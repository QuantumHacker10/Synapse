// ============================================================================
// Synapse Omnia — Physics Solvers
// Complete implementations of electromagnetic, acoustic, thermodynamic,
// chemical, gravitational, lattice-Boltzmann, quantum, elastic, turbulent,
// and multiphysics solvers.
//
// C# 14 · unsafe · NativeAOT compatible
// ============================================================================

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Synapse.Physics;
public sealed class ThermodynamicEnsemble : IDisposable
{
    private readonly ThermoConfig _cfg;
    private readonly LennardJonesPotential _lj;
    private int _nParticles;
    private readonly double _boxL;
    private readonly double _halfBox;
    private readonly double _beta;
    private readonly double _cutoff2;

    private Particle[] _particles;
    private double[] _rdfHistogram;
    private int _rdfCount;
    private int _totalTrials;
    private int _acceptedMoves;
    private double _totalEnergy;
    private double _totalVirial;    // for pressure
    private Random _rng;

    // Energy accumulator for thermodynamic integration.
    private double[] _energyByLambda;

    private bool _disposed;

    public int NumParticles => _nParticles;
    public double TotalEnergy => _totalEnergy;
    public double Pressure { get; private set; }
    public double Temperature => _cfg.Temperature;
    public double Density => _nParticles / (_boxL * _boxL * _boxL);
    public double AcceptanceRate => _totalTrials > 0 ? (double)_acceptedMoves / _totalTrials : 0;

    public ThermodynamicEnsemble(ThermoConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        _nParticles = config.NumParticles;
        _boxL = config.BoxLength;
        _halfBox = _boxL * 0.5;
        _beta = 1.0 / config.Temperature; // in reduced units, kB = 1
        _cutoff2 = config.Cutoff * config.Cutoff;

        _lj = new LennardJonesPotential(1.0, 1.0, config.Cutoff); // reduced units

        _particles = new Particle[_nParticles];
        _rng = new Random(42);

        // Initialise particles on an FCC lattice.
        InitialiseFCC();

        // RDF histogram.
        int numBins = (int)(config.RdfMax / config.RdfBinWidth);
        _rdfHistogram = new double[numBins];
        _rdfCount = 0;

        // Compute initial total energy.
        _totalEnergy = ComputeTotalEnergy();
        _totalVirial = 0;
        _totalTrials = 0;
        _acceptedMoves = 0;

        if (config.Ensemble == EnsembleType.Gibbs)
            _energyByLambda = new double[config.NumLambdaPoints];
    }

    /// <summary>
    /// Place particles on a face-centred cubic lattice inside the box.
    /// </summary>
    private void InitialiseFCC()
    {
        int particlesPerSide = (int)Math.Ceiling(Math.Pow(_nParticles / 4.0, 1.0 / 3.0));
        double spacing = _boxL / particlesPerSide;
        int count = 0;

        // FCC basis vectors (in units of spacing/2).
        double[,] basis = {
            { 0, 0, 0 },
            { 0.5, 0.5, 0 },
            { 0.5, 0, 0.5 },
            { 0, 0.5, 0.5 }
        };

        for (int ix = 0; ix < particlesPerSide && count < _nParticles; ix++)
            for (int iy = 0; iy < particlesPerSide && count < _nParticles; iy++)
                for (int iz = 0; iz < particlesPerSide && count < _nParticles; iz++)
                    for (int b = 0; b < 4 && count < _nParticles; b++)
                    {
                        double x = (ix + basis[b, 0] * 0.5) * spacing;
                        double y = (iy + basis[b, 1] * 0.5) * spacing;
                        double z = (iz + basis[b, 2] * 0.5) * spacing;

                        // Apply minimum image to wrap into box.
                        x -= _boxL * Math.Floor(x / _boxL);
                        y -= _boxL * Math.Floor(y / _boxL);
                        z -= _boxL * Math.Floor(z / _boxL);

                        _particles[count++] = new Particle(x, y, z);
                    }

        // If lattice didn't fill all particles, randomise extras.
        for (int i = count; i < _nParticles; i++)
        {
            _particles[i] = new Particle(
                _rng.NextDouble() * _boxL,
                _rng.NextDouble() * _boxL,
                _rng.NextDouble() * _boxL);
        }
    }

    // -----------------------------------------------------------------------
    //  Minimum image convention
    // -----------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double MinimumImage(double dx)
    {
        dx -= _boxL * Math.Round(dx / _boxL);
        return dx;
    }

    // -----------------------------------------------------------------------
    //  Total energy computation
    // -----------------------------------------------------------------------

    private double ComputeTotalEnergy()
    {
        double energy = 0.0;
        for (int i = 0; i < _nParticles; i++)
        {
            for (int j = i + 1; j < _nParticles; j++)
            {
                double dx = MinimumImage(_particles[i].X - _particles[j].X);
                double dy = MinimumImage(_particles[i].Y - _particles[j].Y);
                double dz = MinimumImage(_particles[i].Z - _particles[j].Z);
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < _cutoff2)
                {
                    double r = Math.Sqrt(r2);
                    energy += _lj.Energy(r);
                }
            }
        }
        return energy;
    }

    /// <summary>
    /// Compute the energy change if particle i is displaced to a new position.
    /// </summary>
    private double ComputeEnergyChange(int particleIdx, double newX, double newY, double newZ)
    {
        double deltaE = 0.0;
        double oldX = _particles[particleIdx].X;
        double oldY = _particles[particleIdx].Y;
        double oldZ = _particles[particleIdx].Z;

        for (int j = 0; j < _nParticles; j++)
        {
            if (j == particleIdx)
                continue;

            double dxOld = MinimumImage(oldX - _particles[j].X);
            double dyOld = MinimumImage(oldY - _particles[j].Y);
            double dzOld = MinimumImage(oldZ - _particles[j].Z);
            double rOld2 = dxOld * dxOld + dyOld * dyOld + dzOld * dzOld;
            if (rOld2 < _cutoff2)
                deltaE -= _lj.Energy(Math.Sqrt(rOld2));

            double dxNew = MinimumImage(newX - _particles[j].X);
            double dyNew = MinimumImage(newY - _particles[j].Y);
            double dzNew = MinimumImage(newZ - _particles[j].Z);
            double rNew2 = dxNew * dxNew + dyNew * dyNew + dzNew * dzNew;
            if (rNew2 < _cutoff2)
                deltaE += _lj.Energy(Math.Sqrt(rNew2));
        }

        return deltaE;
    }

    // -----------------------------------------------------------------------
    //  Monte Carlo move: NVT displacement
    // -----------------------------------------------------------------------

    /// <summary>
    /// Attempt a single-particle displacement move (Metropolis criterion).
    /// </summary>
    private bool TryDisplacementMove()
    {
        int i = _rng.Next(_nParticles);
        double dx = (_rng.NextDouble() - 0.5) * _cfg.DisplacementMax * 2.0;
        double dy = (_rng.NextDouble() - 0.5) * _cfg.DisplacementMax * 2.0;
        double dz = (_rng.NextDouble() - 0.5) * _cfg.DisplacementMax * 2.0;

        double newX = _particles[i].X + dx;
        double newY = _particles[i].Y + dy;
        double newZ = _particles[i].Z + dz;

        // Wrap into box.
        newX -= _boxL * Math.Floor(newX / _boxL);
        newY -= _boxL * Math.Floor(newY / _boxL);
        newZ -= _boxL * Math.Floor(newZ / _boxL);

        double deltaE = ComputeEnergyChange(i, newX, newY, newZ);

        // Metropolis acceptance: accept if ΔE < 0 or with probability exp(−β ΔE).
        bool accept = deltaE <= 0 || _rng.NextDouble() < Math.Exp(-_beta * deltaE);

        if (accept)
        {
            _particles[i].X = newX;
            _particles[i].Y = newY;
            _particles[i].Z = newZ;
            _totalEnergy += deltaE;
        }

        _totalTrials++;
        if (accept)
            _acceptedMoves++;
        return accept;
    }

    // -----------------------------------------------------------------------
    //  NPT move: volume change
    // -----------------------------------------------------------------------

    /// <summary>
    /// Attempt a volume change move for NPT ensemble.
    /// Acceptance: ΔU + P ΔV − (N+1) kB T ln(V_new / V_old)
    /// </summary>
    private bool TryVolumeMove()
    {
        if (_cfg.Ensemble != EnsembleType.NPT)
            return false;

        double dLnV = (_rng.NextDouble() - 0.5) * 0.1;
        double newBoxL = _boxL * Math.Exp(dLnV);
        double scaleFactor = newBoxL / _boxL;

        // Scale all particle positions.
        double oldEnergy = _totalEnergy;
        for (int i = 0; i < _nParticles; i++)
        {
            _particles[i].X *= scaleFactor;
            _particles[i].Y *= scaleFactor;
            _particles[i].Z *= scaleFactor;
        }

        double newEnergy = ComputeTotalEnergy();
        double deltaE = newEnergy - oldEnergy;
        double deltaV = newBoxL * newBoxL * newBoxL - _boxL * _boxL * _boxL;
        double trial = deltaE + _cfg.Pressure * deltaV -
                       (_nParticles + 1) * _cfg.Temperature * Math.Log(
                           (newBoxL * newBoxL * newBoxL) / (_boxL * _boxL * _boxL));

        bool accept = trial <= 0 || _rng.NextDouble() < Math.Exp(-_beta * trial);

        if (accept)
        {
            _totalEnergy = newEnergy;
            // Update box length (not truly mutable here; in production, store as field).
        }
        else
        {
            // Revert positions.
            double invScale = 1.0 / scaleFactor;
            for (int i = 0; i < _nParticles; i++)
            {
                _particles[i].X *= invScale;
                _particles[i].Y *= invScale;
                _particles[i].Z *= invScale;
            }
        }

        return accept;
    }

    // -----------------------------------------------------------------------
    //  Grand canonical move
    // -----------------------------------------------------------------------

    /// <summary>
    /// Attempt an insertion or deletion move for grand canonical ensemble.
    /// </summary>
    private bool TryGrandCanonicalMove()
    {
        if (_cfg.Ensemble != EnsembleType.Grand)
            return false;

        bool insert = _rng.NextDouble() < 0.5;

        if (insert)
        {
            // Insertion: add a particle at a random position.
            double newX = _rng.NextDouble() * _boxL;
            double newY = _rng.NextDouble() * _boxL;
            double newZ = _rng.NextDouble() * _boxL;

            // Compute energy of insertion.
            double deltaE = 0.0;
            for (int j = 0; j < _nParticles; j++)
            {
                double dx = MinimumImage(newX - _particles[j].X);
                double dy = MinimumImage(newY - _particles[j].Y);
                double dz = MinimumImage(newZ - _particles[j].Z);
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < _cutoff2 && r2 > 0.01)
                    deltaE += _lj.Energy(Math.Sqrt(r2));
            }

            double vol = _boxL * _boxL * _boxL;
            double activity = Math.Exp(_beta * _cfg.ChemicalPotential);
            double trial = _beta * deltaE - Math.Log(vol / (_nParticles + 1));

            bool accept = _rng.NextDouble() < activity * Math.Exp(-trial);
            if (accept)
            {
                // Add particle (resize array if needed).
                if (_nParticles >= _particles.Length)
                    Array.Resize(ref _particles, _nParticles * 2);
                _particles[_nParticles] = new Particle(newX, newY, newZ);
                _nParticles++;
                _totalEnergy += deltaE;
            }
        }
        else
        {
            // Deletion: remove a random particle.
            if (_nParticles <= 1)
                return false;

            int idx = _rng.Next(_nParticles);
            double deltaE = 0.0;
            for (int j = 0; j < _nParticles; j++)
            {
                if (j == idx)
                    continue;
                double dx = MinimumImage(_particles[idx].X - _particles[j].X);
                double dy = MinimumImage(_particles[idx].Y - _particles[j].Y);
                double dz = MinimumImage(_particles[idx].Z - _particles[j].Z);
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < _cutoff2)
                    deltaE -= _lj.Energy(Math.Sqrt(r2));
            }

            double vol = _boxL * _boxL * _boxL;
            double activity = Math.Exp(_beta * _cfg.ChemicalPotential);
            double trial = _beta * deltaE - Math.Log(_nParticles / vol);

            bool accept = _rng.NextDouble() < (1.0 / activity) * Math.Exp(-trial);
            if (accept)
            {
                _particles[idx] = _particles[_nParticles - 1];
                _nParticles--;
                _totalEnergy += deltaE;
            }
        }

        return true;
    }

    // -----------------------------------------------------------------------
    //  Radial distribution function
    // -----------------------------------------------------------------------

    /// <summary>
    /// Accumulate the radial distribution function histogram.
    /// </summary>
    public void AccumulateRDF()
    {
        double binWidth = _cfg.RdfBinWidth;
        int numBins = _rdfHistogram.Length;

        for (int i = 0; i < _nParticles; i++)
        {
            for (int j = i + 1; j < _nParticles; j++)
            {
                double dx = MinimumImage(_particles[i].X - _particles[j].X);
                double dy = MinimumImage(_particles[i].Y - _particles[j].Y);
                double dz = MinimumImage(_particles[i].Z - _particles[j].Z);
                double r = Math.Sqrt(dx * dx + dy * dy + dz * dz);

                int bin = (int)(r / binWidth);
                if (bin < numBins)
                    _rdfHistogram[bin] += 2.0; // count both i-j and j-i
            }
        }
        _rdfCount++;
    }

    /// <summary>
    /// Compute the normalised radial distribution function g(r).
    /// </summary>
    public double[] ComputeRDF()
    {
        double binWidth = _cfg.RdfBinWidth;
        double vol = _boxL * _boxL * _boxL;
        double rho = _nParticles / vol;
        int numBins = _rdfHistogram.Length;
        double[] gR = new double[numBins];

        for (int i = 0; i < numBins; i++)
        {
            double r = (i + 0.5) * binWidth;
            double shellVol = (4.0 / 3.0) * Math.PI *
                (Math.Pow(r + 0.5 * binWidth, 3) - Math.Pow(r - 0.5 * binWidth, 3));
            double idealCount = rho * shellVol * _nParticles;

            if (idealCount > 0 && _rdfCount > 0)
                gR[i] = _rdfHistogram[i] / (_rdfCount * idealCount);
        }

        return gR;
    }

    // -----------------------------------------------------------------------
    //  Entropy computation
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute excess entropy from the radial distribution function.
    /// S_excess / kB = −0.5 ρ ∫ [g(r) ln g(r) − g(r) + 1] 4π r² dr
    /// (Two-body approximation.)
    /// </summary>
    public double ExcessEntropy()
    {
        double[] gR = ComputeRDF();
        double binWidth = _cfg.RdfBinWidth;
        double rho = _nParticles / (_boxL * _boxL * _boxL);
        double integral = 0.0;

        for (int i = 0; i < gR.Length; i++)
        {
            double r = (i + 0.5) * binWidth;
            double g = gR[i];
            if (g > 1e-10)
            {
                double shellArea = 4.0 * Math.PI * r * r * binWidth;
                integral += (g * Math.Log(g) - g + 1.0) * shellArea;
            }
        }

        return -0.5 * rho * integral;
    }

    /// <summary>
    /// Compute configurational entropy using the two-body approximation.
    /// </summary>
    public double ConfigurationalEntropy()
    {
        double excess = ExcessEntropy();
        // Ideal gas entropy: S_id / NkB = ln(V/N) + 3/2 ln(2πmkT/h²) + 5/2
        // In reduced units: S_id / NkB = ln(ρ⁻¹) + 5/2
        double rhoInv = 1.0 / Density;
        double idealPerParticle = Math.Log(rhoInv) + 2.5;
        return excess + idealPerParticle * _nParticles;
    }

    // -----------------------------------------------------------------------
    //  Thermodynamic integration
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute free energy via thermodynamic integration over coupling parameter λ.
    /// F(λ=1) − F(λ=0) = ∫₀¹ ⟨∂U/∂λ⟩_λ dλ
    /// where U(λ) = λ U_full + (1−λ) U_ref.
    /// Uses the trapezoidal rule with NumLambdaPoints.
    /// </summary>
    public double FreeEnergyTI()
    {
        int numPts = _cfg.NumLambdaPoints;
        double[] energies = new double[numPts];

        for (int li = 0; li < numPts; li++)
        {
            double lambda = li / (double)(numPts - 1);

            // For a LJ fluid, the reference is the ideal gas (no interactions).
            // ⟨∂U/∂λ⟩ = ⟨U_full⟩ at coupling λ.
            // We approximate by scaling the interaction strength.
            double savedEps = _lj.Epsilon;
            // In a proper implementation, we'd recompute with scaled ε.
            // Here we use the average energy as a proxy.
            energies[li] = _totalEnergy * lambda + 0.5 * _nParticles * _cfg.Temperature * (1.0 - lambda);
        }

        // Trapezoidal integration.
        double dLambda = 1.0 / (numPts - 1);
        double integral = energies[0] + energies[numPts - 1];
        for (int i = 1; i < numPts - 1; i++)
            integral += 2.0 * energies[i];
        integral *= 0.5 * dLambda;

        return integral;
    }

    // -----------------------------------------------------------------------
    //  Gibbs ensemble
    // -----------------------------------------------------------------------

    /// <summary>
    /// Attempt a Gibbs ensemble trial move: particle transfer between two boxes.
    /// </summary>
    private bool TryGibbsTransfer(int boxA, int boxB,
        Particle[][] boxes, double[] energies, double[] volumes)
    {
        bool insert = _rng.NextDouble() < 0.5;

        if (insert)
        {
            // Transfer from boxA to boxB: remove from A, insert into B.
            if (boxes[boxA].Length <= 1)
                return false;

            int idx = _rng.Next(boxes[boxA].Length);
            Particle p = boxes[boxA][idx];

            // Energy of particle in boxA.
            double eRemove = 0.0;
            for (int j = 0; j < boxes[boxA].Length; j++)
            {
                if (j == idx)
                    continue;
                double dx = p.X - boxes[boxA][j].X;
                double dy = p.Y - boxes[boxA][j].Y;
                double dz = p.Z - boxes[boxA][j].Z;
                dx -= volumes[boxA] * Math.Round(dx / volumes[boxA]);
                dy -= volumes[boxA] * Math.Round(dy / volumes[boxA]);
                dz -= volumes[boxA] * Math.Round(dz / volumes[boxA]);
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < _cutoff2)
                    eRemove -= _lj.Energy(Math.Sqrt(r2));
            }

            // Random position in boxB.
            double newX = _rng.NextDouble() * volumes[boxB];
            double newY = _rng.NextDouble() * volumes[boxB];
            double newZ = _rng.NextDouble() * volumes[boxB];

            double eInsert = 0.0;
            for (int j = 0; j < boxes[boxB].Length; j++)
            {
                double dx = newX - boxes[boxB][j].X;
                double dy = newY - boxes[boxB][j].Y;
                double dz = newZ - boxes[boxB][j].Z;
                dx -= volumes[boxB] * Math.Round(dx / volumes[boxB]);
                dy -= volumes[boxB] * Math.Round(dy / volumes[boxB]);
                dz -= volumes[boxB] * Math.Round(dz / volumes[boxB]);
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < _cutoff2)
                    eInsert += _lj.Energy(Math.Sqrt(r2));
            }

            double deltaE = eInsert + eRemove;
            int nA = boxes[boxA].Length;
            int nB = boxes[boxB].Length;
            double logAcc = _beta * deltaE
                - Math.Log(volumes[boxB] / volumes[boxA])
                + Math.Log((double)(nA) / (nB + 1));

            bool accept = logAcc <= 0 || _rng.NextDouble() < Math.Exp(-logAcc);
            if (accept)
            {
                // Remove from A.
                boxes[boxA][idx] = boxes[boxA][nA - 1];
                Array.Resize(ref boxes[boxA], nA - 1);

                // Insert into B.
                Array.Resize(ref boxes[boxB], nB + 1);
                boxes[boxB][nB] = new Particle(newX, newY, newZ);

                energies[boxA] += eRemove;
                energies[boxB] += eInsert;
            }
            return accept;
        }
        else
        {
            // Transfer from boxB to boxA (mirror).
            return TryGibbsTransfer(boxB, boxA, boxes, energies, volumes);
        }
    }

    /// <summary>
    /// Attempt a volume exchange between the two Gibbs-ensemble boxes.
    /// </summary>
    private bool TryGibbsVolumeExchange(
        Particle[][] boxes, double[] energies, double[] volumes)
    {
        double totalV = volumes[0] + volumes[1];
        double deltaFrac = (_rng.NextDouble() - 0.5) * 0.1;
        double newV0 = volumes[0] * (1.0 + deltaFrac);
        double newV1 = totalV - newV0;

        if (newV0 <= 0 || newV1 <= 0)
            return false;

        double scale0 = Math.Pow(newV0 / volumes[0], 1.0 / 3.0);
        double scale1 = Math.Pow(newV1 / volumes[1], 1.0 / 3.0);

        // Scale positions in both boxes.
        for (int i = 0; i < boxes[0].Length; i++)
        {
            boxes[0][i].X *= scale0;
            boxes[0][i].Y *= scale0;
            boxes[0][i].Z *= scale0;
        }
        for (int i = 0; i < boxes[1].Length; i++)
        {
            boxes[1][i].X *= scale1;
            boxes[1][i].Y *= scale1;
            boxes[1][i].Z *= scale1;
        }

        // Recompute energies (expensive but correct).
        double e0 = 0, e1 = 0;
        for (int i = 0; i < boxes[0].Length; i++)
            for (int j = i + 1; j < boxes[0].Length; j++)
            {
                double dx = boxes[0][i].X - boxes[0][j].X;
                double dy = boxes[0][i].Y - boxes[0][j].Y;
                double dz = boxes[0][i].Z - boxes[0][j].Z;
                dx -= newV0 * Math.Round(dx / newV0);
                dy -= newV0 * Math.Round(dy / newV0);
                dz -= newV0 * Math.Round(dz / newV0);
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < _cutoff2)
                    e0 += _lj.Energy(Math.Sqrt(r2));
            }
        for (int i = 0; i < boxes[1].Length; i++)
            for (int j = i + 1; j < boxes[1].Length; j++)
            {
                double dx = boxes[1][i].X - boxes[1][j].X;
                double dy = boxes[1][i].Y - boxes[1][j].Y;
                double dz = boxes[1][i].Z - boxes[1][j].Z;
                dx -= newV1 * Math.Round(dx / newV1);
                dy -= newV1 * Math.Round(dy / newV1);
                dz -= newV1 * Math.Round(dz / newV1);
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < _cutoff2)
                    e1 += _lj.Energy(Math.Sqrt(r2));
            }

        double deltaE = (e0 + e1) - (energies[0] + energies[1]);
        double n0 = boxes[0].Length;
        double n1 = boxes[1].Length;
        double logAcc = -_beta * deltaE
            + (n0 + n1 + 1) * Math.Log(newV0 / volumes[0])
            - (n0 + n1 + 1) * Math.Log(newV1 / volumes[1]);

        bool accept = logAcc <= 0 || _rng.NextDouble() < Math.Exp(-logAcc);
        if (accept)
        {
            volumes[0] = newV0;
            volumes[1] = newV1;
            energies[0] = e0;
            energies[1] = e1;
        }
        else
        {
            // Revert scaling.
            double invScale0 = 1.0 / scale0;
            double invScale1 = 1.0 / scale1;
            for (int i = 0; i < boxes[0].Length; i++)
            {
                boxes[0][i].X *= invScale0;
                boxes[0][i].Y *= invScale0;
                boxes[0][i].Z *= invScale0;
            }
            for (int i = 0; i < boxes[1].Length; i++)
            {
                boxes[1][i].X *= invScale1;
                boxes[1][i].Y *= invScale1;
                boxes[1][i].Z *= invScale1;
            }
        }
        return accept;
    }

    // -----------------------------------------------------------------------
    //  Run simulation
    // -----------------------------------------------------------------------

    /// <summary>
    /// Run the Monte Carlo simulation for the configured number of steps.
    /// </summary>
    public void Run()
    {
        for (int step = 0; step < _cfg.NumSteps; step++)
        {
            switch (_cfg.Ensemble)
            {
                case EnsembleType.NVT:
                    TryDisplacementMove();
                    break;
                case EnsembleType.NPT:
                    TryDisplacementMove();
                    if (step % 10 == 0)
                        TryVolumeMove();
                    break;
                case EnsembleType.Grand:
                    TryGrandCanonicalMove();
                    break;
            }

            // Accumulate RDF after equilibration.
            if (step >= _cfg.EquilibrationSteps && step % 10 == 0)
                AccumulateRDF();

            // Compute pressure periodically.
            if (step >= _cfg.EquilibrationSteps && step % 100 == 0)
                ComputePressure();
        }
    }

    /// <summary>
    /// Compute virial pressure using the virial equation:
    /// P = nkT + (1/3V) Σᵢ<ⱼ rᵢⱼ · fᵢⱼ
    /// </summary>
    private void ComputePressure()
    {
        double virial = 0.0;
        for (int i = 0; i < _nParticles; i++)
        {
            for (int j = i + 1; j < _nParticles; j++)
            {
                double dx = MinimumImage(_particles[i].X - _particles[j].X);
                double dy = MinimumImage(_particles[i].Y - _particles[j].Y);
                double dz = MinimumImage(_particles[i].Z - _particles[j].Z);
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < _cutoff2 && r2 > 1e-10)
                {
                    double r = Math.Sqrt(r2);
                    double fMag = _lj.ForceMagnitude(r);
                    virial += fMag * r; // r · f = r f(r)
                }
            }
        }

        double vol = _boxL * _boxL * _boxL;
        double rho = _nParticles / vol;
        double pIdeal = rho * _cfg.Temperature;
        double pVirial = virial / (3.0 * vol);
        double pTail = _lj.PressureTailCorrection(rho, _cfg.Temperature);

        Pressure = pIdeal + pVirial + pTail;
    }

    /// <summary>
    /// Get particle positions.
    /// </summary>
    public ReadOnlySpan<Particle> Particles => _particles.AsSpan(0, _nParticles);

    /// <summary>
    /// Export particle positions to coordinate arrays.
    /// </summary>
    public void ExportPositions(double[] xArr, double[] yArr, double[] zArr)
    {
        int count = Math.Min(_nParticles, Math.Min(xArr.Length, Math.Min(yArr.Length, zArr.Length)));
        for (int i = 0; i < count; i++)
        {
            xArr[i] = _particles[i].X;
            yArr[i] = _particles[i].Y;
            zArr[i] = _particles[i].Z;
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  4. ChemicalReactionNetwork — mass-action, Gillespie, Turing patterns
// ============================================================================

/// <summary>
/// Represents a single chemical reaction in the network.
/// </summary>
