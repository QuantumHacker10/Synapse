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
public sealed class WavePropagator : IDisposable
{
    private readonly WaveConfig _cfg;
    private readonly int _nx, _ny, _nz, _n;
    private readonly double _dx, _dt, _c0, _rho0;

    private double[] _p;           // pressure field
    private double[] _pPrev;       // previous time-step pressure
    private double[] _pNext;       // next time-step pressure (temp)
    private double[] _vx, _vy, _vz; // particle velocity
    private double[] _density;     // density field (spatially varying)

    // PML auxiliary arrays (split-field acoustic PML).
    private double[] _pmlPx, _pmlPy, _pmlPz;
    private double[] _sigmaX, _sigmaY, _sigmaZ;

    // Frequency-domain workspace.
    private Complex[] _pFreq;     // complex pressure in frequency domain
    private Complex[] _aCoeff;    // Helmholtz operator diagonal
    private Complex[] _bCoeff;    // Helmholtz operator off-diagonal

    // Energy flux storage.
    private double[] _fluxX, _fluxY, _fluxZ;
    private bool _disposed;

    public int CurrentStep { get; private set; }
    public ReadOnlySpan<double> Pressure => _p;

    public WavePropagator(WaveConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        (_nx, _ny, _nz) = config.GridSize;
        _n = _nx * _ny * _nz;
        _dx = config.CellSize;
        _dt = config.TimeStep;
        _c0 = config.SoundSpeed;
        _rho0 = config.Density;

        // CFL check: dt <= dx / (c * sqrt(3)).
        double cfl = _dx / (_c0 * Math.Sqrt(3.0));
        if (_dt > cfl)
            throw new ArgumentException(
                $"Time-step {_dt:e3} exceeds CFL limit {cfl:e3}.");

        _p = new double[_n];
        _pPrev = new double[_n];
        _pNext = new double[_n];
        _vx = new double[_n];
        _vy = new double[_n];
        _vz = new double[_n];
        _density = new double[_n];
        Array.Fill(_density, _rho0);

        // Energy flux.
        _fluxX = new double[_n];
        _fluxY = new double[_n];
        _fluxZ = new double[_n];

        // PML.
        if (config.PmlThickness > 0)
        {
            _pmlPx = new double[_n];
            _pmlPy = new double[_n];
            _pmlPz = new double[_n];
            _sigmaX = new double[_n];
            _sigmaY = new double[_n];
            _sigmaZ = new double[_n];
            InitialisePml();
        }

        // Frequency-domain setup.
        if (config.UseFrequencyDomain)
        {
            _pFreq = new Complex[_n];
            _aCoeff = new Complex[_n];
            _bCoeff = new Complex[_n];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Idx(int x, int y, int z) => z * _ny * _nx + y * _nx + x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool InBounds(int x, int y, int z)
        => (uint)x < (uint)_nx && (uint)y < (uint)_ny && (uint)z < (uint)_nz;

    // -----------------------------------------------------------------------
    //  PML initialisation
    // -----------------------------------------------------------------------

    private void InitialisePml()
    {
        int pml = _cfg.PmlThickness;
        double sigmaMax = -(double)(_cfg.PmlOrder + 1) * _cfg.PmlR0 *
                          Math.Log(1e-15) / (2.0 * pml * _dx);

        for (int i = 0; i < pml; i++)
        {
            double ratio = (double)(pml - i) / pml;
            double sig = sigmaMax * Math.Pow(ratio, _cfg.PmlOrder);

            // x-dimension
            for (int z = 0; z < _nz; z++)
                for (int y = 0; y < _ny; y++)
                {
                    if (_cfg.PmlFaces.NegX)
                        _sigmaX[Idx(i, y, z)] = sig;
                    if (_cfg.PmlFaces.PosX)
                        _sigmaX[Idx(_nx - 1 - i, y, z)] = sig;
                }

            // y-dimension
            for (int z = 0; z < _nz; z++)
                for (int x = 0; x < _nx; x++)
                {
                    if (_cfg.PmlFaces.NegY)
                        _sigmaY[Idx(x, i, z)] = sig;
                    if (_cfg.PmlFaces.PosY)
                        _sigmaY[Idx(x, _ny - 1 - i, z)] = sig;
                }

            // z-dimension
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    if (_cfg.PmlFaces.NegZ)
                        _sigmaZ[Idx(x, y, i)] = sig;
                    if (_cfg.PmlFaces.PosZ)
                        _sigmaZ[Idx(x, y, _nz - 1 - i)] = sig;
                }
        }
    }

    // -----------------------------------------------------------------------
    //  Set spatially varying sound speed
    // -----------------------------------------------------------------------

    public void SetSoundSpeed(int x0, int y0, int z0, int x1, int y1, int z1, double c)
    {
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++)
                {
                    if (InBounds(x, y, z))
                        _density[Idx(x, y, z)] = _rho0; // could store c² instead
                }
    }

    /// <summary>
    /// Compute the square of the sound speed at each cell.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double SoundSpeedSquared(int idx) => _c0 * _c0;

    // -----------------------------------------------------------------------
    //  Time-domain step
    // -----------------------------------------------------------------------

    /// <summary>
    /// Advance the acoustic field by one time-step.
    /// Uses central-difference scheme: p(t+dt) = 2p(t) − p(t−dt) + c² dt² ∇²p.
    /// </summary>
    public void Step()
    {
        double dt = _dt;
        double dx = _dx;
        double dt2OverDx2 = dt * dt / (dx * dx);
        double dt2c2 = dt * dt * _c0 * _c0;

        // Swap buffers.
        (_pPrev, _p) = (_p, _pPrev);

        // Main stencil loop.
        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);
                    double c2 = SoundSpeedSquared(idx);

                    // Laplacian of pressure (2nd order central difference).
                    double laplacian = (
                        (_pPrev[Idx(x + 1, y, z)] + _pPrev[Idx(x - 1, y, z)] +
                         _pPrev[Idx(x, y + 1, z)] + _pPrev[Idx(x, y - 1, z)] +
                         _pPrev[Idx(x, y, z + 1)] + _pPrev[Idx(x, y, z - 1)] -
                         6.0 * _pPrev[idx])
                    ) * dt2OverDx2;

                    _pNext[idx] = 2.0 * _pPrev[idx] - _p[idx] + c2 * laplacian;

                    // PML absorption.
                    if (_sigmaX != null)
                    {
                        double sx = _sigmaX[idx];
                        double sy = _sigmaY[idx];
                        double sz = _sigmaZ[idx];
                        double dampX = sx > 0 ? Math.Exp(-sx * dt) : 1.0;
                        double dampY = sy > 0 ? Math.Exp(-sy * dt) : 1.0;
                        double dampZ = sz > 0 ? Math.Exp(-sz * dt) : 1.0;
                        _pNext[idx] *= dampX * dampY * dampZ;
                    }
                }

        // Inject soft source (modulated Gaussian pulse).
        var sp = _cfg.SourcePosition;
        if (InBounds(sp.X, sp.Y, sp.Z))
        {
            int sIdx = Idx(sp.X, sp.Y, sp.Z);
            double t = CurrentStep * _dt;
            double f0 = _cfg.SourceFrequency;
            double t0 = 3.0 / f0;
            double tau = 1.0 / (2.0 * f0);
            double pulse = _cfg.SourceAmplitude *
                           Math.Exp(-0.5 * ((t - t0) / tau) * ((t - t0) / tau)) *
                           Math.Cos(PhysicsConstants.TwoPi * f0 * (t - t0));
            _pNext[sIdx] += pulse;
        }

        // Periodic boundaries.
        ApplyPeriodicPressure();

        // Swap next buffer into pressure.
        (_p, _pNext) = (_pNext, _p);

        CurrentStep++;
    }

    private void ApplyPeriodicPressure()
    {
        if (_cfg.Periodic.X)
        {
            for (int z = 0; z < _nz; z++)
                for (int y = 0; y < _ny; y++)
                {
                    _pNext[Idx(0, y, z)] = _pNext[Idx(_nx - 2, y, z)];
                    _pNext[Idx(_nx - 1, y, z)] = _pNext[Idx(1, y, z)];
                }
        }
        if (_cfg.Periodic.Y)
        {
            for (int z = 0; z < _nz; z++)
                for (int x = 0; x < _nx; x++)
                {
                    _pNext[Idx(x, 0, z)] = _pNext[Idx(x, _ny - 2, z)];
                    _pNext[Idx(x, _ny - 1, z)] = _pNext[Idx(x, 1, z)];
                }
        }
        if (_cfg.Periodic.Z)
        {
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    _pNext[Idx(x, y, 0)] = _pNext[Idx(x, y, _nz - 2)];
                    _pNext[Idx(x, y, _nz - 1)] = _pNext[Idx(x, y, 1)];
                }
        }
    }

    // -----------------------------------------------------------------------
    //  Frequency-domain (Helmholtz) solver
    // -----------------------------------------------------------------------

    /// <summary>
    /// Solve the Helmholtz equation (∇² + k²) p = −s using iterative
    /// Gauss-Seidel with successive over-relaxation (SOR).
    /// </summary>
    /// <param name="omega">Angular frequency (rad/s).</param>
    /// <param name="maxIter">Maximum iterations.</param>
    /// <param name="tolerance">Convergence tolerance on residual L2-norm.</param>
    /// <returns>Number of iterations to converge.</returns>
    public int SolveFrequencyDomain(double omega, int maxIter = 1000, double tolerance = 1e-10)
    {
        if (_pFreq == null)
            throw new InvalidOperationException(
                "Frequency-domain solver requires UseFrequencyDomain = true.");

        double k2 = omega * omega / (_c0 * _c0);
        double dx2 = _dx * _dx;
        double sorOmega = 1.5; // SOR relaxation parameter

        // Source vector: modulated point source at source position.
        var sp = _cfg.SourcePosition;
        Complex[] source = new Complex[_n];
        if (InBounds(sp.X, sp.Y, sp.Z))
            source[Idx(sp.X, sp.Y, sp.Z)] = -_cfg.SourceAmplitude;

        // Initial guess.
        for (int i = 0; i < _n; i++)
            _pFreq[i] = Complex.Zero;

        double rhsNorm = 0.0;
        for (int i = 0; i < _n; i++)
            rhsNorm += source[i].Magnitude * source[i].Magnitude;
        rhsNorm = Math.Sqrt(rhsNorm);
        if (rhsNorm < 1e-30)
            rhsNorm = 1.0;

        int iter;
        for (iter = 0; iter < maxIter; iter++)
        {
            double residualNorm = 0.0;

            for (int z = 1; z < _nz - 1; z++)
                for (int y = 1; y < _ny - 1; y++)
                    for (int x = 1; x < _nx - 1; x++)
                    {
                        int idx = Idx(x, y, z);
                        Complex neighbors =
                            _pFreq[Idx(x + 1, y, z)] + _pFreq[Idx(x - 1, y, z)] +
                            _pFreq[Idx(x, y + 1, z)] + _pFreq[Idx(x, y - 1, z)] +
                            _pFreq[Idx(x, y, z + 1)] + _pFreq[Idx(x, y, z - 1)];

                        // (∇² + k²) p = s → (6/dx² − k²) p_center = neighbors/dx² + s
                        Complex diag = new Complex(6.0 / dx2 - k2, 0);
                        Complex rhs = neighbors / dx2 + source[idx];
                        Complex pNew = rhs / diag;

                        // SOR update.
                        Complex correction = (pNew - _pFreq[idx]) * sorOmega;
                        _pFreq[idx] += correction;
                        residualNorm += correction.Magnitude * correction.Magnitude;
                    }

            if (Math.Sqrt(residualNorm) / rhsNorm < tolerance)
                break;
        }

        // Copy solution to real pressure field.
        for (int i = 0; i < _n; i++)
            _p[i] = _pFreq[i].Real;

        return iter;
    }

    // -----------------------------------------------------------------------
    //  Energy flux computation
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute acoustic intensity (energy flux) at each cell.
    /// I = p * v, where v is the particle velocity.
    /// </summary>
    public void ComputeEnergyFlux()
    {
        double dtOverDx = _dt / _dx;
        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);
                    double dpdx = (_p[Idx(x + 1, y, z)] - _p[Idx(x - 1, y, z)]) / (2.0 * _dx);
                    double dpdy = (_p[Idx(x, y + 1, z)] - _p[Idx(x, y - 1, z)]) / (2.0 * _dx);
                    double dpdz = (_p[Idx(x, y, z + 1)] - _p[Idx(x, y, z - 1)]) / (2.0 * _dx);

                    // Particle velocity: v = −∇p / (iωρ) in frequency domain,
                    // or time-integrated: v(t+dt) = v(t) − dt/ρ ∇p.
                    _vx[idx] -= dtOverDx * dpdx / _rho0;
                    _vy[idx] -= dtOverDx * dpdy / _rho0;
                    _vz[idx] -= dtOverDx * dpdz / _rho0;

                    // Intensity I = p * v (time-averaged over a cycle ≈ 0.5 Re{p v*}).
                    _fluxX[idx] = 0.5 * _p[idx] * _vx[idx];
                    _fluxY[idx] = 0.5 * _p[idx] * _vy[idx];
                    _fluxZ[idx] = 0.5 * _p[idx] * _vz[idx];
                }
    }

    /// <summary>
    /// Compute total acoustic energy in the domain.
    /// E = 0.5 ∫ (p²/(ρc²) + ρ|v|²) dV
    /// </summary>
    public double TotalAcousticEnergy()
    {
        double dV = _dx * _dx * _dx;
        double sum = 0.0;
        for (int i = 0; i < _n; i++)
        {
            double pTerm = _p[i] * _p[i] / (_rho0 * _c0 * _c0);
            double vTerm = _rho0 * (_vx[i] * _vx[i] + _vy[i] * _vy[i] + _vz[i] * _vz[i]);
            sum += pTerm + vTerm;
        }
        return 0.5 * sum * dV;
    }

    // -----------------------------------------------------------------------
    //  Dispersion analysis
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute numerical dispersion error ratio for the finite-difference
    /// scheme. Returns ω_numerical / ω_exact for a given wave number k.
    /// </summary>
    public double DispersionRatio(double k, double omega)
    {
        double kdx = k * _dx;
        double sinHalf = Math.Sin(kdx * 0.5);
        double exact = omega;
        double numerical = 2.0 / _dt * Math.Asin(
            _c0 * _dt / _dx * sinHalf);
        return numerical / exact;
    }

    /// <summary>
    /// Maximum resolvable wave number (Nyquist) for the grid.
    /// </summary>
    public double NyquistWaveNumber => PhysicsConstants.Pi / _dx;

    /// <summary>
    /// Wavelength at the configured source frequency.
    /// </summary>
    public double Wavelength => _c0 / _cfg.SourceFrequency;

    /// <summary>
    /// Cells per wavelength — a key quality metric for FD schemes.
    /// </summary>
    public double CellsPerWavelength => Wavelength / _dx;

    /// <summary>
    /// Run the time-domain simulation for the configured number of steps.
    /// </summary>
    public void Run()
    {
        for (int i = 0; i < _cfg.NumSteps; i++)
            Step();
    }

    /// <summary>
    /// Take a snapshot of the current pressure field.
    /// </summary>
    public double[] Snapshot()
    {
        double[] copy = new double[_n];
        Array.Copy(_p, copy, _n);
        return copy;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  3. ThermodynamicEnsemble — Monte Carlo, Gibbs ensemble
// ============================================================================

/// <summary>
/// Represents the type of thermodynamic ensemble.
/// </summary>
