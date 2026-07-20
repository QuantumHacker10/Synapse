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

public sealed class SchrodingerSolver : IDisposable
{
    private readonly SchrodingerConfig _cfg;
    private readonly int _nx, _ny, _nz, _n;
    private readonly double _dx, _dt;
    private readonly double _hbar, _m;

    // Wavefunction: real and imaginary parts.
    private double[] _psiRe, _psiIm;
    private double[] _psiRePrev, _psiImPrev;

    // Potential.
    private double[] _V;

    // Time-dependent Crank-Nicolson tridiagonal coefficients.
    private double[] _alphaRe, _alphaIm;    // diagonal
    private double[] _betaRe, _betaIm;      // off-diagonal

    // Eigenstates.
    private double[][] _eigenRe;
    private double[][] _eigenIm;
    private double[] _eigenValues;

    // Density.
    private double[] _density;

    // Time-series storage.
    private List<double[]> _probabilityHistory;

    private bool _disposed;
    private int _currentStep;

    public int CurrentStep => _currentStep;
    public ReadOnlySpan<double> PsiRe => _psiRe;
    public ReadOnlySpan<double> PsiIm => _psiIm;
    public ReadOnlySpan<double> Density => _density;
    public ReadOnlySpan<double> EigenValues => _eigenValues;

    public SchrodingerSolver(SchrodingerConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        (_nx, _ny, _nz) = config.GridSize;
        _n = _nx * _ny * _nz;
        _dx = config.GridSpacing;
        _dt = config.TimeStep;

        // Physical constants in eV·fs units.
        _hbar = 0.6582119569;  // ℏ in eV·fs
        _m = config.ParticleMass; // in units of electron mass

        // Allocate.
        _psiRe = new double[_n];
        _psiIm = new double[_n];
        _psiRePrev = new double[_n];
        _psiImPrev = new double[_n];
        _V = new double[_n];
        _density = new double[_n];

        // Crank-Nicolson coefficients.
        double r = _hbar * _dt / (2.0 * _m * _dx * _dx);
        _alphaRe = new double[_n];
        _alphaIm = new double[_n];
        _betaRe = new double[_n];
        _betaIm = new double[_n];
        for (int i = 0; i < _n; i++)
        {
            _alphaRe[i] = 1.0 + 6.0 * r;
            _alphaIm[i] = _V[i] * _dt / (2.0 * _hbar);
            _betaRe[i] = -r;
            _betaIm[i] = 0;
        }

        // Eigenstates.
        int nEigen = config.NumEigenstates;
        _eigenRe = new double[nEigen][];
        _eigenIm = new double[nEigen][];
        _eigenValues = new double[nEigen];
        for (int i = 0; i < nEigen; i++)
        {
            _eigenRe[i] = new double[_n];
            _eigenIm[i] = new double[_n];
        }

        if (config.RecordProbability)
            _probabilityHistory = new List<double[]>();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Idx(int x, int y, int z) => z * _ny * _nx + y * _nx + x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool InBounds(int x, int y, int z)
        => (uint)x < (uint)_nx && (uint)y < (uint)_ny && (uint)z < (uint)_nz;

    /// <summary>
    /// Set the external potential V(x, y, z) at a grid point.
    /// </summary>
    public void SetPotential(int x, int y, int z, double value)
    {
        if (InBounds(x, y, z))
            _V[Idx(x, y, z)] = value;
    }

    /// <summary>
    /// Set potential on a rectangular region.
    /// </summary>
    public void SetPotentialRegion(
        (int X0, int Y0, int Z0) lo, (int X1, int Y1, int Z1) hi, double value)
    {
        for (int z = lo.Z0; z <= hi.Z1; z++)
            for (int y = lo.Y0; y <= hi.Y1; y++)
                for (int x = lo.X0; x <= hi.X1; x++)
                    if (InBounds(x, y, z))
                        _V[Idx(x, y, z)] = value;
    }

    /// <summary>
    /// Create an infinite square well potential (V = 0 inside, V = ∞ outside).
    /// </summary>
    public void CreateInfiniteWell()
    {
        Array.Fill(_V, 1e10);
        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                    _V[Idx(x, y, z)] = 0;
    }

    /// <summary>
    /// Create a harmonic oscillator potential: V = 0.5 m ω² r².
    /// </summary>
    public void CreateHarmonicOscillator(double omega, (double Cx, double Cy, double Cz) center)
    {
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    double dx = (x - center.Cx) * _dx;
                    double dy = (y - center.Cy) * _dx;
                    double dz = (z - center.Cz) * _dx;
                    double r2 = dx * dx + dy * dy + dz * dz;
                    _V[Idx(x, y, z)] = 0.5 * _m * omega * omega * r2;
                }
    }

    /// <summary>
    /// Create a hydrogen-like Coulomb potential: V = −Ze²/(4πε₀r).
    /// </summary>
    public void CreateCoulombPotential(double Z, (double Cx, double Cy, double Cz) center)
    {
        double eVnm = 14.3996; // e²/(4πε₀) in eV·Å
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    double dx = (x - center.Cx) * _dx;
                    double dy = (y - center.Cy) * _dx;
                    double dz = (z - center.Cz) * _dx;
                    double r = Math.Sqrt(dx * dx + dy * dy + dz * dz + 0.01); // soft-core
                    _V[Idx(x, y, z)] = -Z * eVnm / r;
                }
    }

    /// <summary>
    /// Create a double-well potential along x.
    /// </summary>
    public void CreateDoubleWell(double depth, double separation)
    {
        double cx = _nx * 0.5;
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    double dx1 = (x - cx + separation * 0.5) * _dx;
                    double dx2 = (x - cx - separation * 0.5) * _dx;
                    _V[Idx(x, y, z)] = depth * (dx1 * dx1 * dx2 * dx2);
                }
    }

    /// <summary>
    /// Initialise a Gaussian wave packet: ψ = N exp(−r²/2σ² + ik·r).
    /// </summary>
    public void InitialiseGaussianPacket(
        (double Cx, double Cy, double Cz) center,
        double sigma,
        (double Kx, double Ky, double Kz) k)
    {
        double norm = 0;
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    double dx = (x - center.Cx) * _dx;
                    double dy = (y - center.Cy) * _dx;
                    double dz = (z - center.Cz) * _dx;
                    double r2 = dx * dx + dy * dy + dz * dz;

                    double envelope = Math.Exp(-r2 / (2.0 * sigma * sigma));
                    double phase = k.Kx * dx + k.Ky * dy + k.Kz * dz;

                    int idx = Idx(x, y, z);
                    _psiRe[idx] = envelope * Math.Cos(phase);
                    _psiIm[idx] = envelope * Math.Sin(phase);
                    norm += _psiRe[idx] * _psiRe[idx] + _psiIm[idx] * _psiIm[idx];
                }

        // Normalise.
        norm = Math.Sqrt(norm * _dx * _dx * _dx);
        if (norm > 1e-30)
        {
            for (int i = 0; i < _n; i++)
            {
                _psiRe[i] /= norm;
                _psiIm[i] /= norm;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Time-dependent: Crank-Nicolson
    // -----------------------------------------------------------------------

    /// <summary>
    /// Advance the wavefunction by one time-step using the Crank-Nicolson scheme.
    /// The Schrödinger equation iℏ ∂ψ/∂t = Ĥψ is split as:
    /// ψ(t+dt) = (1 + iĤdt/2ℏ)⁻¹ (1 − iĤdt/2ℏ) ψ(t)
    /// The tridiagonal system is solved line-by-line (ADI approach).
    /// </summary>
    public void StepTDSE()
    {
        int n = _n;
        double dt = _dt;
        double hbar = _hbar;
        double m = _m;

        // Coefficient r = ℏ dt / (2m dx²).
        double r = hbar * dt / (2.0 * m * _dx * _dx);

        // Save previous.
        Array.Copy(_psiRe, _psiRePrev, n);
        Array.Copy(_psiIm, _psiImPrev, n);

        // ADI (Alternating Direction Implicit) split:
        // X-sweep, then Y-sweep, then Z-sweep.

        // X-sweep: implicit in x, explicit in y,z.
        SolveTridiagonalX(r, dt, hbar);

        // Y-sweep: implicit in y, explicit in x,z.
        SolveTridiagonalY(r, dt, hbar);

        // Z-sweep: implicit in z, explicit in x,y.
        if (_nz > 1)
            SolveTridiagonalZ(r, dt, hbar);

        // Update density.
        if (_cfg.ComputeDensity)
            ComputeDensityField();

        // Record if requested.
        if (_cfg.RecordProbability && _probabilityHistory != null &&
            _currentStep % _cfg.RecordInterval == 0)
        {
            double[] prob = new double[n];
            for (int i = 0; i < n; i++)
                prob[i] = _psiRe[i] * _psiRe[i] + _psiIm[i] * _psiIm[i];
            _probabilityHistory.Add(prob);
        }

        _currentStep++;
    }

    private void SolveTridiagonalX(double r, double dt, double hbar)
    {
        double[] a = new double[_nx], b = new double[_nx], c = new double[_nx];
        double[] dRe = new double[_nx], dIm = new double[_nx];
        double[] solRe = new double[_nx], solIm = new double[_nx];

        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
            {
                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    double vi = _V[idx] * dt / (2.0 * hbar);

                    // RHS: (1 − iĤdt/2ℏ) ψ
                    double laplacianY = 0, laplacianZ = 0;
                    if (y > 0 && y < _ny - 1)
                        laplacianY = r * (_psiRePrev[Idx(x, y + 1, z)] +
                                          _psiRePrev[Idx(x, y - 1, z)] -
                                          2.0 * _psiRePrev[idx]);
                    if (_nz > 1 && z > 0 && z < _nz - 1)
                        laplacianZ = r * (_psiRePrev[Idx(x, y, z + 1)] +
                                          _psiRePrev[Idx(x, y, z - 1)] -
                                          2.0 * _psiRePrev[idx]);

                    dRe[x] = _psiRePrev[idx] + vi * _psiImPrev[idx] -
                             laplacianY - laplacianZ;
                    dIm[x] = _psiImPrev[idx] - vi * _psiRePrev[idx];

                    // Tridiagonal coefficients for x-direction.
                    a[x] = -r;  // sub-diagonal
                    b[x] = 1.0 + 2.0 * r + vi;  // diagonal (for Im part)
                    c[x] = -r;  // super-diagonal

                    // Adjust for imaginary part.
                    dIm[x] -= r * (_psiRePrev[Math.Max(0, x - 1)] - 2.0 * _psiRePrev[idx] +
                                   _psiRePrev[Math.Min(_nx - 1, x + 1)]);
                }

                SolveTridiagonalSystem(a, b, c, dRe, solRe, _nx);
                SolveTridiagonalSystem(a, b, c, dIm, solIm, _nx);

                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    _psiRe[idx] = solRe[x];
                    _psiIm[idx] = solIm[x];
                }
            }
    }

    private void SolveTridiagonalY(double r, double dt, double hbar)
    {
        double[] a = new double[_ny], b = new double[_ny], c = new double[_ny];
        double[] dRe = new double[_ny], dIm = new double[_ny];
        double[] solRe = new double[_ny], solIm = new double[_ny];

        for (int z = 0; z < _nz; z++)
            for (int x = 0; x < _nx; x++)
            {
                for (int y = 0; y < _ny; y++)
                {
                    int idx = Idx(x, y, z);
                    double vi = _V[idx] * dt / (2.0 * hbar);

                    dRe[x] = _psiRe[idx];
                    dIm[x] = _psiIm[idx];

                    a[y] = -r;
                    b[y] = 1.0 + 2.0 * r + vi;
                    c[y] = -r;
                }

                SolveTridiagonalSystem(a, b, c, dRe, solRe, _ny);
                SolveTridiagonalSystem(a, b, c, dIm, solIm, _ny);

                for (int y = 0; y < _ny; y++)
                {
                    int idx = Idx(x, y, z);
                    _psiRe[idx] = solRe[y];
                    _psiIm[idx] = solIm[y];
                }
            }
    }

    private void SolveTridiagonalZ(double r, double dt, double hbar)
    {
        double[] a = new double[_nz], b = new double[_nz], c = new double[_nz];
        double[] dRe = new double[_nz], dIm = new double[_nz];
        double[] solRe = new double[_nz], solIm = new double[_nz];

        for (int y = 0; y < _ny; y++)
            for (int x = 0; x < _nx; x++)
            {
                for (int z = 0; z < _nz; z++)
                {
                    int idx = Idx(x, y, z);
                    double vi = _V[idx] * dt / (2.0 * hbar);

                    dRe[z] = _psiRe[idx];
                    dIm[z] = _psiIm[idx];

                    a[z] = -r;
                    b[z] = 1.0 + 2.0 * r + vi;
                    c[z] = -r;
                }

                SolveTridiagonalSystem(a, b, c, dRe, solRe, _nz);
                SolveTridiagonalSystem(a, b, c, dIm, solIm, _nz);

                for (int z = 0; z < _nz; z++)
                {
                    int idx = Idx(x, y, z);
                    _psiRe[idx] = solRe[z];
                    _psiIm[idx] = solIm[z];
                }
            }
    }

    /// <summary>
    /// Thomas algorithm for tridiagonal system Ax = d.
    /// </summary>
    private static void SolveTridiagonalSystem(
        double[] a, double[] b, double[] c, double[] d, double[] x, int n)
    {
        double[] cp = new double[n];
        double[] dp = new double[n];

        cp[0] = c[0] / b[0];
        dp[0] = d[0] / b[0];

        for (int i = 1; i < n; i++)
        {
            double denom = b[i] - a[i] * cp[i - 1];
            if (Math.Abs(denom) < 1e-30)
                denom = 1e-30;
            cp[i] = c[i] / denom;
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
        }

        x[n - 1] = dp[n - 1];
        for (int i = n - 2; i >= 0; i--)
            x[i] = dp[i] - cp[i] * x[i + 1];
    }

    // -----------------------------------------------------------------------
    //  Density and expectation values
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute the probability density |ψ|² = Re² + Im².
    /// </summary>
    public void ComputeDensityField()
    {
        for (int i = 0; i < _n; i++)
            _density[i] = _psiRe[i] * _psiRe[i] + _psiIm[i] * _psiIm[i];
    }

    /// <summary>
    /// Compute the normalisation: ∫ |ψ|² dV.
    /// </summary>
    public double ComputeNorm()
    {
        double sum = 0;
        for (int i = 0; i < _n; i++)
            sum += _psiRe[i] * _psiRe[i] + _psiIm[i] * _psiIm[i];
        return sum * _dx * _dx * _dx;
    }

    /// <summary>
    /// Compute expectation value of position: ⟨x⟩ = ∫ x |ψ|² dV.
    /// </summary>
    public (double X, double Y, double Z) ExpectationPosition()
    {
        double ex = 0, ey = 0, ez = 0;
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    double prob = _psiRe[idx] * _psiRe[idx] + _psiIm[idx] * _psiIm[idx];
                    ex += x * prob;
                    ey += y * prob;
                    ez += z * prob;
                }
        double dV = _dx * _dx * _dx;
        return (ex * dV * _dx, ey * dV * _dx, ez * dV * _dx);
    }

    /// <summary>
    /// Compute expectation value of kinetic energy.
    /// ⟨T⟩ = −ℏ²/(2m) ∫ ψ* ∇²ψ dV
    /// </summary>
    public double ExpectationKineticEnergy()
    {
        double sum = 0;
        double coeff = -_hbar * _hbar / (2.0 * _m);
        double dx2 = _dx * _dx;

        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);
                    // Laplacian of Re and Im parts.
                    double lapRe = (_psiRe[Idx(x + 1, y, z)] + _psiRe[Idx(x - 1, y, z)] +
                                    _psiRe[Idx(x, y + 1, z)] + _psiRe[Idx(x, y - 1, z)] +
                                    (_nz > 1 ? _psiRe[Idx(x, y, z + 1)] + _psiRe[Idx(x, y, z - 1)] : 0) -
                                    (_nz > 1 ? 6.0 : 4.0) * _psiRe[idx]) / dx2;

                    double lapIm = (_psiIm[Idx(x + 1, y, z)] + _psiIm[Idx(x - 1, y, z)] +
                                    _psiIm[Idx(x, y + 1, z)] + _psiIm[Idx(x, y - 1, z)] +
                                    (_nz > 1 ? _psiIm[Idx(x, y, z + 1)] + _psiIm[Idx(x, y, z - 1)] : 0) -
                                    (_nz > 1 ? 6.0 : 4.0) * _psiIm[idx]) / dx2;

                    sum += _psiRe[idx] * lapRe + _psiIm[idx] * lapIm;
                }

        return coeff * sum * _dx * _dx * _dx;
    }

    /// <summary>
    /// Compute expectation value of potential energy: ⟨V⟩ = ∫ V|ψ|² dV.
    /// </summary>
    public double ExpectationPotentialEnergy()
    {
        double sum = 0;
        for (int i = 0; i < _n; i++)
            sum += _V[i] * (_psiRe[i] * _psiRe[i] + _psiIm[i] * _psiIm[i]);
        return sum * _dx * _dx * _dx;
    }

    /// <summary>
    /// Compute total energy: ⟨E⟩ = ⟨T⟩ + ⟨V⟩.
    /// </summary>
    public double TotalEnergy() => ExpectationKineticEnergy() + ExpectationPotentialEnergy();

    /// <summary>
    /// Compute expectation value of momentum: ⟨p⟩ = −iℏ ∫ ψ* ∇ψ dV.
    /// </summary>
    public (double Px, double Py, double Pz) ExpectationMomentum()
    {
        double px = 0, py = 0, pz = 0;
        double coeff = -_hbar;
        double dx2 = 2.0 * _dx;
        int dimFactor = _nz > 1 ? 6 : 4;

        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);
                    // ∂ψ/∂x ≈ (ψ(x+1) − ψ(x−1)) / 2dx
                    double dpsiReX = (_psiRe[Idx(x + 1, y, z)] - _psiRe[Idx(x - 1, y, z)]) / dx2;
                    double dpsiImX = (_psiIm[Idx(x + 1, y, z)] - _psiIm[Idx(x - 1, y, z)]) / dx2;
                    px += _psiRe[idx] * dpsiImX - _psiIm[idx] * dpsiReX;

                    double dpsiReY = (_psiRe[Idx(x, y + 1, z)] - _psiRe[Idx(x, y - 1, z)]) / dx2;
                    double dpsiImY = (_psiIm[Idx(x, y + 1, z)] - _psiIm[Idx(x, y - 1, z)]) / dx2;
                    py += _psiRe[idx] * dpsiImY - _psiIm[idx] * dpsiReY;

                    if (_nz > 1)
                    {
                        double dpsiReZ = (_psiRe[Idx(x, y, z + 1)] - _psiRe[Idx(x, y, z - 1)]) / dx2;
                        double dpsiImZ = (_psiIm[Idx(x, y, z + 1)] - _psiIm[Idx(x, y, z - 1)]) / dx2;
                        pz += _psiRe[idx] * dpsiImZ - _psiIm[idx] * dpsiReZ;
                    }
                }

        double dV = _dx * _dx * _dx;
        return (coeff * px * dV, coeff * py * dV, coeff * pz * dV);
    }

    // -----------------------------------------------------------------------
    //  Time-independent eigenstates (inverse iteration with shifts)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Find eigenstates using shifted inverse power iteration.
    /// For each eigenvalue, we shift Ĥ → (Ĥ − σI)⁻¹ and iterate.
    /// </summary>
    public void ComputeEigenstates()
    {
        int nEigen = _cfg.NumEigenstates;
        double dV = _dx * _dx * _dx;
        int nTotal = _n;
        int maxIter = _cfg.EigenMaxIter;
        double tol = _cfg.EigenTolerance;

        // Pre-compute the kinetic energy matrix as tridiagonal coefficients.
        double r = _hbar * _hbar / (2.0 * _m * _dx * _dx);

        // For each eigenstate, shift by an estimate and iterate.
        for (int n = 0; n < nEigen; n++)
        {
            // Initial guess: random.
            var rng = new Random(n * 12345 + 42);
            double norm = 0;
            for (int i = 0; i < nTotal; i++)
            {
                _eigenRe[n][i] = rng.NextDouble() - 0.5;
                _eigenIm[n][i] = 0;
                norm += _eigenRe[n][i] * _eigenRe[n][i];
            }
            norm = Math.Sqrt(norm);
            for (int i = 0; i < nTotal; i++)
                _eigenRe[n][i] /= norm;

            // Shift estimate: use Vmin + spacing.
            double vMin = double.MaxValue;
            for (int i = 0; i < nTotal; i++)
                if (_V[i] < vMin)
                    vMin = _V[i];
            double shift = vMin + (n + 1) * 1.0; // rough spacing

            for (int iter = 0; iter < maxIter; iter++)
            {
                // Apply (Ĥ − shift·I) to current eigenstate estimate.
                double[] rhsRe = new double[nTotal];
                double[] rhsIm = new double[nTotal];
                double[] solRe = new double[nTotal];
                double[] solIm = new double[nTotal];

                for (int z = 1; z < _nz - 1; z++)
                    for (int y = 1; y < _ny - 1; y++)
                        for (int x = 1; x < _nx - 1; x++)
                        {
                            int idx = Idx(x, y, z);
                            int dimF = _nz > 1 ? 6 : 4;

                            double lapRe = (_eigenRe[n][Idx(x + 1, y, z)] + _eigenRe[n][Idx(x - 1, y, z)] +
                                            _eigenRe[n][Idx(x, y + 1, z)] + _eigenRe[n][Idx(x, y - 1, z)] +
                                            (_nz > 1 ? _eigenRe[n][Idx(x, y, z + 1)] + _eigenRe[n][Idx(x, y, z - 1)] : 0) -
                                            dimF * _eigenRe[n][idx]) / (_dx * _dx);

                            rhsRe[idx] = r * lapRe + (_V[idx] - shift) * _eigenRe[n][idx];
                            rhsIm[idx] = r * lapRe + (_V[idx] - shift) * _eigenIm[n][idx];
                        }

                // Solve approximately: use Gauss-Seidel.
                double[] newRe = new double[nTotal];
                double[] newIm = new double[nTotal];
                Array.Copy(_eigenRe[n], newRe, nTotal);
                Array.Copy(_eigenIm[n], newIm, nTotal);

                for (int gsIter = 0; gsIter < 20; gsIter++)
                {
                    for (int z = 1; z < _nz - 1; z++)
                        for (int y = 1; y < _ny - 1; y++)
                            for (int x = 1; x < _nx - 1; x++)
                            {
                                int idx = Idx(x, y, z);
                                double diag = (_V[idx] - shift);
                                double dimF = _nz > 1 ? 6.0 : 4.0;
                                double diagVal = dimF * r / (_dx * _dx) + diag;

                                double sumRe = 0;
                                sumRe += r / (_dx * _dx) * (newRe[Idx(x + 1, y, z)] + newRe[Idx(x - 1, y, z)] +
                                                            newRe[Idx(x, y + 1, z)] + newRe[Idx(x, y - 1, z)]);
                                if (_nz > 1)
                                    sumRe += r / (_dx * _dx) * (newRe[Idx(x, y, z + 1)] + newRe[Idx(x, y, z - 1)]);

                                newRe[idx] = (rhsRe[idx] + sumRe) / diagVal;
                                newIm[idx] = (rhsIm[idx] + sumRe) / diagVal;
                            }
                }

                // Normalise.
                norm = 0;
                for (int i = 0; i < nTotal; i++)
                    norm += newRe[i] * newRe[i] + newIm[i] * newIm[i];
                norm = Math.Sqrt(norm * dV);
                if (norm < 1e-30)
                    break;

                for (int i = 0; i < nTotal; i++)
                {
                    newRe[i] /= norm;
                    newIm[i] /= norm;
                }

                // Check convergence.
                double change = 0;
                for (int i = 0; i < nTotal; i++)
                {
                    double dr = newRe[i] - _eigenRe[n][i];
                    double di = newIm[i] - _eigenIm[n][i];
                    change += dr * dr + di * di;
                }

                Array.Copy(newRe, _eigenRe[n], nTotal);
                Array.Copy(newIm, _eigenIm[n], nTotal);

                if (Math.Sqrt(change * dV) < tol)
                    break;
            }

            // Compute eigenvalue: E_n = ⟨ψ_n|Ĥ|ψ_n⟩ / ⟨ψ_n|ψ_n⟩.
            double energy = 0;
            for (int z = 1; z < _nz - 1; z++)
                for (int y = 1; y < _ny - 1; y++)
                    for (int x = 1; x < _nx - 1; x++)
                    {
                        int idx = Idx(x, y, z);
                        double lapRe = (_eigenRe[n][Idx(x + 1, y, z)] + _eigenRe[n][Idx(x - 1, y, z)] +
                                        _eigenRe[n][Idx(x, y + 1, z)] + _eigenRe[n][Idx(x, y - 1, z)] +
                                        (_nz > 1 ? _eigenRe[n][Idx(x, y, z + 1)] + _eigenRe[n][Idx(x, y, z - 1)] : 0) -
                                        (_nz > 1 ? 6.0 : 4.0) * _eigenRe[n][idx]) / (_dx * _dx);

                        energy += _eigenRe[n][idx] * (r * lapRe + _V[idx] * _eigenRe[n][idx]);
                    }
            _eigenValues[n] = energy * dV;

            // Orthogonalise against previously computed eigenstates.
            for (int m = 0; m < n; m++)
            {
                double dot = 0;
                for (int i = 0; i < nTotal; i++)
                    dot += _eigenRe[m][i] * _eigenRe[n][i] + _eigenIm[m][i] * _eigenIm[n][i];
                dot *= dV;
                for (int i = 0; i < nTotal; i++)
                {
                    _eigenRe[n][i] -= dot * _eigenRe[m][i];
                    _eigenIm[n][i] -= dot * _eigenIm[m][i];
                }
            }

            // Re-normalise.
            norm = 0;
            for (int i = 0; i < nTotal; i++)
                norm += _eigenRe[n][i] * _eigenRe[n][i] + _eigenIm[n][i] * _eigenIm[n][i];
            norm = Math.Sqrt(norm * dV);
            if (norm > 1e-30)
            {
                for (int i = 0; i < nTotal; i++)
                {
                    _eigenRe[n][i] /= norm;
                    _eigenIm[n][i] /= norm;
                }
            }
        }
    }

    /// <summary>
    /// Get the real part of a computed eigenstate.
    /// </summary>
    public ReadOnlySpan<double> GetEigenstateReal(int n) => _eigenRe[n];

    /// <summary>
    /// Get recorded probability history (if enabled).
    /// </summary>
    public IReadOnlyList<double[]> ProbabilityHistory => _probabilityHistory;

    /// <summary>
    /// Run the time-dependent simulation for the configured number of steps.
    /// </summary>
    public void RunTD()
    {
        for (int i = 0; i < _cfg.NumSteps; i++)
            StepTDSE();
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  8. ElasticitySolver — FEM-like grid elasticity, J2 plasticity, contact
// ============================================================================

/// <summary>
/// Material model for the elasticity solver.
/// </summary>
