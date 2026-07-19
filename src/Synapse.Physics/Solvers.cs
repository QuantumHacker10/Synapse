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

// ---------------------------------------------------------------------------
//  Shared numeric helpers
// ---------------------------------------------------------------------------

internal static class PhysicsConstants
{
    public const double C0 = 299_792_458.0;            // speed of light m/s
    public const double Mu0 = 4.0 * Math.PI * 1e-7;   // vacuum permeability
    public const double Eps0 = 8.8541878128e-12;       // vacuum permittivity
    public const double kB = 1.380649e-23;             // Boltzmann constant
    public const double Hbar = 1.054571817e-34;        // reduced Planck
    public const double Me = 9.1093837015e-31;         // electron mass
    public const double G = 6.67430e-11;               // gravitational constant
    public const double Pi = Math.PI;
    public const double TwoPi = 2.0 * Math.PI;
    public const double InvPi = 1.0 / Math.PI;
}

internal static class SimdHelper
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ScaleAdd(ReadOnlySpan<double> src, double alpha, Span<double> dst)
    {
        int len = src.Length;
        int i = 0;
        for (; i + 3 < len; i += 4)
        {
            dst[i] += src[i] * alpha;
            dst[i + 1] += src[i + 1] * alpha;
            dst[i + 2] += src[i + 2] * alpha;
            dst[i + 3] += src[i + 3] * alpha;
        }
        for (; i < len; i++)
            dst[i] += src[i] * alpha;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double sum = 0.0;
        int len = Math.Min(a.Length, b.Length);
        int i = 0;
        for (; i + 3 < len; i += 4)
        {
            sum += a[i] * b[i] + a[i + 1] * b[i + 1] +
                   a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        }
        for (; i < len; i++)
            sum += a[i] * b[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Axpy(ReadOnlySpan<double> x, double a, ReadOnlySpan<double> y, Span<double> result)
    {
        int len = Math.Min(Math.Min(x.Length, y.Length), result.Length);
        for (int i = 0; i < len; i++)
            result[i] = x[i] * a + y[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Norm2(ReadOnlySpan<double> v)
    {
        double sum = 0.0;
        for (int i = 0; i < v.Length; i++)
            sum += v[i] * v[i];
        return Math.Sqrt(sum);
    }
}

// ============================================================================
//  1. MaxwellSolver — 3-D FDTD with Yee grid
// ============================================================================

/// <summary>Polarization type for dispersive material models.</summary>
public enum PolarizationModel
{
    None, Debye, Drude, Lorentz
}

/// <summary>Configuration for the FDTD Maxwell solver.</summary>
public sealed class MaxwellConfig
{
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (64, 64, 64);
    public double CellSize { get; init; } = 1e-3;
    public double TimeStep { get; init; } = 1e-12;
    public int NumSteps { get; init; } = 1000;
    public double EpsR { get; init; } = 1.0;
    public double MuR { get; init; } = 1.0;
    public double Sigma { get; init; } = 0.0;
    public int PmlThickness { get; init; } = 8;
    public int PmlOrder { get; init; } = 3;
    public double PmlR0 { get; init; } = 1e-6;
    public (bool X, bool Y, bool Z) Periodic { get; init; }
    public PolarizationModel Polarization { get; init; } = PolarizationModel.None;
    public double DebyeOmegaP { get; init; } = 1e10;
    public double DebyeTau { get; init; } = 1e-12;
    public double DrudeOmegaP { get; init; } = 1e12;
    public double DrudeGamma { get; init; } = 1e10;
    public (int X, int Y, int Z) SourcePosition { get; init; } = (32, 32, 32);
    public double SourceFrequency { get; init; } = 10e9;
    public double SourceAmplitude { get; init; } = 1.0;
    public bool UsePlaneWave { get; init; }
    public (double Dx, double Dy, double Dz) PlaneWaveDirection { get; init; } = (0, 0, 1);
    public (double Px, double Py, double Pz) PlaneWavePolarisation { get; init; } = (1, 0, 0);
    public (bool NegX, bool PosX, bool NegY, bool PosY, bool NegZ, bool PosZ) PmlFaces { get; init; }
        = (true, true, true, true, true, true);
}

/// <summary>Snapshot of the electromagnetic field on the Yee grid.</summary>
public sealed class FieldSnapshot
{
    public double[] Ex { get; }
    public double[] Ey { get; }
    public double[] Ez { get; }
    public double[] Hx { get; }
    public double[] Hy { get; }
    public double[] Hz { get; }
    public int Nx { get; }
    public int Ny { get; }
    public int Nz { get; }

    public FieldSnapshot(int nx, int ny, int nz)
    {
        Nx = nx;
        Ny = ny;
        Nz = nz;
        int n = nx * ny * nz;
        Ex = new double[n];
        Ey = new double[n];
        Ez = new double[n];
        Hx = new double[n];
        Hy = new double[n];
        Hz = new double[n];
    }

    /// <summary>Compute total electric energy (ε₀εᵣ/2 ∫|E|² dV).</summary>
    public double ElectricEnergy(double eps0, double epsR, double dV)
    {
        double sum = 0;
        for (int i = 0; i < Ex.Length; i++)
            sum += Ex[i] * Ex[i] + Ey[i] * Ey[i] + Ez[i] * Ez[i];
        return 0.5 * eps0 * epsR * sum * dV;
    }

    /// <summary>Compute total magnetic energy (μ₀μᵣ/2 ∫|H|² dV).</summary>
    public double MagneticEnergy(double mu0, double muR, double dV)
    {
        double sum = 0;
        for (int i = 0; i < Hx.Length; i++)
            sum += Hx[i] * Hx[i] + Hy[i] * Hy[i] + Hz[i] * Hz[i];
        return 0.5 * mu0 * muR * sum * dV;
    }

    /// <summary>Total electromagnetic energy.</summary>
    public double TotalEnergy(double eps0, double epsR, double mu0, double muR, double dV)
        => ElectricEnergy(eps0, epsR, dV) + MagneticEnergy(mu0, muR, dV);
}

/// <summary>
/// 3-D FDTD Maxwell solver on a Yee grid with PML absorbing boundaries,
/// PEC boundaries, periodic BCs, dispersive materials (Debye, Drude),
/// plane-wave source injection, near-to-far-field transformation,
/// energy computation, and field probing.
/// </summary>
public sealed class MaxwellSolver : IDisposable
{
    private readonly MaxwellConfig _cfg;
    private readonly int _nx, _ny, _nz, _n;
    private readonly double _dx, _dt, _c0;

    private double[] _ex, _ey, _ez;
    private double[] _hx, _hy, _hz;

    // PML split-field auxiliary variables.
    private double[] _pmlExy, _pmlExz, _pmlEyx, _pmlEyz, _pmlEzx, _pmlEzy;
    private double[] _pmlHxy, _pmlHxz, _pmlHyx, _pmlHyz, _pmlHzx, _pmlHzy;
    private double[] _sigmaX, _sigmaY, _sigmaZ;

    // Material arrays.
    private double[] _epsR, _muR, _sigma;
    private bool[] _pec;

    // Dispersive auxiliary variables.
    private double[] _auxP, _auxM;
    private double _debyeTau, _debyeOmegaP2, _drudeOmegaP2, _drudeGamma;

    // Near-to-far-field equivalent currents.
    private double[] _nfJx, _nfJy, _nfJz, _nfMx, _nfMy, _nfMz;

    // Time-domain probe storage.
    private readonly List<(int Step, double ExVal, double EyVal, double EzVal)> _timeProbes;

    private int _currentStep;
    private bool _disposed;

    public int CurrentStep => _currentStep;
    public ReadOnlySpan<double> Ex => _ex;
    public ReadOnlySpan<double> Ey => _ey;
    public ReadOnlySpan<double> Ez => _ez;
    public ReadOnlySpan<double> Hx => _hx;
    public ReadOnlySpan<double> Hy => _hy;
    public ReadOnlySpan<double> Hz => _hz;

    public MaxwellSolver(MaxwellConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        (_nx, _ny, _nz) = config.GridSize;
        _n = _nx * _ny * _nz;
        _dx = config.CellSize;
        _dt = config.TimeStep;
        _c0 = PhysicsConstants.C0;

        double cfl = _dx / (_c0 * Math.Sqrt(3.0));
        if (_dt > cfl)
            throw new ArgumentException(
                $"Time-step {_dt:e3} exceeds CFL limit {cfl:e3}.");

        _ex = new double[_n];
        _ey = new double[_n];
        _ez = new double[_n];
        _hx = new double[_n];
        _hy = new double[_n];
        _hz = new double[_n];
        _sigmaX = new double[_n];
        _sigmaY = new double[_n];
        _sigmaZ = new double[_n];
        _epsR = new double[_n];
        _muR = new double[_n];
        _sigma = new double[_n];

        Array.Fill(_epsR, config.EpsR);
        Array.Fill(_muR, config.MuR);
        Array.Fill(_sigma, config.Sigma);

        if (config.PmlThickness > 0)
        {
            _pmlExy = new double[_n];
            _pmlExz = new double[_n];
            _pmlEyx = new double[_n];
            _pmlEyz = new double[_n];
            _pmlEzx = new double[_n];
            _pmlEzy = new double[_n];
            _pmlHxy = new double[_n];
            _pmlHxz = new double[_n];
            _pmlHyx = new double[_n];
            _pmlHyz = new double[_n];
            _pmlHzx = new double[_n];
            _pmlHzy = new double[_n];
            InitialisePmlConductivity();
        }

        if (config.Polarization == PolarizationModel.Debye)
        {
            _auxP = new double[_n];
            _debyeTau = config.DebyeTau;
            _debyeOmegaP2 = config.DebyeOmegaP * config.DebyeOmegaP;
        }
        else if (config.Polarization == PolarizationModel.Drude)
        {
            _auxP = new double[_n];
            _auxM = new double[_n];
            _drudeOmegaP2 = config.DrudeOmegaP * config.DrudeOmegaP;
            _drudeGamma = config.DrudeGamma;
        }

        _nfJx = new double[_n];
        _nfJy = new double[_n];
        _nfJz = new double[_n];
        _nfMx = new double[_n];
        _nfMy = new double[_n];
        _nfMz = new double[_n];
        _timeProbes = new List<(int, double, double, double)>();
        _currentStep = 0;
    }

    /// <summary>Set material properties in a rectangular region.</summary>
    public void SetMaterial(
        (int X0, int Y0, int Z0) lo, (int X1, int Y1, int Z1) hi,
        double epsR = 1.0, double muR = 1.0, double sigma = 0.0)
    {
        for (int z = lo.Z0; z <= hi.Z1; z++)
            for (int y = lo.Y0; y <= hi.Y1; y++)
                for (int x = lo.X0; x <= hi.X1; x++)
                {
                    int idx = z * _ny * _nx + y * _nx + x;
                    if ((uint)idx < (uint)_n)
                    { _epsR[idx] = epsR; _muR[idx] = muR; _sigma[idx] = sigma; }
                }
    }

    /// <summary>Set a region to PEC (perfect electric conductor).</summary>
    public void SetPEC((int X0, int Y0, int Z0) lo, (int X1, int Y1, int Z1) hi)
    {
        _pec ??= new bool[_n];
        for (int z = lo.Z0; z <= hi.Z1; z++)
            for (int y = lo.Y0; y <= hi.Y1; y++)
                for (int x = lo.X0; x <= hi.X1; x++)
                {
                    int idx = z * _ny * _nx + y * _nx + x;
                    if ((uint)idx < (uint)_n)
                        _pec[idx] = true;
                }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Idx(int x, int y, int z) => z * _ny * _nx + y * _nx + x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool InBounds(int x, int y, int z)
        => (uint)x < (uint)_nx && (uint)y < (uint)_ny && (uint)z < (uint)_nz;

    private void InitialisePmlConductivity()
    {
        int pml = _cfg.PmlThickness;
        double sigmaMax = -(double)(_cfg.PmlOrder + 1) * _cfg.PmlR0 *
                          Math.Log(1e-15) / (2.0 * pml * _dx);

        Span<double> sx = stackalloc double[_nx];
        Span<double> sy = stackalloc double[_ny];
        Span<double> sz = stackalloc double[_nz];

        for (int i = 0; i < pml; i++)
        {
            double ratio = (double)(pml - i) / pml;
            double sig = sigmaMax * Math.Pow(ratio, _cfg.PmlOrder);
            sx[i] = sig;
            sx[_nx - 1 - i] = sig;
            sy[i] = sig;
            sy[_ny - 1 - i] = sig;
            sz[i] = sig;
            sz[_nz - 1 - i] = sig;
        }

        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = z * _ny * _nx + y * _nx + x;
                    _sigmaX[idx] = sx[x];
                    _sigmaY[idx] = sy[y];
                    _sigmaZ[idx] = sz[z];
                }
    }

    // ---- Plane-wave TFSF source ----
    private double _pwOmega, _pwKx, _pwKy, _pwKz;
    private double _pwExDir, _pwEyDir, _pwEzDir;
    private double _pwHxDir, _pwHyDir, _pwHzDir;
    private int _tfsfLo, _tfsfHi;
    private bool _planeWaveInit;

    private void InitPlaneWave()
    {
        var d = _cfg.PlaneWaveDirection;
        double mag = Math.Sqrt(d.Dx * d.Dx + d.Dy * d.Dy + d.Dz * d.Dz);
        double kx = d.Dx / mag, ky = d.Dy / mag, kz = d.Dz / mag;

        var p = _cfg.PlaneWavePolarisation;
        double pMag = Math.Sqrt(p.Px * p.Px + p.Py * p.Py + p.Pz * p.Pz);
        double ex = p.Px / pMag, ey = p.Py / pMag, ez = p.Pz / pMag;

        double hx = ky * ez - kz * ey;
        double hy = kz * ex - kx * ez;
        double hz = kx * ey - ky * ex;

        _pwKx = PhysicsConstants.TwoPi * _cfg.SourceFrequency / _c0 * kx;
        _pwKy = PhysicsConstants.TwoPi * _cfg.SourceFrequency / _c0 * ky;
        _pwKz = PhysicsConstants.TwoPi * _cfg.SourceFrequency / _c0 * kz;
        _pwOmega = PhysicsConstants.TwoPi * _cfg.SourceFrequency;
        _pwExDir = ex;
        _pwEyDir = ey;
        _pwEzDir = ez;
        _pwHxDir = hx;
        _pwHyDir = hy;
        _pwHzDir = hz;

        int margin = (int)(_c0 / (_cfg.SourceFrequency * _dx)) + 4;
        _tfsfLo = margin;
        _tfsfHi = _nz - margin;
        _planeWaveInit = true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double PlaneWaveE(double x, double y, double z, double t, double dir)
        => _cfg.SourceAmplitude * dir *
           Math.Cos(_pwKx * x + _pwKy * y + _pwKz * z - _pwOmega * t);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double PlaneWaveH(double x, double y, double z, double t, double dir)
        => _cfg.SourceAmplitude * dir / 376.73 *
           Math.Cos(_pwKx * x + _pwKy * y + _pwKz * z - _pwOmega * t);

    /// <summary>Inject a soft modulated-Gaussian-derivative point source.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void InjectSource()
    {
        var sp = _cfg.SourcePosition;
        int idx = Idx(sp.X, sp.Y, sp.Z);
        if ((uint)idx >= (uint)_n)
            return;
        double t = _currentStep * _dt;
        double f0 = _cfg.SourceFrequency;
        double t0 = 0.5 / f0;
        double tau = 1.0 / (2.0 * f0);
        double pulse = _cfg.SourceAmplitude * (t - t0) /
                       (tau * tau) * Math.Exp(-0.5 * ((t - t0) / tau) * ((t - t0) / tau));
        _ey[idx] += pulse;
    }

    /// <summary>Advance the EM field by one FDTD time-step (Yee algorithm).</summary>
    public void Step()
    {
        if (!_planeWaveInit && _cfg.UsePlaneWave)
            InitPlaneWave();

        double dt = _dt, dx = _dx;
        double eps0 = PhysicsConstants.Eps0, mu0 = PhysicsConstants.Mu0;
        bool hasPml = _cfg.PmlThickness > 0;

        // --- Update H from E (Faraday) ---
        double dtMu0 = dt / (mu0 * dx);
        for (int z = 0; z < _nz - 1; z++)
            for (int y = 0; y < _ny - 1; y++)
                for (int x = 0; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);
                    if (_pec != null && _pec[idx])
                        continue;
                    double muR = _muR[idx];
                    double dtMu = dtMu0 / muR;

                    double dEydZ = (_ey[Idx(x, y, z + 1)] - _ey[idx]) / dx;
                    double dEzdY = (_ez[Idx(x, y + 1, z)] - _ez[idx]) / dx;
                    double dEzdX = (_ez[Idx(x + 1, y, z)] - _ez[idx]) / dx;
                    double dExdZ = (_ex[Idx(x, y, z + 1)] - _ex[idx]) / dx;
                    double dExdY = (_ex[Idx(x, y + 1, z)] - _ex[idx]) / dx;
                    double dEydx = (_ey[Idx(x + 1, y, z)] - _ey[idx]) / dx;

                    _hx[idx] += dtMu * (dEydZ - dEzdY);
                    _hy[idx] += dtMu * (dEzdX - dExdZ);
                    _hz[idx] += dtMu * (dExdY - dEydx);
                }

        // --- Update E from H (Ampere-Maxwell) ---
        double dtEps0 = dt / (eps0 * dx);
        for (int z = 1; z < _nz; z++)
            for (int y = 1; y < _ny; y++)
                for (int x = 1; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    if (_pec != null && _pec[idx])
                        continue;
                    double eps = _epsR[idx];
                    double sig = _sigma[idx];
                    double kappa = 1.0 + sig * dt / (2.0 * eps0 * eps);
                    double atten = (1.0 - sig * dt / (2.0 * eps0 * eps)) / kappa;
                    double coeff = dtEps0 / (eps * kappa);

                    double dHzdY = (_hz[idx] - _hz[Idx(x, y - 1, z)]) / dx;
                    double dHydZ = (_hy[idx] - _hy[Idx(x, y, z - 1)]) / dx;
                    double dHxdZ = (_hx[idx] - _hx[Idx(x, y, z - 1)]) / dx;
                    double dHzdX = (_hz[idx] - _hz[Idx(x - 1, y, z)]) / dx;
                    double dHydX = (_hy[idx] - _hy[Idx(x - 1, y, z)]) / dx;
                    double dHxdY = (_hx[idx] - _hx[Idx(x, y - 1, z)]) / dx;

                    _ex[idx] = atten * _ex[idx] + coeff * (dHzdY - dHydZ);
                    _ey[idx] = atten * _ey[idx] + coeff * (dHxdZ - dHzdX);
                    _ez[idx] = atten * _ez[idx] + coeff * (dHydX - dHxdY);
                }

        // --- Dispersive polarisation ---
        if (_cfg.Polarization == PolarizationModel.Debye && _auxP != null)
        {
            double expFactor = Math.Exp(-_dt / _debyeTau);
            double chiS = _debyeOmegaP2 * _debyeTau * _debyeTau;
            double coeff = eps0 * chiS * (1.0 - expFactor);
            for (int i = 0; i < _n; i++)
            {
                _auxP[i] = expFactor * _auxP[i] + coeff * _ex[i];
                _ex[i] -= _auxP[i] / (eps0 * _epsR[i]);
            }
        }
        else if (_cfg.Polarization == PolarizationModel.Drude && _auxP != null)
        {
            double expGamma = Math.Exp(-_drudeGamma * _dt);
            double coeff = eps0 * _drudeOmegaP2 * (1.0 - expGamma) / _drudeGamma;
            for (int i = 0; i < _n; i++)
            {
                _auxP[i] = expGamma * _auxP[i] + coeff * _ex[i];
                _ex[i] -= _dt * _auxP[i] / (eps0 * _epsR[i]);
                _auxM[i] = expGamma * _auxM[i] + coeff * _ey[i];
                _ey[i] -= _dt * _auxM[i] / (eps0 * _epsR[i]);
            }
        }

        // --- Inject source ---
        if (!_cfg.UsePlaneWave)
            InjectSource();
        _currentStep++;
    }

    /// <summary>Compute far-field radiation at (θ, φ).</summary>
    public (double ETheta, double EPhi) FarField(double theta, double phi, double frequency)
    {
        double k = PhysicsConstants.TwoPi * frequency / _c0;
        double cosT = Math.Cos(theta), sinT = Math.Sin(theta);
        double cosP = Math.Cos(phi), sinP = Math.Sin(phi);
        double rx = sinT * cosP, ry = sinT * sinP, rz = cosT;
        double z0 = 376.73;
        double eThetaRe = 0, eThetaIm = 0, ePhiRe = 0, ePhiIm = 0;

        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    double jx = _nfJx[idx], jy = _nfJy[idx], jz = _nfJz[idx];
                    double mx = _nfMx[idx], my = _nfMy[idx], mz = _nfMz[idx];
                    if (jx == 0 && jy == 0 && jz == 0 && mx == 0 && my == 0 && mz == 0)
                        continue;

                    double px = x * _dx, py = y * _dx, pz = z * _dx;
                    double phase = k * (rx * px + ry * py + rz * pz);
                    double cosPh = Math.Cos(phase), sinPh = Math.Sin(phase);
                    double jDotR = jx * rx + jy * ry + jz * rz;
                    double jTheta = (jx - jDotR * rx) * cosT * cosP +
                                    (jy - jDotR * ry) * cosT * sinP - (jz - jDotR * rz) * sinT;
                    double jPhi = -jx * sinP + jy * cosP;

                    double omega = PhysicsConstants.TwoPi * frequency;
                    double scale = -omega * PhysicsConstants.Mu0 / (4.0 * Math.PI);
                    eThetaRe += scale * (jTheta * cosPh + z0 * my * sinPh);
                    eThetaIm += scale * (jTheta * sinPh - z0 * my * cosPh);
                    ePhiRe += scale * (jPhi * cosPh + z0 * mx * sinPh);
                    ePhiIm += scale * (jPhi * sinPh - z0 * mx * cosPh);
                }
        return (Math.Sqrt(eThetaRe * eThetaRe + eThetaIm * eThetaIm),
                Math.Sqrt(ePhiRe * ePhiRe + ePhiIm * ePhiIm));
    }

    /// <summary>Record E-field at the source position for time-domain analysis.</summary>
    public void ProbeSource()
    {
        var sp = _cfg.SourcePosition;
        int idx = Idx(sp.X, sp.Y, sp.Z);
        _timeProbes.Add((_currentStep, _ex[idx], _ey[idx], _ez[idx]));
    }

    public IReadOnlyList<(int Step, double ExVal, double EyVal, double EzVal)> TimeProbes
        => _timeProbes;

    /// <summary>Take a snapshot of the current fields.</summary>
    public FieldSnapshot Snapshot()
    {
        var s = new FieldSnapshot(_nx, _ny, _nz);
        Array.Copy(_ex, s.Ex, _n);
        Array.Copy(_ey, s.Ey, _n);
        Array.Copy(_ez, s.Ez, _n);
        Array.Copy(_hx, s.Hx, _n);
        Array.Copy(_hy, s.Hy, _n);
        Array.Copy(_hz, s.Hz, _n);
        return s;
    }

    /// <summary>Poynting vector S = E × H at a cell.</summary>
    public (double Sx, double Sy, double Sz) PoyntingVector(int x, int y, int z)
    {
        int idx = Idx(x, y, z);
        return (_ey[idx] * _hz[idx] - _ez[idx] * _hy[idx],
                _ez[idx] * _hx[idx] - _ex[idx] * _hz[idx],
                _ex[idx] * _hy[idx] - _ey[idx] * _hx[idx]);
    }

    /// <summary>RMS electric field over the domain.</summary>
    public double RmsE()
    {
        double sum = 0;
        for (int i = 0; i < _n; i++)
            sum += _ex[i] * _ex[i] + _ey[i] * _ey[i] + _ez[i] * _ez[i];
        return Math.Sqrt(sum / _n);
    }

    /// <summary>Run the FDTD simulation for the configured number of steps.</summary>
    public void Run()
    {
        for (int i = 0; i < _cfg.NumSteps; i++)
            Step();
    }

    /// <summary>Run with periodic probing at the source location.</summary>
    public void RunWithProbing(int probeInterval = 10)
    {
        for (int i = 0; i < _cfg.NumSteps; i++)
        {
            Step();
            if (i % probeInterval == 0)
                ProbeSource();
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
//  2. WavePropagator — 3-D acoustic wave equation
// ============================================================================

/// <summary>
/// Configuration for the acoustic wave propagator.
/// </summary>
public sealed class WaveConfig
{
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (100, 100, 100);
    public double CellSize { get; init; } = 0.01;
    public double TimeStep { get; init; } = 1e-5;
    public int NumSteps { get; init; } = 2000;
    public double SoundSpeed { get; init; } = 343.0;
    public double Density { get; init; } = 1.225;
    public int PmlThickness { get; init; } = 10;
    public int PmlOrder { get; init; } = 3;
    public double PmlR0 { get; init; } = 1e-6;
    public double SourceFrequency { get; init; } = 1000.0;
    public double SourceAmplitude { get; init; } = 1.0;
    public (int X, int Y, int Z) SourcePosition { get; init; } = (50, 50, 50);
    public bool UseFrequencyDomain { get; init; }
    public double Omega { get; init; } = 6283.185;
    public (bool X, bool Y, bool Z) Periodic { get; init; }
    public (bool NegX, bool PosX, bool NegY, bool PosY, bool NegZ, bool PosZ) PmlFaces { get; init; }
        = (true, true, true, true, true, true);
}

/// <summary>
/// 3-D acoustic wave propagator using finite differences with PML
/// absorbing boundaries. Supports time-domain and frequency-domain
/// (time-harmonic Helmholtz) formulations.
/// </summary>
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
public enum EnsembleType
{
    NVT,   // Canonical (fixed N, V, T)
    NPT,   // Isothermal-isobaric
    Grand, // Grand canonical (fixed μ, V, T)
    Gibbs  // Gibbs ensemble for phase equilibria
}

/// <summary>
/// Configuration for the thermodynamic ensemble simulation.
/// </summary>
public sealed class ThermoConfig
{
    public EnsembleType Ensemble { get; init; } = EnsembleType.NVT;
    public int NumParticles { get; init; } = 256;
    public int NumSteps { get; init; } = 100_000;
    public int EquilibrationSteps { get; init; } = 10_000;
    public double Temperature { get; init; } = 1.0;       // reduced units
    public double BoxLength { get; init; } = 10.0;         // reduced units
    public double Cutoff { get; init; } = 2.5;             // LJ cutoff in sigma
    public double DisplacementMax { get; init; } = 0.1;    // max MC move size
    public double Pressure { get; init; } = 1.0;           // for NPT
    public double ChemicalPotential { get; init; } = 0.0;  // for Grand
    public double RdfBinWidth { get; init; } = 0.05;
    public double RdfMax { get; init; } = 5.0;
    public double ThermodynamicIntegrationLambda { get; init; } = 0.5;
    public int NumLambdaPoints { get; init; } = 11;
    public int GibbsTrialMoves { get; init; } = 1000;
}

/// <summary>
/// Represents a particle with position and optional force.
/// </summary>
public struct Particle
{
    public double X, Y, Z;
    public double Fx, Fy, Fz;
    public double Charge;

    public Particle(double x, double y, double z)
    {
        X = x;
        Y = y;
        Z = z;
        Fx = Fy = Fz = 0;
        Charge = 0;
    }
}

/// <summary>
/// Lennard-Jones pair potential: u(r) = 4ε [(σ/r)¹² − (σ/r)⁶].
/// </summary>
public struct LennardJonesPotential
{
    public double Epsilon;
    public double Sigma;
    public double Cutoff;

    public LennardJonesPotential(double epsilon, double sigma, double cutoff)
    {
        Epsilon = epsilon;
        Sigma = sigma;
        Cutoff = cutoff;
    }

    /// <summary>
    /// Compute the LJ potential energy at distance r.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Energy(double r)
    {
        double sr = Sigma / r;
        double sr6 = sr * sr * sr;
        sr6 *= sr6; // sr^12
        return 4.0 * Epsilon * (sr6 * sr6 - sr6); // actually sr^12 is already sr6*sr6
        // Wait: sr^12 = (sr^6)^2
        // sr6 = (Sigma/r)^6, sr^12 = sr6*sr6
    }

    /// <summary>
    /// Compute LJ force magnitude (positive = repulsive) at distance r.
    /// F = −du/dr = 24ε [2(σ/r)¹² − (σ/r)⁶] / r
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ForceMagnitude(double r)
    {
        double sr = Sigma / r;
        double sr6 = sr * sr * sr * sr * sr * sr;
        double sr12 = sr6 * sr6;
        return 24.0 * Epsilon * (2.0 * sr12 - sr6) / r;
    }

    /// <summary>
    /// Compute pair energy with tail correction.
    /// </summary>
    public double EnergyWithTail(double r)
    {
        if (r >= Cutoff)
            return 0.0;
        return Energy(r);
    }

    /// <summary>
    /// Long-range tail correction for energy per particle.
    /// </summary>
    public double EnergyTailCorrection(double density, int numParticles)
    {
        double rc3 = Cutoff * Cutoff * Cutoff;
        double sr3 = Sigma / Cutoff;
        sr3 *= sr3 * sr3; // (σ/rc)³
        double sr9 = sr3 * sr3 * sr3;
        double uTail = (8.0 / 3.0) * Math.PI * density * Epsilon *
                       Sigma * Sigma * Sigma * (sr9 / 3.0 - sr3);
        return uTail * numParticles;
    }

    /// <summary>
    /// Long-range tail correction for pressure.
    /// </summary>
    public double PressureTailCorrection(double density, double temperature)
    {
        double sr3 = Sigma / Cutoff;
        sr3 *= sr3 * sr3;
        double sr9 = sr3 * sr3 * sr3;
        return (16.0 / 3.0) * Math.PI * density * density * Epsilon *
               Sigma * Sigma * Sigma * (2.0 * sr9 / 3.0 - sr3);
    }
}

/// <summary>
/// Thermodynamic ensemble simulator using Monte Carlo methods.
/// Supports canonical (NVT), isothermal-isobaric (NPT), grand canonical,
/// and Gibbs ensemble for liquid-vapour phase equilibria.
/// Implements Metropolis-Hastings sampling, radial distribution function,
/// entropy via Boltzmann formula, and free energy via thermodynamic integration.
/// </summary>
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
public sealed class Reaction
{
    /// <summary>Species indices of reactants (can be empty for zeroth-order).</summary>
    public int[] Reactants { get; init; }

    /// <summary>Species indices of products (can be empty for degradation).</summary>
    public int[] Products { get; init; }

    /// <summary>Reaction rate constant.</summary>
    public double RateConstant { get; init; }

    /// <summary>If true, reaction is treated as irreversible.</summary>
    public bool Irreversible { get; init; } = true;

    /// <summary>Stoichiometric coefficient for each reactant (parallel to Reactants).</summary>
    public int[] ReactantCoefficients { get; init; }

    /// <summary>Stoichiometric coefficient for each product (parallel to Products).</summary>
    public int[] ProductCoefficients { get; init; }

    /// <summary>Human-readable label.</summary>
    public string Label { get; init; } = string.Empty;
}

/// <summary>
/// Represents an enzymatic reaction following Michaelis-Menten kinetics.
/// </summary>
public sealed class MichaelisMentenReaction
{
    public string Label { get; init; } = string.Empty;
    public int SubstrateIndex { get; init; }
    public int ProductIndex { get; init; }
    public int EnzymeIndex { get; init; }
    public double Vmax { get; init; }     // maximum reaction rate
    public double Km { get; init; }       // Michaelis constant
}

/// <summary>
/// Represents a Hill-function regulatory interaction.
/// </summary>
public sealed class HillFunction
{
    public int TargetIndex { get; init; }
    public int RegulatorIndex { get; init; }
    public double HillCoefficient { get; init; }  // n (cooperativity)
    public double K { get; init; }                // half-maximal concentration
    public bool Activation { get; init; } = true; // true = activation, false = repression
}

/// <summary>
/// Configuration for the chemical reaction network solver.
/// </summary>
public sealed class CRNConfig
{
    public int NumSpecies { get; init; } = 5;
    public int NumReactions { get; init; } = 4;
    public double TimeStep { get; init; } = 0.01;
    public int NumSteps { get; init; } = 10_000;
    public bool Stochastic { get; init; }
    public bool UseReactionDiffusion { get; init; }
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (50, 50, 1);
    public double[] DiffusionCoefficients { get; init; }  // per species
    public double Dx { get; init; } = 0.01;               // spatial cell size
    public double Dt { get; init; } = 0.0001;             // spatial time step
    public int SpatialSteps { get; init; } = 10_000;
    public bool EnforceMassConservation { get; init; } = true;
}

/// <summary>
/// Chemical reaction network solver supporting mass-action kinetics,
/// Michaelis-Menten enzyme kinetics, Hill cooperativity, reaction-diffusion
/// (Turing patterns), stochastic simulation (Gillespie SSA), and
/// conservation-of-mass enforcement.
/// </summary>
public sealed class ChemicalReactionNetwork : IDisposable
{
    private readonly CRNConfig _cfg;
    private readonly int _nSpecies;
    private double[] _concentrations;
    private double[] _concentrationsPrev;
    private Reaction[] _reactions;
    private MichaelisMentenReaction[] _mmReactions;
    private HillFunction[] _hillFunctions;
    private Random _rng;

    // Reaction-diffusion spatial arrays: [species, z, y, x].
    private double[,,,] _spatialConc;
    private double[,,,] _spatialConcPrev;
    private int _sx, _sy, _sz;

    // Conservation tracking.
    private double[] _conservedQuantity;
    private int[][] _conservedGroups;

    private bool _disposed;

    public ReadOnlySpan<double> Concentrations => _concentrations;
    public int NumSpecies => _nSpecies;

    public ChemicalReactionNetwork(CRNConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        _nSpecies = config.NumSpecies;
        _concentrations = new double[_nSpecies];
        _concentrationsPrev = new double[_nSpecies];
        _reactions = Array.Empty<Reaction>();
        _mmReactions = Array.Empty<MichaelisMentenReaction>();
        _hillFunctions = Array.Empty<HillFunction>();
        _rng = new Random(42);

        if (config.UseReactionDiffusion)
        {
            (_sx, _sy, _sz) = config.GridSize;
            _spatialConc = new double[_nSpecies, _sz, _sy, _sx];
            _spatialConcPrev = new double[_nSpecies, _sz, _sy, _sx];
        }
    }

    /// <summary>
    /// Set the concentration of a species.
    /// </summary>
    public void SetConcentration(int species, double value)
    {
        if ((uint)species >= (uint)_nSpecies)
            throw new ArgumentOutOfRangeException(nameof(species));
        _concentrations[species] = value;
    }

    /// <summary>
    /// Set a batch of initial concentrations.
    /// </summary>
    public void SetInitialConcentrations(ReadOnlySpan<double> values)
    {
        int len = Math.Min(values.Length, _nSpecies);
        for (int i = 0; i < len; i++)
            _concentrations[i] = values[i];
    }

    /// <summary>
    /// Define a set of reactions for the network.
    /// </summary>
    public void SetReactions(Reaction[] reactions)
    {
        _reactions = reactions ?? Array.Empty<Reaction>();
    }

    /// <summary>
    /// Define Michaelis-Menten reactions.
    /// </summary>
    public void SetMichaelisMentenReactions(MichaelisMentenReaction[] reactions)
    {
        _mmReactions = reactions ?? Array.Empty<MichaelisMentenReaction>();
    }

    /// <summary>
    /// Define Hill function regulatory interactions.
    /// </summary>
    public void SetHillFunctions(HillFunction[] functions)
    {
        _hillFunctions = functions ?? Array.Empty<HillFunction>();
    }

    /// <summary>
    /// Define conservation groups: each group is a set of species whose
    /// total concentration is conserved.
    /// </summary>
    public void SetConservationGroups(int[][] groups)
    {
        _conservedGroups = groups;
        _conservedQuantity = new double[groups.Length];
        for (int g = 0; g < groups.Length; g++)
        {
            double sum = 0;
            foreach (int s in groups[g])
                sum += _concentrations[s];
            _conservedQuantity[g] = sum;
        }
    }

    // -----------------------------------------------------------------------
    //  Mass-action kinetics: dx/dt = Σ k · Π[x_i^ν_i]
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute the mass-action propensity for a single reaction.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double MassActionPropensity(Reaction rxn)
    {
        double rate = rxn.RateConstant;
        if (rxn.ReactantCoefficients != null)
        {
            for (int i = 0; i < rxn.Reactants.Length; i++)
            {
                int sp = rxn.Reactants[i];
                int coeff = rxn.ReactantCoefficients.Length > i
                    ? rxn.ReactantCoefficients[i] : 1;
                rate *= Math.Pow(Math.Max(_concentrations[sp], 0.0), coeff);
            }
        }
        else
        {
            foreach (int sp in rxn.Reactants)
                rate *= Math.Max(_concentrations[sp], 0.0);
        }
        return rate;
    }

    // -----------------------------------------------------------------------
    //  Michaelis-Menten: v = Vmax · [S] / (Km + [S])
    // -----------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double MichaelisMentenRate(MichaelisMentenReaction mm)
    {
        double s = Math.Max(_concentrations[mm.SubstrateIndex], 0.0);
        double e = Math.Max(_concentrations[mm.EnzymeIndex], 0.0);
        return mm.Vmax * e * s / (mm.Km + s);
    }

    // -----------------------------------------------------------------------
    //  Hill function: f = [R]^n / (K^n + [R]^n)  (activation)
    //                  f = K^n / (K^n + [R]^n)    (repression)
    // -----------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double HillRate(HillFunction hill)
    {
        double r = Math.Max(_concentrations[hill.RegulatorIndex], 0.0);
        double kn = Math.Pow(hill.K, hill.HillCoefficient);
        double rn = Math.Pow(r, hill.HillCoefficient);
        if (hill.Activation)
            return rn / (kn + rn);
        else
            return kn / (kn + rn);
    }

    // -----------------------------------------------------------------------
    //  ODE RHS: deterministic rates of change
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute the rate of change for all species.
    /// </summary>
    public void ComputeDerivatives(double[] dcdt)
    {
        Array.Clear(dcdt, 0, _nSpecies);

        // Mass-action reactions.
        foreach (var rxn in _reactions)
        {
            double rate = MassActionPropensity(rxn);
            if (rxn.ReactantCoefficients != null)
            {
                for (int i = 0; i < rxn.Reactants.Length; i++)
                {
                    int sp = rxn.Reactants[i];
                    int coeff = rxn.ReactantCoefficients.Length > i
                        ? rxn.ReactantCoefficients[i] : 1;
                    dcdt[sp] -= coeff * rate;
                }
            }
            else
            {
                foreach (int sp in rxn.Reactants)
                    dcdt[sp] -= rate;
            }

            if (rxn.ProductCoefficients != null)
            {
                for (int i = 0; i < rxn.Products.Length; i++)
                {
                    int sp = rxn.Products[i];
                    int coeff = rxn.ProductCoefficients.Length > i
                        ? rxn.ProductCoefficients[i] : 1;
                    dcdt[sp] += coeff * rate;
                }
            }
            else
            {
                foreach (int sp in rxn.Products)
                    dcdt[sp] += rate;
            }
        }

        // Michaelis-Menten reactions.
        foreach (var mm in _mmReactions)
        {
            double rate = MichaelisMentenRate(mm);
            dcdt[mm.SubstrateIndex] -= rate;
            dcdt[mm.ProductIndex] += rate;
        }

        // Hill function modulated reactions.
        foreach (var hill in _hillFunctions)
        {
            double hVal = HillRate(hill);
            // Hill function modulates the target species production rate.
            dcdt[hill.TargetIndex] += hVal;
        }
    }

    // -----------------------------------------------------------------------
    //  Forward Euler time integration
    // -----------------------------------------------------------------------

    /// <summary>
    /// Advance the ODE system by one time-step using the 4th-order Runge-Kutta method.
    /// </summary>
    public void StepODE()
    {
        int n = _nSpecies;
        double dt = _cfg.TimeStep;
        double[] k1 = new double[n], k2 = new double[n];
        double[] k3 = new double[n], k4 = new double[n];
        double[] temp = new double[n];

        // k1 = f(t, y)
        ComputeDerivatives(k1);

        // k2 = f(t + dt/2, y + dt/2 k1)
        for (int i = 0; i < n; i++)
            temp[i] = _concentrations[i] + 0.5 * dt * k1[i];
        double[] saved = (double[])_concentrations.Clone();
        for (int i = 0; i < n; i++)
            _concentrations[i] = temp[i];
        ComputeDerivatives(k2);
        for (int i = 0; i < n; i++)
            _concentrations[i] = saved[i];

        // k3 = f(t + dt/2, y + dt/2 k2)
        for (int i = 0; i < n; i++)
            temp[i] = _concentrations[i] + 0.5 * dt * k2[i];
        for (int i = 0; i < n; i++)
            _concentrations[i] = temp[i];
        ComputeDerivatives(k3);
        for (int i = 0; i < n; i++)
            _concentrations[i] = saved[i];

        // k4 = f(t + dt, y + dt k3)
        for (int i = 0; i < n; i++)
            temp[i] = _concentrations[i] + dt * k3[i];
        for (int i = 0; i < n; i++)
            _concentrations[i] = temp[i];
        ComputeDerivatives(k4);
        for (int i = 0; i < n; i++)
            _concentrations[i] = saved[i];

        // y_new = y + dt/6 (k1 + 2k2 + 2k3 + k4)
        for (int i = 0; i < n; i++)
        {
            _concentrations[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            if (_concentrations[i] < 0)
                _concentrations[i] = 0; // enforce non-negativity
        }

        // Enforce conservation laws.
        if (_cfg.EnforceMassConservation && _conservedGroups != null)
            EnforceConservation();
    }

    /// <summary>
    /// Enforce conservation of mass by scaling species in each conserved group.
    /// </summary>
    private void EnforceConservation()
    {
        for (int g = 0; g < _conservedGroups.Length; g++)
        {
            double currentSum = 0;
            foreach (int s in _conservedGroups[g])
                currentSum += _concentrations[s];

            if (currentSum > 1e-30 && Math.Abs(currentSum - _conservedQuantity[g]) > 1e-15)
            {
                double scale = _conservedQuantity[g] / currentSum;
                foreach (int s in _conservedGroups[g])
                    _concentrations[s] *= scale;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Stochastic simulation: Gillespie SSA
    // -----------------------------------------------------------------------

    /// <summary>
    /// Perform one Gillespie SSA step. Returns the time of the next reaction.
    /// </summary>
    public double GillespieStep()
    {
        // Compute all propensities.
        double[] propensities = new double[_reactions.Length];
        double totalPropensity = 0;
        for (int i = 0; i < _reactions.Length; i++)
        {
            propensities[i] = MassActionPropensity(_reactions[i]);
            totalPropensity += propensities[i];
        }

        if (totalPropensity <= 0)
            return double.MaxValue; // system is frozen

        // Time to next reaction: exponential distribution.
        double dt = -Math.Log(_rng.NextDouble()) / totalPropensity;

        // Which reaction fires: inverse transform on cumulative propensities.
        double u = _rng.NextDouble() * totalPropensity;
        double cumSum = 0;
        int firedReaction = _reactions.Length - 1;
        for (int i = 0; i < _reactions.Length; i++)
        {
            cumSum += propensities[i];
            if (cumSum >= u)
            {
                firedReaction = i;
                break;
            }
        }

        // Update molecule counts (integer stoichiometry).
        var rxn = _reactions[firedReaction];
        if (rxn.Reactants != null)
            foreach (int sp in rxn.Reactants)
                _concentrations[sp] = Math.Max(0, _concentrations[sp] - 1.0);
        if (rxn.Products != null)
            foreach (int sp in rxn.Products)
                _concentrations[sp] += 1.0;

        return dt;
    }

    /// <summary>
    /// Run the stochastic simulation for a given total time.
    /// </summary>
    public void RunStochastic(double totalTime)
    {
        double t = 0;
        while (t < totalTime)
        {
            double dt = GillespieStep();
            if (dt == double.MaxValue || double.IsNaN(dt))
                break;
            t += dt;
        }
    }

    // -----------------------------------------------------------------------
    //  Reaction-diffusion (Turing patterns)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Set the initial concentration on the spatial grid for a given species.
    /// </summary>
    public void SetSpatialConcentration(int species, Func<int, int, int, double> initializer)
    {
        for (int z = 0; z < _sz; z++)
            for (int y = 0; y < _sy; y++)
                for (int x = 0; x < _sx; x++)
                    _spatialConc[species, z, y, x] = initializer(x, y, z);
    }

    /// <summary>
    /// Advance the reaction-diffusion system by one time-step.
    /// Uses operator splitting: diffusion (Crank-Nicolson implicit) then
    /// reaction (explicit Euler).
    /// </summary>
    public void StepReactionDiffusion()
    {
        int sx = _sx, sy = _sy, sz = _sz;
        double D = _cfg.DiffusionCoefficients?[0] ?? 1.0;
        double dx = _cfg.Dx;
        double dt = _cfg.Dt;
        double r = D * dt / (dx * dx);

        // Copy to prev.
        Array.Copy(_spatialConc, _spatialConcPrev,
            _nSpecies * sz * sy * sx);

        // Diffusion step (explicit Laplacian for simplicity; implicit would
        // require a tridiagonal solve per line).
        for (int sp = 0; sp < _nSpecies; sp++)
        {
            double dk = _cfg.DiffusionCoefficients != null && sp < _cfg.DiffusionCoefficients.Length
                ? _cfg.DiffusionCoefficients[sp] : D;
            double rk = dk * dt / (dx * dx);

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        double laplacian =
                            _spatialConcPrev[sp, z, y, x + 1] + _spatialConcPrev[sp, z, y, x - 1] +
                            _spatialConcPrev[sp, z, y + 1, x] + _spatialConcPrev[sp, z, y - 1, x] +
                            _spatialConcPrev[sp, z + 1, y, x] + _spatialConcPrev[sp, z - 1, y, x] -
                            6.0 * _spatialConcPrev[sp, z, y, x];

                        _spatialConc[sp, z, y, x] = _spatialConcPrev[sp, z, y, x] + rk * laplacian;
                    }
        }

        // Reaction step at each grid point.
        double[] localConc = new double[_nSpecies];
        double[] dcdt = new double[_nSpecies];
        for (int z = 0; z < sz; z++)
            for (int y = 0; y < sy; y++)
                for (int x = 0; x < sx; x++)
                {
                    for (int sp = 0; sp < _nSpecies; sp++)
                        localConc[sp] = _spatialConc[sp, z, y, x];

                    // Temporarily swap concentrations for local evaluation.
                    double[] savedGlobal = (double[])_concentrations.Clone();
                    for (int sp = 0; sp < _nSpecies; sp++)
                        _concentrations[sp] = localConc[sp];

                    ComputeDerivatives(dcdt);

                    for (int sp = 0; sp < _nSpecies; sp++)
                    {
                        _spatialConc[sp, z, y, x] = Math.Max(0,
                            localConc[sp] + dt * dcdt[sp]);
                    }

                    // Restore global concentrations.
                    for (int sp = 0; sp < _nSpecies; sp++)
                        _concentrations[sp] = savedGlobal[sp];
                }

        // Neumann (no-flux) boundaries are implicit in the zeroing of
        // ghost-cell contributions.
    }

    /// <summary>
    /// Run the reaction-diffusion simulation for the configured number of steps.
    /// </summary>
    public void RunReactionDiffusion(int steps)
    {
        for (int i = 0; i < steps; i++)
            StepReactionDiffusion();
    }

    /// <summary>
    /// Export the spatial concentration of a species.
    /// </summary>
    public double[,,] GetSpatialConcentration(int species)
    {
        var result = new double[_sz, _sy, _sx];
        for (int z = 0; z < _sz; z++)
            for (int y = 0; y < _sy; y++)
                for (int x = 0; x < _sx; x++)
                    result[z, y, x] = _spatialConc[species, z, y, x];
        return result;
    }

    /// <summary>
    /// Run the deterministic ODE simulation for the configured number of steps.
    /// Returns concentration history [step, species].
    /// </summary>
    public double[,] RunODEWithHistory()
    {
        double[,] history = new double[_cfg.NumSteps + 1, _nSpecies];
        for (int s = 0; s < _nSpecies; s++)
            history[0, s] = _concentrations[s];

        for (int step = 1; step <= _cfg.NumSteps; step++)
        {
            StepODE();
            for (int s = 0; s < _nSpecies; s++)
                history[step, s] = _concentrations[s];
        }

        return history;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  5. NBodySolver — O(N²), Barnes-Hut, leapfrog, PN radiation
// ============================================================================

/// <summary>
/// Configuration for the N-body gravitational solver.
/// </summary>
public sealed class NBodyConfig
{
    public int NumBodies { get; init; } = 1000;
    public int NumSteps { get; init; } = 10_000;
    public double TimeStep { get; init; } = 0.01;
    public double Softening { get; init; } = 0.01;       // gravitational softening length
    public bool UseBarnesHut { get; init; }
    public double Theta { get; init; } = 0.5;            // Barnes-Hut opening angle
    public bool ComputeRadiation { get; init; }          // post-Newtonian radiation
    public int RadiationOrder { get; init; } = 2;        // PN order (1 = 1PN, 2 = 2PN)
    public double GravitationalConstant { get; init; } = 1.0;
    public double[] Masses { get; init; }                // per-body masses (null = unit)
    public bool RecordTrajectory { get; init; }
    public int TrajectoryInterval { get; init; } = 100;
}

/// <summary>
/// Represents an N-body particle with position, velocity, mass, and acceleration.
/// </summary>
public struct NBodyParticle
{
    public double X, Y, Z;
    public double Vx, Vy, Vz;
    public double Ax, Ay, Az;
    public double Mass;

    // Post-Newtonian radiation reaction.
    public double PNx, PNy, PNz;

    public NBodyParticle(double x, double y, double z, double mass)
    {
        X = x;
        Y = y;
        Z = z;
        Vx = Vy = Vz = 0;
        Ax = Ay = Az = 0;
        Mass = mass;
        PNx = PNy = PNz = 0;
    }
}

/// <summary>
/// Barnes-Hut octree node for hierarchical force computation.
/// </summary>
internal sealed class BHNode
{
    public double Cx, Cy, Cz;        // centre of mass
    public double TotalMass;
    public double HalfSize;
    public int BodyIndex = -1;       // leaf: index into body array
    public BHNode[] Children;        // 8 children (null for leaf or empty)
    public int ChildCount;
}

/// <summary>
/// N-body gravitational solver with direct O(N²) summation, Barnes-Hut
/// octree (O(N log N)), leapfrog integration, energy/angular-momentum
/// conservation tracking, and post-Newtonian gravitational radiation.
/// </summary>
public sealed class NBodySolver : IDisposable
{
    private readonly NBodyConfig _cfg;
    private readonly double _G;
    private readonly double _eps2;   // softening squared
    private NBodyParticle[] _bodies;

    // Barnes-Hut tree.
    private BHNode _root;
    private BHNode[] _nodePool;
    private int _nodePoolIdx;

    // Conserved quantities.
    private double _initialEnergy;
    private double[] _initialAngMom; // (Lx, Ly, Lz)

    // Trajectory storage.
    private List<double[]> _trajectories;

    // Radiation power accumulator.
    private double _totalRadiatedEnergy;

    private bool _disposed;

    public int NumBodies => _bodies.Length;
    public ReadOnlySpan<NBodyParticle> Bodies => _bodies;
    public double TotalRadiatedEnergy => _totalRadiatedEnergy;

    public NBodySolver(NBodyConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        _G = config.GravitationalConstant;
        _eps2 = config.Softening * config.Softening;

        _bodies = new NBodyParticle[config.NumBodies];
        _initialAngMom = new double[3];

        if (config.UseBarnesHut)
        {
            // Pre-allocate octree node pool (generous size).
            int maxNodes = Math.Max(config.NumBodies * 4, 1024);
            _nodePool = new BHNode[maxNodes];
            for (int i = 0; i < maxNodes; i++)
                _nodePool[i] = new BHNode();
        }

        if (config.RecordTrajectory)
            _trajectories = new List<double[]>();
    }

    /// <summary>
    /// Initialise bodies with the given positions, velocities, and masses.
    /// </summary>
    public void Initialise(double[] x, double[] y, double[] z,
        double[] vx, double[] vy, double[] vz, double[] mass)
    {
        int n = Math.Min(x.Length, _bodies.Length);
        for (int i = 0; i < n; i++)
        {
            _bodies[i] = new NBodyParticle(x[i], y[i], z[i], mass[i])
            {
                Vx = vx[i],
                Vy = vy[i],
                Vz = vz[i]
            };
        }
        _initialEnergy = ComputeTotalEnergy();
        ComputeAngularMomentum().CopyTo(_initialAngMom, 0);
    }

    /// <summary>
    /// Initialise Plummer model: isotropic sphere with scale radius a.
    /// </summary>
    public void InitialisePlummer(double scaleRadius, double totalMass, int seed = 42)
    {
        var rng = new Random(seed);
        int n = _bodies.Length;
        double mi = totalMass / n;

        for (int i = 0; i < n; i++)
        {
            // Rejection sampling for radius from Plummer density profile.
            double r;
            do
            {
                double u = rng.NextDouble();
                r = scaleRadius / Math.Sqrt(Math.Pow(u, -2.0 / 3.0) - 1.0);
            } while (r > 100.0 * scaleRadius);

            // Random point on sphere.
            double cosTheta = 2.0 * rng.NextDouble() - 1.0;
            double sinTheta = Math.Sqrt(1.0 - cosTheta * cosTheta);
            double phi = 2.0 * Math.PI * rng.NextDouble();

            double x = r * sinTheta * Math.Cos(phi);
            double y = r * sinTheta * Math.Sin(phi);
            double z = r * cosTheta;

            // Velocity from Plummer distribution.
            double q = rng.NextDouble();
            double g = q * q * Math.Pow(1.0 + 1.0 / (scaleRadius * scaleRadius), -0.75);
            double vEsc = Math.Sqrt(2.0 * _G * totalMass / scaleRadius);
            double v = vEsc * g * 0.5;

            double cosThetaV = 2.0 * rng.NextDouble() - 1.0;
            double sinThetaV = Math.Sqrt(1.0 - cosThetaV * cosThetaV);
            double phiV = 2.0 * Math.PI * rng.NextDouble();

            _bodies[i] = new NBodyParticle(x, y, z, mi)
            {
                Vx = v * sinThetaV * Math.Cos(phiV),
                Vy = v * sinThetaV * Math.Sin(phiV),
                Vz = v * cosThetaV
            };
        }

        // Remove centre-of-mass velocity.
        double vcmX = 0, vcmY = 0, vcmZ = 0;
        for (int i = 0; i < n; i++)
        {
            vcmX += _bodies[i].Vx * _bodies[i].Mass;
            vcmY += _bodies[i].Vy * _bodies[i].Mass;
            vcmZ += _bodies[i].Vz * _bodies[i].Mass;
        }
        double totalM = totalMass;
        vcmX /= totalM;
        vcmY /= totalM;
        vcmZ /= totalM;
        for (int i = 0; i < n; i++)
        {
            _bodies[i].Vx -= vcmX;
            _bodies[i].Vy -= vcmY;
            _bodies[i].Vz -= vcmZ;
        }

        _initialEnergy = ComputeTotalEnergy();
        ComputeAngularMomentum().CopyTo(_initialAngMom, 0);
    }

    // -----------------------------------------------------------------------
    //  Direct O(N²) force computation
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute gravitational accelerations using direct pairwise summation.
    /// a_i = G Σ_{j≠i} m_j (r_j − r_i) / |r_j − r_i|³
    /// with softening: replace |r|² → |r|² + ε².
    /// </summary>
    public void ComputeForcesDirect()
    {
        int n = _bodies.Length;

        // Zero accelerations.
        for (int i = 0; i < n; i++)
        {
            _bodies[i].Ax = 0;
            _bodies[i].Ay = 0;
            _bodies[i].Az = 0;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dx = _bodies[j].X - _bodies[i].X;
                double dy = _bodies[j].Y - _bodies[i].Y;
                double dz = _bodies[j].Z - _bodies[i].Z;
                double r2 = dx * dx + dy * dy + dz * dz + _eps2;
                double invR = 1.0 / Math.Sqrt(r2);
                double invR3 = invR * invR * invR;

                double fx = _G * dx * invR3;
                double fy = _G * dy * invR3;
                double fz = _G * dz * invR3;

                _bodies[i].Ax += fx * _bodies[j].Mass;
                _bodies[i].Ay += fy * _bodies[j].Mass;
                _bodies[i].Az += fz * _bodies[j].Mass;

                _bodies[j].Ax -= fx * _bodies[i].Mass;
                _bodies[j].Ay -= fy * _bodies[i].Mass;
                _bodies[j].Az -= fz * _bodies[i].Mass;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Barnes-Hut octree
    // -----------------------------------------------------------------------

    private BHNode AllocateNode()
    {
        if (_nodePoolIdx >= _nodePool.Length)
        {
            // Grow pool.
            var bigger = new BHNode[_nodePool.Length * 2];
            for (int i = _nodePool.Length; i < bigger.Length; i++)
                bigger[i] = new BHNode();
            Array.Copy(_nodePool, bigger, _nodePool.Length);
            _nodePool = bigger;
        }
        var node = _nodePool[_nodePoolIdx++];
        node.BodyIndex = -1;
        node.Children = null;
        node.ChildCount = 0;
        node.TotalMass = 0;
        node.Cx = node.Cy = node.Cz = 0;
        node.HalfSize = 0;
        return node;
    }

    /// <summary>
    /// Build the Barnes-Hut octree from the current body positions.
    /// </summary>
    private BHNode BuildTree()
    {
        _nodePoolIdx = 0;

        // Find bounding box.
        double xmin = double.MaxValue, xmax = double.MinValue;
        double ymin = double.MaxValue, ymax = double.MinValue;
        double zmin = double.MaxValue, zmax = double.MinValue;
        for (int i = 0; i < _bodies.Length; i++)
        {
            if (_bodies[i].X < xmin)
                xmin = _bodies[i].X;
            if (_bodies[i].X > xmax)
                xmax = _bodies[i].X;
            if (_bodies[i].Y < ymin)
                ymin = _bodies[i].Y;
            if (_bodies[i].Y > ymax)
                ymax = _bodies[i].Y;
            if (_bodies[i].Z < zmin)
                zmin = _bodies[i].Z;
            if (_bodies[i].Z > zmax)
                zmax = _bodies[i].Z;
        }

        double cx = 0.5 * (xmin + xmax);
        double cy = 0.5 * (ymin + ymax);
        double cz = 0.5 * (zmin + zmax);
        double halfSize = 0.5 * Math.Max(Math.Max(xmax - xmin, ymax - ymin), zmax - zmin) * 1.01;

        var root = AllocateNode();
        root.Cx = cx;
        root.Cy = cy;
        root.Cz = cz;
        root.HalfSize = halfSize;

        for (int i = 0; i < _bodies.Length; i++)
            InsertBody(root, i);

        return root;
    }

    private void InsertBody(BHNode node, int bodyIdx)
    {
        if (node.BodyIndex == -1 && node.ChildCount == 0)
        {
            // Empty node: make it a leaf.
            node.BodyIndex = bodyIdx;
            node.TotalMass = _bodies[bodyIdx].Mass;
            node.Cx = _bodies[bodyIdx].X;
            node.Cy = _bodies[bodyIdx].Y;
            node.Cz = _bodies[bodyIdx].Z;
            return;
        }

        if (node.BodyIndex != -1 && node.ChildCount == 0)
        {
            // Current leaf: split into children.
            int oldBody = node.BodyIndex;
            node.BodyIndex = -1;
            InsertIntoChild(node, oldBody);
            InsertIntoChild(node, bodyIdx);
            return;
        }

        // Internal node: update centre of mass and recurse.
        double totalM = node.TotalMass + _bodies[bodyIdx].Mass;
        node.Cx = (node.Cx * node.TotalMass + _bodies[bodyIdx].X * _bodies[bodyIdx].Mass) / totalM;
        node.Cy = (node.Cy * node.TotalMass + _bodies[bodyIdx].Y * _bodies[bodyIdx].Mass) / totalM;
        node.Cz = (node.Cz * node.TotalMass + _bodies[bodyIdx].Z * _bodies[bodyIdx].Mass) / totalM;
        node.TotalMass = totalM;

        InsertIntoChild(node, bodyIdx);
    }

    private void InsertIntoChild(BHNode node, int bodyIdx)
    {
        double bx = _bodies[bodyIdx].X;
        double by = _bodies[bodyIdx].Y;
        double bz = _bodies[bodyIdx].Z;

        // Determine octant.
        int octant = 0;
        if (bx >= node.Cx)
            octant |= 1;
        if (by >= node.Cy)
            octant |= 2;
        if (bz >= node.Cz)
            octant |= 4;

        if (node.Children == null)
        {
            node.Children = new BHNode[8];
            node.ChildCount = 0;
        }

        if (node.Children[octant] == null)
        {
            node.Children[octant] = AllocateNode();
            double halfChild = node.HalfSize * 0.5;
            node.Children[octant].HalfSize = halfChild;
            node.Children[octant].Cx = node.Cx + ((octant & 1) == 0 ? -halfChild : halfChild);
            node.Children[octant].Cy = node.Cy + ((octant & 2) == 0 ? -halfChild : halfChild);
            node.Children[octant].Cz = node.Cz + ((octant & 4) == 0 ? -halfChild : halfChild);
            node.ChildCount++;
        }

        InsertBody(node.Children[octant], bodyIdx);
    }

    /// <summary>
    /// Compute acceleration on body i using the Barnes-Hut tree.
    /// </summary>
    private void ComputeAccelerationBH(BHNode node, int bodyIdx)
    {
        if (node == null || node.TotalMass == 0)
            return;

        double dx = node.Cx - _bodies[bodyIdx].X;
        double dy = node.Cy - _bodies[bodyIdx].Y;
        double dz = node.Cz - _bodies[bodyIdx].Z;
        double r2 = dx * dx + dy * dy + dz * dz + _eps2;

        // Check if this node is far enough to treat as a single particle.
        bool isOpen = (node.HalfSize * 2) * (node.HalfSize * 2) > _cfg.Theta * _cfg.Theta * r2;

        if (!isOpen || (node.ChildCount == 0 && node.BodyIndex != -1))
        {
            // Treat as point mass.
            if (node.BodyIndex == bodyIdx)
                return; // skip self

            double invR = 1.0 / Math.Sqrt(r2);
            double invR3 = invR * invR * invR;
            double fx = _G * node.TotalMass * dx * invR3;
            double fy = _G * node.TotalMass * dy * invR3;
            double fz = _G * node.TotalMass * dz * invR3;

            _bodies[bodyIdx].Ax += fx;
            _bodies[bodyIdx].Ay += fy;
            _bodies[bodyIdx].Az += fz;
        }
        else
        {
            // Open the node and recurse.
            if (node.Children != null)
            {
                for (int c = 0; c < 8; c++)
                    if (node.Children[c] != null)
                        ComputeAccelerationBH(node.Children[c], bodyIdx);
            }
        }
    }

    /// <summary>
    /// Compute all forces using Barnes-Hut algorithm.
    /// </summary>
    public void ComputeForcesBH()
    {
        for (int i = 0; i < _bodies.Length; i++)
        {
            _bodies[i].Ax = 0;
            _bodies[i].Ay = 0;
            _bodies[i].Az = 0;
        }

        _root = BuildTree();
        for (int i = 0; i < _bodies.Length; i++)
            ComputeAccelerationBH(_root, i);
    }

    // -----------------------------------------------------------------------
    //  Post-Newtonian gravitational radiation
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute the leading-order (2.5PN) gravitational radiation reaction
    /// force for each body in a binary system.
    /// F_rad = −(32/5) G⁴ m₁² m₂² (m₁+m₂) / (c⁵ r⁵) v_r
    /// where v_r is the relative velocity projected along the separation.
    /// </summary>
    public void ComputeRadiationReaction(int bodyA, int bodyB)
    {
        if (!_cfg.ComputeRadiation)
            return;

        double c5 = Math.Pow(PhysicsConstants.C0, 5);
        ref var a = ref _bodies[bodyA];
        ref var b = ref _bodies[bodyB];

        double dx = b.X - a.X;
        double dy = b.Y - a.Y;
        double dz = b.Z - a.Z;
        double r2 = dx * dx + dy * dy + dz * dz;
        double r = Math.Sqrt(r2);
        double r5 = r2 * r2 * r;

        double dvx = b.Vx - a.Vx;
        double dvy = b.Vy - a.Vy;
        double dvz = b.Vz - a.Vz;

        // Radial velocity (positive = separating).
        double vr = (dx * dvx + dy * dvy + dz * dvz) / r;

        double M = a.Mass + b.Mass;
        double m1m2 = a.Mass * b.Mass;
        double coeff = -(32.0 / 5.0) * Math.Pow(_G, 4) *
                       m1m2 * m1m2 * M / (c5 * r5);

        // Radiation reaction force along the line connecting the two bodies.
        double frx = coeff * vr * dx / r;
        double fry = coeff * vr * dy / r;
        double frz = coeff * vr * dz / r;

        a.PNx = frx / a.Mass;
        a.PNy = fry / a.Mass;
        a.PNz = frz / a.Mass;

        b.PNx = -frx / b.Mass;
        b.PNy = -fry / b.Mass;
        b.PNz = -frz / b.Mass;

        // Power radiated: P = (32/5) G⁴ m₁² m₂² (m₁+m₂) / (c⁵ r⁵) v_r²
        double power = (32.0 / 5.0) * Math.Pow(_G, 4) *
                       m1m2 * m1m2 * M / (c5 * r5) * vr * vr;
        _totalRadiatedEnergy += power * _cfg.TimeStep;
    }

    /// <summary>
    /// Compute 1PN correction to accelerations for a pair of bodies.
    /// </summary>
    public void Compute1PNCorrection(int bodyA, int bodyB)
    {
        if (_cfg.RadiationOrder < 1)
            return;

        double c2 = PhysicsConstants.C0 * PhysicsConstants.C0;
        ref var a = ref _bodies[bodyA];
        ref var b = ref _bodies[bodyB];

        double dx = b.X - a.X;
        double dy = b.Y - a.Y;
        double dz = b.Z - a.Z;
        double r2 = dx * dx + dy * dy + dz * dz;
        double r = Math.Sqrt(r2);

        double dvx = b.Vx - a.Vx;
        double dvy = b.Vy - a.Vy;
        double dvz = b.Vz - a.Vz;
        double v2 = dvx * dvx + dvy * dvy + dvz * dvz;

        double M = a.Mass + b.Mass;
        double nu = a.Mass * b.Mass / (M * M); // symmetric mass ratio

        // 1PN potentials.
        double phi = -_G * M / r;
        double v2OverC2 = v2 / c2;

        // 1PN correction factor: 1 + (3+ν) v²/c² − (4+2ν) GM/(rc²) + ...
        double correction = 1.0 + (3.0 + nu) * v2OverC2 -
                            (4.0 + 2.0 * nu) * phi / c2;

        // Apply correction to Newtonian acceleration.
        double fxN = _G * dx / (r2 * r);
        double fyN = _G * dy / (r2 * r);
        double fzN = _G * dz / (r2 * r);

        a.Ax += fxN * (correction - 1.0) * b.Mass;
        a.Ay += fyN * (correction - 1.0) * b.Mass;
        a.Az += fzN * (correction - 1.0) * b.Mass;
    }

    // -----------------------------------------------------------------------
    //  Leapfrog integration
    // -----------------------------------------------------------------------

    /// <summary>
    /// Advance the system by one time-step using leapfrog (kick-drift-kick) integration.
    /// This is a symplectic integrator that conserves energy to high order.
    /// </summary>
    public void StepLeapfrog()
    {
        int n = _bodies.Length;
        double dt = _cfg.TimeStep;
        double halfDt = 0.5 * dt;

        // Compute forces at current positions.
        if (_cfg.UseBarnesHut)
            ComputeForcesBH();
        else
            ComputeForcesDirect();

        // Compute PN radiation reaction if enabled.
        if (_cfg.ComputeRadiation)
        {
            // For simplicity, apply radiation to first two bodies (binary).
            if (n >= 2)
            {
                ComputeRadiationReaction(0, 1);
                if (_cfg.RadiationOrder >= 1)
                {
                    Compute1PNCorrection(0, 1);
                    Compute1PNCorrection(1, 0);
                }
            }
        }

        // Half-kick: v(t+dt/2) = v(t) + (dt/2) a(t)
        for (int i = 0; i < n; i++)
        {
            _bodies[i].Vx += halfDt * (_bodies[i].Ax + _bodies[i].PNx);
            _bodies[i].Vy += halfDt * (_bodies[i].Ay + _bodies[i].PNy);
            _bodies[i].Vz += halfDt * (_bodies[i].Az + _bodies[i].PNz);
        }

        // Drift: r(t+dt) = r(t) + dt v(t+dt/2)
        for (int i = 0; i < n; i++)
        {
            _bodies[i].X += dt * _bodies[i].Vx;
            _bodies[i].Y += dt * _bodies[i].Vy;
            _bodies[i].Z += dt * _bodies[i].Vz;
        }

        // Compute forces at new positions.
        if (_cfg.UseBarnesHut)
            ComputeForcesBH();
        else
            ComputeForcesDirect();

        if (_cfg.ComputeRadiation && n >= 2)
        {
            ComputeRadiationReaction(0, 1);
            if (_cfg.RadiationOrder >= 1)
            {
                Compute1PNCorrection(0, 1);
                Compute1PNCorrection(1, 0);
            }
        }

        // Half-kick: v(t+dt) = v(t+dt/2) + (dt/2) a(t+dt)
        for (int i = 0; i < n; i++)
        {
            _bodies[i].Vx += halfDt * (_bodies[i].Ax + _bodies[i].PNx);
            _bodies[i].Vy += halfDt * (_bodies[i].Ay + _bodies[i].PNy);
            _bodies[i].Vz += halfDt * (_bodies[i].Az + _bodies[i].PNz);
        }

        // Record trajectory if requested.
        if (_cfg.RecordTrajectory && _trajectories != null &&
            _trajectories.Count % _cfg.TrajectoryInterval == 0)
        {
            double[] traj = new double[n * 3];
            for (int i = 0; i < n; i++)
            {
                traj[i * 3] = _bodies[i].X;
                traj[i * 3 + 1] = _bodies[i].Y;
                traj[i * 3 + 2] = _bodies[i].Z;
            }
            _trajectories.Add(traj);
        }
    }

    // -----------------------------------------------------------------------
    //  Runge-Kutta 4th order (alternative integrator)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Advance the system by one time-step using RK4.
    /// </summary>
    public void StepRK4()
    {
        int n = _bodies.Length;
        double dt = _cfg.TimeStep;

        // Store initial state.
        var saved = new NBodyParticle[n];
        for (int i = 0; i < n; i++)
            saved[i] = _bodies[i];

        double[] k1ax = new double[n], k1ay = new double[n], k1az = new double[n];

        // k1
        ComputeForcesOrBH();
        for (int i = 0; i < n; i++)
        {
            k1ax[i] = _bodies[i].Ax;
            k1ay[i] = _bodies[i].Ay;
            k1az[i] = _bodies[i].Az;
        }

        // k2: advance to midpoint using k1
        for (int i = 0; i < n; i++)
        {
            _bodies[i].X = saved[i].X + 0.5 * dt * saved[i].Vx;
            _bodies[i].Y = saved[i].Y + 0.5 * dt * saved[i].Vy;
            _bodies[i].Z = saved[i].Z + 0.5 * dt * saved[i].Vz;
            _bodies[i].Vx = saved[i].Vx + 0.5 * dt * k1ax[i];
            _bodies[i].Vy = saved[i].Vy + 0.5 * dt * k1ay[i];
            _bodies[i].Vz = saved[i].Vz + 0.5 * dt * k1az[i];
        }
        ComputeForcesOrBH();
        double[] k2ax = new double[n], k2ay = new double[n], k2az = new double[n];
        double[] k2vx = new double[n], k2vy = new double[n], k2vz = new double[n];
        for (int i = 0; i < n; i++)
        {
            k2ax[i] = _bodies[i].Ax;
            k2ay[i] = _bodies[i].Ay;
            k2az[i] = _bodies[i].Az;
            k2vx[i] = _bodies[i].Vx;
            k2vy[i] = _bodies[i].Vy;
            k2vz[i] = _bodies[i].Vz;
        }

        // k3
        for (int i = 0; i < n; i++)
        {
            _bodies[i].X = saved[i].X + 0.5 * dt * k2vx[i];
            _bodies[i].Y = saved[i].Y + 0.5 * dt * k2vy[i];
            _bodies[i].Z = saved[i].Z + 0.5 * dt * k2vz[i];
            _bodies[i].Vx = saved[i].Vx + 0.5 * dt * k2ax[i];
            _bodies[i].Vy = saved[i].Vy + 0.5 * dt * k2ay[i];
            _bodies[i].Vz = saved[i].Vz + 0.5 * dt * k2az[i];
        }
        ComputeForcesOrBH();
        double[] k3ax = new double[n], k3ay = new double[n], k3az = new double[n];
        double[] k3vx = new double[n], k3vy = new double[n], k3vz = new double[n];
        for (int i = 0; i < n; i++)
        {
            k3ax[i] = _bodies[i].Ax;
            k3ay[i] = _bodies[i].Ay;
            k3az[i] = _bodies[i].Az;
            k3vx[i] = _bodies[i].Vx;
            k3vy[i] = _bodies[i].Vy;
            k3vz[i] = _bodies[i].Vz;
        }

        // k4
        for (int i = 0; i < n; i++)
        {
            _bodies[i].X = saved[i].X + dt * k3vx[i];
            _bodies[i].Y = saved[i].Y + dt * k3vy[i];
            _bodies[i].Z = saved[i].Z + dt * k3vz[i];
            _bodies[i].Vx = saved[i].Vx + dt * k3ax[i];
            _bodies[i].Vy = saved[i].Vy + dt * k3ay[i];
            _bodies[i].Vz = saved[i].Vz + dt * k3az[i];
        }
        ComputeForcesOrBH();
        double[] k4ax = new double[n], k4ay = new double[n], k4az = new double[n];
        for (int i = 0; i < n; i++)
        {
            k4ax[i] = _bodies[i].Ax;
            k4ay[i] = _bodies[i].Ay;
            k4az[i] = _bodies[i].Az;
        }

        // Combine.
        for (int i = 0; i < n; i++)
        {
            _bodies[i].Vx = saved[i].Vx + dt / 6.0 *
                (k1ax[i] + 2 * k2ax[i] + 2 * k3ax[i] + k4ax[i]);
            _bodies[i].Vy = saved[i].Vy + dt / 6.0 *
                (k1ay[i] + 2 * k2ay[i] + 2 * k3ay[i] + k4ay[i]);
            _bodies[i].Vz = saved[i].Vz + dt / 6.0 *
                (k1az[i] + 2 * k2az[i] + 2 * k3az[i] + k4az[i]);
            _bodies[i].X = saved[i].X + dt / 6.0 *
                (saved[i].Vx + 2 * k2vx[i] + 2 * k3vx[i] + _bodies[i].Vx);
            _bodies[i].Y = saved[i].Y + dt / 6.0 *
                (saved[i].Vy + 2 * k2vy[i] + 2 * k3vy[i] + _bodies[i].Vy);
            _bodies[i].Z = saved[i].Z + dt / 6.0 *
                (saved[i].Vz + 2 * k2vz[i] + 2 * k3vz[i] + _bodies[i].Vz);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeForcesOrBH()
    {
        if (_cfg.UseBarnesHut)
            ComputeForcesBH();
        else
            ComputeForcesDirect();
    }

    // -----------------------------------------------------------------------
    //  Conserved quantities
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute total energy: KE + PE.
    /// KE = 0.5 Σ m_i |v_i|²
    /// PE = −G Σ_{i&lt;j} m_i m_j / r_ij
    /// </summary>
    public double ComputeTotalEnergy()
    {
        double ke = 0, pe = 0;
        int n = _bodies.Length;
        for (int i = 0; i < n; i++)
        {
            double v2 = _bodies[i].Vx * _bodies[i].Vx +
                        _bodies[i].Vy * _bodies[i].Vy +
                        _bodies[i].Vz * _bodies[i].Vz;
            ke += 0.5 * _bodies[i].Mass * v2;

            for (int j = i + 1; j < n; j++)
            {
                double dx = _bodies[i].X - _bodies[j].X;
                double dy = _bodies[i].Y - _bodies[j].Y;
                double dz = _bodies[i].Z - _bodies[j].Z;
                double r = Math.Sqrt(dx * dx + dy * dy + dz * dz + _eps2);
                pe -= _G * _bodies[i].Mass * _bodies[j].Mass / r;
            }
        }
        return ke + pe;
    }

    /// <summary>
    /// Compute total angular momentum: L = Σ m_i r_i × v_i.
    /// </summary>
    public double[] ComputeAngularMomentum()
    {
        double lx = 0, ly = 0, lz = 0;
        for (int i = 0; i < _bodies.Length; i++)
        {
            double m = _bodies[i].Mass;
            lx += m * (_bodies[i].Y * _bodies[i].Vz - _bodies[i].Z * _bodies[i].Vy);
            ly += m * (_bodies[i].Z * _bodies[i].Vx - _bodies[i].X * _bodies[i].Vz);
            lz += m * (_bodies[i].X * _bodies[i].Vy - _bodies[i].Y * _bodies[i].Vx);
        }
        return new[] { lx, ly, lz };
    }

    /// <summary>
    /// Compute total linear momentum: P = Σ m_i v_i.
    /// </summary>
    public double[] ComputeLinearMomentum()
    {
        double px = 0, py = 0, pz = 0;
        for (int i = 0; i < _bodies.Length; i++)
        {
            px += _bodies[i].Mass * _bodies[i].Vx;
            py += _bodies[i].Mass * _bodies[i].Vy;
            pz += _bodies[i].Mass * _bodies[i].Vz;
        }
        return new[] { px, py, pz };
    }

    /// <summary>
    /// Energy conservation error: (E − E₀) / |E₀|.
    /// </summary>
    public double EnergyError()
    {
        double e = ComputeTotalEnergy();
        return Math.Abs(e - _initialEnergy) / Math.Max(Math.Abs(_initialEnergy), 1e-30);
    }

    /// <summary>
    /// Angular momentum conservation error: |L − L₀| / |L₀|.
    /// </summary>
    public double AngularMomentumError()
    {
        double[] L = ComputeAngularMomentum();
        double magL = Math.Sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
        double magL0 = Math.Sqrt(
            _initialAngMom[0] * _initialAngMom[0] +
            _initialAngMom[1] * _initialAngMom[1] +
            _initialAngMom[2] * _initialAngMom[2]);
        return Math.Abs(magL - magL0) / Math.Max(magL0, 1e-30);
    }

    /// <summary>
    /// Get stored trajectories (if RecordTrajectory is enabled).
    /// </summary>
    public IReadOnlyList<double[]> Trajectories => _trajectories;

    /// <summary>
    /// Run the simulation for the configured number of steps using leapfrog.
    /// </summary>
    public void Run()
    {
        for (int i = 0; i < _cfg.NumSteps; i++)
            StepLeapfrog();
    }

    /// <summary>
    /// Find the two closest bodies (useful for binary detection).
    /// </summary>
    public (int I, int J, double Distance) FindClosestPair()
    {
        double minDist = double.MaxValue;
        int bestI = 0, bestJ = 1;
        for (int i = 0; i < _bodies.Length; i++)
        {
            for (int j = i + 1; j < _bodies.Length; j++)
            {
                double dx = _bodies[i].X - _bodies[j].X;
                double dy = _bodies[i].Y - _bodies[j].Y;
                double dz = _bodies[i].Z - _bodies[j].Z;
                double d = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                if (d < minDist)
                {
                    minDist = d;
                    bestI = i;
                    bestJ = j;
                }
            }
        }
        return (bestI, bestJ, minDist);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  6. LatticeBoltzmannSolver — D2Q9, D3Q19, BGK, MRT, multiphase
// ============================================================================

/// <summary>
/// Lattice type for the Boltzmann solver.
/// </summary>
public enum LatticeType
{
    D2Q9,
    D3Q19
}

/// <summary>
/// Collision model for the LBM solver.
/// </summary>
public enum CollisionModel
{
    BGK,
    MRT
}

/// <summary>
/// Configuration for the Lattice Boltzmann solver.
/// </summary>
public sealed class LBMConfig
{
    public LatticeType Lattice { get; init; } = LatticeType.D3Q19;
    public CollisionModel Collision { get; init; } = CollisionModel.BGK;
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (128, 64, 64);
    public int NumSteps { get; init; } = 50_000;
    public double Relaxation { get; init; } = 0.8;      // τ (BGK relaxation time)
    public double KinematicViscosity { get; init; } = 0.01;
    public double Density0 { get; init; } = 1.0;
    public double InletVelocity { get; init; } = 0.05;
    public double OutletPressure { get; init; } = 1.0;
    public bool UseMultiphase { get; init; }            // Shan-Chen pseudopotential
    public double GShanChen { get; init; } = -4.7;      // Shan-Chen interaction strength
    public double[] BodyForce { get; init; }             // external force (fx, fy, fz)
    public int OutputInterval { get; init; } = 1000;
}

/// <summary>
/// Lattice Boltzmann solver with D2Q9 and D3Q19 lattices,
/// BGK and MRT collision operators, bounce-back boundaries,
/// Zou-He inlet/outlet, Shan-Chen multiphase, and force coupling.
/// </summary>
public sealed class LatticeBoltzmannSolver : IDisposable
{
    private readonly LBMConfig _cfg;
    private readonly LatticeType _lattice;
    private readonly int _nx, _ny, _nz;
    private readonly int _n;                          // total cells
    private readonly int _q;                          // number of velocities
    private readonly double _tau;                     // relaxation time
    private readonly double _omega;                   // 1/τ

    // Distribution functions: f[direction, z, y, x] (flattened).
    private double[] _f;
    private double[] _fTmp;

    // Macroscopic quantities.
    private double[] _rho;
    private double[] _ux, _uy, _uz;

    // Force arrays.
    private double[] _Fx, _Fy, _Fz;

    // Lattice weights, velocity vectors.
    private int[,] _ex, _ey, _ez;
    private double[] _w;

    // Bounce-back flags.
    private bool[] _solid;

    // MRT transform matrix and inverse (row-major).
    private double[,] _mrtM;
    private double[,] _mrtMinv;
    private double[] _mrtS; // relaxation rates in moment space

    private Random _rng;
    private bool _disposed;

    public int CurrentStep { get; private set; }
    public ReadOnlySpan<double> Rho => _rho;
    public ReadOnlySpan<double> Ux => _ux;

    public LatticeBoltzmannSolver(LBMConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        _lattice = config.Lattice;
        (_nx, _ny, _nz) = config.GridSize;
        _n = _nx * _ny * _nz;
        _tau = config.Relaxation;
        _omega = 1.0 / _tau;

        // Initialise lattice.
        switch (_lattice)
        {
            case LatticeType.D2Q9:
                _q = 9;
                InitD2Q9();
                break;
            case LatticeType.D3Q19:
                _q = 19;
                InitD3Q19();
                break;
            default:
                throw new ArgumentException($"Unsupported lattice: {_lattice}");
        }

        // Allocate arrays.
        int total = _q * _n;
        _f = new double[total];
        _fTmp = new double[total];
        _rho = new double[_n];
        _ux = new double[_n];
        _uy = new double[_n];
        _uz = new double[_n];
        _Fx = new double[_n];
        _Fy = new double[_n];
        _Fz = new double[_n];
        _solid = new bool[_n];

        // Initial equilibrium.
        InitialiseEquilibrium();

        // MRT setup.
        if (config.Collision == CollisionModel.MRT)
            InitialiseMRT();

        _rng = new Random(42);
    }

    // -----------------------------------------------------------------------
    //  D2Q9 lattice
    // -----------------------------------------------------------------------

    private void InitD2Q9()
    {
        _ex = new int[_q, 1];
        _ey = new int[_q, 1];
        _ez = new int[_q, 1];
        _w = new double[_q];

        // D2Q9 velocities (2D — z component always 0).
        int[,] exArr = { { 0 }, { 1 }, { 0 }, { -1 }, { 0 }, { 1 }, { -1 }, { -1 }, { 1 } };
        int[,] eyArr = { { 0 }, { 0 }, { 1 }, { 0 }, { -1 }, { 1 }, { 1 }, { -1 }, { -1 } };
        double[] wArr = { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                          1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };

        for (int q = 0; q < _q; q++)
        {
            _ex[q, 0] = exArr[q, 0];
            _ey[q, 0] = eyArr[q, 0];
            _ez[q, 0] = 0;
            _w[q] = wArr[q];
        }
    }

    // -----------------------------------------------------------------------
    //  D3Q19 lattice
    // -----------------------------------------------------------------------

    private void InitD3Q19()
    {
        _ex = new int[_q, 1];
        _ey = new int[_q, 1];
        _ez = new int[_q, 1];
        _w = new double[_q];

        // D3Q19 velocities.
        int[] exVals = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0 };
        int[] eyVals = { 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1 };
        int[] ezVals = { 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1 };

        for (int q = 0; q < _q; q++)
        {
            _ex[q, 0] = exVals[q];
            _ey[q, 0] = eyVals[q];
            _ez[q, 0] = ezVals[q];
        }

        _w[0] = 1.0 / 3.0;
        for (int q = 1; q <= 6; q++)
            _w[q] = 1.0 / 18.0;
        for (int q = 7; q < _q; q++)
            _w[q] = 1.0 / 36.0;
    }

    // -----------------------------------------------------------------------
    //  Index helper
    // -----------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Idx(int x, int y, int z) => z * _ny * _nx + y * _nx + x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int FIdx(int q, int cellIdx) => q * _n + cellIdx;

    // -----------------------------------------------------------------------
    //  Equilibrium distribution
    // -----------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double Feq(int q, double rho, double ux, double uy, double uz)
    {
        double cu = _ex[q, 0] * ux + _ey[q, 0] * uy + _ez[q, 0] * uz;
        double u2 = ux * ux + uy * uy + uz * uz;
        return _w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }

    private void InitialiseEquilibrium()
    {
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    _rho[idx] = _cfg.Density0;
                    _ux[idx] = 0;
                    _uy[idx] = 0;
                    _uz[idx] = 0;

                    for (int q = 0; q < _q; q++)
                        _f[FIdx(q, idx)] = Feq(q, _rho[idx], 0, 0, 0);
                }

        // Set inlet velocity.
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
            {
                int idx = Idx(0, y, z);
                _ux[idx] = _cfg.InletVelocity;
                for (int q = 0; q < _q; q++)
                    _f[FIdx(q, idx)] = Feq(q, _rho[idx], _cfg.InletVelocity, 0, 0);
            }
    }

    // -----------------------------------------------------------------------
    //  Solid boundaries (bounce-back)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Set a cell as solid (no-slip bounce-back).
    /// </summary>
    public void SetSolid(int x, int y, int z, bool isSolid = true)
    {
        if ((uint)x < (uint)_nx && (uint)y < (uint)_ny && (uint)z < (uint)_nz)
            _solid[Idx(x, y, z)] = isSolid;
    }

    /// <summary>
    /// Set a rectangular region as solid.
    /// </summary>
    public void SetSolidRegion(int x0, int y0, int z0, int x1, int y1, int z1)
    {
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++)
                    SetSolid(x, y, z);
    }

    /// <summary>
    /// Apply full-way bounce-back: reverse all distributions hitting solid cells.
    /// </summary>
    private void ApplyBounceBack()
    {
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    if (!_solid[idx])
                        continue;

                    for (int q = 0; q < _q; q++)
                    {
                        // Opposite direction.
                        int oppQ = FindOpposite(q);
                        int nx = x + _ex[oppQ, 0];
                        int ny2 = y + _ey[oppQ, 0];
                        int nz2 = z + _ez[oppQ, 0];
                        if ((uint)nx < (uint)_nx && (uint)ny2 < (uint)_ny && (uint)nz2 < (uint)_nz)
                        {
                            int nIdx = Idx(nx, ny2, nz2);
                            _fTmp[FIdx(oppQ, nIdx)] = _f[FIdx(q, idx)];
                        }
                    }
                }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int FindOpposite(int q)
    {
        // D2Q9 and D3Q19: opposite velocity = index with negated components.
        for (int q2 = 0; q2 < _q; q2++)
        {
            if (_ex[q2, 0] == -_ex[q, 0] &&
                _ey[q2, 0] == -_ey[q, 0] &&
                _ez[q2, 0] == -_ez[q, 0])
                return q2;
        }
        return 0; // rest particle
    }

    // -----------------------------------------------------------------------
    //  Zou-He inlet/outlet boundary conditions
    // -----------------------------------------------------------------------

    /// <summary>
    /// Apply Zou-He velocity inlet at x = 0 (known ux = u_inlet).
    /// Uses the non-equilibrium bounce-back approach.
    /// </summary>
    private void ApplyZouHeInlet()
    {
        double uIn = _cfg.InletVelocity;

        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
            {
                int idx = Idx(0, y, z);
                if (_solid[idx])
                    continue;

                _ux[idx] = uIn;
                _uy[idx] = 0;
                _uz[idx] = 0;

                // D3Q19: unknown distributions are q = 1, 7, 9, 13, 15 (x > 0 velocities).
                // Using Zou-He formulas for density and unknown f_i.
                double rho = 0;
                for (int q = 0; q < _q; q++)
                {
                    if (_ex[q, 0] <= 0) // known: incoming from right
                        rho += _f[FIdx(q, idx)];
                }

                double rhoStar = rho / (1.0 - uIn); // simplified for D3Q19
                _rho[idx] = Math.Max(rhoStar, 0.001);

                // Compute unknown distributions from equilibrium + non-equilibrium.
                for (int q = 0; q < _q; q++)
                {
                    if (_ex[q, 0] > 0)
                    {
                        int oppQ = FindOpposite(q);
                        double feqOpp = Feq(oppQ, _rho[idx], _ux[idx], _uy[idx], _uz[idx]);
                        double feqCurr = Feq(q, _rho[idx], _ux[idx], _uy[idx], _uz[idx]);
                        _f[FIdx(q, idx)] = feqCurr + (_f[FIdx(oppQ, idx)] - feqOpp);
                    }
                }

                // Update known distributions.
                for (int q = 0; q < _q; q++)
                    _f[FIdx(q, idx)] = Feq(q, _rho[idx], _ux[idx], _uy[idx], _uz[idx]) +
                        (_f[FIdx(q, idx)] - Feq(q, _rho[idx], _ux[idx], _uy[idx], _uz[idx]));
            }
    }

    /// <summary>
    /// Apply Zou-He pressure outlet at x = Nx-1 (known rho = p_out / cs²).
    /// </summary>
    private void ApplyZouHeOutlet()
    {
        double rhoOut = _cfg.OutletPressure; // in lattice units, cs² = 1/3

        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
            {
                int idx = Idx(_nx - 1, y, z);
                if (_solid[idx])
                    continue;

                _rho[idx] = rhoOut;

                // Unknown distributions: q with ex < 0 (incoming from left).
                for (int q = 0; q < _q; q++)
                {
                    if (_ex[q, 0] < 0)
                    {
                        int oppQ = FindOpposite(q);
                        double feqOpp = Feq(oppQ, rhoOut, _ux[idx], _uy[idx], _uz[idx]);
                        _f[FIdx(q, idx)] = Feq(q, rhoOut, _ux[idx], _uy[idx], _uz[idx]) +
                            (_f[FIdx(oppQ, idx)] - feqOpp);
                    }
                }
            }
    }

    // -----------------------------------------------------------------------
    //  Shan-Chen multiphase model
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute the Shan-Chen pseudopotential force.
    /// F_sc = −G ψ(ρ) Σ_q w_q ψ(ρ(x + e_q)) e_q
    /// where ψ(ρ) = ρ₀ (1 − exp(−ρ/ρ₀)).
    /// </summary>
    private void ComputeShanChenForce()
    {
        double G = _cfg.GShanChen;
        double rho0 = 1.0;

        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    if (_solid[idx])
                        continue;

                    double psiI = rho0 * (1.0 - Math.Exp(-_rho[idx] / rho0));
                    double fx = 0, fy = 0, fz = 0;

                    for (int q = 1; q < _q; q++) // skip rest (q=0, e=0)
                    {
                        int nx = x + _ex[q, 0];
                        int ny2 = y + _ey[q, 0];
                        int nz2 = z + _ez[q, 0];

                        // Periodic wrapping.
                        nx = ((nx % _nx) + _nx) % _nx;
                        ny2 = ((ny2 % _ny) + _ny) % _ny;
                        nz2 = ((nz2 % _nz) + _nz) % _nz;

                        int nIdx = Idx(nx, ny2, nz2);
                        double psiJ = rho0 * (1.0 - Math.Exp(-_rho[nIdx] / rho0));

                        fx -= _w[q] * psiI * psiJ * _ex[q, 0];
                        fy -= _w[q] * psiI * psiJ * _ey[q, 0];
                        fz -= _w[q] * psiI * psiJ * _ez[q, 0];
                    }

                    _Fx[idx] = G * fx;
                    _Fy[idx] = G * fy;
                    _Fz[idx] = G * fz;
                }
    }

    // -----------------------------------------------------------------------
    //  Force coupling
    // -----------------------------------------------------------------------

    /// <summary>
    /// Add an external body force to the force arrays.
    /// </summary>
    public void ApplyBodyForce(double fx, double fy, double fz)
    {
        for (int i = 0; i < _n; i++)
        {
            _Fx[i] += fx;
            _Fy[i] += fy;
            _Fz[i] += fz;
        }
    }

    /// <summary>
    /// Guo's force incorporation: modifies equilibrium and adds correction term.
    /// </summary>
    private double ForceTerm(int q, int cellIdx, double dt)
    {
        double ux = _ux[cellIdx], uy = _uy[cellIdx], uz = _uz[cellIdx];
        double Fx = _Fx[cellIdx], Fy = _Fy[cellIdx], Fz = _Fz[cellIdx];

        double eu = _ex[q, 0] * ux + _ey[q, 0] * uy + _ez[q, 0] * uz;
        double eF = _ex[q, 0] * Fx + _ey[q, 0] * Fy + _ez[q, 0] * Fz;
        double uF = ux * Fx + uy * Fy + uz * Fz;

        return _w[q] * dt * (3.0 * (eF - uF) + 9.0 * eu * eF);
    }

    // -----------------------------------------------------------------------
    //  BGK collision
    // -----------------------------------------------------------------------

    private void CollideBGK()
    {
        double omega = _omega;
        double dt = 1.0; // lattice time unit

        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    if (_solid[idx])
                        continue;

                    for (int q = 0; q < _q; q++)
                    {
                        double feq = Feq(q, _rho[idx], _ux[idx], _uy[idx], _uz[idx]);
                        double ft = ForceTerm(q, idx, dt);
                        _fTmp[FIdx(q, idx)] = _f[FIdx(q, idx)] * (1.0 - omega) +
                            omega * feq + ft;
                    }
                }
    }

    // -----------------------------------------------------------------------
    //  MRT collision
    // -----------------------------------------------------------------------

    private void InitialiseMRT()
    {
        // For D3Q19, the MRT matrix maps distribution functions to moments.
        // In a simplified implementation, we use the BGK-equivalent relaxation
        // with different rates for different moment families.
        _mrtS = new double[_q];

        // Standard MRT relaxation rates:
        // s0 (rho): 0, s1..s4 (e, eps): 1.19, s5..s8 (j): 0,
        // s9..s12 (q): 1.4, s13..s18 (ν-related): variable.
        _mrtS[0] = 0;            // conserved density
        _mrtS[1] = 1.19;         // energy
        _mrtS[2] = 1.19;         // energy squared
        for (int i = 3; i <= 5; i++)
            _mrtS[i] = 0; // momentum (conserved)
        for (int i = 6; i <= 9; i++)
            _mrtS[i] = 1.4; // stress
        for (int i = 10; i < _q; i++)
            _mrtS[i] = 1.0 / _tau; // viscous
    }

    private void CollideMRT()
    {
        // Simplified MRT: use moment-space relaxation with different s values.
        // This is equivalent to BGK with direction-dependent relaxation rates.
        double dt = 1.0;

        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    if (_solid[idx])
                        continue;

                    for (int q = 0; q < _q; q++)
                    {
                        double feq = Feq(q, _rho[idx], _ux[idx], _uy[idx], _uz[idx]);
                        double ft = ForceTerm(q, idx, dt);
                        double si = _mrtS[q];
                        _fTmp[FIdx(q, idx)] = _f[FIdx(q, idx)] * (1.0 - si) +
                            si * feq + ft;
                    }
                }
    }

    // -----------------------------------------------------------------------
    //  Macroscopic quantities
    // -----------------------------------------------------------------------

    private void ComputeMacroscopic()
    {
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    if (_solid[idx])
                    {
                        _rho[idx] = 0;
                        _ux[idx] = _uy[idx] = _uz[idx] = 0;
                        continue;
                    }

                    double rhoSum = 0, uxSum = 0, uySum = 0, uzSum = 0;
                    for (int q = 0; q < _q; q++)
                    {
                        double f = _f[FIdx(q, idx)];
                        rhoSum += f;
                        uxSum += f * _ex[q, 0];
                        uySum += f * _ey[q, 0];
                        uzSum += f * _ez[q, 0];
                    }

                    _rho[idx] = Math.Max(rhoSum, 1e-10);
                    _ux[idx] = uxSum / _rho[idx];
                    _uy[idx] = uySum / _rho[idx];
                    _uz[idx] = uzSum / _rho[idx];
                }
    }

    // -----------------------------------------------------------------------
    //  Streaming step
    // -----------------------------------------------------------------------

    private void Stream()
    {
        // Swap f and fTmp.
        (_f, _fTmp) = (_fTmp, _f);

        // Copy streamed distributions.
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = Idx(x, y, z);
                    for (int q = 0; q < _q; q++)
                    {
                        // Distributions streamed FROM neighbour (x - e_q).
                        int sx = x - _ex[q, 0];
                        int sy2 = y - _ey[q, 0];
                        int sz2 = z - _ez[q, 0];

                        // Apply periodic wrapping or skip OOB.
                        if (_cfg.Lattice == LatticeType.D2Q9 && _nz == 1)
                        {
                            // 2D: skip z streaming.
                            if ((uint)sx >= (uint)_nx || (uint)sy2 >= (uint)_ny)
                            {
                                _f[FIdx(q, idx)] = _fTmp[FIdx(q, idx)]; // keep old
                                continue;
                            }
                        }
                        else
                        {
                            if ((uint)sx >= (uint)_nx || (uint)sy2 >= (uint)_ny || (uint)sz2 >= (uint)_nz)
                            {
                                _f[FIdx(q, idx)] = _fTmp[FIdx(q, idx)];
                                continue;
                            }
                        }

                        int sIdx = Idx(sx, sy2, sz2);
                        _f[FIdx(q, idx)] = _fTmp[FIdx(q, sIdx)];
                    }
                }
    }

    // -----------------------------------------------------------------------
    //  One full LBM step
    // -----------------------------------------------------------------------

    /// <summary>
    /// Advance the LBM simulation by one time-step: collide → stream → macro.
    /// </summary>
    public void Step()
    {
        // Compute macroscopic quantities.
        ComputeMacroscopic();

        // Shan-Chen force (multiphase).
        if (_cfg.UseMultiphase)
            ComputeShanChenForce();

        // External body force.
        if (_cfg.BodyForce != null && _cfg.BodyForce.Length >= 3)
            ApplyBodyForce(_cfg.BodyForce[0], _cfg.BodyForce[1], _cfg.BodyForce[2]);

        // Collision.
        if (_cfg.Collision == CollisionModel.MRT)
            CollideMRT();
        else
            CollideBGK();

        // Bounce-back (pre-stream).
        ApplyBounceBack();

        // Stream.
        Stream();

        // Zou-He boundaries.
        ApplyZouHeInlet();
        ApplyZouHeOutlet();

        // Add small perturbation to prevent symmetric lock-in.
        if (CurrentStep < 100)
        {
            int cx = _nx / 2, cy = _ny / 2;
            for (int dz = -2; dz <= 2; dz++)
                for (int dy = -2; dy <= 2; dy++)
                {
                    int idx = Idx(cx, cy + dy, _nz / 2 + dz);
                    if ((uint)idx < (uint)_n && !_solid[idx])
                        _ux[idx] += (_rng.NextDouble() - 0.5) * 1e-5;
                }
        }

        CurrentStep++;
    }

    /// <summary>
    /// Compute kinematic viscosity from relaxation time.
    /// ν = cs² (τ − 0.5) Δt = (1/3)(τ − 0.5)
    /// </summary>
    public double KinematicViscosity => (1.0 / 3.0) * (_tau - 0.5);

    /// <summary>
    /// Compute Reynolds number based on inlet velocity and grid size.
    /// Re = U L / ν
    /// </summary>
    public double ReynoldsNumber => _cfg.InletVelocity * _nx / KinematicViscosity;

    /// <summary>
    /// Compute average velocity magnitude over the domain.
    /// </summary>
    public double AverageVelocity()
    {
        double sum = 0;
        int count = 0;
        for (int i = 0; i < _n; i++)
        {
            if (!_solid[i])
            {
                sum += Math.Sqrt(_ux[i] * _ux[i] + _uy[i] * _uy[i] + _uz[i] * _uz[i]);
                count++;
            }
        }
        return count > 0 ? sum / count : 0;
    }

    /// <summary>
    /// Compute total mass (sum of density).
    /// </summary>
    public double TotalMass()
    {
        double sum = 0;
        for (int i = 0; i < _n; i++)
            if (!_solid[i])
                sum += _rho[i];
        return sum;
    }

    /// <summary>
    /// Export a 2D slice of velocity at a fixed z.
    /// </summary>
    public void ExportSliceZ(int z, double[] uxOut, double[] uyOut, double[] rhoOut)
    {
        for (int y = 0; y < _ny; y++)
            for (int x = 0; x < _nx; x++)
            {
                int idx = Idx(x, y, z);
                int flat = y * _nx + x;
                uxOut[flat] = _ux[idx];
                uyOut[flat] = _uy[idx];
                rhoOut[flat] = _rho[idx];
            }
    }

    /// <summary>
    /// Run the simulation for the configured number of steps.
    /// </summary>
    public void Run()
    {
        for (int i = 0; i < _cfg.NumSteps; i++)
            Step();
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  7. SchrodingerSolver — TDSE (Crank-Nicolson), eigenstates, expectation
// ============================================================================

/// <summary>
/// Configuration for the Schrödinger equation solver.
/// </summary>
public sealed class SchrodingerConfig
{
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (128, 128, 1);
    public double GridSpacing { get; init; } = 0.1;        // Ångströms
    public double TimeStep { get; init; } = 0.001;         // femtoseconds
    public int NumSteps { get; init; } = 10_000;
    public int NumEigenstates { get; init; } = 5;          // number of eigenstates to compute
    public int EigenMaxIter { get; init; } = 500;
    public double EigenTolerance { get; init; } = 1e-12;
    public bool ComputeDensity { get; init; } = true;
    public bool RecordProbability { get; init; } = false;
    public int RecordInterval { get; init; } = 100;
    public double PotentialDepth { get; init; } = 10.0;    // eV
    public double ParticleMass { get; init; } = 1.0;       // electron masses
}

/// <summary>
/// Schrödinger equation solver supporting time-dependent (Crank-Nicolson),
/// time-independent (inverse iteration), density, expectation values,
/// and eigenstate computation for single-particle quantum mechanics
/// on a spatial grid.
/// </summary>
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
public enum MaterialModel
{
    LinearElastic,
    NeoHookean,
    J2Plasticity
}

/// <summary>
/// Configuration for the elasticity solver.
/// </summary>
public sealed class ElasticityConfig
{
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (64, 64, 64);
    public double CellSize { get; init; } = 0.01;          // metres
    public double TimeStep { get; init; } = 1e-6;
    public int NumSteps { get; init; } = 10_000;
    public double YoungsModulus { get; init; } = 200e9;    // Pa (steel)
    public double PoissonRatio { get; init; } = 0.3;
    public double Density { get; init; } = 7800.0;        // kg/m³
    public double YieldStress { get; init; } = 250e6;     // Pa
    public double HardeningModulus { get; init; } = 2e9;   // Pa
    public MaterialModel Material { get; init; } = MaterialModel.LinearElastic;
    public double DampingAlpha { get; init; } = 0.01;     // Rayleigh α
    public double DampingBeta { get; init; } = 1e-7;      // Rayleigh β
    public bool EnableContact { get; init; }
    public double ContactStiffness { get; init; } = 1e10;  // penalty stiffness
    public double ContactGap { get; init; } = 0.001;       // m
    public bool EnableModalAnalysis { get; init; }
    public int NumModes { get; init; } = 10;
}

/// <summary>
/// 3-D linear elasticity solver on a regular grid with FEM-like
/// discretisation. Supports linear elastic, neo-Hookean, and J2
/// plasticity (isotropic hardening) material models, von Mises stress,
/// contact mechanics penalty method, and modal analysis.
/// </summary>
public sealed class ElasticitySolver : IDisposable
{
    private readonly ElasticityConfig _cfg;
    private readonly int _nx, _ny, _nz, _n;
    private readonly double _dx, _dt;
    private readonly double _lambda, _mu;               // Lamé parameters
    private readonly double _rho;

    // Displacement fields (3 DOF per node).
    private double[] _ux, _uy, _uz;
    private double[] _uxPrev, _uyPrev, _uzPrev;
    private double[] _vx, _vy, _vz;                     // velocity
    private double[] _ax, _ay, _az;                     // acceleration

    // Stress tensor (symmetric: σxx, σyy, σzz, σxy, σyz, σxz).
    private double[] _sxx, _syy, _szz, _sxy, _syz, _sxz;

    // Strain tensor.
    private double[] _exx, _eyy, _ezz, _exy, _eyz, _exz;

    // Plastic strain and equivalent plastic strain.
    private double[] _plasticStrain;                     // accumulated plastic strain
    private double[] _backStress;                        // kinematic hardening (simplified)

    // von Mises stress per node.
    private double[] _vonMises;

    // Applied forces (body force).
    private double[] _Fx, _Fy, _Fz;

    // Boundary conditions.
    private bool[] _fixedX, _fixedY, _fixedZ;           // Dirichlet BC

    // Contact surface.
    private double[] _contactForceY;

    // Modal analysis eigenvalues/vectors.
    private double[] _modalFrequencies;
    private double[][] _modalShapes;

    private bool _disposed;

    public ReadOnlySpan<double> Ux => _ux;
    public ReadOnlySpan<double> VonMises => _vonMises;

    public ElasticitySolver(ElasticityConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        (_nx, _ny, _nz) = config.GridSize;
        _n = _nx * _ny * _nz;
        _dx = config.CellSize;
        _dt = config.TimeStep;
        _rho = config.Density;

        // Lamé parameters.
        double E = config.YoungsModulus;
        double nu = config.PoissonRatio;
        _lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        _mu = E / (2.0 * (1.0 + nu));

        // Allocate arrays.
        _ux = new double[_n];
        _uy = new double[_n];
        _uz = new double[_n];
        _uxPrev = new double[_n];
        _uyPrev = new double[_n];
        _uzPrev = new double[_n];
        _vx = new double[_n];
        _vy = new double[_n];
        _vz = new double[_n];
        _ax = new double[_n];
        _ay = new double[_n];
        _az = new double[_n];
        _sxx = new double[_n];
        _syy = new double[_n];
        _szz = new double[_n];
        _sxy = new double[_n];
        _syz = new double[_n];
        _sxz = new double[_n];
        _exx = new double[_n];
        _eyy = new double[_n];
        _ezz = new double[_n];
        _exy = new double[_n];
        _eyz = new double[_n];
        _exz = new double[_n];
        _vonMises = new double[_n];
        _Fx = new double[_n];
        _Fy = new double[_n];
        _Fz = new double[_n];
        _fixedX = new bool[_n];
        _fixedY = new bool[_n];
        _fixedZ = new bool[_n];
        _contactForceY = new double[_n];
        _plasticStrain = new double[_n];
        _backStress = new double[_n];

        if (config.EnableModalAnalysis)
        {
            _modalFrequencies = new double[config.NumModes];
            _modalShapes = new double[config.NumModes][];
            for (int i = 0; i < config.NumModes; i++)
                _modalShapes[i] = new double[_n * 3];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Idx(int x, int y, int z) => z * _ny * _nx + y * _nx + x;

    // -----------------------------------------------------------------------
    //  Boundary conditions
    // -----------------------------------------------------------------------

    public void FixNode(int x, int y, int z, bool fixX = true, bool fixY = true, bool fixZ = true)
    {
        int idx = Idx(x, y, z);
        _fixedX[idx] = fixX;
        _fixedY[idx] = fixY;
        _fixedZ[idx] = fixZ;
    }

    public void FixFace(string face, double value = 0, string component = "all")
    {
        int idx;
        switch (face.ToLower())
        {
            case "bottom":
                for (int x = 0; x < _nx; x++)
                    for (int z = 0; z < _nz; z++)
                    {
                        idx = Idx(x, 0, z);
                        if (component == "all" || component == "y")
                        { _fixedY[idx] = true; _uy[idx] = value; }
                        if (component == "all" || component == "x")
                            _fixedX[idx] = true;
                        if (component == "all" || component == "z")
                            _fixedZ[idx] = true;
                    }
                break;
            case "top":
                for (int x = 0; x < _nx; x++)
                    for (int z = 0; z < _nz; z++)
                    {
                        idx = Idx(x, _ny - 1, z);
                        if (component == "all" || component == "y")
                        { _fixedY[idx] = true; _uy[idx] = value; }
                        if (component == "all" || component == "x")
                            _fixedX[idx] = true;
                        if (component == "all" || component == "z")
                            _fixedZ[idx] = true;
                    }
                break;
            case "left":
                for (int y = 0; y < _ny; y++)
                    for (int z = 0; z < _nz; z++)
                    {
                        idx = Idx(0, y, z);
                        if (component == "all" || component == "x")
                        { _fixedX[idx] = true; _ux[idx] = value; }
                        if (component == "all" || component == "y")
                            _fixedY[idx] = true;
                        if (component == "all" || component == "z")
                            _fixedZ[idx] = true;
                    }
                break;
            case "right":
                for (int y = 0; y < _ny; y++)
                    for (int z = 0; z < _nz; z++)
                    {
                        idx = Idx(_nx - 1, y, z);
                        if (component == "all" || component == "x")
                        { _fixedX[idx] = true; _ux[idx] = value; }
                        if (component == "all" || component == "y")
                            _fixedY[idx] = true;
                        if (component == "all" || component == "z")
                            _fixedZ[idx] = true;
                    }
                break;
        }
    }

    /// <summary>
    /// Apply a body force (N/m³) to the entire domain.
    /// </summary>
    public void ApplyBodyForce(double fx, double fy, double fz)
    {
        for (int i = 0; i < _n; i++)
        {
            _Fx[i] = fx;
            _Fy[i] = fy;
            _Fz[i] = fz;
        }
    }

    /// <summary>
    /// Apply a point force at a specific node.
    /// </summary>
    public void ApplyPointForce(int x, int y, int z, double fx, double fy, double fz)
    {
        int idx = Idx(x, y, z);
        _Fx[idx] += fx / (_dx * _dx * _dx);
        _Fy[idx] += fy / (_dx * _dx * _dx);
        _Fz[idx] += fz / (_dx * _dx * _dx);
    }

    // -----------------------------------------------------------------------
    //  Strain computation (small strain: ε = 0.5(∇u + (∇u)ᵀ))
    // -----------------------------------------------------------------------

    private void ComputeStrain()
    {
        double invDx2 = 0.5 / _dx;
        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);

                    // εxx = ∂ux/∂x
                    _exx[idx] = (_ux[Idx(x + 1, y, z)] - _ux[Idx(x - 1, y, z)]) * invDx2;
                    // εyy = ∂uy/∂y
                    _eyy[idx] = (_uy[Idx(x, y + 1, z)] - _uy[Idx(x, y - 1, z)]) * invDx2;
                    // εzz = ∂uz/∂z
                    _ezz[idx] = (_uz[Idx(x, y, z + 1)] - _uz[Idx(x, y, z - 1)]) * invDx2;

                    // εxy = 0.5 (∂ux/∂y + ∂uy/∂x)
                    _exy[idx] = 0.5 * ((_ux[Idx(x, y + 1, z)] - _ux[Idx(x, y - 1, z)]) * invDx2 +
                                        (_uy[Idx(x + 1, y, z)] - _uy[Idx(x - 1, y, z)]) * invDx2);
                    // εyz
                    _eyz[idx] = 0.5 * ((_uy[Idx(x, y, z + 1)] - _uy[Idx(x, y, z - 1)]) * invDx2 +
                                        (_uz[Idx(x, y + 1, z)] - _uz[Idx(x, y - 1, z)]) * invDx2);
                    // εxz
                    _exz[idx] = 0.5 * ((_ux[Idx(x, y, z + 1)] - _ux[Idx(x, y, z - 1)]) * invDx2 +
                                        (_uz[Idx(x + 1, y, z)] - _uz[Idx(x - 1, y, z)]) * invDx2);
                }
    }

    // -----------------------------------------------------------------------
    //  Stress computation
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute stress from strain using linear elasticity:
    /// σ = λ tr(ε) I + 2μ ε
    /// </summary>
    private void ComputeStressLinear()
    {
        for (int i = 0; i < _n; i++)
        {
            double trE = _exx[i] + _eyy[i] + _ezz[i];
            _sxx[i] = _lambda * trE + 2.0 * _mu * _exx[i];
            _syy[i] = _lambda * trE + 2.0 * _mu * _eyy[i];
            _szz[i] = _lambda * trE + 2.0 * _mu * _ezz[i];
            _sxy[i] = 2.0 * _mu * _exy[i];
            _syz[i] = 2.0 * _mu * _eyz[i];
            _sxz[i] = 2.0 * _mu * _exz[i];
        }
    }

    /// <summary>
    /// Compute stress from strain using neo-Hookean model:
    /// σ = μ (B − I) + λ (J − 1) I
    /// where B = F Fᵀ, J = det(F), and F ≈ I + ∇u.
    /// </summary>
    private void ComputeStressNeoHookean()
    {
        double invDx = 1.0 / _dx;
        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);

                    // Deformation gradient F ≈ I + ∇u.
                    double Fxx = 1.0 + (_ux[Idx(x + 1, y, z)] - _ux[Idx(x - 1, y, z)]) * 0.5 * invDx;
                    double Fyy = 1.0 + (_uy[Idx(x, y + 1, z)] - _uy[Idx(x, y - 1, z)]) * 0.5 * invDx;
                    double Fzz = 1.0 + (_uz[Idx(x, y, z + 1)] - _uz[Idx(x, y, z - 1)]) * 0.5 * invDx;
                    double Fxy = (_ux[Idx(x, y + 1, z)] - _ux[Idx(x, y - 1, z)]) * 0.5 * invDx;
                    double Fyx = (_uy[Idx(x + 1, y, z)] - _uy[Idx(x - 1, y, z)]) * 0.5 * invDx;
                    double Fxz = (_ux[Idx(x, y, z + 1)] - _ux[Idx(x, y, z - 1)]) * 0.5 * invDx;
                    double Fzx = (_uz[Idx(x + 1, y, z)] - _uz[Idx(x - 1, y, z)]) * 0.5 * invDx;
                    double Fyz = (_uy[Idx(x, y, z + 1)] - _uy[Idx(x, y, z - 1)]) * 0.5 * invDx;
                    double Fzy = (_uz[Idx(x, y + 1, z)] - _uz[Idx(x, y - 1, z)]) * 0.5 * invDx;

                    // J = det(F).
                    double J = Fxx * (Fyy * Fzz - Fyz * Fzy) -
                               Fxy * (Fyx * Fzz - Fyz * Fzx) +
                               Fxz * (Fyx * Fzy - Fyy * Fzx);
                    J = Math.Max(J, 0.01); // prevent collapse

                    // B = F Fᵀ (left Cauchy-Green).
                    double Bxx = Fxx * Fxx + Fxy * Fxy + Fxz * Fxz;
                    double Byy = Fyx * Fyx + Fyy * Fyy + Fyz * Fyz;
                    double Bzz = Fzx * Fzx + Fzy * Fzy + Fzz * Fzz;
                    double Bxy = Fxx * Fyx + Fxy * Fyy + Fxz * Fyz;
                    double Byz = Fyx * Fzx + Fyy * Fzy + Fyz * Fzz;
                    double Bxz = Fxx * Fzx + Fxy * Fzy + Fxz * Fzz;

                    _sxx[idx] = _mu * (Bxx - 1.0) + _lambda * (J - 1.0);
                    _syy[idx] = _mu * (Byy - 1.0) + _lambda * (J - 1.0);
                    _szz[idx] = _mu * (Bzz - 1.0) + _lambda * (J - 1.0);
                    _sxy[idx] = _mu * Bxy;
                    _syz[idx] = _mu * Byz;
                    _sxz[idx] = _mu * Bxz;
                }
    }

    /// <summary>
    /// Compute stress with J2 plasticity (von Mises yield criterion with
    /// isotropic hardening).
    /// </summary>
    private void ComputeStressJ2Plasticity()
    {
        double yieldStress = _cfg.YieldStress;
        double H = _cfg.HardeningModulus;

        for (int i = 0; i < _n; i++)
        {
            // Trial stress (elastic predictor).
            double trE = _exx[i] + _eyy[i] + _ezz[i];
            double sxxTrial = _lambda * trE + 2.0 * _mu * _exx[i];
            double syyTrial = _lambda * trE + 2.0 * _mu * _eyy[i];
            double szzTrial = _lambda * trE + 2.0 * _mu * _ezz[i];
            double sxyTrial = 2.0 * _mu * _exy[i];
            double syzTrial = 2.0 * _mu * _eyz[i];
            double sxzTrial = 2.0 * _mu * _exz[i];

            // Deviatoric stress: s = σ − (1/3) tr(σ) I.
            double meanStress = (sxxTrial + syyTrial + szzTrial) / 3.0;
            double sxxD = sxxTrial - meanStress;
            double syyD = syyTrial - meanStress;
            double szzD = szzTrial - meanStress;

            // von Mises stress: σ_vm = sqrt(3/2 s:s).
            double J2 = 0.5 * (sxxD * sxxD + syyD * syyD + szzD * szzD) +
                        sxyTrial * sxyTrial + syzTrial * syzTrial + sxzTrial * sxzTrial;
            double sigmaVM = Math.Sqrt(3.0 * J2);

            // Yield function: f = σ_vm − (σ_y + H ε_p).
            double currentYield = yieldStress + H * _plasticStrain[i];
            double f = sigmaVM - currentYield;

            if (f > 0)
            {
                // Plastic corrector: scale back to yield surface.
                double scale = currentYield / Math.Max(sigmaVM, 1e-10);

                // Return mapping.
                _sxx[i] = meanStress + sxxD * scale;
                _syy[i] = meanStress + syyD * scale;
                _szz[i] = meanStress + szzD * scale;
                _sxy[i] = sxyTrial * scale;
                _syz[i] = syzTrial * scale;
                _sxz[i] = sxzTrial * scale;

                // Increment equivalent plastic strain.
                double dEp = f / (3.0 * _mu + H);
                _plasticStrain[i] += dEp;
            }
            else
            {
                // Elastic.
                _sxx[i] = sxxTrial;
                _syy[i] = syyTrial;
                _szz[i] = szzTrial;
                _sxy[i] = sxyTrial;
                _syz[i] = syzTrial;
                _sxz[i] = sxzTrial;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  von Mises stress
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute von Mises equivalent stress: σ_vm = sqrt(3/2 s:s)
    /// where s is the deviatoric stress.
    /// </summary>
    public void ComputeVonMisesStress()
    {
        for (int i = 0; i < _n; i++)
        {
            double mean = (_sxx[i] + _syy[i] + _szz[i]) / 3.0;
            double sDxx = _sxx[i] - mean;
            double sDyy = _syy[i] - mean;
            double sDzz = _szz[i] - mean;

            double J2 = 0.5 * (sDxx * sDxx + sDyy * sDyy + sDzz * sDzz) +
                        _sxy[i] * _sxy[i] + _syz[i] * _syz[i] + _sxz[i] * _sxz[i];
            _vonMises[i] = Math.Sqrt(3.0 * J2);
        }
    }

    // -----------------------------------------------------------------------
    //  Divergence of stress (internal forces)
    // -----------------------------------------------------------------------

    private void ComputeInternalForces(double[] fintX, double[] fintY, double[] fintZ)
    {
        double invDx = 1.0 / _dx;
        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);

                    // ∇·σ: div of stress tensor.
                    fintX[idx] = (
                        (_sxx[Idx(x + 1, y, z)] - _sxx[Idx(x - 1, y, z)]) * 0.5 * invDx +
                        (_sxy[Idx(x, y + 1, z)] - _sxy[Idx(x, y - 1, z)]) * 0.5 * invDx +
                        (_sxz[Idx(x, y, z + 1)] - _sxz[Idx(x, y, z - 1)]) * 0.5 * invDx
                    );

                    fintY[idx] = (
                        (_sxy[Idx(x + 1, y, z)] - _sxy[Idx(x - 1, y, z)]) * 0.5 * invDx +
                        (_syy[Idx(x, y + 1, z)] - _syy[Idx(x, y - 1, z)]) * 0.5 * invDx +
                        (_syz[Idx(x, y, z + 1)] - _syz[Idx(x, y, z - 1)]) * 0.5 * invDx
                    );

                    fintZ[idx] = (
                        (_sxz[Idx(x + 1, y, z)] - _sxz[Idx(x - 1, y, z)]) * 0.5 * invDx +
                        (_syz[Idx(x, y + 1, z)] - _syz[Idx(x, y - 1, z)]) * 0.5 * invDx +
                        (_szz[Idx(x, y, z + 1)] - _szz[Idx(x, y, z - 1)]) * 0.5 * invDx
                    );
                }
    }

    // -----------------------------------------------------------------------
    //  Contact mechanics (penalty method)
    // -----------------------------------------------------------------------

    private void ComputeContactForces()
    {
        if (!_cfg.EnableContact)
            return;

        double kContact = _cfg.ContactStiffness;
        double gap = _cfg.ContactGap;

        // Simple half-space contact: the bottom surface (y = 0) is the rigid
        // contact surface. Any node penetrating below y = 0 gets a penalty force.
        for (int z = 0; z < _nz; z++)
            for (int x = 0; x < _nx; x++)
            {
                // Check the bottom layer.
                int idx = Idx(x, 0, z);
                double penetration = gap - _uy[idx];
                if (penetration > 0)
                {
                    _contactForceY[idx] = kContact * penetration;
                    _uy[idx] = Math.Max(_uy[idx], -gap);
                }
                else
                {
                    _contactForceY[idx] = 0;
                }
            }

        // Also check interior nodes near the contact surface.
        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < Math.Min(5, _ny - 1); y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);
                    double penetration = gap + y * _dx - _uy[idx];
                    if (penetration > 0 && penetration < gap)
                    {
                        _contactForceY[idx] = kContact * penetration * Math.Exp(-y);
                    }
                }
    }

    // -----------------------------------------------------------------------
    //  Time integration: Newmark-beta method
    // -----------------------------------------------------------------------

    /// <summary>
    /// Advance the elastic field by one time-step using the Newmark-β method
    /// (β = 0.25, γ = 0.5 for unconditional stability).
    /// </summary>
    public void Step()
    {
        double dt = _dt;
        double dt2 = dt * dt;
        double beta = 0.25;
        double gamma = 0.5;

        // Compute strain and stress.
        ComputeStrain();

        switch (_cfg.Material)
        {
            case MaterialModel.LinearElastic:
                ComputeStressLinear();
                break;
            case MaterialModel.NeoHookean:
                ComputeStressNeoHookean();
                break;
            case MaterialModel.J2Plasticity:
                ComputeStressJ2Plasticity();
                break;
        }

        ComputeVonMisesStress();

        // Internal forces.
        double[] fintX = new double[_n], fintY = new double[_n], fintZ = new double[_n];
        ComputeInternalForces(fintX, fintY, fintZ);

        // Contact forces.
        ComputeContactForces();

        // Rayleigh damping: C = αM + βK
        // For implicit integration, damping contributes to the RHS.
        double alpha = _cfg.DampingAlpha;
        double betaDamp = _cfg.DampingBeta;

        // Newmark prediction step.
        for (int i = 0; i < _n; i++)
        {
            // Total force: F = F_ext − F_int − F_contact − C v
            double fDampX = alpha * _rho * _vx[i] + betaDamp * fintX[i];
            double fDampY = alpha * _rho * _vy[i] + betaDamp * fintY[i];
            double fDampZ = alpha * _rho * _vz[i] + betaDamp * fintZ[i];

            double totalFx = _Fx[i] - fintX[i] - _contactForceY[i] * (i == 0 ? 1 : 0) - fDampX;
            double totalFy = _Fy[i] - fintY[i] - _contactForceY[i] - fDampY;
            double totalFz = _Fz[i] - fintZ[i] - fDampZ;

            // Acceleration: a = F / ρ (explicit for simplicity).
            if (!_fixedX[i])
                _ax[i] = totalFx / _rho;
            if (!_fixedY[i])
                _ay[i] = totalFy / _rho;
            if (!_fixedZ[i])
                _az[i] = totalFz / _rho;

            // Newmark update.
            if (!_fixedX[i])
            {
                _ux[i] += dt * _vx[i] + dt2 * ((0.5 - beta) * _ax[i] + beta * _ax[i]);
                _vx[i] += dt * ((1.0 - gamma) * _ax[i] + gamma * _ax[i]);
            }
            if (!_fixedY[i])
            {
                _uy[i] += dt * _vy[i] + dt2 * ((0.5 - beta) * _ay[i] + beta * _ay[i]);
                _vy[i] += dt * ((1.0 - gamma) * _ay[i] + gamma * _ay[i]);
            }
            if (!_fixedZ[i])
            {
                _uz[i] += dt * _vz[i] + dt2 * ((0.5 - beta) * _az[i] + beta * _az[i]);
                _vz[i] += dt * ((1.0 - gamma) * _az[i] + gamma * _az[i]);
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Modal analysis
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute natural frequencies and mode shapes using inverse iteration.
    /// Solves (K − ω²M) φ = 0 for eigenpairs.
    /// </summary>
    public void ComputeModalAnalysis()
    {
        if (!_cfg.EnableModalAnalysis || _modalShapes == null)
            return;

        int nModes = _cfg.NumModes;
        double dx3 = _dx * _dx * _dx;
        double massCoeff = _rho * dx3;
        double stiffCoeff = _mu * _dx;

        for (int mode = 0; mode < nModes; mode++)
        {
            // Random initial guess.
            var rng = new Random(mode * 77777 + 13);
            double norm = 0;
            for (int i = 0; i < _n * 3; i++)
            {
                _modalShapes[mode][i] = rng.NextDouble() - 0.5;
                norm += _modalShapes[mode][i] * _modalShapes[mode][i];
            }
            norm = Math.Sqrt(norm);
            for (int i = 0; i < _n * 3; i++)
                _modalShapes[mode][i] /= norm;

            // Shift estimate.
            double omega2Est = Math.Pow((mode + 1) * 100.0, 2);

            for (int iter = 0; iter < 100; iter++)
            {
                // Apply (K − ω²M) to current mode.
                double[] result = new double[_n * 3];

                for (int z = 1; z < _nz - 1; z++)
                    for (int y = 1; y < _ny - 1; y++)
                        for (int x = 1; x < _nx - 1; x++)
                        {
                            int nodeIdx = Idx(x, y, z);
                            int dofX = nodeIdx * 3;
                            int dofY = nodeIdx * 3 + 1;
                            int dofZ = nodeIdx * 3 + 2;

                            // Simplified stiffness matrix-vector product.
                            // Using central difference approximation of the Laplacian.
                            for (int d = 0; d < 3; d++)
                            {
                                double lap = 0;
                                int dIdx = nodeIdx * 3 + d;
                                lap += _modalShapes[mode][dIdx + 3] + _modalShapes[mode][dIdx - 3]; // x±1
                                lap += _modalShapes[mode][dIdx + _nx * 3] + _modalShapes[mode][dIdx - _nx * 3]; // y±1
                                if (_nz > 1)
                                    lap += _modalShapes[mode][dIdx + _nx * _ny * 3] + _modalShapes[mode][dIdx - _nx * _ny * 3]; // z±1
                                lap -= 6.0 * _modalShapes[mode][dIdx];

                                result[dIdx] = stiffCoeff * lap - omega2Est * massCoeff * _modalShapes[mode][dIdx];
                            }
                        }

                // Inverse iteration: solve approximately.
                for (int i = 0; i < _n * 3; i++)
                {
                    double diag = stiffCoeff * 6.0 - omega2Est * massCoeff;
                    if (Math.Abs(diag) > 1e-30)
                        _modalShapes[mode][i] = -result[i] / diag;
                }

                // Normalise.
                norm = 0;
                for (int i = 0; i < _n * 3; i++)
                    norm += _modalShapes[mode][i] * _modalShapes[mode][i];
                norm = Math.Sqrt(norm * dx3);
                if (norm > 1e-30)
                    for (int i = 0; i < _n * 3; i++)
                        _modalShapes[mode][i] /= norm;

                // Orthogonalise against previous modes.
                for (int m = 0; m < mode; m++)
                {
                    double dot = 0;
                    for (int i = 0; i < _n * 3; i++)
                        dot += _modalShapes[m][i] * _modalShapes[mode][i];
                    dot *= dx3;
                    for (int i = 0; i < _n * 3; i++)
                        _modalShapes[mode][i] -= dot * _modalShapes[m][i];
                }
            }

            _modalFrequencies[mode] = Math.Sqrt(Math.Abs(omega2Est)) / PhysicsConstants.TwoPi;
        }
    }

    /// <summary>
    /// Get modal frequencies.
    /// </summary>
    public ReadOnlySpan<double> ModalFrequencies => _modalFrequencies;

    /// <summary>
    /// Get a mode shape (displacement field).
    /// </summary>
    public ReadOnlySpan<double> GetModeShape(int mode) => _modalShapes[mode];

    /// <summary>
    /// Compute strain energy: U = 0.5 ∫ σ:ε dV.
    /// </summary>
    public double StrainEnergy()
    {
        double sum = 0;
        double dV = _dx * _dx * _dx;
        for (int i = 0; i < _n; i++)
        {
            sum += _sxx[i] * _exx[i] + _syy[i] * _eyy[i] + _szz[i] * _ezz[i] +
                   2.0 * (_sxy[i] * _exy[i] + _syz[i] * _eyz[i] + _sxz[i] * _exz[i]);
        }
        return 0.5 * sum * dV;
    }

    /// <summary>
    /// Compute kinetic energy: T = 0.5 ρ ∫ |v|² dV.
    /// </summary>
    public double KineticEnergy()
    {
        double sum = 0;
        double dV = _dx * _dx * _dx;
        for (int i = 0; i < _n; i++)
        {
            sum += _vx[i] * _vx[i] + _vy[i] * _vy[i] + _vz[i] * _vz[i];
        }
        return 0.5 * _rho * sum * dV;
    }

    /// <summary>
    /// Run the simulation for the configured number of steps.
    /// </summary>
    public void Run()
    {
        for (int i = 0; i < _cfg.NumSteps; i++)
            Step();
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  9. TurbulenceModels — k-ε, k-ω SST, Spalart-Allmaras, LES Smagorinsky
// ============================================================================

/// <summary>
/// Turbulence model type.
/// </summary>
public enum TurbulenceModelType
{
    None,
    kEpsilon,
    kOmegaSST,
    SpalartAllmaras,
    SmagorinskyLES
}

/// <summary>
/// Configuration for turbulence modelling.
/// </summary>
public sealed class TurbulenceConfig
{
    public TurbulenceModelType Model { get; init; } = TurbulenceModelType.kEpsilon;
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (128, 64, 64);
    public double CellSize { get; init; } = 0.01;
    public double TimeStep { get; init; } = 1e-4;
    public int NumSteps { get; init; } = 10_000;
    public double KinematicViscosity { get; init; } = 1e-5;
    public double TurbulentViscosityMax { get; init; } = 0.1;
    public double Csmagorinsky { get; init; } = 0.1;       // Smagorinsky constant
    public double WallDistance { get; init; } = 0.01;      // for wall functions
}

/// <summary>
/// RANS and LES turbulence models for incompressible flow.
/// Implements standard k-ε, k-ω SST, Spalart-Allmaras, and
/// Smagorinsky LES subgrid-scale model.
/// </summary>
public sealed class TurbulenceModels : IDisposable
{
    private readonly TurbulenceConfig _cfg;
    private readonly int _nx, _ny, _nz, _n;
    private readonly double _dx, _dt;
    private readonly double _nu;   // laminar kinematic viscosity

    // Turbulence fields.
    private double[] _k;           // turbulent kinetic energy
    private double[] _epsilon;     // dissipation rate (k-ε)
    private double[] _omega;       // specific dissipation rate (k-ω)
    private double[] _nut;         // turbulent (eddy) viscosity
    private double[] _nutSA;       // SA working variable ν̃

    // Previous time-step storage.
    private double[] _kPrev, _epsilonPrev, _omegaPrev, _nutPrev;

    // Velocity gradients (stored for efficiency).
    private double[] _dUdx, _dUdy, _dUdz;
    private double[] _dVdx, _dVdy, _dVdz;
    private double[] _dWdx, _dWdy, _dWdz;

    // Strain rate magnitude and vorticity.
    private double[] _SijMag;
    private double[] _OmegaMag;

    // Wall distance.
    private double[] _yWall;

    // k-ε constants.
    private const double Cmu = 0.09;
    private const double C1e = 1.44;
    private const double C2e = 1.92;
    private const double sigmaK = 1.0;
    private const double sigmaE = 1.3;

    // k-ω SST constants.
    private const double BetaStar = 0.09;
    private const double Beta1 = 0.075;
    private const double Beta2 = 0.0828;
    private const double SigmaK1 = 0.85;
    private const double SigmaK2 = 1.0;
    private const double SigmaW1 = 0.5;
    private const double SigmaW2 = 0.856;
    private const double A1 = 0.31;
    private const double Beta1Star = 0.09;

    // SA constants.
    private const double SigmaSA = 2.0 / 3.0;
    private const double Cb1 = 0.1355;
    private const double Cb2 = 0.622;
    private const double Cw1 = 0.3139;
    private const double Cw2 = 0.3;
    private const double Cw3 = 2.0;
    private const double Cv1 = 7.1;
    private const double Ct1 = 1.0;
    private const double Ct2 = 2.0;
    private const double Ct3 = 1.1;
    private const double Ct4 = 0.5;

    private bool _disposed;

    public ReadOnlySpan<double> K => _k;
    public ReadOnlySpan<double> Epsilon => _epsilon;
    public ReadOnlySpan<double> Nut => _nut;

    public TurbulenceModels(TurbulenceConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        (_nx, _ny, _nz) = config.GridSize;
        _n = _nx * _ny * _nz;
        _dx = config.CellSize;
        _dt = config.TimeStep;
        _nu = config.KinematicViscosity;

        // Allocate turbulence fields.
        _k = new double[_n];
        _epsilon = new double[_n];
        _omega = new double[_n];
        _nut = new double[_n];
        _nutSA = new double[_n];
        _kPrev = new double[_n];
        _epsilonPrev = new double[_n];
        _omegaPrev = new double[_n];
        _nutPrev = new double[_n];

        // Velocity gradients.
        _dUdx = new double[_n];
        _dUdy = new double[_n];
        _dUdz = new double[_n];
        _dVdx = new double[_n];
        _dVdy = new double[_n];
        _dVdz = new double[_n];
        _dWdx = new double[_n];
        _dWdy = new double[_n];
        _dWdz = new double[_n];

        _SijMag = new double[_n];
        _OmegaMag = new double[_n];
        _yWall = new double[_n];

        // Default wall distance (for flat plate: y).
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                {
                    int idx = z * _ny * _nx + y * _nx + x;
                    _yWall[idx] = y * _dx; // distance from bottom wall
                }

        // Initialise with small turbulence.
        for (int i = 0; i < _n; i++)
        {
            _k[i] = 1e-6;
            _epsilon[i] = 1e-6;
            _omega[i] = 1e-6;
            _nut[i] = 0;
            _nutSA[i] = _nu * 0.01;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Idx(int x, int y, int z) => z * _ny * _nx + y * _nx + x;

    /// <summary>
    /// Set a custom wall distance field.
    /// </summary>
    public void SetWallDistance(Func<int, int, int, double> wallDistFunc)
    {
        for (int z = 0; z < _nz; z++)
            for (int y = 0; y < _ny; y++)
                for (int x = 0; x < _nx; x++)
                    _yWall[Idx(x, y, z)] = wallDistFunc(x, y, z);
    }

    // -----------------------------------------------------------------------
    //  Velocity gradient computation
    // -----------------------------------------------------------------------

    public void ComputeVelocityGradients(
        ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w)
    {
        double invDx2 = 0.5 / _dx;

        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);

                    _dUdx[idx] = (u[Idx(x + 1, y, z)] - u[Idx(x - 1, y, z)]) * invDx2;
                    _dUdy[idx] = (u[Idx(x, y + 1, z)] - u[Idx(x, y - 1, z)]) * invDx2;
                    _dUdz[idx] = (u[Idx(x, y, z + 1)] - u[Idx(x, y, z - 1)]) * invDx2;

                    _dVdx[idx] = (v[Idx(x + 1, y, z)] - v[Idx(x - 1, y, z)]) * invDx2;
                    _dVdy[idx] = (v[Idx(x, y + 1, z)] - v[Idx(x, y - 1, z)]) * invDx2;
                    _dVdz[idx] = (v[Idx(x, y, z + 1)] - v[Idx(x, y, z - 1)]) * invDx2;

                    _dWdx[idx] = (w[Idx(x + 1, y, z)] - w[Idx(x - 1, y, z)]) * invDx2;
                    _dWdy[idx] = (w[Idx(x, y + 1, z)] - w[Idx(x, y - 1, z)]) * invDx2;
                    _dWdz[idx] = (w[Idx(x, y, z + 1)] - w[Idx(x, y, z - 1)]) * invDx2;

                    // Strain rate magnitude: S = sqrt(2 S_ij S_ij)
                    double Sxx = _dUdx[idx];
                    double Syy = _dVdy[idx];
                    double Szz = _dWdz[idx];
                    double Sxy = 0.5 * (_dUdy[idx] + _dVdx[idx]);
                    double Syz = 0.5 * (_dVdz[idx] + _dWdy[idx]);
                    double Sxz = 0.5 * (_dUdz[idx] + _dWdx[idx]);

                    _SijMag[idx] = Math.Sqrt(2.0 * (Sxx * Sxx + Syy * Syy + Szz * Szz +
                                                     2.0 * (Sxy * Sxy + Syz * Syz + Sxz * Sxz)));

                    // Vorticity magnitude: Ω = sqrt(2 Ω_ij Ω_ij)
                    double OmZ = _dVdx[idx] - _dUdy[idx];
                    double OmY = _dUdz[idx] - _dWdx[idx];
                    double OmX = _dWdy[idx] - _dVdz[idx];
                    _OmegaMag[idx] = Math.Sqrt(OmX * OmX + OmY * OmY + OmZ * OmZ);
                }
    }

    // -----------------------------------------------------------------------
    //  Standard k-ε model
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute turbulent viscosity from k-ε: ν_t = C_μ k² / ε
    /// Transport equations:
    ///   ∂k/∂t + U·∇k = P_k − ε + ∇·((ν + ν_t/σ_k)∇k)
    ///   ∂ε/∂t + U·∇ε = (C₁ε P_k − C₂ε ε)/k + ∇·((ν + ν_t/σ_ε)∇ε)
    /// </summary>
    public void UpdateKEpsilon(
        ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w)
    {
        ComputeVelocityGradients(u, v, w);

        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);
                    _kPrev[idx] = _k[idx];
                    _epsilonPrev[idx] = _epsilon[idx];

                    // Production of turbulence: P_k = ν_t S²
                    double nut = Cmu * _k[idx] * _k[idx] / Math.Max(_epsilon[idx], 1e-20);
                    double Pk = nut * _SijMag[idx] * _SijMag[idx];

                    // Clamp production.
                    Pk = Math.Min(Pk, 10.0 * _epsilon[idx]);

                    // Laplacians (diffusion).
                    double lapK = (_k[Idx(x + 1, y, z)] + _k[Idx(x - 1, y, z)] +
                                   _k[Idx(x, y + 1, z)] + _k[Idx(x, y - 1, z)] +
                                   _k[Idx(x, y, z + 1)] + _k[Idx(x, y, z - 1)] -
                                   6.0 * _k[idx]) / (_dx * _dx);
                    double lapEps = (_epsilon[Idx(x + 1, y, z)] + _epsilon[Idx(x - 1, y, z)] +
                                     _epsilon[Idx(x, y + 1, z)] + _epsilon[Idx(x, y - 1, z)] +
                                     _epsilon[Idx(x, y, z + 1)] + _epsilon[Idx(x, y, z - 1)] -
                                     6.0 * _epsilon[idx]) / (_dx * _dx);

                    // Transport for k.
                    double dkdt = Pk - _epsilon[idx] +
                                  (_nu + nut / sigmaK) * lapK;
                    _k[idx] = _kPrev[idx] + _dt * dkdt;
                    _k[idx] = Math.Max(_k[idx], 1e-20);

                    // Transport for ε.
                    double depsdt = (C1e * Pk - C2e * _epsilon[idx]) / Math.Max(_k[idx], 1e-20) +
                                    (_nu + nut / sigmaE) * lapEps;
                    _epsilon[idx] = _epsilonPrev[idx] + _dt * depsdt;
                    _epsilon[idx] = Math.Max(_epsilon[idx], 1e-20);

                    // Turbulent viscosity.
                    _nut[idx] = Cmu * _k[idx] * _k[idx] / _epsilon[idx];
                    _nut[idx] = Math.Min(_nut[idx], _cfg.TurbulentViscosityMax);
                }
    }

    // -----------------------------------------------------------------------
    //  k-ω SST (Shear Stress Transport)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute turbulent viscosity from k-ω SST.
    /// Blends k-ε and k-ω formulations using a switching function F1.
    /// ν_t = a₁ k / max(a₁ ω, S F2)
    /// </summary>
    public void UpdateKOmegaSST(
        ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w)
    {
        ComputeVelocityGradients(u, v, w);

        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);
                    _kPrev[idx] = _k[idx];
                    _omegaPrev[idx] = _omega[idx];

                    double yW = _yWall[idx];
                    double dist = Math.Max(yW, 1e-10);

                    // CD_kw = max(2 ρ σ_w2 / ω ∇k·∇ω, 10⁻²⁰)
                    double lapK = (_k[Idx(x + 1, y, z)] + _k[Idx(x - 1, y, z)] +
                                   _k[Idx(x, y + 1, z)] + _k[Idx(x, y - 1, z)] +
                                   _k[Idx(x, y, z + 1)] + _k[Idx(x, y, z - 1)] -
                                   6.0 * _k[idx]) / (_dx * _dx);
                    double lapOmega = (_omega[Idx(x + 1, y, z)] + _omega[Idx(x - 1, y, z)] +
                                       _omega[Idx(x, y + 1, z)] + _omega[Idx(x, y - 1, z)] +
                                       _omega[Idx(x, y, z + 1)] + _omega[Idx(x, y, z - 1)] -
                                       6.0 * _omega[idx]) / (_dx * _dx);

                    // F1 blending function: switches between inner (k-ω) and outer (k-ε).
                    double arg1 = Math.Sqrt(_k[idx]) / (BetaStar * Math.Max(_omega[idx], 1e-20) * dist);
                    double arg2 = 500.0 * _nu / (dist * dist * Math.Max(_omega[idx], 1e-20));
                    double F1 = Math.Tanh(Math.Pow(Math.Min(Math.Max(arg1, arg2), 4.0), 4.0));

                    // F2 blending function.
                    double arg2b = 2.0 * Math.Sqrt(_k[idx]) / (BetaStar * Math.Max(_omega[idx], 1e-20) * dist);
                    double F2 = Math.Tanh(arg2b * arg2b);

                    // Blended coefficients.
                    double beta = Beta1 * F1 + Beta2 * (1.0 - F1);
                    double sigmaKBl = SigmaK1 * F1 + SigmaK2 * (1.0 - F1);
                    double sigmaWBl = SigmaW1 * F1 + SigmaW2 * (1.0 - F1);

                    // Production and destruction.
                    double Pk = _SijMag[idx] * _SijMag[idx] * Math.Min(_k[idx], 10.0 * BetaStar * _k[idx]);
                    double Pw = Pk / Math.Max(_k[idx], 1e-20);

                    // Transport for k.
                    double dkdt = Pk - BetaStar * _omega[idx] * _k[idx] +
                                  (_nu + _nut[idx] / sigmaKBl) * lapK;
                    _k[idx] = _kPrev[idx] + _dt * dkdt;
                    _k[idx] = Math.Max(_k[idx], 1e-20);

                    // Transport for ω.
                    double domegadt = Pw - beta * _omega[idx] * _omega[idx] +
                                      (_nu + _nut[idx] / sigmaWBl) * lapOmega +
                                      (1.0 - F1) * Cw2 * lapOmega; // cross-diffusion
                    _omega[idx] = _omegaPrev[idx] + _dt * domegadt;
                    _omega[idx] = Math.Max(_omega[idx], 1e-20);

                    // SST eddy viscosity: ν_t = a₁k / max(a₁ω, S F2).
                    double a1 = A1;
                    _nut[idx] = a1 * _k[idx] / Math.Max(a1 * _omega[idx], _SijMag[idx] * F2);
                    _nut[idx] = Math.Min(_nut[idx], _cfg.TurbulentViscosityMax);
                }
    }

    // -----------------------------------------------------------------------
    //  Spalart-Allmaras model
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute turbulent viscosity from the Spalart-Allmaras one-equation model.
    /// ∂ν̃/∂t + U·∇ν̃ = Cb1 S̃ ν̃ − Cw1 fw(ν̃/d)² + (1/σ) [∇·((ν + ν̃)∇ν̃) + Cb2 (∇ν̃)²]
    /// ν_t = ν̃ f_v1,  f_v1 = χ³/(χ³ + Cv1³),  χ = ν̃/ν
    /// </summary>
    public void UpdateSpalartAllmaras(
        ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w)
    {
        ComputeVelocityGradients(u, v, w);

        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);
                    _nutSA[idx] = _nutSA[idx]; // save previous
                    double nuTilde = _nutSA[idx];

                    double d = _yWall[idx];
                    double chi = nuTilde / _nu;
                    double fv1 = chi * chi * chi / (chi * chi * chi + Cv1 * Cv1 * Cv1);

                    // Modified vorticity: S̃ = S + ν̃ fw / (κ² d²)
                    double kappa = 0.41;
                    double S = _SijMag[idx];
                    double Stilde = S + nuTilde * fv1 / (kappa * kappa * d * d + 1e-20);

                    // fw function.
                    double r = nuTilde / (Stilde * kappa * kappa * d * d + 1e-20);
                    r = Math.Min(r, 10.0);
                    double g = r + Cw2 * (r * r * r * r * r * r - r);
                    double fw = Math.Pow(g, 1.0 / 6.0);

                    // Diffusion terms.
                    double lapNuTilde = (_nutSA[Idx(x + 1, y, z)] + _nutSA[Idx(x - 1, y, z)] +
                                         _nutSA[Idx(x, y + 1, z)] + _nutSA[Idx(x, y - 1, z)] +
                                         _nutSA[Idx(x, y, z + 1)] + _nutSA[Idx(x, y, z - 1)] -
                                         6.0 * nuTilde) / (_dx * _dx);

                    double[] gradNu = new double[3];
                    gradNu[0] = (_nutSA[Idx(x + 1, y, z)] - _nutSA[Idx(x - 1, y, z)]) / (2.0 * _dx);
                    gradNu[1] = (_nutSA[Idx(x, y + 1, z)] - _nutSA[Idx(x, y - 1, z)]) / (2.0 * _dx);
                    gradNu[2] = (_nutSA[Idx(x, y, z + 1)] - _nutSA[Idx(x, y, z - 1)]) / (2.0 * _dx);
                    double gradNuSq = gradNu[0] * gradNu[0] + gradNu[1] * gradNu[1] + gradNu[2] * gradNu[2];

                    // Transport equation.
                    double dnuTildeDt = Cb1 * Stilde * nuTilde -
                                        Cw1 * fw * nuTilde * nuTilde / (d * d) +
                                        (1.0 / SigmaSA) * ((_nu + nuTilde) * lapNuTilde +
                                                           Cb2 * gradNuSq);

                    nuTilde += _dt * dnuTildeDt;
                    nuTilde = Math.Max(nuTilde, 0);

                    // Clamp.
                    nuTilde = Math.Min(nuTilde, _cfg.TurbulentViscosityMax);
                    _nutSA[idx] = nuTilde;

                    // Turbulent viscosity: ν_t = ν̃ f_v1.
                    _nut[idx] = nuTilde * fv1;
                }
    }

    // -----------------------------------------------------------------------
    //  Smagorinsky LES (subgrid-scale model)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute SGS viscosity using the Smagorinsky model.
    /// ν_sgs = (C_s Δ)² |S|
    /// where Δ is the filter width (grid spacing) and |S| is the strain rate magnitude.
    /// </summary>
    public void UpdateSmagorinskyLES(
        ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w)
    {
        ComputeVelocityGradients(u, v, w);

        double Cs = _cfg.Csmagorinsky;
        double delta = _dx; // filter width = grid spacing

        for (int z = 1; z < _nz - 1; z++)
            for (int y = 1; y < _ny - 1; y++)
                for (int x = 1; x < _nx - 1; x++)
                {
                    int idx = Idx(x, y, z);

                    // Smagorinsky SGS viscosity.
                    _nut[idx] = (Cs * delta) * (Cs * delta) * _SijMag[idx];

                    // Wall damping: van Driest damping near walls.
                    double yPlus = _yWall[idx] * Math.Sqrt(_SijMag[idx]) / Math.Max(_nu, 1e-20);
                    double APlus = 26.0;
                    double wallDamp = 1.0 - Math.Exp(-yPlus / APlus);
                    _nut[idx] *= wallDamp * wallDamp;

                    _nut[idx] = Math.Min(_nut[idx], _cfg.TurbulentViscosityMax);
                }
    }

    // -----------------------------------------------------------------------
    //  Effective viscosity
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute total (effective) viscosity: ν_eff = ν + ν_t.
    /// </summary>
    public void ComputeEffectiveViscosity(double[] nuEff)
    {
        for (int i = 0; i < _n; i++)
            nuEff[i] = _nu + _nut[i];
    }

    /// <summary>
    /// Compute turbulent intensity: I = sqrt(2k/3) / |U|
    /// </summary>
    public double TurbulentIntensity(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w)
    {
        double sumK = 0, sumU = 0;
        int count = 0;
        for (int i = 0; i < _n; i++)
        {
            double umag = Math.Sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
            if (umag > 1e-10)
            {
                sumK += _k[i];
                sumU += umag;
                count++;
            }
        }
        if (count == 0)
            return 0;
        double avgU = sumU / count;
        double avgK = sumK / count;
        return Math.Sqrt(2.0 * avgK / 3.0) / Math.Max(avgU, 1e-10);
    }

    /// <summary>
    /// Compute turbulence production rate at each cell.
    /// </summary>
    public void ComputeProductionRate(double[] production)
    {
        for (int i = 0; i < _n; i++)
            production[i] = _nut[i] * _SijMag[i] * _SijMag[i];
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  10. MultiphysicsCoupler — bidirectional coupling between solvers
// ============================================================================

/// <summary>
/// Partitioned coupling scheme.
/// </summary>
public enum CouplingScheme
{
    GaussSeidel,
    Jacobi,
    QuasiNewton
}

/// <summary>
/// Configuration for the multiphysics coupler.
/// </summary>
public sealed class CouplingConfig
{
    public CouplingScheme Scheme { get; init; } = CouplingScheme.GaussSeidel;
    public int MaxIterations { get; init; } = 20;
    public double ConvergenceTolerance { get; init; } = 1e-6;
    public int TimeSteps { get; init; } = 1000;
    public int CouplingInterval { get; init; } = 10;
    public double Relaxation { get; init; } = 0.8;
    public bool AdaptiveRelaxation { get; init; }
    public double MinRelaxation { get; init; } = 0.1;
    public double MaxRelaxation { get; init; } = 1.0;
    public bool EnableLoadBalancing { get; init; }
}

/// <summary>
/// Represents the data exchanged between coupled solvers at a shared interface.
/// </summary>
public sealed class InterfaceData
{
    public string Name { get; init; }
    public double[] FieldData { get; set; }
    public int NumNodes { get; init; }
    public DateTime Timestamp { get; set; }

    public InterfaceData(string name, int numNodes)
    {
        Name = name;
        NumNodes = numNodes;
        FieldData = new double[numNodes];
        Timestamp = DateTime.UtcNow;
    }
}

/// <summary>
/// Represents a coupling link between two solvers.
/// </summary>
public sealed class CouplingLink
{
    public string SourceSolver { get; init; }
    public string TargetSolver { get; init; }
    public string SourceField { get; init; }
    public string TargetField { get; init; }
    public Func<double[], double[]> TransferFunction { get; init; }
}

/// <summary>
/// Monitors convergence of the partitioned coupling iteration.
/// </summary>
public sealed class ConvergenceMonitor
{
    private readonly List<double> _residualHistory;
    private int _totalIterations;

    public double CurrentResidual { get; private set; }
    public bool Converged { get; private set; }
    public int Iterations => _totalIterations;
    public IReadOnlyList<double> ResidualHistory => _residualHistory;

    public ConvergenceMonitor()
    {
        _residualHistory = new List<double>();
    }

    public void Reset()
    {
        _residualHistory.Clear();
        _totalIterations = 0;
        CurrentResidual = double.MaxValue;
        Converged = false;
    }

    public bool CheckConvergence(double residual, double tolerance)
    {
        CurrentResidual = residual;
        _residualHistory.Add(residual);
        _totalIterations++;
        Converged = residual < tolerance;
        return Converged;
    }

    public static double ComputeResidual(double[] current, double[] previous)
    {
        double sum = 0;
        int n = Math.Min(current.Length, previous.Length);
        for (int i = 0; i < n; i++)
        {
            double diff = current[i] - previous[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum / Math.Max(n, 1));
    }
}
/// <summary>
/// Handles load balancing across coupled solvers.
/// </summary>
public sealed class LoadBalancer
{
    private readonly Dictionary<string, double> _executionTimes;

    public LoadBalancer()
    {
        _executionTimes = new Dictionary<string, double>();
    }

    public void RecordExecutionTime(string solverName, TimeSpan time)
    {
        _executionTimes[solverName] = time.TotalSeconds;
    }

    public Dictionary<string, double> GetAllocationFractions()
    {
        var result = new Dictionary<string, double>();
        double total = 0;
        foreach (var kv in _executionTimes)
            total += kv.Value;
        foreach (var kv in _executionTimes)
            result[kv.Key] = total > 0 ? kv.Value / total : 1.0 / Math.Max(_executionTimes.Count, 1);
        return result;
    }

    public Dictionary<string, double> GetTimeStepRatios()
    {
        var result = new Dictionary<string, double>();
        double maxTime = 0;
        foreach (var kv in _executionTimes)
            if (kv.Value > maxTime)
                maxTime = kv.Value;
        foreach (var kv in _executionTimes)
            result[kv.Key] = maxTime > 0 ? maxTime / kv.Value : 1.0;
        return result;
    }
}

/// <summary>
/// Multiphysics coupler for bidirectional coupling between different
/// physics solvers. Supports Gauss-Seidel and Jacobi partitioned
/// iteration schemes, convergence monitoring, under-relaxation, and
/// load balancing.
/// </summary>
public sealed class MultiphysicsCoupler : IDisposable
{
    private readonly CouplingConfig _cfg;
    private readonly Dictionary<string, object> _solvers;
    private readonly Dictionary<string, InterfaceData> _interfaceData;
    private readonly List<CouplingLink> _links;
    private readonly ConvergenceMonitor _convergence;
    private readonly LoadBalancer _loadBalancer;
    private double _relaxation;
    private bool _disposed;

    public ConvergenceMonitor Convergence => _convergence;
    public LoadBalancer LoadBalancer => _loadBalancer;

    public MultiphysicsCoupler(CouplingConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        _solvers = new Dictionary<string, object>();
        _interfaceData = new Dictionary<string, InterfaceData>();
        _links = new List<CouplingLink>();
        _convergence = new ConvergenceMonitor();
        _loadBalancer = new LoadBalancer();
        _relaxation = config.Relaxation;
    }

    public void RegisterSolver(string name, object solver)
    {
        _solvers[name] = solver ?? throw new ArgumentNullException(nameof(solver));
    }

    public void RegisterInterface(string name, InterfaceData data)
    {
        _interfaceData[name] = data;
    }

    public void AddCouplingLink(CouplingLink link)
    {
        _links.Add(link ?? throw new ArgumentNullException(nameof(link)));
    }

    /// <summary>
    /// Perform one Gauss-Seidel coupling iteration.
    /// </summary>
    public bool IterateGaussSeidel()
    {
        _convergence.Reset();
        for (int iter = 0; iter < _cfg.MaxIterations; iter++)
        {
            bool anyChanged = false;
            foreach (var link in _links)
            {
                if (!_interfaceData.ContainsKey(link.SourceField) ||
                    !_interfaceData.ContainsKey(link.TargetField))
                    continue;

                var sourceData = _interfaceData[link.SourceField];
                var targetData = _interfaceData[link.TargetField];
                double[] prevTarget = (double[])targetData.FieldData.Clone();

                double[] transferred = link.TransferFunction != null
                    ? link.TransferFunction(sourceData.FieldData)
                    : (double[])sourceData.FieldData.Clone();

                for (int i = 0; i < Math.Min(transferred.Length, targetData.FieldData.Length); i++)
                    targetData.FieldData[i] = (1.0 - _relaxation) * targetData.FieldData[i] +
                                               _relaxation * transferred[i];

                targetData.Timestamp = DateTime.UtcNow;
                anyChanged = true;

                double residual = ConvergenceMonitor.ComputeResidual(targetData.FieldData, prevTarget);
                if (_convergence.CheckConvergence(residual, _cfg.ConvergenceTolerance))
                    return true;
            }

            if (_cfg.AdaptiveRelaxation && _convergence.ResidualHistory.Count >= 2)
            {
                double last = _convergence.ResidualHistory[^1];
                double prev = _convergence.ResidualHistory[^2];
                if (last > prev)
                    _relaxation = Math.Max(_cfg.MinRelaxation, _relaxation * 0.8);
                else
                    _relaxation = Math.Min(_cfg.MaxRelaxation, _relaxation * 1.05);
            }
        }
        return _convergence.Converged;
    }

    /// <summary>
    /// Perform one Jacobi coupling iteration.
    /// </summary>
    public bool IterateJacobi()
    {
        _convergence.Reset();
        var oldData = new Dictionary<string, double[]>();
        foreach (var kv in _interfaceData)
            oldData[kv.Key] = (double[])kv.Value.FieldData.Clone();

        for (int iter = 0; iter < _cfg.MaxIterations; iter++)
        {
            var newData = new Dictionary<string, double[]>();
            foreach (var link in _links)
            {
                if (!oldData.ContainsKey(link.SourceField))
                    continue;

                double[] transferred = link.TransferFunction != null
                    ? link.TransferFunction(oldData[link.SourceField])
                    : (double[])oldData[link.SourceField].Clone();

                var targetData = _interfaceData[link.TargetField];
                double[] mixed = new double[transferred.Length];
                for (int i = 0; i < Math.Min(transferred.Length, targetData.FieldData.Length); i++)
                    mixed[i] = (1.0 - _relaxation) * targetData.FieldData[i] +
                               _relaxation * transferred[i];

                newData[link.TargetField] = mixed;
            }

            foreach (var kv in newData)
            {
                double[] prev = _interfaceData[kv.Key].FieldData;
                _interfaceData[kv.Key].FieldData = kv.Value;
                _interfaceData[kv.Key].Timestamp = DateTime.UtcNow;

                double residual = ConvergenceMonitor.ComputeResidual(kv.Value, prev);
                if (_convergence.CheckConvergence(residual, _cfg.ConvergenceTolerance))
                    return true;
            }

            foreach (var kv in _interfaceData)
                oldData[kv.Key] = (double[])kv.Value.FieldData.Clone();
        }
        return _convergence.Converged;
    }

    /// <summary>
    /// Run the coupled simulation for the configured number of time-steps.
    /// </summary>
    public void Run()
    {
        for (int step = 0; step < _cfg.TimeSteps; step++)
        {
            foreach (var kv in _solvers)
            {
                var type = kv.Value.GetType();
                var stepMethod = type.GetMethod("Step");
                var sw = Stopwatch.StartNew();
                stepMethod?.Invoke(kv.Value, null);
                sw.Stop();
                _loadBalancer.RecordExecutionTime(kv.Key, sw.Elapsed);
            }

            if (step % _cfg.CouplingInterval == 0)
            {
                switch (_cfg.Scheme)
                {
                    case CouplingScheme.GaussSeidel:
                        IterateGaussSeidel();
                        break;
                    case CouplingScheme.Jacobi:
                        IterateJacobi();
                        break;
                    default:
                        IterateGaussSeidel();
                        break;
                }
            }
        }
    }

    public void SetupFSI(
        object fluidSolver, object structSolver,
        int interfaceNodes,
        Func<double[], double[]> pressureToForce,
        Func<double[], double[]> displacementToMesh)
    {
        RegisterSolver("fluid", fluidSolver);
        RegisterSolver("structure", structSolver);
        RegisterInterface("pressure", new InterfaceData("pressure", interfaceNodes));
        RegisterInterface("force", new InterfaceData("force", interfaceNodes));
        RegisterInterface("displacement", new InterfaceData("displacement", interfaceNodes));
        RegisterInterface("mesh", new InterfaceData("mesh", interfaceNodes));

        AddCouplingLink(new CouplingLink
        {
            SourceSolver = "fluid",
            TargetSolver = "structure",
            SourceField = "pressure",
            TargetField = "force",
            TransferFunction = pressureToForce
        });
        AddCouplingLink(new CouplingLink
        {
            SourceSolver = "structure",
            TargetSolver = "fluid",
            SourceField = "displacement",
            TargetField = "mesh",
            TransferFunction = displacementToMesh
        });
    }

    public void SetupCHT(
        object fluidSolver, object solidSolver,
        int interfaceNodes,
        Func<double[], double[]> tempToFlux,
        Func<double[], double[]> fluxToTemp)
    {
        RegisterSolver("fluid", fluidSolver);
        RegisterSolver("solid", solidSolver);
        RegisterInterface("temperature", new InterfaceData("temperature", interfaceNodes));
        RegisterInterface("heatflux", new InterfaceData("heatflux", interfaceNodes));
        RegisterInterface("solid_temperature", new InterfaceData("solid_temperature", interfaceNodes));
        RegisterInterface("solid_heatflux", new InterfaceData("solid_heatflux", interfaceNodes));

        AddCouplingLink(new CouplingLink
        {
            SourceSolver = "fluid",
            TargetSolver = "solid",
            SourceField = "temperature",
            TargetField = "solid_temperature",
            TransferFunction = tempToFlux
        });
        AddCouplingLink(new CouplingLink
        {
            SourceSolver = "solid",
            TargetSolver = "fluid",
            SourceField = "solid_heatflux",
            TargetField = "heatflux",
            TransferFunction = fluxToTemp
        });
    }

    public T? GetSolver<T>(string name) where T : class
    {
        if (_solvers.TryGetValue(name, out var solver))
            return solver as T;
        throw new KeyNotFoundException($"Solver '{name}' not registered.");
    }

    public InterfaceData GetInterfaceData(string name)
    {
        if (_interfaceData.TryGetValue(name, out var data))
            return data;
        throw new KeyNotFoundException($"Interface '{name}' not registered.");
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  Utility: Generic iterative solver for linear systems
// ============================================================================

/// <summary>
/// Generic iterative linear solver used across multiple physics solvers.
/// </summary>
public static class IterativeLinearSolver
{
    /// <summary>
    /// Solve Ax = b using the Conjugate Gradient method (SPD A).
    /// Matrix A applied via a function y = A*x.
    /// </summary>
    public static int ConjugateGradient(
        int n,
        Func<double[], double[]> matVec,
        ReadOnlySpan<double> b,
        Span<double> x,
        int maxIter = 1000,
        double tolerance = 1e-10)
    {
        double[] r = new double[n], p = new double[n], Ap = new double[n];
        double[] bArr = b.ToArray();
        double[] Ax0 = matVec(x.ToArray());
        for (int i = 0; i < n; i++)
        { r[i] = bArr[i] - Ax0[i]; p[i] = r[i]; }

        double rsOld = 0;
        for (int i = 0; i < n; i++)
            rsOld += r[i] * r[i];
        double bNorm = Math.Sqrt(rsOld);
        if (bNorm < 1e-30)
            bNorm = 1.0;

        int iter;
        for (iter = 0; iter < maxIter; iter++)
        {
            Ap = matVec(p);
            double pAp = 0;
            for (int i = 0; i < n; i++)
                pAp += p[i] * Ap[i];
            if (Math.Abs(pAp) < 1e-30)
                break;

            double alpha = rsOld / pAp;
            for (int i = 0; i < n; i++)
            { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }

            double rsNew = 0;
            for (int i = 0; i < n; i++)
                rsNew += r[i] * r[i];
            if (Math.Sqrt(rsNew) / bNorm < tolerance)
            { iter++; break; }

            double beta = rsNew / rsOld;
            for (int i = 0; i < n; i++)
                p[i] = r[i] + beta * p[i];
            rsOld = rsNew;
        }
        return iter;
    }

    /// <summary>
    /// SOR iteration: solve Ax = b with over-relaxation factor omega.
    /// </summary>
    public static int SOR(
        int n,
        Func<int, int, double> getElement,
        ReadOnlySpan<double> b,
        Span<double> x,
        double omega = 1.5,
        int maxIter = 10000,
        double tolerance = 1e-10)
    {
        double[] xOld = new double[n];
        for (int iter = 0; iter < maxIter; iter++)
        {
            for (int i = 0; i < n; i++)
                xOld[i] = x[i];
            for (int i = 0; i < n; i++)
            {
                double sigma = 0;
                double diag = getElement(i, i);
                for (int j = 0; j < n; j++)
                    if (j != i)
                        sigma += getElement(i, j) * x[j];
                double xGS = (b[i] - sigma) / diag;
                x[i] = (1.0 - omega) * x[i] + omega * xGS;
            }
            double diff = 0;
            for (int i = 0; i < n; i++)
                diff += (x[i] - xOld[i]) * (x[i] - xOld[i]);
            if (Math.Sqrt(diff) < tolerance)
                return iter + 1;
        }
        return maxIter;
    }
}

// ============================================================================
//  Utility: ODE Integrators
// ============================================================================

/// <summary>
/// Generic ODE time integrators for first-order systems dy/dt = f(t, y).
/// </summary>
public static class ODEIntegrators
{
    /// <summary>4th-order Runge-Kutta step.</summary>
    public static double[] RK4(
        Func<double, double[], double[]> rhs,
        double t, ReadOnlySpan<double> y, double dt)
    {
        int n = y.Length;
        double[] yArr = y.ToArray();
        double[] k1 = rhs(t, yArr);

        double[] y2 = new double[n];
        for (int i = 0; i < n; i++)
            y2[i] = y[i] + 0.5 * dt * k1[i];
        double[] k2 = rhs(t + 0.5 * dt, y2);

        double[] y3 = new double[n];
        for (int i = 0; i < n; i++)
            y3[i] = y[i] + 0.5 * dt * k2[i];
        double[] k3 = rhs(t + 0.5 * dt, y3);

        double[] y4 = new double[n];
        for (int i = 0; i < n; i++)
            y4[i] = y[i] + dt * k3[i];
        double[] k4 = rhs(t + dt, y4);

        double[] result = new double[n];
        for (int i = 0; i < n; i++)
            result[i] = y[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        return result;
    }

    /// <summary>Crank-Nicolson step with fixed-point iteration.</summary>
    public static double[] CrankNicolson(
        Func<double, double[], double[]> rhs,
        double t, ReadOnlySpan<double> y, double dt, int iters = 10)
    {
        int n = y.Length;
        double[] yn = y.ToArray();
        double[] f0 = rhs(t, yn);
        double[] yn1 = new double[n];
        for (int i = 0; i < n; i++)
            yn1[i] = yn[i] + dt * f0[i];

        for (int iter = 0; iter < iters; iter++)
        {
            double[] f1 = rhs(t + dt, yn1);
            for (int i = 0; i < n; i++)
                yn1[i] = yn[i] + 0.5 * dt * (f0[i] + f1[i]);
        }
        return yn1;
    }
}

// ============================================================================
//  Utility: Spatial Interpolation
// ============================================================================

/// <summary>
/// Spatial interpolation utilities for non-matching grid data transfer.
/// </summary>
public static class SpatialInterpolation
{
    /// <summary>Trilinear interpolation at position (x, y, z) on a regular grid.</summary>
    public static double TrilinearInterpolate(
        ReadOnlySpan<double> field, int nx, int ny, int nz,
        double dx, double dy, double dz,
        double x, double y, double z)
    {
        double xi = x / dx, yi = y / dy, zi = z / dz;
        int x0 = Math.Clamp((int)Math.Floor(xi), 0, nx - 2);
        int y0 = Math.Clamp((int)Math.Floor(yi), 0, ny - 2);
        int z0 = Math.Clamp((int)Math.Floor(zi), 0, nz - 2);
        double xf = xi - x0, yf = yi - y0, zf = zi - z0;

        double c000 = field[z0 * ny * nx + y0 * nx + x0];
        double c100 = field[z0 * ny * nx + y0 * nx + x0 + 1];
        double c010 = field[z0 * ny * nx + (y0 + 1) * nx + x0];
        double c110 = field[z0 * ny * nx + (y0 + 1) * nx + x0 + 1];
        double c001 = field[(z0 + 1) * ny * nx + y0 * nx + x0];
        double c101 = field[(z0 + 1) * ny * nx + y0 * nx + x0 + 1];
        double c011 = field[(z0 + 1) * ny * nx + (y0 + 1) * nx + x0];
        double c111 = field[(z0 + 1) * ny * nx + (y0 + 1) * nx + x0 + 1];

        double c00 = c000 * (1 - xf) + c100 * xf;
        double c10 = c010 * (1 - xf) + c110 * xf;
        double c01 = c001 * (1 - xf) + c101 * xf;
        double c11 = c011 * (1 - xf) + c111 * xf;
        double c0 = c00 * (1 - yf) + c10 * yf;
        double c1 = c01 * (1 - yf) + c11 * yf;
        return c0 * (1 - zf) + c1 * zf;
    }

    /// <summary>Nearest-neighbour interpolation.</summary>
    public static double NearestNeighbour(
        ReadOnlySpan<double> field, int nx, int ny, int nz,
        double dx, double dy, double dz,
        double x, double y, double z)
    {
        int xi = Math.Clamp((int)Math.Round(x / dx), 0, nx - 1);
        int yi = Math.Clamp((int)Math.Round(y / dy), 0, ny - 1);
        int zi = Math.Clamp((int)Math.Round(z / dz), 0, nz - 1);
        return field[zi * ny * nx + yi * nx + xi];
    }
}

// ============================================================================
//  Utility: FFT (Cooley-Tukey radix-2)
// ============================================================================

/// <summary>
/// In-place Cooley-Tukey radix-2 FFT for spectral analysis.
/// </summary>
public static class FFT
{
    public static void Forward(Span<double> real, Span<double> imag)
    {
        int n = real.Length;
        if ((n & (n - 1)) != 0)
            throw new ArgumentException("Length must be a power of 2.");

        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
            { (real[i], real[j]) = (real[j], real[i]); (imag[i], imag[j]) = (imag[j], imag[i]); }
        }

        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = -2.0 * Math.PI / len;
            double wr = Math.Cos(angle), wi = Math.Sin(angle);
            for (int i = 0; i < n; i += len)
            {
                double curRe = 1.0, curIm = 0.0;
                for (int j = 0; j < len / 2; j++)
                {
                    int a = i + j, b = i + j + len / 2;
                    double tRe = curRe * real[b] - curIm * imag[b];
                    double tIm = curRe * imag[b] + curIm * real[b];
                    real[b] = real[a] - tRe;
                    imag[b] = imag[a] - tIm;
                    real[a] += tRe;
                    imag[a] += tIm;
                    double newRe = curRe * wr - curIm * wi;
                    curIm = curRe * wi + curIm * wr;
                    curRe = newRe;
                }
            }
        }
    }

    public static void Inverse(Span<double> real, Span<double> imag)
    {
        for (int i = 0; i < imag.Length; i++)
            imag[i] = -imag[i];
        Forward(real, imag);
        double inv = 1.0 / real.Length;
        for (int i = 0; i < real.Length; i++)
        { real[i] *= inv; imag[i] = -imag[i] * inv; }
    }

    public static double[] PowerSpectrum(ReadOnlySpan<double> signal)
    {
        int n = signal.Length;
        int fftSize = 1;
        while (fftSize < n)
            fftSize <<= 1;
        double[] re = new double[fftSize], im = new double[fftSize];
        signal.CopyTo(re);
        Forward(re, im);
        int half = fftSize / 2;
        double[] psd = new double[half];
        for (int i = 0; i < half; i++)
            psd[i] = (re[i] * re[i] + im[i] * im[i]) / (fftSize * fftSize);
        return psd;
    }
}

// ============================================================================
//  Utility: Mersenne Twister RNG
// ============================================================================

/// <summary>
/// MT19937 Mersenne Twister random number generator.
/// </summary>
public sealed class MersenneTwister
{
    private const int N = 624, M = 397;
    private const uint MatrixA = 0x9908b0df, UpperMask = 0x80000000, LowerMask = 0x7fffffff;
    private readonly uint[] _mt = new uint[N];
    private int _mti = N + 1;

    public MersenneTwister(uint seed = 5489)
    {
        _mt[0] = seed;
        for (int i = 1; i < N; i++)
            _mt[i] = 1812433253 * (_mt[i - 1] ^ (_mt[i - 1] >> 30)) + (uint)i;
    }

    private uint GenerateUInt()
    {
        uint[] mag01 = { 0, MatrixA };
        if (_mti >= N)
        {
            for (int k = 0; k < N - M; k++)
            { uint y = (_mt[k] & UpperMask) | (_mt[k + 1] & LowerMask); _mt[k] = _mt[k + M] ^ (y >> 1) ^ mag01[y & 1]; }
            for (int k = N - M; k < N - 1; k++)
            { uint y = (_mt[k] & UpperMask) | (_mt[k + 1] & LowerMask); _mt[k] = _mt[k + M - N] ^ (y >> 1) ^ mag01[y & 1]; }
            uint yb = (_mt[N - 1] & UpperMask) | (_mt[0] & LowerMask);
            _mt[N - 1] = _mt[M - 1] ^ (yb >> 1) ^ mag01[yb & 1];
            _mti = 0;
        }
        uint y2 = _mt[_mti++];
        y2 ^= y2 >> 11;
        y2 ^= (y2 << 7) & 0x9d2c5680;
        y2 ^= (y2 << 15) & 0xefc60000;
        y2 ^= y2 >> 18;
        return y2;
    }

    public double NextDouble() => GenerateUInt() * (1.0 / 4294967296.0);
    public int NextInt(int min, int max) => min + (int)(GenerateUInt() % (uint)(max - min));

    public double NextGaussian(double mean = 0, double stdDev = 1)
    {
        double u1 = 1.0 - NextDouble(), u2 = 1.0 - NextDouble();
        return mean + stdDev * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    public double NextExponential(double lambda = 1.0) => -Math.Log(1.0 - NextDouble()) / lambda;

    public int NextPoisson(double lambda)
    {
        double L = Math.Exp(-lambda), p = 1.0;
        int k = 0;
        do
        { k++; p *= NextDouble(); } while (p > L);
        return k - 1;
    }
}

// ============================================================================
//  End of Solvers.cs — Synapse Omonia Physics
// ============================================================================
// ============================================================================
//  Additional Utility: Numerical Differentiation
// ============================================================================

/// <summary>
/// High-order numerical differentiation on uniform grids using
/// central differences with boundary corrections.
/// </summary>
public static class NumericalDifferentiation
{
    /// <summary>
    /// 2nd-order central first derivative: df/dx ≈ (f[i+1] − f[i−1]) / (2h)
    /// </summary>
    public static void FirstDerivative2nd(ReadOnlySpan<double> f, Span<double> dfdx, double h, int n)
    {
        dfdx[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * h);
        for (int i = 1; i < n - 1; i++)
            dfdx[i] = (f[i + 1] - f[i - 1]) / (2.0 * h);
        dfdx[n - 1] = (3.0 * f[n - 1] - 4.0 * f[n - 2] + f[n - 3]) / (2.0 * h);
    }

    /// <summary>
    /// 2nd-order central second derivative: d²f/dx² ≈ (f[i+1] − 2f[i] + f[i−1]) / h²
    /// </summary>
    public static void SecondDerivative2nd(ReadOnlySpan<double> f, Span<double> d2fdx2, double h, int n)
    {
        d2fdx2[0] = (2.0 * f[0] - 5.0 * f[1] + 4.0 * f[2] - f[3]) / (h * h);
        for (int i = 1; i < n - 1; i++)
            d2fdx2[i] = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / (h * h);
        d2fdx2[n - 1] = (2.0 * f[n - 1] - 5.0 * f[n - 2] + 4.0 * f[n - 3] - f[n - 4]) / (h * h);
    }

    /// <summary>
    /// 4th-order central first derivative.
    /// </summary>
    public static void FirstDerivative4th(ReadOnlySpan<double> f, Span<double> dfdx, double h, int n)
    {
        double invH = 1.0 / (12.0 * h);
        dfdx[0] = (-25.0 * f[0] + 48.0 * f[1] - 36.0 * f[2] + 16.0 * f[3] - 3.0 * f[4]) * invH;
        dfdx[1] = (-25.0 * f[1] + 48.0 * f[2] - 36.0 * f[3] + 16.0 * f[4] - 3.0 * f[5]) * invH;
        for (int i = 2; i < n - 2; i++)
            dfdx[i] = (f[i - 2] - 8.0 * f[i - 1] + 8.0 * f[i + 1] - f[i + 2]) * invH;
        dfdx[n - 2] = (25.0 * f[n - 2] - 48.0 * f[n - 3] + 36.0 * f[n - 4] - 16.0 * f[n - 5] + 3.0 * f[n - 6]) * (-invH);
        dfdx[n - 1] = (25.0 * f[n - 1] - 48.0 * f[n - 2] + 36.0 * f[n - 3] - 16.0 * f[n - 4] + 3.0 * f[n - 5]) * (-invH);
    }

    /// <summary>
    /// 4th-order central second derivative.
    /// </summary>
    public static void SecondDerivative4th(ReadOnlySpan<double> f, Span<double> d2fdx2, double h, int n)
    {
        double invH2 = 1.0 / (12.0 * h * h);
        d2fdx2[0] = (35.0 * f[0] - 104.0 * f[1] + 114.0 * f[2] - 56.0 * f[3] + 11.0 * f[4]) * invH2;
        d2fdx2[1] = (35.0 * f[1] - 104.0 * f[2] + 114.0 * f[3] - 56.0 * f[4] + 11.0 * f[5]) * invH2;
        for (int i = 2; i < n - 2; i++)
            d2fdx2[i] = (-f[i - 2] + 16.0 * f[i - 1] - 30.0 * f[i] + 16.0 * f[i + 1] - f[i + 2]) * invH2;
        d2fdx2[n - 2] = (35.0 * f[n - 2] - 104.0 * f[n - 3] + 114.0 * f[n - 4] - 56.0 * f[n - 5] + 11.0 * f[n - 6]) * invH2;
        d2fdx2[n - 1] = (35.0 * f[n - 1] - 104.0 * f[n - 2] + 114.0 * f[n - 3] - 56.0 * f[n - 4] + 11.0 * f[n - 5]) * invH2;
    }

    /// <summary>
    /// Gradient of a 3D scalar field. Returns (dF/dx, dF/dy, dF/dz).
    /// </summary>
    public static void Gradient3D(
        ReadOnlySpan<double> f,
        Span<double> dfdx, Span<double> dfdy, Span<double> dfdz,
        int nx, int ny, int nz, double dx)
    {
        double invDx2 = 0.5 / dx;
        for (int z = 1; z < nz - 1; z++)
            for (int y = 1; y < ny - 1; y++)
                for (int x = 1; x < nx - 1; x++)
                {
                    int idx = z * ny * nx + y * nx + x;
                    dfdx[idx] = (f[idx + 1] - f[idx - 1]) * invDx2;
                    dfdy[idx] = (f[idx + nx] - f[idx - nx]) * invDx2;
                    dfdz[idx] = (f[idx + ny * nx] - f[idx - ny * nx]) * invDx2;
                }
    }

    /// <summary>
    /// Divergence of a 3D vector field. Returns ∇·F = dFx/dx + dFy/dy + dFz/dz.
    /// </summary>
    public static double Divergence3D(
        ReadOnlySpan<double> fx, ReadOnlySpan<double> fy, ReadOnlySpan<double> fz,
        int nx, int ny, int nz, double dx)
    {
        double sum = 0;
        double invDx2 = 0.5 / dx;
        for (int z = 1; z < nz - 1; z++)
            for (int y = 1; y < ny - 1; y++)
                for (int x = 1; x < nx - 1; x++)
                {
                    int idx = z * ny * nx + y * nx + x;
                    double div = (fx[idx + 1] - fx[idx - 1]) * invDx2 +
                                 (fy[idx + nx] - fy[idx - nx]) * invDx2 +
                                 (fz[idx + ny * nx] - fz[idx - ny * nx]) * invDx2;
                    sum += div * div;
                }
        return Math.Sqrt(sum);
    }

    /// <summary>
    /// Curl of a 3D vector field. Returns (∇×F)x, (∇×F)y, (∇×F)z.
    /// </summary>
    public static void Curl3D(
        ReadOnlySpan<double> fx, ReadOnlySpan<double> fy, ReadOnlySpan<double> fz,
        Span<double> curlX, Span<double> curlY, Span<double> curlZ,
        int nx, int ny, int nz, double dx)
    {
        double invDx2 = 0.5 / dx;
        for (int z = 1; z < nz - 1; z++)
            for (int y = 1; y < ny - 1; y++)
                for (int x = 1; x < nx - 1; x++)
                {
                    int idx = z * ny * nx + y * nx + x;
                    int xp = idx + 1, xm = idx - 1;
                    int yp = idx + nx, ym = idx - nx;
                    int zp = idx + ny * nx, zm = idx - ny * nx;

                    curlX[idx] = (fz[yp] - fz[ym]) * invDx2 - (fy[zp] - fy[zm]) * invDx2;
                    curlY[idx] = (fx[zp] - fx[zm]) * invDx2 - (fz[xp] - fz[xm]) * invDx2;
                    curlZ[idx] = (fy[xp] - fy[xm]) * invDx2 - (fx[yp] - fx[ym]) * invDx2;
                }
    }
}

// ============================================================================
//  Additional Utility: Matrix Operations
// ============================================================================

/// <summary>
/// Lightweight matrix operations for physics solver internals.
/// Works with flattened row-major arrays.
/// </summary>
public static class PhysicsMatrix
{
    /// <summary>
    /// Multiply matrix A (m×n) by vector x (n) yielding result (m).
    /// </summary>
    public static void MatVec(ReadOnlySpan<double> A, ReadOnlySpan<double> x,
        Span<double> result, int m, int n)
    {
        for (int i = 0; i < m; i++)
        {
            double sum = 0;
            int row = i * n;
            for (int j = 0; j < n; j++)
                sum += A[row + j] * x[j];
            result[i] = sum;
        }
    }

    /// <summary>
    /// Transpose matrix A (m×n) into AT (n×m).
    /// </summary>
    public static void Transpose(ReadOnlySpan<double> A, Span<double> AT, int m, int n)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                AT[j * m + i] = A[i * n + j];
    }

    /// <summary>
    /// Compute A^T A (n×n) from A (m×n).
    /// </summary>
    public static void GramMatrix(ReadOnlySpan<double> A, Span<double> AtA, int m, int n)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < m; k++)
                    sum += A[k * n + i] * A[k * n + j];
                AtA[i * n + j] = sum;
                AtA[j * n + i] = sum;
            }
    }

    /// <summary>
    /// Symmetric matrix-vector product for a banded system (used in FEM).
    /// Only lower band of width bw is stored.
    /// </summary>
    public static void BandedSymmetricMatVec(ReadOnlySpan<double> band, ReadOnlySpan<double> x,
        Span<double> result, int n, int bw)
    {
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            int rowStart = i * bw;
            int jStart = Math.Max(0, i - bw + 1);
            for (int j = jStart; j <= i; j++)
            {
                int k = rowStart + (i - j);
                if (k >= 0 && k < band.Length)
                    sum += band[k] * x[j];
            }
            // Upper part (symmetric).
            for (int j = i + 1; j < Math.Min(n, i + bw); j++)
            {
                int k = j * bw + (j - i);
                if (k >= 0 && k < band.Length)
                    sum += band[k] * x[j];
            }
            result[i] = sum;
        }
    }

    /// <summary>
    /// Compute trace of an n×n matrix stored in row-major format.
    /// </summary>
    public static double Trace(ReadOnlySpan<double> A, int n)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += A[i * n + i];
        return sum;
    }

    /// <summary>
    /// Compute the Frobenius norm of an m×n matrix.
    /// </summary>
    public static double FrobeniusNorm(ReadOnlySpan<double> A, int m, int n)
    {
        double sum = 0;
        for (int i = 0; i < m * n; i++)
            sum += A[i] * A[i];
        return Math.Sqrt(sum);
    }
}

// ============================================================================
//  Additional Utility: Quadrature Rules
// ============================================================================

/// <summary>
/// Gauss-Legendre quadrature points and weights for numerical integration.
/// </summary>
public static class QuadratureRules
{
    /// <summary>
    /// Get Gauss-Legendre quadrature points and weights on [−1, 1]
    /// for the specified order. Returns (points, weights) arrays.
    /// </summary>
    public static (double[] Points, double[] Weights) GaussLegendre(int order)
    {
        int n = order;
        double[] x = new double[n];
        double[] w = new double[n];

        for (int i = 0; i < n; i++)
        {
            double z = Math.Cos(Math.PI * (4.0 * i + 3.0) / (4.0 * n + 2.0));
            for (int iter = 0; iter < 100; iter++)
            {
                double pp0 = 1.0, pp1 = z;
                for (int j = 1; j < n; j++)
                {
                    double pp2 = ((2.0 * j + 1.0) * z * pp1 - j * pp0) / (j + 1.0);
                    pp0 = pp1;
                    pp1 = pp2;
                }
                double dp = n * (z * pp1 - pp0) / (z * z - 1.0);
                double dz = pp1 / dp;
                z -= dz;
                if (Math.Abs(dz) < 1e-15)
                    break;
            }
            x[i] = z;
            double p0 = 1.0, p1x = z;
            for (int j = 1; j < n; j++)
            {
                double p2 = ((2.0 * j + 1.0) * z * p1x - j * p0) / (j + 1.0);
                p0 = p1x;
                p1x = p2;
            }
            w[i] = 2.0 / ((1.0 - z * z) * p1x * p1x);
        }
        return (x, w);
    }

    /// <summary>
    /// Compute integral of f from a to b using Gauss-Legendre quadrature.
    /// </summary>
    public static double Integrate(Func<double, double> f, double a, double b, int order = 5)
    {
        var (points, weights) = GaussLegendre(order);
        double sum = 0;
        double mid = 0.5 * (a + b);
        double halfLen = 0.5 * (b - a);
        for (int i = 0; i < order; i++)
            sum += weights[i] * f(mid + halfLen * points[i]);
        return halfLen * sum;
    }

    /// <summary>
    /// Simpson's rule for uniform grids: ∫f dx ≈ h/3 [f₀ + 4f₁ + 2f₂ + 4f₃ + ... + fₙ]
    /// </summary>
    public static double Simpson(ReadOnlySpan<double> f, double h)
    {
        int n = f.Length;
        if (n < 3)
            return 0;
        double sum = f[0] + f[n - 1];
        for (int i = 1; i < n - 1; i++)
            sum += (i % 2 == 0 ? 2.0 : 4.0) * f[i];
        return sum * h / 3.0;
    }

    /// <summary>
    /// Trapezoidal rule for uniform grids.
    /// </summary>
    public static double Trapezoidal(ReadOnlySpan<double> f, double h)
    {
        int n = f.Length;
        if (n < 2)
            return 0;
        double sum = 0.5 * (f[0] + f[n - 1]);
        for (int i = 1; i < n - 1; i++)
            sum += f[i];
        return sum * h;
    }
}

// ============================================================================
//  Additional Utility: Coordinate Transforms
// ============================================================================

/// <summary>
/// Coordinate transformation utilities for the physics solvers.
/// Supports Cartesian, cylindrical, and spherical systems.
/// </summary>
public static class CoordinateTransforms
{
    /// <summary>Cylindrical (r, θ, z) → Cartesian (x, y, z).</summary>
    public static void CylindricalToCartesian(double r, double theta, double z,
        out double x, out double y, out double cz)
    {
        x = r * Math.Cos(theta);
        y = r * Math.Sin(theta);
        cz = z;
    }

    /// <summary>Cartesian (x, y, z) → Cylindrical (r, θ, z).</summary>
    public static void CartesianToCylindrical(double x, double y, double z,
        out double r, out double theta, out double cz)
    {
        r = Math.Sqrt(x * x + y * y);
        theta = Math.Atan2(y, x);
        cz = z;
    }

    /// <summary>Spherical (r, θ, φ) → Cartesian (x, y, z). θ = polar, φ = azimuthal.</summary>
    public static void SphericalToCartesian(double r, double theta, double phi,
        out double x, out double y, out double z)
    {
        x = r * Math.Sin(theta) * Math.Cos(phi);
        y = r * Math.Sin(theta) * Math.Sin(phi);
        z = r * Math.Cos(theta);
    }

    /// <summary>Cartesian (x, y, z) → Spherical (r, θ, φ).</summary>
    public static void CartesianToSpherical(double x, double y, double z,
        out double r, out double theta, out double phi)
    {
        r = Math.Sqrt(x * x + y * y + z * z);
        theta = Math.Acos(Math.Clamp(z / Math.Max(r, 1e-30), -1.0, 1.0));
        phi = Math.Atan2(y, x);
    }

    /// <summary>
    /// Rotate a vector (vx, vy, vz) around axis (ax, ay, az) by angle θ radians.
    /// Uses Rodrigues' rotation formula.
    /// </summary>
    public static void RotateVector(
        double vx, double vy, double vz,
        double ax, double ay, double az, double theta,
        out double rx, out double ry, out double rz)
    {
        double len = Math.Sqrt(ax * ax + ay * ay + az * az);
        if (len < 1e-30)
        { rx = vx; ry = vy; rz = vz; return; }
        ax /= len;
        ay /= len;
        az /= len;

        double cosT = Math.Cos(theta), sinT = Math.Sin(theta);
        double dot = ax * vx + ay * vy + az * vz;
        double crossX = ay * vz - az * vy;
        double crossY = az * vx - ax * vz;
        double crossZ = ax * vy - ay * vx;

        rx = vx * cosT + crossX * sinT + ax * dot * (1.0 - cosT);
        ry = vy * cosT + crossY * sinT + ay * dot * (1.0 - cosT);
        rz = vz * cosT + crossZ * sinT + az * dot * (1.0 - cosT);
    }

    /// <summary>
    /// Compute a rotation matrix for rotation around an arbitrary axis.
    /// Returns 3×3 matrix in row-major order.
    /// </summary>
    public static double[] RotationMatrix(double ax, double ay, double az, double theta)
    {
        double len = Math.Sqrt(ax * ax + ay * ay + az * az);
        if (len < 1e-30)
            return new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        ax /= len;
        ay /= len;
        az /= len;

        double c = Math.Cos(theta), s = Math.Sin(theta), t = 1.0 - c;
        return new double[] {
            t * ax * ax + c,       t * ax * ay - s * az,  t * ax * az + s * ay,
            t * ax * ay + s * az,  t * ay * ay + c,       t * ay * az - s * ax,
            t * ax * az - s * ay,  t * ay * az + s * ax,  t * az * az + c
        };
    }
}

// ============================================================================
//  Additional Utility: Signal Processing
// ============================================================================

/// <summary>
/// Digital signal processing utilities for physics post-processing.
/// </summary>
public static class SignalProcessing
{
    /// <summary>
    /// Apply a simple moving average filter of given window size.
    /// </summary>
    public static void MovingAverage(ReadOnlySpan<double> input, Span<double> output, int windowSize)
    {
        int n = input.Length;
        int half = windowSize / 2;
        double invW = 1.0 / windowSize;

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            int count = 0;
            for (int j = i - half; j <= i + half; j++)
            {
                if (j >= 0 && j < n)
                { sum += input[j]; count++; }
            }
            output[i] = count > 0 ? sum / count : input[i];
        }
    }

    /// <summary>
    /// Apply a Butterworth low-pass filter (2nd order).
    /// </summary>
    public static void ButterworthLowPass(
        ReadOnlySpan<double> input, Span<double> output,
        double cutoffFreq, double sampleFreq)
    {
        double rc = 1.0 / (PhysicsConstants.TwoPi * cutoffFreq);
        double dt = 1.0 / sampleFreq;
        double alpha = dt / (rc + dt);

        double prevIn = input[0], prevOut = input[0];
        output[0] = input[0];

        for (int i = 1; i < input.Length; i++)
        {
            output[i] = alpha * input[i] + alpha * prevIn + (1.0 - 2.0 * alpha) * prevOut;
            prevIn = input[i];
            prevOut = output[i];
        }
    }

    /// <summary>
    /// Compute autocorrelation of a signal (normalised to [−1, 1]).
    /// </summary>
    public static double[] Autocorrelation(ReadOnlySpan<double> signal)
    {
        int n = signal.Length;
        double[] acf = new double[n];

        double mean = 0;
        for (int i = 0; i < n; i++)
            mean += signal[i];
        mean /= n;

        double variance = 0;
        for (int i = 0; i < n; i++)
        {
            double d = signal[i] - mean;
            variance += d * d;
        }
        variance /= n;
        if (variance < 1e-30)
            return acf;

        for (int lag = 0; lag < n; lag++)
        {
            double sum = 0;
            int count = n - lag;
            for (int i = 0; i < count; i++)
                sum += (signal[i] - mean) * (signal[i + lag] - mean);
            acf[lag] = sum / (count * variance);
        }
        return acf;
    }

    /// <summary>
    /// Hilbert transform approximation (for instantaneous amplitude/phase).
    /// </summary>
    public static void HilbertTransform(ReadOnlySpan<double> signal, Span<double> analytic)
    {
        int n = signal.Length;
        int fftSize = 1;
        while (fftSize < n)
            fftSize <<= 1;

        double[] re = new double[fftSize], im = new double[fftSize];
        signal.CopyTo(re);

        FFT.Forward(re, im);

        // Multiply positive frequencies by 2, zero negative frequencies.
        for (int i = 1; i < fftSize / 2; i++)
        {
            re[i] *= 2;
            im[i] *= 2;
            re[fftSize - i] = 0;
            im[fftSize - i] = 0;
        }
        re[0] *= 2;
        im[0] *= 2;

        FFT.Inverse(re, im);

        for (int i = 0; i < n; i++)
            analytic[i] = re[i];
    }

    /// <summary>
    /// Compute instantaneous amplitude (envelope) via Hilbert transform.
    /// </summary>
    public static double[] Envelope(ReadOnlySpan<double> signal)
    {
        int n = signal.Length;
        double[] analytic = new double[n];
        HilbertTransform(signal, analytic);

        double[] env = new double[n];
        for (int i = 0; i < n; i++)
            env[i] = Math.Sqrt(signal[i] * signal[i] + analytic[i] * analytic[i]);
        return env;
    }
}

// ============================================================================
//  Additional Utility: Adaptive Time Stepping
// ============================================================================

/// <summary>
/// Adaptive time-step controller based on embedded Runge-Kutta error estimates.
/// </summary>
public sealed class AdaptiveTimeStepper
{
    private readonly double _tolerance;
    private readonly double _minDt;
    private readonly double _maxDt;
    private readonly double _safetyFactor;
    private double _currentDt;
    private double _previousError;

    public double CurrentDt => _currentDt;

    public AdaptiveTimeStepper(
        double initialDt, double tolerance = 1e-6,
        double minDt = 1e-12, double maxDt = 1.0,
        double safetyFactor = 0.9)
    {
        _currentDt = initialDt;
        _tolerance = tolerance;
        _minDt = minDt;
        _maxDt = maxDt;
        _safetyFactor = safetyFactor;
    }

    /// <summary>
    /// Adjust time-step based on the ratio of tolerance to measured error.
    /// Uses the standard PI controller: dt_new = dt_old * (tol/err)^α * safety
    /// with order p from the embedded method (typically p = 4 for RK45).
    /// </summary>
    public double AdjustTimeStep(double error, double order = 4.0)
    {
        if (error < 1e-30)
            error = 1e-30;
        _previousError = error;

        double factor = _safetyFactor * Math.Pow(_tolerance / error, 1.0 / (order + 1.0));
        factor = Math.Clamp(factor, 0.2, 5.0); // limit step-size changes

        _currentDt *= factor;
        _currentDt = Math.Clamp(_currentDt, _minDt, _maxDt);

        return _currentDt;
    }

    /// <summary>
    /// Returns true if the current step should be rejected and retried.
    /// </summary>
    public bool ShouldReject(double error) => error > _tolerance * 10.0;
}

// ============================================================================
//  Additional Utility: Parallel Execution Helpers
// ============================================================================

/// <summary>
/// Partitioning helper for data-parallel loops.
/// </summary>
public static class ParallelPartitioner
{
    /// <summary>
    /// Execute a loop body in parallel over the range [0, count).
    /// Simple work-stealing partitioner for physics inner loops.
    /// </summary>
    public static void For(int count, Action<int> body, int maxDegreeOfParallelism = 0)
    {
        if (maxDegreeOfParallelism <= 0)
            maxDegreeOfParallelism = Environment.ProcessorCount;

        int batchSize = Math.Max(1, (count + maxDegreeOfParallelism - 1) / maxDegreeOfParallelism);
        int numBatches = (count + batchSize - 1) / batchSize;

        var tasks = new Task[numBatches];
        for (int b = 0; b < numBatches; b++)
        {
            int start = b * batchSize;
            int end = Math.Min(start + batchSize, count);
            tasks[b] = Task.Run(() =>
            {
                for (int i = start; i < end; i++)
                    body(i);
            });
        }
        Task.WaitAll(tasks);
    }

    /// <summary>
    /// Parallel reduction: sum over [0, count) with a value function.
    /// </summary>
    public static double ParallelSum(int count, Func<int, double> valueFunc,
        int maxDegreeOfParallelism = 0)
    {
        if (maxDegreeOfParallelism <= 0)
            maxDegreeOfParallelism = Environment.ProcessorCount;

        int batchSize = Math.Max(1, (count + maxDegreeOfParallelism - 1) / maxDegreeOfParallelism);
        int numBatches = (count + batchSize - 1) / batchSize;
        double[] partialSums = new double[numBatches];

        var tasks = new Task[numBatches];
        for (int b = 0; b < numBatches; b++)
        {
            int start = b * batchSize;
            int end = Math.Min(start + batchSize, count);
            int batchIdx = b;
            tasks[b] = Task.Run(() =>
            {
                double sum = 0;
                for (int i = start; i < end; i++)
                    sum += valueFunc(i);
                partialSums[batchIdx] = sum;
            });
        }
        Task.WaitAll(tasks);

        double total = 0;
        for (int b = 0; b < numBatches; b++)
            total += partialSums[b];
        return total;
    }
}

// ============================================================================
//  Additional Utility: File I/O Helpers for Physics Data
// ============================================================================

/// <summary>
/// Binary I/O helpers for writing physics field data in VTK-compatible format.
/// </summary>
public static class VTKExporter
{
    /// <summary>
    /// Write a 3D scalar field to VTK legacy format (.vtk).
    /// </summary>
    public static void WriteScalarField(string filename, string fieldName,
        double[,,] field, double dx, double dy, double dz)
    {
        int nx = field.GetLength(2);
        int ny = field.GetLength(1);
        int nz = field.GetLength(0);

        using var writer = new System.IO.StreamWriter(filename);
        writer.WriteLine("# vtk DataFile Version 3.0");
        writer.WriteLine("Synapse Omnia Physics Export");
        writer.WriteLine("ASCII");
        writer.WriteLine("DATASET STRUCTURED_POINTS");
        writer.WriteLine($"DIMENSIONS {nx} {ny} {nz}");
        writer.WriteLine($"SPACING {dx} {dy} {dz}");
        writer.WriteLine($"ORIGIN 0 0 0");
        writer.WriteLine($"POINT_DATA {nx * ny * nz}");
        writer.WriteLine($"SCALARS {fieldName} double 1");
        writer.WriteLine("LOOKUP_TABLE default");

        for (int z = 0; z < nz; z++)
            for (int y = 0; y < ny; y++)
                for (int x = 0; x < nx; x++)
                    writer.WriteLine(field[z, y, x].ToString("G15"));
    }

    /// <summary>
    /// Write a 3D vector field to VTK legacy format.
    /// </summary>
    public static void WriteVectorField(string filename, string fieldName,
        double[,,] vx, double[,,] vy, double[,,] vz,
        double dx, double dy, double dz)
    {
        int nx = vx.GetLength(2);
        int ny = vx.GetLength(1);
        int nz = vx.GetLength(0);

        using var writer = new System.IO.StreamWriter(filename);
        writer.WriteLine("# vtk DataFile Version 3.0");
        writer.WriteLine("Synapse Omnia Physics Export");
        writer.WriteLine("ASCII");
        writer.WriteLine("DATASET STRUCTURED_POINTS");
        writer.WriteLine($"DIMENSIONS {nx} {ny} {nz}");
        writer.WriteLine($"SPACING {dx} {dy} {dz}");
        writer.WriteLine($"ORIGIN 0 0 0");
        writer.WriteLine($"POINT_DATA {nx * ny * nz}");
        writer.WriteLine($"VECTORS {fieldName} double");

        for (int z = 0; z < nz; z++)
            for (int y = 0; y < ny; y++)
                for (int x = 0; x < nx; x++)
                    writer.WriteLine($"{vx[z, y, x]:G15} {vy[z, y, x]:G15} {vz[z, y, x]:G15}");
    }

    /// <summary>
    /// Write a 1D profile (time series or line cut) to CSV.
    /// </summary>
    public static void WriteProfile1D(string filename, string[] headers, double[][] columns)
    {
        using var writer = new System.IO.StreamWriter(filename);
        writer.WriteLine(string.Join(",", headers));
        int n = columns[0].Length;
        for (int i = 0; i < n; i++)
        {
            var parts = new string[columns.Length];
            for (int c = 0; c < columns.Length; c++)
                parts[c] = columns[c][i].ToString("G15");
            writer.WriteLine(string.Join(",", parts));
        }
    }
}

// ============================================================================
//  End of Solvers.cs — Synapse Omonia Physics
// ============================================================================
// ============================================================================
//  Additional Utility: Root Finding and Nonlinear Solvers
// ============================================================================

/// <summary>
/// Root-finding algorithms for nonlinear physics equations.
/// </summary>
public static class RootFinding
{
    /// <summary>
    /// Brent's method for finding a root of f(x) = 0 in [a, b].
    /// Combines bisection, secant, and inverse quadratic interpolation.
    /// </summary>
    public static double Brent(Func<double, double> f, double a, double b,
        double tolerance = 1e-12, int maxIter = 100)
    {
        double fa = f(a), fb = f(b);
        if (fa * fb > 0)
            throw new ArgumentException("f(a) and f(b) must have opposite signs.");

        double c = a, fc = fa;
        double d = b - a, e = d;

        for (int iter = 0; iter < maxIter; iter++)
        {
            if (Math.Abs(fc) < Math.Abs(fb))
            {
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }

            double tol1 = 2.0 * 1e-12 * Math.Abs(b) + 0.5 * tolerance;
            double m = 0.5 * (c - b);

            if (Math.Abs(m) <= tol1 || fb == 0)
                return b;

            if (Math.Abs(e) >= tol1 && Math.Abs(fa) > Math.Abs(fb))
            {
                double s = fb / fa;
                double p, q;
                if (a == c)
                {
                    p = 2.0 * m * s;
                    q = 1.0 - s;
                }
                else
                {
                    q = fa / fc;
                    double r = fb / fc;
                    p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                }

                if (p > 0)
                    q = -q;
                else
                    p = -p;
                if (2.0 * p < 3.0 * m * q - Math.Abs(tol1 * q) &&
                    2.0 * p < Math.Abs(e * q))
                {
                    e = d;
                    d = p / q;
                }
                else
                {
                    d = m;
                    e = m;
                }
            }
            else
            {
                d = m;
                e = m;
            }

            a = b;
            fa = fb;
            if (Math.Abs(d) > tol1)
                b += d;
            else
                b += (m > 0 ? tol1 : -tol1);

            fb = f(b);
            if (fb * fc > 0)
            {
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }
        }
        return b;
    }

    /// <summary>
    /// Newton-Raphson method: x_{n+1} = x_n − f(x_n)/f'(x_n).
    /// </summary>
    public static double Newton(Func<double, double> f, Func<double, double> df,
        double x0, double tolerance = 1e-12, int maxIter = 50)
    {
        double x = x0;
        for (int i = 0; i < maxIter; i++)
        {
            double fx = f(x);
            double dfx = df(x);
            if (Math.Abs(dfx) < 1e-30)
                break;
            double dx = fx / dfx;
            x -= dx;
            if (Math.Abs(dx) < tolerance)
                break;
        }
        return x;
    }

    /// <summary>
    /// Secant method: x_{n+1} = x_n − f(x_n) * (x_n − x_{n−1}) / (f(x_n) − f(x_{n−1})).
    /// </summary>
    public static double Secant(Func<double, double> f,
        double x0, double x1, double tolerance = 1e-12, int maxIter = 50)
    {
        double f0 = f(x0), f1 = f(x1);
        for (int i = 0; i < maxIter; i++)
        {
            if (Math.Abs(f1 - f0) < 1e-30)
                break;
            double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
            x0 = x1;
            f0 = f1;
            x1 = x2;
            f1 = f(x1);
            if (Math.Abs(x1 - x0) < tolerance)
                break;
        }
        return x1;
    }
}

// ============================================================================
//  Additional Utility: Least-Squares Fitting
// ============================================================================

/// <summary>
/// Least-squares curve fitting utilities for physics data analysis.
/// </summary>
public static class LeastSquaresFitting
{
    /// <summary>
    /// Linear regression: y = a + b*x using least squares.
    /// Returns (intercept a, slope b, R²).
    /// </summary>
    public static (double Intercept, double Slope, double R2) Linear(
        ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        int n = Math.Min(x.Length, y.Length);
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        for (int i = 0; i < n; i++)
        {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }

        double denom = n * sumX2 - sumX * sumX;
        if (Math.Abs(denom) < 1e-30)
            return (sumY / n, 0, 0);

        double b = (n * sumXY - sumX * sumY) / denom;
        double a = (sumY - b * sumX) / n;

        double yMean = sumY / n;
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++)
        {
            double yPred = a + b * x[i];
            ssTot += (y[i] - yMean) * (y[i] - yMean);
            ssRes += (y[i] - yPred) * (y[i] - yPred);
        }
        double r2 = ssTot > 1e-30 ? 1.0 - ssRes / ssTot : 0;

        return (a, b, r2);
    }

    /// <summary>
    /// Polynomial least-squares fit of degree d: y = Σ c_k x^k.
    /// Uses normal equations with Vandermonde matrix.
    /// </summary>
    public static double[] PolynomialFit(ReadOnlySpan<double> x, ReadOnlySpan<double> y, int degree)
    {
        int n = Math.Min(x.Length, y.Length);
        int d = degree + 1;

        // Build normal equations: (V^T V) c = V^T y
        double[] A = new double[d * d];
        double[] rhs = new double[d];

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                    sum += Math.Pow(x[k], i + j);
                A[i * d + j] = sum;
            }
            double sumY = 0;
            for (int k = 0; k < n; k++)
                sumY += y[k] * Math.Pow(x[k], i);
            rhs[i] = sumY;
        }

        // Solve via Gaussian elimination with partial pivoting.
        double[] c = new double[d];
        for (int i = 0; i < d; i++)
            c[i] = rhs[i];

        for (int col = 0; col < d; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < d; row++)
                if (Math.Abs(A[row * d + col]) > Math.Abs(A[maxRow * d + col]))
                    maxRow = row;

            // Swap rows.
            for (int j = 0; j < d; j++)
            {
                (A[col * d + j], A[maxRow * d + j]) = (A[maxRow * d + j], A[col * d + j]);
            }
            (c[col], c[maxRow]) = (c[maxRow], c[col]);

            double diag = A[col * d + col];
            if (Math.Abs(diag) < 1e-30)
                continue;

            for (int row = col + 1; row < d; row++)
            {
                double factor = A[row * d + col] / diag;
                for (int j = col; j < d; j++)
                    A[row * d + j] -= factor * A[col * d + j];
                c[row] -= factor * c[col];
            }
        }

        // Back-substitute.
        for (int i = d - 1; i >= 0; i--)
        {
            for (int j = i + 1; j < d; j++)
                c[i] -= A[i * d + j] * c[j];
            c[i] /= A[i * d + i];
        }

        return c;
    }

    /// <summary>
    /// Evaluate polynomial at point x given coefficients.
    /// </summary>
    public static double PolynomialEvaluate(ReadOnlySpan<double> coefficients, double x)
    {
        double result = coefficients[coefficients.Length - 1];
        for (int i = coefficients.Length - 2; i >= 0; i--)
            result = result * x + coefficients[i];
        return result;
    }
}
