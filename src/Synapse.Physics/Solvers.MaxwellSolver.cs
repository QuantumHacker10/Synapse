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
