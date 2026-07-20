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
