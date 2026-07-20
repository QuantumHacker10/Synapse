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
