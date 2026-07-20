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
