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
