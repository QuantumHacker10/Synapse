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
