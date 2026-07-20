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
