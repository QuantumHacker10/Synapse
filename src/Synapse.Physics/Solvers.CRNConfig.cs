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
