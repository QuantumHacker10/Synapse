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
