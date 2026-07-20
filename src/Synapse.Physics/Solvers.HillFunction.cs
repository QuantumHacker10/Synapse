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
