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
public sealed class MichaelisMentenReaction
{
    public string Label { get; init; } = string.Empty;
    public int SubstrateIndex { get; init; }
    public int ProductIndex { get; init; }
    public int EnzymeIndex { get; init; }
    public double Vmax { get; init; }     // maximum reaction rate
    public double Km { get; init; }       // Michaelis constant
}

/// <summary>
/// Represents a Hill-function regulatory interaction.
/// </summary>
