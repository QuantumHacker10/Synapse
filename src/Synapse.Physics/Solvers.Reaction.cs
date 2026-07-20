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
public sealed class Reaction
{
    /// <summary>Species indices of reactants (can be empty for zeroth-order).</summary>
    public int[] Reactants { get; init; }

    /// <summary>Species indices of products (can be empty for degradation).</summary>
    public int[] Products { get; init; }

    /// <summary>Reaction rate constant.</summary>
    public double RateConstant { get; init; }

    /// <summary>If true, reaction is treated as irreversible.</summary>
    public bool Irreversible { get; init; } = true;

    /// <summary>Stoichiometric coefficient for each reactant (parallel to Reactants).</summary>
    public int[] ReactantCoefficients { get; init; }

    /// <summary>Stoichiometric coefficient for each product (parallel to Products).</summary>
    public int[] ProductCoefficients { get; init; }

    /// <summary>Human-readable label.</summary>
    public string Label { get; init; } = string.Empty;
}

/// <summary>
/// Represents an enzymatic reaction following Michaelis-Menten kinetics.
/// </summary>
