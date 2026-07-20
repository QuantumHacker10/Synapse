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
public enum EnsembleType
{
    NVT,   // Canonical (fixed N, V, T)
    NPT,   // Isothermal-isobaric
    Grand, // Grand canonical (fixed μ, V, T)
    Gibbs  // Gibbs ensemble for phase equilibria
}

/// <summary>
/// Configuration for the thermodynamic ensemble simulation.
/// </summary>
