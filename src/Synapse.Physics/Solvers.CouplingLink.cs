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

public sealed class CouplingLink
{
    public string SourceSolver { get; init; }
    public string TargetSolver { get; init; }
    public string SourceField { get; init; }
    public string TargetField { get; init; }
    public Func<double[], double[]> TransferFunction { get; init; }
}

/// <summary>
/// Monitors convergence of the partitioned coupling iteration.
/// </summary>
