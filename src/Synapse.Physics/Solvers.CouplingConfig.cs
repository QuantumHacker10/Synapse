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
public sealed class CouplingConfig
{
    public CouplingScheme Scheme { get; init; } = CouplingScheme.GaussSeidel;
    public int MaxIterations { get; init; } = 20;
    public double ConvergenceTolerance { get; init; } = 1e-6;
    public int TimeSteps { get; init; } = 1000;
    public int CouplingInterval { get; init; } = 10;
    public double Relaxation { get; init; } = 0.8;
    public bool AdaptiveRelaxation { get; init; }
    public double MinRelaxation { get; init; } = 0.1;
    public double MaxRelaxation { get; init; } = 1.0;
    public bool EnableLoadBalancing { get; init; }
}

/// <summary>
/// Represents the data exchanged between coupled solvers at a shared interface.
/// </summary>
