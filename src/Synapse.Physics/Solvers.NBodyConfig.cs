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

public sealed class NBodyConfig
{
    public int NumBodies { get; init; } = 1000;
    public int NumSteps { get; init; } = 10_000;
    public double TimeStep { get; init; } = 0.01;
    public double Softening { get; init; } = 0.01;       // gravitational softening length
    public bool UseBarnesHut { get; init; }
    public double Theta { get; init; } = 0.5;            // Barnes-Hut opening angle
    public bool ComputeRadiation { get; init; }          // post-Newtonian radiation
    public int RadiationOrder { get; init; } = 2;        // PN order (1 = 1PN, 2 = 2PN)
    public double GravitationalConstant { get; init; } = 1.0;
    public double[] Masses { get; init; }                // per-body masses (null = unit)
    public bool RecordTrajectory { get; init; }
    public int TrajectoryInterval { get; init; } = 100;
}

/// <summary>
/// Represents an N-body particle with position, velocity, mass, and acceleration.
/// </summary>
