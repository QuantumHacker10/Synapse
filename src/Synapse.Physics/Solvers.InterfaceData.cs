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
public sealed class InterfaceData
{
    public string Name { get; init; }
    public double[] FieldData { get; set; }
    public int NumNodes { get; init; }
    public DateTime Timestamp { get; set; }

    public InterfaceData(string name, int numNodes)
    {
        Name = name;
        NumNodes = numNodes;
        FieldData = new double[numNodes];
        Timestamp = DateTime.UtcNow;
    }
}

/// <summary>
/// Represents a coupling link between two solvers.
/// </summary>
