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
internal sealed class BHNode
{
    public double Cx, Cy, Cz;        // centre of mass
    public double TotalMass;
    public double HalfSize;
    public int BodyIndex = -1;       // leaf: index into body array
    public BHNode[] Children;        // 8 children (null for leaf or empty)
    public int ChildCount;
}

/// <summary>
/// N-body gravitational solver with direct O(N²) summation, Barnes-Hut
/// octree (O(N log N)), leapfrog integration, energy/angular-momentum
/// conservation tracking, and post-Newtonian gravitational radiation.
/// </summary>
