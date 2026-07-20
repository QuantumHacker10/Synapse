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
public struct NBodyParticle
{
    public double X, Y, Z;
    public double Vx, Vy, Vz;
    public double Ax, Ay, Az;
    public double Mass;

    // Post-Newtonian radiation reaction.
    public double PNx, PNy, PNz;

    public NBodyParticle(double x, double y, double z, double mass)
    {
        X = x;
        Y = y;
        Z = z;
        Vx = Vy = Vz = 0;
        Ax = Ay = Az = 0;
        Mass = mass;
        PNx = PNy = PNz = 0;
    }
}

/// <summary>
/// Barnes-Hut octree node for hierarchical force computation.
/// </summary>
