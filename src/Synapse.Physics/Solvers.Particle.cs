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
public struct Particle
{
    public double X, Y, Z;
    public double Fx, Fy, Fz;
    public double Charge;

    public Particle(double x, double y, double z)
    {
        X = x;
        Y = y;
        Z = z;
        Fx = Fy = Fz = 0;
        Charge = 0;
    }
}

/// <summary>
/// Lennard-Jones pair potential: u(r) = 4ε [(σ/r)¹² − (σ/r)⁶].
/// </summary>
