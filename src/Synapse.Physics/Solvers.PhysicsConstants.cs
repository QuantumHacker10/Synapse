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
internal static class PhysicsConstants
{
    public const double C0 = 299_792_458.0;            // speed of light m/s
    public const double Mu0 = 4.0 * Math.PI * 1e-7;   // vacuum permeability
    public const double Eps0 = 8.8541878128e-12;       // vacuum permittivity
    public const double kB = 1.380649e-23;             // Boltzmann constant
    public const double Hbar = 1.054571817e-34;        // reduced Planck
    public const double Me = 9.1093837015e-31;         // electron mass
    public const double G = 6.67430e-11;               // gravitational constant
    public const double Pi = Math.PI;
    public const double TwoPi = 2.0 * Math.PI;
    public const double InvPi = 1.0 / Math.PI;
}

