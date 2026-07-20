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
public sealed class SchrodingerConfig
{
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (128, 128, 1);
    public double GridSpacing { get; init; } = 0.1;        // Ångströms
    public double TimeStep { get; init; } = 0.001;         // femtoseconds
    public int NumSteps { get; init; } = 10_000;
    public int NumEigenstates { get; init; } = 5;          // number of eigenstates to compute
    public int EigenMaxIter { get; init; } = 500;
    public double EigenTolerance { get; init; } = 1e-12;
    public bool ComputeDensity { get; init; } = true;
    public bool RecordProbability { get; init; } = false;
    public int RecordInterval { get; init; } = 100;
    public double PotentialDepth { get; init; } = 10.0;    // eV
    public double ParticleMass { get; init; } = 1.0;       // electron masses
}

/// <summary>
/// Schrödinger equation solver supporting time-dependent (Crank-Nicolson),
/// time-independent (inverse iteration), density, expectation values,
/// and eigenstate computation for single-particle quantum mechanics
/// on a spatial grid.
/// </summary>
