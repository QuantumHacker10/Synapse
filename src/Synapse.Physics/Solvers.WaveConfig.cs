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

public sealed class WaveConfig
{
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (100, 100, 100);
    public double CellSize { get; init; } = 0.01;
    public double TimeStep { get; init; } = 1e-5;
    public int NumSteps { get; init; } = 2000;
    public double SoundSpeed { get; init; } = 343.0;
    public double Density { get; init; } = 1.225;
    public int PmlThickness { get; init; } = 10;
    public int PmlOrder { get; init; } = 3;
    public double PmlR0 { get; init; } = 1e-6;
    public double SourceFrequency { get; init; } = 1000.0;
    public double SourceAmplitude { get; init; } = 1.0;
    public (int X, int Y, int Z) SourcePosition { get; init; } = (50, 50, 50);
    public bool UseFrequencyDomain { get; init; }
    public double Omega { get; init; } = 6283.185;
    public (bool X, bool Y, bool Z) Periodic { get; init; }
    public (bool NegX, bool PosX, bool NegY, bool PosY, bool NegZ, bool PosZ) PmlFaces { get; init; }
        = (true, true, true, true, true, true);
}

/// <summary>
/// 3-D acoustic wave propagator using finite differences with PML
/// absorbing boundaries. Supports time-domain and frequency-domain
/// (time-harmonic Helmholtz) formulations.
/// </summary>
