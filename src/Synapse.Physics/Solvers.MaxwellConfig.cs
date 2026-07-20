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

public sealed class MaxwellConfig
{
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (64, 64, 64);
    public double CellSize { get; init; } = 1e-3;
    public double TimeStep { get; init; } = 1e-12;
    public int NumSteps { get; init; } = 1000;
    public double EpsR { get; init; } = 1.0;
    public double MuR { get; init; } = 1.0;
    public double Sigma { get; init; } = 0.0;
    public int PmlThickness { get; init; } = 8;
    public int PmlOrder { get; init; } = 3;
    public double PmlR0 { get; init; } = 1e-6;
    public (bool X, bool Y, bool Z) Periodic { get; init; }
    public PolarizationModel Polarization { get; init; } = PolarizationModel.None;
    public double DebyeOmegaP { get; init; } = 1e10;
    public double DebyeTau { get; init; } = 1e-12;
    public double DrudeOmegaP { get; init; } = 1e12;
    public double DrudeGamma { get; init; } = 1e10;
    public (int X, int Y, int Z) SourcePosition { get; init; } = (32, 32, 32);
    public double SourceFrequency { get; init; } = 10e9;
    public double SourceAmplitude { get; init; } = 1.0;
    public bool UsePlaneWave { get; init; }
    public (double Dx, double Dy, double Dz) PlaneWaveDirection { get; init; } = (0, 0, 1);
    public (double Px, double Py, double Pz) PlaneWavePolarisation { get; init; } = (1, 0, 0);
    public (bool NegX, bool PosX, bool NegY, bool PosY, bool NegZ, bool PosZ) PmlFaces { get; init; }
        = (true, true, true, true, true, true);
}

/// <summary>Snapshot of the electromagnetic field on the Yee grid.</summary>
