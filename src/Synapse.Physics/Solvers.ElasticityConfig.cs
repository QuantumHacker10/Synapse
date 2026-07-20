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
public sealed class ElasticityConfig
{
    public (int Nx, int Ny, int Nz) GridSize { get; init; } = (64, 64, 64);
    public double CellSize { get; init; } = 0.01;          // metres
    public double TimeStep { get; init; } = 1e-6;
    public int NumSteps { get; init; } = 10_000;
    public double YoungsModulus { get; init; } = 200e9;    // Pa (steel)
    public double PoissonRatio { get; init; } = 0.3;
    public double Density { get; init; } = 7800.0;        // kg/m³
    public double YieldStress { get; init; } = 250e6;     // Pa
    public double HardeningModulus { get; init; } = 2e9;   // Pa
    public MaterialModel Material { get; init; } = MaterialModel.LinearElastic;
    public double DampingAlpha { get; init; } = 0.01;     // Rayleigh α
    public double DampingBeta { get; init; } = 1e-7;      // Rayleigh β
    public bool EnableContact { get; init; }
    public double ContactStiffness { get; init; } = 1e10;  // penalty stiffness
    public double ContactGap { get; init; } = 0.001;       // m
    public bool EnableModalAnalysis { get; init; }
    public int NumModes { get; init; } = 10;
}

/// <summary>
/// 3-D linear elasticity solver on a regular grid with FEM-like
/// discretisation. Supports linear elastic, neo-Hookean, and J2
/// plasticity (isotropic hardening) material models, von Mises stress,
/// contact mechanics penalty method, and modal analysis.
/// </summary>
