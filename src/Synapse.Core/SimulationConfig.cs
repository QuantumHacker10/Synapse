// SYNAPSE OMNIA — Synapse.Core
// Split from PhysicsState.cs for maintainability.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Synapse.Core;

// SECTION 12: SIMULATION CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Configuration globale du champ physique — parametres de simulation.
/// Definit la grille spatiale, le pas de temps, les conditions aux limites,
/// et les options de performance (SIMD, GPU, threads).
/// </summary>
public sealed class SimulationConfig
{
    // === Parametres temporels ===
    public double TimeStep { get; set; } = 0.001;
    public double MaxTimeStep { get; set; } = 0.1;
    public double MinTimeStep { get; set; } = 1e-8;
    public bool EnableAdaptiveTimeStepping { get; set; } = true;
    public double CourantNumber { get; set; } = 0.5; // CFL condition

    // === Grille spatiale ===
    public int GridResolutionX { get; set; } = 128;
    public int GridResolutionY { get; set; } = 128;
    public int GridResolutionZ { get; set; } = 128;
    public double DomainSizeX { get; set; } = 10.0;
    public double DomainSizeY { get; set; } = 10.0;
    public double DomainSizeZ { get; set; } = 10.0;
    public double DomainOffsetX { get; set; } = -5.0;
    public double DomainOffsetY { get; set; } = -5.0;
    public double DomainOffsetZ { get; set; } = -5.0;

    // === Solveur ===
    public int MaxIterations { get; set; } = 10000;
    public double ConvergenceTolerance { get; set; } = 1e-6;
    public bool EnableBoundaryConditions { get; set; } = true;
    public BoundaryConditionKind DefaultBoundaryCondition { get; set; } = BoundaryConditionKind.Absorbing;

    // === PNS (Progressive Neural Simulation) ===
    public int PnsTransitionNear { get; set; } = 5;
    public int PnsTransitionFar { get; set; } = 50;
    public int PnsMaxPolygonBudget { get; set; } = 10_000_000;
    public double PnsNeuralPrecisionThreshold { get; set; } = 0.001;
    public double PnsPhysicsLODDistance { get; set; } = 100.0;

    // === Performance ===
    public int MaxConcurrentThreads { get; set; } = Environment.ProcessorCount;
    public bool EnableSimd { get; set; } = true;
    public bool EnableGpuAcceleration { get; set; } = true;
    public string DeviceId { get; set; } = "auto";
    public bool EnableDoublePrecision { get; set; } = true;

    // === Proprietes calculees ===
    public int TotalCells => GridResolutionX * GridResolutionY * GridResolutionZ;
    public double CellSizeX => DomainSizeX / GridResolutionX;
    public double CellSizeY => DomainSizeY / GridResolutionY;
    public double CellSizeZ => DomainSizeZ / GridResolutionZ;
    public double CellVolume => CellSizeX * CellSizeY * CellSizeZ;
    public double SmallestCellDimension => Math.Min(Math.Min(CellSizeX, CellSizeY), CellSizeZ);
    public double LargestCellDimension => Math.Max(Math.Max(CellSizeX, CellSizeY), CellSizeZ);

    public Vector3D GridToWorld(int ix, int iy, int iz) => new(DomainOffsetX + ix * CellSizeX, DomainOffsetY + iy * CellSizeY, DomainOffsetZ + iz * CellSizeZ);
    public void WorldToGrid(Vector3D w, out int ix, out int iy, out int iz) { ix = Math.Clamp((int)((w.X - DomainOffsetX) / CellSizeX), 0, GridResolutionX - 1); iy = Math.Clamp((int)((w.Y - DomainOffsetY) / CellSizeY), 0, GridResolutionY - 1); iz = Math.Clamp((int)((w.Z - DomainOffsetZ) / CellSizeZ), 0, GridResolutionZ - 1); }
    public bool IsValidIndex(int ix, int iy, int iz) => ix >= 0 && ix < GridResolutionX && iy >= 0 && iy < GridResolutionY && iz >= 0 && iz < GridResolutionZ;
    public int FlattenIndex(int ix, int iy, int iz) => ix + GridResolutionX * (iy + GridResolutionY * iz);
    public void UnflattenIndex(int flat, out int ix, out int iy, out int iz) { iz = flat / (GridResolutionX * GridResolutionY); int rem = flat % (GridResolutionX * GridResolutionY); iy = rem / GridResolutionX; ix = rem % GridResolutionX; }

    public override string ToString() => $"SimConfig[{GridResolutionX}x{GridResolutionY}x{GridResolutionZ}] dt={TimeStep} domain=[{DomainSizeX},{DomainSizeY},{DomainSizeZ}] cells={TotalCells:N0}";
}
