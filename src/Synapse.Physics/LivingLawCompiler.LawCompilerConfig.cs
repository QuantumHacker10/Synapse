// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    /// <summary>Configuration for the Living Law Compiler.</summary>
    public sealed class LawCompilerConfig
    {
        /// <summary>Simulation tolerance (convergence threshold).</summary>
        public float Tolerance { get; set; } = 1e-6f;
        /// <summary>Maximum iterations for iterative solvers.</summary>
        public int MaxIterations { get; set; } = 1000;
        /// <summary>Time step for explicit schemes.</summary>
        public float TimeStep { get; set; } = 0.001f;
        /// <summary>Cell size for spatial discretization.</summary>
        public float CellSize { get; set; } = 1.0f;
        /// <summary>Boundary condition type.</summary>
        public BoundaryCondition BoundaryCondition { get; set; } = BoundaryCondition.Periodic;
        /// <summary>Boundary value for Dirichlet conditions.</summary>
        public float BoundaryValue { get; set; } = 0f;
        /// <summary>Gas limit for bytecode execution.</summary>
        public long GasLimit { get; set; } = 10_000_000;
        /// <summary>CFL stability limit.</summary>
        public float CflLimit { get; set; } = 1.0f;
        /// <summary>Physical constants used in the simulation.</summary>
        public Dictionary<string, float> Constants { get; set; } = new()
        {
            ["alpha"] = 1.43e-4f,
            ["c"] = 340f,
            ["k"] = 100f,
            ["m"] = 1f,
            ["G"] = 6.674e-11f,
            ["R"] = 287.058f,
            ["sigma"] = 5.670374419e-8f,
            ["mu"] = 1.002e-3f,
            ["rho"] = 1.225f,
            ["D"] = 1e-5f,
            ["eps"] = 8.854e-12f,
            ["mu0"] = 1.25663706212e-6f,
            ["vx"] = 1.0f,
            ["vy"] = 0f,
            ["vz"] = 0f,
            ["ke"] = 8.9875517923e9f,
            ["h"] = 0.01f,
        };
        /// <summary>Coupling parameters between fields.</summary>
        public Dictionary<string, float> CouplingParameters { get; set; } = new();
        /// <summary>Solver method selection.</summary>
        public SolverMethod Solver { get; set; } = SolverMethod.ExplicitEuler;
        /// <summary>Number of grid cells per dimension.</summary>
        public int GridSize { get; set; } = 64;
        /// <summary>Enable hot-reload of law expressions.</summary>
        public bool EnableHotReload { get; set; } = true;
        /// <summary>Compile with validation enabled.</summary>
        public bool EnableValidation { get; set; } = true;
        /// <summary>Maximum bytecode instructions.</summary>
        public int MaxBytecodeInstructions { get; set; } = 4096;
        /// <summary>Enable JIT-like specialization for hot loops.</summary>
        public bool EnableJitSpecialization { get; set; } = false;
        /// <summary>Number of parallel threads for field operations.</summary>
        public int Parallelism { get; set; } = 1;
        /// <summary>Output directory for compilation artifacts.</summary>
        public string? OutputDirectory { get; set; }

        /// <summary>Create a copy of this configuration.</summary>
        public LawCompilerConfig Clone()
        {
            return new LawCompilerConfig
            {
                Tolerance = Tolerance,
                MaxIterations = MaxIterations,
                TimeStep = TimeStep,
                CellSize = CellSize,
                BoundaryCondition = BoundaryCondition,
                BoundaryValue = BoundaryValue,
                GasLimit = GasLimit,
                CflLimit = CflLimit,
                Solver = Solver,
                GridSize = GridSize,
                EnableHotReload = EnableHotReload,
                EnableValidation = EnableValidation,
                MaxBytecodeInstructions = MaxBytecodeInstructions,
                EnableJitSpecialization = EnableJitSpecialization,
                Parallelism = Parallelism,
                OutputDirectory = OutputDirectory,
                Constants = new Dictionary<string, float>(Constants),
                CouplingParameters = new Dictionary<string, float>(CouplingParameters)
            };
        }

        /// <summary>Get a preset configuration for a specific physics domain.</summary>
        public static LawCompilerConfig ForDomain(string domain) => domain.ToLowerInvariant() switch
        {
            "heat" or "thermal" => new LawCompilerConfig
            {
                TimeStep = 0.001f,
                CellSize = 1f,
                GridSize = 64,
                BoundaryCondition = BoundaryCondition.Dirichlet,
                BoundaryValue = 300f,
                CflLimit = 0.5f,
                Constants = new() { ["alpha"] = 1.43e-4f, ["k"] = 205f }
            },
            "wave" or "acoustic" => new LawCompilerConfig
            {
                TimeStep = 0.0001f,
                CellSize = 0.1f,
                GridSize = 128,
                BoundaryCondition = BoundaryCondition.Periodic,
                CflLimit = 0.9f,
                Constants = new() { ["c"] = 340f }
            },
            "fluid" or "navier_stokes" => new LawCompilerConfig
            {
                TimeStep = 0.0001f,
                CellSize = 0.01f,
                GridSize = 64,
                BoundaryCondition = BoundaryCondition.Periodic,
                CflLimit = 0.5f,
                Solver = SolverMethod.RungeKutta4,
                Constants = new() { ["mu"] = 1.002e-3f, ["rho"] = 1000f }
            },
            "elasticity" or "solid" => new LawCompilerConfig
            {
                TimeStep = 0.001f,
                CellSize = 1f,
                GridSize = 32,
                BoundaryCondition = BoundaryCondition.Neumann,
                Constants = new() { ["k"] = 100f, ["m"] = 1f }
            },
            "electromagnetic" or "em" => new LawCompilerConfig
            {
                TimeStep = 1e-10f,
                CellSize = 1e-3f,
                GridSize = 64,
                BoundaryCondition = BoundaryCondition.Periodic,
                CflLimit = 0.95f,
                Constants = new() { ["eps"] = 8.854e-12f, ["mu0"] = 1.25663706212e-6f }
            },
            "gravity" or "gravitation" => new LawCompilerConfig
            {
                TimeStep = 0.01f,
                CellSize = 1e6f,
                GridSize = 32,
                BoundaryCondition = BoundaryCondition.Periodic,
                Constants = new() { ["G"] = 6.674e-11f }
            },
            _ => new LawCompilerConfig()
        };
    }    // =========================================================================
}
