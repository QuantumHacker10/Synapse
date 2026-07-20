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

    // =========================================================================
    // LawSimulationRunner — runs simulations with compiled laws
    // =========================================================================

    /// <summary>Configuration for a simulation run.</summary>
    public sealed class SimulationConfig
    {
        public float Duration { get; set; } = 1.0f;
        public float TimeStep { get; set; } = 0.001f;
        public int GridSize { get; set; } = 64;
        public bool RecordHistory { get; set; } = true;
        public int HistoryInterval { get; set; } = 10;
        public Func<int, PhysicsField, bool>? StopCondition { get; set; }
    }
}
