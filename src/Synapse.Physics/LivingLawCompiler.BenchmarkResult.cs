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
    // LawBenchmark — performance benchmarking for compilation and execution
    // =========================================================================

    /// <summary>Benchmark result for a single operation.</summary>
    public sealed class BenchmarkResult
    {
        public string OperationName { get; init; } = "";
        public long ElapsedTicks { get; init; }
        public double ElapsedMilliseconds { get; init; }
        public int Iterations { get; init; }
        public double OpsPerSecond { get; init; }
        public long MemoryBytes { get; init; }
        public string? AdditionalInfo { get; init; }
    }
}
