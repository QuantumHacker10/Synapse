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
    // LawProfiler — detailed profiling for law operations
    // =========================================================================

    /// <summary>Profile data for a single operation.</summary>
    public sealed class ProfileData
    {
        public string OperationName { get; init; } = "";
        public int CallCount { get; set; }
        public double TotalMilliseconds { get; set; }
        public double MinMilliseconds { get; set; } = double.MaxValue;
        public double MaxMilliseconds { get; set; } = double.MinValue;
        public double LastMilliseconds { get; set; }
        public long TotalBytes { get; set; }
        public double AverageMilliseconds => CallCount > 0 ? TotalMilliseconds / CallCount : 0;
    }
}
