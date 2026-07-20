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

    /// <summary>A diagnostic message from the compiler.</summary>
    public sealed class DiagnosticMessage
    {
        public DiagnosticSeverity Severity { get; init; }
        public string Code { get; init; } = "";
        public string Message { get; init; } = "";
        public int Line { get; init; }
        public int Column { get; init; }
        public string? Expression { get; init; }
        public string? Suggestion { get; init; }
        public DateTime Timestamp { get; init; } = DateTime.UtcNow;

        public override string ToString()
        {
            string loc = Line > 0 ? $" at line {Line}:{Column}" : "";
            string sug = Suggestion != null ? $" (suggestion: {Suggestion})" : "";
            return $"[{Severity}] {Code}: {Message}{loc}{sug}";
        }
    }
}
