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

    /// <summary>Describes a modification to be applied to a law expression.</summary>
    public sealed class LawModification
    {
        public ModificationType Type { get; set; }
        public string? TargetExpression { get; set; }
        public string? ReplacementExpression { get; set; }
        public float? ConstantValue { get; set; }
        public string? VariableName { get; set; }
        public float ScaleFactor { get; set; } = 1f;
        public string? CouplingTerm { get; set; }
        public string? Description { get; set; }
        public Dictionary<string, float>? Metadata { get; set; }
    }
}
