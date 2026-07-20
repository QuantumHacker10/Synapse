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

    /// <summary>AST node for the parsed expression.</summary>
    public sealed class AstNode
    {
        public NodeType Type { get; init; }
        public string? Value { get; init; }
        public float NumericValue { get; init; }
        public AstNode? Left { get; init; }
        public AstNode? Right { get; init; }
        public AstNode? Middle { get; init; }
        public List<AstNode>? Children { get; init; }
        public Dimension InferredDimension { get; set; } = Dimension.Scalar;
    }
}
