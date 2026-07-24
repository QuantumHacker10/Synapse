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
    // LawDimensionalAnalyzer — advanced dimensional analysis
    // =========================================================================

    /// <summary>Unit representation for dimensional analysis.</summary>
    public sealed class PhysicalUnit
    {
        public string Symbol { get; init; } = "";
        public string Name { get; init; } = "";
        public Dimension BaseDimension { get; init; } = Dimension.Scalar;
        public float ConversionFactor { get; init; } = 1.0f;
        public float Offset { get; init; } = 0.0f;

        public float ConvertToBase(float value) => value * ConversionFactor + Offset;
        public float ConvertFromBase(float baseValue) => (baseValue - Offset) / ConversionFactor;
    }
}
