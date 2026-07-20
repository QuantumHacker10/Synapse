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

    /// <summary>Result of a law validation.</summary>
    public readonly struct ValidationResult
    {
        public readonly bool IsValid;
        public readonly string[] Errors;
        public readonly string[] Warnings;
        public readonly Dimension[] TermDimensions;
        public readonly bool DimensionallyConsistent;
        public readonly float StabilityCflRatio;

        public ValidationResult(bool isValid, string[] errors, string[] warnings,
            Dimension[] termDimensions, bool dimensionallyConsistent, float stabilityCflRatio)
        {
            IsValid = isValid;
            Errors = errors;
            Warnings = warnings;
            TermDimensions = termDimensions;
            DimensionallyConsistent = dimensionallyConsistent;
            StabilityCflRatio = stabilityCflRatio;
        }

        public static ValidationResult Valid() =>
            new(true, Array.Empty<string>(), Array.Empty<string>(), Array.Empty<Dimension>(), true, 0f);
    }
}
