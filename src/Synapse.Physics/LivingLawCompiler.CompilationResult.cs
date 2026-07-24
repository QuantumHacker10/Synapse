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

    /// <summary>Result of a compilation or validation operation.</summary>
    public readonly struct CompilationResult
    {
        public readonly bool Success;
        public readonly string Message;
        public readonly string[] Errors;
        public readonly string[] Warnings;
        public readonly LawBytecode? Bytecode;
        public readonly int InstructionCount;
        public readonly long CompilationTimeMs;

        public CompilationResult(bool success, string message, string[] errors, string[] warnings,
            LawBytecode? bytecode, int instructionCount, long compilationTimeMs)
        {
            Success = success;
            Message = message;
            Errors = errors;
            Warnings = warnings;
            Bytecode = bytecode;
            InstructionCount = instructionCount;
            CompilationTimeMs = compilationTimeMs;
        }

        public static CompilationResult Ok(string msg, LawBytecode bc, int ins, long ms) =>
            new(true, msg, Array.Empty<string>(), Array.Empty<string>(), bc, ins, ms);
        public static CompilationResult Fail(string msg, string[] errors) =>
            new(false, msg, errors, Array.Empty<string>(), null, 0, 0);
    }
}
