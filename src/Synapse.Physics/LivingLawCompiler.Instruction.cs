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

    /// <summary>A single bytecode instruction.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct Instruction
    {
        public readonly OpCode Op;
        public readonly int Operand;
        public readonly float FloatOperand;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Instruction(OpCode op, int operand = 0, float floatOperand = 0f)
        {
            Op = op;
            Operand = operand;
            FloatOperand = floatOperand;
        }

        public override string ToString() => Op switch
        {
            OpCode.PushConst => $"PushConst({FloatOperand})",
            OpCode.LoadVar => $"LoadVar({Operand})",
            OpCode.LoadField => $"LoadField({Operand})",
            OpCode.LoadParam => $"LoadParam({Operand})",
            OpCode.ConditionalJump => $"CondJump({Operand})",
            OpCode.UnconditionalJump => $"Jump({Operand})",
            OpCode.GasConsume => $"Gas({Operand})",
            OpCode.BoundsCheck => $"BoundsCheck({Operand})",
            OpCode.Call => $"Call({Operand})",
            _ => Op.ToString()
        };
    }
}
