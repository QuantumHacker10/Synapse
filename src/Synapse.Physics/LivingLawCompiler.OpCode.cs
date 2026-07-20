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
    // LawBytecode — stack-based bytecode for compiled laws
    // =========================================================================

    /// <summary>Opcodes for the bytecode VM.</summary>
    public enum OpCode : byte
    {
        Nop = 0,
        PushConst = 1, Pop = 2, Dup = 3, Swap = 4,
        Add = 10, Sub = 11, Mul = 12, Div = 13, Mod = 14, Pow = 15, Neg = 16, Abs = 17,
        LoadVar = 30, StoreVar = 31, LoadField = 32, LoadParam = 33,
        Equals = 40, NotEquals = 41, LessThan = 42, GreaterThan = 43,
        LessOrEqual = 44, GreaterOrEqual = 45, LogicalAnd = 46, LogicalOr = 47,
        LogicalNot = 48, TernaryJump = 49,
        Sin = 60, Cos = 61, Tan = 62, Asin = 63, Acos = 64, Atan = 65, Atan2 = 66,
        Sinh = 67, Cosh = 68, Tanh = 69, Exp = 70, Log = 71, Log2 = 72, Log10 = 73,
        Sqrt = 74, Cbrt = 75, Ceil = 76, Floor = 77, Round = 78, Clamp = 79, Lerp = 80,
        Min = 81, Max = 82, Sign = 83,
        GradientX = 90, GradientY = 91, GradientZ = 92, Laplacian = 93,
        Divergence = 94, CurlX = 95, CurlY = 96, CurlZ = 97,
        ConditionalJump = 100, UnconditionalJump = 101, Call = 102, Return = 103,
        GasConsume = 110, BoundsCheck = 111,
        Halt = 255
    }
}
