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
    // LawExpressionParser — complete expression parser
    // =========================================================================

    /// <summary>Token types for the expression parser.</summary>
    public enum TokenType
    {
        Number, Identifier,
        Plus, Minus, Star, Slash, Percent, Caret,
        LeftParen, RightParen, LeftBracket, RightBracket,
        Comma, Semicolon, Dot,
        Equals, NotEquals, Less, Greater, LessOrEqual, GreaterOrEqual,
        And, Or, Not,
        Question, Colon, Assign,
        Eof, Error
    }
}
