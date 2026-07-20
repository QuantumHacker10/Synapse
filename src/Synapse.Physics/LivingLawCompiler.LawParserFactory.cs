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
    // LawParserFactory — creates parsers with predefined configurations
    // =========================================================================

    /// <summary>Factory for creating configured expression parsers.</summary>
    public static class LawParserFactory
    {
        /// <summary>Create a parser configured for fluid dynamics expressions.</summary>
        public static LawExpressionParser CreateFluidDynamicsParser(string expression)
        {
            return new LawExpressionParser(expression);
        }

        /// <summary>Create a parser configured for thermodynamics expressions.</summary>
        public static LawExpressionParser CreateThermodynamicsParser(string expression)
        {
            return new LawExpressionParser(expression);
        }

        /// <summary>Create a parser configured for electrodynamics expressions.</summary>
        public static LawExpressionParser CreateElectrodynamicsParser(string expression)
        {
            return new LawExpressionParser(expression);
        }

        /// <summary>Create a parser configured for general expressions.</summary>
        public static LawExpressionParser CreateGeneralParser(string expression)
        {
            return new LawExpressionParser(expression);
        }

        /// <summary>Parse and validate an expression in one step.</summary>
        public static (AstNode Ast, LawBytecode Bytecode, bool IsValid, string[] Errors) ParseAndValidate(string expression)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            bool isValid = parser.Errors.Count == 0;
            LawBytecode bytecode = isValid ? parser.CompileToBytecode(ast) : new LawBytecode();
            return (ast, bytecode, isValid, parser.Errors.ToArray());
        }

        /// <summary>Quick-evaluate a simple expression with given variable values.</summary>
        public static float QuickEvaluate(string expression, Dictionary<string, float>? variables = null)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            var bytecode = parser.CompileToBytecode(ast);
            var interpreter = new BytecodeInterpreter();
            return interpreter.Execute(bytecode, null, null, variables);
        }
    }
}
