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

    /// <summary>Advanced dimensional analysis for law expressions.</summary>
    public sealed class LawDimensionalAnalyzer
    {
        private readonly UnitRegistry _unitRegistry;
        private readonly Dictionary<string, Dimension> _variableDimensions = new();

        public LawDimensionalAnalyzer()
        {
            _unitRegistry = new UnitRegistry();
            InitializeVariableDimensions();
        }

        private void InitializeVariableDimensions()
        {
            _variableDimensions["x"] = Dimension.LengthD;
            _variableDimensions["y"] = Dimension.LengthD;
            _variableDimensions["z"] = Dimension.LengthD;
            _variableDimensions["t"] = Dimension.TimeD;
            _variableDimensions["T"] = Dimension.TemperatureD;
            _variableDimensions["P"] = Dimension.Pressure;
            _variableDimensions["rho"] = Dimension.Density;
            _variableDimensions["v"] = Dimension.Velocity;
            _variableDimensions["u"] = Dimension.Velocity;
            _variableDimensions["w"] = Dimension.Velocity;
            _variableDimensions["F"] = Dimension.Force;
            _variableDimensions["E"] = Dimension.Energy;
            _variableDimensions["k"] = new Dimension(0, 2, -1, 0, 0, 0, 0, 0);
            _variableDimensions["mu"] = Dimension.Viscosity;
            _variableDimensions["alpha"] = new Dimension(0, 2, -1, 0, 0, 0, 0, 0);
            _variableDimensions["sigma"] = new Dimension(1, 0, -3, -4, 0, 0, 0, 0);
            _variableDimensions["G"] = new Dimension(-1, 3, -2, 0, 0, 0, 0, 0);
            _variableDimensions["R"] = new Dimension(0, 2, -2, -1, 0, 0, 0, 0);
            _variableDimensions["c"] = Dimension.Velocity;
            _variableDimensions["I"] = new Dimension(0, 0, 0, 0, 0, 1, 0, 0);
            _variableDimensions["q"] = new Dimension(0, 0, 0, 0, 0, 1, 0, 0);
            _variableDimensions["dt"] = Dimension.TimeD;
            _variableDimensions["dx"] = Dimension.LengthD;
            _variableDimensions["dy"] = Dimension.LengthD;
            _variableDimensions["dz"] = Dimension.LengthD;
            _variableDimensions["m1"] = Dimension.MassD;
            _variableDimensions["m2"] = Dimension.MassD;
            _variableDimensions["r"] = Dimension.LengthD;
        }

        /// <summary>Analyze an expression and return dimension of each sub-expression.</summary>
        public Dictionary<AstNode, Dimension> AnalyzeDimensions(AstNode ast)
        {
            var result = new Dictionary<AstNode, Dimension>();
            AnalyzeNode(ast, result);
            return result;
        }

        private Dimension AnalyzeNode(AstNode node, Dictionary<AstNode, Dimension> result)
        {
            if (node == null)
                return Dimension.Scalar;

            Dimension dim = node.Type switch
            {
                NodeType.NumberLiteral => Dimension.Scalar,
                NodeType.Identifier => _variableDimensions.TryGetValue(node.Value ?? "", out var d) ? d : Dimension.Scalar,
                NodeType.FieldAccess => Dimension.Scalar,
                NodeType.BinaryExpression => AnalyzeBinary(node, result),
                NodeType.UnaryExpression => AnalyzeNode(node.Left!, result),
                NodeType.TernaryExpression => AnalyzeNode(node.Right!, result),
                NodeType.FunctionCall => AnalyzeFunction(node, result),
                _ => Dimension.Scalar
            };

            result[node] = dim;
            return dim;
        }

        private Dimension AnalyzeBinary(AstNode node, Dictionary<AstNode, Dimension> result)
        {
            var leftDim = AnalyzeNode(node.Left!, result);
            var rightDim = AnalyzeNode(node.Right!, result);

            return node.Value switch
            {
                "+" or "-" => leftDim,
                "*" => leftDim.Multiply(rightDim),
                "/" => leftDim.Divide(rightDim),
                "^" => leftDim.Pow(rightDim.IsDimensionless ? 1f : 2f),
                "%" => leftDim,
                "==" or "!=" or "<" or ">" or "<=" or ">=" => Dimension.Scalar,
                "&&" or "||" => Dimension.Scalar,
                _ => Dimension.Scalar
            };
        }

        private Dimension AnalyzeFunction(AstNode node, Dictionary<AstNode, Dimension> result)
        {
            string funcName = (node.Value ?? "").ToLowerInvariant();
            if (node.Children != null && node.Children.Count > 0)
                AnalyzeNode(node.Children[0], result);

            return funcName switch
            {
                "sin" or "cos" or "tan" or "asin" or "acos" or "atan" or
                "sinh" or "cosh" or "tanh" => Dimension.Scalar,
                "exp" or "log" or "log2" or "log10" or "sqrt" or "cbrt" => Dimension.Scalar,
                "abs" or "sign" or "ceil" or "floor" or "round" => Dimension.Scalar,
                "min" or "max" => node.Children?.Count > 0 ? AnalyzeNode(node.Children[0], result) : Dimension.Scalar,
                "pow" => Dimension.Scalar,
                "clamp" => node.Children?.Count > 0 ? AnalyzeNode(node.Children[0], result) : Dimension.Scalar,
                "lerp" => node.Children?.Count > 0 ? AnalyzeNode(node.Children[0], result) : Dimension.Scalar,
                "grad_x" or "grad_y" or "grad_z" => node.Children?.Count > 0
                    ? AnalyzeNode(node.Children[0], result).Divide(Dimension.LengthD) : Dimension.Scalar,
                "laplacian" => node.Children?.Count > 0
                    ? AnalyzeNode(node.Children[0], result).Divide(Dimension.LengthD.Pow(2)) : Dimension.Scalar,
                "divergence" => node.Children?.Count > 0
                    ? AnalyzeNode(node.Children[0], result).Divide(Dimension.LengthD) : Dimension.Scalar,
                "curl_x" or "curl_y" or "curl_z" => node.Children?.Count > 0
                    ? AnalyzeNode(node.Children[0], result).Divide(Dimension.LengthD) : Dimension.Scalar,
                _ => Dimension.Scalar
            };
        }

        /// <summary>Check if an equation is dimensionally homogeneous.</summary>
        public bool IsDimensionallyHomogeneous(AstNode ast)
        {
            var dims = AnalyzeDimensions(ast);
            if (ast.Type == NodeType.BinaryExpression && ast.Value is "=" or "==")
            {
                if (dims.TryGetValue(ast.Left!, out var leftDim) && dims.TryGetValue(ast.Right!, out var rightDim))
                    return leftDim.IsCompatible(rightDim);
            }
            return true;
        }

        /// <summary>Get the dimension of the entire expression.</summary>
        public Dimension GetExpressionDimension(AstNode ast)
        {
            var dims = AnalyzeDimensions(ast);
            return dims.TryGetValue(ast, out var dim) ? dim : Dimension.Scalar;
        }

        /// <summary>Verify that a unit matches the expected dimension.</summary>
        public bool VerifyUnit(string expression, string expectedUnit)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            var exprDim = GetExpressionDimension(ast);
            var unit = _unitRegistry.GetUnit(expectedUnit);
            if (unit == null)
                return false;
            return exprDim.IsCompatible(unit.BaseDimension);
        }

        /// <summary>Get a list of compatible units for a dimension.</summary>
        public List<string> GetCompatibleUnits(Dimension dim)
        {
            var results = new List<string>();
            foreach (var unitName in _variableDimensions)
            {
                if (unitName.Value.IsCompatible(dim))
                    results.Add(unitName.Key);
            }
            return results;
        }
    }
}
