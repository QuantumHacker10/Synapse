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

    /// <summary>Engine for modifying law expressions through various methods.</summary>
    public sealed class LawModificationEngine
    {
        private readonly LawLibrary _library;
        private readonly List<ModificationRecord> _history = new();
        public IReadOnlyList<ModificationRecord> History => _history;

        public LawModificationEngine(LawLibrary library) { _library = library; }

        /// <summary>Modify a constant value in the expression.</summary>
        public string ModifyConstant(string expression, string constantName, float newValue)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            return ReplaceConstantInAst(ast, constantName, newValue);
        }

        private string ReplaceConstantInAst(AstNode node, string name, float newValue)
        {
            if (node == null)
                return "";
            return node.Type switch
            {
                NodeType.NumberLiteral => node.Value ?? "0",
                NodeType.Identifier => node.Value == name
                    ? newValue.ToString(CultureInfo.InvariantCulture) : node.Value ?? "",
                NodeType.BinaryExpression =>
                    $"({ReplaceConstantInAst(node.Left!, name, newValue)} {node.Value} {ReplaceConstantInAst(node.Right!, name, newValue)})",
                NodeType.UnaryExpression =>
                    $"({node.Value}{ReplaceConstantInAst(node.Left!, name, newValue)})",
                NodeType.FunctionCall =>
                    $"{node.Value}({string.Join(", ", node.Children?.Select(c => ReplaceConstantInAst(c, name, newValue)) ?? Array.Empty<string>())})",
                NodeType.FieldAccess => $"field.{node.Value}",
                NodeType.TernaryExpression =>
                    $"({ReplaceConstantInAst(node.Left!, name, newValue)} ? {ReplaceConstantInAst(node.Right!, name, newValue)} : {ReplaceConstantInAst(node.Middle!, name, newValue)})",
                _ => ""
            };
        }

        /// <summary>Replace one operator with another.</summary>
        public string ModifyOperator(string expression, string targetOp, string replacementOp)
        {
            return expression.Replace(targetOp, replacementOp);
        }

        /// <summary>Add a coupling term to the expression.</summary>
        public string AddCouplingTerm(string expression, string couplingTerm, float couplingStrength = 1.0f)
        {
            string strengthStr = couplingStrength == 1.0f ? "" : couplingStrength.ToString(CultureInfo.InvariantCulture) + "*";
            return $"({expression}) + {strengthStr}({couplingTerm})";
        }

        /// <summary>Scale a variable in the expression.</summary>
        public string ScaleVariable(string expression, string variableName, float scaleFactor)
        {
            return expression.Replace(variableName, $"({scaleFactor}*{variableName})");
        }

        /// <summary>Invert the sign of a term.</summary>
        public string InvertSign(string expression, string termName)
        {
            return expression.Replace(termName, $"(-{termName})");
        }

        /// <summary>Exponentiate a variable.</summary>
        public string Exponentiate(string expression, string variableName, float exponent)
        {
            return expression.Replace(variableName, $"({variableName}^{exponent.ToString(CultureInfo.InvariantCulture)})");
        }

        /// <summary>Add damping term to an equation.</summary>
        public string AddDamping(string expression, string variableName, float dampingCoeff)
        {
            string dampingTerm = $"{dampingCoeff.ToString(CultureInfo.InvariantCulture)}*{variableName}";
            return $"({expression}) - {dampingTerm}";
        }

        /// <summary>Add external forcing term.</summary>
        public string AddForcing(string expression, string forcingExpression, float strength = 1.0f)
        {
            string strengthStr = strength == 1.0f ? "" : strength.ToString(CultureInfo.InvariantCulture) + "*";
            return $"({expression}) + {strengthStr}({forcingExpression})";
        }

        /// <summary>Linearize around an operating point.</summary>
        public string Linearize(string expression, string variableName, float operatingPoint)
        {
            string result = expression;
            result = result.Replace($"{variableName}^2", $"(2*{operatingPoint.ToString(CultureInfo.InvariantCulture)}*{variableName})");
            result = result.Replace($"{variableName}^3", $"(3*{MathF.Pow(operatingPoint, 2).ToString(CultureInfo.InvariantCulture)}*{variableName})");
            return result;
        }

        /// <summary>Add nonlinearity (power law).</summary>
        public string Nonlinearize(string expression, string variableName, float exponent)
        {
            return expression.Replace(variableName, $"abs({variableName})^{exponent.ToString(CultureInfo.InvariantCulture)}*sign({variableName})");
        }

        /// <summary>Apply a modification from a LawModification object.</summary>
        public string ApplyModification(string expression, LawModification modification)
        {
            string result = modification.Type switch
            {
                ModificationType.ModifyConstant when modification.ConstantValue.HasValue && modification.VariableName != null =>
                    ModifyConstant(expression, modification.VariableName, modification.ConstantValue.Value),
                ModificationType.ModifyOperator when modification.TargetExpression != null && modification.ReplacementExpression != null =>
                    ModifyOperator(expression, modification.TargetExpression, modification.ReplacementExpression),
                ModificationType.AddCouplingTerm when modification.CouplingTerm != null =>
                    AddCouplingTerm(expression, modification.CouplingTerm, modification.ScaleFactor),
                ModificationType.ScaleVariable when modification.VariableName != null =>
                    ScaleVariable(expression, modification.VariableName, modification.ScaleFactor),
                ModificationType.ReplaceSubexpression when modification.TargetExpression != null && modification.ReplacementExpression != null =>
                    expression.Replace(modification.TargetExpression, modification.ReplacementExpression),
                ModificationType.SimplifyExpression => SimplifyExpression(expression),
                ModificationType.InvertSign when modification.VariableName != null =>
                    InvertSign(expression, modification.VariableName),
                ModificationType.Exponentiate when modification.VariableName != null && modification.ConstantValue.HasValue =>
                    Exponentiate(expression, modification.VariableName, modification.ConstantValue.Value),
                ModificationType.AddDamping when modification.VariableName != null && modification.ConstantValue.HasValue =>
                    AddDamping(expression, modification.VariableName, modification.ConstantValue.Value),
                ModificationType.AddForcing when modification.CouplingTerm != null =>
                    AddForcing(expression, modification.CouplingTerm, modification.ScaleFactor),
                ModificationType.Linearize when modification.VariableName != null && modification.ConstantValue.HasValue =>
                    Linearize(expression, modification.VariableName, modification.ConstantValue.Value),
                ModificationType.Nonlinearize when modification.VariableName != null && modification.ConstantValue.HasValue =>
                    Nonlinearize(expression, modification.VariableName, modification.ConstantValue.Value),
                _ => expression
            };

            _history.Add(new ModificationRecord
            {
                Modification = modification,
                OriginalExpression = expression,
                ResultExpression = result,
                Timestamp = DateTime.UtcNow
            });
            return result;
        }

        /// <summary>Apply a sequence of modifications.</summary>
        public string ApplyModifications(string expression, IReadOnlyList<LawModification> modifications)
        {
            string result = expression;
            foreach (var mod in modifications)
                result = ApplyModification(result, mod);
            return result;
        }

        /// <summary>Simplify an expression (basic algebraic simplifications).</summary>
        public string SimplifyExpression(string expression)
        {
            string result = expression;
            result = System.Text.RegularExpressions.Regex.Replace(result, @"--", "+");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"(\d+\.?\d*)\s*\*\s*1\b", "");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\b1\s*\*\s*(\d+\.?\d*)", "");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"(\d+\.?\d*)\s*\*\s*0\b", "0");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\b0\s*\*\s*(\d+\.?\d*)", "0");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\+\s*0\b", "");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\b0\s*\+", "");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\^\s*1\b", "");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"(\w+)\s*\^\s*0", "1");
            result = System.Text.RegularExpressions.Regex.Replace(result, @"\(\s*(\w+)\s*\)", "");
            return result;
        }

        /// <summary>Parse natural language modification instruction and apply it.</summary>
        public string ApplyNaturalLanguageModification(string expression, string instruction)
        {
            string lower = instruction.ToLowerInvariant();

            var increaseMatch = System.Text.RegularExpressions.Regex.Match(lower, @"increase\s+(\w+)\s+by\s+(\d+)%");
            if (increaseMatch.Success)
            {
                string varName = increaseMatch.Groups[1].Value;
                float percent = float.Parse(increaseMatch.Groups[2].Value) / 100f;
                return ScaleVariable(expression, varName, 1f + percent);
            }

            var decreaseMatch = System.Text.RegularExpressions.Regex.Match(lower, @"decrease\s+(\w+)\s+by\s+(\d+)%");
            if (decreaseMatch.Success)
            {
                string varName = decreaseMatch.Groups[1].Value;
                float percent = float.Parse(decreaseMatch.Groups[2].Value) / 100f;
                return ScaleVariable(expression, varName, 1f - percent);
            }

            var setMatch = System.Text.RegularExpressions.Regex.Match(lower, @"set\s+(\w+)\s+to\s+([\d.]+)");
            if (setMatch.Success)
            {
                string varName = setMatch.Groups[1].Value;
                float value = float.Parse(setMatch.Groups[2].Value, CultureInfo.InvariantCulture);
                return ModifyConstant(expression, varName, value);
            }

            var replaceMatch = System.Text.RegularExpressions.Regex.Match(lower, @"replace\s+(\w+)\s+with\s+(\w+)");
            if (replaceMatch.Success)
            {
                string target = replaceMatch.Groups[1].Value;
                string replacement = replaceMatch.Groups[2].Value;
                return ModifyOperator(expression, target, replacement);
            }

            var couplingMatch = System.Text.RegularExpressions.Regex.Match(lower, @"add\s+coupling\s+with\s+(\w+)");
            if (couplingMatch.Success)
            {
                string coupling = couplingMatch.Groups[1].Value;
                return AddCouplingTerm(expression, coupling);
            }

            var dampingMatch = System.Text.RegularExpressions.Regex.Match(lower, @"add\s+damping\s+to\s+(\w+)\s+with\s+coefficient\s+([\d.]+)");
            if (dampingMatch.Success)
            {
                string varName = dampingMatch.Groups[1].Value;
                float coeff = float.Parse(dampingMatch.Groups[2].Value, CultureInfo.InvariantCulture);
                return AddDamping(expression, varName, coeff);
            }

            var forceMatch = System.Text.RegularExpressions.Regex.Match(lower, @"add\s+forcing\s+(.+)");
            if (forceMatch.Success)
            {
                string forcing = forceMatch.Groups[1].Value;
                return AddForcing(expression, forcing);
            }

            var linMatch = System.Text.RegularExpressions.Regex.Match(lower, @"linearize\s+(\w+)\s+around\s+([\d.]+)");
            if (linMatch.Success)
            {
                string varName = linMatch.Groups[1].Value;
                float op = float.Parse(linMatch.Groups[2].Value, CultureInfo.InvariantCulture);
                return Linearize(expression, varName, op);
            }

            var invMatch = System.Text.RegularExpressions.Regex.Match(lower, @"invert\s+sign\s+of\s+(\w+)");
            if (invMatch.Success)
                return InvertSign(expression, invMatch.Groups[1].Value);

            if (lower.Contains("simplify"))
                return SimplifyExpression(expression);
            if (lower.Contains("double"))
            {
                var doubleVarMatch = System.Text.RegularExpressions.Regex.Match(lower, @"double\s+(\w+)");
                if (doubleVarMatch.Success)
                    return ScaleVariable(expression, doubleVarMatch.Groups[1].Value, 2f);
            }
            if (lower.Contains("half"))
            {
                var halfVarMatch = System.Text.RegularExpressions.Regex.Match(lower, @"half\s+(\w+)");
                if (halfVarMatch.Success)
                    return ScaleVariable(expression, halfVarMatch.Groups[1].Value, 0.5f);
            }

            return expression;
        }

        /// <summary>Get modification history for an expression.</summary>
        public IReadOnlyList<ModificationRecord> GetHistoryForExpression(string expression)
        {
            return _history.Where(r => r.OriginalExpression == expression || r.ResultExpression == expression).ToList();
        }

        /// <summary>Undo the last modification.</summary>
        public string? UndoLastModification()
        {
            if (_history.Count == 0)
                return null;
            var last = _history[^1];
            _history.RemoveAt(_history.Count - 1);
            return last.OriginalExpression;
        }
    }    // =========================================================================
}
