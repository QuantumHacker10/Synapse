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

    /// <summary>Creates new physical laws from templates and parameter exploration.</summary>
    public sealed class LawInventor
    {
        private readonly LawLibrary _library;
        private readonly LawExpressionParser _parser;
        private readonly List<LawTemplate> _templates = new();

        public IReadOnlyList<LawTemplate> Templates => _templates;

        public LawInventor(LawLibrary library)
        {
            _library = library;
            _parser = new LawExpressionParser("");
            RegisterBuiltInTemplates();
        }

        private void RegisterBuiltInTemplates()
        {
            _templates.Add(new LawTemplate
            {
                Name = "Conservation Law",
                Category = "Conservation",
                ExpressionTemplate = "∂({var})/∂t + ∇·({var}*v) = {source}",
                VariableDescriptions = new() { ["var"] = "conserved quantity", ["v"] = "velocity", ["source"] = "source term" },
                ExpectedDimension = new Dimension(0, -3, -1, 0, 0, 0, 0, 0)
            });
            _templates.Add(new LawTemplate
            {
                Name = "Diffusion-Reaction",
                Category = "ReactionDiffusion",
                ExpressionTemplate = "∂u/∂t = D*∇²u + f(u)",
                VariableDescriptions = new() { ["u"] = "concentration", ["D"] = "diffusivity", ["f"] = "reaction rate" },
                ExpectedDimension = new Dimension(0, -3, -1, 0, 0, 0, 0, 0)
            });
            _templates.Add(new LawTemplate
            {
                Name = "Dispersion Relation",
                Category = "WaveDynamics",
                ExpressionTemplate = "ω² = {coeff}*k^n",
                VariableDescriptions = new() { ["ω"] = "angular frequency", ["k"] = "wavenumber", ["n"] = "power" },
                ExpectedDimension = new Dimension(0, 0, -2, 0, 0, 0, 0, 0)
            });
            _templates.Add(new LawTemplate
            {
                Name = "Power Law",
                Category = "Empirical",
                ExpressionTemplate = "{output} = {coeff}*{input}^{exponent}",
                VariableDescriptions = new() { ["output"] = "output", ["input"] = "input", ["coeff"] = "coefficient", ["exponent"] = "exponent" },
                ExpectedDimension = Dimension.Scalar
            });
            _templates.Add(new LawTemplate
            {
                Name = "Exponential Decay",
                Category = "Kinetics",
                ExpressionTemplate = "{var} = {initial}*exp(-{rate}*t)",
                VariableDescriptions = new() { ["var"] = "variable", ["initial"] = "initial value", ["rate"] = "decay rate" },
                ExpectedDimension = Dimension.Scalar
            });
            _templates.Add(new LawTemplate
            {
                Name = "Coupled Oscillators",
                Category = "Dynamics",
                ExpressionTemplate = "m*ẍ + c*ẋ + k*x = F_ext + {coupling}*y",
                VariableDescriptions = new() { ["m"] = "mass", ["c"] = "damping", ["k"] = "stiffness", ["F_ext"] = "external force", ["coupling"] = "coupling constant", ["y"] = "coupled variable" },
                ExpectedDimension = Dimension.Force
            });
        }

        /// <summary>Invent a law from a template with specific parameter values.</summary>
        public LawEntry InventFromTemplate(string templateName, Dictionary<string, float> parameters, string? lawId = null)
        {
            var template = _templates.FirstOrDefault(t => t.Name == templateName);
            if (template == null)
                throw new ArgumentException($"Template '{templateName}' not found");

            string expression = template.ExpressionTemplate;
            foreach (var kv in parameters)
            {
                expression = expression.Replace($"{{{kv.Key}}}", kv.Value.ToString(CultureInfo.InvariantCulture));
            }

            var entry = new LawEntry
            {
                Id = lawId ?? $"invented_{Guid.NewGuid().ToString("N")[..8]}",
                Name = $"Invented: {templateName}",
                Category = template.Category,
                Expression = expression,
                Description = $"Invented from template '{templateName}'",
                ResultDimension = template.ExpectedDimension
            };

            _library.Register(entry);
            return entry;
        }

        /// <summary>Explore parameter space to find valid laws.</summary>
        public List<LawEntry> ExploreParameterSpace(string templateName,
            Dictionary<string, (float Min, float Max, float Step)> parameterRanges,
            Func<string, bool> validator, int maxResults = 100)
        {
            var results = new List<LawEntry>();
            var template = _templates.FirstOrDefault(t => t.Name == templateName);
            if (template == null)
                return results;

            var paramNames = parameterRanges.Keys.ToList();
            var ranges = paramNames.Select(n => parameterRanges[n]).ToList();
            int[] indices = new int[paramNames.Count];
            int[] counts = ranges.Select(r => (int)((r.Max - r.Min) / r.Step) + 1).ToArray();
            int totalCombinations = counts.Aggregate(1, (a, b) => a * b);

            for (int combo = 0; combo < totalCombinations && results.Count < maxResults; combo++)
            {
                var parameters = new Dictionary<string, float>();
                int tempCombo = combo;
                for (int i = 0; i < paramNames.Count; i++)
                {
                    indices[i] = tempCombo % counts[i];
                    tempCombo /= counts[i];
                    float value = ranges[i].Min + indices[i] * ranges[i].Step;
                    parameters[paramNames[i]] = value;
                }

                string expression = template.ExpressionTemplate;
                foreach (var kv in parameters)
                    expression = expression.Replace($"{{{kv.Key}}}", kv.Value.ToString(CultureInfo.InvariantCulture));

                if (validator(expression))
                {
                    var entry = new LawEntry
                    {
                        Id = $"explored_{Guid.NewGuid().ToString("N")[..8]}",
                        Name = $"Explored: {templateName}",
                        Category = template.Category,
                        Expression = expression,
                        Description = $"Parameter exploration from '{templateName}'"
                    };
                    _library.Register(entry);
                    results.Add(entry);
                }
            }
            return results;
        }

        /// <summary>Generate variations of an existing law.</summary>
        public List<string> GenerateVariations(string expression, int count = 10)
        {
            var variations = new List<string>();
            var rng = new Random();
            var operations = new Func<string, string>[]
            {
                e => $"({e}) * (1 + 0.1*sin(t))",
                e => $"({e})^1.1",
                e => $"({e}) * exp(-0.01*t)",
                e => $"({e}) + 0.01*∇²({e})",
                e => $"({e}) * (1 - 0.05*rho/1000)",
                e => $"clamp({e}, -1000, 1000)",
                e => $"({e}) * (1 + 0.05*sign(sin(2*3.14159*x)))",
                e => $"tanh({e}/100)*100",
                e => $"({e}) * exp(-abs(x)/100)",
                e => $"({e}) * (1 + 0.1*cos(y*0.1))",
            };

            for (int i = 0; i < count; i++)
            {
                int opIdx = rng.Next(operations.Length);
                variations.Add(operations[opIdx](expression));
            }
            return variations;
        }

        /// <summary>Blend two law expressions.</summary>
        public string BlendLaws(string expressionA, string expressionB, float weightA = 0.5f)
        {
            float weightB = 1f - weightA;
            return $"({weightA.ToString(CultureInfo.InvariantCulture)}*({expressionA})) + ({weightB.ToString(CultureInfo.InvariantCulture)}*({expressionB}))";
        }

        /// <summary>Apply dimensional constraints to generate valid expressions.</summary>
        public List<string> ApplyDimensionalConstraint(string variableName, Dimension targetDim)
        {
            var results = new List<string>();
            var dimStr = targetDim.ToString();

            if (targetDim.IsCompatible(Dimension.Velocity))
                results.Add($"{variableName} = dx/dt");
            else if (targetDim.IsCompatible(Dimension.Acceleration))
                results.Add($"{variableName} = dv/dt");
            else if (targetDim.IsCompatible(Dimension.Force))
                results.Add($"{variableName} = m*{variableName}_accel");
            else if (targetDim.IsCompatible(Dimension.Energy))
                results.Add($"{variableName} = 0.5*m*v^2");
            else if (targetDim.IsCompatible(Dimension.Pressure))
                results.Add($"{variableName} = F/A");
            else if (targetDim.IsCompatible(Dimension.Density))
                results.Add($"{variableName} = m/V");
            else if (targetDim.Time > 0)
                results.Add($"{variableName} = t^{targetDim.Time.ToString(CultureInfo.InvariantCulture)}");
            else if (targetDim.Length > 0)
                results.Add($"{variableName} = x^{targetDim.Length.ToString(CultureInfo.InvariantCulture)}");
            else
                results.Add($"{variableName} = constant");

            return results;
        }
    }
}
