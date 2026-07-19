// Multi-provider LLM pipeline for Synapse (split from HybridLlmRouter.cs).

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using System.Timers;
using GDNN.Scene;

#nullable enable

namespace GDNN.Llm
{

    /// <summary>
    /// Engine for constructing prompts from templates with variable substitution,
    /// conditional sections, loops, few-shot example injection, and chain-of-thought
    /// prompt construction.
    /// </summary>
    public sealed class PromptTemplateEngine
    {
        private readonly ConcurrentDictionary<string, string> _compiledTemplates;
        private readonly ConcurrentDictionary<string, string> _templateHashes;
        private readonly Random _rng;

        /// <summary>Number of compiled templates cached.</summary>
        public int CompiledTemplateCount => _compiledTemplates.Count;

        /// <summary>
        /// Initializes a new PromptTemplateEngine.
        /// </summary>
        public PromptTemplateEngine()
        {
            _compiledTemplates = new ConcurrentDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            _templateHashes = new ConcurrentDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            _rng = new Random();
        }

        /// <summary>
        /// Renders a template by replacing {{variable}} placeholders with values.
        /// </summary>
        /// <param name="template">Template string with {{variable}} placeholders.</param>
        /// <param name="variables">Dictionary of variable names to values.</param>
        /// <returns>Rendered string with variables substituted.</returns>
        public string Render(string template, IReadOnlyDictionary<string, string> variables)
        {
            if (string.IsNullOrEmpty(template)) return string.Empty;
            if (variables == null || variables.Count == 0) return template;

            var result = template;
            foreach (var (key, value) in variables)
            {
                result = result.Replace("{{" + key + "}}", value ?? "");
            }

            result = ProcessConditionals(result, variables);
            result = ProcessLoops(result, variables);

            return result;
        }

        /// <summary>
        /// Processes conditional sections: {{#if variable}}...{{/if}}.
        /// </summary>
        private string ProcessConditionals(string template, IReadOnlyDictionary<string, string> variables)
        {
            var pattern = @"\{\{#if\s+(\w+)\}\}([\s\S]*?)\{\{/if\}\}";
            return Regex.Replace(template, pattern, match =>
            {
                var varName = match.Groups[1].Value;
                var content = match.Groups[2].Value;
                if (variables.TryGetValue(varName, out var val) &&
                    !string.IsNullOrEmpty(val) &&
                    val != "false" && val != "0")
                    return content;
                return "";
            });
        }

        /// <summary>
        /// Processes loop sections: {{#each items}}...{{/each}} where items is comma-separated.
        /// </summary>
        private string ProcessLoops(string template, IReadOnlyDictionary<string, string> variables)
        {
            var pattern = @"\{\{#each\s+(\w+)\}\}([\s\S]*?)\{\{/each\}\}";
            return Regex.Replace(template, pattern, match =>
            {
                var varName = match.Groups[1].Value;
                var itemTemplate = match.Groups[2].Value;

                if (!variables.TryGetValue(varName, out var csvValue) || string.IsNullOrEmpty(csvValue))
                    return "";

                var items = csvValue.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                var sb = new StringBuilder();

                foreach (var item in items)
                {
                    sb.Append(itemTemplate.Replace("{{this}}", item.Trim()));
                }

                return sb.ToString();
            });
        }

        /// <summary>
        /// Injects few-shot examples into the prompt.
        /// </summary>
        /// <param name="template">The template containing {{examples:count=N}}.</param>
        /// <param name="examples">Available examples to inject.</param>
        /// <param name="maxCount">Maximum number of examples to inject.</param>
        /// <returns>Template with examples injected.</returns>
        public string InjectExamples(
            string template,
            IReadOnlyList<(string Input, string Output)> examples,
            int maxCount = 3)
        {
            var pattern = @"\{\{examples:(?:count=(\d+))?\}\}";
            return Regex.Replace(template, pattern, match =>
            {
                int count = match.Groups[1].Success && int.TryParse(match.Groups[1].Value, out var c)
                    ? c : maxCount;
                count = Math.Min(count, examples.Count);

                var sb = new StringBuilder();
                for (int i = 0; i < count; i++)
                {
                    sb.AppendLine($"Example {i + 1}:");
                    sb.AppendLine($"Input: {examples[i].Input}");
                    sb.AppendLine($"Output: {examples[i].Output}");
                    sb.AppendLine();
                }
                return sb.ToString();
            });
        }

        /// <summary>
        /// Builds a chain-of-thought prompt from a question.
        /// </summary>
        /// <param name="question">The user's question.</param>
        /// <param name="context">Optional additional context.</param>
        /// <returns>A chain-of-thought prompt.</returns>
        public string BuildChainOfThoughtPrompt(string question, string? context = null)
        {
            var sb = new StringBuilder();
            sb.AppendLine("Let's approach this step by step.");
            if (!string.IsNullOrEmpty(context))
                sb.AppendLine($"Context: {context}");
            sb.AppendLine();
            sb.AppendLine($"Question: {question}");
            sb.AppendLine();
            sb.AppendLine("Step 1: Understand the problem.");
            sb.AppendLine("Step 2: Identify key information.");
            sb.AppendLine("Step 3: Apply relevant knowledge.");
            sb.AppendLine("Step 4: Formulate the answer.");
            sb.AppendLine();
            sb.AppendLine("Let me work through this:");
            return sb.ToString();
        }

        /// <summary>
        /// Builds a structured output prompt that instructs the LLM to return JSON.
        /// </summary>
        /// <param name="task">Description of what to generate.</param>
        /// <param name="jsonSchema">Expected JSON schema.</param>
        /// <returns>A structured output prompt.</returns>
        public string BuildStructuredOutputPrompt(string task, string jsonSchema)
        {
            return $"Task: {task}\n\n" +
                   $"You MUST respond with valid JSON that conforms to this schema:\n" +
                   $"```json\n{jsonSchema}\n```\n\n" +
                   $"Do not include any text outside the JSON response. " +
                   $"Do not use markdown code blocks. Return ONLY the raw JSON.";
        }

        /// <summary>
        /// Validates that a template has all required variables available.
        /// </summary>
        /// <param name="template">Template to validate.</param>
        /// <param name="availableVariables">Available variable names.</param>
        /// <returns>True if all required variables are available.</returns>
        public (bool IsValid, IReadOnlyList<string> MissingVariables) ValidateTemplate(
            string template,
            IEnumerable<string> availableVariables)
        {
            var required = ExtractVariables(template);
            var availableSet = new HashSet<string>(availableVariables, StringComparer.OrdinalIgnoreCase);
            var missing = required.Where(v => !availableSet.Contains(v)).ToList();
            return (missing.Count == 0, missing);
        }

        /// <summary>
        /// Extracts all variable names from a template.
        /// </summary>
        /// <param name="template">Template string.</param>
        /// <returns>List of variable names found in the template.</returns>
        public IReadOnlyList<string> ExtractVariables(string template)
        {
            if (string.IsNullOrEmpty(template)) return Array.Empty<string>();

            var matches = Regex.Matches(template, @"\{\{(\w+)\}\}");
            return matches.Select(m => m.Groups[1].Value)
                          .Distinct(StringComparer.OrdinalIgnoreCase)
                          .ToList();
        }

        /// <summary>
        /// Computes a hash of the template content for caching and versioning.
        /// </summary>
        /// <param name="template">Template content.</param>
        /// <returns>SHA-256 hash string.</returns>
        public static string ComputeTemplateHash(string template)
        {
            if (string.IsNullOrEmpty(template)) return "";
            using var sha = SHA256.Create();
            var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(template));
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }

        /// <summary>
        /// Serializes scene context into a natural language prompt.
        /// </summary>
        /// <param name="scene">Scene description to serialize.</param>
        /// <returns>Natural language description of the scene.</returns>
        public string SerializeSceneContext(SceneDescription scene)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Scene: {scene.Name}");
            if (!string.IsNullOrEmpty(scene.Description))
                sb.AppendLine($"Description: {scene.Description}");

            if (scene.Entities != null && scene.Entities.Count > 0)
            {
                sb.AppendLine("\nEntities:");
                foreach (var entity in scene.Entities)
                {
                    sb.AppendLine($"  - {entity.Name} ({entity.Type})");
                    if (entity.Position.HasValue)
                        sb.AppendLine($"    Position: ({entity.Position.Value.X:F1}, {entity.Position.Value.Y:F1}, {entity.Position.Value.Z:F1})");
                    if (!string.IsNullOrEmpty(entity.Material))
                        sb.AppendLine($"    Material: {entity.Material}");
                }
            }

            if (scene.Relationships != null && scene.Relationships.Count > 0)
            {
                sb.AppendLine("\nRelationships:");
                foreach (var rel in scene.Relationships)
                    sb.AppendLine($"  - {rel.Source} --[{rel.Type}]--> {rel.Target}");
            }

            if (!string.IsNullOrEmpty(scene.Lighting))
                sb.AppendLine($"\nLighting: {scene.Lighting}");
            if (!string.IsNullOrEmpty(scene.Camera))
                sb.AppendLine($"Camera: {scene.Camera}");

            return sb.ToString();
        }
    }
}
