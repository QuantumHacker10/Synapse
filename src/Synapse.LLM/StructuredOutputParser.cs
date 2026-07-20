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
using Synapse.Infrastructure.Logging;

#nullable enable

namespace GDNN.Llm
{

    /// <summary>
    /// Parses and validates structured output from LLM responses, including JSON
    /// extraction from markdown, schema validation, repair of common LLM errors,
    /// and extraction of domain-specific structures.
    /// </summary>
    public static class StructuredOutputParser
    {
        private static readonly Regex JsonBlockPattern = new(
            @"```(?:json)?\s*\n([\s\S]*?)\n\s*```",
            RegexOptions.Compiled);

        private static readonly Regex JsonObjectPattern = new(
            @"(\{[\s\S]*\})",
            RegexOptions.Compiled);

        private static readonly Regex TrailingCommaPattern = new(
            @",\s*([}\]])",
            RegexOptions.Compiled);

        private static readonly Regex UnquotedKeyPattern = new(
            @"(?<=[{,]\s*)(\w+)\s*:",
            RegexOptions.Compiled);

        /// <summary>
        /// Parses JSON from an LLM response, extracting it from markdown code blocks if necessary.
        /// </summary>
        /// <typeparam name="T">Target type to deserialize into.</typeparam>
        /// <param name="rawText">Raw LLM response text.</param>
        /// <returns>Parsed and validated result.</returns>
        public static StructuredOutput<T> ParseJson<T>(string rawText) where T : class
        {
            if (string.IsNullOrWhiteSpace(rawText))
                return StructuredOutput<T>.Fail("Empty response", rawText);

            var json = ExtractJson(rawText);
            if (string.IsNullOrEmpty(json))
                return StructuredOutput<T>.Fail("No JSON found in response", rawText);

            try
            {
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    AllowTrailingCommas = true,
                    ReadCommentHandling = JsonCommentHandling.Skip
                };

                var data = JsonSerializer.Deserialize<T>(json, options);
                if (data == null)
                    return StructuredOutput<T>.Fail("Deserialization returned null", rawText);

                return StructuredOutput<T>.Ok(data, 0.9f, rawText);
            }
            catch (JsonException ex)
            {
                var repaired = RepairJson(json);
                try
                {
                    var data = JsonSerializer.Deserialize<T>(repaired);
                    if (data != null)
                        return StructuredOutput<T>.Ok(data, 0.7f, rawText);
                }
                catch (Exception repairEx)
                {
                    SynapseLogger.Default.Debug("StructuredOutputParser", "JSON repair attempt failed.", repairEx);
                }

                return StructuredOutput<T>.Fail($"JSON parse error: {ex.Message}", rawText);
            }
        }

        /// <summary>
        /// Extracts JSON string from LLM response text.
        /// </summary>
        /// <param name="text">Raw text.</param>
        /// <returns>Extracted JSON string, or null.</returns>
        public static string? ExtractJson(string text)
        {
            if (string.IsNullOrEmpty(text))
                return null;

            // Try code block first
            var blockMatch = JsonBlockPattern.Match(text);
            if (blockMatch.Success)
                return blockMatch.Groups[1].Value.Trim();

            // Try finding a JSON object directly
            var objMatch = JsonObjectPattern.Match(text);
            if (objMatch.Success)
                return objMatch.Groups[1].Value;

            return null;
        }

        /// <summary>
        /// Attempts to repair common JSON errors in LLM output.
        /// </summary>
        /// <param name="json">Potentially broken JSON string.</param>
        /// <returns>Repaired JSON string.</returns>
        public static string RepairJson(string json)
        {
            if (string.IsNullOrEmpty(json))
                return json;

            var result = json.Trim();

            // Remove markdown code block markers if present
            if (result.StartsWith("```"))
            {
                var firstNewline = result.IndexOf('\n');
                if (firstNewline >= 0)
                    result = result.Substring(firstNewline + 1);
                if (result.EndsWith("```"))
                    result = result.Substring(0, result.Length - 3);
                result = result.Trim();
            }

            // Fix trailing commas
            result = TrailingCommaPattern.Replace(result, "$1");

            // Fix unquoted keys (simple heuristic)
            result = UnquotedKeyPattern.Replace(result, match =>
            {
                var key = match.Groups[1].Value;
                if (key == "true" || key == "false" || key == "null" ||
                    double.TryParse(key, NumberStyles.Any, CultureInfo.InvariantCulture, out _))
                    return match.Value;
                return $"\"{key}\":";
            });

            // Ensure proper string escaping for common patterns
            result = result.Replace("\\\"", "\"").Replace("\"", "\\\"");

            return result;
        }

        /// <summary>
        /// Extracts a genome specification from natural language LLM output.
        /// </summary>
        /// <param name="text">LLM response describing a genome.</param>
        /// <returns>Extracted genome specification.</returns>
        public static StructuredOutput<GenomeSpecification> ExtractGenomeSpec(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return StructuredOutput<GenomeSpecification>.Fail("Empty input");

            var parameters = new Dictionary<string, object>();
            var name = "";
            var description = text;
            var type = "organoid";

            var nameMatch = Regex.Match(text, @"name[:\s]+[""']?([^""'\n]+)", RegexOptions.IgnoreCase);
            if (nameMatch.Success)
                name = nameMatch.Groups[1].Value.Trim();

            var typeMatch = Regex.Match(text, @"type[:\s]+[""']?(\w+)", RegexOptions.IgnoreCase);
            if (typeMatch.Success)
                type = typeMatch.Groups[1].Value.Trim();

            var paramMatches = Regex.Matches(text, @"(\w+)\s*[=:]\s*([\d.]+)", RegexOptions.IgnoreCase);
            foreach (Match m in paramMatches)
            {
                if (double.TryParse(m.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var val))
                    parameters[m.Groups[1].Value] = val;
            }

            var genome = new GenomeSpecification
            {
                Name = name,
                Description = description,
                Type = type,
                Parameters = parameters
            };

            return StructuredOutput<GenomeSpecification>.Ok(genome, 0.75f, text);
        }

        /// <summary>
        /// Extracts behavior tree nodes from LLM output.
        /// </summary>
        /// <param name="text">LLM response describing behavior.</param>
        /// <returns>List of extracted behavior tree nodes.</returns>
        public static StructuredOutput<IReadOnlyList<BehaviorTreeNode>> ExtractBehaviorTree(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return StructuredOutput<IReadOnlyList<BehaviorTreeNode>>.Fail("Empty input");

            var nodes = new List<BehaviorTreeNode>();

            var actionMatches = Regex.Matches(text, @"(?:action|do|perform)[:\s]+[""']?([^""'\n]+)",
                RegexOptions.IgnoreCase);
            foreach (Match m in actionMatches)
            {
                nodes.Add(new BehaviorTreeNode
                {
                    NodeType = "action",
                    Name = m.Groups[1].Value.Trim(),
                    Description = m.Groups[1].Value.Trim()
                });
            }

            var conditionMatches = Regex.Matches(text, @"(?:if|when|check|condition)[:\s]+[""']?([^""'\n]+)",
                RegexOptions.IgnoreCase);
            foreach (Match m in conditionMatches)
            {
                nodes.Add(new BehaviorTreeNode
                {
                    NodeType = "condition",
                    Name = m.Groups[1].Value.Trim(),
                    Description = m.Groups[1].Value.Trim()
                });
            }

            if (nodes.Count == 0)
            {
                nodes.Add(new BehaviorTreeNode
                {
                    NodeType = "sequence",
                    Name = "DefaultBehavior",
                    Description = text
                });
            }

            return StructuredOutput<IReadOnlyList<BehaviorTreeNode>>.Ok(nodes.AsReadOnly(), 0.6f, text);
        }

        /// <summary>
        /// Extracts material properties from natural language description.
        /// </summary>
        /// <param name="text">LLM response describing a material.</param>
        /// <returns>Extracted material properties.</returns>
        public static StructuredOutput<MaterialProperties> ExtractMaterialProperties(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return StructuredOutput<MaterialProperties>.Fail("Empty input");

            var material = new MaterialProperties { Name = "GeneratedMaterial" };

            var nameMatch = Regex.Match(text, @"(?:name|material)[:\s]+[""']?([^""'\n]+)", RegexOptions.IgnoreCase);
            if (nameMatch.Success)
                material = material with { Name = nameMatch.Groups[1].Value.Trim() };

            var colorMatch = Regex.Match(text, @"(?:color|colour)[:\s]+#?([0-9a-fA-F]{6})", RegexOptions.IgnoreCase);
            if (colorMatch.Success)
                material = material with { BaseColor = "#" + colorMatch.Groups[1].Value };

            var metallicMatch = Regex.Match(text, @"metallic[:\s]+([\d.]+)", RegexOptions.IgnoreCase);
            if (metallicMatch.Success && float.TryParse(metallicMatch.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var metallic))
                material = material with { Metallic = Math.Clamp(metallic, 0f, 1f) };

            var roughnessMatch = Regex.Match(text, @"roughness[:\s]+([\d.]+)", RegexOptions.IgnoreCase);
            if (roughnessMatch.Success && float.TryParse(roughnessMatch.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var roughness))
                material = material with { Roughness = Math.Clamp(roughness, 0f, 1f) };

            return StructuredOutput<MaterialProperties>.Ok(material, 0.7f, text);
        }

        /// <summary>
        /// Parses lighting parameters from JSON or key=value LLM text.
        /// </summary>
        public static bool TryParseLightingParams(string llmText, out LightingParams parameters)
        {
            parameters = new LightingParams();
            if (string.IsNullOrWhiteSpace(llmText))
                return false;

            var json = ExtractJson(llmText);
            if (!string.IsNullOrEmpty(json))
            {
                try
                {
                    using var doc = JsonDocument.Parse(json);
                    var root = doc.RootElement;
                    parameters = ReadLightingParamsFromJson(root);
                    return HasLightingSignal(parameters);
                }
                catch (JsonException)
                {
                    // Fall through to regex parsing.
                }
            }

            parameters = ParseLightingParamsFromText(llmText);
            return HasLightingSignal(parameters);
        }

        /// <summary>
        /// Parses a simple SDF primitive hint (sphere/box) from LLM text.
        /// </summary>
        public static bool TryParseSdfHint(string llmText, out SdfHint hint)
        {
            hint = new SdfHint();
            if (string.IsNullOrWhiteSpace(llmText))
                return false;

            var json = ExtractJson(llmText);
            if (!string.IsNullOrEmpty(json))
            {
                try
                {
                    using var doc = JsonDocument.Parse(json);
                    hint = ReadSdfHintFromJson(doc.RootElement);
                    return HasSdfSignal(hint);
                }
                catch (JsonException)
                {
                    // Fall through to regex parsing.
                }
            }

            hint = ParseSdfHintFromText(llmText);
            return HasSdfSignal(hint);
        }

        private static LightingParams ReadLightingParamsFromJson(JsonElement root)
        {
            var direction = TryReadVector3(root, "directionalDirection", "direction", "sunDirection");
            var color = TryReadString(root, "color", "lightColor", "sunColor");
            var intensity = TryReadFloat(root, "intensity", "lightIntensity", "sunIntensity");
            var fogDensity = TryReadFloat(root, "fogDensity", "fog", "volumeFogDensity");
            var enableClouds = TryReadBool(root, "enableClouds", "clouds");

            return new LightingParams
            {
                DirectionalDirection = direction,
                Color = color,
                Intensity = intensity,
                FogDensity = fogDensity,
                EnableClouds = enableClouds
            };
        }

        private static LightingParams ParseLightingParamsFromText(string text)
        {
            var direction = TryParseDirectionVector(text);
            var colorMatch = Regex.Match(text, @"(?:color|lightColor|sunColor)[:\s]+#?([0-9a-fA-F]{6})", RegexOptions.IgnoreCase);
            var intensityMatch = Regex.Match(text, @"(?:intensity|lightIntensity|sunIntensity)[:\s]+([\d.]+)", RegexOptions.IgnoreCase);
            var fogMatch = Regex.Match(text, @"(?:fogDensity|fog|volumeFogDensity)[:\s]+([\d.]+)", RegexOptions.IgnoreCase);
            var cloudsMatch = Regex.Match(text, @"(?:enableClouds|clouds)[:\s]+(true|false|yes|no|on|off)", RegexOptions.IgnoreCase);

            float? intensity = null;
            if (intensityMatch.Success &&
                float.TryParse(intensityMatch.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsedIntensity))
            {
                intensity = parsedIntensity;
            }

            float? fogDensity = null;
            if (fogMatch.Success &&
                float.TryParse(fogMatch.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsedFog))
            {
                fogDensity = parsedFog;
            }

            bool? enableClouds = null;
            if (cloudsMatch.Success)
            {
                enableClouds = ParseBoolToken(cloudsMatch.Groups[1].Value);
            }

            return new LightingParams
            {
                DirectionalDirection = direction,
                Color = colorMatch.Success ? "#" + colorMatch.Groups[1].Value : null,
                Intensity = intensity,
                FogDensity = fogDensity,
                EnableClouds = enableClouds
            };
        }

        private static SdfHint ReadSdfHintFromJson(JsonElement root)
        {
            var primitive = TryReadString(root, "primitive", "type", "sdfPrimitive") ?? "sphere";
            var center = TryReadVector3(root, "center", "position");
            var radius = TryReadFloat(root, "radius", "size");
            var size = TryReadVector3(root, "size", "halfExtents", "extents");

            return new SdfHint
            {
                Primitive = primitive,
                Center = center,
                Radius = radius,
                Size = size
            };
        }

        private static SdfHint ParseSdfHintFromText(string text)
        {
            var primitiveMatch = Regex.Match(text, @"(?:primitive|type|sdf)[:\s]+(sphere|box|cuboid)", RegexOptions.IgnoreCase);
            var centerMatch = Regex.Match(text, @"(?:center|position)[:\s]+\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?", RegexOptions.IgnoreCase);
            var radiusMatch = Regex.Match(text, @"(?:radius|size)[:\s]+([\d.]+)", RegexOptions.IgnoreCase);

            (float X, float Y, float Z)? center = null;
            if (centerMatch.Success &&
                float.TryParse(centerMatch.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var cx) &&
                float.TryParse(centerMatch.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var cy) &&
                float.TryParse(centerMatch.Groups[3].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var cz))
            {
                center = (cx, cy, cz);
            }

            float? radius = null;
            if (radiusMatch.Success &&
                float.TryParse(radiusMatch.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsedRadius))
            {
                radius = parsedRadius;
            }

            return new SdfHint
            {
                Primitive = primitiveMatch.Success ? primitiveMatch.Groups[1].Value.ToLowerInvariant() : "",
                Center = center,
                Radius = radius
            };
        }

        private static bool HasLightingSignal(LightingParams parameters) =>
            parameters.DirectionalDirection.HasValue ||
            !string.IsNullOrWhiteSpace(parameters.Color) ||
            parameters.Intensity.HasValue ||
            parameters.FogDensity.HasValue ||
            parameters.EnableClouds.HasValue;

        /// <summary>
        /// Requires a geometric cue (center/radius/size). A bare default
        /// "sphere" primitive without measurements is not a signal.
        /// </summary>
        private static bool HasSdfSignal(SdfHint hint) =>
            hint.Center.HasValue ||
            hint.Radius.HasValue ||
            hint.Size.HasValue;

        private static (float X, float Y, float Z)? TryParseDirectionVector(string text)
        {
            var bracketMatch = Regex.Match(
                text,
                @"(?:direction|directionalDirection|sunDirection)[:\s]+\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?",
                RegexOptions.IgnoreCase);
            if (bracketMatch.Success &&
                float.TryParse(bracketMatch.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) &&
                float.TryParse(bracketMatch.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var y) &&
                float.TryParse(bracketMatch.Groups[3].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var z))
            {
                return (x, y, z);
            }

            return null;
        }

        private static (float X, float Y, float Z)? TryReadVector3(JsonElement root, params string[] names)
        {
            foreach (var name in names)
            {
                if (!root.TryGetProperty(name, out var element))
                    continue;

                if (element.ValueKind == JsonValueKind.Array && element.GetArrayLength() >= 3)
                {
                    return (
                        (float)element[0].GetDouble(),
                        (float)element[1].GetDouble(),
                        (float)element[2].GetDouble());
                }

                if (element.ValueKind == JsonValueKind.Object)
                {
                    if (element.TryGetProperty("x", out var x) &&
                        element.TryGetProperty("y", out var y) &&
                        element.TryGetProperty("z", out var z))
                    {
                        return ((float)x.GetDouble(), (float)y.GetDouble(), (float)z.GetDouble());
                    }
                }
            }

            return null;
        }

        private static string? TryReadString(JsonElement root, params string[] names)
        {
            foreach (var name in names)
            {
                if (root.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.String)
                    return element.GetString();
            }

            return null;
        }

        private static float? TryReadFloat(JsonElement root, params string[] names)
        {
            foreach (var name in names)
            {
                if (!root.TryGetProperty(name, out var element))
                    continue;

                if (element.ValueKind == JsonValueKind.Number)
                    return (float)element.GetDouble();

                if (element.ValueKind == JsonValueKind.String &&
                    float.TryParse(element.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed))
                {
                    return parsed;
                }
            }

            return null;
        }

        private static bool? TryReadBool(JsonElement root, params string[] names)
        {
            foreach (var name in names)
            {
                if (!root.TryGetProperty(name, out var element))
                    continue;

                if (element.ValueKind == JsonValueKind.True)
                    return true;
                if (element.ValueKind == JsonValueKind.False)
                    return false;

                if (element.ValueKind == JsonValueKind.String)
                    return ParseBoolToken(element.GetString());
            }

            return null;
        }

        private static bool? ParseBoolToken(string? token) => token?.Trim().ToLowerInvariant() switch
        {
            "true" or "yes" or "on" or "1" => true,
            "false" or "no" or "off" or "0" => false,
            _ => null
        };

        /// <summary>
        /// Extracts entities and their types from text using pattern matching.
        /// </summary>
        /// <param name="text">Input text.</param>
        /// <returns>Extracted entities.</returns>
        public static IReadOnlyList<EntityExtractionResult> ExtractEntities(string text)
        {
            if (string.IsNullOrEmpty(text))
                return Array.Empty<EntityExtractionResult>();

            var entities = new List<EntityExtractionResult>();

            // Extract quoted strings as entities
            var quoteMatches = Regex.Matches(text, @"[""']([^""']+)[""']");
            foreach (Match m in quoteMatches)
            {
                entities.Add(new EntityExtractionResult
                {
                    Text = m.Groups[1].Value,
                    Type = "quoted_string",
                    StartIndex = m.Groups[1].Index,
                    EndIndex = m.Groups[1].Index + m.Groups[1].Length,
                    Confidence = 0.9f
                });
            }

            // Extract numeric values
            var numberMatches = Regex.Matches(text, @"\b(\d+(?:\.\d+)?)\b");
            foreach (Match m in numberMatches)
            {
                entities.Add(new EntityExtractionResult
                {
                    Text = m.Groups[1].Value,
                    Type = "number",
                    StartIndex = m.Groups[1].Index,
                    EndIndex = m.Groups[1].Index + m.Groups[1].Length,
                    NormalizedValue = m.Groups[1].Value,
                    Confidence = 0.95f
                });
            }

            // Extract color references
            var colorMatches = Regex.Matches(text, @"#([0-9a-fA-F]{3,8})\b");
            foreach (Match m in colorMatches)
            {
                entities.Add(new EntityExtractionResult
                {
                    Text = "#" + m.Groups[1].Value,
                    Type = "color",
                    StartIndex = m.Groups[0].Index,
                    EndIndex = m.Groups[0].Index + m.Groups[0].Length,
                    Confidence = 0.95f
                });
            }

            return entities;
        }

        /// <summary>
        /// Validates a JSON string against a basic schema (property presence, types).
        /// </summary>
        /// <param name="json">JSON string to validate.</param>
        /// <param name="requiredProperties">Required property names.</param>
        /// <returns>Validation result.</returns>
        public static (bool IsValid, IReadOnlyList<string> Errors) ValidateJson(
            string json,
            IReadOnlyList<string>? requiredProperties = null)
        {
            var errors = new List<string>();

            try
            {
                using var doc = JsonDocument.Parse(json);

                if (requiredProperties != null)
                {
                    foreach (var prop in requiredProperties)
                    {
                        if (!doc.RootElement.TryGetProperty(prop, out _))
                            errors.Add($"Missing required property: '{prop}'");
                    }
                }
            }
            catch (JsonException ex)
            {
                errors.Add($"Invalid JSON: {ex.Message}");
            }

            return (errors.Count == 0, errors);
        }
    }
}
