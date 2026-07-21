using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;

namespace GDNN.Rendering.MeshIO;

/// <summary>Parses UsdPreviewSurface materials and material:binding relations from USDA text.</summary>
public static class UsdMaterialParser
{
    private static readonly Regex MaterialDef = new(
        @"def\s+Material\s+""([^""]+)""\s*\{",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex ShaderDef = new(
        @"def\s+Shader\s+""([^""]+)""\s*\{",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex Binding = new(
        @"rel\s+material:binding(?:\s*:\s*\w+)?\s*=\s*<([^>]+)>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    public static List<MeshMaterial> ParseMaterials(string usdaText)
    {
        var materials = new List<MeshMaterial>();
        if (string.IsNullOrEmpty(usdaText))
            return materials;

        // Prefer UsdPreviewSurface shader inputs; fall back to Material display name with defaults.
        foreach (Match shader in ShaderDef.Matches(usdaText))
        {
            int bodyStart = shader.Index + shader.Length;
            string body = ExtractBalancedBlock(usdaText, bodyStart - 1);
            if (body.IndexOf("UsdPreviewSurface", StringComparison.OrdinalIgnoreCase) < 0 &&
                body.IndexOf("inputs:diffuseColor", StringComparison.OrdinalIgnoreCase) < 0)
                continue;

            var mat = new MeshMaterial { Name = shader.Groups[1].Value };
            mat.BaseColor = ParseColor3(body, "inputs:diffuseColor", mat.BaseColor);
            mat.Roughness = ParseFloat(body, "inputs:roughness", mat.Roughness);
            mat.Metallic = ParseFloat(body, "inputs:metallic", mat.Metallic);
            mat.Opacity = ParseFloat(body, "inputs:opacity", mat.Opacity);
            mat.EmissiveColor = ParseColor3(body, "inputs:emissiveColor", mat.EmissiveColor);
            materials.Add(mat);
        }

        if (materials.Count == 0)
        {
            foreach (Match m in MaterialDef.Matches(usdaText))
            {
                materials.Add(new MeshMaterial
                {
                    Name = m.Groups[1].Value,
                    BaseColor = new Vector3(0.8f, 0.8f, 0.8f)
                });
            }
        }

        return materials;
    }

    public static string? ParseBindingPath(string usdaText)
    {
        var m = Binding.Match(usdaText);
        return m.Success ? m.Groups[1].Value.Trim() : null;
    }

    public static int ResolveMaterialIndex(IReadOnlyList<MeshMaterial> materials, string? bindingPath)
    {
        if (materials.Count == 0)
            return 0;
        if (string.IsNullOrWhiteSpace(bindingPath))
            return 0;

        var last = bindingPath.Trim().Trim('/');
        if (last.Contains('/'))
            last = last[(last.LastIndexOf('/') + 1)..];

        for (int i = 0; i < materials.Count; i++)
        {
            if (string.Equals(materials[i].Name, last, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        return 0;
    }

    public static Vector3 ParseColor3(string text, string key, Vector3 fallback)
    {
        int idx = text.IndexOf(key, StringComparison.OrdinalIgnoreCase);
        if (idx < 0)
            return fallback;
        int open = text.IndexOf('(', idx);
        int close = open >= 0 ? text.IndexOf(')', open) : -1;
        if (open < 0 || close < open)
            return fallback;
        var parts = text.Substring(open + 1, close - open - 1)
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length < 3)
            return fallback;
        if (float.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out var r) &&
            float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var g) &&
            float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var b))
            return new Vector3(r, g, b);
        return fallback;
    }

    public static float ParseFloat(string text, string key, float fallback)
    {
        int idx = text.IndexOf(key, StringComparison.OrdinalIgnoreCase);
        if (idx < 0)
            return fallback;
        int eq = text.IndexOf('=', idx);
        if (eq < 0)
            return fallback;
        int end = eq + 1;
        while (end < text.Length && char.IsWhiteSpace(text[end]))
            end++;
        int start = end;
        while (end < text.Length && (char.IsDigit(text[end]) || text[end] is '.' or '-' or '+' or 'e' or 'E'))
            end++;
        if (start == end)
            return fallback;
        return float.TryParse(text[start..end], NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
            ? v
            : fallback;
    }

    private static string ExtractBalancedBlock(string text, int openBraceIndex)
    {
        if (openBraceIndex < 0 || openBraceIndex >= text.Length || text[openBraceIndex] != '{')
            return "";
        int depth = 0;
        for (int i = openBraceIndex; i < text.Length; i++)
        {
            if (text[i] == '{')
                depth++;
            else if (text[i] == '}')
            {
                depth--;
                if (depth == 0)
                    return text.Substring(openBraceIndex + 1, i - openBraceIndex - 1);
            }
        }

        return text[(openBraceIndex + 1)..];
    }
}
