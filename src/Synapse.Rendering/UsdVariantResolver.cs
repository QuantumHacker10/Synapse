using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace GDNN.Rendering.MeshIO;

/// <summary>
/// Resolves USD variantSets: selects the active variant body (from config or first authored)
/// and returns USDA text with inactive variant blocks removed.
/// </summary>
public static class UsdVariantResolver
{
    private static readonly Regex VariantSetBlock = new(
        @"variantSet\s+""([^""]+)""\s*=\s*\{",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>
    /// Applies variant selections from <paramref name="config"/>. When a set is missing from config,
    /// the first authored variant name is chosen.
    /// </summary>
    public static string ApplyVariants(string usdaText, MeshLoadConfig? config)
    {
        if (string.IsNullOrEmpty(usdaText))
            return usdaText ?? "";

        var selections = config?.UsdVariantSelections;
        var result = usdaText;
        // Iterate until no variantSet remains (supports one nesting level practically).
        for (int guard = 0; guard < 8; guard++)
        {
            var m = VariantSetBlock.Match(result);
            if (!m.Success)
                break;

            string setName = m.Groups[1].Value;
            int braceOpen = m.Index + m.Length - 1; // points at '{'
            if (!TryExtractBalanced(result, braceOpen, out int braceClose, out string setBody))
                break;

            var variants = ParseVariantBodies(setBody);
            if (variants.Count == 0)
            {
                // Strip empty variantSet
                result = result.Remove(m.Index, braceClose - m.Index + 1);
                continue;
            }

            string chosen;
            if (selections != null && selections.TryGetValue(setName, out var sel) &&
                variants.ContainsKey(sel))
                chosen = sel;
            else
                chosen = variants.Keys.First();

            string replacement = variants[chosen];
            result = result.Remove(m.Index, braceClose - m.Index + 1).Insert(m.Index, replacement);
        }

        return result;
    }

    public static Dictionary<string, string> ParseVariantBodies(string setBody)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        // variant names: "High" { ... }
        var nameRe = new Regex(@"""([^""]+)""\s*\{", RegexOptions.Compiled);
        int search = 0;
        while (search < setBody.Length)
        {
            var m = nameRe.Match(setBody, search);
            if (!m.Success)
                break;
            int open = m.Index + m.Length - 1;
            if (!TryExtractBalanced(setBody, open, out int close, out string body))
                break;
            map[m.Groups[1].Value] = body;
            search = close + 1;
        }

        return map;
    }

    public static IReadOnlyList<string> ListVariantSetNames(string usdaText)
    {
        var list = new List<string>();
        foreach (Match m in VariantSetBlock.Matches(usdaText))
            list.Add(m.Groups[1].Value);
        return list;
    }

    private static bool TryExtractBalanced(string text, int openBraceIndex, out int closeIndex, out string inner)
    {
        closeIndex = -1;
        inner = "";
        if (openBraceIndex < 0 || openBraceIndex >= text.Length || text[openBraceIndex] != '{')
            return false;
        int depth = 0;
        for (int i = openBraceIndex; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '{')
                depth++;
            else if (c == '}')
            {
                depth--;
                if (depth == 0)
                {
                    closeIndex = i;
                    inner = text.Substring(openBraceIndex + 1, i - openBraceIndex - 1);
                    return true;
                }
            }
        }

        return false;
    }
}
