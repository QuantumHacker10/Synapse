using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;

namespace GDNN.Rendering.MeshIO;

/// <summary>UDIM tile map helpers (&lt;UDIM&gt; → 1001+).</summary>
public static class UsdUdim
{
    public const int BaseTile = 1001;

    /// <summary>
    /// Expands a path containing <c>&lt;UDIM&gt;</c> / <c>&lt;udim&gt;</c> into concrete tiles
    /// that exist under <paramref name="baseDirectory"/> (or returns the single resolved path).
    /// </summary>
    public static Dictionary<int, string> ExpandTiles(string pathTemplate, string? baseDirectory, int maxTiles = 100)
    {
        var map = new Dictionary<int, string>();
        if (string.IsNullOrWhiteSpace(pathTemplate))
            return map;

        bool hasToken = pathTemplate.Contains("<UDIM>", StringComparison.OrdinalIgnoreCase) ||
                        pathTemplate.Contains("%(UDIM)", StringComparison.OrdinalIgnoreCase) ||
                        pathTemplate.Contains("<udim>", StringComparison.Ordinal);
        if (!hasToken)
        {
            var single = UsdMaterialParser.ResolveTexturePath(pathTemplate, baseDirectory);
            map[BaseTile] = single;
            return map;
        }

        for (int i = 0; i < maxTiles; i++)
        {
            int tile = BaseTile + i;
            string candidate = pathTemplate
                .Replace("<UDIM>", tile.ToString(CultureInfo.InvariantCulture), StringComparison.OrdinalIgnoreCase)
                .Replace("<udim>", tile.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal)
                .Replace("%(UDIM)", tile.ToString(CultureInfo.InvariantCulture), StringComparison.OrdinalIgnoreCase);
            string resolved = UsdMaterialParser.ResolveTexturePath(candidate, baseDirectory);
            if (File.Exists(resolved))
                map[tile] = resolved;
            else if (i == 0)
                map[tile] = resolved; // keep 1001 even if missing (authoring hint)
            else
                break; // stop at first gap after 1001
        }

        return map;
    }

    public static int TileFromUv(Vector2 uv)
    {
        int u = (int)MathF.Floor(uv.X);
        int v = (int)MathF.Floor(uv.Y);
        return BaseTile + u + v * 10;
    }
}

/// <summary>UsdSkel blend shape (morph target) deltas.</summary>
public sealed class MeshBlendShape
{
    public string Name { get; set; } = "";
    public List<Vector3> DeltaPositions { get; } = new();
    public List<Vector3> DeltaNormals { get; } = new();
    public float DefaultWeight { get; set; }

    /// <summary>Applies weighted deltas onto <paramref name="positions"/> (in-place).</summary>
    public void Apply(IList<Vector3> positions, float weight)
    {
        int n = Math.Min(positions.Count, DeltaPositions.Count);
        for (int i = 0; i < n; i++)
            positions[i] += DeltaPositions[i] * weight;
    }
}

/// <summary>Parses UsdSkel BlendShape prims and mesh blendShape bindings.</summary>
public static class UsdSkelBlendShapeParser
{
    private static readonly Regex BlendShapeDef = new(
        @"def\s+BlendShape\s+""([^""]+)""\s*\{",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex BlendShapeBinding = new(
        @"rel\s+skel:blendShapes\s*=\s*\[([^\]]*)\]",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    public static List<MeshBlendShape> ParseBlendShapes(string usdaText)
    {
        var list = new List<MeshBlendShape>();
        if (string.IsNullOrEmpty(usdaText))
            return list;

        foreach (Match m in BlendShapeDef.Matches(usdaText))
        {
            string name = m.Groups[1].Value;
            string body = ExtractBlock(usdaText, m.Index + m.Length - 1);
            var shape = new MeshBlendShape { Name = name };
            shape.DeltaPositions.AddRange(ParsePointArray(body, "offsets") 
                .Concat(ParsePointArray(body, "point3f[] offsets")));
            if (shape.DeltaPositions.Count == 0)
                shape.DeltaPositions.AddRange(ParsePointArrayLoose(body, "offsets"));
            shape.DeltaNormals.AddRange(ParsePointArrayLoose(body, "normalOffsets"));
            list.Add(shape);
        }

        return list;
    }

    public static IReadOnlyList<string> ParseBlendShapeBindingPaths(string usdaText)
    {
        var list = new List<string>();
        var m = BlendShapeBinding.Match(usdaText);
        if (!m.Success)
            return list;
        foreach (Match at in Regex.Matches(m.Groups[1].Value, @"<([^>]+)>"))
            list.Add(at.Groups[1].Value.Trim());
        return list;
    }

    private static List<Vector3> ParsePointArray(string body, string marker)
    {
        var list = new List<Vector3>();
        int idx = body.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (idx < 0)
            return list;
        int open = body.IndexOf('[', idx);
        int close = open >= 0 ? body.IndexOf(']', open) : -1;
        if (open < 0 || close < open)
            return list;
        return ParsePointArrayLoose(body.Substring(open + 1, close - open - 1), "");
    }

    private static List<Vector3> ParsePointArrayLoose(string body, string marker)
    {
        string slice = body;
        if (!string.IsNullOrEmpty(marker))
        {
            int idx = body.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
            if (idx < 0)
                return new List<Vector3>();
            int open = body.IndexOf('[', idx);
            int close = open >= 0 ? body.IndexOf(']', open) : -1;
            if (open < 0 || close < open)
                return new List<Vector3>();
            slice = body.Substring(open + 1, close - open - 1);
        }

        var list = new List<Vector3>();
        foreach (Match m in Regex.Matches(slice, @"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)"))
        {
            if (float.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) &&
                float.TryParse(m.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var y) &&
                float.TryParse(m.Groups[3].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var z))
                list.Add(new Vector3(x, y, z));
        }

        return list;
    }

    private static string ExtractBlock(string text, int openBrace)
    {
        if (openBrace < 0 || openBrace >= text.Length || text[openBrace] != '{')
            return "";
        int depth = 0;
        for (int i = openBrace; i < text.Length; i++)
        {
            if (text[i] == '{')
                depth++;
            else if (text[i] == '}')
            {
                depth--;
                if (depth == 0)
                    return text.Substring(openBrace + 1, i - openBrace - 1);
            }
        }

        return "";
    }
}
