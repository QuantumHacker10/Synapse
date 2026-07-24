using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;

namespace GDNN.Rendering.MeshIO;

/// <summary>
/// OpenUSD mesh topology helpers: <c>faceVertexCounts</c> triangulation, normals, extents,
/// purpose/visibility filters used by the production MeshIO path.
/// </summary>
public static class UsdMeshTopology
{
    private static readonly Regex MeshDef = new(
        @"def\s+Mesh\s+""([^""]+)""\s*\{",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex PointTuple = new(
        @"\(([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\)",
        RegexOptions.Compiled);

    public static IReadOnlyList<(string Name, string Body)> EnumerateMeshBodies(string text)
    {
        var list = new List<(string, string)>();
        if (string.IsNullOrEmpty(text))
            return list;

        foreach (Match m in MeshDef.Matches(text))
        {
            string name = m.Groups[1].Value;
            string body = ExtractBalancedBlock(text, m.Index + m.Length - 1);
            if (!string.IsNullOrEmpty(body))
                list.Add((name, body));
        }

        return list;
    }

    public static bool ShouldSkipPrim(string body, MeshLoadConfig config, out string? reason)
    {
        reason = null;
        config ??= new MeshLoadConfig();

        var purpose = ParseToken(body, "purpose");
        if (!string.IsNullOrEmpty(purpose))
        {
            var p = purpose.Trim().ToLowerInvariant();
            if (p is "guide" or "proxy")
            {
                if ((config.UsdPurposeMask & UsdPurposeMask.Guide) == 0 && p == "guide")
                {
                    reason = $"skipped purpose=guide";
                    return true;
                }

                if ((config.UsdPurposeMask & UsdPurposeMask.Proxy) == 0 && p == "proxy")
                {
                    reason = $"skipped purpose=proxy";
                    return true;
                }
            }
            else if (p == "render" && (config.UsdPurposeMask & UsdPurposeMask.Render) == 0)
            {
                reason = "skipped purpose=render";
                return true;
            }
        }

        if (config.UsdSkipInvisible)
        {
            var vis = ParseToken(body, "visibility");
            if (string.Equals(vis, "invisible", StringComparison.OrdinalIgnoreCase))
            {
                reason = "skipped visibility=invisible";
                return true;
            }
        }

        return false;
    }

    public static List<Vector3> ParsePoints(string text)
    {
        var points = ParseTypedPointArray(text, "point3f[] points");
        if (points.Count == 0)
            points = ParseTypedPointArray(text, "point3d[] points");
        return points;
    }

    public static List<Vector3> ParseNormals(string text)
    {
        var normals = ParseTypedPointArray(text, "normal3f[] normals");
        if (normals.Count == 0)
            normals = ParseTypedPointArray(text, "normal3f[] primvars:normals");
        if (normals.Count == 0)
            normals = ParseTypedPointArray(text, "float3[] normals");
        return normals;
    }

    public static List<Vector2> ParseUvs(string text)
    {
        var uvs = new List<Vector2>();
        string[] markers =
        {
            "texCoord2f[] primvars:st = [",
            "float2[] primvars:st = [",
            "float2[] primvars:UVMap = [",
            "texCoord2f[] primvars:UVMap = ["
        };
        int start = -1;
        foreach (var marker in markers)
        {
            int idx = text.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
            if (idx >= 0)
            {
                start = idx + marker.Length;
                break;
            }
        }

        if (start < 0)
            return uvs;
        int end = text.IndexOf(']', start);
        if (end < start)
            return uvs;
        var body = text[start..end];
        var pair = new Regex(@"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)", RegexOptions.Compiled);
        foreach (Match match in pair.Matches(body))
        {
            if (float.TryParse(match.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var u) &&
                float.TryParse(match.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                uvs.Add(new Vector2(u, v));
        }

        return uvs;
    }

    /// <summary>
    /// Triangulates faces using <c>faceVertexCounts</c> when present; otherwise fans on <c>-1</c>
    /// sentinels (OBJ-style) or treats a flat index list as already-triangulated triangles.
    /// </summary>
    public static List<uint> ParseTriangulatedIndices(string text, List<string>? warnings = null)
    {
        var counts = ParseIntArray(text, "int[] faceVertexCounts");
        var raw = ParseIntArray(text, "int[] faceVertexIndices");
        if (raw.Count == 0)
            return new List<uint>();

        if (counts.Count > 0)
            return TriangulateFromCounts(counts, raw, warnings);

        // Sentinel style: …, -1, …
        if (raw.Exists(v => v < 0))
            return TriangulateFromSentinels(raw);

        // Flat triangle list (length % 3 == 0) — common after DCC triangulation export.
        if (raw.Count % 3 == 0)
        {
            var indices = new List<uint>(raw.Count);
            foreach (var v in raw)
                indices.Add((uint)v);
            return indices;
        }

        warnings?.Add($"faceVertexIndices length {raw.Count} is not a multiple of 3 and has no faceVertexCounts; treating as triangle fan from vertex 0.");
        var fan = new List<uint>();
        for (int i = 1; i < raw.Count - 1; i++)
        {
            fan.Add((uint)raw[0]);
            fan.Add((uint)raw[i]);
            fan.Add((uint)raw[i + 1]);
        }

        return fan;
    }

    public static bool TryParseExtent(string text, out BoundingBox3D bounds)
    {
        bounds = default;
        var pts = ParseTypedPointArray(text, "float3[] extent");
        if (pts.Count < 2)
            pts = ParseTypedPointArray(text, "float3[] extentHint");
        if (pts.Count < 2)
            return false;
        bounds = new BoundingBox3D(Vector3.Min(pts[0], pts[1]), Vector3.Max(pts[0], pts[1]));
        return true;
    }

    public static bool ParseDoubleSided(string text)
    {
        var m = Regex.Match(text, @"(?:uniform\s+)?bool\s+doubleSided\s*=\s*([01]|true|false)",
            RegexOptions.IgnoreCase);
        if (!m.Success)
            return false;
        var v = m.Groups[1].Value;
        return v is "1" or "true" or "True";
    }

    public static string? ParseToken(string text, string key)
    {
        var m = Regex.Match(text,
            $@"(?:uniform\s+)?token\s+{Regex.Escape(key)}\s*=\s*""([^""]+)""",
            RegexOptions.IgnoreCase);
        return m.Success ? m.Groups[1].Value : null;
    }

    public static float? ParseStageFloat(string text, string key)
    {
        // Stage metadata lives in the header (...) block before prims.
        var m = Regex.Match(text, $@"\b{Regex.Escape(key)}\s*=\s*([-\d.eE+]+)", RegexOptions.IgnoreCase);
        if (!m.Success)
            return null;
        return float.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
            ? v
            : null;
    }

    public static void AssignNormals(
        List<MeshVertex> vertices,
        List<uint> indices,
        List<Vector3> normals,
        MeshProcessFlags flags,
        List<string>? warnings = null)
    {
        if (normals.Count == vertices.Count)
        {
            for (int i = 0; i < vertices.Count; i++)
            {
                var v = vertices[i];
                var n = normals[i];
                if (n.LengthSquared() > 1e-12f)
                    v.Normal = Vector3.Normalize(n);
                vertices[i] = v;
            }

            return;
        }

        if (normals.Count == indices.Count && indices.Count > 0)
        {
            // Face-varying: average into unique vertices.
            var acc = new Vector3[vertices.Count];
            var counts = new int[vertices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                int vi = (int)indices[i];
                if (vi < 0 || vi >= vertices.Count)
                    continue;
                acc[vi] += normals[i];
                counts[vi]++;
            }

            for (int i = 0; i < vertices.Count; i++)
            {
                if (counts[i] == 0)
                    continue;
                var n = acc[i] / counts[i];
                if (n.LengthSquared() > 1e-12f)
                {
                    var v = vertices[i];
                    v.Normal = Vector3.Normalize(n);
                    vertices[i] = v;
                }
            }

            return;
        }

        if (normals.Count > 0)
            warnings?.Add($"normals count {normals.Count} did not match vertices ({vertices.Count}) or indices ({indices.Count}); generating.");

        if ((flags & MeshProcessFlags.CalculateNormals) != 0)
            GenerateSmoothNormals(vertices, indices);
        else
        {
            for (int i = 0; i < vertices.Count; i++)
            {
                var v = vertices[i];
                v.Normal = Vector3.UnitY;
                vertices[i] = v;
            }
        }
    }

    public static void GenerateSmoothNormals(List<MeshVertex> vertices, List<uint> indices)
    {
        var acc = new Vector3[vertices.Count];
        for (int i = 0; i + 2 < indices.Count; i += 3)
        {
            int i0 = (int)indices[i];
            int i1 = (int)indices[i + 1];
            int i2 = (int)indices[i + 2];
            if (i0 < 0 || i1 < 0 || i2 < 0 ||
                i0 >= vertices.Count || i1 >= vertices.Count || i2 >= vertices.Count)
                continue;
            var p0 = vertices[i0].Position;
            var p1 = vertices[i1].Position;
            var p2 = vertices[i2].Position;
            var n = Vector3.Cross(p1 - p0, p2 - p0);
            if (n.LengthSquared() < 1e-20f)
                continue;
            n = Vector3.Normalize(n);
            acc[i0] += n;
            acc[i1] += n;
            acc[i2] += n;
        }

        for (int i = 0; i < vertices.Count; i++)
        {
            var v = vertices[i];
            v.Normal = acc[i].LengthSquared() > 1e-12f ? Vector3.Normalize(acc[i]) : Vector3.UnitY;
            vertices[i] = v;
        }
    }

    public static string ExtractBalancedBlock(string text, int openBrace)
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

    private static List<Vector3> ParseTypedPointArray(string text, string marker)
    {
        var points = new List<Vector3>();
        int idx = text.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (idx < 0)
            return points;
        // Skip the type's own "[]" (e.g. point3f[]) — open the value array after '='.
        int eq = text.IndexOf('=', idx + marker.Length);
        if (eq < 0)
            eq = text.IndexOf('=', idx);
        int open = eq >= 0 ? text.IndexOf('[', eq) : text.IndexOf('[', idx + marker.Length);
        int close = open >= 0 ? text.IndexOf(']', open) : -1;
        if (open < 0 || close < open)
            return points;
        var body = text.Substring(open + 1, close - open - 1);
        foreach (Match match in PointTuple.Matches(body))
        {
            if (float.TryParse(match.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) &&
                float.TryParse(match.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var y) &&
                float.TryParse(match.Groups[3].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var z))
                points.Add(new Vector3(x, y, z));
        }

        return points;
    }

    private static List<int> ParseIntArray(string text, string marker)
    {
        var list = new List<int>();
        int idx = text.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (idx < 0)
            return list;
        int eq = text.IndexOf('=', idx + marker.Length);
        if (eq < 0)
            eq = text.IndexOf('=', idx);
        int open = eq >= 0 ? text.IndexOf('[', eq) : text.IndexOf('[', idx + marker.Length);
        int close = open >= 0 ? text.IndexOf(']', open) : -1;
        if (open < 0 || close < open)
            return list;
        var body = text.Substring(open + 1, close - open - 1);
        foreach (var n in body.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (int.TryParse(n, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v))
                list.Add(v);
        }

        return list;
    }

    private static List<uint> TriangulateFromCounts(List<int> counts, List<int> raw, List<string>? warnings)
    {
        var indices = new List<uint>();
        int cursor = 0;
        for (int f = 0; f < counts.Count; f++)
        {
            int n = counts[f];
            if (n < 3)
            {
                warnings?.Add($"faceVertexCounts[{f}]={n} ignored (need ≥3).");
                cursor += Math.Max(n, 0);
                continue;
            }

            if (cursor + n > raw.Count)
            {
                warnings?.Add("faceVertexCounts overrun faceVertexIndices; stopping.");
                break;
            }

            int i0 = raw[cursor];
            for (int k = 1; k < n - 1; k++)
            {
                indices.Add((uint)i0);
                indices.Add((uint)raw[cursor + k]);
                indices.Add((uint)raw[cursor + k + 1]);
            }

            cursor += n;
        }

        return indices;
    }

    private static List<uint> TriangulateFromSentinels(List<int> raw)
    {
        var indices = new List<uint>();
        var poly = new List<int>();
        foreach (var v in raw)
        {
            if (v < 0)
            {
                for (int i = 1; i < poly.Count - 1; i++)
                {
                    indices.Add((uint)poly[0]);
                    indices.Add((uint)poly[i]);
                    indices.Add((uint)poly[i + 1]);
                }

                poly.Clear();
            }
            else
            {
                poly.Add(v);
            }
        }

        // Trailing poly without sentinel
        for (int i = 1; i < poly.Count - 1; i++)
        {
            indices.Add((uint)poly[0]);
            indices.Add((uint)poly[i]);
            indices.Add((uint)poly[i + 1]);
        }

        return indices;
    }
}

/// <summary>Which USD <c>purpose</c> values to include when importing meshes.</summary>
[Flags]
public enum UsdPurposeMask
{
    None = 0,
    Default = 1 << 0,
    Render = 1 << 1,
    Proxy = 1 << 2,
    Guide = 1 << 3,
    /// <summary>Default production mask: default + render (skip proxy/guide).</summary>
    Production = Default | Render,
    All = Default | Render | Proxy | Guide
}
