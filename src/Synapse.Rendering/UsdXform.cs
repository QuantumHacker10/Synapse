using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;

namespace GDNN.Rendering.MeshIO;

/// <summary>
/// OpenUSD xformOp stack parser (translate, rotateXYZ, scale, transform) with optional xformOpOrder.
/// </summary>
public static class UsdXform
{
    private static readonly Regex FloatTuple3 = new(
        @"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)",
        RegexOptions.Compiled);

    private static readonly Regex FloatTuple4 = new(
        @"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)",
        RegexOptions.Compiled);

    private static readonly Regex MatrixRow = new(
        @"\(([^)]+)\)",
        RegexOptions.Compiled);

    private static readonly Regex OpOrder = new(
        @"uniform\s+token\[\]\s+xformOpOrder\s*=\s*\[([^\]]*)\]",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    private static readonly Regex OpOrderAlt = new(
        @"token\[\]\s+xformOpOrder\s*=\s*\[([^\]]*)\]",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    /// <summary>Builds a local-to-parent matrix from USDA xformOps (identity when none present).</summary>
    public static Matrix4x4 ParseLocalMatrix(string usdaText)
    {
        if (string.IsNullOrEmpty(usdaText))
            return Matrix4x4.Identity;

        var order = ParseOpOrder(usdaText);
        if (order.Count == 0)
        {
            // Default OpenUSD order when ops exist without explicit order: translate, rotateXYZ, scale, transform.
            if (HasOp(usdaText, "xformOp:transform"))
                order.Add("xformOp:transform");
            if (HasOp(usdaText, "xformOp:translate"))
                order.Add("xformOp:translate");
            if (HasOp(usdaText, "xformOp:rotateXYZ"))
                order.Add("xformOp:rotateXYZ");
            if (HasOp(usdaText, "xformOp:scale"))
                order.Add("xformOp:scale");
        }

        var result = Matrix4x4.Identity;
        foreach (var op in order)
        {
            var name = op.Trim().Trim('"').Trim('\'');
            if (string.IsNullOrEmpty(name))
                continue;
            bool inverted = name.StartsWith("!invert!", StringComparison.Ordinal);
            if (inverted)
                name = name["!invert!".Length..];

            Matrix4x4 m = Matrix4x4.Identity;
            if (name.Contains("translate", StringComparison.OrdinalIgnoreCase))
            {
                var t = ParseVec3Op(usdaText, "xformOp:translate");
                m = Matrix4x4.CreateTranslation(t);
            }
            else if (name.Contains("rotateXYZ", StringComparison.OrdinalIgnoreCase))
            {
                var r = ParseVec3Op(usdaText, "xformOp:rotateXYZ");
                // USD rotateXYZ is degrees, applied X then Y then Z in local space.
                m = Matrix4x4.CreateRotationX(DegToRad(r.X)) *
                    Matrix4x4.CreateRotationY(DegToRad(r.Y)) *
                    Matrix4x4.CreateRotationZ(DegToRad(r.Z));
            }
            else if (name.Contains("scale", StringComparison.OrdinalIgnoreCase) &&
                     !name.Contains("rotate", StringComparison.OrdinalIgnoreCase))
            {
                var s = ParseVec3Op(usdaText, "xformOp:scale", Vector3.One);
                m = Matrix4x4.CreateScale(s);
            }
            else if (name.Contains("transform", StringComparison.OrdinalIgnoreCase))
            {
                m = ParseMatrixOp(usdaText, "xformOp:transform");
            }
            else
            {
                continue;
            }

            if (inverted && Matrix4x4.Invert(m, out var inv))
                m = inv;

            // USD applies ops in order as local = opN * ... * op0 * point (row-vector convention varies).
            // System.Numerics uses row vectors (v * M). Compose as result = m * result so first op is applied first.
            result = m * result;
        }

        return result;
    }

    public static Vector3 TransformPoint(Vector3 point, Matrix4x4 matrix) =>
        Vector3.Transform(point, matrix);

    public static void ApplyToPoints(IList<Vector3> points, Matrix4x4 matrix)
    {
        if (matrix.IsIdentity)
            return;
        for (int i = 0; i < points.Count; i++)
            points[i] = Vector3.Transform(points[i], matrix);
    }

    public static List<string> ParseOpOrder(string text)
    {
        var list = new List<string>();
        var m = OpOrder.Match(text);
        if (!m.Success)
            m = OpOrderAlt.Match(text);
        if (!m.Success)
            return list;
        foreach (Match tok in Regex.Matches(m.Groups[1].Value, @"""([^""]+)""|token\s+(\S+)"))
        {
            var v = tok.Groups[1].Success ? tok.Groups[1].Value : tok.Groups[2].Value;
            if (!string.IsNullOrWhiteSpace(v))
                list.Add(v.Trim());
        }

        // Also split on commas for bare names
        if (list.Count == 0)
        {
            foreach (var part in m.Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                var cleaned = part.Trim().Trim('"').Trim('\'');
                if (!string.IsNullOrEmpty(cleaned))
                    list.Add(cleaned);
            }
        }

        return list;
    }

    public static Vector3 ParseVec3Op(string text, string opName, Vector3 fallback = default)
    {
        int idx = IndexOfOpAssignment(text, opName);
        if (idx < 0)
            return fallback;
        var m = FloatTuple3.Match(text, idx);
        if (!m.Success)
            return fallback;
        if (float.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) &&
            float.TryParse(m.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var y) &&
            float.TryParse(m.Groups[3].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var z))
            return new Vector3(x, y, z);
        return fallback;
    }

    public static Matrix4x4 ParseMatrixOp(string text, string opName)
    {
        int idx = IndexOfOpAssignment(text, opName);
        if (idx < 0)
            return Matrix4x4.Identity;
        int eq = text.IndexOf('=', idx);
        if (eq < 0)
            return Matrix4x4.Identity;
        int open = text.IndexOf('(', eq);
        if (open < 0)
            return Matrix4x4.Identity;

        // Collect up to 16 floats from nested tuples: ((r0), (r1), (r2), (r3))
        var floats = new List<float>(16);
        foreach (Match row in MatrixRow.Matches(text.Substring(open, Math.Min(text.Length - open, 800))))
        {
            var nums = row.Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            foreach (var n in nums)
            {
                if (float.TryParse(n, NumberStyles.Float, CultureInfo.InvariantCulture, out var f))
                    floats.Add(f);
                if (floats.Count >= 16)
                    break;
            }
            if (floats.Count >= 16)
                break;
        }

        if (floats.Count < 16)
            return Matrix4x4.Identity;

        // USD matrix4d is row-major; System.Numerics.Matrix4x4 is also row-major in constructor M11..M44.
        return new Matrix4x4(
            floats[0], floats[1], floats[2], floats[3],
            floats[4], floats[5], floats[6], floats[7],
            floats[8], floats[9], floats[10], floats[11],
            floats[12], floats[13], floats[14], floats[15]);
    }

    private static bool HasOp(string text, string opName) =>
        IndexOfOpAssignment(text, opName) >= 0;

    private static int IndexOfOpAssignment(string text, string opName)
    {
        int idx = text.IndexOf(opName, StringComparison.Ordinal);
        while (idx >= 0)
        {
            int eq = text.IndexOf('=', idx);
            if (eq > idx && eq - idx < opName.Length + 24)
                return idx;
            idx = text.IndexOf(opName, idx + opName.Length, StringComparison.Ordinal);
        }

        return -1;
    }

    private static float DegToRad(float deg) => deg * (MathF.PI / 180f);
}
