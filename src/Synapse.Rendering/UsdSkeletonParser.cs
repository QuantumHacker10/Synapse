using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;

namespace GDNN.Rendering.MeshIO;

/// <summary>USD skeleton / skinning payload attached to a <see cref="MeshAsset"/>.</summary>
public sealed class MeshSkeleton
{
    public List<string> JointNames { get; } = new();
    public List<int> ParentIndices { get; } = new();
    public List<Matrix4x4> BindTransforms { get; } = new();
    public List<Matrix4x4> InverseBindMatrices { get; } = new();
}

/// <summary>Parses UsdSkel joints, bindTransforms, and skel:jointIndices/Weights primvars.</summary>
public static class UsdSkeletonParser
{
    private static readonly Regex TokenArray = new(
        @"token\[\]\s+joints\s*=\s*\[([^\]]*)\]",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    private static readonly Regex BindTransforms = new(
        @"matrix4d\[\]\s+bindTransforms\s*=\s*\[",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex RestTransforms = new(
        @"matrix4d\[\]\s+restTransforms\s*=\s*\[",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    public static MeshSkeleton? ParseSkeleton(string usdaText, MeshLoadConfig? config = null)
    {
        if (string.IsNullOrEmpty(usdaText))
            return null;

        var joints = ParseJointNames(usdaText);
        if (joints.Count == 0)
            return null;

        config ??= new MeshLoadConfig();
        if (joints.Count > config.MaxBones)
            joints = joints.GetRange(0, config.MaxBones);

        var skel = new MeshSkeleton();
        skel.JointNames.AddRange(joints);
        for (int i = 0; i < joints.Count; i++)
        {
            // Infer parent from path hierarchy: Root/Hips/Spine → parent = Hips index
            skel.ParentIndices.Add(InferParentIndex(joints, i));
        }

        var binds = ParseMatrixArray(usdaText);
        for (int i = 0; i < joints.Count; i++)
        {
            var m = i < binds.Count ? binds[i] : Matrix4x4.Identity;
            skel.BindTransforms.Add(m);
            if (Matrix4x4.Invert(m, out var inv))
                skel.InverseBindMatrices.Add(inv);
            else
                skel.InverseBindMatrices.Add(Matrix4x4.Identity);
        }

        return skel;
    }

    public static void ApplySkinPrimvars(string usdaText, List<MeshVertex> vertices, MeshLoadConfig? config = null)
    {
        if (vertices.Count == 0)
            return;
        config ??= new MeshLoadConfig();
        int stride = Math.Clamp(config.MaxBoneWeightsPerVertex, 1, 4);

        var indices = ParseIntArray(usdaText, "primvars:skel:jointIndices")
                      ?? ParseIntArray(usdaText, "int[] primvars:skel:jointIndices");
        var weights = ParseFloatArray(usdaText, "primvars:skel:jointWeights")
                      ?? ParseFloatArray(usdaText, "float[] primvars:skel:jointWeights");
        if (indices == null || weights == null || indices.Count == 0)
            return;

        for (int v = 0; v < vertices.Count; v++)
        {
            int baseIdx = v * stride;
            if (baseIdx >= indices.Count)
                break;

            var vert = vertices[v];
            int i0 = SafeIndex(indices, baseIdx);
            int i1 = stride > 1 ? SafeIndex(indices, baseIdx + 1) : 0;
            int i2 = stride > 2 ? SafeIndex(indices, baseIdx + 2) : 0;
            int i3 = stride > 3 ? SafeIndex(indices, baseIdx + 3) : 0;
            vert.BoneIndices = new Vector4i(i0, i1, i2, i3);

            float w0 = SafeWeight(weights, baseIdx);
            float w1 = stride > 1 ? SafeWeight(weights, baseIdx + 1) : 0;
            float w2 = stride > 2 ? SafeWeight(weights, baseIdx + 2) : 0;
            float w3 = stride > 3 ? SafeWeight(weights, baseIdx + 3) : 0;
            float sum = w0 + w1 + w2 + w3;
            if (sum > 1e-6f)
            {
                w0 /= sum;
                w1 /= sum;
                w2 /= sum;
                w3 /= sum;
            }
            else
            {
                w0 = 1f;
            }

            vert.BoneWeights = new Vector4(w0, w1, w2, w3);
            vertices[v] = vert;
        }
    }

    public static List<string> ParseJointNames(string text)
    {
        var list = new List<string>();
        var m = TokenArray.Match(text);
        if (!m.Success)
            return list;
        foreach (Match tok in Regex.Matches(m.Groups[1].Value, @"""([^""]+)"""))
        {
            var name = tok.Groups[1].Value.Trim();
            if (!string.IsNullOrEmpty(name))
                list.Add(name);
        }

        return list;
    }

    public static List<Matrix4x4> ParseMatrixArray(string text)
    {
        var list = new List<Matrix4x4>();
        // Prefer bindTransforms (skin bind pose) over restTransforms.
        var m = BindTransforms.Match(text);
        if (!m.Success)
            m = RestTransforms.Match(text);
        if (!m.Success)
            return list;

        int start = m.Index + m.Length;
        // Collect floats until we've gathered multiples of 16 or hit a reasonable cap.
        var floats = new List<float>();
        int end = Math.Min(text.Length, start + 8000);
        var slice = text[start..end];
        foreach (Match num in Regex.Matches(slice, @"[-\d.eE+]+"))
        {
            if (float.TryParse(num.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var f))
                floats.Add(f);
            // Stop early if we hit a non-matrix section keyword after some data
            if (floats.Count >= 16 * 64)
                break;
        }

        for (int i = 0; i + 15 < floats.Count; i += 16)
        {
            list.Add(new Matrix4x4(
                floats[i], floats[i + 1], floats[i + 2], floats[i + 3],
                floats[i + 4], floats[i + 5], floats[i + 6], floats[i + 7],
                floats[i + 8], floats[i + 9], floats[i + 10], floats[i + 11],
                floats[i + 12], floats[i + 13], floats[i + 14], floats[i + 15]));
        }

        return list;
    }

    private static int InferParentIndex(IReadOnlyList<string> joints, int index)
    {
        var path = joints[index].Replace('\\', '/').Trim('/');
        int slash = path.LastIndexOf('/');
        if (slash <= 0)
            return -1;
        var parentPath = path[..slash];
        for (int i = 0; i < joints.Count; i++)
        {
            if (i == index)
                continue;
            var cand = joints[i].Replace('\\', '/').Trim('/');
            if (cand.Equals(parentPath, StringComparison.OrdinalIgnoreCase) ||
                cand.EndsWith("/" + parentPath, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(cand, parentPath[(parentPath.LastIndexOf('/') + 1)..], StringComparison.OrdinalIgnoreCase))
                return i;
        }

        // Match by leaf name of parent
        var parentLeaf = parentPath.Contains('/') ? parentPath[(parentPath.LastIndexOf('/') + 1)..] : parentPath;
        for (int i = 0; i < joints.Count; i++)
        {
            if (i == index)
                continue;
            var leaf = joints[i].Contains('/') ? joints[i][(joints[i].LastIndexOf('/') + 1)..] : joints[i];
            if (leaf.Equals(parentLeaf, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        return index > 0 ? index - 1 : -1;
    }

    private static List<int>? ParseIntArray(string text, string marker)
    {
        int idx = text.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (idx < 0)
            return null;
        int open = text.IndexOf('[', idx);
        int close = open >= 0 ? text.IndexOf(']', open) : -1;
        if (open < 0 || close < open)
            return null;
        var list = new List<int>();
        foreach (var part in text.Substring(open + 1, close - open - 1)
                     .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v))
                list.Add(v);
        }

        return list;
    }

    private static List<float>? ParseFloatArray(string text, string marker)
    {
        int idx = text.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (idx < 0)
            return null;
        int open = text.IndexOf('[', idx);
        int close = open >= 0 ? text.IndexOf(']', open) : -1;
        if (open < 0 || close < open)
            return null;
        var list = new List<float>();
        foreach (var part in text.Substring(open + 1, close - open - 1)
                     .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (float.TryParse(part, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                list.Add(v);
        }

        return list;
    }

    private static int SafeIndex(List<int> list, int i) => i < list.Count ? list[i] : 0;
    private static float SafeWeight(List<float> list, int i) => i < list.Count ? list[i] : 0f;
}
