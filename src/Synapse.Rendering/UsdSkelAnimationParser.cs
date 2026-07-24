using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;

namespace GDNN.Rendering.MeshIO;

/// <summary>One sampled TRS pose for a joint at a given time.</summary>
public readonly struct MeshJointSample
{
    public MeshJointSample(Vector3 translation, Quaternion rotation, Vector3 scale)
    {
        Translation = translation;
        Rotation = rotation;
        Scale = scale;
    }

    public Vector3 Translation { get; }
    public Quaternion Rotation { get; }
    public Vector3 Scale { get; }

    public Matrix4x4 ToLocalMatrix() =>
        Matrix4x4.CreateScale(Scale) *
        Matrix4x4.CreateFromQuaternion(Rotation) *
        Matrix4x4.CreateTranslation(Translation);
}

/// <summary>Keyframe bag for one joint (times aligned across T/R/S or densified).</summary>
public sealed class MeshJointCurve
{
    public string JointName { get; set; } = "";
    public List<float> Times { get; } = new();
    public List<Vector3> Translations { get; } = new();
    public List<Quaternion> Rotations { get; } = new();
    public List<Vector3> Scales { get; } = new();

    public MeshJointSample Sample(float time)
    {
        if (Times.Count == 0)
            return new MeshJointSample(Vector3.Zero, Quaternion.Identity, Vector3.One);

        if (Times.Count == 1 || time <= Times[0])
            return At(0);

        if (time >= Times[^1])
            return At(Times.Count - 1);

        int hi = Times.FindIndex(t => t >= time);
        if (hi <= 0)
            return At(0);
        int lo = hi - 1;
        float t0 = Times[lo];
        float t1 = Times[hi];
        float a = t1 > t0 ? (time - t0) / (t1 - t0) : 0f;
        var tr = Vector3.Lerp(Translations[lo], Translations[hi], a);
        var rot = Quaternion.Slerp(Rotations[lo], Rotations[hi], a);
        var sc = Vector3.Lerp(Scales[lo], Scales[hi], a);
        return new MeshJointSample(tr, rot, sc);
    }

    private MeshJointSample At(int i) =>
        new(
            i < Translations.Count ? Translations[i] : Vector3.Zero,
            i < Rotations.Count ? Rotations[i] : Quaternion.Identity,
            i < Scales.Count ? Scales[i] : Vector3.One);
}

/// <summary>UsdSkel animation clip (SkelAnimation) stored on a <see cref="MeshAsset"/>.</summary>
public sealed class MeshAnimationClip
{
    public string Name { get; set; } = "";
    public float Duration { get; set; }
    public float FrameRate { get; set; } = 24f;
    public List<string> JointNames { get; } = new();
    public List<MeshJointCurve> Curves { get; } = new();

    /// <summary>Samples local TRS matrices per joint at <paramref name="time"/> (seconds).</summary>
    public MeshJointSample[] Evaluate(float time)
    {
        var result = new MeshJointSample[Curves.Count];
        for (int i = 0; i < Curves.Count; i++)
            result[i] = Curves[i].Sample(time);
        return result;
    }

    public Matrix4x4[] EvaluateLocalMatrices(float time)
    {
        var samples = Evaluate(time);
        var mats = new Matrix4x4[samples.Length];
        for (int i = 0; i < samples.Length; i++)
            mats[i] = samples[i].ToLocalMatrix();
        return mats;
    }
}

/// <summary>Parses UsdSkel <c>SkelAnimation</c> prims (static arrays + timeSamples).</summary>
public static class UsdSkelAnimationParser
{
    private static readonly Regex SkelAnimDef = new(
        @"def\s+SkelAnimation\s+""([^""]+)""\s*\{",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex AnimationSource = new(
        @"rel\s+skel:animationSource\s*=\s*<([^>]+)>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex AnimationSources = new(
        @"rel\s+skel:animationSources?\s*=\s*\[([^\]]*)\]",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    public static List<MeshAnimationClip> ParseClips(string usdaText)
    {
        var clips = new List<MeshAnimationClip>();
        if (string.IsNullOrEmpty(usdaText))
            return clips;

        foreach (Match m in SkelAnimDef.Matches(usdaText))
        {
            string name = m.Groups[1].Value;
            string body = ExtractBalancedBlock(usdaText, m.Index + m.Length - 1);
            var clip = ParseClipBody(name, body);
            if (clip.Curves.Count > 0 || clip.JointNames.Count > 0)
                clips.Add(clip);
        }

        return clips;
    }

    public static IReadOnlyList<string> ParseAnimationSourcePaths(string usdaText)
    {
        var list = new List<string>();
        var single = AnimationSource.Match(usdaText);
        if (single.Success)
            list.Add(single.Groups[1].Value.Trim());

        var multi = AnimationSources.Match(usdaText);
        if (multi.Success)
        {
            foreach (Match at in Regex.Matches(multi.Groups[1].Value, @"<([^>]+)>"))
                list.Add(at.Groups[1].Value.Trim());
        }

        return list;
    }

    public static MeshAnimationClip? FindClip(IEnumerable<MeshAnimationClip> clips, string? primPathOrName)
    {
        if (string.IsNullOrWhiteSpace(primPathOrName))
            return clips.FirstOrDefault();
        var leaf = primPathOrName.Trim().Trim('/').Trim('<', '>');
        if (leaf.Contains('/'))
            leaf = leaf[(leaf.LastIndexOf('/') + 1)..];
        return clips.FirstOrDefault(c => c.Name.Equals(leaf, StringComparison.OrdinalIgnoreCase))
               ?? clips.FirstOrDefault();
    }

    private static MeshAnimationClip ParseClipBody(string name, string body)
    {
        var clip = new MeshAnimationClip { Name = name };
        var joints = UsdSkeletonParser.ParseJointNames(body);
        if (joints.Count == 0)
        {
            // Sometimes joints authored as uniform token[] joints without going through skeleton regex — already covered.
            joints = ParseTokenArray(body, "joints");
        }

        clip.JointNames.AddRange(joints);

        var translationSamples = ParseVec3TimeSamples(body, "translations");
        var rotationSamples = ParseQuatTimeSamples(body, "rotations");
        var scaleSamples = ParseVec3TimeSamples(body, "scales");

        // Static (non-timesampled) arrays → single key at t=0
        if (translationSamples.Count == 0)
        {
            var staticT = ParseVec3Array(body, "translations");
            if (staticT.Count > 0)
                translationSamples[0f] = staticT;
        }

        if (rotationSamples.Count == 0)
        {
            var staticR = ParseQuatArray(body, "rotations");
            if (staticR.Count > 0)
                rotationSamples[0f] = staticR;
        }

        if (scaleSamples.Count == 0)
        {
            var staticS = ParseVec3Array(body, "scales");
            if (staticS.Count > 0)
                scaleSamples[0f] = staticS;
        }

        var allTimes = translationSamples.Keys
            .Concat(rotationSamples.Keys)
            .Concat(scaleSamples.Keys)
            .Distinct()
            .OrderBy(t => t)
            .ToList();

        if (allTimes.Count == 0 && joints.Count > 0)
        {
            // Bind pose only
            allTimes.Add(0f);
            translationSamples[0f] = Enumerable.Repeat(Vector3.Zero, joints.Count).ToList();
            rotationSamples[0f] = Enumerable.Repeat(Quaternion.Identity, joints.Count).ToList();
            scaleSamples[0f] = Enumerable.Repeat(Vector3.One, joints.Count).ToList();
        }

        int jointCount = joints.Count;
        if (jointCount == 0)
        {
            jointCount = Math.Max(
                translationSamples.Values.FirstOrDefault()?.Count ?? 0,
                Math.Max(
                    rotationSamples.Values.FirstOrDefault()?.Count ?? 0,
                    scaleSamples.Values.FirstOrDefault()?.Count ?? 0));
            for (int i = 0; i < jointCount; i++)
                clip.JointNames.Add($"Joint{i}");
            joints = clip.JointNames.ToList();
        }

        for (int j = 0; j < jointCount; j++)
        {
            var curve = new MeshJointCurve { JointName = joints[j] };
            foreach (float t in allTimes)
            {
                curve.Times.Add(t);
                curve.Translations.Add(PickVec3(translationSamples, t, j, Vector3.Zero));
                curve.Rotations.Add(PickQuat(rotationSamples, t, j, Quaternion.Identity));
                curve.Scales.Add(PickVec3(scaleSamples, t, j, Vector3.One));
            }

            clip.Curves.Add(curve);
        }

        clip.Duration = allTimes.Count > 0 ? allTimes[^1] : 0f;
        if (allTimes.Count >= 2)
        {
            float dt = allTimes[1] - allTimes[0];
            if (dt > 1e-6f)
                clip.FrameRate = 1f / dt;
        }

        return clip;
    }

    private static Vector3 PickVec3(Dictionary<float, List<Vector3>> samples, float t, int joint, Vector3 fallback)
    {
        if (samples.TryGetValue(t, out var list) && joint < list.Count)
            return list[joint];
        // Nearest earlier sample
        float best = float.NegativeInfinity;
        Vector3 value = fallback;
        foreach (var kv in samples)
        {
            if (kv.Key <= t && kv.Key >= best && joint < kv.Value.Count)
            {
                best = kv.Key;
                value = kv.Value[joint];
            }
        }

        return value;
    }

    private static Quaternion PickQuat(Dictionary<float, List<Quaternion>> samples, float t, int joint, Quaternion fallback)
    {
        if (samples.TryGetValue(t, out var list) && joint < list.Count)
            return list[joint];
        float best = float.NegativeInfinity;
        Quaternion value = fallback;
        foreach (var kv in samples)
        {
            if (kv.Key <= t && kv.Key >= best && joint < kv.Value.Count)
            {
                best = kv.Key;
                value = kv.Value[joint];
            }
        }

        return value;
    }

    private static Dictionary<float, List<Vector3>> ParseVec3TimeSamples(string body, string attr)
    {
        var result = new Dictionary<float, List<Vector3>>();
        var header = new Regex(
            $@"(?:float3|half3|double3)\[\]\s+{Regex.Escape(attr)}\.timeSamples\s*=\s*\{{",
            RegexOptions.IgnoreCase);
        var m = header.Match(body);
        if (!m.Success)
            return result;

        int start = m.Index + m.Length;
        if (!TryExtractBalanced(body, start - 1, '{', '}', out string block))
            return result;

        foreach (Match entry in Regex.Matches(block, @"([-\d.eE+]+)\s*:\s*\[([^\]]*)\]"))
        {
            if (!float.TryParse(entry.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var time))
                continue;
            result[time] = ParseVec3List(entry.Groups[2].Value);
        }

        return result;
    }

    private static Dictionary<float, List<Quaternion>> ParseQuatTimeSamples(string body, string attr)
    {
        var result = new Dictionary<float, List<Quaternion>>();
        var header = new Regex(
            $@"(?:quatf|quatd|quath)\[\]\s+{Regex.Escape(attr)}\.timeSamples\s*=\s*\{{",
            RegexOptions.IgnoreCase);
        var m = header.Match(body);
        if (!m.Success)
            return result;

        int start = m.Index + m.Length;
        if (!TryExtractBalanced(body, start - 1, '{', '}', out string block))
            return result;

        foreach (Match entry in Regex.Matches(block, @"([-\d.eE+]+)\s*:\s*\[([^\]]*)\]"))
        {
            if (!float.TryParse(entry.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var time))
                continue;
            result[time] = ParseQuatList(entry.Groups[2].Value);
        }

        return result;
    }

    private static List<Vector3> ParseVec3Array(string body, string attr)
    {
        var header = new Regex(
            $@"(?:float3|half3|double3)\[\]\s+{Regex.Escape(attr)}\s*=\s*\[",
            RegexOptions.IgnoreCase);
        var m = header.Match(body);
        if (!m.Success)
            return new List<Vector3>();
        int open = m.Index + m.Length - 1;
        int close = body.IndexOf(']', open);
        if (close < open)
            return new List<Vector3>();
        return ParseVec3List(body.Substring(open + 1, close - open - 1));
    }

    private static List<Quaternion> ParseQuatArray(string body, string attr)
    {
        var header = new Regex(
            $@"(?:quatf|quatd|quath)\[\]\s+{Regex.Escape(attr)}\s*=\s*\[",
            RegexOptions.IgnoreCase);
        var m = header.Match(body);
        if (!m.Success)
            return new List<Quaternion>();
        int open = m.Index + m.Length - 1;
        int close = body.IndexOf(']', open);
        if (close < open)
            return new List<Quaternion>();
        return ParseQuatList(body.Substring(open + 1, close - open - 1));
    }

    private static List<Vector3> ParseVec3List(string body)
    {
        var list = new List<Vector3>();
        foreach (Match m in Regex.Matches(body, @"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)"))
        {
            if (float.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) &&
                float.TryParse(m.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var y) &&
                float.TryParse(m.Groups[3].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var z))
                list.Add(new Vector3(x, y, z));
        }

        return list;
    }

    private static List<Quaternion> ParseQuatList(string body)
    {
        var list = new List<Quaternion>();
        // USD quatf is (x, y, z, w) imaginary + real
        foreach (Match m in Regex.Matches(
                     body,
                     @"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)"))
        {
            if (float.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) &&
                float.TryParse(m.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var y) &&
                float.TryParse(m.Groups[3].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var z) &&
                float.TryParse(m.Groups[4].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var w))
            {
                var q = new Quaternion(x, y, z, w);
                if (q.LengthSquared() > 1e-8f)
                    q = Quaternion.Normalize(q);
                list.Add(q);
            }
        }

        return list;
    }

    private static List<string> ParseTokenArray(string text, string attr)
    {
        var list = new List<string>();
        var m = Regex.Match(text, $@"token\[\]\s+{Regex.Escape(attr)}\s*=\s*\[([^\]]*)\]",
            RegexOptions.IgnoreCase | RegexOptions.Singleline);
        if (!m.Success)
            return list;
        foreach (Match tok in Regex.Matches(m.Groups[1].Value, @"""([^""]+)"""))
            list.Add(tok.Groups[1].Value);
        return list;
    }

    private static string ExtractBalancedBlock(string text, int openBraceIndex)
    {
        if (!TryExtractBalanced(text, openBraceIndex, '{', '}', out string inner))
            return "";
        return inner;
    }

    private static bool TryExtractBalanced(string text, int openIndex, char open, char close, out string inner)
    {
        inner = "";
        if (openIndex < 0 || openIndex >= text.Length || text[openIndex] != open)
            return false;
        int depth = 0;
        for (int i = openIndex; i < text.Length; i++)
        {
            if (text[i] == open)
                depth++;
            else if (text[i] == close)
            {
                depth--;
                if (depth == 0)
                {
                    inner = text.Substring(openIndex + 1, i - openIndex - 1);
                    return true;
                }
            }
        }

        return false;
    }
}
