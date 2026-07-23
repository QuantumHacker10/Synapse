// =============================================================================
// G-DNN — Nanite Neural 3.0 (AAA)
// Virtualized neural geometry: continuous LOD, cluster material resolve,
// opportunistic page selection, and screen-error driven meshlet density.
// =============================================================================

using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace GDNN.Polygonization;

/// <summary>
/// Nanite Neural 3.0 policy: converts viewport / camera error into polygonization
/// density, visibility-buffer resolution, and cluster material weights.
/// Pure algorithms — no Vulkan dependency — so CPU and GPU paths share one brain.
/// </summary>
public static class NaniteNeural30
{
    /// <summary>Industrial defaults tuned for on-screen cluster density.</summary>
    public sealed class Policy
    {
        public int MinPolyResolution { get; init; } = 32;
        public int MaxPolyResolution { get; init; } = 64;
        public int MinVisibilityWidth { get; init; } = 384;
        public int MaxVisibilityWidth { get; init; } = 1280;
        public float ScreenErrorThresholdPx { get; init; } = 1.0f;
        public float ContinuousLodBias { get; init; } = 0.40f;
        public int MeshletTickCadence { get; init; } = 2;
        public float ClusterAlbedoBoost { get; init; } = 1.28f;
        public float ContactAoStrength { get; init; } = 0.60f;
        public float MetallicBias { get; init; } = 0.12f;
        public float NormalDetailStrength { get; init; } = 0.18f;
    }

    public static Policy Industrial { get; } = new();

    /// <summary>Realtime AAA density (Ultra / High).</summary>
    public static Policy Aaa { get; } = new()
    {
        MinPolyResolution = 40,
        MaxPolyResolution = 80,
        MinVisibilityWidth = 512,
        MaxVisibilityWidth = 1600,
        ScreenErrorThresholdPx = 0.85f,
        ContinuousLodBias = 0.48f,
        MeshletTickCadence = 1,
        ClusterAlbedoBoost = 1.32f,
        ContactAoStrength = 0.68f,
        MetallicBias = 0.18f,
        NormalDetailStrength = 0.28f
    };

    /// <summary>
    /// Continuous LOD factor in [0,1] from camera distance and screen error.
    /// 0 = coarsest (far), 1 = densest (close / high error).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ContinuousLod(
        float cameraDistance,
        float projectedRadiusPx,
        Policy? policy = null)
    {
        policy ??= Industrial;
        float distTerm = 1f / (1f + MathF.Max(0f, cameraDistance) * 0.075f);
        float errorTerm = Math.Clamp(projectedRadiusPx / MathF.Max(0.5f, policy.ScreenErrorThresholdPx), 0f, 2.5f);
        float raw = 0.42f * distTerm + 0.58f * MathF.Tanh(errorTerm);
        return Math.Clamp(raw + policy.ContinuousLodBias * 0.18f, 0f, 1f);
    }

    /// <summary>Polygon extract resolution from continuous LOD.</summary>
    public static int PolyResolution(float lod01, int viewportMax, Policy? policy = null)
    {
        policy ??= Industrial;
        int fromViewport = Math.Clamp(viewportMax / 36, policy.MinPolyResolution, policy.MaxPolyResolution);
        int fromLod = (int)MathF.Round(
            policy.MinPolyResolution + lod01 * (policy.MaxPolyResolution - policy.MinPolyResolution));
        return Math.Clamp(Math.Max(fromViewport, fromLod), policy.MinPolyResolution, policy.MaxPolyResolution);
    }

    /// <summary>Visibility-buffer size (Nanite-like cluster tiles).</summary>
    public static (int Width, int Height) VisibilityBufferSize(int viewportW, int viewportH, float lod01, Policy? policy = null)
    {
        policy ??= Industrial;
        float scale = 0.50f + 0.42f * lod01;
        int w = Math.Clamp((int)(viewportW * scale), policy.MinVisibilityWidth, policy.MaxVisibilityWidth);
        int h = Math.Clamp((int)(viewportH * scale), policy.MinVisibilityWidth, policy.MaxVisibilityWidth);
        // Keep even dims for compute workgroup alignment.
        w &= ~1;
        h &= ~1;
        return (Math.Max(2, w), Math.Max(2, h));
    }

    /// <summary>
    /// Screen-space projected radius of an AABB (pixels) under a perspective-like fov.
    /// </summary>
    public static float ProjectedRadiusPx(
        Vector3 boundsCenter,
        float boundsRadius,
        Vector3 cameraPos,
        float fovYRadians,
        int viewportHeight)
    {
        float dist = MathF.Max(0.05f, Vector3.Distance(cameraPos, boundsCenter));
        float halfFov = MathF.Max(1e-3f, fovYRadians * 0.5f);
        float worldPerPixel = 2f * dist * MathF.Tan(halfFov) / Math.Max(1, viewportHeight);
        return boundsRadius / MathF.Max(1e-4f, worldPerPixel);
    }

    /// <summary>
    /// Cluster material resolve: maps meshlet+triangle hash to albedo with
    /// metallic/roughness packed into a Vector4 (rgb + roughness proxy).
    /// AAA: desaturated industrial PBR with controlled metallic bias.
    /// </summary>
    public static Vector4 ResolveClusterMaterial(int meshletIndex, int triangleIndex, float lod01, Policy? policy = null)
    {
        policy ??= Industrial;
        uint h = unchecked((uint)(meshletIndex * 73856093) ^ (uint)(triangleIndex * 19349663) ^ 0x9E3779B9u);
        float r = ((h) & 255) / 255f;
        float g = ((h >> 8) & 255) / 255f;
        float b = ((h >> 16) & 255) / 255f;
        // Slight desaturation toward industrial PBR neutrals at coarse LOD.
        float sat = 0.48f + 0.52f * lod01;
        float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        r = luma + (r - luma) * sat;
        g = luma + (g - luma) * sat;
        b = luma + (b - luma) * sat;
        float boost = policy.ClusterAlbedoBoost;
        float metallic = Math.Clamp(policy.MetallicBias * (0.35f + 0.65f * lod01) * (((h >> 20) & 15) / 15f), 0f, 0.85f);
        // Microfacet roughness: tighter at high LOD (AAA detail).
        float roughness = 0.18f + 0.62f * (((h >> 24) & 255) / 255f) * (1f - 0.42f * lod01);
        roughness = Math.Clamp(roughness * (1f - 0.25f * metallic), 0.04f, 1f);
        // Pre-expose albedo toward metal F0 mix for deferred GGX.
        float metalTint = 1f - 0.35f * metallic;
        return new Vector4(
            Math.Clamp(r * boost * metalTint, 0f, 1.5f),
            Math.Clamp(g * boost * metalTint, 0f, 1.5f),
            Math.Clamp(b * boost * metalTint, 0f, 1.5f),
            roughness);
    }

    /// <summary>
    /// Stable micro-detail normal perturbation for cinematic cluster shading.
    /// </summary>
    public static Vector3 PerturbClusterNormal(Vector3 baseNormal, int meshletIndex, int triangleIndex, float lod01, Policy? policy = null)
    {
        policy ??= Industrial;
        if (baseNormal.LengthSquared() < 1e-8f)
            baseNormal = Vector3.UnitY;
        else
            baseNormal = Vector3.Normalize(baseNormal);

        uint h = unchecked((uint)(meshletIndex * 83492791) ^ (uint)(triangleIndex * 2654435761));
        float dx = (((h) & 255) / 255f - 0.5f) * 2f;
        float dy = (((h >> 8) & 255) / 255f - 0.5f) * 2f;
        float strength = policy.NormalDetailStrength * (0.35f + 0.65f * lod01);
        Vector3 tangent = MathF.Abs(baseNormal.Y) < 0.99f
            ? Vector3.Normalize(Vector3.Cross(baseNormal, Vector3.UnitY))
            : Vector3.Normalize(Vector3.Cross(baseNormal, Vector3.UnitX));
        Vector3 bitangent = Vector3.Normalize(Vector3.Cross(baseNormal, tangent));
        Vector3 n = Vector3.Normalize(baseNormal + (tangent * dx + bitangent * dy) * strength);
        return n;
    }

    /// <summary>
    /// Opportunistic virtual page key from world position (Nanite-style page streaming).
    /// </summary>
    public static ulong VirtualPageKey(Vector3 worldPos, int lodLevel)
    {
        int px = (int)MathF.Floor(worldPos.X / 4f);
        int py = (int)MathF.Floor(worldPos.Y / 4f);
        int pz = (int)MathF.Floor(worldPos.Z / 4f);
        return ((ulong)(uint)px & 0x1FFFFFul)
             | (((ulong)(uint)py & 0x1FFFFFul) << 21)
             | (((ulong)(uint)pz & 0x1FFFFFul) << 42)
             | (((ulong)(uint)(lodLevel & 0x3FF)) << 54);
    }

    /// <summary>
    /// Contact AO from a batch of SDF distances (Nanite Neural secondary shading).
    /// </summary>
    public static float ContactAoFromSdf(ReadOnlySpan<float> distances, Policy? policy = null)
    {
        policy ??= Industrial;
        if (distances.IsEmpty)
            return 1f;
        float min = float.MaxValue;
        for (int i = 0; i < distances.Length; i++)
            min = MathF.Min(min, MathF.Abs(distances[i]));
        float ao = 1f - policy.ContactAoStrength * MathF.Exp(-min * 4.5f);
        return Math.Clamp(ao, 0.04f, 1f);
    }

    /// <summary>Whether this frame should rebuild meshlets given cadence + dirty flag.</summary>
    public static bool ShouldRebuildMeshlets(int frameIndex, bool geometryDirty, Policy? policy = null)
    {
        policy ??= Industrial;
        if (geometryDirty)
            return true;
        int cadence = Math.Max(1, policy.MeshletTickCadence);
        return (frameIndex % cadence) == 0;
    }

    /// <summary>Selects Nanite policy from a quality preset name.</summary>
    public static Policy PolicyFromPreset(string? presetName)
    {
        if (string.IsNullOrWhiteSpace(presetName))
            return Industrial;
        return presetName.Trim().ToLowerInvariant() switch
        {
            "cinematic" => NaniteCinematicResolve.Cinematic,
            "ultra" or "aaa" => Aaa,
            "high" => Aaa,
            _ => Industrial
        };
    }
}
