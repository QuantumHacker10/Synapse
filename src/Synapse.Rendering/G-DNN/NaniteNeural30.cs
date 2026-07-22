// =============================================================================
// G-DNN — Nanite Neural 3.0
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
        public int MinPolyResolution { get; init; } = 24;
        public int MaxPolyResolution { get; init; } = 48;
        public int MinVisibilityWidth { get; init; } = 320;
        public int MaxVisibilityWidth { get; init; } = 768;
        public float ScreenErrorThresholdPx { get; init; } = 1.15f;
        public float ContinuousLodBias { get; init; } = 0.35f;
        public int MeshletTickCadence { get; init; } = 2;
        public float ClusterAlbedoBoost { get; init; } = 1.35f;
        public float ContactAoStrength { get; init; } = 0.55f;
    }

    public static Policy Industrial { get; } = new();

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
        float distTerm = 1f / (1f + MathF.Max(0f, cameraDistance) * 0.08f);
        float errorTerm = Math.Clamp(projectedRadiusPx / MathF.Max(0.5f, policy.ScreenErrorThresholdPx), 0f, 2f);
        float raw = 0.45f * distTerm + 0.55f * MathF.Tanh(errorTerm);
        return Math.Clamp(raw + policy.ContinuousLodBias * 0.15f, 0f, 1f);
    }

    /// <summary>Polygon extract resolution from continuous LOD.</summary>
    public static int PolyResolution(float lod01, int viewportMax, Policy? policy = null)
    {
        policy ??= Industrial;
        int fromViewport = Math.Clamp(viewportMax / 40, policy.MinPolyResolution, policy.MaxPolyResolution);
        int fromLod = (int)MathF.Round(
            policy.MinPolyResolution + lod01 * (policy.MaxPolyResolution - policy.MinPolyResolution));
        return Math.Clamp(Math.Max(fromViewport, fromLod), policy.MinPolyResolution, policy.MaxPolyResolution);
    }

    /// <summary>Visibility-buffer size (Nanite-like cluster tiles).</summary>
    public static (int Width, int Height) VisibilityBufferSize(int viewportW, int viewportH, float lod01, Policy? policy = null)
    {
        policy ??= Industrial;
        float scale = 0.45f + 0.35f * lod01;
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
    /// </summary>
    public static Vector4 ResolveClusterMaterial(int meshletIndex, int triangleIndex, float lod01, Policy? policy = null)
    {
        policy ??= Industrial;
        uint h = unchecked((uint)(meshletIndex * 73856093) ^ (uint)(triangleIndex * 19349663) ^ 0x9E3779B9u);
        float r = ((h) & 255) / 255f;
        float g = ((h >> 8) & 255) / 255f;
        float b = ((h >> 16) & 255) / 255f;
        // Slight desaturation toward industrial PBR neutrals at coarse LOD.
        float sat = 0.55f + 0.45f * lod01;
        float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        r = luma + (r - luma) * sat;
        g = luma + (g - luma) * sat;
        b = luma + (b - luma) * sat;
        float boost = policy.ClusterAlbedoBoost;
        float roughness = 0.25f + 0.55f * (((h >> 24) & 255) / 255f) * (1f - 0.35f * lod01);
        return new Vector4(
            Math.Clamp(r * boost, 0f, 1.5f),
            Math.Clamp(g * boost, 0f, 1.5f),
            Math.Clamp(b * boost, 0f, 1.5f),
            Math.Clamp(roughness, 0.04f, 1f));
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
        float ao = 1f - policy.ContactAoStrength * MathF.Exp(-min * 4f);
        return Math.Clamp(ao, 0.05f, 1f);
    }

    /// <summary>Whether this frame should rebuild meshlets given cadence + dirty flag.</summary>
    public static bool ShouldRebuildMeshlets(int frameIndex, bool geometryDirty, Policy? policy = null)
    {
        policy ??= Industrial;
        if (geometryDirty)
            return true;
        int cadence = Math.Max(1, policy.MeshletTickCadence);
        return (frameIndex % cadence) == 1;
    }
}
