// =============================================================================
// G-DNN — Cinematic Nanite path: primary-device material resolve + page streaming
// =============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using GDNN.Polygonization;

namespace GDNN.Polygonization;

/// <summary>
/// Cinematic Nanite Neural extensions: full-res visibility material resolve and
/// virtual page residency scoring for primary-device present path.
/// </summary>
public static class NaniteCinematicResolve
{
    /// <summary>Cinematic policy: viewport-scale visibility, denser poly extract.</summary>
    public static NaniteNeural30.Policy Cinematic { get; } = new()
    {
        MinPolyResolution = 36,
        MaxPolyResolution = 64,
        MinVisibilityWidth = 512,
        MaxVisibilityWidth = 2048,
        ScreenErrorThresholdPx = 0.75f,
        ContinuousLodBias = 0.55f,
        MeshletTickCadence = 1,
        ClusterAlbedoBoost = 1.45f,
        ContactAoStrength = 0.65f
    };

    /// <summary>
    /// Full-resolution material resolve: expands a (possibly sub-res) visibility
    /// buffer into viewport-sized albedo / normal / roughness MRTs.
    /// </summary>
    public static void ResolveFullResMaterials(
        RasterTarget? visibility,
        IReadOnlyList<NeuralMeshlet>? meshlets,
        float lod01,
        int viewportW,
        int viewportH,
        Span<Vector3> outAlbedo,
        Span<Vector3> outNormal,
        Span<float> outRoughness)
    {
        int n = viewportW * viewportH;
        if (outAlbedo.Length < n || outNormal.Length < n || outRoughness.Length < n)
            throw new ArgumentException("Output buffers too small for viewport.");

        outAlbedo.Clear();
        outNormal.Clear();
        outRoughness.Clear();
        if (visibility == null || meshlets == null || meshlets.Count == 0 || viewportW <= 0 || viewportH <= 0)
            return;

        for (int y = 0; y < viewportH; y++)
        {
            int sy = Math.Min(visibility.Height - 1, y * visibility.Height / viewportH);
            for (int x = 0; x < viewportW; x++)
            {
                int sx = Math.Min(visibility.Width - 1, x * visibility.Width / viewportW);
                if (!visibility.TryDecode(sx, sy, out int meshletIdx, out int triIdx))
                    continue;
                if ((uint)meshletIdx >= (uint)meshlets.Count)
                    continue;

                var mat = NaniteNeural30.ResolveClusterMaterial(meshletIdx, triIdx, lod01, Cinematic);
                int idx = y * viewportW + x;
                outAlbedo[idx] = new Vector3(mat.X, mat.Y, mat.Z);
                outRoughness[idx] = mat.W;

                // Reconstruct a stable cluster normal from meshlet cone / hash.
                var ml = meshlets[meshletIdx];
                Vector3 nrm = ml.ConeAxis;
                if (nrm.LengthSquared() < 1e-6f)
                    nrm = Vector3.UnitY;
                else
                    nrm = Vector3.Normalize(nrm);
                outNormal[idx] = nrm;
            }
        }
    }

    /// <summary>
    /// Scores virtual pages for cinematic streaming (higher = keep resident).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float PageResidencyScore(ulong pageKey, Vector3 cameraPos, float lod01)
    {
        int px = (int)(pageKey & 0x1FFFFFul);
        int py = (int)((pageKey >> 21) & 0x1FFFFFul);
        int pz = (int)((pageKey >> 42) & 0x1FFFFFul);
        // Sign-extend 21-bit
        if ((px & 0x100000) != 0)
            px |= ~0x1FFFFF;
        if ((py & 0x100000) != 0)
            py |= ~0x1FFFFF;
        if ((pz & 0x100000) != 0)
            pz |= ~0x1FFFFF;
        var center = new Vector3(px * 4f + 2f, py * 4f + 2f, pz * 4f + 2f);
        float dist = Vector3.Distance(cameraPos, center);
        return lod01 / (1f + dist * 0.12f);
    }
}
