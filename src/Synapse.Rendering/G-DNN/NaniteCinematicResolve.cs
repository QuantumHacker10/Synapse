// =============================================================================
// G-DNN — Cinematic Nanite path: primary-device material resolve + page streaming (AAA)
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
        MinPolyResolution = 48,
        MaxPolyResolution = 96,
        MinVisibilityWidth = 768,
        MaxVisibilityWidth = 2048,
        ScreenErrorThresholdPx = 0.55f,
        ContinuousLodBias = 0.62f,
        MeshletTickCadence = 1,
        ClusterAlbedoBoost = 1.38f,
        ContactAoStrength = 0.72f,
        MetallicBias = 0.22f,
        NormalDetailStrength = 0.35f
    };

    /// <summary>
    /// Full-resolution material resolve: expands a (possibly sub-res) visibility
    /// buffer into viewport-sized albedo / normal / roughness MRTs.
    /// When <paramref name="sourceMesh"/> is provided, triangle normals use SDF
    /// gradient vertex normals (AAA geometric fidelity).
    /// </summary>
    public static void ResolveFullResMaterials(
        RasterTarget? visibility,
        IReadOnlyList<NeuralMeshlet>? meshlets,
        float lod01,
        int viewportW,
        int viewportH,
        Span<Vector3> outAlbedo,
        Span<Vector3> outNormal,
        Span<float> outRoughness,
        NeuralPolygonMesh? sourceMesh = null,
        NaniteNeural30.Policy? policy = null)
    {
        policy ??= Cinematic;
        int n = viewportW * viewportH;
        if (outAlbedo.Length < n || outNormal.Length < n || outRoughness.Length < n)
            throw new ArgumentException("Output buffers too small for viewport.");

        outAlbedo.Clear();
        outNormal.Clear();
        outRoughness.Clear();
        if (visibility == null || meshlets == null || meshlets.Count == 0 || viewportW <= 0 || viewportH <= 0)
            return;

        bool hasMesh = sourceMesh?.Positions is { Length: > 0 } && sourceMesh.Normals is { Length: > 0 };

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

                var mat = NaniteNeural30.ResolveClusterMaterial(meshletIdx, triIdx, lod01, policy);
                int idx = y * viewportW + x;
                outAlbedo[idx] = new Vector3(mat.X, mat.Y, mat.Z);
                outRoughness[idx] = mat.W;

                var ml = meshlets[meshletIdx];
                Vector3 nrm = ResolveTriangleNormal(ml, triIdx, sourceMesh, hasMesh);
                outNormal[idx] = NaniteNeural30.PerturbClusterNormal(nrm, meshletIdx, triIdx, lod01, policy);
            }
        }
    }

    private static Vector3 ResolveTriangleNormal(
        NeuralMeshlet ml,
        int triIdx,
        NeuralPolygonMesh? mesh,
        bool hasMesh)
    {
        if (hasMesh && mesh != null &&
            triIdx >= 0 &&
            triIdx * 3 + 2 < ml.LocalIndices.Length)
        {
            int li0 = ml.LocalIndices[triIdx * 3];
            int li1 = ml.LocalIndices[triIdx * 3 + 1];
            int li2 = ml.LocalIndices[triIdx * 3 + 2];
            if ((uint)li0 < (uint)ml.VertexIndices.Length &&
                (uint)li1 < (uint)ml.VertexIndices.Length &&
                (uint)li2 < (uint)ml.VertexIndices.Length)
            {
                int i0 = ml.VertexIndices[li0];
                int i1 = ml.VertexIndices[li1];
                int i2 = ml.VertexIndices[li2];
                if ((uint)i0 < (uint)mesh.Normals.Length &&
                    (uint)i1 < (uint)mesh.Normals.Length &&
                    (uint)i2 < (uint)mesh.Normals.Length)
                {
                    Vector3 n = mesh.Normals[i0] + mesh.Normals[i1] + mesh.Normals[i2];
                    if (n.LengthSquared() > 1e-8f)
                        return Vector3.Normalize(n);
                }

                // Geometric face normal fallback from positions.
                if ((uint)i0 < (uint)mesh.Positions.Length &&
                    (uint)i1 < (uint)mesh.Positions.Length &&
                    (uint)i2 < (uint)mesh.Positions.Length)
                {
                    Vector3 e1 = mesh.Positions[i1] - mesh.Positions[i0];
                    Vector3 e2 = mesh.Positions[i2] - mesh.Positions[i0];
                    Vector3 face = Vector3.Cross(e1, e2);
                    if (face.LengthSquared() > 1e-10f)
                        return Vector3.Normalize(face);
                }
            }
        }

        Vector3 cone = ml.ConeAxis;
        if (cone.LengthSquared() < 1e-6f)
            return Vector3.UnitY;
        return Vector3.Normalize(cone);
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
        return lod01 / (1f + dist * 0.10f);
    }
}
