using System;
using System.Collections.Generic;
using System.Numerics;
using GDNN.Core.DataStructures;

namespace GDNN.Polygonization;

/// <summary>Vue caméra minimale pour la sélection et le culling de clusters.</summary>
public readonly struct CameraView
{
    public required Vector3 Position { get; init; }
    public required Matrix4x4 ViewProjection { get; init; }
    public required float VerticalFovRadians { get; init; }
    public required int ScreenHeightPixels { get; init; }

    public static CameraView CreatePerspectiveLookAt(
        Vector3 position, Vector3 target, float verticalFovRadians,
        float aspect, float nearPlane, float farPlane, int screenHeightPixels)
    {
        var view = Matrix4x4.CreateLookAt(position, target, Vector3.UnitY);
        var projection = Matrix4x4.CreatePerspectiveFieldOfView(
            verticalFovRadians, aspect, nearPlane, farPlane);
        return new CameraView
        {
            Position = position,
            ViewProjection = view * projection,
            VerticalFovRadians = verticalFovRadians,
            ScreenHeightPixels = screenHeightPixels
        };
    }
}

/// <summary>Statistiques mesurées d'une passe de sélection de clusters.</summary>
public sealed class ClusterCullStats
{
    public int SelectedLevel { get; init; }
    public int ClustersTotal { get; init; }
    public int FrustumCulled { get; init; }
    public int BackfaceCulled { get; init; }
    public int VisibleClusters { get; init; }
    public int VisibleTriangles { get; init; }

    public override string ToString() =>
        $"LOD{SelectedLevel}: {VisibleClusters}/{ClustersTotal} clusters visibles " +
        $"(frustum −{FrustumCulled}, backface −{BackfaceCulled}), {VisibleTriangles} tris";
}

/// <summary>
/// Sélection de clusters façon Nanite pour une chaîne de polygones neuronaux :
/// 1. choix du niveau par erreur d'écran garantie (borne voxel) ;
/// 2. culling frustum hiérarchique via un AABBTree de meshlets ;
/// 3. culling backface par cône de normales au niveau cluster
///    (test conservatif standard des pipelines mesh-shader).
/// </summary>
public sealed class NeuralClusterRenderer
{
    private readonly NeuralPolygonLodChain _chain;
    private readonly List<AABBTree<int>> _treesPerLevel = [];
    private readonly List<int> _queryScratch = [];
    private readonly AABB _objectBounds;

    public NeuralClusterRenderer(NeuralPolygonLodChain chain)
    {
        ArgumentNullException.ThrowIfNull(chain);
        if (chain.Levels.Count == 0)
            throw new ArgumentException("LOD chain is empty.", nameof(chain));

        _chain = chain;
        _objectBounds = chain.Levels[0].Mesh.ComputeBounds();

        foreach (var level in chain.Levels)
        {
            var tree = new AABBTree<int>(Math.Max(16, level.Meshlets.Count));
            for (int i = 0; i < level.Meshlets.Count; i++)
                tree.Insert(i, level.Meshlets[i].Bounds);
            _treesPerLevel.Add(tree);
        }
    }

    /// <summary>
    /// Sélectionne les meshlets visibles pour cette caméra et remplit les stats.
    /// </summary>
    public IReadOnlyList<NeuralMeshlet> SelectVisible(
        in CameraView camera, out ClusterCullStats stats, float pixelErrorBudget = 1.0f)
    {
        float distance = DistanceToBounds(camera.Position, _objectBounds);
        var lod = _chain.SelectLod(
            distance, camera.VerticalFovRadians, camera.ScreenHeightPixels, pixelErrorBudget);

        var frustum = Frustum.FromViewProjection(camera.ViewProjection);
        var tree = _treesPerLevel[lod.Level];

        _queryScratch.Clear();
        int inFrustum = tree.QueryFrustum(frustum, _queryScratch);

        var visible = new List<NeuralMeshlet>(inFrustum);
        int backfaceCulled = 0;
        int triangles = 0;

        foreach (int index in _queryScratch)
        {
            var meshlet = lod.Meshlets[index];
            if (IsBackfacing(meshlet, camera.Position))
            {
                backfaceCulled++;
                continue;
            }
            visible.Add(meshlet);
            triangles += meshlet.TriangleCount;
        }

        stats = new ClusterCullStats
        {
            SelectedLevel = lod.Level,
            ClustersTotal = lod.Meshlets.Count,
            FrustumCulled = lod.Meshlets.Count - inFrustum,
            BackfaceCulled = backfaceCulled,
            VisibleClusters = visible.Count,
            VisibleTriangles = triangles
        };
        return visible;
    }

    /// <summary>
    /// Test conservatif du cône de normales : le cluster est entièrement dos à la
    /// caméra si l'angle entre l'axe du cône et la direction de vue dépasse
    /// 90° + demi-angle du cône, avec marge sur l'étendue du cluster.
    /// </summary>
    public static bool IsBackfacing(NeuralMeshlet meshlet, Vector3 cameraPosition)
    {
        float cutoff = meshlet.ConeCutoff;
        if (cutoff <= 1e-3f)
            return false; // cône trop ouvert (≥ 90°) : jamais cullable

        Vector3 center = meshlet.Bounds.Center;
        float radius = meshlet.Bounds.HalfExtents.Length();
        Vector3 toCluster = center - cameraPosition;
        float dist = toCluster.Length();
        if (dist <= radius + 1e-6f)
            return false; // caméra dans le cluster

        // sin du demi-angle du cône (cutoff = cos du demi-angle).
        float sinAngle = MathF.Sqrt(MathF.Max(0f, 1f - cutoff * cutoff));
        return Vector3.Dot(toCluster, meshlet.ConeAxis) >= sinAngle * dist + radius;
    }

    private static float DistanceToBounds(Vector3 point, AABB bounds)
    {
        Vector3 closest = Vector3.Clamp(point, bounds.Min, bounds.Max);
        return (point - closest).Length();
    }
}
