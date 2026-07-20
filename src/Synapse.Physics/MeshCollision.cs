// ============================================================================
// Synapse Omnia — Physics/MeshCollision.cs
// Cook mesh colliders (convex hull / triangle mesh) for Synapse worlds.
// ============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;

namespace Synapse.Physics;

/// <summary>
/// Provides mesh geometry to the physics layer without depending on rendering types.
/// Implemented by Synapse Runtime against MeshLoader / scene assets.
/// </summary>
public interface IMeshProvider
{
    /// <summary>Returns true when a mesh id is known.</summary>
    bool TryGetMesh(string meshId, out MeshCollisionSource source);

    /// <summary>Enumerates registered mesh ids.</summary>
    IEnumerable<string> MeshIds { get; }
}

/// <summary>Raw triangle mesh input for cooking.</summary>
public readonly struct MeshCollisionSource
{
    public readonly Vector3[] Vertices;
    public readonly int[] Indices;
    public readonly string Id;

    public MeshCollisionSource(string id, Vector3[] vertices, int[] indices)
    {
        Id = id;
        Vertices = vertices ?? Array.Empty<Vector3>();
        Indices = indices ?? Array.Empty<int>();
    }

    public int TriangleCount => Indices.Length / 3;
}

/// <summary>
/// Cooks Synapse-friendly collision shapes from mesh sources.
/// Convex hull for dynamics; triangle mesh for static scenery / neural form shells.
/// </summary>
public static class MeshCollisionCooker
{
    /// <summary>Builds an AABB-fit box collider (fast path).</summary>
    public static Collider CookBoundingBox(MeshCollisionSource source)
    {
        if (source.Vertices.Length == 0)
            return new Collider { Shape = ColliderShape.Box, Size = new Vector3(0.5f) };

        Vector3 min = source.Vertices[0], max = source.Vertices[0];
        for (int i = 1; i < source.Vertices.Length; i++)
        {
            min = Vector3.Min(min, source.Vertices[i]);
            max = Vector3.Max(max, source.Vertices[i]);
        }
        return new Collider
        {
            Shape = ColliderShape.Box,
            Size = (max - min) * 0.5f,
            LocalOffset = (min + max) * 0.5f,
            SourceMeshId = source.Id
        };
    }

    /// <summary>
    /// Cooks a convex hull via iterative extreme-point expansion (gift-wrapping lite).
    /// Suitable for dynamic Synapse agents / props.
    /// </summary>
    public static Collider CookConvexHull(MeshCollisionSource source, int maxHullVertices = 64)
    {
        if (source.Vertices.Length == 0)
            return CookBoundingBox(source);

        maxHullVertices = Math.Clamp(maxHullVertices, 4, 256);
        var hull = BuildConvexHull(source.Vertices, maxHullVertices);
        Vector3 min = hull[0], max = hull[0];
        for (int i = 1; i < hull.Length; i++)
        {
            min = Vector3.Min(min, hull[i]);
            max = Vector3.Max(max, hull[i]);
        }

        return new Collider
        {
            Shape = ColliderShape.ConvexHull,
            HullVertices = hull,
            Size = (max - min) * 0.5f,
            LocalOffset = (min + max) * 0.5f,
            SourceMeshId = source.Id
        };
    }

    /// <summary>
    /// Cooks a triangle mesh collider (prefer static/kinematic bodies).
    /// </summary>
    public static Collider CookTriangleMesh(MeshCollisionSource source)
    {
        if (source.Vertices.Length == 0 || source.Indices.Length < 3)
            return CookBoundingBox(source);

        var verts = (Vector3[])source.Vertices.Clone();
        var indices = (int[])source.Indices.Clone();
        Vector3 min = verts[0], max = verts[0];
        for (int i = 1; i < verts.Length; i++)
        {
            min = Vector3.Min(min, verts[i]);
            max = Vector3.Max(max, verts[i]);
        }

        return new Collider
        {
            Shape = ColliderShape.TriangleMesh,
            MeshVertices = verts,
            MeshIndices = indices,
            Size = (max - min) * 0.5f,
            LocalOffset = (min + max) * 0.5f,
            SourceMeshId = source.Id
        };
    }

    /// <summary>
    /// Chooses hull for dynamic bodies and triangle mesh for static scenery —
    /// Synapse Omnia default policy.
    /// </summary>
    public static Collider CookForBodyType(MeshCollisionSource source, BodyType bodyType, int maxHullVertices = 48)
    {
        if (bodyType == BodyType.Dynamic)
            return CookConvexHull(source, maxHullVertices);
        return CookTriangleMesh(source);
    }

    private static Vector3[] BuildConvexHull(Vector3[] points, int maxVerts)
    {
        // Seed with AABB extremes + farthest points (robust industrial subset of Qhull).
        var set = new List<Vector3>(Math.Min(points.Length, maxVerts));
        Vector3 min = points[0], max = points[0];
        for (int i = 1; i < points.Length; i++)
        {
            min = Vector3.Min(min, points[i]);
            max = Vector3.Max(max, points[i]);
        }
        AddUnique(set, min);
        AddUnique(set, max);
        AddUnique(set, new Vector3(min.X, min.Y, max.Z));
        AddUnique(set, new Vector3(min.X, max.Y, min.Z));
        AddUnique(set, new Vector3(max.X, min.Y, min.Z));
        AddUnique(set, new Vector3(max.X, max.Y, max.Z));

        Vector3 center = (min + max) * 0.5f;
        // Iteratively add farthest point from current hull centroid.
        while (set.Count < maxVerts && set.Count < points.Length)
        {
            float bestDist = -1f;
            int bestIdx = -1;
            Vector3 hullCenter = Vector3.Zero;
            for (int i = 0; i < set.Count; i++)
                hullCenter += set[i];
            hullCenter /= set.Count;

            for (int i = 0; i < points.Length; i++)
            {
                float d = (points[i] - hullCenter).LengthSquared();
                bool inside = false;
                for (int h = 0; h < set.Count; h++)
                {
                    if ((points[i] - set[h]).LengthSquared() < 1e-8f)
                    {
                        inside = true;
                        break;
                    }
                }
                if (inside) continue;
                if (d > bestDist)
                {
                    bestDist = d;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) break;
            set.Add(points[bestIdx]);
        }

        if (set.Count < 4)
        {
            // Degenerate — return AABB corners.
            return new[]
            {
                new Vector3(min.X, min.Y, min.Z),
                new Vector3(max.X, min.Y, min.Z),
                new Vector3(min.X, max.Y, min.Z),
                new Vector3(max.X, max.Y, min.Z),
                new Vector3(min.X, min.Y, max.Z),
                new Vector3(max.X, min.Y, max.Z),
                new Vector3(min.X, max.Y, max.Z),
                new Vector3(max.X, max.Y, max.Z)
            };
        }

        _ = center;
        return set.ToArray();
    }

    private static void AddUnique(List<Vector3> list, Vector3 v)
    {
        for (int i = 0; i < list.Count; i++)
            if ((list[i] - v).LengthSquared() < 1e-10f)
                return;
        list.Add(v);
    }
}
