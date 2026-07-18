using System;
using System.Collections.Generic;
using System.Numerics;
using GDNN.Core.DataStructures;

namespace GDNN.Polygonization;

/// <summary>
/// Cluster de polygones (meshlet) prêt pour un pipeline mesh-shader :
/// indices locaux compacts, bornes englobantes et cône de normales
/// pour le culling hiérarchique (frustum + backface au niveau cluster).
/// </summary>
public sealed class NeuralMeshlet
{
    /// <summary>Indices des sommets dans le maillage source (≤ MaxVertices).</summary>
    public required int[] VertexIndices { get; init; }

    /// <summary>Triangles en indexation locale au meshlet (triplets, ≤ MaxTriangles).</summary>
    public required byte[] LocalIndices { get; init; }

    public required AABB Bounds { get; init; }

    /// <summary>Axe moyen du cône de normales.</summary>
    public required Vector3 ConeAxis { get; init; }

    /// <summary>cos(angle) du cône de normales ; -1 = cône dégénéré (pas de culling backface).</summary>
    public required float ConeCutoff { get; init; }

    public int VertexCount => VertexIndices.Length;
    public int TriangleCount => LocalIndices.Length / 3;
}

/// <summary>
/// Découpe un <see cref="NeuralPolygonMesh"/> en meshlets bornés
/// (64 sommets / 124 triangles, limites standard des mesh shaders).
/// Croissance gloutonne par adjacence pour maximiser la localité spatiale.
/// </summary>
public sealed class NeuralMeshletBuilder
{
    public const int MaxVertices = 64;
    public const int MaxTriangles = 124;

    public IReadOnlyList<NeuralMeshlet> Build(NeuralPolygonMesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);

        int triangleCount = mesh.TriangleCount;
        var meshlets = new List<NeuralMeshlet>();
        if (triangleCount == 0)
            return meshlets;

        var adjacency = BuildVertexTriangleAdjacency(mesh);
        var used = new bool[triangleCount];
        int usedCount = 0;
        int seedCursor = 0;

        while (usedCount < triangleCount)
        {
            // Graine : premier triangle libre.
            while (seedCursor < triangleCount && used[seedCursor])
                seedCursor++;

            var localVertexMap = new Dictionary<int, byte>(MaxVertices);
            var localIndices = new List<byte>(MaxTriangles * 3);
            var frontier = new Queue<int>();
            frontier.Enqueue(seedCursor);

            while (frontier.Count > 0 && localIndices.Count / 3 < MaxTriangles)
            {
                int tri = frontier.Dequeue();
                if (used[tri])
                    continue;

                // Compter les nouveaux sommets qu'ajouterait ce triangle.
                int newVertices = 0;
                for (int c = 0; c < 3; c++)
                {
                    if (!localVertexMap.ContainsKey(mesh.Indices[tri * 3 + c]))
                        newVertices++;
                }
                if (localVertexMap.Count + newVertices > MaxVertices)
                    continue;

                used[tri] = true;
                usedCount++;

                for (int c = 0; c < 3; c++)
                {
                    int global = mesh.Indices[tri * 3 + c];
                    if (!localVertexMap.TryGetValue(global, out byte local))
                    {
                        local = (byte)localVertexMap.Count;
                        localVertexMap[global] = local;
                    }
                    localIndices.Add(local);

                    // Étendre la frontière aux triangles voisins par sommet partagé.
                    foreach (int neighbor in adjacency[global])
                    {
                        if (!used[neighbor])
                            frontier.Enqueue(neighbor);
                    }
                }
            }

            meshlets.Add(FinalizeMeshlet(mesh, localVertexMap, localIndices));
        }

        return meshlets;
    }

    private static List<int>[] BuildVertexTriangleAdjacency(NeuralPolygonMesh mesh)
    {
        var adjacency = new List<int>[mesh.VertexCount];
        for (int v = 0; v < adjacency.Length; v++)
            adjacency[v] = [];

        for (int t = 0; t < mesh.TriangleCount; t++)
        {
            adjacency[mesh.Indices[t * 3]].Add(t);
            adjacency[mesh.Indices[t * 3 + 1]].Add(t);
            adjacency[mesh.Indices[t * 3 + 2]].Add(t);
        }
        return adjacency;
    }

    private static NeuralMeshlet FinalizeMeshlet(
        NeuralPolygonMesh mesh, Dictionary<int, byte> localVertexMap, List<byte> localIndices)
    {
        var vertexIndices = new int[localVertexMap.Count];
        foreach (var (global, local) in localVertexMap)
            vertexIndices[local] = global;

        // Bornes.
        Vector3 min = mesh.Positions[vertexIndices[0]];
        Vector3 max = min;
        foreach (int v in vertexIndices)
        {
            min = Vector3.Min(min, mesh.Positions[v]);
            max = Vector3.Max(max, mesh.Positions[v]);
        }

        // Cône de normales : axe = normale moyenne, cutoff = pire cos(angle).
        Vector3 axis = Vector3.Zero;
        foreach (int v in vertexIndices)
            axis += mesh.Normals[v];

        float cutoff;
        if (axis.LengthSquared() < 1e-12f)
        {
            axis = Vector3.UnitY;
            cutoff = -1f; // cône dégénéré : culling backface désactivé
        }
        else
        {
            axis = Vector3.Normalize(axis);
            float minDot = 1f;
            foreach (int v in vertexIndices)
                minDot = MathF.Min(minDot, Vector3.Dot(axis, mesh.Normals[v]));
            cutoff = minDot;
        }

        return new NeuralMeshlet
        {
            VertexIndices = vertexIndices,
            LocalIndices = localIndices.ToArray(),
            Bounds = new AABB(min, max, fromCorners: true),
            ConeAxis = axis,
            ConeCutoff = cutoff
        };
    }
}
