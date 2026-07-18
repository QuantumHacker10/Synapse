using System;
using System.Collections.Generic;
using System.Numerics;
using GDNN.Core.Mathematics;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Evaluation;

/// <summary>
/// Maillage de référence pour mesurer l'erreur géométrique d'un SDF neuronal.
/// Fournit des distances non signées point→triangle (approximation Hausdorff).
/// </summary>
public sealed class ReferenceMeshSdf
{
    private readonly Vector3[] _vertices;
    private readonly int[] _indices; // triplets de triangles

    public int TriangleCount => _indices.Length / 3;
    public int VertexCount => _vertices.Length;

    public ReferenceMeshSdf(Vector3[] vertices, int[] indices)
    {
        if (indices.Length % 3 != 0)
            throw new ArgumentException("Indices must be a multiple of 3.");
        _vertices = vertices ?? throw new ArgumentNullException(nameof(vertices));
        _indices = indices ?? throw new ArgumentNullException(nameof(indices));
    }

    /// <summary>
    /// Distance non signée au maillage (min sur tous les triangles).
    /// </summary>
    public float UnsignedDistance(Vector3 point)
    {
        float minDist = float.MaxValue;
        for (int t = 0; t < _indices.Length; t += 3)
        {
            Vector3 v0 = _vertices[_indices[t]];
            Vector3 v1 = _vertices[_indices[t + 1]];
            Vector3 v2 = _vertices[_indices[t + 2]];
            float d = point.PointToTriangleDistance(v0, v1, v2);
            if (d < minDist) minDist = d;
        }
        return minDist;
    }

    /// <summary>
    /// Remplit un buffer de distances de référence pour un nuage de points.
    /// </summary>
    public void SampleDistances(ReadOnlySpan<Vector3> points, Span<float> distancesOut)
    {
        if (points.Length != distancesOut.Length)
            throw new ArgumentException("Points and distances must match.");

        for (int i = 0; i < points.Length; i++)
            distancesOut[i] = UnsignedDistance(points[i]);
    }

    /// <summary>
    /// Erreur de Hausdorff approximée : max |network(|p|) - meshDist(p)|.
    /// Note : le réseau produit une distance signée ; on compare |sdf| à la distance maillage.
    /// </summary>
    public float ComputeHausdorffError(ISdfNetwork network, ReadOnlySpan<Vector3> samplePoints)
    {
        float maxError = 0f;
        for (int i = 0; i < samplePoints.Length; i++)
        {
            float neural = MathF.Abs(network.Evaluate(samplePoints[i]));
            float mesh = UnsignedDistance(samplePoints[i]);
            maxError = MathF.Max(maxError, MathF.Abs(neural - mesh));
        }
        return maxError;
    }

    /// <summary>
    /// Erreur RMS |network| vs distance maillage.
    /// </summary>
    public float ComputeRmsError(ISdfNetwork network, ReadOnlySpan<Vector3> samplePoints)
    {
        float sumSq = 0f;
        for (int i = 0; i < samplePoints.Length; i++)
        {
            float neural = MathF.Abs(network.Evaluate(samplePoints[i]));
            float mesh = UnsignedDistance(samplePoints[i]);
            float diff = neural - mesh;
            sumSq += diff * diff;
        }
        return MathF.Sqrt(sumSq / samplePoints.Length);
    }

    /// <summary>
    /// Distance signée via parity de raycast (+X) pour maillages fermés orientés.
    /// </summary>
    public float SignedDistance(Vector3 point)
    {
        float ud = UnsignedDistance(point);
        return IsInside(point) ? -ud : ud;
    }

    /// <summary>
    /// Test d'inclusion par comptage d'intersections triangle le long de +X.
    /// </summary>
    public bool IsInside(Vector3 point)
    {
        int hits = 0;
        Vector3 dir = Vector3.UnitX;
        for (int t = 0; t < _indices.Length; t += 3)
        {
            Vector3 v0 = _vertices[_indices[t]];
            Vector3 v1 = _vertices[_indices[t + 1]];
            Vector3 v2 = _vertices[_indices[t + 2]];
            if (RayTriangleIntersect(point, dir, v0, v1, v2, out float dist) && dist > 1e-5f)
                hits++;
        }
        return (hits & 1) == 1;
    }

    /// <summary>
    /// Génère un icosaèdre (approximation sphère) pour les tests sans asset externe.
    /// </summary>
    public static ReferenceMeshSdf CreateUnitSphereIcosahedron(float radius = 0.5f, int subdivisions = 0)
    {
        const float t = 1.6180339887f;
        var raw = new Vector3[]
        {
            new(-1,  t, 0), new( 1,  t, 0), new(-1, -t, 0), new( 1, -t, 0),
            new( 0, -1,  t), new( 0,  1,  t), new( 0, -1, -t), new( 0,  1, -t),
            new( t,  0, -1), new( t,  0,  1), new(-t,  0, -1), new(-t,  0,  1),
        };

        // Use BCL List — GDNN.Evaluation also defines a custom List<T>.
        var vertices = new System.Collections.Generic.List<Vector3>(raw.Length);
        for (int i = 0; i < raw.Length; i++)
            vertices.Add(Vector3.Normalize(raw[i]) * radius);

        int[] baseIndices =
        {
            0,11,5,  0,5,1,  0,1,7,  0,7,10,  0,10,11,
            1,5,9,   5,11,4, 11,10,2, 10,7,6,  7,1,8,
            3,9,4,   3,4,2,  3,2,6,  3,6,8,   3,8,9,
            4,9,5,   2,4,11, 6,2,10, 8,6,7,   9,8,1
        };
        var indices = new System.Collections.Generic.List<int>(baseIndices);

        for (int s = 0; s < subdivisions; s++)
            Subdivide(vertices, indices, radius);

        return new ReferenceMeshSdf(vertices.ToArray(), indices.ToArray());
    }

    /// <summary>
    /// SDF analytique d'une sphère (pour entraînement / comparaison signée).
    /// </summary>
    public static float AnalyticSphereSdf(Vector3 point, float radius = 0.5f) =>
        point.Length() - radius;

    private static void Subdivide(
        System.Collections.Generic.List<Vector3> vertices,
        System.Collections.Generic.List<int> indices,
        float radius)
    {
        var midCache = new Dictionary<(int, int), int>();
        var newIndices = new System.Collections.Generic.List<int>(indices.Count * 4);

        int Midpoint(int i0, int i1)
        {
            var key = i0 < i1 ? (i0, i1) : (i1, i0);
            if (midCache.TryGetValue(key, out int existing))
                return existing;

            Vector3 mid = Vector3.Normalize((vertices[i0] + vertices[i1]) * 0.5f) * radius;
            int idx = vertices.Count;
            vertices.Add(mid);
            midCache[key] = idx;
            return idx;
        }

        for (int i = 0; i < indices.Count; i += 3)
        {
            int a = indices[i], b = indices[i + 1], c = indices[i + 2];
            int ab = Midpoint(a, b);
            int bc = Midpoint(b, c);
            int ca = Midpoint(c, a);

            newIndices.Add(a); newIndices.Add(ab); newIndices.Add(ca);
            newIndices.Add(b); newIndices.Add(bc); newIndices.Add(ab);
            newIndices.Add(c); newIndices.Add(ca); newIndices.Add(bc);
            newIndices.Add(ab); newIndices.Add(bc); newIndices.Add(ca);
        }

        indices.Clear();
        for (int i = 0; i < newIndices.Count; i++)
            indices.Add(newIndices[i]);
    }

    private static bool RayTriangleIntersect(
        Vector3 origin, Vector3 dir, Vector3 v0, Vector3 v1, Vector3 v2, out float t)
    {
        t = 0f;
        const float epsilon = 1e-7f;
        Vector3 e1 = v1 - v0;
        Vector3 e2 = v2 - v0;
        Vector3 pvec = Vector3.Cross(dir, e2);
        float det = Vector3.Dot(e1, pvec);
        if (MathF.Abs(det) < epsilon) return false;

        float invDet = 1f / det;
        Vector3 tvec = origin - v0;
        float u = Vector3.Dot(tvec, pvec) * invDet;
        if (u < 0f || u > 1f) return false;

        Vector3 qvec = Vector3.Cross(tvec, e1);
        float v = Vector3.Dot(dir, qvec) * invDet;
        if (v < 0f || u + v > 1f) return false;

        t = Vector3.Dot(e2, qvec) * invDet;
        return t > epsilon;
    }
}
