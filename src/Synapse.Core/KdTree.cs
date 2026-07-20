// SYNAPSE OMNIA — Synapse.Core
// Split from PhysicsState.cs for maintainability.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Synapse.Core;

public class KdTree
{
    private class KdNode
    {
        public int PointIndex;
        public int Axis;
        public KdNode? Left, Right;
        public BoundingBox3D Bounds;
    }

    private KdNode _root;
    private readonly List<Vector3D> _points;
    private readonly int[] _indices;

    public int Count => _points.Count;
    public int MaxDepth { get; private set; }

    public KdTree(List<Vector3D> points)
    {
        _points = points;
        _indices = new int[points.Count];
        for (int i = 0; i < points.Count; i++)
            _indices[i] = i;
        if (points.Count > 0)
            _root = Build(0, points.Count - 1, 0);
    }

    private KdNode? Build(int lo, int hi, int depth)
    {
        if (lo > hi)
            return null;
        int axis = depth % 3;
        int mid = (lo + hi) / 2;
        SortByAxis(lo, hi, mid, axis);
        var node = new KdNode { PointIndex = _indices[mid], Axis = axis };
        node.Left = Build(lo, mid - 1, depth + 1);
        node.Right = Build(mid + 1, hi, depth + 1);
        return node;
    }

    private void SortByAxis(int lo, int hi, int mid, int axis)
    {
        for (int i = lo; i < hi; i++)
        {
            int minIdx = i;
            for (int j = i + 1; j <= hi; j++)
                if (GetComponent(_indices[j], axis) < GetComponent(_indices[minIdx], axis))
                    minIdx = j;
            if (minIdx != i)
            { int t = _indices[i]; _indices[i] = _indices[minIdx]; _indices[minIdx] = t; }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double GetComponent(int idx, int axis)
        => axis == 0 ? _points[idx].X : axis == 1 ? _points[idx].Y : _points[idx].Z;

    /// <summary>Find the K nearest neighbors to a query point.</summary>
    public List<(int index, double distanceSquared)> KNearest(Vector3D query, int k)
    {
        var results = new List<(int, double)>();
        KNearestSearch(_root, query, k, results);
        return results.OrderByDescending(x => x.Item2).Take(k).ToList();
    }

    private void KNearestSearch(KdNode node, Vector3D query, int k, List<(int, double)> results)
    {
        if (node == null)
            return;
        double dx = query.X - _points[node.PointIndex].X;
        double dy = query.Y - _points[node.PointIndex].Y;
        double dz = query.Z - _points[node.PointIndex].Z;
        double distSq = dx * dx + dy * dy + dz * dz;
        if (results.Count < k || distSq < results[0].Item2)
        {
            results.Add((node.PointIndex, distSq));
            if (results.Count > k)
                results.Sort((a, b) => b.Item2.CompareTo(a.Item2));
            if (results.Count > k)
                results.RemoveAt(0);
        }
        double diff = GetComponent(node.PointIndex, node.Axis) - (node.Axis == 0 ? query.X : node.Axis == 1 ? query.Y : query.Z);
        var near = diff < 0 ? node.Right : node.Left;
        var far = diff < 0 ? node.Left : node.Right;
        KNearestSearch(near, query, k, results);
        if (results.Count < k || diff * diff < results[0].Item2)
            KNearestSearch(far, query, k, results);
    }

    /// <summary>Find all points within a radius.</summary>
    public List<int> RangeQuery(Vector3D center, double radius)
    {
        var results = new List<int>();
        RangeSearch(_root, center, radius * radius, results);
        return results;
    }

    private void RangeSearch(KdNode node, Vector3D center, double rSq, List<int> results)
    {
        if (node == null)
            return;
        double dx = center.X - _points[node.PointIndex].X;
        double dy = center.Y - _points[node.PointIndex].Y;
        double dz = center.Z - _points[node.PointIndex].Z;
        double distSq = dx * dx + dy * dy + dz * dz;
        if (distSq <= rSq)
            results.Add(node.PointIndex);
        double diff = GetComponent(node.PointIndex, node.Axis) - (node.Axis == 0 ? center.X : node.Axis == 1 ? center.Y : center.Z);
        var near = diff < 0 ? node.Right : node.Left;
        var far = diff < 0 ? node.Left : node.Right;
        RangeSearch(near, center, rSq, results);
        if (diff * diff <= rSq)
            RangeSearch(far, center, rSq, results);
    }

    /// <summary>Find the single closest point.</summary>
    public int NearestNeighbor(Vector3D query) => KNearest(query, 1)[0].index;
}

/// <summary>Spatial hashing for uniform grid-based particle lookup. O(1) average case neighbor search.</summary>
