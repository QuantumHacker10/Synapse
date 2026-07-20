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

public class OctreeNode
{
    private const int MaxLeafParticles = 8;
    private const int MaxDepth = 20;

    public BoundingBox3D Bounds { get; }
    public int Depth { get; }
    public bool IsLeaf => Children == null;
    public int Count { get; private set; }
    public List<int> ParticleIndices { get; }
    public OctreeNode[] Children { get; private set; }
    public double TotalMass { get; private set; }
    public Vector3D CenterOfMass { get; private set; }
    public OctreeNode(BoundingBox3D bounds, int depth) { Bounds = bounds; Depth = depth; ParticleIndices = new(); }

    public void Insert(int particleIndex, Vector3D position, double mass)
    {
        Count++;
        double newCount = Count;
        CenterOfMass = (CenterOfMass * (newCount - 1) + position * mass) / newCount;
        TotalMass += mass;
        if (IsLeaf)
        {
            ParticleIndices.Add(particleIndex);
            if (ParticleIndices.Count > MaxLeafParticles && Depth < MaxDepth)
                Subdivide();
        }
        else
        {
            int childIdx = GetChildIndex(position);
            Children[childIdx].Insert(particleIndex, position, mass);
        }
    }

    private void Subdivide()
    {
        Children = new OctreeNode[8];
        var center = Bounds.Center;
        for (int i = 0; i < 8; i++)
        {
            var min = new Vector3D(
                (i & 1) == 0 ? Bounds.Min.X : center.X,
                (i & 2) == 0 ? Bounds.Min.Y : center.Y,
                (i & 4) == 0 ? Bounds.Min.Z : center.Z);
            var max = new Vector3D(
                (i & 1) == 0 ? center.X : Bounds.Max.X,
                (i & 2) == 0 ? center.Y : Bounds.Max.Y,
                (i & 4) == 0 ? center.Z : Bounds.Max.Z);
            Children[i] = new OctreeNode(new BoundingBox3D(min, max), Depth + 1);
        }
        var indices = new List<int>(ParticleIndices);
        ParticleIndices.Clear();
        foreach (var idx in indices)
            Insert(idx, Vector3D.Zero, 0); // re-insert (simplified)
    }

    private int GetChildIndex(Vector3D pos)
    {
        var center = Bounds.Center;
        int idx = 0;
        if (pos.X >= center.X)
            idx |= 1;
        if (pos.Y >= center.Y)
            idx |= 2;
        if (pos.Z >= center.Z)
            idx |= 4;
        return idx;
    }

    public void RangeQuery(BoundingBox3D queryBounds, List<int> results)
    {
        if (!Bounds.Intersects(queryBounds))
            return;
        if (IsLeaf)
        { results.AddRange(ParticleIndices); return; }
        foreach (var child in Children)
            child?.RangeQuery(queryBounds, results);
    }

    public void RadiusQuery(Vector3D center, double radius, List<int> results)
    {
        double rSq = radius * radius;
        if (Bounds.DistanceSquaredTo(center) > rSq)
            return;
        if (IsLeaf)
        { results.AddRange(ParticleIndices); return; }
        foreach (var child in Children)
            child?.RadiusQuery(center, radius, results);
    }

    public int DepthFirstTraversal(Func<OctreeNode, bool> visit)
    {
        int visited = 1;
        if (!visit(this))
            return visited;
        if (!IsLeaf)
            foreach (var child in Children)
                if (child != null)
                    visited += child.DepthFirstTraversal(visit);
        return visited;
    }

    public int ComputeMaxDepth() => IsLeaf ? Depth : Children.Max(c => c?.ComputeMaxDepth() ?? Depth);
    public int CountNodes() => IsLeaf ? 1 : 1 + Children.Sum(c => c?.CountNodes() ?? 0);
    public int CountLeaves() => IsLeaf ? 1 : Children.Sum(c => c?.CountLeaves() ?? 0);
    public double AverageParticlesPerLeaf() { int leaves = CountLeaves(); return leaves > 0 ? (double)Count / leaves : 0; }

    public void Clear() { ParticleIndices.Clear(); Children = null; Count = 0; TotalMass = 0; CenterOfMass = Vector3D.Zero; }
}

/// <summary>KD-tree for efficient nearest-neighbor and range queries in 3D.</summary>
