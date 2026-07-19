using System;
// ============================================================
// FILE: LooseOctree.cs
// PATH: Core/DataStructures/LooseOctree.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Core.DataStructures;

/// <summary>
/// A loose octree variant with relaxed boundary constraints.
/// Items can extend beyond node boundaries, reducing object movement between nodes
/// during updates and improving insertion performance.
/// </summary>
/// <typeparam name="T">The type of items stored in the loose octree.</typeparam>
public sealed class LooseOctree<T> : IDisposable, IEnumerable<LooseOctree<T>.LooseNode>
    where T : notnull
{
    /// <summary>
    /// The looseness factor. Higher values allow items to extend further beyond node boundaries.
    /// A value of 1.0 means no looseness (standard octree behavior).
    /// </summary>
    public float Looseness { get; }

    /// <summary>
    /// Maximum items per node before subdivision.
    /// </summary>
    public int MaxItemsPerNode { get; }

    /// <summary>
    /// Maximum tree depth.
    /// </summary>
    public int MaxDepth { get; }

    /// <summary>
    /// The root node.
    /// </summary>
    public LooseNode Root { get; private set; }

    /// <summary>
    /// Total number of items.
    /// </summary>
    public int Count { get; private set; }

    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Initializes a new loose octree.
    /// </summary>
    /// <param name="bounds">The bounding box of the root node.</param>
    /// <param name="looseness">The looseness factor (default 1.5).</param>
    /// <param name="maxItemsPerNode">Maximum items before subdivision (default 8).</param>
    /// <param name="maxDepth">Maximum tree depth (default 8).</param>
    public LooseOctree(AABB bounds, float looseness = 1.5f, int maxItemsPerNode = 8, int maxDepth = 8)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(looseness, 1.0f);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(maxItemsPerNode, 0);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(maxDepth, 0);

        Looseness = looseness;
        MaxItemsPerNode = maxItemsPerNode;
        MaxDepth = maxDepth;
        Root = CreateNode(bounds, 0);
    }

    /// <summary>
    /// Initializes with center and half-extents.
    /// </summary>
    public LooseOctree(Vector3 center, Vector3 halfExtents, float looseness = 1.5f, int maxItemsPerNode = 8, int maxDepth = 8)
        : this(new AABB(center, halfExtents), looseness, maxItemsPerNode, maxDepth)
    {
    }

    /// <summary>
    /// Inserts an item at the specified position.
    /// </summary>
    /// <param name="item">The item to insert.</param>
    /// <param name="position">The world-space position.</param>
    public void Insert(T item, Vector3 position)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(item);

        lock (_lock)
        {
            InsertInternal(Root, item, position, 0);
            Count++;
        }
    }

    /// <summary>
    /// Inserts an item with a specific bounding box extent.
    /// </summary>
    /// <param name="item">The item to insert.</param>
    /// <param name="position">The center position.</param>
    /// <param name="extent">Half-extent of the item's bounding box.</param>
    public void Insert(T item, Vector3 position, Vector3 extent)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(item);

        lock (_lock)
        {
            InsertInternalExtent(Root, item, position, extent, 0);
            Count++;
        }
    }

    /// <summary>
    /// Removes an item from the loose octree.
    /// </summary>
    public bool Remove(T item)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            bool removed = RemoveInternal(Root, item);
            if (removed)
            {
                Count--;
                PruneEmptyNodes(Root);
            }
            return removed;
        }
    }

    /// <summary>
    /// Moves an item to a new position, potentially reinserting it.
    /// </summary>
    public bool Move(T item, Vector3 newPosition)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            if (RemoveInternal(Root, item))
            {
                InsertInternal(Root, item, newPosition, 0);
                return true;
            }
            return false;
        }
    }

    /// <summary>
    /// Queries for items within a sphere.
    /// </summary>
    public int QuerySphere(Vector3 center, float radius, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        return QuerySphereInternal(Root, center, radius, results);
    }

    /// <summary>
    /// Queries for items within an AABB.
    /// </summary>
    public int QueryAABB(AABB queryBounds, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        return QueryAABBInternal(Root, queryBounds, results);
    }

    /// <summary>
    /// Queries for items within a frustum.
    /// </summary>
    public int QueryFrustum(Frustum frustum, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        return QueryFrustumInternal(Root, frustum, results);
    }

    /// <summary>
    /// Queries along a ray.
    /// </summary>
    public int QueryRay(Ray ray, float maxDistance, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        return QueryRayInternal(Root, ray, maxDistance, results);
    }

    /// <summary>
    /// Gets all items at a specific depth level.
    /// </summary>
    public int GetItemsAtDepth(int depth, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        return GetItemsAtDepthInternal(Root, depth, 0, results);
    }

    /// <summary>
    /// Finds the nearest item to a point.
    /// </summary>
    public bool FindNearest(Vector3 point, Func<T, Vector3, float> distanceFunc, out T? nearest, out float minDistance)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        nearest = default;
        minDistance = float.MaxValue;
        return FindNearestInternal(Root, point, distanceFunc, ref nearest, ref minDistance);
    }

    /// <summary>
    /// Traverses the octree with a visitor callback.
    /// </summary>
    public void Traverse(LooseNodeVisitor visitor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        TraverseInternal(Root, visitor);
    }

    /// <summary>
    /// Gets the total number of nodes in the tree.
    /// </summary>
    public int GetNodeCount()
    {
        return CountNodes(Root);
    }

    /// <summary>
    /// Clears the loose octree.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            ClearNode(Root);
            Root = CreateNode(Root.Bounds, 0);
            Count = 0;
        }
    }

    /// <summary>
    /// Estimates memory usage in bytes.
    /// </summary>
    public long EstimateMemoryUsage()
    {
        int nodeCount = GetNodeCount();
        return (long)nodeCount * (Unsafe.SizeOf<LooseNode>() + MaxItemsPerNode * Unsafe.SizeOf<ItemEntry<T>>());
    }

    #region Private Implementation

    private LooseNode CreateNode(AABB bounds, int depth)
    {
        return new LooseNode
        {
            Bounds = bounds,
            LooseBounds = ComputeLooseBounds(bounds),
            Depth = depth,
            Items = new List<ItemEntry<T>>()
        };
    }

    private AABB ComputeLooseBounds(AABB tightBounds)
    {
        Vector3 loosenessHalfExtents = tightBounds.HalfExtents * Looseness;
        return new AABB(tightBounds.Center, loosenessHalfExtents);
    }

    private void InsertInternal(LooseNode node, T item, Vector3 position, int depth)
    {
        if (depth >= MaxDepth || (node.Items.Count < MaxItemsPerNode && !node.HasChildren))
        {
            node.Items.Add(new ItemEntry<T>(item, position));
            return;
        }

        if (!node.HasChildren)
        {
            Subdivide(node);
        }

        int childIndex = GetChildIndex(node.Bounds, position);
        if (childIndex >= 0 && childIndex < 8)
        {
            // Check if item fits within loose bounds of child
            LooseNode child = node.Children[childIndex];
            if (child.LooseBounds.Contains(position))
            {
                InsertInternal(child, item, position, depth + 1);
            }
            else
            {
                // Item doesn't fit in child's loose bounds, store in current node
                node.Items.Add(new ItemEntry<T>(item, position));
            }
        }
        else
        {
            node.Items.Add(new ItemEntry<T>(item, position));
        }
    }

    private void InsertInternalExtent(LooseNode node, T item, Vector3 position, Vector3 extent, int depth)
    {
        if (depth >= MaxDepth || (node.Items.Count < MaxItemsPerNode && !node.HasChildren))
        {
            node.Items.Add(new ItemEntry<T>(item, position));
            return;
        }

        if (!node.HasChildren)
        {
            Subdivide(node);
        }

        int childIndex = GetChildIndex(node.Bounds, position);
        if (childIndex >= 0 && childIndex < 8)
        {
            LooseNode child = node.Children[childIndex];
            // Test if item's AABB fits within child's loose bounds
            AABB itemBounds = new AABB(position, extent);
            if (AABB.Intersects(child.LooseBounds, itemBounds))
            {
                InsertInternalExtent(child, item, position, extent, depth + 1);
            }
            else
            {
                node.Items.Add(new ItemEntry<T>(item, position));
            }
        }
        else
        {
            node.Items.Add(new ItemEntry<T>(item, position));
        }
    }

    private bool RemoveInternal(LooseNode node, T item)
    {
        for (int i = 0; i < node.Items.Count; i++)
        {
            if (EqualityComparer<T>.Default.Equals(node.Items[i].Item, item))
            {
                node.Items.RemoveAt(i);
                return true;
            }
        }

        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                if (RemoveInternal(node.Children[i], item))
                    return true;
            }
        }

        return false;
    }

    private int QuerySphereInternal(LooseNode node, Vector3 center, float radius, List<T> results)
    {
        // Use loose bounds for early rejection
        float distSq = AABB.SqDistance(node.LooseBounds, center);
        if (distSq > radius * radius)
            return 0;

        int count = 0;
        float radiusSq = radius * radius;

        for (int i = 0; i < node.Items.Count; i++)
        {
            float itemDistSq = Vector3.DistanceSquared(node.Items[i].Position, center);
            if (itemDistSq <= radiusSq)
            {
                results.Add(node.Items[i].Item);
                count++;
            }
        }

        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                count += QuerySphereInternal(node.Children[i], center, radius, results);
            }
        }

        return count;
    }

    private int QueryAABBInternal(LooseNode node, AABB queryBounds, List<T> results)
    {
        if (!AABB.Intersects(node.LooseBounds, queryBounds))
            return 0;

        int count = 0;

        for (int i = 0; i < node.Items.Count; i++)
        {
            if (queryBounds.Contains(node.Items[i].Position))
            {
                results.Add(node.Items[i].Item);
                count++;
            }
        }

        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                count += QueryAABBInternal(node.Children[i], queryBounds, results);
            }
        }

        return count;
    }

    private int QueryFrustumInternal(LooseNode node, Frustum frustum, List<T> results)
    {
        FrustumTest result = frustum.TestAABB(node.LooseBounds);
        if (result == FrustumTest.Outside)
            return 0;

        int count = 0;

        if (result == FrustumTest.Inside)
        {
            CollectAllItems(node, results);
            return CountNodeItems(node);
        }

        for (int i = 0; i < node.Items.Count; i++)
        {
            if (frustum.TestPoint(node.Items[i].Position) != FrustumTest.Outside)
            {
                results.Add(node.Items[i].Item);
                count++;
            }
        }

        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                count += QueryFrustumInternal(node.Children[i], frustum, results);
            }
        }

        return count;
    }

    private int QueryRayInternal(LooseNode node, Ray ray, float maxDistance, List<T> results)
    {
        if (!AABB.IntersectsRay(node.LooseBounds, ray, out float tmin, out float tmax))
            return 0;

        if (tmin > maxDistance)
            return 0;

        int count = 0;

        for (int i = 0; i < node.Items.Count; i++)
        {
            Vector3 toItem = node.Items[i].Position - ray.Position;
            float t = Vector3.Dot(toItem, ray.Direction);

            if (t < 0 || t > maxDistance)
                continue;

            Vector3 closest = ray.Position + ray.Direction * t;
            float distSq = Vector3.DistanceSquared(closest, node.Items[i].Position);

            if (distSq < 1.0f)
            {
                results.Add(node.Items[i].Item);
                count++;
            }
        }

        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                count += QueryRayInternal(node.Children[i], ray, maxDistance, results);
            }
        }

        return count;
    }

    private int GetItemsAtDepthInternal(LooseNode node, int targetDepth, int currentDepth, List<T> results)
    {
        if (currentDepth == targetDepth)
        {
            for (int i = 0; i < node.Items.Count; i++)
            {
                results.Add(node.Items[i].Item);
            }
            return node.Items.Count;
        }

        if (!node.HasChildren)
            return 0;

        int count = 0;
        for (int i = 0; i < 8; i++)
        {
            count += GetItemsAtDepthInternal(node.Children[i], targetDepth, currentDepth + 1, results);
        }
        return count;
    }

    private bool FindNearestInternal(LooseNode node, Vector3 point, Func<T, Vector3, float> distanceFunc, ref T? nearest, ref float minDistance)
    {
        float nodeDist = AABB.SqDistance(node.LooseBounds, point);
        if (nodeDist > minDistance)
            return false;

        bool found = false;

        for (int i = 0; i < node.Items.Count; i++)
        {
            float dist = distanceFunc(node.Items[i].Item, point);
            if (dist < minDistance)
            {
                minDistance = dist;
                nearest = node.Items[i].Item;
                found = true;
            }
        }

        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                if (FindNearestInternal(node.Children[i], point, distanceFunc, ref nearest, ref minDistance))
                    found = true;
            }
        }

        return found;
    }

    private void CollectAllItems(LooseNode node, List<T> results)
    {
        for (int i = 0; i < node.Items.Count; i++)
        {
            results.Add(node.Items[i].Item);
        }

        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                CollectAllItems(node.Children[i], results);
            }
        }
    }

    private int CountNodeItems(LooseNode node)
    {
        int count = node.Items.Count;
        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                count += CountNodeItems(node.Children[i]);
            }
        }
        return count;
    }

    private int CountNodes(LooseNode node)
    {
        int count = 1;
        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                count += CountNodes(node.Children[i]);
            }
        }
        return count;
    }

    private void ClearNode(LooseNode node)
    {
        node.Items.Clear();
        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                ClearNode(node.Children[i]);
            }
            node.Children = null!;
            node.HasChildren = false;
        }
    }

    private void Subdivide(LooseNode node)
    {
        node.Children = new LooseNode[8];
        Vector3 half = node.Bounds.HalfExtents * 0.5f;

        for (int i = 0; i < 8; i++)
        {
            Vector3 offset = new(
                (i & 1) == 0 ? -half.X : half.X,
                (i & 2) == 0 ? -half.Y : half.Y,
                (i & 4) == 0 ? -half.Z : half.Z
            );

            AABB childBounds = new AABB(node.Bounds.Center + offset, half);
            node.Children[i] = CreateNode(childBounds, node.Depth + 1);
        }

        node.HasChildren = true;

        // Redistribute items
        var items = node.Items;
        node.Items = new List<ItemEntry<T>>();

        for (int i = 0; i < items.Count; i++)
        {
            int childIndex = GetChildIndex(node.Bounds, items[i].Position);
            if (childIndex >= 0 && childIndex < 8 &&
                node.Children[childIndex].LooseBounds.Contains(items[i].Position))
            {
                InsertInternal(node.Children[childIndex], items[i].Item, items[i].Position, node.Depth + 1);
            }
            else
            {
                node.Items.Add(items[i]);
            }
        }
    }

    private int GetChildIndex(AABB parentBounds, Vector3 position)
    {
        Vector3 local = position - parentBounds.Center;
        int index = 0;
        if (local.X >= 0)
            index |= 1;
        if (local.Y >= 0)
            index |= 2;
        if (local.Z >= 0)
            index |= 4;
        return index;
    }

    private void PruneEmptyNodes(LooseNode node)
    {
        if (!node.HasChildren)
            return;

        bool allEmpty = true;
        for (int i = 0; i < 8; i++)
        {
            if (node.Children[i].Items.Count > 0 || node.Children[i].HasChildren)
            {
                allEmpty = false;
                break;
            }
        }

        if (allEmpty)
        {
            node.Children = null!;
            node.HasChildren = false;
        }
        else
        {
            for (int i = 0; i < 8; i++)
            {
                PruneEmptyNodes(node.Children[i]);
            }
        }
    }

    private void TraverseInternal(LooseNode node, LooseNodeVisitor visitor)
    {
        visitor(node);
        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                TraverseInternal(node.Children[i], visitor);
            }
        }
    }

    #endregion

    #region Iterator

    /// <inheritdoc/>
    public IEnumerator<LooseNode> GetEnumerator()
    {
        var stack = new Stack<LooseNode>();
        stack.Push(Root);

        while (stack.Count > 0)
        {
            var node = stack.Pop();
            yield return node;

            if (node.HasChildren)
            {
                for (int i = 7; i >= 0; i--)
                    stack.Push(node.Children[i]);
            }
        }
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    #endregion

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            Clear();
            GC.SuppressFinalize(this);
        }
    }

    #region Nested Types

    /// <summary>
    /// Represents a node in the loose octree with both tight and loose bounding boxes.
    /// </summary>
    public sealed class LooseNode
    {
        /// <summary>The tight bounding box of this node.</summary>
        public AABB Bounds { get; internal set; }

        /// <summary>The loose bounding box (expanded by looseness factor).</summary>
        public AABB LooseBounds { get; internal set; }

        /// <summary>Items stored in this node.</summary>
        public List<ItemEntry<T>> Items { get; internal set; } = new();

        /// <summary>Child nodes.</summary>
        public LooseNode[]? Children { get; internal set; }

        /// <summary>Whether this node has children.</summary>
        public bool HasChildren { get; internal set; }

        /// <summary>Depth of this node in the tree.</summary>
        public int Depth { get; internal set; }

        /// <summary>Whether this is a leaf node.</summary>
        public bool IsLeaf => !HasChildren;

        /// <summary>The ratio of loose to tight bounds.</summary>
        public float LooseRatio => LooseBounds.HalfExtents.X / Bounds.HalfExtents.X;
    }

    /// <summary>
    /// Delegate for traversing loose octree nodes.
    /// </summary>
    public delegate void LooseNodeVisitor(LooseNode node);

    #endregion
}

/// <summary>
/// An item entry with its position.
/// </summary>
public readonly struct ItemEntry<T>
{
    /// <summary>The stored item.</summary>
    public readonly T Item;

    /// <summary>The position of the item.</summary>
    public readonly Vector3 Position;

    public ItemEntry(T item, Vector3 position)
    {
        Item = item;
        Position = position;
    }
}
