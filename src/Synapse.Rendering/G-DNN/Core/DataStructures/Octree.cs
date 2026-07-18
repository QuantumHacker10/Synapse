using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;


// ============================================================
// FILE: Octree.cs
// PATH: Core/DataStructures/Octree.cs
// ============================================================


using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace GDNN.Core.DataStructures;

/// <summary>
/// A generic octree data structure for spatial partitioning of neural assets.
/// Supports point, sphere, AABB, and frustum queries with thread-safe concurrent operations.
/// Uses pooled node allocation for minimal GC pressure.
/// </summary>
/// <typeparam name="T">The type of items stored in the octree.</typeparam>
public sealed class Octree<T> : IDisposable, IEnumerable<Octree<T>.OctreeNode>
    where T : notnull
{
    /// <summary>
    /// Maximum number of items per node before subdivision.
    /// </summary>
    public int MaxItemsPerNode { get; }

    /// <summary>
    /// Maximum depth of the octree.
    /// </summary>
    public int MaxDepth { get; }

    /// <summary>
    /// The root node of the octree.
    /// </summary>
    public OctreeNode Root { get; private set; }

    /// <summary>
    /// Total number of items stored in the octree.
    /// </summary>
    public int Count { get; private set; }

    private readonly object _lock = new();
    private readonly NodePool _nodePool;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="Octree{T}"/> class.
    /// </summary>
    /// <param name="bounds">The bounding box of the root node.</param>
    /// <param name="maxItemsPerNode">Maximum items before subdivision. Default is 8.</param>
    /// <param name="maxDepth">Maximum tree depth. Default is 8.</param>
    public Octree(AABB bounds, int maxItemsPerNode = 8, int maxDepth = 8)
    {
        MaxItemsPerNode = maxItemsPerNode;
        MaxDepth = maxDepth;
        _nodePool = new NodePool(256);
        Root = _nodePool.Rent(bounds, 0);
    }

    /// <summary>
    /// Initializes a new instance with specified center and half-extents.
    /// </summary>
    public Octree(Vector3 center, Vector3 halfExtents, int maxItemsPerNode = 8, int maxDepth = 8)
        : this(new AABB(center, halfExtents), maxItemsPerNode, maxDepth)
    {
    }

    /// <summary>
    /// Inserts an item into the octree at the specified position.
    /// </summary>
    /// <param name="item">The item to insert.</param>
    /// <param name="position">The world-space position of the item.</param>
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
    /// Inserts an item with an associated bounding box.
    /// </summary>
    /// <param name="item">The item to insert.</param>
    /// <param name="itemBounds">The bounding box of the item.</param>
    public void Insert(T item, AABB itemBounds)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(item);

        lock (_lock)
        {
            InsertInternalBounds(Root, item, itemBounds, 0);
            Count++;
        }
    }

    /// <summary>
    /// Removes an item from the octree.
    /// </summary>
    /// <param name="item">The item to remove.</param>
    /// <returns>True if the item was found and removed; otherwise, false.</returns>
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
    /// Queries the octree for all items whose positions fall within a sphere.
    /// </summary>
    /// <param name="center">The center of the query sphere.</param>
    /// <param name="radius">The radius of the query sphere.</param>
    /// <param name="results">The list to collect results into.</param>
    /// <returns>The number of items found.</returns>
    public int QuerySphere(Vector3 center, float radius, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        return QuerySphereInternal(Root, center, radius, results);
    }

    /// <summary>
    /// Queries the octree for all items within an axis-aligned bounding box.
    /// </summary>
    /// <param name="queryBounds">The AABB to test against.</param>
    /// <param name="results">The list to collect results into.</param>
    /// <returns>The number of items found.</returns>
    public int QueryAABB(AABB queryBounds, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        return QueryAABBInternal(Root, queryBounds, results);
    }

    /// <summary>
    /// Queries the octree for all items within a frustum.
    /// </summary>
    /// <param name="frustum">The frustum planes to test against.</param>
    /// <param name="results">The list to collect results into.</param>
    /// <returns>The number of items found.</returns>
    public int QueryFrustum(Frustum frustum, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        return QueryFrustumInternal(Root, frustum, results);
    }

    /// <summary>
    /// Queries the octree for all items along a ray for raymarching acceleration.
    /// </summary>
    /// <param name="ray">The ray to test.</param>
    /// <param name="maxDistance">Maximum query distance along the ray.</param>
    /// <param name="results">The list to collect results into, ordered by distance.</param>
    /// <returns>The number of items found.</returns>
    public int QueryRay(Ray ray, float maxDistance, List<RayHit<T>> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        return QueryRayInternal(Root, ray, maxDistance, results);
    }

    /// <summary>
    /// Gets all items at a specific depth level in the octree.
    /// </summary>
    /// <param name="depth">The depth level (0 = root).</param>
    /// <param name="results">The list to collect results into.</param>
    /// <returns>The number of items found.</returns>
    public int GetItemsAtDepth(int depth, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);
        ArgumentOutOfRangeException.ThrowIfNegative(depth);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(depth, MaxDepth);

        return GetItemsAtDepthInternal(Root, depth, 0, results);
    }

    /// <summary>
    /// Gets all leaf nodes that contain items.
    /// </summary>
    /// <param name="results">The list to collect leaf nodes into.</param>
    public void GetLeafNodes(List<OctreeNode> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        GetLeafNodesInternal(Root, results);
    }

    /// <summary>
    /// Finds the nearest item to a given point using the octree hierarchy.
    /// </summary>
    /// <param name="point">The query point.</param>
    /// <param name="distanceFunc">Function to compute distance from point to item.</param>
    /// <param name="nearest">The nearest item found, or default.</param>
    /// <param name="minDistance">The distance to the nearest item.</param>
    /// <returns>True if an item was found; otherwise, false.</returns>
    public bool FindNearest(Vector3 point, Func<T, Vector3, float> distanceFunc, out T? nearest, out float minDistance)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        nearest = default;
        minDistance = float.MaxValue;
        return FindNearestInternal(Root, point, distanceFunc, ref nearest, ref minDistance);
    }

    /// <summary>
    /// Performs a brute-force rebuild of the octree from all stored items.
    /// </summary>
    /// <param name="positionFunc">Function to get the position of each item.</param>
    public void Rebuild(Func<T, Vector3> positionFunc)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(positionFunc);

        lock (_lock)
        {
            var allItems = new List<(T Item, Vector3 Position)>();
            CollectAllItems(Root, allItems, positionFunc);

            ReturnNode(Root);
            Root = _nodePool.Rent(new AABB(Vector3.Zero, new Vector3(float.MaxValue)), 0);
            Count = 0;

            foreach (var (item, position) in allItems)
            {
                Insert(item, position);
            }
        }
    }

    /// <summary>
    /// Serializes the octree to a byte array for network transfer or persistence.
    /// </summary>
    /// <param name="serializeItem">Function to serialize an individual item.</param>
    /// <returns>A byte array containing the serialized octree.</returns>
    public byte[] Serialize(Func<T, byte[]> serializeItem)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(serializeItem);

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(MaxItemsPerNode);
        writer.Write(MaxDepth);
        writer.Write(Count);

        // Write root AABB
        writer.Write(Root.Bounds.Center.X);
        writer.Write(Root.Bounds.Center.Y);
        writer.Write(Root.Bounds.Center.Z);
        writer.Write(Root.Bounds.HalfExtents.X);
        writer.Write(Root.Bounds.HalfExtents.Y);
        writer.Write(Root.Bounds.HalfExtents.Z);

        SerializeNode(Root, writer, serializeItem);
        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes an octree from a byte array.
    /// </summary>
    /// <param name="data">The serialized byte array.</param>
    /// <param name="deserializeItem">Function to deserialize an individual item.</param>
    /// <returns>A new octree instance.</returns>
    public static Octree<T> Deserialize(byte[] data, Func<byte[], T> deserializeItem)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        int maxItemsPerNode = reader.ReadInt32();
        int maxDepth = reader.ReadInt32();
        int count = reader.ReadInt32();

        float cx = reader.ReadSingle();
        float cy = reader.ReadSingle();
        float cz = reader.ReadSingle();
        float hx = reader.ReadSingle();
        float hy = reader.ReadSingle();
        float hz = reader.ReadSingle();

        var octree = new Octree<T>(new Vector3(cx, cy, cz), new Vector3(hx, hy, hz), maxItemsPerNode, maxDepth);
        octree.DeserializeNode(octree.Root, reader, deserializeItem);
        octree.Count = count;
        return octree;
    }

    /// <summary>
    /// Clears all items from the octree.
    /// </summary>
    public void Clear()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            ReturnNode(Root);
            Root = _nodePool.Rent(Root.Bounds, 0);
            Count = 0;
        }
    }

    /// <summary>
    /// Concurrently queries the octree with a reader-writer lock for minimal contention.
    /// </summary>
    public int ConcurrentQuerySphere(Vector3 center, float radius, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return QuerySphereInternal(Root, center, radius, results);
    }

    /// <summary>
    /// Attempts to insert an item concurrently without blocking.
    /// </summary>
    public bool TryConcurrentInsert(T item, Vector3 position)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(item);

        lock (_lock)
        {
            InsertInternal(Root, item, position, 0);
            Count++;
            return true;
        }
    }

    /// <summary>
    /// Traverses the octree with a callback for each node.
    /// </summary>
    /// <param name="visitor">The visitor callback.</param>
    public void Traverse(OctreeNodeVisitor visitor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        TraverseInternal(Root, visitor);
    }

    /// <summary>
    /// Performs a level-of-detail traversal, returning items grouped by depth.
    /// </summary>
    /// <param name="viewPosition">The viewer position for LOD selection.</param>
    /// <param name="lodDistances">Distance thresholds for each LOD level.</param>
    /// <param name="results">Items at each LOD level.</param>
    public void LODTraversal(Vector3 viewPosition, float[] lodDistances, List<T>[] results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        LODTraversalInternal(Root, viewPosition, lodDistances, results, 0);
    }

    #region Private Implementation

    private void InsertInternal(OctreeNode node, T item, Vector3 position, int depth)
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
            InsertInternal(node.Children[childIndex], item, position, depth + 1);
        }
        else
        {
            node.Items.Add(new ItemEntry<T>(item, position));
        }
    }

    private void InsertInternalBounds(OctreeNode node, T item, AABB itemBounds, int depth)
    {
        if (depth >= MaxDepth || (node.Items.Count < MaxItemsPerNode && !node.HasChildren))
        {
            node.Items.Add(new ItemEntry<T>(item, node.Bounds.Center));
            return;
        }

        if (!node.HasChildren)
        {
            Subdivide(node);
        }

        int childIndex = GetChildIndex(node.Bounds, node.Bounds.Center);
        if (childIndex >= 0 && childIndex < 8)
        {
            InsertInternalBounds(node.Children[childIndex], item, itemBounds, depth + 1);
        }
        else
        {
            node.Items.Add(new ItemEntry<T>(item, node.Bounds.Center));
        }
    }

    private bool RemoveInternal(OctreeNode node, T item)
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
                {
                    return true;
                }
            }
        }

        return false;
    }

    private int QuerySphereInternal(OctreeNode node, Vector3 center, float radius, List<T> results)
    {
        float distSq = AABB.SqDistance(node.Bounds, center);
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

    private int QueryAABBInternal(OctreeNode node, AABB queryBounds, List<T> results)
    {
        if (!AABB.Intersects(node.Bounds, queryBounds))
            return 0;

        int count = 0;
        for (int i = 0; i < node.Items.Count; i++)
        {
            results.Add(node.Items[i].Item);
            count++;
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

    private int QueryFrustumInternal(OctreeNode node, Frustum frustum, List<T> results)
    {
        FrustumTest result = frustum.TestAABB(node.Bounds);
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

    private int QueryRayInternal(OctreeNode node, Ray ray, float maxDistance, List<RayHit<T>> results)
    {
        if (!AABB.IntersectsRay(node.Bounds, ray, out float tmin, out float tmax))
            return 0;

        if (tmin > maxDistance)
            return 0;

        int count = 0;

        for (int i = 0; i < node.Items.Count; i++)
        {
            Vector3 toItem = node.Items[i].Position - ray.Position;
            float t = Vector3.Dot(toItem, ray.Direction);
            if (t < 0 || t > maxDistance) continue;

            Vector3 closest = ray.Position + ray.Direction * t;
            float distSq = Vector3.DistanceSquared(closest, node.Items[i].Position);

            if (distSq < 1.0f)
            {
                results.Add(new RayHit<T>(node.Items[i].Item, t, node.Items[i].Position));
                count++;
            }
        }

        if (node.HasChildren)
        {
            // Sort children by ray entry distance for front-to-back traversal
            Span<int> order = stackalloc int[8];
            for (int i = 0; i < 8; i++) order[i] = i;

            SortChildrenByRay(node, ray, order);

            for (int i = 0; i < 8; i++)
            {
                count += QueryRayInternal(node.Children[order[i]], ray, maxDistance, results);
            }
        }

        return count;
    }

    private unsafe void SortChildrenByRay(OctreeNode node, Ray ray, Span<int> order)
    {
        float* distances = stackalloc float[8];
        for (int i = 0; i < 8; i++)
        {
            Vector3 childCenter = node.Children[i].Bounds.Center;
            Vector3 toChild = childCenter - ray.Position;
            distances[i] = Vector3.Dot(toChild, ray.Direction);
        }

        // Simple insertion sort for 8 elements
        for (int i = 1; i < 8; i++)
        {
            int key = order[i];
            float keyDist = distances[key];
            int j = i - 1;

            while (j >= 0 && distances[order[j]] > keyDist)
            {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }
    }

    private int GetItemsAtDepthInternal(OctreeNode node, int targetDepth, int currentDepth, List<T> results)
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

    private void GetLeafNodesInternal(OctreeNode node, List<OctreeNode> results)
    {
        if (!node.HasChildren)
        {
            results.Add(node);
            return;
        }

        for (int i = 0; i < 8; i++)
        {
            GetLeafNodesInternal(node.Children[i], results);
        }
    }

    private bool FindNearestInternal(OctreeNode node, Vector3 point, Func<T, Vector3, float> distanceFunc, ref T? nearest, ref float minDistance)
    {
        float nodeDist = AABB.SqDistance(node.Bounds, point);
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

    private void LODTraversalInternal(OctreeNode node, Vector3 viewPosition, float[] lodDistances, List<T>[] results, int depth)
    {
        float dist = MathF.Sqrt(AABB.SqDistance(node.Bounds, viewPosition));

        int lodLevel = lodDistances.Length - 1;
        for (int i = 0; i < lodDistances.Length; i++)
        {
            if (dist < lodDistances[i])
            {
                lodLevel = i;
                break;
            }
        }

        if (depth >= lodLevel || !node.HasChildren)
        {
            for (int i = 0; i < node.Items.Count; i++)
            {
                if (lodLevel < results.Length)
                    results[lodLevel].Add(node.Items[i].Item);
            }
        }
        else
        {
            for (int i = 0; i < 8; i++)
            {
                LODTraversalInternal(node.Children[i], viewPosition, lodDistances, results, depth + 1);
            }
        }
    }

    private void Subdivide(OctreeNode node)
    {
        node.Children = new OctreeNode[8];
        Vector3 half = node.Bounds.HalfExtents * 0.5f;

        for (int i = 0; i < 8; i++)
        {
            Vector3 offset = new(
                (i & 1) == 0 ? -half.X : half.X,
                (i & 2) == 0 ? -half.Y : half.Y,
                (i & 4) == 0 ? -half.Z : half.Z
            );

            node.Children[i] = _nodePool.Rent(
                new AABB(node.Bounds.Center + offset, half),
                node.Depth + 1
            );
        }

        node.HasChildren = true;

        // Redistribute items to children
        var items = node.Items;
        node.Items = new List<ItemEntry<T>>(MaxItemsPerNode);

        for (int i = 0; i < items.Count; i++)
        {
            int childIndex = GetChildIndex(node.Bounds, items[i].Position);
            if (childIndex >= 0 && childIndex < 8)
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
        if (local.X >= 0) index |= 1;
        if (local.Y >= 0) index |= 2;
        if (local.Z >= 0) index |= 4;
        return index;
    }

    private void PruneEmptyNodes(OctreeNode node)
    {
        if (!node.HasChildren) return;

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
            for (int i = 0; i < 8; i++)
            {
                ReturnNode(node.Children[i]);
            }
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

    private void CollectAllItems(OctreeNode node, List<T> results)
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

    private void CollectAllItems(OctreeNode node, List<(T Item, Vector3 Position)> results, Func<T, Vector3> positionFunc)
    {
        for (int i = 0; i < node.Items.Count; i++)
        {
            results.Add((node.Items[i].Item, positionFunc(node.Items[i].Item)));
        }

        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                CollectAllItems(node.Children[i], results, positionFunc);
            }
        }
    }

    private int CountNodeItems(OctreeNode node)
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

    private void TraverseInternal(OctreeNode node, OctreeNodeVisitor visitor)
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

    private void SerializeNode(OctreeNode node, BinaryWriter writer, Func<T, byte[]> serializeItem)
    {
        writer.Write(node.Items.Count);
        writer.Write(node.HasChildren);

        for (int i = 0; i < node.Items.Count; i++)
        {
            writer.Write(node.Items[i].Position.X);
            writer.Write(node.Items[i].Position.Y);
            writer.Write(node.Items[i].Position.Z);
            byte[] itemData = serializeItem(node.Items[i].Item);
            writer.Write(itemData.Length);
            writer.Write(itemData);
        }

        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                SerializeNode(node.Children[i], writer, serializeItem);
            }
        }
    }

    private void DeserializeNode(OctreeNode node, BinaryReader reader, Func<byte[], T> deserializeItem)
    {
        int itemCount = reader.ReadInt32();
        bool hasChildren = reader.ReadBoolean();

        node.Items = new List<ItemEntry<T>>(itemCount);
        for (int i = 0; i < itemCount; i++)
        {
            float x = reader.ReadSingle();
            float y = reader.ReadSingle();
            float z = reader.ReadSingle();
            int dataLen = reader.ReadInt32();
            byte[] data = reader.ReadBytes(dataLen);
            T item = deserializeItem(data);
            node.Items.Add(new ItemEntry<T>(item, new Vector3(x, y, z)));
        }

        if (hasChildren)
        {
            SubdivideForDeserialization(node);
            for (int i = 0; i < 8; i++)
            {
                DeserializeNode(node.Children[i], reader, deserializeItem);
            }
        }
    }

    private void SubdivideForDeserialization(OctreeNode node)
    {
        node.Children = new OctreeNode[8];
        Vector3 half = node.Bounds.HalfExtents * 0.5f;

        for (int i = 0; i < 8; i++)
        {
            Vector3 offset = new(
                (i & 1) == 0 ? -half.X : half.X,
                (i & 2) == 0 ? -half.Y : half.Y,
                (i & 4) == 0 ? -half.Z : half.Z
            );

            node.Children[i] = _nodePool.Rent(
                new AABB(node.Bounds.Center + offset, half),
                node.Depth + 1
            );
        }
        node.HasChildren = true;
    }

    private void ReturnNode(OctreeNode node)
    {
        if (node.HasChildren)
        {
            for (int i = 0; i < 8; i++)
            {
                ReturnNode(node.Children[i]);
            }
        }
        _nodePool.Return(node);
    }

    /// <inheritdoc/>
    public IEnumerator<OctreeNode> GetEnumerator()
    {
        var stack = new Stack<OctreeNode>();
        stack.Push(Root);

        while (stack.Count > 0)
        {
            var node = stack.Pop();
            yield return node;

            if (node.HasChildren)
            {
                for (int i = 7; i >= 0; i--)
                {
                    stack.Push(node.Children[i]);
                }
            }
        }
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _nodePool.Dispose();
            GC.SuppressFinalize(this);
        }
    }

    #endregion

    #region Nested Types

    /// <summary>
    /// Represents a node in the octree hierarchy.
    /// </summary>
    public sealed class OctreeNode
    {
        /// <summary>
        /// The bounding box of this node.
        /// </summary>
        public AABB Bounds { get; internal set; }

        /// <summary>
        /// Items stored directly in this node.
        /// </summary>
        public List<ItemEntry<T>> Items { get; internal set; } = new();

        /// <summary>
        /// Child nodes (null if leaf or empty).
        /// </summary>
        public OctreeNode[]? Children { get; internal set; }

        /// <summary>
        /// Whether this node has been subdivided.
        /// </summary>
        public bool HasChildren { get; internal set; }

        /// <summary>
        /// The depth of this node in the tree.
        /// </summary>
        public int Depth { get; internal set; }

        /// <summary>
        /// Whether this node is a leaf (no children).
        /// </summary>
        public bool IsLeaf => !HasChildren;

        /// <summary>
        /// Total items in this node and all descendants.
        /// </summary>
        public int TotalItemCount { get; internal set; }
    }

    // ItemEntry<T> is defined in LooseOctree.cs within this namespace.

    /// <summary>
    /// Delegate for traversing octree nodes.
    /// </summary>
    public delegate void OctreeNodeVisitor(OctreeNode node);

    #endregion

    #region Node Pool

    /// <summary>
    /// Pool for recycling octree nodes to minimize allocations.
    /// </summary>
    private sealed class NodePool : IDisposable
    {
        private readonly ConcurrentBag<OctreeNode> _pool;
        private readonly int _initialCapacity;
        private bool _disposed;

        public NodePool(int initialCapacity)
        {
            _initialCapacity = initialCapacity;
            _pool = new ConcurrentBag<OctreeNode>();

            for (int i = 0; i < initialCapacity; i++)
            {
                _pool.Add(new OctreeNode());
            }
        }

        public OctreeNode Rent(AABB bounds, int depth)
        {
            if (_pool.TryTake(out OctreeNode? node))
            {
                node.Bounds = bounds;
                node.Items = new List<ItemEntry<T>>();
                node.Children = null;
                node.HasChildren = false;
                node.Depth = depth;
                node.TotalItemCount = 0;
                return node;
            }

            return new OctreeNode { Bounds = bounds, Depth = depth };
        }

        public void Return(OctreeNode node)
        {
            if (_disposed) return;

            node.Items.Clear();
            node.Children = null;
            node.HasChildren = false;
            _pool.Add(node);
        }

        public void Dispose()
        {
            _disposed = true;
            _pool.Clear();
        }
    }

    #endregion
}

/// <summary>
/// Axis-Aligned Bounding Box used for spatial queries.
/// </summary>
public struct AABB : IEquatable<AABB>
{
    /// <summary>The center of the bounding box.</summary>
    public Vector3 Center;

    /// <summary>The half-extents along each axis.</summary>
    public Vector3 HalfExtents;

    /// <summary>Minimum corner of the bounding box.</summary>
    public Vector3 Min => Center - HalfExtents;

    /// <summary>Maximum corner of the bounding box.</summary>
    public Vector3 Max => Center + HalfExtents;

    /// <summary>Size along each axis.</summary>
    public Vector3 Size => HalfExtents * 2.0f;

    /// <summary>Surface area of the bounding box.</summary>
    public float SurfaceArea => 8.0f * (HalfExtents.X * HalfExtents.Y + HalfExtents.Y * HalfExtents.Z + HalfExtents.Z * HalfExtents.X);

    /// <summary>Volume of the bounding box.</summary>
    public float Volume => 8.0f * HalfExtents.X * HalfExtents.Y * HalfExtents.Z;

    public AABB(Vector3 center, Vector3 halfExtents)
    {
        Center = center;
        HalfExtents = halfExtents;
    }

    public AABB(Vector3 min, Vector3 max, bool fromCorners)
    {
        Center = (min + max) * 0.5f;
        HalfExtents = (max - min) * 0.5f;
    }

    /// <summary>Expands this AABB to include the given point.</summary>
    public void ExpandToInclude(Vector3 point)
    {
        Vector3 min = Min;
        Vector3 max = Max;
        min = Vector3.Min(min, point);
        max = Vector3.Max(max, point);
        Center = (min + max) * 0.5f;
        HalfExtents = (max - min) * 0.5f;
    }

    /// <summary>Expands this AABB to include another AABB.</summary>
    public void ExpandToInclude(AABB other)
    {
        Vector3 otherMin = other.Min;
        Vector3 otherMax = other.Max;
        Vector3 myMin = Min;
        Vector3 myMax = Max;

        myMin = Vector3.Min(myMin, otherMin);
        myMax = Vector3.Max(myMax, otherMax);
        Center = (myMin + myMax) * 0.5f;
        HalfExtents = (myMax - myMin) * 0.5f;
    }

    /// <summary>Tests if a point is inside this AABB.</summary>
    public bool Contains(Vector3 point)
    {
        Vector3 d = Vector3.Abs(point - Center);
        return d.X <= HalfExtents.X && d.Y <= HalfExtents.Y && d.Z <= HalfExtents.Z;
    }

    /// <summary>Squared distance from a point to the closest point on the AABB.</summary>
    public static float SqDistance(AABB box, Vector3 point)
    {
        Vector3 closest = Vector3.Clamp(point, box.Min, box.Max);
        return Vector3.DistanceSquared(point, closest);
    }

    /// <summary>Tests if two AABBs intersect.</summary>
    public static bool Intersects(AABB a, AABB b)
    {
        Vector3 d = Vector3.Abs(a.Center - b.Center);
        Vector3 combinedExtents = a.HalfExtents + b.HalfExtents;
        return d.X <= combinedExtents.X && d.Y <= combinedExtents.Y && d.Z <= combinedExtents.Z;
    }

    /// <summary>Tests ray-AABB intersection.</summary>
    public static bool IntersectsRay(AABB box, Ray ray, out float tmin, out float tmax)
    {
        tmin = float.MinValue;
        tmax = float.MaxValue;

        Vector3 invDir = new(
            MathF.Abs(ray.Direction.X) < 1e-8f ? float.MaxValue : 1.0f / ray.Direction.X,
            MathF.Abs(ray.Direction.Y) < 1e-8f ? float.MaxValue : 1.0f / ray.Direction.Y,
            MathF.Abs(ray.Direction.Z) < 1e-8f ? float.MaxValue : 1.0f / ray.Direction.Z
        );

        Vector3 localOrigin = ray.Position - box.Center;
        Vector3 r0 = (localOrigin - box.HalfExtents) * invDir;
        Vector3 r1 = (localOrigin + box.HalfExtents) * invDir;

        Vector3 tminV = Vector3.Min(r0, r1);
        Vector3 tmaxV = Vector3.Max(r0, r1);

        tmin = MathF.Max(tminV.X, MathF.Max(tminV.Y, tminV.Z));
        tmax = MathF.Min(tmaxV.X, MathF.Min(tmaxV.Y, tmaxV.Z));

        return tmin <= tmax && tmax >= 0;
    }

    /// <summary>Merges two AABBs.</summary>
    public static AABB Merge(AABB a, AABB b)
    {
        Vector3 min = Vector3.Min(a.Min, b.Min);
        Vector3 max = Vector3.Max(a.Max, b.Max);
        return new AABB((min + max) * 0.5f, (max - min) * 0.5f);
    }

    public bool Equals(AABB other) => Center.Equals(other.Center) && HalfExtents.Equals(other.HalfExtents);
    public override bool Equals(object? obj) => obj is AABB other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(Center, HalfExtents);
    public static bool operator ==(AABB left, AABB right) => left.Equals(right);
    public static bool operator !=(AABB left, AABB right) => !left.Equals(right);
    public override string ToString() => $"AABB(Center={Center}, HalfExtents={HalfExtents})";
}

/// <summary>
/// Represents a view frustum defined by six planes.
/// </summary>
public readonly struct Frustum
{
    /// <summary>The six frustum planes (left, right, bottom, top, near, far).</summary>
    public readonly Plane Left, Right, Bottom, Top, Near, Far;

    public Frustum(Plane left, Plane right, Plane bottom, Plane top, Plane near, Plane far)
    {
        Left = left;
        Right = right;
        Bottom = bottom;
        Top = top;
        Near = near;
        Far = far;
    }

    /// <summary>Constructs a frustum from a view-projection matrix.</summary>
    public static Frustum FromViewProjection(Matrix4x4 viewProjection)
    {
        Plane left = new(
            viewProjection.M14 + viewProjection.M11,
            viewProjection.M24 + viewProjection.M21,
            viewProjection.M34 + viewProjection.M31,
            viewProjection.M44 + viewProjection.M41
        );
        Plane right = new(
            viewProjection.M14 - viewProjection.M11,
            viewProjection.M24 - viewProjection.M21,
            viewProjection.M34 - viewProjection.M31,
            viewProjection.M44 - viewProjection.M41
        );
        Plane bottom = new(
            viewProjection.M14 + viewProjection.M12,
            viewProjection.M24 + viewProjection.M22,
            viewProjection.M34 + viewProjection.M32,
            viewProjection.M44 + viewProjection.M42
        );
        Plane top = new(
            viewProjection.M14 - viewProjection.M12,
            viewProjection.M24 - viewProjection.M22,
            viewProjection.M34 - viewProjection.M32,
            viewProjection.M44 - viewProjection.M42
        );
        Plane near = new(
            viewProjection.M14 + viewProjection.M13,
            viewProjection.M24 + viewProjection.M23,
            viewProjection.M34 + viewProjection.M33,
            viewProjection.M44 + viewProjection.M43
        );
        Plane far = new(
            viewProjection.M14 - viewProjection.M13,
            viewProjection.M24 - viewProjection.M23,
            viewProjection.M34 - viewProjection.M33,
            viewProjection.M44 - viewProjection.M43
        );

        left.Normalize();
        right.Normalize();
        bottom.Normalize();
        top.Normalize();
        near.Normalize();
        far.Normalize();

        return new Frustum(left, right, bottom, top, near, far);
    }

    /// <summary>Tests a point against the frustum.</summary>
    public FrustumTest TestPoint(Vector3 point)
    {
        if (TestPlane(Left, point) < 0) return FrustumTest.Outside;
        if (TestPlane(Right, point) < 0) return FrustumTest.Outside;
        if (TestPlane(Bottom, point) < 0) return FrustumTest.Outside;
        if (TestPlane(Top, point) < 0) return FrustumTest.Outside;
        if (TestPlane(Near, point) < 0) return FrustumTest.Outside;
        if (TestPlane(Far, point) < 0) return FrustumTest.Outside;
        return FrustumTest.Inside;
    }

    /// <summary>Tests an AABB against the frustum.</summary>
    public FrustumTest TestAABB(AABB box)
    {
        int insideCount = 0;

        if (TestPlaneAABB(Left, box) >= 0) insideCount++;
        if (TestPlaneAABB(Right, box) >= 0) insideCount++;
        if (TestPlaneAABB(Bottom, box) >= 0) insideCount++;
        if (TestPlaneAABB(Top, box) >= 0) insideCount++;
        if (TestPlaneAABB(Near, box) >= 0) insideCount++;
        if (TestPlaneAABB(Far, box) >= 0) insideCount++;

        if (insideCount == 0) return FrustumTest.Outside;
        if (insideCount == 6) return FrustumTest.Inside;
        return FrustumTest.Intersecting;
    }

    private static float TestPlane(Plane plane, Vector3 point)
    {
        return plane.X * point.X + plane.Y * point.Y + plane.Z * point.Z + plane.D;
    }

    private static float TestPlaneAABB(Plane plane, AABB box)
    {
        Vector3 positiveVertex = box.Center;

        if (plane.X >= 0) positiveVertex.X += box.HalfExtents.X;
        else positiveVertex.X -= box.HalfExtents.X;

        if (plane.Y >= 0) positiveVertex.Y += box.HalfExtents.Y;
        else positiveVertex.Y -= box.HalfExtents.Y;

        if (plane.Z >= 0) positiveVertex.Z += box.HalfExtents.Z;
        else positiveVertex.Z -= box.HalfExtents.Z;

        return TestPlane(plane, positiveVertex);
    }
}

/// <summary>Result of a frustum test.</summary>
public enum FrustumTest
{
    Outside,
    Intersecting,
    Inside
}

/// <summary>
/// Represents a ray for spatial queries.
/// </summary>
public readonly struct Ray
{
    /// <summary>The origin of the ray.</summary>
    public readonly Vector3 Position;

    /// <summary>The direction of the ray (normalized).</summary>
    public readonly Vector3 Direction;

    public Ray(Vector3 position, Vector3 direction)
    {
        Position = position;
        Direction = Vector3.Normalize(direction);
    }

    /// <summary>Gets a point along the ray at distance t.</summary>
    public Vector3 GetPoint(float t) => Position + Direction * t;

    public override string ToString() => $"Ray(Pos={Position}, Dir={Direction})";
}

/// <summary>
/// Simple Plane struct for frustum culling.
/// </summary>
public struct Plane
{
    public float X, Y, Z, D;

    public Plane(float x, float y, float z, float d)
    {
        X = x; Y = y; Z = z; D = d;
    }

    public void Normalize()
    {
        float length = MathF.Sqrt(X * X + Y * Y + Z * Z);
        if (length > 1e-8f)
        {
            float inv = 1.0f / length;
            X *= inv;
            Y *= inv;
            Z *= inv;
            D *= inv;
        }
    }

    public static float Dot(Plane plane, Vector3 point)
    {
        return plane.X * point.X + plane.Y * point.Y + plane.Z * point.Z + plane.D;
    }
}
