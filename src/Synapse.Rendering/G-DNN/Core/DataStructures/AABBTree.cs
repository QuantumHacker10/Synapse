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
// FILE: AABBTree.cs
// PATH: Core/DataStructures/AABBTree.cs
// ============================================================


using System;
using System.Buffers;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace GDNN.Core.DataStructures;

/// <summary>
/// Axis-Aligned Bounding Box tree (BVH-like) with Surface Area Heuristic (SAH) construction.
/// Optimized for ray tracing and overlap queries on neural geometry assets.
/// Uses memory-efficient node layout with SoA-inspired design.
/// </summary>
/// <typeparam name="T">The type of items stored in the AABB tree.</typeparam>
public sealed class AABBTree<T> : IDisposable
    where T : notnull
{
    private const int LeafThreshold = 4;
    private const float NullCost = 1e30f;

    private Node[] _nodes;
    private int _rootIndex;
    private int _nodeCount;
    private int _capacity;
    private readonly Dictionary<T, int> _itemToNode;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Total number of items in the tree.
    /// </summary>
    public int Count { get; private set; }

    /// <summary>
    /// Total number of nodes (internal + leaf).
    /// </summary>
    public int NodeCount => _nodeCount;

    /// <summary>
    /// The root node index.
    /// </summary>
    public int RootIndex => _rootIndex;

    /// <summary>
    /// Memory usage in bytes.
    /// </summary>
    public long MemoryUsage => (long)_capacity * Unsafe.SizeOf<Node>();

    /// <summary>
    /// Initializes a new AABB tree.
    /// </summary>
    /// <param name="initialCapacity">Initial node capacity.</param>
    public AABBTree(int initialCapacity = 16)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(initialCapacity, 0);

        _capacity = initialCapacity;
        _nodes = new Node[initialCapacity];
        _nodeCount = 0;
        _rootIndex = -1;
        _itemToNode = new Dictionary<T, int>(initialCapacity);

        // Allocate sentinel node
        AllocateNode();
    }

    /// <summary>
    /// Inserts an item with its bounding box.
    /// </summary>
    /// <param name="item">The item to insert.</param>
    /// <param name="bounds">The bounding box of the item.</param>
    public void Insert(T item, AABB bounds)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(item);

        lock (_lock)
        {
            int leafIndex = InsertLeaf(item, bounds);
            _itemToNode[item] = leafIndex;
            Count++;
        }
    }

    /// <summary>
    /// Removes an item from the tree.
    /// </summary>
    public bool Remove(T item)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            if (!_itemToNode.TryGetValue(item, out int nodeIndex))
                return false;

            RemoveLeaf(nodeIndex);
            _itemToNode.Remove(item);
            Count--;
            return true;
        }
    }

    /// <summary>
    /// Updates the bounding box of an existing item.
    /// </summary>
    public bool Update(T item, AABB newBounds)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            if (!_itemToNode.TryGetValue(item, out int nodeIndex))
                return false;

            // Remove and reinsert for simplicity; could be optimized with refit
            RemoveLeaf(nodeIndex);
            int newLeafIndex = InsertLeaf(item, newBounds);
            _itemToNode[item] = newLeafIndex;
            return true;
        }
    }

    /// <summary>
    /// Refits all bounding boxes bottom-up after items have moved.
    /// </summary>
    public void Refit(Func<T, AABB> boundsFunc)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            if (_rootIndex < 0 || _nodes[_rootIndex].IsLeaf)
                return;

            RefitInternal(_rootIndex, boundsFunc);
        }
    }

    /// <summary>
    /// Rebuilds the tree from scratch using SAH.
    /// </summary>
    /// <param name="items">All items with their bounds.</param>
    public void Rebuild(ReadOnlySpan<(T Item, AABB Bounds)> items)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            // Clear existing
            _nodeCount = 0;
            _itemToNode.Clear();
            Count = 0;

            // Ensure capacity
            int required = items.Length * 2 + 1;
            if (_capacity < required)
            {
                _capacity = required;
                _nodes = new Node[_capacity];
            }

            // Allocate sentinel
            AllocateNode();

            if (items.Length == 0)
            {
                _rootIndex = -1;
                return;
            }

            if (items.Length == 1)
            {
                int leaf = AllocateNode();
                _nodes[leaf].Item = items[0].Item;
                _nodes[leaf].Bounds = items[0].Bounds;
                _nodes[leaf].IsLeaf = true;
                _nodes[leaf].Height = 0;
                _itemToNode[items[0].Item] = leaf;
                _rootIndex = leaf;
                Count = 1;
                return;
            }

            // Build using SAH
            var buildItems = new BuildItem[items.Length];
            for (int i = 0; i < items.Length; i++)
            {
                buildItems[i] = new BuildItem(items[i].Item, items[i].Bounds, i);
            }

            _rootIndex = BuildSubtree(buildItems, 0, items.Length);
            Count = items.Length;
        }
    }

    /// <summary>
    /// Queries for all items that overlap with a given AABB.
    /// </summary>
    public int QueryAABB(AABB queryBounds, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        if (_rootIndex < 0) return 0;
        return QueryAABBInternal(_rootIndex, queryBounds, results);
    }

    /// <summary>
    /// Queries for all items that overlap with a sphere.
    /// </summary>
    public int QuerySphere(Vector3 center, float radius, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        if (_rootIndex < 0) return 0;

        AABB sphereBounds = new(center, new Vector3(radius));
        return QueryAABBInternal(_rootIndex, sphereBounds, results);
    }

    /// <summary>
    /// Queries for all items that overlap with a frustum.
    /// </summary>
    public int QueryFrustum(Frustum frustum, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        if (_rootIndex < 0) return 0;
        return QueryFrustumInternal(_rootIndex, frustum, results);
    }

    /// <summary>
    /// Performs a ray intersection query. Returns all items the ray intersects.
    /// </summary>
    public int QueryRay(Ray ray, float maxDistance, List<RayHit<T>> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        if (_rootIndex < 0) return 0;
        return QueryRayInternal(_rootIndex, ray, maxDistance, results);
    }

    /// <summary>
    /// Finds the closest ray intersection.
    /// </summary>
    public bool RaycastClosest(Ray ray, float maxDistance, out T? hitItem, out float hitDistance, out Vector3 hitPoint)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        hitItem = default;
        hitDistance = float.MaxValue;
        hitPoint = Vector3.Zero;

        if (_rootIndex < 0) return false;

        bool found = false;
        RaycastInternal(_rootIndex, ray, maxDistance, ref found, ref hitItem, ref hitDistance, ref hitPoint);
        return found;
    }

    /// <summary>
    /// Tests if a ray intersects any item in the tree.
    /// </summary>
    public bool RaycastAny(Ray ray, float maxDistance)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_rootIndex < 0) return false;
        return RaycastAnyInternal(_rootIndex, ray, maxDistance);
    }

    /// <summary>
    /// Performs overlap detection between this tree and another.
    /// </summary>
    public int FindOverlaps(AABBTree<T> other, List<(T ItemA, T ItemB)> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(other);
        ArgumentNullException.ThrowIfNull(results);

        if (_rootIndex < 0 || other._rootIndex < 0) return 0;

        return FindOverlapsInternal(_rootIndex, other, other._rootIndex, results);
    }

    /// <summary>
    /// Balances the tree using rotations to reduce tree height.
    /// </summary>
    public void Balance()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            if (_rootIndex >= 0)
            {
                _rootIndex = BalanceNode(_rootIndex);
            }
        }
    }

    /// <summary>
    /// Gets the height of the tree.
    /// </summary>
    public int GetHeight()
    {
        if (_rootIndex < 0) return 0;
        return _nodes[_rootIndex].Height;
    }

    /// <summary>
    /// Validates the tree integrity (for debugging).
    /// </summary>
    public bool Validate()
    {
        if (_rootIndex < 0) return true;
        return ValidateNode(_rootIndex, out _);
    }

    /// <summary>
    /// Enumerates all items in the tree.
    /// </summary>
    public IEnumerable<T> GetAllItems()
    {
        for (int i = 0; i < _nodeCount; i++)
        {
            if (_nodes[i].IsLeaf && _nodes[i].Item != null)
            {
                yield return _nodes[i].Item!;
            }
        }
    }

    /// <summary>
    /// Gets the AABB of an item's leaf node.
    /// </summary>
    public bool TryGetBounds(T item, out AABB bounds)
    {
        bounds = default;
        if (_itemToNode.TryGetValue(item, out int nodeIndex))
        {
            bounds = _nodes[nodeIndex].Bounds;
            return true;
        }
        return false;
    }

    #region Private Implementation

    private struct Node
    {
        public AABB Bounds;
        public T? Item;
        public int Left;
        public int Right;
        public int Parent;
        public int Height;
        public bool IsLeaf;

        public readonly bool IsNull => Height < 0;
    }

    private struct BuildItem
    {
        public readonly T Item;
        public AABB Bounds;
        public readonly int OriginalIndex;

        public BuildItem(T item, AABB bounds, int originalIndex)
        {
            Item = item;
            Bounds = bounds;
            OriginalIndex = originalIndex;
        }
    }

    private int AllocateNode()
    {
        if (_nodeCount >= _capacity)
        {
            _capacity *= 2;
            Array.Resize(ref _nodes, _capacity);
        }

        int index = _nodeCount++;
        _nodes[index] = new Node
        {
            Left = -1,
            Right = -1,
            Parent = -1,
            Height = 0,
            IsLeaf = false
        };
        return index;
    }

    private int InsertLeaf(T item, AABB bounds)
    {
        if (_rootIndex < 0)
        {
            int rootLeaf = AllocateNode();
            _nodes[rootLeaf].Item = item;
            _nodes[rootLeaf].Bounds = bounds;
            _nodes[rootLeaf].IsLeaf = true;
            _nodes[rootLeaf].Height = 0;
            _rootIndex = rootLeaf;
            return rootLeaf;
        }

        // Find best sibling using SAH
        int bestSibling = FindBestSibling(bounds);

        // Create new parent
        int oldParent = _nodes[bestSibling].Parent;
        int newParent = AllocateNode();
        _nodes[newParent].Parent = oldParent;
        _nodes[newParent].Bounds = AABB.Merge(bounds, _nodes[bestSibling].Bounds);
        _nodes[newParent].Height = _nodes[bestSibling].Height + 1;

        // Create leaf
        int leaf = AllocateNode();
        _nodes[leaf].Item = item;
        _nodes[leaf].Bounds = bounds;
        _nodes[leaf].IsLeaf = true;
        _nodes[leaf].Height = 0;
        _nodes[leaf].Parent = newParent;

        // Update parent pointers
        if (oldParent >= 0)
        {
            if (_nodes[oldParent].Left == bestSibling)
                _nodes[oldParent].Left = newParent;
            else
                _nodes[oldParent].Right = newParent;
        }
        else
        {
            _rootIndex = newParent;
        }

        _nodes[newParent].Left = bestSibling;
        _nodes[newParent].Right = leaf;
        _nodes[bestSibling].Parent = newParent;

        // Refit ancestors
        RefitAncestors(newParent);

        return leaf;
    }

    private int FindBestSibling(AABB leafBounds)
    {
        int node = _rootIndex;

        while (!_nodes[node].IsLeaf)
        {
            int left = _nodes[node].Left;
            int right = _nodes[node].Right;

            float surfaceArea = _nodes[node].Bounds.SurfaceArea;

            AABB combinedLeft = AABB.Merge(_nodes[node].Bounds, leafBounds);
            AABB combinedRight = AABB.Merge(_nodes[node].Bounds, leafBounds);

            float costLeft = combinedLeft.SurfaceArea * GetAreaMultiplier(left);
            float costRight = combinedRight.SurfaceArea * GetAreaMultiplier(right);

            float costIncrease = MathF.Min(costLeft, costRight) - surfaceArea * GetAreaMultiplier(node);

            if (costIncrease < 0)
            {
                // Descend into cheaper child
                if (costLeft < costRight)
                    node = left;
                else
                    node = right;
            }
            else
            {
                break;
            }
        }

        return node;
    }

    private float GetAreaMultiplier(int nodeIndex)
    {
        if (nodeIndex < 0 || nodeIndex >= _nodeCount) return 1.0f;
        return _nodes[nodeIndex].Height + 1;
    }

    private void RemoveLeaf(int leafIndex)
    {
        if (leafIndex == _rootIndex)
        {
            _rootIndex = -1;
            return;
        }

        int parent = _nodes[leafIndex].Parent;
        int grandParent = _nodes[parent].Parent;
        int sibling;

        if (_nodes[parent].Left == leafIndex)
            sibling = _nodes[parent].Right;
        else
            sibling = _nodes[parent].Left;

        if (grandParent >= 0)
        {
            if (_nodes[grandParent].Left == parent)
                _nodes[grandParent].Left = sibling;
            else
                _nodes[grandParent].Right = sibling;

            _nodes[sibling].Parent = grandParent;
            RefitAncestors(grandParent);
        }
        else
        {
            _rootIndex = sibling;
            _nodes[sibling].Parent = -1;
        }

        // Mark parent as free (reuse sentinel pattern)
        _nodes[parent].Height = -1;
    }

    private void RefitAncestors(int nodeIndex)
    {
        int node = nodeIndex;
        while (node >= 0)
        {
            int left = _nodes[node].Left;
            int right = _nodes[node].Right;

            _nodes[node].Bounds = AABB.Merge(_nodes[left].Bounds, _nodes[right].Bounds);
            _nodes[node].Height = Math.Max(_nodes[left].Height, _nodes[right].Height) + 1;

            node = _nodes[node].Parent;
        }
    }

    private void RefitInternal(int nodeIndex, Func<T, AABB> boundsFunc)
    {
        if (_nodes[nodeIndex].IsLeaf)
        {
            if (_nodes[nodeIndex].Item != null)
                _nodes[nodeIndex].Bounds = boundsFunc(_nodes[nodeIndex].Item!);
            return;
        }

        int left = _nodes[nodeIndex].Left;
        int right = _nodes[nodeIndex].Right;

        RefitInternal(left, boundsFunc);
        RefitInternal(right, boundsFunc);

        _nodes[nodeIndex].Bounds = AABB.Merge(_nodes[left].Bounds, _nodes[right].Bounds);
        _nodes[nodeIndex].Height = Math.Max(_nodes[left].Height, _nodes[right].Height) + 1;
    }

    private int QueryAABBInternal(int nodeIndex, AABB queryBounds, List<T> results)
    {
        if (nodeIndex < 0) return 0;

        if (!AABB.Intersects(_nodes[nodeIndex].Bounds, queryBounds))
            return 0;

        int count = 0;

        if (_nodes[nodeIndex].IsLeaf)
        {
            if (_nodes[nodeIndex].Item != null)
            {
                results.Add(_nodes[nodeIndex].Item!);
                count++;
            }
        }
        else
        {
            count += QueryAABBInternal(_nodes[nodeIndex].Left, queryBounds, results);
            count += QueryAABBInternal(_nodes[nodeIndex].Right, queryBounds, results);
        }

        return count;
    }

    private int QueryFrustumInternal(int nodeIndex, Frustum frustum, List<T> results)
    {
        if (nodeIndex < 0) return 0;

        FrustumTest test = frustum.TestAABB(_nodes[nodeIndex].Bounds);
        if (test == FrustumTest.Outside)
            return 0;

        int count = 0;

        if (_nodes[nodeIndex].IsLeaf)
        {
            if (_nodes[nodeIndex].Item != null)
            {
                results.Add(_nodes[nodeIndex].Item!);
                count++;
            }
        }
        else if (test == FrustumTest.Inside)
        {
            CollectAllItems(nodeIndex, results);
        }
        else
        {
            count += QueryFrustumInternal(_nodes[nodeIndex].Left, frustum, results);
            count += QueryFrustumInternal(_nodes[nodeIndex].Right, frustum, results);
        }

        return count;
    }

    private int QueryRayInternal(int nodeIndex, Ray ray, float maxDistance, List<RayHit<T>> results)
    {
        if (nodeIndex < 0) return 0;

        if (!AABB.IntersectsRay(_nodes[nodeIndex].Bounds, ray, out float tmin, out float tmax))
            return 0;

        if (tmin > maxDistance)
            return 0;

        int count = 0;

        if (_nodes[nodeIndex].IsLeaf)
        {
            if (_nodes[nodeIndex].Item != null)
            {
                // The node-level slab test already proved the ray enters this
                // leaf's bounds within range; report the exact entry distance
                // (clamped to 0 when the ray starts inside the box).
                float entry = MathF.Max(tmin, 0.0f);
                results.Add(new RayHit<T>(_nodes[nodeIndex].Item!, entry, ray.GetPoint(entry)));
                count++;
            }
        }
        else
        {
            // Front-to-back traversal
            int left = _nodes[nodeIndex].Left;
            int right = _nodes[nodeIndex].Right;

            float leftDist = AABB.SqDistance(_nodes[left].Bounds, ray.Position);
            float rightDist = AABB.SqDistance(_nodes[right].Bounds, ray.Position);

            if (leftDist < rightDist)
            {
                count += QueryRayInternal(left, ray, maxDistance, results);
                count += QueryRayInternal(right, ray, maxDistance, results);
            }
            else
            {
                count += QueryRayInternal(right, ray, maxDistance, results);
                count += QueryRayInternal(left, ray, maxDistance, results);
            }
        }

        return count;
    }

    private void RaycastInternal(int nodeIndex, Ray ray, float maxDistance,
        ref bool found, ref T? hitItem, ref float hitDistance, ref Vector3 hitPoint)
    {
        if (nodeIndex < 0) return;

        if (!AABB.IntersectsRay(_nodes[nodeIndex].Bounds, ray, out float tmin, out float tmax))
            return;

        if (tmin > hitDistance)
            return;

        if (_nodes[nodeIndex].IsLeaf)
        {
            if (_nodes[nodeIndex].Item != null)
            {
                Vector3 toItem = _nodes[nodeIndex].Bounds.Center - ray.Position;
                float t = Vector3.Dot(toItem, ray.Direction);

                if (t >= 0 && t <= maxDistance && t < hitDistance)
                {
                    Vector3 closest = ray.Position + ray.Direction * t;
                    float distSq = Vector3.DistanceSquared(closest, _nodes[nodeIndex].Bounds.Center);

                    if (distSq <= _nodes[nodeIndex].Bounds.HalfExtents.LengthSquared())
                    {
                        found = true;
                        hitItem = _nodes[nodeIndex].Item!;
                        hitDistance = t;
                        hitPoint = closest;
                    }
                }
            }
        }
        else
        {
            RaycastInternal(_nodes[nodeIndex].Left, ray, maxDistance, ref found, ref hitItem, ref hitDistance, ref hitPoint);
            RaycastInternal(_nodes[nodeIndex].Right, ray, maxDistance, ref found, ref hitItem, ref hitDistance, ref hitPoint);
        }
    }

    private bool RaycastAnyInternal(int nodeIndex, Ray ray, float maxDistance)
    {
        if (nodeIndex < 0) return false;

        if (!AABB.IntersectsRay(_nodes[nodeIndex].Bounds, ray, out float tmin, out float tmax))
            return false;

        if (tmin > maxDistance)
            return false;

        if (_nodes[nodeIndex].IsLeaf)
        {
            return _nodes[nodeIndex].Item != null;
        }

        return RaycastAnyInternal(_nodes[nodeIndex].Left, ray, maxDistance) ||
               RaycastAnyInternal(_nodes[nodeIndex].Right, ray, maxDistance);
    }

    private int FindOverlapsInternal(int nodeA, AABBTree<T> other, int nodeB, List<(T ItemA, T ItemB)> results)
    {
        if (nodeA < 0 || nodeB < 0) return 0;

        if (!AABB.Intersects(_nodes[nodeA].Bounds, other._nodes[nodeB].Bounds))
            return 0;

        int count = 0;

        if (_nodes[nodeA].IsLeaf && other._nodes[nodeB].IsLeaf)
        {
            if (_nodes[nodeA].Item != null && other._nodes[nodeB].Item != null)
            {
                results.Add((_nodes[nodeA].Item!, other._nodes[nodeB].Item!));
                count++;
            }
        }
        else if (_nodes[nodeA].IsLeaf)
        {
            count += FindOverlapsInternal(nodeA, other, other._nodes[nodeB].Left, results);
            count += FindOverlapsInternal(nodeA, other, other._nodes[nodeB].Right, results);
        }
        else if (other._nodes[nodeB].IsLeaf)
        {
            count += FindOverlapsInternal(_nodes[nodeA].Left, other, nodeB, results);
            count += FindOverlapsInternal(_nodes[nodeA].Right, other, nodeB, results);
        }
        else
        {
            count += FindOverlapsInternal(_nodes[nodeA].Left, other, other._nodes[nodeB].Left, results);
            count += FindOverlapsInternal(_nodes[nodeA].Left, other, other._nodes[nodeB].Right, results);
            count += FindOverlapsInternal(_nodes[nodeA].Right, other, other._nodes[nodeB].Left, results);
            count += FindOverlapsInternal(_nodes[nodeA].Right, other, other._nodes[nodeB].Right, results);
        }

        return count;
    }

    private void CollectAllItems(int nodeIndex, List<T> results)
    {
        if (nodeIndex < 0) return;

        if (_nodes[nodeIndex].IsLeaf)
        {
            if (_nodes[nodeIndex].Item != null)
                results.Add(_nodes[nodeIndex].Item!);
        }
        else
        {
            CollectAllItems(_nodes[nodeIndex].Left, results);
            CollectAllItems(_nodes[nodeIndex].Right, results);
        }
    }

    #endregion

    #region SAH Build

    private int BuildSubtree(BuildItem[] items, int begin, int end)
    {
        int count = end - begin;

        if (count <= LeafThreshold)
        {
            // Create leaf node
            AABB totalBounds = items[begin].Bounds;
            for (int i = begin + 1; i < end; i++)
            {
                totalBounds = AABB.Merge(totalBounds, items[i].Bounds);
            }

            int leaf = AllocateNode();
            _nodes[leaf].Bounds = totalBounds;
            _nodes[leaf].IsLeaf = true;
            _nodes[leaf].Height = 0;

            if (count == 1)
            {
                _nodes[leaf].Item = items[begin].Item;
                _itemToNode[items[begin].Item] = leaf;
            }
            else
            {
                // Multi-leaf: store items in a list node or first item
                _nodes[leaf].Item = items[begin].Item;
                _itemToNode[items[begin].Item] = leaf;

                // Create additional leaf nodes for remaining items
                int prevLeaf = leaf;
                for (int i = begin + 1; i < end; i++)
                {
                    int extraLeaf = AllocateNode();
                    _nodes[extraLeaf].Item = items[i].Item;
                    _nodes[extraLeaf].Bounds = items[i].Bounds;
                    _nodes[extraLeaf].IsLeaf = true;
                    _nodes[extraLeaf].Height = 0;
                    _itemToNode[items[i].Item] = extraLeaf;

                    // Chain leaves through parent
                    int newParent = AllocateNode();
                    _nodes[newParent].Left = prevLeaf;
                    _nodes[newParent].Right = extraLeaf;
                    _nodes[newParent].Parent = -1;
                    _nodes[newParent].Bounds = AABB.Merge(_nodes[prevLeaf].Bounds, _nodes[extraLeaf].Bounds);
                    _nodes[newParent].Height = 1;

                    _nodes[prevLeaf].Parent = newParent;
                    _nodes[extraLeaf].Parent = newParent;

                    prevLeaf = newParent;
                }

                return prevLeaf;
            }

            return leaf;
        }

        // Choose split axis using SAH
        AABB centroidBounds = items[begin].Bounds;
        for (int i = begin + 1; i < end; i++)
        {
            centroidBounds.ExpandToInclude(items[i].Bounds.Center);
        }

        Vector3 extent = centroidBounds.Size;
        int axis;

        if (extent.X > extent.Y && extent.X > extent.Z)
            axis = 0;
        else if (extent.Y > extent.Z)
            axis = 1;
        else
            axis = 2;

        // Sort along chosen axis
        Array.Sort(items, begin, count, new CentroidComparer(axis));

        int mid = begin + count / 2;

        int left = BuildSubtree(items, begin, mid);
        int right = BuildSubtree(items, mid, end);

        int internalNode = AllocateNode();
        _nodes[internalNode].Left = left;
        _nodes[internalNode].Right = right;
        _nodes[internalNode].Parent = -1;
        _nodes[internalNode].Bounds = AABB.Merge(_nodes[left].Bounds, _nodes[right].Bounds);
        _nodes[internalNode].Height = Math.Max(_nodes[left].Height, _nodes[right].Height) + 1;

        _nodes[left].Parent = internalNode;
        _nodes[right].Parent = internalNode;

        return internalNode;
    }

    private sealed class CentroidComparer : IComparer<BuildItem>
    {
        private readonly int _axis;

        public CentroidComparer(int axis) => _axis = axis;

        public int Compare(BuildItem a, BuildItem b)
        {
            return _axis switch
            {
                0 => a.Bounds.Center.X.CompareTo(b.Bounds.Center.X),
                1 => a.Bounds.Center.Y.CompareTo(b.Bounds.Center.Y),
                2 => a.Bounds.Center.Z.CompareTo(b.Bounds.Center.Z),
                _ => 0
            };
        }
    }

    #endregion

    #region Balance

    private int BalanceNode(int nodeIndex)
    {
        if (nodeIndex < 0 || _nodes[nodeIndex].IsLeaf)
            return nodeIndex;

        int A = nodeIndex;
        int B = _nodes[A].Left;
        int C = _nodes[A].Right;

        int balanceFactor = _nodes[C].Height - _nodes[B].Height;

        // Rotate left
        if (balanceFactor > 1)
        {
            int F = _nodes[C].Left;
            int G = _nodes[C].Right;

            _nodes[C].Left = A;
            _nodes[C].Parent = _nodes[A].Parent;
            _nodes[A].Parent = C;

            if (_nodes[C].Parent >= 0)
            {
                if (_nodes[_nodes[C].Parent].Left == A)
                    _nodes[_nodes[C].Parent].Left = C;
                else
                    _nodes[_nodes[C].Parent].Right = C;
            }
            else
            {
                _rootIndex = C;
            }

            if (_nodes[F].Height > _nodes[G].Height)
            {
                _nodes[C].Right = F;
                _nodes[A].Right = G;
                _nodes[G].Parent = A;
                _nodes[A].Bounds = AABB.Merge(_nodes[B].Bounds, _nodes[G].Bounds);
                _nodes[C].Bounds = AABB.Merge(_nodes[A].Bounds, _nodes[F].Bounds);
                _nodes[A].Height = Math.Max(_nodes[B].Height, _nodes[G].Height) + 1;
                _nodes[C].Height = Math.Max(_nodes[A].Height, _nodes[F].Height) + 1;
            }
            else
            {
                _nodes[C].Right = G;
                _nodes[A].Right = F;
                _nodes[F].Parent = A;
                _nodes[A].Bounds = AABB.Merge(_nodes[B].Bounds, _nodes[F].Bounds);
                _nodes[C].Bounds = AABB.Merge(_nodes[A].Bounds, _nodes[G].Bounds);
                _nodes[A].Height = Math.Max(_nodes[B].Height, _nodes[F].Height) + 1;
                _nodes[C].Height = Math.Max(_nodes[A].Height, _nodes[G].Height) + 1;
            }

            return C;
        }

        // Rotate right
        if (balanceFactor < -1)
        {
            int D = _nodes[B].Left;
            int E = _nodes[B].Right;

            _nodes[B].Left = D;
            _nodes[B].Right = A;
            _nodes[B].Parent = _nodes[A].Parent;
            _nodes[A].Parent = B;

            if (_nodes[B].Parent >= 0)
            {
                if (_nodes[_nodes[B].Parent].Left == A)
                    _nodes[_nodes[B].Parent].Left = B;
                else
                    _nodes[_nodes[B].Parent].Right = B;
            }
            else
            {
                _rootIndex = B;
            }

            if (_nodes[D].Height > _nodes[E].Height)
            {
                _nodes[B].Left = D;
                _nodes[A].Left = E;
                _nodes[E].Parent = A;
                _nodes[A].Bounds = AABB.Merge(_nodes[C].Bounds, _nodes[E].Bounds);
                _nodes[B].Bounds = AABB.Merge(_nodes[A].Bounds, _nodes[D].Bounds);
                _nodes[A].Height = Math.Max(_nodes[C].Height, _nodes[E].Height) + 1;
                _nodes[B].Height = Math.Max(_nodes[A].Height, _nodes[D].Height) + 1;
            }
            else
            {
                _nodes[B].Left = E;
                _nodes[A].Left = D;
                _nodes[D].Parent = A;
                _nodes[A].Bounds = AABB.Merge(_nodes[C].Bounds, _nodes[D].Bounds);
                _nodes[B].Bounds = AABB.Merge(_nodes[A].Bounds, _nodes[E].Bounds);
                _nodes[A].Height = Math.Max(_nodes[C].Height, _nodes[D].Height) + 1;
                _nodes[B].Height = Math.Max(_nodes[A].Height, _nodes[E].Height) + 1;
            }

            return B;
        }

        return nodeIndex;
    }

    #endregion

    #region Validation

    private bool ValidateNode(int nodeIndex, out int height)
    {
        height = 0;

        if (nodeIndex < 0 || nodeIndex >= _nodeCount)
            return false;

        if (_nodes[nodeIndex].IsLeaf)
        {
            height = 0;
            return true;
        }

        int left = _nodes[nodeIndex].Left;
        int right = _nodes[nodeIndex].Right;

        if (left < 0 || right < 0)
            return false;

        if (!ValidateNode(left, out int leftHeight))
            return false;

        if (!ValidateNode(right, out int rightHeight))
            return false;

        height = Math.Max(leftHeight, rightHeight) + 1;

        AABB computed = AABB.Merge(_nodes[left].Bounds, _nodes[right].Bounds);
        return computed == _nodes[nodeIndex].Bounds;
    }

    #endregion

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _nodes = Array.Empty<Node>();
            _itemToNode.Clear();
            GC.SuppressFinalize(this);
        }
    }
}

/// <summary>
/// Represents a ray hit result with distance and position.
/// </summary>
public readonly struct RayHit<TItem>
{
    /// <summary>The hit item.</summary>
    public readonly TItem Item;

    /// <summary>Distance along the ray to the hit.</summary>
    public readonly float Distance;

    /// <summary>The world-space position of the hit.</summary>
    public readonly Vector3 Position;

    public RayHit(TItem item, float distance, Vector3 position)
    {
        Item = item;
        Distance = distance;
        Position = position;
    }
}
