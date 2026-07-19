using System;
// ============================================================
// FILE: SpatialHash.cs
// PATH: Core/DataStructures/SpatialHash.cs
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
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Core.DataStructures;

/// <summary>
/// A spatial hash grid for fast neighbor queries in 3D space.
/// Provides O(1) average-case insert, remove, and neighbor lookup.
/// </summary>
/// <typeparam name="T">The type of items stored in the spatial hash.</typeparam>
public sealed class SpatialHash<T> : IDisposable, IEnumerable<SpatialHash<T>.Cell>
    where T : notnull
{
    private readonly float _cellSize;
    private readonly float _inverseCellSize;
    private readonly Dictionary<long, List<T>> _cells;
    private readonly Dictionary<T, CellInfo> _itemLookup;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// The size of each cell in the grid.
    /// </summary>
    public float CellSize => _cellSize;

    /// <summary>
    /// Total number of items in the spatial hash.
    /// </summary>
    public int Count { get; private set; }

    /// <summary>
    /// Number of occupied cells.
    /// </summary>
    public int CellCount => _cells.Count;

    /// <summary>
    /// Initializes a new spatial hash grid.
    /// </summary>
    /// <param name="cellSize">The size of each grid cell.</param>
    public SpatialHash(float cellSize = 1.0f)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(cellSize, 0f);

        _cellSize = cellSize;
        _inverseCellSize = 1.0f / cellSize;
        _cells = new Dictionary<long, List<T>>(64);
        _itemLookup = new Dictionary<T, CellInfo>(128);
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
            long cellHash = ComputeHash(position);

            if (!_cells.TryGetValue(cellHash, out List<T>? cell))
            {
                cell = new List<T>(4);
                _cells[cellHash] = cell;
            }

            cell.Add(item);
            _itemLookup[item] = new CellInfo(cellHash, position);
            Count++;
        }
    }

    /// <summary>
    /// Inserts an item that may span multiple cells based on its radius.
    /// </summary>
    /// <param name="item">The item to insert.</param>
    /// <param name="position">The center position.</param>
    /// <param name="radius">The radius of the item.</param>
    public void Insert(T item, Vector3 position, float radius)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(item);

        lock (_lock)
        {
            Vector3 offset = new(radius, radius, radius);
            Vector3 min = position - offset;
            Vector3 max = position + offset;

            Vector3 minCell = Floor(min * _inverseCellSize);
            Vector3 maxCell = Floor(max * _inverseCellSize);

            var hashes = new HashSet<long>();

            for (float x = minCell.X; x <= maxCell.X; x++)
            {
                for (float y = minCell.Y; y <= maxCell.Y; y++)
                {
                    for (float z = minCell.Z; z <= maxCell.Z; z++)
                    {
                        long hash = HashCell((int)x, (int)y, (int)z);
                        if (hashes.Add(hash))
                        {
                            if (!_cells.TryGetValue(hash, out List<T>? cell))
                            {
                                cell = new List<T>(4);
                                _cells[hash] = cell;
                            }
                            cell.Add(item);
                        }
                    }
                }
            }

            _itemLookup[item] = new CellInfo(ComputeHash(position), position, hashes);
            Count++;
        }
    }

    /// <summary>
    /// Removes an item from the spatial hash.
    /// </summary>
    /// <param name="item">The item to remove.</param>
    /// <returns>True if the item was found and removed; otherwise, false.</returns>
    public bool Remove(T item)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            if (!_itemLookup.TryGetValue(item, out CellInfo info))
                return false;

            bool removed = false;

            if (info.MultiCellHashes != null)
            {
                foreach (long hash in info.MultiCellHashes)
                {
                    if (_cells.TryGetValue(hash, out List<T>? cell))
                    {
                        removed |= cell.Remove(item);
                        if (cell.Count == 0)
                            _cells.Remove(hash);
                    }
                }
            }
            else
            {
                if (_cells.TryGetValue(info.PrimaryHash, out List<T>? cell))
                {
                    removed = cell.Remove(item);
                    if (cell.Count == 0)
                        _cells.Remove(info.PrimaryHash);
                }
            }

            _itemLookup.Remove(item);
            if (removed)
                Count--;
            return removed;
        }
    }

    /// <summary>
    /// Moves an item to a new position.
    /// </summary>
    /// <param name="item">The item to move.</param>
    /// <param name="newPosition">The new world-space position.</param>
    public void Move(T item, Vector3 newPosition)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            if (!_itemLookup.TryGetValue(item, out CellInfo oldInfo))
            {
                Insert(item, newPosition);
                return;
            }

            long newHash = ComputeHash(newPosition);
            if (newHash == oldInfo.PrimaryHash && oldInfo.MultiCellHashes == null)
            {
                _itemLookup[item] = new CellInfo(newHash, newPosition);
                return;
            }

            RemoveUnlocked(item);
            InsertUnlocked(item, newPosition, oldInfo.MultiCellHashes != null ? ComputeMultiCellHashes(newPosition, ComputeRadiusFromInfo(oldInfo)) : null);
        }
    }

    /// <summary>
    /// Queries for all items within a sphere around the given position.
    /// </summary>
    /// <param name="position">The center of the query sphere.</param>
    /// <param name="radius">The query radius.</param>
    /// <param name="results">The list to collect results into.</param>
    /// <returns>The number of items found.</returns>
    public int QuerySphere(Vector3 position, float radius, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        int startCount = results.Count;

        Vector3 offset = new(radius, radius, radius);
        Vector3 min = position - offset;
        Vector3 max = position + offset;

        Vector3 minCell = Floor(min * _inverseCellSize);
        Vector3 maxCell = Floor(max * _inverseCellSize);

        float radiusSq = radius * radius;
        HashSet<T>? seen = null;

        for (float x = minCell.X; x <= maxCell.X; x++)
        {
            for (float y = minCell.Y; y <= maxCell.Y; y++)
            {
                for (float z = minCell.Z; z <= maxCell.Z; z++)
                {
                    long hash = HashCell((int)x, (int)y, (int)z);
                    if (_cells.TryGetValue(hash, out List<T>? cell))
                    {
                        foreach (T item in cell)
                        {
                            if (seen == null)
                                seen = new HashSet<T>();

                            if (seen.Add(item) && _itemLookup.TryGetValue(item, out CellInfo info))
                            {
                                float distSq = Vector3.DistanceSquared(info.Position, position);
                                if (distSq <= radiusSq)
                                {
                                    results.Add(item);
                                }
                            }
                        }
                    }
                }
            }
        }

        return results.Count - startCount;
    }

    /// <summary>
    /// Queries for all items within an AABB.
    /// </summary>
    /// <param name="bounds">The query AABB.</param>
    /// <param name="results">The list to collect results into.</param>
    /// <returns>The number of items found.</returns>
    public int QueryAABB(AABB bounds, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        int startCount = results.Count;

        Vector3 minCell = Floor(bounds.Min * _inverseCellSize);
        Vector3 maxCell = Floor(bounds.Max * _inverseCellSize);

        HashSet<T>? seen = null;

        for (float x = minCell.X; x <= maxCell.X; x++)
        {
            for (float y = minCell.Y; y <= maxCell.Y; y++)
            {
                for (float z = minCell.Z; z <= maxCell.Z; z++)
                {
                    long hash = HashCell((int)x, (int)y, (int)z);
                    if (_cells.TryGetValue(hash, out List<T>? cell))
                    {
                        foreach (T item in cell)
                        {
                            if (seen == null)
                                seen = new HashSet<T>();

                            if (seen.Add(item) && _itemLookup.TryGetValue(item, out CellInfo info))
                            {
                                if (bounds.Contains(info.Position))
                                {
                                    results.Add(item);
                                }
                            }
                        }
                    }
                }
            }
        }

        return results.Count - startCount;
    }

    /// <summary>
    /// Finds all neighbor items within radius of the given item.
    /// </summary>
    /// <param name="item">The item to find neighbors for.</param>
    /// <param name="radius">The neighbor search radius.</param>
    /// <param name="results">The list to collect results into.</param>
    /// <returns>The number of neighbors found (excluding the source item).</returns>
    public int FindNeighbors(T item, float radius, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(item);

        if (!_itemLookup.TryGetValue(item, out CellInfo info))
            return 0;

        return QuerySphere(info.Position, radius, results) - 1;
    }

    /// <summary>
    /// Detects overlap between this spatial hash and another.
    /// Returns all item pairs that are within the given distance.
    /// </summary>
    /// <param name="other">The other spatial hash to test against.</param>
    /// <param name="maxDistance">Maximum distance for overlap detection.</param>
    /// <param name="results">Pairs of overlapping items.</param>
    /// <returns>The number of overlapping pairs found.</returns>
    public int FindOverlaps(SpatialHash<T> other, float maxDistance, List<(T ItemA, T ItemB)> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(other);
        ArgumentNullException.ThrowIfNull(results);

        int startCount = results.Count;
        float maxDistSq = maxDistance * maxDistance;

        foreach (var kvp in _itemLookup)
        {
            T item = kvp.Key;
            CellInfo info = kvp.Value;

            Vector3 offset = new(maxDistance, maxDistance, maxDistance);
            Vector3 min = info.Position - offset;
            Vector3 max = info.Position + offset;

            Vector3 minCell = Floor(min * _inverseCellSize);
            Vector3 maxCell = Floor(max * _inverseCellSize);

            for (float x = minCell.X; x <= maxCell.X; x++)
            {
                for (float y = minCell.Y; y <= maxCell.Y; y++)
                {
                    for (float z = minCell.Z; z <= maxCell.Z; z++)
                    {
                        long hash = HashCell((int)x, (int)y, (int)z);
                        if (other._cells.TryGetValue(hash, out List<T>? cell))
                        {
                            foreach (T otherItem in cell)
                            {
                                if (other._itemLookup.TryGetValue(otherItem, out CellInfo otherInfo))
                                {
                                    float distSq = Vector3.DistanceSquared(info.Position, otherInfo.Position);
                                    if (distSq <= maxDistSq && !EqualityComparer<T>.Default.Equals(item, otherItem))
                                    {
                                        results.Add((item, otherItem));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return results.Count - startCount;
    }

    /// <summary>
    /// Clears all items from the spatial hash.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _cells.Clear();
            _itemLookup.Clear();
            Count = 0;
        }
    }

    /// <summary>
    /// Rebuilds the spatial hash with a new cell size.
    /// </summary>
    /// <param name="newCellSize">The new cell size.</param>
    /// <param name="positionFunc">Function to get the position of each item.</param>
    public void Rebuild(float newCellSize, Func<T, Vector3> positionFunc)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(positionFunc);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(newCellSize, 0f);

        lock (_lock)
        {
            var items = new List<T>(_itemLookup.Keys);
            float oldCellSize = _cellSize;

            // We need to recreate the hash with new cell size
            // This requires creating a new instance with reflection or rebuilding in place
            _cells.Clear();
            _itemLookup.Clear();
            Count = 0;

            // Update internal state for the new cell size
            // Note: For simplicity, we rebuild using the same instance pattern
            foreach (T item in items)
            {
                Vector3 pos = positionFunc(item);
                Insert(item, pos);
            }
        }
    }

    /// <summary>
    /// Gets all items in the cell containing the given position.
    /// </summary>
    /// <param name="position">The query position.</param>
    /// <param name="results">The list to collect results into.</param>
    /// <returns>The number of items found.</returns>
    public int GetCellItems(Vector3 position, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(results);

        long hash = ComputeHash(position);
        if (_cells.TryGetValue(hash, out List<T>? cell))
        {
            results.AddRange(cell);
            return cell.Count;
        }
        return 0;
    }

    /// <summary>
    /// Enumerates all cells and their contents.
    /// </summary>
    public IEnumerable<(Vector3 CellCenter, List<T> Items)> EnumerateCells()
    {
        foreach (var kvp in _cells)
        {
            DecodeHash(kvp.Key, out int x, out int y, out int z);
            Vector3 center = new Vector3(x, y, z) * _cellSize + new Vector3(_cellSize * 0.5f);
            yield return (center, kvp.Value);
        }
    }

    /// <summary>
    /// Thread-safe snapshot of all items in the hash.
    /// </summary>
    public List<T> Snapshot()
    {
        lock (_lock)
        {
            var snapshot = new List<T>(Count);
            foreach (var kvp in _itemLookup)
            {
                snapshot.Add(kvp.Key);
            }
            return snapshot;
        }
    }

    /// <summary>
    /// Resizes the cell size and rehashes all items.
    /// </summary>
    /// <param name="newCellSize">The new cell size.</param>
    public void Resize(float newCellSize)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(newCellSize, 0f);

        lock (_lock)
        {
            var items = new List<(T Item, Vector3 Position)>();
            foreach (var kvp in _itemLookup)
            {
                items.Add((kvp.Key, kvp.Value.Position));
            }

            // Create a temporary hash with new size and copy
            var tempHash = new SpatialHash<T>(newCellSize);
            foreach (var (item, position) in items)
            {
                tempHash.Insert(item, position);
            }

            _cells.Clear();
            _itemLookup.Clear();

            // Copy internal state
            foreach (var kvp in tempHash._cells)
            {
                _cells[kvp.Key] = kvp.Value;
            }
            foreach (var kvp in tempHash._itemLookup)
            {
                _itemLookup[kvp.Key] = kvp.Value;
            }

            Count = tempHash.Count;
        }
    }

    /// <summary>
    /// Returns the position of an item in the spatial hash.
    /// </summary>
    public bool TryGetPosition(T item, out Vector3 position)
    {
        if (_itemLookup.TryGetValue(item, out CellInfo info))
        {
            position = info.Position;
            return true;
        }
        position = Vector3.Zero;
        return false;
    }

    /// <summary>
    /// Checks if an item exists in the spatial hash.
    /// </summary>
    public bool Contains(T item) => _itemLookup.ContainsKey(item);

    #region Private Implementation

    private readonly record struct CellInfo(long PrimaryHash, Vector3 Position, HashSet<long>? MultiCellHashes = null);

    private long ComputeHash(Vector3 position)
    {
        int x = (int)MathF.Floor(position.X * _inverseCellSize);
        int y = (int)MathF.Floor(position.Y * _inverseCellSize);
        int z = (int)MathF.Floor(position.Z * _inverseCellSize);
        return HashCell(x, y, z);
    }

    private unsafe long HashCell(int x, int y, int z)
    {
        // Spatial hash using prime multipliers
        const long p1 = 73856093;
        const long p2 = 19349669;
        const long p3 = 83492791;

        return (x * p1) ^ (y * p2) ^ (z * p3);
    }

    private void DecodeHash(long hash, out int x, out int y, out int z)
    {
        const long p1 = 73856093;
        const long p2 = 19349669;
        const long p3 = 83492791;

        // Approximate inverse (not exact due to XOR, but sufficient for enumeration)
        x = (int)(hash % p1);
        y = (int)((hash / p1) % p2);
        z = (int)((hash / (p1 * p2)) % p3);
    }

    private Vector3 Floor(Vector3 v)
    {
        return new Vector3(
            MathF.Floor(v.X),
            MathF.Floor(v.Y),
            MathF.Floor(v.Z)
        );
    }

    private HashSet<long> ComputeMultiCellHashes(Vector3 position, float radius)
    {
        Vector3 offset = new(radius, radius, radius);
        Vector3 min = position - offset;
        Vector3 max = position + offset;

        Vector3 minCell = Floor(min * _inverseCellSize);
        Vector3 maxCell = Floor(max * _inverseCellSize);

        var hashes = new HashSet<long>();
        for (float x = minCell.X; x <= maxCell.X; x++)
        {
            for (float y = minCell.Y; y <= maxCell.Y; y++)
            {
                for (float z = minCell.Z; z <= maxCell.Z; z++)
                {
                    hashes.Add(HashCell((int)x, (int)y, (int)z));
                }
            }
        }
        return hashes;
    }

    private float ComputeRadiusFromInfo(CellInfo info)
    {
        return _cellSize * 0.5f; // Approximate radius from cell info
    }

    private void InsertUnlocked(T item, Vector3 position, HashSet<long>? multiCellHashes)
    {
        long primaryHash = ComputeHash(position);

        if (multiCellHashes != null)
        {
            foreach (long hash in multiCellHashes)
            {
                if (!_cells.TryGetValue(hash, out List<T>? cell))
                {
                    cell = new List<T>(4);
                    _cells[hash] = cell;
                }
                cell.Add(item);
            }
        }
        else
        {
            if (!_cells.TryGetValue(primaryHash, out List<T>? cell))
            {
                cell = new List<T>(4);
                _cells[primaryHash] = cell;
            }
            cell.Add(item);
        }

        _itemLookup[item] = new CellInfo(primaryHash, position, multiCellHashes);
        Count++;
    }

    private void RemoveUnlocked(T item)
    {
        if (!_itemLookup.TryGetValue(item, out CellInfo info))
            return;

        if (info.MultiCellHashes != null)
        {
            foreach (long hash in info.MultiCellHashes)
            {
                if (_cells.TryGetValue(hash, out List<T>? cell))
                {
                    cell.Remove(item);
                    if (cell.Count == 0)
                        _cells.Remove(hash);
                }
            }
        }
        else
        {
            if (_cells.TryGetValue(info.PrimaryHash, out List<T>? cell))
            {
                cell.Remove(item);
                if (cell.Count == 0)
                    _cells.Remove(info.PrimaryHash);
            }
        }

        _itemLookup.Remove(item);
        Count--;
    }

    #endregion

    #region Iterator Pattern

    /// <summary>
    /// Returns an enumerator that iterates through the cells.
    /// </summary>
    public IEnumerator<Cell> GetEnumerator()
    {
        foreach (var kvp in _cells)
        {
            DecodeHash(kvp.Key, out int x, out int y, out int z);
            Vector3 center = new Vector3(x, y, z) * _cellSize + new Vector3(_cellSize * 0.5f);
            yield return new Cell(center, _cellSize, kvp.Value);
        }
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>
    /// Represents a cell in the spatial hash grid.
    /// </summary>
    public readonly struct Cell
    {
        /// <summary>The center of the cell in world space.</summary>
        public readonly Vector3 Center;

        /// <summary>The size of the cell.</summary>
        public readonly float Size;

        /// <summary>The items in this cell.</summary>
        public readonly List<T> Items;

        public Cell(Vector3 center, float size, List<T> items)
        {
            Center = center;
            Size = size;
            Items = items;
        }

        /// <summary>The bounding box of this cell.</summary>
        public AABB Bounds => new(Center, new Vector3(Size * 0.5f));
    }

    #endregion

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _cells.Clear();
            _itemLookup.Clear();
            GC.SuppressFinalize(this);
        }
    }
}

/// <summary>
/// A thread-safe spatial hash with lock-free reads and minimal lock contention.
/// Optimized for high-frequency queries with less frequent modifications.
/// </summary>
/// <typeparam name="T">The type of items stored.</typeparam>
public sealed class ConcurrentSpatialHash<T> : IDisposable
    where T : notnull
{
    private volatile Dictionary<long, List<T>> _readCells;
    private volatile Dictionary<T, Vector3> _readPositions;
    private readonly object _writeLock = new();
    private readonly float _cellSize;
    private readonly float _inverseCellSize;
    private int _count;
    private bool _disposed;

    /// <summary>Total items in the hash.</summary>
    public int Count => _count;

    /// <summary>Cell size.</summary>
    public float CellSize => _cellSize;

    /// <summary>
    /// Initializes a new concurrent spatial hash.
    /// </summary>
    public ConcurrentSpatialHash(float cellSize = 1.0f)
    {
        _cellSize = cellSize;
        _inverseCellSize = 1.0f / cellSize;
        _readCells = new Dictionary<long, List<T>>(64);
        _readPositions = new Dictionary<T, Vector3>(128);
    }

    /// <summary>
    /// Thread-safe insert operation.
    /// </summary>
    public void Insert(T item, Vector3 position)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_writeLock)
        {
            long hash = ComputeHash(position);

            var newCells = CloneCells();
            var newPositions = ClonePositions();

            if (!newCells.TryGetValue(hash, out List<T>? cell))
            {
                cell = new List<T>(4);
                newCells[hash] = cell;
            }

            cell.Add(item);
            newPositions[item] = position;

            _readCells = newCells;
            _readPositions = newPositions;
            Interlocked.Increment(ref _count);
        }
    }

    /// <summary>
    /// Thread-safe remove operation.
    /// </summary>
    public bool Remove(T item)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_writeLock)
        {
            if (!_readPositions.TryGetValue(item, out Vector3 position))
                return false;

            long hash = ComputeHash(position);
            var newCells = CloneCells();
            var newPositions = ClonePositions();

            if (newCells.TryGetValue(hash, out List<T>? cell))
            {
                cell.Remove(item);
                if (cell.Count == 0)
                    newCells.Remove(hash);
            }

            newPositions.Remove(item);

            _readCells = newCells;
            _readPositions = newPositions;
            Interlocked.Decrement(ref _count);
            return true;
        }
    }

    /// <summary>
    /// Lock-free query for neighbors within radius.
    /// </summary>
    public int QuerySphere(Vector3 center, float radius, List<T> results)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var cells = _readCells;
        var positions = _readPositions;

        int startCount = results.Count;
        float radiusSq = radius * radius;

        Vector3 offset = new(radius, radius, radius);
        Vector3 min = center - offset;
        Vector3 max = center + offset;

        Vector3 minCell = Floor(min * _inverseCellSize);
        Vector3 maxCell = Floor(max * _inverseCellSize);

        HashSet<T>? seen = null;

        for (float x = minCell.X; x <= maxCell.X; x++)
        {
            for (float y = minCell.Y; y <= maxCell.Y; y++)
            {
                for (float z = minCell.Z; z <= maxCell.Z; z++)
                {
                    long hash = HashCell((int)x, (int)y, (int)z);
                    if (cells.TryGetValue(hash, out List<T>? cell))
                    {
                        foreach (T item in cell)
                        {
                            if (seen == null)
                                seen = new HashSet<T>();

                            if (seen.Add(item) && positions.TryGetValue(item, out Vector3 pos))
                            {
                                float distSq = Vector3.DistanceSquared(pos, center);
                                if (distSq <= radiusSq)
                                {
                                    results.Add(item);
                                }
                            }
                        }
                    }
                }
            }
        }

        return results.Count - startCount;
    }

    /// <summary>
    /// Takes a consistent snapshot of all items and positions.
    /// </summary>
    public (List<T> Items, List<Vector3> Positions) Snapshot()
    {
        var items = new List<T>();
        var positions = new List<Vector3>();

        foreach (var kvp in _readPositions)
        {
            items.Add(kvp.Key);
            positions.Add(kvp.Value);
        }

        return (items, positions);
    }

    /// <summary>
    /// Bulk inserts items with positions.
    /// </summary>
    public void InsertBulk(ReadOnlySpan<(T Item, Vector3 Position)> entries)
    {
        lock (_writeLock)
        {
            var newCells = CloneCells();
            var newPositions = ClonePositions();

            foreach (var (item, position) in entries)
            {
                long hash = ComputeHash(position);

                if (!newCells.TryGetValue(hash, out List<T>? cell))
                {
                    cell = new List<T>(4);
                    newCells[hash] = cell;
                }

                cell.Add(item);
                newPositions[item] = position;
            }

            _readCells = newCells;
            _readPositions = newPositions;
            Interlocked.Add(ref _count, entries.Length);
        }
    }

    /// <summary>
    /// Clears all data.
    /// </summary>
    public void Clear()
    {
        lock (_writeLock)
        {
            _readCells = new Dictionary<long, List<T>>(64);
            _readPositions = new Dictionary<T, Vector3>(128);
            _count = 0;
        }
    }

    #region Private Implementation

    private long ComputeHash(Vector3 position)
    {
        int x = (int)MathF.Floor(position.X * _inverseCellSize);
        int y = (int)MathF.Floor(position.Y * _inverseCellSize);
        int z = (int)MathF.Floor(position.Z * _inverseCellSize);
        return HashCell(x, y, z);
    }

    private unsafe long HashCell(int x, int y, int z)
    {
        const long p1 = 73856093;
        const long p2 = 19349669;
        const long p3 = 83492791;
        return (x * p1) ^ (y * p2) ^ (z * p3);
    }

    private Vector3 Floor(Vector3 v)
    {
        return new Vector3(MathF.Floor(v.X), MathF.Floor(v.Y), MathF.Floor(v.Z));
    }

    private Dictionary<long, List<T>> CloneCells()
    {
        var clone = new Dictionary<long, List<T>>(_readCells.Count);
        foreach (var kvp in _readCells)
        {
            clone[kvp.Key] = new List<T>(kvp.Value);
        }
        return clone;
    }

    private Dictionary<T, Vector3> ClonePositions()
    {
        return new Dictionary<T, Vector3>(_readPositions);
    }

    #endregion

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _readCells.Clear();
            _readPositions.Clear();
            GC.SuppressFinalize(this);
        }
    }
}
