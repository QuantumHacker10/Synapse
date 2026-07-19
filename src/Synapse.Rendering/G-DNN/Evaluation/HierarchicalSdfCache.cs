using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Evaluation;

/// <summary>
/// Configuration for hierarchical SDF caching.
/// </summary>
public sealed class HierarchicalCacheConfig
{
    /// <summary>Number of hierarchy levels (mip levels).</summary>
    public int HierarchyLevels { get; set; } = 6;

    /// <summary>Base resolution for the finest level.</summary>
    public int BaseResolution { get; set; } = 64;

    /// <summary>Cache size in entries per level.</summary>
    public int CacheSizePerLevel { get; set; } = 1024;

    /// <summary>Enable temporal caching (reuse across frames).</summary>
    public bool EnableTemporalCaching { get; set; } = true;

    /// <summary>Maximum age of cached entries (frames).</summary>
    public int MaxCacheAge { get; set; } = 30;

    /// <summary>Enable spatial coherence (reuse nearby evaluations).</summary>
    public bool EnableSpatialCoherence { get; set; } = true;

    /// <summary>Spatial coherence radius for cache lookup.</summary>
    public float SpatialCoherenceRadius { get; set; } = 0.1f;

    /// <summary>Enable predictive prefetching.</summary>
    public bool EnablePredictivePrefetch { get; set; } = true;

    /// <summary>Number of frames to predict ahead.</summary>
    public int PredictionFrames { get; set; } = 2;

    /// <summary>Enable async cache population.</summary>
    public bool EnableAsyncPopulation { get; set; } = true;

    /// <summary>Maximum concurrent cache operations.</summary>
    public int MaxConcurrentOps { get; set; } = 4;

    /// <summary>Enable cache statistics collection.</summary>
    public bool EnableStatistics { get; set; } = true;

    /// <summary>World-space bounds for the cache.</summary>
    public IntervalBox WorldBounds { get; set; } = new IntervalBox
    {
        Min = new Vector3(-100, -100, -100),
        Max = new Vector3(100, 100, 100)
    };
}

/// <summary>
/// Represents a single entry in the SDF cache.
/// </summary>
public readonly struct SdfCacheEntry
{
    /// <summary>Cache key (position quantized to grid).</summary>
    public readonly Vector3Int GridPosition;

    /// <summary>Cached SDF value.</summary>
    public readonly float SdfValue;

    /// <summary>Cached gradient.</summary>
    public readonly Vector3 Gradient;

    /// <summary>Frame when this entry was created.</summary>
    public readonly int FrameCreated;

    /// <summary>Access count for LRU eviction.</summary>
    public readonly int AccessCount;

    /// <summary>LOD level this entry was computed at.</summary>
    public readonly int LodLevel;

    /// <summary>Creates a new cache entry.</summary>
    public SdfCacheEntry(Vector3Int gridPos, float sdf, Vector3 gradient, int frame, int accessCount, int lodLevel)
    {
        GridPosition = gridPos;
        SdfValue = sdf;
        Gradient = gradient;
        FrameCreated = frame;
        AccessCount = accessCount;
        LodLevel = lodLevel;
    }
}

/// <summary>
/// Integer vector for grid-based cache indexing.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Vector3Int : IEquatable<Vector3Int>
{
    public readonly int X;
    public readonly int Y;
    public readonly int Z;

    public Vector3Int(int x, int y, int z) { X = x; Y = y; Z = z; }

    public bool Equals(Vector3Int other) => X == other.X && Y == other.Y && Z == other.Z;
    public override bool Equals(object? obj) => obj is Vector3Int other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(X, Y, Z);
    public override string ToString() => $"({X}, {Y}, {Z})";

    public static bool operator ==(Vector3Int left, Vector3Int right) => left.Equals(right);
    public static bool operator !=(Vector3Int left, Vector3Int right) => !left.Equals(right);
}

/// <summary>
/// Statistics for the hierarchical SDF cache.
/// </summary>
public sealed class CacheStatistics
{
    /// <summary>Total cache lookups.</summary>
    public long TotalLookups { get; set; }

    /// <summary>Cache hits.</summary>
    public long CacheHits { get; set; }

    /// <summary>Cache misses.</summary>
    public long CacheMisses { get; set; }

    /// <summary>Hit rate as a fraction.</summary>
    public float HitRate => TotalLookups > 0 ? (float)CacheHits / TotalLookups : 0;

    /// <summary>Total cache evictions.</summary>
    public long Evictions { get; set; }

    /// <summary>Average access count per entry.</summary>
    public float AverageAccessCount { get; set; }

    /// <summary>Cache entries per level.</summary>
    public int[] EntriesPerLevel { get; set; } = Array.Empty<int>();

    /// <summary>Returns a summary string.</summary>
    public override string ToString()
    {
        return $"HitRate={HitRate:P1} Hits={CacheHits} Misses={CacheMisses} Evictions={Evictions}";
    }
}

/// <summary>
/// Hierarchical SDF cache for accelerating neural SDF evaluations.
/// Implements multi-level caching with temporal coherence, spatial hashing,
/// and predictive prefetching.
/// </summary>
public sealed class HierarchicalSdfCache : IDisposable
{
    private readonly HierarchicalCacheConfig _config;
    private readonly List<Dictionary<Vector3Int, SdfCacheEntry>> _cacheLevels;
    private readonly ReaderWriterLockSlim _cacheLock;
    private readonly Random _rng;
    private int _currentFrame;
    private bool _disposed;

    // Performance counters
    private long _totalLookups;
    private long _cacheHits;
    private long _cacheMisses;
    private long _evictions;

    // World-to-grid transformation
    private readonly float _cellSize;
    private readonly Vector3 _gridOrigin;

    /// <summary>Gets the configuration.</summary>
    public HierarchicalCacheConfig Config => _config;

    /// <summary>Gets cache statistics.</summary>
    public CacheStatistics Statistics
    {
        get
        {
            var stats = new CacheStatistics
            {
                TotalLookups = Interlocked.CompareExchange(ref _totalLookups, 0, 0),
                CacheHits = Interlocked.CompareExchange(ref _cacheHits, 0, 0),
                CacheMisses = Interlocked.CompareExchange(ref _cacheMisses, 0, 0),
                Evictions = Interlocked.CompareExchange(ref _evictions, 0, 0),
                EntriesPerLevel = new int[_cacheLevels.Count]
            };

            _cacheLock.EnterReadLock();
            try
            {
                for (int i = 0; i < _cacheLevels.Count; i++)
                    stats.EntriesPerLevel[i] = _cacheLevels[i].Count;
            }
            finally { _cacheLock.ExitReadLock(); }

            return stats;
        }
    }

    /// <summary>
    /// Initializes a new hierarchical SDF cache.
    /// </summary>
    public HierarchicalSdfCache(HierarchicalCacheConfig? config = null)
    {
        _config = config ?? new HierarchicalCacheConfig();
        _rng = new Random(42);
        _currentFrame = 0;

        // Compute grid parameters
        Vector3 size = _config.WorldBounds.Size;
        float maxSize = MathF.Max(size.X, MathF.Max(size.Y, size.Z));
        _cellSize = maxSize / _config.BaseResolution;
        _gridOrigin = _config.WorldBounds.Min;

        // Initialize cache levels
        _cacheLevels = new List<Dictionary<Vector3Int, SdfCacheEntry>>();
        for (int i = 0; i < _config.HierarchyLevels; i++)
        {
            _cacheLevels.Add(new Dictionary<Vector3Int, SdfCacheEntry>());
        }

        _cacheLock = new ReaderWriterLockSlim();
    }

    /// <summary>
    /// Looks up a cached SDF value at a point.
    /// Returns true if found, false on cache miss.
    /// </summary>
    public bool TryLookup(Vector3 point, out float sdfValue, out Vector3 gradient, int lodLevel = 0)
    {
        sdfValue = 0;
        gradient = Vector3.Zero;

        if (!_config.EnableSpatialCoherence)
            return false;

        Interlocked.Increment(ref _totalLookups);

        Vector3Int gridPos = WorldToGrid(point, lodLevel);

        _cacheLock.EnterReadLock();
        try
        {
            if (lodLevel < _cacheLevels.Count &&
                _cacheLevels[lodLevel].TryGetValue(gridPos, out var entry))
            {
                // Check if entry is still valid
                if (_currentFrame - entry.FrameCreated <= _config.MaxCacheAge)
                {
                    sdfValue = entry.SdfValue;
                    gradient = entry.Gradient;
                    Interlocked.Increment(ref _cacheHits);
                    return true;
                }
            }
        }
        finally { _cacheLock.ExitReadLock(); }

        Interlocked.Increment(ref _cacheMisses);
        return false;
    }

    /// <summary>
    /// Looks up a cached SDF value (gradient-free version).
    /// </summary>
    public bool TryLookupSimple(Vector3 point, out float sdfValue, int lodLevel = 0)
    {
        return TryLookup(point, out sdfValue, out _, lodLevel);
    }

    /// <summary>
    /// Stores a computed SDF value in the cache.
    /// </summary>
    public void Store(Vector3 point, float sdfValue, Vector3 gradient, int lodLevel = 0)
    {
        if (!_config.EnableSpatialCoherence)
            return;

        Vector3Int gridPos = WorldToGrid(point, lodLevel);

        _cacheLock.EnterWriteLock();
        try
        {
            if (lodLevel < _cacheLevels.Count)
            {
                var entry = new SdfCacheEntry(gridPos, sdfValue, gradient, _currentFrame, 1, lodLevel);

                if (_cacheLevels[lodLevel].ContainsKey(gridPos))
                {
                    _cacheLevels[lodLevel][gridPos] = entry;
                }
                else
                {
                    // Check if we need to evict
                    if (_cacheLevels[lodLevel].Count >= _config.CacheSizePerLevel)
                    {
                        EvictOldest(lodLevel);
                    }

                    _cacheLevels[lodLevel][gridPos] = entry;
                }
            }
        }
        finally { _cacheLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Evaluates SDF with cache acceleration.
    /// Returns cached value if available, otherwise evaluates and caches.
    /// </summary>
    public float EvaluateCached(MicroMLP network, Vector3 point, int lodLevel = 0)
    {
        if (TryLookupSimple(point, out float cachedSdf, lodLevel))
        {
            return cachedSdf;
        }

        // Cache miss - evaluate and store
        float sdf = network.Evaluate(point);
        Vector3 gradient = network.ComputeGradient(point);
        Store(point, sdf, gradient, lodLevel);

        return sdf;
    }

    /// <summary>
    /// Evaluates SDF with cache acceleration (DeepMicroMLP version).
    /// </summary>
    public float EvaluateCached(DeepMicroMLP network, Vector3 point, int lodLevel = 0)
    {
        if (TryLookupSimple(point, out float cachedSdf, lodLevel))
        {
            return cachedSdf;
        }

        float sdf = network.Evaluate(point);
        Vector3 gradient = network.ComputeGradient(point);
        Store(point, sdf, gradient, lodLevel);

        return sdf;
    }

    /// <summary>
    /// Evaluates SDF with gradient and cache acceleration.
    /// </summary>
    public float EvaluateCachedWithGradient(MicroMLP network, Vector3 point, out Vector3 gradient, int lodLevel = 0)
    {
        if (TryLookup(point, out float cachedSdf, out gradient, lodLevel))
        {
            return cachedSdf;
        }

        float sdf = network.EvaluateWithGradient(point, out gradient);
        Store(point, sdf, gradient, lodLevel);

        return sdf;
    }

    /// <summary>
    /// Prefetches SDF values for a region around a point.
    /// Useful for predictive caching based on camera motion.
    /// </summary>
    public void PrefetchRegion(MicroMLP network, Vector3 center, float radius, int lodLevel = 0)
    {
        if (!_config.EnablePredictivePrefetch)
            return;

        int prefetchCount = 0;
        float stepSize = _cellSize * (1 << lodLevel);

        for (float z = center.Z - radius; z <= center.Z + radius; z += stepSize)
            for (float y = center.Y - radius; y <= center.Y + radius; y += stepSize)
                for (float x = center.X - radius; x <= center.X + radius; x += stepSize)
                {
                    Vector3 point = new Vector3(x, y, z);
                    if (Vector3.Distance(point, center) <= radius)
                    {
                        EvaluateCached(network, point, lodLevel);
                        prefetchCount++;

                        if (prefetchCount >= _config.CacheSizePerLevel / 4)
                            return; // Limit prefetch work
                    }
                }
    }

    /// <summary>
    /// Prefetches based on camera velocity prediction.
    /// </summary>
    public void PrefetchPredictive(MicroMLP network, CameraState camera, float predictionHorizon = 0.1f)
    {
        if (!_config.EnablePredictivePrefetch)
            return;

        // Predict future camera position
        Vector3 futurePosition = camera.Position + camera.Velocity * predictionHorizon;

        // Prefetch around current and future positions
        PrefetchRegion(network, camera.Position, _config.SpatialCoherenceRadius * 10, 0);
        PrefetchRegion(network, futurePosition, _config.SpatialCoherenceRadius * 10, 0);
    }

    /// <summary>
    /// Advances to the next frame and performs maintenance.
    /// </summary>
    public void AdvanceFrame()
    {
        _currentFrame++;

        // Perform cache maintenance every 30 frames
        if (_currentFrame % 30 == 0)
        {
            PerformMaintenance();
        }
    }

    /// <summary>
    /// Performs cache maintenance (evict old entries, update statistics).
    /// </summary>
    public void PerformMaintenance()
    {
        _cacheLock.EnterWriteLock();
        try
        {
            for (int level = 0; level < _cacheLevels.Count; level++)
            {
                var toRemove = new global::System.Collections.Generic.List<Vector3Int>();

                foreach (var kvp in _cacheLevels[level])
                {
                    if (_currentFrame - kvp.Value.FrameCreated > _config.MaxCacheAge)
                    {
                        toRemove.Add(kvp.Key);
                    }
                }

                foreach (var key in toRemove)
                {
                    _cacheLevels[level].Remove(key);
                    Interlocked.Increment(ref _evictions);
                }
            }
        }
        finally { _cacheLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Clears all cached data.
    /// </summary>
    public void Clear()
    {
        _cacheLock.EnterWriteLock();
        try
        {
            for (int i = 0; i < _cacheLevels.Count; i++)
                _cacheLevels[i].Clear();
        }
        finally { _cacheLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Gets the cache occupancy at each level.
    /// </summary>
    public int[] GetOccupancy()
    {
        var occupancy = new int[_cacheLevels.Count];
        _cacheLock.EnterReadLock();
        try
        {
            for (int i = 0; i < _cacheLevels.Count; i++)
                occupancy[i] = _cacheLevels[i].Count;
        }
        finally { _cacheLock.ExitReadLock(); }
        return occupancy;
    }

    /// <summary>
    /// Resets all statistics.
    /// </summary>
    public void ResetStatistics()
    {
        Interlocked.Exchange(ref _totalLookups, 0);
        Interlocked.Exchange(ref _cacheHits, 0);
        Interlocked.Exchange(ref _cacheMisses, 0);
        Interlocked.Exchange(ref _evictions, 0);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _cacheLock.Dispose();
        _cacheLevels.Clear();
        GC.SuppressFinalize(this);
    }

    #region Private Helpers

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private Vector3Int WorldToGrid(Vector3 worldPos, int lodLevel)
    {
        float scale = 1.0f / (_cellSize * (1 << lodLevel));
        Vector3 local = (worldPos - _gridOrigin) * scale;

        return new Vector3Int(
            (int)MathF.Floor(local.X),
            (int)MathF.Floor(local.Y),
            (int)MathF.Floor(local.Z)
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private Vector3 GridToWorld(Vector3Int gridPos, int lodLevel)
    {
        float scale = _cellSize * (1 << lodLevel);
        return new Vector3(
            gridPos.X * scale + _gridOrigin.X,
            gridPos.Y * scale + _gridOrigin.Y,
            gridPos.Z * scale + _gridOrigin.Z
        );
    }

    private void EvictOldest(int level)
    {
        if (_cacheLevels[level].Count == 0)
            return;

        Vector3Int oldestKey = default;
        int oldestFrame = int.MaxValue;

        foreach (var kvp in _cacheLevels[level])
        {
            if (kvp.Value.FrameCreated < oldestFrame)
            {
                oldestFrame = kvp.Value.FrameCreated;
                oldestKey = kvp.Key;
            }
        }

        if (oldestFrame < int.MaxValue)
        {
            _cacheLevels[level].Remove(oldestKey);
            Interlocked.Increment(ref _evictions);
        }
    }

    #endregion
}


