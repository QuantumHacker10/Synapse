using System;
// ============================================================
// FILE: AssetCache.cs
// PATH: Streaming/AssetCache.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Streaming
{
    /// <summary>
    /// Configuration for the asset cache.
    /// </summary>
    public sealed class AssetCacheConfig
    {
        /// <summary>Maximum number of entries in the cache.</summary>
        public int MaxEntries { get; set; } = 4096;

        /// <summary>Maximum total size in bytes.</summary>
        public long MaxSizeBytes { get; set; } = 256L * 1024 * 1024;

        /// <summary>Default time-to-live for entries in seconds (0 = no expiry).</summary>
        public float DefaultTtlSeconds { get; set; } = 300.0f;

        /// <summary>Interval in seconds for running cleanup of expired entries.</summary>
        public float CleanupIntervalSeconds { get; set; } = 30.0f;

        /// <summary>Minimum number of entries to evict per cleanup cycle.</summary>
        public int MinEvictionBatchSize { get; set; } = 16;

        /// <summary>Maximum eviction batch size.</summary>
        public int MaxEvictionBatchSize { get; set; } = 256;

        /// <summary>Whether to enable time-based expiration.</summary>
        public bool EnableTtl { get; set; } = true;

        /// <summary>Whether to enable size-based eviction.</summary>
        public bool EnableSizeEviction { get; set; } = true;

        /// <summary>Whether to track access statistics.</summary>
        public bool EnableStatistics { get; set; } = true;
    }

    /// <summary>
    /// Represents a single entry in the asset cache.
    /// </summary>
    internal sealed class CacheEntry
    {
        /// <summary>Cache key.</summary>
        public required string Key { get; init; }

        /// <summary>Cached asset value.</summary>
        public required NeuralAsset Value { get; set; }

        /// <summary>Size of this entry in bytes.</summary>
        public long SizeBytes { get; set; }

        /// <summary>Creation time.</summary>
        public DateTime CreatedTime { get; init; } = DateTime.UtcNow;

        /// <summary>Last access time (for LRU).</summary>
        public DateTime LastAccessTime { get; set; } = DateTime.UtcNow;

        /// <summary>Expiration time (null = no expiry).</summary>
        public DateTime? ExpirationTime { get; set; }

        /// <summary>Number of times this entry has been accessed.</summary>
        public long AccessCount;

        /// <summary>Priority level (higher = more important).</summary>
        public int Priority { get; set; }

        /// <summary>Whether this entry is pinned and should not be evicted.</summary>
        public volatile bool IsPinned;

        /// <summary>Whether this entry is currently being evicted.</summary>
        public volatile bool IsEvicting;

        /// <summary>
        /// Whether this entry has expired.
        /// </summary>
        public bool IsExpired => ExpirationTime.HasValue && DateTime.UtcNow > ExpirationTime.Value;

        /// <summary>
        /// Age of the entry since last access.
        /// </summary>
        public TimeSpan TimeSinceLastAccess => DateTime.UtcNow - LastAccessTime;
    }

    /// <summary>
    /// Cache hit/miss statistics for the asset cache.
    /// </summary>
    public sealed class AssetCacheStatistics
    {
        internal long _hits;
        internal long _misses;
        internal long _evictions;
        internal long _expiredEvictions;
        internal long _sizeEvictions;
        internal long _puts;
        internal long _removes;

        /// <summary>Total cache hits.</summary>
        public long Hits
        {
            get => Interlocked.Read(ref _hits);
            set => Interlocked.Exchange(ref _hits, value);
        }

        /// <summary>Total cache misses.</summary>
        public long Misses
        {
            get => Interlocked.Read(ref _misses);
            set => Interlocked.Exchange(ref _misses, value);
        }

        /// <summary>Total evictions (for any reason).</summary>
        public long Evictions
        {
            get => Interlocked.Read(ref _evictions);
            set => Interlocked.Exchange(ref _evictions, value);
        }

        /// <summary>Evictions due to TTL expiry.</summary>
        public long ExpiredEvictions
        {
            get => Interlocked.Read(ref _expiredEvictions);
            set => Interlocked.Exchange(ref _expiredEvictions, value);
        }

        /// <summary>Evictions due to size limits.</summary>
        public long SizeEvictions
        {
            get => Interlocked.Read(ref _sizeEvictions);
            set => Interlocked.Exchange(ref _sizeEvictions, value);
        }

        /// <summary>Total put operations.</summary>
        public long Puts
        {
            get => Interlocked.Read(ref _puts);
            set => Interlocked.Exchange(ref _puts, value);
        }

        /// <summary>Total remove operations.</summary>
        public long Removes
        {
            get => Interlocked.Read(ref _removes);
            set => Interlocked.Exchange(ref _removes, value);
        }

        /// <summary>Cache hit rate (0.0-1.0).</summary>
        public double HitRate
        {
            get
            {
                long h = Hits;
                long m = Misses;
                long total = h + m;
                return total > 0 ? (double)h / total : 0.0;
            }
        }

        /// <summary>Current number of entries in the cache.</summary>
        public int CurrentEntryCount { get; set; }

        /// <summary>Current total size in bytes.</summary>
        public long CurrentSizeBytes { get; set; }

        /// <summary>Peak entry count.</summary>
        public long PeakEntryCount { get; set; }

        /// <summary>Peak size in bytes.</summary>
        public long PeakSizeBytes { get; set; }

        /// <summary>Resets all statistics.</summary>
        public void Reset()
        {
            Interlocked.Exchange(ref _hits, 0);
            Interlocked.Exchange(ref _misses, 0);
            Interlocked.Exchange(ref _evictions, 0);
            Interlocked.Exchange(ref _expiredEvictions, 0);
            Interlocked.Exchange(ref _sizeEvictions, 0);
            Interlocked.Exchange(ref _puts, 0);
            Interlocked.Exchange(ref _removes, 0);
            CurrentEntryCount = 0;
            CurrentSizeBytes = 0;
            PeakEntryCount = 0;
            PeakSizeBytes = 0;
        }
    }

    /// <summary>
    /// Thread-safe LRU cache for decoded neural assets with size-based eviction,
    /// time-based expiration, and comprehensive hit/miss statistics.
    /// </summary>
    public sealed class AssetCache : IDisposable
    {
        private readonly AssetCacheConfig _config;
        private readonly ConcurrentDictionary<string, CacheEntry> _entries;
        private readonly object _lruLock = new();
        private readonly List<string> _lruOrder;
        private readonly AssetCacheStatistics _statistics;
        private readonly Timer _cleanupTimer;
        private long _currentSizeBytes;
        private bool _disposed;

        /// <summary>Raised when an entry is evicted from the cache.</summary>
        public event EventHandler<CacheEvictionEventArgs>? EntryEvicted;

        /// <summary>Gets the current cache statistics.</summary>
        public AssetCacheStatistics Statistics => _statistics;

        /// <summary>Gets the current number of entries.</summary>
        public int Count => _entries.Count;

        /// <summary>Gets the current total size in bytes.</summary>
        public long CurrentSizeBytes => Interlocked.Read(ref _currentSizeBytes);

        /// <summary>
        /// Initializes a new asset cache.
        /// </summary>
        /// <param name="maxEntries">Maximum number of entries.</param>
        /// <param name="maxSizeBytes">Maximum total size in bytes.</param>
        /// <param name="config">Optional configuration.</param>
        public AssetCache(int maxEntries = 4096, long maxSizeBytes = 256L * 1024 * 1024, AssetCacheConfig? config = null)
        {
            _config = config ?? new AssetCacheConfig
            {
                MaxEntries = maxEntries,
                MaxSizeBytes = maxSizeBytes
            };
            _entries = new ConcurrentDictionary<string, CacheEntry>(StringComparer.OrdinalIgnoreCase);
            _lruOrder = new List<string>();
            _statistics = new AssetCacheStatistics();

            _cleanupTimer = new Timer(
                _ => CleanupExpiredEntries(),
                null,
                TimeSpan.FromSeconds(_config.CleanupIntervalSeconds),
                TimeSpan.FromSeconds(_config.CleanupIntervalSeconds));
        }

        /// <summary>
        /// Retrieves a cached asset by key.
        /// Returns null if not found or expired.
        /// Updates LRU order and access statistics.
        /// </summary>
        /// <param name="key">Cache key.</param>
        /// <returns>The cached NeuralAsset, or null.</returns>
        public NeuralAsset? Get(string key)
        {
            if (string.IsNullOrEmpty(key))
                return null;

            if (!_entries.TryGetValue(key, out var entry))
            {
                Interlocked.Increment(ref _statistics._misses);
                return null;
            }

            if (entry.IsExpired)
            {
                Remove(key);
                Interlocked.Increment(ref _statistics._misses);
                return null;
            }

            entry.LastAccessTime = DateTime.UtcNow;
            Interlocked.Increment(ref entry.AccessCount);
            Interlocked.Increment(ref _statistics._hits);

            lock (_lruLock)
            {
                _lruOrder.Remove(key);
                _lruOrder.Add(key);
            }

            return entry.Value;
        }

        /// <summary>
        /// Tries to get a cached asset without modifying LRU order.
        /// </summary>
        /// <param name="key">Cache key.</param>
        /// <param name="asset">The cached asset if found.</param>
        /// <returns>True if found and not expired.</returns>
        public bool TryGet(string key, out NeuralAsset? asset)
        {
            if (string.IsNullOrEmpty(key))
            {
                asset = null;
                return false;
            }

            if (!_entries.TryGetValue(key, out var entry) || entry.IsExpired)
            {
                asset = null;
                if (entry?.IsExpired == true)
                    Remove(key);
                return false;
            }

            entry.LastAccessTime = DateTime.UtcNow;
            Interlocked.Increment(ref entry.AccessCount);
            Interlocked.Increment(ref _statistics._hits);

            asset = entry.Value;
            return true;
        }

        /// <summary>
        /// Adds or updates a cached asset.
        /// </summary>
        /// <param name="key">Cache key.</param>
        /// <param name="asset">NeuralAsset to cache.</param>
        /// <param name="ttlSeconds">Time-to-live in seconds (null = use default).</param>
        /// <param name="priority">Priority level for eviction decisions.</param>
        public void Put(string key, NeuralAsset asset, float? ttlSeconds = null, int priority = 0)
        {
            if (string.IsNullOrEmpty(key) || asset == null)
                return;

            long sizeBytes = asset.ComputeMemoryFootprint();

            if (_entries.TryGetValue(key, out var existing))
            {
                Interlocked.Add(ref _currentSizeBytes, -existing.SizeBytes);
                existing.Value = asset;
                existing.SizeBytes = sizeBytes;
                existing.LastAccessTime = DateTime.UtcNow;
                existing.Priority = priority;

                if (_config.EnableTtl)
                {
                    float ttl = ttlSeconds ?? _config.DefaultTtlSeconds;
                    existing.ExpirationTime = ttl > 0 ? DateTime.UtcNow.AddSeconds(ttl) : null;
                }

                Interlocked.Add(ref _currentSizeBytes, sizeBytes);
            }
            else
            {
                float effectiveTtl = ttlSeconds ?? _config.DefaultTtlSeconds;
                var entry = new CacheEntry
                {
                    Key = key,
                    Value = asset,
                    SizeBytes = sizeBytes,
                    Priority = priority,
                    ExpirationTime = _config.EnableTtl && effectiveTtl > 0
                        ? DateTime.UtcNow.AddSeconds(effectiveTtl)
                        : null
                };

                _entries[key] = entry;

                lock (_lruLock)
                {
                    _lruOrder.Add(key);
                }

                Interlocked.Add(ref _currentSizeBytes, sizeBytes);
            }

            Interlocked.Increment(ref _statistics._puts);
            UpdatePeakStats();
            EnforceSizeLimits();
        }

        /// <summary>
        /// Removes an entry from the cache.
        /// </summary>
        /// <param name="key">Cache key.</param>
        /// <returns>True if the entry was found and removed.</returns>
        public bool Remove(string key)
        {
            if (string.IsNullOrEmpty(key))
                return false;

            if (!_entries.TryRemove(key, out var entry))
                return false;

            Interlocked.Add(ref _currentSizeBytes, -entry.SizeBytes);
            Interlocked.Increment(ref _statistics._removes);

            lock (_lruLock)
            {
                _lruOrder.Remove(key);
            }

            EntryEvicted?.Invoke(this, new CacheEvictionEventArgs
            {
                Key = key,
                SizeBytes = entry.SizeBytes,
                Reason = CacheEvictionReason.Removed,
                AccessCount = Interlocked.Read(ref entry.AccessCount)
            });

            return true;
        }

        /// <summary>
        /// Pins an entry to prevent it from being evicted.
        /// </summary>
        /// <param name="key">Cache key.</param>
        /// <returns>True if the entry was found and pinned.</returns>
        public bool Pin(string key)
        {
            if (_entries.TryGetValue(key, out var entry))
            {
                entry.IsPinned = true;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Unpins an entry, allowing it to be evicted.
        /// </summary>
        /// <param name="key">Cache key.</param>
        /// <returns>True if the entry was found and unpinned.</returns>
        public bool Unpin(string key)
        {
            if (_entries.TryGetValue(key, out var entry))
            {
                entry.IsPinned = false;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Checks if a key exists in the cache and is not expired.
        /// </summary>
        /// <param name="key">Cache key.</param>
        /// <returns>True if the key exists and is valid.</returns>
        public bool Contains(string key)
        {
            if (!_entries.TryGetValue(key, out var entry))
                return false;
            if (entry.IsExpired)
            { Remove(key); return false; }
            return true;
        }

        /// <summary>
        /// Gets all keys currently in the cache.
        /// </summary>
        /// <returns>Collection of cache keys.</returns>
        public IReadOnlyCollection<string> GetKeys()
        {
            return _entries.Keys.ToList().AsReadOnly();
        }

        /// <summary>
        /// Gets cache entries matching a prefix.
        /// </summary>
        /// <param name="keyPrefix">Key prefix to match.</param>
        /// <returns>Matching keys.</returns>
        public IReadOnlyCollection<string> GetKeysByPrefix(string keyPrefix)
        {
            if (string.IsNullOrEmpty(keyPrefix))
                return GetKeys();

            return _entries.Keys
                .Where(k => k.StartsWith(keyPrefix, StringComparison.OrdinalIgnoreCase))
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Evicts the least recently used entries to bring the cache within size limits.
        /// </summary>
        /// <param name="targetSizeBytes">Target size to evict down to (null = use config max).</param>
        /// <returns>Number of entries evicted.</returns>
        public int Evict(long? targetSizeBytes = null)
        {
            long target = targetSizeBytes ?? _config.MaxSizeBytes;
            int evicted = 0;

            lock (_lruLock)
            {
                while (CurrentSizeBytes > target && _lruOrder.Count > 0)
                {
                    string keyToEvict = _lruOrder[0];

                    if (_entries.TryGetValue(keyToEvict, out var entry) && !entry.IsPinned)
                    {
                        entry.IsEvicting = true;

                        if (_entries.TryRemove(keyToEvict, out _))
                        {
                            _lruOrder.RemoveAt(0);
                            Interlocked.Add(ref _currentSizeBytes, -entry.SizeBytes);
                            Interlocked.Increment(ref _statistics._evictions);
                            Interlocked.Increment(ref _statistics._sizeEvictions);

                            EntryEvicted?.Invoke(this, new CacheEvictionEventArgs
                            {
                                Key = keyToEvict,
                                SizeBytes = entry.SizeBytes,
                                Reason = CacheEvictionReason.SizeLimit,
                                AccessCount = Interlocked.Read(ref entry.AccessCount)
                            });

                            evicted++;
                        }
                    }
                    else
                    {
                        _lruOrder.RemoveAt(0);
                    }
                }
            }

            return evicted;
        }

        /// <summary>
        /// Clears all entries from the cache.
        /// </summary>
        /// <returns>Number of entries cleared.</returns>
        public int Clear()
        {
            int count = _entries.Count;
            _entries.Clear();

            lock (_lruLock)
            {
                _lruOrder.Clear();
            }

            Interlocked.Exchange(ref _currentSizeBytes, 0);
            return count;
        }

        /// <summary>
        /// Gets the total size in bytes for entries matching a prefix.
        /// </summary>
        /// <param name="keyPrefix">Key prefix.</param>
        /// <returns>Total size in bytes.</returns>
        public long GetSizeByPrefix(string keyPrefix)
        {
            return _entries.Values
                .Where(e => e.Key.StartsWith(keyPrefix, StringComparison.OrdinalIgnoreCase))
                .Sum(e => e.SizeBytes);
        }

        /// <summary>
        /// Gets the number of entries matching a prefix.
        /// </summary>
        /// <param name="keyPrefix">Key prefix.</param>
        /// <returns>Entry count.</returns>
        public int GetCountByPrefix(string keyPrefix)
        {
            return _entries.Values
                .Count(e => e.Key.StartsWith(keyPrefix, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Removes all expired entries from the cache.
        /// </summary>
        /// <returns>Number of entries removed.</returns>
        public int CleanupExpiredEntries()
        {
            if (!_config.EnableTtl)
                return 0;

            var expiredKeys = _entries.Values
                .Where(e => e.IsExpired)
                .Select(e => e.Key)
                .ToList();

            int removed = 0;
            foreach (var key in expiredKeys)
            {
                if (_entries.TryRemove(key, out var entry))
                {
                    Interlocked.Add(ref _currentSizeBytes, -entry.SizeBytes);
                    Interlocked.Increment(ref _statistics._evictions);
                    Interlocked.Increment(ref _statistics._expiredEvictions);

                    lock (_lruLock)
                    {
                        _lruOrder.Remove(key);
                    }

                    EntryEvicted?.Invoke(this, new CacheEvictionEventArgs
                    {
                        Key = key,
                        SizeBytes = entry.SizeBytes,
                        Reason = CacheEvictionReason.TtlExpired,
                        AccessCount = Interlocked.Read(ref entry.AccessCount)
                    });

                    removed++;
                }
            }

            return removed;
        }

        /// <summary>
        /// Enforces size and count limits by evicting LRU entries.
        /// </summary>
        private void EnforceSizeLimits()
        {
            if (_config.EnableSizeEviction && CurrentSizeBytes > _config.MaxSizeBytes)
            {
                long target = (long)(_config.MaxSizeBytes * 0.8);
                Evict(target);
            }

            if (_entries.Count > _config.MaxEntries)
            {
                int excess = _entries.Count - _config.MaxEntries;
                lock (_lruLock)
                {
                    for (int i = 0; i < Math.Min(excess, _lruOrder.Count); i++)
                    {
                        string key = _lruOrder[0];
                        if (_entries.TryRemove(key, out var entry) && !entry.IsPinned)
                        {
                            Interlocked.Add(ref _currentSizeBytes, -entry.SizeBytes);
                            Interlocked.Increment(ref _statistics._evictions);
                            _lruOrder.RemoveAt(0);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Updates peak usage statistics.
        /// </summary>
        private void UpdatePeakStats()
        {
            long currentEntries = _entries.Count;
            long currentSize = CurrentSizeBytes;

            if (currentEntries > _statistics.PeakEntryCount)
            {
                _statistics.PeakEntryCount = currentEntries;
            }

            if (currentSize > _statistics.PeakSizeBytes)
            {
                _statistics.PeakSizeBytes = currentSize;
            }

            _statistics.CurrentEntryCount = _entries.Count;
            _statistics.CurrentSizeBytes = CurrentSizeBytes;
        }

        /// <summary>
        /// Disposes the cache and releases resources.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;

            _cleanupTimer.Dispose();
            Clear();
        }
    }

    /// <summary>
    /// Reasons for cache entry eviction.
    /// </summary>
    public enum CacheEvictionReason : byte
    {
        /// <summary>Entry expired due to TTL.</summary>
        TtlExpired = 0,

        /// <summary>Evicted to meet size limits.</summary>
        SizeLimit = 1,

        /// <summary>Evicted to meet count limits.</summary>
        CountLimit = 2,

        /// <summary>Explicitly removed by caller.</summary>
        Removed = 3,

        /// <summary>Cache was cleared.</summary>
        Cleared = 4
    }

    /// <summary>
    /// Event arguments for cache eviction events.
    /// </summary>
    public sealed class CacheEvictionEventArgs : EventArgs
    {
        /// <summary>Evicted entry key.</summary>
        public required string Key { get; init; }

        /// <summary>Size of the evicted entry in bytes.</summary>
        public long SizeBytes { get; init; }

        /// <summary>Reason for eviction.</summary>
        public CacheEvictionReason Reason { get; init; }

        /// <summary>Number of times the entry was accessed before eviction.</summary>
        public long AccessCount { get; init; }

        /// <summary>Returns a formatted description.</summary>
        public override string ToString() =>
            $"Evicted '{Key}': {SizeBytes} bytes, {AccessCount} accesses, reason={Reason}";
    }
}
