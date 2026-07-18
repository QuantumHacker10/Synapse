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
// FILE: AssetStreamer.cs
// PATH: Streaming/AssetStreamer.cs
// ============================================================


using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Streaming
{
    /// <summary>
    /// Represents the priority level for asset loading operations.
    /// Higher values indicate higher priority.
    /// </summary>
    public enum AssetPriority : byte
    {
        /// <summary>Background loading, lowest priority.</summary>
        Background = 0,

        /// <summary>Normal priority for standard asset loading.</summary>
        Normal = 1,

        /// <summary>High priority for assets needed by visible geometry.</summary>
        High = 2,

        /// <summary>Critical priority, blocks rendering if not loaded.</summary>
        Critical = 3,

        /// <summary>Immediate priority, bypasses all throttling.</summary>
        Immediate = 4
    }

    /// <summary>
    /// Represents the loading state of a neural asset.
    /// </summary>
    public enum AssetLoadingState : byte
    {
        /// <summary>Asset has not been requested yet.</summary>
        None = 0,

        /// <summary>Asset is queued for loading.</summary>
        Queued = 1,

        /// <summary>Asset dependencies are being resolved.</summary>
        ResolvingDependencies = 2,

        /// <summary>Asset is being downloaded from storage.</summary>
        Downloading = 3,

        /// <summary>Asset is being decompressed.</summary>
        Decompressing = 4,

        /// <summary>Asset is being decoded into usable form.</summary>
        Decoding = 5,

        /// <summary>Asset is being uploaded to GPU memory.</summary>
        Uploading = 6,

        /// <summary>Asset is fully loaded and ready for use.</summary>
        Loaded = 7,

        /// <summary>Asset loading failed.</summary>
        Failed = 8,

        /// <summary>Asset has been unloaded to reclaim memory.</summary>
        Unloaded = 9
    }

    /// <summary>
    /// Represents a loaded or loading neural asset with its associated metadata.
    /// </summary>
    public sealed class AssetEntry
    {
        /// <summary>Unique identifier for this asset.</summary>
        public required string AssetId { get; init; }

        /// <summary>The neural asset data, null if not yet loaded.</summary>
        public NeuralAsset? Asset { get; set; }

        /// <summary>Current loading state.</summary>
        public AssetLoadingState State { get; set; } = AssetLoadingState.None;

        /// <summary>Priority level for loading.</summary>
        public AssetPriority Priority { get; set; } = AssetPriority.Normal;

        /// <summary>Asset IDs that must be loaded before this asset.</summary>
        public List<string> Dependencies { get; init; } = new();

        /// <summary>LOD level, 0 is lowest quality.</summary>
        public int LodLevel { get; set; }

        /// <summary>Target mesh identifier.</summary>
        public int TargetMeshId { get; set; }

        /// <summary>Size in bytes of the asset in memory.</summary>
        public long MemorySize { get; set; }

        /// <summary>Timestamp when loading started.</summary>
        public DateTime? LoadingStartTime { get; set; }

        /// <summary>Timestamp when loading completed.</summary>
        public DateTime? LoadingEndTime { get; set; }

        /// <summary>Number of retry attempts made.</summary>
        public int RetryCount { get; set; }

        /// <summary>Last error message if loading failed.</summary>
        public string? LastError { get; set; }

        /// <summary>Cancellation token source for this asset's loading operation.</summary>
        public CancellationTokenSource? CancellationSource { get; set; }

        /// <summary>LRU timestamp for eviction tracking.</summary>
        public DateTime LastAccessTime { get; set; } = DateTime.UtcNow;

        /// <summary>Total load time in milliseconds.</summary>
        public double LoadTimeMs =>
            LoadingStartTime.HasValue && LoadingEndTime.HasValue
                ? (LoadingEndTime.Value - LoadingStartTime.Value).TotalMilliseconds
                : 0;

        /// <summary>Whether the asset is currently in use by the renderer.</summary>
        public bool IsInUse { get; set; }

        /// <summary>Reference count for pinning the asset in memory.</summary>
        public int ReferenceCount { get; set; }
    }

    /// <summary>
    /// Configuration for the asset streaming pipeline.
    /// </summary>
    public sealed class StreamerConfig
    {
        /// <summary>Maximum memory budget in bytes for loaded assets.</summary>
        public long MemoryBudgetBytes { get; set; } = 512L * 1024 * 1024;

        /// <summary>Maximum concurrent download operations.</summary>
        public int MaxConcurrentDownloads { get; set; } = 8;

        /// <summary>Maximum bandwidth in bytes per second (0 = unlimited).</summary>
        public long MaxBandwidthBytesPerSecond { get; set; } = 0;

        /// <summary>Maximum number of retry attempts for failed loads.</summary>
        public int MaxRetryAttempts { get; set; } = 3;

        /// <summary>Base delay between retry attempts in milliseconds.</summary>
        public int RetryBaseDelayMs { get; set; } = 100;

        /// <summary>Timeout for a single download operation in milliseconds.</summary>
        public int DownloadTimeoutMs { get; set; } = 30000;

        /// <summary>LRU eviction threshold as fraction of memory budget (0.0-1.0).</summary>
        public float EvictionThreshold { get; set; } = 0.85f;

        /// <summary>Time in seconds after which unused assets become eviction candidates.</summary>
        public float UnusedAssetTtlSeconds { get; set; } = 30.0f;

        /// <summary>Maximum number of assets to keep in cache.</summary>
        public int MaxCachedAssets { get; set; } = 4096;

        /// <summary>Batch size for progressive LOD loading.</summary>
        public int ProgressiveLodBatchSize { get; set; } = 4;

        /// <summary>Whether to enable bandwidth throttling.</summary>
        public bool EnableBandwidthThrottling { get; set; } = true;

        /// <summary>Whether to enable progressive LOD loading.</summary>
        public bool EnableProgressiveLoading { get; set; } = true;

        /// <summary>Statistics reporting interval in seconds.</summary>
        public float StatsReportingIntervalSeconds { get; set; } = 5.0f;

        /// <summary>Size classes for memory pool slab allocation in bytes.</summary>
        public int[] SlabSizeClasses { get; set; } =
            [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
    }

    /// <summary>
    /// Comprehensive statistics for the streaming pipeline.
    /// </summary>
    public sealed class StreamerStatistics
    {
        internal long _totalBytesLoaded;
        internal long _totalBytesDecompressed;
        internal long _totalAssetsLoaded;
        internal long _totalAssetsFailed;
        internal long _totalCacheHits;
        internal long _totalCacheMisses;
        internal long _totalEvictions;
        internal long _totalBandwidthBytes;
        private readonly Stopwatch _uptime = Stopwatch.StartNew();
        private readonly object _lock = new();

        /// <summary>Total bytes loaded from storage.</summary>
        public long TotalBytesLoaded
        {
            get => Interlocked.Read(ref _totalBytesLoaded);
            set => Interlocked.Exchange(ref _totalBytesLoaded, value);
        }

        /// <summary>Total bytes after decompression.</summary>
        public long TotalBytesDecompressed
        {
            get => Interlocked.Read(ref _totalBytesDecompressed);
            set => Interlocked.Exchange(ref _totalBytesDecompressed, value);
        }

        /// <summary>Total number of assets successfully loaded.</summary>
        public long TotalAssetsLoaded
        {
            get => Interlocked.Read(ref _totalAssetsLoaded);
            set => Interlocked.Exchange(ref _totalAssetsLoaded, value);
        }

        /// <summary>Total number of asset load failures.</summary>
        public long TotalAssetsFailed
        {
            get => Interlocked.Read(ref _totalAssetsFailed);
            set => Interlocked.Exchange(ref _totalAssetsFailed, value);
        }

        /// <summary>Cache hit count.</summary>
        public long TotalCacheHits
        {
            get => Interlocked.Read(ref _totalCacheHits);
            set => Interlocked.Exchange(ref _totalCacheHits, value);
        }

        /// <summary>Cache miss count.</summary>
        public long TotalCacheMisses
        {
            get => Interlocked.Read(ref _totalCacheMisses);
            set => Interlocked.Exchange(ref _totalCacheMisses, value);
        }

        /// <summary>Total LRU evictions performed.</summary>
        public long TotalEvictions
        {
            get => Interlocked.Read(ref _totalEvictions);
            set => Interlocked.Exchange(ref _totalEvictions, value);
        }

        /// <summary>Total bandwidth consumed in bytes.</summary>
        public long TotalBandwidthBytes
        {
            get => Interlocked.Read(ref _totalBandwidthBytes);
            set => Interlocked.Exchange(ref _totalBandwidthBytes, value);
        }

        /// <summary>Current bytes per second bandwidth usage.</summary>
        public double CurrentBandwidthBytesPerSecond { get; set; }

        /// <summary>Average load time in milliseconds.</summary>
        public double AverageLoadTimeMs { get; set; }

        /// <summary>Peak concurrent downloads.</summary>
        public int PeakConcurrentDownloads { get; set; }

        /// <summary>Current number of assets in cache.</summary>
        public int CachedAssetCount { get; set; }

        /// <summary>Current memory usage in bytes.</summary>
        public long CurrentMemoryUsageBytes { get; set; }

        /// <summary>Memory budget in bytes.</summary>
        public long MemoryBudgetBytes { get; set; }

        /// <summary>Uptime of the streamer.</summary>
        public TimeSpan Uptime => _uptime.Elapsed;

        /// <summary>Cache hit rate as a ratio (0.0-1.0).</summary>
        public double CacheHitRate
        {
            get
            {
                long hits = TotalCacheHits;
                long misses = TotalCacheMisses;
                long total = hits + misses;
                return total > 0 ? (double)hits / total : 0.0;
            }
        }

        /// <summary>Memory utilization as a ratio (0.0-1.0).</summary>
        public double MemoryUtilization
        {
            get => MemoryBudgetBytes > 0
                ? (double)CurrentMemoryUsageBytes / MemoryBudgetBytes
                : 0.0;
        }

        /// <summary>Compression ratio (decompressed size / compressed size).</summary>
        public double CompressionRatio
        {
            get
            {
                long compressed = TotalBytesLoaded;
                long decompressed = TotalBytesDecompressed;
                return compressed > 0 ? (double)decompressed / compressed : 1.0;
            }
        }

        /// <summary>Creates a snapshot of the current statistics.</summary>
        public StreamerStatisticsSnapshot CreateSnapshot()
        {
            lock (_lock)
            {
                return new StreamerStatisticsSnapshot
                {
                    Timestamp = DateTime.UtcNow,
                    TotalBytesLoaded = TotalBytesLoaded,
                    TotalBytesDecompressed = TotalBytesDecompressed,
                    TotalAssetsLoaded = TotalAssetsLoaded,
                    TotalAssetsFailed = TotalAssetsFailed,
                    CacheHitRate = CacheHitRate,
                    MemoryUtilization = MemoryUtilization,
                    CurrentBandwidthBytesPerSecond = CurrentBandwidthBytesPerSecond,
                    AverageLoadTimeMs = AverageLoadTimeMs,
                    CachedAssetCount = CachedAssetCount,
                    CurrentMemoryUsageBytes = CurrentMemoryUsageBytes,
                    TotalEvictions = TotalEvictions,
                    Uptime = Uptime
                };
            }
        }

        /// <summary>Resets all statistics to zero.</summary>
        public void Reset()
        {
            Interlocked.Exchange(ref _totalBytesLoaded, 0);
            Interlocked.Exchange(ref _totalBytesDecompressed, 0);
            Interlocked.Exchange(ref _totalAssetsLoaded, 0);
            Interlocked.Exchange(ref _totalAssetsFailed, 0);
            Interlocked.Exchange(ref _totalCacheHits, 0);
            Interlocked.Exchange(ref _totalCacheMisses, 0);
            Interlocked.Exchange(ref _totalEvictions, 0);
            Interlocked.Exchange(ref _totalBandwidthBytes, 0);
            CurrentBandwidthBytesPerSecond = 0;
            AverageLoadTimeMs = 0;
            PeakConcurrentDownloads = 0;
            CachedAssetCount = 0;
            CurrentMemoryUsageBytes = 0;
        }
    }

    /// <summary>
    /// Immutable snapshot of streaming statistics at a point in time.
    /// </summary>
    public sealed class StreamerStatisticsSnapshot
    {
        /// <summary>Timestamp of the snapshot.</summary>
        public DateTime Timestamp { get; init; }

        /// <summary>Total bytes loaded from storage.</summary>
        public long TotalBytesLoaded { get; init; }

        /// <summary>Total bytes after decompression.</summary>
        public long TotalBytesDecompressed { get; init; }

        /// <summary>Total number of assets successfully loaded.</summary>
        public long TotalAssetsLoaded { get; init; }

        /// <summary>Total number of asset load failures.</summary>
        public long TotalAssetsFailed { get; init; }

        /// <summary>Cache hit rate (0.0-1.0).</summary>
        public double CacheHitRate { get; init; }

        /// <summary>Memory utilization (0.0-1.0).</summary>
        public double MemoryUtilization { get; init; }

        /// <summary>Current bandwidth in bytes per second.</summary>
        public double CurrentBandwidthBytesPerSecond { get; init; }

        /// <summary>Average load time in milliseconds.</summary>
        public double AverageLoadTimeMs { get; init; }

        /// <summary>Number of cached assets.</summary>
        public int CachedAssetCount { get; init; }

        /// <summary>Current memory usage in bytes.</summary>
        public long CurrentMemoryUsageBytes { get; init; }

        /// <summary>Total evictions performed.</summary>
        public long TotalEvictions { get; init; }

        /// <summary>Total uptime.</summary>
        public TimeSpan Uptime { get; init; }

        /// <summary>Returns a formatted summary of the snapshot.</summary>
        public override string ToString() =>
            $"[Streamer Stats @ {Timestamp:HH:mm:ss.fff}] " +
            $"Assets: {TotalAssetsLoaded} loaded / {TotalAssetsFailed} failed | " +
            $"Cache: {CacheHitRate:P1} hit rate | " +
            $"Memory: {MemoryUtilization:P1} ({CurrentMemoryUsageBytes / (1024.0 * 1024.0):F1} MB) | " +
            $"Bandwidth: {CurrentBandwidthBytesPerSecond / (1024.0 * 1024.0):F1} MB/s";
    }

    /// <summary>
    /// Event arguments for asset loading progress notifications.
    /// </summary>
    public sealed class AssetLoadProgressEventArgs : EventArgs
    {
        /// <summary>Asset identifier.</summary>
        public required string AssetId { get; init; }

        /// <summary>Current loading state.</summary>
        public AssetLoadingState State { get; init; }

        /// <summary>Progress as a percentage (0-100).</summary>
        public float ProgressPercent { get; init; }

        /// <summary>Bytes downloaded so far.</summary>
        public long BytesDownloaded { get; init; }

        /// <summary>Total expected bytes.</summary>
        public long TotalBytes { get; init; }

        /// <summary>Error message if state is Failed.</summary>
        public string? ErrorMessage { get; init; }
    }

    /// <summary>
    /// Asynchronous streaming pipeline for loading, caching, and managing neural geometry assets.
    /// Supports priority-based loading, bandwidth throttling, progressive LOD loading,
    /// dependency tracking, memory budget management with LRU eviction, and comprehensive
    /// statistics tracking.
    /// </summary>
    public sealed class AssetStreamer : IDisposable
    {
        private readonly StreamerConfig _config;
        private readonly AssetCache _cache;
        private readonly MemoryPool _memoryPool;
        private readonly CompressionUtils _compression;
        private readonly AsyncPipeline _pipeline;
        private readonly StreamerStatistics _statistics;

        private readonly ConcurrentDictionary<string, AssetEntry> _assets = new();
        private readonly PriorityQueue<string, int> _loadQueue = new();
        private readonly SemaphoreSlim _downloadSemaphore;
        private readonly SemaphoreSlim _bandwidthSemaphore;
        private readonly CancellationTokenSource _shutdownCts;
        private readonly Task _processingTask;
        private readonly Task _statsTask;
        private readonly Task _evictionTask;
        private readonly BandwidthTracker _bandwidthTracker;

        private long _currentMemoryUsage;
        private int _activeDownloads;
        private bool _disposed;

        /// <summary>Raised when an asset's loading state changes.</summary>
        public event EventHandler<AssetLoadProgressEventArgs>? AssetLoadProgress;

        /// <summary>Raised when statistics are updated.</summary>
        public event EventHandler<StreamerStatisticsSnapshot>? StatisticsUpdated;

        /// <summary>Gets the current streaming statistics.</summary>
        public StreamerStatistics Statistics => _statistics;

        /// <summary>Gets the current configuration.</summary>
        public StreamerConfig Configuration => _config;

        /// <summary>Gets the number of currently active downloads.</summary>
        public int ActiveDownloads => _activeDownloads;

        /// <summary>Gets the current memory usage in bytes.</summary>
        public long CurrentMemoryUsage => Interlocked.Read(ref _currentMemoryUsage);

        /// <summary>
        /// Initializes a new instance of the <see cref="AssetStreamer"/> class.
        /// </summary>
        /// <param name="config">Configuration for the streaming pipeline.</param>
        /// <param name="memoryPool">Memory pool for buffer allocation.</param>
        /// <param name="compression">Compression utilities.</param>
        public AssetStreamer(
            StreamerConfig config,
            MemoryPool? memoryPool = null,
            CompressionUtils? compression = null)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _memoryPool = memoryPool ?? new MemoryPool(config.SlabSizeClasses);
            _compression = compression ?? new CompressionUtils();
            _cache = new AssetCache(config.MaxCachedAssets, config.MemoryBudgetBytes);
            _pipeline = new AsyncPipeline();
            _statistics = new StreamerStatistics { MemoryBudgetBytes = config.MemoryBudgetBytes };
            _bandwidthTracker = new BandwidthTracker();

            _downloadSemaphore = new SemaphoreSlim(config.MaxConcurrentDownloads, config.MaxConcurrentDownloads);
            _bandwidthSemaphore = new SemaphoreSlim(config.MaxConcurrentDownloads, config.MaxConcurrentDownloads);
            _shutdownCts = new CancellationTokenSource();
            _processingTask = Task.Run(() => ProcessLoadQueueAsync(_shutdownCts.Token));
            _statsTask = Task.Run(() => ReportStatsAsync(_shutdownCts.Token));
            _evictionTask = Task.Run(() => RunEvictionAsync(_shutdownCts.Token));
        }

        /// <summary>
        /// Requests an asset to be loaded asynchronously with the specified priority.
        /// </summary>
        /// <param name="assetId">Unique identifier of the asset to load.</param>
        /// <param name="priority">Loading priority.</param>
        /// <param name="lodLevel">LOD level to load (0 = lowest quality).</param>
        /// <param name="dependencies">Asset IDs that must be loaded first.</param>
        /// <param name="cancellationToken">Optional cancellation token.</param>
        /// <returns>Task representing the async operation.</returns>
        public async Task RequestAssetAsync(
            string assetId,
            AssetPriority priority = AssetPriority.Normal,
            int lodLevel = 0,
            IEnumerable<string>? dependencies = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(assetId);

            if (_assets.ContainsKey(assetId))
            {
                var existing = _assets[assetId];
                if (existing.State == AssetLoadingState.Loaded)
                {
                    existing.LastAccessTime = DateTime.UtcNow;
                    Interlocked.Increment(ref _statistics._totalCacheHits);
                    return;
                }

                if (existing.State is AssetLoadingState.Queued or AssetLoadingState.ResolvingDependencies
                    or AssetLoadingState.Downloading or AssetLoadingState.Decompressing
                    or AssetLoadingState.Decoding or AssetLoadingState.Uploading)
                {
                    if (existing.Priority < priority)
                    {
                        existing.Priority = priority;
                    }
                    return;
                }
            }

            var entry = new AssetEntry
            {
                AssetId = assetId,
                Priority = priority,
                LodLevel = lodLevel,
                State = AssetLoadingState.Queued,
                CancellationSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken)
            };

            if (dependencies != null)
            {
                entry.Dependencies.AddRange(dependencies);
            }

            _assets[assetId] = entry;

            lock (_loadQueue)
            {
                _loadQueue.Enqueue(assetId, (int)priority);
            }

            OnAssetLoadProgress(new AssetLoadProgressEventArgs
            {
                AssetId = assetId,
                State = AssetLoadingState.Queued,
                ProgressPercent = 0
            });

            await Task.CompletedTask;
        }

        /// <summary>
        /// Requests multiple assets to be loaded asynchronously.
        /// </summary>
        /// <param name="assetIds">Collection of asset identifiers to load.</param>
        /// <param name="priority">Loading priority for all assets.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>Task representing the async operation.</returns>
        public async Task RequestAssetsAsync(
            IEnumerable<string> assetIds,
            AssetPriority priority = AssetPriority.Normal,
            CancellationToken cancellationToken = default)
        {
            var tasks = assetIds.Select(id => RequestAssetAsync(id, priority, cancellationToken: cancellationToken));
            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Requests progressive loading of an asset starting from the lowest LOD level.
        /// </summary>
        /// <param name="assetId">Asset identifier.</param>
        /// <param name="maxLodLevel">Maximum LOD level to load up to.</param>
        /// <param name="priority">Loading priority.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>Async task.</returns>
        public async Task RequestProgressiveAsync(
            string assetId,
            int maxLodLevel,
            AssetPriority priority = AssetPriority.Normal,
            CancellationToken cancellationToken = default)
        {
            if (!_config.EnableProgressiveLoading)
            {
                await RequestAssetAsync(assetId, priority, maxLodLevel, cancellationToken: cancellationToken);
                return;
            }

            for (int lod = 0; lod <= maxLodLevel; lod++)
            {
                string lodAssetId = $"{assetId}_lod{lod}";
                AssetPriority lodPriority = lod == 0
                    ? (AssetPriority)Math.Min((int)priority + 1, (int)AssetPriority.Immediate)
                    : priority;

                await RequestAssetAsync(lodAssetId, lodPriority, lod, cancellationToken: cancellationToken);
            }
        }

        /// <summary>
        /// Gets the current state of an asset.
        /// </summary>
        /// <param name="assetId">Asset identifier.</param>
        /// <returns>Asset entry if found, null otherwise.</returns>
        public AssetEntry? GetAssetState(string assetId)
        {
            return _assets.TryGetValue(assetId, out var entry) ? entry : null;
        }

        /// <summary>
        /// Gets a loaded asset, returning null if not loaded.
        /// Updates the LRU access time.
        /// </summary>
        /// <param name="assetId">Asset identifier.</param>
        /// <returns>The loaded NeuralAsset or null.</returns>
        public NeuralAsset? GetAsset(string assetId)
        {
            if (_assets.TryGetValue(assetId, out var entry) && entry.State == AssetLoadingState.Loaded && entry.Asset != null)
            {
                entry.LastAccessTime = DateTime.UtcNow;
                entry.ReferenceCount++;
                Interlocked.Increment(ref _statistics._totalCacheHits);
                return entry.Asset;
            }

            Interlocked.Increment(ref _statistics._totalCacheMisses);
            return null;
        }

        /// <summary>
        /// Releases a reference to an asset, allowing it to be evicted if no longer referenced.
        /// </summary>
        /// <param name="assetId">Asset identifier.</param>
        public void ReleaseAsset(string assetId)
        {
            if (_assets.TryGetValue(assetId, out var entry))
            {
                int newCount = Math.Max(0, entry.ReferenceCount - 1);
                entry.ReferenceCount = newCount;
                if (newCount <= 0)
                {
                    entry.ReferenceCount = 0;
                    entry.IsInUse = false;
                }
            }
        }

        /// <summary>
        /// Unloads a specific asset from memory.
        /// </summary>
        /// <param name="assetId">Asset identifier.</param>
        /// <returns>True if the asset was unloaded, false if not found or in use.</returns>
        public bool UnloadAsset(string assetId)
        {
            if (!_assets.TryGetValue(assetId, out var entry))
                return false;

            if (entry.ReferenceCount > 0)
                return false;

            if (entry.CancellationSource != null)
            {
                entry.CancellationSource.Cancel();
                entry.CancellationSource.Dispose();
            }

            entry.State = AssetLoadingState.Unloaded;
            entry.Asset = null;

            Interlocked.Add(ref _currentMemoryUsage, -entry.MemorySize);
            _statistics.CurrentMemoryUsageBytes = Interlocked.Read(ref _currentMemoryUsage);

            _cache.Remove(assetId);
            _assets.TryRemove(assetId, out _);

            return true;
        }

        /// <summary>
        /// Cancels loading of a specific asset.
        /// </summary>
        /// <param name="assetId">Asset identifier.</param>
        /// <returns>True if the cancellation was successful.</returns>
        public bool CancelLoading(string assetId)
        {
            if (!_assets.TryGetValue(assetId, out var entry))
                return false;

            if (entry.State is AssetLoadingState.Loaded or AssetLoadingState.Unloaded)
                return false;

            entry.CancellationSource?.Cancel();
            entry.State = AssetLoadingState.Failed;
            entry.LastError = "Cancelled by user";
            return true;
        }

        /// <summary>
        /// Cancels all pending loading operations.
        /// </summary>
        public void CancelAll()
        {
            foreach (var kvp in _assets)
            {
                if (kvp.Value.State is not (AssetLoadingState.Loaded or AssetLoadingState.Unloaded))
                {
                    kvp.Value.CancellationSource?.Cancel();
                    kvp.Value.State = AssetLoadingState.Failed;
                    kvp.Value.LastError = "All operations cancelled";
                }
            }
        }

        /// <summary>
        /// Gets a list of all asset IDs currently in the specified loading state.
        /// </summary>
        /// <param name="state">The loading state to filter by.</param>
        /// <returns>Collection of asset IDs in the specified state.</returns>
        public IReadOnlyCollection<string> GetAssetsByState(AssetLoadingState state)
        {
            return _assets
                .Where(kvp => kvp.Value.State == state)
                .Select(kvp => kvp.Key)
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Gets the total count of assets in each loading state.
        /// </summary>
        /// <returns>Dictionary mapping loading state to count.</returns>
        public IReadOnlyDictionary<AssetLoadingState, int> GetStateCounts()
        {
            return _assets.Values
                .GroupBy(e => e.State)
                .ToDictionary(g => g.Key, g => g.Count());
        }

        /// <summary>
        /// Forces eviction of unused assets to reclaim memory.
        /// </summary>
        /// <param name="targetBytes">Target number of bytes to free.</param>
        /// <returns>Number of bytes actually freed.</returns>
        public long ForceEviction(long targetBytes)
        {
            long freed = 0;
            var candidates = _assets.Values
                .Where(e => e.State == AssetLoadingState.Loaded
                    && e.ReferenceCount <= 0
                    && !e.IsInUse)
                .OrderBy(e => e.LastAccessTime)
                .ThenBy(e => e.Priority)
                .ToList();

            foreach (var candidate in candidates)
            {
                if (freed >= targetBytes)
                    break;

                if (UnloadAsset(candidate.AssetId))
                {
                    freed += candidate.MemorySize;
                    Interlocked.Increment(ref _statistics._totalEvictions);
                }
            }

            return freed;
        }

        /// <summary>
        /// Gets the current memory budget utilization information.
        /// </summary>
        /// <returns>Memory usage details.</returns>
        public MemoryBudgetInfo GetMemoryBudgetInfo()
        {
            long used = Interlocked.Read(ref _currentMemoryUsage);
            return new MemoryBudgetInfo
            {
                BudgetBytes = _config.MemoryBudgetBytes,
                UsedBytes = used,
                AvailableBytes = Math.Max(0, _config.MemoryBudgetBytes - used),
                UtilizationRatio = _config.MemoryBudgetBytes > 0
                    ? (double)used / _config.MemoryBudgetBytes
                    : 0.0,
                AssetCount = _assets.Count(e => e.Value.State == AssetLoadingState.Loaded),
                EvictionThresholdBytes = (long)(_config.MemoryBudgetBytes * _config.EvictionThreshold)
            };
        }

        /// <summary>
        /// Performs a warm-up by pre-loading critical assets.
        /// </summary>
        /// <param name="assetIds">Assets to preload.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>Task representing the async operation.</returns>
        public async Task WarmUpAsync(
            IEnumerable<string> assetIds,
            CancellationToken cancellationToken = default)
        {
            var tasks = assetIds.Select(id =>
                RequestAssetAsync(id, AssetPriority.Critical, cancellationToken: cancellationToken));
            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Main processing loop that dequeues and loads assets.
        /// </summary>
        private async Task ProcessLoadQueueAsync(CancellationToken cancellationToken)
        {
            var bandwidthWindow = new Queue<(DateTime time, long bytes)>();
            var statsTimer = Stopwatch.StartNew();
            double avgLoadTime = 0;
            int loadCount = 0;

            while (!cancellationToken.IsCancellationRequested)
            {
                string? assetId = null;
                lock (_loadQueue)
                {
                    if (_loadQueue.Count > 0)
                    {
                        _loadQueue.TryDequeue(out assetId, out _);
                    }
                }

                if (assetId == null)
                {
                    await Task.Delay(10, cancellationToken).ConfigureAwait(false);
                    continue;
                }

                if (!_assets.TryGetValue(assetId, out var entry))
                    continue;

                if (entry.State == AssetLoadingState.Loaded || entry.State == AssetLoadingState.Failed)
                    continue;

                if (entry.CancellationSource?.IsCancellationRequested == true)
                    continue;

                await _downloadSemaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
                Interlocked.Increment(ref _activeDownloads);

                int currentActive = _activeDownloads;
                if (currentActive > _statistics.PeakConcurrentDownloads)
                {
                    _statistics.PeakConcurrentDownloads = currentActive;
                }

                try
                {
                    await LoadAssetAsync(entry, cancellationToken).ConfigureAwait(false);

                    if (entry.State == AssetLoadingState.Loaded)
                    {
                        Interlocked.Increment(ref _statistics._totalAssetsLoaded);
                        Interlocked.Add(ref _currentMemoryUsage, entry.MemorySize);
                        _statistics.CurrentMemoryUsageBytes = Interlocked.Read(ref _currentMemoryUsage);

                        loadCount++;
                        if (loadCount > 0)
                        {
                            avgLoadTime = avgLoadTime + (entry.LoadTimeMs - avgLoadTime) / loadCount;
                            _statistics.AverageLoadTimeMs = avgLoadTime;
                        }
                    }
                    else if (entry.State == AssetLoadingState.Failed)
                    {
                        Interlocked.Increment(ref _statistics._totalAssetsFailed);

                        if (entry.RetryCount < _config.MaxRetryAttempts)
                        {
                            entry.RetryCount++;
                            entry.State = AssetLoadingState.Queued;
                            int delayMs = _config.RetryBaseDelayMs * (1 << (entry.RetryCount - 1));
                            await Task.Delay(delayMs, cancellationToken).ConfigureAwait(false);
                            lock (_loadQueue)
                            {
                                _loadQueue.Enqueue(assetId, (int)entry.Priority);
                            }
                        }
                    }
                }
                finally
                {
                    Interlocked.Decrement(ref _activeDownloads);
                    _downloadSemaphore.Release();
                }
            }
        }

        /// <summary>
        /// Loads a single asset through the full pipeline.
        /// </summary>
        private async Task LoadAssetAsync(AssetEntry entry, CancellationToken cancellationToken)
        {
            var loadStartTime = DateTime.UtcNow;
            entry.LoadingStartTime = loadStartTime;

            try
            {
                entry.State = AssetLoadingState.ResolvingDependencies;
                OnAssetLoadProgress(new AssetLoadProgressEventArgs
                {
                    AssetId = entry.AssetId,
                    State = entry.State,
                    ProgressPercent = 10
                });

                foreach (var depId in entry.Dependencies)
                {
                    if (cancellationToken.IsCancellationRequested) return;

                    if (_assets.TryGetValue(depId, out var depEntry))
                    {
                        int waitCount = 0;
                        while (depEntry.State != AssetLoadingState.Loaded
                            && depEntry.State != AssetLoadingState.Failed
                            && depEntry.State != AssetLoadingState.Unloaded)
                        {
                            await Task.Delay(50, cancellationToken).ConfigureAwait(false);
                            if (++waitCount > 600) break;
                        }

                        if (depEntry.State == AssetLoadingState.Failed)
                        {
                            entry.State = AssetLoadingState.Failed;
                            entry.LastError = $"Dependency {depId} failed to load";
                            return;
                        }
                    }
                }

                if (_config.EnableBandwidthThrottling && _config.MaxBandwidthBytesPerSecond > 0)
                {
                    await WaitForBandwidthAsync(cancellationToken).ConfigureAwait(false);
                }

                entry.State = AssetLoadingState.Downloading;
                OnAssetLoadProgress(new AssetLoadProgressEventArgs
                {
                    AssetId = entry.AssetId,
                    State = entry.State,
                    ProgressPercent = 30
                });

                byte[] compressedData = await DownloadAssetDataAsync(entry, cancellationToken)
                    .ConfigureAwait(false);

                if (compressedData == null || compressedData.Length == 0)
                {
                    entry.State = AssetLoadingState.Failed;
                    entry.LastError = "Download returned empty data";
                    return;
                }

                Interlocked.Add(ref _statistics._totalBytesLoaded, compressedData.Length);

                if (_config.EnableBandwidthThrottling && _config.MaxBandwidthBytesPerSecond > 0)
                {
                    _bandwidthTracker.RecordBytes(compressedData.Length);
                }

                entry.State = AssetLoadingState.Decompressing;
                OnAssetLoadProgress(new AssetLoadProgressEventArgs
                {
                    AssetId = entry.AssetId,
                    State = entry.State,
                    ProgressPercent = 50
                });

                if (cancellationToken.IsCancellationRequested) return;

                byte[] decompressed = await Task.Run(
                    () => _compression.Decompress(compressedData).ToArray(), cancellationToken)
                    .ConfigureAwait(false);

                Interlocked.Add(ref _statistics._totalBytesDecompressed, decompressed.Length);

                entry.State = AssetLoadingState.Decoding;
                OnAssetLoadProgress(new AssetLoadProgressEventArgs
                {
                    AssetId = entry.AssetId,
                    State = entry.State,
                    ProgressPercent = 70
                });

                if (cancellationToken.IsCancellationRequested) return;

                NeuralAsset asset = await Task.Run(
                    () => DecodeAsset(decompressed, entry), cancellationToken)
                    .ConfigureAwait(false);

                entry.State = AssetLoadingState.Uploading;
                OnAssetLoadProgress(new AssetLoadProgressEventArgs
                {
                    AssetId = entry.AssetId,
                    State = entry.State,
                    ProgressPercent = 90
                });

                if (cancellationToken.IsCancellationRequested) return;

                await UploadToGpuAsync(asset, cancellationToken).ConfigureAwait(false);

                entry.Asset = asset;
                entry.State = AssetLoadingState.Loaded;
                entry.LoadingEndTime = DateTime.UtcNow;
                entry.MemorySize = asset.ComputeMemoryFootprint();

                var validationErrors = asset.Validate();
                if (validationErrors.Count > 0)
                {
                    entry.LastError = $"Validation warnings: {string.Join("; ", validationErrors)}";
                }

                _cache.Put(entry.AssetId, asset);

                OnAssetLoadProgress(new AssetLoadProgressEventArgs
                {
                    AssetId = entry.AssetId,
                    State = AssetLoadingState.Loaded,
                    ProgressPercent = 100,
                    BytesDownloaded = compressedData.Length,
                    TotalBytes = decompressed.Length
                });
            }
            catch (OperationCanceledException)
            {
                if (entry.State != AssetLoadingState.Failed)
                {
                    entry.State = AssetLoadingState.Failed;
                    entry.LastError = "Operation cancelled";
                }
            }
            catch (Exception ex)
            {
                entry.State = AssetLoadingState.Failed;
                entry.LastError = ex.Message;
            }
        }

        /// <summary>
        /// Loads asset bytes from the local project/assets directory, or synthesizes a placeholder MLP.
        /// Set <see cref="AssetRootDirectory"/> to enable real file I/O for <c>{assetId}.gnn</c> / <c>.bin</c>.
        /// </summary>
        public static string? AssetRootDirectory { get; set; }

        private const long MaxAssetFileBytes = 64L * 1024 * 1024;

        private async Task<byte[]> DownloadAssetDataAsync(
            AssetEntry entry,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (!string.IsNullOrWhiteSpace(AssetRootDirectory))
            {
                var safeId = Synapse.Core.Security.PathSecurity.RequireSafeAssetId(entry.AssetId);
                var root = Synapse.Core.Security.PathSecurity.GetFullPathChecked(AssetRootDirectory);

                foreach (var ext in new[] { ".gnn", ".bin", ".neural" })
                {
                    var path = Synapse.Core.Security.PathSecurity.CombineUnderRoot(root, safeId + ext);
                    if (File.Exists(path))
                        return await ReadAssetFileAsync(path, cancellationToken).ConfigureAwait(false);
                }

                var direct = Synapse.Core.Security.PathSecurity.CombineUnderRoot(root, safeId);
                if (File.Exists(direct))
                    return await ReadAssetFileAsync(direct, cancellationToken).ConfigureAwait(false);
            }

            // Fallback: generate a local placeholder asset so streaming never blocks the product.
            await Task.Yield();
            var asset = new NeuralAsset { TargetMeshId = Guid.Empty };
            var mlp = new MicroMLP();
            asset.CompressedWeights = mlp.CompressWeights();
            return asset.Serialize();
        }

        private static async Task<byte[]> ReadAssetFileAsync(string path, CancellationToken cancellationToken)
        {
            var info = new FileInfo(path);
            if (info.Length > MaxAssetFileBytes)
                throw new InvalidDataException($"Asset file exceeds {MaxAssetFileBytes} byte limit.");
            return await File.ReadAllBytesAsync(path, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Decodes compressed data into a NeuralAsset.
        /// </summary>
        private NeuralAsset DecodeAsset(ReadOnlySpan<byte> data, AssetEntry entry)
        {
            using var ms = new MemoryStream(data.ToArray());
            return NeuralAsset.DeserializeAsync(ms).GetAwaiter().GetResult();
        }

        /// <summary>
        /// Simulates uploading asset data to GPU memory.
        /// </summary>
        private async Task UploadToGpuAsync(NeuralAsset asset, CancellationToken cancellationToken)
        {
            await Task.Delay(1, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Waits for bandwidth to become available within the configured limit.
        /// </summary>
        private async Task WaitForBandwidthAsync(CancellationToken cancellationToken)
        {
            if (_config.MaxBandwidthBytesPerSecond <= 0) return;

            double currentBps = _bandwidthTracker.GetCurrentBandwidth();
            while (currentBps > _config.MaxBandwidthBytesPerSecond)
            {
                await Task.Delay(10, cancellationToken).ConfigureAwait(false);
                currentBps = _bandwidthTracker.GetCurrentBandwidth();
            }
        }

        /// <summary>
        /// Background task for reporting statistics at regular intervals.
        /// </summary>
        private async Task ReportStatsAsync(CancellationToken cancellationToken)
        {
            var interval = TimeSpan.FromSeconds(_config.StatsReportingIntervalSeconds);
            while (!cancellationToken.IsCancellationRequested)
            {
                await Task.Delay(interval, cancellationToken).ConfigureAwait(false);

                _statistics.CurrentMemoryUsageBytes = Interlocked.Read(ref _currentMemoryUsage);
                _statistics.CachedAssetCount = _cache.Count;
                _statistics.CurrentBandwidthBytesPerSecond = _bandwidthTracker.GetCurrentBandwidth();

                var snapshot = _statistics.CreateSnapshot();
                StatisticsUpdated?.Invoke(this, snapshot);
            }
        }

        /// <summary>
        /// Background task for periodic LRU eviction of unused assets.
        /// </summary>
        private async Task RunEvictionAsync(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                await Task.Delay(1000, cancellationToken).ConfigureAwait(false);

                long currentMem = Interlocked.Read(ref _currentMemoryUsage);
                double utilization = _config.MemoryBudgetBytes > 0
                    ? (double)currentMem / _config.MemoryBudgetBytes
                    : 0;

                if (utilization < _config.EvictionThreshold)
                    continue;

                long targetFree = currentMem - (long)(_config.MemoryBudgetBytes * 0.7);
                if (targetFree <= 0) continue;

                var expiredCandidates = _assets.Values
                    .Where(e => e.State == AssetLoadingState.Loaded
                        && e.ReferenceCount <= 0
                        && !e.IsInUse
                        && (DateTime.UtcNow - e.LastAccessTime).TotalSeconds > _config.UnusedAssetTtlSeconds)
                    .OrderBy(e => e.LastAccessTime)
                    .ToList();

                long freed = 0;
                foreach (var candidate in expiredCandidates)
                {
                    if (freed >= targetFree) break;

                    if (UnloadAsset(candidate.AssetId))
                    {
                        freed += candidate.MemorySize;
                        Interlocked.Increment(ref _statistics._totalEvictions);
                    }
                }

                if (freed == 0)
                {
                    ForceEviction(targetFree);
                }
            }
        }

        /// <summary>
        /// Raises the AssetLoadProgress event.
        /// </summary>
        private void OnAssetLoadProgress(AssetLoadProgressEventArgs e)
        {
            AssetLoadProgress?.Invoke(this, e);
        }

        /// <summary>
        /// Disposes the asset streamer and releases all resources.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _shutdownCts.Cancel();
            _processingTask.Wait(TimeSpan.FromSeconds(5));
            _statsTask.Wait(TimeSpan.FromSeconds(2));
            _evictionTask.Wait(TimeSpan.FromSeconds(2));

            foreach (var entry in _assets.Values)
            {
                entry.CancellationSource?.Dispose();
            }

            _downloadSemaphore.Dispose();
            _bandwidthSemaphore.Dispose();
            _shutdownCts.Dispose();
            _cache.Dispose();
            _memoryPool.Dispose();
            _pipeline.Dispose();
        }
    }

    /// <summary>
    /// Tracks bandwidth usage over a sliding time window.
    /// </summary>
    internal sealed class BandwidthTracker
    {
        private readonly ConcurrentQueue<(DateTime time, long bytes)> _samples = new();
        private readonly TimeSpan _windowSize = TimeSpan.FromSeconds(1);
        private long _totalBytesInWindow;
        private DateTime _lastCleanup = DateTime.UtcNow;

        /// <summary>
        /// Records a number of bytes transferred.
        /// </summary>
        /// <param name="byteCount">Number of bytes transferred.</param>
        public void RecordBytes(long byteCount)
        {
            var now = DateTime.UtcNow;
            _samples.Enqueue((now, Interlocked.Add(ref _totalBytesInWindow, byteCount)));

            if (now - _lastCleanup > TimeSpan.FromMilliseconds(100))
            {
                CleanupOldSamples(now);
                _lastCleanup = now;
            }
        }

        /// <summary>
        /// Gets the current bandwidth in bytes per second over the sliding window.
        /// </summary>
        /// <returns>Bytes per second.</returns>
        public double GetCurrentBandwidth()
        {
            var now = DateTime.UtcNow;
            CleanupOldSamples(now);

            var cutoff = now - _windowSize;
            long bytesInWindow = 0;
            foreach (var sample in _samples)
            {
                if (sample.time >= cutoff)
                    bytesInWindow += sample.bytes;
            }

            return bytesInWindow / _windowSize.TotalSeconds;
        }

        /// <summary>
        /// Removes samples outside the sliding window.
        /// </summary>
        private void CleanupOldSamples(DateTime now)
        {
            var cutoff = now - _windowSize;
            while (_samples.TryPeek(out var sample) && sample.time < cutoff)
            {
                _samples.TryDequeue(out _);
            }
        }
    }

    /// <summary>
    /// Information about the current memory budget status.
    /// </summary>
    public sealed class MemoryBudgetInfo
    {
        /// <summary>Total memory budget in bytes.</summary>
        public long BudgetBytes { get; init; }

        /// <summary>Currently used memory in bytes.</summary>
        public long UsedBytes { get; init; }

        /// <summary>Available memory in bytes.</summary>
        public long AvailableBytes { get; init; }

        /// <summary>Memory utilization as a ratio (0.0-1.0).</summary>
        public double UtilizationRatio { get; init; }

        /// <summary>Number of assets currently loaded.</summary>
        public int AssetCount { get; init; }

        /// <summary>Memory usage threshold that triggers eviction in bytes.</summary>
        public long EvictionThresholdBytes { get; init; }

        /// <summary>Returns a formatted string representation.</summary>
        public override string ToString() =>
            $"Memory: {UsedBytes / (1024.0 * 1024.0):F1} / {BudgetBytes / (1024.0 * 1024.0):F1} MB " +
            $"({UtilizationRatio:P1}) | Assets: {AssetCount}";
    }

    /// <summary>
    /// Provides slab-based memory pool allocation for streaming buffers.
    /// </summary>
    public sealed class MemoryPool : IDisposable
    {
        private readonly int[] _slabSizeClasses;
        private bool _disposed;

        public MemoryPool(int[] slabSizeClasses)
        {
            _slabSizeClasses = slabSizeClasses ?? [4096];
        }

        public void Dispose()
        {
            _disposed = true;
        }
    }
}
