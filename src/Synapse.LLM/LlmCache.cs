// Multi-provider LLM pipeline for Synapse (split from HybridLlmRouter.cs).

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using System.Timers;
using GDNN.Scene;
using Synapse.Infrastructure.Logging;

#nullable enable

namespace GDNN.Llm
{

    /// <summary>
    /// Multi-tier LLM response cache supporting exact-match and semantic similarity
    /// lookups, TTL expiration, LRU eviction, persistent storage, and compression.
    /// </summary>
    public sealed class LlmCache : IDisposable
    {
        private readonly ConcurrentDictionary<string, CacheEntry> _exactCache;
        private readonly List<CacheEntry> _semanticIndex;
        private readonly int _maxEntries;
        private readonly TimeSpan _defaultTtl;
        private readonly float _semanticThreshold;
        private readonly string? _persistentPath;
        private readonly object _semanticLock = new();
        private int _hitCount;
        private int _missCount;
        private int _evictionCount;
        private bool _disposed;

        /// <summary>Total cache hits.</summary>
        public int HitCount => _hitCount;

        /// <summary>Total cache misses.</summary>
        public int MissCount => _missCount;

        /// <summary>Total evictions.</summary>
        public int EvictionCount => _evictionCount;

        /// <summary>Cache hit rate as a percentage.</summary>
        public float HitRate =>
            (_hitCount + _missCount) > 0
                ? (float)_hitCount / (_hitCount + _missCount) * 100f
                : 0f;

        /// <summary>Current number of entries.</summary>
        public int Count => _exactCache.Count;

        /// <summary>
        /// Initializes a new LLM cache.
        /// </summary>
        /// <param name="maxEntries">Maximum cache entries before LRU eviction.</param>
        /// <param name="defaultTtl">Default time-to-live for entries.</param>
        /// <param name="semanticThreshold">Cosine similarity threshold for semantic cache hits (0-1).</param>
        /// <param name="persistentPath">Optional path for persistent cache storage.</param>
        public LlmCache(
            int maxEntries = 10000,
            TimeSpan? defaultTtl = null,
            float semanticThreshold = 0.95f,
            string? persistentPath = null)
        {
            _maxEntries = maxEntries;
            _defaultTtl = defaultTtl ?? TimeSpan.FromHours(1);
            _semanticThreshold = semanticThreshold;
            _persistentPath = persistentPath;
            _exactCache = new ConcurrentDictionary<string, CacheEntry>(StringComparer.OrdinalIgnoreCase);
            _semanticIndex = new List<CacheEntry>();

            if (!string.IsNullOrEmpty(persistentPath) && Directory.Exists(persistentPath))
            {
                LoadFromDisk();
            }
        }

        /// <summary>
        /// Stores a response in the cache.
        /// </summary>
        /// <param name="prompt">The original prompt.</param>
        /// <param name="response">The LLM response.</param>
        /// <param name="ttl">Optional custom TTL.</param>
        /// <param name="embedding">Optional embedding for semantic lookup.</param>
        public void Store(
            string prompt,
            LlmResponse response,
            TimeSpan? ttl = null,
            EmbeddingVector? embedding = null)
        {
            if (_disposed)
                return;

            var hash = ComputeHash(prompt);
            var entry = new CacheEntry
            {
                Response = response,
                PromptHash = hash,
                CreatedAt = DateTimeOffset.UtcNow,
                ExpiresAt = DateTimeOffset.UtcNow + (ttl ?? _defaultTtl),
                AccessCount = 0,
                SizeBytes = Encoding.UTF8.GetByteCount(response.Content),
                SemanticEmbedding = embedding
            };

            _exactCache[hash] = entry;

            if (embedding != null)
            {
                lock (_semanticLock)
                {
                    _semanticIndex.Add(entry);
                }
            }

            EvictIfNeeded();
        }

        /// <summary>
        /// Retrieves a cached response using exact match.
        /// </summary>
        /// <param name="prompt">The prompt to look up.</param>
        /// <returns>The cached response, or null if not found/expired.</returns>
        public LlmResponse? GetExact(string prompt)
        {
            if (_disposed || string.IsNullOrEmpty(prompt))
                return null;

            var hash = ComputeHash(prompt);
            if (!_exactCache.TryGetValue(hash, out var entry))
                return null;

            if (entry.IsInvalidated || DateTimeOffset.UtcNow > entry.ExpiresAt)
            {
                _exactCache.TryRemove(hash, out _);
                Interlocked.Increment(ref _missCount);
                return null;
            }

            Interlocked.Increment(ref _hitCount);
            var updated = entry with { AccessCount = entry.AccessCount + 1, LastAccessedAt = DateTimeOffset.UtcNow };
            _exactCache[hash] = updated;
            return entry.Response;
        }

        /// <summary>
        /// Retrieves a cached response using semantic similarity.
        /// </summary>
        /// <param name="queryEmbedding">Embedding of the query.</param>
        /// <returns>The best matching cached response, or null.</returns>
        public LlmResponse? GetSemantic(EmbeddingVector queryEmbedding)
        {
            if (_disposed || queryEmbedding.Values.Length == 0)
                return null;

            CacheEntry? bestMatch = null;
            float bestScore = 0f;

            lock (_semanticLock)
            {
                foreach (var entry in _semanticIndex)
                {
                    if (entry.IsInvalidated || DateTimeOffset.UtcNow > entry.ExpiresAt)
                        continue;
                    if (entry.SemanticEmbedding == null)
                        continue;

                    var score = ComputeCosineSimilarity(
                        queryEmbedding.Values, entry.SemanticEmbedding.Values);

                    if (score > bestScore && score >= _semanticThreshold)
                    {
                        bestScore = score;
                        bestMatch = entry;
                    }
                }
            }

            if (bestMatch != null)
            {
                Interlocked.Increment(ref _hitCount);
                return bestMatch.Response;
            }

            Interlocked.Increment(ref _missCount);
            return null;
        }

        /// <summary>
        /// Invalidates all cache entries for a specific model.
        /// </summary>
        /// <param name="model">Model name to invalidate.</param>
        public void InvalidateByModel(string model)
        {
            if (string.IsNullOrEmpty(model))
                return;

            var keysToInvalidate = _exactCache
                .Where(kv => kv.Value.Response.Model == model)
                .Select(kv => kv.Key)
                .ToList();

            foreach (var key in keysToInvalidate)
            {
                if (_exactCache.TryGetValue(key, out var entry))
                {
                    _exactCache[key] = entry with { IsInvalidated = true };
                }
            }
        }

        /// <summary>
        /// Clears all cache entries.
        /// </summary>
        public void Clear()
        {
            _exactCache.Clear();
            lock (_semanticLock)
            { _semanticIndex.Clear(); }
        }

        /// <summary>
        /// Gets cache statistics.
        /// </summary>
        public CacheStatistics GetStatistics()
        {
            return new CacheStatistics
            {
                TotalEntries = _exactCache.Count,
                HitCount = _hitCount,
                MissCount = _missCount,
                EvictionCount = _evictionCount,
                HitRate = HitRate,
                TotalSizeBytes = _exactCache.Values.Sum(e => e.SizeBytes),
                SemanticIndexSize = _semanticIndex.Count
            };
        }

        /// <summary>
        /// Persists the cache to disk.
        /// </summary>
        public void PersistToDisk()
        {
            if (string.IsNullOrEmpty(_persistentPath))
                return;
            if (!Directory.Exists(_persistentPath))
                Directory.CreateDirectory(_persistentPath);

            var filePath = Path.Combine(_persistentPath, "llm_cache.json");
            var entries = _exactCache.Values
                .Where(e => !e.IsInvalidated && DateTimeOffset.UtcNow <= e.ExpiresAt)
                .ToList();

            var json = JsonSerializer.Serialize(entries, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Computes cosine similarity between two vectors.
        /// </summary>
        public static float ComputeCosineSimilarity(float[] a, float[] b)
        {
            if (a == null || b == null || a.Length == 0 || a.Length != b.Length)
                return 0f;

            float dot = 0f, normA = 0f, normB = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            float denominator = MathF.Sqrt(normA) * MathF.Sqrt(normB);
            return denominator > 1e-10f ? dot / denominator : 0f;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            PersistToDisk();
        }

        private static string ComputeHash(string prompt)
        {
            using var sha = SHA256.Create();
            var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(prompt));
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }

        private void EvictIfNeeded()
        {
            while (_exactCache.Count > _maxEntries)
            {
                var oldest = _exactCache.Values
                    .OrderBy(e => e.LastAccessedAt)
                    .FirstOrDefault();

                if (oldest != null)
                {
                    _exactCache.TryRemove(oldest.PromptHash, out _);
                    Interlocked.Increment(ref _evictionCount);
                }
                else
                    break;
            }
        }

        private void LoadFromDisk()
        {
            try
            {
                var filePath = Path.Combine(_persistentPath!, "llm_cache.json");
                if (!File.Exists(filePath))
                    return;

                var json = File.ReadAllText(filePath);
                var entries = JsonSerializer.Deserialize<List<CacheEntry>>(json);
                if (entries != null)
                {
                    foreach (var entry in entries)
                    {
                        if (!entry.IsInvalidated && DateTimeOffset.UtcNow <= entry.ExpiresAt)
                            _exactCache[entry.PromptHash] = entry;
                    }
                }
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("LlmCache", "Best-effort cache load failed.", ex);
            }
        }
    }

    /// <summary>
    /// Cache statistics.
    /// </summary>
    public record CacheStatistics
    {
        /// <summary>Total entries in cache.</summary>
        public int TotalEntries { get; init; }

        /// <summary>Total cache hits.</summary>
        public int HitCount { get; init; }

        /// <summary>Total cache misses.</summary>
        public int MissCount { get; init; }

        /// <summary>Total evictions.</summary>
        public int EvictionCount { get; init; }

        /// <summary>Hit rate percentage.</summary>
        public float HitRate { get; init; }

        /// <summary>Total size in bytes.</summary>
        public long TotalSizeBytes { get; init; }

        /// <summary>Semantic index size.</summary>
        public int SemanticIndexSize { get; init; }
    }
}
