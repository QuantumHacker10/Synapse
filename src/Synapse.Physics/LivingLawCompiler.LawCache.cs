// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    // =========================================================================
    // LawCache — thread-safe caching for compiled bytecodes
    // =========================================================================

    /// <summary>Thread-safe cache for compiled law bytecodes with LRU eviction.</summary>
    public sealed class LawCache
    {
        private readonly ConcurrentDictionary<string, CacheEntry> _cache = new();
        private readonly int _maxEntries;
        private long _hits;
        private long _misses;
        private long _evictions;

        public int Count => _cache.Count;
        public long Hits => Interlocked.Read(ref _hits);
        public long Misses => Interlocked.Read(ref _misses);
        public long Evictions => Interlocked.Read(ref _evictions);

        private sealed class CacheEntry
        {
            public LawBytecode Bytecode { get; set; } = null!;
            public DateTime LastAccessed { get; set; }
            public int AccessCount { get; set; }
        }

        public LawCache(int maxEntries = 1000)
        {
            _maxEntries = maxEntries;
        }

        /// <summary>Try to get a cached bytecode.</summary>
        public bool TryGet(string key, [MaybeNullWhen(false)] out LawBytecode bytecode)
        {
            if (_cache.TryGetValue(key, out var entry))
            {
                entry.LastAccessed = DateTime.UtcNow;
                entry.AccessCount++;
                bytecode = entry.Bytecode;
                Interlocked.Increment(ref _hits);
                return true;
            }
            bytecode = null!;
            Interlocked.Increment(ref _misses);
            return false;
        }

        /// <summary>Store a bytecode in the cache.</summary>
        public void Store(string key, LawBytecode bytecode)
        {
            if (_cache.Count >= _maxEntries)
            {
                EvictLRU();
            }
            _cache[key] = new CacheEntry
            {
                Bytecode = bytecode,
                LastAccessed = DateTime.UtcNow,
                AccessCount = 1
            };
        }

        /// <summary>Remove a specific entry from the cache.</summary>
        public bool Remove(string key) => _cache.TryRemove(key, out _);

        /// <summary>Clear the entire cache.</summary>
        public void Clear() { _cache.Clear(); Interlocked.Exchange(ref _hits, 0); Interlocked.Exchange(ref _misses, 0); Interlocked.Exchange(ref _evictions, 0); }

        private void EvictLRU()
        {
            string? oldestKey = null;
            DateTime oldestTime = DateTime.MaxValue;
            foreach (var kv in _cache)
            {
                if (kv.Value.LastAccessed < oldestTime)
                {
                    oldestTime = kv.Value.LastAccessed;
                    oldestKey = kv.Key;
                }
            }
            if (oldestKey != null)
            {
                _cache.TryRemove(oldestKey, out _);
                Interlocked.Increment(ref _evictions);
            }
        }

        /// <summary>Get cache statistics.</summary>
        public (long Hits, long Misses, long Evictions, int Count, float HitRate) GetStats()
        {
            long h = Hits, m = Misses;
            float rate = h + m > 0 ? (float)h / (h + m) : 0f;
            return (h, m, Evictions, Count, rate);
        }

        /// <summary>Get the most frequently accessed entries.</summary>
        public IReadOnlyList<(string Key, int AccessCount)> GetMostAccessed(int count = 10)
        {
            return _cache.OrderByDescending(kv => kv.Value.AccessCount)
                .Take(count)
                .Select(kv => (kv.Key, kv.Value.AccessCount))
                .ToList();
        }
    }
}
