// =============================================================================
// NeatGEvolutionEngine.cs - NEAT-G Evolution Engine Core
// GDNN.Engine - Geometric Deep Neural Network Engine
// Copyright (c) 2024. All rights reserved.
// =============================================================================
// This file is the heart of the G-DNN Engine implementing the NEAT-G
// (NeuroEvolution of Augmented Topologies - Geometric) algorithm.
// It provides comprehensive evolutionary optimization for neural network
// architectures with geometric awareness, semantic crossover, manifold-based
// speciation, and swarm evolution capabilities.
// =============================================================================

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using GDNN.Core.NEAT.Models;
using Synapse.Infrastructure.Logging;

namespace GDNN.Core.NEAT
{

    /// <summary>
    /// LRU (Least Recently Used) cache for genome fitness evaluations.
    /// Prevents redundant evaluations by caching previously computed fitness values.
    /// Thread-safe for concurrent access during parallel evaluation.
    /// </summary>
    public sealed class GenomeFitnessCache
    {
        private readonly ConcurrentDictionary<string, CacheEntry> _cache;
        private readonly int _maxSize;
        private readonly object _evictionLock = new();
        private long _hits;
        private long _misses;

        /// <summary>
        /// Initializes a new instance of the GenomeFitnessCache class.
        /// </summary>
        /// <param name="maxSize">Maximum number of cache entries.</param>
        public GenomeFitnessCache(int maxSize = 10000)
        {
            _maxSize = maxSize;
            _cache = new ConcurrentDictionary<string, CacheEntry>();
        }

        /// <summary>Cache hit count.</summary>
        public long Hits => Interlocked.Read(ref _hits);

        /// <summary>Cache miss count.</summary>
        public long Misses => Interlocked.Read(ref _misses);

        /// <summary>Cache hit rate.</summary>
        public double HitRate
        {
            get
            {
                long total = Hits + Misses;
                return total > 0 ? (double)Hits / total : 0;
            }
        }

        /// <summary>Current cache size.</summary>
        public int Size => _cache.Count;

        /// <summary>
        /// Tries to get a cached fitness value for a genome.
        /// </summary>
        /// <param name="genome">The genome to look up.</param>
        /// <param name="fitness">The cached fitness value, if found.</param>
        /// <returns>True if found in cache.</returns>
        public bool TryGetFitness(GeoGenome genome, out double fitness)
        {
            string key = ComputeCacheKey(genome);
            if (_cache.TryGetValue(key, out var entry))
            {
                entry.LastAccessTime = DateTime.UtcNow;
                Interlocked.Increment(ref _hits);
                fitness = entry.Fitness;
                return true;
            }

            Interlocked.Increment(ref _misses);
            fitness = 0;
            return false;
        }

        /// <summary>
        /// Adds a fitness value to the cache.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="fitness">The fitness value.</param>
        public void AddFitness(GeoGenome genome, double fitness)
        {
            string key = ComputeCacheKey(genome);
            var entry = new CacheEntry
            {
                Fitness = fitness,
                LastAccessTime = DateTime.UtcNow,
                InsertionTime = DateTime.UtcNow
            };

            _cache[key] = entry;

            if (_cache.Count > _maxSize)
            {
                EvictOldest();
            }
        }

        /// <summary>
        /// Removes a genome from the cache.
        /// </summary>
        /// <param name="genome">The genome to remove.</param>
        /// <returns>True if removed.</returns>
        public bool Remove(GeoGenome genome)
        {
            string key = ComputeCacheKey(genome);
            return _cache.TryRemove(key, out _);
        }

        /// <summary>
        /// Clears the entire cache.
        /// </summary>
        public void Clear()
        {
            _cache.Clear();
        }

        /// <summary>
        /// Gets cache statistics.
        /// </summary>
        public CacheStatistics GetStatistics()
        {
            return new CacheStatistics
            {
                Size = Size,
                MaxSize = _maxSize,
                Hits = Hits,
                Misses = Misses,
                HitRate = HitRate,
                OldestEntry = _cache.Values
                    .OrderBy(e => e.InsertionTime)
                    .FirstOrDefault()?.InsertionTime ?? DateTime.MinValue,
                NewestEntry = _cache.Values
                    .OrderByDescending(e => e.InsertionTime)
                    .FirstOrDefault()?.InsertionTime ?? DateTime.MinValue
            };
        }

        private string ComputeCacheKey(GeoGenome genome)
        {
            long topologyHash = genome.ComputeTopologyHash();
            long weightHash = 0;
            foreach (var synapse in genome.Synapses.Where(s => s.IsActive).OrderBy(s => s.InnovationNumber))
            {
                weightHash = HashCode.Combine(weightHash, synapse.Weight.GetHashCode());
            }
            return $"{topologyHash}_{weightHash}";
        }

        private void EvictOldest()
        {
            lock (_evictionLock)
            {
                if (_cache.Count <= _maxSize)
                    return;

                var toRemove = _cache
                    .OrderBy(kvp => kvp.Value.LastAccessTime)
                    .Take(_cache.Count - _maxSize + 100)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var key in toRemove)
                {
                    _cache.TryRemove(key, out _);
                }
            }
        }

        private sealed class CacheEntry
        {
            public double Fitness { get; set; }
            public DateTime LastAccessTime { get; set; }
            public DateTime InsertionTime { get; set; }
        }
    }

    /// <summary>
    /// Cache statistics.
    /// </summary>
    public record CacheStatistics
    {
        /// <summary>Current cache size.</summary>
        public int Size { get; init; }

        /// <summary>Maximum cache size.</summary>
        public int MaxSize { get; init; }

        /// <summary>Total cache hits.</summary>
        public long Hits { get; init; }

        /// <summary>Total cache misses.</summary>
        public long Misses { get; init; }

        /// <summary>Cache hit rate.</summary>
        public double HitRate { get; init; }

        /// <summary>Timestamp of oldest entry.</summary>
        public DateTime OldestEntry { get; init; }

        /// <summary>Timestamp of newest entry.</summary>
        public DateTime NewestEntry { get; init; }
    }

}
