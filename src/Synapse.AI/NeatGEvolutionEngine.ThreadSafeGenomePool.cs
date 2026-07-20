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
    /// Thread-safe genome pool for concurrent access during parallel evolution.
    /// Uses lock-free data structures for high-performance concurrent operations.
    /// </summary>
    public sealed class ConcurrentGenomePool
    {
        private readonly ConcurrentDictionary<Guid, GeoGenome> _pool;
        private readonly ConcurrentQueue<Guid> _availableIds;
        private readonly ConcurrentBag<Guid> _recycledIds;
        private long _totalCount;
        private long _activeCount;

        /// <summary>
        /// Initializes a new instance of the ConcurrentGenomePool class.
        /// </summary>
        public ConcurrentGenomePool()
        {
            _pool = new ConcurrentDictionary<Guid, GeoGenome>();
            _availableIds = new ConcurrentQueue<Guid>();
            _recycledIds = new ConcurrentBag<Guid>();
        }

        /// <summary>Total genomes ever added to the pool.</summary>
        public long TotalCount => Interlocked.Read(ref _totalCount);

        /// <summary>Currently active genomes in the pool.</summary>
        public long ActiveCount => Interlocked.Read(ref _activeCount);

        /// <summary>Number of available (unclaimed) genomes.</summary>
        public int AvailableCount => _availableIds.Count;

        /// <summary>
        /// Adds a genome to the pool.
        /// </summary>
        /// <param name="genome">The genome to add.</param>
        public void Add(GeoGenome genome)
        {
            if (_pool.TryAdd(genome.Id, genome))
            {
                _availableIds.Enqueue(genome.Id);
                Interlocked.Increment(ref _totalCount);
                Interlocked.Increment(ref _activeCount);
            }
        }

        /// <summary>
        /// Adds multiple genomes to the pool.
        /// </summary>
        /// <param name="genomes">Genomes to add.</param>
        public void AddRange(IEnumerable<GeoGenome> genomes)
        {
            foreach (var genome in genomes)
            {
                Add(genome);
            }
        }

        /// <summary>
        /// Tries to claim a genome from the pool.
        /// </summary>
        /// <param name="genome">The claimed genome, or default if none available.</param>
        /// <returns>True if a genome was successfully claimed.</returns>
        public bool TryClaim(out GeoGenome? genome)
        {
            genome = null;

            while (_availableIds.TryDequeue(out var id))
            {
                if (_pool.TryGetValue(id, out var g))
                {
                    genome = g;
                    Interlocked.Decrement(ref _activeCount);
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Claims a specific genome by ID.
        /// </summary>
        /// <param name="id">Genome ID to claim.</param>
        /// <returns>The claimed genome, or null if not found.</returns>
        public GeoGenome? ClaimById(Guid id)
        {
            if (_pool.TryRemove(id, out var genome))
            {
                Interlocked.Decrement(ref _activeCount);
                return genome;
            }
            return null;
        }

        /// <summary>
        /// Returns a genome to the pool for reuse.
        /// </summary>
        /// <param name="genome">The genome to return.</param>
        public void Return(GeoGenome genome)
        {
            genome.InvalidateFitness();
            if (_pool.TryAdd(genome.Id, genome))
            {
                _availableIds.Enqueue(genome.Id);
                Interlocked.Increment(ref _activeCount);
            }
        }

        /// <summary>
        /// Removes a genome from the pool permanently.
        /// </summary>
        /// <param name="id">Genome ID to remove.</param>
        /// <returns>True if the genome was removed.</returns>
        public bool Remove(Guid id)
        {
            if (_pool.TryRemove(id, out _))
            {
                Interlocked.Decrement(ref _activeCount);
                _recycledIds.Add(id);
                return true;
            }
            return false;
        }

        /// <summary>
        /// Gets a genome by ID without claiming it.
        /// </summary>
        /// <param name="id">Genome ID.</param>
        /// <returns>The genome, or null if not found.</returns>
        public GeoGenome? Peek(Guid id)
        {
            return _pool.TryGetValue(id, out var genome) ? genome : null;
        }

        /// <summary>
        /// Gets all genomes in the pool.
        /// </summary>
        public IReadOnlyList<GeoGenome> GetAll()
        {
            return _pool.Values.ToList().AsReadOnly();
        }

        /// <summary>
        /// Clears the pool.
        /// </summary>
        public void Clear()
        {
            _pool.Clear();
            while (_availableIds.TryDequeue(out _))
            { }
            Interlocked.Exchange(ref _activeCount, 0);
        }

        /// <summary>
        /// Gets the top N genomes by fitness.
        /// </summary>
        /// <param name="count">Number of top genomes to retrieve.</param>
        public IReadOnlyList<GeoGenome> GetTop(int count)
        {
            return _pool.Values
                .OrderByDescending(g => g.Fitness)
                .Take(count)
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Gets genomes matching a predicate.
        /// </summary>
        /// <param name="predicate">Filter predicate.</param>
        public IReadOnlyList<GeoGenome> Where(Func<GeoGenome, bool> predicate)
        {
            return _pool.Values.Where(predicate).ToList().AsReadOnly();
        }

        /// <summary>
        /// Performs a bulk operation on all genomes in the pool.
        /// </summary>
        /// <param name="operation">Operation to perform on each genome.</param>
        public void ForEach(Action<GeoGenome> operation)
        {
            foreach (var genome in _pool.Values)
            {
                operation(genome);
            }
        }

        /// <summary>
        /// Performs a parallel bulk operation on all genomes in the pool.
        /// </summary>
        /// <param name="operation">Operation to perform on each genome.</param>
        /// <param name="maxDegreeOfParallelism">Maximum parallel threads.</param>
        public void ParallelForEach(Action<GeoGenome> operation, int maxDegreeOfParallelism = -1)
        {
            var options = new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism
            };
            Parallel.ForEach(_pool.Values, options, operation);
        }

        /// <summary>
        /// Computes aggregate statistics for all genomes in the pool.
        /// </summary>
        public PoolStatistics GetStatistics()
        {
            var genomes = _pool.Values.ToList();
            if (genomes.Count == 0)
                return new PoolStatistics();

            var fitnesses = genomes.Select(g => g.Fitness).ToArray();
            return new PoolStatistics
            {
                Count = genomes.Count,
                MeanFitness = fitnesses.Average(),
                BestFitness = fitnesses.Max(),
                WorstFitness = fitnesses.Min(),
                AvgNeuronCount = genomes.Average(g => g.ActiveNeuronCount),
                AvgSynapseCount = genomes.Average(g => g.ActiveSynapseCount),
                AvgComplexity = genomes.Average(g => g.Complexity),
                UniqueTopologies = genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count()
            };
        }
    }

    /// <summary>
    /// Statistics for a genome pool.
    /// </summary>
    public record PoolStatistics
    {
        /// <summary>Number of genomes in the pool.</summary>
        public int Count { get; init; }

        /// <summary>Mean fitness.</summary>
        public double MeanFitness { get; init; }

        /// <summary>Best fitness.</summary>
        public double BestFitness { get; init; }

        /// <summary>Worst fitness.</summary>
        public double WorstFitness { get; init; }

        /// <summary>Average neuron count.</summary>
        public double AvgNeuronCount { get; init; }

        /// <summary>Average synapse count.</summary>
        public double AvgSynapseCount { get; init; }

        /// <summary>Average complexity.</summary>
        public double AvgComplexity { get; init; }

        /// <summary>Number of unique topologies.</summary>
        public int UniqueTopologies { get; init; }
    }

}
