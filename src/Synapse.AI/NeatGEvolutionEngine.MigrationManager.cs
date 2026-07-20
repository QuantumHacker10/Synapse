// =============================================================================
// NeatGEvolutionEngine.MigrationManager.cs — NEAT-G partial module
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
    /// Handles gene migration between species pools with semantic compatibility checking,
    /// migration rate control, and anti-premature-convergence measures.
    /// </summary>
    public sealed class MigrationManager
    {
        private readonly EvolutionConfig _config;
        private readonly ConcurrentDictionary<int, Channel<MigrationEvent>> _channels;
        private int _totalMigrations;
        private int _rejectedMigrations;
        private readonly Queue<MigrationEvent> _recentMigrations;

        /// <summary>
        /// Initializes a new instance of the MigrationManager class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public MigrationManager(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _channels = new ConcurrentDictionary<int, Channel<MigrationEvent>>();
            _recentMigrations = new Queue<MigrationEvent>();
        }

        /// <summary>Total migrations performed.</summary>
        public int TotalMigrations => Volatile.Read(ref _totalMigrations);

        /// <summary>Rejected migrations.</summary>
        public int RejectedMigrations => Volatile.Read(ref _rejectedMigrations);

        /// <summary>
        /// Evaluates whether a migration is semantically compatible.
        /// </summary>
        /// <param name="migrant">The genome to migrate.</param>
        /// <param name="targetSpecies">Target species information.</param>
        /// <param name="targetGenomes">Genomes in the target species.</param>
        /// <returns>Compatibility score (0-1) and whether migration is allowed.</returns>
        public (double CompatibilityScore, bool IsAllowed) EvaluateMigration(
            GeoGenome migrant,
            SpeciesInfo targetSpecies,
            IReadOnlyList<GeoGenome> targetGenomes)
        {
            if (targetGenomes.Count == 0)
                return (0.5, true);

            if (migrant.SemanticEmbedding.IsDefaultOrEmpty)
                migrant.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

            double totalDistance = 0;
            int count = 0;

            foreach (var target in targetGenomes.Take(Math.Min(10, targetGenomes.Count)))
            {
                if (target.SemanticEmbedding.IsDefaultOrEmpty)
                    target.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

                int dim = Math.Min(migrant.SemanticEmbedding.Length, target.SemanticEmbedding.Length);
                double dist = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = migrant.SemanticEmbedding[d] - target.SemanticEmbedding[d];
                    dist += diff * diff;
                }
                totalDistance += Math.Sqrt(dist);
                count++;
            }

            double avgDistance = count > 0 ? totalDistance / count : 1.0;
            double compatibilityScore = Math.Exp(-avgDistance * 2.0);

            bool isAllowed = compatibilityScore >= _config.SemanticAlignmentThreshold;

            if (!isAllowed)
            {
                Interlocked.Increment(ref _rejectedMigrations);
            }

            return (compatibilityScore, isAllowed);
        }

        /// <summary>
        /// Performs anti-premature-convergence checks.
        /// </summary>
        /// <param name="species">Current species information.</param>
        /// <param name="population">Current population.</param>
        /// <returns>Whether anti-convergence measures should be activated.</returns>
        public bool CheckAntiConvergence(ImmutableArray<SpeciesInfo> species, GenomePopulation population)
        {
            if (species.Length <= 1)
                return false;

            int totalMembers = species.Sum(s => s.MemberCount);
            if (totalMembers == 0)
                return false;

            double maxSpeciesFraction = species.Max(s => (double)s.MemberCount / totalMembers);
            if (maxSpeciesFraction > 0.6)
                return true;

            double fitnessVariance = population.FitnessStandardDeviation;
            double fitnessRange = population.Genomes.Length > 0
                ? population.Genomes.Max(g => g.Fitness) - population.Genomes.Min(g => g.Fitness)
                : 0;

            if (fitnessRange > 0 && fitnessVariance / fitnessRange < 0.01)
                return true;

            var topologyHashes = population.Genomes
                .Select(g => g.ComputeTopologyHash())
                .Distinct()
                .Count();
            double topologyDiversity = (double)topologyHashes / Math.Max(1, population.Genomes.Length);

            if (topologyDiversity < 0.1)
                return true;

            return false;
        }

        /// <summary>
        /// Gets the recommended migration rate based on current population dynamics.
        /// </summary>
        /// <param name="species">Current species.</param>
        /// <param name="population">Current population.</param>
        /// <returns>Recommended migration rate (0-1).</returns>
        public double GetRecommendedMigrationRate(ImmutableArray<SpeciesInfo> species, GenomePopulation population)
        {
            double baseRate = _config.MigrationRate;

            bool antiConverge = CheckAntiConvergence(species, population);
            if (antiConverge)
            {
                return Math.Min(baseRate * 3.0, 0.3);
            }

            double diversity = 0;
            if (population.Genomes.Length > 0)
            {
                var fitnesses = population.Genomes.Select(g => g.Fitness).ToArray();
                double mean = fitnesses.Average();
                double variance = fitnesses.Average(f => (f - mean) * (f - mean));
                diversity = Math.Min(1.0, Math.Sqrt(variance) / (Math.Abs(mean) + 1e-10));
            }

            if (diversity < 0.1)
                return Math.Min(baseRate * 2.0, 0.2);
            else if (diversity > 0.5)
                return baseRate * 0.5;

            return baseRate;
        }

        /// <summary>
        /// Records a completed migration event.
        /// </summary>
        /// <param name="migrationEvent">The migration event.</param>
        public void RecordMigration(MigrationEvent migrationEvent)
        {
            Interlocked.Increment(ref _totalMigrations);
            lock (_recentMigrations)
            {
                _recentMigrations.Enqueue(migrationEvent);
                while (_recentMigrations.Count > 100)
                    _recentMigrations.Dequeue();
            }
        }

        /// <summary>
        /// Gets recent migration events.
        /// </summary>
        public IReadOnlyList<MigrationEvent> GetRecentMigrations()
        {
            lock (_recentMigrations)
            {
                return _recentMigrations.ToList().AsReadOnly();
            }
        }
    }

}
