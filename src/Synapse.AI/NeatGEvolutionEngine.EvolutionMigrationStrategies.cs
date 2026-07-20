// =============================================================================
// NeatGEvolutionEngine.EvolutionMigrationStrategies.cs — NEAT-G partial module
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
    /// Advanced migration strategies for inter-species gene flow.
    /// Implements island model migration with various topologies.
    /// </summary>
    public sealed class IslandModelMigrationManager
    {
        private readonly EvolutionConfig _config;
        private readonly MigrationTopology _topology;
        private readonly Dictionary<int, List<int>> _neighborMap;
        private readonly Random _rng;

        /// <summary>
        /// Initializes a new instance of the IslandModelMigrationManager class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="topology">Migration topology.</param>
        /// <param name="rng">Random number generator.</param>
        public IslandModelMigrationManager(
            EvolutionConfig config,
            MigrationTopology topology = MigrationTopology.Ring,
            Random? rng = null)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _topology = topology;
            _rng = rng ?? new Random();
            _neighborMap = new Dictionary<int, List<int>>();
        }

        /// <summary>
        /// Configures the neighbor map based on species IDs and topology.
        /// </summary>
        /// <param name="speciesIds">List of species IDs.</param>
        public void ConfigureNeighbors(IReadOnlyList<int> speciesIds)
        {
            _neighborMap.Clear();

            switch (_topology)
            {
                case MigrationTopology.Ring:
                    ConfigureRing(speciesIds);
                    break;
                case MigrationTopology.FullMesh:
                    ConfigureFullMesh(speciesIds);
                    break;
                case MigrationTopology.Random:
                    ConfigureRandom(speciesIds);
                    break;
                case MigrationTopology.Hierarchical:
                    ConfigureHierarchical(speciesIds);
                    break;
                case MigrationTopology.SmallWorld:
                    ConfigureSmallWorld(speciesIds);
                    break;
            }
        }

        /// <summary>
        /// Selects migration partners for each species based on the topology.
        /// </summary>
        /// <param name="species">Current species.</param>
        /// <returns>Migration pairs (source, target).</returns>
        public IReadOnlyList<(int SourceSpeciesId, int TargetSpeciesId)> SelectMigrationPartners(
            ImmutableArray<SpeciesInfo> species)
        {
            var speciesIds = species.Select(s => s.Id).ToList();
            ConfigureNeighbors(speciesIds);

            var pairs = new List<(int, int)>();
            int maxMigrations = Math.Min(_config.MaxMigrationsPerGeneration, speciesIds.Count);

            foreach (var sourceId in speciesIds)
            {
                if (!_neighborMap.TryGetValue(sourceId, out var neighbors) || neighbors.Count == 0)
                    continue;

                int targetId = neighbors[_rng.Next(neighbors.Count)];
                if (sourceId != targetId)
                {
                    pairs.Add((sourceId, targetId));
                }

                if (pairs.Count >= maxMigrations)
                    break;
            }

            return pairs;
        }

        /// <summary>
        /// Selects the best migrants from a species for emigration.
        /// </summary>
        /// <param name="species">Source species info.</param>
        /// <param name="genomes">All genomes.</param>
        /// <param name="count">Number of migrants to select.</param>
        /// <returns>Selected migrant genomes.</returns>
        public IReadOnlyList<GeoGenome> SelectMigrants(
            SpeciesInfo species,
            IReadOnlyList<GeoGenome> genomes,
            int count)
        {
            var speciesGenomes = genomes
                .Where(g => species.MemberIds.Contains(g.Id))
                .ToList();

            if (speciesGenomes.Count == 0)
                return Array.Empty<GeoGenome>();

            var ranked = speciesGenomes
                .OrderByDescending(g => g.Fitness)
                .ToList();

            int eliteBoundary = Math.Max(1, ranked.Count / 3);
            var candidates = ranked.Skip(eliteBoundary).ToList();

            if (candidates.Count == 0)
                candidates = ranked;

            var migrants = new List<GeoGenome>();
            for (int i = 0; i < Math.Min(count, candidates.Count); i++)
            {
                migrants.Add(candidates[i].Clone());
            }

            return migrants;
        }

        /// <summary>
        /// Computes migration acceptance probability based on fitness and diversity.
        /// </summary>
        /// <param name="migrant">The migrant genome.</param>
        /// <param name="targetSpecies">Target species info.</param>
        /// <param name="targetGenomes">Target species genomes.</param>
        /// <returns>Acceptance probability (0-1).</returns>
        public double ComputeAcceptanceProbability(
            GeoGenome migrant,
            SpeciesInfo targetSpecies,
            IReadOnlyList<GeoGenome> targetGenomes)
        {
            if (targetGenomes.Count == 0)
                return 1.0;

            double fitnessRatio = targetSpecies.AverageFitness > 0
                ? migrant.Fitness / targetSpecies.AverageFitness
                : 1.0;

            double fitnessScore = Math.Clamp(fitnessRatio, 0.1, 2.0) / 2.0;

            double diversityBenefit = 0;
            if (targetGenomes.Count > 1)
            {
                var targetHashes = targetGenomes.Select(g => g.ComputeTopologyHash()).ToHashSet();
                long migrantHash = migrant.ComputeTopologyHash();
                bool isNewTopology = !targetHashes.Contains(migrantHash);
                diversityBenefit = isNewTopology ? 0.3 : 0;
            }

            double acceptance = 0.5 * fitnessScore + 0.2 + diversityBenefit;
            return Math.Clamp(acceptance, 0.1, 0.9);
        }

        private void ConfigureRing(IReadOnlyList<int> speciesIds)
        {
            for (int i = 0; i < speciesIds.Count; i++)
            {
                int nextIdx = (i + 1) % speciesIds.Count;
                _neighborMap[speciesIds[i]] = new List<int> { speciesIds[nextIdx] };
            }
        }

        private void ConfigureFullMesh(IReadOnlyList<int> speciesIds)
        {
            foreach (var id in speciesIds)
            {
                _neighborMap[id] = speciesIds.Where(other => other != id).ToList();
            }
        }

        private void ConfigureRandom(IReadOnlyList<int> speciesIds)
        {
            foreach (var id in speciesIds)
            {
                int neighborCount = Math.Max(1, speciesIds.Count / 3);
                var neighbors = speciesIds
                    .Where(other => other != id)
                    .OrderBy(_ => _rng.Next())
                    .Take(neighborCount)
                    .ToList();
                _neighborMap[id] = neighbors;
            }
        }

        private void ConfigureHierarchical(IReadOnlyList<int> speciesIds)
        {
            if (speciesIds.Count <= 2)
            {
                ConfigureFullMesh(speciesIds);
                return;
            }

            int mid = speciesIds.Count / 2;
            var left = speciesIds.Take(mid).ToList();
            var right = speciesIds.Skip(mid).ToList();

            foreach (var id in left)
            {
                _neighborMap[id] = left.Where(other => other != id).ToList();
                if (right.Count > 0)
                    _neighborMap[id].Add(right[_rng.Next(right.Count)]);
            }

            foreach (var id in right)
            {
                _neighborMap[id] = right.Where(other => other != id).ToList();
                if (left.Count > 0)
                    _neighborMap[id].Add(left[_rng.Next(left.Count)]);
            }
        }

        private void ConfigureSmallWorld(IReadOnlyList<int> speciesIds)
        {
            ConfigureRing(speciesIds);

            int rewiredCount = Math.Max(1, speciesIds.Count / 5);
            for (int i = 0; i < rewiredCount; i++)
            {
                int sourceIdx = _rng.Next(speciesIds.Count);
                int sourceId = speciesIds[sourceIdx];

                if (_neighborMap.TryGetValue(sourceId, out var neighbors) && neighbors.Count > 0)
                {
                    int targetIdx = _rng.Next(neighbors.Count);
                    int newTarget = speciesIds[_rng.Next(speciesIds.Count)];
                    if (newTarget != sourceId)
                    {
                        neighbors[targetIdx] = newTarget;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Migration network topologies.
    /// </summary>
    public enum MigrationTopology
    {
        /// <summary>Each species migrates to the next in a ring.</summary>
        Ring,

        /// <summary>Every species can migrate to every other species.</summary>
        FullMesh,

        /// <summary>Random neighbor selection.</summary>
        Random,

        /// <summary>Hierarchical tree structure.</summary>
        Hierarchical,

        /// <summary>Ring with random shortcuts (small-world network).</summary>
        SmallWorld
    }

}
