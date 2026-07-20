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
    /// Tracks detailed diagnostics about the evolution process including
    /// mutation success rates, crossover success rates, species dynamics,
    /// and population diversity over time.
    /// </summary>
    public sealed class EvolutionDiagnostics
    {
        private readonly EvolutionConfig _config;
        private readonly List<DiagnosticsSnapshot> _snapshots;
        private readonly ConcurrentDictionary<MutationType, (int attempts, int successes)> _mutationStats;
        private readonly ConcurrentDictionary<string, (int attempts, int successes)> _crossoverStats;

        /// <summary>
        /// Initializes a new instance of the EvolutionDiagnostics class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public EvolutionDiagnostics(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _snapshots = new List<DiagnosticsSnapshot>();
            _mutationStats = new ConcurrentDictionary<MutationType, (int, int)>();
            _crossoverStats = new ConcurrentDictionary<string, (int, int)>();
        }

        /// <summary>
        /// Records a mutation attempt.
        /// </summary>
        /// <param name="type">Mutation type.</param>
        /// <param name="success">Whether the mutation was successful.</param>
        public void RecordMutation(MutationType type, bool success)
        {
            _mutationStats.AddOrUpdate(type,
                _ => success ? (1, 1) : (1, 0),
                (_, old) => (old.attempts + 1, old.successes + (success ? 1 : 0)));
        }

        /// <summary>
        /// Records a crossover attempt.
        /// </summary>
        /// <param name="strategy">Crossover strategy name.</param>
        /// <param name="success">Whether the crossover was successful.</param>
        public void RecordCrossover(string strategy, bool success)
        {
            _crossoverStats.AddOrUpdate(strategy,
                _ => success ? (1, 1) : (1, 0),
                (_, old) => (old.attempts + 1, old.successes + (success ? 1 : 0)));
        }

        /// <summary>
        /// Records a diagnostics snapshot.
        /// </summary>
        /// <param name="generation">Current generation.</param>
        /// <param name="population">Current population.</param>
        /// <param name="species">Current species.</param>
        public void RecordSnapshot(int generation, GenomePopulation population, ImmutableArray<SpeciesInfo> species)
        {
            var snapshot = new DiagnosticsSnapshot
            {
                Generation = generation,
                MutationSuccessRates = GetMutationSuccessRates(),
                CrossoverSuccessRates = GetCrossoverSuccessRates(),
                PopulationDiversity = ComputeDiversity(population),
                SpeciesCount = species.Length,
                SpeciesSizeVariance = species.Length > 1
                    ? species.Select(s => (double)s.MemberCount).ToArray().Select(m => { var avg = species.Average(s => (double)s.MemberCount); return (m - avg) * (m - avg); }).Average()
                    : 0,
                BestFitness = population.Genomes.Length > 0 ? population.Genomes.Max(g => g.Fitness) : 0,
                AverageFitness = population.AverageFitness,
                StructuralDiversity = ComputeStructuralDiversity(population),
                WeightDiversity = ComputeWeightDiversity(population),
                ActiveGeneRatio = ComputeActiveGeneRatio(population)
            };

            lock (_snapshots)
            {
                _snapshots.Add(snapshot);
            }
        }

        /// <summary>
        /// Gets mutation success rates per type.
        /// </summary>
        public IReadOnlyDictionary<MutationType, double> GetMutationSuccessRates()
        {
            return _mutationStats.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.attempts > 0
                    ? (double)kvp.Value.successes / kvp.Value.attempts
                    : 0);
        }

        /// <summary>
        /// Gets crossover success rates per strategy.
        /// </summary>
        public IReadOnlyDictionary<string, double> GetCrossoverSuccessRates()
        {
            return _crossoverStats.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.attempts > 0
                    ? (double)kvp.Value.successes / kvp.Value.attempts
                    : 0);
        }

        /// <summary>
        /// Gets all recorded snapshots.
        /// </summary>
        public IReadOnlyList<DiagnosticsSnapshot> GetSnapshots()
        {
            lock (_snapshots)
            {
                return _snapshots.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets overall mutation success rate across all types.
        /// </summary>
        public double GetOverallMutationSuccessRate()
        {
            int totalAttempts = _mutationStats.Values.Sum(v => v.attempts);
            int totalSuccesses = _mutationStats.Values.Sum(v => v.successes);
            return totalAttempts > 0 ? (double)totalSuccesses / totalAttempts : 0;
        }

        /// <summary>
        /// Gets overall crossover success rate across all strategies.
        /// </summary>
        public double GetOverallCrossoverSuccessRate()
        {
            int totalAttempts = _crossoverStats.Values.Sum(v => v.attempts);
            int totalSuccesses = _crossoverStats.Values.Sum(v => v.successes);
            return totalAttempts > 0 ? (double)totalSuccesses / totalAttempts : 0;
        }

        /// <summary>
        /// Clears all diagnostic data.
        /// </summary>
        public void Clear()
        {
            lock (_snapshots)
            {
                _snapshots.Clear();
            }
            _mutationStats.Clear();
            _crossoverStats.Clear();
        }

        private double ComputeDiversity(GenomePopulation population)
        {
            if (population.Genomes.Length <= 1)
                return 0;
            return population.FitnessStandardDeviation / (Math.Abs(population.AverageFitness) + 1e-10);
        }

        private double ComputeStructuralDiversity(GenomePopulation population)
        {
            if (population.Genomes.Length <= 1)
                return 0;
            var hashes = population.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count();
            return (double)hashes / population.Genomes.Length;
        }

        private double ComputeWeightDiversity(GenomePopulation population)
        {
            if (population.Genomes.Length <= 1)
                return 0;

            var allWeights = population.Genomes
                .SelectMany(g => g.Synapses.Where(s => s.IsActive).Select(s => s.Weight))
                .ToList();

            if (allWeights.Count == 0)
                return 0;

            double mean = allWeights.Average();
            double variance = allWeights.Average(w => (w - mean) * (w - mean));
            return Math.Sqrt(variance);
        }

        private double ComputeActiveGeneRatio(GenomePopulation population)
        {
            if (population.Genomes.Length == 0)
                return 0;

            double totalActive = population.Genomes.Sum(g => g.ActiveNeuronCount + g.ActiveSynapseCount);
            double totalPossible = population.Genomes.Sum(g => g.TotalNeuronCount + g.TotalSynapseCount);

            return totalPossible > 0 ? totalActive / totalPossible : 0;
        }
    }

    /// <summary>
    /// Snapshot of diagnostics data at a specific generation.
    /// </summary>
    public record DiagnosticsSnapshot
    {
        /// <summary>Generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Mutation success rates per type.</summary>
        public IReadOnlyDictionary<MutationType, double> MutationSuccessRates { get; init; } =
            new Dictionary<MutationType, double>();

        /// <summary>Crossover success rates per strategy.</summary>
        public IReadOnlyDictionary<string, double> CrossoverSuccessRates { get; init; } =
            new Dictionary<string, double>();

        /// <summary>Population diversity metric.</summary>
        public double PopulationDiversity { get; init; }

        /// <summary>Species count.</summary>
        public int SpeciesCount { get; init; }

        /// <summary>Variance of species sizes.</summary>
        public double SpeciesSizeVariance { get; init; }

        /// <summary>Best fitness.</summary>
        public double BestFitness { get; init; }

        /// <summary>Average fitness.</summary>
        public double AverageFitness { get; init; }

        /// <summary>Structural diversity (unique topologies / total).</summary>
        public double StructuralDiversity { get; init; }

        /// <summary>Weight diversity (std dev of weights).</summary>
        public double WeightDiversity { get; init; }

        /// <summary>Ratio of active genes to total genes.</summary>
        public double ActiveGeneRatio { get; init; }
    }

}
