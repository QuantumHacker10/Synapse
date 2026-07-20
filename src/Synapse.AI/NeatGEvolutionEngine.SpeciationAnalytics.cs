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
    /// Provides analytics and diagnostics for species dynamics during evolution.
    /// Tracks diversity indices, inter-species distance matrices, fitness distributions,
    /// and evolutionary lineage.
    /// </summary>
    public sealed class SpeciationAnalytics
    {
        private readonly EvolutionConfig _config;
        private readonly List<SpeciesSnapshot> _snapshots;

        /// <summary>
        /// Initializes a new instance of the SpeciationAnalytics class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public SpeciationAnalytics(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _snapshots = new List<SpeciesSnapshot>();
        }

        /// <summary>
        /// Records a snapshot of species state for analysis.
        /// </summary>
        /// <param name="generation">Current generation.</param>
        /// <param name="species">Current species information.</param>
        /// <param name="population">Current population.</param>
        public void RecordSnapshot(int generation, ImmutableArray<SpeciesInfo> species, GenomePopulation population)
        {
            var snapshot = new SpeciesSnapshot
            {
                Generation = generation,
                SpeciesCount = species.Length,
                SpeciesSizes = species.Select(s => s.MemberCount).ToImmutableArray(),
                SpeciesBestFitness = species.Select(s => s.BestFitness).ToImmutableArray(),
                SpeciesAverageFitness = species.Select(s => s.AverageFitness).ToImmutableArray(),
                ShannonDiversityIndex = ComputeShannonDiversity(species),
                SimpsonDiversityIndex = ComputeSimpsonDiversity(species),
                PopulationDiversity = ComputePopulationDiversity(population),
                InterSpeciesDistances = ComputeInterSpeciesDistanceMatrix(species, population)
            };

            _snapshots.Add(snapshot);
        }

        /// <summary>
        /// Computes the Shannon diversity index for species distribution.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <returns>Shannon diversity index (0 = no diversity, log(n) = max).</returns>
        public double ComputeShannonDiversity(ImmutableArray<SpeciesInfo> species)
        {
            int totalMembers = species.Sum(s => s.MemberCount);
            if (totalMembers == 0)
                return 0;

            double entropy = 0;
            foreach (var s in species)
            {
                if (s.MemberCount > 0)
                {
                    double p = (double)s.MemberCount / totalMembers;
                    entropy -= p * Math.Log2(p);
                }
            }

            return entropy;
        }

        /// <summary>
        /// Computes the Simpson diversity index for species distribution.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <returns>Simpson diversity index (0 = no diversity, 1 = max).</returns>
        public double ComputeSimpsonDiversity(ImmutableArray<SpeciesInfo> species)
        {
            int totalMembers = species.Sum(s => s.MemberCount);
            if (totalMembers <= 1)
                return 0;

            double sumP2 = 0;
            foreach (var s in species)
            {
                double p = (double)s.MemberCount / totalMembers;
                sumP2 += p * p;
            }

            return 1.0 - sumP2;
        }

        /// <summary>
        /// Computes population diversity based on genome feature distances.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <returns>Population diversity metric (0-1).</returns>
        public double ComputePopulationDiversity(GenomePopulation population)
        {
            if (population.Genomes.Length <= 1)
                return 0;

            var fitnesses = population.Genomes.Select(g => g.Fitness).ToArray();
            double range = fitnesses.Max() - fitnesses.Min();
            double stdDev = population.FitnessStandardDeviation;

            double normalizedRange = Math.Min(1.0, range / (Math.Abs(fitnesses.Average()) + 1e-10));
            double normalizedStdDev = Math.Min(1.0, stdDev / (Math.Abs(fitnesses.Average()) + 1e-10));

            return 0.5 * normalizedRange + 0.5 * normalizedStdDev;
        }

        /// <summary>
        /// Computes the inter-species distance matrix.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <param name="population">Current population.</param>
        /// <returns>Distance matrix as a 2D array.</returns>
        public double[,] ComputeInterSpeciesDistanceMatrix(
            ImmutableArray<SpeciesInfo> species,
            GenomePopulation population)
        {
            int n = species.Length;
            var matrix = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    var repA = species[i].Representative;
                    var repB = species[j].Representative;

                    if (repA != null && repB != null)
                    {
                        double dist = ComputeGenomeDistance(repA, repB);
                        matrix[i, j] = dist;
                        matrix[j, i] = dist;
                    }
                }
            }

            return matrix;
        }

        /// <summary>
        /// Gets species fitness distribution statistics.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <returns>Per-species fitness statistics.</returns>
        public IReadOnlyList<SpeciesFitnessStats> GetFitnessDistribution(ImmutableArray<SpeciesInfo> species)
        {
            return species.Select(s => new SpeciesFitnessStats
            {
                SpeciesId = s.Id,
                BestFitness = s.BestFitness,
                AverageFitness = s.AverageFitness,
                MemberCount = s.MemberCount,
                FitnessVariance = ComputeSpeciesVariance(s),
                StagnationCounter = s.StagnationCounter,
                Age = s.Age
            }).ToList().AsReadOnly();
        }

        /// <summary>
        /// Gets snapshots of species over time.
        /// </summary>
        public IReadOnlyList<SpeciesSnapshot> GetSnapshots()
        {
            return _snapshots.ToList().AsReadOnly();
        }

        /// <summary>
        /// Gets species count over generations.
        /// </summary>
        public IReadOnlyList<(int Generation, int SpeciesCount)> GetSpeciesCountOverTime()
        {
            return _snapshots.Select(s => (s.Generation, s.SpeciesCount)).ToList().AsReadOnly();
        }

        /// <summary>
        /// Computes the rate of speciation (new species per generation).
        /// </summary>
        /// <param name="windowSize">Number of generations to average over.</param>
        /// <returns>Average speciation rate.</returns>
        public double ComputeSpeciationRate(int windowSize = 10)
        {
            if (_snapshots.Count < windowSize)
                return 0;

            var recent = _snapshots.Skip(_snapshots.Count - windowSize).ToList();
            int totalNewSpecies = 0;

            for (int i = 1; i < recent.Count; i++)
            {
                totalNewSpecies += Math.Max(0, recent[i].SpeciesCount - recent[i - 1].SpeciesCount);
            }

            return (double)totalNewSpecies / (recent.Count - 1);
        }

        /// <summary>
        /// Computes the extinction rate (species lost per generation).
        /// </summary>
        /// <param name="windowSize">Number of generations to average over.</param>
        /// <returns>Average extinction rate.</returns>
        public double ComputeExtinctionRate(int windowSize = 10)
        {
            if (_snapshots.Count < windowSize)
                return 0;

            var recent = _snapshots.Skip(_snapshots.Count - windowSize).ToList();
            int totalExtinctions = 0;

            for (int i = 1; i < recent.Count; i++)
            {
                totalExtinctions += Math.Max(0, recent[i - 1].SpeciesCount - recent[i].SpeciesCount);
            }

            return (double)totalExtinctions / (recent.Count - 1);
        }

        /// <summary>
        /// Clears recorded snapshots.
        /// </summary>
        public void Clear()
        {
            _snapshots.Clear();
        }

        private double ComputeGenomeDistance(GeoGenome a, GeoGenome b)
        {
            if (a.SemanticEmbedding.IsDefaultOrEmpty || b.SemanticEmbedding.IsDefaultOrEmpty)
                return 1.0;

            int dim = Math.Min(a.SemanticEmbedding.Length, b.SemanticEmbedding.Length);
            double dist = 0;
            for (int d = 0; d < dim; d++)
            {
                double diff = a.SemanticEmbedding[d] - b.SemanticEmbedding[d];
                dist += diff * diff;
            }
            return Math.Sqrt(dist);
        }

        private double ComputeSpeciesVariance(SpeciesInfo species)
        {
            if (species.MemberCount <= 1)
                return 0;
            double mean = species.AverageFitness;
            return species.FitnessHistory.Length > 1
                ? species.FitnessHistory.Select(f => (f - mean) * (f - mean)).Average()
                : 0;
        }
    }

    /// <summary>
    /// Statistics for a single species' fitness distribution.
    /// </summary>
    public record SpeciesFitnessStats
    {
        /// <summary>Species identifier.</summary>
        public int SpeciesId { get; init; }

        /// <summary>Best fitness in the species.</summary>
        public double BestFitness { get; init; }

        /// <summary>Average fitness in the species.</summary>
        public double AverageFitness { get; init; }

        /// <summary>Number of members.</summary>
        public int MemberCount { get; init; }

        /// <summary>Fitness variance within the species.</summary>
        public double FitnessVariance { get; init; }

        /// <summary>Stagnation counter.</summary>
        public int StagnationCounter { get; init; }

        /// <summary>Species age.</summary>
        public int Age { get; init; }
    }

    /// <summary>
    /// Snapshot of species state at a specific generation.
    /// </summary>
    public record SpeciesSnapshot
    {
        /// <summary>Generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Number of species.</summary>
        public int SpeciesCount { get; init; }

        /// <summary>Sizes of each species.</summary>
        public ImmutableArray<int> SpeciesSizes { get; init; }

        /// <summary>Best fitness of each species.</summary>
        public ImmutableArray<double> SpeciesBestFitness { get; init; }

        /// <summary>Average fitness of each species.</summary>
        public ImmutableArray<double> SpeciesAverageFitness { get; init; }

        /// <summary>Shannon diversity index.</summary>
        public double ShannonDiversityIndex { get; init; }

        /// <summary>Simpson diversity index.</summary>
        public double SimpsonDiversityIndex { get; init; }

        /// <summary>Population diversity metric.</summary>
        public double PopulationDiversity { get; init; }

        /// <summary>Inter-species distance matrix.</summary>
        public double[,]? InterSpeciesDistances { get; init; }
    }

}
