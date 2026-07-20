// =============================================================================
// NeatGEvolutionEngine.SelectionStrategies.cs — NEAT-G partial module
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
    /// Implements multiple selection strategies including tournament, roulette wheel,
    /// rank-based, truncation, and stochastic universal sampling.
    /// </summary>
    public sealed class SelectionStrategy : ISelectionStrategy
    {
        private readonly EvolutionConfig _config;
        private readonly FitnessObjective _objective;

        /// <summary>
        /// Initializes a new instance of the SelectionStrategy class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public SelectionStrategy(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _objective = config.Objective;
        }

        /// <inheritdoc/>
        public IReadOnlyList<GeoGenome> SelectParents(GenomePopulation population, int count, Random rng)
        {
            return _config.ParentSelection switch
            {
                SelectionMethod.Tournament => TournamentSelect(population, count, _config.TournamentSize, rng),
                SelectionMethod.RouletteWheel => RouletteWheelSelect(population, count, rng),
                SelectionMethod.RankBased => RankBasedSelect(population, count, rng),
                SelectionMethod.Truncation => TruncationSelect(population, count, rng),
                SelectionMethod.StochasticUniversal => StochasticUniversalSelect(population, count, rng),
                _ => TournamentSelect(population, count, _config.TournamentSize, rng)
            };
        }

        /// <inheritdoc/>
        public IReadOnlyList<GeoGenome> SelectSurvivors(
            IReadOnlyList<GeoGenome> current,
            IReadOnlyList<GeoGenome> offspring,
            int targetSize)
        {
            var combined = new List<GeoGenome>();
            combined.AddRange(current);
            combined.AddRange(offspring);

            return _config.SurvivalSelection switch
            {
                SelectionMethod.Truncation => combined
                    .OrderByDescending(g => GetFitnessForComparison(g))
                    .Take(targetSize)
                    .ToList(),
                SelectionMethod.Tournament => TournamentSelectCombined(combined, targetSize, _config.TournamentSize, new Random()),
                _ => combined
                    .OrderByDescending(g => GetFitnessForComparison(g))
                    .Take(targetSize)
                    .ToList()
            };
        }

        private IReadOnlyList<GeoGenome> TournamentSelect(GenomePopulation population, int count, int tournamentSize, Random rng)
        {
            var selected = new List<GeoGenome>();
            var genomes = population.Genomes.ToArray();

            for (int i = 0; i < count; i++)
            {
                GeoGenome? best = null;
                for (int t = 0; t < tournamentSize; t++)
                {
                    var candidate = genomes[rng.Next(genomes.Length)];
                    if (best == null || GetFitnessForComparison(candidate) > GetFitnessForComparison(best))
                    {
                        best = candidate;
                    }
                }
                if (best != null)
                    selected.Add(best);
            }

            return selected;
        }

        private IReadOnlyList<GeoGenome> TournamentSelectCombined(List<GeoGenome> genomes, int count, int tournamentSize, Random rng)
        {
            var selected = new List<GeoGenome>();
            var array = genomes.ToArray();

            for (int i = 0; i < count; i++)
            {
                GeoGenome? best = null;
                for (int t = 0; t < tournamentSize; t++)
                {
                    var candidate = array[rng.Next(array.Length)];
                    if (best == null || GetFitnessForComparison(candidate) > GetFitnessForComparison(best))
                    {
                        best = candidate;
                    }
                }
                if (best != null)
                    selected.Add(best);
            }

            return selected;
        }

        private IReadOnlyList<GeoGenome> RouletteWheelSelect(GenomePopulation population, int count, Random rng)
        {
            var genomes = population.Genomes.ToArray();
            var fitnesses = genomes.Select(g => GetFitnessForComparison(g)).ToArray();

            double minFitness = fitnesses.Min();
            double shifted = minFitness < 0 ? -minFitness + 1 : 0;
            double total = fitnesses.Sum(f => f + shifted);

            var selected = new List<GeoGenome>();
            for (int i = 0; i < count; i++)
            {
                double r = rng.NextDouble() * total;
                double cumulative = 0;
                for (int j = 0; j < genomes.Length; j++)
                {
                    cumulative += fitnesses[j] + shifted;
                    if (cumulative >= r)
                    {
                        selected.Add(genomes[j]);
                        break;
                    }
                }
            }

            return selected;
        }

        private IReadOnlyList<GeoGenome> RankBasedSelect(GenomePopulation population, int count, Random rng)
        {
            var ranked = population.Genomes
                .OrderByDescending(g => GetFitnessForComparison(g))
                .ToArray();

            int n = ranked.Length;
            var probabilities = new double[n];
            double totalProb = 0;

            for (int i = 0; i < n; i++)
            {
                probabilities[i] = (2.0 - _config.TournamentSize) / n +
                                   2.0 * (n - 1 - i) * (_config.TournamentSize - 1) / (n * (n - 1));
                totalProb += probabilities[i];
            }

            var selected = new List<GeoGenome>();
            for (int i = 0; i < count; i++)
            {
                double r = rng.NextDouble() * totalProb;
                double cumulative = 0;
                for (int j = 0; j < n; j++)
                {
                    cumulative += probabilities[j];
                    if (cumulative >= r)
                    {
                        selected.Add(ranked[j]);
                        break;
                    }
                }
            }

            return selected;
        }

        private IReadOnlyList<GeoGenome> TruncationSelect(GenomePopulation population, int count, Random rng)
        {
            int truncationSize = Math.Max(1, population.Genomes.Length / _config.TournamentSize);
            var top = population.Genomes
                .OrderByDescending(g => GetFitnessForComparison(g))
                .Take(truncationSize)
                .ToArray();

            var selected = new List<GeoGenome>();
            for (int i = 0; i < count; i++)
            {
                selected.Add(top[rng.Next(top.Length)]);
            }

            return selected;
        }

        private IReadOnlyList<GeoGenome> StochasticUniversalSelect(GenomePopulation population, int count, Random rng)
        {
            var ranked = population.Genomes
                .OrderByDescending(g => GetFitnessForComparison(g))
                .ToArray();

            int n = ranked.Length;
            double totalFitness = ranked.Sum(g => GetFitnessForComparison(g));
            if (totalFitness <= 0)
            {
                return ranked.Take(count).ToList();
            }

            double spacing = totalFitness / count;
            double start = rng.NextDouble() * spacing;

            var selected = new List<GeoGenome>();
            double cumulative = 0;
            int idx = 0;

            for (int i = 0; i < count; i++)
            {
                double pointer = start + i * spacing;
                while (idx < n - 1 && cumulative + GetFitnessForComparison(ranked[idx]) < pointer)
                {
                    cumulative += GetFitnessForComparison(ranked[idx]);
                    idx++;
                }
                selected.Add(ranked[Math.Min(idx, n - 1)]);
            }

            return selected;
        }

        private double GetFitnessForComparison(GeoGenome genome)
        {
            double fitness = genome.AdjustedFitness != 0 ? genome.AdjustedFitness : genome.Fitness;
            return _objective == FitnessObjective.Minimize ? -fitness : fitness;
        }
    }

}
