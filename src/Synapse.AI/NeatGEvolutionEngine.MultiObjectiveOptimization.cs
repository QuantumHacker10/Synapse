// =============================================================================
// NeatGEvolutionEngine.MultiObjectiveOptimization.cs — NEAT-G partial module
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
    /// Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) for
    /// multi-objective optimization of genome fitness.
    /// Provides Pareto-optimal solutions for conflicting objectives.
    /// </summary>
    public sealed class NSGAIISelector
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the NSGAIISelector class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public NSGAIISelector(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Performs non-dominated sorting of the population.
        /// </summary>
        /// <param name="population">Population to sort.</param>
        /// <returns>List of fronts, each containing genome indices.</returns>
        public IReadOnlyList<IReadOnlyList<int>> NonDominatedSorting(GenomePopulation population)
        {
            var genomes = population.Genomes.ToArray();
            int n = genomes.Length;
            var dominationCount = new int[n];
            var dominatedSet = new List<int>[n];
            var fronts = new List<IReadOnlyList<int>>();

            for (int i = 0; i < n; i++)
            {
                dominatedSet[i] = new List<int>();
                dominationCount[i] = 0;
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (Domominates(genomes[i], genomes[j]))
                    {
                        dominatedSet[i].Add(j);
                        dominationCount[j]++;
                    }
                    else if (Domominates(genomes[j], genomes[i]))
                    {
                        dominatedSet[j].Add(i);
                        dominationCount[i]++;
                    }
                }
            }

            var currentFront = new List<int>();
            for (int i = 0; i < n; i++)
            {
                if (dominationCount[i] == 0)
                {
                    currentFront.Add(i);
                }
            }

            while (currentFront.Count > 0)
            {
                fronts.Add(currentFront.AsReadOnly());

                var nextFront = new List<int>();
                foreach (int i in currentFront)
                {
                    foreach (int j in dominatedSet[i])
                    {
                        dominationCount[j]--;
                        if (dominationCount[j] == 0)
                        {
                            nextFront.Add(j);
                        }
                    }
                }
                currentFront = nextFront;
            }

            return fronts;
        }

        /// <summary>
        /// Computes crowding distances for a front.
        /// </summary>
        /// <param name="front">Indices of genomes in the front.</param>
        /// <param name="population">The population.</param>
        /// <returns>Crowding distance for each genome in the front.</returns>
        public IReadOnlyList<double> ComputeCrowdingDistances(
            IReadOnlyList<int> front,
            GenomePopulation population)
        {
            int frontSize = front.Count;
            var distances = new double[frontSize];

            if (frontSize <= 2)
            {
                for (int i = 0; i < frontSize; i++)
                    distances[i] = double.PositiveInfinity;
                return distances;
            }

            var objectives = Enum.GetValues<FitnessComponent>();

            foreach (var objective in objectives)
            {
                var genomeObjectives = front
                    .Select((idx, pos) => (Index: pos, Value: GetObjectiveValue(population.Genomes[idx], objective)))
                    .OrderBy(x => x.Value)
                    .ToList();

                distances[genomeObjectives[0].Index] = double.PositiveInfinity;
                distances[genomeObjectives[^1].Index] = double.PositiveInfinity;

                double range = genomeObjectives[^1].Value - genomeObjectives[0].Value;
                if (range < 1e-10)
                    continue;

                for (int i = 1; i < frontSize - 1; i++)
                {
                    double gap = genomeObjectives[i + 1].Value - genomeObjectives[i - 1].Value;
                    distances[genomeObjectives[i].Index] += gap / range;
                }
            }

            return distances.ToList().AsReadOnly();
        }

        /// <summary>
        /// Selects parents using NSGA-II tournament selection.
        /// </summary>
        /// <param name="population">Population to select from.</param>
        /// <param name="count">Number of parents to select.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>Selected parent genomes.</returns>
        public IReadOnlyList<GeoGenome> Select(GenomePopulation population, int count, Random rng)
        {
            var fronts = NonDominatedSorting(population);
            var selected = new List<GeoGenome>();

            var frontRanks = new int[population.Genomes.Length];
            for (int f = 0; f < fronts.Count; f++)
            {
                foreach (int idx in fronts[f])
                {
                    frontRanks[idx] = f;
                }
            }

            var crowdingDistances = new double[population.Genomes.Length];
            for (int f = 0; f < fronts.Count; f++)
            {
                var distances = ComputeCrowdingDistances(fronts[f], population);
                for (int i = 0; i < fronts[f].Count; i++)
                {
                    crowdingDistances[fronts[f][i]] = distances[i];
                }
            }

            for (int i = 0; i < count; i++)
            {
                int candidate1 = rng.Next(population.Genomes.Length);
                int candidate2 = rng.Next(population.Genomes.Length);

                bool candidate1Better = fronts.Count == 0 ||
                    frontRanks[candidate1] < frontRanks[candidate2] ||
                    (frontRanks[candidate1] == frontRanks[candidate2] &&
                     crowdingDistances[candidate1] > crowdingDistances[candidate2]);

                int winner = candidate1Better ? candidate1 : candidate2;
                selected.Add(population.Genomes[winner]);
            }

            return selected;
        }

        private bool Domominates(GeoGenome a, GeoGenome b)
        {
            bool atLeastOneBetter = false;

            foreach (var component in Enum.GetValues<FitnessComponent>())
            {
                double valA = GetObjectiveValue(a, component);
                double valB = GetObjectiveValue(b, component);

                if (valA < valB)
                    return false;
                if (valA > valB)
                    atLeastOneBetter = true;
            }

            return atLeastOneBetter;
        }

        private double GetObjectiveValue(GeoGenome genome, FitnessComponent component)
        {
            if (genome.FitnessComponents.TryGetValue(component, out double value))
                return value;
            return genome.Fitness;
        }
    }

}
