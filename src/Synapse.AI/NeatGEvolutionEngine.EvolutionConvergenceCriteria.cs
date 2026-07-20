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
    /// Provides pluggable convergence criteria for stopping evolution based on
    /// fitness plateaus, diversity thresholds, or custom predicates.
    /// </summary>
    public sealed class ConvergenceCriteria
    {
        private readonly List<Func<GenomePopulation, IReadOnlyList<SpeciesInfo>, int, bool>> _criteria;
        private readonly List<string> _criterionNames;
        private readonly Queue<double> _fitnessHistory;
        private readonly int _historyWindowSize;

        /// <summary>
        /// Initializes a new instance of the ConvergenceCriteria class.
        /// </summary>
        /// <param name="historyWindowSize">Number of generations to track for plateau detection.</param>
        public ConvergenceCriteria(int historyWindowSize = 50)
        {
            _historyWindowSize = historyWindowSize;
            _fitnessHistory = new Queue<double>(historyWindowSize);
            _criteria = new List<Func<GenomePopulation, IReadOnlyList<SpeciesInfo>, int, bool>>();
            _criterionNames = new List<string>();
        }

        /// <summary>
        /// Adds a fitness plateau criterion. Stops when best fitness hasn't improved
        /// for the specified number of generations.
        /// </summary>
        /// <param name="generationsWithoutImprovement">Number of stagnant generations.</param>
        /// <param name="tolerance">Minimum improvement to consider as progress.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddFitnessPlateau(int generationsWithoutImprovement, double tolerance = 1e-8)
        {
            _criteria.Add((pop, species, gen) =>
            {
                if (_fitnessHistory.Count < generationsWithoutImprovement)
                    return false;

                var recent = _fitnessHistory.TakeLast(generationsWithoutImprovement).ToList();
                double improvement = recent[^1] - recent[0];
                return improvement <= tolerance;
            });
            _criterionNames.Add($"FitnessPlateau(gens={generationsWithoutImprovement}, tol={tolerance})");
            return this;
        }

        /// <summary>
        /// Adds a maximum generation limit criterion.
        /// </summary>
        /// <param name="maxGenerations">Maximum number of generations.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddMaxGenerations(int maxGenerations)
        {
            _criteria.Add((pop, species, gen) => gen >= maxGenerations);
            _criterionNames.Add($"MaxGenerations({maxGenerations})");
            return this;
        }

        /// <summary>
        /// Adds a minimum diversity criterion. Stops when topology diversity drops below threshold.
        /// </summary>
        /// <param name="minimumDiversityRatio">Minimum fraction of unique topologies.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddMinimumDiversity(double minimumDiversityRatio)
        {
            _criteria.Add((pop, species, gen) =>
            {
                if (pop.Genomes.Length == 0)
                    return true;
                var uniqueHashes = pop.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count();
                double diversity = (double)uniqueHashes / pop.Genomes.Length;
                return diversity < minimumDiversityRatio;
            });
            _criterionNames.Add($"MinimumDiversity({minimumDiversityRatio})");
            return this;
        }

        /// <summary>
        /// Adds a target fitness criterion. Stops when any genome reaches the target.
        /// </summary>
        /// <param name="targetFitness">Target fitness value.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddTargetFitness(double targetFitness)
        {
            _criteria.Add((pop, species, gen) =>
                pop.Genomes.Any(g => g.Fitness >= targetFitness));
            _criterionNames.Add($"TargetFitness({targetFitness})");
            return this;
        }

        /// <summary>
        /// Adds a species stagnation criterion. Stops when all species are stagnant.
        /// </summary>
        /// <param name="stagnantGenerations">Number of generations without improvement per species.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddSpeciesStagnation(int stagnantGenerations)
        {
            _criteria.Add((pop, species, gen) =>
            {
                if (species.Count == 0)
                    return false;
                return species.All(s => s.GenerationsWithoutImprovement >= stagnantGenerations);
            });
            _criterionNames.Add($"SpeciesStagnation({stagnantGenerations})");
            return this;
        }

        /// <summary>
        /// Adds a custom convergence criterion.
        /// </summary>
        /// <param name="name">Name of the criterion.</param>
        /// <param name="predicate">Predicate that returns true when evolution should stop.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddCustom(string name, Func<GenomePopulation, IReadOnlyList<SpeciesInfo>, int, bool> predicate)
        {
            _criteria.Add(predicate);
            _criterionNames.Add(name);
            return this;
        }

        /// <summary>
        /// Evaluates all convergence criteria against the current state.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <param name="species">Current species.</param>
        /// <param name="generation">Current generation.</param>
        /// <returns>Convergence result with details.</returns>
        public ConvergenceResult Evaluate(GenomePopulation population, IReadOnlyList<SpeciesInfo> species, int generation)
        {
            if (population.Genomes.Length > 0)
            {
                double bestFitness = population.Genomes.Max(g => g.Fitness);
                _fitnessHistory.Enqueue(bestFitness);

                while (_fitnessHistory.Count > _historyWindowSize)
                    _fitnessHistory.Dequeue();
            }

            var triggeredCriteria = new List<string>();

            for (int i = 0; i < _criteria.Count; i++)
            {
                if (_criteria[i](population, species, generation))
                {
                    triggeredCriteria.Add(_criterionNames[i]);
                }
            }

            bool hasConverged = triggeredCriteria.Count > 0;

            double bestFitnessValue = population.Genomes.Length > 0
                ? population.Genomes.Max(g => g.Fitness)
                : double.MinValue;

            double diversityRatio = population.Genomes.Length > 0
                ? (double)population.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count() / population.Genomes.Length
                : 0;

            return new ConvergenceResult
            {
                HasConverged = hasConverged,
                TriggeredCriteria = triggeredCriteria.AsReadOnly(),
                Generation = generation,
                BestFitness = bestFitnessValue,
                TopologyDiversity = diversityRatio,
                SpeciesCount = species.Count,
                FitnessHistorySize = _fitnessHistory.Count
            };
        }

        /// <summary>
        /// Resets the internal state (fitness history).
        /// </summary>
        public void Reset()
        {
            _fitnessHistory.Clear();
        }

        /// <summary>
        /// Gets the names of all registered criteria.
        /// </summary>
        public IReadOnlyList<string> GetCriterionNames()
        {
            return _criterionNames.AsReadOnly();
        }
    }

    /// <summary>
    /// Result of convergence evaluation.
    /// </summary>
    public sealed class ConvergenceResult
    {
        /// <summary>Whether convergence was detected.</summary>
        public bool HasConverged { get; init; }
        /// <summary>Names of triggered criteria.</summary>
        public IReadOnlyList<string> TriggeredCriteria { get; init; } = Array.Empty<string>();
        /// <summary>Current generation.</summary>
        public int Generation { get; init; }
        /// <summary>Current best fitness.</summary>
        public double BestFitness { get; init; }
        /// <summary>Current topology diversity ratio.</summary>
        public double TopologyDiversity { get; init; }
        /// <summary>Number of species.</summary>
        public int SpeciesCount { get; init; }
        /// <summary>Size of fitness history buffer.</summary>
        public int FitnessHistorySize { get; init; }
    }

}
