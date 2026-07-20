// =============================================================================
// NeatGEvolutionEngine.EvolutionConfigurationOptimizer.cs — NEAT-G partial module
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
    /// Optimizes evolution configuration parameters using meta-evolution.
    /// Treats configuration parameters as a genome and evolves them to find
    /// optimal settings for a given problem class.
    /// </summary>
    public sealed class ConfigurationOptimizer
    {
        private readonly int _inputCount;
        private readonly int _outputCount;
        private readonly EvaluationContext _context;
        private readonly EvolutionConfig _baseConfig;
        private readonly Random _rng;

        /// <summary>
        /// Initializes a new instance of the ConfigurationOptimizer class.
        /// </summary>
        /// <param name="inputCount">Network input count.</param>
        /// <param name="outputCount">Network output count.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="baseConfig">Base configuration to optimize from.</param>
        public ConfigurationOptimizer(
            int inputCount,
            int outputCount,
            EvaluationContext context,
            EvolutionConfig baseConfig)
        {
            _inputCount = inputCount;
            _outputCount = outputCount;
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _baseConfig = baseConfig ?? throw new ArgumentNullException(nameof(baseConfig));
            _rng = new Random(42);
        }

        /// <summary>
        /// Optimizes configuration parameters using random search.
        /// </summary>
        /// <param name="trials">Number of random configurations to try.</param>
        /// <param name="maxGenerationsPerTrial">Max generations per trial.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Best configuration found and its performance.</returns>
        public async Task<(EvolutionConfig BestConfig, double BestFitness)> OptimizeRandomSearchAsync(
            int trials,
            int maxGenerationsPerTrial,
            CancellationToken ct = default)
        {
            EvolutionConfig bestConfig = _baseConfig.Clone();
            double bestFitness = double.MinValue;

            for (int trial = 0; trial < trials; trial++)
            {
                ct.ThrowIfCancellationRequested();

                var config = GenerateRandomConfig();
                config.MaxGenerations = maxGenerationsPerTrial;
                config.RandomSeed = _rng.Next();

                var engine = new NeatGEvolutionEngine(config);
                var result = await engine.RunEvolutionAsync(
                    _inputCount, _outputCount, _context, null, ct).ConfigureAwait(false);

                if (result.BestGenome.Fitness > bestFitness)
                {
                    bestFitness = result.BestGenome.Fitness;
                    bestConfig = config;
                }
            }

            return (bestConfig, bestFitness);
        }

        /// <summary>
        /// Optimizes configuration using hill-climbing on parameter space.
        /// </summary>
        /// <param name="iterations">Number of hill-climbing iterations.</param>
        /// <param name="maxGenerationsPerTrial">Max generations per trial.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Best configuration found and its performance.</returns>
        public async Task<(EvolutionConfig BestConfig, double BestFitness)> OptimizeHillClimbingAsync(
            int iterations,
            int maxGenerationsPerTrial,
            CancellationToken ct = default)
        {
            var currentConfig = _baseConfig.Clone();
            currentConfig.MaxGenerations = maxGenerationsPerTrial;
            currentConfig.RandomSeed = _rng.Next();

            var engine = new NeatGEvolutionEngine(currentConfig);
            var result = await engine.RunEvolutionAsync(
                _inputCount, _outputCount, _context, null, ct).ConfigureAwait(false);
            double currentFitness = result.BestGenome.Fitness;

            for (int iter = 0; iter < iterations; iter++)
            {
                ct.ThrowIfCancellationRequested();

                var neighborConfig = PerturbConfig(currentConfig);
                neighborConfig.MaxGenerations = maxGenerationsPerTrial;
                neighborConfig.RandomSeed = _rng.Next();

                var neighborEngine = new NeatGEvolutionEngine(neighborConfig);
                var neighborResult = await neighborEngine.RunEvolutionAsync(
                    _inputCount, _outputCount, _context, null, ct).ConfigureAwait(false);

                if (neighborResult.BestGenome.Fitness > currentFitness)
                {
                    currentConfig = neighborConfig;
                    currentFitness = neighborResult.BestGenome.Fitness;
                }
            }

            return (currentConfig, currentFitness);
        }

        /// <summary>
        /// Optimizes configuration using a genetic algorithm on parameter space.
        /// </summary>
        /// <param name="populationSize">Population size for config evolution.</param>
        /// <param name="generations">Number of generations for config evolution.</param>
        /// <param name="maxGenerationsPerTrial">Max generations per evolution trial.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Best configuration found and its performance.</returns>
        public async Task<(EvolutionConfig BestConfig, double BestFitness)> OptimizeWithGAAsync(
            int populationSize,
            int generations,
            int maxGenerationsPerTrial,
            CancellationToken ct = default)
        {
            var configPopulation = new List<(EvolutionConfig Config, double Fitness)>();

            for (int i = 0; i < populationSize; i++)
            {
                ct.ThrowIfCancellationRequested();

                var config = i == 0 ? _baseConfig.Clone() : GenerateRandomConfig();
                config.MaxGenerations = maxGenerationsPerTrial;
                config.RandomSeed = _rng.Next();

                var engine = new NeatGEvolutionEngine(config);
                var result = await engine.RunEvolutionAsync(
                    _inputCount, _outputCount, _context, null, ct).ConfigureAwait(false);

                configPopulation.Add((config, result.BestGenome.Fitness));
            }

            for (int gen = 0; gen < generations; gen++)
            {
                ct.ThrowIfCancellationRequested();

                configPopulation = configPopulation
                    .OrderByDescending(x => x.Fitness)
                    .ToList();

                var newPopulation = new List<(EvolutionConfig, double)>();

                int eliteCount = Math.Max(1, populationSize / 5);
                newPopulation.AddRange(configPopulation.Take(eliteCount));

                while (newPopulation.Count < populationSize)
                {
                    var parentA = configPopulation[_rng.Next(populationSize)].Config;
                    var parentB = configPopulation[_rng.Next(populationSize)].Config;

                    var child = CrossoverConfigs(parentA, parentB);
                    child = MutateConfig(child);
                    child.MaxGenerations = maxGenerationsPerTrial;
                    child.RandomSeed = _rng.Next();

                    var engine = new NeatGEvolutionEngine(child);
                    var result = await engine.RunEvolutionAsync(
                        _inputCount, _outputCount, _context, null, ct).ConfigureAwait(false);

                    newPopulation.Add((child, result.BestGenome.Fitness));
                }

                configPopulation = newPopulation;
            }

            var best = configPopulation.OrderByDescending(x => x.Fitness).First();
            return (best.Config, best.Fitness);
        }

        private EvolutionConfig GenerateRandomConfig()
        {
            var config = _baseConfig.Clone();
            config.PopulationSize = _rng.Next(100, 800);
            config.CrossoverRate = _rng.NextDouble() * 0.6 + 0.2;
            config.MutationRate = _rng.NextDouble() * 0.5 + 0.1;
            config.SpeciationThreshold = _rng.NextDouble() * 6 + 1;
            config.TargetSpeciesCount = _rng.Next(3, 25);
            config.TournamentSize = _rng.Next(2, 10);
            config.MaxStagnationGenerations = _rng.Next(10, 50);
            config.PerturbationMagnitude = _rng.NextDouble() * 0.3 + 0.01;
            config.MigrationRate = _rng.NextDouble() * 0.15;
            config.MigrationInterval = _rng.Next(3, 20);
            config.LandmarkCount = _rng.Next(10, 50);
            config.SemanticEmbeddingDimension = _rng.Next(16, 64);
            config.EliteFraction = _rng.NextDouble() * 0.1 + 0.01;
            return config;
        }

        private EvolutionConfig PerturbConfig(EvolutionConfig config)
        {
            var perturbed = config.Clone();
            string[] parameters = {
                nameof(EvolutionConfig.CrossoverRate),
                nameof(EvolutionConfig.MutationRate),
                nameof(EvolutionConfig.SpeciationThreshold),
                nameof(EvolutionConfig.TournamentSize),
                nameof(EvolutionConfig.PerturbationMagnitude),
                nameof(EvolutionConfig.MigrationRate),
                nameof(EvolutionConfig.MaxStagnationGenerations)
            };

            string param = parameters[_rng.Next(parameters.Length)];
            double perturbation = (_rng.NextDouble() * 0.4 - 0.2);

            switch (param)
            {
                case nameof(EvolutionConfig.CrossoverRate):
                    perturbed.CrossoverRate = Math.Clamp(perturbed.CrossoverRate + perturbation, 0.2, 0.95);
                    break;
                case nameof(EvolutionConfig.MutationRate):
                    perturbed.MutationRate = Math.Clamp(perturbed.MutationRate + perturbation, 0.05, 0.8);
                    break;
                case nameof(EvolutionConfig.SpeciationThreshold):
                    perturbed.SpeciationThreshold = Math.Clamp(perturbed.SpeciationThreshold + perturbation * 2, 0.5, 8);
                    break;
                case nameof(EvolutionConfig.TournamentSize):
                    perturbed.TournamentSize = Math.Clamp(perturbed.TournamentSize + (perturbation > 0 ? 1 : -1), 2, 15);
                    break;
                case nameof(EvolutionConfig.PerturbationMagnitude):
                    perturbed.PerturbationMagnitude = Math.Clamp(perturbed.PerturbationMagnitude + perturbation * 0.1, 0.001, 0.5);
                    break;
                case nameof(EvolutionConfig.MigrationRate):
                    perturbed.MigrationRate = Math.Clamp(perturbed.MigrationRate + perturbation * 0.05, 0, 0.2);
                    break;
                case nameof(EvolutionConfig.MaxStagnationGenerations):
                    perturbed.MaxStagnationGenerations = Math.Clamp(perturbed.MaxStagnationGenerations + (int)(perturbation * 10), 5, 100);
                    break;
            }

            return perturbed;
        }

        private EvolutionConfig CrossoverConfigs(EvolutionConfig a, EvolutionConfig b)
        {
            var child = a.Clone();
            if (_rng.NextDouble() < 0.5)
                child.CrossoverRate = b.CrossoverRate;
            if (_rng.NextDouble() < 0.5)
                child.MutationRate = b.MutationRate;
            if (_rng.NextDouble() < 0.5)
                child.SpeciationThreshold = b.SpeciationThreshold;
            if (_rng.NextDouble() < 0.5)
                child.TournamentSize = b.TournamentSize;
            if (_rng.NextDouble() < 0.5)
                child.PerturbationMagnitude = b.PerturbationMagnitude;
            if (_rng.NextDouble() < 0.5)
                child.MigrationRate = b.MigrationRate;
            if (_rng.NextDouble() < 0.5)
                child.MaxStagnationGenerations = b.MaxStagnationGenerations;
            return child;
        }

        private EvolutionConfig MutateConfig(EvolutionConfig config)
        {
            if (_rng.NextDouble() < 0.3)
                config.CrossoverRate = Math.Clamp(config.CrossoverRate + (_rng.NextDouble() - 0.5) * 0.2, 0.2, 0.95);
            if (_rng.NextDouble() < 0.3)
                config.MutationRate = Math.Clamp(config.MutationRate + (_rng.NextDouble() - 0.5) * 0.2, 0.05, 0.8);
            if (_rng.NextDouble() < 0.3)
                config.SpeciationThreshold = Math.Clamp(config.SpeciationThreshold + (_rng.NextDouble() - 0.5) * 2, 0.5, 8);
            return config;
        }
    }

}
