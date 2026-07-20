// =============================================================================
// NeatGEvolutionEngine.EvolutionExperimentFramework.cs — NEAT-G partial module
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
    /// Framework for running controlled evolution experiments with
    /// parameter sweeps, ablation studies, and statistical analysis.
    /// </summary>
    public sealed class EvolutionExperimentFramework
    {
        private readonly EvolutionConfig _baseConfig;
        private readonly Random _rng;

        /// <summary>
        /// Initializes a new instance of the EvolutionExperimentFramework class.
        /// </summary>
        public EvolutionExperimentFramework(EvolutionConfig baseConfig)
        {
            _baseConfig = baseConfig ?? throw new ArgumentNullException(nameof(baseConfig));
            _rng = Random.Shared;
        }

        /// <summary>
        /// Runs a parameter sweep experiment.
        /// </summary>
        public async Task<IReadOnlyList<ExperimentResult>> RunParameterSweepAsync(
            string parameterName,
            IReadOnlyList<double> parameterValues,
            Func<EvolutionConfig, Task<GeoGenome>> evolutionRunner,
            CancellationToken cancellationToken = default)
        {
            var results = new List<ExperimentResult>();

            foreach (var value in parameterValues)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var config = ApplyParameter(_baseConfig, parameterName, value);
                var stopwatch = Stopwatch.StartNew();

                try
                {
                    var bestGenome = await evolutionRunner(config);
                    stopwatch.Stop();

                    results.Add(new ExperimentResult
                    {
                        ParameterName = parameterName,
                        ParameterValue = value,
                        BestFitness = bestGenome.Fitness,
                        BestGenome = bestGenome,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = true,
                        Error = null
                    });
                }
                catch (Exception ex)
                {
                    stopwatch.Stop();
                    results.Add(new ExperimentResult
                    {
                        ParameterName = parameterName,
                        ParameterValue = value,
                        BestFitness = double.MinValue,
                        BestGenome = null,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = false,
                        Error = ex.Message
                    });
                }
            }

            return results.AsReadOnly();
        }

        /// <summary>
        /// Runs an ablation study by removing one component at a time.
        /// </summary>
        public async Task<IReadOnlyList<ExperimentResult>> RunAblationStudyAsync(
            IReadOnlyList<string> componentNames,
            Func<EvolutionConfig, Task<GeoGenome>> evolutionRunner,
            CancellationToken cancellationToken = default)
        {
            var baselineResult = await evolutionRunner(_baseConfig);
            double baselineFitness = baselineResult.Fitness;

            var results = new List<ExperimentResult>();

            results.Add(new ExperimentResult
            {
                ParameterName = "Baseline",
                ParameterValue = 1.0,
                BestFitness = baselineFitness,
                BestGenome = baselineResult,
                ExecutionTime = TimeSpan.Zero,
                IsSuccess = true,
                Error = null
            });

            foreach (var component in componentNames)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var config = RemoveComponent(_baseConfig, component);
                var stopwatch = Stopwatch.StartNew();

                try
                {
                    var bestGenome = await evolutionRunner(config);
                    stopwatch.Stop();

                    double fitnessDrop = baselineFitness - bestGenome.Fitness;

                    results.Add(new ExperimentResult
                    {
                        ParameterName = $"Ablated: {component}",
                        ParameterValue = fitnessDrop,
                        BestFitness = bestGenome.Fitness,
                        BestGenome = bestGenome,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = true,
                        Error = null
                    });
                }
                catch (Exception ex)
                {
                    stopwatch.Stop();
                    results.Add(new ExperimentResult
                    {
                        ParameterName = $"Ablated: {component}",
                        ParameterValue = double.MinValue,
                        BestFitness = double.MinValue,
                        BestGenome = null,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = false,
                        Error = ex.Message
                    });
                }
            }

            return results.AsReadOnly();
        }

        /// <summary>
        /// Runs multiple trials of the same configuration for statistical analysis.
        /// </summary>
        public async Task<StatisticalAnalysisResult> RunStatisticalAnalysisAsync(
            Func<EvolutionConfig, Task<GeoGenome>> evolutionRunner,
            int trialCount = 10,
            CancellationToken cancellationToken = default)
        {
            var fitnesses = new List<double>();
            var times = new List<TimeSpan>();
            var genomes = new List<GeoGenome>();
            var errors = new List<string>();

            for (int i = 0; i < trialCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var stopwatch = Stopwatch.StartNew();
                try
                {
                    var genome = await evolutionRunner(_baseConfig);
                    stopwatch.Stop();

                    fitnesses.Add(genome.Fitness);
                    times.Add(stopwatch.Elapsed);
                    genomes.Add(genome);
                }
                catch (Exception ex)
                {
                    stopwatch.Stop();
                    errors.Add(ex.Message);
                    times.Add(stopwatch.Elapsed);
                }
            }

            return new StatisticalAnalysisResult
            {
                TrialCount = trialCount,
                SuccessfulTrials = fitnesses.Count,
                FailedTrials = errors.Count,
                MeanFitness = fitnesses.Count > 0 ? fitnesses.Average() : double.MinValue,
                MedianFitness = fitnesses.Count > 0 ? fitnesses.Median() : double.MinValue,
                StdDevFitness = fitnesses.Count > 1 ? fitnesses.StandardDeviation() : 0,
                MinFitness = fitnesses.Count > 0 ? fitnesses.Min() : double.MinValue,
                MaxFitness = fitnesses.Count > 0 ? fitnesses.Max() : double.MinValue,
                MeanTime = times.Count > 0 ? TimeSpan.FromMilliseconds(times.Average(t => t.TotalMilliseconds)) : TimeSpan.Zero,
                BestGenome = fitnesses.Count > 0 ? genomes[fitnesses.IndexOf(fitnesses.Max())] : null,
                Errors = errors.AsReadOnly(),
                FitnessValues = fitnesses.AsReadOnly()
            };
        }

        /// <summary>
        /// Runs a grid search over multiple parameters.
        /// </summary>
        public async Task<IReadOnlyList<ExperimentResult>> RunGridSearchAsync(
            IReadOnlyDictionary<string, IReadOnlyList<double>> parameterGrid,
            Func<EvolutionConfig, Task<GeoGenome>> evolutionRunner,
            CancellationToken cancellationToken = default)
        {
            var parameterNames = parameterGrid.Keys.ToList();
            var results = new List<ExperimentResult>();

            var combinations = GenerateCombinations(parameterGrid);

            foreach (var combination in combinations)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var config = _baseConfig;
                foreach (var kvp in combination)
                {
                    config = ApplyParameter(config, kvp.Key, kvp.Value);
                }

                var paramDescription = string.Join(", ", combination.Select(kvp => $"{kvp.Key}={kvp.Value}"));
                var stopwatch = Stopwatch.StartNew();

                try
                {
                    var bestGenome = await evolutionRunner(config);
                    stopwatch.Stop();

                    results.Add(new ExperimentResult
                    {
                        ParameterName = paramDescription,
                        ParameterValue = bestGenome.Fitness,
                        BestFitness = bestGenome.Fitness,
                        BestGenome = bestGenome,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = true,
                        Error = null
                    });
                }
                catch (Exception ex)
                {
                    stopwatch.Stop();
                    results.Add(new ExperimentResult
                    {
                        ParameterName = paramDescription,
                        ParameterValue = double.MinValue,
                        BestFitness = double.MinValue,
                        BestGenome = null,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = false,
                        Error = ex.Message
                    });
                }
            }

            return results.OrderByDescending(r => r.BestFitness).ToList().AsReadOnly();
        }

        private EvolutionConfig ApplyParameter(EvolutionConfig config, string parameterName, double value)
        {
            var clone = config.Clone();
            switch (parameterName.ToLowerInvariant())
            {
                case "populationsize":
                    clone.PopulationSize = (int)value;
                    break;
                case "mutationrate":
                    clone.MutationRate = value;
                    break;
                case "crossoverrate":
                    clone.CrossoverRate = value;
                    break;
                case "maxgenerations":
                    clone.MaxGenerations = (int)value;
                    break;
                case "elitismcount":
                    clone.ElitismCount = (int)value;
                    break;
                case "speciationthreshold":
                    clone.SpeciesCompatibilityThreshold = value;
                    break;
            }
            return clone;
        }

        private EvolutionConfig RemoveComponent(EvolutionConfig config, string componentName)
        {
            var clone = config.Clone();
            switch (componentName.ToLowerInvariant())
            {
                case "speciation":
                    clone.SpeciationMethod = SpeciationMethod.None;
                    break;
                case "crossover":
                    clone.CrossoverRate = 0;
                    break;
            }
            return clone;
        }

        private IReadOnlyList<IReadOnlyDictionary<string, double>> GenerateCombinations(
            IReadOnlyDictionary<string, IReadOnlyList<double>> parameterGrid)
        {
            var keys = parameterGrid.Keys.ToList();
            var combinations = new List<IReadOnlyDictionary<string, double>>();

            if (keys.Count == 0)
            {
                return new List<IReadOnlyDictionary<string, double>> { new Dictionary<string, double>() };
            }

            var current = new Dictionary<string, double>();
            GenerateCombinationsRecursive(parameterGrid, keys, 0, current, combinations);

            return combinations;
        }

        private void GenerateCombinationsRecursive(
            IReadOnlyDictionary<string, IReadOnlyList<double>> parameterGrid,
            List<string> keys,
            int index,
            Dictionary<string, double> current,
            List<IReadOnlyDictionary<string, double>> results)
        {
            if (index == keys.Count)
            {
                results.Add(new Dictionary<string, double>(current));
                return;
            }

            string key = keys[index];
            foreach (double value in parameterGrid[key])
            {
                current[key] = value;
                GenerateCombinationsRecursive(parameterGrid, keys, index + 1, current, results);
            }
        }
    }

    /// <summary>
    /// Result of an evolution experiment.
    /// </summary>
    public sealed class ExperimentResult
    {
        /// <summary>Parameter name tested.</summary>
        public string ParameterName { get; init; } = string.Empty;
        /// <summary>Parameter value used.</summary>
        public double ParameterValue { get; init; }
        /// <summary>Best fitness achieved.</summary>
        public double BestFitness { get; init; }
        /// <summary>Best genome found.</summary>
        public GeoGenome? BestGenome { get; init; }
        /// <summary>Execution time.</summary>
        public TimeSpan ExecutionTime { get; init; }
        /// <summary>Whether the experiment succeeded.</summary>
        public bool IsSuccess { get; init; }
        /// <summary>Error message if failed.</summary>
        public string? Error { get; init; }
    }

    /// <summary>
    /// Result of statistical analysis over multiple trials.
    /// </summary>
    public sealed class StatisticalAnalysisResult
    {
        /// <summary>Total number of trials.</summary>
        public int TrialCount { get; init; }
        /// <summary>Number of successful trials.</summary>
        public int SuccessfulTrials { get; init; }
        /// <summary>Number of failed trials.</summary>
        public int FailedTrials { get; init; }
        /// <summary>Mean fitness across trials.</summary>
        public double MeanFitness { get; init; }
        /// <summary>Median fitness across trials.</summary>
        public double MedianFitness { get; init; }
        /// <summary>Standard deviation of fitness.</summary>
        public double StdDevFitness { get; init; }
        /// <summary>Minimum fitness.</summary>
        public double MinFitness { get; init; }
        /// <summary>Maximum fitness.</summary>
        public double MaxFitness { get; init; }
        /// <summary>Mean execution time.</summary>
        public TimeSpan MeanTime { get; init; }
        /// <summary>Best genome across trials.</summary>
        public GeoGenome? BestGenome { get; init; }
        /// <summary>Error messages.</summary>
        public IReadOnlyList<string> Errors { get; init; } = Array.Empty<string>();
        /// <summary>All fitness values.</summary>
        public IReadOnlyList<double> FitnessValues { get; init; } = Array.Empty<double>();
    }

}
