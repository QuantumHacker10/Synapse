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
    /// Automatically tunes evolution parameters based on observed performance.
    /// Uses Bayesian optimization principles to find optimal hyperparameter settings.
    /// Monitors fitness progress, diversity, and convergence to adjust parameters dynamically.
    /// </summary>
    public sealed class EvolutionAutoTuner
    {
        private readonly EvolutionConfig _config;
        private readonly Queue<AutoTunerObservation> _observations;
        private readonly Dictionary<string, ParameterRange> _parameterRanges;
        private int _tuningInterval;
        private int _observationsSinceTuning;

        /// <summary>
        /// Initializes a new instance of the EvolutionAutoTuner class.
        /// </summary>
        /// <param name="config">Evolution configuration to tune.</param>
        /// <param name="tuningInterval">Generations between tuning attempts.</param>
        public EvolutionAutoTuner(EvolutionConfig config, int tuningInterval = 20)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _observations = new Queue<AutoTunerObservation>();
            _parameterRanges = InitializeParameterRanges();
            _tuningInterval = tuningInterval;
            _observationsSinceTuning = 0;
        }

        /// <summary>Gets the number of observations collected.</summary>
        public int ObservationCount => _observations.Count;

        /// <summary>
        /// Records an observation of evolution performance with current parameters.
        /// </summary>
        /// <param name="metrics">Current evolution metrics.</param>
        /// <param name="config">Current configuration used.</param>
        public void RecordObservation(EvolutionMetrics metrics, EvolutionConfig config)
        {
            _observations.Enqueue(new AutoTunerObservation
            {
                Timestamp = DateTime.UtcNow,
                Generation = metrics.Generation,
                BestFitness = metrics.BestFitness,
                AverageFitness = metrics.AverageFitness,
                FitnessImprovement = metrics.FitnessImprovement,
                SpeciesCount = metrics.SpeciesCount,
                DiversityMetric = metrics.DiversityMetric,
                MutationSuccessRate = metrics.MutationSuccessRate,
                CrossoverSuccessRate = metrics.CrossoverSuccessRate,
                AverageComplexity = metrics.AverageComplexity,
                CrossoverRate = config.CrossoverRate,
                MutationRate = config.MutationRate,
                SpeciationThreshold = config.SpeciationThreshold,
                TournamentSize = config.TournamentSize,
                PerturbationMagnitude = config.PerturbationMagnitude,
                MigrationRate = config.MigrationRate,
                PopulationSize = config.PopulationSize
            });

            _observationsSinceTuning++;
        }

        /// <summary>
        /// Analyzes collected observations and suggests parameter adjustments.
        /// </summary>
        /// <returns>Recommended parameter adjustments.</returns>
        public AutoTunerRecommendations AnalyzeAndRecommend()
        {
            var recentObservations = _observations.TakeLast(50).ToList();
            if (recentObservations.Count < 5)
            {
                return new AutoTunerRecommendations { Confidence = 0, Changes = new List<ParameterChange>() };
            }

            var recommendations = new List<ParameterChange>();

            var fitnessTrend = ComputeFitnessTrend(recentObservations);
            var diversityTrend = ComputeDiversityTrend(recentObservations);
            var mutationEffectiveness = ComputeMutationEffectiveness(recentObservations);
            var convergenceRate = ComputeConvergenceRate(recentObservations);

            if (fitnessTrend < 0.001 && diversityTrend < -0.05)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.MutationRate),
                    CurrentValue = _config.MutationRate,
                    RecommendedValue = Math.Min(0.5, _config.MutationRate * 1.3),
                    Reason = "Fitness stagnation with declining diversity"
                });

                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.MigrationRate),
                    CurrentValue = _config.MigrationRate,
                    RecommendedValue = Math.Min(0.2, _config.MigrationRate * 1.5),
                    Reason = "Increase migration to boost diversity"
                });
            }

            if (fitnessTrend > 0.01 && diversityTrend > 0.1)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.MutationRate),
                    CurrentValue = _config.MutationRate,
                    RecommendedValue = Math.Max(0.05, _config.MutationRate * 0.9),
                    Reason = "Good progress; slightly reduce mutation for exploitation"
                });
            }

            if (mutationEffectiveness < 0.05)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.PerturbationMagnitude),
                    CurrentValue = _config.PerturbationMagnitude,
                    RecommendedValue = Math.Min(0.5, _config.PerturbationMagnitude * 1.5),
                    Reason = "Low mutation effectiveness; increase perturbation magnitude"
                });
            }

            if (recentObservations.Average(o => o.SpeciesCount) > _config.TargetSpeciesCount * 1.5)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.SpeciationThreshold),
                    CurrentValue = _config.SpeciationThreshold,
                    RecommendedValue = Math.Min(_config.MaxSpeciationThreshold,
                        _config.SpeciationThreshold * 1.2),
                    Reason = "Too many species; increase threshold to merge species"
                });
            }
            else if (recentObservations.Average(o => o.SpeciesCount) < _config.TargetSpeciesCount * 0.5)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.SpeciationThreshold),
                    CurrentValue = _config.SpeciationThreshold,
                    RecommendedValue = Math.Max(_config.MinSpeciationThreshold,
                        _config.SpeciationThreshold * 0.8),
                    Reason = "Too few species; decrease threshold to create more species"
                });
            }

            if (convergenceRate < 0.0001 && recentObservations.Count > 20)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.CrossoverRate),
                    CurrentValue = _config.CrossoverRate,
                    RecommendedValue = Math.Max(0.3, _config.CrossoverRate - 0.05),
                    Reason = "Very slow convergence; reduce crossover to increase exploration via mutation"
                });
            }

            double avgTournamentSize = recentObservations.Average(o => o.TournamentSize);
            if (avgTournamentSize > 7 && diversityTrend < -0.1)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.TournamentSize),
                    CurrentValue = _config.TournamentSize,
                    RecommendedValue = Math.Max(2, _config.TournamentSize - 1),
                    Reason = "High selection pressure causing diversity loss"
                });
            }

            double confidence = ComputeRecommendationConfidence(recentObservations, recommendations);

            return new AutoTunerRecommendations
            {
                Confidence = confidence,
                Changes = recommendations.AsReadOnly(),
                FitnessTrend = fitnessTrend,
                DiversityTrend = diversityTrend,
                ConvergenceRate = convergenceRate,
                ObservationCount = recentObservations.Count
            };
        }

        /// <summary>
        /// Applies recommended changes to the configuration.
        /// </summary>
        /// <param name="recommendations">Recommendations to apply.</param>
        /// <param name="maxChangesPerIteration">Maximum number of changes to apply.</param>
        /// <returns>Number of changes actually applied.</returns>
        public int ApplyRecommendations(AutoTunerRecommendations recommendations, int maxChangesPerIteration = 2)
        {
            if (recommendations.Confidence < 0.3)
                return 0;

            int applied = 0;
            foreach (var change in recommendations.Changes.Take(maxChangesPerIteration))
            {
                if (ApplyParameterChange(change))
                {
                    applied++;
                }
            }
            _observationsSinceTuning = 0;
            return applied;
        }

        /// <summary>
        /// Gets the tuning history.
        /// </summary>
        public IReadOnlyList<AutoTunerObservation> GetObservations()
        {
            return _observations.ToList().AsReadOnly();
        }

        /// <summary>
        /// Resets the auto-tuner state.
        /// </summary>
        public void Reset()
        {
            _observations.Clear();
            _observationsSinceTuning = 0;
        }

        private bool ApplyParameterChange(ParameterChange change)
        {
            switch (change.ParameterName)
            {
                case nameof(EvolutionConfig.MutationRate):
                    _config.MutationRate = Math.Clamp(change.RecommendedValue, 0.01, 0.8);
                    return true;
                case nameof(EvolutionConfig.CrossoverRate):
                    _config.CrossoverRate = Math.Clamp(change.RecommendedValue, 0.1, 0.95);
                    return true;
                case nameof(EvolutionConfig.SpeciationThreshold):
                    _config.SpeciationThreshold = Math.Clamp(change.RecommendedValue,
                        _config.MinSpeciationThreshold, _config.MaxSpeciationThreshold);
                    return true;
                case nameof(EvolutionConfig.TournamentSize):
                    _config.TournamentSize = Math.Clamp((int)change.RecommendedValue, 2, 15);
                    return true;
                case nameof(EvolutionConfig.PerturbationMagnitude):
                    _config.PerturbationMagnitude = Math.Clamp(change.RecommendedValue, 0.001, 1.0);
                    return true;
                case nameof(EvolutionConfig.MigrationRate):
                    _config.MigrationRate = Math.Clamp(change.RecommendedValue, 0.0, 0.3);
                    return true;
                default:
                    return false;
            }
        }

        private double ComputeFitnessTrend(List<AutoTunerObservation> observations)
        {
            if (observations.Count < 2)
                return 0;

            var fitnesses = observations.Select(o => o.BestFitness).ToList();
            return (fitnesses[^1] - fitnesses[0]) / Math.Max(1, Math.Abs(fitnesses[0]));
        }

        private double ComputeDiversityTrend(List<AutoTunerObservation> observations)
        {
            if (observations.Count < 2)
                return 0;

            var diversities = observations.Select(o => o.DiversityMetric).ToList();
            return (diversities[^1] - diversities[0]) / Math.Max(0.01, diversities[0]);
        }

        private double ComputeMutationEffectiveness(List<AutoTunerObservation> observations)
        {
            return observations.Average(o => o.MutationSuccessRate);
        }

        private double ComputeConvergenceRate(List<AutoTunerObservation> observations)
        {
            if (observations.Count < 5)
                return 0;

            var recent = observations.TakeLast(5).ToList();
            var improvements = new List<double>();
            for (int i = 1; i < recent.Count; i++)
            {
                improvements.Add(recent[i].BestFitness - recent[i - 1].BestFitness);
            }

            return improvements.Count > 0 ? improvements.Average() : 0;
        }

        private double ComputeRecommendationConfidence(
            List<AutoTunerObservation> observations,
            List<ParameterChange> recommendations)
        {
            if (recommendations.Count == 0)
                return 0;

            double dataQuality = Math.Min(1.0, observations.Count / 30.0);
            double signalStrength = Math.Min(1.0, recommendations.Count * 0.3);
            double consistency = ComputeObservationConsistency(observations);

            return 0.4 * dataQuality + 0.3 * signalStrength + 0.3 * consistency;
        }

        private double ComputeObservationConsistency(List<AutoTunerObservation> observations)
        {
            if (observations.Count < 3)
                return 0.5;

            var fitnesses = observations.Select(o => o.BestFitness).ToList();
            int monotonicCount = 0;
            for (int i = 1; i < fitnesses.Count; i++)
            {
                if (fitnesses[i] >= fitnesses[i - 1])
                    monotonicCount++;
            }

            return (double)monotonicCount / (fitnesses.Count - 1);
        }

        private Dictionary<string, ParameterRange> InitializeParameterRanges()
        {
            return new Dictionary<string, ParameterRange>
            {
                [nameof(EvolutionConfig.MutationRate)] = new ParameterRange(0.01, 0.8, 0.25),
                [nameof(EvolutionConfig.CrossoverRate)] = new ParameterRange(0.1, 0.95, 0.75),
                [nameof(EvolutionConfig.SpeciationThreshold)] = new ParameterRange(0.5, 8.0, 3.0),
                [nameof(EvolutionConfig.TournamentSize)] = new ParameterRange(2, 15, 5),
                [nameof(EvolutionConfig.PerturbationMagnitude)] = new ParameterRange(0.001, 1.0, 0.1),
                [nameof(EvolutionConfig.MigrationRate)] = new ParameterRange(0.0, 0.3, 0.05),
                [nameof(EvolutionConfig.PopulationSize)] = new ParameterRange(50, 2000, 300)
            };
        }
    }

    /// <summary>
    /// Observation record for auto-tuning.
    /// </summary>
    public sealed class AutoTunerObservation
    {
        /// <summary>Timestamp of observation.</summary>
        public DateTime Timestamp { get; init; }

        /// <summary>Generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Best fitness.</summary>
        public double BestFitness { get; init; }

        /// <summary>Average fitness.</summary>
        public double AverageFitness { get; init; }

        /// <summary>Fitness improvement over previous generation.</summary>
        public double FitnessImprovement { get; init; }

        /// <summary>Number of species.</summary>
        public int SpeciesCount { get; init; }

        /// <summary>Diversity metric.</summary>
        public double DiversityMetric { get; init; }

        /// <summary>Mutation success rate.</summary>
        public double MutationSuccessRate { get; init; }

        /// <summary>Crossover success rate.</summary>
        public double CrossoverSuccessRate { get; init; }

        /// <summary>Average complexity.</summary>
        public double AverageComplexity { get; init; }

        /// <summary>Configuration parameters at observation time.</summary>
        public double CrossoverRate { get; init; }
        public double MutationRate { get; init; }
        public double SpeciationThreshold { get; init; }
        public int TournamentSize { get; init; }
        public double PerturbationMagnitude { get; init; }
        public double MigrationRate { get; init; }
        public int PopulationSize { get; init; }
    }

    /// <summary>
    /// Recommendations from the auto-tuner.
    /// </summary>
    public sealed class AutoTunerRecommendations
    {
        /// <summary>Confidence in the recommendations (0-1).</summary>
        public double Confidence { get; init; }

        /// <summary>Recommended parameter changes.</summary>
        public IReadOnlyList<ParameterChange> Changes { get; init; } = Array.Empty<ParameterChange>();

        /// <summary>Observed fitness trend.</summary>
        public double FitnessTrend { get; init; }

        /// <summary>Observed diversity trend.</summary>
        public double DiversityTrend { get; init; }

        /// <summary>Observed convergence rate.</summary>
        public double ConvergenceRate { get; init; }

        /// <summary>Number of observations used.</summary>
        public int ObservationCount { get; init; }
    }

    /// <summary>
    /// A single parameter change recommendation.
    /// </summary>
    public sealed class ParameterChange
    {
        /// <summary>Name of the parameter.</summary>
        public string ParameterName { get; init; } = string.Empty;

        /// <summary>Current value.</summary>
        public double CurrentValue { get; init; }

        /// <summary>Recommended value.</summary>
        public double RecommendedValue { get; init; }

        /// <summary>Reason for the change.</summary>
        public string Reason { get; init; } = string.Empty;

        /// <inheritdoc/>
        public override string ToString() =>
            $"{ParameterName}: {CurrentValue:F3} -> {RecommendedValue:F3} ({Reason})";
    }

    /// <summary>
    /// Range definition for a parameter.
    /// </summary>
    public sealed class ParameterRange
    {
        /// <summary>Minimum allowed value.</summary>
        public double Min { get; init; }

        /// <summary>Maximum allowed value.</summary>
        public double Max { get; init; }

        /// <summary>Default value.</summary>
        public double Default { get; init; }

        /// <summary>
        /// Initializes a new ParameterRange.
        /// </summary>
        public ParameterRange(double min, double max, double @default)
        {
            Min = min;
            Max = max;
            Default = @default;
        }

        /// <summary>Clamps a value to the valid range.</summary>
        public double Clamp(double value) => Math.Clamp(value, Min, Max);

        /// <summary>Normalizes a value to [0, 1].</summary>
        public double Normalize(double value) =>
            (Math.Clamp(value, Min, Max) - Min) / Math.Max(1e-10, Max - Min);

        /// <summary>Denormalizes a value from [0, 1].</summary>
        public double Denormalize(double normalized) =>
            Min + Math.Clamp(normalized, 0, 1) * (Max - Min);
    }

}
