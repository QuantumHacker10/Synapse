// =============================================================================
// NeatGEvolutionEngine.EvolutionAnalyticsDashboard.cs — NEAT-G partial module
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
    /// Provides comprehensive analytics and reporting for evolution runs.
    /// Generates detailed reports, statistics, and performance metrics.
    /// </summary>
    public sealed class EvolutionAnalyticsDashboard
    {
        private readonly EvolutionConfig _config;
        private readonly EvolutionHistoryTracker _history;
        private readonly SpeciationAnalytics _speciationAnalytics;
        private readonly EvolutionDiagnostics _diagnostics;

        /// <summary>
        /// Initializes a new instance of the EvolutionAnalyticsDashboard class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="history">Evolution history tracker.</param>
        /// <param name="speciationAnalytics">Speciation analytics.</param>
        /// <param name="diagnostics">Evolution diagnostics.</param>
        public EvolutionAnalyticsDashboard(
            EvolutionConfig config,
            EvolutionHistoryTracker history,
            SpeciationAnalytics speciationAnalytics,
            EvolutionDiagnostics diagnostics)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _history = history ?? throw new ArgumentNullException(nameof(history));
            _speciationAnalytics = speciationAnalytics ?? throw new ArgumentNullException(nameof(speciationAnalytics));
            _diagnostics = diagnostics ?? throw new ArgumentNullException(nameof(diagnostics));
        }

        /// <summary>
        /// Generates a comprehensive evolution report.
        /// </summary>
        /// <returns>A detailed report string.</returns>
        public string GenerateReport()
        {
            var summary = _history.GetSummary();
            var metrics = _history.GetMetricsHistory();
            var mutationRates = _diagnostics.GetMutationSuccessRates();
            var crossoverRates = _diagnostics.GetCrossoverSuccessRates();

            var sb = new StringBuilder();
            sb.AppendLine("=== NEAT-G Evolution Report ===");
            sb.AppendLine($"Total Generations: {summary.TotalGenerations}");
            sb.AppendLine($"Total Evaluations: {summary.TotalEvaluations:N0}");
            sb.AppendLine($"Best Fitness Ever: {summary.BestFitnessEver:F6}");
            sb.AppendLine($"Best Fitness Generation: {summary.BestFitnessGeneration}");
            sb.AppendLine($"Initial Fitness: {summary.InitialFitness:F6}");
            sb.AppendLine($"Final Fitness: {summary.FinalFitness:F6}");
            sb.AppendLine($"Fitness Improvement: {summary.FitnessImprovement:F6}");
            sb.AppendLine($"Peak Species Count: {summary.PeakSpeciesCount}");
            sb.AppendLine($"Final Species Count: {summary.FinalSpeciesCount}");
            sb.AppendLine($"Average Diversity: {summary.AverageDiversity:F4}");
            sb.AppendLine($"Total Events: {summary.TotalEvents}");
            sb.AppendLine();

            if (metrics.Count > 0)
            {
                sb.AppendLine("--- Fitness Progression ---");
                int reportInterval = Math.Max(1, metrics.Count / 20);
                for (int i = 0; i < metrics.Count; i += reportInterval)
                {
                    var m = metrics[i];
                    sb.AppendLine($"Gen {m.Generation,5}: Best={m.BestFitness:F4}, Avg={m.AverageFitness:F4}, " +
                                 $"Species={m.SpeciesCount}, Diversity={m.DiversityMetric:F3}, " +
                                 $"Evals={m.EvaluationsThisGeneration}");
                }
                sb.AppendLine();
            }

            if (mutationRates.Count > 0)
            {
                sb.AppendLine("--- Mutation Success Rates ---");
                foreach (var kvp in mutationRates.OrderByDescending(k => k.Value))
                {
                    sb.AppendLine($"  {kvp.Key}: {kvp.Value:P1}");
                }
                sb.AppendLine();
            }

            if (crossoverRates.Count > 0)
            {
                sb.AppendLine("--- Crossover Success Rates ---");
                foreach (var kvp in crossoverRates.OrderByDescending(k => k.Value))
                {
                    sb.AppendLine($"  {kvp.Key}: {kvp.Value:P1}");
                }
                sb.AppendLine();
            }

            sb.AppendLine("--- Configuration ---");
            sb.AppendLine($"  Population Size: {_config.PopulationSize}");
            sb.AppendLine($"  Max Generations: {_config.MaxGenerations}");
            sb.AppendLine($"  Crossover Rate: {_config.CrossoverRate}");
            sb.AppendLine($"  Mutation Rate: {_config.MutationRate}");
            sb.AppendLine($"  Speciation Threshold: {_config.SpeciationThreshold}");
            sb.AppendLine($"  Target Species: {_config.TargetSpeciesCount}");
            sb.AppendLine($"  Selection Method: {_config.ParentSelection}");
            sb.AppendLine($"  Speciation Method: {_config.SpeciationMethod}");

            return sb.ToString();
        }

        /// <summary>
        /// Generates convergence analysis data.
        /// </summary>
        /// <returns>Convergence analysis results.</returns>
        public ConvergenceAnalysis AnalyzeConvergence()
        {
            var metrics = _history.GetMetricsHistory();
            if (metrics.Count < 2)
            {
                return new ConvergenceAnalysis { HasConverged = false };
            }

            var fitnesses = metrics.Select(m => m.BestFitness).ToList();
            var recentWindow = fitnesses.Skip(Math.Max(0, fitnesses.Count - 20)).ToList();

            double recentMean = recentWindow.Average();
            double recentVariance = recentWindow.Average(f => (f - recentMean) * (f - recentMean));
            double recentStdDev = Math.Sqrt(recentVariance);

            double overallRange = fitnesses.Max() - fitnesses.Min();
            double convergenceRatio = overallRange > 0 ? recentStdDev / overallRange : 0;

            bool hasConverged = convergenceRatio < 0.01 && recentWindow.Count >= 10;

            int stagnationStart = -1;
            double plateauFitness = 0;
            for (int i = fitnesses.Count - 1; i >= 1; i--)
            {
                if (Math.Abs(fitnesses[i] - fitnesses[i - 1]) < _config.FitnessThreshold)
                {
                    if (stagnationStart < 0)
                        stagnationStart = i;
                    plateauFitness = fitnesses[i];
                }
                else
                {
                    break;
                }
            }

            double fitnessGrowthRate = 0;
            if (metrics.Count >= 10)
            {
                var first10 = metrics.Take(10).Select(m => m.BestFitness).ToList();
                var last10 = metrics.TakeLast(10).Select(m => m.BestFitness).ToList();
                fitnessGrowthRate = (last10.Average() - first10.Average()) / Math.Max(1, first10.Average());
            }

            return new ConvergenceAnalysis
            {
                HasConverged = hasConverged,
                ConvergenceRatio = convergenceRatio,
                RecentStdDev = recentStdDev,
                OverallRange = overallRange,
                StagnationStart = stagnationStart,
                PlateauFitness = plateauFitness,
                FitnessGrowthRate = fitnessGrowthRate,
                EstimatedRemainingGenerations = EstimateRemainingGenerations(metrics)
            };
        }

        /// <summary>
        /// Generates species dynamics analysis.
        /// </summary>
        /// <returns>Species dynamics data.</returns>
        public SpeciesDynamicsAnalysis AnalyzeSpeciesDynamics()
        {
            var snapshots = _speciationAnalytics.GetSnapshots();
            var speciesOverTime = _speciationAnalytics.GetSpeciesCountOverTime();

            double speciationRate = _speciationAnalytics.ComputeSpeciationRate();
            double extinctionRate = _speciationAnalytics.ComputeExtinctionRate();

            int peakSpecies = speciesOverTime.Count > 0 ? speciesOverTime.Max(s => s.SpeciesCount) : 0;
            int minSpecies = speciesOverTime.Count > 0 ? speciesOverTime.Min(s => s.SpeciesCount) : 0;
            double avgSpecies = speciesOverTime.Count > 0 ? speciesOverTime.Average(s => s.SpeciesCount) : 0;

            double speciesStability = 0;
            if (speciesOverTime.Count > 1)
            {
                var counts = speciesOverTime.Select(s => s.SpeciesCount).ToList();
                double mean = counts.Average();
                double variance = counts.Average(c => (c - mean) * (c - mean));
                speciesStability = mean > 0 ? 1.0 - Math.Sqrt(variance) / mean : 0;
            }

            var fitnessBySpecies = new Dictionary<int, List<double>>();
            foreach (var snapshot in snapshots)
            {
                for (int i = 0; i < snapshot.SpeciesBestFitness.Length; i++)
                {
                    int speciesId = i;
                    if (!fitnessBySpecies.ContainsKey(speciesId))
                        fitnessBySpecies[speciesId] = new List<double>();
                    fitnessBySpecies[speciesId].Add(snapshot.SpeciesBestFitness[i]);
                }
            }

            var speciesDominance = fitnessBySpecies
                .Select(kvp => new
                {
                    SpeciesId = kvp.Key,
                    MaxFitness = kvp.Value.Max(),
                    AvgFitness = kvp.Value.Average(),
                    Longevity = kvp.Value.Count
                })
                .OrderByDescending(s => s.MaxFitness)
                .ToList();

            return new SpeciesDynamicsAnalysis
            {
                SpeciationRate = speciationRate,
                ExtinctionRate = extinctionRate,
                PeakSpeciesCount = peakSpecies,
                MinSpeciesCount = minSpecies,
                AverageSpeciesCount = avgSpecies,
                SpeciesStability = speciesStability,
                SpeciesDominanceRanking = speciesDominance.Select(s => s.SpeciesId).ToList().AsReadOnly()
            };
        }

        /// <summary>
        /// Generates mutation effectiveness analysis.
        /// </summary>
        /// <returns>Mutation effectiveness data.</returns>
        public MutationEffectivenessAnalysis AnalyzeMutationEffectiveness()
        {
            var mutationRates = _diagnostics.GetMutationSuccessRates();
            var crossoverRates = _diagnostics.GetCrossoverSuccessRates();

            var snapshots = _diagnostics.GetSnapshots();
            var diversityOverTime = snapshots.Select(s => s.PopulationDiversity).ToList();
            var structuralOverTime = snapshots.Select(s => s.StructuralDiversity).ToList();

            double avgMutationRate = mutationRates.Count > 0 ? mutationRates.Values.Average() : 0;
            double avgCrossoverRate = crossoverRates.Count > 0 ? crossoverRates.Values.Average() : 0;

            var mostEffectiveMutation = mutationRates.Count > 0
                ? mutationRates.OrderByDescending(k => k.Value).First()
                : new KeyValuePair<MutationType, double>(MutationType.None, 0);

            var leastEffectiveMutation = mutationRates.Count > 0
                ? mutationRates.OrderBy(k => k.Value).First()
                : new KeyValuePair<MutationType, double>(MutationType.None, 0);

            double diversityTrend = 0;
            if (diversityOverTime.Count >= 10)
            {
                var firstHalf = diversityOverTime.Take(diversityOverTime.Count / 2).Average();
                var secondHalf = diversityOverTime.Skip(diversityOverTime.Count / 2).Average();
                diversityTrend = secondHalf - firstHalf;
            }

            double weightDiversity = 0;
            if (snapshots.Count > 0)
            {
                weightDiversity = snapshots.Average(s => s.WeightDiversity);
            }

            return new MutationEffectivenessAnalysis
            {
                AverageMutationSuccessRate = avgMutationRate,
                AverageCrossoverSuccessRate = avgCrossoverRate,
                MostEffectiveMutation = mostEffectiveMutation.Key,
                MostEffectiveMutationRate = mostEffectiveMutation.Value,
                LeastEffectiveMutation = leastEffectiveMutation.Key,
                LeastEffectiveMutationRate = leastEffectiveMutation.Value,
                DiversityTrend = diversityTrend,
                AverageWeightDiversity = weightDiversity,
                MutationTypeRates = mutationRates.ToImmutableDictionary(),
                CrossoverStrategyRates = crossoverRates.ToImmutableDictionary()
            };
        }

        /// <summary>
        /// Generates a fitness landscape summary.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <returns>Fitness landscape analysis.</returns>
        public FitnessLandscapeAnalysis AnalyzeFitnessLandscape(GenomePopulation population)
        {
            if (population.Genomes.Length == 0)
            {
                return new FitnessLandscapeAnalysis();
            }

            var fitnesses = population.Genomes.Select(g => g.Fitness).ToArray();
            double mean = fitnesses.Average();
            double variance = fitnesses.Average(f => (f - mean) * (f - mean));
            double stdDev = Math.Sqrt(variance);

            double skewness = 0;
            double kurtosis = 0;
            if (stdDev > 0)
            {
                skewness = fitnesses.Average(f => Math.Pow((f - mean) / stdDev, 3));
                kurtosis = fitnesses.Average(f => Math.Pow((f - mean) / stdDev, 4)) - 3;
            }

            var sorted = fitnesses.OrderBy(f => f).ToArray();
            double q1 = sorted[sorted.Length / 4];
            double median = sorted[sorted.Length / 2];
            double q3 = sorted[3 * sorted.Length / 4];
            double iqr = q3 - q1;

            int outlierCount = fitnesses.Count(f => f < q1 - 1.5 * iqr || f > q3 + 1.5 * iqr);

            double fitnessEntropy = ComputeFitnessEntropy(fitnesses, 20);

            bool isMultimodal = ComputeModesCount(fitnesses) > 1;

            return new FitnessLandscapeAnalysis
            {
                Mean = mean,
                Variance = variance,
                StandardDeviation = stdDev,
                Skewness = skewness,
                Kurtosis = kurtosis,
                Q1 = q1,
                Median = median,
                Q3 = q3,
                IQR = iqr,
                OutlierCount = outlierCount,
                FitnessEntropy = fitnessEntropy,
                IsMultimodal = isMultimodal,
                ModeCount = ComputeModesCount(fitnesses)
            };
        }

        private int EstimateRemainingGenerations(IReadOnlyList<EvolutionMetrics> metrics)
        {
            if (metrics.Count < 10)
                return -1;

            var recent = metrics.TakeLast(10).ToList();
            double avgImprovement = 0;
            for (int i = 1; i < recent.Count; i++)
            {
                avgImprovement += recent[i].BestFitness - recent[i - 1].BestFitness;
            }
            avgImprovement /= (recent.Count - 1);

            if (Math.Abs(avgImprovement) < _config.FitnessThreshold)
                return 0;

            double targetGap = _config.TargetFitness - recent.Last().BestFitness;
            if (targetGap <= 0)
                return 0;

            return avgImprovement > 0 ? (int)Math.Ceiling(targetGap / avgImprovement) : -1;
        }

        private double ComputeFitnessEntropy(double[] fitnesses, int bins)
        {
            if (fitnesses.Length == 0 || bins <= 0)
                return 0;

            double min = fitnesses.Min();
            double max = fitnesses.Max();
            double range = max - min;

            if (range < 1e-10)
                return 0;

            var counts = new int[bins];
            foreach (var f in fitnesses)
            {
                int bin = Math.Min(bins - 1, (int)((f - min) / range * bins));
                counts[bin]++;
            }

            double entropy = 0;
            foreach (var count in counts)
            {
                if (count > 0)
                {
                    double p = (double)count / fitnesses.Length;
                    entropy -= p * Math.Log2(p);
                }
            }

            return entropy / Math.Log2(bins);
        }

        private int ComputeModesCount(double[] fitnesses)
        {
            if (fitnesses.Length < 5)
                return 1;

            double min = fitnesses.Min();
            double max = fitnesses.Max();
            double range = max - min;

            if (range < 1e-10)
                return 1;

            int bins = Math.Max(5, (int)Math.Sqrt(fitnesses.Length));
            var counts = new int[bins];

            foreach (var f in fitnesses)
            {
                int bin = Math.Min(bins - 1, (int)((f - min) / range * bins));
                counts[bin]++;
            }

            int modes = 0;
            for (int i = 1; i < counts.Length - 1; i++)
            {
                if (counts[i] > counts[i - 1] && counts[i] > counts[i + 1] && counts[i] >= 3)
                {
                    modes++;
                }
            }

            return Math.Max(1, modes);
        }
    }

    /// <summary>
    /// Convergence analysis results.
    /// </summary>
    public record ConvergenceAnalysis
    {
        /// <summary>Whether the evolution has converged.</summary>
        public bool HasConverged { get; init; }

        /// <summary>Convergence ratio (lower = more converged).</summary>
        public double ConvergenceRatio { get; init; }

        /// <summary>Recent fitness standard deviation.</summary>
        public double RecentStdDev { get; init; }

        /// <summary>Overall fitness range.</summary>
        public double OverallRange { get; init; }

        /// <summary>Generation when stagnation started (-1 if not stagnant).</summary>
        public int StagnationStart { get; init; }

        /// <summary>Fitness at the plateau.</summary>
        public double PlateauFitness { get; init; }

        /// <summary>Rate of fitness growth.</summary>
        public double FitnessGrowthRate { get; init; }

        /// <summary>Estimated remaining generations to target (-1 if unknown).</summary>
        public int EstimatedRemainingGenerations { get; init; }
    }

    /// <summary>
    /// Species dynamics analysis results.
    /// </summary>
    public record SpeciesDynamicsAnalysis
    {
        /// <summary>Rate of new species formation.</summary>
        public double SpeciationRate { get; init; }

        /// <summary>Rate of species extinction.</summary>
        public double ExtinctionRate { get; init; }

        /// <summary>Peak species count.</summary>
        public int PeakSpeciesCount { get; init; }

        /// <summary>Minimum species count.</summary>
        public int MinSpeciesCount { get; init; }

        /// <summary>Average species count.</summary>
        public double AverageSpeciesCount { get; init; }

        /// <summary>Species count stability (0-1).</summary>
        public double SpeciesStability { get; init; }

        /// <summary>Species ranked by fitness dominance.</summary>
        public IReadOnlyList<int> SpeciesDominanceRanking { get; init; } = Array.Empty<int>();
    }

    /// <summary>
    /// Mutation effectiveness analysis results.
    /// </summary>
    public record MutationEffectivenessAnalysis
    {
        /// <summary>Average mutation success rate.</summary>
        public double AverageMutationSuccessRate { get; init; }

        /// <summary>Average crossover success rate.</summary>
        public double AverageCrossoverSuccessRate { get; init; }

        /// <summary>Most effective mutation type.</summary>
        public MutationType MostEffectiveMutation { get; init; }

        /// <summary>Success rate of most effective mutation.</summary>
        public double MostEffectiveMutationRate { get; init; }

        /// <summary>Least effective mutation type.</summary>
        public MutationType LeastEffectiveMutation { get; init; }

        /// <summary>Success rate of least effective mutation.</summary>
        public double LeastEffectiveMutationRate { get; init; }

        /// <summary>Trend in population diversity (positive = increasing).</summary>
        public double DiversityTrend { get; init; }

        /// <summary>Average weight diversity across snapshots.</summary>
        public double AverageWeightDiversity { get; init; }

        /// <summary>Per-type mutation success rates.</summary>
        public ImmutableDictionary<MutationType, double> MutationTypeRates { get; init; } =
            ImmutableDictionary<MutationType, double>.Empty;

        /// <summary>Per-strategy crossover success rates.</summary>
        public ImmutableDictionary<string, double> CrossoverStrategyRates { get; init; } =
            ImmutableDictionary<string, double>.Empty;
    }

    /// <summary>
    /// Fitness landscape analysis results.
    /// </summary>
    public record FitnessLandscapeAnalysis
    {
        /// <summary>Mean fitness.</summary>
        public double Mean { get; init; }

        /// <summary>Fitness variance.</summary>
        public double Variance { get; init; }

        /// <summary>Fitness standard deviation.</summary>
        public double StandardDeviation { get; init; }

        /// <summary>Fitness skewness.</summary>
        public double Skewness { get; init; }

        /// <summary>Fitness kurtosis.</summary>
        public double Kurtosis { get; init; }

        /// <summary>First quartile.</summary>
        public double Q1 { get; init; }

        /// <summary>Median fitness.</summary>
        public double Median { get; init; }

        /// <summary>Third quartile.</summary>
        public double Q3 { get; init; }

        /// <summary>Interquartile range.</summary>
        public double IQR { get; init; }

        /// <summary>Number of outlier genomes.</summary>
        public int OutlierCount { get; init; }

        /// <summary>Fitness entropy (0-1, higher = more diverse).</summary>
        public double FitnessEntropy { get; init; }

        /// <summary>Whether the fitness landscape is multimodal.</summary>
        public bool IsMultimodal { get; init; }

        /// <summary>Estimated number of fitness modes.</summary>
        public int ModeCount { get; init; }
    }

}
