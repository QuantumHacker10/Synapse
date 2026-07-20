// =============================================================================
// NeatGEvolutionEngine.EvolutionQualityMonitor.cs — NEAT-G partial module
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
    /// Monitors the quality of the evolution process and detects issues
    /// like premature convergence, fitness stagnation, and genetic drift.
    /// Provides actionable recommendations for parameter tuning.
    /// </summary>
    public sealed class EvolutionQualityMonitor
    {
        private readonly Queue<EvolutionQualitySnapshot> _snapshots;
        private readonly int _maxSnapshots;
        private readonly object _lock = new();

        /// <summary>
        /// Occurs when a quality issue is detected.
        /// </summary>
        public event EventHandler<EvolutionQualityIssueEventArgs>? IssueDetected;

        /// <summary>
        /// Initializes a new instance of the EvolutionQualityMonitor class.
        /// </summary>
        /// <param name="maxSnapshots">Maximum number of snapshots to retain.</param>
        public EvolutionQualityMonitor(int maxSnapshots = 100)
        {
            _maxSnapshots = maxSnapshots;
            _snapshots = new Queue<EvolutionQualitySnapshot>(maxSnapshots);
        }

        /// <summary>
        /// Records an evolution quality snapshot.
        /// </summary>
        /// <param name="generation">Current generation.</param>
        /// <param name="population">Current population.</param>
        /// <param name="species">Current species.</param>
        /// <param name="config">Evolution configuration.</param>
        public void RecordSnapshot(
            int generation,
            GenomePopulation population,
            IReadOnlyList<SpeciesInfo> species,
            EvolutionConfig config)
        {
            var snapshot = CreateSnapshot(generation, population, species, config);

            lock (_lock)
            {
                _snapshots.Enqueue(snapshot);
                while (_snapshots.Count > _maxSnapshots)
                    _snapshots.Dequeue();
            }

            AnalyzeQuality(snapshot, config);
        }

        /// <summary>
        /// Analyzes the current quality of evolution and provides diagnostics.
        /// </summary>
        /// <returns>Quality analysis result.</returns>
        public EvolutionQualityReport AnalyzeCurrentQuality()
        {
            EvolutionQualitySnapshot[] snapshots;
            lock (_lock)
            {
                snapshots = _snapshots.ToArray();
            }

            if (snapshots.Length == 0)
            {
                return new EvolutionQualityReport
                {
                    OverallScore = 0,
                    Status = EvolutionQualityStatus.InsufficientData,
                    Issues = new List<QualityIssue>().AsReadOnly(),
                    Recommendations = new List<string> { "Run more generations for quality analysis." }.AsReadOnly()
                };
            }

            var issues = new List<QualityIssue>();
            var recommendations = new List<string>();

            AnalyzeConvergence(snapshots, issues, recommendations);
            AnalyzeDiversity(snapshots, issues, recommendations);
            AnalyzeStagnation(snapshots, issues, recommendations);
            AnalyzeFitnessProgression(snapshots, issues, recommendations);
            AnalyzeSpeciesHealth(snapshots, issues, recommendations);

            double overallScore = ComputeOverallScore(snapshots, issues);

            return new EvolutionQualityReport
            {
                OverallScore = overallScore,
                Status = DetermineStatus(overallScore, issues),
                Issues = issues.AsReadOnly(),
                Recommendations = recommendations.AsReadOnly(),
                LatestSnapshot = snapshots[^1],
                TrendAnalysis = ComputeTrend(snapshots)
            };
        }

        /// <summary>
        /// Gets the recommended parameter adjustments based on quality analysis.
        /// </summary>
        /// <returns>Recommended configuration adjustments.</returns>
        public IReadOnlyDictionary<string, double> GetRecommendedAdjustments()
        {
            var report = AnalyzeCurrentQuality();
            var adjustments = new Dictionary<string, double>();

            foreach (var issue in report.Issues)
            {
                switch (issue.Type)
                {
                    case QualityIssueType.PrematureConvergence:
                        adjustments["MutationRate"] = Math.Min(1.0, adjustments.GetValueOrDefault("MutationRate", 0.1) * 1.5);
                        adjustments["CrossoverRate"] = Math.Max(0.1, adjustments.GetValueOrDefault("CrossoverRate", 0.8) * 0.8);
                        break;

                    case QualityIssueType.FitnessStagnation:
                        adjustments["MutationRate"] = Math.Min(1.0, adjustments.GetValueOrDefault("MutationRate", 0.1) * 2.0);
                        break;

                    case QualityIssueType.LowDiversity:
                        adjustments["SpeciesCompatibilityThreshold"] =
                            Math.Max(0.1, adjustments.GetValueOrDefault("SpeciesCompatibilityThreshold", 0.5) * 0.7);
                        break;

                    case QualityIssueType.GeneticDrift:
                        adjustments["PopulationSize"] = adjustments.GetValueOrDefault("PopulationSize", 100) * 1.3;
                        break;

                    case QualityIssueType.UnbalancedSpecies:
                        adjustments["ElitismCount"] = Math.Min(20, adjustments.GetValueOrDefault("ElitismCount", 5) + 2);
                        break;
                }
            }

            return adjustments;
        }

        /// <summary>
        /// Gets all recorded snapshots.
        /// </summary>
        public IReadOnlyList<EvolutionQualitySnapshot> GetSnapshots()
        {
            lock (_lock)
            {
                return _snapshots.ToArray().AsReadOnly();
            }
        }

        /// <summary>
        /// Clears all recorded snapshots.
        /// </summary>
        public void Clear()
        {
            lock (_lock)
            {
                _snapshots.Clear();
            }
        }

        private EvolutionQualitySnapshot CreateSnapshot(
            int generation,
            GenomePopulation population,
            IReadOnlyList<SpeciesInfo> species,
            EvolutionConfig config)
        {
            var fitnesses = population.Genomes.Select(g => g.Fitness).ToList();
            var complexities = population.Genomes.Select(g => g.Complexity).ToList();

            return new EvolutionQualitySnapshot
            {
                Timestamp = DateTime.UtcNow,
                Generation = generation,
                PopulationSize = population.Genomes.Length,
                BestFitness = fitnesses.Max(),
                WorstFitness = fitnesses.Min(),
                AverageFitness = fitnesses.Average(),
                FitnessStdDev = fitnesses.StandardDeviation(),
                FitnessMedian = fitnesses.Median(),
                AverageComplexity = complexities.Average(),
                ComplexityStdDev = complexities.StandardDeviation(),
                SpeciesCount = species.Count,
                AverageSpeciesSize = species.Count > 0 ? species.Average(s => s.MemberCount) : 0,
                LargestSpeciesSize = species.Count > 0 ? species.Max(s => s.MemberCount) : 0,
                SmallestSpeciesSize = species.Count > 0 ? species.Min(s => s.MemberCount) : 0,
                SpeciesSizeStdDev = species.Count > 1
                    ? species.Select(s => (double)s.MemberCount).StandardDeviation()
                    : 0,
                UniqueTopologies = population.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count(),
                TopologyDiversityRatio = (double)population.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count() / Math.Max(1, population.Genomes.Length)
            };
        }

        private void AnalyzeQuality(EvolutionQualitySnapshot snapshot, EvolutionConfig config)
        {
            lock (_lock)
            {
                if (_snapshots.Count < 5)
                    return;
            }

            var allSnapshots = GetSnapshots();
            if (allSnapshots.Count < 5)
                return;

            var issues = new List<QualityIssue>();
            var recommendations = new List<string>();

            AnalyzeConvergence(allSnapshots.ToArray(), issues, recommendations);
            AnalyzeDiversity(allSnapshots.ToArray(), issues, recommendations);
            AnalyzeStagnation(allSnapshots.ToArray(), issues, recommendations);

            foreach (var issue in issues)
            {
                IssueDetected?.Invoke(this, new EvolutionQualityIssueEventArgs(issue, recommendations));
            }
        }

        private void AnalyzeConvergence(
            EvolutionQualitySnapshot[] snapshots,
            List<QualityIssue> issues,
            List<string> recommendations)
        {
            if (snapshots.Length < 10)
                return;

            var recentFitnesses = snapshots.Skip(snapshots.Length - 10).Select(s => s.BestFitness).ToList();
            double variance = recentFitnesses.StandardDeviation();

            if (variance < 1e-6)
            {
                issues.Add(new QualityIssue
                {
                    Type = QualityIssueType.PrematureConvergence,
                    Severity = QualityIssueSeverity.High,
                    Message = "Best fitness has converged to a narrow range, suggesting premature convergence.",
                    DetectedAt = DateTime.UtcNow
                });
                recommendations.Add("Increase mutation rate or reduce selection pressure to escape local optima.");
            }
        }

        private void AnalyzeDiversity(
            EvolutionQualitySnapshot[] snapshots,
            List<QualityIssue> issues,
            List<string> recommendations)
        {
            if (snapshots.Length < 5)
                return;

            var recentDiversity = snapshots.Skip(snapshots.Length - 5).Select(s => s.TopologyDiversityRatio).ToList();
            double avgDiversity = recentDiversity.Average();

            if (avgDiversity < 0.3)
            {
                issues.Add(new QualityIssue
                {
                    Type = QualityIssueType.LowDiversity,
                    Severity = QualityIssueSeverity.Medium,
                    Message = $"Topology diversity ratio is low ({avgDiversity:F3}), limiting evolutionary exploration.",
                    DetectedAt = DateTime.UtcNow
                });
                recommendations.Add("Reduce species compatibility threshold or increase mutation rates.");
            }

            if (snapshots.Length >= 3)
            {
                double diversityTrend = snapshots[^1].TopologyDiversityRatio - snapshots[^3].TopologyDiversityRatio;
                if (diversityTrend < -0.2)
                {
                    issues.Add(new QualityIssue
                    {
                        Type = QualityIssueType.GeneticDrift,
                        Severity = QualityIssueSeverity.Medium,
                        Message = "Topology diversity is declining rapidly, indicating genetic drift.",
                        DetectedAt = DateTime.UtcNow
                    });
                    recommendations.Add("Introduce immigration or increase crossover rate.");
                }
            }
        }

        private void AnalyzeStagnation(
            EvolutionQualitySnapshot[] snapshots,
            List<QualityIssue> issues,
            List<string> recommendations)
        {
            if (snapshots.Length < 15)
                return;

            var recentAvgFitness = snapshots.Skip(snapshots.Length - 15).Select(s => s.AverageFitness).ToList();
            double slope = ComputeLinearSlope(recentAvgFitness);

            if (Math.Abs(slope) < 1e-8)
            {
                issues.Add(new QualityIssue
                {
                    Type = QualityIssueType.FitnessStagnation,
                    Severity = QualityIssueSeverity.High,
                    Message = "Average fitness has stagnated over the last 15 generations.",
                    DetectedAt = DateTime.UtcNow
                });
                recommendations.Add("Significantly increase mutation rate or introduce novel genetic operators.");
            }
        }

        private void AnalyzeFitnessProgression(
            EvolutionQualitySnapshot[] snapshots,
            List<QualityIssue> issues,
            List<string> recommendations)
        {
            if (snapshots.Length < 20)
                return;

            var allBestFitness = snapshots.Select(s => s.BestFitness).ToList();
            double overallSlope = ComputeLinearSlope(allBestFitness);

            if (overallSlope < 0)
            {
                issues.Add(new QualityIssue
                {
                    Type = QualityIssueType.FitnessRegression,
                    Severity = QualityIssueSeverity.Critical,
                    Message = "Overall fitness trend is negative, indicating regression.",
                    DetectedAt = DateTime.UtcNow
                });
                recommendations.Add("Review mutation operators for destructive changes. Consider reducing mutation strength.");
            }
        }

        private void AnalyzeSpeciesHealth(
            EvolutionQualitySnapshot[] snapshots,
            List<QualityIssue> issues,
            List<string> recommendations)
        {
            if (snapshots.Length == 0)
                return;

            var latest = snapshots[^1];
            if (latest.SpeciesCount > 1 && latest.SpeciesSizeStdDev > latest.AverageSpeciesSize * 0.5)
            {
                issues.Add(new QualityIssue
                {
                    Type = QualityIssueType.UnbalancedSpecies,
                    Severity = QualityIssueSeverity.Medium,
                    Message = "Species sizes are highly unbalanced, with dominant species emerging.",
                    DetectedAt = DateTime.UtcNow
                });
                recommendations.Add("Increase diversity preservation mechanisms or adjust fitness sharing.");
            }
        }

        private double ComputeLinearSlope(List<double> values)
        {
            int n = values.Count;
            if (n < 2)
                return 0;

            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            for (int i = 0; i < n; i++)
            {
                sumX += i;
                sumY += values[i];
                sumXY += i * values[i];
                sumX2 += i * i;
            }

            double denominator = n * sumX2 - sumX * sumX;
            return Math.Abs(denominator) > 1e-10
                ? (n * sumXY - sumX * sumY) / denominator
                : 0;
        }

        private double ComputeOverallScore(EvolutionQualitySnapshot[] snapshots, List<QualityIssue> issues)
        {
            double score = 1.0;

            foreach (var issue in issues)
            {
                switch (issue.Severity)
                {
                    case QualityIssueSeverity.Critical:
                        score -= 0.3;
                        break;
                    case QualityIssueSeverity.High:
                        score -= 0.2;
                        break;
                    case QualityIssueSeverity.Medium:
                        score -= 0.1;
                        break;
                    case QualityIssueSeverity.Low:
                        score -= 0.05;
                        break;
                }
            }

            if (snapshots.Length > 0)
            {
                score *= snapshots[^1].TopologyDiversityRatio;
            }

            return Math.Max(0, Math.Min(1, score));
        }

        private EvolutionQualityStatus DetermineStatus(double score, List<QualityIssue> issues)
        {
            if (issues.Any(i => i.Severity == QualityIssueSeverity.Critical))
                return EvolutionQualityStatus.Critical;

            if (score > 0.8)
                return EvolutionQualityStatus.Excellent;
            if (score > 0.6)
                return EvolutionQualityStatus.Good;
            if (score > 0.4)
                return EvolutionQualityStatus.Fair;
            if (score > 0.2)
                return EvolutionQualityStatus.Poor;
            return EvolutionQualityStatus.Critical;
        }

        private TrendAnalysis ComputeTrend(EvolutionQualitySnapshot[] snapshots)
        {
            if (snapshots.Length < 3)
            {
                return new TrendAnalysis { FitnessTrend = TrendDirection.Stable, DiversityTrend = TrendDirection.Stable };
            }

            var recentBestFitness = snapshots.Skip(snapshots.Length - 5).Select(s => s.BestFitness).ToList();
            var recentDiversity = snapshots.Skip(snapshots.Length - 5).Select(s => s.TopologyDiversityRatio).ToList();

            return new TrendAnalysis
            {
                FitnessTrend = ComputeTrendDirection(recentBestFitness),
                DiversityTrend = ComputeTrendDirection(recentDiversity),
                FitnessSlope = ComputeLinearSlope(recentBestFitness),
                DiversitySlope = ComputeLinearSlope(recentDiversity),
                GenerationsAnalyzed = Math.Min(5, snapshots.Length)
            };
        }

        private TrendDirection ComputeTrendDirection(List<double> values)
        {
            double slope = ComputeLinearSlope(values);
            if (slope > 0.01)
                return TrendDirection.Improving;
            if (slope < -0.01)
                return TrendDirection.Declining;
            return TrendDirection.Stable;
        }
    }

    /// <summary>
    /// Snapshot of evolution quality metrics at a specific point.
    /// </summary>
    public sealed class EvolutionQualitySnapshot
    {
        /// <summary>Timestamp.</summary>
        public DateTime Timestamp { get; init; }
        /// <summary>Generation number.</summary>
        public int Generation { get; init; }
        /// <summary>Population size.</summary>
        public int PopulationSize { get; init; }
        /// <summary>Best fitness in population.</summary>
        public double BestFitness { get; init; }
        /// <summary>Worst fitness in population.</summary>
        public double WorstFitness { get; init; }
        /// <summary>Average fitness.</summary>
        public double AverageFitness { get; init; }
        /// <summary>Standard deviation of fitness.</summary>
        public double FitnessStdDev { get; init; }
        /// <summary>Median fitness.</summary>
        public double FitnessMedian { get; init; }
        /// <summary>Average complexity.</summary>
        public double AverageComplexity { get; init; }
        /// <summary>Standard deviation of complexity.</summary>
        public double ComplexityStdDev { get; init; }
        /// <summary>Number of species.</summary>
        public int SpeciesCount { get; init; }
        /// <summary>Average species size.</summary>
        public double AverageSpeciesSize { get; init; }
        /// <summary>Largest species size.</summary>
        public int LargestSpeciesSize { get; init; }
        /// <summary>Smallest species size.</summary>
        public int SmallestSpeciesSize { get; init; }
        /// <summary>Standard deviation of species sizes.</summary>
        public double SpeciesSizeStdDev { get; init; }
        /// <summary>Number of unique topologies.</summary>
        public int UniqueTopologies { get; init; }
        /// <summary>Ratio of unique topologies to population size.</summary>
        public double TopologyDiversityRatio { get; init; }
    }

    /// <summary>
    /// Quality issue detected in the evolution process.
    /// </summary>
    public sealed class QualityIssue
    {
        /// <summary>Issue type.</summary>
        public QualityIssueType Type { get; init; }
        /// <summary>Issue severity.</summary>
        public QualityIssueSeverity Severity { get; init; }
        /// <summary>Human-readable message.</summary>
        public string Message { get; init; } = string.Empty;
        /// <summary>When the issue was detected.</summary>
        public DateTime DetectedAt { get; init; }
    }

    /// <summary>
    /// Quality analysis report.
    /// </summary>
    public sealed class EvolutionQualityReport
    {
        /// <summary>Overall quality score [0, 1].</summary>
        public double OverallScore { get; init; }
        /// <summary>Overall quality status.</summary>
        public EvolutionQualityStatus Status { get; init; }
        /// <summary>Detected issues.</summary>
        public IReadOnlyList<QualityIssue> Issues { get; init; } = Array.Empty<QualityIssue>();
        /// <summary>Recommendations for improvement.</summary>
        public IReadOnlyList<string> Recommendations { get; init; } = Array.Empty<string>();
        /// <summary>Latest quality snapshot.</summary>
        public EvolutionQualitySnapshot? LatestSnapshot { get; init; }
        /// <summary>Trend analysis.</summary>
        public TrendAnalysis? TrendAnalysis { get; init; }
    }

    /// <summary>
    /// Trend analysis result.
    /// </summary>
    public sealed class TrendAnalysis
    {
        /// <summary>Fitness trend direction.</summary>
        public TrendDirection FitnessTrend { get; init; }
        /// <summary>Diversity trend direction.</summary>
        public TrendDirection DiversityTrend { get; init; }
        /// <summary>Fitness slope value.</summary>
        public double FitnessSlope { get; init; }
        /// <summary>Diversity slope value.</summary>
        public double DiversitySlope { get; init; }
        /// <summary>Number of generations analyzed.</summary>
        public int GenerationsAnalyzed { get; init; }
    }

    /// <summary>
    /// Event args for detected quality issues.
    /// </summary>
    public sealed class EvolutionQualityIssueEventArgs : EventArgs
    {
        /// <summary>The detected issue.</summary>
        public QualityIssue Issue { get; }
        /// <summary>Recommendations.</summary>
        public IReadOnlyList<string> Recommendations { get; }

        /// <summary>
        /// Initializes a new instance of the EvolutionQualityIssueEventArgs class.
        /// </summary>
        public EvolutionQualityIssueEventArgs(QualityIssue issue, IReadOnlyList<string> recommendations)
        {
            Issue = issue;
            Recommendations = recommendations;
        }
    }

    /// <summary>
    /// Types of quality issues in evolution.
    /// </summary>
    public enum QualityIssueType
    {
        /// <summary>Premature convergence detected.</summary>
        PrematureConvergence,
        /// <summary>Fitness stagnation.</summary>
        FitnessStagnation,
        /// <summary>Low population diversity.</summary>
        LowDiversity,
        /// <summary>Genetic drift.</summary>
        GeneticDrift,
        /// <summary>Fitness regression.</summary>
        FitnessRegression,
        /// <summary>Unbalanced species sizes.</summary>
        UnbalancedSpecies,
        /// <summary>Excessive complexity growth.</summary>
        ComplexityGrowth,
        /// <summary>Computation inefficiency.</summary>
        ComputationInefficiency
    }

    /// <summary>
    /// Severity levels for quality issues.
    /// </summary>
    public enum QualityIssueSeverity
    {
        /// <summary>Low severity.</summary>
        Low,
        /// <summary>Medium severity.</summary>
        Medium,
        /// <summary>High severity.</summary>
        High,
        /// <summary>Critical severity.</summary>
        Critical
    }

    /// <summary>
    /// Overall evolution quality status.
    /// </summary>
    public enum EvolutionQualityStatus
    {
        /// <summary>Insufficient data for analysis.</summary>
        InsufficientData,
        /// <summary>Critical issues detected.</summary>
        Critical,
        /// <summary>Poor quality.</summary>
        Poor,
        /// <summary>Fair quality.</summary>
        Fair,
        /// <summary>Good quality.</summary>
        Good,
        /// <summary>Excellent quality.</summary>
        Excellent
    }

    /// <summary>
    /// Trend direction.
    /// </summary>
    public enum TrendDirection
    {
        /// <summary>Metric is declining.</summary>
        Declining,
        /// <summary>Metric is stable.</summary>
        Stable,
        /// <summary>Metric is improving.</summary>
        Improving
    }

}
