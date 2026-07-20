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
    /// Generates formatted summary reports for evolution runs.
    /// Supports plain text, markdown, and HTML output formats.
    /// </summary>
    public sealed class EvolutionReportGenerator
    {
        private readonly EvolutionAnalyticsDashboard _dashboard;
        private readonly EvolutionHistoryTracker _history;

        /// <summary>
        /// Initializes a new instance of the EvolutionReportGenerator class.
        /// </summary>
        /// <param name="dashboard">Analytics dashboard.</param>
        /// <param name="history">History tracker.</param>
        public EvolutionReportGenerator(
            EvolutionAnalyticsDashboard dashboard,
            EvolutionHistoryTracker history)
        {
            _dashboard = dashboard ?? throw new ArgumentNullException(nameof(dashboard));
            _history = history ?? throw new ArgumentNullException(nameof(history));
        }

        /// <summary>
        /// Generates a plain text summary report.
        /// </summary>
        /// <returns>Plain text report string.</returns>
        public string GeneratePlainTextReport()
        {
            return _dashboard.GenerateReport();
        }

        /// <summary>
        /// Generates a markdown-formatted summary report.
        /// </summary>
        /// <returns>Markdown report string.</returns>
        public string GenerateMarkdownReport()
        {
            var summary = _history.GetSummary();
            var convergence = _dashboard.AnalyzeConvergence();
            var speciesDynamics = _dashboard.AnalyzeSpeciesDynamics();
            var mutationAnalysis = _dashboard.AnalyzeMutationEffectiveness();

            var sb = new StringBuilder();
            sb.AppendLine("# NEAT-G Evolution Report");
            sb.AppendLine();

            sb.AppendLine("## Overview");
            sb.AppendLine($"| Metric | Value |");
            sb.AppendLine($"|--------|-------|");
            sb.AppendLine($"| Total Generations | {summary.TotalGenerations} |");
            sb.AppendLine($"| Total Evaluations | {summary.TotalEvaluations:N0} |");
            sb.AppendLine($"| Best Fitness | {summary.BestFitnessEver:F6} |");
            sb.AppendLine($"| Fitness Improvement | {summary.FitnessImprovement:F6} |");
            sb.AppendLine($"| Peak Species | {summary.PeakSpeciesCount} |");
            sb.AppendLine($"| Final Species | {summary.FinalSpeciesCount} |");
            sb.AppendLine();

            sb.AppendLine("## Convergence Analysis");
            sb.AppendLine($"- **Converged:** {convergence.HasConverged}");
            sb.AppendLine($"- **Convergence Ratio:** {convergence.ConvergenceRatio:F4}");
            sb.AppendLine($"- **Recent Std Dev:** {convergence.RecentStdDev:F6}");
            sb.AppendLine($"- **Fitness Growth Rate:** {convergence.FitnessGrowthRate:P1}");
            if (convergence.EstimatedRemainingGenerations > 0)
                sb.AppendLine($"- **Est. Remaining Generations:** {convergence.EstimatedRemainingGenerations}");
            sb.AppendLine();

            sb.AppendLine("## Species Dynamics");
            sb.AppendLine($"- **Speciation Rate:** {speciesDynamics.SpeciationRate:F2} species/gen");
            sb.AppendLine($"- **Extinction Rate:** {speciesDynamics.ExtinctionRate:F2} species/gen");
            sb.AppendLine($"- **Species Stability:** {speciesDynamics.SpeciesStability:F3}");
            sb.AppendLine();

            sb.AppendLine("## Mutation Effectiveness");
            sb.AppendLine($"- **Avg Mutation Success:** {mutationAnalysis.AverageMutationSuccessRate:P1}");
            sb.AppendLine($"- **Avg Crossover Success:** {mutationAnalysis.AverageCrossoverSuccessRate:P1}");
            sb.AppendLine($"- **Most Effective:** {mutationAnalysis.MostEffectiveMutation} ({mutationAnalysis.MostEffectiveMutationRate:P1})");
            sb.AppendLine($"- **Diversity Trend:** {mutationAnalysis.DiversityTrend:F4}");
            sb.AppendLine();

            sb.AppendLine("## Fitness Progression");
            var metrics = _history.GetMetricsHistory();
            if (metrics.Count > 0)
            {
                sb.AppendLine("```");
                sb.AppendLine("Gen | Best     | Avg      | Species | Diversity");
                sb.AppendLine("----|----------|----------|---------|----------");
                int interval = Math.Max(1, metrics.Count / 30);
                for (int i = 0; i < metrics.Count; i += interval)
                {
                    var m = metrics[i];
                    sb.AppendLine($"{m.Generation,4} | {m.BestFitness,8:F4} | {m.AverageFitness,8:F4} | {m.SpeciesCount,7} | {m.DiversityMetric,8:F4}");
                }
                sb.AppendLine("```");
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generates an HTML-formatted summary report.
        /// </summary>
        /// <returns>HTML report string.</returns>
        public string GenerateHtmlReport()
        {
            var summary = _history.GetSummary();

            var sb = new StringBuilder();
            sb.AppendLine("<!DOCTYPE html>");
            sb.AppendLine("<html><head><title>NEAT-G Evolution Report</title>");
            sb.AppendLine("<style>");
            sb.AppendLine("body { font-family: Arial, sans-serif; margin: 20px; }");
            sb.AppendLine("h1 { color: #2C3E50; }");
            sb.AppendLine("h2 { color: #34495E; }");
            sb.AppendLine("table { border-collapse: collapse; width: 100%; }");
            sb.AppendLine("th, td { border: 1px solid #BDC3C7; padding: 8px; text-align: left; }");
            sb.AppendLine("th { background-color: #ECF0F1; }");
            sb.AppendLine(".metric { font-size: 1.2em; font-weight: bold; }");
            sb.AppendLine(".improvement { color: #27AE60; }");
            sb.AppendLine(".warning { color: #E67E22; }");
            sb.AppendLine("</style></head><body>");
            sb.AppendLine("<h1>NEAT-G Evolution Report</h1>");

            sb.AppendLine("<h2>Overview</h2>");
            sb.AppendLine("<table>");
            sb.AppendLine("<tr><th>Metric</th><th>Value</th></tr>");
            sb.AppendLine($"<tr><td>Total Generations</td><td>{summary.TotalGenerations}</td></tr>");
            sb.AppendLine($"<tr><td>Total Evaluations</td><td>{summary.TotalEvaluations:N0}</td></tr>");
            sb.AppendLine($"<tr><td>Best Fitness</td><td class='metric improvement'>{summary.BestFitnessEver:F6}</td></tr>");
            sb.AppendLine($"<tr><td>Fitness Improvement</td><td class='improvement'>{summary.FitnessImprovement:F6}</td></tr>");
            sb.AppendLine($"<tr><td>Peak Species</td><td>{summary.PeakSpeciesCount}</td></tr>");
            sb.AppendLine($"<tr><td>Final Species</td><td>{summary.FinalSpeciesCount}</td></tr>");
            sb.AppendLine($"<tr><td>Average Diversity</td><td>{summary.AverageDiversity:F4}</td></tr>");
            sb.AppendLine("</table>");

            sb.AppendLine("<h2>Fitness Over Generations</h2>");
            sb.AppendLine("<table>");
            sb.AppendLine("<tr><th>Generation</th><th>Best Fitness</th><th>Avg Fitness</th><th>Species</th><th>Diversity</th></tr>");

            var metrics = _history.GetMetricsHistory();
            int interval = Math.Max(1, metrics.Count / 20);
            for (int i = 0; i < metrics.Count; i += interval)
            {
                var m = metrics[i];
                sb.AppendLine($"<tr><td>{m.Generation}</td><td>{m.BestFitness:F4}</td><td>{m.AverageFitness:F4}</td><td>{m.SpeciesCount}</td><td>{m.DiversityMetric:F4}</td></tr>");
            }
            sb.AppendLine("</table>");

            sb.AppendLine("</body></html>");
            return sb.ToString();
        }

        /// <summary>
        /// Generates a compact summary suitable for console output.
        /// </summary>
        /// <param name="metrics">Current metrics.</param>
        /// <returns>Compact single-line summary.</returns>
        public string GenerateCompactSummary(EvolutionMetrics metrics)
        {
            return $"Gen {metrics.Generation,5} | Best {metrics.BestFitness,8:F4} | Avg {metrics.AverageFitness,8:F4} | " +
                   $"Species {metrics.SpeciesCount,3} | Div {metrics.DiversityMetric,5:F3} | " +
                   $"Mut {metrics.MutationSuccessRate:P0} | Cross {metrics.CrossoverSuccessRate:P0} | " +
                   $"Time {metrics.GenerationTime.TotalMilliseconds:F0}ms";
        }
    }

}
