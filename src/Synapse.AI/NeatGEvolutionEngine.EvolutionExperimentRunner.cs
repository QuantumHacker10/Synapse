// =============================================================================
// NeatGEvolutionEngine.EvolutionExperimentRunner.cs — NEAT-G partial module
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
    /// Runs controlled evolution experiments with statistical rigor.
    /// Supports multiple runs, parameter sweeps, and comparative analysis.
    /// </summary>
    public sealed class EvolutionExperimentRunner
    {
        /// <summary>
        /// Runs multiple evolution trials with the same configuration for statistical analysis.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="trialCount">Number of trials to run.</param>
        /// <param name="progressCallback">Progress callback.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Statistical summary of all trials.</returns>
        public async Task<ExperimentResults> RunMultipleTrialsAsync(
            EvolutionConfig config,
            int inputCount,
            int outputCount,
            EvaluationContext context,
            int trialCount,
            IProgress<int>? progressCallback = null,
            CancellationToken ct = default)
        {
            var trialResults = new List<TrialResult>();

            for (int trial = 0; trial < trialCount; trial++)
            {
                ct.ThrowIfCancellationRequested();

                var trialConfig = config.Clone();
                trialConfig.RandomSeed = (trial + 1) * 42;

                var engine = new NeatGEvolutionEngine(trialConfig);
                var result = await engine.RunEvolutionAsync(
                    inputCount, outputCount, context, null, ct).ConfigureAwait(false);

                trialResults.Add(new TrialResult
                {
                    TrialNumber = trial + 1,
                    BestFitness = result.BestGenome.Fitness,
                    TotalGenerations = result.TotalGenerations,
                    TotalEvaluations = result.TotalEvaluations,
                    ElapsedTime = result.TotalElapsed,
                    TargetReached = result.TargetReached,
                    FinalSpeciesCount = result.FinalPopulation.Count > 0
                        ? result.MetricsHistory.Last().SpeciesCount
                        : 0
                });

                progressCallback?.Report(trial + 1);
            }

            return ComputeExperimentStatistics(trialResults);
        }

        /// <summary>
        /// Runs a parameter sweep over specified parameter ranges.
        /// </summary>
        /// <param name="baseConfig">Base configuration.</param>
        /// <param name="parameterName">Parameter to sweep.</param>
        /// <param name="values">Values to try.</param>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="trialsPerValue">Number of trials per parameter value.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Results for each parameter value.</returns>
        public async Task<IReadOnlyList<ParameterSweepResult>> RunParameterSweepAsync(
            EvolutionConfig baseConfig,
            string parameterName,
            double[] values,
            int inputCount,
            int outputCount,
            EvaluationContext context,
            int trialsPerValue = 3,
            CancellationToken ct = default)
        {
            var results = new List<ParameterSweepResult>();

            foreach (var value in values)
            {
                ct.ThrowIfCancellationRequested();

                var config = baseConfig.Clone();
                SetParameterValue(config, parameterName, value);

                var trialResults = new List<double>();
                var trialTimes = new List<TimeSpan>();

                for (int trial = 0; trial < trialsPerValue; trial++)
                {
                    ct.ThrowIfCancellationRequested();

                    var trialConfig = config.Clone();
                    trialConfig.RandomSeed = (trial + 1) * 100;

                    var engine = new NeatGEvolutionEngine(trialConfig);
                    var result = await engine.RunEvolutionAsync(
                        inputCount, outputCount, context, null, ct).ConfigureAwait(false);

                    trialResults.Add(result.BestGenome.Fitness);
                    trialTimes.Add(result.TotalElapsed);
                }

                results.Add(new ParameterSweepResult
                {
                    ParameterName = parameterName,
                    ParameterValue = value,
                    AverageFitness = trialResults.Average(),
                    StdDevFitness = trialResults.Count > 1
                        ? Math.Sqrt(trialResults.Average(f => (f - trialResults.Average()) * (f - trialResults.Average())))
                        : 0,
                    BestFitness = trialResults.Max(),
                    WorstFitness = trialResults.Min(),
                    AverageTime = TimeSpan.FromTicks((long)trialTimes.Average(t => t.Ticks)),
                    TrialCount = trialsPerValue
                });
            }

            return results.AsReadOnly();
        }

        /// <summary>
        /// Runs a comparative experiment between two configurations.
        /// </summary>
        public async Task<ComparisonResult> RunComparativeExperimentAsync(
            EvolutionConfig configA,
            EvolutionConfig configB,
            string configAName,
            string configBName,
            int inputCount,
            int outputCount,
            EvaluationContext context,
            int trialCount = 5,
            CancellationToken ct = default)
        {
            var resultsA = new List<double>();
            var resultsB = new List<double>();
            var timesA = new List<TimeSpan>();
            var timesB = new List<TimeSpan>();

            for (int trial = 0; trial < trialCount; trial++)
            {
                ct.ThrowIfCancellationRequested();

                int seed = (trial + 1) * 42;

                var configAClone = configA.Clone();
                configAClone.RandomSeed = seed;
                var engineA = new NeatGEvolutionEngine(configAClone);
                var resultA = await engineA.RunEvolutionAsync(
                    inputCount, outputCount, context, null, ct).ConfigureAwait(false);
                resultsA.Add(resultA.BestGenome.Fitness);
                timesA.Add(resultA.TotalElapsed);

                var configBClone = configB.Clone();
                configBClone.RandomSeed = seed;
                var engineB = new NeatGEvolutionEngine(configBClone);
                var resultB = await engineB.RunEvolutionAsync(
                    inputCount, outputCount, context, null, ct).ConfigureAwait(false);
                resultsB.Add(resultB.BestGenome.Fitness);
                timesB.Add(resultB.TotalElapsed);
            }

            double meanA = resultsA.Average();
            double meanB = resultsB.Average();
            double stdA = resultsA.Count > 1
                ? Math.Sqrt(resultsA.Average(f => (f - meanA) * (f - meanA))) : 0;
            double stdB = resultsB.Count > 1
                ? Math.Sqrt(resultsB.Average(f => (f - meanB) * (f - meanB))) : 0;

            double tStatistic = 0;
            double pValue = 0.5;
            if (stdA > 0 && stdB > 0)
            {
                double pooledSE = Math.Sqrt(stdA * stdA / trialCount + stdB * stdB / trialCount);
                if (pooledSE > 1e-10)
                {
                    tStatistic = (meanA - meanB) / pooledSE;
                    pValue = ComputeTwoTailedPValue(Math.Abs(tStatistic), 2 * trialCount - 2);
                }
            }

            return new ComparisonResult
            {
                ConfigAName = configAName,
                ConfigBName = configBName,
                MeanFitnessA = meanA,
                MeanFitnessB = meanB,
                StdDevA = stdA,
                StdDevB = stdB,
                TStatistic = tStatistic,
                PValue = pValue,
                IsSignificant = pValue < 0.05,
                Winner = meanA > meanB ? configAName : meanB > meanA ? configBName : "Tie",
                AverageTimeA = TimeSpan.FromTicks((long)timesA.Average(t => t.Ticks)),
                AverageTimeB = TimeSpan.FromTicks((long)timesB.Average(t => t.Ticks)),
                TrialCount = trialCount
            };
        }

        private ExperimentResults ComputeExperimentStatistics(List<TrialResult> trialResults)
        {
            var bestFitnesses = trialResults.Select(t => t.BestFitness).ToList();
            var generations = trialResults.Select(t => t.TotalGenerations).ToList();
            var evaluations = trialResults.Select(t => t.TotalEvaluations).ToList();

            double mean = bestFitnesses.Average();
            double variance = bestFitnesses.Average(f => (f - mean) * (f - mean));
            double stdDev = Math.Sqrt(variance);

            return new ExperimentResults
            {
                TrialCount = trialResults.Count,
                MeanBestFitness = mean,
                StdDevBestFitness = stdDev,
                MedianBestFitness = GetMedian(bestFitnesses),
                MinBestFitness = bestFitnesses.Min(),
                MaxBestFitness = bestFitnesses.Max(),
                MeanGenerations = generations.Average(),
                MeanEvaluations = evaluations.Average(),
                TargetReachedCount = trialResults.Count(t => t.TargetReached),
                TargetReachedRate = (double)trialResults.Count(t => t.TargetReached) / trialResults.Count,
                MeanElapsedTime = TimeSpan.FromTicks(
                    (long)trialResults.Average(t => t.ElapsedTime.Ticks)),
                TrialResults = trialResults.AsReadOnly()
            };
        }

        private double GetMedian(List<double> values)
        {
            var sorted = values.OrderBy(v => v).ToList();
            int mid = sorted.Count / 2;
            return sorted.Count % 2 == 0
                ? (sorted[mid - 1] + sorted[mid]) / 2.0
                : sorted[mid];
        }

        private void SetParameterValue(EvolutionConfig config, string parameterName, double value)
        {
            switch (parameterName)
            {
                case nameof(EvolutionConfig.MutationRate):
                    config.MutationRate = value;
                    break;
                case nameof(EvolutionConfig.CrossoverRate):
                    config.CrossoverRate = value;
                    break;
                case nameof(EvolutionConfig.SpeciationThreshold):
                    config.SpeciationThreshold = value;
                    break;
                case nameof(EvolutionConfig.TournamentSize):
                    config.TournamentSize = (int)value;
                    break;
                case nameof(EvolutionConfig.PerturbationMagnitude):
                    config.PerturbationMagnitude = value;
                    break;
                case nameof(EvolutionConfig.MigrationRate):
                    config.MigrationRate = value;
                    break;
                case nameof(EvolutionConfig.PopulationSize):
                    config.PopulationSize = (int)value;
                    break;
                case nameof(EvolutionConfig.MaxStagnationGenerations):
                    config.MaxStagnationGenerations = (int)value;
                    break;
            }
        }

        private double ComputeTwoTailedPValue(double tStat, int degreesOfFreedom)
        {
            double x = (double)degreesOfFreedom / (degreesOfFreedom + tStat * tStat);
            double p = 0.5 * IncompleteBetaFunction(degreesOfFreedom / 2.0, 0.5, x);
            return 2 * p;
        }

        private double IncompleteBetaFunction(double a, double b, double x)
        {
            if (x <= 0)
                return 0;
            if (x >= 1)
                return 1;

            double result = 0;
            double term = Math.Pow(x, a) * Math.Pow(1 - x, b) / a;
            result = term;

            for (int n = 1; n < 200; n++)
            {
                double numerator = n * (b - n) * x / ((a + 2 * n - 1) * (a + 2 * n));
                term *= numerator;
                result += term;
                if (Math.Abs(term) < 1e-10)
                    break;
            }

            double beta = GammaFunction(a) * GammaFunction(b) / GammaFunction(a + b);
            return result / beta;
        }

        private double GammaFunction(double z)
        {
            if (z < 0.5)
                return Math.PI / (Math.Sin(Math.PI * z) * GammaFunction(1 - z));

            z -= 1;
            double[] coefficients = {
                0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                771.32342877765313, -176.61502916214059, 12.507343278686905,
                -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
            };

            double x = coefficients[0];
            for (int i = 1; i < coefficients.Length; i++)
                x += coefficients[i] / (z + i);

            double t = z + coefficients.Length - 1.5;
            return Math.Sqrt(2 * Math.PI) * Math.Pow(t, z + 0.5) * Math.Exp(-t) * x;
        }
    }

    /// <summary>
    /// Results of a single trial run.
    /// </summary>
    public sealed class TrialResult
    {
        public int TrialNumber { get; init; }
        public double BestFitness { get; init; }
        public int TotalGenerations { get; init; }
        public long TotalEvaluations { get; init; }
        public TimeSpan ElapsedTime { get; init; }
        public bool TargetReached { get; init; }
        public int FinalSpeciesCount { get; init; }
    }

    /// <summary>
    /// Statistical results of multiple experiment trials.
    /// </summary>
    public sealed class ExperimentResults
    {
        public int TrialCount { get; init; }
        public double MeanBestFitness { get; init; }
        public double StdDevBestFitness { get; init; }
        public double MedianBestFitness { get; init; }
        public double MinBestFitness { get; init; }
        public double MaxBestFitness { get; init; }
        public double MeanGenerations { get; init; }
        public double MeanEvaluations { get; init; }
        public int TargetReachedCount { get; init; }
        public double TargetReachedRate { get; init; }
        public TimeSpan MeanElapsedTime { get; init; }
        public IReadOnlyList<TrialResult> TrialResults { get; init; } = Array.Empty<TrialResult>();
    }

    /// <summary>
    /// Result of a parameter sweep value.
    /// </summary>
    public sealed class ParameterSweepResult
    {
        public string ParameterName { get; init; } = string.Empty;
        public double ParameterValue { get; init; }
        public double AverageFitness { get; init; }
        public double StdDevFitness { get; init; }
        public double BestFitness { get; init; }
        public double WorstFitness { get; init; }
        public TimeSpan AverageTime { get; init; }
        public int TrialCount { get; init; }
    }

    /// <summary>
    /// Result of a comparative experiment between two configurations.
    /// </summary>
    public sealed class ComparisonResult
    {
        public string ConfigAName { get; init; } = string.Empty;
        public string ConfigBName { get; init; } = string.Empty;
        public double MeanFitnessA { get; init; }
        public double MeanFitnessB { get; init; }
        public double StdDevA { get; init; }
        public double StdDevB { get; init; }
        public double TStatistic { get; init; }
        public double PValue { get; init; }
        public bool IsSignificant { get; init; }
        public string Winner { get; init; } = string.Empty;
        public TimeSpan AverageTimeA { get; init; }
        public TimeSpan AverageTimeB { get; init; }
        public int TrialCount { get; init; }
    }

}
