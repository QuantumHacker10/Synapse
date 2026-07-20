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
    /// Hyper-heuristic controller that selects and tunes evolution operators
    /// dynamically based on problem characteristics and performance feedback.
    /// Uses a multi-armed bandit approach for operator selection.
    /// </summary>
    public sealed class HyperHeuristicController : IDisposable
    {
        private readonly ConcurrentDictionary<string, OperatorStatistics> _operatorStats;
        private readonly Timer _rebalanceTimer;
        private readonly object _lock = new();
        private readonly Random _rng;
        private bool _disposed;

        private const double ExplorationRate = 0.15;
        private const double LearningRate = 0.1;
        private const double DiscountFactor = 0.95;
        private const int MinSamplesBeforeSelection = 5;

        /// <summary>
        /// Occurs when the operator selection strategy changes.
        /// </summary>
        public event EventHandler<HyperHeuristicEventArgs>? StrategyChanged;

        /// <summary>
        /// Initializes a new instance of the HyperHeuristicController class.
        /// </summary>
        public HyperHeuristicController()
        {
            _operatorStats = new ConcurrentDictionary<string, OperatorStatistics>();
            _rng = Random.Shared;
            _rebalanceTimer = new Timer(RebalanceCallback, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
        }

        /// <summary>
        /// Records the result of an operator execution.
        /// </summary>
        /// <param name="operatorName">Name of the operator.</param>
        /// <param name="improvement">Fitness improvement achieved.</param>
        /// <param name="executionTime">Time taken.</param>
        public void RecordResult(string operatorName, double improvement, TimeSpan executionTime)
        {
            var stats = _operatorStats.GetOrAdd(operatorName, _ => new OperatorStatistics(operatorName));

            lock (stats.Lock)
            {
                stats.TotalExecutions++;
                stats.TotalImprovement += improvement;
                stats.TotalExecutionTime += executionTime;
                stats.RecentImprovements.Add(improvement);

                if (stats.RecentImprovements.Count > 100)
                    stats.RecentImprovements.RemoveAt(0);

                double avgImprovement = stats.RecentImprovements.Average();
                stats.QValue = (1 - LearningRate) * stats.QValue + LearningRate * improvement;
                stats.AverageImprovement = avgImprovement;
                stats.LastUpdate = DateTime.UtcNow;
            }
        }

        /// <summary>
        /// Selects the next operator using the UCB1 (Upper Confidence Bound) algorithm.
        /// </summary>
        /// <param name="availableOperators">Available operator names.</param>
        /// <returns>Selected operator name.</returns>
        public string SelectOperator(IReadOnlyList<string> availableOperators)
        {
            if (availableOperators.Count == 0)
                throw new ArgumentException("No operators available.", nameof(availableOperators));

            if (availableOperators.Count == 1)
                return availableOperators[0];

            if (_rng.NextDouble() < ExplorationRate)
            {
                return availableOperators[_rng.Next(availableOperators.Count)];
            }

            int totalExecutions = _operatorStats.Values.Sum(s =>
            {
                lock (s.Lock)
                    return s.TotalExecutions;
            });

            string bestOperator = availableOperators[0];
            double bestScore = double.MinValue;

            foreach (var opName in availableOperators)
            {
                var stats = _operatorStats.GetOrAdd(opName, _ => new OperatorStatistics(opName));

                double qValue;
                int executions;
                lock (stats.Lock)
                {
                    qValue = stats.QValue;
                    executions = stats.TotalExecutions;
                }

                double explorationBonus = executions > 0
                    ? Math.Sqrt(2 * Math.Log(Math.Max(1, totalExecutions)) / executions)
                    : double.MaxValue;

                double score = qValue + explorationBonus;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestOperator = opName;
                }
            }

            return bestOperator;
        }

        /// <summary>
        /// Gets the best performing operators ranked by Q-value.
        /// </summary>
        /// <param name="count">Number of operators to return.</param>
        /// <returns>Ranked list of operator statistics.</returns>
        public IReadOnlyList<OperatorRanking> GetRankedOperators(int count = 10)
        {
            var rankings = new List<OperatorRanking>();

            foreach (var kvp in _operatorStats)
            {
                lock (kvp.Value.Lock)
                {
                    rankings.Add(new OperatorRanking
                    {
                        OperatorName = kvp.Key,
                        QValue = kvp.Value.QValue,
                        AverageImprovement = kvp.Value.AverageImprovement,
                        TotalExecutions = kvp.Value.TotalExecutions,
                        AverageExecutionTime = kvp.Value.TotalExecutions > 0
                            ? kvp.Value.TotalExecutionTime / kvp.Value.TotalExecutions
                            : TimeSpan.Zero,
                        Rank = 0
                    });
                }
            }

            rankings = rankings.OrderByDescending(r => r.QValue).Take(count).ToList();
            for (int i = 0; i < rankings.Count; i++)
                rankings[i].Rank = i + 1;

            return rankings.AsReadOnly();
        }

        /// <summary>
        /// Gets statistics for a specific operator.
        /// </summary>
        /// <param name="operatorName">Operator name.</param>
        /// <returns>Statistics or null if not found.</returns>
        public OperatorStatistics? GetStatistics(string operatorName)
        {
            return _operatorStats.TryGetValue(operatorName, out var stats) ? stats.Clone() : null;
        }

        /// <summary>
        /// Resets all operator statistics.
        /// </summary>
        public void Reset()
        {
            _operatorStats.Clear();
        }

        private void RebalanceCallback(object? state)
        {
            var staleOperators = _operatorStats
                .Where(kvp =>
                {
                    lock (kvp.Value.Lock)
                        return (DateTime.UtcNow - kvp.Value.LastUpdate).TotalMinutes > 5;
                })
                .Select(kvp => kvp.Key)
                .ToList();

            foreach (var opName in staleOperators)
            {
                if (_operatorStats.TryRemove(opName, out var stats))
                {
                    lock (stats.Lock)
                    {
                        stats.QValue *= 0.9;
                    }
                    _operatorStats[opName] = stats;
                }
            }

            StrategyChanged?.Invoke(this, new HyperHeuristicEventArgs(staleOperators));
        }

        /// <summary>
        /// Disposes of the controller.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _rebalanceTimer?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Statistics for an evolution operator in the hyper-heuristic.
    /// </summary>
    public sealed class OperatorStatistics
    {
        internal readonly object Lock = new();

        /// <summary>
        /// Operator name.
        /// </summary>
        public string OperatorName { get; }

        /// <summary>
        /// Q-value (expected reward).
        /// </summary>
        public double QValue { get; internal set; }

        /// <summary>
        /// Average fitness improvement.
        /// </summary>
        public double AverageImprovement { get; internal set; }

        /// <summary>
        /// Total number of executions.
        /// </summary>
        public int TotalExecutions { get; internal set; }

        /// <summary>
        /// Total improvement across all executions.
        /// </summary>
        public double TotalImprovement { get; internal set; }

        /// <summary>
        /// Total execution time.
        /// </summary>
        public TimeSpan TotalExecutionTime { get; internal set; }

        /// <summary>
        /// Last update timestamp.
        /// </summary>
        public DateTime LastUpdate { get; internal set; }

        /// <summary>
        /// Recent improvements for rolling average.
        /// </summary>
        internal List<double> RecentImprovements { get; } = new();

        /// <summary>
        /// Initializes a new instance of the OperatorStatistics class.
        /// </summary>
        public OperatorStatistics(string operatorName)
        {
            OperatorName = operatorName;
            LastUpdate = DateTime.UtcNow;
        }

        /// <summary>
        /// Creates a deep copy of this statistics object.
        /// </summary>
        public OperatorStatistics Clone()
        {
            lock (Lock)
            {
                var clone = new OperatorStatistics(OperatorName)
                {
                    QValue = QValue,
                    AverageImprovement = AverageImprovement,
                    TotalExecutions = TotalExecutions,
                    TotalImprovement = TotalImprovement,
                    TotalExecutionTime = TotalExecutionTime,
                    LastUpdate = LastUpdate
                };
                clone.RecentImprovements.AddRange(RecentImprovements);
                return clone;
            }
        }

        /// <summary>
        /// Gets the success rate (fraction of improvements).
        /// </summary>
        public double GetSuccessRate()
        {
            lock (Lock)
            {
                return TotalExecutions > 0
                    ? RecentImprovements.Count(v => v > 0) / (double)RecentImprovements.Count
                    : 0;
            }
        }
    }

    /// <summary>
    /// Ranked operator in the hyper-heuristic.
    /// </summary>
    public sealed class OperatorRanking
    {
        /// <summary>Operator name.</summary>
        public string OperatorName { get; init; }
        /// <summary>Q-value.</summary>
        public double QValue { get; init; }
        /// <summary>Average improvement.</summary>
        public double AverageImprovement { get; init; }
        /// <summary>Total executions.</summary>
        public int TotalExecutions { get; init; }
        /// <summary>Average execution time.</summary>
        public TimeSpan AverageExecutionTime { get; init; }
        /// <summary>Rank position (1-based).</summary>
        public int Rank { get; set; }
    }

    /// <summary>
    /// Event args for hyper-heuristic strategy changes.
    /// </summary>
    public sealed class HyperHeuristicEventArgs : EventArgs
    {
        /// <summary>Operators that were affected by rebalancing.</summary>
        public IReadOnlyList<string> AffectedOperators { get; }

        /// <summary>
        /// Initializes a new instance of the HyperHeuristicEventArgs class.
        /// </summary>
        public HyperHeuristicEventArgs(IReadOnlyList<string> affectedOperators)
        {
            AffectedOperators = affectedOperators;
        }
    }

}
