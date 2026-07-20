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
    /// Manages computational budgets for evolution runs, including time limits,
    /// generation limits, fitness targets, and adaptive budget allocation.
    /// </summary>
    public sealed class EvolutionBudgetManager
    {
        private long _totalFitnessEvaluations;
        private long _totalMutations;
        private long _totalCrossovers;
        private long _totalSpeciesComputations;
        private readonly Stopwatch _elapsedTimer;
        private readonly object _lock = new();

        /// <summary>Budget configuration.</summary>
        public EvolutionBudgetConfig Config { get; }

        /// <summary>Gets the elapsed evolution time.</summary>
        public TimeSpan Elapsed => _elapsedTimer.Elapsed;

        /// <summary>Gets total fitness evaluations performed.</summary>
        public long TotalFitnessEvaluations => Interlocked.Read(ref _totalFitnessEvaluations);

        /// <summary>Gets total mutations performed.</summary>
        public long TotalMutations => Interlocked.Read(ref _totalMutations);

        /// <summary>Gets total crossovers performed.</summary>
        public long TotalCrossovers => Interlocked.Read(ref _totalCrossovers);

        /// <summary>Gets total species computations performed.</summary>
        public long TotalSpeciesComputations => Interlocked.Read(ref _totalSpeciesComputations);

        /// <summary>
        /// Initializes a new instance of the EvolutionBudgetManager class.
        /// </summary>
        /// <param name="config">Budget configuration.</param>
        public EvolutionBudgetManager(EvolutionBudgetConfig config)
        {
            Config = config ?? throw new ArgumentNullException(nameof(config));
            _elapsedTimer = new Stopwatch();
            _elapsedTimer.Start();
        }

        /// <summary>Records a fitness evaluation.</summary>
        public void RecordFitnessEvaluation(long count = 1)
        {
            Interlocked.Add(ref _totalFitnessEvaluations, count);
        }

        /// <summary>Records a mutation operation.</summary>
        public void RecordMutation(long count = 1)
        {
            Interlocked.Add(ref _totalMutations, count);
        }

        /// <summary>Records a crossover operation.</summary>
        public void RecordCrossover(long count = 1)
        {
            Interlocked.Add(ref _totalCrossovers, count);
        }

        /// <summary>Records a species computation.</summary>
        public void RecordSpeciesComputation(long count = 1)
        {
            Interlocked.Add(ref _totalSpeciesComputations, count);
        }

        /// <summary>
        /// Checks if any budget limit has been exceeded.
        /// </summary>
        public bool IsBudgetExceeded()
        {
            if (Config.MaxTime.HasValue && _elapsedTimer.Elapsed >= Config.MaxTime.Value)
                return true;
            if (Config.MaxFitnessEvaluations.HasValue &&
                _totalFitnessEvaluations >= Config.MaxFitnessEvaluations.Value)
                return true;
            if (Config.MaxGenerations.HasValue)
                return true;
            if (Config.TargetFitness.HasValue)
                return true;
            return false;
        }

        /// <summary>
        /// Checks if a specific budget limit has been exceeded.
        /// </summary>
        public bool IsLimitExceeded(BudgetLimitType type)
        {
            return type switch
            {
                BudgetLimitType.Time => Config.MaxTime.HasValue && _elapsedTimer.Elapsed >= Config.MaxTime.Value,
                BudgetLimitType.FitnessEvaluations => Config.MaxFitnessEvaluations.HasValue &&
                    _totalFitnessEvaluations >= Config.MaxFitnessEvaluations.Value,
                BudgetLimitType.Mutations => Config.MaxMutations.HasValue &&
                    _totalMutations >= Config.MaxMutations.Value,
                BudgetLimitType.Crossovers => Config.MaxCrossovers.HasValue &&
                    _totalCrossovers >= Config.MaxCrossovers.Value,
                _ => false
            };
        }

        /// <summary>
        /// Gets the remaining budget for each limit type.
        /// </summary>
        public BudgetRemaining GetRemainingBudget()
        {
            return new BudgetRemaining
            {
                RemainingTime = Config.MaxTime.HasValue
                    ? Config.MaxTime.Value - _elapsedTimer.Elapsed
                    : (TimeSpan?)null,
                RemainingFitnessEvaluations = Config.MaxFitnessEvaluations.HasValue
                    ? Math.Max(0, Config.MaxFitnessEvaluations.Value - _totalFitnessEvaluations)
                    : (long?)null,
                RemainingMutations = Config.MaxMutations.HasValue
                    ? Math.Max(0, Config.MaxMutations.Value - _totalMutations)
                    : (long?)null,
                RemainingCrossovers = Config.MaxCrossovers.HasValue
                    ? Math.Max(0, Config.MaxCrossovers.Value - _totalCrossovers)
                    : (long?)null
            };
        }

        /// <summary>
        /// Gets the completion percentage for each budget limit.
        /// </summary>
        public IReadOnlyDictionary<BudgetLimitType, double> GetCompletionPercentages()
        {
            var percentages = new Dictionary<BudgetLimitType, double>();

            if (Config.MaxTime.HasValue)
            {
                percentages[BudgetLimitType.Time] = Math.Min(1.0,
                    _elapsedTimer.Elapsed.TotalMilliseconds / Config.MaxTime.Value.TotalMilliseconds);
            }

            if (Config.MaxFitnessEvaluations.HasValue)
            {
                percentages[BudgetLimitType.FitnessEvaluations] = Math.Min(1.0,
                    (double)_totalFitnessEvaluations / Config.MaxFitnessEvaluations.Value);
            }

            if (Config.MaxMutations.HasValue)
            {
                percentages[BudgetLimitType.Mutations] = Math.Min(1.0,
                    (double)_totalMutations / Config.MaxMutations.Value);
            }

            if (Config.MaxCrossovers.HasValue)
            {
                percentages[BudgetLimitType.Crossovers] = Math.Min(1.0,
                    (double)_totalCrossovers / Config.MaxCrossovers.Value);
            }

            return percentages;
        }

        /// <summary>
        /// Gets the most constraining budget limit.
        /// </summary>
        public BudgetLimitType? GetMostConstrainingLimit()
        {
            var percentages = GetCompletionPercentages();
            if (percentages.Count == 0)
                return null;
            return percentages.OrderByDescending(kvp => kvp.Value).First().Key;
        }

        /// <summary>Resets all budget counters.</summary>
        public void Reset()
        {
            Interlocked.Exchange(ref _totalFitnessEvaluations, 0);
            Interlocked.Exchange(ref _totalMutations, 0);
            Interlocked.Exchange(ref _totalCrossovers, 0);
            Interlocked.Exchange(ref _totalSpeciesComputations, 0);
            lock (_lock)
            { _elapsedTimer.Restart(); }
        }

        /// <summary>Stops the elapsed timer.</summary>
        public void Stop()
        {
            lock (_lock)
            { _elapsedTimer.Stop(); }
        }

        /// <summary>Creates a budget report.</summary>
        public BudgetReport CreateReport()
        {
            return new BudgetReport
            {
                ElapsedTime = _elapsedTimer.Elapsed,
                TotalFitnessEvaluations = _totalFitnessEvaluations,
                TotalMutations = _totalMutations,
                TotalCrossovers = _totalCrossovers,
                TotalSpeciesComputations = _totalSpeciesComputations,
                RemainingBudget = GetRemainingBudget(),
                CompletionPercentages = GetCompletionPercentages(),
                MostConstrainingLimit = GetMostConstrainingLimit(),
                IsBudgetExceeded = IsBudgetExceeded()
            };
        }
    }

    /// <summary>
    /// Budget configuration for evolution.
    /// </summary>
    public sealed class EvolutionBudgetConfig
    {
        /// <summary>Maximum time allowed.</summary>
        public TimeSpan? MaxTime { get; init; }
        /// <summary>Maximum fitness evaluations.</summary>
        public long? MaxFitnessEvaluations { get; init; }
        /// <summary>Maximum number of generations.</summary>
        public int? MaxGenerations { get; init; }
        /// <summary>Target fitness value.</summary>
        public double? TargetFitness { get; init; }
        /// <summary>Maximum mutation operations.</summary>
        public long? MaxMutations { get; init; }
        /// <summary>Maximum crossover operations.</summary>
        public long? MaxCrossovers { get; init; }
        /// <summary>Maximum memory usage in bytes.</summary>
        public long? MaxMemoryBytes { get; init; }
        /// <summary>Maximum species computations.</summary>
        public long? MaxSpeciesComputations { get; init; }
    }

    /// <summary>Types of budget limits.</summary>
    public enum BudgetLimitType
    {
        Time,
        FitnessEvaluations,
        Mutations,
        Crossovers,
        Memory,
        SpeciesComputations
    }

    /// <summary>Remaining budget information.</summary>
    public sealed class BudgetRemaining
    {
        public TimeSpan? RemainingTime { get; init; }
        public long? RemainingFitnessEvaluations { get; init; }
        public long? RemainingMutations { get; init; }
        public long? RemainingCrossovers { get; init; }
    }

    /// <summary>Budget status report.</summary>
    public sealed class BudgetReport
    {
        public TimeSpan ElapsedTime { get; init; }
        public long TotalFitnessEvaluations { get; init; }
        public long TotalMutations { get; init; }
        public long TotalCrossovers { get; init; }
        public long TotalSpeciesComputations { get; init; }
        public BudgetRemaining RemainingBudget { get; init; } = new();
        public IReadOnlyDictionary<BudgetLimitType, double> CompletionPercentages { get; init; } =
            new Dictionary<BudgetLimitType, double>();
        public BudgetLimitType? MostConstrainingLimit { get; init; }
        public bool IsBudgetExceeded { get; init; }
    }

}
