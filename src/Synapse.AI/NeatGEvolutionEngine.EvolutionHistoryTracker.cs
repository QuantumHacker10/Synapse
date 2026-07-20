// =============================================================================
// NeatGEvolutionEngine.EvolutionHistoryTracker.cs — NEAT-G partial module
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
    /// Records and tracks all evolution events, statistics, and fitness history.
    /// Provides export capabilities and visualization data for evolution analysis.
    /// </summary>
    public sealed class EvolutionHistoryTracker
    {
        private readonly EvolutionConfig _config;
        private readonly List<EvolutionEvent> _events;
        private readonly List<EvolutionMetrics> _metricsHistory;
        private readonly Dictionary<int, List<double>> _speciesFitnessHistory;
        private readonly Dictionary<int, List<int>> _speciesSizeHistory;
        private readonly ConcurrentQueue<EvolutionEvent> _eventBuffer;
        private readonly object _lock = new();

        /// <summary>
        /// Initializes a new instance of the EvolutionHistoryTracker class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public EvolutionHistoryTracker(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _events = new List<EvolutionEvent>();
            _metricsHistory = new List<EvolutionMetrics>();
            _speciesFitnessHistory = new Dictionary<int, List<double>>();
            _speciesSizeHistory = new Dictionary<int, List<int>>();
            _eventBuffer = new ConcurrentQueue<EvolutionEvent>();
        }

        /// <summary>Gets the total number of recorded events.</summary>
        public int EventCount
        {
            get { lock (_lock) { return _events.Count; } }
        }

        /// <summary>Gets the number of recorded metrics entries.</summary>
        public int MetricsCount
        {
            get { lock (_lock) { return _metricsHistory.Count; } }
        }

        /// <summary>
        /// Records an evolution event.
        /// </summary>
        /// <param name="eventType">Type of event.</param>
        /// <param name="generation">Current generation.</param>
        /// <param name="description">Event description.</param>
        /// <param name="genomeId">Associated genome ID (optional).</param>
        /// <param name="speciesId">Associated species ID (optional).</param>
        /// <param name="value">Numeric value (optional).</param>
        public void RecordEvent(
            EvolutionEventType eventType,
            int generation,
            string description,
            Guid? genomeId = null,
            int? speciesId = null,
            double value = 0)
        {
            var evolutionEvent = new EvolutionEvent
            {
                EventType = eventType,
                Generation = generation,
                Description = description,
                GenomeId = genomeId,
                SpeciesId = speciesId,
                Value = value,
                Timestamp = DateTime.UtcNow
            };

            _eventBuffer.Enqueue(evolutionEvent);

            lock (_lock)
            {
                if (_config.EnableHistoryTracking && _events.Count < _config.MaxHistoryEntries)
                {
                    _events.Add(evolutionEvent);
                }
            }
        }

        /// <summary>
        /// Records evolution metrics for a generation.
        /// </summary>
        /// <param name="metrics">The metrics to record.</param>
        public void RecordMetrics(EvolutionMetrics metrics)
        {
            lock (_lock)
            {
                _metricsHistory.Add(metrics);
            }
        }

        /// <summary>
        /// Records species fitness history.
        /// </summary>
        /// <param name="speciesId">Species identifier.</param>
        /// <param name="fitness">Current best fitness.</param>
        /// <param name="size">Current species size.</param>
        public void RecordSpeciesData(int speciesId, double fitness, int size)
        {
            lock (_lock)
            {
                if (!_speciesFitnessHistory.ContainsKey(speciesId))
                    _speciesFitnessHistory[speciesId] = new List<double>();
                _speciesFitnessHistory[speciesId].Add(fitness);

                if (!_speciesSizeHistory.ContainsKey(speciesId))
                    _speciesSizeHistory[speciesId] = new List<int>();
                _speciesSizeHistory[speciesId].Add(size);
            }
        }

        /// <summary>
        /// Gets all recorded events.
        /// </summary>
        public IReadOnlyList<EvolutionEvent> GetEvents()
        {
            lock (_lock)
            {
                return _events.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets events filtered by type.
        /// </summary>
        /// <param name="eventType">The event type to filter by.</param>
        public IReadOnlyList<EvolutionEvent> GetEventsByType(EvolutionEventType eventType)
        {
            lock (_lock)
            {
                return _events.Where(e => e.EventType == eventType).ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets all recorded metrics.
        /// </summary>
        public IReadOnlyList<EvolutionMetrics> GetMetricsHistory()
        {
            lock (_lock)
            {
                return _metricsHistory.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets fitness over generations as a series of (generation, fitness) pairs.
        /// </summary>
        public IReadOnlyList<(int Generation, double BestFitness, double AverageFitness)> GetFitnessOverGenerations()
        {
            lock (_lock)
            {
                return _metricsHistory
                    .Select(m => (m.Generation, m.BestFitness, m.AverageFitness))
                    .ToList()
                    .AsReadOnly();
            }
        }

        /// <summary>
        /// Gets species fitness history for a specific species.
        /// </summary>
        /// <param name="speciesId">Species identifier.</param>
        public IReadOnlyList<(int Generation, double Fitness, int Size)> GetSpeciesHistory(int speciesId)
        {
            lock (_lock)
            {
                var result = new List<(int, double, int)>();
                if (_speciesFitnessHistory.TryGetValue(speciesId, out var fitnesses))
                {
                    if (_speciesSizeHistory.TryGetValue(speciesId, out var sizes))
                    {
                        for (int i = 0; i < fitnesses.Count; i++)
                        {
                            int size = i < sizes.Count ? sizes[i] : 0;
                            result.Add((i, fitnesses[i], size));
                        }
                    }
                }
                return result.AsReadOnly();
            }
        }

        /// <summary>
        /// Gets summary statistics for the evolution run.
        /// </summary>
        public EvolutionSummary GetSummary()
        {
            lock (_lock)
            {
                if (_metricsHistory.Count == 0)
                {
                    return new EvolutionSummary { TotalGenerations = 0 };
                }

                var bestEver = _metricsHistory.MaxBy(m => m.BestFitness);
                var first = _metricsHistory.First();
                var last = _metricsHistory.Last();

                return new EvolutionSummary
                {
                    TotalGenerations = _metricsHistory.Count,
                    BestFitnessEver = bestEver.BestFitness,
                    BestFitnessGeneration = bestEver.Generation,
                    InitialFitness = first.BestFitness,
                    FinalFitness = last.BestFitness,
                    TotalEvaluations = last.TotalEvaluations,
                    PeakSpeciesCount = _metricsHistory.Max(m => m.SpeciesCount),
                    FinalSpeciesCount = last.SpeciesCount,
                    AverageDiversity = _metricsHistory.Average(m => m.DiversityMetric),
                    FitnessImprovement = last.BestFitness - first.BestFitness,
                    TotalEvents = _events.Count
                };
            }
        }

        /// <summary>
        /// Exports evolution history as a JSON string.
        /// </summary>
        /// <param name="includeEvents">Whether to include detailed events.</param>
        /// <returns>JSON string representation of the history.</returns>
        public string ExportJson(bool includeEvents = false)
        {
            lock (_lock)
            {
                var data = new
                {
                    Summary = GetSummary(),
                    Metrics = _metricsHistory,
                    Events = includeEvents ? _events : new List<EvolutionEvent>()
                };

                return JsonSerializer.Serialize(data, new JsonSerializerOptions
                {
                    WriteIndented = true,
                    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
                });
            }
        }

        /// <summary>
        /// Exports evolution history as CSV-formatted metrics.
        /// </summary>
        /// <returns>CSV string of generation metrics.</returns>
        public string ExportCsv()
        {
            lock (_lock)
            {
                var sb = new StringBuilder();
                sb.AppendLine("Generation,BestFitness,AverageFitness,MedianFitness,StdDev,SpeciesCount,Diversity,Evaluations,GenTime");

                foreach (var m in _metricsHistory)
                {
                    sb.AppendLine($"{m.Generation},{m.BestFitness:F6},{m.AverageFitness:F6}," +
                                 $"{m.MedianFitness:F6},{m.StdDevFitness:F6},{m.SpeciesCount}," +
                                 $"{m.DiversityMetric:F4},{m.EvaluationsThisGeneration}," +
                                 $"{m.GenerationTime.TotalMilliseconds:F1}");
                }

                return sb.ToString();
            }
        }

        /// <summary>
        /// Clears all recorded history.
        /// </summary>
        public void Clear()
        {
            lock (_lock)
            {
                _events.Clear();
                _metricsHistory.Clear();
                _speciesFitnessHistory.Clear();
                _speciesSizeHistory.Clear();
            }

            while (_eventBuffer.TryDequeue(out _))
            { }
        }
    }

    /// <summary>
    /// Summary statistics for an evolution run.
    /// </summary>
    public record EvolutionSummary
    {
        /// <summary>Total generations executed.</summary>
        public int TotalGenerations { get; init; }

        /// <summary>Best fitness achieved across all generations.</summary>
        public double BestFitnessEver { get; init; }

        /// <summary>Generation when best fitness was achieved.</summary>
        public int BestFitnessGeneration { get; init; }

        /// <summary>Initial population best fitness.</summary>
        public double InitialFitness { get; init; }

        /// <summary>Final population best fitness.</summary>
        public double FinalFitness { get; init; }

        /// <summary>Total fitness evaluations.</summary>
        public long TotalEvaluations { get; init; }

        /// <summary>Peak species count during evolution.</summary>
        public int PeakSpeciesCount { get; init; }

        /// <summary>Final species count.</summary>
        public int FinalSpeciesCount { get; init; }

        /// <summary>Average diversity across all generations.</summary>
        public double AverageDiversity { get; init; }

        /// <summary>Total fitness improvement from first to last generation.</summary>
        public double FitnessImprovement { get; init; }

        /// <summary>Total events recorded.</summary>
        public int TotalEvents { get; init; }
    }

}
