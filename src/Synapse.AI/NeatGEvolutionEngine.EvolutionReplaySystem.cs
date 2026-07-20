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
    /// Records and replays evolution runs for debugging, analysis, and visualization.
    /// Captures all genome operations and allows step-by-step replay.
    /// </summary>
    public sealed class EvolutionReplaySystem
    {
        private readonly List<ReplayEvent> _events;
        private int _currentPosition;
        private bool _isRecording;

        /// <summary>
        /// Initializes a new instance of the EvolutionReplaySystem class.
        /// </summary>
        public EvolutionReplaySystem()
        {
            _events = new List<ReplayEvent>();
            _currentPosition = 0;
            _isRecording = false;
        }

        /// <summary>Whether the system is currently recording.</summary>
        public bool IsRecording => _isRecording;

        /// <summary>Total recorded events.</summary>
        public int EventCount => _events.Count;

        /// <summary>Current replay position.</summary>
        public int CurrentPosition => _currentPosition;

        /// <summary>
        /// Starts recording evolution events.
        /// </summary>
        public void StartRecording()
        {
            _isRecording = true;
            _events.Clear();
            _currentPosition = 0;
        }

        /// <summary>
        /// Stops recording.
        /// </summary>
        public void StopRecording()
        {
            _isRecording = false;
        }

        /// <summary>
        /// Records a replay event.
        /// </summary>
        /// <param name="eventType">Type of event.</param>
        /// <param name="generation">Current generation.</param>
        /// <param name="data">Event data.</param>
        /// <param name="genomeId">Associated genome ID (optional).</param>
        public void RecordEvent(ReplayEventType eventType, int generation, string data, Guid? genomeId = null)
        {
            if (!_isRecording)
                return;

            _events.Add(new ReplayEvent
            {
                Position = _events.Count,
                Timestamp = DateTime.UtcNow,
                EventType = eventType,
                Generation = generation,
                Data = data,
                GenomeId = genomeId
            });
        }

        /// <summary>
        /// Records a genome snapshot at a specific point.
        /// </summary>
        /// <param name="genome">Genome to record.</param>
        /// <param name="generation">Current generation.</param>
        public void RecordGenomeSnapshot(GeoGenome genome, int generation)
        {
            if (!_isRecording)
                return;

            _events.Add(new ReplayEvent
            {
                Position = _events.Count,
                Timestamp = DateTime.UtcNow,
                EventType = ReplayEventType.GenomeSnapshot,
                Generation = generation,
                Data = JsonSerializer.Serialize(new
                {
                    genome.Id,
                    genome.Fitness,
                    NeuronCount = genome.ActiveNeuronCount,
                    SynapseCount = genome.ActiveSynapseCount,
                    genome.Complexity
                }),
                GenomeId = genome.Id
            });
        }

        /// <summary>
        /// Gets the next event in replay.
        /// </summary>
        public ReplayEvent? GetNextEvent()
        {
            if (_currentPosition >= _events.Count)
                return null;
            return _events[_currentPosition++];
        }

        /// <summary>
        /// Gets a specific event by position.
        /// </summary>
        /// <param name="position">Event position.</param>
        public ReplayEvent? GetEventAt(int position)
        {
            if (position < 0 || position >= _events.Count)
                return null;
            return _events[position];
        }

        /// <summary>
        /// Seeks to a specific position in the replay.
        /// </summary>
        /// <param name="position">Target position.</param>
        public void SeekTo(int position)
        {
            _currentPosition = Math.Clamp(position, 0, _events.Count);
        }

        /// <summary>
        /// Seeks to a specific generation.
        /// </summary>
        /// <param name="generation">Target generation.</param>
        public void SeekToGeneration(int generation)
        {
            var idx = _events.FindIndex(e => e.Generation >= generation);
            if (idx >= 0)
                _currentPosition = idx;
        }

        /// <summary>
        /// Gets all events for a specific generation.
        /// </summary>
        /// <param name="generation">Generation number.</param>
        public IReadOnlyList<ReplayEvent> GetEventsForGeneration(int generation)
        {
            return _events.Where(e => e.Generation == generation).ToList().AsReadOnly();
        }

        /// <summary>
        /// Gets events filtered by type.
        /// </summary>
        /// <param name="eventType">Event type to filter.</param>
        public IReadOnlyList<ReplayEvent> GetEventsByType(ReplayEventType eventType)
        {
            return _events.Where(e => e.EventType == eventType).ToList().AsReadOnly();
        }

        /// <summary>
        /// Gets the total number of generations recorded.
        /// </summary>
        public int GetTotalGenerations()
        {
            return _events.Count > 0 ? _events.Max(e => e.Generation) + 1 : 0;
        }

        /// <summary>
        /// Exports the replay data.
        /// </summary>
        public string ExportJson()
        {
            return JsonSerializer.Serialize(_events, new JsonSerializerOptions
            {
                WriteIndented = false
            });
        }

        /// <summary>
        /// Imports replay data from JSON.
        /// </summary>
        /// <param name="json">JSON string.</param>
        public void ImportJson(string json)
        {
            var imported = JsonSerializer.Deserialize<List<ReplayEvent>>(json);
            if (imported != null)
            {
                _events.Clear();
                _events.AddRange(imported);
                _currentPosition = 0;
            }
        }

        /// <summary>
        /// Resets the replay system.
        /// </summary>
        public void Reset()
        {
            _events.Clear();
            _currentPosition = 0;
            _isRecording = false;
        }
    }

    /// <summary>
    /// Types of replay events.
    /// </summary>
    public enum ReplayEventType
    {
        GenerationStart,
        GenerationEnd,
        GenomeCreated,
        GenomeEvaluated,
        GenomeMutated,
        GenomeCrossover,
        GenomeSnapshot,
        SpeciesCreated,
        SpeciesEliminated,
        Migration,
        Selection,
        ParameterChange
    }

    /// <summary>
    /// A single replay event.
    /// </summary>
    public sealed class ReplayEvent
    {
        public int Position { get; init; }
        public DateTime Timestamp { get; init; }
        public ReplayEventType EventType { get; init; }
        public int Generation { get; init; }
        public string Data { get; init; } = string.Empty;
        public Guid? GenomeId { get; init; }
    }

}
