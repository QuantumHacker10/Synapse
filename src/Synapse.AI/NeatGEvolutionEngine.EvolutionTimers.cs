// =============================================================================
// NeatGEvolutionEngine.EvolutionTimers.cs — NEAT-G partial module
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
    /// High-resolution timer for tracking evolution phase durations.
    /// Uses Stopwatch for precise timing of evolution operations.
    /// </summary>
    public sealed class EvolutionTimers
    {
        private readonly ConcurrentDictionary<string, PhaseTimer> _timers;
        private readonly List<PhaseTimingRecord> _records;

        /// <summary>
        /// Initializes a new instance of the EvolutionTimers class.
        /// </summary>
        public EvolutionTimers()
        {
            _timers = new ConcurrentDictionary<string, PhaseTimer>();
            _records = new List<PhaseTimingRecord>();
        }

        /// <summary>
        /// Starts timing a phase.
        /// </summary>
        /// <param name="phaseName">Name of the phase.</param>
        public void StartPhase(string phaseName)
        {
            var timer = _timers.GetOrAdd(phaseName, _ => new PhaseTimer());
            timer.Restart();
        }

        /// <summary>
        /// Stops timing a phase and records the result.
        /// </summary>
        /// <param name="phaseName">Name of the phase.</param>
        /// <param name="generation">Current generation.</param>
        public void StopPhase(string phaseName, int generation)
        {
            if (_timers.TryGetValue(phaseName, out var timer))
            {
                timer.Stop();
                lock (_records)
                {
                    _records.Add(new PhaseTimingRecord
                    {
                        PhaseName = phaseName,
                        Duration = timer.Elapsed,
                        Generation = generation,
                        Timestamp = DateTime.UtcNow
                    });
                }
            }
        }

        /// <summary>
        /// Gets the total time spent in a specific phase.
        /// </summary>
        /// <param name="phaseName">Phase name.</param>
        /// <returns>Total time spent.</returns>
        public TimeSpan GetTotalTime(string phaseName)
        {
            lock (_records)
            {
                return _records
                    .Where(r => r.PhaseName == phaseName)
                    .Aggregate(TimeSpan.Zero, (sum, r) => sum + r.Duration);
            }
        }

        /// <summary>
        /// Gets the average time for a specific phase.
        /// </summary>
        /// <param name="phaseName">Phase name.</param>
        /// <returns>Average time per occurrence.</returns>
        public TimeSpan GetAverageTime(string phaseName)
        {
            lock (_records)
            {
                var phaseRecords = _records.Where(r => r.PhaseName == phaseName).ToList();
                return phaseRecords.Count > 0
                    ? TimeSpan.FromTicks((long)phaseRecords.Average(r => r.Duration.Ticks))
                    : TimeSpan.Zero;
            }
        }

        /// <summary>
        /// Gets all timing records.
        /// </summary>
        public IReadOnlyList<PhaseTimingRecord> GetRecords()
        {
            lock (_records)
            {
                return _records.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets timing summary for all phases.
        /// </summary>
        public IReadOnlyDictionary<string, (TimeSpan Total, TimeSpan Average, int Count)> GetSummary()
        {
            lock (_records)
            {
                return _records
                    .GroupBy(r => r.PhaseName)
                    .ToDictionary(
                        g => g.Key,
                        g => (
                            Total: TimeSpan.FromTicks(g.Sum(r => r.Duration.Ticks)),
                            Average: TimeSpan.FromTicks((long)g.Average(r => r.Duration.Ticks)),
                            Count: g.Count()
                        ));
            }
        }

        /// <summary>
        /// Clears all timing records.
        /// </summary>
        public void Clear()
        {
            lock (_records)
            {
                _records.Clear();
            }
            _timers.Clear();
        }

        private sealed class PhaseTimer
        {
            private readonly Stopwatch _stopwatch = new();

            public TimeSpan Elapsed => _stopwatch.Elapsed;

            public void Restart()
            {
                _stopwatch.Restart();
            }

            public void Stop()
            {
                _stopwatch.Stop();
            }
        }
    }

    /// <summary>
    /// Record of a phase timing measurement.
    /// </summary>
    public record PhaseTimingRecord
    {
        /// <summary>Name of the phase.</summary>
        public string PhaseName { get; init; } = string.Empty;

        /// <summary>Duration of the phase.</summary>
        public TimeSpan Duration { get; init; }

        /// <summary>Generation when timing occurred.</summary>
        public int Generation { get; init; }

        /// <summary>Timestamp of the measurement.</summary>
        public DateTime Timestamp { get; init; }
    }

}
