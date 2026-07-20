// =============================================================================
// NeatGEvolutionEngine.EvolutionEventBus.cs — NEAT-G partial module
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
    /// Central event bus for the NEAT-G evolution engine.
    /// Provides publish-subscribe pattern for decoupled communication between
    /// evolution components. Supports typed event handlers and async subscriptions.
    /// </summary>
    public sealed class EvolutionEventBus : IAsyncDisposable
    {
        private readonly ConcurrentDictionary<Type, List<Delegate>> _handlers;
        private readonly ConcurrentQueue<(Type EventType, object EventData, DateTime Timestamp)> _eventLog;
        private readonly int _maxLogSize;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the EvolutionEventBus class.
        /// </summary>
        /// <param name="maxLogSize">Maximum number of events to retain in the log.</param>
        public EvolutionEventBus(int maxLogSize = 10000)
        {
            _handlers = new ConcurrentDictionary<Type, List<Delegate>>();
            _eventLog = new ConcurrentQueue<(Type, object, DateTime)>();
            _maxLogSize = maxLogSize;
        }

        /// <summary>Total events published.</summary>
        private long _totalEventsPublished;
        public long TotalEventsPublished => _totalEventsPublished;

        /// <summary>
        /// Subscribes a handler for a specific event type.
        /// </summary>
        /// <typeparam name="TEvent">The event type.</typeparam>
        /// <param name="handler">The event handler.</param>
        public void Subscribe<TEvent>(Action<TEvent> handler) where TEvent : class
        {
            var eventType = typeof(TEvent);
            _handlers.AddOrUpdate(eventType,
                _ => new List<Delegate> { handler },
                (_, existing) =>
                {
                    lock (existing)
                    {
                        existing.Add(handler);
                    }
                    return existing;
                });
        }

        /// <summary>
        /// Subscribes an async handler for a specific event type.
        /// </summary>
        /// <typeparam name="TEvent">The event type.</typeparam>
        /// <param name="handler">The async event handler.</param>
        public void SubscribeAsync<TEvent>(Func<TEvent, Task> handler) where TEvent : class
        {
            var eventType = typeof(TEvent);
            _handlers.AddOrUpdate(eventType,
                _ => new List<Delegate> { handler },
                (_, existing) =>
                {
                    lock (existing)
                    {
                        existing.Add(handler);
                    }
                    return existing;
                });
        }

        /// <summary>
        /// Unsubscribes a handler for a specific event type.
        /// </summary>
        /// <typeparam name="TEvent">The event type.</typeparam>
        /// <param name="handler">The handler to unsubscribe.</param>
        public void Unsubscribe<TEvent>(Action<TEvent> handler) where TEvent : class
        {
            var eventType = typeof(TEvent);
            if (_handlers.TryGetValue(eventType, out var handlers))
            {
                lock (handlers)
                {
                    handlers.Remove(handler);
                }
            }
        }

        /// <summary>
        /// Publishes an event to all subscribed handlers.
        /// </summary>
        /// <typeparam name="TEvent">The event type.</typeparam>
        /// <param name="eventData">The event data.</param>
        public async Task PublishAsync<TEvent>(TEvent eventData) where TEvent : class
        {
            Interlocked.Increment(ref _totalEventsPublished);

            _eventLog.Enqueue((typeof(TEvent), eventData, DateTime.UtcNow));
            while (_eventLog.Count > _maxLogSize)
            {
                _eventLog.TryDequeue(out _);
            }

            if (!_handlers.TryGetValue(typeof(TEvent), out var handlers))
                return;

            List<Delegate> snapshot;
            lock (handlers)
            {
                snapshot = handlers.ToList();
            }

            foreach (var handler in snapshot)
            {
                try
                {
                    if (handler is Action<TEvent> syncHandler)
                    {
                        syncHandler(eventData);
                    }
                    else if (handler is Func<TEvent, Task> asyncHandler)
                    {
                        await asyncHandler(eventData).ConfigureAwait(false);
                    }
                }
                catch (Exception ex)
                {
                    SynapseLogger.Default.Warn("NeatGEvolution", $"Event handler failed for event type '{typeof(TEvent).Name}'.", ex);
                }
            }
        }

        /// <summary>
        /// Publishes an event synchronously (non-async handlers only).
        /// </summary>
        /// <typeparam name="TEvent">The event type.</typeparam>
        /// <param name="eventData">The event data.</param>
        public void Publish<TEvent>(TEvent eventData) where TEvent : class
        {
            Interlocked.Increment(ref _totalEventsPublished);

            _eventLog.Enqueue((typeof(TEvent), eventData, DateTime.UtcNow));
            while (_eventLog.Count > _maxLogSize)
            {
                _eventLog.TryDequeue(out _);
            }

            if (!_handlers.TryGetValue(typeof(TEvent), out var handlers))
                return;

            List<Delegate> snapshot;
            lock (handlers)
            {
                snapshot = handlers.ToList();
            }

            foreach (var handler in snapshot)
            {
                try
                {
                    if (handler is Action<TEvent> syncHandler)
                    {
                        syncHandler(eventData);
                    }
                }
                catch (Exception ex)
                {
                    SynapseLogger.Default.Warn("NeatGEvolution", $"Sync event handler failed for event type '{typeof(TEvent).Name}'.", ex);
                }
            }
        }

        /// <summary>
        /// Gets the event log filtered by event type.
        /// </summary>
        /// <typeparam name="TEvent">Event type to filter by.</typeparam>
        /// <param name="count">Maximum number of events to return.</param>
        /// <returns>Filtered events.</returns>
        public IReadOnlyList<(TEvent Data, DateTime Timestamp)> GetEventLog<TEvent>(int count = 100) where TEvent : class
        {
            return _eventLog
                .Where(e => e.EventType == typeof(TEvent))
                .Take(count)
                .Select(e => ((TEvent)e.EventData, e.Timestamp))
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Clears the event log.
        /// </summary>
        public void ClearLog()
        {
            while (_eventLog.TryDequeue(out _))
            { }
        }

        /// <summary>
        /// Gets the number of handlers for a specific event type.
        /// </summary>
        /// <typeparam name="TEvent">Event type.</typeparam>
        public int GetHandlerCount<TEvent>() where TEvent : class
        {
            if (_handlers.TryGetValue(typeof(TEvent), out var handlers))
            {
                lock (handlers)
                {
                    return handlers.Count;
                }
            }
            return 0;
        }

        /// <inheritdoc/>
        public ValueTask DisposeAsync()
        {
            if (_disposed)
                return ValueTask.CompletedTask;
            _disposed = true;
            _handlers.Clear();
            ClearLog();
            return ValueTask.CompletedTask;
        }
    }

    /// <summary>
    /// Event data for genome evaluation completion.
    /// </summary>
    public sealed class GenomeEvaluatedEvent
    {
        /// <summary>The evaluated genome.</summary>
        public GeoGenome Genome { get; init; } = null!;

        /// <summary>Time taken for evaluation.</summary>
        public TimeSpan EvaluationTime { get; init; }

        /// <summary>Generation when evaluation occurred.</summary>
        public int Generation { get; init; }
    }

    /// <summary>
    /// Event data for species creation.
    /// </summary>
    public sealed class SpeciesCreatedEvent
    {
        /// <summary>New species information.</summary>
        public SpeciesInfo Species { get; init; } = default!;

        /// <summary>Generation when created.</summary>
        public int Generation { get; init; }

        /// <summary>Number of initial members.</summary>
        public int InitialMemberCount { get; init; }
    }

    /// <summary>
    /// Event data for species extinction.
    /// </summary>
    public sealed class SpeciesExtinctEvent
    {
        /// <summary>Extinct species information.</summary>
        public SpeciesInfo Species { get; init; } = default!;

        /// <summary>Generation when extinct.</summary>
        public int Generation { get; init; }

        /// <summary>Reason for extinction.</summary>
        public string Reason { get; init; } = string.Empty;

        /// <summary>Number of members reassignment.</summary>
        public int MembersReassigned { get; init; }
    }

    /// <summary>
    /// Event data for evolution phase transition.
    /// </summary>
    public sealed class PhaseTransitionEvent
    {
        /// <summary>Previous evolution phase.</summary>
        public string FromPhase { get; init; } = string.Empty;

        /// <summary>New evolution phase.</summary>
        public string ToPhase { get; init; } = string.Empty;

        /// <summary>Generation of transition.</summary>
        public int Generation { get; init; }

        /// <summary>Duration of previous phase.</summary>
        public TimeSpan PhaseDuration { get; init; }
    }

}
