// =============================================================================
// EntityBehaviorSystem.cs
// GDNN.Sentience - Complete Entity Behavior System for G-DNN Engine
// =============================================================================

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace GDNN.Sentience
{

    public class SentienceEventBus
    {
        private readonly Dictionary<string, List<Delegate>> _handlers = new();
        private readonly ConcurrentQueue<SentienceEvent> _eventQueue = new();
        private readonly object _lock = new();

        public void Subscribe<T>(string eventName, Action<T> handler) where T : SentienceEvent
        {
            lock (_lock)
            {
                if (!_handlers.TryGetValue(eventName, out var list))
                { list = new List<Delegate>(); _handlers[eventName] = list; }
                list.Add(handler);
            }
        }

        public void Unsubscribe<T>(string eventName, Action<T> handler) where T : SentienceEvent
        {
            lock (_lock)
            {
                if (_handlers.TryGetValue(eventName, out var list))
                    list.Remove(handler);
            }
        }

        public void Publish<T>(T evt) where T : SentienceEvent
        {
            _eventQueue.Enqueue(evt);
        }

        public void ProcessEvents()
        {
            while (_eventQueue.TryDequeue(out var evt))
            {
                lock (_lock)
                {
                    if (_handlers.TryGetValue(evt.EventName, out var list))
                    {
                        foreach (var handler in list)
                        {
                            if (handler.GetType().IsGenericType && handler.GetType().GetGenericTypeDefinition() == typeof(Action<>))
                            {
                                handler.DynamicInvoke(evt);
                            }
                        }
                    }
                }
            }
        }

        public void Clear() { lock (_lock) _handlers.Clear(); while (_eventQueue.TryDequeue(out _)) { } }
        public int PendingEventCount => _eventQueue.Count;
    }

    public class SentienceEvent
    {
        public string EventName { get; set; } = string.Empty;
        public double Timestamp { get; set; } = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
        public Guid SourceEntity { get; set; }
        public Dictionary<string, object> Data { get; set; } = new();
    }

    public class PerceptionEventData : SentienceEvent
    {
        public PerceptionEvent Perception { get; set; }
    }

    public class DamageEventData : SentienceEvent
    {
        public float DamageAmount { get; set; }
        public Guid DamageSource { get; set; }
    }

    public class StateChangeEventData : SentienceEvent
    {
        public BehaviorState PreviousState { get; set; }
        public BehaviorState NewState { get; set; }
    }

}
