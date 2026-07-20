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

    public class EntityTickScheduler
    {
        private readonly Dictionary<Guid, SentientEntity> _entities = new();
        private readonly SortedSet<(int Priority, Guid EntityId)> _priorityQueue = new(Comparer<(int Priority, Guid EntityId)>.Create((a, b) => a.Priority != b.Priority ? b.Priority.CompareTo(a.Priority) : a.EntityId.CompareTo(b.EntityId)));
        private readonly WorldStateManager _worldState;
        private readonly PerceptionSystem _perceptionSystem;
        private readonly EmotionalModel _emotionalModel;
        private readonly BehaviorDebugger? _debugger;
        private readonly BehaviorAnalytics _analytics;
        private readonly object _lock = new();
        private SchedulerStats _stats = new();

        public EntityTickScheduler(WorldStateManager worldState, PerceptionSystem perceptionSystem, EmotionalModel emotionalModel, BehaviorDebugger? debugger = null, BehaviorAnalytics? analytics = null)
        {
            _worldState = worldState;
            _perceptionSystem = perceptionSystem;
            _emotionalModel = emotionalModel;
            _debugger = debugger;
            _analytics = analytics ?? new BehaviorAnalytics();
        }

        public int ActiveEntityCount { get { lock (_lock) return _entities.Values.Count(e => e.CurrentState == BehaviorState.Active); } }
        public int DormantEntityCount { get { lock (_lock) return _entities.Values.Count(e => e.CurrentState == BehaviorState.Dormant); } }
        public SchedulerStats Stats { get { lock (_lock) return _stats; } }

        public void RegisterEntity(SentientEntity entity)
        {
            lock (_lock)
            {
                _entities[entity.EntityId] = entity;
                _priorityQueue.Add((entity.UpdatePriority, entity.EntityId));
            }
        }

        public void UnregisterEntity(Guid entityId)
        {
            lock (_lock)
            {
                if (_entities.TryGetValue(entityId, out var e))
                    _priorityQueue.Remove((e.UpdatePriority, entityId));
                _entities.Remove(entityId);
            }
        }

        public async Task TickAllEntitiesAsync(float deltaTime, CancellationToken ct = default)
        {
            var startTime = Stopwatch.GetTimestamp();
            _stats = new SchedulerStats { FrameStartTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond };

            var (active, dormant) = PartitionByRelevance();

            var activeEntities = active.ToList();
            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount, CancellationToken = ct };

            await Parallel.ForEachAsync(activeEntities, parallelOptions, async (entity, token) =>
            {
                var entityStart = Stopwatch.GetTimestamp();
                await Task.Run(() => TickEntity(entity, deltaTime), token);
                var entityTime = (Stopwatch.GetTimestamp() - entityStart) / (double)TimeSpan.TicksPerMillisecond;
                lock (_lock)
                { _stats.EntityTickTimes[entity.EntityId] = entityTime; }
            });

            foreach (var entity in dormant)
            {
                SimulateDormant(entity, deltaTime);
            }

            var frameTime = (Stopwatch.GetTimestamp() - startTime) / (double)TimeSpan.TicksPerMillisecond;
            lock (_lock)
            {
                _stats.FrameTimeMs = frameTime;
                _stats.ActiveEntityCount = activeEntities.Count;
                _stats.DormantEntityCount = dormant.Count();
                _stats.TotalEntities = _entities.Count;
            }
        }

        private (IEnumerable<SentientEntity> Active, IEnumerable<SentientEntity> Dormant) PartitionByRelevance()
        {
            var active = new List<SentientEntity>();
            var dormant = new List<SentientEntity>();

            foreach (var entity in _entities.Values)
            {
                if (entity.CurrentState == BehaviorState.Terminated)
                    continue;
                if (entity.CurrentState == BehaviorState.Active || entity.CurrentState == BehaviorState.Transitioning)
                { active.Add(entity); continue; }
                if (entity.CurrentState == BehaviorState.Dormant)
                { dormant.Add(entity); continue; }

                bool isRelevant = entity.CurrentState != BehaviorState.Idle ||
                    entity.Needs.Values.Any(v => v > 0.8f) ||
                    entity.Relationships.Values.Any(r => r.Type == RelationshipType.Hostile && r.Strength > 0.5f);

                if (isRelevant)
                    active.Add(entity);
                else
                    dormant.Add(entity);
            }
            return (active, dormant);
        }

        private void TickEntity(SentientEntity entity, float deltaTime)
        {
            var worldState = _worldState.CreateSnapshot();
            _perceptionSystem.Update(entity, worldState);

            var context = new EntityContext
            {
                Entity = entity,
                WorldState = worldState,
                DeltaTime = deltaTime,
                PerceptionEvents = entity.GetPendingPerceptions().ToList(),
                Memory = entity.MemorySystem!,
                EmotionalState = entity.CurrentEmotion,
                Relationships = entity.Relationships.ToDictionary(kv => kv.Key, kv => kv.Value.Type)
            };

            var tickStart = Stopwatch.GetTimestamp();
            entity.Update(context, deltaTime);
            var tickTime = (float)(Stopwatch.GetTimestamp() - tickStart) / TimeSpan.TicksPerMillisecond;

            _analytics?.RecordBehavior(entity.CurrentState.ToString(), entity.EntityId.ToString(), tickTime);
            _debugger?.RecordReplayFrame(worldState, _worldState.GetAllEntities());
        }

        private void SimulateDormant(SentientEntity entity, float deltaTime)
        {
            entity.Position += entity.Velocity * deltaTime * 0.1f;
            foreach (var need in entity.Needs.Keys.ToList())
                entity.Needs[need] = Math.Min(1f, entity.Needs[need] + deltaTime * 0.005f);
        }

        public void RebuildPriorityQueue()
        {
            lock (_lock)
            {
                _priorityQueue.Clear();
                foreach (var e in _entities.Values)
                    _priorityQueue.Add((e.UpdatePriority, e.EntityId));
            }
        }
    }

    public class SchedulerStats
    {
        public double FrameStartTime { get; set; }
        public double FrameTimeMs { get; set; }
        public int ActiveEntityCount { get; set; }
        public int DormantEntityCount { get; set; }
        public int TotalEntities { get; set; }
        public Dictionary<Guid, double> EntityTickTimes { get; set; } = new();
        public double AverageTickTimeMs => EntityTickTimes.Count > 0 ? EntityTickTimes.Values.Average() : 0;
        public double MaxTickTimeMs => EntityTickTimes.Count > 0 ? EntityTickTimes.Values.Max() : 0;
    }

}
