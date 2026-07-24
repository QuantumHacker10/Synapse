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

    public class WorldStateManager
    {
        private readonly Dictionary<Guid, SentientEntity> _entities = new();
        private readonly Queue<WorldEvent> _events = new();
        private readonly SpatialIndex _spatialIndex;
        private readonly object _lock = new();
        private double _currentTime;
        private WeatherConditions _weather;

        public WorldStateManager()
        {
            _spatialIndex = new SpatialIndex(15f);
            _weather = new WeatherConditions(20f, 0.5f, 5f, 0f, 1f, 1f, WeatherType.Clear);
        }

        public double CurrentTime { get => _currentTime; set => _currentTime = value; }
        public WeatherConditions Weather { get => _weather; set => _weather = value; }

        public void AddEntity(SentientEntity entity) { lock (_lock) { _entities[entity.EntityId] = entity; _spatialIndex.Insert(entity.EntityId, entity.Position); } }
        public void RemoveEntity(Guid entityId) { lock (_lock) { _entities.Remove(entityId); _spatialIndex.Remove(entityId); } }
        public SentientEntity? GetEntity(Guid entityId) { lock (_lock) { return _entities.TryGetValue(entityId, out var e) ? e : null; } }
        public IReadOnlyDictionary<Guid, SentientEntity> GetAllEntities() { lock (_lock) { return _entities.ToDictionary(kv => kv.Key, kv => kv.Value); } }
        public int EntityCount { get { lock (_lock) return _entities.Count; } }

        public void UpdateSpatialIndex()
        {
            lock (_lock)
            {
                _spatialIndex.Clear();
                foreach (var e in _entities.Values)
                    _spatialIndex.Insert(e.EntityId, e.Position);
            }
        }

        public List<Guid> QueryNearby(Vector3 position, float radius) => _spatialIndex.QueryRadius(position, radius);
        public List<Guid> QueryBox(Vector3 min, Vector3 max) => _spatialIndex.QueryBox(min, max);
        public Guid? QueryNearest(Vector3 position, float maxDist = float.MaxValue) => _spatialIndex.QueryNearest(position, _entities.ToDictionary(kv => kv.Key, kv => kv.Value.Position), maxDist);

        public void QueueEvent(WorldEvent evt) { lock (_lock) _events.Enqueue(evt); }
        public WorldEvent? DequeueEvent() { lock (_lock) { return _events.Count > 0 ? _events.Dequeue() : null; } }
        public int EventCount { get { lock (_lock) return _events.Count; } }

        public WorldStateData CreateSnapshot()
        {
            lock (_lock)
            {
                return new WorldStateData
                {
                    Entities = new Dictionary<Guid, SentientEntity>(_entities),
                    Time = _currentTime,
                    Weather = _weather,
                    Events = new Queue<WorldEvent>(_events),
                    SpatialIndex = _spatialIndex
                };
            }
        }

        public void Update(float deltaTime)
        {
            _currentTime += deltaTime;
            UpdateSpatialIndex();
            foreach (var e in _entities.Values)
                e.Update(CreateContext(e), deltaTime);
        }

        private EntityContext CreateContext(SentientEntity entity)
        {
            return new EntityContext
            {
                Entity = entity,
                WorldState = CreateSnapshot(),
                DeltaTime = 0.016f,
                PerceptionEvents = entity.GetPendingPerceptions().ToList(),
                Memory = entity.MemorySystem!,
                EmotionalState = entity.CurrentEmotion,
                Relationships = entity.Relationships.ToDictionary(kv => kv.Key, kv => kv.Value.Type)
            };
        }

        public void ProcessEvents()
        {
            lock (_lock)
            {
                while (_events.Count > 0)
                {
                    var evt = _events.Dequeue();
                    ProcessEvent(evt);
                }
            }
        }

        private void ProcessEvent(WorldEvent evt)
        {
            var nearby = _spatialIndex.QueryRadius(evt.Location, evt.Radius);
            foreach (var eid in nearby)
            {
                if (!_entities.TryGetValue(eid, out var entity))
                    continue;
                if (eid == evt.SourceEntity)
                    continue;
                var perception = new PerceptionEvent(
                    evt.Intensity > 0.5f ? PerceptionType.Auditory : PerceptionType.Visual,
                    evt.SourceEntity, evt.Timestamp, evt.Intensity, evt.Location,
                    Vector3.Normalize(entity.Position - evt.Location),
                    $"{evt.EventType}: {string.Join(", ", evt.Metadata.Take(3).Select(kv => $"{kv.Key}={kv.Value}"))}",
                    Math.Clamp(evt.Intensity * (1f - Vector3.Distance(entity.Position, evt.Location) / evt.Radius), 0, 1));
                entity.AddPerception(perception);
            }
        }

        public Dictionary<string, object> GetWorldStats()
        {
            lock (_lock)
            {
                return new Dictionary<string, object>
                {
                    { "EntityCount", _entities.Count },
                    { "EventCount", _events.Count },
                    { "CurrentTime", _currentTime },
                    { "Weather", _weather.Type.ToString() },
                    { "Temperature", _weather.Temperature },
                    { "Visibility", _weather.Visibility },
                    { "ActiveEntities", _entities.Values.Count(e => e.CurrentState == BehaviorState.Active) },
                    { "IdleEntities", _entities.Values.Count(e => e.CurrentState == BehaviorState.Idle) },
                    { "TerminatedEntities", _entities.Values.Count(e => e.CurrentState == BehaviorState.Terminated) }
                };
            }
        }
    }

}
