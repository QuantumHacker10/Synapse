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

    public class PerceptionSystem : IPerceptionSystem
    {
        private readonly Dictionary<Guid, List<PerceptionEvent>> _entityPerceptions = new();
        private readonly Dictionary<Guid, AttentionModel> _attentionModels = new();
        private readonly SpatialIndex _spatialIndex;
        private readonly object _lock = new();

        public PerceptionSystem() { _spatialIndex = new SpatialIndex(20f); }

        public float VisualRange { get; set; } = 100f;
        public float AuditoryRange { get; set; } = 50f;
        public float ProximityRange { get; set; } = 10f;
        public float PerceptionThreshold { get; set; } = 0.1f;
        public bool UseAttentionFiltering { get; set; } = true;
        public bool SimulateOcclusion { get; set; } = true;

        public void Update(SentientEntity entity, WorldStateData worldState)
        {
            var perceptions = new List<PerceptionEvent>();
            var nearbyIds = _spatialIndex.QueryRadius(entity.Position, entity.PerceptionRadius);

            foreach (var eid in nearbyIds)
            {
                if (eid == entity.EntityId || !worldState.Entities.TryGetValue(eid, out var other))
                    continue;
                float dist = entity.DistanceTo(other);

                if (entity.PerceptionCapabilities.Contains(PerceptionType.Visual) && dist <= VisualRange)
                {
                    var p = ProcessVisual(entity, other, dist, worldState);
                    if (p != null)
                        perceptions.Add(p);
                }
                if (entity.PerceptionCapabilities.Contains(PerceptionType.Auditory) && dist <= AuditoryRange)
                {
                    var p = ProcessAuditory(entity, other, dist, worldState);
                    if (p != null)
                        perceptions.Add(p);
                }
                if (entity.PerceptionCapabilities.Contains(PerceptionType.Proximity) && dist <= ProximityRange)
                {
                    var p = ProcessProximity(entity, other, dist);
                    if (p != null)
                        perceptions.Add(p);
                }
                if (entity.PerceptionCapabilities.Contains(PerceptionType.Semantic))
                {
                    var p = ProcessSemantic(entity, other, dist);
                    if (p != null)
                        perceptions.Add(p);
                }
                if (entity.PerceptionCapabilities.Contains(PerceptionType.Emotional))
                {
                    var p = ProcessEmotional(entity, other, dist);
                    if (p != null)
                        perceptions.Add(p);
                }
            }

            if (UseAttentionFiltering)
                perceptions = ApplyAttention(entity, perceptions);
            lock (_lock)
            { _entityPerceptions[entity.EntityId] = perceptions; }
            foreach (var p in perceptions)
                entity.AddPerception(p);
        }

        public IReadOnlyList<PerceptionEvent> GetPerceptions(SentientEntity entity)
        {
            lock (_lock)
                return _entityPerceptions.TryGetValue(entity.EntityId, out var p) ? p.AsReadOnly() : Array.Empty<PerceptionEvent>();
        }

        public IReadOnlyList<PerceptionEvent> FilterPerceptions(SentientEntity entity, PerceptionFilter filters)
        {
            return ApplyFilter(GetPerceptions(entity), filters);
        }

        private PerceptionEvent? ProcessVisual(SentientEntity obs, SentientEntity tgt, float dist, WorldStateData ws)
        {
            if (!obs.CanSee(tgt))
                return null;
            float visFactor = ws.Weather?.Visibility ?? 1f;
            float distFactor = Math.Max(0, 1f - dist / VisualRange);
            float conf = distFactor * visFactor;
            if (dist > VisualRange * 0.8f)
                conf *= 0.5f;
            if (conf < PerceptionThreshold)
                return null;
            var dir = Vector3.Normalize(tgt.Position - obs.Position);
            return new PerceptionEvent(PerceptionType.Visual, tgt.EntityId, ws.Time, conf * visFactor, tgt.Position, dir, $"Visual contact with {tgt.EntityType}", conf);
        }

        private PerceptionEvent? ProcessAuditory(SentientEntity obs, SentientEntity tgt, float dist, WorldStateData ws)
        {
            float intensity = 1f / (1f + dist * dist / (AuditoryRange * AuditoryRange));
            if (SimulateOcclusion)
                intensity *= 0.8f;
            if (ws.Weather?.WindSpeed > 0)
            {
                var wd = new Vector3((float)Math.Cos(ws.Weather.WindDirection), 0, (float)Math.Sin(ws.Weather.WindDirection));
                float we = Vector3.Dot(wd, Vector3.Normalize(tgt.Position - obs.Position));
                intensity *= 1f + we * 0.2f;
            }
            if (intensity < PerceptionThreshold)
                return null;
            var dir = Vector3.Normalize(tgt.Position - obs.Position);
            return new PerceptionEvent(PerceptionType.Auditory, tgt.EntityId, ws.Time, intensity, tgt.Position, dir, $"Auditory from {tgt.EntityType}", intensity * 0.8f);
        }

        private PerceptionEvent? ProcessProximity(SentientEntity obs, SentientEntity tgt, float dist)
        {
            float intensity = Math.Max(0, 1f - dist / ProximityRange);
            if (intensity < PerceptionThreshold)
                return null;
            var dir = Vector3.Normalize(tgt.Position - obs.Position);
            return new PerceptionEvent(PerceptionType.Proximity, tgt.EntityId, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, intensity, tgt.Position, dir, $"Proximity: {dist:F1} units", 1f);
        }

        private PerceptionEvent? ProcessSemantic(SentientEntity obs, SentientEntity tgt, float dist)
        {
            float conf = 0.5f;
            if (obs.Relationships.TryGetValue(tgt.EntityId, out var rel))
                conf = Math.Min(1f, conf + 0.3f);
            var dir = Vector3.Normalize(tgt.Position - obs.Position);
            return new PerceptionEvent(PerceptionType.Semantic, tgt.EntityId, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, conf, tgt.Position, dir, $"Semantic: {tgt.EntityType} ({rel?.Type.ToString() ?? "Unknown"})", conf);
        }

        private PerceptionEvent? ProcessEmotional(SentientEntity obs, SentientEntity tgt, float dist)
        {
            float distFactor = Math.Max(0, 1f - dist / (obs.PerceptionRadius * 0.5f));
            if (distFactor < PerceptionThreshold)
                return null;
            var dir = Vector3.Normalize(tgt.Position - obs.Position);
            return new PerceptionEvent(PerceptionType.Emotional, tgt.EntityId, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, distFactor * 0.6f, tgt.Position, dir, $"Emotional: {tgt.CurrentEmotion}", distFactor * 0.5f);
        }

        private List<PerceptionEvent> ApplyAttention(SentientEntity entity, List<PerceptionEvent> perceptions)
        {
            if (!_attentionModels.TryGetValue(entity.EntityId, out var am))
            { am = new AttentionModel(); _attentionModels[entity.EntityId] = am; }
            return am.FilterPerceptions(perceptions);
        }

        private static List<PerceptionEvent> ApplyFilter(IReadOnlyList<PerceptionEvent> perceptions, PerceptionFilter filter)
        {
            return perceptions.Where(p =>
                (filter.AllowedTypes.Count == 0 || filter.AllowedTypes.Contains(p.Type)) &&
                p.Intensity >= filter.MinIntensity && p.Confidence >= filter.MinConfidence &&
                (filter.AllowedSources.Count == 0 || filter.AllowedSources.Contains(p.Source)) &&
                !filter.ExcludedSources.Contains(p.Source)).ToList();
        }

        public void UpdateSpatialIndex(IEnumerable<SentientEntity> entities) { _spatialIndex.Clear(); foreach (var e in entities) _spatialIndex.Insert(e.EntityId, e.Position); }
        public void RemoveEntity(Guid eid) { lock (_lock) { _entityPerceptions.Remove(eid); _attentionModels.Remove(eid); } }
    }

    public class AttentionModel
    {
        private readonly Dictionary<PerceptionType, float> _weights = new()
        {
            { PerceptionType.Visual, 1f }, { PerceptionType.Auditory, 0.8f }, { PerceptionType.Tactile, 0.6f },
            { PerceptionType.Proximity, 0.9f }, { PerceptionType.Semantic, 0.7f }, { PerceptionType.Emotional, 0.5f }
        };
        private readonly Dictionary<Guid, float> _focus = new();
        private float _arousal;
        private float _radius = 50f;

        public float ArousalLevel { get => _arousal; set => _arousal = Math.Clamp(value, 0, 1); }
        public float FocusRadius { get => _radius; set => _radius = Math.Max(1, value); }

        public List<PerceptionEvent> FilterPerceptions(List<PerceptionEvent> perceptions)
        {
            var filtered = new List<PerceptionEvent>();
            foreach (var p in perceptions)
            {
                float score = p.Intensity * p.Confidence;
                if (_arousal > 0.7f)
                    score *= 1f + (_arousal - 0.7f) * 2f;
                if (_weights.TryGetValue(p.Type, out var w))
                    score *= w;
                if (_focus.TryGetValue(p.Source, out var f))
                    score *= 1f + f;
                if (score >= 0.3f)
                    filtered.Add(p);
            }
            filtered.Sort((a, b) => (b.Intensity * b.Confidence).CompareTo(a.Intensity * a.Confidence));
            return filtered;
        }

        public void FocusOn(Guid eid, float intensity = 1f) { _focus[eid] = Math.Clamp(intensity, 0, 1); }
        public void Unfocus(Guid eid) { _focus.Remove(eid); }

        public void Update(float deltaTime)
        {
            _arousal = Math.Max(0, _arousal - deltaTime * 0.1f);
            foreach (var k in _focus.Keys.ToList())
            { _focus[k] -= deltaTime * 0.05f; if (_focus[k] <= 0) _focus.Remove(k); }
        }
    }

}
