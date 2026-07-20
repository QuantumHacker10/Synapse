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

    public class AdvancedPerceptionProcessor
    {
        private readonly Dictionary<Guid, PerceptionProcessingState> _processingStates = new();
        private readonly SensoryIntegrationEngine _integrationEngine;
        private readonly PredictionEngine _predictionEngine;

        public AdvancedPerceptionProcessor()
        {
            _integrationEngine = new SensoryIntegrationEngine();
            _predictionEngine = new PredictionEngine();
        }

        public float NoiseFilterStrength { get; set; } = 0.3f;
        public float PredictionHorizon { get; set; } = 2f;
        public bool EnableCrossModalIntegration { get; set; } = true;
        public bool EnableTemporalSmoothing { get; set; } = true;

        public List<PerceptionEvent> ProcessPerceptions(SentientEntity entity, List<PerceptionEvent> rawPerceptions, float deltaTime)
        {
            var state = GetOrCreate(entity.EntityId);
            var processed = new List<PerceptionEvent>();

            var filtered = ApplyNoiseFilter(rawPerceptions, state);
            var integrated = EnableCrossModalIntegration ? _integrationEngine.Integrate(filtered) : filtered;
            var smoothed = EnableTemporalSmoothing ? ApplyTemporalSmoothing(integrated, state, deltaTime) : integrated;

            foreach (var perception in smoothed)
            {
                var relevance = CalculateRelevance(entity, perception);
                if (relevance >= 0.2f)
                {
                    var enhancedConfidence = Math.Clamp(perception.Confidence * (0.5f + relevance * 0.5f), 0, 1);
                    processed.Add(new PerceptionEvent(
                        perception.Type, perception.Source, perception.Timestamp,
                        perception.Intensity, perception.Location, perception.Direction,
                        perception.SemanticContent, enhancedConfidence));
                }
            }

            UpdateState(state, processed, deltaTime);
            return processed;
        }

        private List<PerceptionEvent> ApplyNoiseFilter(List<PerceptionEvent> perceptions, PerceptionProcessingState state)
        {
            var filtered = new List<PerceptionEvent>();
            foreach (var p in perceptions)
            {
                if (state.PerceivedRecently(p.Source, p.Type))
                {
                    float similarity = CalculateSimilarity(p, state.GetLastPerception(p.Source, p.Type));
                    if (similarity > 0.8f && p.Intensity < state.GetLastPerception(p.Source, p.Type).Intensity * 1.2f)
                        continue;
                }
                if (p.Intensity < NoiseFilterStrength * 0.5f)
                    continue;
                filtered.Add(p);
            }
            return filtered;
        }

        private List<PerceptionEvent> ApplyTemporalSmoothing(List<PerceptionEvent> perceptions, PerceptionProcessingState state, float deltaTime)
        {
            var smoothed = new List<PerceptionEvent>();
            foreach (var p in perceptions)
            {
                if (state.GetSmoothingBuffer(p.Source, p.Type).Count > 0)
                {
                    var buffer = state.GetSmoothingBuffer(p.Source, p.Type);
                    buffer.Add(p);
                    if (buffer.Count > 5)
                        buffer.RemoveAt(0);

                    float avgIntensity = buffer.Average(b => b.Intensity);
                    float avgConfidence = buffer.Average(b => b.Confidence);
                    smoothed.Add(new PerceptionEvent(
                        p.Type, p.Source, p.Timestamp, avgIntensity, p.Location, p.Direction,
                        p.SemanticContent, avgConfidence));
                }
                else
                {
                    state.GetSmoothingBuffer(p.Source, p.Type).Add(p);
                    smoothed.Add(p);
                }
            }
            return smoothed;
        }

        private float CalculateRelevance(SentientEntity entity, PerceptionEvent perception)
        {
            float relevance = 0;

            if (entity.Relationships.TryGetValue(perception.Source, out var rel))
            {
                relevance += rel.Type switch
                {
                    RelationshipType.Hostile => 0.8f,
                    RelationshipType.Friendly => 0.6f,
                    RelationshipType.Parent => 0.7f,
                    RelationshipType.Child => 0.7f,
                    RelationshipType.Partner => 0.65f,
                    RelationshipType.Master => 0.5f,
                    RelationshipType.Servant => 0.4f,
                    RelationshipType.Sibling => 0.5f,
                    _ => 0.2f
                };
                relevance *= rel.Strength;
            }

            if (perception.Type == PerceptionType.Tactile)
                relevance += 0.4f;
            if (perception.Type == PerceptionType.Visual && perception.Intensity > 0.7f)
                relevance += 0.3f;
            if (entity.Needs.Values.Any(v => v > 0.8f))
                relevance += 0.2f;

            float distFactor = 1f - (Vector3.Distance(entity.Position, perception.Location) / entity.PerceptionRadius);
            relevance *= (0.5f + distFactor * 0.5f);

            return Math.Clamp(relevance, 0, 1);
        }

        private static float CalculateSimilarity(PerceptionEvent a, PerceptionEvent b)
        {
            if (a.Type != b.Type || a.Source != b.Source)
                return 0;
            float posSim = 1f - Math.Min(1f, Vector3.Distance(a.Location, b.Location) / 10f);
            float intSim = 1f - Math.Abs(a.Intensity - b.Intensity);
            return (posSim + intSim) / 2f;
        }

        private void UpdateState(PerceptionProcessingState state, List<PerceptionEvent> processed, float deltaTime)
        {
            state.UpdateTimestamps(processed);
        }

        private PerceptionProcessingState GetOrCreate(Guid entityId)
        {
            if (!_processingStates.TryGetValue(entityId, out var state))
            {
                state = new PerceptionProcessingState();
                _processingStates[entityId] = state;
            }
            return state;
        }
    }

    public class PerceptionProcessingState
    {
        private readonly Dictionary<(Guid, PerceptionType), PerceptionEvent> _lastPerceptions = new();
        private readonly Dictionary<(Guid, PerceptionType), List<PerceptionEvent>> _smoothingBuffers = new();
        private readonly Dictionary<(Guid, PerceptionType), double> _lastPerceivedTime = new();

        public bool PerceivedRecently(Guid source, PerceptionType type, float threshold = 0.5f)
        {
            var key = (source, type);
            if (!_lastPerceivedTime.TryGetValue(key, out var time))
                return false;
            return (Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond - time) < threshold;
        }

        public PerceptionEvent? GetLastPerception(Guid source, PerceptionType type)
        {
            return _lastPerceptions.TryGetValue((source, type), out var p) ? p : default;
        }

        public List<PerceptionEvent> GetSmoothingBuffer(Guid source, PerceptionType type)
        {
            var key = (source, type);
            if (!_smoothingBuffers.TryGetValue(key, out var buf))
            { buf = new List<PerceptionEvent>(); _smoothingBuffers[key] = buf; }
            return buf;
        }

        public void UpdateTimestamps(List<PerceptionEvent> perceptions)
        {
            double now = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
            foreach (var p in perceptions)
            {
                var key = (p.Source, p.Type);
                _lastPerceptions[key] = p;
                _lastPerceivedTime[key] = now;
            }
        }
    }

    public class SensoryIntegrationEngine
    {
        public List<PerceptionEvent> Integrate(List<PerceptionEvent> perceptions)
        {
            var integrated = new List<PerceptionEvent>();
            var grouped = perceptions.GroupBy(p => p.Source);

            foreach (var group in grouped)
            {
                var sourcePerceptions = group.ToList();
                if (sourcePerceptions.Count <= 1)
                {
                    integrated.AddRange(sourcePerceptions);
                    continue;
                }

                var visual = sourcePerceptions.FirstOrDefault(p => p.Type == PerceptionType.Visual);
                var auditory = sourcePerceptions.FirstOrDefault(p => p.Type == PerceptionType.Auditory);
                var proximity = sourcePerceptions.FirstOrDefault(p => p.Type == PerceptionType.Proximity);

                if (visual.Intensity > 0 && auditory.Intensity > 0)
                {
                    float integratedIntensity = (visual.Intensity * 0.6f + auditory.Intensity * 0.4f);
                    float integratedConfidence = Math.Min(1f, (visual.Confidence + auditory.Confidence) * 0.7f);
                    integrated.Add(new PerceptionEvent(
                        PerceptionType.Visual, group.Key, visual.Timestamp,
                        integratedIntensity, visual.Location, visual.Direction,
                        $"Multimodal: {visual.SemanticContent} + {auditory.SemanticContent}",
                        integratedConfidence));
                }
                else
                {
                    integrated.AddRange(sourcePerceptions);
                }
            }
            return integrated;
        }
    }

    public class PredictionEngine
    {
        private readonly Dictionary<Guid, List<Vector3>> _positionHistory = new();
        private readonly Dictionary<Guid, List<Vector3>> _velocityHistory = new();

        public Vector3 PredictPosition(Guid entityId, Vector3 currentPosition, Vector3 currentVelocity, float timeAhead)
        {
            if (!_positionHistory.TryGetValue(entityId, out var positions) || positions.Count < 3)
                return currentPosition + currentVelocity * timeAhead;

            var recentPositions = positions.TakeLast(10).ToList();
            float totalDeltaX = 0, totalDeltaY = 0, totalDeltaZ = 0;
            for (int i = 1; i < recentPositions.Count; i++)
            {
                totalDeltaX += recentPositions[i].X - recentPositions[i - 1].X;
                totalDeltaY += recentPositions[i].Y - recentPositions[i - 1].Y;
                totalDeltaZ += recentPositions[i].Z - recentPositions[i - 1].Z;
            }
            float avgDeltaX = totalDeltaX / (recentPositions.Count - 1);
            float avgDeltaY = totalDeltaY / (recentPositions.Count - 1);
            float avgDeltaZ = totalDeltaZ / (recentPositions.Count - 1);

            return currentPosition + new Vector3(avgDeltaX, avgDeltaY, avgDeltaZ) * timeAhead;
        }

        public float PredictThreatLevel(Guid entityId, List<PerceptionEvent> recentPerceptions, float timeHorizon)
        {
            var threatPerceptions = recentPerceptions
                .Where(p => p.Source == entityId && (p.Type == PerceptionType.Tactile || p.SemanticContent.Contains("threat")))
                .ToList();

            if (threatPerceptions.Count == 0)
                return 0;

            float avgIntensity = threatPerceptions.Average(p => p.Intensity);
            float frequency = threatPerceptions.Count / Math.Max(1f, timeHorizon);
            float recencyFactor = threatPerceptions.Count > 0 ?
                1f - (float)(Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond - threatPerceptions.Max(p => p.Timestamp)) / timeHorizon : 0;

            return Math.Clamp(avgIntensity * frequency * recencyFactor, 0, 1);
        }

        public void UpdateHistory(Guid entityId, Vector3 position, Vector3 velocity)
        {
            if (!_positionHistory.TryGetValue(entityId, out var positions))
            { positions = new List<Vector3>(); _positionHistory[entityId] = positions; }
            positions.Add(position);
            if (positions.Count > 20)
                positions.RemoveAt(0);

            if (!_velocityHistory.TryGetValue(entityId, out var velocities))
            { velocities = new List<Vector3>(); _velocityHistory[entityId] = velocities; }
            velocities.Add(velocity);
            if (velocities.Count > 20)
                velocities.RemoveAt(0);
        }
    }

}
