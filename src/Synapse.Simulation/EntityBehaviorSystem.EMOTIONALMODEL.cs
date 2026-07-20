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

    public class EmotionalModel : IEmotionalModel
    {
        private readonly Dictionary<Guid, EmotionalStateData> _entityEmotions = new();
        private readonly EmotionWheel _wheel;
        private readonly EmpathyModel _empathy;

        private static readonly Dictionary<EmotionalState, EmotionalState> Opposites = new()
        {
            { EmotionalState.Happy, EmotionalState.Sad }, { EmotionalState.Angry, EmotionalState.Fearful },
            { EmotionalState.Surprised, EmotionalState.Anticipating }, { EmotionalState.Disgusted, EmotionalState.Trusting },
            { EmotionalState.Calm, EmotionalState.Excited }, { EmotionalState.Bored, EmotionalState.Curious },
            { EmotionalState.Confused, EmotionalState.Determined }, { EmotionalState.Anxious, EmotionalState.Calm }
        };

        public EmotionalModel() { _wheel = new EmotionWheel(); _empathy = new EmpathyModel(); }
        public float DefaultInertia { get; set; } = 0.7f;
        public float DefaultDecayRate { get; set; } = 0.1f;

        public void Update(SentientEntity entity, IReadOnlyList<PerceptionEvent> events, float deltaTime)
        {
            var sd = GetOrCreate(entity.EntityId);
            foreach (var e in events)
            { var r = ReactToEvent(entity, e); ApplyResponse(sd, r, entity.EmotionalInertia); }
            _empathy.Update(entity, sd, deltaTime);
            ApplyDecay(sd, deltaTime, entity.EmotionalDecayRate);
            sd.Inertia = entity.EmotionalInertia;
            sd.UpdateDominantEmotion();
        }

        public EmotionalState GetCurrentState(SentientEntity entity) => GetOrCreate(entity.EntityId).DominantEmotion;

        public EmotionalResponse ReactToEvent(SentientEntity entity, PerceptionEvent perception)
        {
            var sd = GetOrCreate(entity.EntityId);
            var (emo, intensity) = DetermineResponse(entity, perception);
            float eff = intensity * (1f - sd.Inertia * 0.5f);
            if (sd.EmotionIntensities.TryGetValue(emo, out var ci) && Opposites.TryGetValue(emo, out var opp) && sd.EmotionIntensities.TryGetValue(opp, out var oi))
                eff *= Math.Max(0, 1f - oi * 0.5f);
            eff = Math.Clamp(eff, 0, 1);
            return new EmotionalResponse(emo, eff, 2f * (1f + eff * 3f), $"Response to {perception.Type}", DefaultDecayRate);
        }

        public IReadOnlyDictionary<EmotionalState, float> GetEmotionalProfile(SentientEntity entity)
        {
            return GetOrCreate(entity.EntityId).EmotionIntensities.ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        private EmotionalStateData GetOrCreate(Guid eid)
        {
            if (!_entityEmotions.TryGetValue(eid, out var sd))
            { sd = new EmotionalStateData(); _entityEmotions[eid] = sd; }
            return sd;
        }

        private (EmotionalState, float) DetermineResponse(SentientEntity entity, PerceptionEvent p)
        {
            float baseInt = p.Intensity * p.Confidence;
            return p.Type switch
            {
                PerceptionType.Visual when entity.Relationships.TryGetValue(p.Source, out var r) => r.Type switch
                {
                    RelationshipType.Hostile => (EmotionalState.Angry, baseInt * 0.8f),
                    RelationshipType.Friendly => (EmotionalState.Happy, baseInt * 0.6f),
                    RelationshipType.Parent => (EmotionalState.Trusting, baseInt * 0.5f),
                    RelationshipType.Child => (EmotionalState.Happy, baseInt * 0.7f),
                    _ => (EmotionalState.Surprised, baseInt * 0.4f)
                },
                PerceptionType.Visual => (EmotionalState.Curious, baseInt * entity.PersonalityTraits.GetValueOrDefault("Openness", 0.5f)),
                PerceptionType.Auditory when p.SemanticContent.Contains("threat") => (EmotionalState.Fearful, baseInt * 0.7f),
                PerceptionType.Auditory when p.SemanticContent.Contains("help") => (EmotionalState.Trusting, baseInt * 0.5f),
                PerceptionType.Auditory => (EmotionalState.Surprised, baseInt * 0.3f),
                PerceptionType.Tactile when p.Intensity > 0.7f => (EmotionalState.Angry, baseInt * 0.8f),
                PerceptionType.Tactile => (EmotionalState.Surprised, baseInt * 0.5f),
                PerceptionType.Proximity when entity.Relationships.TryGetValue(p.Source, out var r2) => r2.Type switch
                {
                    RelationshipType.Hostile => (EmotionalState.Anxious, baseInt * 0.6f),
                    RelationshipType.Friendly => (EmotionalState.Happy, baseInt * 0.4f),
                    _ => (EmotionalState.Curious, baseInt * 0.3f)
                },
                PerceptionType.Proximity => (EmotionalState.Curious, baseInt * 0.4f),
                PerceptionType.Semantic => (EmotionalState.Curious, baseInt * 0.5f),
                PerceptionType.Emotional when entity.Relationships.ContainsKey(p.Source) => (EmotionalState.Curious, baseInt * 0.3f),
                _ => (EmotionalState.Neutral, 0)
            };
        }

        private static void ApplyResponse(EmotionalStateData sd, EmotionalResponse r, float inertia)
        {
            float eff = r.Intensity * (1f - inertia);
            if (sd.EmotionIntensities.TryGetValue(r.Emotion, out var ci))
                sd.EmotionIntensities[r.Emotion] = Math.Min(1f, ci + eff * (1f - ci));
            else
                sd.EmotionIntensities[r.Emotion] = eff;

            if (Opposites.TryGetValue(r.Emotion, out var opp) && sd.EmotionIntensities.TryGetValue(opp, out var oi))
                sd.EmotionIntensities[opp] = Math.Max(0, oi - eff * 0.3f);
        }

        private static void ApplyDecay(EmotionalStateData sd, float dt, float rate)
        {
            foreach (var k in sd.EmotionIntensities.Keys.ToList())
            {
                sd.EmotionIntensities[k] = Math.Max(0, sd.EmotionIntensities[k] - rate * dt);
                if (sd.EmotionIntensities[k] <= 0.001f)
                    sd.EmotionIntensities.Remove(k);
            }
        }

        public EmotionalStateData? GetStateData(Guid eid) => _entityEmotions.TryGetValue(eid, out var d) ? d : null;
        public void RemoveEntity(Guid eid) => _entityEmotions.Remove(eid);
    }

    public class EmotionWheel
    {
        private readonly Dictionary<EmotionalState, float> _angles = new()
        {
            { EmotionalState.Happy, 0 }, { EmotionalState.Trusting, 45 }, { EmotionalState.Fearful, 90 },
            { EmotionalState.Surprised, 135 }, { EmotionalState.Sad, 180 }, { EmotionalState.Disgusted, 225 },
            { EmotionalState.Angry, 270 }, { EmotionalState.Anticipating, 315 }
        };

        public float CalculateDistance(EmotionalState a, EmotionalState b)
        {
            if (!_angles.TryGetValue(a, out float aa) || !_angles.TryGetValue(b, out float ab))
                return 180;
            float d = Math.Abs(aa - ab);
            return Math.Min(d, 360 - d);
        }

        public EmotionalState? Blend(EmotionalState a, EmotionalState b, float factor)
        {
            if (a == b)
                return a;
            float dist = CalculateDistance(a, b);
            if (dist <= 90)
            {
                float mid = ((_angles[a] + _angles[b]) / 2) % 360;
                return FindClosest(mid);
            }
            return null;
        }

        private EmotionalState FindClosest(float angle)
        {
            EmotionalState best = EmotionalState.Neutral;
            float minD = float.MaxValue;
            foreach (var (e, a) in _angles)
            {
                float d = Math.Min(Math.Abs(angle - a), 360 - Math.Abs(angle - a));
                if (d < minD)
                { minD = d; best = e; }
            }
            return best;
        }
    }

    public class EmpathyModel
    {
        public float EmpathyStrength { get; set; } = 0.3f;
        public float EmpathyRange { get; set; } = 30f;

        public void Update(SentientEntity entity, EmotionalStateData stateData, float deltaTime)
        {
            float drift = 0.01f * deltaTime;
            foreach (var e in stateData.EmotionIntensities.Keys.ToList())
            {
                if (e != stateData.BaselineEmotion)
                    stateData.EmotionIntensities[e] = Math.Max(0, stateData.EmotionIntensities[e] - drift);
            }
        }

        public float CalculateInfluence(SentientEntity source, SentientEntity target, float distance)
        {
            float df = Math.Max(0, 1f - distance / EmpathyRange);
            float rf = 0.1f;
            if (source.Relationships.TryGetValue(target.EntityId, out var r))
                rf = r.Strength;
            return df * rf * EmpathyStrength;
        }
    }

}
