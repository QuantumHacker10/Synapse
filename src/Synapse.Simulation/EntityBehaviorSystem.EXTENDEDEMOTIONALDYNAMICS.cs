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

    public class EmotionalDynamicsEngine
    {
        private readonly Dictionary<Guid, EmotionalDynamicsState> _dynamicsStates = new();
        private readonly EmotionTransitionMatrix _transitionMatrix;

        public EmotionalDynamicsEngine()
        {
            _transitionMatrix = new EmotionTransitionMatrix();
        }

        public float BlendingRate { get; set; } = 0.5f;
        public float MoodSwingThreshold { get; set; } = 0.7f;
        public float EmotionalResonanceFactor { get; set; } = 0.3f;
        public float MoodStabilityBase { get; set; } = 0.6f;

        public void UpdateDynamics(Guid entityId, EmotionalStateData stateData, float deltaTime)
        {
            var dynamics = GetOrCreate(entityId);

            UpdateMood(dynamics, stateData, deltaTime);
            UpdateEmotionalMomentum(dynamics, stateData, deltaTime);
            DetectMoodSwings(dynamics, stateData);
            UpdateEmotionalMemory(dynamics, stateData, deltaTime);
            ProcessEmotionalCascade(dynamics, stateData);
            UpdateBaselineShift(dynamics, deltaTime);
        }

        private void UpdateMood(EmotionalDynamicsState dynamics, EmotionalStateData stateData, float deltaTime)
        {
            var currentMood = dynamics.CurrentMood;
            var targetMood = CalculateMoodFromEmotions(stateData);

            float blendSpeed = BlendingRate * (1f - MoodStabilityBase);
            if (dynamics.PersonalityStability > 0.7f)
                blendSpeed *= 0.5f;

            dynamics.CurrentMood = LerpEmotionalState(currentMood, targetMood, blendSpeed * deltaTime);
            dynamics.MoodIntensity = Math.Clamp(
                dynamics.MoodIntensity + (stateData.GetArousalLevel() - dynamics.MoodIntensity) * 0.1f * deltaTime,
                0, 1);
        }

        private void UpdateEmotionalMomentum(EmotionalDynamicsState dynamics, EmotionalStateData stateData, float deltaTime)
        {
            var dominant = stateData.DominantEmotion;
            if (dominant == dynamics.LastDominantEmotion)
            {
                dynamics.EmotionalMomentum = Math.Min(1f, dynamics.EmotionalMomentum + deltaTime * 0.2f);
            }
            else
            {
                dynamics.EmotionalMomentum = Math.Max(0, dynamics.EmotionalMomentum - deltaTime * 0.3f);
                dynamics.LastDominantEmotion = dominant;
                dynamics.MomentumChangeCount++;
            }
        }

        private void DetectMoodSwings(EmotionalDynamicsState dynamics, EmotionalStateData stateData)
        {
            if (dynamics.RecentDominantEmotions.Count >= 5)
            {
                var recent = dynamics.RecentDominantEmotions.TakeLast(5).ToList();
                int changes = 0;
                for (int i = 1; i < recent.Count; i++)
                {
                    if (recent[i] != recent[i - 1])
                        changes++;
                }
                if (changes >= 3)
                {
                    dynamics.IsMoodSwinging = true;
                    dynamics.MoodSwingIntensity = Math.Min(1f, changes / 3f);
                }
                else
                {
                    dynamics.IsMoodSwinging = false;
                    dynamics.MoodSwingIntensity = Math.Max(0, dynamics.MoodSwingIntensity - 0.1f);
                }
            }

            if (dynamics.IsMoodSwinging)
            {
                dynamics.TimeSinceLastSwing = 0;
            }
            else
            {
                dynamics.TimeSinceLastSwing++;
            }
        }

        private void UpdateEmotionalMemory(EmotionalDynamicsState dynamics, EmotionalStateData stateData, float deltaTime)
        {
            foreach (var entry in dynamics.EmotionalMemory.ToList())
            {
                entry.Strength = Math.Max(0, entry.Strength - deltaTime * 0.01f);
                if (entry.Strength <= 0.01f)
                    dynamics.EmotionalMemory.Remove(entry);
            }

            if (stateData.EmotionIntensities.Count > 0)
            {
                var strongest = stateData.EmotionIntensities.OrderByDescending(kv => kv.Value).First();
                if (strongest.Value > 0.3f)
                {
                    var existing = dynamics.EmotionalMemory.FirstOrDefault(m => m.Emotion == strongest.Key);
                    if (existing != null)
                    {
                        existing.Strength = Math.Min(1f, existing.Strength + strongest.Value * deltaTime * 0.1f);
                        existing.LastExperienceTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
                    }
                    else
                    {
                        dynamics.EmotionalMemory.Add(new EmotionalMemoryEntry
                        {
                            Emotion = strongest.Key,
                            Strength = strongest.Value * 0.5f,
                            FirstExperienceTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond,
                            LastExperienceTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond
                        });
                    }
                }
            }
        }

        private void ProcessEmotionalCascade(EmotionalDynamicsState dynamics, EmotionalStateData stateData)
        {
            if (dynamics.EmotionalMomentum > 0.8f)
            {
                var currentEmotions = stateData.EmotionIntensities.ToList();
                foreach (var (emotion, intensity) in currentEmotions)
                {
                    var related = _transitionMatrix.GetRelatedEmotions(emotion);
                    foreach (var (relatedEmotion, probability) in related)
                    {
                        if (intensity > 0.5f && Random.Shared.NextSingle() < probability * 0.1f)
                        {
                            float cascadeStrength = intensity * 0.2f;
                            if (stateData.EmotionIntensities.TryGetValue(relatedEmotion, out var existing))
                                stateData.EmotionIntensities[relatedEmotion] = Math.Min(1f, existing + cascadeStrength);
                            else
                                stateData.EmotionIntensities[relatedEmotion] = cascadeStrength;
                        }
                    }
                }
            }
        }

        private void UpdateBaselineShift(EmotionalDynamicsState dynamics, float deltaTime)
        {
            if (dynamics.MoodIntensity > 0.7f && dynamics.EmotionalMomentum > 0.6f)
            {
                float shiftRate = 0.001f * deltaTime * dynamics.MoodIntensity;
                dynamics.BaselineShiftAccumulator += shiftRate;
                if (dynamics.BaselineShiftAccumulator > 1f)
                {
                    dynamics.BaselineShiftAccumulator = 0;
                    if (_transitionMatrix.GetRelatedEmotions(dynamics.CurrentMood).Count > 0)
                    {
                        var candidates = _transitionMatrix.GetRelatedEmotions(dynamics.CurrentMood);
                        var weighted = candidates.OrderByDescending(c => c.Probability).First();
                        dynamics.ShiftedBaseline = weighted.Emotion;
                    }
                }
            }
        }

        private static EmotionalState CalculateMoodFromEmotions(EmotionalStateData stateData)
        {
            if (stateData.EmotionIntensities.Count == 0)
                return EmotionalState.Neutral;
            return stateData.EmotionIntensities.OrderByDescending(kv => kv.Value).First().Key;
        }

        private static EmotionalState LerpEmotionalState(EmotionalState from, EmotionalState to, float t)
        {
            if (from == to)
                return from;
            if (t >= 1f)
                return to;
            if (t <= 0f)
                return from;
            return Random.Shared.NextSingle() < t ? to : from;
        }

        public EmotionalDynamicsState? GetDynamics(Guid entityId)
        {
            return _dynamicsStates.TryGetValue(entityId, out var d) ? d : null;
        }

        private EmotionalDynamicsState GetOrCreate(Guid entityId)
        {
            if (!_dynamicsStates.TryGetValue(entityId, out var d))
            {
                d = new EmotionalDynamicsState
                {
                    PersonalityStability = MoodStabilityBase + (Random.Shared.NextSingle() - 0.5f) * 0.3f
                };
                _dynamicsStates[entityId] = d;
            }
            return d;
        }
    }

    public class EmotionalDynamicsState
    {
        public EmotionalState CurrentMood { get; set; } = EmotionalState.Neutral;
        public float MoodIntensity { get; set; } = 0.3f;
        public float EmotionalMomentum { get; set; }
        public EmotionalState LastDominantEmotion { get; set; } = EmotionalState.Neutral;
        public int MomentumChangeCount { get; set; }
        public bool IsMoodSwinging { get; set; }
        public float MoodSwingIntensity { get; set; }
        public int TimeSinceLastSwing { get; set; }
        public List<EmotionalMemoryEntry> EmotionalMemory { get; set; } = new();
        public List<EmotionalState> RecentDominantEmotions { get; set; } = new();
        public float PersonalityStability { get; set; } = 0.6f;
        public EmotionalState ShiftedBaseline { get; set; } = EmotionalState.Neutral;
        public float BaselineShiftAccumulator { get; set; }
    }

    public class EmotionalMemoryEntry
    {
        public EmotionalState Emotion { get; set; }
        public float Strength { get; set; }
        public double FirstExperienceTime { get; set; }
        public double LastExperienceTime { get; set; }
        public int ExperienceCount { get; set; } = 1;
    }

    public class EmotionTransitionMatrix
    {
        private readonly Dictionary<EmotionalState, List<(EmotionalState Emotion, float Probability)>> _transitions;

        public EmotionTransitionMatrix()
        {
            _transitions = new Dictionary<EmotionalState, List<(EmotionalState, float)>>
            {
                { EmotionalState.Happy, new List<(EmotionalState, float)>
                    { (EmotionalState.Excited, 0.3f), (EmotionalState.Calm, 0.4f), (EmotionalState.Trusting, 0.2f), (EmotionalState.Surprised, 0.1f) } },
                { EmotionalState.Sad, new List<(EmotionalState, float)>
                    { (EmotionalState.Anxious, 0.2f), (EmotionalState.Angry, 0.15f), (EmotionalState.Bored, 0.25f), (EmotionalState.Fearful, 0.1f) } },
                { EmotionalState.Angry, new List<(EmotionalState, float)>
                    { (EmotionalState.Disgusted, 0.25f), (EmotionalState.Determined, 0.2f), (EmotionalState.Fearful, 0.15f), (EmotionalState.Surprised, 0.1f) } },
                { EmotionalState.Fearful, new List<(EmotionalState, float)>
                    { (EmotionalState.Anxious, 0.3f), (EmotionalState.Surprised, 0.2f), (EmotionalState.Angry, 0.15f), (EmotionalState.Sad, 0.1f) } },
                { EmotionalState.Surprised, new List<(EmotionalState, float)>
                    { (EmotionalState.Curious, 0.3f), (EmotionalState.Fearful, 0.2f), (EmotionalState.Happy, 0.15f), (EmotionalState.Angry, 0.1f) } },
                { EmotionalState.Disgusted, new List<(EmotionalState, float)>
                    { (EmotionalState.Angry, 0.3f), (EmotionalState.Sad, 0.2f), (EmotionalState.Fearful, 0.1f), (EmotionalState.Anxious, 0.15f) } },
                { EmotionalState.Trusting, new List<(EmotionalState, float)>
                    { (EmotionalState.Happy, 0.3f), (EmotionalState.Calm, 0.25f), (EmotionalState.Anticipating, 0.2f), (EmotionalState.Curious, 0.1f) } },
                { EmotionalState.Anticipating, new List<(EmotionalState, float)>
                    { (EmotionalState.Curious, 0.25f), (EmotionalState.Excited, 0.3f), (EmotionalState.Anxious, 0.15f), (EmotionalState.Determined, 0.2f) } },
                { EmotionalState.Calm, new List<(EmotionalState, float)>
                    { (EmotionalState.Trusting, 0.2f), (EmotionalState.Happy, 0.25f), (EmotionalState.Bored, 0.15f), (EmotionalState.Anticipating, 0.1f) } },
                { EmotionalState.Excited, new List<(EmotionalState, float)>
                    { (EmotionalState.Happy, 0.35f), (EmotionalState.Surprised, 0.2f), (EmotionalState.Determined, 0.15f), (EmotionalState.Anxious, 0.1f) } },
                { EmotionalState.Bored, new List<(EmotionalState, float)>
                    { (EmotionalState.Curious, 0.2f), (EmotionalState.Sad, 0.15f), (EmotionalState.Angry, 0.1f), (EmotionalState.Calm, 0.25f) } },
                { EmotionalState.Curious, new List<(EmotionalState, float)>
                    { (EmotionalState.Excited, 0.2f), (EmotionalState.Anticipating, 0.25f), (EmotionalState.Happy, 0.15f), (EmotionalState.Surprised, 0.2f) } },
                { EmotionalState.Confused, new List<(EmotionalState, float)>
                    { (EmotionalState.Curious, 0.25f), (EmotionalState.Anxious, 0.2f), (EmotionalState.Frustrated, 0.15f), (EmotionalState.Surprised, 0.1f) } },
                { EmotionalState.Determined, new List<(EmotionalState, float)>
                    { (EmotionalState.Excited, 0.2f), (EmotionalState.Angry, 0.15f), (EmotionalState.Happy, 0.1f), (EmotionalState.Anticipating, 0.25f) } },
                { EmotionalState.Anxious, new List<(EmotionalState, float)>
                    { (EmotionalState.Fearful, 0.25f), (EmotionalState.Confused, 0.2f), (EmotionalState.Angry, 0.15f), (EmotionalState.Sad, 0.1f) } }
            };
        }

        public List<(EmotionalState Emotion, float Probability)> GetRelatedEmotions(EmotionalState emotion)
        {
            return _transitions.TryGetValue(emotion, out var related) ? related : new List<(EmotionalState, float)>();
        }

        public float GetTransitionProbability(EmotionalState from, EmotionalState to)
        {
            if (_transitions.TryGetValue(from, out var related))
            {
                var match = related.FirstOrDefault(r => r.Emotion == to);
                return match.Emotion == to ? match.Probability : 0;
            }
            return 0;
        }
    }

}
