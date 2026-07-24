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

    public record PerceptionEvent(PerceptionType Type, Guid Source, double Timestamp, float Intensity, Vector3 Location, Vector3 Direction, string SemanticContent, float Confidence);
    public record EmotionalResponse(EmotionalState Emotion, float Intensity, float Duration, string Trigger, float DecayRate);
    public record BehaviorAction(string Type, Dictionary<string, object> Parameters, int Priority, float Duration, Guid Target);
    public record RelationshipConstraint(RelationshipType Type, float Strength, bool Bidirectional, bool Exclusive);
    public record WeatherConditions(float Temperature, float Humidity, float WindSpeed, float WindDirection, float Visibility, float AmbientLight, WeatherType Type);
    public record WorldEvent(Guid Id, string EventType, Vector3 Location, float Radius, float Intensity, double Timestamp, Guid SourceEntity, Dictionary<string, object> Metadata);

    internal static class SentienceQuaternion
    {
        public static Quaternion LookRotation(Vector3 forward, Vector3 up)
        {
            var matrix = Matrix4x4.CreateWorld(Vector3.Zero, Vector3.Normalize(forward), up);
            return Quaternion.CreateFromRotationMatrix(matrix);
        }
    }

    public class WorldStateData
    {
        public Dictionary<Guid, SentientEntity> Entities { get; set; } = new();
        public double Time { get; set; }
        public WeatherConditions Weather { get; set; } = new(20f, 0.5f, 5f, 0f, 1f, 1f, WeatherType.Clear);
        public Queue<WorldEvent> Events { get; set; } = new();
        public SpatialIndex SpatialIndex { get; set; } = new();
    }

    public class EntityContext
    {
        public SentientEntity Entity { get; set; } = null!;
        public WorldStateData WorldState { get; set; } = null!;
        public float DeltaTime { get; set; }
        public List<PerceptionEvent> PerceptionEvents { get; set; } = new();
        public IMemorySystem Memory { get; set; } = null!;
        public EmotionalState EmotionalState { get; set; }
        public Dictionary<Guid, RelationshipType> Relationships { get; set; } = new();
    }

    public class MemoryEntry
    {
        public Guid Id { get; init; } = Guid.NewGuid();
        public string Content { get; set; } = string.Empty;
        public double Timestamp { get; set; }
        public float EmotionalIntensity { get; set; }
        public float Importance { get; set; }
        public int RetrievalCount { get; set; }
        public double LastRetrieved { get; set; }
        public List<Guid> AssociatedEntities { get; set; } = new();
        public Vector3 Location { get; set; }
        public MemoryType Type { get; set; } = MemoryType.Episodic;
        public HashSet<string> Tags { get; set; } = new();
        public float ConsolidationStrength { get; set; } = 0.1f;
        public bool IsConsolidated { get; set; }
        public float DecayFactor { get; set; } = 1.0f;
        public Dictionary<string, object> Context { get; set; } = new();
    }

    public class Relationship
    {
        public Guid EntityA { get; set; }
        public Guid EntityB { get; set; }
        public RelationshipType Type { get; set; }
        public float Strength { get; set; }
        public double EstablishedTime { get; set; }
        public List<RelationshipEvent> History { get; set; } = new();
        public HashSet<string> Tags { get; set; } = new();
        public bool Bidirectional { get; set; } = true;
    }

    public record RelationshipEvent(RelationshipType PreviousType, RelationshipType NewType, float PreviousStrength, float NewStrength, double Timestamp, string Reason);

    public class BehaviorTreeBlueprint
    {
        public BehaviorNodeType NodeType { get; set; }
        public string NodeId { get; set; } = Guid.NewGuid().ToString("N")[..8];
        public string Name { get; set; } = string.Empty;
        public Dictionary<string, object> Parameters { get; set; } = new();
        public List<BehaviorTreeBlueprint> Children { get; set; } = new();
        public string Condition { get; set; } = string.Empty;
        public string ActionType { get; set; } = string.Empty;
        public float Weight { get; set; } = 1.0f;
        public float CooldownDuration { get; set; }
        public int Repetitions { get; set; } = -1;
        public string LLMPrompt { get; set; } = string.Empty;
        public string BlackboardKey { get; set; } = string.Empty;
    }

    public class MemoryQuery
    {
        public string SearchTerms { get; set; } = string.Empty;
        public MemoryType? Type { get; set; }
        public float MinEmotionalIntensity { get; set; }
        public float MinImportance { get; set; }
        public double MaxAge { get; set; } = double.MaxValue;
        public Guid? AssociatedEntity { get; set; }
        public (Vector3 Center, float Radius)? LocationFilter { get; set; }
        public List<string> Tags { get; set; } = new();
        public int MaxResults { get; set; } = 100;
        public bool SortByEmotionalIntensity { get; set; }
        public bool SortByRecency { get; set; } = true;
        public bool SortByImportance { get; set; }
    }

    public class PerceptionFilter
    {
        public HashSet<PerceptionType> AllowedTypes { get; set; } = new();
        public float MinIntensity { get; set; }
        public float MaxDistance { get; set; } = float.MaxValue;
        public HashSet<Guid> AllowedSources { get; set; } = new();
        public HashSet<Guid> ExcludedSources { get; set; } = new();
        public float MinConfidence { get; set; }
    }

    public class EmotionalStateData
    {
        public Dictionary<EmotionalState, float> EmotionIntensities { get; set; } = new();
        public EmotionalState DominantEmotion { get; set; } = EmotionalState.Neutral;
        public float Inertia { get; set; } = 0.7f;
        public float Volatility { get; set; } = 0.3f;
        public EmotionalState BaselineEmotion { get; set; } = EmotionalState.Neutral;
        public List<(EmotionalState Emotion, float Intensity, double Timestamp)> History { get; set; } = new();

        public void UpdateDominantEmotion()
        {
            if (EmotionIntensities.Count == 0)
            { DominantEmotion = BaselineEmotion; return; }
            DominantEmotion = EmotionIntensities.OrderByDescending(kv => kv.Value).First().Key;
        }

        public float GetArousalLevel() => Math.Min(1.0f, EmotionIntensities.Values.Sum());

        public float GetValence()
        {
            float v = 0;
            foreach (var (e, i) in EmotionIntensities)
                v += GetValence(e) * i;
            return Math.Clamp(v, -1, 1);
        }

        private static float GetValence(EmotionalState e) => e switch
        {
            EmotionalState.Happy => 1f,
            EmotionalState.Sad => -1f,
            EmotionalState.Angry => -0.8f,
            EmotionalState.Fearful => -0.9f,
            EmotionalState.Surprised => 0.3f,
            EmotionalState.Disgusted => -0.7f,
            EmotionalState.Trusting => 0.6f,
            EmotionalState.Anticipating => 0.4f,
            EmotionalState.Calm => 0.7f,
            EmotionalState.Excited => 0.8f,
            EmotionalState.Bored => -0.3f,
            EmotionalState.Curious => 0.5f,
            EmotionalState.Confused => -0.2f,
            EmotionalState.Determined => 0.6f,
            EmotionalState.Anxious => -0.6f,
            _ => 0f
        };
    }

}
