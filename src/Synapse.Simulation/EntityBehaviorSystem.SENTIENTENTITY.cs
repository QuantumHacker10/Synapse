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

    public class SentientEntity
    {
        private readonly Dictionary<string, object> _properties = new();
        private readonly List<PerceptionEvent> _pendingPerceptions = new();
        private readonly object _lock = new();

        public SentientEntity(Guid entityId, EntityType entityType)
        {
            EntityId = entityId;
            EntityType = entityType;
            Position = Vector3.Zero;
            Velocity = Vector3.Zero;
            Orientation = Quaternion.Identity;
            CurrentState = BehaviorState.Idle;
            CurrentEmotion = EmotionalState.Neutral;
            PerceptionRadius = 50f;
            PerceptionCapabilities = new HashSet<PerceptionType> { PerceptionType.Visual, PerceptionType.Auditory, PerceptionType.Proximity };
            Relationships = new Dictionary<Guid, Relationship>();
            PersonalityTraits = new Dictionary<string, float>();
            Needs = new Dictionary<string, float>();
            Goals = new List<string>();
        }

        public Guid EntityId { get; }
        public EntityType EntityType { get; set; }
        public Vector3 Position { get; set; }
        public Vector3 Velocity { get; set; }
        public Quaternion Orientation { get; set; }
        public Vector3 Forward => Vector3.Transform(Vector3.UnitZ, Orientation);
        public BehaviorTree? BehaviorTree { get; set; }
        public IEmotionalModel? EmotionalModel { get; set; }
        public IMemorySystem? MemorySystem { get; set; }
        public Dictionary<Guid, Relationship> Relationships { get; }
        public BehaviorState CurrentState { get; set; }
        public EmotionalState CurrentEmotion { get; set; }
        public float PerceptionRadius { get; set; }
        public HashSet<PerceptionType> PerceptionCapabilities { get; }
        public float EmotionalInertia { get; set; } = 0.7f;
        public float EmotionalDecayRate { get; set; } = 0.1f;
        public string EntityGroup { get; set; } = string.Empty;
        public Dictionary<string, float> PersonalityTraits { get; }
        public Dictionary<string, float> Needs { get; }
        public List<string> Goals { get; }
        public float FieldOfView { get; set; } = 120f;
        public bool CanMove { get; set; } = true;
        public float MaxSpeed { get; set; } = 5f;
        public float Health { get; set; } = 100f;
        public float MaxHealth { get; set; } = 100f;
        public bool IsAlive => Health > 0 && CurrentState != BehaviorState.Terminated;
        public int UpdatePriority { get; set; }

        public object? this[string key] { get { lock (_lock) return _properties.TryGetValue(key, out var v) ? v : null; } set { lock (_lock) { if (value != null) _properties[key] = value; else _properties.Remove(key); } } }
        public void SetProperty(string key, object value) { lock (_lock) _properties[key] = value; }
        public T? GetProperty<T>(string key) { lock (_lock) { return _properties.TryGetValue(key, out var v) && v is T t ? t : default; } }
        public bool HasProperty(string key) { lock (_lock) return _properties.ContainsKey(key); }

        public void AddPerception(PerceptionEvent p) { lock (_lock) _pendingPerceptions.Add(p); }
        public IReadOnlyList<PerceptionEvent> GetPendingPerceptions() { lock (_lock) { var r = _pendingPerceptions.ToList().AsReadOnly(); _pendingPerceptions.Clear(); return r; } }

        public virtual void Update(EntityContext context, float deltaTime)
        {
            if (CurrentState == BehaviorState.Terminated)
                return;
            EmotionalModel?.Update(this, context.PerceptionEvents, deltaTime);
            if (EmotionalModel != null)
                CurrentEmotion = EmotionalModel.GetCurrentState(this);
            if (BehaviorTree != null && BehaviorTree.IsValid)
            {
                var s = BehaviorTree.Tick(this, context, deltaTime);
                CurrentState = s switch { TaskStatus.Running => BehaviorState.Active, _ => BehaviorState.Idle };
            }
            if (CanMove)
                Position += Velocity * deltaTime;
            DecayNeeds(deltaTime);
        }

        public virtual void HandlePerception(PerceptionEvent p)
        {
            MemorySystem?.Store(this, new MemoryEntry { Content = $"Perceived: {p.Type} from {p.Source}", Timestamp = p.Timestamp, EmotionalIntensity = p.Intensity, Importance = p.Confidence, Type = MemoryType.Episodic, Location = p.Location, AssociatedEntities = new List<Guid> { p.Source } });
            EmotionalModel?.ReactToEvent(this, p);
        }

        public void TransitionState(BehaviorState newState, string reason = "")
        {
            if (CurrentState == BehaviorState.Terminated)
                return;
            var prev = CurrentState;
            CurrentState = BehaviorState.Transitioning;
            OnStateTransition(prev, newState, reason);
            CurrentState = newState;
        }

        protected virtual void OnStateTransition(BehaviorState from, BehaviorState to, string reason) { }

        public void TakeDamage(float amount, Guid source)
        {
            Health = Math.Max(0, Health - amount);
            MemorySystem?.Store(this, new MemoryEntry { Content = $"Took {amount} damage from {source}", Timestamp = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, EmotionalIntensity = Math.Min(1f, amount / MaxHealth), Importance = 0.9f, Type = MemoryType.Episodic, AssociatedEntities = new List<Guid> { source } });
            EmotionalModel?.ReactToEvent(this, new PerceptionEvent(PerceptionType.Tactile, source, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, Math.Min(1f, amount / MaxHealth), Position, Vector3.Zero, $"Took {amount} damage", 1f));
            if (Health <= 0)
                TransitionState(BehaviorState.Terminated, "Health depleted");
        }

        public void Heal(float amount) { Health = Math.Min(MaxHealth, Health + amount); }

        public bool MoveToward(Vector3 target, float speed, float deltaTime)
        {
            if (!CanMove)
                return false;
            var dir = Vector3.Normalize(target - Position);
            var dist = Vector3.Distance(Position, target);
            var step = speed * deltaTime;
            if (step >= dist)
            { Position = target; Velocity = Vector3.Zero; return true; }
            Position += dir * step;
            Velocity = dir * speed;
            if (dir.LengthSquared() > 0.001f)
                Orientation = SentienceQuaternion.LookRotation(dir, Vector3.UnitY);
            return false;
        }

        public void MoveAwayFrom(Vector3 threat, float speed, float deltaTime)
        {
            if (!CanMove)
                return;
            var dir = Vector3.Normalize(Position - threat);
            Position += dir * speed * deltaTime;
            Velocity = dir * speed;
            if (dir.LengthSquared() > 0.001f)
                Orientation = SentienceQuaternion.LookRotation(dir, Vector3.UnitY);
        }

        public float DistanceTo(SentientEntity other) => Vector3.Distance(Position, other.Position);

        public bool CanSee(SentientEntity other, float maxDist = -1)
        {
            if (maxDist < 0)
                maxDist = PerceptionRadius;
            if (DistanceTo(other) > maxDist)
                return false;
            var toOther = Vector3.Normalize(other.Position - Position);
            var angle = Math.Acos(Math.Clamp(Vector3.Dot(Forward, toOther), -1, 1));
            return angle <= FieldOfView / 2f * (Math.PI / 180);
        }

        public string GetDominantTrait() => PersonalityTraits.Count == 0 ? "Neutral" : PersonalityTraits.OrderByDescending(kv => kv.Value).First().Key;
        public string GetMostUrgentNeed() => Needs.Count == 0 ? "None" : Needs.OrderByDescending(kv => kv.Value).First().Key;

        private void DecayNeeds(float deltaTime)
        {
            foreach (var k in Needs.Keys.ToList())
                Needs[k] = Math.Min(1f, Needs[k] + deltaTime * 0.01f);
        }

        public override string ToString() => $"Entity[{EntityId:N8}] Type={EntityType} State={CurrentState} Emotion={CurrentEmotion} Pos=({Position.X:F1},{Position.Y:F1},{Position.Z:F1}) HP={Health:F0}/{MaxHealth:F0}";
    }

}
