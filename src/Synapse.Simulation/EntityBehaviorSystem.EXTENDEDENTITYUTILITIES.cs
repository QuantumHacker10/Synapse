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

    public static class SentientEntityExtensions
    {
        public static float GetHealthPercentage(this SentientEntity e) => e.MaxHealth > 0 ? e.Health / e.MaxHealth : 0;
        public static bool IsLowHealth(this SentientEntity e, float threshold = 0.3f) => e.GetHealthPercentage() < threshold;
        public static bool IsHighHealth(this SentientEntity e, float threshold = 0.7f) => e.GetHealthPercentage() > threshold;
        public static bool IsEmotionallyStable(this SentientEntity e) => e.CurrentEmotion == EmotionalState.Calm || e.CurrentEmotion == EmotionalState.Neutral;
        public static bool IsEmotionallyUnstable(this SentientEntity e) => e.CurrentEmotion == EmotionalState.Angry || e.CurrentEmotion == EmotionalState.Fearful || e.CurrentEmotion == EmotionalState.Anxious;
        public static bool IsHostileTo(this SentientEntity e, Guid otherId) => e.Relationships.TryGetValue(otherId, out var r) && r.Type == RelationshipType.Hostile;
        public static bool IsFriendlyTo(this SentientEntity e, Guid otherId) => e.Relationships.TryGetValue(otherId, out var r) && r.Type == RelationshipType.Friendly;
        public static bool IsWithinRange(this SentientEntity e, Vector3 position, float range) => Vector3.Distance(e.Position, position) <= range;
        public static bool HasEnemiesNearby(this SentientEntity e, float range = 30f) => e.Relationships.Any(r => r.Value.Type == RelationshipType.Hostile && r.Value.Strength > 0.3f);
        public static int GetNearbyEnemyCount(this SentientEntity e, Dictionary<Guid, SentientEntity> allEntities, float range = 30f)
        {
            return allEntities.Values.Count(o => o.EntityId != e.EntityId && e.IsHostileTo(o.EntityId) && e.DistanceTo(o) <= range);
        }
        public static SentientEntity? GetNearestEnemy(this SentientEntity e, Dictionary<Guid, SentientEntity> allEntities)
        {
            return allEntities.Values.Where(o => o.EntityId != e.EntityId && e.IsHostileTo(o.EntityId)).OrderBy(o => e.DistanceTo(o)).FirstOrDefault();
        }
        public static SentientEntity? GetNearestAlly(this SentientEntity e, Dictionary<Guid, SentientEntity> allEntities)
        {
            return allEntities.Values.Where(o => o.EntityId != e.EntityId && e.IsFriendlyTo(o.EntityId)).OrderBy(o => e.DistanceTo(o)).FirstOrDefault();
        }
        public static float CalculateThreatScore(this SentientEntity e, Dictionary<Guid, SentientEntity> allEntities)
        {
            float score = 0;
            foreach (var other in allEntities.Values)
            {
                if (other.EntityId == e.EntityId || !e.IsHostileTo(other.EntityId))
                    continue;
                float dist = e.DistanceTo(other);
                float distFactor = Math.Max(0, 1f - dist / 50f);
                score += distFactor * (other.Health / other.MaxHealth);
            }
            return Math.Clamp(score, 0, 1);
        }
        public static Vector3 GetFleeDirection(this SentientEntity e, Dictionary<Guid, SentientEntity> allEntities)
        {
            var enemies = allEntities.Values.Where(o => o.EntityId != e.EntityId && e.IsHostileTo(o.EntityId)).ToList();
            if (enemies.Count == 0)
                return e.Forward;
            var avgEnemyPos = enemies.Aggregate(Vector3.Zero, (sum, en) => sum + en.Position) / enemies.Count;
            return Vector3.Normalize(e.Position - avgEnemyPos);
        }
        public static void LookAt(this SentientEntity e, Vector3 target)
        {
            var dir = Vector3.Normalize(target - e.Position);
            if (dir.LengthSquared() > 0.001f)
                e.Orientation = SentienceQuaternion.LookRotation(dir, Vector3.UnitY);
        }
        public static void LookAtEntity(this SentientEntity e, SentientEntity target) => e.LookAt(target.Position);
        public static float GetAlignmentScore(this SentientEntity e, List<SentientEntity> group)
        {
            if (group.Count < 2)
                return 1f;
            var avgForward = group.Where(o => o.EntityId != e.EntityId).Aggregate(Vector3.Zero, (sum, o) => sum + o.Forward) / Math.Max(1, group.Count - 1);
            return Math.Clamp(Vector3.Dot(e.Forward, Vector3.Normalize(avgForward)), 0, 1);
        }
        public static float GetCohesionScore(this SentientEntity e, List<SentientEntity> group)
        {
            if (group.Count < 2)
                return 1f;
            var center = group.Where(o => o.EntityId != e.EntityId).Aggregate(Vector3.Zero, (sum, o) => sum + o.Position) / Math.Max(1, group.Count - 1);
            float dist = Vector3.Distance(e.Position, center);
            return Math.Clamp(1f - dist / 10f, 0, 1);
        }
        public static float GetSeparationScore(this SentientEntity e, List<SentientEntity> group, float desiredSeparation = 2f)
        {
            if (group.Count < 2)
                return 1f;
            float minDist = group.Where(o => o.EntityId != e.EntityId).Min(o => e.DistanceTo(o));
            return Math.Clamp(minDist / desiredSeparation, 0, 1);
        }
        public static Vector3 CalculateFlockingForce(this SentientEntity e, List<SentientEntity> group, float separationWeight = 1.5f, float alignmentWeight = 1f, float cohesionWeight = 1f)
        {
            var separation = Vector3.Zero;
            var alignment = Vector3.Zero;
            var cohesion = Vector3.Zero;
            int count = 0;
            foreach (var other in group)
            {
                if (other.EntityId == e.EntityId)
                    continue;
                float dist = e.DistanceTo(other);
                if (dist < 2f && dist > 0.01f)
                    separation += Vector3.Normalize(e.Position - other.Position) / dist;
                if (dist < 5f)
                { alignment += other.Velocity; cohesion += other.Position; count++; }
            }
            if (count > 0)
            { alignment = Vector3.Normalize(alignment / count - e.Velocity); cohesion = Vector3.Normalize(cohesion / count - e.Position); }
            return separation * separationWeight + alignment * alignmentWeight + cohesion * cohesionWeight;
        }
    }

    public static class TaskStatusExtensions
    {
        public static bool IsComplete(this TaskStatus s) => s == TaskStatus.Success || s == TaskStatus.Failure;
        public static bool IsRunning(this TaskStatus s) => s == TaskStatus.Running;
        public static bool IsPending(this TaskStatus s) => s == TaskStatus.Pending;
        public static TaskStatus Invert(this TaskStatus s) => s switch { TaskStatus.Success => TaskStatus.Failure, TaskStatus.Failure => TaskStatus.Success, _ => s };
        public static string ToDisplayString(this TaskStatus s) => s switch { TaskStatus.Success => "SUCCESS", TaskStatus.Failure => "FAILURE", TaskStatus.Running => "RUNNING", TaskStatus.Pending => "PENDING", _ => "UNKNOWN" };
    }

    public static class EmotionalStateExtensions
    {
        public static float GetValence(this EmotionalState s) => s switch
        {
            EmotionalState.Happy => 1f,
            EmotionalState.Excited => 0.8f,
            EmotionalState.Calm => 0.7f,
            EmotionalState.Trusting => 0.6f,
            EmotionalState.Determined => 0.6f,
            EmotionalState.Curious => 0.5f,
            EmotionalState.Surprised => 0.3f,
            EmotionalState.Bored => -0.3f,
            EmotionalState.Confused => -0.2f,
            EmotionalState.Anxious => -0.6f,
            EmotionalState.Disgusted => -0.7f,
            EmotionalState.Angry => -0.8f,
            EmotionalState.Fearful => -0.9f,
            EmotionalState.Sad => -1f,
            _ => 0f
        };
        public static float GetArousal(this EmotionalState s) => s switch
        {
            EmotionalState.Excited => 0.9f,
            EmotionalState.Angry => 0.85f,
            EmotionalState.Fearful => 0.8f,
            EmotionalState.Surprised => 0.75f,
            EmotionalState.Anxious => 0.7f,
            EmotionalState.Curious => 0.6f,
            EmotionalState.Happy => 0.5f,
            EmotionalState.Determined => 0.5f,
            EmotionalState.Disgusted => 0.4f,
            EmotionalState.Bored => 0.1f,
            EmotionalState.Calm => 0.15f,
            EmotionalState.Sad => 0.2f,
            EmotionalState.Trusting => 0.25f,
            EmotionalState.Confused => 0.35f,
            _ => 0.3f
        };
        public static bool IsPositive(this EmotionalState s) => s.GetValence() > 0.1f;
        public static bool IsNegative(this EmotionalState s) => s.GetValence() < -0.1f;
        public static bool IsHighArousal(this EmotionalState s) => s.GetArousal() > 0.6f;
        public static bool IsLowArousal(this EmotionalState s) => s.GetArousal() < 0.3f;
        public static EmotionalState GetOpposite(this EmotionalState s) => s switch
        {
            EmotionalState.Happy => EmotionalState.Sad,
            EmotionalState.Sad => EmotionalState.Happy,
            EmotionalState.Angry => EmotionalState.Calm,
            EmotionalState.Calm => EmotionalState.Angry,
            EmotionalState.Fearful => EmotionalState.Determined,
            EmotionalState.Determined => EmotionalState.Fearful,
            EmotionalState.Surprised => EmotionalState.Anticipating,
            EmotionalState.Anticipating => EmotionalState.Surprised,
            EmotionalState.Disgusted => EmotionalState.Trusting,
            EmotionalState.Trusting => EmotionalState.Disgusted,
            EmotionalState.Excited => EmotionalState.Bored,
            EmotionalState.Bored => EmotionalState.Excited,
            EmotionalState.Curious => EmotionalState.Confused,
            EmotionalState.Confused => EmotionalState.Curious,
            EmotionalState.Anxious => EmotionalState.Calm,
            _ => EmotionalState.Neutral
        };
        public static string ToEmoji(this EmotionalState s) => s switch
        {
            EmotionalState.Happy => "[HAPPY]",
            EmotionalState.Sad => "[SAD]",
            EmotionalState.Angry => "[ANGRY]",
            EmotionalState.Fearful => "[FEAR]",
            EmotionalState.Surprised => "[SURPRISE]",
            EmotionalState.Disgusted => "[DISGUST]",
            EmotionalState.Trusting => "[TRUST]",
            EmotionalState.Anticipating => "[ANTICIPATE]",
            EmotionalState.Calm => "[CALM]",
            EmotionalState.Excited => "[EXCITED]",
            EmotionalState.Bored => "[BORED]",
            EmotionalState.Curious => "[CURIOUS]",
            EmotionalState.Confused => "[CONFUSED]",
            EmotionalState.Determined => "[DETERMINED]",
            EmotionalState.Anxious => "[ANXIOUS]",
            _ => "[NEUTRAL]"
        };
    }

    public static class BehaviorStateExtensions
    {
        public static bool IsActive(this BehaviorState s) => s == BehaviorState.Active;
        public static bool IsIdle(this BehaviorState s) => s == BehaviorState.Idle;
        public static bool IsDormant(this BehaviorState s) => s == BehaviorState.Dormant;
        public static bool ShouldUpdate(this BehaviorState s) => s == BehaviorState.Active || s == BehaviorState.Transitioning || s == BehaviorState.Idle;
        public static float GetUpdateFrequency(this BehaviorState s) => s switch { BehaviorState.Active => 1f, BehaviorState.Idle => 0.5f, BehaviorState.Transitioning => 0.8f, BehaviorState.Dormant => 0.1f, _ => 0f };
        public static string ToDisplayString(this BehaviorState s) => s switch { BehaviorState.Idle => "IDLE", BehaviorState.Active => "ACTIVE", BehaviorState.Dormant => "DORMANT", BehaviorState.Transitioning => "TRANSITIONING", BehaviorState.Terminated => "TERMINATED", _ => "UNKNOWN" };
    }

    public static class RelationshipTypeExtensions
    {
        public static float GetBaseTrust(this RelationshipType t) => t switch { RelationshipType.Friendly => 0.7f, RelationshipType.Parent => 0.9f, RelationshipType.Child => 0.8f, RelationshipType.Partner => 0.85f, RelationshipType.Sibling => 0.6f, RelationshipType.Master => 0.5f, RelationshipType.Servant => 0.4f, RelationshipType.Neutral => 0.3f, RelationshipType.Hostile => 0.05f, _ => 0.3f };
        public static float GetBaseAffinity(this RelationshipType t) => t switch { RelationshipType.Friendly => 0.8f, RelationshipType.Parent => 0.95f, RelationshipType.Child => 0.9f, RelationshipType.Partner => 0.9f, RelationshipType.Sibling => 0.7f, RelationshipType.Master => 0.4f, RelationshipType.Servant => 0.3f, RelationshipType.Neutral => 0.5f, RelationshipType.Hostile => 0.1f, _ => 0.5f };
        public static bool IsPositive(this RelationshipType t) => t == RelationshipType.Friendly || t == RelationshipType.Parent || t == RelationshipType.Child || t == RelationshipType.Partner || t == RelationshipType.Sibling;
        public static bool IsNegative(this RelationshipType t) => t == RelationshipType.Hostile;
        public static bool IsAuthority(this RelationshipType t) => t == RelationshipType.Master || t == RelationshipType.Parent;
    }

}
