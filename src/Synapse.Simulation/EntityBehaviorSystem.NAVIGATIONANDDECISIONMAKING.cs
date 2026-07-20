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

    public class NavigationAgent
    {
        private readonly SpatialReasoningSystem _spatialReasoning;
        private readonly Dictionary<Guid, NavigationState> _agentStates = new();
        private readonly Dictionary<Guid, List<Vector3>> _waypointSets = new();
        private readonly Dictionary<Guid, float> _pathUpdateTimers = new();
        private readonly float _pathUpdateInterval = 0.5f;

        public NavigationAgent(SpatialReasoningSystem spatialReasoning)
        {
            _spatialReasoning = spatialReasoning;
        }

        public float AcceptanceRadius { get; set; } = 1.0f;
        public float AvoidanceRadius { get; set; } = 2.5f;
        public float MaxAcceleration { get; set; } = 8.0f;
        public float MaxDeceleration { get; set; } = 12.0f;
        public float PathRecalculationThreshold { get; set; } = 5.0f;

        public void SetDestination(Guid entityId, Vector3 destination)
        {
            var state = GetOrCreate(entityId);
            state.Destination = destination;
            state.HasValidPath = false;
            state.PathDirty = true;
            _pathUpdateTimers[entityId] = 0;
        }

        public void SetPatrolRoute(Guid entityId, List<Vector3> waypoints)
        {
            _waypointSets[entityId] = waypoints;
            var state = GetOrCreate(entityId);
            state.CurrentWaypointIndex = 0;
            state.Mode = NavigationMode.Patrol;
            if (waypoints.Count > 0)
                state.Destination = waypoints[0];
            state.PathDirty = true;
        }

        public Vector3 CalculateSteering(SentientEntity entity, float deltaTime)
        {
            var state = GetOrCreate(entity.EntityId);

            _pathUpdateTimers.TryGetValue(entity.EntityId, out float timer);
            timer += deltaTime;
            _pathUpdateTimers[entity.EntityId] = timer;

            if (state.PathDirty || timer >= _pathUpdateInterval)
            {
                RecalculatePath(entity, state);
                state.PathDirty = false;
                _pathUpdateTimers[entity.EntityId] = 0;
            }

            if (state.CurrentPath == null || state.CurrentPath.Count == 0)
                return Vector3.Zero;

            var waypoint = state.CurrentPath[state.CurrentWaypointIndex];
            var toWaypoint = waypoint - entity.Position;
            float distance = toWaypoint.Length();

            if (distance <= AcceptanceRadius)
            {
                state.CurrentWaypointIndex++;
                if (state.CurrentWaypointIndex >= state.CurrentPath.Count)
                {
                    if (state.Mode == NavigationMode.Patrol)
                    {
                        AdvancePatrol(entity, state);
                        state.CurrentWaypointIndex = 0;
                        state.PathDirty = true;
                    }
                    else
                    {
                        state.HasReachedDestination = true;
                        return Vector3.Zero;
                    }
                }
                if (state.CurrentWaypointIndex < state.CurrentPath.Count)
                    waypoint = state.CurrentPath[state.CurrentWaypointIndex];
                else
                    return Vector3.Zero;
                toWaypoint = waypoint - entity.Position;
                distance = toWaypoint.Length();
            }

            var desiredDirection = Vector3.Normalize(toWaypoint);
            float desiredSpeed = Math.Min(entity.MaxSpeed, distance / AcceptanceRadius * entity.MaxSpeed);
            var desiredVelocity = desiredDirection * desiredSpeed;

            var avoidance = CalculateObstacleAvoidance(entity, state);
            desiredVelocity += avoidance * entity.MaxSpeed;

            var steering = desiredVelocity - entity.Velocity;
            if (steering.Length() > MaxAcceleration)
                steering = Vector3.Normalize(steering) * MaxAcceleration;

            return steering;
        }

        public bool HasReachedDestination(Guid entityId)
        {
            return _agentStates.TryGetValue(entityId, out var state) && state.HasReachedDestination;
        }

        public void ClearDestination(Guid entityId)
        {
            if (_agentStates.TryGetValue(entityId, out var state))
            {
                state.Destination = null;
                state.CurrentPath = null;
                state.HasReachedDestination = false;
            }
        }

        private void RecalculatePath(SentientEntity entity, NavigationState state)
        {
            if (!state.Destination.HasValue)
                return;
            if (state.CurrentPath != null && state.CurrentWaypointIndex < state.CurrentPath.Count)
            {
                float distFromPath = Vector3.Distance(entity.Position, state.CurrentPath[state.CurrentWaypointIndex]);
                if (distFromPath < PathRecalculationThreshold && !state.PathDirty)
                    return;
            }
            var start = entity.Position;
            var goal = state.Destination.Value;
            var obstacles = new List<Vector3>();
            state.CurrentPath = _spatialReasoning.FindPath(start, goal, obstacles);
            state.CurrentWaypointIndex = 0;
            state.HasReachedDestination = false;
        }

        private void AdvancePatrol(SentientEntity entity, NavigationState state)
        {
            if (_waypointSets.TryGetValue(entity.EntityId, out var waypoints) && waypoints.Count > 0)
            {
                state.CurrentWaypointIndex = 0;
                state.Destination = waypoints[(state.CurrentWaypointIndex + 1) % waypoints.Count];
            }
        }

        private Vector3 CalculateObstacleAvoidance(SentientEntity entity, NavigationState state)
        {
            var avoidance = Vector3.Zero;
            if (state.NearbyObstacles != null)
            {
                foreach (var obstacle in state.NearbyObstacles)
                {
                    float dist = Vector3.Distance(entity.Position, obstacle);
                    if (dist < AvoidanceRadius && dist > 0.01f)
                    {
                        var away = Vector3.Normalize(entity.Position - obstacle);
                        float strength = (AvoidanceRadius - dist) / AvoidanceRadius;
                        avoidance += away * strength * strength;
                    }
                }
            }
            return avoidance.LengthSquared() > 0.001f ? Vector3.Normalize(avoidance) : Vector3.Zero;
        }

        private NavigationState GetOrCreate(Guid entityId)
        {
            if (!_agentStates.TryGetValue(entityId, out var state))
            { state = new NavigationState(); _agentStates[entityId] = state; }
            return state;
        }
    }

    public enum NavigationMode { Direct, Patrol, Follow, Flee, Wander }

    public class NavigationState
    {
        public Vector3? Destination { get; set; }
        public List<Vector3>? CurrentPath { get; set; }
        public int CurrentWaypointIndex { get; set; }
        public bool HasReachedDestination { get; set; }
        public bool HasValidPath { get; set; }
        public bool PathDirty { get; set; }
        public NavigationMode Mode { get; set; } = NavigationMode.Direct;
        public List<Vector3>? NearbyObstacles { get; set; }
        public Vector3 LastRecordedPosition { get; set; }
        public float StuckTimer { get; set; }
        public float StuckThreshold { get; set; } = 2.0f;
    }

    public class DecisionMakingEngine
    {
        private readonly Dictionary<Guid, UtilityAI> _utilityAIs = new();
        private readonly Dictionary<Guid, GoalStack> _goalStacks = new();

        public float DecisionInterval { get; set; } = 0.5f;

        public void RegisterEntity(Guid entityId)
        {
            _utilityAIs[entityId] = new UtilityAI();
            _goalStacks[entityId] = new GoalStack();
        }

        public string? MakeDecision(SentientEntity entity, EntityContext context)
        {
            if (!_utilityAIs.TryGetValue(entity.EntityId, out var ai))
                return null;
            if (!_goalStacks.TryGetValue(entity.EntityId, out var goals))
                return null;

            var considerations = GenerateConsiderations(entity, context);
            var scored = new List<(string Action, float Score)>();

            foreach (var consideration in considerations)
            {
                float score = 1f;
                foreach (var criterion in consideration.Criteria)
                {
                    score *= criterion.CalculateScore(entity, context);
                }
                scored.Add((consideration.ActionName, score * consideration.BaseScore));
            }

            scored.Sort((a, b) => b.Score.CompareTo(a.Score));

            if (scored.Count > 0 && scored[0].Score > 0.1f)
            {
                return scored[0].Action;
            }
            return null;
        }

        public GoalStack? GetGoalStack(Guid entityId)
        {
            return _goalStacks.TryGetValue(entityId, out var gs) ? gs : null;
        }

        private List<UtilityAIConsideration> GenerateConsiderations(SentientEntity entity, EntityContext context)
        {
            var considerations = new List<UtilityAIConsideration>();

            considerations.Add(new UtilityAIConsideration("Attack",
                new List<UtilityAICriterion>
                {
                    new("HasEnemy", (e, c) => c.Relationships.Values.Any(t => t == RelationshipType.Hostile) ? 1f : 0f),
                    new("HealthHigh", (e, c) => e.Health / e.MaxHealth),
                    new("InMeleeRange", (e, c) => c.PerceptionEvents.Any(p => p.Type == PerceptionType.Proximity && p.Intensity > 0.8f) ? 1f : 0.2f),
                    new("Aggressive", (e, c) => e.PersonalityTraits.GetValueOrDefault("Aggression", 0.5f))
                }, 0.9f));

            considerations.Add(new UtilityAIConsideration("Flee",
                new List<UtilityAICriterion>
                {
                    new("HasEnemy", (e, c) => c.Relationships.Values.Any(t => t == RelationshipType.Hostile) ? 1f : 0f),
                    new("HealthLow", (e, c) => 1f - e.Health / e.MaxHealth),
                    new("Fearful", (e, c) => e.CurrentEmotion == EmotionalState.Fearful || e.CurrentEmotion == EmotionalState.Anxious ? 1f : 0.2f),
                    new("Cowardly", (e, c) => 1f - e.PersonalityTraits.GetValueOrDefault("Courage", 0.5f))
                }, 0.7f));

            considerations.Add(new UtilityAIConsideration("Patrol",
                new List<UtilityAICriterion>
                {
                    new("NoImmediateThreat", (e, c) => c.Relationships.Values.Any(t => t == RelationshipType.Hostile) ? 0.1f : 1f),
                    new("HasPatrolRoute", (e, c) => e.HasProperty("PatrolRoute") ? 1f : 0.3f),
                    new("Bored", (e, c) => e.CurrentEmotion == EmotionalState.Bored ? 0.8f : 0.4f)
                }, 0.5f));

            considerations.Add(new UtilityAIConsideration("Heal",
                new List<UtilityAICriterion>
                {
                    new("HealthLow", (e, c) => 1f - e.Health / e.MaxHealth),
                    new("HasHealing", (e, c) => e.HasProperty("HealingItem") ? 1f : 0.1f)
                }, 0.6f));

            considerations.Add(new UtilityAIConsideration("FollowAlly",
                new List<UtilityAICriterion>
                {
                    new("HasAlly", (e, c) => c.Relationships.Values.Any(t => t == RelationshipType.Friendly || t == RelationshipType.Parent) ? 1f : 0f),
                    new("Lonely", (e, c) => e.Needs.GetValueOrDefault("Social", 0f)),
                    new("Following", (e, c) => e.CurrentEmotion == EmotionalState.Trusting ? 0.8f : 0.3f)
                }, 0.55f));

            considerations.Add(new UtilityAIConsideration("Idle",
                new List<UtilityAICriterion>
                {
                    new("NoThreats", (e, c) => c.Relationships.Values.Any(t => t == RelationshipType.Hostile) ? 0.1f : 1f),
                    new("Healthy", (e, c) => e.Health / e.MaxHealth),
                    new("Calm", (e, c) => e.CurrentEmotion == EmotionalState.Calm ? 1f : 0.5f)
                }, 0.3f));

            return considerations;
        }
    }

    public class UtilityAI
    {
        public Dictionary<string, float> ActionScores { get; set; } = new();
        public string? LastDecision { get; set; }
        public double LastDecisionTime { get; set; }
        public int DecisionCount { get; set; }
    }

    public class UtilityAIConsideration
    {
        public string ActionName { get; set; }
        public List<UtilityAICriterion> Criteria { get; set; }
        public float BaseScore { get; set; }

        public UtilityAIConsideration(string actionName, List<UtilityAICriterion> criteria, float baseScore)
        {
            ActionName = actionName;
            Criteria = criteria;
            BaseScore = baseScore;
        }
    }

    public class UtilityAICriterion
    {
        public string Name { get; set; }
        private readonly Func<SentientEntity, EntityContext, float> _scoreFunc;

        public UtilityAICriterion(string name, Func<SentientEntity, EntityContext, float> scoreFunc)
        {
            Name = name;
            _scoreFunc = scoreFunc;
        }

        public float CalculateScore(SentientEntity entity, EntityContext context) => _scoreFunc(entity, context);
    }

    public class GoalStack
    {
        private readonly Stack<Goal> _goals = new();
        private readonly List<Goal> _activeGoals = new();

        public void PushGoal(Goal goal) { _goals.Push(goal); }
        public Goal? PeekGoal() => _goals.Count > 0 ? _goals.Peek() : null;
        public Goal? PopGoal() => _goals.Count > 0 ? _goals.Pop() : null;
        public int GoalCount => _goals.Count;
        public IReadOnlyList<Goal> ActiveGoals => _activeGoals.AsReadOnly();

        public void UpdateGoals(float deltaTime)
        {
            foreach (var goal in _activeGoals.ToList())
            {
                goal.ElapsedTime += deltaTime;
                if (goal.IsCompleted || goal.ElapsedTime >= goal.MaxDuration)
                {
                    _activeGoals.Remove(goal);
                }
            }
        }
    }

    public class Goal
    {
        public string Name { get; set; } = string.Empty;
        public float Priority { get; set; }
        public float MaxDuration { get; set; } = float.MaxValue;
        public float ElapsedTime { get; set; }
        public bool IsCompleted { get; set; }
        public Dictionary<string, object> Parameters { get; set; } = new();
    }

}
