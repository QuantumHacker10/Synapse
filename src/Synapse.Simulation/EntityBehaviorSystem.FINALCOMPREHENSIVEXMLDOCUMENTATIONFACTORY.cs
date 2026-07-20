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

    /// <summary>
    /// Factory for creating pre-configured sentient entities with common behavior patterns.
    /// Provides convenience methods for spawning entities with appropriate behaviors.
    /// </summary>
    public class SentientEntityFactory
    {
        private readonly SentienceManager _manager;

        public SentientEntityFactory(SentienceManager manager) { _manager = manager; }

        /// <summary>
        /// Spawns a sentient simulation agent (perceives, decides, adapts) — not a scripted game NPC.
        /// </summary>
        public SentientEntity CreateAgent(Vector3 position, string behaviorType, string groupName = "")
        {
            var entity = _manager.CreateEntity(EntityType.Sentient, position);
            entity.CanMove = true;
            entity.MaxSpeed = 5f;
            entity.Health = 100f;
            entity.MaxHealth = 100f;
            entity.PerceptionRadius = 50f;
            entity.FieldOfView = 120f;

            var tree = new BehaviorTree($"Agent_{behaviorType}_{entity.EntityId:N}");
            var root = new SelectorNode("Agent_Behavior");

            switch (behaviorType.ToLower())
            {
                case "guard":
                    root = CreateGuardBehavior(entity);
                    break;
                case "patrol":
                    root = CreatePatrolBehavior(entity);
                    break;
                case "merchant":
                    root = CreateMerchantBehavior(entity);
                    break;
                case "aggressive":
                    root = CreateAggressiveBehavior(entity);
                    root = new SelectorNode("Agent_Behavior", root);
                    entity.PersonalityTraits["Aggression"] = 0.8f;
                    entity.PersonalityTraits["Courage"] = 0.7f;
                    break;
                case "passive":
                    root = new SelectorNode("Agent_Behavior",
                        new ActionNode("Idle", (_, _, _) => TaskStatus.Success));
                    entity.PersonalityTraits["Aggression"] = 0.1f;
                    entity.PersonalityTraits["Agreeableness"] = 0.9f;
                    break;
                case "explorer":
                    root = CreateExplorerBehavior(entity);
                    entity.PersonalityTraits["Openness"] = 0.9f;
                    entity.PersonalityTraits["Courage"] = 0.6f;
                    break;
                case "coward":
                    root = CreateCowardBehavior(entity);
                    entity.PersonalityTraits["Courage"] = 0.1f;
                    entity.PersonalityTraits["Neuroticism"] = 0.8f;
                    break;
                default:
                    root = new SelectorNode("Agent_Behavior",
                        new ActionNode("DefaultBehavior", (_, _, _) => TaskStatus.Success));
                    break;
            }

            tree.Root = root;
            entity.BehaviorTree = tree;

            if (!string.IsNullOrEmpty(groupName))
                _manager.AddGroupMember(groupName, entity.EntityId);

            return entity;
        }

        [Obsolete("Use CreateAgent — Synapse models sentient simulation agents, not game NPCs.")]
        public SentientEntity CreateNPC(Vector3 position, string behaviorType, string groupName = "")
            => CreateAgent(position, behaviorType, groupName);

        /// <summary>
        /// Spawns a simulation observer (operator presence), not a game player avatar.
        /// </summary>
        public SentientEntity CreateObserver(Vector3 position)
        {
            var entity = _manager.CreateEntity(EntityType.Observer, position);
            entity.CanMove = true;
            entity.MaxSpeed = 8f;
            entity.Health = 150f;
            entity.MaxHealth = 150f;
            entity.PerceptionRadius = 80f;
            entity.FieldOfView = 150f;
            entity.UpdatePriority = 100;
            return entity;
        }

        [Obsolete("Use CreateObserver — Synapse is a simulation tool, not a game engine.")]
        public SentientEntity CreatePlayer(Vector3 position) => CreateObserver(position);

        public SentientEntity CreateTrigger(Vector3 position, float radius, Action<SentientEntity, SentientEntity> onEnter)
        {
            var entity = _manager.CreateEntity(EntityType.Trigger, position);
            entity.CanMove = false;
            entity.PerceptionRadius = radius;
            return entity;
        }

        public SentientEntity CreateSensor(Vector3 position, float radius)
        {
            var entity = _manager.CreateEntity(EntityType.Sensor, position);
            entity.CanMove = false;
            entity.PerceptionRadius = radius;
            entity.PerceptionCapabilities.Add(PerceptionType.Visual);
            entity.PerceptionCapabilities.Add(PerceptionType.Auditory);
            entity.PerceptionCapabilities.Add(PerceptionType.Semantic);
            return entity;
        }

        public List<SentientEntity> CreateGroup(string groupName, EntityType type, Vector3 center, int count, float spacing = 3f)
        {
            var entities = new List<SentientEntity>();
            for (int i = 0; i < count; i++)
            {
                float angle = (2 * MathF.PI * i) / count;
                var pos = center + new Vector3(MathF.Cos(angle) * spacing, 0, MathF.Sin(angle) * spacing);
                var entity = CreateAgent(pos, "patrol", groupName);
                entities.Add(entity);
            }
            if (entities.Count > 0)
                _manager.GroupBehavior.SetLeader(groupName, entities[0].EntityId);
            return entities;
        }

        private SelectorNode CreateGuardBehavior(SentientEntity entity) => new("Guard_Root",
            new SequenceNode("Patrol",
                new ActionNode("MoveToGuardPoint", (e, c, dt) => { e.MoveToward(e.Position + e.Forward * 2f, e.MaxSpeed * 0.3f, dt); return TaskStatus.Success; }),
                new WaitNode("Wait", 3f)),
            new SelectorNode("RespondToThreat",
                new SequenceNode("DetectThreat",
                    new ConditionNode("ThreatDetected", (_, c) => c.PerceptionEvents.Any(p => p.Type == PerceptionType.Proximity && p.Intensity > 0.7f)),
                    new ActionNode("EngageThreat", (e, c, dt) => { e.CurrentState = BehaviorState.Active; return TaskStatus.Success; })),
                new ActionNode("IdleGuard", (_, _, _) => TaskStatus.Success)));

        private SelectorNode CreatePatrolBehavior(SentientEntity entity) => new("Patrol_Root",
            new SequenceNode("Patrol_Circuit",
                new ActionNode("MoveToNextPoint", (e, c, dt) =>
                {
                    var points = e.GetProperty<List<Vector3>>("PatrolPoints");
                    int idx = e.GetProperty<int>("PatrolIndex");
                    if (points != null && points.Count > 0)
                    {
                        e.MoveToward(points[idx % points.Count], e.MaxSpeed * 0.5f, dt);
                        if (Vector3.Distance(e.Position, points[idx % points.Count]) < 1.5f)
                            e.SetProperty("PatrolIndex", (idx + 1) % points.Count);
                    }
                    return TaskStatus.Success;
                }),
                new WaitNode("Pause", 1.5f)));

        private SelectorNode CreateMerchantBehavior(SentientEntity entity) => new("Merchant_Root",
            new SequenceNode("OfferTrade",
                new ConditionNode("CustomerNearby", (_, c) => c.PerceptionEvents.Any(p => p.Type == PerceptionType.Proximity)),
                new ActionNode("InitiateTrade", (_, _, _) => TaskStatus.Success)),
            new ActionNode("IdleMerchant", (_, _, _) => TaskStatus.Success));

        private SelectorNode CreateAggressiveBehavior(SentientEntity entity) => new("Aggressive_Root",
            new SequenceNode("HuntTarget",
                new ConditionNode("EnemyVisible", (_, c) => c.Relationships.Values.Any(t => t == RelationshipType.Hostile)),
                new ActionNode("ChaseEnemy", (e, c, dt) => { e.CurrentState = BehaviorState.Active; return TaskStatus.Success; }),
                new ActionNode("Attack", (_, _, _) => TaskStatus.Success)),
            new ActionNode("Provoke", (_, _, _) => TaskStatus.Success));

        private SelectorNode CreateExplorerBehavior(SentientEntity entity) => new("Explorer_Root",
            new SequenceNode("Investigate",
                new ConditionNode("InterestingPerception", (_, c) => c.PerceptionEvents.Any(p => p.Confidence > 0.5f)),
                new ActionNode("MoveToInvestigate", (e, c, dt) => { e.CurrentState = BehaviorState.Active; return TaskStatus.Success; })),
            new SequenceNode("Explore",
                new ActionNode("ChooseDirection", (e, _, _) => { e.SetProperty("ExploreTarget", e.Position + BehaviorRandom.InsideUnitSphere() * 20f); return TaskStatus.Success; }),
                new ActionNode("MoveToExplore", (e, c, dt) =>
                {
                    if (e.HasProperty("ExploreTarget"))
                    {
                        var target = e.GetProperty<Vector3>("ExploreTarget");
                        e.MoveToward(target, e.MaxSpeed * 0.6f, dt);
                    }
                    return TaskStatus.Success;
                }),
                new WaitNode("Observe", 2f)));

        private SelectorNode CreateCowardBehavior(SentientEntity entity) => new("Coward_Root",
            new SequenceNode("FleeFromDanger",
                new ConditionNode("DangerNearby", (_, c) => c.PerceptionEvents.Any(p => p.Intensity > 0.6f)),
                new ActionNode("Flee", (e, c, dt) => { e.MoveAwayFrom(e.Position + e.Forward * 10f, e.MaxSpeed, dt); return TaskStatus.Success; })),
            new ActionNode("Hide", (_, _, _) => TaskStatus.Success));
    }

}
