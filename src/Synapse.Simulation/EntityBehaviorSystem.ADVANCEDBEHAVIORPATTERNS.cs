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

    public class BehaviorPatternLibrary
    {
        private readonly Dictionary<string, Func<SentientEntity, EntityContext, IBehaviorNode>> _patternFactories = new();

        public BehaviorPatternLibrary()
        {
            RegisterDefaultPatterns();
        }

        public void RegisterPattern(string name, Func<SentientEntity, EntityContext, IBehaviorNode> factory)
        {
            _patternFactories[name] = factory;
        }

        public IBehaviorNode? CreatePattern(string name, SentientEntity entity, EntityContext context)
        {
            return _patternFactories.TryGetValue(name, out var factory) ? factory(entity, context) : null;
        }

        public IReadOnlyCollection<string> GetAvailablePatterns() => _patternFactories.Keys.ToList().AsReadOnly();

        private void RegisterDefaultPatterns()
        {
            _patternFactories["AggressiveCombat"] = (e, c) => new SelectorNode("AggressiveCombat",
                new SequenceNode("MeleePriority",
                    new ConditionNode("HasMeleeWeapon", (en, co) => en.HasProperty("MeleeWeapon")),
                    new ActionNode("MeleeAttack", (en, co, dt) => TaskStatus.Success)),
                new SequenceNode("RangedPriority",
                    new ConditionNode("HasRangedWeapon", (en, co) => en.HasProperty("RangedWeapon")),
                    new ActionNode("RangedAttack", (en, co, dt) => TaskStatus.Success)),
                new ActionNode("BasicAttack", (en, co, dt) => TaskStatus.Success));

            _patternFactories["DefensiveCombat"] = (e, c) => new SequenceNode("DefensiveCombat",
                new SelectorNode("DefensiveOptions",
                    new SequenceNode("Retreat", new ConditionNode("HealthLow", (en, co) => en.Health < en.MaxHealth * 0.3f), new ActionNode("MoveToSafety", (en, co, dt) => TaskStatus.Success)),
                    new SequenceNode("Block", new ConditionNode("EnemyNearby", (en, co) => co.PerceptionEvents.Any(p => p.Type == PerceptionType.Proximity)), new ActionNode("Block", (en, co, dt) => TaskStatus.Success)),
                    new ActionNode("CounterAttack", (en, co, dt) => TaskStatus.Success)));

            _patternFactories["WanderBehavior"] = (e, c) => new SequenceNode("Wander",
                new ActionNode("ChooseRandomDirection", (en, co, dt) => { en.SetProperty("WanderTarget", en.Position + BehaviorRandom.InsideUnitSphere() * 10f); return TaskStatus.Success; }),
                new ActionNode("MoveToWanderPoint", (en, co, dt) =>
                {
                    if (en.HasProperty("WanderTarget"))
                    {
                        var target = en.GetProperty<Vector3>("WanderTarget");
                        en.MoveToward(target, en.MaxSpeed * 0.5f, dt);
                    }
                    return TaskStatus.Success;
                }),
                new WaitNode("Pause", 2f));

            _patternFactories["SocialBehavior"] = (e, c) => new SelectorNode("Social",
                new SequenceNode("GreetAlly",
                    new ConditionNode("HasNearbyAlly", (en, co) => co.PerceptionEvents.Any(p => p.Type == PerceptionType.Semantic && co.Relationships.ContainsKey(p.Source))),
                    new ActionNode("PerformGreeting", (en, co, dt) => TaskStatus.Success)),
                new SequenceNode("Trade",
                    new ConditionNode("HasTradePartner", (en, co) => co.PerceptionEvents.Any(p => p.SemanticContent.Contains("trader"))),
                    new ActionNode("InitiateTrade", (en, co, dt) => TaskStatus.Success)),
                new ActionNode("IdleSocial", (en, co, dt) => TaskStatus.Success));

            _patternFactories["ExploreBehavior"] = (e, c) => new SequenceNode("Explore",
                new SelectorNode("ExploreOptions",
                    new SequenceNode("Investigate",
                        new ConditionNode("HasInterestingPerception", (en, co) => co.PerceptionEvents.Any(p => p.Confidence > 0.6f && p.Intensity > 0.5f)),
                        new ActionNode("MoveToInvestigate", (en, co, dt) => TaskStatus.Success)),
                    new SequenceNode("Scout",
                        new ConditionNode("AreaNotExplored", (en, co) => !en.HasProperty("AreaExplored")),
                        new ActionNode("ScoutArea", (en, co, dt) => TaskStatus.Success)),
                    new ActionNode("RandomExplore", (en, co, dt) => TaskStatus.Success)));
        }
    }

    public class BehaviorComposer
    {
        private readonly BehaviorPatternLibrary _patternLibrary;

        public BehaviorComposer(BehaviorPatternLibrary patternLibrary)
        {
            _patternLibrary = patternLibrary;
        }

        public BehaviorTree ComposeBehaviorTree(string name, List<string> patternNames, SentientEntity entity, EntityContext context)
        {
            var tree = new BehaviorTree(name);
            var rootNode = new SelectorNode("Root");

            foreach (var patternName in patternNames)
            {
                var node = _patternLibrary.CreatePattern(patternName, entity, context);
                if (node != null)
                    rootNode = new SelectorNode("Root", rootNode, node);
            }

            tree.Root = rootNode;
            return tree;
        }

        public BehaviorTree ComposeFromRules(string name, List<BehaviorRule> rules, SentientEntity entity, EntityContext context)
        {
            var tree = new BehaviorTree(name);
            var rootNode = new SelectorNode("Root");

            foreach (var rule in rules)
            {
                var condition = new ConditionNode($"Check_{rule.Name}", rule.Condition);
                var action = new ActionNode($"Do_{rule.Name}", rule.Action);
                var sequence = new SequenceNode(rule.Name, condition, action);
                rootNode = new SelectorNode("Root", rootNode, sequence);
            }

            tree.Root = rootNode;
            return tree;
        }
    }

    public class BehaviorRule
    {
        public string Name { get; set; } = string.Empty;
        public Func<SentientEntity, EntityContext, bool> Condition { get; set; } = (_, _) => true;
        public Func<SentientEntity, EntityContext, float, TaskStatus> Action { get; set; } = (_, _, _) => TaskStatus.Success;
        public float Priority { get; set; }
    }

}
