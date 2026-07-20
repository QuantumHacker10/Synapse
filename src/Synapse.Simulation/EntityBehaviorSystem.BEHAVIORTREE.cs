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

    public class BehaviorTree
    {
        private IBehaviorNode? _root;
        private readonly Blackboard _blackboard;
        private readonly Dictionary<string, IBehaviorNode> _nodeIndex;
        private readonly List<string> _validationErrors;
        private bool _isValid;
        private int _tickCount;
        private readonly Stopwatch _tickTimer;

        public BehaviorTree(string name) { Name = name; _blackboard = new Blackboard(); _nodeIndex = new(); _validationErrors = new(); _tickTimer = new(); }

        public string Name { get; set; }
        public IBehaviorNode? Root { get => _root; set { _root = value; RebuildIndex(); Validate(); } }
        public Blackboard Blackboard => _blackboard;
        public bool IsValid => _isValid;
        public IReadOnlyList<string> ValidationErrors => _validationErrors.AsReadOnly();
        public int TickCount => _tickCount;
        public double LastTickTimeMs { get; private set; }

        public TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            if (_root == null || !_isValid)
                return TaskStatus.Failure;
            _tickTimer.Restart();
            var status = _root.Tick(entity, context, deltaTime);
            _tickTimer.Stop();
            LastTickTimeMs = _tickTimer.Elapsed.TotalMilliseconds;
            _tickCount++;
            return status;
        }

        public bool Validate()
        {
            _validationErrors.Clear();
            if (_root == null)
            { _validationErrors.Add("No root node."); _isValid = false; return false; }
            _isValid = _root.Validate(_validationErrors);
            return _isValid;
        }

        public void Reset() { _root?.Reset(); _tickCount = 0; }
        public BehaviorTree Clone() => new($"{Name}_Clone") { Root = _root?.Clone() };
        public IBehaviorNode? FindNode(string id) => _nodeIndex.TryGetValue(id, out var n) ? n : null;
        public IReadOnlyCollection<IBehaviorNode> GetAllNodes() => _nodeIndex.Values.ToList().AsReadOnly();

        public string Serialize()
        {
            if (_root == null)
                return "{}";
            var bp = ToBlueprint(_root);
            return JsonSerializer.Serialize(bp, new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        }

        public static BehaviorTree? Deserialize(string json)
        {
            try
            {
                var bp = JsonSerializer.Deserialize<BehaviorTreeBlueprint>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                if (bp == null)
                    return null;
                return new BehaviorTree("Deserialized") { Root = FromBlueprint(bp) };
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("BehaviorTree", "Failed to deserialize behavior tree JSON.", ex);
                return null;
            }
        }

        public static BehaviorTree FromLLMPrompt(string name, string prompt)
        {
            var tree = new BehaviorTree(name);
            var root = new SequenceNode("Root");
            var words = prompt.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (words.Contains("patrol") || words.Contains("wander"))
                root = CreatePatrol(root);
            if (words.Contains("attack") || words.Contains("fight"))
                root = CreateCombat(root);
            if (words.Contains("flee") || words.Contains("escape"))
                root = CreateFlee(root);
            if (words.Contains("guard") || words.Contains("protect"))
                root = CreateGuard(root);
            if (words.Contains("follow") || words.Contains("chase"))
                root = CreateFollow(root);
            tree.Root = root;
            return tree;
        }

        public void Optimize()
        {
            if (_root == null)
                return;
            _root = OptimizeNode(_root);
            RebuildIndex();
            Validate();
        }

        private IBehaviorNode OptimizeNode(IBehaviorNode node)
        {
            if ((node is SucceederNode or FailerNode) && node.GetChildren().Count == 1)
                return node.GetChildren()[0];
            return node;
        }

        private void RebuildIndex()
        {
            _nodeIndex.Clear();
            if (_root != null)
                IndexNode(_root);
        }

        private void IndexNode(IBehaviorNode node) { _nodeIndex[node.Id] = node; foreach (var c in node.GetChildren()) IndexNode(c); }

        private static BehaviorTreeBlueprint ToBlueprint(IBehaviorNode node)
        {
            var bp = new BehaviorTreeBlueprint { NodeType = node.NodeType, NodeId = node.Id, Name = node.Name };
            foreach (var c in node.GetChildren())
                bp.Children.Add(ToBlueprint(c));
            return bp;
        }

        private static IBehaviorNode FromBlueprint(BehaviorTreeBlueprint bp)
        {
            return bp.NodeType switch
            {
                BehaviorNodeType.Sequence => new SequenceNode(bp.Name, bp.Children.Select(FromBlueprint).ToArray()),
                BehaviorNodeType.Selector => new SelectorNode(bp.Name, bp.Children.Select(FromBlueprint).ToArray()),
                BehaviorNodeType.Parallel => new ParallelNode(bp.Name, 1, bp.Children.Count, bp.Children.Select(FromBlueprint).ToArray()),
                BehaviorNodeType.Condition => new ConditionNode(bp.Name, (_, _) => true),
                BehaviorNodeType.Action => new ActionNode(bp.Name, (_, _, _) => TaskStatus.Success),
                BehaviorNodeType.Inverter => new InverterNode(bp.Name, bp.Children.Count > 0 ? FromBlueprint(bp.Children[0]) : new WaitNode("E", 0)),
                BehaviorNodeType.Repeater => new RepeaterNode(bp.Name, bp.Children.Count > 0 ? FromBlueprint(bp.Children[0]) : new WaitNode("E", 0), bp.Repetitions),
                BehaviorNodeType.Cooldown => new CooldownNode(bp.Name, bp.Children.Count > 0 ? FromBlueprint(bp.Children[0]) : new WaitNode("E", 0), bp.CooldownDuration),
                BehaviorNodeType.RandomSelector => new RandomSelectorNode(bp.Name, bp.Children.Select(c => (FromBlueprint(c), c.Weight)).ToArray()),
                BehaviorNodeType.WeightedRandom => new WeightedRandomNode(bp.Name, 1, bp.Children.Select(c => (FromBlueprint(c), c.Weight)).ToArray()),
                BehaviorNodeType.Loop => new LoopNode(bp.Name, bp.Children.Count > 0 ? FromBlueprint(bp.Children[0]) : new WaitNode("E", 0), bp.Repetitions),
                BehaviorNodeType.Guard => new GuardNode(bp.Name, bp.Children.Count > 0 ? FromBlueprint(bp.Children[0]) : new WaitNode("E", 0), (_, _) => true),
                BehaviorNodeType.Succeeder => new SucceederNode(bp.Name, bp.Children.Count > 0 ? FromBlueprint(bp.Children[0]) : new WaitNode("E", 0)),
                BehaviorNodeType.Failer => new FailerNode(bp.Name, bp.Children.Count > 0 ? FromBlueprint(bp.Children[0]) : new WaitNode("E", 0)),
                BehaviorNodeType.Wait => new WaitNode(bp.Name, bp.CooldownDuration),
                BehaviorNodeType.LLMQuery => new LLMQueryNode(bp.Name, bp.LLMPrompt, BehaviorLlmContext.DefaultHandle),
                _ => new WaitNode(bp.Name, 0)
            };
        }

        private static SequenceNode CreatePatrol(SequenceNode _) => new("Patrol",
            new SelectorNode("FindPoint",
                new ActionNode("MoveToPatrol", (_, _, _) => TaskStatus.Success),
                new ActionNode("ChoosePoint", (_, _, _) => TaskStatus.Success)),
            new WaitNode("Wait", 2f));

        private static SequenceNode CreateCombat(SequenceNode _) => new("Combat",
            new ConditionNode("HasTarget", (_, c) => c.Relationships.Any(r => r.Value == RelationshipType.Hostile)),
            new SelectorNode("Options",
                new SequenceNode("Melee", new ConditionNode("MeleeRange", (_, _) => true), new ActionNode("MeleeAttack", (_, _, _) => TaskStatus.Success)),
                new SequenceNode("Ranged", new ConditionNode("RangedRange", (_, _) => true), new ActionNode("RangedAttack", (_, _, _) => TaskStatus.Success))));

        private static SequenceNode CreateFlee(SequenceNode _) => new("Flee",
            new ConditionNode("ThreatNearby", (_, c) => c.Relationships.Any(r => r.Value == RelationshipType.Hostile)),
            new ActionNode("FindSafeDir", (_, _, _) => TaskStatus.Success),
            new ActionNode("MoveToSafety", (_, _, _) => TaskStatus.Success));

        private static SequenceNode CreateGuard(SequenceNode _) => new("Guard",
            new ActionNode("MoveToGuardPoint", (_, _, _) => TaskStatus.Success),
            new SelectorNode("GuardDuty",
                new SequenceNode("DetectThreat", new ConditionNode("ThreatDetected", (_, c) => c.Relationships.Any(r => r.Value == RelationshipType.Hostile)), new ActionNode("EngageThreat", (_, _, _) => TaskStatus.Success)),
                new ActionNode("IdleGuard", (_, _, _) => TaskStatus.Success)));

        private static SequenceNode CreateFollow(SequenceNode _) => new("Follow",
            new ConditionNode("HasTarget", (_, c) => c.Relationships.Values.Any(t => t == RelationshipType.Parent || t == RelationshipType.Master)),
            new ActionNode("MoveToTarget", (_, _, _) => TaskStatus.Success),
            new WaitNode("MaintainDist", 0.5f));
    }

}
