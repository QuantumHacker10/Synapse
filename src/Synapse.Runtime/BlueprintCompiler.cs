using System;
using System.Collections.Generic;
using System.Linq;
using GDNN.Sentience;

namespace Synapse.Runtime
{
    /// <summary>Compiles a <see cref="BlueprintDocument"/> graph into a simulation <see cref="BehaviorTreeBlueprint"/>.</summary>
    public static class BlueprintCompiler
    {
        public static BehaviorTreeBlueprint Compile(BlueprintDocument doc)
        {
            var (ok, msg) = doc.Validate();
            if (!ok) throw new InvalidOperationException(msg);

            var entry = doc.Nodes.First(n => n.Kind == BlueprintNodeKind.Entry);
            var visited = new HashSet<Guid>();
            var root = BuildExecChain(doc, entry.Id, visited);
            return root ?? new BehaviorTreeBlueprint
            {
                NodeType = BehaviorNodeType.Sequence,
                Name = doc.Name,
                Children = { new BehaviorTreeBlueprint { NodeType = BehaviorNodeType.Wait, Name = "Empty", CooldownDuration = 0.1f } }
            };
        }

        private static BehaviorTreeBlueprint? BuildExecChain(
            BlueprintDocument doc,
            Guid nodeId,
            HashSet<Guid> visited)
        {
            if (!visited.Add(nodeId)) return null;
            var node = doc.Nodes.FirstOrDefault(n => n.Id == nodeId);
            if (node == null) return null;

            if (node.Kind == BlueprintNodeKind.Exit)
                return null;

            var self = MapNode(node);
            var nextIds = doc.Edges
                .Where(e => e.FromNodeId == nodeId)
                .OrderBy(e => e.FromPin)
                .Select(e => e.ToNodeId)
                .ToList();

            var children = new List<BehaviorTreeBlueprint>();
            if (self != null) children.Add(self);

            foreach (var nextId in nextIds)
            {
                var child = BuildExecChain(doc, nextId, visited);
                if (child != null) children.Add(child);
            }

            if (children.Count == 0) return self;
            if (children.Count == 1) return children[0];

            return new BehaviorTreeBlueprint
            {
                NodeType = BehaviorNodeType.Sequence,
                Name = $"Seq_{node.Title}",
                Children = children
            };
        }

        private static BehaviorTreeBlueprint? MapNode(BlueprintNode node)
        {
            return node.Kind switch
            {
                BlueprintNodeKind.Entry => null,
                BlueprintNodeKind.Exit => null,
                BlueprintNodeKind.Sequence => new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.Sequence,
                    Name = node.Title,
                    Children = new List<BehaviorTreeBlueprint>()
                },
                BlueprintNodeKind.Selector => new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.Selector,
                    Name = node.Title,
                    Children = new List<BehaviorTreeBlueprint>()
                },
                BlueprintNodeKind.Condition => new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.Condition,
                    Name = node.Title,
                    Condition = node.Payload ?? "true"
                },
                BlueprintNodeKind.Action => new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.Action,
                    Name = node.Title,
                    ActionType = node.Payload ?? "idle"
                },
                BlueprintNodeKind.Wait => new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.Wait,
                    Name = node.Title,
                    CooldownDuration = float.TryParse(node.Payload, out var d) ? d : 1f
                },
                BlueprintNodeKind.LlmQuery => new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.LLMQuery,
                    Name = node.Title,
                    LLMPrompt = node.Payload ?? node.Title
                },
                BlueprintNodeKind.SpawnAgent => new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.Action,
                    Name = node.Title,
                    ActionType = $"spawn:{node.Payload ?? "patrol"}"
                },
                BlueprintNodeKind.LawApply => new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.Action,
                    Name = node.Title,
                    ActionType = $"law:{node.Payload ?? "heat_equation"}"
                },
                BlueprintNodeKind.EvolveStep => new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.Action,
                    Name = node.Title,
                    ActionType = "evolve"
                },
                _ => new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.Wait,
                    Name = node.Title,
                    CooldownDuration = 0.5f
                }
            };
        }
    }
}
