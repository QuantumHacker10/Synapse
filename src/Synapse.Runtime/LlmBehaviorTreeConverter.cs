using System;
using System.Collections.Generic;
using System.Linq;
using GDNN.Llm;
using GDNN.Sentience;

namespace Synapse.Runtime
{
    /// <summary>Converts LLM-extracted flat nodes into a simulation behavior tree blueprint.</summary>
    public static class LlmBehaviorTreeConverter
    {
        public static BehaviorTreeBlueprint ToBlueprint(IReadOnlyList<BehaviorTreeNode> nodes)
        {
            var children = nodes.Select(MapNode).ToList();
            if (children.Count == 0)
            {
                children.Add(new BehaviorTreeBlueprint
                {
                    NodeType = BehaviorNodeType.Wait,
                    Name = "Idle",
                    CooldownDuration = 1f
                });
            }

            return new BehaviorTreeBlueprint
            {
                NodeType = BehaviorNodeType.Sequence,
                Name = "LLM_Root",
                Children = children
            };
        }

        private static BehaviorTreeBlueprint MapNode(BehaviorTreeNode node)
        {
            var kind = node.NodeType?.ToLowerInvariant() ?? "action";
            return kind switch
            {
                "sequence" => new BehaviorTreeBlueprint { NodeType = BehaviorNodeType.Sequence, Name = node.Name, Children = new List<BehaviorTreeBlueprint>() },
                "selector" => new BehaviorTreeBlueprint { NodeType = BehaviorNodeType.Selector, Name = node.Name, Children = new List<BehaviorTreeBlueprint>() },
                "condition" => new BehaviorTreeBlueprint { NodeType = BehaviorNodeType.Condition, Name = node.Name, Condition = node.Description ?? node.Name },
                "llmquery" or "llm" or "query" => new BehaviorTreeBlueprint { NodeType = BehaviorNodeType.LLMQuery, Name = node.Name, LLMPrompt = node.Description ?? node.Name },
                "wait" => new BehaviorTreeBlueprint { NodeType = BehaviorNodeType.Wait, Name = node.Name, CooldownDuration = 1f },
                _ => new BehaviorTreeBlueprint { NodeType = BehaviorNodeType.Action, Name = node.Name, ActionType = node.Description ?? node.Name }
            };
        }
    }
}
