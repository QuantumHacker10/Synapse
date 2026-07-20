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

    public class BehaviorCompiler
    {
        private readonly BehaviorDebugger? _debugger;

        public BehaviorCompiler(BehaviorDebugger? debugger = null) { _debugger = debugger; }

        public BehaviorTree CompileFromLLMPrompt(string name, string prompt)
        {
            var tree = new BehaviorTree(name);
            var lines = prompt.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            var rootNode = new SelectorNode("Root");

            foreach (var line in lines)
            {
                var trimmed = line.Trim();
                if (trimmed.StartsWith("#"))
                    continue;

                var node = ParseLLMLine(trimmed);
                if (node != null)
                    rootNode.GetChildren().ToList().Add(node);
            }

            if (rootNode.GetChildren().Count == 0)
                rootNode = new SelectorNode("Root", new WaitNode("Idle", 1f));

            tree.Root = rootNode;
            return tree;
        }

        private IBehaviorNode? ParseLLMLine(string line)
        {
            var lower = line.ToLower();
            if (lower.Contains("if ") || lower.Contains("when ") || lower.Contains("check "))
            {
                string condition = ExtractCondition(lower);
                return new ConditionNode($"Check_{condition[..Math.Min(20, condition.Length)]}", (e, c) => EvaluateCondition(condition, e, c));
            }
            if (lower.Contains("do ") || lower.Contains("perform ") || lower.Contains("execute "))
            {
                string action = ExtractAction(lower);
                return new ActionNode($"Action_{action[..Math.Min(20, action.Length)]}", (e, c, dt) => ExecuteAction(action, e, c));
            }
            if (lower.Contains("wait ") || lower.Contains("pause ") || lower.Contains("sleep "))
            {
                float duration = ExtractFloat(line, 1f);
                return new WaitNode($"Wait_{duration:F1}s", duration);
            }
            if (lower.Contains("repeat ") || lower.Contains("loop "))
            {
                int count = ExtractInt(line, 3);
                var child = new ActionNode("RepeatedAction", (_, _, _) => TaskStatus.Success);
                return new LoopNode($"Loop_{count}", child, count);
            }
            if (lower.Contains("randomly ") || lower.Contains("random "))
            {
                return new RandomSelectorNode("RandomChoice",
                    (new ActionNode("OptionA", (_, _, _) => TaskStatus.Success), 1f),
                    (new ActionNode("OptionB", (_, _, _) => TaskStatus.Success), 1f));
            }
            if (lower.Contains("invert ") || lower.Contains("not "))
            {
                var child = new ConditionNode("InvertedCond", (_, _) => true);
                return new InverterNode("Invert", child);
            }
            if (lower.Contains("cooldown ") || lower.Contains("throttle "))
            {
                float cd = ExtractFloat(line, 2f);
                var child = new ActionNode("CooldownAction", (_, _, _) => TaskStatus.Success);
                return new CooldownNode($"Cooldown_{cd:F1}s", child, cd);
            }
            return null;
        }

        private string ExtractCondition(string line)
        {
            var keywords = new[] { "if ", "when ", "check " };
            foreach (var kw in keywords)
            {
                int idx = line.IndexOf(kw, StringComparison.Ordinal);
                if (idx >= 0)
                    return line[(idx + kw.Length)..].Trim();
            }
            return line;
        }

        private string ExtractAction(string line)
        {
            var keywords = new[] { "do ", "perform ", "execute " };
            foreach (var kw in keywords)
            {
                int idx = line.IndexOf(kw, StringComparison.Ordinal);
                if (idx >= 0)
                    return line[(idx + kw.Length)..].Trim();
            }
            return line;
        }

        private static float ExtractFloat(string text, float defaultVal)
        {
            var words = text.Split(' ');
            foreach (var w in words)
            {
                string cleaned = new(w.Where(c => char.IsDigit(c) || c == '.' || c == '-').ToArray());
                if (float.TryParse(cleaned, out float val))
                    return val;
            }
            return defaultVal;
        }

        private static int ExtractInt(string text, int defaultVal)
        {
            var words = text.Split(' ');
            foreach (var w in words)
            {
                string cleaned = new(w.Where(char.IsDigit).ToArray());
                if (int.TryParse(cleaned, out int val))
                    return val;
            }
            return defaultVal;
        }

        private bool EvaluateCondition(string condition, SentientEntity entity, EntityContext context)
        {
            var lower = condition.ToLower();
            if (lower.Contains("health"))
                return entity.Health > entity.MaxHealth * 0.5f;
            if (lower.Contains("enemy") || lower.Contains("hostile"))
                return context.Relationships.Values.Any(t => t == RelationshipType.Hostile);
            if (lower.Contains("friend") || lower.Contains("ally"))
                return context.Relationships.Values.Any(t => t == RelationshipType.Friendly);
            if (lower.Contains("fear"))
                return entity.CurrentEmotion == EmotionalState.Fearful;
            if (lower.Contains("angry"))
                return entity.CurrentEmotion == EmotionalState.Angry;
            if (lower.Contains("alive"))
                return entity.IsAlive;
            if (lower.Contains("move"))
                return entity.CanMove;
            return true;
        }

        private TaskStatus ExecuteAction(string action, SentientEntity entity, EntityContext context)
        {
            var lower = action.ToLower();
            if (lower.Contains("attack") || lower.Contains("fight"))
                return TaskStatus.Success;
            if (lower.Contains("flee") || lower.Contains("escape"))
                return TaskStatus.Success;
            if (lower.Contains("patrol") || lower.Contains("wander"))
                return TaskStatus.Success;
            if (lower.Contains("heal") || lower.Contains("rest"))
                return TaskStatus.Success;
            if (lower.Contains("follow") || lower.Contains("chase"))
                return TaskStatus.Success;
            return TaskStatus.Success;
        }

        public BehaviorTree CompileFromBlueprint(string name, BehaviorTreeBlueprint blueprint)
        {
            var tree = new BehaviorTree(name) { Root = CompileNode(blueprint) };
            return tree;
        }

        private IBehaviorNode CompileNode(BehaviorTreeBlueprint bp)
        {
            return bp.NodeType switch
            {
                BehaviorNodeType.Sequence => new SequenceNode(bp.Name, bp.Children.Select(CompileNode).ToArray()),
                BehaviorNodeType.Selector => new SelectorNode(bp.Name, bp.Children.Select(CompileNode).ToArray()),
                BehaviorNodeType.Parallel => new ParallelNode(bp.Name, 1, bp.Children.Count, bp.Children.Select(CompileNode).ToArray()),
                BehaviorNodeType.Condition => new ConditionNode(bp.Name, (e, c) => EvaluateCondition(bp.Condition, e, c)),
                BehaviorNodeType.Action => new ActionNode(bp.Name, (e, c, dt) => ExecuteAction(bp.ActionType, e, c)),
                BehaviorNodeType.Inverter => new InverterNode(bp.Name, bp.Children.Count > 0 ? CompileNode(bp.Children[0]) : new WaitNode("Empty", 0)),
                BehaviorNodeType.Repeater => new RepeaterNode(bp.Name, bp.Children.Count > 0 ? CompileNode(bp.Children[0]) : new WaitNode("Empty", 0), bp.Repetitions),
                BehaviorNodeType.Cooldown => new CooldownNode(bp.Name, bp.Children.Count > 0 ? CompileNode(bp.Children[0]) : new WaitNode("Empty", 0), bp.CooldownDuration),
                BehaviorNodeType.RandomSelector => new RandomSelectorNode(bp.Name, bp.Children.Select(c => (CompileNode(c), c.Weight)).ToArray()),
                BehaviorNodeType.WeightedRandom => new WeightedRandomNode(bp.Name, 1, bp.Children.Select(c => (CompileNode(c), c.Weight)).ToArray()),
                BehaviorNodeType.Loop => new LoopNode(bp.Name, bp.Children.Count > 0 ? CompileNode(bp.Children[0]) : new WaitNode("Empty", 0), bp.Repetitions),
                BehaviorNodeType.Guard => new GuardNode(bp.Name, bp.Children.Count > 0 ? CompileNode(bp.Children[0]) : new WaitNode("Empty", 0), (e, c) => EvaluateCondition(bp.Condition, e, c)),
                BehaviorNodeType.Succeeder => new SucceederNode(bp.Name, bp.Children.Count > 0 ? CompileNode(bp.Children[0]) : new WaitNode("Empty", 0)),
                BehaviorNodeType.Failer => new FailerNode(bp.Name, bp.Children.Count > 0 ? CompileNode(bp.Children[0]) : new WaitNode("Empty", 0)),
                BehaviorNodeType.Wait => new WaitNode(bp.Name, bp.CooldownDuration > 0 ? bp.CooldownDuration : 1f),
                BehaviorNodeType.LLMQuery => new LLMQueryNode(bp.Name, bp.LLMPrompt, BehaviorLlmContext.DefaultHandle),
                _ => new WaitNode(bp.Name, 0)
            };
        }

        public BehaviorTree OptimizeTree(BehaviorTree tree)
        {
            tree.Optimize();
            return tree;
        }

        public List<string> ValidateTree(BehaviorTree tree)
        {
            var errors = new List<string>();
            tree.Validate();
            errors.AddRange(tree.ValidationErrors);
            return errors;
        }

        public string CompileToText(IBehaviorNode node, int depth = 0)
        {
            var sb = new StringBuilder();
            string indent = new string(' ', depth * 2);
            sb.AppendLine($"{indent}{node.NodeType}: {node.Name} [{node.Id}]");
            foreach (var child in node.GetChildren())
                sb.Append(CompileToText(child, depth + 1));
            return sb.ToString();
        }
    }

}
