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

    public abstract class BehaviorNodeBase : IBehaviorNode
    {
        public string Id { get; init; } = Guid.NewGuid().ToString("N")[..8];
        public string Name { get; set; } = string.Empty;
        public abstract BehaviorNodeType NodeType { get; }
        public TaskStatus CurrentStatus { get; protected set; } = TaskStatus.Pending;
        public Blackboard? Blackboard { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new();

        public abstract TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime);
        public virtual IReadOnlyList<IBehaviorNode> GetChildren() => Array.Empty<IBehaviorNode>();
        public virtual void Reset() { CurrentStatus = TaskStatus.Pending; }
        public abstract IBehaviorNode Clone();

        public virtual bool Validate(List<string> errors)
        {
            if (string.IsNullOrWhiteSpace(Name))
            { errors.Add($"Node {Id} has no name."); return false; }
            return true;
        }

        protected Dictionary<string, object> CloneMetadata() => new(Metadata);
    }

    public class SequenceNode : BehaviorNodeBase
    {
        private readonly List<IBehaviorNode> _children;
        private int _currentChildIndex;

        public SequenceNode(string name, params IBehaviorNode[] children)
        { Name = name; _children = new List<IBehaviorNode>(children); }

        public override BehaviorNodeType NodeType => BehaviorNodeType.Sequence;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            if (_children.Count == 0)
            { CurrentStatus = TaskStatus.Success; return CurrentStatus; }
            for (int i = _currentChildIndex; i < _children.Count; i++)
            {
                var status = _children[i].Tick(entity, context, deltaTime);
                if (status == TaskStatus.Running || status == TaskStatus.Pending)
                { _currentChildIndex = i; CurrentStatus = TaskStatus.Running; return CurrentStatus; }
                if (status == TaskStatus.Failure)
                { _currentChildIndex = 0; CurrentStatus = TaskStatus.Failure; return CurrentStatus; }
            }
            _currentChildIndex = 0;
            CurrentStatus = TaskStatus.Success;
            return CurrentStatus;
        }

        public override IReadOnlyList<IBehaviorNode> GetChildren() => _children.AsReadOnly();
        public override void Reset() { base.Reset(); _currentChildIndex = 0; foreach (var c in _children) c.Reset(); }
        public override IBehaviorNode Clone() => new SequenceNode(Name, _children.Select(c => c.Clone()).ToArray()) { Blackboard = Blackboard, Metadata = CloneMetadata() };

        public override bool Validate(List<string> errors)
        {
            if (!base.Validate(errors))
                return false;
            if (_children.Count == 0)
            { errors.Add($"Sequence '{Name}' has no children."); return false; }
            bool valid = true;
            foreach (var c in _children)
                if (!c.Validate(errors))
                    valid = false;
            return valid;
        }
    }

    public class SelectorNode : BehaviorNodeBase
    {
        private readonly List<IBehaviorNode> _children;
        private int _currentChildIndex;

        public SelectorNode(string name, params IBehaviorNode[] children)
        { Name = name; _children = new List<IBehaviorNode>(children); }

        public override BehaviorNodeType NodeType => BehaviorNodeType.Selector;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            if (_children.Count == 0)
            { CurrentStatus = TaskStatus.Failure; return CurrentStatus; }
            for (int i = _currentChildIndex; i < _children.Count; i++)
            {
                var status = _children[i].Tick(entity, context, deltaTime);
                if (status == TaskStatus.Running || status == TaskStatus.Pending)
                { _currentChildIndex = i; CurrentStatus = TaskStatus.Running; return CurrentStatus; }
                if (status == TaskStatus.Success)
                { _currentChildIndex = 0; CurrentStatus = TaskStatus.Success; return CurrentStatus; }
            }
            _currentChildIndex = 0;
            CurrentStatus = TaskStatus.Failure;
            return CurrentStatus;
        }

        public override IReadOnlyList<IBehaviorNode> GetChildren() => _children.AsReadOnly();
        public override void Reset() { base.Reset(); _currentChildIndex = 0; foreach (var c in _children) c.Reset(); }
        public override IBehaviorNode Clone() => new SelectorNode(Name, _children.Select(c => c.Clone()).ToArray()) { Blackboard = Blackboard, Metadata = CloneMetadata() };

        public override bool Validate(List<string> errors)
        {
            if (!base.Validate(errors))
                return false;
            if (_children.Count == 0)
            { errors.Add($"Selector '{Name}' has no children."); return false; }
            bool valid = true;
            foreach (var c in _children)
                if (!c.Validate(errors))
                    valid = false;
            return valid;
        }
    }

    public class ParallelNode : BehaviorNodeBase
    {
        private readonly List<IBehaviorNode> _children;
        private readonly int _successThreshold;
        private readonly int _failureThreshold;
        private readonly Dictionary<string, TaskStatus> _childStatuses = new();

        public ParallelNode(string name, int successThreshold, int failureThreshold, params IBehaviorNode[] children)
        { Name = name; _successThreshold = successThreshold; _failureThreshold = failureThreshold; _children = new List<IBehaviorNode>(children); }

        public override BehaviorNodeType NodeType => BehaviorNodeType.Parallel;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            if (_children.Count == 0)
            { CurrentStatus = TaskStatus.Success; return CurrentStatus; }
            int sc = 0, fc = 0;
            foreach (var child in _children)
            {
                if (_childStatuses.TryGetValue(child.Id, out var es) && es != TaskStatus.Running && es != TaskStatus.Pending)
                { if (es == TaskStatus.Success) sc++; else if (es == TaskStatus.Failure) fc++; continue; }
                var s = child.Tick(entity, context, deltaTime);
                _childStatuses[child.Id] = s;
                if (s == TaskStatus.Success)
                    sc++;
                else if (s == TaskStatus.Failure)
                    fc++;
            }
            if (fc >= _failureThreshold)
            { CurrentStatus = TaskStatus.Failure; _childStatuses.Clear(); return CurrentStatus; }
            if (sc >= _successThreshold)
            { CurrentStatus = TaskStatus.Success; _childStatuses.Clear(); return CurrentStatus; }
            CurrentStatus = TaskStatus.Running;
            return CurrentStatus;
        }

        public override IReadOnlyList<IBehaviorNode> GetChildren() => _children.AsReadOnly();
        public override void Reset() { base.Reset(); _childStatuses.Clear(); foreach (var c in _children) c.Reset(); }
        public override IBehaviorNode Clone() => new ParallelNode(Name, _successThreshold, _failureThreshold, _children.Select(c => c.Clone()).ToArray()) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class ConditionNode : BehaviorNodeBase
    {
        private readonly Func<SentientEntity, EntityContext, bool> _condition;
        public ConditionNode(string name, Func<SentientEntity, EntityContext, bool> condition) { Name = name; _condition = condition; }
        public override BehaviorNodeType NodeType => BehaviorNodeType.Condition;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            try
            { CurrentStatus = _condition(entity, context) ? TaskStatus.Success : TaskStatus.Failure; }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("ConditionNode", $"Condition '{Name}' threw an exception.", ex);
                CurrentStatus = TaskStatus.Failure;
            }
            return CurrentStatus;
        }

        public override IBehaviorNode Clone() => new ConditionNode(Name, _condition) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class ActionNode : BehaviorNodeBase
    {
        private readonly Func<SentientEntity, EntityContext, float, TaskStatus> _action;
        private float _elapsed;
        private readonly float _duration;

        public ActionNode(string name, Func<SentientEntity, EntityContext, float, TaskStatus> action, float duration = 0f)
        { Name = name; _action = action; _duration = duration; }

        public override BehaviorNodeType NodeType => BehaviorNodeType.Action;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            try
            {
                if (_duration > 0)
                { _elapsed += deltaTime; if (_elapsed < _duration) { CurrentStatus = TaskStatus.Running; return CurrentStatus; } _elapsed = 0; }
                CurrentStatus = _action(entity, context, deltaTime);
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("ActionNode", $"Action '{Name}' threw an exception.", ex);
                CurrentStatus = TaskStatus.Failure;
            }
            return CurrentStatus;
        }

        public override void Reset() { base.Reset(); _elapsed = 0; }
        public override IBehaviorNode Clone() => new ActionNode(Name, _action, _duration) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public abstract class DecoratorNode : BehaviorNodeBase
    {
        protected IBehaviorNode Child { get; }
        protected DecoratorNode(string name, IBehaviorNode child) { Name = name; Child = child; }
        public override IReadOnlyList<IBehaviorNode> GetChildren() => new[] { Child };
        public override void Reset() { base.Reset(); Child.Reset(); }
        public override bool Validate(List<string> errors) { if (!base.Validate(errors)) return false; return Child.Validate(errors); }
    }

    public class InverterNode : DecoratorNode
    {
        public InverterNode(string name, IBehaviorNode child) : base(name, child) { }
        public override BehaviorNodeType NodeType => BehaviorNodeType.Inverter;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            var s = Child.Tick(entity, context, deltaTime);
            CurrentStatus = s switch { TaskStatus.Success => TaskStatus.Failure, TaskStatus.Failure => TaskStatus.Success, _ => s };
            return CurrentStatus;
        }

        public override IBehaviorNode Clone() => new InverterNode(Name, Child.Clone()) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class RepeaterNode : DecoratorNode
    {
        private int _count;
        private readonly int _repetitions;
        public RepeaterNode(string name, IBehaviorNode child, int repetitions = -1) : base(name, child) { _repetitions = repetitions; }
        public override BehaviorNodeType NodeType => BehaviorNodeType.Repeater;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            var s = Child.Tick(entity, context, deltaTime);
            if (s == TaskStatus.Running || s == TaskStatus.Pending)
            { CurrentStatus = TaskStatus.Running; return CurrentStatus; }
            _count++;
            Child.Reset();
            if (_repetitions > 0 && _count >= _repetitions)
            { _count = 0; CurrentStatus = TaskStatus.Success; return CurrentStatus; }
            CurrentStatus = TaskStatus.Running;
            return CurrentStatus;
        }

        public override void Reset() { base.Reset(); _count = 0; }
        public override IBehaviorNode Clone() => new RepeaterNode(Name, Child.Clone(), _repetitions) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class CooldownNode : DecoratorNode
    {
        private float _remaining;
        private readonly float _duration;
        public CooldownNode(string name, IBehaviorNode child, float duration) : base(name, child) { _duration = duration; }
        public override BehaviorNodeType NodeType => BehaviorNodeType.Cooldown;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            if (_remaining > 0)
            { _remaining -= deltaTime; CurrentStatus = TaskStatus.Failure; return CurrentStatus; }
            var s = Child.Tick(entity, context, deltaTime);
            if (s != TaskStatus.Running)
                _remaining = _duration;
            CurrentStatus = s;
            return CurrentStatus;
        }

        public override void Reset() { base.Reset(); _remaining = 0; }
        public override IBehaviorNode Clone() => new CooldownNode(Name, Child.Clone(), _duration) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class RandomSelectorNode : BehaviorNodeBase
    {
        private readonly List<(IBehaviorNode Node, float Weight)> _children;
        private readonly Random _rng;
        private IBehaviorNode? _selected;

        public RandomSelectorNode(string name, params (IBehaviorNode Node, float Weight)[] children)
        { Name = name; _children = new List<(IBehaviorNode, float)>(children); _rng = new Random(); }

        public override BehaviorNodeType NodeType => BehaviorNodeType.RandomSelector;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            if (_children.Count == 0)
            { CurrentStatus = TaskStatus.Failure; return CurrentStatus; }
            if (_selected == null)
            {
                float total = _children.Sum(c => c.Weight);
                float r = (float)(_rng.NextDouble() * total);
                float cum = 0;
                foreach (var (n, w) in _children)
                { cum += w; if (r <= cum) { _selected = n; break; } }
                _selected ??= _children[^1].Node;
            }
            var s = _selected.Tick(entity, context, deltaTime);
            if (s != TaskStatus.Running)
                _selected = null;
            CurrentStatus = s;
            return CurrentStatus;
        }

        public override IReadOnlyList<IBehaviorNode> GetChildren() => _children.Select(c => c.Node).ToList().AsReadOnly();
        public override void Reset() { base.Reset(); _selected = null; foreach (var (n, _) in _children) n.Reset(); }
        public override IBehaviorNode Clone() => new RandomSelectorNode(Name, _children.Select(c => (c.Node.Clone(), c.Weight)).ToArray()) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class WeightedRandomNode : BehaviorNodeBase
    {
        private readonly List<(IBehaviorNode Node, float Weight)> _children;
        private readonly Random _rng;
        private IBehaviorNode? _selected;
        private int _selCount;
        private readonly int _maxSel;

        public WeightedRandomNode(string name, int maxSelections = 1, params (IBehaviorNode Node, float Weight)[] children)
        { Name = name; _children = new List<(IBehaviorNode, float)>(children); _rng = new Random(); _maxSel = maxSelections; }

        public override BehaviorNodeType NodeType => BehaviorNodeType.WeightedRandom;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            if (_children.Count == 0 || _selCount >= _maxSel)
            { CurrentStatus = TaskStatus.Failure; return CurrentStatus; }
            if (_selected == null)
            {
                float total = _children.Sum(c => c.Weight);
                float r = (float)(_rng.NextDouble() * total);
                float cum = 0;
                foreach (var (n, w) in _children)
                { cum += w; if (r <= cum) { _selected = n; break; } }
                _selected ??= _children[^1].Node;
                _selCount++;
            }
            var s = _selected.Tick(entity, context, deltaTime);
            if (s != TaskStatus.Running)
                _selected = null;
            CurrentStatus = s;
            return CurrentStatus;
        }

        public override IReadOnlyList<IBehaviorNode> GetChildren() => _children.Select(c => c.Node).ToList().AsReadOnly();
        public override void Reset() { base.Reset(); _selected = null; _selCount = 0; foreach (var (n, _) in _children) n.Reset(); }
        public override IBehaviorNode Clone() => new WeightedRandomNode(Name, _maxSel, _children.Select(c => (c.Node.Clone(), c.Weight)).ToArray()) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class LoopNode : BehaviorNodeBase
    {
        private readonly IBehaviorNode _child;
        private int _iter;
        private readonly int _iterations;

        public LoopNode(string name, IBehaviorNode child, int iterations = -1) { Name = name; _child = child; _iterations = iterations; }
        public override BehaviorNodeType NodeType => BehaviorNodeType.Loop;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            var s = _child.Tick(entity, context, deltaTime);
            if (s == TaskStatus.Running)
            { CurrentStatus = TaskStatus.Running; return CurrentStatus; }
            _iter++;
            _child.Reset();
            if (_iterations > 0 && _iter >= _iterations)
            { _iter = 0; CurrentStatus = TaskStatus.Success; return CurrentStatus; }
            CurrentStatus = TaskStatus.Running;
            return CurrentStatus;
        }

        public override IReadOnlyList<IBehaviorNode> GetChildren() => new[] { _child };
        public override void Reset() { base.Reset(); _iter = 0; _child.Reset(); }
        public override IBehaviorNode Clone() => new LoopNode(Name, _child.Clone(), _iterations) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class GuardNode : BehaviorNodeBase
    {
        private readonly IBehaviorNode _child;
        private readonly Func<SentientEntity, EntityContext, bool> _condition;

        public GuardNode(string name, IBehaviorNode child, Func<SentientEntity, EntityContext, bool> condition)
        { Name = name; _child = child; _condition = condition; }

        public override BehaviorNodeType NodeType => BehaviorNodeType.Guard;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            try
            { if (!_condition(entity, context)) { CurrentStatus = TaskStatus.Failure; return CurrentStatus; } }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("GuardNode", $"Guard condition '{Name}' threw an exception.", ex);
                CurrentStatus = TaskStatus.Failure;
                return CurrentStatus;
            }
            CurrentStatus = _child.Tick(entity, context, deltaTime);
            return CurrentStatus;
        }

        public override IReadOnlyList<IBehaviorNode> GetChildren() => new[] { _child };
        public override void Reset() { base.Reset(); _child.Reset(); }
        public override IBehaviorNode Clone() => new GuardNode(Name, _child.Clone(), _condition) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class SucceederNode : DecoratorNode
    {
        public SucceederNode(string name, IBehaviorNode child) : base(name, child) { }
        public override BehaviorNodeType NodeType => BehaviorNodeType.Succeeder;
        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime) { Child.Tick(entity, context, deltaTime); CurrentStatus = TaskStatus.Success; return CurrentStatus; }
        public override IBehaviorNode Clone() => new SucceederNode(Name, Child.Clone()) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class FailerNode : DecoratorNode
    {
        public FailerNode(string name, IBehaviorNode child) : base(name, child) { }
        public override BehaviorNodeType NodeType => BehaviorNodeType.Failer;
        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime) { Child.Tick(entity, context, deltaTime); CurrentStatus = TaskStatus.Failure; return CurrentStatus; }
        public override IBehaviorNode Clone() => new FailerNode(Name, Child.Clone()) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class WaitNode : BehaviorNodeBase
    {
        private float _elapsed;
        private readonly float _duration;
        public WaitNode(string name, float duration) { Name = name; _duration = duration; }
        public override BehaviorNodeType NodeType => BehaviorNodeType.Wait;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            _elapsed += deltaTime;
            if (_elapsed >= _duration)
            { _elapsed = 0; CurrentStatus = TaskStatus.Success; return CurrentStatus; }
            CurrentStatus = TaskStatus.Running;
            return CurrentStatus;
        }

        public override void Reset() { base.Reset(); _elapsed = 0; }
        public override IBehaviorNode Clone() => new WaitNode(Name, _duration) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    public class LLMQueryNode : BehaviorNodeBase
    {
        private readonly string _prompt;
        private readonly Func<SentientEntity, EntityContext, string, TaskStatus> _handler;
        private string _response = string.Empty;
        private bool _sent;
        private readonly float _timeout;

        public LLMQueryNode(string name, string prompt, Func<SentientEntity, EntityContext, string, TaskStatus> handler, float timeout = 5f)
        { Name = name; _prompt = prompt; _handler = handler; _timeout = timeout; }

        public override BehaviorNodeType NodeType => BehaviorNodeType.LLMQuery;

        public override TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime)
        {
            if (!_sent)
            { _sent = true; _ = QueryAsync(entity, context); CurrentStatus = TaskStatus.Running; return CurrentStatus; }
            if (!string.IsNullOrEmpty(_response))
            {
                try
                { CurrentStatus = _handler(entity, context, _response); }
                catch (Exception ex)
                {
                    SynapseLogger.Default.Warn("LLMQueryNode", $"Response handler for '{Name}' threw an exception.", ex);
                    CurrentStatus = TaskStatus.Failure;
                }
                _sent = false;
                _response = string.Empty;
                return CurrentStatus;
            }
            CurrentStatus = TaskStatus.Running;
            return CurrentStatus;
        }

        private async Task QueryAsync(SentientEntity entity, EntityContext context)
        {
            try
            {
                var cts = new CancellationTokenSource(TimeSpan.FromSeconds(_timeout));
                if (BehaviorLlmContext.QueryAsync != null)
                    _response = await BehaviorLlmContext.QueryAsync(_prompt, entity, context, cts.Token).ConfigureAwait(false);
                else
                {
                    await Task.Delay(50, cts.Token).ConfigureAwait(false);
                    _response = $"Entity {entity.EntityId} context: {_prompt}";
                }
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("LLMQueryNode", $"LLM query for '{Name}' timed out or failed.", ex);
                _response = "TIMEOUT";
            }
        }

        public override void Reset() { base.Reset(); _sent = false; _response = string.Empty; }
        public override IBehaviorNode Clone() => new LLMQueryNode(Name, _prompt, _handler, _timeout) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

}
