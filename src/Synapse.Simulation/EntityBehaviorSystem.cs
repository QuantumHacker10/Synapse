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

namespace GDNN.Sentience
{
    #region ENUMS

    public enum BehaviorState { Idle, Active, Dormant, Transitioning, Terminated }
    public enum EntityType { Static, Dynamic, Kinematic, Player, NPC, Environmental, Trigger, Sensor }
    public enum PerceptionType { Visual, Auditory, Tactile, Proximity, Semantic, Emotional }
    public enum RelationshipType { Neutral, Friendly, Hostile, Parent, Child, Sibling, Partner, Servant, Master }
    public enum EmotionalState { Neutral, Happy, Sad, Angry, Fearful, Surprised, Disgusted, Trusting, Anticipating, Calm, Excited, Bored, Curious, Confused, Determined, Anxious, Frustrated }
    public enum ScheduleMode { Fixed, Variable, EventDriven, Priority, Adaptive }
    public enum BehaviorNodeType { Sequence, Selector, Parallel, Condition, Action, Decorator, RandomSelector, Loop, Inverter, Repeater, Succeeder, Failer, Wait, Cooldown, Guard, LLMQuery, WeightedRandom }
    public enum TaskStatus { Success, Failure, Running, Pending }
    public enum WeatherType { Clear, Cloudy, Rain, HeavyRain, Snow, Fog, Storm, Sandstorm }
    public enum MemoryType { ShortTerm, LongTerm, Episodic, Semantic, Procedural }
    public enum NodeType { Root, Branch, Leaf }

    #endregion
    #region RECORDS

    public record PerceptionEvent(PerceptionType Type, Guid Source, double Timestamp, float Intensity, Vector3 Location, Vector3 Direction, string SemanticContent, float Confidence);
    public record EmotionalResponse(EmotionalState Emotion, float Intensity, float Duration, string Trigger, float DecayRate);
    public record BehaviorAction(string Type, Dictionary<string, object> Parameters, int Priority, float Duration, Guid Target);
    public record RelationshipConstraint(RelationshipType Type, float Strength, bool Bidirectional, bool Exclusive);
    public record WeatherConditions(float Temperature, float Humidity, float WindSpeed, float WindDirection, float Visibility, float AmbientLight, WeatherType Type);
    public record WorldEvent(Guid Id, string EventType, Vector3 Location, float Radius, float Intensity, double Timestamp, Guid SourceEntity, Dictionary<string, object> Metadata);

    internal static class SentienceQuaternion
    {
        public static Quaternion LookRotation(Vector3 forward, Vector3 up)
        {
            var matrix = Matrix4x4.CreateWorld(Vector3.Zero, Vector3.Normalize(forward), up);
            return Quaternion.CreateFromRotationMatrix(matrix);
        }
    }

    public class WorldStateData
    {
        public Dictionary<Guid, SentientEntity> Entities { get; set; } = new();
        public double Time { get; set; }
        public WeatherConditions Weather { get; set; } = new(20f, 0.5f, 5f, 0f, 1f, 1f, WeatherType.Clear);
        public Queue<WorldEvent> Events { get; set; } = new();
        public SpatialIndex SpatialIndex { get; set; } = new();
    }

    public class EntityContext
    {
        public SentientEntity Entity { get; set; } = null!;
        public WorldStateData WorldState { get; set; } = null!;
        public float DeltaTime { get; set; }
        public List<PerceptionEvent> PerceptionEvents { get; set; } = new();
        public IMemorySystem Memory { get; set; } = null!;
        public EmotionalState EmotionalState { get; set; }
        public Dictionary<Guid, RelationshipType> Relationships { get; set; } = new();
    }

    public class MemoryEntry
    {
        public Guid Id { get; init; } = Guid.NewGuid();
        public string Content { get; set; } = string.Empty;
        public double Timestamp { get; set; }
        public float EmotionalIntensity { get; set; }
        public float Importance { get; set; }
        public int RetrievalCount { get; set; }
        public double LastRetrieved { get; set; }
        public List<Guid> AssociatedEntities { get; set; } = new();
        public Vector3 Location { get; set; }
        public MemoryType Type { get; set; } = MemoryType.Episodic;
        public HashSet<string> Tags { get; set; } = new();
        public float ConsolidationStrength { get; set; } = 0.1f;
        public bool IsConsolidated { get; set; }
        public float DecayFactor { get; set; } = 1.0f;
        public Dictionary<string, object> Context { get; set; } = new();
    }

    public class Relationship
    {
        public Guid EntityA { get; set; }
        public Guid EntityB { get; set; }
        public RelationshipType Type { get; set; }
        public float Strength { get; set; }
        public double EstablishedTime { get; set; }
        public List<RelationshipEvent> History { get; set; } = new();
        public HashSet<string> Tags { get; set; } = new();
        public bool Bidirectional { get; set; } = true;
    }

    public record RelationshipEvent(RelationshipType PreviousType, RelationshipType NewType, float PreviousStrength, float NewStrength, double Timestamp, string Reason);

    public class BehaviorTreeBlueprint
    {
        public BehaviorNodeType NodeType { get; set; }
        public string NodeId { get; set; } = Guid.NewGuid().ToString("N")[..8];
        public string Name { get; set; } = string.Empty;
        public Dictionary<string, object> Parameters { get; set; } = new();
        public List<BehaviorTreeBlueprint> Children { get; set; } = new();
        public string Condition { get; set; } = string.Empty;
        public string ActionType { get; set; } = string.Empty;
        public float Weight { get; set; } = 1.0f;
        public float CooldownDuration { get; set; }
        public int Repetitions { get; set; } = -1;
        public string LLMPrompt { get; set; } = string.Empty;
        public string BlackboardKey { get; set; } = string.Empty;
    }

    public class MemoryQuery
    {
        public string SearchTerms { get; set; } = string.Empty;
        public MemoryType? Type { get; set; }
        public float MinEmotionalIntensity { get; set; }
        public float MinImportance { get; set; }
        public double MaxAge { get; set; } = double.MaxValue;
        public Guid? AssociatedEntity { get; set; }
        public (Vector3 Center, float Radius)? LocationFilter { get; set; }
        public List<string> Tags { get; set; } = new();
        public int MaxResults { get; set; } = 100;
        public bool SortByEmotionalIntensity { get; set; }
        public bool SortByRecency { get; set; } = true;
        public bool SortByImportance { get; set; }
    }

    public class PerceptionFilter
    {
        public HashSet<PerceptionType> AllowedTypes { get; set; } = new();
        public float MinIntensity { get; set; }
        public float MaxDistance { get; set; } = float.MaxValue;
        public HashSet<Guid> AllowedSources { get; set; } = new();
        public HashSet<Guid> ExcludedSources { get; set; } = new();
        public float MinConfidence { get; set; }
    }

    public class EmotionalStateData
    {
        public Dictionary<EmotionalState, float> EmotionIntensities { get; set; } = new();
        public EmotionalState DominantEmotion { get; set; } = EmotionalState.Neutral;
        public float Inertia { get; set; } = 0.7f;
        public float Volatility { get; set; } = 0.3f;
        public EmotionalState BaselineEmotion { get; set; } = EmotionalState.Neutral;
        public List<(EmotionalState Emotion, float Intensity, double Timestamp)> History { get; set; } = new();

        public void UpdateDominantEmotion()
        {
            if (EmotionIntensities.Count == 0)
            { DominantEmotion = BaselineEmotion; return; }
            DominantEmotion = EmotionIntensities.OrderByDescending(kv => kv.Value).First().Key;
        }

        public float GetArousalLevel() => Math.Min(1.0f, EmotionIntensities.Values.Sum());

        public float GetValence()
        {
            float v = 0;
            foreach (var (e, i) in EmotionIntensities)
                v += GetValence(e) * i;
            return Math.Clamp(v, -1, 1);
        }

        private static float GetValence(EmotionalState e) => e switch
        {
            EmotionalState.Happy => 1f,
            EmotionalState.Sad => -1f,
            EmotionalState.Angry => -0.8f,
            EmotionalState.Fearful => -0.9f,
            EmotionalState.Surprised => 0.3f,
            EmotionalState.Disgusted => -0.7f,
            EmotionalState.Trusting => 0.6f,
            EmotionalState.Anticipating => 0.4f,
            EmotionalState.Calm => 0.7f,
            EmotionalState.Excited => 0.8f,
            EmotionalState.Bored => -0.3f,
            EmotionalState.Curious => 0.5f,
            EmotionalState.Confused => -0.2f,
            EmotionalState.Determined => 0.6f,
            EmotionalState.Anxious => -0.6f,
            _ => 0f
        };
    }

    #endregion
    #region INTERFACES

    public interface IBehaviorNode
    {
        string Id { get; }
        string Name { get; }
        BehaviorNodeType NodeType { get; }
        TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime);
        IReadOnlyList<IBehaviorNode> GetChildren();
        void Reset();
        IBehaviorNode Clone();
        bool Validate(List<string> errors);
    }

    public interface IPerceptionSystem
    {
        void Update(SentientEntity entity, WorldStateData worldState);
        IReadOnlyList<PerceptionEvent> GetPerceptions(SentientEntity entity);
        IReadOnlyList<PerceptionEvent> FilterPerceptions(SentientEntity entity, PerceptionFilter filters);
    }

    public interface IEmotionalModel
    {
        void Update(SentientEntity entity, IReadOnlyList<PerceptionEvent> events, float deltaTime);
        EmotionalState GetCurrentState(SentientEntity entity);
        EmotionalResponse ReactToEvent(SentientEntity entity, PerceptionEvent perceptionEvent);
        IReadOnlyDictionary<EmotionalState, float> GetEmotionalProfile(SentientEntity entity);
    }

    public interface IMemorySystem
    {
        void Store(SentientEntity entity, MemoryEntry memory);
        IReadOnlyList<MemoryEntry> Retrieve(SentientEntity entity, MemoryQuery query);
        int Forget(SentientEntity entity, Func<MemoryEntry, bool> criteria);
        void Consolidate(SentientEntity entity);
        IReadOnlyDictionary<MemoryType, int> GetMemoryCounts(SentientEntity entity);
    }

    #endregion
    #region SPATIAL INDEX

    public class SpatialIndex
    {
        private readonly Dictionary<(int, int, int), HashSet<Guid>> _cells;
        private readonly Dictionary<Guid, (int, int, int)> _entityCells;
        private readonly float _cellSize;
        private readonly object _lock = new();

        public SpatialIndex(float cellSize = 10.0f)
        {
            _cellSize = cellSize;
            _cells = new Dictionary<(int, int, int), HashSet<Guid>>();
            _entityCells = new Dictionary<Guid, (int, int, int)>();
        }

        public void Insert(Guid entityId, Vector3 position)
        {
            var cell = GetCell(position);
            lock (_lock)
            {
                if (_entityCells.TryGetValue(entityId, out var oldCell) && _cells.TryGetValue(oldCell, out var oldSet))
                {
                    oldSet.Remove(entityId);
                    if (oldSet.Count == 0)
                        _cells.Remove(oldCell);
                }
                if (!_cells.TryGetValue(cell, out var cellSet))
                { cellSet = new HashSet<Guid>(); _cells[cell] = cellSet; }
                cellSet.Add(entityId);
                _entityCells[entityId] = cell;
            }
        }

        public void Remove(Guid entityId)
        {
            lock (_lock)
            {
                if (_entityCells.TryGetValue(entityId, out var cell) && _cells.TryGetValue(cell, out var cellSet))
                {
                    cellSet.Remove(entityId);
                    if (cellSet.Count == 0)
                        _cells.Remove(cell);
                }
                _entityCells.Remove(entityId);
            }
        }

        public List<Guid> QueryRadius(Vector3 center, float radius)
        {
            var results = new List<Guid>();
            var minCell = GetCell(center - new Vector3(radius));
            var maxCell = GetCell(center + new Vector3(radius));
            lock (_lock)
            {
                for (int x = minCell.Item1; x <= maxCell.Item1; x++)
                    for (int y = minCell.Item2; y <= maxCell.Item2; y++)
                        for (int z = minCell.Item3; z <= maxCell.Item3; z++)
                            if (_cells.TryGetValue((x, y, z), out var s))
                                results.AddRange(s);
            }
            return results;
        }

        public List<Guid> QueryBox(Vector3 min, Vector3 max)
        {
            var minCell = GetCell(min);
            var maxCell = GetCell(max);
            var results = new List<Guid>();
            lock (_lock)
            {
                for (int x = minCell.Item1; x <= maxCell.Item1; x++)
                    for (int y = minCell.Item2; y <= maxCell.Item2; y++)
                        for (int z = minCell.Item3; z <= maxCell.Item3; z++)
                            if (_cells.TryGetValue((x, y, z), out var s))
                                results.AddRange(s);
            }
            return results;
        }

        public Guid? QueryNearest(Vector3 point, Dictionary<Guid, Vector3> positions, float maxDistance = float.MaxValue)
        {
            var candidates = QueryRadius(point, maxDistance);
            Guid? nearest = null;
            float bestDistSq = maxDistance * maxDistance;
            foreach (var id in candidates)
            {
                if (positions.TryGetValue(id, out var pos))
                {
                    float d = Vector3.DistanceSquared(point, pos);
                    if (d < bestDistSq)
                    { bestDistSq = d; nearest = id; }
                }
            }
            return nearest;
        }

        public void Clear() { lock (_lock) { _cells.Clear(); _entityCells.Clear(); } }
        public int Count { get { lock (_lock) return _entityCells.Count; } }

        private (int, int, int) GetCell(Vector3 p) =>
            ((int)Math.Floor(p.X / _cellSize), (int)Math.Floor(p.Y / _cellSize), (int)Math.Floor(p.Z / _cellSize));
    }

    #endregion

    #region BLACKBOARD

    public class Blackboard
    {
        private readonly Dictionary<string, object> _data = new();
        private readonly ReaderWriterLockSlim _lock = new();
        private readonly Dictionary<string, DateTime> _lastModified = new();

        public void Set(string key, object value) { _lock.EnterWriteLock(); try { _data[key] = value; _lastModified[key] = DateTime.UtcNow; } finally { _lock.ExitWriteLock(); } }

        public T? Get<T>(string key) { _lock.EnterReadLock(); try { return _data.TryGetValue(key, out var v) && v is T t ? t : default; } finally { _lock.ExitReadLock(); } }

        public T GetOrSet<T>(string key, Func<T> factory)
        {
            _lock.EnterUpgradeableReadLock();
            try
            {
                if (_data.TryGetValue(key, out var v) && v is T t)
                    return t;
                _lock.EnterWriteLock();
                try
                { var nv = factory(); _data[key] = nv; _lastModified[key] = DateTime.UtcNow; return nv; }
                finally { _lock.ExitWriteLock(); }
            }
            finally { _lock.ExitUpgradeableReadLock(); }
        }

        public bool Has(string key) { _lock.EnterReadLock(); try { return _data.ContainsKey(key); } finally { _lock.ExitReadLock(); } }
        public bool Remove(string key) { _lock.EnterWriteLock(); try { _lastModified.Remove(key); return _data.Remove(key); } finally { _lock.ExitWriteLock(); } }
        public IReadOnlyCollection<string> Keys { get { _lock.EnterReadLock(); try { return _data.Keys.ToList().AsReadOnly(); } finally { _lock.ExitReadLock(); } } }
        public int Count { get { _lock.EnterReadLock(); try { return _data.Count; } finally { _lock.ExitReadLock(); } } }

        public void Clear() { _lock.EnterWriteLock(); try { _data.Clear(); _lastModified.Clear(); } finally { _lock.ExitWriteLock(); } }

        public DateTime? GetLastModified(string key) { _lock.EnterReadLock(); try { return _lastModified.TryGetValue(key, out var t) ? t : null; } finally { _lock.ExitReadLock(); } }
    }

    #endregion
    #region BEHAVIOR TREE NODES

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
            catch { CurrentStatus = TaskStatus.Failure; }
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
            catch { CurrentStatus = TaskStatus.Failure; }
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
            catch { CurrentStatus = TaskStatus.Failure; return CurrentStatus; }
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
                catch { CurrentStatus = TaskStatus.Failure; }
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
            catch { _response = "TIMEOUT"; }
        }

        public override void Reset() { base.Reset(); _sent = false; _response = string.Empty; }
        public override IBehaviorNode Clone() => new LLMQueryNode(Name, _prompt, _handler, _timeout) { Blackboard = Blackboard, Metadata = CloneMetadata() };
    }

    #endregion
    #region BEHAVIOR TREE

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
            catch { return null; }
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

    #endregion
    #region SENTIENT ENTITY

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

    #endregion
    #region PERCEPTION SYSTEM

    public class PerceptionSystem : IPerceptionSystem
    {
        private readonly Dictionary<Guid, List<PerceptionEvent>> _entityPerceptions = new();
        private readonly Dictionary<Guid, AttentionModel> _attentionModels = new();
        private readonly SpatialIndex _spatialIndex;
        private readonly object _lock = new();

        public PerceptionSystem() { _spatialIndex = new SpatialIndex(20f); }

        public float VisualRange { get; set; } = 100f;
        public float AuditoryRange { get; set; } = 50f;
        public float ProximityRange { get; set; } = 10f;
        public float PerceptionThreshold { get; set; } = 0.1f;
        public bool UseAttentionFiltering { get; set; } = true;
        public bool SimulateOcclusion { get; set; } = true;

        public void Update(SentientEntity entity, WorldStateData worldState)
        {
            var perceptions = new List<PerceptionEvent>();
            var nearbyIds = _spatialIndex.QueryRadius(entity.Position, entity.PerceptionRadius);

            foreach (var eid in nearbyIds)
            {
                if (eid == entity.EntityId || !worldState.Entities.TryGetValue(eid, out var other))
                    continue;
                float dist = entity.DistanceTo(other);

                if (entity.PerceptionCapabilities.Contains(PerceptionType.Visual) && dist <= VisualRange)
                {
                    var p = ProcessVisual(entity, other, dist, worldState);
                    if (p != null)
                        perceptions.Add(p);
                }
                if (entity.PerceptionCapabilities.Contains(PerceptionType.Auditory) && dist <= AuditoryRange)
                {
                    var p = ProcessAuditory(entity, other, dist, worldState);
                    if (p != null)
                        perceptions.Add(p);
                }
                if (entity.PerceptionCapabilities.Contains(PerceptionType.Proximity) && dist <= ProximityRange)
                {
                    var p = ProcessProximity(entity, other, dist);
                    if (p != null)
                        perceptions.Add(p);
                }
                if (entity.PerceptionCapabilities.Contains(PerceptionType.Semantic))
                {
                    var p = ProcessSemantic(entity, other, dist);
                    if (p != null)
                        perceptions.Add(p);
                }
                if (entity.PerceptionCapabilities.Contains(PerceptionType.Emotional))
                {
                    var p = ProcessEmotional(entity, other, dist);
                    if (p != null)
                        perceptions.Add(p);
                }
            }

            if (UseAttentionFiltering)
                perceptions = ApplyAttention(entity, perceptions);
            lock (_lock)
            { _entityPerceptions[entity.EntityId] = perceptions; }
            foreach (var p in perceptions)
                entity.AddPerception(p);
        }

        public IReadOnlyList<PerceptionEvent> GetPerceptions(SentientEntity entity)
        {
            lock (_lock)
                return _entityPerceptions.TryGetValue(entity.EntityId, out var p) ? p.AsReadOnly() : Array.Empty<PerceptionEvent>();
        }

        public IReadOnlyList<PerceptionEvent> FilterPerceptions(SentientEntity entity, PerceptionFilter filters)
        {
            return ApplyFilter(GetPerceptions(entity), filters);
        }

        private PerceptionEvent? ProcessVisual(SentientEntity obs, SentientEntity tgt, float dist, WorldStateData ws)
        {
            if (!obs.CanSee(tgt))
                return null;
            float visFactor = ws.Weather?.Visibility ?? 1f;
            float distFactor = Math.Max(0, 1f - dist / VisualRange);
            float conf = distFactor * visFactor;
            if (dist > VisualRange * 0.8f)
                conf *= 0.5f;
            if (conf < PerceptionThreshold)
                return null;
            var dir = Vector3.Normalize(tgt.Position - obs.Position);
            return new PerceptionEvent(PerceptionType.Visual, tgt.EntityId, ws.Time, conf * visFactor, tgt.Position, dir, $"Visual contact with {tgt.EntityType}", conf);
        }

        private PerceptionEvent? ProcessAuditory(SentientEntity obs, SentientEntity tgt, float dist, WorldStateData ws)
        {
            float intensity = 1f / (1f + dist * dist / (AuditoryRange * AuditoryRange));
            if (SimulateOcclusion)
                intensity *= 0.8f;
            if (ws.Weather?.WindSpeed > 0)
            {
                var wd = new Vector3((float)Math.Cos(ws.Weather.WindDirection), 0, (float)Math.Sin(ws.Weather.WindDirection));
                float we = Vector3.Dot(wd, Vector3.Normalize(tgt.Position - obs.Position));
                intensity *= 1f + we * 0.2f;
            }
            if (intensity < PerceptionThreshold)
                return null;
            var dir = Vector3.Normalize(tgt.Position - obs.Position);
            return new PerceptionEvent(PerceptionType.Auditory, tgt.EntityId, ws.Time, intensity, tgt.Position, dir, $"Auditory from {tgt.EntityType}", intensity * 0.8f);
        }

        private PerceptionEvent? ProcessProximity(SentientEntity obs, SentientEntity tgt, float dist)
        {
            float intensity = Math.Max(0, 1f - dist / ProximityRange);
            if (intensity < PerceptionThreshold)
                return null;
            var dir = Vector3.Normalize(tgt.Position - obs.Position);
            return new PerceptionEvent(PerceptionType.Proximity, tgt.EntityId, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, intensity, tgt.Position, dir, $"Proximity: {dist:F1} units", 1f);
        }

        private PerceptionEvent? ProcessSemantic(SentientEntity obs, SentientEntity tgt, float dist)
        {
            float conf = 0.5f;
            if (obs.Relationships.TryGetValue(tgt.EntityId, out var rel))
                conf = Math.Min(1f, conf + 0.3f);
            var dir = Vector3.Normalize(tgt.Position - obs.Position);
            return new PerceptionEvent(PerceptionType.Semantic, tgt.EntityId, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, conf, tgt.Position, dir, $"Semantic: {tgt.EntityType} ({rel?.Type.ToString() ?? "Unknown"})", conf);
        }

        private PerceptionEvent? ProcessEmotional(SentientEntity obs, SentientEntity tgt, float dist)
        {
            float distFactor = Math.Max(0, 1f - dist / (obs.PerceptionRadius * 0.5f));
            if (distFactor < PerceptionThreshold)
                return null;
            var dir = Vector3.Normalize(tgt.Position - obs.Position);
            return new PerceptionEvent(PerceptionType.Emotional, tgt.EntityId, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, distFactor * 0.6f, tgt.Position, dir, $"Emotional: {tgt.CurrentEmotion}", distFactor * 0.5f);
        }

        private List<PerceptionEvent> ApplyAttention(SentientEntity entity, List<PerceptionEvent> perceptions)
        {
            if (!_attentionModels.TryGetValue(entity.EntityId, out var am))
            { am = new AttentionModel(); _attentionModels[entity.EntityId] = am; }
            return am.FilterPerceptions(perceptions);
        }

        private static List<PerceptionEvent> ApplyFilter(IReadOnlyList<PerceptionEvent> perceptions, PerceptionFilter filter)
        {
            return perceptions.Where(p =>
                (filter.AllowedTypes.Count == 0 || filter.AllowedTypes.Contains(p.Type)) &&
                p.Intensity >= filter.MinIntensity && p.Confidence >= filter.MinConfidence &&
                (filter.AllowedSources.Count == 0 || filter.AllowedSources.Contains(p.Source)) &&
                !filter.ExcludedSources.Contains(p.Source)).ToList();
        }

        public void UpdateSpatialIndex(IEnumerable<SentientEntity> entities) { _spatialIndex.Clear(); foreach (var e in entities) _spatialIndex.Insert(e.EntityId, e.Position); }
        public void RemoveEntity(Guid eid) { lock (_lock) { _entityPerceptions.Remove(eid); _attentionModels.Remove(eid); } }
    }

    public class AttentionModel
    {
        private readonly Dictionary<PerceptionType, float> _weights = new()
        {
            { PerceptionType.Visual, 1f }, { PerceptionType.Auditory, 0.8f }, { PerceptionType.Tactile, 0.6f },
            { PerceptionType.Proximity, 0.9f }, { PerceptionType.Semantic, 0.7f }, { PerceptionType.Emotional, 0.5f }
        };
        private readonly Dictionary<Guid, float> _focus = new();
        private float _arousal;
        private float _radius = 50f;

        public float ArousalLevel { get => _arousal; set => _arousal = Math.Clamp(value, 0, 1); }
        public float FocusRadius { get => _radius; set => _radius = Math.Max(1, value); }

        public List<PerceptionEvent> FilterPerceptions(List<PerceptionEvent> perceptions)
        {
            var filtered = new List<PerceptionEvent>();
            foreach (var p in perceptions)
            {
                float score = p.Intensity * p.Confidence;
                if (_arousal > 0.7f)
                    score *= 1f + (_arousal - 0.7f) * 2f;
                if (_weights.TryGetValue(p.Type, out var w))
                    score *= w;
                if (_focus.TryGetValue(p.Source, out var f))
                    score *= 1f + f;
                if (score >= 0.3f)
                    filtered.Add(p);
            }
            filtered.Sort((a, b) => (b.Intensity * b.Confidence).CompareTo(a.Intensity * a.Confidence));
            return filtered;
        }

        public void FocusOn(Guid eid, float intensity = 1f) { _focus[eid] = Math.Clamp(intensity, 0, 1); }
        public void Unfocus(Guid eid) { _focus.Remove(eid); }

        public void Update(float deltaTime)
        {
            _arousal = Math.Max(0, _arousal - deltaTime * 0.1f);
            foreach (var k in _focus.Keys.ToList())
            { _focus[k] -= deltaTime * 0.05f; if (_focus[k] <= 0) _focus.Remove(k); }
        }
    }

    #endregion
    #region EMOTIONAL MODEL

    public class EmotionalModel : IEmotionalModel
    {
        private readonly Dictionary<Guid, EmotionalStateData> _entityEmotions = new();
        private readonly EmotionWheel _wheel;
        private readonly EmpathyModel _empathy;

        private static readonly Dictionary<EmotionalState, EmotionalState> Opposites = new()
        {
            { EmotionalState.Happy, EmotionalState.Sad }, { EmotionalState.Angry, EmotionalState.Fearful },
            { EmotionalState.Surprised, EmotionalState.Anticipating }, { EmotionalState.Disgusted, EmotionalState.Trusting },
            { EmotionalState.Calm, EmotionalState.Excited }, { EmotionalState.Bored, EmotionalState.Curious },
            { EmotionalState.Confused, EmotionalState.Determined }, { EmotionalState.Anxious, EmotionalState.Calm }
        };

        public EmotionalModel() { _wheel = new EmotionWheel(); _empathy = new EmpathyModel(); }
        public float DefaultInertia { get; set; } = 0.7f;
        public float DefaultDecayRate { get; set; } = 0.1f;

        public void Update(SentientEntity entity, IReadOnlyList<PerceptionEvent> events, float deltaTime)
        {
            var sd = GetOrCreate(entity.EntityId);
            foreach (var e in events)
            { var r = ReactToEvent(entity, e); ApplyResponse(sd, r, entity.EmotionalInertia); }
            _empathy.Update(entity, sd, deltaTime);
            ApplyDecay(sd, deltaTime, entity.EmotionalDecayRate);
            sd.Inertia = entity.EmotionalInertia;
            sd.UpdateDominantEmotion();
        }

        public EmotionalState GetCurrentState(SentientEntity entity) => GetOrCreate(entity.EntityId).DominantEmotion;

        public EmotionalResponse ReactToEvent(SentientEntity entity, PerceptionEvent perception)
        {
            var sd = GetOrCreate(entity.EntityId);
            var (emo, intensity) = DetermineResponse(entity, perception);
            float eff = intensity * (1f - sd.Inertia * 0.5f);
            if (sd.EmotionIntensities.TryGetValue(emo, out var ci) && Opposites.TryGetValue(emo, out var opp) && sd.EmotionIntensities.TryGetValue(opp, out var oi))
                eff *= Math.Max(0, 1f - oi * 0.5f);
            eff = Math.Clamp(eff, 0, 1);
            return new EmotionalResponse(emo, eff, 2f * (1f + eff * 3f), $"Response to {perception.Type}", DefaultDecayRate);
        }

        public IReadOnlyDictionary<EmotionalState, float> GetEmotionalProfile(SentientEntity entity)
        {
            return GetOrCreate(entity.EntityId).EmotionIntensities.ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        private EmotionalStateData GetOrCreate(Guid eid)
        {
            if (!_entityEmotions.TryGetValue(eid, out var sd))
            { sd = new EmotionalStateData(); _entityEmotions[eid] = sd; }
            return sd;
        }

        private (EmotionalState, float) DetermineResponse(SentientEntity entity, PerceptionEvent p)
        {
            float baseInt = p.Intensity * p.Confidence;
            return p.Type switch
            {
                PerceptionType.Visual when entity.Relationships.TryGetValue(p.Source, out var r) => r.Type switch
                {
                    RelationshipType.Hostile => (EmotionalState.Angry, baseInt * 0.8f),
                    RelationshipType.Friendly => (EmotionalState.Happy, baseInt * 0.6f),
                    RelationshipType.Parent => (EmotionalState.Trusting, baseInt * 0.5f),
                    RelationshipType.Child => (EmotionalState.Happy, baseInt * 0.7f),
                    _ => (EmotionalState.Surprised, baseInt * 0.4f)
                },
                PerceptionType.Visual => (EmotionalState.Curious, baseInt * entity.PersonalityTraits.GetValueOrDefault("Openness", 0.5f)),
                PerceptionType.Auditory when p.SemanticContent.Contains("threat") => (EmotionalState.Fearful, baseInt * 0.7f),
                PerceptionType.Auditory when p.SemanticContent.Contains("help") => (EmotionalState.Trusting, baseInt * 0.5f),
                PerceptionType.Auditory => (EmotionalState.Surprised, baseInt * 0.3f),
                PerceptionType.Tactile when p.Intensity > 0.7f => (EmotionalState.Angry, baseInt * 0.8f),
                PerceptionType.Tactile => (EmotionalState.Surprised, baseInt * 0.5f),
                PerceptionType.Proximity when entity.Relationships.TryGetValue(p.Source, out var r2) => r2.Type switch
                {
                    RelationshipType.Hostile => (EmotionalState.Anxious, baseInt * 0.6f),
                    RelationshipType.Friendly => (EmotionalState.Happy, baseInt * 0.4f),
                    _ => (EmotionalState.Curious, baseInt * 0.3f)
                },
                PerceptionType.Proximity => (EmotionalState.Curious, baseInt * 0.4f),
                PerceptionType.Semantic => (EmotionalState.Curious, baseInt * 0.5f),
                PerceptionType.Emotional when entity.Relationships.ContainsKey(p.Source) => (EmotionalState.Curious, baseInt * 0.3f),
                _ => (EmotionalState.Neutral, 0)
            };
        }

        private static void ApplyResponse(EmotionalStateData sd, EmotionalResponse r, float inertia)
        {
            float eff = r.Intensity * (1f - inertia);
            if (sd.EmotionIntensities.TryGetValue(r.Emotion, out var ci))
                sd.EmotionIntensities[r.Emotion] = Math.Min(1f, ci + eff * (1f - ci));
            else
                sd.EmotionIntensities[r.Emotion] = eff;

            if (Opposites.TryGetValue(r.Emotion, out var opp) && sd.EmotionIntensities.TryGetValue(opp, out var oi))
                sd.EmotionIntensities[opp] = Math.Max(0, oi - eff * 0.3f);
        }

        private static void ApplyDecay(EmotionalStateData sd, float dt, float rate)
        {
            foreach (var k in sd.EmotionIntensities.Keys.ToList())
            {
                sd.EmotionIntensities[k] = Math.Max(0, sd.EmotionIntensities[k] - rate * dt);
                if (sd.EmotionIntensities[k] <= 0.001f)
                    sd.EmotionIntensities.Remove(k);
            }
        }

        public EmotionalStateData? GetStateData(Guid eid) => _entityEmotions.TryGetValue(eid, out var d) ? d : null;
        public void RemoveEntity(Guid eid) => _entityEmotions.Remove(eid);
    }

    public class EmotionWheel
    {
        private readonly Dictionary<EmotionalState, float> _angles = new()
        {
            { EmotionalState.Happy, 0 }, { EmotionalState.Trusting, 45 }, { EmotionalState.Fearful, 90 },
            { EmotionalState.Surprised, 135 }, { EmotionalState.Sad, 180 }, { EmotionalState.Disgusted, 225 },
            { EmotionalState.Angry, 270 }, { EmotionalState.Anticipating, 315 }
        };

        public float CalculateDistance(EmotionalState a, EmotionalState b)
        {
            if (!_angles.TryGetValue(a, out float aa) || !_angles.TryGetValue(b, out float ab))
                return 180;
            float d = Math.Abs(aa - ab);
            return Math.Min(d, 360 - d);
        }

        public EmotionalState? Blend(EmotionalState a, EmotionalState b, float factor)
        {
            if (a == b)
                return a;
            float dist = CalculateDistance(a, b);
            if (dist <= 90)
            {
                float mid = ((_angles[a] + _angles[b]) / 2) % 360;
                return FindClosest(mid);
            }
            return null;
        }

        private EmotionalState FindClosest(float angle)
        {
            EmotionalState best = EmotionalState.Neutral;
            float minD = float.MaxValue;
            foreach (var (e, a) in _angles)
            {
                float d = Math.Min(Math.Abs(angle - a), 360 - Math.Abs(angle - a));
                if (d < minD)
                { minD = d; best = e; }
            }
            return best;
        }
    }

    public class EmpathyModel
    {
        public float EmpathyStrength { get; set; } = 0.3f;
        public float EmpathyRange { get; set; } = 30f;

        public void Update(SentientEntity entity, EmotionalStateData stateData, float deltaTime)
        {
            float drift = 0.01f * deltaTime;
            foreach (var e in stateData.EmotionIntensities.Keys.ToList())
            {
                if (e != stateData.BaselineEmotion)
                    stateData.EmotionIntensities[e] = Math.Max(0, stateData.EmotionIntensities[e] - drift);
            }
        }

        public float CalculateInfluence(SentientEntity source, SentientEntity target, float distance)
        {
            float df = Math.Max(0, 1f - distance / EmpathyRange);
            float rf = 0.1f;
            if (source.Relationships.TryGetValue(target.EntityId, out var r))
                rf = r.Strength;
            return df * rf * EmpathyStrength;
        }
    }

    #endregion
    #region MEMORY SYSTEM

    public class MemorySystem : IMemorySystem
    {
        private readonly Dictionary<Guid, EntityMemoryBank> _entityMemories = new();

        public int ShortTermCapacity { get; set; } = 7;
        public float ShortTermDuration { get; set; } = 30f;
        public float ConsolidationThreshold { get; set; } = 0.3f;
        public float DecayRate { get; set; } = 0.01f;

        private EntityMemoryBank GetOrCreate(Guid eid)
        {
            if (!_entityMemories.TryGetValue(eid, out var b))
            { b = new EntityMemoryBank(); _entityMemories[eid] = b; }
            return b;
        }

        public void Store(SentientEntity entity, MemoryEntry memory)
        {
            var bank = GetOrCreate(entity.EntityId);
            bank.ShortTerm.Add(memory);
            if (bank.ShortTerm.Count > ShortTermCapacity)
            {
                var oldest = bank.ShortTerm.OrderBy(m => m.Timestamp).First();
                if (oldest.Importance >= ConsolidationThreshold)
                    ConsolidateMemory(bank, oldest);
                else
                    bank.ShortTerm.Remove(oldest);
            }
            bank.Episodic.Add(memory);
            if (memory.Type == MemoryType.Semantic)
                UpdateSemantic(bank, memory);
            if (memory.Type == MemoryType.Procedural)
                UpdateProcedural(bank, memory);
        }

        public IReadOnlyList<MemoryEntry> Retrieve(SentientEntity entity, MemoryQuery query)
        {
            var bank = GetOrCreate(entity.EntityId);
            var results = new List<MemoryEntry>();
            results.AddRange(Search(bank.ShortTerm, query));
            results.AddRange(Search(bank.LongTerm, query));
            results.AddRange(Search(bank.Episodic, query));
            results.AddRange(Search(bank.Semantic, query));
            results.AddRange(Search(bank.Procedural, query));
            results = results.DistinctBy(m => m.Id).ToList();
            if (query.SortByEmotionalIntensity)
                results = results.OrderByDescending(m => m.EmotionalIntensity).ToList();
            else if (query.SortByImportance)
                results = results.OrderByDescending(m => m.Importance).ToList();
            else if (query.SortByRecency)
                results = results.OrderByDescending(m => m.Timestamp).ToList();
            if (results.Count > query.MaxResults)
                results = results.Take(query.MaxResults).ToList();
            foreach (var m in results)
            { m.RetrievalCount++; m.LastRetrieved = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond; }
            return results.AsReadOnly();
        }

        public int Forget(SentientEntity entity, Func<MemoryEntry, bool> criteria)
        {
            var bank = GetOrCreate(entity.EntityId);
            int count = 0;
            count += bank.ShortTerm.RemoveAll(m => criteria(m));
            count += bank.LongTerm.RemoveAll(m => criteria(m));
            count += bank.Episodic.RemoveAll(m => criteria(m));
            count += bank.Semantic.RemoveAll(m => criteria(m));
            count += bank.Procedural.RemoveAll(m => criteria(m));
            return count;
        }

        public void Consolidate(SentientEntity entity)
        {
            var bank = GetOrCreate(entity.EntityId);
            var toCon = bank.ShortTerm.Where(m => m.Importance >= ConsolidationThreshold || m.EmotionalIntensity > 0.7f || m.RetrievalCount > 3).ToList();
            foreach (var m in toCon)
                ConsolidateMemory(bank, m);
            var rehearsed = bank.Episodic.Where(m => m.RetrievalCount > 2 && !m.IsConsolidated).ToList();
            foreach (var m in rehearsed)
            {
                m.ConsolidationStrength += 0.1f * m.RetrievalCount;
                if (m.ConsolidationStrength >= 1f)
                { m.IsConsolidated = true; bank.LongTerm.Add(m); }
            }
            ApplyDecay(bank);
        }

        public IReadOnlyDictionary<MemoryType, int> GetMemoryCounts(SentientEntity entity)
        {
            var bank = GetOrCreate(entity.EntityId);
            return new Dictionary<MemoryType, int>
            {
                { MemoryType.ShortTerm, bank.ShortTerm.Count }, { MemoryType.LongTerm, bank.LongTerm.Count },
                { MemoryType.Episodic, bank.Episodic.Count }, { MemoryType.Semantic, bank.Semantic.Count },
                { MemoryType.Procedural, bank.Procedural.Count }
            };
        }

        private void ConsolidateMemory(EntityMemoryBank bank, MemoryEntry m)
        {
            if (!m.IsConsolidated)
            { m.IsConsolidated = true; m.ConsolidationStrength += 0.2f; bank.LongTerm.Add(m); }
            bank.ShortTerm.Remove(m);
        }

        private static List<MemoryEntry> Search(List<MemoryEntry> memories, MemoryQuery q)
        {
            return memories.Where(m =>
                (string.IsNullOrEmpty(q.SearchTerms) || m.Content.Contains(q.SearchTerms, StringComparison.OrdinalIgnoreCase)) &&
                (!q.Type.HasValue || m.Type == q.Type.Value) &&
                m.EmotionalIntensity >= q.MinEmotionalIntensity &&
                m.Importance >= q.MinImportance &&
                (Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond - m.Timestamp) <= q.MaxAge &&
                (!q.AssociatedEntity.HasValue || m.AssociatedEntities.Contains(q.AssociatedEntity.Value)) &&
                (!q.LocationFilter.HasValue || Vector3.Distance(m.Location, q.LocationFilter.Value.Center) <= q.LocationFilter.Value.Radius) &&
                (q.Tags.Count == 0 || q.Tags.Any(t => m.Tags.Contains(t)))
            ).ToList();
        }

        private void UpdateSemantic(EntityMemoryBank bank, MemoryEntry m)
        {
            var existing = bank.Semantic.FirstOrDefault(s => s.Content == m.Content);
            if (existing != null)
            { existing.RetrievalCount++; existing.Importance = Math.Min(1f, existing.Importance + 0.1f); }
            else
                bank.Semantic.Add(m);
        }

        private void UpdateProcedural(EntityMemoryBank bank, MemoryEntry m)
        {
            var existing = bank.Procedural.FirstOrDefault(p => p.Content == m.Content);
            if (existing != null)
            { existing.RetrievalCount++; existing.ConsolidationStrength = Math.Min(1f, existing.ConsolidationStrength + 0.1f); }
            else
                bank.Procedural.Add(m);
        }

        private void ApplyDecay(EntityMemoryBank bank)
        {
            var now = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
            foreach (var list in new[] { bank.ShortTerm, bank.LongTerm, bank.Episodic, bank.Semantic, bank.Procedural })
            {
                foreach (var m in list.Where(m => m.DecayFactor > 0).ToList())
                {
                    float age = (float)(now - m.Timestamp);
                    m.DecayFactor = Math.Max(0, m.DecayFactor - age * 0.0001f * m.DecayFactor);
                    if (m.DecayFactor <= 0.01f && m.Type != MemoryType.Procedural)
                        list.Remove(m);
                }
            }
        }
    }

    public class EntityMemoryBank
    {
        public List<MemoryEntry> ShortTerm { get; set; } = new();
        public List<MemoryEntry> LongTerm { get; set; } = new();
        public List<MemoryEntry> Episodic { get; set; } = new();
        public List<MemoryEntry> Semantic { get; set; } = new();
        public List<MemoryEntry> Procedural { get; set; } = new();
    }

    #endregion
    #region RELATIONSHIP MANAGER

    public class RelationshipManager
    {
        private readonly Dictionary<(Guid, Guid), Relationship> _relationships = new();
        private readonly Dictionary<string, HashSet<Guid>> _groups = new();
        private readonly object _lock = new();

        public void AddRelationship(Relationship relationship)
        {
            lock (_lock)
            {
                var key = GetKey(relationship.EntityA, relationship.EntityB);
                _relationships[key] = relationship;
            }
        }

        public void RemoveRelationship(Guid entityA, Guid entityB)
        {
            lock (_lock)
            { _relationships.Remove(GetKey(entityA, entityB)); }
        }

        public Relationship? GetRelationship(Guid entityA, Guid entityB)
        {
            lock (_lock)
            { return _relationships.TryGetValue(GetKey(entityA, entityB), out var r) ? r : null; }
        }

        public List<Relationship> GetRelationshipsFor(Guid entityId)
        {
            lock (_lock)
            { return _relationships.Values.Where(r => r.EntityA == entityId || r.EntityB == entityId).ToList(); }
        }

        public List<Relationship> GetRelationshipsOfType(RelationshipType type)
        {
            lock (_lock)
            { return _relationships.Values.Where(r => r.Type == type).ToList(); }
        }

        public void UpdateRelationshipStrength(Guid entityA, Guid entityB, float newStrength, string reason = "")
        {
            lock (_lock)
            {
                if (_relationships.TryGetValue(GetKey(entityA, entityB), out var r))
                {
                    var prev = r.Type;
                    r.History.Add(new RelationshipEvent(prev, r.Type, r.Strength, newStrength, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, reason));
                    r.Strength = Math.Clamp(newStrength, 0, 1);
                    if (r.Strength <= 0.01f)
                        r.Type = RelationshipType.Neutral;
                }
            }
        }

        public void ChangeRelationshipType(Guid entityA, Guid entityB, RelationshipType newType, string reason = "")
        {
            lock (_lock)
            {
                if (_relationships.TryGetValue(GetKey(entityA, entityB), out var r))
                {
                    r.History.Add(new RelationshipEvent(r.Type, newType, r.Strength, r.Strength, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, reason));
                    r.Type = newType;
                }
            }
        }

        public float CalculateInfluencePropagation(Guid source, Guid target, int maxDepth = 3)
        {
            lock (_lock)
            {
                var visited = new HashSet<Guid>();
                return PropagateHelper(source, target, maxDepth, visited);
            }
        }

        private float PropagateHelper(Guid current, Guid target, int depth, HashSet<Guid> visited)
        {
            if (depth <= 0 || !visited.Add(current))
                return 0;
            if (current == target)
                return 1f;

            float maxInfluence = 0;
            foreach (var r in _relationships.Values.Where(r => r.EntityA == current || r.EntityB == current))
            {
                var next = r.EntityA == current ? r.EntityB : r.EntityA;
                float childInfluence = PropagateHelper(next, target, depth - 1, visited) * r.Strength;
                maxInfluence = Math.Max(maxInfluence, childInfluence);
            }
            return maxInfluence;
        }

        public void AddToGroup(Guid entityId, string groupName)
        {
            lock (_lock)
            {
                if (!_groups.TryGetValue(groupName, out var members))
                { members = new HashSet<Guid>(); _groups[groupName] = members; }
                members.Add(entityId);
            }
        }

        public void RemoveFromGroup(Guid entityId, string groupName)
        {
            lock (_lock)
            { if (_groups.TryGetValue(groupName, out var m)) m.Remove(entityId); }
        }

        public List<Guid> GetGroupMembers(string groupName)
        {
            lock (_lock)
            { return _groups.TryGetValue(groupName, out var m) ? m.ToList() : new List<Guid>(); }
        }

        public List<string> GetGroupsFor(Guid entityId)
        {
            lock (_lock)
            { return _groups.Where(kv => kv.Value.Contains(entityId)).Select(kv => kv.Key).ToList(); }
        }

        public bool AreInSameGroup(Guid entityA, Guid entityB)
        {
            lock (_lock)
            { return _groups.Values.Any(g => g.Contains(entityA) && g.Contains(entityB)); }
        }

        public Dictionary<string, float> AnalyzeSocialNetwork(Guid entityId)
        {
            lock (_lock)
            {
                var relationships = GetRelationshipsFor(entityId);
                var analysis = new Dictionary<string, float>
                {
                    { "ConnectionCount", relationships.Count },
                    { "AverageStrength", relationships.Count > 0 ? relationships.Average(r => r.Strength) : 0 },
                    { "StrongestConnection", relationships.Count > 0 ? relationships.Max(r => r.Strength) : 0 },
                    { "WeakestConnection", relationships.Count > 0 ? relationships.Min(r => r.Strength) : 0 },
                    { "GroupCount", GetGroupsFor(entityId).Count },
                    { "CentralityScore", CalculateCentrality(entityId) }
                };
                return analysis;
            }
        }

        private float CalculateCentrality(Guid entityId)
        {
            int totalPairs = _relationships.Count;
            if (totalPairs == 0)
                return 0;
            int involvingEntity = _relationships.Values.Count(r => r.EntityA == entityId || r.EntityB == entityId);
            return (float)involvingEntity / totalPairs;
        }

        public void EvolveRelationships(float deltaTime, float evolutionRate = 0.001f)
        {
            lock (_lock)
            {
                foreach (var r in _relationships.Values)
                {
                    float change = (Random.Shared.NextSingle() - 0.5f) * evolutionRate * deltaTime;
                    r.Strength = Math.Clamp(r.Strength + change, 0, 1);
                    if (r.Strength <= 0.01f)
                        r.Type = RelationshipType.Neutral;
                }
            }
        }

        private static (Guid, Guid) GetKey(Guid a, Guid b) => a < b ? (a, b) : (b, a);
    }

    #endregion
    #region SPATIAL REASONING SYSTEM

    public class SpatialReasoningSystem
    {
        private readonly SpatialIndex _spatialIndex;
        private readonly Dictionary<Guid, List<Vector3>> _pathCache = new();
        private readonly Dictionary<Guid, Territory> _territories = new();

        public SpatialReasoningSystem() { _spatialIndex = new SpatialIndex(15f); }

        public float CellSize { get; set; } = 15f;

        public List<SentientEntity> FindNearestEntities(SentientEntity entity, WorldStateData worldState, int count = 5)
        {
            var candidates = _spatialIndex.QueryRadius(entity.Position, entity.PerceptionRadius);
            return candidates
                .Where(id => id != entity.EntityId && worldState.Entities.ContainsKey(id))
                .Select(id => worldState.Entities[id])
                .OrderBy(e => Vector3.Distance(entity.Position, e.Position))
                .Take(count)
                .ToList();
        }

        public List<SentientEntity> FindEntitiesInRadius(SentientEntity entity, WorldStateData worldState, float radius)
        {
            var candidates = _spatialIndex.QueryRadius(entity.Position, radius);
            return candidates
                .Where(id => id != entity.EntityId && worldState.Entities.ContainsKey(id))
                .Select(id => worldState.Entities[id])
                .ToList();
        }

        public bool HasLineOfSight(Vector3 from, Vector3 to, List<Vector3> obstacles = null)
        {
            obstacles ??= new List<Vector3>();
            var dir = to - from;
            float dist = dir.Length();
            var norm = dir / dist;
            int steps = (int)(dist / 0.5f);
            for (int i = 0; i <= steps; i++)
            {
                var point = from + norm * (i * 0.5f);
                if (obstacles.Any(o => Vector3.Distance(point, o) < 0.5f))
                    return false;
            }
            return true;
        }

        public List<Vector3> FindPath(Vector3 start, Vector3 goal, List<Vector3> obstacles = null, int maxIterations = 1000)
        {
            obstacles ??= new List<Vector3>();
            var openSet = new SortedSet<(float F, Vector3 Pos)>(Comparer<(float, Vector3)>.Create((a, b) => a.Item1.CompareTo(b.Item1)));
            var cameFrom = new Dictionary<Vector3, Vector3>();
            var gScore = new Dictionary<Vector3, float> { [start] = 0 };
            var closedSet = new HashSet<Vector3>();
            var snapStart = SnapToGrid(start);
            var snapGoal = SnapToGrid(goal);

            openSet.Add((Heuristic(snapStart, snapGoal), snapStart));
            int iter = 0;

            while (openSet.Count > 0 && iter < maxIterations)
            {
                iter++;
                var current = openSet.Min;
                openSet.Remove(current);

                if (Vector3.Distance(current.Pos, snapGoal) < 1f)
                    return ReconstructPath(cameFrom, current.Pos);

                closedSet.Add(current.Pos);
                float currentG = gScore.GetValueOrDefault(current.Pos, float.MaxValue);

                foreach (var neighbor in GetNeighbors(current.Pos))
                {
                    if (closedSet.Contains(neighbor))
                        continue;
                    if (obstacles.Any(o => Vector3.Distance(neighbor, o) < 1f))
                        continue;

                    float tentativeG = currentG + Vector3.Distance(current.Pos, neighbor);
                    if (tentativeG < gScore.GetValueOrDefault(neighbor, float.MaxValue))
                    {
                        cameFrom[neighbor] = current.Pos;
                        gScore[neighbor] = tentativeG;
                        float f = tentativeG + Heuristic(neighbor, snapGoal);
                        openSet.Add((f, neighbor));
                    }
                }
            }
            return new List<Vector3> { start, goal };
        }

        public Vector3 CalculateAvoidance(Vector3 position, List<SentientEntity> nearbyEntities, float avoidRadius = 3f)
        {
            var avoidance = Vector3.Zero;
            foreach (var e in nearbyEntities)
            {
                float dist = Vector3.Distance(position, e.Position);
                if (dist < avoidRadius && dist > 0.01f)
                {
                    var dir = Vector3.Normalize(position - e.Position);
                    float strength = (avoidRadius - dist) / avoidRadius;
                    avoidance += dir * strength;
                }
            }
            return avoidance;
        }

        public Vector3 CalculateFormationPosition(Vector3 leaderPos, int index, int total, float spacing = 2f, FormationType formation = FormationType.Circle)
        {
            switch (formation)
            {
                case FormationType.Line:
                    return leaderPos + new Vector3(index * spacing, 0, 0);
                case FormationType.Circle:
                    {
                        float angle = (2 * MathF.PI * index) / Math.Max(1, total);
                        return leaderPos + new Vector3(MathF.Cos(angle) * spacing, 0, MathF.Sin(angle) * spacing);
                    }
                case FormationType.VShape:
                    {
                        int side = index % 2 == 0 ? 1 : -1;
                        int row = (index / 2) + 1;
                        return leaderPos + new Vector3(side * row * spacing * 0.5f, 0, -row * spacing);
                    }
                case FormationType.Grid:
                    {
                        int cols = (int)Math.Ceiling(Math.Sqrt(total));
                        int row = index / cols;
                        int col = index % cols;
                        return leaderPos + new Vector3((col - cols / 2f) * spacing, 0, row * spacing);
                    }
                default:
                    return leaderPos;
            }
        }

        public void RegisterTerritory(Guid ownerId, Vector3 center, float radius)
        {
            _territories[ownerId] = new Territory { OwnerId = ownerId, Center = center, Radius = radius };
        }

        public bool IsInTerritory(Vector3 position, out Guid? ownerId)
        {
            foreach (var t in _territories.Values)
            {
                if (Vector3.Distance(position, t.Center) <= t.Radius)
                { ownerId = t.OwnerId; return true; }
            }
            ownerId = null;
            return false;
        }

        public List<Vector3> FindPatrolPoints(Vector3 center, float radius, int count = 4)
        {
            var points = new List<Vector3>();
            for (int i = 0; i < count; i++)
            {
                float angle = (2 * MathF.PI * i) / count;
                points.Add(center + new Vector3(MathF.Cos(angle) * radius, 0, MathF.Sin(angle) * radius));
            }
            return points;
        }

        public Vector3 PredictPosition(SentientEntity entity, float timeAhead)
        {
            return entity.Position + entity.Velocity * timeAhead;
        }

        public Vector3 InterceptPosition(SentientEntity pursuer, SentientEntity target, float pursuerSpeed)
        {
            var toTarget = target.Position - pursuer.Position;
            float dist = toTarget.Length();
            float targetSpeed = target.Velocity.Length();
            if (targetSpeed < 0.01f)
                return target.Position;
            float t = dist / Math.Max(0.01f, pursuerSpeed + targetSpeed);
            return target.Position + target.Velocity * t;
        }

        public void UpdateSpatialIndex(IEnumerable<SentientEntity> entities) { _spatialIndex.Clear(); foreach (var e in entities) _spatialIndex.Insert(e.EntityId, e.Position); }

        private Vector3 SnapToGrid(Vector3 p) => new((float)Math.Round(p.X), (float)Math.Round(p.Y), (float)Math.Round(p.Z));

        private float Heuristic(Vector3 a, Vector3 b) => Vector3.Distance(a, b);

        private List<Vector3> GetNeighbors(Vector3 pos)
        {
            var neighbors = new List<Vector3>();
            for (int x = -1; x <= 1; x++)
                for (int y = -1; y <= 1; y++)
                    for (int z = -1; z <= 1; z++)
                        if (x != 0 || y != 0 || z != 0)
                            neighbors.Add(pos + new Vector3(x, y, z));
            return neighbors;
        }

        private List<Vector3> ReconstructPath(Dictionary<Vector3, Vector3> cameFrom, Vector3 current)
        {
            var path = new List<Vector3> { current };
            while (cameFrom.TryGetValue(current, out var prev))
            { current = prev; path.Insert(0, current); }
            return path;
        }
    }

    public enum FormationType { Line, Circle, VShape, Grid }

    public class Territory
    {
        public Guid OwnerId { get; set; }
        public Vector3 Center { get; set; }
        public float Radius { get; set; }
    }

    #endregion
    #region GROUP BEHAVIOR SYSTEM

    public class GroupBehaviorSystem
    {
        private readonly Dictionary<string, GroupData> _groups = new();
        private readonly Dictionary<Guid, string> _entityGroups = new();

        public float SeparationRadius { get; set; } = 2f;
        public float AlignmentRadius { get; set; } = 5f;
        public float CohesionRadius { get; set; } = 8f;
        public float SeparationWeight { get; set; } = 1.5f;
        public float AlignmentWeight { get; set; } = 1f;
        public float CohesionWeight { get; set; } = 1f;

        public void RegisterEntity(Guid entityId, string groupName)
        {
            _entityGroups[entityId] = groupName;
            if (!_groups.TryGetValue(groupName, out var g))
            { g = new GroupData { Name = groupName }; _groups[groupName] = g; }
            g.Members.Add(entityId);
        }

        public void UnregisterEntity(Guid entityId)
        {
            if (_entityGroups.TryGetValue(entityId, out var gName) && _groups.TryGetValue(gName, out var g))
                g.Members.Remove(entityId);
            _entityGroups.Remove(entityId);
        }

        public Vector3 CalculateFlocking(SentientEntity entity, Dictionary<Guid, SentientEntity> allEntities)
        {
            if (!_entityGroups.TryGetValue(entity.EntityId, out var groupName) || !_groups.TryGetValue(groupName, out var group))
                return Vector3.Zero;

            var separation = CalculateSeparation(entity, group, allEntities);
            var alignment = CalculateAlignment(entity, group, allEntities);
            var cohesion = CalculateCohesion(entity, group, allEntities);

            return separation * SeparationWeight + alignment * AlignmentWeight + cohesion * CohesionWeight;
        }

        private Vector3 CalculateSeparation(SentientEntity entity, GroupData group, Dictionary<Guid, SentientEntity> allEntities)
        {
            var force = Vector3.Zero;
            int count = 0;
            foreach (var mid in group.Members)
            {
                if (mid == entity.EntityId || !allEntities.TryGetValue(mid, out var other))
                    continue;
                float dist = entity.DistanceTo(other);
                if (dist < SeparationRadius && dist > 0.01f)
                {
                    force += Vector3.Normalize(entity.Position - other.Position) / dist;
                    count++;
                }
            }
            return count > 0 ? force / count : Vector3.Zero;
        }

        private Vector3 CalculateAlignment(SentientEntity entity, GroupData group, Dictionary<Guid, SentientEntity> allEntities)
        {
            var avgVel = Vector3.Zero;
            int count = 0;
            foreach (var mid in group.Members)
            {
                if (mid == entity.EntityId || !allEntities.TryGetValue(mid, out var other))
                    continue;
                if (entity.DistanceTo(other) < AlignmentRadius)
                { avgVel += other.Velocity; count++; }
            }
            return count > 0 ? Vector3.Normalize(avgVel / count - entity.Velocity) : Vector3.Zero;
        }

        private Vector3 CalculateCohesion(SentientEntity entity, GroupData group, Dictionary<Guid, SentientEntity> allEntities)
        {
            var center = Vector3.Zero;
            int count = 0;
            foreach (var mid in group.Members)
            {
                if (mid == entity.EntityId || !allEntities.TryGetValue(mid, out var other))
                    continue;
                if (entity.DistanceTo(other) < CohesionRadius)
                { center += other.Position; count++; }
            }
            if (count == 0)
                return Vector3.Zero;
            center /= count;
            return Vector3.Normalize(center - entity.Position);
        }

        public Guid? FindLeader(string groupName)
        {
            if (!_groups.TryGetValue(groupName, out var g))
                return null;
            return g.LeaderId;
        }

        public void SetLeader(string groupName, Guid entityId)
        {
            if (_groups.TryGetValue(groupName, out var g))
                g.LeaderId = entityId;
        }

        public Vector3 CalculateFollowFormation(SentientEntity follower, SentientEntity leader, int index, int total)
        {
            var spacing = 2.5f;
            var side = index % 2 == 0 ? 1 : -1;
            var row = (index / 2) + 1;
            return leader.Position + new Vector3(side * row * spacing * 0.5f, 0, -row * spacing);
        }

        public Dictionary<Guid, float> CalculateSwarmInfluence(string groupName, Vector3 target, Dictionary<Guid, SentientEntity> allEntities)
        {
            var influences = new Dictionary<Guid, float>();
            if (!_groups.TryGetValue(groupName, out var group))
                return influences;

            float totalDist = 0;
            var distances = new Dictionary<Guid, float>();
            foreach (var mid in group.Members)
            {
                if (!allEntities.TryGetValue(mid, out var e))
                    continue;
                float d = Vector3.Distance(e.Position, target);
                distances[mid] = d;
                totalDist += d;
            }

            if (totalDist > 0)
            {
                foreach (var (id, d) in distances)
                    influences[id] = 1f - (d / totalDist);
            }
            return influences;
        }

        public Dictionary<string, object> MakeGroupDecision(string groupName, List<(string Option, float Score)> options)
        {
            var results = new Dictionary<string, object>();
            if (!_groups.TryGetValue(groupName, out var group))
                return results;

            int totalVotes = group.Members.Count;
            var votes = new Dictionary<string, int>();

            foreach (var member in group.Members)
            {
                var personality = 0.5f;
                var scored = options.OrderByDescending(o => o.Score * personality).First();
                if (!votes.ContainsKey(scored.Option))
                    votes[scored.Option] = 0;
                votes[scored.Option]++;
            }

            var winner = votes.OrderByDescending(kv => kv.Value).First();
            results["Decision"] = winner.Key;
            results["Votes"] = winner.Value;
            results["TotalVoters"] = totalVotes;
            results["Confidence"] = (float)winner.Value / totalVotes;
            return results;
        }

        public void CommunicateWithinGroup(string groupName, Guid senderId, string message, Dictionary<Guid, SentientEntity> allEntities)
        {
            if (!_groups.TryGetValue(groupName, out var group))
                return;
            foreach (var mid in group.Members)
            {
                if (mid == senderId || !allEntities.TryGetValue(mid, out var receiver))
                    continue;
                receiver.SetProperty($"GroupMsg_{senderId}", message);
            }
        }

        public Dictionary<string, float> AnalyzeGroupCohesion(string groupName, Dictionary<Guid, SentientEntity> allEntities)
        {
            if (!_groups.TryGetValue(groupName, out var group) || group.Members.Count < 2)
                return new Dictionary<string, float> { { "Cohesion", 0 }, { "AvgDist", 0 }, { "SpeedMatch", 0 } };

            var members = group.Members.Where(id => allEntities.ContainsKey(id)).Select(id => allEntities[id]).ToList();
            var center = members.Aggregate(Vector3.Zero, (sum, e) => sum + e.Position) / members.Count;
            float avgDist = members.Average(e => Vector3.Distance(e.Position, center));
            var avgVel = members.Aggregate(Vector3.Zero, (sum, e) => sum + e.Velocity) / members.Count;
            float speedMatch = 1f - members.Average(e => Vector3.Distance(e.Velocity, avgVel)) / 10f;

            return new Dictionary<string, float>
            {
                { "Cohesion", Math.Clamp(1f - avgDist / CohesionRadius, 0, 1) },
                { "AvgDist", avgDist },
                { "SpeedMatch", Math.Clamp(speedMatch, 0, 1) },
                { "MemberCount", members.Count }
            };
        }

        public void TaskAllocation(string groupName, List<string> tasks, Dictionary<Guid, SentientEntity> allEntities)
        {
            if (!_groups.TryGetValue(groupName, out var group))
                return;
            var members = group.Members.Where(id => allEntities.ContainsKey(id)).Select(id => allEntities[id]).ToList();
            members = members.OrderBy(e => e.Needs.GetValueOrDefault("TaskLoad", 0)).ToList();

            for (int i = 0; i < Math.Min(tasks.Count, members.Count); i++)
            {
                members[i].SetProperty("AssignedTask", tasks[i]);
                members[i].SetProperty("TaskAssignedTime", Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond);
            }
        }
    }

    public class GroupData
    {
        public string Name { get; set; } = string.Empty;
        public HashSet<Guid> Members { get; set; } = new();
        public Guid? LeaderId { get; set; }
        public Dictionary<string, object> Properties { get; set; } = new();
    }

    #endregion
    #region BEHAVIOR DEBUGGER

    public class BehaviorDebugger
    {
        private readonly List<DebugLogEntry> _logEntries = new();
        private readonly HashSet<string> _breakpoints = new();
        private readonly Dictionary<string, TaskStatus> _nodeStates = new();
        private readonly List<DebugReplayFrame> _replayFrames = new();
        private bool _isStepping;
        private bool _isPaused;
        private int _stepCount;
        private readonly object _lock = new();

        public bool IsPaused { get => _isPaused; set => _isPaused = value; }
        public bool IsStepping { get => _isStepping; }
        public int LogEntryCount { get { lock (_lock) return _logEntries.Count; } }

        public void OnNodeTick(IBehaviorNode node, SentientEntity entity, TaskStatus status, float deltaTime)
        {
            lock (_lock)
            {
                _nodeStates[node.Id] = status;
                var entry = new DebugLogEntry
                {
                    Timestamp = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond,
                    NodeId = node.Id,
                    NodeName = node.Name,
                    NodeType = node.NodeType,
                    EntityId = entity.EntityId,
                    Status = status,
                    DeltaTime = deltaTime
                };
                _logEntries.Add(entry);

                if (_logEntries.Count > 10000)
                    _logEntries.RemoveRange(0, 1000);

                if (_breakpoints.Contains(node.Id))
                {
                    _isPaused = true;
                    _isStepping = false;
                }

                if (_isStepping)
                {
                    _stepCount--;
                    if (_stepCount <= 0)
                    { _isStepping = false; _isPaused = true; }
                }
            }
        }

        public void AddBreakpoint(string nodeId) { lock (_lock) _breakpoints.Add(nodeId); }
        public void RemoveBreakpoint(string nodeId) { lock (_lock) _breakpoints.Remove(nodeId); }
        public void ClearBreakpoints() { lock (_lock) _breakpoints.Clear(); }
        public bool HasBreakpoint(string nodeId) { lock (_lock) return _breakpoints.Contains(nodeId); }

        public void Step(int count = 1) { lock (_lock) { _isPaused = false; _isStepping = true; _stepCount = count; } }
        public void Continue() { lock (_lock) { _isPaused = false; _isStepping = false; } }
        public void Pause() { lock (_lock) _isPaused = true; }

        public void RecordReplayFrame(WorldStateData worldState, IReadOnlyDictionary<Guid, SentientEntity> entities)
        {
            lock (_lock)
            {
                var frame = new DebugReplayFrame
                {
                    Timestamp = worldState.Time,
                    EntityStates = entities.ToDictionary(kv => kv.Key, kv => new EntityStateSnapshot
                    {
                        Position = kv.Value.Position,
                        Velocity = kv.Value.Velocity,
                        Health = kv.Value.Health,
                        State = kv.Value.CurrentState,
                        Emotion = kv.Value.CurrentEmotion,
                        BehaviorTreeState = _nodeStates.ToDictionary(kv2 => kv2.Key, kv2 => kv2.Value)
                    })
                };
                _replayFrames.Add(frame);
                if (_replayFrames.Count > 5000)
                    _replayFrames.RemoveRange(0, 500);
            }
        }

        public DebugReplayFrame? GetReplayFrame(int index)
        {
            lock (_lock)
            { return index >= 0 && index < _replayFrames.Count ? _replayFrames[index] : null; }
        }

        public int ReplayFrameCount { get { lock (_lock) return _replayFrames.Count; } }

        public List<DebugLogEntry> GetLogEntries(int count = 100)
        {
            lock (_lock)
            { return _logEntries.TakeLast(count).ToList(); }
        }

        public List<DebugLogEntry> GetLogEntriesForNode(string nodeId, int count = 50)
        {
            lock (_lock)
            { return _logEntries.Where(e => e.NodeId == nodeId).TakeLast(count).ToList(); }
        }

        public List<DebugLogEntry> GetLogEntriesForEntity(Guid entityId, int count = 100)
        {
            lock (_lock)
            { return _logEntries.Where(e => e.EntityId == entityId).TakeLast(count).ToList(); }
        }

        public Dictionary<string, TaskStatus> GetBehaviorTreeState()
        {
            lock (_lock)
            { return new Dictionary<string, TaskStatus>(_nodeStates); }
        }

        public string VisualizeBehaviorTree(IBehaviorNode root, string indent = "", bool isLast = true)
        {
            var sb = new StringBuilder();
            string connector = isLast ? "└── " : "├── ";
            string statusStr = _nodeStates.TryGetValue(root.Id, out var s) ? $" [{s}]" : "";
            sb.AppendLine($"{indent}{connector}{root.Name} ({root.NodeType}){statusStr}");

            string childIndent = indent + (isLast ? "    " : "│   ");
            var children = root.GetChildren();
            for (int i = 0; i < children.Count; i++)
            {
                sb.Append(VisualizeBehaviorTree(children[i], childIndent, i == children.Count - 1));
            }
            return sb.ToString();
        }

        public Dictionary<string, object> GetStatistics()
        {
            lock (_lock)
            {
                var stats = new Dictionary<string, object>
                {
                    { "TotalLogEntries", _logEntries.Count },
                    { "ActiveBreakpoints", _breakpoints.Count },
                    { "ReplayFrames", _replayFrames.Count },
                    { "UniqueNodesTracked", _nodeStates.Count },
                    { "IsPaused", _isPaused },
                    { "IsStepping", _isStepping }
                };

                if (_logEntries.Count > 0)
                {
                    var statusCounts = _logEntries.GroupBy(e => e.Status)
                        .ToDictionary(g => g.Key.ToString(), g => g.Count());
                    stats["StatusDistribution"] = statusCounts;

                    var avgDt = _logEntries.Average(e => e.DeltaTime);
                    stats["AverageDeltaTime"] = avgDt;
                }
                return stats;
            }
        }

        public void Clear() { lock (_lock) { _logEntries.Clear(); _nodeStates.Clear(); _replayFrames.Clear(); } }
    }

    public class DebugLogEntry
    {
        public double Timestamp { get; set; }
        public string NodeId { get; set; } = string.Empty;
        public string NodeName { get; set; } = string.Empty;
        public BehaviorNodeType NodeType { get; set; }
        public Guid EntityId { get; set; }
        public TaskStatus Status { get; set; }
        public float DeltaTime { get; set; }
    }

    public class DebugReplayFrame
    {
        public double Timestamp { get; set; }
        public Dictionary<Guid, EntityStateSnapshot> EntityStates { get; set; } = new();
    }

    public class EntityStateSnapshot
    {
        public Vector3 Position { get; set; }
        public Vector3 Velocity { get; set; }
        public float Health { get; set; }
        public BehaviorState State { get; set; }
        public EmotionalState Emotion { get; set; }
        public Dictionary<string, TaskStatus> BehaviorTreeState { get; set; } = new();
    }

    #endregion
    #region BEHAVIOR COMPILER

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

    #endregion
    #region WORLD STATE

    public class WorldStateManager
    {
        private readonly Dictionary<Guid, SentientEntity> _entities = new();
        private readonly Queue<WorldEvent> _events = new();
        private readonly SpatialIndex _spatialIndex;
        private readonly object _lock = new();
        private double _currentTime;
        private WeatherConditions _weather;

        public WorldStateManager()
        {
            _spatialIndex = new SpatialIndex(15f);
            _weather = new WeatherConditions(20f, 0.5f, 5f, 0f, 1f, 1f, WeatherType.Clear);
        }

        public double CurrentTime { get => _currentTime; set => _currentTime = value; }
        public WeatherConditions Weather { get => _weather; set => _weather = value; }

        public void AddEntity(SentientEntity entity) { lock (_lock) { _entities[entity.EntityId] = entity; _spatialIndex.Insert(entity.EntityId, entity.Position); } }
        public void RemoveEntity(Guid entityId) { lock (_lock) { _entities.Remove(entityId); _spatialIndex.Remove(entityId); } }
        public SentientEntity? GetEntity(Guid entityId) { lock (_lock) { return _entities.TryGetValue(entityId, out var e) ? e : null; } }
        public IReadOnlyDictionary<Guid, SentientEntity> GetAllEntities() { lock (_lock) { return _entities.ToDictionary(kv => kv.Key, kv => kv.Value); } }
        public int EntityCount { get { lock (_lock) return _entities.Count; } }

        public void UpdateSpatialIndex()
        {
            lock (_lock)
            {
                _spatialIndex.Clear();
                foreach (var e in _entities.Values)
                    _spatialIndex.Insert(e.EntityId, e.Position);
            }
        }

        public List<Guid> QueryNearby(Vector3 position, float radius) => _spatialIndex.QueryRadius(position, radius);
        public List<Guid> QueryBox(Vector3 min, Vector3 max) => _spatialIndex.QueryBox(min, max);
        public Guid? QueryNearest(Vector3 position, float maxDist = float.MaxValue) => _spatialIndex.QueryNearest(position, _entities.ToDictionary(kv => kv.Key, kv => kv.Value.Position), maxDist);

        public void QueueEvent(WorldEvent evt) { lock (_lock) _events.Enqueue(evt); }
        public WorldEvent? DequeueEvent() { lock (_lock) { return _events.Count > 0 ? _events.Dequeue() : null; } }
        public int EventCount { get { lock (_lock) return _events.Count; } }

        public WorldStateData CreateSnapshot()
        {
            lock (_lock)
            {
                return new WorldStateData
                {
                    Entities = new Dictionary<Guid, SentientEntity>(_entities),
                    Time = _currentTime,
                    Weather = _weather,
                    Events = new Queue<WorldEvent>(_events),
                    SpatialIndex = _spatialIndex
                };
            }
        }

        public void Update(float deltaTime)
        {
            _currentTime += deltaTime;
            UpdateSpatialIndex();
            foreach (var e in _entities.Values)
                e.Update(CreateContext(e), deltaTime);
        }

        private EntityContext CreateContext(SentientEntity entity)
        {
            return new EntityContext
            {
                Entity = entity,
                WorldState = CreateSnapshot(),
                DeltaTime = 0.016f,
                PerceptionEvents = entity.GetPendingPerceptions().ToList(),
                Memory = entity.MemorySystem!,
                EmotionalState = entity.CurrentEmotion,
                Relationships = entity.Relationships.ToDictionary(kv => kv.Key, kv => kv.Value.Type)
            };
        }

        public void ProcessEvents()
        {
            lock (_lock)
            {
                while (_events.Count > 0)
                {
                    var evt = _events.Dequeue();
                    ProcessEvent(evt);
                }
            }
        }

        private void ProcessEvent(WorldEvent evt)
        {
            var nearby = _spatialIndex.QueryRadius(evt.Location, evt.Radius);
            foreach (var eid in nearby)
            {
                if (!_entities.TryGetValue(eid, out var entity))
                    continue;
                if (eid == evt.SourceEntity)
                    continue;
                var perception = new PerceptionEvent(
                    evt.Intensity > 0.5f ? PerceptionType.Auditory : PerceptionType.Visual,
                    evt.SourceEntity, evt.Timestamp, evt.Intensity, evt.Location,
                    Vector3.Normalize(entity.Position - evt.Location),
                    $"{evt.EventType}: {string.Join(", ", evt.Metadata.Take(3).Select(kv => $"{kv.Key}={kv.Value}"))}",
                    Math.Clamp(evt.Intensity * (1f - Vector3.Distance(entity.Position, evt.Location) / evt.Radius), 0, 1));
                entity.AddPerception(perception);
            }
        }

        public Dictionary<string, object> GetWorldStats()
        {
            lock (_lock)
            {
                return new Dictionary<string, object>
                {
                    { "EntityCount", _entities.Count },
                    { "EventCount", _events.Count },
                    { "CurrentTime", _currentTime },
                    { "Weather", _weather.Type.ToString() },
                    { "Temperature", _weather.Temperature },
                    { "Visibility", _weather.Visibility },
                    { "ActiveEntities", _entities.Values.Count(e => e.CurrentState == BehaviorState.Active) },
                    { "IdleEntities", _entities.Values.Count(e => e.CurrentState == BehaviorState.Idle) },
                    { "TerminatedEntities", _entities.Values.Count(e => e.CurrentState == BehaviorState.Terminated) }
                };
            }
        }
    }

    #endregion
    #region BEHAVIOR ANALYTICS

    public class BehaviorAnalytics
    {
        private readonly Dictionary<string, BehaviorFrequencyData> _frequencies = new();
        private readonly Dictionary<string, List<double>> _responseTimes = new();
        private readonly Dictionary<string, int> _patternCounts = new();
        private readonly Dictionary<string, List<float>> _nodeTickTimes = new();
        private readonly List<BehaviorSession> _sessions = new();
        private readonly object _lock = new();

        public void RecordBehavior(string behaviorType, string nodeId, float executionTime)
        {
            lock (_lock)
            {
                if (!_frequencies.TryGetValue(behaviorType, out var data))
                { data = new BehaviorFrequencyData { BehaviorType = behaviorType }; _frequencies[behaviorType] = data; }
                data.Count++;
                data.TotalTime += executionTime;
                data.LastOccurrence = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;

                if (!_responseTimes.TryGetValue(behaviorType, out var times))
                { times = new List<double>(); _responseTimes[behaviorType] = times; }
                times.Add(executionTime);
                if (times.Count > 1000)
                    times.RemoveRange(0, 200);

                if (!_nodeTickTimes.TryGetValue(nodeId, out var nodeTimes))
                { nodeTimes = new List<float>(); _nodeTickTimes[nodeId] = nodeTimes; }
                nodeTimes.Add(executionTime);
                if (nodeTimes.Count > 500)
                    nodeTimes.RemoveRange(0, 100);
            }
        }

        public void RecordPattern(string patternName)
        {
            lock (_lock)
            { _patternCounts.TryGetValue(patternName, out var c); _patternCounts[patternName] = c + 1; }
        }

        public void StartSession(Guid entityId, string sessionType)
        {
            lock (_lock)
            {
                _sessions.Add(new BehaviorSession
                {
                    EntityId = entityId,
                    SessionType = sessionType,
                    StartTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond
                });
            }
        }

        public void EndSession(Guid entityId)
        {
            lock (_lock)
            {
                var session = _sessions.LastOrDefault(s => s.EntityId == entityId && s.EndTime == 0);
                if (session != null)
                    session.EndTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
            }
        }

        public Dictionary<string, float> GetBehaviorFrequencies()
        {
            lock (_lock)
            { return _frequencies.ToDictionary(kv => kv.Key, kv => (float)kv.Value.Count); }
        }

        public float GetAverageResponseTime(string behaviorType)
        {
            lock (_lock)
            {
                return _responseTimes.TryGetValue(behaviorType, out var times) && times.Count > 0
                    ? (float)times.Average() : 0;
            }
        }

        public float GetP95ResponseTime(string behaviorType)
        {
            lock (_lock)
            {
                if (!_responseTimes.TryGetValue(behaviorType, out var times) || times.Count == 0)
                    return 0;
                var sorted = times.OrderBy(t => t).ToList();
                return (float)sorted[(int)(sorted.Count * 0.95)];
            }
        }

        public Dictionary<string, int> GetDominantPatterns(int topN = 5)
        {
            lock (_lock)
            { return _patternCounts.OrderByDescending(kv => kv.Value).Take(topN).ToDictionary(kv => kv.Key, kv => kv.Value); }
        }

        public float GetNodeAverageTickTime(string nodeId)
        {
            lock (_lock)
            {
                return _nodeTickTimes.TryGetValue(nodeId, out var times) && times.Count > 0
                    ? times.Average() : 0;
            }
        }

        public float GetNodeP99TickTime(string nodeId)
        {
            lock (_lock)
            {
                if (!_nodeTickTimes.TryGetValue(nodeId, out var times) || times.Count == 0)
                    return 0;
                var sorted = times.OrderBy(t => t).ToList();
                return sorted[(int)(sorted.Count * 0.99)];
            }
        }

        public Dictionary<string, object> GetComprehensiveReport()
        {
            lock (_lock)
            {
                var report = new Dictionary<string, object>
                {
                    { "TotalBehaviorTypes", _frequencies.Count },
                    { "TotalPatternsTracked", _patternCounts.Count },
                    { "TotalSessions", _sessions.Count },
                    { "ActiveSessions", _sessions.Count(s => s.EndTime == 0) },
                    { "TotalBehaviorEvents", _frequencies.Values.Sum(f => f.Count) },
                    { "AverageResponseTime", _responseTimes.Values.SelectMany(t => t).DefaultIfEmpty(0).Average() },
                    { "NodeCount", _nodeTickTimes.Count }
                };

                var topBehaviors = _frequencies.Values.OrderByDescending(f => f.Count).Take(5)
                    .ToDictionary(f => f.BehaviorType, f => new { f.Count, AvgTime = f.TotalTime / f.Count });
                report["TopBehaviors"] = topBehaviors;

                var patternReport = _patternCounts.OrderByDescending(kv => kv.Value).Take(5)
                    .ToDictionary(kv => kv.Key, kv => kv.Value);
                report["TopPatterns"] = patternReport;

                if (_sessions.Count > 0)
                {
                    var completedSessions = _sessions.Where(s => s.EndTime > 0).ToList();
                    if (completedSessions.Count > 0)
                    {
                        report["AvgSessionDuration"] = completedSessions.Average(s => s.EndTime - s.StartTime);
                        report["MaxSessionDuration"] = completedSessions.Max(s => s.EndTime - s.StartTime);
                    }
                }
                return report;
            }
        }

        public Dictionary<string, float> IdentifyAnomalies()
        {
            lock (_lock)
            {
                var anomalies = new Dictionary<string, float>();
                foreach (var (behaviorType, times) in _responseTimes)
                {
                    if (times.Count < 10)
                        continue;
                    var avg = times.Average();
                    var stdDev = Math.Sqrt(times.Average(t => (t - avg) * (t - avg)));
                    var latest = times.Last();
                    if (Math.Abs(latest - avg) > 3 * stdDev)
                        anomalies[behaviorType] = (float)((latest - avg) / stdDev);
                }
                return anomalies;
            }
        }

        public void Clear() { lock (_lock) { _frequencies.Clear(); _responseTimes.Clear(); _patternCounts.Clear(); _nodeTickTimes.Clear(); _sessions.Clear(); } }
    }

    public class BehaviorFrequencyData
    {
        public string BehaviorType { get; set; } = string.Empty;
        public int Count { get; set; }
        public double TotalTime { get; set; }
        public double LastOccurrence { get; set; }
    }

    public class BehaviorSession
    {
        public Guid EntityId { get; set; }
        public string SessionType { get; set; } = string.Empty;
        public double StartTime { get; set; }
        public double EndTime { get; set; }
    }

    #endregion

    #region ENTITY TICK SCHEDULER

    public class EntityTickScheduler
    {
        private readonly Dictionary<Guid, SentientEntity> _entities = new();
        private readonly SortedSet<(int Priority, Guid EntityId)> _priorityQueue = new(Comparer<(int Priority, Guid EntityId)>.Create((a, b) => a.Priority != b.Priority ? b.Priority.CompareTo(a.Priority) : a.EntityId.CompareTo(b.EntityId)));
        private readonly WorldStateManager _worldState;
        private readonly PerceptionSystem _perceptionSystem;
        private readonly EmotionalModel _emotionalModel;
        private readonly BehaviorDebugger? _debugger;
        private readonly BehaviorAnalytics _analytics;
        private readonly object _lock = new();
        private SchedulerStats _stats = new();

        public EntityTickScheduler(WorldStateManager worldState, PerceptionSystem perceptionSystem, EmotionalModel emotionalModel, BehaviorDebugger? debugger = null, BehaviorAnalytics? analytics = null)
        {
            _worldState = worldState;
            _perceptionSystem = perceptionSystem;
            _emotionalModel = emotionalModel;
            _debugger = debugger;
            _analytics = analytics ?? new BehaviorAnalytics();
        }

        public int ActiveEntityCount { get { lock (_lock) return _entities.Values.Count(e => e.CurrentState == BehaviorState.Active); } }
        public int DormantEntityCount { get { lock (_lock) return _entities.Values.Count(e => e.CurrentState == BehaviorState.Dormant); } }
        public SchedulerStats Stats { get { lock (_lock) return _stats; } }

        public void RegisterEntity(SentientEntity entity)
        {
            lock (_lock)
            {
                _entities[entity.EntityId] = entity;
                _priorityQueue.Add((entity.UpdatePriority, entity.EntityId));
            }
        }

        public void UnregisterEntity(Guid entityId)
        {
            lock (_lock)
            {
                if (_entities.TryGetValue(entityId, out var e))
                    _priorityQueue.Remove((e.UpdatePriority, entityId));
                _entities.Remove(entityId);
            }
        }

        public async Task TickAllEntitiesAsync(float deltaTime, CancellationToken ct = default)
        {
            var startTime = Stopwatch.GetTimestamp();
            _stats = new SchedulerStats { FrameStartTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond };

            var (active, dormant) = PartitionByRelevance();

            var activeEntities = active.ToList();
            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount, CancellationToken = ct };

            await Parallel.ForEachAsync(activeEntities, parallelOptions, async (entity, token) =>
            {
                var entityStart = Stopwatch.GetTimestamp();
                await Task.Run(() => TickEntity(entity, deltaTime), token);
                var entityTime = (Stopwatch.GetTimestamp() - entityStart) / (double)TimeSpan.TicksPerMillisecond;
                lock (_lock)
                { _stats.EntityTickTimes[entity.EntityId] = entityTime; }
            });

            foreach (var entity in dormant)
            {
                SimulateDormant(entity, deltaTime);
            }

            var frameTime = (Stopwatch.GetTimestamp() - startTime) / (double)TimeSpan.TicksPerMillisecond;
            lock (_lock)
            {
                _stats.FrameTimeMs = frameTime;
                _stats.ActiveEntityCount = activeEntities.Count;
                _stats.DormantEntityCount = dormant.Count();
                _stats.TotalEntities = _entities.Count;
            }
        }

        private (IEnumerable<SentientEntity> Active, IEnumerable<SentientEntity> Dormant) PartitionByRelevance()
        {
            var active = new List<SentientEntity>();
            var dormant = new List<SentientEntity>();

            foreach (var entity in _entities.Values)
            {
                if (entity.CurrentState == BehaviorState.Terminated)
                    continue;
                if (entity.CurrentState == BehaviorState.Active || entity.CurrentState == BehaviorState.Transitioning)
                { active.Add(entity); continue; }
                if (entity.CurrentState == BehaviorState.Dormant)
                { dormant.Add(entity); continue; }

                bool isRelevant = entity.CurrentState != BehaviorState.Idle ||
                    entity.Needs.Values.Any(v => v > 0.8f) ||
                    entity.Relationships.Values.Any(r => r.Type == RelationshipType.Hostile && r.Strength > 0.5f);

                if (isRelevant)
                    active.Add(entity);
                else
                    dormant.Add(entity);
            }
            return (active, dormant);
        }

        private void TickEntity(SentientEntity entity, float deltaTime)
        {
            var worldState = _worldState.CreateSnapshot();
            _perceptionSystem.Update(entity, worldState);

            var context = new EntityContext
            {
                Entity = entity,
                WorldState = worldState,
                DeltaTime = deltaTime,
                PerceptionEvents = entity.GetPendingPerceptions().ToList(),
                Memory = entity.MemorySystem!,
                EmotionalState = entity.CurrentEmotion,
                Relationships = entity.Relationships.ToDictionary(kv => kv.Key, kv => kv.Value.Type)
            };

            var tickStart = Stopwatch.GetTimestamp();
            entity.Update(context, deltaTime);
            var tickTime = (float)(Stopwatch.GetTimestamp() - tickStart) / TimeSpan.TicksPerMillisecond;

            _analytics?.RecordBehavior(entity.CurrentState.ToString(), entity.EntityId.ToString(), tickTime);
            _debugger?.RecordReplayFrame(worldState, _worldState.GetAllEntities());
        }

        private void SimulateDormant(SentientEntity entity, float deltaTime)
        {
            entity.Position += entity.Velocity * deltaTime * 0.1f;
            foreach (var need in entity.Needs.Keys.ToList())
                entity.Needs[need] = Math.Min(1f, entity.Needs[need] + deltaTime * 0.005f);
        }

        public void RebuildPriorityQueue()
        {
            lock (_lock)
            {
                _priorityQueue.Clear();
                foreach (var e in _entities.Values)
                    _priorityQueue.Add((e.UpdatePriority, e.EntityId));
            }
        }
    }

    public class SchedulerStats
    {
        public double FrameStartTime { get; set; }
        public double FrameTimeMs { get; set; }
        public int ActiveEntityCount { get; set; }
        public int DormantEntityCount { get; set; }
        public int TotalEntities { get; set; }
        public Dictionary<Guid, double> EntityTickTimes { get; set; } = new();
        public double AverageTickTimeMs => EntityTickTimes.Count > 0 ? EntityTickTimes.Values.Average() : 0;
        public double MaxTickTimeMs => EntityTickTimes.Count > 0 ? EntityTickTimes.Values.Max() : 0;
    }

    #endregion
    #region EXTENDED EMOTIONAL DYNAMICS

    public class EmotionalDynamicsEngine
    {
        private readonly Dictionary<Guid, EmotionalDynamicsState> _dynamicsStates = new();
        private readonly EmotionTransitionMatrix _transitionMatrix;

        public EmotionalDynamicsEngine()
        {
            _transitionMatrix = new EmotionTransitionMatrix();
        }

        public float BlendingRate { get; set; } = 0.5f;
        public float MoodSwingThreshold { get; set; } = 0.7f;
        public float EmotionalResonanceFactor { get; set; } = 0.3f;
        public float MoodStabilityBase { get; set; } = 0.6f;

        public void UpdateDynamics(Guid entityId, EmotionalStateData stateData, float deltaTime)
        {
            var dynamics = GetOrCreate(entityId);

            UpdateMood(dynamics, stateData, deltaTime);
            UpdateEmotionalMomentum(dynamics, stateData, deltaTime);
            DetectMoodSwings(dynamics, stateData);
            UpdateEmotionalMemory(dynamics, stateData, deltaTime);
            ProcessEmotionalCascade(dynamics, stateData);
            UpdateBaselineShift(dynamics, deltaTime);
        }

        private void UpdateMood(EmotionalDynamicsState dynamics, EmotionalStateData stateData, float deltaTime)
        {
            var currentMood = dynamics.CurrentMood;
            var targetMood = CalculateMoodFromEmotions(stateData);

            float blendSpeed = BlendingRate * (1f - MoodStabilityBase);
            if (dynamics.PersonalityStability > 0.7f)
                blendSpeed *= 0.5f;

            dynamics.CurrentMood = LerpEmotionalState(currentMood, targetMood, blendSpeed * deltaTime);
            dynamics.MoodIntensity = Math.Clamp(
                dynamics.MoodIntensity + (stateData.GetArousalLevel() - dynamics.MoodIntensity) * 0.1f * deltaTime,
                0, 1);
        }

        private void UpdateEmotionalMomentum(EmotionalDynamicsState dynamics, EmotionalStateData stateData, float deltaTime)
        {
            var dominant = stateData.DominantEmotion;
            if (dominant == dynamics.LastDominantEmotion)
            {
                dynamics.EmotionalMomentum = Math.Min(1f, dynamics.EmotionalMomentum + deltaTime * 0.2f);
            }
            else
            {
                dynamics.EmotionalMomentum = Math.Max(0, dynamics.EmotionalMomentum - deltaTime * 0.3f);
                dynamics.LastDominantEmotion = dominant;
                dynamics.MomentumChangeCount++;
            }
        }

        private void DetectMoodSwings(EmotionalDynamicsState dynamics, EmotionalStateData stateData)
        {
            if (dynamics.RecentDominantEmotions.Count >= 5)
            {
                var recent = dynamics.RecentDominantEmotions.TakeLast(5).ToList();
                int changes = 0;
                for (int i = 1; i < recent.Count; i++)
                {
                    if (recent[i] != recent[i - 1])
                        changes++;
                }
                if (changes >= 3)
                {
                    dynamics.IsMoodSwinging = true;
                    dynamics.MoodSwingIntensity = Math.Min(1f, changes / 3f);
                }
                else
                {
                    dynamics.IsMoodSwinging = false;
                    dynamics.MoodSwingIntensity = Math.Max(0, dynamics.MoodSwingIntensity - 0.1f);
                }
            }

            if (dynamics.IsMoodSwinging)
            {
                dynamics.TimeSinceLastSwing = 0;
            }
            else
            {
                dynamics.TimeSinceLastSwing++;
            }
        }

        private void UpdateEmotionalMemory(EmotionalDynamicsState dynamics, EmotionalStateData stateData, float deltaTime)
        {
            foreach (var entry in dynamics.EmotionalMemory.ToList())
            {
                entry.Strength = Math.Max(0, entry.Strength - deltaTime * 0.01f);
                if (entry.Strength <= 0.01f)
                    dynamics.EmotionalMemory.Remove(entry);
            }

            if (stateData.EmotionIntensities.Count > 0)
            {
                var strongest = stateData.EmotionIntensities.OrderByDescending(kv => kv.Value).First();
                if (strongest.Value > 0.3f)
                {
                    var existing = dynamics.EmotionalMemory.FirstOrDefault(m => m.Emotion == strongest.Key);
                    if (existing != null)
                    {
                        existing.Strength = Math.Min(1f, existing.Strength + strongest.Value * deltaTime * 0.1f);
                        existing.LastExperienceTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
                    }
                    else
                    {
                        dynamics.EmotionalMemory.Add(new EmotionalMemoryEntry
                        {
                            Emotion = strongest.Key,
                            Strength = strongest.Value * 0.5f,
                            FirstExperienceTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond,
                            LastExperienceTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond
                        });
                    }
                }
            }
        }

        private void ProcessEmotionalCascade(EmotionalDynamicsState dynamics, EmotionalStateData stateData)
        {
            if (dynamics.EmotionalMomentum > 0.8f)
            {
                var currentEmotions = stateData.EmotionIntensities.ToList();
                foreach (var (emotion, intensity) in currentEmotions)
                {
                    var related = _transitionMatrix.GetRelatedEmotions(emotion);
                    foreach (var (relatedEmotion, probability) in related)
                    {
                        if (intensity > 0.5f && Random.Shared.NextSingle() < probability * 0.1f)
                        {
                            float cascadeStrength = intensity * 0.2f;
                            if (stateData.EmotionIntensities.TryGetValue(relatedEmotion, out var existing))
                                stateData.EmotionIntensities[relatedEmotion] = Math.Min(1f, existing + cascadeStrength);
                            else
                                stateData.EmotionIntensities[relatedEmotion] = cascadeStrength;
                        }
                    }
                }
            }
        }

        private void UpdateBaselineShift(EmotionalDynamicsState dynamics, float deltaTime)
        {
            if (dynamics.MoodIntensity > 0.7f && dynamics.EmotionalMomentum > 0.6f)
            {
                float shiftRate = 0.001f * deltaTime * dynamics.MoodIntensity;
                dynamics.BaselineShiftAccumulator += shiftRate;
                if (dynamics.BaselineShiftAccumulator > 1f)
                {
                    dynamics.BaselineShiftAccumulator = 0;
                    if (_transitionMatrix.GetRelatedEmotions(dynamics.CurrentMood).Count > 0)
                    {
                        var candidates = _transitionMatrix.GetRelatedEmotions(dynamics.CurrentMood);
                        var weighted = candidates.OrderByDescending(c => c.Probability).First();
                        dynamics.ShiftedBaseline = weighted.Emotion;
                    }
                }
            }
        }

        private static EmotionalState CalculateMoodFromEmotions(EmotionalStateData stateData)
        {
            if (stateData.EmotionIntensities.Count == 0)
                return EmotionalState.Neutral;
            return stateData.EmotionIntensities.OrderByDescending(kv => kv.Value).First().Key;
        }

        private static EmotionalState LerpEmotionalState(EmotionalState from, EmotionalState to, float t)
        {
            if (from == to)
                return from;
            if (t >= 1f)
                return to;
            if (t <= 0f)
                return from;
            return Random.Shared.NextSingle() < t ? to : from;
        }

        public EmotionalDynamicsState? GetDynamics(Guid entityId)
        {
            return _dynamicsStates.TryGetValue(entityId, out var d) ? d : null;
        }

        private EmotionalDynamicsState GetOrCreate(Guid entityId)
        {
            if (!_dynamicsStates.TryGetValue(entityId, out var d))
            {
                d = new EmotionalDynamicsState
                {
                    PersonalityStability = MoodStabilityBase + (Random.Shared.NextSingle() - 0.5f) * 0.3f
                };
                _dynamicsStates[entityId] = d;
            }
            return d;
        }
    }

    public class EmotionalDynamicsState
    {
        public EmotionalState CurrentMood { get; set; } = EmotionalState.Neutral;
        public float MoodIntensity { get; set; } = 0.3f;
        public float EmotionalMomentum { get; set; }
        public EmotionalState LastDominantEmotion { get; set; } = EmotionalState.Neutral;
        public int MomentumChangeCount { get; set; }
        public bool IsMoodSwinging { get; set; }
        public float MoodSwingIntensity { get; set; }
        public int TimeSinceLastSwing { get; set; }
        public List<EmotionalMemoryEntry> EmotionalMemory { get; set; } = new();
        public List<EmotionalState> RecentDominantEmotions { get; set; } = new();
        public float PersonalityStability { get; set; } = 0.6f;
        public EmotionalState ShiftedBaseline { get; set; } = EmotionalState.Neutral;
        public float BaselineShiftAccumulator { get; set; }
    }

    public class EmotionalMemoryEntry
    {
        public EmotionalState Emotion { get; set; }
        public float Strength { get; set; }
        public double FirstExperienceTime { get; set; }
        public double LastExperienceTime { get; set; }
        public int ExperienceCount { get; set; } = 1;
    }

    public class EmotionTransitionMatrix
    {
        private readonly Dictionary<EmotionalState, List<(EmotionalState Emotion, float Probability)>> _transitions;

        public EmotionTransitionMatrix()
        {
            _transitions = new Dictionary<EmotionalState, List<(EmotionalState, float)>>
            {
                { EmotionalState.Happy, new List<(EmotionalState, float)>
                    { (EmotionalState.Excited, 0.3f), (EmotionalState.Calm, 0.4f), (EmotionalState.Trusting, 0.2f), (EmotionalState.Surprised, 0.1f) } },
                { EmotionalState.Sad, new List<(EmotionalState, float)>
                    { (EmotionalState.Anxious, 0.2f), (EmotionalState.Angry, 0.15f), (EmotionalState.Bored, 0.25f), (EmotionalState.Fearful, 0.1f) } },
                { EmotionalState.Angry, new List<(EmotionalState, float)>
                    { (EmotionalState.Disgusted, 0.25f), (EmotionalState.Determined, 0.2f), (EmotionalState.Fearful, 0.15f), (EmotionalState.Surprised, 0.1f) } },
                { EmotionalState.Fearful, new List<(EmotionalState, float)>
                    { (EmotionalState.Anxious, 0.3f), (EmotionalState.Surprised, 0.2f), (EmotionalState.Angry, 0.15f), (EmotionalState.Sad, 0.1f) } },
                { EmotionalState.Surprised, new List<(EmotionalState, float)>
                    { (EmotionalState.Curious, 0.3f), (EmotionalState.Fearful, 0.2f), (EmotionalState.Happy, 0.15f), (EmotionalState.Angry, 0.1f) } },
                { EmotionalState.Disgusted, new List<(EmotionalState, float)>
                    { (EmotionalState.Angry, 0.3f), (EmotionalState.Sad, 0.2f), (EmotionalState.Fearful, 0.1f), (EmotionalState.Anxious, 0.15f) } },
                { EmotionalState.Trusting, new List<(EmotionalState, float)>
                    { (EmotionalState.Happy, 0.3f), (EmotionalState.Calm, 0.25f), (EmotionalState.Anticipating, 0.2f), (EmotionalState.Curious, 0.1f) } },
                { EmotionalState.Anticipating, new List<(EmotionalState, float)>
                    { (EmotionalState.Curious, 0.25f), (EmotionalState.Excited, 0.3f), (EmotionalState.Anxious, 0.15f), (EmotionalState.Determined, 0.2f) } },
                { EmotionalState.Calm, new List<(EmotionalState, float)>
                    { (EmotionalState.Trusting, 0.2f), (EmotionalState.Happy, 0.25f), (EmotionalState.Bored, 0.15f), (EmotionalState.Anticipating, 0.1f) } },
                { EmotionalState.Excited, new List<(EmotionalState, float)>
                    { (EmotionalState.Happy, 0.35f), (EmotionalState.Surprised, 0.2f), (EmotionalState.Determined, 0.15f), (EmotionalState.Anxious, 0.1f) } },
                { EmotionalState.Bored, new List<(EmotionalState, float)>
                    { (EmotionalState.Curious, 0.2f), (EmotionalState.Sad, 0.15f), (EmotionalState.Angry, 0.1f), (EmotionalState.Calm, 0.25f) } },
                { EmotionalState.Curious, new List<(EmotionalState, float)>
                    { (EmotionalState.Excited, 0.2f), (EmotionalState.Anticipating, 0.25f), (EmotionalState.Happy, 0.15f), (EmotionalState.Surprised, 0.2f) } },
                { EmotionalState.Confused, new List<(EmotionalState, float)>
                    { (EmotionalState.Curious, 0.25f), (EmotionalState.Anxious, 0.2f), (EmotionalState.Frustrated, 0.15f), (EmotionalState.Surprised, 0.1f) } },
                { EmotionalState.Determined, new List<(EmotionalState, float)>
                    { (EmotionalState.Excited, 0.2f), (EmotionalState.Angry, 0.15f), (EmotionalState.Happy, 0.1f), (EmotionalState.Anticipating, 0.25f) } },
                { EmotionalState.Anxious, new List<(EmotionalState, float)>
                    { (EmotionalState.Fearful, 0.25f), (EmotionalState.Confused, 0.2f), (EmotionalState.Angry, 0.15f), (EmotionalState.Sad, 0.1f) } }
            };
        }

        public List<(EmotionalState Emotion, float Probability)> GetRelatedEmotions(EmotionalState emotion)
        {
            return _transitions.TryGetValue(emotion, out var related) ? related : new List<(EmotionalState, float)>();
        }

        public float GetTransitionProbability(EmotionalState from, EmotionalState to)
        {
            if (_transitions.TryGetValue(from, out var related))
            {
                var match = related.FirstOrDefault(r => r.Emotion == to);
                return match.Emotion == to ? match.Probability : 0;
            }
            return 0;
        }
    }

    #endregion

    #region EXTENDED PERCEPTION DYNAMICS

    public class AdvancedPerceptionProcessor
    {
        private readonly Dictionary<Guid, PerceptionProcessingState> _processingStates = new();
        private readonly SensoryIntegrationEngine _integrationEngine;
        private readonly PredictionEngine _predictionEngine;

        public AdvancedPerceptionProcessor()
        {
            _integrationEngine = new SensoryIntegrationEngine();
            _predictionEngine = new PredictionEngine();
        }

        public float NoiseFilterStrength { get; set; } = 0.3f;
        public float PredictionHorizon { get; set; } = 2f;
        public bool EnableCrossModalIntegration { get; set; } = true;
        public bool EnableTemporalSmoothing { get; set; } = true;

        public List<PerceptionEvent> ProcessPerceptions(SentientEntity entity, List<PerceptionEvent> rawPerceptions, float deltaTime)
        {
            var state = GetOrCreate(entity.EntityId);
            var processed = new List<PerceptionEvent>();

            var filtered = ApplyNoiseFilter(rawPerceptions, state);
            var integrated = EnableCrossModalIntegration ? _integrationEngine.Integrate(filtered) : filtered;
            var smoothed = EnableTemporalSmoothing ? ApplyTemporalSmoothing(integrated, state, deltaTime) : integrated;

            foreach (var perception in smoothed)
            {
                var relevance = CalculateRelevance(entity, perception);
                if (relevance >= 0.2f)
                {
                    var enhancedConfidence = Math.Clamp(perception.Confidence * (0.5f + relevance * 0.5f), 0, 1);
                    processed.Add(new PerceptionEvent(
                        perception.Type, perception.Source, perception.Timestamp,
                        perception.Intensity, perception.Location, perception.Direction,
                        perception.SemanticContent, enhancedConfidence));
                }
            }

            UpdateState(state, processed, deltaTime);
            return processed;
        }

        private List<PerceptionEvent> ApplyNoiseFilter(List<PerceptionEvent> perceptions, PerceptionProcessingState state)
        {
            var filtered = new List<PerceptionEvent>();
            foreach (var p in perceptions)
            {
                if (state.PerceivedRecently(p.Source, p.Type))
                {
                    float similarity = CalculateSimilarity(p, state.GetLastPerception(p.Source, p.Type));
                    if (similarity > 0.8f && p.Intensity < state.GetLastPerception(p.Source, p.Type).Intensity * 1.2f)
                        continue;
                }
                if (p.Intensity < NoiseFilterStrength * 0.5f)
                    continue;
                filtered.Add(p);
            }
            return filtered;
        }

        private List<PerceptionEvent> ApplyTemporalSmoothing(List<PerceptionEvent> perceptions, PerceptionProcessingState state, float deltaTime)
        {
            var smoothed = new List<PerceptionEvent>();
            foreach (var p in perceptions)
            {
                if (state.GetSmoothingBuffer(p.Source, p.Type).Count > 0)
                {
                    var buffer = state.GetSmoothingBuffer(p.Source, p.Type);
                    buffer.Add(p);
                    if (buffer.Count > 5)
                        buffer.RemoveAt(0);

                    float avgIntensity = buffer.Average(b => b.Intensity);
                    float avgConfidence = buffer.Average(b => b.Confidence);
                    smoothed.Add(new PerceptionEvent(
                        p.Type, p.Source, p.Timestamp, avgIntensity, p.Location, p.Direction,
                        p.SemanticContent, avgConfidence));
                }
                else
                {
                    state.GetSmoothingBuffer(p.Source, p.Type).Add(p);
                    smoothed.Add(p);
                }
            }
            return smoothed;
        }

        private float CalculateRelevance(SentientEntity entity, PerceptionEvent perception)
        {
            float relevance = 0;

            if (entity.Relationships.TryGetValue(perception.Source, out var rel))
            {
                relevance += rel.Type switch
                {
                    RelationshipType.Hostile => 0.8f,
                    RelationshipType.Friendly => 0.6f,
                    RelationshipType.Parent => 0.7f,
                    RelationshipType.Child => 0.7f,
                    RelationshipType.Partner => 0.65f,
                    RelationshipType.Master => 0.5f,
                    RelationshipType.Servant => 0.4f,
                    RelationshipType.Sibling => 0.5f,
                    _ => 0.2f
                };
                relevance *= rel.Strength;
            }

            if (perception.Type == PerceptionType.Tactile)
                relevance += 0.4f;
            if (perception.Type == PerceptionType.Visual && perception.Intensity > 0.7f)
                relevance += 0.3f;
            if (entity.Needs.Values.Any(v => v > 0.8f))
                relevance += 0.2f;

            float distFactor = 1f - (Vector3.Distance(entity.Position, perception.Location) / entity.PerceptionRadius);
            relevance *= (0.5f + distFactor * 0.5f);

            return Math.Clamp(relevance, 0, 1);
        }

        private static float CalculateSimilarity(PerceptionEvent a, PerceptionEvent b)
        {
            if (a.Type != b.Type || a.Source != b.Source)
                return 0;
            float posSim = 1f - Math.Min(1f, Vector3.Distance(a.Location, b.Location) / 10f);
            float intSim = 1f - Math.Abs(a.Intensity - b.Intensity);
            return (posSim + intSim) / 2f;
        }

        private void UpdateState(PerceptionProcessingState state, List<PerceptionEvent> processed, float deltaTime)
        {
            state.UpdateTimestamps(processed);
        }

        private PerceptionProcessingState GetOrCreate(Guid entityId)
        {
            if (!_processingStates.TryGetValue(entityId, out var state))
            {
                state = new PerceptionProcessingState();
                _processingStates[entityId] = state;
            }
            return state;
        }
    }

    public class PerceptionProcessingState
    {
        private readonly Dictionary<(Guid, PerceptionType), PerceptionEvent> _lastPerceptions = new();
        private readonly Dictionary<(Guid, PerceptionType), List<PerceptionEvent>> _smoothingBuffers = new();
        private readonly Dictionary<(Guid, PerceptionType), double> _lastPerceivedTime = new();

        public bool PerceivedRecently(Guid source, PerceptionType type, float threshold = 0.5f)
        {
            var key = (source, type);
            if (!_lastPerceivedTime.TryGetValue(key, out var time))
                return false;
            return (Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond - time) < threshold;
        }

        public PerceptionEvent? GetLastPerception(Guid source, PerceptionType type)
        {
            return _lastPerceptions.TryGetValue((source, type), out var p) ? p : default;
        }

        public List<PerceptionEvent> GetSmoothingBuffer(Guid source, PerceptionType type)
        {
            var key = (source, type);
            if (!_smoothingBuffers.TryGetValue(key, out var buf))
            { buf = new List<PerceptionEvent>(); _smoothingBuffers[key] = buf; }
            return buf;
        }

        public void UpdateTimestamps(List<PerceptionEvent> perceptions)
        {
            double now = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
            foreach (var p in perceptions)
            {
                var key = (p.Source, p.Type);
                _lastPerceptions[key] = p;
                _lastPerceivedTime[key] = now;
            }
        }
    }

    public class SensoryIntegrationEngine
    {
        public List<PerceptionEvent> Integrate(List<PerceptionEvent> perceptions)
        {
            var integrated = new List<PerceptionEvent>();
            var grouped = perceptions.GroupBy(p => p.Source);

            foreach (var group in grouped)
            {
                var sourcePerceptions = group.ToList();
                if (sourcePerceptions.Count <= 1)
                {
                    integrated.AddRange(sourcePerceptions);
                    continue;
                }

                var visual = sourcePerceptions.FirstOrDefault(p => p.Type == PerceptionType.Visual);
                var auditory = sourcePerceptions.FirstOrDefault(p => p.Type == PerceptionType.Auditory);
                var proximity = sourcePerceptions.FirstOrDefault(p => p.Type == PerceptionType.Proximity);

                if (visual.Intensity > 0 && auditory.Intensity > 0)
                {
                    float integratedIntensity = (visual.Intensity * 0.6f + auditory.Intensity * 0.4f);
                    float integratedConfidence = Math.Min(1f, (visual.Confidence + auditory.Confidence) * 0.7f);
                    integrated.Add(new PerceptionEvent(
                        PerceptionType.Visual, group.Key, visual.Timestamp,
                        integratedIntensity, visual.Location, visual.Direction,
                        $"Multimodal: {visual.SemanticContent} + {auditory.SemanticContent}",
                        integratedConfidence));
                }
                else
                {
                    integrated.AddRange(sourcePerceptions);
                }
            }
            return integrated;
        }
    }

    public class PredictionEngine
    {
        private readonly Dictionary<Guid, List<Vector3>> _positionHistory = new();
        private readonly Dictionary<Guid, List<Vector3>> _velocityHistory = new();

        public Vector3 PredictPosition(Guid entityId, Vector3 currentPosition, Vector3 currentVelocity, float timeAhead)
        {
            if (!_positionHistory.TryGetValue(entityId, out var positions) || positions.Count < 3)
                return currentPosition + currentVelocity * timeAhead;

            var recentPositions = positions.TakeLast(10).ToList();
            float totalDeltaX = 0, totalDeltaY = 0, totalDeltaZ = 0;
            for (int i = 1; i < recentPositions.Count; i++)
            {
                totalDeltaX += recentPositions[i].X - recentPositions[i - 1].X;
                totalDeltaY += recentPositions[i].Y - recentPositions[i - 1].Y;
                totalDeltaZ += recentPositions[i].Z - recentPositions[i - 1].Z;
            }
            float avgDeltaX = totalDeltaX / (recentPositions.Count - 1);
            float avgDeltaY = totalDeltaY / (recentPositions.Count - 1);
            float avgDeltaZ = totalDeltaZ / (recentPositions.Count - 1);

            return currentPosition + new Vector3(avgDeltaX, avgDeltaY, avgDeltaZ) * timeAhead;
        }

        public float PredictThreatLevel(Guid entityId, List<PerceptionEvent> recentPerceptions, float timeHorizon)
        {
            var threatPerceptions = recentPerceptions
                .Where(p => p.Source == entityId && (p.Type == PerceptionType.Tactile || p.SemanticContent.Contains("threat")))
                .ToList();

            if (threatPerceptions.Count == 0)
                return 0;

            float avgIntensity = threatPerceptions.Average(p => p.Intensity);
            float frequency = threatPerceptions.Count / Math.Max(1f, timeHorizon);
            float recencyFactor = threatPerceptions.Count > 0 ?
                1f - (float)(Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond - threatPerceptions.Max(p => p.Timestamp)) / timeHorizon : 0;

            return Math.Clamp(avgIntensity * frequency * recencyFactor, 0, 1);
        }

        public void UpdateHistory(Guid entityId, Vector3 position, Vector3 velocity)
        {
            if (!_positionHistory.TryGetValue(entityId, out var positions))
            { positions = new List<Vector3>(); _positionHistory[entityId] = positions; }
            positions.Add(position);
            if (positions.Count > 20)
                positions.RemoveAt(0);

            if (!_velocityHistory.TryGetValue(entityId, out var velocities))
            { velocities = new List<Vector3>(); _velocityHistory[entityId] = velocities; }
            velocities.Add(velocity);
            if (velocities.Count > 20)
                velocities.RemoveAt(0);
        }
    }

    #endregion
    #region EXTENDED MEMORY DYNAMICS

    public class MemoryConsolidationEngine
    {
        private readonly Dictionary<Guid, MemoryConsolidationState> _consolidationStates = new();

        public float ConsolidationThreshold { get; set; } = 0.5f;
        public float RehearsalBoost { get; set; } = 0.15f;
        public float InterferenceDecay { get; set; } = 0.05f;
        public float EmotionalAmplification { get; set; } = 0.3f;

        public void ProcessConsolidation(SentientEntity entity, IMemorySystem memorySystem, float deltaTime)
        {
            var state = GetOrCreate(entity.EntityId);
            state.TimeSinceLastRehearsal += deltaTime;
            if (state.TimeSinceLastRehearsal >= state.NextRehearsalInterval)
            {
                state.NeedsRehearsal = true;
                state.TimeSinceLastRehearsal = 0;
                state.RehearsalCount++;
                state.NextRehearsalInterval = Math.Max(5f, 30f * (float)Math.Pow(2.5, state.RehearsalCount));
            }
            if (state.NeedsRehearsal)
            {
                state.NeedsRehearsal = false;
                var query = new MemoryQuery { MaxResults = 10, SortByEmotionalIntensity = true, MinImportance = 0.3f };
                var memories = memorySystem.Retrieve(entity, query);
                foreach (var m in memories)
                {
                    m.ConsolidationStrength = Math.Min(1f, m.ConsolidationStrength + RehearsalBoost);
                    m.RetrievalCount++;
                    m.LastRetrieved = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
                    if (m.ConsolidationStrength >= ConsolidationThreshold && !m.IsConsolidated)
                    { m.IsConsolidated = true; state.ConsolidatedMemoryCount++; }
                }
            }
            state.GlobalConsolidationRate = Math.Clamp(state.GlobalConsolidationRate + (state.ConsolidatedMemoryCount - state.ForgottenMemoryCount) * 0.01f * deltaTime, 0.01f, 1f);
            foreach (var entry in state.SpacedRepetitionSchedule.ToList())
            {
                entry.ElapsedTime += deltaTime;
                if (entry.ElapsedTime >= entry.NextReviewTime)
                {
                    entry.ReviewCount++;
                    entry.ElapsedTime = 0;
                    entry.NextReviewTime *= entry.EaseFactor;
                    entry.EaseFactor = Math.Max(1.3f, entry.EaseFactor + 0.1f);
                }
            }
        }

        public MemoryConsolidationState? GetState(Guid eid) => _consolidationStates.TryGetValue(eid, out var s) ? s : null;

        private MemoryConsolidationState GetOrCreate(Guid eid)
        {
            if (!_consolidationStates.TryGetValue(eid, out var s))
            { s = new MemoryConsolidationState(); _consolidationStates[eid] = s; }
            return s;
        }
    }

    public class MemoryConsolidationState
    {
        public float TimeSinceLastRehearsal { get; set; }
        public float NextRehearsalInterval { get; set; } = 30f;
        public bool NeedsRehearsal { get; set; }
        public int RehearsalCount { get; set; }
        public int ConsolidatedMemoryCount { get; set; }
        public int ForgottenMemoryCount { get; set; }
        public float GlobalConsolidationRate { get; set; } = 0.5f;
        public List<SpacedRepetitionEntry> SpacedRepetitionSchedule { get; set; } = new();
    }

    public class SpacedRepetitionEntry
    {
        public Guid MemoryId { get; set; }
        public int ReviewCount { get; set; }
        public float EaseFactor { get; set; } = 2.5f;
        public float NextReviewTime { get; set; } = 60f;
        public float ElapsedTime { get; set; }
    }

    public class SemanticNetworkBuilder
    {
        private readonly Dictionary<string, List<string>> _graph = new();

        public void AddConcept(string concept, List<string> related)
        {
            if (!_graph.TryGetValue(concept, out var e))
            { e = new List<string>(); _graph[concept] = e; }
            foreach (var r in related)
            {
                if (!e.Contains(r))
                    e.Add(r);
                if (!_graph.TryGetValue(r, out var rev))
                { rev = new List<string>(); _graph[r] = rev; }
                if (!rev.Contains(concept))
                    rev.Add(concept);
            }
        }

        public List<string> FindRelated(string concept, int depth = 2)
        {
            var result = new HashSet<string>();
            var queue = new Queue<(string, int)>();
            queue.Enqueue((concept, 0));
            while (queue.Count > 0)
            {
                var (cur, d) = queue.Dequeue();
                if (d >= depth)
                    continue;
                if (_graph.TryGetValue(cur, out var neighbors))
                    foreach (var n in neighbors)
                        if (result.Add(n) && n != concept)
                            queue.Enqueue((n, d + 1));
            }
            return result.ToList();
        }

        public float CalculateDistance(string a, string b)
        {
            var relatedA = FindRelated(a, 3);
            if (relatedA.Contains(b))
                return 0.5f;
            var relatedB = FindRelated(b, 3);
            int shared = relatedA.Intersect(relatedB).Count();
            int total = relatedA.Union(relatedB).Count();
            return total > 0 ? 1f - (float)shared / total : 1f;
        }

        public List<string> GetConcepts() => _graph.Keys.ToList();
        public int ConceptCount => _graph.Count;
        public int LinkCount => _graph.Values.Sum(v => v.Count);
    }

    #endregion
    #region SENTIENCE MANAGER - MAIN ORCHESTRATOR

    public class SentienceManager
    {
        private readonly WorldStateManager _worldState;
        private readonly PerceptionSystem _perceptionSystem;
        private readonly EmotionalModel _emotionalModel;
        private readonly MemorySystem _memorySystem;
        private readonly RelationshipManager _relationshipManager;
        private readonly SpatialReasoningSystem _spatialReasoning;
        private readonly GroupBehaviorSystem _groupBehavior;
        private readonly EntityTickScheduler _tickScheduler;
        private readonly BehaviorDebugger _debugger;
        private readonly BehaviorAnalytics _analytics;
        private readonly BehaviorCompiler _compiler;
        private readonly EmotionalDynamicsEngine _emotionalDynamics;
        private readonly AdvancedPerceptionProcessor _advancedPerception;
        private readonly MemoryConsolidationEngine _memoryConsolidation;
        private readonly SemanticNetworkBuilder _semanticNetwork;
        private readonly PredictionEngine _predictionEngine;
        private readonly ConcurrentDictionary<Guid, SentientEntity> _entities = new();
        private readonly ConcurrentDictionary<string, BehaviorTree> _sharedBehaviorTrees = new();
        private bool _isRunning;
        private CancellationTokenSource? _cts;
        private float _globalTimeScale = 1f;

        public SentienceManager()
        {
            _worldState = new WorldStateManager();
            _perceptionSystem = new PerceptionSystem();
            _emotionalModel = new EmotionalModel();
            _memorySystem = new MemorySystem();
            _relationshipManager = new RelationshipManager();
            _spatialReasoning = new SpatialReasoningSystem();
            _groupBehavior = new GroupBehaviorSystem();
            _debugger = new BehaviorDebugger();
            _analytics = new BehaviorAnalytics();
            _compiler = new BehaviorCompiler(_debugger);
            _emotionalDynamics = new EmotionalDynamicsEngine();
            _advancedPerception = new AdvancedPerceptionProcessor();
            _memoryConsolidation = new MemoryConsolidationEngine();
            _semanticNetwork = new SemanticNetworkBuilder();
            _predictionEngine = new PredictionEngine();
            _tickScheduler = new EntityTickScheduler(_worldState, _perceptionSystem, _emotionalModel, _debugger, _analytics);
        }

        public WorldStateManager WorldState => _worldState;
        public PerceptionSystem PerceptionSystem => _perceptionSystem;
        public EmotionalModel EmotionalModel => _emotionalModel;
        public MemorySystem MemorySystem => _memorySystem;
        public RelationshipManager RelationshipManager => _relationshipManager;
        public SpatialReasoningSystem SpatialReasoning => _spatialReasoning;
        public GroupBehaviorSystem GroupBehavior => _groupBehavior;
        public BehaviorDebugger Debugger => _debugger;
        public BehaviorAnalytics Analytics => _analytics;
        public BehaviorCompiler Compiler => _compiler;
        public float GlobalTimeScale { get => _globalTimeScale; set => _globalTimeScale = Math.Max(0, value); }
        public bool IsRunning => _isRunning;
        public int EntityCount => _entities.Count;

        public SentientEntity CreateEntity(EntityType type, Vector3 position, string? behaviorTreeName = null)
        {
            var entity = new SentientEntity(Guid.NewGuid(), type)
            {
                Position = position,
                EmotionalModel = _emotionalModel,
                MemorySystem = _memorySystem
            };

            if (behaviorTreeName != null && _sharedBehaviorTrees.TryGetValue(behaviorTreeName, out var tree))
                entity.BehaviorTree = tree.Clone();

            _entities[entity.EntityId] = entity;
            _worldState.AddEntity(entity);
            _tickScheduler.RegisterEntity(entity);
            return entity;
        }

        public void RemoveEntity(Guid entityId)
        {
            if (_entities.TryRemove(entityId, out var entity))
            {
                _worldState.RemoveEntity(entityId);
                _tickScheduler.UnregisterEntity(entityId);
                _perceptionSystem.RemoveEntity(entityId);
                _emotionalModel.RemoveEntity(entityId);
                _groupBehavior.UnregisterEntity(entityId);
            }
        }

        public SentientEntity? GetEntity(Guid entityId) => _entities.TryGetValue(entityId, out var e) ? e : null;
        public IReadOnlyCollection<SentientEntity> GetAllEntities() => _entities.Values.ToList().AsReadOnly();

        public void RegisterBehaviorTree(string name, BehaviorTree tree) => _sharedBehaviorTrees[name] = tree;
        public BehaviorTree? GetBehaviorTree(string name) => _sharedBehaviorTrees.TryGetValue(name, out var t) ? t : null;

        public void SetRelationship(Guid entityA, Guid entityB, RelationshipType type, float strength = 0.5f)
        {
            _relationshipManager.AddRelationship(new Relationship
            {
                EntityA = entityA,
                EntityB = entityB,
                Type = type,
                Strength = strength,
                EstablishedTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond
            });
            if (_entities.TryGetValue(entityA, out var eA))
                eA.Relationships[entityB] = new Relationship { EntityA = entityA, EntityB = entityB, Type = type, Strength = strength };
            if (_entities.TryGetValue(entityB, out var eB))
                eB.Relationships[entityA] = new Relationship { EntityA = entityA, EntityB = entityB, Type = type, Strength = strength };
        }

        public void AddGroupMember(string groupName, Guid entityId)
        {
            _groupBehavior.RegisterEntity(entityId, groupName);
            if (_entities.TryGetValue(entityId, out var e))
                e.EntityGroup = groupName;
        }

        public async Task RunAsync(float deltaTime, CancellationToken ct)
        {
            _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            _isRunning = true;
            try
            {
                while (!_cts.Token.IsCancellationRequested)
                {
                    await UpdateAsync(deltaTime * _globalTimeScale);
                    await Task.Delay((int)(deltaTime * 1000), _cts.Token);
                }
            }
            catch (OperationCanceledException) { }
            finally { _isRunning = false; }
        }

        public void Stop() { _cts?.Cancel(); _isRunning = false; }

        public async Task UpdateAsync(float deltaTime)
        {
            _worldState.CurrentTime += deltaTime;
            _worldState.ProcessEvents();
            _perceptionSystem.UpdateSpatialIndex(_entities.Values);
            await _tickScheduler.TickAllEntitiesAsync(deltaTime);

            foreach (var entity in _entities.Values)
            {
                if (entity.CurrentState == BehaviorState.Terminated)
                    continue;
                _predictionEngine.UpdateHistory(entity.EntityId, entity.Position, entity.Velocity);
                _emotionalDynamics.UpdateDynamics(entity.EntityId, _emotionalModel.GetStateData(entity.EntityId) ?? new EmotionalStateData(), deltaTime);
                _memoryConsolidation.ProcessConsolidation(entity, _memorySystem, deltaTime);
                _relationshipManager.EvolveRelationships(deltaTime, 0.001f);
            }
        }

        public Dictionary<string, object> GetSystemReport()
        {
            var report = new Dictionary<string, object>
            {
                { "EntityCount", _entities.Count },
                { "IsRunning", _isRunning },
                { "GlobalTimeScale", _globalTimeScale },
                { "WorldTime", _worldState.CurrentTime },
                { "Weather", _worldState.Weather.Type.ToString() },
                { "Analytics", _analytics.GetComprehensiveReport() },
                { "WorldStats", _worldState.GetWorldStats() }
            };
            return report;
        }
    }

    #endregion

    #region UTILITY: VECTOR3 MATH HELPERS

    public static class VectorMath
    {
        public static float AngleBetween(Vector3 a, Vector3 b)
        {
            float dot = Vector3.Dot(Vector3.Normalize(a), Vector3.Normalize(b));
            return (float)Math.Acos(Math.Clamp(dot, -1f, 1f));
        }

        public static Vector3 Slerp(Vector3 a, Vector3 b, float t)
        {
            float angle = AngleBetween(a, b);
            if (angle < 0.001f)
                return Vector3.Lerp(a, b, t);
            float sinAngle = (float)Math.Sin(angle);
            float factorA = (float)Math.Sin((1 - t) * angle) / sinAngle;
            float factorB = (float)Math.Sin(t * angle) / sinAngle;
            return a * factorA + b * factorB;
        }

        public static Vector3 ClampMagnitude(Vector3 v, float maxLength)
        {
            if (v.LengthSquared() > maxLength * maxLength)
                return Vector3.Normalize(v) * maxLength;
            return v;
        }

        public static Vector3 RandomPointInSphere(float radius)
        {
            var rng = new Random();
            float theta = (float)(rng.NextDouble() * 2 * Math.PI);
            float phi = (float)(Math.Acos(2 * rng.NextDouble() - 1));
            float r = radius * (float)Math.Pow(rng.NextDouble(), 1.0 / 3.0);
            return new Vector3(
                r * (float)Math.Sin(phi) * (float)Math.Cos(theta),
                r * (float)Math.Sin(phi) * (float)Math.Sin(theta),
                r * (float)Math.Cos(phi));
        }

        public static Vector3 RandomPointOnCircle(float radius)
        {
            float angle = (float)(Random.Shared.NextDouble() * 2 * Math.PI);
            return new Vector3(MathF.Cos(angle) * radius, 0, MathF.Sin(angle) * radius);
        }

        public static float Lerp(float a, float b, float t) => a + (b - a) * Math.Clamp(t, 0, 1);
        public static Vector3 Lerp(Vector3 a, Vector3 b, float t) => Vector3.Lerp(a, b, Math.Clamp(t, 0, 1));
        public static float SmoothStep(float edge0, float edge1, float x)
        {
            float t = Math.Clamp((x - edge0) / (edge1 - edge0), 0, 1);
            return t * t * (3 - 2 * t);
        }
    }

    #endregion

    #region UTILITY: RANDOM UTILITIES

    public static class BehaviorRandom
    {
        [ThreadStatic] private static Random? _threadRandom;

        public static Random Instance => _threadRandom ??= new Random(Guid.NewGuid().GetHashCode());

        public static float NextFloat() => (float)Instance.NextDouble();
        public static float NextFloat(float min, float max) => min + (float)Instance.NextDouble() * (max - min);
        public static int NextInt(int min, int max) => Instance.Next(min, max);
        public static bool NextBool(float probability = 0.5f) => Instance.NextDouble() < probability;
        public static T PickRandom<T>(IReadOnlyList<T> items) => items[Instance.Next(items.Count)];

        public static T WeightedPick<T>(IReadOnlyList<T> items, IReadOnlyList<float> weights)
        {
            float total = weights.Sum();
            float r = (float)(Instance.NextDouble() * total);
            float cum = 0;
            for (int i = 0; i < items.Count; i++)
            {
                cum += weights[i];
                if (r <= cum)
                    return items[i];
            }
            return items[^1];
        }

        public static Vector3 InsideUnitSphere() => VectorMath.RandomPointInSphere(1f);
        public static Vector3 OnUnitCircle() => VectorMath.RandomPointOnCircle(1f);
        public static Vector3 InsideCircle(float radius) => VectorMath.RandomPointOnCircle(radius * (float)Math.Sqrt(Instance.NextDouble()));
    }

    #endregion

    #region UTILITY: EVENT BUS

    public class SentienceEventBus
    {
        private readonly Dictionary<string, List<Delegate>> _handlers = new();
        private readonly ConcurrentQueue<SentienceEvent> _eventQueue = new();
        private readonly object _lock = new();

        public void Subscribe<T>(string eventName, Action<T> handler) where T : SentienceEvent
        {
            lock (_lock)
            {
                if (!_handlers.TryGetValue(eventName, out var list))
                { list = new List<Delegate>(); _handlers[eventName] = list; }
                list.Add(handler);
            }
        }

        public void Unsubscribe<T>(string eventName, Action<T> handler) where T : SentienceEvent
        {
            lock (_lock)
            {
                if (_handlers.TryGetValue(eventName, out var list))
                    list.Remove(handler);
            }
        }

        public void Publish<T>(T evt) where T : SentienceEvent
        {
            _eventQueue.Enqueue(evt);
        }

        public void ProcessEvents()
        {
            while (_eventQueue.TryDequeue(out var evt))
            {
                lock (_lock)
                {
                    if (_handlers.TryGetValue(evt.EventName, out var list))
                    {
                        foreach (var handler in list)
                        {
                            if (handler.GetType().IsGenericType && handler.GetType().GetGenericTypeDefinition() == typeof(Action<>))
                            {
                                handler.DynamicInvoke(evt);
                            }
                        }
                    }
                }
            }
        }

        public void Clear() { lock (_lock) _handlers.Clear(); while (_eventQueue.TryDequeue(out _)) { } }
        public int PendingEventCount => _eventQueue.Count;
    }

    public class SentienceEvent
    {
        public string EventName { get; set; } = string.Empty;
        public double Timestamp { get; set; } = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
        public Guid SourceEntity { get; set; }
        public Dictionary<string, object> Data { get; set; } = new();
    }

    public class PerceptionEventData : SentienceEvent
    {
        public PerceptionEvent Perception { get; set; }
    }

    public class DamageEventData : SentienceEvent
    {
        public float DamageAmount { get; set; }
        public Guid DamageSource { get; set; }
    }

    public class StateChangeEventData : SentienceEvent
    {
        public BehaviorState PreviousState { get; set; }
        public BehaviorState NewState { get; set; }
    }

    #endregion
    #region NAVIGATION AND DECISION MAKING

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

    #endregion

    #region ADVANCED BEHAVIOR PATTERNS

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

    #endregion

    #region ENVIRONMENTAL RESPONSE SYSTEM

    public class EnvironmentalResponseSystem
    {
        private readonly Dictionary<Guid, EnvironmentalState> _entityEnvStates = new();

        public float WeatherResponseFactor { get; set; } = 0.3f;
        public float TimeOfDayResponseFactor { get; set; } = 0.2f;
        public float DangerResponseFactor { get; set; } = 0.5f;

        public void UpdateResponses(SentientEntity entity, WorldStateData worldState, float deltaTime)
        {
            var state = GetOrCreate(entity.EntityId);
            UpdateWeatherResponse(entity, worldState.Weather, state, deltaTime);
            UpdateTimeResponse(entity, worldState.Time, state, deltaTime);
            UpdateDangerResponse(entity, worldState, state, deltaTime);
            UpdateComfortResponse(entity, state, deltaTime);
            ApplyBehaviorModifiers(entity, state);
        }

        private void UpdateWeatherResponse(SentientEntity entity, WeatherConditions weather, EnvironmentalState state, float deltaTime)
        {
            var prevComfort = state.WeatherComfort;
            state.WeatherComfort = weather.Type switch
            {
                WeatherType.Clear => 1.0f,
                WeatherType.Cloudy => 0.8f,
                WeatherType.Rain => 0.4f,
                WeatherType.HeavyRain => 0.2f,
                WeatherType.Snow => 0.5f,
                WeatherType.Fog => 0.6f,
                WeatherType.Storm => 0.1f,
                WeatherType.Sandstorm => 0.15f,
                _ => 0.7f
            };
            state.WeatherComfort *= weather.Visibility;
            float change = state.WeatherComfort - prevComfort;
            if (Math.Abs(change) > 0.1f)
            {
                entity.AddPerception(new PerceptionEvent(
                    PerceptionType.Proximity, Guid.Empty, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond,
                    Math.Abs(change), entity.Position, Vector3.Zero,
                    $"Weather changed to {weather.Type}", Math.Abs(change)));
            }
        }

        private void UpdateTimeResponse(SentientEntity entity, double worldTime, EnvironmentalState state, float deltaTime)
        {
            double hourOfDay = (worldTime / 3600.0) % 24.0;
            state.TimeOfDayFactor = hourOfDay switch
            {
                >= 6 and < 8 => 0.8f,
                >= 8 and < 17 => 1.0f,
                >= 17 and < 20 => 0.7f,
                >= 20 or < 6 => 0.3f,
                _ => 0.5f
            };
        }

        private void UpdateDangerResponse(SentientEntity entity, WorldStateData worldState, EnvironmentalState state, float deltaTime)
        {
            float dangerLevel = 0;
            foreach (var rel in entity.Relationships.Values)
            {
                if (rel.Type == RelationshipType.Hostile)
                    dangerLevel += rel.Strength * 0.3f;
            }
            var nearbyHostiles = worldState.Entities.Values
                .Where(e => e.EntityId != entity.EntityId && entity.Relationships.TryGetValue(e.EntityId, out var r) && r.Type == RelationshipType.Hostile)
                .ToList();
            foreach (var hostile in nearbyHostiles)
            {
                float dist = entity.DistanceTo(hostile);
                dangerLevel += Math.Max(0, 1f - dist / 30f) * 0.5f;
            }
            state.DangerLevel = Math.Clamp(state.DangerLevel + (dangerLevel - state.DangerLevel) * deltaTime * 2f, 0, 1);
        }

        private void UpdateComfortResponse(SentientEntity entity, EnvironmentalState state, float deltaTime)
        {
            float overallComfort = (state.WeatherComfort * 0.4f + state.TimeOfDayFactor * 0.3f + (1f - state.DangerLevel) * 0.3f);
            state.OverallComfort = Math.Clamp(state.OverallComfort + (overallComfort - state.OverallComfort) * deltaTime, 0, 1);
        }

        private void ApplyBehaviorModifiers(SentientEntity entity, EnvironmentalState state)
        {
            entity.SetProperty("WeatherComfort", state.WeatherComfort);
            entity.SetProperty("TimeFactor", state.TimeOfDayFactor);
            entity.SetProperty("DangerLevel", state.DangerLevel);
            entity.SetProperty("OverallComfort", state.OverallComfort);
        }

        private EnvironmentalState GetOrCreate(Guid entityId)
        {
            if (!_entityEnvStates.TryGetValue(entityId, out var s))
            { s = new EnvironmentalState(); _entityEnvStates[entityId] = s; }
            return s;
        }
    }

    public class EnvironmentalState
    {
        public float WeatherComfort { get; set; } = 0.7f;
        public float TimeOfDayFactor { get; set; } = 1.0f;
        public float DangerLevel { get; set; }
        public float OverallComfort { get; set; } = 0.7f;
    }

    #endregion

    #region COMMUNICATION SYSTEM

    public class CommunicationSystem
    {
        private readonly Dictionary<Guid, List<Message>> _inbox = new();
        private readonly List<BroadcastMessage> _broadcasts = new();
        private readonly Dictionary<string, List<Guid>> _channels = new();
        private readonly object _lock = new();

        public void SendMessage(Guid from, Guid to, string content, MessageType type = MessageType.Normal, float urgency = 0.5f)
        {
            lock (_lock)
            {
                if (!_inbox.TryGetValue(to, out var box))
                { box = new List<Message>(); _inbox[to] = box; }
                box.Add(new Message
                {
                    SenderId = from,
                    Content = content,
                    Type = type,
                    Urgency = urgency,
                    Timestamp = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond
                });
                if (box.Count > 50)
                    box.RemoveAt(0);
            }
        }

        public void Broadcast(Guid from, string channel, string content, float urgency = 0.5f)
        {
            lock (_lock)
            {
                if (!_channels.TryGetValue(channel, out var subscribers))
                    return;
                foreach (var subId in subscribers)
                {
                    if (subId == from)
                        continue;
                    SendMessage(from, subId, content, MessageType.Broadcast, urgency);
                }
            }
        }

        public List<Message> ReceiveMessages(Guid entityId)
        {
            lock (_lock)
            {
                if (!_inbox.TryGetValue(entityId, out var box))
                    return new List<Message>();
                var messages = box.OrderByDescending(m => m.Urgency).ThenByDescending(m => m.Timestamp).ToList();
                box.Clear();
                return messages;
            }
        }

        public void Subscribe(Guid entityId, string channel)
        {
            lock (_lock)
            {
                if (!_channels.TryGetValue(channel, out var subs))
                { subs = new List<Guid>(); _channels[channel] = subs; }
                if (!subs.Contains(entityId))
                    subs.Add(entityId);
            }
        }

        public void Unsubscribe(Guid entityId, string channel)
        {
            lock (_lock)
            { if (_channels.TryGetValue(channel, out var subs)) subs.Remove(entityId); }
        }

        public int GetInboxCount(Guid entityId) { lock (_lock) { return _inbox.TryGetValue(entityId, out var box) ? box.Count : 0; } }
        public int GetChannelSubscriberCount(string channel) { lock (_lock) { return _channels.TryGetValue(channel, out var s) ? s.Count : 0; } }
    }

    public class Message
    {
        public Guid SenderId { get; set; }
        public string Content { get; set; } = string.Empty;
        public MessageType Type { get; set; }
        public float Urgency { get; set; }
        public double Timestamp { get; set; }
        public bool IsRead { get; set; }
    }

    public class BroadcastMessage
    {
        public Guid SenderId { get; set; }
        public string Channel { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
        public float Urgency { get; set; }
        public double Timestamp { get; set; }
    }

    public enum MessageType { Normal, Urgent, Broadcast, System, Response }

    #endregion
    #region TASK AND ACTIVITY SYSTEM

    public class EntityTaskSystem
    {
        private readonly Dictionary<Guid, TaskQueue> _taskQueues = new();
        private readonly Dictionary<Guid, ActiveTask> _activeTasks = new();
        private readonly Dictionary<string, TaskTemplate> _templates = new();
        private readonly object _lock = new();

        public float TaskTimeoutSeconds { get; set; } = 60f;
        public int MaxConcurrentTasks { get; set; } = 3;

        public void AssignTask(Guid entityId, EntityTask task)
        {
            lock (_lock)
            {
                if (!_taskQueues.TryGetValue(entityId, out var queue))
                { queue = new TaskQueue(); _taskQueues[entityId] = queue; }
                task.AssignedTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
                task.Status = TaskStatus.Pending;
                queue.PendingTasks.Add(task);
                queue.PendingTasks.Sort((a, b) => b.Priority.CompareTo(a.Priority));
            }
        }

        public void CancelTask(Guid entityId, Guid taskId)
        {
            lock (_lock)
            {
                if (_taskQueues.TryGetValue(entityId, out var queue))
                {
                    var task = queue.PendingTasks.FirstOrDefault(t => t.Id == taskId);
                    if (task != null)
                    { task.Status = TaskStatus.Failure; queue.PendingTasks.Remove(task); }
                }
                if (_activeTasks.TryGetValue(entityId, out var active) && active.Task.Id == taskId)
                { active.Task.Status = TaskStatus.Failure; _activeTasks.Remove(entityId); }
            }
        }

        public ActiveTask? GetActiveTask(Guid entityId)
        {
            lock (_lock)
            { return _activeTasks.TryGetValue(entityId, out var t) ? t : null; }
        }

        public List<EntityTask> GetPendingTasks(Guid entityId)
        {
            lock (_lock)
            { return _taskQueues.TryGetValue(entityId, out var q) ? q.PendingTasks.ToList() : new List<EntityTask>(); }
        }

        public EntityTask? GetNextTask(Guid entityId)
        {
            lock (_lock)
            {
                if (_activeTasks.ContainsKey(entityId))
                    return null;
                if (_taskQueues.TryGetValue(entityId, out var q) && q.PendingTasks.Count > 0)
                    return q.PendingTasks[0];
                return null;
            }
        }

        public void StartTask(Guid entityId, EntityTask task)
        {
            lock (_lock)
            {
                _activeTasks[entityId] = new ActiveTask
                {
                    Task = task,
                    StartTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond,
                    Progress = 0
                };
                task.Status = TaskStatus.Running;
                if (_taskQueues.TryGetValue(entityId, out var q))
                    q.PendingTasks.Remove(task);
            }
        }

        public void CompleteTask(Guid entityId, bool success)
        {
            lock (_lock)
            {
                if (_activeTasks.TryGetValue(entityId, out var active))
                {
                    active.Task.Status = success ? TaskStatus.Success : TaskStatus.Failure;
                    active.Task.CompletionTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
                    active.Task.Result = success ? "Completed" : "Failed";
                    _activeTasks.Remove(entityId);
                }
            }
        }

        public void Update(float deltaTime)
        {
            lock (_lock)
            {
                foreach (var (entityId, active) in _activeTasks.ToList())
                {
                    float elapsed = (float)((Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond) - active.StartTime);
                    if (elapsed > TaskTimeoutSeconds)
                    {
                        active.Task.Status = TaskStatus.Failure;
                        active.Task.Result = "Timeout";
                        _activeTasks.Remove(entityId);
                    }
                    else
                    {
                        active.Progress = Math.Min(1f, elapsed / Math.Max(0.1f, active.Task.ExpectedDuration));
                    }
                }
            }
        }

        public void RegisterTemplate(string name, TaskTemplate template) => _templates[name] = template;
        public TaskTemplate? GetTemplate(string name) => _templates.TryGetValue(name, out var t) ? t : null;
    }

    public class EntityTask
    {
        public Guid Id { get; init; } = Guid.NewGuid();
        public string Name { get; set; } = string.Empty;
        public string TaskType { get; set; } = string.Empty;
        public int Priority { get; set; }
        public TaskStatus Status { get; set; }
        public float ExpectedDuration { get; set; } = 5f;
        public double AssignedTime { get; set; }
        public double CompletionTime { get; set; }
        public string Result { get; set; } = string.Empty;
        public Guid TargetEntity { get; set; }
        public Vector3 TargetPosition { get; set; }
        public Dictionary<string, object> Parameters { get; set; } = new();
    }

    public class ActiveTask
    {
        public EntityTask Task { get; set; } = null!;
        public double StartTime { get; set; }
        public float Progress { get; set; }
    }

    public class TaskQueue
    {
        public List<EntityTask> PendingTasks { get; set; } = new();
        public int MaxSize { get; set; } = 20;
    }

    public class TaskTemplate
    {
        public string Name { get; set; } = string.Empty;
        public string TaskType { get; set; } = string.Empty;
        public float DefaultDuration { get; set; } = 5f;
        public int DefaultPriority { get; set; } = 1;
        public Dictionary<string, object> DefaultParameters { get; set; } = new();
    }

    #endregion

    #region PERSONALITY AND TRAIT SYSTEM

    public class PersonalitySystem
    {
        private readonly Dictionary<Guid, PersonalityProfile> _profiles = new();
        private static readonly string[] CoreTraits = { "Aggression", "Courage", "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism", "Intelligence", "Empathy", "Patience" };

        public void InitializePersonality(Guid entityId, float variance = 0.2f)
        {
            var profile = new PersonalityProfile();
            var rng = new Random(entityId.GetHashCode());
            foreach (var trait in CoreTraits)
            {
                profile.Traits[trait] = Math.Clamp(0.5f + (float)(rng.NextDouble() * 2 - 1) * variance, 0, 1);
            }
            var temperament = DetermineTemperament(profile);
            profile.Temperament = temperament;
            _profiles[entityId] = profile;
        }

        public float GetTrait(Guid entityId, string trait)
        {
            if (_profiles.TryGetValue(entityId, out var p) && p.Traits.TryGetValue(trait, out var v))
                return v;
            return 0.5f;
        }

        public void ModifyTrait(Guid entityId, string trait, float delta)
        {
            if (_profiles.TryGetValue(entityId, out var p))
            {
                p.Traits.TryGetValue(trait, out float current);
                p.Traits[trait] = Math.Clamp(current + delta, 0, 1);
            }
        }

        public string GetTemperament(Guid entityId)
        {
            return _profiles.TryGetValue(entityId, out var p) ? p.Temperament : "Unknown";
        }

        public PersonalityProfile? GetProfile(Guid entityId)
        {
            return _profiles.TryGetValue(entityId, out var p) ? p : null;
        }

        public float CalculateBehaviorWeight(Guid entityId, string behavior)
        {
            if (!_profiles.TryGetValue(entityId, out var profile))
                return 0.5f;
            return behavior.ToLower() switch
            {
                "attack" or "combat" or "fight" => (profile.Traits.GetValueOrDefault("Aggression", 0.5f) + profile.Traits.GetValueOrDefault("Courage", 0.5f)) / 2f,
                "flee" or "escape" or "hide" => 1f - (profile.Traits.GetValueOrDefault("Courage", 0.5f) + profile.Traits.GetValueOrDefault("Aggression", 0.5f)) / 2f,
                "explore" or "investigate" => profile.Traits.GetValueOrDefault("Openness", 0.5f),
                "socialize" or "communicate" => profile.Traits.GetValueOrDefault("Extraversion", 0.5f),
                "help" or "protect" or "heal" => profile.Traits.GetValueOrDefault("Agreeableness", 0.5f) * profile.Traits.GetValueOrDefault("Empathy", 0.5f),
                "wait" or "idle" or "rest" => profile.Traits.GetValueOrDefault("Patience", 0.5f),
                "patrol" or "guard" => profile.Traits.GetValueOrDefault("Conscientiousness", 0.5f),
                _ => 0.5f
            };
        }

        public EmotionalState GetEmotionalTendency(Guid entityId)
        {
            if (!_profiles.TryGetValue(entityId, out var profile))
                return EmotionalState.Neutral;
            float neuroticism = profile.Traits.GetValueOrDefault("Neuroticism", 0.5f);
            float extraversion = profile.Traits.GetValueOrDefault("Extraversion", 0.5f);

            if (neuroticism > 0.7f)
                return EmotionalState.Anxious;
            if (extraversion > 0.7f)
                return EmotionalState.Excited;
            if (neuroticism < 0.3f && extraversion < 0.3f)
                return EmotionalState.Calm;
            if (extraversion > 0.5f)
                return EmotionalState.Happy;
            return EmotionalState.Neutral;
        }

        public Dictionary<string, float> GenerateCompatibility(Guid entityA, Guid entityB)
        {
            var profileA = _profiles.TryGetValue(entityA, out var pa) ? pa : null;
            var profileB = _profiles.TryGetValue(entityB, out var pb) ? pb : null;
            if (profileA == null || profileB == null)
                return new Dictionary<string, float> { { "Overall", 0.5f } };

            var compatibility = new Dictionary<string, float>();
            float totalCompat = 0;
            int traitCount = 0;

            foreach (var trait in CoreTraits)
            {
                float valA = profileA.Traits.GetValueOrDefault(trait, 0.5f);
                float valB = profileB.Traits.GetValueOrDefault(trait, 0.5f);
                float traitCompat = 1f - Math.Abs(valA - valB);

                if (trait == "Aggression" || trait == "Neuroticism")
                    traitCompat = 1f - (valA + valB) / 2f;

                compatibility[trait] = traitCompat;
                totalCompat += traitCompat;
                traitCount++;
            }

            compatibility["Overall"] = traitCount > 0 ? totalCompat / traitCount : 0.5f;
            return compatibility;
        }

        private static string DetermineTemperament(PersonalityProfile profile)
        {
            float aggression = profile.Traits.GetValueOrDefault("Aggression", 0.5f);
            float extraversion = profile.Traits.GetValueOrDefault("Extraversion", 0.5f);
            float neuroticism = profile.Traits.GetValueOrDefault("Neuroticism", 0.5f);
            float agreeableness = profile.Traits.GetValueOrDefault("Agreeableness", 0.5f);

            if (aggression > 0.7f && extraversion > 0.6f)
                return "Choleric";
            if (extraversion > 0.7f && agreeableness > 0.6f)
                return "Sanguine";
            if (neuroticism > 0.6f && extraversion < 0.4f)
                return "Melancholic";
            if (agreeableness > 0.7f && extraversion < 0.4f)
                return "Phlegmatic";
            return "Mixed";
        }
    }

    public class PersonalityProfile
    {
        public Dictionary<string, float> Traits { get; set; } = new();
        public string Temperament { get; set; } = "Mixed";
        public double LastUpdated { get; set; } = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
        public int TraitUpdateCount { get; set; }
    }

    #endregion

    #region UTILITY: POOLING AND CACHE

    public class ObjectPool<T>
    {
        private readonly ConcurrentBag<T> _objects;
        private readonly Func<T> _generator;
        private readonly Action<T> _reset;
        private int _totalCreated;

        public ObjectPool(Func<T> generator, Action<T>? reset = null, int initialSize = 10)
        {
            _generator = generator;
            _reset = reset ?? (_ => { });
            _objects = new ConcurrentBag<T>();
            for (int i = 0; i < initialSize; i++)
            { _objects.Add(generator()); _totalCreated++; }
        }

        public T Get()
        {
            if (_objects.TryTake(out var item))
                return item;
            _totalCreated++;
            return _generator();
        }

        public void Return(T item) { _reset(item); _objects.Add(item); }
        public int AvailableCount => _objects.Count;
        public int TotalCreated => _totalCreated;
    }

    public class TimedCache<TKey, TValue> where TKey : notnull
    {
        private readonly Dictionary<TKey, (TValue Value, DateTime Expiry)> _cache = new();
        private readonly TimeSpan _defaultTtl;
        private readonly object _lock = new();

        public TimedCache(TimeSpan defaultTtl) { _defaultTtl = defaultTtl; }

        public void Set(TKey key, TValue value, TimeSpan? ttl = null)
        {
            lock (_lock)
                _cache[key] = (value, DateTime.UtcNow + (ttl ?? _defaultTtl));
        }

        public bool TryGet(TKey key, out TValue? value)
        {
            lock (_lock)
            {
                if (_cache.TryGetValue(key, out var entry) && entry.Expiry > DateTime.UtcNow)
                { value = entry.Value; return true; }
                _cache.Remove(key);
                value = default;
                return false;
            }
        }

        public void Remove(TKey key) { lock (_lock) _cache.Remove(key); }
        public void Clear() { lock (_lock) _cache.Clear(); }
        public int Count { get { lock (_lock) return _cache.Count; } }

        public void Cleanup()
        {
            lock (_lock)
            {
                var expired = _cache.Where(kv => kv.Value.Expiry <= DateTime.UtcNow).Select(kv => kv.Key).ToList();
                foreach (var key in expired)
                    _cache.Remove(key);
            }
        }
    }

    public class FrameBudgetManager
    {
        private readonly Stopwatch _frameTimer = new();
        private readonly List<double> _frameTimes = new();
        private readonly int _maxSamples;

        public FrameBudgetManager(int maxSamples = 120) { _maxSamples = maxSamples; }

        public double BudgetMs { get; set; } = 16.67;
        public double ElapsedMs => _frameTimer.Elapsed.TotalMilliseconds;
        public double RemainingMs => Math.Max(0, BudgetMs - ElapsedMs);
        public float BudgetUsage => (float)(ElapsedMs / BudgetMs);
        public bool IsOverBudget => ElapsedMs > BudgetMs;

        public void StartFrame() { _frameTimer.Restart(); }

        public void EndFrame()
        {
            _frameTimer.Stop();
            _frameTimes.Add(_frameTimer.Elapsed.TotalMilliseconds);
            if (_frameTimes.Count > _maxSamples)
                _frameTimes.RemoveAt(0);
        }

        public double GetAverageFrameTime() => _frameTimes.Count > 0 ? _frameTimes.Average() : 0;
        public double GetMaxFrameTime() => _frameTimes.Count > 0 ? _frameTimes.Max() : 0;
        public double GetMinFrameTime() => _frameTimes.Count > 0 ? _frameTimes.Min() : 0;
        public double GetPercentileFrameTime(float percentile)
        {
            if (_frameTimes.Count == 0)
                return 0;
            var sorted = _frameTimes.OrderBy(t => t).ToList();
            return sorted[Math.Clamp((int)(sorted.Count * percentile), 0, sorted.Count - 1)];
        }

        public float GetFrameRate() => _frameTimes.Count > 0 ? (float)(1000.0 / GetAverageFrameTime()) : 0;

        public Dictionary<string, object> GetStats()
        {
            return new Dictionary<string, object>
            {
                { "BudgetMs", BudgetMs },
                { "AverageFrameMs", GetAverageFrameTime() },
                { "MaxFrameMs", GetMaxFrameTime() },
                { "MinFrameMs", GetMinFrameTime() },
                { "P95FrameMs", GetPercentileFrameTime(0.95f) },
                { "P99FrameMs", GetPercentileFrameTime(0.99f) },
                { "FrameRate", GetFrameRate() },
                { "BudgetUsage", BudgetUsage },
                { "IsOverBudget", IsOverBudget },
                { "SampleCount", _frameTimes.Count }
            };
        }
    }

    #endregion

    #region WORLD STATE MANAGER EXTENSIONS

    public static class WorldStateExtensions
    {
        public static List<SentientEntity> FindEntitiesByType(this WorldStateManager wsm, EntityType type)
        {
            return wsm.GetAllEntities().Values.Where(e => e.EntityType == type).ToList();
        }

        public static List<SentientEntity> FindEntitiesByEmotion(this WorldStateManager wsm, EmotionalState emotion)
        {
            return wsm.GetAllEntities().Values.Where(e => e.CurrentEmotion == emotion).ToList();
        }

        public static List<SentientEntity> FindEntitiesByState(this WorldStateManager wsm, BehaviorState state)
        {
            return wsm.GetAllEntities().Values.Where(e => e.CurrentState == state).ToList();
        }

        public static SentientEntity? FindNearestEnemy(this WorldStateManager wsm, SentientEntity entity)
        {
            var enemies = wsm.GetAllEntities().Values
                .Where(e => e.EntityId != entity.EntityId && entity.Relationships.TryGetValue(e.EntityId, out var r) && r.Type == RelationshipType.Hostile)
                .OrderBy(e => entity.DistanceTo(e))
                .ToList();
            return enemies.Count > 0 ? enemies[0] : null;
        }

        public static SentientEntity? FindNearestAlly(this WorldStateManager wsm, SentientEntity entity)
        {
            var allies = wsm.GetAllEntities().Values
                .Where(e => e.EntityId != entity.EntityId && entity.Relationships.TryGetValue(e.EntityId, out var r) && (r.Type == RelationshipType.Friendly || r.Type == RelationshipType.Parent))
                .OrderBy(e => entity.DistanceTo(e))
                .ToList();
            return allies.Count > 0 ? allies[0] : null;
        }

        public static int CountEnemiesInRange(this WorldStateManager wsm, SentientEntity entity, float range)
        {
            return wsm.GetAllEntities().Values
                .Count(e => e.EntityId != entity.EntityId && entity.DistanceTo(e) <= range && entity.Relationships.TryGetValue(e.EntityId, out var r) && r.Type == RelationshipType.Hostile);
        }

        public static float CalculateThreatLevel(this WorldStateManager wsm, SentientEntity entity)
        {
            float threat = 0;
            foreach (var e in wsm.GetAllEntities().Values)
            {
                if (e.EntityId == entity.EntityId)
                    continue;
                if (!entity.Relationships.TryGetValue(e.EntityId, out var r) || r.Type != RelationshipType.Hostile)
                    continue;
                float dist = entity.DistanceTo(e);
                float distFactor = Math.Max(0, 1f - dist / 50f);
                threat += distFactor * r.Strength * (e.Health / e.MaxHealth);
            }
            return Math.Clamp(threat, 0, 1);
        }

        public static WeatherConditions LerpWeather(this WeatherConditions a, WeatherConditions b, float t)
        {
            return new WeatherConditions(
                a.Temperature + (b.Temperature - a.Temperature) * t,
                a.Humidity + (b.Humidity - a.Humidity) * t,
                a.WindSpeed + (b.WindSpeed - a.WindSpeed) * t,
                a.WindDirection + (b.WindDirection - a.WindDirection) * t,
                a.Visibility + (b.Visibility - a.Visibility) * t,
                a.AmbientLight + (b.AmbientLight - a.AmbientLight) * t,
                t > 0.5f ? b.Type : a.Type);
        }
    }

    #endregion
    #region EXTENDED ENTITY UTILITIES

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

    #endregion
    #region FINAL: COMPREHENSIVE XML DOCUMENTATION FACTORY

    /// <summary>
    /// Factory for creating pre-configured sentient entities with common behavior patterns.
    /// Provides convenience methods for spawning entities with appropriate behaviors.
    /// </summary>
    public class SentientEntityFactory
    {
        private readonly SentienceManager _manager;

        public SentientEntityFactory(SentienceManager manager) { _manager = manager; }

        public SentientEntity CreateNPC(Vector3 position, string behaviorType, string groupName = "")
        {
            var entity = _manager.CreateEntity(EntityType.NPC, position);
            entity.CanMove = true;
            entity.MaxSpeed = 5f;
            entity.Health = 100f;
            entity.MaxHealth = 100f;
            entity.PerceptionRadius = 50f;
            entity.FieldOfView = 120f;

            var tree = new BehaviorTree($"NPC_{behaviorType}_{entity.EntityId:N}");
            var root = new SelectorNode("NPC_Behavior");

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
                    root = new SelectorNode("NPC_Behavior", root);
                    entity.PersonalityTraits["Aggression"] = 0.8f;
                    entity.PersonalityTraits["Courage"] = 0.7f;
                    break;
                case "passive":
                    root = new SelectorNode("NPC_Behavior",
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
                    root = new SelectorNode("NPC_Behavior",
                        new ActionNode("DefaultBehavior", (_, _, _) => TaskStatus.Success));
                    break;
            }

            tree.Root = root;
            entity.BehaviorTree = tree;

            if (!string.IsNullOrEmpty(groupName))
                _manager.AddGroupMember(groupName, entity.EntityId);

            return entity;
        }

        public SentientEntity CreatePlayer(Vector3 position)
        {
            var entity = _manager.CreateEntity(EntityType.Player, position);
            entity.CanMove = true;
            entity.MaxSpeed = 8f;
            entity.Health = 150f;
            entity.MaxHealth = 150f;
            entity.PerceptionRadius = 80f;
            entity.FieldOfView = 150f;
            entity.UpdatePriority = 100;
            return entity;
        }

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
                var entity = CreateNPC(pos, "patrol", groupName);
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

    #endregion
} // namespace GDNN.Sentience
