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

}
