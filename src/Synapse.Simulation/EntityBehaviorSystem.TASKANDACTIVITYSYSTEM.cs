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

}
