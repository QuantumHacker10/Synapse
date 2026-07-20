// =============================================================================
// NeatGEvolutionEngine.cs - NEAT-G Evolution Engine Core
// GDNN.Engine - Geometric Deep Neural Network Engine
// Copyright (c) 2024. All rights reserved.
// =============================================================================
// This file is the heart of the G-DNN Engine implementing the NEAT-G
// (NeuroEvolution of Augmented Topologies - Geometric) algorithm.
// It provides comprehensive evolutionary optimization for neural network
// architectures with geometric awareness, semantic crossover, manifold-based
// speciation, and swarm evolution capabilities.
// =============================================================================

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using GDNN.Core.NEAT.Models;
using Synapse.Infrastructure.Logging;

namespace GDNN.Core.NEAT
{

    /// <summary>
    /// Monitors resource usage during evolution including CPU, memory, and time.
    /// Provides alerts when resource limits are approached.
    /// </summary>
    public sealed class EvolutionResourceMonitor
    {
        private readonly long _maxMemoryBytes;
        private readonly TimeSpan _maxTotalTime;
        private readonly TimeSpan _maxGenerationTime;
        private readonly Queue<ResourceSnapshot> _snapshots;
        private readonly List<ResourceAlert> _alerts;

        /// <summary>
        /// Initializes a new instance of the EvolutionResourceMonitor class.
        /// </summary>
        /// <param name="maxMemoryBytes">Maximum allowed memory usage.</param>
        /// <param name="maxTotalTime">Maximum total evolution time.</param>
        /// <param name="maxGenerationTime">Maximum time per generation.</param>
        public EvolutionResourceMonitor(
            long maxMemoryBytes = 2L * 1024 * 1024 * 1024,
            TimeSpan? maxTotalTime = null,
            TimeSpan? maxGenerationTime = null)
        {
            _maxMemoryBytes = maxMemoryBytes;
            _maxTotalTime = maxTotalTime ?? TimeSpan.FromHours(1);
            _maxGenerationTime = maxGenerationTime ?? TimeSpan.FromMinutes(5);
            _snapshots = new Queue<ResourceSnapshot>();
            _alerts = new List<ResourceAlert>();
        }

        /// <summary>Gets all triggered alerts.</summary>
        public IReadOnlyList<ResourceAlert> Alerts => _alerts.AsReadOnly();

        /// <summary>
        /// Takes a resource snapshot.
        /// </summary>
        /// <param name="generation">Current generation.</param>
        /// <returns>Resource snapshot.</returns>
        public ResourceSnapshot TakeSnapshot(int generation)
        {
            var process = Process.GetCurrentProcess();
            var snapshot = new ResourceSnapshot
            {
                Timestamp = DateTime.UtcNow,
                Generation = generation,
                MemoryUsedBytes = process.WorkingSet64,
                MemoryAvailableBytes = GC.GetTotalMemory(false),
                Gen0Collections = GC.CollectionCount(0),
                Gen1Collections = GC.CollectionCount(1),
                Gen2Collections = GC.CollectionCount(2),
                ThreadCount = Environment.ProcessorCount,
                CpuTimeMs = process.TotalProcessorTime.TotalMilliseconds
            };

            _snapshots.Enqueue(snapshot);
            while (_snapshots.Count > 1000)
                _snapshots.Dequeue();

            CheckLimits(snapshot);
            return snapshot;
        }

        /// <summary>
        /// Gets the current memory usage.
        /// </summary>
        public long GetCurrentMemoryUsage()
        {
            return Process.GetCurrentProcess().WorkingSet64;
        }

        /// <summary>
        /// Gets the memory usage trend.
        /// </summary>
        public double GetMemoryTrend()
        {
            var recent = _snapshots.TakeLast(10).ToList();
            if (recent.Count < 2)
                return 0;

            return (recent[^1].MemoryUsedBytes - recent[0].MemoryUsedBytes) /
                   Math.Max(1, recent[0].MemoryUsedBytes);
        }

        /// <summary>
        /// Forces a garbage collection if memory usage is high.
        /// </summary>
        /// <returns>Memory freed in bytes.</returns>
        public long ForceCollectionIfNeeded()
        {
            long before = GC.GetTotalMemory(false);
            if (before > _maxMemoryBytes * 0.8)
            {
                GC.Collect(2, GCCollectionMode.Forced, true);
                GC.WaitForPendingFinalizers();
            }
            long after = GC.GetTotalMemory(false);
            return Math.Max(0, before - after);
        }

        /// <summary>
        /// Gets a summary of resource usage.
        /// </summary>
        public ResourceSummary GetSummary()
        {
            var allSnapshots = _snapshots.ToList();
            if (allSnapshots.Count == 0)
                return new ResourceSummary();

            return new ResourceSummary
            {
                PeakMemoryBytes = allSnapshots.Max(s => s.MemoryUsedBytes),
                AverageMemoryBytes = (long)allSnapshots.Average(s => s.MemoryUsedBytes),
                CurrentMemoryBytes = allSnapshots.Last().MemoryUsedBytes,
                TotalCpuTimeMs = allSnapshots.Last().CpuTimeMs,
                SnapshotCount = allSnapshots.Count,
                AlertCount = _alerts.Count,
                MemoryUtilization = (double)allSnapshots.Last().MemoryUsedBytes / _maxMemoryBytes
            };
        }

        /// <summary>
        /// Clears all snapshots and alerts.
        /// </summary>
        public void Clear()
        {
            _snapshots.Clear();
            _alerts.Clear();
        }

        private void CheckLimits(ResourceSnapshot snapshot)
        {
            double memoryRatio = (double)snapshot.MemoryUsedBytes / _maxMemoryBytes;

            if (memoryRatio > 0.9)
            {
                _alerts.Add(new ResourceAlert
                {
                    Timestamp = DateTime.UtcNow,
                    AlertType = ResourceAlertType.MemoryCritical,
                    Message = $"Memory usage at {memoryRatio:P0} of limit",
                    Severity = AlertSeverity.Critical
                });
            }
            else if (memoryRatio > 0.75)
            {
                _alerts.Add(new ResourceAlert
                {
                    Timestamp = DateTime.UtcNow,
                    AlertType = ResourceAlertType.MemoryWarning,
                    Message = $"Memory usage at {memoryRatio:P0} of limit",
                    Severity = AlertSeverity.Warning
                });
            }

            var process = Process.GetCurrentProcess();
            if (process.TotalProcessorTime > _maxTotalTime)
            {
                _alerts.Add(new ResourceAlert
                {
                    Timestamp = DateTime.UtcNow,
                    AlertType = ResourceAlertType.TimeLimit,
                    Message = $"Total CPU time exceeded limit",
                    Severity = AlertSeverity.Critical
                });
            }
        }
    }

    /// <summary>
    /// Resource usage snapshot.
    /// </summary>
    public sealed class ResourceSnapshot
    {
        public DateTime Timestamp { get; init; }
        public int Generation { get; init; }
        public long MemoryUsedBytes { get; init; }
        public long MemoryAvailableBytes { get; init; }
        public int Gen0Collections { get; init; }
        public int Gen1Collections { get; init; }
        public int Gen2Collections { get; init; }
        public int ThreadCount { get; init; }
        public double CpuTimeMs { get; init; }
    }

    /// <summary>
    /// Resource usage summary.
    /// </summary>
    public sealed class ResourceSummary
    {
        public long PeakMemoryBytes { get; init; }
        public long AverageMemoryBytes { get; init; }
        public long CurrentMemoryBytes { get; init; }
        public double TotalCpuTimeMs { get; init; }
        public int SnapshotCount { get; init; }
        public int AlertCount { get; init; }
        public double MemoryUtilization { get; init; }
    }

    /// <summary>
    /// Resource usage alert.
    /// </summary>
    public sealed class ResourceAlert
    {
        public DateTime Timestamp { get; init; }
        public ResourceAlertType AlertType { get; init; }
        public string Message { get; init; } = string.Empty;
        public AlertSeverity Severity { get; init; }
    }

    /// <summary>
    /// Types of resource alerts.
    /// </summary>
    public enum ResourceAlertType
    {
        MemoryWarning,
        MemoryCritical,
        TimeLimit,
        GenerationTimeout,
        HighCPU
    }

    /// <summary>
    /// Alert severity levels.
    /// </summary>
    public enum AlertSeverity
    {
        Info,
        Warning,
        Critical
    }

}
