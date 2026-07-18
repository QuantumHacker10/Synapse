using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;


// ============================================================
// FILE: MemoryTracker.cs
// PATH: Memory/MemoryTracker.cs
// ============================================================


// SPDX-License-Identifier: MIT
// GDNN Neural Geometry Engine - Memory Tracker
// Global memory tracking, leak detection, and per-category budget management.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;

namespace GDNN.Memory;

/// <summary>
/// Categories for memory allocation tracking.
/// Allows per-category budget enforcement and reporting.
/// </summary>
[Flags]
public enum MemoryCategory
{
    None = 0,
    NeuralWeights = 1,
    MeshData = 2,
    TextureData = 4,
    AnimationData = 8,
    StreamingBuffers = 16,
    FrameScratch = 32,
    GPUStaging = 64,
    SpatialStructures = 128,
    EvaluationScratch = 256,
    PhysicsData = 512,
    General = 1024,
    All = 2047
}

/// <summary>
/// Severity levels for memory pressure notifications.
/// </summary>
public enum MemoryPressureLevel
{
    Low,
    Medium,
    High
}

/// <summary>
/// Event arguments for memory pressure notifications.
/// </summary>
public sealed class MemoryPressureEventArgs : EventArgs
{
    public MemoryPressureLevel Level { get; init; }
    public long TotalAllocatedBytes { get; init; }
    public MemoryCategory Category { get; init; }
    public int AllocationCount { get; init; }
    public long? BudgetBytes { get; init; }
    public double BudgetUtilization { get; init; }
}

/// <summary>
/// Records a single allocation or deallocation event for diagnostic tracking.
/// </summary>
[DebuggerDisplay("Id={AllocationId}, Category={Category}, Size={Size} bytes")]
public sealed class AllocationRecord
{
    public long AllocationId { get; init; }
    public MemoryCategory Category { get; init; }
    public long Size { get; init; }
    public string? StackTrace { get; init; }
    public DateTime Timestamp { get; init; }
    public int ThreadId { get; init; }
    public string? Tag { get; init; }
    public bool IsFree { get; init; }
    public bool IsFreed { get; set; }
    public DateTime? FreeTimestamp { get; set; }
}

/// <summary>
/// Per-category budget and statistics.
/// </summary>
[DebuggerDisplay("Category={Category}, Used={UsedBytes}/{BudgetBytes} ({Utilization:P0})")]
public sealed class CategoryStats
{
    public MemoryCategory Category { get; init; }
    public long UsedBytes { get; set; }
    public long? BudgetBytes { get; set; }
    public long TotalAllocatedBytes { get; set; }
    public long AllocationCount { get; set; }
    public int ActiveAllocationCount { get; set; }
    public long PeakUsedBytes { get; set; }
    public int PeakActiveAllocationCount { get; set; }

    public double Utilization => BudgetBytes.HasValue && BudgetBytes.Value > 0
        ? (double)UsedBytes / BudgetBytes.Value
        : 0.0;
}

/// <summary>
/// Global memory tracker for the GDNN engine.
/// Provides allocation/deallocation recording with callstacks, memory leak detection,
/// per-category budget tracking, and memory pressure notifications.
/// Thread-safe for concurrent use from multiple threads.
/// </summary>
public sealed class MemoryTracker : IDisposable
{
    /// <summary>
    /// Event raised when memory pressure changes level.
    /// </summary>
    public static event EventHandler<MemoryPressureEventArgs>? MemoryPressureChanged;

    public long MediumPressureThreshold { get; set; } = 512L * 1024 * 1024;
    public long HighPressureThreshold { get; set; } = 1024L * 1024 * 1024;
    public bool CaptureStackTraces { get; set; }
    public bool RetainFreedAllocations { get; set; }
    public int MaxHistorySize { get; set; } = 100_000;

    private long _nextAllocationId;
    private long _totalAllocatedBytes;
    private long _totalAllocationCount;
    private long _totalFreeCount;
    private long _currentAllocatedBytes;
    private int _activeAllocationCount;
    private MemoryPressureLevel _currentPressureLevel;
    private bool _disposed;

    private readonly ConcurrentDictionary<long, AllocationRecord> _activeAllocations;
    private readonly ConcurrentQueue<AllocationRecord> _freedHistory;
    private readonly ConcurrentDictionary<MemoryCategory, CategoryStats> _categoryStats;
    private readonly ConcurrentDictionary<int, long> _threadAllocationCount;
    private readonly List<Action<AllocationRecord>> _allocationListeners;
    private readonly List<Action<AllocationRecord>> _freeListeners;

    public long CurrentAllocatedBytes => Interlocked.Read(ref _currentAllocatedBytes);
    public long TotalAllocatedBytes => Interlocked.Read(ref _totalAllocatedBytes);
    public long TotalAllocationCount => Interlocked.Read(ref _totalAllocationCount);
    public long TotalFreeCount => Interlocked.Read(ref _totalFreeCount);
    public int ActiveAllocationCount => _activeAllocationCount;
    public MemoryPressureLevel CurrentPressureLevel => _currentPressureLevel;
    public int FreedHistoryCount => _freedHistory.Count;

    /// <summary>
    /// Initializes a new memory tracker.
    /// </summary>
    public MemoryTracker()
    {
        _activeAllocations = new ConcurrentDictionary<long, AllocationRecord>();
        _freedHistory = new ConcurrentQueue<AllocationRecord>();
        _categoryStats = new ConcurrentDictionary<MemoryCategory, CategoryStats>();
        _threadAllocationCount = new ConcurrentDictionary<int, long>();
        _allocationListeners = new List<Action<AllocationRecord>>();
        _freeListeners = new List<Action<AllocationRecord>>();
    }

    /// <summary>
    /// Records a new allocation.
    /// </summary>
    public AllocationRecord TrackAllocation(long size, MemoryCategory category = MemoryCategory.General, string? tag = null)
    {
        long allocId = Interlocked.Increment(ref _nextAllocationId);
        int threadId = Thread.CurrentThread.ManagedThreadId;

        var record = new AllocationRecord
        {
            AllocationId = allocId,
            Category = category,
            Size = size,
            Timestamp = DateTime.UtcNow,
            ThreadId = threadId,
            Tag = tag,
            StackTrace = CaptureStackTraces ? Environment.StackTrace : null
        };

        Interlocked.Add(ref _totalAllocatedBytes, size);
        Interlocked.Increment(ref _totalAllocationCount);
        Interlocked.Add(ref _currentAllocatedBytes, size);
        Interlocked.Increment(ref _activeAllocationCount);

        UpdateCategoryAllocation(category, size);
        _threadAllocationCount.AddOrUpdate(threadId, 1, (_, count) => count + 1);
        _activeAllocations[allocId] = record;

        CheckBudgetAndPressure(category);
        NotifyAllocation(record);

        return record;
    }

    /// <summary>
    /// Records a deallocation for a previously tracked allocation.
    /// </summary>
    public bool TrackFree(long allocationId)
    {
        if (!_activeAllocations.TryRemove(allocationId, out var record))
            return false;

        record.IsFreed = true;
        record.FreeTimestamp = DateTime.UtcNow;

        Interlocked.Add(ref _currentAllocatedBytes, -record.Size);
        Interlocked.Increment(ref _totalFreeCount);
        Interlocked.Decrement(ref _activeAllocationCount);

        UpdateCategoryFree(record.Category, record.Size);

        if (_threadAllocationCount.TryGetValue(record.ThreadId, out long count) && count > 0)
            _threadAllocationCount[record.ThreadId] = count - 1;

        if (RetainFreedAllocations)
        {
            _freedHistory.Enqueue(record);
            TrimHistory();
        }

        NotifyFree(record);
        return true;
    }

    /// <summary>
    /// Records a free operation with the specified size and category.
    /// </summary>
    public void TrackFree(long size, MemoryCategory category)
    {
        Interlocked.Add(ref _currentAllocatedBytes, -size);
        Interlocked.Increment(ref _totalFreeCount);
        Interlocked.Decrement(ref _activeAllocationCount);
        UpdateCategoryFree(category, size);
    }

    /// <summary>
    /// Sets the budget limit for a specific memory category.
    /// </summary>
    public void SetBudget(MemoryCategory category, long? budgetBytes)
    {
        var stats = _categoryStats.GetOrAdd(category, c => new CategoryStats { Category = c });
        stats.BudgetBytes = budgetBytes;
    }

    /// <summary>
    /// Gets the current statistics for a specific memory category.
    /// </summary>
    public CategoryStats? GetCategoryStats(MemoryCategory category)
    {
        _categoryStats.TryGetValue(category, out var stats);
        return stats;
    }

    /// <summary>
    /// Gets statistics for all active categories.
    /// </summary>
    public IReadOnlyList<CategoryStats> GetAllCategoryStats()
    {
        return _categoryStats.Values.ToList();
    }

    /// <summary>
    /// Detects potential memory leaks by examining active allocations.
    /// </summary>
    public IReadOnlyList<AllocationRecord> DetectLeaks(double minAgeSeconds = 30.0)
    {
        var leaks = new List<AllocationRecord>();
        var cutoff = DateTime.UtcNow.AddSeconds(-minAgeSeconds);

        foreach (var kvp in _activeAllocations)
        {
            if (kvp.Value.Timestamp < cutoff)
                leaks.Add(kvp.Value);
        }

        return leaks;
    }

    /// <summary>
    /// Detects per-thread leaks by checking outstanding allocations.
    /// </summary>
    public IReadOnlyDictionary<int, long> DetectThreadLeaks()
    {
        var leaks = new Dictionary<int, long>();
        foreach (var kvp in _threadAllocationCount)
        {
            if (kvp.Value > 0)
                leaks[kvp.Key] = kvp.Value;
        }
        return leaks;
    }

    /// <summary>
    /// Gets the top N largest active allocations.
    /// </summary>
    public IReadOnlyList<AllocationRecord> GetTopAllocations(int count = 10)
    {
        var sorted = new SortedList<long, AllocationRecord>();
        foreach (var kvp in _activeAllocations)
        {
            if (sorted.Count < count)
            {
                sorted.Add(kvp.Value.Size, kvp.Value);
            }
            else if (kvp.Value.Size > sorted.Keys[0])
            {
                sorted.RemoveAt(0);
                sorted.Add(kvp.Value.Size, kvp.Value);
            }
        }
        var result = new List<AllocationRecord>(sorted.Values);
        result.Reverse();
        return result;
    }

    /// <summary>
    /// Generates a comprehensive diagnostic report.
    /// </summary>
    public MemoryTrackerReport GenerateReport()
    {
        var categorySnapshots = new List<CategorySnapshot>();
        foreach (var kvp in _categoryStats)
        {
            categorySnapshots.Add(new CategorySnapshot
            {
                Category = kvp.Key,
                UsedBytes = kvp.Value.UsedBytes,
                BudgetBytes = kvp.Value.BudgetBytes,
                AllocationCount = kvp.Value.AllocationCount,
                ActiveCount = kvp.Value.ActiveAllocationCount,
                PeakUsedBytes = kvp.Value.PeakUsedBytes,
                Utilization = kvp.Value.Utilization
            });
        }

        return new MemoryTrackerReport
        {
            Timestamp = DateTime.UtcNow,
            CurrentAllocatedBytes = CurrentAllocatedBytes,
            TotalAllocatedBytes = TotalAllocatedBytes,
            TotalAllocationCount = TotalAllocationCount,
            TotalFreeCount = TotalFreeCount,
            ActiveAllocationCount = ActiveAllocationCount,
            CurrentPressureLevel = CurrentPressureLevel,
            CategorySnapshots = categorySnapshots,
            TopAllocations = GetTopAllocations(20),
            PotentialLeaks = DetectLeaks(60.0)
        };
    }

    /// <summary>
    /// Registers a listener that is called on every allocation.
    /// </summary>
    public void OnAllocation(Action<AllocationRecord> listener)
    {
        lock (_allocationListeners) { _allocationListeners.Add(listener); }
    }

    /// <summary>
    /// Registers a listener that is called on every free operation.
    /// </summary>
    public void OnFree(Action<AllocationRecord> listener)
    {
        lock (_freeListeners) { _freeListeners.Add(listener); }
    }

    /// <summary>
    /// Resets all tracking state. Does not affect existing allocations.
    /// </summary>
    public void Reset()
    {
        Interlocked.Exchange(ref _nextAllocationId, 0);
        Interlocked.Exchange(ref _totalAllocatedBytes, 0);
        Interlocked.Exchange(ref _totalAllocationCount, 0);
        Interlocked.Exchange(ref _totalFreeCount, 0);
        Interlocked.Exchange(ref _currentAllocatedBytes, 0);
        Interlocked.Exchange(ref _activeAllocationCount, 0);
        _activeAllocations.Clear();
        while (_freedHistory.TryDequeue(out _)) { }
        _categoryStats.Clear();
        _threadAllocationCount.Clear();
    }

    private void UpdateCategoryAllocation(MemoryCategory category, long size)
    {
        var stats = _categoryStats.GetOrAdd(category, c => new CategoryStats { Category = c });
        stats.UsedBytes += size;
        stats.TotalAllocatedBytes += size;
        stats.AllocationCount++;
        stats.ActiveAllocationCount++;
        if (stats.UsedBytes > stats.PeakUsedBytes) stats.PeakUsedBytes = stats.UsedBytes;
        if (stats.ActiveAllocationCount > stats.PeakActiveAllocationCount)
            stats.PeakActiveAllocationCount = stats.ActiveAllocationCount;
    }

    private void UpdateCategoryFree(MemoryCategory category, long size)
    {
        if (_categoryStats.TryGetValue(category, out var stats))
        {
            stats.UsedBytes -= size;
            if (stats.UsedBytes < 0) stats.UsedBytes = 0;
            stats.ActiveAllocationCount--;
            if (stats.ActiveAllocationCount < 0) stats.ActiveAllocationCount = 0;
        }
    }

    private void CheckBudgetAndPressure(MemoryCategory category)
    {
        if (!_categoryStats.TryGetValue(category, out var stats)) return;

        if (stats.BudgetBytes.HasValue && stats.UsedBytes > stats.BudgetBytes.Value)
        {
            RaisePressure(new MemoryPressureEventArgs
            {
                Level = MemoryPressureLevel.High,
                TotalAllocatedBytes = CurrentAllocatedBytes,
                Category = category,
                AllocationCount = stats.ActiveAllocationCount,
                BudgetBytes = stats.BudgetBytes,
                BudgetUtilization = stats.Utilization
            });
            return;
        }

        MemoryPressureLevel level;
        if (CurrentAllocatedBytes >= HighPressureThreshold)
            level = MemoryPressureLevel.High;
        else if (CurrentAllocatedBytes >= MediumPressureThreshold)
            level = MemoryPressureLevel.Medium;
        else
            level = MemoryPressureLevel.Low;

        if (level != _currentPressureLevel)
        {
            _currentPressureLevel = level;
            RaisePressure(new MemoryPressureEventArgs
            {
                Level = level,
                TotalAllocatedBytes = CurrentAllocatedBytes,
                Category = category,
                AllocationCount = stats.ActiveAllocationCount,
                BudgetBytes = stats.BudgetBytes,
                BudgetUtilization = stats.Utilization
            });
        }
    }

    private void RaisePressure(MemoryPressureEventArgs args)
    {
        MemoryPressureChanged?.Invoke(this, args);
    }

    private void NotifyAllocation(AllocationRecord record)
    {
        lock (_allocationListeners)
        {
            foreach (var listener in _allocationListeners)
            {
                try { listener(record); } catch { }
            }
        }
    }

    private void NotifyFree(AllocationRecord record)
    {
        lock (_freeListeners)
        {
            foreach (var listener in _freeListeners)
            {
                try { listener(record); } catch { }
            }
        }
    }

    private void TrimHistory()
    {
        while (_freedHistory.Count > MaxHistorySize)
            _freedHistory.TryDequeue(out _);
    }

    /// <summary>
    /// Disposes the memory tracker.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _activeAllocations.Clear();
        while (_freedHistory.TryDequeue(out _)) { }
        _categoryStats.Clear();
        _threadAllocationCount.Clear();
        lock (_allocationListeners) _allocationListeners.Clear();
        lock (_freeListeners) _freeListeners.Clear();
    }
}

/// <summary>
/// A snapshot of category statistics for reporting.
/// </summary>
public sealed class CategorySnapshot
{
    public MemoryCategory Category { get; init; }
    public long UsedBytes { get; init; }
    public long? BudgetBytes { get; init; }
    public long AllocationCount { get; init; }
    public int ActiveCount { get; init; }
    public long PeakUsedBytes { get; init; }
    public double Utilization { get; init; }
}

/// <summary>
/// Comprehensive diagnostic report from the memory tracker.
/// </summary>
[DebuggerDisplay("Allocated={CurrentAllocatedBytes / 1024.0:F1}KB, Active={ActiveAllocationCount}, Pressure={CurrentPressureLevel}")]
public sealed class MemoryTrackerReport
{
    public DateTime Timestamp { get; init; }
    public long CurrentAllocatedBytes { get; init; }
    public long TotalAllocatedBytes { get; init; }
    public long TotalAllocationCount { get; init; }
    public long TotalFreeCount { get; init; }
    public int ActiveAllocationCount { get; init; }
    public MemoryPressureLevel CurrentPressureLevel { get; init; }
    public IReadOnlyList<CategorySnapshot> CategorySnapshots { get; init; } = [];
    public IReadOnlyList<AllocationRecord> TopAllocations { get; init; } = [];
    public IReadOnlyList<AllocationRecord> PotentialLeaks { get; init; } = [];

    public override string ToString() =>
        $"MemoryReport: {CurrentAllocatedBytes / 1024.0:F1}KB, " +
        $"{ActiveAllocationCount} active, Pressure={CurrentPressureLevel}";
}

/// <summary>
/// Provides a RAII-style scope guard that tracks an allocation and automatically
/// frees it when disposed.
/// </summary>
public readonly struct TrackedAllocationScope : IDisposable
{
    private readonly MemoryTracker _tracker;
    private readonly long _allocationId;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TrackedAllocationScope(MemoryTracker tracker, long size, MemoryCategory category = MemoryCategory.General, string? tag = null)
    {
        _tracker = tracker;
        var record = tracker.TrackAllocation(size, category, tag);
        _allocationId = record.AllocationId;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Dispose()
    {
        _tracker?.TrackFree(_allocationId);
    }
}


