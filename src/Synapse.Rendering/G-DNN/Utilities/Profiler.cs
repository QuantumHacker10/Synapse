using System;
// ============================================================
// FILE: Profiler.cs
// PATH: Utilities/Profiler.cs
// ============================================================

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics;
using System.IO;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Utilities;

/// <summary>
/// High-resolution performance profiler with hierarchical section tracking,
/// moving averages, percentile calculations, and GPU timestamp integration.
/// </summary>
public sealed class Profiler : IDisposable
{
    private readonly Dictionary<string, SectionData> _sections = new();
    private readonly Dictionary<int, Stack<string>> _threadLocalSections = new();
    private readonly ReaderWriterLockSlim _lock = new(LockRecursionPolicy.SupportsRecursion);
    private readonly Stopwatch _frameStopwatch = new();
    private readonly List<double> _frameTimes = new();
    private readonly int _maxFrameSamples;
    private long _frameCount;
    private bool _disposed;

    /// <summary>Maximum number of frame time samples kept for statistics.</summary>
    public int MaxFrameSamples => _maxFrameSamples;

    /// <summary>Total number of frames profiled.</summary>
    public long FrameCount => Interlocked.Read(ref _frameCount);

    /// <summary>GPU timestamp provider, or null if unavailable.</summary>
    public IGpuTimestampProvider? GpuProvider { get; set; }

    /// <summary>Memory allocation tracker, or null if disabled.</summary>
    public IMemoryTracker? MemoryTracker { get; set; }

    /// <summary>
    /// Creates a new profiler with the specified maximum frame sample count.
    /// </summary>
    /// <param name="maxFrameSamples">Number of frame times to retain for statistics (default 1024).</param>
    public Profiler(int maxFrameSamples = 1024)
    {
        _maxFrameSamples = maxFrameSamples;
        _frameStopwatch.Start();
    }

    /// <summary>
    /// Begins a named profiling section on the calling thread.
    /// </summary>
    /// <param name="sectionName">Unique name identifying this section.</param>
    public void BeginSection(string sectionName)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(sectionName);

        int threadId = Thread.CurrentThread.ManagedThreadId;
        long timestamp = Stopwatch.GetTimestamp();
        long gpuTimestamp = GpuProvider?.GetCurrentTimestamp() ?? 0;
        long allocBytes = MemoryTracker?.GetAllocatedBytes() ?? 0;

        _lock.EnterWriteLock();
        try
        {
            if (!_sections.TryGetValue(sectionName, out SectionData? data))
            {
                data = new SectionData(sectionName);
                _sections[sectionName] = data;
            }

            data.Begin(threadId, timestamp, gpuTimestamp, allocBytes);

            if (!_threadLocalSections.TryGetValue(threadId, out Stack<string>? stack))
            {
                stack = new Stack<string>();
                _threadLocalSections[threadId] = stack;
            }

            if (stack.Count > 0)
            {
                string parentName = stack.Peek();
                if (_sections.TryGetValue(parentName, out SectionData? parent))
                {
                    parent.AddChild(sectionName);
                    data.ParentSection = parentName;
                }
            }

            stack.Push(sectionName);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Ends the most recent profiling section on the calling thread.
    /// </summary>
    public void EndSection()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int threadId = Thread.CurrentThread.ManagedThreadId;
        long timestamp = Stopwatch.GetTimestamp();
        long gpuTimestamp = GpuProvider?.GetCurrentTimestamp() ?? 0;
        long allocBytes = MemoryTracker?.GetAllocatedBytes() ?? 0;

        _lock.EnterWriteLock();
        try
        {
            if (!_threadLocalSections.TryGetValue(threadId, out Stack<string>? stack) || stack.Count == 0)
            {
                throw new InvalidOperationException("No active profiling section to end.");
            }

            string sectionName = stack.Pop();
            if (_sections.TryGetValue(sectionName, out SectionData? data))
            {
                data.End(threadId, timestamp, gpuTimestamp, allocBytes);
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Records a single frame time and increments the frame counter.
    /// </summary>
    public void EndFrame()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        double elapsed = _frameStopwatch.Elapsed.TotalMilliseconds;
        _frameStopwatch.Restart();
        Interlocked.Increment(ref _frameCount);

        _lock.EnterWriteLock();
        try
        {
            _frameTimes.Add(elapsed);
            if (_frameTimes.Count > _maxFrameSamples)
            {
                _frameTimes.RemoveAt(0);
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Gets the snapshot of all profiling sections.
    /// </summary>
    public ProfilerSnapshot GetSnapshot()
    {
        _lock.EnterReadLock();
        try
        {
            var sections = _sections.Values.Select(s => s.ToSnapshot()).ToList();
            double[] frameTimesCopy;
            _lock.EnterUpgradeableReadLock();
            try
            {
                frameTimesCopy = _frameTimes.ToArray();
            }
            finally
            {
                _lock.ExitUpgradeableReadLock();
            }

            return new ProfilerSnapshot(sections, frameTimesCopy, FrameCount);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Gets computed statistics for the specified section.
    /// </summary>
    public SectionStatistics? GetSectionStatistics(string sectionName)
    {
        _lock.EnterReadLock();
        try
        {
            if (_sections.TryGetValue(sectionName, out SectionData? data))
            {
                return data.ComputeStatistics();
            }
            return null;
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Gets frame time statistics across all sampled frames.
    /// </summary>
    public FrameStatistics GetFrameStatistics()
    {
        _lock.EnterReadLock();
        try
        {
            if (_frameTimes.Count == 0)
                return new FrameStatistics(0, 0, 0, 0, 0, 0, 0, 0);

            double[] sorted = _frameTimes.ToArray();
            Array.Sort(sorted);

            double sum = 0;
            foreach (double t in sorted)
                sum += t;

            double mean = sum / sorted.Length;
            double variance = 0;
            foreach (double t in sorted)
                variance += (t - mean) * (t - mean);
            variance /= sorted.Length;

            return new FrameStatistics(
                mean: mean,
                min: sorted[0],
                max: sorted[^1],
                median: GetPercentile(sorted, 50),
                p95: GetPercentile(sorted, 95),
                p99: GetPercentile(sorted, 99),
                stdDev: Math.Sqrt(variance),
                fps: 1000.0 / Math.Max(mean, 0.001)
            );
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Resets all profiling data.
    /// </summary>
    public void Reset()
    {
        _lock.EnterWriteLock();
        try
        {
            _sections.Clear();
            _frameTimes.Clear();
            _threadLocalSections.Clear();
            Interlocked.Exchange(ref _frameCount, 0);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Exports profiling data in CSV format.
    /// </summary>
    public string ExportCsv()
    {
        _lock.EnterReadLock();
        try
        {
            var sb = new StringBuilder();
            sb.AppendLine("Section,Count,TotalMs,AvgMs,MinMs,MaxMs,P95Ms,P99Ms,Parent");

            foreach (var section in _sections.Values.OrderBy(s => s.Name))
            {
                var stats = section.ComputeStatistics();
                sb.AppendLine($"{section.Name},{stats.Count},{stats.TotalMs:F4}," +
                    $"{stats.AverageMs:F4},{stats.MinMs:F4},{stats.MaxMs:F4}," +
                    $"{stats.P95Ms:F4},{stats.P99Ms:F4},{section.ParentSection ?? ""}");
            }

            return sb.ToString();
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Exports profiling data in JSON format.
    /// </summary>
    public string ExportJson()
    {
        _lock.EnterReadLock();
        try
        {
            var sb = new StringBuilder();
            sb.AppendLine("{");
            sb.AppendLine("  \"frameStatistics\": {");
            var frameStats = GetFrameStatistics();
            sb.AppendLine($"    \"meanMs\": {frameStats.Mean:F4},");
            sb.AppendLine($"    \"minMs\": {frameStats.Min:F4},");
            sb.AppendLine($"    \"maxMs\": {frameStats.Max:F4},");
            sb.AppendLine($"    \"fps\": {frameStats.Fps:F2},");
            sb.AppendLine($"    \"frameCount\": {FrameCount}");
            sb.AppendLine("  },");
            sb.AppendLine("  \"sections\": [");

            bool first = true;
            foreach (var section in _sections.Values.OrderBy(s => s.Name))
            {
                if (!first)
                    sb.AppendLine(",");
                first = false;
                var stats = section.ComputeStatistics();
                sb.AppendLine("    {");
                sb.AppendLine($"      \"name\": \"{section.Name}\",");
                sb.AppendLine($"      \"count\": {stats.Count},");
                sb.AppendLine($"      \"totalMs\": {stats.TotalMs:F4},");
                sb.AppendLine($"      \"avgMs\": {stats.AverageMs:F4},");
                sb.AppendLine($"      \"minMs\": {stats.MinMs:F4},");
                sb.AppendLine($"      \"maxMs\": {stats.MaxMs:F4},");
                sb.AppendLine($"      \"p95Ms\": {stats.P95Ms:F4},");
                sb.AppendLine($"      \"p99Ms\": {stats.P99Ms:F4},");
                sb.AppendLine($"      \"parent\": \"{section.ParentSection ?? ""}\"");
                sb.Append("    }");
            }

            sb.AppendLine();
            sb.AppendLine("  ]");
            sb.AppendLine("}");
            return sb.ToString();
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Exports profiling data to a file.
    /// </summary>
    /// <param name="filePath">Path to write the export file.</param>
    /// <param name="format">Export format (csv or json).</param>
    public void ExportToFile(string filePath, string format = "csv")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        string content = format.ToLowerInvariant() switch
        {
            "json" => ExportJson(),
            "csv" => ExportCsv(),
            _ => throw new ArgumentException($"Unsupported format: {format}")
        };
        File.WriteAllText(filePath, content);
    }

    /// <summary>
    /// Gets all section names currently tracked.
    /// </summary>
    public IReadOnlyCollection<string> GetSectionNames()
    {
        _lock.EnterReadLock();
        try
        {
            return _sections.Keys.ToArray();
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Gets all sections that are children of the specified parent.
    /// </summary>
    public IReadOnlyList<string> GetChildSections(string parentName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(parentName);
        _lock.EnterReadLock();
        try
        {
            if (_sections.TryGetValue(parentName, out SectionData? data))
            {
                return data.ChildSections.ToArray();
            }
            return Array.Empty<string>();
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    private static double GetPercentile(double[] sorted, double percentile)
    {
        if (sorted.Length == 0)
            return 0;
        double index = (percentile / 100.0) * (sorted.Length - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);
        if (lower == upper)
            return sorted[lower];
        double frac = index - lower;
        return sorted[lower] * (1 - frac) + sorted[upper] * frac;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _lock.Dispose();
    }

    private sealed class SectionData
    {
        public string Name { get; }
        public string? ParentSection { get; set; }
        private readonly List<string> _childSections = new();
        private readonly List<SectionTiming> _timings = new();
        private readonly object _timingLock = new();
        private readonly Dictionary<int, long> _activeGpuStarts = new();
        private readonly Dictionary<int, long> _activeAllocStarts = new();

        public IReadOnlyCollection<string> ChildSections => _childSections;

        public SectionData(string name)
        {
            Name = name;
        }

        public void AddChild(string childName)
        {
            lock (_timingLock)
            {
                if (!_childSections.Contains(childName))
                    _childSections.Add(childName);
            }
        }

        public void Begin(int threadId, long timestamp, long gpuTimestamp, long allocBytes)
        {
            lock (_timingLock)
            {
                _activeGpuStarts[threadId] = gpuTimestamp;
                _activeAllocStarts[threadId] = allocBytes;
            }
        }

        public void End(int threadId, long timestamp, long gpuTimestamp, long allocBytes)
        {
            SectionTiming timing;
            lock (_timingLock)
            {
                long startGpu = _activeGpuStarts.GetValueOrDefault(threadId);
                long startAlloc = _activeAllocStarts.GetValueOrDefault(threadId);
                timing = new SectionTiming
                {
                    ThreadId = threadId,
                    StartTimestamp = timestamp,
                    GpuStartTimestamp = startGpu,
                    GpuEndTimestamp = gpuTimestamp,
                    AllocationDelta = allocBytes - startAlloc
                };
                _activeGpuStarts.Remove(threadId);
                _activeAllocStarts.Remove(threadId);
            }

            timing.Duration = timestamp - timing.StartTimestamp;
            lock (_timingLock)
            {
                _timings.Add(timing);
            }
        }

        public SectionStatistics ComputeStatistics()
        {
            lock (_timingLock)
            {
                if (_timings.Count == 0)
                    return new SectionStatistics(Name, 0, 0, 0, 0, 0, 0, 0, 0);

                double freq = Stopwatch.Frequency;
                double[] durations = _timings.Select(t => (t.Duration / freq) * 1000.0).ToArray();
                Array.Sort(durations);

                double sum = durations.Sum();
                double mean = sum / durations.Length;
                double variance = durations.Sum(d => (d - mean) * (d - mean)) / durations.Length;

                double gpuTotal = 0;
                if (_timings[0].GpuStartTimestamp != 0)
                {
                    gpuTotal = _timings.Sum(t => ((t.GpuEndTimestamp - t.GpuStartTimestamp) / freq) * 1000.0);
                }

                long totalAllocDelta = _timings.Sum(t => t.AllocationDelta);

                return new SectionStatistics(
                    name: Name,
                    count: durations.Length,
                    totalMs: sum,
                    averageMs: mean,
                    minMs: durations[0],
                    maxMs: durations[^1],
                    p95Ms: GetPercentile(durations, 95),
                    p99Ms: GetPercentile(durations, 99),
                    totalGpuMs: gpuTotal,
                    totalAllocatedBytes: totalAllocDelta,
                    stdDevMs: Math.Sqrt(variance)
                );
            }
        }

        public SectionSnapshot ToSnapshot()
        {
            var stats = ComputeStatistics();
            return new SectionSnapshot(
                name: Name,
                parent: ParentSection,
                children: _childSections.ToArray(),
                statistics: stats
            );
        }
    }

    private struct SectionTiming
    {
        public int ThreadId;
        public long StartTimestamp;
        public long Duration;
        public long GpuStartTimestamp;
        public long GpuEndTimestamp;
        public long AllocationDelta;
    }
}

/// <summary>
/// Provides GPU timestamp queries for profiler integration.
/// </summary>
public interface IGpuTimestampProvider
{
    /// <summary>Gets the current GPU timestamp.</summary>
    long GetCurrentTimestamp();

    /// <summary>Converts a GPU timestamp to milliseconds.</summary>
    double TimestampToMs(long timestamp);
}

/// <summary>
/// Tracks memory allocations for profiler section accounting.
/// </summary>
public interface IMemoryTracker
{
    /// <summary>Gets the current allocated bytes.</summary>
    long GetAllocatedBytes();
}

/// <summary>
/// Immutable snapshot of all profiling data at a point in time.
/// </summary>
public sealed class ProfilerSnapshot
{
    /// <summary>Snapshots of all sections.</summary>
    public IReadOnlyList<SectionSnapshot> Sections { get; }

    /// <summary>Raw frame time samples in milliseconds.</summary>
    public double[] FrameTimes { get; }

    /// <summary>Total frame count.</summary>
    public long FrameCount { get; }

    public ProfilerSnapshot(IReadOnlyList<SectionSnapshot> sections, double[] frameTimes, long frameCount)
    {
        Sections = sections;
        FrameTimes = frameTimes;
        FrameCount = frameCount;
    }
}

/// <summary>
/// Snapshot of a single profiling section.
/// </summary>
public sealed class SectionSnapshot
{
    /// <summary>Section name.</summary>
    public string Name { get; }

    /// <summary>Parent section name, or null.</summary>
    public string? Parent { get; }

    /// <summary>Child section names.</summary>
    public string[] Children { get; }

    /// <summary>Computed statistics for this section.</summary>
    public SectionStatistics Statistics { get; }

    public SectionSnapshot(string name, string? parent, string[] children, SectionStatistics statistics)
    {
        Name = name;
        Parent = parent;
        Children = children;
        Statistics = statistics;
    }
}

/// <summary>
/// Computed statistics for a profiling section.
/// </summary>
public sealed class SectionStatistics
{
    /// <summary>Section name.</summary>
    public string Name { get; }

    /// <summary>Number of times the section was entered.</summary>
    public int Count { get; }

    /// <summary>Total time spent in the section (ms).</summary>
    public double TotalMs { get; }

    /// <summary>Average time per invocation (ms).</summary>
    public double AverageMs { get; }

    /// <summary>Minimum time per invocation (ms).</summary>
    public double MinMs { get; }

    /// <summary>Maximum time per invocation (ms).</summary>
    public double MaxMs { get; }

    /// <summary>95th percentile time (ms).</summary>
    public double P95Ms { get; }

    /// <summary>99th percentile time (ms).</summary>
    public double P99Ms { get; }

    /// <summary>Total GPU time if tracked (ms).</summary>
    public double TotalGpuMs { get; }

    /// <summary>Total bytes allocated during section execution.</summary>
    public long TotalAllocatedBytes { get; }

    /// <summary>Standard deviation of section durations (ms).</summary>
    public double StdDevMs { get; }

    public SectionStatistics(string name, int count, double totalMs, double averageMs,
        double minMs, double maxMs, double p95Ms, double p99Ms,
        double totalGpuMs = 0, long totalAllocatedBytes = 0, double stdDevMs = 0)
    {
        Name = name;
        Count = count;
        TotalMs = totalMs;
        AverageMs = averageMs;
        MinMs = minMs;
        MaxMs = maxMs;
        P95Ms = p95Ms;
        P99Ms = p99Ms;
        TotalGpuMs = totalGpuMs;
        TotalAllocatedBytes = totalAllocatedBytes;
        StdDevMs = stdDevMs;
    }

    public override string ToString() =>
        $"{Name}: {Count} calls, {TotalMs:F2}ms total, {AverageMs:F4}ms avg, " +
        $"[{MinMs:F4}..{MaxMs:F4}], P95={P95Ms:F4}, P99={P99Ms:F4}";
}

/// <summary>
/// Frame time statistics.
/// </summary>
public sealed class FrameStatistics
{
    /// <summary>Average frame time (ms).</summary>
    public double Mean { get; }

    /// <summary>Minimum frame time (ms).</summary>
    public double Min { get; }

    /// <summary>Maximum frame time (ms).</summary>
    public double Max { get; }

    /// <summary>Median frame time (ms).</summary>
    public double Median { get; }

    /// <summary>95th percentile frame time (ms).</summary>
    public double P95 { get; }

    /// <summary>99th percentile frame time (ms).</summary>
    public double P99 { get; }

    /// <summary>Standard deviation of frame times (ms).</summary>
    public double StdDev { get; }

    /// <summary>Average frames per second.</summary>
    public double Fps { get; }

    public FrameStatistics(double mean, double min, double max, double median,
        double p95, double p99, double stdDev, double fps)
    {
        Mean = mean;
        Min = min;
        Max = max;
        Median = median;
        P95 = p95;
        P99 = p99;
        StdDev = stdDev;
        Fps = fps;
    }

    public override string ToString() =>
        $"FPS: {Fps:F1}, Frame: {Mean:F2}ms, Min: {Min:F2}ms, Max: {Max:F2}ms, " +
        $"P95: {P95:F2}ms, P99: {P99:F2}ms";
}

/// <summary>
/// RAII-style profiling scope that begins a section on construction
/// and ends it on disposal.
/// </summary>
public readonly struct ProfileScope : IDisposable
{
    private readonly Profiler? _profiler;

    /// <summary>
    /// Creates a new profiling scope.
    /// </summary>
    /// <param name="profiler">The profiler instance.</param>
    /// <param name="sectionName">The section name to profile.</param>
    public ProfileScope(Profiler profiler, string sectionName)
    {
        _profiler = profiler;
        profiler.BeginSection(sectionName);
    }

    public void Dispose()
    {
        _profiler?.EndSection();
    }
}

/// <summary>
/// Static helper for creating profiling scopes using a global profiler.
/// </summary>
public static class ProfilerGlobal
{
    [ThreadStatic]
    private static Profiler? _threadLocal;

    /// <summary>Gets or sets the profiler for the current thread.</summary>
    public static Profiler? Current
    {
        get => _threadLocal;
        set => _threadLocal = value;
    }

    /// <summary>
    /// Creates a profiling scope using the thread-local profiler.
    /// </summary>
    public static ProfileScope BeginScope(string sectionName)
    {
        if (_threadLocal is null)
            throw new InvalidOperationException("No profiler set for current thread.");
        return new ProfileScope(_threadLocal, sectionName);
    }

    /// <summary>
    /// Begins a section on the thread-local profiler.
    /// </summary>
    public static void Begin(string sectionName) => _threadLocal?.BeginSection(sectionName);

    /// <summary>
    /// Ends a section on the thread-local profiler.
    /// </summary>
    public static void End() => _threadLocal?.EndSection();
}

/// <summary>
/// Default memory tracker using GC.GetTotalMemory.
/// </summary>
public sealed class GcMemoryTracker : IMemoryTracker
{
    /// <inheritdoc/>
    public long GetAllocatedBytes() => GC.GetTotalMemory(false);
}

/// <summary>
/// Composite profiler that aggregates data from multiple threads.
/// </summary>
public sealed class CompositeProfiler : IDisposable
{
    private readonly Dictionary<int, Profiler> _profilers = new();
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Gets or creates a profiler for the specified thread.
    /// </summary>
    public Profiler GetProfiler(int threadId)
    {
        lock (_lock)
        {
            if (!_profilers.TryGetValue(threadId, out Profiler? profiler))
            {
                profiler = new Profiler();
                _profilers[threadId] = profiler;
            }
            return profiler;
        }
    }

    /// <summary>
    /// Gets the profiler for the current thread.
    /// </summary>
    public Profiler GetCurrentThreadProfiler() =>
        GetProfiler(Thread.CurrentThread.ManagedThreadId);

    /// <summary>
    /// Gets combined statistics from all thread profilers.
    /// </summary>
    public ProfilerSnapshot GetCombinedSnapshot()
    {
        lock (_lock)
        {
            var allSections = new Dictionary<string, SectionStatistics>();

            foreach (var profiler in _profilers.Values)
            {
                var snapshot = profiler.GetSnapshot();
                foreach (var section in snapshot.Sections)
                {
                    if (allSections.TryGetValue(section.Name, out SectionStatistics? existing))
                    {
                        allSections[section.Name] = MergeStatistics(existing, section.Statistics);
                    }
                    else
                    {
                        allSections[section.Name] = section.Statistics;
                    }
                }
            }

            var sections = allSections.Values.Select(s => new SectionSnapshot(
                s.Name, null, Array.Empty<string>(), s
            )).ToList();

            long totalFrames = _profilers.Values.Sum(p => p.FrameCount);

            return new ProfilerSnapshot(sections, Array.Empty<double>(), totalFrames);
        }
    }

    private static SectionStatistics MergeStatistics(SectionStatistics a, SectionStatistics b)
    {
        return new SectionStatistics(
            name: a.Name,
            count: a.Count + b.Count,
            totalMs: a.TotalMs + b.TotalMs,
            averageMs: (a.TotalMs + b.TotalMs) / Math.Max(1, a.Count + b.Count),
            minMs: Math.Min(a.MinMs, b.MinMs),
            maxMs: Math.Max(a.MaxMs, b.MaxMs),
            p95Ms: Math.Max(a.P95Ms, b.P95Ms),
            p99Ms: Math.Max(a.P99Ms, b.P99Ms),
            totalGpuMs: a.TotalGpuMs + b.TotalGpuMs,
            totalAllocatedBytes: a.TotalAllocatedBytes + b.TotalAllocatedBytes,
            stdDevMs: Math.Max(a.StdDevMs, b.StdDevMs)
        );
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        lock (_lock)
        {
            foreach (var profiler in _profilers.Values)
                profiler.Dispose();
            _profilers.Clear();
        }
    }
}

/// <summary>
/// Utility class for manual GPU timestamp queries via direct buffer writes.
/// </summary>
public sealed unsafe class GpuTimestampBuffer : IDisposable
{
    private long* _buffer;
    private int _capacity;
    private int _count;
    private bool _disposed;

    /// <summary>Number of timestamps recorded.</summary>
    public int Count => _count;

    /// <summary>
    /// Creates a GPU timestamp buffer with the specified capacity.
    /// </summary>
    /// <param name="capacity">Maximum number of timestamps.</param>
    public GpuTimestampBuffer(int capacity = 4096)
    {
        _capacity = capacity;
        _buffer = (long*)System.Runtime.InteropServices.Marshal.AllocHGlobal(capacity * sizeof(long)).ToPointer();
        _count = 0;
    }

    /// <summary>
    /// Records a timestamp in the buffer.
    /// </summary>
    /// <param name="timestamp">The GPU timestamp value.</param>
    public void Record(long timestamp)
    {
        if (_count >= _capacity)
            throw new InvalidOperationException("Timestamp buffer full.");
        _buffer[_count++] = timestamp;
    }

    /// <summary>
    /// Gets a span over the recorded timestamps.
    /// </summary>
    public Span<long> GetTimestamps()
    {
        return new Span<long>(_buffer, _count);
    }

    /// <summary>
    /// Computes the elapsed time between the first and last timestamp.
    /// </summary>
    public double GetElapsedMs(double frequency)
    {
        if (_count < 2)
            return 0;
        return ((_buffer[_count - 1] - _buffer[0]) / frequency) * 1000.0;
    }

    /// <summary>
    /// Resets the buffer without freeing memory.
    /// </summary>
    public void Reset() => _count = 0;

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        if (_buffer != null)
        {
            System.Runtime.InteropServices.Marshal.FreeHGlobal((IntPtr)_buffer);
            _buffer = null;
        }
    }
}

/// <summary>
/// Thread-safe profiling accumulator for collecting timing data over
/// long-running operations without holding locks during measurement.
/// </summary>
public sealed class ProfilingAccumulator
{
    private long _totalTicks;
    private long _count;
    private long _minTicks = long.MaxValue;
    private long _maxTicks;
    private readonly string _name;
    private readonly object _lock = new();

    /// <summary>Name of this accumulator.</summary>
    public string Name => _name;

    /// <summary>Number of recorded samples.</summary>
    public long Count => Interlocked.Read(ref _count);

    /// <summary>
    /// Creates a new accumulator with the given name.
    /// </summary>
    public ProfilingAccumulator(string name)
    {
        _name = name ?? throw new ArgumentNullException(nameof(name));
    }

    /// <summary>
    /// Records a duration in ticks.
    /// </summary>
    public void Record(long ticks)
    {
        Interlocked.Add(ref _totalTicks, ticks);
        Interlocked.Increment(ref _count);

        lock (_lock)
        {
            if (ticks < _minTicks)
                _minTicks = ticks;
            if (ticks > _maxTicks)
                _maxTicks = ticks;
        }
    }

    /// <summary>
    /// Records a duration from a Stopwatch.
    /// </summary>
    public void Record(Stopwatch stopwatch) => Record(stopwatch.ElapsedTicks);

    /// <summary>
    /// Records a duration in milliseconds.
    /// </summary>
    public void RecordMs(double ms) => Record((long)(ms * Stopwatch.Frequency / 1000.0));

    /// <summary>
    /// Gets the average duration in milliseconds.
    /// </summary>
    public double GetAverageMs()
    {
        long count = Interlocked.Read(ref _count);
        if (count == 0)
            return 0;
        long total = Interlocked.Read(ref _totalTicks);
        return ((double)total / count / Stopwatch.Frequency) * 1000.0;
    }

    /// <summary>
    /// Gets the minimum recorded duration in milliseconds.
    /// </summary>
    public double GetMinMs()
    {
        lock (_lock)
            return _minTicks == long.MaxValue ? 0 : (_minTicks / (double)Stopwatch.Frequency) * 1000.0;
    }

    /// <summary>
    /// Gets the maximum recorded duration in milliseconds.
    /// </summary>
    public double GetMaxMs()
    {
        lock (_lock)
            return (_maxTicks / (double)Stopwatch.Frequency) * 1000.0;
    }

    /// <summary>
    /// Resets all accumulated data.
    /// </summary>
    public void Reset()
    {
        Interlocked.Exchange(ref _totalTicks, 0);
        Interlocked.Exchange(ref _count, 0);
        lock (_lock)
        {
            _minTicks = long.MaxValue;
            _maxTicks = 0;
        }
    }

    public override string ToString() =>
        $"{_name}: {Count} samples, avg={GetAverageMs():F4}ms, min={GetMinMs():F4}ms, max={GetMaxMs():F4}ms";
}

/// <summary>
/// Provides disposable timers for measuring code blocks, writing
/// results to a callback on disposal.
/// </summary>
public sealed class MeasureBlock : IDisposable
{
    private readonly Stopwatch _stopwatch;
    private readonly Action<double> _onComplete;
    private bool _disposed;

    /// <summary>
    /// Creates a measurement block that invokes the callback with elapsed ms on dispose.
    /// </summary>
    /// <param name="onComplete">Callback receiving elapsed time in milliseconds.</param>
    public MeasureBlock(Action<double> onComplete)
    {
        _onComplete = onComplete ?? throw new ArgumentNullException(nameof(onComplete));
        _stopwatch = Stopwatch.StartNew();
    }

    /// <summary>
    /// Gets the elapsed time so far without stopping the measurement.
    /// </summary>
    public double ElapsedMs => _stopwatch.Elapsed.TotalMilliseconds;

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _stopwatch.Stop();
        _onComplete(_stopwatch.Elapsed.TotalMilliseconds);
    }
}
