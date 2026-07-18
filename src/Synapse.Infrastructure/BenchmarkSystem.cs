// =============================================================================
// BenchmarkSystem.cs - G-DNN Engine: Performance Benchmarking
// GDNN.Engine - GDNN.Diagnostics
// Comprehensive benchmark system for profiling rendering performance
// =============================================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;

namespace GDNN.Diagnostics
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    /// <summary>Benchmark test types.</summary>
    public enum BenchmarkTestType
    {
        FrameTime,
        GPUTime,
        CPUTime,
        MemoryAllocation,
        DrawCallThroughput,
        TriangleThroughput,
        ShaderCompilation,
        TextureStreaming,
        SceneLoading,
        LODTransition,
        PostProcessing
    }

    /// <summary>Benchmark quality preset.</summary>
    public enum BenchmarkPreset
    {
        Quick,
        Standard,
        Extended,
        Custom
    }

    /// <summary>Thermal throttling state.</summary>
    public enum ThrottleState
    {
        None,
        Warning,
        Critical,
        Throttled
    }

    // =========================================================================
    // DATA STRUCTURES
    // =========================================================================

    /// <summary>Single benchmark sample.</summary>
    [DebuggerDisplay("Sample: {TestType} = {Value:F3}{Unit} ({Timestamp})")]
    public class BenchmarkSample
    {
        public BenchmarkTestType TestType { get; set; }
        public string TestName { get; set; } = "";
        public float Value { get; set; }
        public string Unit { get; set; } = "ms";
        public long Timestamp { get; set; }
        public int FrameNumber { get; set; }
        public float GpuTimeMs { get; set; }
        public float CpuTimeMs { get; set; }
        public long MemoryBytes { get; set; }
        public int DrawCalls { get; set; }
        public int TriangleCount { get; set; }
        public int TriangleCountK { get; set; }
        public int InstanceCount { get; set; }
        public int TextureBindings { get; set; }
        public int ShaderSwitches { get; set; }
    }

    /// <summary>Aggregated statistics for a benchmark run.</summary>
    [DebuggerDisplay("Stats: Avg={AverageMs:F2}ms, Min={MinMs:F2}ms, Max={MaxMs:F2}ms, P99={P99Ms:F2}ms")]
    public class BenchmarkStats
    {
        public string TestName { get; set; } = "";
        public BenchmarkTestType TestType { get; set; }
        public int SampleCount { get; set; }
        public float AverageMs { get; set; }
        public float MinMs { get; set; }
        public float MaxMs { get; set; }
        public float MedianMs { get; set; }
        public float P95Ms { get; set; }
        public float P99Ms { get; set; }
        public float StandardDeviation { get; set; }
        public float Percentile1Ms { get; set; }
        public float Percentile5Ms { get; set; }
        public float Fps { get; set; }
        public float FrameTimeMs { get; set; }
        public float FrameTimeVariance { get; set; }
        public int TotalDrawCalls { get; set; }
        public long TotalTriangles { get; set; }
        public long TotalMemoryBytes { get; set; }
        public float AverageGpuTimeMs { get; set; }
        public float AverageCpuTimeMs { get; set; }
    }

    /// <summary>Complete benchmark report.</summary>
    public class BenchmarkReport
    {
        public string ReportName { get; set; } = "";
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public TimeSpan Duration => EndTime - StartTime;
        public BenchmarkPreset Preset { get; set; }
        public string SystemInfo { get; set; } = "";
        public List<BenchmarkStats> Results { get; set; } = new();
        public List<BenchmarkSample> AllSamples { get; set; } = new();
        public string Summary { get; set; } = "";
        public float OverallScore { get; set; }
    }

    /// <summary>Configuration for a benchmark run.</summary>
    public class BenchmarkConfig
    {
        public BenchmarkPreset Preset { get; set; } = BenchmarkPreset.Standard;
        public int WarmupFrames { get; set; } = 60;
        public int TestFrames { get; set; } = 300;
        public int CoolDownFrames { get; set; } = 30;
        public bool RecordGpuTiming { get; set; } = true;
        public bool RecordCpuTiming { get; set; } = true;
        public bool RecordMemory { get; set; } = true;
        public bool RecordDrawCalls { get; set; } = true;
        public bool AutoDetectThrottling { get; set; } = true;
        public float ThrottleTemperatureThreshold { get; set; } = 85.0f;
        public List<BenchmarkTestType> TestsToRun { get; set; } = new()
        {
            BenchmarkTestType.FrameTime,
            BenchmarkTestType.GPUTime,
            BenchmarkTestType.DrawCallThroughput,
            BenchmarkTestType.TriangleThroughput,
            BenchmarkTestType.PostProcessing
        };
    }

    /// <summary>System information snapshot.</summary>
    public class SystemInfo
    {
        public string CpuName { get; set; } = "Unknown CPU";
        public int CpuCoreCount { get; set; }
        public long TotalMemoryBytes { get; set; }
        public string GpuName { get; set; } = "Unknown GPU";
        public long GpuMemoryBytes { get; set; }
        public string DriverVersion { get; set; } = "";
        public string OSVersion { get; set; } = "";
        public string RendererVersion { get; set; } = "";
    }

    // =========================================================================
    // FRAME TIMER
    // =========================================================================

    /// <summary>
    /// High-precision frame timing using Stopwatch and GPU timestamp queries.
    /// </summary>
    public class FrameTimer
    {
        private readonly Stopwatch _cpuTimer = new();
        private readonly Stopwatch _frameTimer = new();
        private long _lastGpuTimestamp;
        private float _cpuFrameTimeMs;
        private float _gpuFrameTimeMs;
        private int _frameCount;
        private float _fps;
        private float _fpsAccumulator;
        private int _fpsFrameCount;
        private float _fpsUpdateTimer;
        private readonly Queue<float> _frameTimes = new(120);

        public float CpuFrameTimeMs => _cpuFrameTimeMs;
        public float GpuFrameTimeMs => _gpuFrameTimeMs;
        public float TotalFrameTimeMs => _cpuFrameTimeMs + _gpuFrameTimeMs;
        public float Fps => _fps;
        public int FrameCount => _frameCount;

        public void BeginFrame()
        {
            _frameTimer.Restart();
            _cpuTimer.Restart();
        }

        public void EndFrame()
        {
            _cpuTimer.Stop();
            _cpuFrameTimeMs = (float)_cpuTimer.Elapsed.TotalMilliseconds;
            _frameCount++;

            _frameTimes.Enqueue(_cpuFrameTimeMs);
            if (_frameTimes.Count > 120) _frameTimes.Dequeue();

            _fpsAccumulator += _cpuFrameTimeMs;
            _fpsFrameCount++;

            if (_fpsAccumulator >= 1000.0f)
            {
                _fps = _fpsFrameCount / (_fpsAccumulator / 1000.0f);
                _fpsFrameCount = 0;
                _fpsAccumulator = 0;
            }
        }

        public void ReportGpuTime(float gpuMs)
        {
            _gpuFrameTimeMs = gpuMs;
        }

        public float GetAverageFrameTime()
        {
            return _frameTimes.Count > 0 ? _frameTimes.Average() : 0;
        }

        public float GetMedianFrameTime()
        {
            if (_frameTimes.Count == 0) return 0;
            var sorted = _frameTimes.OrderBy(x => x).ToList();
            return sorted[sorted.Count / 2];
        }

        public float GetPercentile(float percentile)
        {
            if (_frameTimes.Count == 0) return 0;
            var sorted = _frameTimes.OrderBy(x => x).ToList();
            int idx = (int)(sorted.Count * percentile / 100.0f);
            return sorted[Math.Min(idx, sorted.Count - 1)];
        }

        public float Get1PercentLow()
        {
            return GetPercentile(99);
        }

        public void Reset()
        {
            _frameCount = 0;
            _fps = 0;
            _fpsAccumulator = 0;
            _fpsFrameCount = 0;
            _frameTimes.Clear();
        }
    }

    // =========================================================================
    // BENCHMARK RUNNER
    // =========================================================================

    /// <summary>
    /// Main benchmark execution engine. Runs tests, collects samples,
    /// and generates comprehensive performance reports.
    /// </summary>
    public class BenchmarkRunner
    {
        private readonly FrameTimer _frameTimer = new();
        private readonly ConcurrentBag<BenchmarkSample> _samples = new();
        private readonly BenchmarkConfig _config;
        private BenchmarkState _state = BenchmarkState.Idle;
        private int _currentFrame;
        private int _warmupRemaining;
        private int _testRemaining;
        private int _cooldownRemaining;
        private BenchmarkTestType _currentTest;
        private readonly List<BenchmarkStats> _results = new();
        private ThrottleState _throttleState = ThrottleState.None;
        private float _peakTemperature;
        private readonly System.Diagnostics.Process _currentProcess = System.Diagnostics.Process.GetCurrentProcess();

        public BenchmarkState State => _state;
        public float Progress => _config.TestFrames > 0 ? 1.0f - (_testRemaining + _cooldownRemaining) / (float)_config.TestFrames : 0;
        public ThrottleState ThrottleState => _throttleState;
        public float PeakTemperature => _peakTemperature;

        public event Action<BenchmarkSample>? OnSampleRecorded;
        public event Action<BenchmarkState>? OnStateChanged;
        public event Action<BenchmarkReport>? OnBenchmarkComplete;

        public BenchmarkRunner(BenchmarkConfig? config = null)
        {
            _config = config ?? new BenchmarkConfig();
        }

        public void Start(string reportName = "Benchmark")
        {
            _state = BenchmarkState.Warmup;
            _currentFrame = 0;
            _warmupRemaining = _config.WarmupFrames;
            _testRemaining = _config.TestFrames;
            _cooldownRemaining = _config.CoolDownFrames;
            _results.Clear();
            while (_samples.TryTake(out _)) { }
            _frameTimer.Reset();
            OnStateChanged?.Invoke(_state);
        }

        public void RecordFrame()
        {
            if (_state == BenchmarkState.Idle) return;

            _frameTimer.EndFrame();

            switch (_state)
            {
                case BenchmarkState.Warmup:
                    _warmupRemaining--;
                    if (_warmupRemaining <= 0)
                    {
                        _state = BenchmarkState.Running;
                        _currentTest = _config.TestsToRun.Count > 0 ? _config.TestsToRun[0] : BenchmarkTestType.FrameTime;
                        OnStateChanged?.Invoke(_state);
                    }
                    break;

                case BenchmarkState.Running:
                    _currentFrame++;
                    _testRemaining--;
                    RecordSample();
                    UpdateThrottling();
                    if (_testRemaining <= 0)
                    {
                        _state = BenchmarkState.Cooldown;
                        OnStateChanged?.Invoke(_state);
                    }
                    break;

                case BenchmarkState.Cooldown:
                    _cooldownRemaining--;
                    if (_cooldownRemaining <= 0)
                    {
                        _state = BenchmarkState.Complete;
                        OnStateChanged?.Invoke(_state);
                        GenerateReport();
                    }
                    break;
            }

            _frameTimer.BeginFrame();
        }

        private void RecordSample()
        {
            var sample = new BenchmarkSample
            {
                TestType = _currentTest,
                TestName = _currentTest.ToString(),
                Timestamp = Stopwatch.GetTimestamp(),
                FrameNumber = _currentFrame,
                Value = _frameTimer.CpuFrameTimeMs,
                Unit = "ms",
                CpuTimeMs = _frameTimer.CpuFrameTimeMs,
                GpuTimeMs = _frameTimer.GpuFrameTimeMs,
                MemoryBytes = _currentProcess.WorkingSet64,
                DrawCalls = 0,
                TriangleCount = 0
            };

            _samples.Add(sample);
            OnSampleRecorded?.Invoke(sample);
        }

        private void UpdateThrottling()
        {
            if (!_config.AutoDetectThrottling) return;

            float avgFrameTime = _frameTimer.GetAverageFrameTime();
            float targetFrameTime = 1000.0f / 60.0f;

            if (avgFrameTime > targetFrameTime * 1.5f)
                _throttleState = ThrottleState.Warning;
            else if (avgFrameTime > targetFrameTime * 2.0f)
                _throttleState = ThrottleState.Critical;
            else if (avgFrameTime > targetFrameTime * 3.0f)
                _throttleState = ThrottleState.Throttled;
            else
                _throttleState = ThrottleState.None;
        }

        private void GenerateReport()
        {
            var samples = _samples.ToArray();
            var grouped = samples.GroupBy(s => s.TestType);

            foreach (var group in grouped)
            {
                var testSamples = group.OrderBy(s => s.Value).ToList();
                int count = testSamples.Count;
                if (count == 0) continue;

                float avg = testSamples.Average(s => s.Value);
                float min = testSamples.First().Value;
                float max = testSamples.Last().Value;
                float median = testSamples[count / 2].Value;
                float p95 = testSamples[(int)(count * 0.95f)].Value;
                float p99 = testSamples[(int)(count * 0.99f)].Value;
                float p1 = testSamples[(int)(count * 0.01f)].Value;
                float p5 = testSamples[(int)(count * 0.05f)].Value;
                float variance = testSamples.Sum(s => (s.Value - avg) * (s.Value - avg)) / count;

                var stats = new BenchmarkStats
                {
                    TestType = group.Key,
                    TestName = group.Key.ToString(),
                    SampleCount = count,
                    AverageMs = avg,
                    MinMs = min,
                    MaxMs = max,
                    MedianMs = median,
                    P95Ms = p95,
                    P99Ms = p99,
                    StandardDeviation = MathF.Sqrt(variance),
                    Percentile1Ms = p1,
                    Percentile5Ms = p5,
                    Fps = avg > 0 ? 1000.0f / avg : 0,
                    FrameTimeMs = avg,
                    FrameTimeVariance = variance,
                    TotalDrawCalls = testSamples.Sum(s => s.DrawCalls),
                    TotalTriangles = testSamples.Sum(s => s.TriangleCount),
                    TotalMemoryBytes = testSamples.Max(s => s.MemoryBytes),
                    AverageGpuTimeMs = testSamples.Average(s => s.GpuTimeMs),
                    AverageCpuTimeMs = testSamples.Average(s => s.CpuTimeMs)
                };

                _results.Add(stats);
            }

            float overallScore = CalculateOverallScore(_results);

            var report = new BenchmarkReport
            {
                ReportName = $"Benchmark_{DateTime.Now:yyyyMMdd_HHmmss}",
                StartTime = DateTime.Now,
                EndTime = DateTime.Now,
                Preset = _config.Preset,
                Results = _results,
                AllSamples = samples.ToList(),
                OverallScore = overallScore,
                Summary = GenerateSummary(_results, overallScore)
            };

            OnBenchmarkComplete?.Invoke(report);
        }

        private float CalculateOverallScore(List<BenchmarkStats> results)
        {
            if (results.Count == 0) return 0;

            float fpsScore = 0;
            var frameTimeStats = results.FirstOrDefault(r => r.TestType == BenchmarkTestType.FrameTime);
            if (frameTimeStats != null && frameTimeStats.Fps > 0)
            {
                fpsScore = MathF.Min(100, frameTimeStats.Fps / 60.0f * 100);
            }

            float consistencyScore = 0;
            if (frameTimeStats != null && frameTimeStats.AverageMs > 0)
            {
                float jitter = frameTimeStats.StandardDeviation / frameTimeStats.AverageMs;
                consistencyScore = MathF.Max(0, 100 - jitter * 100);
            }

            float p99Score = 0;
            if (frameTimeStats != null && frameTimeStats.P99Ms > 0)
            {
                p99Score = MathF.Max(0, MathF.Min(100, 16.67f / frameTimeStats.P99Ms * 100));
            }

            return fpsScore * 0.5f + consistencyScore * 0.3f + p99Score * 0.2f;
        }

        private string GenerateSummary(List<BenchmarkStats> results, float score)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"=== BENCHMARK SUMMARY ===");
            sb.AppendLine($"Overall Score: {score:F1}/100");
            sb.AppendLine();

            foreach (var r in results)
            {
                sb.AppendLine($"[{r.TestName}]");
                sb.AppendLine($"  Average: {r.AverageMs:F2} ms ({r.Fps:F1} FPS)");
                sb.AppendLine($"  Min: {r.MinMs:F2} ms | Max: {r.MaxMs:F2} ms");
                sb.AppendLine($"  Median: {r.MedianMs:F2} ms | P99: {r.P99Ms:F2} ms");
                sb.AppendLine($"  StdDev: {r.StandardDeviation:F2} ms");
                sb.AppendLine();
            }

            sb.AppendLine($"Throttle State: {_throttleState}");
            sb.AppendLine($"Peak Temperature: {_peakTemperature:F1}°C");
            return sb.ToString();
        }

        public BenchmarkReport? GetLastReport()
        {
            return _results.Count > 0 ? new BenchmarkReport
            {
                ReportName = $"Benchmark_{DateTime.Now:yyyyMMdd_HHmmss}",
                Preset = _config.Preset,
                Results = new List<BenchmarkStats>(_results),
                OverallScore = CalculateOverallScore(_results)
            } : null;
        }

        public void Reset()
        {
            _state = BenchmarkState.Idle;
            _results.Clear();
            while (_samples.TryTake(out _)) { }
            _frameTimer.Reset();
        }
    }

    public enum BenchmarkState
    {
        Idle,
        Warmup,
        Running,
        Cooldown,
        Complete
    }

    // =========================================================================
    // BENCHMARK REPORT EXPORTER
    // =========================================================================

    /// <summary>
    /// Exports benchmark reports to various formats (JSON, CSV, text).
    /// </summary>
    public static class BenchmarkExporter
    {
        public static string ToJson(BenchmarkReport report, bool indented = true)
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = indented,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
            return JsonSerializer.Serialize(report, options);
        }

        public static void SaveToFile(BenchmarkReport report, string filePath)
        {
            string ext = Path.GetExtension(filePath).ToLowerInvariant();
            string content = ext switch
            {
                ".json" => ToJson(report),
                ".csv" => ToCsv(report),
                _ => report.Summary
            };
            File.WriteAllText(filePath, content);
        }

        public static string ToCsv(BenchmarkReport report)
        {
            var sb = new StringBuilder();
            sb.AppendLine("TestType,AverageMs,MinMs,MaxMs,MedianMs,P95Ms,P99Ms,Fps,SampleCount");
            foreach (var r in report.Results)
            {
                sb.AppendLine($"{r.TestType},{r.AverageMs:F3},{r.MinMs:F3},{r.MaxMs:F3},{r.MedianMs:F3},{r.P95Ms:F3},{r.P99Ms:F3},{r.Fps:F1},{r.SampleCount}");
            }
            return sb.ToString();
        }
    }
}