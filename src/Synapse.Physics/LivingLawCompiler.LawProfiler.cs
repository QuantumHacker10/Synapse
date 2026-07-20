// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    /// <summary>Profiles law compilation and execution operations.</summary>
    public sealed class LawProfiler
    {
        private readonly ConcurrentDictionary<string, ProfileData> _profiles = new();
        private bool _enabled = true;

        public bool Enabled { get => _enabled; set => _enabled = value; }

        public IReadOnlyDictionary<string, ProfileData> Profiles => _profiles;

        /// <summary>Start timing an operation.</summary>
        public Stopwatch StartProfile(string operationName)
        {
            if (!_enabled)
                return Stopwatch.StartNew();
            return Stopwatch.StartNew();
        }

        /// <summary>Stop timing and record the result.</summary>
        public void StopProfile(string operationName, Stopwatch sw, long allocatedBytes = 0)
        {
            if (!_enabled)
                return;
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;

            var profile = _profiles.GetOrAdd(operationName, _ => new ProfileData { OperationName = operationName });
            lock (profile)
            {
                profile.CallCount++;
                profile.TotalMilliseconds += ms;
                profile.LastMilliseconds = ms;
                profile.TotalBytes += allocatedBytes;
                if (ms < profile.MinMilliseconds)
                    profile.MinMilliseconds = ms;
                if (ms > profile.MaxMilliseconds)
                    profile.MaxMilliseconds = ms;
            }
        }

        /// <summary>Profile a compiled action.</summary>
        public void ProfileAction(string operationName, Action action)
        {
            var sw = StartProfile(action.Method.Name);
            action();
            StopProfile(operationName, sw);
        }

        /// <summary>Profile a compiled function.</summary>
        public T ProfileFunction<T>(string operationName, Func<T> func)
        {
            var sw = StartProfile(operationName);
            T result = func();
            StopProfile(operationName, sw);
            return result;
        }

        /// <summary>Reset all profiling data.</summary>
        public void Reset()
        {
            _profiles.Clear();
        }

        /// <summary>Generate a profiling report.</summary>
        public string GenerateReport()
        {
            var sb = new StringBuilder();
            sb.AppendLine("=== Law Profiler Report ===");
            sb.AppendLine($"Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss UTC}");
            sb.AppendLine($"Total operations: {_profiles.Count}");
            sb.AppendLine();
            sb.AppendLine($"{"Operation",-30} {"Count",8} {"Total(ms)",12} {"Avg(ms)",10} {"Min(ms)",10} {"Max(ms)",10}");
            sb.AppendLine(new string('-', 82));

            foreach (var kv in _profiles.OrderByDescending(p => p.Value.TotalMilliseconds))
            {
                var p = kv.Value;
                sb.AppendLine($"{p.OperationName,-30} {p.CallCount,8} {p.TotalMilliseconds,12:F2} {p.AverageMilliseconds,10:F4} {p.MinMilliseconds,10:F4} {p.MaxMilliseconds,10:F4}");
            }

            sb.AppendLine();
            long totalBytes = _profiles.Values.Sum(p => p.TotalBytes);
            sb.AppendLine($"Total memory allocated: {totalBytes:N0} bytes");
            double totalTime = _profiles.Values.Sum(p => p.TotalMilliseconds);
            sb.AppendLine($"Total time: {totalTime:F2} ms");

            return sb.ToString();
        }

        /// <summary>Get the top N most time-consuming operations.</summary>
        public IReadOnlyList<ProfileData> GetTopOperations(int count = 10)
        {
            return _profiles.Values.OrderByDescending(p => p.TotalMilliseconds).Take(count).ToList();
        }

        /// <summary>Get the top N most frequently called operations.</summary>
        public IReadOnlyList<ProfileData> GetMostFrequent(int count = 10)
        {
            return _profiles.Values.OrderByDescending(p => p.CallCount).Take(count).ToList();
        }
    }
}
