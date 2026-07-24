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

}
