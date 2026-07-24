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

}
