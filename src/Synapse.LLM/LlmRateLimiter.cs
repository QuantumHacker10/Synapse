// Multi-provider LLM pipeline for Synapse (split from HybridLlmRouter.cs).

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using System.Timers;
using GDNN.Scene;

#nullable enable

namespace GDNN.Llm
{

    /// <summary>
    /// Advanced rate limiter for LLM requests supporting token bucket, sliding window,
    /// priority queuing, per-provider limits, and graceful degradation.
    /// </summary>
    public sealed class LlmRateLimiter
    {
        private readonly ConcurrentDictionary<string, TokenBucketState> _tokenBuckets;
        private readonly ConcurrentDictionary<string, SlidingWindowState> _slidingWindows;
        private readonly ConcurrentDictionary<string, PriorityQueue<RateLimitedRequest, int>> _priorityQueues;
        private readonly RateLimitConfig _globalConfig;
        private readonly int _globalMaxConcurrent;
        private int _globalActiveRequests;
        private readonly object _globalLock = new();

        /// <summary>Number of requests currently in flight globally.</summary>
        public int GlobalActiveRequests => _globalActiveRequests;

        /// <summary>
        /// Initializes a new rate limiter.
        /// </summary>
        /// <param name="globalConfig">Global rate limit configuration.</param>
        public LlmRateLimiter(RateLimitConfig? globalConfig = null)
        {
            _globalConfig = globalConfig ?? new RateLimitConfig();
            _globalMaxConcurrent = _globalConfig.MaxConcurrent;
            _tokenBuckets = new ConcurrentDictionary<string, TokenBucketState>(StringComparer.OrdinalIgnoreCase);
            _slidingWindows = new ConcurrentDictionary<string, SlidingWindowState>(StringComparer.OrdinalIgnoreCase);
            _priorityQueues = new ConcurrentDictionary<string, PriorityQueue<RateLimitedRequest, int>>(StringComparer.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Checks whether a request to the specified provider is allowed.
        /// </summary>
        /// <param name="providerName">Provider name.</param>
        /// <param name="priority">Request priority.</param>
        /// <param name="estimatedTokens">Estimated tokens for this request.</param>
        /// <returns>True if the request is allowed.</returns>
        public bool IsAllowed(string providerName, RequestPriority priority, int estimatedTokens = 1)
        {
            lock (_globalLock)
            {
                if (_globalActiveRequests >= _globalMaxConcurrent)
                    return false;
            }

            if (_tokenBuckets.TryGetValue(providerName, out var bucket))
            {
                if (!bucket.TryConsume(estimatedTokens))
                    return false;
            }

            if (_slidingWindows.TryGetValue(providerName, out var window))
            {
                if (!window.TryRecord(DateTimeOffset.UtcNow))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Attempts to acquire a rate limit slot. Returns false if rate limited.
        /// </summary>
        /// <param name="providerName">Provider name.</param>
        /// <param name="priority">Request priority.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>True if slot acquired.</returns>
        public async Task<bool> AcquireAsync(
            string providerName,
            RequestPriority priority,
            CancellationToken cancellationToken = default)
        {
            var maxWait = TimeSpan.FromSeconds(30);
            var stopwatch = Stopwatch.StartNew();

            while (stopwatch.Elapsed < maxWait)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (IsAllowed(providerName, priority))
                {
                    Interlocked.Increment(ref _globalActiveRequests);
                    return true;
                }

                await Task.Delay(100, cancellationToken);
            }

            return false;
        }

        /// <summary>
        /// Releases a previously acquired rate limit slot.
        /// </summary>
        /// <param name="providerName">Provider name.</param>
        public void Release(string providerName)
        {
            Interlocked.Decrement(ref _globalActiveRequests);
        }

        /// <summary>
        /// Configures per-provider rate limits.
        /// </summary>
        /// <param name="providerName">Provider name.</param>
        /// <param name="config">Rate limit configuration.</param>
        public void ConfigureProvider(string providerName, RateLimitConfig config)
        {
            _tokenBuckets[providerName] = new TokenBucketState(
                config.TokensPerMinute, config.BurstSize, TimeSpan.FromMinutes(1));

            _slidingWindows[providerName] = new SlidingWindowState(
                config.RequestsPerMinute, TimeSpan.FromMinutes(1));
        }

        /// <summary>
        /// Gets the estimated wait time before a request would be allowed.
        /// </summary>
        /// <param name="providerName">Provider name.</param>
        /// <returns>Estimated wait time.</returns>
        public TimeSpan GetWaitTime(string providerName)
        {
            if (_tokenBuckets.TryGetValue(providerName, out var bucket))
                return bucket.EstimateWaitTime(1);
            return TimeSpan.Zero;
        }

        /// <summary>
        /// Resets rate limit state for a provider.
        /// </summary>
        /// <param name="providerName">Provider name.</param>
        public void ResetProvider(string providerName)
        {
            _tokenBuckets.TryRemove(providerName, out _);
            _slidingWindows.TryRemove(providerName, out _);
        }

        /// <summary>
        /// Gets current rate limit status for all providers.
        /// </summary>
        public IReadOnlyDictionary<string, RateLimitStatus> GetStatus()
        {
            var status = new Dictionary<string, RateLimitStatus>(StringComparer.OrdinalIgnoreCase);

            foreach (var (name, bucket) in _tokenBuckets)
            {
                status[name] = new RateLimitStatus
                {
                    ProviderName = name,
                    RemainingTokens = bucket.RemainingTokens,
                    TokensPerMinute = bucket.Capacity,
                    IsThrottled = bucket.RemainingTokens <= 0
                };
            }

            return status;
        }
    }

    /// <summary>
    /// Token bucket state for rate limiting.
    /// </summary>
    internal sealed class TokenBucketState
    {
        private readonly int _refillRate;
        private readonly int _capacity;
        private readonly TimeSpan _refillInterval;
        private double _tokens;
        private DateTimeOffset _lastRefill;
        private readonly object _lock = new();

        public int Capacity => _capacity;
        public int RemainingTokens => (int)_tokens;

        public TokenBucketState(int refillRate, int capacity, TimeSpan refillInterval)
        {
            _refillRate = refillRate;
            _capacity = capacity;
            _refillInterval = refillInterval;
            _tokens = capacity;
            _lastRefill = DateTimeOffset.UtcNow;
        }

        public bool TryConsume(int tokens)
        {
            lock (_lock)
            {
                Refill();
                if (_tokens >= tokens)
                {
                    _tokens -= tokens;
                    return true;
                }
                return false;
            }
        }

        public TimeSpan EstimateWaitTime(int tokens)
        {
            lock (_lock)
            {
                Refill();
                if (_tokens >= tokens) return TimeSpan.Zero;
                double deficit = tokens - _tokens;
                double seconds = deficit / _refillRate * _refillInterval.TotalSeconds;
                return TimeSpan.FromSeconds(Math.Ceiling(seconds));
            }
        }

        private void Refill()
        {
            var now = DateTimeOffset.UtcNow;
            var elapsed = now - _lastRefill;
            var intervals = (int)(elapsed.TotalMilliseconds / _refillInterval.TotalMilliseconds);
            if (intervals > 0)
            {
                _tokens = Math.Min(_capacity, _tokens + intervals * _refillRate);
                _lastRefill = now;
            }
        }
    }

    /// <summary>
    /// Sliding window rate limit state.
    /// </summary>
    internal sealed class SlidingWindowState
    {
        private readonly int _maxRequests;
        private readonly TimeSpan _windowSize;
        private readonly Queue<DateTimeOffset> _timestamps;
        private readonly object _lock = new();

        public SlidingWindowState(int maxRequests, TimeSpan windowSize)
        {
            _maxRequests = maxRequests;
            _windowSize = windowSize;
            _timestamps = new Queue<DateTimeOffset>();
        }

        public bool TryRecord(DateTimeOffset now)
        {
            lock (_lock)
            {
                var cutoff = now - _windowSize;
                while (_timestamps.Count > 0 && _timestamps.Peek() < cutoff)
                    _timestamps.Dequeue();

                if (_timestamps.Count < _maxRequests)
                {
                    _timestamps.Enqueue(now);
                    return true;
                }
                return false;
            }
        }
    }

    /// <summary>
    /// Rate limit status for a provider.
    /// </summary>
    public record RateLimitStatus
    {
        /// <summary>Provider name.</summary>
        public string ProviderName { get; init; } = string.Empty;

        /// <summary>Remaining tokens in the bucket.</summary>
        public int RemainingTokens { get; init; }

        /// <summary>Total token capacity.</summary>
        public int TokensPerMinute { get; init; }

        /// <summary>Whether the provider is currently throttled.</summary>
        public bool IsThrottled { get; init; }
    }

    /// <summary>
    /// A rate-limited request with priority.
    /// </summary>
    internal sealed class RateLimitedRequest
    {
        public string ProviderName { get; set; } = string.Empty;
        public RequestPriority Priority { get; set; }
        public int EstimatedTokens { get; set; }
        public DateTimeOffset QueuedAt { get; set; } = DateTimeOffset.UtcNow;
        public TaskCompletionSource<bool>? CompletionSource { get; set; }
    }
}
