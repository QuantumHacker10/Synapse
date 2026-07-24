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
using Synapse.Infrastructure.Logging;

#nullable enable

namespace GDNN.Llm
{

    /// <summary>
    /// Core router that intelligently directs LLM requests to the best provider
    /// based on complexity, cost, latency, privacy, and provider health. Implements
    /// circuit breaker, rate limiting, cost tracking, load balancing, and fallback chains.
    /// </summary>
    public sealed class HybridLlmRouter : ILlmInferenceProvider
    {
        private readonly ConcurrentDictionary<string, ILlmInferenceProvider> _providers;
        private readonly ConcurrentDictionary<string, CircuitBreakerState> _circuitBreakers;
        private readonly ConcurrentDictionary<string, int> _consecutiveFailures;
        private readonly ConcurrentDictionary<string, DateTimeOffset> _circuitOpenTimestamps;
        private readonly ConcurrentDictionary<string, long> _requestCounts;
        private readonly ConcurrentDictionary<string, long> _roundRobinCounters;
        private readonly ConcurrentDictionary<string, ConcurrentQueue<DateTimeOffset>> _requestTimestamps;
        private readonly ConcurrentDictionary<string, TokenBucket> _tokenBuckets;
        private readonly ConcurrentDictionary<string, ProviderHealth> _healthCache;
        private readonly ConcurrentDictionary<string, ProviderConfig> _configs;
        private readonly ConcurrentDictionary<string, object> _providerLocks;
        private readonly List<string> _fallbackChain;
        private readonly LoadBalancingStrategy _loadBalancingStrategy;
        private readonly TokenBudget _globalBudget;
        private readonly LlmCostTracker _costTracker;
        private readonly LlmRateLimiter _rateLimiter;
        private readonly RetryConfig _retryConfig;
        private readonly object _fallbackLock = new();
        private int _roundRobinIndex;
        private volatile bool _disposed;
        private volatile bool _isPaused;

        /// <summary>Event raised when a provider's status changes.</summary>
        public event EventHandler<ProviderStatusChangedEventArgs>? ProviderStatusChanged;

        /// <summary>Event raised when a cost threshold is exceeded.</summary>
        public event EventHandler<CostAlertEventArgs>? CostAlert;

        /// <summary>Event raised when a request is routed.</summary>
        public event EventHandler<RequestRoutedEventArgs>? RequestRouted;

        /// <inheritdoc/>
        public string ProviderName => "HybridLlmRouter";

        /// <inheritdoc/>
        public bool RequiresNetwork => false;

        /// <inheritdoc/>
        public LlmStatus Status => _isPaused ? LlmStatus.Unavailable : LlmStatus.Ready;

        /// <inheritdoc/>
        public LlmProviderMode Mode => LlmProviderMode.Hybrid;

        /// <inheritdoc/>
        public IReadOnlyList<ModelInfo> AvailableModels =>
            _providers.Values.SelectMany(p => p.AvailableModels).ToList().AsReadOnly();

        /// <inheritdoc/>
        public ProviderHealth Health { get; private set; } = new() { ProviderName = "HybridLlmRouter" };

        /// <summary>Strategy used for load balancing across providers.</summary>
        public LoadBalancingStrategy LoadBalancing => _loadBalancingStrategy;

        /// <summary>The current fallback chain order.</summary>
        public IReadOnlyList<string> FallbackChain => _fallbackChain.AsReadOnly();

        /// <summary>Current global budget status.</summary>
        public TokenBudget CurrentBudget => _globalBudget;

        /// <summary>Number of registered providers.</summary>
        public int ProviderCount => _providers.Count;

        /// <summary>Gets all registered provider names.</summary>
        public IEnumerable<string> RegisteredProviders => _providers.Keys;

        /// <summary>
        /// Initializes a new instance of the HybridLlmRouter with default settings.
        /// </summary>
        public HybridLlmRouter()
            : this(new HybridLlmRouterConfig())
        {
        }

        /// <summary>
        /// Initializes a new instance of the HybridLlmRouter with the specified configuration.
        /// </summary>
        /// <param name="config">Router configuration.</param>
        /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
        public HybridLlmRouter(HybridLlmRouterConfig config)
        {
            _providers = new ConcurrentDictionary<string, ILlmInferenceProvider>(StringComparer.OrdinalIgnoreCase);
            _circuitBreakers = new ConcurrentDictionary<string, CircuitBreakerState>(StringComparer.OrdinalIgnoreCase);
            _consecutiveFailures = new ConcurrentDictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            _circuitOpenTimestamps = new ConcurrentDictionary<string, DateTimeOffset>(StringComparer.OrdinalIgnoreCase);
            _requestCounts = new ConcurrentDictionary<string, long>(StringComparer.OrdinalIgnoreCase);
            _roundRobinCounters = new ConcurrentDictionary<string, long>(StringComparer.OrdinalIgnoreCase);
            _requestTimestamps = new ConcurrentDictionary<string, ConcurrentQueue<DateTimeOffset>>(StringComparer.OrdinalIgnoreCase);
            _tokenBuckets = new ConcurrentDictionary<string, TokenBucket>(StringComparer.OrdinalIgnoreCase);
            _healthCache = new ConcurrentDictionary<string, ProviderHealth>(StringComparer.OrdinalIgnoreCase);
            _configs = new ConcurrentDictionary<string, ProviderConfig>(StringComparer.OrdinalIgnoreCase);
            _providerLocks = new ConcurrentDictionary<string, object>(StringComparer.OrdinalIgnoreCase);
            _fallbackChain = config.FallbackChain?.ToList() ?? new List<string>
            {
                "OnnxLocal", "OllamaLocal", "OpenAI", "Anthropic", "GoogleGemini"
            };
            _loadBalancingStrategy = config.LoadBalancingStrategy;
            _globalBudget = config.GlobalBudget ?? new TokenBudget
            {
                DailyCostLimitUsd = 10.0m,
                MonthlyCostLimitUsd = 200.0m,
                DailyTokenLimit = 500000,
                MonthlyTokenLimit = 10000000
            };
            _retryConfig = config.RetryConfig ?? new RetryConfig();
            _costTracker = new LlmCostTracker(config.CostTrackerConfig ?? new CostTrackerConfig());
            _rateLimiter = new LlmRateLimiter(config.RateLimitConfig ?? new RateLimitConfig());

            foreach (var provider in _fallbackChain)
            {
                _circuitBreakers[provider] = CircuitBreakerState.Closed;
                _consecutiveFailures[provider] = 0;
                _requestCounts[provider] = 0;
                _roundRobinCounters[provider] = 0;
                _requestTimestamps[provider] = new ConcurrentQueue<DateTimeOffset>();
                _healthCache[provider] = new ProviderHealth { ProviderName = provider };
            }
        }

        /// <summary>
        /// Registers an inference provider with the router.
        /// </summary>
        /// <param name="name">Unique name for this provider instance.</param>
        /// <param name="provider">The provider implementation.</param>
        /// <param name="config">Optional provider-specific configuration.</param>
        public void RegisterProvider(string name, ILlmInferenceProvider provider, ProviderConfig? config = null)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(name);
            ArgumentNullException.ThrowIfNull(provider);

            _providers[name] = provider;
            if (config != null)
                _configs[name] = config;

            _circuitBreakers.GetOrAdd(name, _ => CircuitBreakerState.Closed);
            _consecutiveFailures.GetOrAdd(name, _ => 0);
            _requestCounts.GetOrAdd(name, _ => 0);
            _roundRobinCounters.GetOrAdd(name, _ => 0);
            _requestTimestamps.GetOrAdd(name, _ => new ConcurrentQueue<DateTimeOffset>());
            _providerLocks.GetOrAdd(name, _ => new object());

            if (config?.RateLimit != null)
            {
                _tokenBuckets[name] = new TokenBucket(
                    config.RateLimit.TokensPerMinute,
                    config.RateLimit.BurstSize,
                    TimeSpan.FromMinutes(1));
            }

            _healthCache[name] = new ProviderHealth
            {
                ProviderName = name,
                Status = HealthCheckStatus.Healthy,
                CircuitBreakerState = CircuitBreakerState.Closed
            };

            if (!_fallbackChain.Contains(name))
            {
                lock (_fallbackLock)
                {
                    _fallbackChain.Add(name);
                }
            }

            ProviderStatusChanged?.Invoke(this, new ProviderStatusChangedEventArgs
            {
                ProviderName = name,
                NewStatus = LlmStatus.Ready,
                OldStatus = LlmStatus.Unavailable
            });
        }

        /// <summary>
        /// Unregisters a provider from the router.
        /// </summary>
        /// <param name="name">The provider name to remove.</param>
        /// <returns>True if the provider was found and removed.</returns>
        public bool UnregisterProvider(string name)
        {
            ThrowIfDisposed();
            if (_providers.TryRemove(name, out var provider))
            {
                provider.Dispose();
                _circuitBreakers.TryRemove(name, out _);
                _consecutiveFailures.TryRemove(name, out _);
                _circuitOpenTimestamps.TryRemove(name, out _);
                _requestCounts.TryRemove(name, out _);
                _roundRobinCounters.TryRemove(name, out _);
                _requestTimestamps.TryRemove(name, out _);
                _tokenBuckets.TryRemove(name, out _);
                _healthCache.TryRemove(name, out _);
                _configs.TryRemove(name, out _);
                _providerLocks.TryRemove(name, out _);

                lock (_fallbackLock)
                {
                    _fallbackChain.Remove(name);
                }
                return true;
            }
            return false;
        }

        /// <summary>
        /// Routes a prompt to the best available provider based on the given context.
        /// </summary>
        /// <param name="prompt">The prompt text.</param>
        /// <param name="context">Context for routing decisions.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The LLM response from the selected provider.</returns>
        public async Task<LlmResponse> RouteAsync(
            string prompt,
            PromptContext context,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
            ArgumentNullException.ThrowIfNull(context);

            if (_isPaused)
                throw new InvalidOperationException("Router is currently paused.");

            var selectedProvider = SelectProvider(context);
            if (selectedProvider == null)
                throw new InvalidOperationException("No available provider matches the routing criteria.");

            var latency = Stopwatch.StartNew();
            LlmResponse response;
            int attempt = 0;
            string? lastError = null;

            while (attempt <= _retryConfig.MaxRetries)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (!IsProviderHealthy(selectedProvider))
                {
                    var fallback = GetFallbackProvider(selectedProvider, context);
                    if (fallback == null)
                        throw new InvalidOperationException(
                            $"No healthy provider available. Last error: {lastError}");
                    selectedProvider = fallback;
                    attempt = 0;
                    continue;
                }

                if (!await CheckRateLimitAsync(selectedProvider, cancellationToken))
                {
                    var fallback = GetFallbackProvider(selectedProvider, context);
                    if (fallback != null)
                    {
                        selectedProvider = fallback;
                        continue;
                    }
                    await Task.Delay(1000, cancellationToken);
                    attempt++;
                    continue;
                }

                if (!CheckBudget(context))
                {
                    CostAlert?.Invoke(this, new CostAlertEventArgs
                    {
                        Severity = CostAlertSeverity.Critical,
                        Message = "Budget limit reached. Request denied.",
                        Budget = _globalBudget
                    });
                    throw new InvalidOperationException("Budget limit exceeded.");
                }

                try
                {
                    var provider = _providers[selectedProvider];
                    var messages = new List<ChatMessage> { ChatMessage.User(prompt) };
                    var messagesReadOnly = (IReadOnlyList<ChatMessage>)messages;

                    var streamingTokens = provider.StreamAsync(messagesReadOnly, context, cancellationToken);
                    var contentBuilder = new StringBuilder();
                    TokenUsage? usage = null;
                    FinishReason finishReason = FinishReason.StopToken;
                    string model = context.ExtraParameters?.ContainsKey("model") == true
                        ? context.ExtraParameters["model"]?.ToString() ?? ""
                        : "";

                    await foreach (var token in streamingTokens.WithCancellation(cancellationToken))
                    {
                        contentBuilder.Append(token.Text);
                        if (token.IsFinal)
                        {
                            finishReason = token.FinishReason ?? FinishReason.StopToken;
                        }
                    }

                    var finalContent = contentBuilder.ToString();
                    var estimatedTokens = provider.EstimateTokens(finalContent);

                    usage = new TokenUsage
                    {
                        PromptTokens = provider.EstimateTokens(prompt),
                        CompletionTokens = estimatedTokens,
                        Model = model,
                        Provider = selectedProvider
                    };

                    latency.Stop();

                    response = new LlmResponse
                    {
                        Content = finalContent,
                        Model = model,
                        Provider = selectedProvider,
                        Usage = usage,
                        FinishReason = finishReason,
                        Latency = latency.Elapsed
                    };

                    RecordSuccess(selectedProvider, latency.Elapsed);
                    await _costTracker.TrackAsync(new CostEntry
                    {
                        Provider = selectedProvider,
                        Model = model,
                        InputTokens = usage.PromptTokens,
                        OutputTokens = usage.CompletionTokens,
                        CostUsd = usage.EstimatedCostUsd,
                        TaskType = context.TaskType,
                        SessionId = context.SessionId
                    }, cancellationToken);

                    RequestRouted?.Invoke(this, new RequestRoutedEventArgs
                    {
                        ProviderName = selectedProvider,
                        Complexity = context.Complexity,
                        Latency = latency.Elapsed,
                        TokenCount = usage.TotalTokens
                    });

                    return response;
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    lastError = ex.Message;
                    RecordFailure(selectedProvider, ex);
                    attempt++;

                    if (attempt <= _retryConfig.MaxRetries)
                    {
                        var delay = CalculateRetryDelay(attempt);
                        await Task.Delay(delay, cancellationToken);
                    }
                }
            }

            throw new InvalidOperationException(
                $"All retry attempts exhausted for provider '{selectedProvider}'. Last error: {lastError}");
        }

        /// <summary>
        /// Routes a chat completion request with full message history.
        /// </summary>
        /// <param name="messages">The conversation messages.</param>
        /// <param name="context">Context for routing decisions.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The LLM response.</returns>
        public async Task<LlmResponse> RouteChatAsync(
            IReadOnlyList<ChatMessage> messages,
            PromptContext context,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(messages);
            ArgumentNullException.ThrowIfNull(context);

            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            if (context.MaxLatencyMs is > 0 and int latencyMs)
                timeoutCts.CancelAfter(latencyMs);
            cancellationToken = timeoutCts.Token;

            var selectedProvider = SelectProvider(context)
                ?? throw new InvalidOperationException("No available provider matches the routing criteria.");

            var latency = Stopwatch.StartNew();
            int attempt = 0;
            string? lastError = null;

            while (attempt <= _retryConfig.MaxRetries)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (!IsProviderHealthy(selectedProvider))
                {
                    var fallback = GetFallbackProvider(selectedProvider, context);
                    if (fallback == null)
                        throw new InvalidOperationException(
                            $"No healthy provider available. Last error: {lastError}");
                    selectedProvider = fallback;
                    attempt = 0;
                    continue;
                }

                try
                {
                    var provider = _providers[selectedProvider];
                    var contentBuilder = new StringBuilder();
                    FinishReason finishReason = FinishReason.StopToken;
                    string model = context.ExtraParameters?.ContainsKey("model") == true
                        ? context.ExtraParameters["model"]?.ToString() ?? ""
                        : "";

                    await foreach (var token in provider.StreamAsync(messages, context, cancellationToken))
                    {
                        contentBuilder.Append(token.Text);
                        if (token.IsFinal)
                            finishReason = token.FinishReason ?? FinishReason.StopToken;
                    }

                    latency.Stop();

                    var promptTokenCount = messages.Sum(m => provider.EstimateTokens(m.Content));
                    var completionTokenCount = provider.EstimateTokens(contentBuilder.ToString());

                    var response = new LlmResponse
                    {
                        Content = contentBuilder.ToString(),
                        Model = model,
                        Provider = selectedProvider,
                        Usage = new TokenUsage
                        {
                            PromptTokens = promptTokenCount,
                            CompletionTokens = completionTokenCount,
                            Model = model,
                            Provider = selectedProvider
                        },
                        FinishReason = finishReason,
                        Latency = latency.Elapsed
                    };

                    RecordSuccess(selectedProvider, latency.Elapsed);
                    return response;
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    lastError = ex.Message;
                    RecordFailure(selectedProvider, ex);
                    attempt++;
                    if (attempt <= _retryConfig.MaxRetries)
                        await Task.Delay(CalculateRetryDelay(attempt), cancellationToken);
                }
            }

            throw new InvalidOperationException(
                $"All retry attempts exhausted for provider '{selectedProvider}'. Last error: {lastError}");
        }

        /// <summary>
        /// Routes a streaming request and returns the async enumerable of tokens.
        /// </summary>
        /// <param name="prompt">The prompt text.</param>
        /// <param name="context">Context for routing decisions.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>An async enumerable of streaming tokens.</returns>
        public async IAsyncEnumerable<StreamingToken> RouteStreamAsync(
            string prompt,
            PromptContext context,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
            ArgumentNullException.ThrowIfNull(context);

            var selectedProvider = SelectProvider(context)
                ?? throw new InvalidOperationException("No available provider matches the routing criteria.");

            ILlmInferenceProvider? provider = null;
            int attempts = 0;

            while (attempts <= _retryConfig.MaxRetries)
            {
                if (IsProviderHealthy(selectedProvider) && _providers.TryGetValue(selectedProvider, out provider))
                    break;

                var fallback = GetFallbackProvider(selectedProvider, context);
                if (fallback == null)
                    throw new InvalidOperationException("No healthy provider available.");

                selectedProvider = fallback;
                attempts++;
            }

            if (provider == null)
                throw new InvalidOperationException("Failed to select a provider.");

            var messages = new List<ChatMessage> { ChatMessage.User(prompt) };
            StreamingToken? lastToken = null;

            await foreach (var token in provider.StreamAsync((IReadOnlyList<ChatMessage>)messages, context, cancellationToken))
            {
                lastToken = token;
                yield return token;
            }

            if (lastToken != null)
            {
                RecordSuccess(selectedProvider, TimeSpan.Zero);
            }
        }

        /// <summary>
        /// Gets the best provider for a specific task type based on model capabilities.
        /// </summary>
        /// <param name="taskType">The type of task.</param>
        /// <returns>The name of the best provider, or null if none found.</returns>
        public string? GetBestProviderForTask(LlmTaskType taskType)
        {
            ThrowIfDisposed();

            foreach (var name in _fallbackChain)
            {
                if (!_providers.TryGetValue(name, out var provider))
                    continue;
                if (!IsProviderHealthy(name))
                    continue;

                var model = provider.GetModelInfo(provider.AvailableModels.FirstOrDefault()?.ModelId ?? "");
                if (model?.PreferredTasks != null && model.PreferredTasks.Contains(taskType))
                    return name;
            }

            return _fallbackChain.FirstOrDefault(IsProviderHealthy);
        }

        /// <summary>
        /// Estimates the cost of a request across all providers.
        /// </summary>
        /// <param name="prompt">The prompt text.</param>
        /// <param name="estimatedOutputTokens">Estimated output tokens.</param>
        /// <returns>Dictionary of provider name to estimated cost in USD.</returns>
        public IReadOnlyDictionary<string, decimal> EstimateCosts(string prompt, int estimatedOutputTokens)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(prompt);

            var costs = new Dictionary<string, decimal>(StringComparer.OrdinalIgnoreCase);

            foreach (var (name, provider) in _providers)
            {
                if (!IsProviderHealthy(name))
                    continue;

                var promptTokens = provider.EstimateTokens(prompt);
                var model = provider.AvailableModels.FirstOrDefault();
                if (model != null)
                {
                    var inputCost = (decimal)promptTokens / 1_000_000m * model.InputCostPer1MTokens;
                    var outputCost = (decimal)estimatedOutputTokens / 1_000_000m * model.OutputCostPer1MTokens;
                    costs[name] = inputCost + outputCost;
                }
            }

            return costs;
        }

        /// <summary>
        /// Gets latency statistics for all providers.
        /// </summary>
        /// <returns>Dictionary of provider name to average latency.</returns>
        public IReadOnlyDictionary<string, TimeSpan> GetProviderLatencies()
        {
            ThrowIfDisposed();
            var latencies = new Dictionary<string, TimeSpan>(StringComparer.OrdinalIgnoreCase);

            foreach (var (name, _) in _providers)
            {
                if (_healthCache.TryGetValue(name, out var health))
                    latencies[name] = health.AverageLatency;
            }

            return latencies;
        }

        /// <summary>
        /// Resets the circuit breaker for a specific provider, allowing requests again.
        /// </summary>
        /// <param name="providerName">The provider name.</param>
        public void ResetCircuitBreaker(string providerName)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(providerName);

            _circuitBreakers[providerName] = CircuitBreakerState.Closed;
            _consecutiveFailures[providerName] = 0;
            _circuitOpenTimestamps.TryRemove(providerName, out _);

            if (_healthCache.TryGetValue(providerName, out var health))
            {
                _healthCache[providerName] = health with
                {
                    CircuitBreakerState = CircuitBreakerState.Closed,
                    ConsecutiveFailures = 0
                };
            }

            ProviderStatusChanged?.Invoke(this, new ProviderStatusChangedEventArgs
            {
                ProviderName = providerName,
                NewStatus = LlmStatus.Ready,
                OldStatus = LlmStatus.Error
            });
        }

        /// <summary>
        /// Gets the current state of all circuit breakers.
        /// </summary>
        public IReadOnlyDictionary<string, CircuitBreakerState> GetCircuitBreakerStates()
        {
            ThrowIfDisposed();
            return new Dictionary<string, CircuitBreakerState>(_circuitBreakers);
        }

        /// <summary>
        /// Pauses routing of new requests. In-flight requests continue.
        /// </summary>
        public void Pause()
        {
            _isPaused = true;
        }

        /// <summary>
        /// Resumes routing of new requests after a pause.
        /// </summary>
        public void Resume()
        {
            _isPaused = false;
        }

        /// <summary>
        /// Performs health checks on all registered providers.
        /// </summary>
        /// <returns>Health check results for all providers.</returns>
        public async Task<IReadOnlyDictionary<string, HealthCheckStatus>> CheckAllHealthAsync(
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            var results = new Dictionary<string, HealthCheckStatus>(StringComparer.OrdinalIgnoreCase);
            var tasks = new List<Task>();

            foreach (var (name, provider) in _providers)
            {
                var capturedName = name;
                var capturedProvider = provider;
                tasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        var status = await capturedProvider.CheckHealthAsync(cancellationToken);
                        results[capturedName] = status;
                        UpdateHealth(capturedName, status);
                    }
                    catch
                    {
                        results[capturedName] = HealthCheckStatus.Unhealthy;
                        UpdateHealth(capturedName, HealthCheckStatus.Unhealthy);
                    }
                }, cancellationToken));
            }

            await Task.WhenAll(tasks);
            return results;
        }

        /// <summary>
        /// Gets comprehensive statistics about router usage.
        /// </summary>
        public RouterStatistics GetStatistics()
        {
            ThrowIfDisposed();
            var stats = new RouterStatistics
            {
                TotalRequests = _requestCounts.Values.Sum(),
                ProviderRequestCounts = new Dictionary<string, long>(_requestCounts),
                CircuitBreakerStates = new Dictionary<string, CircuitBreakerState>(_circuitBreakers),
                HealthStatuses = new Dictionary<string, ProviderHealth>(_healthCache),
                ProviderCount = _providers.Count,
                ActiveProviderCount = _providers.Count(kv => IsProviderHealthy(kv.Key)),
                BudgetStatus = _globalBudget
            };
            return stats;
        }

        /// <summary>
        /// Adds a provider to the fallback chain at the specified position.
        /// </summary>
        /// <param name="providerName">The provider name.</param>
        /// <param name="position">Position in the chain (0-based). Use -1 for end.</param>
        public void AddToFallbackChain(string providerName, int position = -1)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(providerName);

            lock (_fallbackLock)
            {
                _fallbackChain.Remove(providerName);
                if (position < 0 || position >= _fallbackChain.Count)
                    _fallbackChain.Add(providerName);
                else
                    _fallbackChain.Insert(position, providerName);
            }
        }

        /// <inheritdoc/>
        public async Task<StructuredOutput<T>> InferStructuredAsync<T>(
            string prompt,
            string? systemPrompt,
            string? jsonSchema,
            PromptContext context,
            CancellationToken cancellationToken = default) where T : class
        {
            ThrowIfDisposed();

            var selectedProvider = SelectProvider(context)
                ?? throw new InvalidOperationException("No available provider.");

            int attempts = 0;
            while (attempts <= _retryConfig.MaxRetries)
            {
                if (IsProviderHealthy(selectedProvider) &&
                    _providers.TryGetValue(selectedProvider, out var provider))
                {
                    try
                    {
                        var result = await provider.InferStructuredAsync<T>(
                            prompt, systemPrompt, jsonSchema, context, cancellationToken);
                        RecordSuccess(selectedProvider, TimeSpan.Zero);
                        return result;
                    }
                    catch (Exception ex)
                    {
                        RecordFailure(selectedProvider, ex);
                        attempts++;
                    }
                }
                else
                {
                    var fallback = GetFallbackProvider(selectedProvider, context);
                    if (fallback == null)
                        throw new InvalidOperationException("No healthy provider available.");
                    selectedProvider = fallback;
                }
            }

            return StructuredOutput<T>.Fail("All providers exhausted.");
        }

        /// <inheritdoc/>
        public async IAsyncEnumerable<StreamingToken> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            PromptContext context,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            var selectedProvider = SelectProvider(context)
                ?? throw new InvalidOperationException("No available provider.");

            if (!_providers.TryGetValue(selectedProvider, out var provider))
                throw new InvalidOperationException($"Provider '{selectedProvider}' not found.");

            await foreach (var token in provider.StreamAsync(messages, context, cancellationToken))
            {
                yield return token;
            }
        }

        /// <inheritdoc/>
        public async Task<EmbeddingResult> GetEmbeddingAsync(
            string text,
            EmbeddingModel model = EmbeddingModel.TextEmbedding3Small,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            foreach (var name in _fallbackChain)
            {
                if (!IsProviderHealthy(name) || !_providers.TryGetValue(name, out var provider))
                    continue;

                try
                {
                    return await provider.GetEmbeddingAsync(text, model, cancellationToken);
                }
                catch
                {
                    continue;
                }
            }

            throw new InvalidOperationException("No provider available for embeddings.");
        }

        /// <inheritdoc/>
        public int EstimateTokens(string text)
        {
            if (string.IsNullOrEmpty(text))
                return 0;
            foreach (var (_, provider) in _providers)
            {
                return provider.EstimateTokens(text);
            }
            return text.Length / 4;
        }

        /// <inheritdoc/>
        public ModelInfo? GetModelInfo(string modelId)
        {
            foreach (var (_, provider) in _providers)
            {
                var info = provider.GetModelInfo(modelId);
                if (info != null)
                    return info;
            }
            return null;
        }

        /// <inheritdoc/>
        public async Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
        {
            if (_isPaused || _disposed)
                return false;
            foreach (var name in _fallbackChain)
            {
                if (IsProviderHealthy(name) && _providers.TryGetValue(name, out var provider))
                {
                    if (await provider.IsAvailableAsync(cancellationToken))
                        return true;
                }
            }
            return false;
        }

        /// <inheritdoc/>
        public void CancelCurrentInference()
        {
            foreach (var (_, provider) in _providers)
            {
                try
                { provider.CancelCurrentInference(); }
                catch (Exception ex)
                {
                    SynapseLogger.Default.Debug("HybridLlmRouter", "Provider cancel failed during best-effort shutdown.", ex);
                }
            }
        }

        /// <inheritdoc/>
        public async Task WarmUpAsync(CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            var tasks = _providers.Select(kv => kv.Value.WarmUpAsync(cancellationToken));
            try
            {
                await Task.WhenAll(tasks);
            }
            catch
            {
                // Individual warmup failures are non-fatal
            }
        }

        /// <inheritdoc/>
        public async Task ShutdownAsync()
        {
            if (_disposed)
                return;
            _isPaused = true;

            var tasks = _providers.Select(kv => kv.Value.ShutdownAsync());
            try
            {
                await Task.WhenAll(tasks);
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("HybridLlmRouter", "One or more providers failed during graceful shutdown.", ex);
            }
        }

        /// <inheritdoc/>
        public async Task<HealthCheckStatus> CheckHealthAsync(CancellationToken cancellationToken = default)
        {
            if (_disposed)
                return HealthCheckStatus.Unhealthy;
            if (_isPaused)
                return HealthCheckStatus.Degraded;

            var results = await CheckAllHealthAsync(cancellationToken);
            if (results.Values.All(v => v == HealthCheckStatus.Healthy))
                return HealthCheckStatus.Healthy;
            if (results.Values.Any(v => v == HealthCheckStatus.Unhealthy))
                return results.Values.All(v => v == HealthCheckStatus.Unhealthy)
                    ? HealthCheckStatus.Unhealthy
                    : HealthCheckStatus.Degraded;
            return HealthCheckStatus.Degraded;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _isPaused = true;

            foreach (var (_, provider) in _providers)
            {
                try
                { provider.Dispose(); }
                catch (Exception ex)
                {
                    SynapseLogger.Default.Debug("HybridLlmRouter", "Provider dispose failed during best-effort cleanup.", ex);
                }
            }

            _providers.Clear();
            _costTracker?.Dispose();
        }

        // ─── Private Helpers ───────────────────────────────────────────────

        private string? SelectProvider(PromptContext context)
        {
            if (context.Sensitivity == DataSensitivity.Confidential ||
                context.Sensitivity == DataSensitivity.Restricted)
            {
                var localProviders = _fallbackChain.Where(n =>
                    _providers.ContainsKey(n) && IsProviderHealthy(n) &&
                    _providers[n].Mode is LlmProviderMode.LocalOnnx or LlmProviderMode.LocalOllama);

                if (context.PreferredMode.HasValue)
                {
                    var preferred = localProviders.FirstOrDefault(n =>
                        _providers[n].Mode == context.PreferredMode.Value);
                    if (preferred != null)
                        return preferred;
                }

                return localProviders.FirstOrDefault();
            }

            if (context.PreferredMode.HasValue)
            {
                var preferred = _fallbackChain.FirstOrDefault(n =>
                    _providers.ContainsKey(n) && IsProviderHealthy(n) &&
                    _providers[n].Mode == context.PreferredMode.Value);
                if (preferred != null)
                    return preferred;
            }

            return _loadBalancingStrategy switch
            {
                LoadBalancingStrategy.RoundRobin => SelectRoundRobin(),
                LoadBalancingStrategy.LeastLoaded => SelectLeastLoaded(),
                LoadBalancingStrategy.FastestResponse => SelectFastestResponse(),
                LoadBalancingStrategy.QualityFirst => SelectQualityFirst(context),
                _ => SelectRoundRobin()
            };
        }

        private string? SelectRoundRobin()
        {
            var healthy = _fallbackChain.Where(IsProviderHealthy).ToList();
            if (healthy.Count == 0)
                return null;

            var index = Interlocked.Increment(ref _roundRobinIndex) % healthy.Count;
            return healthy[index];
        }

        private string? SelectLeastLoaded()
        {
            string? best = null;
            long minRequests = long.MaxValue;

            foreach (var name in _fallbackChain)
            {
                if (!IsProviderHealthy(name))
                    continue;
                if (!_requestTimestamps.TryGetValue(name, out var timestamps))
                    continue;

                CleanupTimestamps(timestamps);
                var activeCount = timestamps.Count;

                if (activeCount < minRequests)
                {
                    minRequests = activeCount;
                    best = name;
                }
            }

            return best;
        }

        private string? SelectFastestResponse()
        {
            string? best = null;
            var bestLatency = TimeSpan.MaxValue;

            foreach (var name in _fallbackChain)
            {
                if (!IsProviderHealthy(name))
                    continue;
                if (!_healthCache.TryGetValue(name, out var health))
                    continue;

                if (health.AverageLatency < bestLatency && health.AverageLatency > TimeSpan.Zero)
                {
                    bestLatency = health.AverageLatency;
                    best = name;
                }
            }

            return best ?? _fallbackChain.FirstOrDefault(IsProviderHealthy);
        }

        private string? SelectQualityFirst(PromptContext context)
        {
            decimal? maxCost = context.MaxCostUsd;
            var costs = EstimateCosts("sample", 100);

            var candidates = _fallbackChain.Where(n =>
                IsProviderHealthy(n) &&
                (!maxCost.HasValue || !costs.ContainsKey(n) || costs[n] <= maxCost.Value))
                .ToList();

            if (candidates.Count == 0)
                return _fallbackChain.FirstOrDefault(IsProviderHealthy);

            return candidates.FirstOrDefault();
        }

        private string? GetFallbackProvider(string currentProvider, PromptContext context)
        {
            lock (_fallbackLock)
            {
                var currentIndex = _fallbackChain.IndexOf(currentProvider);
                for (int i = currentIndex + 1; i < _fallbackChain.Count; i++)
                {
                    if (IsProviderHealthy(_fallbackChain[i]))
                    {
                        if (context.Sensitivity >= DataSensitivity.Confidential &&
                            _providers.TryGetValue(_fallbackChain[i], out var p) &&
                            p.Mode is LlmProviderMode.LocalOnnx or LlmProviderMode.LocalOllama)
                            return _fallbackChain[i];
                        if (context.Sensitivity < DataSensitivity.Confidential)
                            return _fallbackChain[i];
                    }
                }

                for (int i = 0; i < currentIndex; i++)
                {
                    if (IsProviderHealthy(_fallbackChain[i]))
                        return _fallbackChain[i];
                }
            }

            return null;
        }

        private bool IsProviderHealthy(string name)
        {
            if (!_circuitBreakers.TryGetValue(name, out var state))
                return false;
            if (state == CircuitBreakerState.Closed)
                return true;
            if (state == CircuitBreakerState.Open)
            {
                if (_circuitOpenTimestamps.TryGetValue(name, out var openTime) &&
                    openTime + TimeSpan.FromSeconds(30) < DateTimeOffset.UtcNow)
                {
                    _circuitBreakers[name] = CircuitBreakerState.HalfOpen;
                    return true;
                }
                return false;
            }
            return true; // HalfOpen: allow one request through
        }

        private void RecordSuccess(string providerName, TimeSpan latency)
        {
            _consecutiveFailures[providerName] = 0;
            _requestCounts.AddOrUpdate(providerName, 1, (_, c) => c + 1);

            if (_circuitBreakers.TryGetValue(providerName, out var state) &&
                state != CircuitBreakerState.Closed)
            {
                _circuitBreakers[providerName] = CircuitBreakerState.Closed;
                _circuitOpenTimestamps.TryRemove(providerName, out _);

                ProviderStatusChanged?.Invoke(this, new ProviderStatusChangedEventArgs
                {
                    ProviderName = providerName,
                    NewStatus = LlmStatus.Ready,
                    OldStatus = LlmStatus.Error
                });
            }

            if (_healthCache.TryGetValue(providerName, out var health))
            {
                var newAvg = health.TotalRequests == 0
                    ? latency
                    : TimeSpan.FromTicks(
                        (health.AverageLatency.Ticks * health.TotalRequests + latency.Ticks)
                        / (health.TotalRequests + 1));

                _healthCache[providerName] = health with
                {
                    AverageLatency = newAvg,
                    SuccessRate = (float)(health.TotalRequests + 1 - health.FailedRequests) /
                                  (health.TotalRequests + 1),
                    TotalRequests = health.TotalRequests + 1,
                    LastSuccessAt = DateTimeOffset.UtcNow,
                    LastCheckedAt = DateTimeOffset.UtcNow,
                    CircuitBreakerState = CircuitBreakerState.Closed,
                    ConsecutiveFailures = 0
                };
            }
        }

        private void RecordFailure(string providerName, Exception ex)
        {
            var failures = _consecutiveFailures.AddOrUpdate(providerName, 1, (_, c) => c + 1);
            _requestCounts.AddOrUpdate(providerName, 1, (_, c) => c + 1);

            int threshold = 5;
            if (_configs.TryGetValue(providerName, out var config) &&
                config.ExtraSettings != null &&
                config.ExtraSettings.TryGetValue("CircuitBreakerThreshold", out var thresholdStr) &&
                int.TryParse(thresholdStr, out var parsed))
            {
                threshold = parsed;
            }

            if (failures >= threshold)
            {
                _circuitBreakers[providerName] = CircuitBreakerState.Open;
                _circuitOpenTimestamps[providerName] = DateTimeOffset.UtcNow;

                ProviderStatusChanged?.Invoke(this, new ProviderStatusChangedEventArgs
                {
                    ProviderName = providerName,
                    NewStatus = LlmStatus.Error,
                    OldStatus = LlmStatus.Ready
                });
            }

            if (_healthCache.TryGetValue(providerName, out var health))
            {
                _healthCache[providerName] = health with
                {
                    FailedRequests = health.FailedRequests + 1,
                    TotalRequests = health.TotalRequests + 1,
                    LastFailureAt = DateTimeOffset.UtcNow,
                    LastError = ex.Message,
                    ConsecutiveFailures = failures,
                    CircuitBreakerState = failures >= threshold
                        ? CircuitBreakerState.Open
                        : CircuitBreakerState.Closed
                };
            }
        }

        private bool CheckBudget(PromptContext context)
        {
            if (_globalBudget.IsDailyExhausted || _globalBudget.IsMonthlyExhausted)
                return false;

            if (context.MaxCostUsd.HasValue)
            {
                if (_globalBudget.CurrentDailyCostUsd + context.MaxCostUsd.Value >
                    (_globalBudget.DailyCostLimitUsd ?? decimal.MaxValue))
                    return false;
            }

            return true;
        }

        private async Task<bool> CheckRateLimitAsync(string providerName, CancellationToken ct)
        {
            if (_tokenBuckets.TryGetValue(providerName, out var bucket))
                return bucket.TryConsume(1);

            if (_configs.TryGetValue(providerName, out var config) && config.RateLimit != null)
            {
                if (_requestTimestamps.TryGetValue(providerName, out var timestamps))
                {
                    CleanupTimestamps(timestamps);
                    if (timestamps.Count >= config.RateLimit.RequestsPerMinute)
                        return false;
                    timestamps.Enqueue(DateTimeOffset.UtcNow);
                }
            }

            return true;
        }

        private void CleanupTimestamps(ConcurrentQueue<DateTimeOffset> timestamps)
        {
            var cutoff = DateTimeOffset.UtcNow.AddMinutes(-1);
            while (timestamps.TryPeek(out var oldest) && oldest < cutoff)
            {
                timestamps.TryDequeue(out _);
            }
        }

        private TimeSpan CalculateRetryDelay(int attempt)
        {
            var baseDelay = _retryConfig.InitialDelay.TotalMilliseconds;
            var delay = baseDelay * Math.Pow(_retryConfig.BackoffMultiplier, attempt - 1);
            delay = Math.Min(delay, _retryConfig.MaxDelay.TotalMilliseconds);

            if (_retryConfig.UseJitter)
            {
                var jitter = delay * 0.2 * (Random.Shared.NextDouble() * 2 - 1);
                delay += jitter;
            }

            return TimeSpan.FromMilliseconds(Math.Max(0, delay));
        }

        private void UpdateHealth(string providerName, HealthCheckStatus status)
        {
            if (_healthCache.TryGetValue(providerName, out var health))
            {
                _healthCache[providerName] = health with
                {
                    Status = status,
                    LastCheckedAt = DateTimeOffset.UtcNow
                };
            }
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }

    /// <summary>
    /// Configuration for the HybridLlmRouter.
    /// </summary>
    public record HybridLlmRouterConfig
    {
        /// <summary>Fallback chain of provider names in priority order.</summary>
        public IReadOnlyList<string>? FallbackChain { get; init; }

        /// <summary>Load balancing strategy for providers.</summary>
        public LoadBalancingStrategy LoadBalancingStrategy { get; init; } = LoadBalancingStrategy.RoundRobin;

        /// <summary>Global token budget.</summary>
        public TokenBudget? GlobalBudget { get; init; }

        /// <summary>Retry configuration.</summary>
        public RetryConfig? RetryConfig { get; init; }

        /// <summary>Cost tracker configuration.</summary>
        public CostTrackerConfig? CostTrackerConfig { get; init; }

        /// <summary>Rate limit configuration.</summary>
        public RateLimitConfig? RateLimitConfig { get; init; }
    }

    /// <summary>
    /// Event args for provider status change events.
    /// </summary>
    public class ProviderStatusChangedEventArgs : EventArgs
    {
        /// <summary>Name of the provider.</summary>
        public string ProviderName { get; set; } = string.Empty;

        /// <summary>New status of the provider.</summary>
        public LlmStatus NewStatus { get; set; }

        /// <summary>Previous status of the provider.</summary>
        public LlmStatus OldStatus { get; set; }
    }

    /// <summary>
    /// Event args for cost alert events.
    /// </summary>
    public class CostAlertEventArgs : EventArgs
    {
        /// <summary>Severity of the alert.</summary>
        public CostAlertSeverity Severity { get; set; }

        /// <summary>Alert message.</summary>
        public string Message { get; set; } = string.Empty;

        /// <summary>Current budget status.</summary>
        public TokenBudget? Budget { get; set; }
    }

    /// <summary>
    /// Event args for request routed events.
    /// </summary>
    public class RequestRoutedEventArgs : EventArgs
    {
        /// <summary>Provider that handled the request.</summary>
        public string ProviderName { get; set; } = string.Empty;

        /// <summary>Complexity of the routed request.</summary>
        public PromptComplexity Complexity { get; set; }

        /// <summary>Request latency.</summary>
        public TimeSpan Latency { get; set; }

        /// <summary>Number of tokens processed.</summary>
        public int TokenCount { get; set; }
    }

    /// <summary>
    /// Aggregated statistics about router usage.
    /// </summary>
    public class RouterStatistics
    {
        /// <summary>Total requests processed.</summary>
        public long TotalRequests { get; set; }

        /// <summary>Request counts per provider.</summary>
        public IReadOnlyDictionary<string, long> ProviderRequestCounts { get; set; } =
            new Dictionary<string, long>();

        /// <summary>Circuit breaker states per provider.</summary>
        public IReadOnlyDictionary<string, CircuitBreakerState> CircuitBreakerStates { get; set; } =
            new Dictionary<string, CircuitBreakerState>();

        /// <summary>Health information per provider.</summary>
        public IReadOnlyDictionary<string, ProviderHealth> HealthStatuses { get; set; } =
            new Dictionary<string, ProviderHealth>();

        /// <summary>Total registered providers.</summary>
        public int ProviderCount { get; set; }

        /// <summary>Currently healthy providers.</summary>
        public int ActiveProviderCount { get; set; }

        /// <summary>Current budget status.</summary>
        public TokenBudget? BudgetStatus { get; set; }
    }

    /// <summary>
    /// Token bucket for rate limiting.
    /// </summary>
    internal sealed class TokenBucket
    {
        private readonly int _maxTokens;
        private readonly int _refillRate;
        private readonly TimeSpan _refillInterval;
        private double _currentTokens;
        private DateTimeOffset _lastRefill;
        private readonly object _lock = new();

        /// <summary>
        /// Initializes a new token bucket.
        /// </summary>
        /// <param name="refillRate">Tokens to add per interval.</param>
        /// <param name="maxTokens">Maximum bucket capacity.</param>
        /// <param name="refillInterval">Interval between refills.</param>
        public TokenBucket(int refillRate, int maxTokens, TimeSpan refillInterval)
        {
            _refillRate = refillRate;
            _maxTokens = maxTokens;
            _refillInterval = refillInterval;
            _currentTokens = maxTokens;
            _lastRefill = DateTimeOffset.UtcNow;
        }

        /// <summary>
        /// Attempts to consume tokens from the bucket.
        /// </summary>
        /// <param name="tokens">Number of tokens to consume.</param>
        /// <returns>True if tokens were available and consumed.</returns>
        public bool TryConsume(int tokens)
        {
            lock (_lock)
            {
                Refill();
                if (_currentTokens >= tokens)
                {
                    _currentTokens -= tokens;
                    return true;
                }
                return false;
            }
        }

        private void Refill()
        {
            var now = DateTimeOffset.UtcNow;
            var elapsed = now - _lastRefill;
            var intervals = (int)(elapsed.TotalMilliseconds / _refillInterval.TotalMilliseconds);
            if (intervals > 0)
            {
                _currentTokens = Math.Min(_maxTokens, _currentTokens + intervals * _refillRate);
                _lastRefill = now;
            }
        }
    }
}
