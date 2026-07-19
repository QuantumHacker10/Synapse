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
    /// Represents a single message in an LLM conversation.
    /// </summary>
    public record ChatMessage
    {
        /// <summary>The role of the message sender.</summary>
        public MessageRole Role { get; init; } = MessageRole.User;

        /// <summary>The text content of the message.</summary>
        public string Content { get; init; } = string.Empty;

        /// <summary>Optional name for the message sender.</summary>
        public string? Name { get; init; }

        /// <summary>Optional function call data.</summary>
        public FunctionCall? FunctionCall { get; init; }

        /// <summary>Optional tool calls.</summary>
        public IReadOnlyList<FunctionCall>? ToolCalls { get; init; }

        /// <summary>Timestamp when the message was created.</summary>
        public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>Optional metadata attached to the message.</summary>
        public IReadOnlyDictionary<string, string>? Metadata { get; init; }

        /// <summary>Creates a system message.</summary>
        public static ChatMessage System(string content) => new() { Role = MessageRole.System, Content = content };

        /// <summary>Creates a user message.</summary>
        public static ChatMessage User(string content) => new() { Role = MessageRole.User, Content = content };

        /// <summary>Creates an assistant message.</summary>
        public static ChatMessage Assistant(string content) => new() { Role = MessageRole.Assistant, Content = content };
    }

    /// <summary>
    /// Provides context information for prompt construction and routing.
    /// </summary>
    public record PromptContext
    {
        /// <summary>The task type being performed.</summary>
        public LlmTaskType TaskType { get; init; } = LlmTaskType.QueryAnswering;

        /// <summary>Complexity of the prompt.</summary>
        public PromptComplexity Complexity { get; init; } = PromptComplexity.Simple;

        /// <summary>Data sensitivity level for privacy routing.</summary>
        public DataSensitivity Sensitivity { get; init; } = DataSensitivity.Public;

        /// <summary>Preferred provider mode, if any.</summary>
        public LlmProviderMode? PreferredMode { get; init; }

        /// <summary>Maximum acceptable latency in milliseconds.</summary>
        public int? MaxLatencyMs { get; init; }

        /// <summary>Maximum cost budget for this request in USD.</summary>
        public decimal? MaxCostUsd { get; init; }

        /// <summary>Whether streaming output is required.</summary>
        public bool RequiresStreaming { get; init; }

        /// <summary>Whether a structured JSON output is required.</summary>
        public bool RequiresStructuredOutput { get; init; }

        /// <summary>JSON schema for structured output, if applicable.</summary>
        public string? OutputSchema { get; init; }

        /// <summary>Additional parameters for the request.</summary>
        public IReadOnlyDictionary<string, object>? ExtraParameters { get; init; }

        /// <summary>The session or conversation ID for context tracking.</summary>
        public string? SessionId { get; init; }

        /// <summary>The priority of the request.</summary>
        public RequestPriority Priority { get; init; } = RequestPriority.Normal;
    }

    /// <summary>
    /// Tracks token usage for a single request or aggregated over time.
    /// </summary>
    public record TokenUsage
    {
        /// <summary>Number of prompt/input tokens.</summary>
        public int PromptTokens { get; init; }

        /// <summary>Number of completion/output tokens.</summary>
        public int CompletionTokens { get; init; }

        /// <summary>Total tokens used (prompt + completion).</summary>
        public int TotalTokens => PromptTokens + CompletionTokens;

        /// <summary>Estimated cost in USD.</summary>
        public decimal EstimatedCostUsd { get; init; }

        /// <summary>The model that was used.</summary>
        public string Model { get; init; } = string.Empty;

        /// <summary>Provider that processed the request.</summary>
        public string Provider { get; init; } = string.Empty;
    }

    /// <summary>
    /// Represents a complete response from an LLM provider.
    /// </summary>
    public record LlmResponse
    {
        /// <summary>The generated content.</summary>
        public string Content { get; init; } = string.Empty;

        /// <summary>The model that generated the response.</summary>
        public string Model { get; init; } = string.Empty;

        /// <summary>Provider that processed the request.</summary>
        public string Provider { get; init; } = string.Empty;

        /// <summary>Token usage statistics.</summary>
        public TokenUsage Usage { get; init; } = new();

        /// <summary>Reason generation finished.</summary>
        public FinishReason FinishReason { get; init; } = FinishReason.StopToken;

        /// <summary>Unique identifier for this response.</summary>
        public string Id { get; init; } = Guid.NewGuid().ToString("N");

        /// <summary>Timestamp of the response.</summary>
        public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>Time taken to generate the response.</summary>
        public TimeSpan Latency { get; init; }

        /// <summary>Function calls requested by the model, if any.</summary>
        public IReadOnlyList<FunctionCall>? FunctionCalls { get; init; }

        /// <summary>Parsed structured output, if applicable.</summary>
        public object? StructuredData { get; init; }

        /// <summary>Whether this response was served from cache.</summary>
        public bool FromCache { get; init; }

        /// <summary>Raw response JSON from the provider, if available.</summary>
        public string? RawResponse { get; init; }
    }

    /// <summary>
    /// Information about a loaded or available LLM model.
    /// </summary>
    public record ModelInfo
    {
        /// <summary>Model identifier (e.g., "gpt-4o", "llama3:8b").</summary>
        public string ModelId { get; init; } = string.Empty;

        /// <summary>Display name of the model.</summary>
        public string DisplayName { get; init; } = string.Empty;

        /// <summary>Provider that offers this model.</summary>
        public string Provider { get; init; } = string.Empty;

        /// <summary>Maximum context window in tokens.</summary>
        public int ContextWindow { get; init; }

        /// <summary>Maximum output tokens.</summary>
        public int MaxOutputTokens { get; init; }

        /// <summary>Tokenizer type used by this model.</summary>
        public TokenizerType Tokenizer { get; init; } = TokenizerType.BPE;

        /// <summary>Vocabulary size of the tokenizer.</summary>
        public int VocabularySize { get; init; }

        /// <summary>Estimated VRAM usage in MB.</summary>
        public int EstimatedVramMb { get; init; }

        /// <summary>Whether this model supports vision inputs.</summary>
        public bool SupportsVision { get; init; }

        /// <summary>Whether this model supports function calling.</summary>
        public bool SupportsFunctionCalling { get; init; }

        /// <summary>Whether this model supports embeddings.</summary>
        public bool SupportsEmbeddings { get; init; }

        /// <summary>Input cost per 1M tokens in USD.</summary>
        public decimal InputCostPer1MTokens { get; init; }

        /// <summary>Output cost per 1M tokens in USD.</summary>
        public decimal OutputCostPer1MTokens { get; init; }

        /// <summary>Task types this model excels at.</summary>
        public IReadOnlyList<LlmTaskType>? PreferredTasks { get; init; }
    }

    /// <summary>
    /// Represents a vector embedding for semantic operations.
    /// </summary>
    public record EmbeddingVector
    {
        /// <summary>The embedding dimensions.</summary>
        public int Dimensions { get; init; }

        /// <summary>The raw embedding values.</summary>
        public float[] Values { get; init; } = Array.Empty<float>();

        /// <summary>The text that was embedded.</summary>
        public string Text { get; init; } = string.Empty;

        /// <summary>The model used to generate the embedding.</summary>
        public string Model { get; init; } = string.Empty;

        /// <summary>Optional identifier for the embedding.</summary>
        public string? Id { get; init; }

        /// <summary>Metadata associated with the embedding.</summary>
        public IReadOnlyDictionary<string, string>? Metadata { get; init; }

        /// <summary>Normalizes the embedding vector to unit length.</summary>
        public EmbeddingVector Normalize()
        {
            if (Values.Length == 0)
                return this;
            float magnitude = 0f;
            for (int i = 0; i < Values.Length; i++)
                magnitude += Values[i] * Values[i];
            magnitude = MathF.Sqrt(magnitude);
            if (magnitude < 1e-10f)
                return this;
            var normalized = new float[Values.Length];
            for (int i = 0; i < Values.Length; i++)
                normalized[i] = Values[i] / magnitude;
            return this with { Values = normalized };
        }
    }

    /// <summary>
    /// A generic wrapper for structured output with validation.
    /// </summary>
    public record StructuredOutput<T> where T : class
    {
        /// <summary>The parsed data.</summary>
        public T? Data { get; init; }

        /// <summary>Whether parsing was successful.</summary>
        public bool Success { get; init; }

        /// <summary>Error message if parsing failed.</summary>
        public string? ErrorMessage { get; init; }

        /// <summary>Validation errors if schema validation failed.</summary>
        public IReadOnlyList<string>? ValidationErrors { get; init; }

        /// <summary>Confidence score (0.0 to 1.0) in the parsed output.</summary>
        public float Confidence { get; init; }

        /// <summary>Raw text that was parsed.</summary>
        public string RawText { get; init; } = string.Empty;

        /// <summary>Creates a successful result.</summary>
        public static StructuredOutput<T> Ok(T data, float confidence = 1.0f, string rawText = "") =>
            new() { Data = data, Success = true, Confidence = confidence, RawText = rawText };

        /// <summary>Creates a failure result.</summary>
        public static StructuredOutput<T> Fail(string error, string rawText = "") =>
            new() { Success = false, ErrorMessage = error, RawText = rawText, Confidence = 0f };
    }

    /// <summary>
    /// A prompt template with variable placeholders.
    /// </summary>
    public record PromptTemplate
    {
        /// <summary>Unique template identifier.</summary>
        public string Id { get; init; } = Guid.NewGuid().ToString("N");

        /// <summary>Template name for human reference.</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>The template content with {{variable}} placeholders.</summary>
        public string Template { get; init; } = string.Empty;

        /// <summary>Default values for template variables.</summary>
        public IReadOnlyDictionary<string, string>? Defaults { get; init; }

        /// <summary>Required variable names.</summary>
        public IReadOnlyList<string>? RequiredVariables { get; init; }

        /// <summary>Template version for A/B testing.</summary>
        public int Version { get; init; } = 1;

        /// <summary>Hash of the template content for caching.</summary>
        public string ContentHash { get; init; } = string.Empty;

        /// <summary>The task type this template is designed for.</summary>
        public LlmTaskType TaskType { get; init; } = LlmTaskType.QueryAnswering;
    }

    /// <summary>
    /// Defines a function that can be called by the LLM.
    /// </summary>
    public record FunctionDefinition
    {
        /// <summary>The function name.</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>Description of what the function does.</summary>
        public string Description { get; init; } = string.Empty;

        /// <summary>Parameters the function accepts.</summary>
        public IReadOnlyList<FunctionParameter> Parameters { get; init; } = Array.Empty<FunctionParameter>();

        /// <summary>Whether all parameters are required.</summary>
        public bool AllParametersRequired { get; init; } = true;
    }

    /// <summary>
    /// Represents a function call made by the LLM.
    /// </summary>
    public record FunctionCall
    {
        /// <summary>The name of the function to call.</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>Unique identifier for this call.</summary>
        public string Id { get; init; } = Guid.NewGuid().ToString("N");

        /// <summary>Arguments as a JSON string.</summary>
        public string Arguments { get; init; } = "{}";

        /// <summary>Parsed arguments dictionary.</summary>
        public IReadOnlyDictionary<string, object>? ParsedArguments { get; init; }
    }

    /// <summary>
    /// Defines a parameter for a function definition.
    /// </summary>
    public record FunctionParameter
    {
        /// <summary>Parameter name.</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>Parameter description.</summary>
        public string Description { get; init; } = string.Empty;

        /// <summary>Parameter type (string, number, boolean, object, array).</summary>
        public string Type { get; init; } = "string";

        /// <summary>Whether this parameter is required.</summary>
        public bool Required { get; init; } = true;

        /// <summary>Allowed enum values, if applicable.</summary>
        public IReadOnlyList<string>? EnumValues { get; init; }

        /// <summary>Default value if not provided.</summary>
        public object? DefaultValue { get; init; }
    }

    /// <summary>
    /// Represents a single turn in a multi-turn conversation.
    /// </summary>
    public record ConversationTurn
    {
        /// <summary>The user's message.</summary>
        public ChatMessage UserMessage { get; init; } = new();

        /// <summary>The assistant's response.</summary>
        public ChatMessage? AssistantMessage { get; init; }

        /// <summary>Token usage for this turn.</summary>
        public TokenUsage? Usage { get; init; }

        /// <summary>Timestamp of this turn.</summary>
        public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>Latency of the assistant response.</summary>
        public TimeSpan? Latency { get; init; }

        /// <summary>Which provider served this turn.</summary>
        public string? Provider { get; init; }
    }

    /// <summary>
    /// Configuration for a specific LLM provider.
    /// </summary>
    public record ProviderConfig
    {
        /// <summary>The provider mode.</summary>
        public LlmProviderMode Mode { get; init; } = LlmProviderMode.Hybrid;

        /// <summary>API key or access token.</summary>
        public string? ApiKey { get; init; }

        /// <summary>Base URL for the provider API.</summary>
        public string? BaseUrl { get; init; }

        /// <summary>Default model to use.</summary>
        public string? DefaultModel { get; init; }

        /// <summary>Request timeout in seconds.</summary>
        public int TimeoutSeconds { get; init; } = 60;

        /// <summary>Maximum number of retry attempts.</summary>
        public int MaxRetries { get; init; } = 3;

        /// <summary>Rate limit configuration.</summary>
        public RateLimitConfig? RateLimit { get; init; }

        /// <summary>Whether this provider is enabled.</summary>
        public bool Enabled { get; init; } = true;

        /// <summary>Additional provider-specific settings.</summary>
        public IReadOnlyDictionary<string, string>? ExtraSettings { get; init; }
    }

    /// <summary>
    /// Configuration for retry behavior.
    /// </summary>
    public record RetryConfig
    {
        /// <summary>Maximum number of retry attempts.</summary>
        public int MaxRetries { get; init; } = 3;

        /// <summary>Initial delay before the first retry.</summary>
        public TimeSpan InitialDelay { get; init; } = TimeSpan.FromSeconds(1);

        /// <summary>Backoff multiplier for subsequent retries.</summary>
        public double BackoffMultiplier { get; init; } = 2.0;

        /// <summary>Maximum delay between retries.</summary>
        public TimeSpan MaxDelay { get; init; } = TimeSpan.FromSeconds(30);

        /// <summary>Whether to add jitter to retry delays.</summary>
        public bool UseJitter { get; init; } = true;

        /// <summary>HTTP status codes that should trigger a retry.</summary>
        public IReadOnlyList<int> RetryableStatusCodes { get; init; } =
            new[] { 429, 500, 502, 503, 504 };
    }

    /// <summary>
    /// A cached LLM response entry.
    /// </summary>
    public record CacheEntry
    {
        /// <summary>The cached response.</summary>
        public LlmResponse Response { get; init; } = new();

        /// <summary>Hash of the prompt used as cache key.</summary>
        public string PromptHash { get; init; } = string.Empty;

        /// <summary>When this entry was created.</summary>
        public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>When this entry expires.</summary>
        public DateTimeOffset ExpiresAt { get; init; }

        /// <summary>Number of times this entry has been accessed.</summary>
        public int AccessCount { get; init; }

        /// <summary>Last time this entry was accessed.</summary>
        public DateTimeOffset LastAccessedAt { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>Size of the cached data in bytes.</summary>
        public long SizeBytes { get; init; }

        /// <summary>The embedding for semantic similarity cache lookups.</summary>
        public EmbeddingVector? SemanticEmbedding { get; init; }

        /// <summary>Whether this entry has been invalidated.</summary>
        public bool IsInvalidated { get; init; }
    }

    /// <summary>
    /// Result of an embedding operation.
    /// </summary>
    public record EmbeddingResult
    {
        /// <summary>The generated embedding vector.</summary>
        public EmbeddingVector Vector { get; init; } = new();

        /// <summary>Token usage for the embedding request.</summary>
        public TokenUsage Usage { get; init; } = new();

        /// <summary>Model used to generate the embedding.</summary>
        public string Model { get; init; } = string.Empty;

        /// <summary>Latency of the embedding request.</summary>
        public TimeSpan Latency { get; init; }
    }

    /// <summary>
    /// Result of a similarity comparison between two texts or embeddings.
    /// </summary>
    public record SimilarityResult
    {
        /// <summary>Similarity score (0.0 to 1.0).</summary>
        public float Score { get; init; }

        /// <summary>Whether the similarity exceeds the threshold.</summary>
        public bool IsMatch { get; init; }

        /// <summary>Distance metric used (cosine, euclidean, dot).</summary>
        public string Metric { get; init; } = "cosine";

        /// <summary>The first text/embedding compared.</summary>
        public string? Text1 { get; init; }

        /// <summary>The second text/embedding compared.</summary>
        public string? Text2 { get; init; }
    }

    /// <summary>
    /// Result of intent classification from natural language input.
    /// </summary>
    public record IntentClassificationResult
    {
        /// <summary>The classified intent.</summary>
        public string Intent { get; init; } = string.Empty;

        /// <summary>Confidence score (0.0 to 1.0).</summary>
        public float Confidence { get; init; }

        /// <summary>Extracted entities from the input.</summary>
        public IReadOnlyList<EntityExtractionResult>? Entities { get; init; }

        /// <summary>Extracted parameters.</summary>
        public IReadOnlyDictionary<string, object>? Parameters { get; init; }

        /// <summary>The original input text.</summary>
        public string InputText { get; init; } = string.Empty;

        /// <summary>Alternative intents considered.</summary>
        public IReadOnlyList<(string Intent, float Confidence)>? Alternatives { get; init; }
    }

    /// <summary>
    /// Result of entity extraction from text.
    /// </summary>
    public record EntityExtractionResult
    {
        /// <summary>The entity text.</summary>
        public string Text { get; init; } = string.Empty;

        /// <summary>The entity type (person, location, object, parameter, etc.).</summary>
        public string Type { get; init; } = string.Empty;

        /// <summary>Start position in the source text.</summary>
        public int StartIndex { get; init; }

        /// <summary>End position in the source text.</summary>
        public int EndIndex { get; init; }

        /// <summary>Confidence score (0.0 to 1.0).</summary>
        public float Confidence { get; init; }

        /// <summary>Normalized value if applicable.</summary>
        public string? NormalizedValue { get; init; }

        /// <summary>Additional properties of the entity.</summary>
        public IReadOnlyDictionary<string, string>? Properties { get; init; }
    }

    /// <summary>
    /// A summary of a conversation generated by the LLM.
    /// </summary>
    public record ConversationSummary
    {
        /// <summary>The summary text.</summary>
        public string Summary { get; init; } = string.Empty;

        /// <summary>Key topics discussed.</summary>
        public IReadOnlyList<string>? Topics { get; init; }

        /// <summary>Entities mentioned.</summary>
        public IReadOnlyList<string>? Entities { get; init; }

        /// <summary>Number of turns summarized.</summary>
        public int TurnCount { get; init; }

        /// <summary>Token count of the summary.</summary>
        public int SummaryTokenCount { get; init; }

        /// <summary>Token count of the original conversation.</summary>
        public int OriginalTokenCount { get; init; }

        /// <summary>Compression ratio.</summary>
        public float CompressionRatio => OriginalTokenCount > 0
            ? (float)SummaryTokenCount / OriginalTokenCount
            : 0f;
    }

    /// <summary>
    /// Tracks cost information for LLM usage.
    /// </summary>
    public record CostEntry
    {
        /// <summary>Unique identifier for this cost entry.</summary>
        public string Id { get; init; } = Guid.NewGuid().ToString("N");

        /// <summary>Provider that incurred the cost.</summary>
        public string Provider { get; init; } = string.Empty;

        /// <summary>Model that was used.</summary>
        public string Model { get; init; } = string.Empty;

        /// <summary>Number of input tokens.</summary>
        public int InputTokens { get; init; }

        /// <summary>Number of output tokens.</summary>
        public int OutputTokens { get; init; }

        /// <summary>Cost in USD.</summary>
        public decimal CostUsd { get; init; }

        /// <summary>Timestamp of the request.</summary>
        public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>The task type that incurred this cost.</summary>
        public LlmTaskType TaskType { get; init; }

        /// <summary>Session or project identifier.</summary>
        public string? SessionId { get; init; }
    }

    /// <summary>
    /// Represents a token budget for cost management.
    /// </summary>
    public record TokenBudget
    {
        /// <summary>Maximum tokens allowed per day.</summary>
        public int? DailyTokenLimit { get; init; }

        /// <summary>Maximum cost allowed per day in USD.</summary>
        public decimal? DailyCostLimitUsd { get; init; }

        /// <summary>Maximum tokens allowed per month.</summary>
        public int? MonthlyTokenLimit { get; init; }

        /// <summary>Maximum cost allowed per month in USD.</summary>
        public decimal? MonthlyCostLimitUsd { get; init; }

        /// <summary>Per-request token limit.</summary>
        public int? PerRequestTokenLimit { get; init; }

        /// <summary>Per-request cost limit in USD.</summary>
        public decimal? PerRequestCostLimitUsd { get; init; }

        /// <summary>Current tokens used today.</summary>
        public int CurrentDailyTokens { get; init; }

        /// <summary>Current cost today in USD.</summary>
        public decimal CurrentDailyCostUsd { get; init; }

        /// <summary>Current tokens used this month.</summary>
        public int CurrentMonthlyTokens { get; init; }

        /// <summary>Current cost this month in USD.</summary>
        public decimal CurrentMonthlyCostUsd { get; init; }

        /// <summary>Whether the daily budget is exhausted.</summary>
        public bool IsDailyExhausted =>
            (DailyTokenLimit.HasValue && CurrentDailyTokens >= DailyTokenLimit.Value) ||
            (DailyCostLimitUsd.HasValue && CurrentDailyCostUsd >= DailyCostLimitUsd.Value);

        /// <summary>Whether the monthly budget is exhausted.</summary>
        public bool IsMonthlyExhausted =>
            (MonthlyTokenLimit.HasValue && CurrentMonthlyTokens >= MonthlyTokenLimit.Value) ||
            (MonthlyCostLimitUsd.HasValue && CurrentMonthlyCostUsd >= MonthlyCostLimitUsd.Value);
    }

    /// <summary>
    /// Configuration for rate limiting a provider.
    /// </summary>
    public record RateLimitConfig
    {
        /// <summary>Maximum requests per minute.</summary>
        public int RequestsPerMinute { get; init; } = 60;

        /// <summary>Maximum tokens per minute.</summary>
        public int TokensPerMinute { get; init; } = 90000;

        /// <summary>Maximum concurrent requests.</summary>
        public int MaxConcurrent { get; init; } = 10;

        /// <summary>Burst size allowance.</summary>
        public int BurstSize { get; init; } = 5;

        /// <summary>Cooldown period after hitting rate limit.</summary>
        public TimeSpan CooldownPeriod { get; init; } = TimeSpan.FromSeconds(30);
    }

    /// <summary>
    /// Tracks the health status of an LLM provider.
    /// </summary>
    public record ProviderHealth
    {
        /// <summary>The provider name.</summary>
        public string ProviderName { get; init; } = string.Empty;

        /// <summary>Current health status.</summary>
        public HealthCheckStatus Status { get; init; } = HealthCheckStatus.Healthy;

        /// <summary>Circuit breaker state.</summary>
        public CircuitBreakerState CircuitBreakerState { get; init; } = CircuitBreakerState.Closed;

        /// <summary>Average response latency.</summary>
        public TimeSpan AverageLatency { get; init; }

        /// <summary>Success rate (0.0 to 1.0).</summary>
        public float SuccessRate { get; init; } = 1.0f;

        /// <summary>Number of consecutive failures.</summary>
        public int ConsecutiveFailures { get; init; }

        /// <summary>Total requests processed.</summary>
        public long TotalRequests { get; init; }

        /// <summary>Total failed requests.</summary>
        public long FailedRequests { get; init; }

        /// <summary>Last successful request timestamp.</summary>
        public DateTimeOffset? LastSuccessAt { get; init; }

        /// <summary>Last failure timestamp.</summary>
        public DateTimeOffset? LastFailureAt { get; init; }

        /// <summary>Last health check timestamp.</summary>
        public DateTimeOffset? LastCheckedAt { get; init; }

        /// <summary>Error message from last failure, if any.</summary>
        public string? LastError { get; init; }

        /// <summary>Current number of active requests.</summary>
        public int ActiveRequests { get; init; }

        /// <summary>Available VRAM in MB (for local providers).</summary>
        public int? AvailableVramMb { get; init; }
    }

    /// <summary>
    /// Configuration for the cost tracker.
    /// </summary>
    public record CostTrackerConfig
    {
        /// <summary>Daily budget limit in USD.</summary>
        public decimal? DailyBudgetUsd { get; init; } = 10.0m;

        /// <summary>Monthly budget limit in USD.</summary>
        public decimal? MonthlyBudgetUsd { get; init; } = 200.0m;

        /// <summary>Per-request cost ceiling in USD.</summary>
        public decimal? PerRequestCeilingUsd { get; init; } = 1.0m;

        /// <summary>Warning threshold (percentage of budget).</summary>
        public float WarningThresholdPercent { get; init; } = 80f;

        /// <summary>Whether to enforce budget limits strictly.</summary>
        public bool EnforceLimits { get; init; } = true;

        /// <summary>Persist cost data to disk.</summary>
        public bool PersistToDisk { get; init; } = true;

        /// <summary>Path for cost data storage.</summary>
        public string? StoragePath { get; init; }
    }

    /// <summary>
    /// Represents a streaming token from the LLM.
    /// </summary>
    public record StreamingToken
    {
        /// <summary>The token text.</summary>
        public string Text { get; init; } = string.Empty;

        /// <summary>Token ID if available.</summary>
        public int? TokenId { get; init; }

        /// <summary>Log probabilities if available.</summary>
        public float[]? LogProbs { get; init; }

        /// <summary>Whether this is the final token.</summary>
        public bool IsFinal { get; init; }

        /// <summary>Finish reason if this is the final token.</summary>
        public FinishReason? FinishReason { get; init; }
    }

    /// <summary>
    /// Represents a complete genome specification for the G-DNN Engine.
    /// </summary>
    public record GenomeSpecification
    {
        /// <summary>Unique genome identifier.</summary>
        public string Id { get; init; } = Guid.NewGuid().ToString("N");

        /// <summary>Human-readable name.</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>Natural language description.</summary>
        public string Description { get; init; } = string.Empty;

        /// <summary>Genome type (organoid, structure, behavior, etc.).</summary>
        public string Type { get; init; } = string.Empty;

        /// <summary>Genome parameters as key-value pairs.</summary>
        public IReadOnlyDictionary<string, object> Parameters { get; init; } =
            new Dictionary<string, object>();

        /// <summary>Parent genome IDs if mutated.</summary>
        public IReadOnlyList<string>? ParentIds { get; init; }

        /// <summary>Generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Fit score if evaluated.</summary>
        public float? FitnessScore { get; init; }

        /// <summary>Tags for categorization.</summary>
        public IReadOnlyList<string>? Tags { get; init; }
    }

    /// <summary>
    /// Represents a behavior tree node extracted from LLM output.
    /// </summary>
    public record BehaviorTreeNode
    {
        /// <summary>Unique node identifier.</summary>
        public string Id { get; init; } = Guid.NewGuid().ToString("N");

        /// <summary>Node type (selector, sequence, action, condition, decorator).</summary>
        public string NodeType { get; init; } = "action";

        /// <summary>Display name.</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>Description of what this node does.</summary>
        public string Description { get; init; } = string.Empty;

        /// <summary>Child node IDs.</summary>
        public IReadOnlyList<string>? ChildIds { get; init; }

        /// <summary>Node-specific parameters.</summary>
        public IReadOnlyDictionary<string, object>? Parameters { get; init; }

        /// <summary>Decorator type if applicable.</summary>
        public string? DecoratorType { get; init; }
    }

    /// <summary>
    /// Represents material properties extracted from LLM output.
    /// </summary>
    public record MaterialProperties
    {
        /// <summary>Material name.</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>Base color as hex string.</summary>
        public string BaseColor { get; init; } = "#FFFFFF";

        /// <summary>Metallic value (0.0 to 1.0).</summary>
        public float Metallic { get; init; }

        /// <summary>Roughness value (0.0 to 1.0).</summary>
        public float Roughness { get; init; } = 0.5f;

        /// <summary>Normal map intensity.</summary>
        public float NormalStrength { get; init; } = 1.0f;

        /// <summary>Emission color if emissive.</summary>
        public string? EmissionColor { get; init; }

        /// <summary>Emission intensity.</summary>
        public float EmissionIntensity { get; init; }

        /// <summary>Transparency/alpha value.</summary>
        public float Alpha { get; init; } = 1.0f;

        /// <summary>Index of refraction for glass-like materials.</summary>
        public float Ior { get; init; } = 1.5f;

        /// <summary>Texture generation parameters.</summary>
        public IReadOnlyDictionary<string, object>? TextureParams { get; init; }

        /// <summary>Material category.</summary>
        public string Category { get; init; } = string.Empty;

        /// <summary>Tags for the material.</summary>
        public IReadOnlyList<string>? Tags { get; init; }
    }

    /// <summary>
    /// Represents a scene description for the G-DNN Engine.
    /// </summary>
    public record SceneDescription
    {
        /// <summary>Scene name.</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>Brief description.</summary>
        public string Description { get; init; } = string.Empty;

        /// <summary>Entities in the scene.</summary>
        public IReadOnlyList<SceneEntity>? Entities { get; init; }

        /// <summary>Relationships between entities.</summary>
        public IReadOnlyList<SceneRelationship>? Relationships { get; init; }

        /// <summary>Lighting configuration.</summary>
        public string? Lighting { get; init; }

        /// <summary>Camera configuration.</summary>
        public string? Camera { get; init; }
    }

    /// <summary>
    /// An entity within a scene.
    /// </summary>
    public record SceneEntity
    {
        /// <summary>Entity name.</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>Entity type.</summary>
        public string Type { get; init; } = string.Empty;

        /// <summary>Position in 3D space.</summary>
        public (float X, float Y, float Z)? Position { get; init; }

        /// <summary>Rotation in Euler angles.</summary>
        public (float X, float Y, float Z)? Rotation { get; init; }

        /// <summary>Scale factors.</summary>
        public (float X, float Y, float Z)? Scale { get; init; }

        /// <summary>Associated material name.</summary>
        public string? Material { get; init; }

        /// <summary>Associated behavior name.</summary>
        public string? Behavior { get; init; }

        /// <summary>Additional properties.</summary>
        public IReadOnlyDictionary<string, object>? Properties { get; init; }
    }

    /// <summary>
    /// Represents a relationship between two scene entities.
    /// </summary>
    public record SceneRelationship
    {
        /// <summary>Source entity name.</summary>
        public string Source { get; init; } = string.Empty;

        /// <summary>Target entity name.</summary>
        public string Target { get; init; } = string.Empty;

        /// <summary>Relationship type (parent, child, attached-to, follows, etc.).</summary>
        public string Type { get; init; } = string.Empty;

        /// <summary>Additional properties.</summary>
        public IReadOnlyDictionary<string, object>? Properties { get; init; }
    }
}
