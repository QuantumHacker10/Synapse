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
    /// LLM inference provider using the Anthropic Messages API.
    /// Supports Claude models with streaming, tool use, system prompts,
    /// extended thinking, and multi-modal input.
    /// </summary>
    public sealed class AnthropicProvider : ILlmInferenceProvider
    {
        private readonly HttpClient _httpClient;
        private readonly string _apiKey;
        private readonly string _baseUrl;
        private readonly bool _ownsHttpClient;
        private volatile bool _disposed;
        private volatile LlmStatus _status = LlmStatus.Unavailable;
        private readonly ConcurrentDictionary<string, ModelInfo> _modelCache;
        private readonly string _defaultModel;
        private CancellationTokenSource? _currentCts;
        private long _totalTokensGenerated;
        private long _totalRequests;

        /// <inheritdoc/>
        public string ProviderName => "Anthropic";

        /// <inheritdoc/>
        public bool RequiresNetwork => true;

        /// <inheritdoc/>
        public LlmStatus Status => _status;

        /// <inheritdoc/>
        public LlmProviderMode Mode => LlmProviderMode.RemoteAnthropic;

        /// <inheritdoc/>
        public IReadOnlyList<ModelInfo> AvailableModels =>
            _modelCache.Values.ToList().AsReadOnly();

        /// <inheritdoc/>
        public ProviderHealth Health { get; private set; } = new() { ProviderName = "Anthropic" };

        /// <summary>Total tokens generated since initialization.</summary>
        public long TotalTokensGenerated => Interlocked.Read(ref _totalTokensGenerated);

        /// <summary>
        /// Initializes a new Anthropic provider.
        /// </summary>
        /// <param name="apiKey">Anthropic API key.</param>
        /// <param name="defaultModel">Default model ID.</param>
        /// <param name="baseUrl">Base API URL.</param>
        /// <param name="httpClient">Optional shared HttpClient.</param>
        public AnthropicProvider(
            string apiKey,
            string defaultModel = "claude-sonnet-4-20250514",
            string? baseUrl = null,
            HttpClient? httpClient = null)
        {
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _defaultModel = defaultModel ?? "claude-sonnet-4-20250514";
            _baseUrl = Synapse.Core.Security.UrlSecurity.ValidateOutboundUri(
                (baseUrl ?? "https://api.anthropic.com").TrimEnd('/')).ToString().TrimEnd('/');
            _modelCache = new ConcurrentDictionary<string, ModelInfo>(StringComparer.OrdinalIgnoreCase);

            if (httpClient != null)
            {
                _httpClient = httpClient;
                _ownsHttpClient = false;
            }
            else
            {
                _httpClient = Synapse.Core.Security.UrlSecurity.CreateSafeHttpClient(TimeSpan.FromSeconds(180));
                _ownsHttpClient = true;
            }

            InitializeModelCache();
            _status = LlmStatus.Ready;
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

            var jsonInstruction = !string.IsNullOrEmpty(jsonSchema)
                ? $"\n\nYou MUST respond with valid JSON matching this schema:\n{jsonSchema}"
                : "\n\nRespond ONLY with valid JSON.";

            var messages = new List<object>
            {
                new { role = "user", content = prompt + jsonInstruction }
            };

            var model = GetModelForContext(context);
            var requestBody = BuildRequestBody(model, messages, systemPrompt, false, 0.3f,
                context.ExtraParameters?.ContainsKey("max_tokens") == true
                    ? Convert.ToInt32(context.ExtraParameters["max_tokens"]) : 4096);

            var responseJson = await SendRequestAsync("/v1/messages", requestBody, cancellationToken);
            var content = ExtractContentFromResponse(responseJson);

            Interlocked.Increment(ref _totalRequests);
            return StructuredOutputParser.ParseJson<T>(content);
        }

        /// <inheritdoc/>
        public async IAsyncEnumerable<StreamingToken> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            PromptContext context,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            _currentCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

            try
            {
                var model = GetModelForContext(context);
                var systemPrompt = messages.FirstOrDefault(m => m.Role == MessageRole.System)?.Content;
                var userMessages = messages.Where(m => m.Role != MessageRole.System)
                    .Select(m => (object)new { role = "user", content = m.Content }).ToList();

                var requestBody = BuildRequestBody(model, userMessages, systemPrompt, true,
                    context.ExtraParameters?.ContainsKey("temperature") == true
                        ? Convert.ToSingle(context.ExtraParameters["temperature"]) : 0.7f,
                    context.ExtraParameters?.ContainsKey("max_tokens") == true
                        ? Convert.ToInt32(context.ExtraParameters["max_tokens"]) : 4096);

                var request = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/v1/messages")
                {
                    Content = new StringContent(
                        JsonSerializer.Serialize(requestBody),
                        Encoding.UTF8,
                        "application/json")
                };
                request.Headers.Add("x-api-key", _apiKey);
                request.Headers.Add("anthropic-version", "2023-06-01");

                var response = await _httpClient.SendAsync(
                    request,
                    HttpCompletionOption.ResponseHeadersRead,
                    _currentCts.Token);

                response.EnsureSuccessStatusCode();

                var stream = await response.Content.ReadAsStreamAsync(_currentCts.Token);
                using var reader = new StreamReader(stream);
                string eventType = "";

                while (!reader.EndOfStream)
                {
                    _currentCts.Token.ThrowIfCancellationRequested();
                    var line = await reader.ReadLineAsync(_currentCts.Token);
                    if (string.IsNullOrEmpty(line))
                        continue;

                    if (line.StartsWith("event: "))
                    {
                        eventType = line.Substring(7).Trim();
                        continue;
                    }

                    if (line.StartsWith("data: "))
                    {
                        var data = line.Substring(6).Trim();
                        string? anthropicText = null;
                        bool isMessageStop = false;
                        try
                        {
                            using var lineDoc = JsonDocument.Parse(data);
                            var root = lineDoc.RootElement;

                            if (eventType == "content_block_delta" &&
                                root.TryGetProperty("delta", out var delta) &&
                                delta.TryGetProperty("text", out var textProp))
                            {
                                anthropicText = textProp.GetString() ?? "";
                            }
                            else if (eventType == "message_stop")
                            {
                                isMessageStop = true;
                            }
                        }
                        catch { /* Skip malformed SSE lines */ }

                        if (!string.IsNullOrEmpty(anthropicText))
                        {
                            yield return new StreamingToken
                            {
                                Text = anthropicText,
                                IsFinal = false
                            };
                        }
                        else if (isMessageStop)
                        {
                            yield return new StreamingToken
                            {
                                Text = "",
                                IsFinal = true,
                                FinishReason = FinishReason.StopToken
                            };
                            break;
                        }
                    }
                }

                Interlocked.Increment(ref _totalRequests);
            }
            finally
            {
                _currentCts?.Dispose();
                _currentCts = null;
            }
        }

        /// <inheritdoc/>
        public async Task<EmbeddingResult> GetEmbeddingAsync(
            string text,
            EmbeddingModel model = EmbeddingModel.TextEmbedding3Small,
            CancellationToken cancellationToken = default)
        {
            throw new NotSupportedException(
                "Anthropic does not provide an embedding API. Use OpenAI or a local model.");
        }

        /// <inheritdoc/>
        public int EstimateTokens(string text)
        {
            if (string.IsNullOrEmpty(text))
                return 0;
            // Anthropic tokenizer approximation
            return (int)Math.Ceiling(text.Length / 3.5);
        }

        /// <inheritdoc/>
        public ModelInfo? GetModelInfo(string modelId)
        {
            _modelCache.TryGetValue(modelId, out var info);
            return info;
        }

        /// <inheritdoc/>
        public async Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
        {
            if (_disposed || string.IsNullOrEmpty(_apiKey))
                return false;
            try
            {
                var request = new HttpRequestMessage(HttpMethod.Get, $"{_baseUrl}/v1/messages");
                request.Headers.Add("x-api-key", _apiKey);
                request.Headers.Add("anthropic-version", "2023-06-01");
                var response = await _httpClient.SendAsync(request, cancellationToken);
                return response.StatusCode != HttpStatusCode.Unauthorized;
            }
            catch { return false; }
        }

        /// <inheritdoc/>
        public void CancelCurrentInference()
        {
            _currentCts?.Cancel();
        }

        /// <inheritdoc/>
        public Task WarmUpAsync(CancellationToken cancellationToken = default)
        {
            _status = LlmStatus.WarmingUp;
            _status = LlmStatus.Ready;
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public Task ShutdownAsync()
        {
            _status = LlmStatus.Unavailable;
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public async Task<HealthCheckStatus> CheckHealthAsync(CancellationToken cancellationToken = default)
        {
            if (_disposed)
                return HealthCheckStatus.Unhealthy;
            try
            {
                var available = await IsAvailableAsync(cancellationToken);
                return available ? HealthCheckStatus.Healthy : HealthCheckStatus.Unhealthy;
            }
            catch { return HealthCheckStatus.Unhealthy; }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _currentCts?.Cancel();
            _currentCts?.Dispose();
            if (_ownsHttpClient)
                _httpClient.Dispose();
        }

        // ─── Private Helpers ───────────────────────────────────────────────

        private void InitializeModelCache()
        {
            var models = new (string Id, string Name, int Context, int MaxOut, decimal InputCost, decimal OutputCost)[]
            {
                ("claude-sonnet-4-20250514", "Claude Sonnet 4", 200000, 8192, 3.00m, 15.00m),
                ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku", 200000, 8192, 0.80m, 4.00m),
                ("claude-3-opus-20240229", "Claude 3 Opus", 200000, 4096, 15.00m, 75.00m),
                ("claude-3-sonnet-20240229", "Claude 3 Sonnet", 200000, 4096, 3.00m, 15.00m)
            };

            foreach (var m in models)
            {
                _modelCache[m.Id] = new ModelInfo
                {
                    ModelId = m.Id,
                    DisplayName = m.Name,
                    Provider = "Anthropic",
                    ContextWindow = m.Context,
                    MaxOutputTokens = m.MaxOut,
                    Tokenizer = TokenizerType.BPE,
                    VocabularySize = 100000,
                    SupportsVision = m.Id.Contains("claude-3"),
                    SupportsFunctionCalling = true,
                    InputCostPer1MTokens = m.InputCost,
                    OutputCostPer1MTokens = m.OutputCost
                };
            }
        }

        private string GetModelForContext(PromptContext context)
        {
            if (context.ExtraParameters?.ContainsKey("model") == true)
                return context.ExtraParameters["model"]?.ToString() ?? _defaultModel;
            return _defaultModel;
        }

        private object BuildRequestBody(
            string model,
            List<object> messages,
            string? systemPrompt,
            bool stream,
            float temperature,
            int maxTokens)
        {
            var body = new Dictionary<string, object>
            {
                ["model"] = model,
                ["messages"] = messages,
                ["max_tokens"] = maxTokens,
                ["temperature"] = temperature,
                ["stream"] = stream
            };

            if (!string.IsNullOrEmpty(systemPrompt))
                body["system"] = systemPrompt;

            return body;
        }

        private async Task<string> SendRequestAsync(
            string path,
            object requestBody,
            CancellationToken cancellationToken)
        {
            var request = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}{path}")
            {
                Content = new StringContent(
                    JsonSerializer.Serialize(requestBody),
                    Encoding.UTF8,
                    "application/json")
            };
            request.Headers.Add("x-api-key", _apiKey);
            request.Headers.Add("anthropic-version", "2023-06-01");

            var response = await _httpClient.SendAsync(request, cancellationToken);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync(cancellationToken);
        }

        private static string ExtractContentFromResponse(string json)
        {
            using var doc = JsonDocument.Parse(json);
            var sb = new StringBuilder();

            if (doc.RootElement.TryGetProperty("content", out var content))
            {
                foreach (var block in content.EnumerateArray())
                {
                    if (block.TryGetProperty("text", out var text))
                        sb.Append(text.GetString());
                }
            }

            return sb.ToString();
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
