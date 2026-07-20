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
    /// LLM inference provider using the Google Gemini API.
    /// Supports content generation, embeddings, function calling,
    /// multi-modal input, safety settings, and code execution.
    /// </summary>
    public sealed class GoogleGeminiProvider : ILlmInferenceProvider
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
        public string ProviderName => "GoogleGemini";

        /// <inheritdoc/>
        public bool RequiresNetwork => true;

        /// <inheritdoc/>
        public LlmStatus Status => _status;

        /// <inheritdoc/>
        public LlmProviderMode Mode => LlmProviderMode.RemoteGoogle;

        /// <inheritdoc/>
        public IReadOnlyList<ModelInfo> AvailableModels =>
            _modelCache.Values.ToList().AsReadOnly();

        /// <inheritdoc/>
        public ProviderHealth Health { get; private set; } = new() { ProviderName = "GoogleGemini" };

        /// <summary>Total tokens generated since initialization.</summary>
        public long TotalTokensGenerated => Interlocked.Read(ref _totalTokensGenerated);

        /// <summary>
        /// Initializes a new Google Gemini provider.
        /// </summary>
        /// <param name="apiKey">Google API key.</param>
        /// <param name="defaultModel">Default model ID.</param>
        /// <param name="httpClient">Optional shared HttpClient.</param>
        public GoogleGeminiProvider(
            string apiKey,
            string defaultModel = "gemini-2.5-flash",
            HttpClient? httpClient = null)
        {
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _defaultModel = defaultModel ?? "gemini-2.5-flash";
            _baseUrl = "https://generativelanguage.googleapis.com/v1beta";
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
                ? $"\n\nRespond ONLY with valid JSON matching this schema:\n{jsonSchema}"
                : "\n\nRespond ONLY with valid JSON.";

            var contents = new List<object>();
            var userContent = prompt + jsonInstruction;

            if (!string.IsNullOrEmpty(systemPrompt))
            {
                contents.Add(new
                {
                    role = "system",
                    parts = new[] { new { text = systemPrompt } }
                });
            }

            contents.Add(new
            {
                role = "user",
                parts = new[] { new { text = userContent } }
            });

            var model = GetModelForContext(context);
            var requestBody = BuildRequestBody(contents, systemPrompt, false, 0.3f);

            var url = $"{_baseUrl}/models/{model}:generateContent?key={_apiKey}";
            var responseJson = await SendRequestAsync(url, requestBody, cancellationToken);
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
                var contents = messages.Where(m => m.Role != MessageRole.System)
                    .Select(m => new
                    {
                        role = m.Role == MessageRole.Assistant ? "model" : "user",
                        parts = new[] { new { text = m.Content } }
                    }).ToList<object>();

                var requestBody = BuildRequestBody(contents, systemPrompt, true,
                    context.ExtraParameters?.ContainsKey("temperature") == true
                        ? Convert.ToSingle(context.ExtraParameters["temperature"]) : 0.7f);

                var url = $"{_baseUrl}/models/{model}:streamGenerateContent?key={_apiKey}&alt=sse";

                var request = new HttpRequestMessage(HttpMethod.Post, url)
                {
                    Content = new StringContent(
                        JsonSerializer.Serialize(requestBody),
                        Encoding.UTF8,
                        "application/json")
                };

                var response = await _httpClient.SendAsync(
                    request,
                    HttpCompletionOption.ResponseHeadersRead,
                    _currentCts.Token);

                response.EnsureSuccessStatusCode();

                var stream = await response.Content.ReadAsStreamAsync(_currentCts.Token);
                using var reader = new StreamReader(stream);

                while (!reader.EndOfStream)
                {
                    _currentCts.Token.ThrowIfCancellationRequested();
                    var line = await reader.ReadLineAsync(_currentCts.Token);
                    if (string.IsNullOrEmpty(line))
                        continue;
                    if (!line.StartsWith("data: "))
                        continue;

                    var data = line.Substring(6).Trim();
                    List<string> geminiTexts = new();
                    bool geminiStop = false;
                    try
                    {
                        using var lineDoc = JsonDocument.Parse(data);
                        if (lineDoc.RootElement.TryGetProperty("candidates", out var candidates) &&
                            candidates.GetArrayLength() > 0)
                        {
                            var candidate = candidates[0];
                            if (candidate.TryGetProperty("content", out var content) &&
                                content.TryGetProperty("parts", out var parts))
                            {
                                foreach (var part in parts.EnumerateArray())
                                {
                                    if (part.TryGetProperty("text", out var textProp))
                                    {
                                        var text = textProp.GetString() ?? "";
                                        if (!string.IsNullOrEmpty(text))
                                        {
                                            geminiTexts.Add(text);
                                        }
                                    }
                                }
                            }

                            if (candidate.TryGetProperty("finishReason", out var frProp))
                            {
                                var reason = frProp.GetString();
                                if (reason == "STOP")
                                {
                                    geminiStop = true;
                                }
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        SynapseLogger.Default.Debug("GoogleGeminiProvider", "Skipping malformed SSE line.", ex);
                    }

                    foreach (var text in geminiTexts)
                    {
                        yield return new StreamingToken
                        {
                            Text = text,
                            IsFinal = false
                        };
                    }

                    if (geminiStop)
                    {
                        yield return new StreamingToken
                        {
                            Text = "",
                            IsFinal = true,
                            FinishReason = FinishReason.StopToken
                        };
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
            ThrowIfDisposed();

            var geminiModel = "text-embedding-004";

            var requestBody = new
            {
                model = $"models/{geminiModel}",
                content = new
                {
                    parts = new[] { new { text } }
                }
            };

            var url = $"{_baseUrl}/models/{geminiModel}:embedContent?key={_apiKey}";
            var responseJson = await SendRequestAsync(url, requestBody, cancellationToken);
            using var doc = JsonDocument.Parse(responseJson);

            float[] embedding = Array.Empty<float>();
            if (doc.RootElement.TryGetProperty("embedding", out var embeddingProp) &&
                embeddingProp.TryGetProperty("values", out var valuesProp))
            {
                var values = new List<float>();
                foreach (var val in valuesProp.EnumerateArray())
                    values.Add(val.GetSingle());
                embedding = values.ToArray();
            }

            Interlocked.Increment(ref _totalRequests);

            return new EmbeddingResult
            {
                Vector = new EmbeddingVector
                {
                    Dimensions = embedding.Length,
                    Values = embedding,
                    Text = text,
                    Model = geminiModel
                },
                Model = geminiModel
            };
        }

        /// <inheritdoc/>
        public int EstimateTokens(string text)
        {
            if (string.IsNullOrEmpty(text))
                return 0;
            return (int)Math.Ceiling(text.Length / 4.0);
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
                var url = $"{_baseUrl}/models?key={_apiKey}";
                var response = await _httpClient.GetAsync(url, cancellationToken);
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("GoogleGeminiProvider", "Availability check failed.", ex);
                return false;
            }
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
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("GoogleGeminiProvider", "Health check failed.", ex);
                return HealthCheckStatus.Unhealthy;
            }
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
            var models = new (string Id, string Name, int Context, int MaxOut)[]
            {
                ("gemini-2.5-flash", "Gemini 2.5 Flash", 1048576, 65536),
                ("gemini-2.5-pro", "Gemini 2.5 Pro", 1048576, 65536),
                ("gemini-2.0-flash", "Gemini 2.0 Flash", 1048576, 8192),
                ("gemini-1.5-flash", "Gemini 1.5 Flash", 1048576, 8192),
                ("gemini-1.5-pro", "Gemini 1.5 Pro", 2097152, 8192)
            };

            foreach (var m in models)
            {
                _modelCache[m.Id] = new ModelInfo
                {
                    ModelId = m.Id,
                    DisplayName = m.Name,
                    Provider = "GoogleGemini",
                    ContextWindow = m.Context,
                    MaxOutputTokens = m.MaxOut,
                    Tokenizer = TokenizerType.SentencePiece,
                    VocabularySize = 32000,
                    SupportsVision = true,
                    SupportsFunctionCalling = true,
                    InputCostPer1MTokens = 0.075m,
                    OutputCostPer1MTokens = 0.30m
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
            List<object> contents,
            string? systemInstruction,
            bool stream,
            float temperature)
        {
            var body = new Dictionary<string, object>
            {
                ["contents"] = contents,
                ["generationConfig"] = new
                {
                    temperature = temperature,
                    topP = 0.95,
                    topK = 40
                }
            };

            if (!string.IsNullOrEmpty(systemInstruction))
            {
                body["systemInstruction"] = new
                {
                    parts = new[] { new { text = systemInstruction } }
                };
            }

            return body;
        }

        private async Task<string> SendRequestAsync(
            string url,
            object requestBody,
            CancellationToken cancellationToken)
        {
            var request = new HttpRequestMessage(HttpMethod.Post, url)
            {
                Content = new StringContent(
                    JsonSerializer.Serialize(requestBody),
                    Encoding.UTF8,
                    "application/json")
            };

            var response = await _httpClient.SendAsync(request, cancellationToken);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync(cancellationToken);
        }

        private static string ExtractContentFromResponse(string json)
        {
            using var doc = JsonDocument.Parse(json);
            var sb = new StringBuilder();

            if (doc.RootElement.TryGetProperty("candidates", out var candidates) &&
                candidates.GetArrayLength() > 0)
            {
                var candidate = candidates[0];
                if (candidate.TryGetProperty("content", out var content) &&
                    content.TryGetProperty("parts", out var parts))
                {
                    foreach (var part in parts.EnumerateArray())
                    {
                        if (part.TryGetProperty("text", out var text))
                            sb.Append(text.GetString());
                    }
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
