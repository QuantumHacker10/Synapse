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
    /// LLM inference provider using the OpenAI API.
    /// Supports chat completions, function calling, streaming via SSE, structured output,
    /// embeddings, vision, rate limit handling, and Azure OpenAI integration.
    /// </summary>
    public sealed class OpenAiProvider : ILlmInferenceProvider
    {
        private readonly HttpClient _httpClient;
        private readonly string _apiKey;
        private readonly string _baseUrl;
        private readonly bool _ownsHttpClient;
        private volatile bool _disposed;
        private volatile LlmStatus _status = LlmStatus.Unavailable;
        private readonly ConcurrentDictionary<string, ModelInfo> _modelCache;
        private readonly string _defaultModel;
        private readonly bool _isAzureEndpoint;
        private readonly string? _azureDeploymentName;
        private readonly string? _azureApiVersion;
        private CancellationTokenSource? _currentCts;
        private long _totalTokensGenerated;
        private long _totalRequests;
        private readonly object _rateLimitLock = new();
        private DateTimeOffset _rateLimitResetTime = DateTimeOffset.MinValue;
        private int _remainingRequests;
        private int _remainingTokens;

        /// <inheritdoc/>
        public string ProviderName => "OpenAI";

        /// <inheritdoc/>
        public bool RequiresNetwork => true;

        /// <inheritdoc/>
        public LlmStatus Status => _status;

        /// <inheritdoc/>
        public LlmProviderMode Mode => _isAzureEndpoint ? LlmProviderMode.RemoteAzure : LlmProviderMode.RemoteOpenAI;

        /// <inheritdoc/>
        public IReadOnlyList<ModelInfo> AvailableModels =>
            _modelCache.Values.ToList().AsReadOnly();

        /// <inheritdoc/>
        public ProviderHealth Health { get; private set; } = new() { ProviderName = "OpenAI" };

        /// <summary>Total tokens generated since initialization.</summary>
        public long TotalTokensGenerated => Interlocked.Read(ref _totalTokensGenerated);

        /// <summary>
        /// Initializes a new OpenAI provider.
        /// </summary>
        /// <param name="apiKey">OpenAI API key.</param>
        /// <param name="defaultModel">Default model ID.</param>
        /// <param name="baseUrl">Base API URL (overrides for Azure).</param>
        /// <param name="isAzure">Whether this is an Azure OpenAI endpoint.</param>
        /// <param name="azureDeploymentName">Azure deployment name.</param>
        /// <param name="azureApiVersion">Azure API version.</param>
        /// <param name="httpClient">Optional shared HttpClient.</param>
        public OpenAiProvider(
            string apiKey,
            string defaultModel = "gpt-4o-mini",
            string? baseUrl = null,
            bool isAzure = false,
            string? azureDeploymentName = null,
            string? azureApiVersion = null,
            HttpClient? httpClient = null)
        {
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _defaultModel = defaultModel ?? "gpt-4o-mini";
            _isAzureEndpoint = isAzure;
            _azureDeploymentName = azureDeploymentName;
            _azureApiVersion = azureApiVersion ?? "2024-02-15-preview";
            _modelCache = new ConcurrentDictionary<string, ModelInfo>(StringComparer.OrdinalIgnoreCase);

            if (isAzure)
            {
                if (string.IsNullOrWhiteSpace(baseUrl))
                    throw new ArgumentException("Azure OpenAI base URL is required.", nameof(baseUrl));
                _baseUrl = Synapse.Core.Security.UrlSecurity.ValidateOutboundUri(baseUrl.TrimEnd('/')).ToString().TrimEnd('/');
            }
            else
            {
                _baseUrl = Synapse.Core.Security.UrlSecurity.ValidateOutboundUri(
                    (baseUrl ?? "https://api.openai.com/v1").TrimEnd('/')).ToString().TrimEnd('/');
            }

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

            var messages = new List<object>();
            if (!string.IsNullOrEmpty(systemPrompt))
                messages.Add(new { role = "system", content = systemPrompt });

            var userContent = !string.IsNullOrEmpty(jsonSchema)
                ? $"{prompt}\n\nRespond ONLY with valid JSON matching this schema:\n{jsonSchema}"
                : $"{prompt}\n\nRespond ONLY with valid JSON.";

            messages.Add(new { role = "user", content = userContent });

            var model = GetModelForContext(context);

            var requestBody = new
            {
                model = model,
                messages = messages.ToArray(),
                temperature = Math.Max(0, (context.ExtraParameters?.ContainsKey("temperature") == true
                    ? Convert.ToSingle(context.ExtraParameters["temperature"]) : 0.3f)),
                response_format = new { type = "json_object" },
                max_tokens = context.ExtraParameters?.ContainsKey("max_tokens") == true
                    ? Convert.ToInt32(context.ExtraParameters["max_tokens"]) : 4096
            };

            var responseJson = await SendRequestAsync("/chat/completions", requestBody, cancellationToken);
            using var doc = JsonDocument.Parse(responseJson);

            var content = "";
            if (doc.RootElement.TryGetProperty("choices", out var choices) &&
                choices.GetArrayLength() > 0)
            {
                var firstChoice = choices[0];
                if (firstChoice.TryGetProperty("message", out var msg) &&
                    msg.TryGetProperty("content", out var contentProp))
                {
                    content = contentProp.GetString() ?? "";
                }
            }

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
                var formattedMessages = FormatMessagesForApi(messages, context);
                var model = GetModelForContext(context);

                var requestBody = new
                {
                    model = model,
                    messages = formattedMessages,
                    temperature = context.ExtraParameters?.ContainsKey("temperature") == true
                        ? Convert.ToSingle(context.ExtraParameters["temperature"]) : 0.7f,
                    max_tokens = context.ExtraParameters?.ContainsKey("max_tokens") == true
                        ? Convert.ToInt32(context.ExtraParameters["max_tokens"]) : 4096,
                    stream = true
                };

                var url = BuildUrl("/chat/completions");
                var request = new HttpRequestMessage(HttpMethod.Post, url)
                {
                    Content = new StringContent(
                        JsonSerializer.Serialize(requestBody),
                        Encoding.UTF8,
                        "application/json")
                };
                request.Headers.Add("Authorization", $"Bearer {_apiKey}");

                var response = await _httpClient.SendAsync(
                    request,
                    HttpCompletionOption.ResponseHeadersRead,
                    _currentCts.Token);

                HandleRateLimitHeaders(response);
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
                    if (data == "[DONE]")
                        break;

                    string? openaiText = null;
                    FinishReason? openaiFinishReason = null;
                    try
                    {
                        using var lineDoc = JsonDocument.Parse(data);
                        var root = lineDoc.RootElement;

                        if (root.TryGetProperty("choices", out var choices) &&
                            choices.GetArrayLength() > 0)
                        {
                            var choice = choices[0];
                            if (choice.TryGetProperty("delta", out var delta) &&
                                delta.TryGetProperty("content", out var contentProp))
                            {
                                openaiText = contentProp.GetString() ?? "";
                                openaiFinishReason = choice.TryGetProperty("finish_reason", out var frProp)
                                    ? ParseFinishReason(frProp.GetString())
                                    : (FinishReason?)null;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        SynapseLogger.Default.Debug("OpenAiProvider", "Skipping malformed SSE line.", ex);
                    }

                    if (!string.IsNullOrEmpty(openaiText))
                    {
                        yield return new StreamingToken
                        {
                            Text = openaiText,
                            IsFinal = openaiFinishReason.HasValue,
                            FinishReason = openaiFinishReason
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

            var modelId = model switch
            {
                EmbeddingModel.TextEmbedding3Small => "text-embedding-3-small",
                EmbeddingModel.TextEmbedding3Large => "text-embedding-3-large",
                _ => "text-embedding-3-small"
            };

            var requestBody = new
            {
                model = modelId,
                input = text
            };

            var responseJson = await SendRequestAsync("/embeddings", requestBody, cancellationToken);
            using var doc = JsonDocument.Parse(responseJson);

            float[] embedding = Array.Empty<float>();
            int dimensions = 0;
            int promptTokens = 0;

            if (doc.RootElement.TryGetProperty("data", out var data) &&
                data.GetArrayLength() > 0)
            {
                var firstEmbedding = data[0];
                if (firstEmbedding.TryGetProperty("embedding", out var embeddingProp))
                {
                    var values = new List<float>();
                    foreach (var val in embeddingProp.EnumerateArray())
                        values.Add(val.GetSingle());
                    embedding = values.ToArray();
                    dimensions = embedding.Length;
                }
            }

            if (doc.RootElement.TryGetProperty("usage", out var usage) &&
                usage.TryGetProperty("prompt_tokens", out var ptProp))
            {
                promptTokens = ptProp.GetInt32();
            }

            Interlocked.Increment(ref _totalRequests);

            return new EmbeddingResult
            {
                Vector = new EmbeddingVector
                {
                    Dimensions = dimensions,
                    Values = embedding,
                    Text = text,
                    Model = modelId
                },
                Usage = new TokenUsage
                {
                    PromptTokens = promptTokens,
                    CompletionTokens = 0,
                    Model = modelId,
                    Provider = "OpenAI"
                },
                Model = modelId
            };
        }

        /// <inheritdoc/>
        public int EstimateTokens(string text)
        {
            if (string.IsNullOrEmpty(text))
                return 0;
            // cl100k_base approximation: ~0.75 tokens per word, ~4 chars per token
            int wordCount = text.Split(new[] { ' ', '\t', '\n', '\r' },
                StringSplitOptions.RemoveEmptyEntries).Length;
            int charEstimate = text.Length / 4;
            return Math.Max(1, (int)(wordCount * 0.75 + charEstimate * 0.25));
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
            if (_disposed)
                return false;
            if (string.IsNullOrEmpty(_apiKey))
                return false;

            try
            {
                var request = new HttpRequestMessage(HttpMethod.Get, BuildUrl("/models"));
                request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                var response = await _httpClient.SendAsync(request, cancellationToken);
                return response.IsSuccessStatusCode;
            }
            catch
            {
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
            _modelCache.Clear();
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public async Task<HealthCheckStatus> CheckHealthAsync(CancellationToken cancellationToken = default)
        {
            if (_disposed)
                return HealthCheckStatus.Unhealthy;

            try
            {
                var request = new HttpRequestMessage(HttpMethod.Get, BuildUrl("/models"));
                request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                var response = await _httpClient.SendAsync(request, cancellationToken);

                if (response.IsSuccessStatusCode)
                {
                    Health = Health with
                    {
                        Status = HealthCheckStatus.Healthy,
                        LastCheckedAt = DateTimeOffset.UtcNow
                    };
                    return HealthCheckStatus.Healthy;
                }

                return HealthCheckStatus.Degraded;
            }
            catch
            {
                return HealthCheckStatus.Unhealthy;
            }
        }

        /// <summary>
        /// Sends a function/tool calling request.
        /// </summary>
        /// <param name="messages">Chat messages.</param>
        /// <param name="functions">Available functions.</param>
        /// <param name="context">Request context.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The response with potential function calls.</returns>
        public async Task<LlmResponse> ChatWithFunctionsAsync(
            IReadOnlyList<ChatMessage> messages,
            IReadOnlyList<FunctionDefinition> functions,
            PromptContext context,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            var formattedMessages = FormatMessagesForApi(messages, context);
            var model = GetModelForContext(context);

            var tools = functions.Select(f => new
            {
                type = "function",
                function = new
                {
                    name = f.Name,
                    description = f.Description,
                    parameters = new
                    {
                        type = "object",
                        properties = f.Parameters.ToDictionary(
                            p => p.Name,
                            p => new
                            {
                                type = p.Type,
                                description = p.Description
                            }),
                        required = f.Parameters.Where(p => p.Required).Select(p => p.Name).ToArray()
                    }
                }
            }).ToArray();

            var requestBody = new
            {
                model = model,
                messages = formattedMessages,
                tools = tools,
                temperature = 0.7f
            };

            var responseJson = await SendRequestAsync("/chat/completions", requestBody, cancellationToken);
            using var doc = JsonDocument.Parse(responseJson);

            string content = "";
            var functionCalls = new List<FunctionCall>();

            if (doc.RootElement.TryGetProperty("choices", out var choices) &&
                choices.GetArrayLength() > 0)
            {
                var choice = choices[0];
                if (choice.TryGetProperty("message", out var msg))
                {
                    if (msg.TryGetProperty("content", out var contentProp))
                        content = contentProp.GetString() ?? "";

                    if (msg.TryGetProperty("tool_calls", out var toolCalls))
                    {
                        foreach (var tc in toolCalls.EnumerateArray())
                        {
                            if (tc.TryGetProperty("function", out var func))
                            {
                                functionCalls.Add(new FunctionCall
                                {
                                    Name = func.TryGetProperty("name", out var nameProp)
                                        ? nameProp.GetString() ?? "" : "",
                                    Arguments = func.TryGetProperty("arguments", out var argsProp)
                                        ? argsProp.GetString() ?? "{}" : "{}"
                                });
                            }
                        }
                    }
                }
            }

            TokenUsage? usage = null;
            if (doc.RootElement.TryGetProperty("usage", out var usageProp))
            {
                int promptTokens = 0, completionTokens = 0;
                if (usageProp.TryGetProperty("prompt_tokens", out var pt))
                    promptTokens = pt.GetInt32();
                if (usageProp.TryGetProperty("completion_tokens", out var ct))
                    completionTokens = ct.GetInt32();

                usage = new TokenUsage
                {
                    PromptTokens = promptTokens,
                    CompletionTokens = completionTokens,
                    Model = model,
                    Provider = "OpenAI"
                };
            }

            Interlocked.Increment(ref _totalRequests);

            return new LlmResponse
            {
                Content = content,
                Model = model,
                Provider = "OpenAI",
                Usage = usage ?? new TokenUsage { Model = model, Provider = "OpenAI" },
                FinishReason = functionCalls.Count > 0 ? FinishReason.FunctionCall : FinishReason.StopToken,
                FunctionCalls = functionCalls.AsReadOnly()
            };
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _currentCts?.Cancel();
            _currentCts?.Dispose();
            _modelCache.Clear();
            if (_ownsHttpClient)
                _httpClient.Dispose();
        }

        // ─── Private Helpers ───────────────────────────────────────────────

        private void InitializeModelCache()
        {
            var models = new (string Id, string Name, int Context, int MaxOut, decimal InputCost, decimal OutputCost, bool Vision)[]
            {
                ("gpt-4o", "GPT-4o", 128000, 16384, 2.50m, 10.00m, true),
                ("gpt-4o-mini", "GPT-4o Mini", 128000, 16384, 0.15m, 0.60m, true),
                ("gpt-4-turbo", "GPT-4 Turbo", 128000, 4096, 10.00m, 30.00m, true),
                ("gpt-4", "GPT-4", 8192, 4096, 30.00m, 60.00m, false),
                ("gpt-3.5-turbo", "GPT-3.5 Turbo", 16385, 4096, 0.50m, 1.50m, false),
                ("text-embedding-3-small", "Text Embedding 3 Small", 8191, 0, 0.02m, 0m, false),
                ("text-embedding-3-large", "Text Embedding 3 Large", 8191, 0, 0.13m, 0m, false)
            };

            foreach (var m in models)
            {
                _modelCache[m.Id] = new ModelInfo
                {
                    ModelId = m.Id,
                    DisplayName = m.Name,
                    Provider = "OpenAI",
                    ContextWindow = m.Context,
                    MaxOutputTokens = m.MaxOut,
                    Tokenizer = TokenizerType.Tiktoken,
                    VocabularySize = 100000,
                    SupportsVision = m.Vision,
                    SupportsFunctionCalling = m.Id.StartsWith("gpt-4"),
                    SupportsEmbeddings = m.Id.Contains("embedding"),
                    InputCostPer1MTokens = m.InputCost,
                    OutputCostPer1MTokens = m.OutputCost
                };
            }
        }

        private string GetModelForContext(PromptContext context)
        {
            if (context.ExtraParameters?.ContainsKey("model") == true)
                return context.ExtraParameters["model"]?.ToString() ?? _defaultModel;

            return context.TaskType switch
            {
                LlmTaskType.CodeGeneration => "gpt-4o",
                LlmTaskType.GenomeCreation => "gpt-4o",
                LlmTaskType.BehaviorGeneration => "gpt-4o-mini",
                LlmTaskType.MaterialSuggestion => "gpt-4o-mini",
                LlmTaskType.EmbeddingGeneration => "text-embedding-3-small",
                LlmTaskType.NarrativeGeneration => "gpt-4o",
                _ => _defaultModel
            };
        }

        private string BuildUrl(string path)
        {
            if (_isAzureEndpoint)
            {
                return $"{_baseUrl}/openai/deployments/{_azureDeploymentName}{path}?api-version={_azureApiVersion}";
            }
            return $"{_baseUrl}{path}";
        }

        private object[] FormatMessagesForApi(IReadOnlyList<ChatMessage> messages, PromptContext context)
        {
            return messages.Select(m => (object)new
            {
                role = m.Role switch
                {
                    MessageRole.System => "system",
                    MessageRole.User => "user",
                    MessageRole.Assistant => "assistant",
                    MessageRole.Function => "function",
                    MessageRole.Tool => "tool",
                    _ => "user"
                },
                content = m.Content,
                name = m.Name
            }).ToArray();
        }

        private async Task<string> SendRequestAsync(
            string path,
            object requestBody,
            CancellationToken cancellationToken)
        {
            const int maxResponseBytes = 8 * 1024 * 1024;
            var url = BuildUrl(path);
            var json = JsonSerializer.Serialize(requestBody);

            async Task<HttpResponseMessage> SendOnceAsync()
            {
                var request = new HttpRequestMessage(HttpMethod.Post, url)
                {
                    Content = new StringContent(json, Encoding.UTF8, "application/json")
                };
                request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                return await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
            }

            using var response = await SendOnceAsync().ConfigureAwait(false);
            HandleRateLimitHeaders(response);

            if (response.StatusCode == (HttpStatusCode)429)
            {
                var retryAfter = response.Headers.RetryAfter?.Delta ?? TimeSpan.FromSeconds(5);
                await Task.Delay(retryAfter, cancellationToken).ConfigureAwait(false);
                response.Dispose();
                using var retry = await SendOnceAsync().ConfigureAwait(false);
                HandleRateLimitHeaders(retry);
                retry.EnsureSuccessStatusCode();
                return await ReadCappedAsync(retry, maxResponseBytes, cancellationToken).ConfigureAwait(false);
            }

            response.EnsureSuccessStatusCode();
            return await ReadCappedAsync(response, maxResponseBytes, cancellationToken).ConfigureAwait(false);
        }

        private static async Task<string> ReadCappedAsync(
            HttpResponseMessage response,
            int maxBytes,
            CancellationToken cancellationToken)
        {
            if (response.Content.Headers.ContentLength is long declared && declared > maxBytes)
                throw new InvalidDataException($"LLM response exceeds {maxBytes} byte limit.");

            await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            using var ms = new MemoryStream();
            var buffer = new byte[81920];
            long total = 0;
            int read;
            while ((read = await stream.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken).ConfigureAwait(false)) > 0)
            {
                total += read;
                if (total > maxBytes)
                    throw new InvalidDataException($"LLM response exceeds {maxBytes} byte limit.");
                ms.Write(buffer, 0, read);
            }
            return Encoding.UTF8.GetString(ms.GetBuffer(), 0, (int)ms.Length);
        }

        private void HandleRateLimitHeaders(HttpResponseMessage response)
        {
            lock (_rateLimitLock)
            {
                if (response.Headers.TryGetValues("x-ratelimit-remaining-requests", out var reqValues))
                {
                    if (int.TryParse(reqValues.FirstOrDefault(), out var remaining))
                        _remainingRequests = remaining;
                }
                if (response.Headers.TryGetValues("x-ratelimit-reset-requests", out var resetValues))
                {
                    if (double.TryParse(resetValues.FirstOrDefault()?.Replace("s", ""),
                        NumberStyles.Float, CultureInfo.InvariantCulture, out var resetSeconds))
                    {
                        _rateLimitResetTime = DateTimeOffset.UtcNow.AddSeconds(resetSeconds);
                    }
                }
            }
        }

        private static FinishReason? ParseFinishReason(string? reason)
        {
            return reason switch
            {
                "stop" => FinishReason.StopToken,
                "length" => FinishReason.MaxTokens,
                "content_filter" => FinishReason.ContentFilter,
                "function_call" => FinishReason.FunctionCall,
                "tool_calls" => FinishReason.FunctionCall,
                _ => null
            };
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
