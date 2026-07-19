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
    /// LLM inference provider using a local Ollama server.
    /// Supports text generation, chat completions, embeddings, and model management
    /// via Ollama's REST API.
    /// </summary>
    public sealed class OllamaProvider : ILlmInferenceProvider
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;
        private readonly int _timeoutSeconds;
        private readonly bool _ownsHttpClient;
        private volatile bool _disposed;
        private volatile LlmStatus _status = LlmStatus.Unavailable;
        private readonly ConcurrentDictionary<string, ModelInfo> _modelCache;
        private readonly List<string> _availableModelNames;
        private string _defaultModel;
        private CancellationTokenSource? _currentCts;
        private long _totalTokensGenerated;
        private long _totalRequests;

        /// <inheritdoc/>
        public string ProviderName => "Ollama";

        /// <inheritdoc/>
        public bool RequiresNetwork => false;

        /// <inheritdoc/>
        public LlmStatus Status => _status;

        /// <inheritdoc/>
        public LlmProviderMode Mode => LlmProviderMode.LocalOllama;

        /// <inheritdoc/>
        public IReadOnlyList<ModelInfo> AvailableModels =>
            _modelCache.Values.ToList().AsReadOnly();

        /// <inheritdoc/>
        public ProviderHealth Health { get; private set; } = new() { ProviderName = "Ollama" };

        /// <summary>Total tokens generated since initialization.</summary>
        public long TotalTokensGenerated => Interlocked.Read(ref _totalTokensGenerated);

        /// <summary>
        /// Initializes a new Ollama provider.
        /// </summary>
        /// <param name="baseUrl">Base URL of the Ollama server (e.g., "http://localhost:11434").</param>
        /// <param name="defaultModel">Default model to use for requests.</param>
        /// <param name="timeoutSeconds">Request timeout in seconds.</param>
        /// <param name="httpClient">Optional shared HttpClient instance.</param>
        public OllamaProvider(
            string baseUrl = "http://localhost:11434",
            string defaultModel = "llama3:8b",
            int timeoutSeconds = 120,
            HttpClient? httpClient = null)
        {
            var validated = Synapse.Core.Security.UrlSecurity.ValidateOutboundUri(
                baseUrl?.TrimEnd('/') ?? "http://localhost:11434",
                allowLoopbackHttp: true);
            _baseUrl = validated.ToString().TrimEnd('/');
            _defaultModel = defaultModel ?? "llama3:8b";
            _timeoutSeconds = timeoutSeconds;
            _modelCache = new ConcurrentDictionary<string, ModelInfo>(StringComparer.OrdinalIgnoreCase);
            _availableModelNames = new List<string>();

            if (httpClient != null)
            {
                _httpClient = httpClient;
                _ownsHttpClient = false;
            }
            else
            {
                _httpClient = Synapse.Core.Security.UrlSecurity.CreateSafeHttpClient(
                    TimeSpan.FromSeconds(timeoutSeconds + 30));
                _ownsHttpClient = true;
            }
        }

        /// <summary>
        /// Discovers available models from the Ollama server.
        /// </summary>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>List of available model names.</returns>
        public async Task<IReadOnlyList<string>> DiscoverModelsAsync(
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            try
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/api/tags", cancellationToken);
                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync(cancellationToken);
                using var doc = JsonDocument.Parse(json);

                _availableModelNames.Clear();
                _modelCache.Clear();

                if (doc.RootElement.TryGetProperty("models", out var models))
                {
                    foreach (var model in models.EnumerateArray())
                    {
                        var name = model.TryGetProperty("name", out var nameProp)
                            ? nameProp.GetString() ?? ""
                            : "";

                        if (string.IsNullOrEmpty(name))
                            continue;

                        _availableModelNames.Add(name);

                        long size = 0;
                        if (model.TryGetProperty("size", out var sizeProp))
                            size = sizeProp.GetInt64();

                        string modifiedAt = "";
                        if (model.TryGetProperty("modified_at", out var modProp))
                            modifiedAt = modProp.GetString() ?? "";

                        _modelCache[name] = new ModelInfo
                        {
                            ModelId = name,
                            DisplayName = name,
                            Provider = "Ollama",
                            ContextWindow = 4096,
                            MaxOutputTokens = 2048,
                            Tokenizer = TokenizerType.BPE,
                            VocabularySize = 32000,
                            EstimatedVramMb = (int)(size / (1024 * 1024)),
                            SupportsVision = name.Contains("vision", StringComparison.OrdinalIgnoreCase),
                            SupportsFunctionCalling = false,
                            SupportsEmbeddings = name.Contains("embed", StringComparison.OrdinalIgnoreCase),
                            InputCostPer1MTokens = 0m,
                            OutputCostPer1MTokens = 0m
                        };
                    }
                }

                _status = LlmStatus.Ready;
                return _availableModelNames.AsReadOnly();
            }
            catch (HttpRequestException ex)
            {
                _status = LlmStatus.Error;
                Health = Health with
                {
                    Status = HealthCheckStatus.Unhealthy,
                    LastError = ex.Message,
                    LastFailureAt = DateTimeOffset.UtcNow
                };
                return Array.Empty<string>();
            }
        }

        /// <summary>
        /// Downloads/pulls a model from the Ollama registry.
        /// </summary>
        /// <param name="modelName">Model name to pull (e.g., "llama3:8b").</param>
        /// <param name="progressCallback">Optional callback for pull progress.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        public async Task PullModelAsync(
            string modelName,
            Action<string>? progressCallback = null,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(modelName);

            var request = new { name = modelName };
            var content = new StringContent(
                JsonSerializer.Serialize(request),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/api/pull", content, cancellationToken);
            response.EnsureSuccessStatusCode();

            var responseStream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(responseStream);

            while (!reader.EndOfStream)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var line = await reader.ReadLineAsync(cancellationToken);
                if (string.IsNullOrEmpty(line))
                    continue;

                try
                {
                    using var lineDoc = JsonDocument.Parse(line);
                    if (lineDoc.RootElement.TryGetProperty("status", out var status))
                    {
                        progressCallback?.Invoke(status.GetString() ?? "");
                    }
                }
                catch { /* Skip malformed lines */ }
            }
        }

        /// <summary>
        /// Deletes a model from the local Ollama instance.
        /// </summary>
        /// <param name="modelName">Model name to delete.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        public async Task DeleteModelAsync(string modelName, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(modelName);

            var request = new { name = modelName };
            var content = new StringContent(
                JsonSerializer.Serialize(request),
                Encoding.UTF8,
                "application/json");

            var requestMessage = new HttpRequestMessage(HttpMethod.Delete, $"{_baseUrl}/api/delete")
            {
                Content = content
            };

            var response = await _httpClient.SendAsync(requestMessage, cancellationToken);
            response.EnsureSuccessStatusCode();

            _modelCache.TryRemove(modelName, out _);
            lock (_availableModelNames)
            {
                _availableModelNames.Remove(modelName);
            }
        }

        /// <summary>
        /// Gets detailed information about a specific model.
        /// </summary>
        /// <param name="modelName">Model name.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>Model information dictionary.</returns>
        public async Task<IReadOnlyDictionary<string, string>> ShowModelInfoAsync(
            string modelName,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(modelName);

            var request = new { name = modelName };
            var content = new StringContent(
                JsonSerializer.Serialize(request),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/api/show", content, cancellationToken);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync(cancellationToken);
            using var doc = JsonDocument.Parse(json);

            var info = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                info[prop.Name] = prop.Value.ToString();
            }

            return info;
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
                ? $"\n\nRespond ONLY with valid JSON matching this schema:\n{jsonSchema}\n"
                : "\n\nRespond ONLY with valid JSON.\n";

            var fullPrompt = prompt + jsonInstruction;

            var chatMessages = new List<ChatMessage>();
            if (!string.IsNullOrEmpty(systemPrompt))
                chatMessages.Add(ChatMessage.System(systemPrompt));
            chatMessages.Add(ChatMessage.User(fullPrompt));

            var response = await ChatCompletionAsync(chatMessages, context, cancellationToken);
            return StructuredOutputParser.ParseJson<T>(response);
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
                var request = BuildChatRequest(messages, model, stream: true);

                var httpContent = new StringContent(
                    JsonSerializer.Serialize(request),
                    Encoding.UTF8,
                    "application/json");

                var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/api/chat")
                {
                    Content = httpContent
                };

                var response = await _httpClient.SendAsync(
                    httpRequest,
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

                    string? ollamaText = null;
                    bool ollamaDone = false;
                    try
                    {
                        using var lineDoc = JsonDocument.Parse(line);
                        var root = lineDoc.RootElement;

                        if (root.TryGetProperty("message", out var message) &&
                            message.TryGetProperty("content", out var content))
                        {
                            ollamaText = content.GetString() ?? "";
                            ollamaDone = root.TryGetProperty("done", out var doneProp) &&
                                       doneProp.GetBoolean();
                        }
                    }
                    catch { /* Skip malformed lines */ }

                    if (!string.IsNullOrEmpty(ollamaText))
                    {
                        yield return new StreamingToken
                        {
                            Text = ollamaText,
                            IsFinal = ollamaDone,
                            FinishReason = ollamaDone ? FinishReason.StopToken : default
                        };
                    }

                    if (ollamaDone)
                        break;
                }
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
            EmbeddingModel model = EmbeddingModel.NomicEmbed,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            var embeddingModel = model switch
            {
                EmbeddingModel.NomicEmbed => "nomic-embed-text",
                EmbeddingModel.AllMiniLM => "all-minilm",
                EmbeddingModel.TextEmbedding3Small => "nomic-embed-text",
                EmbeddingModel.TextEmbedding3Large => "nomic-embed-text",
                _ => "nomic-embed-text"
            };

            var request = new
            {
                model = embeddingModel,
                prompt = text
            };

            var content = new StringContent(
                JsonSerializer.Serialize(request),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/api/embeddings", content, cancellationToken);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            using var doc = JsonDocument.Parse(responseJson);

            float[] embedding = Array.Empty<float>();
            if (doc.RootElement.TryGetProperty("embedding", out var embeddingProp))
            {
                var values = new List<float>();
                foreach (var val in embeddingProp.EnumerateArray())
                    values.Add(val.GetSingle());
                embedding = values.ToArray();
            }

            return new EmbeddingResult
            {
                Vector = new EmbeddingVector
                {
                    Dimensions = embedding.Length,
                    Values = embedding,
                    Text = text,
                    Model = embeddingModel
                },
                Model = embeddingModel
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
            if (_disposed)
                return false;
            try
            {
                var response = await _httpClient.GetAsync(
                    $"{_baseUrl}/api/tags", cancellationToken);
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
        public async Task WarmUpAsync(CancellationToken cancellationToken = default)
        {
            _status = LlmStatus.WarmingUp;
            try
            {
                await DiscoverModelsAsync(cancellationToken);
                if (_availableModelNames.Count > 0)
                    _status = LlmStatus.Ready;
                else
                    _status = LlmStatus.Error;
            }
            catch
            {
                _status = LlmStatus.Error;
            }
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
                var response = await _httpClient.GetAsync(
                    $"{_baseUrl}/api/tags", cancellationToken);

                if (response.IsSuccessStatusCode)
                {
                    Health = Health with
                    {
                        Status = HealthCheckStatus.Healthy,
                        LastCheckedAt = DateTimeOffset.UtcNow
                    };
                    return HealthCheckStatus.Healthy;
                }

                Health = Health with
                {
                    Status = HealthCheckStatus.Unhealthy,
                    LastCheckedAt = DateTimeOffset.UtcNow,
                    LastError = $"HTTP {(int)response.StatusCode}"
                };
                return HealthCheckStatus.Unhealthy;
            }
            catch (Exception ex)
            {
                Health = Health with
                {
                    Status = HealthCheckStatus.Unhealthy,
                    LastCheckedAt = DateTimeOffset.UtcNow,
                    LastError = ex.Message
                };
                return HealthCheckStatus.Unhealthy;
            }
        }

        /// <summary>
        /// Sends a chat completion request without streaming.
        /// </summary>
        /// <param name="messages">Chat messages.</param>
        /// <param name="context">Request context.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The generated response text.</returns>
        public async Task<string> ChatCompletionAsync(
            IReadOnlyList<ChatMessage> messages,
            PromptContext? context = null,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            var model = context != null ? GetModelForContext(context) : _defaultModel;
            var request = BuildChatRequest(messages, model, stream: false);

            var content = new StringContent(
                JsonSerializer.Serialize(request),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/api/chat", content, cancellationToken);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            using var doc = JsonDocument.Parse(responseJson);

            if (doc.RootElement.TryGetProperty("message", out var message) &&
                message.TryGetProperty("content", out var contentProp))
            {
                Interlocked.Increment(ref _totalRequests);
                return contentProp.GetString() ?? "";
            }

            return "";
        }

        /// <summary>
        /// Generates text using the /api/generate endpoint.
        /// </summary>
        /// <param name="prompt">The prompt text.</param>
        /// <param name="model">Model name (uses default if null).</param>
        /// <param name="stream">Whether to stream the response.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>Generated text.</returns>
        public async Task<string> GenerateAsync(
            string prompt,
            string? model = null,
            bool stream = false,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            var request = new
            {
                model = model ?? _defaultModel,
                prompt = prompt,
                stream = stream,
                options = new
                {
                    temperature = 0.7,
                    top_p = 0.9,
                    top_k = 40
                }
            };

            var content = new StringContent(
                JsonSerializer.Serialize(request),
                Encoding.UTF8,
                "application/json");

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/api/generate", content, cancellationToken);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            using var doc = JsonDocument.Parse(responseJson);

            if (doc.RootElement.TryGetProperty("response", out var responseProp))
            {
                Interlocked.Increment(ref _totalRequests);
                return responseProp.GetString() ?? "";
            }

            return "";
        }

        /// <summary>
        /// Sets the default model for subsequent requests.
        /// </summary>
        /// <param name="modelName">Model name to use as default.</param>
        public void SetDefaultModel(string modelName)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(modelName);
            _defaultModel = modelName;
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

        private string GetModelForContext(PromptContext context)
        {
            if (context.ExtraParameters?.ContainsKey("model") == true)
                return context.ExtraParameters["model"]?.ToString() ?? _defaultModel;

            if (context.TaskType == LlmTaskType.EmbeddingGeneration)
                return "nomic-embed-text";

            if (context.TaskType == LlmTaskType.CodeGeneration && _modelCache.ContainsKey("codellama:13b"))
                return "codellama:13b";

            return _defaultModel;
        }

        private object BuildChatRequest(
            IReadOnlyList<ChatMessage> messages,
            string model,
            bool stream)
        {
            var formattedMessages = messages.Select(m => new
            {
                role = m.Role switch
                {
                    MessageRole.System => "system",
                    MessageRole.User => "user",
                    MessageRole.Assistant => "assistant",
                    _ => "user"
                },
                content = m.Content
            }).ToList();

            return new
            {
                model = model,
                messages = formattedMessages,
                stream = stream,
                options = new
                {
                    temperature = 0.7,
                    top_p = 0.9,
                    top_k = 40,
                    num_predict = 512
                }
            };
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
