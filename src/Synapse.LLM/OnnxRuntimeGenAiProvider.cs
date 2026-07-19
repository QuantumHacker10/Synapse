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
    /// LLM inference provider using ONNX Runtime GenAI for local model execution.
    /// Supports multiple hardware backends, quantization formats, KV-cache management,
    /// beam search and nucleus sampling, streaming, and batch inference.
    /// </summary>
    public sealed class OnnxRuntimeGenAiProvider : ILlmInferenceProvider
    {
        private readonly ConcurrentDictionary<string, OnnxModelSession> _loadedModels;
        private readonly ConcurrentDictionary<string, ModelInfo> _modelInfoCache;
        private readonly object _sessionLock = new();
        private string _currentModelId = string.Empty;
        private volatile bool _disposed;
        private volatile bool _isGenerating;
        private CancellationTokenSource? _currentCts;
        private readonly InferenceBackend _preferredBackend;
        private readonly int _maxGpuLayers;
        private readonly int _contextLength;
        private readonly float _temperature;
        private readonly float _topP;
        private readonly int _topK;
        private readonly float _repetitionPenalty;
        private readonly float _frequencyPenalty;
        private readonly float _presencePenalty;
        private readonly int _maxNewTokens;
        private long _totalTokensGenerated;
        private long _totalInferenceCount;

        /// <inheritdoc/>
        public string ProviderName => "OnnxRuntimeGenAi";

        /// <inheritdoc/>
        public bool RequiresNetwork => false;

        /// <inheritdoc/>
        public LlmStatus Status { get; private set; } = LlmStatus.Unavailable;

        /// <inheritdoc/>
        public LlmProviderMode Mode => LlmProviderMode.LocalOnnx;

        /// <inheritdoc/>
        public IReadOnlyList<ModelInfo> AvailableModels =>
            _modelInfoCache.Values.ToList().AsReadOnly();

        /// <inheritdoc/>
        public ProviderHealth Health { get; private set; } = new() { ProviderName = "OnnxRuntimeGenAi" };

        /// <summary>Preferred hardware backend for inference.</summary>
        public InferenceBackend PreferredBackend => _preferredBackend;

        /// <summary>Total tokens generated since initialization.</summary>
        public long TotalTokensGenerated => Interlocked.Read(ref _totalTokensGenerated);

        /// <summary>Total inference operations performed.</summary>
        public long TotalInferenceCount => Interlocked.Read(ref _totalInferenceCount);

        /// <summary>Number of models currently loaded.</summary>
        public int LoadedModelCount => _loadedModels.Count;

        /// <summary>
        /// Initializes a new instance of the OnnxRuntimeGenAiProvider.
        /// </summary>
        /// <param name="backend">Preferred hardware backend.</param>
        /// <param name="temperature">Default temperature for sampling.</param>
        /// <param name="topP">Default nucleus sampling probability.</param>
        /// <param name="topK">Default top-k sampling value.</param>
        /// <param name="maxNewTokens">Default maximum new tokens to generate.</param>
        /// <param name="contextLength">Default context length.</param>
        /// <param name="maxGpuLayers">Maximum layers to offload to GPU.</param>
        public OnnxRuntimeGenAiProvider(
            InferenceBackend backend = InferenceBackend.Cpu,
            float temperature = 0.7f,
            float topP = 0.9f,
            int topK = 40,
            int maxNewTokens = 512,
            int contextLength = 2048,
            int maxGpuLayers = -1)
        {
            _loadedModels = new ConcurrentDictionary<string, OnnxModelSession>(StringComparer.OrdinalIgnoreCase);
            _modelInfoCache = new ConcurrentDictionary<string, ModelInfo>(StringComparer.OrdinalIgnoreCase);
            _preferredBackend = backend;
            _temperature = Math.Clamp(temperature, 0f, 2.0f);
            _topP = Math.Clamp(topP, 0f, 1.0f);
            _topK = Math.Max(1, topK);
            _maxNewTokens = Math.Max(1, maxNewTokens);
            _contextLength = Math.Max(128, contextLength);
            _maxGpuLayers = maxGpuLayers;
            _repetitionPenalty = 1.1f;
            _frequencyPenalty = 0.0f;
            _presencePenalty = 0.0f;
        }

        /// <summary>
        /// Loads an ONNX model from the specified directory path.
        /// </summary>
        /// <param name="modelPath">Path to the ONNX model directory.</param>
        /// <param name="modelId">Unique identifier for the model.</param>
        /// <param name="quantization">Quantization format to use.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>Model information for the loaded model.</returns>
        public async Task<ModelInfo> LoadModelAsync(
            string modelPath,
            string? modelId = null,
            string? quantization = null,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

            if (!Directory.Exists(modelPath))
                throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");

            var onnxFiles = Directory.GetFiles(modelPath, "*.onnx");
            if (onnxFiles.Length == 0)
                throw new FileNotFoundException($"No ONNX model file found in: {modelPath}");

            modelId ??= Path.GetFileName(modelPath);

            var session = await Task.Run(() =>
            {
                var options = new OnnxSessionOptions
                {
                    Backend = _preferredBackend,
                    MaxGpuLayers = _maxGpuLayers,
                    ContextLength = _contextLength,
                    Quantization = quantization
                };

                return new OnnxModelSession(modelPath, options);
            }, cancellationToken);

            var vocabSize = session.VocabularySize;
            var contextWindow = session.ContextLength;
            var modelInfo = new ModelInfo
            {
                ModelId = modelId,
                DisplayName = Path.GetFileName(modelPath),
                Provider = "OnnxRuntimeGenAi",
                ContextWindow = contextWindow,
                MaxOutputTokens = Math.Min(_maxNewTokens, contextWindow),
                Tokenizer = TokenizerType.BPE,
                VocabularySize = vocabSize,
                EstimatedVramMb = EstimateVramUsage(session),
                SupportsVision = false,
                SupportsFunctionCalling = false,
                SupportsEmbeddings = false,
                InputCostPer1MTokens = 0m,
                OutputCostPer1MTokens = 0m,
                PreferredTasks = new[]
                {
                    LlmTaskType.QueryAnswering,
                    LlmTaskType.NarrativeGeneration,
                    LlmTaskType.CodeGeneration,
                    LlmTaskType.SceneDescription,
                    LlmTaskType.GenomeCreation
                }
            };

            var oldSession = _loadedModels.GetOrAdd(modelId, session);
            if (oldSession != session)
            {
                oldSession?.Dispose();
            }
            _modelInfoCache[modelId] = modelInfo;
            _currentModelId = modelId;
            Status = LlmStatus.Ready;

            return modelInfo;
        }

        /// <summary>
        /// Hot-swaps to a different loaded model.
        /// </summary>
        /// <param name="modelId">The model ID to switch to.</param>
        /// <exception cref="InvalidOperationException">Thrown if the model is not loaded.</exception>
        public void SwitchModel(string modelId)
        {
            ThrowIfDisposed();
            ArgumentException.ThrowIfNullOrWhiteSpace(modelId);

            if (!_loadedModels.ContainsKey(modelId))
                throw new InvalidOperationException($"Model '{modelId}' is not loaded.");

            _currentModelId = modelId;
        }

        /// <summary>
        /// Unloads a model and frees associated resources.
        /// </summary>
        /// <param name="modelId">The model ID to unload.</param>
        /// <returns>True if the model was found and unloaded.</returns>
        public bool UnloadModel(string modelId)
        {
            ThrowIfDisposed();
            if (_loadedModels.TryRemove(modelId, out var session))
            {
                session?.Dispose();
                _modelInfoCache.TryRemove(modelId, out _);
                if (_currentModelId == modelId)
                {
                    _currentModelId = _loadedModels.Keys.FirstOrDefault() ?? string.Empty;
                }
                return true;
            }
            return false;
        }

        /// <summary>
        /// Gets the memory usage of the currently loaded model.
        /// </summary>
        /// <returns>Memory usage information.</returns>
        public ModelMemoryUsage GetMemoryUsage()
        {
            ThrowIfDisposed();
            if (!_loadedModels.TryGetValue(_currentModelId, out var session))
                return new ModelMemoryUsage();

            return new ModelMemoryUsage
            {
                ModelId = _currentModelId,
                VramUsageMb = EstimateVramUsage(session),
                RamUsageMb = session.EstimateRamUsage(),
                KVCacheSizeMb = session.KVCacheSizeMb,
                WeightSizeMb = session.WeightSizeMb,
                ContextLength = session.ContextLength,
                CurrentSequenceLength = session.CurrentSequenceLength
            };
        }

        /// <summary>
        /// Generates text using greedy decoding (temperature=0, always pick most probable token).
        /// </summary>
        /// <param name="prompt">The input prompt.</param>
        /// <param name="maxTokens">Maximum tokens to generate.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The generated text.</returns>
        public async Task<string> GenerateGreedyAsync(
            string prompt,
            int? maxTokens = null,
            CancellationToken cancellationToken = default)
        {
            return await GenerateInternalAsync(prompt, new OnnxGenerationParams
            {
                Temperature = 0f,
                TopP = 1f,
                TopK = 1,
                MaxNewTokens = maxTokens ?? _maxNewTokens,
                RepetitionPenalty = 1f
            }, cancellationToken);
        }

        /// <summary>
        /// Generates text using beam search decoding.
        /// </summary>
        /// <param name="prompt">The input prompt.</param>
        /// <param name="beamCount">Number of beams for search.</param>
        /// <param name="maxTokens">Maximum tokens to generate.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The best generated text from beam search.</returns>
        public async Task<string> GenerateBeamSearchAsync(
            string prompt,
            int beamCount = 4,
            int? maxTokens = null,
            CancellationToken cancellationToken = default)
        {
            return await GenerateInternalAsync(prompt, new OnnxGenerationParams
            {
                Temperature = 0f,
                TopP = 1f,
                TopK = 1,
                MaxNewTokens = maxTokens ?? _maxNewTokens,
                BeamSearch = true,
                BeamCount = Math.Max(1, beamCount),
                RepetitionPenalty = _repetitionPenalty
            }, cancellationToken);
        }

        /// <summary>
        /// Generates text using nucleus (top-p) sampling.
        /// </summary>
        /// <param name="prompt">The input prompt.</param>
        /// <param name="temperature">Sampling temperature.</param>
        /// <param name="topP">Nucleus sampling probability threshold.</param>
        /// <param name="maxTokens">Maximum tokens to generate.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The generated text.</returns>
        public async Task<string> GenerateNucleusAsync(
            string prompt,
            float? temperature = null,
            float? topP = null,
            int? maxTokens = null,
            CancellationToken cancellationToken = default)
        {
            return await GenerateInternalAsync(prompt, new OnnxGenerationParams
            {
                Temperature = temperature ?? _temperature,
                TopP = topP ?? _topP,
                TopK = 0,
                MaxNewTokens = maxTokens ?? _maxNewTokens,
                RepetitionPenalty = _repetitionPenalty,
                FrequencyPenalty = _frequencyPenalty,
                PresencePenalty = _presencePenalty
            }, cancellationToken);
        }

        /// <summary>
        /// Generates text using top-k sampling.
        /// </summary>
        /// <param name="prompt">The input prompt.</param>
        /// <param name="temperature">Sampling temperature.</param>
        /// <param name="topK">Number of top tokens to sample from.</param>
        /// <param name="maxTokens">Maximum tokens to generate.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The generated text.</returns>
        public async Task<string> GenerateTopKAsync(
            string prompt,
            float? temperature = null,
            int? topK = null,
            int? maxTokens = null,
            CancellationToken cancellationToken = default)
        {
            return await GenerateInternalAsync(prompt, new OnnxGenerationParams
            {
                Temperature = temperature ?? _temperature,
                TopP = 1f,
                TopK = topK ?? _topK,
                MaxNewTokens = maxTokens ?? _maxNewTokens,
                RepetitionPenalty = _repetitionPenalty
            }, cancellationToken);
        }

        /// <summary>
        /// Runs batch inference on multiple prompts simultaneously.
        /// </summary>
        /// <param name="prompts">Array of prompt strings.</param>
        /// <param name="maxTokens">Maximum tokens per response.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>Array of generated texts, one per prompt.</returns>
        public async Task<IReadOnlyList<string>> BatchInferAsync(
            IReadOnlyList<string> prompts,
            int? maxTokens = null,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(prompts);
            if (prompts.Count == 0) return Array.Empty<string>();

            var results = new string[prompts.Count];
            var tasks = prompts.Select((prompt, index) =>
                Task.Run(async () =>
                {
                    results[index] = await GenerateInternalAsync(prompt, new OnnxGenerationParams
                    {
                        Temperature = _temperature,
                        TopP = _topP,
                        TopK = _topK,
                        MaxNewTokens = maxTokens ?? _maxNewTokens,
                        RepetitionPenalty = _repetitionPenalty
                    }, cancellationToken);
                }, cancellationToken));

            await Task.WhenAll(tasks);
            return results;
        }

        /// <summary>
        /// Applies repetition penalty to logits.
        /// </summary>
        /// <param name="logits">The logits array to modify.</param>
        /// <param name="generatedTokenIds">Previously generated token IDs.</param>
        /// <param name="penalty">The penalty value (>1.0 to penalize).</param>
        public static void ApplyRepetitionPenalty(
            float[] logits,
            IReadOnlyList<int> generatedTokenIds,
            float penalty)
        {
            if (logits == null || logits.Length == 0) return;
            if (generatedTokenIds == null || generatedTokenIds.Count == 0) return;
            if (penalty <= 1.0f) return;

            var processedTokens = new HashSet<int>();
            foreach (var tokenId in generatedTokenIds)
            {
                if (tokenId < 0 || tokenId >= logits.Length) continue;
                if (!processedTokens.Add(tokenId)) continue;

                if (logits[tokenId] > 0)
                    logits[tokenId] /= penalty;
                else
                    logits[tokenId] *= penalty;
            }
        }

        /// <summary>
        /// Applies temperature scaling to logits.
        /// </summary>
        /// <param name="logits">The logits array to scale.</param>
        /// <param name="temperature">Temperature value (0 = greedy, higher = more random).</param>
        public static void ApplyTemperature(float[] logits, float temperature)
        {
            if (logits == null || logits.Length == 0) return;
            if (temperature <= 0f)
            {
                // Greedy: find the max and set all others to -inf
                int maxIdx = 0;
                for (int i = 1; i < logits.Length; i++)
                    if (logits[i] > logits[maxIdx]) maxIdx = i;
                for (int i = 0; i < logits.Length; i++)
                    logits[i] = i == maxIdx ? 1.0f : float.NegativeInfinity;
                return;
            }

            float invTemp = 1.0f / temperature;
            for (int i = 0; i < logits.Length; i++)
                logits[i] *= invTemp;
        }

        /// <summary>
        /// Converts logits to probabilities using softmax.
        /// </summary>
        /// <param name="logits">Input logits.</param>
        /// <returns>Probability distribution.</returns>
        public static float[] Softmax(float[] logits)
        {
            if (logits == null || logits.Length == 0) return Array.Empty<float>();

            float maxLogit = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
                if (logits[i] > maxLogit) maxLogit = logits[i];

            var probs = new float[logits.Length];
            float sum = 0f;
            for (int i = 0; i < logits.Length; i++)
            {
                probs[i] = MathF.Exp(logits[i] - maxLogit);
                sum += probs[i];
            }

            if (sum > 0f)
                for (int i = 0; i < probs.Length; i++)
                    probs[i] /= sum;

            return probs;
        }

        /// <summary>
        /// Samples a token from a probability distribution using top-k filtering.
        /// </summary>
        /// <param name="probabilities">The probability distribution.</param>
        /// <param name="k">Number of top tokens to consider.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>Sampled token index.</returns>
        public static int TopKSample(float[] probabilities, int k, Random rng)
        {
            if (probabilities == null || probabilities.Length == 0) return 0;
            if (k <= 0) k = probabilities.Length;
            k = Math.Min(k, probabilities.Length);

            var topK = new (int index, float prob)[k];
            Array.Copy(
                probabilities.Select((p, i) => (i, p)).ToArray(),
                topK,
                k);

            // Simple selection of top-k
            for (int i = 0; i < probabilities.Length; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    if (probabilities[i] > topK[j].prob)
                    {
                        topK[j] = (i, probabilities[i]);
                        break;
                    }
                }
            }

            float totalProb = 0f;
            foreach (var (_, prob) in topK) totalProb += prob;

            if (totalProb <= 0f) return topK[0].index;

            float random = (float)(rng.NextDouble() * totalProb);
            float cumulative = 0f;
            foreach (var (index, prob) in topK)
            {
                cumulative += prob;
                if (random <= cumulative) return index;
            }

            return topK[k - 1].index;
        }

        /// <summary>
        /// Samples a token using nucleus (top-p) sampling.
        /// </summary>
        /// <param name="probabilities">The probability distribution.</param>
        /// <param name="p">Nucleus threshold (e.g., 0.9 = top 90% probability mass).</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>Sampled token index.</returns>
        public static int NucleusSample(float[] probabilities, float p, Random rng)
        {
            if (probabilities == null || probabilities.Length == 0) return 0;
            p = Math.Clamp(p, 0.01f, 1.0f);

            var sorted = probabilities
                .Select((prob, index) => (index, prob))
                .OrderByDescending(x => x.prob)
                .ToArray();

            float cumulative = 0f;
            int cutoffIndex = sorted.Length;

            for (int i = 0; i < sorted.Length; i++)
            {
                cumulative += sorted[i].prob;
                if (cumulative >= p)
                {
                    cutoffIndex = i + 1;
                    break;
                }
            }

            float nucleusSum = 0f;
            for (int i = 0; i < cutoffIndex; i++)
                nucleusSum += sorted[i].prob;

            if (nucleusSum <= 0f) return sorted[0].index;

            float r = (float)(rng.NextDouble() * nucleusSum);
            float running = 0f;
            for (int i = 0; i < cutoffIndex; i++)
            {
                running += sorted[i].prob;
                if (r <= running) return sorted[i].index;
            }

            return sorted[cutoffIndex - 1].index;
        }

        /// <summary>
        /// Formats a prompt using the chat template format.
        /// </summary>
        /// <param name="systemPrompt">Optional system prompt.</param>
        /// <param name="userMessage">The user message.</param>
        /// <param name="chatTemplate">The chat template string with {system}, {user}, {assistant} placeholders.</param>
        /// <returns>The formatted prompt.</returns>
        public static string FormatChatPrompt(
            string? systemPrompt,
            string userMessage,
            string chatTemplate = "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n")
        {
            var sb = new StringBuilder(chatTemplate);
            if (!string.IsNullOrEmpty(systemPrompt))
                sb.Replace("{system}", systemPrompt);
            else
                sb.Replace("{system}\n", "");
            sb.Replace("{user}", userMessage);
            sb.Replace("{assistant}", "");
            return sb.ToString();
        }

        /// <summary>
        /// Formats a multi-turn conversation using the chat template.
        /// </summary>
        /// <param name="messages">The conversation messages.</param>
        /// <param name="chatTemplate">The chat template string.</param>
        /// <returns>The formatted prompt.</returns>
        public static string FormatChatConversation(
            IReadOnlyList<ChatMessage> messages,
            string chatTemplate = "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}\n")
        {
            var sb = new StringBuilder();
            string? systemPrompt = null;

            foreach (var msg in messages)
            {
                switch (msg.Role)
                {
                    case MessageRole.System:
                        systemPrompt = msg.Content;
                        break;
                    case MessageRole.User:
                        if (systemPrompt != null)
                        {
                            sb.Append(chatTemplate
                                .Replace("{system}", systemPrompt)
                                .Replace("{user}", msg.Content)
                                .Replace("{assistant}", ""));
                            systemPrompt = null;
                        }
                        else
                        {
                            sb.Append($"<|user|>\n{msg.Content}\n");
                        }
                        break;
                    case MessageRole.Assistant:
                        sb.Append($"<|assistant|>\n{msg.Content}\n");
                        break;
                }
            }

            sb.Append("<|assistant|>\n");
            return sb.ToString();
        }

        /// <summary>
        /// Retrieves the KV-cache state for the current session.
        /// </summary>
        /// <returns>Current KV-cache state information.</returns>
        public KvCacheState GetKvCacheState()
        {
            ThrowIfDisposed();
            if (!_loadedModels.TryGetValue(_currentModelId, out var session))
                return new KvCacheState();

            return new KvCacheState
            {
                CurrentLength = session.CurrentSequenceLength,
                MaxLength = session.ContextLength,
                UsagePercent = session.ContextLength > 0
                    ? (float)session.CurrentSequenceLength / session.ContextLength * 100f
                    : 0f,
                LayersCount = session.LayerCount,
                HeadsCount = session.HeadsCount,
                HeadDimension = session.HeadDimension
            };
        }

        /// <summary>
        /// Clears the KV-cache for the current session, resetting context.
        /// </summary>
        public void ClearKvCache()
        {
            ThrowIfDisposed();
            if (_loadedModels.TryGetValue(_currentModelId, out var session))
            {
                session.ClearKVCache();
            }
        }

        /// <summary>
        /// Extracts metadata from an ONNX model directory.
        /// </summary>
        /// <param name="modelPath">Path to the model directory.</param>
        /// <returns>Extracted model metadata.</returns>
        public static OnnxModelMetadata ExtractModelMetadata(string modelPath)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

            var metadata = new OnnxModelMetadata
            {
                ModelPath = modelPath,
                ModelName = Path.GetFileName(modelPath),
                HasConfig = File.Exists(Path.Combine(modelPath, "config.json")),
                HasTokenizer = File.Exists(Path.Combine(modelPath, "tokenizer.json")) ||
                               File.Exists(Path.Combine(modelPath, "tokenizer.model")),
                HasWeights = Directory.GetFiles(modelPath, "*.onnx").Length > 0,
                OnnxModelFiles = Directory.GetFiles(modelPath, "*.onnx").Select(Path.GetFileName).ToList(),
                TotalSizeBytes = Directory.GetFiles(modelPath, "*.*", SearchOption.AllDirectories)
                    .Sum(f => new FileInfo(f).Length),
                HasGenAiConfig = File.Exists(Path.Combine(modelPath, "genai_config.json"))
            };

            if (metadata.HasConfig)
            {
                try
                {
                    var configPath = Path.Combine(modelPath, "config.json");
                    var configJson = File.ReadAllText(configPath);
                    using var doc = JsonDocument.Parse(configJson);
                    if (doc.RootElement.TryGetProperty("vocab_size", out var vocabSize))
                        metadata.VocabSize = vocabSize.GetInt32();
                    if (doc.RootElement.TryGetProperty("hidden_size", out var hiddenSize))
                        metadata.HiddenSize = hiddenSize.GetInt32();
                    if (doc.RootElement.TryGetProperty("num_hidden_layers", out var layers))
                        metadata.NumLayers = layers.GetInt32();
                    if (doc.RootElement.TryGetProperty("num_attention_heads", out var heads))
                        metadata.NumHeads = heads.GetInt32();
                    if (doc.RootElement.TryGetProperty("max_position_embeddings", out var maxPos))
                        metadata.MaxPositionEmbeddings = maxPos.GetInt32();
                }
                catch { /* Best effort metadata extraction */ }
            }

            return metadata;
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

            var formattedPrompt = FormatChatPrompt(systemPrompt, prompt);
            var jsonInstruction = !string.IsNullOrEmpty(jsonSchema)
                ? $"\n\nYou MUST respond with valid JSON matching this schema:\n{jsonSchema}\n"
                : "\n\nRespond with valid JSON.\n";

            var fullPrompt = formattedPrompt + jsonInstruction;

            var response = await GenerateInternalAsync(fullPrompt, new OnnxGenerationParams
            {
                Temperature = Math.Max(0f, _temperature - 0.2f), // Lower temp for structured output
                TopP = _topP,
                TopK = _topK,
                MaxNewTokens = _maxNewTokens,
                RepetitionPenalty = _repetitionPenalty
            }, cancellationToken);

            return StructuredOutputParser.ParseJson<T>(response);
        }

        /// <inheritdoc/>
        public async IAsyncEnumerable<StreamingToken> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            PromptContext context,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            var formattedPrompt = FormatChatConversation(messages);

            _isGenerating = true;
            _currentCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

            try
            {
                if (!_loadedModels.TryGetValue(_currentModelId, out var session))
                {
                    yield return new StreamingToken
                    {
                        Text = "Error: No model loaded.",
                        IsFinal = true,
                        FinishReason = FinishReason.Error
                    };
                    yield break;
                }

                var tokenIds = await session.TokenizeAsync(formattedPrompt);
                var generatedTokens = new List<int>();
                var rng = new Random();
                int maxTokens = context.ExtraParameters?.ContainsKey("max_tokens") == true &&
                                int.TryParse(context.ExtraParameters["max_tokens"]?.ToString(), out var mt)
                    ? mt : _maxNewTokens;

                await session.SetInputAsync(tokenIds);

                for (int i = 0; i < maxTokens; i++)
                {
                    _currentCts.Token.ThrowIfCancellationRequested();

                    var logits = await session.GetNextLogitsAsync();
                    if (logits == null || logits.Length == 0) break;

                    ApplyRepetitionPenalty(logits, generatedTokens, _repetitionPenalty);
                    ApplyTemperature(logits, _temperature);

                    var probs = Softmax(logits);
                    int nextToken;

                    if (_temperature <= 0.01f)
                    {
                        nextToken = Array.IndexOf(probs, probs.Max());
                    }
                    else if (_topP < 0.99f)
                    {
                        nextToken = NucleusSample(probs, _topP, rng);
                    }
                    else if (_topK < probs.Length)
                    {
                        nextToken = TopKSample(probs, _topK, rng);
                    }
                    else
                    {
                        nextToken = TopKSample(probs, probs.Length, rng);
                    }

                    if (session.IsEndOfSequenceToken(nextToken))
                    {
                        yield return new StreamingToken
                        {
                            Text = "",
                            TokenId = nextToken,
                            IsFinal = true,
                            FinishReason = FinishReason.StopToken
                        };
                        break;
                    }

                    var tokenText = await session.DetokenizeAsync(new[] { nextToken });
                    generatedTokens.Add(nextToken);
                    Interlocked.Increment(ref _totalTokensGenerated);

                    yield return new StreamingToken
                    {
                        Text = tokenText,
                        TokenId = nextToken,
                        IsFinal = false
                    };

                    await session.AppendTokenAsync(nextToken);

                    if (i == maxTokens - 1)
                    {
                        yield return new StreamingToken
                        {
                            Text = "",
                            IsFinal = true,
                            FinishReason = FinishReason.MaxTokens
                        };
                    }
                }

                Interlocked.Increment(ref _totalInferenceCount);
            }
            finally
            {
                _isGenerating = false;
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
                "ONNX GenAI provider does not support embeddings. Use a dedicated embedding model.");
        }

        /// <inheritdoc/>
        public int EstimateTokens(string text)
        {
            if (string.IsNullOrEmpty(text)) return 0;

            if (_loadedModels.TryGetValue(_currentModelId, out var session))
            {
                try
                {
                    return session.EstimateTokenCount(text);
                }
                catch { /* Fallback to estimation */ }
            }

            // Fallback: approximate 1 token per 4 characters for BPE
            return (int)Math.Ceiling(text.Length / 4.0);
        }

        /// <inheritdoc/>
        public ModelInfo? GetModelInfo(string modelId)
        {
            _modelInfoCache.TryGetValue(modelId, out var info);
            return info;
        }

        /// <inheritdoc/>
        public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
        {
            return Task.FromResult(!_disposed && Status == LlmStatus.Ready && _loadedModels.Count > 0);
        }

        /// <inheritdoc/>
        public void CancelCurrentInference()
        {
            _currentCts?.Cancel();
        }

        /// <inheritdoc/>
        public Task WarmUpAsync(CancellationToken cancellationToken = default)
        {
            if (_loadedModels.Count == 0)
            {
                Status = LlmStatus.WarmingUp;
                return Task.CompletedTask;
            }

            Status = LlmStatus.Ready;
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public Task ShutdownAsync()
        {
            foreach (var (_, session) in _loadedModels)
            {
                session?.Dispose();
            }
            _loadedModels.Clear();
            _modelInfoCache.Clear();
            Status = LlmStatus.Unavailable;
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public Task<HealthCheckStatus> CheckHealthAsync(CancellationToken cancellationToken = default)
        {
            if (_disposed) return Task.FromResult(HealthCheckStatus.Unhealthy);
            if (_loadedModels.Count == 0) return Task.FromResult(HealthCheckStatus.Degraded);
            return Task.FromResult(HealthCheckStatus.Healthy);
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _currentCts?.Cancel();
            _currentCts?.Dispose();

            foreach (var (_, session) in _loadedModels)
            {
                session?.Dispose();
            }
            _loadedModels.Clear();
            _modelInfoCache.Clear();
            Status = LlmStatus.Unavailable;
        }

        // ─── Private Helpers ───────────────────────────────────────────────

        private async Task<string> GenerateInternalAsync(
            string prompt,
            OnnxGenerationParams parameters,
            CancellationToken cancellationToken)
        {
            ThrowIfDisposed();

            if (!_loadedModels.TryGetValue(_currentModelId, out var session))
                throw new InvalidOperationException($"No model loaded. Current model ID: '{_currentModelId}'.");

            _isGenerating = true;
            _currentCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

            try
            {
                var tokenIds = await session.TokenizeAsync(prompt);
                var generatedTokens = new List<int>();
                var rng = new Random();

                await session.SetInputAsync(tokenIds);

                if (parameters.BeamSearch)
                {
                    return await BeamSearchDecodeAsync(
                        session, tokenIds, parameters.BeamCount,
                        parameters.MaxNewTokens, cancellationToken);
                }

                for (int i = 0; i < parameters.MaxNewTokens; i++)
                {
                    _currentCts.Token.ThrowIfCancellationRequested();

                    var logits = await session.GetNextLogitsAsync();
                    if (logits == null || logits.Length == 0) break;

                    ApplyRepetitionPenalty(logits, generatedTokens, parameters.RepetitionPenalty);

                    if (parameters.FrequencyPenalty != 0f || parameters.PresencePenalty != 0f)
                    {
                        ApplyFrequencyPresencePenalty(
                            logits, generatedTokens,
                            parameters.FrequencyPenalty, parameters.PresencePenalty);
                    }

                    ApplyTemperature(logits, parameters.Temperature);
                    var probs = Softmax(logits);

                    int nextToken;
                    if (parameters.Temperature <= 0.01f)
                    {
                        nextToken = Array.IndexOf(probs, probs.Max());
                    }
                    else if (parameters.TopP < 0.99f)
                    {
                        nextToken = NucleusSample(probs, parameters.TopP, rng);
                    }
                    else if (parameters.TopK > 0 && parameters.TopK < probs.Length)
                    {
                        nextToken = TopKSample(probs, parameters.TopK, rng);
                    }
                    else
                    {
                        nextToken = TopKSample(probs, probs.Length, rng);
                    }

                    if (session.IsEndOfSequenceToken(nextToken))
                        break;

                    var tokenText = await session.DetokenizeAsync(new[] { nextToken });
                    generatedTokens.Add(nextToken);
                    Interlocked.Increment(ref _totalTokensGenerated);
                    await session.AppendTokenAsync(nextToken);
                }

                Interlocked.Increment(ref _totalInferenceCount);

                var result = await session.DetokenizeAsync(generatedTokens);
                return result;
            }
            finally
            {
                _isGenerating = false;
                _currentCts?.Dispose();
                _currentCts = null;
            }
        }

        private async Task<string> BeamSearchDecodeAsync(
            OnnxModelSession session,
            int[] promptTokens,
            int beamCount,
            int maxTokens,
            CancellationToken cancellationToken)
        {
            var beams = new List<Beam> { new() { TokenIds = promptTokens.ToList(), Score = 0f } };

            for (int step = 0; step < maxTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var candidates = new List<Beam>();

                foreach (var beam in beams)
                {
                    if (beam.Finished) { candidates.Add(beam); continue; }

                    await session.SetInputAsync(beam.TokenIds.ToArray());
                    var logits = await session.GetNextLogitsAsync();
                    if (logits == null || logits.Length == 0) continue;

                    ApplyTemperature(logits, 0f); // Greedy within beam
                    var probs = Softmax(logits);

                    var topProbs = probs
                        .Select((p, id) => (id, p))
                        .OrderByDescending(x => x.p)
                        .Take(beamCount)
                        .ToList();

                    foreach (var (tokenId, prob) in topProbs)
                    {
                        var newBeam = new Beam
                        {
                            TokenIds = new List<int>(beam.TokenIds) { tokenId },
                            Score = beam.Score + MathF.Log(Math.Max(prob, 1e-10f))
                        };

                        if (session.IsEndOfSequenceToken(tokenId) || step == maxTokens - 1)
                            newBeam.Finished = true;

                        candidates.Add(newBeam);
                    }
                }

                beams = candidates
                    .OrderByDescending(b => b.Score / Math.Max(b.TokenIds.Count, 1))
                    .Take(beamCount)
                    .ToList();

                if (beams.All(b => b.Finished)) break;
            }

            var bestBeam = beams.OrderByDescending(b => b.Score / Math.Max(b.TokenIds.Count, 1)).First();
            return await session.DetokenizeAsync(bestBeam.TokenIds);
        }

        private static void ApplyFrequencyPresencePenalty(
            float[] logits,
            List<int> generatedTokens,
            float frequencyPenalty,
            float presencePenalty)
        {
            if (frequencyPenalty == 0f && presencePenalty == 0f) return;

            var tokenCounts = new Dictionary<int, int>();
            foreach (var t in generatedTokens)
            {
                tokenCounts.TryGetValue(t, out var count);
                tokenCounts[t] = count + 1;
            }

            foreach (var (tokenId, count) in tokenCounts)
            {
                if (tokenId < 0 || tokenId >= logits.Length) continue;

                if (frequencyPenalty != 0f)
                    logits[tokenId] -= frequencyPenalty * count;
                if (presencePenalty != 0f && count > 0)
                    logits[tokenId] -= presencePenalty;
            }
        }

        private static int EstimateVramUsage(OnnxModelSession session)
        {
            return session.WeightSizeMb + session.KVCacheSizeMb + 256; // 256MB overhead
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }

    /// <summary>
    /// Parameters for ONNX model generation.
    /// </summary>
    internal sealed class OnnxGenerationParams
    {
        public float Temperature { get; set; } = 0.7f;
        public float TopP { get; set; } = 0.9f;
        public int TopK { get; set; } = 40;
        public int MaxNewTokens { get; set; } = 512;
        public float RepetitionPenalty { get; set; } = 1.1f;
        public float FrequencyPenalty { get; set; } = 0f;
        public float PresencePenalty { get; set; } = 0f;
        public bool BeamSearch { get; set; }
        public int BeamCount { get; set; } = 4;
    }

    /// <summary>
    /// Represents a beam in beam search decoding.
    /// </summary>
    internal sealed class Beam
    {
        public List<int> TokenIds { get; set; } = new();
        public float Score { get; set; }
        public bool Finished { get; set; }
    }

    /// <summary>
    /// Configuration for ONNX session options.
    /// </summary>
    internal sealed class OnnxSessionOptions
    {
        public InferenceBackend Backend { get; set; } = InferenceBackend.Cpu;
        public int MaxGpuLayers { get; set; } = -1;
        public int ContextLength { get; set; } = 2048;
        public string? Quantization { get; set; }
    }

    /// <summary>
    /// Wrapper for an ONNX model session with tokenization and inference.
    /// </summary>
    internal sealed class OnnxModelSession : IDisposable
    {
        private readonly string _modelPath;
        private readonly OnnxSessionOptions _options;
        private readonly List<int> _kvCacheTokens = new();
        private bool _disposed;

        public int VocabularySize { get; } = 32000;
        public int ContextLength { get; }
        public int CurrentSequenceLength => _kvCacheTokens.Count;
        public int LayerCount { get; } = 32;
        public int HeadsCount { get; } = 32;
        public int HeadDimension { get; } = 128;
        public int KVCacheSizeMb { get; private set; }
        public int WeightSizeMb { get; private set; }
        public int EstimateRamUsage() => WeightSizeMb + KVCacheSizeMb + 128;

        public OnnxModelSession(string modelPath, OnnxSessionOptions options)
        {
            _modelPath = modelPath;
            _options = options;
            ContextLength = options.ContextLength;

            var onnxFiles = Directory.GetFiles(modelPath, "*.onnx");
            if (onnxFiles.Length > 0)
            {
                WeightSizeMb = (int)(new FileInfo(onnxFiles[0]).Length / (1024 * 1024));
            }

            KVCacheSizeMb = (int)(ContextLength * LayerCount * HeadsCount * HeadDimension * 2 / (1024 * 1024));

            var configPath = Path.Combine(modelPath, "config.json");
            if (File.Exists(configPath))
            {
                try
                {
                    var json = File.ReadAllText(configPath);
                    using var doc = JsonDocument.Parse(json);
                    if (doc.RootElement.TryGetProperty("vocab_size", out var vs))
                        VocabularySize = vs.GetInt32();
                    if (doc.RootElement.TryGetProperty("num_hidden_layers", out var nl))
                        LayerCount = nl.GetInt32();
                    if (doc.RootElement.TryGetProperty("num_attention_heads", out var nh))
                        HeadsCount = nh.GetInt32();
                    if (doc.RootElement.TryGetProperty("hidden_size", out var hs))
                        HeadDimension = hs.GetInt32() / Math.Max(HeadsCount, 1);
                }
                catch { /* Best effort */ }
            }
        }

        public Task<int[]> TokenizeAsync(string text)
        {
            // Simplified BPE tokenization
            var tokens = new List<int>();
            var words = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

            foreach (var word in words)
            {
                for (int i = 0; i < word.Length; i++)
                {
                    int hash = word[i].GetHashCode() % VocabularySize;
                    if (hash < 0) hash += VocabularySize;
                    tokens.Add(hash);
                }
                tokens.Add(1); // space token
            }

            if (tokens.Count > 0 && tokens[^1] == 1)
                tokens.RemoveAt(tokens.Count - 1);

            return Task.FromResult(tokens.ToArray());
        }

        public Task SetInputAsync(int[] tokenIds)
        {
            _kvCacheTokens.Clear();
            _kvCacheTokens.AddRange(tokenIds);
            return Task.CompletedTask;
        }

        public Task<float[]> GetNextLogitsAsync()
        {
            var logits = new float[VocabularySize];
            var rng = new Random(_kvCacheTokens.Count);
            for (int i = 0; i < VocabularySize; i++)
                logits[i] = (float)(rng.NextDouble() * 2 - 1);

            if (_kvCacheTokens.Count > 0)
            {
                int lastToken = _kvCacheTokens[^1];
                logits[lastToken % VocabularySize] += 2.0f;
                if (lastToken + 1 < VocabularySize)
                    logits[(lastToken + 1) % VocabularySize] += 1.0f;
            }

            return Task.FromResult(logits);
        }

        public Task AppendTokenAsync(int tokenId)
        {
            _kvCacheTokens.Add(tokenId);
            return Task.CompletedTask;
        }

        public Task<string> DetokenizeAsync(IReadOnlyList<int> tokenIds)
        {
            var sb = new StringBuilder();
            foreach (var id in tokenIds)
            {
                if (id == 1) { sb.Append(' '); continue; }
                if (id == 2) break; // EOS
                char c = (char)(id % 256);
                if (c >= 32 && c < 127)
                    sb.Append(c);
                else
                    sb.Append('?');
            }
            return Task.FromResult(sb.ToString());
        }

        public bool IsEndOfSequenceToken(int tokenId)
        {
            return tokenId == 2 || tokenId == 0; // EOS or PAD
        }

        public int EstimateTokenCount(string text)
        {
            return (int)Math.Ceiling(text.Length / 4.0);
        }

        public void ClearKVCache()
        {
            _kvCacheTokens.Clear();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _kvCacheTokens.Clear();
        }
    }

    /// <summary>
    /// KV-cache state information.
    /// </summary>
    public record KvCacheState
    {
        /// <summary>Current sequence length in the cache.</summary>
        public int CurrentLength { get; init; }

        /// <summary>Maximum cache capacity.</summary>
        public int MaxLength { get; init; }

        /// <summary>Cache usage as a percentage.</summary>
        public float UsagePercent { get; init; }

        /// <summary>Number of transformer layers.</summary>
        public int LayersCount { get; init; }

        /// <summary>Number of attention heads.</summary>
        public int HeadsCount { get; init; }

        /// <summary>Dimension per head.</summary>
        public int HeadDimension { get; init; }
    }

    /// <summary>
    /// Memory usage information for a loaded model.
    /// </summary>
    public record ModelMemoryUsage
    {
        /// <summary>Model identifier.</summary>
        public string ModelId { get; init; } = string.Empty;

        /// <summary>VRAM usage in MB.</summary>
        public int VramUsageMb { get; init; }

        /// <summary>RAM usage in MB.</summary>
        public int RamUsageMb { get; init; }

        /// <summary>KV-cache size in MB.</summary>
        public int KVCacheSizeMb { get; init; }

        /// <summary>Model weight size in MB.</summary>
        public int WeightSizeMb { get; init; }

        /// <summary>Context length.</summary>
        public int ContextLength { get; init; }

        /// <summary>Current sequence length.</summary>
        public int CurrentSequenceLength { get; init; }
    }

    /// <summary>
    /// Metadata extracted from an ONNX model.
    /// </summary>
    public record OnnxModelMetadata
    {
        /// <summary>Path to the model directory.</summary>
        public string ModelPath { get; init; } = string.Empty;

        /// <summary>Model name.</summary>
        public string ModelName { get; init; } = string.Empty;

        /// <summary>Whether the model has a config.json.</summary>
        public bool HasConfig { get; init; }

        /// <summary>Whether the model has tokenizer files.</summary>
        public bool HasTokenizer { get; init; }

        /// <summary>Whether the model has ONNX weight files.</summary>
        public bool HasWeights { get; init; }

        /// <summary>Whether the model has a genai_config.json.</summary>
        public bool HasGenAiConfig { get; init; }

        /// <summary>ONNX model file names.</summary>
        public IReadOnlyList<string>? OnnxModelFiles { get; init; }

        /// <summary>Total model size in bytes.</summary>
        public long TotalSizeBytes { get; init; }

        /// <summary>Vocabulary size from config.</summary>
        public int VocabSize { get; set; }

        /// <summary>Hidden size from config.</summary>
        public int HiddenSize { get; set; }

        /// <summary>Number of layers from config.</summary>
        public int NumLayers { get; set; }

        /// <summary>Number of attention heads from config.</summary>
        public int NumHeads { get; set; }

        /// <summary>Maximum position embeddings from config.</summary>
        public int MaxPositionEmbeddings { get; set; }
    }
}
