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
    /// Contract for all LLM inference providers in the G-DNN Engine pipeline.
    /// Each provider (ONNX, Ollama, OpenAI, Anthropic, Google) implements this
    /// interface to provide a unified abstraction for the router.
    /// </summary>
    public interface ILlmInferenceProvider : IDisposable
    {
        /// <summary>Human-readable name of this provider.</summary>
        string ProviderName { get; }

        /// <summary>Whether this provider requires network access.</summary>
        bool RequiresNetwork { get; }

        /// <summary>Current operational status.</summary>
        LlmStatus Status { get; }

        /// <summary>The provider's operational mode.</summary>
        LlmProviderMode Mode { get; }

        /// <summary>Supported models by this provider.</summary>
        IReadOnlyList<ModelInfo> AvailableModels { get; }

        /// <summary>Current provider health information.</summary>
        ProviderHealth Health { get; }

        /// <summary>
        /// Performs structured inference, returning a typed result parsed from the LLM output.
        /// </summary>
        Task<StructuredOutput<T>> InferStructuredAsync<T>(
            string prompt,
            string? systemPrompt,
            string? jsonSchema,
            PromptContext context,
            CancellationToken cancellationToken = default) where T : class;

        /// <summary>
        /// Streams tokens from the LLM as they are generated.
        /// </summary>
        IAsyncEnumerable<StreamingToken> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            PromptContext context,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates an embedding vector for the given text.
        /// </summary>
        Task<EmbeddingResult> GetEmbeddingAsync(
            string text,
            EmbeddingModel model = EmbeddingModel.TextEmbedding3Small,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Estimates the number of tokens for the given text.
        /// </summary>
        int EstimateTokens(string text);

        /// <summary>
        /// Returns model information for a specific model ID.
        /// </summary>
        ModelInfo? GetModelInfo(string modelId);

        /// <summary>
        /// Checks if the provider is currently available for inference.
        /// </summary>
        Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Cancels the current inference operation if one is in progress.
        /// </summary>
        void CancelCurrentInference();

        /// <summary>
        /// Warms up the provider by loading models or establishing connections.
        /// </summary>
        Task WarmUpAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Gracefully shuts down the provider, releasing resources.
        /// </summary>
        Task ShutdownAsync();

        /// <summary>
        /// Performs a health check on the provider.
        /// </summary>
        Task<HealthCheckStatus> CheckHealthAsync(CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Contract for intent parsing services that convert natural language to structured intents.
    /// </summary>
    public interface IIntentParser
    {
        /// <summary>
        /// Classifies the intent of a user's input text.
        /// </summary>
        Task<IntentClassificationResult> ClassifyIntentAsync(
            string input,
            PromptContext? context = null,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Extracts named entities from text.
        /// </summary>
        Task<IReadOnlyList<EntityExtractionResult>> ExtractEntitiesAsync(
            string text,
            CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Contract for adapters that convert between LLM output and G-DNN Engine data types.
    /// </summary>
    public interface IGdnAdapter<TInput, TOutput> where TOutput : class
    {
        /// <summary>
        /// Converts LLM output to a G-DNN Engine data type.
        /// </summary>
        StructuredOutput<TOutput> Convert(TInput llmOutput, string rawText);

        /// <summary>
        /// Validates that converted output meets constraints.
        /// </summary>
        (bool IsValid, IReadOnlyList<string> Errors) Validate(TOutput output);

        /// <summary>
        /// Generates a natural language description of the output.
        /// </summary>
        string Describe(TOutput output);
    }
}
