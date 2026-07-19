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
    /// Specifies the operational mode of the LLM provider.
    /// </summary>
    public enum LlmProviderMode
    {
        /// <summary>Local inference using ONNX Runtime GenAI.</summary>
        LocalOnnx,
        /// <summary>Local inference using Ollama server.</summary>
        LocalOllama,
        /// <summary>Remote inference via OpenAI API.</summary>
        RemoteOpenAI,
        /// <summary>Remote inference via Anthropic API.</summary>
        RemoteAnthropic,
        /// <summary>Remote inference via Google Gemini API.</summary>
        RemoteGoogle,
        /// <summary>Remote inference via Azure OpenAI.</summary>
        RemoteAzure,
        /// <summary>Hybrid mode selecting best available provider.</summary>
        Hybrid
    }

    /// <summary>
    /// Classifies the complexity of a prompt for routing decisions.
    /// </summary>
    public enum PromptComplexity
    {
        /// <summary>Single-word or trivial responses.</summary>
        Trivial,
        /// <summary>Simple factual queries.</summary>
        Simple,
        /// <summary>Moderate reasoning required.</summary>
        Moderate,
        /// <summary>Complex multi-step reasoning.</summary>
        Complex,
        /// <summary>Expert-level domain knowledge.</summary>
        Expert
    }

    /// <summary>
    /// Identifies the type of task being requested from the LLM.
    /// </summary>
    public enum LlmTaskType
    {
        /// <summary>Creating genome specifications from descriptions.</summary>
        GenomeCreation,
        /// <summary>Generating mutations for existing genomes.</summary>
        GenomeMutation,
        /// <summary>Generating behavior descriptions for entities.</summary>
        BehaviorGeneration,
        /// <summary>Suggesting material properties and textures.</summary>
        MaterialSuggestion,
        /// <summary>Describing scene compositions in natural language.</summary>
        SceneDescription,
        /// <summary>Answering general knowledge queries.</summary>
        QueryAnswering,
        /// <summary>Generating code snippets or logic.</summary>
        CodeGeneration,
        /// <summary>Generating narrative or story content.</summary>
        NarrativeGeneration,
        /// <summary>Classifying text into semantic categories.</summary>
        SemanticClassification,
        /// <summary>Generating vector embeddings for text.</summary>
        EmbeddingGeneration,
        /// <summary>Classifying user intent from input.</summary>
        IntentClassification,
        /// <summary>Extracting named entities from text.</summary>
        EntityExtraction,
        /// <summary>Summarizing longer content.</summary>
        Summarization
    }

    /// <summary>
    /// Specifies the hardware backend for model inference.
    /// </summary>
    public enum InferenceBackend
    {
        /// <summary>CPU inference (slowest, most compatible).</summary>
        Cpu,
        /// <summary>NVIDIA CUDA GPU acceleration.</summary>
        Cuda,
        /// <summary>Microsoft DirectML acceleration.</summary>
        DirectMl,
        /// <summary>Apple CoreML acceleration.</summary>
        CoreMl,
        /// <summary>Vulkan GPU acceleration.</summary>
        Vulkan,
        /// <summary>Apple Metal GPU acceleration.</summary>
        Metal,
        /// <summary>OpenCL GPU acceleration.</summary>
        OpenCL
    }

    /// <summary>
    /// Identifies the tokenizer algorithm used by a model.
    /// </summary>
    public enum TokenizerType
    {
        /// <summary>Byte-Pair Encoding (GPT-2, LLaMA, etc.).</summary>
        BPE,
        /// <summary>WordPiece (BERT, DistilBERT, etc.).</summary>
        WordPiece,
        /// <summary>SentencePiece (T5, PaLM, etc.).</summary>
        SentencePiece,
        /// <summary>Tiktoken (OpenAI models).</summary>
        Tiktoken
    }

    /// <summary>
    /// Identifies the role of a message in a conversation.
    /// </summary>
    public enum MessageRole
    {
        /// <summary>System-level instructions.</summary>
        System,
        /// <summary>User input message.</summary>
        User,
        /// <summary>Assistant (LLM) response.</summary>
        Assistant,
        /// <summary>Function call result.</summary>
        Function,
        /// <summary>Tool call result.</summary>
        Tool
    }

    /// <summary>
    /// Current operational status of an LLM provider.
    /// </summary>
    public enum LlmStatus
    {
        /// <summary>Provider is ready to accept requests.</summary>
        Ready,
        /// <summary>Provider is loading a model.</summary>
        Loading,
        /// <summary>Provider is currently unavailable.</summary>
        Unavailable,
        /// <summary>Provider encountered an error.</summary>
        Error,
        /// <summary>Provider is rate-limited.</summary>
        Throttled,
        /// <summary>Provider is warming up models.</summary>
        WarmingUp
    }

    /// <summary>
    /// State of the circuit breaker pattern for provider health management.
    /// </summary>
    public enum CircuitBreakerState
    {
        /// <summary>Normal operation, requests pass through.</summary>
        Closed,
        /// <summary>Circuit is open, requests are blocked.</summary>
        Open,
        /// <summary>Testing if the provider has recovered.</summary>
        HalfOpen
    }

    /// <summary>
    /// Reason the LLM stopped generating tokens.
    /// </summary>
    public enum FinishReason
    {
        /// <summary>Model generated an end-of-text token.</summary>
        StopToken,
        /// <summary>Reached maximum token limit.</summary>
        MaxTokens,
        /// <summary>Content was filtered by safety systems.</summary>
        ContentFilter,
        /// <summary>Model requested a function call.</summary>
        FunctionCall,
        /// <summary>An error occurred during generation.</summary>
        Error
    }

    /// <summary>
    /// Identifies a specific embedding model.
    /// </summary>
    public enum EmbeddingModel
    {
        /// <summary>OpenAI text-embedding-3-small (1536 dimensions).</summary>
        TextEmbedding3Small,
        /// <summary>OpenAI text-embedding-3-large (3072 dimensions).</summary>
        TextEmbedding3Large,
        /// <summary>Nomic Embed (768 dimensions).</summary>
        NomicEmbed,
        /// <summary>Sentence-Transformers all-MiniLM-L6-v2 (384 dimensions).</summary>
        AllMiniLM,
        /// <summary>Custom embedding model with configurable dimensions.</summary>
        Custom
    }

    /// <summary>
    /// Priority levels for LLM request queuing.
    /// </summary>
    public enum RequestPriority
    {
        /// <summary>Low priority background tasks.</summary>
        Low = 0,
        /// <summary>Normal priority requests.</summary>
        Normal = 1,
        /// <summary>High priority for interactive use.</summary>
        High = 2,
        /// <summary>Critical priority for UI-blocking operations.</summary>
        Critical = 3
    }

    /// <summary>
    /// Truncation strategy when content exceeds token limits.
    /// </summary>
    public enum TruncationStrategy
    {
        /// <summary>Keep the beginning, truncate the end.</summary>
        Head,
        /// <summary>Truncate the beginning, keep the end.</summary>
        Tail,
        /// <summary>Keep beginning and end, truncate the middle.</summary>
        Middle,
        /// <summary>Balanced truncation from both ends.</summary>
        Balanced
    }

    /// <summary>
    /// Type of cache lookup to perform.
    /// </summary>
    public enum CacheLookupType
    {
        /// <summary>Exact match on prompt hash.</summary>
        Exact,
        /// <summary>Semantic similarity match using embeddings.</summary>
        Semantic,
        /// <summary>Try exact first, fall back to semantic.</summary>
        Hybrid
    }

    /// <summary>
    /// Severity level for LLM cost alerts.
    /// </summary>
    public enum CostAlertSeverity
    {
        /// <summary>Informational notice.</summary>
        Info,
        /// <summary>Warning approaching budget threshold.</summary>
        Warning,
        /// <summary>Critical budget threshold exceeded.</summary>
        Critical
    }

    /// <summary>
    /// Type of load balancing strategy for providers.
    /// </summary>
    public enum LoadBalancingStrategy
    {
        /// <summary>Round-robin across available providers.</summary>
        RoundRobin,
        /// <summary>Route to provider with fewest active requests.</summary>
        LeastLoaded,
        /// <summary>Route to fastest responding provider.</summary>
        FastestResponse,
        /// <summary>Route to highest quality provider within budget.</summary>
        QualityFirst
    }

    /// <summary>
    /// Status of a health check probe.
    /// </summary>
    public enum HealthCheckStatus
    {
        /// <summary>Health check passed.</summary>
        Healthy,
        /// <summary>Health check degraded but functional.</summary>
        Degraded,
        /// <summary>Health check failed.</summary>
        Unhealthy
    }

    /// <summary>
    /// Classification of data sensitivity for privacy-aware routing.
    /// </summary>
    public enum DataSensitivity
    {
        /// <summary>Non-sensitive data, can be sent to any provider.</summary>
        Public,
        /// <summary>Internal data, prefer local or trusted providers.</summary>
        Internal,
        /// <summary>Sensitive data, must stay on local providers.</summary>
        Confidential,
        /// <summary>Restricted data, only local inference allowed.</summary>
        Restricted
    }
}
