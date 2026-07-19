using System;
using System.Collections.Generic;
using GDNN.Llm;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;

namespace Synapse.Runtime
{
    /// <summary>
    /// Registers configured LLM providers on the hybrid router at startup.
    /// </summary>
    public static class LlmProviderBootstrap
    {
        public sealed class BootstrapResult
        {
            public IReadOnlyList<string> Registered { get; init; } = Array.Empty<string>();
            public string Summary { get; init; } = "No LLM providers configured";
        }

        public static BootstrapResult Register(HybridLlmRouter router, SynapseConfig config, ISynapseLogger logger)
        {
            ArgumentNullException.ThrowIfNull(router);
            ArgumentNullException.ThrowIfNull(config);
            ArgumentNullException.ThrowIfNull(logger);

            var registered = new List<string>();
            var llm = config.Llm;

            if (!string.IsNullOrWhiteSpace(llm.OllamaBaseUrl))
            {
                var ollama = new OllamaProvider(
                    llm.OllamaBaseUrl,
                    llm.DefaultModel,
                    timeoutSeconds: 120);
                router.RegisterProvider("OllamaLocal", ollama, new ProviderConfig
                {
                    Mode = LlmProviderMode.LocalOllama,
                    BaseUrl = llm.OllamaBaseUrl,
                    DefaultModel = llm.DefaultModel
                });
                registered.Add("Ollama (local)");
                logger.Info("LLM", $"Registered Ollama at {llm.OllamaBaseUrl} model={llm.DefaultModel}");
            }

            if (!string.IsNullOrWhiteSpace(llm.OpenAiApiKey))
            {
                router.RegisterProvider("OpenAI", new OpenAiProvider(llm.OpenAiApiKey, llm.DefaultModel));
                registered.Add("OpenAI");
            }

            if (!string.IsNullOrWhiteSpace(llm.AnthropicApiKey))
            {
                router.RegisterProvider("Anthropic", new AnthropicProvider(llm.AnthropicApiKey));
                registered.Add("Anthropic");
            }

            if (!string.IsNullOrWhiteSpace(llm.GeminiApiKey))
            {
                router.RegisterProvider("Gemini", new GoogleGeminiProvider(llm.GeminiApiKey));
                registered.Add("Gemini");
            }

            string summary = registered.Count == 0
                ? "LLM offline — install Ollama or set API keys"
                : string.Join(" · ", registered);

            logger.Info("LLM", $"Providers ready ({registered.Count}): {summary}");
            return new BootstrapResult { Registered = registered, Summary = summary };
        }
    }
}
