using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using Synapse.Core.Security;

namespace Synapse.Infrastructure.Configuration
{
    public sealed class SynapseConfig
    {
        public int Width { get; set; } = 1280;
        public int Height { get; set; } = 720;
        public bool EnableValidation { get; set; } = true;
        public string? ScenePath { get; set; }
        public string QualityPreset { get; set; } = "High";
        public float PhysicsBudgetMs { get; set; } = 4.0f;
        public float SimulationBudgetMs { get; set; } = 4.0f;
        public bool Headless { get; set; }
        public string LogLevel { get; set; } = "Information";
        public LlmConfig Llm { get; set; } = new();
        public string ProjectsDirectory { get; set; } =
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "Synapse", "Projects");

        public static string UserConfigDirectory =>
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Synapse");

        public static string UserConfigPath => Path.Combine(UserConfigDirectory, "appsettings.json");

        public static SynapseConfig Load(string? path = null, string[]? args = null)
        {
            var config = new SynapseConfig();
            // Prefer user config; fall back to shipped appsettings next to the exe.
            path ??= File.Exists(UserConfigPath)
                ? UserConfigPath
                : Path.Combine(AppContext.BaseDirectory, "appsettings.json");

            if (File.Exists(path))
            {
                var json = File.ReadAllText(path);
                if (json.Length > 1_000_000)
                    throw new InvalidOperationException("Config file exceeds size limit.");
                var loaded = JsonSerializer.Deserialize(json, SynapseConfigJsonContext.Default.SynapseConfig);
                if (loaded != null) config = loaded;
            }

            ApplyEnvironment(config);
            if (args != null) ApplyCli(config, args);

            // Normalize projects directory and validate optional scene path.
            config.ProjectsDirectory = Path.GetFullPath(config.ProjectsDirectory);
            Directory.CreateDirectory(config.ProjectsDirectory);
            if (!string.IsNullOrWhiteSpace(config.ScenePath))
            {
                var scene = Path.GetFullPath(config.ScenePath);
                if (!PathSecurity.IsUnderRoot(config.ProjectsDirectory, scene) &&
                    !PathSecurity.IsUnderRoot(AppContext.BaseDirectory, scene) &&
                    !PathSecurity.IsUnderRoot(Path.Combine(AppContext.BaseDirectory, "samples"), scene))
                {
                    throw new UnauthorizedAccessException("Scene path must be under Projects or application samples.");
                }
                config.ScenePath = scene;
            }

            if (!string.IsNullOrWhiteSpace(config.Llm.OllamaBaseUrl))
                config.Llm.OllamaBaseUrl = UrlSecurity.ValidateOutboundUri(config.Llm.OllamaBaseUrl, allowLoopbackHttp: true).ToString();

            return config;
        }

        public void Save(string? path = null)
        {
            // Never write secrets-bearing user config into the install directory.
            path ??= UserConfigPath;
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
            var json = JsonSerializer.Serialize(this, SynapseConfigJsonContext.Default.SynapseConfig);
            File.WriteAllText(path, json);
        }

        private static void ApplyEnvironment(SynapseConfig config)
        {
            var w = Environment.GetEnvironmentVariable("SYNAPSE_WIDTH");
            if (int.TryParse(w, out var width)) config.Width = width;
            var h = Environment.GetEnvironmentVariable("SYNAPSE_HEIGHT");
            if (int.TryParse(h, out var height)) config.Height = height;
            var scene = Environment.GetEnvironmentVariable("SYNAPSE_SCENE");
            if (!string.IsNullOrWhiteSpace(scene)) config.ScenePath = scene;

            config.Llm.OpenAiApiKey ??= Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            config.Llm.AnthropicApiKey ??= Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY");
            config.Llm.GeminiApiKey ??= Environment.GetEnvironmentVariable("GEMINI_API_KEY");
            config.Llm.AzureApiKey ??= Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");
            config.Llm.OllamaBaseUrl ??= Environment.GetEnvironmentVariable("OLLAMA_HOST") ?? "http://127.0.0.1:11434";
        }

        private static void ApplyCli(SynapseConfig config, string[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                string? Next() => i + 1 < args.Length ? args[++i] : null;

                switch (arg)
                {
                    case "--width":
                        if (int.TryParse(Next(), out var w)) config.Width = w;
                        break;
                    case "--height":
                        if (int.TryParse(Next(), out var h)) config.Height = h;
                        break;
                    case "--validation":
                        config.EnableValidation = true;
                        break;
                    case "--no-validation":
                        config.EnableValidation = false;
                        break;
                    case "--scene":
                        config.ScenePath = Next();
                        break;
                    case "--headless":
                        config.Headless = true;
                        break;
                    case "--quality":
                        var q = Next();
                        if (!string.IsNullOrWhiteSpace(q)) config.QualityPreset = q!;
                        break;
                }
            }
        }
    }

    public sealed class LlmConfig
    {
        public string PreferredProvider { get; set; } = "OllamaLocal";
        public string? OllamaBaseUrl { get; set; } = "http://127.0.0.1:11434";
        public string DefaultModel { get; set; } = "llama3.2";
        [JsonIgnore] public string? OpenAiApiKey { get; set; }
        [JsonIgnore] public string? AnthropicApiKey { get; set; }
        [JsonIgnore] public string? GeminiApiKey { get; set; }
        [JsonIgnore] public string? AzureApiKey { get; set; }
    }

    [JsonSerializable(typeof(SynapseConfig))]
    [JsonSerializable(typeof(LlmConfig))]
    [JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
    internal sealed partial class SynapseConfigJsonContext : JsonSerializerContext
    {
    }
}
