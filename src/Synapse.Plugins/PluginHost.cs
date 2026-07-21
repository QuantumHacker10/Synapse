using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Synapse.Core.Maturity;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;

namespace Synapse.Plugins;

/// <summary>
/// Trust level for plugin loading. AssemblyLoadContext isolation is <b>not</b> a security sandbox —
/// plugins run with full host privileges. Prefer <see cref="PluginTrustMode.RequireManifest"/> in production.
/// </summary>
public enum PluginTrustMode
{
    /// <summary>Load any ISynapsePlugin DLL (lab / developer default).</summary>
    Permissive = 0,

    /// <summary>Require a sidecar <c>plugin.synapse.json</c> with SHA-256 of the assembly.</summary>
    RequireManifest = 1
}

/// <summary>Sidecar manifest next to a plugin DLL (<c>plugin.synapse.json</c>).</summary>
public sealed class PluginManifest
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Version { get; set; } = "";
    public string AssemblyFile { get; set; } = "";
    public string Sha256 { get; set; } = "";
}

/// <summary>
/// Loads and manages Synapse plugins in isolated <see cref="AssemblyLoadContext"/> instances.
/// ALC isolation enables unload; it does not sandbox filesystem, network, or process access.
/// </summary>
[SynapseExperimental("Plugins.CSharp", "ALC isolation is not a security sandbox; use PluginTrustMode.RequireManifest for production.")]
public sealed class PluginHost : IDisposable
{
    public const int MaxPlugins = 64;

    private readonly ISynapseLogger _logger;
    private readonly PluginTrustMode _trustMode;
    private readonly List<(ISynapsePlugin Plugin, PluginLoadContext Context)> _loaded = new();

    public PluginHost(ISynapseLogger logger, PluginTrustMode trustMode = PluginTrustMode.Permissive)
    {
        _logger = logger;
        _trustMode = ResolveTrustMode(trustMode);
    }

    public PluginTrustMode TrustMode => _trustMode;

    public IReadOnlyList<PluginMetadata> LoadedPlugins =>
        _loaded.ConvertAll(p => p.Plugin.Metadata);

    public int LoadFromDirectory(string directory, EngineHost host)
    {
        if (!Directory.Exists(directory))
        {
            _logger.Warn("Plugins", $"Plugin directory not found: {directory}");
            return 0;
        }

        int count = 0;
        foreach (var dll in Directory.EnumerateFiles(directory, "*.dll", SearchOption.TopDirectoryOnly))
        {
            if (_loaded.Count >= MaxPlugins)
            {
                _logger.Warn("Plugins", $"Plugin limit ({MaxPlugins}) reached; skipping remaining assemblies.");
                break;
            }

            if (LoadPluginAssembly(dll, host))
                count++;
        }

        _logger.Info("Plugins", $"Loaded {count} plugin(s) from {directory} (trust={_trustMode})");
        return count;
    }

    public bool LoadPluginAssembly(string assemblyPath, EngineHost host)
    {
        PluginLoadContext? context = null;
        lock (_loaded)
        {
            if (_loaded.Count >= MaxPlugins)
            {
                _logger.Warn("Plugins", $"Cannot load {assemblyPath}: plugin limit reached.");
                return false;
            }
        }

        try
        {
            var full = Path.GetFullPath(assemblyPath);
            if (!File.Exists(full) || !full.EndsWith(".dll", StringComparison.OrdinalIgnoreCase))
            {
                _logger.Warn("Plugins", $"Not a plugin DLL: {assemblyPath}");
                return false;
            }

            if (!VerifyTrust(full))
                return false;

            context = new PluginLoadContext(full);
            var assembly = context.LoadFromAssemblyPath(full);
            var pluginType = assembly.GetTypes()
                .FirstOrDefault(t => typeof(ISynapsePlugin).IsAssignableFrom(t) && t is { IsAbstract: false, IsInterface: false });

            if (pluginType == null)
            {
                _logger.Warn("Plugins", $"No ISynapsePlugin in {Path.GetFileName(full)}");
                context.Unload();
                return false;
            }

            if (Activator.CreateInstance(pluginType) is not ISynapsePlugin plugin)
            {
                context.Unload();
                return false;
            }

            var pluginDir = Path.GetDirectoryName(full) ?? AppContext.BaseDirectory;
            plugin.OnLoad(new PluginHostContext { Host = host, PluginDirectory = pluginDir });
            lock (_loaded)
            {
                if (_loaded.Count >= MaxPlugins)
                {
                    try
                    { plugin.OnUnload(); }
                    catch { /* ignore */ }
                    context.Unload();
                    return false;
                }
                _loaded.Add((plugin, context));
            }
            context = null;
            _logger.Info("Plugins", $"Loaded {plugin.Metadata.Name} v{plugin.Metadata.Version}");
            return true;
        }
        catch (Exception ex)
        {
            if (context != null)
            {
                try
                { context.Unload(); }
                catch { /* ignore */ }
            }
            _logger.Error("Plugins", $"Failed to load {assemblyPath}: {ex.Message}");
            return false;
        }
    }

    public void Dispose()
    {
        List<(ISynapsePlugin Plugin, PluginLoadContext Context)> snapshot;
        lock (_loaded)
        {
            snapshot = _loaded.ToList();
            _loaded.Clear();
        }

        foreach (var (plugin, context) in snapshot)
        {
            try
            { plugin.OnUnload(); }
            catch (Exception ex) { _logger.Warn("Plugins", $"Unload error: {ex.Message}"); }
            context.Unload();
        }
    }

    private bool VerifyTrust(string assemblyPath)
    {
        var manifestPath = Path.Combine(Path.GetDirectoryName(assemblyPath)!, "plugin.synapse.json");
        var hasManifest = File.Exists(manifestPath);

        if (_trustMode == PluginTrustMode.RequireManifest && !hasManifest)
        {
            _logger.Error("Plugins",
                $"Refusing {Path.GetFileName(assemblyPath)}: missing plugin.synapse.json (RequireManifest).");
            return false;
        }

        if (!hasManifest)
        {
            _logger.Warn("Plugins",
                $"Loading {Path.GetFileName(assemblyPath)} without manifest — ALC is not a security sandbox.");
            return true;
        }

        try
        {
            var json = File.ReadAllText(manifestPath);
            var manifest = JsonSerializer.Deserialize(json, PluginManifestJsonContext.Default.PluginManifest)
                           ?? throw new InvalidDataException("Invalid plugin manifest.");

            if (string.IsNullOrWhiteSpace(manifest.Sha256))
                throw new InvalidDataException("plugin.synapse.json must include Sha256.");

            var actual = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(assemblyPath)));
            if (!string.Equals(actual, manifest.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                _logger.Error("Plugins",
                    $"SHA-256 mismatch for {Path.GetFileName(assemblyPath)} (manifest invalid).");
                return false;
            }

            if (!string.IsNullOrWhiteSpace(manifest.AssemblyFile) &&
                !string.Equals(Path.GetFileName(assemblyPath), manifest.AssemblyFile, StringComparison.OrdinalIgnoreCase))
            {
                _logger.Error("Plugins", "Manifest AssemblyFile does not match DLL name.");
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.Error("Plugins", $"Manifest verification failed: {ex.Message}");
            return _trustMode != PluginTrustMode.RequireManifest;
        }
    }

    private static PluginTrustMode ResolveTrustMode(PluginTrustMode requested)
    {
        var env = Environment.GetEnvironmentVariable("SYNAPSE_PLUGIN_TRUST");
        if (string.Equals(env, "require-manifest", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(env, "strict", StringComparison.OrdinalIgnoreCase))
            return PluginTrustMode.RequireManifest;
        if (string.Equals(env, "permissive", StringComparison.OrdinalIgnoreCase))
            return PluginTrustMode.Permissive;
        return requested;
    }
}

internal sealed class PluginLoadContext : AssemblyLoadContext
{
    private readonly AssemblyDependencyResolver _resolver;

    public PluginLoadContext(string pluginPath) : base(isCollectible: true)
    {
        _resolver = new AssemblyDependencyResolver(pluginPath);
    }

    protected override Assembly? Load(AssemblyName assemblyName)
    {
        var path = _resolver.ResolveAssemblyToPath(assemblyName);
        return path != null ? LoadFromAssemblyPath(path) : null;
    }
}

[JsonSerializable(typeof(PluginManifest))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class PluginManifestJsonContext : JsonSerializerContext;
