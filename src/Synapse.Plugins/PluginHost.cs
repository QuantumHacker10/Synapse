using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;

namespace Synapse.Plugins;

/// <summary>Loads and manages Synapse plugins in isolated <see cref="AssemblyLoadContext"/> instances with path jail + optional hash allowlist.</summary>
public sealed class PluginHost : IDisposable
{
    private readonly ISynapseLogger _logger;
    private readonly List<(ISynapsePlugin Plugin, PluginLoadContext Context)> _loaded = new();
    private string? _trustedRoot;
    private HashSet<string>? _allowHashes;

    public PluginHost(ISynapseLogger logger) => _logger = logger;

    public IReadOnlyList<PluginMetadata> LoadedPlugins =>
        _loaded.ConvertAll(p => p.Plugin.Metadata);

    /// <summary>
    /// Loads plugins from a directory. Paths outside the directory are rejected.
    /// Optional <c>plugins.allow</c> (SHA-256 hex lines) restricts which DLLs may load.
    /// </summary>
    public int LoadFromDirectory(string directory, EngineHost host)
    {
        if (!Directory.Exists(directory))
        {
            _logger.Warn("Plugins", $"Plugin directory not found: {directory}");
            return 0;
        }

        _trustedRoot = Path.GetFullPath(directory);
        _allowHashes = TryLoadAllowlist(Path.Combine(_trustedRoot, "plugins.allow"));

        int count = 0;
        foreach (var dll in Directory.EnumerateFiles(_trustedRoot, "*.dll", SearchOption.TopDirectoryOnly))
        {
            if (LoadPluginAssembly(dll, host))
                count++;
        }

        _logger.Info("Plugins", $"Loaded {count} plugin(s) from {_trustedRoot}");
        return count;
    }

    public bool LoadPluginAssembly(string assemblyPath, EngineHost host)
    {
        try
        {
            if (IsBlockedRemotePath(assemblyPath))
            {
                _logger.Warn("Plugins", $"Blocked remote/UNC plugin path: {assemblyPath}");
                return false;
            }

            var full = Path.GetFullPath(assemblyPath);
            if (_trustedRoot != null && !IsUnderRoot(full, _trustedRoot))
            {
                _logger.Warn("Plugins", $"Plugin path escapes trusted root: {full}");
                return false;
            }

            if (_allowHashes is { Count: > 0 })
            {
                var hash = ComputeSha256Hex(full);
                if (!_allowHashes.Contains(hash))
                {
                    _logger.Warn("Plugins", $"DLL not in plugins.allow allowlist: {Path.GetFileName(full)}");
                    return false;
                }
            }

            var context = new PluginLoadContext(full, _trustedRoot ?? Path.GetDirectoryName(full)!);
            var assembly = context.LoadFromAssemblyPath(full);
            var pluginType = assembly.GetExportedTypes()
                .FirstOrDefault(t => typeof(ISynapsePlugin).IsAssignableFrom(t) && t is { IsAbstract: false, IsInterface: false });

            if (pluginType == null)
            {
                _logger.Warn("Plugins", $"No ISynapsePlugin in {Path.GetFileName(full)}");
                context.Unload();
                return false;
            }

            if (Activator.CreateInstance(pluginType) is not ISynapsePlugin plugin)
                return false;

            var pluginDir = Path.GetDirectoryName(full) ?? AppContext.BaseDirectory;
            plugin.OnLoad(new PluginHostContext { Host = host, PluginDirectory = pluginDir });
            _loaded.Add((plugin, context));
            _logger.Info("Plugins", $"Loaded {plugin.Metadata.Name} v{plugin.Metadata.Version}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.Error("Plugins", $"Failed to load {assemblyPath}: {ex.Message}");
            return false;
        }
    }

    public void Dispose()
    {
        foreach (var (plugin, context) in _loaded)
        {
            try
            { plugin.OnUnload(); }
            catch (Exception ex) { _logger.Warn("Plugins", $"Unload error: {ex.Message}"); }

            try
            { context.Unload(); }
            catch (Exception ex) { _logger.Warn("Plugins", $"ALC unload error: {ex.Message}"); }
        }
        _loaded.Clear();
    }

    public static bool IsUnderRoot(string fullPath, string root)
    {
        var normalizedRoot = Path.GetFullPath(root)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            + Path.DirectorySeparatorChar;
        var normalizedPath = Path.GetFullPath(fullPath);
        return normalizedPath.StartsWith(normalizedRoot, StringComparison.OrdinalIgnoreCase)
               || string.Equals(normalizedPath.TrimEnd(Path.DirectorySeparatorChar),
                   normalizedRoot.TrimEnd(Path.DirectorySeparatorChar), StringComparison.OrdinalIgnoreCase);
    }

    public static bool IsBlockedRemotePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return true;
        if (path.StartsWith(@"\\", StringComparison.Ordinal) || path.StartsWith("//", StringComparison.Ordinal))
            return true;
        if (path.Contains("://", StringComparison.OrdinalIgnoreCase))
            return true;
        return false;
    }

    private static HashSet<string>? TryLoadAllowlist(string allowPath)
    {
        if (!File.Exists(allowPath))
            return null;
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var line in File.ReadAllLines(allowPath))
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith('#') || trimmed.StartsWith("//", StringComparison.Ordinal))
                continue;
            set.Add(trimmed.ToLowerInvariant());
        }
        return set;
    }

    private static string ComputeSha256Hex(string filePath)
    {
        var hash = SHA256.HashData(File.ReadAllBytes(filePath));
        var sb = new StringBuilder(hash.Length * 2);
        foreach (var b in hash)
            sb.Append(b.ToString("x2"));
        return sb.ToString();
    }
}

internal sealed class PluginLoadContext : AssemblyLoadContext
{
    private readonly AssemblyDependencyResolver _resolver;
    private readonly string _pluginRoot;

    public PluginLoadContext(string pluginPath, string pluginRoot) : base(isCollectible: true)
    {
        _resolver = new AssemblyDependencyResolver(pluginPath);
        _pluginRoot = Path.GetFullPath(pluginRoot);
    }

    protected override Assembly? Load(AssemblyName assemblyName)
    {
        var path = _resolver.ResolveAssemblyToPath(assemblyName);
        if (path == null)
            return null;
        var full = Path.GetFullPath(path);
        if (!PluginHost.IsUnderRoot(full, _pluginRoot))
            return null; // refuse dependencies outside plugin directory
        return LoadFromAssemblyPath(full);
    }
}
