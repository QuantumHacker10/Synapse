using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Loader;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;

namespace Synapse.Plugins;

/// <summary>Loads and manages Synapse plugins in isolated <see cref="AssemblyLoadContext"/> instances.</summary>
public sealed class PluginHost : IDisposable
{
    private readonly ISynapseLogger _logger;
    private readonly List<(ISynapsePlugin Plugin, PluginLoadContext Context)> _loaded = new();

    public PluginHost(ISynapseLogger logger) => _logger = logger;

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
            if (LoadPluginAssembly(dll, host))
                count++;
        }

        _logger.Info("Plugins", $"Loaded {count} plugin(s) from {directory}");
        return count;
    }

    public bool LoadPluginAssembly(string assemblyPath, EngineHost host)
    {
        try
        {
            var full = Path.GetFullPath(assemblyPath);
            var context = new PluginLoadContext(full);
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
            context.Unload();
        }
        _loaded.Clear();
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
