using Synapse.Runtime;

namespace Synapse.Plugins;

/// <summary>Contract for Synapse extensions loaded in an isolated AssemblyLoadContext (v2 plugin API).
/// Isolation is not a security sandbox — plugins share host privileges.</summary>
public interface ISynapsePlugin
{
    PluginMetadata Metadata { get; }

    void OnLoad(PluginHostContext context);

    void OnUnload();
}

public sealed class PluginMetadata
{
    public required string Id { get; init; }
    public required string Name { get; init; }
    public required string Version { get; init; }
    public string Author { get; init; } = "";
    public string Description { get; init; } = "";
}

public sealed class PluginHostContext
{
    public required EngineHost Host { get; init; }
    public required string PluginDirectory { get; init; }
}
