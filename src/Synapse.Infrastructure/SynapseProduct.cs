using System.Reflection;

namespace Synapse.Infrastructure;

/// <summary>Product name and version (from <c>Directory.Build.props</c> at build time).</summary>
public static class SynapseProduct
{
    public const string Name = "SYNAPSE OMNIA";

    public static string Version =>
        typeof(SynapseProduct).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? "2.0.0";
}
