using System.Reflection;

namespace Synapse.Infrastructure;

/// <summary>Product name and version (from <c>Directory.Build.props</c> at build time).</summary>
public static class SynapseProduct
{
    public const string Name = "SYNAPSE OMNIA";

    public static string Version =>
        typeof(SynapseProduct).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? "1.0.0";

    public const string LicenseTierUrl = "https://synapse-omnia.com/pricing";
    public const string SupportEmail = "support@synapse-omnia.com";
}
