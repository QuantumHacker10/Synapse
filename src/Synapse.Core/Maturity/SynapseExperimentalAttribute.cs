using System;

namespace Synapse.Core.Maturity;

/// <summary>
/// Marks a type or member as experimental: scaffolding, localhost-only, or synthetic.
/// Does not emit compiler warnings (unlike <see cref="System.Diagnostics.CodeAnalysis.ExperimentalAttribute"/>).
/// See <c>docs/MATURITY.md</c>.
/// </summary>
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct | AttributeTargets.Interface | AttributeTargets.Method | AttributeTargets.Property, Inherited = false, AllowMultiple = false)]
public sealed class SynapseExperimentalAttribute : Attribute
{
    public SynapseExperimentalAttribute(string featureId, string reason)
    {
        FeatureId = featureId ?? throw new ArgumentNullException(nameof(featureId));
        Reason = reason ?? throw new ArgumentNullException(nameof(reason));
    }

    /// <summary>Stable id, e.g. <c>VR.OpenXR</c> or <c>Network.WAN</c>.</summary>
    public string FeatureId { get; }

    /// <summary>Why this surface is not production-supported yet.</summary>
    public string Reason { get; }
}
