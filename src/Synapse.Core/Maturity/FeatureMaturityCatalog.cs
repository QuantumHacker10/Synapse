using System;
using System.Collections.Generic;

namespace Synapse.Core.Maturity;

/// <summary>Maturity tier for a product surface. See <c>docs/MATURITY.md</c>.</summary>
public enum MaturityTier
{
    /// <summary>Validated on the official target (Windows x64 + Vulkan GPU) for local use.</summary>
    Supported = 0,

    /// <summary>Usable with gaps; expect churn and incomplete validation.</summary>
    EarlyAccess = 1,

    /// <summary>Scaffold / stub / localhost-only. Do not ship as production capability.</summary>
    Experimental = 2
}

/// <summary>One row in the public maturity matrix.</summary>
public sealed class FeatureMaturityEntry
{
    public FeatureMaturityEntry(string id, string name, MaturityTier tier, string notes)
    {
        Id = id ?? throw new ArgumentNullException(nameof(id));
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Tier = tier;
        Notes = notes ?? throw new ArgumentNullException(nameof(notes));
    }

    public string Id { get; }
    public string Name { get; }
    public MaturityTier Tier { get; }
    public string Notes { get; }
}

/// <summary>
/// Canonical list of what Synapse OMNIA claims as Supported vs Experimental.
/// Keep this in sync with <c>docs/MATURITY.md</c> and README maturity callouts.
/// </summary>
public static class FeatureMaturityCatalog
{
    public static IReadOnlyList<FeatureMaturityEntry> All { get; } =
    [
        new("Studio.Editor", "Synapse Studio (édition / inspecteur / LLM console)", MaturityTier.EarlyAccess,
            "UI Avalonia utilisable; viewport Vulkan officiel sur Windows x64 + GPU réel."),
        new("Runtime.Local", "Runtime local (EngineHost, lois, multiphysique, scènes .synapse)", MaturityTier.EarlyAccess,
            "Boucle headless et Studio local; durcissement et couverture encore incomplets."),
        new("Runtime.Benchmarks", "Benchmarks headless reproductibles", MaturityTier.EarlyAccess,
            "Suite JSON + SYNAPSE_SEED; utile pour régression, pas une certification industrielle."),
        new("IO.GlTF", "Export scène glTF / import mesh", MaturityTier.EarlyAccess,
            "Export entités + métadonnées; imports FBX ASCII / USDA limités."),
        new("Plugins.CSharp", "API plugins C# (ALC isolé)", MaturityTier.EarlyAccess,
            "AssemblyLoadContext collectible — pas un sandbox sécurité. Utiliser SYNAPSE_PLUGIN_TRUST=require-manifest."),
        new("Network.P2P", "P2P multi-pairs", MaturityTier.Experimental,
            "TCP localhost / labo avec framing exact; pas un réseau collaboratif production."),
        new("Network.WAN", "P2P WAN (NAT + AES-GCM)", MaturityTier.EarlyAccess,
            "Intégré à EngineHost/Studio : STUN + rendez-vous + hole-punch + patches scène ; NAT symétrique peut encore nécessiter un relay."),
        new("VR.OpenXR", "OpenXR / swapchain Vulkan", MaturityTier.EarlyAccess,
            "Intégré à EngineHost/FrameOrchestrator/Studio ; Silk.NET natif + SYNAPSE_VR_SIMULATE=1."),
        new("Web.Editor", "Éditeur web WASM / WebGPU", MaturityTier.EarlyAccess,
            "Synapse.Web.Studio + ExportWebStudioAsync / menu Studio ; scene.synapse.json partagé."),
    ];

    public static IEnumerable<FeatureMaturityEntry> OfTier(MaturityTier tier)
    {
        foreach (var entry in All)
        {
            if (entry.Tier == tier)
                yield return entry;
        }
    }

    public static FeatureMaturityEntry? Find(string id)
    {
        foreach (var entry in All)
        {
            if (string.Equals(entry.Id, id, StringComparison.Ordinal))
                return entry;
        }

        return null;
    }
}
