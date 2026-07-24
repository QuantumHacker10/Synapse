// Unified LLM → Physics → Rendering → Simulation world-delta DTOs.

using System.Collections.Generic;

namespace GDNN.Scene;

/// <summary>
/// Living-law hint extracted from an LLM reply (expression and/or library id).
/// </summary>
public record LivingLawHint
{
    /// <summary>Stable law id (library key or generated slug).</summary>
    public string LawId { get; init; } = "";

    /// <summary>Human-readable name.</summary>
    public string? Name { get; init; }

    /// <summary>Living-law expression (compiler source).</summary>
    public string? Expression { get; init; }

    /// <summary>Optional scalar parameters (alpha, k, etc.).</summary>
    public IReadOnlyDictionary<string, float>? Parameters { get; init; }

    /// <summary>Optional continuum module flags: sph, elasticity, rigid, laws.</summary>
    public string[]? EnableModules { get; init; }
}

/// <summary>
/// Material / substrate hint for PBR surfaces driven by LLM.
/// </summary>
public record MaterialHint
{
    public string? Name { get; init; }
    public float? Metallic { get; init; }
    public float? Roughness { get; init; }
    public string? BaseColor { get; init; }
    public float? EmissiveStrength { get; init; }
}

/// <summary>
/// Agent / simulation impulse hint (Simulation → Physics coupling seed).
/// </summary>
public record SimulationImpulseHint
{
    public string? Profile { get; init; }
    public (float X, float Y, float Z)? Position { get; init; }
    public (float X, float Y, float Z)? Impulse { get; init; }
    public float? HeatDeposit { get; init; }
    public string? BehaviorTreeName { get; init; }
}

/// <summary>
/// Full industrial world delta: one LLM reply can drive the entire cascade
/// LLM → Physics → Rendering → Simulation.
/// </summary>
public record LlmWorldDelta
{
    public LightingParams? Lighting { get; init; }
    public SdfHint? Sdf { get; init; }
    public LivingLawHint? Law { get; init; }
    public MaterialHint? Material { get; init; }
    public SimulationImpulseHint? Impulse { get; init; }
    public IReadOnlyList<BehaviorTreeNodeHint>? BehaviorNodes { get; init; }
    public bool HasAny =>
        Lighting != null ||
        Sdf != null ||
        Law != null ||
        Material != null ||
        Impulse != null ||
        (BehaviorNodes is { Count: > 0 });
}

/// <summary>Minimal BT node hint for world-delta payloads (avoids circular refs).</summary>
public record BehaviorTreeNodeHint
{
    public string Type { get; init; } = "action";
    public string? Name { get; init; }
    public string? Action { get; init; }
    public string? Condition { get; init; }
}
