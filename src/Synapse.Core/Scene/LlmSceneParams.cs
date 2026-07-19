// LLM scene hint DTOs (LightingParams, SdfHint) used by HybridLlmRouter, EngineHost, and LlmSceneApplicator.

namespace GDNN.Scene;

/// <summary>
/// Lighting parameters extracted from LLM scene or material responses.
/// </summary>
public record LightingParams
{
    /// <summary>Normalized directional light direction vector.</summary>
    public (float X, float Y, float Z)? DirectionalDirection { get; init; }

    /// <summary>Light color as hex string (e.g. #FFEECC).</summary>
    public string? Color { get; init; }

    /// <summary>Directional light intensity multiplier.</summary>
    public float? Intensity { get; init; }

    /// <summary>Volumetric fog density override.</summary>
    public float? FogDensity { get; init; }

    /// <summary>Whether procedural clouds are enabled in volumetric fog.</summary>
    public bool? EnableClouds { get; init; }
}

/// <summary>
/// Simple SDF primitive hint parsed from LLM output.
/// </summary>
public record SdfHint
{
    /// <summary>Primitive type (sphere, box).</summary>
    public string Primitive { get; init; } = "sphere";

    /// <summary>Primitive center in world space.</summary>
    public (float X, float Y, float Z)? Center { get; init; }

    /// <summary>Sphere radius or box half-extent scalar.</summary>
    public float? Radius { get; init; }

    /// <summary>Box half-extents when Primitive is box.</summary>
    public (float X, float Y, float Z)? Size { get; init; }
}
