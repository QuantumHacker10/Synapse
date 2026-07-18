using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using GDNN.Lighting.LDNN;
using GDNN.Scene;

namespace GDNN.Rendering;

/// <summary>
/// Applies parsed LLM scene hints to L-DNN lighting and fog configuration.
/// </summary>
public static class LlmSceneApplicator
{
    /// <summary>
    /// Converts parsed lighting parameters into runtime light configs.
    /// </summary>
    public static List<LightConfig> ApplyLighting(
        LightingParams parameters,
        IList<LightConfig>? existingLights = null)
    {
        var lights = existingLights != null
            ? new List<LightConfig>(existingLights)
            : new List<LightConfig>();

        if (!parameters.DirectionalDirection.HasValue &&
            string.IsNullOrWhiteSpace(parameters.Color) &&
            !parameters.Intensity.HasValue)
        {
            return lights;
        }

        var direction = parameters.DirectionalDirection.HasValue
            ? Normalize(new Vector3(
                parameters.DirectionalDirection.Value.X,
                parameters.DirectionalDirection.Value.Y,
                parameters.DirectionalDirection.Value.Z))
            : new Vector3(0.3f, -0.8f, 0.5f);

        var color = ParseColor(parameters.Color) ?? Vector3.One;
        var intensity = Math.Max(0f, parameters.Intensity ?? 1f);

        lights.Add(new LightConfig
        {
            Type = LightType.Directional,
            Direction = direction,
            Color = color,
            Intensity = intensity,
            ShadowMethod = ShadowMethod.None,
            Importance = 1f
        });

        return lights;
    }

    /// <summary>
    /// Applies fog and cloud hints to a volumetric fog configuration.
    /// </summary>
    public static VolumeFogConfig ApplyFog(LightingParams parameters, VolumeFogConfig? baseConfig = null)
    {
        var config = baseConfig ?? new VolumeFogConfig
        {
            MaxDensity = 0.02f,
            HeightFalloff = 0.02f,
            ReferenceHeight = 0f,
            FogColor = new Vector3(0.7f, 0.75f, 0.85f),
            EnableClouds = false
        };

        if (parameters.FogDensity.HasValue)
            config = config with { MaxDensity = Math.Max(0f, parameters.FogDensity.Value) };

        if (parameters.EnableClouds.HasValue)
            config = config with { EnableClouds = parameters.EnableClouds.Value };

        return config;
    }

    private static Vector3 Normalize(Vector3 vector)
    {
        if (vector.LengthSquared() <= float.Epsilon)
            return new Vector3(0f, -1f, 0f);

        return Vector3.Normalize(vector);
    }

    private static Vector3? ParseColor(string? colorText)
    {
        if (string.IsNullOrWhiteSpace(colorText))
            return null;

        var hex = colorText.Trim().TrimStart('#');
        if (hex.Length != 6 ||
            !int.TryParse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var rgb))
        {
            return null;
        }

        return new Vector3(
            ((rgb >> 16) & 0xFF) / 255f,
            ((rgb >> 8) & 0xFF) / 255f,
            (rgb & 0xFF) / 255f);
    }
}
