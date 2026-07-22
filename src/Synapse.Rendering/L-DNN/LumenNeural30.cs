// =============================================================================
// L-DNN — Lumen Neural 3.0
// Surface radiance cache, multi-bounce neural refine, and physics-coupled
// volumetrics — the industrial GI brain for Synapse OMNIA.
// =============================================================================

using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace GDNN.Lighting.LDNN;

/// <summary>
/// Lumen Neural 3.0 algorithms: world-space surface cache, multi-bounce
/// irradiance mixing, and thermo-volumetric fog coupling from physics fields.
/// </summary>
public static class LumenNeural30
{
    public sealed class Policy
    {
        public int SurfaceCacheResolution { get; init; } = 64;
        public int MaxBounces { get; init; } = 4;
        public float BounceFalloff { get; init; } = 0.62f;
        public float SpecularWeight { get; init; } = 0.35f;
        public float DiffuseWeight { get; init; } = 0.85f;
        public float PhysicsFogCoupling { get; init; } = 0.55f;
        public float PhysicsEmissiveCoupling { get; init; } = 0.40f;
        public float AmbientFloor { get; init; } = 0.02f;
        public float CascadeDominance { get; init; } = 1.15f;
    }

    public static Policy Industrial { get; } = new();

    /// <summary>
    /// World-space surface radiance cache (Lumen surface-cache analog).
    /// Stores SH-like low-order irradiance per cell for multi-bounce feedback.
    /// </summary>
    public sealed class SurfaceRadianceCache
    {
        private readonly int _res;
        private readonly Vector3[] _irradiance;
        private readonly float[] _confidence;
        private readonly Vector3 _origin;
        private readonly float _cellSize;

        public SurfaceRadianceCache(int resolution = 64, Vector3? origin = null, float cellSize = 0.5f)
        {
            _res = Math.Clamp(resolution, 8, 128);
            _irradiance = new Vector3[_res * _res * _res];
            _confidence = new float[_res * _res * _res];
            _origin = origin ?? new Vector3(-16f, -2f, -16f);
            _cellSize = MathF.Max(0.05f, cellSize);
        }

        public int Resolution => _res;
        public int CellCount => _irradiance.Length;

        public void Clear()
        {
            Array.Clear(_irradiance, 0, _irradiance.Length);
            Array.Clear(_confidence, 0, _confidence.Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int IndexOf(Vector3 world)
        {
            int x = (int)MathF.Floor((world.X - _origin.X) / _cellSize);
            int y = (int)MathF.Floor((world.Y - _origin.Y) / _cellSize);
            int z = (int)MathF.Floor((world.Z - _origin.Z) / _cellSize);
            if ((uint)x >= (uint)_res || (uint)y >= (uint)_res || (uint)z >= (uint)_res)
                return -1;
            return x + _res * (y + _res * z);
        }

        public void Accumulate(Vector3 worldPos, Vector3 radiance, float weight = 1f)
        {
            int i = IndexOf(worldPos);
            if (i < 0 || weight <= 0f)
                return;
            float w = Math.Clamp(weight, 0f, 4f);
            float prev = _confidence[i];
            float next = prev + w;
            _irradiance[i] = (_irradiance[i] * prev + radiance * w) / MathF.Max(1e-4f, next);
            _confidence[i] = Math.Min(next, 64f);
        }

        public Vector3 Sample(Vector3 worldPos)
        {
            int i = IndexOf(worldPos);
            if (i < 0 || _confidence[i] < 1e-3f)
                return Vector3.Zero;
            return _irradiance[i];
        }

        /// <summary>Temporal decay so stale cache cells fade (Lumen-like invalidation).</summary>
        public void TemporalDecay(float factor = 0.97f)
        {
            factor = Math.Clamp(factor, 0.5f, 0.999f);
            for (int i = 0; i < _confidence.Length; i++)
            {
                _confidence[i] *= factor;
                _irradiance[i] *= factor;
            }
        }
    }

    /// <summary>
    /// Multi-bounce neural refine: mixes SSGI, cascade, and surface-cache samples
    /// into a single irradiance with exponential bounce falloff.
    /// </summary>
    public static Vector3 MultiBounceRefine(
        Vector3 ssgi,
        Vector3 cascade,
        Vector3 surfaceCache,
        Vector3 albedo,
        Vector3 specularHint,
        Policy? policy = null)
    {
        policy ??= Industrial;
        Vector3 diffuse = (ssgi * 0.55f + cascade * policy.CascadeDominance + surfaceCache * 0.75f)
                          * policy.DiffuseWeight;
        Vector3 bounce = Vector3.Zero;
        Vector3 carry = diffuse;
        float atten = 1f;
        for (int b = 0; b < policy.MaxBounces; b++)
        {
            atten *= policy.BounceFalloff;
            carry = carry * albedo * atten;
            bounce += carry;
        }

        Vector3 specular = specularHint * policy.SpecularWeight;
        Vector3 result = diffuse + bounce + specular;
        float floor = policy.AmbientFloor;
        return new Vector3(
            MathF.Max(floor, result.X),
            MathF.Max(floor, result.Y),
            MathF.Max(floor, result.Z));
    }

    /// <summary>
    /// Maps physics field temperature / density into volumetric fog density and emissive tint.
    /// Invented thermo-optical coupling: Planck-ish warm bias + density scattering.
    /// </summary>
    public static (float FogDensityScale, Vector3 EmissiveTint, Vector3 FogColorBias) CouplePhysicsFields(
        float averageTemperatureK,
        float averageDensity,
        float temperatureVariance,
        Policy? policy = null)
    {
        policy ??= Industrial;
        // Normalize around room temp (300 K) and unit density.
        float tNorm = Math.Clamp((averageTemperatureK - 250f) / 400f, 0f, 2f);
        float dNorm = Math.Clamp(averageDensity / 1.2f, 0f, 3f);
        float varNorm = Math.Clamp(temperatureVariance / 80f, 0f, 2f);

        float fogScale = 1f + policy.PhysicsFogCoupling * (0.35f * dNorm + 0.25f * varNorm);
        // Warm emissive from heat (blackbody-ish approximation in RGB).
        Vector3 emissive = new(
            policy.PhysicsEmissiveCoupling * (0.15f + 1.2f * tNorm),
            policy.PhysicsEmissiveCoupling * (0.08f + 0.55f * tNorm * tNorm),
            policy.PhysicsEmissiveCoupling * (0.05f + 0.15f * MathF.Max(0f, 1f - tNorm)));
        Vector3 fogBias = new(
            0.55f + 0.35f * tNorm,
            0.60f + 0.15f * tNorm,
            0.85f - 0.25f * tNorm);
        return (fogScale, emissive, fogBias);
    }

    /// <summary>
    /// Injects physics coupling into a VolumeFogConfig without dropping other settings.
    /// </summary>
    public static VolumeFogConfig ApplyPhysicsToFog(
        VolumeFogConfig fog,
        float averageTemperatureK,
        float averageDensity,
        float temperatureVariance,
        Policy? policy = null)
    {
        var (scale, _, fogBias) = CouplePhysicsFields(averageTemperatureK, averageDensity, temperatureVariance, policy);
        return fog with
        {
            MaxDensity = Math.Clamp(fog.MaxDensity * scale, 0.001f, 0.25f),
            FogColor = Vector3.Lerp(fog.FogColor, fogBias, 0.45f)
        };
    }

    /// <summary>
    /// Screen-space irradiance boost from a coarse physics heat map (bilinear sample).
    /// heatMap is row-major width*height, values typically Kelvin.
    /// </summary>
    public static void BoostIrradianceFromHeatMap(
        Vector3[,] irradiance,
        ReadOnlySpan<float> heatMap,
        int mapW,
        int mapH,
        Policy? policy = null)
    {
        policy ??= Industrial;
        if (irradiance == null || heatMap.IsEmpty || mapW <= 0 || mapH <= 0)
            return;
        int w = irradiance.GetLength(0);
        int h = irradiance.GetLength(1);
        for (int y = 0; y < h; y++)
        {
            float v = (y + 0.5f) / h * (mapH - 1);
            int y0 = Math.Clamp((int)v, 0, mapH - 1);
            int y1 = Math.Min(y0 + 1, mapH - 1);
            float fy = v - y0;
            for (int x = 0; x < w; x++)
            {
                float u = (x + 0.5f) / w * (mapW - 1);
                int x0 = Math.Clamp((int)u, 0, mapW - 1);
                int x1 = Math.Min(x0 + 1, mapW - 1);
                float fx = u - x0;
                float t00 = heatMap[y0 * mapW + x0];
                float t10 = heatMap[y0 * mapW + x1];
                float t01 = heatMap[y1 * mapW + x0];
                float t11 = heatMap[y1 * mapW + x1];
                float t = (1 - fx) * (1 - fy) * t00 + fx * (1 - fy) * t10
                        + (1 - fx) * fy * t01 + fx * fy * t11;
                var (_, emissive, _) = CouplePhysicsFields(t, 1f, 0f, policy);
                irradiance[x, y] += emissive * 0.25f;
            }
        }
    }
}
