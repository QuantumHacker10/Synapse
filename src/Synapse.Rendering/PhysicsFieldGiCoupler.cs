// Physics → Rendering coupler: living-law fields paint L-DNN GI / fog / emissive.

using System;
using System.Numerics;
using GDNN.Lighting.LDNN;

namespace GDNN.Rendering;

/// <summary>
/// Samples a physics temperature/density field into screen-aligned heat maps and
/// applies Lumen Neural 3.0 thermo-optical coupling into fog + irradiance.
/// </summary>
public sealed class PhysicsFieldGiCoupler
{
    private float[] _heatMap = Array.Empty<float>();
    private int _mapW;
    private int _mapH;

    public float LastAverageTemperature { get; private set; } = 300f;
    public float LastAverageDensity { get; private set; } = 1f;
    public float LastTemperatureVariance { get; private set; }
    public float LastFogDensityScale { get; private set; } = 1f;
    public Vector3 LastEmissiveTint { get; private set; }

    /// <summary>
    /// Rebuilds a coarse 2D heat map from a 3D field slab (Y mid-plane projection).
    /// temperature/density are flat arrays in x + sx*(y + sy*z) layout.
    /// </summary>
    public void IngestField(
        ReadOnlySpan<float> temperature,
        ReadOnlySpan<float> density,
        int sx,
        int sy,
        int sz,
        int mapWidth = 48,
        int mapHeight = 48)
    {
        if (sx <= 0 || sy <= 0 || sz <= 0 || temperature.Length < sx * sy * sz)
            return;

        _mapW = Math.Clamp(mapWidth, 8, 128);
        _mapH = Math.Clamp(mapHeight, 8, 128);
        if (_heatMap.Length != _mapW * _mapH)
            _heatMap = new float[_mapW * _mapH];

        int midY = sy / 2;
        double sumT = 0, sumT2 = 0, sumD = 0;
        int n = sx * sy * sz;

        for (int i = 0; i < n; i++)
        {
            float t = temperature[i];
            sumT += t;
            sumT2 += t * t;
        }

        if (density.Length >= n)
        {
            for (int i = 0; i < n; i++)
                sumD += density[i];
        }
        else
        {
            sumD = n;
        }

        LastAverageTemperature = (float)(sumT / n);
        LastAverageDensity = (float)(sumD / n);
        float mean = LastAverageTemperature;
        LastTemperatureVariance = (float)Math.Max(0, sumT2 / n - mean * mean);

        for (int my = 0; my < _mapH; my++)
        {
            int iz = Math.Clamp(my * sz / _mapH, 0, sz - 1);
            for (int mx = 0; mx < _mapW; mx++)
            {
                int ix = Math.Clamp(mx * sx / _mapW, 0, sx - 1);
                int idx = ix + sx * (midY + sy * iz);
                _heatMap[my * _mapW + mx] = temperature[idx];
            }
        }

        var (fogScale, emissive, _) = LumenNeural30.CouplePhysicsFields(
            LastAverageTemperature,
            LastAverageDensity,
            LastTemperatureVariance);
        LastFogDensityScale = fogScale;
        LastEmissiveTint = emissive;
    }

    public VolumeFogConfig ApplyToFog(VolumeFogConfig fog) =>
        LumenNeural30.ApplyPhysicsToFog(
            fog,
            LastAverageTemperature,
            LastAverageDensity,
            LastTemperatureVariance);

    public void BoostIrradiance(Vector3[,] irradiance)
    {
        if (_heatMap.Length == 0 || irradiance == null)
            return;
        LumenNeural30.BoostIrradianceFromHeatMap(irradiance, _heatMap, _mapW, _mapH);
    }

    public ReadOnlySpan<float> HeatMap => _heatMap;
    public int MapWidth => _mapW;
    public int MapHeight => _mapH;
}
