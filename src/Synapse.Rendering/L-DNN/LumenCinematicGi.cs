// =============================================================================
// L-DNN — Cinematic GI: GPU-resident surface cache + full path-trace mode (AAA)
// =============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using GDNN.Lighting.LDNN;

namespace GDNN.Lighting.LDNN;

/// <summary>
/// Cinematic Lumen Neural path: larger GPU-staged surface cache and full-frame
/// path-trace refine for offline / Cinematic quality.
/// </summary>
public sealed class LumenCinematicGi
{
    public enum Mode : byte
    {
        Hybrid = 0,
        GpuSurfaceCache = 1,
        FullPathTrace = 2
    }

    private readonly LumenNeural30.SurfaceRadianceCache _cache;
    private readonly ReferencePathTracer _pathTracer = new();
    private bool _pathTracerReady;
    private int _width;
    private int _height;

    public LumenCinematicGi(int cacheResolution = 128)
    {
        _cache = new LumenNeural30.SurfaceRadianceCache(
            resolution: Math.Clamp(cacheResolution, 32, 256),
            origin: new Vector3(-24f, -4f, -24f),
            cellSize: 0.28f);
    }

    public LumenNeural30.SurfaceRadianceCache Cache => _cache;
    public Mode LastMode { get; private set; } = Mode.Hybrid;
    public int LastPathTraceSamples { get; private set; }
    public bool UsedRealPathTracer { get; private set; }

    public void EnsurePathTracer(int width, int height)
    {
        if (_pathTracerReady && _width == width && _height == height)
            return;
        _width = Math.Max(1, width);
        _height = Math.Max(1, height);
        _pathTracer.Initialize(_width, _height);
        _pathTracerReady = true;
    }

    /// <summary>
    /// Refines Hybrid irradiance with surface-cache multi-bounce; optionally
    /// runs a dense path-trace pass for Cinematic/FullPathTrace via
    /// <see cref="ReferencePathTracer.EstimateRadiance"/>.
    /// </summary>
    public Vector3[,] Refine(
        Vector3[,] hybridIrradiance,
        GBuffer gbuffer,
        CameraState camera,
        List<LightConfig> lights,
        Mode mode,
        int pathTraceSpp = 4,
        LumenNeural30.Policy? policyOverride = null)
    {
        ArgumentNullException.ThrowIfNull(hybridIrradiance);
        ArgumentNullException.ThrowIfNull(gbuffer);
        ArgumentNullException.ThrowIfNull(camera);

        int w = hybridIrradiance.GetLength(0);
        int h = hybridIrradiance.GetLength(1);
        var result = new Vector3[w, h];
        LastMode = mode;
        UsedRealPathTracer = false;

        // Dense accumulation into GPU-staged surface cache from hybrid.
        int step = Math.Max(1, Math.Max(w, h) / (mode == Mode.FullPathTrace ? 96 : 64));
        for (int y = 0; y < h; y += step)
        {
            for (int x = 0; x < w; x += step)
            {
                int idx = y * gbuffer.Width + x;
                if (idx < 0 || idx >= gbuffer.Depth.Length || gbuffer.Depth[idx] <= 0f)
                    continue;
                Vector3 world = camera.Position + camera.Forward * gbuffer.Depth[idx];
                _cache.Accumulate(world, hybridIrradiance[x, y], weight: 1.25f);
            }
        }

        var policy = policyOverride ?? new LumenNeural30.Policy
        {
            MaxBounces = mode == Mode.FullPathTrace ? 8 : 6,
            BounceFalloff = 0.55f,
            SpecularWeight = 0.50f,
            DiffuseWeight = 0.93f,
            CascadeDominance = 1.32f,
            AmbientFloor = 0.01f
        };

        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int idx = y * gbuffer.Width + x;
                Vector3 albedo = idx < gbuffer.Albedo.Length ? gbuffer.Albedo[idx] : Vector3.One * 0.5f;
                float depth = idx < gbuffer.Depth.Length ? gbuffer.Depth[idx] : 1f;
                Vector3 world = camera.Position + camera.Forward * MathF.Max(0.5f, depth);
                Vector3 cache = _cache.SampleFiltered(world);
                // Cascade proxy: slightly lifted hybrid (real cascade field mixed upstream).
                Vector3 cascade = hybridIrradiance[x, y] * 1.18f + cache * 0.15f;
                Vector3 specularHint = idx < gbuffer.Specular.Length && gbuffer.Specular[idx].LengthSquared() > 1e-6f
                    ? gbuffer.Specular[idx]
                    : Vector3.One * 0.22f;
                result[x, y] = LumenNeural30.MultiBounceRefine(
                    hybridIrradiance[x, y],
                    cascade,
                    cache,
                    albedo,
                    specularHint,
                    policy);
            }
        });

        if (mode == Mode.FullPathTrace)
        {
            EnsurePathTracer(w, h);
            int spp = Math.Clamp(pathTraceSpp, 1, 16);
            int maxDepth = Math.Max(4, policy.MaxBounces);
            // Dense cinematic stride (≈ full-res / 128 on the short axis).
            int stride = Math.Max(1, Math.Min(w, h) / 128);
            int collected = 0;
            var lightList = lights ?? new List<LightConfig>();

            for (int y = 0; y < h; y += stride)
            {
                for (int x = 0; x < w; x += stride)
                {
                    int idx = y * gbuffer.Width + x;
                    if (idx >= gbuffer.Depth.Length || gbuffer.Depth[idx] <= 0f)
                        continue;

                    var rng = new RandomNumberGenerator((uint)(x * 73856093 ^ y * 19349663 ^ spp * 83492791));
                    Vector3 gt = Vector3.Zero;
                    for (int s = 0; s < spp; s++)
                    {
                        float jitterX = rng.NextFloat() - 0.5f;
                        float jitterY = rng.NextFloat() - 0.5f;
                        float u = (x + 0.5f + jitterX) / w;
                        float v = (y + 0.5f + jitterY) / h;
                        RayPayload ray = _pathTracer.GenerateCameraRay(u, v, camera, ref rng);
                        gt += _pathTracer.EstimateRadiance(ray, gbuffer, lightList, maxDepth, ref rng);
                    }

                    gt /= spp;
                    UsedRealPathTracer = true;
                    // Blend path-traced ground truth into hybrid multi-bounce (AAA offline refine).
                    result[x, y] = Vector3.Lerp(result[x, y], gt * 0.72f + result[x, y] * 0.28f, 0.62f);
                    var sample = gbuffer.GetSample(x, y);
                    _cache.Accumulate(camera.Position + camera.Forward * sample.Depth, result[x, y], 2.2f);

                    // Fill immediate neighborhood so sparse PT doesn't leave block artifacts.
                    int x1 = Math.Min(w, x + stride);
                    int y1 = Math.Min(h, y + stride);
                    for (int yy = y; yy < y1; yy++)
                    {
                        for (int xx = x; xx < x1; xx++)
                        {
                            if (xx == x && yy == y)
                                continue;
                            result[xx, yy] = Vector3.Lerp(result[xx, yy], result[x, y], 0.35f);
                        }
                    }

                    collected++;
                }
            }

            LastPathTraceSamples = collected * spp;
        }
        else
        {
            LastPathTraceSamples = 0;
        }

        _cache.TemporalDecay(mode == Mode.FullPathTrace ? 0.994f : 0.985f);
        return result;
    }
}
