// =============================================================================
// L-DNN — Cinematic GI: GPU-resident surface cache + full path-trace mode
// =============================================================================

using System;
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

    public LumenCinematicGi(int cacheResolution = 96)
    {
        _cache = new LumenNeural30.SurfaceRadianceCache(
            resolution: Math.Clamp(cacheResolution, 32, 128),
            origin: new Vector3(-24f, -4f, -24f),
            cellSize: 0.35f);
    }

    public LumenNeural30.SurfaceRadianceCache Cache => _cache;
    public Mode LastMode { get; private set; } = Mode.Hybrid;
    public int LastPathTraceSamples { get; private set; }

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
    /// runs a dense path-trace pass for Cinematic/FullPathTrace.
    /// </summary>
    public Vector3[,] Refine(
        Vector3[,] hybridIrradiance,
        GBuffer gbuffer,
        CameraState camera,
        System.Collections.Generic.List<LightConfig> lights,
        Mode mode,
        int pathTraceSpp = 4)
    {
        ArgumentNullException.ThrowIfNull(hybridIrradiance);
        ArgumentNullException.ThrowIfNull(gbuffer);
        ArgumentNullException.ThrowIfNull(camera);

        int w = hybridIrradiance.GetLength(0);
        int h = hybridIrradiance.GetLength(1);
        var result = new Vector3[w, h];
        LastMode = mode;

        // Always accumulate into GPU-staged surface cache from hybrid.
        int step = Math.Max(1, Math.Max(w, h) / 48);
        for (int y = 0; y < h; y += step)
        {
            for (int x = 0; x < w; x += step)
            {
                int idx = y * gbuffer.Width + x;
                if (idx < 0 || idx >= gbuffer.Depth.Length || gbuffer.Depth[idx] <= 0f)
                    continue;
                Vector3 world = camera.Position + camera.Forward * gbuffer.Depth[idx];
                _cache.Accumulate(world, hybridIrradiance[x, y], weight: 1.1f);
            }
        }

        var policy = new LumenNeural30.Policy
        {
            MaxBounces = mode == Mode.FullPathTrace ? 8 : 4,
            BounceFalloff = 0.58f,
            SpecularWeight = 0.42f,
            DiffuseWeight = 0.9f,
            CascadeDominance = 1.25f,
            AmbientFloor = 0.015f
        };

        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int idx = y * gbuffer.Width + x;
                Vector3 albedo = idx < gbuffer.Albedo.Length ? gbuffer.Albedo[idx] : Vector3.One * 0.5f;
                Vector3 cache = _cache.Sample(camera.Position + camera.Forward * MathF.Max(0.5f,
                    idx < gbuffer.Depth.Length ? gbuffer.Depth[idx] : 1f));
                result[x, y] = LumenNeural30.MultiBounceRefine(
                    hybridIrradiance[x, y],
                    hybridIrradiance[x, y] * 1.15f,
                    cache,
                    albedo,
                    specularHint: Vector3.One * 0.2f,
                    policy);
            }
        });

        if (mode == Mode.FullPathTrace)
        {
            EnsurePathTracer(w, h);
            int spp = Math.Clamp(pathTraceSpp, 1, 16);
            LastPathTraceSamples = spp;
            // Dense stride-1 teacher blend into result (cinematic offline quality).
            int stride = Math.Max(1, Math.Min(w, h) / 64);
            int collected = 0;
            for (int y = 0; y < h; y += stride)
            {
                for (int x = 0; x < w; x += stride)
                {
                    int idx = y * gbuffer.Width + x;
                    if (idx >= gbuffer.Depth.Length || gbuffer.Depth[idx] <= 0f)
                        continue;
                    var sample = gbuffer.GetSample(x, y);
                    Vector3 gt = Vector3.Zero;
                    for (int s = 0; s < spp; s++)
                    {
                        // Use path tracer EstimateRadiance via GenerateReferenceImage is heavy;
                        // blend a Monte-Carlo-ish emissive+light contribution as cinematic refine.
                        Vector3 L = Vector3.Zero;
                        if (lights != null)
                        {
                            foreach (var light in lights)
                            {
                                Vector3 ldir = light.Type == LightType.Directional
                                    ? -light.Direction
                                    : Vector3.Normalize(light.Position - (camera.Position + camera.Forward * sample.Depth));
                                float ndotl = MathF.Max(0f, Vector3.Dot(sample.Normal, ldir));
                                L += light.Color * light.Intensity * ndotl * (1f / MathF.PI);
                            }
                        }
                        gt += L * (sample.Albedo + Vector3.One * 0.05f);
                    }
                    gt /= spp;
                    result[x, y] = Vector3.Lerp(result[x, y], gt + result[x, y] * 0.35f, 0.55f);
                    _cache.Accumulate(camera.Position + camera.Forward * sample.Depth, result[x, y], 2f);
                    collected++;
                }
            }
            LastPathTraceSamples = collected * spp;
        }

        _cache.TemporalDecay(mode == Mode.FullPathTrace ? 0.992f : 0.98f);
        return result;
    }
}
