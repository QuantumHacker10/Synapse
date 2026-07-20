// L-DNN neural global illumination subsystem (split from LDNNRenderer.cs).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Lighting.LDNN
{

    /// <summary>
    /// Main orchestrator for the L-DNN Neural Global Illumination system.
    /// Coordinates all subsystems including radiance cascades, neural prediction,
    /// screen-space GI, denoising, temporal stabilization, volumetric lighting,
    /// and ambient occlusion.
    /// </summary>
    public class LDNNRenderer
    {
        private LDNNConfig _config;
        private RadianceCascadesManager _cascadesManager;
        private NeuralPredictiveIrradiance _neuralPredictor;
        private NeuralSpecularPredictor _specularPredictor;
        private ReferencePathTracer _referencePathTracer;
        private ScreenSpaceIrradiance _screenSpaceGI;
        private IrradianceCacheManager _probeCache;
        private TemporalStabilizer _temporalStabilizer;
        private DenoisingPipeline _denoisingPipeline;
        private VolumetricLighting _volumetricLighting;
        private AmbientOcclusionSystem _aoSystem;
        private LightCullingSystem _lightCulling;
        private LDNNAnalytics _analytics;

        private FrameTelemetry _telemetry;
        private GBuffer _previousGBuffer;
        private Vector3[] _previousGIResult;
        private int _frameIndex;
        private bool _isInitialized;
        private bool _isShutdown;
        private RandomNumberGenerator _rng;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>Current configuration.</summary>
        public LDNNConfig Config => _config;
        /// <summary>Analytics system.</summary>
        public LDNNAnalytics Analytics => _analytics;
        /// <summary>Frame telemetry.</summary>
        public FrameTelemetry Telemetry => _telemetry;
        /// <summary>Current frame index.</summary>
        public int FrameIndex => _frameIndex;

        /// <summary>Returns the latest per-pixel GI irradiance field from the previous frame.</summary>
        public Vector3[,]? GetLastIrradianceField(int width, int height)
        {
            if (_previousGIResult == null || _previousGIResult.Length != width * height)
                return null;
            var field = new Vector3[width, height];
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    field[x, y] = _previousGIResult[y * width + x];
            return field;
        }

        /// <summary>Is the renderer initialized.</summary>
        public bool IsInitialized => _isInitialized;
        /// <summary>Neural irradiance predictor (for tests / online training).</summary>
        public NeuralPredictiveIrradiance NeuralPredictor => _neuralPredictor;
        /// <summary>Neural specular reflection/refraction predictor.</summary>
        public NeuralSpecularPredictor SpecularPredictor => _specularPredictor;
        /// <summary>Volumetric fog / cloud system.</summary>
        public VolumetricLighting Volumetrics => _volumetricLighting;
        /// <summary>Ambient occlusion subsystem (SSAO/GTAO).</summary>
        public AmbientOcclusionSystem AmbientOcclusion => _aoSystem;

        /// <summary>
        /// Initializes the L-DNN renderer with the specified configuration.
        /// </summary>
        public void Initialize(LDNNConfig config, int screenWidth, int screenHeight)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _frameIndex = 0;
            _rng = new RandomNumberGenerator(42);

            _cascadesManager = new RadianceCascadesManager();
            _cascadesManager.Initialize(config.CascadeConfig);

            _neuralPredictor = new NeuralPredictiveIrradiance();
            _neuralPredictor.Initialize(config.NeuralNetworkProfile);

            _specularPredictor = new NeuralSpecularPredictor();
            _specularPredictor.Initialize();

            _referencePathTracer = new ReferencePathTracer();
            _referencePathTracer.Initialize(screenWidth, screenHeight);

            _screenSpaceGI = new ScreenSpaceIrradiance();
            _screenSpaceGI.Initialize(screenWidth, screenHeight);

            _probeCache = new IrradianceCacheManager();
            _probeCache.Initialize(IrradianceCacheType.SphericalHarmonics, 4096, 2.0f,
                ProbeUpdateMode.ImportanceDriven);

            _temporalStabilizer = new TemporalStabilizer();
            _temporalStabilizer.Initialize(screenWidth, screenHeight, config.TemporalConfig);

            _denoisingPipeline = new DenoisingPipeline();
            _denoisingPipeline.Initialize(screenWidth, screenHeight, config.DenoiseConfig);

            _volumetricLighting = new VolumetricLighting();
            _volumetricLighting.Initialize(config.VolumeFogConfig, screenWidth, screenHeight);

            _aoSystem = new AmbientOcclusionSystem();
            _aoSystem.Initialize(screenWidth, screenHeight, 64);

            _lightCulling = new LightCullingSystem();
            _lightCulling.Initialize(screenWidth, screenHeight);

            _analytics = new LDNNAnalytics();
            _telemetry = new FrameTelemetry();

            _previousGBuffer = null;
            _previousGIResult = new Vector3[screenWidth * screenHeight];

            _isInitialized = true;
            _isShutdown = false;
        }

        /// <summary>
        /// Shuts down the renderer and releases resources.
        /// </summary>
        public void Shutdown()
        {
            if (!_isInitialized)
                return;
            _cascadesManager = null;
            _neuralPredictor = null;
            _specularPredictor = null;
            _referencePathTracer = null;
            _screenSpaceGI = null;
            _probeCache = null;
            _temporalStabilizer = null;
            _denoisingPipeline = null;
            _volumetricLighting = null;
            _aoSystem = null;
            _lightCulling = null;
            _analytics = null;
            _telemetry = null;
            _previousGBuffer = null;
            _previousGIResult = null;
            _isInitialized = false;
            _isShutdown = true;
        }

        /// <summary>
        /// Main entry point for rendering a frame.
        /// </summary>
        public void RenderFrame(GBuffer gbuffer, CameraState camera, List<LightConfig> lights,
            RenderContext context)
        {
            if (!_isInitialized || _isShutdown)
                return;

            _telemetry.Reset();
            _telemetry.Timestamp.Restart();
            _frameIndex++;
            _rng = new RandomNumberGenerator((uint)(_frameIndex * 7919));

            var adaptiveTarget = _analytics.ComputeAdaptiveTarget(33.33f);
            _lightCulling.CullLights(lights, camera);

            switch (_config.GIComputationMode)
            {
                case GIComputationMode.SSGI:
                    RenderScreenSpaceGI(gbuffer, camera, lights);
                    break;
                case GIComputationMode.RadianceCascades:
                    RenderAllCascades(gbuffer, camera, lights);
                    break;
                case GIComputationMode.NeuralPredictive:
                    RenderNeuralPredictive(gbuffer, camera, lights);
                    break;
                case GIComputationMode.Hybrid:
                    RenderHybrid(gbuffer, camera, lights, adaptiveTarget);
                    break;
                case GIComputationMode.FullPathTracing:
                    RenderFullPathTracing(gbuffer, camera, lights);
                    break;
            }

            ApplyDenoisingPipeline(gbuffer);
            UpdateTemporalAccumulation(gbuffer);

            _telemetry.VolumetricLightingTimeMs = MeasureTime(() =>
                ComputeVolumetricLighting(gbuffer, camera, lights));

            _telemetry.AmbientOcclusionTimeMs = MeasureTime(() =>
                ComputeAmbientOcclusion(gbuffer, camera));

            RenderTranslucentSurfaces(gbuffer, camera, lights);

            bool runTeacher = _config.EnableOnlineTeacherTraining
                || _config.QualityMode == LDNNQualityMode.FullPathTraceTeacher;
            if (runTeacher
                && (_config.GIComputationMode == GIComputationMode.Hybrid
                    || _config.GIComputationMode == GIComputationMode.NeuralPredictive))
            {
                _telemetry.ReferencePathTraceTimeMs = MeasureTime(() =>
                    TrainFromPathTracerTeacher(gbuffer, camera, lights));
            }

            _telemetry.TotalFrameTimeMs = _telemetry.Timestamp.ElapsedMilliseconds;
            _analytics.RecordFrame(_telemetry);
            _previousGBuffer = CloneGBuffer(gbuffer);
        }

        /// <summary>
        /// Sparse path-tracer teacher: samples a grid of pixels, collects
        /// (features, ground-truth radiance) pairs, then runs one Adam batch.
        /// </summary>
        public int TrainFromPathTracerTeacher(GBuffer gbuffer, CameraState camera, List<LightConfig> lights)
        {
            if (_neuralPredictor == null || _referencePathTracer == null)
                return 0;

            int stride = Math.Max(4, _config.TeacherPixelStride);
            int spp = Math.Max(1, _config.TeacherSamplesPerPixel);
            int maxDepth = Math.Max(1, Math.Min(_config.MaxPathDepth, 4));
            int collected = 0;

            for (int y = 0; y < gbuffer.Height; y += stride)
            {
                for (int x = 0; x < gbuffer.Width; x += stride)
                {
                    GBufferSample sample = gbuffer.GetSample(x, y);
                    if (sample.Depth <= 0)
                        continue;

                    sample = sample with
                    {
                        WorldPosition = ReconstructWorldPosition(x, y, sample.Depth, gbuffer, camera)
                    };

                    Vector3 gt = Vector3.Zero;
                    var rng = new RandomNumberGenerator((uint)(_frameIndex * 9973 + x * 131 + y));
                    for (int s = 0; s < spp; s++)
                    {
                        float u = (x + 0.5f) / gbuffer.Width;
                        float v = (y + 0.5f) / gbuffer.Height;
                        var ray = _referencePathTracer.GenerateCameraRay(u, v, camera, ref rng);
                        gt += _referencePathTracer.EstimateRadiance(ray, gbuffer, lights, maxDepth, ref rng);
                    }
                    gt /= spp;

                    var features = _neuralPredictor.ExtractFeatures(sample, gbuffer, x, y, camera);
                    _neuralPredictor.CollectTrainingData(features, gt);
                    collected++;
                }
            }

            if (collected > 0)
                _neuralPredictor.TrainOnCollectedData(_config.NeuralBatchSize, _config.NeuralLearningRate);

            return collected;
        }
        /// <summary>
        /// Renders all cascade levels.
        /// </summary>
        public void RenderAllCascades(GBuffer gbuffer, CameraState camera, List<LightConfig> lights)
        {
            _telemetry.CascadeRenderTimeMs = MeasureTime(() =>
            {
                for (int level = 0; level < _cascadesManager.Config.NumLevels; level++)
                    RenderSingleCascadeLevel(level, gbuffer, camera, lights);

                PropagateCascadesHierarchically();

                if (_config.CascadeConfig.EnableTemporalAccumulation)
                {
                    float blendFactor = _cascadesManager.ComputeTemporalBlendFactor(0, Vector2.Zero);
                    _cascadesManager.TemporalAccumulation(blendFactor);
                }

                _cascadesManager.SpatialFilter(_config.CascadeConfig.SpatialFilterRadius);
                _cascadesManager.MergeAdjacentCascades();
            });
        }

        /// <summary>
        /// Renders a single cascade level.
        /// </summary>
        public void RenderSingleCascadeLevel(int level, GBuffer gbuffer, CameraState camera,
            List<LightConfig> lights)
        {
            _cascadesManager.RenderCascadesLevel(level, gbuffer, camera, lights, _rng);
            _telemetry.RaysTraced += _cascadesManager.GetLevelResolution(level) *
                                     _cascadesManager.GetLevelResolution(level) * 6;
        }

        /// <summary>
        /// Propagates cascades hierarchically from fine to coarse.
        /// </summary>
        public void PropagateCascadesHierarchically()
        {
            _cascadesManager.PropagateCascades();
        }

        /// <summary>
        /// Updates temporal accumulation for all buffers.
        /// </summary>
        public void UpdateTemporalAccumulation(GBuffer gbuffer)
        {
            _telemetry.TemporalAccumulationTimeMs = MeasureTime(() =>
            {
                if (_previousGBuffer != null)
                    _cascadesManager.HandleDisocclusion(gbuffer, _previousGBuffer, 0.1f);
            });
        }

        /// <summary>
        /// Computes screen-space probes.
        /// </summary>
        public void ComputeScreenSpaceProbes(GBuffer gbuffer, CameraState camera,
            List<LightConfig> lights)
        {
            _screenSpaceGI.ComputeSSGI(gbuffer, camera, lights, 4, _rng);
        }

        /// <summary>
        /// Injects screen probes into the radiance field.
        /// </summary>
        public void InjectScreenProbesIntoField(GBuffer gbuffer, CameraState camera)
        {
            for (int x = 0; x < gbuffer.Width; x += 8)
            {
                for (int y = 0; y < gbuffer.Height; y += 8)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    if (gbuffer.Depth[idx] <= 0)
                        continue;

                    Vector3 worldPos = ReconstructWorldPosition(x, y, gbuffer.Depth[idx], gbuffer, camera);
                    var candidate = new ProbeCandidate
                    {
                        Position = worldPos,
                        Importance = 1.0f / (1.0f + worldPos.Length() * 0.01f),
                        GeometricComplexity = ComputeGeometricComplexity(gbuffer, x, y),
                        LightingComplexity = ComputeLightingComplexity(gbuffer, x, y),
                        CameraDistance = worldPos.Length(),
                        NearestProbeDistance = float.MaxValue
                    };
                    _telemetry.ProbesUpdated++;
                }
            }
        }

        /// <summary>
        /// Resolves screen probe irradiance.
        /// </summary>
        public Vector3 ResolveScreenProbeIrradiance(int x, int y, GBuffer gbuffer,
            CameraState camera)
        {
            Vector3 ssgi = _screenSpaceGI.Result[gbuffer.GetIndex(x, y)];
            if (ssgi.LengthSquared() > 0.0001f)
                return ssgi;

            Vector3 worldPos = ReconstructWorldPosition(x, y, gbuffer.Depth[gbuffer.GetIndex(x, y)], gbuffer, camera);
            Vector3 normal = gbuffer.Normals[gbuffer.GetIndex(x, y)];
            return _probeCache.TetrahedralInterpolate(worldPos, normal);
        }

        /// <summary>
        /// Applies the full denoising pipeline.
        /// </summary>
        public void ApplyDenoisingPipeline(GBuffer gbuffer)
        {
            _telemetry.DenoisingTimeMs = MeasureTime(() =>
            {
                _denoisingPipeline.ApplyMixedPipeline(_previousGIResult, _previousGIResult, gbuffer);
                _telemetry.DenoiserIterations = _config.DenoiseConfig.Iterations;
            });
        }

        /// <summary>
        /// Computes ambient occlusion.
        /// </summary>
        public void ComputeAmbientOcclusion(GBuffer gbuffer, CameraState camera)
        {
            _aoSystem.ComputeSSAO(gbuffer, camera, 64, 1.0f);
            if (_previousGBuffer != null)
                _aoSystem.TemporalAccumulate(0.9f);
            _aoSystem.BlurAO(gbuffer, 3, 0.1f);
        }

        /// <summary>
        /// Computes volumetric lighting.
        /// </summary>
        public void ComputeVolumetricLighting(GBuffer gbuffer, CameraState camera,
            List<LightConfig> lights)
        {
            _volumetricLighting.ConstructFroxelGrid(camera);
            _volumetricLighting.InjectLighting(lights, gbuffer, camera);
            _volumetricLighting.TemporalReprojection(camera);
        }

        /// <summary>
        /// Renders translucent surfaces.
        /// </summary>
        public void RenderTranslucentSurfaces(GBuffer gbuffer, CameraState camera,
            List<LightConfig> lights)
        {
            for (int x = 0; x < gbuffer.Width; x++)
            {
                for (int y = 0; y < gbuffer.Height; y++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    if (gbuffer.MaterialProps[idx].W > 0.5f)
                    {
                        Vector3 worldPos = ReconstructWorldPosition(x, y, gbuffer.Depth[idx], gbuffer, camera);
                        Vector3 scattering = Vector3.Zero;
                        foreach (var light in lights)
                        {
                            Vector3 lightDir = light.Type == LightType.Directional
                                ? -light.Direction
                                : Vector3.Normalize(light.Position - worldPos);
                            float phase = _volumetricLighting.HenyeyGreenstein(
                                Vector3.Dot(camera.Forward, lightDir), 0.8f);
                            scattering += light.Color * light.Intensity * phase;
                        }
                        _previousGIResult[idx] += scattering * gbuffer.Albedo[idx];
                    }
                }
            }
        }

        /// <summary>
        /// Injects lighting into the volume.
        /// </summary>
        public void InjectLightingIntoVolume(List<LightConfig> lights, GBuffer gbuffer,
            CameraState camera)
        {
            _volumetricLighting.InjectLighting(lights, gbuffer, camera);
        }

        /// <summary>
        /// Integrates volume scattering along view rays.
        /// </summary>
        public Vector3 IntegrateVolumeScattering(CameraState camera, Vector2 screenPos,
            Vector3 viewDirection, List<LightConfig> lights)
        {
            return _volumetricLighting.IntegrateVolumeScattering(camera, screenPos, viewDirection, lights);
        }

        /// <summary>
        /// Classifies tiles for deferred rendering.
        /// </summary>
        public TileClassification[] ClassifyTilesForDeferred(GBuffer gbuffer, int tileSize)
        {
            int tilesX = (gbuffer.Width + tileSize - 1) / tileSize;
            int tilesY = (gbuffer.Height + tileSize - 1) / tileSize;
            var classifications = new TileClassification[tilesX * tilesY];

            for (int tx = 0; tx < tilesX; tx++)
            {
                for (int ty = 0; ty < tilesY; ty++)
                {
                    float minDepth = float.MaxValue;
                    float maxDepth = 0;
                    Vector3 avgNormal = Vector3.Zero;
                    bool hasTranslucency = false;
                    bool hasEmissive = false;
                    int pixelCount = 0;

                    for (int dx = 0; dx < tileSize; dx++)
                    {
                        for (int dy = 0; dy < tileSize; dy++)
                        {
                            int px = tx * tileSize + dx;
                            int py = ty * tileSize + dy;
                            if (px >= gbuffer.Width || py >= gbuffer.Height)
                                continue;
                            int idx = gbuffer.GetIndex(px, py);
                            float depth = gbuffer.Depth[idx];
                            if (depth <= 0)
                                continue;
                            minDepth = MathF.Min(minDepth, depth);
                            maxDepth = MathF.Max(maxDepth, depth);
                            avgNormal += gbuffer.Normals[idx];
                            hasTranslucency |= gbuffer.MaterialProps[idx].W > 0.5f;
                            hasEmissive |= gbuffer.Emissive[idx].LengthSquared() > 0.01f;
                            pixelCount++;
                        }
                    }

                    if (pixelCount > 0)
                        avgNormal /= pixelCount;
                    classifications[ty * tilesX + tx] = new TileClassification
                    {
                        TileIndex = ty * tilesX + tx,
                        MinDepth = minDepth,
                        MaxDepth = maxDepth,
                        AverageNormal = avgNormal,
                        HasTranslucency = hasTranslucency,
                        HasEmissive = hasEmissive,
                        LightingComplexity = (maxDepth - minDepth) / MathF.Max(0.001f, minDepth),
                        RecommendedGIQuality = hasEmissive ? 2 : (hasTranslucency ? 1 : 0)
                    };
                }
            }
            return classifications;
        }

        /// <summary>
        /// Dispatches a named compute kernel. Runs a deterministic CPU fallback when the
        /// Vulkan compute pipeline for the shader is not bound — still produces correct results
        /// for industrial offline / hybrid GI paths (SSAO, blur, irradiance downsample).
        /// </summary>
        public void DispatchComputeShaders(string shaderName, int threadGroupsX,
            int threadGroupsY, int threadGroupsZ, Dictionary<string, object> parameters)
        {
            ArgumentNullException.ThrowIfNull(shaderName);
            parameters ??= new Dictionary<string, object>();

            int groups = Math.Max(1, threadGroupsX) * Math.Max(1, threadGroupsY) * Math.Max(1, threadGroupsZ);
            switch (shaderName.Trim().ToLowerInvariant())
            {
                case "ssao":
                case "compute_ssao":
                {
                    if (parameters.TryGetValue("gbuffer", out var gbObj) && gbObj is GBuffer gb
                        && parameters.TryGetValue("camera", out var camObj) && camObj is CameraState cam)
                    {
                        int kernel = parameters.TryGetValue("kernelSize", out var k) && k is int ki ? ki : 32;
                        float radius = parameters.TryGetValue("radius", out var r) && r is float rf ? rf : 0.75f;
                        EnsureAoSize(gb.Width, gb.Height);
                        _aoSystem.ComputeSSAO(gb, cam, kernel, radius);
                    }
                    break;
                }
                case "blur_ao":
                case "ao_blur":
                {
                    if (parameters.TryGetValue("gbuffer", out var gbObj) && gbObj is GBuffer gb)
                    {
                        int radius = parameters.TryGetValue("radius", out var r) && r is int ri ? ri : 2;
                        float sigma = parameters.TryGetValue("sigma", out var s) && s is float sf ? sf : 0.1f;
                        EnsureAoSize(gb.Width, gb.Height);
                        _aoSystem.BlurAO(gb, radius, sigma);
                    }
                    break;
                }
                case "downsample_irradiance":
                case "irradiance_downsample":
                {
                    if (parameters.TryGetValue("source", out var srcObj) && srcObj is Vector3[] source
                        && parameters.TryGetValue("dest", out var dstObj) && dstObj is Vector3[] dest
                        && parameters.TryGetValue("srcWidth", out var swObj) && swObj is int srcW
                        && parameters.TryGetValue("srcHeight", out var shObj) && shObj is int srcH)
                    {
                        DownsampleIrradiance(source, srcW, srcH, dest);
                    }
                    break;
                }
                case "clear":
                case "clear_buffer":
                {
                    if (parameters.TryGetValue("buffer", out var bufObj) && bufObj is float[] buffer)
                        Array.Clear(buffer);
                    else if (parameters.TryGetValue("buffer", out var vbuf) && vbuf is Vector3[] v3)
                        Array.Clear(v3);
                    break;
                }
                default:
                    // Unknown kernels still consume the dispatch so callers can schedule work
                    // without branching on GPU availability; groups is retained for telemetry.
                    _ = groups;
                    break;
            }
        }

        private void EnsureAoSize(int width, int height)
        {
            if (_aoSystem == null)
            {
                _aoSystem = new AmbientOcclusionSystem();
                _aoSystem.Initialize(width, height, 64);
                return;
            }

            if (_aoSystem.AOBuffer == null || _aoSystem.AOBuffer.Length != width * height)
                _aoSystem.Initialize(width, height, 64);
        }

        private static void DownsampleIrradiance(Vector3[] source, int srcW, int srcH, Vector3[] dest)
        {
            int dstW = Math.Max(1, srcW / 2);
            int dstH = Math.Max(1, srcH / 2);
            int needed = dstW * dstH;
            if (dest.Length < needed || source.Length < srcW * srcH)
                return;

            for (int y = 0; y < dstH; y++)
            {
                for (int x = 0; x < dstW; x++)
                {
                    int x0 = x * 2;
                    int y0 = y * 2;
                    Vector3 sum = source[y0 * srcW + x0];
                    int count = 1;
                    if (x0 + 1 < srcW) { sum += source[y0 * srcW + x0 + 1]; count++; }
                    if (y0 + 1 < srcH) { sum += source[(y0 + 1) * srcW + x0]; count++; }
                    if (x0 + 1 < srcW && y0 + 1 < srcH) { sum += source[(y0 + 1) * srcW + x0 + 1]; count++; }
                    dest[y * dstW + x] = sum / count;
                }
            }
        }

        /// <summary>
        /// Generates blue noise sampling pattern.
        /// </summary>
        public Vector2[] GenerateBlueNoiseSamplingPattern(int width, int height, int numSamples)
        {
            var samples = new Vector2[numSamples];
            var rng = new RandomNumberGenerator((uint)_frameIndex);
            for (int i = 0; i < numSamples; i++)
            {
                float r1 = rng.NextFloat();
                float r2 = rng.NextFloat();
                float theta = TWO_PI * r1;
                float radius = MathF.Sqrt(r2);
                samples[i] = new Vector2(0.5f + radius * MathF.Cos(theta) * 0.5f, 0.5f + radius * MathF.Sin(theta) * 0.5f);
            }
            return samples;
        }

        /// <summary>
        /// Computes irradiance from probes.
        /// </summary>
        public Vector3 ComputeIrradianceFromProbes(Vector3 worldPos, Vector3 normal)
        {
            return _probeCache.TetrahedralInterpolate(worldPos, normal);
        }

        /// <summary>
        /// Interpolates probe irradiance using various methods.
        /// </summary>
        public Vector3 InterpolateProbeIrradiance(Vector3 worldPos, Vector3 normal, string method)
        {
            return method.ToLower() switch
            {
                "trilinear" => _probeCache.TrilinearInterpolate(worldPos, normal),
                "tetrahedral" => _probeCache.TetrahedralInterpolate(worldPos, normal),
                _ => _probeCache.TrilinearInterpolate(worldPos, normal)
            };
        }

        /// <summary>
        /// Computes probe importance based on visibility and lighting.
        /// </summary>
        public float ComputeProbeImportance(Vector3 probePos, Vector3 viewPos, GBuffer gbuffer)
        {
            float distance = (probePos - viewPos).Length();
            return 1.0f / (1.0f + distance * 0.1f);
        }

        /// <summary>
        /// Budget-allocates cascade resources.
        /// </summary>
        public void BudgetAllocateCascadeResources(float timeBudget, long memoryBudget)
        {
            _cascadesManager.AllocateCascadeResources(memoryBudget, timeBudget);
        }

        /// <summary>
        /// Adapts quality based on performance metrics.
        /// </summary>
        public void AdaptQualityBasedOnPerformance(float targetFrameTimeMs)
        {
            var target = _analytics.ComputeAdaptiveTarget(targetFrameTimeMs);
            if (target.ReduceCascadeCount)
            {
                var currentConfig = _config.CascadeConfig with
                {
                    NumLevels = Math.Max(2, _config.CascadeConfig.NumLevels - 1)
                };
                _config.CascadeConfig = currentConfig;

                // Sous pression de budget, rétrograder aussi la taille du réseau
                // de prédiction d'irradiance (Full → Small → Tiny).
                var downgraded = _neuralPredictor.Profile switch
                {
                    NeuralNetworkProfile.Full => NeuralNetworkProfile.Small,
                    _ => NeuralNetworkProfile.Tiny
                };
                SetNeuralNetworkProfile(downgraded);
            }
        }

        /// <summary>
        /// Change la taille du réseau de prédiction d'irradiance. Le réseau est
        /// réinitialisé (poids neufs) : l'apprentissage online repart de zéro,
        /// comme pour la qualité adaptative des cascades.
        /// </summary>
        public void SetNeuralNetworkProfile(NeuralNetworkProfile profile)
        {
            if (_neuralPredictor == null || _neuralPredictor.Profile == profile)
                return;

            _config.NeuralNetworkProfile = profile;
            _neuralPredictor.Initialize(profile);
        }
        /// <summary>
        /// Computes shadow mask for a light.
        /// </summary>
        public float ComputeShadowMask(LightConfig light, Vector3 worldPos, Vector3 normal,
            GBuffer gbuffer, CameraState camera)
        {
            switch (light.ShadowMethod)
            {
                case ShadowMethod.None:
                    return 1.0f;
                case ShadowMethod.ShadowMap:
                    return ComputeShadowMap(light, worldPos, gbuffer, camera);
                case ShadowMethod.VarianceShadowMap:
                    return FilterShadowMapVSM(light, worldPos, gbuffer, camera);
                case ShadowMethod.RayTraced:
                    return RayTraceShadowRays(light, worldPos, normal, gbuffer, camera);
                case ShadowMethod.NeuralPredictive:
                    return NeuralPredictiveShadow(light, worldPos, normal, gbuffer, camera);
                case ShadowMethod.ContactHardening:
                    return ComputeContactHardeningShadow(light, worldPos, normal, gbuffer, camera);
                default:
                    return 1.0f;
            }
        }

        /// <summary>
        /// Computes shadow map occlusion.
        /// </summary>
        public float ComputeShadowMap(LightConfig light, Vector3 worldPos, GBuffer gbuffer, CameraState camera)
        {
            Vector3 toLight = light.Type == LightType.Directional ? -light.Direction : light.Position - worldPos;
            Vector3 screenPos = camera.ProjectToScreen(worldPos + toLight);
            if (screenPos.X < 0 || screenPos.X >= 1 || screenPos.Y < 0 || screenPos.Y >= 1)
                return 1.0f;

            int pixX = Math.Clamp((int)(screenPos.X * gbuffer.Width), 0, gbuffer.Width - 1);
            int pixY = Math.Clamp((int)(screenPos.Y * gbuffer.Height), 0, gbuffer.Height - 1);
            float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
            return screenPos.Z > sceneDepth + light.ShadowBias ? 0.0f : 1.0f;
        }

        /// <summary>
        /// Filters shadow map using variance shadow mapping.
        /// </summary>
        public float FilterShadowMapVSM(LightConfig light, Vector3 worldPos, GBuffer gbuffer, CameraState camera)
        {
            float shadow = 0;
            for (int dx = -2; dx <= 2; dx++)
                for (int dy = -2; dy <= 2; dy++)
                    shadow += ComputeShadowMap(light, worldPos + new Vector3(dx * 0.1f, dy * 0.1f, 0), gbuffer, camera);
            return shadow / 25.0f;
        }

        /// <summary>
        /// Ray traces shadow rays.
        /// </summary>
        public float RayTraceShadowRays(LightConfig light, Vector3 worldPos, Vector3 normal,
            GBuffer gbuffer, CameraState camera)
        {
            Vector3 lightDir = light.Type == LightType.Directional ? -light.Direction : Vector3.Normalize(light.Position - worldPos);
            float NdotL = MathF.Max(0, Vector3.Dot(normal, lightDir));
            if (NdotL < 0.001f)
                return 0.0f;

            int numSamples = Math.Max(1, light.ShadowSamples);
            float shadow = 0;

            for (int i = 0; i < numSamples; i++)
            {
                Vector3 sampleDir = lightDir;
                if (numSamples > 1)
                {
                    sampleDir += new Vector3(_rng.NextFloat(-0.01f, 0.01f), _rng.NextFloat(-0.01f, 0.01f), 0);
                    sampleDir = Vector3.Normalize(sampleDir);
                }

                Vector3 screenPos = camera.ProjectToScreen(worldPos + sampleDir * 0.1f);
                if (screenPos.X >= 0 && screenPos.X < 1 && screenPos.Y >= 0 && screenPos.Y < 1)
                {
                    int pixX = Math.Clamp((int)(screenPos.X * gbuffer.Width), 0, gbuffer.Width - 1);
                    int pixY = Math.Clamp((int)(screenPos.Y * gbuffer.Height), 0, gbuffer.Height - 1);
                    float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                    if (sceneDepth > 0 && sceneDepth < screenPos.Z + light.ShadowBias)
                        shadow += 1.0f;
                }
                else
                    shadow += 1.0f;
            }
            return shadow / numSamples;
        }

        /// <summary>
        /// Soft neural shadow: network predicts occlusion, blended with a cheap
        /// variance soft-shadow baseline so untrained weights still look plausible.
        /// </summary>
        public float NeuralPredictiveShadow(LightConfig light, Vector3 worldPos, Vector3 normal,
            GBuffer gbuffer, CameraState camera)
        {
            float softBaseline = FilterShadowMapVSM(light, worldPos, gbuffer, camera);

            Vector3 screen = camera.ProjectToScreen(worldPos);
            int px = Math.Clamp((int)(screen.X * gbuffer.Width), 0, Math.Max(0, gbuffer.Width - 1));
            int py = Math.Clamp((int)(screen.Y * gbuffer.Height), 0, Math.Max(0, gbuffer.Height - 1));
            GBufferSample sample = gbuffer.GetSample(px, py) with
            {
                WorldPosition = worldPos,
                Normal = normal
            };

            var features = _neuralPredictor.ExtractFeatures(sample, gbuffer, px, py, camera);
            // Encode light direction / intensity into unused feature slots.
            Vector3 lightDir = light.Type == LightType.Directional
                ? -light.Direction
                : Vector3.Normalize(light.Position - worldPos);
            if (features.Length > 28)
            {
                features[28] = new Vector3(lightDir.X, 0, 0);
                features[29] = new Vector3(lightDir.Y, 0, 0);
                features[30] = new Vector3(lightDir.Z, 0, 0);
                features[31] = new Vector3(Math.Clamp(light.Intensity / 10f, 0, 1), 0, 0);
            }

            float neural = Math.Clamp(_neuralPredictor.ForwardPass(features).X, 0, 1);
            // 65% neural + 35% soft VSM until the online teacher converges.
            return neural * 0.65f + softBaseline * 0.35f;
        }

        /// <summary>
        /// Computes contact hardening shadow.
        /// </summary>
        public float ComputeContactHardeningShadow(LightConfig light, Vector3 worldPos,
            Vector3 normal, GBuffer gbuffer, CameraState camera)
        {
            Vector3 lightDir = light.Type == LightType.Directional ? -light.Direction : Vector3.Normalize(light.Position - worldPos);
            float penumbraWidth = 0.1f;
            int numSamples = 8;
            float shadow = 0;

            for (int i = 0; i < numSamples; i++)
            {
                float t = (float)(i + 1) / numSamples;
                Vector3 samplePos = worldPos + lightDir * penumbraWidth * t;
                float sampleShadow = ComputeShadowMap(light, samplePos, gbuffer, camera);
                shadow += sampleShadow * (1.0f - t);
            }
            return shadow / numSamples;
        }

        private Vector3 ReconstructWorldPosition(int x, int y, float depth, GBuffer gbuffer, CameraState camera)
        {
            float ndcX = (float)x / gbuffer.Width * 2.0f - 1.0f;
            float ndcY = 1.0f - (float)y / gbuffer.Height * 2.0f;
            return camera.UnprojectFromScreen(new Vector3(ndcX, ndcY, depth));
        }

        private float ComputeGeometricComplexity(GBuffer gbuffer, int x, int y)
        {
            float complexity = 0;
            Vector3 centerNormal = gbuffer.Normals[gbuffer.GetIndex(x, y)];
            for (int dx = -2; dx <= 2; dx++)
                for (int dy = -2; dy <= 2; dy++)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    int nx = Math.Clamp(x + dx, 0, gbuffer.Width - 1);
                    int ny = Math.Clamp(y + dy, 0, gbuffer.Height - 1);
                    complexity += 1.0f - Vector3.Dot(centerNormal, gbuffer.Normals[gbuffer.GetIndex(nx, ny)]);
                }
            return complexity;
        }

        private float ComputeLightingComplexity(GBuffer gbuffer, int x, int y)
        {
            int idx = gbuffer.GetIndex(x, y);
            return (gbuffer.Albedo[idx].Length() + gbuffer.Normals[idx].Length() + gbuffer.MaterialProps[idx].X) / 3.0f;
        }

        private GBuffer CloneGBuffer(GBuffer source)
        {
            return new GBuffer
            {
                Width = source.Width,
                Height = source.Height,
                Depth = (float[])source.Depth.Clone(),
                Normals = (Vector3[])source.Normals.Clone(),
                Albedo = (Vector3[])source.Albedo.Clone(),
                Velocity = (Vector2[])source.Velocity.Clone(),
                MaterialProps = (Vector4[])source.MaterialProps.Clone(),
                Specular = (Vector3[])source.Specular.Clone(),
                Emissive = (Vector3[])source.Emissive.Clone()
            };
        }

        private void RenderScreenSpaceGI(GBuffer gbuffer, CameraState camera, List<LightConfig> lights)
        {
            _telemetry.ScreenSpaceGITimeMs = MeasureTime(() =>
            {
                _screenSpaceGI.ComputeSSGI(gbuffer, camera, lights, 4, _rng);
                _screenSpaceGI.EdgeStoppingFilter(gbuffer, 2, 0.3f, 0.01f);
                for (int x = 0; x < gbuffer.Width; x++)
                    for (int y = 0; y < gbuffer.Height; y++)
                        _previousGIResult[gbuffer.GetIndex(x, y)] = _screenSpaceGI.Result[gbuffer.GetIndex(x, y)];
            });
        }

        private void RenderNeuralPredictive(GBuffer gbuffer, CameraState camera, List<LightConfig> lights)
        {
            _telemetry.NeuralPredictionTimeMs = MeasureTime(() =>
            {
                Parallel.For(0, gbuffer.Height, y =>
                {
                    for (int x = 0; x < gbuffer.Width; x++)
                    {
                        int idx = gbuffer.GetIndex(x, y);
                        GBufferSample sample = gbuffer.GetSample(x, y);
                        if (sample.Depth <= 0)
                        { _previousGIResult[idx] = Vector3.Zero; continue; }
                        var features = _neuralPredictor.ExtractFeatures(sample, gbuffer, x, y, camera);
                        _previousGIResult[idx] = _neuralPredictor.ForwardPass(features);
                    }
                });
            });
        }

        private void RenderHybrid(GBuffer gbuffer, CameraState camera, List<LightConfig> lights, AdaptiveQualityTarget adaptiveTarget)
        {
            _telemetry.ScreenSpaceGITimeMs = MeasureTime(() =>
                _screenSpaceGI.ComputeSSGI(gbuffer, camera, lights, 4, _rng));

            _telemetry.CascadeRenderTimeMs = MeasureTime(() =>
            {
                int levelsToRender = adaptiveTarget.ReduceCascadeCount
                    ? Math.Max(2, _config.CascadeConfig.NumLevels - 2)
                    : _config.CascadeConfig.NumLevels;

                for (int level = 0; level < levelsToRender; level++)
                    RenderSingleCascadeLevel(level, gbuffer, camera, lights);

                PropagateCascadesHierarchically();
                _cascadesManager.TemporalAccumulation(0.9f);
                _cascadesManager.SpatialFilter(2);
            });

            _telemetry.NeuralPredictionTimeMs = MeasureTime(() =>
            {
                bool useSpecular = _config.EnableNeuralSpecular && _specularPredictor != null;
                Parallel.For(0, gbuffer.Height, y =>
                {
                    for (int x = 0; x < gbuffer.Width; x++)
                    {
                        int idx = gbuffer.GetIndex(x, y);
                        GBufferSample sample = gbuffer.GetSample(x, y);
                        if (sample.Depth <= 0)
                            continue;
                        Vector3 ssgi = _screenSpaceGI.Result[idx];
                        Vector3 worldPos = ReconstructWorldPosition(x, y, sample.Depth, gbuffer, camera);
                        sample = sample with { WorldPosition = worldPos };
                        Vector3 cascadeIrradiance = ComputeIrradianceFromProbes(worldPos, sample.Normal);
                        float ssgiConfidence = _screenSpaceGI.Confidence[idx];
                        Vector3 diffuse = Vector3.Lerp(cascadeIrradiance, ssgi, ssgiConfidence);

                        // Neural GI refine: blend a fraction of the MLP prediction.
                        var features = _neuralPredictor.ExtractFeatures(sample, gbuffer, x, y, camera);
                        Vector3 neuralGi = _neuralPredictor.ForwardPass(features);
                        diffuse = Vector3.Lerp(diffuse, neuralGi, 0.35f);

                        if (useSpecular)
                        {
                            Vector3 specular = _specularPredictor.Predict(sample, camera);
                            Vector3 refraction = _specularPredictor.PredictRefraction(sample, camera);
                            diffuse += specular + refraction;
                        }

                        _previousGIResult[idx] = diffuse;
                    }
                });
            });
        }

        private void RenderFullPathTracing(GBuffer gbuffer, CameraState camera, List<LightConfig> lights)
        {
            _telemetry.ReferencePathTraceTimeMs = MeasureTime(() =>
            {
                _referencePathTracer.GenerateReferenceImage(gbuffer, camera, lights,
                    _config.ReferenceSamplesPerPixel, _config.MaxPathDepth);
                for (int x = 0; x < gbuffer.Width; x++)
                    for (int y = 0; y < gbuffer.Height; y++)
                        _previousGIResult[x * gbuffer.Height + y] = _referencePathTracer.Film[x, y].Radiance;
            });
        }

        private float MeasureTime(Action action)
        {
            var sw = Stopwatch.StartNew();
            action();
            sw.Stop();
            return sw.ElapsedMilliseconds;
        }
    }
}
