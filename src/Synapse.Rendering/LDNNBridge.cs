using System;
using System.Collections.Generic;
using System.Numerics;
using GDNN.Lighting.LDNN;
using GDNN.Rendering;
using GDNN.RHI.Vulkan;
using GDNN.Scene;

namespace GDNN.Rendering.Bridge
{
    /// <summary>How the L-DNN bridge last populated its G-buffer.</summary>
    public enum GiGBufferFillMode
    {
        None,
        Constants,
        ProceduralPreview,
        GpuResident,
        GpuReadback
    }

    /// <summary>
    /// Thin integration layer between deferred rendering and <see cref="LDNNRenderer"/>.
    /// Owns G-Buffer proxies, camera/light state, static GI caching, and LLM lighting apply.
    /// </summary>
    public class LDNNBridge : IDisposable
    {
        private LDNNRenderer _ldnn;
        private GBuffer _gbuffer;
        private CameraState _cameraState;
        private bool _initialized;
        private int _width;
        private int _height;
        private LDNNConfig _config;
        private List<LightConfig> _lights;
        private bool _disposed;

        private int _gbufferVersion;
        private int? _lastStateHash;
        private Vector3[,]? _cachedIrradiance;
        private int _aoGbufferVersion = -1;
        private readonly GpuResidentGiPipeline _residentGi = new();
        private bool _lastFillWasConstants;

        /// <summary>Tracks the most recent G-buffer population strategy.</summary>
        public GiGBufferFillMode LastFillMode { get; private set; } = GiGBufferFillMode.None;

        /// <summary>Monotonic frame counter from the underlying L-DNN renderer.</summary>
        public int FrameIndex => _ldnn?.FrameIndex ?? 0;

        /// <summary>True after <see cref="Initialize"/> completes.</summary>
        public bool IsInitialized => _initialized;

        /// <summary>
        /// Reuses the previous GI result when camera, lights, and G-Buffer version are unchanged.
        /// </summary>
        public bool EnableStaticSceneCache { get; set; } = true;

        /// <summary>Number of frames served from the static GI cache.</summary>
        public long StaticCacheHits { get; private set; }

        /// <summary>GPU-resident GI pipeline (SSGI without per-frame readback).</summary>
        public GpuResidentGiPipeline ResidentGi => _residentGi;

        /// <summary>Which GI path produced the last irradiance.</summary>
        public GiComputePath LastGiPath => _residentGi.LastPath;

        /// <summary>True when a GPU-origin G-buffer is resident and usable without constants.</summary>
        public bool HasResidentGBuffer => _residentGi.HasResidentGBuffer;

        /// <summary>Allocates G-Buffer storage for the given viewport size.</summary>
        public LDNNBridge(int width, int height)
        {
            _width = width;
            _height = height;
            _gbuffer = new GBuffer { Width = width, Height = height };
            AllocateGBufferArrays();
            _cameraState = new CameraState();
            _lights = new List<LightConfig>();
        }

        private void AllocateGBufferArrays()
        {
            int count = Math.Max(1, _width * _height);
            _gbuffer.Depth = new float[count];
            _gbuffer.Normals = new Vector3[count];
            _gbuffer.Albedo = new Vector3[count];
            _gbuffer.Velocity = new Vector2[count];
            _gbuffer.MaterialProps = new Vector4[count];
            _gbuffer.Specular = new Vector3[count];
            _gbuffer.Emissive = new Vector3[count];
        }

        /// <summary>Constructs and initializes the L-DNN renderer with the given or default config.</summary>
        public void Initialize(LDNNConfig config = null)
        {
            _config = config ?? CreateDefaultConfig();
            _ldnn = new LDNNRenderer();
            _ldnn.Initialize(_config, _width, _height);
            _initialized = true;
        }

        /// <summary>Resizes internal buffers and invalidates cached GI.</summary>
        public void Resize(int width, int height)
        {
            InvalidateGICache();
            _width = width;
            _height = height;
            _gbuffer.Width = width;
            _gbuffer.Height = height;
            AllocateGBufferArrays();
        }

        /// <summary>Updates view/projection matrices and derived camera vectors for GI.</summary>
        public void UpdateCamera(
            Matrix4x4 view, Matrix4x4 projection,
            Vector3 position, Vector3 forward, Vector3 right, Vector3 up,
            float fov, float aspect, float nearPlane, float farPlane)
        {
            _cameraState.ViewMatrix = view;
            _cameraState.ProjectionMatrix = projection;
            _cameraState.ViewProjectionMatrix = view * projection;
            Matrix4x4.Invert(view, out var invView);
            Matrix4x4.Invert(projection, out var invProj);
            _cameraState.InverseViewMatrix = invView;
            _cameraState.InverseProjectionMatrix = invProj;
            _cameraState.InverseViewProjectionMatrix = invView * invProj;
            _cameraState.Position = position;
            _cameraState.Forward = forward;
            _cameraState.Right = right;
            _cameraState.Up = up;
            _cameraState.FieldOfView = fov;
            _cameraState.AspectRatio = aspect;
            _cameraState.NearPlane = nearPlane;
            _cameraState.FarPlane = farPlane;
            _cameraState.Recompute();
        }

        /// <summary>Replaces the light list used by the next GI pass.</summary>
        public void SetLights(List<LightConfig> lights)
        {
            _lights = lights ?? new List<LightConfig>();
        }

        /// <summary>Appends a directional light with neural shadow settings.</summary>
        public void AddDirectionalLight(Vector3 direction, Vector3 color, float intensity)
        {
            _lights.Add(new LightConfig
            {
                Type = LightType.Directional,
                Direction = Vector3.Normalize(direction),
                Color = color,
                Intensity = intensity,
                ShadowMethod = ShadowMethod.NeuralPredictive,
                ShadowBias = 0.005f,
                ShadowSamples = 16,
                Importance = 1.0f
            });
        }

        /// <summary>Appends a point light with neural shadow settings.</summary>
        public void AddPointLight(Vector3 position, Vector3 color, float intensity, float range)
        {
            _lights.Add(new LightConfig
            {
                Type = LightType.Point,
                Position = position,
                Color = color,
                Intensity = intensity,
                Range = range,
                ShadowMethod = ShadowMethod.NeuralPredictive,
                ShadowBias = 0.005f,
                ShadowSamples = 8,
                Importance = 0.8f
            });
        }

        /// <summary>
        /// Overlays meshlet visibility (depth/normal/albedo) onto the CPU G-buffer so Hybrid GI
        /// samples real cluster geometry instead of constant fill.
        /// </summary>
        public void OverlayMeshletGBuffer(
            Action<float[], Vector3[], Vector3[], int, int> painter)
        {
            if (!_initialized || painter == null)
                return;
            painter(_gbuffer.Depth, _gbuffer.Normals, _gbuffer.Albedo, _width, _height);
            _gbufferVersion++;
            _lastFillWasConstants = false;
            _residentGi.UpdateFromGBuffer(_gbuffer);
        }

        /// <summary>Copies depth/normal/albedo/emissive arrays into the internal G-Buffer.</summary>
        public void FillGBuffer(
            float[] depthData,
            Vector3[] normalData,
            Vector3[] albedoData,
            Vector3[] emissiveData)
        {
            _gbufferVersion++;
            _lastFillWasConstants = false;
            LastFillMode = GiGBufferFillMode.GpuReadback;
            if (depthData != null && depthData.Length == _gbuffer.Depth.Length)
                Array.Copy(depthData, _gbuffer.Depth, depthData.Length);
            if (normalData != null && normalData.Length == _gbuffer.Normals.Length)
                Array.Copy(normalData, _gbuffer.Normals, normalData.Length);
            if (albedoData != null && albedoData.Length == _gbuffer.Albedo.Length)
                Array.Copy(albedoData, _gbuffer.Albedo, albedoData.Length);
            if (emissiveData != null && emissiveData.Length == _gbuffer.Emissive.Length)
                Array.Copy(emissiveData, _gbuffer.Emissive, emissiveData.Length);
            _residentGi.UpdateFromGBuffer(_gbuffer);
        }

        /// <summary>Copies all G-buffer channels including velocity, material, and specular.</summary>
        public void FillGBufferComplete(
            float[] depthData,
            Vector3[] normalData,
            Vector3[] albedoData,
            Vector3[] emissiveData,
            Vector2[] velocityData,
            Vector4[] materialProps,
            Vector3[] specularData)
        {
            FillGBuffer(depthData, normalData, albedoData, emissiveData);
            if (velocityData != null && velocityData.Length == _gbuffer.Velocity.Length)
                Array.Copy(velocityData, _gbuffer.Velocity, velocityData.Length);
            if (materialProps != null && materialProps.Length == _gbuffer.MaterialProps.Length)
                Array.Copy(materialProps, _gbuffer.MaterialProps, materialProps.Length);
            if (specularData != null && specularData.Length == _gbuffer.Specular.Length)
                Array.Copy(specularData, _gbuffer.Specular, specularData.Length);
        }

        /// <summary>Fills the G-Buffer with uniform constants (used when GPU readback is not wired).</summary>
        public void FillGBufferFromConstants(
            float fillDepth, Vector3 fillNormal, Vector3 fillAlbedo)
        {
            _gbufferVersion++;
            _lastFillWasConstants = true;
            LastFillMode = GiGBufferFillMode.Constants;
            int count = _width * _height;
            for (int i = 0; i < count; i++)
            {
                _gbuffer.Depth[i] = fillDepth;
                _gbuffer.Normals[i] = fillNormal;
                _gbuffer.Albedo[i] = fillAlbedo;
                _gbuffer.Velocity[i] = Vector2.Zero;
                _gbuffer.MaterialProps[i] = Vector4.Zero;
                _gbuffer.Specular[i] = new Vector3(0.04f, 0.04f, 0.04f);
                _gbuffer.Emissive[i] = Vector3.Zero;
            }
            _residentGi.MarkConstantFallback();
        }

        /// <summary>
        /// Fills the G-buffer with a lightweight procedural preview (ground + sphere)
        /// so L-DNN can compute spatial GI without a GPU readback path.
        /// </summary>
        public void FillGBufferProceduralPreview()
        {
            _gbufferVersion++;
            _lastFillWasConstants = false;
            LastFillMode = GiGBufferFillMode.ProceduralPreview;

            var forward = _cameraState.Forward;
            if (forward.LengthSquared() < 1e-6f)
                forward = Vector3.UnitZ;
            forward = Vector3.Normalize(forward);

            var right = _cameraState.Right;
            if (right.LengthSquared() < 1e-6f)
                right = Vector3.UnitX;
            right = Vector3.Normalize(right);

            var up = Vector3.Normalize(Vector3.Cross(right, forward));
            float aspect = _width > 0 ? (float)_width / _height : 16f / 9f;
            float tanHalfFov = MathF.Tan(_cameraState.FieldOfView * 0.5f);

            var sphereCenter = new Vector3(0f, 0.55f, 0f);
            const float sphereRadius = 0.45f;

            for (int y = 0; y < _height; y++)
            {
                float v = (_height > 1 ? y / (float)(_height - 1) : 0.5f) * 2f - 1f;
                for (int x = 0; x < _width; x++)
                {
                    float u = (_width > 1 ? x / (float)(_width - 1) : 0.5f) * 2f - 1f;
                    var rayDir = Vector3.Normalize(forward + right * (u * tanHalfFov * aspect) + up * (-v * tanHalfFov));
                    int idx = y * _width + x;

                    if (TryTracePreviewScene(_cameraState.Position, rayDir, sphereCenter, sphereRadius,
                            out float depth, out Vector3 normal, out Vector3 albedo))
                    {
                        _gbuffer.Depth[idx] = depth;
                        _gbuffer.Normals[idx] = normal;
                        _gbuffer.Albedo[idx] = albedo;
                    }
                    else
                    {
                        _gbuffer.Depth[idx] = _cameraState.FarPlane;
                        _gbuffer.Normals[idx] = Vector3.UnitY;
                        _gbuffer.Albedo[idx] = new Vector3(0.02f, 0.025f, 0.03f);
                    }

                    _gbuffer.Velocity[idx] = Vector2.Zero;
                    _gbuffer.MaterialProps[idx] = new Vector4(0.35f, 0.08f, 0f, 0f);
                    _gbuffer.Specular[idx] = new Vector3(0.06f);
                    _gbuffer.Emissive[idx] = Vector3.Zero;
                }
            }

            _residentGi.UpdateFromGBuffer(_gbuffer);
        }

        private static bool TryTracePreviewScene(
            Vector3 origin,
            Vector3 dir,
            Vector3 sphereCenter,
            float sphereRadius,
            out float depth,
            out Vector3 normal,
            out Vector3 albedo)
        {
            depth = 0f;
            normal = Vector3.UnitY;
            albedo = Vector3.One;

            float nearest = float.MaxValue;
            bool hit = false;

            if (MathF.Abs(dir.Y) > 1e-5f)
            {
                float tPlane = -origin.Y / dir.Y;
                if (tPlane > 0.01f && tPlane < nearest)
                {
                    nearest = tPlane;
                    hit = true;
                    normal = Vector3.UnitY;
                    albedo = new Vector3(0.22f, 0.24f, 0.28f);
                }
            }

            var oc = origin - sphereCenter;
            float b = Vector3.Dot(oc, dir);
            float c = oc.LengthSquared() - sphereRadius * sphereRadius;
            float disc = b * b - c;
            if (disc >= 0f)
            {
                float t = -b - MathF.Sqrt(disc);
                if (t > 0.01f && t < nearest)
                {
                    nearest = t;
                    hit = true;
                    var p = origin + dir * t;
                    normal = Vector3.Normalize(p - sphereCenter);
                    albedo = new Vector3(0.72f, 0.45f, 0.38f);
                }
            }

            depth = hit ? nearest : 0f;
            return hit;
        }

        /// <summary>
        /// Restores the last GPU-origin G-buffer into the bridge without a new Vulkan readback.
        /// Returns false when no resident buffer exists.
        /// </summary>
        public bool TryRestoreResidentGBuffer()
        {
            if (!_residentGi.HasResidentGBuffer)
                return false;
            _residentGi.CopyResidentTo(_gbuffer);
            _gbufferVersion++;
            _lastFillWasConstants = false;
            LastFillMode = GiGBufferFillMode.GpuResident;
            return true;
        }

        /// <summary>Ingests a Vulkan G-buffer snapshot into the resident GI store.</summary>
        public void IngestGpuSnapshot(GDNN.Rendering.Engine.GBufferSnapshot snapshot)
        {
            ArgumentNullException.ThrowIfNull(snapshot);
            FillGBufferComplete(
                snapshot.Depth,
                snapshot.Normals,
                snapshot.Albedo,
                snapshot.Emissive,
                snapshot.Velocity,
                snapshot.MaterialProps,
                snapshot.Specular);
            _residentGi.UpdateFromSnapshot(snapshot);
            _lastFillWasConstants = false;
            LastFillMode = GiGBufferFillMode.GpuReadback;
        }

        /// <summary>
        /// Clears cached GI output. Call when the scene changes through a path
        /// not tracked by camera, lights, or G-Buffer version.
        /// </summary>
        public void InvalidateGICache()
        {
            _lastStateHash = null;
            _cachedIrradiance = null;
        }

        /// <summary>
        /// Applies LLM-parsed lighting/fog parameters to the L-DNN light list
        /// and volumetric config, then invalidates the GI cache.
        /// </summary>
        public void ApplyLlmLighting(LightingParams parameters)
        {
            ArgumentNullException.ThrowIfNull(parameters);
            if (!_initialized || _ldnn == null || _config == null)
                return;

            SetLights(LlmSceneApplicator.ApplyLighting(parameters));
            _config.VolumeFogConfig = LlmSceneApplicator.ApplyFog(parameters, _config.VolumeFogConfig);
            _ldnn.Config.VolumeFogConfig = _config.VolumeFogConfig;
            _ldnn.Volumetrics?.Initialize(_config.VolumeFogConfig, _width, _height);
            InvalidateGICache();
        }

        /// <summary>
        /// Hash of state that affects GI: dimensions, G-Buffer version, camera, and lights.
        /// </summary>
        private int ComputeStateHash()
        {
            var hash = new HashCode();
            hash.Add(_width);
            hash.Add(_height);
            hash.Add(_gbufferVersion);
            hash.Add(_cameraState.ViewMatrix);
            hash.Add(_cameraState.ProjectionMatrix);
            hash.Add(_cameraState.Position);

            hash.Add(_lights.Count);
            foreach (var light in _lights)
            {
                hash.Add(light.Type);
                hash.Add(light.Position);
                hash.Add(light.Direction);
                hash.Add(light.Color);
                hash.Add(light.Intensity);
                hash.Add(light.Range);
                hash.Add(light.ShadowMethod);
            }

            return hash.ToHashCode();
        }

        /// <summary>Runs one L-DNN GI frame and returns per-pixel irradiance (may be cache-hit).</summary>
        public Vector3[,] RenderGI()
        {
            if (!_initialized || _ldnn == null)
                return new Vector3[_width, _height];

            if (EnableStaticSceneCache)
            {
                int stateHash = ComputeStateHash();
                if (_cachedIrradiance != null && _lastStateHash == stateHash)
                {
                    StaticCacheHits++;
                    return _cachedIrradiance;
                }
                _lastStateHash = stateHash;
            }

            // Always run the full Hybrid L-DNN stack (SSGI + Radiance Cascades + neural + specular).
            // The resident-only SSGI shortcut skipped cascades and killed visual quality.
            var context = new RenderContext
            {
                FrameIndex = _ldnn.FrameIndex,
                RenderWidth = _width,
                RenderHeight = _height,
                ResourcePool = new Dictionary<string, object>(),
                ConstantBufferData = new Dictionary<string, float>()
            };

            _ldnn.RenderFrame(_gbuffer, _cameraState, _lights, context);

            var irradiance = new Vector3[_width, _height];
            var giField = _ldnn.GetLastIrradianceField(_width, _height);
            if (giField != null && giField.GetLength(0) == _width && giField.GetLength(1) == _height)
            {
                for (int y = 0; y < _height; y++)
                    for (int x = 0; x < _width; x++)
                        irradiance[x, y] = giField[x, y];
            }
            else if (!_lastFillWasConstants && _residentGi.HasResidentGBuffer)
            {
                irradiance = _residentGi.ComputeResidentIrradiance(_ldnn, _cameraState, _lights);
            }
            else
            {
                for (int y = 0; y < _height; y++)
                    for (int x = 0; x < _width; x++)
                        irradiance[x, y] = _gbuffer.Albedo[y * _width + x] * 0.3f;
            }

            if (_lastFillWasConstants)
                _residentGi.MarkConstantFallback();
            else
                _residentGi.UpdateFromGBuffer(_gbuffer);

            _cachedIrradiance = irradiance;
            return irradiance;
        }

        /// <summary>
        /// Returns screen-space ambient occlusion at a pixel using the industrial SSAO kernel
        /// (hemisphere sampling against the G-buffer depth/normals).
        /// </summary>
        public float ComputeAO(int px, int py)
        {
            if (px < 0 || px >= _width || py < 0 || py >= _height)
                return 1.0f;

            var ao = EnsureAoBuffer();
            if (ao == null || ao.Length != _width * _height)
                return SampleLocalDepthAO(px, py);

            return Math.Clamp(ao[py * _width + px], 0f, 1f);
        }

        /// <summary>
        /// Ensures the industrial SSAO buffer is up to date and returns it (row-major, length = width*height).
        /// </summary>
        public float[]? EnsureAoBuffer()
        {
            if (!_initialized || _ldnn == null)
                return null;

            if (_aoGbufferVersion != _gbufferVersion)
            {
                _ldnn.DispatchComputeShaders(
                    "ssao",
                    (_width + 7) / 8,
                    (_height + 7) / 8,
                    1,
                    new Dictionary<string, object>
                    {
                        ["gbuffer"] = _gbuffer,
                        ["camera"] = _cameraState,
                        ["kernelSize"] = 32,
                        ["radius"] = 0.75f
                    });
                _ldnn.DispatchComputeShaders(
                    "blur_ao",
                    (_width + 7) / 8,
                    (_height + 7) / 8,
                    1,
                    new Dictionary<string, object>
                    {
                        ["gbuffer"] = _gbuffer,
                        ["radius"] = 2,
                        ["sigma"] = 0.1f
                    });
                _aoGbufferVersion = _gbufferVersion;
            }

            var ao = _ldnn.AmbientOcclusion?.AOBuffer;
            if (ao == null || ao.Length != _width * _height)
                return null;
            return ao;
        }

        /// <summary>Screen-space AO as a dense WxH field for GPU upload (1 = unoccluded).</summary>
        public float[,] GetAoField()
        {
            var field = new float[_width, _height];
            var ao = EnsureAoBuffer();
            if (ao == null)
            {
                for (int y = 0; y < _height; y++)
                    for (int x = 0; x < _width; x++)
                        field[x, y] = SampleLocalDepthAO(x, y);
                return field;
            }

            for (int y = 0; y < _height; y++)
            {
                int row = y * _width;
                for (int x = 0; x < _width; x++)
                    field[x, y] = Math.Clamp(ao[row + x], 0f, 1f);
            }
            return field;
        }

        /// <summary>
        /// Collapses L-DNN froxel in-scatter to a screen-sized RGB field for the deferred fog composite.
        /// Uses a sparse sample grid then bilinear-fills for speed.
        /// </summary>
        public Vector3[,] GetFogInScatterField()
        {
            var field = new Vector3[_width, _height];
            var vol = _ldnn?.Volumetrics;
            if (!_initialized || vol == null || _width <= 0 || _height <= 0)
                return field;

            int step = Math.Max(2, Math.Max(_width, _height) / 64);
            for (int y = 0; y < _height; y += step)
            {
                for (int x = 0; x < _width; x += step)
                {
                    float u = (x + 0.5f) / _width;
                    float v = (y + 0.5f) / _height;
                    var screen = new Vector2(u, v);
                    var viewDir = Vector3.Normalize(
                        _cameraState.Forward
                        + _cameraState.Right * ((u - 0.5f) * 2f * _cameraState.AspectRatio * MathF.Tan(_cameraState.FieldOfView * 0.5f * MathF.PI / 180f))
                        + _cameraState.Up * ((0.5f - v) * 2f * MathF.Tan(_cameraState.FieldOfView * 0.5f * MathF.PI / 180f)));
                    var scatter = vol.IntegrateVolumeScattering(_cameraState, screen, viewDir, _lights);
                    int x1 = Math.Min(_width, x + step);
                    int y1 = Math.Min(_height, y + step);
                    for (int yy = y; yy < y1; yy++)
                        for (int xx = x; xx < x1; xx++)
                            field[xx, yy] = scatter;
                }
            }

            return field;
        }

        /// <summary>Fast local depth-based AO fallback when the full SSAO pass is unavailable.</summary>
        private float SampleLocalDepthAO(int px, int py)
        {
            if (_gbuffer.Depth == null || _gbuffer.Normals == null)
                return 1.0f;

            int idx = py * _width + px;
            float depth = _gbuffer.Depth[idx];
            if (depth <= 0f)
                return 1.0f;

            Vector3 normal = _gbuffer.Normals[idx];
            float occlusion = 0f;
            int samples = 0;
            ReadOnlySpan<int> ox = stackalloc int[] { -2, -1, 1, 2, -2, 2, -1, 1 };
            ReadOnlySpan<int> oy = stackalloc int[] { -2, -1, 1, 2, 1, -1, 2, -2 };
            for (int i = 0; i < ox.Length; i++)
            {
                int sx = Math.Clamp(px + ox[i], 0, _width - 1);
                int sy = Math.Clamp(py + oy[i], 0, _height - 1);
                float sampleDepth = _gbuffer.Depth[sy * _width + sx];
                float delta = depth - sampleDepth;
                if (delta > 0.002f && delta < 0.5f)
                {
                    float range = 1f - delta / 0.5f;
                    float ndot = MathF.Max(0f, Vector3.Dot(normal, _gbuffer.Normals[sy * _width + sx]));
                    occlusion += range * (0.5f + 0.5f * ndot);
                }
                samples++;
            }

            return samples == 0 ? 1f : Math.Clamp(1f - occlusion / samples, 0f, 1f);
        }

        private static LDNNConfig CreateDefaultConfig()
        {
            return new LDNNConfig
            {
                QualityMode = LDNNQualityMode.HybridRT,
                // Hybrid = SSGI + Radiance Cascades + neural refine + specular (L-DNN full stack).
                GIComputationMode = GIComputationMode.Hybrid,
                CascadeConfig = new CascadeConfig
                {
                    NumLevels = 6,
                    BaseResolution = CascadeResolution.High256,
                    AllocationStrategy = CascadeAllocationStrategy.Logarithmic,
                    MemoryBudgetBytes = 512 * 1024 * 1024,
                    TimeBudgetMs = 10.0f,
                    EnableTemporalAccumulation = true,
                    TemporalBlendFactor = 0.10f,
                    SpatialFilterRadius = 3,
                    DistanceScale = 1.0f,
                    AngularCoverage = 1.0f
                },
                DenoiseConfig = new DenoiseConfig
                {
                    PrimaryDenoiser = DenoiserType.SpatialBilateral,
                    SecondaryDenoiser = DenoiserType.TemporalAccumulation,
                    SpatialRadius = 4,
                    TemporalFrames = 8,
                    NormalThreshold = 0.45f,
                    DepthThreshold = 0.01f,
                    LuminanceThreshold = 0.25f,
                    Strength = 0.9f,
                    Iterations = 3,
                    UseHalfRes = false
                },
                TemporalConfig = new TemporalConfig
                {
                    Mode = TemporalFilterMode.VarianceClipping,
                    BaseBlendFactor = 0.08f,
                    MaxHistoryLength = 24,
                    VarianceClippingStrength = 0.55f,
                    DisocclusionThreshold = 0.25f,
                    ResponseSpeed = 0.4f,
                    AccumulationSpeed = 0.12f,
                    UseYCoCg = true
                },
                VolumeFogConfig = new VolumeFogConfig
                {
                    MaxDensity = 0.035f,
                    HeightFalloff = 0.35f,
                    ReferenceHeight = 0.0f,
                    FogColor = new Vector3(0.65f, 0.75f, 0.95f),
                    Anisotropy = 0.45f,
                    StartDistance = 3.0f,
                    MaxDistance = 180.0f,
                    DepthSlices = 96,
                    GridResolutionXY = 48,
                    TemporalReprojectionStrength = 0.85f,
                    NoiseScale = 0.04f,
                    NoiseSpeed = 0.12f,
                    ShadowIntensity = 0.65f,
                    EnableClouds = true,
                    CloudAltitude = 90.0f,
                    CloudThickness = 45.0f,
                    CloudCoverage = 0.4f,
                    CloudDensityScale = 2.2f,
                    CloudNoiseScale = 0.018f
                },
                SamplingMode = HemisphereSamplingMode.CosineWeighted,
                MaxPathDepth = 6,
                ReferenceSamplesPerPixel = 32,
                EnableNeuralTemporal = true,
                NeuralLearningRate = 0.001f,
                NeuralBatchSize = 32,
                NeuralNetworkProfile = NeuralNetworkProfile.Full,
                // Teacher path-trace every frame kills realtime — keep off the present path.
                EnableOnlineTeacherTraining = false,
                TeacherPixelStride = 16,
                TeacherSamplesPerPixel = 1,
                EnableNeuralSpecular = true
            };
        }

        /// <summary>Underlying L-DNN renderer (tests and tooling).</summary>
        public LDNNRenderer Renderer => _ldnn;

        /// <summary>Shuts down the L-DNN renderer and releases GPU resources.</summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_initialized)
            {
                _ldnn?.Shutdown();
                _ldnn = null;
            }
        }
    }
}
