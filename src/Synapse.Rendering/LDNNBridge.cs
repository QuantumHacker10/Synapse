using System;
using System.Collections.Generic;
using System.Numerics;
using GDNN.Lighting.LDNN;
using GDNN.RHI.Vulkan;

namespace GDNN.Rendering.Bridge
{
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

        public int FrameIndex => _ldnn?.FrameIndex ?? 0;
        public bool IsInitialized => _initialized;

        public LDNNBridge(int width, int height)
        {
            _width = width;
            _height = height;
            _gbuffer = new GBuffer { Width = width, Height = height };
            _cameraState = new CameraState();
            _lights = new List<LightConfig>();
        }

        public void Initialize(LDNNConfig config = null)
        {
            _config = config ?? CreateDefaultConfig();
            _ldnn = new LDNNRenderer();
            _ldnn.Initialize(_config, _width, _height);
            _initialized = true;
        }

        public void Resize(int width, int height)
        {
            _width = width;
            _height = height;
            _gbuffer.Width = width;
            _gbuffer.Height = height;
            _gbuffer.Depth = new float[width * height];
            _gbuffer.Normals = new Vector3[width * height];
            _gbuffer.Albedo = new Vector3[width * height];
            _gbuffer.Velocity = new Vector2[width * height];
            _gbuffer.MaterialProps = new Vector4[width * height];
            _gbuffer.Specular = new Vector3[width * height];
            _gbuffer.Emissive = new Vector3[width * height];
        }

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

        public void SetLights(List<LightConfig> lights)
        {
            _lights = lights ?? new List<LightConfig>();
        }

        public void AddDirectionalLight(Vector3 direction, Vector3 color, float intensity)
        {
            _lights.Add(new LightConfig
            {
                Type = LightType.Directional,
                Direction = Vector3.Normalize(direction),
                Color = color,
                Intensity = intensity,
                ShadowMethod = ShadowMethod.ShadowMap,
                ShadowBias = 0.005f,
                ShadowSamples = 16,
                Importance = 1.0f
            });
        }

        public void AddPointLight(Vector3 position, Vector3 color, float intensity, float range)
        {
            _lights.Add(new LightConfig
            {
                Type = LightType.Point,
                Position = position,
                Color = color,
                Intensity = intensity,
                Range = range,
                ShadowMethod = ShadowMethod.ShadowMap,
                ShadowBias = 0.005f,
                ShadowSamples = 8,
                Importance = 0.8f
            });
        }

        public void FillGBuffer(
            float[] depthData,
            Vector3[] normalData,
            Vector3[] albedoData,
            Vector3[] emissiveData)
        {
            if (depthData != null && depthData.Length == _gbuffer.Depth.Length)
                Array.Copy(depthData, _gbuffer.Depth, depthData.Length);
            if (normalData != null && normalData.Length == _gbuffer.Normals.Length)
                Array.Copy(normalData, _gbuffer.Normals, normalData.Length);
            if (albedoData != null && albedoData.Length == _gbuffer.Albedo.Length)
                Array.Copy(albedoData, _gbuffer.Albedo, albedoData.Length);
            if (emissiveData != null && emissiveData.Length == _gbuffer.Emissive.Length)
                Array.Copy(emissiveData, _gbuffer.Emissive, emissiveData.Length);
        }

        public void FillGBufferFromConstants(
            float fillDepth, Vector3 fillNormal, Vector3 fillAlbedo)
        {
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
        }

        public Vector3[,] RenderGI()
        {
            if (!_initialized || _ldnn == null)
                return new Vector3[_width, _height];

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
            if (_ldnn.Analytics != null)
            {
                for (int y = 0; y < _height; y++)
                    for (int x = 0; x < _width; x++)
                        irradiance[x, y] = _gbuffer.Albedo[y * _width + x] * 0.3f;
            }

            return irradiance;
        }

        public float ComputeAO(int px, int py)
        {
            if (px < 0 || px >= _width || py < 0 || py >= _height) return 1.0f;
            return 1.0f;
        }

        private static LDNNConfig CreateDefaultConfig()
        {
            return new LDNNConfig
            {
                QualityMode = LDNNQualityMode.HybridRT,
                GIComputationMode = GIComputationMode.Hybrid,
                CascadeConfig = new CascadeConfig
                {
                    NumLevels = 4,
                    BaseResolution = CascadeResolution.Medium128,
                    AllocationStrategy = CascadeAllocationStrategy.Logarithmic,
                    MemoryBudgetBytes = 256 * 1024 * 1024,
                    TimeBudgetMs = 4.0f,
                    EnableTemporalAccumulation = true,
                    TemporalBlendFactor = 0.1f,
                    SpatialFilterRadius = 2,
                    DistanceScale = 1.0f,
                    AngularCoverage = 1.0f
                },
                DenoiseConfig = new DenoiseConfig
                {
                    PrimaryDenoiser = DenoiserType.SpatialBilateral,
                    SecondaryDenoiser = DenoiserType.TemporalAccumulation,
                    SpatialRadius = 3,
                    TemporalFrames = 4,
                    NormalThreshold = 0.5f,
                    DepthThreshold = 0.01f,
                    LuminanceThreshold = 0.3f,
                    Strength = 0.8f,
                    Iterations = 2,
                    UseHalfRes = false
                },
                TemporalConfig = new TemporalConfig
                {
                    Mode = TemporalFilterMode.VarianceClipping,
                    BaseBlendFactor = 0.1f,
                    MaxHistoryLength = 16,
                    VarianceClippingStrength = 0.5f,
                    DisocclusionThreshold = 0.3f,
                    ResponseSpeed = 0.5f,
                    AccumulationSpeed = 0.1f,
                    UseYCoCg = true
                },
                VolumeFogConfig = new VolumeFogConfig
                {
                    MaxDensity = 0.02f,
                    HeightFalloff = 0.5f,
                    ReferenceHeight = 0.0f,
                    FogColor = new Vector3(0.7f, 0.8f, 1.0f),
                    Anisotropy = 0.3f,
                    StartDistance = 5.0f,
                    MaxDistance = 100.0f,
                    DepthSlices = 64,
                    GridResolutionXY = 16,
                    TemporalReprojectionStrength = 0.8f,
                    NoiseScale = 0.05f,
                    NoiseSpeed = 0.1f,
                    ShadowIntensity = 0.5f
                },
                SamplingMode = HemisphereSamplingMode.CosineWeighted,
                MaxPathDepth = 4,
                ReferenceSamplesPerPixel = 64,
                EnableNeuralTemporal = false,
                NeuralLearningRate = 0.001f,
                NeuralBatchSize = 32
            };
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_initialized)
            {
                _ldnn?.Shutdown();
                _ldnn = null;
            }
        }
    }
}
