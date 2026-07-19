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
    /// Camera state for view and projection information.
    /// </summary>
    public class CameraState
    {
        /// <summary>View matrix (world to view space).</summary>
        public Matrix4x4 ViewMatrix { get; set; }
        /// <summary>Projection matrix (view to clip space).</summary>
        public Matrix4x4 ProjectionMatrix { get; set; }
        /// <summary>View-projection combined matrix.</summary>
        public Matrix4x4 ViewProjectionMatrix { get; set; }
        /// <summary>Inverse view matrix.</summary>
        public Matrix4x4 InverseViewMatrix { get; set; }
        /// <summary>Inverse projection matrix.</summary>
        public Matrix4x4 InverseProjectionMatrix { get; set; }
        /// <summary>Inverse view-projection matrix.</summary>
        public Matrix4x4 InverseViewProjectionMatrix { get; set; }
        /// <summary>Camera position in world space.</summary>
        public Vector3 Position { get; set; }
        /// <summary>Camera forward direction.</summary>
        public Vector3 Forward { get; set; }
        /// <summary>Camera right direction.</summary>
        public Vector3 Right { get; set; }
        /// <summary>Camera up direction.</summary>
        public Vector3 Up { get; set; }
        /// <summary>Field of view in radians.</summary>
        public float FieldOfView { get; set; }
        /// <summary>Aspect ratio (width / height).</summary>
        public float AspectRatio { get; set; }
        /// <summary>Near clipping plane distance.</summary>
        public float NearPlane { get; set; }
        /// <summary>Far clipping plane distance.</summary>
        public float FarPlane { get; set; }
        /// <summary>Previous frame view-projection matrix for velocity computation.</summary>
        public Matrix4x4 PreviousViewProjectionMatrix { get; set; }
        /// <summary>Jitter offset for temporal anti-aliasing.</summary>
        public Vector2 JitterOffset { get; set; }

        /// <summary>
        /// Recomputes derived matrices from view and projection.
        /// </summary>
        public void Recompute()
        {
            ViewProjectionMatrix = ViewMatrix * ProjectionMatrix;
            Matrix4x4.Invert(ViewMatrix, out var invView);
            Matrix4x4.Invert(ProjectionMatrix, out var invProj);
            Matrix4x4.Invert(ViewProjectionMatrix, out var invVP);
            InverseViewMatrix = invView;
            InverseProjectionMatrix = invProj;
            InverseViewProjectionMatrix = invVP;
            Forward = new Vector3(ViewMatrix.M13, ViewMatrix.M23, ViewMatrix.M33);
            Right = new Vector3(ViewMatrix.M11, ViewMatrix.M21, ViewMatrix.M31);
            Up = new Vector3(ViewMatrix.M12, ViewMatrix.M22, ViewMatrix.M32);
        }

        /// <summary>
        /// Projects a world-space point to screen-space coordinates.
        /// </summary>
        public Vector3 ProjectToScreen(Vector3 worldPos)
        {
            Vector4 clip = Vector4.Transform(new Vector4(worldPos.X, worldPos.Y, worldPos.Z, 1.0f), ViewProjectionMatrix);
            if (MathF.Abs(clip.W) < 0.0001f)
                return new Vector3(-1, -1, -1);
            float ndcX = clip.X / clip.W;
            float ndcY = clip.Y / clip.W;
            float ndcZ = clip.Z / clip.W;
            return new Vector3((ndcX + 1.0f) * 0.5f, (1.0f - ndcY) * 0.5f, ndcZ);
        }

        /// <summary>
        /// Unprojects a screen-space point to world-space.
        /// </summary>
        public Vector3 UnprojectFromScreen(Vector3 screenPos)
        {
            float ndcX = screenPos.X * 2.0f - 1.0f;
            float ndcY = 1.0f - screenPos.Y * 2.0f;
            Vector4 clip = new Vector4(ndcX, ndcY, screenPos.Z, 1.0f);
            Vector4 world = Vector4.Transform(clip, InverseViewProjectionMatrix);
            return world.XYZ();
        }

        /// <summary>
        /// Generates a ray from screen-space coordinates.
        /// </summary>
        public (Vector3 Origin, Vector3 Direction) GenerateRay(float u, float v)
        {
            float tanHalfFov = MathF.Tan(FieldOfView * 0.5f);
            float ndcX = (2.0f * u - 1.0f) * AspectRatio * tanHalfFov;
            float ndcY = (1.0f - 2.0f * v) * tanHalfFov;
            Vector3 dir = Vector3.Normalize(Right * ndcX + Up * ndcY + Forward);
            return (Position, dir);
        }
    }

    /// <summary>
    /// Geometry buffer containing screen-space data.
    /// </summary>
    public class GBuffer
    {
        /// <summary>Depth buffer (linear view-space depth).</summary>
        public required float[] Depth { get; set; }
        /// <summary>Normal buffer (world-space normals, XYZ).</summary>
        public required Vector3[] Normals { get; set; }
        /// <summary>Albedo buffer (base color).</summary>
        public required Vector3[] Albedo { get; set; }
        /// <summary>Velocity buffer (screen-space motion vectors).</summary>
        public required Vector2[] Velocity { get; set; }
        /// <summary>Material buffer (roughness, metallic packed).</summary>
        public required Vector4[] MaterialProps { get; set; }
        /// <summary>Specular buffer.</summary>
        public required Vector3[] Specular { get; set; }
        /// <summary>Emissive buffer.</summary>
        public required Vector3[] Emissive { get; set; }
        /// <summary>Width of the G-Buffer.</summary>
        public int Width { get; set; }
        /// <summary>Height of the G-Buffer.</summary>
        public int Height { get; set; }

        /// <summary>
        /// Gets the linear index for a pixel coordinate.
        /// </summary>
        public int GetIndex(int x, int y) => y * Width + x;

        /// <summary>
        /// Gets the linear index for a pixel coordinate.
        /// </summary>
        public int GetIndex(Vector2Int pos) => pos.Y * Width + pos.X;

        /// <summary>
        /// Gets the G-Buffer sample at a pixel coordinate.
        /// </summary>
        public GBufferSample GetSample(int x, int y)
        {
            int idx = GetIndex(x, y);
            return new GBufferSample
            {
                ScreenPosition = new Vector2(x, y),
                Depth = Depth[idx],
                WorldPosition = Vector3.Zero,
                Normal = Normals[idx],
                Albedo = Albedo[idx],
                Specular = Specular[idx],
                Roughness = MaterialProps[idx].X,
                Metallic = MaterialProps[idx].Y,
                Velocity = Velocity[idx],
                MaterialID = (int)MaterialProps[idx].Z,
                IsDynamic = Velocity[idx].LengthSquared() > 0.0001f,
                IsTranslucent = MaterialProps[idx].W > 0.5f
            };
        }

        /// <summary>
        /// Returns the total number of pixels in the buffer.
        /// </summary>
        public int GetPixelCount() => Width * Height;
    }

    /// <summary>
    /// Rendering context providing access to GPU resources.
    /// </summary>
    public class RenderContext
    {
        /// <summary>Command list for GPU command submission.</summary>
        public required object CommandList { get; set; }
        /// <summary>Current frame index.</summary>
        public int FrameIndex { get; set; }
        /// <summary>Render target width.</summary>
        public int RenderWidth { get; set; }
        /// <summary>Render target height.</summary>
        public int RenderHeight { get; set; }
        /// <summary>Available GPU memory in bytes.</summary>
        public long AvailableGPUMemory { get; set; }
        /// <summary>GPU frame time in milliseconds.</summary>
        public float GPUFrameTimeMs { get; set; }
        /// <summary>Resource pool for texture/buffer allocation.</summary>
        public Dictionary<string, object> ResourcePool { get; set; } = new();
        /// <summary>Global constant buffer data.</summary>
        public Dictionary<string, float> ConstantBufferData { get; set; } = new();

        /// <summary>
        /// Gets or creates a named resource from the pool.
        /// </summary>
        public T GetOrCreateResource<T>(string name, Func<T> factory)
        {
            if (!ResourcePool.TryGetValue(name, out var resource))
            {
                resource = factory();
                ResourcePool[name] = resource;
            }
            return (T)resource;
        }
    }

    /// <summary>
    /// Performance telemetry data for a frame.
    /// </summary>
    public class FrameTelemetry
    {
        /// <summary>Total frame time in milliseconds.</summary>
        public float TotalFrameTimeMs { get; set; }
        /// <summary>Cascade rendering time in milliseconds.</summary>
        public float CascadeRenderTimeMs { get; set; }
        /// <summary>Neural prediction time in milliseconds.</summary>
        public float NeuralPredictionTimeMs { get; set; }
        /// <summary>Denoising time in milliseconds.</summary>
        public float DenoisingTimeMs { get; set; }
        /// <summary>Temporal accumulation time in milliseconds.</summary>
        public float TemporalAccumulationTimeMs { get; set; }
        /// <summary>Volumetric lighting time in milliseconds.</summary>
        public float VolumetricLightingTimeMs { get; set; }
        /// <summary>Ambient occlusion time in milliseconds.</summary>
        public float AmbientOcclusionTimeMs { get; set; }
        /// <summary>Screen-space GI time in milliseconds.</summary>
        public float ScreenSpaceGITimeMs { get; set; }
        /// <summary>Reference path tracing time in milliseconds.</summary>
        public float ReferencePathTraceTimeMs { get; set; }
        /// <summary>Total GPU memory used in bytes.</summary>
        public long GPUMemoryUsedBytes { get; set; }
        /// <summary>Number of rays traced this frame.</summary>
        public int RaysTraced { get; set; }
        /// <summary>Number of probes updated this frame.</summary>
        public int ProbesUpdated { get; set; }
        /// <summary>Number of denoiser iterations.</summary>
        public int DenoiserIterations { get; set; }
        /// <summary>Frame timestamp.</summary>
        public Stopwatch Timestamp { get; set; } = new();

        /// <summary>
        /// Resets all telemetry values for a new frame.
        /// </summary>
        public void Reset()
        {
            TotalFrameTimeMs = 0;
            CascadeRenderTimeMs = 0;
            NeuralPredictionTimeMs = 0;
            DenoisingTimeMs = 0;
            TemporalAccumulationTimeMs = 0;
            VolumetricLightingTimeMs = 0;
            AmbientOcclusionTimeMs = 0;
            ScreenSpaceGITimeMs = 0;
            ReferencePathTraceTimeMs = 0;
            GPUMemoryUsedBytes = 0;
            RaysTraced = 0;
            ProbesUpdated = 0;
            DenoiserIterations = 0;
        }
    }

    /// <summary>
    /// Global configuration for the L-DNN system.
    /// </summary>
    /// <summary>
    /// Taille du réseau de prédiction d'irradiance : les scènes simples peuvent
    /// utiliser des variantes plus petites (moins de coût par pixel).
    /// </summary>
    public enum NeuralNetworkProfile
    {
        /// <summary>32 → 32 → 32 → 3 : scènes très simples, coût minimal.</summary>
        Tiny,
        /// <summary>32 → 64 → 64 → 3 : compromis qualité/perfs.</summary>
        Small,
        /// <summary>32 → 128 → 128 → 3 : architecture complète (défaut).</summary>
        Full
    }

    public class LDNNConfig
    {
        /// <summary>Quality mode for rendering.</summary>
        public LDNNQualityMode QualityMode { get; set; } = LDNNQualityMode.HybridRT;
        /// <summary>GI computation mode.</summary>
        public GIComputationMode GIComputationMode { get; set; } = GIComputationMode.Hybrid;
        /// <summary>Cascade configuration.</summary>
        public CascadeConfig CascadeConfig { get; set; } = new()
        {
            NumLevels = 6,
            BaseResolution = CascadeResolution.High256,
            AllocationStrategy = CascadeAllocationStrategy.Adaptive,
            MemoryBudgetBytes = 512 * 1024 * 1024,
            TimeBudgetMs = 4.0f,
            EnableTemporalAccumulation = true,
            TemporalBlendFactor = 0.9f,
            SpatialFilterRadius = 2,
            EnableAdaptiveAllocation = true,
            DistanceScale = 1.0f,
            AngularCoverage = MathF.PI / 3.0f
        };
        /// <summary>Denoiser configuration.</summary>
        public DenoiseConfig DenoiseConfig { get; set; } = new()
        {
            PrimaryDenoiser = DenoiserType.TemporalAccumulation,
            SecondaryDenoiser = DenoiserType.SpatialBilateral,
            SpatialRadius = 3,
            TemporalFrames = 8,
            NormalThreshold = 0.3f,
            DepthThreshold = 0.01f,
            LuminanceThreshold = 0.8f,
            Strength = 0.8f,
            Iterations = 2,
            UseHalfRes = true
        };
        /// <summary>Temporal configuration.</summary>
        public TemporalConfig TemporalConfig { get; set; } = new()
        {
            Mode = TemporalFilterMode.VarianceClipping,
            BaseBlendFactor = 0.9f,
            MaxHistoryLength = 32,
            VarianceClippingStrength = 0.5f,
            DisocclusionThreshold = 0.1f,
            ResponseSpeed = 0.1f,
            AccumulationSpeed = 0.95f,
            UseYCoCg = true,
            ReprojectionJitter = 0.5f,
            MaxVelocity = 100.0f
        };
        /// <summary>Volumetric fog configuration.</summary>
        public VolumeFogConfig VolumeFogConfig { get; set; } = new()
        {
            MaxDensity = 0.02f,
            HeightFalloff = 0.1f,
            ReferenceHeight = 50.0f,
            FogColor = new Vector3(0.7f, 0.8f, 0.9f),
            Anisotropy = 0.5f,
            StartDistance = 5.0f,
            MaxDistance = 500.0f,
            DepthSlices = 128,
            GridResolutionXY = 160,
            TemporalReprojectionStrength = 0.75f,
            NoiseScale = 0.01f,
            NoiseSpeed = 0.5f,
            ShadowIntensity = 0.8f
        };
        /// <summary>Hemisphere sampling mode.</summary>
        public HemisphereSamplingMode SamplingMode { get; set; } = HemisphereSamplingMode.CosineWeighted;
        /// <summary>Maximum path depth for path tracing.</summary>
        public int MaxPathDepth { get; set; } = 8;
        /// <summary>Number of samples per pixel for reference rendering.</summary>
        public int ReferenceSamplesPerPixel { get; set; } = 256;
        /// <summary>Whether to enable neural temporal filtering.</summary>
        public bool EnableNeuralTemporal { get; set; } = false;
        /// <summary>Learning rate for online neural training.</summary>
        public float NeuralLearningRate { get; set; } = 0.001f;
        /// <summary>Batch size for neural training.</summary>
        public int NeuralBatchSize { get; set; } = 64;
        /// <summary>Taille du réseau de prédiction d'irradiance.</summary>
        public NeuralNetworkProfile NeuralNetworkProfile { get; set; } = NeuralNetworkProfile.Full;
        /// <summary>
        /// Sparse path-tracer teacher: sample pixels each frame and train
        /// <see cref="NeuralPredictiveIrradiance"/> online (Hybrid / NeuralPredictive).
        /// </summary>
        public bool EnableOnlineTeacherTraining { get; set; }
        /// <summary>Pixel stride for teacher sampling (16 = ~1/256 of the frame).</summary>
        public int TeacherPixelStride { get; set; } = 16;
        /// <summary>Path-tracer samples per teacher pixel (keep small for real-time).</summary>
        public int TeacherSamplesPerPixel { get; set; } = 1;
        /// <summary>Enable dedicated neural specular reflection/refraction pass.</summary>
        public bool EnableNeuralSpecular { get; set; } = true;
    }

    /// <summary>
    /// Budget allocation for cascade resources.
    /// </summary>
    public record CascadeBudget
    {
        /// <summary>Total time budget in milliseconds.</summary>
        public float TotalTimeBudgetMs { get; init; }
        /// <summary>Total memory budget in bytes.</summary>
        public long TotalMemoryBudgetBytes { get; init; }
        /// <summary>Time allocated per cascade level.</summary>
        public required float[] TimePerLevelMs { get; init; }
        /// <summary>Memory allocated per cascade level.</summary>
        public required long[] MemoryPerLevelBytes { get; init; }
        /// <summary>Ray count allocated per cascade level.</summary>
        public required int[] RaysPerLevel { get; init; }
        /// <summary>Resolution per cascade level.</summary>
        public required int[] ResolutionPerLevel { get; init; }
        /// <summary>Whether allocation was successful within budget.</summary>
        public bool WithinBudget { get; init; }
    }
}
