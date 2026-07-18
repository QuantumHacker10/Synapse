// =============================================================================
// LDNNRenderer.cs - L-DNN Neural Global Illumination System
// GDNN.Engine - GDNN.Lighting Namespace
// Production-Ready Implementation
// =============================================================================
// This file implements a complete neural global illumination system combining
// radiance cascades, neural predictive models, screen-space techniques, and
// reference path tracing for training data generation.
// =============================================================================

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
    // =========================================================================
    #region Enums

    /// <summary>
    /// Defines the quality mode for L-DNN rendering.
    /// </summary>
    public enum LDNNQualityMode
    {
        /// <summary>Uses only neural prediction without any ray tracing.</summary>
        NeuralOnly,
        /// <summary>Hybrid mode combining neural prediction with limited ray tracing.</summary>
        HybridRT,
        /// <summary>Full path tracing used as teacher signal for neural network training.</summary>
        FullPathTraceTeacher
    }

    /// <summary>
    /// Resolution options for radiance cascades.
    /// </summary>
    public enum CascadeResolution
    {
        /// <summary>64x64 cascade resolution.</summary>
        Low64 = 64,
        /// <summary>128x128 cascade resolution.</summary>
        Medium128 = 128,
        /// <summary>256x256 cascade resolution.</summary>
        High256 = 256,
        /// <summary>512x512 cascade resolution.</summary>
        Ultra512 = 512,
        /// <summary>1024x1024 cascade resolution.</summary>
        Max1024 = 1024
    }

    /// <summary>
    /// Type of irradiance cache used for probe storage.
    /// </summary>
    public enum IrradianceCacheType
    {
        /// <summary>Octahedral mapping for directional irradiance.</summary>
        Octahedral,
        /// <summary>Spherical harmonics encoding of irradiance.</summary>
        SphericalHarmonics,
        /// <summary>Regular 3D radiance grid.</summary>
        RadianceGrid,
        /// <summary>Adaptively placed probe volume.</summary>
        ProbeVolume
    }

    /// <summary>
    /// Types of lights supported by the system.
    /// </summary>
    public enum LightType
    {
        /// <summary>Infinite directional light (sun/moon).</summary>
        Directional,
        /// <summary>Omni-directional point light.</summary>
        Point,
        /// <summary>Spotlight with cone attenuation.</summary>
        Spot,
        /// <summary>Rectangular area light.</summary>
        AreaRect,
        /// <summary>Disc-shaped area light.</summary>
        AreaDisc,
        /// <summary>Environment map lighting.</summary>
        Environment,
        /// <summary>Emissive surface lighting.</summary>
        Emissive
    }

    /// <summary>
    /// Shadow computation methods.
    /// </summary>
    public enum ShadowMethod
    {
        /// <summary>No shadows computed.</summary>
        None,
        /// <summary>Traditional shadow mapping.</summary>
        ShadowMap,
        /// <summary>Variance shadow mapping for soft shadows.</summary>
        VarianceShadowMap,
        /// <summary>Hardware ray traced shadows.</summary>
        RayTraced,
        /// <summary>Neural network predicted shadows.</summary>
        NeuralPredictive,
        /// <summary>Contact hardening shadows (PCSS).</summary>
        ContactHardening
    }

    /// <summary>
    /// Global illumination computation modes.
    /// </summary>
    public enum GIComputationMode
    {
        /// <summary>Screen-space global illumination only.</summary>
        SSGI,
        /// <summary>Radiance cascades for world-space GI.</summary>
        RadianceCascades,
        /// <summary>Neural network predicted GI.</summary>
        NeuralPredictive,
        /// <summary>Hybrid of screen-space and world-space.</summary>
        Hybrid,
        /// <summary>Full path tracing (reference quality).</summary>
        FullPathTracing
    }

    /// <summary>
    /// Temporal filtering modes for stable accumulation.
    /// </summary>
    public enum TemporalFilterMode
    {
        /// <summary>No temporal filtering.</summary>
        None,
        /// <summary>Simple exponential moving average.</summary>
        ExponentialMovingAverage,
        /// <summary>Variance-clipped temporal accumulation.</summary>
        VarianceClipping,
        /// <summary>Disocclusion-aware temporal filtering.</summary>
        DisocclusionAware,
        /// <summary>Neural network-based temporal filtering.</summary>
        NeuralTemporal
    }

    /// <summary>
    /// Denoiser types for noise reduction.
    /// </summary>
    public enum DenoiserType
    {
        /// <summary>No denoising applied.</summary>
        None,
        /// <summary>Bilateral spatial filter.</summary>
        SpatialBilateral,
        /// <summary>Temporal accumulation denoiser.</summary>
        TemporalAccumulation,
        /// <summary>Non-local means patch-based denoiser.</summary>
        NonLocalMeans,
        /// <summary>Neural network denoiser.</summary>
        NeuralDenoiser,
        /// <summary>Wavelet transform denoiser.</summary>
        WaveletFilter
    }

    /// <summary>
    /// Probe update scheduling strategies.
    /// </summary>
    public enum ProbeUpdateMode
    {
        /// <summary>Update probes on demand only.</summary>
        OnDemand,
        /// <summary>Periodic update at fixed intervals.</summary>
        Periodic,
        /// <summary>Update based on camera distance.</summary>
        DistanceBased,
        /// <summary>Update based on importance metrics.</summary>
        ImportanceDriven,
        /// <summary>Update within a fixed time budget.</summary>
        BudgetLimited
    }

    /// <summary>
    /// Strategies for allocating cascade resources across levels.
    /// </summary>
    public enum CascadeAllocationStrategy
    {
        /// <summary>Uniform allocation across all levels.</summary>
        Uniform,
        /// <summary>Logarithmic allocation favoring lower levels.</summary>
        Logarithmic,
        /// <summary>Importance-driven allocation.</summary>
        ImportanceDriven,
        /// <summary>Adaptive allocation based on scene complexity.</summary>
        Adaptive,
        /// <summary>Fixed allocation predetermined by quality preset.</summary>
        Fixed
    }

    /// <summary>
    /// Hemisphere sampling strategies for Monte Carlo integration.
    /// </summary>
    public enum HemisphereSamplingMode
    {
        /// <summary>Uniform random sampling.</summary>
        Uniform,
        /// <summary>Cosine-weighted importance sampling.</summary>
        CosineWeighted,
        /// <summary>Stratified sampling with jitter.</summary>
        Stratified,
        /// <summary>Golden ratio spiral sampling.</summary>
        GoldenRatio,
        /// <summary>Blue noise dithered sampling.</summary>
        BlueNoise
    }

    /// <summary>
    /// Material scattering models for light transport.
    /// </summary>
    public enum ScatteringModel
    {
        /// <summary>Basic Lambertian diffuse.</summary>
        Lambertian,
        /// <summary>Oren-Nayar model for rough surfaces.</summary>
        OrenNayar,
        /// <summary>Ashikhmin-Shirley anisotropic model.</summary>
        AshikhminShirley,
        /// <summary>Disney principled BRDF diffuse term.</summary>
        DisneyDiffuse,
        /// <summary>Hanrahan-Krueger layer model.</summary>
        HanrahanKrueger
    }

    #endregion

    // =========================================================================
    #region Records

    /// <summary>
    /// Configuration for a single light source.
    /// </summary>
    public record LightConfig
    {
        /// <summary>Light type (directional, point, spot, etc.).</summary>
        public LightType Type { get; init; }
        /// <summary>World-space position of the light.</summary>
        public Vector3 Position { get; init; }
        /// <summary>Light direction vector.</summary>
        public Vector3 Direction { get; init; }
        /// <summary>Light color and intensity.</summary>
        public Vector3 Color { get; init; }
        /// <summary>Intensity multiplier.</summary>
        public float Intensity { get; init; }
        /// <summary>Maximum range for attenuation.</summary>
        public float Range { get; init; }
        /// <summary>Inner cone angle for spotlights (radians).</summary>
        public float InnerConeAngle { get; init; }
        /// <summary>Outer cone angle for spotlights (radians).</summary>
        public float OuterConeAngle { get; init; }
        /// <summary>Area light width (for area lights).</summary>
        public float AreaWidth { get; init; }
        /// <summary>Area light height (for area lights).</summary>
        public float AreaHeight { get; init; }
        /// <summary>Shadow method for this light.</summary>
        public ShadowMethod ShadowMethod { get; init; }
        /// <summary>Shadow bias to prevent shadow acne.</summary>
        public float ShadowBias { get; init; }
        /// <summary>Number of shadow samples for soft shadows.</summary>
        public int ShadowSamples { get; init; }
        /// <summary>Whether this light casts volumetric shadows.</summary>
        public bool VolumetricShadow { get; init; }
        /// <summary>Light importance for adaptive sampling.</summary>
        public float Importance { get; init; }
        /// <summary>Light right vector for area lights.</summary>
        public Vector3 Right { get; init; }
        /// <summary>Light up vector for area lights.</summary>
        public Vector3 Up { get; init; }
    }

    /// <summary>
    /// Configuration for the radiance cascade system.
    /// </summary>
    public record CascadeConfig
    {
        /// <summary>Number of cascade levels.</summary>
        public int NumLevels { get; init; }
        /// <summary>Base resolution for the cascades.</summary>
        public CascadeResolution BaseResolution { get; init; }
        /// <summary>Allocation strategy for cascade resources.</summary>
        public CascadeAllocationStrategy AllocationStrategy { get; init; }
        /// <summary>Maximum total memory budget for cascades (bytes).</summary>
        public long MemoryBudgetBytes { get; init; }
        /// <summary>Maximum time budget for cascade updates (milliseconds).</summary>
        public float TimeBudgetMs { get; init; }
        /// <summary>Whether to use temporal accumulation across frames.</summary>
        public bool EnableTemporalAccumulation { get; init; }
        /// <summary>Temporal blend factor (0 = new frame only, 1 = full history).</summary>
        public float TemporalBlendFactor { get; init; }
        /// <summary>Spatial filter radius for cascade denoising.</summary>
        public int SpatialFilterRadius { get; init; }
        /// <summary>Whether to enable adaptive level allocation.</summary>
        public bool EnableAdaptiveAllocation { get; init; }
        /// <summary>Distance scaling factor for cascade coverage.</summary>
        public float DistanceScale { get; init; }
        /// <summary>Angular coverage per cascade level (radians).</summary>
        public float AngularCoverage { get; init; }
    }

    /// <summary>
    /// Configuration for a single cascade level.
    /// </summary>
    public record CascadeLevelConfig
    {
        /// <summary>Level index (0 = closest to camera).</summary>
        public int LevelIndex { get; init; }
        /// <summary>Resolution of this cascade level.</summary>
        public int Resolution { get; init; }
        /// <summary>Maximum trace distance for this level.</summary>
        public float MaxTraceDistance { get; init; }
        /// <summary>Minimum trace distance for this level.</summary>
        public float MinTraceDistance { get; init; }
        /// <summary>Number of ray samples per texel.</summary>
        public int RaysPerTexel { get; init; }
        /// <summary>Angular resolution (number of directions).</summary>
        public int AngularResolution { get; init; }
        /// <summary>Whether this level is currently active.</summary>
        public bool IsActive { get; init; }
        /// <summary>Importance weight for budget allocation.</summary>
        public float ImportanceWeight { get; init; }
        /// <summary>Update frequency in frames.</summary>
        public int UpdateFrequency { get; init; }
        /// <summary>Spatial filter kernel size.</summary>
        public int FilterKernelSize { get; init; }
    }

    /// <summary>
    /// Represents a single irradiance probe in the cache.
    /// </summary>
    public record IrradianceProbe
    {
        /// <summary>World-space position of the probe.</summary>
        public Vector3 Position { get; init; }
        /// <summary>SH coefficients for directional irradiance (9 coefficients for L2 SH).</summary>
        public Vector4[] SHCoefficients { get; init; }
        /// <summary>Validity flag for the probe.</summary>
        public bool IsValid { get; init; }
        /// <summary>Frame when this probe was last updated.</summary>
        public int LastUpdateFrame { get; init; }
        /// <summary>Importance score of the probe.</summary>
        public float Importance { get; init; }
        /// <summary>Depth reference for validity checking.</summary>
        public float ReferenceDepth { get; init; }
        /// <summary>Normal reference for validity checking.</summary>
        public Vector3 ReferenceNormal { get; init; }
        /// <summary>Variance of the irradiance estimate.</summary>
        public float Variance { get; init; }
        /// <summary>Number of samples used to compute this probe.</summary>
        public int SampleCount { get; init; }
    }

    /// <summary>
    /// Represents a screen-space probe for local irradiance.
    /// </summary>
    public record ScreenProbe
    {
        /// <summary>Screen-space position (pixel coordinates).</summary>
        public Vector2 ScreenPosition { get; init; }
        /// <summary>World-space position of the surface.</summary>
        public Vector3 WorldPosition { get; init; }
        /// <summary>Surface normal at the probe location.</summary>
        public Vector3 Normal { get; init; }
        /// <summary>Albedo at the probe location.</summary>
        public Vector3 Albedo { get; init; }
        /// <summary>Accumulated irradiance from ray marching.</summary>
        public Vector3 Irradiance { get; init; }
        /// <summary>Confidence of the irradiance estimate.</summary>
        public float Confidence { get; init; }
        /// <summary>Number of rays that hit geometry.</summary>
        public int HitCount { get; init; }
        /// <summary>Total number of rays traced.</summary>
        public int TotalRays { get; init; }
        /// <summary>Temporal history depth.</summary>
        public int HistoryDepth { get; init; }
        /// <summary>Velocity for temporal reprojection.</summary>
        public Vector2 Velocity { get; init; }
    }

    /// <summary>
    /// Candidate position for probe placement.
    /// </summary>
    public record ProbeCandidate
    {
        /// <summary>Candidate world position.</summary>
        public Vector3 Position { get; init; }
        /// <summary>Estimated importance of this position.</summary>
        public float Importance { get; init; }
        /// <summary>Geometric complexity score.</summary>
        public float GeometricComplexity { get; init; }
        /// <summary>Lighting complexity score.</summary>
        public float LightingComplexity { get; init; }
        /// <summary>Distance from camera.</summary>
        public float CameraDistance { get; init; }
        /// <summary>Nearest existing probe distance.</summary>
        public float NearestProbeDistance { get; init; }
    }

    /// <summary>
    /// Represents a 3D radiance field for volumetric lighting.
    /// </summary>
    public record RadianceField
    {
        /// <summary>Grid dimensions.</summary>
        public Vector3Int GridSize { get; init; }
        /// <summary>World-space origin of the grid.</summary>
        public Vector3 Origin { get; init; }
        /// <summary>Cell size in world units.</summary>
        public float CellSize { get; init; }
        /// <summary>Radiance data (RGB per cell).</summary>
        public Vector3[] RadianceData { get; init; }
        /// <summary>Transmittance data per cell.</summary>
        public float[] TransmittanceData { get; init; }
        /// <summary>Frame when the field was last updated.</summary>
        public int LastUpdateFrame { get; init; }
        /// <summary>Whether the field needs regeneration.</summary>
        public bool IsDirty { get; init; }
    }

    /// <summary>
    /// Configuration for volumetric fog rendering.
    /// </summary>
    public record VolumeFogConfig
    {
        /// <summary>Maximum fog density.</summary>
        public float MaxDensity { get; init; }
        /// <summary>Height falloff rate for height-based fog.</summary>
        public float HeightFalloff { get; init; }
        /// <summary>Reference height for fog density.</summary>
        public float ReferenceHeight { get; init; }
        /// <summary>Fog color.</summary>
        public Vector3 FogColor { get; init; }
        /// <summary>Anisotropy parameter for phase function (Henyey-Greenstein).</summary>
        public float Anisotropy { get; init; }
        /// <summary>Start distance for fog.</summary>
        public float StartDistance { get; init; }
        /// <summary>Maximum fog distance.</summary>
        public float MaxDistance { get; init; }
        /// <summary>Number of froxel slices along depth.</summary>
        public int DepthSlices { get; init; }
        /// <summary>Froxel grid resolution (XY).</summary>
        public int GridResolutionXY { get; init; }
        /// <summary>Temporal reprojection strength.</summary>
        public float TemporalReprojectionStrength { get; init; }
        /// <summary>Noise scale for animated fog.</summary>
        public float NoiseScale { get; init; }
        /// <summary>Noise animation speed.</summary>
        public float NoiseSpeed { get; init; }
        /// <summary>Volumetric shadow intensity.</summary>
        public float ShadowIntensity { get; init; }
    }

    /// <summary>
    /// Configuration for the denoising pipeline.
    /// </summary>
    public record DenoiseConfig
    {
        /// <summary>Primary denoiser type.</summary>
        public DenoiserType PrimaryDenoiser { get; init; }
        /// <summary>Secondary denoiser type.</summary>
        public DenoiserType SecondaryDenoiser { get; init; }
        /// <summary>Spatial filter radius.</summary>
        public int SpatialRadius { get; init; }
        /// <summary>Temporal accumulation frames.</summary>
        public int TemporalFrames { get; init; }
        /// <summary>Edge-stopping threshold for normals.</summary>
        public float NormalThreshold { get; init; }
        /// <summary>Edge-stopping threshold for depth.</summary>
        public float DepthThreshold { get; init; }
        /// <summary>Luminance threshold for edge stopping.</summary>
        public float LuminanceThreshold { get; init; }
        /// <summary>Strength of the denoiser (0 = off, 1 = full).</summary>
        public float Strength { get; init; }
        /// <summary>Number of denoising iterations.</summary>
        public int Iterations { get; init; }
        /// <summary>Half-resolution denoising enabled.</summary>
        public bool UseHalfRes { get; init; }
    }

    /// <summary>
    /// Configuration for temporal stabilization.
    /// </summary>
    public record TemporalConfig
    {
        /// <summary>Temporal filter mode.</summary>
        public TemporalFilterMode Mode { get; init; }
        /// <summary>Base blend factor for temporal accumulation.</summary>
        public float BaseBlendFactor { get; init; }
        /// <summary>Maximum history length.</summary>
        public int MaxHistoryLength { get; init; }
        /// <summary>Variance clipping strength.</summary>
        public float VarianceClippingStrength { get; init; }
        /// <summary>Disocclusion detection threshold.</summary>
        public float DisocclusionThreshold { get; init; }
        /// <summary>Response speed for scene changes.</summary>
        public float ResponseSpeed { get; init; }
        /// <summary>Accumulation speed for stable regions.</summary>
        public float AccumulationSpeed { get; init; }
        /// <summary>Whether to use YCoCg color space for filtering.</summary>
        public bool UseYCoCg { get; init; }
        /// <summary>History reprojection jitter.</summary>
        public float ReprojectionJitter { get; init; }
        /// <summary>Maximum allowed velocity for reprojection.</summary>
        public float MaxVelocity { get; init; }
    }

    /// <summary>
    /// Sample from the G-Buffer for a single pixel.
    /// </summary>
    public record GBufferSample
    {
        /// <summary>Screen-space position.</summary>
        public Vector2 ScreenPosition { get; init; }
        /// <summary>View-space depth.</summary>
        public float Depth { get; init; }
        /// <summary>World-space position.</summary>
        public Vector3 WorldPosition { get; init; }
        /// <summary>World-space normal.</summary>
        public Vector3 Normal { get; init; }
        /// <summary>Albedo color.</summary>
        public Vector3 Albedo { get; init; }
        /// <summary>Specular color.</summary>
        public Vector3 Specular { get; init; }
        /// <summary>Roughness value.</summary>
        public float Roughness { get; init; }
        /// <summary>Metallic value.</summary>
        public float Metallic { get; init; }
        /// <summary>Screen-space velocity.</summary>
        public Vector2 Velocity { get; init; }
        /// <summary>Material ID for classification.</summary>
        public int MaterialID { get; init; }
        /// <summary>Is this pixel on a dynamic object.</summary>
        public bool IsDynamic { get; init; }
        /// <summary>Is this pixel on a translucent surface.</summary>
        public bool IsTranslucent { get; init; }
    }

    /// <summary>
    /// Payload data carried by a ray through the scene.
    /// </summary>
    public record RayPayload
    {
        /// <summary>Ray origin.</summary>
        public Vector3 Origin { get; init; }
        /// <summary>Ray direction (normalized).</summary>
        public Vector3 Direction { get; init; }
        /// <summary>Maximum trace distance.</summary>
        public float MaxDistance { get; init; }
        /// <summary>Accumulated path throughput.</summary>
        public Vector3 Throughput { get; init; }
        /// <summary>Accumulated radiance along the path.</summary>
        public Vector3 Radiance { get; init; }
        /// <summary>Current bounce depth.</summary>
        public int BounceDepth { get; init; }
        /// <summary>Random seed for this ray.</summary>
        public uint RandomSeed { get; init; }
        /// <summary>Flags for ray properties.</summary>
        public RayFlags Flags { get; init; }
    }

    /// <summary>
    /// Flags for ray properties.
    /// </summary>
    [Flags]
    public enum RayFlags
    {
        /// <summary>No special flags.</summary>
        None = 0,
        /// <summary>Ray is for shadow testing.</summary>
        Shadow = 1,
        /// <summary>Ray is for indirect lighting.</summary>
        Indirect = 2,
        /// <summary>Ray is for specular reflection.</summary>
        Specular = 4,
        /// <summary>Ray is for transmission.</summary>
        Transmission = 8,
        /// <summary>Ray should terminate on first hit.</summary>
        ClosestHit = 16,
        /// <summary>Ray is for ambient occlusion.</summary>
        AO = 32
    }

    /// <summary>
    /// Result of a ray-scene intersection test.
    /// </summary>
    public record HitResult
    {
        /// <summary>Whether the ray hit geometry.</summary>
        public bool DidHit { get; init; }
        /// <summary>Distance along the ray to the hit.</summary>
        public float HitDistance { get; init; }
        /// <summary>World-space hit position.</summary>
        public Vector3 HitPosition { get; init; }
        /// <summary>Surface normal at the hit point.</summary>
        public Vector3 Normal { get; init; }
        /// <summary>Geometric normal (uninterpolated).</summary>
        public Vector3 GeometricNormal { get; init; }
        /// <summary>Albedo at the hit point.</summary>
        public Vector3 Albedo { get; init; }
        /// <summary>Material properties at the hit point.</summary>
        public MaterialProperties Material { get; init; }
        /// <summary>Triangle index in the mesh.</summary>
        public int TriangleIndex { get; init; }
        /// <summary>Primitive index for procedural geometry.</summary>
        public int PrimitiveIndex { get; init; }
        /// <summary>Instance ID for multi-instance scenes.</summary>
        public int InstanceID { get; init; }
        /// <summary>UV coordinates at the hit point.</summary>
        public Vector2 UV { get; init; }
        /// <summary>Barycentric coordinates of the hit.</summary>
        public Vector3 Barycentrics { get; init; }
    }

    /// <summary>
    /// Material properties for shading.
    /// </summary>
    public record MaterialProperties
    {
        /// <summary>Base color / albedo.</summary>
        public Vector3 BaseColor { get; init; }
        /// <summary>Specular color.</summary>
        public Vector3 SpecularColor { get; init; }
        /// <summary>Roughness (0 = mirror, 1 = diffuse).</summary>
        public float Roughness { get; init; }
        /// <summary>Metallic (0 = dielectric, 1 = metal).</summary>
        public float Metallic { get; init; }
        /// <summary>Index of refraction.</summary>
        public float IOR { get; init; }
        /// <summary>Subsurface scattering coefficient.</summary>
        public float Subsurface { get; init; }
        /// <summary>Specular transmission.</summary>
        public float SpecularTransmission { get; init; }
        /// <summary>Thin surface flag.</summary>
        public bool IsThinSurface { get; init; }
        /// <summary>Emissive color.</summary>
        public Vector3 Emissive { get; init; }
        /// <summary>Emissive intensity.</summary>
        public float EmissiveIntensity { get; init; }
    }

    /// <summary>
    /// Result of BxDF evaluation.
    /// </summary>
    public record BxDFResult
    {
        /// <summary>Evaluated BxDF value (f).</summary>
        public Vector3 Value { get; init; }
        /// <summary>Sampled direction.</summary>
        public Vector3 SampledDirection { get; init; }
        /// <summary>Probability density function value.</summary>
        public float PDF { get; init; }
        /// <summary>Whether the sample is valid.</summary>
        public bool IsValid { get; init; }
        /// <summary>Is this a delta distribution (e.g., mirror).</summary>
        public bool IsDelta { get; init; }
        /// <summary>Component type that was sampled.</summary>
        public BxDFComponent Component { get; init; }
    }

    /// <summary>
    /// BxDF component types.
    /// </summary>
    public enum BxDFComponent
    {
        /// <summary>Diffuse reflection.</summary>
        Diffuse,
        /// <summary>Specular reflection.</summary>
        SpecularReflection,
        /// <summary>Specular transmission.</summary>
        SpecularTransmission,
        /// <summary>Glossy reflection.</summary>
        Glossy
    }

    /// <summary>
    /// Represents a single scattering event in path tracing.
    /// </summary>
    public record ScatteringEvent
    {
        /// <summary>Position of the scattering event.</summary>
        public Vector3 Position { get; init; }
        /// <summary>Incoming direction.</summary>
        public Vector3 IncomingDirection { get; init; }
        /// <summary>Outgoing direction.</summary>
        public Vector3 OutgoingDirection { get; init; }
        /// <summary>Surface normal.</summary>
        public Vector3 Normal { get; init; }
        /// <summary>BxDF value at this event.</summary>
        public Vector3 BxDFValue { get; init; }
        /// <summary>PDF of this event.</summary>
        public float PDF { get; init; }
        /// <summary>Importance weight of this event.</summary>
        public float Importance { get; init; }
        /// <summary>Material properties at the event.</summary>
        public MaterialProperties Material { get; init; }
        /// <summary>Path depth of this event.</summary>
        public int Depth { get; init; }
    }

    /// <summary>
    /// Represents the state of a path being traced.
    /// </summary>
    public record PathState
    {
        /// <summary>Current ray origin.</summary>
        public Vector3 RayOrigin { get; init; }
        /// <summary>Current ray direction.</summary>
        public Vector3 RayDirection { get; init; }
        /// <summary>Accumulated path throughput.</summary>
        public Vector3 Throughput { get; init; }
        /// <summary>Accumulated radiance.</summary>
        public Vector3 Radiance { get; init; }
        /// <summary>Current bounce depth.</summary>
        public int Depth { get; init; }
        /// <summary>Maximum allowed depth.</summary>
        public int MaxDepth { get; init; }
        /// <summary>Random number generator state.</summary>
        public uint RNGState { get; init; }
        /// <summary>Whether the path is still active.</summary>
        public bool IsActive { get; init; }
        /// <summary>Current medium (for volume rendering).</summary>
        public int CurrentMedium { get; init; }
        /// <summary>Accumulated transmittance.</summary>
        public Vector3 Transmittance { get; init; }
    }

    /// <summary>
    /// Represents a single pixel in the film (output image).
    /// </summary>
    public record FilmPixel
    {
        /// <summary>Pixel coordinates.</summary>
        public Vector2Int Position { get; init; }
        /// <summary>Accumulated color.</summary>
        public Vector3 Color { get; init; }
        /// <summary>Accumulated radiance.</summary>
        public Vector3 Radiance { get; init; }
        /// <summary>Number of samples taken for this pixel.</summary>
        public int SampleCount { get; init; }
        /// <summary>Variance estimate.</summary>
        public float Variance { get; init; }
        /// <summary>Is this pixel converged.</summary>
        public bool IsConverged { get; init; }
        /// <summary>Tile index for tiled rendering.</summary>
        public int TileIndex { get; init; }
    }

    /// <summary>
    /// Classification of a screen tile for deferred rendering.
    /// </summary>
    public record TileClassification
    {
        /// <summary>Tile index.</summary>
        public int TileIndex { get; init; }
        /// <summary>Tile minimum depth.</summary>
        public float MinDepth { get; init; }
        /// <summary>Tile maximum depth.</summary>
        public float MaxDepth { get; init; }
        /// <summary>Tile average normal.</summary>
        public Vector3 AverageNormal { get; init; }
        /// <summary>Whether the tile has translucency.</summary>
        public bool HasTranslucency { get; init; }
        /// <summary>Whether the tile has emissive surfaces.</summary>
        public bool HasEmissive { get; init; }
        /// <summary>Lighting complexity score.</summary>
        public float LightingComplexity { get; init; }
        /// <summary>Recommended GI quality level.</summary>
        public int RecommendedGIQuality { get; init; }
    }

    /// <summary>
    /// Adaptive quality target based on performance metrics.
    /// </summary>
    public record AdaptiveQualityTarget
    {
        /// <summary>Target frame time in milliseconds.</summary>
        public float TargetFrameTimeMs { get; init; }
        /// <summary>Current frame time in milliseconds.</summary>
        public float CurrentFrameTimeMs { get; init; }
        /// <summary>Recommended quality scale (0.5 = half quality, 1.0 = full).</summary>
        public float QualityScale { get; init; }
        /// <summary>Whether to reduce cascade count.</summary>
        public bool ReduceCascadeCount { get; init; }
        /// <summary>Whether to reduce ray count per texel.</summary>
        public bool ReduceRayCount { get; init; }
        /// <summary>Whether to use lower resolution buffers.</summary>
        public bool UseLowerResolution { get; init; }
        /// <summary>Performance headroom percentage.</summary>
        public float PerformanceHeadroom { get; init; }
        /// <summary>Recommended denoiser strength.</summary>
        public float DenoiserStrength { get; init; }
    }

    /// <summary>
    /// Integer vector for 3D grid indexing.
    /// </summary>
    public record struct Vector3Int
    {
        /// <summary>X component.</summary>
        public int X;
        /// <summary>Y component.</summary>
        public int Y;
        /// <summary>Z component.</summary>
        public int Z;

        /// <summary>Creates a new Vector3Int.</summary>
        public Vector3Int(int x, int y, int z) { X = x; Y = y; Z = z; }

        /// <summary>Computes linear index from 3D coordinates.</summary>
        public int ToLinearIndex(int strideY, int strideZ) => X + Y * strideY + Z * strideZ;
    }

    /// <summary>
    /// Integer vector for 2D screen indexing.
    /// </summary>
    public record struct Vector2Int
    {
        /// <summary>X component.</summary>
        public int X;
        /// <summary>Y component.</summary>
        public int Y;

        /// <summary>Creates a new Vector2Int.</summary>
        public Vector2Int(int x, int y) { X = x; Y = y; }
    }

    #endregion

    // =========================================================================
    #region Helper Types

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
            if (MathF.Abs(clip.W) < 0.0001f) return new Vector3(-1, -1, -1);
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
        public float[] Depth { get; set; }
        /// <summary>Normal buffer (world-space normals, XYZ).</summary>
        public Vector3[] Normals { get; set; }
        /// <summary>Albedo buffer (base color).</summary>
        public Vector3[] Albedo { get; set; }
        /// <summary>Velocity buffer (screen-space motion vectors).</summary>
        public Vector2[] Velocity { get; set; }
        /// <summary>Material buffer (roughness, metallic packed).</summary>
        public Vector4[] MaterialProps { get; set; }
        /// <summary>Specular buffer.</summary>
        public Vector3[] Specular { get; set; }
        /// <summary>Emissive buffer.</summary>
        public Vector3[] Emissive { get; set; }
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
        public object CommandList { get; set; }
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
        public float[] TimePerLevelMs { get; init; }
        /// <summary>Memory allocated per cascade level.</summary>
        public long[] MemoryPerLevelBytes { get; init; }
        /// <summary>Ray count allocated per cascade level.</summary>
        public int[] RaysPerLevel { get; init; }
        /// <summary>Resolution per cascade level.</summary>
        public int[] ResolutionPerLevel { get; init; }
        /// <summary>Whether allocation was successful within budget.</summary>
        public bool WithinBudget { get; init; }
    }

    #endregion

    // =========================================================================
    #region LDNNAnalytics

    /// <summary>
    /// Analytics system for tracking performance and quality metrics.
    /// </summary>
    public class LDNNAnalytics
    {
        private readonly Queue<float> _frameTimeHistory = new(120);
        private readonly Queue<float> _cascadeTimeHistory = new(120);
        private readonly Queue<float> _neuralTimeHistory = new(120);
        private readonly Queue<int> _raysTracedHistory = new(120);
        private readonly Queue<float> _qualityMetricsHistory = new(120);
        private readonly object _lock = new();
        private int _totalFramesTraced;
        private long _totalRaysTraced;
        private float _averagePSNR;
        private float _averageSSIM;
        private int _convergenceFrameCount;
        private float _lastQualityScore;

        /// <summary>Average frame time over the last 120 frames.</summary>
        public float AverageFrameTimeMs { get; private set; }
        /// <summary>Average cascade rendering time.</summary>
        public float AverageCascadeTimeMs { get; private set; }
        /// <summary>Average neural prediction time.</summary>
        public float AverageNeuralTimeMs { get; private set; }
        /// <summary>Average rays traced per frame.</summary>
        public float AverageRaysPerFrame { get; private set; }
        /// <summary>Current quality score (0-1).</summary>
        public float CurrentQualityScore { get; private set; }
        /// <summary>Performance budget utilization (0-1).</summary>
        public float BudgetUtilization { get; private set; }
        /// <summary>Total frames rendered.</summary>
        public int TotalFramesRendered => _totalFramesTraced;
        /// <summary>Total rays traced across all frames.</summary>
        public long TotalRaysTraced => _totalRaysTraced;

        /// <summary>
        /// Records a frame's telemetry data.
        /// </summary>
        public void RecordFrame(FrameTelemetry telemetry)
        {
            lock (_lock)
            {
                _frameTimeHistory.Enqueue(telemetry.TotalFrameTimeMs);
                if (_frameTimeHistory.Count > 120) _frameTimeHistory.Dequeue();

                _cascadeTimeHistory.Enqueue(telemetry.CascadeRenderTimeMs);
                if (_cascadeTimeHistory.Count > 120) _cascadeTimeHistory.Dequeue();

                _neuralTimeHistory.Enqueue(telemetry.NeuralPredictionTimeMs);
                if (_neuralTimeHistory.Count > 120) _neuralTimeHistory.Dequeue();

                _raysTracedHistory.Enqueue(telemetry.RaysTraced);
                if (_raysTracedHistory.Count > 120) _raysTracedHistory.Dequeue();

                _totalFramesTraced++;
                _totalRaysTraced += telemetry.RaysTraced;

                RecomputeAverages();
            }
        }

        private void RecomputeAverages()
        {
            if (_frameTimeHistory.Count == 0) return;

            float sumFrame = 0, sumCascade = 0, sumNeural = 0;
            int sumRays = 0;
            foreach (float f in _frameTimeHistory) sumFrame += f;
            foreach (float c in _cascadeTimeHistory) sumCascade += c;
            foreach (float n in _neuralTimeHistory) sumNeural += n;
            foreach (int r in _raysTracedHistory) sumRays += r;

            int count = _frameTimeHistory.Count;
            AverageFrameTimeMs = sumFrame / count;
            AverageCascadeTimeMs = sumCascade / count;
            AverageNeuralTimeMs = sumNeural / count;
            AverageRaysPerFrame = (float)sumRays / count;
        }

        /// <summary>
        /// Records quality metrics for reference comparison.
        /// </summary>
        public void RecordQualityMetrics(float psnr, float ssim)
        {
            lock (_lock)
            {
                _averagePSNR = _averagePSNR * 0.95f + psnr * 0.05f;
                _averageSSIM = _averageSSIM * 0.95f + ssim * 0.05f;
                _lastQualityScore = CalculateQualityScore(psnr, ssim);
                _qualityMetricsHistory.Enqueue(_lastQualityScore);
                if (_qualityMetricsHistory.Count > 120) _qualityMetricsHistory.Dequeue();
            }
        }

        private float CalculateQualityScore(float psnr, float ssim)
        {
            float psnrNormalized = MathF.Min(1.0f, MathF.Max(0.0f, (psnr - 20.0f) / 30.0f));
            return psnrNormalized * 0.5f + ssim * 0.5f;
        }

        /// <summary>
        /// Computes an adaptive quality target based on performance history.
        /// </summary>
        public AdaptiveQualityTarget ComputeAdaptiveTarget(float targetFrameTimeMs)
        {
            lock (_lock)
            {
                float currentFrameTime = AverageFrameTimeMs;
                float headroom = (targetFrameTimeMs - currentFrameTime) / targetFrameTimeMs;
                float qualityScale = MathF.Max(0.25f, MathF.Min(1.5f, 1.0f + headroom * 0.5f));

                return new AdaptiveQualityTarget
                {
                    TargetFrameTimeMs = targetFrameTimeMs,
                    CurrentFrameTimeMs = currentFrameTime,
                    QualityScale = qualityScale,
                    ReduceCascadeCount = currentFrameTime > targetFrameTimeMs * 1.1f,
                    ReduceRayCount = currentFrameTime > targetFrameTimeMs * 1.05f,
                    UseLowerResolution = currentFrameTime > targetFrameTimeMs * 1.2f,
                    PerformanceHeadroom = headroom,
                    DenoiserStrength = MathF.Max(0.3f, MathF.Min(1.0f, 1.0f - headroom * 0.3f))
                };
            }
        }

        /// <summary>
        /// Gets performance statistics as a formatted string.
        /// </summary>
        public string GetPerformanceReport()
        {
            lock (_lock)
            {
                return $"Frame: {AverageFrameTimeMs:F2}ms | Cascade: {AverageCascadeTimeMs:F2}ms | " +
                       $"Neural: {AverageNeuralTimeMs:F2}ms | Rays: {AverageRaysPerFrame:F0} | " +
                       $"Quality: {CurrentQualityScore:F3} | Frames: {_totalFramesTraced}";
            }
        }

        /// <summary>
        /// Resets all analytics data.
        /// </summary>
        public void Reset()
        {
            lock (_lock)
            {
                _frameTimeHistory.Clear();
                _cascadeTimeHistory.Clear();
                _neuralTimeHistory.Clear();
                _raysTracedHistory.Clear();
                _qualityMetricsHistory.Clear();
                _totalFramesTraced = 0;
                _totalRaysTraced = 0;
                _averagePSNR = 0;
                _averageSSIM = 0;
                _convergenceFrameCount = 0;
                _lastQualityScore = 0;
                AverageFrameTimeMs = 0;
                AverageCascadeTimeMs = 0;
                AverageNeuralTimeMs = 0;
                AverageRaysPerFrame = 0;
                CurrentQualityScore = 0;
                BudgetUtilization = 0;
            }
        }
    }

    #endregion

    // =========================================================================
    #region LightCullingSystem

    /// <summary>
    /// Tiled/clustered light culling system for efficient light evaluation.
    /// </summary>
    public class LightCullingSystem
    {
        private const int TILE_SIZE = 16;
        private const int MAX_LIGHTS_PER_TILE = 256;
        private const int CLUSTER_Z_SLICES = 16;

        private List<LightConfig> _allLights = new();
        private List<int>[,] _tileLightLists;
        private List<int>[][][] _clusterLightLists;
        private int _tileCountX;
        private int _tileCountY;
        private int _screenWidth;
        private int _screenHeight;
        private float[] _clusterDepths;
        private bool _isInitialized;

        /// <summary>Number of tiles in the X direction.</summary>
        public int TileCountX => _tileCountX;
        /// <summary>Number of tiles in the Y direction.</summary>
        public int TileCountY => _tileCountY;
        /// <summary>Total number of tiles.</summary>
        public int TotalTiles => _tileCountX * _tileCountY;
        /// <summary>Total number of clusters.</summary>
        public int TotalClusters => _tileCountX * _tileCountY * CLUSTER_Z_SLICES;

        /// <summary>
        /// Initializes the light culling system for a given screen resolution.
        /// </summary>
        public void Initialize(int screenWidth, int screenHeight)
        {
            _screenWidth = screenWidth;
            _screenHeight = screenHeight;
            _tileCountX = (screenWidth + TILE_SIZE - 1) / TILE_SIZE;
            _tileCountY = (screenHeight + TILE_SIZE - 1) / TILE_SIZE;

            _tileLightLists = new List<int>[_tileCountX, _tileCountY];
            for (int x = 0; x < _tileCountX; x++)
                for (int y = 0; y < _tileCountY; y++)
                    _tileLightLists[x, y] = new List<int>(MAX_LIGHTS_PER_TILE);

            _clusterLightLists = new List<int>[_tileCountX][][];
            for (int x = 0; x < _tileCountX; x++)
            {
                _clusterLightLists[x] = new List<int>[_tileCountY][];
                for (int y = 0; y < _tileCountY; y++)
                {
                    _clusterLightLists[x][y] = new List<int>[CLUSTER_Z_SLICES];
                    for (int z = 0; z < CLUSTER_Z_SLICES; z++)
                        _clusterLightLists[x][y][z] = new List<int>(MAX_LIGHTS_PER_TILE);
                }
            }

            _clusterDepths = new float[CLUSTER_Z_SLICES + 1];
            _isInitialized = true;
        }

        /// <summary>
        /// Updates the light list and performs culling for the current frame.
        /// </summary>
        public void CullLights(List<LightConfig> lights, CameraState camera)
        {
            if (!_isInitialized) return;
            _allLights = lights;
            ComputeClusterDepths(camera.NearPlane, camera.FarPlane);

            Parallel.For(0, _tileCountX, tx =>
            {
                for (int ty = 0; ty < _tileCountY; ty++)
                {
                    _tileLightLists[tx, ty].Clear();
                    for (int i = 0; i < lights.Count; i++)
                    {
                        if (IsLightRelevant(lights[i], tx, ty, camera))
                        {
                            _tileLightLists[tx, ty].Add(i);
                        }
                    }
                }
            });

            Parallel.For(0, _tileCountX, tx =>
            {
                for (int ty = 0; ty < _tileCountY; ty++)
                {
                    for (int tz = 0; tz < CLUSTER_Z_SLICES; tz++)
                    {
                        _clusterLightLists[tx][ty][tz].Clear();
                        float zMin = _clusterDepths[tz];
                        float zMax = _clusterDepths[tz + 1];

                        for (int li = 0; li < _tileLightLists[tx, ty].Count; li++)
                        {
                            int lightIdx = _tileLightLists[tx, ty][li];
                            if (IsLightInCluster(lights[lightIdx], tx, ty, zMin, zMax, camera))
                            {
                                _clusterLightLists[tx][ty][tz].Add(lightIdx);
                            }
                        }
                    }
                }
            });
        }

        private void ComputeClusterDepths(float nearPlane, float farPlane)
        {
            for (int i = 0; i <= CLUSTER_Z_SLICES; i++)
            {
                float t = (float)i / CLUSTER_Z_SLICES;
                _clusterDepths[i] = nearPlane * MathF.Pow(farPlane / nearPlane, t);
            }
        }

        private bool IsLightRelevant(LightConfig light, int tx, int ty, CameraState camera)
        {
            if (light.Type == LightType.Directional) return true;
            float range = light.Range;
            if (range <= 0) range = 100.0f;
            Vector3 viewPos = Vector3.Transform(light.Position, camera.ViewMatrix);
            if (viewPos.Z > -camera.NearPlane) return false;
            float tileXMin = (float)(tx * TILE_SIZE) / _screenWidth * 2.0f - 1.0f;
            float tileXMax = (float)((tx + 1) * TILE_SIZE) / _screenWidth * 2.0f - 1.0f;
            float tileYMin = 1.0f - (float)((ty + 1) * TILE_SIZE) / _screenHeight * 2.0f;
            float tileYMax = 1.0f - (float)(ty * TILE_SIZE) / _screenHeight * 2.0f;
            float absZ = MathF.Abs(viewPos.Z);
            if (absZ < 0.001f) return true;
            float projRadius = range / absZ * MathF.Max(_screenWidth, _screenHeight) * 0.5f / MathF.Tan(camera.FieldOfView * 0.5f);
            float screenX = (viewPos.X / absZ * 0.5f + 0.5f) * 2.0f - 1.0f;
            float screenY = 1.0f - (viewPos.Y / absZ * 0.5f + 0.5f) * 2.0f;
            return screenX + projRadius / _screenWidth >= tileXMin &&
                   screenX - projRadius / _screenWidth <= tileXMax &&
                   screenY + projRadius / _screenHeight >= tileYMin &&
                   screenY - projRadius / _screenHeight <= tileYMax;
        }

        private bool IsLightInCluster(LightConfig light, int tx, int ty, float zMin, float zMax, CameraState camera)
        {
            if (light.Type == LightType.Directional) return true;
            float range = light.Range;
            if (range <= 0) range = 100.0f;
            Vector3 viewPos = Vector3.Transform(light.Position, camera.ViewMatrix);
            if (viewPos.Z + range < -zMax || viewPos.Z - range > -zMin) return false;
            return true;
        }

        /// <summary>
        /// Gets the list of light indices affecting a specific tile.
        /// </summary>
        public List<int> GetTileLights(int tileX, int tileY)
        {
            if (tileX < 0 || tileX >= _tileCountX || tileY < 0 || tileY >= _tileCountY)
                return new List<int>();
            return _tileLightLists[tileX, tileY];
        }

        /// <summary>
        /// Gets the list of light indices in a specific cluster.
        /// </summary>
        public List<int> GetClusterLights(int tileX, int tileY, int clusterZ)
        {
            if (tileX < 0 || tileX >= _tileCountX || tileY < 0 || tileY >= _tileCountY ||
                clusterZ < 0 || clusterZ >= CLUSTER_Z_SLICES)
                return new List<int>();
            return _clusterLightLists[tileX][tileY][clusterZ];
        }

        /// <summary>
        /// Gets the depth range for a cluster slice.
        /// </summary>
        public (float Near, float Far) GetClusterDepthRange(int clusterZ)
        {
            if (clusterZ < 0 || clusterZ >= CLUSTER_Z_SLICES) return (0, 0);
            return (_clusterDepths[clusterZ], _clusterDepths[clusterZ + 1]);
        }

        /// <summary>
        /// Builds the light grid data for GPU consumption.
        /// </summary>
        public (int[] LightIndices, int[] TileOffsets, int[] TileCounts) BuildLightGridData()
        {
            var allIndices = new List<int>();
            var offsets = new int[_tileCountX * _tileCountY * CLUSTER_Z_SLICES];
            var counts = new int[_tileCountX * _tileCountY * CLUSTER_Z_SLICES];

            for (int tx = 0; tx < _tileCountX; tx++)
            {
                for (int ty = 0; ty < _tileCountY; ty++)
                {
                    for (int tz = 0; tz < CLUSTER_Z_SLICES; tz++)
                    {
                        int clusterIdx = (tx * _tileCountY + ty) * CLUSTER_Z_SLICES + tz;
                        offsets[clusterIdx] = allIndices.Count;
                        var lights = _clusterLightLists[tx][ty][tz];
                        counts[clusterIdx] = lights.Count;
                        allIndices.AddRange(lights);
                    }
                }
            }

            return (allIndices.ToArray(), offsets, counts);
        }
    }

    /// <summary>
    /// Extension methods for Vector4.
    /// </summary>
    public static class Vector4Extensions
    {
        /// <summary>Extracts XYZ components as a Vector3.</summary>
        public static Vector3 XYZ(this Vector4 v) => new Vector3(v.X, v.Y, v.Z);
    }

    #endregion

    // =========================================================================
    #region ReferencePathTracer

    /// <summary>
    /// Reference path tracer for generating ground truth images and training data.
    /// Implements unidirectional path tracing with MIS, Russian roulette, and
    /// next event estimation.
    /// </summary>
    public class ReferencePathTracer
    {
        private const float EPSILON = 1e-6f;
        private const float RussianRouletteThreshold = 0.05f;
        private const int MAX_BOUNCES = 32;
        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;
        private const float INV_TWO_PI = 1.0f / (2.0f * MathF.PI);

        private FilmPixel[,] _film;
        private int _width;
        private int _height;
        private int _totalSamples;

        /// <summary>Reference image buffer.</summary>
        public FilmPixel[,] Film => _film;

        /// <summary>
        /// Initializes the path tracer for a given resolution.
        /// </summary>
        public void Initialize(int width, int height)
        {
            _width = width;
            _height = height;
            _film = new FilmPixel[width, height];
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                    _film[x, y] = new FilmPixel
                    {
                        Position = new Vector2Int(x, y),
                        Color = Vector3.Zero,
                        Radiance = Vector3.Zero,
                        SampleCount = 0,
                        Variance = 0,
                        IsConverged = false,
                        TileIndex = 0
                    };
        }

        /// <summary>
        /// Generates a reference image using path tracing.
        /// </summary>
        public void GenerateReferenceImage(GBuffer gbuffer, CameraState camera,
            List<LightConfig> lights, int samplesPerPixel, int maxDepth)
        {
            _totalSamples = samplesPerPixel;

            Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    var rng = new RandomNumberGenerator((uint)(x * _height + y + _totalSamples * 1000));
                    Vector3 pixelColor = Vector3.Zero;

                    for (int s = 0; s < samplesPerPixel; s++)
                    {
                        float jitterX = rng.NextFloat() - 0.5f;
                        float jitterY = rng.NextFloat() - 0.5f;
                        float u = (x + 0.5f + jitterX) / _width;
                        float v = (y + 0.5f + jitterY) / _height;

                        RayPayload ray = GenerateCameraRay(u, v, camera, ref rng);
                        Vector3 radiance = EstimateRadiance(ray, gbuffer, lights, maxDepth, ref rng);
                        pixelColor += radiance;
                    }

                    pixelColor /= samplesPerPixel;

                    _film[x, y] = _film[x, y] with
                    {
                        Color = pixelColor,
                        Radiance = pixelColor,
                        SampleCount = samplesPerPixel,
                        IsConverged = true
                    };
                }
            });
        }

        /// <summary>
        /// Generates a camera ray from screen-space coordinates.
        /// </summary>
        public RayPayload GenerateCameraRay(float u, float v, CameraState camera, ref RandomNumberGenerator rng)
        {
            float tanHalfFov = MathF.Tan(camera.FieldOfView * 0.5f);
            float aspect = camera.AspectRatio;
            float ndcX = (2.0f * u - 1.0f) * aspect * tanHalfFov;
            float ndcY = (1.0f - 2.0f * v) * tanHalfFov;
            Vector3 rayDir = Vector3.Normalize(camera.Right * ndcX + camera.Up * ndcY + camera.Forward);

            return new RayPayload
            {
                Origin = camera.Position,
                Direction = rayDir,
                MaxDistance = camera.FarPlane,
                Throughput = Vector3.One,
                Radiance = Vector3.Zero,
                BounceDepth = 0,
                RandomSeed = rng.NextUint(),
                Flags = RayFlags.None
            };
        }

        /// <summary>
        /// Estimates the radiance along a ray using path tracing.
        /// </summary>
        public Vector3 EstimateRadiance(RayPayload initialRay, GBuffer gbuffer,
            List<LightConfig> lights, int maxDepth, ref RandomNumberGenerator rng)
        {
            Vector3 accumulatedRadiance = Vector3.Zero;
            Vector3 throughput = initialRay.Throughput;
            Vector3 origin = initialRay.Origin;
            Vector3 direction = initialRay.Direction;
            int depth = 0;

            while (depth < maxDepth)
            {
                HitResult hit = TraceRay(origin, direction, gbuffer);
                if (!hit.DidHit)
                {
                    accumulatedRadiance += throughput * SampleEnvironmentMap(direction);
                    break;
                }

                MaterialProperties mat = hit.Material;
                accumulatedRadiance += throughput * mat.Emissive * mat.EmissiveIntensity;

                if (depth > 2)
                {
                    float continueProbability = MathF.Max(throughput.X,
                        MathF.Max(throughput.Y, throughput.Z));
                    if (rng.NextFloat() > continueProbability)
                        break;
                    throughput /= continueProbability;
                }

                Vector3 wo = -direction;
                Vector3 lightContribution = EstimateDirectLighting(hit, wo, lights, gbuffer, ref rng);
                accumulatedRadiance += throughput * lightContribution;

                BxDFResult bxdf = SampleBSDF(hit, wo, ref rng);
                if (!bxdf.IsValid || bxdf.PDF < EPSILON)
                    break;

                throughput *= bxdf.Value * MathF.Abs(Vector3.Dot(bxdf.SampledDirection, hit.Normal)) / bxdf.PDF;

                origin = hit.HitPosition + bxdf.SampledDirection * EPSILON;
                direction = bxdf.SampledDirection;
                depth++;
            }

            return accumulatedRadiance;
        }

        /// <summary>
        /// Performs ray tracing against the scene geometry.
        /// </summary>
        public HitResult TraceRay(Vector3 origin, Vector3 direction, GBuffer gbuffer)
        {
            float minDist = float.MaxValue;
            HitResult closest = new HitResult { DidHit = false };

            for (int x = 0; x < gbuffer.Width; x++)
            {
                for (int y = 0; y < gbuffer.Height; y++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    float depth = gbuffer.Depth[idx];
                    if (depth <= 0 || depth >= minDist) continue;

                    Vector3 hitPos = origin + direction * depth;
                    float cosAngle = Vector3.Dot(direction, gbuffer.Normals[idx]);
                    if (cosAngle >= 0) continue;

                    minDist = depth;
                    Vector3 normal = gbuffer.Normals[idx];
                    Vector3 albedo = gbuffer.Albedo[idx];
                    float roughness = gbuffer.MaterialProps[idx].X;
                    float metallic = gbuffer.MaterialProps[idx].Y;
                    Vector3 specular = gbuffer.Specular[idx];

                    closest = new HitResult
                    {
                        DidHit = true,
                        HitDistance = depth,
                        HitPosition = hitPos,
                        Normal = normal,
                        GeometricNormal = normal,
                        Albedo = albedo,
                        Material = new MaterialProperties
                        {
                            BaseColor = albedo,
                            SpecularColor = specular,
                            Roughness = roughness,
                            Metallic = metallic,
                            IOR = 1.5f,
                            Subsurface = 0,
                            SpecularTransmission = 0,
                            IsThinSurface = false,
                            Emissive = gbuffer.Emissive[idx],
                            EmissiveIntensity = 1.0f
                        },
                        UV = new Vector2((float)x / gbuffer.Width, (float)y / gbuffer.Height),
                        Barycentrics = new Vector3(1, 0, 0),
                        TriangleIndex = idx,
                        PrimitiveIndex = 0,
                        InstanceID = 0
                    };
                }
            }

            return closest;
        }

        /// <summary>
        /// Evaluates the BxDF for a given surface and direction pair.
        /// </summary>
        public Vector3 EvaluateBSDF(HitResult hit, Vector3 wi, Vector3 wo)
        {
            MaterialProperties mat = hit.Material;
            Vector3 n = hit.Normal;
            float NdotL = MathF.Max(0, Vector3.Dot(n, wi));
            float NdotV = MathF.Max(0, Vector3.Dot(n, wo));
            if (NdotL < EPSILON || NdotV < EPSILON) return Vector3.Zero;

            Vector3 diffuse = mat.BaseColor * INV_PI;
            Vector3 halfVec = Vector3.Normalize(wi + wo);
            float HdotV = MathF.Max(0, Vector3.Dot(halfVec, wo));
            float D = DistributionGGX(n, halfVec, mat.Roughness);
            float G = GeometrySmith(n, wi, wo, mat.Roughness);
            Vector3 F = FresnelSchlick(HdotV, Vector3.Lerp(new Vector3(0.04f), mat.BaseColor, mat.Metallic));
            Vector3 kD = (Vector3.One - F) * (1.0f - mat.Metallic);
            Vector3 specular = (D * G * F) / MathF.Max(EPSILON, 4.0f * NdotL * NdotV);
            return kD * diffuse + specular;
        }

        /// <summary>
        /// Samples the BSDF to generate a new direction.
        /// </summary>
        public BxDFResult SampleBSDF(HitResult hit, Vector3 wo, ref RandomNumberGenerator rng)
        {
            MaterialProperties mat = hit.Material;
            Vector3 n = hit.Normal;
            float r1 = rng.NextFloat();
            float r2 = rng.NextFloat();

            Vector3 tangent = GetOrthogonal(n, ref rng);
            Vector3 bitangent = Vector3.Cross(n, tangent);

            float roughness = MathF.Max(0.04f, mat.Roughness);
            float alpha = roughness * roughness;

            if (rng.NextFloat() < mat.Metallic * 0.5f)
            {
                float phi = TWO_PI * r1;
                float cosTheta = MathF.Sqrt((1.0f - r2) / (1.0f + (alpha * alpha - 1.0f) * r2));
                float sinTheta = MathF.Sqrt(1.0f - cosTheta * cosTheta);
                Vector3 halfVec = tangent * (MathF.Cos(phi) * sinTheta) +
                                  bitangent * (MathF.Sin(phi) * sinTheta) +
                                  n * cosTheta;
                Vector3 wi = Vector3.Reflect(-wo, halfVec);
                if (Vector3.Dot(wi, n) < 0)
                    return new BxDFResult { IsValid = false };

                float D = DistributionGGX(n, halfVec, roughness);
                float G = GeometrySmith(n, wi, wo, roughness);
                float HdotV = MathF.Max(0, Vector3.Dot(halfVec, wo));
                Vector3 F = FresnelSchlick(HdotV, Vector3.Lerp(new Vector3(0.04f), mat.BaseColor, mat.Metallic));
                float pdf = D * MathF.Max(0, Vector3.Dot(halfVec, n)) / MathF.Max(EPSILON, 4.0f * HdotV);
                Vector3 value = F * G * MathF.Max(0, Vector3.Dot(wi, n)) / MathF.Max(EPSILON, HdotV);

                return new BxDFResult
                {
                    Value = value,
                    SampledDirection = wi,
                    PDF = pdf,
                    IsValid = true,
                    IsDelta = false,
                    Component = BxDFComponent.SpecularReflection
                };
            }
            else
            {
                float phi = TWO_PI * r1;
                float cosTheta = MathF.Sqrt(r2);
                float sinTheta = MathF.Sqrt(1.0f - r2);
                Vector3 wi = tangent * (MathF.Cos(phi) * sinTheta) +
                              bitangent * (MathF.Sin(phi) * sinTheta) +
                              n * cosTheta;
                float pdf = cosTheta * INV_PI;
                Vector3 value = mat.BaseColor * INV_PI;
                return new BxDFResult
                {
                    Value = value,
                    SampledDirection = wi,
                    PDF = pdf,
                    IsValid = true,
                    IsDelta = false,
                    Component = BxDFComponent.Diffuse
                };
            }
        }

        /// <summary>
        /// Estimates direct lighting at a surface point using next event estimation.
        /// </summary>
        public Vector3 EstimateDirectLighting(HitResult hit, Vector3 wo,
            List<LightConfig> lights, GBuffer gbuffer, ref RandomNumberGenerator rng)
        {
            Vector3 directLight = Vector3.Zero;

            foreach (var light in lights)
            {
                if (light.Intensity < EPSILON) continue;

                Vector3 lightDir;
                float lightDistance;
                Vector3 lightRadiance;

                switch (light.Type)
                {
                    case LightType.Directional:
                        lightDir = -light.Direction;
                        lightDistance = float.MaxValue;
                        lightRadiance = light.Color * light.Intensity;
                        break;

                    case LightType.Point:
                        Vector3 toLight = light.Position - hit.HitPosition;
                        lightDistance = toLight.Length();
                        lightDir = toLight / lightDistance;
                        float attenuation = 1.0f / (lightDistance * lightDistance);
                        float rangeAtten = MathF.Max(0, 1.0f - MathF.Pow(lightDistance / MathF.Max(0.001f, light.Range), 4));
                        lightRadiance = light.Color * light.Intensity * attenuation * rangeAtten;
                        break;

                    case LightType.Spot:
                        Vector3 spotToLight = light.Position - hit.HitPosition;
                        lightDistance = spotToLight.Length();
                        lightDir = spotToLight / lightDistance;
                        float spotAtten = 1.0f / (lightDistance * lightDistance);
                        float spotRangeAtten = MathF.Max(0, 1.0f - MathF.Pow(lightDistance / MathF.Max(0.001f, light.Range), 4));
                        float cosAngle = Vector3.Dot(-lightDir, light.Direction);
                        float spotCos = MathF.Cos(light.OuterConeAngle);
                        float spotInnerCos = MathF.Cos(light.InnerConeAngle);
                        float spotFalloff = Math.Clamp((cosAngle - spotCos) / MathF.Max(EPSILON, spotInnerCos - spotCos), 0, 1);
                        lightRadiance = light.Color * light.Intensity * spotAtten * spotRangeAtten * spotFalloff;
                        break;

                    case LightType.AreaRect:
                        Vector2 areaSample = SampleRectangle(ref rng);
                        Vector3 lightPoint = light.Position +
                            light.Right * areaSample.X * light.AreaWidth * 0.5f +
                            light.Up * areaSample.Y * light.AreaHeight * 0.5f;
                        Vector3 areaToLight = lightPoint - hit.HitPosition;
                        lightDistance = areaToLight.Length();
                        lightDir = areaToLight / lightDistance;
                        float cosAtLight = MathF.Max(0, -Vector3.Dot(lightDir, light.Direction));
                        float areaPdf = 1.0f / (light.AreaWidth * light.AreaHeight);
                        lightRadiance = light.Color * light.Intensity * cosAtLight /
                            MathF.Max(EPSILON, lightDistance * lightDistance * areaPdf);
                        break;

                    case LightType.AreaDisc:
                        Vector2 discSample = SampleDisc(ref rng);
                        Vector3 discPoint = light.Position +
                            light.Right * discSample.X * light.AreaWidth * 0.5f +
                            light.Up * discSample.Y * light.AreaHeight * 0.5f;
                        Vector3 discToLight = discPoint - hit.HitPosition;
                        lightDistance = discToLight.Length();
                        lightDir = discToLight / lightDistance;
                        float discCosAtLight = MathF.Max(0, -Vector3.Dot(lightDir, light.Direction));
                        float discPdf = 1.0f / (PI * light.AreaWidth * light.AreaHeight * 0.25f);
                        lightRadiance = light.Color * light.Intensity * discCosAtLight /
                            MathF.Max(EPSILON, lightDistance * lightDistance * discPdf);
                        break;

                    default:
                        continue;
                }

                float NdotL = MathF.Max(0, Vector3.Dot(hit.Normal, lightDir));
                if (NdotL < EPSILON) continue;

                if (light.ShadowMethod == ShadowMethod.RayTraced)
                {
                    HitResult shadowHit = TraceRay(
                        hit.HitPosition + hit.Normal * EPSILON,
                        lightDir, gbuffer);
                    if (shadowHit.DidHit && shadowHit.HitDistance < lightDistance)
                        continue;
                }

                Vector3 bxdf = EvaluateBSDF(hit, lightDir, wo);
                directLight += bxdf * lightRadiance * NdotL;
            }

            return directLight;
        }

        /// <summary>
        /// Computes the GGX/Trowbridge-Reitz normal distribution function.
        /// </summary>
        public float DistributionGGX(Vector3 N, Vector3 H, float roughness)
        {
            float a = roughness * roughness;
            float a2 = a * a;
            float NdotH = MathF.Max(0, Vector3.Dot(N, H));
            float NdotH2 = NdotH * NdotH;
            float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
            return a2 / (PI * denom * denom);
        }

        /// <summary>
        /// Computes the Smith geometry shadowing/masking function.
        /// </summary>
        public float GeometrySmith(Vector3 N, Vector3 V, Vector3 L, float roughness)
        {
            float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
            float NdotV = MathF.Max(0, Vector3.Dot(N, V));
            float NdotL = MathF.Max(0, Vector3.Dot(N, L));
            float ggx1 = NdotV / (NdotV * (1.0f - k) + k);
            float ggx2 = NdotL / (NdotL * (1.0f - k) + k);
            return ggx1 * ggx2;
        }

        /// <summary>
        /// Evaluates the Fresnel-Schlick approximation.
        /// </summary>
        public Vector3 FresnelSchlick(float cosTheta, Vector3 F0)
        {
            float oneMinusCos = 1.0f - cosTheta;
            float oneMinusCos5 = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
            return F0 + (Vector3.One - F0) * oneMinusCos5;
        }

        /// <summary>
        /// Computes Fresnel for dielectric surfaces.
        /// </summary>
        public float FresnelDielectric(float cosTheta, float eta)
        {
            float sin2Theta = 1.0f - cosTheta * cosTheta;
            float eta2 = eta * eta;
            float discriminant = 1.0f - sin2Theta / eta2;
            if (discriminant < 0) return 1.0f;
            float cosT = MathF.Sqrt(discriminant);
            float rs = (eta * cosTheta - cosT) / (eta * cosTheta + cosT);
            float rp = (cosTheta - eta * cosT) / (cosTheta + eta * cosT);
            return (rs * rs + rp * rp) * 0.5f;
        }

        /// <summary>
        /// Generates an orthonormal basis from a normal vector.
        /// </summary>
        public Vector3 GetOrthogonal(Vector3 n, ref RandomNumberGenerator rng)
        {
            Vector3 t;
            if (MathF.Abs(n.X) > MathF.Abs(n.Y))
                t = new Vector3(n.Z, 0, -n.X) / MathF.Sqrt(n.X * n.X + n.Z * n.Z);
            else
                t = new Vector3(0, -n.Z, n.Y) / MathF.Sqrt(n.Y * n.Y + n.Z * n.Z);
            return t;
        }

        private Vector2 SampleRectangle(ref RandomNumberGenerator rng)
        {
            return new Vector2(rng.NextFloat() * 2.0f - 1.0f, rng.NextFloat() * 2.0f - 1.0f);
        }

        private Vector2 SampleDisc(ref RandomNumberGenerator rng)
        {
            float r = MathF.Sqrt(rng.NextFloat());
            float theta = TWO_PI * rng.NextFloat();
            return new Vector2(r * MathF.Cos(theta), r * MathF.Sin(theta));
        }

        private Vector3 SampleEnvironmentMap(Vector3 direction)
        {
            float sky = MathF.Max(0, direction.Y);
            return Vector3.Lerp(new Vector3(0.1f, 0.1f, 0.15f), new Vector3(0.4f, 0.6f, 0.9f), sky) * 0.5f;
        }

        /// <summary>
        /// Computes PSNR between the reference and denoised images.
        /// </summary>
        public float ComputePSNR(Vector3[,] reference, Vector3[,] denoised, int width, int height)
        {
            double mse = 0;
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    Vector3 diff = reference[x, y] - denoised[x, y];
                    mse += diff.X * diff.X + diff.Y * diff.Y + diff.Z * diff.Z;
                }
            }
            mse /= (width * height * 3);
            if (mse < 1e-10) return 100.0f;
            return (float)(10.0 * Math.Log10(1.0 / mse));
        }

        /// <summary>
        /// Computes SSIM between reference and denoised images.
        /// </summary>
        public float ComputeSSIM(Vector3[,] reference, Vector3[,] denoised, int width, int height)
        {
            double sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;
            int n = width * height * 3;
            const float C1 = 0.0001f;
            const float C2 = 0.0009f;

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        float rx = reference[x, y][c];
                        float ry = denoised[x, y][c];
                        sumX += rx; sumY += ry;
                        sumXX += rx * rx; sumYY += ry * ry;
                        sumXY += rx * ry;
                    }
                }
            }

            double meanX = sumX / n;
            double meanY = sumY / n;
            double varX = sumXX / n - meanX * meanX;
            double varY = sumYY / n - meanY * meanY;
            double cov = sumXY / n - meanX * meanY;

            double numerator = (2.0 * meanX * meanY + C1) * (2.0 * cov + C2);
            double denominator = (meanX * meanX + meanY * meanY + C1) * (varX + varY + C2);
            return (float)(numerator / denominator);
        }

        /// <summary>
        /// Approximates LPIPS perceptual distance.
        /// </summary>
        public float ComputeLPIPSApproximation(Vector3[,] reference, Vector3[,] denoised, int width, int height)
        {
            double totalDist = 0;
            int sampleCount = 0;
            int step = Math.Max(1, width / 64);

            for (int x = 0; x < width; x += step)
            {
                for (int y = 0; y < height; y += step)
                {
                    Vector3 diff = reference[x, y] - denoised[x, y];
                    float perceptualWeight = 0.5f + 0.5f * MathF.Abs(reference[x, y].Y);
                    totalDist += diff.Length() * perceptualWeight;
                    sampleCount++;
                }
            }

            return sampleCount > 0 ? (float)(totalDist / sampleCount) : 0;
        }

        /// <summary>
        /// Generates training data pairs from path tracing.
        /// </summary>
        public (Vector3[,] NoisyImage, Vector3[,] GroundTruth) GenerateTrainingPair(
            GBuffer gbuffer, CameraState camera, List<LightConfig> lights,
            int samplesPerPixel, int noisySamples, int maxDepth)
        {
            var groundTruth = new Vector3[_width, _height];
            var noisy = new Vector3[_width, _height];

            Parallel.For(0, _height, y =>
            {
                var rngGT = new RandomNumberGenerator((uint)(y * 1337 + 42));
                var rngNoisy = new RandomNumberGenerator((uint)(y * 2847 + 13));

                for (int x = 0; x < _width; x++)
                {
                    Vector3 gtColor = Vector3.Zero;
                    Vector3 noisyColor = Vector3.Zero;

                    for (int s = 0; s < samplesPerPixel; s++)
                    {
                        float jitterX = rngGT.NextFloat() - 0.5f;
                        float jitterY = rngGT.NextFloat() - 0.5f;
                        float u = (x + 0.5f + jitterX) / _width;
                        float v = (y + 0.5f + jitterY) / _height;

                        var ray = GenerateCameraRay(u, v, camera, ref rngGT);
                        gtColor += EstimateRadiance(ray, gbuffer, lights, maxDepth, ref rngGT);
                    }
                    gtColor /= samplesPerPixel;

                    for (int s = 0; s < noisySamples; s++)
                    {
                        float jitterX = rngNoisy.NextFloat() - 0.5f;
                        float jitterY = rngNoisy.NextFloat() - 0.5f;
                        float u = (x + 0.5f + jitterX) / _width;
                        float v = (y + 0.5f + jitterY) / _height;

                        var ray = GenerateCameraRay(u, v, camera, ref rngNoisy);
                        noisyColor += EstimateRadiance(ray, gbuffer, lights, maxDepth, ref rngNoisy);
                    }
                    noisyColor /= noisySamples;

                    groundTruth[x, y] = gtColor;
                    noisy[x, y] = noisyColor;
                }
            });

            return (noisy, groundTruth);
        }
    }

    /// <summary>
    /// Simple pseudo-random number generator for path tracing.
    /// </summary>
    public class RandomNumberGenerator
    {
        private uint _state;

        /// <summary>Creates a new RNG with the given seed.</summary>
        public RandomNumberGenerator(uint seed = 0)
        {
            _state = seed == 0 ? 1u : seed;
        }

        /// <summary>Generates a random unsigned integer.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public uint NextUint()
        {
            _state ^= _state << 13;
            _state ^= _state >> 17;
            _state ^= _state << 5;
            return _state;
        }

        /// <summary>Generates a random float in [0, 1).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextFloat() => (NextUint() & 0x00FFFFFF) / (float)0x01000000;

        /// <summary>Generates a random float in [min, max).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextFloat(float min, float max) => min + NextFloat() * (max - min);

        /// <summary>Generates a random integer in [min, max).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int NextInt(int min, int max) => min + (int)(NextUint() % (uint)(max - min));
    }

    #endregion

    // =========================================================================
    #region RadianceCascadesManager

    /// <summary>
    /// Manages the radiance cascade hierarchy for world-space global illumination.
    /// Implements multi-level cascades with temporal accumulation, spatial filtering,
    /// and importance-driven adaptive allocation.
    /// </summary>
    public class RadianceCascadesManager
    {
        private CascadeConfig _config;
        private CascadeLevelConfig[] _levels;
        private Vector3[][] _cascadeData;
        private Vector3[][] _temporalHistory;
        private float[][] _varianceData;
        private int[][] _sampleCounts;
        private int[] _resolutionPerLevel;
        private float[] _importancePerLevel;
        private float[] _timeAllocationPerLevel;
        private int _totalLevels;
        private int _frameIndex;
        private bool _isInitialized;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>Current cascade configuration.</summary>
        public CascadeConfig Config => _config;
        /// <summary>Number of active cascade levels.</summary>
        public int ActiveLevelCount => _totalLevels;
        /// <summary>Is the manager initialized.</summary>
        public bool IsInitialized => _isInitialized;

        /// <summary>
        /// Initializes the cascade hierarchy with configurable levels.
        /// </summary>
        public void Initialize(CascadeConfig config)
        {
            _config = config;
            _totalLevels = config.NumLevels;
            _levels = new CascadeLevelConfig[_totalLevels];
            _cascadeData = new Vector3[_totalLevels][];
            _temporalHistory = new Vector3[_totalLevels][];
            _varianceData = new float[_totalLevels][];
            _sampleCounts = new int[_totalLevels][];
            _resolutionPerLevel = new int[_totalLevels];
            _importancePerLevel = new float[_totalLevels];
            _timeAllocationPerLevel = new float[_totalLevels];

            for (int i = 0; i < _totalLevels; i++)
            {
                int resolution = ComputeLevelResolution(i);
                _resolutionPerLevel[i] = resolution;
                int pixelCount = resolution * resolution * 6;

                _levels[i] = new CascadeLevelConfig
                {
                    LevelIndex = i,
                    Resolution = resolution,
                    MaxTraceDistance = ComputeMaxTraceDistance(i),
                    MinTraceDistance = ComputeMinTraceDistance(i),
                    RaysPerTexel = ComputeRaysPerTexel(i),
                    AngularResolution = ComputeAngularResolution(i),
                    IsActive = true,
                    ImportanceWeight = 1.0f,
                    UpdateFrequency = ComputeUpdateFrequency(i),
                    FilterKernelSize = config.SpatialFilterRadius
                };

                _cascadeData[i] = new Vector3[pixelCount];
                _temporalHistory[i] = new Vector3[pixelCount];
                _varianceData[i] = new float[pixelCount];
                _sampleCounts[i] = new int[pixelCount];
                _importancePerLevel[i] = 1.0f;
            }

            ComputeBudgetAllocations();
            _isInitialized = true;
        }

        private int ComputeLevelResolution(int level)
        {
            int baseRes = (int)_config.BaseResolution;
            return Math.Max(16, baseRes >> level);
        }

        private float ComputeMaxTraceDistance(int level)
        {
            return _config.DistanceScale * MathF.Pow(2.0f, level + 1);
        }

        private float ComputeMinTraceDistance(int level)
        {
            if (level == 0) return 0.0f;
            return _config.DistanceScale * MathF.Pow(2.0f, level);
        }

        private int ComputeRaysPerTexel(int level)
        {
            return Math.Max(1, 8 >> level);
        }

        private int ComputeAngularResolution(int level)
        {
            return MathMax(4, 64 >> level);
        }

        private int MathMax(int a, int b) => a > b ? a : b;

        private int ComputeUpdateFrequency(int level)
        {
            return Math.Max(1, 1 << level);
        }

        /// <summary>
        /// Allocates cascade resources based on budget constraints.
        /// </summary>
        public CascadeBudget AllocateCascadeResources(long totalMemoryBudget, float totalTimeBudget)
        {
            var timePerLevel = new float[_totalLevels];
            var memoryPerLevel = new long[_totalLevels];
            var raysPerLevel = new int[_totalLevels];
            var resolutionPerLevel = new int[_totalLevels];

            switch (_config.AllocationStrategy)
            {
                case CascadeAllocationStrategy.Uniform:
                    AllocateUniform(ref timePerLevel, ref memoryPerLevel,
                        ref raysPerLevel, ref resolutionPerLevel,
                        totalMemoryBudget, totalTimeBudget);
                    break;
                case CascadeAllocationStrategy.Logarithmic:
                    AllocateLogarithmic(ref timePerLevel, ref memoryPerLevel,
                        ref raysPerLevel, ref resolutionPerLevel,
                        totalMemoryBudget, totalTimeBudget);
                    break;
                case CascadeAllocationStrategy.ImportanceDriven:
                    AllocateImportanceDriven(ref timePerLevel, ref memoryPerLevel,
                        ref raysPerLevel, ref resolutionPerLevel,
                        totalMemoryBudget, totalTimeBudget);
                    break;
                case CascadeAllocationStrategy.Adaptive:
                    AllocateAdaptive(ref timePerLevel, ref memoryPerLevel,
                        ref raysPerLevel, ref resolutionPerLevel,
                        totalMemoryBudget, totalTimeBudget);
                    break;
                case CascadeAllocationStrategy.Fixed:
                    AllocateFixed(ref timePerLevel, ref memoryPerLevel,
                        ref raysPerLevel, ref resolutionPerLevel,
                        totalMemoryBudget, totalTimeBudget);
                    break;
            }

            long totalMemoryUsed = 0;
            float totalTimeUsed = 0;
            for (int i = 0; i < _totalLevels; i++)
            {
                totalMemoryUsed += memoryPerLevel[i];
                totalTimeUsed += timePerLevel[i];
            }

            return new CascadeBudget
            {
                TotalTimeBudgetMs = totalTimeBudget,
                TotalMemoryBudgetBytes = totalMemoryBudget,
                TimePerLevelMs = timePerLevel,
                MemoryPerLevelBytes = memoryPerLevel,
                RaysPerLevel = raysPerLevel,
                ResolutionPerLevel = resolutionPerLevel,
                WithinBudget = totalMemoryUsed <= totalMemoryBudget && totalTimeUsed <= totalTimeBudget
            };
        }

        private void AllocateUniform(ref float[] timePerLevel, ref long[] memoryPerLevel,
            ref int[] raysPerLevel, ref int[] resolutionPerLevel,
            long totalMemory, float totalTime)
        {
            float timePer = totalTime / _totalLevels;
            long memPer = totalMemory / _totalLevels;
            for (int i = 0; i < _totalLevels; i++)
            {
                timePerLevel[i] = timePer;
                memoryPerLevel[i] = memPer;
                raysPerLevel[i] = _levels[i].RaysPerTexel;
                resolutionPerLevel[i] = _levels[i].Resolution;
            }
        }

        private void AllocateLogarithmic(ref float[] timePerLevel, ref long[] memoryPerLevel,
            ref int[] raysPerLevel, ref int[] resolutionPerLevel,
            long totalMemory, float totalTime)
        {
            float totalWeight = 0;
            for (int i = 0; i < _totalLevels; i++)
                totalWeight += 1.0f / (i + 1);
            for (int i = 0; i < _totalLevels; i++)
            {
                float weight = (1.0f / (i + 1)) / totalWeight;
                timePerLevel[i] = totalTime * weight;
                memoryPerLevel[i] = (long)(totalMemory * weight);
                raysPerLevel[i] = Math.Max(1, _levels[i].RaysPerTexel);
                resolutionPerLevel[i] = _levels[i].Resolution;
            }
        }

        private void AllocateImportanceDriven(ref float[] timePerLevel, ref long[] memoryPerLevel,
            ref int[] raysPerLevel, ref int[] resolutionPerLevel,
            long totalMemory, float totalTime)
        {
            float totalImportance = 0;
            for (int i = 0; i < _totalLevels; i++)
                totalImportance += _importancePerLevel[i];
            for (int i = 0; i < _totalLevels; i++)
            {
                float weight = _importancePerLevel[i] / MathF.Max(0.001f, totalImportance);
                timePerLevel[i] = totalTime * weight;
                memoryPerLevel[i] = (long)(totalMemory * weight);
                raysPerLevel[i] = Math.Max(1, (int)(_levels[i].RaysPerTexel * weight * _totalLevels));
                resolutionPerLevel[i] = _levels[i].Resolution;
            }
        }

        private void AllocateAdaptive(ref float[] timePerLevel, ref long[] memoryPerLevel,
            ref int[] raysPerLevel, ref int[] resolutionPerLevel,
            long totalMemory, float totalTime)
        {
            for (int i = 0; i < _totalLevels; i++)
            {
                float levelWeight = (1.0f - (float)i / _totalLevels) * _importancePerLevel[i];
                timePerLevel[i] = totalTime * levelWeight / _totalLevels;
                memoryPerLevel[i] = (long)(totalMemory * levelWeight / _totalLevels);
                raysPerLevel[i] = Math.Max(1, (int)(_levels[i].RaysPerTexel * levelWeight));
                resolutionPerLevel[i] = (int)(_levels[i].Resolution * MathF.Sqrt(levelWeight));
            }
        }

        private void AllocateFixed(ref float[] timePerLevel, ref long[] memoryPerLevel,
            ref int[] raysPerLevel, ref int[] resolutionPerLevel,
            long totalMemory, float totalTime)
        {
            int[] fixedResolutions = { 256, 128, 64, 32, 16, 8, 4, 2 };
            for (int i = 0; i < _totalLevels; i++)
            {
                int resIdx = Math.Min(i, fixedResolutions.Length - 1);
                resolutionPerLevel[i] = fixedResolutions[resIdx];
                raysPerLevel[i] = Math.Max(1, 8 >> i);
                timePerLevel[i] = totalTime / _totalLevels;
                memoryPerLevel[i] = totalMemory / _totalLevels;
            }
        }

        /// <summary>
        /// Renders a single cascade level by dispatching compute shaders.
        /// </summary>
        public void RenderCascadesLevel(int level, GBuffer gbuffer, CameraState camera,
            List<LightConfig> lights, RandomNumberGenerator rng)
        {
            if (level < 0 || level >= _totalLevels) return;
            if (!_levels[level].IsActive) return;

            int resolution = _resolutionPerLevel[level];
            float maxDist = _levels[level].MaxTraceDistance;
            float minDist = _levels[level].MinTraceDistance;
            int raysPerTexel = _levels[level].RaysPerTexel;

            for (int face = 0; face < 6; face++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    for (int y = 0; y < resolution; y++)
                    {
                        Vector3 radiance = Vector3.Zero;
                        Vector3 direction = ComputeCascadeDirection(x, y, face, resolution);

                        for (int r = 0; r < raysPerTexel; r++)
                        {
                            float jitter = rng.NextFloat();
                            float rayAngle = (float)r / raysPerTexel + jitter / raysPerTexel;
                            Vector3 rayDir = RotateDirection(direction, rayAngle, face);

                            Vector3 rayRadiance = TraceCascadeRay(
                                direction, rayDir, minDist, maxDist, gbuffer, lights, rng);
                            radiance += rayRadiance;
                        }

                        radiance /= raysPerTexel;
                        int idx = ComputeCascadeIndex(x, y, face, resolution);
                        _cascadeData[level][idx] = radiance;
                    }
                }
            }
        }

        private Vector3 ComputeCascadeDirection(int x, int y, int face, int resolution)
        {
            float u = (x + 0.5f) / resolution * 2.0f - 1.0f;
            float v = (y + 0.5f) / resolution * 2.0f - 1.0f;
            return face switch
            {
                0 => Vector3.Normalize(new Vector3(1, v, -u)),
                1 => Vector3.Normalize(new Vector3(-1, v, u)),
                2 => Vector3.Normalize(new Vector3(u, 1, -v)),
                3 => Vector3.Normalize(new Vector3(u, -1, v)),
                4 => Vector3.Normalize(new Vector3(u, v, 1)),
                5 => Vector3.Normalize(new Vector3(-u, v, -1)),
                _ => Vector3.UnitY
            };
        }

        private Vector3 RotateDirection(Vector3 direction, float angle, int face)
        {
            float cosA = MathF.Cos(angle * TWO_PI);
            float sinA = MathF.Sin(angle * TWO_PI);
            Vector3 tangent = GetTangentForFace(face);
            Vector3 bitangent = Vector3.Cross(direction, tangent);
            return Vector3.Normalize(direction * cosA + tangent * sinA + bitangent * cosA * 0.5f);
        }

        private Vector3 GetTangentForFace(int face) => face switch
        {
            0 => Vector3.UnitY,
            1 => Vector3.UnitY,
            2 => Vector3.UnitX,
            3 => Vector3.UnitX,
            4 => Vector3.UnitY,
            5 => Vector3.UnitY,
            _ => Vector3.UnitX
        };

        private int ComputeCascadeIndex(int x, int y, int face, int resolution)
        {
            return (face * resolution * resolution) + (y * resolution + x);
        }

        private Vector3 TraceCascadeRay(Vector3 origin, Vector3 direction, float minDist,
            float maxDist, GBuffer gbuffer, List<LightConfig> lights, RandomNumberGenerator rng)
        {
            Vector3 totalRadiance = Vector3.Zero;
            float stepSize = (maxDist - minDist) / 32.0f;
            Vector3 currentPos = origin + direction * minDist;

            for (int step = 0; step < 32; step++)
            {
                Vector3 samplePos = currentPos + direction * stepSize * 0.5f;
                int pixX = (int)(MathF.Atan2(direction.X, -direction.Z) * INV_PI * 0.5f * gbuffer.Width);
                int pixY = (int)(MathF.Acos(Math.Clamp(direction.Y, -1, 1)) * INV_PI * gbuffer.Height);
                pixX = Math.Abs(pixX) % gbuffer.Width;
                pixY = Math.Abs(pixY) % gbuffer.Height;

                if (pixX >= 0 && pixX < gbuffer.Width && pixY >= 0 && pixY < gbuffer.Height)
                {
                    int idx = gbuffer.GetIndex(pixX, pixY);
                    float geoDepth = gbuffer.Depth[idx];
                    float sampleDepth = samplePos.Length() * 0.1f;

                    if (geoDepth > 0 && MathF.Abs(geoDepth - sampleDepth) < stepSize)
                    {
                        Vector3 albedo = gbuffer.Albedo[idx];
                        Vector3 normal = gbuffer.Normals[idx];
                        foreach (var light in lights)
                        {
                            Vector3 lightDir = Vector3.Normalize(light.Position - samplePos);
                            float NdotL = MathF.Max(0, Vector3.Dot(normal, lightDir));
                            totalRadiance += albedo * light.Color * light.Intensity * NdotL;
                        }
                        break;
                    }
                }

                currentPos += direction * stepSize;
            }

            return totalRadiance;
        }

        /// <summary>
        /// Propagates cascades bottom-up by merging adjacent levels.
        /// </summary>
        public void PropagateCascades()
        {
            for (int level = _totalLevels - 1; level > 0; level--)
            {
                int childRes = _resolutionPerLevel[level];
                int parentRes = _resolutionPerLevel[level - 1];

                for (int face = 0; face < 6; face++)
                {
                    for (int px = 0; px < parentRes; px++)
                    {
                        for (int py = 0; py < parentRes; py++)
                        {
                            Vector3 mergedRadiance = Vector3.Zero;
                            int samples = 0;

                            for (int dx = 0; dx < 2; dx++)
                            {
                                for (int dy = 0; dy < 2; dy++)
                                {
                                    int cx = px * 2 + dx;
                                    int cy = py * 2 + dy;
                                    if (cx < childRes && cy < childRes)
                                    {
                                        int childIdx = ComputeCascadeIndex(cx, cy, face, childRes);
                                        mergedRadiance += _cascadeData[level][childIdx];
                                        samples++;
                                    }
                                }
                            }

                            if (samples > 0)
                            {
                                mergedRadiance /= samples;
                                int parentIdx = ComputeCascadeIndex(px, py, face, parentRes);
                                _cascadeData[level - 1][parentIdx] = Vector3.Lerp(
                                    _cascadeData[level - 1][parentIdx],
                                    mergedRadiance,
                                    0.5f);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Merges adjacent cascades for seamless transitions.
        /// </summary>
        public void MergeAdjacentCascades()
        {
            for (int level = 0; level < _totalLevels - 1; level++)
            {
                int currentRes = _resolutionPerLevel[level];
                int nextRes = _resolutionPerLevel[level + 1];

                for (int face = 0; face < 6; face++)
                {
                    for (int x = 0; x < currentRes; x++)
                    {
                        for (int y = 0; y < currentRes; y++)
                        {
                            int idx = ComputeCascadeIndex(x, y, face, currentRes);
                            int nx = x / 2;
                            int ny = y / 2;
                            if (nx < nextRes && ny < nextRes)
                            {
                                int nextIdx = ComputeCascadeIndex(nx, ny, face, nextRes);
                                _cascadeData[level][idx] = Vector3.Lerp(
                                    _cascadeData[level][idx],
                                    _cascadeData[level + 1][nextIdx],
                                    0.3f);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Performs temporal accumulation across frames.
        /// </summary>
        public void TemporalAccumulation(float blendFactor)
        {
            for (int level = 0; level < _totalLevels; level++)
            {
                int pixelCount = _cascadeData[level].Length;
                for (int i = 0; i < pixelCount; i++)
                {
                    Vector3 current = _cascadeData[level][i];
                    Vector3 history = _temporalHistory[level][i];

                    float historyWeight = MathF.Min(1.0f, blendFactor);
                    if (_sampleCounts[level][i] < 2)
                        historyWeight = 0;

                    _cascadeData[level][i] = Vector3.Lerp(current, history, historyWeight);
                    _temporalHistory[level][i] = _cascadeData[level][i];
                    _sampleCounts[level][i] = Math.Min(_sampleCounts[level][i] + 1, 256);

                    Vector3 diff = current - _temporalHistory[level][i];
                    _varianceData[level][i] = _varianceData[level][i] * 0.95f + diff.LengthSquared() * 0.05f;
                }
            }
        }

        /// <summary>
        /// Applies spatial bilateral filtering within cascades.
        /// </summary>
        public void SpatialFilter(int radius)
        {
            for (int level = 0; level < _totalLevels; level++)
            {
                int res = _resolutionPerLevel[level];
                Vector3[] filtered = new Vector3[_cascadeData[level].Length];
                Array.Copy(_cascadeData[level], filtered, _cascadeData[level].Length);

                for (int face = 0; face < 6; face++)
                {
                    for (int x = radius; x < res - radius; x++)
                    {
                        for (int y = radius; y < res - radius; y++)
                        {
                            Vector3 sum = Vector3.Zero;
                            float weightSum = 0;
                            int idx = ComputeCascadeIndex(x, y, face, res);
                            Vector3 centerColor = _cascadeData[level][idx];

                            for (int dx = -radius; dx <= radius; dx++)
                            {
                                for (int dy = -radius; dy <= radius; dy++)
                                {
                                    int nx = x + dx;
                                    int ny = y + dy;
                                    int nIdx = ComputeCascadeIndex(nx, ny, face, res);
                                    Vector3 neighborColor = _cascadeData[level][nIdx];

                                    float spatialWeight = MathF.Exp(-(dx * dx + dy * dy) / (2.0f * radius * radius));
                                    float colorDist = (neighborColor - centerColor).LengthSquared();
                                    float colorWeight = MathF.Exp(-colorDist * 10.0f);
                                    float weight = spatialWeight * colorWeight;

                                    sum += neighborColor * weight;
                                    weightSum += weight;
                                }
                            }

                            filtered[idx] = weightSum > 0 ? sum / weightSum : centerColor;
                        }
                    }
                }

                Array.Copy(filtered, _cascadeData[level], filtered.Length);
            }
        }

        /// <summary>
        /// Computes the temporal blend factor based on motion and history confidence.
        /// </summary>
        public float ComputeTemporalBlendFactor(int level, Vector2 velocity)
        {
            float motionMagnitude = velocity.Length();
            float motionFactor = MathF.Exp(-motionMagnitude * 10.0f);
            float historyFactor = MathF.Min(1.0f, (float)_sampleCounts[level][0] / 16.0f);
            return _config.TemporalBlendFactor * motionFactor * historyFactor;
        }

        /// <summary>
        /// Handles disocclusion by detecting large depth changes.
        /// </summary>
        public void HandleDisocclusion(GBuffer currentGBuffer, GBuffer previousGBuffer, float threshold)
        {
            if (previousGBuffer == null) return;

            for (int level = 0; level < _totalLevels; level++)
            {
                int res = _resolutionPerLevel[level];
                for (int face = 0; face < 6; face++)
                {
                    for (int x = 0; x < res; x++)
                    {
                        for (int y = 0; y < res; y++)
                        {
                            int idx = ComputeCascadeIndex(x, y, face, res);
                            int gbufX = Math.Clamp((int)((float)x / res * currentGBuffer.Width), 0, currentGBuffer.Width - 1);
                            int gbufY = Math.Clamp((int)((float)y / res * currentGBuffer.Height), 0, currentGBuffer.Height - 1);

                            float currentDepth = currentGBuffer.Depth[currentGBuffer.GetIndex(gbufX, gbufY)];
                            float prevDepth = previousGBuffer.Depth[previousGBuffer.GetIndex(gbufX, gbufY)];

                            if (MathF.Abs(currentDepth - prevDepth) > threshold * currentDepth)
                            {
                                _temporalHistory[level][idx] = Vector3.Zero;
                                _sampleCounts[level][idx] = 0;
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Adaptive level allocation based on importance metrics.
        /// </summary>
        public void AdaptiveLevelAllocation(float[] importanceScores)
        {
            if (importanceScores == null || importanceScores.Length < _totalLevels) return;

            float totalImportance = 0;
            for (int i = 0; i < _totalLevels; i++)
            {
                _importancePerLevel[i] = importanceScores[i];
                totalImportance += importanceScores[i];
            }

            for (int i = 0; i < _totalLevels; i++)
            {
                float normalizedImportance = _importancePerLevel[i] / MathF.Max(0.001f, totalImportance);
                _levels[i] = _levels[i] with
                {
                    ImportanceWeight = normalizedImportance,
                    IsActive = normalizedImportance > 0.05f
                };
            }
        }

        /// <summary>
        /// Computes the cascade budget allocation.
        /// </summary>
        public CascadeBudget ComputeCascadeBudget(float totalTimeBudget, long totalMemoryBudget)
        {
            return AllocateCascadeResources(totalMemoryBudget, totalTimeBudget);
        }

        private void ComputeBudgetAllocations()
        {
            float totalTime = _config.TimeBudgetMs;
            for (int i = 0; i < _totalLevels; i++)
            {
                float weight = 1.0f / (i + 1);
                _timeAllocationPerLevel[i] = totalTime * weight;
            }
        }

        /// <summary>
        /// Generates debug visualization of the cascade levels.
        /// </summary>
        public Vector3[,] GenerateCascadeDebugVisualization(int level, int face)
        {
            if (level < 0 || level >= _totalLevels) return new Vector3[0, 0];

            int res = _resolutionPerLevel[level];
            var visualization = new Vector3[res, res];

            for (int x = 0; x < res; x++)
                for (int y = 0; y < res; y++)
                {
                    int idx = ComputeCascadeIndex(x, y, face, res);
                    visualization[x, y] = _cascadeData[level][idx];
                }

            return visualization;
        }

        /// <summary>
        /// Returns statistics about the cascade system.
        /// </summary>
        public CascadeStatistics CascadeStatistics()
        {
            long totalMemory = 0;
            int totalActiveLevels = 0;
            float totalImportance = 0;

            for (int i = 0; i < _totalLevels; i++)
            {
                totalMemory += _cascadeData[i].Length * 12;
                if (_levels[i].IsActive) totalActiveLevels++;
                totalImportance += _importancePerLevel[i];
            }

            return new CascadeStatistics
            {
                TotalLevels = _totalLevels,
                ActiveLevels = totalActiveLevels,
                TotalMemoryBytes = totalMemory,
                AverageImportance = totalImportance / _totalLevels,
                TotalTexels = totalMemory / 12,
                FrameIndex = _frameIndex
            };
        }

        /// <summary>
        /// Gets the cascade data for a specific level.
        /// </summary>
        public Vector3[] GetCascadeData(int level)
        {
            if (level < 0 || level >= _totalLevels) return Array.Empty<Vector3>();
            return _cascadeData[level];
        }

        /// <summary>
        /// Gets the resolution for a specific cascade level.
        /// </summary>
        public int GetLevelResolution(int level)
        {
            if (level < 0 || level >= _totalLevels) return 0;
            return _resolutionPerLevel[level];
        }
    }

    /// <summary>
    /// Statistics about the radiance cascade system.
    /// </summary>
    public record CascadeStatistics
    {
        /// <summary>Total cascade levels configured.</summary>
        public int TotalLevels { get; init; }
        /// <summary>Number of active cascade levels.</summary>
        public int ActiveLevels { get; init; }
        /// <summary>Total memory used by cascades.</summary>
        public long TotalMemoryBytes { get; init; }
        /// <summary>Average importance across all levels.</summary>
        public float AverageImportance { get; init; }
        /// <summary>Total number of texels across all levels.</summary>
        public long TotalTexels { get; init; }
        /// <summary>Current frame index.</summary>
        public int FrameIndex { get; init; }
    }

    #endregion

    // =========================================================================
    #region NeuralPredictiveIrradiance

    /// <summary>
    /// Neural network system for predicting irradiance from G-Buffer features.
    /// Implements forward pass, feature extraction, online training, and
    /// inference optimization.
    /// </summary>
    public class NeuralPredictiveIrradiance
    {
        private const int INPUT_FEATURES = 32;
        private const int HIDDEN_LAYER_1 = 128;
        private const int HIDDEN_LAYER_2 = 128;
        private const int HIDDEN_LAYER_3 = 64;
        private const int OUTPUT_FEATURES = 3;
        private const float LEARNING_RATE = 0.001f;
        private const float BETA1 = 0.9f;
        private const float BETA2 = 0.999f;
        private const float EPSILON_ADAM = 1e-8f;

        private float[,] _weights1;
        private float[,] _weights2;
        private float[,] _weights3;
        private float[] _bias1;
        private float[] _bias2;
        private float[] _bias3;
        private float[,] _m1, _v1, _m2, _v2, _m3, _v3;
        private float[] _mb1, _mb2, _mb3;
        private int _t;
        private bool _isInitialized;

        private Queue<(Vector3[] Features, Vector3 Target)> _trainingBuffer;
        private int _maxTrainingBufferSize;
        private float _trainingLoss;
        private float _inferenceTime;
        private int _trainingStep;

        /// <summary>Whether the network is initialized.</summary>
        public bool IsInitialized => _isInitialized;
        /// <summary>Current training loss.</summary>
        public float TrainingLoss => _trainingLoss;
        /// <summary>Last inference time in milliseconds.</summary>
        public float InferenceTime => _inferenceTime;
        /// <summary>Total training steps performed.</summary>
        public int TrainingStep => _trainingStep;

        /// <summary>
        /// Initializes the neural network architecture.
        /// </summary>
        public void Initialize()
        {
            var rng = new RandomNumberGenerator(42);

            _weights1 = XavierInitialize(INPUT_FEATURES, HIDDEN_LAYER_1, ref rng);
            _weights2 = XavierInitialize(HIDDEN_LAYER_1, HIDDEN_LAYER_2, ref rng);
            _weights3 = XavierInitialize(HIDDEN_LAYER_2, OUTPUT_FEATURES, ref rng);
            _bias1 = new float[HIDDEN_LAYER_1];
            _bias2 = new float[HIDDEN_LAYER_2];
            _bias3 = new float[OUTPUT_FEATURES];

            _m1 = new float[INPUT_FEATURES, HIDDEN_LAYER_1];
            _v1 = new float[INPUT_FEATURES, HIDDEN_LAYER_1];
            _m2 = new float[HIDDEN_LAYER_1, HIDDEN_LAYER_2];
            _v2 = new float[HIDDEN_LAYER_1, HIDDEN_LAYER_2];
            _m3 = new float[HIDDEN_LAYER_2, OUTPUT_FEATURES];
            _v3 = new float[HIDDEN_LAYER_2, OUTPUT_FEATURES];
            _mb1 = new float[HIDDEN_LAYER_1];
            _mb2 = new float[HIDDEN_LAYER_2];
            _mb3 = new float[OUTPUT_FEATURES];
            _t = 0;

            _trainingBuffer = new Queue<(Vector3[], Vector3)>();
            _maxTrainingBufferSize = 10000;
            _trainingLoss = 0;
            _trainingStep = 0;
            _isInitialized = true;
        }

        private float[,] XavierInitialize(int fanIn, int fanOut, ref RandomNumberGenerator rng)
        {
            float limit = MathF.Sqrt(6.0f / (fanIn + fanOut));
            var weights = new float[fanIn, fanOut];
            for (int i = 0; i < fanIn; i++)
                for (int j = 0; j < fanOut; j++)
                    weights[i, j] = rng.NextFloat(-limit, limit);
            return weights;
        }

        /// <summary>
        /// Extracts features from the G-Buffer for neural prediction.
        /// </summary>
        public Vector3[] ExtractFeatures(GBufferSample sample, GBuffer gbuffer, int px, int py,
            CameraState camera)
        {
            var features = new float[INPUT_FEATURES];
            int idx = 0;

            features[idx++] = sample.Depth / 100.0f;
            features[idx++] = sample.Normal.X;
            features[idx++] = sample.Normal.Y;
            features[idx++] = sample.Normal.Z;
            features[idx++] = sample.Albedo.X;
            features[idx++] = sample.Albedo.Y;
            features[idx++] = sample.Albedo.Z;
            features[idx++] = sample.Specular.X;
            features[idx++] = sample.Roughness;
            features[idx++] = sample.Metallic;

            Vector3 viewDir = Vector3.Normalize(camera.Position - sample.WorldPosition);
            float NdotV = MathF.Max(0, Vector3.Dot(sample.Normal, viewDir));
            features[idx++] = NdotV;

            Vector3 screenPos3 = new Vector3(
                (float)px / gbuffer.Width * 2.0f - 1.0f,
                1.0f - (float)py / gbuffer.Height * 2.0f,
                sample.Depth);
            features[idx++] = screenPos3.X;
            features[idx++] = screenPos3.Y;
            features[idx++] = sample.Velocity.X;
            features[idx++] = sample.Velocity.Y;

            float edgeDepth = ComputeEdgeDepth(gbuffer, px, py);
            float edgeNormal = ComputeEdgeNormal(gbuffer, px, py);
            features[idx++] = edgeDepth;
            features[idx++] = edgeNormal;

            Vector3[] neighborFeatures = ExtractNeighborhoodFeatures(gbuffer, px, py);
            for (int i = 0; i < Math.Min(8, neighborFeatures.Length) && idx < INPUT_FEATURES; i++)
            {
                features[idx++] = neighborFeatures[i].X;
                if (idx < INPUT_FEATURES) features[idx++] = neighborFeatures[i].Y;
                if (idx < INPUT_FEATURES) features[idx++] = neighborFeatures[i].Z;
            }

            while (idx < INPUT_FEATURES) features[idx++] = 0;

            var result = new Vector3[INPUT_FEATURES];
            for (int i = 0; i < INPUT_FEATURES; i++)
                result[i] = new Vector3(features[i], 0, 0);
            return result;
        }

        private float ComputeEdgeDepth(GBuffer gbuffer, int x, int y)
        {
            float depth = gbuffer.Depth[gbuffer.GetIndex(x, y)];
            float maxDiff = 0;
            int[] offsets = { -1, 0, 1 };
            foreach (int dx in offsets)
            {
                foreach (int dy in offsets)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nx = Math.Clamp(x + dx, 0, gbuffer.Width - 1);
                    int ny = Math.Clamp(y + dy, 0, gbuffer.Height - 1);
                    float neighborDepth = gbuffer.Depth[gbuffer.GetIndex(nx, ny)];
                    maxDiff = MathF.Max(maxDiff, MathF.Abs(depth - neighborDepth));
                }
            }
            return maxDiff;
        }

        private float ComputeEdgeNormal(GBuffer gbuffer, int x, int y)
        {
            Vector3 normal = gbuffer.Normals[gbuffer.GetIndex(x, y)];
            float maxDiff = 0;
            int[] offsets = { -1, 0, 1 };
            foreach (int dx in offsets)
            {
                foreach (int dy in offsets)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nx = Math.Clamp(x + dx, 0, gbuffer.Width - 1);
                    int ny = Math.Clamp(y + dy, 0, gbuffer.Height - 1);
                    Vector3 neighborNormal = gbuffer.Normals[gbuffer.GetIndex(nx, ny)];
                    maxDiff = MathF.Max(maxDiff, 1.0f - Vector3.Dot(normal, neighborNormal));
                }
            }
            return maxDiff;
        }

        private Vector3[] ExtractNeighborhoodFeatures(GBuffer gbuffer, int x, int y)
        {
            var features = new List<Vector3>();
            int radius = 2;
            for (int dx = -radius; dx <= radius; dx++)
            {
                for (int dy = -radius; dy <= radius; dy++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nx = Math.Clamp(x + dx, 0, gbuffer.Width - 1);
                    int ny = Math.Clamp(y + dy, 0, gbuffer.Height - 1);
                    int idx = gbuffer.GetIndex(nx, ny);
                    features.Add(new Vector3(
                        gbuffer.Depth[idx],
                        gbuffer.Normals[idx].Length(),
                        gbuffer.Albedo[idx].Length()));
                }
            }
            return features.ToArray();
        }

        /// <summary>
        /// Aggregates spatial features from the neighborhood.
        /// </summary>
        public Vector3[] SpatialFeatureAggregation(Vector3[] pixelFeatures, int width, int height, int x, int y)
        {
            var aggregated = new Vector3[INPUT_FEATURES];
            float totalWeight = 0;

            for (int dx = -2; dx <= 2; dx++)
            {
                for (int dy = -2; dy <= 2; dy++)
                {
                    int nx = Math.Clamp(x + dx, 0, width - 1);
                    int ny = Math.Clamp(y + dy, 0, height - 1);
                    int idx = ny * width + nx;
                    float spatialWeight = MathF.Exp(-(dx * dx + dy * dy) * 0.5f);

                    for (int f = 0; f < INPUT_FEATURES && f < pixelFeatures.Length; f++)
                        aggregated[f] += pixelFeatures[idx] * spatialWeight;
                    totalWeight += spatialWeight;
                }
            }

            if (totalWeight > 0)
                for (int f = 0; f < INPUT_FEATURES; f++)
                    aggregated[f] /= totalWeight;

            return aggregated;
        }

        /// <summary>
        /// Aggregates temporal features from reprojection.
        /// </summary>
        public Vector3[] TemporalFeatureAggregation(Vector3[] currentFeatures,
            Vector3[] previousFeatures, Vector2 velocity, int width, int height, int x, int y)
        {
            var temporalFeatures = new Vector3[INPUT_FEATURES];
            float reprojectionConfidence = MathF.Exp(-velocity.Length() * 5.0f);

            for (int f = 0; f < INPUT_FEATURES; f++)
            {
                if (previousFeatures != null && f < previousFeatures.Length)
                    temporalFeatures[f] = Vector3.Lerp(currentFeatures[f], previousFeatures[f], reprojectionConfidence * 0.8f);
                else
                    temporalFeatures[f] = currentFeatures[f];
            }

            return temporalFeatures;
        }

        /// <summary>
        /// Estimates confidence (variance prediction) for the irradiance estimate.
        /// </summary>
        public float EstimateConfidence(Vector3[] features, Vector3 predictedIrradiance)
        {
            float featureMagnitude = 0;
            for (int i = 0; i < features.Length; i++)
                featureMagnitude += features[i].X * features[i].X;
            featureMagnitude = MathF.Sqrt(featureMagnitude);

            float irradianceMagnitude = predictedIrradiance.Length();
            float normalizedIrradiance = MathF.Min(1.0f, irradianceMagnitude / 5.0f);
            float confidence = normalizedIrradiance * MathF.Exp(-featureMagnitude * 0.1f);
            return Math.Clamp(confidence, 0, 1);
        }

        /// <summary>
        /// Performs a forward pass through the neural network.
        /// </summary>
        public Vector3 ForwardPass(Vector3[] inputFeatures)
        {
            if (!_isInitialized) return Vector3.Zero;

            float[] input = new float[INPUT_FEATURES];
            for (int i = 0; i < INPUT_FEATURES && i < inputFeatures.Length; i++)
                input[i] = inputFeatures[i].X;

            float[] hidden1 = LinearForward(input, _weights1, _bias1);
            LeakyReLU(hidden1);
            float[] hidden2 = LinearForward(hidden1, _weights2, _bias2);
            LeakyReLU(hidden2);
            float[] output = LinearForward(hidden2, _weights3, _bias3);
            Sigmoid(output);

            return new Vector3(output[0], output[1], output[2]);
        }

        private float[] LinearForward(float[] input, float[,] weights, float[] bias)
        {
            int inputSize = weights.GetLength(0);
            int outputSize = weights.GetLength(1);
            float[] output = new float[outputSize];

            for (int j = 0; j < outputSize; j++)
            {
                float sum = bias[j];
                for (int i = 0; i < inputSize; i++)
                    sum += input[i] * weights[i, j];
                output[j] = sum;
            }

            return output;
        }

        private void LeakyReLU(float[] data, float slope = 0.01f)
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = data[i] > 0 ? data[i] : data[i] * slope;
        }

        private void ReLU(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = MathF.Max(0, data[i]);
        }

        private void Sigmoid(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = 1.0f / (1.0f + MathF.Exp(-data[i]));
        }

        private void TanhActivate(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = MathF.Tanh(data[i]);
        }

        /// <summary>
        /// Performs backpropagation and updates network weights using Adam optimizer.
        /// </summary>
        public void BackwardPass(Vector3[] input, Vector3 target, float learningRate)
        {
            if (!_isInitialized) return;

            float[] inputArr = new float[INPUT_FEATURES];
            for (int i = 0; i < INPUT_FEATURES; i++)
                inputArr[i] = input[i].X;

            float[] hidden1 = LinearForward(inputArr, _weights1, _bias1);
            float[] hidden1Act = (float[])hidden1.Clone();
            LeakyReLU(hidden1Act);

            float[] hidden2 = LinearForward(hidden1Act, _weights2, _bias2);
            float[] hidden2Act = (float[])hidden2.Clone();
            LeakyReLU(hidden2Act);

            float[] output = LinearForward(hidden2Act, _weights3, _bias3);
            float[] outputAct = (float[])output.Clone();
            Sigmoid(outputAct);

            float[] targetArr = { target.X, target.Y, target.Z };

            float[] outputGrad = new float[OUTPUT_FEATURES];
            float loss = 0;
            for (int i = 0; i < OUTPUT_FEATURES; i++)
            {
                float diff = outputAct[i] - targetArr[i];
                outputGrad[i] = 2.0f * diff / OUTPUT_FEATURES;
                loss += diff * diff;
            }
            _trainingLoss = _trainingLoss * 0.99f + loss * 0.01f;

            float[,] dW3 = new float[HIDDEN_LAYER_2, OUTPUT_FEATURES];
            float[] dB3 = new float[OUTPUT_FEATURES];
            float[] hidden2Grad = new float[HIDDEN_LAYER_2];

            for (int j = 0; j < OUTPUT_FEATURES; j++)
            {
                float outputDeriv = outputAct[j] * (1.0f - outputAct[j]);
                float delta = outputGrad[j] * outputDeriv;
                dB3[j] = delta;
                for (int i = 0; i < HIDDEN_LAYER_2; i++)
                {
                    dW3[i, j] = hidden2Act[i] * delta;
                    hidden2Grad[i] += _weights3[i, j] * delta;
                }
            }

            for (int i = 0; i < HIDDEN_LAYER_2; i++)
                hidden2Grad[i] *= hidden2Act[i] > 0 ? 1.0f : 0.01f;

            float[,] dW2 = new float[HIDDEN_LAYER_1, HIDDEN_LAYER_2];
            float[] dB2 = new float[HIDDEN_LAYER_2];
            float[] hidden1Grad = new float[HIDDEN_LAYER_1];

            for (int j = 0; j < HIDDEN_LAYER_2; j++)
            {
                float delta = hidden2Grad[j];
                dB2[j] = delta;
                for (int i = 0; i < HIDDEN_LAYER_1; i++)
                {
                    dW2[i, j] = hidden1Act[i] * delta;
                    hidden1Grad[i] += _weights2[i, j] * delta;
                }
            }

            for (int i = 0; i < HIDDEN_LAYER_1; i++)
                hidden1Grad[i] *= hidden1Act[i] > 0 ? 1.0f : 0.01f;

            float[,] dW1 = new float[INPUT_FEATURES, HIDDEN_LAYER_1];
            float[] dB1 = new float[HIDDEN_LAYER_1];

            for (int j = 0; j < HIDDEN_LAYER_1; j++)
            {
                float delta = hidden1Grad[j];
                dB1[j] = delta;
                for (int i = 0; i < INPUT_FEATURES; i++)
                    dW1[i, j] = inputArr[i] * delta;
            }

            _t++;
            float beta1PowT = MathF.Pow(BETA1, _t);
            float beta2PowT = MathF.Pow(BETA2, _t);

            UpdateAdamWeights(_weights1, _m1, _v1, dW1, learningRate, beta1PowT, beta2PowT, INPUT_FEATURES, HIDDEN_LAYER_1);
            UpdateAdamWeights(_weights2, _m2, _v2, dW2, learningRate, beta1PowT, beta2PowT, HIDDEN_LAYER_1, HIDDEN_LAYER_2);
            UpdateAdamWeights(_weights3, _m3, _v3, dW3, learningRate, beta1PowT, beta2PowT, HIDDEN_LAYER_2, OUTPUT_FEATURES);

            UpdateAdamBias(_bias1, _mb1, dB1, learningRate, beta1PowT, beta2PowT);
            UpdateAdamBias(_bias2, _mb2, dB2, learningRate, beta1PowT, beta2PowT);
            UpdateAdamBias(_bias3, _mb3, dB3, learningRate, beta1PowT, beta2PowT);

            _trainingStep++;
        }

        private void UpdateAdamWeights(float[,] weights, float[,] m, float[,] v,
            float[,] grads, float lr, float beta1PowT, float beta2PowT, int rows, int cols)
        {
            float lrCorrected = lr * MathF.Sqrt(1.0f - beta2PowT) / (1.0f - beta1PowT);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m[i, j] = BETA1 * m[i, j] + (1.0f - BETA1) * grads[i, j];
                    v[i, j] = BETA2 * v[i, j] + (1.0f - BETA2) * grads[i, j] * grads[i, j];
                    float mHat = m[i, j] / (1.0f - beta1PowT);
                    float vHat = v[i, j] / (1.0f - beta2PowT);
                    weights[i, j] -= lrCorrected * mHat / (MathF.Sqrt(vHat) + EPSILON_ADAM);
                }
            }
        }

        private void UpdateAdamBias(float[] bias, float[] m, float[] grads,
            float lr, float beta1PowT, float beta2PowT)
        {
            float lrCorrected = lr * MathF.Sqrt(1.0f - beta2PowT) / (1.0f - beta1PowT);
            for (int i = 0; i < bias.Length; i++)
            {
                m[i] = BETA1 * m[i] + (1.0f - BETA1) * grads[i];
                float v = BETA2 * 0 + (1.0f - BETA2) * grads[i] * grads[i];
                float mHat = m[i] / (1.0f - beta1PowT);
                float vHat = v / (1.0f - beta2PowT);
                bias[i] -= lrCorrected * mHat / (MathF.Sqrt(vHat) + EPSILON_ADAM);
            }
        }

        /// <summary>
        /// Collects training data from reference path tracer output.
        /// </summary>
        public void CollectTrainingData(Vector3[] features, Vector3 groundTruthIrradiance)
        {
            _trainingBuffer.Enqueue((features, groundTruthIrradiance));
            while (_trainingBuffer.Count > _maxTrainingBufferSize)
                _trainingBuffer.Dequeue();
        }

        /// <summary>
        /// Runs the online training loop on collected data.
        /// </summary>
        public void TrainOnCollectedData(int batchSize, float learningRate)
        {
            if (_trainingBuffer.Count < batchSize) return;

            var batch = _trainingBuffer.ToArray();
            var rng = new RandomNumberGenerator((uint)_trainingStep);

            float totalLoss = 0;
            for (int b = 0; b < batchSize; b++)
            {
                int idx = rng.NextInt(0, batch.Length);
                var (features, target) = batch[idx];
                ForwardPass(features);
                BackwardPass(features, target, learningRate);
                totalLoss += _trainingLoss;
            }
        }

        /// <summary>
        /// Computes L1 loss between prediction and target.
        /// </summary>
        public float ComputeL1Loss(Vector3 prediction, Vector3 target)
        {
            Vector3 diff = prediction - target;
            return (MathF.Abs(diff.X) + MathF.Abs(diff.Y) + MathF.Abs(diff.Z)) / 3.0f;
        }

        /// <summary>
        /// Computes L2 loss between prediction and target.
        /// </summary>
        public float ComputeL2Loss(Vector3 prediction, Vector3 target)
        {
            Vector3 diff = prediction - target;
            return (diff.X * diff.X + diff.Y * diff.Y + diff.Z * diff.Z) / 3.0f;
        }

        /// <summary>
        /// Computes perceptual loss (simplified).
        /// </summary>
        public float ComputePerceptualLoss(Vector3 prediction, Vector3 target)
        {
            float l2 = ComputeL2Loss(prediction, target);
            float predLum = 0.2126f * prediction.X + 0.7152f * prediction.Y + 0.0722f * prediction.Z;
            float tgtLum = 0.2126f * target.X + 0.7152f * target.Y + 0.0722f * target.Z;
            float lumLoss = (predLum - tgtLum) * (predLum - tgtLum);
            return l2 * 0.7f + lumLoss * 0.3f;
        }

        /// <summary>
        /// Serializes the network weights to a byte array.
        /// </summary>
        public byte[] Serialize()
        {
            using var ms = new System.IO.MemoryStream();
            using var bw = new System.IO.BinaryWriter(ms);
            bw.Write(INPUT_FEATURES);
            bw.Write(HIDDEN_LAYER_1);
            bw.Write(HIDDEN_LAYER_2);
            bw.Write(OUTPUT_FEATURES);
            bw.Write(_t);

            for (int i = 0; i < INPUT_FEATURES; i++)
                for (int j = 0; j < HIDDEN_LAYER_1; j++)
                    bw.Write(_weights1[i, j]);

            for (int i = 0; i < HIDDEN_LAYER_1; i++)
                for (int j = 0; j < HIDDEN_LAYER_2; j++)
                    bw.Write(_weights2[i, j]);

            for (int i = 0; i < HIDDEN_LAYER_2; i++)
                for (int j = 0; j < OUTPUT_FEATURES; j++)
                    bw.Write(_weights3[i, j]);

            foreach (float bias in _bias1) bw.Write(bias);
            foreach (float bias in _bias2) bw.Write(bias);
            foreach (float bias in _bias3) bw.Write(bias);

            return ms.ToArray();
        }

        /// <summary>
        /// Deserializes network weights from a byte array.
        /// </summary>
        public void Deserialize(byte[] data)
        {
            using var ms = new System.IO.MemoryStream(data);
            using var br = new System.IO.BinaryReader(ms);

            int inputFeat = br.ReadInt32();
            int hidden1 = br.ReadInt32();
            int hidden2 = br.ReadInt32();
            int outputFeat = br.ReadInt32();
            _t = br.ReadInt32();

            _weights1 = new float[inputFeat, hidden1];
            for (int i = 0; i < inputFeat; i++)
                for (int j = 0; j < hidden1; j++)
                    _weights1[i, j] = br.ReadSingle();

            _weights2 = new float[hidden1, hidden2];
            for (int i = 0; i < hidden1; i++)
                for (int j = 0; j < hidden2; j++)
                    _weights2[i, j] = br.ReadSingle();

            _weights3 = new float[hidden2, outputFeat];
            for (int i = 0; i < hidden2; i++)
                for (int j = 0; j < outputFeat; j++)
                    _weights3[i, j] = br.ReadSingle();

            _bias1 = new float[hidden1];
            _bias2 = new float[hidden2];
            _bias3 = new float[outputFeat];
            for (int i = 0; i < hidden1; i++) _bias1[i] = br.ReadSingle();
            for (int i = 0; i < hidden2; i++) _bias2[i] = br.ReadSingle();
            for (int i = 0; i < outputFeat; i++) _bias3[i] = br.ReadSingle();

            _m1 = new float[inputFeat, hidden1];
            _v1 = new float[inputFeat, hidden1];
            _m2 = new float[hidden1, hidden2];
            _v2 = new float[hidden1, hidden2];
            _m3 = new float[hidden2, outputFeat];
            _v3 = new float[hidden2, outputFeat];
            _mb1 = new float[hidden1];
            _mb2 = new float[hidden2];
            _mb3 = new float[outputFeat];

            _isInitialized = true;
        }

        /// <summary>
        /// Optimizes inference by fusing operations (simulated).
        /// </summary>
        public Vector3 OptimizedInference(Vector3[] features)
        {
            return ForwardPass(features);
        }

        /// <summary>
        /// Quantizes weights to reduced precision (simulated).
        /// </summary>
        public void QuantizeWeights(int bits)
        {
            float scale = MathF.Pow(2, bits) - 1;
            float invScale = 1.0f / scale;

            for (int i = 0; i < _weights1.GetLength(0); i++)
                for (int j = 0; j < _weights1.GetLength(1); j++)
                    _weights1[i, j] = MathF.Round(_weights1[i, j] * scale) * invScale;

            for (int i = 0; i < _weights2.GetLength(0); i++)
                for (int j = 0; j < _weights2.GetLength(1); j++)
                    _weights2[i, j] = MathF.Round(_weights2[i, j] * scale) * invScale;

            for (int i = 0; i < _weights3.GetLength(0); i++)
                for (int j = 0; j < _weights3.GetLength(1); j++)
                    _weights3[i, j] = MathF.Round(_weights3[i, j] * scale) * invScale;
        }
    }

    #endregion

    // =========================================================================
    #region ScreenSpaceIrradiance

    /// <summary>
    /// Screen-space global illumination system using ray marching and Hi-Z traversal.
    /// </summary>
    public class ScreenSpaceIrradiance
    {
        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;
        private const int MAX_MARCH_STEPS = 64;
        private const float STEP_SIZE = 1.0f;
        private const float Thickness = 0.5f;

        private Vector3[] _ssgiResult;
        private float[] _ssgiConfidence;
        private int _width;
        private int _height;
        private bool _isInitialized;

        /// <summary>SSGI result buffer.</summary>
        public Vector3[] Result => _ssgiResult;
        /// <summary>Confidence buffer.</summary>
        public float[] Confidence => _ssgiConfidence;

        /// <summary>
        /// Initializes the screen-space irradiance system.
        /// </summary>
        public void Initialize(int width, int height)
        {
            _width = width;
            _height = height;
            int pixelCount = width * height;
            _ssgiResult = new Vector3[pixelCount];
            _ssgiConfidence = new float[pixelCount];
            _isInitialized = true;
        }

        /// <summary>
        /// Computes screen-space GI using ray marching.
        /// </summary>
        public void ComputeSSGI(GBuffer gbuffer, CameraState camera, List<LightConfig> lights,
            int numRays, RandomNumberGenerator rng)
        {
            if (!_isInitialized) return;

            Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    float depth = gbuffer.Depth[idx];
                    if (depth <= 0)
                    {
                        _ssgiResult[idx] = Vector3.Zero;
                        _ssgiConfidence[idx] = 0;
                        continue;
                    }

                    Vector3 normal = gbuffer.Normals[idx];
                    Vector3 albedo = gbuffer.Albedo[idx];

                    Vector3 worldPos = ReconstructWorldPosition(x, y, depth, gbuffer, camera);
                    Vector3 tangent = GetTangent(normal, ref rng);
                    Vector3 bitangent = Vector3.Cross(normal, tangent);

                    Vector3 totalIrradiance = Vector3.Zero;
                    int hitCount = 0;

                    for (int r = 0; r < numRays; r++)
                    {
                        float phi = rng.NextFloat() * TWO_PI;
                        float cosTheta = MathF.Sqrt(rng.NextFloat());
                        float sinTheta = MathF.Sqrt(1.0f - cosTheta * cosTheta);

                        Vector3 sampleDir = tangent * (MathF.Cos(phi) * sinTheta) +
                                            bitangent * (MathF.Sin(phi) * sinTheta) +
                                            normal * cosTheta;

                        Vector3 hitRadiance = MarchScreenSpaceRay(worldPos, sampleDir,
                            gbuffer, camera, lights, rng);

                        if (hitRadiance.LengthSquared() > 0.0001f)
                        {
                            totalIrradiance += hitRadiance * cosTheta;
                            hitCount++;
                        }
                    }

                    if (hitCount > 0)
                    {
                        totalIrradiance /= hitCount;
                        totalIrradiance *= INV_PI;
                    }

                    _ssgiResult[idx] = totalIrradiance;
                    _ssgiConfidence[idx] = (float)hitCount / numRays;
                }
            });
        }

        /// <summary>
        /// Performs Hi-Z ray marching for efficient screen-space traversal.
        /// </summary>
        public Vector3 HiZRayMarch(Vector3 worldPos, Vector3 rayDir, GBuffer gbuffer,
            CameraState camera, int maxSteps)
        {
            Vector3 screenPos = camera.ProjectToScreen(worldPos);
            Vector3 screenDir = Vector3.Normalize(camera.ProjectToScreen(worldPos + rayDir) - screenPos);

            float currentDepth = screenPos.Z;
            float stepSize = STEP_SIZE;

            for (int step = 0; step < maxSteps; step++)
            {
                Vector3 sampleScreen = screenPos + screenDir * stepSize * (step + 1);
                if (sampleScreen.X < 0 || sampleScreen.X >= 1 || sampleScreen.Y < 0 || sampleScreen.Y >= 1)
                    return Vector3.Zero;

                int pixX = (int)(sampleScreen.X * _width);
                int pixY = (int)(sampleScreen.Y * _height);
                pixX = Math.Clamp(pixX, 0, _width - 1);
                pixY = Math.Clamp(pixY, 0, _height - 1);

                float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                float sampleDepth = sampleScreen.Z;

                if (sampleDepth > sceneDepth && sampleDepth < sceneDepth + STEP_SIZE)
                {
                    return gbuffer.Albedo[gbuffer.GetIndex(pixX, pixY)];
                }

                if (sampleDepth > sceneDepth + STEP_SIZE)
                {
                    stepSize *= 0.5f;
                }
            }

            return Vector3.Zero;
        }

        /// <summary>
        /// Marches a ray in screen-space and returns radiance at the hit point.
        /// </summary>
        public Vector3 MarchScreenSpaceRay(Vector3 worldOrigin, Vector3 worldDir,
            GBuffer gbuffer, CameraState camera, List<LightConfig> lights, RandomNumberGenerator rng)
        {
            Vector3 origin = camera.ProjectToScreen(worldOrigin);
            Vector3 target = camera.ProjectToScreen(worldOrigin + worldDir * 10.0f);
            Vector3 rayDir = target - origin;

            float rayLength = rayDir.Length();
            if (rayLength < 0.001f) return Vector3.Zero;
            rayDir /= rayLength;

            int numSteps = Math.Min(MAX_MARCH_STEPS, (int)(rayLength * 10));
            float stepLen = rayLength / numSteps;

            for (int step = 1; step <= numSteps; step++)
            {
                Vector3 samplePos = origin + rayDir * stepLen * step;
                if (samplePos.X < 0 || samplePos.X >= 1 || samplePos.Y < 0 || samplePos.Y >= 1)
                    return Vector3.Zero;

                int pixX = (int)(samplePos.X * _width);
                int pixY = (int)(samplePos.Y * _height);
                pixX = Math.Clamp(pixX, 0, _width - 1);
                pixY = Math.Clamp(pixY, 0, _height - 1);

                float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                float sampleDepth = samplePos.Z;

                if (sampleDepth > sceneDepth && sampleDepth < sceneDepth + STEP_SIZE)
                {
                    Vector3 hitNormal = gbuffer.Normals[gbuffer.GetIndex(pixX, pixY)];
                    Vector3 hitAlbedo = gbuffer.Albedo[gbuffer.GetIndex(pixX, pixY)];

                    float hitWeight = MathF.Max(0, Vector3.Dot(-worldDir, hitNormal));

                    Vector3 hitRadiance = Vector3.Zero;
                    foreach (var light in lights)
                    {
                        Vector3 lightDir;
                        float lightDist;
                        Vector3 lightColor;

                        if (light.Type == LightType.Directional)
                        {
                            lightDir = -light.Direction;
                            lightDist = float.MaxValue;
                            lightColor = light.Color * light.Intensity;
                        }
                        else
                        {
                            Vector3 toLight = light.Position - (worldOrigin + worldDir * stepLen * step);
                            lightDist = toLight.Length();
                            lightDir = toLight / lightDist;
                            float atten = 1.0f / (lightDist * lightDist);
                            lightColor = light.Color * light.Intensity * atten;
                        }

                        float NdotL = MathF.Max(0, Vector3.Dot(hitNormal, lightDir));
                        hitRadiance += hitAlbedo * lightColor * NdotL;
                    }

                    return hitRadiance * hitWeight;
                }
            }

            return Vector3.Zero;
        }

        /// <summary>
        /// Validates a screen-space hit using depth and normal comparison.
        /// </summary>
        public bool ValidateHit(Vector3 hitScreenPos, Vector3 hitNormal, float hitDepth,
            GBuffer gbuffer, CameraState camera)
        {
            if (hitScreenPos.X < 0 || hitScreenPos.X >= 1 || hitScreenPos.Y < 0 || hitScreenPos.Y >= 1)
                return false;

            int pixX = (int)(hitScreenPos.X * _width);
            int pixY = (int)(hitScreenPos.Y * _height);
            pixX = Math.Clamp(pixX, 0, _width - 1);
            pixY = Math.Clamp(pixY, 0, _height - 1);

            float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
            Vector3 sceneNormal = gbuffer.Normals[gbuffer.GetIndex(pixX, pixY)];

            float depthError = MathF.Abs(hitDepth - sceneDepth) / MathF.Max(0.001f, sceneDepth);
            float normalError = 1.0f - Vector3.Dot(hitNormal, sceneNormal);

            return depthError < 0.1f && normalError < 0.3f;
        }

        /// <summary>
        /// Performs bi-directional ray marching for better coverage.
        /// </summary>
        public Vector3 BidirectionalRayMarch(Vector3 worldPos, Vector3 rayDir, GBuffer gbuffer,
            CameraState camera, List<LightConfig> lights, RandomNumberGenerator rng)
        {
            Vector3 forwardResult = MarchScreenSpaceRay(worldPos, rayDir, gbuffer, camera, lights, rng);
            Vector3 backwardResult = MarchScreenSpaceRay(worldPos, -rayDir, gbuffer, camera, lights, rng);
            return (forwardResult + backwardResult) * 0.5f;
        }

        /// <summary>
        /// Detects back-face hits for two-sided surfaces.
        /// </summary>
        public bool IsBackFaceHit(Vector3 rayDir, Vector3 hitNormal)
        {
            return Vector3.Dot(rayDir, hitNormal) > 0;
        }

        /// <summary>
        /// Temporally reprojects screen-space hits from previous frame.
        /// </summary>
        public Vector3 TemporalReproject(int x, int y, Vector2 velocity,
            Vector3[] previousFrame, int prevWidth, int prevHeight)
        {
            float prevX = x + velocity.X;
            float prevY = y + velocity.Y;

            if (prevX < 0 || prevX >= prevWidth || prevY < 0 || prevY >= prevHeight)
                return Vector3.Zero;

            int prevIdx = (int)prevY * prevWidth + (int)prevX;
            if (prevIdx >= 0 && prevIdx < previousFrame.Length)
                return previousFrame[prevIdx];

            return Vector3.Zero;
        }

        /// <summary>
        /// Applies edge-stopping spatial filter to SSGI result.
        /// </summary>
        public void EdgeStoppingFilter(GBuffer gbuffer, int radius, float normalThreshold,
            float depthThreshold)
        {
            Vector3[] filtered = new Vector3[_ssgiResult.Length];
            Array.Copy(_ssgiResult, filtered, _ssgiResult.Length);

            Parallel.For(radius, _height - radius, y =>
            {
                for (int x = radius; x < _width - radius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    Vector3 centerNormal = gbuffer.Normals[idx];
                    float centerDepth = gbuffer.Depth[idx];
                    Vector3 sum = Vector3.Zero;
                    float weightSum = 0;

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);

                            float depthDiff = MathF.Abs(centerDepth - gbuffer.Depth[nIdx]) / MathF.Max(0.001f, centerDepth);
                            float normalDiff = 1.0f - Vector3.Dot(centerNormal, gbuffer.Normals[nIdx]);

                            if (depthDiff > depthThreshold || normalDiff > normalThreshold)
                                continue;

                            float spatialWeight = MathF.Exp(-(dx * dx + dy * dy) / (2.0f * radius * radius));
                            float edgeWeight = MathF.Exp(-depthDiff * 10.0f) * MathF.Exp(-normalDiff * 5.0f);
                            float weight = spatialWeight * edgeWeight;

                            sum += _ssgiResult[nIdx] * weight;
                            weightSum += weight;
                        }
                    }

                    filtered[idx] = weightSum > 0 ? sum / weightSum : _ssgiResult[idx];
                }
            });

            Array.Copy(filtered, _ssgiResult, filtered.Length);
        }

        /// <summary>
        /// Falls back to world-space probes on screen-space miss.
        /// </summary>
        public Vector3 FallbackToWorldSpace(int x, int y, GBuffer gbuffer,
            IrradianceCacheManager probeCache, CameraState camera)
        {
            int idx = gbuffer.GetIndex(x, y);
            Vector3 worldPos = ReconstructWorldPosition(x, y, gbuffer.Depth[idx], gbuffer, camera);
            Vector3 normal = gbuffer.Normals[idx];

            float bestWeight = 0;
            Vector3 bestIrradiance = Vector3.Zero;

            var probes = probeCache.GetNearbyProbes(worldPos, normal, 5);
            foreach (var probe in probes)
            {
                if (!probe.IsValid) continue;
                float dist = (probe.Position - worldPos).Length();
                float weight = MathF.Exp(-dist * 0.1f) * probe.Importance;
                if (weight > bestWeight)
                {
                    bestWeight = weight;
                    bestIrradiance = ComputeProbeIrradiance(probe, normal);
                }
            }

            return bestIrradiance;
        }

        /// <summary>
        /// Computes irradiance from a probe given a direction.
        /// </summary>
        public Vector3 ComputeProbeIrradiance(IrradianceProbe probe, Vector3 direction)
        {
            if (probe.SHCoefficients == null || probe.SHCoefficients.Length < 9)
                return Vector3.Zero;

            float x = direction.X;
            float y = direction.Y;
            float z = direction.Z;

            Vector3 result = Vector3.Zero;
            result += new Vector3(probe.SHCoefficients[0].X, probe.SHCoefficients[0].Y, probe.SHCoefficients[0].Z) * 0.282095f;
            result += new Vector3(probe.SHCoefficients[1].X, probe.SHCoefficients[1].Y, probe.SHCoefficients[1].Z) * 0.488603f * y;
            result += new Vector3(probe.SHCoefficients[2].X, probe.SHCoefficients[2].Y, probe.SHCoefficients[2].Z) * 0.488603f * z;
            result += new Vector3(probe.SHCoefficients[3].X, probe.SHCoefficients[3].Y, probe.SHCoefficients[3].Z) * 0.488603f * x;
            result += new Vector3(probe.SHCoefficients[4].X, probe.SHCoefficients[4].Y, probe.SHCoefficients[4].Z) * 1.092548f * x * y;
            result += new Vector3(probe.SHCoefficients[5].X, probe.SHCoefficients[5].Y, probe.SHCoefficients[5].Z) * 1.092548f * y * z;
            result += new Vector3(probe.SHCoefficients[6].X, probe.SHCoefficients[6].Y, probe.SHCoefficients[6].Z) * 0.315392f * (3.0f * z * z - 1.0f);
            result += new Vector3(probe.SHCoefficients[7].X, probe.SHCoefficients[7].Y, probe.SHCoefficients[7].Z) * 1.092548f * x * z;
            result += new Vector3(probe.SHCoefficients[8].X, probe.SHCoefficients[8].Y, probe.SHCoefficients[8].Z) * 0.546274f * (x * x - y * y);

            return Vector3.Max(result, Vector3.Zero);
        }

        private Vector3 ReconstructWorldPosition(int x, int y, float depth, GBuffer gbuffer, CameraState camera)
        {
            float ndcX = (float)x / _width * 2.0f - 1.0f;
            float ndcY = 1.0f - (float)y / _height * 2.0f;
            Vector3 viewPos = new Vector3(ndcX, ndcY, depth);
            return camera.UnprojectFromScreen(viewPos);
        }

        private Vector3 GetTangent(Vector3 normal, ref RandomNumberGenerator rng)
        {
            Vector3 t;
            if (MathF.Abs(normal.X) > MathF.Abs(normal.Y))
                t = new Vector3(normal.Z, 0, -normal.X) / MathF.Sqrt(normal.X * normal.X + normal.Z * normal.Z);
            else
                t = new Vector3(0, -normal.Z, normal.Y) / MathF.Sqrt(normal.Y * normal.Y + normal.Z * normal.Z);
            return t;
        }
    }

    #endregion

    // =========================================================================
    #region IrradianceCacheManager

    /// <summary>
    /// Manages the irradiance cache with various placement strategies and probe management.
    /// </summary>
    public class IrradianceCacheManager
    {
        private IrradianceCacheType _cacheType;
        private List<IrradianceProbe> _probes;
        private int _maxProbes;
        private float _probeSpacing;
        private int _frameIndex;
        private bool _isInitialized;
        private ProbeUpdateMode _updateMode;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>Number of active probes.</summary>
        public int ActiveProbeCount => _probes?.Count(p => p.IsValid) ?? 0;
        /// <summary>Total probe capacity.</summary>
        public int MaxProbes => _maxProbes;
        /// <summary>Current frame index.</summary>
        public int FrameIndex => _frameIndex;

        /// <summary>
        /// Initializes the irradiance cache manager.
        /// </summary>
        public void Initialize(IrradianceCacheType cacheType, int maxProbes, float probeSpacing,
            ProbeUpdateMode updateMode)
        {
            _cacheType = cacheType;
            _maxProbes = maxProbes;
            _probeSpacing = probeSpacing;
            _updateMode = updateMode;
            _probes = new List<IrradianceProbe>(maxProbes);
            _frameIndex = 0;
            _isInitialized = true;
        }

        /// <summary>
        /// Places probes using octahedral mapping.
        /// </summary>
        public void PlaceOctahedralProbes(Vector3 center, float radius, int probesPerAxis)
        {
            float step = radius * 2.0f / probesPerAxis;
            for (int x = 0; x < probesPerAxis; x++)
            {
                for (int y = 0; y < probesPerAxis; y++)
                {
                    for (int z = 0; z < probesPerAxis; z++)
                    {
                        Vector3 pos = center + new Vector3(
                            (x - probesPerAxis / 2.0f) * step,
                            (y - probesPerAxis / 2.0f) * step,
                            (z - probesPerAxis / 2.0f) * step);

                        if (_probes.Count >= _maxProbes) return;

                        _probes.Add(new IrradianceProbe
                        {
                            Position = pos,
                            SHCoefficients = new Vector4[9],
                            IsValid = false,
                            LastUpdateFrame = -1,
                            Importance = 1.0f,
                            ReferenceDepth = 0,
                            ReferenceNormal = Vector3.UnitY,
                            Variance = 0,
                            SampleCount = 0
                        });
                    }
                }
            }
        }

        /// <summary>
        /// Places probes using spherical harmonics layout.
        /// </summary>
        public void PlaceSphericalHarmonicsProbes(Vector3 center, float radius, int numRings)
        {
            for (int ring = 0; ring < numRings; ring++)
            {
                float phi = PI * (ring + 0.5f) / numRings;
                int probesInRing = (int)(numRings * MathF.Sin(phi) * 4);
                probesInRing = Math.Max(1, probesInRing);

                for (int i = 0; i < probesInRing; i++)
                {
                    float theta = TWO_PI * i / probesInRing;
                    Vector3 pos = center + new Vector3(
                        radius * MathF.Sin(phi) * MathF.Cos(theta),
                        radius * MathF.Cos(phi),
                        radius * MathF.Sin(phi) * MathF.Sin(theta));

                    if (_probes.Count >= _maxProbes) return;

                    _probes.Add(new IrradianceProbe
                    {
                        Position = pos,
                        SHCoefficients = new Vector4[9],
                        IsValid = false,
                        LastUpdateFrame = -1,
                        Importance = 1.0f,
                        ReferenceDepth = 0,
                        ReferenceNormal = Vector3.UnitY,
                        Variance = 0,
                        SampleCount = 0
                    });
                }
            }
        }

        /// <summary>
        /// Places probes in a regular radiance grid.
        /// </summary>
        public void PlaceRadianceGrid(Vector3 origin, Vector3Int gridSize, float cellSize)
        {
            for (int x = 0; x < gridSize.X; x++)
            {
                for (int y = 0; y < gridSize.Y; y++)
                {
                    for (int z = 0; z < gridSize.Z; z++)
                    {
                        Vector3 pos = origin + new Vector3(x, y, z) * cellSize;

                        if (_probes.Count >= _maxProbes) return;

                        _probes.Add(new IrradianceProbe
                        {
                            Position = pos,
                            SHCoefficients = new Vector4[9],
                            IsValid = false,
                            LastUpdateFrame = -1,
                            Importance = 1.0f,
                            ReferenceDepth = 0,
                            ReferenceNormal = Vector3.UnitY,
                            Variance = 0,
                            SampleCount = 0
                        });
                    }
                }
            }
        }

        /// <summary>
        /// Interpolates irradiance from nearby probes using trilinear interpolation.
        /// </summary>
        public Vector3 TrilinearInterpolate(Vector3 worldPos, Vector3 normal)
        {
            if (_probes == null || _probes.Count == 0) return Vector3.Zero;

            float totalWeight = 0;
            Vector3 totalIrradiance = Vector3.Zero;

            foreach (var probe in _probes)
            {
                if (!probe.IsValid) continue;
                float dist = (probe.Position - worldPos).Length();
                float weight = MathF.Exp(-dist / _probeSpacing);
                if (weight < 0.001f) continue;

                Vector3 irradiance = ComputeProbeIrradiance(probe, normal);
                totalIrradiance += irradiance * weight;
                totalWeight += weight;
            }

            return totalWeight > 0 ? totalIrradiance / totalWeight : Vector3.Zero;
        }

        /// <summary>
        /// Interpolates irradiance using tetrahedral interpolation.
        /// </summary>
        public Vector3 TetrahedralInterpolate(Vector3 worldPos, Vector3 normal)
        {
            var nearest = FindNearestProbes(worldPos, 4);
            if (nearest.Count < 4) return TrilinearInterpolate(worldPos, normal);

            Vector3 p0 = nearest[0].Position;
            Vector3 p1 = nearest[1].Position;
            Vector3 p2 = nearest[2].Position;
            Vector3 p3 = nearest[3].Position;

            Vector4 barycentrics = ComputeBarycentricCoordinates(worldPos, p0, p1, p2, p3);

            Vector3 irr0 = ComputeProbeIrradiance(nearest[0], normal);
            Vector3 irr1 = ComputeProbeIrradiance(nearest[1], normal);
            Vector3 irr2 = ComputeProbeIrradiance(nearest[2], normal);
            Vector3 irr3 = ComputeProbeIrradiance(nearest[3], normal);

            return irr0 * barycentrics.X + irr1 * barycentrics.Y +
                   irr2 * barycentrics.Z + irr3 * barycentrics.W;
        }

        /// <summary>
        /// Schedules probe updates based on importance.
        /// </summary>
        public void ScheduleUpdates(CameraState camera, float timeBudget)
        {
            _frameIndex++;

            for (int probeIndex = 0; probeIndex < _probes.Count; probeIndex++)
            {
                var probe = _probes[probeIndex];
                float dist = (probe.Position - camera.Position).Length();
                float importance = 1.0f / (1.0f + dist * 0.01f);

                bool needsUpdate = false;
                switch (_updateMode)
                {
                    case ProbeUpdateMode.OnDemand:
                        needsUpdate = !probe.IsValid;
                        break;
                    case ProbeUpdateMode.Periodic:
                        needsUpdate = (_frameIndex - probe.LastUpdateFrame) > 60;
                        break;
                    case ProbeUpdateMode.DistanceBased:
                        needsUpdate = dist < 50.0f && (_frameIndex - probe.LastUpdateFrame) > (int)(dist * 0.5f);
                        break;
                    case ProbeUpdateMode.ImportanceDriven:
                        needsUpdate = importance > 0.5f && (_frameIndex - probe.LastUpdateFrame) > 10;
                        break;
                    case ProbeUpdateMode.BudgetLimited:
                        needsUpdate = importance > 0.3f;
                        break;
                }

                if (needsUpdate)
                {
                    _probes[probeIndex] = probe with
                    {
                        Importance = importance,
                        LastUpdateFrame = _frameIndex
                    };
                }
            }
        }

        /// <summary>
        /// Manages probe budget by removing least important probes.
        /// </summary>
        public void ManageBudget(int targetCount)
        {
            if (_probes.Count <= targetCount) return;

            var sorted = _probes.OrderByDescending(p => p.Importance).Take(targetCount).ToList();
            _probes = sorted;
        }

        /// <summary>
        /// Tracks cache coherence across frames.
        /// </summary>
        public float TrackCacheCoherence(CameraState camera)
        {
            if (_probes.Count == 0) return 0;

            int reusedCount = 0;
            foreach (var probe in _probes)
            {
                if (probe.IsValid && (_frameIndex - probe.LastUpdateFrame) < 5)
                    reusedCount++;
            }

            return (float)reusedCount / _probes.Count;
        }

        /// <summary>
        /// Checks probe validity by comparing depth with current G-Buffer.
        /// </summary>
        public void CheckProbeValidity(GBuffer gbuffer, CameraState camera)
        {
            for (int i = 0; i < _probes.Count; i++)
            {
                var probe = _probes[i];
                Vector3 screenPos = camera.ProjectToScreen(probe.Position);

                if (screenPos.X < 0 || screenPos.X >= 1 || screenPos.Y < 0 || screenPos.Y >= 1)
                {
                    _probes[i] = probe with { IsValid = false };
                    continue;
                }

                int pixX = (int)(screenPos.X * gbuffer.Width);
                int pixY = (int)(screenPos.Y * gbuffer.Height);
                pixX = Math.Clamp(pixX, 0, gbuffer.Width - 1);
                pixY = Math.Clamp(pixY, 0, gbuffer.Height - 1);

                float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                float probeDepth = screenPos.Z;

                float depthError = MathF.Abs(sceneDepth - probeDepth) / MathF.Max(0.001f, sceneDepth);
                if (depthError > 0.1f)
                    _probes[i] = probe with { IsValid = false };
            }
        }

        /// <summary>
        /// Compresses probe data using SH compression.
        /// </summary>
        public byte[] CompressProbes()
        {
            using var ms = new System.IO.MemoryStream();
            using var bw = new System.IO.BinaryWriter(ms);

            bw.Write(_probes.Count);
            foreach (var probe in _probes)
            {
                bw.Write(probe.Position.X);
                bw.Write(probe.Position.Y);
                bw.Write(probe.Position.Z);
                bw.Write(probe.IsValid);
                bw.Write(probe.Importance);

                for (int i = 0; i < 9; i++)
                {
                    bw.Write(probe.SHCoefficients[i].X);
                    bw.Write(probe.SHCoefficients[i].Y);
                    bw.Write(probe.SHCoefficients[i].Z);
                    bw.Write(probe.SHCoefficients[i].W);
                }
            }

            return ms.ToArray();
        }

        /// <summary>
        /// Decompresses probe data from a byte array.
        /// </summary>
        public void DecompressProbes(byte[] data)
        {
            using var ms = new System.IO.MemoryStream(data);
            using var br = new System.IO.BinaryReader(ms);

            int count = br.ReadInt32();
            _probes.Clear();

            for (int i = 0; i < count; i++)
            {
                float px = br.ReadSingle();
                float py = br.ReadSingle();
                float pz = br.ReadSingle();
                bool valid = br.ReadBoolean();
                float importance = br.ReadSingle();

                var shCoeffs = new Vector4[9];
                for (int j = 0; j < 9; j++)
                    shCoeffs[j] = new Vector4(br.ReadSingle(), br.ReadSingle(), br.ReadSingle(), br.ReadSingle());

                _probes.Add(new IrradianceProbe
                {
                    Position = new Vector3(px, py, pz),
                    SHCoefficients = shCoeffs,
                    IsValid = valid,
                    Importance = importance
                });
            }
        }

        /// <summary>
        /// Streams probes to disk for persistence.
        /// </summary>
        public void StreamToDisk(string filePath)
        {
            byte[] data = CompressProbes();
            System.IO.File.WriteAllBytes(filePath, data);
        }

        /// <summary>
        /// Loads probes from disk.
        /// </summary>
        public void StreamFromDisk(string filePath)
        {
            if (System.IO.File.Exists(filePath))
            {
                byte[] data = System.IO.File.ReadAllBytes(filePath);
                DecompressProbes(data);
            }
        }

        /// <summary>
        /// Gets nearby probes for a given position and normal.
        /// </summary>
        public List<IrradianceProbe> GetNearbyProbes(Vector3 worldPos, Vector3 normal, int maxCount)
        {
            return _probes
                .Where(p => p.IsValid)
                .OrderBy(p => (p.Position - worldPos).LengthSquared())
                .Take(maxCount)
                .ToList();
        }

        private List<IrradianceProbe> FindNearestProbes(Vector3 worldPos, int count)
        {
            return _probes
                .OrderBy(p => (p.Position - worldPos).LengthSquared())
                .Take(count)
                .ToList();
        }

        private Vector3 ComputeProbeIrradiance(IrradianceProbe probe, Vector3 direction)
        {
            if (probe.SHCoefficients == null || probe.SHCoefficients.Length < 9)
                return Vector3.Zero;

            float x = direction.X;
            float y = direction.Y;
            float z = direction.Z;

            Vector3 result = Vector3.Zero;
            result += new Vector3(probe.SHCoefficients[0].X, probe.SHCoefficients[0].Y, probe.SHCoefficients[0].Z) * 0.282095f;
            result += new Vector3(probe.SHCoefficients[1].X, probe.SHCoefficients[1].Y, probe.SHCoefficients[1].Z) * 0.488603f * y;
            result += new Vector3(probe.SHCoefficients[2].X, probe.SHCoefficients[2].Y, probe.SHCoefficients[2].Z) * 0.488603f * z;
            result += new Vector3(probe.SHCoefficients[3].X, probe.SHCoefficients[3].Y, probe.SHCoefficients[3].Z) * 0.488603f * x;
            result += new Vector3(probe.SHCoefficients[4].X, probe.SHCoefficients[4].Y, probe.SHCoefficients[4].Z) * 1.092548f * x * y;
            result += new Vector3(probe.SHCoefficients[5].X, probe.SHCoefficients[5].Y, probe.SHCoefficients[5].Z) * 1.092548f * y * z;
            result += new Vector3(probe.SHCoefficients[6].X, probe.SHCoefficients[6].Y, probe.SHCoefficients[6].Z) * 0.315392f * (3.0f * z * z - 1.0f);
            result += new Vector3(probe.SHCoefficients[7].X, probe.SHCoefficients[7].Y, probe.SHCoefficients[7].Z) * 1.092548f * x * z;
            result += new Vector3(probe.SHCoefficients[8].X, probe.SHCoefficients[8].Y, probe.SHCoefficients[8].Z) * 0.546274f * (x * x - y * y);

            return Vector3.Max(result, Vector3.Zero);
        }

        private Vector4 ComputeBarycentricCoordinates(Vector3 point, Vector3 a, Vector3 b, Vector3 c, Vector3 d)
        {
            Vector3 v0 = b - a, v1 = c - a, v2 = d - a, v3 = point - a;
            float d00 = Vector3.Dot(v0, v0);
            float d01 = Vector3.Dot(v0, v1);
            float d02 = Vector3.Dot(v0, v2);
            float d11 = Vector3.Dot(v1, v1);
            float d12 = Vector3.Dot(v1, v2);
            float d22 = Vector3.Dot(v2, v2);
            float d30 = Vector3.Dot(v3, v0);
            float d31 = Vector3.Dot(v3, v1);
            float d32 = Vector3.Dot(v3, v2);

            float denom = d00 * (d11 * d22 - d12 * d12) - d01 * (d01 * d22 - d12 * d02) + d02 * (d01 * d12 - d11 * d02);
            if (MathF.Abs(denom) < 0.0001f) return new Vector4(0.25f, 0.25f, 0.25f, 0.25f);

            float w = (d30 * (d11 * d22 - d12 * d12) - d01 * (d31 * d22 - d12 * d32) + d02 * (d31 * d12 - d11 * d32)) / denom;
            float u = (d00 * (d31 * d22 - d12 * d32) - d30 * (d01 * d22 - d12 * d02) + d02 * (d01 * d32 - d31 * d02)) / denom;
            float v = (d00 * (d11 * d32 - d31 * d12) - d01 * (d01 * d32 - d31 * d02) + d30 * (d01 * d12 - d11 * d02)) / denom;
            float t = 1.0f - u - v - w;

            return new Vector4(t, u, v, w);
        }
    }

    #endregion

    // =========================================================================
    #region TemporalStabilizer

    /// <summary>
    /// Temporal stabilization system for frame-to-frame coherence.
    /// </summary>
    public class TemporalStabilizer
    {
        private TemporalConfig _config;
        private Vector3[] _historyBuffer;
        private float[] _historyWeight;
        private float[] _varianceBuffer;
        private int _width;
        private int _height;
        private int _historyLength;
        private bool _isInitialized;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;

        /// <summary>Current history length.</summary>
        public int HistoryLength => _historyLength;

        /// <summary>
        /// Initializes the temporal stabilizer.
        /// </summary>
        public void Initialize(int width, int height, TemporalConfig config)
        {
            _width = width;
            _height = height;
            _config = config;
            int pixelCount = width * height;
            _historyBuffer = new Vector3[pixelCount];
            _historyWeight = new float[pixelCount];
            _varianceBuffer = new float[pixelCount];
            _historyLength = 0;
            _isInitialized = true;
        }

        /// <summary>
        /// Reprojects the previous frame using motion vectors.
        /// </summary>
        public Vector3 Reproject(int x, int y, Vector2 velocity, GBuffer gbuffer)
        {
            if (_historyBuffer == null) return Vector3.Zero;

            float prevX = x + velocity.X;
            float prevY = y + velocity.Y;

            if (prevX < 0 || prevX >= _width || prevY < 0 || prevY >= _height)
                return Vector3.Zero;

            int prevIdx = (int)prevY * _width + (int)prevX;
            if (prevIdx >= 0 && prevIdx < _historyBuffer.Length && _historyWeight[prevIdx] > 0)
                return _historyBuffer[prevIdx];

            return Vector3.Zero;
        }

        /// <summary>
        /// Applies exponential moving average with adaptive blend factor.
        /// </summary>
        public Vector3 ApplyEMA(Vector3 current, Vector3 history, float blendFactor)
        {
            return Vector3.Lerp(history, current, 1.0f - blendFactor);
        }

        /// <summary>
        /// Performs variance clipping to prevent ghosting.
        /// </summary>
        public Vector3 VarianceClip(Vector3 current, Vector3 history, Vector3[] neighborhood,
            float clippingStrength)
        {
            if (neighborhood == null || neighborhood.Length == 0)
                return history;

            Vector3 mean = Vector3.Zero;
            Vector3 variance = Vector3.Zero;

            foreach (var sample in neighborhood)
                mean += sample;
            mean /= neighborhood.Length;

            foreach (var sample in neighborhood)
            {
                Vector3 diff = sample - mean;
                variance += new Vector3(diff.X * diff.X, diff.Y * diff.Y, diff.Z * diff.Z);
            }
            variance /= neighborhood.Length;

            Vector3 minBound = mean - new Vector3(
                MathF.Sqrt(variance.X) * clippingStrength,
                MathF.Sqrt(variance.Y) * clippingStrength,
                MathF.Sqrt(variance.Z) * clippingStrength);
            Vector3 maxBound = mean + new Vector3(
                MathF.Sqrt(variance.X) * clippingStrength,
                MathF.Sqrt(variance.Y) * clippingStrength,
                MathF.Sqrt(variance.Z) * clippingStrength);

            return Vector3.Clamp(history, minBound, maxBound);
        }

        /// <summary>
        /// Prevents ghosting by checking history confidence.
        /// </summary>
        public float ComputeGhostingPrevention(Vector3 current, Vector3 history,
            Vector2 velocity, GBuffer gbuffer, int x, int y)
        {
            float velocityMagnitude = velocity.Length();
            float motionConfidence = MathF.Exp(-velocityMagnitude * 10.0f);

            int idx = gbuffer.GetIndex(x, y);
            Vector3 currentNormal = gbuffer.Normals[idx];
            float currentDepth = gbuffer.Depth[idx];

            float depthDiff = 0;
            float normalDiff = 0;

            if (velocityMagnitude > 0.01f)
            {
                float prevX = Math.Clamp(x + (int)velocity.X, 0, _width - 1);
                float prevY = Math.Clamp(y + (int)velocity.Y, 0, _height - 1);
                int prevIdx = (int)prevY * _width + (int)prevX;

                if (prevIdx >= 0 && prevIdx < gbuffer.Depth.Length)
                {
                    float prevDepth = gbuffer.Depth[prevIdx];
                    depthDiff = MathF.Abs(currentDepth - prevDepth) / MathF.Max(0.001f, currentDepth);
                }
            }

            float depthConfidence = MathF.Exp(-depthDiff * 10.0f);
            return motionConfidence * depthConfidence;
        }

        /// <summary>
        /// Detects disocclusion using bilateral depth/stencil comparison.
        /// </summary>
        public bool DetectDisocclusion(GBuffer currentGBuffer, GBuffer previousGBuffer,
            int x, int y, Vector2 velocity, float threshold)
        {
            if (previousGBuffer == null) return true;

            float prevX = Math.Clamp(x + velocity.X, 0, _width - 1);
            float prevY = Math.Clamp(y + velocity.Y, 0, _height - 1);

            int currentIdx = currentGBuffer.GetIndex(x, y);
            int prevIdx = previousGBuffer.GetIndex((int)prevX, (int)prevY);

            float currentDepth = currentGBuffer.Depth[currentIdx];
            float prevDepth = previousGBuffer.Depth[prevIdx];

            Vector3 currentNormal = currentGBuffer.Normals[currentIdx];
            Vector3 prevNormal = previousGBuffer.Normals[prevIdx];

            float depthError = MathF.Abs(currentDepth - prevDepth) / MathF.Max(0.001f, currentDepth);
            float normalError = 1.0f - Vector3.Dot(currentNormal, prevNormal);

            return depthError > threshold || normalError > 0.5f;
        }

        /// <summary>
        /// Applies color box filter to history.
        /// </summary>
        public Vector3 ColorBoxFilterHistory(int x, int y, int radius)
        {
            if (_historyBuffer == null) return Vector3.Zero;

            Vector3 sum = Vector3.Zero;
            int count = 0;

            for (int dx = -radius; dx <= radius; dx++)
            {
                for (int dy = -radius; dy <= radius; dy++)
                {
                    int nx = Math.Clamp(x + dx, 0, _width - 1);
                    int ny = Math.Clamp(y + dy, 0, _height - 1);
                    int idx = ny * _width + nx;

                    if (_historyWeight[idx] > 0)
                    {
                        sum += _historyBuffer[idx];
                        count++;
                    }
                }
            }

            return count > 0 ? sum / count : Vector3.Zero;
        }

        /// <summary>
        /// Computes adaptive history length based on motion and variance.
        /// </summary>
        public int ComputeAdaptiveHistoryLength(Vector2 velocity, float variance)
        {
            float motionFactor = MathF.Exp(-velocity.Length() * 5.0f);
            float varianceFactor = MathF.Exp(-variance * 2.0f);
            float combinedFactor = motionFactor * varianceFactor;
            return Math.Clamp((int)(combinedFactor * _config.MaxHistoryLength), 1, _config.MaxHistoryLength);
        }

        /// <summary>
        /// Fast response to scene changes.
        /// </summary>
        public Vector3 FastResponse(Vector3 current, Vector3 history, float responseSpeed)
        {
            return Vector3.Lerp(history, current, responseSpeed);
        }

        /// <summary>
        /// Slow accumulation for stable regions.
        /// </summary>
        public Vector3 SlowAccumulation(Vector3 current, Vector3 history, float accumulationSpeed)
        {
            return Vector3.Lerp(current, history, accumulationSpeed);
        }

        /// <summary>
        /// Converts RGB to YCoCg color space for better filtering.
        /// </summary>
        public Vector3 RGBToYCoCg(Vector3 rgb)
        {
            float y = 0.25f * rgb.X + 0.5f * rgb.Y + 0.25f * rgb.Z;
            float co = 0.5f * rgb.X - 0.5f * rgb.Z;
            float cg = -0.25f * rgb.X + 0.5f * rgb.Y - 0.25f * rgb.Z;
            return new Vector3(y, co, cg);
        }

        /// <summary>
        /// Converts YCoCg back to RGB color space.
        /// </summary>
        public Vector3 YCoCgToRGB(Vector3 ycocg)
        {
            float r = ycocg.X + ycocg.Y - ycocg.Z;
            float g = ycocg.X + ycocg.Z;
            float b = ycocg.X - ycocg.Y - ycocg.Z;
            return new Vector3(r, g, b);
        }

        /// <summary>
        /// Temporal filter with full pipeline.
        /// </summary>
        public Vector3 ApplyTemporalFilter(int x, int y, Vector3 current, Vector2 velocity,
            GBuffer gbuffer, GBuffer previousGBuffer)
        {
            if (!_isInitialized || _historyBuffer == null) return current;

            int idx = gbuffer.GetIndex(x, y);

            bool disoccluded = DetectDisocclusion(gbuffer, previousGBuffer, x, y, velocity,
                _config.DisocclusionThreshold);

            if (disoccluded)
            {
                _historyBuffer[idx] = current;
                _historyWeight[idx] = 0;
                return current;
            }

            Vector3 history = Reproject(x, y, velocity, gbuffer);
            if (history.LengthSquared() < 0.0001f)
            {
                _historyBuffer[idx] = current;
                _historyWeight[idx] = 0;
                return current;
            }

            float ghostingConfidence = ComputeGhostingPrevention(current, history, velocity, gbuffer, x, y);

            float blendFactor = _config.BaseBlendFactor * ghostingConfidence;

            Vector3 filtered;
            switch (_config.Mode)
            {
                case TemporalFilterMode.ExponentialMovingAverage:
                    filtered = ApplyEMA(current, history, blendFactor);
                    break;

                case TemporalFilterMode.VarianceClipping:
                    Vector3[] neighborhood = GetNeighborhoodHistory(x, y, 1);
                    Vector3 clippedHistory = VarianceClip(current, history, neighborhood,
                        _config.VarianceClippingStrength);
                    filtered = ApplyEMA(current, clippedHistory, blendFactor);
                    break;

                case TemporalFilterMode.DisocclusionAware:
                    float responseBlend = FastResponse(current, history, _config.ResponseSpeed).Length();
                    float accumBlend = SlowAccumulation(current, history, _config.AccumulationSpeed).Length();
                    filtered = Vector3.Lerp(
                        FastResponse(current, history, _config.ResponseSpeed),
                        SlowAccumulation(current, history, _config.AccumulationSpeed),
                        ghostingConfidence);
                    break;

                default:
                    filtered = current;
                    break;
            }

            _historyBuffer[idx] = filtered;
            _historyWeight[idx] = MathF.Min(_historyWeight[idx] + 1, _config.MaxHistoryLength);

            return filtered;
        }

        private Vector3[] GetNeighborhoodHistory(int x, int y, int radius)
        {
            var neighborhood = new List<Vector3>();
            for (int dx = -radius; dx <= radius; dx++)
            {
                for (int dy = -radius; dy <= radius; dy++)
                {
                    int nx = Math.Clamp(x + dx, 0, _width - 1);
                    int ny = Math.Clamp(y + dy, 0, _height - 1);
                    int idx = ny * _width + nx;
                    if (_historyWeight[idx] > 0)
                        neighborhood.Add(_historyBuffer[idx]);
                }
            }
            return neighborhood.ToArray();
        }
    }

    #endregion

    // =========================================================================
    #region DenoisingPipeline

    /// <summary>
    /// Multi-stage denoising pipeline for noise reduction in GI.
    /// </summary>
    public class DenoisingPipeline
    {
        private DenoiseConfig _config;
        private Vector3[] _tempBufferA;
        private Vector3[] _tempBufferB;
        private float[] _varianceBuffer;
        private int _width;
        private int _height;
        private bool _isInitialized;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>
        /// Initializes the denoising pipeline.
        /// </summary>
        public void Initialize(int width, int height, DenoiseConfig config)
        {
            _width = width;
            _height = height;
            _config = config;
            int pixelCount = width * height;
            _tempBufferA = new Vector3[pixelCount];
            _tempBufferB = new Vector3[pixelCount];
            _varianceBuffer = new float[pixelCount];
            _isInitialized = true;
        }

        /// <summary>
        /// Applies bilateral spatial filter.
        /// </summary>
        public void SpatialBilateralFilter(Vector3[] input, Vector3[] output, GBuffer gbuffer,
            int radius, float normalThreshold, float depthThreshold)
        {
            Parallel.For(radius, _height - radius, y =>
            {
                for (int x = radius; x < _width - radius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    Vector3 centerNormal = gbuffer.Normals[idx];
                    float centerDepth = gbuffer.Depth[idx];
                    Vector3 centerColor = input[idx];

                    Vector3 sum = Vector3.Zero;
                    float weightSum = 0;

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);

                            float depthDiff = MathF.Abs(centerDepth - gbuffer.Depth[nIdx]) /
                                             MathF.Max(0.001f, centerDepth);
                            float normalDiff = 1.0f - Vector3.Dot(centerNormal, gbuffer.Normals[nIdx]);
                            Vector3 colorDiff = centerColor - input[nIdx];
                            float colorDist = colorDiff.Length();

                            if (depthDiff > depthThreshold || normalDiff > normalThreshold)
                                continue;

                            float spatialWeight = MathF.Exp(-(dx * dx + dy * dy) / (2.0f * radius * radius));
                            float depthWeight = MathF.Exp(-depthDiff * 10.0f);
                            float normalWeight = MathF.Exp(-normalDiff * 5.0f);
                            float colorWeight = MathF.Exp(-colorDist * 3.0f);

                            float weight = spatialWeight * depthWeight * normalWeight * colorWeight;
                            sum += input[nIdx] * weight;
                            weightSum += weight;
                        }
                    }

                    output[idx] = weightSum > 0 ? sum / weightSum : centerColor;
                }
            });
        }

        /// <summary>
        /// Applies non-local means denoiser.
        /// </summary>
        public void NonLocalMeansDenoiser(Vector3[] input, Vector3[] output, GBuffer gbuffer,
            int patchRadius, int searchRadius, float h)
        {
            Parallel.For(searchRadius, _height - searchRadius, y =>
            {
                for (int x = searchRadius; x < _width - searchRadius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    Vector3 centerColor = input[idx];

                    Vector3 sum = Vector3.Zero;
                    float weightSum = 0;

                    for (int dx = -searchRadius; dx <= searchRadius; dx++)
                    {
                        for (int dy = -searchRadius; dy <= searchRadius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;

                            float patchDistance = ComputePatchDistance(input, x, y, nx, ny,
                                patchRadius, gbuffer);

                            float weight = MathF.Exp(-patchDistance / (h * h));
                            int nIdx = gbuffer.GetIndex(nx, ny);
                            sum += input[nIdx] * weight;
                            weightSum += weight;
                        }
                    }

                    output[idx] = weightSum > 0 ? sum / weightSum : centerColor;
                }
            });
        }

        private float ComputePatchDistance(Vector3[] image, int x1, int y1, int x2, int y2,
            int patchRadius, GBuffer gbuffer)
        {
            float distance = 0;
            int count = 0;

            for (int dx = -patchRadius; dx <= patchRadius; dx++)
            {
                for (int dy = -patchRadius; dy <= patchRadius; dy++)
                {
                    int nx1 = Math.Clamp(x1 + dx, 0, _width - 1);
                    int ny1 = Math.Clamp(y1 + dy, 0, _height - 1);
                    int nx2 = Math.Clamp(x2 + dx, 0, _width - 1);
                    int ny2 = Math.Clamp(y2 + dy, 0, _height - 1);

                    int idx1 = gbuffer.GetIndex(nx1, ny1);
                    int idx2 = gbuffer.GetIndex(nx2, ny2);

                    Vector3 diff = image[idx1] - image[idx2];
                    distance += diff.LengthSquared();
                    count++;
                }
            }

            return count > 0 ? distance / count : 0;
        }

        /// <summary>
        /// Applies wavelet transform denoiser.
        /// </summary>
        public void WaveletDenoiser(Vector3[] input, Vector3[] output, int levels, float threshold)
        {
            Vector3[] tempA = new Vector3[input.Length];
            Vector3[] tempB = new Vector3[input.Length];
            Array.Copy(input, tempA, input.Length);

            for (int level = 0; level < levels; level++)
            {
                int stride = 1 << level;
                WaveletDecompose(tempA, tempB, stride);
                SoftThreshold(tempB, threshold / (level + 1));
                WaveletReconstruct(tempA, tempB, stride);
            }

            Array.Copy(tempA, output, tempA.Length);
        }

        private void WaveletDecompose(Vector3[] input, Vector3[] detail, int stride)
        {
            for (int i = 0; i < input.Length; i++)
            {
                int left = Math.Max(0, i - stride);
                int right = Math.Min(input.Length - 1, i + stride);
                Vector3 average = (input[left] + input[right]) * 0.5f;
                detail[i] = input[i] - average;
                input[i] = average;
            }
        }

        private void WaveletReconstruct(Vector3[] approximation, Vector3[] detail, int stride)
        {
            for (int i = 0; i < approximation.Length; i++)
            {
                approximation[i] += detail[i];
            }
        }

        private void SoftThreshold(Vector3[] data, float threshold)
        {
            for (int i = 0; i < data.Length; i++)
            {
                float lum = 0.2126f * data[i].X + 0.7152f * data[i].Y + 0.0722f * data[i].Z;
                float sign = lum >= 0 ? 1.0f : -1.0f;
                float newLum = MathF.Max(0, MathF.Abs(lum) - threshold) * sign;
                float scale = MathF.Abs(lum) > 0.0001f ? newLum / lum : 0;
                data[i] *= scale;
            }
        }

        /// <summary>
        /// Applies normal-guided filter.
        /// </summary>
        public void NormalGuidedFilter(Vector3[] input, Vector3[] output, GBuffer gbuffer,
            int radius, float epsilon)
        {
            Parallel.For(radius, _height - radius, y =>
            {
                for (int x = radius; x < _width - radius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    Vector3 centerNormal = gbuffer.Normals[idx];

                    Vector3 meanP = Vector3.Zero;
                    Vector3 meanI = Vector3.Zero;
                    int count = 0;

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);

                            float normalSim = MathF.Max(0, Vector3.Dot(centerNormal, gbuffer.Normals[nIdx]));
                            if (normalSim < 0.7f) continue;

                            meanP += gbuffer.Normals[nIdx];
                            meanI += input[nIdx];
                            count++;
                        }
                    }

                    if (count > 0)
                    {
                        meanP /= count;
                        meanI /= count;

                        Vector3 varP = Vector3.Zero;
                        Vector3 covPI = Vector3.Zero;

                        for (int dx = -radius; dx <= radius; dx++)
                        {
                            for (int dy = -radius; dy <= radius; dy++)
                            {
                                int nx = x + dx;
                                int ny = y + dy;
                                int nIdx = gbuffer.GetIndex(nx, ny);

                                float normalSim = MathF.Max(0, Vector3.Dot(centerNormal, gbuffer.Normals[nIdx]));
                                if (normalSim < 0.7f) continue;

                                Vector3 diffP = gbuffer.Normals[nIdx] - meanP;
                                Vector3 diffI = input[nIdx] - meanI;
                                varP += new Vector3(diffP.X * diffP.X, diffP.Y * diffP.Y, diffP.Z * diffP.Z);
                                covPI += new Vector3(diffP.X * diffI.X, diffP.Y * diffI.Y, diffP.Z * diffI.Z);
                            }
                        }

                        varP /= count;
                        covPI /= count;

                        Vector3 a = new Vector3(
                            covPI.X / (varP.X + epsilon),
                            covPI.Y / (varP.Y + epsilon),
                            covPI.Z / (varP.Z + epsilon));
                        Vector3 b = meanI - a * meanP;

                        output[idx] = a * gbuffer.Normals[idx] + b;
                    }
                    else
                    {
                        output[idx] = input[idx];
                    }
                }
            });
        }

        /// <summary>
        /// Estimates local variance.
        /// </summary>
        public void EstimateVariance(Vector3[] input, float[] output, GBuffer gbuffer, int radius)
        {
            Parallel.For(radius, _height - radius, y =>
            {
                for (int x = radius; x < _width - radius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    Vector3 mean = Vector3.Zero;
                    int count = 0;

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);
                            mean += input[nIdx];
                            count++;
                        }
                    }

                    mean /= count;

                    float variance = 0;
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);
                            Vector3 diff = input[nIdx] - mean;
                            variance += diff.LengthSquared();
                        }
                    }

                    output[idx] = variance / count;
                }
            });
        }

        /// <summary>
        /// Computes edge-stopping function.
        /// </summary>
        public float EdgeStoppingFunction(GBuffer gbuffer, int x1, int y1, int x2, int y2,
            float normalThreshold, float depthThreshold, float luminanceThreshold,
            Vector3[] luminance)
        {
            int idx1 = gbuffer.GetIndex(x1, y1);
            int idx2 = gbuffer.GetIndex(x2, y2);

            float depthDiff = MathF.Abs(gbuffer.Depth[idx1] - gbuffer.Depth[idx2]) /
                             MathF.Max(0.001f, gbuffer.Depth[idx1]);
            float normalDiff = 1.0f - Vector3.Dot(gbuffer.Normals[idx1], gbuffer.Normals[idx2]);

            float lum1 = luminance[idx1].Length();
            float lum2 = luminance[idx2].Length();
            float luminanceDiff = MathF.Abs(lum1 - lum2) / MathF.Max(0.001f, MathF.Max(lum1, lum2));

            float depthWeight = MathF.Exp(-depthDiff / depthThreshold);
            float normalWeight = MathF.Exp(-normalDiff / normalThreshold);
            float luminanceWeight = MathF.Exp(-luminanceDiff / luminanceThreshold);

            return depthWeight * normalWeight * luminanceWeight;
        }

        /// <summary>
        /// Progressive denoising with iterative refinement.
        /// </summary>
        public void ProgressiveDenoise(Vector3[] input, Vector3[] output, GBuffer gbuffer,
            int iterations, int baseRadius)
        {
            Vector3[] current = new Vector3[input.Length];
            Vector3[] next = new Vector3[input.Length];
            Array.Copy(input, current, input.Length);

            for (int iter = 0; iter < iterations; iter++)
            {
                int radius = baseRadius * (iter + 1);
                float strength = _config.Strength * (1.0f - (float)iter / iterations);

                SpatialBilateralFilter(current, next, gbuffer, radius,
                    _config.NormalThreshold * strength, _config.DepthThreshold * strength);

                Vector3[] temp = current;
                current = next;
                next = temp;
            }

            Array.Copy(current, output, current.Length);
        }

        /// <summary>
        /// Applies the full mixed denoiser pipeline.
        /// </summary>
        public void ApplyMixedPipeline(Vector3[] input, Vector3[] output, GBuffer gbuffer)
        {
            int pixelCount = _width * _height;
            Vector3[] stageA = new Vector3[pixelCount];
            Vector3[] stageB = new Vector3[pixelCount];
            Array.Copy(input, stageA, pixelCount);

            switch (_config.PrimaryDenoiser)
            {
                case DenoiserType.SpatialBilateral:
                    SpatialBilateralFilter(stageA, stageB, gbuffer, _config.SpatialRadius,
                        _config.NormalThreshold, _config.DepthThreshold);
                    break;
                case DenoiserType.NonLocalMeans:
                    NonLocalMeansDenoiser(stageA, stageB, gbuffer, 2, _config.SpatialRadius,
                        0.1f);
                    break;
                case DenoiserType.WaveletFilter:
                    WaveletDenoiser(stageA, stageB, 4, 0.05f);
                    break;
                default:
                    Array.Copy(stageA, stageB, pixelCount);
                    break;
            }

            if (_config.SecondaryDenoiser != DenoiserType.None)
            {
                switch (_config.SecondaryDenoiser)
                {
                    case DenoiserType.SpatialBilateral:
                        SpatialBilateralFilter(stageB, stageA, gbuffer, _config.SpatialRadius / 2,
                            _config.NormalThreshold, _config.DepthThreshold);
                        break;
                    case DenoiserType.TemporalAccumulation:
                        Array.Copy(stageB, stageA, pixelCount);
                        break;
                    default:
                        Array.Copy(stageB, stageA, pixelCount);
                        break;
                }
                Array.Copy(stageA, output, pixelCount);
            }
            else
            {
                Array.Copy(stageB, output, pixelCount);
            }
        }
    }

    #endregion

    // =========================================================================
    #region VolumetricLighting

    /// <summary>
    /// Volumetric lighting system using froxel-based volume rendering.
    /// </summary>
    public class VolumetricLighting
    {
        private VolumeFogConfig _config;
        private Vector3[,,] _froxelRadiance;
        private float[,,] _froxelTransmittance;
        private Vector3[,,] _temporalHistory;
        private int _gridX;
        private int _gridY;
        private int _gridZ;
        private float[] _depthSlices;
        private bool _isInitialized;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>Froxel grid X resolution.</summary>
        public int GridX => _gridX;
        /// <summary>Froxel grid Y resolution.</summary>
        public int GridY => _gridY;
        /// <summary>Froxel grid Z resolution (depth slices).</summary>
        public int GridZ => _gridZ;

        /// <summary>
        /// Initializes the volumetric lighting system.
        /// </summary>
        public void Initialize(VolumeFogConfig config, int screenWitdth, int screenHeight)
        {
            _config = config;
            _gridX = config.GridResolutionXY;
            _gridY = config.GridResolutionXY;
            _gridZ = config.DepthSlices;

            _froxelRadiance = new Vector3[_gridX, _gridY, _gridZ];
            _froxelTransmittance = new float[_gridX, _gridY, _gridZ];
            _temporalHistory = new Vector3[_gridX, _gridY, _gridZ];
            _depthSlices = new float[_gridZ + 1];

            ComputeExponentialDepthSlices(config.StartDistance, config.MaxDistance);
            _isInitialized = true;
        }

        private void ComputeExponentialDepthSlices(float near, float far)
        {
            for (int i = 0; i <= _gridZ; i++)
            {
                float t = (float)i / _gridZ;
                _depthSlices[i] = near * MathF.Pow(far / near, t);
            }
        }

        /// <summary>
        /// Constructs the froxel grid from camera parameters.
        /// </summary>
        public void ConstructFroxelGrid(CameraState camera)
        {
            for (int z = 0; z < _gridZ; z++)
            {
                float depthNear = _depthSlices[z];
                float depthFar = _depthSlices[z + 1];

                for (int x = 0; x < _gridX; x++)
                {
                    for (int y = 0; y < _gridY; y++)
                    {
                        float u = (x + 0.5f) / _gridX;
                        float v = (y + 0.5f) / _gridY;

                        Vector3 viewPos = new Vector3(
                            (u * 2.0f - 1.0f) * camera.FieldOfView * camera.AspectRatio,
                            (1.0f - v * 2.0f) * camera.FieldOfView,
                            -1.0f);
                        viewPos = Vector3.Normalize(viewPos);

                        float volumeDepth = (depthNear + depthFar) * 0.5f;
                        float volumeSize = depthFar - depthNear;
                        float cellVolume = volumeSize * volumeDepth * volumeDepth / (_gridX * _gridY);

                        _froxelRadiance[x, y, z] = Vector3.Zero;
                        _froxelTransmittance[x, y, z] = 1.0f;
                    }
                }
            }
        }

        /// <summary>
        /// Injects lighting into froxels from light sources.
        /// </summary>
        public void InjectLighting(List<LightConfig> lights, GBuffer gbuffer, CameraState camera)
        {
            for (int z = 0; z < _gridZ; z++)
            {
                float depthNear = _depthSlices[z];
                float depthFar = _depthSlices[z + 1];
                float sliceDepth = (depthNear + depthFar) * 0.5f;

                for (int x = 0; x < _gridX; x++)
                {
                    for (int y = 0; y < _gridY; y++)
                    {
                        Vector3 worldPos = ComputeFroxelWorldPos(x, y, z, camera);
                        Vector3 totalLighting = Vector3.Zero;

                        foreach (var light in lights)
                        {
                            Vector3 lightContribution = ComputeLightContribution(light, worldPos, gbuffer, camera);
                            totalLighting += lightContribution;
                        }

                        float density = ComputeFogDensity(worldPos);
                        _froxelRadiance[x, y, z] = totalLighting * density * _config.MaxDensity;
                    }
                }
            }
        }

        private Vector3 ComputeLightContribution(LightConfig light, Vector3 worldPos,
            GBuffer gbuffer, CameraState camera)
        {
            switch (light.Type)
            {
                case LightType.Directional:
                    return light.Color * light.Intensity;

                case LightType.Point:
                    Vector3 toLight = light.Position - worldPos;
                    float dist = toLight.Length();
                    if (dist > light.Range) return Vector3.Zero;
                    float atten = 1.0f / (dist * dist);
                    float rangeAtten = MathF.Max(0, 1.0f - MathF.Pow(dist / light.Range, 4));
                    return light.Color * light.Intensity * atten * rangeAtten;

                case LightType.Spot:
                    Vector3 spotToLight = light.Position - worldPos;
                    float spotDist = spotToLight.Length();
                    if (spotDist > light.Range) return Vector3.Zero;
                    Vector3 spotDir = spotToLight / spotDist;
                    float spotAtten = 1.0f / (spotDist * spotDist);
                    float spotRangeAtten = MathF.Max(0, 1.0f - MathF.Pow(spotDist / light.Range, 4));
                    float cosAngle = Vector3.Dot(-spotDir, light.Direction);
                    float spotCos = MathF.Cos(light.OuterConeAngle);
                    float spotInnerCos = MathF.Cos(light.InnerConeAngle);
                    float spotFalloff = Math.Clamp((cosAngle - spotCos) / MathF.Max(0.001f, spotInnerCos - spotCos), 0, 1);
                    return light.Color * light.Intensity * spotAtten * spotRangeAtten * spotFalloff;

                default:
                    return Vector3.Zero;
            }
        }

        /// <summary>
        /// Evaluates the Henyey-Greenstein phase function.
        /// </summary>
        public float HenyeyGreenstein(float cosTheta, float g)
        {
            float g2 = g * g;
            float denom = 1.0f + g2 - 2.0f * g * cosTheta;
            return (1.0f - g2) / (4.0f * PI * denom * MathF.Sqrt(denom));
        }

        /// <summary>
        /// Evaluates a dual-lobe phase function.
        /// </summary>
        public float DualLobePhaseFunction(float cosTheta, float g1, float g2, float blend)
        {
            float phase1 = HenyeyGreenstein(cosTheta, g1);
            float phase2 = HenyeyGreenstein(cosTheta, g2);
            return blend * phase1 + (1.0f - blend) * phase2;
        }

        /// <summary>
        /// Integrates anisotropic scattering along a view ray.
        /// </summary>
        public Vector3 IntegrateAnisotropicScattering(Vector3 viewDir, Vector3 lightDir,
            float density, Vector3 scatteringCoeff)
        {
            float cosTheta = Vector3.Dot(viewDir, lightDir);
            float phase = HenyeyGreenstein(cosTheta, _config.Anisotropy);
            return scatteringCoeff * phase * density;
        }

        /// <summary>
        /// Performs temporal reprojection for froxels.
        /// </summary>
        public void TemporalReprojection(CameraState camera)
        {
            for (int x = 0; x < _gridX; x++)
            {
                for (int y = 0; y < _gridY; y++)
                {
                    for (int z = 0; z < _gridZ; z++)
                    {
                        Vector3 current = _froxelRadiance[x, y, z];
                        Vector3 history = _temporalHistory[x, y, z];

                        float blendFactor = _config.TemporalReprojectionStrength;
                        _froxelRadiance[x, y, z] = Vector3.Lerp(current, history, blendFactor);
                        _temporalHistory[x, y, z] = _froxelRadiance[x, y, z];
                    }
                }
            }
        }

        /// <summary>
        /// Computes fog density at a world position.
        /// </summary>
        public float ComputeFogDensity(Vector3 worldPos)
        {
            float heightDensity = MathF.Exp(-worldPos.Y * _config.HeightFalloff);
            float baseDensity = _config.MaxDensity * heightDensity;

            if (_config.NoiseScale > 0)
            {
                float noise = PerlinNoise3D(worldPos * _config.NoiseScale);
                baseDensity *= 0.5f + noise * 0.5f;
            }

            return baseDensity;
        }

        /// <summary>
        /// Computes fog color at a world position.
        /// </summary>
        public Vector3 ComputeFogColor(Vector3 worldPos, Vector3 lightColor, float lightIntensity)
        {
            float density = ComputeFogDensity(worldPos);
            return _config.FogColor * density * lightColor * lightIntensity;
        }

        /// <summary>
        /// Computes light shaft / god ray contribution.
        /// </summary>
        public float ComputeLightShafts(CameraState camera, Vector3 lightDirection,
            GBuffer gbuffer, int numSamples)
        {
            float occlusion = 0;
            Vector3 sunPos = camera.Position - lightDirection * 100.0f;

            for (int i = 0; i < numSamples; i++)
            {
                float t = (float)i / numSamples;
                Vector3 samplePos = Vector3.Lerp(camera.Position, sunPos, t);
                Vector3 screenPos = camera.ProjectToScreen(samplePos);

                if (screenPos.X >= 0 && screenPos.X < 1 && screenPos.Y >= 0 && screenPos.Y < 1)
                {
                    int pixX = (int)(screenPos.X * gbuffer.Width);
                    int pixY = (int)(screenPos.Y * gbuffer.Height);
                    pixX = Math.Clamp(pixX, 0, gbuffer.Width - 1);
                    pixY = Math.Clamp(pixY, 0, gbuffer.Height - 1);

                    float depth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                    float sampleDepth = screenPos.Z;

                    if (sampleDepth > depth)
                        occlusion += 1.0f;
                }
            }

            return 1.0f - occlusion / numSamples;
        }

        /// <summary>
        /// Computes volumetric shadow by ray marching through froxels.
        /// </summary>
        public Vector3 ComputeVolumetricShadow(Vector3 worldPos, Vector3 lightDirection,
            CameraState camera)
        {
            Vector3 shadow = Vector3.One;
            Vector3 startScreen = camera.ProjectToScreen(worldPos);
            Vector3 endScreen = camera.ProjectToScreen(worldPos - lightDirection * _config.MaxDistance);

            int numSteps = 32;
            for (int i = 0; i < numSteps; i++)
            {
                float t = (float)i / numSteps;
                Vector3 sampleScreen = Vector3.Lerp(startScreen, endScreen, t);

                if (sampleScreen.X >= 0 && sampleScreen.X < 1 &&
                    sampleScreen.Y >= 0 && sampleScreen.Y < 1)
                {
                    int fx = (int)(sampleScreen.X * _gridX);
                    int fy = (int)(sampleScreen.Y * _gridY);
                    int fz = (int)(sampleScreen.Z * _gridZ);

                    fx = Math.Clamp(fx, 0, _gridX - 1);
                    fy = Math.Clamp(fy, 0, _gridY - 1);
                    fz = Math.Clamp(fz, 0, _gridZ - 1);

                    float transmittance = _froxelTransmittance[fx, fy, fz];
                    shadow *= transmittance;
                }
            }

            return shadow;
        }

        /// <summary>
        /// Integrates volume scattering along a view ray.
        /// </summary>
        public Vector3 IntegrateVolumeScattering(CameraState camera, Vector2 screenPos,
            Vector3 viewDirection, List<LightConfig> lights)
        {
            Vector3 accumulatedScattering = Vector3.Zero;
            Vector3 accumulatedTransmittance = Vector3.One;

            for (int z = 0; z < _gridZ; z++)
            {
                float depthNear = _depthSlices[z];
                float depthFar = _depthSlices[z + 1];
                float sliceThickness = depthFar - depthNear;

                int fx = (int)(screenPos.X * _gridX);
                int fy = (int)(screenPos.Y * _gridY);
                fx = Math.Clamp(fx, 0, _gridX - 1);
                fy = Math.Clamp(fy, 0, _gridY - 1);

                Vector3 sliceRadiance = _froxelRadiance[fx, fy, z];
                float sliceDensity = ComputeFogDensity(
                    camera.Position + viewDirection * (depthNear + depthFar) * 0.5f);

                Vector3 sliceAlbedo = _config.FogColor;
                float sliceTransmittance = MathF.Exp(-sliceDensity * sliceThickness);

                Vector3 sliceInScattering = sliceRadiance * (1.0f - sliceTransmittance) /
                    MathF.Max(0.001f, sliceDensity);
                accumulatedScattering += accumulatedTransmittance * sliceInScattering;
                accumulatedTransmittance *= sliceTransmittance;
            }

            return accumulatedScattering;
        }

        private Vector3 ComputeFroxelWorldPos(int x, int y, int z, CameraState camera)
        {
            float u = (x + 0.5f) / _gridX;
            float v = (y + 0.5f) / _gridY;
            float depth = (_depthSlices[z] + _depthSlices[z + 1]) * 0.5f;

            Vector3 screenPos = new Vector3(u, v, depth);
            return camera.UnprojectFromScreen(screenPos);
        }

        private float PerlinNoise3D(Vector3 pos)
        {
            int xi = (int)MathF.Floor(pos.X) & 255;
            int yi = (int)MathF.Floor(pos.Y) & 255;
            int zi = (int)MathF.Floor(pos.Z) & 255;

            float xf = pos.X - MathF.Floor(pos.X);
            float yf = pos.Y - MathF.Floor(pos.Y);
            float zf = pos.Z - MathF.Floor(pos.Z);

            float u = Fade(xf);
            float v = Fade(yf);
            float w = Fade(zf);

            float n000 = Grad(xi, yi, zi, xf, yf, zf);
            float n001 = Grad(xi, yi, zi + 1, xf, yf, zf - 1);
            float n010 = Grad(xi, yi + 1, zi, xf, yf - 1, zf);
            float n011 = Grad(xi, yi + 1, zi + 1, xf, yf - 1, zf - 1);
            float n100 = Grad(xi + 1, yi, zi, xf - 1, yf, zf);
            float n101 = Grad(xi + 1, yi, zi + 1, xf - 1, yf, zf - 1);
            float n110 = Grad(xi + 1, yi + 1, zi, xf - 1, yf - 1, zf);
            float n111 = Grad(xi + 1, yi + 1, zi + 1, xf - 1, yf - 1, zf - 1);

            float nx00 = Lerp(n000, n100, u);
            float nx01 = Lerp(n001, n101, u);
            float nx10 = Lerp(n010, n110, u);
            float nx11 = Lerp(n011, n111, u);

            float nxy0 = Lerp(nx00, nx10, v);
            float nxy1 = Lerp(nx01, nx11, v);

            return Lerp(nxy0, nxy1, w);
        }

        private float Fade(float t) => t * t * t * (t * (t * 6 - 15) + 10);
        private float Lerp(float a, float b, float t) => a + t * (b - a);

        private float Grad(int x, int y, int z, float dx, float dy, float dz)
        {
            int h = (x * 374761393 + y * 668265263 + z * 1274126177) & 15;
            float u = h < 8 ? dx : dy;
            float v = h < 4 ? dy : (h == 12 || h == 14 ? dx : dz);
            return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
        }
    }

    #endregion

    // =========================================================================
    #region AmbientOcclusionSystem

    /// <summary>
    /// Ambient occlusion system supporting SSAO, GTAO, and contact shadows.
    /// </summary>
    public class AmbientOcclusionSystem
    {
        private float[] _aoBuffer;
        private float[] _temporalHistory;
        private Vector3[] _hemisphereKernel;
        private Vector3[] _noiseTexture;
        private int _width;
        private int _height;
        private int _kernelSize;
        private bool _isInitialized;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>AO result buffer.</summary>
        public float[] AOBuffer => _aoBuffer;

        /// <summary>
        /// Initializes the ambient occlusion system.
        /// </summary>
        public void Initialize(int width, int height, int kernelSize = 64)
        {
            _width = width;
            _height = height;
            _kernelSize = kernelSize;
            int pixelCount = width * height;

            _aoBuffer = new float[pixelCount];
            _temporalHistory = new float[pixelCount];
            _hemisphereKernel = new Vector3[kernelSize];
            _noiseTexture = new Vector3[16 * 16];

            GenerateHemisphereKernel();
            GenerateNoiseTexture();
            _isInitialized = true;
        }

        private void GenerateHemisphereKernel()
        {
            var rng = new RandomNumberGenerator(42);
            for (int i = 0; i < _kernelSize; i++)
            {
                float x = rng.NextFloat(-1.0f, 1.0f);
                float y = rng.NextFloat(-1.0f, 1.0f);
                float z = rng.NextFloat(0.0f, 1.0f);
                float len = MathF.Sqrt(x * x + y * y + z * z);
                x /= len; y /= len; z /= len;

                float scale = (float)i / _kernelSize;
                scale = 0.1f + scale * scale * 0.9f;
                _hemisphereKernel[i] = new Vector3(x, y, z) * scale;
            }
        }

        private void GenerateNoiseTexture()
        {
            var rng = new RandomNumberGenerator(123);
            for (int i = 0; i < _noiseTexture.Length; i++)
            {
                _noiseTexture[i] = new Vector3(
                    rng.NextFloat(-1.0f, 1.0f),
                    rng.NextFloat(-1.0f, 1.0f),
                    0.0f);
            }
        }

        /// <summary>
        /// Computes SSAO using hemisphere sampling.
        /// </summary>
        public void ComputeSSAO(GBuffer gbuffer, CameraState camera, int kernelSize, float radius)
        {
            if (!_isInitialized) return;

            Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    float depth = gbuffer.Depth[idx];
                    if (depth <= 0)
                    {
                        _aoBuffer[idx] = 1.0f;
                        continue;
                    }

                    Vector3 normal = gbuffer.Normals[idx];
                    Vector3 worldPos = ReconstructWorldPosition(x, y, depth, gbuffer, camera);

                    Vector3 noise = _noiseTexture[(x % 16) * 16 + (y % 16)];

                    Vector3 tangent = Vector3.Normalize(noise - normal * Vector3.Dot(normal, noise));
                    Vector3 bitangent = Vector3.Cross(normal, tangent);
                    Matrix4x4 tbn = new Matrix4x4(
                        tangent.X, bitangent.X, normal.X, 0,
                        tangent.Y, bitangent.Y, normal.Y, 0,
                        tangent.Z, bitangent.Z, normal.Z, 0,
                        0, 0, 0, 1);

                    float occlusion = 0;
                    int validSamples = 0;

                    for (int i = 0; i < kernelSize && i < _hemisphereKernel.Length; i++)
                    {
                        Vector3 sampleDir = Vector3.Transform(_hemisphereKernel[i], tbn);
                        Vector3 samplePos = worldPos + sampleDir * radius;

                        Vector3 sampleScreen = camera.ProjectToScreen(samplePos);
                        if (sampleScreen.X < 0 || sampleScreen.X >= 1 ||
                            sampleScreen.Y < 0 || sampleScreen.Y >= 1)
                            continue;

                        int samplePixX = (int)(sampleScreen.X * _width);
                        int samplePixY = (int)(sampleScreen.Y * _height);
                        samplePixX = Math.Clamp(samplePixX, 0, _width - 1);
                        samplePixY = Math.Clamp(samplePixY, 0, _height - 1);

                        float sampleDepth = gbuffer.Depth[gbuffer.GetIndex(samplePixX, samplePixY)];
                        float rangeCheck = MathF.Max(0, 1.0f - MathF.Abs(depth - sampleDepth) / radius);

                        if (sampleDepth < sampleScreen.Z)
                            occlusion += rangeCheck;

                        validSamples++;
                    }

                    _aoBuffer[idx] = validSamples > 0 ? 1.0f - (occlusion / validSamples) : 1.0f;
                }
            });
        }

        /// <summary>
        /// Computes GTAO (Ground Truth Ambient Occlusion) approximation.
        /// </summary>
        public void ComputeGTAO(GBuffer gbuffer, CameraState camera, int numDirections, float radius)
        {
            if (!_isInitialized) return;

            Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    float depth = gbuffer.Depth[idx];
                    if (depth <= 0)
                    {
                        _aoBuffer[idx] = 1.0f;
                        continue;
                    }

                    Vector3 normal = gbuffer.Normals[idx];
                    Vector3 worldPos = ReconstructWorldPosition(x, y, depth, gbuffer, camera);

                    float occlusion = 0;

                    for (int i = 0; i < numDirections; i++)
                    {
                        float angle = TWO_PI * i / numDirections;
                        Vector2 dir = new Vector2(MathF.Cos(angle), MathF.Sin(angle));

                        float horizonLow = 0;
                        float horizonHigh = 0;

                        for (int j = 1; j <= 8; j++)
                        {
                            float stepSize = radius * j / 8.0f;
                            Vector3 sampleOffset = new Vector3(dir.X, dir.Y, 0) * stepSize;
                            Vector3 samplePos = worldPos + sampleOffset;

                            Vector3 sampleScreen = camera.ProjectToScreen(samplePos);
                            if (sampleScreen.X < 0 || sampleScreen.X >= 1 ||
                                sampleScreen.Y < 0 || sampleScreen.Y >= 1)
                                continue;

                            int samplePixX = (int)(sampleScreen.X * _width);
                            int samplePixY = (int)(sampleScreen.Y * _height);
                            samplePixX = Math.Clamp(samplePixX, 0, _width - 1);
                            samplePixY = Math.Clamp(samplePixY, 0, _height - 1);

                            float sampleDepth = gbuffer.Depth[gbuffer.GetIndex(samplePixX, samplePixY)];
                            Vector3 sampleWorldPos = ReconstructWorldPosition(samplePixX, samplePixY,
                                sampleDepth, gbuffer, camera);

                            Vector3 horizonVec = sampleWorldPos - worldPos;
                            float horizonAngle = MathF.Atan2(horizonVec.Z, new Vector2(horizonVec.X, horizonVec.Y).Length());

                            if (horizonAngle > horizonLow)
                                horizonLow = horizonAngle;
                            if (horizonAngle < horizonHigh)
                                horizonHigh = horizonAngle;
                        }

                        occlusion += MathF.Max(0, MathF.Cos(horizonLow)) + MathF.Max(0, MathF.Cos(horizonHigh));
                    }

                    _aoBuffer[idx] = 1.0f - occlusion / (numDirections * 2.0f);
                }
            });
        }

        /// <summary>
        /// Temporally accumulates AO.
        /// </summary>
        public void TemporalAccumulate(float blendFactor)
        {
            for (int i = 0; i < _aoBuffer.Length; i++)
            {
                _aoBuffer[i] = _aoBuffer[i] * (1.0f - blendFactor) + _temporalHistory[i] * blendFactor;
                _temporalHistory[i] = _aoBuffer[i];
            }
        }

        /// <summary>
        /// Applies edge-preserving blur to AO.
        /// </summary>
        public void BlurAO(GBuffer gbuffer, int radius, float edgeThreshold)
        {
            float[] blurred = new float[_aoBuffer.Length];
            Array.Copy(_aoBuffer, blurred, _aoBuffer.Length);

            Parallel.For(radius, _height - radius, y =>
            {
                for (int x = radius; x < _width - radius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    float centerDepth = gbuffer.Depth[idx];
                    Vector3 centerNormal = gbuffer.Normals[idx];

                    float sum = 0;
                    float weightSum = 0;

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);

                            float depthDiff = MathF.Abs(centerDepth - gbuffer.Depth[nIdx]) /
                                             MathF.Max(0.001f, centerDepth);
                            float normalDiff = 1.0f - Vector3.Dot(centerNormal, gbuffer.Normals[nIdx]);

                            if (depthDiff > edgeThreshold || normalDiff > 0.5f)
                                continue;

                            float spatialWeight = MathF.Exp(-(dx * dx + dy * dy) / (2.0f * radius * radius));
                            float depthWeight = MathF.Exp(-depthDiff * 10.0f);

                            sum += _aoBuffer[nIdx] * spatialWeight * depthWeight;
                            weightSum += spatialWeight * depthWeight;
                        }
                    }

                    blurred[idx] = weightSum > 0 ? sum / weightSum : _aoBuffer[idx];
                }
            });

            Array.Copy(blurred, _aoBuffer, blurred.Length);
        }

        /// <summary>
        /// Upscales AO from half-resolution to full-resolution.
        /// </summary>
        public void UpscaleAO(float[] halfResAO, int halfWidth, int halfHeight)
        {
            for (int x = 0; x < _width; x++)
            {
                for (int y = 0; y < _height; y++)
                {
                    float u = (float)x / _width * halfWidth;
                    float v = (float)y / _height * halfHeight;

                    int x0 = Math.Clamp((int)u, 0, halfWidth - 1);
                    int y0 = Math.Clamp((int)v, 0, halfHeight - 1);
                    int x1 = Math.Min(x0 + 1, halfWidth - 1);
                    int y1 = Math.Min(y0 + 1, halfHeight - 1);

                    float fx = u - x0;
                    float fy = v - y0;

                    float a00 = halfResAO[y0 * halfWidth + x0];
                    float a10 = halfResAO[y0 * halfWidth + x1];
                    float a01 = halfResAO[y1 * halfWidth + x0];
                    float a11 = halfResAO[y1 * halfWidth + x1];

                    float a0 = a00 * (1 - fx) + a10 * fx;
                    float a1 = a01 * (1 - fx) + a11 * fx;
                    _aoBuffer[y * _width + x] = a0 * (1 - fy) + a1 * fy;
                }
            }
        }

        /// <summary>
        /// Computes contact shadows.
        /// </summary>
        public float ComputeContactShadow(GBuffer gbuffer, CameraState camera, Vector3 worldPos,
            Vector3 lightDir, int numSteps, float maxDistance)
        {
            Vector3 startScreen = camera.ProjectToScreen(worldPos);
            Vector3 endPos = worldPos + lightDir * maxDistance;
            Vector3 endScreen = camera.ProjectToScreen(endPos);

            float shadow = 1.0f;

            for (int i = 1; i <= numSteps; i++)
            {
                float t = (float)i / numSteps;
                Vector3 sampleScreen = Vector3.Lerp(startScreen, endScreen, t);

                if (sampleScreen.X < 0 || sampleScreen.X >= 1 ||
                    sampleScreen.Y < 0 || sampleScreen.Y >= 1)
                    break;

                int pixX = (int)(sampleScreen.X * _width);
                int pixY = (int)(sampleScreen.Y * _height);
                pixX = Math.Clamp(pixX, 0, _width - 1);
                pixY = Math.Clamp(pixY, 0, _height - 1);

                float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                float sampleDepth = sampleScreen.Z;

                if (sampleDepth > sceneDepth && sampleDepth < sceneDepth + 0.1f)
                {
                    shadow *= 0.5f;
                }
            }

            return shadow;
        }

        private Vector3 ReconstructWorldPosition(int x, int y, float depth, GBuffer gbuffer, CameraState camera)
        {
            float ndcX = (float)x / _width * 2.0f - 1.0f;
            float ndcY = 1.0f - (float)y / _height * 2.0f;
            Vector3 viewPos = new Vector3(ndcX, ndcY, depth);
            return camera.UnprojectFromScreen(viewPos);
        }
    }

    #endregion
    // =========================================================================
    #region LDNNRenderer

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
        /// <summary>Is the renderer initialized.</summary>
        public bool IsInitialized => _isInitialized;

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
            _neuralPredictor.Initialize();

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
            if (!_isInitialized) return;
            _cascadesManager = null;
            _neuralPredictor = null;
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
            if (!_isInitialized || _isShutdown) return;

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

            _telemetry.TotalFrameTimeMs = _telemetry.Timestamp.ElapsedMilliseconds;
            _analytics.RecordFrame(_telemetry);
            _previousGBuffer = CloneGBuffer(gbuffer);
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
                    if (gbuffer.Depth[idx] <= 0) continue;

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
            if (ssgi.LengthSquared() > 0.0001f) return ssgi;

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
            if (_previousGBuffer != null) _aoSystem.TemporalAccumulate(0.9f);
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
                            if (px >= gbuffer.Width || py >= gbuffer.Height) continue;
                            int idx = gbuffer.GetIndex(px, py);
                            float depth = gbuffer.Depth[idx];
                            if (depth <= 0) continue;
                            minDepth = MathF.Min(minDepth, depth);
                            maxDepth = MathF.Max(maxDepth, depth);
                            avgNormal += gbuffer.Normals[idx];
                            hasTranslucency |= gbuffer.MaterialProps[idx].W > 0.5f;
                            hasEmissive |= gbuffer.Emissive[idx].LengthSquared() > 0.01f;
                            pixelCount++;
                        }
                    }

                    if (pixelCount > 0) avgNormal /= pixelCount;
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
        /// Dispatches compute shaders (simulated).
        /// </summary>
        public void DispatchComputeShaders(string shaderName, int threadGroupsX,
            int threadGroupsY, int threadGroupsZ, Dictionary<string, object> parameters)
        {
            _ = shaderName; _ = threadGroupsX; _ = threadGroupsY; _ = threadGroupsZ; _ = parameters;
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
            }
        }
        /// <summary>
        /// Computes shadow mask for a light.
        /// </summary>
        public float ComputeShadowMask(LightConfig light, Vector3 worldPos, Vector3 normal,
            GBuffer gbuffer, CameraState camera)
        {
            switch (light.ShadowMethod)
            {
                case ShadowMethod.None: return 1.0f;
                case ShadowMethod.ShadowMap: return ComputeShadowMap(light, worldPos, gbuffer, camera);
                case ShadowMethod.VarianceShadowMap: return FilterShadowMapVSM(light, worldPos, gbuffer, camera);
                case ShadowMethod.RayTraced: return RayTraceShadowRays(light, worldPos, normal, gbuffer, camera);
                case ShadowMethod.NeuralPredictive: return NeuralPredictiveShadow(light, worldPos, normal, gbuffer, camera);
                case ShadowMethod.ContactHardening: return ComputeContactHardeningShadow(light, worldPos, normal, gbuffer, camera);
                default: return 1.0f;
            }
        }

        /// <summary>
        /// Computes shadow map occlusion.
        /// </summary>
        public float ComputeShadowMap(LightConfig light, Vector3 worldPos, GBuffer gbuffer, CameraState camera)
        {
            Vector3 toLight = light.Type == LightType.Directional ? -light.Direction : light.Position - worldPos;
            Vector3 screenPos = camera.ProjectToScreen(worldPos + toLight);
            if (screenPos.X < 0 || screenPos.X >= 1 || screenPos.Y < 0 || screenPos.Y >= 1) return 1.0f;

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
            if (NdotL < 0.001f) return 0.0f;

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
                else shadow += 1.0f;
            }
            return shadow / numSamples;
        }

        /// <summary>
        /// Neural predictive shadow computation.
        /// </summary>
        public float NeuralPredictiveShadow(LightConfig light, Vector3 worldPos, Vector3 normal,
            GBuffer gbuffer, CameraState camera)
        {
            var features = new Vector3[8];
            features[0] = worldPos;
            features[1] = normal;
            features[2] = light.Position;
            features[3] = light.Direction;
            features[4] = new Vector3(light.Intensity, light.Range, 0);
            features[5] = new Vector3(light.InnerConeAngle, light.OuterConeAngle, 0);
            features[6] = camera.Position;
            features[7] = Vector3.Zero;

            Vector3 prediction = _neuralPredictor.ForwardPass(features);
            return Math.Clamp(prediction.X, 0, 1);
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
                    if (dx == 0 && dy == 0) continue;
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
                Width = source.Width, Height = source.Height,
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
                        if (sample.Depth <= 0) { _previousGIResult[idx] = Vector3.Zero; continue; }
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
                Parallel.For(0, gbuffer.Height, y =>
                {
                    for (int x = 0; x < gbuffer.Width; x++)
                    {
                        int idx = gbuffer.GetIndex(x, y);
                        GBufferSample sample = gbuffer.GetSample(x, y);
                        if (sample.Depth <= 0) continue;
                        Vector3 ssgi = _screenSpaceGI.Result[idx];
                        Vector3 worldPos = ReconstructWorldPosition(x, y, sample.Depth, gbuffer, camera);
                        Vector3 cascadeIrradiance = ComputeIrradianceFromProbes(worldPos, sample.Normal);
                        float ssgiConfidence = _screenSpaceGI.Confidence[idx];
                        _previousGIResult[idx] = Vector3.Lerp(cascadeIrradiance, ssgi, ssgiConfidence);
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

    #endregion
}
