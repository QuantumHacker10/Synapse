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
        /// <summary>Enable procedural cloud layer on top of height fog.</summary>
        public bool EnableClouds { get; init; }
        /// <summary>World-space altitude of the cloud layer center.</summary>
        public float CloudAltitude { get; init; } = 80.0f;
        /// <summary>Half-thickness of the cloud slab.</summary>
        public float CloudThickness { get; init; } = 40.0f;
        /// <summary>Coverage in [0,1] — higher means denser cloud banks.</summary>
        public float CloudCoverage { get; init; } = 0.45f;
        /// <summary>Multiplier applied to cloud density relative to MaxDensity.</summary>
        public float CloudDensityScale { get; init; } = 2.0f;
        /// <summary>Horizontal noise frequency for cloud shapes.</summary>
        public float CloudNoiseScale { get; init; } = 0.02f;
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
}
