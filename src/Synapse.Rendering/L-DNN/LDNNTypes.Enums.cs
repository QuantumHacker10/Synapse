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
}
