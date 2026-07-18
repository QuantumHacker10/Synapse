using System.Numerics;
using FluentAssertions;
using GDNN.Lighting.LDNN;
using GDNN.Rendering.Bridge;
using Xunit;

namespace Synapse.Tests.Rendering;

/// <summary>
/// Cache de résultats L-DNN pour scènes statiques : une frame identique est
/// servie sans relancer le renderer, toute mutation (caméra, lumières,
/// G-Buffer) force un recalcul.
/// </summary>
public class LDNNBridgeCacheTests
{
    private const int Size = 8;

    private static LDNNConfig CreateTinyConfig() => new()
    {
        QualityMode = LDNNQualityMode.NeuralOnly,
        GIComputationMode = GIComputationMode.SSGI,
        CascadeConfig = new CascadeConfig
        {
            NumLevels = 2,
            BaseResolution = CascadeResolution.Low64,
            AllocationStrategy = CascadeAllocationStrategy.Logarithmic,
            MemoryBudgetBytes = 16 * 1024 * 1024,
            TimeBudgetMs = 4.0f,
            EnableTemporalAccumulation = false,
            TemporalBlendFactor = 0.1f,
            SpatialFilterRadius = 1,
            DistanceScale = 1.0f,
            AngularCoverage = 1.0f
        },
        DenoiseConfig = new DenoiseConfig
        {
            PrimaryDenoiser = DenoiserType.SpatialBilateral,
            SecondaryDenoiser = DenoiserType.TemporalAccumulation,
            SpatialRadius = 1,
            TemporalFrames = 2,
            NormalThreshold = 0.5f,
            DepthThreshold = 0.01f,
            LuminanceThreshold = 0.3f,
            Strength = 0.8f,
            Iterations = 1,
            UseHalfRes = false
        },
        TemporalConfig = new TemporalConfig
        {
            Mode = TemporalFilterMode.None,
            BaseBlendFactor = 0.1f,
            MaxHistoryLength = 2,
            VarianceClippingStrength = 0.5f,
            DisocclusionThreshold = 0.3f,
            ResponseSpeed = 0.5f,
            AccumulationSpeed = 0.1f,
            UseYCoCg = false
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
            DepthSlices = 8,
            GridResolutionXY = 4,
            TemporalReprojectionStrength = 0.8f,
            NoiseScale = 0.05f,
            NoiseSpeed = 0.1f,
            ShadowIntensity = 0.5f
        },
        SamplingMode = HemisphereSamplingMode.CosineWeighted,
        MaxPathDepth = 2,
        ReferenceSamplesPerPixel = 4,
        EnableNeuralTemporal = false,
        NeuralLearningRate = 0.001f,
        NeuralBatchSize = 8,
        NeuralNetworkProfile = NeuralNetworkProfile.Tiny
    };

    private static LDNNBridge CreateStaticSceneBridge()
    {
        var bridge = new LDNNBridge(Size, Size);
        bridge.Initialize(CreateTinyConfig());
        bridge.Resize(Size, Size);
        bridge.UpdateCamera(
            Matrix4x4.CreateLookAt(new Vector3(0, 0, -3), Vector3.Zero, Vector3.UnitY),
            Matrix4x4.CreatePerspectiveFieldOfView(1.0f, 1.0f, 0.1f, 100f),
            new Vector3(0, 0, -3), Vector3.UnitZ, Vector3.UnitX, Vector3.UnitY,
            1.0f, 1.0f, 0.1f, 100f);
        bridge.AddDirectionalLight(new Vector3(0, -1, 0), Vector3.One, 1.0f);
        bridge.FillGBufferFromConstants(5.0f, Vector3.UnitY, new Vector3(0.5f, 0.5f, 0.5f));
        return bridge;
    }

    [Fact]
    public void RenderGI_StaticScene_SecondFrameIsServedFromCache()
    {
        using var bridge = CreateStaticSceneBridge();

        var first = bridge.RenderGI();
        var second = bridge.RenderGI();

        bridge.StaticCacheHits.Should().Be(1);
        second.Should().BeSameAs(first, "aucun état n'a changé entre les deux frames");
    }

    [Fact]
    public void RenderGI_CameraMoved_Recomputes()
    {
        using var bridge = CreateStaticSceneBridge();
        var first = bridge.RenderGI();

        bridge.UpdateCamera(
            Matrix4x4.CreateLookAt(new Vector3(1, 0, -3), Vector3.Zero, Vector3.UnitY),
            Matrix4x4.CreatePerspectiveFieldOfView(1.0f, 1.0f, 0.1f, 100f),
            new Vector3(1, 0, -3), Vector3.UnitZ, Vector3.UnitX, Vector3.UnitY,
            1.0f, 1.0f, 0.1f, 100f);
        var second = bridge.RenderGI();

        bridge.StaticCacheHits.Should().Be(0);
        second.Should().NotBeSameAs(first);
    }

    [Fact]
    public void RenderGI_LightAdded_Recomputes()
    {
        using var bridge = CreateStaticSceneBridge();
        var first = bridge.RenderGI();

        bridge.AddPointLight(new Vector3(0, 2, 0), Vector3.One, 2.0f, 10.0f);
        var second = bridge.RenderGI();

        bridge.StaticCacheHits.Should().Be(0);
        second.Should().NotBeSameAs(first);
    }

    [Fact]
    public void RenderGI_GBufferRefilled_Recomputes()
    {
        using var bridge = CreateStaticSceneBridge();
        var first = bridge.RenderGI();

        bridge.FillGBufferFromConstants(5.0f, Vector3.UnitY, new Vector3(0.9f, 0.1f, 0.1f));
        var second = bridge.RenderGI();

        bridge.StaticCacheHits.Should().Be(0);
        second.Should().NotBeSameAs(first);
    }

    [Fact]
    public void RenderGI_CacheDisabled_AlwaysRecomputes()
    {
        using var bridge = CreateStaticSceneBridge();
        bridge.EnableStaticSceneCache = false;

        var first = bridge.RenderGI();
        var second = bridge.RenderGI();

        bridge.StaticCacheHits.Should().Be(0);
        second.Should().NotBeSameAs(first);
    }

    [Fact]
    public void RenderGI_AfterExplicitInvalidation_Recomputes()
    {
        using var bridge = CreateStaticSceneBridge();
        var first = bridge.RenderGI();

        bridge.InvalidateGICache();
        var second = bridge.RenderGI();

        bridge.StaticCacheHits.Should().Be(0);
        second.Should().NotBeSameAs(first);
    }
}
