using System.Numerics;
using FluentAssertions;
using GDNN.Lighting.LDNN;
using GDNN.Rendering.Bridge;
using Xunit;

namespace Synapse.Tests.Rendering;

/// <summary>
/// Lot Réalisme L-DNN : teacher path tracing, ombres neuronales,
/// spéculaire neural, nuages volumétriques.
/// </summary>
public class LDNNRealismTests
{
    private const int Size = 16;

    private static LDNNConfig CreateConfig(bool teacher = true) => new()
    {
        QualityMode = teacher ? LDNNQualityMode.FullPathTraceTeacher : LDNNQualityMode.HybridRT,
        GIComputationMode = GIComputationMode.Hybrid,
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
            MaxDensity = 0.05f,
            HeightFalloff = 0.1f,
            ReferenceHeight = 0.0f,
            FogColor = new Vector3(0.7f, 0.8f, 1.0f),
            Anisotropy = 0.3f,
            StartDistance = 1.0f,
            MaxDistance = 100.0f,
            DepthSlices = 8,
            GridResolutionXY = 4,
            TemporalReprojectionStrength = 0.5f,
            NoiseScale = 0.05f,
            NoiseSpeed = 0.1f,
            ShadowIntensity = 0.5f,
            EnableClouds = true,
            CloudAltitude = 10.0f,
            CloudThickness = 5.0f,
            CloudCoverage = 0.7f,
            CloudDensityScale = 3.0f,
            CloudNoiseScale = 0.1f
        },
        SamplingMode = HemisphereSamplingMode.CosineWeighted,
        MaxPathDepth = 2,
        ReferenceSamplesPerPixel = 1,
        EnableNeuralTemporal = false,
        NeuralLearningRate = 0.01f,
        NeuralBatchSize = 8,
        NeuralNetworkProfile = NeuralNetworkProfile.Tiny,
        EnableOnlineTeacherTraining = teacher,
        TeacherPixelStride = 8,
        TeacherSamplesPerPixel = 1,
        EnableNeuralSpecular = true
    };

    private static (LDNNRenderer Renderer, GBuffer GBuffer, CameraState Camera, List<LightConfig> Lights)
        CreateScene(LDNNConfig config)
    {
        var renderer = new LDNNRenderer();
        renderer.Initialize(config, Size, Size);

        var gbuffer = new GBuffer
        {
            Width = Size,
            Height = Size,
            Depth = new float[Size * Size],
            Normals = new Vector3[Size * Size],
            Albedo = new Vector3[Size * Size],
            Velocity = new Vector2[Size * Size],
            MaterialProps = new Vector4[Size * Size],
            Specular = new Vector3[Size * Size],
            Emissive = new Vector3[Size * Size]
        };
        for (int i = 0; i < Size * Size; i++)
        {
            gbuffer.Depth[i] = 5.0f;
            gbuffer.Normals[i] = Vector3.UnitY;
            gbuffer.Albedo[i] = new Vector3(0.6f, 0.5f, 0.4f);
            gbuffer.Specular[i] = new Vector3(0.04f);
            gbuffer.MaterialProps[i] = new Vector4(0.3f, 0.1f, 0, 0);
        }

        var camera = new CameraState
        {
            Position = new Vector3(0, 2, -6),
            Forward = Vector3.UnitZ,
            Right = Vector3.UnitX,
            Up = Vector3.UnitY,
            FieldOfView = 1.0f,
            AspectRatio = 1.0f,
            NearPlane = 0.1f,
            FarPlane = 100f
        };
        camera.ViewMatrix = Matrix4x4.CreateLookAt(camera.Position, Vector3.Zero, Vector3.UnitY);
        camera.ProjectionMatrix = Matrix4x4.CreatePerspectiveFieldOfView(1.0f, 1.0f, 0.1f, 100f);
        camera.Recompute();

        var lights = new List<LightConfig>
        {
            new()
            {
                Type = LightType.Directional,
                Direction = Vector3.Normalize(new Vector3(0.2f, -1f, 0.3f)),
                Color = Vector3.One,
                Intensity = 3.0f,
                ShadowMethod = ShadowMethod.NeuralPredictive,
                ShadowBias = 0.005f,
                ShadowSamples = 4
            }
        };

        return (renderer, gbuffer, camera, lights);
    }

    [Fact]
    public void TrainFromPathTracerTeacher_CollectsSamplesAndAdvancesTraining()
    {
        var config = CreateConfig(teacher: true);
        var (renderer, gbuffer, camera, lights) = CreateScene(config);

        int before = renderer.NeuralPredictor.TrainingStep;
        int collected = renderer.TrainFromPathTracerTeacher(gbuffer, camera, lights);

        collected.Should().BeGreaterThan(0);
        renderer.NeuralPredictor.TrainingStep.Should().BeGreaterThanOrEqualTo(before);
    }

    [Fact]
    public void RenderFrame_WithTeacherEnabled_RunsWithoutThrowing()
    {
        var config = CreateConfig(teacher: true);
        var (renderer, gbuffer, camera, lights) = CreateScene(config);
        var context = new RenderContext
        {
            FrameIndex = 0,
            RenderWidth = Size,
            RenderHeight = Size,
            ResourcePool = new Dictionary<string, object>(),
            ConstantBufferData = new Dictionary<string, float>()
        };

        var act = () => renderer.RenderFrame(gbuffer, camera, lights, context);
        act.Should().NotThrow();
        renderer.FrameIndex.Should().Be(1);
    }

    [Fact]
    public void NeuralPredictiveShadow_ReturnsSoftOcclusionInUnitRange()
    {
        var config = CreateConfig(teacher: false);
        var (renderer, gbuffer, camera, lights) = CreateScene(config);

        float shadow = renderer.ComputeShadowMask(
            lights[0], new Vector3(0, 0, 0), Vector3.UnitY, gbuffer, camera);

        shadow.Should().BeInRange(0f, 1f);
    }

    [Fact]
    public void NeuralSpecular_PredictsGlossyAndRefraction()
    {
        var predictor = new NeuralSpecularPredictor();
        predictor.Initialize();

        var glossy = new GBufferSample
        {
            WorldPosition = Vector3.Zero,
            Normal = Vector3.UnitY,
            Albedo = new Vector3(0.9f, 0.8f, 0.7f),
            Specular = new Vector3(0.9f),
            Roughness = 0.05f,
            Metallic = 1.0f,
            IsTranslucent = false
        };
        var camera = new CameraState
        {
            Position = new Vector3(0, 1, -2),
            Forward = Vector3.UnitZ,
            Up = Vector3.UnitY,
            Right = Vector3.UnitX
        };

        Vector3 reflection = predictor.Predict(glossy, camera);
        reflection.Length().Should().BeGreaterThan(0f);

        var glass = glossy with { IsTranslucent = true, Metallic = 0f, Roughness = 0.1f };
        Vector3 refraction = predictor.PredictRefraction(glass, camera);
        refraction.Length().Should().BeGreaterThan(0f);
    }

    [Fact]
    public void ComputeCloudDensity_PeaksNearCloudAltitude()
    {
        var config = CreateConfig(teacher: false);
        var (renderer, _, _, _) = CreateScene(config);

        float atCloud = renderer.Volumetrics.ComputeCloudDensity(new Vector3(0, 10, 0));
        float farBelow = renderer.Volumetrics.ComputeCloudDensity(new Vector3(0, -50, 0));

        atCloud.Should().BeGreaterThanOrEqualTo(0f);
        farBelow.Should().Be(0f);
    }

    [Fact]
    public void Bridge_Defaults_EnableTeacherSpecularAndNeuralShadows()
    {
        using var bridge = new LDNNBridge(8, 8);
        bridge.Initialize();
        bridge.AddDirectionalLight(Vector3.UnitY * -1, Vector3.One, 1f);

        bridge.Renderer.Config.EnableOnlineTeacherTraining.Should().BeTrue();
        bridge.Renderer.Config.EnableNeuralSpecular.Should().BeTrue();
        bridge.Renderer.Config.VolumeFogConfig.EnableClouds.Should().BeTrue();
    }
}
