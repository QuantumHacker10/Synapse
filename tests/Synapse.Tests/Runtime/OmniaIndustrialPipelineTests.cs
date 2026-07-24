using FluentAssertions;
using GDNN.Llm;
using GDNN.Polygonization;
using GDNN.Lighting.LDNN;
using GDNN.Rendering;
using GDNN.Scene;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Physics;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public class OmniaIndustrialPipelineTests
{
    [Fact]
    public void ParseWorldDelta_LawAndLighting_ExtractsBoth()
    {
        const string reply = """
            {
              "directionalDirection": [0.2, -1.0, 0.1],
              "intensity": 1.8,
              "fogDensity": 0.04,
              "law": {
                "lawId": "heat_equation",
                "enableModules": ["sph"]
              },
              "primitive": "sphere",
              "center": [0, 1, 0],
              "radius": 0.5
            }
            """;

        var delta = StructuredOutputParser.ParseWorldDelta(reply);

        delta.HasAny.Should().BeTrue();
        delta.Lighting.Should().NotBeNull();
        delta.Lighting!.Intensity.Should().BeApproximately(1.8f, 1e-4f);
        delta.Law.Should().NotBeNull();
        delta.Law!.LawId.Should().Be("heat_equation");
        delta.Sdf.Should().NotBeNull();
        delta.Sdf!.Primitive.Should().Be("sphere");
    }

    [Fact]
    public void ApplyLlmWorldDelta_LawHint_ActivatesLivingLaw()
    {
        using var logger = new SynapseLogger(null, LogLevel.Warning, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);

        const string reply = """
            {
              "lawId": "heat_equation",
              "enableModules": ["sph"]
            }
            """;

        string status = host.ApplyLlmWorldDelta(reply);

        status.Should().Contain("law:heat_equation");
        host.ActiveLawId.Should().Be("heat_equation");
        host.Multiphysics.Should().NotBeNull();
        (host.Multiphysics!.Config.EnabledModules & ContinuumModules.Sph)
            .Should().Be(ContinuumModules.Sph);
    }

    [Fact]
    public void ApplyLlmWorldDelta_FullCascade_AppliesLightingLawAndImpulse()
    {
        using var logger = new SynapseLogger(null, LogLevel.Warning, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);
        host.InitializeModules();
        host.Multiphysics!.SeedDemoBodies();

        const string reply = """
            {
              "intensity": 2.0,
              "color": "#FFAA66",
              "fogDensity": 0.05,
              "lawId": "heat_equation",
              "impulse": {
                "position": [0, 3, 0],
                "impulse": [0, 5, 0],
                "heatDeposit": 2.5,
                "profile": "patrol"
              }
            }
            """;

        string status = host.ApplyLlmWorldDelta(reply);

        status.Should().Contain("lighting");
        status.Should().Contain("law:heat_equation");
        host.EntityCount.Should().BeGreaterThan(0);
        host.AverageFieldTemperature.Should().BeGreaterThan(0f);
    }

    [Fact]
    public async Task FrameOrchestrator_Tick_RunsCouplingStage()
    {
        using var logger = new SynapseLogger(null, LogLevel.Warning, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);
        host.InitializeModules();
        var orch = new FrameOrchestrator(host, logger);

        var stats = await orch.TickAsync();

        stats.PhysicsMs.Should().BeGreaterThanOrEqualTo(0f);
        stats.CouplingMs.Should().BeGreaterThanOrEqualTo(0f);
        host.IndustrialPipeline.Should().NotBeNull();
        host.IndustrialPipeline!.FieldCoupler.LastAverageTemperature.Should().BeGreaterThan(0f);
    }

    [Fact]
    public void PhysicsActuator_DepositHeat_RaisesFieldTemperature()
    {
        using var logger = new SynapseLogger(null, LogLevel.Warning, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);
        host.InitializeModules();
        float before = host.Multiphysics!.LastStats.AverageTemperature;
        // Force a sample after deposit
        host.Multiphysics.DepositHeat(System.Numerics.Vector3.Zero, 10f);
        host.TickPhysics(1f / 60f);
        host.AverageFieldTemperature.Should().BeGreaterThanOrEqualTo(before);
    }
}

public class NaniteLumenNeural30Tests
{
    [Fact]
    public void NaniteNeural30_ContinuousLod_IncreasesWhenClose()
    {
        float far = NaniteNeural30.ContinuousLod(40f, 2f);
        float near = NaniteNeural30.ContinuousLod(2f, 40f);
        near.Should().BeGreaterThan(far);
        NaniteNeural30.PolyResolution(near, 1080).Should().BeGreaterThanOrEqualTo(
            NaniteNeural30.PolyResolution(far, 1080));
    }

    [Fact]
    public void NaniteNeural30_VisibilityBuffer_ScalesWithLod()
    {
        var (w0, _) = NaniteNeural30.VisibilityBufferSize(1920, 1080, 0.1f);
        var (w1, _) = NaniteNeural30.VisibilityBufferSize(1920, 1080, 0.9f);
        w1.Should().BeGreaterThanOrEqualTo(w0);
    }

    [Fact]
    public void LumenNeural30_MultiBounce_AmplifiesDiffuse()
    {
        var result = LumenNeural30.MultiBounceRefine(
            new System.Numerics.Vector3(0.2f),
            new System.Numerics.Vector3(0.3f),
            new System.Numerics.Vector3(0.1f),
            new System.Numerics.Vector3(0.8f),
            new System.Numerics.Vector3(0.05f));
        result.X.Should().BeGreaterThan(0.2f);
    }

    [Fact]
    public void LumenNeural30_PhysicsCoupling_WarmsFog()
    {
        var fog = new VolumeFogConfig { MaxDensity = 0.02f, FogColor = new System.Numerics.Vector3(0.7f) };
        var hot = LumenNeural30.ApplyPhysicsToFog(fog, 600f, 1.5f, 40f);
        hot.MaxDensity.Should().BeGreaterThan(fog.MaxDensity);
    }

    [Fact]
    public void PhysicsFieldGiCoupler_Ingest_ComputesAverages()
    {
        var coupler = new PhysicsFieldGiCoupler();
        int g = 8;
        var temp = new float[g * g * g];
        var dens = new float[g * g * g];
        for (int i = 0; i < temp.Length; i++)
        {
            temp[i] = 320f;
            dens[i] = 1.1f;
        }

        coupler.IngestField(temp, dens, g, g, g);
        coupler.LastAverageTemperature.Should().BeApproximately(320f, 1e-3f);
        coupler.LastFogDensityScale.Should().BeGreaterThan(1f);
    }

    [Fact]
    public void TryParseLivingLawHint_FromText_FindsLibraryId()
    {
        bool ok = StructuredOutputParser.TryParseLivingLawHint(
            "Please apply law heat_equation to the scene",
            out var hint);
        ok.Should().BeTrue();
        hint.LawId.Should().Be("heat_equation");
    }
}
