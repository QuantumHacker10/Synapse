using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Rendering.FrameGraph;
using GDNN.Rendering.FrameGraph.Passes;
using GDNN.Rendering.Quality;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Physics;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class NativeFramePipelineTests
{
    [Fact]
    public void FrameGraphFactory_RegistersNativeModulePasses()
    {
        var graph = FrameGraphFactory.CreateDefault();
        graph.Passes.Should().HaveCount(8);
        graph.Passes[0].Should().BeOfType<LdnnGiPass>();
        graph.Passes[0].Phase.Should().Be(RenderPassPhase.CpuProducer);
        graph.Passes[1].Phase.Should().Be(RenderPassPhase.Gpu);
        graph.Passes[^1].Should().BeOfType<PostTonemapPass>();
    }

    [Fact]
    public void RuntimeQualityMapper_MapsInfrastructureLevel()
    {
        var mapped = RuntimeQualityMapper.FromLevel(QualityPresets.High);
        mapped.EnableGlobalIllumination.Should().BeTrue();
        mapped.ShadowCascades.Should().BeGreaterThan(0);
        mapped.PresetName.Should().Be("High");
    }

    [Fact]
    public async Task NativePipeline_TicksPhysicsSimulationWithoutRender()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { Headless = true, PhysicsBudgetMs = 50 }, logger);
        host.InitializeModules();

        host.Multiphysics!.Config.EnabledModules.Should().HaveFlag(ContinuumModules.Sph);
        host.Multiphysics.Config.EnabledModules.Should().HaveFlag(ContinuumModules.Elasticity);

        var orch = new FrameOrchestrator(host, logger);
        var stats = await orch.TickAsync();

        stats.PhysicsMs.Should().BeGreaterThanOrEqualTo(0);
        stats.EntityCount.Should().BeGreaterThan(0);
        stats.ActiveLawId.Should().NotBeNullOrEmpty();
        stats.RenderReady.Should().BeFalse();
        stats.ContinuumModules.Should().Contain("Sph");
        host.AverageFieldTemperature.Should().BeGreaterThan(0f);
    }

    [Fact]
    public void EngineHost_SyncSimulationAndFieldBridges_AreSafeWithoutRender()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { Headless = true }, logger);
        host.InitializeModules();

        host.Invoking(h => h.SyncSimulationTransformsToScene()).Should().NotThrow();
        host.Invoking(h => h.PushPhysicsFieldToRenderer()).Should().NotThrow();
        host.IsRenderInitialized.Should().BeFalse();
    }
}
