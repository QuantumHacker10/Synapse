using FluentAssertions;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public class IndustrialPhysicsRuntimeTests
{
    [Fact]
    public void EngineHost_TickPhysics_ShouldDriveRigidBodiesAndField()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var config = new SynapseConfig { PhysicsBudgetMs = 50, QualityPreset = "High" };
        var host = new EngineHost(config, logger);

        host.InitializeModules();
        host.Multiphysics.Should().NotBeNull();
        host.RigidWorld.Should().NotBeNull();
        host.RigidWorld!.Bodies.Count.Should().BeGreaterThan(0);

        for (int i = 0; i < 30; i++)
            host.TickPhysics(1f / 60f);

        host.Multiphysics!.LastStats.SubSteps.Should().BeGreaterThan(0);
        host.AverageFieldTemperature.Should().BeGreaterThan(0f);
        float.IsFinite(host.AverageFieldTemperature).Should().BeTrue();
    }
}
