using System;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;
using FluentAssertions;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class EngineHostSceneLifecycleTests
{
    [Fact]
    public async Task CreateUpdateDelete_SaveLoadScene()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { SimulationSeed = 3 }, logger);
        host.InitializeModules();

        var id = host.CreateSceneEntity("ProdBox", "Empty");
        id.Should().NotBe(Guid.Empty);
        host.Scene.Entities.Should().Contain(e => e.Id == id);

        host.DeleteSceneEntity(id).Should().BeTrue();
        host.Scene.Entities.Should().NotContain(e => e.Id == id);

        var path = Path.Combine(Path.GetTempPath(), $"scene-{Guid.NewGuid():N}.synapse");
        try
        {
            host.CreateSceneEntity("Persist", "Empty");
            await host.SaveSceneAsync(path);
            File.Exists(path).Should().BeTrue();

            var host2 = new EngineHost(new SynapseConfig(), logger);
            host2.InitializeModules();
            await host2.LoadSceneAsync(path);
            host2.Scene.Entities.Should().Contain(e => e.Name == "Persist");
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public void TickPhysics_AdvancesWithoutThrow()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);
        host.InitializeModules();
        host.ApplyLaw("heat_equation");
        host.TickPhysics(1f / 60f);
        host.AverageFieldTemperature.Should().BeGreaterThan(0);
    }

    [Fact]
    public void SetEntityAsVehicle_AndHinge_DoNotThrow()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig(), logger);
        host.InitializeModules();
        var id = host.CreateSceneEntity("Car", "Empty");
        host.SetEntityAsVehicle(id, true).Should().BeTrue();
        // Hinge may return null if body not synced — must not throw.
        _ = host.AddHingeToWorld(id, Vector3.UnitY);
    }
}
