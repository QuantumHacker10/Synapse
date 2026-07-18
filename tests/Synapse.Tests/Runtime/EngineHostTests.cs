using System;
using System.IO;
using System.Threading.Tasks;
using FluentAssertions;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime
{
    public sealed class EngineHostTests
    {
        [Fact]
        public void InitializeModules_WiresPhysicsSimulationAndLaws()
        {
            using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
            var host = new EngineHost(new SynapseConfig(), logger);

            host.InitializeModules();

            host.ListLaws().Should().NotBeEmpty();
            host.ActiveLawId.Should().NotBeNullOrWhiteSpace();
            host.EntityCount.Should().BeGreaterThan(0);
        }

        [Fact]
        public async Task FrameOrchestrator_TicksWithoutRender()
        {
            using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
            var host = new EngineHost(new SynapseConfig { Headless = true }, logger);
            host.InitializeModules();
            var orch = new FrameOrchestrator(host, logger);

            var stats = await orch.TickAsync();

            stats.PhysicsMs.Should().BeGreaterThanOrEqualTo(0);
            stats.EntityCount.Should().BeGreaterThan(0);
            stats.ActiveLawId.Should().NotBeNullOrEmpty();
        }

        [Fact]
        public async Task SceneDocument_RoundTrips()
        {
            var path = Path.Combine(Path.GetTempPath(), $"synapse-test-{Guid.NewGuid():N}.synapse");
            try
            {
                var demo = SceneDocument.CreateDemo();
                await demo.SaveAsync(path);
                var loaded = await SceneDocument.LoadAsync(path);
                loaded.Name.Should().Be(demo.Name);
                loaded.Entities.Should().HaveCount(demo.Entities.Count);
                loaded.ActiveLawId.Should().Be("heat_equation");
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Fact]
        public void CompileLaw_HotReloadsExpression()
        {
            using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
            var host = new EngineHost(new SynapseConfig(), logger);
            host.InitializeModules();

            var result = host.CompileLaw("heat_equation", "T");
            result.Success.Should().BeTrue();
            host.ActiveLawId.Should().Be("heat_equation");
        }

        [Fact]
        public void DigitalTwinRegistry_RegistersAndSearches()
        {
            var registry = new InMemoryDigitalTwinRegistry();
            var twin = new InMemoryDigitalTwin { PhysicalId = "sensor-1" };
            twin.SetProperty("Name", "Alpha");
            registry.Register(twin);

            registry.GetById(twin.Id).Should().NotBeNull();
            registry.Search("Alpha").Should().ContainSingle();
        }
    }
}
