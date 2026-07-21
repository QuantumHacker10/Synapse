using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

/// <summary>CI guards: shipped samples load and tick without a GPU.</summary>
public sealed class SampleSceneHardeningTests
{
    [Theory]
    [InlineData("samples/demo.synapse", "Demo Scene", 4)]
    [InlineData("samples/lab-heat-agents.synapse", "Lab — Heat + Agents", 6)]
    public async Task ShippedSample_LoadsAndTicksHeadless(string relativePath, string expectedName, int minEntities)
    {
        var path = ResolveSamplePath(relativePath);
        File.Exists(path).Should().BeTrue($"missing sample {relativePath}");

        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var host = new EngineHost(new SynapseConfig { Headless = true, SimulationSeed = 42 }, logger);
        await host.LoadSceneAsync(path);

        host.Scene.Name.Should().Be(expectedName);
        host.Scene.Entities.Count.Should().BeGreaterThanOrEqualTo(minEntities);
        host.ActiveLawId.Should().Be("heat_equation");
        host.EntityCount.Should().BeGreaterThan(0);

        host.TickPhysics(1f / 60f);
        await host.TickSimulationAsync(1f / 60f, CancellationToken.None);
        host.EntityCount.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task LabHeatAgents_PersistsJointAndLawExpression()
    {
        var path = ResolveSamplePath("samples/lab-heat-agents.synapse");
        var doc = await SceneDocument.LoadAsync(path);

        doc.ActiveLawExpression.Should().Contain("laplacian");
        doc.Joints.Should().ContainSingle(j => j.Type.Equals("Hinge", StringComparison.OrdinalIgnoreCase));
        doc.Entities.Count(e => e.Type.Equals("Agent", StringComparison.OrdinalIgnoreCase)).Should().Be(2);
        doc.Entities.Should().Contain(e => e.Type.Equals("Genome", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task LabHeatAgents_RoundTripsThroughEngineHost()
    {
        var path = ResolveSamplePath("samples/lab-heat-agents.synapse");
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var host = new EngineHost(new SynapseConfig { Headless = true, SimulationSeed = 7 }, logger);
        await host.LoadSceneAsync(path);

        var outPath = Path.Combine(Path.GetTempPath(), $"synapse-lab-rt-{Guid.NewGuid():N}.synapse");
        try
        {
            await host.SaveSceneAsync(outPath);
            var reloaded = await SceneDocument.LoadAsync(outPath);
            reloaded.Entities.Should().HaveCount(host.Scene.Entities.Count);
            reloaded.Joints.Should().HaveCount(host.Scene.Joints.Count);
            reloaded.ActiveLawId.Should().Be("heat_equation");
        }
        finally
        {
            if (File.Exists(outPath))
                File.Delete(outPath);
        }
    }

    private static string ResolveSamplePath(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }

        return Path.GetFullPath(relative);
    }
}
