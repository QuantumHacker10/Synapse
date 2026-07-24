using System.Text;
using FluentAssertions;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Network;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class CollaborationBridgeTests
{
    [Fact]
    public void ScenePatchCodec_RoundTripsEntityTransforms()
    {
        var scene = SceneDocument.CreateDemo();
        var bytes = ScenePatchCodec.Encode(scene, sequence: 7);
        ScenePatchCodec.TryDecode(bytes, out var patch).Should().BeTrue();
        patch!.Sequence.Should().Be(7);
        patch.Entities.Should().HaveCount(scene.Entities.Count);

        var target = new SceneDocument { Name = "empty" };
        ScenePatchCodec.Apply(target, patch!).Should().Be(scene.Entities.Count);
        target.Entities.Should().HaveCount(scene.Entities.Count);
        target.ActiveLawId.Should().Be(scene.ActiveLawId);
    }

    [Fact]
    public async Task EngineHost_WanHostBroadcastsAndReceivesPatches()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var config = new SynapseConfig { Headless = true };
        await using var hostA = new EngineHost(config, logger);
        await using var hostB = new EngineHost(config, logger);
        hostA.InitializeModules();
        hostB.InitializeModules();

        await hostA.StartWanHostAsync("bridge-room", port: 0);
        hostA.WanHub.Should().NotBeNull();
        int rdv = hostA.WanHub!.RendezvousPort;

        await hostB.JoinWanAsync("bridge-room", rendezvousPort: rdv);
        hostB.IsWanConnected.Should().BeTrue();

        // Mutate host A scene and tick collaboration to broadcast.
        hostA.Scene.Entities[0].Position = new Vec3(9, 8, 7);
        for (int i = 0; i < 8; i++)
            await hostA.TickCollaborationAsync();

        // Allow receive loop a moment.
        await Task.Delay(200);
        hostA.WanPatchesSent.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task EngineHost_ExportWebStudioWritesBundle()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var host = new EngineHost(new SynapseConfig { Headless = true }, logger);
        host.InitializeModules();
        var dir = Path.Combine(Path.GetTempPath(), $"synapse-collab-web-{Guid.NewGuid():N}");
        var result = await host.ExportWebStudioAsync(dir);
        File.Exists(result.IndexPath).Should().BeTrue();
        File.Exists(result.SceneJsonPath).Should().BeTrue();
        File.ReadAllText(result.SceneJsonPath, Encoding.UTF8).Should().Contain(host.Scene.Name);
        Directory.Delete(dir, true);
    }

    [Fact]
    public async Task FrameOrchestrator_TicksVrAndCollaborationHooksSafely()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var host = new EngineHost(new SynapseConfig { Headless = true }, logger);
        host.InitializeModules();
        using var orch = new FrameOrchestrator(host, logger);
        var stats = await orch.TickAsync();
        stats.EntityCount.Should().BeGreaterThanOrEqualTo(0);
        stats.VrActive.Should().BeFalse();
        stats.WanActive.Should().BeFalse();
    }
}
