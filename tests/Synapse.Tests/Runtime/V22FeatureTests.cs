using System.Security.Cryptography;
using System.Text;
using FluentAssertions;
using Synapse.Infrastructure.Logging;
using Synapse.Network;
using Synapse.VR;
using Synapse.Web;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class V22FeatureTests
{
    [Fact]
    public void PeerEncryption_RoundTripsPayload()
    {
        using var enc = PeerEncryption.FromSessionCode("synapse-demo-room");
        var plain = Encoding.UTF8.GetBytes("scene-patch-v2");
        var cipher = enc.Encrypt(plain);
        enc.Decrypt(cipher).Should().Equal(plain);
    }

    [Fact]
    public async Task OpenXrSwapchain_AcquireReleaseCycle_IsSyntheticScaffold()
    {
        var swap = new OpenXrVulkanSwapchain(3, 1280, 720);
        swap.UsesSyntheticImageHandles.Should().BeTrue("v2.2 OpenXR path is experimental scaffolding");
        swap.TryAcquire(out var idx).Should().BeTrue();
        idx.Should().BeInRange(0, 2);
        swap.PrepareSubmit(0, 0).ImageHandle.Should().BeGreaterThan(0);
        swap.Release();
        swap.IsAcquired.Should().BeFalse();
    }

    [Fact]
    public async Task WanPeerHub_HostRegistersAndJoinDiscoversPort()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var host = new WanSimulationPeerHub(logger, "test-session-discover");
        await host.StartHostAsync(port: 0);
        host.ListenPort.Should().BeGreaterThan(0);
        host.RendezvousPort.Should().BeGreaterThan(0);

        using var clientNat = new NatTraversalCoordinator(logger, "test-session-discover", host.RendezvousPort);
        var endpoint = await clientNat.DiscoverPeerAsync();
        endpoint.Should().NotBeNull();
        endpoint!.Port.Should().Be(host.ListenPort);
    }

    [Fact]
    public async Task WebEditorBundle_WritesV22Site()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"synapse-web22-{Guid.NewGuid():N}");
        var bundle = WebEditorBuilder.FromScene("Demo", "demo.gltf", "heat_equation", 2);
        await WebEditorBuilder.WriteSiteAsync(dir, bundle);
        File.ReadAllText(Path.Combine(dir, "index.html")).Should().Contain("2.2");
        Directory.Delete(dir, true);
    }

    [Fact]
    public void OpenXrSession_FailClosedWithoutSimulateFlag()
    {
        var previous = Environment.GetEnvironmentVariable("SYNAPSE_VR_SIMULATE");
        try
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_SIMULATE", null);
            using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
            var session = VrSessionFactory.Create(logger);
            session.TryInitializeAsync().GetAwaiter().GetResult().Should().BeFalse();
            session.IsAvailable.Should().BeFalse();
            session.Swapchain.Should().BeNull();
        }
        finally
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_SIMULATE", previous);
        }
    }

    [Fact]
    public void OpenXrSession_SimulateModeExposesSyntheticSwapchain()
    {
        var previous = Environment.GetEnvironmentVariable("SYNAPSE_VR_SIMULATE");
        try
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_SIMULATE", "1");
            using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
            var session = VrSessionFactory.Create(logger);
            session.TryInitializeAsync(64, 64).GetAwaiter().GetResult().Should().BeTrue();
            session.IsAvailable.Should().BeTrue();
            session.Swapchain.Should().NotBeNull();
            session.Swapchain!.UsesSyntheticImageHandles.Should().BeTrue();
        }
        finally
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_SIMULATE", previous);
        }
    }
}
