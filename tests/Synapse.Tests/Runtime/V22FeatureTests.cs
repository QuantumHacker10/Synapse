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
    public async Task OpenXrSwapchain_AcquireReleaseCycle_SupportsSyntheticAndNativeHandles()
    {
        var swap = new OpenXrVulkanSwapchain(3, 1280, 720);
        swap.UsesSyntheticImageHandles.Should().BeTrue();
        swap.TryAcquire(out var idx).Should().BeTrue();
        idx.Should().BeInRange(0, 2);
        swap.PrepareSubmit(0, 0).ImageHandle.Should().BeGreaterThan(0);
        swap.Release();
        swap.IsAcquired.Should().BeFalse();

        using var native = OpenXrVulkanSwapchain.FromNative(2, 64, 64, [0xABCDEF00UL, 0xABCDEF01UL]);
        native.UsesSyntheticImageHandles.Should().BeFalse();
        native.VulkanImageHandles[0].Should().Be(0xABCDEF00UL);
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
        endpoint!.TcpPort.Should().Be(host.ListenPort);
    }

    [Fact]
    public void StunClient_ParsesXorMappedAddress()
    {
        // Craft a minimal Binding Success with XOR-MAPPED-ADDRESS for 203.0.113.5:54320
        Span<byte> txn = stackalloc byte[12];
        txn.Clear();
        txn[0] = 0x11;
        txn[1] = 0x22;

        const uint magic = 0x2112A442;
        ushort port = 54320;
        uint ip = (203u << 24) | (0u << 16) | (113u << 8) | 5u;

        var packet = new byte[32];
        // header
        packet[0] = 0x01;
        packet[1] = 0x01; // success
        packet[2] = 0x00;
        packet[3] = 12; // attr length
        packet[4] = 0x21;
        packet[5] = 0x12;
        packet[6] = 0xA4;
        packet[7] = 0x42;
        txn.CopyTo(packet.AsSpan(8, 12));
        // XOR-MAPPED-ADDRESS attr
        packet[20] = 0x00;
        packet[21] = 0x20;
        packet[22] = 0x00;
        packet[23] = 0x08;
        packet[24] = 0x00;
        packet[25] = 0x01; // IPv4
        ushort xport = (ushort)(port ^ (magic >> 16));
        packet[26] = (byte)(xport >> 8);
        packet[27] = (byte)xport;
        uint xip = ip ^ magic;
        packet[28] = (byte)(xip >> 24);
        packet[29] = (byte)(xip >> 16);
        packet[30] = (byte)(xip >> 8);
        packet[31] = (byte)xip;

        StunClient.TryParseMappedAddress(packet, txn, out var mapped).Should().BeTrue();
        mapped!.Address.ToString().Should().Be("203.0.113.5");
        mapped.Port.Should().Be(54320);
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
    public async Task WasmStudioPublisher_WritesSceneBundle()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"synapse-wasm-{Guid.NewGuid():N}");
        var result = await WasmStudioPublisher.PublishAsync(
            dir,
            "DemoWASM",
            "heat_equation",
            3,
            sceneJson: """{"name":"DemoWASM","entities":[]}""",
            webStudioProjectDirectory: Path.Combine(Path.GetTempPath(), "synapse-no-web-studio"));
        File.Exists(result.SceneJsonPath).Should().BeTrue();
        File.Exists(result.IndexPath).Should().BeTrue();
        result.UsedDotnetPublish.Should().BeFalse();
        Directory.Delete(dir, true);
    }

    [Fact]
    public async Task OpenXrSession_FailClosedWithoutSimulateFlagWhenNoRuntime()
    {
        var previous = Environment.GetEnvironmentVariable("SYNAPSE_VR_SIMULATE");
        try
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_SIMULATE", null);
            using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
            var session = VrSessionFactory.Create(logger);
            var ok = await session.TryInitializeAsync();
            if (ok)
            {
                // Native OpenXR runtime present (e.g. Monado headless) — accept real path.
                session.UsesNativeOpenXr.Should().BeTrue();
                session.Swapchain.Should().NotBeNull();
                session.Swapchain!.UsesSyntheticImageHandles.Should().BeFalse();
            }
            else
            {
                session.IsAvailable.Should().BeFalse();
                session.Swapchain.Should().BeNull();
                session.UsesNativeOpenXr.Should().BeFalse();
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_SIMULATE", previous);
        }
    }

    [Fact]
    public async Task OpenXrSession_SimulateModeExposesSyntheticSwapchain()
    {
        var previous = Environment.GetEnvironmentVariable("SYNAPSE_VR_SIMULATE");
        try
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_SIMULATE", "1");
            using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
            var session = VrSessionFactory.Create(logger);
            // Force simulate by skipping native if somehow available: still OK if native wins first.
            (await session.TryInitializeAsync(64, 64)).Should().BeTrue();
            session.IsAvailable.Should().BeTrue();
            session.Swapchain.Should().NotBeNull();
            if (!session.UsesNativeOpenXr)
                session.Swapchain!.UsesSyntheticImageHandles.Should().BeTrue();
        }
        finally
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_SIMULATE", previous);
        }
    }
}
