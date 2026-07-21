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
    public async Task WanPeerHub_EncryptedHostStarts()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var hub = new WanSimulationPeerHub(logger, "test-session-v22");
        await hub.StartHostAsync(port: 0);
        hub.ListenPort.Should().BeGreaterThan(0);
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
    public void OpenXrSession_ExposesSwapchainWhenLoaderPresent()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var session = VrSessionFactory.Create(logger);
        session.TryInitializeAsync().GetAwaiter().GetResult();
        if (session.IsAvailable)
            session.Swapchain.Should().NotBeNull();
    }
}
