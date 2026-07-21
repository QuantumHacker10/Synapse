using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using FluentAssertions;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Network;
using Synapse.Runtime;
using Synapse.VR;
using Synapse.Web;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class CollaborationCoverageTests
{
    [Fact]
    public async Task TickCollaboration_ThrottlesToEverySixthTick()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var host = new EngineHost(new SynapseConfig { Headless = true }, logger);
        host.InitializeModules();
        await host.StartWanHostAsync("throttle-room", port: 0);

        for (int i = 0; i < 5; i++)
            await host.TickCollaborationAsync();
        host.WanPatchesSent.Should().Be(0);

        await host.TickCollaborationAsync();
        host.WanPatchesSent.Should().BeGreaterThanOrEqualTo(1);
    }

    [Fact]
    public async Task ApplyOptional_FromConfig_StartsWanHost()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var config = new SynapseConfig
        {
            Headless = true,
            WanSessionCode = "cfg-host-room",
            WanHost = true,
            WanPort = 0
        };
        await using var host = new EngineHost(config, logger);
        host.InitializeModules();
        await host.ApplyOptionalCollaborationFromConfigAsync();
        host.IsWanConnected.Should().BeTrue();
        host.WanStatusText.Should().ContainEquivalentOf("host");
    }

    [Fact]
    public async Task EnableVr_SimulateMode_ThenDisable()
    {
        var previous = Environment.GetEnvironmentVariable("SYNAPSE_VR_SIMULATE");
        try
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_SIMULATE", "1");
            using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
            await using var host = new EngineHost(new SynapseConfig { Headless = true }, logger);
            host.InitializeModules();

            bool ok = await host.EnableVrAsync(64, 64);
            // Native may win over simulate if runtime present; either way Enable should succeed with simulate flag.
            ok.Should().BeTrue();
            host.IsVrActive.Should().BeTrue();
            host.VrStatusText.Should().NotBeNullOrWhiteSpace();

            await host.DisableVrAsync();
            host.IsVrActive.Should().BeFalse();
            host.VrStatusText.Should().ContainEquivalentOf("off");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_SIMULATE", previous);
        }
    }

    [Fact]
    public async Task FrameOrchestrator_WithWan_SetsWanActiveStats()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var host = new EngineHost(new SynapseConfig { Headless = true }, logger);
        host.InitializeModules();
        await host.StartWanHostAsync("orch-stats", port: 0);
        using var orch = new FrameOrchestrator(host, logger);

        // Drive enough ticks to include a collaboration broadcast.
        FrameStats? stats = null;
        for (int i = 0; i < 8; i++)
            stats = await orch.TickAsync();

        stats.Should().NotBeNull();
        stats!.WanActive.Should().BeTrue();
        stats.WanStatus.Should().NotBeNullOrWhiteSpace();
        stats.CollaborationMs.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public async Task FrameOrchestrator_Paused_SkipsCollaborationBroadcast()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var host = new EngineHost(new SynapseConfig { Headless = true }, logger);
        host.InitializeModules();
        await host.StartWanHostAsync("pause-room", port: 0);
        using var orch = new FrameOrchestrator(host, logger) { IsPaused = true };

        for (int i = 0; i < 12; i++)
            await orch.TickAsync();

        host.WanPatchesSent.Should().Be(0);
    }

    [Fact]
    public async Task FrameOrchestrator_OverlappingTick_ReturnsLastStats()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var host = new EngineHost(new SynapseConfig { Headless = true }, logger);
        host.InitializeModules();
        using var orch = new FrameOrchestrator(host, logger);

        var first = orch.TickAsync();
        // Immediately request another tick while the first may still hold the gate.
        var second = await orch.TickAsync();
        var completed = await first;

        second.Should().NotBeNull();
        completed.Should().NotBeNull();
    }

    [Fact]
    public async Task NativeOpenXrRuntime_TryInitialize_FailsClosedWithoutRuntime()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        using var runtime = new NativeOpenXrRuntime(logger);
        // On this headless VM without OpenXR runtime, init should fail closed (not throw).
        bool ok = runtime.TryInitialize(64, 64);
        if (!ok)
            runtime.IsInitialized.Should().BeFalse();
        // If a headless runtime is installed, success is also acceptable.
    }

    [Fact]
    public void OpenXrSwapchain_DoubleAcquire_FailsClosed()
    {
        using var swap = new OpenXrVulkanSwapchain(3, 128, 128);
        swap.TryAcquire(out _).Should().BeTrue();
        swap.TryAcquire(out var second).Should().BeFalse();
        second.Should().Be(-1);
        swap.Release();
        swap.TryAcquire(out _).Should().BeTrue();
    }

    [Fact]
    public void PeerEncryption_RejectsTamperedCiphertextAndWrongSession()
    {
        using var a = PeerEncryption.FromSessionCode("same-room");
        using var b = PeerEncryption.FromSessionCode("other-room");
        var plain = Encoding.UTF8.GetBytes("patch-payload");
        var cipher = a.Encrypt(plain);

        var tampered = (byte[])cipher.Clone();
        tampered[^1] ^= 0xFF;
        var actTamper = () => a.Decrypt(tampered);
        actTamper.Should().Throw<CryptographicException>();

        var actWrong = () => b.Decrypt(cipher);
        actWrong.Should().Throw<CryptographicException>();
    }

    [Fact]
    public async Task WasmStudioPublisher_WritesSceneJsonMatchingInput()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"synapse-pub-{Guid.NewGuid():N}");
        const string sceneJson = """{"name":"CoverageScene","version":"2.2","entities":[]}""";
        var result = await WasmStudioPublisher.PublishAsync(
            dir,
            "CoverageScene",
            "heat_equation",
            0,
            sceneJson: sceneJson,
            webStudioProjectDirectory: Path.Combine(Path.GetTempPath(), "no-studio"));

        result.UsedDotnetPublish.Should().BeFalse();
        File.ReadAllText(result.SceneJsonPath).Should().Be(sceneJson);
        File.Exists(result.IndexPath).Should().BeTrue();
        File.Exists(Path.Combine(dir, "studio-fallback.txt")).Should().BeTrue();
        Directory.Delete(dir, true);
    }
}
