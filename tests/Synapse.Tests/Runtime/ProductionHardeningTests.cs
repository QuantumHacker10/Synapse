using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Core.NeuralNetwork;
using GDNN.Memory;
using GDNN.Rendering.MeshIO;
using GDNN.Streaming;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Network;
using Synapse.Plugins;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class ProductionHardeningTests
{
    [Fact]
    public async Task UsdComposition_ResolvesUsdaReference()
    {
        var root = ResolveSample("samples/meshes/composed_root.usda");
        File.Exists(root).Should().BeTrue();
        var result = await new MeshLoader().LoadAsync(root);
        result.Success.Should().BeTrue(result.ErrorMessage);
        result.Asset!.Primitives.Should().NotBeEmpty();
        result.Asset.Primitives[0].Vertices.Count.Should().BeGreaterThanOrEqualTo(4);
    }

    [Fact]
    public async Task UsdComposition_ResolvesUsdcReference()
    {
        var root = ResolveSample("samples/meshes/composed_usdc.usda");
        var result = await new UsdAsciiLoader().LoadAsync(root);
        result.Success.Should().BeTrue(result.ErrorMessage);
        result.Asset!.Primitives.Should().NotBeEmpty();
    }

    [Fact]
    public void UsdComposition_DetectsCycleWithoutHang()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"usd-cycle-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var a = Path.Combine(dir, "a.usda");
            var b = Path.Combine(dir, "b.usda");
            File.WriteAllText(a, "#usda 1.0\ndef Xform \"A\" { prepend references = @./b.usda@ }\n");
            File.WriteAllText(b, "#usda 1.0\ndef Xform \"B\" { prepend references = @./a.usda@ }\n");
            var result = new UsdAsciiLoader().LoadAsync(a).GetAwaiter().GetResult();
            result.Success.Should().BeFalse();
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void PluginHost_RejectsPathTraversalAndRemote()
    {
        PluginHost.IsBlockedRemotePath(@"\\server\share\evil.dll").Should().BeTrue();
        PluginHost.IsBlockedRemotePath("https://evil.example/p.dll").Should().BeTrue();
        var root = Path.GetTempPath();
        var outside = Path.GetFullPath(Path.Combine(root, "..", "outside.dll"));
        PluginHost.IsUnderRoot(outside, root).Should().BeFalse();
    }

    [Fact]
    public void PluginHost_LoadOutsideRoot_Fails()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        using var plugins = new PluginHost(logger);
        var host = new EngineHost(new SynapseConfig(), logger);
        var dir = Path.Combine(Path.GetTempPath(), $"plugroot-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            plugins.LoadFromDirectory(dir, host).Should().Be(0);
            var escape = Path.Combine(dir, "..", "nope.dll");
            plugins.LoadPluginAssembly(escape, host).Should().BeFalse();
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void PluginHost_RequireManifest_RefusesBareDll()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        using var hostPlugins = new PluginHost(logger, PluginTrustMode.RequireManifest);
        var dir = Path.Combine(Path.GetTempPath(), $"synapse-plugin-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        var dll = Path.Combine(dir, "fake.dll");
        File.WriteAllBytes(dll, Encoding.UTF8.GetBytes("not-a-real-assembly"));
        try
        {
            var engine = new EngineHost(new SynapseConfig(), logger);
            hostPlugins.LoadPluginAssembly(dll, engine).Should().BeFalse();
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void PeerEncryption_AuthToken_AndTamperDetect()
    {
        using var enc = PeerEncryption.FromSessionCode("prod-room-01");
        var nonce = new byte[16];
        Random.Shared.NextBytes(nonce);
        var token = enc.ComputeAuthToken(nonce, "client");
        enc.VerifyAuthToken(nonce, "client", token).Should().BeTrue();
        enc.VerifyAuthToken(nonce, "host", token).Should().BeFalse();

        var cipher = enc.Encrypt(new byte[] { 1, 2, 3, 4 });
        cipher[20] ^= 0xFF;
        var act = () => enc.Decrypt(cipher);
        act.Should().Throw<System.Security.Cryptography.CryptographicException>();
    }

    [Fact]
    public async Task WanHub_DropsTamperedPacket()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var hub = new WanSimulationPeerHub(logger, "prod-wan-drop");
        await hub.StartHostAsync(0);
        hub.ListenPort.Should().BeGreaterThan(0);

        using var other = PeerEncryption.FromSessionCode("other-session");
        var junk = other.Encrypt(new byte[] { 9, 9, 9 });
        var before = hub.DroppedPackets;
        var act = () => hub.DecryptPatch(junk);
        act.Should().Throw<System.Security.Cryptography.CryptographicException>();
        hub.DroppedPackets.Should().Be(before);
    }

    [Fact]
    public async Task MultiPeer_PublicBindWithoutAuth_Throws()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        await using var hub = SimulationPeerHub.CreateMultiPeerHub(logger);
        var act = async () => await hub.StartHostAsync(0, publicBind: true);
        await act.Should().ThrowAsync<InvalidOperationException>();
    }

    [Fact]
    public async Task AuthenticatedHub_ConnectsLocally()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        using var auth = PeerEncryption.FromSessionCode("auth-room-99");
        await using var host = SimulationPeerHub.CreateAuthenticatedHub(logger, auth);
        await host.StartHostAsync(0, publicBind: false);
        await using var client = SimulationPeerHub.CreateAuthenticatedHub(logger, auth);
        await client.ConnectAsync("127.0.0.1", host.ListenPort);
        await Task.Delay(80);
        host.PeerCount.Should().BeGreaterOrEqualTo(2);
    }

    [Fact]
    public void BlueprintCompiler_SelectorAndWaitKinds()
    {
        var entry = new BlueprintNode { Kind = BlueprintNodeKind.Entry, Title = "E", Outputs = { new BlueprintPin() } };
        var sel = new BlueprintNode
        {
            Kind = BlueprintNodeKind.Selector,
            Title = "Sel",
            Inputs = { new BlueprintPin { IsInput = true } },
            Outputs = { new BlueprintPin() }
        };
        var wait = new BlueprintNode
        {
            Kind = BlueprintNodeKind.Wait,
            Title = "W",
            Payload = "0.01",
            Inputs = { new BlueprintPin { IsInput = true } },
            Outputs = { new BlueprintPin() }
        };
        var law = new BlueprintNode
        {
            Kind = BlueprintNodeKind.LawApply,
            Title = "L",
            Payload = "heat_equation",
            Inputs = { new BlueprintPin { IsInput = true } },
            Outputs = { new BlueprintPin() }
        };
        var exit = new BlueprintNode { Kind = BlueprintNodeKind.Exit, Title = "X", Inputs = { new BlueprintPin { IsInput = true } } };
        var doc = new BlueprintDocument
        {
            Name = "ProdKinds",
            Nodes = { entry, sel, wait, law, exit },
            Edges =
            {
                new BlueprintEdge { FromNodeId = entry.Id, ToNodeId = sel.Id },
                new BlueprintEdge { FromNodeId = sel.Id, ToNodeId = wait.Id },
                new BlueprintEdge { FromNodeId = wait.Id, ToNodeId = law.Id },
                new BlueprintEdge { FromNodeId = law.Id, ToNodeId = exit.Id }
            }
        };
        BlueprintCompiler.Compile(doc).Should().NotBeNull();
    }

    [Fact]
    public void ZeroCopyBuffer_CreateFromFile_MapsAndRoundTrips()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-zcb-{Guid.NewGuid():N}.bin");
        try
        {
            using (var buffer = ZeroCopyBuffer.CreateFromFile(path, capacity: 64))
            {
                buffer.Capacity.Should().Be(64);
                buffer.GetSpan<byte>(0, 64)[0] = 0xAB;
                buffer.GetSpan<byte>(0, 64)[63] = 0xCD;
            }

            using var reopen = ZeroCopyBuffer.CreateFromFile(path, capacity: 64);
            reopen.GetSpan<byte>(0, 64)[0].Should().Be(0xAB);
            reopen.GetSpan<byte>(0, 64)[63].Should().Be(0xCD);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public void MappedBuffer_KeepsPointerAliveAcrossOpen()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-mapped-{Guid.NewGuid():N}.bin");
        try
        {
            using var mapped = new MappedBuffer(path, 32);
            mapped.Open();
            mapped.IsOpen.Should().BeTrue();
            mapped.GetSpan<byte>(0, 32)[0] = 42;
            mapped.GetSpan<byte>(0, 32)[0].Should().Be(42);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public async Task AssetStreamer_FailsClosedWhenAssetMissing()
    {
        var previousRoot = AssetStreamer.AssetRootDirectory;
        var previousSynth = AssetStreamer.AllowSyntheticPlaceholders;
        var root = Path.Combine(Path.GetTempPath(), $"synapse-assets-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        AssetStreamer? streamer = null;
        try
        {
            AssetStreamer.AssetRootDirectory = root;
            AssetStreamer.AllowSyntheticPlaceholders = false;
            streamer = new AssetStreamer(new StreamerConfig
            {
                MaxConcurrentDownloads = 1,
                MemoryBudgetBytes = 16 * 1024 * 1024
            });

            await streamer.RequestAssetAsync("missing-asset-id", AssetPriority.Immediate);

            AssetEntry? entry = null;
            for (int i = 0; i < 50; i++)
            {
                entry = streamer.GetAssetState("missing-asset-id");
                if (entry?.State is AssetLoadingState.Failed or AssetLoadingState.Loaded)
                    break;
                await Task.Delay(50);
            }

            entry.Should().NotBeNull();
            entry!.State.Should().Be(AssetLoadingState.Failed);
            entry.LastError.Should().NotBeNullOrWhiteSpace();
        }
        finally
        {
            streamer?.Dispose();
            AssetStreamer.AssetRootDirectory = previousRoot;
            AssetStreamer.AllowSyntheticPlaceholders = previousSynth;
            if (Directory.Exists(root))
                Directory.Delete(root, true);
        }
    }

    [Fact]
    public void LawMarketplace_RejectsOversizedExpression()
    {
        var package = new SynapseLawPackage
        {
            Id = "bad",
            Name = "Bad",
            Expression = new string('x', LawMarketplace.MaxExpressionLength + 1)
        };
        var act = () => LawMarketplace.ValidatePackage(package);
        act.Should().Throw<InvalidDataException>();
    }

    [Fact]
    public async Task GpuUploadStage_RejectsEmptyAsset()
    {
        var stage = StreamingPipelineStages.CreateGpuUploadStage();
        var asset = new NeuralAsset();
        var act = async () => await stage.ProcessAsync(asset, default);
        await act.Should().ThrowAsync<InvalidDataException>();
    }

    [Fact]
    public async Task GpuUploadStage_PreparesValidWeights()
    {
        var stage = StreamingPipelineStages.CreateGpuUploadStage();
        var mlp = new MicroMLP();
        var asset = new NeuralAsset
        {
            CompressedWeights = mlp.CompressWeights()
        };

        var prepared = await stage.ProcessAsync(asset, default);
        prepared.IsGpuUploadPrepared.Should().BeTrue();
        prepared.Metadata.ContentHash.Should().NotBeNullOrWhiteSpace();
    }

    private static string ResolveSample(string relative)
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
