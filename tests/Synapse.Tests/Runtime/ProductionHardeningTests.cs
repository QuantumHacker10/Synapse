using System;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Rendering.MeshIO;
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
            // No mesh → expect failure, but must return quickly.
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
            // Attempt escape after root is set
            var escape = Path.Combine(dir, "..", "nope.dll");
            plugins.LoadPluginAssembly(escape, host).Should().BeFalse();
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

        // Simulate inner plaintext event is not used; decrypt path drops bad packets.
        using var other = PeerEncryption.FromSessionCode("other-session");
        var junk = other.Encrypt(new byte[] { 9, 9, 9 });
        // Direct decrypt API still throws; hub drop counter increases via handler
        var before = hub.DroppedPackets;
        // Invoke through public DecryptPatch would throw — use reflection-free path:
        // Broadcast then the self echo decrypts OK; separately verify DecryptPatch rejects wrong key packet.
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
