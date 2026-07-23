using System;
using System.Net;
using System.Numerics;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Rendering.MeshIO;
using Synapse.Network;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class StunTurnUsdDccTests
{
    [Fact]
    public void Stun_XorMappedAddress_RoundTrips()
    {
        var txn = new byte[12];
        Random.Shared.NextBytes(txn);
        var mapped = new IPEndPoint(IPAddress.Parse("203.0.113.10"), 54320);
        var success = StunClient.BuildBindingSuccess(txn, mapped);
        StunClient.TryParseMappedAddress(success, txn).Should().Be(mapped);

        var req = StunClient.BuildBindingRequest(txn);
        req.Length.Should().BeGreaterThan(20);
        req[0].Should().Be(0x00);
        req[1].Should().Be(0x01);
    }

    [Fact]
    public void Turn_AllocateSuccess_ParsesRelayed()
    {
        var txn = new byte[12];
        Random.Shared.NextBytes(txn);
        var relayed = new IPEndPoint(IPAddress.Parse("198.51.100.2"), 49152);
        var mapped = new IPEndPoint(IPAddress.Parse("203.0.113.5"), 40000);
        var msg = TurnClient.BuildAllocateSuccessForTests(txn, relayed, mapped, lifetime: 600);
        TurnClient.TryParseAllocateSuccess(msg, txn, out var r, out var m, out var life).Should().BeTrue();
        r.Should().Be(relayed);
        m.Should().Be(mapped);
        life.Should().Be(600);
    }

    [Fact]
    public void Turn_DeriveLongTermKey_IsDeterministicMd5()
    {
        var a = TurnClient.DeriveLongTermKey("user", "example.org", "pass");
        var b = TurnClient.DeriveLongTermKey("user", "example.org", "pass");
        a.Should().Equal(b);
        a.Length.Should().Be(16);
    }

    [Fact]
    public void NatIce_ParseHostPort()
    {
        NatIceOptions.ParseHostPort("stun.l.google.com:19302", 3478)
            .Should().Be(("stun.l.google.com", 19302));
        NatIceOptions.ParseHostPort("turn.example.com", 3478)
            .Should().Be(("turn.example.com", 3478));
    }

    [Fact]
    public void PeerCandidate_ParsesMode()
    {
        var c = NatTraversalCoordinator.ParsePeerCandidate("PEER|room|10.0.0.1|7777|stun", "room");
        c.Should().NotBeNull();
        c!.Value.Mode.Should().Be("stun");
        c.Value.Endpoint.Port.Should().Be(7777);
    }

    [Fact]
    public void UsdMaterial_ParsesPreviewSurface()
    {
        var path = Resolve("samples/meshes/tetra_preview_skel.usda");
        var result = new UsdAsciiLoader().LoadLeafMeshAsync(path, null, default).GetAwaiter().GetResult();
        result.Success.Should().BeTrue(result.ErrorMessage);
        result.Asset!.Materials.Should().NotBeEmpty();
        result.Asset.Materials[0].BaseColor.X.Should().BeApproximately(1f, 0.01f);
        result.Asset.Primitives[0].MaterialIndex.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public void UsdSkeleton_ParsesJointsAndWeights()
    {
        var path = Resolve("samples/meshes/tetra_preview_skel.usda");
        var result = new UsdAsciiLoader().LoadLeafMeshAsync(path, null, default).GetAwaiter().GetResult();
        result.Asset!.Skeleton.Should().NotBeNull();
        result.Asset.Skeleton!.JointNames.Should().HaveCount(2);
        result.Asset.Primitives[0].Vertices[0].BoneWeights.X.Should().BeApproximately(1f, 0.01f);
    }

    [Fact]
    public async Task UsdVariants_SelectsHighByConfig()
    {
        var path = Resolve("samples/meshes/variant_modeling.usda");
        var cfg = new MeshLoadConfig
        {
            UsdVariantSelections = { ["modelingVariant"] = "High" }
        };
        var high = await new UsdAsciiLoader().LoadAsync(path, cfg);
        high.Success.Should().BeTrue(high.ErrorMessage);
        high.Asset!.Primitives[0].Vertices.Count.Should().Be(4);

        var lowCfg = new MeshLoadConfig
        {
            UsdVariantSelections = { ["modelingVariant"] = "Low" }
        };
        var low = await new UsdAsciiLoader().LoadAsync(path, lowCfg);
        low.Asset!.Primitives[0].Vertices.Count.Should().Be(3);
    }

    [Fact]
    public void UsdVariantResolver_ListsSets()
    {
        var text = System.IO.File.ReadAllText(Resolve("samples/meshes/variant_modeling.usda"));
        UsdVariantResolver.ListVariantSetNames(text).Should().Contain("modelingVariant");
    }

    private static string Resolve(string relative)
    {
        var dir = new System.IO.DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = System.IO.Path.Combine(dir.FullName, relative);
            if (System.IO.File.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }

        throw new System.IO.FileNotFoundException(relative);
    }
}
