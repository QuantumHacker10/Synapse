using System;
using System.IO;
using System.Net;
using System.Numerics;
using System.Threading.Tasks;
using FluentAssertions;
using GDNN.Rendering.MeshIO;
using Synapse.Infrastructure.Logging;
using Synapse.Network;
using Synapse.VR;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class ProductionVrWanUsdTests
{
    [Fact]
    public async Task OpenXr_SimulatedPath_IsProductionLabeled()
    {
        Environment.SetEnvironmentVariable("SYNAPSE_VR_FORCE_SIMULATED", "1");
        try
        {
            using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
            await using var session = VrSessionFactory.Create(logger);
            (await session.TryInitializeAsync(640, 480)).Should().BeTrue();
            session.IsAvailable.Should().BeTrue();
            session.IsSimulated.Should().BeTrue();
            session.Swapchain.Should().NotBeNull();
            session.Swapchain!.IsSimulated.Should().BeTrue();
            session.Swapchain.TryAcquire(out var idx).Should().BeTrue();
            idx.Should().BeInRange(0, session.Swapchain.ImageCount - 1);
            session.Swapchain.Release();
        }
        finally
        {
            Environment.SetEnvironmentVariable("SYNAPSE_VR_FORCE_SIMULATED", null);
        }
    }

    [Fact]
    public void NatPeerReply_ParsesIpAndPort()
    {
        var ep = NatTraversalCoordinator.ParsePeerReply("PEER|room42|10.0.0.5|7777", "room42");
        ep.Should().NotBeNull();
        ep!.Address.Should().Be(IPAddress.Parse("10.0.0.5"));
        ep.Port.Should().Be(7777);

        NatTraversalCoordinator.ParsePeerReply("PEER|room42|0.0.0.0|0", "room42").Should().BeNull();
        NatTraversalCoordinator.ParsePeerReply("PEER|room42|9000", "room42")!
            .Address.Should().Be(IPAddress.Loopback);
    }

    [Fact]
    public async Task WanNat_DiscoverReturnsObservedEndpoint()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var code = "wan-prod-" + Guid.NewGuid().ToString("N")[..8];
        int rvPort = 18000 + Random.Shared.Next(1000, 2000);

        using var relay = NatTraversalCoordinator.StartRelay(logger, code, rvPort, IPAddress.Any);
        using var hostNat = new NatTraversalCoordinator(logger, code, rvPort, IPAddress.Loopback);
        await hostNat.RegisterPublicEndpointAsync(tcpPort: 5555);

        using var clientNat = new NatTraversalCoordinator(logger, code, rvPort, IPAddress.Loopback);
        var ep = await clientNat.DiscoverPeerAsync();
        ep.Should().NotBeNull();
        ep!.Port.Should().Be(5555);
        // Same-host UDP source is loopback (or a local interface mapped to the relay).
        ep.Address.Should().NotBe(IPAddress.Any);
    }

    [Fact]
    public void UsdXform_ParsesTranslateRotateScaleOrder()
    {
        const string usda = """
            double3 xformOp:translate = (10, 0, 0)
            float3 xformOp:rotateXYZ = (0, 0, 90)
            float3 xformOp:scale = (2, 2, 2)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
            """;
        var m = UsdXform.ParseLocalMatrix(usda);
        var p = UsdXform.TransformPoint(new Vector3(1, 0, 0), m);
        // scale → rotateZ90 → translate: (1,0,0)*S=(2,0,0); *Rz90≈(0,2,0); +T≈(10,2,0)
        p.X.Should().BeApproximately(10f, 0.05f);
        p.Y.Should().BeApproximately(2f, 0.05f);
        p.Z.Should().BeApproximately(0f, 0.05f);
    }

    [Fact]
    public async Task UsdComposition_AppliesParentXformStack()
    {
        var root = Path.GetFullPath(Path.Combine(FindRepoRoot(), "samples", "meshes", "composed_xform.usda"));
        File.Exists(root).Should().BeTrue();
        var result = await new UsdAsciiLoader().LoadAsync(root);
        result.Success.Should().BeTrue(result.ErrorMessage);
        result.Asset!.Primitives.Should().NotBeEmpty();
        // Root translate x=5 should move composed mesh away from origin.
        var minX = float.MaxValue;
        foreach (var prim in result.Asset.Primitives)
        {
            foreach (var v in prim.Vertices)
                minX = Math.Min(minX, v.Position.X);
        }

        minX.Should().BeGreaterThan(1f);
    }

    [Fact]
    public void UsdComposition_ExtractsPrimPathTarget()
    {
        var refs = UsdCompositionResolver.ExtractCompositionRefs(
            @"prepend references = @./tetra.usda@</Cube>");
        refs.Should().ContainSingle();
        refs[0].AssetPath.Should().Be("./tetra.usda");
        refs[0].PrimPath.Should().Be("/Cube");
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Synapse.slnx")))
                return dir.FullName;
            dir = dir.Parent;
        }

        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
    }
}
