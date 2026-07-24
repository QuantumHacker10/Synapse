using System;
using System.IO;
using System.Numerics;
using FluentAssertions;
using GDNN.Rendering.MeshIO;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class UsdProductionReadyTests
{
    [Fact]
    public void ProductionSmoke_EmbeddedStagePasses()
    {
        UsdProductionSmoke.TryVerify(out var detail).Should().BeTrue(detail);
    }

    [Fact]
    public void FaceVertexCounts_TriangulatesQuad()
    {
        var path = Resolve("samples/meshes/production_dcc.usda");
        var result = new UsdAsciiLoader().LoadLeafMeshAsync(path, null, default).GetAwaiter().GetResult();
        result.Success.Should().BeTrue(result.ErrorMessage);
        // Quad + Tri; ProxyCage skipped by purpose mask
        result.Asset!.Primitives.Should().HaveCount(2);
        var quad = result.Asset.Primitives.Should().ContainSingle(p => p.Name == "Quad").Subject;
        quad.Indices.Should().HaveCount(6); // one quad → 2 tris
        quad.Vertices[0].Normal.Y.Should().BeApproximately(1f, 0.01f);
        quad.Bounds.Min.X.Should().BeApproximately(-1f, 0.01f);

        var mat = result.Asset.Materials.Should().ContainSingle(m => m.Name == "PbrMat").Subject;
        mat.DoubleSided.Should().BeTrue();
        mat.AlphaCutoff.Should().BeApproximately(0.1f, 0.001f);
        mat.EmissiveIntensity.Should().BeApproximately(1.5f, 0.001f);
    }

    [Fact]
    public void PurposeMask_CanIncludeProxy()
    {
        var path = Resolve("samples/meshes/production_dcc.usda");
        var cfg = new MeshLoadConfig { UsdPurposeMask = UsdPurposeMask.All };
        var result = new UsdAsciiLoader().LoadLeafMeshAsync(path, cfg, default).GetAwaiter().GetResult();
        result.Success.Should().BeTrue(result.ErrorMessage);
        result.Asset!.Primitives.Should().Contain(p => p.Name == "ProxyCage");
    }

    [Fact]
    public void LegacySentinelFaces_StillLoad()
    {
        var path = Resolve("samples/meshes/tetra.usda");
        var result = new UsdAsciiLoader().LoadLeafMeshAsync(path, null, default).GetAwaiter().GetResult();
        result.Success.Should().BeTrue(result.ErrorMessage);
        result.Asset!.Primitives.Should().NotBeEmpty();
        result.Asset.TotalTriangleCount.Should().BeGreaterThan(0);
    }

    [Fact]
    public void BlendShape_AppliesNormalDeltas()
    {
        var shape = new MeshBlendShape { Name = "n" };
        shape.DeltaPositions.Add(new Vector3(0, 0.5f, 0));
        shape.DeltaNormals.Add(new Vector3(0, 0, 1));
        var verts = new System.Collections.Generic.List<MeshVertex>
        {
            new() { Position = Vector3.Zero, Normal = Vector3.UnitY }
        };
        shape.Apply(verts, 1f);
        verts[0].Position.Y.Should().BeApproximately(0.5f, 0.001f);
        verts[0].Normal.Z.Should().BeGreaterThan(0.1f);
    }

    private static string Resolve(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }

        throw new FileNotFoundException(relative);
    }
}
