using System;
using System.IO;
using FluentAssertions;
using GDNN.Rendering.MeshIO;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class UsdAdvancedTexturesTests
{
    [Fact]
    public void UsdTextures_MapsPbrChannelsAndUvs()
    {
        var path = Resolve("samples/meshes/textured_pbr.usda");
        var result = new UsdAsciiLoader().LoadLeafMeshAsync(path, null, default).GetAwaiter().GetResult();
        result.Success.Should().BeTrue(result.ErrorMessage);
        result.Asset!.Materials.Should().ContainSingle(m => m.Name == "PbrMat");
        var mat = result.Asset.Materials[0];
        mat.AlbedoTexturePath.Should().EndWith(Path.Combine("textures", "albedo.png"));
        mat.NormalTexturePath.Should().EndWith(Path.Combine("textures", "normal.png"));
        mat.MetallicRoughnessTexturePath.Should().EndWith(Path.Combine("textures", "orm.png"));
        mat.AOTexturePath.Should().EndWith(Path.Combine("textures", "orm.png"));
        mat.EmissiveTexturePath.Should().EndWith(Path.Combine("textures", "emissive.png"));
        mat.HeightTexturePath.Should().EndWith(Path.Combine("textures", "height.png"));
        mat.Clearcoat.Should().BeApproximately(0.25f, 0.001f);
        mat.Ior.Should().BeApproximately(1.5f, 0.001f);

        result.Asset.Primitives[0].ActiveAttributes.Should().HaveFlag(VertexAttribute.TexCoord0);
        result.Asset.Primitives[0].Vertices[1].TexCoord0.X.Should().BeApproximately(1f, 0.001f);
    }

    [Fact]
    public void UsdTexture_ResolveRelativePath()
    {
        var resolved = UsdMaterialParser.ResolveTexturePath("./textures/albedo.png", "/tmp/stage");
        resolved.Should().Be(Path.GetFullPath(Path.Combine("/tmp/stage", "textures", "albedo.png")));
    }

    [Fact]
    public void UsdTexture_ExtractFileFromUvShader()
    {
        const string body = """
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @./foo/bar.png@
            """;
        UsdMaterialParser.ExtractUvTextureFile(body).Should().Be("./foo/bar.png");
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
