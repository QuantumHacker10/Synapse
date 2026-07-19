using System.Numerics;
using FluentAssertions;
using GDNN.Core.NeuralNetwork;
using GDNN.Evaluation;
using GDNN.Rendering.MeshIO;
using GDNN.Rendering.PostProcess;
using Xunit;

namespace Synapse.Tests.Rendering;

public class MeshToSdfPipelineTests
{
    [Fact]
    public void BuildReferenceMesh_FromSyntheticCube_ProducesTriangles()
    {
        var asset = MeshAsset.CreateUnitCube(1f, "TestCube");
        var mesh = MeshToSdfPipeline.BuildReferenceMesh(asset);

        mesh.TriangleCount.Should().Be(12);
        mesh.VertexCount.Should().Be(8);
        mesh.UnsignedDistance(Vector3.Zero).Should().BeGreaterThan(0f);
    }

    [Fact]
    public void TrainFromAsset_OnSyntheticCube_CompletesWithReport()
    {
        var asset = MeshAsset.CreateUnitCube(1f, "TrainCube");
        var result = MeshToSdfPipeline.TrainFromAsset(
            asset,
            new MeshToSdfOptions
            {
                SampleCount = 1024,
                Epochs = 20,
                RandomSeed = 7
            });

        result.Success.Should().BeTrue(result.ErrorMessage);
        result.Network.Should().NotBeNull();
        result.Report.Should().NotBeNull();
        result.Report!.MeshTriangles.Should().Be(12);
        result.Report.SampleCount.Should().Be(1024);
    }

    [Fact]
    public void CreateNeuralAsset_RoundTripsHashNetwork()
    {
        using var network = new HashEncodedDeepMLP(new Random(3));
        var asset = MeshToSdfPipeline.CreateNeuralAsset(network, MeshAsset.CreateUnitCube());
        using var restored = asset.ToHashEncodedDeepMLP();

        var point = new Vector3(0.1f, 0.2f, 0.3f);
        network.Evaluate(point).Should().BeApproximately(restored.Evaluate(point), 1e-5f);
    }
}

public class GlTFExporterTests
{
    [Fact]
    public async Task ExportAndLoad_RoundTripsUnitCube()
    {
        var source = MeshAsset.CreateUnitCube(1f, "RoundTripCube");
        var loader = new MeshLoader();
        var gltfPath = Path.Combine(Path.GetTempPath(), $"synapse_cube_{Guid.NewGuid():N}.gltf");

        try
        {
            var export = await loader.ExportAsync(gltfPath, source);
            export.Success.Should().BeTrue(export.ErrorMessage);
            File.Exists(gltfPath).Should().BeTrue();

            var load = await loader.LoadAsync(gltfPath);
            load.Success.Should().BeTrue(load.ErrorMessage);
            load.Asset!.TotalVertexCount.Should().Be(source.TotalVertexCount);
            load.Asset.TotalTriangleCount.Should().Be(source.TotalTriangleCount);
        }
        finally
        {
            if (File.Exists(gltfPath))
                File.Delete(gltfPath);
            var binPath = Path.ChangeExtension(gltfPath, ".bin");
            if (File.Exists(binPath))
                File.Delete(binPath);
        }
    }

    [Fact]
    public async Task ExportGlb_WritesBinaryContainer()
    {
        var source = MeshAsset.CreateUnitCube(0.5f, "GlbCube");
        var loader = new MeshLoader();
        var glbPath = Path.Combine(Path.GetTempPath(), $"synapse_cube_{Guid.NewGuid():N}.glb");

        try
        {
            var export = await loader.ExportAsync(glbPath, source);
            export.Success.Should().BeTrue(export.ErrorMessage);
            File.Exists(glbPath).Should().BeTrue();
            new FileInfo(glbPath).Length.Should().BeGreaterThan(64);
        }
        finally
        {
            if (File.Exists(glbPath))
                File.Delete(glbPath);
        }
    }
}

public class ArtisticStylePassTests
{
    [Fact]
    public void Grayscale_ConvertsColorToLuminance()
    {
        var color = new Vector3(0.2f, 0.5f, 0.9f);
        var gray = ArtisticStylePass.ApplyGrayscale(color);
        gray.X.Should().Be(gray.Y);
        gray.Y.Should().Be(gray.Z);
        gray.X.Should().BeApproximately(ArtisticStylePass.Luminance(color), 1e-5f);
    }

    [Fact]
    public void Noir_CrushesShadowsAndBoostsContrast()
    {
        var config = new ArtisticStyleConfig { NoirContrast = 2.0f, NoirCrush = 0.2f };
        var dark = ArtisticStylePass.ApplyNoir(new Vector3(0.05f), config);
        dark.X.Should().BeGreaterOrEqualTo(0.2f);

        var bright = ArtisticStylePass.ApplyNoir(new Vector3(0.9f), config);
        bright.X.Should().BeGreaterThan(dark.X);
    }

    [Fact]
    public void Cartoon_QuantizesColorsOnBuffer()
    {
        using var input = new HDRFrameBuffer(2, 2);
        input.SetPixel(0, 0, new Vector4(0.42f, 0.58f, 0.73f, 1f));

        var config = new ArtisticStyleConfig
        {
            CartoonColorLevels = 4,
            CartoonEdgeStrength = 0f
        };

        var quantized = ArtisticStylePass.ApplyCartoon(
            new Vector3(0.42f, 0.58f, 0.73f),
            input,
            0,
            0,
            input.Width,
            input.Height,
            config);

        quantized.X.Should().BeOneOf(0.0f, 1f / 3f, 2f / 3f, 1.0f);
    }
}
