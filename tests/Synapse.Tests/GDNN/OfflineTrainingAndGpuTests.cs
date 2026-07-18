using FluentAssertions;
using GDNN.Core.NeuralNetwork;
using GDNN.Evaluation;
using GDNN.GPU;
using Xunit;
using Xunit.Abstractions;

namespace Synapse.Tests.GDNN;

public class OfflineTrainingAndGpuTests
{
    private readonly ITestOutputHelper _output;

    public OfflineTrainingAndGpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SubdividedSphere_HasMoreTrianglesThanBase()
    {
        var coarse = ReferenceMeshSdf.CreateUnitSphereIcosahedron(0.5f, subdivisions: 0);
        var fine = ReferenceMeshSdf.CreateUnitSphereIcosahedron(0.5f, subdivisions: 3);

        fine.TriangleCount.Should().BeGreaterThan(coarse.TriangleCount);
        fine.VertexCount.Should().BeGreaterThan(coarse.VertexCount);
        fine.TriangleCount.Should().Be(20 * (int)Math.Pow(4, 3)); // 1280
    }

    [Fact]
    public void OfflineHashMeshTrainer_ImprovesOnHighResSphere()
    {
        using var network = new HashEncodedDeepMLP(new Random(11));
        var trainer = new OfflineHashMeshTrainer
        {
            LearningRate = 5e-3f,
            HashLearningRate = 5e-2f
        };

        var report = trainer.TrainOnSubdividedSphere(
            network,
            subdivisions: 2,
            radius: 0.5f,
            sampleCount: 1536,
            epochs: 35,
            random: new Random(11));

        _output.WriteLine(report.ToString());

        report.Improved.Should().BeTrue();
        report.MeshTriangles.Should().BeGreaterThan(20);
        report.LossAfter.Should().BeLessThan(report.LossBefore);
        report.RmsError.Should().BeLessThan(0.5f);
    }

    [Fact]
    public void SpirvEmitter_GeneratesValidGlsl()
    {
        string glsl = DeepMicroMLPSpirvEmitter.GenerateGlsl();
        glsl.Should().Contain("#version 450");
        glsl.Should().Contain("layout(local_size_x");
        glsl.Should().Contain("float evaluate(vec3 p)");
        glsl.Should().Contain("void main()");
    }

    [Fact]
    public void GpuTestHarness_ReportsStatus_AndStillProducesResults()
    {
        _output.WriteLine("GPU status: " + GpuTestHarness.GpuStatus);
        _output.WriteLine($"IsGpuAvailable={GpuTestHarness.IsGpuAvailable}");
        _output.WriteLine($"DXC={SpirvToolchain.IsDxcAvailable}, glslang={SpirvToolchain.IsGlslangAvailable}");

        var network = TrainedNetworkFixture.LoadTestNetwork();
        var points = TrainedNetworkFixture.SamplePointCloud(128);
        var (compilation, results) = GpuTestHarness.CompileAndRun(network, points);

        compilation.Success.Should().BeTrue();
        results.Should().HaveCount(128);

        // Regardless of GPU availability, results must stay close to HLSL-compatible CPU
        var hlsl = new HlslCompatibleEvaluator(network);
        for (int i = 0; i < points.Length; i++)
        {
            float expected = hlsl.Evaluate(points[i]);
            // If real GPU ran, allow slightly looser tolerance; proxy should be near-exact
            MathF.Abs(results[i] - expected).Should().BeLessThan(GpuTestHarness.IsGpuAvailable ? 1e-2f : 1e-4f);
        }
    }

    [Fact]
    public void ValidateAgainstMesh_HighResSphere()
    {
        using var network = new HashEncodedDeepMLP(new Random(5));
        var trainer = new OfflineHashMeshTrainer();
        trainer.TrainOnSubdividedSphere(network, subdivisions: 2, sampleCount: 768, epochs: 20, random: new Random(5));

        var mesh = ReferenceMeshSdf.CreateUnitSphereIcosahedron(0.5f, subdivisions: 2);
        var report = GDNNValidationProtocol.ValidateAgainstMesh(network, mesh, 256);

        report.RmsError.Should().BeLessThan(0.5f);
        report.MemoryBytesFp32.Should().BeGreaterThan(0);
    }
}
