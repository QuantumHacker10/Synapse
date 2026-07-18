using FluentAssertions;
using GDNN.Core.NeuralNetwork;
using GDNN.Evaluation;
using GDNN.GPU;
using Xunit;

namespace Synapse.Tests.GDNN;

public class GpuShaderIntegrationTests
{
    [Fact]
    public void GeneratedShader_CompilesSuccessfully()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        string source = NeuralComputeShaderGenerator.GenerateComputeShader(network);

        source.Should().Contain("EvaluateNeuralSDF");
        source.Should().Contain("CSMain");

        using var compiler = new ShaderCompiler();
        var result = compiler.Compile(source, "CSMain", ShaderType.ComputeShader);

        result.Success.Should().BeTrue(result.GetErrorSummary());
        result.Bytecode.Should().NotBeNull();
    }

    [Fact]
    public void GeneratedShader_MatchesCpuInference_WithinTolerance()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var testPoints = TrainedNetworkFixture.SamplePointCloud(256);

        var (compilation, gpuResults) = GpuTestHarness.CompileAndRun(network, testPoints);
        compilation.Success.Should().BeTrue();

        for (int i = 0; i < testPoints.Length; i++)
        {
            float cpu = network.Evaluate(testPoints[i]);
            MathF.Abs(cpu - gpuResults[i]).Should().BeLessThan(1e-3f,
                $"divergence at point {i}");
        }
    }
}

public class GDNNValidationProtocolTests
{
    [Fact]
    public void ValidateAgainstReference_IdenticalNetworks_ZeroError()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var report = GDNNValidationProtocol.ValidateAgainstReference(network, network, 256);

        report.HausdorffError.Should().Be(0f);
        report.RmsError.Should().Be(0f);
    }

    [Fact]
    public void RunFullBenchmark_ProducesMetrics()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var report = GDNNValidationProtocol.RunFullBenchmark(network, 512);

        report.MemoryBytesFp32.Should().BeGreaterThan(0);
        report.MemoryBytesQuantized.Should().BeLessThan(report.MemoryBytesFp32);
        report.QuantizationRmsError.Should().BeLessThan(0.05f);
        report.BatchThroughputPointsPerMs.Should().BeGreaterThan(0);
    }

    [Fact]
    public void BenchmarkHashEncoded_ProducesMetrics()
    {
        using var network = new HashEncodedDeepMLP(new Random(42));
        var report = GDNNValidationProtocol.BenchmarkHashEncoded(network, 256);

        report.MemoryBytesFp32.Should().BeGreaterThan(0);
        report.BatchThroughputPointsPerMs.Should().BeGreaterThan(0);
    }
}
