using FluentAssertions;
using GDNN.Core.NeuralNetwork;
using System.Numerics;
using Xunit;

namespace Synapse.Tests.GDNN;

public class QuantizedDeepMLPTests
{
    [Fact]
    public void FromTrained_QuantizationError_IsSmall()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var quantized = QuantizedDeepMLP.FromTrained(network);
        var points = TrainedNetworkFixture.SamplePointCloud(512);

        float rmsError = quantized.MeasureQuantizationError(network, points);
        rmsError.Should().BeLessThan(0.05f);
    }

    [Fact]
    public void Quantized_UsesLessMemoryThanFp32()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var quantized = QuantizedDeepMLP.FromTrained(network);

        int fp32Bytes = DeepMicroMLP.GetTotalWeightCount() * sizeof(float);
        int qBytes = quantized.GetMemoryFootprintBytes();

        qBytes.Should().BeLessThan(fp32Bytes);
    }

    [Fact]
    public void Quantized_Evaluate_MatchesFp32WithinTolerance()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var quantized = QuantizedDeepMLP.FromTrained(network);
        var point = new Vector3(0.3f, -0.2f, 0.7f);

        float fp32 = network.Evaluate(point);
        float q = quantized.Evaluate(point);

        MathF.Abs(fp32 - q).Should().BeLessThan(0.01f);
    }
}
