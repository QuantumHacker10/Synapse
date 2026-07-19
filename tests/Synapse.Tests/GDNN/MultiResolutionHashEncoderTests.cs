using System.Numerics;
using FluentAssertions;
using GDNN.Core.NeuralNetwork;
using Xunit;

namespace Synapse.Tests.GDNN;

public class MultiResolutionHashEncoderTests
{
    [Fact]
    public void Encode_ProducesCorrectOutputDimension()
    {
        var encoder = new MultiResolutionHashEncoder(new Random(42));
        encoder.OutputDimension.Should().Be(16);

        Span<float> output = stackalloc float[encoder.OutputDimension];
        encoder.Encode(new Vector3(0.5f, -0.3f, 0.1f), output);

        output.Length.Should().Be(16);
        float.IsNaN(output[0]).Should().BeFalse();
    }

    [Fact]
    public void Encode_IsContinuous_SmallPerturbationSmallChange()
    {
        var encoder = new MultiResolutionHashEncoder(new Random(42));
        Span<float> a = stackalloc float[16];
        Span<float> b = stackalloc float[16];

        encoder.Encode(new Vector3(0.5f, 0.5f, 0.5f), a);
        encoder.Encode(new Vector3(0.501f, 0.5f, 0.5f), b);

        float diff = 0f;
        for (int i = 0; i < 16; i++)
            diff += MathF.Abs(a[i] - b[i]);

        diff.Should().BeLessThan(1.0f);
    }

    [Fact]
    public void Serialize_RoundTrips()
    {
        var encoder = new MultiResolutionHashEncoder(new Random(99));
        float[] serialized = encoder.Serialize();
        var restored = new MultiResolutionHashEncoder(serialized);

        Span<float> original = stackalloc float[16];
        Span<float> roundTrip = stackalloc float[16];
        var point = new Vector3(0.2f, -0.7f, 0.4f);

        encoder.Encode(point, original);
        restored.Encode(point, roundTrip);

        for (int i = 0; i < 16; i++)
            roundTrip[i].Should().BeApproximately(original[i], 1e-6f);
    }

    [Fact]
    public void HashEncodedDeepMLP_Evaluate_ReturnsFiniteValue()
    {
        using var network = new HashEncodedDeepMLP(new Random(42));
        float sdf = network.Evaluate(new Vector3(0f, 0f, 0f));

        float.IsNaN(sdf).Should().BeFalse();
        float.IsInfinity(sdf).Should().BeFalse();
    }
}
