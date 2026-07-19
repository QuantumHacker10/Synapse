using System.Numerics;
using FluentAssertions;
using GDNN.Core.NeuralNetwork;
using GDNN.SIMD;
using Xunit;

namespace Synapse.Tests.GDNN;

public class BatchSdfEvaluatorTests
{
    [Fact]
    public void EvaluateVectorized_MatchesScalarEvaluation()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var points = TrainedNetworkFixture.SamplePointCloud(32);

        int width = System.Numerics.Vector<float>.Count;
        for (int i = 0; i + width <= points.Length; i += width)
        {
            Span<float> vectorized = stackalloc float[width];
            network.EvaluateVectorized(points.AsSpan(i, width), vectorized);

            for (int lane = 0; lane < width; lane++)
            {
                float scalar = network.Evaluate(points[i + lane]);
                MathF.Abs(scalar - vectorized[lane]).Should().BeLessThan(1e-4f,
                    $"divergence at lane {lane}, point index {i + lane}");
            }
        }
    }

    [Fact]
    public void BatchSdfEvaluator_MatchesScalarBatch()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var points = TrainedNetworkFixture.SamplePointCloud(64);

        var scalar = new float[points.Length];
        var batched = new float[points.Length];

        for (int i = 0; i < points.Length; i++)
            scalar[i] = network.Evaluate(points[i]);

        BatchSdfEvaluator.EvaluateBatch(network, points, batched);

        for (int i = 0; i < points.Length; i++)
            batched[i].Should().BeApproximately(scalar[i], 1e-4f);
    }

    [Fact]
    public void BatchSdfEvaluator_ReportsThroughput()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var points = TrainedNetworkFixture.SamplePointCloud(256);
        var distances = new float[points.Length];

        BatchSdfEvaluator.EvaluateBatch(network, points, distances);

        BatchSdfEvaluator.LastBatchTimeMs.Should().BeGreaterThan(0);
        BatchSdfEvaluator.GetThroughputPointsPerMs(points.Length).Should().BeGreaterThan(0);
    }
}
