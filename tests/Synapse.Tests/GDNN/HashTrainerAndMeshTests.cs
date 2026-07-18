using FluentAssertions;
using GDNN.Core.NeuralNetwork;
using GDNN.Evaluation;
using GDNN.GPU;
using System.Numerics;
using Xunit;

namespace Synapse.Tests.GDNN;

public class HashTrainerAndMeshTests
{
    [Fact]
    public void HashTrainer_ReducesLossOnSphereSdf()
    {
        using var network = new HashEncodedDeepMLP(new Random(7));
        var trainer = new HashEncodedDeepMLPTrainer(network)
        {
            LearningRate = 5e-3f,
            HashLearningRate = 5e-2f
        };

        Func<Vector3, float> target = p => ReferenceMeshSdf.AnalyticSphereSdf(p);
        float lossBefore = MeasureMse(network, target, 256, new Random(1));
        trainer.FitToTarget(target, sampleCount: 512, epochs: 20, random: new Random(7));
        float lossAfter = MeasureMse(network, target, 256, new Random(1));

        lossAfter.Should().BeLessThan(lossBefore);
    }

    [Fact]
    public void ReferenceMesh_HausdorffAgainstTrainedHashNetwork()
    {
        using var network = new HashEncodedDeepMLP(new Random(3));
        var trainer = new HashEncodedDeepMLPTrainer(network)
        {
            LearningRate = 5e-3f,
            HashLearningRate = 5e-2f
        };
        trainer.FitToTarget(p => ReferenceMeshSdf.AnalyticSphereSdf(p, 0.5f), sampleCount: 1024, epochs: 25, random: new Random(3));

        var mesh = ReferenceMeshSdf.CreateUnitSphereIcosahedron(0.5f);
        var points = GDNNValidationProtocol.SamplePointCloud(256, new Random(9));

        float rms = mesh.ComputeRmsError(network, points);
        // After training, |sdf| should track mesh distance better than random (~0.5+ typically)
        rms.Should().BeLessThan(0.45f);
    }

    [Fact]
    public void ValidateAgainstMesh_ProducesReport()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var mesh = ReferenceMeshSdf.CreateUnitSphereIcosahedron();
        var report = GDNNValidationProtocol.ValidateAgainstMesh(network, mesh, 128);

        report.SampleCount.Should().Be(128);
        report.HausdorffError.Should().BeGreaterThanOrEqualTo(0f);
        report.MemoryBytesFp32.Should().BeGreaterThan(0);
    }

    [Fact]
    public void HlslCompatibleEvaluator_MatchesDeepMicroMLP_WhenScaleIsOne()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        network.PositionalScale.Should().Be(1f);

        var hlsl = new HlslCompatibleEvaluator(network);
        var points = TrainedNetworkFixture.SamplePointCloud(64);

        for (int i = 0; i < points.Length; i++)
        {
            float cpu = network.Evaluate(points[i]);
            float proxy = hlsl.Evaluate(points[i]);
            MathF.Abs(cpu - proxy).Should().BeLessThan(1e-4f, $"point {i}");
        }
    }

    private static float MeasureMse(HashEncodedDeepMLP network, Func<Vector3, float> target, int count, Random random)
    {
        float sum = 0f;
        for (int i = 0; i < count; i++)
        {
            var p = new Vector3(
                (float)(random.NextDouble() * 2 - 1),
                (float)(random.NextDouble() * 2 - 1),
                (float)(random.NextDouble() * 2 - 1));
            float diff = network.Evaluate(p) - target(p);
            sum += diff * diff;
        }
        return sum / count;
    }
}
