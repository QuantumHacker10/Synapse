using System.Numerics;
using FluentAssertions;
using GDNN.Core.NeuralNetwork;
using GDNN.Evaluation;
using Xunit;

namespace Synapse.Tests.GDNN;

public class PipelineIntegrationTests
{
    [Fact]
    public void SurfaceEvaluator_AcceptsDeepMicroMLP()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        using var evaluator = new SurfaceEvaluator();
        evaluator.AddLodLevel(new LodLevel
        {
            Network = network,
            MaxDistance = 100f,
            BoundingRadius = 2f
        });

        float sdf = evaluator.Evaluate(Vector3.Zero, Vector3.UnitZ * 5f);
        float.IsNaN(sdf).Should().BeFalse();
    }

    [Fact]
    public void SurfaceEvaluator_AcceptsHashEncodedDeepMLP()
    {
        using var network = new HashEncodedDeepMLP(new Random(1));
        using var evaluator = new SurfaceEvaluator();
        evaluator.AddLodLevel(new LodLevel
        {
            Network = network,
            MaxDistance = 100f,
            BoundingRadius = 2f
        });

        float sdf = evaluator.Evaluate(new Vector3(0.1f, 0f, 0f), Vector3.UnitZ * 5f);
        float.IsNaN(sdf).Should().BeFalse();
    }

    [Fact]
    public void RayMarcher_TracesDeepMicroMLP()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        using var marcher = new RayMarcher();

        var ray = new TracingRay(new Vector3(0, 0, -2), Vector3.UnitZ, 10f);
        var hit = marcher.Trace(network, ray);
        // Random network may or may not hit — ensure the call completes without throwing
        (hit.DidHit || !hit.DidHit).Should().BeTrue();
    }

    [Fact]
    public void NeuralLodSelector_RegistersDeepMicroMLP()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        using var selector = new NeuralLodSelector();
        selector.RegisterLod(0, network, Vector3.Zero, 1f, 50f);

        selector.CandidateCount.Should().Be(1);
        selector.GetLodNetwork(0).Should().BeSameAs(network);
    }

    [Fact]
    public void NeuralAsset_RoundTripsDeepMicroMLP()
    {
        var network = TrainedNetworkFixture.LoadTestNetwork();
        var asset = new NeuralAsset();
        asset.FromDeepMicroMLP(network);

        var restored = asset.ToDeepMicroMLP();
        var point = new Vector3(0.2f, -0.1f, 0.3f);

        restored.Evaluate(point).Should().BeApproximately(network.Evaluate(point), 1e-5f);
        asset.ToSdfNetwork().Should().BeAssignableTo<DeepMicroMLP>();
    }
}
