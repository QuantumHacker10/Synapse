using System.Numerics;
using FluentAssertions;
using GDNN.Lighting.LDNN;
using Xunit;

namespace Synapse.Tests.Rendering;

/// <summary>
/// Profils de taille du réseau de prédiction d'irradiance (Tiny/Small/Full) :
/// initialisation, inférence, entraînement et sérialisation sur chaque variante.
/// </summary>
public class LDNNNetworkProfileTests
{
    private static Vector3[] CreateFeatures(float value = 0.5f)
    {
        var features = new Vector3[32];
        for (int i = 0; i < features.Length; i++)
            features[i] = new Vector3(value, 0, 0);
        return features;
    }

    [Theory]
    [InlineData(NeuralNetworkProfile.Tiny, 32, 32)]
    [InlineData(NeuralNetworkProfile.Small, 64, 64)]
    [InlineData(NeuralNetworkProfile.Full, 128, 128)]
    public void Initialize_SetsHiddenLayerSizes(NeuralNetworkProfile profile, int hidden1, int hidden2)
    {
        var network = new NeuralPredictiveIrradiance();
        network.Initialize(profile);

        network.Profile.Should().Be(profile);
        network.HiddenLayer1Size.Should().Be(hidden1);
        network.HiddenLayer2Size.Should().Be(hidden2);
        network.IsInitialized.Should().BeTrue();
    }

    [Theory]
    [InlineData(NeuralNetworkProfile.Tiny)]
    [InlineData(NeuralNetworkProfile.Small)]
    [InlineData(NeuralNetworkProfile.Full)]
    public void ForwardPass_ProducesBoundedOutput(NeuralNetworkProfile profile)
    {
        var network = new NeuralPredictiveIrradiance();
        network.Initialize(profile);

        Vector3 output = network.ForwardPass(CreateFeatures());

        // Sortie sigmoïde : chaque canal dans [0, 1].
        output.X.Should().BeInRange(0f, 1f);
        output.Y.Should().BeInRange(0f, 1f);
        output.Z.Should().BeInRange(0f, 1f);
    }

    [Theory]
    [InlineData(NeuralNetworkProfile.Tiny)]
    [InlineData(NeuralNetworkProfile.Small)]
    [InlineData(NeuralNetworkProfile.Full)]
    public void Training_ReducesLossTowardConstantTarget(NeuralNetworkProfile profile)
    {
        var network = new NeuralPredictiveIrradiance();
        network.Initialize(profile);

        var features = CreateFeatures();
        var target = new Vector3(0.8f, 0.2f, 0.4f);

        Vector3 before = network.ForwardPass(features);
        for (int step = 0; step < 200; step++)
            network.BackwardPass(features, target, 0.01f);
        Vector3 after = network.ForwardPass(features);

        (after - target).Length().Should().BeLessThan((before - target).Length(),
            "l'entraînement doit rapprocher la prédiction de la cible");
        network.TrainingStep.Should().Be(200);
    }

    [Theory]
    [InlineData(NeuralNetworkProfile.Tiny)]
    [InlineData(NeuralNetworkProfile.Small)]
    [InlineData(NeuralNetworkProfile.Full)]
    public void SerializeDeserialize_PreservesProfileAndPredictions(NeuralNetworkProfile profile)
    {
        var network = new NeuralPredictiveIrradiance();
        network.Initialize(profile);
        var features = CreateFeatures(0.3f);
        Vector3 expected = network.ForwardPass(features);

        var restored = new NeuralPredictiveIrradiance();
        restored.Deserialize(network.Serialize());

        restored.Profile.Should().Be(profile);
        restored.HiddenLayer1Size.Should().Be(network.HiddenLayer1Size);
        Vector3 actual = restored.ForwardPass(features);
        (actual - expected).Length().Should().BeLessThan(1e-5f);
    }

    [Fact]
    public void Renderer_AdaptQuality_DowngradesProfileUnderBudgetPressure()
    {
        var renderer = new LDNNRenderer();
        var config = new LDNNConfig { NeuralNetworkProfile = NeuralNetworkProfile.Full };
        renderer.Initialize(config, 8, 8);

        renderer.SetNeuralNetworkProfile(NeuralNetworkProfile.Small);

        renderer.Config.NeuralNetworkProfile.Should().Be(NeuralNetworkProfile.Small);
    }
}
