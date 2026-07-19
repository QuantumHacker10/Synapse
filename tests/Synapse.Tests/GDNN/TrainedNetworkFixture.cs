using System;
using System.Numerics;
using GDNN.Core.NeuralNetwork;

namespace Synapse.Tests.GDNN;

/// <summary>
/// Fixture réseau déterministe pour les tests d'intégration GDNN.
/// </summary>
public static class TrainedNetworkFixture
{
    private static DeepMicroMLP? _cached;

    public static DeepMicroMLP LoadTestNetwork()
    {
        if (_cached != null)
            return _cached;
        _cached = new DeepMicroMLP(new Random(12345));
        _cached.PositionalScale = 1.0f;
        return _cached;
    }

    public static Vector3[] SamplePointCloud(int count, int seed = 42)
    {
        var random = new Random(seed);
        var points = new Vector3[count];
        for (int i = 0; i < count; i++)
        {
            points[i] = new Vector3(
                (float)(random.NextDouble() * 2.0 - 1.0),
                (float)(random.NextDouble() * 2.0 - 1.0),
                (float)(random.NextDouble() * 2.0 - 1.0));
        }
        return points;
    }
}
