using System;
using System.Collections.Generic;
using System.Numerics;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Entraîne un HashEncodedDeepMLP (tables de hachage + poids MLP) pour coller à un SDF cible.
/// Utilise SGD avec rétropropagation analytique à travers SiLU et l'interpolation trilinéaire.
/// </summary>
public sealed class HashEncodedDeepMLPTrainer
{
    private readonly HashEncodedDeepMLP _network;
    private readonly List<MultiResolutionHashEncoder.CornerContribution> _contributions = new(128);

    public float LearningRate { get; set; } = 1e-2f;
    public float HashLearningRate { get; set; } = 1e-1f;
    public float LastLoss { get; private set; }

    public HashEncodedDeepMLPTrainer(HashEncodedDeepMLP network)
    {
        _network = network ?? throw new ArgumentNullException(nameof(network));
    }

    /// <summary>
    /// Une étape d'entraînement sur un point (perte MSE).
    /// </summary>
    public float TrainStep(Vector3 point, float targetDistance)
    {
        int encDim = _network.EncodedDimension;
        int hidden = HashEncodedDeepMLP.HiddenSize;

        Span<float> encoded = stackalloc float[encDim];
        _network.Encoder.Encode(point, encoded, _contributions);

        Span<float> h1 = stackalloc float[hidden];
        Span<float> h1Pre = stackalloc float[hidden];
        Span<float> h2 = stackalloc float[hidden];
        Span<float> h2Pre = stackalloc float[hidden];

        // Forward layer 1
        for (int o = 0; o < hidden; o++)
        {
            float sum = _network.Layer1Bias[o];
            int row = o * encDim;
            for (int j = 0; j < encDim; j++)
                sum += _network.Layer1Weights[row + j] * encoded[j];
            h1Pre[o] = sum;
            h1[o] = SiLU(sum);
        }

        // Forward layer 2
        for (int o = 0; o < hidden; o++)
        {
            float sum = _network.Layer2Bias[o];
            int row = o * hidden;
            for (int j = 0; j < hidden; j++)
                sum += _network.Layer2Weights[row + j] * h1[j];
            h2Pre[o] = sum;
            h2[o] = SiLU(sum);
        }

        float pred = _network.OutputBias;
        for (int j = 0; j < hidden; j++)
            pred += _network.OutputWeights[j] * h2[j];

        float error = pred - targetDistance;
        LastLoss = error * error;
        float dLoss = 2f * error; // d(MSE)/d(pred)

        // Backprop output
        Span<float> dH2 = stackalloc float[hidden];
        for (int j = 0; j < hidden; j++)
        {
            dH2[j] = dLoss * _network.OutputWeights[j] * SiLUDerivative(h2Pre[j]);
            _network.OutputWeights[j] -= LearningRate * dLoss * h2[j];
        }
        _network.OutputBias -= LearningRate * dLoss;

        // Backprop layer 2
        Span<float> dH1 = stackalloc float[hidden];
        dH1.Clear();
        for (int o = 0; o < hidden; o++)
        {
            int row = o * hidden;
            _network.Layer2Bias[o] -= LearningRate * dH2[o];
            for (int j = 0; j < hidden; j++)
            {
                dH1[j] += dH2[o] * _network.Layer2Weights[row + j];
                _network.Layer2Weights[row + j] -= LearningRate * dH2[o] * h1[j];
            }
        }

        for (int j = 0; j < hidden; j++)
            dH1[j] *= SiLUDerivative(h1Pre[j]);

        // Backprop layer 1 → encoded
        Span<float> dEncoded = stackalloc float[encDim];
        dEncoded.Clear();
        for (int o = 0; o < hidden; o++)
        {
            int row = o * encDim;
            _network.Layer1Bias[o] -= LearningRate * dH1[o];
            for (int j = 0; j < encDim; j++)
            {
                dEncoded[j] += dH1[o] * _network.Layer1Weights[row + j];
                _network.Layer1Weights[row + j] -= LearningRate * dH1[o] * encoded[j];
            }
        }

        // Backprop into hash tables
        _network.Encoder.AccumulateFeatureGradient(_contributions.ToArray(), dEncoded, HashLearningRate);

        return LastLoss;
    }

    /// <summary>
    /// Entraîne sur un ensemble de points pendant plusieurs époques.
    /// </summary>
    public float Fit(
        ReadOnlySpan<Vector3> points,
        ReadOnlySpan<float> targets,
        int epochs = 50,
        Random? random = null)
    {
        if (points.Length != targets.Length)
            throw new ArgumentException("Points and targets must have the same length.");

        random ??= Random.Shared;
        float meanLoss = 0f;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float epochLoss = 0f;
            // Shuffle indices lightly
            for (int i = 0; i < points.Length; i++)
            {
                int idx = random.Next(points.Length);
                epochLoss += TrainStep(points[idx], targets[idx]);
            }
            meanLoss = epochLoss / points.Length;
        }

        return meanLoss;
    }

    /// <summary>
    /// Ajuste le réseau pour coller à une fonction SDF analytique (ex. sphère).
    /// </summary>
    public float FitToTarget(
        Func<Vector3, float> targetSdf,
        int sampleCount = 2048,
        int epochs = 30,
        Random? random = null)
    {
        random ??= new Random(42);
        var points = new Vector3[sampleCount];
        var targets = new float[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            points[i] = new Vector3(
                (float)(random.NextDouble() * 2.0 - 1.0),
                (float)(random.NextDouble() * 2.0 - 1.0),
                (float)(random.NextDouble() * 2.0 - 1.0));
            targets[i] = targetSdf(points[i]);
        }

        return Fit(points, targets, epochs, random);
    }

    private static float SiLU(float x) => x / (1f + MathF.Exp(-x));

    private static float SiLUDerivative(float x)
    {
        float sig = 1f / (1f + MathF.Exp(-x));
        return sig + x * sig * (1f - sig);
    }
}
