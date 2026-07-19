using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using GDNN.Core.NeuralNetwork;
using GDNN.SIMD;

namespace GDNN.SIMD;

/// <summary>
/// Évalue le SDF neuronal pour un lot de positions en parallèle vectoriel,
/// au lieu d'un appel par point comme dans RayMarcher.
/// </summary>
public static class BatchSdfEvaluator
{
    /// <summary>Temps d'inférence cumulé (ms) du dernier batch.</summary>
    public static double LastBatchTimeMs { get; private set; }

    /// <summary>
    /// Évalue le SDF neuronal pour un lot de positions (dispatch selon le type concret).
    /// </summary>
    public static void EvaluateBatch(
        ISdfNetwork network,
        ReadOnlySpan<Vector3> positions,
        Span<float> distancesOut)
    {
        switch (network)
        {
            case DeepMicroMLP deep:
                EvaluateBatch(deep, positions, distancesOut);
                break;
            case HashEncodedDeepMLP hash:
                EvaluateBatch(hash, positions, distancesOut);
                break;
            case QuantizedDeepMLP quantized:
                EvaluateBatch(quantized, positions, distancesOut);
                break;
            default:
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    network.EvaluateBatch(positions, distancesOut);
                    sw.Stop();
                    LastBatchTimeMs = sw.Elapsed.TotalMilliseconds;
                    break;
                }
        }
    }

    /// <summary>
    /// Évalue le SDF DeepMicroMLP pour un lot de positions (chemin vectorisé SoA).
    /// </summary>
    public static void EvaluateBatch(
        DeepMicroMLP network,
        ReadOnlySpan<Vector3> positions,
        Span<float> distancesOut)
    {
        if (positions.Length != distancesOut.Length)
            throw new ArgumentException("Positions and distances must have the same length.");

        var sw = System.Diagnostics.Stopwatch.StartNew();

        int width = Vector<float>.Count;
        int i = 0;
        for (; i + width <= positions.Length; i += width)
            network.EvaluateVectorized(positions.Slice(i, width), distancesOut.Slice(i, width));

        for (; i < positions.Length; i++)
            distancesOut[i] = network.Evaluate(positions[i]);

        sw.Stop();
        LastBatchTimeMs = sw.Elapsed.TotalMilliseconds;
    }

    /// <summary>
    /// Évalue le SDF hash-encodé pour un lot de positions.
    /// </summary>
    public static void EvaluateBatch(
        HashEncodedDeepMLP network,
        ReadOnlySpan<Vector3> positions,
        Span<float> distancesOut)
    {
        network.EvaluateBatch(positions, distancesOut);
    }

    /// <summary>
    /// Évalue le SDF quantifié pour un lot de positions.
    /// </summary>
    public static void EvaluateBatch(
        QuantizedDeepMLP network,
        ReadOnlySpan<Vector3> positions,
        Span<float> distancesOut)
    {
        network.EvaluateBatch(positions, distancesOut);
    }

    /// <summary>
    /// Débit estimé en points/ms sur le dernier batch.
    /// </summary>
    public static double GetThroughputPointsPerMs(int pointCount) =>
        LastBatchTimeMs > 0 ? pointCount / LastBatchTimeMs : 0;
}
