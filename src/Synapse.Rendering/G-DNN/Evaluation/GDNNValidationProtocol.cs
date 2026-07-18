using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Evaluation;

/// <summary>
/// Protocole de validation global GDNN — condition pour affirmer des gains mesurables.
/// Remplit le tableau de métriques : Hausdorff, mémoire, débit, erreur de quantification.
/// </summary>
public sealed class GDNNValidationProtocol
{
    /// <summary>Résultat d'une campagne de validation.</summary>
    public sealed class ValidationReport
    {
        public float HausdorffError { get; init; }
        public float RmsError { get; init; }
        public int MemoryBytesFp32 { get; init; }
        public int MemoryBytesQuantized { get; init; }
        public double InferenceMsPerPoint { get; init; }
        public double BatchThroughputPointsPerMs { get; init; }
        public float QuantizationRmsError { get; init; }
        public int SampleCount { get; init; }
        public double ElapsedMs { get; init; }

        public override string ToString() =>
            $"Hausdorff={HausdorffError:F6}, RMS={RmsError:F6}, " +
            $"Mem(fp32)={MemoryBytesFp32}, Mem(q)={MemoryBytesQuantized}, " +
            $"Inf={InferenceMsPerPoint:F4}ms/pt, Batch={BatchThroughputPointsPerMs:F0}pt/ms, " +
            $"QuantErr={QuantizationRmsError:F6}";
    }

    /// <summary>
    /// Compare un réseau candidat contre un réseau de référence sur un nuage de points.
    /// </summary>
    public static ValidationReport ValidateAgainstReference(
        DeepMicroMLP candidate,
        DeepMicroMLP reference,
        int sampleCount = 4096,
        Random? random = null)
    {
        random ??= new Random(42);
        var sw = Stopwatch.StartNew();

        var points = SamplePointCloud(sampleCount, random);
        float maxError = 0f;
        float sumSq = 0f;

        foreach (var p in points)
        {
            float diff = MathF.Abs(candidate.Evaluate(p) - reference.Evaluate(p));
            maxError = MathF.Max(maxError, diff);
            sumSq += diff * diff;
        }

        sw.Stop();
        return new ValidationReport
        {
            HausdorffError = maxError,
            RmsError = MathF.Sqrt(sumSq / sampleCount),
            MemoryBytesFp32 = DeepMicroMLP.GetTotalWeightCount() * sizeof(float),
            SampleCount = sampleCount,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            InferenceMsPerPoint = sw.Elapsed.TotalMilliseconds / sampleCount
        };
    }

    /// <summary>
    /// Mesure complète incluant quantification et batch SIMD.
    /// </summary>
    public static ValidationReport RunFullBenchmark(
        DeepMicroMLP network,
        int sampleCount = 4096,
        Random? random = null)
    {
        random ??= new Random(42);
        var points = SamplePointCloud(sampleCount, random);
        var distances = new float[sampleCount];

        // Scalar inference timing
        var swScalar = Stopwatch.StartNew();
        for (int i = 0; i < sampleCount; i++)
            distances[i] = network.Evaluate(points[i]);
        swScalar.Stop();

        // Batch SIMD timing
        var swBatch = Stopwatch.StartNew();
        network.EvaluateBatch(points, distances);
        swBatch.Stop();

        // Quantization error
        var quantized = QuantizedDeepMLP.FromTrained(network);
        float quantErr = quantized.MeasureQuantizationError(network, points);

        return new ValidationReport
        {
            MemoryBytesFp32 = DeepMicroMLP.GetTotalWeightCount() * sizeof(float),
            MemoryBytesQuantized = quantized.GetMemoryFootprintBytes(),
            InferenceMsPerPoint = swScalar.Elapsed.TotalMilliseconds / sampleCount,
            BatchThroughputPointsPerMs = sampleCount / Math.Max(swBatch.Elapsed.TotalMilliseconds, 1e-6),
            QuantizationRmsError = quantErr,
            SampleCount = sampleCount,
            ElapsedMs = swScalar.Elapsed.TotalMilliseconds + swBatch.Elapsed.TotalMilliseconds
        };
    }

    /// <summary>
    /// Mesure la mémoire et le débit d'un réseau hash-encodé.
    /// </summary>
    public static ValidationReport BenchmarkHashEncoded(
        HashEncodedDeepMLP network,
        int sampleCount = 4096,
        Random? random = null)
    {
        random ??= new Random(42);
        var points = SamplePointCloud(sampleCount, random);
        var distances = new float[sampleCount];

        var sw = Stopwatch.StartNew();
        network.EvaluateBatch(points, distances);
        sw.Stop();

        return new ValidationReport
        {
            MemoryBytesFp32 = network.GetMemoryFootprintBytes(),
            InferenceMsPerPoint = sw.Elapsed.TotalMilliseconds / sampleCount,
            BatchThroughputPointsPerMs = sampleCount / Math.Max(sw.Elapsed.TotalMilliseconds, 1e-6),
            SampleCount = sampleCount,
            ElapsedMs = sw.Elapsed.TotalMilliseconds
        };
    }

    /// <summary>
    /// Erreur de Hausdorff approximée entre un réseau et des distances de référence.
    /// </summary>
    public static float ComputeHausdorffError(
        ISdfNetwork network,
        ReadOnlySpan<Vector3> points,
        ReadOnlySpan<float> referenceDistances)
    {
        float maxError = 0f;
        for (int i = 0; i < points.Length; i++)
        {
            float diff = MathF.Abs(network.Evaluate(points[i]) - referenceDistances[i]);
            maxError = MathF.Max(maxError, diff);
        }
        return maxError;
    }

    /// <summary>
    /// Benchmark géométrique contre un maillage de référence (Hausdorff + RMS + mémoire).
    /// </summary>
    public static ValidationReport ValidateAgainstMesh(
        ISdfNetwork network,
        ReferenceMeshSdf mesh,
        int sampleCount = 1024,
        Random? random = null)
    {
        random ??= new Random(42);
        var sw = Stopwatch.StartNew();
        var points = SamplePointCloud(sampleCount, random);

        float hausdorff = mesh.ComputeHausdorffError(network, points);
        float rms = mesh.ComputeRmsError(network, points);

        sw.Stop();
        return new ValidationReport
        {
            HausdorffError = hausdorff,
            RmsError = rms,
            MemoryBytesFp32 = EstimateNetworkMemory(network),
            SampleCount = sampleCount,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            InferenceMsPerPoint = sw.Elapsed.TotalMilliseconds / sampleCount
        };
    }

    private static int EstimateNetworkMemory(ISdfNetwork network) => network switch
    {
        DeepMicroMLP => DeepMicroMLP.GetTotalWeightCount() * sizeof(float),
        HashEncodedDeepMLP hash => hash.GetMemoryFootprintBytes(),
        QuantizedDeepMLP q => q.GetMemoryFootprintBytes(),
        MicroMLP => MicroMLP.TotalWeightCount * sizeof(float),
        _ => 0
    };

    public static Vector3[] SamplePointCloud(int count, Random random)
    {
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
