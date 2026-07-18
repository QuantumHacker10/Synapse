using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Version quantifiée int8 d'un DeepMicroMLP entraîné, pour l'inférence seule.
/// Le facteur d'échelle (scale) est calculé une fois après entraînement.
/// </summary>
public sealed class QuantizedDeepMLP : ISdfNetwork
{
    private readonly sbyte[] _layer1WeightsQ;
    private readonly float _layer1Scale;
    private readonly sbyte[] _layer1BiasQ;
    private readonly float _layer1BiasScale;

    private readonly sbyte[][] _residualWeightsQ;
    private readonly float[] _residualScales;
    private readonly sbyte[][] _residualBiasesQ;
    private readonly float[] _residualBiasScales;

    private readonly float[][] _layerNormGamma;
    private readonly float[][] _layerNormBeta;

    private readonly sbyte[] _outputWeightsQ;
    private readonly float _outputScale;
    private readonly float _outputBias;

    public float PositionalScale { get; }

    /// <summary>Temps d'inférence cumulé (ms) pour profiling.</summary>
    public double InferenceTimeMs { get; private set; }

    public static QuantizedDeepMLP FromTrained(DeepMicroMLP source)
    {
        var (layer1W, layer1S) = QuantizeSymmetric(source.Layer1Weights);
        var (layer1B, layer1BS) = QuantizeSymmetric(source.Layer1Bias);

        var resW = new sbyte[DeepMicroMLP.ResidualBlockCount][];
        var resS = new float[DeepMicroMLP.ResidualBlockCount];
        var resBQ = new sbyte[DeepMicroMLP.ResidualBlockCount][];
        var resBSS = new float[DeepMicroMLP.ResidualBlockCount];

        for (int b = 0; b < DeepMicroMLP.ResidualBlockCount; b++)
        {
            (resW[b], resS[b]) = QuantizeSymmetric(source.ResidualWeights[b]);
            (resBQ[b], resBSS[b]) = QuantizeSymmetric(source.ResidualBiases[b]);
        }

        var (outW, outS) = QuantizeSymmetric(source.OutputWeights);

        return new QuantizedDeepMLP(
            layer1W, layer1S, layer1B, layer1BS,
            resW, resS, resBQ, resBSS,
            source.LayerNormGamma, source.LayerNormBeta,
            outW, outS, source.OutputBias[0],
            source.PositionalScale);
    }

    private QuantizedDeepMLP(
        sbyte[] layer1WeightsQ, float layer1Scale,
        sbyte[] layer1BiasQ, float layer1BiasScale,
        sbyte[][] residualWeightsQ, float[] residualScales,
        sbyte[][] residualBiasesQ, float[] residualBiasScales,
        float[][] layerNormGamma, float[][] layerNormBeta,
        sbyte[] outputWeightsQ, float outputScale, float outputBias,
        float positionalScale)
    {
        _layer1WeightsQ = layer1WeightsQ;
        _layer1Scale = layer1Scale;
        _layer1BiasQ = layer1BiasQ;
        _layer1BiasScale = layer1BiasScale;
        _residualWeightsQ = residualWeightsQ;
        _residualScales = residualScales;
        _residualBiasesQ = residualBiasesQ;
        _residualBiasScales = residualBiasScales;
        _layerNormGamma = layerNormGamma;
        _layerNormBeta = layerNormBeta;
        _outputWeightsQ = outputWeightsQ;
        _outputScale = outputScale;
        _outputBias = outputBias;
        PositionalScale = positionalScale;
    }

    public static (sbyte[] quantized, float scale) QuantizeSymmetric(ReadOnlySpan<float> weights)
    {
        float maxAbs = 1e-8f;
        foreach (var w in weights)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(w));

        float scale = maxAbs / 127f;
        var quantized = new sbyte[weights.Length];
        for (int i = 0; i < weights.Length; i++)
            quantized[i] = (sbyte)Math.Clamp(MathF.Round(weights[i] / scale), -127, 127);

        return (quantized, scale);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Evaluate(Vector3 point)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();

        Span<float> encoded = stackalloc float[DeepMicroMLP.EncodedDimension];
        DeepMicroMLP.PositionalEncode(point, encoded, PositionalScale);

        Span<float> hidden = stackalloc float[DeepMicroMLP.HiddenSize];
        Span<float> hidden2 = stackalloc float[DeepMicroMLP.HiddenSize];
        Span<float> residual = stackalloc float[DeepMicroMLP.HiddenSize];

        QuantizedMatVecWithBias(_layer1WeightsQ, _layer1Scale, encoded,
            _layer1BiasQ, _layer1BiasScale, hidden,
            DeepMicroMLP.HiddenSize, DeepMicroMLP.EncodedDimension);
        ApplySiLU(hidden);

        for (int b = 0; b < DeepMicroMLP.ResidualBlockCount; b++)
        {
            hidden.CopyTo(residual);
            LayerNorm(hidden, _layerNormGamma[b], _layerNormBeta[b]);
            QuantizedMatVecWithBias(_residualWeightsQ[b], _residualScales[b], hidden,
                _residualBiasesQ[b], _residualBiasScales[b], hidden2,
                DeepMicroMLP.HiddenSize, DeepMicroMLP.HiddenSize);
            ApplySiLU(hidden2);
            for (int i = 0; i < DeepMicroMLP.HiddenSize; i++)
                hidden[i] = hidden2[i] + residual[i];
        }

        float result = QuantizedDotProduct(hidden, _outputWeightsQ, _outputScale) + _outputBias;

        sw.Stop();
        InferenceTimeMs += sw.Elapsed.TotalMilliseconds;
        return result;
    }

    public void EvaluateBatch(ReadOnlySpan<Vector3> points, Span<float> distances)
    {
        for (int i = 0; i < points.Length; i++)
            distances[i] = Evaluate(points[i]);
    }

    public float EvaluateWithGradient(Vector3 point, out Vector3 gradient)
    {
        float d = Evaluate(point);
        gradient = ComputeGradient(point);
        return d;
    }

    public Vector3 ComputeGradient(Vector3 point)
    {
        const float epsilon = 0.0001f;
        float dx = (Evaluate(new Vector3(point.X + epsilon, point.Y, point.Z)) -
                    Evaluate(new Vector3(point.X - epsilon, point.Y, point.Z))) / (2 * epsilon);
        float dy = (Evaluate(new Vector3(point.X, point.Y + epsilon, point.Z)) -
                    Evaluate(new Vector3(point.X, point.Y - epsilon, point.Z))) / (2 * epsilon);
        float dz = (Evaluate(new Vector3(point.X, point.Y, point.Z + epsilon)) -
                    Evaluate(new Vector3(point.X, point.Y, point.Z - epsilon))) / (2 * epsilon);

        Vector3 gradient = new Vector3(dx, dy, dz);
        float length = gradient.Length();
        return length > 0 ? gradient / length : Vector3.UnitY;
    }

    /// <summary>
    /// Erreur RMS entre l'inférence quantifiée et fp32 sur un jeu de points.
    /// </summary>
    public float MeasureQuantizationError(DeepMicroMLP reference, ReadOnlySpan<Vector3> testPoints)
    {
        float sumSq = 0f;
        for (int i = 0; i < testPoints.Length; i++)
        {
            float diff = Evaluate(testPoints[i]) - reference.Evaluate(testPoints[i]);
            sumSq += diff * diff;
        }
        return MathF.Sqrt(sumSq / testPoints.Length);
    }

    public int GetMemoryFootprintBytes()
    {
        int qBytes = _layer1WeightsQ.Length + _layer1BiasQ.Length + _outputWeightsQ.Length;
        for (int b = 0; b < DeepMicroMLP.ResidualBlockCount; b++)
            qBytes += _residualWeightsQ[b].Length + _residualBiasesQ[b].Length;
        return qBytes + sizeof(float) * (4 + DeepMicroMLP.ResidualBlockCount * 2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void QuantizedMatVecWithBias(
        ReadOnlySpan<sbyte> weightsQ, float weightScale,
        ReadOnlySpan<float> input,
        ReadOnlySpan<sbyte> biasQ, float biasScale,
        Span<float> output, int outSize, int inSize)
    {
        for (int o = 0; o < outSize; o++)
        {
            float sum = biasQ[o] * biasScale;
            int rowStart = o * inSize;
            for (int i = 0; i < inSize; i++)
                sum += weightsQ[rowStart + i] * input[i] * weightScale;
            output[o] = sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float QuantizedDotProduct(ReadOnlySpan<float> input, ReadOnlySpan<sbyte> weightsQ, float scale)
    {
        float sum = 0f;
        for (int i = 0; i < input.Length; i++)
            sum += input[i] * weightsQ[i] * scale;
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplySiLU(Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            float x = values[i];
            values[i] = x / (1.0f + MathF.Exp(-x));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LayerNorm(Span<float> values, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta)
    {
        int length = values.Length;
        float mean = 0f;
        for (int i = 0; i < length; i++) mean += values[i];
        mean /= length;

        float variance = 0f;
        for (int i = 0; i < length; i++)
        {
            float diff = values[i] - mean;
            variance += diff * diff;
        }
        variance /= length;

        float invStd = 1.0f / MathF.Sqrt(variance + 1e-5f);
        for (int i = 0; i < length; i++)
            values[i] = (values[i] - mean) * invStd * gamma[i] + beta[i];
    }
}
