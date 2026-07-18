using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using GDNN.SIMD;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Réseau compact à encodage par grille de hachage multi-résolution.
/// Architecture : HashEncode(3→16) → Dense(16→32, SiLU) → Dense(32→32, SiLU) → Dense(32→1).
/// La précision locale est portée par les tables de hachage ; le MLP reste minuscule.
/// </summary>
public sealed class HashEncodedDeepMLP : ISdfNetwork, IDisposable
{
    public const int HiddenSize = 32;

    private readonly MultiResolutionHashEncoder _encoder;

    public float[] Layer1Weights { get; private set; }
    public float[] Layer1Bias { get; private set; }
    public float[] Layer2Weights { get; private set; }
    public float[] Layer2Bias { get; private set; }
    public float[] OutputWeights { get; private set; }
    public float OutputBias { get; set; }

    public int EncodedDimension => _encoder.OutputDimension;
    public MultiResolutionHashEncoder Encoder => _encoder;

    /// <summary>Temps d'inférence cumulé (ms) pour profiling.</summary>
    public double InferenceTimeMs { get; private set; }

    public HashEncodedDeepMLP(Random? random = null)
    {
        random ??= Random.Shared;
        _encoder = new MultiResolutionHashEncoder(random);

        int encDim = EncodedDimension;
        Layer1Weights = CreateXavierRandom(encDim, HiddenSize, random);
        Layer1Bias = new float[HiddenSize];
        Layer2Weights = CreateXavierRandom(HiddenSize, HiddenSize, random);
        Layer2Bias = new float[HiddenSize];
        OutputWeights = CreateXavierRandom(HiddenSize, 1, random);
    }

    private HashEncodedDeepMLP(
        MultiResolutionHashEncoder encoder,
        float[] layer1Weights,
        float[] layer1Bias,
        float[] layer2Weights,
        float[] layer2Bias,
        float[] outputWeights,
        float outputBias)
    {
        _encoder = encoder;
        Layer1Weights = layer1Weights;
        Layer1Bias = layer1Bias;
        Layer2Weights = layer2Weights;
        Layer2Bias = layer2Bias;
        OutputWeights = outputWeights;
        OutputBias = outputBias;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Evaluate(Vector3 point)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();

        Span<float> encoded = stackalloc float[EncodedDimension];
        _encoder.Encode(point, encoded);

        Span<float> hidden = stackalloc float[HiddenSize];
        Span<float> hidden2 = stackalloc float[HiddenSize];

        BatchOps.MatVecMultiplyBias(Layer1Weights, encoded, Layer1Bias, hidden, HiddenSize, EncodedDimension);
        BatchOps.ApplySiluInPlace(hidden);

        BatchOps.MatVecMultiplyBias(Layer2Weights, hidden, Layer2Bias, hidden2, HiddenSize, HiddenSize);
        BatchOps.ApplySiluInPlace(hidden2);

        float result = DotProduct(hidden2, OutputWeights) + OutputBias;

        sw.Stop();
        InferenceTimeMs += sw.Elapsed.TotalMilliseconds;
        return result;
    }

    public void EvaluateBatch(ReadOnlySpan<Vector3> points, Span<float> distances)
    {
        if (points.Length != distances.Length)
            throw new ArgumentException("Points and distances must have the same length.");

        var sw = System.Diagnostics.Stopwatch.StartNew();

        Span<float> encoded = stackalloc float[EncodedDimension];
        Span<float> hidden = stackalloc float[HiddenSize];
        Span<float> hidden2 = stackalloc float[HiddenSize];

        for (int i = 0; i < points.Length; i++)
        {
            _encoder.Encode(points[i], encoded);
            BatchOps.MatVecMultiplyBias(Layer1Weights, encoded, Layer1Bias, hidden, HiddenSize, EncodedDimension);
            BatchOps.ApplySiluInPlace(hidden);
            BatchOps.MatVecMultiplyBias(Layer2Weights, hidden, Layer2Bias, hidden2, HiddenSize, HiddenSize);
            BatchOps.ApplySiluInPlace(hidden2);
            distances[i] = DotProduct(hidden2, OutputWeights) + OutputBias;
        }

        sw.Stop();
        InferenceTimeMs += sw.Elapsed.TotalMilliseconds;
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

    public static int GetTotalWeightCount()
    {
        int mlpWeights = EncodedDimensionStatic() * HiddenSize + HiddenSize
                       + HiddenSize * HiddenSize + HiddenSize
                       + HiddenSize + 1;
        return MultiResolutionHashEncoder.GetTotalParameterCount() + mlpWeights;
    }

    /// <summary>Serializes hash tables and MLP weights to a flat float array.</summary>
    public float[] Serialize()
    {
        float[] result = new float[GetTotalWeightCount()];
        int offset = 0;

        float[] hashTables = _encoder.Serialize();
        hashTables.CopyTo(result, offset);
        offset += hashTables.Length;

        Layer1Weights.CopyTo(result, offset);
        offset += Layer1Weights.Length;
        Layer1Bias.CopyTo(result, offset);
        offset += Layer1Bias.Length;
        Layer2Weights.CopyTo(result, offset);
        offset += Layer2Weights.Length;
        Layer2Bias.CopyTo(result, offset);
        offset += Layer2Bias.Length;
        OutputWeights.CopyTo(result, offset);
        offset += OutputWeights.Length;
        result[offset] = OutputBias;

        return result;
    }

    /// <summary>Reconstructs a network from <see cref="Serialize"/> output.</summary>
    public static HashEncodedDeepMLP FromSerialized(ReadOnlySpan<float> weights)
    {
        int expected = GetTotalWeightCount();
        if (weights.Length < expected)
            throw new ArgumentException($"Expected at least {expected} weights, got {weights.Length}.");

        int hashCount = MultiResolutionHashEncoder.GetTotalParameterCount();
        var encoder = new MultiResolutionHashEncoder(weights[..hashCount]);

        int offset = hashCount;
        int encDim = EncodedDimensionStatic();

        var layer1Weights = weights.Slice(offset, encDim * HiddenSize).ToArray();
        offset += encDim * HiddenSize;
        var layer1Bias = weights.Slice(offset, HiddenSize).ToArray();
        offset += HiddenSize;
        var layer2Weights = weights.Slice(offset, HiddenSize * HiddenSize).ToArray();
        offset += HiddenSize * HiddenSize;
        var layer2Bias = weights.Slice(offset, HiddenSize).ToArray();
        offset += HiddenSize;
        var outputWeights = weights.Slice(offset, HiddenSize).ToArray();
        offset += HiddenSize;
        float outputBias = weights[offset];

        return new HashEncodedDeepMLP(
            encoder,
            layer1Weights,
            layer1Bias,
            layer2Weights,
            layer2Bias,
            outputWeights,
            outputBias);
    }

    private static int EncodedDimensionStatic() =>
        MultiResolutionHashEncoder.NumLevels * MultiResolutionHashEncoder.FeaturesPerLevel;

    public int GetMemoryFootprintBytes()
    {
        int hashBytes = MultiResolutionHashEncoder.GetTotalParameterCount() * sizeof(float);
        int mlpBytes = (Layer1Weights.Length + Layer1Bias.Length +
                        Layer2Weights.Length + Layer2Bias.Length +
                        OutputWeights.Length + 1) * sizeof(float);
        return hashBytes + mlpBytes;
    }

    public void Dispose() => GC.SuppressFinalize(this);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static float[] CreateXavierRandom(int fanIn, int fanOut, Random random)
    {
        float scale = MathF.Sqrt(2.0f / fanIn);
        float[] weights = new float[fanIn * fanOut];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(random.NextDouble() * 2.0 - 1.0) * scale;
        return weights;
    }
}
