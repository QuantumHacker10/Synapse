using System;
// ============================================================
// FILE: MicroMLP.cs
// PATH: Core/NeuralNetwork/MicroMLP.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Micro-MLP neural network for continuous surface representation.
/// Architecture: 3D input → 8 hidden → 8 hidden → 1 output (signed distance)
/// </summary>
public sealed class MicroMLP : ISdfNetwork, IDisposable
{
    /// <summary>
    /// Input dimension (3D coordinates).
    /// </summary>
    public const int InputDimension = 3;

    /// <summary>
    /// Hidden layer size.
    /// </summary>
    public const int HiddenSize = NeuralLayerWeights.MaxNeurons;

    /// <summary>
    /// Output dimension (signed distance).
    /// </summary>
    public const int OutputDimension = 1;

    /// <summary>
    /// Number of layers in the network.
    /// </summary>
    public const int LayerCount = 3;

    /// <summary>
    /// Total weight count across all layers.
    /// </summary>
    public const int TotalWeightCount = NeuralLayerWeights.TotalFloatCount * LayerCount;

    /// <summary>
    /// Layer 1 weights: 3 inputs → 8 hidden neurons.
    /// </summary>
    public NeuralLayerWeights Layer1 { get; private set; }

    /// <summary>
    /// Layer 2 weights: 8 hidden → 8 hidden neurons.
    /// </summary>
    public NeuralLayerWeights Layer2 { get; private set; }

    /// <summary>
    /// Layer 3 weights: 8 hidden → 1 output neuron.
    /// </summary>
    public NeuralLayerWeights Layer3 { get; private set; }

    /// <summary>
    /// Activation function type for hidden layers.
    /// </summary>
    public ActivationFunction Activation { get; set; } = ActivationFunction.ReLU;

    /// <summary>
    /// Whether to use SIMD-optimized evaluation.
    /// </summary>
    public bool UseSimd { get; set; } = true;

    /// <summary>
    /// Creates a new MicroMLP with random weights.
    /// </summary>
    public MicroMLP(Random? random = null)
    {
        random ??= Random.Shared;
        Layer1 = NeuralLayerWeights.CreateRandom(InputDimension, random);
        Layer2 = NeuralLayerWeights.CreateRandom(HiddenSize, random);
        Layer3 = NeuralLayerWeights.CreateRandom(HiddenSize, random);
    }

    /// <summary>
    /// Creates a new MicroMLP with specified weights.
    /// </summary>
    public MicroMLP(NeuralLayerWeights layer1, NeuralLayerWeights layer2, NeuralLayerWeights layer3)
    {
        Layer1 = layer1;
        Layer2 = layer2;
        Layer3 = layer3;
    }

    /// <summary>
    /// Creates a MicroMLP from a flat weight array.
    /// </summary>
    public MicroMLP(ReadOnlySpan<float> weights)
    {
        if (weights.Length < TotalWeightCount)
            throw new ArgumentException($"Expected at least {TotalWeightCount} weights.");

        int offset = 0;
        Layer1 = new NeuralLayerWeights(weights.Slice(offset, NeuralLayerWeights.TotalFloatCount));
        offset += NeuralLayerWeights.TotalFloatCount;
        Layer2 = new NeuralLayerWeights(weights.Slice(offset, NeuralLayerWeights.TotalFloatCount));
        offset += NeuralLayerWeights.TotalFloatCount;
        Layer3 = new NeuralLayerWeights(weights.Slice(offset, NeuralLayerWeights.TotalFloatCount));
    }

    /// <summary>
    /// Loads weights from a flat buffer into this network.
    /// </summary>
    public void LoadFrom(ReadOnlySpan<float> weights)
    {
        if (weights.Length < TotalWeightCount)
            throw new ArgumentException($"Expected at least {TotalWeightCount} weights.");

        int offset = 0;
        Layer1 = new NeuralLayerWeights(weights.Slice(offset, NeuralLayerWeights.TotalFloatCount));
        offset += NeuralLayerWeights.TotalFloatCount;
        Layer2 = new NeuralLayerWeights(weights.Slice(offset, NeuralLayerWeights.TotalFloatCount));
        offset += NeuralLayerWeights.TotalFloatCount;
        Layer3 = new NeuralLayerWeights(weights.Slice(offset, NeuralLayerWeights.TotalFloatCount));
    }

    /// <summary>
    /// Evaluates the network at a 3D point, returning signed distance.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Evaluate(Vector3 point)
    {
        Span<float> layer1Output = stackalloc float[HiddenSize];
        Span<float> layer2Output = stackalloc float[HiddenSize];

        EvaluateLayer1(point, layer1Output);
        ApplyActivation(layer1Output);

        EvaluateLayer2(layer1Output, layer2Output);
        ApplyActivation(layer2Output);

        return EvaluateLayer3Output(layer2Output);
    }

    /// <summary>
    /// Evaluates the network with full pipeline including gradient computation.
    /// </summary>
    public float EvaluateWithGradient(Vector3 point, out Vector3 gradient)
    {
        float d = Evaluate(point);
        gradient = ComputeGradient(point);
        return d;
    }

    /// <summary>
    /// Evaluates the network using SIMD-optimized operations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float EvaluateSimd(Vector3 point)
    {
        Span<float> layer1Output = stackalloc float[HiddenSize];
        Span<float> layer2Output = stackalloc float[HiddenSize];

        EvaluateLayer1Simd(point, layer1Output);
        ApplyActivation(layer1Output);

        EvaluateLayer2Simd(layer1Output, layer2Output);
        ApplyActivation(layer2Output);

        return EvaluateLayer3OutputSimd(layer2Output);
    }

    /// <summary>
    /// Batch evaluation of multiple points for improved cache utilization.
    /// </summary>
    public void EvaluateBatch(ReadOnlySpan<Vector3> points, Span<float> distances)
    {
        if (points.Length != distances.Length)
            throw new ArgumentException("Points and distances must have the same length.");

        Span<float> layer1Output = stackalloc float[HiddenSize];
        Span<float> layer2Output = stackalloc float[HiddenSize];

        for (int i = 0; i < points.Length; i++)
        {
            EvaluateLayer1(points[i], layer1Output);
            ApplyActivation(layer1Output);

            EvaluateLayer2(layer1Output, layer2Output);
            ApplyActivation(layer2Output);

            distances[i] = EvaluateLayer3Output(layer2Output);
        }
    }

    /// <summary>
    /// SIMD-optimized batch evaluation.
    /// </summary>
    public unsafe void EvaluateBatchSimd(ReadOnlySpan<Vector3> points, Span<float> distances)
    {
        if (points.Length != distances.Length)
            throw new ArgumentException("Points and distances must have the same length.");

        int vectorSize = Vector<float>.Count;
        Span<float> layer1Output = stackalloc float[HiddenSize];
        Span<float> layer2Output = stackalloc float[HiddenSize];

        fixed (Vector3* pPoints = points)
        fixed (float* pDistances = distances)
        {
            for (int i = 0; i < points.Length; i++)
            {
                EvaluateLayer1(pPoints[i], layer1Output);
                ApplyActivation(layer1Output);

                EvaluateLayer2(layer1Output, layer2Output);
                ApplyActivation(layer2Output);

                pDistances[i] = EvaluateLayer3Output(layer2Output);
            }
        }
    }

    /// <summary>
    /// Computes the gradient (surface normal) via finite differences.
    /// </summary>
    public Vector3 ComputeGradient(Vector3 point)
    {
        const float epsilon = 0.0001f;

        float d = Evaluate(point);
        float dx = Evaluate(new Vector3(point.X + epsilon, point.Y, point.Z)) - d;
        float dy = Evaluate(new Vector3(point.X, point.Y + epsilon, point.Z)) - d;
        float dz = Evaluate(new Vector3(point.X, point.Y, point.Z + epsilon)) - d;

        Vector3 gradient = new Vector3(dx, dy, dz);
        float length = gradient.Length();

        return length > 0 ? gradient / length : Vector3.UnitY;
    }

    /// <summary>
    /// Computes the gradient using central differences for better accuracy.
    /// </summary>
    public Vector3 ComputeGradientCentral(Vector3 point)
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
    /// Computes the Jacobian matrix of the network output with respect to input.
    /// </summary>
    public void ComputeJacobian(Vector3 point, Span<float> jacobian)
    {
        const float epsilon = 0.0001f;

        float d = Evaluate(point);
        float dx = (Evaluate(new Vector3(point.X + epsilon, point.Y, point.Z)) -
                    Evaluate(new Vector3(point.X - epsilon, point.Y, point.Z))) / (2 * epsilon);
        float dy = (Evaluate(new Vector3(point.X, point.Y + epsilon, point.Z)) -
                    Evaluate(new Vector3(point.X, point.Y - epsilon, point.Z))) / (2 * epsilon);
        float dz = (Evaluate(new Vector3(point.X, point.Y, point.Z + epsilon)) -
                    Evaluate(new Vector3(point.X, point.Y, point.Z - epsilon))) / (2 * epsilon);

        jacobian[0] = dx;
        jacobian[1] = dy;
        jacobian[2] = dz;
    }

    /// <summary>
    /// Determines if a ray intersects the surface defined by this SDF.
    /// </summary>
    public bool RayMarch(Vector3 origin, Vector3 direction, float maxDistance, out float distance, out Vector3 hitPoint, out Vector3 normal, int maxSteps = 32)
    {
        distance = 0f;
        hitPoint = origin;
        normal = Vector3.UnitY;

        float totalDistance = 0f;

        for (int step = 0; step < maxSteps; step++)
        {
            Vector3 currentPoint = origin + direction * totalDistance;
            float dist = Evaluate(currentPoint);

            if (Math.Abs(dist) < 0.0005f)
            {
                hitPoint = currentPoint;
                normal = ComputeGradient(currentPoint);
                distance = totalDistance;
                return true;
            }

            totalDistance += dist;

            if (totalDistance > maxDistance)
                break;
        }

        return false;
    }

    /// <summary>
    /// Finds the closest point on the surface to a given point.
    /// </summary>
    public Vector3 FindClosestPoint(Vector3 queryPoint, int maxSteps = 32, float tolerance = 0.0001f)
    {
        Vector3 current = queryPoint;

        for (int step = 0; step < maxSteps; step++)
        {
            float dist = Evaluate(current);
            if (Math.Abs(dist) < tolerance)
                break;

            Vector3 gradient = ComputeGradient(current);
            current -= gradient * dist;
        }

        return current;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EvaluateLayer1(Vector3 input, Span<float> output)
    {
        for (int neuron = 0; neuron < HiddenSize; neuron++)
        {
            float sum = input.X * Layer1[neuron, 0] +
                        input.Y * Layer1[neuron, 1] +
                        input.Z * Layer1[neuron, 2];
            output[neuron] = sum + Layer1.GetBias(neuron);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EvaluateLayer1Simd(Vector3 input, Span<float> output)
    {
        Span<float> inputBuf = stackalloc float[3];
        inputBuf[0] = input.X;
        inputBuf[1] = input.Y;
        inputBuf[2] = input.Z;

        ReadOnlySpan<float> weights = Layer1.AsReadOnlySpan();
        int vectorSize = Vector<float>.Count;

        for (int neuron = 0; neuron < HiddenSize; neuron++)
        {
            float sum = 0f;
            int weightOffset = neuron * NeuralLayerWeights.MaxInputs;

            int i = 0;
            if (vectorSize <= 3)
            {
                var vSum = Vector<float>.Zero;
                var vInput = new Vector<float>(inputBuf);

                for (; i <= 3 - vectorSize; i += vectorSize)
                {
                    var vWeight = new Vector<float>(weights.Slice(weightOffset + i, vectorSize));
                    vSum += vInput * vWeight;
                }

                for (int j = 0; j < vectorSize; j++)
                    sum += vSum[j];
            }

            for (; i < 3; i++)
                sum += inputBuf[i] * weights[weightOffset + i];

            output[neuron] = sum + weights[NeuralLayerWeights.MaxWeightCount + neuron];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EvaluateLayer2(ReadOnlySpan<float> input, Span<float> output)
    {
        for (int neuron = 0; neuron < HiddenSize; neuron++)
        {
            float sum = 0f;
            for (int i = 0; i < HiddenSize; i++)
            {
                sum += input[i] * Layer2[neuron, i];
            }
            output[neuron] = sum + Layer2.GetBias(neuron);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EvaluateLayer2Simd(ReadOnlySpan<float> input, Span<float> output)
    {
        ReadOnlySpan<float> weights = Layer2.AsReadOnlySpan();
        int vectorSize = Vector<float>.Count;

        for (int neuron = 0; neuron < HiddenSize; neuron++)
        {
            float sum = 0f;
            int weightOffset = neuron * NeuralLayerWeights.MaxInputs;

            int i = 0;
            if (vectorSize <= HiddenSize)
            {
                var vSum = Vector<float>.Zero;
                var vInput = new Vector<float>(input.Slice(0, Math.Min(vectorSize, HiddenSize)));

                for (; i <= HiddenSize - vectorSize; i += vectorSize)
                {
                    var vWeight = new Vector<float>(weights.Slice(weightOffset + i, vectorSize));
                    vSum += vInput * vWeight;
                }

                for (int j = 0; j < vectorSize; j++)
                    sum += vSum[j];
            }

            for (; i < HiddenSize; i++)
                sum += input[i] * weights[weightOffset + i];

            output[neuron] = sum + weights[NeuralLayerWeights.MaxWeightCount + neuron];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float EvaluateLayer3Output(ReadOnlySpan<float> input)
    {
        float sum = 0f;
        for (int i = 0; i < HiddenSize; i++)
        {
            sum += input[i] * Layer3[0, i];
        }
        return sum + Layer3.GetBias(0);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float EvaluateLayer3OutputSimd(ReadOnlySpan<float> input)
    {
        ReadOnlySpan<float> weights = Layer3.AsReadOnlySpan();
        float sum = 0f;
        int vectorSize = Vector<float>.Count;
        int i = 0;

        if (vectorSize <= HiddenSize)
        {
            var vSum = Vector<float>.Zero;
            var vInput = new Vector<float>(input.Slice(0, Math.Min(vectorSize, HiddenSize)));

            for (; i <= HiddenSize - vectorSize; i += vectorSize)
            {
                var vWeight = new Vector<float>(weights.Slice(i, vectorSize));
                vSum += vInput * vWeight;
            }

            for (int j = 0; j < vectorSize; j++)
                sum += vSum[j];
        }

        for (; i < HiddenSize; i++)
            sum += input[i] * weights[i];

        return sum + weights[NeuralLayerWeights.MaxWeightCount];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyActivation(Span<float> values)
    {
        switch (Activation)
        {
            case ActivationFunction.ReLU:
                for (int i = 0; i < values.Length; i++)
                    values[i] = Math.Max(0f, values[i]);
                break;

            case ActivationFunction.LeakyReLU:
                for (int i = 0; i < values.Length; i++)
                    values[i] = values[i] > 0 ? values[i] : 0.01f * values[i];
                break;

            case ActivationFunction.Sigmoid:
                for (int i = 0; i < values.Length; i++)
                    values[i] = 1.0f / (1.0f + MathF.Exp(-values[i]));
                break;

            case ActivationFunction.Tanh:
                for (int i = 0; i < values.Length; i++)
                    values[i] = MathF.Tanh(values[i]);
                break;

            case ActivationFunction.SiLU:
                for (int i = 0; i < values.Length; i++)
                    values[i] = values[i] / (1.0f + MathF.Exp(-values[i]));
                break;

            case ActivationFunction.GELU:
                for (int i = 0; i < values.Length; i++)
                {
                    float x = values[i];
                    values[i] = 0.5f * x * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (x + 0.044715f * x * x * x)));
                }
                break;
        }
    }

    /// <summary>
    /// Serializes all weights to a flat float array.
    /// </summary>
    public float[] Serialize()
    {
        float[] result = new float[TotalWeightCount];
        int offset = 0;

        Layer1.AsReadOnlySpan().CopyTo(result.AsSpan(offset));
        offset += NeuralLayerWeights.TotalFloatCount;

        Layer2.AsReadOnlySpan().CopyTo(result.AsSpan(offset));
        offset += NeuralLayerWeights.TotalFloatCount;

        Layer3.AsReadOnlySpan().CopyTo(result.AsSpan(offset));

        return result;
    }

    /// <summary>
    /// Compresses all weights to FP8 format.
    /// </summary>
    public byte[] CompressWeights()
    {
        byte[] result = new byte[TotalWeightCount];
        int offset = 0;

        Layer1.CompressToFP8().CopyTo(result, offset);
        offset += NeuralLayerWeights.TotalFloatCount;

        Layer2.CompressToFP8().CopyTo(result, offset);
        offset += NeuralLayerWeights.TotalFloatCount;

        Layer3.CompressToFP8().CopyTo(result, offset);

        return result;
    }

    /// <summary>
    /// Creates a MicroMLP from compressed FP8 weights.
    /// </summary>
    public static MicroMLP FromCompressedWeights(ReadOnlySpan<byte> compressed)
    {
        if (compressed.Length < TotalWeightCount)
            throw new ArgumentException($"Compressed data must have at least {TotalWeightCount} bytes.");

        int offset = 0;
        var layer1 = NeuralLayerWeights.DecompressFromFP8(compressed.Slice(offset, NeuralLayerWeights.TotalFloatCount));
        offset += NeuralLayerWeights.TotalFloatCount;
        var layer2 = NeuralLayerWeights.DecompressFromFP8(compressed.Slice(offset, NeuralLayerWeights.TotalFloatCount));
        offset += NeuralLayerWeights.TotalFloatCount;
        var layer3 = NeuralLayerWeights.DecompressFromFP8(compressed.Slice(offset, NeuralLayerWeights.TotalFloatCount));

        return new MicroMLP(layer1, layer2, layer3);
    }

    /// <summary>
    /// Creates a deep copy of this network.
    /// </summary>
    public MicroMLP Clone()
    {
        return new MicroMLP(Layer1, Layer2, Layer3)
        {
            Activation = Activation,
            UseSimd = UseSimd
        };
    }

    /// <summary>
    /// Linearly interpolates between two MicroMLPs.
    /// </summary>
    public MicroMLP Lerp(MicroMLP other, float t)
    {
        return new MicroMLP(
            Layer1.Lerp(other.Layer1, t),
            Layer2.Lerp(other.Layer2, t),
            Layer3.Lerp(other.Layer3, t)
        )
        {
            Activation = Activation,
            UseSimd = UseSimd
        };
    }

    /// <summary>
    /// Computes mean squared error against ground truth network.
    /// </summary>
    public float ComputeLoss(MicroMLP groundTruth, int sampleCount = 1000, Random? random = null)
    {
        random ??= Random.Shared;
        float totalLoss = 0f;

        for (int i = 0; i < sampleCount; i++)
        {
            float x = (float)(random.NextDouble() * 2.0 - 1.0);
            float y = (float)(random.NextDouble() * 2.0 - 1.0);
            float z = (float)(random.NextDouble() * 2.0 - 1.0);
            Vector3 point = new Vector3(x, y, z);

            float predicted = Evaluate(point);
            float target = groundTruth.Evaluate(point);
            float diff = predicted - target;
            totalLoss += diff * diff;
        }

        return totalLoss / sampleCount;
    }

    /// <summary>
    /// Generates HLSL shader code for GPU evaluation.
    /// </summary>
    public string GenerateHLSL()
    {
        var sb = new System.Text.StringBuilder();

        sb.AppendLine("// Auto-generated MicroMLP HLSL shader");
        sb.AppendLine("cbuffer NeuralWeights : register(b0)");
        sb.AppendLine("{");
        sb.AppendLine("    float4 Layer1_Weights[6]; // 3×8 weights packed into float4s");
        sb.AppendLine("    float4 Layer2_Weights[16]; // 8×8 weights packed into float4s");
        sb.AppendLine("    float4 Output_Weights[2]; // 8 weights packed into float4s");
        sb.AppendLine("};");
        sb.AppendLine();
        sb.AppendLine("float EvaluateNeuralSDF(float3 localPos)");
        sb.AppendLine("{");
        sb.AppendLine("    float layer1[8];");
        sb.AppendLine("    float layer2[8];");
        sb.AppendLine();
        sb.AppendLine("    [unroll]");
        sb.AppendLine("    for(int i = 0; i < 8; i++)");
        sb.AppendLine("    {");
        sb.AppendLine("        layer1[i] = max(0.0, dot(localPos, Layer1_Weights[i].xyz) + Layer1_Weights[i].w);");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    [unroll]");
        sb.AppendLine("    for(int j = 0; j < 8; j++)");
        sb.AppendLine("    {");
        sb.AppendLine("        float sum = 0.0;");
        sb.AppendLine("        sum += layer1[0] * Layer2_Weights[j * 2].x;");
        sb.AppendLine("        sum += layer1[1] * Layer2_Weights[j * 2].y;");
        sb.AppendLine("        sum += layer1[2] * Layer2_Weights[j * 2].z;");
        sb.AppendLine("        sum += layer1[3] * Layer2_Weights[j * 2].w;");
        sb.AppendLine("        sum += layer1[4] * Layer2_Weights[j * 2 + 1].x;");
        sb.AppendLine("        sum += layer1[5] * Layer2_Weights[j * 2 + 1].y;");
        sb.AppendLine("        sum += layer1[6] * Layer2_Weights[j * 2 + 1].z;");
        sb.AppendLine("        sum += layer1[7] * Layer2_Weights[j * 2 + 1].w;");
        sb.AppendLine("        layer2[j] = max(0.0, sum);");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    float distance = 0.0;");
        sb.AppendLine("    distance += layer2[0] * Output_Weights[0].x;");
        sb.AppendLine("    distance += layer2[1] * Output_Weights[0].y;");
        sb.AppendLine("    distance += layer2[2] * Output_Weights[0].z;");
        sb.AppendLine("    distance += layer2[3] * Output_Weights[0].w;");
        sb.AppendLine("    distance += layer2[4] * Output_Weights[1].x;");
        sb.AppendLine("    distance += layer2[5] * Output_Weights[1].y;");
        sb.AppendLine("    distance += layer2[6] * Output_Weights[1].z;");
        sb.AppendLine("    distance += layer2[7] * Output_Weights[1].w;");
        sb.AppendLine();
        sb.AppendLine("    return distance;");
        sb.AppendLine("}");

        return sb.ToString();
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Copies all weights into the provided buffer.
    /// </summary>
    public void CopyTo(Span<float> destination)
    {
        Debug.Assert(destination.Length >= TotalWeightCount);
        int offset = 0;
        Layer1.AsReadOnlySpan().CopyTo(destination.Slice(offset));
        offset += NeuralLayerWeights.TotalFloatCount;
        Layer2.AsReadOnlySpan().CopyTo(destination.Slice(offset));
        offset += NeuralLayerWeights.TotalFloatCount;
        Layer3.AsReadOnlySpan().CopyTo(destination.Slice(offset));
    }

    /// <summary>
    /// Returns all weights as a flat float array.
    /// </summary>
    public float[] AllWeights()
    {
        float[] result = new float[TotalWeightCount];
        CopyTo(result);
        return result;
    }
}

/// <summary>
/// Activation functions supported by MicroMLP.
/// </summary>
public enum ActivationFunction
{
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    SiLU,
    GELU
}
