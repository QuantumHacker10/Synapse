using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;


// ============================================================
// FILE: NeuralLayerWeights.cs
// PATH: Core/NeuralNetwork/NeuralLayerWeights.cs
// ============================================================


using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Numerics;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Represents the weights of a single layer in the Micro-MLP neural network.
/// Aligned on 64 bytes for optimal CPU cache efficiency (L1/L2) and SIMD access.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 64)]
public readonly struct NeuralLayerWeights : IEquatable<NeuralLayerWeights>
{
    /// <summary>
    /// Maximum number of neurons per layer in the Micro-MLP architecture.
    /// </summary>
    public const int MaxNeurons = 8;

    /// <summary>
    /// Maximum number of inputs per layer.
    /// </summary>
    public const int MaxInputs = 8;

    /// <summary>
    /// Total weight count: 8 neurons × 8 inputs = 64 weights maximum.
    /// </summary>
    public const int MaxWeightCount = MaxNeurons * MaxInputs;

    /// <summary>
    /// Bias terms for each neuron in this layer.
    /// </summary>
    public const int BiasCount = MaxNeurons;

    /// <summary>
    /// Total size in floats: weights + biases.
    /// </summary>
    public const int TotalFloatCount = MaxWeightCount + BiasCount;

    /// <summary>
    /// Size in bytes for this structure.
    /// </summary>
    public const int SizeInBytes = TotalFloatCount * sizeof(float);

    /// <summary>
    /// Raw weight storage, aligned for SIMD operations.
    /// </summary>
    private readonly float _w00, _w01, _w02, _w03, _w04, _w05, _w06, _w07;
    private readonly float _w10, _w11, _w12, _w13, _w14, _w15, _w16, _w17;
    private readonly float _w20, _w21, _w22, _w23, _w24, _w25, _w26, _w27;
    private readonly float _w30, _w31, _w32, _w33, _w34, _w35, _w36, _w37;
    private readonly float _w40, _w41, _w42, _w43, _w44, _w45, _w46, _w47;
    private readonly float _w50, _w51, _w52, _w53, _w54, _w55, _w56, _w57;
    private readonly float _w60, _w61, _w62, _w63, _w64, _w65, _w66, _w67;
    private readonly float _w70, _w71, _w72, _w73, _w74, _w75, _w76, _w77;

    /// <summary>
    /// Bias terms for each output neuron.
    /// </summary>
    private readonly float _b0, _b1, _b2, _b3, _b4, _b5, _b6, _b7;

    /// <summary>
    /// Creates a new NeuralLayerWeights from a span of floats.
    /// </summary>
    /// <param name="weights">Weight values (must be at least TotalFloatCount elements).</param>
    public NeuralLayerWeights(ReadOnlySpan<float> weights)
    {
        if (weights.Length < TotalFloatCount)
            throw new ArgumentException($"Expected at least {TotalFloatCount} weights, got {weights.Length}.");

        _w00 = weights[0]; _w01 = weights[1]; _w02 = weights[2]; _w03 = weights[3];
        _w04 = weights[4]; _w05 = weights[5]; _w06 = weights[6]; _w07 = weights[7];
        _w10 = weights[8]; _w11 = weights[9]; _w12 = weights[10]; _w13 = weights[11];
        _w14 = weights[12]; _w15 = weights[13]; _w16 = weights[14]; _w17 = weights[15];
        _w20 = weights[16]; _w21 = weights[17]; _w22 = weights[18]; _w23 = weights[19];
        _w24 = weights[20]; _w25 = weights[21]; _w26 = weights[22]; _w27 = weights[23];
        _w30 = weights[24]; _w31 = weights[25]; _w32 = weights[26]; _w33 = weights[27];
        _w34 = weights[28]; _w35 = weights[29]; _w36 = weights[30]; _w37 = weights[31];
        _w40 = weights[32]; _w41 = weights[33]; _w42 = weights[34]; _w43 = weights[35];
        _w44 = weights[36]; _w45 = weights[37]; _w46 = weights[38]; _w47 = weights[39];
        _w50 = weights[40]; _w51 = weights[41]; _w52 = weights[42]; _w53 = weights[43];
        _w54 = weights[44]; _w55 = weights[45]; _w56 = weights[46]; _w57 = weights[47];
        _w60 = weights[48]; _w61 = weights[49]; _w62 = weights[50]; _w63 = weights[51];
        _w64 = weights[52]; _w65 = weights[53]; _w66 = weights[54]; _w67 = weights[55];
        _w70 = weights[56]; _w71 = weights[57]; _w72 = weights[58]; _w73 = weights[59];
        _w74 = weights[60]; _w75 = weights[61]; _w76 = weights[62]; _w77 = weights[63];

        _b0 = weights[64]; _b1 = weights[65]; _b2 = weights[66]; _b3 = weights[67];
        _b4 = weights[68]; _b5 = weights[69]; _b6 = weights[70]; _b7 = weights[71];
    }

    /// <summary>
    /// Gets weight at specified neuron and input index.
    /// </summary>
    public float this[int neuron, int input]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            int index = neuron * MaxInputs + input;
            return GetWeightAt(index);
        }
    }

    /// <summary>
    /// Gets bias for specified neuron.
    /// </summary>
    public float GetBias(int neuron)
    {
        return neuron switch
        {
            0 => _b0,
            1 => _b1,
            2 => _b2,
            3 => _b3,
            4 => _b4,
            5 => _b5,
            6 => _b6,
            7 => _b7,
            _ => throw new ArgumentOutOfRangeException(nameof(neuron))
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float GetWeightAt(int index)
    {
        return index switch
        {
            0 => _w00, 1 => _w01, 2 => _w02, 3 => _w03, 4 => _w04, 5 => _w05, 6 => _w06, 7 => _w07,
            8 => _w10, 9 => _w11, 10 => _w12, 11 => _w13, 12 => _w14, 13 => _w15, 14 => _w16, 15 => _w17,
            16 => _w20, 17 => _w21, 18 => _w22, 19 => _w23, 20 => _w24, 21 => _w25, 22 => _w26, 23 => _w27,
            24 => _w30, 25 => _w31, 26 => _w32, 27 => _w33, 28 => _w34, 29 => _w35, 30 => _w36, 31 => _w37,
            32 => _w40, 33 => _w41, 34 => _w42, 35 => _w43, 36 => _w44, 37 => _w45, 38 => _w46, 39 => _w47,
            40 => _w50, 41 => _w51, 42 => _w52, 43 => _w53, 44 => _w54, 45 => _w55, 46 => _w56, 47 => _w57,
            48 => _w60, 49 => _w61, 50 => _w62, 51 => _w63, 52 => _w64, 53 => _w65, 54 => _w66, 55 => _w67,
            56 => _w70, 57 => _w71, 58 => _w72, 59 => _w73, 60 => _w74, 61 => _w75, 62 => _w76, 63 => _w77,
            _ => throw new ArgumentOutOfRangeException(nameof(index))
        };
    }

    /// <summary>
    /// Creates a read-only span view of the weights data.
    /// </summary>
    public unsafe ReadOnlySpan<float> AsReadOnlySpan()
    {
        fixed (float* ptr = &_w00)
        {
            return new ReadOnlySpan<float>(ptr, TotalFloatCount);
        }
    }

    /// <summary>
    /// Creates a new NeuralLayerWeights with random initialization (Xavier/Glorot).
    /// </summary>
    public static NeuralLayerWeights CreateRandom(int inputCount, Random? random = null)
    {
        random ??= Random.Shared;
        float scale = MathF.Sqrt(2.0f / inputCount);

        Span<float> data = stackalloc float[TotalFloatCount];
        for (int i = 0; i < MaxWeightCount; i++)
        {
            data[i] = (float)(random.NextDouble() * 2.0 - 1.0) * scale;
        }
        for (int i = 0; i < BiasCount; i++)
        {
            data[MaxWeightCount + i] = 0f;
        }
        return new NeuralLayerWeights(data);
    }

    /// <summary>
    /// Creates a new NeuralLayerWeights initialized to zeros.
    /// </summary>
    public static NeuralLayerWeights CreateZero()
    {
        Span<float> data = stackalloc float[TotalFloatCount];
        data.Clear();
        return new NeuralLayerWeights(data);
    }

    /// <summary>
    /// Creates a new NeuralLayerWeights with uniform values.
    /// </summary>
    public static NeuralLayerWeights CreateUniform(float value)
    {
        Span<float> data = stackalloc float[TotalFloatCount];
        data.Fill(value);
        return new NeuralLayerWeights(data);
    }

    /// <summary>
    /// Performs matrix-vector multiplication for this layer: output = weights × input + bias.
    /// </summary>
    /// <param name="input">Input vector (must have MaxInputs elements).</param>
    /// <param name="output">Output buffer (must have MaxNeurons elements).</param>
    public unsafe void Evaluate(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length < MaxInputs)
            throw new ArgumentException($"Input must have at least {MaxInputs} elements.");
        if (output.Length < MaxNeurons)
            throw new ArgumentException($"Output must have at least {MaxNeurons} elements.");

        fixed (float* pInput = input)
        fixed (float* pOutput = output)
        fixed (float* pWeights = &_w00)
        {
            for (int neuron = 0; neuron < MaxNeurons; neuron++)
            {
                float sum = 0f;
                int weightOffset = neuron * MaxInputs;

                for (int inputIdx = 0; inputIdx < MaxInputs; inputIdx++)
                {
                    sum += pInput[inputIdx] * pWeights[weightOffset + inputIdx];
                }

                pOutput[neuron] = sum + pWeights[MaxWeightCount + neuron];
            }
        }
    }

    /// <summary>
    /// Performs SIMD-optimized matrix-vector multiplication using Vector<T>.
    /// </summary>
    public unsafe void EvaluateSimd(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length < MaxInputs)
            throw new ArgumentException($"Input must have at least {MaxInputs} elements.");
        if (output.Length < MaxNeurons)
            throw new ArgumentException($"Output must have at least {MaxNeurons} elements.");

        int vectorSize = Vector<float>.Count;

        fixed (float* pInput = input)
        fixed (float* pOutput = output)
        fixed (float* pWeights = &_w00)
        {
            for (int neuron = 0; neuron < MaxNeurons; neuron++)
            {
                float sum = 0f;
                int weightOffset = neuron * MaxInputs;

                int i = 0;
                if (vectorSize <= MaxInputs)
                {
                    var vSum = Vector<float>.Zero;
                    var vInput = Unsafe.Read<Vector<float>>(pInput);

                    for (; i <= MaxInputs - vectorSize; i += vectorSize)
                    {
                        var vWeight = Unsafe.Read<Vector<float>>(pWeights + weightOffset + i);
                        vSum += vInput * vWeight;
                    }

                    for (int j = 0; j < vectorSize; j++)
                    {
                        sum += vSum[j];
                    }
                }

                for (; i < MaxInputs; i++)
                {
                    sum += pInput[i] * pWeights[weightOffset + i];
                }

                pOutput[neuron] = sum + pWeights[MaxWeightCount + neuron];
            }
        }
    }

    /// <summary>
    /// Compresses weights to FP8 format for GPU transfer.
    /// </summary>
    public unsafe byte[] CompressToFP8()
    {
        byte[] result = new byte[TotalFloatCount];
        fixed (float* pWeights = &_w00)
        fixed (byte* pResult = result)
        {
            for (int i = 0; i < TotalFloatCount; i++)
            {
                float normalized = (pWeights[i] + 1.0f) * 0.5f;
                pResult[i] = (byte)Math.Clamp((int)(normalized * 255.0f), 0, 255);
            }
        }
        return result;
    }

    /// <summary>
    /// Decompresses FP8 weights back to FP32.
    /// </summary>
    public static NeuralLayerWeights DecompressFromFP8(ReadOnlySpan<byte> compressed)
    {
        if (compressed.Length < TotalFloatCount)
            throw new ArgumentException($"Compressed data must have at least {TotalFloatCount} bytes.");

        Span<float> data = stackalloc float[TotalFloatCount];
        for (int i = 0; i < TotalFloatCount; i++)
        {
            data[i] = compressed[i] / 255.0f * 2.0f - 1.0f;
        }
        return new NeuralLayerWeights(data);
    }

    /// <summary>
    /// Linearly interpolates between two weight sets.
    /// </summary>
    public NeuralLayerWeights Lerp(NeuralLayerWeights other, float t)
    {
        Span<float> result = stackalloc float[TotalFloatCount];
        ReadOnlySpan<float> self = AsReadOnlySpan();
        ReadOnlySpan<float> otherSpan = other.AsReadOnlySpan();

        for (int i = 0; i < TotalFloatCount; i++)
        {
            result[i] = self[i] + (otherSpan[i] - self[i]) * t;
        }
        return new NeuralLayerWeights(result);
    }

    /// <summary>
    /// Computes the mean squared error between two weight sets.
    /// </summary>
    public float MeanSquaredError(NeuralLayerWeights other)
    {
        ReadOnlySpan<float> selfSpan = AsReadOnlySpan();
        ReadOnlySpan<float> otherSpan = other.AsReadOnlySpan();

        float sum = 0f;
        for (int i = 0; i < TotalFloatCount; i++)
        {
            float diff = selfSpan[i] - otherSpan[i];
            sum += diff * diff;
        }
        return sum / TotalFloatCount;
    }

    /// <summary>
    /// Computes L2 norm of all weights.
    /// </summary>
    public float L2Norm()
    {
        ReadOnlySpan<float> span = AsReadOnlySpan();
        float sum = 0f;
        for (int i = 0; i < TotalFloatCount; i++)
        {
            sum += span[i] * span[i];
        }
        return MathF.Sqrt(sum);
    }

    /// <summary>
    /// Applies weight decay regularization.
    /// </summary>
    public NeuralLayerWeights ApplyWeightDecay(float decayRate)
    {
        Span<float> result = stackalloc float[TotalFloatCount];
        ReadOnlySpan<float> span = AsReadOnlySpan();

        for (int i = 0; i < TotalFloatCount; i++)
        {
            result[i] = span[i] * (1.0f - decayRate);
        }
        return new NeuralLayerWeights(result);
    }

    /// <summary>
    /// Applies gradient descent update: weights = weights - learningRate * gradient.
    /// </summary>
    public NeuralLayerWeights ApplyGradient(NeuralLayerWeights gradient, float learningRate)
    {
        Span<float> result = stackalloc float[TotalFloatCount];
        ReadOnlySpan<float> selfSpan = AsReadOnlySpan();
        ReadOnlySpan<float> gradSpan = gradient.AsReadOnlySpan();

        for (int i = 0; i < TotalFloatCount; i++)
        {
            result[i] = selfSpan[i] - learningRate * gradSpan[i];
        }
        return new NeuralLayerWeights(result);
    }

    /// <summary>
    /// Adds another weight set to this one (used for gradient accumulation).
    /// </summary>
    public NeuralLayerWeights Add(NeuralLayerWeights other)
    {
        Span<float> result = stackalloc float[TotalFloatCount];
        ReadOnlySpan<float> selfSpan = AsReadOnlySpan();
        ReadOnlySpan<float> otherSpan = other.AsReadOnlySpan();

        for (int i = 0; i < TotalFloatCount; i++)
        {
            result[i] = selfSpan[i] + otherSpan[i];
        }
        return new NeuralLayerWeights(result);
    }

    /// <summary>
    /// Scales all weights by a scalar value.
    /// </summary>
    public NeuralLayerWeights Scale(float scalar)
    {
        Span<float> result = stackalloc float[TotalFloatCount];
        ReadOnlySpan<float> span = AsReadOnlySpan();

        for (int i = 0; i < TotalFloatCount; i++)
        {
            result[i] = span[i] * scalar;
        }
        return new NeuralLayerWeights(result);
    }

    public bool Equals(NeuralLayerWeights other)
    {
        ReadOnlySpan<float> selfSpan = AsReadOnlySpan();
        ReadOnlySpan<float> otherSpan = other.AsReadOnlySpan();

        for (int i = 0; i < TotalFloatCount; i++)
        {
            if (selfSpan[i] != otherSpan[i]) return false;
        }
        return true;
    }

    public override bool Equals(object? obj) => obj is NeuralLayerWeights other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(_w00, _w33, _w77, _b0, _b7);

    public static bool operator ==(NeuralLayerWeights left, NeuralLayerWeights right) => left.Equals(right);
    public static bool operator !=(NeuralLayerWeights left, NeuralLayerWeights right) => !left.Equals(right);

    public static NeuralLayerWeights operator +(NeuralLayerWeights left, NeuralLayerWeights right) => left.Add(right);
    public static NeuralLayerWeights operator *(NeuralLayerWeights left, float scalar) => left.Scale(scalar);
    public static NeuralLayerWeights operator *(float scalar, NeuralLayerWeights right) => right.Scale(scalar);
}
