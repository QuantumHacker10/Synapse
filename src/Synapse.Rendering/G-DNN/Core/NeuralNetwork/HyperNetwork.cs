using System;
// ============================================================
// FILE: HyperNetwork.cs
// PATH: Core/NeuralNetwork/HyperNetwork.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers;
using System.Buffers.Binary;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
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
using System.Threading;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks;
using GDNN.Rendering.Compat;

namespace GDNN.Core.NeuralNetwork;

// NOTE: NeuralLayerWeights, MicroMLP, and related types are defined in their dedicated
// source files (NeuralLayerWeights.cs, MicroMLP.cs) later in this merged compilation unit.
// The HyperNetwork uses the class-based MicroMLP from MicroMLP.cs.

/// <summary>
/// Represents a 16-dimensional geometry descriptor (Z vector) that encodes
/// the essential characteristics of a 3D mesh patch for weight generation.
/// </summary>
[StructLayout(LayoutKind.Sequential, Size = 16 * sizeof(float))]
public struct GeometryDescriptor
{
    public const int Dimension = 16;

    public float V00, V01, V02, V03;
    public float V04, V05, V06, V07;
    public float V08, V09, V10, V11;
    public float V12, V13, V14, V15;

    /// <summary>
    /// Provides a Span&lt;float&gt; view over the 16 descriptor values.
    /// </summary>
    public Span<float> AsSpan()
    {
        return MemoryMarshal.CreateSpan(ref V00, Dimension);
    }

    /// <summary>
    /// Provides a read-only span over the 16 descriptor values.
    /// </summary>
    public readonly ReadOnlySpan<float> AsReadOnlySpan()
    {
        return MemoryMarshal.CreateReadOnlySpan(ref Unsafe.AsRef(in V00), Dimension);
    }

    /// <summary>
    /// Computes the L2 norm of the descriptor vector.
    /// </summary>
    public readonly float Norm()
    {
        return MathF.Sqrt(V00 * V00 + V01 * V01 + V02 * V02 + V03 * V03 +
                          V04 * V04 + V05 * V05 + V06 * V06 + V07 * V07 +
                          V08 * V08 + V09 * V09 + V10 * V10 + V11 * V11 +
                          V12 * V12 + V13 * V13 + V14 * V14 + V15 * V15);
    }

    /// <summary>
    /// Normalizes the descriptor to unit length in-place.
    /// </summary>
    public void Normalize()
    {
        float norm = Norm();
        if (norm < 1e-8f)
            return;
        float inv = 1.0f / norm;
        Span<float> s = AsSpan();
        for (int i = 0; i < Dimension; i++)
            s[i] *= inv;
    }

    /// <summary>
    /// Serializes the descriptor to a byte span.
    /// </summary>
    public readonly void Serialize(Span<byte> destination)
    {
        MemoryMarshal.AsBytes(AsReadOnlySpan()).CopyTo(destination);
    }

    /// <summary>
    /// Deserializes a descriptor from a byte span.
    /// </summary>
    public static GeometryDescriptor Deserialize(ReadOnlySpan<byte> source)
    {
        var desc = new GeometryDescriptor();
        MemoryMarshal.AsBytes(desc.AsSpan()).CopyFrom(source);
        return desc;
    }

    /// <summary>
    /// Encodes a GeometryDescriptor from raw vertex data (positions and normals).
    /// </summary>
    public static GeometryDescriptor Encode(ReadOnlySpan<float> positions, ReadOnlySpan<float> normals)
    {
        var desc = new GeometryDescriptor();
        var s = desc.AsSpan();

        int posCount = Math.Min(positions.Length, 6);
        for (int i = 0; i < posCount; i++)
            s[i] = positions[i];

        int normCount = Math.Min(normals.Length, 6);
        for (int i = 0; i < normCount; i++)
            s[6 + i] = normals[i];

        float area = 0f;
        for (int i = 0; i < positions.Length - 3; i += 3)
        {
            float dx = positions[i + 3] - positions[i];
            float dy = positions[i + 4] - positions[i + 1];
            float dz = positions[i + 5] - positions[i + 2];
            area += MathF.Sqrt(dx * dx + dy * dy + dz * dz);
        }
        s[12] = area;
        s[13] = positions.Length / 3f;

        float curvature = 0f;
        for (int i = 0; i < normals.Length - 3; i += 3)
        {
            float dx = normals[i + 3] - normals[i];
            float dy = normals[i + 4] - normals[i + 1];
            float dz = normals[i + 5] - normals[i + 2];
            curvature += dx * dx + dy * dy + dz * dz;
        }
        s[14] = curvature;

        s[15] = MathF.Min(posCount / 3f, 8f);

        return desc;
    }
}

/// <summary>
/// HyperNetwork that generates MicroMLP weights from compact 16-dimensional geometry descriptors.
/// Uses a learned mapping Z → weights to produce neural geometry representations.
/// Architecture: Z(16) → Dense(64, ReLU) → Dense(128, ReLU) → Dense(256, ReLU)
///              → Dense(128, ReLU) → Dense(64, ReLU) → Output(216, Linear).
/// </summary>
public sealed class HyperNetwork
{
    /// <summary>Input dimension of the geometry descriptor.</summary>
    public const int ZDimension = GeometryDescriptor.Dimension; // 16

    /// <summary>Output dimension matching MicroMLP.TotalWeightCount.</summary>
    public const int OutputDimension = MicroMLP.TotalWeightCount; // 216

    private readonly float[] _hyperWeights;
    private readonly float[] _hyperBiases;

    private const int Hidden0 = 64;
    private const int Hidden1 = 128;
    private const int Hidden2 = 256;
    private const int Hidden3 = 128;
    private const int Hidden4 = 64;

    private static readonly int[] _layerSizes = { ZDimension, Hidden0, Hidden1, Hidden2, Hidden3, Hidden4, OutputDimension };

    private static readonly int[] _weightCounts;
    private static readonly int[] _biasOffsets;
    private static readonly int _totalParams;

    static HyperNetwork()
    {
        _weightCounts = new int[_layerSizes.Length - 1];
        _biasOffsets = new int[_layerSizes.Length - 1];
        int total = 0;
        for (int i = 0; i < _layerSizes.Length - 1; i++)
        {
            _weightCounts[i] = _layerSizes[i] * _layerSizes[i + 1];
            _biasOffsets[i] = total + _weightCounts[i];
            total += _weightCounts[i] + _layerSizes[i + 1];
        }
        _totalParams = total;
    }

    /// <summary>
    /// Initializes the HyperNetwork with randomly sampled Xavier-normal weights.
    /// </summary>
    /// <param name="seed">Random seed for reproducibility.</param>
    public HyperNetwork(int seed = 42)
    {
        _hyperWeights = new float[_totalParams];
        var rng = new Random(seed);

        for (int layer = 0; layer < _layerSizes.Length - 1; layer++)
        {
            int fanIn = _layerSizes[layer];
            int fanOut = _layerSizes[layer + 1];
            float stdDev = MathF.Sqrt(2.0f / (fanIn + fanOut));

            int weightStart = layer == 0 ? 0 : _biasOffsets[layer - 1] + _layerSizes[layer];
            int biasStart = _biasOffsets[layer];

            for (int i = 0; i < _weightCounts[layer]; i++)
                _hyperWeights[weightStart + i] = (float)(rng.NextDouble() * 2 - 1) * stdDev;

            for (int i = 0; i < fanOut; i++)
                _hyperWeights[biasStart + i] = 0f;
        }
    }

    /// <summary>
    /// Initializes the HyperNetwork from a pre-computed parameter buffer.
    /// </summary>
    /// <param name="parameters">Flat parameter buffer of length <see cref="_totalParams"/>.</param>
    public HyperNetwork(ReadOnlySpan<float> parameters)
    {
        Debug.Assert(parameters.Length >= _totalParams);
        _hyperWeights = new float[_totalParams];
        parameters.Slice(0, _totalParams).CopyTo(_hyperWeights);
    }

    /// <summary>
    /// Gets the total number of parameters in the hyper-network.
    /// </summary>
    public int TotalParameterCount => _totalParams;

    /// <summary>
    /// Gets the flat parameter buffer (for serialization or evolutionary strategies).
    /// </summary>
    public ReadOnlySpan<float> Parameters => _hyperWeights;

    /// <summary>
    /// Gets a mutable reference to the parameter buffer.
    /// </summary>
    public Span<float> MutableParameters => _hyperWeights;

    /// <summary>
    /// Generates a MicroMLP from a geometry descriptor using stackalloc for hidden layers.
    /// </summary>
    /// <param name="descriptor">16-dimensional geometry descriptor.</param>
    /// <returns>A fully weighted MicroMLP instance.</returns>
    public MicroMLP GenerateMicroMLP(in GeometryDescriptor descriptor)
    {
        Span<float> weights = stackalloc float[OutputDimension];
        GenerateWeights(descriptor, weights);
        return new MicroMLP(weights);
    }

    /// <summary>
    /// Generates raw weight floats into the provided destination buffer.
    /// Zero-allocation: no MicroMLP struct is created.
    /// </summary>
    /// <param name="descriptor">16-dimensional geometry descriptor.</param>
    /// <param name="destination">Buffer of at least <see cref="OutputDimension"/> floats.</param>
    public void GenerateWeights(in GeometryDescriptor descriptor, Span<float> destination)
    {
        Debug.Assert(destination.Length >= OutputDimension);

        Span<float> z = stackalloc float[ZDimension];
        descriptor.AsReadOnlySpan().CopyTo(z);

        Span<float> h0 = stackalloc float[Hidden0];
        Span<float> h1 = stackalloc float[Hidden1];
        Span<float> h2 = stackalloc float[Hidden2];
        Span<float> h3 = stackalloc float[Hidden3];
        Span<float> h4 = stackalloc float[Hidden4];

        ForwardLayer(z, h0, 0, LeakyReLUSimd);
        ForwardLayer(h0, h1, 1, LeakyReLUSimd);
        ForwardLayer(h1, h2, 2, LeakyReLUSimd);
        ForwardLayer(h2, h3, 3, LeakyReLUSimd);
        ForwardLayer(h3, h4, 4, LeakyReLUSimd);

        ForwardLayerLinear(h4, destination, 5);
    }

    /// <summary>
    /// Asynchronously generates a MicroMLP, offloading the compute to a thread pool thread.
    /// Useful for background asset streaming.
    /// </summary>
    /// <param name="descriptor">16-dimensional geometry descriptor.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The generated MicroMLP.</returns>
    public Task<MicroMLP> GenerateMicroMLPAsync(GeometryDescriptor descriptor, CancellationToken cancellationToken = default)
    {
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            return GenerateMicroMLP(descriptor);
        }, cancellationToken);
    }

    /// <summary>
    /// Generates MicroMLPs for a batch of geometry descriptors.
    /// Writes results into the provided span of MicroMLPs.
    /// </summary>
    /// <param name="descriptors">Source descriptors.</param>
    /// <param name="results">Destination buffer for generated MicroMLPs.</param>
    public void GenerateBatch(ReadOnlySpan<GeometryDescriptor> descriptors, Span<MicroMLP> results)
    {
        Debug.Assert(descriptors.Length <= results.Length);

        for (int i = 0; i < descriptors.Length; i++)
        {
            results[i] = GenerateMicroMLP(descriptors[i]);
        }
    }

    /// <summary>
    /// Asynchronously generates a batch of MicroMLPs in parallel.
    /// </summary>
    /// <param name="descriptors">Source descriptors.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Array of generated MicroMLPs.</returns>
    public async Task<MicroMLP[]> GenerateBatchAsync(ReadOnlyMemory<GeometryDescriptor> descriptors, CancellationToken cancellationToken = default)
    {
        var results = new MicroMLP[descriptors.Length];
        var tasks = new Task[descriptors.Length];

        for (int i = 0; i < descriptors.Length; i++)
        {
            int idx = i;
            tasks[i] = Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                var desc = descriptors.Span[idx];
                results[idx] = GenerateMicroMLP(desc);
            }, cancellationToken);
        }

        await Task.WhenAll(tasks).ConfigureAwait(false);
        return results;
    }

    /// <summary>
    /// Extracts a geometry descriptor from an existing MicroMLP by inverting the hyper-network.
    /// Uses gradient-free optimization (random search with refinement) to find the best-fit descriptor.
    /// </summary>
    /// <param name="target">The MicroMLP to reconstruct a descriptor for.</param>
    /// <param name="iterations">Number of optimization iterations.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>The best-matching geometry descriptor.</returns>
    public GeometryDescriptor ExtractDescriptor(in MicroMLP target, int iterations = 1000, int seed = 1337)
    {
        var rng = new Random(seed);
        GeometryDescriptor bestDesc = default;
        float bestLoss = float.MaxValue;

        Span<float> candidate = stackalloc float[ZDimension];
        Span<float> generated = stackalloc float[OutputDimension];
        Span<float> targetWeights = stackalloc float[OutputDimension];
        target.CopyTo(targetWeights);

        for (int i = 0; i < ZDimension; i++)
            candidate[i] = (float)(rng.NextDouble() * 2 - 1);

        float bestScale = 1.0f;

        for (int iter = 0; iter < iterations; iter++)
        {
            GenerateWeights(bestDesc, generated);
            float loss = ComputeMSELoss(generated, targetWeights);

            if (loss < bestLoss)
            {
                bestLoss = loss;
            }

            float perturbScale = bestScale * (1.0f - (float)iter / iterations);
            GeometryDescriptor trialDesc = bestDesc;
            Span<float> trialSpan = trialDesc.AsSpan();

            for (int d = 0; d < ZDimension; d++)
            {
                float noise = (float)(rng.NextDouble() * 2 - 1) * perturbScale;
                trialSpan[d] = bestDesc.AsSpan()[d] + noise;
            }

            GenerateWeights(trialDesc, generated);
            float trialLoss = ComputeMSELoss(generated, targetWeights);

            if (trialLoss < bestLoss)
            {
                bestLoss = trialLoss;
                bestDesc = trialDesc;
            }
            else
            {
                bestScale *= 0.99f;
            }
        }

        return bestDesc;
    }

    /// <summary>
    /// Computes the MSE reconstruction loss between a target MicroMLP and the weights
    /// generated from the given descriptor.
    /// </summary>
    /// <param name="descriptor">Geometry descriptor to evaluate.</param>
    /// <param name="target">Target MicroMLP to reconstruct.</param>
    /// <returns>MSE loss value.</returns>
    public float ComputeReconstructionLoss(in GeometryDescriptor descriptor, in MicroMLP target)
    {
        Span<float> generated = stackalloc float[OutputDimension];
        Span<float> targetWeights = stackalloc float[OutputDimension];
        target.CopyTo(targetWeights);

        GenerateWeights(descriptor, generated);
        return ComputeMSELoss(generated, targetWeights);
    }

    /// <summary>
    /// Performs a single evolutionary strategy training step.
    /// Evaluates perturbations of the hyper-network parameters and updates via ES gradient estimate.
    /// </summary>
    /// <param name="descriptors">Batch of geometry descriptors for training.</param>
    /// <param name="targets">Corresponding target MicroMLPs.</param>
    /// <param name="sigma">Standard deviation of the exploration noise.</param>
    /// <param name="learningRate">Learning rate for the parameter update.</param>
    /// <param name="populationSize">Number of perturbations per parameter vector.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>Average loss before the update step.</returns>
    public float TrainStep(
        ReadOnlySpan<GeometryDescriptor> descriptors,
        ReadOnlySpan<MicroMLP> targets,
        float sigma = 0.05f,
        float learningRate = 0.01f,
        int populationSize = 16,
        int seed = 0)
    {
        Debug.Assert(descriptors.Length == targets.Length);
        int batchSize = descriptors.Length;

        var rng = new Random(seed);

        float baseLoss = 0f;
        for (int i = 0; i < batchSize; i++)
            baseLoss += ComputeReconstructionLoss(descriptors[i], targets[i]);
        baseLoss /= batchSize;

        Span<float> noise = stackalloc float[_totalParams];
        Span<float> perturbedParams = stackalloc float[_totalParams];
        Span<float> gradient = stackalloc float[_totalParams];
        gradient.Clear();

        for (int pop = 0; pop < populationSize; pop++)
        {
            for (int p = 0; p < _totalParams; p++)
                noise[p] = (float)(rng.NextDouble() * 2 - 1);

            float lossPos = 0f;
            float lossNeg = 0f;

            for (int p = 0; p < _totalParams; p++)
                perturbedParams[p] = _hyperWeights[p] + sigma * noise[p];

            for (int i = 0; i < batchSize; i++)
            {
                var tempNet = new HyperNetwork(perturbedParams);
                lossPos += tempNet.ComputeReconstructionLoss(descriptors[i], targets[i]);
            }
            lossPos /= batchSize;

            for (int p = 0; p < _totalParams; p++)
                perturbedParams[p] = _hyperWeights[p] - sigma * noise[p];

            for (int i = 0; i < batchSize; i++)
            {
                var tempNet = new HyperNetwork(perturbedParams);
                lossNeg += tempNet.ComputeReconstructionLoss(descriptors[i], targets[i]);
            }
            lossNeg /= batchSize;

            float factor = (lossPos - lossNeg) / (2f * sigma * populationSize);
            for (int p = 0; p < _totalParams; p++)
                gradient[p] += factor * noise[p];
        }

        for (int p = 0; p < _totalParams; p++)
            _hyperWeights[p] -= learningRate * gradient[p];

        return baseLoss;
    }

    /// <summary>
    /// Serializes the hyper-network parameters to a byte array.
    /// </summary>
    public byte[] Serialize()
    {
        var bytes = new byte[_totalParams * sizeof(float)];
        MemoryMarshal.AsBytes(_hyperWeights.AsSpan()).CopyTo(bytes);
        return bytes;
    }

    /// <summary>
    /// Deserializes a HyperNetwork from a byte span.
    /// </summary>
    public static HyperNetwork Deserialize(ReadOnlySpan<byte> data)
    {
        var floats = MemoryMarshal.Cast<byte, float>(data);
        return new HyperNetwork(floats);
    }

    /// <summary>
    /// Saves the hyper-network to a file.
    /// </summary>
    public async Task SaveAsync(string path, CancellationToken cancellationToken = default)
    {
        var data = Serialize();
        await System.IO.File.WriteAllBytesAsync(path, data, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Loads a HyperNetwork from a file.
    /// </summary>
    public static async Task<HyperNetwork> LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        var data = await System.IO.File.ReadAllBytesAsync(path, cancellationToken).ConfigureAwait(false);
        return Deserialize(data);
    }

    /// <summary>
    /// Computes the memory footprint of this HyperNetwork in bytes.
    /// </summary>
    public long ComputeMemoryFootprint()
    {
        return sizeof(float) * _totalParams + sizeof(float) * _totalParams; // weights + biases (flat array)
    }

    #region Private Forward Helpers

    private void ForwardLayer(ReadOnlySpan<float> input, Span<float> output, int layerIndex, Action<Span<float>> activation)
    {
        int fanIn = _layerSizes[layerIndex];
        int fanOut = _layerSizes[layerIndex + 1];

        int weightStart = layerIndex == 0 ? 0 : _biasOffsets[layerIndex - 1] + _layerSizes[layerIndex];
        int biasStart = _biasOffsets[layerIndex];

        for (int o = 0; o < fanOut; o++)
        {
            float sum = _hyperWeights[biasStart + o];
            int rowStart = weightStart + o * fanIn;

            if (Vector.IsHardwareAccelerated && fanIn >= Vector<float>.Count)
            {
                var acc = Vector<float>.Zero;
                int j = 0;
                for (; j + Vector<float>.Count <= fanIn; j += Vector<float>.Count)
                {
                    var w = new Vector<float>(_hyperWeights.AsSpan(rowStart + j, Vector<float>.Count));
                    var inp = new Vector<float>(input.Slice(j, Vector<float>.Count));
                    acc += w * inp;
                }
                for (int k = 0; k < Vector<float>.Count; k++)
                    sum += acc[k];
                for (; j < fanIn; j++)
                    sum += _hyperWeights[rowStart + j] * input[j];
            }
            else
            {
                for (int j = 0; j < fanIn; j++)
                    sum += _hyperWeights[rowStart + j] * input[j];
            }

            output[o] = sum;
        }

        activation(output);
    }

    private void ForwardLayerLinear(ReadOnlySpan<float> input, Span<float> output, int layerIndex)
    {
        int fanIn = _layerSizes[layerIndex];
        int fanOut = _layerSizes[layerIndex + 1];

        int weightStart = _biasOffsets[layerIndex - 1] + _layerSizes[layerIndex];
        int biasStart = _biasOffsets[layerIndex];

        for (int o = 0; o < fanOut; o++)
        {
            float sum = _hyperWeights[biasStart + o];
            int rowStart = weightStart + o * fanIn;

            for (int j = 0; j < fanIn; j++)
                sum += _hyperWeights[rowStart + j] * input[j];

            output[o] = sum;
        }
    }

    #endregion

    #region Activation Functions

    private static void LeakyReLUSimd(Span<float> values)
    {
        const float alpha = 0.01f;
        if (Vector.IsHardwareAccelerated && values.Length >= Vector<float>.Count)
        {
            var zero = Vector<float>.Zero;
            var alphaVec = new Vector<float>(alpha);
            int i = 0;
            for (; i + Vector<float>.Count <= values.Length; i += Vector<float>.Count)
            {
                var v = new Vector<float>(values.Slice(i, Vector<float>.Count));
                var mask = Vector.GreaterThan(v, zero);
                var result = Vector.ConditionalSelect(mask, v, v * alphaVec);
                result.CopyTo(values.Slice(i));
            }
            for (; i < values.Length; i++)
                values[i] = values[i] > 0 ? values[i] : values[i] * alpha;
        }
        else
        {
            for (int i = 0; i < values.Length; i++)
                values[i] = values[i] > 0 ? values[i] : values[i] * alpha;
        }
    }

    private static float ComputeMSELoss(ReadOnlySpan<float> predicted, ReadOnlySpan<float> target)
    {
        Debug.Assert(predicted.Length == target.Length);
        float sum = 0f;
        int len = predicted.Length;

        if (Vector.IsHardwareAccelerated && len >= Vector<float>.Count)
        {
            var acc = Vector<float>.Zero;
            int i = 0;
            for (; i + Vector<float>.Count <= len; i += Vector<float>.Count)
            {
                var p = new Vector<float>(predicted.Slice(i, Vector<float>.Count));
                var t = new Vector<float>(target.Slice(i, Vector<float>.Count));
                var diff = p - t;
                acc += diff * diff;
            }
            for (int k = 0; k < Vector<float>.Count; k++)
                sum += acc[k];
            for (; i < len; i++)
            {
                float d = predicted[i] - target[i];
                sum += d * d;
            }
        }
        else
        {
            for (int i = 0; i < len; i++)
            {
                float d = predicted[i] - target[i];
                sum += d * d;
            }
        }

        return sum / len;
    }

    #endregion
}
