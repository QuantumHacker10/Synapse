using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Enhanced Micro-MLP with residual connections, positional encoding, and wider layers.
/// Architecture: PositionalEncode(3→30) → Dense(30→32, SiLU) + Residual → Dense(32→32, SiLU) + Residual
///              → Dense(32→32, SiLU) + Residual → Dense(32→1, Linear)
/// Total weights: ~4,600 (vs 160 in original MicroMLP)
/// </summary>
public sealed class DeepMicroMLP : ISdfNetwork, IDisposable
{
    /// <summary>
    /// Input dimension (3D coordinates).
    /// </summary>
    public const int InputDimension = 3;

    /// <summary>
    /// Positional encoding frequency count. Each frequency produces sin and cos for each dimension.
    /// Output dimension = InputDimension * FrequencyCount * 2 + InputDimension = 3 * 5 * 2 + 3 = 33.
    /// </summary>
    public const int FrequencyCount = 5;

    /// <summary>
    /// Encoded input dimension after positional encoding.
    /// </summary>
    public const int EncodedDimension = InputDimension * FrequencyCount * 2 + InputDimension;

    /// <summary>
    /// Hidden layer size (wider than original 8).
    /// </summary>
    public const int HiddenSize = 32;

    /// <summary>
    /// Output dimension (signed distance).
    /// </summary>
    public const int OutputDimension = 1;

    /// <summary>
    /// Number of residual blocks.
    /// </summary>
    public const int ResidualBlockCount = 3;

    /// <summary>
    /// Layer 1 weights: EncodedDimension → HiddenSize.
    /// </summary>
    public float[] Layer1Weights { get; private set; }

    /// <summary>
    /// Layer 1 bias.
    /// </summary>
    public float[] Layer1Bias { get; private set; }

    /// <summary>
    /// Residual block weights (ResidualBlockCount blocks, each with 2 linear layers).
    /// Format: [block][layer] where layer 0 is HiddenSize→HiddenSize, layer 1 is projection.
    /// </summary>
    public float[][] ResidualWeights { get; private set; }

    /// <summary>
    /// Residual block biases.
    /// </summary>
    public float[][] ResidualBiases { get; private set; }

    /// <summary>
    /// Layer norm gamma for each residual block.
    /// </summary>
    public float[][] LayerNormGamma { get; private set; }

    /// <summary>
    /// Layer norm beta for each residual block.
    /// </summary>
    public float[][] LayerNormBeta { get; private set; }

    /// <summary>
    /// Output layer weights: HiddenSize → OutputDimension.
    /// </summary>
    public float[] OutputWeights { get; private set; }

    /// <summary>
    /// Output layer bias.
    /// </summary>
    public float[] OutputBias { get; private set; }

    /// <summary>
    /// Positional encoding scale factor.
    /// </summary>
    public float PositionalScale { get; set; } = 1.0f;

    /// <summary>
    /// Creates a new DeepMicroMLP with random weights.
    /// </summary>
    public DeepMicroMLP(Random? random = null)
    {
        random ??= Random.Shared;

        // Layer 1: EncodedDimension → HiddenSize
        Layer1Weights = CreateXavierRandom(EncodedDimension, HiddenSize, random);
        Layer1Bias = new float[HiddenSize];

        // Residual blocks
        ResidualWeights = new float[ResidualBlockCount][];
        ResidualBiases = new float[ResidualBlockCount][];
        LayerNormGamma = new float[ResidualBlockCount][];
        LayerNormBeta = new float[ResidualBlockCount][];

        for (int b = 0; b < ResidualBlockCount; b++)
        {
            ResidualWeights[b] = CreateXavierRandom(HiddenSize, HiddenSize, random);
            ResidualBiases[b] = new float[HiddenSize];
            LayerNormGamma[b] = new float[HiddenSize];
            LayerNormBeta[b] = new float[HiddenSize];

            // Initialize gamma to 1, beta to 0
            Array.Fill(LayerNormGamma[b], 1.0f);
        }

        // Output layer: HiddenSize → OutputDimension
        OutputWeights = CreateXavierRandom(HiddenSize, OutputDimension, random);
        OutputBias = new float[OutputDimension];
    }

    /// <summary>
    /// Creates a DeepMicroMLP from a flat weight array (for compatibility with HyperNetwork).
    /// </summary>
    public DeepMicroMLP(ReadOnlySpan<float> weights)
    {
        int expected = GetTotalWeightCount();
        if (weights.Length < expected)
            throw new ArgumentException($"Expected at least {expected} weights, got {weights.Length}.");

        int offset = 0;

        // Layer 1
        Layer1Weights = new float[EncodedDimension * HiddenSize];
        weights.Slice(offset, EncodedDimension * HiddenSize).CopyTo(Layer1Weights);
        offset += EncodedDimension * HiddenSize;

        Layer1Bias = new float[HiddenSize];
        weights.Slice(offset, HiddenSize).CopyTo(Layer1Bias);
        offset += HiddenSize;

        // Residual blocks
        ResidualWeights = new float[ResidualBlockCount][];
        ResidualBiases = new float[ResidualBlockCount][];
        LayerNormGamma = new float[ResidualBlockCount][];
        LayerNormBeta = new float[ResidualBlockCount][];

        for (int b = 0; b < ResidualBlockCount; b++)
        {
            ResidualWeights[b] = new float[HiddenSize * HiddenSize];
            weights.Slice(offset, HiddenSize * HiddenSize).CopyTo(ResidualWeights[b]);
            offset += HiddenSize * HiddenSize;

            ResidualBiases[b] = new float[HiddenSize];
            weights.Slice(offset, HiddenSize).CopyTo(ResidualBiases[b]);
            offset += HiddenSize;

            LayerNormGamma[b] = new float[HiddenSize];
            weights.Slice(offset, HiddenSize).CopyTo(LayerNormGamma[b]);
            offset += HiddenSize;

            LayerNormBeta[b] = new float[HiddenSize];
            weights.Slice(offset, HiddenSize).CopyTo(LayerNormBeta[b]);
            offset += HiddenSize;
        }

        // Output layer
        OutputWeights = new float[HiddenSize * OutputDimension];
        weights.Slice(offset, HiddenSize * OutputDimension).CopyTo(OutputWeights);
        offset += HiddenSize * OutputDimension;

        OutputBias = new float[OutputDimension];
        weights.Slice(offset, OutputDimension).CopyTo(OutputBias);
    }

    /// <summary>
    /// Gets the total number of weights in this network.
    /// </summary>
    public static int GetTotalWeightCount()
    {
        int count = 0;
        count += EncodedDimension * HiddenSize;  // Layer1 weights
        count += HiddenSize;                      // Layer1 bias
        // Per residual block: weights (H×H) + bias (H) + gamma (H) + beta (H)
        count += ResidualBlockCount * (HiddenSize * HiddenSize + HiddenSize * 3);
        count += HiddenSize * OutputDimension;    // Output weights
        count += OutputDimension;                  // Output bias
        return count;
    }

    /// <summary>
    /// Applies positional encoding to a 3D point.
    /// Maps (x,y,z) → (x,y,z, sin(2π·f·x), cos(2π·f·x), ..., sin(2π·f·z), cos(2π·f·z)) for f=1..FrequencyCount.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PositionalEncode(Vector3 point, Span<float> output, float scale = 1.0f)
    {
        float sx = point.X * scale;
        float sy = point.Y * scale;
        float sz = point.Z * scale;

        int idx = 0;

        // Raw coordinates
        output[idx++] = sx;
        output[idx++] = sy;
        output[idx++] = sz;

        // Frequency bands
        for (int f = 1; f <= FrequencyCount; f++)
        {
            float freq = 2.0f * MathF.PI * f;
            output[idx++] = MathF.Sin(freq * sx);
            output[idx++] = MathF.Cos(freq * sx);
            output[idx++] = MathF.Sin(freq * sy);
            output[idx++] = MathF.Cos(freq * sy);
            output[idx++] = MathF.Sin(freq * sz);
            output[idx++] = MathF.Cos(freq * sz);
        }
    }

    /// <summary>
    /// Evaluates the network at a 3D point, returning signed distance.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Evaluate(Vector3 point)
    {
        Span<float> encoded = stackalloc float[EncodedDimension];
        PositionalEncode(point, encoded, PositionalScale);

        Span<float> hidden = stackalloc float[HiddenSize];
        Span<float> hidden2 = stackalloc float[HiddenSize];
        Span<float> residual = stackalloc float[HiddenSize];

        // Layer 1: Encoded → Hidden with SiLU
        MatVecWithBias(Layer1Weights, encoded, Layer1Bias, hidden, HiddenSize, EncodedDimension);
        ApplySiLU(hidden);

        // Residual blocks
        for (int b = 0; b < ResidualBlockCount; b++)
        {
            hidden.CopyTo(residual);

            // Layer norm
            LayerNorm(hidden, LayerNormGamma[b], LayerNormBeta[b]);

            // Linear layer
            MatVecWithBias(ResidualWeights[b], hidden, ResidualBiases[b], hidden2, HiddenSize, HiddenSize);

            // SiLU activation
            ApplySiLU(hidden2);

            // Skip connection
            for (int i = 0; i < HiddenSize; i++)
                hidden[i] = hidden2[i] + residual[i];
        }

        // Output layer
        return DotProduct(hidden, OutputWeights, HiddenSize) + OutputBias[0];
    }

    /// <summary>
    /// Evaluates the network with gradient computation via central differences.
    /// </summary>
    public float EvaluateWithGradient(Vector3 point, out Vector3 gradient)
    {
        const float epsilon = 0.0001f;

        float d = Evaluate(point);
        float dx = (Evaluate(new Vector3(point.X + epsilon, point.Y, point.Z)) -
                    Evaluate(new Vector3(point.X - epsilon, point.Y, point.Z))) / (2 * epsilon);
        float dy = (Evaluate(new Vector3(point.X, point.Y + epsilon, point.Z)) -
                    Evaluate(new Vector3(point.X, point.Y - epsilon, point.Z))) / (2 * epsilon);
        float dz = (Evaluate(new Vector3(point.X, point.Y, point.Z + epsilon)) -
                    Evaluate(new Vector3(point.X, point.Y, point.Z - epsilon))) / (2 * epsilon);

        gradient = new Vector3(dx, dy, dz);
        float length = gradient.Length();
        if (length > 0) gradient /= length;

        return d;
    }

    /// <summary>
    /// Batch evaluation of multiple points.
    /// </summary>
    public void EvaluateBatch(ReadOnlySpan<Vector3> points, Span<float> distances)
    {
        if (points.Length != distances.Length)
            throw new ArgumentException("Points and distances must have the same length.");

        int width = Vector<float>.Count;
        int i = 0;
        for (; i + width <= points.Length; i += width)
            EvaluateVectorized(points.Slice(i, width), distances.Slice(i, width));

        for (; i < points.Length; i++)
            distances[i] = Evaluate(points[i]);
    }

    /// <summary>
    /// Évalue un petit lot de points (≤ Vector&lt;float&gt;.Count) avec transposition SoA
    /// pour exploiter le SIMD sur les dimensions cachées.
    /// </summary>
    public void EvaluateVectorized(ReadOnlySpan<Vector3> points, Span<float> distancesOut)
    {
        if (points.Length != distancesOut.Length)
            throw new ArgumentException("Points and distances must have the same length.");
        if (points.Length == 0) return;

        int batch = points.Length;
        int encDim = EncodedDimension;
        int hidden = HiddenSize;

        // SoA encoded features: encodedSoA[feature][lane]
        Span<float> encodedSoA = stackalloc float[encDim * batch];
        for (int lane = 0; lane < batch; lane++)
        {
            Span<float> enc = stackalloc float[encDim];
            PositionalEncode(points[lane], enc, PositionalScale);
            for (int f = 0; f < encDim; f++)
                encodedSoA[f * batch + lane] = enc[f];
        }

        // hiddenSoA[neuron][lane]
        Span<float> hiddenSoA = stackalloc float[hidden * batch];
        Span<float> hidden2SoA = stackalloc float[hidden * batch];
        Span<float> residualSoA = stackalloc float[hidden * batch];

        // Layer 1
        for (int o = 0; o < hidden; o++)
        {
            for (int lane = 0; lane < batch; lane++)
            {
                float sum = Layer1Bias[o];
                int rowStart = o * encDim;
                for (int j = 0; j < encDim; j++)
                    sum += Layer1Weights[rowStart + j] * encodedSoA[j * batch + lane];
                hiddenSoA[o * batch + lane] = sum;
            }
        }
        ApplySiLUSoA(hiddenSoA, hidden, batch);

        // Residual blocks
        for (int b = 0; b < ResidualBlockCount; b++)
        {
            for (int n = 0; n < hidden; n++)
                for (int lane = 0; lane < batch; lane++)
                    residualSoA[n * batch + lane] = hiddenSoA[n * batch + lane];

            LayerNormSoA(hiddenSoA, LayerNormGamma[b], LayerNormBeta[b], hidden, batch);

            for (int o = 0; o < hidden; o++)
            {
                for (int lane = 0; lane < batch; lane++)
                {
                    float sum = ResidualBiases[b][o];
                    int rowStart = o * hidden;
                    for (int j = 0; j < hidden; j++)
                        sum += ResidualWeights[b][rowStart + j] * hiddenSoA[j * batch + lane];
                    hidden2SoA[o * batch + lane] = sum;
                }
            }
            ApplySiLUSoA(hidden2SoA, hidden, batch);

            for (int n = 0; n < hidden; n++)
                for (int lane = 0; lane < batch; lane++)
                    hiddenSoA[n * batch + lane] = hidden2SoA[n * batch + lane] + residualSoA[n * batch + lane];
        }

        // Output
        for (int lane = 0; lane < batch; lane++)
        {
            float result = OutputBias[0];
            for (int n = 0; n < hidden; n++)
                result += hiddenSoA[n * batch + lane] * OutputWeights[n];
            distancesOut[lane] = result;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplySiLUSoA(Span<float> values, int neurons, int batch)
    {
        for (int n = 0; n < neurons; n++)
        {
            int baseIdx = n * batch;
            for (int lane = 0; lane < batch; lane++)
            {
                float x = values[baseIdx + lane];
                values[baseIdx + lane] = x / (1.0f + MathF.Exp(-x));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LayerNormSoA(Span<float> values, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta, int neurons, int batch)
    {
        for (int lane = 0; lane < batch; lane++)
        {
            float mean = 0f;
            for (int n = 0; n < neurons; n++)
                mean += values[n * batch + lane];
            mean /= neurons;

            float variance = 0f;
            for (int n = 0; n < neurons; n++)
            {
                float diff = values[n * batch + lane] - mean;
                variance += diff * diff;
            }
            variance /= neurons;

            float invStd = 1.0f / MathF.Sqrt(variance + 1e-5f);
            for (int n = 0; n < neurons; n++)
                values[n * batch + lane] = (values[n * batch + lane] - mean) * invStd * gamma[n] + beta[n];
        }
    }

    /// <summary>
    /// Computes the gradient (surface normal) via central differences.
    /// </summary>
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
    /// Serializes all weights to a flat float array.
    /// </summary>
    public float[] Serialize()
    {
        float[] result = new float[GetTotalWeightCount()];
        int offset = 0;

        Layer1Weights.CopyTo(result, offset);
        offset += EncodedDimension * HiddenSize;

        Layer1Bias.CopyTo(result, offset);
        offset += HiddenSize;

        for (int b = 0; b < ResidualBlockCount; b++)
        {
            ResidualWeights[b].CopyTo(result, offset);
            offset += HiddenSize * HiddenSize;

            ResidualBiases[b].CopyTo(result, offset);
            offset += HiddenSize;

            LayerNormGamma[b].CopyTo(result, offset);
            offset += HiddenSize;

            LayerNormBeta[b].CopyTo(result, offset);
            offset += HiddenSize;
        }

        OutputWeights.CopyTo(result, offset);
        offset += HiddenSize * OutputDimension;

        OutputBias.CopyTo(result, offset);

        return result;
    }

    /// <summary>
    /// Creates a deep copy of this network.
    /// </summary>
    public DeepMicroMLP Clone()
    {
        var clone = new DeepMicroMLP();
        clone.PositionalScale = PositionalScale;

        Layer1Weights.CopyTo(clone.Layer1Weights, 0);
        Layer1Bias.CopyTo(clone.Layer1Bias, 0);

        for (int b = 0; b < ResidualBlockCount; b++)
        {
            ResidualWeights[b].CopyTo(clone.ResidualWeights[b], 0);
            ResidualBiases[b].CopyTo(clone.ResidualBiases[b], 0);
            LayerNormGamma[b].CopyTo(clone.LayerNormGamma[b], 0);
            LayerNormBeta[b].CopyTo(clone.LayerNormBeta[b], 0);
        }

        OutputWeights.CopyTo(clone.OutputWeights, 0);
        OutputBias.CopyTo(clone.OutputBias, 0);

        return clone;
    }

    /// <summary>
    /// Linearly interpolates between two DeepMicroMLPs.
    /// </summary>
    public DeepMicroMLP Lerp(DeepMicroMLP other, float t)
    {
        var result = new DeepMicroMLP();
        result.PositionalScale = PositionalScale + (other.PositionalScale - PositionalScale) * t;

        LinearInterpolate(Layer1Weights, other.Layer1Weights, result.Layer1Weights, t);
        LinearInterpolate(Layer1Bias, other.Layer1Bias, result.Layer1Bias, t);

        for (int b = 0; b < ResidualBlockCount; b++)
        {
            LinearInterpolate(ResidualWeights[b], other.ResidualWeights[b], result.ResidualWeights[b], t);
            LinearInterpolate(ResidualBiases[b], other.ResidualBiases[b], result.ResidualBiases[b], t);
            LinearInterpolate(LayerNormGamma[b], other.LayerNormGamma[b], result.LayerNormGamma[b], t);
            LinearInterpolate(LayerNormBeta[b], other.LayerNormBeta[b], result.LayerNormBeta[b], t);
        }

        LinearInterpolate(OutputWeights, other.OutputWeights, result.OutputWeights, t);
        LinearInterpolate(OutputBias, other.OutputBias, result.OutputBias, t);

        return result;
    }

    /// <summary>
    /// Generates HLSL shader code for GPU evaluation.
    /// </summary>
    public string GenerateHLSL()
    {
        var sb = new StringBuilder();

        sb.AppendLine("// Auto-generated DeepMicroMLP HLSL shader");
        sb.AppendLine($"// Positional encoding: {FrequencyCount} frequencies, {EncodedDimension}D encoded input");
        sb.AppendLine($"// Hidden size: {HiddenSize}, Residual blocks: {ResidualBlockCount}");
        sb.AppendLine();

        sb.AppendLine("cbuffer NeuralWeights : register(b0)");
        sb.AppendLine("{");
        sb.AppendLine($"    float Layer1_Weights[{EncodedDimension * HiddenSize}];");
        sb.AppendLine($"    float Layer1_Bias[{HiddenSize}];");

        for (int b = 0; b < ResidualBlockCount; b++)
        {
            sb.AppendLine($"    float Res{b}_Weights[{HiddenSize * HiddenSize}];");
            sb.AppendLine($"    float Res{b}_Bias[{HiddenSize}];");
            sb.AppendLine($"    float Res{b}_Gamma[{HiddenSize}];");
            sb.AppendLine($"    float Res{b}_Beta[{HiddenSize}];");
        }

        sb.AppendLine($"    float Output_Weights[{HiddenSize}];");
        sb.AppendLine($"    float Output_Bias[1];");
        sb.AppendLine("};");
        sb.AppendLine();

        // Positional encoding function
        sb.AppendLine($"static const int FREQ_COUNT = {FrequencyCount};");
        sb.AppendLine();
        sb.AppendLine("void PositionalEncode(float3 p, out float[{EncodedDimension}] encoded)");
        sb.AppendLine("{");
        sb.AppendLine("    encoded[0] = p.x; encoded[1] = p.y; encoded[2] = p.z;");
        sb.AppendLine("    int idx = 3;");
        sb.AppendLine("    [unroll]");
        sb.AppendLine("    for(int f = 1; f <= FREQ_COUNT; f++)");
        sb.AppendLine("    {");
        sb.AppendLine("        float freq = 6.2831853 * f;");
        sb.AppendLine("        encoded[idx++] = sin(freq * p.x); encoded[idx++] = cos(freq * p.x);");
        sb.AppendLine("        encoded[idx++] = sin(freq * p.y); encoded[idx++] = cos(freq * p.y);");
        sb.AppendLine("        encoded[idx++] = sin(freq * p.z); encoded[idx++] = cos(freq * p.z);");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine();

        // Main evaluation function
        sb.AppendLine("float EvaluateNeuralSDF(float3 localPos)");
        sb.AppendLine("{");
        sb.AppendLine($"    float encoded[{EncodedDimension}];");
        sb.AppendLine("    PositionalEncode(localPos, encoded);");
        sb.AppendLine();
        sb.AppendLine($"    float hidden[{HiddenSize}];");
        sb.AppendLine($"    float temp[{HiddenSize}];");
        sb.AppendLine($"    float residual[{HiddenSize}];");
        sb.AppendLine();

        // Layer 1
        sb.AppendLine("    [unroll]");
        sb.AppendLine($"    for(int i = 0; i < {HiddenSize}; i++)");
        sb.AppendLine("    {");
        sb.AppendLine("        float sum = Layer1_Bias[i];");
        sb.AppendLine($"        [unroll] for(int j = 0; j < {EncodedDimension}; j++)");
        sb.AppendLine("            sum += encoded[j] * Layer1_Weights[i * {EncodedDimension} + j];");
        sb.AppendLine("        hidden[i] = sum / (1.0 + exp(-sum));"); // SiLU
        sb.AppendLine("    }");
        sb.AppendLine();

        // Residual blocks
        for (int b = 0; b < ResidualBlockCount; b++)
        {
            sb.AppendLine($"    // Residual block {b}");
            sb.AppendLine("    [unroll]");
            sb.AppendLine($"    for(int i = 0; i < {HiddenSize}; i++) residual[i] = hidden[i];");
            sb.AppendLine();

            // Layer norm
            sb.AppendLine("    // Layer norm");
            sb.AppendLine("    float mean = 0;");
            sb.AppendLine($"    [unroll] for(int i = 0; i < {HiddenSize}; i++) mean += hidden[i];");
            sb.AppendLine($"    mean /= {HiddenSize};");
            sb.AppendLine("    float var = 0;");
            sb.AppendLine($"    [unroll] for(int i = 0; i < {HiddenSize}; i++) var += (hidden[i]-mean)*(hidden[i]-mean);");
            sb.AppendLine($"    var /= {HiddenSize};");
            sb.AppendLine("    float invStd = 1.0 / sqrt(var + 1e-5);");
            sb.AppendLine($"    [unroll] for(int i = 0; i < {HiddenSize}; i++)");
            sb.AppendLine("        hidden[i] = (hidden[i] - mean) * invStd * Res{b}_Gamma[i] + Res{b}_Beta[i];");
            sb.AppendLine();

            // Linear + SiLU
            sb.AppendLine("    [unroll]");
            sb.AppendLine($"    for(int i = 0; i < {HiddenSize}; i++)");
            sb.AppendLine("    {");
            sb.AppendLine($"        float sum = Res{b}_Bias[i];");
            sb.AppendLine($"        [unroll] for(int j = 0; j < {HiddenSize}; j++)");
            sb.AppendLine($"            sum += hidden[j] * Res{b}_Weights[i * {HiddenSize} + j];");
            sb.AppendLine($"        temp[i] = sum / (1.0 + exp(-sum));"); // SiLU
            sb.AppendLine("    }");
            sb.AppendLine();

            // Skip connection
            sb.AppendLine("    [unroll]");
            sb.AppendLine($"    for(int i = 0; i < {HiddenSize}; i++)");
            sb.AppendLine("        hidden[i] = temp[i] + residual[i];");
            sb.AppendLine();
        }

        // Output layer
        sb.AppendLine("    float result = Output_Bias[0];");
        sb.AppendLine("    [unroll]");
        sb.AppendLine($"    for(int i = 0; i < {HiddenSize}; i++)");
        sb.AppendLine("        result += hidden[i] * Output_Weights[i];");
        sb.AppendLine("    return result;");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Computes mean squared error against ground truth network.
    /// </summary>
    public float ComputeLoss(DeepMicroMLP groundTruth, int sampleCount = 1000, Random? random = null)
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

    public void Dispose()
    {
        GC.SuppressFinalize(this);
    }

    #region Private Helpers

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int length)
    {
        float sum = 0f;
        int i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            int vectorSize = Vector<float>.Count;
            var vSum = Vector<float>.Zero;

            for (; i <= length - vectorSize; i += vectorSize)
            {
                var va = new Vector<float>(a.Slice(i));
                var vb = new Vector<float>(b.Slice(i));
                vSum += va * vb;
            }

            for (int j = 0; j < vectorSize; j++)
                sum += vSum[j];
        }

        for (; i < length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MatVecWithBias(
        ReadOnlySpan<float> weights, ReadOnlySpan<float> input, ReadOnlySpan<float> bias,
        Span<float> output, int outSize, int inSize)
    {
        for (int o = 0; o < outSize; o++)
        {
            float sum = bias[o];
            int rowStart = o * inSize;
            int i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                int vectorSize = Vector<float>.Count;
                var vSum = Vector<float>.Zero;

                for (; i <= inSize - vectorSize; i += vectorSize)
                {
                    var vw = new Vector<float>(weights.Slice(rowStart + i));
                    var vi = new Vector<float>(input.Slice(i));
                    vSum += vw * vi;
                }

                for (int j = 0; j < vectorSize; j++)
                    sum += vSum[j];
            }

            for (; i < inSize; i++)
                sum += weights[rowStart + i] * input[i];

            output[o] = sum;
        }
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
        for (int i = 0; i < length; i++)
            mean += values[i];
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

    private static float[] CreateXavierRandom(int fanIn, int fanOut, Random random)
    {
        float scale = MathF.Sqrt(2.0f / fanIn);
        float[] weights = new float[fanIn * fanOut];

        for (int i = 0; i < weights.Length; i++)
            weights[i] = (float)(random.NextDouble() * 2.0 - 1.0) * scale;

        return weights;
    }

    private static void LinearInterpolate(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, float t)
    {
        float oneMinusT = 1.0f - t;
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] * oneMinusT + b[i] * t;
    }

    #endregion
}
