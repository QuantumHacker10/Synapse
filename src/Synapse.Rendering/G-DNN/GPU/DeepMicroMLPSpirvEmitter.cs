using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using GDNN.Core.NeuralNetwork;

namespace GDNN.GPU;

/// <summary>
/// Produit du GLSL compute (et SPIR-V via toolchain) pour évaluer un DeepMicroMLP
/// depuis des SSBOs de poids + positions. Aligné sur HlslCompatibleEvaluator.
/// </summary>
public static class DeepMicroMLPSpirvEmitter
{
    public const int LocalSizeX = 64;

    /// <summary>
    /// GLSL compute complet pour DeepMicroMLP (std430, push constant count).
    /// </summary>
    public static string GenerateGlsl()
    {
        int enc = DeepMicroMLP.EncodedDimension;
        int hidden = DeepMicroMLP.HiddenSize;
        int res = DeepMicroMLP.ResidualBlockCount;
        int freq = DeepMicroMLP.FrequencyCount;

        var sb = new StringBuilder();
        sb.AppendLine("#version 450");
        sb.AppendLine($"layout(local_size_x = {LocalSizeX}) in;");
        sb.AppendLine();
        sb.AppendLine("layout(set = 0, binding = 0) readonly buffer Positions { vec4 positions[]; };");
        sb.AppendLine("layout(set = 0, binding = 1) readonly buffer Weights { float weights[]; };");
        sb.AppendLine("layout(set = 0, binding = 2) buffer Distances { float distances[]; };");
        sb.AppendLine("layout(push_constant) uniform PushConstants { uint count; } pc;");
        sb.AppendLine();
        sb.AppendLine("float silu(float x) { return x / (1.0 + exp(-x)); }");
        sb.AppendLine();
        sb.AppendLine("float evaluate(vec3 p)");
        sb.AppendLine("{");
        sb.AppendLine($"    float encoded[{enc}];");
        sb.AppendLine("    encoded[0] = p.x; encoded[1] = p.y; encoded[2] = p.z;");
        sb.AppendLine("    int idx = 3;");
        sb.AppendLine("    for (int f = 1; f <= " + freq + "; f++)");
        sb.AppendLine("    {");
        sb.AppendLine("        float freq = 6.2831853 * float(f);");
        sb.AppendLine("        encoded[idx++] = sin(freq * p.x); encoded[idx++] = cos(freq * p.x);");
        sb.AppendLine("        encoded[idx++] = sin(freq * p.y); encoded[idx++] = cos(freq * p.y);");
        sb.AppendLine("        encoded[idx++] = sin(freq * p.z); encoded[idx++] = cos(freq * p.z);");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    int w = 0;");
        sb.AppendLine($"    float hidden[{hidden}];");
        sb.AppendLine($"    float temp[{hidden}];");
        sb.AppendLine($"    float residual[{hidden}];");
        sb.AppendLine();
        // Layer1 weights then bias
        sb.AppendLine($"    for (int i = 0; i < {hidden}; i++)");
        sb.AppendLine("    {");
        sb.AppendLine($"        float sum = weights[w + {enc * hidden} + i];");
        sb.AppendLine($"        for (int j = 0; j < {enc}; j++)");
        sb.AppendLine($"            sum += encoded[j] * weights[w + i * {enc} + j];");
        sb.AppendLine("        hidden[i] = silu(sum);");
        sb.AppendLine("    }");
        sb.AppendLine($"    w += {enc * hidden + hidden};");
        sb.AppendLine();

        for (int b = 0; b < res; b++)
        {
            sb.AppendLine($"    // Residual block {b}");
            sb.AppendLine($"    for (int i = 0; i < {hidden}; i++) residual[i] = hidden[i];");
            sb.AppendLine("    {");
            sb.AppendLine("        float mean = 0.0;");
            sb.AppendLine($"        for (int i = 0; i < {hidden}; i++) mean += hidden[i];");
            sb.AppendLine($"        mean /= {hidden}.0;");
            sb.AppendLine("        float variance = 0.0;");
            sb.AppendLine($"        for (int i = 0; i < {hidden}; i++) {{ float d = hidden[i] - mean; variance += d * d; }}");
            sb.AppendLine($"        variance /= {hidden}.0;");
            sb.AppendLine("        float invStd = inversesqrt(variance + 1e-5);");
            // gamma at w + H*H + H, beta at w + H*H + 2H — after linear weights/bias
            // Layout per block: weights[H*H], bias[H], gamma[H], beta[H]
            sb.AppendLine($"        int gammaBase = w + {hidden * hidden + hidden};");
            sb.AppendLine($"        int betaBase = w + {hidden * hidden + hidden * 2};");
            sb.AppendLine($"        for (int i = 0; i < {hidden}; i++)");
            sb.AppendLine("            hidden[i] = (hidden[i] - mean) * invStd * weights[gammaBase + i] + weights[betaBase + i];");
            sb.AppendLine("    }");
            sb.AppendLine($"    for (int i = 0; i < {hidden}; i++)");
            sb.AppendLine("    {");
            sb.AppendLine($"        float sum = weights[w + {hidden * hidden} + i];");
            sb.AppendLine($"        for (int j = 0; j < {hidden}; j++)");
            sb.AppendLine($"            sum += hidden[j] * weights[w + i * {hidden} + j];");
            sb.AppendLine("        temp[i] = silu(sum);");
            sb.AppendLine("    }");
            sb.AppendLine($"    for (int i = 0; i < {hidden}; i++) hidden[i] = temp[i] + residual[i];");
            sb.AppendLine($"    w += {hidden * hidden + hidden * 3};");
            sb.AppendLine();
        }

        sb.AppendLine("    float result = weights[w + " + hidden + "];");
        sb.AppendLine($"    for (int i = 0; i < {hidden}; i++)");
        sb.AppendLine("        result += hidden[i] * weights[w + i];");
        sb.AppendLine("    return result;");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("void main()");
        sb.AppendLine("{");
        sb.AppendLine("    uint i = gl_GlobalInvocationID.x;");
        sb.AppendLine("    if (i >= pc.count) return;");
        sb.AppendLine("    distances[i] = evaluate(positions[i].xyz);");
        sb.AppendLine("}");
        return sb.ToString();
    }

    /// <summary>
    /// Obtient du SPIR-V : toolchain (glslang/DXC) en priorité, sinon null.
    /// </summary>
    public static bool TryGetSpirv(out byte[] spirv, out string log)
    {
        string glsl = GenerateGlsl();
        if (SpirvToolchain.TryCompileGlsl(glsl, out spirv, out log))
            return true;

        string hlsl = NeuralComputeShaderGenerator.GenerateComputeShaderForSpirv();
        if (SpirvToolchain.TryCompileHlsl(hlsl, "CSMain", out spirv, out string hlslLog))
        {
            log = hlslLog;
            return true;
        }

        log = $"No SPIR-V compiler available. glslang: {log}; dxc: {hlslLog}";
        spirv = Array.Empty<byte>();
        return false;
    }

    /// <summary>
    /// Aplatit les poids DeepMicroMLP dans l'ordre attendu par le GLSL.
    /// </summary>
    public static float[] FlattenWeights(DeepMicroMLP network)
    {
        // Same order as Serialize()
        return network.Serialize();
    }

    /// <summary>
    /// Pack Vector3 → Vector4 (std430 alignment) pour le buffer Positions.
    /// </summary>
    public static float[] PackPositions(ReadOnlySpan<System.Numerics.Vector3> points)
    {
        var packed = new float[points.Length * 4];
        for (int i = 0; i < points.Length; i++)
        {
            packed[i * 4] = points[i].X;
            packed[i * 4 + 1] = points[i].Y;
            packed[i * 4 + 2] = points[i].Z;
            packed[i * 4 + 3] = 0f;
        }
        return packed;
    }
}
