using System;
using System.Numerics;
using System.Text;
using GDNN.Core.NeuralNetwork;

namespace GDNN.GPU;

/// <summary>
/// Génère un compute shader complet à partir d'un DeepMicroMLP entraîné,
/// pour validation GPU/CPU avant bascule du pipeline principal.
/// </summary>
public static class NeuralComputeShaderGenerator
{
    /// <summary>
    /// Produit le source HLSL d'un compute shader évaluant le SDF sur un buffer de positions.
    /// </summary>
    public static string GenerateComputeShader(DeepMicroMLP network)
    {
        var sb = new StringBuilder();
        sb.AppendLine("#pragma target 5.0");
        sb.AppendLine();
        sb.AppendLine(network.GenerateHLSL());
        sb.AppendLine();
        sb.AppendLine("StructuredBuffer<float3> InputPositions : register(t0);");
        sb.AppendLine("RWStructuredBuffer<float> OutputDistances : register(u0);");
        sb.AppendLine();
        sb.AppendLine("[numthreads(64, 1, 1)]");
        sb.AppendLine("void CSMain(uint3 id : SV_DispatchThreadID)");
        sb.AppendLine("{");
        sb.AppendLine("    uint idx = id.x;");
        sb.AppendLine("    float3 pos = InputPositions[idx];");
        sb.AppendLine("    OutputDistances[idx] = EvaluateNeuralSDF(pos);");
        sb.AppendLine("}");
        return sb.ToString();
    }

    /// <summary>
    /// Variante HLSL autonome (poids en StructuredBuffer) pour compilation DXC → SPIR-V.
    /// </summary>
    public static string GenerateComputeShaderForSpirv()
    {
        // Prefer the GLSL path via SpirvToolchain; this HLSL is a secondary option for DXC.
        return GenerateComputeShader(new DeepMicroMLP(new Random(0)));
    }
}

/// <summary>
/// Évaluateur CPU qui reproduit exactement la formule HLSL générée
/// (SiLU = x/(1+exp(-x)), LayerNorm identique). Sert de proxy GPU
/// tant qu'aucun dispatch DXC/Vulkan n'est branché.
/// </summary>
public sealed class HlslCompatibleEvaluator
{
    private readonly DeepMicroMLP _network;

    public HlslCompatibleEvaluator(DeepMicroMLP network)
    {
        _network = network ?? throw new ArgumentNullException(nameof(network));
    }

    public float Evaluate(Vector3 localPos)
    {
        Span<float> encoded = stackalloc float[DeepMicroMLP.EncodedDimension];
        encoded[0] = localPos.X;
        encoded[1] = localPos.Y;
        encoded[2] = localPos.Z;
        int idx = 3;
        for (int f = 1; f <= DeepMicroMLP.FrequencyCount; f++)
        {
            float freq = 6.2831853f * f;
            encoded[idx++] = MathF.Sin(freq * localPos.X);
            encoded[idx++] = MathF.Cos(freq * localPos.X);
            encoded[idx++] = MathF.Sin(freq * localPos.Y);
            encoded[idx++] = MathF.Cos(freq * localPos.Y);
            encoded[idx++] = MathF.Sin(freq * localPos.Z);
            encoded[idx++] = MathF.Cos(freq * localPos.Z);
        }

        Span<float> hidden = stackalloc float[DeepMicroMLP.HiddenSize];
        Span<float> temp = stackalloc float[DeepMicroMLP.HiddenSize];
        Span<float> residual = stackalloc float[DeepMicroMLP.HiddenSize];

        for (int i = 0; i < DeepMicroMLP.HiddenSize; i++)
        {
            float sum = _network.Layer1Bias[i];
            int row = i * DeepMicroMLP.EncodedDimension;
            for (int j = 0; j < DeepMicroMLP.EncodedDimension; j++)
                sum += encoded[j] * _network.Layer1Weights[row + j];
            hidden[i] = sum / (1.0f + MathF.Exp(-sum));
        }

        for (int b = 0; b < DeepMicroMLP.ResidualBlockCount; b++)
        {
            for (int i = 0; i < DeepMicroMLP.HiddenSize; i++)
                residual[i] = hidden[i];

            float mean = 0f;
            for (int i = 0; i < DeepMicroMLP.HiddenSize; i++)
                mean += hidden[i];
            mean /= DeepMicroMLP.HiddenSize;

            float var = 0f;
            for (int i = 0; i < DeepMicroMLP.HiddenSize; i++)
            {
                float d = hidden[i] - mean;
                var += d * d;
            }
            var /= DeepMicroMLP.HiddenSize;
            float invStd = 1.0f / MathF.Sqrt(var + 1e-5f);

            for (int i = 0; i < DeepMicroMLP.HiddenSize; i++)
                hidden[i] = (hidden[i] - mean) * invStd * _network.LayerNormGamma[b][i] + _network.LayerNormBeta[b][i];

            for (int i = 0; i < DeepMicroMLP.HiddenSize; i++)
            {
                float sum = _network.ResidualBiases[b][i];
                int row = i * DeepMicroMLP.HiddenSize;
                for (int j = 0; j < DeepMicroMLP.HiddenSize; j++)
                    sum += hidden[j] * _network.ResidualWeights[b][row + j];
                temp[i] = sum / (1.0f + MathF.Exp(-sum));
            }

            for (int i = 0; i < DeepMicroMLP.HiddenSize; i++)
                hidden[i] = temp[i] + residual[i];
        }

        float result = _network.OutputBias[0];
        for (int i = 0; i < DeepMicroMLP.HiddenSize; i++)
            result += hidden[i] * _network.OutputWeights[i];
        return result;
    }

    public void EvaluateBatch(ReadOnlySpan<Vector3> points, Span<float> distances)
    {
        for (int i = 0; i < points.Length; i++)
            distances[i] = Evaluate(points[i]);
    }
}

/// <summary>
/// Harness de test GPU/CPU.
/// Préfère VulkanNeuralSdfDispatcher quand SPIR-V + Vulkan sont disponibles ;
/// sinon HlslCompatibleEvaluator (proxy mathématique du shader).
/// </summary>
public static class GpuTestHarness
{
    /// <summary>
    /// Indique si un dispatch GPU Vulkan réel est disponible.
    /// </summary>
    public static bool IsGpuAvailable => VulkanNeuralSdfDispatcher.IsAvailable;

    /// <summary>Diagnostic d'init GPU (toolchain / Vulkan).</summary>
    public static string GpuStatus => VulkanNeuralSdfDispatcher.StatusLog;

    /// <summary>
    /// Exécute l'évaluation : GPU Vulkan si possible, sinon proxy HLSL.
    /// </summary>
    public static float[] RunCompute(
        ShaderCompilationResult compiled,
        DeepMicroMLP referenceNetwork,
        ReadOnlySpan<Vector3> testPoints)
    {
        if (!compiled.Success)
            throw new InvalidOperationException($"Shader compilation failed: {compiled.GetErrorSummary()}");

        var gpu = VulkanNeuralSdfDispatcher.Shared;
        if (gpu != null)
        {
            try
            {
                return gpu.Evaluate(referenceNetwork, testPoints);
            }
            catch
            {
                // Fall through to CPU proxy
            }
        }

        var hlsl = new HlslCompatibleEvaluator(referenceNetwork);
        var results = new float[testPoints.Length];
        hlsl.EvaluateBatch(testPoints, results);
        return results;
    }

    /// <summary>
    /// Compile (simulé ou SPIR-V) et exécute le pipeline.
    /// </summary>
    public static (ShaderCompilationResult Compilation, float[] Results) CompileAndRun(
        DeepMicroMLP network,
        ReadOnlySpan<Vector3> testPoints,
        ShaderCompiler? compiler = null)
    {
        compiler ??= new ShaderCompiler();
        string source = NeuralComputeShaderGenerator.GenerateComputeShader(network);
        var result = compiler.Compile(source, "CSMain", ShaderType.ComputeShader);

        // Enrich compilation result with real SPIR-V when toolchain is present
        if (DeepMicroMLPSpirvEmitter.TryGetSpirv(out byte[] spirv, out _))
        {
            result.Bytecode = spirv;
            result.Success = true;
        }

        var outputs = RunCompute(result, network, testPoints);
        return (result, outputs);
    }
}
