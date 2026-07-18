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
// FILE: ShaderGenerator.cs
// PATH: GPU/ShaderGenerator.cs
// ============================================================


using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using GDNN.Core.NeuralNetwork;

namespace GDNN.GPU;

/// <summary>
/// Activation function types that can be emitted as HLSL code.
/// </summary>
public enum ShaderActivationFunction
{
    /// <summary>Rectified Linear Unit: max(0, x).</summary>
    ReLU,

    /// <summary>Leaky ReLU: x > 0 ? x : alpha * x.</summary>
    LeakyReLU,

    /// <summary>Sigmoid: 1 / (1 + exp(-x)).</summary>
    Sigmoid,

    /// <summary>Hyperbolic tangent.</summary>
    Tanh,

    /// <summary>SiLU / Swish: x * sigmoid(x).</summary>
    SiLU,

    /// <summary>Gaussian Error Linear Unit.</summary>
    GELU,

    /// <summary>No activation (linear pass-through).</summary>
    None
}

/// <summary>
/// Quality levels that control shader complexity and feature inclusion.
/// </summary>
public enum ShaderQualityLevel
{
    /// <summary>Minimal features, reduced precision, fewer ray march steps.</summary>
    Low = 0,

    /// <summary>Balanced features for mid-range hardware.</summary>
    Medium = 1,

    /// <summary>Full features with high-precision paths.</summary>
    High = 2,

    /// <summary>Ultra quality with maximum sample counts and effects.</summary>
    Ultra = 3
}

/// <summary>
/// Shader variant feature flags used for permutation generation.
/// </summary>
[Flags]
public enum ShaderFeatures
{
    /// <summary>No features enabled.</summary>
    None = 0,

    /// <summary>Enable shadow ray evaluation.</summary>
    Shadows = 1 << 0,

    /// <summary>Enable reflection rays.</summary>
    Reflections = 1 << 1,

    /// <summary>Enable refraction rays.</summary>
    Refractions = 1 << 2,

    /// <summary>Enable ambient occlusion.</summary>
    AmbientOcclusion = 1 << 3,

    /// <summary>Enable hierarchical sphere tracing acceleration.</summary>
    HierarchicalTracing = 1 << 4,

    /// <summary>Enable binary search refinement.</summary>
    BinarySearch = 1 << 5,

    /// <summary>Enable normal mapping perturbation.</summary>
    NormalMapping = 1 << 6,

    /// <summary>Enable parallax occlusion mapping.</summary>
    ParallaxOcclusion = 1 << 7,

    /// <summary>Enable temporal reprojection.</summary>
    TemporalReprojection = 1 << 8,

    /// <summary>Enable screen-space ambient occlusion.</summary>
    SSAO = 1 << 9,

    /// <summary>Enable subsurface scattering approximation.</summary>
    SubsurfaceScattering = 1 << 10,

    /// <summary>Enable volumetric fog integration.</summary>
    VolumetricFog = 1 << 11,

    /// <summary>Enable curvature-based adaptive step scaling.</summary>
    AdaptiveStepping = 1 << 12,

    /// <summary>Enable two-sided surface rendering.</summary>
    TwoSided = 1 << 13,

    /// <summary>All features enabled.</summary>
    All = (1 << 14) - 1
}

/// <summary>
/// Configuration parameters for shader generation.
/// </summary>
public sealed class ShaderGeneratorConfig
{
    /// <summary>Number of hidden layers in the neural network architecture.</summary>
    public int LayerCount { get; set; } = 3;

    /// <summary>Neurons per hidden layer (max 8).</summary>
    public int HiddenSize { get; set; } = 8;

    /// <summary>Input dimension (3 for spatial coordinates).</summary>
    public int InputDimension { get; set; } = 3;

    /// <summary>Output dimension (1 for SDF, 3 for color, etc.).</summary>
    public int OutputDimension { get; set; } = 1;

    /// <summary>Activation function for hidden layers.</summary>
    public ShaderActivationFunction Activation { get; set; } = ShaderActivationFunction.ReLU;

    /// <summary>Output layer activation function.</summary>
    public ShaderActivationFunction OutputActivation { get; set; } = ShaderActivationFunction.None;

    /// <summary>Shader quality level.</summary>
    public ShaderQualityLevel Quality { get; set; } = ShaderQualityLevel.High;

    /// <summary>Enabled feature flags.</summary>
    public ShaderFeatures Features { get; set; } = ShaderFeatures.Shadows | ShaderFeatures.BinarySearch | ShaderFeatures.HierarchicalTracing;

    /// <summary>Maximum sphere tracing steps per quality level.</summary>
    public int MaxRayMarchSteps => Quality switch
    {
        ShaderQualityLevel.Low => 32,
        ShaderQualityLevel.Medium => 64,
        ShaderQualityLevel.High => 128,
        ShaderQualityLevel.Ultra => 256,
        _ => 128
    };

    /// <summary>Surface intersection threshold.</summary>
    public float SurfaceThreshold => Quality switch
    {
        ShaderQualityLevel.Low => 0.01f,
        ShaderQualityLevel.Medium => 0.005f,
        ShaderQualityLevel.High => 0.001f,
        ShaderQualityLevel.Ultra => 0.0005f,
        _ => 0.001f
    };

    /// <summary>Number of binary search refinement iterations.</summary>
    public int BinarySearchIterations => Quality switch
    {
        ShaderQualityLevel.Low => 2,
        ShaderQualityLevel.Medium => 4,
        ShaderQualityLevel.High => 8,
        ShaderQualityLevel.Ultra => 16,
        _ => 8
    };

    /// <summary>Shadow ray softness factor.</summary>
    public float ShadowSoftness => Quality switch
    {
        ShaderQualityLevel.Low => 0.5f,
        ShaderQualityLevel.Medium => 1.0f,
        ShaderQualityLevel.High => 2.0f,
        ShaderQualityLevel.Ultra => 4.0f,
        _ => 1.0f
    };

    /// <summary>AO sample count.</summary>
    public int AOSampleCount => Quality switch
    {
        ShaderQualityLevel.Low => 4,
        ShaderQualityLevel.Medium => 8,
        ShaderQualityLevel.High => 16,
        ShaderQualityLevel.Ultra => 32,
        _ => 16
    };

    /// <summary>Maximum reflection bounces.</summary>
    public int MaxReflectionBounces => Quality switch
    {
        ShaderQualityLevel.Low => 0,
        ShaderQualityLevel.Medium => 1,
        ShaderQualityLevel.High => 2,
        ShaderQualityLevel.Ultra => 4,
        _ => 2
    };

    /// <summary>Hierarchical tracing level count.</summary>
    public int HierarchicalLevels => Quality switch
    {
        ShaderQualityLevel.Low => 0,
        ShaderQualityLevel.Medium => 2,
        ShaderQualityLevel.High => 4,
        ShaderQualityLevel.Ultra => 6,
        _ => 4
    };

    /// <summary>Sphere tracing relaxation factor (0..1).</summary>
    public float Relaxation { get; set; } = 0.8f;

    /// <summary>Maximum trace distance.</summary>
    public float MaxTraceDistance { get; set; } = 100.0f;

    /// <summary>Normal bias to prevent self-intersection.</summary>
    public float NormalBias { get; set; } = 0.002f;

    /// <summary>Leaky ReLU alpha parameter.</summary>
    public float LeakyReLUAlpha { get; set; } = 0.01f;

    /// <summary>GELU approximation constant.</summary>
    public float GELUConstant { get; set; } = 0.044715f;

    /// <summary>Whether to emit debug visualization outputs.</summary>
    public bool EmitDebugOutput { get; set; } = false;

    /// <summary>Whether to use half-precision for intermediate calculations where possible.</summary>
    public bool UseHalfPrecision { get; set; } = false;

    /// <summary>Target shader model version (e.g. "5_0", "6_0", "6_5").</summary>
    public string ShaderModel { get; set; } = "5_0";

    /// <summary>Entry point name for the pixel shader.</summary>
    public string PixelShaderEntryPoint { get; set; } = "PSMain";

    /// <summary>Entry point name for the vertex shader.</summary>
    public string VertexShaderEntryPoint { get; set; } = "VSMain";

    /// <summary>Creates a configuration for a MicroMLP-compatible architecture.</summary>
    public static ShaderGeneratorConfig ForMicroMLP(ShaderQualityLevel quality = ShaderQualityLevel.High)
    {
        return new ShaderGeneratorConfig
        {
            LayerCount = 3,
            HiddenSize = 8,
            InputDimension = 3,
            OutputDimension = 1,
            Activation = ShaderActivationFunction.ReLU,
            OutputActivation = ShaderActivationFunction.None,
            Quality = quality
        };
    }

    /// <summary>
    /// Computes a unique variant key from the configuration for caching.
    /// </summary>
    public ulong ComputeVariantKey()
    {
        ulong key = 0;
        key |= (ulong)(uint)LayerCount;
        key |= (ulong)(uint)HiddenSize << 4;
        key |= (ulong)(uint)InputDimension << 8;
        key |= (ulong)(uint)OutputDimension << 12;
        key |= (ulong)(uint)Activation << 16;
        key |= (ulong)(uint)OutputActivation << 20;
        key |= (ulong)(uint)Quality << 24;
        key |= (ulong)(uint)Features << 28;
        return key;
    }
}

/// <summary>
/// Generates HLSL shader code for neural SDF evaluation on the GPU.
/// Produces complete pixel shaders with sphere tracing, constant buffer layouts
/// matching NeuralLayerWeights, and vertex shaders for full-screen quad rendering.
/// Supports parameterized network architectures, activation functions, quality
/// levels, and feature permutations.
/// </summary>
public sealed class ShaderGenerator
{
    private readonly ShaderGeneratorConfig _config;
    private readonly HLSLCodeGenerator _hlsl;

    /// <summary>
    /// Initializes a new ShaderGenerator with the specified configuration.
    /// </summary>
    /// <param name="config">Shader generation configuration. Uses defaults if null.</param>
    public ShaderGenerator(ShaderGeneratorConfig? config = null)
    {
        _config = config ?? new ShaderGeneratorConfig();
        _hlsl = new HLSLCodeGenerator();
    }

    /// <summary>
    /// Gets the current configuration.
    /// </summary>
    public ShaderGeneratorConfig Config => _config;

    /// <summary>
    /// Generates the complete HLSL constant buffer declaration for neural network weights.
    /// Matches the NeuralLayerWeights struct layout with 16-byte aligned float4 packing.
    /// </summary>
    /// <returns>HLSL constant buffer source code.</returns>
    public string GenerateConstantBufferDeclaration()
    {
        var sb = new StringBuilder();
        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Neural SDF Constant Buffer Layout");
        sb.AppendLine($"// Architecture: {_config.InputDimension} -> {_config.HiddenSize}x{_config.LayerCount} -> {_config.OutputDimension}");
        sb.AppendLine("// All constants are packed into float4 arrays for 16-byte alignment.");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();

        int hiddenSize = _config.HiddenSize;
        int inputDim = _config.InputDimension;
        int outputDim = _config.OutputDimension;

        // Layer 1: inputDim -> hiddenSize
        int layer1WeightFloats = inputDim * hiddenSize;
        int layer1BiasFloats = hiddenSize;
        int layer1TotalFloats = layer1WeightFloats + layer1BiasFloats;
        int layer1Float4Count = (layer1TotalFloats + 3) / 4;

        // Hidden layers: hiddenSize -> hiddenSize
        int hiddenWeightFloats = hiddenSize * hiddenSize;
        int hiddenBiasFloats = hiddenSize;
        int hiddenTotalFloats = hiddenWeightFloats + hiddenBiasFloats;
        int hiddenFloat4Count = (hiddenTotalFloats + 3) / 4;

        // Output layer: hiddenSize -> outputDim
        int outputWeightFloats = hiddenSize * outputDim;
        int outputBiasFloats = outputDim;
        int outputTotalFloats = outputWeightFloats + outputBiasFloats;
        int outputFloat4Count = (outputTotalFloats + 3) / 4;

        sb.AppendLine("cbuffer NeuralWeights : register(b0)");
        sb.AppendLine("{");
        sb.AppendLine($"    // Layer 1: {inputDim} inputs -> {hiddenSize} neurons ({layer1TotalFloats} floats)");
        sb.AppendLine($"    float4 Layer0_Weights[{layer1Float4Count}];");
        sb.AppendLine();
        sb.AppendLine($"    // Hidden layers (x{_config.LayerCount - 2}): {hiddenSize} -> {hiddenSize} ({hiddenTotalFloats} floats each)");
        for (int i = 1; i < _config.LayerCount - 1; i++)
        {
            sb.AppendLine($"    float4 Layer{i}_Weights[{hiddenFloat4Count}];");
        }
        sb.AppendLine();
        sb.AppendLine($"    // Output layer: {hiddenSize} -> {outputDim} ({outputTotalFloats} floats)");
        sb.AppendLine($"    float4 Output_Weights[{outputFloat4Count}];");
        sb.AppendLine("};");
        sb.AppendLine();

        // Scene parameters buffer
        sb.AppendLine("cbuffer SceneParams : register(b1)");
        sb.AppendLine("{");
        sb.AppendLine("    float4x4 ViewMatrix;");
        sb.AppendLine("    float4x4 ProjectionMatrix;");
        sb.AppendLine("    float4x4 ViewProjectionMatrix;");
        sb.AppendLine("    float4x4 InvViewMatrix;");
        sb.AppendLine("    float4x4 InvProjectionMatrix;");
        sb.AppendLine("    float4 CameraPosition;   // xyz = pos, w = time");
        sb.AppendLine("    float4 CameraForward;    // xyz = forward, w = nearPlane");
        sb.AppendLine("    float4 CameraRight;      // xyz = right, w = farPlane");
        sb.AppendLine("    float4 CameraUp;         // xyz = up, w = fovY");
        sb.AppendLine("    float4 LightDirection;   // xyz = dir, w = intensity");
        sb.AppendLine("    float4 LightColor;       // xyz = color, w = ambient");
        sb.AppendLine("    float4 ScreenParams;     // x = width, y = height, z = 1/width, w = 1/height");
        sb.AppendLine("    float4 TraceParams;      // x = maxSteps, y = threshold, z = relaxation, w = maxDist");
        sb.AppendLine("};");
        sb.AppendLine();

        return sb.ToString();
    }

    /// <summary>
    /// Generates the HLSL activation function definition.
    /// </summary>
    /// <param name="function">The activation function to generate.</param>
    /// <param name="functionName">Name for the generated function.</param>
    /// <returns>HLSL function source code.</returns>
    public string GenerateActivationFunction(ShaderActivationFunction function, string functionName)
    {
        var sb = new StringBuilder();
        float alpha = _config.LeakyReLUAlpha;
        float geluC = _config.GELUConstant;

        switch (function)
        {
            case ShaderActivationFunction.ReLU:
                sb.AppendLine($"float {functionName}(float x)");
                sb.AppendLine("{");
                sb.AppendLine("    return max(0.0, x);");
                sb.AppendLine("}");
                break;

            case ShaderActivationFunction.LeakyReLU:
                sb.AppendLine($"float {functionName}(float x)");
                sb.AppendLine("{");
                sb.AppendLine($"    return x > 0.0 ? x : {alpha:F4} * x;");
                sb.AppendLine("}");
                break;

            case ShaderActivationFunction.Sigmoid:
                sb.AppendLine($"float {functionName}(float x)");
                sb.AppendLine("{");
                sb.AppendLine("    return 1.0 / (1.0 + exp(-x));");
                sb.AppendLine("}");
                break;

            case ShaderActivationFunction.Tanh:
                sb.AppendLine($"float {functionName}(float x)");
                sb.AppendLine("{");
                sb.AppendLine("    return tanh(x);");
                sb.AppendLine("}");
                break;

            case ShaderActivationFunction.SiLU:
                sb.AppendLine($"float {functionName}(float x)");
                sb.AppendLine("{");
                sb.AppendLine("    return x / (1.0 + exp(-x));");
                sb.AppendLine("}");
                break;

            case ShaderActivationFunction.GELU:
                sb.AppendLine($"float {functionName}(float x)");
                sb.AppendLine("{");
                sb.AppendLine($"    float c = {geluC:F6};");
                sb.AppendLine("    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / 3.14159265) * (x + c * x * x * x)));");
                sb.AppendLine("}");
                break;

            case ShaderActivationFunction.None:
                sb.AppendLine($"float {functionName}(float x)");
                sb.AppendLine("{");
                sb.AppendLine("    return x;");
                sb.AppendLine("}");
                break;
        }

        return sb.ToString();
    }

    /// <summary>
    /// Generates the batch activation function that operates on an array of values.
    /// </summary>
    /// <param name="function">The activation function type.</param>
    /// <param name="functionName">Name for the generated function.</param>
    /// <param name="size">Number of elements to process.</param>
    /// <returns>HLSL function source code.</returns>
    public string GenerateBatchActivationFunction(ShaderActivationFunction function, string functionName, int size)
    {
        var sb = new StringBuilder();
        string singleName = $"_{functionName}_single";
        sb.Append(GenerateActivationFunction(function, singleName));

        sb.AppendLine($"void {functionName}(inout float values[{size}])");
        sb.AppendLine("{");
        sb.AppendLine($"    [unroll]");
        sb.AppendLine($"    for (int i = 0; i < {size}; i++)");
        sb.AppendLine($"    {{");
        sb.AppendLine($"        values[i] = {singleName}(values[i]);");
        sb.AppendLine($"    }}");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates the neural SDF evaluation function that performs a complete forward pass.
    /// </summary>
    /// <returns>HLSL function source code.</returns>
    public string GenerateEvaluateNeuralSDF()
    {
        var sb = new StringBuilder();
        int hidden = _config.HiddenSize;
        int inputDim = _config.InputDimension;
        int layers = _config.LayerCount;

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Neural SDF Evaluation Function");
        sb.AppendLine("// Performs forward pass through the Micro-MLP network.");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();

        // Emit single-value activation
        sb.Append(GenerateActivationFunction(_config.Activation, "ActivationFunc"));
        sb.AppendLine();

        // Emit batch activation
        sb.Append(GenerateBatchActivationFunction(_config.Activation, "ActivationBatch", hidden));
        sb.AppendLine();

        // EvaluateNeuralSDF function
        sb.AppendLine("float EvaluateNeuralSDF(float3 localPos)");
        sb.AppendLine("{");

        // Declare intermediate buffers
        if (layers > 2)
        {
            sb.AppendLine($"    float layer0[{hidden}];");
            for (int i = 1; i < layers - 1; i++)
            {
                sb.AppendLine($"    float layer{i}[{hidden}];");
            }
        }
        sb.AppendLine($"    float output[{_config.OutputDimension}];");
        sb.AppendLine();

        // Layer 0: input -> hidden
        sb.AppendLine("    // Layer 0: Input -> Hidden");
        int layer0WeightFloats = inputDim * hidden;
        if (inputDim == 3)
        {
            // Optimized path for 3D input
            sb.AppendLine("    [unroll]");
            sb.AppendLine($"    for (int n0 = 0; n0 < {hidden}; n0++)");
            sb.AppendLine("    {");
            sb.AppendLine("        int wIdx = n0;");
            sb.AppendLine($"        float sum = dot(localPos, Layer0_Weights[wIdx].xyz) + Layer0_Weights[wIdx].w;");
            sb.AppendLine($"        layer0[n0] = ActivationFunc(sum);");
            sb.AppendLine("    }");
        }
        else
        {
            // General path
            int float4Count0 = (inputDim + 3) / 4;
            sb.AppendLine("    [unroll]");
            sb.AppendLine($"    for (int n0 = 0; n0 < {hidden}; n0++)");
            sb.AppendLine("    {");
            sb.AppendLine("        float sum = 0.0;");
            sb.AppendLine($"        [unroll]");
            sb.AppendLine($"        for (int i0 = 0; i0 < {inputDim}; i0++)");
            sb.AppendLine("        {");
            sb.AppendLine("            int wIdx = n0 * inputDim + i0;");
            sb.AppendLine($"            int bufIdx = wIdx / 4;");
            sb.AppendLine($"            int compIdx = wIdx % 4;");
            sb.AppendLine($"            sum += Layer0_Weights[bufIdx][compIdx] * ");
            if (inputDim <= 4)
                sb.AppendLine($"                (i0 == 0 ? localPos.x : (i0 == 1 ? localPos.y : (i0 == 2 ? localPos.z : 0.0)));");
            else
                sb.AppendLine($"                localPos[i0];");
            sb.AppendLine("        }");
            sb.AppendLine($"        sum += Layer0_Weights[{hidden} + n0 / 4][n0 % 4];");
            sb.AppendLine($"        layer0[n0] = ActivationFunc(sum);");
            sb.AppendLine("    }");
        }
        sb.AppendLine();

        // Hidden layers
        for (int layer = 1; layer < layers - 1; layer++)
        {
            string prevLayer = $"layer{layer - 1}";
            string currLayer = $"layer{layer}";

            sb.AppendLine($"    // Layer {layer}: Hidden -> Hidden");
            sb.AppendLine("    [unroll]");
            sb.AppendLine($"    for (int n{layer} = 0; n{layer} < {hidden}; n{layer}++)");
            sb.AppendLine("    {");
            sb.AppendLine("        float sum = 0.0;");
            sb.AppendLine($"        [unroll]");
            sb.AppendLine($"        for (int i{layer} = 0; i{layer} < {hidden}; i{layer}++)");
            sb.AppendLine("        {");
            sb.AppendLine($"            int wIdx = n{layer} * {hidden} + i{layer};");
            sb.AppendLine($"            int bufIdx = wIdx / 4;");
            sb.AppendLine($"            int compIdx = wIdx % 4;");
            sb.AppendLine($"            sum += Layer{layer}_Weights[bufIdx][compIdx] * {prevLayer}[i{layer}];");
            sb.AppendLine("        }");
            sb.AppendLine($"        sum += Layer{layer}_Weights[{hidden} + n{layer} / 4][n{layer} % 4];");
            sb.AppendLine($"        {currLayer}[n{layer}] = sum;");
            sb.AppendLine("    }");
            sb.AppendLine($"    ActivationBatch({currLayer});");
            sb.AppendLine();
        }

        // Output layer
        string lastHidden = $"layer{layers - 2}";
        sb.AppendLine("    // Output layer");
        for (int outIdx = 0; outIdx < _config.OutputDimension; outIdx++)
        {
            sb.AppendLine("    {");
            sb.AppendLine("        float sum = 0.0;");
            sb.AppendLine($"        [unroll]");
            sb.AppendLine($"        for (int io = 0; io < {hidden}; io++)");
            sb.AppendLine("        {");
            sb.AppendLine($"            int wIdx = io * {_config.OutputDimension} + {outIdx};");
            sb.AppendLine($"            int bufIdx = wIdx / 4;");
            sb.AppendLine($"            int compIdx = wIdx % 4;");
            sb.AppendLine($"            sum += Output_Weights[bufIdx][compIdx] * {lastHidden}[io];");
            sb.AppendLine("        }");
            sb.AppendLine($"        sum += Output_Weights[{hidden * _config.OutputDimension / 4 + outIdx / 4}][{outIdx % 4}];");
            if (_config.OutputActivation != ShaderActivationFunction.None)
            {
                sb.AppendLine($"        output[{outIdx}] = ActivationFunc(sum);");
            }
            else
            {
                sb.AppendLine($"        output[{outIdx}] = sum;");
            }
            sb.AppendLine("    }");
        }
        sb.AppendLine();

        sb.AppendLine("    return output[0];");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates the surface normal computation function using central differences.
    /// </summary>
    /// <returns>HLSL function source code.</returns>
    public string GenerateComputeNormal()
    {
        var sb = new StringBuilder();
        float eps = _config.Quality switch
        {
            ShaderQualityLevel.Low => 0.01f,
            ShaderQualityLevel.Medium => 0.005f,
            ShaderQualityLevel.High => 0.001f,
            ShaderQualityLevel.Ultra => 0.0005f,
            _ => 0.001f
        };

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Surface Normal Computation via Central Differences");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();
        sb.AppendLine("float3 ComputeNormal(float3 p)");
        sb.AppendLine("{");
        sb.AppendLine($"    const float eps = {eps:F6};");
        sb.AppendLine("    float d = EvaluateNeuralSDF(p);");
        sb.AppendLine("    float dx = EvaluateNeuralSDF(p + float3(eps, 0, 0)) - d;");
        sb.AppendLine("    float dy = EvaluateNeuralSDF(p + float3(0, eps, 0)) - d;");
        sb.AppendLine("    float dz = EvaluateNeuralSDF(p + float3(0, 0, eps)) - d;");
        sb.AppendLine("    float3 gradient = float3(dx, dy, dz);");
        sb.AppendLine("    float len = length(gradient);");
        sb.AppendLine("    return len > 0.0 ? gradient / len : float3(0, 1, 0);");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates the hierarchical sphere tracing lower-bound function.
    /// </summary>
    /// <returns>HLSL function source code.</returns>
    public string GenerateHierarchicalTrace()
    {
        if (!_config.Features.HasFlag(ShaderFeatures.HierarchicalTracing))
            return string.Empty;

        var sb = new StringBuilder();
        int levels = _config.HierarchicalLevels;

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Hierarchical Sphere Tracing Acceleration");
        sb.AppendLine("// Uses mip-mapped SDF lower bounds for conservative stepping.");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();

        // Texture bindings for hierarchical levels
        for (int i = 0; i < levels; i++)
        {
            sb.AppendLine($"Texture3D<float> HierarchyLevel{i} : register(t{i});");
            sb.AppendLine($"SamplerState HierarchySampler{i} : register(s{i});");
        }
        sb.AppendLine();

        sb.AppendLine("float GetHierarchicalLowerBound(float3 point, float baseLevel)");
        sb.AppendLine("{");
        sb.AppendLine("    float lowerBound = 0.0;");
        sb.AppendLine("    float3 uvw = (point - HierarchyOrigin) * HierarchyScale;");

        sb.AppendLine("    [unroll]");
        sb.AppendLine($"    for (int lvl = 0; lvl < {levels}; lvl++)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (baseLevel >= float(lvl))");
        sb.AppendLine("        {");
        for (int i = 0; i < levels; i++)
        {
            string prefix = i == 0 ? "    " : "    else ";
            sb.AppendLine($"        {prefix}(lvl == {i})");
            sb.AppendLine("        {");
            sb.AppendLine($"            float sampled = HierarchyLevel{i}.SampleLevel(HierarchySampler{i}, uvw, 0).r;");
            sb.AppendLine("            lowerBound = max(lowerBound, sampled);");
            sb.AppendLine("        }");
        }
        sb.AppendLine("        }");
        sb.AppendLine("        uvw *= 0.5;");
        sb.AppendLine("    }");

        sb.AppendLine("    return lowerBound;");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates the shadow evaluation function.
    /// </summary>
    /// <returns>HLSL function source code.</returns>
    public string GenerateShadowFunction()
    {
        if (!_config.Features.HasFlag(ShaderFeatures.Shadows))
            return string.Empty;

        var sb = new StringBuilder();
        int maxSteps = _config.MaxRayMarchSteps / 2;
        float softness = _config.ShadowSoftness;
        float bias = _config.NormalBias * 2.5f;

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Shadow Ray Evaluation");
        sb.AppendLine("// Returns 0.0 (fully shadowed) to 1.0 (fully lit).");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();
        sb.AppendLine("float EvaluateShadow(float3 origin, float3 lightDir)");
        sb.AppendLine("{");
        sb.AppendLine($"    const float bias = {bias:F4};");
        sb.AppendLine($"    const float maxDist = {maxSteps}.0;");
        sb.AppendLine($"    const float softness = {softness:F2};");
        sb.AppendLine();
        sb.AppendLine("    float3 rayOrigin = origin + lightDir * bias;");
        sb.AppendLine("    float totalDist = 0.0;");
        sb.AppendLine("    float result = 1.0;");
        sb.AppendLine();
        sb.AppendLine($"    [loop]");
        sb.AppendLine($"    for (int step = 0; step < {maxSteps}; step++)");
        sb.AppendLine("    {");
        sb.AppendLine("        float3 currentPoint = rayOrigin + lightDir * totalDist;");
        sb.AppendLine("        float sdf = EvaluateNeuralSDF(currentPoint);");
        sb.AppendLine();
        sb.AppendLine("        if (abs(sdf) < 0.001)");
        sb.AppendLine("            return 0.0;");
        sb.AppendLine();
        sb.AppendLine("        result = min(result, softness * sdf / totalDist);");
        sb.AppendLine("        totalDist += abs(sdf) * 0.8;");
        sb.AppendLine();
        sb.AppendLine("        if (totalDist >= maxDist || result < 0.001)");
        sb.AppendLine("            break;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    return max(0.0, result);");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates the ambient occlusion evaluation function.
    /// </summary>
    /// <returns>HLSL function source code.</returns>
    public string GenerateAOFunction()
    {
        if (!_config.Features.HasFlag(ShaderFeatures.AmbientOcclusion))
            return string.Empty;

        var sb = new StringBuilder();
        int samples = _config.AOSampleCount;
        float radius = _config.Quality switch
        {
            ShaderQualityLevel.Low => 0.5f,
            ShaderQualityLevel.Medium => 1.0f,
            ShaderQualityLevel.High => 1.5f,
            ShaderQualityLevel.Ultra => 2.0f,
            _ => 1.0f
        };

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Ambient Occlusion Evaluation");
        sb.AppendLine("// Hemisphere sampling with distance-based weighting.");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();
        sb.AppendLine("float EvaluateAO(float3 point, float3 normal)");
        sb.AppendLine("{");
        sb.AppendLine($"    const int samples = {samples};");
        sb.AppendLine($"    const float radius = {radius:F2};");
        sb.AppendLine($"    const float intensity = 2.0;");
        sb.AppendLine();
        sb.AppendLine("    float occlusion = 0.0;");
        sb.AppendLine("    float stepSize = radius / float(samples);");
        sb.AppendLine();
        sb.AppendLine("    [unroll]");
        sb.AppendLine($"    for (int i = 0; i < {samples}; i++)");
        sb.AppendLine("    {");
        sb.AppendLine("        float t = float(i + 1) * stepSize;");
        sb.AppendLine("        float phi = float(i) * 2.39996323;"); // golden angle
        sb.AppendLine("        float cosTheta = 1.0 - float(i) / float(samples);");
        sb.AppendLine("        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);");
        sb.AppendLine();
        sb.AppendLine("        float3 sampleDir = float3(");
        sb.AppendLine("            sinTheta * cos(phi),");
        sb.AppendLine("            sinTheta * sin(phi),");
        sb.AppendLine("            cosTheta);");
        sb.AppendLine();
        sb.AppendLine("        // Align to hemisphere");
        sb.AppendLine("        sampleDir = normalize(sampleDir + normal);");
        sb.AppendLine();
        sb.AppendLine("        float3 samplePoint = point + sampleDir * t;");
        sb.AppendLine("        float sdf = EvaluateNeuralSDF(samplePoint);");
        sb.AppendLine();
        sb.AppendLine("        if (sdf < 0.0)");
        sb.AppendLine("            occlusion += 1.0;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    return 1.0 - intensity * occlusion / float(samples);");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates the Fresnel reflectance computation function.
    /// </summary>
    /// <returns>HLSL function source code.</returns>
    public string GenerateFresnelFunction()
    {
        var sb = new StringBuilder();

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Fresnel Reflectance (Schlick Approximation)");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();
        sb.AppendLine("float FresnelSchlick(float cosTheta, float f0)");
        sb.AppendLine("{");
        sb.AppendLine("    float oneMinusCos = 1.0 - cosTheta;");
        sb.AppendLine("    float oneMinusCos2 = oneMinusCos * oneMinusCos;");
        sb.AppendLine("    return f0 + (1.0 - f0) * oneMinusCos2 * oneMinusCos2 * oneMinusCos;");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates the full vertex shader for rendering a full-screen quad.
    /// </summary>
    /// <returns>HLSL vertex shader source code.</returns>
    public string GenerateVertexShader()
    {
        var sb = new StringBuilder();

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Full-Screen Quad Vertex Shader");
        sb.AppendLine("// Generates a screen-filling triangle from vertex ID.");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();
        sb.AppendLine("struct VSOutput");
        sb.AppendLine("{");
        sb.AppendLine("    float4 position : SV_POSITION;");
        sb.AppendLine("    float2 texCoord : TEXCOORD0;");
        sb.AppendLine("    float3 viewRay  : TEXCOORD1;");
        sb.AppendLine("};");
        sb.AppendLine();
        sb.AppendLine("cbuffer QuadParams : register(b2)");
        sb.AppendLine("{");
        sb.AppendLine("    float4x4 InvViewProjection;");
        sb.AppendLine("    float4 QuadCameraPos;    // xyz = position, w = tan(fovY/2)");
        sb.AppendLine("    float4 QuadScreenParams; // x = aspect, y = 1/width, z = 1/height, w = 0");
        sb.AppendLine("};");
        sb.AppendLine();
        sb.AppendLine("VSOutput VSMain(uint vertexID : SV_VertexID)");
        sb.AppendLine("{");
        sb.AppendLine("    VSOutput output;");
        sb.AppendLine();
        sb.AppendLine("    // Generate full-screen triangle from vertex ID");
        sb.AppendLine("    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);");
        sb.AppendLine("    output.texCoord = uv;");
        sb.AppendLine("    output.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);");
        sb.AppendLine();
        sb.AppendLine("    // Compute view ray for ray marching");
        sb.AppendLine("    float aspect = QuadScreenParams.x;");
        sb.AppendLine("    float halfHeight = QuadCameraPos.w;");
        sb.AppendLine("    float halfWidth = halfHeight * aspect;");
        sb.AppendLine();
        sb.AppendLine("    float3 right = CameraRight.xyz * halfWidth;");
        sb.AppendLine("    float3 up = CameraUp.xyz * halfHeight;");
        sb.AppendLine("    float3 forward = CameraForward.xyz;");
        sb.AppendLine();
        sb.AppendLine("    output.viewRay = forward + right * (uv.x * 2.0 - 1.0) + up * (1.0 - uv.y * 2.0);");
        sb.AppendLine();
        sb.AppendLine("    return output;");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates the primary pixel shader with sphere tracing and lighting.
    /// </summary>
    /// <returns>HLSL pixel shader source code.</returns>
    public string GeneratePixelShader()
    {
        var sb = new StringBuilder();

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Neural SDF Pixel Shader - Sphere Tracing with Lighting");
        sb.AppendLine($"// Quality: {_config.Quality} | Features: {_config.Features}");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();

        // Include all generated functions
        sb.Append(GenerateConstantBufferDeclaration());
        sb.AppendLine();

        // Texture resources
        if (_config.Features.HasFlag(ShaderFeatures.HierarchicalTracing) && _config.HierarchicalLevels > 0)
        {
            sb.AppendLine("// Hierarchical acceleration textures");
            for (int i = 0; i < _config.HierarchicalLevels; i++)
            {
                sb.AppendLine($"Texture3D<float> HierarchyLevel{i} : register(t{i});");
            }
            sb.AppendLine("SamplerState HierarchySampler : register(s0);");
            sb.AppendLine("float3 HierarchyOrigin = float3(-10, -10, -10);");
            sb.AppendLine("float3 HierarchyScale = float3(0.05, 0.05, 0.05);");
            sb.AppendLine();
        }

        // Generate all helper functions
        sb.Append(GenerateEvaluateNeuralSDF());
        sb.AppendLine();
        sb.Append(GenerateComputeNormal());
        sb.AppendLine();
        sb.Append(GenerateShadowFunction());
        sb.AppendLine();
        sb.Append(GenerateAOFunction());
        sb.AppendLine();
        sb.Append(GenerateFresnelFunction());
        sb.AppendLine();

        // Pixel shader structure
        sb.AppendLine("struct PSOutput");
        sb.AppendLine("{");
        sb.AppendLine("    float4 color : SV_TARGET0;");
        if (_config.EmitDebugOutput)
        {
            sb.AppendLine("    float4 debug  : SV_TARGET1;");
        }
        sb.AppendLine("};");
        sb.AppendLine();

        // Main pixel shader
        sb.AppendLine("PSOutput PSMain(VSOutput input)");
        sb.AppendLine("{");
        sb.AppendLine("    PSOutput result;");
        sb.AppendLine();
        sb.AppendLine("    // Reconstruct ray from interpolated view direction");
        sb.AppendLine("    float3 rayOrigin = CameraPosition.xyz;");
        sb.AppendLine("    float3 rayDir = normalize(input.viewRay);");
        sb.AppendLine();
        sb.AppendLine("    // Sphere tracing");
        sb.AppendLine($"    float totalDistance = 0.0;");
        sb.AppendLine($"    float prevSdf = 1e10;");
        sb.AppendLine($"    bool hit = false;");
        sb.AppendLine($"    float3 hitPoint = float3(0, 0, 0);");
        sb.AppendLine();
        sb.AppendLine($"    [loop]");
        sb.AppendLine($"    for (int step = 0; step < {_config.MaxRayMarchSteps}; step++)");
        sb.AppendLine("    {");
        sb.AppendLine("        float3 currentPoint = rayOrigin + rayDir * totalDistance;");
        sb.AppendLine("        float sdf = EvaluateNeuralSDF(currentPoint);");
        sb.AppendLine();
        sb.AppendLine($"        if (abs(sdf) < {_config.SurfaceThreshold:F6})");
        sb.AppendLine("        {");
        sb.AppendLine("            hitPoint = currentPoint;");
        sb.AppendLine("            hit = true;");
        sb.AppendLine("            break;");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        // Adaptive stepping");
        sb.AppendLine($"        float stepSize = abs(sdf) * {_config.Relaxation:F2};");

        if (_config.Features.HasFlag(ShaderFeatures.HierarchicalTracing))
        {
            sb.AppendLine("        float hierBound = GetHierarchicalLowerBound(currentPoint, 0.0);");
            sb.AppendLine("        stepSize = max(stepSize, hierBound);");
        }

        sb.AppendLine("        stepSize = max(stepSize, 0.0001);");
        sb.AppendLine("        stepSize = min(stepSize, 50.0);");
        sb.AppendLine();
        sb.AppendLine("        // Binary search on sign change");
        if (_config.Features.HasFlag(ShaderFeatures.BinarySearch))
        {
            sb.AppendLine("        if (prevSdf > 0.0 && sdf < 0.0)");
            sb.AppendLine("        {");
            sb.AppendLine("            float lo = totalDistance - abs(prevSdf);");
            sb.AppendLine("            float hi = totalDistance;");
            sb.AppendLine($"            [unroll]");
            sb.AppendLine($"            for (int bs = 0; bs < {_config.BinarySearchIterations}; bs++)");
            sb.AppendLine("            {");
            sb.AppendLine("                float mid = (lo + hi) * 0.5;");
            sb.AppendLine("                float midSdf = EvaluateNeuralSDF(rayOrigin + rayDir * mid);");
            sb.AppendLine("                if (midSdf < 0.0) hi = mid; else lo = mid;");
            sb.AppendLine("            }");
            sb.AppendLine("            hitPoint = rayOrigin + rayDir * ((lo + hi) * 0.5);");
            sb.AppendLine("            hit = true;");
            sb.AppendLine("            break;");
            sb.AppendLine("        }");
        }

        sb.AppendLine();
        sb.AppendLine("        totalDistance += stepSize;");
        sb.AppendLine("        prevSdf = sdf;");
        sb.AppendLine();
        sb.AppendLine($"        if (totalDistance > {_config.MaxTraceDistance:F1})");
        sb.AppendLine("            break;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // Early exit for miss
        sb.AppendLine("    if (!hit)");
        sb.AppendLine("    {");
        sb.AppendLine("        // Sky color / background");
        sb.AppendLine("        float3 skyColor = float3(0.5, 0.7, 1.0);");
        sb.AppendLine("        float3 horizonColor = float3(1.0, 1.0, 1.0);");
        sb.AppendLine("        float skyGradient = pow(max(0.0, rayDir.y) * 0.5 + 0.5, 0.5);");
        sb.AppendLine("        result.color = float4(lerp(horizonColor, skyColor, skyGradient), 1.0);");
        if (_config.EmitDebugOutput)
        {
            sb.AppendLine("        result.debug = float4(0, 0, 0, 1);");
        }
        sb.AppendLine("        return result;");
        sb.AppendLine("    }");
        sb.AppendLine();

        // Surface evaluation
        sb.AppendLine("    // Compute surface properties");
        sb.AppendLine("    float3 normal = ComputeNormal(hitPoint);");
        sb.AppendLine($"    hitPoint += normal * {_config.NormalBias:F4};");
        sb.AppendLine();

        // Lighting model
        sb.AppendLine("    // Lighting");
        sb.AppendLine("    float3 lightDir = normalize(-LightDirection.xyz);");
        sb.AppendLine("    float3 viewDir = normalize(CameraPosition.xyz - hitPoint);");
        sb.AppendLine("    float3 halfVec = normalize(lightDir + viewDir);");
        sb.AppendLine();
        sb.AppendLine("    float nDotL = max(0.0, dot(normal, lightDir));");
        sb.AppendLine("    float nDotH = max(0.0, dot(normal, halfVec));");
        sb.AppendLine("    float nDotV = max(0.0, dot(normal, viewDir));");
        sb.AppendLine();

        // Shadow
        if (_config.Features.HasFlag(ShaderFeatures.Shadows))
        {
            sb.AppendLine("    float shadow = EvaluateShadow(hitPoint, lightDir);");
        }
        else
        {
            sb.AppendLine("    float shadow = 1.0;");
        }

        // AO
        if (_config.Features.HasFlag(ShaderFeatures.AmbientOcclusion))
        {
            sb.AppendLine("    float ao = EvaluateAO(hitPoint, normal);");
        }
        else
        {
            sb.AppendLine("    float ao = 1.0;");
        }

        // Material properties
        sb.AppendLine("    float3 albedo = float3(0.8, 0.8, 0.8);");
        sb.AppendLine("    float specular = pow(nDotH, 64.0);");
        sb.AppendLine("    float3 ambient = LightColor.xyz * LightColor.w * ao;");
        sb.AppendLine("    float3 diffuse = LightColor.xyz * nDotL * shadow;");
        sb.AppendLine("    float3 fresnelColor = float3(1.0, 1.0, 1.0);");
        sb.AppendLine("    float fresnel = FresnelSchlick(nDotV, 0.04);");
        sb.AppendLine();
        sb.AppendLine("    float3 finalColor = albedo * (ambient + diffuse) + fresnelColor * specular * fresnel;");
        sb.AppendLine();
        sb.AppendLine("    result.color = float4(finalColor, 1.0);");

        if (_config.EmitDebugOutput)
        {
            sb.AppendLine("    result.debug = float4(normal * 0.5 + 0.5, 1.0);");
        }

        sb.AppendLine("    return result;");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates a complete shader source file combining all components.
    /// </summary>
    /// <returns>Complete HLSL shader source code.</returns>
    public string GenerateCompleteShader()
    {
        var sb = new StringBuilder();

        sb.AppendLine("// ==================================================================");
        sb.AppendLine($"// G-DNN Neural SDF Shader - {_config.Quality} Quality");
        sb.AppendLine($"// Auto-generated by GDNN GPU ShaderGenerator");
        sb.AppendLine($"// Architecture: {_config.InputDimension}D -> {_config.HiddenSize}x{_config.LayerCount} -> {_config.OutputDimension}D");
        sb.AppendLine($"// Features: {_config.Features}");
        sb.AppendLine($"// Shader Model: {_config.ShaderModel}");
        sb.AppendLine("// ==================================================================");
        sb.AppendLine();
        sb.AppendLine("#pragma target 5_0");
        sb.AppendLine("#pragma enable_d3d11_debug_symbols");
        sb.AppendLine();

        // Feature defines
        sb.AppendLine("// Feature defines");
        if (_config.Features.HasFlag(ShaderFeatures.Shadows))
            sb.AppendLine("#define FEATURE_SHADOWS");
        if (_config.Features.HasFlag(ShaderFeatures.Reflections))
            sb.AppendLine("#define FEATURE_REFLECTIONS");
        if (_config.Features.HasFlag(ShaderFeatures.AmbientOcclusion))
            sb.AppendLine("#define FEATURE_AO");
        if (_config.Features.HasFlag(ShaderFeatures.HierarchicalTracing))
            sb.AppendLine("#define FEATURE_HIERARCHICAL");
        if (_config.Features.HasFlag(ShaderFeatures.BinarySearch))
            sb.AppendLine("#define FEATURE_BINARY_SEARCH");
        if (_config.Features.HasFlag(ShaderFeatures.NormalMapping))
            sb.AppendLine("#define FEATURE_NORMAL_MAPPING");
        if (_config.Features.HasFlag(ShaderFeatures.TemporalReprojection))
            sb.AppendLine("#define FEATURE_TEMPORAL");
        if (_config.Features.HasFlag(ShaderFeatures.SubsurfaceScattering))
            sb.AppendLine("#define FEATURE_SSS");
        if (_config.Features.HasFlag(ShaderFeatures.VolumetricFog))
            sb.AppendLine("#define FEATURE_FOG");
        if (_config.Features.HasFlag(ShaderFeatures.AdaptiveStepping))
            sb.AppendLine("#define FEATURE_ADAPTIVE");
        if (_config.Features.HasFlag(ShaderFeatures.TwoSided))
            sb.AppendLine("#define FEATURE_TWO_SIDED");
        sb.AppendLine();

        // Quality defines
        sb.AppendLine("// Quality defines");
        sb.AppendLine($"#define QUALITY_LEVEL {(int)_config.Quality}");
        sb.AppendLine($"#define MAX_STEPS {_config.MaxRayMarchSteps}");
        sb.AppendLine($"#define SURFACE_THRESHOLD {_config.SurfaceThreshold:F6}");
        sb.AppendLine($"#define BINARY_SEARCH_ITERS {_config.BinarySearchIterations}");
        sb.AppendLine($"#define MAX_TRACE_DIST {_config.MaxTraceDistance:F1}");
        sb.AppendLine($"#define RELAXATION {_config.Relaxation:F2}");
        sb.AppendLine($"#define NORMAL_BIAS {_config.NormalBias:F4}");
        sb.AppendLine();

        // Combine all sections
        sb.Append(GenerateVertexShader());
        sb.AppendLine();
        sb.Append(GeneratePixelShader());

        return sb.ToString();
    }

    /// <summary>
    /// Generates all shader variants for the current configuration.
    /// Each variant corresponds to a unique combination of feature flags.
    /// </summary>
    /// <param name="featureCombinations">Specific feature flag combinations to generate. If null, generates common permutations.</param>
    /// <returns>Dictionary mapping variant keys to shader source code.</returns>
    public Dictionary<ulong, string> GenerateVariants(ReadOnlySpan<ShaderFeatures> featureCombinations = default)
    {
        var variants = new Dictionary<ulong, string>();

        ShaderFeatures[] featureSets;
        if (!featureCombinations.IsEmpty && featureCombinations.Length > 0)
        {
            featureSets = featureCombinations.ToArray();
        }
        else
        {
            // Generate common permutations
            featureSets = new ShaderFeatures[]
            {
                ShaderFeatures.None,
                ShaderFeatures.Shadows,
                ShaderFeatures.Shadows | ShaderFeatures.AmbientOcclusion,
                ShaderFeatures.Shadows | ShaderFeatures.AmbientOcclusion | ShaderFeatures.BinarySearch,
                ShaderFeatures.Shadows | ShaderFeatures.AmbientOcclusion | ShaderFeatures.BinarySearch | ShaderFeatures.HierarchicalTracing,
                ShaderFeatures.All
            };
        }

        foreach (var features in featureSets)
        {
            var variantConfig = new ShaderGeneratorConfig
            {
                LayerCount = _config.LayerCount,
                HiddenSize = _config.HiddenSize,
                InputDimension = _config.InputDimension,
                OutputDimension = _config.OutputDimension,
                Activation = _config.Activation,
                OutputActivation = _config.OutputActivation,
                Quality = _config.Quality,
                Features = features,
                Relaxation = _config.Relaxation,
                MaxTraceDistance = _config.MaxTraceDistance,
                NormalBias = _config.NormalBias,
                LeakyReLUAlpha = _config.LeakyReLUAlpha,
                GELUConstant = _config.GELUConstant,
                EmitDebugOutput = _config.EmitDebugOutput,
                UseHalfPrecision = _config.UseHalfPrecision,
                ShaderModel = _config.ShaderModel
            };

            var generator = new ShaderGenerator(variantConfig);
            ulong key = variantConfig.ComputeVariantKey();
            variants[key] = generator.GenerateCompleteShader();
        }

        return variants;
    }

    /// <summary>
    /// Generates a shader variant key for the given feature flags.
    /// </summary>
    /// <param name="features">Feature flags.</param>
    /// <returns>Unique variant key.</returns>
    public ulong GenerateVariantKey(ShaderFeatures features)
    {
        var config = new ShaderGeneratorConfig
        {
            LayerCount = _config.LayerCount,
            HiddenSize = _config.HiddenSize,
            InputDimension = _config.InputDimension,
            OutputDimension = _config.OutputDimension,
            Activation = _config.Activation,
            OutputActivation = _config.OutputActivation,
            Quality = _config.Quality,
            Features = features
        };
        return config.ComputeVariantKey();
    }

    /// <summary>
    /// Generates weight packing function that loads NeuralLayerWeights into float4 constant buffers.
    /// </summary>
    /// <returns>HLSL helper function source code.</returns>
    public string GenerateWeightPackingHelper()
    {
        var sb = new StringBuilder();
        int hidden = _config.HiddenSize;
        int inputDim = _config.InputDimension;

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Weight Packing Helper");
        sb.AppendLine("// Converts raw weight floats into constant buffer float4 layout.");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();

        // Layer 1 packing
        int layer1Total = inputDim * hidden + hidden;
        int layer1Float4 = (layer1Total + 3) / 4;
        sb.AppendLine($"// Pack Layer 1 weights ({inputDim}x{hidden} + {hidden} biases = {layer1Total} floats) into {layer1Float4} float4s");
        sb.AppendLine($"// Layout: first {hidden} float4s hold weight rows (4 weights per float4),");
        sb.AppendLine($"// remaining floats hold biases (one per neuron, packed in last float4s).");
        sb.AppendLine();

        // Hidden layer packing
        int hiddenTotal = hidden * hidden + hidden;
        int hiddenFloat4 = (hiddenTotal + 3) / 4;
        sb.AppendLine($"// Pack Hidden layer weights ({hidden}x{hidden} + {hidden} biases = {hiddenTotal} floats) into {hiddenFloat4} float4s");
        sb.AppendLine();

        // Output layer packing
        int outputDim = _config.OutputDimension;
        int outputTotal = hidden * outputDim + outputDim;
        int outputFloat4 = (outputTotal + 3) / 4;
        sb.AppendLine($"// Pack Output layer weights ({hidden}x{outputDim} + {outputDim} biases = {outputTotal} floats) into {outputFloat4} float4s");
        sb.AppendLine();

        sb.AppendLine("void PackWeights(ReadOnlySpan<float> rawWeights, Span<float4> packedWeights)");
        sb.AppendLine("{");
        sb.AppendLine("    int srcIdx = 0;");
        sb.AppendLine("    int dstIdx = 0;");
        sb.AppendLine();
        sb.AppendLine("    // Pack Layer 1");
        for (int i = 0; i < layer1Float4; i++)
        {
            int baseIdx = i * 4;
            sb.AppendLine($"    packedWeights[{i}] = float4(rawWeights[{baseIdx}], rawWeights[{baseIdx + 1}], rawWeights[{baseIdx + 2}], rawWeights[{baseIdx + 3}]);");
        }
        sb.AppendLine();
        sb.AppendLine("    // Pack Hidden layers");
        for (int layer = 1; layer < _config.LayerCount - 1; layer++)
        {
            int offset = layer1Float4 + (layer - 1) * hiddenFloat4;
            for (int i = 0; i < hiddenFloat4; i++)
            {
                int baseIdx = offset * 4 + i * 4;
                sb.AppendLine($"    packedWeights[{offset + i}] = float4(rawWeights[{baseIdx}], rawWeights[{baseIdx + 1}], rawWeights[{baseIdx + 2}], rawWeights[{baseIdx + 3}]);");
            }
        }
        sb.AppendLine();
        sb.AppendLine("    // Pack Output layer");
        int outputOffset = layer1Float4 + (_config.LayerCount - 2) * hiddenFloat4;
        for (int i = 0; i < outputFloat4; i++)
        {
            int baseIdx = outputOffset * 4 + i * 4;
            sb.AppendLine($"    packedWeights[{outputOffset + i}] = float4(rawWeights[{baseIdx}], rawWeights[{baseIdx + 1}], rawWeights[{baseIdx + 2}], rawWeights[{baseIdx + 3}]);");
        }

        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates reflection/refraction ray functions for multi-bounce rendering.
    /// </summary>
    /// <returns>HLSL function source code.</returns>
    public string GenerateReflectionFunctions()
    {
        if (!_config.Features.HasFlag(ShaderFeatures.Reflections))
            return string.Empty;

        var sb = new StringBuilder();
        int maxBounces = _config.MaxReflectionBounces;

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Reflection and Refraction Ray Functions");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();
        sb.AppendLine("struct BounceResult");
        sb.AppendLine("{");
        sb.AppendLine("    float3 color;");
        sb.AppendLine("    float3 throughput;");
        sb.AppendLine("    int bounces;");
        sb.AppendLine("};");
        sb.AppendLine();
        sb.AppendLine("float3 ComputeReflection(float3 incident, float3 normal)");
        sb.AppendLine("{");
        sb.AppendLine("    return incident - 2.0 * dot(incident, normal) * normal;");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("float3 ComputeRefraction(float3 incident, float3 normal, float etaRatio)");
        sb.AppendLine("{");
        sb.AppendLine("    float cosI = -dot(incident, normal);");
        sb.AppendLine("    float sinT2 = etaRatio * etaRatio * (1.0 - cosI * cosI);");
        sb.AppendLine("    if (sinT2 > 1.0) return float3(0, 0, 0);");
        sb.AppendLine("    float cosT = sqrt(1.0 - sinT2);");
        sb.AppendLine("    return etaRatio * incident + (etaRatio * cosI - cosT) * normal;");
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine($"BounceResult TraceReflections(float3 origin, float3 dir, float3 normal)");
        sb.AppendLine("{");
        sb.AppendLine("    BounceResult result;");
        sb.AppendLine("    result.color = float3(0, 0, 0);");
        sb.AppendLine("    result.throughput = float3(1, 1, 1);");
        sb.AppendLine("    result.bounces = 0;");
        sb.AppendLine();
        sb.AppendLine("    float3 currentDir = dir;");
        sb.AppendLine("    float3 currentOrigin = origin;");
        sb.AppendLine();
        sb.AppendLine($"    [loop]");
        sb.AppendLine($"    for (int bounce = 0; bounce < {maxBounces}; bounce++)");
        sb.AppendLine("    {");
        sb.AppendLine("        float3 reflDir = ComputeReflection(currentDir, normal);");
        sb.AppendLine("        float3 reflOrigin = currentOrigin + normal * 0.01;");
        sb.AppendLine();
        sb.AppendLine("        // Trace reflection ray");
        sb.AppendLine("        float totalDist = 0.0;");
        sb.AppendLine("        bool hit = false;");
        sb.AppendLine($"        for (int s = 0; s < {_config.MaxRayMarchSteps / 2}; s++)");
        sb.AppendLine("        {");
        sb.AppendLine("            float3 pt = reflOrigin + reflDir * totalDist;");
        sb.AppendLine("            float sdf = EvaluateNeuralSDF(pt);");
        sb.AppendLine("            if (abs(sdf) < 0.005) { hit = true; break; }");
        sb.AppendLine("            totalDist += abs(sdf) * 0.8;");
        sb.AppendLine("            if (totalDist > 20.0) break;");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        if (!hit) break;");
        sb.AppendLine();
        sb.AppendLine("        float3 hitPt = reflOrigin + reflDir * totalDist;");
        sb.AppendLine("        float3 hitN = ComputeNormal(hitPt);");
        sb.AppendLine("        float3 hitAlbedo = float3(0.8, 0.8, 0.8);");
        sb.AppendLine("        float fresnel = FresnelSchlick(max(0.0, dot(hitN, -reflDir)), 0.04);");
        sb.AppendLine();
        sb.AppendLine("        result.color += result.throughput * hitAlbedo * fresnel;");
        sb.AppendLine("        result.throughput *= (1.0 - fresnel);");
        sb.AppendLine();
        sb.AppendLine("        currentDir = reflDir;");
        sb.AppendLine("        currentOrigin = hitPt;");
        sb.AppendLine("        normal = hitN;");
        sb.AppendLine("        result.bounces++;");
        sb.AppendLine();
        sb.AppendLine("        if (result.throughput.x < 0.01 && result.throughput.y < 0.01 && result.throughput.z < 0.01)");
        sb.AppendLine("            break;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    return result;");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates volumetric fog integration function.
    /// </summary>
    /// <returns>HLSL function source code.</returns>
    public string GenerateVolumetricFogFunction()
    {
        if (!_config.Features.HasFlag(ShaderFeatures.VolumetricFog))
            return string.Empty;

        var sb = new StringBuilder();
        int sampleCount = _config.Quality switch
        {
            ShaderQualityLevel.Low => 8,
            ShaderQualityLevel.Medium => 16,
            ShaderQualityLevel.High => 32,
            ShaderQualityLevel.Ultra => 64,
            _ => 16
        };

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Volumetric Fog Integration");
        sb.AppendLine("// Exponential height fog with ray marching.");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();
        sb.AppendLine("struct FogParams");
        sb.AppendLine("{");
        sb.AppendLine("    float3 fogColor;");
        sb.AppendLine("    float fogDensity;");
        sb.AppendLine("    float fogHeightFalloff;");
        sb.AppendLine("    float fogHeightBase;");
        sb.AppendLine("    float fogStartDistance;");
        sb.AppendLine("    float fogEndDistance;");
        sb.AppendLine("};");
        sb.AppendLine();
        sb.AppendLine("float IntegrateFog(float3 origin, float3 direction, float maxDist, FogParams params)");
        sb.AppendLine("{");
        sb.AppendLine($"    const int samples = {sampleCount};");
        sb.AppendLine("    float stepSize = maxDist / float(samples);");
        sb.AppendLine("    float transmittance = 1.0;");
        sb.AppendLine();
        sb.AppendLine("    [unroll]");
        sb.AppendLine($"    for (int i = 0; i < {sampleCount}; i++)");
        sb.AppendLine("    {");
        sb.AppendLine("        float t = (float(i) + 0.5) * stepSize;");
        sb.AppendLine("        float3 samplePoint = origin + direction * t;");
        sb.AppendLine();
        sb.AppendLine("        // Height-based density");
        sb.AppendLine("        float heightDiff = samplePoint.y - params.fogHeightBase;");
        sb.AppendLine("        float heightFactor = exp(-heightDiff * params.fogHeightFalloff);");
        sb.AppendLine();
        sb.AppendLine("        // Distance-based density");
        sb.AppendLine("        float distFactor = stepSize * params.fogDensity;");
        sb.AppendLine();
        sb.AppendLine("        // Combine");
        sb.AppendLine("        float density = distFactor * heightFactor;");
        sb.AppendLine("        transmittance *= exp(-density);");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    return transmittance;");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates subsurface scattering approximation function.
    /// </summary>
    /// <returns>HLSL function source code.</returns>
    public string GenerateSSSFunction()
    {
        if (!_config.Features.HasFlag(ShaderFeatures.SubsurfaceScattering))
            return string.Empty;

        var sb = new StringBuilder();

        sb.AppendLine("// ============================================================");
        sb.AppendLine("// Subsurface Scattering Approximation");
        sb.AppendLine("// Pre-integrated skin SSS LUT approach.");
        sb.AppendLine("// ============================================================");
        sb.AppendLine();
        sb.AppendLine("float3 ComputeSSS(float3 normal, float3 lightDir, float3 viewDir, float3 sssColor)");
        sb.AppendLine("{");
        sb.AppendLine("    float nDotL = dot(normal, lightDir);");
        sb.AppendLine("    float nDotV = dot(normal, viewDir);");
        sb.AppendLine();
        sb.AppendLine("    // Wrap lighting for SSS approximation");
        sb.AppendLine("    float wrapDiffuse = max(0.0, (nDotL + 0.5) / 1.5);");
        sb.AppendLine();
        sb.AppendLine("    // View-dependent SSS");
        sb.AppendLine("    float3 H = normalize(lightDir + viewDir);");
        sb.AppendLine("    float vDotH = max(0.0, dot(viewDir, H));");
        sb.AppendLine("    float3 sssTerm = sssColor * wrapDiffuse * pow(vDotH, 3.0);");
        sb.AppendLine();
        sb.AppendLine("    // Thickness estimation");
        sb.AppendLine("    float thickness = saturate(1.0 - nDotV) * saturate(1.0 - nDotL);");
        sb.AppendLine("    sssTerm *= (1.0 - thickness * 0.5);");

        sb.AppendLine();
        sb.AppendLine("    return sssTerm;");
        sb.AppendLine("}");

        return sb.ToString();
    }
}
