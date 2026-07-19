using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
// ============================================================
// FILE: ShaderCompiler.cs
// PATH: GPU/ShaderCompiler.cs
// ============================================================


using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Security.Cryptography;
using System.Text;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks;

namespace GDNN.GPU;

/// <summary>
/// Shader model versions supported by the compilation pipeline.
/// </summary>
public enum ShaderModelVersion
{
    /// <summary>Shader Model 5.0 (DirectX 11).</summary>
    SM_5_0,

    /// <summary>Shader Model 5.1 (DirectX 11.3).</summary>
    SM_5_1,

    /// <summary>Shader Model 6.0 (DirectX 12).</summary>
    SM_6_0,

    /// <summary>Shader Model 6.1.</summary>
    SM_6_1,

    /// <summary>Shader Model 6.2.</summary>
    SM_6_2,

    /// <summary>Shader Model 6.3.</summary>
    SM_6_3,

    /// <summary>Shader Model 6.4.</summary>
    SM_6_4,

    /// <summary>Shader Model 6.5 (DirectX 12 Ultimate).</summary>
    SM_6_5,

    /// <summary>Shader Model 6.6.</summary>
    SM_6_6,

    /// <summary>Shader Model 6.7.</summary>
    SM_6_7,

    /// <summary>Vulkan SPIR-V target.</summary>
    SPIRV,

    /// <summary>Metal Shading Language target.</summary>
    Metal
}

/// <summary>
/// Shader compilation target platform.
/// </summary>
public enum ShaderTargetPlatform
{
    /// <summary>DirectX 11/12 with D3D compiler.</summary>
    Direct3D,

    /// <summary>Vulkan with SPIR-V.</summary>
    Vulkan,

    /// <summary>Metal for Apple platforms.</summary>
    Metal,

    /// <summary>OpenGL / GLSL.</summary>
    OpenGL,

    /// <summary>Cross-platform SPIR-V.</summary>
    SPIRV_Cross
}

/// <summary>
/// Severity levels for shader compilation diagnostics.
/// </summary>
public enum ShaderDiagnosticSeverity
{
    /// <summary>Informational message.</summary>
    Info,

    /// <summary>Warning that may affect performance or correctness.</summary>
    Warning,

    /// <summary>Error that prevents compilation.</summary>
    Error
}

/// <summary>
/// A single diagnostic message from shader compilation.
/// </summary>
public sealed class ShaderDiagnostic
{
    /// <summary>Severity level.</summary>
    public ShaderDiagnosticSeverity Severity { get; init; }

    /// <summary>Error or warning code.</summary>
    public string? Code { get; init; }

    /// <summary>Human-readable message.</summary>
    public string Message { get; init; }

    /// <summary>Source file name (if applicable).</summary>
    public string? FileName { get; init; }

    /// <summary>Line number in source (1-based, 0 if unknown).</summary>
    public int Line { get; init; }

    /// <summary>Column number in source (1-based, 0 if unknown).</summary>
    public int Column { get; init; }

    /// <summary>The preprocessed source line (if available).</summary>
    public string? SourceLine { get; init; }

    /// <summary>Returns a formatted diagnostic string.</summary>
    public override string ToString()
    {
        string location = !string.IsNullOrEmpty(FileName) ? $"{FileName}" : "<input>";
        if (Line > 0)
            location += $"({Line},{Column})";
        return $"[{Severity}] {location}: {Code ?? ""} {Message}";
    }
}

/// <summary>
/// Result of a shader compilation operation.
/// </summary>
public sealed class ShaderCompilationResult
{
    /// <summary>Whether compilation succeeded.</summary>
    public bool Success { get; set; }

    /// <summary>Compiled bytecode (HLSL bytecode, SPIR-V, etc.).</summary>
    public byte[]? Bytecode { get; set; }

    /// <summary>Assembly listing (if requested).</summary>
    public string? AssemblyListing { get; set; }

    /// <summary>Preprocessed source code.</summary>
    public string? PreprocessedSource { get; set; }

    /// <summary>Diagnostics from compilation.</summary>
    public List<ShaderDiagnostic> Diagnostics { get; init; } = new();

    /// <summary>Shader variant key that was compiled.</summary>
    public ulong VariantKey { get; init; }

    /// <summary>Shader entry point name.</summary>
    public string? EntryPoint { get; init; }

    /// <summary>Target shader model.</summary>
    public ShaderModelVersion TargetModel { get; init; }

    /// <summary>Target platform.</summary>
    public ShaderTargetPlatform TargetPlatform { get; init; }

    /// <summary>Compilation time in milliseconds.</summary>
    public double CompilationTimeMs { get; set; }

    /// <summary>Hash of the preprocessed source (for caching).</summary>
    public string? SourceHash { get; set; }

    /// <summary>Number of warnings.</summary>
    public int WarningCount => Diagnostics.Count(d => d.Severity == ShaderDiagnosticSeverity.Warning);

    /// <summary>Number of errors.</summary>
    public int ErrorCount => Diagnostics.Count(d => d.Severity == ShaderDiagnosticSeverity.Error);

    /// <summary>Gets all errors as a combined string.</summary>
    public string GetErrorSummary()
    {
        var errors = Diagnostics.Where(d => d.Severity == ShaderDiagnosticSeverity.Error).ToList();
        if (errors.Count == 0)
            return "No errors.";
        return string.Join("\n", errors.Select(e => e.ToString()));
    }
}

/// <summary>
/// Configuration for the shader compiler.
/// </summary>
public sealed class ShaderCompilerConfig
{
    /// <summary>Target shader model.</summary>
    public ShaderModelVersion ShaderModel { get; set; } = ShaderModelVersion.SM_5_0;

    /// <summary>Target platform.</summary>
    public ShaderTargetPlatform Platform { get; set; } = ShaderTargetPlatform.Direct3D;

    /// <summary>Enable debug information in compiled shaders.</summary>
    public bool EnableDebugInfo { get; set; } = true;

    /// <summary>Enable optimization passes.</summary>
    public bool EnableOptimization { get; set; } = true;

    /// <summary>Optimization level (0-3).</summary>
    public int OptimizationLevel { get; set; } = 2;

    /// <summary>Maximum number of instruction slots (0 = unlimited).</summary>
    public int MaxInstructionSlots { get; set; } = 0;

    /// <summary>Maximum texture sample count (0 = unlimited).</summary>
    public int MaxTextureSamples { get; set; } = 0;

    /// <summary>Enable packing validation.</summary>
    public bool ValidatePacking { get; set; } = true;

    /// <summary>Treat warnings as errors.</summary>
    public bool WarningsAsErrors { get; set; } = false;

    /// <summary>Enable preprocessor output for debugging.</summary>
    public bool DumpPreprocessed { get; set; } = false;

    /// <summary>Additional preprocessor defines.</summary>
    public Dictionary<string, string?> Defines { get; init; } = new();

    /// <summary>Include search paths.</summary>
    public List<string> IncludePaths { get; init; } = new();

    /// <summary>Enable shader caching.</summary>
    public bool EnableCaching { get; set; } = true;

    /// <summary>Maximum cache size in MB.</summary>
    public int MaxCacheSizeMB { get; set; } = 256;

    /// <summary>Enables strict floating-point compliance.</summary>
    public bool StrictFloatCompliance { get; set; } = false;

    /// <summary>Enable denormalized number handling.</summary>
    public bool FlushDenormals { get; set; } = true;

    /// <summary>Enable partial precision (min16float) where supported.</summary>
    public bool EnablePartialPrecision { get; set; } = false;

    /// <summary>Creates a default config for DirectX 11.</summary>
    public static ShaderCompilerConfig ForDirectX11() => new()
    {
        ShaderModel = ShaderModelVersion.SM_5_0,
        Platform = ShaderTargetPlatform.Direct3D,
        EnableDebugInfo = true,
        EnableOptimization = true
    };

    /// <summary>Creates a default config for DirectX 12.</summary>
    public static ShaderCompilerConfig ForDirectX12() => new()
    {
        ShaderModel = ShaderModelVersion.SM_6_5,
        Platform = ShaderTargetPlatform.Direct3D,
        EnableDebugInfo = true,
        EnableOptimization = true
    };

    /// <summary>Creates a default config for Vulkan.</summary>
    public static ShaderCompilerConfig ForVulkan() => new()
    {
        ShaderModel = ShaderModelVersion.SPIRV,
        Platform = ShaderTargetPlatform.Vulkan,
        EnableDebugInfo = true,
        EnableOptimization = true
    };
}

/// <summary>
/// Shader compilation pipeline with preprocessor handling, validation,
/// error reporting, and compiled shader caching.
/// </summary>
public sealed class ShaderCompiler : IDisposable
{
    private readonly ShaderCompilerConfig _config;
    private readonly ConcurrentDictionary<string, ShaderCompilationResult> _cache;
    private readonly object _compilationLock;
    private bool _disposed;

    /// <summary>Creates a new shader compiler.</summary>
    /// <param name="config">Compiler configuration.</param>
    public ShaderCompiler(ShaderCompilerConfig? config = null)
    {
        _config = config ?? new ShaderCompilerConfig();
        _cache = new ConcurrentDictionary<string, ShaderCompilationResult>();
        _compilationLock = new object();
    }

    /// <summary>Gets the current configuration.</summary>
    public ShaderCompilerConfig Config => _config;

    /// <summary>Gets the number of cached shaders.</summary>
    public int CacheCount => _cache.Count;

    /// <summary>
    /// Compiles a shader source string.
    /// </summary>
    /// <param name="source">HLSL source code.</param>
    /// <param name="entryPoint">Entry point function name.</param>
    /// <param name="shaderType">Shader type (pixel, vertex, compute).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Compilation result.</returns>
    public async Task<ShaderCompilationResult> CompileAsync(
        string source,
        string entryPoint,
        ShaderType shaderType = ShaderType.PixelShader,
        CancellationToken cancellationToken = default)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        // Check cache
        string cacheKey = ComputeCacheKey(source, entryPoint, shaderType);
        if (_config.EnableCaching && _cache.TryGetValue(cacheKey, out var cached))
        {
            return cached;
        }

        var result = new ShaderCompilationResult
        {
            EntryPoint = entryPoint,
            TargetModel = _config.ShaderModel,
            TargetPlatform = _config.Platform
        };

        try
        {
            // Phase 1: Preprocessing
            string preprocessed = PreprocessSource(source, result);

            if (_config.DumpPreprocessed)
            {
                result.PreprocessedSource = preprocessed;
            }

            // Phase 2: Validation
            bool valid = ValidateSource(preprocessed, result);

            if (!valid && result.Diagnostics.Any(d => d.Severity == ShaderDiagnosticSeverity.Error))
            {
                result.Success = false;
                stopwatch.Stop();
                result.CompilationTimeMs = stopwatch.Elapsed.TotalMilliseconds;
                return result;
            }

            // Phase 3: Compilation (simulated - in production, call DxcCompiler or similar)
            await SimulateCompilationAsync(preprocessed, entryPoint, shaderType, result, cancellationToken);

            // Phase 4: Post-compilation validation
            if (result.Success && _config.ValidatePacking)
            {
                ValidateConstantBufferPacking(preprocessed, result);
            }

            // Cache result
            if (_config.EnableCaching && result.Success)
            {
                _cache[cacheKey] = result;
            }
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            result.Diagnostics.Add(new ShaderDiagnostic
            {
                Severity = ShaderDiagnosticSeverity.Error,
                Code = "GDNN_COMPILE_EXCEPTION",
                Message = $"Compilation failed: {ex.Message}"
            });
            result.Success = false;
        }

        stopwatch.Stop();
        result.CompilationTimeMs = stopwatch.Elapsed.TotalMilliseconds;
        return result;
    }

    /// <summary>
    /// Compiles a shader synchronously.
    /// </summary>
    public ShaderCompilationResult Compile(string source, string entryPoint, ShaderType shaderType = ShaderType.PixelShader)
    {
        return CompileAsync(source, entryPoint, shaderType).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Compiles multiple shader variants in parallel.
    /// </summary>
    /// <param name="variants">Shader variants to compile (source, entry point, type).</param>
    /// <param name="maxParallelism">Maximum parallel compilations.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Compilation results.</returns>
    public async Task<ShaderCompilationResult[]> CompileBatchAsync(
        ReadOnlyMemory<(string Source, string EntryPoint, ShaderType Type)> variants,
        int maxParallelism = 4,
        CancellationToken cancellationToken = default)
    {
        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxParallelism,
            CancellationToken = cancellationToken
        };

        var results = new ShaderCompilationResult[variants.Length];

        await Parallel.ForAsync(0, variants.Length, options, async (i, ct) =>
        {
            var (source, entryPoint, type) = variants.Span[i];
            results[i] = await CompileAsync(source, entryPoint, type, ct);
        });

        return results;
    }

    /// <summary>
    /// Preprocesses HLSL source code, handling #define, #ifdef, #include, etc.
    /// </summary>
    /// <param name="source">Raw HLSL source.</param>
    /// <param name="result">Compilation result to add diagnostics to.</param>
    /// <returns>Preprocessed source.</returns>
    public string PreprocessSource(string source, ShaderCompilationResult? result = null)
    {
        var sb = new StringBuilder(source);
        var defines = new Dictionary<string, string?>(_config.Defines);

        // Add built-in defines based on shader model
        defines["GDNN_SM_5_0"] = _config.ShaderModel >= ShaderModelVersion.SM_5_0 ? "1" : null;
        defines["GDNN_SM_5_1"] = _config.ShaderModel >= ShaderModelVersion.SM_5_1 ? "1" : null;
        defines["GDNN_SM_6_0"] = _config.ShaderModel >= ShaderModelVersion.SM_6_0 ? "1" : null;
        defines["GDNN_SM_6_5"] = _config.ShaderModel >= ShaderModelVersion.SM_6_5 ? "1" : null;
        defines["GDNN_DEBUG"] = _config.EnableDebugInfo ? "1" : null;
        defines["GDNN_OPTIMIZED"] = _config.EnableOptimization ? "1" : null;

        // Platform defines
        defines["GDNN_PLATFORM_D3D"] = _config.Platform == ShaderTargetPlatform.Direct3D ? "1" : null;
        defines["GDNN_PLATFORM_VULKAN"] = _config.Platform == ShaderTargetPlatform.Vulkan ? "1" : null;
        defines["GDNN_PLATFORM_METAL"] = _config.Platform == ShaderTargetPlatform.Metal ? "1" : null;

        // Process #define directives
        ProcessDefines(sb, defines, result);

        // Process #ifdef / #ifndef / #if directives
        ProcessConditionals(sb, defines, result);

        // Process #include directives
        ProcessIncludes(sb, result);

        // Remove comments
        RemoveComments(sb);

        // Process #pragma directives
        ProcessPragmas(sb, result);

        // Remove remaining preprocessor directives
        RemovePreprocessorDirectives(sb);

        // Normalize whitespace
        NormalizeWhitespace(sb);

        return sb.ToString();
    }

    /// <summary>
    /// Processes #define directives and expands macro definitions.
    /// </summary>
    private void ProcessDefines(StringBuilder sb, Dictionary<string, string?> defines, ShaderCompilationResult? result)
    {
        string source = sb.ToString();
        var defineRegex = new Regex(@"#define\s+(\w+)(?:\s+(.+))?$", RegexOptions.Multiline);
        var usages = new Dictionary<string, string?>();

        // First pass: collect all #define directives
        var definesToProcess = new List<(string Name, string? Value)>();
        foreach (Match match in defineRegex.Matches(source))
        {
            string name = match.Groups[1].Value;
            string? value = match.Groups[2].Success ? match.Groups[2].Value.Trim() : null;
            definesToProcess.Add((name, value));
            defines[name] = value;
        }

        // Remove #define lines
        source = defineRegex.Replace(source, "");

        // Second pass: expand macro usages
        foreach (var kvp in defines)
        {
            if (kvp.Value != null)
            {
                var usageRegex = new Regex($@"\b{Regex.Escape(kvp.Key)}\b");
                source = usageRegex.Replace(source, kvp.Value);
            }
        }

        sb.Clear();
        sb.Append(source);
    }

    /// <summary>
    /// Processes #ifdef, #ifndef, #if, #elif, #else, #endif conditionals.
    /// </summary>
    private void ProcessConditionals(StringBuilder sb, Dictionary<string, string?> defines, ShaderCompilationResult? result)
    {
        string source = sb.ToString();
        var lines = source.Split('\n');
        var output = new List<string>();
        var stack = new Stack<bool>();
        stack.Push(true); // outermost scope is always active

        foreach (string line in lines)
        {
            string trimmed = line.Trim();

            if (trimmed.StartsWith("#ifdef "))
            {
                string symbol = trimmed.Substring(7).Trim();
                bool active = stack.Peek() && defines.ContainsKey(symbol);
                stack.Push(active);
                continue;
            }
            else if (trimmed.StartsWith("#ifndef "))
            {
                string symbol = trimmed.Substring(8).Trim();
                bool active = stack.Peek() && !defines.ContainsKey(symbol);
                stack.Push(active);
                continue;
            }
            else if (trimmed.StartsWith("#if "))
            {
                string condition = trimmed.Substring(4).Trim();
                bool active = stack.Peek() && EvaluatePreprocessorCondition(condition, defines);
                stack.Push(active);
                continue;
            }
            else if (trimmed.StartsWith("#elif "))
            {
                if (stack.Count > 1)
                {
                    bool parentActive = stack.Count > 1 ? stack.ElementAt(1) : true;
                    string condition = trimmed.Substring(6).Trim();
                    bool active = parentActive && !stack.Peek() && EvaluatePreprocessorCondition(condition, defines);
                    stack.Pop();
                    stack.Push(active);
                }
                continue;
            }
            else if (trimmed == "#else")
            {
                if (stack.Count > 1)
                {
                    bool parentActive = stack.Count > 1 ? stack.ElementAt(1) : true;
                    stack.Pop();
                    stack.Push(parentActive && !stack.Peek());
                }
                continue;
            }
            else if (trimmed == "#endif")
            {
                if (stack.Count > 1)
                    stack.Pop();
                continue;
            }

            if (stack.Peek())
            {
                output.Add(line);
            }
        }

        sb.Clear();
        sb.Append(string.Join("\n", output));
    }

    /// <summary>
    /// Evaluates a preprocessor condition expression.
    /// </summary>
    private bool EvaluatePreprocessorCondition(string condition, Dictionary<string, string?> defines)
    {
        condition = condition.Trim();

        // Handle defined(X) or defined X
        var definedRegex = new Regex(@"defined\s*\(?\s*(\w+)\s*\)?");
        condition = definedRegex.Replace(condition, match =>
        {
            string symbol = match.Groups[1].Value;
            return defines.ContainsKey(symbol) ? "1" : "0";
        });

        // Handle simple symbol references
        foreach (var kvp in defines)
        {
            var symbolRegex = new Regex($@"\b{Regex.Escape(kvp.Key)}\b");
            condition = symbolRegex.Replace(condition, kvp.Value != null ? "1" : "0");
        }

        // Handle && and || operators
        condition = condition.Replace("&&", "&&");
        condition = condition.Replace("||", "||");

        // Simple evaluation: if contains 0 after all substitutions, condition is false
        try
        {
            // Very simple evaluator for common preprocessor conditions
            if (condition.Contains("||"))
            {
                var parts = condition.Split("||");
                foreach (var part in parts)
                {
                    if (EvaluateSimpleCondition(part.Trim()))
                        return true;
                }
                return false;
            }
            else if (condition.Contains("&&"))
            {
                var parts = condition.Split("&&");
                foreach (var part in parts)
                {
                    if (!EvaluateSimpleCondition(part.Trim()))
                        return false;
                }
                return true;
            }
            else
            {
                return EvaluateSimpleCondition(condition);
            }
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Evaluates a simple preprocessor condition (single term).
    /// </summary>
    private bool EvaluateSimpleCondition(string condition)
    {
        condition = condition.Trim();

        if (condition == "1")
            return true;
        if (condition == "0")
            return false;

        // Handle ! prefix
        if (condition.StartsWith('!'))
            return !EvaluateSimpleCondition(condition.Substring(1));

        // Handle parenthesized expressions
        if (condition.StartsWith('(') && condition.EndsWith(')'))
            return EvaluateSimpleCondition(condition.Substring(1, condition.Length - 2));

        // If it's a number, evaluate as integer
        if (int.TryParse(condition, out int value))
            return value != 0;

        // If it's still a symbol, check if it's defined (should have been substituted)
        return false;
    }

    /// <summary>
    /// Processes #include directives.
    /// </summary>
    private void ProcessIncludes(StringBuilder sb, ShaderCompilationResult? result)
    {
        var includeRegex = new Regex(@"#include\s+[<""]([^>""]+)[>""]", RegexOptions.Multiline);
        string source = sb.ToString();

        var matches = includeRegex.Matches(source);
        foreach (Match match in matches)
        {
            string includePath = match.Groups[1].Value;

            // Check configured include paths
            string? resolvedPath = null;
            foreach (string searchPath in _config.IncludePaths)
            {
                string fullPath = Path.Combine(searchPath, includePath);
                if (File.Exists(fullPath))
                {
                    resolvedPath = fullPath;
                    break;
                }
            }

            if (resolvedPath != null)
            {
                try
                {
                    string includeContent = File.ReadAllText(resolvedPath);
                    source = source.Replace(match.Value, $"// -- begin include: {includePath} --\n{includeContent}\n// -- end include: {includePath} --");
                }
                catch (Exception ex)
                {
                    result?.Diagnostics.Add(new ShaderDiagnostic
                    {
                        Severity = ShaderDiagnosticSeverity.Warning,
                        Code = "GDNN_INCLUDE_FAILED",
                        Message = $"Failed to include '{includePath}': {ex.Message}"
                    });
                    source = source.Replace(match.Value, $"// -- include failed: {includePath} --");
                }
            }
            else
            {
                // Check if it's a standard library include
                if (IsStandardHLSLInclude(includePath))
                {
                    source = source.Replace(match.Value, $"// -- standard include: {includePath} --");
                }
                else
                {
                    result?.Diagnostics.Add(new ShaderDiagnostic
                    {
                        Severity = ShaderDiagnosticSeverity.Warning,
                        Code = "GDNN_INCLUDE_NOT_FOUND",
                        Message = $"Include file not found: '{includePath}'"
                    });
                    source = source.Replace(match.Value, $"// -- include not found: {includePath} --");
                }
            }
        }

        sb.Clear();
        sb.Append(source);
    }

    /// <summary>
    /// Checks if an include path is a standard HLSL/graphics library.
    /// </summary>
    private static bool IsStandardHLSLInclude(string path)
    {
        string[] standardHeaders = new[]
        {
            "d3d11.hlsli", "d3d12.hlsli", "windows.hlsli",
            "packingutils.hlsli", "semantics.hlsli",
            "HLSLExtensions.hlsl", "HlslDefines.hlsli"
        };

        return standardHeaders.Any(h => path.EndsWith(h, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Removes C-style and C++-style comments from the source.
    /// </summary>
    private void RemoveComments(StringBuilder sb)
    {
        string source = sb.ToString();

        // Remove multi-line comments
        source = Regex.Replace(source, @"/\*[\s\S]*?\*/", "", RegexOptions.Multiline);

        // Remove single-line comments
        source = Regex.Replace(source, @"//[^\n]*", "", RegexOptions.Multiline);

        sb.Clear();
        sb.Append(source);
    }

    /// <summary>
    /// Processes #pragma directives and emits them as-is (they are kept for the compiler).
    /// </summary>
    private void ProcessPragmas(StringBuilder sb, ShaderCompilationResult? result)
    {
        // Pragmas are left intact for the actual shader compiler
        // We just validate them
        string source = sb.ToString();
        var pragmaRegex = new Regex(@"#pragma\s+(.+)$", RegexOptions.Multiline);

        foreach (Match match in pragmaRegex.Matches(source))
        {
            string pragma = match.Groups[1].Value.Trim();

            if (pragma.StartsWith("target "))
            {
                // Validate target pragma matches config
                string targetModel = pragma.Substring(7).Trim();
                string expectedModel = _config.ShaderModel switch
                {
                    ShaderModelVersion.SM_5_0 => "5_0",
                    ShaderModelVersion.SM_5_1 => "5_1",
                    ShaderModelVersion.SM_6_0 => "6_0",
                    ShaderModelVersion.SM_6_1 => "6_1",
                    ShaderModelVersion.SM_6_2 => "6_2",
                    ShaderModelVersion.SM_6_3 => "6_3",
                    ShaderModelVersion.SM_6_4 => "6_4",
                    ShaderModelVersion.SM_6_5 => "6_5",
                    ShaderModelVersion.SM_6_6 => "6_6",
                    ShaderModelVersion.SM_6_7 => "6_7",
                    _ => "5_0"
                };

                if (targetModel != expectedModel && _config.ShaderModel != ShaderModelVersion.SPIRV)
                {
                    result?.Diagnostics.Add(new ShaderDiagnostic
                    {
                        Severity = ShaderDiagnosticSeverity.Info,
                        Code = "GDNN_PRAGMA_TARGET",
                        Message = $"Shader model pragma '{targetModel}' will be used. Config targets SM {expectedModel}."
                    });
                }
            }
        }
    }

    /// <summary>
    /// Removes preprocessor directives that were not handled.
    /// </summary>
    private void RemovePreprocessorDirectives(StringBuilder sb)
    {
        string source = sb.ToString();

        // Remove lines starting with # that aren't important
        source = Regex.Replace(source, @"^#\s*(?!pragma|include|define|ifdef|ifndef|if|elif|else|endif)[^\n]*$", "",
            RegexOptions.Multiline | RegexOptions.Compiled);

        sb.Clear();
        sb.Append(source);
    }

    /// <summary>
    /// Normalizes whitespace in the source.
    /// </summary>
    private void NormalizeWhitespace(StringBuilder sb)
    {
        string source = sb.ToString();

        // Replace multiple blank lines with a single blank line
        source = Regex.Replace(source, @"\n\s*\n\s*\n", "\n\n");

        // Remove trailing whitespace on lines
        source = Regex.Replace(source, @"[ \t]+$", "", RegexOptions.Multiline);

        // Ensure file ends with a newline
        source = source.TrimEnd('\n', '\r') + "\n";

        sb.Clear();
        sb.Append(source);
    }

    /// <summary>
    /// Validates the preprocessed source for common issues.
    /// </summary>
    private bool ValidateSource(string source, ShaderCompilationResult result)
    {
        bool valid = true;

        // Check for undefined function calls
        var functionCallRegex = new Regex(@"\b([a-zA-Z_]\w*)\s*\(", RegexOptions.Multiline);
        var definedFunctions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            // HLSL built-in functions
            "abs", "acos", "all", "any", "asin", "atan", "atan2",
            "ceil", "clamp", "clip", "cos", "cosh", "cross",
            "ddx", "ddx_coarse", "ddx_fine", "ddy", "ddy_coarse", "ddy_fine",
            "degrees", "determinant", "distance", "dot",
            "exp", "exp2",
            "floor", "fmod", "frac",
            "frexp", "fwidth",
            "isfinite", "isinf", "isnan",
            "ldexp", "length", "lerp", "lit", "log", "log10", "log2",
            "max", "min", "modf", "mul",
            "noise", "normalize", "pow",
            "radians", "rcp", "reflect", "refract", "round", "rsqrt",
            "saturate", "sign", "sin", "sincos", "sinh", "smoothstep",
            "sqrt", "step",
            "tan", "tanh", "tex1D", "tex2D", "tex3D", "texCUBE",
            "tex2Dgrad", "tex2Dlod", "tex2Dproj",
            "transpose",
            "trunc",
            // G-DNN-specific
            "EvaluateNeuralSDF", "ComputeNormal", "EvaluateShadow",
            "EvaluateAO", "FresnelSchlick", "ActivationFunc", "ActivationBatch",
            "GetHierarchicalLowerBound", "IntegrateFog", "ComputeSSS",
            "ComputeReflection", "ComputeRefraction", "TraceReflections",
            // Common utilities
            "float2", "float3", "float4", "int2", "int3", "int4",
            "uint2", "uint3", "uint4", "bool2", "bool3", "bool4",
            "float2x2", "float3x3", "float4x4",
            "asfloat", "asint", "asuint",
            "saturate", "clamp", "step", "smoothstep",
            "PSMain", "VSMain", "CSMain"
        };

        // Collect user-defined functions
        var funcDefRegex = new Regex(@"\b(?:void|float|int|uint|bool|half|float[234]|int[234]|uint[234])\s+([a-zA-Z_]\w*)\s*\(");
        foreach (Match match in funcDefRegex.Matches(source))
        {
            definedFunctions.Add(match.Groups[1].Value);
        }

        // Check for common issues
        if (!source.Contains("SV_TARGET") && !source.Contains("SV_Position") && !source.Contains("SV_Depth"))
        {
            result.Diagnostics.Add(new ShaderDiagnostic
            {
                Severity = ShaderDiagnosticSeverity.Warning,
                Code = "GDNN_MISSING_SEMANTIC",
                Message = "No SV_TARGET or SV_Position semantics found. Ensure output semantics are defined."
            });
        }

        // Check for potential division by zero
        var divByZeroRegex = new Regex(@"/\s*0\.0?\b");
        if (divByZeroRegex.IsMatch(source))
        {
            result.Diagnostics.Add(new ShaderDiagnostic
            {
                Severity = ShaderDiagnosticSeverity.Warning,
                Code = "GDNN_DIVISION_BY_ZERO",
                Message = "Potential division by zero detected."
            });
        }

        // Check for texture sampling without sampler
        if (Regex.IsMatch(source, @"\bSample\s*\(") && !Regex.IsMatch(source, @"SamplerState\s+\w+"))
        {
            result.Diagnostics.Add(new ShaderDiagnostic
            {
                Severity = ShaderDiagnosticSeverity.Warning,
                Code = "GDNN_MISSING_SAMPLER",
                Message = "Texture sampling found but no SamplerState declaration detected."
            });
        }

        // Check instruction count estimate
        if (_config.MaxInstructionSlots > 0)
        {
            int estimatedInstructions = EstimateInstructionCount(source);
            if (estimatedInstructions > _config.MaxInstructionSlots)
            {
                result.Diagnostics.Add(new ShaderDiagnostic
                {
                    Severity = ShaderDiagnosticSeverity.Warning,
                    Code = "GDNN_HIGH_INSTRUCTION_COUNT",
                    Message = $"Estimated instruction count ({estimatedInstructions}) exceeds limit ({_config.MaxInstructionSlots})."
                });
            }
        }

        return valid;
    }

    /// <summary>
    /// Estimates the instruction count of the shader source.
    /// </summary>
    private int EstimateInstructionCount(string source)
    {
        int count = 0;

        // Count arithmetic operations
        count += Regex.Matches(source, @"[\+\-\*/]").Count;

        // Count function calls
        count += Regex.Matches(source, @"\w+\s*\(").Count;

        // Count loops
        count += Regex.Matches(source, @"\bfor\s*\(").Count * 8;
        count += Regex.Matches(source, @"\bwhile\s*\(").Count * 8;

        // Count texture samples (expensive)
        count += Regex.Matches(source, @"\bSample\s*\(").Count * 16;
        count += Regex.Matches(source, @"\bSampleLevel\s*\(").Count * 12;

        return count;
    }

    /// <summary>
    /// Validates constant buffer packing rules.
    /// </summary>
    private void ValidateConstantBufferPacking(string source, ShaderCompilationResult result)
    {
        // Check for constant buffer declarations
        var cbufferRegex = new Regex(@"cbuffer\s+(\w+)\s*:\s*register\((\w+)\)\s*\{([^}]+)\}", RegexOptions.Singleline);

        foreach (Match match in cbufferRegex.Matches(source))
        {
            string bufferName = match.Groups[1].Value;
            string body = match.Groups[3].Value;

            // Parse fields
            var fieldRegex = new Regex(@"(float[234]|float|int[234]|int|uint[234]|uint|bool|float4x4)\s+(\w+)(?:\s*\[(\d+)\])?\s*;");
            var fields = new List<(string Type, string Name, int ArraySize)>();

            foreach (Match fieldMatch in fieldRegex.Matches(body))
            {
                string type = fieldMatch.Groups[1].Value;
                string name = fieldMatch.Groups[2].Value;
                int arraySize = fieldMatch.Groups[3].Success ? int.Parse(fieldMatch.Groups[3].Value) : 1;
                fields.Add((type, name, arraySize));
            }

            // Check packing rules
            int currentOffset = 0;
            foreach (var (type, name, arraySize) in fields)
            {
                int fieldSize = type switch
                {
                    "float" => 4,
                    "float2" => 8,
                    "float3" => 12,
                    "float4" => 16,
                    "int" => 4,
                    "int2" => 8,
                    "int3" => 12,
                    "int4" => 16,
                    "uint" => 4,
                    "uint2" => 8,
                    "uint3" => 12,
                    "uint4" => 16,
                    "float4x4" => 64,
                    "bool" => 4,
                    _ => 4
                };

                int alignment = fieldSize >= 16 ? 16 : fieldSize >= 8 ? 8 : 4;
                int padding = (alignment - (currentOffset % alignment)) % alignment;

                if (padding > 0)
                {
                    result.Diagnostics.Add(new ShaderDiagnostic
                    {
                        Severity = ShaderDiagnosticSeverity.Info,
                        Code = "GDNN_CB_PADDING",
                        Message = $"Constant buffer '{bufferName}': {padding} bytes padding before field '{name}' (offset {currentOffset + padding})."
                    });
                }

                currentOffset += padding + fieldSize * arraySize;
            }

            // Check total size
            if (currentOffset > 4096)
            {
                result.Diagnostics.Add(new ShaderDiagnostic
                {
                    Severity = ShaderDiagnosticSeverity.Error,
                    Code = "GDNN_CB_TOO_LARGE",
                    Message = $"Constant buffer '{bufferName}' exceeds maximum size of 4096 bytes (actual: {currentOffset})."
                });
            }
        }
    }

    /// <summary>
    /// Simulates shader compilation (in production, this would call DxcCompiler or fxc).
    /// </summary>
    private Task SimulateCompilationAsync(string source, string entryPoint, ShaderType shaderType,
        ShaderCompilationResult result, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // Compute source hash for cache key
        byte[] sourceBytes = Encoding.UTF8.GetBytes(source);
        string sourceHash = Convert.ToHexString(MD5.HashData(sourceBytes));
        result.SourceHash = sourceHash;

        // Validate entry point exists
        string entryPattern = $@"\b{Regex.Escape(entryPoint)}\s*\(";
        if (!Regex.IsMatch(source, entryPattern))
        {
            result.Diagnostics.Add(new ShaderDiagnostic
            {
                Severity = ShaderDiagnosticSeverity.Error,
                Code = "GDNN_ENTRY_POINT_NOT_FOUND",
                Message = $"Entry point '{entryPoint}' not found in shader source."
            });
            result.Success = false;
            return Task.CompletedTask;
        }

        // Simulate bytecode generation
        // In production: call dxcCompiler.Compile() or fxcCompiler.Compile()
        byte[] simulatedBytecode = GenerateSimulatedBytecode(source, entryPoint, shaderType);
        result.Bytecode = simulatedBytecode;

        // Generate assembly listing
        if (_config.DumpPreprocessed || _config.EnableDebugInfo)
        {
            result.AssemblyListing = GenerateAssemblyListing(source, entryPoint, shaderType);
        }

        result.Success = true;

        return Task.CompletedTask;
    }

    /// <summary>
    /// Generates simulated bytecode for testing purposes.
    /// </summary>
    private byte[] GenerateSimulatedBytecode(string source, string entryPoint, ShaderType shaderType)
    {
        // DXBC/DXIL header simulation
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Magic number (DXBC)
        writer.Write((uint)0x43425844); // 'DXBC'

        // Version (simulated)
        uint version = _config.ShaderModel switch
        {
            ShaderModelVersion.SM_5_0 => 0x00050000,
            ShaderModelVersion.SM_5_1 => 0x00050100,
            ShaderModelVersion.SM_6_0 => 0x00060000,
            ShaderModelVersion.SM_6_5 => 0x00060500,
            _ => 0x00050000
        };
        writer.Write(version);

        // Shader type
        writer.Write((uint)shaderType);

        // Byte count placeholder
        int bytecodeOffset = (int)ms.Position;
        writer.Write(0); // placeholder

        // Write source hash
        writer.Write(Encoding.UTF8.GetBytes(entryPoint.PadRight(256, '\0'), 0, 256));

        // Instruction count (estimated)
        int instructionCount = EstimateInstructionCount(source);
        writer.Write(instructionCount);

        // Pad to alignment
        while (ms.Position % 16 != 0)
            writer.Write((byte)0);

        // Update bytecode size
        long endPos = ms.Position;
        ms.Seek(bytecodeOffset, SeekOrigin.Begin);
        writer.Write((int)(endPos - bytecodeOffset - 4));
        ms.Seek(endPos, SeekOrigin.Begin);

        return ms.ToArray();
    }

    /// <summary>
    /// Generates a simulated assembly listing for debugging.
    /// </summary>
    private string GenerateAssemblyListing(string source, string entryPoint, ShaderType shaderType)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"// Shader Assembly Listing");
        sb.AppendLine($"// Entry Point: {entryPoint}");
        sb.AppendLine($"// Shader Type: {shaderType}");
        sb.AppendLine($"// Shader Model: {_config.ShaderModel}");
        sb.AppendLine($"// Platform: {_config.Platform}");
        sb.AppendLine();

        // Count resources
        int cbCount = Regex.Matches(source, @"cbuffer\s+\w+").Count;
        int texCount = Regex.Matches(source, @"Texture\w*<").Count;
        int sampCount = Regex.Matches(source, @"SamplerState\s+\w+").Count;

        sb.AppendLine($"// Constant Buffers: {cbCount}");
        sb.AppendLine($"// Textures: {texCount}");
        sb.AppendLine($"// Samplers: {sampCount}");
        sb.AppendLine();

        // Simulated instruction listing
        sb.AppendLine($"// Estimated instructions: {EstimateInstructionCount(source)}");
        sb.AppendLine();

        var functionRegex = new Regex(@"\b(?:void|float|int|uint|bool|half|float[234])\s+([a-zA-Z_]\w*)\s*\([^)]*\)\s*\{");
        foreach (Match match in functionRegex.Matches(source))
        {
            sb.AppendLine($"// Function: {match.Groups[1].Value}");
            sb.AppendLine($"//   Estimated instructions: {EstimateInstructionCount(source.Substring(match.Index, Math.Min(500, source.Length - match.Index)))}");
        }

        return sb.ToString();
    }

    /// <summary>
    /// Computes a cache key for a shader variant.
    /// </summary>
    private string ComputeCacheKey(string source, string entryPoint, ShaderType shaderType)
    {
        var combined = $"{source}|{entryPoint}|{shaderType}|{_config.ShaderModel}|{_config.Platform}|{_config.EnableDebugInfo}|{_config.EnableOptimization}";
        byte[] bytes = Encoding.UTF8.GetBytes(combined);
        byte[] hash = MD5.HashData(bytes);
        return Convert.ToHexString(hash);
    }

    /// <summary>
    /// Clears the shader cache.
    /// </summary>
    public void ClearCache()
    {
        _cache.Clear();
    }

    /// <summary>
    /// Removes a specific entry from the cache.
    /// </summary>
    public bool RemoveFromCache(string sourceHash)
    {
        return _cache.TryRemove(sourceHash, out _);
    }

    /// <summary>
    /// Gets cache statistics.
    /// </summary>
    public (int Entries, long SizeEstimateBytes) GetCacheStats()
    {
        long size = _cache.Values.Sum(r => r.Bytecode?.Length ?? 0);
        return (_cache.Count, size);
    }

    /// <summary>
    /// Disposes resources held by the compiler.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _cache.Clear();
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Shader type enumeration for compilation targets.
/// </summary>
public enum ShaderType
{
    /// <summary>Pixel/fragment shader.</summary>
    PixelShader = 0,

    /// <summary>Vertex shader.</summary>
    VertexShader = 1,

    /// <summary>Geometry shader.</summary>
    GeometryShader = 2,

    /// <summary>Hull ( tessellation control) shader.</summary>
    HullShader = 3,

    /// <summary>Domain (tessellation evaluation) shader.</summary>
    DomainShader = 4,

    /// <summary>Compute shader.</summary>
    ComputeShader = 5,

    /// <summary>Mesh shader (SM 6.5+).</summary>
    MeshShader = 6,

    /// <summary>Amplification shader (SM 6.5+).</summary>
    AmplificationShader = 7
}
