using System;
// ============================================================
// FILE: HLSLCodeGenerator.cs
// PATH: GPU/HLSLCodeGenerator.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.GPU;

/// <summary>
/// HLSL scalar types.
/// </summary>
public enum HLSLScalarType
{
    /// <summary>32-bit float.</summary>
    Float,

    /// <summary>16-bit half precision float.</summary>
    Half,

    /// <summary>32-bit integer.</summary>
    Int,

    /// <summary>Unsigned 32-bit integer.</summary>
    UInt,

    /// <summary>Boolean.</summary>
    Bool
}

/// <summary>
/// HLSL vector dimension.
/// </summary>
public enum HLSLVectorDimension
{
    /// <summary>Scalar (1 component).</summary>
    Scalar = 1,

    /// <summary>2-component vector.</summary>
    Vector2 = 2,

    /// <summary>3-component vector.</summary>
    Vector3 = 3,

    /// <summary>4-component vector.</summary>
    Vector4 = 4
}

/// <summary>
/// HLSL register types for resource binding.
/// </summary>
public enum HLSLRegisterType
{
    /// <summary>Constant buffer register (b).</summary>
    ConstantBuffer,

    /// <summary>Texture register (t).</summary>
    Texture,

    /// <summary>Sampler register (s).</summary>
    Sampler,

    /// <summary>Unordered access view register (u).</summary>
    UAV,

    /// <summary>Temporary register (r).</summary>
    Temp,

    /// <summary>Input register (v).</summary>
    Input,

    /// <summary>Output register (o).</summary>
    Output
}

/// <summary>
/// Represents an HLSL type with its scalar base and vector dimension.
/// </summary>
public readonly struct HLSLType : IEquatable<HLSLType>
{
    /// <summary>Scalar base type.</summary>
    public readonly HLSLScalarType ScalarType;

    /// <summary>Vector dimension.</summary>
    public readonly HLSLVectorDimension Dimension;

    /// <summary>Array size (0 for non-arrays).</summary>
    public readonly int ArraySize;

    /// <summary>Creates a new HLSL type.</summary>
    public HLSLType(HLSLScalarType scalarType, HLSLVectorDimension dimension = HLSLVectorDimension.Scalar, int arraySize = 0)
    {
        ScalarType = scalarType;
        Dimension = dimension;
        ArraySize = arraySize;
    }

    /// <summary>Gets the HLSL type name string.</summary>
    public string TypeName => (ScalarType, Dimension) switch
    {
        (HLSLScalarType.Float, HLSLVectorDimension.Scalar) => "float",
        (HLSLScalarType.Float, HLSLVectorDimension.Vector2) => "float2",
        (HLSLScalarType.Float, HLSLVectorDimension.Vector3) => "float3",
        (HLSLScalarType.Float, HLSLVectorDimension.Vector4) => "float4",
        (HLSLScalarType.Half, HLSLVectorDimension.Scalar) => "half",
        (HLSLScalarType.Half, HLSLVectorDimension.Vector2) => "half2",
        (HLSLScalarType.Half, HLSLVectorDimension.Vector3) => "half3",
        (HLSLScalarType.Half, HLSLVectorDimension.Vector4) => "half4",
        (HLSLScalarType.Int, HLSLVectorDimension.Scalar) => "int",
        (HLSLScalarType.Int, HLSLVectorDimension.Vector2) => "int2",
        (HLSLScalarType.Int, HLSLVectorDimension.Vector3) => "int3",
        (HLSLScalarType.Int, HLSLVectorDimension.Vector4) => "int4",
        (HLSLScalarType.UInt, HLSLVectorDimension.Scalar) => "uint",
        (HLSLScalarType.UInt, HLSLVectorDimension.Vector2) => "uint2",
        (HLSLScalarType.UInt, HLSLVectorDimension.Vector3) => "uint3",
        (HLSLScalarType.UInt, HLSLVectorDimension.Vector4) => "uint4",
        (HLSLScalarType.Bool, HLSLVectorDimension.Scalar) => "bool",
        (HLSLScalarType.Bool, HLSLVectorDimension.Vector2) => "bool2",
        (HLSLScalarType.Bool, HLSLVectorDimension.Vector3) => "bool3",
        (HLSLScalarType.Bool, HLSLVectorDimension.Vector4) => "bool4",
        _ => "float"
    };

    /// <summary>Gets the full type name including array suffix.</summary>
    public string FullName => ArraySize > 0 ? $"{TypeName}[{ArraySize}]" : TypeName;

    /// <summary>Gets the number of scalar components.</summary>
    public int ComponentCount => (int)Dimension;

    /// <summary>Gets the stride in bytes.</summary>
    public int StrideBytes => ScalarType switch
    {
        HLSLScalarType.Float => ComponentCount * 4,
        HLSLScalarType.Half => ComponentCount * 2,
        HLSLScalarType.Int => ComponentCount * 4,
        HLSLScalarType.UInt => ComponentCount * 4,
        HLSLScalarType.Bool => ComponentCount * 4,
        _ => ComponentCount * 4
    };

    /// <summary>Creates a float type.</summary>
    public static HLSLType Float(HLSLVectorDimension dim = HLSLVectorDimension.Scalar, int arraySize = 0)
        => new(HLSLScalarType.Float, dim, arraySize);

    /// <summary>Creates a float2 type.</summary>
    public static HLSLType Float2(int arraySize = 0)
        => new(HLSLScalarType.Float, HLSLVectorDimension.Vector2, arraySize);

    /// <summary>Creates a float3 type.</summary>
    public static HLSLType Float3(int arraySize = 0)
        => new(HLSLScalarType.Float, HLSLVectorDimension.Vector3, arraySize);

    /// <summary>Creates a float4 type.</summary>
    public static HLSLType Float4(int arraySize = 0)
        => new(HLSLScalarType.Float, HLSLVectorDimension.Vector4, arraySize);

    /// <summary>Creates an int type.</summary>
    public static HLSLType Int(HLSLVectorDimension dim = HLSLVectorDimension.Scalar, int arraySize = 0)
        => new(HLSLScalarType.Int, dim, arraySize);

    /// <summary>Creates a matrix type (float4x4).</summary>
    public static HLSLType Matrix4x4()
        => new(HLSLScalarType.Float, HLSLVectorDimension.Vector4, 4);

    public bool Equals(HLSLType other) =>
        ScalarType == other.ScalarType && Dimension == other.Dimension && ArraySize == other.ArraySize;

    public override bool Equals(object? obj) => obj is HLSLType other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(ScalarType, Dimension, ArraySize);
    public static bool operator ==(HLSLType left, HLSLType right) => left.Equals(right);
    public static bool operator !=(HLSLType left, HLSLType right) => !left.Equals(right);

    public override string ToString() => FullName;
}

/// <summary>
/// Represents a named HLSL variable with its type and optional semantic.
/// </summary>
public sealed class HLSLVariable
{
    /// <summary>Variable name.</summary>
    public required string Name { get; init; }

    /// <summary>HLSL type.</summary>
    public required HLSLType Type { get; init; }

    /// <summary>Optional semantic (e.g. SV_POSITION).</summary>
    public string? Semantic { get; init; }

    /// <summary>Register binding (e.g. "b0", "t1").</summary>
    public string? Register { get; init; }

    /// <summary>Whether this is a constant/static variable.</summary>
    public bool IsConst { get; init; }

    /// <summary>Whether this is an inout parameter.</summary>
    public bool IsInout { get; init; }

    /// <summary>Initial value expression (if any).</summary>
    public string? InitialValue { get; init; }

    /// <summary>Generates the HLSL declaration string.</summary>
    public string ToDeclaration()
    {
        var sb = new StringBuilder();
        if (IsConst)
            sb.Append("const ");
        sb.Append(Type.FullName);
        sb.Append(' ');
        sb.Append(Name);
        if (!string.IsNullOrEmpty(Register))
            sb.Append($" : register({Register})");
        if (!string.IsNullOrEmpty(Semantic))
            sb.Append($" : {Semantic}");
        if (!string.IsNullOrEmpty(InitialValue))
            sb.Append($" = {InitialValue}");
        sb.Append(';');
        return sb.ToString();
    }

    /// <summary>Generates the HLSL parameter string for function signatures.</summary>
    public string ToParameter()
    {
        var sb = new StringBuilder();
        if (IsInout)
            sb.Append("inout ");
        sb.Append(Type.FullName);
        sb.Append(' ');
        sb.Append(Name);
        return sb.ToString();
    }
}

/// <summary>
/// Represents an HLSL function parameter.
/// </summary>
public sealed class HLSLParameter
{
    /// <summary>Parameter name.</summary>
    public required string Name { get; init; }

    /// <summary>HLSL type.</summary>
    public required HLSLType Type { get; init; }

    /// <summary>Whether this is an input parameter.</summary>
    public bool IsInput { get; init; } = true;

    /// <summary>Whether this is an output parameter.</summary>
    public bool IsOutput { get; init; }

    /// <summary>Whether this is an inout parameter.</summary>
    public bool IsInout { get; init; }

    /// <summary>Optional semantic.</summary>
    public string? Semantic { get; init; }

    /// <summary>Generates the HLSL parameter string.</summary>
    public string ToParameterString()
    {
        var sb = new StringBuilder();
        if (IsInout)
            sb.Append("inout ");
        else if (IsOutput)
            sb.Append("out ");
        sb.Append(Type.FullName);
        sb.Append(' ');
        sb.Append(Name);
        if (!string.IsNullOrEmpty(Semantic))
            sb.Append($" : {Semantic}");
        return sb.ToString();
    }
}

/// <summary>
/// Represents an HLSL function definition.
/// </summary>
public sealed class HLSLFunction
{
    /// <summary>Return type.</summary>
    public required HLSLType ReturnType { get; init; }

    /// <summary>Function name.</summary>
    public required string Name { get; init; }

    /// <summary>Function parameters.</summary>
    public List<HLSLParameter> Parameters { get; init; } = new();

    /// <summary>Function body (HLSL code).</summary>
    public string Body { get; set; } = string.Empty;

    /// <summary>Whether to emit [unroll] attribute on loops in the body.</summary>
    public bool UnrollLoops { get; init; }

    /// <summary>Whether this function is a pixel shader entry point.</summary>
    public bool IsEntryPoint { get; init; }

    /// <summary>Optional semantics for entry points.</summary>
    public string? EntryPointSemantic { get; init; }

    /// <summary>Generates the complete HLSL function.</summary>
    public string ToHLSL()
    {
        var sb = new StringBuilder();

        sb.Append($"{ReturnType.FullName} {Name}(");
        for (int i = 0; i < Parameters.Count; i++)
        {
            if (i > 0)
                sb.Append(", ");
            sb.Append(Parameters[i].ToParameterString());
        }
        sb.Append(')');

        if (!string.IsNullOrEmpty(EntryPointSemantic))
            sb.Append($" : {EntryPointSemantic}");

        sb.AppendLine();
        sb.AppendLine("{");

        if (UnrollLoops)
        {
            // Wrap body with unroll hints
            string[] lines = Body.Split('\n');
            foreach (string line in lines)
            {
                if (line.TrimStart().StartsWith("for (") && !line.Contains("[unroll]"))
                {
                    sb.AppendLine("    [unroll]");
                }
                sb.AppendLine(line);
            }
        }
        else
        {
            sb.AppendLine(Body);
        }

        sb.AppendLine("}");
        return sb.ToString();
    }
}

/// <summary>
/// Low-level HLSL code generation utilities.
/// Provides expression tree translation, type system, built-in function generation,
/// loop unrolling, branch optimization, and register allocation.
/// </summary>
public sealed class HLSLCodeGenerator
{
    private readonly StringBuilder _sb;
    private readonly List<HLSLVariable> _globalVariables;
    private readonly List<HLSLFunction> _functions;
    private readonly Dictionary<string, int> _registerCounts;
    private int _tempCounter;

    /// <summary>Creates a new HLSL code generator.</summary>
    public HLSLCodeGenerator()
    {
        _sb = new StringBuilder();
        _globalVariables = new List<HLSLVariable>();
        _functions = new List<HLSLFunction>();
        _registerCounts = new Dictionary<string, int>();
        _tempCounter = 0;
    }

    /// <summary>Gets or sets the indentation level.</summary>
    public int IndentLevel { get; set; }

    /// <summary>Gets the current indentation string.</summary>
    public string Indent => new(' ', IndentLevel * 4);

    /// <summary>
    /// Generates an HLSL float literal.
    /// </summary>
    public static string FloatLiteral(float value)
    {
        if (value == 0.0f)
            return "0.0";
        if (value == 1.0f)
            return "1.0";
        if (value == -1.0f)
            return "-1.0";
        if (float.IsNaN(value))
            return "asfloat(0x7FC00000)";
        if (float.IsPositiveInfinity(value))
            return "asfloat(0x7F800000)";
        if (float.IsNegativeInfinity(value))
            return "asfloat(0xFF800000)";

        string formatted = value.ToString("G");
        if (!formatted.Contains('.') && !formatted.Contains('e') && !formatted.Contains('E'))
            formatted += ".0";
        return formatted;
    }

    /// <summary>
    /// Generates an HLSL float2 constructor.
    /// </summary>
    public static string Float2(float x, float y)
        => $"float2({FloatLiteral(x)}, {FloatLiteral(y)})";

    /// <summary>
    /// Generates an HLSL float2 constructor from expressions.
    /// </summary>
    public static string Float2(string x, string y)
        => $"float2({x}, {y})";

    /// <summary>
    /// Generates an HLSL float3 constructor.
    /// </summary>
    public static string Float3(float x, float y, float z)
        => $"float3({FloatLiteral(x)}, {FloatLiteral(y)}, {FloatLiteral(z)})";

    /// <summary>
    /// Generates an HLSL float3 constructor from expressions.
    /// </summary>
    public static string Float3(string x, string y, string z)
        => $"float3({x}, {y}, {z})";

    /// <summary>
    /// Generates an HLSL float4 constructor.
    /// </summary>
    public static string Float4(float x, float y, float z, float w)
        => $"float4({FloatLiteral(x)}, {FloatLiteral(y)}, {FloatLiteral(z)}, {FloatLiteral(w)})";

    /// <summary>
    /// Generates an HLSL float4 constructor from expressions.
    /// </summary>
    public static string Float4(string x, string y, string z, string w)
        => $"float4({x}, {y}, {z}, {w})";

    /// <summary>
    /// Generates an HLSL float4x4 identity matrix.
    /// </summary>
    public static string IdentityMatrix()
        => "float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)";

    /// <summary>
    /// Generates an HLSL dot product expression.
    /// </summary>
    public static string Dot(string a, string b)
        => $"dot({a}, {b})";

    /// <summary>
    /// Generates an HLSL cross product expression.
    /// </summary>
    public static string Cross(string a, string b)
        => $"cross({a}, {b})";

    /// <summary>
    /// Generates an HLSL normalize expression.
    /// </summary>
    public static string Normalize(string v)
        => $"normalize({v})";

    /// <summary>
    /// Generates an HLSL length expression.
    /// </summary>
    public static string Length(string v)
        => $"length({v})";

    /// <summary>
    /// Generates an HLSL distance expression.
    /// </summary>
    public static string Distance(string a, string b)
        => $"distance({a}, {b})";

    /// <summary>
    /// Generates an HLSL lerp expression.
    /// </summary>
    public static string Lerp(string a, string b, string t)
        => $"lerp({a}, {b}, {t})";

    /// <summary>
    /// Generates an HLSL clamp expression.
    /// </summary>
    public static string Clamp(string value, string min, string max)
        => $"clamp({value}, {min}, {max})";

    /// <summary>
    /// Generates an HLSL saturate expression.
    /// </summary>
    public static string Saturate(string v)
        => $"saturate({v})";

    /// <summary>
    /// Generates an HLSL step expression.
    /// </summary>
    public static string Step(string edge, string x)
        => $"step({edge}, {x})";

    /// <summary>
    /// Generates an HLSL smoothstep expression.
    /// </summary>
    public static string Smoothstep(string edge0, string edge1, string x)
        => $"smoothstep({edge0}, {edge1}, {x})";

    /// <summary>
    /// Generates an HLSL min expression.
    /// </summary>
    public static string Min(string a, string b)
        => $"min({a}, {b})";

    /// <summary>
    /// Generates an HLSL max expression.
    /// </summary>
    public static string Max(string a, string b)
        => $"max({a}, {b})";

    /// <summary>
    /// Generates an HLSL abs expression.
    /// </summary>
    public static string Abs(string v)
        => $"abs({v})";

    /// <summary>
    /// Generates an HLSL sqrt expression.
    /// </summary>
    public static string Sqrt(string v)
        => $"sqrt({v})";

    /// <summary>
    /// Generates an HLSL exp expression.
    /// </summary>
    public static string Exp(string v)
        => $"exp({v})";

    /// <summary>
    /// Generates an HLSL exp2 expression.
    /// </summary>
    public static string Exp2(string v)
        => $"exp2({v})";

    /// <summary>
    /// Generates an HLSL log expression.
    /// </summary>
    public static string Log(string v)
        => $"log({v})";

    /// <summary>
    /// Generates an HLSL log2 expression.
    /// </summary>
    public static string Log2(string v)
        => $"log2({v})";

    /// <summary>
    /// Generates an HLSL pow expression.
    /// </summary>
    public static string Pow(string baseVal, string exponent)
        => $"pow({baseVal}, {exponent})";

    /// <summary>
    /// Generates an HLSL sin expression.
    /// </summary>
    public static string Sin(string v)
        => $"sin({v})";

    /// <summary>
    /// Generates an HLSL cos expression.
    /// </summary>
    public static string Cos(string v)
        => $"cos({v})";

    /// <summary>
    /// Generates an HLSL tan expression.
    /// </summary>
    public static string Tan(string v)
        => $"tan({v})";

    /// <summary>
    /// Generates an HLSL asin expression.
    /// </summary>
    public static string Asin(string v)
        => $"asin({v})";

    /// <summary>
    /// Generates an HLSL acos expression.
    /// </summary>
    public static string Acos(string v)
        => $"acos({v})";

    /// <summary>
    /// Generates an HLSL atan2 expression.
    /// </summary>
    public static string Atan2(string y, string x)
        => $"atan2({y}, {x})";

    /// <summary>
    /// Generates an HLSL frac expression.
    /// </summary>
    public static string Frac(string v)
        => $"frac({v})";

    /// <summary>
    /// Generates an HLSL floor expression.
    /// </summary>
    public static string Floor(string v)
        => $"floor({v})";

    /// <summary>
    /// Generates an HLSL ceil expression.
    /// </summary>
    public static string Ceil(string v)
        => $"ceil({v})";

    /// <summary>
    /// Generates an HLSL round expression.
    /// </summary>
    public static string Round(string v)
        => $"round({v})";

    /// <summary>
    /// Generates an HLSL sign expression.
    /// </summary>
    public static string Sign(string v)
        => $"sign({v})";

    /// <summary>
    /// Generates an HLSL trunc expression.
    /// </summary>
    public static string Trunc(string v)
        => $"trunc({v})";

    /// <summary>
    /// Generates an HLSL fmod expression.
    /// </summary>
    public static string Fmod(string a, string b)
        => $"fmod({a}, {b})";

    /// <summary>
    /// Generates an HLSL reflect expression.
    /// </summary>
    public static string Reflect(string incident, string normal)
        => $"reflect({incident}, {normal})";

    /// <summary>
    /// Generates an HLSL refract expression.
    /// </summary>
    public static string Refract(string incident, string normal, string eta)
        => $"refract({incident}, {normal}, {eta})";

    /// <summary>
    /// Generates an HLSL mul expression.
    /// </summary>
    public static string Mul(string matrix, string vector)
        => $"mul({matrix}, {vector})";

    /// <summary>
    /// Generates an HLSL transpose expression.
    /// </summary>
    public static string Transpose(string m)
        => $"transpose({m})";

    /// <summary>
    /// Generates an HLSL determinant expression.
    /// </summary>
    public static string Determinant(string m)
        => $"determinant({m})";

    /// <summary>
    /// Generates an HLSL saturate expression with custom bounds.
    /// </summary>
    public static string SaturateCustom(string v, string min, string max)
        => $"saturate(({v} - {min}) / ({max} - {min}))";

    /// <summary>
    /// Generates an HLSL remap expression (linear interpolation from one range to another).
    /// </summary>
    public static string Remap(string value, string inMin, string inMax, string outMin, string outMax)
        => $"lerp({outMin}, {outMax}, saturate(({value} - {inMin}) / ({inMax} - {inMin})))";

    /// <summary>
    /// Generates an HLSL noise function call.
    /// </summary>
    public static string Noise(string v)
        => $"noise({v})";

    /// <summary>
    /// Generates an HLSL dd{x,y,z} derivative function.
    /// </summary>
    public static string Ddx(string v) => $"ddx({v})";

    /// <summary>
    /// Generates an HLSL ddy function.
    /// </summary>
    public static string Ddy(string v) => $"ddy({v})";

    /// <summary>
    /// Generates an HLSL ddx_coarse function.
    /// </summary>
    public static string DdxCoarse(string v) => $"ddx_coarse({v})";

    /// <summary>
    /// Generates an HLSL ddy_fine function.
    /// </summary>
    public static string DdyFine(string v) => $"ddy_fine({v})";

    /// <summary>
    /// Generates an HLSL any function.
    /// </summary>
    public static string Any(string v) => $"any({v})";

    /// <summary>
    /// Generates an HLSL all function.
    /// </summary>
    public static string All(string v) => $"all({v})";

    /// <summary>
    /// Generates an HLSL asfloat reinterpretation.
    /// </summary>
    public static string AsFloat(string v) => $"asfloat({v})";

    /// <summary>
    /// Generates an HLSL asuint reinterpretation.
    /// </summary>
    public static string AsUInt(string v) => $"asuint({v})";

    /// <summary>
    /// Generates an HLSL asint reinterpretation.
    /// </summary>
    public static string AsInt(string v) => $"asint({v})";

    /// <summary>
    /// Generates an HLSL if-else statement.
    /// </summary>
    public string EmitIfElse(string condition, string trueBlock, string falseBlock)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"{Indent}if ({condition})");
        sb.AppendLine($"{Indent}{{");
        IndentLevel++;
        sb.AppendLine($"{Indent}{trueBlock.Trim()}");
        IndentLevel--;
        sb.AppendLine($"{Indent}}}");
        if (!string.IsNullOrWhiteSpace(falseBlock))
        {
            sb.AppendLine($"{Indent}else");
            sb.AppendLine($"{Indent}{{");
            IndentLevel++;
            sb.AppendLine($"{Indent}{falseBlock.Trim()}");
            IndentLevel--;
            sb.AppendLine($"{Indent}}}");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL for loop statement.
    /// </summary>
    public string EmitForLoop(string init, string condition, string increment, string body, bool unroll = false)
    {
        var sb = new StringBuilder();
        if (unroll)
            sb.AppendLine($"{Indent}[unroll]");
        sb.AppendLine($"{Indent}for ({init}; {condition}; {increment})");
        sb.AppendLine($"{Indent}{{");
        IndentLevel++;
        sb.AppendLine($"{Indent}{body.Trim()}");
        IndentLevel--;
        sb.AppendLine($"{Indent}}}");
        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL while loop statement.
    /// </summary>
    public string EmitWhileLoop(string condition, string body, bool unroll = false)
    {
        var sb = new StringBuilder();
        if (unroll)
            sb.AppendLine($"{Indent}[unroll]");
        sb.AppendLine($"{Indent}while ({condition})");
        sb.AppendLine($"{Indent}{{");
        IndentLevel++;
        sb.AppendLine($"{Indent}{body.Trim()}");
        IndentLevel--;
        sb.AppendLine($"{Indent}}}");
        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL loop with a fixed iteration count using [unroll].
    /// </summary>
    public string EmitUnrolledLoop(string iteratorVar, int count, string body)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"{Indent}[unroll]");
        sb.AppendLine($"{Indent}for (int {iteratorVar} = 0; {iteratorVar} < {count}; {iteratorVar}++)");
        sb.AppendLine($"{Indent}{{");
        IndentLevel++;
        foreach (string line in body.Split('\n'))
        {
            if (!string.IsNullOrWhiteSpace(line))
                sb.AppendLine($"{Indent}{line.Trim()}");
        }
        IndentLevel--;
        sb.AppendLine($"{Indent}}}");
        return sb.ToString();
    }

    /// <summary>
    /// Generates a switch-case statement.
    /// </summary>
    public string EmitSwitch(string expression, Dictionary<int, string> cases, string? defaultCase = null)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"{Indent}switch ({expression})");
        sb.AppendLine($"{Indent}{{");
        IndentLevel++;

        foreach (var kvp in cases)
        {
            sb.AppendLine($"{Indent}case {kvp.Key}:");
            IndentLevel++;
            sb.AppendLine($"{Indent}{kvp.Value.Trim()}");
            sb.AppendLine($"{Indent}break;");
            IndentLevel--;
        }

        if (defaultCase != null)
        {
            sb.AppendLine($"{Indent}default:");
            IndentLevel++;
            sb.AppendLine($"{Indent}{defaultCase.Trim()}");
            sb.AppendLine($"{Indent}break;");
            IndentLevel--;
        }

        IndentLevel--;
        sb.AppendLine($"{Indent}}}");
        return sb.ToString();
    }

    /// <summary>
    /// Generates a ternary expression.
    /// </summary>
    public static string Ternary(string condition, string trueValue, string falseValue)
        => $"({condition}) ? ({trueValue}) : ({falseValue})";

    /// <summary>
    /// Generates an HLSL variable declaration with optional initialization.
    /// </summary>
    public string EmitVariableDeclaration(HLSLType type, string name, string? initialValue = null)
    {
        var sb = new StringBuilder();
        sb.Append($"{Indent}{type.FullName} {name}");
        if (initialValue != null)
            sb.Append($" = {initialValue}");
        sb.AppendLine(";");
        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL variable declaration for a float4 array.
    /// </summary>
    public string EmitFloat4ArrayDeclaration(string name, int size, ReadOnlySpan<float> initialValues = default)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"{Indent}float4 {name}[{size}];");

        if (initialValues.Length > 0)
        {
            for (int i = 0; i < Math.Min(initialValues.Length / 4, size); i++)
            {
                int baseIdx = i * 4;
                float x = initialValues.Length > baseIdx ? initialValues[baseIdx] : 0;
                float y = initialValues.Length > baseIdx + 1 ? initialValues[baseIdx + 1] : 0;
                float z = initialValues.Length > baseIdx + 2 ? initialValues[baseIdx + 2] : 0;
                float w = initialValues.Length > baseIdx + 3 ? initialValues[baseIdx + 3] : 0;
                sb.AppendLine($"{Indent}{name}[{i}] = {Float4(x, y, z, w)};");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL return statement.
    /// </summary>
    public string EmitReturn(string? value = null)
    {
        if (value == null)
            return $"{Indent}return;";
        return $"{Indent}return {value};";
    }

    /// <summary>
    /// Generates an HLSL discard statement.
    /// </summary>
    public string EmitDiscard()
        => $"{Indent}discard;";

    /// <summary>
    /// Generates an HLSL comment block.
    /// </summary>
    public string EmitComment(string text)
    {
        var sb = new StringBuilder();
        string[] lines = text.Split('\n');
        if (lines.Length == 1)
        {
            sb.AppendLine($"{Indent}// {text}");
        }
        else
        {
            sb.AppendLine($"{Indent}/*");
            foreach (string line in lines)
                sb.AppendLine($"{Indent} * {line}");
            sb.AppendLine($"{Indent} */");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL texture sample expression.
    /// </summary>
    public static string TextureSample(string textureName, string samplerName, string uv)
        => $"{textureName}.Sample({samplerName}, {uv})";

    /// <summary>
    /// Generates an HLSL texture sample with LOD.
    /// </summary>
    public static string TextureSampleLevel(string textureName, string samplerName, string uv, string lod)
        => $"{textureName}.SampleLevel({samplerName}, {uv}, {lod})";

    /// <summary>
    /// Generates an HLSL texture sample with gradient.
    /// </summary>
    public static string TextureSampleGrad(string textureName, string samplerName, string uv, string ddx, string ddy)
        => $"{textureName}.SampleGrad({samplerName}, {uv}, {ddx}, {ddy})";

    /// <summary>
    /// Generates an HLSL texture load expression.
    /// </summary>
    public static string TextureLoad(string textureName, string location, int mipLevel = 0)
        => $"{textureName}.Load(int4({location}, {mipLevel}))";

    /// <summary>
    /// Generates an HLSL texture dimensions query.
    /// </summary>
    public static string TextureDimensions(string textureName, int mipLevel = 0)
        => $"{textureName}.Dimensions(mipLevel)";

    /// <summary>
    /// Generates an HLSL texture store expression (UAV).
    /// </summary>
    public static string TextureStore(string textureName, string location, string value)
        => $"{textureName}[{location}] = {value}";

    /// <summary>
    /// Generates an HLSL buffer load expression.
    /// </summary>
    public static string BufferLoad(string bufferName, string index)
        => $"{bufferName}[{index}]";

    /// <summary>
    /// Generates an HLSL structured buffer load.
    /// </summary>
    public static string StructuredBufferLoad(string bufferName, string index)
        => $"{bufferName}[{index}]";

    /// <summary>
    /// Generates an HLSL RWBuffer store expression.
    /// </summary>
    public static string RWBufferStore(string bufferName, string index, string value)
        => $"{bufferName}[{index}] = {value}";

    /// <summary>
    /// Generates an HLSL groupshared memory declaration.
    /// </summary>
    public string EmitGroupSharedDeclaration(HLSLType type, string name, int size)
        => $"{Indent}groupshared {type.FullName} {name}[{size}];";

    /// <summary>
    /// Generates an HLSL barrier/sync statement.
    /// </summary>
    public string EmitBarrier(string syncType = "GroupMemoryWithGroupSync")
        => $"{Indent}GroupMemoryBarrierWithGroupSync();";

    /// <summary>
    /// Generates an HLSL [branch] attribute.
    /// </summary>
    public string EmitBranch(string condition, string body)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"{Indent}[branch]");
        sb.AppendLine($"{Indent}if ({condition})");
        sb.AppendLine($"{Indent}{{");
        IndentLevel++;
        sb.AppendLine($"{Indent}{body.Trim()}");
        IndentLevel--;
        sb.AppendLine($"{Indent}}}");
        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL [flatten] attribute for avoiding dynamic branching.
    /// </summary>
    public string EmitFlatten(string condition, string trueBody, string falseBody = "")
    {
        var sb = new StringBuilder();
        sb.AppendLine($"{Indent}[flatten]");
        sb.AppendLine($"{Indent}if ({condition})");
        sb.AppendLine($"{Indent}{{");
        IndentLevel++;
        sb.AppendLine($"{Indent}{trueBody.Trim()}");
        IndentLevel--;
        sb.AppendLine($"{Indent}}}");
        if (!string.IsNullOrWhiteSpace(falseBody))
        {
            sb.AppendLine($"{Indent}else");
            sb.AppendLine($"{Indent}{{");
            IndentLevel++;
            sb.AppendLine($"{Indent}{falseBody.Trim()}");
            IndentLevel--;
            sb.AppendLine($"{Indent}}}");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL [call] attribute for function calls.
    /// </summary>
    public string EmitCall(string functionName, params string[] args)
    {
        return $"{Indent}{functionName}({string.Join(", ", args)});";
    }

    /// <summary>
    /// Generates an HLSL assignment statement.
    /// </summary>
    public string EmitAssignment(string target, string value)
        => $"{Indent}{target} = {value};";

    /// <summary>
    /// Generates an HLSL compound assignment (+=, -=, *=, /=).
    /// </summary>
    public string EmitCompoundAssignment(string target, string op, string value)
        => $"{Indent}{target} {op}= {value};";

    /// <summary>
    /// Generates a temporary variable and returns its name.
    /// </summary>
    public string GenerateTempVariable(HLSLType type, string? initialValue = null)
    {
        string name = $"_tmp{_tempCounter++}";
        return name;
    }

    /// <summary>
    /// Allocates a register for the given resource type.
    /// </summary>
    public string AllocateRegister(HLSLRegisterType registerType)
    {
        string prefix = registerType switch
        {
            HLSLRegisterType.ConstantBuffer => "b",
            HLSLRegisterType.Texture => "t",
            HLSLRegisterType.Sampler => "s",
            HLSLRegisterType.UAV => "u",
            HLSLRegisterType.Temp => "r",
            HLSLRegisterType.Input => "v",
            HLSLRegisterType.Output => "o",
            _ => "r"
        };

        if (!_registerCounts.ContainsKey(prefix))
            _registerCounts[prefix] = 0;

        int index = _registerCounts[prefix]++;
        return $"{prefix}{index}";
    }

    /// <summary>
    /// Gets the next available register index for a type.
    /// </summary>
    public int PeekNextRegister(HLSLRegisterType registerType)
    {
        string prefix = registerType switch
        {
            HLSLRegisterType.ConstantBuffer => "b",
            HLSLRegisterType.Texture => "t",
            HLSLRegisterType.Sampler => "s",
            HLSLRegisterType.UAV => "u",
            HLSLRegisterType.Temp => "r",
            HLSLRegisterType.Input => "v",
            HLSLRegisterType.Output => "o",
            _ => "r"
        };

        return _registerCounts.GetValueOrDefault(prefix, 0);
    }

    /// <summary>
    /// Resets the temporary variable counter.
    /// </summary>
    public void ResetTemps()
    {
        _tempCounter = 0;
    }

    /// <summary>
    /// Resets all register allocations.
    /// </summary>
    public void ResetRegisters()
    {
        _registerCounts.Clear();
        _tempCounter = 0;
    }

    /// <summary>
    /// Generates an HLSL pragma directive.
    /// </summary>
    public static string EmitPragma(string directive)
        => $"#pragma {directive}";

    /// <summary>
    /// Generates an HLSL #define directive.
    /// </summary>
    public static string EmitDefine(string name, string? value = null)
    {
        if (value == null)
            return $"#define {name}";
        return $"#define {name} {value}";
    }

    /// <summary>
    /// Generates an HLSL #ifdef block.
    /// </summary>
    public string EmitIfDef(string symbol, string trueBlock, string? falseBlock = null)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"#ifdef {symbol}");
        sb.AppendLine($"{trueBlock}");
        if (falseBlock != null)
        {
            sb.AppendLine("#else");
            sb.AppendLine($"{falseBlock}");
        }
        sb.AppendLine("#endif");
        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL #if block with a constant expression.
    /// </summary>
    public string EmitIf(string condition, string trueBlock, string? falseBlock = null)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"#if {condition}");
        sb.AppendLine($"{trueBlock}");
        if (falseBlock != null)
        {
            sb.AppendLine("#else");
            sb.AppendLine($"{falseBlock}");
        }
        sb.AppendLine("#endif");
        return sb.ToString();
    }

    /// <summary>
    /// Generates a matrix column access expression.
    /// </summary>
    public static string MatrixColumn(string matrix, int column)
        => $"{matrix}._m{0}{column}{1}{column}{2}{column}{3}{column}";

    /// <summary>
    /// Generates a matrix element access expression.
    /// </summary>
    public static string MatrixElement(string matrix, int row, int col)
        => $"{matrix}._m{row}{col}";

    /// <summary>
    /// Generates a swizzle expression.
    /// </summary>
    public static string Swizzle(string vector, string components)
        => $"{vector}.{components}";

    /// <summary>
    /// Generates a component extraction from a float4.
    /// </summary>
    public static string Component(string vector, int index)
    {
        return index switch
        {
            0 => $"{vector}.x",
            1 => $"{vector}.y",
            2 => $"{vector}.z",
            3 => $"{vector}.w",
            _ => $"{vector}.x"
        };
    }

    /// <summary>
    /// Generates an HLSL static variable declaration.
    /// </summary>
    public static string EmitStaticConst(string type, string name, string value)
        => $"static const {type} {name} = {value};";

    /// <summary>
    /// Generates an HLSL branchless select using lerp and step.
    /// </summary>
    public static string BranchlessSelect(string condition, string trueValue, string falseValue)
        => $"lerp({falseValue}, {trueValue}, step(0.5, {condition}))";

    /// <summary>
    /// Generates an HLSL branchless clamp using min/max.
    /// </summary>
    public static string BranchlessClamp(string value, string min, string max)
        => $"min(max({value}, {min}), {max})";

    /// <summary>
    /// Generates an HLSL inverse lerp (recovery of t from interpolation).
    /// </summary>
    public static string InverseLerp(string a, string b, string value)
        => $"saturate(({value} - {a}) / ({b} - {a}))";

    /// <summary>
    /// Generates an HLSL map (remap from one range to another).
    /// </summary>
    public static string Map(string value, string inMin, string inMax, string outMin, string outMax)
        => $"lerp({outMin}, {outMax}, saturate(({value} - {inMin}) / ({inMax} - {inMin})))";

    /// <summary>
    /// Generates an HLSL step function chain for multi-range selection.
    /// </summary>
    public static string MultiStep(string value, ReadOnlySpan<float> thresholds, ReadOnlySpan<string> values)
    {
        if (thresholds.Length == 0 || values.Length == 0)
            return "0.0";

        var sb = new StringBuilder();
        sb.Append(values[values.Length - 1]);

        for (int i = thresholds.Length - 1; i >= 0; i--)
        {
            if (i < values.Length)
            {
                sb.Insert(0, $"lerp({values[i]}, ");
                sb.Append($", step({FloatLiteral(thresholds[i])}, {value}))");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL smooth hermite interpolation.
    /// </summary>
    public static string HermiteInterpolation(string t)
        => $"({t} * {t} * (3.0 - 2.0 * {t}))";

    /// <summary>
    /// Generates an HLSL smootherstep (Ken Perlin's improved version).
    /// </summary>
    public static string SmootherStep(string t)
        => $"({t} * {t} * {t} * ({t} * ({t} * 6.0 - 15.0) + 10.0))";

    /// <summary>
    /// Generates an HLSL ping-pong function (oscillating between 0 and 1).
    /// </summary>
    public static string PingPong(string t)
        => $"1.0 - abs(2.0 * frac({t}) - 1.0)";

    /// <summary>
    /// Generates an HLSL circular repeat function.
    /// </summary>
    public static string CircularRepeat(string t)
        => $"frac({t})";

    /// <summary>
    /// Generates an HLSL noise-based hash function.
    /// </summary>
    public static string Hash21(string p)
        => $"frac(sin(dot({p}, float2(12.9898, 78.233))) * 43758.5453)";

    /// <summary>
    /// Generates an HLSL 3D noise function using hash.
    /// </summary>
    public static string Hash31(string p)
        => $"frac(sin(dot({p}, float3(12.9898, 78.233, 45.164))) * 43758.5453)";

    /// <summary>
    /// Generates an HLSL value noise function.
    /// </summary>
    public static string ValueNoise(string p)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"floor({p})");
        sb.AppendLine($"frac({p})");
        sb.AppendLine($"lerp(...)");
        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL gradient noise function.
    /// </summary>
    public static string GradientNoise(string p)
    {
        return $"(frac(sin(dot(floor({p}), float2(12.9898, 78.233))) * 43758.5453) * 2.0 - 1.0)";
    }

    /// <summary>
    /// Generates an HLSL FBM (Fractal Brownian Motion) function.
    /// </summary>
    public static string FBM(string p, int octaves, float lacunarity = 2.0f, float gain = 0.5f)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"// FBM with {octaves} octaves");
        sb.AppendLine($"float fbm({p})");
        sb.AppendLine("{");
        sb.AppendLine("    float value = 0.0;");
        sb.AppendLine("    float amplitude = 0.5;");
        sb.AppendLine($"    float frequency = 1.0;");
        for (int i = 0; i < octaves; i++)
        {
            sb.AppendLine($"    value += amplitude * noise(p * frequency);");
            sb.AppendLine($"    frequency *= {FloatLiteral(lacunarity)};");
            sb.AppendLine($"    amplitude *= {FloatLiteral(gain)};");
        }
        sb.AppendLine("    return value;");
        sb.AppendLine("}");
        return sb.ToString();
    }

    /// <summary>
    /// Generates an HLSL coordinate transform from world to screen space.
    /// </summary>
    public static string WorldToScreen(string worldPos, string viewProjection)
        => $"mul({viewProjection}, float4({worldPos}, 1.0))";

    /// <summary>
    /// Generates an HLSL perspective divide.
    /// </summary>
    public static string PerspectiveDivide(string clipPos)
        => $"({clipPos}.xyz / {clipPos}.w)";

    /// <summary>
    /// Generates an HLSL UV coordinate from screen position.
    /// </summary>
    public static string ScreenToUV(string screenPos, string screenSize)
        => $"({screenPos}.xy / {screenSize})";

    /// <summary>
    /// Generates an HLSL depth linearization.
    /// </summary>
    public static string LinearizeDepth(string depth, string nearPlane, string farPlane)
        => $"({nearPlane} * {farPlane}) / ({farPlane} - {depth} * ({farPlane} - {nearPlane}))";

    /// <summary>
    /// Generates an HLSL depth linearization for reverse-Z.
    /// </summary>
    public static string LinearizeDepthReverseZ(string depth, string nearPlane, string farPlane)
        => $"{farPlane} * {nearPlane} / ({farPlane} + {depth} * ({nearPlane} - {farPlane}))";

    /// <summary>
    /// Generates an HLSL view-space position reconstruction from depth.
    /// </summary>
    public static string ReconstructViewPos(string uv, string depth, string invProjection)
        => $"mul({invProjection}, float4(uv * 2.0 - 1.0, {depth}, 1.0)).xyz";

    /// <summary>
    /// Generates an HLSL view-space position reconstruction from depth (reverse-Z).
    /// </summary>
    public static string ReconstructViewPosReverseZ(string uv, string depth, string invProjection)
        => $"mul({invProjection}, float4(uv * 2.0 - 1.0, 1.0 - {depth}, 1.0)).xyz";

    /// <summary>
    /// Generates an HLSL world-space position reconstruction from depth.
    /// </summary>
    public static string ReconstructWorldPos(string viewPos, string invView)
        => $"mul({invView}, float4({viewPos}, 1.0)).xyz";

    /// <summary>
    /// Generates an HLSL edge-aware normal computation.
    /// </summary>
    public static string EdgeAwareNormal(string depth, string uv, string texelSize)
    {
        return $"normalize(float3(" +
               $"({texelSize}.x * 2.0) * (tex2D(_DepthTex, {uv} - float2({texelSize}.x, 0)).r - tex2D(_DepthTex, {uv} + float2({texelSize}.x, 0)).r)," +
               $"({texelSize}.y * 2.0) * (tex2D(_DepthTex, {uv} - float2(0, {texelSize}.y)).r - tex2D(_DepthTex, {uv} + float2(0, {texelSize}.y)).r)," +
               "1.0))";
    }

    /// <summary>
    /// Generates an HLSL checkerboard pattern.
    /// </summary>
    public static string Checkerboard(string uv, float scale = 8.0f)
        => $"frac(floor({uv}.x * {FloatLiteral(scale)}) + floor({uv}.y * {FloatLiteral(scale)}))";

    /// <summary>
    /// Generates an HLSL vignette effect.
    /// </summary>
    public static string Vignette(string uv, string intensity)
        => $"(1.0 - dot({uv} - 0.5, {uv} - 0.5) * {intensity})";

    /// <summary>
    /// Generates an HLSL chromatic aberration offset.
    /// </summary>
    public static string ChromaticAberration(string uv, string center, string strength)
        => $"float3({uv}.x + ({uv}.x - {center}.x) * {strength}, {uv}.y + ({uv}.y - {center}.y) * {strength}, 0.0)";

    /// <summary>
    /// Generates an HLSL dithering function.
    /// </summary>
    public static string Dither(string uv)
        => $"frac(sin(dot({uv}, float2(12.9898, 78.233))) * 43758.5453)";

    /// <summary>
    /// Generates an HLSL temporal AA jitter.
    /// </summary>
    public static string TemporalJitter(string uv, int frameIndex)
        => $"frac({uv} + float2(({frameIndex} * 0.75487766624669276005) % 1.0, ({frameIndex} * 0.56984029099805323579) % 1.0) * 0.25)";

    /// <summary>
    /// Generates an HLSL tone mapping (ACES approximation).
    /// </summary>
    public static string ACESToneMapping(string color)
    {
        return $"saturate(({color} * (2.51 * {color} + 0.03)) / ({color} * (2.43 * {color} + 0.59) + 0.14))";
    }

    /// <summary>
    /// Generates an HLSL sRGB gamma correction.
    /// </summary>
    public static string LinearToSRGB(string color)
        => $"pow({color}, 1.0 / 2.2)";

    /// <summary>
    /// Generates an HLSL linear from sRGB.
    /// </summary>
    public static string SRGBToLinear(string color)
        => $"pow({color}, 2.2)";

    /// <summary>
    /// Generates an HDR luminance calculation.
    /// </summary>
    public static string Luminance(string color)
        => $"dot({color}, float3(0.2126, 0.7152, 0.0722))";

    /// <summary>
    /// Generates an HSV to RGB conversion.
    /// </summary>
    public static string HSVtoRGB(string hsv)
    {
        return $"float3(" +
               $"abs(fmod(({hsv}.x * 6.0 + float3(0.0, 4.0, 2.0)), 6.0) - 3.0) - 1.0) * {hsv}.y * {hsv}.z";
    }

    /// <summary>
    /// Generates an RGB to HSV conversion.
    /// </summary>
    public static string RGBtoHSV(string rgb)
    {
        return $"float3(" +
               $"fmod((({rgb}.b - {rgb}.g) / ({rgb}.r - min({rgb}.g, {rgb}.b) + 0.0001) + 6.0) / 6.0, 1.0), " +
               $"max(max({rgb}.r, {rgb}.g), {rgb}.b) - min(min({rgb}.r, {rgb}.g), {rgb}.b), " +
               $"max(max({rgb}.r, {rgb}.g), {rgb}.b))";
    }

    /// <summary>
    /// Generates an HLSL matrix look-at construction.
    /// </summary>
    public static string LookAtMatrix(string eye, string target, string up)
        => $"float4x4( " +
           $"normalize({target} - {eye}), " +
           $"cross(normalize({target} - {eye}), {up}), " +
           $"{up}, " +
           $"{eye})";

    /// <summary>
    /// Generates an HLSL perspective projection matrix.
    /// </summary>
    public static string PerspectiveMatrix(string fovY, string aspect, string nearPlane, string farPlane)
    {
        return $"float4x4(" +
               $"1.0 / ({aspect} * tan({fovY} / 2.0)), 0, 0, 0," +
               $"0, 1.0 / tan({fovY} / 2.0), 0, 0," +
               $"0, 0, {farPlane} / ({nearPlane} - {farPlane}), -1," +
               $"0, 0, {nearPlane} * {farPlane} / ({nearPlane} - {farPlane}), 0)";
    }

    /// <summary>
    /// Generates an HLSL orthographic projection matrix.
    /// </summary>
    public static string OrthographicMatrix(string width, string height, string nearPlane, string farPlane)
    {
        return $"float4x4(" +
               $"2.0 / {width}, 0, 0, 0," +
               $"0, 2.0 / {height}, 0, 0," +
               $"0, 0, 1.0 / ({nearPlane} - {farPlane}), 0," +
               $"0, 0, {nearPlane} / ({nearPlane} - {farPlane}), 1)";
    }

    /// <summary>
    /// Generates an HLSL rotation matrix around an axis.
    /// </summary>
    public static string RotationMatrix(string axis, string angle)
    {
        string c = $"cos({angle})";
        string s = $"sin({angle})";
        string t = $"1.0 - cos({angle})";
        return $"float4x4(" +
               $"{t} * {axis}.x * {axis}.x + {c}, {t} * {axis}.x * {axis}.y - {s} * {axis}.z, {t} * {axis}.x * {axis}.z + {s} * {axis}.y, 0," +
               $"{t} * {axis}.x * {axis}.y + {s} * {axis}.z, {t} * {axis}.y * {axis}.y + {c}, {t} * {axis}.y * {axis}.z - {s} * {axis}.x, 0," +
               $"{t} * {axis}.x * {axis}.z - {s} * {axis}.y, {t} * {axis}.y * {axis}.z + {s} * {axis}.x, {t} * {axis}.z * {axis}.z + {c}, 0," +
               "0, 0, 0, 1)";
    }

    /// <summary>
    /// Generates an HLSL scale matrix.
    /// </summary>
    public static string ScaleMatrix(string scale)
        => $"float4x4({scale}.x, 0, 0, 0, 0, {scale}.y, 0, 0, 0, 0, {scale}.z, 0, 0, 0, 0, 1)";

    /// <summary>
    /// Generates an HLSL translation matrix.
    /// </summary>
    public static string TranslationMatrix(string translation)
        => $"float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, {translation}.x, {translation}.y, {translation}.z, 1)";

    /// <summary>
    /// Resets the generator state for reuse.
    /// </summary>
    public void Reset()
    {
        _sb.Clear();
        _globalVariables.Clear();
        _functions.Clear();
        _registerCounts.Clear();
        _tempCounter = 0;
        IndentLevel = 0;
    }
}
