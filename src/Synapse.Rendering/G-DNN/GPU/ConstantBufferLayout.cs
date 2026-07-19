using System;
// ============================================================
// FILE: ConstantBufferLayout.cs
// PATH: GPU/ConstantBufferLayout.cs
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
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Core.NeuralNetwork;

namespace GDNN.GPU;

/// <summary>
/// Alignment requirements for HLSL constant buffer packing.
/// </summary>
public enum ConstantBufferAlignment
{
    /// <summary>No specific alignment.</summary>
    None = 0,

    /// <summary>4-byte alignment (float, int).</summary>
    Align4 = 4,

    /// <summary>8-byte alignment (float2, int2).</summary>
    Align8 = 8,

    /// <summary>16-byte alignment (float4, float4x4, arrays).</summary>
    Align16 = 16,

    /// <summary>64-byte alignment (cbuffer boundary).</summary>
    Align64 = 64
}

/// <summary>
/// Represents a single field within a constant buffer layout.
/// </summary>
public sealed class ConstantBufferField
{
    /// <summary>Field name.</summary>
    public string Name { get; init; }

    /// <summary>HLSL type name.</summary>
    public string TypeName { get; init; }

    /// <summary>Offset in bytes from the start of the constant buffer.</summary>
    public int Offset { get; set; }

    /// <summary>Size in bytes of this field.</summary>
    public int SizeBytes { get; init; }

    /// <summary>Alignment requirement in bytes.</summary>
    public ConstantBufferAlignment Alignment { get; init; } = ConstantBufferAlignment.Align16;

    /// <summary>Array element count (1 for non-arrays).</summary>
    public int ArrayElementCount { get; init; } = 1;

    /// <summary>Stride between array elements in bytes.</summary>
    public int ArrayStride { get; init; }

    /// <summary>Whether this field contains packed float4 arrays.</summary>
    public bool IsPackedArray { get; init; }

    /// <summary>Sub-fields for struct members (if this is a struct type).</summary>
    public List<ConstantBufferField> SubFields { get; init; } = new();

    /// <summary>Gets the total footprint including array padding.</summary>
    public int TotalFootprint => ArrayElementCount > 1 ? ArrayStride * ArrayElementCount : SizeBytes;

    /// <summary>Gets the HLSL declaration string.</summary>
    public string ToDeclaration()
    {
        if (ArrayElementCount > 1)
            return $"    {TypeName} {Name}[{ArrayElementCount}];";
        return $"    {TypeName} {Name};";
    }
}

/// <summary>
/// Represents a complete constant buffer layout with proper alignment and padding.
/// </summary>
public sealed class ConstantBufferLayout
{
    /// <summary>Constant buffer name.</summary>
    public string Name { get; init; }

    /// <summary>Register binding (e.g. "b0").</summary>
    public string Register { get; init; }

    /// <summary>Fields in this constant buffer.</summary>
    public List<ConstantBufferField> Fields { get; init; } = new();

    /// <summary>Total size in bytes.</summary>
    public int TotalSizeBytes { get; set; }

    /// <summary>Total size in float4 elements.</summary>
    public int TotalFloat4Elements => (TotalSizeBytes + 15) / 16;

    /// <summary>Gets the field at the given byte offset, or null if none.</summary>
    public ConstantBufferField? GetFieldAtOffset(int offset)
    {
        foreach (var field in Fields)
        {
            if (offset >= field.Offset && offset < field.Offset + field.TotalFootprint)
                return field;
        }
        return null;
    }

    /// <summary>Gets the byte offset of a named field, or -1 if not found.</summary>
    public int GetFieldOffset(string name)
    {
        foreach (var field in Fields)
        {
            if (field.Name == name)
                return field.Offset;
        }
        return -1;
    }

    /// <summary>Generates the HLSL constant buffer declaration.</summary>
    public string ToHLSL()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"cbuffer {Name} : register({Register})");
        sb.AppendLine("{");
        foreach (var field in Fields)
        {
            sb.AppendLine(field.ToDeclaration());
        }
        sb.AppendLine("};");
        return sb.ToString();
    }
}

/// <summary>
/// Provides layout computation for GPU constant buffers with 16-byte packing rules.
/// Handles padding, alignment, struct packing, weight-to-float4 conversion,
/// and layout serialization for shader compilation.
/// </summary>
public sealed class ConstantBufferLayoutBuilder
{
    /// <summary>HLSL packing alignment constant (16 bytes).</summary>
    public const int HLSLAlignment = 16;

    /// <summary>Maximum constant buffer size on most GPUs (4096 bytes = 256 float4s).</summary>
    public const int MaxConstantBufferSize = 4096;

    /// <summary>Maximum number of float4 elements per constant buffer (D3D11 limit).</summary>
    public const int MaxFloat4Elements = 256;

    private readonly List<ConstantBufferField> _pendingFields;
    private int _currentOffset;
    private string _bufferName = "ConstantBuffer";
    private string _register = "b0";

    /// <summary>Creates a new layout builder.</summary>
    public ConstantBufferLayoutBuilder()
    {
        _pendingFields = new List<ConstantBufferField>();
        _currentOffset = 0;
    }

    /// <summary>Creates a new layout builder with a name and register.</summary>
    public ConstantBufferLayoutBuilder(string name, string register = "b0")
    {
        _pendingFields = new List<ConstantBufferField>();
        _currentOffset = 0;
        _bufferName = name;
        _register = register;
    }

    /// <summary>Resets the builder for reuse.</summary>
    public void Reset(string? name = null, string? register = null)
    {
        _pendingFields.Clear();
        _currentOffset = 0;
        if (name != null)
            _bufferName = name;
        if (register != null)
            _register = register;
    }

    /// <summary>Computes the required alignment for a given type size.</summary>
    public static ConstantBufferAlignment ComputeAlignment(int sizeBytes)
    {
        if (sizeBytes >= 16)
            return ConstantBufferAlignment.Align16;
        if (sizeBytes >= 8)
            return ConstantBufferAlignment.Align8;
        if (sizeBytes >= 4)
            return ConstantBufferAlignment.Align4;
        return ConstantBufferAlignment.None;
    }

    /// <summary>Computes padding bytes needed to reach the next alignment boundary.</summary>
    public static int ComputePadding(int currentOffset, ConstantBufferAlignment alignment)
    {
        int align = (int)alignment;
        if (align <= 0)
            return 0;
        int remainder = currentOffset % align;
        if (remainder == 0)
            return 0;
        return align - remainder;
    }

    /// <summary>Computes padding for float4 (16-byte) alignment.</summary>
    public static int ComputeFloat4Padding(int currentOffset)
    {
        return ComputePadding(currentOffset, ConstantBufferAlignment.Align16);
    }

    /// <summary>Adds a float field to the layout.</summary>
    public ConstantBufferLayoutBuilder AddFloat(string name)
    {
        var alignment = ComputeAlignment(4);
        int padding = ComputePadding(_currentOffset, alignment);

        _pendingFields.Add(new ConstantBufferField
        {
            Name = name,
            TypeName = "float",
            Offset = _currentOffset + padding,
            SizeBytes = 4,
            Alignment = alignment
        });

        _currentOffset += padding + 4;
        return this;
    }

    /// <summary>Adds a float2 field to the layout.</summary>
    public ConstantBufferLayoutBuilder AddFloat2(string name)
    {
        var alignment = ComputeAlignment(8);
        int padding = ComputePadding(_currentOffset, alignment);

        _pendingFields.Add(new ConstantBufferField
        {
            Name = name,
            TypeName = "float2",
            Offset = _currentOffset + padding,
            SizeBytes = 8,
            Alignment = alignment
        });

        _currentOffset += padding + 8;
        return this;
    }

    /// <summary>Adds a float3 field to the layout.</summary>
    public ConstantBufferLayoutBuilder AddFloat3(string name)
    {
        var alignment = ComputeAlignment(12);

        _pendingFields.Add(new ConstantBufferField
        {
            Name = name,
            TypeName = "float3",
            Offset = _currentOffset,
            SizeBytes = 12,
            Alignment = alignment
        });

        _currentOffset += 12;
        return this;
    }

    /// <summary>Adds a float4 field to the layout.</summary>
    public ConstantBufferLayoutBuilder AddFloat4(string name)
    {
        int padding = ComputeFloat4Padding(_currentOffset);

        _pendingFields.Add(new ConstantBufferField
        {
            Name = name,
            TypeName = "float4",
            Offset = _currentOffset + padding,
            SizeBytes = 16,
            Alignment = ConstantBufferAlignment.Align16
        });

        _currentOffset += padding + 16;
        return this;
    }

    /// <summary>Adds an int field to the layout.</summary>
    public ConstantBufferLayoutBuilder AddInt(string name)
    {
        var alignment = ComputeAlignment(4);
        int padding = ComputePadding(_currentOffset, alignment);

        _pendingFields.Add(new ConstantBufferField
        {
            Name = name,
            TypeName = "int",
            Offset = _currentOffset + padding,
            SizeBytes = 4,
            Alignment = alignment
        });

        _currentOffset += padding + 4;
        return this;
    }

    /// <summary>Adds an int4 field to the layout.</summary>
    public ConstantBufferLayoutBuilder AddInt4(string name)
    {
        int padding = ComputeFloat4Padding(_currentOffset);

        _pendingFields.Add(new ConstantBufferField
        {
            Name = name,
            TypeName = "int4",
            Offset = _currentOffset + padding,
            SizeBytes = 16,
            Alignment = ConstantBufferAlignment.Align16
        });

        _currentOffset += padding + 16;
        return this;
    }

    /// <summary>Adds a float4x4 matrix field to the layout.</summary>
    public ConstantBufferLayoutBuilder AddMatrix4x4(string name)
    {
        int padding = ComputeFloat4Padding(_currentOffset);

        _pendingFields.Add(new ConstantBufferField
        {
            Name = name,
            TypeName = "float4x4",
            Offset = _currentOffset + padding,
            SizeBytes = 64,
            Alignment = ConstantBufferAlignment.Align16
        });

        _currentOffset += padding + 64;
        return this;
    }

    /// <summary>Adds a float4 array field to the layout.</summary>
    /// <param name="name">Field name.</param>
    /// <param name="elementCount">Number of float4 elements.</param>
    public ConstantBufferLayoutBuilder AddFloat4Array(string name, int elementCount)
    {
        int padding = ComputeFloat4Padding(_currentOffset);

        _pendingFields.Add(new ConstantBufferField
        {
            Name = name,
            TypeName = "float4",
            Offset = _currentOffset + padding,
            SizeBytes = 16,
            ArrayElementCount = elementCount,
            ArrayStride = 16,
            Alignment = ConstantBufferAlignment.Align16,
            IsPackedArray = true
        });

        _currentOffset += padding + 16 * elementCount;
        return this;
    }

    /// <summary>Adds a float array field to the layout.</summary>
    /// <param name="name">Field name.</param>
    /// <param name="elementCount">Number of float elements.</param>
    public ConstantBufferLayoutBuilder AddFloatArray(string name, int elementCount)
    {
        int padding = ComputeFloat4Padding(_currentOffset);

        // Float arrays are packed into float4 chunks for HLSL
        int float4Count = (elementCount + 3) / 4;

        _pendingFields.Add(new ConstantBufferField
        {
            Name = name,
            TypeName = "float",
            Offset = _currentOffset + padding,
            SizeBytes = 4,
            ArrayElementCount = elementCount,
            ArrayStride = 4,
            Alignment = ConstantBufferAlignment.Align16,
            IsPackedArray = false
        });

        _currentOffset += padding + 4 * elementCount;

        // Ensure final alignment to float4 boundary
        int finalPadding = ComputeFloat4Padding(_currentOffset);
        _currentOffset += finalPadding;

        return this;
    }

    /// <summary>Adds a raw byte range to the layout (padded to float4 boundary).</summary>
    /// <param name="name">Field name.</param>
    /// <param name="byteCount">Number of bytes.</param>
    public ConstantBufferLayoutBuilder AddRawBytes(string name, int byteCount)
    {
        int padding = ComputeFloat4Padding(_currentOffset);

        _pendingFields.Add(new ConstantBufferField
        {
            Name = name,
            TypeName = "float4",
            Offset = _currentOffset + padding,
            SizeBytes = 16,
            ArrayElementCount = (byteCount + 15) / 16,
            ArrayStride = 16,
            Alignment = ConstantBufferAlignment.Align16,
            IsPackedArray = true
        });

        int alignedBytes = (byteCount + 15) & ~15;
        _currentOffset += padding + alignedBytes;
        return this;
    }

    /// <summary>Ensures the current offset is aligned to a float4 boundary.</summary>
    public ConstantBufferLayoutBuilder AlignToFloat4()
    {
        int padding = ComputeFloat4Padding(_currentOffset);
        _currentOffset += padding;
        return this;
    }

    /// <summary>Builds the final constant buffer layout.</summary>
    public ConstantBufferLayout Build()
    {
        // Ensure final float4 alignment
        AlignToFloat4();

        var layout = new ConstantBufferLayout
        {
            Name = _bufferName,
            Register = _register,
            TotalSizeBytes = _currentOffset
        };

        layout.Fields.AddRange(_pendingFields);

        return layout;
    }

    /// <summary>Gets the current byte offset.</summary>
    public int CurrentOffset => _currentOffset;

    /// <summary>Gets the current offset as a float4 index.</summary>
    public int CurrentFloat4Index => _currentOffset / 16;

    /// <summary>
    /// Packs NeuralLayerWeights into float4 arrays for constant buffer upload.
    /// </summary>
    /// <param name="weights">The neural layer weights to pack.</param>
    /// <returns>Array of float4 values ready for GPU upload.</returns>
    public static float4[] PackNeuralLayerWeights(in NeuralLayerWeights weights)
    {
        const int floatsPerLayer = NeuralLayerWeights.TotalFloatCount; // 72
        const int float4PerLayer = (floatsPerLayer + 3) / 4; // 18

        var packed = new float4[float4PerLayer];
        ReadOnlySpan<float> weightSpan = weights.AsReadOnlySpan();

        for (int i = 0; i < float4PerLayer; i++)
        {
            int baseIdx = i * 4;
            float x = baseIdx < floatsPerLayer ? weightSpan[baseIdx] : 0f;
            float y = baseIdx + 1 < floatsPerLayer ? weightSpan[baseIdx + 1] : 0f;
            float z = baseIdx + 2 < floatsPerLayer ? weightSpan[baseIdx + 2] : 0f;
            float w = baseIdx + 3 < floatsPerLayer ? weightSpan[baseIdx + 3] : 0f;
            packed[i] = new float4(x, y, z, w);
        }

        return packed;
    }

    /// <summary>
    /// Packs all layers of a MicroMLP into float4 arrays.
    /// </summary>
    /// <param name="mlp">The MicroMLP network to pack.</param>
    /// <returns>Packed float4 arrays for each layer.</returns>
    public static float4[][] PackMicroMLP(in MicroMLP mlp)
    {
        var result = new float4[MicroMLP.LayerCount][];

        ReadOnlySpan<float> allWeights = mlp.AllWeights();
        const int floatsPerLayer = NeuralLayerWeights.TotalFloatCount;
        const int float4PerLayer = (floatsPerLayer + 3) / 4;

        for (int layer = 0; layer < MicroMLP.LayerCount; layer++)
        {
            result[layer] = new float4[float4PerLayer];
            int layerOffset = layer * floatsPerLayer;

            for (int i = 0; i < float4PerLayer; i++)
            {
                int baseIdx = layerOffset + i * 4;
                float x = baseIdx < allWeights.Length ? allWeights[baseIdx] : 0f;
                float y = baseIdx + 1 < allWeights.Length ? allWeights[baseIdx + 1] : 0f;
                float z = baseIdx + 2 < allWeights.Length ? allWeights[baseIdx + 2] : 0f;
                float w = baseIdx + 3 < allWeights.Length ? allWeights[baseIdx + 3] : 0f;
                result[layer][i] = new float4(x, y, z, w);
            }
        }

        return result;
    }

    /// <summary>
    /// Packs raw float weights into float4 arrays for constant buffer upload.
    /// </summary>
    /// <param name="weights">Raw weight floats.</param>
    /// <returns>Packed float4 array.</returns>
    public static float4[] PackFloatsToFloat4(ReadOnlySpan<float> weights)
    {
        int float4Count = (weights.Length + 3) / 4;
        var packed = new float4[float4Count];

        for (int i = 0; i < float4Count; i++)
        {
            int baseIdx = i * 4;
            float x = baseIdx < weights.Length ? weights[baseIdx] : 0f;
            float y = baseIdx + 1 < weights.Length ? weights[baseIdx + 1] : 0f;
            float z = baseIdx + 2 < weights.Length ? weights[baseIdx + 2] : 0f;
            float w = baseIdx + 3 < weights.Length ? weights[baseIdx + 3] : 0f;
            packed[i] = new float4(x, y, z, w);
        }

        return packed;
    }

    /// <summary>
    /// Computes the byte layout for a MicroMLP constant buffer.
    /// Includes all three layers with proper float4 alignment.
    /// </summary>
    /// <param name="layerCount">Number of hidden layers (default 3).</param>
    /// <param name="hiddenSize">Hidden layer size (default 8).</param>
    /// <returns>Constant buffer layout.</returns>
    public static ConstantBufferLayout ComputeMicroMLPLayout(int layerCount = 3, int hiddenSize = 8)
    {
        var builder = new ConstantBufferLayoutBuilder("NeuralWeights", "b0");

        // Layer 0: input -> hidden
        int layer0WeightFloats = 3 * hiddenSize;
        int layer0BiasFloats = hiddenSize;
        int layer0Float4Count = ((layer0WeightFloats + layer0BiasFloats) + 3) / 4;
        builder.AddFloat4Array($"Layer0_Weights", layer0Float4Count);

        // Hidden layers
        for (int i = 1; i < layerCount - 1; i++)
        {
            int hiddenWeightFloats = hiddenSize * hiddenSize;
            int hiddenBiasFloats = hiddenSize;
            int hiddenFloat4Count = ((hiddenWeightFloats + hiddenBiasFloats) + 3) / 4;
            builder.AddFloat4Array($"Layer{i}_Weights", hiddenFloat4Count);
        }

        // Output layer
        int outputWeightFloats = hiddenSize * 1;
        int outputBiasFloats = 1;
        int outputFloat4Count = ((outputWeightFloats + outputBiasFloats) + 3) / 4;
        builder.AddFloat4Array("Output_Weights", outputFloat4Count);

        return builder.Build();
    }

    /// <summary>
    /// Computes the scene parameter constant buffer layout.
    /// </summary>
    public static ConstantBufferLayout ComputeSceneParamsLayout()
    {
        var builder = new ConstantBufferLayoutBuilder("SceneParams", "b1");

        builder.AddMatrix4x4("ViewMatrix");
        builder.AddMatrix4x4("ProjectionMatrix");
        builder.AddMatrix4x4("ViewProjectionMatrix");
        builder.AddMatrix4x4("InvViewMatrix");
        builder.AddMatrix4x4("InvProjectionMatrix");
        builder.AddFloat4("CameraPosition");
        builder.AddFloat4("CameraForward");
        builder.AddFloat4("CameraRight");
        builder.AddFloat4("CameraUp");
        builder.AddFloat4("LightDirection");
        builder.AddFloat4("LightColor");
        builder.AddFloat4("ScreenParams");
        builder.AddFloat4("TraceParams");

        return builder.Build();
    }

    /// <summary>
    /// Serializes the layout to a binary format for shader compilation.
    /// Format: [FieldCount:int32][For each field: NameLength:int32, NameBytes:UTF8, Offset:int32, Size:int32, ArrayCount:int32, Alignment:int32]
    /// </summary>
    /// <param name="layout">The layout to serialize.</param>
    /// <returns>Binary data.</returns>
    public static byte[] SerializeLayout(ConstantBufferLayout layout)
    {
        using var ms = new System.IO.MemoryStream();
        using var writer = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

        // Header
        writer.Write(layout.Fields.Count);
        writer.Write(layout.TotalSizeBytes);

        // Name
        byte[] nameBytes = Encoding.UTF8.GetBytes(layout.Name);
        writer.Write(nameBytes.Length);
        writer.Write(nameBytes);

        // Register
        byte[] regBytes = Encoding.UTF8.GetBytes(layout.Register);
        writer.Write(regBytes.Length);
        writer.Write(regBytes);

        // Fields
        foreach (var field in layout.Fields)
        {
            byte[] fieldNameBytes = Encoding.UTF8.GetBytes(field.Name);
            writer.Write(fieldNameBytes.Length);
            writer.Write(fieldNameBytes);

            byte[] typeNameBytes = Encoding.UTF8.GetBytes(field.TypeName);
            writer.Write(typeNameBytes.Length);
            writer.Write(typeNameBytes);

            writer.Write(field.Offset);
            writer.Write(field.SizeBytes);
            writer.Write(field.ArrayElementCount);
            writer.Write(field.ArrayStride);
            writer.Write((int)field.Alignment);
            writer.Write(field.IsPackedArray);
        }

        writer.Flush();
        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes a layout from binary data.
    /// </summary>
    /// <param name="data">Binary layout data.</param>
    /// <returns>Deserialized layout.</returns>
    public static ConstantBufferLayout DeserializeLayout(ReadOnlySpan<byte> data)
    {
        using var ms = new System.IO.MemoryStream(data.ToArray());
        using var reader = new System.IO.BinaryReader(ms, Encoding.UTF8, leaveOpen: true);

        int fieldCount = reader.ReadInt32();
        int totalSize = reader.ReadInt32();

        int nameLen = reader.ReadInt32();
        string name = Encoding.UTF8.GetString(reader.ReadBytes(nameLen));

        int regLen = reader.ReadInt32();
        string register = Encoding.UTF8.GetString(reader.ReadBytes(regLen));

        var layout = new ConstantBufferLayout
        {
            Name = name,
            Register = register,
            TotalSizeBytes = totalSize
        };

        for (int i = 0; i < fieldCount; i++)
        {
            int fieldNameLen = reader.ReadInt32();
            string fieldName = Encoding.UTF8.GetString(reader.ReadBytes(fieldNameLen));

            int typeNameLen = reader.ReadInt32();
            string typeName = Encoding.UTF8.GetString(reader.ReadBytes(typeNameLen));

            int offset = reader.ReadInt32();
            int sizeBytes = reader.ReadInt32();
            int arrayCount = reader.ReadInt32();
            int arrayStride = reader.ReadInt32();
            var alignment = (ConstantBufferAlignment)reader.ReadInt32();
            bool isPackedArray = reader.ReadBoolean();

            layout.Fields.Add(new ConstantBufferField
            {
                Name = fieldName,
                TypeName = typeName,
                Offset = offset,
                SizeBytes = sizeBytes,
                ArrayElementCount = arrayCount,
                ArrayStride = arrayStride,
                Alignment = alignment,
                IsPackedArray = isPackedArray
            });
        }

        return layout;
    }

    /// <summary>
    /// Validates a layout against GPU constant buffer limits.
    /// </summary>
    /// <param name="layout">The layout to validate.</param>
    /// <returns>List of validation errors (empty if valid).</returns>
    public static List<string> ValidateLayout(ConstantBufferLayout layout)
    {
        var errors = new List<string>();

        if (layout.TotalSizeBytes > MaxConstantBufferSize)
        {
            errors.Add($"Constant buffer size {layout.TotalSizeBytes} exceeds maximum {MaxConstantBufferSize} bytes.");
        }

        if (layout.TotalFloat4Elements > MaxFloat4Elements)
        {
            errors.Add($"Float4 element count {layout.TotalFloat4Elements} exceeds maximum {MaxFloat4Elements}.");
        }

        int expectedOffset = 0;
        foreach (var field in layout.Fields)
        {
            if (field.Offset < expectedOffset)
            {
                errors.Add($"Field '{field.Name}' at offset {field.Offset} overlaps previous field ending at {expectedOffset}.");
            }

            if (field.Offset % (int)field.Alignment != 0)
            {
                errors.Add($"Field '{field.Name}' at offset {field.Offset} is not aligned to {field.Alignment} bytes.");
            }

            if (field.ArrayElementCount > 1)
            {
                if (field.ArrayStride % 16 != 0)
                {
                    errors.Add($"Array field '{field.Name}' stride {field.ArrayStride} is not a multiple of 16.");
                }
            }

            expectedOffset = field.Offset + field.TotalFootprint;
        }

        if (layout.TotalSizeBytes % HLSLAlignment != 0)
        {
            errors.Add($"Total buffer size {layout.TotalSizeBytes} is not aligned to {HLSLAlignment} bytes.");
        }

        return errors;
    }

    /// <summary>
    /// Generates the HLSL constant buffer declaration from a layout.
    /// </summary>
    /// <param name="layout">The buffer layout.</param>
    /// <returns>HLSL source code.</returns>
    public static string GenerateHLSL(ConstantBufferLayout layout)
    {
        return layout.ToHLSL();
    }

    /// <summary>
    /// Generates a C# struct that matches the constant buffer layout for unsafe upload.
    /// </summary>
    /// <param name="layout">The buffer layout.</param>
    /// <returns>C# struct declaration string.</returns>
    public static string GenerateCSharpStruct(ConstantBufferLayout layout)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"[StructLayout(LayoutKind.Sequential, Pack = 16)]");
        sb.AppendLine($"public struct {layout.Name}");
        sb.AppendLine("{");

        foreach (var field in layout.Fields)
        {
            string csType = field.TypeName switch
            {
                "float" => "float",
                "float2" => "Vector2",
                "float3" => "Vector3",
                "float4" => "Vector4",
                "float4x4" => "Matrix4x4",
                "int" => "int",
                "int2" => "Vector2Int",
                "int3" => "Vector3Int",
                "int4" => "Vector4Int",
                "uint" => "uint",
                "bool" => "int",
                _ => "float"
            };

            if (field.ArrayElementCount > 1)
            {
                sb.AppendLine($"    [FieldOffset({field.Offset})]");
                sb.AppendLine($"    public unsafe fixed {csType} {field.Name}[{field.ArrayElementCount}];");
            }
            else
            {
                sb.AppendLine($"    [FieldOffset({field.Offset})]");
                sb.AppendLine($"    public {csType} {field.Name};");
            }
        }

        sb.AppendLine();
        sb.AppendLine($"    public const int SizeInBytes = {layout.TotalSizeBytes};");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Computes a merged layout from multiple layouts (for bindless rendering).
    /// </summary>
    /// <param name="layouts">Layouts to merge.</param>
    /// <returns>Merged layout with all fields.</returns>
    public static ConstantBufferLayout MergeLayouts(params ConstantBufferLayout[] layouts)
    {
        var builder = new ConstantBufferLayoutBuilder("MergedBuffer", "b0");

        foreach (var layout in layouts)
        {
            foreach (var field in layout.Fields)
            {
                if (field.ArrayElementCount > 1)
                {
                    builder.AddFloat4Array(field.Name, field.ArrayElementCount);
                }
                else
                {
                    switch (field.TypeName)
                    {
                        case "float":
                            builder.AddFloat(field.Name);
                            break;
                        case "float2":
                            builder.AddFloat2(field.Name);
                            break;
                        case "float3":
                            builder.AddFloat3(field.Name);
                            break;
                        case "float4":
                            builder.AddFloat4(field.Name);
                            break;
                        case "float4x4":
                            builder.AddMatrix4x4(field.Name);
                            break;
                        case "int":
                            builder.AddInt(field.Name);
                            break;
                        case "int4":
                            builder.AddInt4(field.Name);
                            break;
                    }
                }
            }
        }

        return builder.Build();
    }

    /// <summary>
    /// Computes the offset map for weight data within a constant buffer.
    /// Used for efficient GPU weight upload.
    /// </summary>
    /// <param name="layerCount">Number of layers.</param>
    /// <param name="hiddenSize">Hidden layer size.</param>
    /// <returns>Dictionary mapping layer index to byte offset in constant buffer.</returns>
    public static Dictionary<int, int> ComputeWeightOffsets(int layerCount = 3, int hiddenSize = 8)
    {
        var offsets = new Dictionary<int, int>();
        int currentOffset = 0;

        for (int layer = 0; layer < layerCount; layer++)
        {
            offsets[layer] = currentOffset;

            int inputSize = layer == 0 ? 3 : hiddenSize;
            int outputSize = layer == layerCount - 1 ? 1 : hiddenSize;
            int totalFloats = inputSize * outputSize + outputSize;
            int float4Count = (totalFloats + 3) / 4;

            currentOffset += float4Count * 16;
        }

        return offsets;
    }
}

/// <summary>
/// Represents a 4-component float vector (HLSL float4 equivalent).
/// </summary>
public struct float4
{
    public float X;
    public float Y;
    public float Z;
    public float W;

    public float4(float x, float y, float z, float w)
    {
        X = x;
        Y = y;
        Z = z;
        W = w;
    }
}
