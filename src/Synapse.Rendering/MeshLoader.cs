// =============================================================================
// MeshLoader.cs - G-DNN Engine: 3D Model Loading System
// GDNN.Engine - GDNN.Rendering.MeshIO
// Complete mesh import pipeline supporting glTF, FBX, and OBJ formats
// =============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Rendering.MeshIO
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    /// <summary>Supported 3D model file formats.</summary>
    public enum MeshFileType
    {
        Unknown,
        glTF,
        glTFBinary,
        OBJ,
        FBX,
        STL,
        PLY
    }

    /// <summary>Vertex attribute flags for mesh data.</summary>
    [Flags]
    public enum VertexAttribute : uint
    {
        None = 0,
        Position = 1 << 0,
        Normal = 1 << 1,
        Tangent = 1 << 2,
        Bitangent = 1 << 3,
        Color0 = 1 << 4,
        Color1 = 1 << 5,
        TexCoord0 = 1 << 6,
        TexCoord1 = 1 << 7,
        TexCoord2 = 1 << 8,
        TexCoord3 = 1 << 9,
        BoneWeights = 1 << 10,
        BoneIndices = 1 << 11,
        InstanceTransform = 1 << 12,
        All = 0x1FFF
    }

    /// <summary>Primitive topology types.</summary>
    public enum PrimitiveTopology
    {
        TriangleList,
        TriangleStrip,
        TriangleFan,
        PointList,
        LineList,
        LineStrip
    }

    /// <summary>Mesh processing flags.</summary>
    [Flags]
    public enum MeshProcessFlags : uint
    {
        None = 0,
        CalculateNormals = 1 << 0,
        CalculateTangents = 1 << 1,
        FlipUVs = 1 << 2,
        Triangulate = 1 << 3,
        GenerateMipmaps = 1 << 4,
        OptimizeVertexCache = 1 << 5,
        OptimizeOverdraw = 1 << 6,
        OptimizeVertexFetch = 1 << 7,
        FlipWindingOrder = 1 << 8,
        MakeLeftHanded = 1 << 9,
        PreTransformVertices = 1 << 10,
        SortByPrimitiveType = 1 << 11,
        JoinIdenticalVertices = 1 << 12,
        SplitLargeMeshes = 1 << 13,
        Debone = 1 << 14
    }

    // =========================================================================
    // DATA STRUCTURES
    // =========================================================================

    /// <summary>Represents a vertex with all possible attributes.</summary>
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct MeshVertex
    {
        public Vector3 Position;
        public Vector3 Normal;
        public Vector4 Tangent;
        public Vector4 Bitangent;
        public Vector4 Color0;
        public Vector4 Color1;
        public Vector2 TexCoord0;
        public Vector2 TexCoord1;
        public Vector2 TexCoord2;
        public Vector2 TexCoord3;
        public Vector4 BoneWeights;
        public Vector4i BoneIndices;

        public static readonly uint SizeInBytes = (uint)(sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 4 * 7 + sizeof(float) * 2 * 4 + sizeof(int) * 4);
    }

    /// <summary>Integer 4-component vector for bone indices.</summary>
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct Vector4i
    {
        public int X, Y, Z, W;
        public Vector4i(int x, int y, int z, int w) { X = x; Y = y; Z = z; W = w; }
    }

    /// <summary>Represents a single triangle face with vertex indices.</summary>
    public struct TriangleFace
    {
        public int V0, V1, V2;
        public int MaterialIndex;
        public TriangleFace(int v0, int v1, int v2, int materialIndex = 0)
        { V0 = v0; V1 = v1; V2 = v2; MaterialIndex = materialIndex; }
    }

    /// <summary>Represents a mesh primitive (submesh).</summary>
    public class MeshPrimitive
    {
        public string Name { get; set; } = "";
        public List<MeshVertex> Vertices { get; set; } = new();
        public List<uint> Indices { get; set; } = new();
        public int MaterialIndex { get; set; }
        public PrimitiveTopology Topology { get; set; } = PrimitiveTopology.TriangleList;
        public VertexAttribute ActiveAttributes { get; set; }
        public BoundingBox3D Bounds { get; set; }
    }

    /// <summary>Axis-aligned bounding box.</summary>
    public struct BoundingBox3D
    {
        public Vector3 Min;
        public Vector3 Max;
        public BoundingBox3D(Vector3 min, Vector3 max) { Min = min; Max = max; }
        public Vector3 Center => (Min + Max) * 0.5f;
        public Vector3 Size => Max - Min;
        public Vector3 Extents => Size * 0.5f;
        public static BoundingBox3D Invalid => new(Vector3.One * float.MaxValue, Vector3.One * float.MinValue);
        public void Encapsulate(Vector3 point) { Min = Vector3.Min(Min, point); Max = Vector3.Max(Max, point); }
        public void Encapsulate(BoundingBox3D other) { Min = Vector3.Min(Min, other.Min); Max = Vector3.Max(Max, other.Max); }
        public bool Intersects(BoundingBox3D other) => Min.X <= other.Max.X && Max.X >= other.Min.X && Min.Y <= other.Max.Y && Max.Y >= other.Min.Y && Min.Z <= other.Max.Z && Max.Z >= other.Min.Z;
    }

    /// <summary>Complete mesh asset with multiple primitives and materials.</summary>
    public class MeshAsset
    {
        public string Name { get; set; } = "";
        public string SourcePath { get; set; } = "";
        public MeshFileType SourceFormat { get; set; }
        public List<MeshPrimitive> Primitives { get; set; } = new();
        public List<MeshMaterial> Materials { get; set; } = new();
        public MeshSkeleton? Skeleton { get; set; }
        public List<MeshAnimationClip> AnimationClips { get; set; } = new();
        public BoundingBox3D Bounds { get; set; }
        public int TotalVertexCount => Primitives.Sum(p => p.Vertices.Count);
        public int TotalIndexCount => Primitives.Sum(p => p.Indices.Count);
        public int TotalTriangleCount => Primitives.Sum(p => p.Indices.Count / 3);
        public VertexAttribute GlobalAttributes { get; set; }

        /// <summary>Creates a simple axis-aligned cube mesh for tests and placeholder exports.</summary>
        public static MeshAsset CreateUnitCube(float size = 1f, string name = "Cube")
        {
            float h = size * 0.5f;
            var positions = new[]
            {
                new Vector3(-h, -h, -h), new Vector3(h, -h, -h), new Vector3(h, h, -h), new Vector3(-h, h, -h),
                new Vector3(-h, -h, h), new Vector3(h, -h, h), new Vector3(h, h, h), new Vector3(-h, h, h)
            };

            var asset = new MeshAsset
            {
                Name = name,
                SourceFormat = MeshFileType.Unknown,
                Materials = { new MeshMaterial { Name = "Default", BaseColor = Vector3.One * 0.8f } }
            };

            var primitive = new MeshPrimitive
            {
                Name = name,
                MaterialIndex = 0,
                ActiveAttributes = VertexAttribute.Position | VertexAttribute.Normal
            };

            foreach (var position in positions)
            {
                primitive.Vertices.Add(new MeshVertex
                {
                    Position = position,
                    Normal = Vector3.Normalize(position)
                });
            }

            int[] faces =
            {
                0, 1, 2, 0, 2, 3,
                4, 6, 5, 4, 7, 6,
                0, 4, 5, 0, 5, 1,
                2, 6, 7, 2, 7, 3,
                0, 3, 7, 0, 7, 4,
                1, 5, 6, 1, 6, 2
            };

            foreach (int index in faces)
                primitive.Indices.Add((uint)index);

            primitive.Bounds = new BoundingBox3D(new Vector3(-h), new Vector3(h));
            asset.Primitives.Add(primitive);
            asset.Bounds = primitive.Bounds;
            asset.GlobalAttributes = primitive.ActiveAttributes;
            return asset;
        }
    }

    /// <summary>Material definition for mesh loading.</summary>
    public class MeshMaterial
    {
        public string Name { get; set; } = "";
        public Vector3 BaseColor { get; set; } = Vector3.One;
        public float Metallic { get; set; }
        public float Roughness { get; set; } = 0.5f;
        public Vector3 EmissiveColor { get; set; }
        public float EmissiveIntensity { get; set; }
        public float Opacity { get; set; } = 1.0f;
        public float AlphaCutoff { get; set; }
        public bool DoubleSided { get; set; }
        public float Clearcoat { get; set; }
        public float ClearcoatRoughness { get; set; }
        public float Ior { get; set; } = 1.5f;
        public float Occlusion { get; set; } = 1.0f;
        public string AlbedoTexturePath { get; set; } = "";
        public string NormalTexturePath { get; set; } = "";
        public string MetallicRoughnessTexturePath { get; set; } = "";
        public string EmissiveTexturePath { get; set; } = "";
        public string AOTexturePath { get; set; } = "";
        public string HeightTexturePath { get; set; } = "";
        public string ClearcoatTexturePath { get; set; } = "";
        public string SpecularTexturePath { get; set; } = "";
        public string OpacityTexturePath { get; set; } = "";
    }

    /// <summary>Result of a mesh loading operation.</summary>
    public class MeshLoadResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; } = "";
        public MeshAsset? Asset { get; set; }
        public TimeSpan LoadTime { get; set; }
        public int WarningsCount { get; set; }
        public List<string> Warnings { get; set; } = new();
    }

    /// <summary>Configuration for mesh loading.</summary>
    public class MeshLoadConfig
    {
        public MeshProcessFlags ProcessFlags { get; set; } = MeshProcessFlags.CalculateNormals | MeshProcessFlags.CalculateTangents | MeshProcessFlags.Triangulate;
        public float ScaleFactor { get; set; } = 1.0f;
        public bool FlipUVs { get; set; }
        public bool MakeLeftHanded { get; set; }
        public int MaxBoneWeightsPerVertex { get; set; } = 4;
        public int MaxBones { get; set; } = 256;
        public float ImportDepth { get; set; } = -1;
        /// <summary>Active USD variantSelections: variantSet name → variant name.</summary>
        public Dictionary<string, string> UsdVariantSelections { get; set; } = new(StringComparer.Ordinal);
    }

    /// <summary>Configuration for mesh export.</summary>
    public class MeshExportConfig
    {
        public bool ExportMaterials { get; init; } = true;
        public bool EmbedBinaryData { get; init; }
    }

    /// <summary>Result of a mesh export operation.</summary>
    public class MeshExportResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; } = "";
        public TimeSpan ExportTime { get; set; }
        public string OutputPath { get; set; } = "";
    }

    // =========================================================================
    // glTF LOADER
    // =========================================================================

    /// <summary>
    /// Loads glTF 2.0 (.gltf) and glTF Binary (.glb) files.
    /// Supports meshes, materials, textures, animations, and scene hierarchy.
    /// </summary>
    public class GlTFLoader
    {
        public async Task<MeshLoadResult> LoadAsync(string filePath, MeshLoadConfig? config = null, CancellationToken ct = default)
        {
            config ??= new MeshLoadConfig();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = new MeshLoadResult { Asset = new MeshAsset { SourcePath = filePath } };

            try
            {
                if (!File.Exists(filePath))
                {
                    result.ErrorMessage = $"File not found: {filePath}";
                    return result;
                }

                var ext = Path.GetExtension(filePath).ToLowerInvariant();
                result.Asset.SourceFormat = ext == ".glb" ? MeshFileType.glTFBinary : MeshFileType.glTF;

                byte[] fileData = await File.ReadAllBytesAsync(filePath, ct);
                GlTFRoot? root;

                if (ext == ".glb")
                {
                    root = ParseGLB(fileData);
                }
                else
                {
                    root = JsonSerializer.Deserialize<GlTFRoot>(Encoding.UTF8.GetString(fileData));
                }

                if (root == null)
                {
                    result.ErrorMessage = "Failed to parse glTF file";
                    return result;
                }

                result.Asset.Name = root.Asset?.Generator ?? Path.GetFileNameWithoutExtension(filePath);
                byte[] bufferData = root.Buffers != null && root.Buffers.Length > 0
                    ? await LoadBufferData(filePath, root, ct)
                    : Array.Empty<byte>();

                if (root.Meshes != null)
                {
                    foreach (var gltfMesh in root.Meshes)
                    {
                        var primitive = new MeshPrimitive { Name = gltfMesh.Name ?? "" };
                        primitive.ActiveAttributes = ParseAttributes(gltfMesh);

                        if (gltfMesh.Primitives != null)
                        {
                            foreach (var gltfPrim in gltfMesh.Primitives)
                            {
                                ReadPrimitive(gltfPrim, root, bufferData, primitive, config);
                            }
                        }

                        ComputeBounds(primitive);
                        result.Asset.Primitives.Add(primitive);
                    }
                }

                if (root.Materials != null)
                {
                    foreach (var mat in root.Materials)
                    {
                        result.Asset.Materials.Add(ConvertMaterial(mat));
                    }
                }

                ComputeAssetBounds(result.Asset);
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = $"glTF load error: {ex.Message}";
            }

            sw.Stop();
            result.LoadTime = sw.Elapsed;
            return result;
        }

        private GlTFRoot? ParseGLB(byte[] data)
        {
            if (data.Length < 12)
                return null;
            uint magic = BitConverter.ToUInt32(data, 0);
            if (magic != 0x46546C67)
                return null;

            uint version = BitConverter.ToUInt32(data, 4);
            if (version != 2)
                return null;

            uint totalLength = BitConverter.ToUInt32(data, 8);
            int offset = 12;
            byte[]? jsonChunk = null;
            byte[]? binChunk = null;

            while (offset < totalLength)
            {
                if (offset + 8 > data.Length)
                    break;
                uint chunkLength = BitConverter.ToUInt32(data, offset);
                uint chunkType = BitConverter.ToUInt32(data, offset + 4);
                offset += 8;

                if (chunkType == 0x4E4F534A) // JSON
                    jsonChunk = new byte[chunkLength];
                else if (chunkType == 0x004E4942) // BIN
                    binChunk = new byte[chunkLength];

                if (jsonChunk != null && chunkType == 0x4E4F534A)
                    Buffer.BlockCopy(data, offset, jsonChunk, 0, (int)chunkLength);
                else if (binChunk != null && chunkType == 0x004E4942)
                    Buffer.BlockCopy(data, offset, binChunk, 0, (int)chunkLength);

                offset += (int)chunkLength;
            }

            if (jsonChunk == null)
                return null;
            var root = JsonSerializer.Deserialize<GlTFRoot>(Encoding.UTF8.GetString(jsonChunk));

            if (root != null && binChunk != null && root.Buffers is { Length: > 0 })
            {
                root.Buffers[0].BinaryData = binChunk;
            }

            return root;
        }

        private async Task<byte[]> LoadBufferData(string basePath, GlTFRoot root, CancellationToken ct)
        {
            if (root.Buffers == null || root.Buffers.Length == 0)
                return Array.Empty<byte>();

            var buffer = root.Buffers[0];
            if (buffer.BinaryData != null)
                return buffer.BinaryData;

            if (!string.IsNullOrEmpty(buffer.Uri))
            {
                if (buffer.Uri.StartsWith("data:"))
                {
                    int commaIdx = buffer.Uri.IndexOf(',');
                    if (commaIdx > 0)
                    {
                        string header = buffer.Uri.Substring(0, commaIdx);
                        string data = buffer.Uri.Substring(commaIdx + 1);
                        if (header.Contains("base64"))
                            return Convert.FromBase64String(data);
                    }
                }

                var meshRoot = Path.GetDirectoryName(basePath) ?? "";
                if (buffer.Uri.Contains("..", StringComparison.Ordinal) || Path.IsPathRooted(buffer.Uri))
                    throw new UnauthorizedAccessException("glTF buffer URI escapes asset directory.");
                string bufferPath = Synapse.Core.Security.PathSecurity.EnsureUnderRoot(
                    meshRoot, Path.Combine(meshRoot, buffer.Uri));
                var info = new FileInfo(bufferPath);
                if (info.Length > 256L * 1024 * 1024)
                    throw new InvalidDataException("glTF buffer exceeds size limit.");
                return await File.ReadAllBytesAsync(bufferPath, ct);
            }

            return Array.Empty<byte>();
        }

        private VertexAttribute ParseAttributes(GlTFMesh mesh)
        {
            var attrs = VertexAttribute.None;
            if (mesh.Primitives == null || mesh.Primitives.Length == 0)
                return attrs;

            var prim = mesh.Primitives[0];
            if (prim.Attributes != null)
            {
                foreach (var key in prim.Attributes.Keys)
                {
                    attrs |= key switch
                    {
                        "POSITION" => VertexAttribute.Position,
                        "NORMAL" => VertexAttribute.Normal,
                        "TANGENT" => VertexAttribute.Tangent,
                        "TEXCOORD_0" => VertexAttribute.TexCoord0,
                        "TEXCOORD_1" => VertexAttribute.TexCoord1,
                        "TEXCOORD_2" => VertexAttribute.TexCoord2,
                        "TEXCOORD_3" => VertexAttribute.TexCoord3,
                        "COLOR_0" => VertexAttribute.Color0,
                        "COLOR_1" => VertexAttribute.Color1,
                        "JOINTS_0" => VertexAttribute.BoneIndices,
                        "WEIGHTS_0" => VertexAttribute.BoneWeights,
                        _ => VertexAttribute.None
                    };
                }
            }

            return attrs;
        }

        private void ReadPrimitive(GlTFPrimitive prim, GlTFRoot root, byte[] bufferData, MeshPrimitive primitive, MeshLoadConfig config)
        {
            if (prim.Attributes == null)
                return;

            List<Vector3> positions = new();
            List<Vector3> normals = new();
            List<Vector4> tangents = new();
            List<Vector2> texCoords0 = new();
            List<Vector2> texCoords1 = new();
            List<Vector4> colors0 = new();

            if (prim.Attributes.TryGetValue("POSITION", out int posIdx))
                ReadAccessor(root, bufferData, posIdx, positions);

            if (prim.Attributes.TryGetValue("NORMAL", out int nrmIdx))
                ReadAccessor(root, bufferData, nrmIdx, normals);

            if (prim.Attributes.TryGetValue("TANGENT", out int tanIdx))
                ReadAccessor(root, bufferData, tanIdx, tangents);

            if (prim.Attributes.TryGetValue("TEXCOORD_0", out int uv0Idx))
                ReadAccessor(root, bufferData, uv0Idx, texCoords0);

            if (prim.Attributes.TryGetValue("TEXCOORD_1", out int uv1Idx))
                ReadAccessor(root, bufferData, uv1Idx, texCoords1);

            if (prim.Attributes.TryGetValue("COLOR_0", out int colIdx))
                ReadAccessor(root, bufferData, colIdx, colors0);

            if ((config.ProcessFlags & MeshProcessFlags.FlipUVs) != 0)
            {
                for (int i = 0; i < texCoords0.Count; i++)
                    texCoords0[i] = new Vector2(texCoords0[i].X, 1.0f - texCoords0[i].Y);
            }

            int vertexCount = positions.Count;
            for (int i = 0; i < vertexCount; i++)
            {
                var v = new MeshVertex
                {
                    Position = positions[i] * config.ScaleFactor,
                    Normal = i < normals.Count ? normals[i] : Vector3.UnitY,
                    Tangent = i < tangents.Count ? tangents[i] : new Vector4(1, 0, 0, 1),
                    TexCoord0 = i < texCoords0.Count ? texCoords0[i] : Vector2.Zero,
                    TexCoord1 = i < texCoords1.Count ? texCoords1[i] : Vector2.Zero,
                    Color0 = i < colors0.Count ? colors0[i] : Vector4.One,
                    BoneWeights = Vector4.UnitX,
                    BoneIndices = new Vector4i(0, 0, 0, 0)
                };
                primitive.Vertices.Add(v);
            }

            if (prim.Indices.HasValue)
            {
                List<uint> indices = new();
                ReadAccessorIndices(root, bufferData, prim.Indices.Value, indices);
                primitive.Indices.AddRange(indices);
            }
            else
            {
                for (uint i = 0; i < vertexCount; i++)
                    primitive.Indices.Add(i);
            }

            primitive.MaterialIndex = prim.Material ?? 0;
        }

        private void ReadAccessor<T>(GlTFRoot root, byte[] bufferData, int accessorIndex, List<T> output) where T : struct
        {
            if (root.Accessors == null || accessorIndex >= root.Accessors.Length)
                return;
            var accessor = root.Accessors[accessorIndex];
            if (root.BufferViews == null || accessor.BufferView >= root.BufferViews.Length)
                return;

            var bufferView = root.BufferViews[accessor.BufferView];
            int byteOffset = accessor.ByteOffset + (bufferView.ByteOffset ?? 0);
            int componentSize = GetComponentSize(accessor.ComponentType);
            int elementSize = componentSize * GetDimension(accessor.Type);
            int count = accessor.Count;
            byte[] data = bufferView.Buffer == 0 ? bufferData : Array.Empty<byte>();

            if (data.Length == 0)
                return;

            Type t = typeof(T);
            if (t == typeof(Vector3))
            {
                for (int i = 0; i < count; i++)
                {
                    int off = byteOffset + i * elementSize;
                    if (off + 12 > data.Length)
                        break;
                    float x = BitConverter.ToSingle(data, off);
                    float y = BitConverter.ToSingle(data, off + 4);
                    float z = BitConverter.ToSingle(data, off + 8);
                    output.Add((T)(object)new Vector3(x, y, z));
                }
            }
            else if (t == typeof(Vector4))
            {
                for (int i = 0; i < count; i++)
                {
                    int off = byteOffset + i * elementSize;
                    if (off + 16 > data.Length)
                        break;
                    float x = BitConverter.ToSingle(data, off);
                    float y = BitConverter.ToSingle(data, off + 4);
                    float z = BitConverter.ToSingle(data, off + 8);
                    float w = accessor.Type == "VEC3" ? 1.0f : BitConverter.ToSingle(data, off + 12);
                    output.Add((T)(object)new Vector4(x, y, z, w));
                }
            }
            else if (t == typeof(Vector2))
            {
                for (int i = 0; i < count; i++)
                {
                    int off = byteOffset + i * elementSize;
                    if (off + 8 > data.Length)
                        break;
                    float x = BitConverter.ToSingle(data, off);
                    float y = BitConverter.ToSingle(data, off + 4);
                    output.Add((T)(object)new Vector2(x, y));
                }
            }
        }

        private void ReadAccessorIndices(GlTFRoot root, byte[] bufferData, int accessorIndex, List<uint> output)
        {
            if (root.Accessors == null || accessorIndex >= root.Accessors.Length)
                return;
            var accessor = root.Accessors[accessorIndex];
            if (root.BufferViews == null || accessor.BufferView >= root.BufferViews.Length)
                return;

            var bufferView = root.BufferViews[accessor.BufferView];
            int byteOffset = accessor.ByteOffset + (bufferView.ByteOffset ?? 0);
            int componentSize = GetComponentSize(accessor.ComponentType);
            int count = accessor.Count;
            byte[] data = bufferView.Buffer == 0 ? bufferData : Array.Empty<byte>();

            if (data.Length == 0)
                return;

            for (int i = 0; i < count; i++)
            {
                int off = byteOffset + i * componentSize;
                if (off + componentSize > data.Length)
                    break;

                uint idx = accessor.ComponentType switch
                {
                    5120 => (uint)(sbyte)data[off],
                    5121 => data[off],
                    5122 => (uint)BitConverter.ToInt16(data, off),
                    5123 => BitConverter.ToUInt16(data, off),
                    5125 => BitConverter.ToUInt32(data, off),
                    _ => BitConverter.ToUInt32(data, off)
                };
                output.Add(idx);
            }
        }

        private int GetComponentSize(int componentType) => componentType switch
        {
            5120 => 1,
            5121 => 1,
            5122 => 2,
            5123 => 2,
            5125 => 4,
            5126 => 4,
            _ => 4
        };

        private int GetDimension(string type) => type switch
        {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT2" => 4,
            "MAT3" => 9,
            "MAT4" => 16,
            _ => 1
        };

        private void ComputeBounds(MeshPrimitive prim)
        {
            prim.Bounds = BoundingBox3D.Invalid;
            foreach (var v in prim.Vertices)
                prim.Bounds.Encapsulate(v.Position);
        }

        private void ComputeAssetBounds(MeshAsset asset)
        {
            asset.Bounds = BoundingBox3D.Invalid;
            foreach (var prim in asset.Primitives)
                asset.Bounds.Encapsulate(prim.Bounds);
        }

        private MeshMaterial ConvertMaterial(GlTFMaterial mat)
        {
            var m = new MeshMaterial { Name = mat.Name ?? "" };
            if (mat.PbrMetallicRoughness != null)
            {
                if (mat.PbrMetallicRoughness.BaseColorFactor != null && mat.PbrMetallicRoughness.BaseColorFactor.Length >= 3)
                    m.BaseColor = new Vector3(mat.PbrMetallicRoughness.BaseColorFactor[0], mat.PbrMetallicRoughness.BaseColorFactor[1], mat.PbrMetallicRoughness.BaseColorFactor[2]);
                if (mat.PbrMetallicRoughness.BaseColorFactor != null && mat.PbrMetallicRoughness.BaseColorFactor.Length >= 4)
                    m.Opacity = mat.PbrMetallicRoughness.BaseColorFactor[3];
                m.Metallic = mat.PbrMetallicRoughness.MetallicFactor;
                m.Roughness = mat.PbrMetallicRoughness.RoughnessFactor;
            }
            if (mat.EmissiveFactor != null && mat.EmissiveFactor.Length >= 3)
                m.EmissiveColor = new Vector3(mat.EmissiveFactor[0], mat.EmissiveFactor[1], mat.EmissiveFactor[2]);
            m.DoubleSided = mat.DoubleSided;
            return m;
        }
    }

    // =========================================================================
    // OBJ LOADER
    // =========================================================================

    /// <summary>
    /// Loads Wavefront OBJ files with MTL material support.
    /// Supports vertices, normals, texture coordinates, faces, and groups.
    /// </summary>
    public class ObjLoader
    {
        public async Task<MeshLoadResult> LoadAsync(string filePath, MeshLoadConfig? config = null, CancellationToken ct = default)
        {
            config ??= new MeshLoadConfig();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = new MeshLoadResult { Asset = new MeshAsset { SourcePath = filePath, SourceFormat = MeshFileType.OBJ } };

            try
            {
                if (!File.Exists(filePath))
                {
                    result.ErrorMessage = $"File not found: {filePath}";
                    return result;
                }

                var lines = await File.ReadAllLinesAsync(filePath, ct);
                string basePath = Path.GetDirectoryName(filePath) ?? "";
                List<Vector3> rawPositions = new();
                List<Vector3> rawNormals = new();
                List<Vector2> rawTexCoords = new();
                List<ObjFace> rawFaces = new();
                Dictionary<string, int> materialMap = new();
                List<MeshMaterial> materials = new();
                string currentMaterial = "";
                string currentGroup = "";
                int materialIndex = 0;

                foreach (var line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;
                    var parts = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 0)
                        continue;

                    switch (parts[0].ToLowerInvariant())
                    {
                        case "mtllib":
                            if (parts.Length > 1)
                            {
                                var mtlName = parts[1];
                                if (Path.IsPathRooted(mtlName) || mtlName.Contains("..", StringComparison.Ordinal))
                                    break;
                                string mtlPath = Synapse.Core.Security.PathSecurity.EnsureUnderRoot(
                                    basePath, Path.Combine(basePath, mtlName));
                                if (File.Exists(mtlPath))
                                {
                                    var mtlResult = await LoadMTLAsync(mtlPath, ct);
                                    materials.AddRange(mtlResult.materials);
                                    foreach (var kv in mtlResult.nameToIndex)
                                        materialMap[kv.Key] = kv.Value + materialIndex;
                                    materialIndex += mtlResult.materials.Count;
                                }
                            }
                            break;

                        case "usemtl":
                            if (parts.Length > 1)
                            {
                                currentMaterial = parts[1];
                                if (!materialMap.ContainsKey(currentMaterial))
                                {
                                    materialMap[currentMaterial] = materials.Count;
                                    materials.Add(new MeshMaterial { Name = currentMaterial });
                                }
                                materialIndex = materialMap[currentMaterial];
                            }
                            break;

                        case "v":
                            if (parts.Length >= 4 && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float vx) &&
                                float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float vy) &&
                                float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float vz))
                            {
                                rawPositions.Add(new Vector3(vx, vy, vz) * config.ScaleFactor);
                            }
                            break;

                        case "vn":
                            if (parts.Length >= 4 && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float vnx) &&
                                float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float vny) &&
                                float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float vnz))
                            {
                                rawNormals.Add(Vector3.Normalize(new Vector3(vnx, vny, vnz)));
                            }
                            break;

                        case "vt":
                            if (parts.Length >= 3 && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float u) &&
                                float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float v))
                            {
                                rawTexCoords.Add(new Vector2(u, (config.ProcessFlags & MeshProcessFlags.FlipUVs) != 0 ? 1.0f - v : v));
                            }
                            break;

                        case "f":
                            if (parts.Length >= 4)
                                ParseObjFace(parts, rawFaces, materialIndex);
                            break;

                        case "g":
                            currentGroup = parts.Length > 1 ? parts[1] : "";
                            break;
                    }
                }

                var vertMap = new Dictionary<ObjVertexKey, int>();
                var vertices = new List<MeshVertex>();
                var indices = new List<uint>();

                foreach (var face in rawFaces)
                {
                    var faceIndices = new[] { face.V0, face.V1, face.V2 };
                    foreach (var fv in faceIndices)
                    {
                        var key = new ObjVertexKey(fv.Position, fv.Normal, fv.TexCoord);
                        if (!vertMap.TryGetValue(key, out int idx))
                        {
                            idx = vertices.Count;
                            vertMap[key] = idx;
                            var vert = new MeshVertex
                            {
                                Position = fv.Position < rawPositions.Count ? rawPositions[fv.Position] : Vector3.Zero,
                                Normal = fv.Normal >= 0 && fv.Normal < rawNormals.Count ? rawNormals[fv.Normal] : Vector3.UnitY,
                                TexCoord0 = fv.TexCoord >= 0 && fv.TexCoord < rawTexCoords.Count ? rawTexCoords[fv.TexCoord] : Vector2.Zero,
                                Color0 = Vector4.One,
                                BoneWeights = Vector4.UnitX,
                                BoneIndices = new Vector4i(0, 0, 0, 0)
                            };
                            vertices.Add(vert);
                        }
                        indices.Add((uint)idx);
                    }
                }

                var primitive = new MeshPrimitive
                {
                    Name = currentGroup,
                    Vertices = vertices,
                    Indices = indices,
                    MaterialIndex = materialMap.ContainsKey(currentMaterial) ? materialMap[currentMaterial] : 0,
                    ActiveAttributes = VertexAttribute.Position | VertexAttribute.Normal | VertexAttribute.TexCoord0
                };

                ComputeBounds(primitive);
                result.Asset.Primitives.Add(primitive);
                result.Asset.Materials = materials;
                ComputeAssetBounds(result.Asset);
                result.Success = true;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = $"OBJ load error: {ex.Message}";
            }

            sw.Stop();
            result.LoadTime = sw.Elapsed;
            return result;
        }

        private void ParseObjFace(string[] parts, List<ObjFace> faces, int materialIndex)
        {
            var faceVerts = new List<ObjVertexRef>();
            for (int i = 1; i < parts.Length; i++)
            {
                var indices = parts[i].Split('/');
                int pos = int.Parse(indices[0], CultureInfo.InvariantCulture) - 1;
                int tex = indices.Length > 1 && !string.IsNullOrEmpty(indices[1]) ? int.Parse(indices[1], CultureInfo.InvariantCulture) - 1 : -1;
                int nrm = indices.Length > 2 && !string.IsNullOrEmpty(indices[2]) ? int.Parse(indices[2], CultureInfo.InvariantCulture) - 1 : -1;
                faceVerts.Add(new ObjVertexRef(pos, nrm, tex));
            }

            for (int i = 1; i < faceVerts.Count - 1; i++)
            {
                faces.Add(new ObjFace(faceVerts[0], faceVerts[i], faceVerts[i + 1], materialIndex));
            }
        }

        private void ComputeBounds(MeshPrimitive prim)
        {
            prim.Bounds = BoundingBox3D.Invalid;
            foreach (var v in prim.Vertices)
                prim.Bounds.Encapsulate(v.Position);
        }

        private void ComputeAssetBounds(MeshAsset asset)
        {
            asset.Bounds = BoundingBox3D.Invalid;
            foreach (var prim in asset.Primitives)
                asset.Bounds.Encapsulate(prim.Bounds);
        }

        private async Task<(List<MeshMaterial> materials, Dictionary<string, int> nameToIndex)> LoadMTLAsync(string mtlPath, CancellationToken ct)
        {
            var materials = new List<MeshMaterial>();
            var nameToIndex = new Dictionary<string, int>();
            var lines = await File.ReadAllLinesAsync(mtlPath, ct);
            MeshMaterial? current = null;

            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                var parts = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 0)
                    continue;

                switch (parts[0].ToLowerInvariant())
                {
                    case "newmtl":
                        current = new MeshMaterial { Name = parts.Length > 1 ? parts[1] : "" };
                        nameToIndex[current.Name] = materials.Count;
                        materials.Add(current);
                        break;
                    case "kd":
                        if (current != null && parts.Length >= 4 &&
                            float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float dr) &&
                            float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float dg) &&
                            float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float db))
                            current.BaseColor = new Vector3(dr, dg, db);
                        break;
                    case "ka":
                        if (current != null && parts.Length >= 4 &&
                            float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float ar) &&
                            float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float ag) &&
                            float.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float ab))
                            current.EmissiveColor = new Vector3(ar, ag, ab);
                        break;
                    case "ns":
                        if (current != null && parts.Length >= 2 &&
                            float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float ns))
                            current.Roughness = 1.0f - Math.Clamp(ns / 1000.0f, 0, 1);
                        break;
                    case "d":
                        if (current != null && parts.Length >= 2 &&
                            float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out float d))
                            current.Opacity = d;
                        break;
                    case "map_kd":
                        if (current != null && parts.Length >= 2)
                            current.AlbedoTexturePath = parts[1];
                        break;
                    case "map_bump":
                    case "bump":
                        if (current != null && parts.Length >= 2)
                            current.NormalTexturePath = parts[1];
                        break;
                    case "map_ns":
                        if (current != null && parts.Length >= 2)
                            current.MetallicRoughnessTexturePath = parts[1];
                        break;
                    case "map_ka":
                        if (current != null && parts.Length >= 2)
                            current.EmissiveTexturePath = parts[1];
                        break;
                }
            }

            return (materials, nameToIndex);
        }

        private readonly record struct ObjVertexRef(int Position, int Normal, int TexCoord);
        private readonly record struct ObjFace(ObjVertexRef V0, ObjVertexRef V1, ObjVertexRef V2, int MaterialIndex);

        private readonly struct ObjVertexKey : IEquatable<ObjVertexKey>
        {
            public readonly int Pos, Nrm, Tex;
            public ObjVertexKey(int pos, int nrm, int tex) { Pos = pos; Nrm = nrm; Tex = tex; }
            public bool Equals(ObjVertexKey other) => Pos == other.Pos && Nrm == other.Nrm && Tex == other.Tex;
            public override bool Equals(object? obj) => obj is ObjVertexKey o && Equals(o);
            public override int GetHashCode() => HashCode.Combine(Pos, Nrm, Tex);
        }
    }

    // =========================================================================
    // glTF EXPORTER
    // =========================================================================

    /// <summary>
    /// Exports a <see cref="MeshAsset"/> to glTF 2.0 JSON (.gltf) or binary (.glb).
    /// </summary>
    public class GlTFExporter
    {
        private const int ComponentTypeFloat = 5126;
        private const int ComponentTypeUnsignedInt = 5125;
        private const int ComponentTypeUnsignedShort = 5123;

        public async Task<MeshExportResult> ExportAsync(
            string filePath,
            MeshAsset asset,
            MeshExportConfig? config = null,
            CancellationToken ct = default)
        {
            config ??= new MeshExportConfig();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = new MeshExportResult { OutputPath = filePath };

            try
            {
                if (asset == null)
                {
                    result.ErrorMessage = "Mesh asset is null.";
                    return result;
                }

                if (asset.Primitives.Count == 0 || asset.TotalVertexCount == 0)
                {
                    result.ErrorMessage = "Mesh asset has no geometry.";
                    return result;
                }

                var ext = Path.GetExtension(filePath).ToLowerInvariant();
                if (ext is not (".gltf" or ".glb"))
                {
                    result.ErrorMessage = $"Unsupported export format: {ext}";
                    return result;
                }

                byte[] binary = BuildBinaryPayload(asset, out int indexComponentType);

                if (ext == ".glb")
                {
                    await WriteGlbAsync(filePath, asset, binary, indexComponentType, config, ct);
                }
                else
                {
                    await WriteGltfAsync(filePath, asset, binary, indexComponentType, config, ct);
                }

                result.Success = true;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = $"glTF export error: {ex.Message}";
            }

            sw.Stop();
            result.ExportTime = sw.Elapsed;
            return result;
        }

        private static byte[] BuildBinaryPayload(MeshAsset asset, out int indexComponentType)
        {
            EnsureBounds(asset);

            int vertexCount = asset.TotalVertexCount;
            int indexCount = asset.TotalIndexCount;

            var positions = new float[vertexCount * 3];
            var normals = new float[vertexCount * 3];
            var indices = new List<uint>(indexCount);

            int vertexOffset = 0;
            foreach (var primitive in asset.Primitives)
            {
                for (int i = 0; i < primitive.Vertices.Count; i++)
                {
                    var v = primitive.Vertices[i];
                    int baseIdx = (vertexOffset + i) * 3;
                    positions[baseIdx] = v.Position.X;
                    positions[baseIdx + 1] = v.Position.Y;
                    positions[baseIdx + 2] = v.Position.Z;

                    var normal = v.Normal;
                    if (normal.LengthSquared() < 1e-6f)
                        normal = Vector3.UnitY;
                    else
                        normal = Vector3.Normalize(normal);

                    normals[baseIdx] = normal.X;
                    normals[baseIdx + 1] = normal.Y;
                    normals[baseIdx + 2] = normal.Z;
                }

                foreach (uint index in primitive.Indices)
                    indices.Add((uint)(vertexOffset + index));

                vertexOffset += primitive.Vertices.Count;
            }

            byte[] positionBytes = MemoryMarshal.AsBytes(positions.AsSpan()).ToArray();
            byte[] normalBytes = MemoryMarshal.AsBytes(normals.AsSpan()).ToArray();

            bool useShortIndices = vertexCount <= ushort.MaxValue;
            byte[] indexBytes;
            if (useShortIndices)
            {
                indexComponentType = ComponentTypeUnsignedShort;
                var shortIndices = indices.Select(i => (ushort)i).ToArray();
                indexBytes = MemoryMarshal.AsBytes(shortIndices.AsSpan()).ToArray();
            }
            else
            {
                indexComponentType = ComponentTypeUnsignedInt;
                indexBytes = MemoryMarshal.AsBytes(indices.ToArray().AsSpan()).ToArray();
            }

            int posOffset = 0;
            int nrmOffset = Align4(posOffset + positionBytes.Length);
            int idxOffset = Align4(nrmOffset + normalBytes.Length);

            var binary = new byte[Align4(idxOffset + indexBytes.Length)];
            positionBytes.CopyTo(binary, posOffset);
            normalBytes.CopyTo(binary, nrmOffset);
            indexBytes.CopyTo(binary, idxOffset);
            return binary;
        }

        private static void EnsureBounds(MeshAsset asset)
        {
            if (asset.Bounds.Min.X <= asset.Bounds.Max.X)
                return;

            asset.Bounds = BoundingBox3D.Invalid;
            foreach (var primitive in asset.Primitives)
            {
                if (primitive.Bounds.Min.X <= primitive.Bounds.Max.X)
                {
                    asset.Bounds.Encapsulate(primitive.Bounds);
                    continue;
                }

                foreach (var vertex in primitive.Vertices)
                    asset.Bounds.Encapsulate(vertex.Position);
            }
        }

        private async Task WriteGltfAsync(
            string filePath,
            MeshAsset asset,
            byte[] binary,
            int indexComponentType,
            MeshExportConfig config,
            CancellationToken ct)
        {
            string binFileName = Path.GetFileNameWithoutExtension(filePath) + ".bin";
            string binPath = Path.Combine(Path.GetDirectoryName(filePath)!, binFileName);
            await File.WriteAllBytesAsync(binPath, binary, ct);

            int posOffset = 0;
            int posLength = asset.TotalVertexCount * 3 * sizeof(float);
            int nrmOffset = Align4(posLength);
            int nrmLength = asset.TotalVertexCount * 3 * sizeof(float);
            int idxOffset = Align4(nrmOffset + nrmLength);
            int idxLength = indexComponentType == ComponentTypeUnsignedShort
                ? asset.TotalIndexCount * sizeof(ushort)
                : asset.TotalIndexCount * sizeof(uint);

            string json = BuildSceneJson(asset, binFileName, binary.Length, posOffset, posLength, nrmOffset, nrmLength, idxOffset, idxLength, indexComponentType, config);
            await File.WriteAllTextAsync(filePath, json, ct);
        }

        private async Task WriteGlbAsync(
            string filePath,
            MeshAsset asset,
            byte[] binary,
            int indexComponentType,
            MeshExportConfig config,
            CancellationToken ct)
        {
            int posOffset = 0;
            int posLength = asset.TotalVertexCount * 3 * sizeof(float);
            int nrmOffset = Align4(posLength);
            int nrmLength = asset.TotalVertexCount * 3 * sizeof(float);
            int idxOffset = Align4(nrmOffset + nrmLength);
            int idxLength = indexComponentType == ComponentTypeUnsignedShort
                ? asset.TotalIndexCount * sizeof(ushort)
                : asset.TotalIndexCount * sizeof(uint);

            string json = BuildSceneJson(asset, null, binary.Length, posOffset, posLength, nrmOffset, nrmLength, idxOffset, idxLength, indexComponentType, config);
            byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
            int jsonPadding = (4 - jsonBytes.Length % 4) % 4;
            int jsonChunkLength = jsonBytes.Length + jsonPadding;
            int binPadding = (4 - binary.Length % 4) % 4;
            int totalLength = 12 + 8 + jsonChunkLength + 8 + binary.Length + binPadding;

            await using var stream = File.Create(filePath);
            await using var writer = new BinaryWriter(stream);
            writer.Write(0x46546C67); // glTF
            writer.Write(2u);
            writer.Write((uint)totalLength);
            writer.Write((uint)jsonChunkLength);
            writer.Write(0x4E4F534Au); // JSON
            writer.Write(jsonBytes);
            for (int i = 0; i < jsonPadding; i++)
                writer.Write((byte)' ');
            writer.Write((uint)(binary.Length + binPadding));
            writer.Write(0x004E4942u); // BIN
            writer.Write(binary);
            for (int i = 0; i < binPadding; i++)
                writer.Write((byte)0);
            await stream.FlushAsync(ct);
        }

        private static string BuildSceneJson(
            MeshAsset asset,
            string? externalBinUri,
            int bufferLength,
            int posOffset,
            int posLength,
            int nrmOffset,
            int nrmLength,
            int idxOffset,
            int idxLength,
            int indexComponentType,
            MeshExportConfig config)
        {
            var root = new Dictionary<string, object?>
            {
                ["asset"] = new Dictionary<string, object?> { ["version"] = "2.0", ["generator"] = "Synapse MeshLoader" },
                ["scene"] = 0,
                ["scenes"] = new[] { new Dictionary<string, object?> { ["nodes"] = new[] { 0 } } },
                ["nodes"] = new[] { new Dictionary<string, object?> { ["mesh"] = 0, ["name"] = asset.Name } },
                ["meshes"] = new[] { BuildMeshJson(asset, config) },
                ["buffers"] = new[]
                {
                    new Dictionary<string, object?>
                    {
                        ["byteLength"] = bufferLength,
                        ["uri"] = externalBinUri
                    }
                },
                ["bufferViews"] = new object[]
                {
                    new Dictionary<string, object?> { ["buffer"] = 0, ["byteOffset"] = posOffset, ["byteLength"] = posLength, ["target"] = 34962 },
                    new Dictionary<string, object?> { ["buffer"] = 0, ["byteOffset"] = nrmOffset, ["byteLength"] = nrmLength, ["target"] = 34962 },
                    new Dictionary<string, object?> { ["buffer"] = 0, ["byteOffset"] = idxOffset, ["byteLength"] = idxLength, ["target"] = 34963 }
                },
                ["accessors"] = new object[]
                {
                    new Dictionary<string, object?> { ["bufferView"] = 0, ["componentType"] = ComponentTypeFloat, ["count"] = asset.TotalVertexCount, ["type"] = "VEC3", ["max"] = new[] { asset.Bounds.Max.X, asset.Bounds.Max.Y, asset.Bounds.Max.Z }, ["min"] = new[] { asset.Bounds.Min.X, asset.Bounds.Min.Y, asset.Bounds.Min.Z } },
                    new Dictionary<string, object?> { ["bufferView"] = 1, ["componentType"] = ComponentTypeFloat, ["count"] = asset.TotalVertexCount, ["type"] = "VEC3" },
                    new Dictionary<string, object?> { ["bufferView"] = 2, ["componentType"] = indexComponentType, ["count"] = asset.TotalIndexCount, ["type"] = "SCALAR" }
                }
            };

            if (config.ExportMaterials && asset.Materials.Count > 0)
                root["materials"] = asset.Materials.Select(BuildMaterialJson).ToArray();

            return JsonSerializer.Serialize(root, new JsonSerializerOptions { WriteIndented = true });
        }

        private static Dictionary<string, object?> BuildMeshJson(MeshAsset asset, MeshExportConfig config)
        {
            var primitive = new Dictionary<string, object?>
            {
                ["attributes"] = new Dictionary<string, int> { ["POSITION"] = 0, ["NORMAL"] = 1 },
                ["indices"] = 2,
                ["mode"] = 4
            };

            if (config.ExportMaterials && asset.Materials.Count > 0)
                primitive["material"] = 0;

            return new Dictionary<string, object?>
            {
                ["name"] = asset.Name,
                ["primitives"] = new[] { primitive }
            };
        }

        private static Dictionary<string, object?> BuildMaterialJson(MeshMaterial material)
        {
            return new Dictionary<string, object?>
            {
                ["name"] = material.Name,
                ["pbrMetallicRoughness"] = new Dictionary<string, object?>
                {
                    ["baseColorFactor"] = new[] { material.BaseColor.X, material.BaseColor.Y, material.BaseColor.Z, material.Opacity },
                    ["metallicFactor"] = material.Metallic,
                    ["roughnessFactor"] = material.Roughness
                },
                ["emissiveFactor"] = new[] { material.EmissiveColor.X, material.EmissiveColor.Y, material.EmissiveColor.Z },
                ["doubleSided"] = material.DoubleSided
            };
        }

        private static int Align4(int value) => (value + 3) & ~3;
    }

    // =========================================================================
    // UNIFIED MESH LOADER
    // =========================================================================

    /// <summary>
    /// Unified mesh loading interface that dispatches to format-specific loaders.
    /// Provides async loading, progress reporting, and caching.
    /// </summary>
    public class MeshLoader
    {
        private readonly GlTFLoader _gltfLoader = new();
        private readonly ObjLoader _objLoader = new();
        private readonly FbxAsciiLoader _fbxLoader = new();
        private readonly UsdAsciiLoader _usdLoader = new();
        private readonly UsdBinaryLoader _usdcLoader = new();
        private readonly GlTFExporter _gltfExporter = new();
        private readonly Dictionary<string, MeshAsset> _cache = new();
        private readonly object _cacheLock = new();

        public IReadOnlyList<string> SupportedFormats { get; } =
            new[] { ".gltf", ".glb", ".obj", ".fbx", ".stl", ".ply", ".usd", ".usda", ".usdc" };

        public int CacheSize { get { lock (_cacheLock) { return _cache.Count; } } }

        public void ClearCache()
        {
            lock (_cacheLock)
            { _cache.Clear(); }
        }

        public void RemoveFromCache(string path)
        {
            lock (_cacheLock)
            { _cache.Remove(path); }
        }

        public bool CanLoad(string filePath)
        {
            var ext = Path.GetExtension(filePath).ToLowerInvariant();
            return ext is ".gltf" or ".glb" or ".obj" or ".fbx" or ".stl" or ".ply" or ".usd" or ".usda" or ".usdc";
        }

        public async Task<MeshLoadResult> LoadAsync(string filePath, MeshLoadConfig? config = null, CancellationToken ct = default)
        {
            lock (_cacheLock)
            {
                if (_cache.TryGetValue(filePath, out var cached))
                {
                    return new MeshLoadResult { Success = true, Asset = cached, LoadTime = TimeSpan.Zero };
                }
            }

            var ext = Path.GetExtension(filePath).ToLowerInvariant();
            MeshLoadResult result = ext switch
            {
                ".gltf" or ".glb" => await _gltfLoader.LoadAsync(filePath, config, ct),
                ".obj" => await _objLoader.LoadAsync(filePath, config, ct),
                ".fbx" => await _fbxLoader.LoadAsync(filePath, config, ct),
                ".usdc" => await _usdcLoader.LoadAsync(filePath, config, ct),
                ".usd" or ".usda" => await _usdLoader.LoadAsync(filePath, config, ct),
                _ => new MeshLoadResult { ErrorMessage = $"Unsupported format: {ext}" }
            };

            if (result.Success && result.Asset != null)
            {
                lock (_cacheLock)
                { _cache[filePath] = result.Asset; }
            }

            return result;
        }

        public MeshAsset? LoadSync(string filePath, MeshLoadConfig? config = null)
        {
            return LoadAsync(filePath, config).GetAwaiter().GetResult().Asset;
        }

        public async Task<MeshExportResult> ExportAsync(
            string filePath,
            MeshAsset asset,
            MeshExportConfig? config = null,
            CancellationToken ct = default)
        {
            var ext = Path.GetExtension(filePath).ToLowerInvariant();
            return ext switch
            {
                ".gltf" or ".glb" => await _gltfExporter.ExportAsync(filePath, asset, config, ct),
                _ => new MeshExportResult { ErrorMessage = $"Unsupported export format: {ext}" }
            };
        }
    }

    // =========================================================================
    // glTF JSON TYPES (minimal for deserialization)
    // =========================================================================

    internal class GlTFRoot
    {
        [JsonPropertyName("asset")] public GlTFAsset? Asset { get; set; }
        [JsonPropertyName("buffers")] public GlTFBuffer[]? Buffers { get; set; }
        [JsonPropertyName("bufferViews")] public GlTFBufferView[]? BufferViews { get; set; }
        [JsonPropertyName("accessors")] public GlTFAccessor[]? Accessors { get; set; }
        [JsonPropertyName("meshes")] public GlTFMesh[]? Meshes { get; set; }
        [JsonPropertyName("materials")] public GlTFMaterial[]? Materials { get; set; }
        [JsonPropertyName("nodes")] public GlTFNode[]? Nodes { get; set; }
        [JsonPropertyName("scene")] public int Scene { get; set; }
        [JsonPropertyName("scenes")] public GlTFScene[]? Scenes { get; set; }
    }

    internal class GlTFAsset
    {
        [JsonPropertyName("generator")] public string? Generator { get; set; }
        [JsonPropertyName("version")] public string? Version { get; set; }
    }

    internal class GlTFBuffer
    {
        [JsonPropertyName("uri")] public string? Uri { get; set; }
        [JsonPropertyName("byteLength")] public int ByteLength { get; set; }
        [JsonIgnore] public byte[]? BinaryData { get; set; }
    }

    internal class GlTFBufferView
    {
        [JsonPropertyName("buffer")] public int Buffer { get; set; }
        [JsonPropertyName("byteOffset")] public int? ByteOffset { get; set; }
        [JsonPropertyName("byteLength")] public int ByteLength { get; set; }
    }

    internal class GlTFAccessor
    {
        [JsonPropertyName("bufferView")] public int BufferView { get; set; }
        [JsonPropertyName("byteOffset")] public int ByteOffset { get; set; }
        [JsonPropertyName("componentType")] public int ComponentType { get; set; }
        [JsonPropertyName("count")] public int Count { get; set; }
        [JsonPropertyName("type")] public string Type { get; set; } = "";
    }

    internal class GlTFMesh
    {
        [JsonPropertyName("name")] public string? Name { get; set; }
        [JsonPropertyName("primitives")] public GlTFPrimitive[]? Primitives { get; set; }
    }

    internal class GlTFPrimitive
    {
        [JsonPropertyName("attributes")] public Dictionary<string, int>? Attributes { get; set; }
        [JsonPropertyName("indices")] public int? Indices { get; set; }
        [JsonPropertyName("material")] public int? Material { get; set; }
    }

    internal class GlTFMaterial
    {
        [JsonPropertyName("name")] public string? Name { get; set; }
        [JsonPropertyName("pbrMetallicRoughness")] public GlTFPbrMetallicRoughness? PbrMetallicRoughness { get; set; }
        [JsonPropertyName("emissiveFactor")] public float[]? EmissiveFactor { get; set; }
        [JsonPropertyName("doubleSided")] public bool DoubleSided { get; set; }
    }

    internal class GlTFPbrMetallicRoughness
    {
        [JsonPropertyName("baseColorFactor")] public float[]? BaseColorFactor { get; set; }
        [JsonPropertyName("metallicFactor")] public float MetallicFactor { get; set; } = 1.0f;
        [JsonPropertyName("roughnessFactor")] public float RoughnessFactor { get; set; } = 1.0f;
    }

    internal class GlTFNode
    {
        [JsonPropertyName("name")] public string? Name { get; set; }
        [JsonPropertyName("mesh")] public int? Mesh { get; set; }
        [JsonPropertyName("children")] public int[]? Children { get; set; }
        [JsonPropertyName("translation")] public float[]? Translation { get; set; }
        [JsonPropertyName("rotation")] public float[]? Rotation { get; set; }
        [JsonPropertyName("scale")] public float[]? Scale { get; set; }
        [JsonPropertyName("matrix")] public float[]? Matrix { get; set; }
    }

    internal class GlTFScene
    {
        [JsonPropertyName("name")] public string? Name { get; set; }
        [JsonPropertyName("nodes")] public int[]? Nodes { get; set; }
    }
}
