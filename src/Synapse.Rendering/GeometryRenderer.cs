// =============================================================================
// GeometryRenderer.cs - G-DNN Engine: 3D Geometry Rendering Pipeline
// GDNN.Engine - GDNN.Rendering.Geometry
// Manages vertex/index buffers, draw call batching, and mesh rendering
// =============================================================================

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace GDNN.Rendering.Geometry
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    /// <summary>Types of render primitives.</summary>
    public enum RenderPrimitiveType
    {
        TriangleList,
        TriangleStrip,
        LineList,
        LineStrip,
        PointList
    }

    /// <summary>Sort order for transparent rendering.</summary>
    public enum RenderQueue
    {
        Background = 0,
        Opaque = 1000,
        AlphaTest = 2000,
        Transparent = 3000,
        Overlay = 4000
    }

    /// <summary>Draw call status.</summary>
    public enum DrawCallStatus
    {
        Pending,
        Submitted,
        Executed,
        Failed
    }

    // =========================================================================
    // VERTEX & INDEX BUFFERS
    // =========================================================================

    /// <summary>
    /// GPU-ready vertex buffer holding pre-transformed and formatted vertex data.
    /// </summary>
    [DebuggerDisplay("VertexBuffer: {VertexCount} vertices, {Stride} bytes/vertex")]
    public class VertexBuffer : IDisposable
    {
        private float[]? _data;
        private bool _disposed;

        public int VertexCount { get; private set; }
        public int Stride { get; }
        public int ByteSize => VertexCount * Stride * sizeof(float);
        public bool IsEmpty => VertexCount == 0;
        public bool IsDirty { get; set; } = true;

        public VertexBuffer(int stride)
        {
            Stride = stride;
        }

        public void SetData(float[] data, int vertexCount)
        {
            _data = data;
            VertexCount = vertexCount;
            IsDirty = true;
        }

        public void SetData(ReadOnlySpan<float> data, int vertexCount)
        {
            _data = data.ToArray();
            VertexCount = vertexCount;
            IsDirty = true;
        }

        public float[] GetData() => _data ?? Array.Empty<float>();
        public ReadOnlySpan<float> AsSpan() => _data ?? Array.Empty<float>();

        public void Dispose()
        {
            if (!_disposed)
            {
                _data = null;
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// GPU-ready index buffer for indexed drawing.
    /// </summary>
    [DebuggerDisplay("IndexBuffer: {IndexCount} indices")]
    public class IndexBuffer : IDisposable
    {
        private uint[]? _data;
        private bool _disposed;

        public int IndexCount { get; private set; }
        public int ByteSize => IndexCount * sizeof(uint);
        public bool IsEmpty => IndexCount == 0;
        public bool IsDirty { get; set; } = true;

        public void SetData(uint[] data, int indexCount)
        {
            _data = data;
            IndexCount = indexCount;
            IsDirty = true;
        }

        public void SetData(ReadOnlySpan<uint> data)
        {
            _data = data.ToArray();
            IndexCount = data.Length;
            IsDirty = true;
        }

        public uint[] GetData() => _data ?? Array.Empty<uint>();
        public ReadOnlySpan<uint> AsSpan() => _data ?? Array.Empty<uint>();

        public void Dispose()
        {
            if (!_disposed)
            {
                _data = null;
                _disposed = true;
            }
        }
    }

    // =========================================================================
    // MESH DATA
    // =========================================================================

    /// <summary>
    /// Complete mesh data container with vertex/index buffers and metadata.
    /// </summary>
    [DebuggerDisplay("MeshData: {Name}, {VertexCount} verts, {IndexCount} indices")]
    public class MeshData : IDisposable
    {
        private static int _nextId;

        public int Id { get; } = System.Threading.Interlocked.Increment(ref _nextId);
        public string Name { get; set; } = "";
        public VertexBuffer? Vertices { get; set; }
        public IndexBuffer? Indices { get; set; }
        public BoundingBox3D Bounds { get; set; }
        public RenderPrimitiveType PrimitiveType { get; set; } = RenderPrimitiveType.TriangleList;
        public int MaterialIndex { get; set; }
        public bool IsStatic { get; set; }
        public bool CastShadows { get; set; } = true;
        public bool ReceiveShadows { get; set; } = true;

        public int VertexCount => Vertices?.VertexCount ?? 0;
        public int IndexCount => Indices?.IndexCount ?? 0;
        public int TriangleCount => PrimitiveType == RenderPrimitiveType.TriangleList ? IndexCount / 3 : 0;

        public void Dispose()
        {
            Vertices?.Dispose();
            Indices?.Dispose();
        }
    }

    /// <summary>
    /// GPU mesh handle representing uploaded buffer resources.
    /// </summary>
    [DebuggerDisplay("GPUMesh: Id={GpuId}, VBO={VertexBufferHandle}, IBO={IndexBufferHandle}")]
    public class GPUMesh : IDisposable
    {
        private bool _disposed;

        public int GpuId { get; set; }
        public IntPtr VertexBufferHandle { get; set; }
        public IntPtr IndexBufferHandle { get; set; }
        public IntPtr PipelineHandle { get; set; }
        public IntPtr DescriptorSetHandle { get; set; }
        public int VertexCount { get; set; }
        public int IndexCount { get; set; }
        public int MaterialId { get; set; }
        public bool IsUploaded { get; set; }

        public void Dispose()
        {
            if (!_disposed)
            {
                VertexBufferHandle = IntPtr.Zero;
                IndexBufferHandle = IntPtr.Zero;
                PipelineHandle = IntPtr.Zero;
                DescriptorSetHandle = IntPtr.Zero;
                _disposed = true;
            }
        }
    }

    // =========================================================================
    // RENDER COMMAND
    // =========================================================================

    /// <summary>
    /// A single draw command to be submitted to the GPU.
    /// </summary>
    [DebuggerDisplay("DrawCommand: Mesh={MeshId}, Instances={InstanceCount}, Queue={Queue}")]
    public class DrawCommand
    {
        public int MeshId { get; set; }
        public int InstanceCount { get; set; } = 1;
        public int FirstVertex { get; set; }
        public int FirstIndex { get; set; }
        public int IndexCount { get; set; }
        public int MaterialId { get; set; }
        public RenderQueue Queue { get; set; } = RenderQueue.Opaque;
        public Matrix4x4 WorldMatrix { get; set; } = Matrix4x4.Identity;
        public float SortKey { get; set; }
        public DrawCallStatus Status { get; set; }
        public long SubmittedTimestamp { get; set; }
        public long ExecutedTimestamp { get; set; }
    }

    /// <summary>
    /// Batch of draw commands sharing the same material/state.
    /// </summary>
    [DebuggerDisplay("DrawBatch: Material={MaterialId}, Commands={Commands.Count}")]
    public class DrawBatch
    {
        public int MaterialId { get; set; }
        public List<DrawCommand> Commands { get; } = new();
        public int TotalInstances => Commands.Sum(c => c.InstanceCount);
        public int TotalTriangles => Commands.Sum(c => c.IndexCount / 3);
    }

    // =========================================================================
    // GEOMETRY RENDERER
    // =========================================================================

    /// <summary>
    /// Core geometry rendering system managing mesh uploads, draw call submission,
    /// and rendering state. Integrates with the Vulkan RHI backend.
    /// </summary>
    public class GeometryRenderer : IDisposable
    {
        private readonly Dictionary<int, MeshData> _meshData = new();
        private readonly Dictionary<int, GPUMesh> _gpuMeshes = new();
        private readonly List<DrawCommand> _pendingCommands = new();
        private readonly List<DrawBatch> _batches = new();
        private readonly Dictionary<int, MaterialState> _materials = new();
        private readonly object _lock = new();
        private int _nextMeshId;
        private int _nextGpuId;
        private int _drawCallCount;
        private int _triangleCount;
        private int _vertexCount;
        private bool _disposed;

        public int MeshCount { get { lock (_lock) { return _meshData.Count; } } }
        public int GPUMeshCount { get { lock (_lock) { return _gpuMeshes.Count; } } }
        public int PendingDrawCalls { get { lock (_lock) { return _pendingCommands.Count; } } }
        public int TotalDrawCalls => _drawCallCount;
        public int TotalTriangles => _triangleCount;
        public int TotalVertices => _vertexCount;
        public IReadOnlyList<DrawBatch> Batches => _batches;

        public int CreateMesh(string name = "")
        {
            lock (_lock)
            {
                int id = System.Threading.Interlocked.Increment(ref _nextMeshId);
                _meshData[id] = new MeshData { Name = name };
                return id;
            }
        }

        public void SetMeshData(int meshId, float[] vertices, int vertexStride, uint[]? indices)
        {
            lock (_lock)
            {
                if (!_meshData.TryGetValue(meshId, out var mesh)) return;

                mesh.Vertices?.Dispose();
                mesh.Vertices = new VertexBuffer(vertexStride);
                mesh.Vertices.SetData(vertices, vertices.Length / vertexStride);

                if (indices != null && indices.Length > 0)
                {
                    mesh.Indices?.Dispose();
                    mesh.Indices = new IndexBuffer();
                    mesh.Indices.SetData(indices, indices.Length);
                }

                ComputeBounds(mesh);
            }
        }

        public void SetMeshData(int meshId, ReadOnlySpan<float> vertices, int vertexStride, ReadOnlySpan<uint> indices)
        {
            lock (_lock)
            {
                if (!_meshData.TryGetValue(meshId, out var mesh)) return;

                mesh.Vertices?.Dispose();
                mesh.Vertices = new VertexBuffer(vertexStride);
                mesh.Vertices.SetData(vertices, vertices.Length / vertexStride);

                if (indices.Length > 0)
                {
                    mesh.Indices?.Dispose();
                    mesh.Indices = new IndexBuffer();
                    mesh.Indices.SetData(indices);
                }

                ComputeBounds(mesh);
            }
        }

        public void SetMeshBounds(int meshId, BoundingBox3D bounds)
        {
            lock (_lock)
            {
                if (_meshData.TryGetValue(meshId, out var mesh))
                    mesh.Bounds = bounds;
            }
        }

        public void SetMeshMaterial(int meshId, int materialId)
        {
            lock (_lock)
            {
                if (_meshData.TryGetValue(meshId, out var mesh))
                    mesh.MaterialIndex = materialId;
            }
        }

        public MeshData? GetMeshData(int meshId)
        {
            lock (_lock) { return _meshData.TryGetValue(meshId, out var mesh) ? mesh : null; }
        }

        public void DestroyMesh(int meshId)
        {
            lock (_lock)
            {
                if (_meshData.TryGetValue(meshId, out var mesh))
                {
                    mesh.Dispose();
                    _meshData.Remove(meshId);
                }
                if (_gpuMeshes.TryGetValue(meshId, out var gpu))
                {
                    gpu.Dispose();
                    _gpuMeshes.Remove(meshId);
                }
            }
        }

        public void SubmitDraw(DrawCommand command)
        {
            lock (_lock) { _pendingCommands.Add(command); }
        }

        public void SubmitDraw(int meshId, Matrix4x4 worldMatrix, int materialId = -1, RenderQueue queue = RenderQueue.Opaque, int instanceCount = 1)
        {
            lock (_lock)
            {
                var mesh = _meshData.GetValueOrDefault(meshId);
                _pendingCommands.Add(new DrawCommand
                {
                    MeshId = meshId,
                    WorldMatrix = worldMatrix,
                    MaterialId = materialId >= 0 ? materialId : mesh?.MaterialIndex ?? 0,
                    Queue = queue,
                    InstanceCount = instanceCount,
                    IndexCount = mesh?.IndexCount ?? 0,
                    FirstIndex = 0,
                    FirstVertex = 0
                });
            }
        }

        public void SubmitMesh(int meshId, Matrix4x4 worldMatrix, int materialId = -1)
        {
            SubmitDraw(meshId, worldMatrix, materialId, RenderQueue.Opaque);
        }

        public List<DrawBatch> FlushCommands()
        {
            lock (_lock)
            {
                _batches.Clear();

                var sorted = _pendingCommands
                    .OrderBy(c => c.Queue)
                    .ThenBy(c => c.MaterialId)
                    .ThenBy(c => c.SortKey)
                    .ToList();

                DrawBatch? currentBatch = null;
                foreach (var cmd in sorted)
                {
                    if (currentBatch == null || currentBatch.MaterialId != cmd.MaterialId)
                    {
                        currentBatch = new DrawBatch { MaterialId = cmd.MaterialId };
                        _batches.Add(currentBatch);
                    }
                    currentBatch.Commands.Add(cmd);
                }

                _drawCallCount += _pendingCommands.Count;
                foreach (var cmd in _pendingCommands)
                {
                    if (_meshData.TryGetValue(cmd.MeshId, out var mesh))
                    {
                        _triangleCount += cmd.IndexCount / 3 * cmd.InstanceCount;
                        _vertexCount += mesh.VertexCount * cmd.InstanceCount;
                    }
                }

                _pendingCommands.Clear();
                return new List<DrawBatch>(_batches);
            }
        }

        public void ClearCommands()
        {
            lock (_lock) { _pendingCommands.Clear(); }
        }

        public void ResetStats()
        {
            _drawCallCount = 0;
            _triangleCount = 0;
            _vertexCount = 0;
        }

        public GPUMesh? GetGPUMesh(int meshId)
        {
            lock (_lock) { return _gpuMeshes.TryGetValue(meshId, out var gpu) ? gpu : null; }
        }

        public GPUMesh UploadToGPU(int meshId)
        {
            lock (_lock)
            {
                if (_gpuMeshes.TryGetValue(meshId, out var existing))
                    return existing;

                if (!_meshData.TryGetValue(meshId, out var mesh))
                    return new GPUMesh();

                var gpu = new GPUMesh
                {
                    GpuId = System.Threading.Interlocked.Increment(ref _nextGpuId),
                    VertexCount = mesh.VertexCount,
                    IndexCount = mesh.IndexCount,
                    MaterialId = mesh.MaterialIndex,
                    IsUploaded = mesh.Vertices != null && !mesh.Vertices.IsEmpty
                };

                _gpuMeshes[meshId] = gpu;
                return gpu;
            }
        }

        public void SetMaterial(int materialId, MaterialState state)
        {
            lock (_lock) { _materials[materialId] = state; }
        }

        public MaterialState? GetMaterial(int materialId)
        {
            lock (_lock) { return _materials.TryGetValue(materialId, out var state) ? state : null; }
        }

        private void ComputeBounds(MeshData mesh)
        {
            if (mesh.Vertices == null || mesh.Vertices.IsEmpty) return;

            var data = mesh.Vertices.GetData();
            int stride = mesh.Vertices.Stride;
            int count = mesh.Vertices.VertexCount;

            if (count == 0 || stride < 3) return;

            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;

            for (int i = 0; i < count; i++)
            {
                int offset = i * stride;
                if (offset + 2 >= data.Length) break;

                float x = data[offset];
                float y = data[offset + 1];
                float z = data[offset + 2];

                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (z < minZ) minZ = z;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
                if (z > maxZ) maxZ = z;
            }

            mesh.Bounds = new BoundingBox3D(new Vector3(minX, minY, minZ), new Vector3(maxX, maxY, maxZ));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            foreach (var mesh in _meshData.Values)
                mesh.Dispose();
            foreach (var gpu in _gpuMeshes.Values)
                gpu.Dispose();

            _meshData.Clear();
            _gpuMeshes.Clear();
            _pendingCommands.Clear();
            _batches.Clear();
            _materials.Clear();
        }
    }

    // =========================================================================
    // MATERIAL STATE
    // =========================================================================

    /// <summary>Rendering material state for draw call batching.</summary>
    [DebuggerDisplay("MaterialState: Id={Id}, Shader={ShaderId}")]
    public class MaterialState
    {
        public int Id { get; set; }
        public string Name { get; set; } = "";
        public int ShaderId { get; set; }
        public Vector3 BaseColor { get; set; } = Vector3.One;
        public float Metallic { get; set; }
        public float Roughness { get; set; } = 0.5f;
        public float Alpha { get; set; } = 1.0f;
        public bool DoubleSided { get; set; }
        public bool IsTransparent => Alpha < 1.0f;
        public int AlbedoTextureId { get; set; } = -1;
        public int NormalTextureId { get; set; } = -1;
        public int MetallicRoughnessTextureId { get; set; } = -1;
        public int EmissiveTextureId { get; set; } = -1;
        public int AOTextureId { get; set; } = -1;
        public Dictionary<string, object> Parameters { get; } = new();
    }

    /// <summary>
    /// Simple bounding box type for geometry (single precision).
    /// </summary>
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
        public bool Contains(Vector3 point) => point.X >= Min.X && point.X <= Max.X && point.Y >= Min.Y && point.Y <= Max.Y && point.Z >= Min.Z && point.Z <= Max.Z;
        public bool Intersects(BoundingBox3D other) => Min.X <= other.Max.X && Max.X >= other.Min.X && Min.Y <= other.Max.Y && Max.Y >= other.Min.Y && Min.Z <= other.Max.Z && Max.Z >= other.Min.Z;
    }
}