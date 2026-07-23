using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using GDNN.Polygonization;
using GDNN.RHI.Vulkan;

namespace GDNN.GPU;

/// <summary>
/// Dispatch Vulkan compute réel pour rasteriser des meshlets G-DNN :
/// upload headers/positions/triangles → vkCmdDispatch (1 workgroup / meshlet)
/// → readback depthKeys + payloads → <see cref="RasterTarget"/>.
/// Utilise un 2ᵉ <see cref="VulkanRhiDevice"/> (comme le SDF), indépendant du swapchain Studio.
/// </summary>
public sealed class VulkanMeshletRasterizerDispatcher : IDisposable
{
    private static readonly object InitLock = new();
    private static VulkanMeshletRasterizerDispatcher? _shared;
    private static bool _initAttempted;
    private static string _initLog = "not initialized";

    private readonly VulkanRhiDevice _device;
    private readonly byte[] _spirv;
    private readonly string _entryPoint;
    private bool _disposed;

    public static string StatusLog => _initLog;
    public static bool IsAvailable
    {
        get
        {
            EnsureInit();
            return _shared != null;
        }
    }

    public static VulkanMeshletRasterizerDispatcher? Shared
    {
        get
        {
            EnsureInit();
            return _shared;
        }
    }

    private VulkanMeshletRasterizerDispatcher(VulkanRhiDevice device, byte[] spirv, string entryPoint)
    {
        _device = device;
        _spirv = spirv;
        _entryPoint = string.IsNullOrEmpty(entryPoint) ? "main" : entryPoint;
    }

    private static void EnsureInit()
    {
        if (_initAttempted)
            return;
        lock (InitLock)
        {
            if (_initAttempted)
                return;
            _initAttempted = true;
            try
            {
                if (!MeshletRasterizerShaderGenerator.TryGetSpirv(
                        out byte[] spirv, out string entryPoint, out string spirvLog))
                {
                    _initLog = "SPIR-V unavailable: " + spirvLog;
                    return;
                }

                var info = new RhiDeviceCreationInfo
                {
                    ApplicationName = "GDNN MeshletRaster",
                    EnableValidation = false,
                    RequiredExtensions = Array.Empty<string>(),
                    RequiredLayers = Array.Empty<string>()
                };

                var device = new VulkanRhiDevice(info);
                using var module = device.CreateShaderModule(spirv);
                _shared = new VulkanMeshletRasterizerDispatcher(device, spirv, entryPoint);
                _initLog = $"Vulkan ready; SPIR-V {spirv.Length} bytes; entry={entryPoint}; " +
                           (SpirvToolchain.IsGlslangAvailable ? "glslang" :
                            SpirvToolchain.IsDxcAvailable ? "dxc" : "unknown");
            }
            catch (Exception ex)
            {
                _initLog = "Vulkan init failed: " + ex.Message;
                _shared = null;
            }
        }
    }

    /// <summary>
    /// Rasterise les meshlets sur GPU dans un <see cref="RasterTarget"/> (visibilité R32).
    /// </summary>
    public RasterTarget Rasterize(
        NeuralPolygonMesh mesh,
        IReadOnlyList<NeuralMeshlet> meshlets,
        in CameraView camera,
        int width,
        int height)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (width < 1 || height < 1)
            throw new ArgumentOutOfRangeException(nameof(width));
        if (meshlets == null || meshlets.Count == 0)
            return new RasterTarget(width, height);

        MeshletRasterizerShaderGenerator.PackMeshlets(
            mesh, meshlets, out uint[] headers, out float[] positions, out uint[] packedTriangles);

        int pixelCount = width * height;
        var depthKeys = new uint[pixelCount];
        var payloads = new uint[pixelCount];

        ulong headerSize = (ulong)(headers.Length * sizeof(uint));
        ulong posSize = (ulong)(positions.Length * sizeof(float));
        ulong triSize = (ulong)(packedTriangles.Length * sizeof(uint));
        ulong visSize = (ulong)(pixelCount * sizeof(uint));

        VulkanBuffer? headerBuf = null;
        VulkanBuffer? posBuf = null;
        VulkanBuffer? triBuf = null;
        VulkanBuffer? depthBuf = null;
        VulkanBuffer? payloadBuf = null;
        VulkanShaderModule? shaderModule = null;
        DescriptorSetLayout? setLayout = null;
        DescriptorPool? pool = null;
        VulkanPipelineLayout? pipelineLayout = null;
        VulkanComputePipeline? pipeline = null;

        try
        {
            headerBuf = CreateSsbo(headerSize);
            posBuf = CreateSsbo(posSize);
            triBuf = CreateSsbo(triSize);
            depthBuf = CreateSsbo(visSize);
            payloadBuf = CreateSsbo(visSize);

            headerBuf.SetData(headers);
            posBuf.SetData(positions);
            triBuf.SetData(packedTriangles);
            depthBuf.SetData(depthKeys);
            payloadBuf.SetData(payloads);

            shaderModule = _device.CreateShaderModule(_spirv);

            setLayout = _device.CreateDescriptorSetLayout(new LayoutDescription
            {
                Bindings = new[]
                {
                    SsboBinding(0),
                    SsboBinding(1),
                    SsboBinding(2),
                    SsboBinding(3),
                    SsboBinding(4)
                }
            });

            pipelineLayout = _device.CreatePipelineLayout(
                new[] { setLayout.Handle },
                new[] { (ShaderStageFlag.Compute, 0u, (uint)MeshletRasterizerShaderGenerator.PushConstantBytes) });

            pipeline = _device.CreateComputePipeline(new ComputePipelineDescription
            {
                PipelineLayout = pipelineLayout.Handle,
                Stage = new PipelineShaderStageCreateInfo
                {
                    Stage = ShaderStageFlag.Compute,
                    Module = shaderModule.Handle,
                    Name = _entryPoint
                }
            });

            pool = _device.CreateDescriptorPool(new PoolDescription
            {
                MaxSets = 1,
                PoolSizes = new[]
                {
                    new DescriptorPoolSize
                    {
                        Type = DescriptorType.StorageBuffer,
                        DescriptorCount = 5
                    }
                }
            });

            var sets = _device.AllocateDescriptorSets(new DescriptorSetAllocation
            {
                Pool = pool.Handle,
                Layouts = new[] { setLayout.Handle }
            });

            _device.UpdateDescriptorSets(new[]
            {
                BufferWrite(sets[0].Handle, 0, headerBuf, headerSize),
                BufferWrite(sets[0].Handle, 1, posBuf, posSize),
                BufferWrite(sets[0].Handle, 2, triBuf, triSize),
                BufferWrite(sets[0].Handle, 3, depthBuf, visSize),
                BufferWrite(sets[0].Handle, 4, payloadBuf, visSize)
            });

            byte[] push = MeshletRasterizerShaderGenerator.PackPushConstants(
                camera.ViewProjection, (uint)width, (uint)height, (uint)meshlets.Count);

            var cmd = _device.CreateCommandBuffer();
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            cmd.BindComputePipeline(pipeline);
            cmd.BindDescriptorSets(PipelineBindPoint.Compute, pipelineLayout.Handle, 0, new[] { sets[0].Handle });
            cmd.PushConstants(pipelineLayout.Handle, ShaderStageFlag.Compute, 0, push);
            cmd.Dispatch((uint)meshlets.Count, 1, 1);
            cmd.End();

            _device.SubmitCommandBuffer(cmd, _device.ComputeQueue, null!);
            _device.WaitForIdle();

            ReadUints(depthBuf, depthKeys);
            ReadUints(payloadBuf, payloads);

            var target = new RasterTarget(width, height);
            target.ApplyGpuVisibility(depthKeys, payloads);
            return target;
        }
        finally
        {
            pipeline?.Dispose();
            pipelineLayout?.Dispose();
            pool?.Dispose();
            setLayout?.Dispose();
            shaderModule?.Dispose();
            payloadBuf?.Dispose();
            depthBuf?.Dispose();
            triBuf?.Dispose();
            posBuf?.Dispose();
            headerBuf?.Dispose();
        }
    }

    private VulkanBuffer CreateSsbo(ulong size) => _device.CreateBuffer(new BufferDescription
    {
        Size = Math.Max(size, 4),
        Usage = BufferUsageFlag.StorageBuffer,
        MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
    });

    private static DescriptorSetLayoutBinding SsboBinding(uint binding) => new()
    {
        Binding = binding,
        DescriptorType = DescriptorType.StorageBuffer,
        DescriptorCount = 1,
        StageFlags = ShaderStageFlag.Compute
    };

    private static DescriptorWrite BufferWrite(IntPtr set, uint binding, VulkanBuffer buffer, ulong size) => new()
    {
        DescriptorSet = set,
        DstBinding = binding,
        DstArrayElement = 0,
        DescriptorType = DescriptorType.StorageBuffer,
        BufferInfos = new[]
        {
            new DescriptorBufferInfo
            {
                Buffer = buffer.Handle,
                Offset = 0,
                Range = Math.Max(size, 4)
            }
        }
    };

    private static void ReadUints(VulkanBuffer buffer, uint[] destination)
    {
        IntPtr mapped = buffer.Map();
        try
        {
            var ints = new int[destination.Length];
            Marshal.Copy(mapped, ints, 0, ints.Length);
            for (int i = 0; i < destination.Length; i++)
                destination[i] = unchecked((uint)ints[i]);
        }
        finally
        {
            buffer.Unmap();
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _device.Dispose();
    }
}
