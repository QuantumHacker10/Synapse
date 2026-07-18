using System;
using System.Numerics;
using System.Runtime.InteropServices;
using GDNN.RHI.Vulkan;
using GDNN.Core.NeuralNetwork;

namespace GDNN.GPU;

/// <summary>
/// Dispatch Vulkan compute réel pour DeepMicroMLP :
/// upload positions/poids → vkCmdDispatch → readback distances.
/// Disponible uniquement si Vulkan + SPIR-V (DXC/glslang) sont présents.
/// </summary>
public sealed class VulkanNeuralSdfDispatcher : IDisposable
{
    private static readonly object InitLock = new();
    private static VulkanNeuralSdfDispatcher? _shared;
    private static bool _initAttempted;
    private static string _initLog = "not initialized";

    private readonly VulkanRhiDevice _device;
    private readonly byte[] _spirv;
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

    public static VulkanNeuralSdfDispatcher? Shared
    {
        get
        {
            EnsureInit();
            return _shared;
        }
    }

    private VulkanNeuralSdfDispatcher(VulkanRhiDevice device, byte[] spirv)
    {
        _device = device;
        _spirv = spirv;
    }

    private static void EnsureInit()
    {
        if (_initAttempted) return;
        lock (InitLock)
        {
            if (_initAttempted) return;
            _initAttempted = true;
            try
            {
                if (!DeepMicroMLPSpirvEmitter.TryGetSpirv(out byte[] spirv, out string spirvLog))
                {
                    _initLog = "SPIR-V unavailable: " + spirvLog;
                    return;
                }

                var info = new RhiDeviceCreationInfo
                {
                    ApplicationName = "GDNN GpuTestHarness",
                    EnableValidation = false,
                    RequiredExtensions = Array.Empty<string>(),
                    RequiredLayers = Array.Empty<string>()
                };

                var device = new VulkanRhiDevice(info);

                // Validate shader module creation early
                using var module = device.CreateShaderModule(spirv);
                _shared = new VulkanNeuralSdfDispatcher(device, spirv);
                _initLog = $"Vulkan ready; SPIR-V {spirv.Length} bytes; " +
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
    /// Évalue le réseau sur GPU et renvoie les distances.
    /// </summary>
    public float[] Evaluate(DeepMicroMLP network, ReadOnlySpan<Vector3> points)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (points.Length == 0) return Array.Empty<float>();

        float[] packedPositions = DeepMicroMLPSpirvEmitter.PackPositions(points);
        float[] weights = DeepMicroMLPSpirvEmitter.FlattenWeights(network);
        var distances = new float[points.Length];

        ulong posSize = (ulong)(packedPositions.Length * sizeof(float));
        ulong weightSize = (ulong)(weights.Length * sizeof(float));
        ulong distSize = (ulong)(distances.Length * sizeof(float));

        VulkanBuffer? posBuf = null;
        VulkanBuffer? weightBuf = null;
        VulkanBuffer? distBuf = null;
        VulkanShaderModule? shaderModule = null;
        DescriptorSetLayout? setLayout = null;
        DescriptorPool? pool = null;
        VulkanPipelineLayout? pipelineLayout = null;
        VulkanComputePipeline? pipeline = null;

        try
        {
            posBuf = _device.CreateBuffer(new BufferDescription
            {
                Size = posSize,
                Usage = BufferUsageFlag.StorageBuffer,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });
            weightBuf = _device.CreateBuffer(new BufferDescription
            {
                Size = weightSize,
                Usage = BufferUsageFlag.StorageBuffer,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });
            distBuf = _device.CreateBuffer(new BufferDescription
            {
                Size = distSize,
                Usage = BufferUsageFlag.StorageBuffer,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });

            posBuf.SetData(packedPositions);
            weightBuf.SetData(weights);

            shaderModule = _device.CreateShaderModule(_spirv);

            setLayout = _device.CreateDescriptorSetLayout(new LayoutDescription
            {
                Bindings = new[]
                {
                    new DescriptorSetLayoutBinding
                    {
                        Binding = 0,
                        DescriptorType = DescriptorType.StorageBuffer,
                        DescriptorCount = 1,
                        StageFlags = ShaderStageFlag.Compute
                    },
                    new DescriptorSetLayoutBinding
                    {
                        Binding = 1,
                        DescriptorType = DescriptorType.StorageBuffer,
                        DescriptorCount = 1,
                        StageFlags = ShaderStageFlag.Compute
                    },
                    new DescriptorSetLayoutBinding
                    {
                        Binding = 2,
                        DescriptorType = DescriptorType.StorageBuffer,
                        DescriptorCount = 1,
                        StageFlags = ShaderStageFlag.Compute
                    }
                }
            });

            pipelineLayout = _device.CreatePipelineLayout(
                new[] { setLayout.Handle },
                new[] { (ShaderStageFlag.Compute, 0u, 4u) });

            // Entry point name as unmanaged string for stage info
            pipeline = _device.CreateComputePipeline(new ComputePipelineDescription
            {
                PipelineLayout = pipelineLayout.Handle,
                Stage = new PipelineShaderStageCreateInfo
                {
                    Stage = ShaderStageFlag.Compute,
                    Module = shaderModule.Handle,
                    Name = "main"
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
                        DescriptorCount = 3
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
                BufferWrite(sets[0].Handle, 0, posBuf, posSize),
                BufferWrite(sets[0].Handle, 1, weightBuf, weightSize),
                BufferWrite(sets[0].Handle, 2, distBuf, distSize)
            });

            var cmd = _device.CreateCommandBuffer();
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            cmd.BindComputePipeline(pipeline);
            cmd.BindDescriptorSets(PipelineBindPoint.Compute, pipelineLayout.Handle, 0, new[] { sets[0].Handle });

            uint count = (uint)points.Length;
            cmd.PushConstants(pipelineLayout.Handle, ShaderStageFlag.Compute, 0, BitConverter.GetBytes(count));

            uint groups = (count + DeepMicroMLPSpirvEmitter.LocalSizeX - 1) / DeepMicroMLPSpirvEmitter.LocalSizeX;
            cmd.Dispatch(groups, 1, 1);
            cmd.End();

            _device.SubmitCommandBuffer(cmd, _device.ComputeQueue, null!);
            _device.WaitForIdle();

            ReadBuffer(distBuf, distances);
            return distances;
        }
        finally
        {
            pipeline?.Dispose();
            pipelineLayout?.Dispose();
            pool?.Dispose();
            setLayout?.Dispose();
            shaderModule?.Dispose();
            distBuf?.Dispose();
            weightBuf?.Dispose();
            posBuf?.Dispose();
        }
    }

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
                Range = size
            }
        }
    };

    private static void ReadBuffer(VulkanBuffer buffer, float[] destination)
    {
        IntPtr mapped = buffer.Map();
        try
        {
            Marshal.Copy(mapped, destination, 0, destination.Length);
        }
        finally
        {
            buffer.Unmap();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _device.Dispose();
    }
}
