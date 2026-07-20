// =============================================================================
// GDNN Engine - Vulkan 1.4 Render Hardware Interface Backend
// File: VulkanRhiDevice.cs
// Description: Complete Vulkan RHI implementation for the G-DNN Engine
// Author: GDNN Engine Team
// Version: 1.0.0
// =============================================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Rendering.Compat;

namespace GDNN.RHI.Vulkan
{
    public class RhiDeviceCreationInfo
    {
        public string ApplicationName { get; set; } = "GDNN Engine";
        public uint ApplicationVersion { get; set; } = 1;
        public string EngineName { get; set; } = "GDNN";
        public uint EngineVersion { get; set; } = 1;
        public uint ApiVersion { get; set; } = VK_API_VERSION_1_4;
        public string[] RequiredExtensions { get; set; } = Array.Empty<string>();
        public string[] RequiredLayers { get; set; } = Array.Empty<string>();
        public bool EnableValidation { get; set; } = true;
        public IntPtr SurfaceHandle { get; set; }

        public const uint VK_API_VERSION_1_4 = (1 << 22) | (4 << 12);
        public const uint VK_API_VERSION_1_3 = (1 << 22) | (3 << 12);
        public const uint VK_API_VERSION_1_2 = (1 << 22) | (2 << 12);
        public const uint VK_API_VERSION_1_1 = (1 << 22) | (1 << 12);
        public const uint VK_API_VERSION_1_0 = (1 << 22);
    }

    /// <summary>Buffer creation description</summary>
    public class BufferDescription
    {
        public ulong Size { get; set; }
        public BufferUsageFlag Usage { get; set; }
        public MemoryPropertyFlag MemoryProperties { get; set; }
        public SharingMode SharingMode { get; set; } = SharingMode.Exclusive;
        public uint[] QueueFamilyIndices { get; set; }
        public string DebugName { get; set; }
    }

    /// <summary>Texture creation description</summary>
    public class TextureDescription
    {
        public ImageType Type { get; set; } = ImageType.Type2D;
        public VulkanFormat Format { get; set; } = VulkanFormat.R8G8B8A8Unorm;
        public uint Width { get; set; } = 1;
        public uint Height { get; set; } = 1;
        public uint Depth { get; set; } = 1;
        public uint MipLevels { get; set; } = 1;
        public uint ArrayLayers { get; set; } = 1;
        public SampleCountFlag Samples { get; set; } = SampleCountFlag.Count1;
        public ImageTiling Tiling { get; set; } = ImageTiling.Optimal;
        public ImageUsageFlag Usage { get; set; } = ImageUsageFlag.Sampled | ImageUsageFlag.TransferDst;
        public SharingMode SharingMode { get; set; } = SharingMode.Exclusive;
        public ImageLayout InitialLayout { get; set; } = ImageLayout.Undefined;
        public uint[] QueueFamilyIndices { get; set; }
        public string DebugName { get; set; }
    }

    /// <summary>Render pass creation description</summary>
    public class RenderPassDescription
    {
        public AttachmentDescription[] Attachments { get; set; }
        public SubpassDescription[] Subpasses { get; set; }
        public SubpassDependency[] Dependencies { get; set; }
        public string DebugName { get; set; }
    }

    /// <summary>Framebuffer creation description</summary>
    public class FramebufferDescription
    {
        public IntPtr RenderPass { get; set; }
        public IntPtr[] Attachments { get; set; }
        public uint Width { get; set; }
        public uint Height { get; set; }
        public uint Layers { get; set; } = 1;
        public string DebugName { get; set; }
    }

    /// <summary>Pipeline creation description</summary>
    public class PipelineDescription
    {
        public PipelineShaderStageCreateInfo[] ShaderStages { get; set; }
        public PipelineVertexInputStateCreateInfo VertexInputState { get; set; }
        public PipelineInputAssemblyStateCreateInfo InputAssemblyState { get; set; }
        public PipelineTessellationStateCreateInfo TessellationState { get; set; }
        public PipelineViewportStateCreateInfo ViewportState { get; set; }
        public PipelineRasterizationStateCreateInfo RasterizationState { get; set; }
        public PipelineMultisampleStateCreateInfo MultisampleState { get; set; }
        public PipelineDepthStencilStateCreateInfo DepthStencilState { get; set; }
        public PipelineColorBlendStateCreateInfo ColorBlendState { get; set; }
        public PipelineDynamicStateCreateInfo DynamicState { get; set; }
        public IntPtr PipelineLayout { get; set; }
        public IntPtr RenderPass { get; set; }
        public uint Subpass { get; set; }
        public PipelineCreateFlag Flags { get; set; }
        public IntPtr BasePipelineHandle { get; set; } = IntPtr.Zero;
        public int BasePipelineIndex { get; set; } = -1;
        public string DebugName { get; set; }
    }

    /// <summary>Compute pipeline creation description</summary>
    public class ComputePipelineDescription
    {
        public PipelineShaderStageCreateInfo Stage { get; set; }
        public IntPtr PipelineLayout { get; set; }
        public PipelineCreateFlag Flags { get; set; }
        public IntPtr BasePipelineHandle { get; set; } = IntPtr.Zero;
        public int BasePipelineIndex { get; set; } = -1;
        public string DebugName { get; set; }
    }

    /// <summary>Descriptor set layout description</summary>
    public class LayoutDescription
    {
        public DescriptorSetLayoutBinding[] Bindings { get; set; }
        public DescriptorSetLayoutCreateFlag Flags { get; set; }
        public string DebugName { get; set; }
    }

    /// <summary>Descriptor set layout binding</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorSetLayoutBinding
    {
        public uint Binding;
        public DescriptorType DescriptorType;
        public uint DescriptorCount;
        public ShaderStageFlag StageFlags;
        public IntPtr[] ImmutableSamplers;
    }

    /// <summary>Descriptor pool description</summary>
    public class PoolDescription
    {
        public DescriptorPoolCreateFlag Flags { get; set; }
        public uint MaxSets { get; set; }
        public DescriptorPoolSize[] PoolSizes { get; set; }
        public string DebugName { get; set; }
    }

    /// <summary>Descriptor pool size</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorPoolSize
    {
        public DescriptorType Type;
        public uint DescriptorCount;
    }

    /// <summary>Descriptor set allocation info</summary>
    public class DescriptorSetAllocation
    {
        public IntPtr Pool { get; set; }
        public IntPtr[] Layouts { get; set; }
    }

    /// <summary>Descriptor write operation</summary>
    public class DescriptorWrite
    {
        public IntPtr DescriptorSet { get; set; }
        public uint DstBinding { get; set; }
        public uint DstArrayElement { get; set; }
        public DescriptorType DescriptorType { get; set; }
        public DescriptorImageInfo[] ImageInfos { get; set; }
        public DescriptorBufferInfo[] BufferInfos { get; set; }
    }

    /// <summary>Descriptor image info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorImageInfo
    {
        public IntPtr Sampler;
        public IntPtr ImageView;
        public ImageLayout ImageLayout;
    }

    /// <summary>Descriptor buffer info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorBufferInfo
    {
        public IntPtr Buffer;
        public ulong Offset;
        public ulong Range;
    }

    /// <summary>Sampler creation description</summary>
    public class SamplerDescription
    {
        public Filter MagFilter { get; set; } = Filter.Linear;
        public Filter MinFilter { get; set; } = Filter.Linear;
        public SamplerAddressMode AddressModeU { get; set; } = SamplerAddressMode.Repeat;
        public SamplerAddressMode AddressModeV { get; set; } = SamplerAddressMode.Repeat;
        public SamplerAddressMode AddressModeW { get; set; } = SamplerAddressMode.Repeat;
        public float MipLodBias { get; set; } = 0.0f;
        public bool AnisotropyEnable { get; set; } = true;
        public float MaxAnisotropy { get; set; } = 16.0f;
        public bool CompareEnable { get; set; } = false;
        public CompareOp CompareOp { get; set; } = CompareOp.Always;
        public float MinLod { get; set; } = 0.0f;
        public float MaxLod { get; set; } = 1000.0f;
        public SampleCountFlag SampleCountFlags { get; set; } = SampleCountFlag.Count1;
        public string DebugName { get; set; }
    }

    /// <summary>Hardware capabilities query result</summary>
    public class HardwareCapabilities
    {
        public string DeviceName { get; set; }
        public string DriverVersion { get; set; }
        public string VendorId { get; set; }
        public uint DeviceId { get; set; }
        public uint ApiVersion { get; set; }
        public ulong DedicatedVideoMemory { get; set; }
        public ulong DedicatedHostMemory { get; set; }
        public ulong DedicatedDeviceMemory { get; set; }
        public ulong SharedVideoMemory { get; set; }
        public ulong SharedHostMemory { get; set; }
        public ulong SharedDeviceMemory { get; set; }
        public SampleCountFlag MaxColorSampleCounts { get; set; }
        public SampleCountFlag MaxDepthSampleCounts { get; set; }
        public SampleCountFlag MaxStencilSampleCounts { get; set; }
        public SampleCountFlag MaxStorageImageSampleCounts { get; set; }
        public uint MaxTextureDimension1D { get; set; }
        public uint MaxTextureDimension2D { get; set; }
        public uint MaxTextureDimension3D { get; set; }
        public uint MaxTextureArrayLayers { get; set; }
        public uint MaxUniformBufferRange { get; set; }
        public uint MaxStorageBufferRange { get; set; }
        public uint MaxPushConstantsSize { get; set; }
        public uint MaxComputeWorkGroupCount0 { get; set; }
        public uint MaxComputeWorkGroupCount1 { get; set; }
        public uint MaxComputeWorkGroupCount2 { get; set; }
        public uint MaxComputeWorkGroupSize0 { get; set; }
        public uint MaxComputeWorkGroupSize1 { get; set; }
        public uint MaxComputeWorkGroupSize2 { get; set; }
        public uint MaxComputeWorkGroupInvocations { get; set; }
        public uint MaxBoundDescriptorSets { get; set; }
        public uint MaxColorAttachments { get; set; }
        public uint MaxVertexInputAttributes { get; set; }
        public uint MaxVertexInputBindings { get; set; }
        public uint MaxVertexInputAttributeOffset { get; set; }
        public uint MaxVertexInputBindingStride { get; set; }
        public uint MaxVertexOutputComponents { get; set; }
        public float MaxSamplerAnisotropy { get; set; }
        public uint SubgroupPropertiesSubgroupSize { get; set; }
        public bool SupportsTimelineSemaphores { get; set; }
        public bool SupportsBufferDeviceAddress { get; set; }
        public bool SupportsDynamicRendering { get; set; }
        public bool SupportsSynchronization2 { get; set; }
        public PhysicalDeviceMemoryProperties MemoryProperties { get; set; }
        public string[] SupportedExtensions { get; set; }
    }

    /// <summary>Pipeline cache creation info</summary>
    public class PipelineCacheInfo
    {
        public byte[] InitialData { get; set; }
    }

    /// <summary>Command buffer allocation info</summary>
    public class CommandBufferAllocationInfo
    {
        public IntPtr CommandPool { get; set; }
        public uint Level { get; set; } = 0;
        public uint CommandBufferCount { get; set; } = 1;
    }

    // =========================================================================
    // INTERFACES
    // =========================================================================

    /// <summary>Core RHI device interface</summary>
    public interface IRhiDevice : IDisposable
    {
        IntPtr PhysicalDevice { get; }
        IntPtr Device { get; }
        IntPtr Instance { get; }
        IntPtr Surface { get; }
        IntPtr Queue { get; }
        VulkanSwapchain Swapchain { get; }

        VulkanSwapchain CreateSwapchain(IntPtr surface);
        QueueFamilyIndices GetQueueFamilies();
        VulkanBuffer CreateBuffer(BufferDescription description);
        VulkanTexture CreateTexture(TextureDescription description);
        VulkanRenderPass CreateRenderPass(RenderPassDescription description);
        VulkanFramebuffer CreateFramebuffer(FramebufferDescription description);
        VulkanPipeline CreatePipeline(PipelineDescription description);
        VulkanComputePipeline CreateComputePipeline(ComputePipelineDescription description);
        DescriptorSetLayout CreateDescriptorSetLayout(LayoutDescription description);
        DescriptorPool CreateDescriptorPool(PoolDescription description);
        DescriptorSet[] AllocateDescriptorSets(DescriptorSetAllocation allocation);
        void UpdateDescriptorSets(DescriptorWrite[] writes);
        VulkanShaderModule CreateShaderModule(byte[] spirvCode);
        Sampler CreateSampler(SamplerDescription description);
        VulkanCommandBuffer CreateCommandBuffer();
        void SubmitCommandBuffer(VulkanCommandBuffer commandBuffer, IntPtr queue, Fence fence);
        void WaitForIdle();
        HardwareCapabilities QueryCapabilities();
    }

    /// <summary>Interface for disposable Vulkan objects</summary>
    public interface IVulkanObject : IDisposable
    {
        IntPtr Handle { get; }
        string DebugName { get; set; }
    }

    // =========================================================================
    // HELPER CLASSES
    // =========================================================================

    /// <summary>Queue family index lookup results</summary>
    public class QueueFamilyIndices
    {
        public int GraphicsFamily { get; set; } = -1;
        public int ComputeFamily { get; set; } = -1;
        public int TransferFamily { get; set; } = -1;
        public int PresentFamily { get; set; } = -1;
        public int SparseBindingFamily { get; set; } = -1;

        public bool IsComplete => GraphicsFamily >= 0 && PresentFamily >= 0;
        public bool HasCompute => ComputeFamily >= 0;
        public bool HasTransfer => TransferFamily >= 0;

        public uint[] UniqueQueueFamilies
        {
            get
            {
                var families = new HashSet<int>();
                if (GraphicsFamily >= 0)
                    families.Add(GraphicsFamily);
                if (ComputeFamily >= 0)
                    families.Add(ComputeFamily);
                if (TransferFamily >= 0)
                    families.Add(TransferFamily);
                if (PresentFamily >= 0)
                    families.Add(PresentFamily);
                return families.Select(x => (uint)x).ToArray();
            }
        }
    }

    /// <summary>Physical device info</summary>
    public class VulkanPhysicalDeviceInfo
    {
        public IntPtr Handle { get; set; }
        public string DeviceName { get; set; }
        public uint VendorId { get; set; }
        public uint DeviceId { get; set; }
        public uint ApiVersion { get; set; }
        public uint DriverVersion { get; set; }
        public QueueFamilyProperties[] QueueFamilyProperties { get; set; }
        public PhysicalDeviceMemoryProperties MemoryProperties { get; set; }
        public SampleCountFlag MaxColorBufferSampleCounts { get; set; }
        public SampleCountFlag MaxDepthBufferSampleCounts { get; set; }
        public uint MaxTextureDimension1D { get; set; }
        public uint MaxTextureDimension2D { get; set; }
        public uint MaxTextureDimension3D { get; set; }
        public uint MaxTextureArrayLayers { get; set; }
        public uint MaxUniformBufferRange { get; set; }
        public uint MaxStorageBufferRange { get; set; }
        public uint MaxPushConstantsSize { get; set; }
        public uint MaxBoundDescriptorSets { get; set; }
        public uint MaxColorAttachments { get; set; }
        public uint MaxVertexInputAttributes { get; set; }
        public uint MaxVertexInputBindings { get; set; }
        public uint MaxVertexInputAttributeOffset { get; set; }
        public uint MaxVertexInputBindingStride { get; set; }
        public uint MaxVertexOutputComponents { get; set; }
        public float MaxSamplerAnisotropy { get; set; }
        public uint MaxComputeWorkGroupCount0 { get; set; }
        public uint MaxComputeWorkGroupCount1 { get; set; }
        public uint MaxComputeWorkGroupCount2 { get; set; }
        public uint MaxComputeWorkGroupSize0 { get; set; }
        public uint MaxComputeWorkGroupSize1 { get; set; }
        public uint MaxComputeWorkGroupSize2 { get; set; }
        public uint MaxComputeWorkGroupInvocations { get; set; }
        public bool SupportsTimelineSemaphores { get; set; }
        public bool SupportsBufferDeviceAddress { get; set; }
        public bool SupportsDynamicRendering { get; set; }
        public bool SupportsSynchronization2 { get; set; }
        public bool SupportsDescriptorIndexing { get; set; }
        public bool SupportsMaintenance4 { get; set; }
        public float GpuTimestampPeriod { get; set; }
    }

    /// <summary>Swapchain support details</summary>
    public class VulkanSwapchainSupportDetails
    {
        public SurfaceCapabilities Capabilities { get; set; }
        public SurfaceFormatKHR[] Formats { get; set; }
        public PresentMode[] PresentModes { get; set; }

        public SurfaceFormatKHR ChooseSurfaceFormat(VulkanFormat preferred = VulkanFormat.B8G8R8A8Srgb, PresentMode colorSpace = PresentMode.Fifo)
        {
            if (Formats != null)
            {
                foreach (var format in Formats)
                {
                    if (format.Format == preferred && (int)format.ColorSpace == (int)colorSpace)
                        return format;
                }
                if (Formats.Length > 0)
                    return Formats[0];
            }
            return new SurfaceFormatKHR { Format = preferred, ColorSpace = colorSpace };
        }

        public PresentMode ChoosePresentMode(PresentMode preferred = PresentMode.Mailbox)
        {
            if (PresentModes != null)
            {
                foreach (var mode in PresentModes)
                {
                    if (mode == preferred)
                        return mode;
                }
            }
            return PresentMode.Fifo;
        }

        public Extent2D ChooseExtent(uint width, uint height)
        {
            var extent = Capabilities.CurrentExtent;
            if (extent.Width != 0xFFFFFFFF && extent.Height != 0xFFFFFFFF)
                return extent;

            return new Extent2D
            {
                Width = Math.Max(Capabilities.MinImageExtent.Width, Math.Min(Capabilities.MaxImageExtent.Width, width)),
                Height = Math.Max(Capabilities.MinImageExtent.Height, Math.Min(Capabilities.MaxImageExtent.Height, height))
            };
        }
    }

    /// <summary>Vulkan handle wrapper base class</summary>
    public abstract class VulkanHandle : IVulkanObject
    {
        protected IntPtr _handle;
        protected VulkanDevice _device;
        protected bool _disposed;

        public IntPtr Handle => _handle;
        public string DebugName { get; set; }

        protected VulkanHandle(VulkanDevice device, IntPtr handle)
        {
            _device = device;
            _handle = handle;
        }

        public abstract void Dispose();
    }

    /// <summary>Wrapper for Vulkan logical device</summary>
    public class VulkanDevice
    {
        public IntPtr Instance { get; set; }
        public IntPtr PhysicalDevice { get; set; }
        public IntPtr LogicalDevice { get; set; }
        public IntPtr Surface { get; set; }
        public IntPtr GraphicsQueue { get; set; }
        public IntPtr ComputeQueue { get; set; }
        public IntPtr TransferQueue { get; set; }
        public IntPtr PresentQueue { get; set; }
        public QueueFamilyIndices QueueIndices { get; set; }
        public VulkanPhysicalDeviceInfo PhysicalDeviceInfo { get; set; }
        public VulkanMemoryAllocator MemoryAllocator { get; set; }
        public VulkanSyncManager SyncManager { get; set; }
        public VulkanDescriptorManager DescriptorManager { get; set; }
        public VulkanPipelineCache PipelineCache { get; set; }
        public VulkanResourceTracker ResourceTracker { get; set; }

        // Pfn delegates for extension functions
        internal IntPtr pfnGetDeviceProcAddr;

        public VulkanDevice()
        {
            QueueIndices = new QueueFamilyIndices();
            PhysicalDeviceInfo = new VulkanPhysicalDeviceInfo();
        }

        public IntPtr GetDeviceProcAddr(string name)
        {
            if (pfnGetDeviceProcAddr != IntPtr.Zero)
            {
                var namePtr = Marshal.StringToHGlobalAnsi(name);
                try
                {
                    return vkGetDeviceProcAddr(LogicalDevice, namePtr);
                }
                finally
                {
                    Marshal.FreeHGlobal(namePtr);
                }
            }
            return IntPtr.Zero;
        }

        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern IntPtr vkGetDeviceProcAddr(IntPtr device, IntPtr pName);
    }

    /// <summary>
    /// Vulkan 1.4 Render Hardware Interface device implementation.
    /// Manages all Vulkan resources and provides a complete RHI abstraction
    /// for the G-DNN Engine rendering pipeline.
    /// </summary>
}
