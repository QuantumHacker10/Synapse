// =============================================================================
// GDNN Engine - Vulkan 1.4 Render Hardware Interface Backend
// VulkanRhiDevice.Device.cs — Vulkan RHI partial module
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
using Synapse.Infrastructure.Logging;

namespace GDNN.RHI.Vulkan
{
    public class VulkanRhiDevice : IRhiDevice
    {
        // Vulkan native handles
        private IntPtr _instance;
        private IntPtr _physicalDevice;
        private IntPtr _logicalDevice;
        private IntPtr _surface;
        private IntPtr _graphicsQueue;
        private IntPtr _computeQueue;
        private IntPtr _transferQueue;
        private IntPtr _presentQueue;
        private IntPtr _pipelineCache;
        private IntPtr _commandPool;

        // Supporting objects
        private VulkanDevice _device;
        private VulkanSwapchain _swapchain;
        private VulkanMemoryAllocator _memoryAllocator;
        private VulkanSyncManager _syncManager;
        private VulkanDescriptorManager _descriptorManager;
        private VulkanPipelineCache _pipelineCacheWrapper;
        private VulkanResourceTracker _resourceTracker;
        private VulkanPhysicalDeviceInfo _physicalDeviceInfo;
        private QueueFamilyIndices _queueFamilyIndices;
        private RhiDeviceCreationInfo _creationInfo;

        // Vulkan function pointers
        private IntPtr _vkGetInstanceProcAddr;
        private IntPtr _vkGetDeviceProcAddr;

        // Validation
        private IntPtr _debugUtilsMessenger;
        private IntPtr _validationLayer;

        // Device state
        private bool _disposed;
        private readonly object _deviceLock = new object();
        private readonly List<VulkanBuffer> _buffers = new List<VulkanBuffer>();
        private readonly List<VulkanTexture> _textures = new List<VulkanTexture>();
        private readonly List<VulkanRenderPass> _renderPasses = new List<VulkanRenderPass>();
        private readonly List<VulkanFramebuffer> _framebuffers = new List<VulkanFramebuffer>();
        private readonly List<VulkanPipeline> _pipelines = new List<VulkanPipeline>();
        private readonly List<VulkanComputePipeline> _computePipelines = new List<VulkanComputePipeline>();
        private readonly List<VulkanCommandBuffer> _commandBuffers = new List<VulkanCommandBuffer>();
        private readonly List<VulkanShaderModule> _shaderModules = new List<VulkanShaderModule>();
        private readonly List<Sampler> _samplers = new List<Sampler>();
        private readonly List<DescriptorPool> _descriptorPools = new List<DescriptorPool>();
        private readonly List<DescriptorSetLayout> _descriptorSetLayouts = new List<DescriptorSetLayout>();

        // Delegate types
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VkBool32 VkDebugUtilsMessengerCallbackDelegate(uint messageSeverity, uint messageTypes, IntPtr pCallbackData, IntPtr pUserData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult VkCreateDebugUtilsMessengerDelegate(IntPtr instance, ref VkDebugUtilsMessengerCreateInfoEXT pCreateInfo, IntPtr pAllocator, ref IntPtr pMessenger);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void VkDestroyDebugUtilsMessengerDelegate(IntPtr instance, IntPtr messenger, IntPtr pAllocator);

        private VkCreateDebugUtilsMessengerDelegate _vkCreateDebugUtilsMessenger;
        private VkDestroyDebugUtilsMessengerDelegate _vkDestroyDebugUtilsMessenger;

        // Core Vulkan P/Invokes
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern IntPtr vkGetInstanceProcAddr(IntPtr instance, IntPtr pName);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern IntPtr vkGetDeviceProcAddr(IntPtr device, IntPtr pName);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkCreateInstance(ref VkInstanceCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pInstance);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern void vkDestroyInstance(IntPtr instance, IntPtr pAllocator);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkEnumeratePhysicalDevices(IntPtr instance, ref uint pPhysicalDeviceCount, IntPtr pPhysicalDevices);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern void vkGetPhysicalDeviceProperties(IntPtr physicalDevice, IntPtr pProperties);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern void vkGetPhysicalDeviceQueueFamilyProperties(IntPtr physicalDevice, ref uint pQueueFamilyPropertyCount, IntPtr pQueueFamilyProperties);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkGetPhysicalDeviceSurfaceSupport(IntPtr physicalDevice, uint queueFamilyIndex, IntPtr surface, ref IntPtr pSupported);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkGetPhysicalDeviceSurfaceCapabilities(IntPtr physicalDevice, IntPtr surface, IntPtr pSurfaceCapabilities);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkGetPhysicalDeviceSurfaceFormats(IntPtr physicalDevice, IntPtr surface, ref uint pSurfaceFormatCount, IntPtr pSurfaceFormats);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkGetPhysicalDeviceSurfacePresentModes(IntPtr physicalDevice, IntPtr surface, ref uint pPresentModeCount, IntPtr pPresentModes);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern void vkGetPhysicalDeviceMemoryProperties(IntPtr physicalDevice, IntPtr pMemoryProperties);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkCreateDevice(IntPtr physicalDevice, ref VkDeviceCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pDevice);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern void vkDestroyDevice(IntPtr device, IntPtr pAllocator);

        private T? GetDeviceProc<T>(string name) where T : Delegate
        {
            var namePtr = Marshal.StringToHGlobalAnsi(name);
            try
            {
                var ptr = vkGetDeviceProcAddr(_logicalDevice, namePtr);
                if (ptr == IntPtr.Zero)
                    return null;
                return Marshal.GetDelegateForFunctionPointer<T>(ptr);
            }
            finally { Marshal.FreeHGlobal(namePtr); }
        }

        private T? GetInstanceProc<T>(string name) where T : Delegate
        {
            var namePtr = Marshal.StringToHGlobalAnsi(name);
            try
            {
                var ptr = vkGetInstanceProcAddr(_instance, namePtr);
                if (ptr == IntPtr.Zero)
                    return null;
                return Marshal.GetDelegateForFunctionPointer<T>(ptr);
            }
            finally { Marshal.FreeHGlobal(namePtr); }
        }

        // Device function delegate types
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void GetDeviceQueueDelegate(IntPtr device, uint queueFamilyIndex, uint queueIndex, ref IntPtr pQueue);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult DeviceWaitIdleDelegate(IntPtr device);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateSwapchainKHRDelegate(IntPtr device, ref VkSwapchainCreateInfoKHR pCreateInfo, IntPtr pAllocator, ref IntPtr pSwapchain);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroySwapchainKHRDelegate(IntPtr device, IntPtr swapchain, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AcquireNextImageKHRDelegate(IntPtr device, IntPtr swapchain, ulong timeout, IntPtr semaphore, IntPtr fence, ref uint pImageIndex);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult QueuePresentKHRDelegate(IntPtr queue, ref VkPresentInfoKHR pPresentInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult QueueSubmitDelegate(IntPtr queue, uint submitCount, ref VkSubmitInfo pSubmits, IntPtr fence);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult QueueWaitIdleDelegate(IntPtr queue);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateFenceDelegate(IntPtr device, ref VkFenceCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pFence);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyFenceDelegate(IntPtr device, IntPtr fence, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult WaitForFencesDelegate(IntPtr device, uint fenceCount, IntPtr pFences, uint waitAll, ulong timeout);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult ResetFencesDelegate(IntPtr device, uint fenceCount, IntPtr pFences);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateSemaphoreDelegate(IntPtr device, ref VkSemaphoreCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pSemaphore);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroySemaphoreDelegate(IntPtr device, IntPtr semaphore, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateCommandPoolDelegate(IntPtr device, ref VkCommandPoolCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pCommandPool);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyCommandPoolDelegate(IntPtr device, IntPtr commandPool, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AllocateCommandBuffersDelegate(IntPtr device, ref VkCommandBufferAllocateInfo pAllocateInfo, IntPtr pCommandBuffers);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void FreeCommandBuffersDelegate(IntPtr device, IntPtr commandPool, uint commandBufferCount, IntPtr pCommandBuffers);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BeginCommandBufferDelegate(IntPtr commandBuffer, ref VkCommandBufferBeginInfo pBeginInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult EndCommandBufferDelegate(IntPtr commandBuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginRenderPassDelegate(IntPtr cmdBuffer, ref VkRenderPassBeginInfo pRenderPassBegin, uint contents);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndRenderPassDelegate(IntPtr cmdBuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindPipelineDelegate(IntPtr cmdBuffer, PipelineBindPoint pipelineBindPoint, IntPtr pipeline);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDrawDelegate(IntPtr cmdBuffer, uint vertexCount, uint instanceCount, uint firstVertex, uint firstInstance);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDrawIndexedDelegate(IntPtr cmdBuffer, uint indexCount, uint instanceCount, uint firstIndex, int vertexOffset, uint firstInstance);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDispatchDelegate(IntPtr cmdBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdPipelineBarrierDelegate(IntPtr cmdBuffer, PipelineStageFlag srcStageMask, PipelineStageFlag dstStageMask, uint dependencyFlags, uint memoryBarrierCount, IntPtr pMemoryBarriers, uint bufferMemoryBarrierCount, IntPtr pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferDelegate(IntPtr cmdBuffer, IntPtr srcBuffer, IntPtr dstBuffer, uint regionCount, ref BufferCopy pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferToImageDelegate(IntPtr cmdBuffer, IntPtr buffer, IntPtr image, uint imageLayout, uint regionCount, ref BufferImageCopy pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdPushConstantsDelegate(IntPtr cmdBuffer, IntPtr layout, ShaderStageFlag stageFlags, uint offset, uint size, IntPtr pValues);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetViewportDelegate(IntPtr cmdBuffer, uint firstViewport, uint viewportCount, ref Viewport pViewports);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetScissorDelegate(IntPtr cmdBuffer, uint firstScissor, uint scissorCount, ref Rect2D pScissors);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindVertexBuffersDelegate(IntPtr cmdBuffer, uint firstBinding, uint bindingCount, ref IntPtr pBuffers, ref ulong pOffsets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindIndexBufferDelegate(IntPtr cmdBuffer, IntPtr buffer, ulong offset, IndexType indexType);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateBufferDelegate(IntPtr device, ref VkBufferCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pBuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyBufferDelegate(IntPtr device, IntPtr buffer, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void GetBufferMemoryRequirementsDelegate(IntPtr device, IntPtr buffer, out VkMemoryRequirements pMemoryRequirements);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindBufferMemoryDelegate(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AllocateMemoryDelegate(IntPtr device, ref VkMemoryAllocateInfo pAllocateInfo, IntPtr pAllocator, ref IntPtr pMemory);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void FreeMemoryDelegate(IntPtr device, IntPtr memory, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult MapMemoryDelegate(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, ref IntPtr ppData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void UnmapMemoryDelegate(IntPtr device, IntPtr memory);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult FlushMappedMemoryRangesDelegate(IntPtr device, uint memoryRangeCount, ref VkMappedMemoryRange pMemoryRanges);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateImageDelegate(IntPtr device, ref VkImageCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pImage);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageDelegate(IntPtr device, IntPtr image, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void GetImageMemoryRequirementsDelegate(IntPtr device, IntPtr image, out VkMemoryRequirements pMemoryRequirements);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindImageMemoryDelegate(IntPtr device, IntPtr image, IntPtr memory, ulong memoryOffset);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateImageViewDelegate(IntPtr device, ref VkImageViewCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pView);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageViewDelegate(IntPtr device, IntPtr imageView, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateSamplerDelegate(IntPtr device, ref VkSamplerCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pSampler);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroySamplerDelegate(IntPtr device, IntPtr sampler, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateRenderPassDelegate(IntPtr device, ref VkRenderPassCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pRenderPass);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyRenderPassDelegate(IntPtr device, IntPtr renderPass, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateFramebufferDelegate(IntPtr device, ref VkFramebufferCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pFramebuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyFramebufferDelegate(IntPtr device, IntPtr framebuffer, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateShaderModuleDelegate(IntPtr device, ref VkShaderModuleCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pShaderModule);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyShaderModuleDelegate(IntPtr device, IntPtr shaderModule, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateGraphicsPipelinesDelegate(IntPtr device, IntPtr pipelineCache, uint createInfoCount, ref VkGraphicsPipelineCreateInfo pCreateInfos, IntPtr pAllocator, ref IntPtr pPipelines);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateComputePipelinesDelegate(IntPtr device, IntPtr pipelineCache, uint createInfoCount, ref VkComputePipelineCreateInfo pCreateInfos, IntPtr pAllocator, ref IntPtr pPipelines);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineDelegate(IntPtr device, IntPtr pipeline, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreatePipelineLayoutDelegate(IntPtr device, ref VkPipelineLayoutCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pPipelineLayout);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineLayoutDelegate(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateDescriptorSetLayoutDelegate(IntPtr device, ref VkDescriptorSetLayoutCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pSetLayout);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyDescriptorSetLayoutDelegate(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateDescriptorPoolDelegate(IntPtr device, ref VkDescriptorPoolCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pDescriptorPool);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyDescriptorPoolDelegate(IntPtr device, IntPtr descriptorPool, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AllocateDescriptorSetsDelegate(IntPtr device, ref VkDescriptorSetAllocateInfo pAllocateInfo, IntPtr pDescriptorSets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult FreeDescriptorSetsDelegate(IntPtr device, IntPtr descriptorPool, uint descriptorSetCount, IntPtr pDescriptorSets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void UpdateDescriptorSetsDelegate(IntPtr device, uint descriptorWriteCount, IntPtr pDescriptorWrites, uint descriptorCopyCount, IntPtr pDescriptorCopies);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreatePipelineCacheDelegate(IntPtr device, ref VkPipelineCacheCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pPipelineCache);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineCacheDelegate(IntPtr device, IntPtr pipelineCache, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult GetPipelineCacheDataDelegate(IntPtr device, IntPtr pipelineCache, ref IntPtr pDataSize, IntPtr pData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult MergePipelineCachesDelegate(IntPtr device, IntPtr dstCache, uint srcCacheCount, IntPtr pSrcCaches);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyImageDelegate(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBlitImageDelegate(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions, Filter filter);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdResolveImageDelegate(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateQueryPoolDelegate(IntPtr device, IntPtr pCreateInfo, IntPtr pAllocator, ref IntPtr pQueryPool);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyQueryPoolDelegate(IntPtr device, IntPtr queryPool, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult ResetQueryPoolDelegate(IntPtr device, IntPtr queryPool, uint firstQuery, uint queryCount);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult GetQueryPoolResultsDelegate(IntPtr device, IntPtr queryPool, uint firstQuery, uint queryCount, IntPtr dataSize, IntPtr pData, ulong stride, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdWriteTimestampDelegate(IntPtr cmdBuffer, PipelineStageFlag pipelineStage, IntPtr queryPool, uint query);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdResetQueryPoolDelegate(IntPtr cmdBuffer, IntPtr queryPool, uint firstQuery, uint queryCount);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyQueryPoolResultsDelegate(IntPtr cmdBuffer, IntPtr queryPool, uint firstQuery, uint queryCount, IntPtr dstBuffer, ulong dstOffset, ulong stride, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginQueryDelegate(IntPtr cmdBuffer, IntPtr queryPool, uint query, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndQueryDelegate(IntPtr cmdBuffer, IntPtr queryPool, uint query);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginRenderingDelegate(IntPtr cmdBuffer, IntPtr pRenderingInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndRenderingDelegate(IntPtr cmdBuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetDepthBiasEnableDelegate(IntPtr cmdBuffer, uint depthBiasEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetPrimitiveRestartEnableDelegate(IntPtr cmdBuffer, uint primitiveRestartEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetRasterizerDiscardEnableDelegate(IntPtr cmdBuffer, uint rasterizerDiscardEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateRenderPass2Delegate(IntPtr device, IntPtr pCreateInfo, IntPtr pAllocator, ref IntPtr pRenderPass);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginRenderPass2Delegate(IntPtr cmdBuffer, IntPtr pRenderPassBegin, IntPtr pSubpassBeginInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdNextSubpass2Delegate(IntPtr cmdBuffer, IntPtr pSubpassBeginInfo, IntPtr pSubpassEndInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndRenderPass2Delegate(IntPtr cmdBuffer, IntPtr pSubpassEndInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetDepthTestEnableDelegate(IntPtr cmdBuffer, uint depthTestEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetDepthWriteEnableDelegate(IntPtr cmdBuffer, uint depthWriteEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetDepthCompareOpDelegate(IntPtr cmdBuffer, CompareOp depthCompareOp);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetDepthBoundsTestEnableDelegate(IntPtr cmdBuffer, uint depthBoundsTestEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetStencilTestEnableDelegate(IntPtr cmdBuffer, uint stencilTestEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetStencilOpDelegate(IntPtr cmdBuffer, uint faceMask, StencilOp failOp, StencilOp passOp, StencilOp depthFailOp, CompareOp compareOp);

        private GetDeviceQueueDelegate _vkGetDeviceQueue;
        private DeviceWaitIdleDelegate _vkDeviceWaitIdle;
        private CreateSwapchainKHRDelegate _vkCreateSwapchainKHR;
        private DestroySwapchainKHRDelegate _vkDestroySwapchainKHR;
        private AcquireNextImageKHRDelegate _vkAcquireNextImageKHR;
        private QueuePresentKHRDelegate _vkQueuePresentKHR;
        private QueueSubmitDelegate _vkQueueSubmit;
        private QueueWaitIdleDelegate _vkQueueWaitIdle;
        private CreateFenceDelegate _vkCreateFence;
        private DestroyFenceDelegate _vkDestroyFence;
        private WaitForFencesDelegate _vkWaitForFences;
        private ResetFencesDelegate _vkResetFences;
        private CreateSemaphoreDelegate _vkCreateSemaphore;
        private DestroySemaphoreDelegate _vkDestroySemaphore;
        private CreateCommandPoolDelegate _vkCreateCommandPool;
        private DestroyCommandPoolDelegate _vkDestroyCommandPool;
        private AllocateCommandBuffersDelegate _vkAllocateCommandBuffers;
        private FreeCommandBuffersDelegate _vkFreeCommandBuffers;
        private BeginCommandBufferDelegate _vkBeginCommandBuffer;
        private EndCommandBufferDelegate _vkEndCommandBuffer;
        private CmdBeginRenderPassDelegate _vkCmdBeginRenderPass;
        private CmdEndRenderPassDelegate _vkCmdEndRenderPass;
        private CmdBindPipelineDelegate _vkCmdBindPipeline;
        private CmdDrawDelegate _vkCmdDraw;
        private CmdDrawIndexedDelegate _vkCmdDrawIndexed;
        private CmdDispatchDelegate _vkCmdDispatch;
        private CmdPipelineBarrierDelegate _vkCmdPipelineBarrier;
        private CmdCopyBufferDelegate _vkCmdCopyBuffer;
        private CmdCopyBufferToImageDelegate _vkCmdCopyBufferToImage;
        private CmdPushConstantsDelegate _vkCmdPushConstants;
        private CmdSetViewportDelegate _vkCmdSetViewport;
        private CmdSetScissorDelegate _vkCmdSetScissor;
        private CmdBindVertexBuffersDelegate _vkCmdBindVertexBuffers;
        private CmdBindIndexBufferDelegate _vkCmdBindIndexBuffer;
        private CreateBufferDelegate _vkCreateBuffer;
        private DestroyBufferDelegate _vkDestroyBuffer;
        private GetBufferMemoryRequirementsDelegate _vkGetBufferMemoryRequirements;
        private BindBufferMemoryDelegate _vkBindBufferMemory;
        private AllocateMemoryDelegate _vkAllocateMemory;
        private FreeMemoryDelegate _vkFreeMemory;
        private MapMemoryDelegate _vkMapMemory;
        private UnmapMemoryDelegate _vkUnmapMemory;
        private FlushMappedMemoryRangesDelegate _vkFlushMappedMemoryRanges;
        private CreateImageDelegate _vkCreateImage;
        private DestroyImageDelegate _vkDestroyImage;
        private GetImageMemoryRequirementsDelegate _vkGetImageMemoryRequirements;
        private BindImageMemoryDelegate _vkBindImageMemory;
        private CreateImageViewDelegate _vkCreateImageView;
        private DestroyImageViewDelegate _vkDestroyImageView;
        private CreateSamplerDelegate _vkCreateSampler;
        private DestroySamplerDelegate _vkDestroySampler;
        private CreateRenderPassDelegate _vkCreateRenderPass;
        private DestroyRenderPassDelegate _vkDestroyRenderPass;
        private CreateFramebufferDelegate _vkCreateFramebuffer;
        private DestroyFramebufferDelegate _vkDestroyFramebuffer;
        private CreateShaderModuleDelegate _vkCreateShaderModule;
        private DestroyShaderModuleDelegate _vkDestroyShaderModule;
        private CreateGraphicsPipelinesDelegate _vkCreateGraphicsPipelines;
        private CreateComputePipelinesDelegate _vkCreateComputePipelines;
        private DestroyPipelineDelegate _vkDestroyPipeline;
        private CreatePipelineLayoutDelegate _vkCreatePipelineLayout;
        private DestroyPipelineLayoutDelegate _vkDestroyPipelineLayout;
        private CreateDescriptorSetLayoutDelegate _vkCreateDescriptorSetLayout;
        private DestroyDescriptorSetLayoutDelegate _vkDestroyDescriptorSetLayout;
        private CreateDescriptorPoolDelegate _vkCreateDescriptorPool;
        private DestroyDescriptorPoolDelegate _vkDestroyDescriptorPool;
        private AllocateDescriptorSetsDelegate _vkAllocateDescriptorSets;
        private FreeDescriptorSetsDelegate _vkFreeDescriptorSets;
        private UpdateDescriptorSetsDelegate _vkUpdateDescriptorSets;
        private CreatePipelineCacheDelegate _vkCreatePipelineCache;
        private DestroyPipelineCacheDelegate _vkDestroyPipelineCache;
        private GetPipelineCacheDataDelegate _vkGetPipelineCacheData;
        private MergePipelineCachesDelegate _vkMergePipelineCaches;
        private CmdCopyImageDelegate _vkCmdCopyImage;
        private CmdBlitImageDelegate _vkCmdBlitImage;
        private CmdResolveImageDelegate _vkCmdResolveImage;
        private CreateQueryPoolDelegate _vkCreateQueryPool;
        private DestroyQueryPoolDelegate _vkDestroyQueryPool;
        private ResetQueryPoolDelegate _vkResetQueryPool;
        private GetQueryPoolResultsDelegate _vkGetQueryPoolResults;
        private CmdWriteTimestampDelegate _vkCmdWriteTimestamp;
        private CmdResetQueryPoolDelegate _vkCmdResetQueryPool;
        private CmdCopyQueryPoolResultsDelegate _vkCmdCopyQueryPoolResults;
        private CmdBeginQueryDelegate _vkCmdBeginQuery;
        private CmdEndQueryDelegate _vkCmdEndQuery;
        private CmdBeginRenderingDelegate _vkCmdBeginRendering;
        private CmdEndRenderingDelegate _vkCmdEndRendering;
        private CmdSetDepthBiasEnableDelegate _vkCmdSetDepthBiasEnable;
        private CmdSetPrimitiveRestartEnableDelegate _vkCmdSetPrimitiveRestartEnable;
        private CmdSetRasterizerDiscardEnableDelegate _vkCmdSetRasterizerDiscardEnable;
        private CreateRenderPass2Delegate _vkCreateRenderPass2;
        private CmdBeginRenderPass2Delegate _vkCmdBeginRenderPass2;
        private CmdNextSubpass2Delegate _vkCmdNextSubpass2;
        private CmdEndRenderPass2Delegate _vkCmdEndRenderPass2;
        private CmdSetDepthTestEnableDelegate _vkCmdSetDepthTestEnable;
        private CmdSetDepthWriteEnableDelegate _vkCmdSetDepthWriteEnable;
        private CmdSetDepthCompareOpDelegate _vkCmdSetDepthCompareOp;
        private CmdSetDepthBoundsTestEnableDelegate _vkCmdSetDepthBoundsTestEnable;
        private CmdSetStencilTestEnableDelegate _vkCmdSetStencilTestEnable;
        private CmdSetStencilOpDelegate _vkCmdSetStencilOp;

        // VK constants
        private const uint VK_STRUCTURE_TYPE_APPLICATION_INFO = 1;
        private const uint VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
        private const uint VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 3;
        private const uint VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
        private const uint VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR = 1000001000;
        private const uint VK_STRUCTURE_TYPE_PRESENT_INFO_KHR = 1000001001;
        private const uint VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
        private const uint VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
        private const uint VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 7;
        private const uint VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
        private const uint VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
        private const uint VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 41;
        private const uint VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO = 42;
        private const uint VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
        private const uint VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
        private const uint VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE = 6;
        private const uint VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO = 18;
        private const uint VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = 15;
        private const uint VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO = 10;
        private const uint VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO = 37;
        private const uint VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO = 38;
        private const uint VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
        private const uint VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO = 28;
        private const uint VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
        private const uint VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
        private const uint VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
        private const uint VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
        private const uint VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO = 17;
        private const uint VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = 36;
        private const uint VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2 = 1000299000;
        private const uint VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO = 1000106001;
        private const uint VK_STRUCTURE_TYPE_SUBPASS_END_INFO = 1000106002;
        private const uint VK_STRUCTURE_TYPE_RENDERING_INFO = 1000245000;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO = 1000127000;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO = 1000127001;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO = 1000127002;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO = 1000127003;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO = 1000127004;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO = 1000127005;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO = 1000127007;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO = 1000127008;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO = 1000127009;
        private const ulong VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF;

        [StructLayout(LayoutKind.Sequential)]
        private struct VkApplicationInfo
        {
            public uint sType; public IntPtr pNext; public IntPtr pApplicationName;
            public uint applicationVersion; public IntPtr pEngineName;
            public uint engineVersion; public uint apiVersion;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkDebugUtilsMessengerCreateInfoEXT
        {
            public uint sType; public IntPtr pNext; public uint flags;
            public uint messageSeverity; public uint messageType; public IntPtr pfnUserCallback; public IntPtr pUserData;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkDebugUtilsMessengerCallbackDataEXT
        {
            public uint sType; public IntPtr pNext; public IntPtr pMessageIdNumber; public IntPtr pMessage; public uint queueLabelCount;
            public IntPtr pQueueLabels; public uint cmdBufLabelCount; public IntPtr pCmdBufLabels; public uint objectCount; public IntPtr pObjects;
        }

        /// <summary>Creates the Vulkan debug messenger</summary>
        private void CreateDebugMessenger()
        {
            if (!_creationInfo.EnableValidation)
                return;
            try
            {
                _vkCreateDebugUtilsMessenger = GetInstanceProc<VkCreateDebugUtilsMessengerDelegate>("vkCreateDebugUtilsMessengerEXT");
                _vkDestroyDebugUtilsMessenger = GetInstanceProc<VkDestroyDebugUtilsMessengerDelegate>("vkDestroyDebugUtilsMessengerEXT");

                var callback = new VkDebugUtilsMessengerCallbackDelegate(DebugUtilsCallback);
                var createInfo = new VkDebugUtilsMessengerCreateInfoEXT
                {
                    sType = 1000028001,
                    messageSeverity = 0x00000001 | 0x00000100 | 0x00001000,
                    messageType = 0x00000001 | 0x00000002 | 0x00000004,
                    pfnUserCallback = Marshal.GetFunctionPointerForDelegate(callback)
                };

                _vkCreateDebugUtilsMessenger(_instance, ref createInfo, IntPtr.Zero, ref _debugUtilsMessenger);
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Debug("VulkanRhi", "Debug utils messenger unavailable; validation layers disabled.", ex);
            }
        }

        private VkBool32 DebugUtilsCallback(uint messageSeverity, uint messageTypes, IntPtr pCallbackData, IntPtr pUserData)
        {
            if (pCallbackData != IntPtr.Zero)
            {
                var callbackData = Marshal.PtrToStructure<VkDebugUtilsMessengerCallbackDataEXT>(pCallbackData);
                if (callbackData.pMessage != IntPtr.Zero)
                {
                    var message = Marshal.PtrToStringAnsi(callbackData.pMessage);
                    if ((messageSeverity & 0x00001000) != 0)
                        Debug.WriteLine($"[Vulkan ERROR] {message}");
                    else if ((messageSeverity & 0x00000100) != 0)
                        Debug.WriteLine($"[Vulkan WARNING] {message}");
                    else
                        Debug.WriteLine($"[Vulkan INFO] {message}");
                }
            }
            return VkBool32.False;
        }

        private void LoadDeviceFunctions()
        {
            _vkGetDeviceQueue = GetDeviceProc<GetDeviceQueueDelegate>("vkGetDeviceQueue");
            _vkDeviceWaitIdle = GetDeviceProc<DeviceWaitIdleDelegate>("vkDeviceWaitIdle");
            _vkCreateSwapchainKHR = GetDeviceProc<CreateSwapchainKHRDelegate>("vkCreateSwapchainKHR");
            _vkDestroySwapchainKHR = GetDeviceProc<DestroySwapchainKHRDelegate>("vkDestroySwapchainKHR");
            _vkAcquireNextImageKHR = GetDeviceProc<AcquireNextImageKHRDelegate>("vkAcquireNextImageKHR");
            _vkQueuePresentKHR = GetDeviceProc<QueuePresentKHRDelegate>("vkQueuePresentKHR");
            _vkQueueSubmit = GetDeviceProc<QueueSubmitDelegate>("vkQueueSubmit");
            _vkQueueWaitIdle = GetDeviceProc<QueueWaitIdleDelegate>("vkQueueWaitIdle");
            _vkCreateFence = GetDeviceProc<CreateFenceDelegate>("vkCreateFence");
            _vkDestroyFence = GetDeviceProc<DestroyFenceDelegate>("vkDestroyFence");
            _vkWaitForFences = GetDeviceProc<WaitForFencesDelegate>("vkWaitForFences");
            _vkResetFences = GetDeviceProc<ResetFencesDelegate>("vkResetFences");
            _vkCreateSemaphore = GetDeviceProc<CreateSemaphoreDelegate>("vkCreateSemaphore");
            _vkDestroySemaphore = GetDeviceProc<DestroySemaphoreDelegate>("vkDestroySemaphore");
            _vkCreateCommandPool = GetDeviceProc<CreateCommandPoolDelegate>("vkCreateCommandPool");
            _vkDestroyCommandPool = GetDeviceProc<DestroyCommandPoolDelegate>("vkDestroyCommandPool");
            _vkAllocateCommandBuffers = GetDeviceProc<AllocateCommandBuffersDelegate>("vkAllocateCommandBuffers");
            _vkFreeCommandBuffers = GetDeviceProc<FreeCommandBuffersDelegate>("vkFreeCommandBuffers");
            _vkBeginCommandBuffer = GetDeviceProc<BeginCommandBufferDelegate>("vkBeginCommandBuffer");
            _vkEndCommandBuffer = GetDeviceProc<EndCommandBufferDelegate>("vkEndCommandBuffer");
            _vkCmdBeginRenderPass = GetDeviceProc<CmdBeginRenderPassDelegate>("vkCmdBeginRenderPass");
            _vkCmdEndRenderPass = GetDeviceProc<CmdEndRenderPassDelegate>("vkCmdEndRenderPass");
            _vkCmdBindPipeline = GetDeviceProc<CmdBindPipelineDelegate>("vkCmdBindPipeline");
            _vkCmdDraw = GetDeviceProc<CmdDrawDelegate>("vkCmdDraw");
            _vkCmdDrawIndexed = GetDeviceProc<CmdDrawIndexedDelegate>("vkCmdDrawIndexed");
            _vkCmdDispatch = GetDeviceProc<CmdDispatchDelegate>("vkCmdDispatch");
            _vkCmdPipelineBarrier = GetDeviceProc<CmdPipelineBarrierDelegate>("vkCmdPipelineBarrier");
            _vkCmdCopyBuffer = GetDeviceProc<CmdCopyBufferDelegate>("vkCmdCopyBuffer");
            _vkCmdCopyBufferToImage = GetDeviceProc<CmdCopyBufferToImageDelegate>("vkCmdCopyBufferToImage");
            _vkCmdPushConstants = GetDeviceProc<CmdPushConstantsDelegate>("vkCmdPushConstants");
            _vkCmdSetViewport = GetDeviceProc<CmdSetViewportDelegate>("vkCmdSetViewport");
            _vkCmdSetScissor = GetDeviceProc<CmdSetScissorDelegate>("vkCmdSetScissor");
            _vkCmdBindVertexBuffers = GetDeviceProc<CmdBindVertexBuffersDelegate>("vkCmdBindVertexBuffers");
            _vkCmdBindIndexBuffer = GetDeviceProc<CmdBindIndexBufferDelegate>("vkCmdBindIndexBuffer");
            _vkCreateBuffer = GetDeviceProc<CreateBufferDelegate>("vkCreateBuffer");
            _vkDestroyBuffer = GetDeviceProc<DestroyBufferDelegate>("vkDestroyBuffer");
            _vkGetBufferMemoryRequirements = GetDeviceProc<GetBufferMemoryRequirementsDelegate>("vkGetBufferMemoryRequirements");
            _vkBindBufferMemory = GetDeviceProc<BindBufferMemoryDelegate>("vkBindBufferMemory");
            _vkAllocateMemory = GetDeviceProc<AllocateMemoryDelegate>("vkAllocateMemory");
            _vkFreeMemory = GetDeviceProc<FreeMemoryDelegate>("vkFreeMemory");
            _vkMapMemory = GetDeviceProc<MapMemoryDelegate>("vkMapMemory");
            _vkUnmapMemory = GetDeviceProc<UnmapMemoryDelegate>("vkUnmapMemory");
            _vkFlushMappedMemoryRanges = GetDeviceProc<FlushMappedMemoryRangesDelegate>("vkFlushMappedMemoryRanges");
            _vkCreateImage = GetDeviceProc<CreateImageDelegate>("vkCreateImage");
            _vkDestroyImage = GetDeviceProc<DestroyImageDelegate>("vkDestroyImage");
            _vkGetImageMemoryRequirements = GetDeviceProc<GetImageMemoryRequirementsDelegate>("vkGetImageMemoryRequirements");
            _vkBindImageMemory = GetDeviceProc<BindImageMemoryDelegate>("vkBindImageMemory");
            _vkCreateImageView = GetDeviceProc<CreateImageViewDelegate>("vkCreateImageView");
            _vkDestroyImageView = GetDeviceProc<DestroyImageViewDelegate>("vkDestroyImageView");
            _vkCreateSampler = GetDeviceProc<CreateSamplerDelegate>("vkCreateSampler");
            _vkDestroySampler = GetDeviceProc<DestroySamplerDelegate>("vkDestroySampler");
            _vkCreateRenderPass = GetDeviceProc<CreateRenderPassDelegate>("vkCreateRenderPass");
            _vkDestroyRenderPass = GetDeviceProc<DestroyRenderPassDelegate>("vkDestroyRenderPass");
            _vkCreateFramebuffer = GetDeviceProc<CreateFramebufferDelegate>("vkCreateFramebuffer");
            _vkDestroyFramebuffer = GetDeviceProc<DestroyFramebufferDelegate>("vkDestroyFramebuffer");
            _vkCreateShaderModule = GetDeviceProc<CreateShaderModuleDelegate>("vkCreateShaderModule");
            _vkDestroyShaderModule = GetDeviceProc<DestroyShaderModuleDelegate>("vkDestroyShaderModule");
            _vkCreateGraphicsPipelines = GetDeviceProc<CreateGraphicsPipelinesDelegate>("vkCreateGraphicsPipelines");
            _vkCreateComputePipelines = GetDeviceProc<CreateComputePipelinesDelegate>("vkCreateComputePipelines");
            _vkDestroyPipeline = GetDeviceProc<DestroyPipelineDelegate>("vkDestroyPipeline");
            _vkCreatePipelineLayout = GetDeviceProc<CreatePipelineLayoutDelegate>("vkCreatePipelineLayout");
            _vkDestroyPipelineLayout = GetDeviceProc<DestroyPipelineLayoutDelegate>("vkDestroyPipelineLayout");
            _vkCreateDescriptorSetLayout = GetDeviceProc<CreateDescriptorSetLayoutDelegate>("vkCreateDescriptorSetLayout");
            _vkDestroyDescriptorSetLayout = GetDeviceProc<DestroyDescriptorSetLayoutDelegate>("vkDestroyDescriptorSetLayout");
            _vkCreateDescriptorPool = GetDeviceProc<CreateDescriptorPoolDelegate>("vkCreateDescriptorPool");
            _vkDestroyDescriptorPool = GetDeviceProc<DestroyDescriptorPoolDelegate>("vkDestroyDescriptorPool");
            _vkAllocateDescriptorSets = GetDeviceProc<AllocateDescriptorSetsDelegate>("vkAllocateDescriptorSets");
            _vkFreeDescriptorSets = GetDeviceProc<FreeDescriptorSetsDelegate>("vkFreeDescriptorSets");
            _vkUpdateDescriptorSets = GetDeviceProc<UpdateDescriptorSetsDelegate>("vkUpdateDescriptorSets");
            _vkCreatePipelineCache = GetDeviceProc<CreatePipelineCacheDelegate>("vkCreatePipelineCache");
            _vkDestroyPipelineCache = GetDeviceProc<DestroyPipelineCacheDelegate>("vkDestroyPipelineCache");
            _vkGetPipelineCacheData = GetDeviceProc<GetPipelineCacheDataDelegate>("vkGetPipelineCacheData");
            _vkMergePipelineCaches = GetDeviceProc<MergePipelineCachesDelegate>("vkMergePipelineCaches");
            _vkCmdCopyImage = GetDeviceProc<CmdCopyImageDelegate>("vkCmdCopyImage");
            _vkCmdBlitImage = GetDeviceProc<CmdBlitImageDelegate>("vkCmdBlitImage");
            _vkCmdResolveImage = GetDeviceProc<CmdResolveImageDelegate>("vkCmdResolveImage");
            _vkCreateQueryPool = GetDeviceProc<CreateQueryPoolDelegate>("vkCreateQueryPool");
            _vkDestroyQueryPool = GetDeviceProc<DestroyQueryPoolDelegate>("vkDestroyQueryPool");
            _vkResetQueryPool = GetDeviceProc<ResetQueryPoolDelegate>("vkResetQueryPool");
            _vkGetQueryPoolResults = GetDeviceProc<GetQueryPoolResultsDelegate>("vkGetQueryPoolResults");
            _vkCmdWriteTimestamp = GetDeviceProc<CmdWriteTimestampDelegate>("vkCmdWriteTimestamp");
            _vkCmdResetQueryPool = GetDeviceProc<CmdResetQueryPoolDelegate>("vkCmdResetQueryPool");
            _vkCmdCopyQueryPoolResults = GetDeviceProc<CmdCopyQueryPoolResultsDelegate>("vkCmdCopyQueryPoolResults");
            _vkCmdBeginQuery = GetDeviceProc<CmdBeginQueryDelegate>("vkCmdBeginQuery");
            _vkCmdEndQuery = GetDeviceProc<CmdEndQueryDelegate>("vkCmdEndQuery");
            _vkCmdBeginRendering = GetDeviceProc<CmdBeginRenderingDelegate>("vkCmdBeginRendering");
            _vkCmdEndRendering = GetDeviceProc<CmdEndRenderingDelegate>("vkCmdEndRendering");
            _vkCmdSetDepthBiasEnable = GetDeviceProc<CmdSetDepthBiasEnableDelegate>("vkCmdSetDepthBiasEnable");
            _vkCmdSetPrimitiveRestartEnable = GetDeviceProc<CmdSetPrimitiveRestartEnableDelegate>("vkCmdSetPrimitiveRestartEnable");
            _vkCmdSetRasterizerDiscardEnable = GetDeviceProc<CmdSetRasterizerDiscardEnableDelegate>("vkCmdSetRasterizerDiscardEnable");
            _vkCreateRenderPass2 = GetDeviceProc<CreateRenderPass2Delegate>("vkCreateRenderPass2");
            _vkCmdBeginRenderPass2 = GetDeviceProc<CmdBeginRenderPass2Delegate>("vkCmdBeginRenderPass2");
            _vkCmdNextSubpass2 = GetDeviceProc<CmdNextSubpass2Delegate>("vkCmdNextSubpass2");
            _vkCmdEndRenderPass2 = GetDeviceProc<CmdEndRenderPass2Delegate>("vkCmdEndRenderPass2");
            _vkCmdSetDepthTestEnable = GetDeviceProc<CmdSetDepthTestEnableDelegate>("vkCmdSetDepthTestEnable");
            _vkCmdSetDepthWriteEnable = GetDeviceProc<CmdSetDepthWriteEnableDelegate>("vkCmdSetDepthWriteEnable");
            _vkCmdSetDepthCompareOp = GetDeviceProc<CmdSetDepthCompareOpDelegate>("vkCmdSetDepthCompareOp");
            _vkCmdSetDepthBoundsTestEnable = GetDeviceProc<CmdSetDepthBoundsTestEnableDelegate>("vkCmdSetDepthBoundsTestEnable");
            _vkCmdSetStencilTestEnable = GetDeviceProc<CmdSetStencilTestEnableDelegate>("vkCmdSetStencilTestEnable");
            _vkCmdSetStencilOp = GetDeviceProc<CmdSetStencilOpDelegate>("vkCmdSetStencilOp");
        }
        private void PickPhysicalDevice()
        {
            uint deviceCount = 0;
            vkEnumeratePhysicalDevices(_instance, ref deviceCount, IntPtr.Zero);
            if (deviceCount == 0)
                throw new InvalidOperationException("No Vulkan physical devices found");

            var devicesPtr = Marshal.AllocHGlobal((int)(deviceCount * IntPtr.Size));
            try
            {
                vkEnumeratePhysicalDevices(_instance, ref deviceCount, devicesPtr);
                _physicalDevice = Marshal.ReadIntPtr(devicesPtr);
            }
            finally { Marshal.FreeHGlobal(devicesPtr); }
            _device.PhysicalDevice = _physicalDevice;

            // Get device properties
            var properties = new VkPhysicalDeviceProperties();
            var propertiesPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPhysicalDeviceProperties>());
            try
            {
                vkGetPhysicalDeviceProperties(_physicalDevice, propertiesPtr);
                properties = Marshal.PtrToStructure<VkPhysicalDeviceProperties>(propertiesPtr);
                _physicalDeviceInfo.Handle = _physicalDevice;
                _physicalDeviceInfo.DeviceName = Marshal.PtrToStringAnsi((IntPtr)((long)propertiesPtr + 16)) ?? "Unknown";
                _physicalDeviceInfo.VendorId = properties.vendorID;
                _physicalDeviceInfo.DeviceId = properties.deviceID;
                _physicalDeviceInfo.ApiVersion = properties.apiVersion;
                _physicalDeviceInfo.DriverVersion = properties.driverVersion;
                _device.PhysicalDeviceInfo = _physicalDeviceInfo;
            }
            finally { Marshal.FreeHGlobal(propertiesPtr); }

            // Get queue family properties
            uint queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref queueFamilyCount, IntPtr.Zero);
            var queueFamilyProperties = new VkQueueFamilyProperties[queueFamilyCount];
            var qfpPtr = Marshal.AllocHGlobal((int)(queueFamilyCount * Marshal.SizeOf<VkQueueFamilyProperties>()));
            try
            {
                vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref queueFamilyCount, qfpPtr);
                for (int i = 0; i < queueFamilyCount; i++)
                    queueFamilyProperties[i] = Marshal.PtrToStructure<VkQueueFamilyProperties>(qfpPtr + i * Marshal.SizeOf<VkQueueFamilyProperties>());
            }
            finally { Marshal.FreeHGlobal(qfpPtr); }

            _queueFamilyIndices = new QueueFamilyIndices();
            for (uint i = 0; i < queueFamilyCount; i++)
            {
                var props = queueFamilyProperties[i];
                if ((props.queueFlags & QueueFlag.Graphics) != 0)
                    _queueFamilyIndices.GraphicsFamily = (int)i;
                if ((props.queueFlags & QueueFlag.Compute) != 0 && _queueFamilyIndices.ComputeFamily < 0)
                    _queueFamilyIndices.ComputeFamily = (int)i;
                if ((props.queueFlags & QueueFlag.Transfer) != 0 && _queueFamilyIndices.TransferFamily < 0)
                    _queueFamilyIndices.TransferFamily = (int)i;

                if (_surface != IntPtr.Zero)
                {
                    IntPtr supported = IntPtr.Zero;
                    vkGetPhysicalDeviceSurfaceSupport(_physicalDevice, i, _surface, ref supported);
                    if (supported != IntPtr.Zero)
                        _queueFamilyIndices.PresentFamily = (int)i;
                }
            }
            _device.QueueIndices = _queueFamilyIndices;

            // Get memory properties
            var memProps = new VkPhysicalDeviceMemoryProperties();
            var memPropsPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPhysicalDeviceMemoryProperties>());
            try
            {
                vkGetPhysicalDeviceMemoryProperties(_physicalDevice, memPropsPtr);
                memProps = Marshal.PtrToStructure<VkPhysicalDeviceMemoryProperties>(memPropsPtr);
                _physicalDeviceInfo.MemoryProperties = new PhysicalDeviceMemoryProperties
                {
                    MemoryTypeCount = memProps.memoryTypeCount,
                    MemoryTypes = new MemoryType[memProps.memoryTypeCount],
                    MemoryHeapCount = memProps.memoryHeapCount,
                    MemoryHeaps = new MemoryHeap[memProps.memoryHeapCount]
                };
                for (int i = 0; i < memProps.memoryTypeCount; i++)
                {
                    var mt = Marshal.PtrToStructure<VkMemoryType>(memPropsPtr + 4 + i * Marshal.SizeOf<VkMemoryType>());
                    _physicalDeviceInfo.MemoryProperties.MemoryTypes[i] = new MemoryType { PropertyFlags = mt.propertyFlags, HeapIndex = mt.heapIndex };
                }
                for (int i = 0; i < memProps.memoryHeapCount; i++)
                {
                    var mh = Marshal.PtrToStructure<VkMemoryHeap>(memPropsPtr + 4 + (int)(memProps.memoryTypeCount * Marshal.SizeOf<VkMemoryType>()) + i * Marshal.SizeOf<VkMemoryHeap>());
                    _physicalDeviceInfo.MemoryProperties.MemoryHeaps[i] = new MemoryHeap { Size = mh.size, Flags = mh.flags };
                }
            }
            finally { Marshal.FreeHGlobal(memPropsPtr); }
        }

        private void CreateLogicalDevice()
        {
            var queueCreateInfos = new List<VkDeviceQueueCreateInfo>();
            var uniqueFamilies = _queueFamilyIndices.UniqueQueueFamilies;
            float queuePriority = 1.0f;
            var priorityPtr = Marshal.AllocHGlobal(sizeof(float));
            MarshalCompat.WriteFloat(priorityPtr, queuePriority);

            foreach (var family in uniqueFamilies)
            {
                var qci = new VkDeviceQueueCreateInfo
                {
                    sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex = family,
                    queueCount = 1,
                    pQueuePriorities = priorityPtr
                };
                queueCreateInfos.Add(qci);
            }

            var deviceFeatures = new VkPhysicalDeviceFeatures();
            deviceFeatures.samplerAnisotropy = VulkanBool32.True;
            deviceFeatures.fillModeNonSolid = VulkanBool32.True;
            deviceFeatures.wideLines = VulkanBool32.True;
            deviceFeatures.largePoints = VulkanBool32.True;
            deviceFeatures.shaderImageGatherExtended = VulkanBool32.True;
            deviceFeatures.samplerLodClamp = VulkanBool32.True;

            var featuresPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPhysicalDeviceFeatures>());
            Marshal.StructureToPtr(deviceFeatures, featuresPtr, false);

            var extensions = new List<string>(_creationInfo.RequiredExtensions);
            extensions.Add("VK_KHR_swapchain");
            extensions.Add("VK_KHR_maintenance1");
            extensions.Add("VK_KHR_maintenance2");
            extensions.Add("VK_KHR_maintenance3");
            extensions.Add("VK_KHR_maintenance4");
            extensions.Add("VK_KHR_synchronization2");
            extensions.Add("VK_KHR_dynamic_rendering");
            extensions.Add("VK_KHR_depth_stencil_resolve");
            extensions.Add("VK_KHR_create_renderpass2");
            extensions.Add("VK_EXT_descriptor_indexing");
            extensions.Add("VK_EXT_subgroup_size_control");
            extensions.Add("VK_KHR_8bit_storage");
            extensions.Add("VK_KHR_16bit_storage");
            extensions.Add("VK_KHR_shader_float16_int8");

            var extensionPtrs = extensions.Select(e => Marshal.StringToHGlobalAnsi(e)).ToArray();
            var layerNames = new List<string>();
            if (_creationInfo.EnableValidation)
                layerNames.Add("VK_LAYER_KHRONOS_validation");
            var layerPtrs = layerNames.Select(l => Marshal.StringToHGlobalAnsi(l)).ToArray();

            var createInfo = new VkDeviceCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                queueCreateInfoCount = (uint)queueCreateInfos.Count,
                enabledExtensionCount = (uint)extensionPtrs.Length,
                enabledLayerCount = (uint)layerPtrs.Length,
                pEnabledFeatures = featuresPtr
            };

            unsafe
            {
                var queueInfoArray = queueCreateInfos.ToArray();
                createInfo.pQueueCreateInfos = Marshal.AllocHGlobal(queueInfoArray.Length * Marshal.SizeOf<VkDeviceQueueCreateInfo>());
                for (int i = 0; i < queueInfoArray.Length; i++)
                    Marshal.StructureToPtr(queueInfoArray[i], createInfo.pQueueCreateInfos + i * Marshal.SizeOf<VkDeviceQueueCreateInfo>(), false);

                createInfo.ppEnabledExtensionNames = Marshal.AllocHGlobal(extensionPtrs.Length * IntPtr.Size);
                createInfo.ppEnabledLayerNames = Marshal.AllocHGlobal(layerPtrs.Length * IntPtr.Size);
                for (int i = 0; i < extensionPtrs.Length; i++)
                    Marshal.WriteIntPtr(createInfo.ppEnabledExtensionNames + i * IntPtr.Size, extensionPtrs[i]);
                for (int i = 0; i < layerPtrs.Length; i++)
                    Marshal.WriteIntPtr(createInfo.ppEnabledLayerNames + i * IntPtr.Size, layerPtrs[i]);
            }

            var result = vkCreateDevice(_physicalDevice, ref createInfo, IntPtr.Zero, ref _logicalDevice);
            Marshal.FreeHGlobal(featuresPtr);
            Marshal.FreeHGlobal(priorityPtr);
            foreach (var ptr in extensionPtrs)
                Marshal.FreeHGlobal(ptr);
            foreach (var ptr in layerPtrs)
                Marshal.FreeHGlobal(ptr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create Vulkan logical device: {result}");

            _device.LogicalDevice = _logicalDevice;

            _vkGetDeviceProcAddr = _device.GetDeviceProcAddr("vkGetDeviceProcAddr");
            LoadDeviceFunctions();
        }

        private void InitializeQueues()
        {
            _vkGetDeviceQueue = GetDeviceProc<GetDeviceQueueDelegate>("vkGetDeviceQueue");

            if (_queueFamilyIndices.GraphicsFamily >= 0)
            {
                _vkGetDeviceQueue(_logicalDevice, (uint)_queueFamilyIndices.GraphicsFamily, 0, ref _graphicsQueue);
                _device.GraphicsQueue = _graphicsQueue;
            }
            if (_queueFamilyIndices.ComputeFamily >= 0)
            {
                _vkGetDeviceQueue(_logicalDevice, (uint)_queueFamilyIndices.ComputeFamily, 0, ref _computeQueue);
                _device.ComputeQueue = _computeQueue;
            }
            if (_queueFamilyIndices.TransferFamily >= 0)
            {
                _vkGetDeviceQueue(_logicalDevice, (uint)_queueFamilyIndices.TransferFamily, 0, ref _transferQueue);
                _device.TransferQueue = _transferQueue;
            }
            if (_queueFamilyIndices.PresentFamily >= 0)
            {
                _vkGetDeviceQueue(_logicalDevice, (uint)_queueFamilyIndices.PresentFamily, 0, ref _presentQueue);
                _device.PresentQueue = _presentQueue;
            }
        }

        private void CreatePipelineCacheInternal()
        {
            var createInfo = new VkPipelineCacheCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
            };
            _vkCreatePipelineCache(_logicalDevice, ref createInfo, IntPtr.Zero, ref _pipelineCache);
            _device.PipelineCache = _pipelineCacheWrapper;
        }

        private void InitializeSubsystems()
        {
            _memoryAllocator = new VulkanMemoryAllocator(_device);
            _device.MemoryAllocator = _memoryAllocator;

            _syncManager = new VulkanSyncManager(_device);
            _device.SyncManager = _syncManager;

            _descriptorManager = new VulkanDescriptorManager(_device);
            _device.DescriptorManager = _descriptorManager;

            _pipelineCacheWrapper = new VulkanPipelineCache(_device, _pipelineCache);
            _device.PipelineCache = _pipelineCacheWrapper;

            _resourceTracker = new VulkanResourceTracker(_device);
            _device.ResourceTracker = _resourceTracker;

            // Create default command pool
            var poolCreateInfo = new VkCommandPoolCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                flags = (uint)CommandPoolCreateFlag.ResetCommandBuffer,
                queueFamilyIndex = (uint)_queueFamilyIndices.GraphicsFamily
            };
            _vkCreateCommandPool(_logicalDevice, ref poolCreateInfo, IntPtr.Zero, ref _commandPool);
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkPhysicalDeviceProperties
        {
            public uint deviceType; public uint vendorID; public uint deviceID;
            public uint apiVersion; public uint driverVersion;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkQueueFamilyProperties
        {
            public QueueFlag queueFlags; public uint queueCount;
            public uint timestampValidBits;
            public uint minImageTransferGranularityX; public uint minImageTransferGranularityY; public uint minImageTransferGranularityZ;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkPhysicalDeviceMemoryProperties
        {
            public uint memoryTypeCount;
            public VkMemoryType memoryType0; public VkMemoryType memoryType1; public VkMemoryType memoryType2;
            public VkMemoryType memoryType3; public VkMemoryType memoryType4; public VkMemoryType memoryType5;
            public VkMemoryType memoryType6; public VkMemoryType memoryType7; public VkMemoryType memoryType8;
            public VkMemoryType memoryType9; public VkMemoryType memoryType10; public VkMemoryType memoryType11;
            public VkMemoryType memoryType12; public VkMemoryType memoryType13; public VkMemoryType memoryType14;
            public VkMemoryType memoryType15; public VkMemoryType memoryType16; public VkMemoryType memoryType17;
            public VkMemoryType memoryType18; public VkMemoryType memoryType19; public VkMemoryType memoryType20;
            public VkMemoryType memoryType21; public VkMemoryType memoryType22; public VkMemoryType memoryType23;
            public VkMemoryType memoryType24; public VkMemoryType memoryType25; public VkMemoryType memoryType26;
            public VkMemoryType memoryType27; public VkMemoryType memoryType28; public VkMemoryType memoryType29;
            public VkMemoryType memoryType30; public VkMemoryType memoryType31;
            public uint memoryHeapCount;
            public VkMemoryHeap heap0; public VkMemoryHeap heap1; public VkMemoryHeap heap2;
            public VkMemoryHeap heap3; public VkMemoryHeap heap4; public VkMemoryHeap heap5;
            public VkMemoryHeap heap6; public VkMemoryHeap heap7; public VkMemoryHeap heap8;
            public VkMemoryHeap heap9; public VkMemoryHeap heap10; public VkMemoryHeap heap11;
            public VkMemoryHeap heap12; public VkMemoryHeap heap13; public VkMemoryHeap heap14;
            public VkMemoryHeap heap15;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkMemoryType { public MemoryPropertyFlag propertyFlags; public uint heapIndex; }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkMemoryHeap { public ulong size; public MemoryHeapFlag flags; }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkPhysicalDeviceFeatures
        {
            public uint sType; public VulkanBool32 samplerAnisotropy; public VulkanBool32 fillModeNonSolid;
            public VulkanBool32 wideLines; public VulkanBool32 largePoints;
            public VulkanBool32 shaderImageGatherExtended; public VulkanBool32 samplerLodClamp;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkDeviceQueueCreateInfo
        {
            public uint sType; public IntPtr pNext; public uint flags;
            public uint queueFamilyIndex; public uint queueCount; public IntPtr pQueuePriorities;
        }

        private VkBool32 DebugUtilsCallbackNative(uint messageSeverity, uint messageTypes, IntPtr pCallbackData, IntPtr pUserData) => DebugUtilsCallback(messageSeverity, messageTypes, pCallbackData, pUserData);
        // =====================================================================
        // PUBLIC PROPERTIES
        // =====================================================================
        public IntPtr PhysicalDevice => _physicalDevice;
        public IntPtr Device => _logicalDevice;
        public IntPtr Instance => _instance;
        public IntPtr Surface => _surface;
        public IntPtr Queue => _graphicsQueue;
        public VulkanSwapchain Swapchain => _swapchain;

        // =====================================================================
        // PUBLIC API METHODS
        // =====================================================================

        /// <summary>Creates a Vulkan swapchain for the given surface</summary>
        public VulkanSwapchain CreateSwapchain(IntPtr surface)
        {
            _surface = surface;
            _device.Surface = surface;

            _queueFamilyIndices = new QueueFamilyIndices();
            FindQueueFamiliesForSurface(surface);

            var supportDetails = QuerySwapchainSupport(surface);
            var surfaceFormat = supportDetails.ChooseSurfaceFormat();
            var presentMode = supportDetails.ChoosePresentMode(PresentMode.Mailbox);
            var extent = supportDetails.ChooseExtent(1280, 720);

            uint imageCount = supportDetails.Capabilities.MinImageCount + 1;
            if (supportDetails.Capabilities.MaxImageCount > 0 && imageCount > supportDetails.Capabilities.MaxImageCount)
                imageCount = supportDetails.Capabilities.MaxImageCount;

            var createInfo = new VkSwapchainCreateInfoKHR
            {
                sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                surface = surface,
                minImageCount = imageCount,
                imageFormat = surfaceFormat.Format,
                imageColorSpace = (uint)surfaceFormat.ColorSpace,
                imageExtent = new VkExtent2D { width = extent.Width, height = extent.Height },
                imageArrayLayers = 1,
                imageUsage = ImageUsageFlag.ColorAttachment | ImageUsageFlag.TransferDst,
                preTransform = supportDetails.Capabilities.CurrentTransform,
                compositeAlpha = CompositeAlphaFlag.Opaque,
                presentMode = presentMode,
                clipped = VulkanBool32.True,
                oldSwapchain = _swapchain?.Handle ?? IntPtr.Zero
            };

            if (_queueFamilyIndices.GraphicsFamily != _queueFamilyIndices.PresentFamily)
            {
                var queueFamilyIndices = new uint[] { (uint)_queueFamilyIndices.GraphicsFamily, (uint)_queueFamilyIndices.PresentFamily };
                createInfo.imageSharingMode = SharingMode.Concurrent;
                createInfo.queueFamilyIndexCount = 2;
                unsafe
                {
                    fixed (uint* pIndices = queueFamilyIndices)
                        createInfo.pQueueFamilyIndices = (IntPtr)pIndices;
                }
            }
            else
            {
                createInfo.imageSharingMode = SharingMode.Exclusive;
            }

            IntPtr swapchain = IntPtr.Zero;
            var result = _vkCreateSwapchainKHR(_logicalDevice, ref createInfo, IntPtr.Zero, ref swapchain);
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create swapchain: {result}");

            var vulkanSwapchain = new VulkanSwapchain(_device, swapchain, surfaceFormat.Format, presentMode, extent);
            _swapchain = vulkanSwapchain;
            return vulkanSwapchain;
        }

        private void FindQueueFamiliesForSurface(IntPtr surface)
        {
            uint queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref queueFamilyCount, IntPtr.Zero);
            var qfpPtr = Marshal.AllocHGlobal((int)(queueFamilyCount * 48));
            try
            {
                vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref queueFamilyCount, qfpPtr);
                for (uint i = 0; i < queueFamilyCount; i++)
                {
                    var qfp = Marshal.PtrToStructure<VkQueueFamilyProperties>(qfpPtr + (int)(i * 48));
                    if ((qfp.queueFlags & QueueFlag.Graphics) != 0)
                        _queueFamilyIndices.GraphicsFamily = (int)i;
                    if ((qfp.queueFlags & QueueFlag.Compute) != 0 && _queueFamilyIndices.ComputeFamily < 0)
                        _queueFamilyIndices.ComputeFamily = (int)i;

                    IntPtr supported = new IntPtr(1);
                    vkGetPhysicalDeviceSurfaceSupport(_physicalDevice, i, surface, ref supported);
                    if (supported != IntPtr.Zero)
                        _queueFamilyIndices.PresentFamily = (int)i;
                }
            }
            finally { Marshal.FreeHGlobal(qfpPtr); }
        }

        /// <summary>Queries swapchain support details for a surface</summary>
        public VulkanSwapchainSupportDetails QuerySwapchainSupport(IntPtr surface)
        {
            var details = new VulkanSwapchainSupportDetails();

            // Capabilities
            var capsPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkSurfaceCapabilitiesKHR>());
            try
            {
                vkGetPhysicalDeviceSurfaceCapabilities(_physicalDevice, surface, capsPtr);
                var caps = Marshal.PtrToStructure<VkSurfaceCapabilitiesKHR>(capsPtr);
                details.Capabilities = new SurfaceCapabilities
                {
                    MinImageCount = caps.minImageCount,
                    MaxImageCount = caps.maxImageCount,
                    CurrentExtent = new Extent2D(caps.currentImageWidth, caps.currentImageHeight),
                    MinImageExtent = new Extent2D(caps.minImageWidth, caps.minImageHeight),
                    MaxImageExtent = new Extent2D(caps.maxImageWidth, caps.maxImageHeight),
                    MaxImageArrayLayers = caps.maxImageArrayLayers,
                    SupportedTransforms = caps.supportedTransforms,
                    CurrentTransform = caps.currentTransform,
                    SupportedCompositeAlpha = caps.supportedCompositeAlpha,
                    SupportedUsageFlags = caps.supportedUsageFlags
                };
            }
            finally { Marshal.FreeHGlobal(capsPtr); }

            // Formats
            uint formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormats(_physicalDevice, surface, ref formatCount, IntPtr.Zero);
            if (formatCount > 0)
            {
                var formatPtr = Marshal.AllocHGlobal((int)(formatCount * 8));
                try
                {
                    vkGetPhysicalDeviceSurfaceFormats(_physicalDevice, surface, ref formatCount, formatPtr);
                    details.Formats = new SurfaceFormatKHR[formatCount];
                    for (int i = 0; i < formatCount; i++)
                    {
                        var fmt = Marshal.PtrToStructure<SurfaceFormatKHR>(formatPtr + i * 8);
                        details.Formats[i] = fmt;
                    }
                }
                finally { Marshal.FreeHGlobal(formatPtr); }
            }

            // Present modes
            uint presentModeCount = 0;
            vkGetPhysicalDeviceSurfacePresentModes(_physicalDevice, surface, ref presentModeCount, IntPtr.Zero);
            if (presentModeCount > 0)
            {
                var modePtr = Marshal.AllocHGlobal((int)(presentModeCount * 4));
                try
                {
                    vkGetPhysicalDeviceSurfacePresentModes(_physicalDevice, surface, ref presentModeCount, modePtr);
                    details.PresentModes = new PresentMode[presentModeCount];
                    for (int i = 0; i < presentModeCount; i++)
                        details.PresentModes[i] = (PresentMode)Marshal.ReadInt32(modePtr + i * 4);
                }
                finally { Marshal.FreeHGlobal(modePtr); }
            }

            return details;
        }

        /// <summary>Returns queue family indices</summary>
        public QueueFamilyIndices GetQueueFamilies() => _queueFamilyIndices;

        /// <summary>Creates a GPU buffer</summary>
        public VulkanBuffer CreateBuffer(BufferDescription description)
        {
            var createInfo = new VkBufferCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size = description.Size,
                usage = description.Usage,
                sharingMode = description.SharingMode
            };

            IntPtr buffer = IntPtr.Zero;
            var result = _vkCreateBuffer(_logicalDevice, ref createInfo, IntPtr.Zero, ref buffer);
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create buffer: {result}");

            _vkGetBufferMemoryRequirements(_logicalDevice, buffer, out var memReqs);

            var vulkanBuffer = new VulkanBuffer(_device, buffer, description, memReqs);
            _memoryAllocator.AllocateBuffer(vulkanBuffer, description.MemoryProperties);
            _buffers.Add(vulkanBuffer);
            _resourceTracker.TrackResource(vulkanBuffer);
            return vulkanBuffer;
        }

        /// <summary>Creates a 2D/3D texture</summary>
        public VulkanTexture CreateTexture(TextureDescription description)
        {
            var extent3D = new VkExtent3D { width = description.Width, height = description.Height, depth = description.Depth };

            var createInfo = new VkImageCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                flags = 0,
                imageType = description.Type,
                format = description.Format,
                extent = extent3D,
                mipLevels = description.MipLevels,
                arrayLayers = description.ArrayLayers,
                samples = description.Samples,
                tiling = description.Tiling,
                usage = description.Usage,
                sharingMode = description.SharingMode,
                initialLayout = description.InitialLayout
            };

            IntPtr image = IntPtr.Zero;
            var result = _vkCreateImage(_logicalDevice, ref createInfo, IntPtr.Zero, ref image);
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create image: {result}");

            _vkGetImageMemoryRequirements(_logicalDevice, image, out var memReqs);

            var vulkanTexture = new VulkanTexture(_device, image, description, memReqs);
            _memoryAllocator.AllocateImage(vulkanTexture, MemoryPropertyFlag.DeviceLocal);
            _textures.Add(vulkanTexture);
            _resourceTracker.TrackResource(vulkanTexture);
            return vulkanTexture;
        }

        /// <summary>Creates a render pass from description</summary>
        public VulkanRenderPass CreateRenderPass(RenderPassDescription description)
        {
            uint attachmentCount = description.Attachments != null ? (uint)description.Attachments.Length : 0;
            uint subpassCount = description.Subpasses != null ? (uint)description.Subpasses.Length : 0;
            uint dependencyCount = description.Dependencies != null ? (uint)description.Dependencies.Length : 0;

            IntPtr renderPass = IntPtr.Zero;
            var createInfo = new VkRenderPassCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                attachmentCount = attachmentCount,
                subpassCount = subpassCount,
                dependencyCount = dependencyCount
            };

            // Marshal attachment descriptions
            IntPtr attachmentPtr = IntPtr.Zero;
            if (attachmentCount > 0)
            {
                int structSize = Marshal.SizeOf<VkAttachmentDescription>();
                attachmentPtr = Marshal.AllocHGlobal((int)(attachmentCount * structSize));
                for (int i = 0; i < description.Attachments.Length; i++)
                {
                    var src = description.Attachments[i];
                    var vkAtt = new VkAttachmentDescription
                    {
                        format = src.Format,
                        samples = src.Samples,
                        loadOp = src.LoadOp,
                        storeOp = src.StoreOp,
                        stencilLoadOp = src.StencilLoadOp,
                        stencilStoreOp = src.StencilStoreOp,
                        initialLayout = src.InitialLayout,
                        finalLayout = src.FinalLayout
                    };
                    Marshal.StructureToPtr(vkAtt, attachmentPtr + i * structSize, false);
                }
                createInfo.pAttachments = attachmentPtr;
            }

            // Marshal subpass descriptions
            IntPtr subpassPtr = IntPtr.Zero;
            var subpassDataList = new List<VkSubpassDescription>();
            var colorAttachRefs = new List<IntPtr>();
            var inputAttachRefs = new List<IntPtr>();
            var depthAttachRef = IntPtr.Zero;
            IntPtr resolveAttachPtr = IntPtr.Zero;

            if (subpassCount > 0)
            {
                for (int s = 0; s < description.Subpasses.Length; s++)
                {
                    var sub = description.Subpasses[s];
                    IntPtr colorRefPtr = IntPtr.Zero;
                    IntPtr inputRefPtr = IntPtr.Zero;
                    int structSize = Marshal.SizeOf<VkAttachmentReference>();

                    if (sub.ColorAttachments != null && sub.ColorAttachments.Length > 0)
                    {
                        colorRefPtr = Marshal.AllocHGlobal(sub.ColorAttachments.Length * structSize);
                        for (int i = 0; i < sub.ColorAttachments.Length; i++)
                        {
                            var aref = new VkAttachmentReference { attachment = sub.ColorAttachments[i].Attachment, layout = sub.ColorAttachments[i].Layout };
                            Marshal.StructureToPtr(aref, colorRefPtr + i * structSize, false);
                        }
                    }
                    colorAttachRefs.Add(colorRefPtr);

                    if (sub.InputAttachments != null && sub.InputAttachments.Length > 0)
                    {
                        inputRefPtr = Marshal.AllocHGlobal(sub.InputAttachments.Length * structSize);
                        for (int i = 0; i < sub.InputAttachments.Length; i++)
                        {
                            var aref = new VkAttachmentReference { attachment = sub.InputAttachments[i].Attachment, layout = sub.InputAttachments[i].Layout };
                            Marshal.StructureToPtr(aref, inputRefPtr + i * structSize, false);
                        }
                    }
                    inputAttachRefs.Add(inputRefPtr);

                    IntPtr depthRefPtr = IntPtr.Zero;
                    if (sub.DepthStencilAttachment.Attachment != 0 || sub.DepthStencilAttachment.Layout != 0)
                    {
                        depthRefPtr = Marshal.AllocHGlobal(structSize);
                        var depthRef = new VkAttachmentReference { attachment = sub.DepthStencilAttachment.Attachment, layout = sub.DepthStencilAttachment.Layout };
                        Marshal.StructureToPtr(depthRef, depthRefPtr, false);
                    }

                    var subDesc = new VkSubpassDescription
                    {
                        pipelineBindPoint = sub.PipelineBindPoint,
                        colorAttachmentCount = (uint)(sub.ColorAttachments?.Length ?? 0),
                        pColorAttachments = colorRefPtr,
                        inputAttachmentCount = (uint)(sub.InputAttachments?.Length ?? 0),
                        pInputAttachments = inputRefPtr,
                        pDepthStencilAttachment = depthRefPtr,
                        pResolveAttachments = resolveAttachPtr
                    };
                    subpassDataList.Add(subDesc);
                }

                int subpassStructSize = Marshal.SizeOf<VkSubpassDescription>();
                subpassPtr = Marshal.AllocHGlobal((int)(subpassCount * subpassStructSize));
                for (int i = 0; i < subpassDataList.Count; i++)
                    Marshal.StructureToPtr(subpassDataList[i], subpassPtr + i * subpassStructSize, false);
                createInfo.pSubpasses = subpassPtr;
            }

            // Marshal dependencies
            IntPtr dependencyPtr = IntPtr.Zero;
            if (dependencyCount > 0)
            {
                int depStructSize = Marshal.SizeOf<VkSubpassDependency>();
                dependencyPtr = Marshal.AllocHGlobal((int)(dependencyCount * depStructSize));
                for (int i = 0; i < description.Dependencies.Length; i++)
                {
                    var dep = description.Dependencies[i];
                    var vkDep = new VkSubpassDependency
                    {
                        srcSubpass = dep.SrcSubpass,
                        dstSubpass = dep.DstSubpass,
                        srcStageMask = dep.SrcStageMask,
                        dstStageMask = dep.DstStageMask,
                        srcAccessMask = dep.SrcAccessMask,
                        dstAccessMask = dep.DstAccessMask,
                        dependencyFlags = (uint)dep.DependencyFlags
                    };
                    Marshal.StructureToPtr(vkDep, dependencyPtr + i * depStructSize, false);
                }
                createInfo.pDependencies = dependencyPtr;
            }

            var res = _vkCreateRenderPass(_logicalDevice, ref createInfo, IntPtr.Zero, ref renderPass);
            if (res != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create render pass: {res}");

            // Free allocated memory
            if (attachmentPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(attachmentPtr);
            if (subpassPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(subpassPtr);
            if (dependencyPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(dependencyPtr);
            foreach (var ptr in colorAttachRefs)
                if (ptr != IntPtr.Zero)
                    Marshal.FreeHGlobal(ptr);
            foreach (var ptr in inputAttachRefs)
                if (ptr != IntPtr.Zero)
                    Marshal.FreeHGlobal(ptr);

            var rp = new VulkanRenderPass(_device, renderPass, description);
            _renderPasses.Add(rp);
            return rp;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkAttachmentDescription
        {
            public uint flags; public VulkanFormat format; public SampleCountFlag samples;
            public AttachmentLoadOp loadOp; public AttachmentStoreOp storeOp;
            public AttachmentLoadOp stencilLoadOp; public AttachmentStoreOp stencilStoreOp;
            public ImageLayout initialLayout; public ImageLayout finalLayout;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkAttachmentReference { public uint attachment; public ImageLayout layout; }
        /// <summary>Creates a framebuffer for a render pass</summary>
        public VulkanFramebuffer CreateFramebuffer(FramebufferDescription description)
        {
            IntPtr framebuffer = IntPtr.Zero;
            var createInfo = new VkFramebufferCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass = description.RenderPass,
                attachmentCount = (uint)(description.Attachments?.Length ?? 0),
                width = description.Width,
                height = description.Height,
                layers = description.Layers
            };

            if (description.Attachments != null && description.Attachments.Length > 0)
            {
                var attPtr = Marshal.AllocHGlobal(description.Attachments.Length * IntPtr.Size);
                for (int i = 0; i < description.Attachments.Length; i++)
                    Marshal.WriteIntPtr(attPtr + i * IntPtr.Size, description.Attachments[i]);
                createInfo.pAttachments = attPtr;

                var result = _vkCreateFramebuffer(_logicalDevice, ref createInfo, IntPtr.Zero, ref framebuffer);
                Marshal.FreeHGlobal(attPtr);
                if (result != VulkanResult.Success)
                    throw new InvalidOperationException($"Failed to create framebuffer: {result}");
            }
            else
            {
                var result = _vkCreateFramebuffer(_logicalDevice, ref createInfo, IntPtr.Zero, ref framebuffer);
                if (result != VulkanResult.Success)
                    throw new InvalidOperationException($"Failed to create framebuffer: {result}");
            }

            var fb = new VulkanFramebuffer(_device, framebuffer, description);
            _framebuffers.Add(fb);
            return fb;
        }

        /// <summary>Creates a graphics pipeline</summary>
        public VulkanPipeline CreatePipeline(PipelineDescription description)
        {
            // Marshal shader stages
            int stageSize = Marshal.SizeOf<PipelineShaderStageCreateInfo>();
            IntPtr stagesPtr = Marshal.AllocHGlobal(description.ShaderStages.Length * stageSize);
            for (int i = 0; i < description.ShaderStages.Length; i++)
                Marshal.StructureToPtr(description.ShaderStages[i], stagesPtr + i * stageSize, false);

            // Create sub-state structs
            IntPtr vertexInputPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineVertexInputStateCreateInfo>());
            IntPtr inputAssemblyPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineInputAssemblyStateCreateInfo>());
            IntPtr tessellationPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineTessellationStateCreateInfo>());
            IntPtr viewportPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineViewportStateCreateInfo>());
            IntPtr rasterizerPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineRasterizationStateCreateInfo>());
            IntPtr multisamplePtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineMultisampleStateCreateInfo>());
            IntPtr depthStencilPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineDepthStencilStateCreateInfo>());
            IntPtr colorBlendPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineColorBlendStateCreateInfo>());
            IntPtr dynamicStatePtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineDynamicStateCreateInfo>());

            // Vertex input
            var vis = description.VertexInputState;
            var vkVis = new VkPipelineVertexInputStateCreateInfo
            {
                sType = 1000027000,
                vertexBindingDescriptionCount = vis.VertexBindingDescriptions != null ? (uint)vis.VertexBindingDescriptions.Length : 0,
                vertexAttributeDescriptionCount = vis.VertexAttributeDescriptions != null ? (uint)vis.VertexAttributeDescriptions.Length : 0
            };
            Marshal.StructureToPtr(vkVis, vertexInputPtr, false);

            // Input assembly
            var ias = description.InputAssemblyState;
            var vkIas = new VkPipelineInputAssemblyStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                topology = ias.Topology,
                primitiveRestartEnable = ias.PrimitiveRestartEnable ? VulkanBool32.True : VulkanBool32.False
            };
            Marshal.StructureToPtr(vkIas, inputAssemblyPtr, false);

            // Tessellation
            var tess = description.TessellationState;
            var vkTess = new VkPipelineTessellationStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                patchControlPoints = tess.PatchControlPoints
            };
            Marshal.StructureToPtr(vkTess, tessellationPtr, false);

            // Viewport
            var vp = description.ViewportState;
            var vkVp = new VkPipelineViewportStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                viewportCount = vp.ViewportCount,
                scissorCount = vp.ScissorCount
            };
            Marshal.StructureToPtr(vkVp, viewportPtr, false);

            // Rasterizer
            var rs = description.RasterizationState;
            var vkRs = new VkPipelineRasterizationStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                depthClampEnable = rs.DepthClampEnable ? VulkanBool32.True : VulkanBool32.False,
                rasterizerDiscardEnable = rs.RasterizerDiscardEnable ? VulkanBool32.True : VulkanBool32.False,
                polygonMode = rs.PolygonMode,
                cullMode = rs.CullMode,
                frontFace = rs.FrontFace,
                depthBiasEnable = rs.DepthBiasEnable ? VulkanBool32.True : VulkanBool32.False,
                depthBiasConstantFactor = rs.DepthBiasConstantFactor,
                depthBiasClamp = rs.DepthBiasClamp,
                depthBiasSlopeFactor = rs.DepthBiasSlopeFactor,
                lineWidth = rs.LineWidth
            };
            Marshal.StructureToPtr(vkRs, rasterizerPtr, false);

            // Multisample
            var ms = description.MultisampleState;
            var vkMs = new VkPipelineMultisampleStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                rasterizationSamples = ms.RasterizationSamples,
                sampleShadingEnable = ms.SampleShadingEnable ? VulkanBool32.True : VulkanBool32.False,
                minSampleShading = ms.MinSampleShading,
                alphaToCoverageEnable = ms.AlphaToCoverageEnable ? VulkanBool32.True : VulkanBool32.False,
                alphaToOneEnable = ms.AlphaToOneEnable ? VulkanBool32.True : VulkanBool32.False
            };
            Marshal.StructureToPtr(vkMs, multisamplePtr, false);

            // Depth stencil
            var ds = description.DepthStencilState;
            var vkDs = new VkPipelineDepthStencilStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                depthTestEnable = ds.DepthTestEnable,
                depthWriteEnable = ds.DepthWriteEnable,
                depthCompareOp = ds.DepthCompareOp,
                depthBoundsTestEnable = ds.DepthBoundsTestEnable,
                stencilTestEnable = ds.StencilTestEnable,
                minDepthBounds = ds.MinDepthBounds,
                maxDepthBounds = ds.MaxDepthBounds
            };
            Marshal.StructureToPtr(vkDs, depthStencilPtr, false);

            // Color blend
            var cb = description.ColorBlendState;
            var vkCb = new VkPipelineColorBlendStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                logicOpEnable = cb.LogicOpEnable ? VulkanBool32.True : VulkanBool32.False,
                logicOp = cb.LogicOp,
                attachmentCount = cb.Attachments != null ? (uint)cb.Attachments.Length : 0
            };
            Marshal.StructureToPtr(vkCb, colorBlendPtr, false);

            // Dynamic state
            var vkDyn = new VkPipelineDynamicStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                dynamicStateCount = description.DynamicState.DynamicStates != null ? (uint)description.DynamicState.DynamicStates.Length : 0
            };
            Marshal.StructureToPtr(vkDyn, dynamicStatePtr, false);

            // Create graphics pipeline
            var pipelineCI = new VkGraphicsPipelineCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                flags = description.Flags,
                stageCount = (uint)description.ShaderStages.Length,
                pStages = stagesPtr,
                pVertexInputState = vertexInputPtr,
                pInputAssemblyState = inputAssemblyPtr,
                pTessellationState = tessellationPtr,
                pViewportState = viewportPtr,
                pRasterizationState = rasterizerPtr,
                pMultisampleState = multisamplePtr,
                pDepthStencilState = depthStencilPtr,
                pColorBlendState = colorBlendPtr,
                pDynamicState = dynamicStatePtr,
                layout = description.PipelineLayout,
                renderPass = description.RenderPass,
                subpass = description.Subpass,
                basePipelineHandle = description.BasePipelineHandle,
                basePipelineIndex = description.BasePipelineIndex
            };

            IntPtr pipeline = IntPtr.Zero;
            var result = _vkCreateGraphicsPipelines(_logicalDevice, _pipelineCache, 1, ref pipelineCI, IntPtr.Zero, ref pipeline);

            // Free marshaled memory
            Marshal.FreeHGlobal(stagesPtr);
            Marshal.FreeHGlobal(vertexInputPtr);
            Marshal.FreeHGlobal(inputAssemblyPtr);
            Marshal.FreeHGlobal(tessellationPtr);
            Marshal.FreeHGlobal(viewportPtr);
            Marshal.FreeHGlobal(rasterizerPtr);
            Marshal.FreeHGlobal(multisamplePtr);
            Marshal.FreeHGlobal(depthStencilPtr);
            Marshal.FreeHGlobal(colorBlendPtr);
            Marshal.FreeHGlobal(dynamicStatePtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create graphics pipeline: {result}");

            var vp2 = new VulkanPipeline(_device, pipeline, description);
            _pipelines.Add(vp2);
            return vp2;
        }

        /// <summary>Creates a compute pipeline</summary>
        public VulkanComputePipeline CreateComputePipeline(ComputePipelineDescription description)
        {
            var stageInfo = description.Stage;
            int stageSize = Marshal.SizeOf<PipelineShaderStageCreateInfo>();
            IntPtr stagePtr = Marshal.AllocHGlobal(stageSize);
            Marshal.StructureToPtr(stageInfo, stagePtr, false);

            var pipelineCI = new VkComputePipelineCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                flags = description.Flags,
                stageCount = 1,
                pStage = stagePtr,
                layout = description.PipelineLayout,
                basePipelineHandle = description.BasePipelineHandle,
                basePipelineIndex = description.BasePipelineIndex
            };

            IntPtr pipeline = IntPtr.Zero;
            var result = _vkCreateComputePipelines(_logicalDevice, _pipelineCache, 1, ref pipelineCI, IntPtr.Zero, ref pipeline);
            Marshal.FreeHGlobal(stagePtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create compute pipeline: {result}");

            var cp = new VulkanComputePipeline(_device, pipeline, description);
            _computePipelines.Add(cp);
            return cp;
        }

        /// <summary>Creates a descriptor set layout</summary>
        public DescriptorSetLayout CreateDescriptorSetLayout(LayoutDescription description)
        {
            IntPtr bindingsPtr = IntPtr.Zero;
            int bindingCount = description.Bindings?.Length ?? 0;

            if (bindingCount > 0)
            {
                int structSize = Marshal.SizeOf<VkDescriptorSetLayoutBinding>();
                bindingsPtr = Marshal.AllocHGlobal(bindingCount * structSize);
                for (int i = 0; i < bindingCount; i++)
                {
                    var src = description.Bindings[i];
                    var vkBinding = new VkDescriptorSetLayoutBinding
                    {
                        binding = src.Binding,
                        descriptorType = src.DescriptorType,
                        descriptorCount = src.DescriptorCount,
                        stageFlags = src.StageFlags
                    };
                    Marshal.StructureToPtr(vkBinding, bindingsPtr + i * structSize, false);
                }
            }

            var createInfo = new VkDescriptorSetLayoutCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                flags = (uint)description.Flags,
                bindingCount = (uint)bindingCount,
                pBindings = bindingsPtr
            };

            IntPtr layout = IntPtr.Zero;
            var result = _vkCreateDescriptorSetLayout(_logicalDevice, ref createInfo, IntPtr.Zero, ref layout);
            if (bindingsPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(bindingsPtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create descriptor set layout: {result}");

            var dsl = new DescriptorSetLayout(_device, layout, description);
            _descriptorSetLayouts.Add(dsl);
            return dsl;
        }

        /// <summary>Creates a descriptor pool</summary>
        public DescriptorPool CreateDescriptorPool(PoolDescription description)
        {
            IntPtr poolSizesPtr = IntPtr.Zero;
            int poolSizeCount = description.PoolSizes?.Length ?? 0;

            if (poolSizeCount > 0)
            {
                int structSize = Marshal.SizeOf<VkDescriptorPoolSize>();
                poolSizesPtr = Marshal.AllocHGlobal(poolSizeCount * structSize);
                for (int i = 0; i < poolSizeCount; i++)
                {
                    var vkSize = new VkDescriptorPoolSize
                    {
                        type = description.PoolSizes[i].Type,
                        descriptorCount = description.PoolSizes[i].DescriptorCount
                    };
                    Marshal.StructureToPtr(vkSize, poolSizesPtr + i * structSize, false);
                }
            }

            var createInfo = new VkDescriptorPoolCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                flags = (uint)description.Flags,
                maxSets = description.MaxSets,
                poolSizeCount = (uint)poolSizeCount,
                pPoolSizes = poolSizesPtr
            };

            IntPtr pool = IntPtr.Zero;
            var result = _vkCreateDescriptorPool(_logicalDevice, ref createInfo, IntPtr.Zero, ref pool);
            if (poolSizesPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(poolSizesPtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create descriptor pool: {result}");

            var dp = new DescriptorPool(_device, pool, description);
            _descriptorPools.Add(dp);
            return dp;
        }

        /// <summary>Allocates descriptor sets from a pool</summary>
        public DescriptorSet[] AllocateDescriptorSets(DescriptorSetAllocation allocation)
        {
            int layoutCount = allocation.Layouts?.Length ?? 0;
            var layoutsPtr = Marshal.AllocHGlobal(layoutCount * IntPtr.Size);
            for (int i = 0; i < layoutCount; i++)
                Marshal.WriteIntPtr(layoutsPtr + i * IntPtr.Size, allocation.Layouts[i]);

            var allocInfo = new VkDescriptorSetAllocateInfo
            {
                sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool = allocation.Pool,
                descriptorSetCount = (uint)layoutCount,
                pSetLayouts = layoutsPtr
            };

            var descriptorSetsPtr = Marshal.AllocHGlobal(layoutCount * IntPtr.Size);
            var result = _vkAllocateDescriptorSets(_logicalDevice, ref allocInfo, descriptorSetsPtr);
            Marshal.FreeHGlobal(layoutsPtr);

            var descriptorSets = new IntPtr[layoutCount];
            for (int i = 0; i < layoutCount; i++)
                descriptorSets[i] = Marshal.ReadIntPtr(descriptorSetsPtr + i * IntPtr.Size);
            Marshal.FreeHGlobal(descriptorSetsPtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to allocate descriptor sets: {result}");

            return descriptorSets.Select(h => new DescriptorSet(_device, h)).ToArray();
        }

        /// <summary>Updates descriptor sets with new binding data</summary>
        public void UpdateDescriptorSets(DescriptorWrite[] writes)
        {
            if (writes == null || writes.Length == 0)
                return;

            int writeStructSize = Marshal.SizeOf<VkWriteDescriptorSet>();
            var writePtrs = new IntPtr[writes.Length];
            var tempAllocs = new List<IntPtr>();

            for (int w = 0; w < writes.Length; w++)
            {
                var write = writes[w];
                IntPtr imageInfoPtr = IntPtr.Zero;
                IntPtr bufferInfoPtr = IntPtr.Zero;

                if (write.ImageInfos != null && write.ImageInfos.Length > 0)
                {
                    int imgInfoSize = Marshal.SizeOf<VkDescriptorImageInfo>();
                    imageInfoPtr = Marshal.AllocHGlobal(write.ImageInfos.Length * imgInfoSize);
                    tempAllocs.Add(imageInfoPtr);
                    for (int i = 0; i < write.ImageInfos.Length; i++)
                    {
                        var imgInfo = new VkDescriptorImageInfo
                        {
                            sampler = write.ImageInfos[i].Sampler,
                            imageView = write.ImageInfos[i].ImageView,
                            imageLayout = write.ImageInfos[i].ImageLayout
                        };
                        Marshal.StructureToPtr(imgInfo, imageInfoPtr + i * imgInfoSize, false);
                    }
                }

                if (write.BufferInfos != null && write.BufferInfos.Length > 0)
                {
                    int bufInfoSize = Marshal.SizeOf<VkDescriptorBufferInfo>();
                    bufferInfoPtr = Marshal.AllocHGlobal(write.BufferInfos.Length * bufInfoSize);
                    tempAllocs.Add(bufferInfoPtr);
                    for (int i = 0; i < write.BufferInfos.Length; i++)
                    {
                        var bufInfo = new VkDescriptorBufferInfo
                        {
                            buffer = write.BufferInfos[i].Buffer,
                            offset = write.BufferInfos[i].Offset,
                            range = write.BufferInfos[i].Range
                        };
                        Marshal.StructureToPtr(bufInfo, bufferInfoPtr + i * bufInfoSize, false);
                    }
                }

                var vkWrite = new VkWriteDescriptorSet
                {
                    sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet = write.DescriptorSet,
                    dstBinding = write.DstBinding,
                    dstArrayElement = write.DstArrayElement,
                    descriptorCount = (uint)(write.ImageInfos?.Length ?? write.BufferInfos?.Length ?? 0),
                    descriptorType = write.DescriptorType,
                    pImageInfo = imageInfoPtr,
                    pBufferInfo = bufferInfoPtr
                };

                var writeAlloc = Marshal.AllocHGlobal(writeStructSize);
                tempAllocs.Add(writeAlloc);
                Marshal.StructureToPtr(vkWrite, writeAlloc, false);
                writePtrs[w] = writeAlloc;
            }

            var writesArrayPtr = Marshal.AllocHGlobal(writes.Length * writeStructSize);
            tempAllocs.Add(writesArrayPtr);
            for (int i = 0; i < writes.Length; i++)
            {
                var src = Marshal.PtrToStructure<VkWriteDescriptorSet>(writePtrs[i]);
                Marshal.StructureToPtr(src, writesArrayPtr + i * writeStructSize, false);
            }

            _vkUpdateDescriptorSets(_logicalDevice, (uint)writes.Length, writesArrayPtr, 0, IntPtr.Zero);

            foreach (var ptr in tempAllocs)
                Marshal.FreeHGlobal(ptr);
        }

        /// <summary>Creates a shader module from SPIR-V bytecode</summary>
        public VulkanShaderModule CreateShaderModule(byte[] spirvCode)
        {
            if (spirvCode == null || spirvCode.Length == 0)
                throw new ArgumentException("SPIR-V code cannot be null or empty", nameof(spirvCode));

            if (spirvCode.Length % 4 != 0)
                throw new ArgumentException("SPIR-V code size must be a multiple of 4", nameof(spirvCode));

            var codePtr = Marshal.AllocHGlobal(spirvCode.Length);
            Marshal.Copy(spirvCode, 0, codePtr, spirvCode.Length);

            var createInfo = new VkShaderModuleCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize = (IntPtr)spirvCode.Length,
                pCode = codePtr
            };

            IntPtr module = IntPtr.Zero;
            var result = _vkCreateShaderModule(_logicalDevice, ref createInfo, IntPtr.Zero, ref module);
            Marshal.FreeHGlobal(codePtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create shader module: {result}");

            var sm = new VulkanShaderModule(_device, module, spirvCode);
            _shaderModules.Add(sm);
            return sm;
        }

        /// <summary>Creates a sampler</summary>
        public Sampler CreateSampler(SamplerDescription description)
        {
            var createInfo = new VkSamplerCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                magFilter = description.MagFilter,
                minFilter = description.MinFilter,
                mipmapMode = SamplerAddressMode.Repeat,
                addressModeU = description.AddressModeU,
                addressModeV = description.AddressModeV,
                addressModeW = description.AddressModeW,
                mipLodBias = description.MipLodBias,
                anisotropyEnable = description.AnisotropyEnable ? VulkanBool32.True : VulkanBool32.False,
                maxAnisotropy = description.MaxAnisotropy,
                compareEnable = description.CompareEnable ? VulkanBool32.True : VulkanBool32.False,
                compareOp = description.CompareOp,
                minLod = description.MinLod,
                maxLod = description.MaxLod,
                borderColor = 0,
                unnormalizedCoordinates = VulkanBool32.False
            };

            IntPtr sampler = IntPtr.Zero;
            var result = _vkCreateSampler(_logicalDevice, ref createInfo, IntPtr.Zero, ref sampler);
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create sampler: {result}");

            var s = new Sampler(_device, sampler, description);
            _samplers.Add(s);
            return s;
        }

        /// <summary>Creates a command buffer</summary>
        public VulkanCommandBuffer CreateCommandBuffer()
        {
            var allocInfo = new VkCommandBufferAllocateInfo
            {
                sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool = _commandPool,
                level = 0,
                commandBufferCount = 1
            };

            var cmdBufferPtr = Marshal.AllocHGlobal(IntPtr.Size);
            var result = _vkAllocateCommandBuffers(_logicalDevice, ref allocInfo, cmdBufferPtr);
            var cmdBuffer = Marshal.ReadIntPtr(cmdBufferPtr);
            Marshal.FreeHGlobal(cmdBufferPtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to allocate command buffer: {result}");

            var cb = new VulkanCommandBuffer(_device, cmdBuffer, _commandPool);
            _commandBuffers.Add(cb);
            return cb;
        }

        /// <summary>Submits a command buffer to a queue</summary>
        public void SubmitCommandBuffer(VulkanCommandBuffer commandBuffer, IntPtr queue, Fence fence)
        {
            var cmdBufferPtr = commandBuffer.Handle;
            var submitInfo = new VkSubmitInfo
            {
                sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount = 1,
                pCommandBuffers = Marshal.AllocHGlobal(IntPtr.Size),
                waitSemaphoreCount = 0,
                signalSemaphoreCount = 0
            };
            Marshal.WriteIntPtr(submitInfo.pCommandBuffers, cmdBufferPtr);

            var result = _vkQueueSubmit(queue, 1, ref submitInfo, fence?.Handle ?? IntPtr.Zero);
            Marshal.FreeHGlobal(submitInfo.pCommandBuffers);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to submit command buffer: {result}");
        }

        /// <summary>Waits for all device operations to complete</summary>
        public void WaitForIdle()
        {
            _vkDeviceWaitIdle(_logicalDevice);
        }

        /// <summary>Queries hardware capabilities</summary>
        public HardwareCapabilities QueryCapabilities()
        {
            return new HardwareCapabilities
            {
                DeviceName = _physicalDeviceInfo.DeviceName,
                VendorId = _physicalDeviceInfo.VendorId.ToString("X"),
                DeviceId = _physicalDeviceInfo.DeviceId,
                ApiVersion = _physicalDeviceInfo.ApiVersion,
                MaxTextureDimension1D = _physicalDeviceInfo.MaxTextureDimension1D,
                MaxTextureDimension2D = _physicalDeviceInfo.MaxTextureDimension2D,
                MaxTextureDimension3D = _physicalDeviceInfo.MaxTextureDimension3D,
                MaxTextureArrayLayers = _physicalDeviceInfo.MaxTextureArrayLayers,
                MaxPushConstantsSize = _physicalDeviceInfo.MaxPushConstantsSize,
                MaxBoundDescriptorSets = _physicalDeviceInfo.MaxBoundDescriptorSets,
                MaxColorAttachments = _physicalDeviceInfo.MaxColorAttachments,
                MaxVertexInputAttributes = _physicalDeviceInfo.MaxVertexInputAttributes,
                MaxVertexInputBindings = _physicalDeviceInfo.MaxVertexInputBindings,
                MaxVertexInputAttributeOffset = _physicalDeviceInfo.MaxVertexInputAttributeOffset,
                MaxVertexInputBindingStride = _physicalDeviceInfo.MaxVertexInputBindingStride,
                MaxVertexOutputComponents = _physicalDeviceInfo.MaxVertexOutputComponents,
                MaxSamplerAnisotropy = _physicalDeviceInfo.MaxSamplerAnisotropy,
                SupportsTimelineSemaphores = _physicalDeviceInfo.SupportsTimelineSemaphores,
                SupportsBufferDeviceAddress = _physicalDeviceInfo.SupportsBufferDeviceAddress,
                SupportsDynamicRendering = _physicalDeviceInfo.SupportsDynamicRendering,
                SupportsSynchronization2 = _physicalDeviceInfo.SupportsSynchronization2,
                MaxComputeWorkGroupCount0 = _physicalDeviceInfo.MaxComputeWorkGroupCount0,
                MaxComputeWorkGroupCount1 = _physicalDeviceInfo.MaxComputeWorkGroupCount1,
                MaxComputeWorkGroupCount2 = _physicalDeviceInfo.MaxComputeWorkGroupCount2,
                MaxComputeWorkGroupSize0 = _physicalDeviceInfo.MaxComputeWorkGroupSize0,
                MaxComputeWorkGroupSize1 = _physicalDeviceInfo.MaxComputeWorkGroupSize1,
                MaxComputeWorkGroupSize2 = _physicalDeviceInfo.MaxComputeWorkGroupSize2,
                MaxComputeWorkGroupInvocations = _physicalDeviceInfo.MaxComputeWorkGroupInvocations,
                MemoryProperties = _physicalDeviceInfo.MemoryProperties
            };
        }

        /// <summary>Creates the Vulkan RHI device with the given configuration</summary>
        public VulkanRhiDevice(RhiDeviceCreationInfo creationInfo)
        {
            _creationInfo = creationInfo;
            _device = new VulkanDevice();
            _physicalDeviceInfo = new VulkanPhysicalDeviceInfo();
            _queueFamilyIndices = new QueueFamilyIndices();

            CreateInstance();
            PickPhysicalDevice();
            _device.Instance = _instance;
            _device.PhysicalDevice = _physicalDevice;
            _device.PhysicalDeviceInfo = _physicalDeviceInfo;
            CreateLogicalDevice();
            InitializeQueues();
            CreatePipelineCacheInternal();
            InitializeSubsystems();
            CreateDebugMessenger();
        }

        private void CreateInstance()
        {
            var appInfo = new VkApplicationInfo
            {
                sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pApplicationName = Marshal.StringToHGlobalAnsi(_creationInfo.ApplicationName),
                applicationVersion = _creationInfo.ApplicationVersion,
                pEngineName = Marshal.StringToHGlobalAnsi(_creationInfo.EngineName),
                engineVersion = _creationInfo.EngineVersion,
                apiVersion = _creationInfo.ApiVersion
            };

            var extensions = new List<string>();
            if (_creationInfo.RequiredExtensions is { Length: > 0 })
            {
                foreach (var ext in _creationInfo.RequiredExtensions)
                {
                    if (!string.IsNullOrWhiteSpace(ext) && !extensions.Contains(ext))
                        extensions.Add(ext);
                }
            }
            else
            {
                // Legacy default: Windows Win32 surface when caller does not specify.
                extensions.Add("VK_KHR_surface");
                if (OperatingSystem.IsWindows())
                    extensions.Add("VK_KHR_win32_surface");
            }

            if (!extensions.Contains("VK_KHR_surface"))
                extensions.Insert(0, "VK_KHR_surface");

            if (_creationInfo.EnableValidation && !extensions.Contains("VK_EXT_debug_utils"))
                extensions.Add("VK_EXT_debug_utils");

            // MoltenVK / portability on macOS
            if (OperatingSystem.IsMacOS() && !extensions.Contains("VK_KHR_portability_enumeration"))
                extensions.Add("VK_KHR_portability_enumeration");

            var extensionPtrs = extensions.Select(e => Marshal.StringToHGlobalAnsi(e)).ToArray();
            var layerNames = new List<string>();
            if (_creationInfo.EnableValidation)
                layerNames.Add("VK_LAYER_KHRONOS_validation");
            var layerPtrs = layerNames.Select(l => Marshal.StringToHGlobalAnsi(l)).ToArray();

            var createInfo = new VkInstanceCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR — required for MoltenVK on macOS
                flags = OperatingSystem.IsMacOS() ? 0x00000001u : 0u,
                pApplicationInfo = Marshal.AllocHGlobal(Marshal.SizeOf<VkApplicationInfo>())
            };
            Marshal.StructureToPtr(appInfo, createInfo.pApplicationInfo, false);

            unsafe
            {
                createInfo.enabledExtensionCount = (uint)extensionPtrs.Length;
                createInfo.ppEnabledExtensionNames = Marshal.AllocHGlobal(extensionPtrs.Length * IntPtr.Size);
                for (int i = 0; i < extensionPtrs.Length; i++)
                    Marshal.WriteIntPtr(createInfo.ppEnabledExtensionNames + i * IntPtr.Size, extensionPtrs[i]);

                createInfo.enabledLayerCount = (uint)layerPtrs.Length;
                createInfo.ppEnabledLayerNames = Marshal.AllocHGlobal(layerPtrs.Length * IntPtr.Size);
                for (int i = 0; i < layerPtrs.Length; i++)
                    Marshal.WriteIntPtr(createInfo.ppEnabledLayerNames + i * IntPtr.Size, layerPtrs[i]);
            }

            var result = vkCreateInstance(ref createInfo, IntPtr.Zero, ref _instance);
            Marshal.FreeHGlobal(createInfo.pApplicationInfo);
            foreach (var ptr in extensionPtrs)
                Marshal.FreeHGlobal(ptr);
            foreach (var ptr in layerPtrs)
                Marshal.FreeHGlobal(ptr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create Vulkan instance: {result}");
        }

        /// <summary>Creates a pipeline layout</summary>
        public VulkanPipelineLayout CreatePipelineLayout(IntPtr[] setLayouts, (ShaderStageFlag stage, uint offset, uint size)[] pushConstants = null)
        {
            IntPtr setLayoutsPtr = IntPtr.Zero;
            if (setLayouts != null && setLayouts.Length > 0)
            {
                setLayoutsPtr = Marshal.AllocHGlobal(setLayouts.Length * IntPtr.Size);
                for (int i = 0; i < setLayouts.Length; i++)
                    Marshal.WriteIntPtr(setLayoutsPtr + i * IntPtr.Size, setLayouts[i]);
            }

            VkPushConstantRange[]? pushRanges = null;
            IntPtr pushRangesPtr = IntPtr.Zero;
            if (pushConstants != null && pushConstants.Length > 0)
            {
                pushRanges = new VkPushConstantRange[pushConstants.Length];
                for (int i = 0; i < pushConstants.Length; i++)
                {
                    pushRanges[i] = new VkPushConstantRange
                    {
                        stageFlags = pushConstants[i].stage,
                        offset = pushConstants[i].offset,
                        size = pushConstants[i].size
                    };
                }
                int structSize = Marshal.SizeOf<VkPushConstantRange>();
                pushRangesPtr = Marshal.AllocHGlobal(pushRanges.Length * structSize);
                for (int i = 0; i < pushRanges.Length; i++)
                    Marshal.StructureToPtr(pushRanges[i], pushRangesPtr + i * structSize, false);
            }

            var createInfo = new VkPipelineLayoutCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount = (uint)(setLayouts?.Length ?? 0),
                pSetLayouts = setLayoutsPtr,
                pushConstantRangeCount = (uint)(pushConstants?.Length ?? 0),
                pPushConstantRanges = pushRangesPtr
            };

            IntPtr layout = IntPtr.Zero;
            var result = _vkCreatePipelineLayout(_logicalDevice, ref createInfo, IntPtr.Zero, ref layout);

            if (setLayoutsPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(setLayoutsPtr);
            if (pushRangesPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(pushRangesPtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create pipeline layout: {result}");

            return new VulkanPipelineLayout(_device, layout);
        }

        /// <summary>Submits a command buffer with semaphore synchronization</summary>
        public void SubmitCommandBufferWithSemaphores(
            VulkanCommandBuffer commandBuffer,
            IntPtr waitSemaphore,
            PipelineStageFlag waitStage,
            IntPtr signalSemaphore,
            IntPtr fence)
        {
            var cmdBufferPtr = commandBuffer.Handle;

            var waitSemaphoreCount = waitSemaphore != IntPtr.Zero ? 1u : 0u;
            var signalSemaphoreCount = signalSemaphore != IntPtr.Zero ? 1u : 0u;

            var waitStagePtr = Marshal.AllocHGlobal(sizeof(uint));
            Marshal.WriteInt32(waitStagePtr, (int)waitStage);

            var waitSemaphorePtr = IntPtr.Zero;
            if (waitSemaphore != IntPtr.Zero)
            {
                waitSemaphorePtr = Marshal.AllocHGlobal(IntPtr.Size);
                Marshal.WriteIntPtr(waitSemaphorePtr, waitSemaphore);
            }

            var signalSemaphorePtr = IntPtr.Zero;
            if (signalSemaphore != IntPtr.Zero)
            {
                signalSemaphorePtr = Marshal.AllocHGlobal(IntPtr.Size);
                Marshal.WriteIntPtr(signalSemaphorePtr, signalSemaphore);
            }

            var cmdBufferArrayPtr = Marshal.AllocHGlobal(IntPtr.Size);
            Marshal.WriteIntPtr(cmdBufferArrayPtr, cmdBufferPtr);

            var submitInfo = new VkSubmitInfo
            {
                sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount = 1,
                pCommandBuffers = cmdBufferArrayPtr,
                waitSemaphoreCount = waitSemaphoreCount,
                pWaitSemaphores = waitSemaphorePtr,
                pWaitDstStageMask = waitStagePtr,
                signalSemaphoreCount = signalSemaphoreCount,
                pSignalSemaphores = signalSemaphorePtr
            };

            var result = _vkQueueSubmit(_graphicsQueue, 1, ref submitInfo, fence);
            Marshal.FreeHGlobal(cmdBufferArrayPtr);
            Marshal.FreeHGlobal(waitStagePtr);
            if (waitSemaphorePtr != IntPtr.Zero)
                Marshal.FreeHGlobal(waitSemaphorePtr);
            if (signalSemaphorePtr != IntPtr.Zero)
                Marshal.FreeHGlobal(signalSemaphorePtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to submit command buffer: {result}");
        }

        /// <summary>Gets the graphics queue handle</summary>
        public IntPtr GraphicsQueue => _graphicsQueue;
        public IntPtr ComputeQueue => _computeQueue != IntPtr.Zero ? _computeQueue : _graphicsQueue;
        /// <summary>Gets the present queue handle</summary>
        public IntPtr PresentQueue => _presentQueue;

        /// <summary>Disposes the device and all resources</summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            WaitForIdle();
            Cleanup();
            GC.SuppressFinalize(this);
        }

        private void Cleanup()
        {
            foreach (var s in _samplers)
                s?.Dispose();
            foreach (var sm in _shaderModules)
                sm?.Dispose();
            foreach (var p in _pipelines)
                p?.Dispose();
            foreach (var cp in _computePipelines)
                cp?.Dispose();
            foreach (var fb in _framebuffers)
                fb?.Dispose();
            foreach (var rp in _renderPasses)
                rp?.Dispose();
            foreach (var t in _textures)
                t?.Dispose();
            foreach (var b in _buffers)
                b?.Dispose();
            foreach (var dpl in _descriptorPools)
                dpl?.Dispose();
            foreach (var dsl in _descriptorSetLayouts)
                dsl?.Dispose();

            _memoryAllocator?.Dispose();
            _syncManager?.Dispose();
            _resourceTracker?.Dispose();

            if (_pipelineCache != IntPtr.Zero)
                _vkDestroyPipelineCache?.Invoke(_logicalDevice, _pipelineCache, IntPtr.Zero);

            if (_commandPool != IntPtr.Zero && _vkDestroyCommandPool != null)
                _vkDestroyCommandPool(_logicalDevice, _commandPool, IntPtr.Zero);

            if (_logicalDevice != IntPtr.Zero)
                vkDestroyDevice(_logicalDevice, IntPtr.Zero);

            if (_debugUtilsMessenger != IntPtr.Zero && _vkDestroyDebugUtilsMessenger != null)
                _vkDestroyDebugUtilsMessenger(_instance, _debugUtilsMessenger, IntPtr.Zero);

            if (_instance != IntPtr.Zero)
                vkDestroyInstance(_instance, IntPtr.Zero);
        }
    }
    // =========================================================================
    // VulkanSwapchain
    // =========================================================================

    /// <summary>
    /// Manages a Vulkan swapchain and its associated images.
    /// Handles image acquisition, presentation, and resize operations.
    /// </summary>
}
