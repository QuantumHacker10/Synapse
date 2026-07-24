// =============================================================================
// GDNN Engine - Vulkan 1.4 Render Hardware Interface Backend
// VulkanRhiDevice.Resources.cs — Vulkan RHI partial module
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
    public class VulkanSwapchain : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _swapchain;
        private IntPtr _surface;
        private IntPtr[] _images;
        private VulkanTexture[] _imageWrappers;
        private IntPtr[] _imageViews;
        private VulkanFormat _surfaceFormat;
        private PresentMode _presentMode;
        private Extent2D _extent;
        private uint _currentImageIndex;
        private bool _disposed;

        // Function pointers
        private GetSwapchainImagesKHRDel _vkGetSwapchainImagesKHR;
        private AcquireNextImageKHRDel _vkAcquireNextImageKHR;
        private QueuePresentKHRDel _vkQueuePresentKHR;
        private CreateImageViewDel _vkCreateImageView;
        private DestroyImageViewDel _vkDestroyImageView;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate VulkanResult GetSwapchainImagesKHRDel(IntPtr device, IntPtr swapchain, ref uint pSwapchainImageCount, IntPtr pSwapchainImages);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate VulkanResult AcquireNextImageKHRDel(IntPtr device, IntPtr swapchain, ulong timeout, IntPtr semaphore, IntPtr fence, ref uint pImageIndex);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate VulkanResult QueuePresentKHRDel(IntPtr queue, ref VkPresentInfoKHR pPresentInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate VulkanResult CreateImageViewDel(IntPtr device, ref VkImageViewCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pView);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate void DestroyImageViewDel(IntPtr device, IntPtr imageView, IntPtr pAllocator);

        /// <summary>Swapchain handle</summary>
        public IntPtr Handle => _swapchain;

        /// <summary>Current extent of swapchain images</summary>
        public Extent2D Extent => _extent;

        /// <summary>Surface format of the swapchain</summary>
        public VulkanFormat SurfaceFormat => _surfaceFormat;

        /// <summary>Present mode of the swapchain</summary>
        public PresentMode CurrentPresentMode => _presentMode;

        public VulkanSwapchain(VulkanDevice device, IntPtr swapchain, VulkanFormat surfaceFormat, PresentMode presentMode, Extent2D extent)
        {
            _device = device;
            _swapchain = swapchain;
            _surfaceFormat = surfaceFormat;
            _presentMode = presentMode;
            _extent = extent;

            LoadFunctions();
            RetrieveImages();
            CreateImageViews();
        }

        private void LoadFunctions()
        {
            _vkGetSwapchainImagesKHR = LoadDeviceFunction<GetSwapchainImagesKHRDel>("vkGetSwapchainImagesKHR");
            _vkAcquireNextImageKHR = LoadDeviceFunction<AcquireNextImageKHRDel>("vkAcquireNextImageKHR");
            _vkQueuePresentKHR = LoadDeviceFunction<QueuePresentKHRDel>("vkQueuePresentKHR");
            _vkCreateImageView = LoadDeviceFunction<CreateImageViewDel>("vkCreateImageView");
            _vkDestroyImageView = LoadDeviceFunction<DestroyImageViewDel>("vkDestroyImageView");
        }

        private T? LoadDeviceFunction<T>(string name) where T : Delegate
        {
            var namePtr = Marshal.StringToHGlobalAnsi(name);
            try
            {
                var ptr = _device.GetDeviceProcAddr(name);
                if (ptr == IntPtr.Zero)
                    ptr = GetProcAddress("vulkan-1", namePtr);
                if (ptr != IntPtr.Zero)
                    return Marshal.GetDelegateForFunctionPointer<T>(ptr);
            }
            finally { Marshal.FreeHGlobal(namePtr); }
            return null;
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr GetProcAddress(string hModule, IntPtr procName);

        private void RetrieveImages()
        {
            uint imageCount = 0;
            _vkGetSwapchainImagesKHR(_device.LogicalDevice, _swapchain, ref imageCount, IntPtr.Zero);
            _images = new IntPtr[imageCount];
            var imagesHandle = GCHandle.Alloc(_images, GCHandleType.Pinned);
            try
            {
                _vkGetSwapchainImagesKHR(_device.LogicalDevice, _swapchain, ref imageCount, imagesHandle.AddrOfPinnedObject());
            }
            finally
            {
                imagesHandle.Free();
            }
            _imageWrappers = new VulkanTexture[imageCount];

            for (int i = 0; i < imageCount; i++)
            {
                _imageWrappers[i] = new VulkanTexture(_device, _images[i], new TextureDescription
                {
                    Width = _extent.Width,
                    Height = _extent.Height,
                    Format = _surfaceFormat
                });
            }
        }

        private void CreateImageViews()
        {
            _imageViews = new IntPtr[_images.Length];
            for (int i = 0; i < _images.Length; i++)
            {
                var createInfo = new VkImageViewCreateInfo
                {
                    sType = 15,
                    image = _images[i],
                    viewType = ImageViewType.Type2D,
                    format = _surfaceFormat,
                    subresourceRange = new VkImageSubresourceRange
                    {
                        aspectMask = ImageAspectFlag.Color,
                        baseMipLevel = 0,
                        levelCount = 1,
                        baseArrayLayer = 0,
                        layerCount = 1
                    }
                };
                createInfo.components = new VkComponentMapping { r = 0, g = 1, b = 2, a = 3 };

                _vkCreateImageView(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref _imageViews[i]);
            }
        }

        /// <summary>Acquires the next presentable image</summary>
        public uint AcquireNextImage(ulong timeout = 0xFFFFFFFFFFFFFFFF)
        {
            uint imageIndex = 0;
            var semaphore = _device.SyncManager?.CreateSemaphore() ?? IntPtr.Zero;
            var fence = _device.SyncManager?.CreateFence(false) ?? IntPtr.Zero;

            var result = _vkAcquireNextImageKHR(_device.LogicalDevice, _swapchain, timeout, semaphore, fence, ref imageIndex);

            if (result == VulkanResult.Success || result == VulkanResult.SuboptimalKHR || result == VulkanResult.ErrorOutOfDateKHR)
            {
                _currentImageIndex = imageIndex;
                return imageIndex;
            }

            return 0;
        }

        /// <summary>Presents the current image to the display</summary>
        public VulkanResult Present(IntPtr queue)
        {
            var imageIndex = _currentImageIndex;
            var presentInfo = new VkPresentInfoKHR
            {
                sType = 1000001001,
                swapchainCount = 1,
                pSwapchains = Marshal.AllocHGlobal(IntPtr.Size),
                pImageIndices = Marshal.AllocHGlobal(sizeof(uint))
            };
            Marshal.WriteIntPtr(presentInfo.pSwapchains, _swapchain);
            Marshal.WriteInt32(presentInfo.pImageIndices, (int)imageIndex);

            var result = _vkQueuePresentKHR(queue, ref presentInfo);

            Marshal.FreeHGlobal(presentInfo.pSwapchains);
            Marshal.FreeHGlobal(presentInfo.pImageIndices);

            return result;
        }

        /// <summary>Resizes the swapchain</summary>
        public void Resize(uint width, uint height)
        {
            if (width == 0 || height == 0)
                return;
            _extent = new Extent2D(width, height);
            CleanupImageViews();
            CleanupImages();

            // Old swapchain is passed for recreation
            _swapchain = IntPtr.Zero;
            RetrieveImages();
            CreateImageViews();
        }

        /// <summary>Gets all swapchain images</summary>
        public VulkanTexture[] GetImages() => _imageWrappers;

        /// <summary>Gets the current swapchain image</summary>
        public VulkanTexture GetCurrentImage() => _imageWrappers[_currentImageIndex];

        /// <summary>Gets the surface format</summary>
        public VulkanFormat GetSurfaceFormat() => _surfaceFormat;

        /// <summary>Gets the present mode</summary>
        public PresentMode GetPresentMode() => _presentMode;

        /// <summary>Gets an image view handle by index</summary>
        public IntPtr GetImageView(uint index) => index < _imageViews.Length ? _imageViews[index] : IntPtr.Zero;

        /// <summary>Gets the image handle by index</summary>
        public IntPtr GetImageHandle(uint index) => index < _images.Length ? _images[index] : IntPtr.Zero;

        /// <summary>Gets the number of images in the swapchain</summary>
        public uint ImageCount => (uint)_images.Length;

        /// <summary>Gets the current image index</summary>
        public uint CurrentImageIndex => _currentImageIndex;

        private void CleanupImageViews()
        {
            if (_imageViews == null)
                return;
            foreach (var iv in _imageViews)
                if (iv != IntPtr.Zero)
                    _vkDestroyImageView?.Invoke(_device.LogicalDevice, iv, IntPtr.Zero);
        }

        private void CleanupImages()
        {
            if (_imageWrappers == null)
                return;
            foreach (var img in _imageWrappers)
                img?.Dispose();
            _imageWrappers = null;
            _images = null;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            CleanupImageViews();
            CleanupImages();
            GC.SuppressFinalize(this);
        }
    }
    // =========================================================================
    // VulkanCommandBuffer
    // =========================================================================

    /// <summary>
    /// Wraps a Vulkan command buffer for recording and submitting GPU commands.
    /// Provides methods for render pass management, draw calls, compute dispatch,
    /// and resource transitions.
    /// </summary>
    public class VulkanCommandBuffer : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _commandBuffer;
        private IntPtr _commandPool;
        private bool _isRecording;
        private bool _disposed;

        // Cached function pointers
        private CmdBeginRenderPassDel _vkCmdBeginRenderPass;
        private CmdEndRenderPassDel _vkCmdEndRenderPass;
        private CmdBindPipelineDel _vkCmdBindPipeline;
        private CmdDrawDel _vkCmdDraw;
        private CmdDrawIndexedDel _vkCmdDrawIndexed;
        private CmdDispatchDel _vkCmdDispatch;
        private CmdPipelineBarrierDel _vkCmdPipelineBarrier;
        private CmdCopyBufferDel _vkCmdCopyBuffer;
        private CmdCopyBufferToImageDel _vkCmdCopyBufferToImage;
        private CmdCopyImageToBufferDel _vkCmdCopyImageToBuffer;
        private CmdPushConstantsDel _vkCmdPushConstants;
        private CmdSetViewportDel _vkCmdSetViewport;
        private CmdSetScissorDel _vkCmdSetScissor;
        private CmdBindVertexBuffersDel _vkCmdBindVertexBuffers;
        private CmdBindIndexBufferDel _vkCmdBindIndexBuffer;
        private CmdCopyImageDel _vkCmdCopyImage;
        private CmdBlitImageDel _vkCmdBlitImage;
        private CmdResolveImageDel _vkCmdResolveImage;
        private CmdWriteTimestampDel _vkCmdWriteTimestamp;
        private CmdBeginQueryDel _vkCmdBeginQuery;
        private CmdEndQueryDel _vkCmdEndQuery;
        private CmdCopyQueryPoolResultsDel _vkCmdCopyQueryPoolResults;
        private CmdBindDescriptorSetsDel _vkCmdBindDescriptorSets;

        // Delegates
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginRenderPassDel(IntPtr cmdBuffer, ref VkRenderPassBeginInfo pRenderPassBegin, uint contents);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndRenderPassDel(IntPtr cmdBuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindPipelineDel(IntPtr cmdBuffer, PipelineBindPoint pipelineBindPoint, IntPtr pipeline);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDrawDel(IntPtr cmdBuffer, uint vertexCount, uint instanceCount, uint firstVertex, uint firstInstance);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDrawIndexedDel(IntPtr cmdBuffer, uint indexCount, uint instanceCount, uint firstIndex, int vertexOffset, uint firstInstance);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDispatchDel(IntPtr cmdBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdPipelineBarrierDel(IntPtr cmdBuffer, PipelineStageFlag srcStageMask, PipelineStageFlag dstStageMask, uint dependencyFlags, uint memoryBarrierCount, IntPtr pMemoryBarriers, uint bufferMemoryBarrierCount, IntPtr pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferDel(IntPtr cmdBuffer, IntPtr srcBuffer, IntPtr dstBuffer, uint regionCount, ref BufferCopy pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferToImageDel(IntPtr cmdBuffer, IntPtr buffer, IntPtr image, uint imageLayout, uint regionCount, ref BufferImageCopy pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyImageToBufferDel(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstBuffer, uint regionCount, ref BufferImageCopy pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdPushConstantsDel(IntPtr cmdBuffer, IntPtr layout, ShaderStageFlag stageFlags, uint offset, uint size, IntPtr pValues);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetViewportDel(IntPtr cmdBuffer, uint firstViewport, uint viewportCount, ref Viewport pViewports);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetScissorDel(IntPtr cmdBuffer, uint firstScissor, uint scissorCount, ref Rect2D pScissors);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindVertexBuffersDel(IntPtr cmdBuffer, uint firstBinding, uint bindingCount, ref IntPtr pBuffers, ref ulong pOffsets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindIndexBufferDel(IntPtr cmdBuffer, IntPtr buffer, ulong offset, IndexType indexType);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyImageDel(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBlitImageDel(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions, Filter filter);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdResolveImageDel(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdWriteTimestampDel(IntPtr cmdBuffer, PipelineStageFlag pipelineStage, IntPtr queryPool, uint query);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginQueryDel(IntPtr cmdBuffer, IntPtr queryPool, uint query, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndQueryDel(IntPtr cmdBuffer, IntPtr queryPool, uint query);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyQueryPoolResultsDel(IntPtr cmdBuffer, IntPtr queryPool, uint firstQuery, uint queryCount, IntPtr dstBuffer, ulong dstOffset, ulong stride, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindDescriptorSetsDel(IntPtr cmdBuffer, uint pipelineBindPoint, IntPtr layout, uint firstSet, uint descriptorSetCount, IntPtr pDescriptorSets, uint dynamicOffsetCount, IntPtr pDynamicOffsets);

        public IntPtr Handle => _commandBuffer;
        public bool IsRecording => _isRecording;

        public VulkanCommandBuffer(VulkanDevice device, IntPtr commandBuffer, IntPtr commandPool)
        {
            _device = device;
            _commandBuffer = commandBuffer;
            _commandPool = commandPool;
            LoadFunctions();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkCmdBeginRenderPass = Marshal.GetDelegateForFunctionPointer<CmdBeginRenderPassDel>(load("vkCmdBeginRenderPass"));
            _vkCmdEndRenderPass = Marshal.GetDelegateForFunctionPointer<CmdEndRenderPassDel>(load("vkCmdEndRenderPass"));
            _vkCmdBindPipeline = Marshal.GetDelegateForFunctionPointer<CmdBindPipelineDel>(load("vkCmdBindPipeline"));
            _vkCmdDraw = Marshal.GetDelegateForFunctionPointer<CmdDrawDel>(load("vkCmdDraw"));
            _vkCmdDrawIndexed = Marshal.GetDelegateForFunctionPointer<CmdDrawIndexedDel>(load("vkCmdDrawIndexed"));
            _vkCmdDispatch = Marshal.GetDelegateForFunctionPointer<CmdDispatchDel>(load("vkCmdDispatch"));
            _vkCmdPipelineBarrier = Marshal.GetDelegateForFunctionPointer<CmdPipelineBarrierDel>(load("vkCmdPipelineBarrier"));
            _vkCmdCopyBuffer = Marshal.GetDelegateForFunctionPointer<CmdCopyBufferDel>(load("vkCmdCopyBuffer"));
            _vkCmdCopyBufferToImage = Marshal.GetDelegateForFunctionPointer<CmdCopyBufferToImageDel>(load("vkCmdCopyBufferToImage"));
            _vkCmdCopyImageToBuffer = Marshal.GetDelegateForFunctionPointer<CmdCopyImageToBufferDel>(load("vkCmdCopyImageToBuffer"));
            _vkCmdPushConstants = Marshal.GetDelegateForFunctionPointer<CmdPushConstantsDel>(load("vkCmdPushConstants"));
            _vkCmdSetViewport = Marshal.GetDelegateForFunctionPointer<CmdSetViewportDel>(load("vkCmdSetViewport"));
            _vkCmdSetScissor = Marshal.GetDelegateForFunctionPointer<CmdSetScissorDel>(load("vkCmdSetScissor"));
            _vkCmdBindVertexBuffers = Marshal.GetDelegateForFunctionPointer<CmdBindVertexBuffersDel>(load("vkCmdBindVertexBuffers"));
            _vkCmdBindIndexBuffer = Marshal.GetDelegateForFunctionPointer<CmdBindIndexBufferDel>(load("vkCmdBindIndexBuffer"));
            _vkCmdCopyImage = Marshal.GetDelegateForFunctionPointer<CmdCopyImageDel>(load("vkCmdCopyImage"));
            _vkCmdBlitImage = Marshal.GetDelegateForFunctionPointer<CmdBlitImageDel>(load("vkCmdBlitImage"));
            _vkCmdResolveImage = Marshal.GetDelegateForFunctionPointer<CmdResolveImageDel>(load("vkCmdResolveImage"));
            _vkCmdWriteTimestamp = Marshal.GetDelegateForFunctionPointer<CmdWriteTimestampDel>(load("vkCmdWriteTimestamp"));
            _vkCmdBeginQuery = Marshal.GetDelegateForFunctionPointer<CmdBeginQueryDel>(load("vkCmdBeginQuery"));
            _vkCmdEndQuery = Marshal.GetDelegateForFunctionPointer<CmdEndQueryDel>(load("vkCmdEndQuery"));
            _vkCmdCopyQueryPoolResults = Marshal.GetDelegateForFunctionPointer<CmdCopyQueryPoolResultsDel>(load("vkCmdCopyQueryPoolResults"));
            _vkCmdBindDescriptorSets = Marshal.GetDelegateForFunctionPointer<CmdBindDescriptorSetsDel>(load("vkCmdBindDescriptorSets"));
        }

        [DllImport("vulkan-1")]
        private static extern VulkanResult vkBeginCommandBuffer(IntPtr commandBuffer, ref VkCommandBufferBeginInfo pBeginInfo);
        [DllImport("vulkan-1")]
        private static extern VulkanResult vkEndCommandBuffer(IntPtr commandBuffer);
        [DllImport("vulkan-1")]
        private static extern void vkCmdPipelineBarrier(IntPtr cmdBuffer, PipelineStageFlag srcStageMask, PipelineStageFlag dstStageMask, uint dependencyFlags, uint memoryBarrierCount, IntPtr pMemoryBarriers, uint bufferMemoryBarrierCount, IntPtr pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers);

        [StructLayout(LayoutKind.Sequential)]
        internal struct VkImageCopy
        {
            public VkImageCopy_SrcSubresource srcSubresource;
            public VkImageCopy_Offset3D srcOffset;
            public VkImageCopy_SrcSubresource dstSubresource;
            public VkImageCopy_Offset3D dstOffset;
            public VkImageCopy_Extent3D extent;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct VkImageCopy_SrcSubresource
        {
            public uint aspectMask;
            public uint mipLevel;
            public uint baseArrayLayer;
            public uint layerCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct VkImageCopy_Offset3D
        {
            public int x, y, z;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct VkImageCopy_Extent3D
        {
            public uint width, height, depth;
        }

        /// <summary>Begins recording commands</summary>
        public VulkanResult Begin(CommandBufferUsageFlag flags = 0)
        {
            var beginInfo = new VkCommandBufferBeginInfo
            {
                sType = 41,
                flags = (uint)flags
            };
            var result = vkBeginCommandBuffer(_commandBuffer, ref beginInfo);
            if (result == VulkanResult.Success)
                _isRecording = true;
            return result;
        }

        /// <summary>Ends recording commands</summary>
        public VulkanResult End()
        {
            var result = vkEndCommandBuffer(_commandBuffer);
            if (result == VulkanResult.Success)
                _isRecording = false;
            return result;
        }

        /// <summary>Begins a render pass</summary>
        public void BeginRenderPass(VulkanRenderPass renderPass, VulkanFramebuffer framebuffer, ClearValue[] clearValues, Rect2D renderArea = default)
        {
            var clearValuesPtr = IntPtr.Zero;
            int clearValueSize = 16;
            if (clearValues != null && clearValues.Length > 0)
            {
                clearValuesPtr = Marshal.AllocHGlobal(clearValues.Length * clearValueSize);
                for (int i = 0; i < clearValues.Length; i++)
                    Marshal.StructureToPtr(clearValues[i], clearValuesPtr + i * clearValueSize, false);
            }

            var beginInfo = new VkRenderPassBeginInfo
            {
                sType = 42,
                renderPass = renderPass.Handle,
                framebuffer = framebuffer.Handle,
                clearValueCount = (uint)(clearValues?.Length ?? 0),
                pClearValues = clearValuesPtr
            };

            if (renderArea.Extent.Width == 0 && renderArea.Extent.Height == 0)
            {
                beginInfo.renderArea = new VkRenderPassBeginInfo_Rect2D
                {
                    x = 0,
                    y = 0,
                    width = framebuffer.Width,
                    height = framebuffer.Height
                };
            }
            else
            {
                beginInfo.renderArea = new VkRenderPassBeginInfo_Rect2D
                {
                    x = renderArea.Offset.X,
                    y = renderArea.Offset.Y,
                    width = renderArea.Extent.Width,
                    height = renderArea.Extent.Height
                };
            }

            _vkCmdBeginRenderPass(_commandBuffer, ref beginInfo, 0);
            if (clearValuesPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(clearValuesPtr);
        }

        /// <summary>Ends the current render pass</summary>
        public void EndRenderPass() => _vkCmdEndRenderPass(_commandBuffer);

        /// <summary>Binds a graphics or compute pipeline</summary>
        public void BindPipeline(VulkanPipeline pipeline) => _vkCmdBindPipeline(_commandBuffer, PipelineBindPoint.Graphics, pipeline.Handle);

        /// <summary>Binds a compute pipeline</summary>
        public void BindComputePipeline(VulkanComputePipeline pipeline) => _vkCmdBindPipeline(_commandBuffer, PipelineBindPoint.Compute, pipeline.Handle);

        /// <summary>Binds vertex buffers</summary>
        public void BindVertexBuffer(VulkanBuffer buffer, ulong offset = 0)
        {
            var bufferHandle = buffer.Handle;
            _vkCmdBindVertexBuffers(_commandBuffer, 0, 1, ref bufferHandle, ref offset);
        }

        /// <summary>Binds multiple vertex buffers</summary>
        public void BindVertexBuffers(VulkanBuffer[] buffers, ulong[] offsets)
        {
            var bufferHandles = new IntPtr[buffers.Length];
            for (int i = 0; i < buffers.Length; i++)
                bufferHandles[i] = buffers[i].Handle;
            var bufferPtr = Marshal.AllocHGlobal(buffers.Length * IntPtr.Size);
            for (int i = 0; i < buffers.Length; i++)
                Marshal.WriteIntPtr(bufferPtr + i * IntPtr.Size, bufferHandles[i]);
            var offsetPtr = Marshal.AllocHGlobal(buffers.Length * sizeof(ulong));
            for (int i = 0; i < buffers.Length; i++)
                Marshal.WriteInt64(offsetPtr + i * sizeof(ulong), (long)offsets[i]);
            _vkCmdBindVertexBuffers(_commandBuffer, 0, (uint)buffers.Length, ref bufferPtr, ref offsets[0]);
            Marshal.FreeHGlobal(bufferPtr);
            Marshal.FreeHGlobal(offsetPtr);
        }

        /// <summary>Binds an index buffer</summary>
        public void BindIndexBuffer(VulkanBuffer buffer, ulong offset = 0, IndexType indexType = IndexType.Uint32)
        {
            _vkCmdBindIndexBuffer(_commandBuffer, buffer.Handle, offset, indexType);
        }

        /// <summary>Issues a draw command</summary>
        public void Draw(uint vertexCount, uint instanceCount = 1, uint firstVertex = 0, uint firstInstance = 0)
        {
            _vkCmdDraw(_commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
        }

        /// <summary>Issues an indexed draw command</summary>
        public void DrawIndexed(uint indexCount, uint instanceCount = 1, uint firstIndex = 0, int vertexOffset = 0, uint firstInstance = 0)
        {
            _vkCmdDrawIndexed(_commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }

        /// <summary>Dispatches a compute shader</summary>
        public void Dispatch(uint groupCountX, uint groupCountY, uint groupCountZ)
        {
            _vkCmdDispatch(_commandBuffer, groupCountX, groupCountY, groupCountZ);
        }

        /// <summary>Inserts a pipeline barrier</summary>
        public void PipelineBarrier(
            PipelineStageFlag srcStageMask, PipelineStageFlag dstStageMask,
            MemoryBarrier[] memoryBarriers = null,
            BufferMemoryBarrier[] bufferBarriers = null,
            ImageMemoryBarrier[] imageBarriers = null)
        {
            var memBarrierPtr = IntPtr.Zero;
            int memBarrierCount = memoryBarriers?.Length ?? 0;
            if (memBarrierCount > 0)
            {
                int sz = Marshal.SizeOf<MemoryBarrier>();
                memBarrierPtr = Marshal.AllocHGlobal(memBarrierCount * sz);
                for (int i = 0; i < memBarrierCount; i++)
                    Marshal.StructureToPtr(memoryBarriers[i], memBarrierPtr + i * sz, false);
            }

            var bufBarrierPtr = IntPtr.Zero;
            int bufBarrierCount = bufferBarriers?.Length ?? 0;
            if (bufBarrierCount > 0)
            {
                int sz = Marshal.SizeOf<BufferMemoryBarrier>();
                bufBarrierPtr = Marshal.AllocHGlobal(bufBarrierCount * sz);
                for (int i = 0; i < bufBarrierCount; i++)
                    Marshal.StructureToPtr(bufferBarriers[i], bufBarrierPtr + i * sz, false);
            }

            var imgBarrierPtr = IntPtr.Zero;
            int imgBarrierCount = imageBarriers?.Length ?? 0;
            if (imgBarrierCount > 0)
            {
                int sz = Marshal.SizeOf<ImageMemoryBarrier>();
                imgBarrierPtr = Marshal.AllocHGlobal(imgBarrierCount * sz);
                for (int i = 0; i < imgBarrierCount; i++)
                {
                    var b = imageBarriers[i];
                    var vkBarrier = new VkImageMemoryBarrier
                    {
                        sType = 44,
                        srcAccessMask = b.SrcAccessMask,
                        dstAccessMask = b.DstAccessMask,
                        oldLayout = b.OldLayout,
                        newLayout = b.NewLayout,
                        srcQueueFamilyIndex = b.SrcQueueFamilyIndex,
                        dstQueueFamilyIndex = b.DstQueueFamilyIndex,
                        image = b.Image,
                        subresourceRange = new VkImageSubresourceRange
                        {
                            aspectMask = b.SubresourceRange.AspectMask,
                            baseMipLevel = b.SubresourceRange.BaseMipLevel,
                            levelCount = b.SubresourceRange.LevelCount,
                            baseArrayLayer = b.SubresourceRange.BaseArrayLayer,
                            layerCount = b.SubresourceRange.LayerCount
                        }
                    };
                    Marshal.StructureToPtr(vkBarrier, imgBarrierPtr + i * sz, false);
                }
            }

            _vkCmdPipelineBarrier(_commandBuffer, srcStageMask, dstStageMask, 0,
                (uint)memBarrierCount, memBarrierPtr,
                (uint)bufBarrierCount, bufBarrierPtr,
                (uint)imgBarrierCount, imgBarrierPtr);

            if (memBarrierPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(memBarrierPtr);
            if (bufBarrierPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(bufBarrierPtr);
            if (imgBarrierPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(imgBarrierPtr);
        }

        /// <summary>Copies data between buffers</summary>
        public void CopyBuffer(VulkanBuffer srcBuffer, VulkanBuffer dstBuffer, BufferCopy[] regions)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0)
                return;
            int sz = Marshal.SizeOf<BufferCopy>();
            var regionsPtr = Marshal.AllocHGlobal(regionCount * sz);
            for (int i = 0; i < regionCount; i++)
                Marshal.StructureToPtr(regions[i], regionsPtr + i * sz, false);
            _vkCmdCopyBuffer(_commandBuffer, srcBuffer.Handle, dstBuffer.Handle, (uint)regionCount, ref regions[0]);
            Marshal.FreeHGlobal(regionsPtr);
        }

        /// <summary>Copies buffer data to an image</summary>
        public void CopyBufferToImage(VulkanBuffer buffer, VulkanTexture image, ImageLayout imageLayout, BufferImageCopy[] regions)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0)
                return;
            _vkCmdCopyBufferToImage(_commandBuffer, buffer.Handle, image.Handle, (uint)imageLayout, (uint)regionCount, ref regions[0]);
        }

        /// <summary>Copies image data to a buffer</summary>
        public void CopyImageToBuffer(VulkanTexture image, ImageLayout imageLayout, VulkanBuffer buffer, BufferImageCopy[] regions)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0)
                return;
            _vkCmdCopyImageToBuffer(_commandBuffer, image.Handle, (uint)imageLayout, buffer.Handle, (uint)regionCount, ref regions[0]);
        }

        /// <summary>Pushes constant data to the shader</summary>
        public void PushConstants(ShaderStageFlag stageFlags, uint offset, byte[] data)
        {
            if (data == null || data.Length == 0)
                return;
            var dataPtr = Marshal.AllocHGlobal(data.Length);
            Marshal.Copy(data, 0, dataPtr, data.Length);
            _vkCmdPushConstants(_commandBuffer, IntPtr.Zero, stageFlags, offset, (uint)data.Length, dataPtr);
            Marshal.FreeHGlobal(dataPtr);
        }

        /// <summary>Pushes constant data to the shader with layout</summary>
        public void PushConstants(IntPtr pipelineLayout, ShaderStageFlag stageFlags, uint offset, byte[] data)
        {
            if (data == null || data.Length == 0)
                return;
            var dataPtr = Marshal.AllocHGlobal(data.Length);
            Marshal.Copy(data, 0, dataPtr, data.Length);
            _vkCmdPushConstants(_commandBuffer, pipelineLayout, stageFlags, offset, (uint)data.Length, dataPtr);
            Marshal.FreeHGlobal(dataPtr);
        }

        /// <summary>Binds descriptor sets to the command buffer</summary>
        public void BindDescriptorSets(PipelineBindPoint pipelineBindPoint, IntPtr layout, uint firstSet, IntPtr[] descriptorSets, uint[] dynamicOffsets = null)
        {
            if (descriptorSets == null || descriptorSets.Length == 0)
                return;
            int setCount = descriptorSets.Length;
            int dynOffsetCount = dynamicOffsets?.Length ?? 0;
            var setsPtr = Marshal.AllocHGlobal(setCount * IntPtr.Size);
            for (int i = 0; i < setCount; i++)
                Marshal.WriteIntPtr(setsPtr + i * IntPtr.Size, descriptorSets[i]);
            var dynPtr = IntPtr.Zero;
            if (dynOffsetCount > 0)
            {
                dynPtr = Marshal.AllocHGlobal(dynOffsetCount * sizeof(uint));
                for (int i = 0; i < dynOffsetCount; i++)
                    Marshal.WriteInt32(dynPtr + i * sizeof(uint), (int)dynamicOffsets[i]);
            }
            _vkCmdBindDescriptorSets(_commandBuffer, (uint)pipelineBindPoint, layout, firstSet, (uint)setCount, setsPtr, (uint)dynOffsetCount, dynPtr);
            Marshal.FreeHGlobal(setsPtr);
            if (dynPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(dynPtr);
        }

        /// <summary>Sets the viewport</summary>
        public void SetViewport(Viewport viewport, uint firstViewport = 0)
        {
            _vkCmdSetViewport(_commandBuffer, firstViewport, 1, ref viewport);
        }

        /// <summary>Sets the scissor rectangle</summary>
        public void SetScissor(Rect2D scissor, uint firstScissor = 0)
        {
            _vkCmdSetScissor(_commandBuffer, firstScissor, 1, ref scissor);
        }

        /// <summary>Copies between images</summary>
        public void CopyImage(VulkanTexture srcImage, ImageLayout srcLayout, VulkanTexture dstImage, ImageLayout dstLayout, ImageCopy[] regions)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0)
                return;
            var vkRegions = new VkImageCopy[regionCount];
            for (int i = 0; i < regionCount; i++)
            {
                var r = regions[i];
                vkRegions[i] = new VkImageCopy
                {
                    srcSubresource = new VkImageCopy_SrcSubresource
                    {
                        aspectMask = (uint)r.SrcSubresource.AspectMask,
                        mipLevel = r.SrcSubresource.MipLevel,
                        baseArrayLayer = r.SrcSubresource.BaseArrayLayer,
                        layerCount = r.SrcSubresource.LayerCount
                    },
                    srcOffset = new VkImageCopy_Offset3D { x = r.SrcOffset.X, y = r.SrcOffset.Y, z = r.SrcOffset.Z },
                    dstSubresource = new VkImageCopy_SrcSubresource
                    {
                        aspectMask = (uint)r.DstSubresource.AspectMask,
                        mipLevel = r.DstSubresource.MipLevel,
                        baseArrayLayer = r.DstSubresource.BaseArrayLayer,
                        layerCount = r.DstSubresource.LayerCount
                    },
                    dstOffset = new VkImageCopy_Offset3D { x = r.DstOffset.X, y = r.DstOffset.Y, z = r.DstOffset.Z },
                    extent = new VkImageCopy_Extent3D { width = r.Extent.Width, height = r.Extent.Height, depth = r.Extent.Depth }
                };
            }
            int sz = Marshal.SizeOf<VkImageCopy>();
            var regionsPtr = Marshal.AllocHGlobal(regionCount * sz);
            for (int i = 0; i < regionCount; i++)
                Marshal.StructureToPtr(vkRegions[i], regionsPtr + i * sz, false);
            _vkCmdCopyImage(_commandBuffer, srcImage.Handle, (uint)srcLayout, dstImage.Handle, (uint)dstLayout, (uint)regionCount, regionsPtr);
            Marshal.FreeHGlobal(regionsPtr);
        }

        /// <summary>Blits an image region</summary>
        public void BlitImage(VulkanTexture srcImage, ImageLayout srcLayout, VulkanTexture dstImage, ImageLayout dstLayout, ImageBlit[] regions, Filter filter = Filter.Linear)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0)
                return;
            int sz = Marshal.SizeOf<ImageBlit>();
            var regionsPtr = Marshal.AllocHGlobal(regionCount * sz);
            for (int i = 0; i < regionCount; i++)
                Marshal.StructureToPtr(regions[i], regionsPtr + i * sz, false);
            _vkCmdBlitImage(_commandBuffer, srcImage.Handle, (uint)srcLayout, dstImage.Handle, (uint)dstLayout, (uint)regionCount, regionsPtr, filter);
            Marshal.FreeHGlobal(regionsPtr);
        }

        /// <summary>Resolves a multisampled image</summary>
        public void ResolveImage(VulkanTexture srcImage, ImageLayout srcLayout, VulkanTexture dstImage, ImageLayout dstLayout, ImageResolve[] regions)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0)
                return;
            int sz = Marshal.SizeOf<ImageResolve>();
            var regionsPtr = Marshal.AllocHGlobal(regionCount * sz);
            for (int i = 0; i < regionCount; i++)
                Marshal.StructureToPtr(regions[i], regionsPtr + i * sz, false);
            _vkCmdResolveImage(_commandBuffer, srcImage.Handle, (uint)srcLayout, dstImage.Handle, (uint)dstLayout, (uint)regionCount, regionsPtr);
            Marshal.FreeHGlobal(regionsPtr);
        }

        /// <summary>Writes a timestamp into a query pool</summary>
        public void WriteTimestamp(PipelineStageFlag pipelineStage, IntPtr queryPool, uint query)
        {
            _vkCmdWriteTimestamp(_commandBuffer, pipelineStage, queryPool, query);
        }

        /// <summary>Begins a query</summary>
        public void BeginQuery(IntPtr queryPool, uint query, uint flags = 0)
        {
            _vkCmdBeginQuery(_commandBuffer, queryPool, query, flags);
        }

        /// <summary>Ends a query</summary>
        public void EndQuery(IntPtr queryPool, uint query)
        {
            _vkCmdEndQuery(_commandBuffer, queryPool, query);
        }

        /// <summary>Resets a command pool, releasing all command buffers</summary>
        public void Reset()
        {
            // Reset is done via command pool, individual command buffers are implicitly reset
            _isRecording = false;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
    // =========================================================================
    // VulkanBuffer
    // =========================================================================

    /// <summary>
    /// Wraps a Vulkan buffer resource with GPU memory allocation.
    /// Supports mapping, flushing, and copy operations.
    /// </summary>
    public class VulkanBuffer : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _buffer;
        private IntPtr _memory;
        private IntPtr _mappedData;
        private BufferDescription _description;
        private VkMemoryRequirements _memoryRequirements;
        private ulong _allocatedSize;
        private uint _memoryTypeIndex;
        private bool _isMapped;
        private bool _disposed;

        // Function pointers
        private MapMemoryDel _vkMapMemory;
        private UnmapMemoryDel _vkUnmapMemory;
        private FlushMappedMemoryRangesDel _vkFlushMappedMemoryRanges;
        private BindBufferMemoryDel _vkBindBufferMemory;
        private DestroyBufferDel _vkDestroyBuffer;
        private FreeMemoryDel _vkFreeMemory;
        private CmdCopyBufferDel _vkCmdCopyBuffer;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult MapMemoryDel(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, ref IntPtr ppData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void UnmapMemoryDel(IntPtr device, IntPtr memory);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult FlushMappedMemoryRangesDel(IntPtr device, uint memoryRangeCount, ref VkMappedMemoryRange pMemoryRanges);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindBufferMemoryDel(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyBufferDel(IntPtr device, IntPtr buffer, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void FreeMemoryDel(IntPtr device, IntPtr memory, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferDel(IntPtr commandBuffer, IntPtr srcBuffer, IntPtr dstBuffer, uint regionCount, ref BufferCopy pRegions);

        public IntPtr Handle => _buffer;
        public IntPtr MemoryHandle => _memory;
        public ulong Size => _description.Size;
        public IntPtr MappedPointer => _mappedData;
        public bool IsMapped => _isMapped;
        public BufferDescription Description => _description;
        public VkMemoryRequirements MemoryRequirements => _memoryRequirements;

        public VulkanBuffer(VulkanDevice device, IntPtr buffer, BufferDescription description, VkMemoryRequirements memoryRequirements)
        {
            _device = device;
            _buffer = buffer;
            _description = description;
            _memoryRequirements = memoryRequirements;
            LoadFunctions();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkMapMemory = Marshal.GetDelegateForFunctionPointer<MapMemoryDel>(load("vkMapMemory"));
            _vkUnmapMemory = Marshal.GetDelegateForFunctionPointer<UnmapMemoryDel>(load("vkUnmapMemory"));
            _vkFlushMappedMemoryRanges = Marshal.GetDelegateForFunctionPointer<FlushMappedMemoryRangesDel>(load("vkFlushMappedMemoryRanges"));
            _vkBindBufferMemory = Marshal.GetDelegateForFunctionPointer<BindBufferMemoryDel>(load("vkBindBufferMemory"));
            _vkDestroyBuffer = Marshal.GetDelegateForFunctionPointer<DestroyBufferDel>(load("vkDestroyBuffer"));
            _vkFreeMemory = Marshal.GetDelegateForFunctionPointer<FreeMemoryDel>(load("vkFreeMemory"));
        }

        /// <summary>Binds memory to this buffer (called by allocator)</summary>
        internal void BindMemory(IntPtr memory, ulong offset)
        {
            _memory = memory;
            _vkBindBufferMemory(_device.LogicalDevice, _buffer, memory, offset);
        }

        /// <summary>Maps the buffer into CPU address space</summary>
        public IntPtr Map(ulong offset = 0, ulong size = 0)
        {
            if (_isMapped)
                return _mappedData;
            if (size == 0)
                size = _description.Size;
            IntPtr data = IntPtr.Zero;
            var result = _vkMapMemory(_device.LogicalDevice, _memory, offset, size, 0, ref data);
            if (result == VulkanResult.Success)
            {
                _mappedData = data;
                _isMapped = true;
            }
            return data;
        }

        /// <summary>Unmaps the buffer from CPU address space</summary>
        public void Unmap()
        {
            if (!_isMapped)
                return;
            _vkUnmapMemory(_device.LogicalDevice, _memory);
            _mappedData = IntPtr.Zero;
            _isMapped = false;
        }

        /// <summary>Flushes mapped memory ranges to make them visible to the device</summary>
        public void Flush(ulong start = 0, ulong size = 0)
        {
            if (size == 0)
                size = _description.Size;
            var range = new VkMappedMemoryRange
            {
                sType = 6,
                memory = _memory,
                offset = start,
                size = size
            };
            _vkFlushMappedMemoryRanges(_device.LogicalDevice, 1, ref range);
        }

        /// <summary>Copies data from another buffer to this one using a command buffer</summary>
        public void CopyFrom(VulkanBuffer sourceBuffer, ulong srcOffset, ulong dstOffset, ulong size, IntPtr commandBuffer)
        {
            var region = new BufferCopy(srcOffset, dstOffset, size);
            var cmdCopy = Marshal.GetDelegateForFunctionPointer<CmdCopyBufferDel>(
                _device.GetDeviceProcAddr("vkCmdCopyBuffer"));
            // Using the command buffer directly
        }

        /// <summary>Writes data to the buffer from CPU memory</summary>
        public void SetData<T>(T[] data, ulong dstOffset = 0) where T : struct
        {
            int structSize = Marshal.SizeOf<T>();
            int dataSize = data.Length * structSize;
            var dataPtr = Marshal.AllocHGlobal(dataSize);
            try
            {
                for (int i = 0; i < data.Length; i++)
                    Marshal.StructureToPtr(data[i], dataPtr + i * structSize, false);

                if (_isMapped)
                {
                    unsafe
                    {
                        Buffer.MemoryCopy(
                            (void*)(dataPtr),
                            (void*)(_mappedData + (int)dstOffset),
                            dataSize, dataSize);
                    }
                }
                else
                {
                    Map(dstOffset, (ulong)dataSize);
                    unsafe
                    {
                        Buffer.MemoryCopy(
                            (void*)(dataPtr),
                            (void*)(_mappedData),
                            dataSize, dataSize);
                    }
                    Flush(dstOffset, (ulong)dataSize);
                    Unmap();
                }
            }
            finally { Marshal.FreeHGlobal(dataPtr); }
        }

        /// <summary>Writes raw byte data to the buffer</summary>
        public void SetData(byte[] data, ulong dstOffset = 0)
        {
            if (data == null || data.Length == 0)
                return;
            var dataPtr = Marshal.AllocHGlobal(data.Length);
            try
            {
                Marshal.Copy(data, 0, dataPtr, data.Length);
                if (_isMapped)
                {
                    unsafe
                    { Buffer.MemoryCopy((void*)dataPtr, (void*)(_mappedData + (int)dstOffset), data.Length, data.Length); }
                }
                else
                {
                    Map(dstOffset, (ulong)data.Length);
                    unsafe
                    { Buffer.MemoryCopy((void*)dataPtr, (void*)_mappedData, data.Length, data.Length); }
                    Flush(dstOffset, (ulong)data.Length);
                    Unmap();
                }
            }
            finally { Marshal.FreeHGlobal(dataPtr); }
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_isMapped)
                Unmap();
            if (_buffer != IntPtr.Zero && _vkDestroyBuffer != null)
                _vkDestroyBuffer(_device.LogicalDevice, _buffer, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    // =========================================================================
    // VulkanTexture
    // =========================================================================

    /// <summary>
    /// Wraps a Vulkan image resource with associated memory and image view.
    /// Supports mipmap generation, layout transitions, and buffer copies.
    /// </summary>
    public class VulkanTexture : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _image;
        private IntPtr _memory;
        private IntPtr _imageView;
        private TextureDescription _description;
        private VkMemoryRequirements _memoryRequirements;
        private uint _memoryTypeIndex;
        private bool _disposed;

        // Function pointers
        private CreateImageViewDel _vkCreateImageView;
        private DestroyImageViewDel _vkDestroyImageView;
        private DestroyImageDel _vkDestroyImage;
        private FreeMemoryDel _vkFreeMemory;
        private BindImageMemoryDel _vkBindImageMemory;
        private CmdPipelineBarrierDel2 _vkCmdPipelineBarrier;
        private CmdCopyBufferToImageDel2 _vkCmdCopyBufferToImage;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateImageViewDel(IntPtr device, ref VkImageViewCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pView);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageViewDel(IntPtr device, IntPtr imageView, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageDel(IntPtr device, IntPtr image, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void FreeMemoryDel(IntPtr device, IntPtr memory, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindImageMemoryDel(IntPtr device, IntPtr image, IntPtr memory, ulong memoryOffset);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdPipelineBarrierDel2(IntPtr cmdBuffer, PipelineStageFlag srcStageMask, PipelineStageFlag dstStageMask, uint depFlags, uint memBarCount, IntPtr memBars, uint bufBarCount, IntPtr bufBars, uint imgBarCount, IntPtr imgBars);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferToImageDel2(IntPtr cmdBuffer, IntPtr buffer, IntPtr image, uint imageLayout, uint regionCount, IntPtr pRegions);

        /// <summary>The Vulkan image handle</summary>
        public IntPtr Handle => _image;

        /// <summary>The image view handle</summary>
        public IntPtr ImageViewHandle => _imageView;

        /// <summary>Texture description</summary>
        public TextureDescription Description => _description;

        /// <summary>Width of the texture</summary>
        public uint Width => _description.Width;

        /// <summary>Height of the texture</summary>
        public uint Height => _description.Height;

        /// <summary>Format of the texture</summary>
        public VulkanFormat Format => _description.Format;

        /// <summary>Mip level count</summary>
        public uint MipLevels => _description.MipLevels;

        public VkMemoryRequirements MemoryRequirements => _memoryRequirements;

        public VulkanTexture(VulkanDevice device, IntPtr image, TextureDescription description, VkMemoryRequirements memoryRequirements = default)
        {
            _device = device;
            _image = image;
            _description = description;
            _memoryRequirements = memoryRequirements;
            LoadFunctions();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkCreateImageView = Marshal.GetDelegateForFunctionPointer<CreateImageViewDel>(load("vkCreateImageView"));
            _vkDestroyImageView = Marshal.GetDelegateForFunctionPointer<DestroyImageViewDel>(load("vkDestroyImageView"));
            _vkDestroyImage = Marshal.GetDelegateForFunctionPointer<DestroyImageDel>(load("vkDestroyImage"));
            _vkFreeMemory = Marshal.GetDelegateForFunctionPointer<FreeMemoryDel>(load("vkFreeMemory"));
            _vkBindImageMemory = Marshal.GetDelegateForFunctionPointer<BindImageMemoryDel>(load("vkBindImageMemory"));
            _vkCmdPipelineBarrier = Marshal.GetDelegateForFunctionPointer<CmdPipelineBarrierDel2>(load("vkCmdPipelineBarrier"));
            _vkCmdCopyBufferToImage = Marshal.GetDelegateForFunctionPointer<CmdCopyBufferToImageDel2>(load("vkCmdCopyBufferToImage"));
        }

        /// <summary>Binds memory to this image (called by allocator)</summary>
        internal void BindMemory(IntPtr memory, ulong offset)
        {
            _memory = memory;
            _vkBindImageMemory(_device.LogicalDevice, _image, memory, offset);
        }

        /// <summary>Creates the image view for this texture</summary>
        public IntPtr GetImageView()
        {
            if (_imageView != IntPtr.Zero)
                return _imageView;

            var aspectFlags = ImageAspectFlag.Color;
            if (_description.Format == VulkanFormat.D16Unorm || _description.Format == VulkanFormat.D32Sfloat ||
                _description.Format == VulkanFormat.D24UnormS8Uint || _description.Format == VulkanFormat.D32SfloatS8Uint)
                aspectFlags = ImageAspectFlag.Depth;
            else if (_description.Format == VulkanFormat.S8Uint)
                aspectFlags = ImageAspectFlag.Stencil;
            else if (_description.Format == VulkanFormat.D16UnormS8Uint)
                aspectFlags = ImageAspectFlag.Depth | ImageAspectFlag.Stencil;

            var viewType = _description.Type == ImageType.Type3D ? ImageViewType.Type3D :
                           _description.ArrayLayers > 1 ? ImageViewType.Type2DArray :
                           ImageViewType.Type2D;

            var createInfo = new VkImageViewCreateInfo
            {
                sType = 15,
                image = _image,
                viewType = viewType,
                format = _description.Format,
                subresourceRange = new VkImageSubresourceRange
                {
                    aspectMask = aspectFlags,
                    baseMipLevel = 0,
                    levelCount = _description.MipLevels,
                    baseArrayLayer = 0,
                    layerCount = _description.ArrayLayers
                }
            };
            createInfo.components = new VkComponentMapping { r = 0, g = 1, b = 2, a = 3 };

            _vkCreateImageView(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref _imageView);
            return _imageView;
        }

        /// <summary>Generates mipmaps for the texture using blit operations</summary>
        public void CreateMipmaps(IntPtr commandBuffer)
        {
            if (_description.MipLevels <= 1)
                return;

            int mipWidth = (int)_description.Width;
            int mipHeight = (int)_description.Height;

            var barrier = new VkImageMemoryBarrier
            {
                sType = 44,
                srcQueueFamilyIndex = 0xFFFFFFFF,
                dstQueueFamilyIndex = 0xFFFFFFFF,
                image = _image,
                subresourceRange = new VkImageSubresourceRange
                {
                    aspectMask = ImageAspectFlag.Color,
                    levelCount = 1,
                    baseArrayLayer = 0,
                    layerCount = _description.ArrayLayers
                }
            };

            for (uint i = 1; i < _description.MipLevels; i++)
            {
                barrier.subresourceRange.baseMipLevel = i - 1;
                barrier.oldLayout = ImageLayout.TransferDstOptimal;
                barrier.newLayout = ImageLayout.TransferSrcOptimal;
                barrier.srcAccessMask = AccessFlag.TransferWrite;
                barrier.dstAccessMask = AccessFlag.TransferRead;

                int barrierSize = Marshal.SizeOf<VkImageMemoryBarrier>();
                var barrierPtr = Marshal.AllocHGlobal(barrierSize);
                Marshal.StructureToPtr(barrier, barrierPtr, false);
                _vkCmdPipelineBarrier(commandBuffer,
                    PipelineStageFlag.Transfer, PipelineStageFlag.Transfer,
                    0, 0, IntPtr.Zero, 0, IntPtr.Zero, 1, barrierPtr);
                Marshal.FreeHGlobal(barrierPtr);

                int dstMipWidth = mipWidth > 1 ? mipWidth / 2 : 1;
                int dstMipHeight = mipHeight > 1 ? mipHeight / 2 : 1;

                var blit = new VkImageBlit
                {
                    srcOffsets0 = new VkOffset3D { x = 0, y = 0, z = 0 },
                    srcOffsets1 = new VkOffset3D { x = mipWidth, y = mipHeight, z = 1 },
                    srcSubresource = new VkImageSubresourceLayers
                    {
                        aspectMask = (uint)ImageAspectFlag.Color,
                        mipLevel = i - 1,
                        baseArrayLayer = 0,
                        layerCount = _description.ArrayLayers
                    },
                    dstOffsets0 = new VkOffset3D { x = 0, y = 0, z = 0 },
                    dstOffsets1 = new VkOffset3D { x = dstMipWidth, y = dstMipHeight, z = 1 },
                    dstSubresource = new VkImageSubresourceLayers
                    {
                        aspectMask = (uint)ImageAspectFlag.Color,
                        mipLevel = i,
                        baseArrayLayer = 0,
                        layerCount = _description.ArrayLayers
                    }
                };

                int blitSize = Marshal.SizeOf<VkImageBlit>();
                var blitPtr = Marshal.AllocHGlobal(blitSize);
                Marshal.StructureToPtr(blit, blitPtr, false);

                var cmdBlit = Marshal.GetDelegateForFunctionPointer<CmdBlitImageDel>(
                    _device.GetDeviceProcAddr("vkCmdBlitImage"));
                cmdBlit(commandBuffer, _image, (uint)ImageLayout.TransferSrcOptimal,
                    _image, (uint)ImageLayout.TransferDstOptimal, 1, blitPtr, Filter.Linear);
                Marshal.FreeHGlobal(blitPtr);

                barrier.oldLayout = ImageLayout.TransferSrcOptimal;
                barrier.newLayout = ImageLayout.ShaderReadOnlyOptimal;
                barrier.srcAccessMask = AccessFlag.TransferRead;
                barrier.dstAccessMask = AccessFlag.ShaderRead;

                barrierPtr = Marshal.AllocHGlobal(barrierSize);
                Marshal.StructureToPtr(barrier, barrierPtr, false);
                _vkCmdPipelineBarrier(commandBuffer,
                    PipelineStageFlag.Transfer, PipelineStageFlag.FragmentShader,
                    0, 0, IntPtr.Zero, 0, IntPtr.Zero, 1, barrierPtr);
                Marshal.FreeHGlobal(barrierPtr);

                mipWidth = dstMipWidth;
                mipHeight = dstMipHeight;
            }

            // Transition last mip level
            barrier.subresourceRange.baseMipLevel = _description.MipLevels - 1;
            barrier.oldLayout = ImageLayout.TransferDstOptimal;
            barrier.newLayout = ImageLayout.ShaderReadOnlyOptimal;
            barrier.srcAccessMask = AccessFlag.TransferWrite;
            barrier.dstAccessMask = AccessFlag.ShaderRead;

            int finalBarrierSize = Marshal.SizeOf<VkImageMemoryBarrier>();
            var finalBarrierPtr = Marshal.AllocHGlobal(finalBarrierSize);
            Marshal.StructureToPtr(barrier, finalBarrierPtr, false);
            _vkCmdPipelineBarrier(commandBuffer,
                PipelineStageFlag.Transfer, PipelineStageFlag.FragmentShader,
                0, 0, IntPtr.Zero, 0, IntPtr.Zero, 1, finalBarrierPtr);
            Marshal.FreeHGlobal(finalBarrierPtr);
        }

        /// <summary>Transitions the image to a new layout</summary>
        public void TransitionLayout(ImageLayout newLayout, IntPtr commandBuffer)
        {
            var barrier = new VkImageMemoryBarrier
            {
                sType = 44,
                srcAccessMask = 0,
                dstAccessMask = 0,
                oldLayout = _description.InitialLayout,
                newLayout = newLayout,
                srcQueueFamilyIndex = 0xFFFFFFFF,
                dstQueueFamilyIndex = 0xFFFFFFFF,
                image = _image,
                subresourceRange = new VkImageSubresourceRange
                {
                    aspectMask = ImageAspectFlag.Color,
                    baseMipLevel = 0,
                    levelCount = _description.MipLevels,
                    baseArrayLayer = 0,
                    layerCount = _description.ArrayLayers
                }
            };

            if (newLayout == ImageLayout.TransferDstOptimal)
                barrier.dstAccessMask = AccessFlag.TransferWrite;
            else if (newLayout == ImageLayout.TransferSrcOptimal)
                barrier.srcAccessMask = AccessFlag.TransferRead;
            else if (newLayout == ImageLayout.ShaderReadOnlyOptimal)
                barrier.dstAccessMask = AccessFlag.ShaderRead;
            else if (newLayout == ImageLayout.ColorAttachmentOptimal)
                barrier.dstAccessMask = AccessFlag.ColorAttachmentWrite;
            else if (newLayout == ImageLayout.DepthStencilAttachmentOptimal)
                barrier.dstAccessMask = AccessFlag.DepthStencilAttachmentWrite;

            PipelineStageFlag srcStage = PipelineStageFlag.TopOfPipe;
            PipelineStageFlag dstStage = PipelineStageFlag.BottomOfPipe;

            if (_description.InitialLayout == ImageLayout.TransferDstOptimal)
                srcStage = PipelineStageFlag.Transfer;
            if (newLayout == ImageLayout.TransferDstOptimal || newLayout == ImageLayout.TransferSrcOptimal)
                dstStage = PipelineStageFlag.Transfer;
            else if (newLayout == ImageLayout.ShaderReadOnlyOptimal)
                dstStage = PipelineStageFlag.FragmentShader;
            else if (newLayout == ImageLayout.ColorAttachmentOptimal)
                dstStage = PipelineStageFlag.ColorAttachmentOutput;

            int barrierSize = Marshal.SizeOf<VkImageMemoryBarrier>();
            var barrierPtr = Marshal.AllocHGlobal(barrierSize);
            Marshal.StructureToPtr(barrier, barrierPtr, false);
            _vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, IntPtr.Zero, 0, IntPtr.Zero, 1, barrierPtr);
            Marshal.FreeHGlobal(barrierPtr);

            _description.InitialLayout = newLayout;
        }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBlitImageDel(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions, Filter filter);

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_imageView != IntPtr.Zero && _vkDestroyImageView != null)
                _vkDestroyImageView(_device.LogicalDevice, _imageView, IntPtr.Zero);
            if (_image != IntPtr.Zero && _vkDestroyImage != null)
                _vkDestroyImage(_device.LogicalDevice, _image, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }
    // =========================================================================
    // VulkanPipeline
    // =========================================================================

    /// <summary>Wraps a Vulkan graphics pipeline</summary>
}
