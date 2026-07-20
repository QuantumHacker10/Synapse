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
    public class VulkanResourceTracker : IDisposable
    {
        private VulkanDevice _device;
        private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();
        private bool _disposed;

        // Resource tracking
        private readonly Dictionary<IntPtr, TrackedResource> _trackedResources = new Dictionary<IntPtr, TrackedResource>();

        // Deferred deletion queue
        private readonly Queue<DeferredDeletion> _deletionQueue = new Queue<DeferredDeletion>();
        private readonly List<DeferredDeletion> _activeDeletions = new List<DeferredDeletion>();

        // Resource state tracking
        private readonly Dictionary<IntPtr, ResourceState> _resourceStates = new Dictionary<IntPtr, ResourceState>();

        // Statistics
        private long _totalTracked;
        private long _totalDeleted;
        private long _pendingDeletions;

        // Function pointers
        private DestroyBufferDel _vkDestroyBuffer;
        private DestroyImageDel _vkDestroyImage;
        private DestroyImageViewDel _vkDestroyImageView;
        private DestroySamplerDel _vkDestroySampler;
        private DestroyRenderPassDel _vkDestroyRenderPass;
        private DestroyFramebufferDel _vkDestroyFramebuffer;
        private DestroyPipelineDel _vkDestroyPipeline;
        private DestroyPipelineLayoutDel _vkDestroyPipelineLayout;
        private DestroyShaderModuleDel _vkDestroyShaderModule;
        private DestroyDescriptorSetLayoutDel _vkDestroyDescriptorSetLayout;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyBufferDel(IntPtr device, IntPtr buffer, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageDel(IntPtr device, IntPtr image, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageViewDel(IntPtr device, IntPtr imageView, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroySamplerDel(IntPtr device, IntPtr sampler, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyRenderPassDel(IntPtr device, IntPtr renderPass, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyFramebufferDel(IntPtr device, IntPtr framebuffer, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineDel(IntPtr device, IntPtr pipeline, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineLayoutDel(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyShaderModuleDel(IntPtr device, IntPtr shaderModule, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyDescriptorSetLayoutDel(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);

        public int TrackedCount => _trackedResources.Count;
        public int PendingDeletions => _deletionQueue.Count;

        public VulkanResourceTracker(VulkanDevice device)
        {
            _device = device;
            LoadFunctions();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkDestroyBuffer = Marshal.GetDelegateForFunctionPointer<DestroyBufferDel>(load("vkDestroyBuffer"));
            _vkDestroyImage = Marshal.GetDelegateForFunctionPointer<DestroyImageDel>(load("vkDestroyImage"));
            _vkDestroyImageView = Marshal.GetDelegateForFunctionPointer<DestroyImageViewDel>(load("vkDestroyImageView"));
            _vkDestroySampler = Marshal.GetDelegateForFunctionPointer<DestroySamplerDel>(load("vkDestroySampler"));
            _vkDestroyRenderPass = Marshal.GetDelegateForFunctionPointer<DestroyRenderPassDel>(load("vkDestroyRenderPass"));
            _vkDestroyFramebuffer = Marshal.GetDelegateForFunctionPointer<DestroyFramebufferDel>(load("vkDestroyFramebuffer"));
            _vkDestroyPipeline = Marshal.GetDelegateForFunctionPointer<DestroyPipelineDel>(load("vkDestroyPipeline"));
            _vkDestroyPipelineLayout = Marshal.GetDelegateForFunctionPointer<DestroyPipelineLayoutDel>(load("vkDestroyPipelineLayout"));
            _vkDestroyShaderModule = Marshal.GetDelegateForFunctionPointer<DestroyShaderModuleDel>(load("vkDestroyShaderModule"));
            _vkDestroyDescriptorSetLayout = Marshal.GetDelegateForFunctionPointer<DestroyDescriptorSetLayoutDel>(load("vkDestroyDescriptorSetLayout"));
        }

        /// <summary>Tracks a resource for lifetime management</summary>
        public void TrackResource(VulkanBuffer resource)
        {
            if (resource == null)
                return;
            _lock.EnterWriteLock();
            try
            {
                _trackedResources[resource.Handle] = new TrackedResource
                {
                    Handle = resource.Handle,
                    Type = ResourceType.Buffer,
                    ReferenceCount = 1,
                    CreationFrame = Interlocked.Read(ref _totalTracked)
                };
                Interlocked.Increment(ref _totalTracked);
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Tracks a texture resource</summary>
        public void TrackResource(VulkanTexture resource)
        {
            if (resource == null)
                return;
            _lock.EnterWriteLock();
            try
            {
                _trackedResources[resource.Handle] = new TrackedResource
                {
                    Handle = resource.Handle,
                    Type = ResourceType.Image,
                    ReferenceCount = 1,
                    CreationFrame = Interlocked.Read(ref _totalTracked)
                };
                Interlocked.Increment(ref _totalTracked);
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Tracks a render pass resource</summary>
        public void TrackResource(VulkanRenderPass resource)
        {
            if (resource == null)
                return;
            _lock.EnterWriteLock();
            try
            {
                _trackedResources[resource.Handle] = new TrackedResource
                {
                    Handle = resource.Handle,
                    Type = ResourceType.RenderPass,
                    ReferenceCount = 1,
                    CreationFrame = Interlocked.Read(ref _totalTracked)
                };
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Adds a reference to a tracked resource</summary>
        public void AddReference(IntPtr handle)
        {
            _lock.EnterUpgradeableReadLock();
            try
            {
                if (_trackedResources.TryGetValue(handle, out var resource))
                {
                    _lock.EnterWriteLock();
                    try
                    { resource.ReferenceCount++; }
                    finally { _lock.ExitWriteLock(); }
                }
            }
            finally { _lock.ExitUpgradeableReadLock(); }
        }

        /// <summary>Removes a reference and queues for deletion if count reaches zero</summary>
        public void RemoveReference(IntPtr handle)
        {
            _lock.EnterUpgradeableReadLock();
            try
            {
                if (_trackedResources.TryGetValue(handle, out var resource))
                {
                    _lock.EnterWriteLock();
                    try
                    {
                        resource.ReferenceCount--;
                        if (resource.ReferenceCount <= 0)
                        {
                            QueueDeletion(resource);
                            _trackedResources.Remove(handle);
                        }
                    }
                    finally { _lock.ExitWriteLock(); }
                }
            }
            finally { _lock.ExitUpgradeableReadLock(); }
        }

        /// <summary>Queues a resource for deferred deletion</summary>
        private void QueueDeletion(TrackedResource resource)
        {
            var deletion = new DeferredDeletion
            {
                Resource = resource,
                DeletionFrame = Interlocked.Read(ref _totalTracked),
                FramesToWait = 3 // Triple-buffered safety
            };
            _deletionQueue.Enqueue(deletion);
            Interlocked.Increment(ref _pendingDeletions);
        }

        /// <summary>Processes deferred deletions for completed frames</summary>
        public void ProcessDeletions()
        {
            _lock.EnterWriteLock();
            try
            {
                while (_deletionQueue.Count > 0)
                {
                    var deletion = _deletionQueue.Peek();
                    long currentFrame = Interlocked.Read(ref _totalTracked);

                    if (currentFrame - deletion.DeletionFrame >= deletion.FramesToWait)
                    {
                        _deletionQueue.Dequeue();
                        DestroyResource(deletion.Resource);
                        Interlocked.Decrement(ref _pendingDeletions);
                        Interlocked.Increment(ref _totalDeleted);
                    }
                    else
                    {
                        break;
                    }
                }
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Destroys a tracked resource</summary>
        private void DestroyResource(TrackedResource resource)
        {
            switch (resource.Type)
            {
                case ResourceType.Buffer:
                    _vkDestroyBuffer?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.Image:
                    _vkDestroyImage?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.ImageView:
                    _vkDestroyImageView?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.Sampler:
                    _vkDestroySampler?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.RenderPass:
                    _vkDestroyRenderPass?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.Framebuffer:
                    _vkDestroyFramebuffer?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.Pipeline:
                    _vkDestroyPipeline?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.PipelineLayout:
                    _vkDestroyPipelineLayout?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.ShaderModule:
                    _vkDestroyShaderModule?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.DescriptorSetLayout:
                    _vkDestroyDescriptorSetLayout?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
            }
        }

        /// <summary>Updates the state of a tracked resource</summary>
        public void SetResourceState(IntPtr handle, ResourceState state)
        {
            _lock.EnterWriteLock();
            try
            { _resourceStates[handle] = state; }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Gets the current state of a tracked resource</summary>
        public ResourceState GetResourceState(IntPtr handle)
        {
            _lock.EnterReadLock();
            try
            {
                return _resourceStates.TryGetValue(handle, out var state) ? state : ResourceState.Unknown;
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Forces immediate deletion of all pending resources</summary>
        public void FlushAll()
        {
            _lock.EnterWriteLock();
            try
            {
                while (_deletionQueue.Count > 0)
                {
                    var deletion = _deletionQueue.Dequeue();
                    DestroyResource(deletion.Resource);
                    Interlocked.Decrement(ref _pendingDeletions);
                    Interlocked.Increment(ref _totalDeleted);
                }
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Gets tracking statistics</summary>
        public TrackerStats GetStats()
        {
            return new TrackerStats
            {
                TotalTracked = _totalTracked,
                TotalDeleted = _totalDeleted,
                PendingDeletions = _pendingDeletions,
                ActiveResources = _trackedResources.Count
            };
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            FlushAll();
            _lock.Dispose();
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Tracked resource entry</summary>
    internal class TrackedResource
    {
        public IntPtr Handle { get; set; }
        public ResourceType Type { get; set; }
        public int ReferenceCount { get; set; }
        public long CreationFrame { get; set; }
    }

    /// <summary>Deferred deletion entry</summary>
    internal class DeferredDeletion
    {
        public TrackedResource Resource { get; set; }
        public long DeletionFrame { get; set; }
        public int FramesToWait { get; set; }
    }

    /// <summary>Resource types for tracking</summary>
    public enum ResourceType
    {
        Buffer,
        Image,
        ImageView,
        Sampler,
        RenderPass,
        Framebuffer,
        Pipeline,
        PipelineLayout,
        ShaderModule,
        DescriptorSetLayout,
        CommandBuffer,
        QueryPool
    }

    /// <summary>Resource state for state tracking</summary>
    public enum ResourceState
    {
        Unknown,
        Common,
        VertexBuffer,
        IndexBuffer,
        ConstantBuffer,
        StreamOutput,
        IndirectArgument,
        Predication,
        ShaderResource,
        UnorderedAccess,
        RenderTarget,
        DepthWrite,
        DepthRead,
        NonPixelShaderResource,
        PixelShaderResource,
        CopyDest,
        CopySource,
        ResolveDest,
        ResolveSource,
        GenericRead,
        Present
    }

    /// <summary>Tracker statistics</summary>
    public class TrackerStats
    {
        public long TotalTracked { get; set; }
        public long TotalDeleted { get; set; }
        public long PendingDeletions { get; set; }
        public int ActiveResources { get; set; }
    }
    // =========================================================================
    // EXTENSION METHODS
    // =========================================================================

    /// <summary>Extension methods for Vulkan utility types</summary>
    public static class VulkanExtensions
    {
        /// <summary>Calculates the aligned size for a given size and alignment</summary>
        public static ulong AlignSize(this ulong size, ulong alignment)
        {
            return (size + alignment - 1) & ~(alignment - 1);
        }

        /// <summary>Calculates the number of mip levels for a texture of given dimensions</summary>
        public static uint CalculateMipLevels(uint width, uint height, uint depth = 1)
        {
            return (uint)(Math.Floor(Math.Log2(Math.Max(width, Math.Max(height, depth)))) + 1);
        }

        /// <summary>Calculates the row pitch for a texture format and width</summary>
        public static uint CalculateRowPitch(VulkanFormat format, uint width, uint alignment = 4)
        {
            uint bitsPerPixel = GetBitsPerPixel(format);
            uint rowBytes = (width * bitsPerPixel + 7) / 8;
            return (rowBytes + alignment - 1) & ~(alignment - 1);
        }

        /// <summary>Calculates the total image size in bytes</summary>
        public static ulong CalculateImageSize(VulkanFormat format, uint width, uint height, uint depth = 1, uint mipLevel = 0)
        {
            uint mipWidth = Math.Max(1, width >> (int)mipLevel);
            uint mipHeight = Math.Max(1, height >> (int)mipLevel);
            uint mipDepth = Math.Max(1, depth >> (int)mipLevel);

            if (IsCompressedFormat(format))
                return CalculateCompressedSize(format, mipWidth, mipHeight, mipDepth);

            uint bitsPerPixel = GetBitsPerPixel(format);
            ulong rowPitch = CalculateRowPitch(format, mipWidth, 4);
            return rowPitch * mipHeight * mipDepth;
        }

        /// <summary>Gets the bits per pixel for a format</summary>
        public static uint GetBitsPerPixel(VulkanFormat format)
        {
            return format switch
            {
                VulkanFormat.R4G4UnormPack8 => 8,
                VulkanFormat.R4G4B4A4UnormPack16 or VulkanFormat.B4G4R4A4UnormPack16 => 16,
                VulkanFormat.R5G6B5UnormPack16 or VulkanFormat.B5G6R5UnormPack16 => 16,
                VulkanFormat.R5G5B5A1UnormPack16 or VulkanFormat.B5G5R5A1UnormPack16 or VulkanFormat.A1R5G5B5UnormPack16 => 16,
                VulkanFormat.R8Unorm or VulkanFormat.R8Snorm or VulkanFormat.R8Uscaled or VulkanFormat.R8Sscaled or
                VulkanFormat.R8Uint or VulkanFormat.R8Sint or VulkanFormat.R8Srgb => 8,
                VulkanFormat.R8G8Unorm or VulkanFormat.R8G8Snorm or VulkanFormat.R8G8Uscaled or VulkanFormat.R8G8Sscaled or
                VulkanFormat.R8G8Uint or VulkanFormat.R8G8Sint or VulkanFormat.R8G8Srgb => 16,
                VulkanFormat.R8G8B8Unorm or VulkanFormat.R8G8B8Snorm or VulkanFormat.R8G8B8Uscaled or VulkanFormat.R8G8B8Sscaled or
                VulkanFormat.R8G8B8Uint or VulkanFormat.R8G8B8Sint or VulkanFormat.R8G8B8Srgb or
                VulkanFormat.B8G8R8Unorm or VulkanFormat.B8G8R8Snorm or VulkanFormat.B8G8R8Uscaled or VulkanFormat.B8G8R8Sscaled or
                VulkanFormat.B8G8R8Uint or VulkanFormat.B8G8R8Sint or VulkanFormat.B8G8R8Srgb => 24,
                VulkanFormat.R8G8B8A8Unorm or VulkanFormat.R8G8B8A8Snorm or VulkanFormat.R8G8B8A8Uscaled or VulkanFormat.R8G8B8A8Sscaled or
                VulkanFormat.R8G8B8A8Uint or VulkanFormat.R8G8B8A8Sint or VulkanFormat.R8G8B8A8Srgb or
                VulkanFormat.B8G8R8A8Unorm or VulkanFormat.B8G8R8A8Snorm or VulkanFormat.B8G8R8A8Uscaled or VulkanFormat.B8G8R8A8Sscaled or
                VulkanFormat.B8G8R8A8Uint or VulkanFormat.B8G8R8A8Sint or VulkanFormat.B8G8R8A8Srgb or
                VulkanFormat.A8B8G8R8UnormPack32 or VulkanFormat.A8B8G8R8SnormPack32 or VulkanFormat.A8B8G8R8UintPack32 or
                VulkanFormat.A8B8G8R8SintPack32 or VulkanFormat.A8B8G8R8SrgbPack32 or
                VulkanFormat.A2R10G10B10UnormPack32 or VulkanFormat.A2R10G10B10SnormPack32 or VulkanFormat.A2R10G10B10UintPack32 or
                VulkanFormat.A2B10G10R10UnormPack32 or VulkanFormat.A2B10G10R10SnormPack32 or VulkanFormat.A2B10G10R10UintPack32 or
                VulkanFormat.R16Unorm or VulkanFormat.R16Snorm or VulkanFormat.R16Uscaled or VulkanFormat.R16Sscaled or
                VulkanFormat.R16Uint or VulkanFormat.R16Sint or VulkanFormat.R16Sfloat or
                VulkanFormat.D16Unorm or VulkanFormat.B10G11R11UfloatPack32 or VulkanFormat.E5B9G9R9UfloatPack32 => 32,
                VulkanFormat.R16G16Unorm or VulkanFormat.R16G16Snorm or VulkanFormat.R16G16Uscaled or VulkanFormat.R16G16Sscaled or
                VulkanFormat.R16G16Uint or VulkanFormat.R16G16Sint or VulkanFormat.R16G16Sfloat or
                VulkanFormat.D32Sfloat or VulkanFormat.X8D24UnormPack32 => 32,
                VulkanFormat.R16G16B16Unorm or VulkanFormat.R16G16B16Snorm or VulkanFormat.R16G16B16Uint or
                VulkanFormat.R16G16B16Sint or VulkanFormat.R16G16B16Sfloat => 48,
                VulkanFormat.R16G16B16A16Unorm or VulkanFormat.R16G16B16A16Snorm or VulkanFormat.R16G16B16A16Uint or
                VulkanFormat.R16G16B16A16Sint or VulkanFormat.R16G16B16A16Sfloat or VulkanFormat.D32SfloatS8Uint or
                VulkanFormat.D24UnormS8Uint => 64,
                VulkanFormat.R32Uint or VulkanFormat.R32Sint or VulkanFormat.R32Sfloat => 32,
                VulkanFormat.R32G32Uint or VulkanFormat.R32G32Sint or VulkanFormat.R32G32Sfloat => 64,
                VulkanFormat.R32G32B32Uint or VulkanFormat.R32G32B32Sint or VulkanFormat.R32G32B32Sfloat => 96,
                VulkanFormat.R32G32B32A32Uint or VulkanFormat.R32G32B32A32Sint or VulkanFormat.R32G32B32A32Sfloat => 128,
                VulkanFormat.R64Uint or VulkanFormat.R64Sint or VulkanFormat.R64Sfloat => 64,
                VulkanFormat.R64G64Uint or VulkanFormat.R64G64Sint or VulkanFormat.R64G64Sfloat => 128,
                VulkanFormat.R64G64B64Uint or VulkanFormat.R64G64B64Sint or VulkanFormat.R64G64B64Sfloat => 192,
                VulkanFormat.R64G64B64A64Uint or VulkanFormat.R64G64B64A64Sint or VulkanFormat.R64G64B64A64Sfloat => 256,
                VulkanFormat.S8Uint => 8,
                VulkanFormat.D16UnormS8Uint => 24,
                _ => 32
            };
        }

        /// <summary>Checks if a format is a compressed format</summary>
        public static bool IsCompressedFormat(VulkanFormat format)
        {
            return (format >= VulkanFormat.BC1RgbUnormBlock && format <= VulkanFormat.BC7SrgbBlock) ||
                   (format >= VulkanFormat.Etc2R8G8B8UnormBlock && format <= VulkanFormat.Astc12x12SrgbBlock) ||
                   (format >= VulkanFormat.EacR11UnormBlock && format <= VulkanFormat.EacR11G11SnormBlock);
        }

        private static ulong CalculateCompressedSize(VulkanFormat format, uint width, uint height, uint depth)
        {
            uint blockSizeX = 4, blockSizeY = 4;
            if (format >= VulkanFormat.Astc4x4UnormBlock && format <= VulkanFormat.Astc4x4SrgbBlock)
            { blockSizeX = 4; blockSizeY = 4; }
            else if (format >= VulkanFormat.Astc5x4UnormBlock && format <= VulkanFormat.Astc5x5SrgbBlock)
            { blockSizeX = 5; blockSizeY = 5; }
            else if (format >= VulkanFormat.BC1RgbUnormBlock && format <= VulkanFormat.BC4SnormBlock)
            { blockSizeX = 4; blockSizeY = 4; }
            else if (format >= VulkanFormat.BC5UnormBlock && format <= VulkanFormat.BC7SrgbBlock)
            { blockSizeX = 4; blockSizeY = 4; }

            uint blocksX = (width + blockSizeX - 1) / blockSizeX;
            uint blocksY = (height + blockSizeY - 1) / blockSizeY;
            uint bytesPerBlock = GetCompressedBlockSize(format);
            return (ulong)blocksX * blocksY * depth * bytesPerBlock;
        }

        private static uint GetCompressedBlockSize(VulkanFormat format)
        {
            if (format >= VulkanFormat.BC1RgbUnormBlock && format <= VulkanFormat.BC1RgbaSrgbBlock)
                return 8;
            if (format >= VulkanFormat.BC2UnormBlock && format <= VulkanFormat.BC3SrgbBlock)
                return 16;
            if (format >= VulkanFormat.BC4UnormBlock && format <= VulkanFormat.BC4SnormBlock)
                return 8;
            if (format >= VulkanFormat.BC5UnormBlock && format <= VulkanFormat.BC5SnormBlock)
                return 16;
            if (format >= VulkanFormat.BC6HUfloatBlock && format <= VulkanFormat.BC6HSfloatBlock)
                return 16;
            if (format >= VulkanFormat.BC7UnormBlock && format <= VulkanFormat.BC7SrgbBlock)
                return 16;
            if (format >= VulkanFormat.Etc2R8G8B8UnormBlock && format <= VulkanFormat.Etc2R8G8B8A1SrgbBlock)
                return 8;
            if (format >= VulkanFormat.EacR11UnormBlock && format <= VulkanFormat.EacR11G11SnormBlock)
                return 8;
            if (format >= VulkanFormat.Astc4x4UnormBlock && format <= VulkanFormat.Astc4x4SrgbBlock)
                return 16;
            return 16;
        }

        /// <summary>Converts a VulkanFormat to bytes per pixel for non-compressed formats</summary>
        public static uint GetBytesPerPixel(VulkanFormat format)
        {
            return (GetBitsPerPixel(format) + 7) / 8;
        }

        /// <summary>Checks if the format has a depth component</summary>
        public static bool IsDepthFormat(VulkanFormat format)
        {
            return format is VulkanFormat.D16Unorm or VulkanFormat.X8D24UnormPack32 or
                VulkanFormat.D32Sfloat or VulkanFormat.D16UnormS8Uint or
                VulkanFormat.D24UnormS8Uint or VulkanFormat.D32SfloatS8Uint;
        }

        /// <summary>Checks if the format has a stencil component</summary>
        public static bool IsStencilFormat(VulkanFormat format)
        {
            return format is VulkanFormat.S8Uint or VulkanFormat.D16UnormS8Uint or
                VulkanFormat.D24UnormS8Uint or VulkanFormat.D32SfloatS8Uint;
        }

        /// <summary>Checks if the format has both depth and stencil</summary>
        public static bool IsDepthStencilFormat(VulkanFormat format)
        {
            return format is VulkanFormat.D16UnormS8Uint or VulkanFormat.D24UnormS8Uint or VulkanFormat.D32SfloatS8Uint;
        }

        /// <summary>Returns the corresponding depth-only format for a depth-stencil format</summary>
        public static VulkanFormat GetDepthOnlyFormat(VulkanFormat format)
        {
            return format switch
            {
                VulkanFormat.D16UnormS8Uint => VulkanFormat.D16Unorm,
                VulkanFormat.D24UnormS8Uint => VulkanFormat.X8D24UnormPack32,
                VulkanFormat.D32SfloatS8Uint => VulkanFormat.D32Sfloat,
                _ => format
            };
        }

        /// <summary>Returns the corresponding stencil-only format for a depth-stencil format</summary>
        public static VulkanFormat GetStencilOnlyFormat(VulkanFormat format)
        {
            return format switch
            {
                VulkanFormat.D16UnormS8Uint => VulkanFormat.S8Uint,
                VulkanFormat.D24UnormS8Uint => VulkanFormat.S8Uint,
                VulkanFormat.D32SfloatS8Uint => VulkanFormat.S8Uint,
                _ => format
            };
        }

        /// <summary>Converts an ImageLayout to a string representation</summary>
        public static string ToDisplayString(this ImageLayout layout)
        {
            return layout switch
            {
                ImageLayout.Undefined => "Undefined",
                ImageLayout.General => "General",
                ImageLayout.ColorAttachmentOptimal => "Color Attachment",
                ImageLayout.DepthStencilAttachmentOptimal => "Depth/Stencil Attachment",
                ImageLayout.DepthStencilReadOnlyOptimal => "Depth/Stencil Read-Only",
                ImageLayout.ShaderReadOnlyOptimal => "Shader Read-Only",
                ImageLayout.TransferSrcOptimal => "Transfer Source",
                ImageLayout.TransferDstOptimal => "Transfer Destination",
                ImageLayout.Preinitialized => "Preinitialized",
                ImageLayout.PresentSrcKHR => "Present Source",
                _ => "Unknown"
            };
        }

        /// <summary>Converts a PipelineStageFlag to a string representation</summary>
        public static string ToDisplayString(this PipelineStageFlag flag)
        {
            if (flag == PipelineStageFlag.TopOfPipe)
                return "Top of Pipe";
            if (flag == PipelineStageFlag.BottomOfPipe)
                return "Bottom of Pipe";
            if (flag == PipelineStageFlag.AllCommands)
                return "All Commands";
            if (flag == PipelineStageFlag.AllGraphics)
                return "All Graphics";

            var parts = new List<string>();
            if ((flag & PipelineStageFlag.VertexInput) != 0)
                parts.Add("Vertex Input");
            if ((flag & PipelineStageFlag.VertexShader) != 0)
                parts.Add("Vertex Shader");
            if ((flag & PipelineStageFlag.FragmentShader) != 0)
                parts.Add("Fragment Shader");
            if ((flag & PipelineStageFlag.GeometryShader) != 0)
                parts.Add("Geometry Shader");
            if ((flag & PipelineStageFlag.ComputeShader) != 0)
                parts.Add("Compute Shader");
            if ((flag & PipelineStageFlag.Transfer) != 0)
                parts.Add("Transfer");
            if ((flag & PipelineStageFlag.ColorAttachmentOutput) != 0)
                parts.Add("Color Attachment Output");
            if ((flag & PipelineStageFlag.EarlyFragmentTests) != 0)
                parts.Add("Early Fragment Tests");
            if ((flag & PipelineStageFlag.LateFragmentTests) != 0)
                parts.Add("Late Fragment Tests");
            if ((flag & PipelineStageFlag.DrawIndirect) != 0)
                parts.Add("Draw Indirect");
            if ((flag & PipelineStageFlag.Host) != 0)
                parts.Add("Host");
            return parts.Count > 0 ? string.Join(" | ", parts) : flag.ToString();
        }

        /// <summary>Creates an identity 4x4 matrix</summary>
        public static float[] CreateIdentityMatrix4x4()
        {
            return new float[]
            {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
            };
        }

        /// <summary>Creates a perspective projection matrix</summary>
        public static float[] CreatePerspectiveMatrix(float fovY, float aspectRatio, float nearPlane, float farPlane)
        {
            float tanHalfFov = (float)Math.Tan(fovY * 0.5f);
            return new float[]
            {
                1.0f / (aspectRatio * tanHalfFov), 0, 0, 0,
                0, -1.0f / tanHalfFov, 0, 0,
                0, 0, farPlane / (nearPlane - farPlane), -1,
                0, 0, (farPlane * nearPlane) / (nearPlane - farPlane), 0
            };
        }

        /// <summary>Creates an orthographic projection matrix</summary>
        public static float[] CreateOrthographicMatrix(float left, float right, float bottom, float top, float nearPlane, float farPlane)
        {
            return new float[]
            {
                2.0f / (right - left), 0, 0, 0,
                0, 2.0f / (top - bottom), 0, 0,
                0, 0, 1.0f / (nearPlane - farPlane), 0,
                -(right + left) / (right - left), -(top + bottom) / (top - bottom), nearPlane / (nearPlane - farPlane), 1
            };
        }

        /// <summary>Creates a look-at view matrix</summary>
        public static float[] CreateLookAtMatrix(float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ)
        {
            float fx = centerX - eyeX, fy = centerY - eyeY, fz = centerZ - eyeZ;
            float len = (float)Math.Sqrt(fx * fx + fy * fy + fz * fz);
            fx /= len;
            fy /= len;
            fz /= len;

            float sx = fy * upZ - fz * upY;
            float sy = fz * upX - fx * upZ;
            float sz = fx * upY - fy * upX;
            len = (float)Math.Sqrt(sx * sx + sy * sy + sz * sz);
            sx /= len;
            sy /= len;
            sz /= len;

            float ux = sy * fz - sz * fy;
            float uy = sz * fx - sx * fz;
            float uz = sx * fy - sy * fx;

            return new float[]
            {
                sx, ux, -fx, 0,
                sy, uy, -fy, 0,
                sz, uz, -fz, 0,
                -(sx * eyeX + sy * eyeY + sz * eyeZ),
                -(ux * eyeX + uy * eyeY + uz * eyeZ),
                (fx * eyeX + fy * eyeY + fz * eyeZ),
                1
            };
        }

        /// <summary>Checks if the format is a sRGB format</summary>
        public static bool IsSRGBFormat(VulkanFormat format)
        {
            return format is VulkanFormat.R8Srgb or VulkanFormat.R8G8Srgb or VulkanFormat.R8G8B8Srgb or
                VulkanFormat.B8G8R8Srgb or VulkanFormat.R8G8B8A8Srgb or VulkanFormat.B8G8R8A8Srgb or
                VulkanFormat.A8B8G8R8SrgbPack32;
        }

        /// <summary>Returns the corresponding linear format for a sRGB format</summary>
        public static VulkanFormat ToLinearFormat(VulkanFormat format)
        {
            return format switch
            {
                VulkanFormat.R8Srgb => VulkanFormat.R8Unorm,
                VulkanFormat.R8G8Srgb => VulkanFormat.R8G8Unorm,
                VulkanFormat.R8G8B8Srgb => VulkanFormat.R8G8B8Unorm,
                VulkanFormat.B8G8R8Srgb => VulkanFormat.B8G8R8Unorm,
                VulkanFormat.R8G8B8A8Srgb => VulkanFormat.R8G8B8A8Unorm,
                VulkanFormat.B8G8R8A8Srgb => VulkanFormat.B8G8R8A8Unorm,
                VulkanFormat.A8B8G8R8SrgbPack32 => VulkanFormat.A8B8G8R8UnormPack32,
                _ => format
            };
        }

        /// <summary>Returns the corresponding sRGB format for a linear format</summary>
        public static VulkanFormat ToSRGBFormat(VulkanFormat format)
        {
            return format switch
            {
                VulkanFormat.R8Unorm => VulkanFormat.R8Srgb,
                VulkanFormat.R8G8Unorm => VulkanFormat.R8G8Srgb,
                VulkanFormat.R8G8B8Unorm => VulkanFormat.R8G8B8Srgb,
                VulkanFormat.B8G8R8Unorm => VulkanFormat.B8G8R8Srgb,
                VulkanFormat.R8G8B8A8Unorm => VulkanFormat.R8G8B8A8Srgb,
                VulkanFormat.B8G8R8A8Unorm => VulkanFormat.B8G8R8A8Srgb,
                VulkanFormat.A8B8G8R8UnormPack32 => VulkanFormat.A8B8G8R8SrgbPack32,
                _ => format
            };
        }

        /// <summary>Validates Vulkan result code and throws on error</summary>
        public static void ThrowOnError(this VulkanResult result, string operation = "")
        {
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Vulkan error {result}" + (string.IsNullOrEmpty(operation) ? "" : $" during {operation}"));
        }

        /// <summary>Converts a ShaderStageFlag to a human-readable name</summary>
        public static string ToDisplayString(this ShaderStageFlag stage)
        {
            return stage switch
            {
                ShaderStageFlag.Vertex => "Vertex",
                ShaderStageFlag.TessellationControl => "Tessellation Control",
                ShaderStageFlag.TessellationEvaluation => "Tessellation Evaluation",
                ShaderStageFlag.Geometry => "Geometry",
                ShaderStageFlag.Fragment => "Fragment",
                ShaderStageFlag.Compute => "Compute",
                ShaderStageFlag.AllGraphics => "All Graphics",
                ShaderStageFlag.All => "All Stages",
                ShaderStageFlag.RayGenerationKHR => "Ray Generation",
                ShaderStageFlag.AnyHitKHR => "Any Hit",
                ShaderStageFlag.ClosestHitKHR => "Closest Hit",
                ShaderStageFlag.MissKHR => "Miss",
                ShaderStageFlag.IntersectionKHR => "Intersection",
                ShaderStageFlag.CallableKHR => "Callable",
                _ => stage.ToString()
            };
        }
    }

    // =========================================================================
    // VkImageBlit struct for blit operations
    // =========================================================================
    [StructLayout(LayoutKind.Sequential)]
    public struct VkImageBlit
    {
        public VkImageSubresourceLayers srcSubresource;
        public VkOffset3D srcOffsets0;
        public VkOffset3D srcOffsets1;
        public VkImageSubresourceLayers dstSubresource;
        public VkOffset3D dstOffsets0;
        public VkOffset3D dstOffsets1;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkInstanceCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public IntPtr pApplicationInfo; public uint enabledLayerCount;
        public IntPtr ppEnabledLayerNames; public uint enabledExtensionCount;
        public IntPtr ppEnabledExtensionNames;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineLayoutCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint setLayoutCount; public IntPtr pSetLayouts;
        public uint pushConstantRangeCount; public IntPtr pPushConstantRanges;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPushConstantRange
    {
        public ShaderStageFlag stageFlags;
        public uint offset;
        public uint size;
    }

    // =========================================================================
    // VULKAN INTEROP TYPES
    // =========================================================================

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkBool32
    {
        private uint _value;
        public static readonly VkBool32 False = new VkBool32 { _value = 0 };
        public static readonly VkBool32 True = new VkBool32 { _value = 1 };
        public static implicit operator bool(VkBool32 v) => v._value != 0;
        public static implicit operator VkBool32(bool b) => b ? True : False;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VulkanBool32
    {
        private uint _value;
        public static readonly VulkanBool32 False = new VulkanBool32 { _value = 0 };
        public static readonly VulkanBool32 True = new VulkanBool32 { _value = 1 };
        public static implicit operator bool(VulkanBool32 v) => v._value != 0;
        public static implicit operator VulkanBool32(bool b) => b ? True : False;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkExtent2D { public uint width; public uint height; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkExtent3D { public uint width; public uint height; public uint depth; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkOffset3D { public int x; public int y; public int z; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryRequirements
    {
        public ulong size;
        public ulong alignment;
        public uint memoryTypeBits;

        public ulong Size => size;
        public ulong Alignment => alignment;
        public uint MemoryTypeBits => memoryTypeBits;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMappedMemoryRange { public uint sType; public IntPtr pNext; public IntPtr memory; public ulong offset; public ulong size; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkImageSubresourceLayers { public uint aspectMask; public uint mipLevel; public uint baseArrayLayer; public uint layerCount; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkImageSubresourceRange
    {
        public ImageAspectFlag aspectMask; public uint baseMipLevel; public uint levelCount; public uint baseArrayLayer; public uint layerCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkComponentMapping { public uint r; public uint g; public uint b; public uint a; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkMemoryAllocateInfo { public uint sType; public IntPtr pNext; public ulong allocationSize; public uint memoryTypeIndex; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkFenceCreateInfo { public uint sType; public IntPtr pNext; public uint flags; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSemaphoreCreateInfo { public uint sType; public IntPtr pNext; public uint flags; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDeviceCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint queueCreateInfoCount; public IntPtr pQueueCreateInfos;
        public uint enabledLayerCount; public IntPtr ppEnabledLayerNames;
        public uint enabledExtensionCount; public IntPtr ppEnabledExtensionNames;
        public IntPtr pEnabledFeatures;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkBufferCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public ulong size; public BufferUsageFlag usage; public SharingMode sharingMode;
        public uint queueFamilyIndexCount; public IntPtr pQueueFamilyIndices;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkImageCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public ImageType imageType; public VulkanFormat format; public VkExtent3D extent;
        public uint mipLevels; public uint arrayLayers; public SampleCountFlag samples;
        public ImageTiling tiling; public ImageUsageFlag usage; public SharingMode sharingMode;
        public uint queueFamilyIndexCount; public IntPtr pQueueFamilyIndices; public ImageLayout initialLayout;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkImageViewCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public IntPtr image; public ImageViewType viewType; public VulkanFormat format;
        public VkComponentMapping components; public VkImageSubresourceRange subresourceRange;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPresentInfoKHR
    {
        public uint sType; public IntPtr pNext;
        public uint waitSemaphoreCount; public IntPtr pWaitSemaphores;
        public uint swapchainCount; public IntPtr pSwapchains;
        public IntPtr pImageIndices; public IntPtr pResults;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkRenderPassCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint attachmentCount; public IntPtr pAttachments;
        public uint subpassCount; public IntPtr pSubpasses;
        public uint dependencyCount; public IntPtr pDependencies;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSubpassDescription
    {
        public uint flags; public PipelineBindPoint pipelineBindPoint;
        public uint inputAttachmentCount; public IntPtr pInputAttachments;
        public uint colorAttachmentCount; public IntPtr pColorAttachments;
        public IntPtr pResolveAttachments; public IntPtr pDepthStencilAttachment;
        public uint preserveAttachmentCount; public IntPtr pPreserveAttachments;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSubpassDependency
    {
        public uint srcSubpass; public uint dstSubpass;
        public PipelineStageFlag srcStageMask; public PipelineStageFlag dstStageMask;
        public AccessFlag srcAccessMask; public AccessFlag dstAccessMask;
        public uint dependencyFlags;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkFramebufferCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public IntPtr renderPass; public uint attachmentCount; public IntPtr pAttachments;
        public uint width; public uint height; public uint layers;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSurfaceCapabilitiesKHR
    {
        public uint minImageCount; public uint maxImageCount;
        public uint currentImageWidth; public uint currentImageHeight;
        public uint minImageWidth; public uint minImageHeight;
        public uint maxImageWidth; public uint maxImageHeight;
        public uint maxImageArrayLayers;
        public SurfaceTransformFlag supportedTransforms;
        public SurfaceTransformFlag currentTransform;
        public CompositeAlphaFlag supportedCompositeAlpha;
        public ImageUsageFlag supportedUsageFlags;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSwapchainCreateInfoKHR
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public IntPtr surface; public uint minImageCount;
        public VulkanFormat imageFormat; public uint imageColorSpace;
        public VkExtent2D imageExtent; public uint imageArrayLayers;
        public ImageUsageFlag imageUsage; public SharingMode imageSharingMode;
        public uint queueFamilyIndexCount; public IntPtr pQueueFamilyIndices;
        public SurfaceTransformFlag preTransform; public CompositeAlphaFlag compositeAlpha;
        public PresentMode presentMode; public VulkanBool32 clipped; public IntPtr oldSwapchain;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSubmitInfo
    {
        public uint sType; public IntPtr pNext;
        public uint waitSemaphoreCount; public IntPtr pWaitSemaphores;
        public IntPtr pWaitDstStageMask;
        public uint commandBufferCount; public IntPtr pCommandBuffers;
        public uint signalSemaphoreCount; public IntPtr pSignalSemaphores;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkCommandPoolCreateInfo { public uint sType; public IntPtr pNext; public uint flags; public uint queueFamilyIndex; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkCommandBufferAllocateInfo { public uint sType; public IntPtr pNext; public IntPtr commandPool; public uint level; public uint commandBufferCount; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkCommandBufferBeginInfo
    {
        public uint sType;
        public IntPtr pNext;
        public uint flags;
        public IntPtr pInheritanceInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkRenderPassBeginInfo_Rect2D
    {
        public int x, y;
        public uint width, height;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkRenderPassBeginInfo
    {
        public uint sType;
        public IntPtr pNext;
        public IntPtr renderPass;
        public IntPtr framebuffer;
        public VkRenderPassBeginInfo_Rect2D renderArea;
        public uint clearValueCount;
        public IntPtr pClearValues;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkShaderModuleCreateInfo { public uint sType; public IntPtr pNext; public uint flags; public IntPtr codeSize; public IntPtr pCode; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSamplerCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public Filter magFilter; public Filter minFilter; public SamplerAddressMode mipmapMode;
        public SamplerAddressMode addressModeU; public SamplerAddressMode addressModeV; public SamplerAddressMode addressModeW;
        public float mipLodBias; public VulkanBool32 anisotropyEnable; public float maxAnisotropy;
        public VulkanBool32 compareEnable; public CompareOp compareOp;
        public float minLod; public float maxLod; public uint borderColor; public VulkanBool32 unnormalizedCoordinates;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorSetLayoutBinding
    {
        public uint binding; public DescriptorType descriptorType; public uint descriptorCount;
        public ShaderStageFlag stageFlags; public IntPtr pImmutableSamplers;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorPoolSize { public DescriptorType type; public uint descriptorCount; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorPoolCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint maxSets; public uint poolSizeCount; public IntPtr pPoolSizes;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorSetAllocateInfo
    {
        public uint sType; public IntPtr pNext;
        public IntPtr descriptorPool; public uint descriptorSetCount; public IntPtr pSetLayouts;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorImageInfo { public IntPtr sampler; public IntPtr imageView; public ImageLayout imageLayout; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorBufferInfo { public IntPtr buffer; public ulong offset; public ulong range; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkWriteDescriptorSet
    {
        public uint sType; public IntPtr pNext;
        public IntPtr dstSet; public uint dstBinding; public uint dstArrayElement;
        public uint descriptorCount; public DescriptorType descriptorType;
        public IntPtr pImageInfo; public IntPtr pBufferInfo; public IntPtr pTexelBufferView;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkImageMemoryBarrier
    {
        public uint sType; public IntPtr pNext;
        public AccessFlag srcAccessMask; public AccessFlag dstAccessMask;
        public ImageLayout oldLayout; public ImageLayout newLayout;
        public uint srcQueueFamilyIndex; public uint dstQueueFamilyIndex;
        public IntPtr image; public VkImageSubresourceRange subresourceRange;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineVertexInputStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint vertexBindingDescriptionCount; public IntPtr pVertexBindingDescriptions;
        public uint vertexAttributeDescriptionCount; public IntPtr pVertexAttributeDescriptions;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineInputAssemblyStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public PrimitiveTopology topology; public VulkanBool32 primitiveRestartEnable;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineTessellationStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint patchControlPoints;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineViewportStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint viewportCount; public IntPtr pViewports;
        public uint scissorCount; public IntPtr pScissors;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineRasterizationStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public VulkanBool32 depthClampEnable; public VulkanBool32 rasterizerDiscardEnable;
        public PolygonMode polygonMode; public CullModeFlag cullMode; public FrontFace frontFace;
        public VulkanBool32 depthBiasEnable; public float depthBiasConstantFactor;
        public float depthBiasClamp; public float depthBiasSlopeFactor; public float lineWidth;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineMultisampleStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public SampleCountFlag rasterizationSamples; public VulkanBool32 sampleShadingEnable;
        public float minSampleShading; public IntPtr pSampleMask;
        public VulkanBool32 alphaToCoverageEnable; public VulkanBool32 alphaToOneEnable;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkStencilOpState
    {
        public uint failOp; public uint passOp; public uint depthFailOp; public uint compareOp;
        public uint compareMask; public uint writeMask; public uint reference;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineDepthStencilStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public VulkanBool32 depthTestEnable; public VulkanBool32 depthWriteEnable; public CompareOp depthCompareOp;
        public VulkanBool32 depthBoundsTestEnable; public VulkanBool32 stencilTestEnable;
        public VkStencilOpState front; public VkStencilOpState back;
        public float minDepthBounds; public float maxDepthBounds;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineColorBlendStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public VulkanBool32 logicOpEnable; public LogicOp logicOp;
        public uint attachmentCount; public IntPtr pAttachments;
        public float blendConstant0; public float blendConstant1; public float blendConstant2; public float blendConstant3;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineDynamicStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint dynamicStateCount; public IntPtr pDynamicStates;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkGraphicsPipelineCreateInfo
    {
        public uint sType; public IntPtr pNext; public PipelineCreateFlag flags;
        public uint stageCount; public IntPtr pStages;
        public IntPtr pVertexInputState; public IntPtr pInputAssemblyState;
        public IntPtr pTessellationState; public IntPtr pViewportState;
        public IntPtr pRasterizationState; public IntPtr pMultisampleState;
        public IntPtr pDepthStencilState; public IntPtr pColorBlendState;
        public IntPtr pDynamicState; public IntPtr layout;
        public IntPtr renderPass; public uint subpass;
        public IntPtr basePipelineHandle; public int basePipelineIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkComputePipelineCreateInfo
    {
        public uint sType; public IntPtr pNext; public PipelineCreateFlag flags;
        public uint stageCount; public IntPtr pStage;
        public IntPtr layout;
        public IntPtr basePipelineHandle; public int basePipelineIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineCacheCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public IntPtr initialDataSize; public IntPtr pInitialData;
    }

    /// <summary>Vulkan pipeline layout wrapper</summary>
}
