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
    public class VulkanSyncManager : IDisposable
    {
        private VulkanDevice _device;
        private readonly object _lock = new object();
        private bool _disposed;

        // Fence pool
        private readonly ConcurrentBag<IntPtr> _fencePool = new ConcurrentBag<IntPtr>();
        private readonly List<IntPtr> _allFences = new List<IntPtr>();
        private int _activeFenceCount;

        // Semaphore pool
        private readonly ConcurrentBag<IntPtr> _semaphorePool = new ConcurrentBag<IntPtr>();
        private readonly List<IntPtr> _allSemaphores = new List<IntPtr>();
        private int _activeSemaphoreCount;

        // Timeline semaphores
        private readonly List<IntPtr> _timelineSemaphores = new List<IntPtr>();
        private readonly Dictionary<IntPtr, ulong> _timelineValues = new Dictionary<IntPtr, ulong>();

        // Function pointers
        private CreateFenceDel _vkCreateFence;
        private DestroyFenceDel _vkDestroyFence;
        private WaitForFencesDel _vkWaitForFences;
        private ResetFencesDel _vkResetFences;
        private CreateSemaphoreDel _vkCreateSemaphore;
        private DestroySemaphoreDel _vkDestroySemaphore;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateFenceDel(IntPtr device, ref VkFenceCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pFence);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyFenceDel(IntPtr device, IntPtr fence, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult WaitForFencesDel(IntPtr device, uint fenceCount, ref IntPtr pFences, uint waitAll, ulong timeout);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult ResetFencesDel(IntPtr device, uint fenceCount, ref IntPtr pFences);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateSemaphoreDel(IntPtr device, ref VkSemaphoreCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pSemaphore);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroySemaphoreDel(IntPtr device, IntPtr semaphore, IntPtr pAllocator);

        public int ActiveFences => _activeFenceCount;
        public int ActiveSemaphores => _activeSemaphoreCount;

        public VulkanSyncManager(VulkanDevice device)
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

            _vkCreateFence = Marshal.GetDelegateForFunctionPointer<CreateFenceDel>(load("vkCreateFence"));
            _vkDestroyFence = Marshal.GetDelegateForFunctionPointer<DestroyFenceDel>(load("vkDestroyFence"));
            _vkWaitForFences = Marshal.GetDelegateForFunctionPointer<WaitForFencesDel>(load("vkWaitForFences"));
            _vkResetFences = Marshal.GetDelegateForFunctionPointer<ResetFencesDel>(load("vkResetFences"));
            _vkCreateSemaphore = Marshal.GetDelegateForFunctionPointer<CreateSemaphoreDel>(load("vkCreateSemaphore"));
            _vkDestroySemaphore = Marshal.GetDelegateForFunctionPointer<DestroySemaphoreDel>(load("vkDestroySemaphore"));
        }

        /// <summary>Acquires a fence from the pool or creates a new one</summary>
        public IntPtr CreateFence(bool signaled = false)
        {
            lock (_lock)
            {
                if (!signaled && _fencePool.TryTake(out var fence))
                {
                    Interlocked.Increment(ref _activeFenceCount);
                    return fence;
                }
            }

            var createInfo = new VkFenceCreateInfo
            {
                sType = 8,
                flags = signaled ? 1u : 0u
            };

            IntPtr newFence = IntPtr.Zero;
            _vkCreateFence(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref newFence);

            lock (_lock)
            {
                _allFences.Add(newFence);
                Interlocked.Increment(ref _activeFenceCount);
            }

            return newFence;
        }

        /// <summary>Returns a fence to the pool for reuse</summary>
        public void RecycleFence(IntPtr fence)
        {
            if (fence == IntPtr.Zero)
                return;

            lock (_lock)
            {
                Interlocked.Decrement(ref _activeFenceCount);
                _fencePool.Add(fence);
            }
        }

        /// <summary>Waits for a single fence with timeout</summary>
        public VulkanResult WaitForFence(IntPtr fence, ulong timeout = 0xFFFFFFFFFFFFFFFF)
        {
            if (fence == IntPtr.Zero)
                return VulkanResult.ErrorUnknown;
            return _vkWaitForFences(_device.LogicalDevice, 1, ref fence, 1, timeout);
        }

        /// <summary>Waits for multiple fences</summary>
        public VulkanResult WaitForFences(IntPtr[] fences, bool waitAll = true, ulong timeout = 0xFFFFFFFFFFFFFFFF)
        {
            if (fences == null || fences.Length == 0)
                return VulkanResult.Success;
            uint count = (uint)fences.Length;
            var fenceArray = Marshal.AllocHGlobal(fences.Length * IntPtr.Size);
            try
            {
                for (int i = 0; i < fences.Length; i++)
                    Marshal.WriteIntPtr(fenceArray + i * IntPtr.Size, fences[i]);
                return _vkWaitForFences(_device.LogicalDevice, count, ref fenceArray, waitAll ? 1u : 0u, timeout);
            }
            finally { Marshal.FreeHGlobal(fenceArray); }
        }

        /// <summary>Resets one or more fences</summary>
        public VulkanResult ResetFence(IntPtr fence)
        {
            if (fence == IntPtr.Zero)
                return VulkanResult.ErrorUnknown;
            return _vkResetFences(_device.LogicalDevice, 1, ref fence);
        }

        public VulkanResult ResetFences(IntPtr[] fences)
        {
            if (fences == null || fences.Length == 0)
                return VulkanResult.Success;
            uint count = (uint)fences.Length;
            var fenceArray = Marshal.AllocHGlobal(fences.Length * IntPtr.Size);
            try
            {
                for (int i = 0; i < fences.Length; i++)
                    Marshal.WriteIntPtr(fenceArray + i * IntPtr.Size, fences[i]);
                return _vkResetFences(_device.LogicalDevice, count, ref fenceArray);
            }
            finally { Marshal.FreeHGlobal(fenceArray); }
        }

        /// <summary>Acquires a semaphore from the pool or creates a new one</summary>
        public IntPtr CreateSemaphore()
        {
            lock (_lock)
            {
                if (_semaphorePool.TryTake(out var semaphore))
                {
                    Interlocked.Increment(ref _activeSemaphoreCount);
                    return semaphore;
                }
            }

            var createInfo = new VkSemaphoreCreateInfo { sType = 7 };
            IntPtr newSemaphore = IntPtr.Zero;
            _vkCreateSemaphore(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref newSemaphore);

            lock (_lock)
            {
                _allSemaphores.Add(newSemaphore);
                Interlocked.Increment(ref _activeSemaphoreCount);
            }

            return newSemaphore;
        }

        /// <summary>Returns a semaphore to the pool</summary>
        public void RecycleSemaphore(IntPtr semaphore)
        {
            if (semaphore == IntPtr.Zero)
                return;
            lock (_lock)
            {
                Interlocked.Decrement(ref _activeSemaphoreCount);
                _semaphorePool.Add(semaphore);
            }
        }

        /// <summary>Creates a timeline semaphore</summary>
        public IntPtr CreateTimelineSemaphore(ulong initialValue = 0)
        {
            var timelineFeatures = new VkTimelineSemaphoreCreateInfo
            {
                sType = 1000076002,
                semaphoreType = 1,
                initialValue = initialValue
            };

            IntPtr semaphore = IntPtr.Zero;
            ref VkSemaphoreCreateInfo createInfo = ref Unsafe.As<VkTimelineSemaphoreCreateInfo, VkSemaphoreCreateInfo>(ref timelineFeatures);
            _vkCreateSemaphore(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref semaphore);

            lock (_lock)
            {
                _timelineSemaphores.Add(semaphore);
                _timelineValues[semaphore] = initialValue;
            }

            return semaphore;
        }

        /// <summary>Gets the current timeline value for a semaphore</summary>
        public ulong GetTimelineValue(IntPtr semaphore)
        {
            lock (_lock)
            {
                return _timelineValues.ContainsKey(semaphore) ? _timelineValues[semaphore] : 0;
            }
        }

        /// <summary>Recycles all unused synchronization objects</summary>
        public void RecycleAll()
        {
            lock (_lock)
            {
                while (_fencePool.TryTake(out _))
                { }
            }
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;

            lock (_lock)
            {
                foreach (var fence in _allFences)
                    _vkDestroyFence?.Invoke(_device.LogicalDevice, fence, IntPtr.Zero);
                foreach (var semaphore in _allSemaphores)
                    _vkDestroySemaphore?.Invoke(_device.LogicalDevice, semaphore, IntPtr.Zero);
                foreach (var timeline in _timelineSemaphores)
                    _vkDestroySemaphore?.Invoke(_device.LogicalDevice, timeline, IntPtr.Zero);

                _allFences.Clear();
                _allSemaphores.Clear();
                _timelineSemaphores.Clear();
            }
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Timeline semaphore create info extension</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct VkTimelineSemaphoreCreateInfo
    {
        public uint sType;
        public IntPtr pNext;
        public uint flags;
        public uint semaphoreType;
        public ulong initialValue;
    }

    /// <summary>Vulkan fence wrapper</summary>
    public class Fence : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _fence;
        private VulkanSyncManager _syncManager;
        private bool _disposed;

        public IntPtr Handle => _fence;

        public Fence(VulkanDevice device, IntPtr fence, VulkanSyncManager syncManager)
        {
            _device = device;
            _fence = fence;
            _syncManager = syncManager;
        }

        public VulkanResult Wait(ulong timeout = 0xFFFFFFFFFFFFFFFF) => _syncManager.WaitForFence(_fence, timeout);
        public VulkanResult Reset() => _syncManager.ResetFence(_fence);

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _syncManager?.RecycleFence(_fence);
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Vulkan semaphore wrapper</summary>
    public class Semaphore : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _semaphore;
        private VulkanSyncManager _syncManager;
        private bool _disposed;

        public IntPtr Handle => _semaphore;

        public Semaphore(VulkanDevice device, IntPtr semaphore, VulkanSyncManager syncManager)
        {
            _device = device;
            _semaphore = semaphore;
            _syncManager = syncManager;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _syncManager?.RecycleSemaphore(_semaphore);
            GC.SuppressFinalize(this);
        }
    }
    // =========================================================================
    // VulkanResourceTracker
    // =========================================================================

    /// <summary>
    /// Tracks resource lifetime with reference counting, deferred deletion,
    /// and resource state tracking for safe GPU resource management.
    /// </summary>
}
