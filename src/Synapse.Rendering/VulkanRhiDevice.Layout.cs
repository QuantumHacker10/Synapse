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
    public class VulkanPipelineLayout : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _layout;
        private bool _disposed;

        public IntPtr Handle => _layout;

        public VulkanPipelineLayout(VulkanDevice device, IntPtr layout)
        {
            _device = device;
            _layout = layout;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_layout != IntPtr.Zero && _device?.LogicalDevice != IntPtr.Zero)
            {
                var pfn = _device.GetDeviceProcAddr("vkDestroyPipelineLayout");
                if (pfn != IntPtr.Zero)
                {
                    var del = Marshal.GetDelegateForFunctionPointer<DestroyPipelineLayoutDel>(pfn);
                    del(_device.LogicalDevice, _layout, IntPtr.Zero);
                }
            }
            GC.SuppressFinalize(this);
        }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate void DestroyPipelineLayoutDel(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator);
    }
}
