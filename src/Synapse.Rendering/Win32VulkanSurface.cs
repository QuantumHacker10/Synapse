using System;
using System.Runtime.InteropServices;

namespace GDNN.Platform
{
    /// <summary>
    /// Creates a Vulkan Win32 surface from an HWND for Avalonia NativeControlHost embedding.
    /// </summary>
    public static class Win32VulkanSurface
    {
        private const string VulkanDll = "vulkan-1.dll";

        [StructLayout(LayoutKind.Sequential)]
        private struct VkWin32SurfaceCreateInfoKHR
        {
            public int sType; // VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR = 1000009000
            public IntPtr pNext;
            public uint flags;
            public IntPtr hinstance;
            public IntPtr hwnd;
        }

        private delegate int VkCreateWin32SurfaceKHR(
            IntPtr instance,
            ref VkWin32SurfaceCreateInfoKHR createInfo,
            IntPtr allocator,
            out IntPtr surface);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr GetModuleHandle(string? lpModuleName);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern IntPtr CreateWindowEx(
            uint dwExStyle,
            string lpClassName,
            string lpWindowName,
            uint dwStyle,
            int x,
            int y,
            int nWidth,
            int nHeight,
            IntPtr hWndParent,
            IntPtr hMenu,
            IntPtr hInstance,
            IntPtr lpParam);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool DestroyWindow(IntPtr hWnd);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern IntPtr SetParent(IntPtr hWndChild, IntPtr hWndNewParent);

        [DllImport("user32.dll")]
        public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        [DllImport(VulkanDll, EntryPoint = "vkGetInstanceProcAddr")]
        private static extern IntPtr vkGetInstanceProcAddr(IntPtr instance, IntPtr name);

        public const uint WS_CHILD = 0x40000000;
        public const uint WS_VISIBLE = 0x10000000;
        public const uint WS_CLIPSIBLINGS = 0x04000000;
        public const int SW_SHOW = 5;
        public const uint SWP_NOZORDER = 0x0004;
        public const uint SWP_NOACTIVATE = 0x0010;

        public static IntPtr CreateChildWindow(IntPtr parentHwnd, int width, int height)
        {
            var hInstance = GetModuleHandle(null);
            var hwnd = CreateWindowEx(
                0,
                "STATIC",
                "SynapseViewport",
                WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS,
                0, 0, width, height,
                parentHwnd,
                IntPtr.Zero,
                hInstance,
                IntPtr.Zero);

            if (hwnd == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Win32 child viewport window.");

            ShowWindow(hwnd, SW_SHOW);
            return hwnd;
        }

        public static void ResizeChild(IntPtr hwnd, int width, int height)
        {
            if (hwnd == IntPtr.Zero)
                return;
            SetWindowPos(hwnd, IntPtr.Zero, 0, 0, Math.Max(1, width), Math.Max(1, height), SWP_NOZORDER | SWP_NOACTIVATE);
        }

        public static unsafe IntPtr CreateSurface(IntPtr vulkanInstance, IntPtr hwnd)
        {
            var namePtr = Marshal.StringToHGlobalAnsi("vkCreateWin32SurfaceKHR");
            try
            {
                var proc = vkGetInstanceProcAddr(vulkanInstance, namePtr);
                if (proc == IntPtr.Zero)
                    throw new InvalidOperationException("vkCreateWin32SurfaceKHR not available.");

                var create = Marshal.GetDelegateForFunctionPointer<VkCreateWin32SurfaceKHR>(proc);
                var info = new VkWin32SurfaceCreateInfoKHR
                {
                    sType = 1000009000,
                    pNext = IntPtr.Zero,
                    flags = 0,
                    hinstance = GetModuleHandle(null),
                    hwnd = hwnd
                };

                var result = create(vulkanInstance, ref info, IntPtr.Zero, out var surface);
                if (result != 0 || surface == IntPtr.Zero)
                    throw new InvalidOperationException($"vkCreateWin32SurfaceKHR failed: {result}");

                return surface;
            }
            finally
            {
                Marshal.FreeHGlobal(namePtr);
            }
        }
    }
}
