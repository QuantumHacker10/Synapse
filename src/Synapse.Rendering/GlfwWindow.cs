// =============================================================================
// GlfwWindow.cs - GDNN Engine: Cross-platform GLFW via P/Invoke
// Windows (glfw3.dll), Linux (libglfw.so.3), macOS (libglfw.3.dylib)
// =============================================================================

using System;
using System.Runtime.InteropServices;

namespace GDNN.Platform
{
    public static class GlfwWindow
    {
        // Resolved by NativeLibraryResolver to the correct OS library name.
        private const string GLFW_DLL = "glfw3";

        static GlfwWindow()
        {
            NativeLibraryResolver.EnsureRegistered();
        }

        public delegate void GLFWerrorfun(int errorcode, IntPtr description);
        public delegate void GLFWwindowposfun(IntPtr window, int xpos, int ypos);
        public delegate void GLFWframesizefun(IntPtr window, int width, int height);
        public delegate void GLFWkeyfun(IntPtr window, int key, int scancode, int action, int mods);
        public delegate void GLFWmousebuttonfun(IntPtr window, int button, int action, int mods);
        public delegate void GLFWscrollfun(IntPtr window, double xoffset, double yoffset);

        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern int glfwInit();
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwTerminate();
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwGetVersion(ref int major, ref int minor, ref int rev);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr glfwGetError(out IntPtr description);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwSetErrorCallback(GLFWerrorfun callback);

        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr glfwCreateWindow(int width, int height, IntPtr title, IntPtr monitor, IntPtr share);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwDestroyWindow(IntPtr window);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern int glfwWindowShouldClose(IntPtr window);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwSetWindowShouldClose(IntPtr window, int value);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwSetWindowTitle(IntPtr window, IntPtr title);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwGetWindowSize(IntPtr window, out int width, out int height);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwSetWindowSizeCallback(IntPtr window, GLFWframesizefun callback);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwSetFramebufferSizeCallback(IntPtr window, GLFWframesizefun callback);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwGetFramebufferSize(IntPtr window, out int width, out int height);

        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern int glfwGetKey(IntPtr window, int key);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern int glfwGetMouseButton(IntPtr window, int button);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwGetCursorPos(IntPtr window, out double xpos, out double ypos);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwSetKeyCallback(IntPtr window, GLFWkeyfun callback);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwSetMouseButtonCallback(IntPtr window, GLFWmousebuttonfun callback);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwSetScrollCallback(IntPtr window, GLFWscrollfun callback);

        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwPollEvents();
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwWaitEvents();
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern double glfwGetTime();
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwSwapBuffers(IntPtr window);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern int glfwVulkanSupported();

        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr glfwGetRequiredInstanceExtensions(ref uint count);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern int glfwCreateWindowSurface(IntPtr instance, IntPtr window, IntPtr allocator, ref IntPtr surface);

        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] public static extern void glfwSetInputMode(IntPtr window, int mode, int value);
        [DllImport(GLFW_DLL, CallingConvention = CallingConvention.Cdecl)] private static extern void glfwWindowHint(int hint, int value);

        public const int GLFW_KEY_ESCAPE = 256;
        public const int GLFW_KEY_W = 87;
        public const int GLFW_KEY_A = 65;
        public const int GLFW_KEY_S = 83;
        public const int GLFW_KEY_D = 68;
        public const int GLFW_KEY_SPACE = 32;
        public const int GLFW_KEY_LEFT = 263;
        public const int GLFW_KEY_RIGHT = 262;
        public const int GLFW_KEY_UP = 265;
        public const int GLFW_KEY_DOWN = 264;

        public const int GLFW_PRESS = 1;
        public const int GLFW_RELEASE = 0;
        public const int GLFW_REPEAT = 2;

        public const int GLFW_MOUSE_BUTTON_LEFT = 0;
        public const int GLFW_MOUSE_BUTTON_RIGHT = 1;
        public const int GLFW_MOUSE_BUTTON_MIDDLE = 2;

        public const int GLFW_CURSOR_DISABLED = 21939;
        public const int GLFW_CURSOR_NORMAL = 21299;
        public const int GLFW_CURSOR_CAPTURED = 21941;

        public const int GLFW_CONNECTED = 1;
        public const int GLFW_DISCONNECTED = 0;

        // GLFW_CLIENT_API / GLFW_NO_API
        private const int GLFW_CLIENT_API = 0x00022001;
        private const int GLFW_NO_API = 0;

        public static IntPtr CreateWindow(int width, int height, string title)
        {
            glfwSetErrorCallback(static (error, desc) =>
            {
                var msg = Marshal.PtrToStringAnsi(desc);
                Console.WriteLine($"[GLFW] Error {error}: {msg}");
            });

            if (glfwInit() == 0)
                throw new InvalidOperationException("Failed to initialize GLFW");

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

            var titlePtr = Marshal.StringToHGlobalAnsi(title);
            try
            {
                var window = glfwCreateWindow(width, height, titlePtr, IntPtr.Zero, IntPtr.Zero);
                if (window == IntPtr.Zero)
                    throw new InvalidOperationException("Failed to create GLFW window");
                return window;
            }
            finally { Marshal.FreeHGlobal(titlePtr); }
        }

        public static string[] GetRequiredInstanceExtensions()
        {
            uint count = 0;
            var ptr = glfwGetRequiredInstanceExtensions(ref count);
            if (ptr == IntPtr.Zero || count == 0)
            {
                // Sensible defaults per platform if GLFW cannot query
                if (OperatingSystem.IsWindows())
                    return new[] { "VK_KHR_surface", "VK_KHR_win32_surface" };
                if (OperatingSystem.IsMacOS())
                    return new[] { "VK_KHR_surface", "VK_EXT_metal_surface", "VK_KHR_portability_enumeration" };
                return new[] { "VK_KHR_surface", "VK_KHR_xcb_surface" };
            }

            var extensions = new string[count];
            for (uint i = 0; i < count; i++)
            {
                var extPtr = Marshal.ReadIntPtr(ptr, (int)(i * (uint)IntPtr.Size));
                extensions[i] = Marshal.PtrToStringAnsi(extPtr) ?? "";
            }
            return extensions;
        }

        public static IntPtr CreateVulkanSurface(IntPtr instance, IntPtr window)
        {
            IntPtr surface = IntPtr.Zero;
            var result = glfwCreateWindowSurface(instance, window, IntPtr.Zero, ref surface);
            if (result != 0 || surface == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create Vulkan surface: GLFW/Vulkan error {result}");
            return surface;
        }

        public static bool ShouldClose(IntPtr window) => glfwWindowShouldClose(window) != 0;

        public static void PollEvents() => glfwPollEvents();
        public static double GetTime() => glfwGetTime();

        public static void DestroyWindow(IntPtr window)
        {
            glfwDestroyWindow(window);
            glfwTerminate();
        }

        public static int GetKey(IntPtr window, int key) => glfwGetKey(window, key);

        public static void GetFramebufferSize(IntPtr window, out int width, out int height) =>
            glfwGetFramebufferSize(window, out width, out height);

        public static void GetCursorPos(IntPtr window, out double xpos, out double ypos) =>
            glfwGetCursorPos(window, out xpos, out ypos);

        public static void SetWindowShouldClose(IntPtr window, int value) =>
            glfwSetWindowShouldClose(window, value);
    }
}
