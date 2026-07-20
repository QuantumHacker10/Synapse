// =============================================================================
// NativePlatform.cs — Synapse Omnia cross-platform surface & capability facade
// GLFW is the native primary path on Windows / Linux / macOS (MoltenVK).
// HWND embedding remains an optional Windows Studio-only path.
// =============================================================================

using System;
using System.Runtime.InteropServices;

namespace GDNN.Platform
{
    /// <summary>Which native windowing / surface backend is active.</summary>
    public enum NativeSurfaceBackend : byte
    {
        None = 0,
        Glfw = 1,
        Win32Hwnd = 2
    }

    /// <summary>Detected OS / GPU loader capabilities for Synapse Omnia.</summary>
    public sealed class PlatformCapabilities
    {
        public OSPlatform Os { get; init; }
        public Architecture ProcessArch { get; init; }
        public string Rid { get; init; } = "";
        public bool IsWindows { get; init; }
        public bool IsLinux { get; init; }
        public bool IsMacOs { get; init; }
        public bool GlfwAvailable { get; init; }
        public bool VulkanLoaderAvailable { get; init; }
        public bool MoltenVkLikely { get; init; }
        public bool HwndEmbedSupported { get; init; }
        public NativeSurfaceBackend PreferredBackend { get; init; }
        public string Summary { get; init; } = "";
    }

    /// <summary>
    /// Creates a Vulkan WSI surface for the render engine.
    /// Implementations: GLFW (all platforms), Win32 HWND (Windows embed only).
    /// </summary>
    public interface IVulkanSurfaceFactory
    {
        NativeSurfaceBackend Backend { get; }
        /// <summary>Creates or returns the native window handle used for presentation.</summary>
        IntPtr CreateWindow(int width, int height, string title);
        /// <summary>Creates a VkSurfaceKHR for the given Vulkan instance.</summary>
        IntPtr CreateSurface(IntPtr vulkanInstance, IntPtr windowOrHwnd);
        /// <summary>Instance extensions required by this backend (may be empty if GLFW reports them).</summary>
        string[] GetRequiredInstanceExtensions();
        void DestroyWindow(IntPtr window);
        void PollEvents();
        bool ShouldClose(IntPtr window);
    }

    /// <summary>GLFW-backed surface factory — primary native path on all OSes.</summary>
    public sealed class GlfwSurfaceFactory : IVulkanSurfaceFactory
    {
        public NativeSurfaceBackend Backend => NativeSurfaceBackend.Glfw;

        public IntPtr CreateWindow(int width, int height, string title) =>
            GlfwWindow.CreateWindow(width, height, title);

        public IntPtr CreateSurface(IntPtr vulkanInstance, IntPtr window) =>
            GlfwWindow.CreateVulkanSurface(vulkanInstance, window);

        public string[] GetRequiredInstanceExtensions() =>
            GlfwWindow.GetRequiredInstanceExtensions();

        public void DestroyWindow(IntPtr window)
        {
            if (window != IntPtr.Zero)
                GlfwWindow.DestroyWindow(window);
        }

        public void PollEvents() => GlfwWindow.PollEvents();

        public bool ShouldClose(IntPtr window) =>
            window != IntPtr.Zero && GlfwWindow.ShouldClose(window);
    }

    /// <summary>Win32 HWND surface factory — Studio viewport embedding on Windows only.</summary>
    public sealed class Win32HwndSurfaceFactory : IVulkanSurfaceFactory
    {
        private readonly IntPtr _parentHwnd;

        public Win32HwndSurfaceFactory(IntPtr parentHwnd)
        {
            if (!OperatingSystem.IsWindows())
                throw new PlatformNotSupportedException("Win32 HWND surfaces require Windows.");
            _parentHwnd = parentHwnd;
        }

        public NativeSurfaceBackend Backend => NativeSurfaceBackend.Win32Hwnd;

        public IntPtr CreateWindow(int width, int height, string title) =>
            Win32VulkanSurface.CreateChildWindow(_parentHwnd, width, height);

        public IntPtr CreateSurface(IntPtr vulkanInstance, IntPtr hwnd) =>
            Win32VulkanSurface.CreateSurface(vulkanInstance, hwnd);

        public string[] GetRequiredInstanceExtensions() => new[] { "VK_KHR_surface", "VK_KHR_win32_surface" };

        public void DestroyWindow(IntPtr window)
        {
            // Child HWND lifetime is owned by the Avalonia host.
        }

        public void PollEvents() { }

        public bool ShouldClose(IntPtr window) => false;
    }

    /// <summary>
    /// Cross-platform entry point for Synapse Omnia: discover capabilities and
    /// pick the correct surface factory (GLFW by default on every OS).
    /// </summary>
    public static class NativePlatform
    {
        private static PlatformCapabilities? _cached;

        public static PlatformCapabilities Probe()
        {
            if (_cached != null)
                return _cached;

            NativeLibraryResolver.EnsureRegistered();

            bool glfw = TryProbeGlfw();
            bool vulkan = TryProbeVulkan();
            bool mac = OperatingSystem.IsMacOS();
            bool win = OperatingSystem.IsWindows();
            bool linux = OperatingSystem.IsLinux();

            var caps = new PlatformCapabilities
            {
                Os = win ? OSPlatform.Windows : mac ? OSPlatform.OSX : OSPlatform.Linux,
                ProcessArch = RuntimeInformation.ProcessArchitecture,
                Rid = RuntimeInformation.RuntimeIdentifier,
                IsWindows = win,
                IsLinux = linux,
                IsMacOs = mac,
                GlfwAvailable = glfw,
                VulkanLoaderAvailable = vulkan,
                MoltenVkLikely = mac,
                HwndEmbedSupported = win,
                PreferredBackend = glfw ? NativeSurfaceBackend.Glfw : NativeSurfaceBackend.None,
                Summary = BuildSummary(win, linux, mac, glfw, vulkan)
            };
            _cached = caps;
            return caps;
        }

        /// <summary>Creates the preferred native surface factory (GLFW). Throws if unavailable.</summary>
        public static IVulkanSurfaceFactory CreatePrimarySurfaceFactory()
        {
            var caps = Probe();
            if (!caps.GlfwAvailable)
                throw new DllNotFoundException(
                    "GLFW is required for native multiplatform rendering. " +
                    "Install glfw (Windows: glfw3.dll beside the exe; Linux: libglfw.so.3; macOS: libglfw.3.dylib).");
            return new GlfwSurfaceFactory();
        }

        /// <summary>Creates an HWND embed factory on Windows; throws elsewhere.</summary>
        public static IVulkanSurfaceFactory CreateHwndSurfaceFactory(IntPtr parentHwnd) =>
            new Win32HwndSurfaceFactory(parentHwnd);

        public static void InvalidateProbeCache() => _cached = null;

        private static bool TryProbeGlfw()
        {
            try
            {
                foreach (var c in new[] { "glfw3", "libglfw.so.3", "libglfw.3" })
                {
                    if (NativeLibrary.TryLoad(c, out var h) || NativeLibrary.TryLoad(NativeLibraryResolver.GlfwLibraryName, out h))
                    {
                        if (h != IntPtr.Zero)
                            return true;
                    }
                }
                // GlfwWindow.Init may still succeed via DllImport resolver with fuller candidate list.
                return GlfwWindow.IsAvailable();
            }
            catch
            {
                return false;
            }
        }

        private static bool TryProbeVulkan()
        {
            try
            {
                var h = NativeLibraryResolver.LoadVulkan();
                return h != IntPtr.Zero;
            }
            catch
            {
                return false;
            }
        }

        private static string BuildSummary(bool win, bool linux, bool mac, bool glfw, bool vulkan)
        {
            string os = win ? "Windows" : mac ? "macOS (MoltenVK)" : linux ? "Linux" : "Unknown";
            return $"{os}/{RuntimeInformation.RuntimeIdentifier}: GLFW={(glfw ? "ok" : "missing")}, Vulkan={(vulkan ? "ok" : "missing")}, primary=GLFW";
        }
    }
}
