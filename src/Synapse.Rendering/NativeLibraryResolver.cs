using System;
using System.Reflection;
using System.Runtime.InteropServices;

namespace GDNN.Platform
{
    /// <summary>
    /// Resolves GLFW and Vulkan native libraries across Windows, Linux, and macOS.
    /// </summary>
    public static class NativeLibraryResolver
    {
        private static bool _registered;

        public static void EnsureRegistered()
        {
            if (_registered) return;
            NativeLibrary.SetDllImportResolver(typeof(NativeLibraryResolver).Assembly, Resolve);
            _registered = true;
        }

        public static string GlfwLibraryName =>
            OperatingSystem.IsWindows() ? "glfw3" :
            OperatingSystem.IsMacOS() ? "libglfw.3" :
            "libglfw.so.3";

        public static string VulkanLibraryName =>
            OperatingSystem.IsWindows() ? "vulkan-1" :
            OperatingSystem.IsMacOS() ? "libvulkan.1" :
            "libvulkan.so.1";

        public static IntPtr LoadVulkan()
        {
            EnsureRegistered();
            if (NativeLibrary.TryLoad(VulkanLibraryName, out var handle))
                return handle;

            // Fallbacks
            foreach (var candidate in VulkanCandidates())
            {
                if (NativeLibrary.TryLoad(candidate, out handle))
                    return handle;
            }

            throw new DllNotFoundException(
                $"Unable to load Vulkan loader. Tried: {string.Join(", ", VulkanCandidates())}");
        }

        public static IntPtr GetExport(IntPtr library, string name)
        {
            if (library == IntPtr.Zero) return IntPtr.Zero;
            return NativeLibrary.TryGetExport(library, name, out var export) ? export : IntPtr.Zero;
        }

        private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            var candidates = libraryName switch
            {
                "glfw3" or "glfw" or "libglfw" or "libglfw.so.3" or "libglfw.3" => GlfwCandidates(),
                "vulkan-1" or "vulkan" or "libvulkan" or "libvulkan.so.1" or "libvulkan.1" => VulkanCandidates(),
                _ => new[] { libraryName }
            };

            foreach (var candidate in candidates)
            {
                if (NativeLibrary.TryLoad(candidate, assembly, searchPath, out var handle))
                    return handle;
            }

            return IntPtr.Zero;
        }

        private static string[] GlfwCandidates()
        {
            if (OperatingSystem.IsWindows())
                return new[] { "glfw3", "glfw3.dll" };
            if (OperatingSystem.IsMacOS())
                return new[] { "libglfw.3.dylib", "libglfw.dylib", "glfw" };
            return new[] { "libglfw.so.3", "libglfw.so", "glfw" };
        }

        private static string[] VulkanCandidates()
        {
            if (OperatingSystem.IsWindows())
                return new[] { "vulkan-1", "vulkan-1.dll" };
            if (OperatingSystem.IsMacOS())
                return new[] { "libvulkan.1.dylib", "libMoltenVK.dylib", "libvulkan.dylib", "vulkan" };
            return new[] { "libvulkan.so.1", "libvulkan.so", "vulkan" };
        }
    }
}
