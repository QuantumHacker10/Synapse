using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace GDNN.Platform
{
    /// <summary>
    /// Resolves GLFW and Vulkan native libraries across Windows, Linux, and macOS,
    /// including app-local and <c>runtimes/{rid}/native</c> layout for publish.
    /// </summary>
    public static class NativeLibraryResolver
    {
        private static bool _registered;
        private static readonly object RegisterLock = new();

        public static void EnsureRegistered()
        {
            if (_registered)
                return;

            lock (RegisterLock)
            {
                if (_registered)
                    return;
                try
                {
                    NativeLibrary.SetDllImportResolver(typeof(NativeLibraryResolver).Assembly, Resolve);
                }
                catch (InvalidOperationException)
                {
                    // Resolver already installed for this assembly (multi-host / parallel tests).
                }
                _registered = true;
            }
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

            foreach (var candidate in VulkanCandidates())
            {
                if (TryLoadPath(candidate, out handle))
                    return handle;
            }

            throw new DllNotFoundException(
                $"Unable to load Vulkan loader. Tried: {string.Join(", ", VulkanCandidates())}");
        }

        public static IntPtr GetExport(IntPtr library, string name)
        {
            if (library == IntPtr.Zero)
                return IntPtr.Zero;
            return NativeLibrary.TryGetExport(library, name, out var export) ? export : IntPtr.Zero;
        }

        private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            var candidates = libraryName switch
            {
                "glfw3" or "glfw" or "libglfw" or "libglfw.so.3" or "libglfw.3" => GlfwCandidates(),
                "vulkan-1" or "vulkan" or "libvulkan" or "libvulkan.so.1" or "libvulkan.1" or "libMoltenVK" => VulkanCandidates(),
                _ => new[] { libraryName }
            };

            foreach (var candidate in candidates)
            {
                if (TryLoadPath(candidate, out var handle))
                    return handle;
                if (NativeLibrary.TryLoad(candidate, assembly, searchPath, out handle))
                    return handle;
            }

            return IntPtr.Zero;
        }

        private static bool TryLoadPath(string candidate, out IntPtr handle)
        {
            handle = IntPtr.Zero;
            // Absolute / relative file path
            if (candidate.Contains(Path.DirectorySeparatorChar) || candidate.Contains(Path.AltDirectorySeparatorChar)
                || candidate.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)
                || candidate.EndsWith(".so", StringComparison.OrdinalIgnoreCase)
                || candidate.EndsWith(".dylib", StringComparison.OrdinalIgnoreCase))
            {
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out handle))
                    return true;
            }

            return NativeLibrary.TryLoad(candidate, out handle);
        }

        private static IEnumerable<string> AppLocalNativeDirs()
        {
            string baseDir = AppContext.BaseDirectory;
            yield return baseDir;
            yield return Path.Combine(baseDir, "native");
            string rid = RuntimeInformation.RuntimeIdentifier;
            yield return Path.Combine(baseDir, "runtimes", rid, "native");
            // Common publish RIDs when running under a different host RID
            if (OperatingSystem.IsWindows())
            {
                yield return Path.Combine(baseDir, "runtimes", "win-x64", "native");
                yield return Path.Combine(baseDir, "runtimes", "win-arm64", "native");
            }
            else if (OperatingSystem.IsMacOS())
            {
                yield return Path.Combine(baseDir, "runtimes", "osx-arm64", "native");
                yield return Path.Combine(baseDir, "runtimes", "osx-x64", "native");
            }
            else
            {
                yield return Path.Combine(baseDir, "runtimes", "linux-x64", "native");
                yield return Path.Combine(baseDir, "runtimes", "linux-arm64", "native");
            }
        }

        private static string[] GlfwCandidates()
        {
            var list = new List<string>();
            if (OperatingSystem.IsWindows())
            {
                list.Add("glfw3");
                list.Add("glfw3.dll");
                foreach (var dir in AppLocalNativeDirs())
                    list.Add(Path.Combine(dir, "glfw3.dll"));
            }
            else if (OperatingSystem.IsMacOS())
            {
                list.Add("libglfw.3.dylib");
                list.Add("libglfw.dylib");
                list.Add("glfw");
                foreach (var dir in AppLocalNativeDirs())
                {
                    list.Add(Path.Combine(dir, "libglfw.3.dylib"));
                    list.Add(Path.Combine(dir, "libglfw.dylib"));
                }
                list.Add("/opt/homebrew/lib/libglfw.3.dylib");
                list.Add("/usr/local/lib/libglfw.3.dylib");
            }
            else
            {
                list.Add("libglfw.so.3");
                list.Add("libglfw.so");
                list.Add("glfw");
                foreach (var dir in AppLocalNativeDirs())
                {
                    list.Add(Path.Combine(dir, "libglfw.so.3"));
                    list.Add(Path.Combine(dir, "libglfw.so"));
                }
                list.Add("/usr/lib/x86_64-linux-gnu/libglfw.so.3");
                list.Add("/usr/lib/libglfw.so.3");
            }
            return list.ToArray();
        }

        private static string[] VulkanCandidates()
        {
            var list = new List<string>();
            if (OperatingSystem.IsWindows())
            {
                list.Add("vulkan-1");
                list.Add("vulkan-1.dll");
                foreach (var dir in AppLocalNativeDirs())
                    list.Add(Path.Combine(dir, "vulkan-1.dll"));
            }
            else if (OperatingSystem.IsMacOS())
            {
                list.Add("libvulkan.1.dylib");
                list.Add("libMoltenVK.dylib");
                list.Add("libvulkan.dylib");
                list.Add("vulkan");
                foreach (var dir in AppLocalNativeDirs())
                {
                    list.Add(Path.Combine(dir, "libMoltenVK.dylib"));
                    list.Add(Path.Combine(dir, "libvulkan.1.dylib"));
                }
                string? sdk = Environment.GetEnvironmentVariable("VULKAN_SDK");
                if (!string.IsNullOrEmpty(sdk))
                {
                    list.Add(Path.Combine(sdk, "lib", "libvulkan.1.dylib"));
                    list.Add(Path.Combine(sdk, "macOS", "lib", "libMoltenVK.dylib"));
                }
            }
            else
            {
                list.Add("libvulkan.so.1");
                list.Add("libvulkan.so");
                list.Add("vulkan");
                foreach (var dir in AppLocalNativeDirs())
                {
                    list.Add(Path.Combine(dir, "libvulkan.so.1"));
                    list.Add(Path.Combine(dir, "libvulkan.so"));
                }
                list.Add("/usr/lib/x86_64-linux-gnu/libvulkan.so.1");
                list.Add("/usr/lib/libvulkan.so.1");
                string? sdk = Environment.GetEnvironmentVariable("VULKAN_SDK");
                if (!string.IsNullOrEmpty(sdk))
                    list.Add(Path.Combine(sdk, "lib", "libvulkan.so.1"));
            }
            return list.ToArray();
        }
    }
}
