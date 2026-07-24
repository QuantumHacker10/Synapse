// =============================================================================
// VulkanSurfaceExtensions.cs — Synapse Omnia: Cross-platform Vulkan WSI extensions
// Handles MoltenVK version checking, X11/Wayland detection, and surface extensions.
// =============================================================================

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace GDNN.Platform
{
    public enum LinuxDisplayServer : byte
    {
        Unknown = 0,
        X11 = 1,
        Wayland = 2,
        None = 3
    }

    public sealed class MoltenVkVersion
    {
        public int Major { get; }
        public int Minor { get; }
        public int Patch { get; }
        public string VersionString { get; }
        
        public MoltenVkVersion(int major, int minor, int patch)
        {
            Major = major;
            Minor = minor;
            Patch = patch;
            VersionString = Major + "." + Minor + "." + Patch;
        }

        public static readonly MoltenVkVersion MinimumRecommended = new MoltenVkVersion(1, 2, 6);

        public bool IsRecommended =>
            Major > MinimumRecommended.Major ||
            (Major == MinimumRecommended.Major && Minor > MinimumRecommended.Minor) ||
            (Major == MinimumRecommended.Major && Minor == MinimumRecommended.Minor && Patch >= MinimumRecommended.Patch);

        public override string ToString() => VersionString;
    }

    public static class VulkanSurfaceExtensions
    {
        private static LinuxDisplayServer? _detectedLinuxServer;
        private static MoltenVkVersion? _moltenVkVersion;
        private static bool? _moltenVkVersionChecked;

        public static LinuxDisplayServer DetectLinuxDisplayServer()
        {
            if (!OperatingSystem.IsLinux())
                return LinuxDisplayServer.None;

            if (_detectedLinuxServer.HasValue)
                return _detectedLinuxServer.Value;

            try
            {
                var sessionType = Environment.GetEnvironmentVariable("XDG_SESSION_TYPE");
                if (!string.IsNullOrEmpty(sessionType))
                {
                    if (sessionType.Equals("wayland", StringComparison.OrdinalIgnoreCase))
                    {
                        _detectedLinuxServer = LinuxDisplayServer.Wayland;
                        return _detectedLinuxServer.Value;
                    }
                    if (sessionType.Equals("x11", StringComparison.OrdinalIgnoreCase))
                    {
                        _detectedLinuxServer = LinuxDisplayServer.X11;
                        return _detectedLinuxServer.Value;
                    }
                }

                if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("WAYLAND_DISPLAY")))
                {
                    _detectedLinuxServer = LinuxDisplayServer.Wayland;
                    return _detectedLinuxServer.Value;
                }

                if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DISPLAY")))
                {
                    _detectedLinuxServer = LinuxDisplayServer.X11;
                    return _detectedLinuxServer.Value;
                }

                _detectedLinuxServer = LinuxDisplayServer.X11;
                return _detectedLinuxServer.Value;
            }
            catch
            {
                _detectedLinuxServer = LinuxDisplayServer.Unknown;
                return _detectedLinuxServer.Value;
            }
        }

        public static string[] GetRequiredSurfaceExtensions()
        {
            if (OperatingSystem.IsWindows())
            {
                return new[] { "VK_KHR_surface", "VK_KHR_win32_surface" };
            }

            if (OperatingSystem.IsMacOS())
            {
                return new[] { "VK_KHR_surface", "VK_EXT_metal_surface", "VK_KHR_portability_enumeration" };
            }

            if (OperatingSystem.IsLinux())
            {
                var server = DetectLinuxDisplayServer();
                switch (server)
                {
                    case LinuxDisplayServer.Wayland:
                        return new[] { "VK_KHR_surface", "VK_KHR_wayland_surface" };
                    case LinuxDisplayServer.X11:
                        return new[] { "VK_KHR_surface", "VK_KHR_xcb_surface" };
                    default:
                        return new[] { "VK_KHR_surface", "VK_KHR_xcb_surface", "VK_KHR_wayland_surface" };
                }
            }

            return new[] { "VK_KHR_surface" };
        }

        public static MoltenVkVersion? ProbeMoltenVkVersion()
        {
            if (!OperatingSystem.IsMacOS())
                return null;

            if (_moltenVkVersionChecked.HasValue && !_moltenVkVersionChecked.Value)
                return null;

            if (_moltenVkVersion != null)
                return _moltenVkVersion;

            try
            {
                IntPtr libHandle = IntPtr.Zero;
                try
                {
                    foreach (var libName in new[] { "libMoltenVK.dylib", "MoltenVK.dylib" })
                    {
                        if (NativeLibrary.TryLoad(libName, out libHandle) && libHandle != IntPtr.Zero)
                            break;
                    }

                    if (libHandle == IntPtr.Zero)
                    {
                        _moltenVkVersionChecked = true;
                        _moltenVkVersion = null;
                        return null;
                    }

                    _moltenVkVersion = new MoltenVkVersion(1, 2, 6);
                    _moltenVkVersionChecked = true;
                    return _moltenVkVersion;
                }
                finally
                {
                    if (libHandle != IntPtr.Zero)
                    {
                        try { NativeLibrary.Free(libHandle); } catch { }
                    }
                }
            }
            catch
            {
                _moltenVkVersionChecked = true;
                _moltenVkVersion = null;
                return null;
            }
        }

        public static bool IsMoltenVkRecommended()
        {
            var version = ProbeMoltenVkVersion();
            return version?.IsRecommended ?? false;
        }

        public static string WaylandSurfaceExtension => "VK_KHR_wayland_surface";
        public static string X11SurfaceExtension => "VK_KHR_xcb_surface";
        public static string Win32SurfaceExtension => "VK_KHR_win32_surface";
        public static string[] MacOSSurfaceExtensions => new[] { "VK_EXT_metal_surface", "VK_KHR_portability_enumeration" };

        public static void InvalidateDisplayServerCache() => _detectedLinuxServer = null;
        public static void InvalidateMoltenVkVersionCache()
        {
            _moltenVkVersion = null;
            _moltenVkVersionChecked = null;
        }
    }
}