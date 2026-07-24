using System.Runtime.InteropServices;
using System.Text;

namespace Synapse.VR;

/// <summary>
/// Minimal OpenXR 1.0 P/Invoke surface for instance + Vulkan2 session + swapchain acquire/release.
/// Loader name: <c>openxr_loader</c> (Windows: openxr_loader.dll; Linux: libopenxr_loader.so).
/// </summary>
internal static class OpenXrNative
{
    public const int XR_SUCCESS = 0;
    public const int XR_TIMEOUT_EXPIRED = 1;
    public const int XR_SESSION_LOSS_PENDING = 3;
    public const int XR_ERROR_FORM_FACTOR_UNAVAILABLE = -33;

    public const ulong XR_NULL_HANDLE = 0;
    public const long XR_INFINITE_DURATION = 0x7fffffffffffffffL;

    public const int XR_TYPE_INSTANCE_CREATE_INFO = 3;
    public const int XR_TYPE_SYSTEM_GET_INFO = 4;
    public const int XR_TYPE_SESSION_CREATE_INFO = 22;
    public const int XR_TYPE_SWAPCHAIN_CREATE_INFO = 9;
    public const int XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO = 11;
    public const int XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO = 12;
    public const int XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO = 13;
    public const int XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR = 1000150000;
    public const int XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR = 1000150001;

    public const int XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY = 1;
    public const int XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO = 2;
    public const int XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT = 0x00000001;
    public const int XR_SWAPCHAIN_USAGE_SAMPLED_BIT = 0x00000020;

    // VK_FORMAT_R8G8B8A8_SRGB
    public const long XR_VK_FORMAT_R8G8B8A8_SRGB = 43;

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrCreateInstance")]
    public static extern int xrCreateInstance(in XrInstanceCreateInfo createInfo, out ulong instance);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrDestroyInstance")]
    public static extern int xrDestroyInstance(ulong instance);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrGetInstanceProcAddr", CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
    public static extern int xrGetInstanceProcAddr(ulong instance, string name, out IntPtr function);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrGetSystem")]
    public static extern int xrGetSystem(ulong instance, in XrSystemGetInfo getInfo, out ulong systemId);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrCreateSession")]
    public static extern int xrCreateSession(ulong instance, in XrSessionCreateInfo createInfo, out ulong session);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrDestroySession")]
    public static extern int xrDestroySession(ulong session);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrCreateSwapchain")]
    public static extern int xrCreateSwapchain(ulong session, in XrSwapchainCreateInfo createInfo, out ulong swapchain);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrDestroySwapchain")]
    public static extern int xrDestroySwapchain(ulong swapchain);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrEnumerateSwapchainImages")]
    public static extern int xrEnumerateSwapchainImages(ulong swapchain, uint imageCapacityInput, out uint imageCountOutput, IntPtr images);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrAcquireSwapchainImage")]
    public static extern int xrAcquireSwapchainImage(ulong swapchain, in XrSwapchainImageAcquireInfo acquireInfo, out uint index);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrWaitSwapchainImage")]
    public static extern int xrWaitSwapchainImage(ulong swapchain, in XrSwapchainImageWaitInfo waitInfo);

    [DllImport("openxr_loader", CallingConvention = CallingConvention.Cdecl, EntryPoint = "xrReleaseSwapchainImage")]
    public static extern int xrReleaseSwapchainImage(ulong swapchain, in XrSwapchainImageReleaseInfo releaseInfo);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate int XrGetVulkanGraphicsRequirements2KHR(ulong instance, ulong systemId, ref XrGraphicsRequirementsVulkan2KHR requirements);

    public static bool TryGetProc<T>(ulong instance, string name, out T? del) where T : Delegate
    {
        del = null;
        if (xrGetInstanceProcAddr(instance, name, out IntPtr ptr) != XR_SUCCESS || ptr == IntPtr.Zero)
            return false;
        del = Marshal.GetDelegateForFunctionPointer<T>(ptr);
        return true;
    }

    public static unsafe void WriteAscii(byte* dest, int capacity, string value)
    {
        int n = Math.Min(capacity - 1, Encoding.ASCII.GetByteCount(value));
        fixed (byte* src = Encoding.ASCII.GetBytes(value))
            Buffer.MemoryCopy(src, dest, capacity - 1, n);
        dest[n] = 0;
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct XrApplicationInfo
    {
        public fixed byte applicationName[128];
        public uint applicationVersion;
        public fixed byte engineName[128];
        public uint engineVersion;
        public ulong apiVersion;
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct XrInstanceCreateInfo
    {
        public int type;
        public IntPtr next;
        public ulong createFlags;
        public XrApplicationInfo applicationInfo;
        public uint enabledApiLayerCount;
        public IntPtr enabledApiLayerNames;
        public uint enabledExtensionCount;
        public IntPtr enabledExtensionNames;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct XrSystemGetInfo
    {
        public int type;
        public IntPtr next;
        public int formFactor;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct XrSessionCreateInfo
    {
        public int type;
        public IntPtr next;
        public ulong createFlags;
        public ulong systemId;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct XrGraphicsBindingVulkan2KHR
    {
        public int type;
        public IntPtr next;
        public IntPtr instance;
        public IntPtr physicalDevice;
        public IntPtr device;
        public uint queueFamilyIndex;
        public uint queueIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct XrGraphicsRequirementsVulkan2KHR
    {
        public int type;
        public IntPtr next;
        public ulong minApiVersionSupported;
        public ulong maxApiVersionSupported;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct XrSwapchainCreateInfo
    {
        public int type;
        public IntPtr next;
        public ulong createFlags;
        public ulong usageFlags;
        public long format;
        public uint sampleCount;
        public uint width;
        public uint height;
        public uint faceCount;
        public uint arraySize;
        public uint mipCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct XrSwapchainImageAcquireInfo
    {
        public int type;
        public IntPtr next;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct XrSwapchainImageWaitInfo
    {
        public int type;
        public IntPtr next;
        public long timeout;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct XrSwapchainImageReleaseInfo
    {
        public int type;
        public IntPtr next;
    }

    /// <summary>XR_KHR_vulkan_swapchain_format_list / Vulkan2 image header (type + next + VkImage).</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct XrSwapchainImageVulkan2KHR
    {
        public int type; // XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR = 1000150002
        public IntPtr next;
        public IntPtr image;
    }

    public const int XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR = 1000150002;
}
