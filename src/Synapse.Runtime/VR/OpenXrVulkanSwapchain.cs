using System.Runtime.InteropServices;

namespace Synapse.VR;

/// <summary>
/// OpenXR swapchain (Vulkan2 images when a native session exists; deterministic simulated images otherwise).
/// </summary>
public sealed class OpenXrVulkanSwapchain : IDisposable
{
    private readonly object _gate = new();
    private readonly ulong _nativeSwapchain;
    private int _currentIndex;
    private bool _acquired;
    private bool _disposed;

    private OpenXrVulkanSwapchain(
        int imageCount,
        int width,
        int height,
        ulong vulkanFormat,
        ulong[] imageHandles,
        ulong nativeSwapchain,
        bool isSimulated)
    {
        ImageCount = Math.Max(2, imageCount);
        Width = width;
        Height = height;
        VulkanFormat = vulkanFormat;
        VulkanImageHandles = imageHandles;
        _nativeSwapchain = nativeSwapchain;
        IsSimulated = isSimulated;
    }

    public int ImageCount { get; }
    public int Width { get; }
    public int Height { get; }
    public ulong VulkanFormat { get; }
    public ulong[] VulkanImageHandles { get; }
    public int CurrentIndex => _currentIndex;
    public bool IsAcquired => _acquired;
    public bool IsSimulated { get; }
    public ulong NativeHandle => _nativeSwapchain;

    /// <summary>QA / headless swapchain with stable placeholder VkImage handles.</summary>
    public static OpenXrVulkanSwapchain CreateSimulated(
        int imageCount,
        int width,
        int height,
        ulong vulkanFormat = 44 /* VK_FORMAT_B8G8R8A8_SRGB */)
    {
        int count = Math.Max(2, imageCount);
        var handles = new ulong[count];
        for (int i = 0; i < count; i++)
            handles[i] = (ulong)(0x1000 + i);
        return new OpenXrVulkanSwapchain(count, width, height, vulkanFormat, handles, 0, isSimulated: true);
    }

    /// <summary>
    /// Creates a real OpenXR swapchain and enumerates <c>XrSwapchainImageVulkan2KHR</c> images.
    /// Returns null on any native failure (caller should fall back to simulated).
    /// </summary>
    public static OpenXrVulkanSwapchain? TryCreateNative(
        ulong session,
        int width,
        int height,
        long vulkanFormat = OpenXrNative.XR_VK_FORMAT_R8G8B8A8_SRGB)
    {
        if (session == OpenXrNative.XR_NULL_HANDLE)
            return null;

        var createInfo = new OpenXrNative.XrSwapchainCreateInfo
        {
            type = OpenXrNative.XR_TYPE_SWAPCHAIN_CREATE_INFO,
            next = IntPtr.Zero,
            createFlags = 0,
            usageFlags = OpenXrNative.XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
                         OpenXrNative.XR_SWAPCHAIN_USAGE_SAMPLED_BIT,
            format = vulkanFormat,
            sampleCount = 1,
            width = (uint)Math.Max(1, width),
            height = (uint)Math.Max(1, height),
            faceCount = 1,
            arraySize = 1,
            mipCount = 1
        };

        if (OpenXrNative.xrCreateSwapchain(session, in createInfo, out ulong swapchain) != OpenXrNative.XR_SUCCESS ||
            swapchain == OpenXrNative.XR_NULL_HANDLE)
            return null;

        if (OpenXrNative.xrEnumerateSwapchainImages(swapchain, 0, out uint count, IntPtr.Zero) != OpenXrNative.XR_SUCCESS ||
            count < 2)
        {
            _ = OpenXrNative.xrDestroySwapchain(swapchain);
            return null;
        }

        int stride = Marshal.SizeOf<OpenXrNative.XrSwapchainImageVulkan2KHR>();
        IntPtr imagesPtr = Marshal.AllocHGlobal((int)count * stride);
        try
        {
            for (int i = 0; i < count; i++)
            {
                var hdr = new OpenXrNative.XrSwapchainImageVulkan2KHR
                {
                    type = OpenXrNative.XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR,
                    next = IntPtr.Zero,
                    image = IntPtr.Zero
                };
                Marshal.StructureToPtr(hdr, imagesPtr + i * stride, false);
            }

            if (OpenXrNative.xrEnumerateSwapchainImages(swapchain, count, out count, imagesPtr) != OpenXrNative.XR_SUCCESS)
            {
                _ = OpenXrNative.xrDestroySwapchain(swapchain);
                return null;
            }

            var handles = new ulong[count];
            for (int i = 0; i < count; i++)
            {
                var img = Marshal.PtrToStructure<OpenXrNative.XrSwapchainImageVulkan2KHR>(imagesPtr + i * stride);
                handles[i] = (ulong)img.image;
            }

            return new OpenXrVulkanSwapchain(
                (int)count,
                width,
                height,
                (ulong)vulkanFormat,
                handles,
                swapchain,
                isSimulated: false);
        }
        catch
        {
            _ = OpenXrNative.xrDestroySwapchain(swapchain);
            return null;
        }
        finally
        {
            Marshal.FreeHGlobal(imagesPtr);
        }
    }

    // Back-compat ctor used by older tests.
    public OpenXrVulkanSwapchain(int imageCount, int width, int height, ulong vulkanFormat = 44)
        : this(
            Math.Max(2, imageCount),
            width,
            height,
            vulkanFormat,
            CreatePlaceholderHandles(Math.Max(2, imageCount)),
            0,
            isSimulated: true)
    {
    }

    private static ulong[] CreatePlaceholderHandles(int count)
    {
        var handles = new ulong[count];
        for (int i = 0; i < count; i++)
            handles[i] = (ulong)(0x1000 + i);
        return handles;
    }

    public bool TryAcquire(out int imageIndex)
    {
        lock (_gate)
        {
            if (_disposed || _acquired)
            {
                imageIndex = -1;
                return false;
            }

            if (!IsSimulated && _nativeSwapchain != OpenXrNative.XR_NULL_HANDLE)
            {
                var acquire = new OpenXrNative.XrSwapchainImageAcquireInfo
                {
                    type = OpenXrNative.XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO,
                    next = IntPtr.Zero
                };
                int rc = OpenXrNative.xrAcquireSwapchainImage(_nativeSwapchain, in acquire, out uint index);
                if (rc != OpenXrNative.XR_SUCCESS && rc != OpenXrNative.XR_SESSION_LOSS_PENDING)
                {
                    imageIndex = -1;
                    return false;
                }

                var wait = new OpenXrNative.XrSwapchainImageWaitInfo
                {
                    type = OpenXrNative.XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO,
                    next = IntPtr.Zero,
                    timeout = OpenXrNative.XR_INFINITE_DURATION
                };
                rc = OpenXrNative.xrWaitSwapchainImage(_nativeSwapchain, in wait);
                if (rc != OpenXrNative.XR_SUCCESS && rc != OpenXrNative.XR_TIMEOUT_EXPIRED)
                {
                    imageIndex = -1;
                    return false;
                }

                _currentIndex = (int)index;
                _acquired = true;
                imageIndex = _currentIndex;
                return true;
            }

            _currentIndex = (_currentIndex + 1) % ImageCount;
            _acquired = true;
            imageIndex = _currentIndex;
            return true;
        }
    }

    public void Release()
    {
        lock (_gate)
        {
            if (!_acquired)
                return;

            if (!IsSimulated && _nativeSwapchain != OpenXrNative.XR_NULL_HANDLE)
            {
                var release = new OpenXrNative.XrSwapchainImageReleaseInfo
                {
                    type = OpenXrNative.XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO,
                    next = IntPtr.Zero
                };
                _ = OpenXrNative.xrReleaseSwapchainImage(_nativeSwapchain, in release);
            }

            _acquired = false;
        }
    }

    public VrSwapchainFrame PrepareSubmit(ulong vulkanQueueFamily, ulong vulkanQueue)
    {
        return new VrSwapchainFrame
        {
            ImageIndex = _currentIndex,
            ImageHandle = VulkanImageHandles[_currentIndex],
            Width = Width,
            Height = Height,
            Format = VulkanFormat,
            QueueFamily = vulkanQueueFamily,
            Queue = vulkanQueue,
            IsSimulated = IsSimulated,
            WaitSemaphores = Array.Empty<ulong>(),
            SignalSemaphores = Array.Empty<ulong>()
        };
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_acquired)
            {
                if (!IsSimulated && _nativeSwapchain != OpenXrNative.XR_NULL_HANDLE)
                {
                    var release = new OpenXrNative.XrSwapchainImageReleaseInfo
                    {
                        type = OpenXrNative.XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO,
                        next = IntPtr.Zero
                    };
                    _ = OpenXrNative.xrReleaseSwapchainImage(_nativeSwapchain, in release);
                }

                _acquired = false;
            }

            if (!IsSimulated && _nativeSwapchain != OpenXrNative.XR_NULL_HANDLE)
                _ = OpenXrNative.xrDestroySwapchain(_nativeSwapchain);
        }
    }
}

public sealed class VrSwapchainFrame
{
    public int ImageIndex { get; init; }
    public ulong ImageHandle { get; init; }
    public int Width { get; init; }
    public int Height { get; init; }
    public ulong Format { get; init; }
    public ulong QueueFamily { get; init; }
    public ulong Queue { get; init; }
    public bool IsSimulated { get; init; }
    public ulong[] WaitSemaphores { get; init; } = Array.Empty<ulong>();
    public ulong[] SignalSemaphores { get; init; } = Array.Empty<ulong>();
}

/// <summary>Optional Vulkan device handles for XR_KHR_vulkan_enable2 session binding.</summary>
public readonly struct OpenXrVulkanBinding
{
    public OpenXrVulkanBinding(IntPtr instance, IntPtr physicalDevice, IntPtr device, uint queueFamilyIndex, uint queueIndex = 0)
    {
        Instance = instance;
        PhysicalDevice = physicalDevice;
        Device = device;
        QueueFamilyIndex = queueFamilyIndex;
        QueueIndex = queueIndex;
    }

    public IntPtr Instance { get; }
    public IntPtr PhysicalDevice { get; }
    public IntPtr Device { get; }
    public uint QueueFamilyIndex { get; }
    public uint QueueIndex { get; }

    public bool IsValid => Instance != IntPtr.Zero && PhysicalDevice != IntPtr.Zero && Device != IntPtr.Zero;
}
