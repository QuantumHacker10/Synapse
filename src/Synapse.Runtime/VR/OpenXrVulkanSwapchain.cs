using System;
using Synapse.Core.Maturity;

namespace Synapse.VR;

/// <summary>
/// OpenXR-style Vulkan swapchain wrapper. Supports native image handles from
/// <see cref="NativeOpenXrRuntime"/> or synthetic lab handles for simulate mode.
/// See <c>docs/MATURITY.md</c> (<c>VR.OpenXR</c>).
/// </summary>
[SynapseExperimental("VR.OpenXR", "Native or synthetic swapchain image handles depending on OpenXR path.")]
public sealed class OpenXrVulkanSwapchain : IDisposable
{
    private readonly object _gate = new();
    private int _currentIndex;
    private bool _acquired;
    private readonly bool _synthetic;

    private OpenXrVulkanSwapchain(int imageCount, int width, int height, ulong[] handles, bool synthetic, ulong vulkanFormat)
    {
        ImageCount = Math.Max(1, imageCount);
        Width = width;
        Height = height;
        VulkanFormat = vulkanFormat;
        _synthetic = synthetic;
        VulkanImageHandles = new ulong[ImageCount];
        for (int i = 0; i < ImageCount; i++)
        {
            VulkanImageHandles[i] = i < handles.Length && handles[i] != 0
                ? handles[i]
                : (synthetic ? (ulong)(0x1000 + i) : (0xA000_0000UL + (ulong)i));
        }
    }

    public static OpenXrVulkanSwapchain CreateSynthetic(int imageCount, int width, int height, ulong vulkanFormat = 44)
        => new(imageCount, width, height, Array.Empty<ulong>(), synthetic: true, vulkanFormat);

    public static OpenXrVulkanSwapchain FromNative(int imageCount, int width, int height, ulong[] nativeHandles, ulong vulkanFormat = 44)
        => new(imageCount, width, height, nativeHandles ?? Array.Empty<ulong>(), synthetic: false, vulkanFormat);

    /// <summary>Compatibility ctor — creates a synthetic swapchain (lab / tests).</summary>
    public OpenXrVulkanSwapchain(int imageCount, int width, int height, ulong vulkanFormat = 44 /* VK_FORMAT_B8G8R8A8_SRGB */)
        : this(imageCount, width, height, Array.Empty<ulong>(), synthetic: true, vulkanFormat)
    {
    }

    /// <summary>True when handles are fabricated locally (simulate mode), not from an OpenXR runtime.</summary>
    public bool UsesSyntheticImageHandles => _synthetic;

    public int ImageCount { get; }
    public int Width { get; }
    public int Height { get; }
    public ulong VulkanFormat { get; }
    public ulong[] VulkanImageHandles { get; }
    public int CurrentIndex => _currentIndex;
    public bool IsAcquired => _acquired;

    public bool TryAcquire(out int imageIndex)
    {
        lock (_gate)
        {
            if (_acquired)
            {
                imageIndex = -1;
                return false;
            }

            _currentIndex = (_currentIndex + 1) % ImageCount;
            _acquired = true;
            imageIndex = _currentIndex;
            return true;
        }
    }

    /// <summary>Marks a native-acquired swapchain image as current (index from xrAcquireSwapchainImage).</summary>
    public void MarkAcquired(int imageIndex)
    {
        lock (_gate)
        {
            if (imageIndex < 0 || imageIndex >= ImageCount)
                throw new ArgumentOutOfRangeException(nameof(imageIndex));
            _currentIndex = imageIndex;
            _acquired = true;
        }
    }

    public void Release()
    {
        lock (_gate)
            _acquired = false;
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
            WaitSemaphores = Array.Empty<ulong>(),
            SignalSemaphores = Array.Empty<ulong>()
        };
    }

    public void Dispose()
    {
        Release();
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
    public ulong[] WaitSemaphores { get; init; } = Array.Empty<ulong>();
    public ulong[] SignalSemaphores { get; init; } = Array.Empty<ulong>();
}
