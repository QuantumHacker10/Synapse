namespace Synapse.VR;

/// <summary>Vulkan swapchain image managed through OpenXR (v2.2).</summary>
public sealed class OpenXrVulkanSwapchain : IDisposable
{
    private readonly object _gate = new();
    private int _currentIndex;
    private bool _acquired;

    public OpenXrVulkanSwapchain(int imageCount, int width, int height, ulong vulkanFormat = 44 /* VK_FORMAT_B8G8R8A8_SRGB */)
    {
        ImageCount = Math.Max(2, imageCount);
        Width = width;
        Height = height;
        VulkanFormat = vulkanFormat;
        VulkanImageHandles = new ulong[ImageCount];
        for (int i = 0; i < ImageCount; i++)
            VulkanImageHandles[i] = (ulong)(0x1000 + i);
    }

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
