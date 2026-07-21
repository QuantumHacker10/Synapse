using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Core.Maturity;
using Synapse.Infrastructure.Logging;

namespace Synapse.VR;

/// <summary>
/// EXPERIMENTAL — OpenXR session lifecycle scaffold with synthetic Vulkan swapchain (v2.2).
/// Detects a loader when present, but does not drive a real XR frame loop.
/// See <c>docs/MATURITY.md</c> (<c>VR.OpenXR</c>).
/// </summary>
[SynapseExperimental("VR.OpenXR", "Loader probe + synthetic swapchain; not a production OpenXR session.")]
public interface IVrSession : IAsyncDisposable
{
    bool IsAvailable { get; }
    bool IsRunning { get; }
    string RuntimeName { get; }
    OpenXrVulkanSwapchain? Swapchain { get; }

    Task<bool> TryInitializeAsync(int width = 1280, int height = 720, CancellationToken cancellationToken = default);

    Task BeginFrameAsync(CancellationToken cancellationToken = default);

    Task EndFrameAsync(CancellationToken cancellationToken = default);

    Task PollEventsAsync(CancellationToken cancellationToken = default);
}

/// <summary>EXPERIMENTAL OpenXR session: loader detection only, synthetic swapchain images.</summary>
[SynapseExperimental("VR.OpenXR", "Loader probe + synthetic swapchain; not a production OpenXR session.")]
public sealed class OpenXrVulkanSession : IVrSession
{
    private readonly ISynapseLogger? _logger;
    private bool _initialized;
    private int _frameCounter;

    public OpenXrVulkanSession(ISynapseLogger? logger = null) => _logger = logger;

    public bool IsAvailable { get; private set; }
    public bool IsRunning { get; private set; }
    public string RuntimeName { get; private set; } = "none";
    public OpenXrVulkanSwapchain? Swapchain { get; private set; }

    public Task<bool> TryInitializeAsync(int width = 1280, int height = 720, CancellationToken cancellationToken = default)
    {
        if (TryLoadOpenXr())
        {
            IsAvailable = true;
            IsRunning = true;
            RuntimeName = Environment.GetEnvironmentVariable("XR_RUNTIME") ?? "OpenXR-Loader";
            Swapchain = new OpenXrVulkanSwapchain(imageCount: 3, width, height);
            _logger?.Warn("VR",
                $"EXPERIMENTAL OpenXR scaffold active ({width}x{height}, {RuntimeName}) — synthetic swapchain, not production XR");
            _initialized = true;
            return Task.FromResult(true);
        }

        IsAvailable = false;
        IsRunning = false;
        RuntimeName = "unavailable";
        Swapchain = null;
        _logger?.Warn("VR", "OpenXR runtime not found — install OpenXR loader or set XR_RUNTIME_JSON");
        return Task.FromResult(false);
    }

    public Task BeginFrameAsync(CancellationToken cancellationToken = default)
    {
        if (!_initialized || Swapchain == null)
            return Task.CompletedTask;

        if (!Swapchain.TryAcquire(out var index))
            _logger?.Warn("VR", "Swapchain acquire skipped (already acquired)");
        else
            _logger?.Debug("VR", $"Acquire synthetic swapchain image {index}");

        return Task.CompletedTask;
    }

    public Task EndFrameAsync(CancellationToken cancellationToken = default)
    {
        if (!_initialized || Swapchain == null)
            return Task.CompletedTask;

        var frame = Swapchain.PrepareSubmit(vulkanQueueFamily: 0, vulkanQueue: 0);
        _frameCounter++;
        _logger?.Debug("VR", $"Submit synthetic XR frame #{_frameCounter} image={frame.ImageHandle:x}");
        Swapchain.Release();
        return Task.CompletedTask;
    }

    public Task PollEventsAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public ValueTask DisposeAsync()
    {
        IsRunning = false;
        _initialized = false;
        Swapchain?.Dispose();
        Swapchain = null;
        return ValueTask.CompletedTask;
    }

    private static bool TryLoadOpenXr()
    {
        if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("XR_RUNTIME_JSON")))
            return true;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return NativeLibrary.TryLoad("openxr_loader.dll", out _);

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return NativeLibrary.TryLoad("libopenxr_loader.so", out _) ||
                   NativeLibrary.TryLoad("libopenxr_loader.so.1", out _);

        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return NativeLibrary.TryLoad("libopenxr_loader.dylib", out _);

        return false;
    }
}

public sealed class HeadlessVrSession : IVrSession
{
    public bool IsAvailable { get; private set; }
    public bool IsRunning { get; private set; }
    public string RuntimeName { get; private set; } = "headless";
    public OpenXrVulkanSwapchain? Swapchain { get; private set; }

    public Task<bool> TryInitializeAsync(int width = 1280, int height = 720, CancellationToken cancellationToken = default)
    {
        IsAvailable = false;
        IsRunning = false;
        Swapchain = null;
        return Task.FromResult(false);
    }

    public Task BeginFrameAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task EndFrameAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task PollEventsAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public ValueTask DisposeAsync()
    {
        IsRunning = false;
        return ValueTask.CompletedTask;
    }
}

public static class VrSessionFactory
{
    public static IVrSession Create(ISynapseLogger? logger = null) => new OpenXrVulkanSession(logger);

    public static IVrSession CreateHeadless() => new HeadlessVrSession();
}
