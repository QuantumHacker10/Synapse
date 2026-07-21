using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.VR;

/// <summary>
/// OpenXR session lifecycle. Prefers a native Silk.NET OpenXR session when a loader/runtime
/// is available; falls back to synthetic lab mode only when <c>SYNAPSE_VR_SIMULATE=1</c>.
/// Owned by <see cref="Synapse.Runtime.EngineHost"/> and ticked by <see cref="Synapse.Runtime.FrameOrchestrator"/>.
/// </summary>
public interface IVrSession : IAsyncDisposable
{
    bool IsAvailable { get; }
    bool IsRunning { get; }
    string RuntimeName { get; }
    bool UsesNativeOpenXr { get; }
    OpenXrVulkanSwapchain? Swapchain { get; }

    Task<bool> TryInitializeAsync(int width = 1280, int height = 720, CancellationToken cancellationToken = default);

    Task BeginFrameAsync(CancellationToken cancellationToken = default);

    Task EndFrameAsync(CancellationToken cancellationToken = default);

    Task PollEventsAsync(CancellationToken cancellationToken = default);
}

/// <summary>OpenXR session: native Silk.NET path, optional synthetic simulate fallback.</summary>
public sealed class OpenXrVulkanSession : IVrSession
{
    private readonly ISynapseLogger? _logger;
    private NativeOpenXrRuntime? _native;
    private bool _initialized;
    private int _frameCounter;

    public OpenXrVulkanSession(ISynapseLogger? logger = null) => _logger = logger;

    public bool IsAvailable { get; private set; }
    public bool IsRunning { get; private set; }
    public string RuntimeName { get; private set; } = "none";
    public bool UsesNativeOpenXr { get; private set; }
    public OpenXrVulkanSwapchain? Swapchain { get; private set; }

    public Task<bool> TryInitializeAsync(int width = 1280, int height = 720, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // 1) Prefer real native OpenXR when a loader/runtime answers.
        var native = new NativeOpenXrRuntime(_logger);
        if (native.TryInitialize(width, height))
        {
            _native = native;
            UsesNativeOpenXr = true;
            IsAvailable = true;
            IsRunning = native.IsSessionRunning || native.IsInitialized;
            RuntimeName = native.RuntimeName;
            int w = native.RecommendedWidth > 0 ? native.RecommendedWidth : width;
            int h = native.RecommendedHeight > 0 ? native.RecommendedHeight : height;
            Swapchain = OpenXrVulkanSwapchain.FromNative(
                native.SwapchainImageHandles.Length > 0 ? native.SwapchainImageHandles.Length : 3,
                w,
                h,
                native.SwapchainImageHandles);
            _initialized = true;
            _logger?.Info("VR", $"Native OpenXR session active ({RuntimeName}, {w}x{h}, headless={native.UsesHeadless})");
            return Task.FromResult(true);
        }

        native.Dispose();
        _native = null;
        UsesNativeOpenXr = false;

        // 2) Lab path: synthetic swapchain only when explicitly requested.
        if (!IsSimulateModeEnabled())
        {
            IsAvailable = false;
            IsRunning = false;
            RuntimeName = TryProbeOpenXrLoader() ? "loader-detected-init-failed" : "unavailable";
            Swapchain = null;
            _logger?.Warn("VR",
                "OpenXR native init failed — install an OpenXR runtime (SteamVR/Monado) " +
                "or set SYNAPSE_VR_SIMULATE=1 for lab synthetic mode");
            return Task.FromResult(false);
        }

        IsAvailable = true;
        IsRunning = true;
        RuntimeName = Environment.GetEnvironmentVariable("XR_RUNTIME")
                      ?? (TryProbeOpenXrLoader() ? "OpenXR-Loader+simulate" : "simulate");
        Swapchain = OpenXrVulkanSwapchain.CreateSynthetic(imageCount: 3, width, height);
        _logger?.Warn("VR",
            $"OpenXR simulate mode ({width}x{height}, {RuntimeName}) — synthetic swapchain for lab only");
        _initialized = true;
        return Task.FromResult(true);
    }

    public Task BeginFrameAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!_initialized || Swapchain == null)
            return Task.CompletedTask;

        if (_native != null)
        {
            if (!_native.TryBeginFrame())
            {
                _logger?.Debug("VR", "Native OpenXR begin-frame skipped (session not ready)");
                return Task.CompletedTask;
            }

            int index = _native.CurrentSwapchainIndex;
            if (index >= 0)
                Swapchain.MarkAcquired(index);
            return Task.CompletedTask;
        }

        if (!Swapchain.TryAcquire(out var idx))
            _logger?.Warn("VR", "Swapchain acquire skipped (already acquired)");
        else
            _logger?.Debug("VR", $"Acquire synthetic swapchain image {idx}");

        return Task.CompletedTask;
    }

    public Task EndFrameAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!_initialized || Swapchain == null)
            return Task.CompletedTask;

        var frame = Swapchain.PrepareSubmit(vulkanQueueFamily: 0, vulkanQueue: 0);
        _frameCounter++;
        _logger?.Debug("VR", $"Submit XR frame #{_frameCounter} image={frame.ImageHandle:x} native={UsesNativeOpenXr}");

        if (_native != null)
        {
            _native.EndFrame();
            Swapchain.Release();
            return Task.CompletedTask;
        }

        Swapchain.Release();
        return Task.CompletedTask;
    }

    public Task PollEventsAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _native?.PollEvents();
        return Task.CompletedTask;
    }

    public ValueTask DisposeAsync()
    {
        IsRunning = false;
        _initialized = false;
        Swapchain?.Dispose();
        Swapchain = null;
        _native?.Dispose();
        _native = null;
        UsesNativeOpenXr = false;
        return ValueTask.CompletedTask;
    }

    private static bool IsSimulateModeEnabled()
    {
        var value = Environment.GetEnvironmentVariable("SYNAPSE_VR_SIMULATE");
        return string.Equals(value, "1", StringComparison.OrdinalIgnoreCase)
               || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase)
               || string.Equals(value, "yes", StringComparison.OrdinalIgnoreCase);
    }

    private static bool TryProbeOpenXrLoader()
    {
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
    public bool UsesNativeOpenXr => false;
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
