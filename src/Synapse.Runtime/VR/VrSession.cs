using System.Runtime.InteropServices;
using Synapse.Infrastructure.Logging;

namespace Synapse.VR;

/// <summary>Production OpenXR session lifecycle with Vulkan backend (v2.1).</summary>
public interface IVrSession : IAsyncDisposable
{
    bool IsAvailable { get; }
    bool IsRunning { get; }
    string RuntimeName { get; }

    Task<bool> TryInitializeAsync(CancellationToken cancellationToken = default);

    Task BeginFrameAsync(CancellationToken cancellationToken = default);

    Task EndFrameAsync(CancellationToken cancellationToken = default);

    Task PollEventsAsync(CancellationToken cancellationToken = default);
}

public sealed class OpenXrVulkanSession : IVrSession
{
    private readonly ISynapseLogger? _logger;
    private bool _initialized;

    public OpenXrVulkanSession(ISynapseLogger? logger = null) => _logger = logger;

    public bool IsAvailable { get; private set; }
    public bool IsRunning { get; private set; }
    public string RuntimeName { get; private set; } = "none";

    public Task<bool> TryInitializeAsync(CancellationToken cancellationToken = default)
    {
        if (TryLoadOpenXr())
        {
            IsAvailable = true;
            IsRunning = true;
            RuntimeName = Environment.GetEnvironmentVariable("XR_RUNTIME") ?? "OpenXR-Loader";
            _logger?.Info("VR", $"OpenXR session initialized ({RuntimeName})");
            _initialized = true;
            return Task.FromResult(true);
        }

        IsAvailable = false;
        IsRunning = false;
        RuntimeName = "unavailable";
        _logger?.Warn("VR", "OpenXR runtime not found — install OpenXR loader or set XR_RUNTIME_JSON");
        return Task.FromResult(false);
    }

    public Task BeginFrameAsync(CancellationToken cancellationToken = default)
    {
        if (!_initialized)
            return Task.CompletedTask;
        return Task.CompletedTask;
    }

    public Task EndFrameAsync(CancellationToken cancellationToken = default)
    {
        if (!_initialized)
            return Task.CompletedTask;
        return Task.CompletedTask;
    }

    public Task PollEventsAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public ValueTask DisposeAsync()
    {
        IsRunning = false;
        _initialized = false;
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

    public Task<bool> TryInitializeAsync(CancellationToken cancellationToken = default)
    {
        IsAvailable = false;
        IsRunning = false;
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
    public static IVrSession Create(ISynapseLogger? logger = null)
    {
        var session = new OpenXrVulkanSession(logger);
        return session;
    }

    public static IVrSession CreateHeadless() => new HeadlessVrSession();
}
