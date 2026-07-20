namespace Synapse.VR;

/// <summary>OpenXR + Vulkan VR session (v2 foundation).</summary>
public interface IVrSession : IAsyncDisposable
{
    bool IsAvailable { get; }
    bool IsRunning { get; }

    Task<bool> TryInitializeAsync(CancellationToken cancellationToken = default);

    Task PollEventsAsync(CancellationToken cancellationToken = default);
}

public sealed class HeadlessVrSession : IVrSession
{
    public bool IsAvailable { get; private set; }
    public bool IsRunning { get; private set; }

    public Task<bool> TryInitializeAsync(CancellationToken cancellationToken = default)
    {
        IsAvailable = false;
        IsRunning = false;
        return Task.FromResult(false);
    }

    public Task PollEventsAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public ValueTask DisposeAsync()
    {
        IsRunning = false;
        return ValueTask.CompletedTask;
    }
}

public static class VrSessionFactory
{
    public static IVrSession Create() => new HeadlessVrSession();
}
