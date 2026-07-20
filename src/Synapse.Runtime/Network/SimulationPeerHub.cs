using System;
using System.Threading;
using System.Threading.Tasks;

namespace Synapse.Network;

/// <summary>P2P session for collaborative simulations (v2 foundation).</summary>
public interface ISimulationPeerSession : IAsyncDisposable
{
    string SessionId { get; }
    bool IsHost { get; }
    int PeerCount { get; }

    Task BroadcastScenePatchAsync(ReadOnlyMemory<byte> patch, CancellationToken ct = default);

    event Action<string, ReadOnlyMemory<byte>>? ScenePatchReceived;
}

public sealed class LocalSimulationPeerSession : ISimulationPeerSession
{
    public string SessionId { get; } = Guid.NewGuid().ToString("N");
    public bool IsHost => true;
    public int PeerCount => 1;

    public event Action<string, ReadOnlyMemory<byte>>? ScenePatchReceived;

    public Task BroadcastScenePatchAsync(ReadOnlyMemory<byte> patch, CancellationToken ct = default)
    {
        ScenePatchReceived?.Invoke("local", patch);
        return Task.CompletedTask;
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

public static class SimulationPeerHub
{
    public static ISimulationPeerSession CreateLocalSession() => new LocalSimulationPeerSession();
}
