using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Core.Maturity;
using Synapse.Infrastructure.Logging;

namespace Synapse.Network;

/// <summary>EXPERIMENTAL — P2P session contract for collaborative simulations (lab only).</summary>
[SynapseExperimental("Network.P2P", "Local/lab P2P surface; not a production collaborative network.")]
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

/// <summary>
/// EXPERIMENTAL — multi-peer TCP hub for lab collaborative sessions (v2.1+).
/// Suitable for localhost experiments; not a production mesh. See <c>docs/MATURITY.md</c>.
/// </summary>
[SynapseExperimental("Network.P2P", "TCP multi-peer hub for localhost/lab; not production WAN mesh.")]
public sealed class MultiPeerSimulationHub : IAsyncDisposable
{
    private readonly ISynapseLogger _logger;
    private readonly ConcurrentDictionary<string, PeerConnection> _peers = new();
    private TcpListener? _listener;
    private CancellationTokenSource? _cts;
    private Task? _acceptTask;

    public string SessionId { get; } = Guid.NewGuid().ToString("N");
    public bool IsHost { get; private set; }
    public int PeerCount => _peers.Count + 1;

    public int ListenPort => _listener is null ? 0 : ((IPEndPoint)_listener.LocalEndpoint).Port;

    public event Action<string, ReadOnlyMemory<byte>>? ScenePatchReceived;

    public MultiPeerSimulationHub(ISynapseLogger logger) => _logger = logger;

    public async Task StartHostAsync(int port = 0, bool publicBind = false, CancellationToken ct = default)
    {
        _listener = new TcpListener(publicBind ? IPAddress.Any : IPAddress.Loopback, port);
        _listener.Start();
        IsHost = true;
        _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        _acceptTask = AcceptLoopAsync(_cts.Token);
        _logger.Info("Network", $"P2P host listening on {((IPEndPoint)_listener.LocalEndpoint).Port}");
        await Task.CompletedTask;
    }

    public async Task ConnectAsync(string host, int port, CancellationToken ct = default)
    {
        var client = new TcpClient();
        await client.ConnectAsync(host, port, ct).ConfigureAwait(false);
        var peerId = Guid.NewGuid().ToString("N");
        var conn = new PeerConnection(peerId, client);
        _peers[peerId] = conn;
        _ = ReceiveLoopAsync(conn, ct);
        _logger.Info("Network", $"Connected to peer at {host}:{port}");
    }

    public async Task BroadcastScenePatchAsync(ReadOnlyMemory<byte> patch, CancellationToken ct = default)
    {
        ScenePatchReceived?.Invoke("self", patch);
        foreach (var peer in _peers.Values)
            await peer.SendAsync(patch, ct).ConfigureAwait(false);
    }

    private async Task AcceptLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested && _listener != null)
        {
            try
            {
                var client = await _listener.AcceptTcpClientAsync(ct).ConfigureAwait(false);
                var peerId = Guid.NewGuid().ToString("N");
                var conn = new PeerConnection(peerId, client);
                _peers[peerId] = conn;
                _ = ReceiveLoopAsync(conn, ct);
                _logger.Info("Network", $"Peer joined: {peerId} (total {_peers.Count + 1})");
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }
    }

    private async Task ReceiveLoopAsync(PeerConnection peer, CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested)
            {
                var payload = await peer.ReceiveAsync(ct).ConfigureAwait(false);
                if (payload.Length == 0)
                    break;
                ScenePatchReceived?.Invoke(peer.PeerId, payload);
            }
        }
        catch (Exception ex)
        {
            _logger.Warn("Network", $"Peer {peer.PeerId} disconnected: {ex.Message}");
        }
        finally
        {
            _peers.TryRemove(peer.PeerId, out _);
            peer.Dispose();
        }
    }

    public async ValueTask DisposeAsync()
    {
        _cts?.Cancel();
        if (_acceptTask != null)
        {
            try
            { await _acceptTask.ConfigureAwait(false); }
            catch (OperationCanceledException) { }
        }
        foreach (var peer in _peers.Values)
            peer.Dispose();
        _peers.Clear();
        _listener?.Stop();
    }

    private sealed class PeerConnection : IDisposable
    {
        private readonly NetworkStream _stream;
        private readonly SemaphoreSlim _sendLock = new(1, 1);

        public PeerConnection(string peerId, TcpClient client)
        {
            PeerId = peerId;
            _stream = client.GetStream();
        }

        public string PeerId { get; }

        public async Task SendAsync(ReadOnlyMemory<byte> payload, CancellationToken ct)
        {
            await _sendLock.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                var header = BitConverter.GetBytes(payload.Length);
                await _stream.WriteAsync(header, ct).ConfigureAwait(false);
                await _stream.WriteAsync(payload, ct).ConfigureAwait(false);
            }
            finally
            {
                _sendLock.Release();
            }
        }

        public async Task<byte[]> ReceiveAsync(CancellationToken ct)
        {
            var header = new byte[4];
            try
            {
                await _stream.ReadExactlyAsync(header, ct).ConfigureAwait(false);
            }
            catch (EndOfStreamException)
            {
                return Array.Empty<byte>();
            }

            int length = BitConverter.ToInt32(header, 0);
            if (length <= 0 || length > 4 * 1024 * 1024)
                throw new InvalidDataException($"Invalid P2P frame length: {length}");

            var buffer = new byte[length];
            await _stream.ReadExactlyAsync(buffer, ct).ConfigureAwait(false);
            return buffer;
        }

        public void Dispose() => _stream.Dispose();
    }
}

public static class SimulationPeerHub
{
    public static ISimulationPeerSession CreateLocalSession() => new LocalSimulationPeerSession();

    public static MultiPeerSimulationHub CreateMultiPeerHub(ISynapseLogger logger) =>
        new(logger);
}
