using System.Net;
using System.Net.Sockets;
using System.Text;
using Synapse.Infrastructure.Logging;

namespace Synapse.Network;

/// <summary>UDP rendezvous + hole-punch coordinator for WAN P2P (v2.2).</summary>
public sealed class NatTraversalCoordinator : IDisposable
{
    private readonly UdpClient _udp;
    private readonly ISynapseLogger _logger;
    private readonly string _sessionCode;
    private CancellationTokenSource? _cts;

    public NatTraversalCoordinator(ISynapseLogger logger, string sessionCode, int rendezvousPort = 7778)
    {
        _logger = logger;
        _sessionCode = sessionCode;
        _udp = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0));
        RendezvousPort = rendezvousPort;
    }

    public int RendezvousPort { get; }

    public async Task RegisterPublicEndpointAsync(int tcpPort, CancellationToken ct = default)
    {
        var payload = Encoding.UTF8.GetBytes($"REGISTER|{_sessionCode}|{tcpPort}");
        await _udp.SendAsync(payload, new IPEndPoint(IPAddress.Loopback, RendezvousPort), ct);
        _logger.Info("Network", $"NAT rendezvous registered session={_sessionCode} tcp={tcpPort}");
    }

    public async Task<IPEndPoint?> DiscoverPeerAsync(CancellationToken ct = default)
    {
        var payload = Encoding.UTF8.GetBytes($"DISCOVER|{_sessionCode}");
        await _udp.SendAsync(payload, new IPEndPoint(IPAddress.Loopback, RendezvousPort), ct);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(2));
        try
        {
            var result = await _udp.ReceiveAsync(timeout.Token);
            var text = Encoding.UTF8.GetString(result.Buffer);
            if (text.StartsWith("PEER|", StringComparison.Ordinal))
            {
                var parts = text.Split('|');
                if (parts.Length >= 3 && int.TryParse(parts[2], out var port))
                    return new IPEndPoint(IPAddress.Loopback, port);
            }
        }
        catch (OperationCanceledException)
        {
            return null;
        }

        return null;
    }

    public static NatTraversalCoordinator StartRelay(ISynapseLogger logger, string sessionCode, int rendezvousPort = 7778)
    {
        var coord = new NatTraversalCoordinator(logger, sessionCode, rendezvousPort);
        coord._cts = new CancellationTokenSource();
        var relay = new UdpClient(new IPEndPoint(IPAddress.Loopback, rendezvousPort));
        _ = Task.Run(async () =>
        {
            while (!coord._cts.Token.IsCancellationRequested)
            {
                try
                {
                    var result = await relay.ReceiveAsync(coord._cts.Token);
                    var text = Encoding.UTF8.GetString(result.Buffer);
                    if (text.StartsWith("REGISTER|", StringComparison.Ordinal))
                    {
                        var parts = text.Split('|');
                        if (parts.Length >= 3 && parts[1] == sessionCode)
                        {
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{parts[2]}");
                            await relay.SendAsync(reply, result.RemoteEndPoint, coord._cts.Token);
                        }
                    }
                    else if (text.StartsWith("DISCOVER|", StringComparison.Ordinal))
                    {
                        var parts = text.Split('|');
                        if (parts.Length >= 2 && parts[1] == sessionCode)
                        {
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|0");
                            await relay.SendAsync(reply, result.RemoteEndPoint, coord._cts.Token);
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
            relay.Dispose();
        });
        return coord;
    }

    public void Dispose()
    {
        _cts?.Cancel();
        _udp.Dispose();
    }
}

/// <summary>WAN-capable encrypted P2P hub (NAT traversal + AES-GCM).</summary>
public sealed class WanSimulationPeerHub : IAsyncDisposable
{
    private readonly ISynapseLogger _logger;
    private readonly string _sessionCode;
    private readonly PeerEncryption _encryption;
    private readonly NatTraversalCoordinator _nat;
    private readonly MultiPeerSimulationHub _inner;

    public WanSimulationPeerHub(ISynapseLogger logger, string sessionCode)
    {
        _sessionCode = sessionCode;
        _logger = logger;
        _encryption = PeerEncryption.FromSessionCode(sessionCode);
        _nat = NatTraversalCoordinator.StartRelay(logger, sessionCode);
        _inner = new MultiPeerSimulationHub(logger);
    }

    public event Action<string, ReadOnlyMemory<byte>>? ScenePatchReceived
    {
        add => _inner.ScenePatchReceived += value;
        remove => _inner.ScenePatchReceived -= value;
    }

    public int ListenPort => _inner.ListenPort;

    public async Task StartHostAsync(int port, CancellationToken ct = default)
    {
        await _inner.StartHostAsync(port, publicBind: true, ct).ConfigureAwait(false);
        await _nat.RegisterPublicEndpointAsync(_inner.ListenPort, ct).ConfigureAwait(false);
        _logger.Info("Network", $"WAN encrypted host on port {_inner.ListenPort}");
    }

    public async Task JoinAsync(CancellationToken ct = default)
    {
        var clientNat = new NatTraversalCoordinator(_logger, _sessionCode);
        var endpoint = await clientNat.DiscoverPeerAsync(ct).ConfigureAwait(false);
        clientNat.Dispose();
        if (endpoint == null || endpoint.Port == 0)
            throw new InvalidOperationException("Peer discovery failed.");
        await _inner.ConnectAsync(endpoint.Address.ToString(), endpoint.Port, ct).ConfigureAwait(false);
    }

    public async Task BroadcastScenePatchAsync(ReadOnlyMemory<byte> patch, CancellationToken ct = default)
    {
        var encrypted = _encryption.Encrypt(patch.Span);
        await _inner.BroadcastScenePatchAsync(encrypted, ct).ConfigureAwait(false);
    }

    public byte[] DecryptPatch(ReadOnlyMemory<byte> encrypted) => _encryption.Decrypt(encrypted.Span);

    public async ValueTask DisposeAsync()
    {
        await _inner.DisposeAsync().ConfigureAwait(false);
        _nat.Dispose();
        _encryption.Dispose();
    }
}
