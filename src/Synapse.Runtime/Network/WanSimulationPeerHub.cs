using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Core.Maturity;
using Synapse.Infrastructure.Logging;

namespace Synapse.Network;

/// <summary>
/// EXPERIMENTAL — UDP rendezvous scaffold for WAN P2P (v2.2).
/// Binds and discovers peers on <see cref="IPAddress.Loopback"/> only; not real NAT traversal.
/// See <c>docs/MATURITY.md</c> (<c>Network.WAN</c>).
/// </summary>
[SynapseExperimental("Network.WAN", "UDP rendezvous is loopback-only; not production NAT traversal.")]
public sealed class NatTraversalCoordinator : IDisposable
{
    private static readonly ConcurrentDictionary<string, int> RegisteredTcpPorts = new(StringComparer.Ordinal);

    private readonly UdpClient _udp;
    private readonly ISynapseLogger _logger;
    private readonly string _sessionCode;
    private CancellationTokenSource? _cts;
    private UdpClient? _relay;
    private bool _ownsRelay;

    public NatTraversalCoordinator(ISynapseLogger logger, string sessionCode, int rendezvousPort = 7778)
    {
        _logger = logger;
        _sessionCode = sessionCode;
        _udp = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0));
        RendezvousPort = rendezvousPort;
    }

    /// <summary>Always true in v2.2: relay and discovery use loopback endpoints only.</summary>
    public bool IsLoopbackOnly => true;

    public int RendezvousPort { get; private set; }

    public async Task RegisterPublicEndpointAsync(int tcpPort, CancellationToken ct = default)
    {
        if (tcpPort <= 0 || tcpPort > 65535)
            throw new ArgumentOutOfRangeException(nameof(tcpPort));

        RegisteredTcpPorts[_sessionCode] = tcpPort;
        var payload = Encoding.UTF8.GetBytes($"REGISTER|{_sessionCode}|{tcpPort}");
        await _udp.SendAsync(payload, new IPEndPoint(IPAddress.Loopback, RendezvousPort), ct)
            .ConfigureAwait(false);
        _logger.Info("Network", $"NAT rendezvous registered session={_sessionCode} tcp={tcpPort}");
    }

    public async Task<IPEndPoint?> DiscoverPeerAsync(CancellationToken ct = default)
    {
        var payload = Encoding.UTF8.GetBytes($"DISCOVER|{_sessionCode}");
        await _udp.SendAsync(payload, new IPEndPoint(IPAddress.Loopback, RendezvousPort), ct)
            .ConfigureAwait(false);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(2));
        try
        {
            var result = await _udp.ReceiveAsync(timeout.Token).ConfigureAwait(false);
            var text = Encoding.UTF8.GetString(result.Buffer);
            if (!text.StartsWith("PEER|", StringComparison.Ordinal))
                return null;

            var parts = text.Split('|');
            if (parts.Length >= 3 &&
                int.TryParse(parts[2], out var port) &&
                port > 0 &&
                port <= 65535)
            {
                return new IPEndPoint(IPAddress.Loopback, port);
            }
        }
        catch (OperationCanceledException)
        {
            return null;
        }

        return null;
    }

    private NatTraversalCoordinator(
        ISynapseLogger logger,
        string sessionCode,
        int rendezvousPort,
        UdpClient relay,
        CancellationTokenSource cts)
        : this(logger, sessionCode, rendezvousPort)
    {
        _relay = relay;
        _ownsRelay = true;
        _cts = cts;
    }

    public static NatTraversalCoordinator StartRelay(
        ISynapseLogger logger,
        string sessionCode,
        int rendezvousPort = 0)
    {
        var relay = new UdpClient(new IPEndPoint(IPAddress.Loopback, rendezvousPort));
        int boundPort = ((IPEndPoint)relay.Client.LocalEndPoint!).Port;
        var cts = new CancellationTokenSource();
        var coord = new NatTraversalCoordinator(logger, sessionCode, boundPort, relay, cts);
        var token = cts.Token;
        _ = Task.Run(async () =>
        {
            while (!token.IsCancellationRequested)
            {
                try
                {
                    var result = await relay.ReceiveAsync(token).ConfigureAwait(false);
                    var text = Encoding.UTF8.GetString(result.Buffer);
                    if (text.StartsWith("REGISTER|", StringComparison.Ordinal))
                    {
                        var parts = text.Split('|');
                        if (parts.Length >= 3 &&
                            parts[1] == sessionCode &&
                            int.TryParse(parts[2], out var tcpPort) &&
                            tcpPort > 0)
                        {
                            RegisteredTcpPorts[sessionCode] = tcpPort;
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{tcpPort}");
                            await relay.SendAsync(reply, result.RemoteEndPoint, token).ConfigureAwait(false);
                        }
                    }
                    else if (text.StartsWith("DISCOVER|", StringComparison.Ordinal))
                    {
                        var parts = text.Split('|');
                        if (parts.Length >= 2 && parts[1] == sessionCode)
                        {
                            int port = RegisteredTcpPorts.TryGetValue(sessionCode, out var registered)
                                ? registered
                                : 0;
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{port}");
                            await relay.SendAsync(reply, result.RemoteEndPoint, token).ConfigureAwait(false);
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
            }
        }, token);

        return coord;
    }

    public void Dispose()
    {
        _cts?.Cancel();
        if (_ownsRelay)
            _relay?.Dispose();
        _udp.Dispose();
        RegisteredTcpPorts.TryRemove(_sessionCode, out _);
    }
}

/// <summary>
/// EXPERIMENTAL — encrypted P2P hub with loopback NAT scaffold + AES-GCM (v2.2).
/// Encryption is real; WAN/NAT claims are not. See <c>docs/MATURITY.md</c>.
/// </summary>
[SynapseExperimental("Network.WAN", "AES-GCM ok; NAT rendezvous is loopback lab scaffolding.")]
public sealed class WanSimulationPeerHub : IAsyncDisposable
{
    private readonly ISynapseLogger _logger;
    private readonly string _sessionCode;
    private readonly PeerEncryption _encryption;
    private readonly NatTraversalCoordinator _nat;
    private readonly MultiPeerSimulationHub _inner;

    public WanSimulationPeerHub(ISynapseLogger logger, string sessionCode, int rendezvousPort = 0)
    {
        _sessionCode = sessionCode;
        _logger = logger;
        _encryption = PeerEncryption.FromSessionCode(sessionCode);
        _nat = NatTraversalCoordinator.StartRelay(logger, sessionCode, rendezvousPort);
        _inner = new MultiPeerSimulationHub(logger);
    }

    public event Action<string, ReadOnlyMemory<byte>>? ScenePatchReceived
    {
        add => _inner.ScenePatchReceived += value;
        remove => _inner.ScenePatchReceived -= value;
    }

    public int ListenPort => _inner.ListenPort;
    public int RendezvousPort => _nat.RendezvousPort;

    public async Task StartHostAsync(int port, CancellationToken ct = default)
    {
        await _inner.StartHostAsync(port, publicBind: true, ct).ConfigureAwait(false);
        await _nat.RegisterPublicEndpointAsync(_inner.ListenPort, ct).ConfigureAwait(false);
        _logger.Info("Network", $"WAN encrypted host on port {_inner.ListenPort} (rendezvous={_nat.RendezvousPort})");
    }

    public async Task JoinAsync(CancellationToken ct = default)
    {
        using var clientNat = new NatTraversalCoordinator(_logger, _sessionCode, _nat.RendezvousPort);
        var endpoint = await clientNat.DiscoverPeerAsync(ct).ConfigureAwait(false);
        if (endpoint == null || endpoint.Port == 0)
            throw new InvalidOperationException(
                "Peer discovery failed. Ensure the host registered before join, and rendezvous ports match.");
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
