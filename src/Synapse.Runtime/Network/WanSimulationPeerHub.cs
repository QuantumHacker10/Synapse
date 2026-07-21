using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Network;

/// <summary>
/// Registered peer endpoint exchanged through the UDP rendezvous.
/// </summary>
public sealed record NatPeerEndpoint(
    string SessionCode,
    IPAddress Address,
    int TcpPort,
    int UdpPort,
    bool IsMapped);

/// <summary>
/// NAT traversal coordinator: STUN mapped-address discovery, UDP rendezvous registration,
/// and bidirectional UDP hole-punching. Falls back to loopback-only when STUN is unreachable
/// (offline CI). Bridged by <see cref="Synapse.Runtime.EngineHost"/>.
/// </summary>
public sealed class NatTraversalCoordinator : IDisposable
{
    private static readonly ConcurrentDictionary<string, NatPeerEndpoint> RegisteredPeers = new(StringComparer.Ordinal);

    private readonly UdpClient _udp;
    private readonly ISynapseLogger _logger;
    private readonly string _sessionCode;
    private readonly StunClient _stun;
    private readonly IPAddress _rendezvousHost;
    private CancellationTokenSource? _cts;
    private UdpClient? _relay;
    private bool _ownsRelay;
    private bool _disposed;

    public NatTraversalCoordinator(
        ISynapseLogger logger,
        string sessionCode,
        int rendezvousPort = 7778,
        IPAddress? rendezvousHost = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _sessionCode = sessionCode ?? throw new ArgumentNullException(nameof(sessionCode));
        _stun = new StunClient(logger);
        _rendezvousHost = rendezvousHost ?? IPAddress.Loopback;
        _udp = new UdpClient(new IPEndPoint(IPAddress.Any, 0));
        try
        { _udp.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true); }
        catch { /* best-effort */ }
        RendezvousPort = rendezvousPort;
        LocalUdpPort = ((IPEndPoint)_udp.Client.LocalEndPoint!).Port;
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

    /// <summary>True until a non-loopback STUN mapping is confirmed.</summary>
    public bool IsLoopbackOnly { get; private set; } = true;

    public int RendezvousPort { get; private set; }
    public int LocalUdpPort { get; }
    public IPEndPoint? MappedEndpoint { get; private set; }

    /// <summary>
    /// Discovers the public mapped UDP endpoint via STUN. On failure, uses the local Any-bound port
    /// advertised as loopback for same-host lab scenarios.
    /// </summary>
    public async Task<IPEndPoint> DiscoverPublicEndpointAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var mapped = await _stun.DiscoverMappedAddressAsync(_udp, timeout: null, ct).ConfigureAwait(false);
        if (mapped != null)
        {
            MappedEndpoint = mapped;
            IsLoopbackOnly = mapped.Address.Equals(IPAddress.Loopback) || IPAddress.IsLoopback(mapped.Address);
            return mapped;
        }

        // Offline / blocked STUN: fall back to loopback advertisement for same-host tests.
        IsLoopbackOnly = true;
        MappedEndpoint = new IPEndPoint(IPAddress.Loopback, LocalUdpPort);
        _logger.Warn("Network", "STUN unavailable — falling back to loopback NAT advertisement");
        return MappedEndpoint;
    }

    public async Task RegisterPublicEndpointAsync(int tcpPort, CancellationToken ct = default)
    {
        if (tcpPort is <= 0 or > 65535)
            throw new ArgumentOutOfRangeException(nameof(tcpPort));

        var mapped = MappedEndpoint ?? await DiscoverPublicEndpointAsync(ct).ConfigureAwait(false);
        var peer = new NatPeerEndpoint(_sessionCode, mapped.Address, tcpPort, mapped.Port, !IsLoopbackOnly);
        RegisteredPeers[_sessionCode] = peer;

        var payload = Encoding.UTF8.GetBytes(
            $"REGISTER|{_sessionCode}|{peer.Address}|{peer.TcpPort}|{peer.UdpPort}|{(peer.IsMapped ? 1 : 0)}");
        await _udp.SendAsync(payload, new IPEndPoint(_rendezvousHost, RendezvousPort), ct)
            .ConfigureAwait(false);
        _logger.Info("Network",
            $"NAT rendezvous registered session={_sessionCode} tcp={tcpPort} mapped={peer.Address}:{peer.UdpPort} loopback={IsLoopbackOnly}");
    }

    public async Task<NatPeerEndpoint?> DiscoverPeerAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var payload = Encoding.UTF8.GetBytes($"DISCOVER|{_sessionCode}");
        await _udp.SendAsync(payload, new IPEndPoint(_rendezvousHost, RendezvousPort), ct)
            .ConfigureAwait(false);

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(3));
        try
        {
            while (!timeout.IsCancellationRequested)
            {
                var result = await _udp.ReceiveAsync(timeout.Token).ConfigureAwait(false);
                var text = Encoding.UTF8.GetString(result.Buffer);
                if (TryParsePeerMessage(text, out var peer) && peer!.SessionCode == _sessionCode)
                    return peer;
            }
        }
        catch (OperationCanceledException)
        {
            // Fall through to in-process registry (same-process host/join).
        }

        return RegisteredPeers.TryGetValue(_sessionCode, out var local) ? local : null;
    }

    /// <summary>
    /// Sends UDP hole-punch packets toward the peer's mapped UDP endpoint to open NAT bindings.
    /// </summary>
    public async Task HolePunchAsync(NatPeerEndpoint peer, int bursts = 8, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(peer);
        var target = new IPEndPoint(peer.Address, peer.UdpPort > 0 ? peer.UdpPort : peer.TcpPort);
        var punch = Encoding.UTF8.GetBytes($"PUNCH|{_sessionCode}|{LocalUdpPort}");
        for (int i = 0; i < Math.Max(1, bursts); i++)
        {
            ct.ThrowIfCancellationRequested();
            await _udp.SendAsync(punch, target, ct).ConfigureAwait(false);
            await Task.Delay(40, ct).ConfigureAwait(false);
        }

        _logger.Info("Network", $"UDP hole-punch sent to {target} ({bursts} bursts)");
    }

    public static NatTraversalCoordinator StartRelay(
        ISynapseLogger logger,
        string sessionCode,
        int rendezvousPort = 0)
    {
        var relay = new UdpClient(new IPEndPoint(IPAddress.Any, rendezvousPort));
        try
        { relay.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true); }
        catch { /* best-effort */ }
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
                        if (TryParseRegisterMessage(text, out var peer) && peer!.SessionCode == sessionCode)
                        {
                            RegisteredPeers[sessionCode] = peer;
                            var reply = Encoding.UTF8.GetBytes(FormatPeerMessage(peer));
                            await relay.SendAsync(reply, result.RemoteEndPoint, token).ConfigureAwait(false);
                        }
                    }
                    else if (text.StartsWith("DISCOVER|", StringComparison.Ordinal))
                    {
                        var parts = text.Split('|');
                        if (parts.Length >= 2 && parts[1] == sessionCode)
                        {
                            if (RegisteredPeers.TryGetValue(sessionCode, out var peer))
                            {
                                var reply = Encoding.UTF8.GetBytes(FormatPeerMessage(peer));
                                await relay.SendAsync(reply, result.RemoteEndPoint, token).ConfigureAwait(false);
                            }
                            else
                            {
                                var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|127.0.0.1|0|0|0");
                                await relay.SendAsync(reply, result.RemoteEndPoint, token).ConfigureAwait(false);
                            }
                        }
                    }
                    else if (text.StartsWith("PUNCH|", StringComparison.Ordinal))
                    {
                        // Reflect punch to keep bindings warm when relay is co-located.
                        var ack = Encoding.UTF8.GetBytes($"PUNCH-ACK|{sessionCode}");
                        await relay.SendAsync(ack, result.RemoteEndPoint, token).ConfigureAwait(false);
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
                catch (Exception)
                {
                    // Keep relay alive across transient errors.
                }
            }
        }, token);

        return coord;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _cts?.Cancel();
        if (_ownsRelay)
            _relay?.Dispose();
        _udp.Dispose();
        RegisteredPeers.TryRemove(_sessionCode, out _);
    }

    private static string FormatPeerMessage(NatPeerEndpoint peer)
        => $"PEER|{peer.SessionCode}|{peer.Address}|{peer.TcpPort}|{peer.UdpPort}|{(peer.IsMapped ? 1 : 0)}";

    private static bool TryParsePeerMessage(string text, out NatPeerEndpoint? peer)
    {
        peer = null;
        if (!text.StartsWith("PEER|", StringComparison.Ordinal))
            return false;
        var parts = text.Split('|');
        if (parts.Length < 5)
            return false;
        if (!IPAddress.TryParse(parts[2], out var address))
            return false;
        if (!int.TryParse(parts[3], out var tcpPort) || tcpPort < 0 || tcpPort > 65535)
            return false;
        if (!int.TryParse(parts[4], out var udpPort) || udpPort < 0 || udpPort > 65535)
            return false;
        bool mapped = parts.Length >= 6 && parts[5] == "1";
        peer = new NatPeerEndpoint(parts[1], address, tcpPort, udpPort, mapped);
        return true;
    }

    private static bool TryParseRegisterMessage(string text, out NatPeerEndpoint? peer)
    {
        peer = null;
        if (!text.StartsWith("REGISTER|", StringComparison.Ordinal))
            return false;
        var parts = text.Split('|');
        // New format: REGISTER|code|ip|tcp|udp|mapped
        if (parts.Length >= 5 && IPAddress.TryParse(parts[2], out var address))
        {
            if (!int.TryParse(parts[3], out var tcpPort) || tcpPort <= 0 || tcpPort > 65535)
                return false;
            int udpPort = 0;
            if (parts.Length >= 5 && !int.TryParse(parts[4], out udpPort))
                udpPort = 0;
            bool mapped = parts.Length >= 6 && parts[5] == "1";
            peer = new NatPeerEndpoint(parts[1], address, tcpPort, udpPort, mapped);
            return true;
        }

        // Legacy format: REGISTER|code|tcpPort
        if (parts.Length >= 3 && int.TryParse(parts[2], out var legacyTcp) && legacyTcp > 0)
        {
            peer = new NatPeerEndpoint(parts[1], IPAddress.Loopback, legacyTcp, 0, false);
            return true;
        }

        return false;
    }
}

/// <summary>
/// Encrypted P2P hub with STUN-backed NAT traversal + AES-GCM.
/// Owned by <see cref="Synapse.Runtime.EngineHost"/> for Studio/CLI collaboration.
/// </summary>
public sealed class WanSimulationPeerHub : IAsyncDisposable
{
    private readonly ISynapseLogger _logger;
    private readonly string _sessionCode;
    private readonly PeerEncryption _encryption;
    private readonly NatTraversalCoordinator _nat;
    private readonly MultiPeerSimulationHub _inner;

    public WanSimulationPeerHub(ISynapseLogger logger, string sessionCode, int rendezvousPort = 0)
    {
        _sessionCode = sessionCode ?? throw new ArgumentNullException(nameof(sessionCode));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
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
    public bool IsLoopbackOnly => _nat.IsLoopbackOnly;
    public IPEndPoint? MappedEndpoint => _nat.MappedEndpoint;

    public async Task StartHostAsync(int port, CancellationToken ct = default)
    {
        await _inner.StartHostAsync(port, publicBind: true, ct).ConfigureAwait(false);
        await _nat.DiscoverPublicEndpointAsync(ct).ConfigureAwait(false);
        await _nat.RegisterPublicEndpointAsync(_inner.ListenPort, ct).ConfigureAwait(false);
        _logger.Info("Network",
            $"WAN encrypted host on port {_inner.ListenPort} (rendezvous={_nat.RendezvousPort}, mapped={_nat.MappedEndpoint}, loopback={_nat.IsLoopbackOnly})");
    }

    public async Task JoinAsync(int? remoteRendezvousPort = null, CancellationToken ct = default)
    {
        int port = remoteRendezvousPort is > 0 ? remoteRendezvousPort.Value : _nat.RendezvousPort;
        using var clientNat = new NatTraversalCoordinator(_logger, _sessionCode, port);
        await clientNat.DiscoverPublicEndpointAsync(ct).ConfigureAwait(false);
        var peer = await clientNat.DiscoverPeerAsync(ct).ConfigureAwait(false);
        if (peer == null || peer.TcpPort == 0)
            throw new InvalidOperationException(
                "Peer discovery failed. Ensure the host registered before join, and rendezvous ports match.");

        await clientNat.HolePunchAsync(peer, ct: ct).ConfigureAwait(false);

        var connectHost = peer.Address.Equals(IPAddress.Any) || peer.Address.Equals(IPAddress.IPv6Any)
            ? IPAddress.Loopback.ToString()
            : peer.Address.ToString();

        await _inner.ConnectAsync(connectHost, peer.TcpPort, ct).ConfigureAwait(false);
        _logger.Info("Network", $"WAN join connected to {connectHost}:{peer.TcpPort} (rdv={port})");
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
