using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using Synapse.Infrastructure.Logging;

namespace Synapse.Network;

/// <summary>Discovered peer endpoint plus transport mode (tcp|stun|turn).</summary>
public readonly record struct PeerCandidate(IPEndPoint Endpoint, string Mode);

/// <summary>Registered peer endpoint exchanged through the UDP rendezvous.</summary>
public sealed record NatPeerEndpoint(
    string SessionCode,
    IPAddress Address,
    int TcpPort,
    int UdpPort,
    bool IsMapped);

/// <summary>
/// UDP rendezvous + hole-punch coordinator for WAN/LAN P2P.
/// Binds <see cref="IPAddress.Any"/> so peers across NAT (with port-forward / same LAN) can discover.
/// Optional STUN-advertised IPs improve reflexive addressing; TURN mode registers relayed candidates.
/// </summary>
public sealed class NatTraversalCoordinator : IDisposable
{
    private const int MaxUdpPayload = 512;
    private readonly UdpClient _udp;
    private readonly ISynapseLogger _logger;
    private readonly string _sessionCode;
    private readonly StunClient _stun;
    private readonly IPAddress _rendezvousAddress;
    private CancellationTokenSource? _cts;
    private UdpClient? _relay;
    private bool _ownsRelay;
    private bool _disposed;

    private static readonly ConcurrentDictionary<string, NatPeerEndpoint> RegisteredPeers =
        new(StringComparer.Ordinal);

    private static readonly ConcurrentDictionary<string, (IPAddress Address, int TcpPort, string Mode, DateTime Expiry)> Registrations =
        new(StringComparer.Ordinal);

    public NatTraversalCoordinator(
        ISynapseLogger logger,
        string sessionCode,
        int rendezvousPort = 7778,
        IPAddress? rendezvousAddress = null)
    {
        if (string.IsNullOrWhiteSpace(sessionCode) || sessionCode.Length < 4)
            throw new ArgumentException("Session code must be at least 4 characters.", nameof(sessionCode));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _sessionCode = sessionCode;
        _stun = new StunClient(logger);
        _rendezvousAddress = rendezvousAddress ?? IPAddress.Loopback;
        RendezvousPort = rendezvousPort;
        _udp = new UdpClient(new IPEndPoint(IPAddress.Any, 0));
        try
        { _udp.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true); }
        catch { /* best-effort */ }
        LocalUdpPort = ((IPEndPoint)_udp.Client.LocalEndPoint!).Port;
    }

    private NatTraversalCoordinator(
        ISynapseLogger logger,
        string sessionCode,
        int rendezvousPort,
        UdpClient relay,
        CancellationTokenSource cts,
        IPAddress rendezvousAddress)
        : this(logger, sessionCode, rendezvousPort, rendezvousAddress)
    {
        _relay = relay;
        _ownsRelay = true;
        _cts = cts;
    }

    public bool IsLoopbackOnly { get; private set; } = true;
    public int RendezvousPort { get; }
    public int LocalUdpPort { get; }
    public IPAddress RendezvousAddress => _rendezvousAddress;
    public IPEndPoint? MappedEndpoint { get; private set; }

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

        IsLoopbackOnly = true;
        MappedEndpoint = new IPEndPoint(IPAddress.Loopback, LocalUdpPort);
        _logger.Warn("Network", "STUN unavailable — falling back to loopback NAT advertisement");
        return MappedEndpoint;
    }

    public async Task RegisterPublicEndpointAsync(
        int tcpPort,
        IPAddress? advertisedAddress = null,
        string mode = "tcp",
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tcpPort is <= 0 or > 65535)
            throw new ArgumentOutOfRangeException(nameof(tcpPort));

        var mapped = MappedEndpoint ?? await DiscoverPublicEndpointAsync(ct).ConfigureAwait(false);
        var adv = advertisedAddress ?? mapped.Address;
        var peer = new NatPeerEndpoint(_sessionCode, adv, tcpPort, mapped.Port, !IsLoopbackOnly);
        RegisteredPeers[RegistryKey(_sessionCode, RendezvousPort)] = peer;
        Registrations[RegistryKey(_sessionCode, RendezvousPort)] = (adv, tcpPort, mode, DateTime.UtcNow.AddMinutes(5));

        var payload = Encoding.UTF8.GetBytes(
            $"REGISTER|{_sessionCode}|{tcpPort}|{ComputeMac(_sessionCode, tcpPort)}|{adv}|{mode}");
        if (payload.Length > MaxUdpPayload)
            throw new InvalidOperationException("REGISTER payload too large.");
        await _udp.SendAsync(payload, new IPEndPoint(_rendezvousAddress, RendezvousPort), ct).ConfigureAwait(false);

        _logger.Info("Network",
            $"NAT rendezvous REGISTER session={_sessionCode} tcp={tcpPort} adv={adv} mode={mode} mapped={peer.Address}:{peer.UdpPort} via {_rendezvousAddress}:{RendezvousPort}");
    }

    public async Task<NatPeerEndpoint?> DiscoverPeerAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var payload = Encoding.UTF8.GetBytes($"DISCOVER|{_sessionCode}|{ComputeMac(_sessionCode, 0)}");
        await _udp.SendAsync(payload, new IPEndPoint(_rendezvousAddress, RendezvousPort), ct).ConfigureAwait(false);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(3));
        try
        {
            while (!timeout.IsCancellationRequested)
            {
                var result = await _udp.ReceiveAsync(timeout.Token).ConfigureAwait(false);
                if (result.Buffer.Length > MaxUdpPayload)
                    continue;
                var text = Encoding.UTF8.GetString(result.Buffer);
                if (TryParsePeerMessage(text, out var peer) && peer!.SessionCode == _sessionCode)
                    return peer;
                var candidate = ParsePeerCandidate(text, _sessionCode);
                if (candidate != null)
                {
                    return new NatPeerEndpoint(
                        _sessionCode,
                        candidate.Value.Endpoint.Address,
                        candidate.Value.Endpoint.Port,
                        candidate.Value.Endpoint.Port,
                        candidate.Value.Mode != "tcp");
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Fall through to in-process registry.
        }

        if (RegisteredPeers.TryGetValue(RegistryKey(_sessionCode, RendezvousPort), out var local))
            return local;
        if (Registrations.TryGetValue(RegistryKey(_sessionCode, RendezvousPort), out var reg) && reg.Expiry > DateTime.UtcNow)
            return new NatPeerEndpoint(_sessionCode, reg.Address, reg.TcpPort, 0, reg.Mode != "tcp");
        return null;
    }

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
        int rendezvousPort = 7778,
        IPAddress? bindAddress = null)
    {
        ArgumentNullException.ThrowIfNull(logger);
        if (string.IsNullOrWhiteSpace(sessionCode) || sessionCode.Length < 4)
            throw new ArgumentException("Session code must be at least 4 characters.", nameof(sessionCode));

        var bind = bindAddress ?? IPAddress.Any;
        UdpClient relay;
        int boundPort;
        if (rendezvousPort == 0)
        {
            relay = new UdpClient(new IPEndPoint(bind, 0));
            try
            { relay.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true); }
            catch { /* best-effort */ }
            boundPort = ((IPEndPoint)relay.Client.LocalEndPoint!).Port;
        }
        else
        {
            relay = new UdpClient(new IPEndPoint(bind, rendezvousPort));
            boundPort = rendezvousPort;
        }

        var cts = new CancellationTokenSource();
        var coord = new NatTraversalCoordinator(
            logger,
            sessionCode,
            boundPort,
            relay,
            cts,
            PreferLoopbackHint(bind));
        var token = cts.Token;
        _ = Task.Run(async () =>
        {
            while (!token.IsCancellationRequested)
            {
                try
                {
                    var result = await relay.ReceiveAsync(token).ConfigureAwait(false);
                    if (result.Buffer.Length == 0 || result.Buffer.Length > MaxUdpPayload)
                        continue;
                    var text = Encoding.UTF8.GetString(result.Buffer);
                    var parts = text.Split('|');
                    if (parts.Length < 2)
                        continue;

                    if (text.StartsWith("REGISTER|", StringComparison.Ordinal))
                    {
                        if (TryParseRegisterMessage(text, out var regPeer) && regPeer!.SessionCode == sessionCode)
                        {
                            RegisteredPeers[RegistryKey(sessionCode, boundPort)] = regPeer;
                            var reply = Encoding.UTF8.GetBytes(FormatPeerMessage(regPeer));
                            await relay.SendAsync(reply, result.RemoteEndPoint, token).ConfigureAwait(false);
                            continue;
                        }

                        if (parts.Length >= 4 && parts[1] == sessionCode &&
                            int.TryParse(parts[2], out var tcpPort) && tcpPort > 0 &&
                            string.Equals(parts[3], ComputeMac(sessionCode, tcpPort), StringComparison.Ordinal))
                        {
                            var peerIp = result.RemoteEndPoint.Address;
                            if (peerIp.IsIPv4MappedToIPv6)
                                peerIp = peerIp.MapToIPv4();
                            string mode = "tcp";
                            if (parts.Length >= 5 && IPAddress.TryParse(parts[4], out var adv) &&
                                !adv.Equals(IPAddress.Any) && !adv.Equals(IPAddress.None))
                                peerIp = adv;
                            if (parts.Length >= 6 && !string.IsNullOrWhiteSpace(parts[5]))
                                mode = parts[5].Trim().ToLowerInvariant();

                            Registrations[RegistryKey(sessionCode, boundPort)] = (peerIp, tcpPort, mode, DateTime.UtcNow.AddMinutes(5));
                            RegisteredPeers[RegistryKey(sessionCode, boundPort)] = new NatPeerEndpoint(sessionCode, peerIp, tcpPort, 0, mode != "tcp");
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{peerIp}|{tcpPort}|{mode}");
                            await relay.SendAsync(reply, result.RemoteEndPoint, token).ConfigureAwait(false);
                        }
                    }
                    else if (text.StartsWith("DISCOVER|", StringComparison.Ordinal) && parts.Length >= 3)
                    {
                        if (parts[1] != sessionCode)
                            continue;
                        if (!string.Equals(parts[2], ComputeMac(sessionCode, 0), StringComparison.Ordinal) &&
                            parts[2] != sessionCode)
                            continue;

                        if (RegisteredPeers.TryGetValue(RegistryKey(sessionCode, boundPort), out var peer))
                        {
                            var reply = Encoding.UTF8.GetBytes(FormatPeerMessage(peer));
                            await relay.SendAsync(reply, result.RemoteEndPoint, token).ConfigureAwait(false);
                        }
                        else if (Registrations.TryGetValue(RegistryKey(sessionCode, boundPort), out var reg) && reg.Expiry > DateTime.UtcNow)
                        {
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{reg.Address}|{reg.TcpPort}|{reg.Mode}");
                            await relay.SendAsync(reply, result.RemoteEndPoint, token).ConfigureAwait(false);
                        }
                        else
                        {
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|127.0.0.1|0|0|0");
                            await relay.SendAsync(reply, result.RemoteEndPoint, token).ConfigureAwait(false);
                        }
                    }
                    else if (text.StartsWith("PUNCH|", StringComparison.Ordinal))
                    {
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

            try
            { relay.Dispose(); }
            catch { /* ignore */ }
        }, token);

        logger.Info("Network", $"NAT rendezvous relay on {bind}:{boundPort}");
        return coord;
    }

    public static PeerCandidate? ParsePeerCandidate(string text, string sessionCode)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(sessionCode);
        if (!text.StartsWith("PEER|", StringComparison.Ordinal))
            return null;
        var parts = text.Split('|');
        if (parts.Length >= 4 && parts[1] == sessionCode)
        {
            if (!IPAddress.TryParse(parts[2], out var ip))
                return null;
            if (!int.TryParse(parts[3], out var port) || port <= 0)
                return null;
            if (ip.Equals(IPAddress.Any) || ip.Equals(IPAddress.IPv6Any))
                return null;
            var mode = parts.Length >= 5 && !string.IsNullOrWhiteSpace(parts[4]) ? parts[4].Trim() : "tcp";
            return new PeerCandidate(new IPEndPoint(ip, port), mode);
        }

        if (parts.Length >= 3 && parts[1] == sessionCode && int.TryParse(parts[2], out var legacyPort) && legacyPort > 0)
            return new PeerCandidate(new IPEndPoint(IPAddress.Loopback, legacyPort), "tcp");

        return null;
    }

    public static IPEndPoint? ParsePeerReply(string text, string sessionCode) =>
        ParsePeerCandidate(text, sessionCode)?.Endpoint;

    private static IPAddress PreferLoopbackHint(IPAddress bind) =>
        bind.Equals(IPAddress.Any) || bind.Equals(IPAddress.IPv6Any) ? IPAddress.Loopback : bind;

    public static string ComputeMac(string sessionCode, int tcpPort)
    {
        var raw = Encoding.UTF8.GetBytes($"{sessionCode}|{tcpPort}|Synapse.NAT.v1");
        var hash = SHA256.HashData(raw);
        return Convert.ToHexString(hash.AsSpan(0, 8));
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
        RegisteredPeers.TryRemove(RegistryKey(_sessionCode, RendezvousPort), out _);
    }

    private static string RegistryKey(string sessionCode, int rendezvousPort) =>
        $"{sessionCode}@{rendezvousPort}";

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

        if (parts.Length >= 3 && int.TryParse(parts[2], out var legacyTcp) && legacyTcp > 0)
        {
            peer = new NatPeerEndpoint(parts[1], IPAddress.Loopback, legacyTcp, 0, false);
            return true;
        }

        return false;
    }
}

/// <summary>WAN-capable encrypted P2P hub (NAT traversal on Any + STUN/TURN + AES-GCM + auth).</summary>
public sealed class WanSimulationPeerHub : IAsyncDisposable
{
    private readonly ISynapseLogger _logger;
    private readonly string _sessionCode;
    private readonly PeerEncryption _encryption;
    private readonly NatTraversalCoordinator _nat;
    private readonly MultiPeerSimulationHub _inner;
    private readonly Action<string, ReadOnlyMemory<byte>> _decryptHandler;
    private readonly IPAddress _rendezvousAddress;
    private readonly int _rendezvousPort;
    private readonly bool _hostRelay;
    private readonly NatIceOptions _ice;
    private TurnClient? _turn;

    public WanSimulationPeerHub(
        ISynapseLogger logger,
        string sessionCode,
        IPAddress? rendezvousAddress = null,
        int rendezvousPort = 7778,
        bool hostRelay = true,
        NatIceOptions? ice = null)
    {
        _sessionCode = sessionCode ?? throw new ArgumentNullException(nameof(sessionCode));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _encryption = PeerEncryption.FromSessionCode(sessionCode);
        _rendezvousAddress = rendezvousAddress ?? IPAddress.Loopback;
        _rendezvousPort = rendezvousPort;
        _hostRelay = hostRelay;
        _ice = ice ?? new NatIceOptions();
        _nat = hostRelay
            ? NatTraversalCoordinator.StartRelay(logger, sessionCode, rendezvousPort, IPAddress.Any)
            : new NatTraversalCoordinator(logger, sessionCode, rendezvousPort, _rendezvousAddress);
        _inner = new MultiPeerSimulationHub(logger, _encryption);
        _decryptHandler = OnEncryptedPatch;
        _inner.ScenePatchReceived += _decryptHandler;
    }

    public event Action<string, ReadOnlyMemory<byte>>? ScenePatchReceived;

    public int ListenPort => _inner.ListenPort;
    public int DroppedPackets { get; private set; }
    public IPAddress RendezvousAddress => _rendezvousAddress;
    public int RendezvousPort => _hostRelay ? _nat.RendezvousPort : _rendezvousPort;
    public bool IsLoopbackOnly => _nat.IsLoopbackOnly;
    public IPEndPoint? MappedEndpoint => _nat.MappedEndpoint;
    public NatIceCandidates? LastIce { get; private set; }
    public string TransportMode { get; private set; } = "tcp";

    public async Task StartHostAsync(int port, CancellationToken ct = default)
    {
        await _inner.StartHostAsync(port, publicBind: true, ct).ConfigureAwait(false);
        await _nat.DiscoverPublicEndpointAsync(ct).ConfigureAwait(false);

        LastIce = await NatIceAssist.GatherAsync(_ice, _logger, ct).ConfigureAwait(false);
        _turn = LastIce.Turn;
        TransportMode = LastIce.Mode;

        int advertisePort = LastIce.AdvertisedPort ?? _inner.ListenPort;
        if (TransportMode != "turn")
            advertisePort = _inner.ListenPort;

        IPAddress? adv = LastIce.AdvertisedAddress;
        if (adv == null || _nat.IsLoopbackOnly || _rendezvousAddress.Equals(IPAddress.Loopback))
            adv = IPAddress.Loopback;

        await _nat.RegisterPublicEndpointAsync(
            advertisePort,
            advertisedAddress: adv,
            mode: TransportMode,
            ct: ct).ConfigureAwait(false);

        _logger.Info("Network",
            $"WAN encrypted+auth host on 0.0.0.0:{_inner.ListenPort} mode={TransportMode} " +
            $"(rendezvous {_rendezvousAddress}:{RendezvousPort}" +
            (LastIce.StunMapped != null ? $", stun={LastIce.StunMapped}" : "") +
            (LastIce.TurnRelayed != null ? $", turn={LastIce.TurnRelayed}" : "") +
            $", loopback={IsLoopbackOnly})");
    }

    public async Task JoinAsync(CancellationToken ct = default)
    {
        await JoinAsync(_rendezvousAddress, _rendezvousPort, ct).ConfigureAwait(false);
    }

    public async Task JoinAsync(int? remoteRendezvousPort, CancellationToken ct = default)
    {
        int port = remoteRendezvousPort is > 0 ? remoteRendezvousPort.Value : _rendezvousPort;
        await JoinAsync(_rendezvousAddress, port, ct).ConfigureAwait(false);
    }

    public async Task JoinAsync(IPAddress rendezvousHost, int rendezvousPort, CancellationToken ct = default)
    {
        using var clientNat = new NatTraversalCoordinator(_logger, _sessionCode, rendezvousPort, rendezvousHost);
        await clientNat.DiscoverPublicEndpointAsync(ct).ConfigureAwait(false);
        LastIce = await NatIceAssist.GatherAsync(_ice, _logger, ct).ConfigureAwait(false);
        _turn = LastIce.Turn;

        var peer = await clientNat.DiscoverPeerAsync(ct).ConfigureAwait(false);
        if (peer == null || peer.TcpPort == 0)
            throw new InvalidOperationException("Peer discovery failed.");

        await clientNat.HolePunchAsync(peer, ct: ct).ConfigureAwait(false);

        var connectHost = rendezvousHost.Equals(IPAddress.Loopback)
            ? IPAddress.Loopback.ToString()
            : peer.Address.Equals(IPAddress.Any) || peer.Address.Equals(IPAddress.IPv6Any)
                ? IPAddress.Loopback.ToString()
                : peer.Address.ToString();

        try
        {
            await _inner.ConnectAsync(connectHost, peer.TcpPort, ct).ConfigureAwait(false);
            TransportMode = "tcp";
            _logger.Info("Network", $"WAN joined peer {connectHost}:{peer.TcpPort} via TCP");
        }
        catch (Exception ex) when (_turn?.RelayedEndpoint != null)
        {
            var endpoint = new IPEndPoint(peer.Address, peer.TcpPort);
            await _turn.CreatePermissionAsync(endpoint, ct: ct).ConfigureAwait(false);
            var channel = await _turn.ChannelBindAsync(endpoint, ct: ct).ConfigureAwait(false);
            TransportMode = "turn";
            _logger.Warn("Network",
                $"TCP join failed ({ex.Message}); TURN fallback active channel={channel} via {_turn.RelayedEndpoint}");
        }
    }

    public async Task BroadcastScenePatchAsync(ReadOnlyMemory<byte> patch, CancellationToken ct = default)
    {
        var encrypted = _encryption.Encrypt(patch.Span);
        if (TransportMode == "turn" && _turn != null)
            await _turn.SendChannelDataAsync(0x4000, encrypted, ct).ConfigureAwait(false);

        await _inner.BroadcastScenePatchAsync(encrypted, ct).ConfigureAwait(false);
    }

    public byte[] DecryptPatch(ReadOnlyMemory<byte> encrypted) => _encryption.Decrypt(encrypted.Span);

    private void OnEncryptedPatch(string peerId, ReadOnlyMemory<byte> encrypted)
    {
        try
        {
            var plain = _encryption.Decrypt(encrypted.Span);
            ScenePatchReceived?.Invoke(peerId, plain);
        }
        catch (CryptographicException ex)
        {
            DroppedPackets++;
            _logger.Warn("Network", $"Dropped tampered/invalid WAN patch from {peerId}: {ex.Message}");
        }
    }

    public async ValueTask DisposeAsync()
    {
        _inner.ScenePatchReceived -= _decryptHandler;
        await _inner.DisposeAsync().ConfigureAwait(false);
        if (_turn != null)
            await _turn.DisposeAsync().ConfigureAwait(false);
        _nat.Dispose();
        _encryption.Dispose();
    }
}
