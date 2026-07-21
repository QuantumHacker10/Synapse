using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using Synapse.Infrastructure.Logging;

namespace Synapse.Network;

/// <summary>Discovered peer endpoint plus transport mode (tcp|stun|turn).</summary>
public readonly record struct PeerCandidate(IPEndPoint Endpoint, string Mode);

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
    private readonly IPAddress _rendezvousAddress;
    private CancellationTokenSource? _cts;
    private UdpClient? _relay;

    // Shared static registry for in-process / same-host relay (stores observed public endpoint).
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
        _logger = logger;
        _sessionCode = sessionCode;
        _rendezvousAddress = rendezvousAddress ?? IPAddress.Loopback;
        RendezvousPort = rendezvousPort;
        // Ephemeral client socket on Any so replies from remote rendezvous work through NAT.
        _udp = new UdpClient(new IPEndPoint(IPAddress.Any, 0));
    }

    public int RendezvousPort { get; }
    public IPAddress RendezvousAddress => _rendezvousAddress;

    public async Task RegisterPublicEndpointAsync(
        int tcpPort,
        IPAddress? advertisedAddress = null,
        string mode = "tcp",
        CancellationToken ct = default)
    {
        var adv = advertisedAddress?.ToString() ?? "";
        var payload = Encoding.UTF8.GetBytes(
            $"REGISTER|{_sessionCode}|{tcpPort}|{ComputeMac(_sessionCode, tcpPort)}|{adv}|{mode}");
        if (payload.Length > MaxUdpPayload)
            throw new InvalidOperationException("REGISTER payload too large.");
        await _udp.SendAsync(payload, new IPEndPoint(_rendezvousAddress, RendezvousPort), ct).ConfigureAwait(false);
        var localIp = advertisedAddress ?? IPAddress.Loopback;
        Registrations[_sessionCode] = (localIp, tcpPort, mode, DateTime.UtcNow.AddMinutes(5));
        _logger.Info("Network",
            $"NAT rendezvous REGISTER session={_sessionCode} tcp={tcpPort} adv={localIp} mode={mode} via {_rendezvousAddress}:{RendezvousPort}");
    }

    public async Task<IPEndPoint?> DiscoverPeerAsync(CancellationToken ct = default)
    {
        var payload = Encoding.UTF8.GetBytes($"DISCOVER|{_sessionCode}|{ComputeMac(_sessionCode, 0)}");
        await _udp.SendAsync(payload, new IPEndPoint(_rendezvousAddress, RendezvousPort), ct).ConfigureAwait(false);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(3));
        try
        {
            var result = await _udp.ReceiveAsync(timeout.Token).ConfigureAwait(false);
            if (result.Buffer.Length > MaxUdpPayload)
                return null;
            return ParsePeerCandidate(Encoding.UTF8.GetString(result.Buffer), _sessionCode)?.Endpoint;
        }
        catch (OperationCanceledException)
        {
            return null;
        }
    }

    /// <summary>
    /// Starts a UDP rendezvous relay bound to <see cref="IPAddress.Any"/> (or an explicit address).
    /// </summary>
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
        var coord = new NatTraversalCoordinator(logger, sessionCode, rendezvousPort, rendezvousAddress: PreferLoopbackHint(bind));
        coord._cts = new CancellationTokenSource();
        coord._relay = new UdpClient(new IPEndPoint(bind, rendezvousPort));
        var relay = coord._relay;
        _ = Task.Run(async () =>
        {
            while (!coord._cts.Token.IsCancellationRequested)
            {
                try
                {
                    var result = await relay.ReceiveAsync(coord._cts.Token).ConfigureAwait(false);
                    if (result.Buffer.Length == 0 || result.Buffer.Length > MaxUdpPayload)
                        continue;
                    var text = Encoding.UTF8.GetString(result.Buffer);
                    var parts = text.Split('|');
                    if (parts.Length < 2)
                        continue;

                    if (text.StartsWith("REGISTER|", StringComparison.Ordinal) && parts.Length >= 4)
                    {
                        if (parts[1] != sessionCode)
                            continue;
                        if (!int.TryParse(parts[2], out var tcpPort) || tcpPort <= 0)
                            continue;
                        if (!string.Equals(parts[3], ComputeMac(sessionCode, tcpPort), StringComparison.Ordinal))
                            continue;

                        // Prefer STUN-advertised IP when present; else UDP source (LAN/WAN with port-forward).
                        var peerIp = result.RemoteEndPoint.Address;
                        if (peerIp.IsIPv4MappedToIPv6)
                            peerIp = peerIp.MapToIPv4();
                        string mode = "tcp";
                        if (parts.Length >= 5 && IPAddress.TryParse(parts[4], out var adv) &&
                            !adv.Equals(IPAddress.Any) && !adv.Equals(IPAddress.None))
                            peerIp = adv;
                        if (parts.Length >= 6 && !string.IsNullOrWhiteSpace(parts[5]))
                            mode = parts[5].Trim().ToLowerInvariant();

                        Registrations[sessionCode] = (peerIp, tcpPort, mode, DateTime.UtcNow.AddMinutes(5));
                        var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{peerIp}|{tcpPort}|{mode}");
                        await relay.SendAsync(reply, result.RemoteEndPoint, coord._cts.Token).ConfigureAwait(false);
                    }
                    else if (text.StartsWith("DISCOVER|", StringComparison.Ordinal) && parts.Length >= 3)
                    {
                        if (parts[1] != sessionCode)
                            continue;
                        if (!string.Equals(parts[2], ComputeMac(sessionCode, 0), StringComparison.Ordinal))
                            continue;
                        if (Registrations.TryGetValue(sessionCode, out var reg) && reg.Expiry > DateTime.UtcNow)
                        {
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{reg.Address}|{reg.TcpPort}|{reg.Mode}");
                            await relay.SendAsync(reply, result.RemoteEndPoint, coord._cts.Token).ConfigureAwait(false);
                        }
                        else
                        {
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|0.0.0.0|0|tcp");
                            await relay.SendAsync(reply, result.RemoteEndPoint, coord._cts.Token).ConfigureAwait(false);
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

            try
            {
                relay.Dispose();
            }
            catch
            {
                // ignore
            }
        });
        logger.Info("Network", $"NAT rendezvous relay on {bind}:{rendezvousPort}");
        return coord;
    }

    public static PeerCandidate? ParsePeerCandidate(string text, string sessionCode)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(sessionCode);
        if (!text.StartsWith("PEER|", StringComparison.Ordinal))
            return null;
        var parts = text.Split('|');
        // PEER|session|ip|port[|mode]
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

        // Legacy QA: PEER|session|port → Loopback
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
        _cts?.Cancel();
        _udp.Dispose();
        try
        {
            _relay?.Dispose();
        }
        catch
        {
            // ignore relay dispose races
        }
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
        _sessionCode = sessionCode;
        _logger = logger;
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
    public int RendezvousPort => _rendezvousPort;
    public NatIceCandidates? LastIce { get; private set; }
    public string TransportMode { get; private set; } = "tcp";

    public async Task StartHostAsync(int port, CancellationToken ct = default)
    {
        // Auth is present → publicBind allowed (LAN/WAN with port-forward).
        await _inner.StartHostAsync(port, publicBind: true, ct).ConfigureAwait(false);

        LastIce = await NatIceAssist.GatherAsync(_ice, _logger, ct).ConfigureAwait(false);
        _turn = LastIce.Turn;
        TransportMode = LastIce.Mode;

        int advertisePort = LastIce.AdvertisedPort ?? _inner.ListenPort;
        // For turn mode the registered port is the TURN relay UDP port; TCP remains for LAN/direct.
        if (TransportMode != "turn")
            advertisePort = _inner.ListenPort;

        await _nat.RegisterPublicEndpointAsync(
            advertisePort,
            advertisedAddress: LastIce.AdvertisedAddress,
            mode: TransportMode,
            ct: ct).ConfigureAwait(false);

        _logger.Info("Network",
            $"WAN encrypted+auth host on 0.0.0.0:{_inner.ListenPort} mode={TransportMode} " +
            $"(rendezvous {_rendezvousAddress}:{_rendezvousPort}" +
            (LastIce.StunMapped != null ? $", stun={LastIce.StunMapped}" : "") +
            (LastIce.TurnRelayed != null ? $", turn={LastIce.TurnRelayed}" : "") +
            ")");
    }

    public async Task JoinAsync(CancellationToken ct = default)
    {
        await JoinAsync(_rendezvousAddress, _rendezvousPort, ct).ConfigureAwait(false);
    }

    public async Task JoinAsync(IPAddress rendezvousHost, int rendezvousPort, CancellationToken ct = default)
    {
        using var clientNat = new NatTraversalCoordinator(_logger, _sessionCode, rendezvousPort, rendezvousHost);
        // Gather local STUN/TURN for diagnostics and symmetric-NAT fallback.
        LastIce = await NatIceAssist.GatherAsync(_ice, _logger, ct).ConfigureAwait(false);
        _turn = LastIce.Turn;

        var endpoint = await clientNat.DiscoverPeerAsync(ct).ConfigureAwait(false);
        if (endpoint == null || endpoint.Port == 0)
            throw new InvalidOperationException("Peer discovery failed.");

        // Direct TCP for tcp/stun modes (stun advertises public IP for port-forwarded hosts).
        // Turn mode: keep TURN allocation for ChannelData relay; still attempt TCP when port is host listen.
        try
        {
            await _inner.ConnectAsync(endpoint.Address.ToString(), endpoint.Port, ct).ConfigureAwait(false);
            TransportMode = "tcp";
            _logger.Info("Network", $"WAN joined peer {endpoint} via TCP");
        }
        catch (Exception ex) when (_turn?.RelayedEndpoint != null)
        {
            // Symmetric NAT fallback: CreatePermission toward discovered peer and keep TURN session.
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
        {
            // Best-effort ChannelData on channel 0x4000 (bound during join/host assist).
            await _turn.SendChannelDataAsync(0x4000, encrypted, ct).ConfigureAwait(false);
        }

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
