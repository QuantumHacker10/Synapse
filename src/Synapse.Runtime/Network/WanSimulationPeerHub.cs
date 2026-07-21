using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using Synapse.Infrastructure.Logging;

namespace Synapse.Network;

/// <summary>
/// UDP rendezvous + hole-punch coordinator for WAN/LAN P2P.
/// Binds <see cref="IPAddress.Any"/> so peers across NAT (with port-forward / same LAN) can discover.
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
    private static readonly ConcurrentDictionary<string, (IPAddress Address, int TcpPort, DateTime Expiry)> Registrations =
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

    public async Task RegisterPublicEndpointAsync(int tcpPort, CancellationToken ct = default)
    {
        var payload = Encoding.UTF8.GetBytes($"REGISTER|{_sessionCode}|{tcpPort}|{ComputeMac(_sessionCode, tcpPort)}");
        if (payload.Length > MaxUdpPayload)
            throw new InvalidOperationException("REGISTER payload too large.");
        await _udp.SendAsync(payload, new IPEndPoint(_rendezvousAddress, RendezvousPort), ct).ConfigureAwait(false);
        // Local optimistic registration (same-process host); relay overwrites with observed remote IP.
        Registrations[_sessionCode] = (IPAddress.Loopback, tcpPort, DateTime.UtcNow.AddMinutes(5));
        _logger.Info("Network", $"NAT rendezvous REGISTER session={_sessionCode} tcp={tcpPort} via {_rendezvousAddress}:{RendezvousPort}");
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
            return ParsePeerReply(Encoding.UTF8.GetString(result.Buffer), _sessionCode);
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

                        // Prefer the UDP source address (works across LAN/WAN with port-forward).
                        var peerIp = result.RemoteEndPoint.Address;
                        if (peerIp.IsIPv4MappedToIPv6)
                            peerIp = peerIp.MapToIPv4();
                        Registrations[sessionCode] = (peerIp, tcpPort, DateTime.UtcNow.AddMinutes(5));
                        var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{peerIp}|{tcpPort}");
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
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{reg.Address}|{reg.TcpPort}");
                            await relay.SendAsync(reply, result.RemoteEndPoint, coord._cts.Token).ConfigureAwait(false);
                        }
                        else
                        {
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|0.0.0.0|0");
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

    public static IPEndPoint? ParsePeerReply(string text, string sessionCode)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(sessionCode);
        if (!text.StartsWith("PEER|", StringComparison.Ordinal))
            return null;
        var parts = text.Split('|');
        // New: PEER|session|ip|port
        if (parts.Length >= 4 && parts[1] == sessionCode)
        {
            if (!IPAddress.TryParse(parts[2], out var ip))
                return null;
            if (!int.TryParse(parts[3], out var port) || port <= 0)
                return null;
            if (ip.Equals(IPAddress.Any) || ip.Equals(IPAddress.IPv6Any))
                return null;
            return new IPEndPoint(ip, port);
        }

        // Legacy QA: PEER|session|port → Loopback
        if (parts.Length >= 3 && parts[1] == sessionCode && int.TryParse(parts[2], out var legacyPort) && legacyPort > 0)
            return new IPEndPoint(IPAddress.Loopback, legacyPort);

        return null;
    }

    private static IPAddress PreferLoopbackHint(IPAddress bind) =>
        bind.Equals(IPAddress.Any) || bind.Equals(IPAddress.IPv6Any) ? IPAddress.Loopback : bind;

    internal static string ComputeMac(string sessionCode, int tcpPort)
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

/// <summary>WAN-capable encrypted P2P hub (NAT traversal on Any + AES-GCM + auth handshake).</summary>
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

    public WanSimulationPeerHub(
        ISynapseLogger logger,
        string sessionCode,
        IPAddress? rendezvousAddress = null,
        int rendezvousPort = 7778,
        bool hostRelay = true)
    {
        _sessionCode = sessionCode;
        _logger = logger;
        _encryption = PeerEncryption.FromSessionCode(sessionCode);
        _rendezvousAddress = rendezvousAddress ?? IPAddress.Loopback;
        _rendezvousPort = rendezvousPort;
        _hostRelay = hostRelay;
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

    public async Task StartHostAsync(int port, CancellationToken ct = default)
    {
        // Auth is present → publicBind allowed (LAN/WAN with port-forward).
        await _inner.StartHostAsync(port, publicBind: true, ct).ConfigureAwait(false);
        await _nat.RegisterPublicEndpointAsync(_inner.ListenPort, ct).ConfigureAwait(false);
        _logger.Info("Network", $"WAN encrypted+auth host on 0.0.0.0:{_inner.ListenPort} (rendezvous {_rendezvousAddress}:{_rendezvousPort})");
    }

    public async Task JoinAsync(CancellationToken ct = default)
    {
        await JoinAsync(_rendezvousAddress, _rendezvousPort, ct).ConfigureAwait(false);
    }

    public async Task JoinAsync(IPAddress rendezvousHost, int rendezvousPort, CancellationToken ct = default)
    {
        using var clientNat = new NatTraversalCoordinator(_logger, _sessionCode, rendezvousPort, rendezvousHost);
        var endpoint = await clientNat.DiscoverPeerAsync(ct).ConfigureAwait(false);
        if (endpoint == null || endpoint.Port == 0)
            throw new InvalidOperationException("Peer discovery failed.");
        await _inner.ConnectAsync(endpoint.Address.ToString(), endpoint.Port, ct).ConfigureAwait(false);
        _logger.Info("Network", $"WAN joined peer {endpoint}");
    }

    public async Task BroadcastScenePatchAsync(ReadOnlyMemory<byte> patch, CancellationToken ct = default)
    {
        var encrypted = _encryption.Encrypt(patch.Span);
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
        _nat.Dispose();
        _encryption.Dispose();
    }
}
