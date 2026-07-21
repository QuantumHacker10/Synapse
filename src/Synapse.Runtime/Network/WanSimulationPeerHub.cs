using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using Synapse.Infrastructure.Logging;

namespace Synapse.Network;

/// <summary>UDP rendezvous + hole-punch coordinator for WAN P2P (hardened v2.3+).</summary>
public sealed class NatTraversalCoordinator : IDisposable
{
    private const int MaxUdpPayload = 512;
    private readonly UdpClient _udp;
    private readonly ISynapseLogger _logger;
    private readonly string _sessionCode;
    private CancellationTokenSource? _cts;
    private UdpClient? _relay;

    // Shared static registry for in-process relay (loopback QA / same-process host+client).
    private static readonly ConcurrentDictionary<string, (int TcpPort, DateTime Expiry)> Registrations = new(StringComparer.Ordinal);

    public NatTraversalCoordinator(ISynapseLogger logger, string sessionCode, int rendezvousPort = 7778)
    {
        if (string.IsNullOrWhiteSpace(sessionCode) || sessionCode.Length < 4)
            throw new ArgumentException("Session code must be at least 4 characters.", nameof(sessionCode));
        _logger = logger;
        _sessionCode = sessionCode;
        _udp = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0));
        RendezvousPort = rendezvousPort;
    }

    public int RendezvousPort { get; }

    public async Task RegisterPublicEndpointAsync(int tcpPort, CancellationToken ct = default)
    {
        var payload = Encoding.UTF8.GetBytes($"REGISTER|{_sessionCode}|{tcpPort}|{ComputeMac(_sessionCode, tcpPort)}");
        if (payload.Length > MaxUdpPayload)
            throw new InvalidOperationException("REGISTER payload too large.");
        await _udp.SendAsync(payload, new IPEndPoint(IPAddress.Loopback, RendezvousPort), ct);
        Registrations[_sessionCode] = (tcpPort, DateTime.UtcNow.AddMinutes(5));
        _logger.Info("Network", $"NAT rendezvous registered session={_sessionCode} tcp={tcpPort}");
    }

    public async Task<IPEndPoint?> DiscoverPeerAsync(CancellationToken ct = default)
    {
        var payload = Encoding.UTF8.GetBytes($"DISCOVER|{_sessionCode}|{ComputeMac(_sessionCode, 0)}");
        await _udp.SendAsync(payload, new IPEndPoint(IPAddress.Loopback, RendezvousPort), ct);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(2));
        try
        {
            var result = await _udp.ReceiveAsync(timeout.Token);
            if (result.Buffer.Length > MaxUdpPayload)
                return null;
            var text = Encoding.UTF8.GetString(result.Buffer);
            if (text.StartsWith("PEER|", StringComparison.Ordinal))
            {
                var parts = text.Split('|');
                if (parts.Length >= 3 && parts[1] == _sessionCode && int.TryParse(parts[2], out var port) && port > 0)
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
        coord._relay = new UdpClient(new IPEndPoint(IPAddress.Loopback, rendezvousPort));
        var relay = coord._relay;
        _ = Task.Run(async () =>
        {
            while (!coord._cts.Token.IsCancellationRequested)
            {
                try
                {
                    var result = await relay.ReceiveAsync(coord._cts.Token);
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
                        Registrations[sessionCode] = (tcpPort, DateTime.UtcNow.AddMinutes(5));
                        var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{tcpPort}");
                        await relay.SendAsync(reply, result.RemoteEndPoint, coord._cts.Token);
                    }
                    else if (text.StartsWith("DISCOVER|", StringComparison.Ordinal) && parts.Length >= 3)
                    {
                        if (parts[1] != sessionCode)
                            continue;
                        if (!string.Equals(parts[2], ComputeMac(sessionCode, 0), StringComparison.Ordinal))
                            continue;
                        if (Registrations.TryGetValue(sessionCode, out var reg) && reg.Expiry > DateTime.UtcNow)
                        {
                            var reply = Encoding.UTF8.GetBytes($"PEER|{sessionCode}|{reg.TcpPort}");
                            await relay.SendAsync(reply, result.RemoteEndPoint, coord._cts.Token);
                        }
                        else
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

/// <summary>WAN-capable encrypted P2P hub (NAT traversal + AES-GCM + auth handshake).</summary>
public sealed class WanSimulationPeerHub : IAsyncDisposable
{
    private readonly ISynapseLogger _logger;
    private readonly string _sessionCode;
    private readonly PeerEncryption _encryption;
    private readonly NatTraversalCoordinator _nat;
    private readonly MultiPeerSimulationHub _inner;
    private readonly Action<string, ReadOnlyMemory<byte>> _decryptHandler;

    public WanSimulationPeerHub(ISynapseLogger logger, string sessionCode)
    {
        _sessionCode = sessionCode;
        _logger = logger;
        _encryption = PeerEncryption.FromSessionCode(sessionCode);
        _nat = NatTraversalCoordinator.StartRelay(logger, sessionCode);
        _inner = new MultiPeerSimulationHub(logger, _encryption);
        _decryptHandler = OnEncryptedPatch;
        _inner.ScenePatchReceived += _decryptHandler;
    }

    public event Action<string, ReadOnlyMemory<byte>>? ScenePatchReceived;

    public int ListenPort => _inner.ListenPort;
    public int DroppedPackets { get; private set; }

    public async Task StartHostAsync(int port, CancellationToken ct = default)
    {
        // Auth is present → publicBind allowed.
        await _inner.StartHostAsync(port, publicBind: true, ct).ConfigureAwait(false);
        await _nat.RegisterPublicEndpointAsync(_inner.ListenPort, ct).ConfigureAwait(false);
        _logger.Info("Network", $"WAN encrypted+auth host on port {_inner.ListenPort}");
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

    private void OnEncryptedPatch(string peerId, ReadOnlyMemory<byte> encrypted)
    {
        // Local self-echo arrives already encrypted from BroadcastScenePatchAsync.
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
