using System;
using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Network;

/// <summary>
/// RFC 5389 STUN Binding client used to discover the public (mapped) UDP endpoint
/// behind NAT. Defaults to Google's public STUN servers.
/// </summary>
public sealed class StunClient
{
    public const int DefaultStunPort = 3478;
    public static readonly string[] DefaultServers =
    [
        "stun.l.google.com",
        "stun1.l.google.com",
        "stun2.l.google.com"
    ];

    private readonly ISynapseLogger? _logger;
    private readonly string[] _servers;

    public StunClient(ISynapseLogger? logger = null, string[]? servers = null)
    {
        _logger = logger;
        _servers = servers is { Length: > 0 } ? servers : DefaultServers;
    }

    /// <summary>
    /// Sends a STUN Binding Request from <paramref name="udp"/> and returns the XOR-MAPPED-ADDRESS
    /// (or MAPPED-ADDRESS) advertised by the server.
    /// </summary>
    public async Task<IPEndPoint?> DiscoverMappedAddressAsync(
        UdpClient udp,
        TimeSpan? timeout = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(udp);
        var wait = timeout ?? TimeSpan.FromSeconds(2);

        foreach (var server in _servers)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                var addresses = await Dns.GetHostAddressesAsync(server, ct).ConfigureAwait(false);
                IPAddress? ipv4 = null;
                foreach (var a in addresses)
                {
                    if (a.AddressFamily == AddressFamily.InterNetwork)
                    {
                        ipv4 = a;
                        break;
                    }
                }

                if (ipv4 is null)
                    continue;

                var remote = new IPEndPoint(ipv4, DefaultStunPort);
                var mapped = await QueryServerAsync(udp, remote, wait, ct).ConfigureAwait(false);
                if (mapped != null)
                {
                    _logger?.Info("Network", $"STUN mapped address {mapped} via {server}");
                    return mapped;
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger?.Debug("Network", $"STUN query to {server} failed: {ex.Message}");
            }
        }

        return null;
    }

    private static async Task<IPEndPoint?> QueryServerAsync(
        UdpClient udp,
        IPEndPoint server,
        TimeSpan timeout,
        CancellationToken ct)
    {
        // STUN Binding Request header (20 bytes) + no attributes.
        var request = new byte[20];
        BinaryPrimitives.WriteUInt16BigEndian(request.AsSpan(), 0x0001); // Binding Request
        BinaryPrimitives.WriteUInt16BigEndian(request.AsSpan(2), 0); // length
        BinaryPrimitives.WriteUInt32BigEndian(request.AsSpan(4), 0x2112A442); // magic cookie
        var txn = request.AsSpan(8, 12);
        Random.Shared.NextBytes(txn);
        var txnCopy = request.AsSpan(8, 12).ToArray();

        await udp.SendAsync(request, server, ct).ConfigureAwait(false);

        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct);
        linked.CancelAfter(timeout);
        try
        {
            while (!linked.IsCancellationRequested)
            {
                var result = await udp.ReceiveAsync(linked.Token).ConfigureAwait(false);
                if (TryParseMappedAddress(result.Buffer, txnCopy, out var mapped))
                    return mapped;
            }
        }
        catch (OperationCanceledException)
        {
            return null;
        }

        return null;
    }

    public static bool TryParseMappedAddress(ReadOnlySpan<byte> packet, ReadOnlySpan<byte> expectedTxn, out IPEndPoint? mapped)
    {
        mapped = null;
        if (packet.Length < 20)
            return false;

        ushort messageType = BinaryPrimitives.ReadUInt16BigEndian(packet);
        // Binding Success Response = 0x0101
        if (messageType != 0x0101)
            return false;

        uint magic = BinaryPrimitives.ReadUInt32BigEndian(packet[4..]);
        if (magic != 0x2112A442)
            return false;

        if (!packet.Slice(8, 12).SequenceEqual(expectedTxn))
            return false;

        ushort length = BinaryPrimitives.ReadUInt16BigEndian(packet[2..]);
        int end = Math.Min(packet.Length, 20 + length);
        int offset = 20;
        IPEndPoint? xorMapped = null;
        IPEndPoint? plainMapped = null;

        while (offset + 4 <= end)
        {
            ushort attrType = BinaryPrimitives.ReadUInt16BigEndian(packet[offset..]);
            ushort attrLen = BinaryPrimitives.ReadUInt16BigEndian(packet[(offset + 2)..]);
            offset += 4;
            if (offset + attrLen > end)
                break;

            var value = packet.Slice(offset, attrLen);
            if (attrType == 0x0020) // XOR-MAPPED-ADDRESS
                xorMapped = ParseAddress(value, xor: true, magic, expectedTxn);
            else if (attrType == 0x0001) // MAPPED-ADDRESS
                plainMapped = ParseAddress(value, xor: false, magic, expectedTxn);

            // Attributes are padded to 4-byte boundary.
            offset += attrLen;
            int pad = (4 - (attrLen % 4)) % 4;
            offset += pad;
        }

        mapped = xorMapped ?? plainMapped;
        return mapped != null;
    }

    private static IPEndPoint? ParseAddress(ReadOnlySpan<byte> value, bool xor, uint magic, ReadOnlySpan<byte> txn)
    {
        if (value.Length < 8)
            return null;

        byte family = value[1];
        ushort portRaw = BinaryPrimitives.ReadUInt16BigEndian(value[2..]);
        ushort port = xor ? (ushort)(portRaw ^ (magic >> 16)) : portRaw;

        if (family == 0x01 && value.Length >= 8) // IPv4
        {
            uint addrRaw = BinaryPrimitives.ReadUInt32BigEndian(value[4..]);
            uint addr = xor ? addrRaw ^ magic : addrRaw;
            var ip = new IPAddress(new byte[]
            {
                (byte)(addr >> 24),
                (byte)(addr >> 16),
                (byte)(addr >> 8),
                (byte)addr
            });
            return new IPEndPoint(ip, port);
        }

        if (family == 0x02 && value.Length >= 20) // IPv6
        {
            Span<byte> addr = stackalloc byte[16];
            value.Slice(4, 16).CopyTo(addr);
            if (xor)
            {
                // XOR with magic || transaction id
                Span<byte> mask = stackalloc byte[16];
                BinaryPrimitives.WriteUInt32BigEndian(mask, magic);
                txn.CopyTo(mask[4..]);
                for (int i = 0; i < 16; i++)
                    addr[i] ^= mask[i];
            }

            return new IPEndPoint(new IPAddress(addr), port);
        }

        return null;
    }
}
