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
using System.Buffers.Binary;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;

namespace Synapse.Network;

/// <summary>RFC 5389 STUN Binding client (XOR-MAPPED-ADDRESS).</summary>
public static class StunClient
{
    public const uint MagicCookie = 0x2112A442;
    public const ushort BindingRequest = 0x0001;
    public const ushort BindingSuccess = 0x0101;
    public const ushort AttrMappedAddress = 0x0001;
    public const ushort AttrXorMappedAddress = 0x0020;
    public const ushort AttrSoftware = 0x8022;
    public const ushort AttrErrorCode = 0x0009;

    public static async Task<IPEndPoint?> QueryMappedAddressAsync(
        string serverHost,
        int serverPort = 3478,
        TimeSpan? timeout = null,
        CancellationToken ct = default)
    {
        var addresses = await Dns.GetHostAddressesAsync(serverHost, ct).ConfigureAwait(false);
        var serverIp = addresses.FirstOrDefault(a => a.AddressFamily == AddressFamily.InterNetwork)
                       ?? addresses.FirstOrDefault();
        if (serverIp == null)
            return null;

        using var udp = new UdpClient(new IPEndPoint(IPAddress.Any, 0));
        var txn = new byte[12];
        RandomNumberGenerator.Fill(txn);
        var request = BuildBindingRequest(txn);
        var serverEp = new IPEndPoint(serverIp, serverPort);
        await udp.SendAsync(request, serverEp, ct).ConfigureAwait(false);

        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct);
        linked.CancelAfter(timeout ?? TimeSpan.FromSeconds(2));
        try
        {
            var resp = await udp.ReceiveAsync(linked.Token).ConfigureAwait(false);
            return TryParseMappedAddress(resp.Buffer, txn);
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

    public static byte[] BuildBindingRequest(ReadOnlySpan<byte> transactionId12, string software = "Synapse/2.6")
    {
        if (transactionId12.Length != 12)
            throw new ArgumentException("Transaction ID must be 12 bytes.", nameof(transactionId12));

        var softwareBytes = Encoding.UTF8.GetBytes(software);
        int softwarePad = (4 - (softwareBytes.Length % 4)) % 4;
        int attrLen = 4 + softwareBytes.Length + softwarePad;
        var buf = new byte[20 + attrLen];
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(0), BindingRequest);
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(2), (ushort)attrLen);
        BinaryPrimitives.WriteUInt32BigEndian(buf.AsSpan(4), MagicCookie);
        transactionId12.CopyTo(buf.AsSpan(8));

        // SOFTWARE attribute
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(20), AttrSoftware);
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(22), (ushort)softwareBytes.Length);
        softwareBytes.CopyTo(buf.AsSpan(24));
        return buf;
    }

    public static IPEndPoint? TryParseMappedAddress(ReadOnlySpan<byte> message, ReadOnlySpan<byte> expectedTxn)
    {
        if (message.Length < 20)
            return null;
        ushort msgType = BinaryPrimitives.ReadUInt16BigEndian(message);
        if (msgType != BindingSuccess)
            return null;
        uint cookie = BinaryPrimitives.ReadUInt32BigEndian(message.Slice(4));
        if (cookie != MagicCookie)
            return null;
        if (!message.Slice(8, 12).SequenceEqual(expectedTxn))
            return null;

        ushort length = BinaryPrimitives.ReadUInt16BigEndian(message.Slice(2));
        int end = Math.Min(message.Length, 20 + length);
        int offset = 20;
        IPEndPoint? xorMapped = null;
        IPEndPoint? mapped = null;
        while (offset + 4 <= end)
        {
            ushort attrType = BinaryPrimitives.ReadUInt16BigEndian(message.Slice(offset));
            ushort attrLen = BinaryPrimitives.ReadUInt16BigEndian(message.Slice(offset + 2));
            int valueStart = offset + 4;
            int valueEnd = valueStart + attrLen;
            if (valueEnd > end)
                break;

            if (attrType == AttrXorMappedAddress)
                xorMapped = ParseAddressAttribute(message.Slice(valueStart, attrLen), xor: true, expectedTxn);
            else if (attrType == AttrMappedAddress)
                mapped = ParseAddressAttribute(message.Slice(valueStart, attrLen), xor: false, expectedTxn);

            int padded = attrLen + (4 - (attrLen % 4)) % 4;
            offset = valueStart + padded;
        }

        return xorMapped ?? mapped;
    }

    /// <summary>Builds a Binding Success with XOR-MAPPED-ADDRESS (for unit tests / mock servers).</summary>
    public static byte[] BuildBindingSuccess(ReadOnlySpan<byte> transactionId12, IPEndPoint mapped)
    {
        ArgumentNullException.ThrowIfNull(mapped);
        var buf = new byte[32];
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(0), BindingSuccess);
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(2), 12); // attr length
        BinaryPrimitives.WriteUInt32BigEndian(buf.AsSpan(4), MagicCookie);
        transactionId12.CopyTo(buf.AsSpan(8));
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(20), AttrXorMappedAddress);
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(22), 8);
        buf[24] = 0;
        buf[25] = 0x01; // IPv4
        ushort xport = (ushort)(mapped.Port ^ (MagicCookie >> 16));
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(26), xport);
        var addr = mapped.Address.GetAddressBytes();
        if (addr.Length != 4)
            throw new ArgumentException("IPv4 only for test helper.", nameof(mapped));
        uint ip = BinaryPrimitives.ReadUInt32BigEndian(addr);
        BinaryPrimitives.WriteUInt32BigEndian(buf.AsSpan(28), ip ^ MagicCookie);
        return buf;
    }

    private static IPEndPoint? ParseAddressAttribute(ReadOnlySpan<byte> value, bool xor, ReadOnlySpan<byte> txn)
    {
        if (value.Length < 8)
            return null;
        byte family = value[1];
        ushort port = BinaryPrimitives.ReadUInt16BigEndian(value.Slice(2));
        if (xor)
            port ^= (ushort)(MagicCookie >> 16);

        if (family == 0x01) // IPv4
        {
            uint ip = BinaryPrimitives.ReadUInt32BigEndian(value.Slice(4));
            if (xor)
                ip ^= MagicCookie;
            var bytes = new byte[4];
            BinaryPrimitives.WriteUInt32BigEndian(bytes, ip);
            return new IPEndPoint(new IPAddress(bytes), port);
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
            var bytes = new byte[16];
            value.Slice(4, 16).CopyTo(bytes);
            if (xor)
            {
                // XOR with Magic Cookie || Transaction ID
                Span<byte> mask = stackalloc byte[16];
                BinaryPrimitives.WriteUInt32BigEndian(mask, MagicCookie);
                txn.CopyTo(mask.Slice(4));
                for (int i = 0; i < 16; i++)
                    bytes[i] ^= mask[i];
            }

            return new IPEndPoint(new IPAddress(bytes), port);
        }

        return null;
    }
}
