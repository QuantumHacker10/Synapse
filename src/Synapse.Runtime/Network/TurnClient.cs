using System.Buffers.Binary;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;

namespace Synapse.Network;

/// <summary>
/// Minimal RFC 5766 TURN client: Allocate (long-term credentials), CreatePermission, ChannelBind, ChannelData.
/// Suitable for symmetric-NAT relay when direct TCP fails.
/// </summary>
public sealed class TurnClient : IAsyncDisposable
{
    public const ushort AllocateRequest = 0x0003;
    public const ushort AllocateSuccess = 0x0103;
    public const ushort AllocateError = 0x0113;
    public const ushort CreatePermissionRequest = 0x0008;
    public const ushort CreatePermissionSuccess = 0x0108;
    public const ushort ChannelBindRequest = 0x0009;
    public const ushort ChannelBindSuccess = 0x0109;
    public const ushort RefreshRequest = 0x0004;

    public const ushort AttrXorPeerAddress = 0x0012;
    public const ushort AttrXorRelayedAddress = 0x0016;
    public const ushort AttrXorMappedAddress = 0x0020;
    public const ushort AttrUsername = 0x0006;
    public const ushort AttrMessageIntegrity = 0x0008;
    public const ushort AttrErrorCode = 0x0009;
    public const ushort AttrLifetime = 0x000D;
    public const ushort AttrRequestedTransport = 0x0019;
    public const ushort AttrNonce = 0x0015;
    public const ushort AttrRealm = 0x0014;
    public const ushort AttrChannelNumber = 0x000C;

    private readonly UdpClient _udp;
    private readonly IPEndPoint _server;
    private readonly string _username;
    private readonly string _password;
    private string? _realm;
    private string? _nonce;
    private ushort _nextChannel = 0x4000;

    public TurnClient(IPEndPoint server, string username, string password)
    {
        ArgumentNullException.ThrowIfNull(server);
        _server = server;
        _username = username ?? throw new ArgumentNullException(nameof(username));
        _password = password ?? throw new ArgumentNullException(nameof(password));
        _udp = new UdpClient(new IPEndPoint(IPAddress.Any, 0));
    }

    public IPEndPoint? RelayedEndpoint { get; private set; }
    public IPEndPoint? MappedEndpoint { get; private set; }
    public int LifetimeSeconds { get; private set; } = 600;

    public static async Task<TurnClient?> TryAllocateAsync(
        string serverHost,
        int serverPort,
        string username,
        string password,
        TimeSpan? timeout = null,
        CancellationToken ct = default)
    {
        var addrs = await Dns.GetHostAddressesAsync(serverHost, ct).ConfigureAwait(false);
        var ip = addrs.FirstOrDefault(a => a.AddressFamily == AddressFamily.InterNetwork) ?? addrs.FirstOrDefault();
        if (ip == null)
            return null;

        var client = new TurnClient(new IPEndPoint(ip, serverPort), username, password);
        try
        {
            if (!await client.AllocateAsync(timeout ?? TimeSpan.FromSeconds(3), ct).ConfigureAwait(false))
            {
                await client.DisposeAsync().ConfigureAwait(false);
                return null;
            }

            return client;
        }
        catch
        {
            await client.DisposeAsync().ConfigureAwait(false);
            return null;
        }
    }

    public async Task<bool> AllocateAsync(TimeSpan timeout, CancellationToken ct = default)
    {
        // First attempt without integrity → capture REALM/NONCE from 401.
        var txn1 = NewTxn();
        var req1 = BuildAllocateRequest(txn1, includeIntegrity: false);
        await _udp.SendAsync(req1, _server, ct).ConfigureAwait(false);
        var resp1 = await ReceiveMatchingAsync(txn1, timeout, ct).ConfigureAwait(false);
        if (resp1 != null)
            TryCaptureChallenge(resp1);

        var txn2 = NewTxn();
        var req2 = BuildAllocateRequest(txn2, includeIntegrity: !string.IsNullOrEmpty(_realm));
        await _udp.SendAsync(req2, _server, ct).ConfigureAwait(false);
        var resp2 = await ReceiveMatchingAsync(txn2, timeout, ct).ConfigureAwait(false);
        if (resp2 == null)
            return false;

        if (!TryParseAllocateSuccess(resp2, txn2, out var relayed, out var mapped, out var lifetime))
        {
            // Some servers succeed on first message (no auth / open relay for QA).
            if (!TryParseAllocateSuccess(resp1 ?? resp2, txn1, out relayed, out mapped, out lifetime))
                return false;
        }

        RelayedEndpoint = relayed;
        MappedEndpoint = mapped;
        LifetimeSeconds = lifetime;
        return RelayedEndpoint != null;
    }

    public async Task<bool> CreatePermissionAsync(IPEndPoint peer, TimeSpan? timeout = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(peer);
        var txn = NewTxn();
        var msg = BuildCreatePermission(txn, peer);
        await _udp.SendAsync(msg, _server, ct).ConfigureAwait(false);
        var resp = await ReceiveMatchingAsync(txn, timeout ?? TimeSpan.FromSeconds(2), ct).ConfigureAwait(false);
        return resp != null && BinaryPrimitives.ReadUInt16BigEndian(resp.AsSpan(0, 2)) == CreatePermissionSuccess;
    }

    public async Task<ushort?> ChannelBindAsync(IPEndPoint peer, ushort? channel = null, TimeSpan? timeout = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(peer);
        ushort ch = channel ?? _nextChannel++;
        if (ch < 0x4000 || ch > 0x7FFE)
            throw new ArgumentOutOfRangeException(nameof(channel));

        var txn = NewTxn();
        var msg = BuildChannelBind(txn, peer, ch);
        await _udp.SendAsync(msg, _server, ct).ConfigureAwait(false);
        var resp = await ReceiveMatchingAsync(txn, timeout ?? TimeSpan.FromSeconds(2), ct).ConfigureAwait(false);
        if (resp == null || BinaryPrimitives.ReadUInt16BigEndian(resp.AsSpan(0, 2)) != ChannelBindSuccess)
            return null;
        return ch;
    }

    public async Task SendChannelDataAsync(ushort channel, ReadOnlyMemory<byte> payload, CancellationToken ct = default)
    {
        // ChannelData header: channel(2) + length(2) + data + pad to 4
        int pad = (4 - (payload.Length % 4)) % 4;
        var buf = new byte[4 + payload.Length + pad];
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(0), channel);
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(2), (ushort)payload.Length);
        payload.Span.CopyTo(buf.AsSpan(4));
        await _udp.SendAsync(buf, _server, ct).ConfigureAwait(false);
    }

    public async Task<(ushort Channel, byte[] Payload)?> TryReceiveChannelDataAsync(TimeSpan timeout, CancellationToken ct = default)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct);
        linked.CancelAfter(timeout);
        try
        {
            var result = await _udp.ReceiveAsync(linked.Token).ConfigureAwait(false);
            if (result.Buffer.Length < 4)
                return null;
            ushort ch = BinaryPrimitives.ReadUInt16BigEndian(result.Buffer.AsSpan(0, 2));
            if (ch < 0x4000)
                return null; // STUN message
            ushort len = BinaryPrimitives.ReadUInt16BigEndian(result.Buffer.AsSpan(2, 2));
            if (4 + len > result.Buffer.Length)
                return null;
            var payload = new byte[len];
            Buffer.BlockCopy(result.Buffer, 4, payload, 0, len);
            return (ch, payload);
        }
        catch (OperationCanceledException)
        {
            return null;
        }
    }

    public ValueTask DisposeAsync()
    {
        _udp.Dispose();
        return ValueTask.CompletedTask;
    }

    // ---- message builders (also used by tests) ----

    public byte[] BuildAllocateRequest(ReadOnlySpan<byte> txn, bool includeIntegrity)
    {
        var attrs = new List<byte[]>();
        // REQUESTED-TRANSPORT = UDP (17)
        var transport = new byte[8];
        BinaryPrimitives.WriteUInt16BigEndian(transport.AsSpan(0), AttrRequestedTransport);
        BinaryPrimitives.WriteUInt16BigEndian(transport.AsSpan(2), 4);
        transport[4] = 17;
        attrs.Add(transport);

        if (!string.IsNullOrEmpty(_username))
            attrs.Add(BuildStringAttr(AttrUsername, _username));
        if (!string.IsNullOrEmpty(_realm))
            attrs.Add(BuildStringAttr(AttrRealm, _realm!));
        if (!string.IsNullOrEmpty(_nonce))
            attrs.Add(BuildStringAttr(AttrNonce, _nonce!));

        return BuildStunMessage(AllocateRequest, txn, attrs, includeIntegrity);
    }

    public static bool TryParseAllocateSuccess(
        byte[] message,
        ReadOnlySpan<byte> txn,
        out IPEndPoint? relayed,
        out IPEndPoint? mapped,
        out int lifetime)
    {
        relayed = null;
        mapped = null;
        lifetime = 600;
        ArgumentNullException.ThrowIfNull(message);
        if (message.Length < 20)
            return false;
        ushort type = BinaryPrimitives.ReadUInt16BigEndian(message.AsSpan(0, 2));
        if (type != AllocateSuccess)
            return false;
        if (BinaryPrimitives.ReadUInt32BigEndian(message.AsSpan(4, 4)) != StunClient.MagicCookie)
            return false;
        if (!message.AsSpan(8, 12).SequenceEqual(txn))
            return false;

        ushort length = BinaryPrimitives.ReadUInt16BigEndian(message.AsSpan(2, 2));
        int end = Math.Min(message.Length, 20 + length);
        int offset = 20;
        while (offset + 4 <= end)
        {
            ushort attrType = BinaryPrimitives.ReadUInt16BigEndian(message.AsSpan(offset, 2));
            ushort attrLen = BinaryPrimitives.ReadUInt16BigEndian(message.AsSpan(offset + 2, 2));
            int vs = offset + 4;
            if (vs + attrLen > end)
                break;
            var value = message.AsSpan(vs, attrLen);
            if (attrType == AttrXorRelayedAddress)
                relayed = ParseXorAddress(value, txn);
            else if (attrType == AttrXorMappedAddress)
                mapped = ParseXorAddress(value, txn);
            else if (attrType == AttrLifetime && attrLen >= 4)
                lifetime = (int)BinaryPrimitives.ReadUInt32BigEndian(value);

            int padded = attrLen + (4 - (attrLen % 4)) % 4;
            offset = vs + padded;
        }

        return relayed != null;
    }

    /// <summary>Test helper: craft Allocate Success with XOR-RELAYED-ADDRESS.</summary>
    public static byte[] BuildAllocateSuccessForTests(ReadOnlySpan<byte> txn, IPEndPoint relayed, IPEndPoint mapped, int lifetime = 600)
    {
        ArgumentNullException.ThrowIfNull(relayed);
        ArgumentNullException.ThrowIfNull(mapped);
        var relayedAttr = EncodeXorAddress(AttrXorRelayedAddress, relayed, txn);
        var mappedAttr = EncodeXorAddress(AttrXorMappedAddress, mapped, txn);
        var life = new byte[8];
        BinaryPrimitives.WriteUInt16BigEndian(life.AsSpan(0), AttrLifetime);
        BinaryPrimitives.WriteUInt16BigEndian(life.AsSpan(2), 4);
        BinaryPrimitives.WriteUInt32BigEndian(life.AsSpan(4), (uint)lifetime);

        int attrLen = relayedAttr.Length + mappedAttr.Length + life.Length;
        var buf = new byte[20 + attrLen];
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(0), AllocateSuccess);
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(2), (ushort)attrLen);
        BinaryPrimitives.WriteUInt32BigEndian(buf.AsSpan(4), StunClient.MagicCookie);
        txn.CopyTo(buf.AsSpan(8));
        int o = 20;
        relayedAttr.CopyTo(buf.AsSpan(o));
        o += relayedAttr.Length;
        mappedAttr.CopyTo(buf.AsSpan(o));
        o += mappedAttr.Length;
        life.CopyTo(buf.AsSpan(o));
        return buf;
    }

    private byte[] BuildCreatePermission(ReadOnlySpan<byte> txn, IPEndPoint peer)
    {
        var attrs = new List<byte[]> { EncodeXorAddress(AttrXorPeerAddress, peer, txn) };
        if (!string.IsNullOrEmpty(_username))
            attrs.Add(BuildStringAttr(AttrUsername, _username));
        if (!string.IsNullOrEmpty(_realm))
            attrs.Add(BuildStringAttr(AttrRealm, _realm!));
        if (!string.IsNullOrEmpty(_nonce))
            attrs.Add(BuildStringAttr(AttrNonce, _nonce!));
        return BuildStunMessage(CreatePermissionRequest, txn, attrs, includeIntegrity: !string.IsNullOrEmpty(_realm));
    }

    private byte[] BuildChannelBind(ReadOnlySpan<byte> txn, IPEndPoint peer, ushort channel)
    {
        var chAttr = new byte[8];
        BinaryPrimitives.WriteUInt16BigEndian(chAttr.AsSpan(0), AttrChannelNumber);
        BinaryPrimitives.WriteUInt16BigEndian(chAttr.AsSpan(2), 4);
        BinaryPrimitives.WriteUInt16BigEndian(chAttr.AsSpan(4), channel);
        var attrs = new List<byte[]> { chAttr, EncodeXorAddress(AttrXorPeerAddress, peer, txn) };
        if (!string.IsNullOrEmpty(_username))
            attrs.Add(BuildStringAttr(AttrUsername, _username));
        if (!string.IsNullOrEmpty(_realm))
            attrs.Add(BuildStringAttr(AttrRealm, _realm!));
        if (!string.IsNullOrEmpty(_nonce))
            attrs.Add(BuildStringAttr(AttrNonce, _nonce!));
        return BuildStunMessage(ChannelBindRequest, txn, attrs, includeIntegrity: !string.IsNullOrEmpty(_realm));
    }

    private byte[] BuildStunMessage(ushort method, ReadOnlySpan<byte> txn, List<byte[]> attrs, bool includeIntegrity)
    {
        int attrBytes = attrs.Sum(a => a.Length);
        int integrityLen = includeIntegrity ? 24 : 0; // type+len+20 hmac
        var buf = new byte[20 + attrBytes + integrityLen];
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(0), method);
        BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(2), (ushort)(attrBytes + integrityLen));
        BinaryPrimitives.WriteUInt32BigEndian(buf.AsSpan(4), StunClient.MagicCookie);
        txn.CopyTo(buf.AsSpan(8));
        int o = 20;
        foreach (var a in attrs)
        {
            a.CopyTo(buf.AsSpan(o));
            o += a.Length;
        }

        if (includeIntegrity)
        {
            // MESSAGE-INTEGRITY covers header+attrs up to but not including this attribute;
            // length field must already include the 24-byte MI attribute.
            byte[] key = DeriveLongTermKey(_username, _realm ?? "", _password);
            int miCovered = o;
            BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(o), AttrMessageIntegrity);
            BinaryPrimitives.WriteUInt16BigEndian(buf.AsSpan(o + 2), 20);
#pragma warning disable CA5350 // HMAC-SHA1 required by RFC 5766 MESSAGE-INTEGRITY
            using var hmac = new HMACSHA1(key);
            var hash = hmac.ComputeHash(buf, 0, miCovered);
#pragma warning restore CA5350
            hash.AsSpan(0, 20).CopyTo(buf.AsSpan(o + 4));
        }

        return buf;
    }

    private void TryCaptureChallenge(byte[] message)
    {
        if (message.Length < 20)
            return;
        ushort length = BinaryPrimitives.ReadUInt16BigEndian(message.AsSpan(2, 2));
        int end = Math.Min(message.Length, 20 + length);
        int offset = 20;
        while (offset + 4 <= end)
        {
            ushort attrType = BinaryPrimitives.ReadUInt16BigEndian(message.AsSpan(offset, 2));
            ushort attrLen = BinaryPrimitives.ReadUInt16BigEndian(message.AsSpan(offset + 2, 2));
            int vs = offset + 4;
            if (vs + attrLen > end)
                break;
            if (attrType == AttrRealm)
                _realm = Encoding.UTF8.GetString(message, vs, attrLen);
            else if (attrType == AttrNonce)
                _nonce = Encoding.UTF8.GetString(message, vs, attrLen);
            int padded = attrLen + (4 - (attrLen % 4)) % 4;
            offset = vs + padded;
        }
    }

    private async Task<byte[]?> ReceiveMatchingAsync(byte[] expectedTxn, TimeSpan timeout, CancellationToken ct)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct);
        linked.CancelAfter(timeout);
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            try
            {
                var result = await _udp.ReceiveAsync(linked.Token).ConfigureAwait(false);
                if (result.Buffer.Length >= 20 &&
                    result.Buffer.AsSpan(8, 12).SequenceEqual(expectedTxn))
                    return result.Buffer;
            }
            catch (OperationCanceledException)
            {
                return null;
            }
        }

        return null;
    }

    private static byte[] NewTxn()
    {
        var t = new byte[12];
        RandomNumberGenerator.Fill(t);
        return t;
    }

    private static byte[] BuildStringAttr(ushort type, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        int pad = (4 - (bytes.Length % 4)) % 4;
        var attr = new byte[4 + bytes.Length + pad];
        BinaryPrimitives.WriteUInt16BigEndian(attr.AsSpan(0), type);
        BinaryPrimitives.WriteUInt16BigEndian(attr.AsSpan(2), (ushort)bytes.Length);
        bytes.CopyTo(attr.AsSpan(4));
        return attr;
    }

    private static byte[] EncodeXorAddress(ushort attrType, IPEndPoint ep, ReadOnlySpan<byte> txn)
    {
        var ip = ep.Address.GetAddressBytes();
        if (ip.Length != 4)
            throw new NotSupportedException("TURN helper encodes IPv4 only.");
        var attr = new byte[12];
        BinaryPrimitives.WriteUInt16BigEndian(attr.AsSpan(0), attrType);
        BinaryPrimitives.WriteUInt16BigEndian(attr.AsSpan(2), 8);
        attr[4] = 0;
        attr[5] = 0x01;
        ushort xport = (ushort)(ep.Port ^ (StunClient.MagicCookie >> 16));
        BinaryPrimitives.WriteUInt16BigEndian(attr.AsSpan(6), xport);
        uint ipNum = BinaryPrimitives.ReadUInt32BigEndian(ip);
        BinaryPrimitives.WriteUInt32BigEndian(attr.AsSpan(8), ipNum ^ StunClient.MagicCookie);
        return attr;
    }

    private static IPEndPoint? ParseXorAddress(ReadOnlySpan<byte> value, ReadOnlySpan<byte> txn)
    {
        if (value.Length < 8 || value[1] != 0x01)
            return null;
        ushort port = (ushort)(BinaryPrimitives.ReadUInt16BigEndian(value.Slice(2)) ^ (StunClient.MagicCookie >> 16));
        uint ip = BinaryPrimitives.ReadUInt32BigEndian(value.Slice(4)) ^ StunClient.MagicCookie;
        var bytes = new byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(bytes, ip);
        return new IPEndPoint(new IPAddress(bytes), port);
    }

    public static byte[] DeriveLongTermKey(string username, string realm, string password)
    {
        // key = MD5(username ":" realm ":" password)
#pragma warning disable CA5351
        var material = Encoding.UTF8.GetBytes($"{username}:{realm}:{password}");
        return MD5.HashData(material);
#pragma warning restore CA5351
    }
}
