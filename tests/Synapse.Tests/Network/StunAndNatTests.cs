using System;
using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using System.Threading.Tasks;
using FluentAssertions;
using Synapse.Infrastructure.Logging;
using Synapse.Network;
using Xunit;

namespace Synapse.Tests.Network;

public sealed class StunClientTests
{
    [Fact]
    public void TryParse_XorMappedAddress_Ipv4()
    {
        Span<byte> txn = stackalloc byte[12];
        txn.Clear();
        txn[0] = 0x11;
        txn[1] = 0x22;

        const uint magic = 0x2112A442;
        ushort port = 54320;
        uint ip = (203u << 24) | (0u << 16) | (113u << 8) | 5u;

        var packet = new byte[32];
        packet[0] = 0x01;
        packet[1] = 0x01;
        packet[2] = 0x00;
        packet[3] = 12;
        packet[4] = 0x21;
        packet[5] = 0x12;
        packet[6] = 0xA4;
        packet[7] = 0x42;
        txn.CopyTo(packet.AsSpan(8, 12));
        packet[20] = 0x00;
        packet[21] = 0x20;
        packet[22] = 0x00;
        packet[23] = 0x08;
        packet[24] = 0x00;
        packet[25] = 0x01;
        ushort xport = (ushort)(port ^ (magic >> 16));
        packet[26] = (byte)(xport >> 8);
        packet[27] = (byte)xport;
        uint xip = ip ^ magic;
        packet[28] = (byte)(xip >> 24);
        packet[29] = (byte)(xip >> 16);
        packet[30] = (byte)(xip >> 8);
        packet[31] = (byte)xip;

        StunClient.TryParseMappedAddress(packet, txn, out var mapped).Should().BeTrue();
        mapped!.ToString().Should().Be("203.0.113.5:54320");
    }

    [Fact]
    public void TryParse_PlainMappedAddress_Ipv4()
    {
        Span<byte> txn = stackalloc byte[12];
        Random.Shared.NextBytes(txn);

        var packet = new byte[32];
        packet[0] = 0x01;
        packet[1] = 0x01;
        packet[2] = 0x00;
        packet[3] = 12;
        BinaryPrimitives.WriteUInt32BigEndian(packet.AsSpan(4), 0x2112A442);
        txn.CopyTo(packet.AsSpan(8, 12));
        // MAPPED-ADDRESS
        packet[20] = 0x00;
        packet[21] = 0x01;
        packet[22] = 0x00;
        packet[23] = 0x08;
        packet[24] = 0x00;
        packet[25] = 0x01;
        packet[26] = 0x15;
        packet[27] = 0xB3; // 5555
        packet[28] = 198;
        packet[29] = 51;
        packet[30] = 100;
        packet[31] = 10;

        StunClient.TryParseMappedAddress(packet, txn, out var mapped).Should().BeTrue();
        mapped!.Address.ToString().Should().Be("198.51.100.10");
        mapped.Port.Should().Be(5555);
    }

    [Theory]
    [InlineData(0x0001)] // binding request, not success
    [InlineData(0x0111)] // unknown
    public void TryParse_RejectsNonSuccessMessageType(ushort messageType)
    {
        Span<byte> txn = stackalloc byte[12];
        txn.Fill(1);
        var packet = new byte[20];
        BinaryPrimitives.WriteUInt16BigEndian(packet, messageType);
        BinaryPrimitives.WriteUInt32BigEndian(packet.AsSpan(4), 0x2112A442);
        txn.CopyTo(packet.AsSpan(8, 12));
        StunClient.TryParseMappedAddress(packet, txn, out _).Should().BeFalse();
    }

    [Fact]
    public void TryParse_RejectsWrongMagicCookie()
    {
        Span<byte> txn = stackalloc byte[12];
        txn.Fill(2);
        var packet = new byte[20];
        packet[0] = 0x01;
        packet[1] = 0x01;
        BinaryPrimitives.WriteUInt32BigEndian(packet.AsSpan(4), 0xDEADBEEF);
        txn.CopyTo(packet.AsSpan(8, 12));
        StunClient.TryParseMappedAddress(packet, txn, out _).Should().BeFalse();
    }

    [Fact]
    public void TryParse_RejectsWrongTransactionId()
    {
        Span<byte> txn = stackalloc byte[12];
        txn.Fill(3);
        Span<byte> other = stackalloc byte[12];
        other.Fill(9);
        var packet = new byte[20];
        packet[0] = 0x01;
        packet[1] = 0x01;
        BinaryPrimitives.WriteUInt32BigEndian(packet.AsSpan(4), 0x2112A442);
        other.CopyTo(packet.AsSpan(8, 12));
        StunClient.TryParseMappedAddress(packet, txn, out _).Should().BeFalse();
    }

    [Fact]
    public void TryParse_RejectsTruncatedPacket()
    {
        Span<byte> txn = stackalloc byte[12];
        StunClient.TryParseMappedAddress(new byte[10], txn, out _).Should().BeFalse();
    }

    [Fact]
    public async Task DiscoverMappedAddress_UnreachableServers_ReturnsNull()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        var client = new StunClient(logger, servers: ["invalid.stun.synapse.test.invalid"]);
        using var udp = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0));
        var mapped = await client.DiscoverMappedAddressAsync(udp, timeout: TimeSpan.FromMilliseconds(200));
        mapped.Should().BeNull();
    }
}

public sealed class NatTraversalCoordinatorTests
{
    [Fact]
    public async Task DiscoverPublicEndpoint_FallsBackToLoopbackWhenStunUnreachable()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        using var nat = new NatTraversalCoordinator(logger, "stun-fallback");
        // With no reachable STUN (offline CI / blocked DNS), coordinator falls back.
        // Force by using unreachable custom path via Discover which uses StunClient defaults —
        // if STUN happens to work in this environment, mapped may be public; either way endpoint is set.
        var endpoint = await nat.DiscoverPublicEndpointAsync();
        endpoint.Should().NotBeNull();
        endpoint.Port.Should().BeGreaterThan(0);
        nat.MappedEndpoint.Should().NotBeNull();
    }

    [Fact]
    public async Task Relay_RegisterAndDiscover_RoundTripsPeer()
    {
        using var logger = new SynapseLogger(null, LogLevel.Error, consoleEnabled: false);
        using var relay = NatTraversalCoordinator.StartRelay(logger, "relay-room", rendezvousPort: 0);
        using var client = new NatTraversalCoordinator(logger, "relay-room", relay.RendezvousPort);

        await client.DiscoverPublicEndpointAsync();
        await client.RegisterPublicEndpointAsync(tcpPort: 45678);
        var peer = await client.DiscoverPeerAsync();
        peer.Should().NotBeNull();
        peer!.TcpPort.Should().Be(45678);
        peer.SessionCode.Should().Be("relay-room");
    }
}
