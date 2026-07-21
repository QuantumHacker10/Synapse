using System.Net;
using Synapse.Infrastructure.Logging;

namespace Synapse.Network;

/// <summary>STUN/TURN options for symmetric-NAT traversal.</summary>
public sealed class NatIceOptions
{
    /// <summary>STUN host (e.g. stun.l.google.com). Port defaults to 3478 unless host contains :port.</summary>
    public string? StunServer { get; set; }

    /// <summary>TURN host:port or host (default port 3478).</summary>
    public string? TurnServer { get; set; }

    public string? TurnUsername { get; set; }
    public string? TurnPassword { get; set; }

    /// <summary>When true, prefer advertising TURN relayed candidate over STUN/TCP.</summary>
    public bool PreferTurn { get; set; }

    public bool HasStun => !string.IsNullOrWhiteSpace(StunServer);
    public bool HasTurn =>
        !string.IsNullOrWhiteSpace(TurnServer) &&
        !string.IsNullOrWhiteSpace(TurnUsername) &&
        !string.IsNullOrWhiteSpace(TurnPassword);

    public static (string Host, int Port) ParseHostPort(string value, int defaultPort)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        var trimmed = value.Trim();
        int idx = trimmed.LastIndexOf(':');
        if (idx > 0 && idx < trimmed.Length - 1 &&
            int.TryParse(trimmed.AsSpan(idx + 1), out var port) &&
            port > 0 && port < 65536 &&
            !trimmed.Contains(']', StringComparison.Ordinal)) // crude IPv6 skip
        {
            return (trimmed[..idx], port);
        }

        return (trimmed, defaultPort);
    }
}

/// <summary>Resolved ICE-lite candidates after STUN and/or TURN probes.</summary>
public sealed class NatIceCandidates
{
    public IPEndPoint? StunMapped { get; init; }
    public IPEndPoint? TurnRelayed { get; init; }
    public TurnClient? Turn { get; init; }
    public string Mode { get; init; } = "tcp"; // tcp | stun | turn

    public IPAddress? AdvertisedAddress =>
        Mode == "turn" ? TurnRelayed?.Address :
        StunMapped?.Address;

    public int? AdvertisedPort =>
        Mode == "turn" ? TurnRelayed?.Port : null; // null → keep TCP listen port
}

/// <summary>Runs STUN Binding and optional TURN Allocate for WAN hosts.</summary>
public static class NatIceAssist
{
    public static async Task<NatIceCandidates> GatherAsync(
        NatIceOptions? options,
        ISynapseLogger? logger = null,
        CancellationToken ct = default)
    {
        options ??= new NatIceOptions();
        IPEndPoint? stunMapped = null;
        TurnClient? turn = null;
        IPEndPoint? relayed = null;

        if (options.HasStun)
        {
            var (host, port) = NatIceOptions.ParseHostPort(options.StunServer!, 3478);
            stunMapped = await StunClient.QueryMappedAddressAsync(host, port, ct: ct).ConfigureAwait(false);
            logger?.Info("Network", stunMapped != null
                ? $"STUN mapped {stunMapped}"
                : $"STUN failed for {host}:{port}");
        }

        if (options.HasTurn)
        {
            var (host, port) = NatIceOptions.ParseHostPort(options.TurnServer!, 3478);
            turn = await TurnClient.TryAllocateAsync(
                host, port, options.TurnUsername!, options.TurnPassword!, ct: ct).ConfigureAwait(false);
            relayed = turn?.RelayedEndpoint;
            logger?.Info("Network", relayed != null
                ? $"TURN relayed {relayed}"
                : $"TURN allocate failed for {host}:{port}");
        }

        string mode = "tcp";
        if (options.PreferTurn && relayed != null)
            mode = "turn";
        else if (relayed != null && stunMapped == null)
            mode = "turn";
        else if (stunMapped != null)
            mode = "stun";

        return new NatIceCandidates
        {
            StunMapped = stunMapped,
            TurnRelayed = relayed,
            Turn = turn,
            Mode = mode
        };
    }
}
