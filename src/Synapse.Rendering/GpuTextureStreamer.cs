using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Core.Security;
using Synapse.Infrastructure.Logging;

namespace GDNN.Rendering.Streaming;

/// <summary>
/// GPU-oriented texture streamer: async page-in of texture bytes (disk or HTTPS),
/// LRU residency, and mip-level hints. Does not bind Vulkan images itself — consumers
/// upload <see cref="StreamedTexturePage.Bytes"/> to the RHI.
/// </summary>
public sealed class GpuTextureStreamer : IAsyncDisposable
{
    private readonly ISynapseLogger? _logger;
    private readonly ConcurrentDictionary<string, StreamedTexturePage> _resident = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentQueue<string> _lru = new();
    private long _bytesResident;
    private int _hits;
    private int _misses;

    public GpuTextureStreamer(ISynapseLogger? logger = null, long maxResidentBytes = 256L * 1024 * 1024)
    {
        _logger = logger;
        MaxResidentBytes = Math.Max(8L * 1024 * 1024, maxResidentBytes);
    }

    public long MaxResidentBytes { get; }
    public long BytesResident => Interlocked.Read(ref _bytesResident);
    public int ResidentCount => _resident.Count;
    public int CacheHits => _hits;
    public int CacheMisses => _misses;

    /// <summary>Requests a texture page. Returns cached page or loads from disk/URL.</summary>
    public async Task<StreamedTexturePage> RequestAsync(
        string key,
        string sourcePathOrUrl,
        int mipLevel = 0,
        IReadOnlyCollection<string>? extraAllowedHosts = null,
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        string cacheKey = $"{key}|m{mipLevel}";
        if (_resident.TryGetValue(cacheKey, out var hit))
        {
            Interlocked.Increment(ref _hits);
            hit.LastAccessUtc = DateTime.UtcNow;
            return hit;
        }

        Interlocked.Increment(ref _misses);
        byte[] bytes;
        string source = sourcePathOrUrl;
        if (Uri.TryCreate(sourcePathOrUrl, UriKind.Absolute, out var uri) &&
            (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps))
        {
            var safe = UrlSecurity.ValidateOutboundUri(sourcePathOrUrl, allowLoopbackHttp: true, extraAllowedHosts);
            using var client = UrlSecurity.CreateSafeHttpClient(TimeSpan.FromSeconds(60));
            bytes = await client.GetByteArrayAsync(safe, ct).ConfigureAwait(false);
            source = safe.ToString();
        }
        else
        {
            var path = Path.GetFullPath(sourcePathOrUrl);
            if (!File.Exists(path))
            {
                var empty = new StreamedTexturePage
                {
                    Key = cacheKey,
                    Source = path,
                    MipLevel = mipLevel,
                    Bytes = Array.Empty<byte>(),
                    LastAccessUtc = DateTime.UtcNow
                };
                _logger?.Warn("TextureStream", $"Missing texture {path}");
                return empty;
            }

            bytes = await File.ReadAllBytesAsync(path, ct).ConfigureAwait(false);
            source = path;
        }

        // Mip hint: for production streaming we keep full bytes; consumers may generate mips.
        // Optional half-res stub when mipLevel > 0 and payload is large — keep full for correctness.
        var page = new StreamedTexturePage
        {
            Key = cacheKey,
            Source = source,
            MipLevel = mipLevel,
            Bytes = bytes,
            LastAccessUtc = DateTime.UtcNow
        };

        _resident[cacheKey] = page;
        _lru.Enqueue(cacheKey);
        Interlocked.Add(ref _bytesResident, bytes.LongLength);
        await EvictIfNeededAsync().ConfigureAwait(false);
        _logger?.Debug("TextureStream", $"Paged in {cacheKey} ({bytes.Length} bytes)");
        return page;
    }

    /// <summary>Prefetches a UDIM tile set (tile → path).</summary>
    public async Task PrefetchUdimAsync(
        string materialKey,
        IReadOnlyDictionary<int, string> tiles,
        CancellationToken ct = default)
    {
        foreach (var kv in tiles.OrderBy(k => k.Key))
            await RequestAsync($"{materialKey}:{kv.Key}", kv.Value, mipLevel: 0, ct: ct).ConfigureAwait(false);
    }

    public bool TryGetResident(string key, int mipLevel, out StreamedTexturePage? page) =>
        _resident.TryGetValue($"{key}|m{mipLevel}", out page);

    public void EvictAll()
    {
        _resident.Clear();
        while (_lru.TryDequeue(out _)) { }
        Interlocked.Exchange(ref _bytesResident, 0);
    }

    private Task EvictIfNeededAsync()
    {
        while (BytesResident > MaxResidentBytes && _lru.TryDequeue(out var key))
        {
            if (_resident.TryRemove(key, out var page))
                Interlocked.Add(ref _bytesResident, -page.Bytes.LongLength);
        }

        return Task.CompletedTask;
    }

    public ValueTask DisposeAsync()
    {
        EvictAll();
        return ValueTask.CompletedTask;
    }
}

public sealed class StreamedTexturePage
{
    public string Key { get; init; } = "";
    public string Source { get; init; } = "";
    public int MipLevel { get; init; }
    public byte[] Bytes { get; init; } = Array.Empty<byte>();
    public DateTime LastAccessUtc { get; set; }
}
