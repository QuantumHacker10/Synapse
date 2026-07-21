using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Core.Security;
using Synapse.Infrastructure.Logging;

namespace Synapse.Plugins;

/// <summary>
/// Remote plugin marketplace: HTTPS catalog fetch + hash-verified download into a jailed plugin directory.
/// </summary>
public sealed class RemotePluginMarketplace
{
    private readonly ISynapseLogger _logger;
    private readonly HttpClient? _ownedClient;

    public RemotePluginMarketplace(ISynapseLogger logger, HttpClient? httpClient = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _ownedClient = httpClient;
    }

    /// <summary>Fetches a remote catalog JSON (array of <see cref="PluginCatalogEntry"/>).</summary>
    public async Task<IReadOnlyList<PluginCatalogEntry>> FetchCatalogAsync(
        string catalogUrl,
        IReadOnlyCollection<string>? extraAllowedHosts = null,
        CancellationToken ct = default)
    {
        var uri = UrlSecurity.ValidateOutboundUri(catalogUrl, allowLoopbackHttp: true, extraAllowedHosts);
        using var client = _ownedClient ?? UrlSecurity.CreateSafeHttpClient(TimeSpan.FromSeconds(60));
        var json = await client.GetStringAsync(uri, ct).ConfigureAwait(false);
        var entries = JsonSerializer.Deserialize<List<PluginCatalogEntry>>(json) ?? new();
        return entries.Where(e => !string.IsNullOrWhiteSpace(e.Id)).ToList();
    }

    /// <summary>
    /// Downloads catalog entries that have <see cref="PluginCatalogEntry.DownloadUrl"/> into
    /// <paramref name="pluginDirectory"/> (path-jailed), verifying SHA-256 when provided.
    /// </summary>
    public async Task<int> InstallAsync(
        IEnumerable<PluginCatalogEntry> entries,
        string pluginDirectory,
        IReadOnlyCollection<string>? extraAllowedHosts = null,
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pluginDirectory);
        var root = Path.GetFullPath(pluginDirectory);
        Directory.CreateDirectory(root);

        int installed = 0;
        using var client = _ownedClient ?? UrlSecurity.CreateSafeHttpClient(TimeSpan.FromSeconds(120));
        foreach (var entry in entries)
        {
            if (string.IsNullOrWhiteSpace(entry.DownloadUrl) || string.IsNullOrWhiteSpace(entry.FileName))
                continue;

            var dest = Path.GetFullPath(Path.Combine(root, entry.FileName));
            if (!PluginHost.IsUnderRoot(dest, root))
            {
                _logger.Warn("Plugins", $"Remote install blocked (path jail): {entry.FileName}");
                continue;
            }

            if (PluginHost.IsBlockedRemotePath(dest))
            {
                _logger.Warn("Plugins", $"Remote install blocked (remote path): {entry.FileName}");
                continue;
            }

            try
            {
                var uri = UrlSecurity.ValidateOutboundUri(entry.DownloadUrl, allowLoopbackHttp: true, extraAllowedHosts);
                var bytes = await client.GetByteArrayAsync(uri, ct).ConfigureAwait(false);
                await File.WriteAllBytesAsync(dest, bytes, ct).ConfigureAwait(false);

                if (!string.IsNullOrWhiteSpace(entry.Sha256))
                {
                    var actual = PluginMarketplace.ComputeFileSha256(dest);
                    if (!string.Equals(actual, entry.Sha256, StringComparison.OrdinalIgnoreCase))
                    {
                        File.Delete(dest);
                        _logger.Warn("Plugins", $"Hash mismatch for remote {entry.Id}: expected {entry.Sha256}, got {actual}");
                        continue;
                    }
                }

                installed++;
                _logger.Info("Plugins", $"Installed remote plugin {entry.Id} → {entry.FileName}");
            }
            catch (Exception ex)
            {
                _logger.Warn("Plugins", $"Remote install failed for {entry.Id}: {ex.Message}");
            }
        }

        // Refresh local catalog snapshot
        var localEntries = entries.Where(e => !string.IsNullOrWhiteSpace(e.FileName)).ToList();
        if (localEntries.Count > 0)
            PluginMarketplace.WriteCatalogTemplate(root, localEntries);

        return installed;
    }

    /// <summary>Fetch catalog then install all entries with download URLs.</summary>
    public async Task<int> SyncAsync(
        string catalogUrl,
        string pluginDirectory,
        IReadOnlyCollection<string>? extraAllowedHosts = null,
        CancellationToken ct = default)
    {
        var catalog = await FetchCatalogAsync(catalogUrl, extraAllowedHosts, ct).ConfigureAwait(false);
        return await InstallAsync(catalog, pluginDirectory, extraAllowedHosts, ct).ConfigureAwait(false);
    }
}
