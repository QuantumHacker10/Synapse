using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Synapse.Infrastructure.Logging;

namespace Synapse.Plugins;

/// <summary>
/// Local plugin marketplace catalog: <c>marketplace.json</c> + optional SHA-256 allowlist.
/// Hosted remote marketplace remains a separate product surface; this is the production-safe local gate.
/// </summary>
public sealed class PluginMarketplace
{
    private readonly string _directory;
    private readonly ISynapseLogger _logger;
    private readonly List<PluginCatalogEntry> _entries;

    private PluginMarketplace(string directory, ISynapseLogger logger, List<PluginCatalogEntry> entries)
    {
        _directory = directory;
        _logger = logger;
        _entries = entries;
    }

    public IReadOnlyList<PluginCatalogEntry> Entries => _entries;

    public static PluginMarketplace FromDirectory(string directory, ISynapseLogger logger)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        var full = Path.GetFullPath(directory);
        var entries = new List<PluginCatalogEntry>();
        var manifest = Path.Combine(full, "marketplace.json");
        if (File.Exists(manifest))
        {
            try
            {
                var json = File.ReadAllText(manifest);
                var parsed = JsonSerializer.Deserialize<List<PluginCatalogEntry>>(json) ?? new();
                entries.AddRange(parsed.Where(e => !string.IsNullOrWhiteSpace(e.Id)));
            }
            catch (Exception ex)
            {
                logger.Warn("Plugins", $"marketplace.json unreadable: {ex.Message}");
            }
        }

        return new PluginMarketplace(full, logger, entries);
    }

    /// <summary>Writes a catalog entry template next to the plugin directory.</summary>
    public static void WriteCatalogTemplate(string directory, IEnumerable<PluginCatalogEntry> entries)
    {
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "marketplace.json");
        var json = JsonSerializer.Serialize(entries.ToList(), new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    public static string ComputeFileSha256(string filePath)
    {
        var hash = SHA256.HashData(File.ReadAllBytes(filePath));
        var sb = new StringBuilder(hash.Length * 2);
        foreach (var b in hash)
            sb.Append(b.ToString("x2"));
        return sb.ToString();
    }

    /// <summary>Verifies on-disk DLLs against catalog hashes when present; warns otherwise.</summary>
    public void VerifyInstalledOrWarn()
    {
        if (_entries.Count == 0)
        {
            _logger.Info("Plugins", $"No marketplace.json in {_directory} (optional).");
            return;
        }

        foreach (var entry in _entries)
        {
            if (string.IsNullOrWhiteSpace(entry.FileName))
                continue;
            var path = Path.Combine(_directory, entry.FileName);
            if (!File.Exists(path))
            {
                _logger.Warn("Plugins", $"Catalog plugin missing: {entry.Id} ({entry.FileName})");
                continue;
            }

            if (string.IsNullOrWhiteSpace(entry.Sha256))
                continue;

            var actual = ComputeFileSha256(path);
            if (!string.Equals(actual, entry.Sha256, StringComparison.OrdinalIgnoreCase))
                _logger.Warn("Plugins", $"Hash mismatch for {entry.Id}: expected {entry.Sha256}, got {actual}");
            else
                _logger.Info("Plugins", $"Verified {entry.Id} ({entry.FileName})");
        }
    }
}

public sealed class PluginCatalogEntry
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Version { get; set; } = "1.0.0";
    public string FileName { get; set; } = "";
    public string Sha256 { get; set; } = "";
    public string Description { get; set; } = "";
}
