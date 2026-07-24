using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace Synapse.Runtime;

/// <summary>Marketplace package format for shareable living laws (.synapse-law).</summary>
public sealed class SynapseLawPackage
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Version { get; set; } = "1.0";
    public string Author { get; set; } = "";
    public string Category { get; set; } = "general";
    public string Expression { get; set; } = "";
    public string Description { get; set; } = "";
    public Dictionary<string, string> Parameters { get; set; } = new();
    public Dictionary<string, float> Constants { get; set; } = new();
    public List<string> Tags { get; set; } = new();
}

public sealed class LawMarketplace
{
    public const int MaxExpressionLength = 8_192;
    public const int MaxDescriptionLength = 4_096;
    public const int MaxPackageBytes = 256 * 1024;
    public const int MaxCatalogEntries = 2_048;
    public const int MaxDirectoryFiles = 512;

    private readonly List<SynapseLawPackage> _catalog = new();

    public IReadOnlyList<SynapseLawPackage> Catalog => _catalog;

    public async Task<SynapseLawPackage> ImportAsync(string path, CancellationToken ct = default)
    {
        var full = Path.GetFullPath(path);
        if (!full.EndsWith(".synapse-law", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Expected .synapse-law package file.");

        var info = new FileInfo(full);
        if (!info.Exists)
            throw new FileNotFoundException("Law package not found.", full);
        if (info.Length > MaxPackageBytes)
            throw new InvalidDataException($"Law package exceeds {MaxPackageBytes} byte limit.");

        await using var stream = File.OpenRead(full);
        var package = await JsonSerializer.DeserializeAsync(stream, LawMarketplaceJsonContext.Default.SynapseLawPackage, ct)
                      ?? throw new InvalidDataException("Invalid law package.");

        if (string.IsNullOrWhiteSpace(package.Id))
            package.Id = Path.GetFileNameWithoutExtension(full);

        ValidatePackage(package);

        if (_catalog.Count >= MaxCatalogEntries)
            throw new InvalidOperationException($"Marketplace catalog exceeds {MaxCatalogEntries} entries.");

        var existing = _catalog.FindIndex(p => string.Equals(p.Id, package.Id, StringComparison.OrdinalIgnoreCase));
        if (existing >= 0)
            _catalog[existing] = package;
        else
            _catalog.Add(package);

        return package;
    }

    public async Task ExportAsync(SynapseLawPackage package, string path, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(package);
        ValidatePackage(package);
        var full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        await using var stream = File.Create(full);
        await JsonSerializer.SerializeAsync(stream, package, LawMarketplaceJsonContext.Default.SynapseLawPackage, ct);
    }

    public int LoadCatalogFromDirectory(string directory)
    {
        if (!Directory.Exists(directory))
            return 0;

        int count = 0;
        int seen = 0;
        foreach (var file in Directory.EnumerateFiles(directory, "*.synapse-law", SearchOption.AllDirectories))
        {
            if (++seen > MaxDirectoryFiles)
                throw new InvalidDataException(
                    $"Too many .synapse-law files under '{directory}' (max {MaxDirectoryFiles}).");

            ImportAsync(file).GetAwaiter().GetResult();
            count++;
        }
        return count;
    }

    public SynapseLawPackage? FindById(string id) =>
        _catalog.FirstOrDefault(p => string.Equals(p.Id, id, StringComparison.OrdinalIgnoreCase));

    public IReadOnlyList<SynapseLawPackage> Search(string query) =>
        _catalog.Where(p =>
            p.Name.Contains(query, StringComparison.OrdinalIgnoreCase) ||
            p.Tags.Any(t => t.Contains(query, StringComparison.OrdinalIgnoreCase))).ToList();

    public static void ValidatePackage(SynapseLawPackage package)
    {
        ArgumentNullException.ThrowIfNull(package);

        if (string.IsNullOrWhiteSpace(package.Id))
            throw new InvalidDataException("Law package Id is required.");
        if (package.Id.Length > 128 || package.Id.Contains('/') || package.Id.Contains('\\') || package.Id.Contains(".."))
            throw new InvalidDataException("Law package Id is invalid.");
        if (string.IsNullOrWhiteSpace(package.Name))
            throw new InvalidDataException("Law package Name is required.");
        if (string.IsNullOrWhiteSpace(package.Expression))
            throw new InvalidDataException("Law package Expression is required.");
        if (package.Expression.Length > MaxExpressionLength)
            throw new InvalidDataException($"Law expression exceeds {MaxExpressionLength} characters.");
        if (package.Description.Length > MaxDescriptionLength)
            throw new InvalidDataException($"Law description exceeds {MaxDescriptionLength} characters.");
        if (package.Parameters.Count > 64 || package.Constants.Count > 64 || package.Tags.Count > 32)
            throw new InvalidDataException("Law package metadata collections exceed size limits.");

        foreach (var key in package.Parameters.Keys)
        {
            if (string.IsNullOrWhiteSpace(key) || key.Length > 64)
                throw new InvalidDataException("Law parameter key is invalid.");
        }

        foreach (var tag in package.Tags)
        {
            if (string.IsNullOrWhiteSpace(tag) || tag.Length > 64)
                throw new InvalidDataException("Law tag is invalid.");
        }
    }
}

[JsonSerializable(typeof(SynapseLawPackage))]
[JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class LawMarketplaceJsonContext : JsonSerializerContext;
