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
    private readonly List<SynapseLawPackage> _catalog = new();

    public IReadOnlyList<SynapseLawPackage> Catalog => _catalog;

    public async Task<SynapseLawPackage> ImportAsync(string path, CancellationToken ct = default)
    {
        var full = Path.GetFullPath(path);
        if (!full.EndsWith(".synapse-law", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Expected .synapse-law package file.");

        await using var stream = File.OpenRead(full);
        var package = await JsonSerializer.DeserializeAsync(stream, LawMarketplaceJsonContext.Default.SynapseLawPackage, ct)
                      ?? throw new InvalidDataException("Invalid law package.");

        if (string.IsNullOrWhiteSpace(package.Id))
            package.Id = Path.GetFileNameWithoutExtension(full);

        _catalog.Add(package);
        return package;
    }

    public async Task ExportAsync(SynapseLawPackage package, string path, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(package);
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
        foreach (var file in Directory.EnumerateFiles(directory, "*.synapse-law", SearchOption.AllDirectories))
        {
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
}

[JsonSerializable(typeof(SynapseLawPackage))]
[JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class LawMarketplaceJsonContext : JsonSerializerContext;
