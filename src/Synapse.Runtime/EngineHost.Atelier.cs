using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Core.NEAT;
using GDNN.Sentience;
using Synapse.Infrastructure;
using Synapse.Simulation.DigitalTwins;

namespace Synapse.Runtime;

/// <summary>
/// First-class atelier surfaces: law marketplace, glTF export, digital twins,
/// behavior-tree inspection, and NEAT-G genome export — owned by <see cref="EngineHost"/>.
/// </summary>
public sealed partial class EngineHost
{
    private static readonly JsonSerializerOptions TwinSnapshotJsonOptions = new() { WriteIndented = true };

    private readonly LawMarketplace _lawMarketplace = new();
    private string _marketplaceStatus = "Marketplace : prêt";
    private string _twinStatus = "Jumeaux : —";

    public LawMarketplace LawMarketplace => _lawMarketplace;
    public IDigitalTwinSynchronizer TwinSync => _twins;
    public string MarketplaceStatusText => _marketplaceStatus;
    public string TwinStatusText => _twinStatus;

    public async Task<SynapseLawPackage> ImportLawPackageAsync(string path, bool compileAndApply = true, CancellationToken ct = default)
    {
        InitializeModules();
        var package = await _lawMarketplace.ImportAsync(path, ct).ConfigureAwait(false);
        _marketplaceStatus = $"Importé : {package.Id} v{package.Version}";
        RaiseInspector("Marketplace", "Import", $"{package.Name} ({package.Id})");

        if (compileAndApply)
        {
            var result = CompileLaw(package.Id, package.Expression);
            if (!result.Success)
                _marketplaceStatus = $"Importé mais compile échoué : {result.Message}";
        }

        return package;
    }

    public async Task ExportLawPackageAsync(string lawId, string path, CancellationToken ct = default)
    {
        InitializeModules();
        ArgumentException.ThrowIfNullOrWhiteSpace(lawId);

        var fromMarket = _lawMarketplace.FindById(lawId);
        SynapseLawPackage package;
        if (fromMarket != null)
        {
            package = fromMarket;
        }
        else
        {
            var entry = _lawCompiler!.Library.GetLaw(lawId);
            if (entry == null && !string.Equals(lawId, _activeLawId, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"Unknown law '{lawId}'.");

            package = new SynapseLawPackage
            {
                Id = lawId,
                Name = entry?.Name ?? lawId,
                Category = entry?.Category ?? "custom",
                Description = entry?.Description ?? "",
                Expression = entry?.Expression ?? _scene.ActiveLawExpression ?? "",
                Author = SynapseProduct.Name,
                Version = SynapseProduct.Version,
                Tags = { "studio-export" }
            };
        }

        await _lawMarketplace.ExportAsync(package, path, ct).ConfigureAwait(false);
        _marketplaceStatus = $"Exporté : {package.Id} → {Path.GetFileName(path)}";
        RaiseInspector("Marketplace", "Export", _marketplaceStatus);
    }

    public Task ExportActiveLawPackageAsync(string path, CancellationToken ct = default)
    {
        var id = _activeLawId ?? throw new InvalidOperationException("No active law to export.");
        return ExportLawPackageAsync(id, path, ct);
    }

    public int LoadMarketplaceCatalog(string directory)
    {
        int count = _lawMarketplace.LoadCatalogFromDirectory(directory);
        _marketplaceStatus = $"Catalogue : {count} package(s)";
        return count;
    }

    public IReadOnlyList<SynapseLawPackage> ListMarketplaceLaws() => _lawMarketplace.Catalog;

    public Task<SceneGlTFExporter.SceneExportResult> ExportSceneGlTFAsync(string outputPath, CancellationToken ct = default)
    {
        InitializeModules();
        return SceneGlTFExporter.ExportAsync(_scene, outputPath, _meshProvider, ct);
    }

    public IDigitalTwin RegisterTwin(string physicalId, IReadOnlyDictionary<string, object>? properties = null)
    {
        InitializeModules();
        ArgumentException.ThrowIfNullOrWhiteSpace(physicalId);
        var twin = new InMemoryDigitalTwin { PhysicalId = physicalId };
        if (properties != null)
        {
            foreach (var kv in properties)
                twin.SetProperty(kv.Key, kv.Value);
        }

        _twins.Register(twin);
        RefreshTwinStatus();
        RaiseInspector("Twins", "Register", physicalId);
        return twin;
    }

    public async Task ExportTwinSnapshotAsync(Guid twinId, string path, CancellationToken ct = default)
    {
        InitializeModules();
        var twin = _twins.GetById(twinId)
                   ?? throw new InvalidOperationException($"Twin '{twinId}' not found.");
        var snapshot = twin.TakeSnapshot();
        var payload = new Dictionary<string, object?>
        {
            ["twinId"] = snapshot.TwinId,
            ["physicalId"] = twin.PhysicalId,
            ["timestamp"] = snapshot.Timestamp,
            ["properties"] = snapshot.Properties
        };
        var full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        await File.WriteAllTextAsync(
            full,
            JsonSerializer.Serialize(payload, TwinSnapshotJsonOptions),
            ct).ConfigureAwait(false);
        _twinStatus = $"Snapshot → {Path.GetFileName(full)}";
        RaiseInspector("Twins", "Export", _twinStatus);
    }

    public async Task SynchronizeTwinsAsync(CancellationToken ct = default)
    {
        InitializeModules();
        await _twins.SynchronizeAllAsync(ct).ConfigureAwait(false);
        RefreshTwinStatus();
        RaiseInspector("Twins", "Sync", _twinStatus);
    }

    public IReadOnlyList<IDigitalTwin> ListTwins()
    {
        InitializeModules();
        return _twins.Twins.Values.ToList();
    }

    /// <summary>ASCII visualization of the selected agent's behavior tree, if any.</summary>
    public string? GetAgentBehaviorTreeText(Guid sceneEntityId)
    {
        InitializeModules();
        var agent = _sentience?.GetEntity(sceneEntityId);
        var root = agent?.BehaviorTree?.Root;
        if (root == null)
            return null;
        return _sentience!.Debugger.VisualizeBehaviorTree(root);
    }

    public IReadOnlyList<string> ListAgentBehaviorTrees()
    {
        InitializeModules();
        if (_sentience == null)
            return Array.Empty<string>();

        return _sentience.GetAllEntities()
            .Where(e => e.BehaviorTree != null)
            .Select(e => $"{e.EntityId.ToString()[..8]} · {e.BehaviorTree!.Name}")
            .ToList();
    }

    public string? GetSentienceSystemReportText()
    {
        InitializeModules();
        if (_sentience == null)
            return null;
        var report = _sentience.GetSystemReport();
        return string.Join(Environment.NewLine, report.Select(kv => $"{kv.Key}: {kv.Value}"));
    }

    public GeoGenome? GetBestGenome()
    {
        InitializeModules();
        return _evolution?.GetBestGenome();
    }

    public string? ExportBestGenomeJson()
    {
        InitializeModules();
        var genome = _evolution?.GetBestGenome();
        if (genome == null)
            return null;
        var serializer = new GenomeSerializer();
        return serializer.SerializeToJson(genome);
    }

    public async Task ExportBestGenomeAsync(string path, CancellationToken ct = default)
    {
        var json = ExportBestGenomeJson()
                   ?? throw new InvalidOperationException("No evolved genome available yet.");
        var full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        await File.WriteAllTextAsync(full, json, ct).ConfigureAwait(false);
        RaiseInspector("Evolution", "Genome export", Path.GetFileName(full));
    }

    public string? ExportEvolutionStateJson(bool includeEvents = false)
    {
        InitializeModules();
        return _evolution?.ExportStateJson(includeEvents);
    }

    public IReadOnlyList<EvolutionMetrics> GetEvolutionMetricsHistory()
    {
        InitializeModules();
        return _evolution?.GetMetricsHistory() ?? Array.Empty<EvolutionMetrics>();
    }

    private void RefreshTwinStatus()
    {
        int n = _twins.Twins.Count;
        _twinStatus = n == 0 ? "Jumeaux : aucun" : $"Jumeaux : {n} enregistré(s)";
    }
}
