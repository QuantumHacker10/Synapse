using System;
using System.Collections.Generic;
using System.Diagnostics;
using GDNN.Core.DataStructures;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Polygonization;

/// <summary>Options du pipeline de géométrie neuronale.</summary>
public sealed class NeuralGeometryPipelineOptions
{
    public int BaseResolution { get; set; } = 32;
    public int LevelCount { get; set; } = 3;
    public float PixelErrorBudget { get; set; } = 1.0f;

    /// <summary>
    /// Pendant qu'un flux d'édits draine, ré-extraire au plus une fois tous les
    /// N ticks (la ré-extraction finale a lieu dès que la file est vide).
    /// </summary>
    public int RebuildIntervalTicks { get; set; } = 8;

    /// <summary>
    /// Répertoire du cache disque des chaînes polygonisées (null = désactivé).
    /// Un asset statique dont le réseau n'a pas changé recharge sa chaîne
    /// depuis le disque au lieu de repolygoniser au démarrage.
    /// </summary>
    public string? CacheDirectory { get; set; }
}

/// <summary>Rapport d'un tick du pipeline — toutes les métriques d'une frame.</summary>
public sealed class PipelineFrameReport
{
    public TrainingSliceReport Training { get; init; }
    public bool Rebuilt { get; init; }
    public double RebuildMs { get; init; }

    /// <summary>La chaîne de ce tick provient-elle du cache disque ?</summary>
    public bool LoadedFromCache { get; init; }
    public ClusterCullStats Culling { get; init; }
    public IReadOnlyList<NeuralMeshlet> VisibleClusters { get; init; }
    public long ExtractedGeometryVersion { get; init; }

    public override string ToString() =>
        $"[train {Training}] [rebuild={(Rebuilt ? $"{RebuildMs:F1}ms" : "non")}] [{Culling}]";
}

/// <summary>
/// Pipeline de géométrie neuronale temps réel, à tick-er chaque frame :
///
/// 1. tranche d'entraînement SGD budgétée (édits sculpt + replay anti-oubli) ;
/// 2. si la version de géométrie a changé, ré-extraction éparse de la chaîne de
///    LOD polygonale depuis le réseau (throttlée pendant un flux d'édits) ;
/// 3. sélection des clusters visibles (LOD par erreur d'écran garantie,
///    culling frustum hiérarchique, culling backface par cône).
///
/// La boucle complète — apprendre, re-polygoniser, culler — est mesurée dans le
/// rapport de frame ; les gains se démontrent au banc, pas sur l'étiquette.
/// </summary>
public sealed class NeuralGeometryPipeline
{
    private readonly HashEncodedDeepMLP _network;
    private readonly AABB _bounds;
    private readonly NeuralGeometryPipelineOptions _options;
    private readonly PolygonizationCache? _cache;
    private NeuralPolygonLodChain _chain;
    private NeuralClusterRenderer _renderer;
    private long _extractedVersion;
    private int _ticksSinceRebuild;
    private bool _chainFromCache;

    public OnlineSdfTrainer Trainer { get; }
    public NeuralPolygonizer Polygonizer { get; }
    public SoftwareRasterizer Rasterizer { get; } = new();
    public NeuralPolygonLodChain Chain => _chain;

    /// <summary>La chaîne initiale a-t-elle été rechargée depuis le cache disque ?</summary>
    public bool InitialChainFromCache { get; }

    public NeuralGeometryPipeline(
        HashEncodedDeepMLP network, AABB bounds, NeuralGeometryPipelineOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(network);
        _network = network;
        _bounds = bounds;
        _options = options ?? new NeuralGeometryPipelineOptions();
        _cache = _options.CacheDirectory is { Length: > 0 } dir
            ? new PolygonizationCache(dir)
            : null;

        Trainer = new OnlineSdfTrainer(network, bounds);
        Polygonizer = new NeuralPolygonizer(); // épars par défaut

        if (_cache != null &&
            _cache.TryLoad(ComputeCacheKey(), out var cachedChain))
        {
            _chain = cachedChain!;
            InitialChainFromCache = true;
            _chainFromCache = true;
        }
        else
        {
            _chain = BuildChain();
            StoreChainInCache();
        }

        _renderer = new NeuralClusterRenderer(_chain);
        _extractedVersion = Trainer.GeometryVersion;
    }

    /// <summary>Un tick de frame : entraînement budgété, ré-extraction si besoin, culling.</summary>
    public PipelineFrameReport Tick(in CameraView camera, double trainBudgetMs = 2.0)
    {
        var training = Trainer.TrainSlice(trainBudgetMs);
        _ticksSinceRebuild++;

        bool rebuilt = false;
        double rebuildMs = 0;
        if (Trainer.GeometryVersion != _extractedVersion && ShouldRebuildNow())
        {
            var sw = Stopwatch.StartNew();
            _chain = BuildChain();
            _renderer = new NeuralClusterRenderer(_chain);
            sw.Stop();

            _extractedVersion = Trainer.GeometryVersion;
            _ticksSinceRebuild = 0;
            rebuilt = true;
            rebuildMs = sw.Elapsed.TotalMilliseconds;
            _chainFromCache = false;
            Trainer.TryConsumeDirtyBounds(out _);

            // Géométrie stabilisée (file d'édits vide) : la ré-extraction est
            // définitive pour ces poids, on la persiste pour le prochain run.
            if (Trainer.PendingCount == 0)
                StoreChainInCache();
        }

        var visible = _renderer.SelectVisible(camera, out var stats, _options.PixelErrorBudget);

        return new PipelineFrameReport
        {
            Training = training,
            Rebuilt = rebuilt,
            RebuildMs = rebuildMs,
            LoadedFromCache = _chainFromCache,
            Culling = stats,
            VisibleClusters = visible,
            ExtractedGeometryVersion = _extractedVersion
        };
    }

    /// <summary>
    /// Passe de rendu complète : sélection de clusters (LOD + frustum + cône)
    /// puis rasterisation software dans le visibility buffer. Le payload de
    /// chaque pixel indexe la liste de clusters visibles retournée.
    /// </summary>
    public RasterStats RenderFrame(
        in CameraView camera, RasterTarget target, out IReadOnlyList<NeuralMeshlet> visible)
    {
        visible = _renderer.SelectVisible(camera, out var cullStats, _options.PixelErrorBudget);
        var levelMesh = _chain.Levels[cullStats.SelectedLevel].Mesh;
        return Rasterizer.Rasterize(target, levelMesh, visible, camera);
    }

    /// <summary>
    /// Ré-extrait dès que la file d'édits est vide (géométrie stabilisée) ;
    /// pendant un flux continu, au rythme de <see cref="NeuralGeometryPipelineOptions.RebuildIntervalTicks"/>.
    /// </summary>
    private bool ShouldRebuildNow()
        => Trainer.PendingCount == 0 || _ticksSinceRebuild >= _options.RebuildIntervalTicks;

    private NeuralPolygonLodChain BuildChain()
        => NeuralPolygonLodChain.Build(
            _network, _bounds, _options.BaseResolution, _options.LevelCount, Polygonizer);

    private string ComputeCacheKey()
        => PolygonizationCache.ComputeKey(
            _network, _bounds, _options.BaseResolution, _options.LevelCount);

    private void StoreChainInCache()
    {
        if (_cache == null)
            return;

        try
        {
            _cache.Store(ComputeCacheKey(), _chain);
        }
        catch (IOException)
        {
            // Cache disque indisponible : le pipeline reste fonctionnel sans lui.
        }
    }
}
