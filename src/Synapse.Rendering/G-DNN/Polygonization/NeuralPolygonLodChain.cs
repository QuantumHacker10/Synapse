using System;
using System.Collections.Generic;
using System.Numerics;
using GDNN.Core.DataStructures;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Polygonization;

/// <summary>Un niveau de la chaîne de polygones neuronaux.</summary>
public sealed class NeuralPolygonLod
{
    public required int Level { get; init; }
    public required int GridResolution { get; init; }
    public required NeuralPolygonMesh Mesh { get; init; }
    public required IReadOnlyList<NeuralMeshlet> Meshlets { get; init; }

    /// <summary>Erreur géométrique world-space de ce niveau (diagonale de voxel).</summary>
    public required float GeometricError { get; init; }
}

/// <summary>
/// Chaîne de LOD polygonale ré-extraite du SDF neuronal à résolutions décroissantes.
///
/// Différence structurelle avec une chaîne de LOD de maillages figés : chaque
/// niveau est régénéré depuis la fonction continue, pas décimé depuis le niveau
/// supérieur — pas d'accumulation d'erreur de simplification, et l'erreur
/// géométrique de chaque niveau est bornée par la taille de voxel, donc la
/// sélection par erreur d'écran est une garantie et non une heuristique.
/// La sélection projette l'erreur en pixels, comme le fait la hiérarchie de
/// clusters de Nanite — c'en est l'analogue, à prouver au banc d'essai
/// (GDNNValidationProtocol), pas dans le marketing.
/// </summary>
public sealed class NeuralPolygonLodChain
{
    private readonly List<NeuralPolygonLod> _levels = [];

    public IReadOnlyList<NeuralPolygonLod> Levels => _levels;

    /// <summary>Bornes d'extraction (domaine du SDF).</summary>
    public AABB Bounds { get; private set; }

    /// <summary>
    /// Construit la chaîne : la résolution est divisée par deux à chaque niveau
    /// (LOD 0 = <paramref name="baseResolution"/>, le plus fin).
    /// </summary>
    public static NeuralPolygonLodChain Build(
        ISdfNetwork network,
        AABB bounds,
        int baseResolution = 64,
        int levelCount = 4,
        NeuralPolygonizer? polygonizer = null)
    {
        ArgumentNullException.ThrowIfNull(network);
        if (levelCount < 1)
            throw new ArgumentOutOfRangeException(nameof(levelCount));

        polygonizer ??= new NeuralPolygonizer();
        var meshletBuilder = new NeuralMeshletBuilder();
        var chain = new NeuralPolygonLodChain { Bounds = bounds };

        var resolutions = new List<int>();
        for (int level = 0, r = baseResolution; level < levelCount && r >= 2; level++, r /= 2)
            resolutions.Add(r);

        // Du plus grossier au plus fin : le LastReport du polygoniseur reflète
        // ainsi le niveau le plus détaillé (le plus coûteux, le plus parlant).
        for (int level = resolutions.Count - 1; level >= 0; level--)
        {
            var mesh = polygonizer.Extract(network, bounds, resolutions[level]);
            chain._levels.Insert(0, new NeuralPolygonLod
            {
                Level = level,
                GridResolution = resolutions[level],
                Mesh = mesh,
                Meshlets = meshletBuilder.Build(mesh),
                GeometricError = polygonizer.LastReport.GeometricError
            });
        }

        return chain;
    }

    /// <summary>
    /// Reconstruit une chaîne depuis des niveaux déjà extraits (ex : cache disque).
    /// Les niveaux sont ordonnés par <see cref="NeuralPolygonLod.Level"/> croissant.
    /// </summary>
    public static NeuralPolygonLodChain FromLevels(AABB bounds, IEnumerable<NeuralPolygonLod> levels)
    {
        ArgumentNullException.ThrowIfNull(levels);

        var chain = new NeuralPolygonLodChain { Bounds = bounds };
        chain._levels.AddRange(levels);
        chain._levels.Sort((a, b) => a.Level.CompareTo(b.Level));

        if (chain._levels.Count == 0)
            throw new ArgumentException("At least one LOD level is required.", nameof(levels));

        return chain;
    }

    /// <summary>
    /// Sélectionne le LOD le plus grossier dont l'erreur projetée à l'écran
    /// reste sous <paramref name="pixelErrorThreshold"/>.
    /// </summary>
    /// <param name="distance">Distance caméra→objet (world).</param>
    /// <param name="verticalFovRadians">FOV vertical.</param>
    /// <param name="screenHeightPixels">Hauteur du viewport en pixels.</param>
    /// <param name="pixelErrorThreshold">Erreur d'écran tolérée (pixels).</param>
    public NeuralPolygonLod SelectLod(
        float distance,
        float verticalFovRadians,
        int screenHeightPixels,
        float pixelErrorThreshold = 1.0f)
    {
        if (_levels.Count == 0)
            throw new InvalidOperationException("LOD chain is empty.");

        distance = MathF.Max(distance, 1e-4f);
        float pixelsPerWorldUnit =
            screenHeightPixels / (2f * distance * MathF.Tan(verticalFovRadians * 0.5f));

        // Du plus grossier au plus fin : premier niveau qui satisfait le budget.
        for (int i = _levels.Count - 1; i >= 0; i--)
        {
            float screenError = _levels[i].GeometricError * pixelsPerWorldUnit;
            if (screenError <= pixelErrorThreshold)
                return _levels[i];
        }

        return _levels[0]; // budget intenable : rendre le plus fin disponible
    }

    /// <summary>
    /// Erreur d'écran (pixels) d'un niveau donné à une distance donnée —
    /// exposé pour instrumentation et tests.
    /// </summary>
    public static float ProjectedScreenError(
        float geometricError, float distance, float verticalFovRadians, int screenHeightPixels)
    {
        distance = MathF.Max(distance, 1e-4f);
        return geometricError * screenHeightPixels
            / (2f * distance * MathF.Tan(verticalFovRadians * 0.5f));
    }
}
