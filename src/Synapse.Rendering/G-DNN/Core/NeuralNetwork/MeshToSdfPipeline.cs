using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Evaluation;
using GDNN.Rendering.MeshIO;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Training options for <see cref="MeshToSdfPipeline"/>.
/// </summary>
public sealed class MeshToSdfOptions
{
    /// <summary>Number of surface/near-surface samples per training epoch.</summary>
    public int SampleCount { get; init; } = 4096;

    /// <summary>Training epoch count.</summary>
    public int Epochs { get; init; } = 40;

    /// <summary>Optional RNG seed for reproducible runs.</summary>
    public int? RandomSeed { get; init; }

    /// <summary>MLP weight learning rate.</summary>
    public float LearningRate { get; init; } = 5e-3f;

    /// <summary>Hash-grid learning rate.</summary>
    public float HashLearningRate { get; init; } = 5e-2f;

    /// <summary>Mesh loader overrides (scale, axis flip, etc.).</summary>
    public MeshLoadConfig? LoadConfig { get; init; }

    /// <summary>When set, writes a serialized <see cref="NeuralAsset"/> to this path.</summary>
    public string? OutputAssetPath { get; init; }

    /// <summary>Stable mesh id stored in the exported neural asset.</summary>
    public Guid? TargetMeshId { get; init; }
}

/// <summary>Outcome of a mesh-to-SDF training run.</summary>
public sealed class MeshToSdfResult
{
    /// <summary>True when training and optional export succeeded.</summary>
    public bool Success { get; init; }

    /// <summary>Human-readable failure reason when <see cref="Success"/> is false.</summary>
    public string ErrorMessage { get; init; } = "";

    /// <summary>Trained hash-encoded SDF network.</summary>
    public HashEncodedDeepMLP? Network { get; init; }

    /// <summary>Per-epoch loss metrics from the offline trainer.</summary>
    public OfflineHashMeshTrainer.TrainingReport? Report { get; init; }

    /// <summary>Triangle mesh used as the ground-truth SDF oracle.</summary>
    public ReferenceMeshSdf? ReferenceMesh { get; init; }

    /// <summary>Serialized asset when <see cref="MeshToSdfOptions.OutputAssetPath"/> was set.</summary>
    public NeuralAsset? SavedAsset { get; init; }
}

/// <summary>
/// Offline workflow: load a triangle mesh, train a hash-encoded neural SDF, optionally export a neural asset.
/// </summary>
public static class MeshToSdfPipeline
{
    private static readonly MeshLoader SharedLoader = new();

    /// <summary>Loads a mesh from disk and trains a hash-encoded SDF against it.</summary>
    public static async Task<MeshToSdfResult> TrainFromFileAsync(
        string meshPath,
        MeshToSdfOptions? options = null,
        CancellationToken ct = default)
    {
        options ??= new MeshToSdfOptions();

        if (string.IsNullOrWhiteSpace(meshPath))
            return new MeshToSdfResult { ErrorMessage = "Mesh path is required." };

        ct.ThrowIfCancellationRequested();

        var loadResult = await SharedLoader.LoadAsync(meshPath, options.LoadConfig, ct);
        if (!loadResult.Success || loadResult.Asset == null)
        {
            return new MeshToSdfResult
            {
                ErrorMessage = string.IsNullOrWhiteSpace(loadResult.ErrorMessage)
                    ? "Failed to load mesh."
                    : loadResult.ErrorMessage
            };
        }

        return await Task.Run(() => TrainFromAsset(loadResult.Asset, options, ct), ct);
    }

    /// <summary>Trains a hash-encoded SDF from an already-loaded mesh asset.</summary>
    public static MeshToSdfResult TrainFromAsset(
        MeshAsset asset,
        MeshToSdfOptions? options = null,
        CancellationToken ct = default)
    {
        options ??= new MeshToSdfOptions();
        ArgumentNullException.ThrowIfNull(asset);

        ct.ThrowIfCancellationRequested();

        ReferenceMeshSdf referenceMesh;
        try
        {
            referenceMesh = BuildReferenceMesh(asset);
        }
        catch (Exception ex)
        {
            return new MeshToSdfResult { ErrorMessage = $"Failed to build reference mesh: {ex.Message}" };
        }

        if (referenceMesh.TriangleCount == 0)
            return new MeshToSdfResult { ErrorMessage = "Mesh contains no triangles." };

        var random = options.RandomSeed.HasValue ? new Random(options.RandomSeed.Value) : new Random(42);
        var network = new HashEncodedDeepMLP(random);
        var trainer = new OfflineHashMeshTrainer
        {
            LearningRate = options.LearningRate,
            HashLearningRate = options.HashLearningRate
        };

        ct.ThrowIfCancellationRequested();
        var report = trainer.TrainOnMesh(
            network,
            referenceMesh,
            options.SampleCount,
            options.Epochs,
            random);

        NeuralAsset? savedAsset = null;
        if (!string.IsNullOrWhiteSpace(options.OutputAssetPath))
        {
            savedAsset = CreateNeuralAsset(network, asset, options);
            savedAsset.SaveAsync(options.OutputAssetPath!, ct).GetAwaiter().GetResult();
        }

        return new MeshToSdfResult
        {
            Success = true,
            Network = network,
            Report = report,
            ReferenceMesh = referenceMesh,
            SavedAsset = savedAsset
        };
    }

    /// <summary>Builds a triangle oracle used as ground truth during offline SDF training.</summary>
    public static ReferenceMeshSdf BuildReferenceMesh(MeshAsset asset)
    {
        ArgumentNullException.ThrowIfNull(asset);

        var vertices = new System.Collections.Generic.List<Vector3>();
        var indices = new System.Collections.Generic.List<int>();

        foreach (var primitive in asset.Primitives)
        {
            if (primitive.Vertices.Count == 0 || primitive.Indices.Count < 3)
                continue;

            int baseIndex = vertices.Count;
            foreach (var vertex in primitive.Vertices)
                vertices.Add(vertex.Position);

            for (int i = 0; i + 2 < primitive.Indices.Count; i += 3)
            {
                indices.Add(baseIndex + (int)primitive.Indices[i]);
                indices.Add(baseIndex + (int)primitive.Indices[i + 1]);
                indices.Add(baseIndex + (int)primitive.Indices[i + 2]);
            }
        }

        if (vertices.Count == 0 || indices.Count == 0)
            throw new InvalidOperationException("Mesh asset has no usable triangle data.");

        return new ReferenceMeshSdf(vertices.ToArray(), indices.ToArray());
    }

    /// <summary>Wraps a trained network and source mesh metadata into a serializable neural asset.</summary>
    public static NeuralAsset CreateNeuralAsset(
        HashEncodedDeepMLP network,
        MeshAsset sourceMesh,
        MeshToSdfOptions? options = null)
    {
        options ??= new MeshToSdfOptions();

        var asset = new NeuralAsset
        {
            AssetId = Guid.NewGuid(),
            TargetMeshId = options.TargetMeshId ?? Guid.NewGuid(),
            Metadata = new NeuralAssetMetadata
            {
                Name = string.IsNullOrWhiteSpace(sourceMesh.Name) ? "MeshToSdf" : sourceMesh.Name,
                SourceEngine = "MeshToSdfPipeline",
                Description = $"Hash-encoded SDF trained from {sourceMesh.SourcePath}"
            }
        };

        var bounds = sourceMesh.Bounds;
        if (bounds.Min.X <= bounds.Max.X)
        {
            asset.BoundingBox = new BoundingBox { Min = bounds.Min, Max = bounds.Max };
            asset.BoundingSphere = new BoundingSphere
            {
                Center = bounds.Center,
                Radius = bounds.Extents.Length()
            };
        }

        asset.FromHashEncodedDeepMLP(network);
        return asset;
    }
}
