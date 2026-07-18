using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Evaluation;
using GDNN.Rendering.MeshIO;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Offline pipeline: load a triangle mesh and train a hash-encoded SDF network against it.
/// </summary>
public sealed class MeshToSdfOptions
{
    public int SampleCount { get; init; } = 4096;
    public int Epochs { get; init; } = 40;
    public int? RandomSeed { get; init; }
    public float LearningRate { get; init; } = 5e-3f;
    public float HashLearningRate { get; init; } = 5e-2f;
    public MeshLoadConfig? LoadConfig { get; init; }
    public string? OutputAssetPath { get; init; }
    public Guid? TargetMeshId { get; init; }
}

public sealed class MeshToSdfResult
{
    public bool Success { get; init; }
    public string ErrorMessage { get; init; } = "";
    public HashEncodedDeepMLP? Network { get; init; }
    public OfflineHashMeshTrainer.TrainingReport? Report { get; init; }
    public ReferenceMeshSdf? ReferenceMesh { get; init; }
    public NeuralAsset? SavedAsset { get; init; }
}

public static class MeshToSdfPipeline
{
    private static readonly MeshLoader SharedLoader = new();

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
