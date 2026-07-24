using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Rendering.MeshIO;
using Synapse.Infrastructure;

namespace Synapse.Runtime;

/// <summary>Exports a full <see cref="SceneDocument"/> to glTF 2.0 with entity transforms and metadata.</summary>
public static class SceneGlTFExporter
{
    public sealed class SceneExportResult
    {
        public bool Success { get; set; }
        public string? OutputPath { get; set; }
        public string? ErrorMessage { get; set; }
        public int EntityCount { get; set; }
    }

    public static async Task<SceneExportResult> ExportAsync(
        SceneDocument scene,
        string outputPath,
        SynapseMeshProvider? meshProvider = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(scene);
        var result = new SceneExportResult { OutputPath = outputPath };

        try
        {
            var full = Path.GetFullPath(outputPath);
            Directory.CreateDirectory(Path.GetDirectoryName(full)!);
            var ext = Path.GetExtension(full).ToLowerInvariant();
            if (ext != ".gltf")
            {
                result.ErrorMessage = "Scene export v2 supports .gltf (JSON) with embedded extras.";
                return result;
            }

            var loader = new MeshLoader();
            var exporter = new GlTFExporter();
            var nodeArray = new JsonArray();

            foreach (var entity in scene.Entities.Where(e => e.Visible))
            {
                var extras = new JsonObject
                {
                    ["synapseType"] = entity.Type,
                    ["synapseId"] = entity.Id.ToString(),
                    ["behaviorProfile"] = entity.BehaviorProfile,
                    ["generator"] = $"{SynapseProduct.Name} v{SynapseProduct.Version}"
                };

                var node = new JsonObject
                {
                    ["name"] = entity.Name,
                    ["translation"] = new JsonArray(entity.Position.X, entity.Position.Y, entity.Position.Z),
                    ["scale"] = new JsonArray(entity.Scale.X, entity.Scale.Y, entity.Scale.Z),
                    ["extras"] = extras
                };

                if (!string.IsNullOrWhiteSpace(entity.MeshPath) && File.Exists(entity.MeshPath))
                {
                    var meshResult = await loader.LoadAsync(entity.MeshPath, ct: cancellationToken);
                    if (meshResult.Success && meshResult.Asset != null)
                    {
                        var meshPath = Path.Combine(Path.GetDirectoryName(full)!, $"{Sanitize(entity.Name)}.mesh.glb");
                        var export = await exporter.ExportAsync(meshPath, meshResult.Asset, ct: cancellationToken);
                        if (export.Success)
                            extras["meshGlb"] = Path.GetFileName(meshPath);
                    }
                }

                nodeArray.Add(node);
            }

            var sceneNodes = new JsonArray();
            for (int i = 0; i < nodeArray.Count; i++)
                sceneNodes.Add(i);

            var root = new JsonObject
            {
                ["asset"] = new JsonObject
                {
                    ["generator"] = $"{SynapseProduct.Name} Scene Exporter",
                    ["version"] = "2.0"
                },
                ["scene"] = 0,
                ["scenes"] = new JsonArray(new JsonObject
                {
                    ["name"] = scene.Name,
                    ["nodes"] = sceneNodes
                }),
                ["nodes"] = nodeArray,
                ["extras"] = new JsonObject
                {
                    ["synapseVersion"] = scene.Version,
                    ["activeLawId"] = scene.ActiveLawId,
                    ["activeLawExpression"] = scene.ActiveLawExpression,
                    ["entityCount"] = scene.Entities.Count
                }
            };

            var options = new JsonSerializerOptions { WriteIndented = true };
            await File.WriteAllTextAsync(full, root.ToJsonString(options), cancellationToken);
            result.Success = true;
            result.EntityCount = scene.Entities.Count;
        }
        catch (Exception ex)
        {
            result.ErrorMessage = ex.Message;
        }

        return result;
    }

    private static string Sanitize(string name) =>
        string.Concat(name.Select(c => char.IsLetterOrDigit(c) ? c : '_'));
}
