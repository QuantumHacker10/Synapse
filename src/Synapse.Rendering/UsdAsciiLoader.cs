using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Rendering.MeshIO;

/// <summary>USDA (ASCII USD) mesh importer with composition, materials, skeletons, and variants.</summary>
public sealed class UsdAsciiLoader
{
    public Task<MeshLoadResult> LoadAsync(string filePath, MeshLoadConfig? config = null, CancellationToken ct = default)
    {
        config ??= new MeshLoadConfig();
        var ext = Path.GetExtension(filePath).ToLowerInvariant();
        if (ext is not (".usd" or ".usda" or ".usdc"))
        {
            return Task.FromResult(new MeshLoadResult { ErrorMessage = "Expected .usd, .usda or .usdc extension." });
        }

        if (ext == ".usdc" || (ext == ".usd" && IsBinaryUsd(filePath)))
            return new UsdBinaryLoader().LoadAsync(filePath, config, ct);

        return UsdCompositionResolver.LoadWithCompositionAsync(
            filePath,
            (path, cfg, token) => LoadLeafMeshAsync(path, cfg, applyLocalXform: false, token),
            config,
            ct);
    }

    public Task<MeshLoadResult> LoadLeafMeshAsync(string filePath, MeshLoadConfig? config, CancellationToken ct) =>
        LoadLeafMeshAsync(filePath, config, applyLocalXform: true, ct);

    public Task<MeshLoadResult> LoadLeafMeshAsync(
        string filePath,
        MeshLoadConfig? config,
        bool applyLocalXform,
        CancellationToken ct)
    {
        config ??= new MeshLoadConfig();
        var result = new MeshLoadResult();
        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            var ext = Path.GetExtension(filePath).ToLowerInvariant();
            if (ext == ".usdc" || (ext == ".usd" && IsBinaryUsd(filePath)))
                return new UsdBinaryLoader().LoadAsync(filePath, config, ct);

            var raw = File.ReadAllText(filePath);
            var text = UsdVariantResolver.ApplyVariants(raw, config);

            if (UsdCompositionResolver.ExtractReferencePaths(text).Count > 0 &&
                text.IndexOf("point3f[] points", StringComparison.Ordinal) < 0 &&
                text.IndexOf("point3d[] points", StringComparison.Ordinal) < 0)
            {
                result.Success = true;
                result.Asset = new MeshAsset { Name = Path.GetFileNameWithoutExtension(filePath) };
                sw.Stop();
                result.LoadTime = sw.Elapsed;
                return Task.FromResult(result);
            }

            result = LoadLeafFromText(
                text,
                Path.GetFileNameWithoutExtension(filePath),
                config,
                applyLocalXform,
                Path.GetDirectoryName(Path.GetFullPath(filePath)));
        }
        catch (Exception ex)
        {
            result.ErrorMessage = $"USD import error: {ex.Message}";
        }

        sw.Stop();
        result.LoadTime = sw.Elapsed;
        return Task.FromResult(result);
    }

    /// <summary>Parses a USDA text buffer (already variant-resolved) into a mesh asset.</summary>
    public MeshLoadResult LoadLeafFromText(
        string text,
        string assetName,
        MeshLoadConfig? config,
        bool applyLocalXform,
        string? baseDirectory = null)
    {
        config ??= new MeshLoadConfig();
        var result = new MeshLoadResult();
        var warnings = config.CollectUsdDiagnostics ? result.Warnings : null;

        var materials = UsdMaterialParser.ParseMaterials(text, baseDirectory);
        var skeleton = UsdSkeletonParser.ParseSkeleton(text, config);
        var clips = UsdSkelAnimationParser.ParseClips(text);
        ApplyStageTiming(text, clips);
        var blends = UsdSkelBlendShapeParser.ParseBlendShapes(text);
        var binding = UsdMaterialParser.ParseBindingPath(text);
        int defaultMatIndex = UsdMaterialParser.ResolveMaterialIndex(materials, binding);

        var asset = new MeshAsset { Name = assetName };
        asset.Materials.AddRange(materials);
        asset.Skeleton = skeleton;
        asset.AnimationClips.AddRange(clips);
        asset.BlendShapes.AddRange(blends);

        var meshBodies = UsdMeshTopology.EnumerateMeshBodies(text);
        if (meshBodies.Count == 0)
        {
            // Flat stage without def Mesh — treat whole buffer as one mesh body.
            meshBodies = new List<(string Name, string Body)> { (assetName, text) };
        }

        Matrix4x4 local = applyLocalXform ? UsdXform.ParseLocalMatrix(text) : Matrix4x4.Identity;
        var allPoints = new List<Vector3>();

        foreach (var (meshName, body) in meshBodies)
        {
            if (UsdMeshTopology.ShouldSkipPrim(body, config, out var skipReason))
            {
                warnings?.Add($"{meshName}: {skipReason}");
                continue;
            }

            var points = UsdMeshTopology.ParsePoints(body);
            if (points.Count == 0)
                continue;

            if (applyLocalXform && !local.IsIdentity)
                UsdXform.ApplyToPoints(points, local);

            var indices = UsdMeshTopology.ParseTriangulatedIndices(body, warnings);
            if (indices.Count == 0)
            {
                for (uint i = 0; i < points.Count; i++)
                    indices.Add(i);
                warnings?.Add($"{meshName}: no faceVertexIndices; using point order.");
            }

            var uvs = UsdMeshTopology.ParseUvs(body);
            var normals = UsdMeshTopology.ParseNormals(body);

            // Per-mesh material binding overrides stage binding when present.
            var meshBinding = UsdMaterialParser.ParseBindingPath(body) ?? binding;
            int matIndex = UsdMaterialParser.ResolveMaterialIndex(materials, meshBinding);
            if (matIndex < 0 || matIndex >= materials.Count)
                matIndex = defaultMatIndex;

            if (UsdMeshTopology.ParseDoubleSided(body) && matIndex >= 0 && matIndex < materials.Count)
                materials[matIndex].DoubleSided = true;

            var primitive = new MeshPrimitive
            {
                Name = meshName,
                Topology = PrimitiveTopology.TriangleList,
                MaterialIndex = matIndex,
                ActiveAttributes = VertexAttribute.Position | VertexAttribute.Normal
            };

            for (int i = 0; i < points.Count; i++)
            {
                var v = new MeshVertex { Position = points[i], Normal = Vector3.UnitY };
                if (i < uvs.Count)
                {
                    v.TexCoord0 = uvs[i];
                    primitive.ActiveAttributes |= VertexAttribute.TexCoord0;
                }

                primitive.Vertices.Add(v);
            }

            primitive.Indices.AddRange(indices);
            UsdMeshTopology.AssignNormals(primitive.Vertices, primitive.Indices, normals, config.ProcessFlags, warnings);

            // Skin primvars are usually authored on the mesh body.
            UsdSkeletonParser.ApplySkinPrimvars(body, primitive.Vertices, config);
            if (body.Contains("primvars:skel:jointIndices", StringComparison.OrdinalIgnoreCase) ||
                text.Contains("primvars:skel:jointIndices", StringComparison.OrdinalIgnoreCase))
            {
                // Fall back to whole-stage primvars when mesh body lacks them.
                if (!body.Contains("primvars:skel:jointIndices", StringComparison.OrdinalIgnoreCase))
                    UsdSkeletonParser.ApplySkinPrimvars(text, primitive.Vertices, config);
                primitive.ActiveAttributes |= VertexAttribute.BoneWeights | VertexAttribute.BoneIndices;
            }

            if (UsdMeshTopology.TryParseExtent(body, out var extent))
                primitive.Bounds = extent;
            else
            {
                primitive.Bounds = new BoundingBox3D
                {
                    Min = points.Aggregate(Vector3.One * float.MaxValue, Vector3.Min),
                    Max = points.Aggregate(Vector3.One * float.MinValue, Vector3.Max)
                };
            }

            allPoints.AddRange(points);
            asset.Primitives.Add(primitive);
            asset.GlobalAttributes |= primitive.ActiveAttributes;
        }

        if (asset.Primitives.Count == 0)
        {
            if (materials.Count > 0 || skeleton != null || clips.Count > 0 || blends.Count > 0)
            {
                result.Success = true;
                result.Asset = asset;
                result.WarningsCount = result.Warnings.Count;
                return result;
            }

            result.ErrorMessage = "No point positions found in USD file.";
            result.WarningsCount = result.Warnings.Count;
            return result;
        }

        if (UsdMeshTopology.TryParseExtent(text, out var stageExtent))
            asset.Bounds = stageExtent;
        else if (allPoints.Count > 0)
        {
            asset.Bounds = new BoundingBox3D
            {
                Min = allPoints.Aggregate(Vector3.One * float.MaxValue, Vector3.Min),
                Max = allPoints.Aggregate(Vector3.One * float.MinValue, Vector3.Max)
            };
        }

        result.Success = true;
        result.Asset = asset;
        result.WarningsCount = result.Warnings.Count;
        return result;
    }

    public static Vector3 ParseTranslate(string text) =>
        UsdXform.ParseVec3Op(text, "xformOp:translate");

    private static void ApplyStageTiming(string text, List<MeshAnimationClip> clips)
    {
        if (clips.Count == 0)
            return;
        float? tps = UsdMeshTopology.ParseStageFloat(text, "timeCodesPerSecond")
                     ?? UsdMeshTopology.ParseStageFloat(text, "framesPerSecond");
        float? start = UsdMeshTopology.ParseStageFloat(text, "startTimeCode");
        float? end = UsdMeshTopology.ParseStageFloat(text, "endTimeCode");
        foreach (var clip in clips)
        {
            if (tps is > 0f)
                clip.FrameRate = tps.Value;
            if (end is float e && start is float s && e >= s && tps is > 0f)
                clip.Duration = (e - s) / tps.Value;
            else if (end is float e2 && tps is > 0f && clip.Duration <= 0f)
                clip.Duration = e2 / tps.Value;
        }
    }

    private static bool IsBinaryUsd(string filePath)
    {
        try
        {
            Span<byte> header = stackalloc byte[8];
            using var fs = File.OpenRead(filePath);
            int read = fs.Read(header);
            return read == 8 && UsdBinaryLoader.IsUsdc(header);
        }
        catch
        {
            return false;
        }
    }
}
