using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Rendering.MeshIO;

/// <summary>
/// Resolves USD composition arcs (references / payloads / subLayers / inherits) with cycle detection,
/// depth limits, optional prim-path targets (<c>@file@&lt;/Prim&gt;</c>), and cumulative xform stacks.
/// </summary>
public static class UsdCompositionResolver
{
    public const int DefaultMaxDepth = 8;
    public const int DefaultMaxReferences = 32;

    private static readonly Regex ReferencePath = new(
        @"(?:references|payload|payloads|subLayers|inherits)\s*=\s*@([^@]+)@(?:\s*<([^>]+)>)?",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Multiline);

    private static readonly Regex ReferenceListPaths = new(
        @"(?:references|payload|payloads|subLayers|inherits)\s*=\s*\[([^\]]*)\]",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    private static readonly Regex AtPath = new(@"@([^@]+)@(?:\s*<([^>]+)>)?", RegexOptions.Compiled);

    public readonly record struct CompositionRef(string AssetPath, string? PrimPath);

    public static IReadOnlyList<string> ExtractReferencePaths(string usdaText) =>
        ExtractCompositionRefs(usdaText).Select(r => r.AssetPath).ToList();

    public static IReadOnlyList<CompositionRef> ExtractCompositionRefs(string usdaText)
    {
        var paths = new List<CompositionRef>();
        foreach (Match m in ReferencePath.Matches(usdaText))
        {
            var p = m.Groups[1].Value.Trim();
            if (string.IsNullOrEmpty(p))
                continue;
            var prim = m.Groups[2].Success ? m.Groups[2].Value.Trim() : null;
            paths.Add(new CompositionRef(p, string.IsNullOrEmpty(prim) ? null : prim));
        }

        foreach (Match m in ReferenceListPaths.Matches(usdaText))
        {
            foreach (Match at in AtPath.Matches(m.Groups[1].Value))
            {
                var p = at.Groups[1].Value.Trim();
                if (string.IsNullOrEmpty(p))
                    continue;
                var prim = at.Groups[2].Success ? at.Groups[2].Value.Trim() : null;
                paths.Add(new CompositionRef(p, string.IsNullOrEmpty(prim) ? null : prim));
            }
        }

        return paths;
    }

    public static string ResolvePath(string parentFile, string reference)
    {
        var cleaned = reference.Trim();
        // Strip optional prim path if caller passed "@file@</Prim>" style accidentally.
        int at2 = cleaned.IndexOf('@', cleaned.StartsWith('@') ? 1 : 0);
        if (cleaned.StartsWith('@') && at2 > 0)
            cleaned = cleaned[1..at2];

        while (cleaned.StartsWith("./", StringComparison.Ordinal) || cleaned.StartsWith(".\\", StringComparison.Ordinal))
            cleaned = cleaned[2..];
        if (Path.IsPathRooted(cleaned))
            return Path.GetFullPath(cleaned);
        var dir = Path.GetDirectoryName(Path.GetFullPath(parentFile)) ?? ".";
        return Path.GetFullPath(Path.Combine(dir, cleaned));
    }

    public static async Task<MeshLoadResult> LoadWithCompositionAsync(
        string filePath,
        Func<string, MeshLoadConfig?, CancellationToken, Task<MeshLoadResult>> loadLeafAsync,
        MeshLoadConfig? config = null,
        CancellationToken ct = default,
        int maxDepth = DefaultMaxDepth,
        int maxReferences = DefaultMaxReferences)
    {
        config ??= new MeshLoadConfig();
        var visited = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var merged = new MeshAsset { Name = Path.GetFileNameWithoutExtension(filePath) };
        var errors = new List<string>();
        int refCount = 0;

        async Task VisitAsync(string path, int depth, Matrix4x4 parentXform, string? primFilter)
        {
            ct.ThrowIfCancellationRequested();
            var full = Path.GetFullPath(path);
            var visitKey = primFilter == null ? full : $"{full}|{primFilter}";
            if (!visited.Add(visitKey))
                return; // cycle
            if (depth > maxDepth)
            {
                errors.Add($"Composition depth exceeded ({maxDepth}) at {full}");
                return;
            }

            if (!File.Exists(full))
            {
                errors.Add($"Missing composition target: {full}");
                return;
            }

            Matrix4x4 localXform = Matrix4x4.Identity;
            var ext = Path.GetExtension(full).ToLowerInvariant();
            if (ext is ".usda" or ".usd")
            {
                if (!(ext == ".usd" && IsBinary(full)))
                {
                    var text = await File.ReadAllTextAsync(full, ct).ConfigureAwait(false);
                    text = UsdVariantResolver.ApplyVariants(text, config);
                    localXform = UsdXform.ParseLocalMatrix(text);
                    var world = localXform * parentXform;

                    var refs = ExtractCompositionRefs(text);
                    foreach (var r in refs)
                    {
                        if (refCount++ >= maxReferences)
                        {
                            errors.Add($"Composition reference limit exceeded ({maxReferences}).");
                            return;
                        }

                        var child = ResolvePath(full, r.AssetPath);
                        await VisitAsync(child, depth + 1, world, r.PrimPath).ConfigureAwait(false);
                    }

                    var leaf = await loadLeafAsync(full, config, ct).ConfigureAwait(false);
                    MergeTransformed(leaf, world, primFilter, full, merged, errors);
                    return;
                }
            }

            // Binary / non-ASCII leaf
            {
                var leaf = await loadLeafAsync(full, config, ct).ConfigureAwait(false);
                var world = localXform * parentXform;
                MergeTransformed(leaf, world, primFilter, full, merged, errors);
            }
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        await VisitAsync(filePath, 0, Matrix4x4.Identity, null).ConfigureAwait(false);
        sw.Stop();

        if (merged.Primitives.Count == 0)
        {
            return new MeshLoadResult
            {
                Success = false,
                ErrorMessage = errors.Count > 0
                    ? string.Join("; ", errors)
                    : "No mesh primitives after composition.",
                LoadTime = sw.Elapsed
            };
        }

        var allPos = merged.Primitives.SelectMany(p => p.Vertices.Select(v => v.Position)).ToList();
        if (allPos.Count > 0)
        {
            merged.Bounds = new BoundingBox3D
            {
                Min = allPos.Aggregate(Vector3.One * float.MaxValue, Vector3.Min),
                Max = allPos.Aggregate(Vector3.One * float.MinValue, Vector3.Max)
            };
        }

        return new MeshLoadResult
        {
            Success = true,
            Asset = merged,
            LoadTime = sw.Elapsed,
            WarningsCount = errors.Count,
            Warnings = errors
        };
    }

    private static void MergeTransformed(
        MeshLoadResult leaf,
        Matrix4x4 world,
        string? primFilter,
        string fullPath,
        MeshAsset merged,
        List<string> errors)
    {
        if (leaf.Success && leaf.Asset != null)
        {
            int matOffset = merged.Materials.Count;
            foreach (var mat in leaf.Asset.Materials)
                merged.Materials.Add(mat);

            if (merged.Skeleton == null && leaf.Asset.Skeleton != null)
                merged.Skeleton = leaf.Asset.Skeleton;

            bool invertOk = Matrix4x4.Invert(world, out var inv);
            var nMat = invertOk ? Matrix4x4.Transpose(inv) : Matrix4x4.Identity;

            foreach (var prim in leaf.Asset.Primitives)
            {
                if (!string.IsNullOrEmpty(primFilter))
                {
                    var filter = primFilter.Trim().Trim('/');
                    var leafName = leaf.Asset.Name ?? "";
                    var fileStem = Path.GetFileNameWithoutExtension(fullPath);
                    var last = filter.Contains('/') ? filter[(filter.LastIndexOf('/') + 1)..] : filter;
                    if (!leafName.Equals(last, StringComparison.OrdinalIgnoreCase) &&
                        !fileStem.Equals(last, StringComparison.OrdinalIgnoreCase) &&
                        !string.Equals(prim.Name, last, StringComparison.OrdinalIgnoreCase))
                    {
                        // Soft filter: still include when leaf is a single-mesh file.
                        if (leaf.Asset.Primitives.Count > 1)
                            continue;
                    }
                }

                for (int i = 0; i < prim.Vertices.Count; i++)
                {
                    var v = prim.Vertices[i];
                    v.Position = Vector3.Transform(v.Position, world);
                    if (!world.IsIdentity && invertOk)
                    {
                        var n = Vector3.TransformNormal(v.Normal, nMat);
                        if (n.LengthSquared() > 1e-12f)
                            v.Normal = Vector3.Normalize(n);
                    }

                    prim.Vertices[i] = v;
                }

                prim.MaterialIndex += matOffset;
                merged.Primitives.Add(prim);
            }
        }
        else if (!string.IsNullOrEmpty(leaf.ErrorMessage) && merged.Primitives.Count == 0)
        {
            errors.Add(leaf.ErrorMessage!);
        }
    }

    private static bool IsBinary(string path)
    {
        try
        {
            Span<byte> header = stackalloc byte[8];
            using var fs = File.OpenRead(path);
            return fs.Read(header) == 8 && UsdBinaryLoader.IsUsdc(header);
        }
        catch
        {
            return false;
        }
    }
}
