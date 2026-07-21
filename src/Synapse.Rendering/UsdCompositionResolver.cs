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
/// Resolves USD composition arcs (references / payloads / subLayers) with cycle detection
/// and depth limits, then merges mesh primitives into a single <see cref="MeshAsset"/>.
/// </summary>
public static class UsdCompositionResolver
{
    public const int DefaultMaxDepth = 8;
    public const int DefaultMaxReferences = 32;

    private static readonly Regex ReferencePath = new(
        @"(?:references|payload|payloads|subLayers)\s*=\s*@([^@]+)@",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Multiline);

    private static readonly Regex ReferenceListPaths = new(
        @"(?:references|payload|payloads|subLayers)\s*=\s*\[([^\]]*)\]",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    private static readonly Regex AtPath = new(@"@([^@]+)@", RegexOptions.Compiled);

    public static IReadOnlyList<string> ExtractReferencePaths(string usdaText)
    {
        var paths = new List<string>();
        foreach (Match m in ReferencePath.Matches(usdaText))
        {
            var p = m.Groups[1].Value.Trim();
            if (!string.IsNullOrEmpty(p))
                paths.Add(p);
        }

        foreach (Match m in ReferenceListPaths.Matches(usdaText))
        {
            foreach (Match at in AtPath.Matches(m.Groups[1].Value))
            {
                var p = at.Groups[1].Value.Trim();
                if (!string.IsNullOrEmpty(p))
                    paths.Add(p);
            }
        }

        return paths;
    }

    public static string ResolvePath(string parentFile, string reference)
    {
        var cleaned = reference.Trim();
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

        async Task VisitAsync(string path, int depth)
        {
            ct.ThrowIfCancellationRequested();
            var full = Path.GetFullPath(path);
            if (!visited.Add(full))
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

            var ext = Path.GetExtension(full).ToLowerInvariant();
            if (ext is ".usda" or ".usd")
            {
                // Prefer ASCII composition walk when file is text USDA.
                if (!(ext == ".usd" && IsBinary(full)))
                {
                    var text = await File.ReadAllTextAsync(full, ct).ConfigureAwait(false);
                    var refs = ExtractReferencePaths(text);
                    foreach (var r in refs)
                    {
                        if (refCount++ >= maxReferences)
                        {
                            errors.Add($"Composition reference limit exceeded ({maxReferences}).");
                            return;
                        }
                        var child = ResolvePath(full, r);
                        await VisitAsync(child, depth + 1).ConfigureAwait(false);
                    }
                }
            }

            var leaf = await loadLeafAsync(full, config, ct).ConfigureAwait(false);
            if (leaf.Success && leaf.Asset != null)
            {
                foreach (var prim in leaf.Asset.Primitives)
                    merged.Primitives.Add(prim);
            }
            else if (!string.IsNullOrEmpty(leaf.ErrorMessage))
            {
                // References-only USDA may have no local mesh — not fatal if children contributed.
                if (merged.Primitives.Count == 0)
                    errors.Add(leaf.ErrorMessage);
            }
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        await VisitAsync(filePath, 0).ConfigureAwait(false);
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

        // Recompute bounds
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
