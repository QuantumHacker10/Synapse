using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Rendering.MeshIO;

/// <summary>Minimal USDA (ASCII USD) mesh importer with composition-arc resolution.</summary>
public sealed class UsdAsciiLoader
{
    private static readonly Regex PointTuple = new(@"\(([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\)", RegexOptions.Compiled);

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

        // Composition walk for USDA (and ASCII .usd): resolve references then merge meshes.
        return UsdCompositionResolver.LoadWithCompositionAsync(
            filePath,
            LoadLeafMeshAsync,
            config,
            ct);
    }

    /// <summary>Loads a single file's local mesh without following composition arcs.</summary>
    public Task<MeshLoadResult> LoadLeafMeshAsync(string filePath, MeshLoadConfig? config, CancellationToken ct)
    {
        config ??= new MeshLoadConfig();
        var result = new MeshLoadResult();
        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            var ext = Path.GetExtension(filePath).ToLowerInvariant();
            if (ext == ".usdc" || (ext == ".usd" && IsBinaryUsd(filePath)))
                return new UsdBinaryLoader().LoadAsync(filePath, config, ct);

            var text = File.ReadAllText(filePath);
            // References-only stage files are valid during composition.
            if (UsdCompositionResolver.ExtractReferencePaths(text).Count > 0 &&
                text.IndexOf("point3f[] points", StringComparison.Ordinal) < 0)
            {
                result.Success = true;
                result.Asset = new MeshAsset { Name = Path.GetFileNameWithoutExtension(filePath) };
                sw.Stop();
                result.LoadTime = sw.Elapsed;
                return Task.FromResult(result);
            }

            var points = ParsePointsArray(text);
            var indices = ParseFaceIndices(text);
            var translate = ParseTranslate(text);
            if (translate != Vector3.Zero)
            {
                for (int i = 0; i < points.Count; i++)
                    points[i] += translate;
            }

            if (points.Count == 0)
            {
                result.ErrorMessage = "No point positions found in USD file.";
                return Task.FromResult(result);
            }

            if (indices.Count == 0)
            {
                for (uint i = 0; i < points.Count; i++)
                    indices.Add(i);
            }

            var asset = new MeshAsset { Name = Path.GetFileNameWithoutExtension(filePath) };
            var primitive = new MeshPrimitive { Topology = PrimitiveTopology.TriangleList };
            foreach (var p in points)
                primitive.Vertices.Add(new MeshVertex { Position = p, Normal = Vector3.UnitY });
            primitive.Indices.AddRange(indices);
            asset.Primitives.Add(primitive);
            asset.Bounds = new BoundingBox3D
            {
                Min = points.Aggregate(Vector3.One * float.MaxValue, Vector3.Min),
                Max = points.Aggregate(Vector3.One * float.MinValue, Vector3.Max)
            };

            result.Success = true;
            result.Asset = asset;
        }
        catch (Exception ex)
        {
            result.ErrorMessage = $"USD import error: {ex.Message}";
        }

        sw.Stop();
        result.LoadTime = sw.Elapsed;
        return Task.FromResult(result);
    }

    private static List<Vector3> ParsePointsArray(string text)
    {
        var points = new List<Vector3>();
        var marker = "point3f[] points = [";
        int idx = text.IndexOf(marker, StringComparison.Ordinal);
        if (idx < 0)
            return points;

        int start = idx + marker.Length;
        int end = text.IndexOf(']', start);
        if (end < start)
            return points;

        var body = text[start..end];
        foreach (Match match in PointTuple.Matches(body))
        {
            if (float.TryParse(match.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) &&
                float.TryParse(match.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var y) &&
                float.TryParse(match.Groups[3].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var z))
            {
                points.Add(new Vector3(x, y, z));
            }
        }

        return points;
    }

    private static List<uint> ParseFaceIndices(string text)
    {
        var indices = new List<uint>();
        var marker = "int[] faceVertexIndices = [";
        int idx = text.IndexOf(marker, StringComparison.Ordinal);
        if (idx < 0)
            return indices;

        int start = idx + marker.Length;
        int end = text.IndexOf(']', start);
        if (end < start)
            return indices;

        var body = text[start..end];
        var nums = body.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var poly = new List<int>();
        foreach (var n in nums)
        {
            if (!int.TryParse(n, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v))
                continue;
            if (v == -1)
            {
                for (int i = 1; i < poly.Count - 1; i++)
                {
                    indices.Add((uint)poly[0]);
                    indices.Add((uint)poly[i]);
                    indices.Add((uint)poly[i + 1]);
                }
                poly.Clear();
            }
            else
            {
                poly.Add(v);
            }
        }

        return indices;
    }

    /// <summary>Parses <c>double3 xformOp:translate = (x, y, z)</c> or <c>float3 xformOp:translate</c>.</summary>
    public static Vector3 ParseTranslate(string text)
    {
        var marker = "xformOp:translate";
        int idx = text.IndexOf(marker, StringComparison.Ordinal);
        if (idx < 0)
            return Vector3.Zero;
        int eq = text.IndexOf('=', idx);
        if (eq < 0)
            return Vector3.Zero;
        int open = text.IndexOf('(', eq);
        int close = open >= 0 ? text.IndexOf(')', open) : -1;
        if (open < 0 || close < open)
            return Vector3.Zero;
        var body = text.Substring(open + 1, close - open - 1);
        var parts = body.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length < 3)
            return Vector3.Zero;
        if (float.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out var x) &&
            float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var y) &&
            float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var z))
            return new Vector3(x, y, z);
        return Vector3.Zero;
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
