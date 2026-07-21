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

/// <summary>Minimal USDA (ASCII USD) mesh importer for simple triangle meshes.</summary>
public sealed class UsdAsciiLoader
{
    private static readonly Regex PointTuple = new(@"\(([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\)", RegexOptions.Compiled);

    public Task<MeshLoadResult> LoadAsync(string filePath, MeshLoadConfig? config = null, CancellationToken ct = default)
    {
        config ??= new MeshLoadConfig();
        var result = new MeshLoadResult();
        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            var ext = Path.GetExtension(filePath).ToLowerInvariant();
            if (ext is not (".usd" or ".usda" or ".usdc"))
            {
                result.ErrorMessage = "Expected .usd, .usda or .usdc extension.";
                return Task.FromResult(result);
            }

            if (ext == ".usdc" || (ext == ".usd" && IsBinaryUsd(filePath)))
                return new UsdBinaryLoader().LoadAsync(filePath, config, ct);

            var text = File.ReadAllText(filePath);
            var points = ParsePointsArray(text);
            var indices = ParseFaceIndices(text);

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
