using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Rendering.MeshIO;

/// <summary>Minimal FBX ASCII mesh importer (vertices + polygon indices).</summary>
public sealed class FbxAsciiLoader
{
    public Task<MeshLoadResult> LoadAsync(string filePath, MeshLoadConfig? config = null, CancellationToken ct = default)
    {
        config ??= new MeshLoadConfig();
        var result = new MeshLoadResult();
        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            if (!filePath.EndsWith(".fbx", StringComparison.OrdinalIgnoreCase))
            {
                result.ErrorMessage = "Not an FBX file.";
                return Task.FromResult(result);
            }

            var lines = File.ReadAllLines(filePath);
            var vertices = new List<Vector3>();
            var indices = new List<uint>();

            foreach (var raw in lines)
            {
                ct.ThrowIfCancellationRequested();
                var line = raw.Trim();
                if (line.StartsWith("Vertices:", StringComparison.Ordinal))
                {
                    var coords = ParseFloats(line);
                    for (int i = 0; i + 2 < coords.Count; i += 3)
                        vertices.Add(new Vector3(coords[i], coords[i + 1], coords[i + 2]));
                }
                else if (line.StartsWith("PolygonVertexIndex:", StringComparison.Ordinal))
                {
                    foreach (var poly in ParseFbxPolygons(line))
                        TriangulateFan(poly, indices);
                }
            }

            if (vertices.Count == 0)
            {
                result.ErrorMessage = "No vertices found in FBX ASCII file.";
                return Task.FromResult(result);
            }

            var asset = BuildAsset(Path.GetFileNameWithoutExtension(filePath), vertices, indices, config);
            result.Success = true;
            result.Asset = asset;
        }
        catch (Exception ex)
        {
            result.ErrorMessage = $"FBX import error: {ex.Message}";
        }

        sw.Stop();
        result.LoadTime = sw.Elapsed;
        return Task.FromResult(result);
    }

    private static List<float> ParseFloats(string line)
    {
        var values = new List<float>();
        int start = line.IndexOf(':') + 1;
        if (start <= 0)
            return values;
        foreach (var part in line[start..].Split(','))
        {
            if (float.TryParse(part.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var f))
                values.Add(f);
        }
        return values;
    }

    private static IEnumerable<List<int>> ParseFbxPolygons(string line)
    {
        int start = line.IndexOf(':') + 1;
        if (start <= 0)
            yield break;

        var poly = new List<int>();
        foreach (var part in line[start..].Split(','))
        {
            if (!int.TryParse(part.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var v))
                continue;

            if (v < 0)
            {
                poly.Add((-v) - 1);
                if (poly.Count >= 3)
                    yield return poly;
                poly = new List<int>();
            }
            else
            {
                poly.Add(v);
            }
        }
    }

    private static List<int> ParseInts(string line)
    {
        var values = new List<int>();
        int start = line.IndexOf(':') + 1;
        if (start <= 0)
            return values;
        foreach (var part in line[start..].Split(','))
        {
            if (int.TryParse(part.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var v))
            {
                if (v < 0)
                {
                    values.Add((-v) - 1);
                    break;
                }
                values.Add(v);
            }
        }
        return values;
    }

    private static void TriangulateFan(List<int> poly, List<uint> indices)
    {
        for (int i = 1; i < poly.Count - 1; i++)
        {
            indices.Add((uint)poly[0]);
            indices.Add((uint)poly[i]);
            indices.Add((uint)poly[i + 1]);
        }
    }

    private static MeshAsset BuildAsset(string name, List<Vector3> vertices, List<uint> indices, MeshLoadConfig config)
    {
        var meshVertices = new List<MeshVertex>(vertices.Count);
        foreach (var v in vertices)
            meshVertices.Add(new MeshVertex { Position = v, Normal = Vector3.UnitY });

        var primitive = new MeshPrimitive
        {
            Vertices = meshVertices,
            Indices = indices,
            Topology = PrimitiveTopology.TriangleList
        };

        return new MeshAsset
        {
            Name = name,
            Primitives = { primitive },
            Bounds = new BoundingBox3D
            {
                Min = vertices.Aggregate(Vector3.One * float.MaxValue, Vector3.Min),
                Max = vertices.Aggregate(Vector3.One * float.MinValue, Vector3.Max)
            }
        };
    }
}
