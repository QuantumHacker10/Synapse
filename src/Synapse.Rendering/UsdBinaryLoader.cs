using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Rendering.MeshIO;

/// <summary>
/// Minimal USDC (binary USD crate) mesh importer/exporter.
/// Supports Synapse mesh-pack crates (round-trip) and best-effort extraction from
/// OpenUSD-compatible crates that embed uncompressed point/index arrays.
/// </summary>
public sealed class UsdBinaryLoader
{
    public static ReadOnlySpan<byte> Magic => "PXR-USDC"u8;
    public const byte SynapseMeshPackVersion = 1;
    private const uint SectionPoints = 0x4D455348; // 'MESH' points payload marker in Synapse packs
    private const uint SectionIndices = 0x49445833; // 'IDX3'
    private const uint SectionTokens = 0x544F4B4E; // 'TOKN'

    public Task<MeshLoadResult> LoadAsync(string filePath, MeshLoadConfig? config = null, CancellationToken ct = default)
    {
        config ??= new MeshLoadConfig();
        var result = new MeshLoadResult();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            var bytes = File.ReadAllBytes(filePath);
            ct.ThrowIfCancellationRequested();
            if (!TryParse(bytes, Path.GetFileNameWithoutExtension(filePath), out var asset, out var error))
            {
                result.ErrorMessage = error;
                return Task.FromResult(result);
            }

            result.Success = true;
            result.Asset = asset;
        }
        catch (Exception ex)
        {
            result.ErrorMessage = $"USDC import error: {ex.Message}";
        }

        sw.Stop();
        result.LoadTime = sw.Elapsed;
        return Task.FromResult(result);
    }

    public static bool IsUsdc(ReadOnlySpan<byte> data) =>
        data.Length >= 8 && data[..8].SequenceEqual(Magic);

    public static bool TryParse(ReadOnlySpan<byte> data, string name, out MeshAsset? asset, out string? error)
    {
        asset = null;
        error = null;
        if (!IsUsdc(data))
        {
            error = "Not a USDC crate (missing PXR-USDC magic).";
            return false;
        }

        if (data.Length < 24)
        {
            error = "USDC file too small.";
            return false;
        }

        // Synapse mesh-pack: magic(8) + version(1) + reserved(7) + tocOffset(8) + sections…
        byte packVersion = data[8];
        if (packVersion == SynapseMeshPackVersion && TryParseSynapsePack(data, name, out asset, out error))
            return asset != null;

        // Best-effort OpenUSD / unknown crate: token scan + float/int array heuristics.
        if (TryExtractHeuristic(data, name, out asset, out error))
            return asset != null;

        error ??= "USDC crate recognized but no mesh points/indices could be extracted. " +
                  "Synapse supports Synapse mesh-pack USDC round-trip and best-effort OpenUSD crates; " +
                  "export USDA or a Synapse mesh-pack USDC for production MeshIO.";
        return false;
    }

    /// <summary>Writes a Synapse USDC mesh-pack (valid PXR-USDC magic, portable, round-trippable).</summary>
    public static void WriteSynapseMeshPack(string filePath, IReadOnlyList<Vector3> points, IReadOnlyList<uint> indices)
    {
        ArgumentNullException.ThrowIfNull(points);
        ArgumentNullException.ThrowIfNull(indices);
        using var ms = new MemoryStream();
        using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            bw.Write(Magic);
            bw.Write(SynapseMeshPackVersion);
            bw.Write(new byte[7]); // reserved
            long tocOffsetPos = ms.Position;
            bw.Write(0L); // toc offset placeholder

            long pointsOffset = ms.Position;
            bw.Write(SectionPoints);
            bw.Write(points.Count);
            foreach (var p in points)
            {
                bw.Write(p.X);
                bw.Write(p.Y);
                bw.Write(p.Z);
            }

            long indicesOffset = ms.Position;
            bw.Write(SectionIndices);
            bw.Write(indices.Count);
            foreach (var i in indices)
                bw.Write(i);

            // Optional TOKENS for OpenUSD-like discovery
            long tokensOffset = ms.Position;
            bw.Write(SectionTokens);
            var tokens = new[] { "points", "faceVertexIndices", "Mesh" };
            bw.Write(tokens.Length);
            foreach (var t in tokens)
            {
                var tb = Encoding.UTF8.GetBytes(t);
                bw.Write(tb.Length);
                bw.Write(tb);
            }

            long tocOffset = ms.Position;
            bw.Write(3); // section count
            WriteTocEntry(bw, SectionPoints, pointsOffset);
            WriteTocEntry(bw, SectionIndices, indicesOffset);
            WriteTocEntry(bw, SectionTokens, tokensOffset);

            ms.Position = tocOffsetPos;
            bw.Write(tocOffset);
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(filePath))!);
        File.WriteAllBytes(filePath, ms.ToArray());
    }

    private static void WriteTocEntry(BinaryWriter bw, uint tag, long offset)
    {
        bw.Write(tag);
        bw.Write(offset);
    }

    private static bool TryParseSynapsePack(ReadOnlySpan<byte> data, string name, out MeshAsset? asset, out string? error)
    {
        asset = null;
        error = null;
        long tocOffset = BitConverter.ToInt64(data.Slice(16, 8));
        if (tocOffset <= 0 || tocOffset >= data.Length)
        {
            error = "Invalid Synapse USDC TOC offset.";
            return false;
        }

        var points = new List<Vector3>();
        var indices = new List<uint>();

        int toc = (int)tocOffset;
        if (toc + 4 > data.Length)
            return false;
        int sectionCount = BitConverter.ToInt32(data.Slice(toc, 4));
        int cursor = toc + 4;
        for (int s = 0; s < sectionCount; s++)
        {
            if (cursor + 12 > data.Length)
                break;
            uint tag = BitConverter.ToUInt32(data.Slice(cursor, 4));
            long offset = BitConverter.ToInt64(data.Slice(cursor + 4, 8));
            cursor += 12;
            if (offset <= 0 || offset >= data.Length)
                continue;

            int o = (int)offset;
            if (tag == SectionPoints)
            {
                if (o + 8 > data.Length)
                    continue;
                int count = BitConverter.ToInt32(data.Slice(o + 4, 4));
                int body = o + 8;
                for (int i = 0; i < count; i++)
                {
                    if (body + 12 > data.Length)
                        break;
                    float x = BitConverter.ToSingle(data.Slice(body, 4));
                    float y = BitConverter.ToSingle(data.Slice(body + 4, 4));
                    float z = BitConverter.ToSingle(data.Slice(body + 8, 4));
                    points.Add(new Vector3(x, y, z));
                    body += 12;
                }
            }
            else if (tag == SectionIndices)
            {
                if (o + 8 > data.Length)
                    continue;
                int count = BitConverter.ToInt32(data.Slice(o + 4, 4));
                int body = o + 8;
                for (int i = 0; i < count; i++)
                {
                    if (body + 4 > data.Length)
                        break;
                    indices.Add(BitConverter.ToUInt32(data.Slice(body, 4)));
                    body += 4;
                }
            }
        }

        if (points.Count == 0)
        {
            error = "Synapse USDC pack has no points.";
            return false;
        }

        asset = BuildAsset(name, points, indices);
        return true;
    }

    private static bool TryExtractHeuristic(ReadOnlySpan<byte> data, string name, out MeshAsset? asset, out string? error)
    {
        asset = null;
        error = null;

        // Scan for ASCII "points" / "faceVertexIndices" near plausible arrays (OpenUSD often keeps tokens readable).
        string ascii = Encoding.ASCII.GetString(data);
        bool mentionsPoints = ascii.Contains("points", StringComparison.Ordinal);
        bool mentionsFaces = ascii.Contains("faceVertexIndices", StringComparison.Ordinal)
                             || ascii.Contains("faceVertexCounts", StringComparison.Ordinal);

        var points = ExtractFloat3Runs(data, minCount: 3, maxCount: 50_000);
        var indices = ExtractIndexRuns(data, minCount: 3, maxCount: 200_000, maxVertex: points.Count > 0 ? points.Count : 0);

        if (points.Count >= 3 && indices.Count >= 3 && (mentionsPoints || mentionsFaces || points.Count <= 10_000))
        {
            // Fan-triangulate if indices look like face lists with -1 sentinels
            var tris = TriangulateIfNeeded(indices);
            asset = BuildAsset(name, points, tris);
            return true;
        }

        if (points.Count >= 3 && indices.Count == 0)
        {
            var auto = new List<uint>();
            for (uint i = 0; i < points.Count; i++)
                auto.Add(i);
            asset = BuildAsset(name, points, auto);
            return true;
        }

        error = "Could not extract mesh arrays from USDC crate.";
        return false;
    }

    private static List<Vector3> ExtractFloat3Runs(ReadOnlySpan<byte> data, int minCount, int maxCount)
    {
        var best = new List<Vector3>();
        // Align to 4 bytes; look for long runs of finite floats in a plausible mesh range.
        for (int align = 0; align < 4; align++)
        {
            var current = new List<Vector3>();
            for (int i = align; i + 12 <= data.Length; i += 4)
            {
                float x = BitConverter.ToSingle(data.Slice(i, 4));
                float y = BitConverter.ToSingle(data.Slice(i + 4, 4));
                float z = BitConverter.ToSingle(data.Slice(i + 8, 4));
                if (IsPlausibleMeshFloat(x) && IsPlausibleMeshFloat(y) && IsPlausibleMeshFloat(z))
                {
                    current.Add(new Vector3(x, y, z));
                    i += 8; // consume full triplet (loop adds 4)
                    if (current.Count > maxCount)
                        break;
                }
                else
                {
                    if (current.Count >= minCount && current.Count > best.Count)
                        best = new List<Vector3>(current);
                    current.Clear();
                }
            }
            if (current.Count >= minCount && current.Count > best.Count)
                best = current;
        }
        return best;
    }

    private static List<int> ExtractIndexRuns(ReadOnlySpan<byte> data, int minCount, int maxCount, int maxVertex)
    {
        var best = new List<int>();
        for (int align = 0; align < 4; align++)
        {
            var current = new List<int>();
            for (int i = align; i + 4 <= data.Length; i += 4)
            {
                int v = BitConverter.ToInt32(data.Slice(i, 4));
                bool ok = v == -1 || (v >= 0 && (maxVertex <= 0 || v < maxVertex));
                if (ok && v < 1_000_000)
                {
                    current.Add(v);
                    if (current.Count > maxCount)
                        break;
                }
                else
                {
                    if (current.Count >= minCount && current.Count > best.Count)
                        best = new List<int>(current);
                    current.Clear();
                }
            }
            if (current.Count >= minCount && current.Count > best.Count)
                best = current;
        }
        return best;
    }

    private static List<uint> TriangulateIfNeeded(List<int> indices)
    {
        if (!indices.Contains(-1))
            return indices.Select(i => (uint)Math.Max(0, i)).ToList();

        var tris = new List<uint>();
        var poly = new List<int>();
        foreach (var v in indices)
        {
            if (v == -1)
            {
                for (int i = 1; i < poly.Count - 1; i++)
                {
                    tris.Add((uint)poly[0]);
                    tris.Add((uint)poly[i]);
                    tris.Add((uint)poly[i + 1]);
                }
                poly.Clear();
            }
            else if (v >= 0)
            {
                poly.Add(v);
            }
        }
        return tris;
    }

    private static bool IsPlausibleMeshFloat(float f) =>
        !float.IsNaN(f) && !float.IsInfinity(f) && MathF.Abs(f) < 1e6f;

    private static MeshAsset BuildAsset(string name, List<Vector3> points, List<uint> indices)
    {
        if (indices.Count == 0)
        {
            for (uint i = 0; i < points.Count; i++)
                indices.Add(i);
        }

        var asset = new MeshAsset { Name = name };
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
        return asset;
    }
}
