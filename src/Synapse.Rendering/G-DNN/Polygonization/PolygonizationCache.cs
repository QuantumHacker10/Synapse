using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using GDNN.Core.DataStructures;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Polygonization;

/// <summary>
/// On-disk cache for polygonized neural SDF LOD chains.
///
/// Cache keys hash network weights, extraction bounds, and polygonization parameters so
/// unchanged static assets reload from disk instead of re-polygonizing every level at startup.
/// </summary>
public sealed class PolygonizationCache
{
    private const uint Magic = 0x474E4C43; // "CLNG" — Chain of Lods, Neural Geometry
    private const int FormatVersion = 1;

    private readonly string _directory;

    public PolygonizationCache(string directory)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        _directory = directory;
        Directory.CreateDirectory(directory);
    }

    /// <summary>Root directory where cached LOD chains are stored.</summary>
    public string CacheDirectory => _directory;

    /// <summary>
    /// Computes a stable cache key from network weights and extraction parameters.
    /// Two networks with identical weights produce the same key.
    /// </summary>
    public static string ComputeKey(
        HashEncodedDeepMLP network, AABB bounds, int baseResolution, int levelCount)
    {
        ArgumentNullException.ThrowIfNull(network);

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        AppendFloats(hash, network.Encoder.Serialize());
        AppendFloats(hash, network.Layer1Weights);
        AppendFloats(hash, network.Layer1Bias);
        AppendFloats(hash, network.Layer2Weights);
        AppendFloats(hash, network.Layer2Bias);
        AppendFloats(hash, network.OutputWeights);
        AppendFloats(hash, [network.OutputBias]);

        Span<float> parameters =
        [
            bounds.Center.X, bounds.Center.Y, bounds.Center.Z,
            bounds.HalfExtents.X, bounds.HalfExtents.Y, bounds.HalfExtents.Z,
            baseResolution, levelCount, FormatVersion
        ];
        AppendFloats(hash, parameters);

        return Convert.ToHexString(hash.GetHashAndReset());
    }

    /// <summary>
    /// Tries to load a LOD chain from disk. Returns false when missing or corrupt
    /// (stale entries are deleted automatically).
    /// </summary>
    public bool TryLoad(string key, out NeuralPolygonLodChain? chain)
    {
        chain = null;
        string path = PathForKey(key);
        if (!File.Exists(path))
            return false;

        try
        {
            using var file = File.OpenRead(path);
            using var brotli = new BrotliStream(file, CompressionMode.Decompress);
            using var reader = new BinaryReader(brotli);

            if (reader.ReadUInt32() != Magic || reader.ReadInt32() != FormatVersion)
            {
                reader.Dispose();
                File.Delete(path);
                return false;
            }

            var bounds = new AABB(ReadVector3(reader), ReadVector3(reader));
            int levelCount = reader.ReadInt32();
            var levels = new List<NeuralPolygonLod>(levelCount);

            for (int i = 0; i < levelCount; i++)
                levels.Add(ReadLod(reader));

            chain = NeuralPolygonLodChain.FromLevels(bounds, levels);
            return true;
        }
        catch (Exception e) when (e is IOException or EndOfStreamException or InvalidDataException)
        {
            TryDelete(path);
            return false;
        }
    }

    /// <summary>Writes a LOD chain to disk (replaces any existing entry).</summary>
    public void Store(string key, NeuralPolygonLodChain chain)
    {
        ArgumentNullException.ThrowIfNull(chain);

        string path = PathForKey(key);
        string tempPath = path + ".tmp";

        using (var file = File.Create(tempPath))
        using (var brotli = new BrotliStream(file, CompressionLevel.Fastest))
        using (var writer = new BinaryWriter(brotli))
        {
            writer.Write(Magic);
            writer.Write(FormatVersion);
            WriteVector3(writer, chain.Bounds.Center);
            WriteVector3(writer, chain.Bounds.HalfExtents);
            writer.Write(chain.Levels.Count);

            foreach (var lod in chain.Levels)
                WriteLod(writer, lod);
        }

        File.Move(tempPath, path, overwrite: true);
    }

    /// <summary>Removes a cache entry when present.</summary>
    public bool Evict(string key) => TryDelete(PathForKey(key));

    private string PathForKey(string key) => Path.Combine(_directory, key + ".gdnnlod");

    private static bool TryDelete(string path)
    {
        try
        {
            if (!File.Exists(path)) return false;
            File.Delete(path);
            return true;
        }
        catch (IOException)
        {
            return false;
        }
    }

    private static void AppendFloats(IncrementalHash hash, ReadOnlySpan<float> values)
        => hash.AppendData(MemoryMarshal.AsBytes(values));

    private static void WriteLod(BinaryWriter writer, NeuralPolygonLod lod)
    {
        writer.Write(lod.Level);
        writer.Write(lod.GridResolution);
        writer.Write(lod.GeometricError);

        WriteFloatArray(writer, MemoryMarshal.Cast<Vector3, float>(lod.Mesh.Positions));
        WriteFloatArray(writer, MemoryMarshal.Cast<Vector3, float>(lod.Mesh.Normals));
        WriteIntArray(writer, lod.Mesh.Indices);

        writer.Write(lod.Meshlets.Count);
        foreach (var meshlet in lod.Meshlets)
        {
            WriteIntArray(writer, meshlet.VertexIndices);
            writer.Write(meshlet.LocalIndices.Length);
            writer.Write(meshlet.LocalIndices);
            WriteVector3(writer, meshlet.Bounds.Center);
            WriteVector3(writer, meshlet.Bounds.HalfExtents);
            WriteVector3(writer, meshlet.ConeAxis);
            writer.Write(meshlet.ConeCutoff);
        }
    }

    private static NeuralPolygonLod ReadLod(BinaryReader reader)
    {
        int level = reader.ReadInt32();
        int gridResolution = reader.ReadInt32();
        float geometricError = reader.ReadSingle();

        float[] positionData = ReadFloatArray(reader);
        float[] normalData = ReadFloatArray(reader);
        int[] indices = ReadIntArray(reader);

        var mesh = new NeuralPolygonMesh
        {
            Positions = MemoryMarshal.Cast<float, Vector3>(positionData).ToArray(),
            Normals = MemoryMarshal.Cast<float, Vector3>(normalData).ToArray(),
            Indices = indices
        };

        int meshletCount = reader.ReadInt32();
        var meshlets = new List<NeuralMeshlet>(meshletCount);
        for (int i = 0; i < meshletCount; i++)
        {
            int[] vertexIndices = ReadIntArray(reader);
            int localLength = reader.ReadInt32();
            byte[] localIndices = reader.ReadBytes(localLength);
            var bounds = new AABB(ReadVector3(reader), ReadVector3(reader));
            Vector3 coneAxis = ReadVector3(reader);
            float coneCutoff = reader.ReadSingle();

            meshlets.Add(new NeuralMeshlet
            {
                VertexIndices = vertexIndices,
                LocalIndices = localIndices,
                Bounds = bounds,
                ConeAxis = coneAxis,
                ConeCutoff = coneCutoff
            });
        }

        return new NeuralPolygonLod
        {
            Level = level,
            GridResolution = gridResolution,
            Mesh = mesh,
            Meshlets = meshlets,
            GeometricError = geometricError
        };
    }

    private static void WriteFloatArray(BinaryWriter writer, ReadOnlySpan<float> values)
    {
        writer.Write(values.Length);
        writer.Write(MemoryMarshal.AsBytes(values));
    }

    private static float[] ReadFloatArray(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        var values = new float[length];
        ReadExactly(reader, MemoryMarshal.AsBytes(values.AsSpan()));
        return values;
    }

    private static void WriteIntArray(BinaryWriter writer, ReadOnlySpan<int> values)
    {
        writer.Write(values.Length);
        writer.Write(MemoryMarshal.AsBytes(values));
    }

    private static int[] ReadIntArray(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        var values = new int[length];
        ReadExactly(reader, MemoryMarshal.AsBytes(values.AsSpan()));
        return values;
    }

    private static void ReadExactly(BinaryReader reader, Span<byte> destination)
    {
        int total = 0;
        while (total < destination.Length)
        {
            int read = reader.Read(destination[total..]);
            if (read <= 0)
                throw new EndOfStreamException("Truncated polygonization cache entry.");
            total += read;
        }
    }

    private static void WriteVector3(BinaryWriter writer, Vector3 v)
    {
        writer.Write(v.X);
        writer.Write(v.Y);
        writer.Write(v.Z);
    }

    private static Vector3 ReadVector3(BinaryReader reader)
        => new(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
}
