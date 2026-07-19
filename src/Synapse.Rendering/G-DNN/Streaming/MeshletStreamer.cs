using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using GDNN.Core.DataStructures;
using GDNN.Polygonization;

namespace GDNN.Streaming;

/// <summary>Identifiant d'un cluster dans un fichier de pages (niveau LOD, index).</summary>
public readonly record struct ClusterKey(int Level, int Index);

/// <summary>
/// Cluster décodé, autonome (positions et normales locales, plus de référence
/// au maillage source) — l'unité de résidence mémoire du streaming.
/// </summary>
public sealed class StreamedCluster
{
    public Vector3[] Positions { get; init; }
    public Vector3[] Normals { get; init; }
    public byte[] LocalIndices { get; init; }
    public AABB Bounds { get; init; }
    public Vector3 ConeAxis { get; init; }
    public float ConeCutoff { get; init; }

    public int VertexCount => Positions.Length;
    public int TriangleCount => LocalIndices.Length / 3;

    /// <summary>Empreinte mémoire décodée (pour le budget de résidence).</summary>
    public int DecodedBytes =>
        Positions.Length * 12 + Normals.Length * 12 + LocalIndices.Length + 64;
}

/// <summary>
/// Codec de cluster : positions quantifiées 16 bits par axe dans les bornes du
/// meshlet, normales octaédriques 2×8 bits, indices locaux bruts, puis Deflate.
/// L'erreur de quantification est bornée par la taille du meshlet / 65535 —
/// une garantie, vérifiée par test, pas une promesse.
/// </summary>
public static class MeshletCodec
{
    public static byte[] Encode(NeuralPolygonMesh mesh, NeuralMeshlet meshlet)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(meshlet);

        using var raw = new MemoryStream();
        using (var w = new BinaryWriter(raw, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            Vector3 min = meshlet.Bounds.Min;
            Vector3 size = meshlet.Bounds.Size;
            Vector3 invSize = new(
                size.X > 0 ? 1f / size.X : 0f,
                size.Y > 0 ? 1f / size.Y : 0f,
                size.Z > 0 ? 1f / size.Z : 0f);

            w.Write((ushort)meshlet.VertexCount);
            w.Write((ushort)(meshlet.LocalIndices.Length / 3));
            WriteVector3(w, min);
            WriteVector3(w, size);
            WriteVector3(w, meshlet.ConeAxis);
            w.Write(meshlet.ConeCutoff);

            foreach (int vi in meshlet.VertexIndices)
            {
                Vector3 local = (mesh.Positions[vi] - min) * invSize;
                w.Write((ushort)Math.Clamp(local.X * 65535f, 0f, 65535f));
                w.Write((ushort)Math.Clamp(local.Y * 65535f, 0f, 65535f));
                w.Write((ushort)Math.Clamp(local.Z * 65535f, 0f, 65535f));
            }

            foreach (int vi in meshlet.VertexIndices)
            {
                var (ox, oy) = OctEncode(mesh.Normals[vi]);
                w.Write(ox);
                w.Write(oy);
            }

            w.Write(meshlet.LocalIndices);
        }

        using var compressed = new MemoryStream();
        using (var deflate = new DeflateStream(compressed, CompressionLevel.Optimal, leaveOpen: true))
            deflate.Write(raw.GetBuffer(), 0, (int)raw.Length);
        return compressed.ToArray();
    }

    public static StreamedCluster Decode(ReadOnlySpan<byte> data)
    {
        using var input = new MemoryStream(data.ToArray());
        using var deflate = new DeflateStream(input, CompressionMode.Decompress);
        using var raw = new MemoryStream();
        deflate.CopyTo(raw);
        raw.Position = 0;

        using var r = new BinaryReader(raw);
        int vertexCount = r.ReadUInt16();
        int triangleCount = r.ReadUInt16();
        Vector3 min = ReadVector3(r);
        Vector3 size = ReadVector3(r);
        Vector3 coneAxis = ReadVector3(r);
        float coneCutoff = r.ReadSingle();

        var positions = new Vector3[vertexCount];
        for (int i = 0; i < vertexCount; i++)
        {
            positions[i] = min + new Vector3(
                r.ReadUInt16() / 65535f * size.X,
                r.ReadUInt16() / 65535f * size.Y,
                r.ReadUInt16() / 65535f * size.Z);
        }

        var normals = new Vector3[vertexCount];
        for (int i = 0; i < vertexCount; i++)
            normals[i] = OctDecode(r.ReadByte(), r.ReadByte());

        var indices = r.ReadBytes(triangleCount * 3);

        return new StreamedCluster
        {
            Positions = positions,
            Normals = normals,
            LocalIndices = indices,
            Bounds = new AABB(min, min + size, fromCorners: true),
            ConeAxis = coneAxis,
            ConeCutoff = coneCutoff
        };
    }

    /// <summary>Encodage octaédrique d'une normale unitaire vers 2×snorm8.</summary>
    internal static (byte X, byte Y) OctEncode(Vector3 n)
    {
        float sum = MathF.Abs(n.X) + MathF.Abs(n.Y) + MathF.Abs(n.Z);
        if (sum < 1e-12f)
            return (127, 127);

        float ox = n.X / sum, oy = n.Y / sum;
        if (n.Z < 0f)
        {
            (ox, oy) = (
                (1f - MathF.Abs(oy)) * (ox >= 0f ? 1f : -1f),
                (1f - MathF.Abs(ox)) * (oy >= 0f ? 1f : -1f));
        }
        return (
            (byte)Math.Clamp((ox * 0.5f + 0.5f) * 255f, 0f, 255f),
            (byte)Math.Clamp((oy * 0.5f + 0.5f) * 255f, 0f, 255f));
    }

    internal static Vector3 OctDecode(byte bx, byte by)
    {
        float ox = bx / 255f * 2f - 1f;
        float oy = by / 255f * 2f - 1f;
        float oz = 1f - MathF.Abs(ox) - MathF.Abs(oy);
        if (oz < 0f)
        {
            (ox, oy) = (
                (1f - MathF.Abs(oy)) * (ox >= 0f ? 1f : -1f),
                (1f - MathF.Abs(ox)) * (oy >= 0f ? 1f : -1f));
        }
        var v = new Vector3(ox, oy, oz);
        return v.LengthSquared() > 1e-12f ? Vector3.Normalize(v) : Vector3.UnitY;
    }

    private static void WriteVector3(BinaryWriter w, Vector3 v)
    {
        w.Write(v.X);
        w.Write(v.Y);
        w.Write(v.Z);
    }

    private static Vector3 ReadVector3(BinaryReader r)
        => new(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
}

/// <summary>Entrée du répertoire d'un fichier de pages : tout ce qu'il faut pour
/// prioriser et culler un cluster SANS le charger.</summary>
public sealed class ClusterDirectoryEntry
{
    public ClusterKey Key { get; init; }
    public AABB Bounds { get; init; }
    public float GeometricError { get; init; }
    public long Offset { get; init; }
    public int Length { get; init; }
}

/// <summary>
/// Fichier de pages hors-cœur : tous les clusters d'une chaîne de LOD,
/// compressés, avec un répertoire en tête (bornes + erreur par cluster) qui
/// reste seul résident en mémoire.
/// </summary>
public sealed class MeshletPageFile : IDisposable
{
    private const uint Magic = 0x4E504C47; // "GLPN"
    private readonly FileStream _stream;
    private readonly object _ioLock = new();
    private readonly List<ClusterDirectoryEntry> _directory;
    private readonly Dictionary<ClusterKey, ClusterDirectoryEntry> _byKey;
    private bool _disposed;

    public IReadOnlyList<ClusterDirectoryEntry> Directory => _directory;
    public int LevelCount { get; }

    private MeshletPageFile(FileStream stream, List<ClusterDirectoryEntry> directory, int levelCount)
    {
        _stream = stream;
        _directory = directory;
        _byKey = directory.ToDictionary(e => e.Key);
        LevelCount = levelCount;
    }

    /// <summary>Écrit la chaîne complète dans un fichier de pages.</summary>
    public static void Build(NeuralPolygonLodChain chain, string path)
    {
        ArgumentNullException.ThrowIfNull(chain);

        var payloads = new List<(ClusterKey Key, AABB Bounds, float Error, byte[] Data)>();
        for (int level = 0; level < chain.Levels.Count; level++)
        {
            var lod = chain.Levels[level];
            for (int i = 0; i < lod.Meshlets.Count; i++)
            {
                payloads.Add((
                    new ClusterKey(level, i),
                    lod.Meshlets[i].Bounds,
                    lod.GeometricError,
                    MeshletCodec.Encode(lod.Mesh, lod.Meshlets[i])));
            }
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var w = new BinaryWriter(stream);
        w.Write(Magic);
        w.Write(chain.Levels.Count);
        w.Write(payloads.Count);

        // Répertoire d'abord (offsets calculés après l'en-tête fixe).
        long headerSize = 4 + 4 + 4 + payloads.Count * (4 + 4 + 24 + 4 + 8 + 4);
        long offset = headerSize;
        foreach (var (key, bounds, error, data) in payloads)
        {
            w.Write(key.Level);
            w.Write(key.Index);
            w.Write(bounds.Min.X);
            w.Write(bounds.Min.Y);
            w.Write(bounds.Min.Z);
            w.Write(bounds.Max.X);
            w.Write(bounds.Max.Y);
            w.Write(bounds.Max.Z);
            w.Write(error);
            w.Write(offset);
            w.Write(data.Length);
            offset += data.Length;
        }

        foreach (var (_, _, _, data) in payloads)
            w.Write(data);
    }

    public static MeshletPageFile Open(string path)
    {
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        var r = new BinaryReader(stream);
        if (r.ReadUInt32() != Magic)
        {
            stream.Dispose();
            throw new InvalidDataException("Not a meshlet page file.");
        }

        int levelCount = r.ReadInt32();
        int clusterCount = r.ReadInt32();
        var directory = new List<ClusterDirectoryEntry>(clusterCount);
        for (int i = 0; i < clusterCount; i++)
        {
            var key = new ClusterKey(r.ReadInt32(), r.ReadInt32());
            var min = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
            var max = new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());
            float error = r.ReadSingle();
            long clusterOffset = r.ReadInt64();
            int length = r.ReadInt32();
            directory.Add(new ClusterDirectoryEntry
            {
                Key = key,
                Bounds = new AABB(min, max, fromCorners: true),
                GeometricError = error,
                Offset = clusterOffset,
                Length = length
            });
        }

        return new MeshletPageFile(stream, directory, levelCount);
    }

    public StreamedCluster ReadCluster(ClusterKey key)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var entry = _byKey[key];
        var buffer = new byte[entry.Length];
        lock (_ioLock)
        {
            _stream.Seek(entry.Offset, SeekOrigin.Begin);
            _stream.ReadExactly(buffer);
        }
        return MeshletCodec.Decode(buffer);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _stream.Dispose();
    }
}

/// <summary>Compteurs mesurés du streamer.</summary>
public sealed class StreamerStats
{
    public long ResidentBytes { get; internal set; }
    public int ResidentClusters { get; internal set; }
    public long CacheHits { get; internal set; }
    public long CacheMisses { get; internal set; }
    public long Evictions { get; internal set; }

    public override string ToString() =>
        $"{ResidentClusters} clusters / {ResidentBytes} o résidents, " +
        $"hits={CacheHits}, misses={CacheMisses}, évictions={Evictions}";
}

/// <summary>
/// Streaming de clusters hors-cœur : cache LRU sous budget mémoire strict,
/// priorisation par visibilité (frustum + distance sur le répertoire, sans
/// charger les données), préchargement asynchrone.
/// </summary>
public sealed class MeshletStreamer : IDisposable
{
    private readonly MeshletPageFile _file;
    private readonly long _memoryBudgetBytes;
    private readonly Dictionary<ClusterKey, LinkedListNode<(ClusterKey Key, StreamedCluster Cluster)>> _resident = new();
    private readonly LinkedList<(ClusterKey Key, StreamedCluster Cluster)> _lru = new();
    private readonly object _lock = new();
    private bool _disposed;

    public StreamerStats Stats { get; } = new();

    public MeshletStreamer(MeshletPageFile file, long memoryBudgetBytes)
    {
        ArgumentNullException.ThrowIfNull(file);
        if (memoryBudgetBytes < 1)
            throw new ArgumentOutOfRangeException(nameof(memoryBudgetBytes));
        _file = file;
        _memoryBudgetBytes = memoryBudgetBytes;
    }

    /// <summary>Cluster résident sans provoquer de chargement.</summary>
    public bool TryGetResident(ClusterKey key, out StreamedCluster cluster)
    {
        lock (_lock)
        {
            if (_resident.TryGetValue(key, out var node))
            {
                Touch(node);
                Stats.CacheHits++;
                cluster = node.Value.Cluster;
                return true;
            }
        }
        cluster = null!;
        return false;
    }

    /// <summary>Chargement bloquant avec insertion LRU et éviction sous budget.</summary>
    public StreamedCluster GetOrLoad(ClusterKey key)
    {
        lock (_lock)
        {
            if (_resident.TryGetValue(key, out var node))
            {
                Touch(node);
                Stats.CacheHits++;
                return node.Value.Cluster;
            }
        }

        // I/O + décodage hors verrou : les chargements concurrents restent parallèles.
        var cluster = _file.ReadCluster(key);

        lock (_lock)
        {
            if (_resident.TryGetValue(key, out var existing))
            {
                Touch(existing);
                return existing.Value.Cluster;
            }

            Stats.CacheMisses++;
            var node = _lru.AddFirst((key, cluster));
            _resident[key] = node;
            Stats.ResidentBytes += cluster.DecodedBytes;
            Stats.ResidentClusters++;
            EvictOverBudget(justLoaded: key);
            return cluster;
        }
    }

    /// <summary>
    /// Clusters d'un niveau visibles par la caméra, triés du plus proche au plus
    /// lointain — décision prise sur le répertoire seul (0 octet chargé).
    /// </summary>
    public IReadOnlyList<ClusterKey> QueryVisible(in CameraView camera, int level)
    {
        var frustum = Frustum.FromViewProjection(camera.ViewProjection);
        Vector3 position = camera.Position;

        var visible = new List<(ClusterKey Key, float DistSq)>();
        foreach (var entry in _file.Directory)
        {
            if (entry.Key.Level != level)
                continue;
            if (frustum.TestAABB(entry.Bounds) == FrustumTest.Outside)
                continue;
            visible.Add((entry.Key, (entry.Bounds.Center - position).LengthSquared()));
        }

        visible.Sort((a, b) => a.DistSq.CompareTo(b.DistSq));
        return visible.Select(v => v.Key).ToList();
    }

    /// <summary>Préchargement asynchrone (ex. pendant que la frame se rend).</summary>
    public Task PrefetchAsync(IEnumerable<ClusterKey> keys)
    {
        var snapshot = keys.ToArray();
        return Task.Run(() =>
        {
            foreach (var key in snapshot)
                GetOrLoad(key);
        });
    }

    private void Touch(LinkedListNode<(ClusterKey, StreamedCluster)> node)
    {
        _lru.Remove(node);
        _lru.AddFirst(node);
    }

    private void EvictOverBudget(ClusterKey justLoaded)
    {
        var node = _lru.Last;
        while (Stats.ResidentBytes > _memoryBudgetBytes && node != null)
        {
            var previous = node.Previous;
            if (!node.Value.Key.Equals(justLoaded)) // jamais évincer ce qu'on vient de servir
            {
                _lru.Remove(node);
                _resident.Remove(node.Value.Key);
                Stats.ResidentBytes -= node.Value.Cluster.DecodedBytes;
                Stats.ResidentClusters--;
                Stats.Evictions++;
            }
            node = previous;
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        lock (_lock)
        {
            _lru.Clear();
            _resident.Clear();
        }
    }
}
