using System;
// ============================================================
// FILE: NeuralAsset.cs
// PATH: Core/NeuralNetwork/NeuralAsset.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers;
using System.Buffers.Binary;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics;
using System.IO;
using System.IO;
using System.IO.Compression;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text;
using System.Text.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks;
using GDNN.Rendering.Compat;

namespace GDNN.Core.NeuralNetwork;

/// <summary>
/// Represents the LOD (Level of Detail) tier for a neural asset.
/// Each tier contains a different resolution MicroMLP for distance-based rendering.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter))]
public enum LODLevel : byte
{
    /// <summary>Full resolution — closest camera distances.</summary>
    LOD0 = 0,

    /// <summary>Medium-high resolution.</summary>
    LOD1 = 1,

    /// <summary>Medium resolution.</summary>
    LOD2 = 2,

    /// <summary>Low resolution — farthest camera distances.</summary>
    LOD3 = 3,

    /// <summary>Impostor / billboard — extreme distance.</summary>
    LOD4 = 4
}

/// <summary>
/// Represents a single LOD tier containing a MicroMLP and associated metadata.
/// </summary>
public sealed class LODTier
{
    /// <summary>Gets or sets the LOD level for this tier.</summary>
    public LODLevel Level { get; set; }

    /// <summary>Gets or sets the compressed neural weights for this LOD tier.</summary>
    public byte[] CompressedWeights { get; set; } = [];

    /// <summary>Gets or sets the uncompressed MicroMLP weights.</summary>
    public float[] UncompressedWeights { get; set; } = [];

    /// <summary>Gets or sets the maximum screen-space pixel error allowed at this LOD.</summary>
    public float MaxScreenError { get; set; }

    /// <summary>Gets or sets the distance range (start, end) at which this LOD is used.</summary>
    public Vector2 DistanceRange { get; set; }

    /// <summary>Gets or sets the triangle count approximation for this LOD tier.</summary>
    public int ApproximateTriangleCount { get; set; }

    /// <summary>Gets the uncompressed weight count.</summary>
    public int WeightCount => UncompressedWeights?.Length ?? 0;

    /// <summary>
    /// Compresses the uncompressed weights using Brotli compression.
    /// </summary>
    public void Compress()
    {
        if (UncompressedWeights == null || UncompressedWeights.Length == 0)
            return;

        byte[] floatBytes = MemoryMarshal.AsBytes(UncompressedWeights.AsSpan()).ToArray();
        CompressedWeights = BrotliCompat.Compress(floatBytes, System.IO.Compression.CompressionLevel.Optimal);
    }

    /// <summary>
    /// Decompresses the compressed weights into the uncompressed buffer.
    /// </summary>
    public void Decompress()
    {
        if (CompressedWeights == null || CompressedWeights.Length == 0)
            return;

        byte[] decompressed = BrotliCompat.Decompress(CompressedWeights);
        UncompressedWeights = MemoryMarshal.Cast<byte, float>(decompressed).ToArray();
    }

    /// <summary>
    /// Computes the memory footprint of this LOD tier in bytes.
    /// </summary>
    public long ComputeMemoryFootprint()
    {
        long compressed = CompressedWeights?.Length ?? 0;
        long uncompressed = (UncompressedWeights?.Length ?? 0) * sizeof(float);
        return compressed + uncompressed + sizeof(float) * 4 + sizeof(int); // metadata overhead
    }
}

/// <summary>
/// Stores micro-texture coordinates for neural geometry patches.
/// These encode UV mapping information for texture-space neural rendering.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct MicroTextureCoordinates
{
    /// <summary>First UV coordinate pair (u0, v0).</summary>
    public Vector2 UV0;

    /// <summary>Second UV coordinate pair (u1, v1).</summary>
    public Vector2 UV1;

    /// <summary>Third UV coordinate pair (u2, v2).</summary>
    public Vector2 UV2;

    /// <summary>Texture atlas index.</summary>
    public int AtlasIndex;

    /// <summary>Mip level bias for this coordinate set.</summary>
    public float MipBias;

    /// <summary>Gets the total byte size of this struct.</summary>
    public const int SizeInBytes = 32;

    /// <summary>
    /// Interpolates between UV0, UV1, UV2 using barycentric coordinates.
    /// </summary>
    /// <param name="bary">Barycentric coordinates (λ0, λ1, λ2).</param>
    /// <returns>Interpolated UV coordinate.</returns>
    public readonly Vector2 Interpolate(Vector3 bary)
    {
        return UV0 * bary.X + UV1 * bary.Y + UV2 * bary.Z;
    }

    /// <summary>
    /// Serializes to a byte span.
    /// </summary>
    public readonly void Serialize(Span<byte> destination)
    {
        Debug.Assert(destination.Length >= SizeInBytes);
        MemoryMarshal.Write(destination, ref Unsafe.AsRef(in this));
    }

    /// <summary>
    /// Deserializes from a byte span.
    /// </summary>
    public static MicroTextureCoordinates Deserialize(ReadOnlySpan<byte> source)
    {
        Debug.Assert(source.Length >= SizeInBytes);
        return MemoryMarshal.Read<MicroTextureCoordinates>(source);
    }
}

/// <summary>
/// Represents a reference to an animation clip associated with a neural asset.
/// </summary>
public sealed class AnimationClipReference
{
    /// <summary>Gets or sets the unique identifier for this animation clip.</summary>
    public Guid ClipId { get; set; }

    /// <summary>Gets or sets the display name of the animation clip.</summary>
    public string ClipName { get; set; } = string.Empty;

    /// <summary>Gets or sets the duration in seconds.</summary>
    public float DurationSeconds { get; set; }

    /// <summary>Gets or sets the frame rate (frames per second).</summary>
    public float FrameRate { get; set; }

    /// <summary>Gets or sets the number of keyframes in the clip.</summary>
    public int KeyframeCount { get; set; }

    /// <summary>Gets or sets the compressed keyframe data.</summary>
    public byte[] CompressedKeyframes { get; set; } = [];

    /// <summary>Gets or sets the animation channel mask (which joints/vertices are affected).</summary>
    public ulong ChannelMask { get; set; }

    /// <summary>
    /// Computes the data rate in bytes per second for this clip.
    /// </summary>
    public float ComputeDataRate()
    {
        if (DurationSeconds <= 0)
            return 0f;
        return CompressedKeyframes.Length / DurationSeconds;
    }

    /// <summary>
    /// Serializes the animation clip reference to a byte array.
    /// </summary>
    public byte[] Serialize()
    {
        using var ms = new MemoryStream();
        Span<byte> header = stackalloc byte[64];

        BinaryPrimitives.WriteUInt32LittleEndian(header[0..4], 0x414E494D); // "ANIM"
        BinaryPrimitives.WriteInt32LittleEndian(header[4..8], 1); // version
        header.Slice(8, 16).CopyTo(header); // ClipId bytes
        GuidCompat.WriteBytes(header.Slice(8, 16), ClipId);
        BinaryPrimitives.WriteSingleLittleEndian(header[24..28], DurationSeconds);
        BinaryPrimitives.WriteSingleLittleEndian(header[28..32], FrameRate);
        BinaryPrimitives.WriteInt32LittleEndian(header[32..36], KeyframeCount);
        BinaryPrimitives.WriteUInt64LittleEndian(header[40..48], ChannelMask);

        ms.Write(header);
        ms.Write(CompressedKeyframes);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes an animation clip reference from a byte span.
    /// </summary>
    public static AnimationClipReference Deserialize(ReadOnlySpan<byte> data)
    {
        Debug.Assert(data.Length >= 64);

        var clip = new AnimationClipReference
        {
            ClipId = new Guid(data.Slice(8, 16)),
            DurationSeconds = BinaryPrimitives.ReadSingleLittleEndian(data[24..28]),
            FrameRate = BinaryPrimitives.ReadSingleLittleEndian(data[28..32]),
            KeyframeCount = BinaryPrimitives.ReadInt32LittleEndian(data[32..36]),
            ChannelMask = BinaryPrimitives.ReadUInt64LittleEndian(data[40..48])
        };

        if (data.Length > 64)
        {
            clip.CompressedKeyframes = data[64..].ToArray();
        }

        return clip;
    }
}

/// <summary>
/// Represents an axis-aligned bounding box for a neural asset.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct BoundingBox
{
    /// <summary>Minimum corner of the AABB.</summary>
    public Vector3 Min;

    /// <summary>Maximum corner of the AABB.</summary>
    public Vector3 Max;

    /// <summary>Gets the size (extents) of the bounding box.</summary>
    public readonly Vector3 Size => Max - Min;

    /// <summary>Gets the center of the bounding box.</summary>
    public readonly Vector3 Center => (Min + Max) * 0.5f;

    /// <summary>Gets the surface area of the bounding box.</summary>
    public readonly float SurfaceArea
    {
        get
        {
            var s = Size;
            return 2f * (s.X * s.Y + s.Y * s.Z + s.Z * s.X);
        }
    }

    /// <summary>Gets the volume of the bounding box.</summary>
    public readonly float Volume
    {
        get
        {
            var s = Size;
            return s.X * s.Y * s.Z;
        }
    }

    /// <summary>Gets the diagonal length of the bounding box.</summary>
    public readonly float DiagonalLength => Size.Length();

    /// <summary>
    /// Tests whether a point is inside the AABB.
    /// </summary>
    public readonly bool Contains(Vector3 point)
    {
        return point.X >= Min.X && point.X <= Max.X &&
               point.Y >= Min.Y && point.Y <= Max.Y &&
               point.Z >= Min.Z && point.Z <= Max.Z;
    }

    /// <summary>
    /// Tests whether this AABB intersects another AABB.
    /// </summary>
    public readonly bool Intersects(in BoundingBox other)
    {
        return Min.X <= other.Max.X && Max.X >= other.Min.X &&
               Min.Y <= other.Max.Y && Max.Y >= other.Min.Y &&
               Min.Z <= other.Max.Z && Max.Z >= other.Min.Z;
    }

    /// <summary>
    /// Expands this AABB to include the given point.
    /// </summary>
    public void Encapsulate(Vector3 point)
    {
        Min = Vector3.Min(Min, point);
        Max = Vector3.Max(Max, point);
    }

    /// <summary>
    /// Expands this AABB to include another AABB.
    /// </summary>
    public void Encapsulate(in BoundingBox other)
    {
        Min = Vector3.Min(Min, other.Min);
        Max = Vector3.Max(Max, other.Max);
    }

    /// <summary>
    /// Gets the 8 corner vertices of the AABB.
    /// </summary>
    public readonly void GetCorners(Span<Vector3> corners)
    {
        Debug.Assert(corners.Length >= 8);
        corners[0] = new Vector3(Min.X, Min.Y, Min.Z);
        corners[1] = new Vector3(Max.X, Min.Y, Min.Z);
        corners[2] = new Vector3(Min.X, Max.Y, Min.Z);
        corners[3] = new Vector3(Max.X, Max.Y, Min.Z);
        corners[4] = new Vector3(Min.X, Min.Y, Max.Z);
        corners[5] = new Vector3(Max.X, Min.Y, Max.Z);
        corners[6] = new Vector3(Min.X, Max.Y, Max.Z);
        corners[7] = new Vector3(Max.X, Max.Y, Max.Z);
    }

    /// <summary>Gets the byte size of this struct.</summary>
    public const int SizeInBytes = 24;

    /// <summary>
    /// Serializes the bounding box to a byte span.
    /// </summary>
    public readonly void Serialize(Span<byte> destination)
    {
        Debug.Assert(destination.Length >= SizeInBytes);
        MemoryMarshal.Write(destination, ref Unsafe.AsRef(in this));
    }

    /// <summary>
    /// Deserializes a bounding box from a byte span.
    /// </summary>
    public static BoundingBox Deserialize(ReadOnlySpan<byte> source)
    {
        Debug.Assert(source.Length >= SizeInBytes);
        return MemoryMarshal.Read<BoundingBox>(source);
    }
}

/// <summary>
/// Represents a bounding sphere for a neural asset.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct BoundingSphere
{
    /// <summary>Center of the sphere in world space.</summary>
    public Vector3 Center;

    /// <summary>Radius of the sphere.</summary>
    public float Radius;

    /// <summary>Gets the diameter of the sphere.</summary>
    public readonly float Diameter => Radius * 2f;

    /// <summary>Gets the volume of the sphere.</summary>
    public readonly float Volume => (4f / 3f) * MathF.PI * Radius * Radius * Radius;

    /// <summary>Gets the surface area of the sphere.</summary>
    public readonly float SurfaceArea => 4f * MathF.PI * Radius * Radius;

    /// <summary>
    /// Tests whether a point is inside the sphere.
    /// </summary>
    public readonly bool Contains(Vector3 point)
    {
        return Vector3.DistanceSquared(point, Center) <= Radius * Radius;
    }

    /// <summary>
    /// Tests whether this sphere intersects another sphere.
    /// </summary>
    public readonly bool Intersects(in BoundingSphere other)
    {
        float distSq = Vector3.DistanceSquared(Center, other.Center);
        float combinedRadius = Radius + other.Radius;
        return distSq <= combinedRadius * combinedRadius;
    }

    /// <summary>
    /// Tests whether this sphere intersects an AABB.
    /// </summary>
    public readonly bool Intersects(in BoundingBox box)
    {
        Vector3 closest = Vector3.Clamp(Center, box.Min, box.Max);
        float distSq = Vector3.DistanceSquared(Center, closest);
        return distSq <= Radius * Radius;
    }

    /// <summary>
    /// Creates a bounding sphere that encloses the given AABB.
    /// </summary>
    public static BoundingSphere FromBoundingBox(in BoundingBox box)
    {
        return new BoundingSphere
        {
            Center = box.Center,
            Radius = box.DiagonalLength * 0.5f
        };
    }

    /// <summary>Gets the byte size of this struct.</summary>
    public const int SizeInBytes = 16;

    /// <summary>
    /// Serializes the bounding sphere to a byte span.
    /// </summary>
    public readonly void Serialize(Span<byte> destination)
    {
        Debug.Assert(destination.Length >= SizeInBytes);
        MemoryMarshal.Write(destination, ref Unsafe.AsRef(in this));
    }

    /// <summary>
    /// Deserializes a bounding sphere from a byte span.
    /// </summary>
    public static BoundingSphere Deserialize(ReadOnlySpan<byte> source)
    {
        Debug.Assert(source.Length >= SizeInBytes);
        return MemoryMarshal.Read<BoundingSphere>(source);
    }
}

/// <summary>
/// Represents metadata for a neural 3D asset including versioning, authoring, and provenance.
/// </summary>
public sealed class NeuralAssetMetadata
{
    /// <summary>Gets or sets the human-readable asset name.</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Gets or sets the asset version string (semver format).</summary>
    public string Version { get; set; } = "1.0.0";

    /// <summary>Gets or sets the creation timestamp (UTC).</summary>
    public DateTime CreationDate { get; set; } = DateTime.UtcNow;

    /// <summary>Gets or sets the last modification timestamp (UTC).</summary>
    public DateTime LastModifiedDate { get; set; } = DateTime.UtcNow;

    /// <summary>Gets or sets the author or creator of the asset.</summary>
    public string Author { get; set; } = string.Empty;

    /// <summary>Gets or sets the description or notes for the asset.</summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>Gets or sets a content hash for integrity verification.</summary>
    public string ContentHash { get; set; } = string.Empty;

    /// <summary>Gets or sets the source engine or tool that produced this asset.</summary>
    public string SourceEngine { get; set; } = "GDNN";

    /// <summary>Gets or sets the license identifier for this asset.</summary>
    public string License { get; set; } = string.Empty;

    /// <summary>Gets or sets arbitrary tag strings for categorization.</summary>
    public string[] Tags { get; set; } = [];

    /// <summary>
    /// Serializes the metadata to JSON.
    /// </summary>
    public string ToJson()
    {
        return JsonSerializer.Serialize(this, new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
    }

    /// <summary>
    /// Deserializes metadata from JSON.
    /// </summary>
    public static NeuralAssetMetadata FromJson(string json)
    {
        return JsonSerializer.Deserialize<NeuralAssetMetadata>(json, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        }) ?? new NeuralAssetMetadata();
    }

    /// <summary>
    /// Computes a content hash from the metadata fields.
    /// </summary>
    public string ComputeContentHash()
    {
        var sb = new StringBuilder();
        sb.Append(Name);
        sb.Append(Version);
        sb.Append(CreationDate.Ticks);
        sb.Append(Author);

        byte[] bytes = Encoding.UTF8.GetBytes(sb.ToString());
        var hash = System.Security.Cryptography.SHA256.HashData(bytes);
        return Convert.ToHexString(hash);
    }
}

/// <summary>
/// Represents a compressed neural 3D asset descriptor.
/// Contains MicroMLP weights, bounding geometry, LOD tiers, texture coordinates,
/// animation references, and metadata for a single neural mesh patch.
/// </summary>
public sealed class NeuralAsset
{
    /// <summary>Gets or sets the unique identifier for this asset.</summary>
    public Guid AssetId { get; set; } = Guid.NewGuid();

    /// <summary>
    /// Gets or sets the compressed neural weights for the base LOD level.
    /// Field-backed property with explicit backing field for C# 14 interop.
    /// </summary>
    private byte[] _compressedWeights = [];

    /// <summary>Gets or sets the compressed neural weights.</summary>
    public byte[] CompressedWeights
    {
        get => _compressedWeights;
        set => _compressedWeights = value ?? [];
    }

    /// <summary>Gets or sets the target mesh identifier that this neural asset is associated with.</summary>
    public Guid TargetMeshId { get; set; }

    /// <summary>
    /// Gets or sets the bounding box size (extents) of the asset.
    /// This is a convenience accessor; the full AABB is in <see cref="BoundingBox"/>.
    /// </summary>
    public Vector3 BoundingBoxSize
    {
        get => BoundingBox.Size;
        set
        {
            var center = BoundingBox.Center;
            var bbox = BoundingBox;
            bbox.Min = center - value * 0.5f;
            bbox.Max = center + value * 0.5f;
            BoundingBox = bbox;
            BoundingSphere = BoundingSphere.FromBoundingBox(BoundingBox);
        }
    }

    /// <summary>Gets or sets the axis-aligned bounding box for this asset.</summary>
    public BoundingBox BoundingBox { get; set; }

    /// <summary>Gets or sets the bounding sphere for this asset.</summary>
    public BoundingSphere BoundingSphere { get; set; }

    /// <summary>Gets or sets the LOD tiers for this asset.</summary>
    public List<LODTier> LODTiers { get; set; } = [];

    /// <summary>Gets or sets the micro-texture coordinates for this asset.</summary>
    public MicroTextureCoordinates TextureCoordinates { get; set; }

    /// <summary>Gets or sets the animation clip references for this asset.</summary>
    public List<AnimationClipReference> AnimationClips { get; set; } = [];

    /// <summary>Gets or sets the streaming priority (higher = load first).</summary>
    public int StreamingPriority { get; set; } = 0;

    /// <summary>Gets or sets the maximum loading distance in world units.</summary>
    public float MaxLoadDistance { get; set; } = 1000f;

    /// <summary>Gets or sets the minimum unloading distance in world units.</summary>
    public float MinUnloadDistance { get; set; } = 1200f;

    /// <summary>Gets or sets the LOD bias (positive = lower detail, negative = higher detail).</summary>
    public float LODBias { get; set; } = 0f;

    /// <summary>Gets or sets the asset metadata.</summary>
    public NeuralAssetMetadata Metadata { get; set; } = new();

    /// <summary>Gets or sets the uncompressed base MicroMLP weights (null if not loaded).</summary>
    [JsonIgnore]
    public float[]? UncompressedWeights { get; set; }

    /// <summary>Gets whether the base weights are currently loaded in memory.</summary>
    [JsonIgnore]
    public bool IsLoaded => UncompressedWeights != null && UncompressedWeights.Length > 0;

    /// <summary>Gets whether the asset is marked as dirty (needs re-serialization).</summary>
    [JsonIgnore]
    public bool IsDirty { get; private set; }

    /// <summary>Gets the LOD level closest to the specified screen-space pixel error.</summary>
    /// <param name="screenError">The target screen-space pixel error.</param>
    /// <returns>The best LOD tier, or null if none exist.</returns>
    public LODTier? GetBestLOD(float screenError)
    {
        LODTier? best = null;
        foreach (var tier in LODTiers)
        {
            if (tier.MaxScreenError >= screenError)
            {
                best = tier;
                break;
            }
        }
        return best ?? LODTiers.FirstOrDefault();
    }

    /// <summary>
    /// Gets the LOD tier for the specified LOD level.
    /// </summary>
    /// <param name="level">The desired LOD level.</param>
    /// <returns>The LOD tier, or null if not found.</returns>
    public LODTier? GetLOD(LODLevel level)
    {
        foreach (var tier in LODTiers)
        {
            if (tier.Level == level)
                return tier;
        }
        return null;
    }

    /// <summary>
    /// Selects the appropriate LOD level based on camera distance and LOD bias.
    /// </summary>
    /// <param name="cameraDistance">Distance from camera to asset center.</param>
    /// <returns>The selected LOD level.</returns>
    public LODLevel SelectLOD(float cameraDistance)
    {
        float adjustedDistance = cameraDistance * MathF.Pow(2f, LODBias);

        if (LODTiers.Count == 0)
            return LODLevel.LOD0;

        for (int i = LODTiers.Count - 1; i >= 0; i--)
        {
            if (adjustedDistance >= LODTiers[i].DistanceRange.X)
                return LODTiers[i].Level;
        }

        return LODTiers[0].Level;
    }

    /// <summary>
    /// Computes the streaming priority score based on distance, bounding size, and priority.
    /// Higher scores indicate higher loading priority.
    /// </summary>
    /// <param name="cameraDistance">Distance from camera to asset center.</param>
    /// <param name="screenSize">Projected screen size in pixels.</param>
    /// <returns>Priority score (higher = more important).</returns>
    public float ComputeStreamingPriority(float cameraDistance, float screenSize)
    {
        if (cameraDistance <= 0)
            return float.MaxValue;

        float distanceFactor = 1.0f / (1.0f + cameraDistance * 0.01f);
        float sizeFactor = screenSize / 100.0f;
        float priorityFactor = StreamingPriority * 0.1f;

        return distanceFactor * sizeFactor + priorityFactor;
    }

    /// <summary>
    /// Tests whether this asset should be loaded given the camera distance.
    /// </summary>
    /// <param name="cameraDistance">Distance from camera to asset center.</param>
    /// <returns>True if the asset should be loaded.</returns>
    public bool ShouldLoad(float cameraDistance)
    {
        return cameraDistance <= MaxLoadDistance;
    }

    /// <summary>
    /// Tests whether this asset should be unloaded given the camera distance.
    /// </summary>
    /// <param name="cameraDistance">Distance from camera to asset center.</param>
    /// <returns>True if the asset should be unloaded.</returns>
    public bool ShouldUnload(float cameraDistance)
    {
        return cameraDistance > MinUnloadDistance;
    }

    /// <summary>
    /// Compresses the uncompressed weights into the CompressedWeights buffer using Brotli.
    /// </summary>
    public void Compress()
    {
        if (UncompressedWeights == null || UncompressedWeights.Length == 0)
            return;

        byte[] floatBytes = MemoryMarshal.AsBytes(UncompressedWeights.AsSpan()).ToArray();
        _compressedWeights = BrotliCompat.Compress(floatBytes, CompressionLevel.Optimal);

        foreach (var tier in LODTiers)
            tier.Compress();

        IsDirty = true;
    }

    /// <summary>
    /// Decompresses the CompressedWeights buffer into UncompressedWeights.
    /// </summary>
    public void Decompress()
    {
        if (_compressedWeights.Length == 0)
            return;

        byte[] decompressed = BrotliCompat.Decompress(_compressedWeights);
        UncompressedWeights = MemoryMarshal.Cast<byte, float>(decompressed).ToArray();

        foreach (var tier in LODTiers)
            tier.Decompress();
    }

    /// <summary>
    /// Gets or creates a MicroMLP from the current uncompressed weights.
    /// </summary>
    /// <returns>The MicroMLP, or default if not loaded.</returns>
    public MicroMLP ToMicroMLP()
    {
        var mlp = new MicroMLP();
        if (UncompressedWeights != null && UncompressedWeights.Length >= MicroMLP.TotalWeightCount)
        {
            mlp.LoadFrom(UncompressedWeights.AsSpan(0, MicroMLP.TotalWeightCount));
        }
        return mlp;
    }

    /// <summary>
    /// Loads a MicroMLP into this asset, updating the uncompressed weights.
    /// </summary>
    /// <param name="mlp">The MicroMLP to load.</param>
    public void FromMicroMLP(in MicroMLP mlp)
    {
        UncompressedWeights = new float[MicroMLP.TotalWeightCount];
        mlp.CopyTo(UncompressedWeights);
        IsDirty = true;
    }

    /// <summary>
    /// Reconstructs a DeepMicroMLP when the asset stores Deep weights (~4385 floats).
    /// </summary>
    public DeepMicroMLP ToDeepMicroMLP()
    {
        int expected = DeepMicroMLP.GetTotalWeightCount();
        if (UncompressedWeights == null || UncompressedWeights.Length < expected)
            throw new InvalidOperationException(
                $"Asset weights ({UncompressedWeights?.Length ?? 0}) are too small for DeepMicroMLP ({expected}).");
        return new DeepMicroMLP(UncompressedWeights.AsSpan(0, expected));
    }

    /// <summary>
    /// Stores a DeepMicroMLP into this asset's uncompressed weights.
    /// </summary>
    public void FromDeepMicroMLP(DeepMicroMLP mlp)
    {
        ArgumentNullException.ThrowIfNull(mlp);
        UncompressedWeights = mlp.Serialize();
        IsDirty = true;
    }

    /// <summary>
    /// Stores a HashEncodedDeepMLP into this asset's uncompressed weights.
    /// </summary>
    public void FromHashEncodedDeepMLP(HashEncodedDeepMLP network)
    {
        ArgumentNullException.ThrowIfNull(network);
        UncompressedWeights = network.Serialize();
        Metadata.Tags = [.. Metadata.Tags, "HashEncodedDeepMLP"];
        IsDirty = true;
    }

    /// <summary>
    /// Reconstructs a HashEncodedDeepMLP when the asset stores hash-encoded weights.
    /// </summary>
    public HashEncodedDeepMLP ToHashEncodedDeepMLP()
    {
        int expected = HashEncodedDeepMLP.GetTotalWeightCount();
        if (UncompressedWeights == null || UncompressedWeights.Length < expected)
            throw new InvalidOperationException(
                $"Asset weights ({UncompressedWeights?.Length ?? 0}) are too small for HashEncodedDeepMLP ({expected}).");
        return HashEncodedDeepMLP.FromSerialized(UncompressedWeights.AsSpan(0, expected));
    }

    /// <summary>
    /// Returns the best matching ISdfNetwork based on weight count.
    /// </summary>
    public ISdfNetwork ToSdfNetwork()
    {
        if (UncompressedWeights == null || UncompressedWeights.Length == 0)
            return new MicroMLP();

        int hashCount = HashEncodedDeepMLP.GetTotalWeightCount();
        if (UncompressedWeights.Length >= hashCount)
            return ToHashEncodedDeepMLP();

        int deepCount = DeepMicroMLP.GetTotalWeightCount();
        if (UncompressedWeights.Length >= deepCount)
            return ToDeepMicroMLP();

        return ToMicroMLP();
    }

    /// <summary>
    /// Extracts a geometry descriptor from the current asset state.
    /// Computes a 16-dimensional descriptor from the bounding geometry and weight statistics.
    /// </summary>
    /// <returns>A geometry descriptor representing this asset.</returns>
    public GeometryDescriptor ExtractDescriptor()
    {
        var desc = new GeometryDescriptor();
        var s = desc.AsSpan();

        s[0] = BoundingBox.Center.X;
        s[1] = BoundingBox.Center.Y;
        s[2] = BoundingBox.Center.Z;
        s[3] = BoundingBoxSize.Length();

        s[4] = BoundingSphere.Radius;
        s[5] = BoundingBox.Volume;
        s[6] = BoundingBox.SurfaceArea;
        s[7] = LODTiers.Count;

        if (UncompressedWeights != null && UncompressedWeights.Length > 0)
        {
            float mean = 0f;
            float variance = 0f;
            int count = Math.Min(UncompressedWeights.Length, 216);

            for (int i = 0; i < count; i++)
                mean += UncompressedWeights[i];
            mean /= count;

            for (int i = 0; i < count; i++)
            {
                float d = UncompressedWeights[i] - mean;
                variance += d * d;
            }
            variance /= count;

            s[8] = mean;
            s[9] = variance;
            s[10] = UncompressedWeights.Length;
            s[11] = StreamingPriority;
        }

        s[12] = TextureCoordinates.UV0.X;
        s[13] = TextureCoordinates.UV0.Y;
        s[14] = AnimationClips.Count;
        s[15] = LODBias;

        return desc;
    }

    /// <summary>
    /// Validates the integrity and consistency of this neural asset.
    /// </summary>
    /// <returns>A list of validation error messages. Empty if valid.</returns>
    public List<string> Validate()
    {
        var errors = new List<string>();

        if (AssetId == Guid.Empty)
            errors.Add("AssetId is empty.");

        if (TargetMeshId == Guid.Empty)
            errors.Add("TargetMeshId is empty.");

        if (BoundingBoxSize.Length() <= 0)
            errors.Add("BoundingBoxSize has zero or negative length.");

        if (BoundingSphere.Radius <= 0)
            errors.Add("BoundingSphere radius is zero or negative.");

        if (MaxLoadDistance <= 0)
            errors.Add("MaxLoadDistance must be positive.");

        if (MinUnloadDistance <= MaxLoadDistance)
            errors.Add("MinUnloadDistance must be greater than MaxLoadDistance.");

        if (_compressedWeights.Length == 0 && (UncompressedWeights == null || UncompressedWeights.Length == 0))
            errors.Add("No weight data present (both compressed and uncompressed are empty).");

        if (UncompressedWeights != null && UncompressedWeights.Length > 0 &&
            UncompressedWeights.Length < MicroMLP.TotalWeightCount)
            errors.Add($"UncompressedWeights length ({UncompressedWeights.Length}) is less than MicroMLP.TotalWeightCount ({MicroMLP.TotalWeightCount}).");

        if (_compressedWeights.Length > 0)
        {
            try
            {
                byte[] test = BrotliCompat.Decompress(_compressedWeights);
                if (test.Length != UncompressedWeights?.Length * sizeof(float) &&
                    UncompressedWeights != null)
                {
                    errors.Add("Compressed weight decompression size mismatch.");
                }
            }
            catch (Exception ex)
            {
                errors.Add($"Compressed weights are corrupt: {ex.Message}");
            }
        }

        for (int i = 0; i < LODTiers.Count; i++)
        {
            var tier = LODTiers[i];
            if (tier.CompressedWeights.Length == 0 && tier.UncompressedWeights.Length == 0)
                errors.Add($"LOD tier {i} (LOD{tier.Level}) has no weight data.");

            if (tier.MaxScreenError <= 0)
                errors.Add($"LOD tier {i} has invalid MaxScreenError ({tier.MaxScreenError}).");
        }

        foreach (var clip in AnimationClips)
        {
            if (clip.ClipId == Guid.Empty)
                errors.Add($"Animation clip '{clip.ClipName}' has empty ClipId.");

            if (clip.DurationSeconds <= 0)
                errors.Add($"Animation clip '{clip.ClipName}' has non-positive duration.");
        }

        return errors;
    }

    /// <summary>
    /// Computes the total memory footprint of this asset in bytes.
    /// </summary>
    /// <returns>Total memory usage in bytes.</returns>
    public long ComputeMemoryFootprint()
    {
        long total = 0;

        total += _compressedWeights.Length;
        total += (UncompressedWeights?.Length ?? 0) * sizeof(float);

        total += BoundingBox.SizeInBytes;
        total += BoundingSphere.SizeInBytes;
        total += MicroTextureCoordinates.SizeInBytes;

        foreach (var tier in LODTiers)
            total += tier.ComputeMemoryFootprint();

        foreach (var clip in AnimationClips)
            total += clip.CompressedKeyframes.Length + 64; // header overhead

        total += 1024; // estimated metadata overhead

        return total;
    }

    /// <summary>
    /// Serializes the entire neural asset to a binary format.
    /// Format: [header(64)] [metadata(json)] [compressed_weights] [lod_tiers] [texture_coords] [animation_clips]
    /// </summary>
    /// <returns>Serialized byte array.</returns>
    public byte[] Serialize()
    {
        using var ms = new MemoryStream();

        // Header
        Span<byte> header = stackalloc byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(header[0..4], 0x4E4E4552); // "NNER"
        BinaryPrimitives.WriteInt32LittleEndian(header[4..8], 2); // version
        GuidCompat.WriteBytes(header.Slice(8, 16), AssetId);
        GuidCompat.WriteBytes(header.Slice(24, 16), TargetMeshId);
        BinaryPrimitives.WriteInt32LittleEndian(header[40..44], StreamingPriority);
        BinaryPrimitives.WriteSingleLittleEndian(header[44..48], MaxLoadDistance);
        BinaryPrimitives.WriteSingleLittleEndian(header[48..52], MinUnloadDistance);
        BinaryPrimitives.WriteSingleLittleEndian(header[52..56], LODBias);
        BinaryPrimitives.WriteInt32LittleEndian(header[56..60], _compressedWeights.Length);
        BinaryPrimitives.WriteInt32LittleEndian(header[60..64], LODTiers.Count);
        ms.Write(header);

        // Bounding geometry
        Span<byte> bboxData = stackalloc byte[BoundingBox.SizeInBytes + BoundingSphere.SizeInBytes + MicroTextureCoordinates.SizeInBytes];
        BoundingBox.Serialize(bboxData);
        BoundingSphere.Serialize(bboxData[BoundingBox.SizeInBytes..]);
        TextureCoordinates.Serialize(bboxData[(BoundingBox.SizeInBytes + BoundingSphere.SizeInBytes)..]);
        ms.Write(bboxData);

        // Metadata JSON
        byte[] metaBytes = Encoding.UTF8.GetBytes(Metadata.ToJson());
        BinaryPrimitives.WriteInt32LittleEndian(ms.GetSpan(4), metaBytes.Length);
        ms.Advance(4);
        ms.Write(metaBytes);

        // Compressed weights
        ms.Write(_compressedWeights);

        // LOD tiers
        foreach (var tier in LODTiers)
        {
            Span<byte> tierHeader = stackalloc byte[16];
            tierHeader[0] = (byte)tier.Level;
            BinaryPrimitives.WriteInt32LittleEndian(tierHeader[1..5], tier.CompressedWeights.Length);
            BinaryPrimitives.WriteInt32LittleEndian(tierHeader[5..9], tier.UncompressedWeights.Length);
            BinaryPrimitives.WriteSingleLittleEndian(tierHeader[9..13], tier.MaxScreenError);
            BinaryPrimitives.WriteInt32LittleEndian(tierHeader[13..17], tier.ApproximateTriangleCount);
            ms.Write(tierHeader);

            ms.Write(tier.CompressedWeights);
            if (tier.UncompressedWeights.Length > 0)
            {
                byte[] tierFloats = MemoryMarshal.AsBytes(tier.UncompressedWeights.AsSpan()).ToArray();
                ms.Write(tierFloats);
            }
        }

        // Animation clips
        BinaryPrimitives.WriteInt32LittleEndian(ms.GetSpan(4), AnimationClips.Count);
        ms.Advance(4);
        foreach (var clip in AnimationClips)
        {
            byte[] clipData = clip.Serialize();
            BinaryPrimitives.WriteInt32LittleEndian(ms.GetSpan(4), clipData.Length);
            ms.Advance(4);
            ms.Write(clipData);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes a neural asset from a binary stream.
    /// </summary>
    /// <param name="stream">The input stream.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The deserialized NeuralAsset.</returns>
    private const int MaxCompressedWeightBytes = 64 * 1024 * 1024;
    private const int MaxMetadataBytes = 1_000_000;
    private const int MaxLodTierCount = 32;
    private const int MaxLodTierWeightBytes = 32 * 1024 * 1024;
    private const int MaxUncompressedFloatCount = 8 * 1024 * 1024;
    private const int MaxAnimationClipCount = 256;
    private const int MaxAnimationClipBytes = 16 * 1024 * 1024;

    public static async Task<NeuralAsset> DeserializeAsync(Stream stream, CancellationToken cancellationToken = default)
    {
        var asset = new NeuralAsset();

        // Header
        byte[] header = new byte[64];
        await stream.ReadExactlyAsync(header, cancellationToken).ConfigureAwait(false);

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(0..4));
        if (magic != 0x4E4E4552)
            throw new InvalidDataException($"Invalid magic number: 0x{magic:X8}, expected 0x4E4E4552 (\"NNER\").");

        int version = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(4..8));
        if (version > 2)
            throw new InvalidDataException($"Unsupported version: {version}");

        asset.AssetId = new Guid(header.AsSpan(8, 16));
        asset.TargetMeshId = new Guid(header.AsSpan(24, 16));
        asset.StreamingPriority = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(40..44));
        asset.MaxLoadDistance = BinaryPrimitives.ReadSingleLittleEndian(header.AsSpan(44..48));
        asset.MinUnloadDistance = BinaryPrimitives.ReadSingleLittleEndian(header.AsSpan(48..52));
        asset.LODBias = BinaryPrimitives.ReadSingleLittleEndian(header.AsSpan(52..56));
        int compressedWeightLength = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(56..60));
        int lodTierCount = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(60..64));

        if (compressedWeightLength < 0 || compressedWeightLength > MaxCompressedWeightBytes)
            throw new InvalidDataException("Compressed weight length out of bounds.");
        if (lodTierCount < 0 || lodTierCount > MaxLodTierCount)
            throw new InvalidDataException("LOD tier count out of bounds.");

        // Bounding geometry
        byte[] bboxData = new byte[BoundingBox.SizeInBytes + BoundingSphere.SizeInBytes + MicroTextureCoordinates.SizeInBytes];
        await stream.ReadExactlyAsync(bboxData, cancellationToken).ConfigureAwait(false);
        asset.BoundingBox = BoundingBox.Deserialize(bboxData);
        asset.BoundingSphere = BoundingSphere.Deserialize(bboxData.AsSpan(BoundingBox.SizeInBytes));
        asset.TextureCoordinates = MicroTextureCoordinates.Deserialize(bboxData.AsSpan(BoundingBox.SizeInBytes + BoundingSphere.SizeInBytes));

        // Metadata JSON
        byte[] metaLenBuf = new byte[4];
        await stream.ReadExactlyAsync(metaLenBuf, cancellationToken).ConfigureAwait(false);
        int metaLen = BinaryPrimitives.ReadInt32LittleEndian(metaLenBuf);
        if (metaLen < 0 || metaLen > MaxMetadataBytes)
            throw new InvalidDataException("Metadata length out of bounds.");
        byte[] metaBytes = new byte[metaLen];
        await stream.ReadExactlyAsync(metaBytes, cancellationToken).ConfigureAwait(false);
        asset.Metadata = NeuralAssetMetadata.FromJson(Encoding.UTF8.GetString(metaBytes));

        // Compressed weights
        asset._compressedWeights = new byte[compressedWeightLength];
        await stream.ReadExactlyAsync(asset._compressedWeights, cancellationToken).ConfigureAwait(false);

        // LOD tiers
        for (int i = 0; i < lodTierCount; i++)
        {
            byte[] tierHeader = new byte[17];
            await stream.ReadExactlyAsync(tierHeader, cancellationToken).ConfigureAwait(false);

            int tierCompressedLen = BinaryPrimitives.ReadInt32LittleEndian(tierHeader.AsSpan(1..5));
            int uncompLen = BinaryPrimitives.ReadInt32LittleEndian(tierHeader.AsSpan(5..9));
            if (tierCompressedLen < 0 || tierCompressedLen > MaxLodTierWeightBytes)
                throw new InvalidDataException("LOD tier compressed weight length out of bounds.");
            if (uncompLen < 0 || uncompLen > MaxUncompressedFloatCount)
                throw new InvalidDataException("LOD tier uncompressed float count out of bounds.");

            var tier = new LODTier
            {
                Level = (LODLevel)tierHeader[0],
                CompressedWeights = new byte[tierCompressedLen],
                MaxScreenError = BinaryPrimitives.ReadSingleLittleEndian(tierHeader.AsSpan(9..13)),
                ApproximateTriangleCount = BinaryPrimitives.ReadInt32LittleEndian(tierHeader.AsSpan(13..17))
            };

            await stream.ReadExactlyAsync(tier.CompressedWeights, cancellationToken).ConfigureAwait(false);

            if (uncompLen > 0)
            {
                byte[] tierFloats = new byte[uncompLen * sizeof(float)];
                await stream.ReadExactlyAsync(tierFloats, cancellationToken).ConfigureAwait(false);
                tier.UncompressedWeights = MemoryMarshal.Cast<byte, float>(tierFloats).ToArray();
            }

            asset.LODTiers.Add(tier);
        }

        // Animation clips
        byte[] clipCountBuf = new byte[4];
        await stream.ReadExactlyAsync(clipCountBuf, cancellationToken).ConfigureAwait(false);
        int clipCount = BinaryPrimitives.ReadInt32LittleEndian(clipCountBuf);
        if (clipCount < 0 || clipCount > MaxAnimationClipCount)
            throw new InvalidDataException("Animation clip count out of bounds.");

        for (int i = 0; i < clipCount; i++)
        {
            byte[] clipLenBuf = new byte[4];
            await stream.ReadExactlyAsync(clipLenBuf, cancellationToken).ConfigureAwait(false);
            int clipLen = BinaryPrimitives.ReadInt32LittleEndian(clipLenBuf);
            if (clipLen < 0 || clipLen > MaxAnimationClipBytes)
                throw new InvalidDataException("Animation clip length out of bounds.");

            byte[] clipData = new byte[clipLen];
            await stream.ReadExactlyAsync(clipData, cancellationToken).ConfigureAwait(false);
            asset.AnimationClips.Add(AnimationClipReference.Deserialize(clipData));
        }

        return asset;
    }

    /// <summary>
    /// Serializes and saves the neural asset to a file.
    /// </summary>
    /// <param name="path">File path to save to.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task SaveAsync(string path, CancellationToken cancellationToken = default)
    {
        byte[] data = Serialize();
        await File.WriteAllBytesAsync(path, data, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Loads and deserializes a neural asset from a file.
    /// </summary>
    /// <param name="path">File path to load from.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The deserialized NeuralAsset.</returns>
    public static async Task<NeuralAsset> LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        using var fs = File.OpenRead(path);
        return await DeserializeAsync(fs, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Creates a deep copy of this neural asset.
    /// </summary>
    /// <returns>A new NeuralAsset instance with copied data.</returns>
    public NeuralAsset Clone()
    {
        var clone = new NeuralAsset
        {
            AssetId = AssetId,
            TargetMeshId = TargetMeshId,
            BoundingBox = BoundingBox,
            BoundingSphere = BoundingSphere,
            TextureCoordinates = TextureCoordinates,
            StreamingPriority = StreamingPriority,
            MaxLoadDistance = MaxLoadDistance,
            MinUnloadDistance = MinUnloadDistance,
            LODBias = LODBias,
            Metadata = new NeuralAssetMetadata
            {
                Name = Metadata.Name,
                Version = Metadata.Version,
                CreationDate = Metadata.CreationDate,
                LastModifiedDate = Metadata.LastModifiedDate,
                Author = Metadata.Author,
                Description = Metadata.Description,
                ContentHash = Metadata.ContentHash,
                SourceEngine = Metadata.SourceEngine,
                License = Metadata.License,
                Tags = (string[])Metadata.Tags.Clone()
            },
            _compressedWeights = (byte[])_compressedWeights.Clone()
        };

        if (UncompressedWeights != null)
            clone.UncompressedWeights = (float[])UncompressedWeights.Clone();

        foreach (var tier in LODTiers)
        {
            clone.LODTiers.Add(new LODTier
            {
                Level = tier.Level,
                CompressedWeights = (byte[])tier.CompressedWeights.Clone(),
                UncompressedWeights = (float[])tier.UncompressedWeights.Clone(),
                MaxScreenError = tier.MaxScreenError,
                DistanceRange = tier.DistanceRange,
                ApproximateTriangleCount = tier.ApproximateTriangleCount
            });
        }

        foreach (var clip in AnimationClips)
        {
            clone.AnimationClips.Add(new AnimationClipReference
            {
                ClipId = clip.ClipId,
                ClipName = clip.ClipName,
                DurationSeconds = clip.DurationSeconds,
                FrameRate = clip.FrameRate,
                KeyframeCount = clip.KeyframeCount,
                CompressedKeyframes = (byte[])clip.CompressedKeyframes.Clone(),
                ChannelMask = clip.ChannelMask
            });
        }

        return clone;
    }

    /// <summary>
    /// Generates a summary string for debugging.
    /// </summary>
    public override string ToString()
    {
        return $"NeuralAsset[{AssetId:B}]: {Metadata.Name} v{Metadata.Version}, " +
               $"LODs={LODTiers.Count}, Clips={AnimationClips.Count}, " +
               $"Compressed={_compressedWeights.Length}B, " +
               $"MemoryFootprint={ComputeMemoryFootprint()}B";
    }
}
