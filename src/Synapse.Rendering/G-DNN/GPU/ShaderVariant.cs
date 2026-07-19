using System;
// ============================================================
// FILE: ShaderVariant.cs
// PATH: GPU/ShaderVariant.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.GPU;

/// <summary>
/// Represents the quality level for a shader variant.
/// </summary>
public enum ShaderVariantQuality
{
    /// <summary>Low quality - minimal features, reduced precision.</summary>
    Low = 0,

    /// <summary>Medium quality - balanced features.</summary>
    Medium = 1,

    /// <summary>High quality - full features.</summary>
    High = 2,

    /// <summary>Ultra quality - maximum features and precision.</summary>
    Ultra = 3
}

/// <summary>
/// Feature flags for shader variants. Each flag enables a specific feature path.
/// </summary>
[Flags]
public enum ShaderVariantFeature : uint
{
    /// <summary>No features enabled.</summary>
    None = 0,

    /// <summary>Shadow ray evaluation.</summary>
    Shadows = 1u << 0,

    /// <summary>Reflection ray tracing.</summary>
    Reflections = 1u << 1,

    /// <summary>Refraction ray tracing.</summary>
    Refractions = 1u << 2,

    /// <summary>Ambient occlusion.</summary>
    AmbientOcclusion = 1u << 3,

    /// <summary>Hierarchical sphere tracing acceleration.</summary>
    HierarchicalTracing = 1u << 4,

    /// <summary>Binary search refinement for sphere tracing.</summary>
    BinarySearch = 1u << 5,

    /// <summary>Normal mapping perturbation.</summary>
    NormalMapping = 1u << 6,

    /// <summary>Parallax occlusion mapping.</summary>
    ParallaxOcclusion = 1u << 7,

    /// <summary>Temporal reprojection for coherence.</summary>
    TemporalReprojection = 1u << 8,

    /// <summary>Screen-space ambient occlusion.</summary>
    SSAO = 1u << 9,

    /// <summary>Subsurface scattering approximation.</summary>
    SubsurfaceScattering = 1u << 10,

    /// <summary>Volumetric fog integration.</summary>
    VolumetricFog = 1u << 11,

    /// <summary>Curvature-based adaptive step scaling.</summary>
    AdaptiveStepping = 1u << 12,

    /// <summary>Two-sided surface rendering.</summary>
    TwoSided = 1u << 13,

    /// <summary>Emissive surface support.</summary>
    Emissive = 1u << 14,

    /// <summary>Alpha testing / cutout.</summary>
    AlphaTest = 1u << 15,

    /// <summary>Wireframe overlay visualization.</summary>
    Wireframe = 1u << 16,

    /// <summary>Debug visualization mode.</summary>
    DebugVisualization = 1u << 17,

    /// <summary>Half-precision floating point paths.</summary>
    HalfPrecision = 1u << 18,

    /// <summary>Dynamic branching in loops.</summary>
    DynamicBranching = 1u << 19,

    /// <summary>All standard features.</summary>
    AllStandard = Shadows | Reflections | AmbientOcclusion | HierarchicalTracing | BinarySearch | AdaptiveStepping,

    /// <summary>All features enabled.</summary>
    All = 0x7FFFFFFFu
}

/// <summary>
/// Represents a unique shader variant identified by feature flags and quality level.
/// </summary>
public readonly struct ShaderVariantKey : IEquatable<ShaderVariantKey>, IComparable<ShaderVariantKey>
{
    /// <summary>Feature flags for this variant.</summary>
    public readonly ShaderVariantFeature Features;

    /// <summary>Quality level for this variant.</summary>
    public readonly ShaderVariantQuality Quality;

    /// <summary>Additional variant parameter hash (for custom permutations).</summary>
    public readonly uint ParameterHash;

    /// <summary>Creates a new variant key.</summary>
    public ShaderVariantKey(ShaderVariantFeature features, ShaderVariantQuality quality, uint parameterHash = 0)
    {
        Features = features;
        Quality = quality;
        ParameterHash = parameterHash;
    }

    /// <summary>Creates a variant key from a 64-bit encoded value.</summary>
    public static ShaderVariantKey FromEncoded(ulong encoded)
    {
        return new ShaderVariantKey(
            (ShaderVariantFeature)(uint)(encoded & 0xFFFFFFFF),
            (ShaderVariantQuality)((encoded >> 32) & 0xFF),
            (uint)((encoded >> 40) & 0xFFFFFFFF)
        );
    }

    /// <summary>Encodes the variant key to a 64-bit value.</summary>
    public ulong Encode()
    {
        ulong key = (ulong)(uint)Features;
        key |= (ulong)(byte)Quality << 32;
        key |= (ulong)ParameterHash << 40;
        return key;
    }

    /// <summary>Gets a 64-bit hash for use in dictionaries.</summary>
    public ulong Hash64 => Encode();

    /// <summary>Gets a 32-bit hash for compact storage.</summary>
    public uint Hash32 => (uint)HashCode.Combine(Features, Quality, ParameterHash);

    /// <summary>Checks if a specific feature is enabled.</summary>
    public bool HasFeature(ShaderVariantFeature feature) => Features.HasFlag(feature);

    /// <summary>Creates a new key with a feature added.</summary>
    public ShaderVariantKey WithFeature(ShaderVariantFeature feature)
        => new(Features | feature, Quality, ParameterHash);

    /// <summary>Creates a new key with a feature removed.</summary>
    public ShaderVariantKey WithoutFeature(ShaderVariantFeature feature)
        => new(Features & ~feature, Quality, ParameterHash);

    /// <summary>Creates a new key with a different quality level.</summary>
    public ShaderVariantKey WithQuality(ShaderVariantQuality quality)
        => new(Features, quality, ParameterHash);

    /// <summary>Creates a new key with a different parameter hash.</summary>
    public ShaderVariantKey WithParameterHash(uint hash)
        => new(Features, Quality, hash);

    /// <summary>Gets the display name for this variant.</summary>
    public string DisplayName
    {
        get
        {
            var sb = new StringBuilder();
            sb.Append(Quality);
            if (Features != ShaderVariantFeature.None)
            {
                sb.Append('_');
                sb.Append(Features.ToString().Replace(", ", "+"));
            }
            return sb.ToString();
        }
    }

    /// <summary>Gets all individual features enabled in this key.</summary>
    public IEnumerable<ShaderVariantFeature> EnabledFeatures
    {
        get
        {
            uint val = (uint)Features;
            for (uint bit = 1; bit <= 0x40000000u; bit <<= 1)
            {
                if ((val & bit) != 0)
                    yield return (ShaderVariantFeature)bit;
            }
        }
    }

    /// <summary>Gets the number of enabled features.</summary>
    public int FeatureCount => BitOperations.PopCount((uint)Features);

    /// <summary>Creates a minimal variant (no features, low quality).</summary>
    public static ShaderVariantKey Minimal => new(ShaderVariantFeature.None, ShaderVariantQuality.Low);

    /// <summary>Creates a shadow-only variant.</summary>
    public static ShaderVariantKey ShadowOnly => new(ShaderVariantFeature.Shadows, ShaderVariantQuality.Medium);

    /// <summary>Creates a full-featured variant at high quality.</summary>
    public static ShaderVariantKey FullFeatured => new(ShaderVariantFeature.AllStandard, ShaderVariantQuality.High);

    /// <summary>Creates an ultra quality variant with all features.</summary>
    public static ShaderVariantKey Ultra => new(ShaderVariantFeature.All, ShaderVariantQuality.Ultra);

    public bool Equals(ShaderVariantKey other) =>
        Features == other.Features && Quality == other.Quality && ParameterHash == other.ParameterHash;

    public override bool Equals(object? obj) => obj is ShaderVariantKey other && Equals(other);
    public override int GetHashCode() => Hash32.GetHashCode();
    public override string ToString() => DisplayName;

    public static bool operator ==(ShaderVariantKey left, ShaderVariantKey right) => left.Equals(right);
    public static bool operator !=(ShaderVariantKey left, ShaderVariantKey right) => !left.Equals(right);

    public int CompareTo(ShaderVariantKey other)
    {
        int cmp = Features.CompareTo(other.Features);
        if (cmp != 0)
            return cmp;
        cmp = Quality.CompareTo(other.Quality);
        if (cmp != 0)
            return cmp;
        return ParameterHash.CompareTo(other.ParameterHash);
    }

    public static bool operator <(ShaderVariantKey left, ShaderVariantKey right) => left.CompareTo(right) < 0;
    public static bool operator >(ShaderVariantKey left, ShaderVariantKey right) => left.CompareTo(right) > 0;
    public static bool operator <=(ShaderVariantKey left, ShaderVariantKey right) => left.CompareTo(right) <= 0;
    public static bool operator >=(ShaderVariantKey left, ShaderVariantKey right) => left.CompareTo(right) >= 0;
}

/// <summary>
/// Represents a compiled shader variant with its metadata.
/// </summary>
public sealed class ShaderVariant
{
    /// <summary>The unique key for this variant.</summary>
    public required ShaderVariantKey Key { get; init; }

    /// <summary>Compiled shader bytecode.</summary>
    public required byte[] Bytecode { get; init; }

    /// <summary>Vertex shader bytecode (for pixel shaders, may be null).</summary>
    public byte[]? VertexBytecode { get; init; }

    /// <summary>Shader entry point name.</summary>
    public required string EntryPoint { get; init; }

    /// <summary>Shader source hash for cache validation.</summary>
    public string? SourceHash { get; init; }

    /// <summary>Timestamp when this variant was compiled.</summary>
    public DateTime CompiledAt { get; init; } = DateTime.UtcNow;

    /// <summary>Estimated GPU instruction count.</summary>
    public int EstimatedInstructions { get; init; }

    /// <summary>Estimated constant buffer size in bytes.</summary>
    public int EstimatedConstantBufferSize { get; init; }

    /// <summary>Whether this variant is currently active/loaded.</summary>
    public bool IsActive { get; set; }

    /// <summary>Whether this variant failed compilation.</summary>
    public bool CompilationFailed { get; init; }

    /// <summary>Compilation error message (if failed).</summary>
    public string? CompilationError { get; init; }

    /// <summary>Gets the feature description string.</summary>
    public string FeatureDescription
    {
        get
        {
            var sb = new StringBuilder();
            sb.Append($"Quality: {Key.Quality}");

            if (Key.Features != ShaderVariantFeature.None)
            {
                sb.Append(" | Features: ");
                sb.Append(string.Join(", ", Key.EnabledFeatures.Select(f => f.ToString())));
            }

            return sb.ToString();
        }
    }
}

/// <summary>
/// Manages shader variant generation, permutation creation, and variant lookup.
/// Handles feature flag combinations, quality level transitions, and variant optimization.
/// </summary>
public sealed class ShaderVariantManager
{
    private readonly Dictionary<ShaderVariantKey, ShaderVariant> _variants;
    private readonly Dictionary<ulong, ShaderVariantKey> _keyLookup;
    private readonly List<ShaderVariantKey> _activeVariants;
    private readonly object _lock;

    /// <summary>Creates a new variant manager.</summary>
    public ShaderVariantManager()
    {
        _variants = new Dictionary<ShaderVariantKey, ShaderVariant>();
        _keyLookup = new Dictionary<ulong, ShaderVariantKey>();
        _activeVariants = new List<ShaderVariantKey>();
        _lock = new object();
    }

    /// <summary>Gets the total number of registered variants.</summary>
    public int VariantCount
    {
        get { lock (_lock) return _variants.Count; }
    }

    /// <summary>Gets the number of active variants.</summary>
    public int ActiveVariantCount
    {
        get { lock (_lock) return _activeVariants.Count; }
    }

    /// <summary>Gets all registered variant keys.</summary>
    public IEnumerable<ShaderVariantKey> RegisteredKeys
    {
        get { lock (_lock) return _variants.Keys.ToList(); }
    }

    /// <summary>Gets all active variant keys.</summary>
    public IReadOnlyList<ShaderVariantKey> ActiveKeys
    {
        get { lock (_lock) return _activeVariants.AsReadOnly(); }
    }

    /// <summary>
    /// Registers a compiled shader variant.
    /// </summary>
    /// <param name="variant">The compiled variant to register.</param>
    public void RegisterVariant(ShaderVariant variant)
    {
        lock (_lock)
        {
            _variants[variant.Key] = variant;
            _keyLookup[variant.Key.Hash64] = variant.Key;
        }
    }

    /// <summary>
    /// Unregisters a variant by key.
    /// </summary>
    public bool UnregisterVariant(ShaderVariantKey key)
    {
        lock (_lock)
        {
            _activeVariants.Remove(key);
            return _variants.Remove(key) && _keyLookup.Remove(key.Hash64);
        }
    }

    /// <summary>
    /// Gets a registered variant by key.
    /// </summary>
    public ShaderVariant? GetVariant(ShaderVariantKey key)
    {
        lock (_lock)
        {
            return _variants.GetValueOrDefault(key);
        }
    }

    /// <summary>
    /// Gets a registered variant by its 64-bit hash.
    /// </summary>
    public ShaderVariant? GetVariantByHash(ulong hash64)
    {
        lock (_lock)
        {
            if (_keyLookup.TryGetValue(hash64, out var key))
                return _variants.GetValueOrDefault(key);
            return null;
        }
    }

    /// <summary>
    /// Checks if a variant is registered.
    /// </summary>
    public bool HasVariant(ShaderVariantKey key)
    {
        lock (_lock)
            return _variants.ContainsKey(key);
    }

    /// <summary>
    /// Sets a variant as active for rendering.
    /// </summary>
    public bool SetActive(ShaderVariantKey key)
    {
        lock (_lock)
        {
            if (!_variants.ContainsKey(key))
                return false;
            if (!_activeVariants.Contains(key))
                _activeVariants.Add(key);

            if (_variants.TryGetValue(key, out var variant))
                variant.IsActive = true;

            return true;
        }
    }

    /// <summary>
    /// Deactivates a variant.
    /// </summary>
    public bool SetInactive(ShaderVariantKey key)
    {
        lock (_lock)
        {
            _activeVariants.Remove(key);
            if (_variants.TryGetValue(key, out var variant))
                variant.IsActive = false;
            return true;
        }
    }

    /// <summary>
    /// Gets the best matching variant for the requested features and quality.
    /// Falls back to lower quality or fewer features if an exact match is not available.
    /// </summary>
    /// <param name="desired">Desired variant key.</param>
    /// <returns>Best matching variant, or null if none available.</returns>
    public ShaderVariant? GetBestMatch(ShaderVariantKey desired)
    {
        lock (_lock)
        {
            // Try exact match
            if (_variants.TryGetValue(desired, out var exact))
                return exact;

            // Try same features, lower quality
            for (int q = (int)desired.Quality; q >= 0; q--)
            {
                var key = new ShaderVariantKey(desired.Features, (ShaderVariantQuality)q, desired.ParameterHash);
                if (_variants.TryGetValue(key, out var variant))
                    return variant;
            }

            // Try fewer features at same or lower quality
            var candidates = _variants
                .Where(kv => kv.Key.Features <= desired.Features && kv.Key.Quality <= desired.Quality)
                .OrderByDescending(kv => kv.Key.FeatureCount)
                .ThenByDescending(kv => kv.Key.Quality)
                .FirstOrDefault();

            return candidates.Value;
        }
    }

    /// <summary>
    /// Generates all permutations of feature flags for a given quality level.
    /// </summary>
    /// <param name="quality">Quality level.</param>
    /// <param name="maxFeatures">Maximum number of features to combine.</param>
    /// <returns>Generated variant keys.</returns>
    public static IEnumerable<ShaderVariantKey> GeneratePermutations(ShaderVariantQuality quality, int maxFeatures = int.MaxValue)
    {
        var allFeatures = Enum.GetValues<ShaderVariantFeature>()
            .Where(f => f != ShaderVariantFeature.None && f != ShaderVariantFeature.All && f != ShaderVariantFeature.AllStandard)
            .ToList();

        // Generate combinations up to maxFeatures
        for (int count = 0; count <= Math.Min(maxFeatures, allFeatures.Count); count++)
        {
            foreach (var combo in Combinations(allFeatures, count))
            {
                var features = ShaderVariantFeature.None;
                foreach (var f in combo)
                    features |= f;

                yield return new ShaderVariantKey(features, quality);
            }
        }
    }

    /// <summary>
    /// Generates common (most useful) permutations without combinatorial explosion.
    /// </summary>
    /// <param name="quality">Quality level.</param>
    /// <returns>Common variant key permutations.</returns>
    public static IEnumerable<ShaderVariantKey> GenerateCommonPermutations(ShaderVariantQuality quality)
    {
        // Minimal set
        yield return new ShaderVariantKey(ShaderVariantFeature.None, quality);

        // Individual features
        yield return new ShaderVariantKey(ShaderVariantFeature.Shadows, quality);
        yield return new ShaderVariantKey(ShaderVariantFeature.AmbientOcclusion, quality);
        yield return new ShaderVariantKey(ShaderVariantFeature.Reflections, quality);

        // Common combinations
        yield return new ShaderVariantKey(
            ShaderVariantFeature.Shadows | ShaderVariantFeature.AmbientOcclusion, quality);
        yield return new ShaderVariantKey(
            ShaderVariantFeature.Shadows | ShaderVariantFeature.AmbientOcclusion | ShaderVariantFeature.BinarySearch, quality);
        yield return new ShaderVariantKey(
            ShaderVariantFeature.Shadows | ShaderVariantFeature.AmbientOcclusion | ShaderVariantFeature.BinarySearch | ShaderVariantFeature.HierarchicalTracing, quality);
        yield return new ShaderVariantKey(
            ShaderVariantFeature.Shadows | ShaderVariantFeature.AmbientOcclusion | ShaderVariantFeature.Reflections, quality);

        // Full featured
        yield return new ShaderVariantKey(ShaderVariantFeature.AllStandard, quality);
    }

    /// <summary>
    /// Generates permutations across all quality levels for a base feature set.
    /// </summary>
    /// <param name="baseFeatures">Base feature flags.</param>
    /// <returns>Variant keys across all quality levels.</returns>
    public static IEnumerable<ShaderVariantKey> GenerateQualityPermutations(ShaderVariantFeature baseFeatures)
    {
        foreach (ShaderVariantQuality quality in Enum.GetValues<ShaderVariantQuality>())
        {
            yield return new ShaderVariantKey(baseFeatures, quality);
        }
    }

    /// <summary>
    /// Finds the nearest variant that is a subset of the requested features.
    /// Useful for fallback when an exact feature match is not available.
    /// </summary>
    /// <param name="desired">Desired feature flags.</param>
    /// <param name="quality">Quality level.</param>
    /// <returns>Best subset variant key, or null.</returns>
    public ShaderVariantKey? FindSubsetVariant(ShaderVariantFeature desired, ShaderVariantQuality quality)
    {
        lock (_lock)
        {
            // Find variants where features are a subset of desired
            var candidates = _variants.Keys
                .Where(k => (k.Features & desired) == k.Features && k.Quality <= quality)
                .OrderByDescending(k => k.FeatureCount)
                .ThenByDescending(k => k.Quality)
                .ToList();

            return candidates.FirstOrDefault();
        }
    }

    /// <summary>
    /// Computes the memory footprint of all registered variants.
    /// </summary>
    public long ComputeMemoryFootprint()
    {
        lock (_lock)
        {
            return _variants.Values.Sum(v =>
                (long)(v.Bytecode?.Length ?? 0) + (long)(v.VertexBytecode?.Length ?? 0));
        }
    }

    /// <summary>
    /// Gets statistics about registered variants.
    /// </summary>
    public ShaderVariantStats GetStats()
    {
        lock (_lock)
        {
            var stats = new ShaderVariantStats
            {
                TotalVariants = _variants.Count,
                ActiveVariants = _activeVariants.Count,
                FailedVariants = _variants.Values.Count(v => v.CompilationFailed),
                TotalBytecodeBytes = _variants.Values.Sum(v => (long)(v.Bytecode?.Length ?? 0)),
                VariantsByQuality = _variants.Keys
                    .GroupBy(k => k.Quality)
                    .ToDictionary(g => g.Key, g => g.Count()),
                VariantsByFeature = new Dictionary<ShaderVariantFeature, int>()
            };

            // Count variants by feature
            foreach (ShaderVariantFeature feature in Enum.GetValues<ShaderVariantFeature>())
            {
                if (feature == ShaderVariantFeature.None || feature == ShaderVariantFeature.All || feature == ShaderVariantFeature.AllStandard)
                    continue;

                int count = _variants.Keys.Count(k => k.HasFeature(feature));
                if (count > 0)
                    stats.VariantsByFeature[feature] = count;
            }

            return stats;
        }
    }

    /// <summary>
    /// Removes variants that haven't been accessed recently (LRU eviction).
    /// </summary>
    /// <param name="maxVariants">Maximum number of variants to keep.</param>
    /// <param name="keepActive">Always keep active variants.</param>
    /// <returns>Number of variants evicted.</returns>
    public int EvictLeastRecentlyUsed(int maxVariants, bool keepActive = true)
    {
        lock (_lock)
        {
            if (_variants.Count <= maxVariants)
                return 0;

            var toRemove = _variants
                .Where(kv => !keepActive || !_activeVariants.Contains(kv.Key))
                .OrderBy(kv => kv.Value.CompiledAt)
                .Take(_variants.Count - maxVariants)
                .Select(kv => kv.Key)
                .ToList();

            foreach (var key in toRemove)
            {
                _variants.Remove(key);
                _keyLookup.Remove(key.Hash64);
            }

            return toRemove.Count;
        }
    }

    /// <summary>
    /// Clears all registered variants.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _variants.Clear();
            _keyLookup.Clear();
            _activeVariants.Clear();
        }
    }

    /// <summary>
    /// Generates a display-friendly list of all variant configurations.
    /// </summary>
    public string GenerateVariantReport()
    {
        lock (_lock)
        {
            var sb = new StringBuilder();
            sb.AppendLine("=== Shader Variant Report ===");
            sb.AppendLine($"Total Variants: {_variants.Count}");
            sb.AppendLine($"Active Variants: {_activeVariants.Count}");
            sb.AppendLine($"Total Bytecode: {ComputeMemoryFootprint() / 1024.0:F1} KB");
            sb.AppendLine();

            // Group by quality
            sb.AppendLine("By Quality Level:");
            foreach (var group in _variants.Keys.GroupBy(k => k.Quality).OrderBy(g => g.Key))
            {
                sb.AppendLine($"  {group.Key}: {group.Count()} variants");
            }
            sb.AppendLine();

            // Group by feature
            sb.AppendLine("By Feature:");
            foreach (ShaderVariantFeature feature in Enum.GetValues<ShaderVariantFeature>())
            {
                if (feature == ShaderVariantFeature.None || (uint)feature > 0x10000)
                    continue;
                int count = _variants.Keys.Count(k => k.HasFeature(feature));
                if (count > 0)
                    sb.AppendLine($"  {feature}: {count} variants");
            }
            sb.AppendLine();

            // List all variants
            sb.AppendLine("All Variants:");
            foreach (var key in _variants.Keys.OrderBy(k => k))
            {
                var variant = _variants[key];
                string status = variant.IsActive ? "[ACTIVE]" : variant.CompilationFailed ? "[FAILED]" : "[READY]";
                sb.AppendLine($"  {key.DisplayName} {status} ({variant.Bytecode?.Length ?? 0} bytes)");
            }

            return sb.ToString();
        }
    }

    /// <summary>
    /// Generates all binary combinations of a set of features (for exhaustive permutation generation).
    /// </summary>
    private static IEnumerable<T[]> Combinations<T>(List<T> items, int count)
    {
        if (count == 0)
        {
            yield return Array.Empty<T>();
            yield break;
        }

        for (int i = 0; i <= items.Count - count; i++)
        {
            var rest = items.Skip(i + 1).Take(items.Count - i - 1).ToList();
            foreach (var combo in Combinations(rest, count - 1))
            {
                var result = new T[count];
                result[0] = items[i];
                Array.Copy(combo, 0, result, 1, combo.Length);
                yield return result;
            }
        }
    }
}

/// <summary>
/// Statistics about registered shader variants.
/// </summary>
public sealed class ShaderVariantStats
{
    /// <summary>Total number of registered variants.</summary>
    public int TotalVariants { get; set; }

    /// <summary>Number of active variants.</summary>
    public int ActiveVariants { get; set; }

    /// <summary>Number of variants that failed compilation.</summary>
    public int FailedVariants { get; set; }

    /// <summary>Total bytecode size in bytes.</summary>
    public long TotalBytecodeBytes { get; set; }

    /// <summary>Variant count grouped by quality level.</summary>
    public Dictionary<ShaderVariantQuality, int> VariantsByQuality { get; set; } = new();

    /// <summary>Variant count grouped by feature flag.</summary>
    public Dictionary<ShaderVariantFeature, int> VariantsByFeature { get; set; } = new();

    /// <summary>Returns a summary string.</summary>
    public override string ToString()
    {
        return $"Variants: {TotalVariants} (Active: {ActiveVariants}, Failed: {FailedVariants}) " +
               $"Bytecode: {TotalBytecodeBytes / 1024.0:F1} KB";
    }
}

/// <summary>
/// Provides extension methods for shader variant feature manipulation.
/// </summary>
public static class ShaderVariantExtensions
{
    /// <summary>
    /// Converts ShaderFeatures to ShaderVariantFeature.
    /// </summary>
    public static ShaderVariantFeature ToVariantFeature(this ShaderFeatures features)
    {
        return (ShaderVariantFeature)(uint)features;
    }

    /// <summary>
    /// Converts ShaderVariantFeature to ShaderFeatures.
    /// </summary>
    public static ShaderFeatures ToShaderFeatures(this ShaderVariantFeature features)
    {
        return (ShaderFeatures)(uint)features;
    }

    /// <summary>
    /// Gets the quality level as a string for shader preprocessor defines.
    /// </summary>
    public static string ToDefineString(this ShaderVariantQuality quality)
    {
        return quality switch
        {
            ShaderVariantQuality.Low => "QUALITY_LOW",
            ShaderVariantQuality.Medium => "QUALITY_MEDIUM",
            ShaderVariantQuality.High => "QUALITY_HIGH",
            ShaderVariantQuality.Ultra => "QUALITY_ULTRA",
            _ => "QUALITY_MEDIUM"
        };
    }

    /// <summary>
    /// Gets the maximum sphere tracing steps for a quality level.
    /// </summary>
    public static int GetMaxRayMarchSteps(this ShaderVariantQuality quality)
    {
        return quality switch
        {
            ShaderVariantQuality.Low => 32,
            ShaderVariantQuality.Medium => 64,
            ShaderVariantQuality.High => 128,
            ShaderVariantQuality.Ultra => 256,
            _ => 64
        };
    }

    /// <summary>
    /// Gets the surface intersection threshold for a quality level.
    /// </summary>
    public static float GetSurfaceThreshold(this ShaderVariantQuality quality)
    {
        return quality switch
        {
            ShaderVariantQuality.Low => 0.01f,
            ShaderVariantQuality.Medium => 0.005f,
            ShaderVariantQuality.High => 0.001f,
            ShaderVariantQuality.Ultra => 0.0005f,
            _ => 0.001f
        };
    }

    /// <summary>
    /// Gets the binary search iteration count for a quality level.
    /// </summary>
    public static int GetBinarySearchIterations(this ShaderVariantQuality quality)
    {
        return quality switch
        {
            ShaderVariantQuality.Low => 2,
            ShaderVariantQuality.Medium => 4,
            ShaderVariantQuality.High => 8,
            ShaderVariantQuality.Ultra => 16,
            _ => 8
        };
    }

    /// <summary>
    /// Gets the AO sample count for a quality level.
    /// </summary>
    public static int GetAOSampleCount(this ShaderVariantQuality quality)
    {
        return quality switch
        {
            ShaderVariantQuality.Low => 4,
            ShaderVariantQuality.Medium => 8,
            ShaderVariantQuality.High => 16,
            ShaderVariantQuality.Ultra => 32,
            _ => 16
        };
    }

    /// <summary>
    /// Gets the hierarchical tracing level count for a quality level.
    /// </summary>
    public static int GetHierarchicalLevels(this ShaderVariantQuality quality)
    {
        return quality switch
        {
            ShaderVariantQuality.Low => 0,
            ShaderVariantQuality.Medium => 2,
            ShaderVariantQuality.High => 4,
            ShaderVariantQuality.Ultra => 6,
            _ => 4
        };
    }

    /// <summary>
    /// Gets the reflection bounce count for a quality level.
    /// </summary>
    public static int GetMaxReflectionBounces(this ShaderVariantQuality quality)
    {
        return quality switch
        {
            ShaderVariantQuality.Low => 0,
            ShaderVariantQuality.Medium => 1,
            ShaderVariantQuality.High => 2,
            ShaderVariantQuality.Ultra => 4,
            _ => 2
        };
    }

    /// <summary>
    /// Gets the sphere tracing relaxation factor for a quality level.
    /// </summary>
    public static float GetRelaxation(this ShaderVariantQuality quality)
    {
        return quality switch
        {
            ShaderVariantQuality.Low => 0.6f,
            ShaderVariantQuality.Medium => 0.7f,
            ShaderVariantQuality.High => 0.8f,
            ShaderVariantQuality.Ultra => 0.9f,
            _ => 0.8f
        };
    }

    /// <summary>
    /// Generates HLSL preprocessor defines for a variant key.
    /// </summary>
    public static string GeneratePreprocessorDefines(this ShaderVariantKey key)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"#define {key.Quality.ToDefineString()} 1");
        sb.AppendLine($"#define QUALITY_LEVEL {(int)key.Quality}");
        sb.AppendLine($"#define MAX_STEPS {key.Quality.GetMaxRayMarchSteps()}");
        sb.AppendLine($"#define SURFACE_THRESHOLD {key.Quality.GetSurfaceThreshold():F6}");
        sb.AppendLine($"#define BINARY_SEARCH_ITERS {key.Quality.GetBinarySearchIterations()}");
        sb.AppendLine($"#define AO_SAMPLES {key.Quality.GetAOSampleCount()}");
        sb.AppendLine($"#define HIERARCHICAL_LEVELS {key.Quality.GetHierarchicalLevels()}");
        sb.AppendLine($"#define MAX_REFLECTION_BOUNCES {key.Quality.GetMaxReflectionBounces()}");
        sb.AppendLine($"#define RELAXATION {key.Quality.GetRelaxation():F2}");
        sb.AppendLine();

        if (key.HasFeature(ShaderVariantFeature.Shadows))
            sb.AppendLine("#define FEATURE_SHADOWS 1");
        if (key.HasFeature(ShaderVariantFeature.Reflections))
            sb.AppendLine("#define FEATURE_REFLECTIONS 1");
        if (key.HasFeature(ShaderVariantFeature.Refractions))
            sb.AppendLine("#define FEATURE_REFRACTIONS 1");
        if (key.HasFeature(ShaderVariantFeature.AmbientOcclusion))
            sb.AppendLine("#define FEATURE_AO 1");
        if (key.HasFeature(ShaderVariantFeature.HierarchicalTracing))
            sb.AppendLine("#define FEATURE_HIERARCHICAL 1");
        if (key.HasFeature(ShaderVariantFeature.BinarySearch))
            sb.AppendLine("#define FEATURE_BINARY_SEARCH 1");
        if (key.HasFeature(ShaderVariantFeature.NormalMapping))
            sb.AppendLine("#define FEATURE_NORMAL_MAPPING 1");
        if (key.HasFeature(ShaderVariantFeature.ParallaxOcclusion))
            sb.AppendLine("#define FEATURE_PARALLAX_OCCLUSION 1");
        if (key.HasFeature(ShaderVariantFeature.TemporalReprojection))
            sb.AppendLine("#define FEATURE_TEMPORAL 1");
        if (key.HasFeature(ShaderVariantFeature.SSAO))
            sb.AppendLine("#define FEATURE_SSAO 1");
        if (key.HasFeature(ShaderVariantFeature.SubsurfaceScattering))
            sb.AppendLine("#define FEATURE_SSS 1");
        if (key.HasFeature(ShaderVariantFeature.VolumetricFog))
            sb.AppendLine("#define FEATURE_FOG 1");
        if (key.HasFeature(ShaderVariantFeature.AdaptiveStepping))
            sb.AppendLine("#define FEATURE_ADAPTIVE 1");
        if (key.HasFeature(ShaderVariantFeature.TwoSided))
            sb.AppendLine("#define FEATURE_TWO_SIDED 1");
        if (key.HasFeature(ShaderVariantFeature.Emissive))
            sb.AppendLine("#define FEATURE_EMISSIVE 1");
        if (key.HasFeature(ShaderVariantFeature.AlphaTest))
            sb.AppendLine("#define FEATURE_ALPHA_TEST 1");
        if (key.HasFeature(ShaderVariantFeature.Wireframe))
            sb.AppendLine("#define FEATURE_WIREFRAME 1");
        if (key.HasFeature(ShaderVariantFeature.DebugVisualization))
            sb.AppendLine("#define FEATURE_DEBUG 1");
        if (key.HasFeature(ShaderVariantFeature.HalfPrecision))
            sb.AppendLine("#define FEATURE_HALF_PRECISION 1");
        if (key.HasFeature(ShaderVariantFeature.DynamicBranching))
            sb.AppendLine("#define FEATURE_DYNAMIC_BRANCHING 1");

        return sb.ToString();
    }

    /// <summary>
    /// Gets the relative rendering cost of a variant (0.0 = free, 1.0 = most expensive).
    /// </summary>
    public static float GetRelativeCost(this ShaderVariantKey key)
    {
        float cost = 0f;

        // Quality cost
        cost += (int)key.Quality * 0.15f;

        // Feature costs
        if (key.HasFeature(ShaderVariantFeature.Shadows))
            cost += 0.10f;
        if (key.HasFeature(ShaderVariantFeature.Reflections))
            cost += 0.15f;
        if (key.HasFeature(ShaderVariantFeature.Refractions))
            cost += 0.12f;
        if (key.HasFeature(ShaderVariantFeature.AmbientOcclusion))
            cost += 0.08f;
        if (key.HasFeature(ShaderVariantFeature.HierarchicalTracing))
            cost += 0.05f;
        if (key.HasFeature(ShaderVariantFeature.BinarySearch))
            cost += 0.03f;
        if (key.HasFeature(ShaderVariantFeature.NormalMapping))
            cost += 0.04f;
        if (key.HasFeature(ShaderVariantFeature.ParallaxOcclusion))
            cost += 0.08f;
        if (key.HasFeature(ShaderVariantFeature.TemporalReprojection))
            cost += 0.06f;
        if (key.HasFeature(ShaderVariantFeature.SSAO))
            cost += 0.10f;
        if (key.HasFeature(ShaderVariantFeature.SubsurfaceScattering))
            cost += 0.07f;
        if (key.HasFeature(ShaderVariantFeature.VolumetricFog))
            cost += 0.12f;
        if (key.HasFeature(ShaderVariantFeature.AdaptiveStepping))
            cost += 0.02f;

        return Math.Clamp(cost, 0f, 1f);
    }
}
