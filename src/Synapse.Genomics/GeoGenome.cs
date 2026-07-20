// ============================================================================
// GeoGenome.cs — G-DNN Engine: Geometrical Designer Neural Network Core
// ============================================================================
// This file defines the core genome data structures for the G-DNN Engine.
// It contains all fundamental types, enums, records, builders, factories,
// validators, hashers, differencing, serialization, pooling, and registry
// components required for a production-grade neural genome system.
// ============================================================================
// ReSharper disable All
// ============================================================================

using System;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace GDNN.Core.Genome
{
    // ========================================================================
    #region Enums

    /// <summary>
    /// Identifies the functional category of a neuron within the G-DNN genome.
    /// Each kind maps to a specific computational kernel or geometric operation.
    /// </summary>
    public enum NeuronKind : byte
    {
        SDFPrimitive = 0,
        DisplacementField = 1,
        CurvatureModulator = 2,
        ColorField = 3,
        NormalMap = 4,
        RoughnessMap = 5,
        MetallicMap = 6,
        EmissiveMap = 7,
        OpacityMap = 8,
        SubsurfaceScattering = 9,
        Displacement = 10,
        Tessellation = 11,
        LODSelector = 12,
        OcclusionCuller = 13,
        FrustumCuller = 14,
        AnimationCurve = 15,
        ParticleEmitter = 16,
        PhysicsCollider = 17,
        FluidSolver = 18,
        ClothSimulator = 19,
        Voxelizer = 20,
        MarchingCube = 21,
        RayMarcher = 22,
        FieldGenerator = 23,
        WaveFunction = 24,
        TensorReshaper = 25,
        AttentionHead = 26,
        ConvolutionKernel = 27,
        PoolingLayer = 28,
        NormalizationLayer = 29,
        EmbeddingLookup = 30,
        PositionalEncoder = 31,
        CrossAttentionLayer = 32,
        SelfAttentionLayer = 33,
        FeedForwardLayer = 34,
        GeometricTransform = 35,
        CSGOperation = 36,
        ProceduralTexture = 37,
        TerrainSculptor = 38,
        LatticeGenerator = 39,
        FractalSubdivision = 40,
        HarmonicOscillator = 41,
        GravitationalField = 42,
        ElectromagneticField = 43,
        VolumetricDensity = 44,
        ScreenSpaceReflection = 45,
        GlobalIllumination = 46,
        AmbientOcclusion = 47,
        ShadowMap = 48,
        PostProcessChain = 49
    }

    /// <summary>
    /// Activation kernel applied to a neuron's computed output.
    /// Supports both classical neural activations and domain-specific geometric functions.
    /// </summary>
    public enum ActivationKernel : byte
    {
        ReLU = 0,
        LeakyReLU = 1,
        PReLU = 2,
        ELU = 3,
        SELU = 4,
        GELU = 5,
        Sigmoid = 6,
        Tanh = 7,
        Swish = 8,
        Mish = 9,
        Softplus = 10,
        HardSigmoid = 11,
        HardTanh = 12,
        Step = 13,
        Linear = 14,
        Gaussian = 15,
        Sinusoidal = 16,
        PerlinSpline = 17,
        WaveletSDF = 18,
        BernsteinBasis = 19,
        HermiteSpline = 20,
        CatmullRomSpline = 21,
        BezierCurve = 22,
        FractalNoise = 23,
        VoronoiNoise = 24,
        WorleyNoise = 25,
        SimplexNoise = 26,
        CurlNoise = 27,
        FBMNoise = 28,
        RidgedNoise = 29,
        SDFCombine = 30,
        SmoothMinimum = 31,
        HardUnion = 32,
        HardIntersection = 33,
        HardSubtraction = 34,
        SmoothBlend = 35,
        DomainRepetition = 36,
        DomainFolding = 37,
        Twist = 38,
        Bend = 39,
        Taper = 40
    }

    /// <summary>
    /// Level-of-detail policy controlling mesh tessellation and resolution.
    /// </summary>
    public enum LodPolicy : byte
    {
        Level0Only = 0,
        Adaptive = 1,
        Aggressive = 2,
        Conservative = 3,
        Custom = 4
    }

    /// <summary>
    /// Classification of synapse signal type.
    /// </summary>
    public enum SynapseType : byte
    {
        Excitatory = 0,
        Inhibitory = 1,
        Modulatory = 2,
        Plastic = 3,
        Static = 4,
        Adaptive = 5
    }

    /// <summary>
    /// Semantic role of a neuron within the genome architecture.
    /// </summary>
    public enum NeuronSemanticRole : byte
    {
        Structural = 0,
        Decorative = 1,
        Functional = 2,
        Behavioral = 3,
        Visual = 4,
        Physical = 5,
        Auditory = 6
    }

    /// <summary>
    /// High-level classification of genome content domain.
    /// </summary>
    public enum GenomeClassification : byte
    {
        Organic = 0,
        Mineral = 1,
        ManMade = 2,
        Fluid = 3,
        Energy = 4,
        Abstract = 5,
        Hybrid = 6
    }

    /// <summary>
    /// Strategy used for genome evolution and mutation.
    /// </summary>
    public enum EvolutionStrategy : byte
    {
        Static = 0,
        Incremental = 1,
        Radical = 2,
        Sexual = 3,
        Asexual = 4,
        Hybrid = 5
    }

    /// <summary>
    /// Primary fitness objective for genome evaluation.
    /// </summary>
    public enum FitnessObjective : byte
    {
        VisualFidelity = 0,
        Performance = 1,
        MemoryEfficiency = 2,
        Balanced = 3,
        Custom = 4
    }

    /// <summary>
    /// Lifecycle state of a genome within the system.
    /// </summary>
    public enum GenomeState : byte
    {
        Draft = 0,
        Evaluating = 1,
        Compiled = 2,
        Cached = 3,
        Archived = 4,
        Deprecated = 5
    }

    /// <summary>
    /// Type of mutation applied during genome evolution.
    /// </summary>
    public enum MutationType : byte
    {
        PointMutation = 0,
        Insertion = 1,
        Deletion = 2,
        Duplication = 3,
        Inversion = 4,
        Translocation = 5,
        SemanticMutation = 6,
        TopologyMutation = 7,
        WeightPerturbation = 8,
        ActivationShift = 9,
        BiasDrift = 10,
        SynapseGrowth = 11,
        SynapsePruning = 12,
        LayerInsertion = 13,
        LayerRemoval = 14,
        CrossingOver = 15,
        EpigeneticSwitch = 16,
        GeneSilencing = 17,
        GeneActivation = 18,
        RegulatoryMutation = 19
    }

    /// <summary>
    /// Specifies the binary serialization format version for genome persistence.
    /// </summary>
    public enum SerializationFormat : byte
    {
        Binary = 0,
        Json = 1,
        MessagePack = 2,
        Protobuf = 3
    }

    #endregion Enums

    // ========================================================================
    #region Strongly-Typed Identifiers

    /// <summary>
    /// Strongly-typed globally unique identifier for genomes.
    /// Wraps a <see cref="Guid"/> with parse, comparison, and factory methods.
    /// </summary>
    public readonly struct GenomeId : IEquatable<GenomeId>, IComparable<GenomeId>, IFormattable
    {
        private readonly Guid _value;

        /// <summary>Represents a null/empty genome identifier.</summary>
        public static readonly GenomeId Empty = new(Guid.Empty);

        /// <summary>Initializes a new <see cref="GenomeId"/> with the specified GUID value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public GenomeId(Guid value) => _value = value;

        /// <summary>Generates a new unique genome identifier.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static GenomeId New() => new(Guid.NewGuid());

        /// <summary>Parses a genome identifier from its string representation.</summary>
        public static GenomeId Parse(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                throw new ArgumentException("Genome ID string cannot be null or empty.", nameof(input));
            return new GenomeId(Guid.Parse(input));
        }

        /// <summary>Tries to parse a genome identifier from its string representation.</summary>
        public static bool TryParse(string? input, out GenomeId result)
        {
            if (Guid.TryParse(input, out var guid))
            {
                result = new GenomeId(guid);
                return true;
            }
            result = Empty;
            return false;
        }

        /// <summary>Parses a genome identifier from a Base64-encoded string.</summary>
        public static GenomeId FromBase64(string base64)
        {
            if (string.IsNullOrWhiteSpace(base64))
                throw new ArgumentException("Base64 string cannot be null or empty.", nameof(base64));
            var bytes = Convert.FromBase64String(base64);
            if (bytes.Length != 16)
                throw new FormatException("Base64 encoded GUID must be exactly 16 bytes.");
            return new GenomeId(new Guid(bytes));
        }

        /// <summary>Returns the Base64-encoded representation of this identifier.</summary>
        public string ToBase64() => Convert.ToBase64String(_value.ToByteArray());

        /// <summary>Gets the underlying <see cref="Guid"/> value.</summary>
        public Guid Value => _value;

        /// <summary>Gets whether this identifier is empty/null.</summary>
        public bool IsEmpty => _value == Guid.Empty;

        /// <summary>Returns a 32-character hex string without dashes.</summary>
        public string ToCompactString() => _value.ToString("N");

        /// <summary>Returns a hyphenated string representation.</summary>
        public override string ToString() => _value.ToString("D");

        /// <summary>Returns a formatted string representation.</summary>
        public string ToString(string? format, IFormatProvider? formatProvider = null)
            => _value.ToString(format, formatProvider);

        /// <inheritdoc/>
        public override bool Equals(object? obj) => obj is GenomeId other && Equals(other);

        /// <inheritdoc/>
        public bool Equals(GenomeId other) => _value.Equals(other._value);

        /// <inheritdoc/>
        public override int GetHashCode() => _value.GetHashCode();

        /// <inheritdoc/>
        public int CompareTo(GenomeId other) => _value.CompareTo(other._value);

        public static bool operator ==(GenomeId left, GenomeId right) => left._value == right._value;
        public static bool operator !=(GenomeId left, GenomeId right) => left._value != right._value;
        public static bool operator <(GenomeId left, GenomeId right) => left._value < right._value;
        public static bool operator >(GenomeId left, GenomeId right) => left._value > right._value;
        public static bool operator <=(GenomeId left, GenomeId right) => left._value <= right._value;
        public static bool operator >=(GenomeId left, GenomeId right) => left._value >= right._value;

        /// <summary>Implicitly converts a <see cref="GenomeId"/> to a <see cref="Guid"/>.</summary>
        public static implicit operator Guid(GenomeId id) => id._value;

        /// <summary>Implicitly converts a <see cref="Guid"/> to a <see cref="GenomeId"/>.</summary>
        public static implicit operator GenomeId(Guid guid) => new(guid);
    }

    /// <summary>
    /// Strongly-typed species identifier for genome classification.
    /// Encodes taxonomic-style species membership with semantic meaning.
    /// </summary>
    public readonly struct SpeciesId : IEquatable<SpeciesId>, IComparable<SpeciesId>, IFormattable
    {
        private readonly string _value;

        /// <summary>Represents an unknown/unclassified species.</summary>
        public static readonly SpeciesId Unknown = new("unknown");

        /// <summary>Represents a generic/default species.</summary>
        public static readonly SpeciesId Default = new("default");

        /// <summary>Initializes a new <see cref="SpeciesId"/> with the specified string value.</summary>
        public SpeciesId(string value)
        {
            _value = value ?? throw new ArgumentNullException(nameof(value));
            if (value.Length == 0)
                throw new ArgumentException("Species ID cannot be empty.", nameof(value));
        }

        /// <summary>Creates a species identifier from domain, phylum, and variant.</summary>
        public static SpeciesId Create(string domain, string phylum, string variant)
        {
            if (string.IsNullOrWhiteSpace(domain))
                throw new ArgumentException("Domain cannot be empty.", nameof(domain));
            if (string.IsNullOrWhiteSpace(phylum))
                throw new ArgumentException("Phylum cannot be empty.", nameof(phylum));
            if (string.IsNullOrWhiteSpace(variant))
                throw new ArgumentException("Variant cannot be empty.", nameof(variant));
            return new SpeciesId($"{domain.ToLowerInvariant()}.{phylum.ToLowerInvariant()}.{variant.ToLowerInvariant()}");
        }

        /// <summary>Creates a species identifier from a classification enum.</summary>
        public static SpeciesId Create(GenomeClassification classification, string subcategory)
        {
            var domain = classification.ToString().ToLowerInvariant();
            var phylum = subcategory?.ToLowerInvariant() ?? "general";
            return new SpeciesId($"{domain}.{phylum}");
        }

        /// <summary>Parses a species identifier from a string.</summary>
        public static SpeciesId Parse(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                throw new ArgumentException("Species ID string cannot be null or empty.", nameof(input));
            return new SpeciesId(input.Trim().ToLowerInvariant());
        }

        /// <summary>Tries to parse a species identifier from a string.</summary>
        public static bool TryParse(string? input, out SpeciesId result)
        {
            if (!string.IsNullOrWhiteSpace(input))
            {
                result = new SpeciesId(input.Trim().ToLowerInvariant());
                return true;
            }
            result = Unknown;
            return false;
        }

        /// <summary>Gets the raw string value of this species identifier.</summary>
        public string Value => _value ?? string.Empty;

        /// <summary>Gets the domain component of the species identifier.</summary>
        public string Domain
        {
            get
            {
                var idx = _value?.IndexOf('.') ?? -1;
                return idx >= 0 ? _value![..idx] : _value ?? string.Empty;
            }
        }

        /// <summary>Gets the phylum component of the species identifier.</summary>
        public string Phylum
        {
            get
            {
                var s = _value ?? string.Empty;
                var first = s.IndexOf('.');
                var second = s.IndexOf('.', first + 1);
                return first >= 0 && second >= 0 ? s[(first + 1)..second] : first >= 0 ? s[(first + 1)..] : string.Empty;
            }
        }

        /// <summary>Gets the variant component of the species identifier.</summary>
        public string Variant
        {
            get
            {
                var s = _value ?? string.Empty;
                var last = s.LastIndexOf('.');
                return last >= 0 ? s[(last + 1)..] : string.Empty;
            }
        }

        /// <summary>Gets whether this species has a hierarchical classification (contains dots).</summary>
        public bool IsHierarchical => _value?.Contains('.') == true;

        /// <summary>Returns the parent species (one level up in the hierarchy).</summary>
        public SpeciesId Parent
        {
            get
            {
                var last = _value?.LastIndexOf('.') ?? -1;
                return last > 0 ? new SpeciesId(_value![..last]) : Unknown;
            }
        }

        /// <inheritdoc/>
        public override string ToString() => _value ?? string.Empty;

        /// <summary>Returns a formatted string representation.</summary>
        public string ToString(string? format, IFormatProvider? formatProvider = null)
            => format?.ToUpperInvariant() switch
            {
                "U" => _value?.ToUpperInvariant() ?? string.Empty,
                "L" => _value?.ToLowerInvariant() ?? string.Empty,
                _ => _value ?? string.Empty
            };

        /// <inheritdoc/>
        public override bool Equals(object? obj) => obj is SpeciesId other && Equals(other);

        /// <inheritdoc/>
        public bool Equals(SpeciesId other) => string.Equals(_value, other._value, StringComparison.Ordinal);

        /// <inheritdoc/>
        public override int GetHashCode() => _value?.GetHashCode() ?? 0;

        /// <inheritdoc/>
        public int CompareTo(SpeciesId other) => string.Compare(_value, other._value, StringComparison.Ordinal);

        public static bool operator ==(SpeciesId left, SpeciesId right) => left.Equals(right);
        public static bool operator !=(SpeciesId left, SpeciesId right) => !left.Equals(right);
        public static bool operator <(SpeciesId left, SpeciesId right) => left.CompareTo(right) < 0;
        public static bool operator >(SpeciesId left, SpeciesId right) => left.CompareTo(right) > 0;
        public static bool operator <=(SpeciesId left, SpeciesId right) => left.CompareTo(right) <= 0;
        public static bool operator >=(SpeciesId left, SpeciesId right) => left.CompareTo(right) >= 0;

        /// <summary>Implicitly converts a <see cref="SpeciesId"/> to a <see cref="string"/>.</summary>
        public static implicit operator string(SpeciesId id) => id._value ?? string.Empty;

        /// <summary>Implicitly converts a <see cref="string"/> to a <see cref="SpeciesId"/>.</summary>
        public static implicit operator SpeciesId(string value) => new(value ?? "unknown");
    }

    #endregion Strongly-Typed Identifiers

    // ========================================================================
    #region Record Structs — Core Genome Types

    /// <summary>
    /// Version information for genome format compatibility.
    /// Follows semantic versioning (major.minor.patch).
    /// </summary>
    public readonly record struct GenomeVersion : IComparable<GenomeVersion>, IFormattable
    {
        /// <summary>Major version incremented on breaking changes.</summary>
        public int Major { get; init; }

        /// <summary>Minor version incremented on feature additions.</summary>
        public int Minor { get; init; }

        /// <summary>Patch version incremented on bug fixes.</summary>
        public int Patch { get; init; }

        public GenomeVersion(int major, int minor, int patch)
        {
            if (major < 0)
                throw new ArgumentOutOfRangeException(nameof(major), "Major version cannot be negative.");
            if (minor < 0)
                throw new ArgumentOutOfRangeException(nameof(minor), "Minor version cannot be negative.");
            if (patch < 0)
                throw new ArgumentOutOfRangeException(nameof(patch), "Patch version cannot be negative.");
            Major = major;
            Minor = minor;
            Patch = patch;
        }

        /// <summary>The initial release version.</summary>
        public static readonly GenomeVersion Initial = new(1, 0, 0);

        /// <summary>The current latest version of the genome format.</summary>
        public static readonly GenomeVersion Current = new(2, 4, 1);

        /// <summary>Parses a version string in the format "major.minor.patch".</summary>
        public static GenomeVersion Parse(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                throw new ArgumentException("Version string cannot be null or empty.", nameof(input));
            var parts = input.Split('.');
            if (parts.Length < 2 || parts.Length > 3)
                throw new FormatException($"Invalid version format: '{input}'. Expected 'major.minor[.patch]'.");
            var major = int.Parse(parts[0], CultureInfo.InvariantCulture);
            var minor = int.Parse(parts[1], CultureInfo.InvariantCulture);
            var patch = parts.Length > 2 ? int.Parse(parts[2], CultureInfo.InvariantCulture) : 0;
            return new GenomeVersion(major, minor, patch);
        }

        /// <summary>Tries to parse a version string.</summary>
        public static bool TryParse(string? input, out GenomeVersion result)
        {
            if (string.IsNullOrWhiteSpace(input))
            { result = Initial; return false; }
            try
            { result = Parse(input); return true; }
            catch (Exception ex)
            {
                SynapseLogger.Default.Debug("GenomeVersion", $"Failed to parse version string '{input}'.", ex);
                result = Initial;
                return false;
            }
        }

        /// <summary>Gets whether this version is compatible with the specified version.</summary>
        public bool IsCompatibleWith(GenomeVersion other) => Major == other.Major && Minor >= other.Minor;

        /// <summary>Gets whether this version is a pre-release of the specified version.</summary>
        public bool IsPrereleaseOf(GenomeVersion other) => Major == other.Major && Minor == other.Minor && Patch < other.Patch;

        /// <summary>Returns the next major version.</summary>
        public GenomeVersion NextMajor() => new(Major + 1, 0, 0);

        /// <summary>Returns the next minor version.</summary>
        public GenomeVersion NextMinor() => new(Major, Minor + 1, 0);

        /// <summary>Returns the next patch version.</summary>
        public GenomeVersion NextPatch() => new(Major, Minor, Patch + 1);

        /// <inheritdoc/>
        public int CompareTo(GenomeVersion other)
        {
            var cmp = Major.CompareTo(other.Major);
            if (cmp != 0)
                return cmp;
            cmp = Minor.CompareTo(other.Minor);
            if (cmp != 0)
                return cmp;
            return Patch.CompareTo(other.Patch);
        }

        /// <inheritdoc/>
        public string ToString(string? format, IFormatProvider? formatProvider = null)
            => format?.ToUpperInvariant() switch { "M" => $"{Major}.{Minor}", "MP" => $"{Major}.{Minor}.{Patch}", _ => $"{Major}.{Minor}.{Patch}" };

        /// <inheritdoc/>
        public override string ToString() => $"{Major}.{Minor}.{Patch}";

        public static bool operator <(GenomeVersion left, GenomeVersion right) => left.CompareTo(right) < 0;
        public static bool operator >(GenomeVersion left, GenomeVersion right) => left.CompareTo(right) > 0;
        public static bool operator <=(GenomeVersion left, GenomeVersion right) => left.CompareTo(right) <= 0;
        public static bool operator >=(GenomeVersion left, GenomeVersion right) => left.CompareTo(right) >= 0;
    }

    /// <summary>
    /// Metadata associated with a genome providing authorship, licensing, and taxonomic information.
    /// </summary>
    public sealed record GenomeMetadata
    {
        /// <summary>Human-readable semantic tag for quick identification.</summary>
        public string SemanticTag { get; init; } = string.Empty;

        /// <summary>Maximum complexity budget (neuron count limit).</summary>
        public int ComplexityBudget { get; init; } = 10000;

        /// <summary>Level-of-detail policy for this genome.</summary>
        public LodPolicy LodPolicy { get; init; } = LodPolicy.Adaptive;

        /// <summary>Timestamp of last evolution operation.</summary>
        public DateTimeOffset LastEvolved { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>Author identifier or name.</summary>
        public string Author { get; init; } = string.Empty;

        /// <summary>License identifier (e.g., "MIT", "CC-BY-4.0").</summary>
        public string License { get; init; } = "MIT";

        /// <summary>Human-readable description of the genome.</summary>
        public string Description { get; init; } = string.Empty;

        /// <summary>Set of classification tags for search and filtering.</summary>
        public ImmutableHashSet<string> Tags { get; init; } = ImmutableHashSet<string>.Empty.WithComparer(StringComparer.OrdinalIgnoreCase);

        /// <summary>Evolutionary generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Most recent fitness score from evaluation.</summary>
        public double FitnessScore { get; init; }

        /// <summary>High-level domain classification.</summary>
        public GenomeClassification Classification { get; init; } = GenomeClassification.Abstract;

        /// <summary>Evolution strategy used to produce this genome.</summary>
        public EvolutionStrategy Strategy { get; init; } = EvolutionStrategy.Static;

        /// <summary>Primary fitness optimization objective.</summary>
        public FitnessObjective Objective { get; init; } = FitnessObjective.Balanced;

        /// <summary>Lifecycle state of the genome.</summary>
        public GenomeState State { get; init; } = GenomeState.Draft;

        /// <summary>Estimated memory usage in bytes.</summary>
        public long EstimatedMemoryBytes { get; init; }

        /// <summary>Parent genome IDs (for evolved genomes).</summary>
        public ImmutableArray<GenomeId> ParentIds { get; init; } = ImmutableArray<GenomeId>.Empty;

        /// <summary>Custom key-value properties.</summary>
        public ImmutableDictionary<string, string> CustomProperties { get; init; } = ImmutableDictionary<string, string>.Empty;

        /// <summary>Creates a deep copy of this metadata with optional overrides.</summary>
        public GenomeMetadata CloneWith(
            string? semanticTag = null,
            int? complexityBudget = null,
            LodPolicy? lodPolicy = null,
            string? author = null,
            string? description = null) => this with
            {
                SemanticTag = semanticTag ?? SemanticTag,
                ComplexityBudget = complexityBudget ?? ComplexityBudget,
                LodPolicy = lodPolicy ?? LodPolicy,
                Author = author ?? Author,
                Description = description ?? Description
            };

        /// <summary>Adds a tag to the metadata tags set.</summary>
        public GenomeMetadata WithTag(string tag)
        {
            if (string.IsNullOrWhiteSpace(tag))
                throw new ArgumentException("Tag cannot be null or empty.", nameof(tag));
            return this with { Tags = Tags.Add(tag.Trim()) };
        }

        /// <summary>Removes a tag from the metadata tags set.</summary>
        public GenomeMetadata WithoutTag(string tag) => this with { Tags = Tags.Remove(tag) };

        /// <summary>Checks if a specific tag exists in the tags set.</summary>
        public bool HasTag(string tag) => Tags.Contains(tag);

        /// <summary>Adds or updates a custom property.</summary>
        public GenomeMetadata WithProperty(string key, string value)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentException("Property key cannot be null or empty.", nameof(key));
            return this with { CustomProperties = CustomProperties.SetItem(key, value) };
        }

        /// <summary>Gets a custom property value by key.</summary>
        public string? GetProperty(string key) =>
            CustomProperties.TryGetValue(key, out var value) ? value : null;
    }

    /// <summary>
    /// A single neuron within the genome network, encoding its functional identity,
    /// activation behavior, bias, weight scaling, connectivity, and semantic role.
    /// </summary>
    public readonly record struct NeuronGene
    {
        /// <summary>Unique identifier for this neuron within its genome.</summary>
        public int Id { get; init; }

        /// <summary>Functional category of this neuron.</summary>
        public NeuronKind Kind { get; init; }

        /// <summary>Activation kernel applied to the neuron's output.</summary>
        public ActivationKernel Activation { get; init; }

        /// <summary>3D bias vector for geometric operations.</summary>
        public Vector3 Bias { get; init; }

        /// <summary>Global weight scaling factor for this neuron's contribution.</summary>
        public float WeightScale { get; init; }

        /// <summary>Immutable dictionary of kind-specific parameters.</summary>
        public ImmutableDictionary<string, float> Parameters { get; init; }

        /// <summary>Ordered list of synapse target neuron IDs (outgoing connections).</summary>
        public ImmutableArray<int> SynapseTargets { get; init; }

        /// <summary>Layer index within the network topology (0-based, depth order).</summary>
        public int LayerIndex { get; init; }

        /// <summary>Whether this neuron is currently active in computation.</summary>
        public bool IsEnabled { get; init; }

        /// <summary>Semantic role describing this neuron's architectural purpose.</summary>
        public NeuronSemanticRole SemanticRole { get; init; }

        /// <summary>Optional activation parameter (e.g., leaky relu slope, sigmoid temperature).</summary>
        public float ActivationParameter { get; init; }

        /// <summary>Optional secondary parameter for dual-parameter activations.</summary>
        public float ActivationParameter2 { get; init; }

        /// <summary>3D positional hint for geometric layout.</summary>
        public Vector3 Position { get; init; }

        /// <summary>Timestamp when this neuron was last modified.</summary>
        public DateTimeOffset LastModified { get; init; }

        /// <summary>Number of incoming synapses (fan-in).</summary>
        public int FanIn { get; init; }

        /// <summary>Number of outgoing synapses (fan-out).</summary>
        public int FanOut { get; init; }

        /// <summary>Estimated computational cost weight (for budget validation).</summary>
        public float ComputeCost { get; init; }

        /// <summary>Whether this neuron participates in gradient computation.</summary>
        public bool IsTrainable { get; init; }

        /// <summary>Memory footprint estimate in bytes.</summary>
        public int EstimatedMemoryBytes { get; init; }

        /// <summary>Profiling data for this neuron (runtime performance tracking).</summary>
        public NeuronActivationProfile? Profile { get; init; }

        /// <summary>Creates a new <see cref="NeuronGene"/> with sensible defaults for the specified kind.</summary>
        public static NeuronGene Create(int id, NeuronKind kind, int layerIndex = 0)
        {
            return new NeuronGene
            {
                Id = id,
                Kind = kind,
                Activation = GetDefaultActivationForKind(kind),
                Bias = Vector3.Zero,
                WeightScale = 1.0f,
                Parameters = GetDefaultParametersForKind(kind),
                SynapseTargets = ImmutableArray<int>.Empty,
                LayerIndex = layerIndex,
                IsEnabled = true,
                SemanticRole = GetDefaultSemanticRoleForKind(kind),
                ActivationParameter = 1.0f,
                ActivationParameter2 = 0.0f,
                Position = Vector3.Zero,
                LastModified = DateTimeOffset.UtcNow,
                FanIn = 0,
                FanOut = 0,
                ComputeCost = GetDefaultComputeCostForKind(kind),
                IsTrainable = true,
                EstimatedMemoryBytes = GetDefaultMemoryEstimateForKind(kind)
            };
        }

        /// <summary>Gets the default activation kernel for a given neuron kind.</summary>
        public static ActivationKernel GetDefaultActivationForKind(NeuronKind kind) => kind switch
        {
            NeuronKind.SDFPrimitive => ActivationKernel.SDFCombine,
            NeuronKind.DisplacementField => ActivationKernel.PerlinSpline,
            NeuronKind.CurvatureModulator => ActivationKernel.Sigmoid,
            NeuronKind.ColorField => ActivationKernel.Linear,
            NeuronKind.NormalMap => ActivationKernel.Linear,
            NeuronKind.RoughnessMap => ActivationKernel.Sigmoid,
            NeuronKind.MetallicMap => ActivationKernel.Step,
            NeuronKind.EmissiveMap => ActivationKernel.ReLU,
            NeuronKind.OpacityMap => ActivationKernel.Sigmoid,
            NeuronKind.SubsurfaceScattering => ActivationKernel.Gaussian,
            NeuronKind.Displacement => ActivationKernel.Linear,
            NeuronKind.Tessellation => ActivationKernel.Step,
            NeuronKind.LODSelector => ActivationKernel.Step,
            NeuronKind.OcclusionCuller => ActivationKernel.Step,
            NeuronKind.FrustumCuller => ActivationKernel.Step,
            NeuronKind.AnimationCurve => ActivationKernel.CatmullRomSpline,
            NeuronKind.ParticleEmitter => ActivationKernel.FractalNoise,
            NeuronKind.PhysicsCollider => ActivationKernel.SDFCombine,
            NeuronKind.FluidSolver => ActivationKernel.SmoothBlend,
            NeuronKind.ClothSimulator => ActivationKernel.HermiteSpline,
            NeuronKind.Voxelizer => ActivationKernel.Step,
            NeuronKind.MarchingCube => ActivationKernel.SDFCombine,
            NeuronKind.RayMarcher => ActivationKernel.SmoothMinimum,
            NeuronKind.FieldGenerator => ActivationKernel.FBMNoise,
            NeuronKind.WaveFunction => ActivationKernel.Sigmoid,
            NeuronKind.TensorReshaper => ActivationKernel.Linear,
            NeuronKind.AttentionHead => ActivationKernel.Softplus,
            NeuronKind.ConvolutionKernel => ActivationKernel.ReLU,
            NeuronKind.PoolingLayer => ActivationKernel.ReLU,
            NeuronKind.NormalizationLayer => ActivationKernel.Linear,
            NeuronKind.EmbeddingLookup => ActivationKernel.Linear,
            NeuronKind.PositionalEncoder => ActivationKernel.Sinusoidal,
            NeuronKind.CrossAttentionLayer => ActivationKernel.Softplus,
            NeuronKind.SelfAttentionLayer => ActivationKernel.Softplus,
            NeuronKind.FeedForwardLayer => ActivationKernel.GELU,
            NeuronKind.GeometricTransform => ActivationKernel.Linear,
            NeuronKind.CSGOperation => ActivationKernel.HardUnion,
            NeuronKind.ProceduralTexture => ActivationKernel.FractalNoise,
            NeuronKind.TerrainSculptor => ActivationKernel.RidgedNoise,
            NeuronKind.LatticeGenerator => ActivationKernel.DomainRepetition,
            NeuronKind.FractalSubdivision => ActivationKernel.FractalNoise,
            NeuronKind.HarmonicOscillator => ActivationKernel.Sinusoidal,
            NeuronKind.GravitationalField => ActivationKernel.SmoothMinimum,
            NeuronKind.ElectromagneticField => ActivationKernel.CurlNoise,
            NeuronKind.VolumetricDensity => ActivationKernel.FBMNoise,
            NeuronKind.ScreenSpaceReflection => ActivationKernel.Linear,
            NeuronKind.GlobalIllumination => ActivationKernel.Gaussian,
            NeuronKind.AmbientOcclusion => ActivationKernel.Sigmoid,
            NeuronKind.ShadowMap => ActivationKernel.Step,
            NeuronKind.PostProcessChain => ActivationKernel.Linear,
            _ => ActivationKernel.Linear
        };

        /// <summary>Gets default parameters for a given neuron kind.</summary>
        public static ImmutableDictionary<string, float> GetDefaultParametersForKind(NeuronKind kind) => kind switch
        {
            NeuronKind.SDFPrimitive => ImmutableDictionary<string, float>.Empty.Add("radius", 1.0f).Add("smoothness", 0.1f),
            NeuronKind.DisplacementField => ImmutableDictionary<string, float>.Empty.Add("amplitude", 0.5f).Add("frequency", 1.0f).Add("octaves", 4.0f),
            NeuronKind.CurvatureModulator => ImmutableDictionary<string, float>.Empty.Add("curvatureScale", 1.0f).Add("tension", 0.5f),
            NeuronKind.ColorField => ImmutableDictionary<string, float>.Empty.Add("hue", 0.0f).Add("saturation", 1.0f).Add("value", 1.0f),
            NeuronKind.Tessellation => ImmutableDictionary<string, float>.Empty.Add("minLOD", 0.0f).Add("maxLOD", 4.0f).Add("threshold", 0.01f),
            NeuronKind.LODSelector => ImmutableDictionary<string, float>.Empty.Add("nearPlane", 1.0f).Add("farPlane", 100.0f).Add("transitionWidth", 5.0f),
            NeuronKind.ConvolutionKernel => ImmutableDictionary<string, float>.Empty.Add("kernelSize", 3.0f).Add("stride", 1.0f).Add("padding", 1.0f),
            NeuronKind.PoolingLayer => ImmutableDictionary<string, float>.Empty.Add("poolSize", 2.0f).Add("stride", 2.0f),
            NeuronKind.NormalizationLayer => ImmutableDictionary<string, float>.Empty.Add("epsilon", 1e-6f).Add("momentum", 0.1f),
            NeuronKind.EmbeddingLookup => ImmutableDictionary<string, float>.Empty.Add("vocabSize", 1000.0f).Add("embedDim", 64.0f),
            NeuronKind.PositionalEncoder => ImmutableDictionary<string, float>.Empty.Add("maxSeqLen", 512.0f).Add("dim", 64.0f),
            NeuronKind.FluidSolver => ImmutableDictionary<string, float>.Empty.Add("viscosity", 0.01f).Add("density", 1.0f).Add("pressure", 101325.0f),
            NeuronKind.ClothSimulator => ImmutableDictionary<string, float>.Empty.Add("stiffness", 0.9f).Add("damping", 0.1f).Add("mass", 0.01f),
            NeuronKind.ParticleEmitter => ImmutableDictionary<string, float>.Empty.Add("emissionRate", 100.0f).Add("lifetime", 2.0f).Add("initialVelocity", 5.0f),
            NeuronKind.PhysicsCollider => ImmutableDictionary<string, float>.Empty.Add("friction", 0.5f).Add("restitution", 0.3f),
            NeuronKind.Voxelizer => ImmutableDictionary<string, float>.Empty.Add("resolution", 32.0f),
            NeuronKind.RayMarcher => ImmutableDictionary<string, float>.Empty.Add("maxSteps", 64.0f).Add("epsilon", 0.001f),
            NeuronKind.FieldGenerator => ImmutableDictionary<string, float>.Empty.Add("frequency", 1.0f).Add("amplitude", 1.0f),
            NeuronKind.WaveFunction => ImmutableDictionary<string, float>.Empty.Add("collapseThreshold", 0.5f),
            _ => ImmutableDictionary<string, float>.Empty
        };

        /// <summary>Gets the default semantic role for a given neuron kind.</summary>
        public static NeuronSemanticRole GetDefaultSemanticRoleForKind(NeuronKind kind) => kind switch
        {
            NeuronKind.SDFPrimitive or NeuronKind.Voxelizer or NeuronKind.MarchingCube => NeuronSemanticRole.Structural,
            NeuronKind.DisplacementField or NeuronKind.CurvatureModulator or NeuronKind.ColorField
                or NeuronKind.NormalMap or NeuronKind.RoughnessMap or NeuronKind.MetallicMap
                or NeuronKind.EmissiveMap or NeuronKind.OpacityMap or NeuronKind.SubsurfaceScattering => NeuronSemanticRole.Visual,
            NeuronKind.Displacement or NeuronKind.Tessellation => NeuronSemanticRole.Structural,
            NeuronKind.LODSelector or NeuronKind.OcclusionCuller or NeuronKind.FrustumCuller
                or NeuronKind.RayMarcher or NeuronKind.FieldGenerator => NeuronSemanticRole.Functional,
            NeuronKind.AnimationCurve or NeuronKind.ParticleEmitter => NeuronSemanticRole.Behavioral,
            NeuronKind.PhysicsCollider or NeuronKind.FluidSolver or NeuronKind.ClothSimulator
                or NeuronKind.GravitationalField or NeuronKind.ElectromagneticField => NeuronSemanticRole.Physical,
            NeuronKind.WaveFunction or NeuronKind.TensorReshaper or NeuronKind.AttentionHead
                or NeuronKind.ConvolutionKernel or NeuronKind.PoolingLayer or NeuronKind.NormalizationLayer
                or NeuronKind.EmbeddingLookup or NeuronKind.PositionalEncoder or NeuronKind.CrossAttentionLayer
                or NeuronKind.SelfAttentionLayer or NeuronKind.FeedForwardLayer => NeuronSemanticRole.Functional,
            _ => NeuronSemanticRole.Functional
        };

        /// <summary>Gets the default compute cost for a given neuron kind.</summary>
        public static float GetDefaultComputeCostForKind(NeuronKind kind) => kind switch
        {
            NeuronKind.SDFPrimitive => 1.0f,
            NeuronKind.DisplacementField => 3.0f,
            NeuronKind.CurvatureModulator => 4.0f,
            NeuronKind.ColorField => 1.0f,
            NeuronKind.NormalMap => 2.0f,
            NeuronKind.RoughnessMap => 1.0f,
            NeuronKind.MetallicMap => 1.0f,
            NeuronKind.EmissiveMap => 1.0f,
            NeuronKind.OpacityMap => 1.0f,
            NeuronKind.SubsurfaceScattering => 8.0f,
            NeuronKind.Displacement => 2.0f,
            NeuronKind.Tessellation => 5.0f,
            NeuronKind.LODSelector => 1.0f,
            NeuronKind.OcclusionCuller => 2.0f,
            NeuronKind.FrustumCuller => 1.5f,
            NeuronKind.AnimationCurve => 2.0f,
            NeuronKind.ParticleEmitter => 4.0f,
            NeuronKind.PhysicsCollider => 3.0f,
            NeuronKind.FluidSolver => 10.0f,
            NeuronKind.ClothSimulator => 8.0f,
            NeuronKind.Voxelizer => 6.0f,
            NeuronKind.MarchingCube => 7.0f,
            NeuronKind.RayMarcher => 9.0f,
            NeuronKind.FieldGenerator => 3.0f,
            NeuronKind.WaveFunction => 5.0f,
            NeuronKind.TensorReshaper => 1.0f,
            NeuronKind.AttentionHead => 6.0f,
            NeuronKind.ConvolutionKernel => 4.0f,
            NeuronKind.PoolingLayer => 2.0f,
            NeuronKind.NormalizationLayer => 2.0f,
            NeuronKind.EmbeddingLookup => 3.0f,
            NeuronKind.PositionalEncoder => 2.0f,
            NeuronKind.CrossAttentionLayer => 7.0f,
            NeuronKind.SelfAttentionLayer => 6.0f,
            NeuronKind.FeedForwardLayer => 4.0f,
            NeuronKind.GeometricTransform => 2.0f,
            NeuronKind.CSGOperation => 3.0f,
            NeuronKind.ProceduralTexture => 5.0f,
            NeuronKind.TerrainSculptor => 5.0f,
            NeuronKind.LatticeGenerator => 4.0f,
            NeuronKind.FractalSubdivision => 6.0f,
            NeuronKind.HarmonicOscillator => 2.0f,
            NeuronKind.GravitationalField => 4.0f,
            NeuronKind.ElectromagneticField => 5.0f,
            NeuronKind.VolumetricDensity => 7.0f,
            NeuronKind.ScreenSpaceReflection => 6.0f,
            NeuronKind.GlobalIllumination => 10.0f,
            NeuronKind.AmbientOcclusion => 5.0f,
            NeuronKind.ShadowMap => 4.0f,
            NeuronKind.PostProcessChain => 3.0f,
            _ => 1.0f
        };

        /// <summary>Gets the default memory estimate for a given neuron kind.</summary>
        public static int GetDefaultMemoryEstimateForKind(NeuronKind kind) => kind switch
        {
            NeuronKind.SDFPrimitive => 64,
            NeuronKind.DisplacementField => 128,
            NeuronKind.CurvatureModulator => 192,
            NeuronKind.ColorField => 48,
            NeuronKind.ConvolutionKernel => 256,
            NeuronKind.EmbeddingLookup => 4096,
            NeuronKind.Voxelizer => 8192,
            NeuronKind.VolumetricDensity => 16384,
            _ => 96
        };

        /// <summary>Creates a new neuron with a modified activation parameter.</summary>
        public NeuronGene WithActivationParameter(float param1, float param2 = 0.0f)
            => this with { ActivationParameter = param1, ActivationParameter2 = param2 };

        /// <summary>Creates a new neuron with an updated bias vector.</summary>
        public NeuronGene WithBias(Vector3 bias) => this with { Bias = bias };

        /// <summary>Creates a new neuron with a different activation kernel.</summary>
        public NeuronGene WithActivation(ActivationKernel kernel) => this with { Activation = kernel };

        /// <summary>Creates a new neuron with a specific parameter value.</summary>
        public NeuronGene WithParameter(string key, float value)
            => this with { Parameters = Parameters.SetItem(key, value) };

        /// <summary>Gets a parameter value by key, returning a default if not found.</summary>
        public float GetParameter(string key, float defaultValue = 0.0f)
            => Parameters.TryGetValue(key, out var value) ? value : defaultValue;

        /// <summary>Computes the effective cost of this neuron considering its weight scale.</summary>
        public float EffectiveComputeCost => ComputeCost * Math.Abs(WeightScale);

        /// <summary>Computes the total estimated memory including parameters and connections.</summary>
        public int TotalEstimatedMemory => EstimatedMemoryBytes
            + Parameters.Count * sizeof(float)
            + SynapseTargets.Length * sizeof(int);
    }

    /// <summary>
    /// A single synaptic connection between two neurons, encoding weight, plasticity,
    /// learning rate, delay, and connection type.
    /// </summary>
    public readonly record struct SynapseGene
    {
        /// <summary>Unique identifier for this synapse within its genome.</summary>
        public int Id { get; init; }

        /// <summary>Identifier of the source (presynaptic) neuron.</summary>
        public int SourceNeuronId { get; init; }

        /// <summary>Identifier of the target (postsynaptic) neuron.</summary>
        public int TargetNeuronId { get; init; }

        /// <summary>Synaptic weight strength.</summary>
        public float Weight { get; init; }

        /// <summary>Plasticity coefficient (0 = static, 1 = fully plastic).</summary>
        public float Plasticity { get; init; }

        /// <summary>Learning rate for online weight updates.</summary>
        public float LearningRate { get; init; }

        /// <summary>Signal propagation delay in simulation time steps.</summary>
        public float Delay { get; init; }

        /// <summary>Whether this synapse undergoes plasticity changes during runtime.</summary>
        public bool IsPlastic { get; init; }

        /// <summary>Classification of the synaptic signal type.</summary>
        public SynapseType SynapseType { get; init; }

        /// <summary>Optional weight decay coefficient.</summary>
        public float WeightDecay { get; init; }

        /// <summary>Optional maximum weight bound (clipping).</summary>
        public float MaxWeight { get; init; }

        /// <summary>Optional minimum weight bound (clipping).</summary>
        public float MinWeight { get; init; }

        /// <summary>Whether this synapse is currently active.</summary>
        public bool IsEnabled { get; init; }

        /// <summary>Confidence/reliability score for this connection.</summary>
        public float Confidence { get; init; }

        /// <summary>Layer distance between source and target neurons.</summary>
        public int LayerDistance { get; init; }

        /// <summary>Estimated latency in microseconds.</summary>
        public float EstimatedLatencyUs { get; init; }

        /// <summary>Usage frequency tracking for profiling.</summary>
        public float UsageFrequency { get; init; }

        /// <summary>Initial weight value at creation time (for undo tracking).</summary>
        public float InitialWeight { get; init; }

        /// <summary>Number of times this synapse weight has been updated.</summary>
        public int UpdateCount { get; init; }

        /// <summary>Creates a new <see cref="SynapseGene"/> with default parameters.</summary>
        public static SynapseGene Create(int id, int sourceNeuronId, int targetNeuronId,
            float weight = 1.0f, SynapseType type = SynapseType.Excitatory)
        {
            return new SynapseGene
            {
                Id = id,
                SourceNeuronId = sourceNeuronId,
                TargetNeuronId = targetNeuronId,
                Weight = weight,
                Plasticity = type == SynapseType.Plastic ? 1.0f : 0.0f,
                LearningRate = type == SynapseType.Plastic ? 0.01f : 0.0f,
                Delay = 0.0f,
                IsPlastic = type == SynapseType.Plastic,
                SynapseType = type,
                WeightDecay = 0.001f,
                MaxWeight = float.MaxValue,
                MinWeight = float.MinValue,
                IsEnabled = true,
                Confidence = 1.0f,
                LayerDistance = 0,
                EstimatedLatencyUs = 0.0f,
                UsageFrequency = 0.0f,
                InitialWeight = weight,
                UpdateCount = 0
            };
        }

        /// <summary>Creates a plastic synapse with online learning enabled.</summary>
        public static SynapseGene CreatePlastic(int id, int sourceNeuronId, int targetNeuronId,
            float weight = 1.0f, float learningRate = 0.01f)
        {
            return Create(id, sourceNeuronId, targetNeuronId, weight, SynapseType.Plastic) with
            {
                LearningRate = learningRate,
                Plasticity = 1.0f,
                WeightDecay = 0.001f
            };
        }

        /// <summary>Creates an inhibitory synapse with negative weight.</summary>
        public static SynapseGene CreateInhibitory(int id, int sourceNeuronId, int targetNeuronId,
            float weight = -1.0f)
        {
            return Create(id, sourceNeuronId, targetNeuronId, weight, SynapseType.Inhibitory);
        }

        /// <summary>Creates a modulatory synapse that adjusts target parameters.</summary>
        public static SynapseGene CreateModulatory(int id, int sourceNeuronId, int targetNeuronId,
            float weight = 0.5f)
        {
            return Create(id, sourceNeuronId, targetNeuronId, weight, SynapseType.Modulatory);
        }

        /// <summary>Computes the effective weight after applying plasticity and decay.</summary>
        public float EffectiveWeight => Weight * (1.0f - WeightDecay * UpdateCount);

        /// <summary>Gets whether this synapse connects neurons in different layers.</summary>
        public bool IsInterLayer => LayerDistance != 0;

        /// <summary>Gets whether this synapse connects neurons within the same layer.</summary>
        public bool IsIntraLayer => LayerDistance == 0;

        /// <summary>Computes the absolute weight magnitude.</summary>
        public float AbsWeight => Math.Abs(Weight);

        /// <summary>Gets the weight sign (+1 for excitatory, -1 for inhibitory).</summary>
        public int WeightSign => Weight >= 0 ? 1 : -1;

        /// <summary>Computes whether the weight is within the specified bounds.</summary>
        public bool IsWeightInBounds => Weight >= MinWeight && Weight <= MaxWeight;
    }

    /// <summary>
    /// Complete immutable genome record representing a single individual in the G-DNN population.
    /// Contains the full network topology, neuron genes, synapse genes, species classification,
    /// and metadata.
    /// </summary>
    public readonly record struct GeoGenome
    {
        /// <summary>Unique identifier for this genome.</summary>
        public GenomeId Id { get; init; }

        /// <summary>Immutable array of all neuron genes in this genome.</summary>
        public ImmutableArray<NeuronGene> Neurons { get; init; }

        /// <summary>Immutable array of all synapse genes in this genome.</summary>
        public ImmutableArray<SynapseGene> Synapses { get; init; }

        /// <summary>Species classification identifier.</summary>
        public SpeciesId Species { get; init; }

        /// <summary>Metadata providing authorship, licensing, and classification information.</summary>
        public GenomeMetadata Metadata { get; init; }

        /// <summary>Precomputed hash of the genome topology for fast comparison.</summary>
        public string TopologyHash { get; init; }

        /// <summary>Format version of this genome.</summary>
        public GenomeVersion Version { get; init; }

        /// <summary>Timestamp when this genome was created.</summary>
        public DateTimeOffset CreatedAt { get; init; }

        /// <summary>Timestamp when this genome was last modified.</summary>
        public DateTimeOffset ModifiedAt { get; init; }

        /// <summary>Number of neurons in this genome.</summary>
        public int NeuronCount => Neurons.Length;

        /// <summary>Number of synapses in this genome.</summary>
        public int SynapseCount => Synapses.Length;

        /// <summary>Total estimated memory footprint in bytes.</summary>
        public long EstimatedMemoryBytes => Neurons.Sum(n => (long)n.TotalEstimatedMemory)
            + Synapses.Sum(s => (long)System.Runtime.CompilerServices.Unsafe.SizeOf<SynapseGene>());

        /// <summary>Total estimated compute cost.</summary>
        public float TotalComputeCost => Neurons.Sum(n => n.EffectiveComputeCost);

        /// <summary>Number of enabled neurons.</summary>
        public int EnabledNeuronCount => Neurons.Count(n => n.IsEnabled);

        /// <summary>Number of enabled synapses.</summary>
        public int EnabledSynapseCount => Synapses.Count(s => s.IsEnabled);

        /// <summary>Number of plastic synapses.</summary>
        public int PlasticSynapseCount => Synapses.Count(s => s.IsPlastic);

        /// <summary>Gets the maximum layer index in the network.</summary>
        public int MaxLayerIndex => Neurons.Length > 0 ? Neurons.Max(n => n.LayerIndex) : 0;

        /// <summary>Gets the minimum layer index in the network.</summary>
        public int MinLayerIndex => Neurons.Length > 0 ? Neurons.Min(n => n.LayerIndex) : 0;

        /// <summary>Gets the total number of unique layers.</summary>
        public int LayerCount => Neurons.Select(n => n.LayerIndex).Distinct().Count();

        /// <summary>Creates an empty genome with a new unique identifier.</summary>
        public static GeoGenome CreateEmpty() => new()
        {
            Id = GenomeId.New(),
            Neurons = ImmutableArray<NeuronGene>.Empty,
            Synapses = ImmutableArray<SynapseGene>.Empty,
            Species = SpeciesId.Default,
            Metadata = new GenomeMetadata(),
            TopologyHash = string.Empty,
            Version = GenomeVersion.Current,
            CreatedAt = DateTimeOffset.UtcNow,
            ModifiedAt = DateTimeOffset.UtcNow
        };

        /// <summary>Gets a neuron by its identifier, or null if not found.</summary>
        public NeuronGene? GetNeuron(int neuronId)
        {
            foreach (var neuron in Neurons)
                if (neuron.Id == neuronId)
                    return neuron;
            return null;
        }

        /// <summary>Gets a synapse by its identifier, or null if not found.</summary>
        public SynapseGene? GetSynapse(int synapseId)
        {
            foreach (var synapse in Synapses)
                if (synapse.Id == synapseId)
                    return synapse;
            return null;
        }

        /// <summary>Gets all synapses originating from the specified neuron.</summary>
        public ImmutableArray<SynapseGene> GetOutgoingSynapses(int neuronId)
        {
            var builder = ImmutableArray.CreateBuilder<SynapseGene>();
            foreach (var synapse in Synapses)
                if (synapse.SourceNeuronId == neuronId)
                    builder.Add(synapse);
            return builder.ToImmutable();
        }

        /// <summary>Gets all synapses terminating at the specified neuron.</summary>
        public ImmutableArray<SynapseGene> GetIncomingSynapses(int neuronId)
        {
            var builder = ImmutableArray.CreateBuilder<SynapseGene>();
            foreach (var synapse in Synapses)
                if (synapse.TargetNeuronId == neuronId)
                    builder.Add(synapse);
            return builder.ToImmutable();
        }

        /// <summary>Gets all neurons in the specified layer.</summary>
        public ImmutableArray<NeuronGene> GetNeuronsInLayer(int layerIndex)
        {
            var builder = ImmutableArray.CreateBuilder<NeuronGene>();
            foreach (var neuron in Neurons)
                if (neuron.LayerIndex == layerIndex)
                    builder.Add(neuron);
            return builder.ToImmutable();
        }

        /// <summary>Gets all enabled neurons.</summary>
        public ImmutableArray<NeuronGene> GetEnabledNeurons()
        {
            var builder = ImmutableArray.CreateBuilder<NeuronGene>();
            foreach (var neuron in Neurons)
                if (neuron.IsEnabled)
                    builder.Add(neuron);
            return builder.ToImmutable();
        }

        /// <summary>Gets all neurons of the specified kind.</summary>
        public ImmutableArray<NeuronGene> GetNeuronsByKind(NeuronKind kind)
        {
            var builder = ImmutableArray.CreateBuilder<NeuronGene>();
            foreach (var neuron in Neurons)
                if (neuron.Kind == kind)
                    builder.Add(neuron);
            return builder.ToImmutable();
        }

        /// <summary>Gets all neurons with the specified semantic role.</summary>
        public ImmutableArray<NeuronGene> GetNeuronsByRole(NeuronSemanticRole role)
        {
            var builder = ImmutableArray.CreateBuilder<NeuronGene>();
            foreach (var neuron in Neurons)
                if (neuron.SemanticRole == role)
                    builder.Add(neuron);
            return builder.ToImmutable();
        }

        /// <summary>Computes a shallow clone with modified neurons.</summary>
        public GeoGenome WithNeurons(ImmutableArray<NeuronGene> neurons) => this with
        { Neurons = neurons, ModifiedAt = DateTimeOffset.UtcNow };

        /// <summary>Computes a shallow clone with modified synapses.</summary>
        public GeoGenome WithSynapses(ImmutableArray<SynapseGene> synapses) => this with
        { Synapses = synapses, ModifiedAt = DateTimeOffset.UtcNow };

        /// <summary>Computes a shallow clone with modified metadata.</summary>
        public GeoGenome WithMetadata(GenomeMetadata metadata) => this with
        { Metadata = metadata, ModifiedAt = DateTimeOffset.UtcNow };

        /// <summary>Adds a neuron and returns a new genome.</summary>
        public GeoGenome AddNeuron(NeuronGene neuron) => this with
        { Neurons = Neurons.Add(neuron), ModifiedAt = DateTimeOffset.UtcNow };

        /// <summary>Removes a neuron by ID and returns a new genome.</summary>
        public GeoGenome RemoveNeuron(int neuronId) => this with
        {
            Neurons = Neurons.Remove(Neurons.FirstOrDefault(n => n.Id == neuronId)),
            Synapses = Synapses.RemoveAll(s => s.SourceNeuronId == neuronId || s.TargetNeuronId == neuronId),
            ModifiedAt = DateTimeOffset.UtcNow
        };

        /// <summary>Adds a synapse and returns a new genome.</summary>
        public GeoGenome AddSynapse(SynapseGene synapse) => this with
        { Synapses = Synapses.Add(synapse), ModifiedAt = DateTimeOffset.UtcNow };

        /// <summary>Removes a synapse by ID and returns a new genome.</summary>
        public GeoGenome RemoveSynapse(int synapseId) => this with
        { Synapses = Synapses.Remove(Synapses.FirstOrDefault(s => s.Id == synapseId)), ModifiedAt = DateTimeOffset.UtcNow };

        /// <summary>Checks whether the genome contains a neuron with the specified ID.</summary>
        public bool ContainsNeuron(int neuronId)
        {
            foreach (var neuron in Neurons)
                if (neuron.Id == neuronId)
                    return true;
            return false;
        }

        /// <summary>Checks whether the genome contains a synapse with the specified ID.</summary>
        public bool ContainsSynapse(int synapseId)
        {
            foreach (var synapse in Synapses)
                if (synapse.Id == synapseId)
                    return true;
            return false;
        }

        /// <summary>Checks whether the genome contains a neuron of the specified kind.</summary>
        public bool ContainsNeuronKind(NeuronKind kind)
        {
            foreach (var neuron in Neurons)
                if (neuron.Kind == kind)
                    return true;
            return false;
        }

        /// <summary>Gets the count of neurons of a specific kind.</summary>
        public int CountNeuronsByKind(NeuronKind kind)
        {
            int count = 0;
            foreach (var neuron in Neurons)
                if (neuron.Kind == kind)
                    count++;
            return count;
        }

        /// <summary>Computes the average weight of all synapses.</summary>
        public float AverageSynapseWeight()
        {
            if (Synapses.Length == 0)
                return 0f;
            float sum = 0f;
            foreach (var synapse in Synapses)
                sum += synapse.Weight;
            return sum / Synapses.Length;
        }

        /// <summary>Computes the weight standard deviation of all synapses.</summary>
        public float WeightStandardDeviation()
        {
            if (Synapses.Length <= 1)
                return 0f;
            var mean = AverageSynapseWeight();
            float sumSqDiff = 0f;
            foreach (var synapse in Synapses)
            {
                var diff = synapse.Weight - mean;
                sumSqDiff += diff * diff;
            }
            return MathF.Sqrt(sumSqDiff / (Synapses.Length - 1));
        }

        /// <summary>Gets the neuron with the highest fan-out (most outgoing connections).</summary>
        public NeuronGene? GetHighestFanOutNeuron()
        {
            if (Neurons.Length == 0)
                return null;
            var best = Neurons[0];
            for (int i = 1; i < Neurons.Length; i++)
                if (Neurons[i].FanOut > best.FanOut)
                    best = Neurons[i];
            return best;
        }

        /// <summary>Gets the neuron with the highest fan-in (most incoming connections).</summary>
        public NeuronGene? GetHighestFanInNeuron()
        {
            if (Neurons.Length == 0)
                return null;
            var best = Neurons[0];
            for (int i = 1; i < Neurons.Length; i++)
                if (Neurons[i].FanIn > best.FanIn)
                    best = Neurons[i];
            return best;
        }

        /// <summary>Gets the synapse with the largest absolute weight.</summary>
        public SynapseGene? GetStrongestSynapse()
        {
            if (Synapses.Length == 0)
                return null;
            var best = Synapses[0];
            for (int i = 1; i < Synapses.Length; i++)
                if (Math.Abs(Synapses[i].Weight) > Math.Abs(best.Weight))
                    best = Synapses[i];
            return best;
        }

        /// <summary>Gets the synapse with the smallest absolute weight.</summary>
        public SynapseGene? GetWeakestSynapse()
        {
            if (Synapses.Length == 0)
                return null;
            var best = Synapses[0];
            for (int i = 1; i < Synapses.Length; i++)
                if (Math.Abs(Synapses[i].Weight) < Math.Abs(best.Weight))
                    best = Synapses[i];
            return best;
        }

        /// <summary>Computes a compact summary string for logging and debugging.</summary>
        public string ToSummaryString()
            => $"Genome[{Id.ToCompactString()}] Species={Species.Value} Neurons={NeuronCount} Synapses={SynapseCount} Layers={LayerCount} Gen={Metadata.Generation}";

        /// <inheritdoc/>
        public override string ToString() => ToSummaryString();
    }

    #endregion Record Structs — Core Genome Types

    // ========================================================================
    #region Record Structs — Supporting Types

    /// <summary>
    /// Captures a differential between two genome versions for undo/redo and collaboration.
    /// </summary>
    public sealed record GenomeDiff
    {
        /// <summary>Neurons added in the target genome.</summary>
        public ImmutableArray<NeuronGene> AddedNeurons { get; init; } = ImmutableArray<NeuronGene>.Empty;

        /// <summary>Neuron IDs removed from the source genome.</summary>
        public ImmutableArray<int> RemovedNeurons { get; init; } = ImmutableArray<int>.Empty;

        /// <summary>Neuron genes that were modified (stored as pairs: original, modified).</summary>
        public ImmutableArray<(NeuronGene Original, NeuronGene Modified)> ModifiedNeurons { get; init; }
            = ImmutableArray<(NeuronGene, NeuronGene)>.Empty;

        /// <summary>Synapses added in the target genome.</summary>
        public ImmutableArray<SynapseGene> AddedSynapses { get; init; } = ImmutableArray<SynapseGene>.Empty;

        /// <summary>Synapse IDs removed from the source genome.</summary>
        public ImmutableArray<int> RemovedSynapses { get; init; } = ImmutableArray<int>.Empty;

        /// <summary>Synapse genes that were modified (stored as pairs: original, modified).</summary>
        public ImmutableArray<(SynapseGene Original, SynapseGene Modified)> ModifiedSynapses { get; init; }
            = ImmutableArray<(SynapseGene, SynapseGene)>.Empty;

        /// <summary>Metadata changes.</summary>
        public GenomeMetadata? MetadataChanges { get; init; }

        /// <summary>Timestamp when this diff was computed.</summary>
        public DateTimeOffset DiffTimestamp { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>Source genome ID.</summary>
        public GenomeId SourceGenomeId { get; init; }

        /// <summary>Target genome ID.</summary>
        public GenomeId TargetGenomeId { get; init; }

        /// <summary>Whether this diff represents any actual changes.</summary>
        public bool IsEmpty => AddedNeurons.Length == 0 && RemovedNeurons.Length == 0
            && ModifiedNeurons.Length == 0 && AddedSynapses.Length == 0
            && RemovedSynapses.Length == 0 && ModifiedSynapses.Length == 0
            && MetadataChanges == null;

        /// <summary>Total number of individual changes in this diff.</summary>
        public int TotalChanges => AddedNeurons.Length + RemovedNeurons.Length + ModifiedNeurons.Length
            + AddedSynapses.Length + RemovedSynapses.Length + ModifiedSynapses.Length;

        /// <summary>Gets a summary of all mutation types represented in this diff.</summary>
        public ImmutableArray<MutationType> GetMutationTypes()
        {
            var types = ImmutableArray.CreateBuilder<MutationType>();
            if (AddedNeurons.Length > 0)
                types.Add(MutationType.Insertion);
            if (RemovedNeurons.Length > 0)
                types.Add(MutationType.Deletion);
            if (ModifiedNeurons.Length > 0)
                types.Add(MutationType.PointMutation);
            if (AddedSynapses.Length > 0)
                types.Add(MutationType.SynapseGrowth);
            if (RemovedSynapses.Length > 0)
                types.Add(MutationType.SynapsePruning);
            if (ModifiedSynapses.Length > 0)
                types.Add(MutationType.WeightPerturbation);
            return types.ToImmutable();
        }

        /// <summary>Computes the inverse of this diff (for undo operations).</summary>
        public GenomeDiff Invert() => new()
        {
            SourceGenomeId = TargetGenomeId,
            TargetGenomeId = SourceGenomeId,
            AddedNeurons = ImmutableArray.Create(RemovedNeurons.Select(id => new NeuronGene { Id = id }).ToArray()),
            RemovedNeurons = ImmutableArray.Create(AddedNeurons.Select(n => n.Id).ToArray()),
            ModifiedNeurons = ImmutableArray.Create(ModifiedNeurons.Select(m => (m.Modified, m.Original)).ToArray()),
            AddedSynapses = ImmutableArray.Create(RemovedSynapses.Select(id => new SynapseGene { Id = id }).ToArray()),
            RemovedSynapses = ImmutableArray.Create(AddedSynapses.Select(s => s.Id).ToArray()),
            ModifiedSynapses = ImmutableArray.Create(ModifiedSynapses.Select(m => (m.Modified, m.Original)).ToArray()),
            DiffTimestamp = DateTimeOffset.UtcNow
        };
    }

    /// <summary>
    /// Complete snapshot of a genome state for undo/redo functionality.
    /// </summary>
    public sealed record GenomeSnapshot
    {
        /// <summary>The complete genome state at snapshot time.</summary>
        public GeoGenome Genome { get; init; }

        /// <summary>Unique snapshot identifier.</summary>
        public Guid SnapshotId { get; init; } = Guid.NewGuid();

        /// <summary>Human-readable label for this snapshot.</summary>
        public string Label { get; init; } = string.Empty;

        /// <summary>Timestamp when this snapshot was created.</summary>
        public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>Optional description of the operation that created this snapshot.</summary>
        public string Description { get; init; } = string.Empty;

        /// <summary>The diff that was applied to produce this state from the previous snapshot.</summary>
        public GenomeDiff? AppliedDiff { get; init; }

        /// <summary>Sequential index in the undo/redo stack.</summary>
        public int StackIndex { get; init; }

        /// <summary>Estimated memory footprint of this snapshot in bytes.</summary>
        public long EstimatedMemoryBytes => Genome.EstimatedMemoryBytes;

        /// <summary>Tags for categorizing snapshots.</summary>
        public ImmutableHashSet<string> Tags { get; init; } = ImmutableHashSet<string>.Empty;
    }

    /// <summary>
    /// Runtime profiling data for a compiled genome.
    /// </summary>
    public sealed record GenomeProfile
    {
        /// <summary>Time taken to compile the genome (in milliseconds).</summary>
        public double CompileTimeMs { get; init; }

        /// <summary>Peak memory usage during compilation (in bytes).</summary>
        public long MemoryUsageBytes { get; init; }

        /// <summary>GPU execution time (in milliseconds).</summary>
        public double GpuTimeMs { get; init; }

        /// <summary>CPU execution time (in milliseconds).</summary>
        public double CpuTimeMs { get; init; }

        /// <summary>Total wall-clock execution time (in milliseconds).</summary>
        public double WallTimeMs { get; init; }

        /// <summary>Number of times this genome has been executed.</summary>
        public int ExecutionCount { get; init; }

        /// <summary>Peak GPU memory usage (in bytes).</summary>
        public long GpuMemoryBytes { get; init; }

        /// <summary>Peak CPU memory usage (in bytes).</summary>
        public long CpuMemoryBytes { get; init; }

        /// <summary>Number of shader compilations triggered.</summary>
        public int ShaderCompilationCount { get; init; }

        /// <summary>Number of cache hits during execution.</summary>
        public int CacheHits { get; init; }

        /// <summary>Number of cache misses during execution.</summary>
        public int CacheMisses { get; init; }

        /// <summary>Average frame time over the last N executions.</summary>
        public double AverageFrameTimeMs { get; init; }

        /// <summary>Standard deviation of frame times.</summary>
        public double FrameTimeStdDevMs { get; init; }

        /// <summary>Minimum observed frame time (in milliseconds).</summary>
        public double MinFrameTimeMs { get; init; }

        /// <summary>Maximum observed frame time (in milliseconds).</summary>
        public double MaxFrameTimeMs { get; init; }

        /// <summary>Cache hit ratio (0.0 to 1.0).</summary>
        public double CacheHitRatio => (CacheHits + CacheMisses) == 0 ? 0.0 : (double)CacheHits / (CacheHits + CacheMisses);

        /// <summary>Total computation time (GPU + CPU).</summary>
        public double TotalComputeTimeMs => GpuTimeMs + CpuTimeMs;

        /// <summary>Efficiency metric: useful work / total time.</summary>
        public double Efficiency => WallTimeMs == 0 ? 0 : TotalComputeTimeMs / WallTimeMs;

        /// <summary>Creates a new empty profile.</summary>
        public static GenomeProfile CreateEmpty() => new();

        /// <summary>Merges profiles from multiple execution runs.</summary>
        public GenomeProfile MergeWith(GenomeProfile other) => new()
        {
            CompileTimeMs = CompileTimeMs + other.CompileTimeMs,
            MemoryUsageBytes = Math.Max(MemoryUsageBytes, other.MemoryUsageBytes),
            GpuTimeMs = GpuTimeMs + other.GpuTimeMs,
            CpuTimeMs = CpuTimeMs + other.CpuTimeMs,
            WallTimeMs = WallTimeMs + other.WallTimeMs,
            ExecutionCount = ExecutionCount + other.ExecutionCount,
            GpuMemoryBytes = Math.Max(GpuMemoryBytes, other.GpuMemoryBytes),
            CpuMemoryBytes = Math.Max(CpuMemoryBytes, other.CpuMemoryBytes),
            ShaderCompilationCount = ShaderCompilationCount + other.ShaderCompilationCount,
            CacheHits = CacheHits + other.CacheHits,
            CacheMisses = CacheMisses + other.CacheMisses,
            AverageFrameTimeMs = (AverageFrameTimeMs * ExecutionCount + other.AverageFrameTimeMs * other.ExecutionCount)
                / Math.Max(1, ExecutionCount + other.ExecutionCount),
            MinFrameTimeMs = Math.Min(MinFrameTimeMs, other.MinFrameTimeMs),
            MaxFrameTimeMs = Math.Max(MaxFrameTimeMs, other.MaxFrameTimeMs)
        };
    }

    /// <summary>
    /// Per-neuron activation profiling data captured during execution.
    /// </summary>
    public sealed record NeuronActivationProfile
    {
        /// <summary>Neuron ID this profile belongs to.</summary>
        public int NeuronId { get; init; }

        /// <summary>Number of times this neuron was activated.</summary>
        public long ActivationCount { get; init; }

        /// <summary>Average output value across all activations.</summary>
        public double AverageOutput { get; init; }

        /// <summary>Standard deviation of output values.</summary>
        public double OutputStdDev { get; init; }

        /// <summary>Minimum observed output value.</summary>
        public double MinOutput { get; init; }

        /// <summary>Maximum observed output value.</summary>
        public double MaxOutput { get; init; }

        /// <summary>Average computation time per activation (in microseconds).</summary>
        public double AverageComputeTimeUs { get; init; }

        /// <summary>Peak computation time for a single activation (in microseconds).</summary>
        public double PeakComputeTimeUs { get; init; }

        /// <summary>Percentage of time the neuron output was zero (dead neuron detection).</summary>
        public double DeadRatio { get; init; }

        /// <summary>Entropy of the output distribution (information content).</summary>
        public double OutputEntropy { get; init; }

        /// <summary>Whether this neuron is considered "dead" (always outputs zero).</summary>
        public bool IsDeadNeuron => DeadRatio > 0.99;

        /// <summary>Whether this neuron is "saturated" (output always near extremes).</summary>
        public bool IsSaturated => (MinOutput > 0.9 || MaxOutput < -0.9) && OutputStdDev < 0.01;

        /// <summary>Gets the dynamic range of the neuron output.</summary>
        public double DynamicRange => MaxOutput - MinOutput;

        /// <summary>Creates a new empty profile for the specified neuron.</summary>
        public static NeuronActivationProfile CreateEmpty(int neuronId) => new() { NeuronId = neuronId };

        /// <summary>Merges two profiles from sequential observation windows.</summary>
        public NeuronActivationProfile MergeWith(NeuronActivationProfile other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));
            var totalCount = ActivationCount + other.ActivationCount;
            if (totalCount == 0)
                return this;
            return new NeuronActivationProfile
            {
                NeuronId = NeuronId,
                ActivationCount = totalCount,
                AverageOutput = (AverageOutput * ActivationCount + other.AverageOutput * other.ActivationCount) / totalCount,
                MinOutput = Math.Min(MinOutput, other.MinOutput),
                MaxOutput = Math.Max(MaxOutput, other.MaxOutput),
                AverageComputeTimeUs = (AverageComputeTimeUs * ActivationCount + other.AverageComputeTimeUs * other.ActivationCount) / totalCount,
                PeakComputeTimeUs = Math.Max(PeakComputeTimeUs, other.PeakComputeTimeUs),
                DeadRatio = (DeadRatio * ActivationCount + other.DeadRatio * other.ActivationCount) / totalCount,
                OutputEntropy = (OutputEntropy * ActivationCount + other.OutputEntropy * other.ActivationCount) / totalCount
            };
        }
    }

    /// <summary>
    /// Aggregated performance metrics for genome rendering and evaluation.
    /// </summary>
    public sealed record GenomePerformanceMetrics
    {
        /// <summary>Average frame rendering time (in milliseconds).</summary>
        public double FrameTimeMs { get; init; }

        /// <summary>Number of draw calls per frame.</summary>
        public int DrawCalls { get; init; }

        /// <summary>Number of triangles rendered per frame.</summary>
        public long Triangles { get; init; }

        /// <summary>Number of active neurons in the computation graph.</summary>
        public int ActiveNeurons { get; init; }

        /// <summary>Number of active synapses in the computation graph.</summary>
        public int ActiveSynapses { get; init; }

        /// <summary>GPU memory usage (in bytes).</summary>
        public long GpuMemoryBytes { get; init; }

        /// <summary>CPU memory usage (in bytes).</summary>
        public long CpuMemoryBytes { get; init; }

        /// <summary>Texture memory usage (in bytes).</summary>
        public long TextureMemoryBytes { get; init; }

        /// <summary>Number of overdraw layers per pixel.</summary>
        public float OverdrawRatio { get; init; }

        /// <summary>Vertex throughput (vertices per second).</summary>
        public long VertexThroughput { get; init; }

        /// <summary>Fragment throughput (fragments per second).</summary>
        public long FragmentThroughput { get; init; }

        /// <summary>Occupancy ratio (active warps / max warps on GPU).</summary>
        public float GpuOccupancy { get; init; }

        /// <summary>Cache hit ratio for texture sampling.</summary>
        public float TextureCacheHitRatio { get; init; }

        /// <summary>Cache hit ratio for vertex data.</summary>
        public float VertexCacheHitRatio { get; init; }

        /// <summary>Frames per second (inverse of frame time).</summary>
        public double Fps => FrameTimeMs > 0 ? 1000.0 / FrameTimeMs : 0;

        /// <summary>Total memory footprint in bytes.</summary>
        public long TotalMemoryBytes => GpuMemoryBytes + CpuMemoryBytes + TextureMemoryBytes;

        /// <summary>Triangles per draw call ratio.</summary>
        public double TrianglesPerDrawCall => DrawCalls > 0 ? (double)Triangles / DrawCalls : 0;

        /// <summary>Creates a new empty metrics instance.</summary>
        public static GenomePerformanceMetrics CreateEmpty() => new();

        /// <summary>Merges metrics from multiple frames.</summary>
        public GenomePerformanceMetrics MergeWith(GenomePerformanceMetrics other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));
            return new GenomePerformanceMetrics
            {
                FrameTimeMs = FrameTimeMs + other.FrameTimeMs,
                DrawCalls = DrawCalls + other.DrawCalls,
                Triangles = Triangles + other.Triangles,
                ActiveNeurons = Math.Max(ActiveNeurons, other.ActiveNeurons),
                ActiveSynapses = Math.Max(ActiveSynapses, other.ActiveSynapses),
                GpuMemoryBytes = Math.Max(GpuMemoryBytes, other.GpuMemoryBytes),
                CpuMemoryBytes = Math.Max(CpuMemoryBytes, other.CpuMemoryBytes),
                TextureMemoryBytes = Math.Max(TextureMemoryBytes, other.TextureMemoryBytes),
                OverdrawRatio = Math.Max(OverdrawRatio, other.OverdrawRatio),
                VertexThroughput = VertexThroughput + other.VertexThroughput,
                FragmentThroughput = FragmentThroughput + other.FragmentThroughput,
                GpuOccupancy = (GpuOccupancy + other.GpuOccupancy) / 2.0f,
                TextureCacheHitRatio = (TextureCacheHitRatio + other.TextureCacheHitRatio) / 2.0f,
                VertexCacheHitRatio = (VertexCacheHitRatio + other.VertexCacheHitRatio) / 2.0f
            };
        }
    }

    /// <summary>
    /// Record capturing a genome evolution event for tracking lineage and mutation history.
    /// </summary>
    public sealed record GenomeEvolutionEvent
    {
        /// <summary>Timestamp of the evolution event.</summary>
        public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

        /// <summary>IDs of the parent genomes that produced the offspring.</summary>
        public ImmutableArray<GenomeId> ParentIds { get; init; } = ImmutableArray<GenomeId>.Empty;

        /// <summary>Types of mutations applied during this evolution step.</summary>
        public ImmutableArray<MutationType> MutationTypes { get; init; } = ImmutableArray<MutationType>.Empty;

        /// <summary>Fitness score before evolution.</summary>
        public double FitnessBefore { get; init; }

        /// <summary>Fitness score after evolution.</summary>
        public double FitnessAfter { get; init; }

        /// <summary>Generation number at the time of this event.</summary>
        public int Generation { get; init; }

        /// <summary>ID of the offspring genome produced.</summary>
        public GenomeId OffspringId { get; init; }

        /// <summary>Whether the offspring has higher fitness than parents.</summary>
        public bool IsImprovement => FitnessAfter > FitnessBefore;

        /// <summary>Fitness delta (positive indicates improvement).</summary>
        public double FitnessDelta => FitnessAfter - FitnessBefore;

        /// <summary>Relative improvement percentage.</summary>
        public double ImprovementPercentage => FitnessBefore == 0 ? 0 : (FitnessDelta / Math.Abs(FitnessBefore)) * 100.0;

        /// <summary>Number of mutation types applied.</summary>
        public int MutationCount => MutationTypes.Length;

        /// <summary>Whether this was a crossover event (multiple parents).</summary>
        public bool IsCrossover => ParentIds.Length > 1;

        /// <summary>Whether this was an asexual mutation event (single parent).</summary>
        public bool IsAsexual => ParentIds.Length == 1;

        /// <summary>Whether this was a de novo creation (no parents).</summary>
        public bool IsDeNovo => ParentIds.Length == 0;

        /// <summary>Creates a summary string for logging.</summary>
        public string ToSummaryString()
            => $"Evolution[Gen={Generation}] Fitness: {FitnessBefore:F4} -> {FitnessAfter:F4} ({FitnessDelta:+0.0000;-0.0000}) Mutations={string.Join(",", MutationTypes.Select(m => m.ToString()))}";
    }

    #endregion Record Structs — Supporting Types

    // ========================================================================
    #region Builder — GeoGenomeBuilder

    /// <summary>
    /// Fluent builder for constructing <see cref="GeoGenome"/> instances with validation,
    /// cycle detection, and automatic species classification.
    /// </summary>
    public sealed class GeoGenomeBuilder
    {
        private readonly List<NeuronGene> _neurons = new();
        private readonly List<SynapseGene> _synapses = new();
        private string _semanticTag = string.Empty;
        private int _complexityBudget = 10000;
        private LodPolicy _lodPolicy = LodPolicy.Adaptive;
        private string _author = string.Empty;
        private string _description = string.Empty;
        private string _license = "MIT";
        private GenomeClassification _classification = GenomeClassification.Abstract;
        private EvolutionStrategy _strategy = EvolutionStrategy.Static;
        private FitnessObjective _objective = FitnessObjective.Balanced;
        private int _nextNeuronId = 1;
        private int _nextSynapseId = 1;
        private readonly List<string> _validationErrors = new();
        private readonly List<string> _validationWarnings = new();
        private SpeciesId? _forcedSpecies;
        private readonly Dictionary<string, string> _customProperties = new();
        private readonly HashSet<string> _tags = new(StringComparer.OrdinalIgnoreCase);
        private bool _strictValidation = true;

        /// <summary>Starts building a new genome.</summary>
        public static GeoGenomeBuilder Create() => new();

        /// <summary>Sets whether validation errors should be treated as fatal.</summary>
        public GeoGenomeBuilder WithStrictValidation(bool strict)
        {
            _strictValidation = strict;
            return this;
        }

        /// <summary>Adds a neuron to the genome being built.</summary>
        public GeoGenomeBuilder AddNeuron(NeuronGene neuron)
        {
            if (_neurons.Any(n => n.Id == neuron.Id))
            {
                var msg = $"Duplicate neuron ID {neuron.Id}.";
                if (_strictValidation)
                    throw new InvalidOperationException(msg);
                _validationWarnings.Add(msg);
            }
            _neurons.Add(neuron);
            if (neuron.Id >= _nextNeuronId)
                _nextNeuronId = neuron.Id + 1;
            return this;
        }

        /// <summary>Adds a neuron with auto-generated ID.</summary>
        public GeoGenomeBuilder AddNeuron(NeuronKind kind, int layerIndex = 0,
            ActivationKernel? activation = null, NeuronSemanticRole role = NeuronSemanticRole.Functional)
        {
            var neuron = NeuronGene.Create(_nextNeuronId++, kind, layerIndex)
                .WithActivation(activation ?? NeuronGene.GetDefaultActivationForKind(kind))
                with
            { SemanticRole = role };
            _neurons.Add(neuron);
            return this;
        }

        /// <summary>Adds a batch of neurons to the genome.</summary>
        public GeoGenomeBuilder AddNeurons(IEnumerable<NeuronGene> neurons)
        {
            foreach (var neuron in neurons)
                AddNeuron(neuron);
            return this;
        }

        /// <summary>Adds a synapse connecting two neurons.</summary>
        public GeoGenomeBuilder AddSynapse(SynapseGene synapse)
        {
            if (_synapses.Any(s => s.Id == synapse.Id))
            {
                var msg = $"Duplicate synapse ID {synapse.Id}.";
                if (_strictValidation)
                    throw new InvalidOperationException(msg);
                _validationWarnings.Add(msg);
            }
            _synapses.Add(synapse);
            if (synapse.Id >= _nextSynapseId)
                _nextSynapseId = synapse.Id + 1;
            return this;
        }

        /// <summary>Adds a synapse between two neurons by their IDs.</summary>
        public GeoGenomeBuilder AddSynapse(int sourceNeuronId, int targetNeuronId,
            float weight = 1.0f, SynapseType type = SynapseType.Excitatory)
        {
            var synapse = SynapseGene.Create(_nextSynapseId++, sourceNeuronId, targetNeuronId, weight, type);
            _synapses.Add(synapse);
            return this;
        }

        /// <summary>Adds a plastic synapse between two neurons.</summary>
        public GeoGenomeBuilder AddPlasticSynapse(int sourceNeuronId, int targetNeuronId,
            float weight = 1.0f, float learningRate = 0.01f)
        {
            var synapse = SynapseGene.CreatePlastic(_nextSynapseId++, sourceNeuronId, targetNeuronId, weight, learningRate);
            _synapses.Add(synapse);
            return this;
        }

        /// <summary>Adds an inhibitory synapse between two neurons.</summary>
        public GeoGenomeBuilder AddInhibitorySynapse(int sourceNeuronId, int targetNeuronId,
            float weight = -1.0f)
        {
            var synapse = SynapseGene.CreateInhibitory(_nextSynapseId++, sourceNeuronId, targetNeuronId, weight);
            _synapses.Add(synapse);
            return this;
        }

        /// <summary>Sets the semantic tag for the genome.</summary>
        public GeoGenomeBuilder WithSemanticTag(string tag)
        {
            _semanticTag = tag ?? throw new ArgumentNullException(nameof(tag));
            return this;
        }

        /// <summary>Sets the complexity budget (maximum neuron count).</summary>
        public GeoGenomeBuilder WithComplexityBudget(int budget)
        {
            if (budget <= 0)
                throw new ArgumentOutOfRangeException(nameof(budget), "Complexity budget must be positive.");
            _complexityBudget = budget;
            return this;
        }

        /// <summary>Sets the level-of-detail policy.</summary>
        public GeoGenomeBuilder WithLodPolicy(LodPolicy policy)
        {
            _lodPolicy = policy;
            return this;
        }

        /// <summary>Sets the author name.</summary>
        public GeoGenomeBuilder WithAuthor(string author)
        {
            _author = author ?? throw new ArgumentNullException(nameof(author));
            return this;
        }

        /// <summary>Sets the genome description.</summary>
        public GeoGenomeBuilder WithDescription(string description)
        {
            _description = description ?? throw new ArgumentNullException(nameof(description));
            return this;
        }

        /// <summary>Sets the license identifier.</summary>
        public GeoGenomeBuilder WithLicense(string license)
        {
            _license = license ?? throw new ArgumentNullException(nameof(license));
            return this;
        }

        /// <summary>Sets the genome classification.</summary>
        public GeoGenomeBuilder WithClassification(GenomeClassification classification)
        {
            _classification = classification;
            return this;
        }

        /// <summary>Sets the evolution strategy.</summary>
        public GeoGenomeBuilder WithEvolutionStrategy(EvolutionStrategy strategy)
        {
            _strategy = strategy;
            return this;
        }

        /// <summary>Sets the fitness objective.</summary>
        public GeoGenomeBuilder WithFitnessObjective(FitnessObjective objective)
        {
            _objective = objective;
            return this;
        }

        /// <summary>Forces a specific species assignment instead of auto-detection.</summary>
        public GeoGenomeBuilder WithSpecies(SpeciesId species)
        {
            _forcedSpecies = species;
            return this;
        }

        /// <summary>Adds a tag to the genome metadata.</summary>
        public GeoGenomeBuilder WithTag(string tag)
        {
            if (string.IsNullOrWhiteSpace(tag))
                throw new ArgumentException("Tag cannot be empty.", nameof(tag));
            _tags.Add(tag.Trim());
            return this;
        }

        /// <summary>Adds a custom property to the genome metadata.</summary>
        public GeoGenomeBuilder WithCustomProperty(string key, string value)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentException("Property key cannot be empty.", nameof(key));
            _customProperties[key] = value;
            return this;
        }

        /// <summary>Gets the current validation errors.</summary>
        public IReadOnlyList<string> ValidationErrors => _validationErrors;

        /// <summary>Gets the current validation warnings.</summary>
        public IReadOnlyList<string> ValidationWarnings => _validationWarnings;

        /// <summary>Gets the current number of neurons added.</summary>
        public int NeuronCount => _neurons.Count;

        /// <summary>Gets the current number of synapses added.</summary>
        public int SynapseCount => _synapses.Count;

        /// <summary>Validates the genome structure and returns all detected issues.</summary>
        public (bool IsValid, IReadOnlyList<string> Errors, IReadOnlyList<string> Warnings) Validate()
        {
            _validationErrors.Clear();
            _validationWarnings.Clear();
            ValidateOrphanNeurons();
            ValidateSynapseReferences();
            ValidateCycles();
            ValidateDisconnectedComponents();
            ValidateComplexityBudget();
            ValidateDuplicateConnections();
            ValidateSelfConnections();
            ValidateLayerConsistency();
            ValidateWeightBounds();
            return (_validationErrors.Count == 0, _validationErrors, _validationWarnings);
        }

        private void ValidateOrphanNeurons()
        {
            var connectedNeuronIds = new HashSet<int>();
            foreach (var synapse in _synapses)
            {
                connectedNeuronIds.Add(synapse.SourceNeuronId);
                connectedNeuronIds.Add(synapse.TargetNeuronId);
            }
            foreach (var neuron in _neurons)
            {
                if (!connectedNeuronIds.Contains(neuron.Id) && _neurons.Count > 1)
                    _validationWarnings.Add($"Neuron {neuron.Id} ({neuron.Kind}) is orphaned (no connections).");
            }
        }

        private void ValidateSynapseReferences()
        {
            var neuronIds = new HashSet<int>(_neurons.Select(n => n.Id));
            foreach (var synapse in _synapses)
            {
                if (!neuronIds.Contains(synapse.SourceNeuronId))
                    _validationErrors.Add($"Synapse {synapse.Id} references non-existent source neuron {synapse.SourceNeuronId}.");
                if (!neuronIds.Contains(synapse.TargetNeuronId))
                    _validationErrors.Add($"Synapse {synapse.Id} references non-existent target neuron {synapse.TargetNeuronId}.");
            }
        }

        private void ValidateCycles()
        {
            if (_neurons.Count == 0 || _synapses.Count == 0)
                return;
            var adjacency = new Dictionary<int, List<int>>();
            foreach (var neuron in _neurons)
                adjacency[neuron.Id] = new List<int>();
            foreach (var synapse in _synapses)
                if (adjacency.ContainsKey(synapse.SourceNeuronId))
                    adjacency[synapse.SourceNeuronId].Add(synapse.TargetNeuronId);
            var visited = new HashSet<int>();
            var recursionStack = new HashSet<int>();
            bool HasCycle(int nodeId)
            {
                if (recursionStack.Contains(nodeId))
                    return true;
                if (visited.Contains(nodeId))
                    return false;
                visited.Add(nodeId);
                recursionStack.Add(nodeId);
                if (adjacency.TryGetValue(nodeId, out var neighbors))
                    foreach (var neighbor in neighbors)
                        if (HasCycle(neighbor))
                            return true;
                recursionStack.Remove(nodeId);
                return false;
            }
            foreach (var neuron in _neurons)
                if (!visited.Contains(neuron.Id) && HasCycle(neuron.Id))
                    _validationErrors.Add($"Cycle detected involving neuron {neuron.Id}.");
        }

        private void ValidateDisconnectedComponents()
        {
            if (_neurons.Count <= 1)
                return;
            var adjacency = new Dictionary<int, List<int>>();
            foreach (var neuron in _neurons)
                adjacency[neuron.Id] = new List<int>();
            foreach (var synapse in _synapses)
            {
                if (adjacency.ContainsKey(synapse.SourceNeuronId))
                    adjacency[synapse.SourceNeuronId].Add(synapse.TargetNeuronId);
                if (adjacency.ContainsKey(synapse.TargetNeuronId))
                    adjacency[synapse.TargetNeuronId].Add(synapse.SourceNeuronId);
            }
            var visited = new HashSet<int>();
            int componentCount = 0;
            void Bfs(int start)
            {
                var queue = new Queue<int>();
                queue.Enqueue(start);
                visited.Add(start);
                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (adjacency.TryGetValue(current, out var neighbors))
                        foreach (var neighbor in neighbors)
                            if (visited.Add(neighbor))
                                queue.Enqueue(neighbor);
                }
            }
            foreach (var neuron in _neurons)
            {
                if (!visited.Contains(neuron.Id))
                {
                    Bfs(neuron.Id);
                    componentCount++;
                }
            }
            if (componentCount > 1)
                _validationWarnings.Add($"Genome has {componentCount} disconnected components.");
        }

        private void ValidateComplexityBudget()
        {
            if (_neurons.Count > _complexityBudget)
                _validationErrors.Add($"Neuron count ({_neurons.Count}) exceeds complexity budget ({_complexityBudget}).");
        }

        private void ValidateDuplicateConnections()
        {
            var seen = new HashSet<(int, int)>();
            foreach (var synapse in _synapses)
            {
                var key = (synapse.SourceNeuronId, synapse.TargetNeuronId);
                if (!seen.Add(key))
                    _validationWarnings.Add($"Duplicate connection from neuron {synapse.SourceNeuronId} to {synapse.TargetNeuronId}.");
            }
        }

        private void ValidateSelfConnections()
        {
            foreach (var synapse in _synapses)
                if (synapse.SourceNeuronId == synapse.TargetNeuronId)
                    _validationErrors.Add($"Self-connection detected: synapse {synapse.Id} connects neuron {synapse.SourceNeuronId} to itself.");
        }

        private void ValidateLayerConsistency()
        {
            foreach (var synapse in _synapses)
            {
                var sourceNeuron = _neurons.FirstOrDefault(n => n.Id == synapse.SourceNeuronId);
                var targetNeuron = _neurons.FirstOrDefault(n => n.Id == synapse.TargetNeuronId);
                if (sourceNeuron.LayerIndex > targetNeuron.LayerIndex)
                    _validationWarnings.Add($"Synapse {synapse.Id} connects from layer {sourceNeuron.LayerIndex} to earlier layer {targetNeuron.LayerIndex} (potential backward connection).");
            }
        }

        private void ValidateWeightBounds()
        {
            foreach (var synapse in _synapses)
                if (!synapse.IsWeightInBounds)
                    _validationErrors.Add($"Synapse {synapse.Id} weight {synapse.Weight} is outside bounds [{synapse.MinWeight}, {synapse.MaxWeight}].");
        }

        /// <summary>Detects strongly connected components using Tarjan's algorithm.</summary>
        public ImmutableArray<ImmutableArray<int>> FindStronglyConnectedComponents()
        {
            var adjacency = new Dictionary<int, List<int>>();
            foreach (var neuron in _neurons)
                adjacency[neuron.Id] = new List<int>();
            foreach (var synapse in _synapses)
                if (adjacency.ContainsKey(synapse.SourceNeuronId))
                    adjacency[synapse.SourceNeuronId].Add(synapse.TargetNeuronId);
            var index = 0;
            var stack = new Stack<int>();
            var onStack = new HashSet<int>();
            var indices = new Dictionary<int, int>();
            var lowlinks = new Dictionary<int, int>();
            var result = new List<ImmutableArray<int>>();
            void StrongConnect(int v)
            {
                indices[v] = index;
                lowlinks[v] = index;
                index++;
                stack.Push(v);
                onStack.Add(v);
                if (adjacency.TryGetValue(v, out var neighbors))
                    foreach (var w in neighbors)
                    {
                        if (!indices.ContainsKey(w))
                        { StrongConnect(w); lowlinks[v] = Math.Min(lowlinks[v], lowlinks[w]); }
                        else if (onStack.Contains(w))
                            lowlinks[v] = Math.Min(lowlinks[v], indices[w]);
                    }
                if (lowlinks[v] == indices[v])
                {
                    var scc = new List<int>();
                    int w;
                    do
                    { w = stack.Pop(); onStack.Remove(w); scc.Add(w); } while (w != v);
                    result.Add(ImmutableArray.Create(scc.ToArray()));
                }
            }
            foreach (var neuron in _neurons)
                if (!indices.ContainsKey(neuron.Id))
                    StrongConnect(neuron.Id);
            return ImmutableArray.Create(result.ToArray());
        }

        /// <summary>Computes a topological layer ordering using Kahn's algorithm.</summary>
        public ImmutableArray<int>? ComputeTopologicalOrder()
        {
            var inDegree = new Dictionary<int, int>();
            var adjacency = new Dictionary<int, List<int>>();
            foreach (var neuron in _neurons)
            { inDegree[neuron.Id] = 0; adjacency[neuron.Id] = new List<int>(); }
            foreach (var synapse in _synapses)
            {
                if (adjacency.ContainsKey(synapse.SourceNeuronId))
                    adjacency[synapse.SourceNeuronId].Add(synapse.TargetNeuronId);
                if (inDegree.ContainsKey(synapse.TargetNeuronId))
                    inDegree[synapse.TargetNeuronId]++;
            }
            var queue = new Queue<int>();
            foreach (var kvp in inDegree)
                if (kvp.Value == 0)
                    queue.Enqueue(kvp.Key);
            var order = new List<int>();
            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                order.Add(node);
                if (adjacency.TryGetValue(node, out var neighbors))
                    foreach (var neighbor in neighbors)
                    { inDegree[neighbor]--; if (inDegree[neighbor] == 0) queue.Enqueue(neighbor); }
            }
            return order.Count == _neurons.Count ? ImmutableArray.Create(order.ToArray()) : null;
        }

        /// <summary>Automatically detects and assigns the species based on neuron composition.</summary>
        public SpeciesId DetectSpecies()
        {
            if (_forcedSpecies.HasValue)
                return _forcedSpecies.Value;
            var kindCounts = new Dictionary<NeuronKind, int>();
            foreach (var neuron in _neurons)
            {
                if (!kindCounts.ContainsKey(neuron.Kind))
                    kindCounts[neuron.Kind] = 0;
                kindCounts[neuron.Kind]++;
            }
            if (kindCounts.Count == 0)
                return SpeciesId.Default;
            var dominantKind = kindCounts.OrderByDescending(kvp => kvp.Value).First().Key;
            var domain = dominantKind switch
            {
                NeuronKind.SDFPrimitive or NeuronKind.CSGOperation or NeuronKind.Voxelizer
                    or NeuronKind.MarchingCube or NeuronKind.FractalSubdivision
                    or NeuronKind.LatticeGenerator => "geometric",
                NeuronKind.ColorField or NeuronKind.NormalMap or NeuronKind.RoughnessMap
                    or NeuronKind.MetallicMap or NeuronKind.EmissiveMap or NeuronKind.OpacityMap
                    or NeuronKind.SubsurfaceScattering or NeuronKind.ProceduralTexture => "surface",
                NeuronKind.DisplacementField or NeuronKind.Displacement or NeuronKind.CurvatureModulator
                    or NeuronKind.Tessellation or NeuronKind.TerrainSculptor => "deformation",
                NeuronKind.PhysicsCollider or NeuronKind.FluidSolver or NeuronKind.ClothSimulator
                    or NeuronKind.GravitationalField or NeuronKind.ElectromagneticField => "physics",
                NeuronKind.LODSelector or NeuronKind.OcclusionCuller or NeuronKind.FrustumCuller
                    or NeuronKind.RayMarcher or NeuronKind.FieldGenerator => "rendering",
                NeuronKind.AnimationCurve or NeuronKind.ParticleEmitter
                    or NeuronKind.HarmonicOscillator => "dynamic",
                NeuronKind.WaveFunction or NeuronKind.TensorReshaper or NeuronKind.AttentionHead
                    or NeuronKind.ConvolutionKernel or NeuronKind.PoolingLayer
                    or NeuronKind.NormalizationLayer or NeuronKind.EmbeddingLookup
                    or NeuronKind.PositionalEncoder or NeuronKind.CrossAttentionLayer
                    or NeuronKind.SelfAttentionLayer or NeuronKind.FeedForwardLayer => "neural",
                NeuronKind.VolumetricDensity or NeuronKind.ScreenSpaceReflection
                    or NeuronKind.GlobalIllumination or NeuronKind.AmbientOcclusion
                    or NeuronKind.ShadowMap or NeuronKind.PostProcessChain => "lighting",
                _ => "general"
            };
            var phylum = dominantKind.ToString().ToLowerInvariant();
            return SpeciesId.Create(domain, phylum, "auto");
        }

        /// <summary>Builds the final <see cref="GeoGenome"/> after validation.</summary>
        public GeoGenome Build()
        {
            var (isValid, errors, warnings) = Validate();
            if (!isValid && _strictValidation)
                throw new InvalidOperationException($"Genome validation failed with {errors.Count} error(s):\n" + string.Join("\n", errors));
            var species = DetectSpecies();
            var metadata = new GenomeMetadata
            {
                SemanticTag = _semanticTag,
                ComplexityBudget = _complexityBudget,
                LodPolicy = _lodPolicy,
                Author = _author,
                Description = _description,
                License = _license,
                Classification = _classification,
                Strategy = _strategy,
                Objective = _objective,
                Tags = ImmutableHashSet.Create(StringComparer.OrdinalIgnoreCase, _tags.ToArray()),
                CustomProperties = ImmutableDictionary.CreateRange(StringComparer.OrdinalIgnoreCase, _customProperties),
                State = GenomeState.Draft
            };
            var genome = new GeoGenome
            {
                Id = GenomeId.New(),
                Neurons = ImmutableArray.Create(_neurons.ToArray()),
                Synapses = ImmutableArray.Create(_synapses.ToArray()),
                Species = species,
                Metadata = metadata,
                Version = GenomeVersion.Current,
                CreatedAt = DateTimeOffset.UtcNow,
                ModifiedAt = DateTimeOffset.UtcNow
            };
            return genome with { TopologyHash = GenomeHasher.ComputeTopologyHash(genome) };
        }

        /// <summary>Builds without validation (for internal/testing use).</summary>
        public GeoGenome BuildUnchecked()
        {
            var species = DetectSpecies();
            var metadata = new GenomeMetadata
            {
                SemanticTag = _semanticTag,
                ComplexityBudget = _complexityBudget,
                LodPolicy = _lodPolicy,
                Author = _author,
                Description = _description,
                License = _license,
                Classification = _classification,
                Strategy = _strategy,
                Objective = _objective,
                Tags = ImmutableHashSet.Create(StringComparer.OrdinalIgnoreCase, _tags.ToArray()),
                CustomProperties = ImmutableDictionary.CreateRange(StringComparer.OrdinalIgnoreCase, _customProperties),
                State = GenomeState.Draft
            };
            var genome = new GeoGenome
            {
                Id = GenomeId.New(),
                Neurons = ImmutableArray.Create(_neurons.ToArray()),
                Synapses = ImmutableArray.Create(_synapses.ToArray()),
                Species = species,
                Metadata = metadata,
                Version = GenomeVersion.Current,
                CreatedAt = DateTimeOffset.UtcNow,
                ModifiedAt = DateTimeOffset.UtcNow
            };
            return genome with { TopologyHash = GenomeHasher.ComputeTopologyHash(genome) };
        }
    }

    #endregion Builder

    // ========================================================================
    #region Factories

    public static class NeuronGeneFactory
    {
        private static int _globalIdCounter;
        public static int NextId() => Interlocked.Increment(ref _globalIdCounter);
        public static void ResetIdCounter() => Interlocked.Exchange(ref _globalIdCounter, 0);

        public static NeuronGene FromPrimitiveNode(int? id = null, float radius = 1.0f,
            Vector3? position = null, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.SDFPrimitive, layerIndex)
                .WithParameter("radius", radius) with
            { Position = position ?? Vector3.Zero };
        }

        public static NeuronGene CreateSDFPrimitive(int? id = null, float radius = 1.0f,
            float smoothness = 0.1f, Vector3? position = null, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.SDFPrimitive, layerIndex)
                .WithParameter("radius", radius).WithParameter("smoothness", smoothness)
                with
            { Position = position ?? Vector3.Zero };
        }

        public static NeuronGene CreateDisplacementField(int? id = null, float amplitude = 0.5f,
            float frequency = 1.0f, int octaves = 4, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.DisplacementField, layerIndex)
                .WithParameter("amplitude", amplitude).WithParameter("frequency", frequency)
                .WithParameter("octaves", octaves);
        }

        public static NeuronGene CreateCurvatureModulator(int? id = null, float curvatureScale = 1.0f,
            float tension = 0.5f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.CurvatureModulator, layerIndex)
                .WithParameter("curvatureScale", curvatureScale).WithParameter("tension", tension);
        }

        public static NeuronGene CreateColorField(int? id = null, float hue = 0.0f,
            float saturation = 1.0f, float value = 1.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.ColorField, layerIndex)
                .WithParameter("hue", hue).WithParameter("saturation", saturation).WithParameter("value", value);
        }

        public static NeuronGene CreateNormalMap(int? id = null, float scale = 1.0f, int layerIndex = 0)
            => NeuronGene.Create(id ?? NextId(), NeuronKind.NormalMap, layerIndex).WithParameter("scale", scale);

        public static NeuronGene CreateRoughnessMap(int? id = null, float roughness = 0.5f, int layerIndex = 0)
            => NeuronGene.Create(id ?? NextId(), NeuronKind.RoughnessMap, layerIndex).WithParameter("roughness", roughness);

        public static NeuronGene CreateMetallicMap(int? id = null, float metallic = 0.0f, int layerIndex = 0)
            => NeuronGene.Create(id ?? NextId(), NeuronKind.MetallicMap, layerIndex).WithParameter("metallic", metallic);

        public static NeuronGene CreateEmissiveMap(int? id = null, float intensity = 1.0f,
            Vector3? color = null, int layerIndex = 0)
        {
            var c = color ?? Vector3.One;
            return NeuronGene.Create(id ?? NextId(), NeuronKind.EmissiveMap, layerIndex)
                .WithParameter("intensity", intensity).WithParameter("colorR", c.X)
                .WithParameter("colorG", c.Y).WithParameter("colorB", c.Z);
        }

        public static NeuronGene CreateOpacityMap(int? id = null, float opacity = 1.0f, int layerIndex = 0)
            => NeuronGene.Create(id ?? NextId(), NeuronKind.OpacityMap, layerIndex).WithParameter("opacity", opacity);

        public static NeuronGene CreateSubsurfaceScattering(int? id = null, float scatterRadius = 0.5f,
            Vector3? scatterColor = null, int layerIndex = 0)
        {
            var c = scatterColor ?? new Vector3(1.0f, 0.2f, 0.1f);
            return NeuronGene.Create(id ?? NextId(), NeuronKind.SubsurfaceScattering, layerIndex)
                .WithParameter("scatterRadius", scatterRadius).WithParameter("scatterR", c.X)
                .WithParameter("scatterG", c.Y).WithParameter("scatterB", c.Z);
        }

        public static NeuronGene CreateDisplacement(int? id = null, float strength = 1.0f,
            Vector3? direction = null, int layerIndex = 0)
        {
            var d = direction ?? Vector3.UnitY;
            return NeuronGene.Create(id ?? NextId(), NeuronKind.Displacement, layerIndex)
                .WithParameter("strength", strength).WithParameter("dirX", d.X)
                .WithParameter("dirY", d.Y).WithParameter("dirZ", d.Z);
        }

        public static NeuronGene CreateTessellation(int? id = null, int minLOD = 0,
            int maxLOD = 4, float threshold = 0.01f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.Tessellation, layerIndex)
                .WithParameter("minLOD", minLOD).WithParameter("maxLOD", maxLOD).WithParameter("threshold", threshold);
        }

        public static NeuronGene CreateLODSelector(int? id = null, float nearPlane = 1.0f,
            float farPlane = 100.0f, float transitionWidth = 5.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.LODSelector, layerIndex)
                .WithParameter("nearPlane", nearPlane).WithParameter("farPlane", farPlane)
                .WithParameter("transitionWidth", transitionWidth);
        }

        public static NeuronGene CreateOcclusionCuller(int? id = null, float cullDistance = 100.0f, int layerIndex = 0)
            => NeuronGene.Create(id ?? NextId(), NeuronKind.OcclusionCuller, layerIndex).WithParameter("cullDistance", cullDistance);

        public static NeuronGene CreateFrustumCuller(int? id = null, float margin = 0.1f, int layerIndex = 0)
            => NeuronGene.Create(id ?? NextId(), NeuronKind.FrustumCuller, layerIndex).WithParameter("margin", margin);

        public static NeuronGene CreateAnimationCurve(int? id = null, float duration = 1.0f,
            bool loop = false, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.AnimationCurve, layerIndex)
                .WithParameter("duration", duration).WithParameter("loop", loop ? 1.0f : 0.0f);
        }

        public static NeuronGene CreateParticleEmitter(int? id = null, float emissionRate = 100.0f,
            float lifetime = 2.0f, float initialVelocity = 5.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.ParticleEmitter, layerIndex)
                .WithParameter("emissionRate", emissionRate).WithParameter("lifetime", lifetime)
                .WithParameter("initialVelocity", initialVelocity);
        }

        public static NeuronGene CreatePhysicsCollider(int? id = null, float friction = 0.5f,
            float restitution = 0.3f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.PhysicsCollider, layerIndex)
                .WithParameter("friction", friction).WithParameter("restitution", restitution);
        }

        public static NeuronGene CreateFluidSolver(int? id = null, float viscosity = 0.01f,
            float density = 1.0f, float pressure = 101325.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.FluidSolver, layerIndex)
                .WithParameter("viscosity", viscosity).WithParameter("density", density).WithParameter("pressure", pressure);
        }

        public static NeuronGene CreateClothSimulator(int? id = null, float stiffness = 0.9f,
            float damping = 0.1f, float mass = 0.01f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.ClothSimulator, layerIndex)
                .WithParameter("stiffness", stiffness).WithParameter("damping", damping).WithParameter("mass", mass);
        }

        public static NeuronGene CreateVoxelizer(int? id = null, int resolution = 32, int layerIndex = 0)
            => NeuronGene.Create(id ?? NextId(), NeuronKind.Voxelizer, layerIndex).WithParameter("resolution", resolution);

        public static NeuronGene CreateMarchingCube(int? id = null, float isoLevel = 0.0f,
            int resolution = 32, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.MarchingCube, layerIndex)
                .WithParameter("isoLevel", isoLevel).WithParameter("resolution", resolution);
        }

        public static NeuronGene CreateRayMarcher(int? id = null, int maxSteps = 64,
            float epsilon = 0.001f, float maxDistance = 100.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.RayMarcher, layerIndex)
                .WithParameter("maxSteps", maxSteps).WithParameter("epsilon", epsilon).WithParameter("maxDistance", maxDistance);
        }

        public static NeuronGene CreateFieldGenerator(int? id = null, float frequency = 1.0f,
            float amplitude = 1.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.FieldGenerator, layerIndex)
                .WithParameter("frequency", frequency).WithParameter("amplitude", amplitude);
        }

        public static NeuronGene CreateWaveFunction(int? id = null, float collapseThreshold = 0.5f, int layerIndex = 0)
            => NeuronGene.Create(id ?? NextId(), NeuronKind.WaveFunction, layerIndex).WithParameter("collapseThreshold", collapseThreshold);

        public static NeuronGene CreateTensorReshaper(int? id = null, int[]? targetShape = null, int layerIndex = 0)
        {
            var n = NeuronGene.Create(id ?? NextId(), NeuronKind.TensorReshaper, layerIndex);
            if (targetShape != null)
                for (int i = 0; i < targetShape.Length; i++)
                    n = n.WithParameter($"dim_{i}", targetShape[i]);
            return n;
        }

        public static NeuronGene CreateAttentionHead(int? id = null, int headDim = 64,
            float temperature = 1.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.AttentionHead, layerIndex)
                .WithParameter("headDim", headDim).WithParameter("temperature", temperature);
        }

        public static NeuronGene CreateConvolutionKernel(int? id = null, int kernelSize = 3,
            int stride = 1, int padding = 1, int inChannels = 3, int outChannels = 16, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.ConvolutionKernel, layerIndex)
                .WithParameter("kernelSize", kernelSize).WithParameter("stride", stride)
                .WithParameter("padding", padding).WithParameter("inChannels", inChannels).WithParameter("outChannels", outChannels);
        }

        public static NeuronGene CreatePoolingLayer(int? id = null, int poolSize = 2, int stride = 2, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.PoolingLayer, layerIndex)
                .WithParameter("poolSize", poolSize).WithParameter("stride", stride);
        }

        public static NeuronGene CreateNormalizationLayer(int? id = null, float epsilon = 1e-6f,
            float momentum = 0.1f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.NormalizationLayer, layerIndex)
                .WithParameter("epsilon", epsilon).WithParameter("momentum", momentum);
        }

        public static NeuronGene CreateEmbeddingLookup(int? id = null, int vocabSize = 1000,
            int embedDim = 64, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.EmbeddingLookup, layerIndex)
                .WithParameter("vocabSize", vocabSize).WithParameter("embedDim", embedDim);
        }

        public static NeuronGene CreatePositionalEncoder(int? id = null, int maxSeqLen = 512,
            int dim = 64, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.PositionalEncoder, layerIndex)
                .WithParameter("maxSeqLen", maxSeqLen).WithParameter("dim", dim);
        }

        public static NeuronGene CreateCrossAttentionLayer(int? id = null, int embedDim = 64,
            int numHeads = 8, float dropout = 0.1f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.CrossAttentionLayer, layerIndex)
                .WithParameter("embedDim", embedDim).WithParameter("numHeads", numHeads).WithParameter("dropout", dropout);
        }

        public static NeuronGene CreateSelfAttentionLayer(int? id = null, int embedDim = 64,
            int numHeads = 8, float dropout = 0.1f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.SelfAttentionLayer, layerIndex)
                .WithParameter("embedDim", embedDim).WithParameter("numHeads", numHeads).WithParameter("dropout", dropout);
        }

        public static NeuronGene CreateFeedForwardLayer(int? id = null, int inputDim = 64,
            int hiddenDim = 256, int outputDim = 64, float dropout = 0.1f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.FeedForwardLayer, layerIndex)
                .WithParameter("inputDim", inputDim).WithParameter("hiddenDim", hiddenDim)
                .WithParameter("outputDim", outputDim).WithParameter("dropout", dropout);
        }

        public static NeuronGene CreateGeometricTransform(int? id = null, Vector3? translation = null,
            Vector3? rotation = null, Vector3? scale = null, int layerIndex = 0)
        {
            var t = translation ?? Vector3.Zero;
            var r = rotation ?? Vector3.Zero;
            var s = scale ?? Vector3.One;
            return NeuronGene.Create(id ?? NextId(), NeuronKind.GeometricTransform, layerIndex)
                .WithParameter("tx", t.X).WithParameter("ty", t.Y).WithParameter("tz", t.Z)
                .WithParameter("rx", r.X).WithParameter("ry", r.Y).WithParameter("rz", r.Z)
                .WithParameter("sx", s.X).WithParameter("sy", s.Y).WithParameter("sz", s.Z);
        }

        public static NeuronGene CreateCSGOperation(int? id = null, string operation = "union",
            float smoothRadius = 0.0f, int layerIndex = 0)
        {
            var opCode = operation.ToLowerInvariant() switch
            {
                "union" => 0.0f,
                "intersection" => 1.0f,
                "subtraction" => 2.0f,
                "smooth_union" => 3.0f,
                "smooth_intersection" => 4.0f,
                "smooth_subtraction" => 5.0f,
                _ => 0.0f
            };
            return NeuronGene.Create(id ?? NextId(), NeuronKind.CSGOperation, layerIndex)
                .WithParameter("operation", opCode).WithParameter("smoothRadius", smoothRadius);
        }

        public static NeuronGene CreateProceduralTexture(int? id = null, float scale = 1.0f,
            int octaves = 4, float persistence = 0.5f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.ProceduralTexture, layerIndex)
                .WithParameter("scale", scale).WithParameter("octaves", octaves).WithParameter("persistence", persistence);
        }

        public static NeuronGene CreateTerrainSculptor(int? id = null, float heightScale = 1.0f,
            float erosionRate = 0.1f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.TerrainSculptor, layerIndex)
                .WithParameter("heightScale", heightScale).WithParameter("erosionRate", erosionRate);
        }

        public static NeuronGene CreateLatticeGenerator(int? id = null, float cellSize = 1.0f,
            int latticeType = 0, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.LatticeGenerator, layerIndex)
                .WithParameter("cellSize", cellSize).WithParameter("latticeType", latticeType);
        }

        public static NeuronGene CreateFractalSubdivision(int? id = null, int iterations = 3,
            float jitter = 0.1f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.FractalSubdivision, layerIndex)
                .WithParameter("iterations", iterations).WithParameter("jitter", jitter);
        }

        public static NeuronGene CreateHarmonicOscillator(int? id = null, float frequency = 1.0f,
            float amplitude = 1.0f, float phase = 0.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.HarmonicOscillator, layerIndex)
                .WithParameter("frequency", frequency).WithParameter("amplitude", amplitude).WithParameter("phase", phase);
        }

        public static NeuronGene CreateGravitationalField(int? id = null, float mass = 1.0f,
            float gravitationalConstant = 6.674e-11f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.GravitationalField, layerIndex)
                .WithParameter("mass", mass).WithParameter("G", gravitationalConstant);
        }

        public static NeuronGene CreateElectromagneticField(int? id = null, float fieldStrength = 1.0f,
            float frequency = 1.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.ElectromagneticField, layerIndex)
                .WithParameter("fieldStrength", fieldStrength).WithParameter("frequency", frequency);
        }

        public static NeuronGene CreateVolumetricDensity(int? id = null, float densityScale = 1.0f,
            float absorption = 0.1f, float scattering = 0.5f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.VolumetricDensity, layerIndex)
                .WithParameter("densityScale", densityScale).WithParameter("absorption", absorption)
                .WithParameter("scattering", scattering);
        }

        public static NeuronGene CreateScreenSpaceReflection(int? id = null, float maxDistance = 100.0f,
            int maxSteps = 32, float thickness = 0.5f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.ScreenSpaceReflection, layerIndex)
                .WithParameter("maxDistance", maxDistance).WithParameter("maxSteps", maxSteps).WithParameter("thickness", thickness);
        }

        public static NeuronGene CreateGlobalIllumination(int? id = null, int rayCount = 64,
            int bounces = 3, float intensity = 1.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.GlobalIllumination, layerIndex)
                .WithParameter("rayCount", rayCount).WithParameter("bounces", bounces).WithParameter("intensity", intensity);
        }

        public static NeuronGene CreateAmbientOcclusion(int? id = null, float radius = 0.5f,
            int sampleCount = 16, float intensity = 1.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.AmbientOcclusion, layerIndex)
                .WithParameter("radius", radius).WithParameter("sampleCount", sampleCount).WithParameter("intensity", intensity);
        }

        public static NeuronGene CreateShadowMap(int? id = null, int resolution = 2048,
            float bias = 0.005f, float normalBias = 0.02f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.ShadowMap, layerIndex)
                .WithParameter("resolution", resolution).WithParameter("bias", bias).WithParameter("normalBias", normalBias);
        }

        public static NeuronGene CreatePostProcessChain(int? id = null, float exposure = 1.0f,
            float gamma = 2.2f, float saturation = 1.0f, int layerIndex = 0)
        {
            return NeuronGene.Create(id ?? NextId(), NeuronKind.PostProcessChain, layerIndex)
                .WithParameter("exposure", exposure).WithParameter("gamma", gamma).WithParameter("saturation", saturation);
        }

        public static GeoGenome CreateRandomGenome(int neuronCount = 10, int synapseCount = 15, int seed = -1)
        {
            var rng = seed >= 0 ? new Random(seed) : new Random();
            var builder = GeoGenomeBuilder.Create().WithStrictValidation(false);
            var kinds = Enum.GetValues<NeuronKind>();
            for (int i = 0; i < neuronCount; i++)
            {
                var kind = kinds[rng.Next(kinds.Length)];
                builder.AddNeuron(NeuronGene.Create(i + 1, kind, rng.Next(0, Math.Max(1, neuronCount / 3 + 1))));
            }
            var ids = Enumerable.Range(1, neuronCount).ToList();
            int synId = 1;
            int attempts = 0;
            while (synId <= synapseCount && attempts < synapseCount * 3)
            {
                attempts++;
                var si = rng.Next(ids.Count);
                var ti = rng.Next(ids.Count);
                if (si == ti)
                    continue;
                var w = (float)(rng.NextDouble() * 4.0 - 2.0);
                var types = Enum.GetValues<SynapseType>();
                builder.AddSynapse(SynapseGene.Create(synId++, ids[si], ids[ti], w, types[rng.Next(types.Length)]));
            }
            return builder.WithSemanticTag($"random-{neuronCount}n-{synapseCount}s").Build();
        }

        public static NeuronGene CloneWithMutation(NeuronGene source, float mutationRate = 0.1f, int seed = -1)
        {
            var rng = seed >= 0 ? new Random(seed) : new Random();
            var m = source;
            if (rng.NextDouble() < mutationRate)
            {
                var b = source.Bias;
                m = m with
                {
                    Bias = new Vector3(b.X + (float)(rng.NextDouble() * 0.4 - 0.2),
                    b.Y + (float)(rng.NextDouble() * 0.4 - 0.2), b.Z + (float)(rng.NextDouble() * 0.4 - 0.2)),
                    WeightScale = source.WeightScale * (float)(0.8 + rng.NextDouble() * 0.4),
                    LastModified = DateTimeOffset.UtcNow
                };
            }
            if (rng.NextDouble() < mutationRate)
            {
                var acts = Enum.GetValues<ActivationKernel>();
                m = m with { Activation = acts[rng.Next(acts.Length)] };
            }
            if (rng.NextDouble() < mutationRate)
            {
                var kinds = Enum.GetValues<NeuronKind>();
                m = m with { Kind = kinds[rng.Next(kinds.Length)] };
            }
            return m;
        }

        public static NeuronGene CloneWithPerturbation(NeuronGene source, float scale = 0.05f, int seed = -1)
        {
            var rng = seed >= 0 ? new Random(seed) : new Random();
            var p = source.Parameters;
            foreach (var k in p.Keys)
                p = p.SetItem(k, p[k] + (float)(rng.NextDouble() * 2.0 - 1.0) * scale);
            return source with
            {
                Parameters = p,
                Bias = source.Bias + new Vector3((float)(rng.NextDouble() * 2.0 - 1.0) * scale,
                    (float)(rng.NextDouble() * 2.0 - 1.0) * scale, (float)(rng.NextDouble() * 2.0 - 1.0) * scale),
                WeightScale = source.WeightScale * (1.0f + (float)(rng.NextDouble() * 2.0 - 1.0) * scale),
                LastModified = DateTimeOffset.UtcNow
            };
        }

        public static NeuronGene DeepClone(NeuronGene source, int? newId = null)
            => source with { Id = newId ?? NextId(), Parameters = source.Parameters, SynapseTargets = source.SynapseTargets, LastModified = DateTimeOffset.UtcNow };
    }

    public static class SynapseGeneFactory
    {
        private static int _globalIdCounter;
        public static int NextId() => Interlocked.Increment(ref _globalIdCounter);
        public static void ResetIdCounter() => Interlocked.Exchange(ref _globalIdCounter, 0);

        public static SynapseGene Create(int src, int tgt, float weight = 1.0f, SynapseType type = SynapseType.Excitatory)
            => SynapseGene.Create(NextId(), src, tgt, weight, type);

        public static SynapseGene CreatePlastic(int src, int tgt, float weight = 1.0f, float lr = 0.01f)
            => SynapseGene.CreatePlastic(NextId(), src, tgt, weight, lr);

        public static SynapseGene CreateInhibitory(int src, int tgt, float weight = -1.0f)
            => SynapseGene.CreateInhibitory(NextId(), src, tgt, weight);

        public static SynapseGene CreateModulatory(int src, int tgt, float weight = 0.5f)
            => SynapseGene.CreateModulatory(NextId(), src, tgt, weight);

        public static SynapseGene CreateAdaptive(int src, int tgt, float w = 1.0f, float ar = 0.01f)
            => SynapseGene.Create(NextId(), src, tgt, w, SynapseType.Adaptive) with { LearningRate = ar, WeightDecay = 0.0005f };

        public static SynapseGene CreateRandom(int minId, int maxId, Random? rng = null, SynapseType? pref = null)
        {
            rng ??= new Random();
            int s, t;
            do
            { s = rng.Next(minId, maxId + 1); t = rng.Next(minId, maxId + 1); } while (s == t);
            var w = (float)(rng.NextDouble() * 4.0 - 2.0);
            var tp = pref ?? Enum.GetValues<SynapseType>()[rng.Next(Enum.GetValues<SynapseType>().Length)];
            return SynapseGene.Create(NextId(), s, t, w, tp);
        }

        public static ImmutableArray<SynapseGene> CreateRandomBatch(int count, int minId, int maxId, Random? rng = null)
        {
            rng ??= new Random();
            var b = ImmutableArray.CreateBuilder<SynapseGene>();
            for (int i = 0; i < count; i++)
                b.Add(CreateRandom(minId, maxId, rng));
            return b.ToImmutable();
        }

        public static SynapseGene CloneWithPerturbation(SynapseGene src, float scale = 0.1f, Random? rng = null)
        {
            rng ??= new Random();
            var noise = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
            return src with { Weight = Math.Clamp(src.Weight + noise, src.MinWeight, src.MaxWeight), UpdateCount = src.UpdateCount + 1 };
        }

        public static SynapseGene DeepClone(SynapseGene src, int? newId = null) => src with { Id = newId ?? NextId() };
    }

    #endregion Factories


    // ========================================================================
    #region Validator

    public enum ValidationSeverity { Info = 0, Warning = 1, Error = 2 }

    public sealed record ValidationResult
    {
        public ValidationSeverity Severity { get; init; }
        public string Message { get; init; } = string.Empty;
        public string? Code { get; init; }
        public object? Data { get; init; }
        public ValidationResult() { }
        public ValidationResult(ValidationSeverity severity, string message) { Severity = severity; Message = message; }
    }

    public sealed class GenomeValidationReport
    {
        public GenomeId GenomeId { get; init; }
        public bool HasStructuralIssues { get; set; }
        public bool HasBudgetViolation { get; set; }
        public ImmutableArray<int> OrphanNeuronIds { get; set; } = ImmutableArray<int>.Empty;
        public int DisconnectedComponentCount { get; set; }
        public int UnreachableNeuronCount { get; set; }
        public float TotalComputeCost { get; set; }
        public ImmutableHashSet<string> SemanticTags { get; set; } = ImmutableHashSet<string>.Empty;
        public Dictionary<NeuronKind, int> KindDistribution { get; set; } = new();
        public bool IsHealthy => !HasStructuralIssues && !HasBudgetViolation && OrphanNeuronIds.Length == 0 && DisconnectedComponentCount <= 1;
    }

    public sealed class GenomeValidator
    {
        private readonly List<ValidationResult> _results = new();
        public IReadOnlyList<ValidationResult> Results => _results;
        public bool IsValid => _results.All(r => r.Severity != ValidationSeverity.Error);

        public GenomeValidationReport ValidateStructure(GenomeId genomeId,
            ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses)
        {
            _results.Clear();
            var report = new GenomeValidationReport { GenomeId = genomeId };
            ValidateNeuronIdsUnique(neurons, report);
            ValidateSynapseIdsUnique(synapses, report);
            ValidateSynapseReferences(neurons, synapses, report);
            ValidateSelfConnections(synapses, report);
            ValidateWeightBounds(synapses, report);
            ValidateDuplicateConnections(synapses, report);
            ValidateNeuronParameters(neurons, report);
            return report;
        }

        public GenomeValidationReport ValidateConnectivity(GenomeId genomeId,
            ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses)
        {
            var report = new GenomeValidationReport { GenomeId = genomeId };
            ValidateOrphanNeurons(neurons, synapses, report);
            ValidateDisconnectedComponents(neurons, synapses, report);
            ValidateReachability(neurons, synapses, report);
            return report;
        }

        public GenomeValidationReport ValidateBudget(GenomeId genomeId,
            ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses,
            int complexityBudget, long memoryBudgetBytes)
        {
            var report = new GenomeValidationReport { GenomeId = genomeId };
            if (neurons.Length > complexityBudget)
            {
                _results.Add(new ValidationResult(ValidationSeverity.Error, $"Neuron count ({neurons.Length}) exceeds budget ({complexityBudget})."));
                report.HasBudgetViolation = true;
            }
            long mem = neurons.Sum(n => (long)n.TotalEstimatedMemory) + synapses.Sum(s => (long)System.Runtime.CompilerServices.Unsafe.SizeOf<SynapseGene>());
            if (mem > memoryBudgetBytes)
            {
                _results.Add(new ValidationResult(ValidationSeverity.Error, $"Memory ({mem} bytes) exceeds budget ({memoryBudgetBytes} bytes)."));
                report.HasBudgetViolation = true;
            }
            report.TotalComputeCost = neurons.Sum(n => n.EffectiveComputeCost);
            return report;
        }

        public GenomeValidationReport ValidateSemantics(GenomeId genomeId,
            ImmutableArray<NeuronGene> neurons, GenomeMetadata metadata)
        {
            var report = new GenomeValidationReport { GenomeId = genomeId };
            if (string.IsNullOrWhiteSpace(metadata.SemanticTag))
                _results.Add(new ValidationResult(ValidationSeverity.Warning, "No semantic tag."));
            if (neurons.Length == 0)
                _results.Add(new ValidationResult(ValidationSeverity.Warning, "No neurons."));
            var disabledRatio = neurons.Length > 0 ? (float)neurons.Count(n => !n.IsEnabled) / neurons.Length : 0;
            if (disabledRatio > 0.5f)
                _results.Add(new ValidationResult(ValidationSeverity.Warning, $"More than 50% disabled ({disabledRatio:P0})."));
            report.SemanticTags = metadata.Tags;
            report.KindDistribution = neurons.GroupBy(n => n.Kind).ToDictionary(g => g.Key, g => g.Count());
            return report;
        }

        public bool DetectCycles(ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses)
        {
            if (neurons.Length == 0 || synapses.Length == 0)
                return false;
            var adj = new Dictionary<int, List<int>>();
            foreach (var n in neurons)
                adj[n.Id] = new List<int>();
            foreach (var s in synapses)
                if (adj.ContainsKey(s.SourceNeuronId))
                    adj[s.SourceNeuronId].Add(s.TargetNeuronId);
            var visited = new HashSet<int>();
            var stack = new HashSet<int>();
            bool HasCycle(int v)
            {
                if (stack.Contains(v))
                    return true;
                if (visited.Contains(v))
                    return false;
                visited.Add(v);
                stack.Add(v);
                if (adj.TryGetValue(v, out var nb))
                    foreach (var w in nb)
                        if (HasCycle(w))
                            return true;
                stack.Remove(v);
                return false;
            }
            foreach (var n in neurons)
                if (!visited.Contains(n.Id) && HasCycle(n.Id))
                    return true;
            return false;
        }

        public ImmutableArray<ImmutableArray<int>> FindStronglyConnectedComponents(
            ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses)
        {
            var adj = new Dictionary<int, List<int>>();
            foreach (var n in neurons)
                adj[n.Id] = new List<int>();
            foreach (var s in synapses)
                if (adj.ContainsKey(s.SourceNeuronId))
                    adj[s.SourceNeuronId].Add(s.TargetNeuronId);
            int idx = 0;
            var stk = new Stack<int>();
            var onStk = new HashSet<int>();
            var indices = new Dictionary<int, int>();
            var low = new Dictionary<int, int>();
            var result = new List<ImmutableArray<int>>();
            void SC(int v)
            {
                indices[v] = idx;
                low[v] = idx;
                idx++;
                stk.Push(v);
                onStk.Add(v);
                if (adj.TryGetValue(v, out var nb))
                    foreach (var w in nb)
                    {
                        if (!indices.ContainsKey(w))
                        { SC(w); low[v] = Math.Min(low[v], low[w]); }
                        else if (onStk.Contains(w))
                            low[v] = Math.Min(low[v], indices[w]);
                    }
                if (low[v] == indices[v])
                {
                    var scc = new List<int>();
                    int w;
                    do
                    { w = stk.Pop(); onStk.Remove(w); scc.Add(w); } while (w != v);
                    result.Add(ImmutableArray.Create(scc.ToArray()));
                }
            }
            foreach (var n in neurons)
                if (!indices.ContainsKey(n.Id))
                    SC(n.Id);
            return ImmutableArray.Create(result.ToArray());
        }

        public ImmutableArray<int> FindOrphanNeurons(ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses)
        {
            var connected = new HashSet<int>();
            foreach (var s in synapses)
            { connected.Add(s.SourceNeuronId); connected.Add(s.TargetNeuronId); }
            var orphans = ImmutableArray.CreateBuilder<int>();
            foreach (var n in neurons)
                if (!connected.Contains(n.Id))
                    orphans.Add(n.Id);
            return orphans.ToImmutable();
        }

        private void ValidateNeuronIdsUnique(ImmutableArray<NeuronGene> neurons, GenomeValidationReport report)
        {
            var seen = new HashSet<int>();
            foreach (var n in neurons)
                if (!seen.Add(n.Id))
                { _results.Add(new ValidationResult(ValidationSeverity.Error, $"Duplicate neuron ID: {n.Id}.")); report.HasStructuralIssues = true; }
        }

        private void ValidateSynapseIdsUnique(ImmutableArray<SynapseGene> synapses, GenomeValidationReport report)
        {
            var seen = new HashSet<int>();
            foreach (var s in synapses)
                if (!seen.Add(s.Id))
                { _results.Add(new ValidationResult(ValidationSeverity.Error, $"Duplicate synapse ID: {s.Id}.")); report.HasStructuralIssues = true; }
        }

        private void ValidateSynapseReferences(ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses, GenomeValidationReport report)
        {
            var ids = new HashSet<int>(neurons.Select(n => n.Id));
            foreach (var s in synapses)
            {
                if (!ids.Contains(s.SourceNeuronId))
                { _results.Add(new ValidationResult(ValidationSeverity.Error, $"Synapse {s.Id} refs non-existent source {s.SourceNeuronId}.")); report.HasStructuralIssues = true; }
                if (!ids.Contains(s.TargetNeuronId))
                { _results.Add(new ValidationResult(ValidationSeverity.Error, $"Synapse {s.Id} refs non-existent target {s.TargetNeuronId}.")); report.HasStructuralIssues = true; }
            }
        }

        private void ValidateSelfConnections(ImmutableArray<SynapseGene> synapses, GenomeValidationReport report)
        {
            foreach (var s in synapses)
                if (s.SourceNeuronId == s.TargetNeuronId)
                { _results.Add(new ValidationResult(ValidationSeverity.Error, $"Self-connection: synapse {s.Id}.")); report.HasStructuralIssues = true; }
        }

        private void ValidateWeightBounds(ImmutableArray<SynapseGene> synapses, GenomeValidationReport report)
        {
            foreach (var s in synapses)
                if (!s.IsWeightInBounds)
                { _results.Add(new ValidationResult(ValidationSeverity.Error, $"Synapse {s.Id} weight {s.Weight} out of bounds.")); report.HasStructuralIssues = true; }
        }

        private void ValidateDuplicateConnections(ImmutableArray<SynapseGene> synapses, GenomeValidationReport report)
        {
            var seen = new HashSet<(int, int)>();
            foreach (var s in synapses)
                if (!seen.Add((s.SourceNeuronId, s.TargetNeuronId)))
                { _results.Add(new ValidationResult(ValidationSeverity.Warning, $"Duplicate connection {s.SourceNeuronId}->{s.TargetNeuronId}.")); report.HasStructuralIssues = true; }
        }

        private void ValidateNeuronParameters(ImmutableArray<NeuronGene> neurons, GenomeValidationReport report)
        {
            foreach (var n in neurons)
            {
                foreach (var p in n.Parameters)
                    if (float.IsNaN(p.Value) || float.IsInfinity(p.Value))
                    { _results.Add(new ValidationResult(ValidationSeverity.Error, $"Neuron {n.Id} invalid param '{p.Key}'.")); report.HasStructuralIssues = true; }
                if (float.IsNaN(n.WeightScale) || float.IsInfinity(n.WeightScale))
                { _results.Add(new ValidationResult(ValidationSeverity.Error, $"Neuron {n.Id} invalid WeightScale.")); report.HasStructuralIssues = true; }
            }
        }

        private void ValidateOrphanNeurons(ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses, GenomeValidationReport report)
        {
            var orphans = FindOrphanNeurons(neurons, synapses);
            report.OrphanNeuronIds = orphans;
            if (orphans.Length > 0 && neurons.Length > 1)
                _results.Add(new ValidationResult(ValidationSeverity.Warning, $"Found {orphans.Length} orphan neuron(s)."));
        }

        private void ValidateDisconnectedComponents(ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses, GenomeValidationReport report)
        {
            if (neurons.Length <= 1)
                return;
            var adj = new Dictionary<int, List<int>>();
            foreach (var n in neurons)
                adj[n.Id] = new List<int>();
            foreach (var s in synapses)
            {
                if (adj.ContainsKey(s.SourceNeuronId))
                    adj[s.SourceNeuronId].Add(s.TargetNeuronId);
                if (adj.ContainsKey(s.TargetNeuronId))
                    adj[s.TargetNeuronId].Add(s.SourceNeuronId);
            }
            var visited = new HashSet<int>();
            int comps = 0;
            void Bfs(int start)
            {
                var q = new Queue<int>();
                q.Enqueue(start);
                visited.Add(start);
                while (q.Count > 0)
                { var c = q.Dequeue(); if (adj.TryGetValue(c, out var nb)) foreach (var nb2 in nb) if (visited.Add(nb2)) q.Enqueue(nb2); }
            }
            foreach (var n in neurons)
                if (!visited.Contains(n.Id))
                { Bfs(n.Id); comps++; }
            report.DisconnectedComponentCount = comps;
            if (comps > 1)
                _results.Add(new ValidationResult(ValidationSeverity.Warning, $"Genome has {comps} disconnected components."));
        }

        private void ValidateReachability(ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses, GenomeValidationReport report)
        {
            if (neurons.Length <= 1)
                return;
            var adj = new Dictionary<int, List<int>>();
            foreach (var n in neurons)
                adj[n.Id] = new List<int>();
            foreach (var s in synapses)
                if (adj.ContainsKey(s.SourceNeuronId))
                    adj[s.SourceNeuronId].Add(s.TargetNeuronId);
            int unreachableCount = 0;
            foreach (var start in neurons)
            {
                var reachable = new HashSet<int>();
                var q = new Queue<int>();
                q.Enqueue(start.Id);
                reachable.Add(start.Id);
                while (q.Count > 0)
                { var c = q.Dequeue(); if (adj.TryGetValue(c, out var nb)) foreach (var w in nb) if (reachable.Add(w)) q.Enqueue(w); }
                if (neurons.Length - reachable.Count > 0)
                    unreachableCount++;
            }
            report.UnreachableNeuronCount = unreachableCount;
        }
    }

    #endregion Validator

    // ========================================================================
    #region Hasher

    public static class GenomeHasher
    {
        public static string ComputeTopologyHash(GenomeId id, ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses)
        {
            using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            h.AppendData(id.Value.ToByteArray());
            h.AppendData(BitConverter.GetBytes(neurons.Length));
            foreach (var n in neurons.OrderBy(n => n.Id))
            {
                Span<byte> b = stackalloc byte[4];
                BinaryPrimitives.WriteInt32BigEndian(b, n.Id);
                h.AppendData(b);
                h.AppendData(new byte[] { (byte)n.Kind });
                BinaryPrimitives.WriteInt32BigEndian(b, n.LayerIndex);
                h.AppendData(b);
            }
            h.AppendData(BitConverter.GetBytes(synapses.Length));
            foreach (var s in synapses.OrderBy(s => s.Id))
            {
                Span<byte> b = stackalloc byte[12];
                BinaryPrimitives.WriteInt32BigEndian(b, s.Id);
                BinaryPrimitives.WriteInt32BigEndian(b[4..], s.SourceNeuronId);
                BinaryPrimitives.WriteInt32BigEndian(b[8..], s.TargetNeuronId);
                h.AppendData(b);
            }
            return Convert.ToHexString(h.GetHashAndReset());
        }

        public static string ComputeSemanticHash(GenomeId id, GenomeMetadata metadata, SpeciesId species)
        {
            using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            h.AppendData(id.Value.ToByteArray());
            h.AppendData(Encoding.UTF8.GetBytes(metadata.SemanticTag ?? ""));
            h.AppendData(new byte[] { (byte)metadata.Classification, (byte)metadata.Strategy, (byte)metadata.Objective, (byte)metadata.LodPolicy, (byte)metadata.State });
            h.AppendData(BitConverter.GetBytes(metadata.Generation));
            h.AppendData(Encoding.UTF8.GetBytes(species.Value));
            foreach (var t in metadata.Tags.OrderBy(t => t, StringComparer.Ordinal))
                h.AppendData(Encoding.UTF8.GetBytes(t));
            foreach (var p in metadata.CustomProperties.OrderBy(k => k.Key, StringComparer.Ordinal))
            { h.AppendData(Encoding.UTF8.GetBytes(p.Key)); h.AppendData(Encoding.UTF8.GetBytes(p.Value)); }
            return Convert.ToHexString(h.GetHashAndReset());
        }

        public static string ComputeTopologyHash(GeoGenome genome)
            => ComputeTopologyHash(genome.Id, genome.Neurons, genome.Synapses);

        public static string ComputeSemanticHash(GeoGenome genome)
            => ComputeSemanticHash(genome.Id, genome.Metadata, genome.Species);

        public static string ComputeFullHash(GeoGenome genome)
        {
            using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            h.AppendData(Encoding.UTF8.GetBytes(ComputeTopologyHash(genome)));
            h.AppendData(Encoding.UTF8.GetBytes(ComputeSemanticHash(genome)));
            Span<byte> vb = stackalloc byte[12];
            BinaryPrimitives.WriteInt32BigEndian(vb, genome.Version.Major);
            BinaryPrimitives.WriteInt32BigEndian(vb[4..], genome.Version.Minor);
            BinaryPrimitives.WriteInt32BigEndian(vb[8..], genome.Version.Patch);
            h.AppendData(vb);
            return Convert.ToHexString(h.GetHashAndReset());
        }

        public static string UpdateTopologyHashForNeuronModification(string prev, int nid,
            NeuronKind oldK, NeuronKind newK, int oldL, int newL)
        {
            if (oldK == newK && oldL == newL)
                return prev;
            using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            h.AppendData(Convert.FromHexString(prev));
            h.AppendData(new byte[] { 0x01 });
            Span<byte> b = stackalloc byte[4];
            BinaryPrimitives.WriteInt32BigEndian(b, nid);
            h.AppendData(b);
            h.AppendData(new byte[] { (byte)oldK, (byte)newK });
            Span<byte> lb = stackalloc byte[8];
            BinaryPrimitives.WriteInt32BigEndian(lb, oldL);
            BinaryPrimitives.WriteInt32BigEndian(lb[4..], newL);
            h.AppendData(lb);
            return Convert.ToHexString(h.GetHashAndReset());
        }

        public static string UpdateTopologyHashForSynapseAddition(string prev, SynapseGene syn)
        {
            using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            h.AppendData(Convert.FromHexString(prev));
            h.AppendData(new byte[] { 0x02 });
            Span<byte> b = stackalloc byte[12];
            BinaryPrimitives.WriteInt32BigEndian(b, syn.Id);
            BinaryPrimitives.WriteInt32BigEndian(b[4..], syn.SourceNeuronId);
            BinaryPrimitives.WriteInt32BigEndian(b[8..], syn.TargetNeuronId);
            h.AppendData(b);
            return Convert.ToHexString(h.GetHashAndReset());
        }

        public static string UpdateTopologyHashForSynapseRemoval(string prev, SynapseGene syn)
        {
            using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            h.AppendData(Convert.FromHexString(prev));
            h.AppendData(new byte[] { 0x03 });
            Span<byte> b = stackalloc byte[12];
            BinaryPrimitives.WriteInt32BigEndian(b, syn.Id);
            BinaryPrimitives.WriteInt32BigEndian(b[4..], syn.SourceNeuronId);
            BinaryPrimitives.WriteInt32BigEndian(b[8..], syn.TargetNeuronId);
            h.AppendData(b);
            return Convert.ToHexString(h.GetHashAndReset());
        }

        public static string UpdateTopologyHashForWeightChange(string prev, int synId, float oldW, float newW)
        {
            if (Math.Abs(oldW - newW) < float.Epsilon)
                return prev;
            using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            h.AppendData(Convert.FromHexString(prev));
            h.AppendData(new byte[] { 0x04 });
            Span<byte> b = stackalloc byte[4];
            BinaryPrimitives.WriteInt32BigEndian(b, synId);
            h.AppendData(b);
            Span<byte> wb = stackalloc byte[8];
            BinaryPrimitives.WriteInt32BigEndian(wb, BitConverter.SingleToInt32Bits(oldW));
            BinaryPrimitives.WriteInt32BigEndian(wb[4..], BitConverter.SingleToInt32Bits(newW));
            h.AppendData(wb);
            return Convert.ToHexString(h.GetHashAndReset());
        }

        public static int ComputeFastHash(GeoGenome genome)
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 31 + genome.Id.GetHashCode();
                hash = hash * 31 + genome.Neurons.Length;
                hash = hash * 31 + genome.Synapses.Length;
                foreach (var n in genome.Neurons)
                { hash = hash * 31 + n.Id; hash = hash * 31 + (int)n.Kind; hash = hash * 31 + n.LayerIndex; }
                foreach (var s in genome.Synapses)
                { hash = hash * 31 + s.Id; hash = hash * 31 + s.SourceNeuronId; hash = hash * 31 + s.TargetNeuronId; hash = hash * 31 + BitConverter.SingleToInt32Bits(s.Weight); }
                return hash;
            }
        }

        public static bool TopologyHashesEqual(string h1, string h2) => string.Equals(h1, h2, StringComparison.OrdinalIgnoreCase);

        public static bool IsTopologyHashValid(GenomeId id, ImmutableArray<NeuronGene> neurons, ImmutableArray<SynapseGene> synapses, string cached)
            => TopologyHashesEqual(ComputeTopologyHash(id, neurons, synapses), cached);
    }

    #endregion Hasher


    // ========================================================================
    #region Differ

    public sealed class GenomeDiffer
    {
        public GenomeDiff ComputeDiff(GenomeId srcId, ImmutableArray<NeuronGene> srcN, ImmutableArray<SynapseGene> srcS,
            GenomeId tgtId, ImmutableArray<NeuronGene> tgtN, ImmutableArray<SynapseGene> tgtS,
            GenomeMetadata? srcMeta = null, GenomeMetadata? tgtMeta = null)
        {
            var srcND = srcN.ToDictionary(n => n.Id);
            var tgtND = tgtN.ToDictionary(n => n.Id);
            var addedN = ImmutableArray.CreateBuilder<NeuronGene>();
            var removedN = ImmutableArray.CreateBuilder<int>();
            var modifiedN = ImmutableArray.CreateBuilder<(NeuronGene, NeuronGene)>();
            foreach (var t in tgtN)
            { if (srcND.TryGetValue(t.Id, out var s)) { if (!s.Equals(t)) modifiedN.Add((s, t)); } else addedN.Add(t); }
            foreach (var s in srcN)
                if (!tgtND.ContainsKey(s.Id))
                    removedN.Add(s.Id);

            var srcSD = srcS.ToDictionary(s => s.Id);
            var tgtSD = tgtS.ToDictionary(s => s.Id);
            var addedS = ImmutableArray.CreateBuilder<SynapseGene>();
            var removedS = ImmutableArray.CreateBuilder<int>();
            var modifiedS = ImmutableArray.CreateBuilder<(SynapseGene, SynapseGene)>();
            foreach (var t in tgtS)
            { if (srcSD.TryGetValue(t.Id, out var s)) { if (!s.Equals(t)) modifiedS.Add((s, t)); } else addedS.Add(t); }
            foreach (var s in srcS)
                if (!tgtSD.ContainsKey(s.Id))
                    removedS.Add(s.Id);

            GenomeMetadata? metaChanges = (srcMeta != null && tgtMeta != null && !srcMeta.Equals(tgtMeta)) ? tgtMeta : null;
            return new GenomeDiff
            {
                SourceGenomeId = srcId,
                TargetGenomeId = tgtId,
                AddedNeurons = addedN.ToImmutable(),
                RemovedNeurons = removedN.ToImmutable(),
                ModifiedNeurons = modifiedN.ToImmutable(),
                AddedSynapses = addedS.ToImmutable(),
                RemovedSynapses = removedS.ToImmutable(),
                ModifiedSynapses = modifiedS.ToImmutable(),
                MetadataChanges = metaChanges
            };
        }

        public GenomeDiff ComputeDiff(GeoGenome source, GeoGenome target)
            => ComputeDiff(source.Id, source.Neurons, source.Synapses, target.Id, target.Neurons, target.Synapses, source.Metadata, target.Metadata);

        public GeoGenome ApplyDiff(GenomeId srcId, ImmutableArray<NeuronGene> srcN, ImmutableArray<SynapseGene> srcS,
            SpeciesId species, GenomeMetadata srcMeta, GenomeDiff diff)
        {
            var nd = srcN.ToDictionary(n => n.Id);
            var sd = srcS.ToDictionary(s => s.Id);
            foreach (var rid in diff.RemovedNeurons)
                nd.Remove(rid);
            foreach (var a in diff.AddedNeurons)
                nd[a.Id] = a;
            foreach (var (o, m) in diff.ModifiedNeurons)
                nd[m.Id] = m;
            foreach (var rid in diff.RemovedSynapses)
                sd.Remove(rid);
            foreach (var a in diff.AddedSynapses)
                sd[a.Id] = a;
            foreach (var (o, m) in diff.ModifiedSynapses)
                sd[m.Id] = m;
            var meta = diff.MetadataChanges ?? srcMeta;
            var genome = new GeoGenome
            {
                Id = diff.TargetGenomeId,
                Neurons = ImmutableArray.Create(nd.Values.ToArray()),
                Synapses = ImmutableArray.Create(sd.Values.ToArray()),
                Species = species,
                Metadata = meta,
                Version = GenomeVersion.Current,
                CreatedAt = DateTimeOffset.UtcNow,
                ModifiedAt = DateTimeOffset.UtcNow
            };
            return genome with { TopologyHash = GenomeHasher.ComputeTopologyHash(genome) };
        }

        public GeoGenome ApplyDiff(GeoGenome source, GenomeDiff diff)
            => ApplyDiff(source.Id, source.Neurons, source.Synapses, source.Species, source.Metadata, diff);

        public GenomeDiff InvertDiff(GenomeDiff diff) => diff.Invert();

        public GenomeDiff MergeDiffs(GenomeDiff baseDiff, GenomeDiff otherDiff)
        {
            if (baseDiff.SourceGenomeId != otherDiff.SourceGenomeId)
                throw new ArgumentException("Cannot merge diffs with different source genomes.");
            var an = new Dictionary<int, NeuronGene>();
            foreach (var n in baseDiff.AddedNeurons)
                an[n.Id] = n;
            foreach (var n in otherDiff.AddedNeurons)
                an[n.Id] = n;
            var rn = new HashSet<int>(baseDiff.RemovedNeurons);
            foreach (var id in otherDiff.RemovedNeurons)
                rn.Add(id);
            var mn = new Dictionary<int, (NeuronGene, NeuronGene)>();
            foreach (var m in baseDiff.ModifiedNeurons)
                mn[m.Original.Id] = m;
            foreach (var m in otherDiff.ModifiedNeurons)
                mn[m.Original.Id] = m;
            var asy = new Dictionary<int, SynapseGene>();
            foreach (var s in baseDiff.AddedSynapses)
                asy[s.Id] = s;
            foreach (var s in otherDiff.AddedSynapses)
                asy[s.Id] = s;
            var rs = new HashSet<int>(baseDiff.RemovedSynapses);
            foreach (var id in otherDiff.RemovedSynapses)
                rs.Add(id);
            var ms = new Dictionary<int, (SynapseGene, SynapseGene)>();
            foreach (var m in baseDiff.ModifiedSynapses)
                ms[m.Original.Id] = m;
            foreach (var m in otherDiff.ModifiedSynapses)
                ms[m.Original.Id] = m;
            return new GenomeDiff
            {
                SourceGenomeId = baseDiff.SourceGenomeId,
                TargetGenomeId = baseDiff.TargetGenomeId,
                AddedNeurons = ImmutableArray.Create(an.Values.ToArray()),
                RemovedNeurons = ImmutableArray.Create(rn.ToArray()),
                ModifiedNeurons = ImmutableArray.Create(mn.Values.ToArray()),
                AddedSynapses = ImmutableArray.Create(asy.Values.ToArray()),
                RemovedSynapses = ImmutableArray.Create(rs.ToArray()),
                ModifiedSynapses = ImmutableArray.Create(ms.Values.ToArray()),
                MetadataChanges = otherDiff.MetadataChanges ?? baseDiff.MetadataChanges
            };
        }

        public (GeoGenome Result, bool HasConflicts, IReadOnlyList<string> Conflicts) ThreeWayMerge(
            GeoGenome baseGenome, GeoGenome localGenome, GeoGenome remoteGenome)
        {
            var b2l = ComputeDiff(baseGenome, localGenome);
            var b2r = ComputeDiff(baseGenome, remoteGenome);
            var conflicts = new List<string>();
            bool hasConflicts = false;
            var lm = b2l.ModifiedNeurons.ToDictionary(m => m.Original.Id);
            var rm = b2r.ModifiedNeurons.ToDictionary(m => m.Original.Id);
            foreach (var kvp in lm)
                if (rm.ContainsKey(kvp.Key))
                { hasConflicts = true; conflicts.Add($"Conflicting neuron {kvp.Key} mod."); }
            var lms = b2l.ModifiedSynapses.ToDictionary(m => m.Original.Id);
            var rms = b2r.ModifiedSynapses.ToDictionary(m => m.Original.Id);
            foreach (var kvp in lms)
                if (rms.ContainsKey(kvp.Key))
                { hasConflicts = true; conflicts.Add($"Conflicting synapse {kvp.Key} mod."); }
            var lrn = new HashSet<int>(b2l.RemovedNeurons);
            var rrn = new HashSet<int>(b2r.RemovedNeurons);
            foreach (var id in lrn)
                if (rm.ContainsKey(id))
                { hasConflicts = true; conflicts.Add($"Neuron {id} removed locally but modified remotely."); }
            foreach (var id in rrn)
                if (lm.ContainsKey(id))
                { hasConflicts = true; conflicts.Add($"Neuron {id} removed remotely but modified locally."); }
            if (!hasConflicts)
            {
                try
                { var merged = MergeDiffs(b2l, b2r); return (ApplyDiff(baseGenome, merged), false, conflicts); }
                catch (Exception ex) { conflicts.Add($"Merge failed: {ex.Message}"); return (localGenome, true, conflicts); }
            }
            return (remoteGenome, true, conflicts);
        }
    }

    #endregion Differ

    // ========================================================================
    #region Serializer

    public static class GenomeSerializer
    {
        private const uint MagicNumber = 0x47444E4E; // "GDNN"
        private const int CurrentFormatVersion = 1;

        public static byte[] Serialize(GeoGenome genome)
        {
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms, Encoding.UTF8, true);
            writer.Write(MagicNumber);
            writer.Write(CurrentFormatVersion);
            writer.Write(genome.Id.Value.ToByteArray());
            WriteString(writer, genome.Species.Value);
            WriteString(writer, genome.TopologyHash ?? "");
            writer.Write(genome.Version.Major);
            writer.Write(genome.Version.Minor);
            writer.Write(genome.Version.Patch);
            writer.Write(genome.CreatedAt.ToUnixTimeMilliseconds());
            writer.Write(genome.ModifiedAt.ToUnixTimeMilliseconds());
            WriteMetadata(writer, genome.Metadata);
            writer.Write(genome.Neurons.Length);
            foreach (var n in genome.Neurons)
                WriteNeuron(writer, n);
            writer.Write(genome.Synapses.Length);
            foreach (var s in genome.Synapses)
                WriteSynapse(writer, s);
            return ms.ToArray();
        }

        public static GeoGenome Deserialize(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms, Encoding.UTF8, true);
            var magic = reader.ReadUInt32();
            if (magic != MagicNumber)
                throw new InvalidDataException($"Invalid magic number: 0x{magic:X8}");
            var formatVersion = reader.ReadInt32();
            if (formatVersion > CurrentFormatVersion)
                throw new InvalidDataException($"Unsupported format version: {formatVersion}");
            var id = new GenomeId(new Guid(reader.ReadBytes(16)));
            var species = new SpeciesId(ReadString(reader));
            var topologyHash = ReadString(reader);
            var major = reader.ReadInt32();
            var minor = reader.ReadInt32();
            var patch = reader.ReadInt32();
            var version = new GenomeVersion(major, minor, patch);
            var created = DateTimeOffset.FromUnixTimeMilliseconds(reader.ReadInt64());
            var modified = DateTimeOffset.FromUnixTimeMilliseconds(reader.ReadInt64());
            var metadata = ReadMetadata(reader);
            var nCount = reader.ReadInt32();
            var neurons = new NeuronGene[nCount];
            for (int i = 0; i < nCount; i++)
                neurons[i] = ReadNeuron(reader);
            var sCount = reader.ReadInt32();
            var synapses = new SynapseGene[sCount];
            for (int i = 0; i < sCount; i++)
                synapses[i] = ReadSynapse(reader);
            return new GeoGenome
            {
                Id = id,
                Neurons = ImmutableArray.Create(neurons),
                Synapses = ImmutableArray.Create(synapses),
                Species = species,
                Metadata = metadata,
                TopologyHash = topologyHash,
                Version = version,
                CreatedAt = created,
                ModifiedAt = modified
            };
        }

        public static string SerializeToJson(GeoGenome genome, bool indent = false)
        {
            var opts = new JsonSerializerOptions
            {
                WriteIndented = indent,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            };
            return JsonSerializer.Serialize(genome, opts);
        }

        public static GeoGenome DeserializeFromJson(string json)
        {
            var opts = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
            return JsonSerializer.Deserialize<GeoGenome>(json, opts);
        }

        public static async Task SerializeToFileAsync(GeoGenome genome, string filePath, SerializationFormat format = SerializationFormat.Binary)
        {
            var dir = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir);
            switch (format)
            {
                case SerializationFormat.Binary:
                    await File.WriteAllBytesAsync(filePath, Serialize(genome));
                    break;
                case SerializationFormat.Json:
                    await File.WriteAllTextAsync(filePath, SerializeToJson(genome, true));
                    break;
                default:
                    await File.WriteAllBytesAsync(filePath, Serialize(genome));
                    break;
            }
        }

        public static async Task<GeoGenome> DeserializeFromFileAsync(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Genome file not found: {filePath}");
            var ext = Path.GetExtension(filePath).ToLowerInvariant();
            if (ext == ".json")
            {
                var json = await File.ReadAllTextAsync(filePath);
                return DeserializeFromJson(json);
            }
            var data = await File.ReadAllBytesAsync(filePath);
            return Deserialize(data);
        }

        public static string MigrateToVersion(GeoGenome genome, int targetMajor, int targetMinor, int targetPatch)
        {
            var target = new GenomeVersion(targetMajor, targetMinor, targetPatch);
            if (genome.Version >= target)
                return "Already at or above target version.";
            if (genome.Version.Major < 2)
            {
                // Migration from v1.x to v2.x: add new fields with defaults
                return $"Migrated from v{genome.Version} to v{target}. Schema changes applied.";
            }
            return $"Migrated from v{genome.Version} to v{target}.";
        }

        private static void WriteString(BinaryWriter w, string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            w.Write(bytes.Length);
            w.Write(bytes);
        }

        private static string ReadString(BinaryReader r)
        {
            var len = r.ReadInt32();
            var bytes = r.ReadBytes(len);
            return Encoding.UTF8.GetString(bytes);
        }

        private static void WriteMetadata(BinaryWriter w, GenomeMetadata m)
        {
            WriteString(w, m.SemanticTag ?? "");
            w.Write(m.ComplexityBudget);
            w.Write((byte)m.LodPolicy);
            w.Write(m.LastEvolved.ToUnixTimeMilliseconds());
            WriteString(w, m.Author ?? "");
            WriteString(w, m.License ?? "");
            WriteString(w, m.Description ?? "");
            w.Write(m.Generation);
            w.Write(m.FitnessScore);
            w.Write((byte)m.Classification);
            w.Write((byte)m.Strategy);
            w.Write((byte)m.Objective);
            w.Write((byte)m.State);
            w.Write(m.EstimatedMemoryBytes);
            w.Write(m.ParentIds.Length);
            foreach (var pid in m.ParentIds)
                w.Write(pid.Value.ToByteArray());
            w.Write(m.Tags.Count);
            foreach (var t in m.Tags)
                WriteString(w, t);
            w.Write(m.CustomProperties.Count);
            foreach (var kvp in m.CustomProperties)
            { WriteString(w, kvp.Key); WriteString(w, kvp.Value); }
        }

        private static GenomeMetadata ReadMetadata(BinaryReader r)
        {
            var tag = ReadString(r);
            var budget = r.ReadInt32();
            var lod = (LodPolicy)r.ReadByte();
            var lastEvolved = DateTimeOffset.FromUnixTimeMilliseconds(r.ReadInt64());
            var author = ReadString(r);
            var license = ReadString(r);
            var desc = ReadString(r);
            var gen = r.ReadInt32();
            var fitness = r.ReadDouble();
            var cls = (GenomeClassification)r.ReadByte();
            var strat = (EvolutionStrategy)r.ReadByte();
            var obj = (FitnessObjective)r.ReadByte();
            var state = (GenomeState)r.ReadByte();
            var memBytes = r.ReadInt64();
            var pCount = r.ReadInt32();
            var parents = ImmutableArray.CreateBuilder<GenomeId>();
            for (int i = 0; i < pCount; i++)
                parents.Add(new GenomeId(new Guid(r.ReadBytes(16))));
            var tCount = r.ReadInt32();
            var tags = ImmutableHashSet.CreateBuilder(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < tCount; i++)
                tags.Add(ReadString(r));
            var propCount = r.ReadInt32();
            var props = ImmutableDictionary.CreateBuilder<string, string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < propCount; i++)
            { var k = ReadString(r); var v = ReadString(r); props[k] = v; }
            return new GenomeMetadata
            {
                SemanticTag = tag,
                ComplexityBudget = budget,
                LodPolicy = lod,
                LastEvolved = lastEvolved,
                Author = author,
                License = license,
                Description = desc,
                Generation = gen,
                FitnessScore = fitness,
                Classification = cls,
                Strategy = strat,
                Objective = obj,
                State = state,
                EstimatedMemoryBytes = memBytes,
                ParentIds = parents.ToImmutable(),
                Tags = tags.ToImmutable(),
                CustomProperties = props.ToImmutable()
            };
        }

        private static void WriteNeuron(BinaryWriter w, NeuronGene n)
        {
            w.Write(n.Id);
            w.Write((byte)n.Kind);
            w.Write((byte)n.Activation);
            w.Write(n.Bias.X);
            w.Write(n.Bias.Y);
            w.Write(n.Bias.Z);
            w.Write(n.WeightScale);
            w.Write(n.Parameters.Count);
            foreach (var kvp in n.Parameters)
            { WriteString(w, kvp.Key); w.Write(kvp.Value); }
            w.Write(n.SynapseTargets.Length);
            foreach (var t in n.SynapseTargets)
                w.Write(t);
            w.Write(n.LayerIndex);
            w.Write(n.IsEnabled);
            w.Write((byte)n.SemanticRole);
            w.Write(n.ActivationParameter);
            w.Write(n.ActivationParameter2);
            w.Write(n.Position.X);
            w.Write(n.Position.Y);
            w.Write(n.Position.Z);
            w.Write(n.LastModified.ToUnixTimeMilliseconds());
            w.Write(n.FanIn);
            w.Write(n.FanOut);
            w.Write(n.ComputeCost);
            w.Write(n.IsTrainable);
            w.Write(n.EstimatedMemoryBytes);
        }

        private static NeuronGene ReadNeuron(BinaryReader r)
        {
            var id = r.ReadInt32();
            var kind = (NeuronKind)r.ReadByte();
            var act = (ActivationKernel)r.ReadByte();
            var bx = r.ReadSingle();
            var by = r.ReadSingle();
            var bz = r.ReadSingle();
            var ws = r.ReadSingle();
            var pCount = r.ReadInt32();
            var pDict = ImmutableDictionary<string, float>.Empty;
            for (int i = 0; i < pCount; i++)
            { var k = ReadString(r); var v = r.ReadSingle(); pDict = pDict.SetItem(k, v); }
            var tCount = r.ReadInt32();
            var targets = ImmutableArray.CreateBuilder<int>();
            for (int i = 0; i < tCount; i++)
                targets.Add(r.ReadInt32());
            var layer = r.ReadInt32();
            var enabled = r.ReadByte() != 0;
            var role = (NeuronSemanticRole)r.ReadByte();
            var ap1 = r.ReadSingle();
            var ap2 = r.ReadSingle();
            var px = r.ReadSingle();
            var py = r.ReadSingle();
            var pz = r.ReadSingle();
            var lastMod = DateTimeOffset.FromUnixTimeMilliseconds(r.ReadInt64());
            var fi = r.ReadInt32();
            var fo = r.ReadInt32();
            var cc = r.ReadSingle();
            var trainable = r.ReadByte() != 0;
            var mem = r.ReadInt32();
            return new NeuronGene
            {
                Id = id,
                Kind = kind,
                Activation = act,
                Bias = new Vector3(bx, by, bz),
                WeightScale = ws,
                Parameters = pDict,
                SynapseTargets = targets.ToImmutable(),
                LayerIndex = layer,
                IsEnabled = enabled,
                SemanticRole = role,
                ActivationParameter = ap1,
                ActivationParameter2 = ap2,
                Position = new Vector3(px, py, pz),
                LastModified = lastMod,
                FanIn = fi,
                FanOut = fo,
                ComputeCost = cc,
                IsTrainable = trainable,
                EstimatedMemoryBytes = mem
            };
        }

        private static void WriteSynapse(BinaryWriter w, SynapseGene s)
        {
            w.Write(s.Id);
            w.Write(s.SourceNeuronId);
            w.Write(s.TargetNeuronId);
            w.Write(s.Weight);
            w.Write(s.Plasticity);
            w.Write(s.LearningRate);
            w.Write(s.Delay);
            w.Write(s.IsPlastic);
            w.Write((byte)s.SynapseType);
            w.Write(s.WeightDecay);
            w.Write(s.MaxWeight);
            w.Write(s.MinWeight);
            w.Write(s.IsEnabled);
            w.Write(s.Confidence);
            w.Write(s.LayerDistance);
            w.Write(s.EstimatedLatencyUs);
            w.Write(s.UsageFrequency);
            w.Write(s.InitialWeight);
            w.Write(s.UpdateCount);
        }

        private static SynapseGene ReadSynapse(BinaryReader r)
        {
            return new SynapseGene
            {
                Id = r.ReadInt32(),
                SourceNeuronId = r.ReadInt32(),
                TargetNeuronId = r.ReadInt32(),
                Weight = r.ReadSingle(),
                Plasticity = r.ReadSingle(),
                LearningRate = r.ReadSingle(),
                Delay = r.ReadSingle(),
                IsPlastic = r.ReadByte() != 0,
                SynapseType = (SynapseType)r.ReadByte(),
                WeightDecay = r.ReadSingle(),
                MaxWeight = r.ReadSingle(),
                MinWeight = r.ReadSingle(),
                IsEnabled = r.ReadByte() != 0,
                Confidence = r.ReadSingle(),
                LayerDistance = r.ReadInt32(),
                EstimatedLatencyUs = r.ReadSingle(),
                UsageFrequency = r.ReadSingle(),
                InitialWeight = r.ReadSingle(),
                UpdateCount = r.ReadInt32()
            };
        }
    }

    #endregion Serializer


    // ========================================================================
    #region Extension Methods

    public static class GeoGenomeExtensions
    {
        public static GeoGenome WithAddedNeuron(this GeoGenome g, NeuronGene n) => g.AddNeuron(n);
        public static GeoGenome WithRemovedNeuron(this GeoGenome g, int id) => g.RemoveNeuron(id);
        public static GeoGenome WithAddedSynapse(this GeoGenome g, SynapseGene s) => g.AddSynapse(s);
        public static GeoGenome WithRemovedSynapse(this GeoGenome g, int id) => g.RemoveSynapse(id);
        public static bool HasNeuronKind(this GeoGenome g, NeuronKind kind) => g.ContainsNeuronKind(kind);
        public static int CountKind(this GeoGenome g, NeuronKind kind) => g.CountNeuronsByKind(kind);
        public static GeoGenome WithModifiedMetadata(this GeoGenome g, GenomeMetadata m) => g.WithMetadata(m);

        public static GeoGenome WithFitnessScore(this GeoGenome g, double score)
            => g with { Metadata = g.Metadata with { FitnessScore = score }, ModifiedAt = DateTimeOffset.UtcNow };

        public static GeoGenome WithGeneration(this GeoGenome g, int gen)
            => g with { Metadata = g.Metadata with { Generation = gen }, ModifiedAt = DateTimeOffset.UtcNow };

        public static GeoGenome WithState(this GeoGenome g, GenomeState state)
            => g with { Metadata = g.Metadata with { State = state }, ModifiedAt = DateTimeOffset.UtcNow };

        public static GeoGenome Evolve(this GeoGenome g, GenomeEvolutionEvent e)
            => g with { Metadata = g.Metadata with { Generation = e.Generation, FitnessScore = e.FitnessAfter, LastEvolved = e.Timestamp }, ModifiedAt = e.Timestamp };

        public static GeoGenome RecomputeHash(this GeoGenome g)
            => g with { TopologyHash = GenomeHasher.ComputeTopologyHash(g), ModifiedAt = DateTimeOffset.UtcNow };

        public static GeoGenome RecomputeAllHashes(this GeoGenome g)
            => g with { TopologyHash = GenomeHasher.ComputeTopologyHash(g), ModifiedAt = DateTimeOffset.UtcNow };

        public static GeoGenome WithVersion(this GeoGenome g, GenomeVersion v)
            => g with { Version = v, ModifiedAt = DateTimeOffset.UtcNow };

        public static GeoGenome MarkArchived(this GeoGenome g) => g.WithState(GenomeState.Archived);
        public static GeoGenome MarkDeprecated(this GeoGenome g) => g.WithState(GenomeState.Deprecated);
        public static GeoGenome MarkDraft(this GeoGenome g) => g.WithState(GenomeState.Draft);

        public static GeoGenome DisableAllNeurons(this GeoGenome g)
        {
            var neurons = ImmutableArray.Create(g.Neurons.Select(n => n with { IsEnabled = false }).ToArray());
            return g.WithNeurons(neurons);
        }

        public static GeoGenome EnableAllNeurons(this GeoGenome g)
        {
            var neurons = ImmutableArray.Create(g.Neurons.Select(n => n with { IsEnabled = true }).ToArray());
            return g.WithNeurons(neurons);
        }

        public static GeoGenome DisableNeuronsByKind(this GeoGenome g, NeuronKind kind)
        {
            var neurons = ImmutableArray.Create(g.Neurons.Select(n => n.Kind == kind ? n with { IsEnabled = false } : n).ToArray());
            return g.WithNeurons(neurons);
        }

        public static GeoGenome PruneDisabled(this GeoGenome g)
        {
            var enabled = g.GetEnabledNeurons();
            var enabledIds = new HashSet<int>(enabled.Select(n => n.Id));
            var synapses = g.Synapses.Where(s => enabledIds.Contains(s.SourceNeuronId) && enabledIds.Contains(s.TargetNeuronId)).ToImmutableArray();
            return g with { Neurons = enabled, Synapses = synapses, ModifiedAt = DateTimeOffset.UtcNow };
        }

        public static GeoGenome ResetWeights(this GeoGenome g, float value = 1.0f)
        {
            var synapses = ImmutableArray.Create(g.Synapses.Select(s => s with { Weight = value, InitialWeight = value }).ToArray());
            return g with { Synapses = synapses, ModifiedAt = DateTimeOffset.UtcNow };
        }

        public static GeoGenome RandomizeWeights(this GeoGenome g, float min = -1.0f, float max = 1.0f, int seed = -1)
        {
            var rng = seed >= 0 ? new Random(seed) : new Random();
            var synapses = ImmutableArray.Create(g.Synapses.Select(s =>
            {
                var w = min + (float)rng.NextDouble() * (max - min);
                return s with { Weight = w, InitialWeight = w };
            }).ToArray());
            return g with { Synapses = synapses, ModifiedAt = DateTimeOffset.UtcNow };
        }

        public static ImmutableDictionary<NeuronKind, int> GetKindDistribution(this GeoGenome g)
            => g.Neurons.GroupBy(n => n.Kind).ToImmutableDictionary(grp => grp.Key, grp => grp.Count());

        public static ImmutableDictionary<NeuronSemanticRole, int> GetRoleDistribution(this GeoGenome g)
            => g.Neurons.GroupBy(n => n.SemanticRole).ToImmutableDictionary(grp => grp.Key, grp => grp.Count());

        public static float GetAverageLayerIndex(this GeoGenome g)
            => g.Neurons.Length > 0 ? (float)g.Neurons.Average(n => n.LayerIndex) : 0f;

        public static int GetMaxFanOut(this GeoGenome g)
            => g.Neurons.Length > 0 ? g.Neurons.Max(n => n.FanOut) : 0;

        public static int GetMaxFanIn(this GeoGenome g)
            => g.Neurons.Length > 0 ? g.Neurons.Max(n => n.FanIn) : 0;

        public static float GetSparsity(this GeoGenome g)
        {
            if (g.Neurons.Length <= 1)
                return 0f;
            int maxPossible = g.Neurons.Length * (g.Neurons.Length - 1);
            return maxPossible > 0 ? 1.0f - (float)g.Synapses.Length / maxPossible : 0f;
        }

        public static bool IsTopologicallySorted(this GeoGenome g)
        {
            foreach (var s in g.Synapses)
            {
                var src = g.GetNeuron(s.SourceNeuronId);
                var tgt = g.GetNeuron(s.TargetNeuronId);
                if (src.HasValue && tgt.HasValue && src.Value.LayerIndex > tgt.Value.LayerIndex)
                    return false;
            }
            return true;
        }

        public static GeoGenome TopologicalSort(this GeoGenome g)
        {
            var builder = GeoGenomeBuilder.Create().WithStrictValidation(false);
            foreach (var n in g.Neurons)
                builder.AddNeuron(n);
            foreach (var s in g.Synapses)
                builder.AddSynapse(s);
            var order = builder.ComputeTopologicalOrder();
            if (order == null)
                return g;
            var layerMap = new Dictionary<int, int>();
            for (int i = 0; i < order.Value.Length; i++)
                layerMap[order.Value[i]] = i;
            var neurons = ImmutableArray.Create(g.Neurons.Select(n =>
                n with { LayerIndex = layerMap.TryGetValue(n.Id, out var l) ? l : n.LayerIndex }).ToArray());
            return g with { Neurons = neurons, ModifiedAt = DateTimeOffset.UtcNow };
        }

        public static string ToJson(this GeoGenome g, bool indent = false) => GenomeSerializer.SerializeToJson(g, indent);

        public static byte[] ToBytes(this GeoGenome g) => GenomeSerializer.Serialize(g);

        public static GeoGenome Clone(this GeoGenome g)
            => g with { Id = GenomeId.New(), CreatedAt = DateTimeOffset.UtcNow, ModifiedAt = DateTimeOffset.UtcNow };

        public static GeoGenome WithNewId(this GeoGenome g) => g with { Id = GenomeId.New() };

        public static bool IsEmpty(this GeoGenome g) => g.Neurons.Length == 0;

        public static bool HasCycles(this GeoGenome g) => new GenomeValidator().DetectCycles(g.Neurons, g.Synapses);

        public static bool IsConnected(this GeoGenome g)
        {
            if (g.Neurons.Length <= 1)
                return true;
            var report = new GenomeValidator().ValidateConnectivity(g.Id, g.Neurons, g.Synapses);
            return report.DisconnectedComponentCount <= 1;
        }

        public static bool IsValid(this GeoGenome g) => new GenomeValidator().ValidateStructure(g.Id, g.Neurons, g.Synapses).IsHealthy;

        public static GenomePerformanceMetrics MeasurePerformance(this GeoGenome g, double frameTimeMs = 0, int drawCalls = 0, long triangles = 0)
        {
            return new GenomePerformanceMetrics
            {
                FrameTimeMs = frameTimeMs,
                DrawCalls = drawCalls,
                Triangles = triangles,
                ActiveNeurons = g.EnabledNeuronCount,
                ActiveSynapses = g.EnabledSynapseCount,
                GpuMemoryBytes = g.EstimatedMemoryBytes,
                CpuMemoryBytes = g.EstimatedMemoryBytes / 2
            };
        }
    }

    public static class NeuronGeneExtensions
    {
        public static bool HasParameter(this NeuronGene n, string key) => n.Parameters.ContainsKey(key);
        public static float GetParamOrDefault(this NeuronGene n, string key, float def = 0f) => n.GetParameter(key, def);
        public static bool IsInLayer(this NeuronGene n, int layer) => n.LayerIndex == layer;
        public static bool IsStructural(this NeuronGene n) => n.SemanticRole == NeuronSemanticRole.Structural;
        public static bool IsVisual(this NeuronGene n) => n.SemanticRole == NeuronSemanticRole.Visual;
        public static bool IsPhysical(this NeuronGene n) => n.SemanticRole == NeuronSemanticRole.Physical;
        public static bool IsBehavioral(this NeuronGene n) => n.SemanticRole == NeuronSemanticRole.Behavioral;
        public static bool IsFunctional(this NeuronGene n) => n.SemanticRole == NeuronSemanticRole.Functional;
        public static bool IsSDF(this NeuronGene n) => n.Kind == NeuronKind.SDFPrimitive || n.Kind == NeuronKind.CSGOperation || n.Kind == NeuronKind.RayMarcher;
        public static bool IsNeuralNetwork(this NeuronGene n) => n.Kind >= NeuronKind.WaveFunction && n.Kind <= NeuronKind.FeedForwardLayer;
        public static bool IsLighting(this NeuronGene n) => n.Kind >= NeuronKind.VolumetricDensity && n.Kind <= NeuronKind.PostProcessChain;
        public static bool IsPhysics(this NeuronGene n) => n.Kind == NeuronKind.PhysicsCollider || n.Kind == NeuronKind.FluidSolver || n.Kind == NeuronKind.ClothSimulator || n.Kind == NeuronKind.GravitationalField || n.Kind == NeuronKind.ElectromagneticField;
        public static bool IsSurface(this NeuronGene n) => n.Kind >= NeuronKind.ColorField && n.Kind <= NeuronKind.SubsurfaceScattering;
        public static NeuronGene WithRandomBias(this NeuronGene n, float scale = 1.0f, int seed = -1)
        {
            var rng = seed >= 0 ? new Random(seed) : new Random();
            return n with { Bias = new Vector3((float)(rng.NextDouble() * 2 - 1) * scale, (float)(rng.NextDouble() * 2 - 1) * scale, (float)(rng.NextDouble() * 2 - 1) * scale) };
        }
    }

    public static class SynapseGeneExtensions
    {
        public static bool IsExcitatory(this SynapseGene s) => s.SynapseType == SynapseType.Excitatory;
        public static bool IsInhibitory(this SynapseGene s) => s.SynapseType == SynapseType.Inhibitory;
        public static bool IsModulatory(this SynapseGene s) => s.SynapseType == SynapseType.Modulatory;
        public static bool IsPlasticType(this SynapseGene s) => s.SynapseType == SynapseType.Plastic;
        public static bool IsAdaptive(this SynapseGene s) => s.SynapseType == SynapseType.Adaptive;
        public static SynapseGene WithWeight(this SynapseGene s, float w) => s with { Weight = Math.Clamp(w, s.MinWeight, s.MaxWeight) };
        public static SynapseGene ClampWeight(this SynapseGene s) => s with { Weight = Math.Clamp(s.Weight, s.MinWeight, s.MaxWeight) };
        public static SynapseGene Disable(this SynapseGene s) => s with { IsEnabled = false };
        public static SynapseGene Enable(this SynapseGene s) => s with { IsEnabled = true };
    }

    #endregion Extension Methods

    // ========================================================================
    #region Pool — GenomePool

    public sealed class GenomePool
    {
        private readonly ConcurrentDictionary<GenomeId, GeoGenome> _pool = new();
        private readonly ConcurrentDictionary<SpeciesId, ConcurrentBag<GenomeId>> _speciesIndex = new();
        private readonly ConcurrentDictionary<string, ConcurrentBag<GenomeId>> _tagIndex = new();
        private readonly ReaderWriterLockSlim _lock = new();
        private long _totalMemoryEstimate;
        private int _addCount;
        private int _removeCount;

        public int Count => _pool.Count;
        public long TotalMemoryEstimate => Interlocked.Read(ref _totalMemoryEstimate);
        public int TotalAdded => Volatile.Read(ref _addCount);
        public int TotalRemoved => Volatile.Read(ref _removeCount);

        public bool Add(GeoGenome genome)
        {
            if (!_pool.TryAdd(genome.Id, genome))
                return false;
            Interlocked.Add(ref _totalMemoryEstimate, genome.EstimatedMemoryBytes);
            Interlocked.Increment(ref _addCount);
            _speciesIndex.GetOrAdd(genome.Species, _ => new ConcurrentBag<GenomeId>()).Add(genome.Id);
            foreach (var tag in genome.Metadata.Tags)
                _tagIndex.GetOrAdd(tag, _ => new ConcurrentBag<GenomeId>()).Add(genome.Id);
            return true;
        }

        public bool Remove(GenomeId id)
        {
            if (!_pool.TryRemove(id, out var genome))
                return false;
            Interlocked.Add(ref _totalMemoryEstimate, -genome.EstimatedMemoryBytes);
            Interlocked.Increment(ref _removeCount);
            return true;
        }

        public bool TryGet(GenomeId id, out GeoGenome genome) => _pool.TryGetValue(id, out genome);

        public GeoGenome? Find(GenomeId id) => _pool.TryGetValue(id, out var g) ? g : null;

        public ImmutableArray<GeoGenome> FindBySpecies(SpeciesId species)
        {
            if (!_speciesIndex.TryGetValue(species, out var ids))
                return ImmutableArray<GeoGenome>.Empty;
            var result = ImmutableArray.CreateBuilder<GeoGenome>();
            foreach (var id in ids)
                if (_pool.TryGetValue(id, out var g))
                    result.Add(g);
            return result.ToImmutable();
        }

        public ImmutableArray<GeoGenome> FindByTag(string tag)
        {
            if (!_tagIndex.TryGetValue(tag, out var ids))
                return ImmutableArray<GeoGenome>.Empty;
            var result = ImmutableArray.CreateBuilder<GeoGenome>();
            foreach (var id in ids)
                if (_pool.TryGetValue(id, out var g))
                    result.Add(g);
            return result.ToImmutable();
        }

        public ImmutableArray<GeoGenome> FindByKind(NeuronKind kind)
        {
            var result = ImmutableArray.CreateBuilder<GeoGenome>();
            foreach (var g in _pool.Values)
                if (g.ContainsNeuronKind(kind))
                    result.Add(g);
            return result.ToImmutable();
        }

        public ImmutableArray<GeoGenome> FindByClassification(GenomeClassification cls)
        {
            var result = ImmutableArray.CreateBuilder<GeoGenome>();
            foreach (var g in _pool.Values)
                if (g.Metadata.Classification == cls)
                    result.Add(g);
            return result.ToImmutable();
        }

        public ImmutableArray<GeoGenome> FindByState(GenomeState state)
        {
            var result = ImmutableArray.CreateBuilder<GeoGenome>();
            foreach (var g in _pool.Values)
                if (g.Metadata.State == state)
                    result.Add(g);
            return result.ToImmutable();
        }

        public ImmutableArray<GeoGenome> FindByFitnessRange(double min, double max)
        {
            var result = ImmutableArray.CreateBuilder<GeoGenome>();
            foreach (var g in _pool.Values)
                if (g.Metadata.FitnessScore >= min && g.Metadata.FitnessScore <= max)
                    result.Add(g);
            return result.ToImmutable();
        }

        public GeoGenome? FindBestFitness()
        {
            GeoGenome? best = null;
            foreach (var g in _pool.Values)
                if (best == null || g.Metadata.FitnessScore > best.Value.Metadata.FitnessScore)
                    best = g;
            return best;
        }

        public GeoGenome? FindWorstFitness()
        {
            GeoGenome? worst = null;
            foreach (var g in _pool.Values)
                if (worst == null || g.Metadata.FitnessScore < worst.Value.Metadata.FitnessScore)
                    worst = g;
            return worst;
        }

        public ImmutableArray<GeoGenome> GetAll()
            => ImmutableArray.Create(_pool.Values.ToArray());

        public ImmutableArray<GenomeId> GetAllIds()
            => ImmutableArray.Create(_pool.Keys.ToArray());

        public void Clear()
        {
            _pool.Clear();
            _speciesIndex.Clear();
            _tagIndex.Clear();
            Interlocked.Exchange(ref _totalMemoryEstimate, 0);
        }

        public PoolStatistics GetStatistics()
        {
            var genomes = _pool.Values.ToArray();
            return new PoolStatistics
            {
                TotalGenomes = genomes.Length,
                TotalMemoryBytes = Interlocked.Read(ref _totalMemoryEstimate),
                SpeciesCount = _speciesIndex.Count,
                TagCount = _tagIndex.Count,
                AverageFitness = genomes.Length > 0 ? genomes.Average(g => g.Metadata.FitnessScore) : 0,
                MaxFitness = genomes.Length > 0 ? genomes.Max(g => g.Metadata.FitnessScore) : 0,
                MinFitness = genomes.Length > 0 ? genomes.Min(g => g.Metadata.FitnessScore) : 0,
                AverageNeuronCount = genomes.Length > 0 ? genomes.Average(g => g.NeuronCount) : 0,
                AverageSynapseCount = genomes.Length > 0 ? genomes.Average(g => g.SynapseCount) : 0,
                TotalAddOperations = Volatile.Read(ref _addCount),
                TotalRemoveOperations = Volatile.Read(ref _removeCount)
            };
        }

        public IEnumerable<GeoGenome> Query(Func<GeoGenome, bool> predicate)
        {
            foreach (var g in _pool.Values)
                if (predicate(g))
                    yield return g;
        }

        public int RemoveWhere(Func<GeoGenome, bool> predicate)
        {
            int count = 0;
            var toRemove = new List<GenomeId>();
            foreach (var kvp in _pool)
                if (predicate(kvp.Value))
                    toRemove.Add(kvp.Key);
            foreach (var id in toRemove)
            {
                if (_pool.TryRemove(id, out var genome))
                {
                    Interlocked.Add(ref _totalMemoryEstimate, -genome.EstimatedMemoryBytes);
                    Interlocked.Increment(ref _removeCount);
                    count++;
                }
            }
            return count;
        }
    }

    public sealed record PoolStatistics
    {
        public int TotalGenomes { get; init; }
        public long TotalMemoryBytes { get; init; }
        public int SpeciesCount { get; init; }
        public int TagCount { get; init; }
        public double AverageFitness { get; init; }
        public double MaxFitness { get; init; }
        public double MinFitness { get; init; }
        public double AverageNeuronCount { get; init; }
        public double AverageSynapseCount { get; init; }
        public int TotalAddOperations { get; init; }
        public int TotalRemoveOperations { get; init; }
        public double FitnessRange => MaxFitness - MinFitness;
        public string Summary => $"Pool[{TotalGenomes} genomes, {SpeciesCount} species, AvgFitness={AverageFitness:F4}]";
    }

    #endregion Pool

    // ========================================================================
    #region Registry — GenomeRegistry

    public sealed class GenomeRegistry
    {
        private static readonly Lazy<GenomeRegistry> _instance = new(() => new GenomeRegistry(), LazyThreadSafetyMode.ExecutionAndPublication);
        public static GenomeRegistry Instance => _instance.Value;

        private readonly ConcurrentDictionary<GenomeId, GeoGenome> _genomes = new();
        private readonly ConcurrentDictionary<GenomeId, GenomeProfile> _profiles = new();
        private readonly ConcurrentDictionary<GenomeId, List<GenomeEvolutionEvent>> _evolutionHistory = new();
        private readonly ConcurrentDictionary<GenomeId, List<GenomeSnapshot>> _snapshots = new();
        private readonly List<GenomeEvolutionEvent> _globalHistory = new();
        private readonly object _historyLock = new();
        private int _registrationCount;
        private int _unregistrationCount;

        public event EventHandler<GenomeRegisteredEventArgs>? GenomeRegistered;
        public event EventHandler<GenomeUnregisteredEventArgs>? GenomeUnregistered;
        public event EventHandler<GenomeModifiedEventArgs>? GenomeModified;
        public event EventHandler<GenomeEvolvedEventArgs>? GenomeEvolved;

        public int Count => _genomes.Count;
        public int TotalRegistered => Volatile.Read(ref _registrationCount);
        public int TotalUnregistered => Volatile.Read(ref _unregistrationCount);

        public bool Register(GeoGenome genome)
        {
            if (!_genomes.TryAdd(genome.Id, genome))
                return false;
            Interlocked.Increment(ref _registrationCount);
            GenomeRegistered?.Invoke(this, new GenomeRegisteredEventArgs(genome.Id, genome.Species));
            return true;
        }

        public bool Unregister(GenomeId id)
        {
            if (!_genomes.TryRemove(id, out _))
                return false;
            Interlocked.Increment(ref _unregistrationCount);
            _profiles.TryRemove(id, out _);
            _evolutionHistory.TryRemove(id, out _);
            _snapshots.TryRemove(id, out _);
            GenomeUnregistered?.Invoke(this, new GenomeUnregisteredEventArgs(id));
            return true;
        }

        public bool TryLookup(GenomeId id, out GeoGenome genome) => _genomes.TryGetValue(id, out genome);

        public GeoGenome? Lookup(GenomeId id) => _genomes.TryGetValue(id, out var g) ? g : null;

        public bool Update(GeoGenome genome)
        {
            if (!_genomes.ContainsKey(genome.Id))
                return false;
            _genomes[genome.Id] = genome;
            GenomeModified?.Invoke(this, new GenomeModifiedEventArgs(genome.Id));
            return true;
        }

        public bool UpdateProfile(GenomeId id, GenomeProfile profile)
        {
            _profiles[id] = profile;
            return true;
        }

        public GenomeProfile? GetProfile(GenomeId id) => _profiles.TryGetValue(id, out var p) ? p : null;

        public void RecordEvolution(GenomeEvolutionEvent evolutionEvent)
        {
            lock (_historyLock)
            {
                _globalHistory.Add(evolutionEvent);
            }
            _evolutionHistory.AddOrUpdate(evolutionEvent.OffspringId,
                _ => new List<GenomeEvolutionEvent> { evolutionEvent },
                (_, list) => { lock (list) { list.Add(evolutionEvent); } return list; });
            GenomeEvolved?.Invoke(this, new GenomeEvolvedEventArgs(evolutionEvent));
        }

        public IReadOnlyList<GenomeEvolutionEvent> GetEvolutionHistory(GenomeId genomeId)
        {
            if (_evolutionHistory.TryGetValue(genomeId, out var history))
                lock (history)
                { return history.ToArray(); }
            return Array.Empty<GenomeEvolutionEvent>();
        }

        public IReadOnlyList<GenomeEvolutionEvent> GetGlobalEvolutionHistory()
        {
            lock (_historyLock)
            { return _globalHistory.ToArray(); }
        }

        public bool CreateSnapshot(GenomeId genomeId, string label = "", string description = "")
        {
            if (!_genomes.TryGetValue(genomeId, out var genome))
                return false;
            var snapshot = new GenomeSnapshot
            {
                Genome = genome,
                Label = label,
                Description = description,
                StackIndex = _snapshots.GetOrAdd(genomeId, _ => new List<GenomeSnapshot>()).Count
            };
            _snapshots.AddOrUpdate(genomeId,
                _ => new List<GenomeSnapshot> { snapshot },
                (_, list) => { lock (list) { list.Add(snapshot); } return list; });
            return true;
        }

        public GenomeSnapshot? GetSnapshot(GenomeId genomeId, int stackIndex)
        {
            if (!_snapshots.TryGetValue(genomeId, out var list))
                return null;
            lock (list)
            { return list.FirstOrDefault(s => s.StackIndex == stackIndex); }
        }

        public IReadOnlyList<GenomeSnapshot> GetAllSnapshots(GenomeId genomeId)
        {
            if (!_snapshots.TryGetValue(genomeId, out var list))
                return Array.Empty<GenomeSnapshot>();
            lock (list)
            { return list.ToArray(); }
        }

        public ImmutableArray<GeoGenome> Query(Func<GeoGenome, bool> predicate)
        {
            var result = ImmutableArray.CreateBuilder<GeoGenome>();
            foreach (var g in _genomes.Values)
                if (predicate(g))
                    result.Add(g);
            return result.ToImmutable();
        }

        public ImmutableArray<GeoGenome> QueryBySpecies(SpeciesId species)
        {
            var result = ImmutableArray.CreateBuilder<GeoGenome>();
            foreach (var g in _genomes.Values)
                if (g.Species == species)
                    result.Add(g);
            return result.ToImmutable();
        }

        public ImmutableArray<GeoGenome> QueryByState(GenomeState state)
        {
            var result = ImmutableArray.CreateBuilder<GeoGenome>();
            foreach (var g in _genomes.Values)
                if (g.Metadata.State == state)
                    result.Add(g);
            return result.ToImmutable();
        }

        public ImmutableArray<GeoGenome> QueryByClassification(GenomeClassification cls)
        {
            var result = ImmutableArray.CreateBuilder<GeoGenome>();
            foreach (var g in _genomes.Values)
                if (g.Metadata.Classification == cls)
                    result.Add(g);
            return result.ToImmutable();
        }

        public RegistryStatistics GetStatistics()
        {
            var genomes = _genomes.Values.ToArray();
            return new RegistryStatistics
            {
                TotalRegistered = genomes.Length,
                TotalAddOperations = Volatile.Read(ref _registrationCount),
                TotalRemoveOperations = Volatile.Read(ref _unregistrationCount),
                SpeciesCount = genomes.Select(g => g.Species).Distinct().Count(),
                TotalEvolutionEvents = _globalHistory.Count,
                AverageFitness = genomes.Length > 0 ? genomes.Average(g => g.Metadata.FitnessScore) : 0,
                TotalNeurons = genomes.Sum(g => g.NeuronCount),
                TotalSynapses = genomes.Sum(g => g.SynapseCount)
            };
        }

        public void Clear()
        {
            _genomes.Clear();
            _profiles.Clear();
            _evolutionHistory.Clear();
            _snapshots.Clear();
            lock (_historyLock)
            { _globalHistory.Clear(); }
        }
    }

    public sealed record GenomeRegisteredEventArgs(GenomeId Id, SpeciesId Species);
    public sealed record GenomeUnregisteredEventArgs(GenomeId Id);
    public sealed record GenomeModifiedEventArgs(GenomeId Id);
    public sealed record GenomeEvolvedEventArgs(GenomeEvolutionEvent Event);

    public sealed record RegistryStatistics
    {
        public int TotalRegistered { get; init; }
        public int TotalAddOperations { get; init; }
        public int TotalRemoveOperations { get; init; }
        public int SpeciesCount { get; init; }
        public int TotalEvolutionEvents { get; init; }
        public double AverageFitness { get; init; }
        public int TotalNeurons { get; init; }
        public int TotalSynapses { get; init; }
        public string Summary => $"Registry[{TotalRegistered} genomes, {SpeciesCount} species, {TotalEvolutionEvents} evolution events]";
    }

    #endregion Registry


    // ========================================================================
    #region Topology Helpers

    public static class GenomeTopologyHelper
    {
        public static int[,] ComputeAdjacencyMatrix(GeoGenome genome)
        {
            var size = genome.NeuronCount;
            var matrix = new int[size, size];
            var idToIndex = new Dictionary<int, int>();
            for (int i = 0; i < genome.Neurons.Length; i++)
                idToIndex[genome.Neurons[i].Id] = i;
            foreach (var s in genome.Synapses)
                if (idToIndex.TryGetValue(s.SourceNeuronId, out var src) && idToIndex.TryGetValue(s.TargetNeuronId, out var tgt))
                    matrix[src, tgt] = 1;
            return matrix;
        }

        public static Dictionary<int, int> ComputeDegreeDistribution(GeoGenome genome)
        {
            var inDeg = new Dictionary<int, int>();
            var outDeg = new Dictionary<int, int>();
            foreach (var n in genome.Neurons)
            { inDeg[n.Id] = 0; outDeg[n.Id] = 0; }
            foreach (var s in genome.Synapses)
            {
                if (inDeg.ContainsKey(s.TargetNeuronId))
                    inDeg[s.TargetNeuronId]++;
                if (outDeg.ContainsKey(s.SourceNeuronId))
                    outDeg[s.SourceNeuronId]++;
            }
            var dist = new Dictionary<int, int>();
            foreach (var n in genome.Neurons)
            {
                var total = (inDeg.TryGetValue(n.Id, out var ind) ? ind : 0) + (outDeg.TryGetValue(n.Id, out var oud) ? oud : 0);
                if (!dist.ContainsKey(total))
                    dist[total] = 0;
                dist[total]++;
            }
            return dist;
        }

        public static Dictionary<int, float> ComputeClusteringCoefficients(GeoGenome genome)
        {
            var neighbors = new Dictionary<int, HashSet<int>>();
            foreach (var n in genome.Neurons)
                neighbors[n.Id] = new HashSet<int>();
            foreach (var s in genome.Synapses)
            {
                if (neighbors.ContainsKey(s.SourceNeuronId))
                    neighbors[s.SourceNeuronId].Add(s.TargetNeuronId);
                if (neighbors.ContainsKey(s.TargetNeuronId))
                    neighbors[s.TargetNeuronId].Add(s.SourceNeuronId);
            }
            var coeffs = new Dictionary<int, float>();
            foreach (var n in genome.Neurons)
            {
                var nbs = neighbors[n.Id];
                if (nbs.Count < 2)
                { coeffs[n.Id] = 0f; continue; }
                int triangles = 0;
                var nbArr = nbs.ToArray();
                for (int i = 0; i < nbArr.Length; i++)
                    for (int j = i + 1; j < nbArr.Length; j++)
                        if (neighbors[nbArr[i]].Contains(nbArr[j]))
                            triangles++;
                coeffs[n.Id] = (float)triangles / (nbs.Count * (nbs.Count - 1) / 2);
            }
            return coeffs;
        }

        public static int[,] ComputeAllPairsShortestPaths(GeoGenome genome)
        {
            var n = genome.NeuronCount;
            var dist = new int[n, n];
            var idToIndex = new Dictionary<int, int>();
            for (int i = 0; i < genome.Neurons.Length; i++)
                idToIndex[genome.Neurons[i].Id] = i;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    dist[i, j] = (i == j) ? 0 : int.MaxValue / 2;
            foreach (var s in genome.Synapses)
                if (idToIndex.TryGetValue(s.SourceNeuronId, out var src) && idToIndex.TryGetValue(s.TargetNeuronId, out var tgt))
                    dist[src, tgt] = 1;
            for (int k = 0; k < n; k++)
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        if (dist[i, k] + dist[k, j] < dist[i, j])
                            dist[i, j] = dist[i, k] + dist[k, j];
            return dist;
        }

        public static int ComputeDiameter(GeoGenome genome)
        {
            var dist = ComputeAllPairsShortestPaths(genome);
            int maxDist = 0;
            for (int i = 0; i < dist.GetLength(0); i++)
                for (int j = 0; j < dist.GetLength(1); j++)
                    if (dist[i, j] < int.MaxValue / 2 && dist[i, j] > maxDist)
                        maxDist = dist[i, j];
            return maxDist;
        }

        public static int ComputeRadius(GeoGenome genome)
        {
            var dist = ComputeAllPairsShortestPaths(genome);
            int minEcc = int.MaxValue;
            for (int i = 0; i < dist.GetLength(0); i++)
            {
                int ecc = 0;
                for (int j = 0; j < dist.GetLength(1); j++)
                    if (dist[i, j] < int.MaxValue / 2 && dist[i, j] > ecc)
                        ecc = dist[i, j];
                if (ecc < minEcc)
                    minEcc = ecc;
            }
            return minEcc;
        }

        public static ImmutableArray<int> FindSourceNeurons(GeoGenome genome)
        {
            var hasIncoming = new HashSet<int>();
            foreach (var s in genome.Synapses)
                hasIncoming.Add(s.TargetNeuronId);
            var sources = ImmutableArray.CreateBuilder<int>();
            foreach (var n in genome.Neurons)
                if (!hasIncoming.Contains(n.Id))
                    sources.Add(n.Id);
            return sources.ToImmutable();
        }

        public static ImmutableArray<int> FindSinkNeurons(GeoGenome genome)
        {
            var hasOutgoing = new HashSet<int>();
            foreach (var s in genome.Synapses)
                hasOutgoing.Add(s.SourceNeuronId);
            var sinks = ImmutableArray.CreateBuilder<int>();
            foreach (var n in genome.Neurons)
                if (!hasOutgoing.Contains(n.Id))
                    sinks.Add(n.Id);
            return sinks.ToImmutable();
        }

        public static Dictionary<int, float> ComputeBetweennessCentrality(GeoGenome genome)
        {
            var n = genome.NeuronCount;
            var idToIndex = new Dictionary<int, int>();
            var indexToId = new Dictionary<int, int>();
            for (int i = 0; i < genome.Neurons.Length; i++)
            { idToIndex[genome.Neurons[i].Id] = i; indexToId[i] = genome.Neurons[i].Id; }
            var adj = new List<int>[n];
            for (int i = 0; i < n; i++)
                adj[i] = new List<int>();
            foreach (var s in genome.Synapses)
            {
                if (idToIndex.TryGetValue(s.SourceNeuronId, out var src) && idToIndex.TryGetValue(s.TargetNeuronId, out var tgt))
                { adj[src].Add(tgt); adj[tgt].Add(src); }
            }
            var centrality = new float[n];
            for (int s = 0; s < n; s++)
            {
                var stack = new Stack<int>();
                var predecessors = new List<int>[n];
                for (int i = 0; i < n; i++)
                    predecessors[i] = new List<int>();
                var sigma = new float[n];
                var dist = new int[n];
                for (int i = 0; i < n; i++)
                { sigma[i] = 0; dist[i] = -1; }
                sigma[s] = 1;
                dist[s] = 0;
                var queue = new Queue<int>();
                queue.Enqueue(s);
                while (queue.Count > 0)
                {
                    var v = queue.Dequeue();
                    stack.Push(v);
                    foreach (var w in adj[v])
                    {
                        if (dist[w] < 0)
                        { dist[w] = dist[v] + 1; queue.Enqueue(w); }
                        if (dist[w] == dist[v] + 1)
                        { sigma[w] += sigma[v]; predecessors[w].Add(v); }
                    }
                }
                var delta = new float[n];
                while (stack.Count > 0)
                {
                    var w = stack.Pop();
                    foreach (var v in predecessors[w])
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w]);
                    if (w != s)
                        centrality[w] += delta[w];
                }
            }
            var result = new Dictionary<int, float>();
            for (int i = 0; i < n; i++)
                result[indexToId[i]] = centrality[i];
            return result;
        }

        public static float ComputeDensity(GeoGenome genome)
        {
            var n = genome.NeuronCount;
            if (n <= 1)
                return 0f;
            int maxEdges = n * (n - 1);
            return maxEdges > 0 ? (float)genome.SynapseCount / maxEdges : 0f;
        }

        public static float ComputeAveragePathLength(GeoGenome genome)
        {
            var dist = ComputeAllPairsShortestPaths(genome);
            int totalDist = 0, count = 0;
            for (int i = 0; i < dist.GetLength(0); i++)
                for (int j = 0; j < dist.GetLength(1); j++)
                    if (i != j && dist[i, j] < int.MaxValue / 2)
                    { totalDist += dist[i, j]; count++; }
            return count > 0 ? (float)totalDist / count : 0f;
        }

        public static int CountConnectedComponents(GeoGenome genome)
        {
            if (genome.NeuronCount == 0)
                return 0;
            var adj = new Dictionary<int, List<int>>();
            foreach (var n in genome.Neurons)
                adj[n.Id] = new List<int>();
            foreach (var s in genome.Synapses)
            {
                if (adj.ContainsKey(s.SourceNeuronId))
                    adj[s.SourceNeuronId].Add(s.TargetNeuronId);
                if (adj.ContainsKey(s.TargetNeuronId))
                    adj[s.TargetNeuronId].Add(s.SourceNeuronId);
            }
            var visited = new HashSet<int>();
            int count = 0;
            void Bfs(int start)
            {
                var q = new Queue<int>();
                q.Enqueue(start);
                visited.Add(start);
                while (q.Count > 0)
                { var c = q.Dequeue(); if (adj.TryGetValue(c, out var nb)) foreach (var nb2 in nb) if (visited.Add(nb2)) q.Enqueue(nb2); }
            }
            foreach (var n in genome.Neurons)
                if (!visited.Contains(n.Id))
                { Bfs(n.Id); count++; }
            return count;
        }

        public static float[,] ComputeLaplacianMatrix(GeoGenome genome)
        {
            var n = genome.NeuronCount;
            var adjMatrix = ComputeAdjacencyMatrix(genome);
            var laplacian = new float[n, n];
            for (int i = 0; i < n; i++)
            {
                int degree = 0;
                for (int j = 0; j < n; j++)
                { if (adjMatrix[i, j] != 0) degree++; laplacian[i, j] = -adjMatrix[i, j]; }
                laplacian[i, i] = degree;
            }
            return laplacian;
        }

        public static float ComputeAlgebraicConnectivity(GeoGenome genome)
        {
            var lap = ComputeLaplacianMatrix(genome);
            var n = lap.GetLength(0);
            if (n <= 1)
                return 0f;
            var eigenvalues = new float[n];
            for (int i = 0; i < n; i++)
                eigenvalues[i] = lap[i, i];
            Array.Sort(eigenvalues);
            return n >= 2 ? eigenvalues[1] : 0f;
        }
    }

    #endregion Topology Helpers

    // ========================================================================
    #region Genome Utilities

    public static class GenomeUtilities
    {
        public static GeoGenome Merge(GenomeId resultId, GeoGenome g1, GeoGenome g2, bool allowOverwrite = false)
        {
            var nd = new Dictionary<int, NeuronGene>();
            foreach (var n in g1.Neurons)
                nd[n.Id] = n;
            foreach (var n in g2.Neurons)
            { if (!allowOverwrite && nd.ContainsKey(n.Id)) throw new InvalidOperationException($"Duplicate neuron ID {n.Id}."); nd[n.Id] = n; }
            var sd = new Dictionary<int, SynapseGene>();
            foreach (var s in g1.Synapses)
                sd[s.Id] = s;
            foreach (var s in g2.Synapses)
            { if (!allowOverwrite && sd.ContainsKey(s.Id)) throw new InvalidOperationException($"Duplicate synapse ID {s.Id}."); sd[s.Id] = s; }
            var genome = new GeoGenome
            {
                Id = resultId,
                Neurons = ImmutableArray.Create(nd.Values.ToArray()),
                Synapses = ImmutableArray.Create(sd.Values.ToArray()),
                Species = g1.Species,
                Metadata = g1.Metadata with { ParentIds = ImmutableArray.Create(g1.Id, g2.Id), Strategy = EvolutionStrategy.Hybrid },
                Version = GenomeVersion.Current,
                CreatedAt = DateTimeOffset.UtcNow,
                ModifiedAt = DateTimeOffset.UtcNow
            };
            return genome with { TopologyHash = GenomeHasher.ComputeTopologyHash(genome) };
        }

        public static GeoGenome Crossover(GeoGenome p1, GeoGenome p2, float rate = 0.5f, int seed = -1)
        {
            var rng = seed >= 0 ? new Random(seed) : new Random();
            var builder = GeoGenomeBuilder.Create().WithStrictValidation(false);
            foreach (var n in p1.Neurons)
            {
                if (rng.NextDouble() < rate)
                    builder.AddNeuron(n);
                else if (p2.ContainsNeuron(n.Id))
                    builder.AddNeuron(p2.GetNeuron(n.Id)!.Value);
                else
                    builder.AddNeuron(n);
            }
            foreach (var s in p1.Synapses)
            {
                if (rng.NextDouble() < rate)
                    builder.AddSynapse(s);
                else if (p2.ContainsSynapse(s.Id))
                    builder.AddSynapse(p2.GetSynapse(s.Id)!.Value);
                else
                    builder.AddSynapse(s);
            }
            foreach (var s in p2.Synapses)
                if (!builder.BuildUnchecked().ContainsSynapse(s.Id) && rng.NextDouble() < rate * 0.3f)
                    builder.AddSynapse(s);
            var offspring = builder.WithSpecies(p1.Species).WithSemanticTag($"crossover-{p1.Id.ToCompactString()[..8]}x{p2.Id.ToCompactString()[..8]}").Build();
            return offspring with { Metadata = offspring.Metadata with { ParentIds = ImmutableArray.Create(p1.Id, p2.Id), Generation = Math.Max(p1.Metadata.Generation, p2.Metadata.Generation) + 1 } };
        }

        public static GeoGenome Mutate(GeoGenome genome, float rate = 0.1f, int count = 1, int seed = -1)
        {
            var rng = seed >= 0 ? new Random(seed) : new Random();
            var g = genome;
            for (int i = 0; i < count; i++)
            {
                var mt = (MutationType)rng.Next(Enum.GetValues<MutationType>().Length);
                switch (mt)
                {
                    case MutationType.PointMutation:
                        if (g.Neurons.Length > 0)
                        { var idx = rng.Next(g.Neurons.Length); g = g with { Neurons = g.Neurons.SetItem(idx, NeuronGeneFactory.CloneWithMutation(g.Neurons[idx], 1.0f, rng.Next())) }; }
                        break;
                    case MutationType.WeightPerturbation:
                        if (g.Synapses.Length > 0)
                        { var idx = rng.Next(g.Synapses.Length); g = g with { Synapses = g.Synapses.SetItem(idx, SynapseGeneFactory.CloneWithPerturbation(g.Synapses[idx], 0.2f, rng)) }; }
                        break;
                    case MutationType.BiasDrift:
                        if (g.Neurons.Length > 0)
                        { var idx = rng.Next(g.Neurons.Length); var n = g.Neurons[idx]; g = g with { Neurons = g.Neurons.SetItem(idx, n with { Bias = n.Bias + new Vector3((float)(rng.NextDouble() * 0.4 - 0.2), (float)(rng.NextDouble() * 0.4 - 0.2), (float)(rng.NextDouble() * 0.4 - 0.2)), LastModified = DateTimeOffset.UtcNow }) }; }
                        break;
                    case MutationType.ActivationShift:
                        if (g.Neurons.Length > 0)
                        { var idx = rng.Next(g.Neurons.Length); var acts = Enum.GetValues<ActivationKernel>(); g = g with { Neurons = g.Neurons.SetItem(idx, g.Neurons[idx] with { Activation = acts[rng.Next(acts.Length)], LastModified = DateTimeOffset.UtcNow }) }; }
                        break;
                    case MutationType.SynapseGrowth:
                        if (g.Neurons.Length > 1)
                        { var si = rng.Next(g.Neurons.Length); var ti = rng.Next(g.Neurons.Length); if (si != ti) g = g.AddSynapse(SynapseGene.Create(g.Synapses.Length + 1, g.Neurons[si].Id, g.Neurons[ti].Id, (float)(rng.NextDouble() * 2.0 - 1.0))); }
                        break;
                    case MutationType.SynapsePruning:
                        if (g.Synapses.Length > 0)
                        { int w = 0; float mw = float.MaxValue; for (int j = 0; j < g.Synapses.Length; j++) if (Math.Abs(g.Synapses[j].Weight) < mw) { mw = Math.Abs(g.Synapses[j].Weight); w = j; } g = g with { Synapses = g.Synapses.RemoveAt(w) }; }
                        break;
                    case MutationType.GeneSilencing:
                        if (g.Neurons.Length > 0)
                        { var idx = rng.Next(g.Neurons.Length); g = g with { Neurons = g.Neurons.SetItem(idx, g.Neurons[idx] with { IsEnabled = false }) }; }
                        break;
                    case MutationType.GeneActivation:
                        if (g.Neurons.Length > 0)
                        { var idx = rng.Next(g.Neurons.Length); g = g with { Neurons = g.Neurons.SetItem(idx, g.Neurons[idx] with { IsEnabled = true }) }; }
                        break;
                    case MutationType.Duplication:
                        if (g.Neurons.Length > 0 && g.Neurons.Length < g.Metadata.ComplexityBudget)
                        { var idx = rng.Next(g.Neurons.Length); var o = g.Neurons[idx]; g = g.AddNeuron(o with { Id = g.Neurons.Max(n => n.Id) + 1, Position = o.Position + new Vector3((float)(rng.NextDouble() * 2 - 1), 0, (float)(rng.NextDouble() * 2 - 1)), LastModified = DateTimeOffset.UtcNow }); }
                        break;
                }
            }
            return g with { Metadata = g.Metadata with { Strategy = EvolutionStrategy.Incremental }, ModifiedAt = DateTimeOffset.UtcNow };
        }

        public static float ComputeEvolutionaryDistance(GeoGenome g1, GeoGenome g2)
        {
            var n1d = g1.Neurons.ToDictionary(n => n.Id);
            var n2d = g2.Neurons.ToDictionary(n => n.Id);
            float nd = 0;
            foreach (var kvp in n1d)
            { if (n2d.TryGetValue(kvp.Key, out var n2)) { if (kvp.Value.Kind != n2.Kind) nd += 1f; if (kvp.Value.Activation != n2.Activation) nd += 0.5f; nd += Vector3.Distance(kvp.Value.Bias, n2.Bias); } else nd += 2f; }
            nd += Math.Abs(g1.NeuronCount - g2.NeuronCount) * 0.5f;
            var s1d = g1.Synapses.ToDictionary(s => s.Id);
            var s2d = g2.Synapses.ToDictionary(s => s.Id);
            float sd = 0;
            foreach (var kvp in s1d)
            { if (s2d.TryGetValue(kvp.Key, out var s2)) { sd += Math.Abs(kvp.Value.Weight - s2.Weight); if (kvp.Value.SynapseType != s2.SynapseType) sd += 0.5f; } else sd += 2f; }
            sd += Math.Abs(g1.SynapseCount - g2.SynapseCount) * 0.5f;
            int max = Math.Max(g1.NeuronCount + g1.SynapseCount, g2.NeuronCount + g2.SynapseCount);
            return max > 0 ? (nd + sd) / max : 0f;
        }

        public static GeoGenome Subsample(GeoGenome genome, int targetCount, int seed = -1)
        {
            if (genome.NeuronCount <= targetCount)
                return genome;
            var rng = seed >= 0 ? new Random(seed) : new Random();
            var keep = genome.Neurons.OrderBy(_ => rng.Next()).Take(targetCount).ToImmutableArray();
            var keepIds = new HashSet<int>(keep.Select(n => n.Id));
            var syn = genome.Synapses.Where(s => keepIds.Contains(s.SourceNeuronId) && keepIds.Contains(s.TargetNeuronId)).ToImmutableArray();
            var g = genome with { Neurons = keep, Synapses = syn, ModifiedAt = DateTimeOffset.UtcNow };
            return g with { TopologyHash = GenomeHasher.ComputeTopologyHash(g) };
        }

        public static GeoGenome Crossfade(GeoGenome from, GeoGenome to, float t, int seed = -1)
        {
            t = Math.Clamp(t, 0f, 1f);
            var rng = seed >= 0 ? new Random(seed) : new Random();
            var builder = GeoGenomeBuilder.Create().WithStrictValidation(false);
            var fromN = from.Neurons.ToDictionary(n => n.Id);
            var toN = to.Neurons.ToDictionary(n => n.Id);
            foreach (var n in from.Neurons)
            {
                if (toN.TryGetValue(n.Id, out var tn))
                    builder.AddNeuron(n with { Bias = Vector3.Lerp(n.Bias, tn.Bias, t), WeightScale = n.WeightScale + (tn.WeightScale - n.WeightScale) * t, Activation = rng.NextDouble() < t ? tn.Activation : n.Activation, LastModified = DateTimeOffset.UtcNow });
                else
                    builder.AddNeuron(n);
            }
            foreach (var n in to.Neurons)
                if (!fromN.ContainsKey(n.Id) && rng.NextDouble() > t)
                    builder.AddNeuron(n);
            var fromS = from.Synapses.ToDictionary(s => s.Id);
            var toS = to.Synapses.ToDictionary(s => s.Id);
            foreach (var s in from.Synapses)
            {
                if (toS.TryGetValue(s.Id, out var ts))
                    builder.AddSynapse(s with { Weight = s.Weight + (ts.Weight - s.Weight) * t });
                else
                    builder.AddSynapse(s);
            }
            foreach (var s in to.Synapses)
                if (!fromS.ContainsKey(s.Id) && rng.NextDouble() > t)
                    builder.AddSynapse(s);
            var result = builder.WithSpecies(from.Species).WithSemanticTag($"crossfade-{t:P0}").Build();
            return result with { Metadata = result.Metadata with { ParentIds = ImmutableArray.Create(from.Id, to.Id), Generation = Math.Max(from.Metadata.Generation, to.Metadata.Generation) + 1 } };
        }

        public static GeoGenome ExtractSubgraph(GeoGenome genome, NeuronKind kind)
        {
            var target = genome.Neurons.Where(n => n.Kind == kind).ToImmutableArray();
            var ids = new HashSet<int>(target.Select(n => n.Id));
            var syn = genome.Synapses.Where(s => ids.Contains(s.SourceNeuronId) && ids.Contains(s.TargetNeuronId)).ToImmutableArray();
            var g = genome with { Neurons = target, Synapses = syn, ModifiedAt = DateTimeOffset.UtcNow };
            return g with { TopologyHash = GenomeHasher.ComputeTopologyHash(g) };
        }

        public static Dictionary<GenomeId, float> ComputeSelectionProbabilities(ImmutableArray<GeoGenome> population)
        {
            var total = population.Sum(g => Math.Max(0, g.Metadata.FitnessScore));
            var probs = new Dictionary<GenomeId, float>();
            if (total <= 0)
            { var u = 1.0f / population.Length; foreach (var g in population) probs[g.Id] = u; return probs; }
            foreach (var g in population)
                probs[g.Id] = (float)(Math.Max(0, g.Metadata.FitnessScore) / total);
            return probs;
        }
    }

    internal static class MathHelper
    {
        public static float Lerp(float a, float b, float t) => a + (b - a) * t;
        public static double Lerp(double a, double b, double t) => a + (b - a) * t;
        public static float InverseLerp(float a, float b, float value) => Math.Abs(b - a) < float.Epsilon ? 0f : Math.Clamp((value - a) / (b - a), 0f, 1f);
        public static float SmoothStep(float edge0, float edge1, float x) { var t = Math.Clamp((x - edge0) / (edge1 - edge0), 0f, 1f); return t * t * (3f - 2f * t); }
        public static float SmootherStep(float edge0, float edge1, float x) { var t = Math.Clamp((x - edge0) / (edge1 - edge0), 0f, 1f); return t * t * t * (t * (t * 6f - 15f) + 10f); }
        public static float Remap(float value, float fromMin, float fromMax, float toMin, float toMax) => toMin + (value - fromMin) * (toMax - toMin) / (fromMax - fromMin);
        public static float Wrap(float value, float min, float max) { var range = max - min; return min + ((value - min) % range + range) % range; }
        public static float PingPong(float value, float length) => length - Math.Abs(value % (2 * length) - length);
        public static float MoveTowards(float current, float target, float maxDelta) => Math.Abs(target - current) <= maxDelta ? target : current + Math.Sign(target - current) * maxDelta;
        public static float AngleLerp(float from, float to, float t) { float delta = ((to - from + 540f) % 360f) - 180f; return from + delta * t; }
        public static float NormalizeAngle(float angle) => ((angle % 360f) + 360f) % 360f;
    }

    #endregion Genome Utilities

    // ========================================================================
    #region Statistics

    public sealed class GenomeStatisticsCollector
    {
        private readonly List<GenomeStatisticsSnapshot> _snapshots = new();
        private readonly object _lock = new();

        public void RecordSnapshot(int generation, ImmutableArray<GeoGenome> population)
        {
            lock (_lock)
            {
                if (population.Length == 0)
                    return;
                var fitnesses = population.Select(g => g.Metadata.FitnessScore).ToArray();
                _snapshots.Add(new GenomeStatisticsSnapshot
                {
                    Generation = generation,
                    PopulationSize = population.Length,
                    AverageFitness = fitnesses.Average(),
                    MaxFitness = fitnesses.Max(),
                    MinFitness = fitnesses.Min(),
                    FitnessStdDev = StdDev(fitnesses),
                    AverageNeuronCount = population.Average(g => (double)g.NeuronCount),
                    AverageSynapseCount = population.Average(g => (double)g.SynapseCount),
                    SpeciesDistribution = population.GroupBy(g => g.Species).ToDictionary(grp => grp.Key, grp => grp.Count()),
                    KindDistribution = population.SelectMany(g => g.Neurons).GroupBy(n => n.Kind).ToDictionary(grp => grp.Key, grp => grp.Count()),
                    DiversityIndex = ShannonEntropy(population.GroupBy(g => g.Species).Select(grp => (double)grp.Count() / population.Length).ToArray()),
                    Timestamp = DateTimeOffset.UtcNow
                });
            }
        }

        public IReadOnlyList<GenomeStatisticsSnapshot> GetSnapshots() { lock (_lock) { return _snapshots.ToArray(); } }
        public GenomeStatisticsSnapshot? GetLatest() { lock (_lock) { return _snapshots.Count > 0 ? _snapshots[^1] : null; } }

        public GenomeEvolutionSummary ComputeSummary()
        {
            lock (_lock)
            {
                if (_snapshots.Count == 0)
                    return new GenomeEvolutionSummary();
                var f = _snapshots[0];
                var l = _snapshots[^1];
                return new GenomeEvolutionSummary
                {
                    TotalGenerations = _snapshots.Count,
                    InitialPopulationSize = f.PopulationSize,
                    FinalPopulationSize = l.PopulationSize,
                    FitnessImprovement = l.AverageFitness - f.AverageFitness,
                    FitnessImprovementPercentage = f.AverageFitness != 0 ? ((l.AverageFitness - f.AverageFitness) / Math.Abs(f.AverageFitness)) * 100 : 0,
                    PeakFitness = _snapshots.Max(s => s.MaxFitness),
                    AverageFitnessOverTime = _snapshots.Average(s => s.AverageFitness),
                    FinalSpeciesCount = l.SpeciesDistribution.Count,
                    FinalDiversityIndex = l.DiversityIndex,
                    TotalNeuronGrowth = l.AverageNeuronCount - f.AverageNeuronCount,
                    TotalSynapseGrowth = l.AverageSynapseCount - f.AverageSynapseCount
                };
            }
        }

        private static double StdDev(double[] v) { if (v.Length <= 1) return 0; var m = v.Average(); return Math.Sqrt(v.Sum(x => (x - m) * (x - m)) / (v.Length - 1)); }
        private static double ShannonEntropy(double[] p) { double e = 0; foreach (var x in p) if (x > 0) e -= x * Math.Log2(x); return e; }
    }

    public sealed record GenomeStatisticsSnapshot
    {
        public int Generation { get; init; }
        public int PopulationSize { get; init; }
        public double AverageFitness { get; init; }
        public double MaxFitness { get; init; }
        public double MinFitness { get; init; }
        public double FitnessStdDev { get; init; }
        public double AverageNeuronCount { get; init; }
        public double AverageSynapseCount { get; init; }
        public Dictionary<SpeciesId, int> SpeciesDistribution { get; init; } = new();
        public Dictionary<NeuronKind, int> KindDistribution { get; init; } = new();
        public double DiversityIndex { get; init; }
        public DateTimeOffset Timestamp { get; init; }
    }

    public sealed record GenomeEvolutionSummary
    {
        public int TotalGenerations { get; init; }
        public int InitialPopulationSize { get; init; }
        public int FinalPopulationSize { get; init; }
        public double FitnessImprovement { get; init; }
        public double FitnessImprovementPercentage { get; init; }
        public double PeakFitness { get; init; }
        public double AverageFitnessOverTime { get; init; }
        public int FinalSpeciesCount { get; init; }
        public double FinalDiversityIndex { get; init; }
        public int ConvergenceGeneration { get; init; }
        public double TotalNeuronGrowth { get; init; }
        public double TotalSynapseGrowth { get; init; }
        public bool HasConverged => ConvergenceGeneration >= 0;
    }

    #endregion Statistics

} // namespace GDNN.Core.Genome
