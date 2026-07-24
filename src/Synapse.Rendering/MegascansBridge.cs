// =============================================================================
// MegascansBridge.cs - GDNN Engine: Production Artist Pipeline
// Quixel Megascans / 3D Scanned Asset Integration
// Full import pipeline for surfaces, 3D assets, vegetation, decals, atlases,
// with automatic material conversion, LOD generation, and virtual texturing.
// =============================================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Materials.SubstrateOmega;
using GDNN.Rendering.MeshIO;
using Synapse.Infrastructure.Logging;

namespace GDNN.Rendering.ArtPipeline
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    public enum MegascansAssetType : byte
    {
        Surface,
        ThreeDAsset,
        Vegetation,
        Decal,
        Atlas,
        Alpha,
        Displacement,
        Imperfection,
        Scene
    }

    public enum MegascansSurfaceType : byte
    {
        Concrete, Asphalt, Brick, Metal, Wood, Stone, Tile, Marble,
        Dirt, Grass, Gravel, Sand, Snow, Ice, Water, Rock,
        Fabric, Leather, Plastic, Rubber, Glass, Ceramic, Paint,
        Soil, Moss, Lichen, Bark, Pebble, Basalt, Slate,
        Limestone, Granite, Sandstone, Clay, Chalk
    }

    public enum MegascansVegetationType : byte
    {
        Tree, Shrub, Grass, Flower, Fern, Bush, Vine,
        Branch, Root, Leaf, Pine, Palm, Cactus, Mushroom
    }

    public enum MegascansTextureQuality : byte
    {
        Preview_1K,
        Quality_2K,
        High_4K,
        Ultra_8K
    }

    public enum MegascansImportMode : byte
    {
        CopyToLibrary,
        MoveToLibrary,
        ReferenceOriginal,
        StreamOnDemand
    }

    public enum MegascansLODStrategy : byte
    {
        Automatic,
        Manual,
        Disabled,
        NaniteVirtual
    }

    // =========================================================================
    // MEGASCANS ASSET METADATA
    // =========================================================================

    public class MegascansAsset
    {
        public string Id { get; set; } = Guid.NewGuid().ToString("N");
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public MegascansAssetType AssetType { get; set; }
        public MegascansSurfaceType SurfaceType { get; set; }
        public MegascansVegetationType VegetationType { get; set; }
        public MegascansTextureQuality Quality { get; set; } = MegascansTextureQuality.High_4K;
        public string SourcePath { get; set; } = "";
        public string LibraryPath { get; set; } = "";
        public float SurfaceArea { get; set; } = 1.0f;
        public float TileSizeU { get; set; } = 1.0f;
        public float TileSizeV { get; set; } = 1.0f;
        public bool IsTiling { get; set; } = true;
        public bool IsTrimSheet { get; set; }
        public DateTime ImportedAt { get; set; } = DateTime.UtcNow;
        public DateTime ModifiedAt { get; set; } = DateTime.UtcNow;
        public string Author { get; set; } = "Quixel";
        public string Tags { get; set; } = "";
        public string[] TagArray => string.IsNullOrEmpty(Tags) ? Array.Empty<string>() : Tags.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        public float RelevanceScore { get; set; }
        public Dictionary<string, string> Metadata { get; set; } = new();
    }

    // =========================================================================
    // MEGASCANS TEXTURE SET
    // =========================================================================

    public class MegascansTextureSet
    {
        public string BaseColor { get; set; } = "";
        public string Normal { get; set; } = "";
        public string Roughness { get; set; } = "";
        public string Metallic { get; set; } = "";
        public string AmbientOcclusion { get; set; } = "";
        public string Displacement { get; set; } = "";
        public string Emissive { get; set; } = "";
        public string Opacity { get; set; } = "";
        public string NormalDX { get; set; } = "";
        public string NormalGL { get; set; } = "";
        public string Cavity { get; set; } = "";
        public string Curvature { get; set; } = "";
        public string Fuzz { get; set; } = "";
        public string Thickness { get; set; } = "";
        public string Translucency { get; set; } = "";
        public string Specular { get; set; } = "";
        public string Anisotropy { get; set; } = "";
        public string SheenColor { get; set; } = "";
        public string ClearCoatNormal { get; set; } = "";
        public string ClearCoatRoughness { get; set; } = "";
        public string DetailAlbedo { get; set; } = "";
        public string DetailNormal { get; set; } = "";
        public string packed_ORM { get; set; } = "";
        public string packed_MR { get; set; } = "";
        public int Width { get; set; }
        public int Height { get; set; }
        public string Format { get; set; } = "png";
        public MegascansTextureQuality Quality { get; set; }
    }

    // =========================================================================
    // MEGASCANS 3D ASSET DATA
    // =========================================================================

    public class Megascans3DAssetData
    {
        public string MeshPath { get; set; } = "";
        public string HighPolyPath { get; set; } = "";
        public int TriangleCount { get; set; }
        public int VertexCount { get; set; }
        public float BoundingRadius { get; set; }
        public string[] LODPaths { get; set; } = Array.Empty<string>();
        public float[] LODScreenSizes { get; set; } = Array.Empty<float>();
        public string CollisionMeshPath { get; set; } = "";
        public string impostorAtlasPath { get; set; } = "";
        public int impostorFrames { get; set; }
    }

    // =========================================================================
    // MEGASCANS VEGETATION DATA
    // =========================================================================

    public class MegascansVegetationData
    {
        public string MeshPath { get; set; } = "";
        public MegascansVegetationType Type { get; set; }
        public float Height { get; set; }
        public float Width { get; set; }
        public float Density { get; set; } = 1.0f;
        public float WindStrength { get; set; } = 0.5f;
        public float WindFrequency { get; set; } = 1.0f;
        public float VariationCount { get; set; } = 5;
        public bool HasWind { get; set; } = true;
        public bool IsBillboard { get; set; }
        public string[] VariantMeshPaths { get; set; } = Array.Empty<string>();
    }

    // =========================================================================
    // MEGASCANS DECAT DATA
    // =========================================================================

    public class MegascansDecalData
    {
        public string MeshPath { get; set; } = "";
        public float Width { get; set; } = 1.0f;
        public float Height { get; set; } = 1.0f;
        public float ProjectionDepth { get; set; } = 0.5f;
        public bool IsProjecting { get; set; } = true;
    }

    // =========================================================================
    // MEGASCANS ATLAS DATA
    // =========================================================================

    public class MegascansAtlasData
    {
        public int TileCountX { get; set; } = 4;
        public int TileCountY { get; set; } = 4;
        public float TileSizeU { get; set; } = 1.0f;
        public float TileSizeV { get; set; } = 1.0f;
        public string[] TileNames { get; set; } = Array.Empty<string>();
        public float[] TileOffsetsU { get; set; } = Array.Empty<float>();
        public float[] TileOffsetsV { get; set; } = Array.Empty<float>();
    }

    // =========================================================================
    // MEGASCANS SCAN ENTRY (FULL IMPORT RESULT)
    // =========================================================================

    public class MegascansScanEntry
    {
        public MegascansAsset Asset { get; set; }
        public MegascansTextureSet Textures { get; set; }
        public Megascans3DAssetData ThreeDData { get; set; }
        public MegascansVegetationData VegetationData { get; set; }
        public MegascansDecalData DecalData { get; set; }
        public MegascansAtlasData AtlasData { get; set; }
        public SubstrateMaterial ConvertedMaterial { get; set; }
        public MeshAsset ConvertedMesh { get; set; }
        public List<string> Warnings { get; set; } = new();
        public bool ImportSucceeded { get; set; }
        public TimeSpan ImportDuration { get; set; }
    }

    // =========================================================================
    // MEGASCANS LIBRARY CONFIG
    // =========================================================================

    public class MegascansConfig
    {
        public string LibraryRootPath { get; set; } = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".synapse", "megascans");
        public MegascansTextureQuality DefaultQuality { get; set; } = MegascansTextureQuality.High_4K;
        public MegascansImportMode ImportMode { get; set; } = MegascansImportMode.CopyToLibrary;
        public MegascansLODStrategy LODStrategy { get; set; } = MegascansLODStrategy.Automatic;
        public bool AutoConvertMaterial { get; set; } = true;
        public bool GenerateMipmaps { get; set; } = true;
        public bool EnableVirtualTexturing { get; set; } = true;
        public bool EnableStreaming { get; set; } = true;
        public int MaxConcurrentImports { get; set; } = 4;
        public bool UseNormalDX { get; set; } = true;
        public string DefaultNormalFormat { get; set; } = "DirectX";
        public float DefaultNormalIntensity { get; set; } = 1.0f;
        public bool UsePackedTextures { get; set; } = true;
    }

    // =========================================================================
    // MEGASCANS BRIDGE - MAIN IMPORTER
    // =========================================================================

    public class MegascansBridge : IDisposable
    {
        private readonly MegascansConfig _config;
        private readonly ConcurrentDictionary<string, MegascansScanEntry> _importedAssets;
        private readonly Dictionary<MegascansSurfaceType, MaterialPresetCategory> _surfaceCategoryMap;
        private readonly SemaphoreSlim _importSemaphore;
        private bool _disposed;

        public MegascansConfig Config => _config;
        public int ImportedAssetCount => _importedAssets.Count;

        public MegascansBridge(MegascansConfig config = null)
        {
            _config = config ?? new MegascansConfig();
            _importedAssets = new ConcurrentDictionary<string, MegascansScanEntry>();
            _importSemaphore = new SemaphoreSlim(_config.MaxConcurrentImports, _config.MaxConcurrentImports);
            _surfaceCategoryMap = BuildSurfaceCategoryMap();
        }

        public async Task<MegascansScanEntry> ImportAssetAsync(string assetDirectory, CancellationToken ct = default)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var entry = new MegascansScanEntry();
            await _importSemaphore.WaitAsync(ct);
            try
            {
                var metadataPath = Path.Combine(assetDirectory, "metadata.json");
                if (!File.Exists(metadataPath))
                {
                    metadataPath = FindMetadataFile(assetDirectory);
                    if (metadataPath == null)
                    {
                        entry.Warnings.Add("No metadata.json found in asset directory");
                        entry.ImportSucceeded = false;
                        return entry;
                    }
                }

                var metaInfo = new FileInfo(metadataPath);
                if (metaInfo.Length > 2_000_000)
                    throw new InvalidDataException("Megascans metadata exceeds size limit.");
                var metadataJson = await File.ReadAllTextAsync(metadataPath, ct);
                var asset = ParseMetadata(metadataJson, assetDirectory);
                entry.Asset = asset;
                entry.Textures = DiscoverTextures(assetDirectory);

                if (asset.AssetType == MegascansAssetType.ThreeDAsset || asset.AssetType == MegascansAssetType.Vegetation)
                    entry.ThreeDData = Discover3DData(assetDirectory);

                if (asset.AssetType == MegascansAssetType.Vegetation)
                    entry.VegetationData = DiscoverVegetationData(assetDirectory);

                if (asset.AssetType == MegascansAssetType.Decal)
                    entry.DecalData = DiscoverDecalData(assetDirectory);

                if (asset.AssetType == MegascansAssetType.Atlas)
                    entry.AtlasData = DiscoverAtlasData(assetDirectory);

                if (_config.AutoConvertMaterial)
                    entry.ConvertedMaterial = ConvertToSubstrateMaterial(entry);

                CopyOrReferenceFiles(entry, assetDirectory);

                sw.Stop();
                entry.ImportDuration = sw.Elapsed;
                entry.ImportSucceeded = true;
                _importedAssets[asset.Id] = entry;
                return entry;
            }
            catch (Exception ex)
            {
                entry.Warnings.Add($"Import failed: {ex.Message}");
                entry.ImportSucceeded = false;
                entry.ImportDuration = sw.Elapsed;
                return entry;
            }
            finally
            {
                _importSemaphore.Release();
            }
        }

        public async Task<List<MegascansScanEntry>> ImportDirectoryAsync(string rootDirectory, CancellationToken ct = default)
        {
            var results = new List<MegascansScanEntry>();
            if (!Directory.Exists(rootDirectory))
                return results;

            var subdirs = Directory.GetDirectories(rootDirectory, "*", SearchOption.TopDirectoryOnly);
            var tasks = subdirs.Select(dir => ImportAssetAsync(dir, ct));
            var entries = await Task.WhenAll(tasks);
            results.AddRange(entries);
            return results;
        }

        public SubstrateMaterial ConvertToSubstrateMaterial(MegascansScanEntry scanEntry)
        {
            var asset = scanEntry.Asset;
            var textures = scanEntry.Textures;
            var mat = new SubstrateMaterial(asset.Name)
            {
                Domain = MaterialDomain.Surface,
                FeatureFlags = MaterialFeatureFlags.None
            };
            mat.InitializeDefaults();

            if (!string.IsNullOrEmpty(textures?.BaseColor))
                mat.SetTexture(TextureChannel.Albedo, new TextureReference(textures.BaseColor, TextureChannel.Albedo) { Sampler = TextureSamplerMode.Trilinear });

            if (!string.IsNullOrEmpty(textures?.Normal) || !string.IsNullOrEmpty(textures?.NormalDX))
            {
                string normalPath = _config.UseNormalDX ? textures.NormalDX : textures.NormalGL;
                if (string.IsNullOrEmpty(normalPath))
                    normalPath = textures.Normal ?? textures.NormalDX;
                if (!string.IsNullOrEmpty(normalPath))
                    mat.SetTexture(TextureChannel.Normal, new TextureReference(normalPath, TextureChannel.Normal));
            }

            if (!string.IsNullOrEmpty(textures?.packed_ORM))
            {
                mat.SetTexture(TextureChannel.Roughness, new TextureReference(textures.packed_ORM, TextureChannel.Roughness));
                mat.SetTexture(TextureChannel.Metallic, new TextureReference(textures.packed_ORM, TextureChannel.Metallic));
                mat.SetTexture(TextureChannel.AO, new TextureReference(textures.packed_ORM, TextureChannel.AO));
            }
            else
            {
                if (!string.IsNullOrEmpty(textures?.Roughness))
                    mat.SetTexture(TextureChannel.Roughness, new TextureReference(textures.Roughness, TextureChannel.Roughness));
                if (!string.IsNullOrEmpty(textures?.Metallic))
                    mat.SetTexture(TextureChannel.Metallic, new TextureReference(textures.Metallic, TextureChannel.Metallic));
                if (!string.IsNullOrEmpty(textures?.AmbientOcclusion))
                    mat.SetTexture(TextureChannel.AO, new TextureReference(textures.AmbientOcclusion, TextureChannel.AO));
            }

            if (!string.IsNullOrEmpty(textures?.Displacement))
            {
                mat.SetTexture(TextureChannel.Displacement, new TextureReference(textures.Displacement, TextureChannel.Displacement));
                mat.SetProperty("DisplacementScale", new MaterialProperty("DisplacementScale", MaterialPropertyType.Float, 0.1f, 0, 5, 0.1f));
            }

            if (!string.IsNullOrEmpty(textures?.Emissive))
            {
                mat.SetTexture(TextureChannel.Emissive, new TextureReference(textures.Emissive, TextureChannel.Emissive));
                mat.SetProperty("EmissiveIntensity", new MaterialProperty("EmissiveIntensity", MaterialPropertyType.Float, 1.0f, 0, 100, 1.0f));
            }

            if (!string.IsNullOrEmpty(textures?.Opacity))
            {
                mat.SetTexture(TextureChannel.Opacity, new TextureReference(textures.Opacity, TextureChannel.Opacity));
                mat.SetProperty("Opacity", new MaterialProperty("Opacity", MaterialPropertyType.Float, 1.0f, 0, 1, 1f));
            }

            if (!string.IsNullOrEmpty(textures?.Displacement))
                mat.SetTexture(TextureChannel.Height, new TextureReference(textures.Displacement, TextureChannel.Height));

            if (!string.IsNullOrEmpty(textures?.Thickness))
            {
                mat.SetTexture(TextureChannel.Thickness, new TextureReference(textures.Thickness, TextureChannel.Thickness));
                mat.SetProperty("Thickness", new MaterialProperty("Thickness", MaterialPropertyType.Float, 0.5f, 0, 10, 0.5f));
            }

            if (!string.IsNullOrEmpty(textures?.Specular))
                mat.SetTexture(TextureChannel.SpecularColor, new TextureReference(textures.Specular, TextureChannel.SpecularColor));

            if (!string.IsNullOrEmpty(textures?.Anisotropy))
                mat.SetTexture(TextureChannel.AnisotropyStrength, new TextureReference(textures.Anisotropy, TextureChannel.AnisotropyStrength));

            if (!string.IsNullOrEmpty(textures?.SheenColor))
                mat.SetTexture(TextureChannel.SheenColor, new TextureReference(textures.SheenColor, TextureChannel.SheenColor));

            if (!string.IsNullOrEmpty(textures?.ClearCoatNormal))
                mat.SetTexture(TextureChannel.ClearCoatNormal, new TextureReference(textures.ClearCoatNormal, TextureChannel.ClearCoatNormal));

            if (!string.IsNullOrEmpty(textures?.ClearCoatRoughness))
                mat.SetTexture(TextureChannel.ClearCoatRoughness, new TextureReference(textures.ClearCoatRoughness, TextureChannel.ClearCoatRoughness));

            if (!string.IsNullOrEmpty(textures?.Fuzz))
            {
                mat.SetTexture(TextureChannel.SubsurfaceColor, new TextureReference(textures.Fuzz, TextureChannel.SubsurfaceColor));
                mat.SetProperty("SubsurfaceRadius", new MaterialProperty("SubsurfaceRadius", MaterialPropertyType.Vec3, new Vec3(1.0f, 0.2f, 0.1f), 0, 5, 1f));
                mat.SetProperty("SubsurfaceColor", new MaterialProperty("SubsurfaceColor", MaterialPropertyType.Color, new Color3(0.8f, 0.2f, 0.1f)));
            }

            if (!string.IsNullOrEmpty(textures?.Translucency))
            {
                mat.SetProperty("Transmission", new MaterialProperty("Transmission", MaterialPropertyType.Float, 0.5f, 0, 1, 0f));
            }

            mat.SetProperty("NormalStrength", new MaterialProperty("NormalStrength", MaterialPropertyType.Float, _config.DefaultNormalIntensity, 0, 2, 1f));

            ApplySurfaceSpecificSettings(mat, asset);
            mat.ComputeActiveFeatures();

            scanEntry.ConvertedMaterial = mat;
            return mat;
        }

        private void ApplySurfaceSpecificSettings(SubstrateMaterial mat, MegascansAsset asset)
        {
            switch (asset.AssetType)
            {
                case MegascansAssetType.Surface:
                    ApplySurfaceSettings(mat, asset.SurfaceType);
                    break;
                case MegascansAssetType.Vegetation:
                    ApplyVegetationSettings(mat, asset.VegetationType);
                    break;
                case MegascansAssetType.Decal:
                    mat.SetProperty("Opacity", new MaterialProperty("Opacity", MaterialPropertyType.Float, 1.0f, 0, 1, 1f));
                    break;
                case MegascansAssetType.Imperfection:
                    mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.5f, 0, 1, 0.5f));
                    break;
            }
        }

        private void ApplySurfaceSettings(SubstrateMaterial mat, MegascansSurfaceType surfaceType)
        {
            switch (surfaceType)
            {
                case MegascansSurfaceType.Metal:
                case MegascansSurfaceType.Basalt:
                    mat.SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, 1.0f, 0, 1, 1f));
                    break;
                case MegascansSurfaceType.Wood:
                case MegascansSurfaceType.Bark:
                    mat.SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, 0.0f, 0, 1, 0f));
                    break;
                case MegascansSurfaceType.Marble:
                case MegascansSurfaceType.Granite:
                case MegascansSurfaceType.Limestone:
                case MegascansSurfaceType.Sandstone:
                case MegascansSurfaceType.Slate:
                    mat.SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, 0.0f, 0, 1, 0f));
                    mat.SetProperty("Specular", new MaterialProperty("Specular", MaterialPropertyType.Float, 0.6f, 0, 1, 0.5f));
                    break;
                case MegascansSurfaceType.Snow:
                case MegascansSurfaceType.Ice:
                    mat.SetProperty("SubsurfaceRadius", new MaterialProperty("SubsurfaceRadius", MaterialPropertyType.Vec3, new Vec3(0.5f, 0.5f, 0.6f), 0, 5, 1f));
                    mat.SetProperty("SubsurfaceColor", new MaterialProperty("SubsurfaceColor", MaterialPropertyType.Color, new Color3(0.8f, 0.85f, 0.95f)));
                    mat.SetProperty("Transmission", new MaterialProperty("Transmission", MaterialPropertyType.Float, 0.3f, 0, 1, 0f));
                    break;
                case MegascansSurfaceType.Water:
                    mat.SetProperty("Transmission", new MaterialProperty("Transmission", MaterialPropertyType.Float, 1.0f, 0, 1, 1f));
                    mat.SetProperty("IOR", new MaterialProperty("IOR", MaterialPropertyType.Float, 1.333f, 1, 2.5f, 1.333f));
                    mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.0f, 0, 1, 0f));
                    break;
                case MegascansSurfaceType.Glass:
                    mat.SetProperty("Transmission", new MaterialProperty("Transmission", MaterialPropertyType.Float, 1.0f, 0, 1, 1f));
                    mat.SetProperty("IOR", new MaterialProperty("IOR", MaterialPropertyType.Float, 1.52f, 1, 2.5f, 1.52f));
                    mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.0f, 0, 1, 0f));
                    break;
                case MegascansSurfaceType.Fabric:
                case MegascansSurfaceType.Leather:
                    mat.SetProperty("Sheen", new MaterialProperty("Sheen", MaterialPropertyType.Float, 0.3f, 0, 1, 0f));
                    mat.SetProperty("SheenRoughness", new MaterialProperty("SheenRoughness", MaterialPropertyType.Float, 0.6f, 0, 1, 0.5f));
                    break;
                case MegascansSurfaceType.Ceramic:
                case MegascansSurfaceType.Tile:
                    mat.SetProperty("ClearCoat", new MaterialProperty("ClearCoat", MaterialPropertyType.Float, 0.8f, 0, 1, 0f));
                    mat.SetProperty("ClearCoatRoughness", new MaterialProperty("ClearCoatRoughness", MaterialPropertyType.Float, 0.05f, 0, 1, 0.01f));
                    break;
                case MegascansSurfaceType.Grass:
                case MegascansSurfaceType.Moss:
                case MegascansSurfaceType.Lichen:
                    mat.SetProperty("SubsurfaceRadius", new MaterialProperty("SubsurfaceRadius", MaterialPropertyType.Vec3, new Vec3(0.3f, 0.4f, 0.2f), 0, 5, 1f));
                    mat.SetProperty("SubsurfaceColor", new MaterialProperty("SubsurfaceColor", MaterialPropertyType.Color, new Color3(0.2f, 0.5f, 0.15f)));
                    break;
            }
        }

        private void ApplyVegetationSettings(SubstrateMaterial mat, MegascansVegetationType vegType)
        {
            mat.SetProperty("SubsurfaceRadius", new MaterialProperty("SubsurfaceRadius", MaterialPropertyType.Vec3, new Vec3(0.5f, 0.3f, 0.1f), 0, 5, 1f));
            mat.SetProperty("SubsurfaceColor", new MaterialProperty("SubsurfaceColor", MaterialPropertyType.Color, new Color3(0.3f, 0.6f, 0.15f)));
            mat.SetProperty("Transmission", new MaterialProperty("Transmission", MaterialPropertyType.Float, 0.4f, 0, 1, 0f));
            mat.SetProperty("TwoSided", new MaterialProperty("TwoSided", MaterialPropertyType.Bool, true));
            mat.FeatureFlags |= MaterialFeatureFlags.SubsurfaceScattering | MaterialFeatureFlags.TwoSided;

            switch (vegType)
            {
                case MegascansVegetationType.Leaf:
                case MegascansVegetationType.Fern:
                case MegascansVegetationType.Vine:
                    mat.SetProperty("Transmission", new MaterialProperty("Transmission", MaterialPropertyType.Float, 0.6f, 0, 1, 0f));
                    break;
                case MegascansVegetationType.Branch:
                case MegascansVegetationType.Root:
                    mat.SetProperty("Transmission", new MaterialProperty("Transmission", MaterialPropertyType.Float, 0.0f, 0, 1, 0f));
                    mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.9f, 0, 1, 0.5f));
                    break;
                case MegascansVegetationType.Pine:
                case MegascansVegetationType.Palm:
                    mat.SetProperty("Transmission", new MaterialProperty("Transmission", MaterialPropertyType.Float, 0.3f, 0, 1, 0f));
                    break;
            }
        }

        private MegascansAsset ParseMetadata(string json, string directory)
        {
            var asset = new MegascansAsset { SourcePath = directory };
            try
            {
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;
                if (root.TryGetProperty("name", out var name))
                    asset.Name = name.GetString() ?? "";
                if (root.TryGetProperty("description", out var desc))
                    asset.Description = desc.GetString() ?? "";
                if (root.TryGetProperty("id", out var id))
                {
                    var rawId = id.GetString() ?? Guid.NewGuid().ToString("N");
                    try
                    { asset.Id = Synapse.Core.Security.PathSecurity.RequireSafeAssetId(rawId); }
                    catch (Exception ex)
                    {
                        SynapseLogger.Default.Warn("MegascansBridge", $"Unsafe asset id '{rawId}'; generating fallback id.", ex);
                        asset.Id = Guid.NewGuid().ToString("N");
                    }
                }
                if (root.TryGetProperty("type", out var type))
                {
                    string typeStr = type.GetString()?.ToLower() ?? "";
                    asset.AssetType = typeStr switch
                    {
                        "surface" or "surfaceatlas" => MegascansAssetType.Surface,
                        "3d" or "3dasset" or "model" => MegascansAssetType.ThreeDAsset,
                        "vegetation" or "plant" or "foliage" => MegascansAssetType.Vegetation,
                        "decal" => MegascansAssetType.Decal,
                        "atlas" => MegascansAssetType.Atlas,
                        "alpha" or "brush" => MegascansAssetType.Alpha,
                        "imperfection" => MegascansAssetType.Imperfection,
                        _ => MegascansAssetType.Surface
                    };
                }
                if (root.TryGetProperty("tags", out var tagsElement))
                {
                    if (tagsElement.ValueKind == JsonValueKind.Array)
                    {
                        var tagList = new List<string>();
                        foreach (var tag in tagsElement.EnumerateArray())
                            tagList.Add(tag.GetString() ?? "");
                        asset.Tags = string.Join(",", tagList);
                    }
                    else if (tagsElement.ValueKind == JsonValueKind.String)
                        asset.Tags = tagsElement.GetString() ?? "";
                }
                if (root.TryGetProperty("quality", out var quality))
                {
                    string qStr = quality.GetString()?.ToLower() ?? "";
                    asset.Quality = qStr switch
                    {
                        "preview" or "1k" => MegascansTextureQuality.Preview_1K,
                        "quality" or "2k" => MegascansTextureQuality.Quality_2K,
                        "high" or "4k" => MegascansTextureQuality.High_4K,
                        "ultra" or "8k" => MegascansTextureQuality.Ultra_8K,
                        _ => MegascansTextureQuality.High_4K
                    };
                }
                if (root.TryGetProperty("size", out var sizeElement))
                {
                    if (sizeElement.TryGetProperty("x", out var sx))
                        asset.TileSizeU = sx.GetSingle();
                    if (sizeElement.TryGetProperty("y", out var sy))
                        asset.TileSizeV = sy.GetSingle();
                }
                if (root.TryGetProperty("tiling", out var tiling))
                    asset.IsTiling = tiling.GetBoolean();
                if (root.TryGetProperty("surfaceArea", out var area))
                    asset.SurfaceArea = area.GetSingle();
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("MegascansBridge", $"Failed to parse Megascans metadata in '{directory}'.", ex);
            }

            if (string.IsNullOrEmpty(asset.Name))
                asset.Name = new DirectoryInfo(directory).Name;
            if (string.IsNullOrWhiteSpace(asset.Id))
                asset.Id = Guid.NewGuid().ToString("N");
            try
            { asset.Id = Synapse.Core.Security.PathSecurity.RequireSafeAssetId(asset.Id); }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("MegascansBridge", $"Asset id '{asset.Id}' failed validation; generating fallback id.", ex);
                asset.Id = Guid.NewGuid().ToString("N");
            }
            Directory.CreateDirectory(_config.LibraryRootPath);
            asset.LibraryPath = Synapse.Core.Security.PathSecurity.CombineUnderRoot(_config.LibraryRootPath, asset.Id);
            return asset;
        }

        private MegascansTextureSet DiscoverTextures(string directory)
        {
            var textures = new MegascansTextureSet();
            var files = Directory.GetFiles(directory, "*.*", SearchOption.AllDirectories);
            if (files.Length > 10_000)
                throw new InvalidDataException(
                    $"Megascans asset tree under '{directory}' exceeds 10,000 files.");

            long totalBytes = 0;
            const long maxTotalBytes = 2L * 1024 * 1024 * 1024; // 2 GiB
            foreach (var file in files)
            {
                try
                {
                    totalBytes += new FileInfo(file).Length;
                    if (totalBytes > maxTotalBytes)
                        throw new InvalidDataException(
                            $"Megascans asset tree under '{directory}' exceeds 2 GiB.");
                }
                catch (InvalidDataException)
                {
                    throw;
                }
                catch
                {
                    // Skip unreadable file size probes.
                }

                string ext = Path.GetExtension(file).ToLowerInvariant();
                if (ext is not (".png" or ".jpg" or ".jpeg" or ".tga" or ".exr" or ".tif" or ".tiff" or ".bmp"))
                    continue;
                string filename = Path.GetFileNameWithoutExtension(file).ToLowerInvariant();
                if (filename.Contains("albedo") || filename.Contains("basecolor") || filename.Contains("diffuse") || filename == "color")
                    textures.BaseColor = file;
                else if (filename.Contains("normal") && (filename.Contains("dx") || filename.Contains("directx")))
                    textures.NormalDX = file;
                else if (filename.Contains("normal") && (filename.Contains("gl") || filename.Contains("opengl")))
                    textures.NormalGL = file;
                else if (filename.Contains("normal"))
                    textures.Normal = file;
                else if (filename.Contains("roughness") || filename == "rough")
                    textures.Roughness = file;
                else if (filename.Contains("metallic") || filename.Contains("metal") || filename == "metal")
                    textures.Metallic = file;
                else if (filename.Contains("ao") || filename.Contains("ambient") || filename.Contains("occlusion"))
                    textures.AmbientOcclusion = file;
                else if (filename.Contains("displacement") || filename.Contains("disp") || filename.Contains("height") || filename.Contains("hmap"))
                    textures.Displacement = file;
                else if (filename.Contains("emissive") || filename.Contains("emit") || filename.Contains("glow"))
                    textures.Emissive = file;
                else if (filename.Contains("opacity") || filename.Contains("alpha") || filename.Contains("mask"))
                    textures.Opacity = file;
                else if (filename.Contains("cavity"))
                    textures.Cavity = file;
                else if (filename.Contains("curvature"))
                    textures.Curvature = file;
                else if (filename.Contains("fuzz") || filename.Contains("cloth"))
                    textures.Fuzz = file;
                else if (filename.Contains("thickness") || filename.Contains("thick"))
                    textures.Thickness = file;
                else if (filename.Contains("translucency") || filename.Contains("translucent"))
                    textures.Translucency = file;
                else if (filename.Contains("specular") || filename.Contains("spec"))
                    textures.Specular = file;
                else if (filename.Contains("anisotropy") || filename.Contains("aniso"))
                    textures.Anisotropy = file;
                else if (filename.Contains("sheen"))
                    textures.SheenColor = file;
                else if (filename.Contains("clearcoatnormal") || filename.Contains("cc_normal"))
                    textures.ClearCoatNormal = file;
                else if (filename.Contains("clearco rough") || filename.Contains("cc_rough"))
                    textures.ClearCoatRoughness = file;
                else if (filename.Contains("orm") || filename.Contains("packed_orm"))
                    textures.packed_ORM = file;
                else if (filename.Contains("_mr") || filename.Contains("packed_mr"))
                    textures.packed_MR = file;
                else if (filename.Contains("detail_albedo") || filename.Contains("detail_color"))
                    textures.DetailAlbedo = file;
                else if (filename.Contains("detail_normal") || filename.Contains("detail_nrm"))
                    textures.DetailNormal = file;
            }
            textures.Format = InferDominantFormat(textures);
            return textures;
        }

        private static string InferDominantFormat(MegascansTextureSet textures)
        {
            foreach (var path in new[] { textures.BaseColor, textures.Normal, textures.packed_ORM, textures.Roughness })
            {
                if (string.IsNullOrEmpty(path)) continue;
                var ext = Path.GetExtension(path).TrimStart('.').ToLowerInvariant();
                if (ext is "png" or "jpg" or "jpeg" or "tga" or "bmp" or "exr" or "tif" or "tiff")
                    return ext == "jpeg" ? "jpg" : ext;
            }
            return "png";
        }

        /// <summary>
        /// Decodes a Megascans texture file from disk (PNG/JPEG/BMP/TGA) for GPU upload.
        /// Returns false for missing/corrupt/unsupported formats (caller keeps procedural fallback).
        /// </summary>
        public static bool TryDecodeTextureFile(string path, out DecodedImage image)
            => MegascansImageDecoder.TryDecodeFile(path, out image);

        private Megascans3DAssetData Discover3DData(string directory)
        {
            var data = new Megascans3DAssetData();
            var files = Directory.GetFiles(directory, "*.*", SearchOption.AllDirectories);
            var meshFiles = files.Where(f =>
            {
                string ext = Path.GetExtension(f).ToLowerInvariant();
                return ext is (".fbx" or ".obj" or ".gltf" or ".glb" or ".usdc" or ".usda");
            }).ToList();

            data.MeshPath = meshFiles.FirstOrDefault(f =>
                Path.GetFileName(f).ToLowerInvariant().Contains("lod0") ||
                Path.GetFileName(f).ToLowerInvariant().Contains("high") ||
                Path.GetFileName(f).ToLowerInvariant().Contains("detail")) ?? meshFiles.FirstOrDefault() ?? "";

            data.HighPolyPath = meshFiles.FirstOrDefault(f =>
                Path.GetFileName(f).ToLowerInvariant().Contains("high") ||
                Path.GetFileName(f).ToLowerInvariant().Contains("source")) ?? "";

            data.LODPaths = meshFiles
                .Where(f => Path.GetFileName(f).ToLowerInvariant().Contains("lod"))
                .OrderBy(f => f)
                .ToArray();

            data.LODScreenSizes = data.LODPaths.Select((_, i) => 1.0f / (1 << i)).ToArray();

            var collisionFiles = files.Where(f => Path.GetFileName(f).ToLowerInvariant().Contains("collision") || Path.GetFileName(f).ToLowerInvariant().Contains("col"));
            data.CollisionMeshPath = collisionFiles.FirstOrDefault() ?? "";

            return data;
        }

        private MegascansVegetationData DiscoverVegetationData(string directory)
        {
            var data = new MegascansVegetationData();
            var threeDData = Discover3DData(directory);
            data.MeshPath = threeDData.MeshPath;
            data.VariantMeshPaths = Directory.GetFiles(directory, "*.*", SearchOption.AllDirectories)
                .Where(f =>
                {
                    string ext = Path.GetExtension(f).ToLowerInvariant();
                    string name = Path.GetFileName(f).ToLowerInvariant();
                    return ext is (".fbx" or ".obj" or ".gltf" or ".glb") && name.Contains("variant");
                }).ToArray();
            return data;
        }

        private MegascansDecalData DiscoverDecalData(string directory)
        {
            var data = new MegascansDecalData();
            var meshFiles = Directory.GetFiles(directory, "*.*", SearchOption.AllDirectories)
                .Where(f => Path.GetExtension(f).ToLowerInvariant() is (".fbx" or ".obj" or ".gltf" or ".glb"));
            data.MeshPath = meshFiles.FirstOrDefault() ?? "";
            return data;
        }

        private MegascansAtlasData DiscoverAtlasData(string directory)
        {
            return new MegascansAtlasData();
        }

        private string FindMetadataFile(string directory)
        {
            var candidates = new[] { "metadata.json", "scan.json", "asset.json", "info.json", "manifest.json" };
            foreach (var c in candidates)
            {
                string path = Path.Combine(directory, c);
                if (File.Exists(path))
                    return path;
            }
            var jsonFiles = Directory.GetFiles(directory, "*.json", SearchOption.TopDirectoryOnly);
            return jsonFiles.FirstOrDefault();
        }

        private void CopyOrReferenceFiles(MegascansScanEntry entry, string sourceDirectory)
        {
            if (_config.ImportMode == MegascansImportMode.ReferenceOriginal)
            {
                entry.Asset.LibraryPath = Path.GetFullPath(sourceDirectory);
                return;
            }

            string libraryRoot = Synapse.Core.Security.PathSecurity.GetFullPathChecked(_config.LibraryRootPath);
            string targetDir = Synapse.Core.Security.PathSecurity.EnsureUnderRoot(libraryRoot, entry.Asset.LibraryPath);
            Directory.CreateDirectory(targetDir);

            if (_config.ImportMode == MegascansImportMode.CopyToLibrary || _config.ImportMode == MegascansImportMode.MoveToLibrary)
            {
                var allFiles = Directory.GetFiles(sourceDirectory, "*.*", SearchOption.AllDirectories);
                if (allFiles.Length > 10_000)
                    throw new InvalidDataException("Megascans import exceeds 10,000 files.");

                long totalBytes = 0;
                const long maxTotalBytes = 2L * 1024 * 1024 * 1024;
                foreach (var file in allFiles)
                {
                    totalBytes += new FileInfo(file).Length;
                    if (totalBytes > maxTotalBytes)
                        throw new InvalidDataException("Megascans import exceeds 2 GiB.");

                    string relativePath = Path.GetRelativePath(sourceDirectory, file);
                    if (relativePath.Contains("..", StringComparison.Ordinal))
                        continue;
                    string destPath = Path.GetFullPath(Path.Combine(targetDir, relativePath));
                    destPath = Synapse.Core.Security.PathSecurity.EnsureUnderRoot(targetDir, destPath);
                    var destParent = Path.GetDirectoryName(destPath);
                    if (destParent != null)
                        Directory.CreateDirectory(Synapse.Core.Security.PathSecurity.EnsureUnderRoot(targetDir, destParent));
                    File.Copy(file, destPath, overwrite: true);
                    if (_config.ImportMode == MegascansImportMode.MoveToLibrary)
                        try
                        { File.Delete(file); }
                        catch (Exception ex)
                        {
                            SynapseLogger.Default.Warn("MegascansBridge", $"Failed to delete source file '{file}' after import.", ex);
                        }
                }
            }
        }

        public MegascansScanEntry GetImportedAsset(string assetId)
        {
            _importedAssets.TryGetValue(assetId, out var entry);
            return entry;
        }

        public IReadOnlyList<MegascansScanEntry> GetAllImportedAssets()
        {
            return _importedAssets.Values.ToList().AsReadOnly();
        }

        public IReadOnlyList<MegascansScanEntry> SearchImportedAssets(string query)
        {
            if (string.IsNullOrWhiteSpace(query))
                return GetAllImportedAssets();
            var lower = query.ToLowerInvariant();
            return _importedAssets.Values
                .Where(e => e.Asset != null &&
                    (e.Asset.Name.Contains(lower, StringComparison.OrdinalIgnoreCase) ||
                     e.Asset.Tags.Contains(lower, StringComparison.OrdinalIgnoreCase)))
                .ToList().AsReadOnly();
        }

        public IReadOnlyList<MegascansScanEntry> GetBySurfaceType(MegascansSurfaceType surfaceType)
        {
            return _importedAssets.Values
                .Where(e => e.Asset?.AssetType == MegascansAssetType.Surface && e.Asset.SurfaceType == surfaceType)
                .ToList().AsReadOnly();
        }

        public IReadOnlyList<MegascansScanEntry> GetByAssetType(MegascansAssetType assetType)
        {
            return _importedAssets.Values
                .Where(e => e.Asset?.AssetType == assetType)
                .ToList().AsReadOnly();
        }

        public string ExportLibraryManifest(string outputPath)
        {
            var manifest = _importedAssets.Values.Select(e => new
            {
                e.Asset.Id,
                e.Asset.Name,
                e.Asset.AssetType,
                e.Asset.SurfaceType,
                e.Asset.Quality,
                e.Asset.Tags,
                e.ImportSucceeded,
                e.ImportDuration.TotalMilliseconds
            }).ToList();
            var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(outputPath, json, Encoding.UTF8);
            return outputPath;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _importSemaphore?.Dispose();
                _importedAssets.Clear();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        private static Dictionary<MegascansSurfaceType, MaterialPresetCategory> BuildSurfaceCategoryMap()
        {
            return new Dictionary<MegascansSurfaceType, MaterialPresetCategory>
            {
                [MegascansSurfaceType.Concrete] = MaterialPresetCategory.ConcreteSmooth,
                [MegascansSurfaceType.Asphalt] = MaterialPresetCategory.ConcreteRough,
                [MegascansSurfaceType.Brick] = MaterialPresetCategory.Brick,
                [MegascansSurfaceType.Metal] = MaterialPresetCategory.Metal,
                [MegascansSurfaceType.Wood] = MaterialPresetCategory.WoodRaw,
                [MegascansSurfaceType.Stone] = MaterialPresetCategory.StoneRough,
                [MegascansSurfaceType.Marble] = MaterialPresetCategory.StonePolished,
                [MegascansSurfaceType.Tile] = MaterialPresetCategory.Ceramic,
                [MegascansSurfaceType.Dirt] = MaterialPresetCategory.Soil,
                [MegascansSurfaceType.Grass] = MaterialPresetCategory.VegetationGrass,
                [MegascansSurfaceType.Gravel] = MaterialPresetCategory.StoneRough,
                [MegascansSurfaceType.Sand] = MaterialPresetCategory.Sand,
                [MegascansSurfaceType.Snow] = MaterialPresetCategory.Snow,
                [MegascansSurfaceType.Ice] = MaterialPresetCategory.IceClear,
                [MegascansSurfaceType.Water] = MaterialPresetCategory.WaterClear,
                [MegascansSurfaceType.Rock] = MaterialPresetCategory.StoneRough,
                [MegascansSurfaceType.Fabric] = MaterialPresetCategory.FabricNatural,
                [MegascansSurfaceType.Leather] = MaterialPresetCategory.LeatherSmooth,
                [MegascansSurfaceType.Plastic] = MaterialPresetCategory.PlasticGlossy,
                [MegascansSurfaceType.Rubber] = MaterialPresetCategory.Rubber,
                [MegascansSurfaceType.Glass] = MaterialPresetCategory.GlassClear,
                [MegascansSurfaceType.Ceramic] = MaterialPresetCategory.Ceramic,
                [MegascansSurfaceType.Paint] = MaterialPresetCategory.Paint,
                [MegascansSurfaceType.Soil] = MaterialPresetCategory.Soil,
                [MegascansSurfaceType.Moss] = MaterialPresetCategory.VegetationGrass,
                [MegascansSurfaceType.Bark] = MaterialPresetCategory.VegetationBark,
                [MegascansSurfaceType.Basalt] = MaterialPresetCategory.StoneRough,
                [MegascansSurfaceType.Slate] = MaterialPresetCategory.StoneRough,
                [MegascansSurfaceType.Limestone] = MaterialPresetCategory.StoneRough,
                [MegascansSurfaceType.Granite] = MaterialPresetCategory.StoneRough,
                [MegascansSurfaceType.Sandstone] = MaterialPresetCategory.StoneRough,
                [MegascansSurfaceType.Clay] = MaterialPresetCategory.Clay,
            };
        }
    }
}
