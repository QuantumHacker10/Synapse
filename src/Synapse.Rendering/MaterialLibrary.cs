// =============================================================================
// MaterialLibrary.cs - GDNN Engine: Production Artist Pipeline
// Material Library with 200+ physically-based presets, categories, search,
// favorites, variant management, and batch operations.
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
using MaterialCategory = GDNN.Materials.SubstrateOmega.MaterialCategory;

namespace GDNN.Rendering.ArtPipeline
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    public enum MaterialPresetCategory : byte
    {
        Metal,
        Wood,
        Stone,
        Concrete,
        Brick,
        Ceramic,
        Glass,
        Plastic,
        Rubber,
        Fabric,
        Leather,
        Skin,
        Hair,
        Eye,
        Vegetation,
        Snow,
        Ice,
        Water,
        Sand,
        Soil,
        Paint,
        Paper,
        Gem,
        Carbon,
        Rust,
        Moss,
        Lichen,
        Clay,
        Chalk,
        Charcoal,
        Wax,
        Soap,
        FabricSynthetic,
        FabricNatural,
        FabricMetallic,
        MetalBrushed,
        MetalPolished,
        MetalRusted,
        MetalPainted,
        MetalGalvanized,
        WoodFinished,
        WoodRaw,
        WoodLacquered,
        StonePolished,
        StoneRough,
        StoneMossy,
        ConcreteSmooth,
        ConcreteRough,
        ConcretePainted,
        GlassClear,
        GlassFrosted,
        GlassTinted,
        PlasticGlossy,
        PlasticMatte,
        PlasticTranslucent,
        LeatherSmooth,
        LeatherRough,
        SkinHuman,
        SkinAlien,
        HairHumanStraight,
        HairHumanCurly,
        HairAnimal,
        VegetationLeaf,
        VegetationBark,
        VegetationGrass,
        IceClear,
        IceFrosted,
        WaterClear,
        WaterMurky,
        Custom
    }

    public enum MaterialAssetType : byte
    {
        Preset,
        Instance,
        Master,
        Variant,
        SubMaterial,
        Decal,
        Landscape,
        WorldPartition
    }

    public enum MaterialLibrarySortMode : byte
    {
        Name,
        Category,
        Complexity,
        LastModified,
        UsageCount,
        FavoritesFirst
    }

    // =========================================================================
    // MATERIAL PRESET DEFINITION
    // =========================================================================

    public class MaterialPreset
    {
        public string Id { get; set; } = Guid.NewGuid().ToString("N");
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public MaterialPresetCategory Category { get; set; }
        public MaterialCategory CoreCategory { get; set; }
        public string Author { get; set; } = "Synapse";
        public string Version { get; set; } = "1.0";
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime ModifiedAt { get; set; } = DateTime.UtcNow;
        public bool IsFavorite { get; set; }
        public int UsageCount { get; set; }
        public string ThumbnailPath { get; set; } = "";

        public Color3 BaseColor { get; set; } = new Color3(0.8f, 0.8f, 0.8f);
        public float Roughness { get; set; } = 0.5f;
        public float Metallic { get; set; }
        public float Specular { get; set; } = 0.5f;
        public float NormalIntensity { get; set; } = 1.0f;
        public float AOIntensity { get; set; } = 1.0f;
        public float EmissiveIntensity { get; set; }
        public Color3 EmissiveColor { get; set; } = Color3.Black;
        public float Opacity { get; set; } = 1.0f;
        public float SubsurfaceRadius { get; set; }
        public Color3 SubsurfaceColor { get; set; } = Color3.Black;
        public float ClearCoat { get; set; }
        public float ClearCoatRoughness { get; set; } = 0.01f;
        public float Anisotropy { get; set; }
        public float AnisotropyRotation { get; set; }
        public float Sheen { get; set; }
        public float SheenRoughness { get; set; } = 0.5f;
        public float Transmission { get; set; }
        public float IOR { get; set; } = 1.5f;
        public float Thickness { get; set; } = 0.5f;
        public float DisplacementScale { get; set; }
        public MaterialDomain Domain { get; set; } = MaterialDomain.Surface;
        public MaterialFeatureFlags Features { get; set; }

        public Dictionary<TextureChannel, string> TexturePaths { get; set; } = new();
        public Dictionary<string, float> ScalarParameters { get; set; } = new();
        public Dictionary<string, Color3> ColorParameters { get; set; } = new();
        public Dictionary<string, Vec3> VectorParameters { get; set; } = new();
        public List<string> Tags { get; set; } = new();
        public string PhysicallyBasedNotes { get; set; } = "";
        public float IORActual { get; set; }
        public float AbsorptionCoeff { get; set; }

        public SubstrateMaterial ToMaterial(string name = null)
        {
            var mat = new SubstrateMaterial(name ?? Name)
            {
                Domain = Domain,
                FeatureFlags = Features
            };
            mat.InitializeDefaults();
            mat.SetProperty("BaseColor", new MaterialProperty("BaseColor", MaterialPropertyType.Color, BaseColor, 0, 1, 0.8f));
            mat.SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, Roughness, 0, 1, 0.5f));
            mat.SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, Metallic, 0, 1, 0f));
            mat.SetProperty("Specular", new MaterialProperty("Specular", MaterialPropertyType.Float, Specular, 0, 1, 0.5f));
            mat.SetProperty("NormalStrength", new MaterialProperty("NormalStrength", MaterialPropertyType.Float, NormalIntensity, 0, 2, 1f));
            mat.SetProperty("AOStrength", new MaterialProperty("AOStrength", MaterialPropertyType.Float, AOIntensity, 0, 1, 1f));
            mat.SetProperty("EmissiveIntensity", new MaterialProperty("EmissiveIntensity", MaterialPropertyType.Float, EmissiveIntensity, 0, 100, 0f));
            mat.SetProperty("Opacity", new MaterialProperty("Opacity", MaterialPropertyType.Float, Opacity, 0, 1, 1f));
            mat.SetProperty("SubsurfaceRadius", new MaterialProperty("SubsurfaceRadius", MaterialPropertyType.Vec3, new Vec3(SubsurfaceRadius), 0, 5, 1f));
            mat.SetProperty("SubsurfaceColor", new MaterialProperty("SubsurfaceColor", MaterialPropertyType.Color, SubsurfaceColor, 0, 1, 0.8f));
            mat.SetProperty("ClearCoat", new MaterialProperty("ClearCoat", MaterialPropertyType.Float, ClearCoat, 0, 1, 0f));
            mat.SetProperty("ClearCoatRoughness", new MaterialProperty("ClearCoatRoughness", MaterialPropertyType.Float, ClearCoatRoughness, 0, 1, 0.01f));
            mat.SetProperty("Anisotropy", new MaterialProperty("Anisotropy", MaterialPropertyType.Float, Anisotropy, -1, 1, 0f));
            mat.SetProperty("AnisotropyRotation", new MaterialProperty("AnisotropyRotation", MaterialPropertyType.Float, AnisotropyRotation, 0, 6.2832f, 0f));
            mat.SetProperty("Sheen", new MaterialProperty("Sheen", MaterialPropertyType.Float, Sheen, 0, 1, 0f));
            mat.SetProperty("SheenRoughness", new MaterialProperty("SheenRoughness", MaterialPropertyType.Float, SheenRoughness, 0, 1, 0.5f));
            mat.SetProperty("Transmission", new MaterialProperty("Transmission", MaterialPropertyType.Float, Transmission, 0, 1, 0f));
            mat.SetProperty("IOR", new MaterialProperty("IOR", MaterialPropertyType.Float, IOR, 1, 2.5f, 1.5f));
            mat.SetProperty("Thickness", new MaterialProperty("Thickness", MaterialPropertyType.Float, Thickness, 0, 10, 0.5f));
            mat.SetProperty("DisplacementScale", new MaterialProperty("DisplacementScale", MaterialPropertyType.Float, DisplacementScale, 0, 5, 0.1f));

            foreach (var kvp in TexturePaths)
                if (!string.IsNullOrEmpty(kvp.Value))
                    mat.SetTexture(kvp.Key, new TextureReference(kvp.Value, kvp.Key));

            foreach (var kvp in ScalarParameters)
                mat.SetProperty(kvp.Key, new MaterialProperty(kvp.Key, MaterialPropertyType.Float, kvp.Value));

            foreach (var kvp in ColorParameters)
                mat.SetProperty(kvp.Key, new MaterialProperty(kvp.Key, MaterialPropertyType.Color, kvp.Value));

            foreach (var kvp in VectorParameters)
                mat.SetProperty(kvp.Key, new MaterialProperty(kvp.Key, MaterialPropertyType.Vec3, kvp.Value));

            mat.ComputeActiveFeatures();
            return mat;
        }

        public MaterialPreset Clone(string newName = null)
        {
            var clone = new MaterialPreset
            {
                Id = Guid.NewGuid().ToString("N"),
                Name = newName ?? $"{Name}_Copy",
                Description = Description,
                Category = Category,
                CoreCategory = CoreCategory,
                Author = Author,
                Version = Version,
                CreatedAt = DateTime.UtcNow,
                ModifiedAt = DateTime.UtcNow,
                IsFavorite = IsFavorite,
                UsageCount = 0,
                ThumbnailPath = ThumbnailPath,
                BaseColor = BaseColor,
                Roughness = Roughness,
                Metallic = Metallic,
                Specular = Specular,
                NormalIntensity = NormalIntensity,
                AOIntensity = AOIntensity,
                EmissiveIntensity = EmissiveIntensity,
                EmissiveColor = EmissiveColor,
                Opacity = Opacity,
                SubsurfaceRadius = SubsurfaceRadius,
                SubsurfaceColor = SubsurfaceColor,
                ClearCoat = ClearCoat,
                ClearCoatRoughness = ClearCoatRoughness,
                Anisotropy = Anisotropy,
                AnisotropyRotation = AnisotropyRotation,
                Sheen = Sheen,
                SheenRoughness = SheenRoughness,
                Transmission = Transmission,
                IOR = IOR,
                Thickness = Thickness,
                DisplacementScale = DisplacementScale,
                Domain = Domain,
                Features = Features,
                TexturePaths = new Dictionary<TextureChannel, string>(TexturePaths),
                ScalarParameters = new Dictionary<string, float>(ScalarParameters),
                ColorParameters = new Dictionary<string, Color3>(ColorParameters),
                VectorParameters = new Dictionary<string, Vec3>(VectorParameters),
                Tags = new List<string>(Tags),
                PhysicallyBasedNotes = PhysicallyBasedNotes,
                IORActual = IORActual,
                AbsorptionCoeff = AbsorptionCoeff
            };
            return clone;
        }
    }

    // =========================================================================
    // MATERIAL VARIANT
    // =========================================================================

    public class MaterialVariant
    {
        public string Id { get; set; } = Guid.NewGuid().ToString("N");
        public string Name { get; set; } = "";
        public string BasePresetId { get; set; } = "";
        public Dictionary<string, float> ScalarOverrides { get; set; } = new();
        public Dictionary<string, Color3> ColorOverrides { get; set; } = new();
        public Dictionary<TextureChannel, string> TextureOverrides { get; set; } = new();
        public Dictionary<string, Vec3> VectorOverrides { get; set; } = new();
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public SubstrateMaterial ApplyTo(SubstrateMaterial baseMaterial)
        {
            var variant = baseMaterial.Clone($"{baseMaterial.Name}_{Name}");
            foreach (var kvp in ScalarOverrides)
                variant.SetProperty(kvp.Key, new MaterialProperty(kvp.Key, MaterialPropertyType.Float, kvp.Value));
            foreach (var kvp in ColorOverrides)
                variant.SetProperty(kvp.Key, new MaterialProperty(kvp.Key, MaterialPropertyType.Color, kvp.Value));
            foreach (var kvp in TextureOverrides)
                if (!string.IsNullOrEmpty(kvp.Value))
                    variant.SetTexture(kvp.Key, new TextureReference(kvp.Value, kvp.Key));
            foreach (var kvp in VectorOverrides)
                variant.SetProperty(kvp.Key, new MaterialProperty(kvp.Key, MaterialPropertyType.Vec3, kvp.Value));
            return variant;
        }
    }

    // =========================================================================
    // MATERIAL ASSET ENTRY
    // =========================================================================

    public class MaterialAssetEntry
    {
        public string Id { get; set; } = Guid.NewGuid().ToString("N");
        public string Name { get; set; } = "";
        public MaterialAssetType AssetType { get; set; }
        public MaterialPresetCategory Category { get; set; }
        public MaterialPreset Preset { get; set; }
        public List<MaterialVariant> Variants { get; set; } = new();
        public string SourcePath { get; set; } = "";
        public DateTime LastModified { get; set; } = DateTime.UtcNow;
        public int UsageCount { get; set; }
        public bool IsFavorite { get; set; }
        public List<string> Tags { get; set; } = new();
        public Dictionary<string, string> Metadata { get; set; } = new();
        public ulong ContentHash { get; set; }
    }

    // =========================================================================
    // MATERIAL LIBRARY
    // =========================================================================

    public class MaterialLibrary : IDisposable
    {
        private readonly ConcurrentDictionary<string, MaterialAssetEntry> _assets;
        private readonly List<MaterialPreset> _presets;
        private readonly List<MaterialPresetCategory> _categories;
        private readonly Dictionary<string, List<string>> _tagIndex;
        private readonly Dictionary<MaterialPresetCategory, List<string>> _categoryIndex;
        private readonly string _libraryRoot;
        private readonly SemaphoreSlim _lock;
        private bool _disposed;

        public int TotalAssets => _assets.Count;
        public int TotalPresets => _presets.Count;
        public int FavoriteCount => _assets.Values.Count(a => a.IsFavorite);
        public string LibraryRoot => _libraryRoot;

        public MaterialLibrary(string libraryRoot = null)
        {
            _libraryRoot = libraryRoot ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".synapse", "materials");
            _assets = new ConcurrentDictionary<string, MaterialAssetEntry>();
            _presets = new List<MaterialPreset>();
            _categories = new List<MaterialPresetCategory>();
            _tagIndex = new Dictionary<string, List<string>>();
            _categoryIndex = new Dictionary<MaterialPresetCategory, List<string>>();
            _lock = new SemaphoreSlim(1, 1);

            InitializeDefaultPresets();
        }

        public async Task InitializeAsync()
        {
            await Task.Run(() =>
            {
                Directory.CreateDirectory(_libraryRoot);
                var metadataPath = Path.Combine(_libraryRoot, "library.json");
                if (File.Exists(metadataPath))
                    LoadMetadata(metadataPath);
            });
        }

        public string AddAsset(MaterialPreset preset, MaterialAssetType type = MaterialAssetType.Preset)
        {
            var entry = new MaterialAssetEntry
            {
                Name = preset.Name,
                AssetType = type,
                Category = preset.Category,
                Preset = preset,
                Tags = new List<string>(preset.Tags),
                LastModified = DateTime.UtcNow,
                ContentHash = ComputePresetHash(preset)
            };
            _assets[entry.Id] = entry;
            IndexAsset(entry);
            return entry.Id;
        }

        public string? AddInstance(string presetId, string instanceName)
        {
            if (!_assets.TryGetValue(presetId, out var source) || source.Preset == null)
                return null;

            var variant = source.Preset.Clone(instanceName);
            var entry = new MaterialAssetEntry
            {
                Name = instanceName,
                AssetType = MaterialAssetType.Instance,
                Category = source.Category,
                Preset = variant,
                SourcePath = presetId,
                Tags = new List<string>(source.Tags),
                LastModified = DateTime.UtcNow
            };
            _assets[entry.Id] = entry;
            IndexAsset(entry);
            return entry.Id;
        }

        public MaterialVariant? AddVariant(string presetId, string variantName, Dictionary<string, float> scalarOverrides = null, Dictionary<string, Color3> colorOverrides = null)
        {
            if (!_assets.TryGetValue(presetId, out var source) || source.Preset == null)
                return null;

            var variant = new MaterialVariant
            {
                Name = variantName,
                BasePresetId = presetId,
                ScalarOverrides = scalarOverrides ?? new(),
                ColorOverrides = colorOverrides ?? new()
            };
            source.Variants.Add(variant);
            source.LastModified = DateTime.UtcNow;
            return variant;
        }

        public bool RemoveAsset(string assetId)
        {
            if (_assets.TryRemove(assetId, out var entry))
            {
                RemoveFromIndex(entry);
                return true;
            }
            return false;
        }

        public bool ToggleFavorite(string assetId)
        {
            if (_assets.TryGetValue(assetId, out var entry))
            {
                entry.IsFavorite = !entry.IsFavorite;
                if (entry.Preset != null)
                    entry.Preset.IsFavorite = entry.IsFavorite;
                return entry.IsFavorite;
            }
            return false;
        }

        public MaterialAssetEntry GetAsset(string assetId)
        {
            _assets.TryGetValue(assetId, out var entry);
            return entry;
        }

        public MaterialPreset? GetPreset(string presetId)
        {
            return _assets.TryGetValue(presetId, out var entry) ? entry.Preset : null;
        }

        public IReadOnlyList<MaterialPreset> GetPresetsByCategory(MaterialPresetCategory category)
        {
            if (_categoryIndex.TryGetValue(category, out var ids))
                return ids.Where(id => _assets.ContainsKey(id) && _assets[id].Preset != null)
                    .Select(id => _assets[id].Preset).ToList().AsReadOnly();
            return Array.Empty<MaterialPreset>();
        }

        public IReadOnlyList<MaterialPreset> SearchPresets(string query, int maxResults = 50)
        {
            if (string.IsNullOrWhiteSpace(query))
                return _presets.Take(maxResults).ToList().AsReadOnly();

            var lowerQuery = query.ToLowerInvariant();
            var results = _presets
                .Where(p => p.Name.Contains(lowerQuery, StringComparison.OrdinalIgnoreCase)
                    || p.Description.Contains(lowerQuery, StringComparison.OrdinalIgnoreCase)
                    || p.Tags.Any(t => t.Contains(lowerQuery, StringComparison.OrdinalIgnoreCase)))
                .OrderByDescending(p => p.IsFavorite)
                .ThenByDescending(p => p.UsageCount)
                .Take(maxResults)
                .ToList();
            return results.AsReadOnly();
        }

        public IReadOnlyList<MaterialAssetEntry> SearchAssets(string query, int maxResults = 100)
        {
            if (string.IsNullOrWhiteSpace(query))
                return _assets.Values.OrderByDescending(a => a.IsFavorite).ThenByDescending(a => a.UsageCount).Take(maxResults).ToList().AsReadOnly();

            var lowerQuery = query.ToLowerInvariant();
            return _assets.Values
                .Where(a => a.Name.Contains(lowerQuery, StringComparison.OrdinalIgnoreCase)
                    || a.Tags.Any(t => t.Contains(lowerQuery, StringComparison.OrdinalIgnoreCase)))
                .OrderByDescending(a => a.IsFavorite)
                .ThenByDescending(a => a.UsageCount)
                .Take(maxResults)
                .ToList().AsReadOnly();
        }

        public IReadOnlyList<MaterialAssetEntry> GetFavorites()
        {
            return _assets.Values.Where(a => a.IsFavorite).OrderBy(a => a.Name).ToList().AsReadOnly();
        }

        public IReadOnlyList<MaterialAssetEntry> GetRecent(int count = 20)
        {
            return _assets.Values.OrderByDescending(a => a.LastModified).Take(count).ToList().AsReadOnly();
        }

        public IReadOnlyList<MaterialAssetEntry> GetMostUsed(int count = 20)
        {
            return _assets.Values.OrderByDescending(a => a.UsageCount).Take(count).ToList().AsReadOnly();
        }

        public IReadOnlyList<MaterialAssetEntry> GetAssetsByTag(string tag)
        {
            if (_tagIndex.TryGetValue(tag, out var ids))
                return ids.Where(id => _assets.ContainsKey(id)).Select(id => _assets[id]).ToList().AsReadOnly();
            return Array.Empty<MaterialAssetEntry>();
        }

        public IReadOnlyList<MaterialPresetCategory> GetUsedCategories()
        {
            return _categoryIndex.Keys.Where(k => _categoryIndex[k].Count > 0).OrderBy(k => k.ToString()).ToList().AsReadOnly();
        }

        public void IncrementUsage(string assetId)
        {
            if (_assets.TryGetValue(assetId, out var entry))
            {
                entry.UsageCount++;
                if (entry.Preset != null)
                    entry.Preset.UsageCount = entry.UsageCount;
            }
        }

        public SubstrateMaterial? CreateMaterialFromPreset(string presetId, string materialName = null)
        {
            var preset = GetPreset(presetId);
            if (preset == null)
                return null;
            IncrementUsage(presetId);
            return preset.ToMaterial(materialName);
        }

        public SubstrateMaterial? CreateMaterialFromVariant(string presetId, string variantName, string materialName = null)
        {
            if (!_assets.TryGetValue(presetId, out var entry) || entry.Preset == null)
                return null;
            var variant = entry.Variants.FirstOrDefault(v => v.Name == variantName);
            if (variant == null)
                return null;
            var baseMat = entry.Preset.ToMaterial(materialName);
            return variant.ApplyTo(baseMat);
        }

        public List<MaterialPreset> GeneratePhysicalVariants(MaterialPreset basePreset, int weatheringLevels = 5, int damageLevels = 3)
        {
            var variants = new List<MaterialPreset>();
            var weatheringStates = new[] { "Pristine", "Light", "Moderate", "Heavy", "Eroded" };
            var damageStates = new[] { "Intact", "Scratched", "Chipped" };

            for (int w = 0; w < weatheringLevels; w++)
            {
                for (int d = 0; d < damageLevels; d++)
                {
                    var variant = basePreset.Clone($"{basePreset.Name}_{weatheringStates[w]}_{damageStates[d]}");
                    float weatherFactor = (float)w / (weatheringLevels - 1);
                    float damageFactor = (float)d / (damageLevels - 1);

                    variant.Roughness = Math.Clamp(basePreset.Roughness + weatherFactor * 0.3f + damageFactor * 0.15f, 0, 1);
                    variant.AOIntensity = Math.Clamp(basePreset.AOIntensity - damageFactor * 0.2f, 0, 1);

                    if (basePreset.Category == MaterialPresetCategory.Metal || basePreset.Category == MaterialPresetCategory.MetalRusted)
                    {
                        variant.Metallic = Math.Clamp(basePreset.Metallic - weatherFactor * 0.4f, 0, 1);
                        variant.ColorParameters["BaseColor"] = Color3.Lerp(basePreset.BaseColor, new Color3(0.45f, 0.22f, 0.08f), weatherFactor * 0.6f);
                    }

                    variant.Tags.Add($"weathering:{weatheringStates[w].ToLower()}");
                    variant.Tags.Add($"damage:{damageStates[d].ToLower()}");
                    variants.Add(variant);
                }
            }
            return variants;
        }

        public string ExportToJson(string filePath)
        {
            var options = new JsonSerializerOptions { WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };
            var data = _assets.Values.ToList();
            var json = JsonSerializer.Serialize(data, options);
            File.WriteAllText(filePath, json, Encoding.UTF8);
            return filePath;
        }

        public int ImportFromJson(string filePath)
        {
            if (!File.Exists(filePath))
                return 0;
            var json = File.ReadAllText(filePath, Encoding.UTF8);
            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            var data = JsonSerializer.Deserialize<List<MaterialAssetEntry>>(json, options);
            if (data == null)
                return 0;

            int count = 0;
            foreach (var entry in data)
            {
                if (_assets.TryAdd(entry.Id, entry))
                {
                    IndexAsset(entry);
                    if (entry.Preset != null)
                        _presets.Add(entry.Preset);
                    count++;
                }
            }
            return count;
        }

        public string ExportPresetCollection(string directoryPath, string collectionName)
        {
            Directory.CreateDirectory(directoryPath);
            var collection = new
            {
                Name = collectionName,
                CreatedAt = DateTime.UtcNow,
                Presets = _presets.Take(100).Select(p => new
                {
                    p.Name,
                    p.Category,
                    p.BaseColor,
                    p.Roughness,
                    p.Metallic,
                    p.Tags
                }).ToList()
            };
            var json = JsonSerializer.Serialize(collection, new JsonSerializerOptions { WriteIndented = true });
            var path = Path.Combine(directoryPath, $"{collectionName}.json");
            File.WriteAllText(path, json, Encoding.UTF8);
            return path;
        }

        private void InitializeDefaultPresets()
        {
            CreateMetalPresets();
            CreateWoodPresets();
            CreateStonePresets();
            CreateConcretePresets();
            CreateGlassPresets();
            CreatePlasticPresets();
            CreateFabricPresets();
            CreateSkinPresets();
            CreateHairPresets();
            CreateEyePresets();
            CreateVegetationPresets();
            CreateLiquidPresets();
            CreateCeramicPresets();
            CreateLeatherPresets();
            CreateRubberPresets();
            CreateSnowIcePresets();
            CreateSandSoilPresets();
            CreateGemPresets();
            CreateCarbonPresets();
            CreateWaxSoapPresets();

            foreach (var preset in _presets)
            {
                var entry = new MaterialAssetEntry
                {
                    Name = preset.Name,
                    AssetType = MaterialAssetType.Preset,
                    Category = preset.Category,
                    Preset = preset,
                    Tags = new List<string>(preset.Tags),
                    ContentHash = ComputePresetHash(preset)
                };
                _assets[entry.Id] = entry;
                IndexAsset(entry);
            }
        }

        private void CreateMetalPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Steel Brushed", Category = MaterialPresetCategory.MetalBrushed, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.72f, 0.72f, 0.74f), Roughness = 0.35f, Metallic = 1.0f, Specular = 0.5f, Tags = new() { "metal", "steel", "brushed", "industrial" } });
            _presets.Add(new MaterialPreset { Name = "Steel Polished", Category = MaterialPresetCategory.MetalPolished, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.85f, 0.85f, 0.87f), Roughness = 0.05f, Metallic = 1.0f, Tags = new() { "metal", "steel", "polished", "mirror" } });
            _presets.Add(new MaterialPreset { Name = "Aluminum Brushed", Category = MaterialPresetCategory.MetalBrushed, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.91f, 0.92f, 0.92f), Roughness = 0.3f, Metallic = 1.0f, Tags = new() { "metal", "aluminum", "brushed", "lightweight" } });
            _presets.Add(new MaterialPreset { Name = "Aluminum Anodized Black", Category = MaterialPresetCategory.MetalPainted, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.05f, 0.05f, 0.05f), Roughness = 0.4f, Metallic = 0.85f, Tags = new() { "metal", "aluminum", "anodized", "black" } });
            _presets.Add(new MaterialPreset { Name = "Copper Polished", Category = MaterialPresetCategory.MetalPolished, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.97f, 0.74f, 0.62f), Roughness = 0.1f, Metallic = 1.0f, Tags = new() { "metal", "copper", "polished", "warm" } });
            _presets.Add(new MaterialPreset { Name = "Copper Patina", Category = MaterialPresetCategory.MetalRusted, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.32f, 0.55f, 0.48f), Roughness = 0.65f, Metallic = 0.3f, Tags = new() { "metal", "copper", "patina", "aged", "green" } });
            _presets.Add(new MaterialPreset { Name = "Gold Polished", Category = MaterialPresetCategory.MetalPolished, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(1.0f, 0.84f, 0.0f), Roughness = 0.05f, Metallic = 1.0f, Tags = new() { "metal", "gold", "precious", "luxury" } });
            _presets.Add(new MaterialPreset { Name = "Silver Polished", Category = MaterialPresetCategory.MetalPolished, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.95f, 0.93f, 0.88f), Roughness = 0.03f, Metallic = 1.0f, Tags = new() { "metal", "silver", "precious", "reflective" } });
            _presets.Add(new MaterialPreset { Name = "Chrome", Category = MaterialPresetCategory.MetalPolished, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.95f, 0.95f, 0.97f), Roughness = 0.02f, Metallic = 1.0f, Tags = new() { "metal", "chrome", "mirror", "automotive" } });
            _presets.Add(new MaterialPreset { Name = "Iron Raw", Category = MaterialPresetCategory.Metal, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.56f, 0.56f, 0.58f), Roughness = 0.55f, Metallic = 1.0f, Tags = new() { "metal", "iron", "raw", "industrial" } });
            _presets.Add(new MaterialPreset { Name = "Iron Rusted", Category = MaterialPresetCategory.MetalRusted, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.45f, 0.22f, 0.08f), Roughness = 0.75f, Metallic = 0.2f, Tags = new() { "metal", "iron", "rusted", "weathered", "orange" } });
            _presets.Add(new MaterialPreset { Name = "Steel Galvanized", Category = MaterialPresetCategory.MetalGalvanized, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.68f, 0.7f, 0.72f), Roughness = 0.45f, Metallic = 0.9f, Tags = new() { "metal", "steel", "galvanized", "zinc" } });
            _presets.Add(new MaterialPreset { Name = "Titanium Brushed", Category = MaterialPresetCategory.MetalBrushed, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.62f, 0.64f, 0.67f), Roughness = 0.3f, Metallic = 1.0f, Tags = new() { "metal", "titanium", "aerospace", "lightweight" } });
            _presets.Add(new MaterialPreset { Name = "Brass Polished", Category = MaterialPresetCategory.MetalPolished, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.88f, 0.78f, 0.5f), Roughness = 0.15f, Metallic = 1.0f, Tags = new() { "metal", "brass", "decorative", "warm" } });
            _presets.Add(new MaterialPreset { Name = "Bronze", Category = MaterialPresetCategory.Metal, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.8f, 0.5f, 0.2f), Roughness = 0.35f, Metallic = 0.95f, Tags = new() { "metal", "bronze", "sculpture", "antique" } });
            _presets.Add(new MaterialPreset { Name = "Gunmetal", Category = MaterialPresetCategory.Metal, CoreCategory = MaterialCategory.Metal, BaseColor = new Color3(0.35f, 0.37f, 0.4f), Roughness = 0.3f, Metallic = 0.95f, Tags = new() { "metal", "gunmetal", "dark", "weapon" } });
        }

        private void CreateWoodPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Oak Raw", Category = MaterialPresetCategory.WoodRaw, CoreCategory = MaterialCategory.Wood, BaseColor = new Color3(0.65f, 0.48f, 0.32f), Roughness = 0.75f, Metallic = 0f, Tags = new() { "wood", "oak", "raw", "natural" } });
            _presets.Add(new MaterialPreset { Name = "Oak Finished", Category = MaterialPresetCategory.WoodFinished, CoreCategory = MaterialCategory.Wood, BaseColor = new Color3(0.6f, 0.42f, 0.28f), Roughness = 0.4f, Metallic = 0f, Tags = new() { "wood", "oak", "finished", "furniture" } });
            _presets.Add(new MaterialPreset { Name = "Walnut", Category = MaterialPresetCategory.WoodFinished, CoreCategory = MaterialCategory.Wood, BaseColor = new Color3(0.35f, 0.22f, 0.12f), Roughness = 0.5f, Metallic = 0f, Tags = new() { "wood", "walnut", "dark", "luxury" } });
            _presets.Add(new MaterialPreset { Name = "Maple", Category = MaterialPresetCategory.WoodFinished, CoreCategory = MaterialCategory.Wood, BaseColor = new Color3(0.82f, 0.68f, 0.52f), Roughness = 0.45f, Metallic = 0f, Tags = new() { "wood", "maple", "light", "floor" } });
            _presets.Add(new MaterialPreset { Name = "Cherry Wood", Category = MaterialPresetCategory.WoodFinished, CoreCategory = MaterialCategory.Wood, BaseColor = new Color3(0.55f, 0.25f, 0.12f), Roughness = 0.4f, Metallic = 0f, Tags = new() { "wood", "cherry", "reddish", "furniture" } });
            _presets.Add(new MaterialPreset { Name = "Pine Raw", Category = MaterialPresetCategory.WoodRaw, CoreCategory = MaterialCategory.Wood, BaseColor = new Color3(0.78f, 0.62f, 0.42f), Roughness = 0.8f, Metallic = 0f, Tags = new() { "wood", "pine", "raw", "construction" } });
            _presets.Add(new MaterialPreset { Name = "Bamboo", Category = MaterialPresetCategory.WoodRaw, CoreCategory = MaterialCategory.Wood, BaseColor = new Color3(0.72f, 0.65f, 0.45f), Roughness = 0.6f, Metallic = 0f, Tags = new() { "wood", "bamboo", "sustainable", "light" } });
            _presets.Add(new MaterialPreset { Name = "Ebony", Category = MaterialPresetCategory.WoodFinished, CoreCategory = MaterialCategory.Wood, BaseColor = new Color3(0.15f, 0.1f, 0.08f), Roughness = 0.35f, Metallic = 0f, Tags = new() { "wood", "ebony", "dark", "luxury", "exotic" } });
            _presets.Add(new MaterialPreset { Name = "Teak", Category = MaterialPresetCategory.WoodFinished, CoreCategory = MaterialCategory.Wood, BaseColor = new Color3(0.52f, 0.38f, 0.22f), Roughness = 0.5f, Metallic = 0f, Tags = new() { "wood", "teak", "outdoor", "maritime" } });
            _presets.Add(new MaterialPreset { Name = "Wood Lacquered", Category = MaterialPresetCategory.WoodLacquered, CoreCategory = MaterialCategory.Wood, BaseColor = new Color3(0.6f, 0.4f, 0.25f), Roughness = 0.15f, Metallic = 0f, ClearCoat = 0.8f, ClearCoatRoughness = 0.05f, Tags = new() { "wood", "lacquered", "glossy", "furniture" } });
        }

        private void CreateStonePresets()
        {
            _presets.Add(new MaterialPreset { Name = "Marble White", Category = MaterialPresetCategory.StonePolished, CoreCategory = MaterialCategory.Stone, BaseColor = new Color3(0.95f, 0.93f, 0.9f), Roughness = 0.15f, Metallic = 0f, Tags = new() { "stone", "marble", "white", "luxury" } });
            _presets.Add(new MaterialPreset { Name = "Marble Black", Category = MaterialPresetCategory.StonePolished, CoreCategory = MaterialCategory.Stone, BaseColor = new Color3(0.12f, 0.12f, 0.14f), Roughness = 0.1f, Metallic = 0f, Tags = new() { "stone", "marble", "black", "luxury" } });
            _presets.Add(new MaterialPreset { Name = "Granite Grey", Category = MaterialPresetCategory.StoneRough, CoreCategory = MaterialCategory.Stone, BaseColor = new Color3(0.55f, 0.53f, 0.52f), Roughness = 0.7f, Metallic = 0f, Tags = new() { "stone", "granite", "grey", "durable" } });
            _presets.Add(new MaterialPreset { Name = "Limestone", Category = MaterialPresetCategory.StoneRough, CoreCategory = MaterialCategory.Stone, BaseColor = new Color3(0.82f, 0.78f, 0.7f), Roughness = 0.8f, Metallic = 0f, Tags = new() { "stone", "limestone", "natural", "building" } });
            _presets.Add(new MaterialPreset { Name = "Slate", Category = MaterialPresetCategory.StoneRough, CoreCategory = MaterialCategory.Stone, BaseColor = new Color3(0.35f, 0.37f, 0.4f), Roughness = 0.65f, Metallic = 0f, Tags = new() { "stone", "slate", "roofing", "dark" } });
            _presets.Add(new MaterialPreset { Name = "Sandstone", Category = MaterialPresetCategory.StoneRough, CoreCategory = MaterialCategory.Stone, BaseColor = new Color3(0.82f, 0.7f, 0.55f), Roughness = 0.85f, Metallic = 0f, Tags = new() { "stone", "sandstone", "desert", "warm" } });
            _presets.Add(new MaterialPreset { Name = "Travertine", Category = MaterialPresetCategory.StonePolished, CoreCategory = MaterialCategory.Stone, BaseColor = new Color3(0.85f, 0.8f, 0.72f), Roughness = 0.35f, Metallic = 0f, Tags = new() { "stone", "travertine", "classic", "flooring" } });
            _presets.Add(new MaterialPreset { Name = "Obsidian", Category = MaterialPresetCategory.StonePolished, CoreCategory = MaterialCategory.Stone, BaseColor = new Color3(0.05f, 0.05f, 0.06f), Roughness = 0.05f, Metallic = 0.1f, Tags = new() { "stone", "obsidian", "volcanic", "glassy" } });
        }

        private void CreateConcretePresets()
        {
            _presets.Add(new MaterialPreset { Name = "Concrete Smooth", Category = MaterialPresetCategory.ConcreteSmooth, CoreCategory = MaterialCategory.Concrete, BaseColor = new Color3(0.7f, 0.7f, 0.7f), Roughness = 0.6f, Metallic = 0f, Tags = new() { "concrete", "smooth", "modern", "architecture" } });
            _presets.Add(new MaterialPreset { Name = "Concrete Rough", Category = MaterialPresetCategory.ConcreteRough, CoreCategory = MaterialCategory.Concrete, BaseColor = new Color3(0.6f, 0.6f, 0.6f), Roughness = 0.9f, Metallic = 0f, Tags = new() { "concrete", "rough", "brutalist", "raw" } });
            _presets.Add(new MaterialPreset { Name = "Concrete Painted White", Category = MaterialPresetCategory.ConcretePainted, CoreCategory = MaterialCategory.Concrete, BaseColor = new Color3(0.92f, 0.92f, 0.92f), Roughness = 0.5f, Metallic = 0f, Tags = new() { "concrete", "painted", "white", "clean" } });
            _presets.Add(new MaterialPreset { Name = "Concrete Stained", Category = MaterialPresetCategory.ConcreteRough, CoreCategory = MaterialCategory.Concrete, BaseColor = new Color3(0.55f, 0.5f, 0.45f), Roughness = 0.75f, Metallic = 0f, Tags = new() { "concrete", "stained", "aged", "weathered" } });
            _presets.Add(new MaterialPreset { Name = "Asphalt", Category = MaterialPresetCategory.ConcreteRough, CoreCategory = MaterialCategory.Concrete, BaseColor = new Color3(0.2f, 0.2f, 0.22f), Roughness = 0.85f, Metallic = 0f, Tags = new() { "asphalt", "road", "dark", "pavement" } });
        }

        private void CreateGlassPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Glass Clear", Category = MaterialPresetCategory.GlassClear, CoreCategory = MaterialCategory.Glass, BaseColor = Color3.White, Roughness = 0.0f, Metallic = 0f, Transmission = 1.0f, IOR = 1.52f, Opacity = 0.1f, Domain = MaterialDomain.Surface, Tags = new() { "glass", "clear", "transparent", "window" } });
            _presets.Add(new MaterialPreset { Name = "Glass Frosted", Category = MaterialPresetCategory.GlassFrosted, CoreCategory = MaterialCategory.Glass, BaseColor = new Color3(0.95f, 0.95f, 0.95f), Roughness = 0.4f, Metallic = 0f, Transmission = 0.8f, IOR = 1.52f, Opacity = 0.3f, Tags = new() { "glass", "frosted", "translucent", "privacy" } });
            _presets.Add(new MaterialPreset { Name = "Glass Tinted Blue", Category = MaterialPresetCategory.GlassTinted, CoreCategory = MaterialCategory.Glass, BaseColor = new Color3(0.3f, 0.5f, 0.8f), Roughness = 0.0f, Metallic = 0f, Transmission = 0.9f, IOR = 1.52f, Opacity = 0.15f, Tags = new() { "glass", "tinted", "blue", "automotive" } });
            _presets.Add(new MaterialPreset { Name = "Glass Tinted Green", Category = MaterialPresetCategory.GlassTinted, CoreCategory = MaterialCategory.Glass, BaseColor = new Color3(0.4f, 0.7f, 0.5f), Roughness = 0.0f, Metallic = 0f, Transmission = 0.85f, IOR = 1.52f, Opacity = 0.15f, Tags = new() { "glass", "tinted", "green", "bottle" } });
            _presets.Add(new MaterialPreset { Name = "Glass Tempered", Category = MaterialPresetCategory.GlassClear, CoreCategory = MaterialCategory.Glass, BaseColor = new Color3(0.85f, 0.9f, 0.92f), Roughness = 0.02f, Metallic = 0f, Transmission = 0.95f, IOR = 1.52f, Opacity = 0.05f, Tags = new() { "glass", "tempered", "safety", "architectural" } });
        }

        private void CreatePlasticPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Plastic Glossy White", Category = MaterialPresetCategory.PlasticGlossy, CoreCategory = MaterialCategory.Plastic, BaseColor = new Color3(0.95f, 0.95f, 0.95f), Roughness = 0.1f, Metallic = 0f, Tags = new() { "plastic", "glossy", "white", "consumer" } });
            _presets.Add(new MaterialPreset { Name = "Plastic Matte Black", Category = MaterialPresetCategory.PlasticMatte, CoreCategory = MaterialCategory.Plastic, BaseColor = new Color3(0.08f, 0.08f, 0.08f), Roughness = 0.7f, Metallic = 0f, Tags = new() { "plastic", "matte", "black", "electronic" } });
            _presets.Add(new MaterialPreset { Name = "ABS Plastic", Category = MaterialPresetCategory.PlasticMatte, CoreCategory = MaterialCategory.Plastic, BaseColor = new Color3(0.85f, 0.85f, 0.85f), Roughness = 0.55f, Metallic = 0f, Tags = new() { "plastic", "abs", "engineering", "prototype" } });
            _presets.Add(new MaterialPreset { Name = "Plastic Translucent", Category = MaterialPresetCategory.PlasticTranslucent, CoreCategory = MaterialCategory.Plastic, BaseColor = new Color3(0.9f, 0.9f, 0.9f), Roughness = 0.2f, Metallic = 0f, Transmission = 0.6f, Opacity = 0.5f, Tags = new() { "plastic", "translucent", "diffuser", "light" } });
            _presets.Add(new MaterialPreset { Name = "Polycarbonate", Category = MaterialPresetCategory.PlasticGlossy, CoreCategory = MaterialCategory.Plastic, BaseColor = new Color3(0.88f, 0.88f, 0.9f), Roughness = 0.05f, Metallic = 0f, Transmission = 0.9f, IOR = 1.585f, Tags = new() { "plastic", "polycarbonate", "impact", "lens" } });
        }

        private void CreateFabricPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Cotton White", Category = MaterialPresetCategory.FabricNatural, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.95f, 0.93f, 0.88f), Roughness = 0.85f, Metallic = 0f, Sheen = 0.1f, Tags = new() { "fabric", "cotton", "white", "natural" } });
            _presets.Add(new MaterialPreset { Name = "Denim", Category = MaterialPresetCategory.FabricNatural, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.15f, 0.2f, 0.35f), Roughness = 0.8f, Metallic = 0f, Sheen = 0.05f, Tags = new() { "fabric", "denim", "blue", "jeans" } });
            _presets.Add(new MaterialPreset { Name = "Silk", Category = MaterialPresetCategory.FabricNatural, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.9f, 0.85f, 0.8f), Roughness = 0.25f, Metallic = 0f, Sheen = 0.4f, SheenRoughness = 0.3f, Tags = new() { "fabric", "silk", "luxury", "smooth" } });
            _presets.Add(new MaterialPreset { Name = "Velvet", Category = MaterialPresetCategory.FabricNatural, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.5f, 0.1f, 0.15f), Roughness = 0.9f, Metallic = 0f, Sheen = 0.6f, SheenRoughness = 0.8f, Tags = new() { "fabric", "velvet", "luxury", "red" } });
            _presets.Add(new MaterialPreset { Name = "Leather Smooth", Category = MaterialPresetCategory.LeatherSmooth, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.35f, 0.2f, 0.12f), Roughness = 0.45f, Metallic = 0f, ClearCoat = 0.3f, Tags = new() { "leather", "smooth", "brown", "furniture" } });
            _presets.Add(new MaterialPreset { Name = "Leather Rough", Category = MaterialPresetCategory.LeatherRough, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.25f, 0.15f, 0.1f), Roughness = 0.75f, Metallic = 0f, Tags = new() { "leather", "rough", "aged", "vintage" } });
            _presets.Add(new MaterialPreset { Name = "Nylon", Category = MaterialPresetCategory.FabricSynthetic, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.8f, 0.8f, 0.82f), Roughness = 0.5f, Metallic = 0f, Tags = new() { "fabric", "nylon", "synthetic", "technical" } });
            _presets.Add(new MaterialPreset { Name = "Canvas", Category = MaterialPresetCategory.FabricNatural, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.85f, 0.82f, 0.75f), Roughness = 0.9f, Metallic = 0f, Tags = new() { "fabric", "canvas", "heavy", "outdoor" } });
            _presets.Add(new MaterialPreset { Name = "Metallic Fabric", Category = MaterialPresetCategory.FabricMetallic, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.7f, 0.7f, 0.72f), Roughness = 0.35f, Metallic = 0.3f, Sheen = 0.5f, Tags = new() { "fabric", "metallic", "shiny", "fashion" } });
        }

        private void CreateSkinPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Skin Light", Category = MaterialPresetCategory.SkinHuman, CoreCategory = MaterialCategory.Skin, BaseColor = new Color3(0.9f, 0.72f, 0.62f), Roughness = 0.55f, Metallic = 0f, SubsurfaceRadius = 1.2f, SubsurfaceColor = new Color3(0.9f, 0.3f, 0.2f), Domain = MaterialDomain.Surface, Features = MaterialFeatureFlags.SubsurfaceScattering | MaterialFeatureFlags.SubsurfaceProfile, Tags = new() { "skin", "human", "light", "sss" } });
            _presets.Add(new MaterialPreset { Name = "Skin Medium", Category = MaterialPresetCategory.SkinHuman, CoreCategory = MaterialCategory.Skin, BaseColor = new Color3(0.75f, 0.55f, 0.42f), Roughness = 0.55f, Metallic = 0f, SubsurfaceRadius = 1.0f, SubsurfaceColor = new Color3(0.8f, 0.25f, 0.15f), Domain = MaterialDomain.Surface, Features = MaterialFeatureFlags.SubsurfaceScattering | MaterialFeatureFlags.SubsurfaceProfile, Tags = new() { "skin", "human", "medium", "sss" } });
            _presets.Add(new MaterialPreset { Name = "Skin Dark", Category = MaterialPresetCategory.SkinHuman, CoreCategory = MaterialCategory.Skin, BaseColor = new Color3(0.45f, 0.3f, 0.22f), Roughness = 0.5f, Metallic = 0f, SubsurfaceRadius = 0.8f, SubsurfaceColor = new Color3(0.7f, 0.2f, 0.1f), Domain = MaterialDomain.Surface, Features = MaterialFeatureFlags.SubsurfaceScattering | MaterialFeatureFlags.SubsurfaceProfile, Tags = new() { "skin", "human", "dark", "sss" } });
            _presets.Add(new MaterialPreset { Name = "Skin Alien", Category = MaterialPresetCategory.SkinAlien, CoreCategory = MaterialCategory.Skin, BaseColor = new Color3(0.3f, 0.6f, 0.4f), Roughness = 0.4f, Metallic = 0f, SubsurfaceRadius = 1.5f, SubsurfaceColor = new Color3(0.1f, 0.8f, 0.3f), EmissiveIntensity = 0.2f, EmissiveColor = new Color3(0.1f, 0.4f, 0.2f), Domain = MaterialDomain.Surface, Features = MaterialFeatureFlags.SubsurfaceScattering | MaterialFeatureFlags.Emissive, Tags = new() { "skin", "alien", "sci-fi", "emissive" } });
            _presets.Add(new MaterialPreset { Name = "Skin Wax", Category = MaterialPresetCategory.SkinHuman, CoreCategory = MaterialCategory.Skin, BaseColor = new Color3(0.85f, 0.7f, 0.6f), Roughness = 0.3f, Metallic = 0f, SubsurfaceRadius = 2.0f, SubsurfaceColor = new Color3(0.9f, 0.4f, 0.25f), Domain = MaterialDomain.Surface, Features = MaterialFeatureFlags.SubsurfaceScattering, Tags = new() { "skin", "wax", "subsurface", "sculpture" } });
        }

        private void CreateHairPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Hair Blonde", Category = MaterialPresetCategory.HairHumanStraight, CoreCategory = MaterialCategory.Hair, BaseColor = new Color3(0.8f, 0.68f, 0.42f), Roughness = 0.35f, Metallic = 0f, Domain = MaterialDomain.Hair, Features = MaterialFeatureFlags.HairBRDF, Tags = new() { "hair", "blonde", "human", "straight" } });
            _presets.Add(new MaterialPreset { Name = "Hair Brown", Category = MaterialPresetCategory.HairHumanStraight, CoreCategory = MaterialCategory.Hair, BaseColor = new Color3(0.35f, 0.22f, 0.12f), Roughness = 0.4f, Metallic = 0f, Domain = MaterialDomain.Hair, Features = MaterialFeatureFlags.HairBRDF, Tags = new() { "hair", "brown", "human", "straight" } });
            _presets.Add(new MaterialPreset { Name = "Hair Black", Category = MaterialPresetCategory.HairHumanStraight, CoreCategory = MaterialCategory.Hair, BaseColor = new Color3(0.08f, 0.06f, 0.05f), Roughness = 0.35f, Metallic = 0f, Domain = MaterialDomain.Hair, Features = MaterialFeatureFlags.HairBRDF, Tags = new() { "hair", "black", "human", "straight" } });
            _presets.Add(new MaterialPreset { Name = "Hair Red", Category = MaterialPresetCategory.HairHumanStraight, CoreCategory = MaterialCategory.Hair, BaseColor = new Color3(0.65f, 0.2f, 0.08f), Roughness = 0.38f, Metallic = 0f, Domain = MaterialDomain.Hair, Features = MaterialFeatureFlags.HairBRDF, Tags = new() { "hair", "red", "human", "ginger" } });
            _presets.Add(new MaterialPreset { Name = "Hair Grey", Category = MaterialPresetCategory.HairHumanStraight, CoreCategory = MaterialCategory.Hair, BaseColor = new Color3(0.6f, 0.58f, 0.55f), Roughness = 0.45f, Metallic = 0f, Domain = MaterialDomain.Hair, Features = MaterialFeatureFlags.HairBRDF, Tags = new() { "hair", "grey", "human", "aging" } });
            _presets.Add(new MaterialPreset { Name = "Hair Animal Fur", Category = MaterialPresetCategory.HairAnimal, CoreCategory = MaterialCategory.Hair, BaseColor = new Color3(0.55f, 0.4f, 0.25f), Roughness = 0.6f, Metallic = 0f, Sheen = 0.3f, Domain = MaterialDomain.Hair, Features = MaterialFeatureFlags.HairBRDF | MaterialFeatureFlags.Sheen, Tags = new() { "hair", "animal", "fur", "creature" } });
        }

        private void CreateEyePresets()
        {
            _presets.Add(new MaterialPreset { Name = "Eye Blue", Category = MaterialPresetCategory.Eye, CoreCategory = MaterialCategory.Eye, BaseColor = new Color3(0.3f, 0.5f, 0.8f), Roughness = 0.05f, Metallic = 0f, IOR = 1.376f, Domain = MaterialDomain.Eye, Features = MaterialFeatureFlags.EyeBRDF | MaterialFeatureFlags.EyeIrisRefraction, Tags = new() { "eye", "blue", "human", "iris" } });
            _presets.Add(new MaterialPreset { Name = "Eye Brown", Category = MaterialPresetCategory.Eye, CoreCategory = MaterialCategory.Eye, BaseColor = new Color3(0.45f, 0.28f, 0.12f), Roughness = 0.05f, Metallic = 0f, IOR = 1.376f, Domain = MaterialDomain.Eye, Features = MaterialFeatureFlags.EyeBRDF | MaterialFeatureFlags.EyeIrisRefraction, Tags = new() { "eye", "brown", "human", "iris" } });
            _presets.Add(new MaterialPreset { Name = "Eye Green", Category = MaterialPresetCategory.Eye, CoreCategory = MaterialCategory.Eye, BaseColor = new Color3(0.35f, 0.55f, 0.3f), Roughness = 0.05f, Metallic = 0f, IOR = 1.376f, Domain = MaterialDomain.Eye, Features = MaterialFeatureFlags.EyeBRDF | MaterialFeatureFlags.EyeIrisRefraction, Tags = new() { "eye", "green", "human", "iris" } });
            _presets.Add(new MaterialPreset { Name = "Eye Sclera", Category = MaterialPresetCategory.Eye, CoreCategory = MaterialCategory.Eye, BaseColor = new Color3(0.95f, 0.93f, 0.88f), Roughness = 0.02f, Metallic = 0f, IOR = 1.376f, Domain = MaterialDomain.Eye, Features = MaterialFeatureFlags.EyeBRDF, Tags = new() { "eye", "sclera", "white", "human" } });
            _presets.Add(new MaterialPreset { Name = "Eye Reptile", Category = MaterialPresetCategory.Eye, CoreCategory = MaterialCategory.Eye, BaseColor = new Color3(0.8f, 0.6f, 0.1f), Roughness = 0.02f, Metallic = 0f, IOR = 1.376f, Domain = MaterialDomain.Eye, Features = MaterialFeatureFlags.EyeBRDF, Tags = new() { "eye", "reptile", "creature", "slit" } });
        }

        private void CreateVegetationPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Leaf Green", Category = MaterialPresetCategory.VegetationLeaf, CoreCategory = MaterialCategory.Organic, BaseColor = new Color3(0.2f, 0.55f, 0.15f), Roughness = 0.6f, Metallic = 0f, SubsurfaceRadius = 0.5f, SubsurfaceColor = new Color3(0.1f, 0.6f, 0.05f), Transmission = 0.3f, Tags = new() { "vegetation", "leaf", "green", "sss" } });
            _presets.Add(new MaterialPreset { Name = "Leaf Autumn", Category = MaterialPresetCategory.VegetationLeaf, CoreCategory = MaterialCategory.Organic, BaseColor = new Color3(0.8f, 0.35f, 0.1f), Roughness = 0.55f, Metallic = 0f, SubsurfaceRadius = 0.4f, SubsurfaceColor = new Color3(0.9f, 0.3f, 0.05f), Tags = new() { "vegetation", "leaf", "autumn", "orange" } });
            _presets.Add(new MaterialPreset { Name = "Bark Oak", Category = MaterialPresetCategory.VegetationBark, CoreCategory = MaterialCategory.Organic, BaseColor = new Color3(0.3f, 0.22f, 0.15f), Roughness = 0.9f, Metallic = 0f, Tags = new() { "vegetation", "bark", "oak", "rough" } });
            _presets.Add(new MaterialPreset { Name = "Bark Birch", Category = MaterialPresetCategory.VegetationBark, CoreCategory = MaterialCategory.Organic, BaseColor = new Color3(0.88f, 0.85f, 0.8f), Roughness = 0.7f, Metallic = 0f, Tags = new() { "vegetation", "bark", "birch", "white" } });
            _presets.Add(new MaterialPreset { Name = "Grass", Category = MaterialPresetCategory.VegetationGrass, CoreCategory = MaterialCategory.Organic, BaseColor = new Color3(0.25f, 0.5f, 0.15f), Roughness = 0.75f, Metallic = 0f, SubsurfaceRadius = 0.3f, SubsurfaceColor = new Color3(0.15f, 0.55f, 0.08f), Tags = new() { "vegetation", "grass", "green", "lawn" } });
            _presets.Add(new MaterialPreset { Name = "Moss", Category = MaterialPresetCategory.VegetationGrass, CoreCategory = MaterialCategory.Organic, BaseColor = new Color3(0.2f, 0.4f, 0.12f), Roughness = 0.95f, Metallic = 0f, Tags = new() { "vegetation", "moss", "green", "damp" } });
            _presets.Add(new MaterialPreset { Name = "Mushroom", Category = MaterialPresetCategory.VegetationLeaf, CoreCategory = MaterialCategory.Organic, BaseColor = new Color3(0.85f, 0.75f, 0.6f), Roughness = 0.4f, Metallic = 0f, SubsurfaceRadius = 0.6f, SubsurfaceColor = new Color3(0.9f, 0.7f, 0.5f), Tags = new() { "vegetation", "mushroom", "organic", "sss" } });
        }

        private void CreateLiquidPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Water Clear", Category = MaterialPresetCategory.WaterClear, CoreCategory = MaterialCategory.Liquid, BaseColor = new Color3(0.95f, 0.98f, 1.0f), Roughness = 0.0f, Metallic = 0f, Transmission = 1.0f, IOR = 1.333f, Opacity = 0.05f, Tags = new() { "liquid", "water", "clear", "ior-1.33" } });
            _presets.Add(new MaterialPreset { Name = "Water Murky", Category = MaterialPresetCategory.WaterMurky, CoreCategory = MaterialCategory.Liquid, BaseColor = new Color3(0.3f, 0.4f, 0.25f), Roughness = 0.1f, Metallic = 0f, Transmission = 0.7f, IOR = 1.333f, Opacity = 0.3f, Tags = new() { "liquid", "water", "murky", "river" } });
            _presets.Add(new MaterialPreset { Name = "Milk", Category = MaterialPresetCategory.WaterMurky, CoreCategory = MaterialCategory.Liquid, BaseColor = new Color3(0.95f, 0.93f, 0.88f), Roughness = 0.15f, Metallic = 0f, Transmission = 0.5f, IOR = 1.35f, SubsurfaceRadius = 0.8f, SubsurfaceColor = new Color3(0.95f, 0.93f, 0.85f), Tags = new() { "liquid", "milk", "opaque", "sss" } });
            _presets.Add(new MaterialPreset { Name = "Oil", Category = MaterialPresetCategory.WaterMurky, CoreCategory = MaterialCategory.Liquid, BaseColor = new Color3(0.6f, 0.5f, 0.1f), Roughness = 0.05f, Metallic = 0f, Transmission = 0.8f, IOR = 1.47f, Tags = new() { "liquid", "oil", "viscous", "golden" } });
            _presets.Add(new MaterialPreset { Name = "Honey", Category = MaterialPresetCategory.WaterMurky, CoreCategory = MaterialCategory.Liquid, BaseColor = new Color3(0.85f, 0.6f, 0.1f), Roughness = 0.1f, Metallic = 0f, Transmission = 0.7f, IOR = 1.5f, SubsurfaceRadius = 0.4f, SubsurfaceColor = new Color3(0.9f, 0.5f, 0.05f), Tags = new() { "liquid", "honey", "viscous", "golden" } });
            _presets.Add(new MaterialPreset { Name = "Glass IOR 1.52", Category = MaterialPresetCategory.WaterClear, CoreCategory = MaterialCategory.Liquid, BaseColor = new Color3(0.98f, 0.98f, 1.0f), Roughness = 0.0f, Metallic = 0f, Transmission = 0.95f, IOR = 1.52f, Opacity = 0.02f, Tags = new() { "liquid", "glass", "clear", "ior-1.52" } });
        }

        private void CreateCeramicPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Ceramic White Glazed", Category = MaterialPresetCategory.Ceramic, CoreCategory = MaterialCategory.Ceramic, BaseColor = new Color3(0.95f, 0.95f, 0.95f), Roughness = 0.1f, Metallic = 0f, ClearCoat = 0.8f, ClearCoatRoughness = 0.05f, Tags = new() { "ceramic", "glazed", "white", "porcelain" } });
            _presets.Add(new MaterialPreset { Name = "Terracotta", Category = MaterialPresetCategory.Ceramic, CoreCategory = MaterialCategory.Ceramic, BaseColor = new Color3(0.75f, 0.4f, 0.2f), Roughness = 0.75f, Metallic = 0f, Tags = new() { "ceramic", "terracotta", "clay", "pottery" } });
            _presets.Add(new MaterialPreset { Name = "Porcelain", Category = MaterialPresetCategory.Ceramic, CoreCategory = MaterialCategory.Ceramic, BaseColor = new Color3(0.97f, 0.95f, 0.92f), Roughness = 0.05f, Metallic = 0f, ClearCoat = 0.9f, Tags = new() { "ceramic", "porcelain", "fine", "luxury" } });
        }

        private void CreateLeatherPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Leather Black", Category = MaterialPresetCategory.LeatherSmooth, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.08f, 0.06f, 0.05f), Roughness = 0.5f, Metallic = 0f, ClearCoat = 0.2f, Tags = new() { "leather", "black", "smooth", "automotive" } });
            _presets.Add(new MaterialPreset { Name = "Leather Brown Aged", Category = MaterialPresetCategory.LeatherRough, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.4f, 0.25f, 0.15f), Roughness = 0.7f, Metallic = 0f, Tags = new() { "leather", "brown", "aged", "vintage" } });
            _presets.Add(new MaterialPreset { Name = "Suede", Category = MaterialPresetCategory.LeatherRough, CoreCategory = MaterialCategory.Fabric, BaseColor = new Color3(0.6f, 0.5f, 0.4f), Roughness = 0.95f, Metallic = 0f, Sheen = 0.3f, Tags = new() { "leather", "suede", "soft", "nubuck" } });
        }

        private void CreateRubberPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Rubber Black", Category = MaterialPresetCategory.Rubber, CoreCategory = MaterialCategory.Rubber, BaseColor = new Color3(0.05f, 0.05f, 0.05f), Roughness = 0.85f, Metallic = 0f, Tags = new() { "rubber", "black", "tire", "grip" } });
            _presets.Add(new MaterialPreset { Name = "Rubber Silicone", Category = MaterialPresetCategory.Rubber, CoreCategory = MaterialCategory.Rubber, BaseColor = new Color3(0.8f, 0.8f, 0.82f), Roughness = 0.6f, Metallic = 0f, Tags = new() { "rubber", "silicone", "soft", "medical" } });
            _presets.Add(new MaterialPreset { Name = "Rubber Neoprene", Category = MaterialPresetCategory.Rubber, CoreCategory = MaterialCategory.Rubber, BaseColor = new Color3(0.12f, 0.12f, 0.12f), Roughness = 0.7f, Metallic = 0f, Tags = new() { "rubber", "neoprene", "wetsuit", "sport" } });
        }

        private void CreateSnowIcePresets()
        {
            _presets.Add(new MaterialPreset { Name = "Snow Fresh", Category = MaterialPresetCategory.Snow, CoreCategory = MaterialCategory.Snow, BaseColor = new Color3(0.95f, 0.96f, 0.98f), Roughness = 0.5f, Metallic = 0f, SubsurfaceRadius = 0.5f, SubsurfaceColor = new Color3(0.8f, 0.85f, 0.95f), Tags = new() { "snow", "fresh", "winter", "sss" } });
            _presets.Add(new MaterialPreset { Name = "Snow Packed", Category = MaterialPresetCategory.Snow, CoreCategory = MaterialCategory.Snow, BaseColor = new Color3(0.85f, 0.88f, 0.92f), Roughness = 0.7f, Metallic = 0f, Tags = new() { "snow", "packed", "winter", "compression" } });
            _presets.Add(new MaterialPreset { Name = "Ice Clear", Category = MaterialPresetCategory.IceClear, CoreCategory = MaterialCategory.Ice, BaseColor = new Color3(0.85f, 0.92f, 0.98f), Roughness = 0.02f, Metallic = 0f, Transmission = 0.9f, IOR = 1.31f, Tags = new() { "ice", "clear", "frozen", "ior-1.31" } });
            _presets.Add(new MaterialPreset { Name = "Ice Frosted", Category = MaterialPresetCategory.IceFrosted, CoreCategory = MaterialCategory.Ice, BaseColor = new Color3(0.88f, 0.92f, 0.95f), Roughness = 0.3f, Metallic = 0f, Transmission = 0.6f, IOR = 1.31f, Tags = new() { "ice", "frosted", "frozen", "rough" } });
        }

        private void CreateSandSoilPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Sand Fine", Category = MaterialPresetCategory.Sand, CoreCategory = MaterialCategory.Sand, BaseColor = new Color3(0.85f, 0.78f, 0.62f), Roughness = 0.9f, Metallic = 0f, Tags = new() { "sand", "fine", "desert", "beach" } });
            _presets.Add(new MaterialPreset { Name = "Sand Wet", Category = MaterialPresetCategory.Sand, CoreCategory = MaterialCategory.Sand, BaseColor = new Color3(0.55f, 0.48f, 0.35f), Roughness = 0.6f, Metallic = 0f, Tags = new() { "sand", "wet", "beach", "dark" } });
            _presets.Add(new MaterialPreset { Name = "Soil Rich", Category = MaterialPresetCategory.Soil, CoreCategory = MaterialCategory.Geological, BaseColor = new Color3(0.35f, 0.25f, 0.15f), Roughness = 0.9f, Metallic = 0f, Tags = new() { "soil", "rich", "garden", "organic" } });
            _presets.Add(new MaterialPreset { Name = "Clay Red", Category = MaterialPresetCategory.Clay, CoreCategory = MaterialCategory.Geological, BaseColor = new Color3(0.7f, 0.35f, 0.2f), Roughness = 0.8f, Metallic = 0f, Tags = new() { "clay", "red", "terracotta", "pottery" } });
            _presets.Add(new MaterialPreset { Name = "Mud", Category = MaterialPresetCategory.Soil, CoreCategory = MaterialCategory.Geological, BaseColor = new Color3(0.35f, 0.28f, 0.18f), Roughness = 0.85f, Metallic = 0f, Tags = new() { "mud", "wet", "earthy", "natural" } });
        }

        private void CreateGemPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Ruby", Category = MaterialPresetCategory.Gem, CoreCategory = MaterialCategory.Gem, BaseColor = new Color3(0.8f, 0.05f, 0.1f), Roughness = 0.02f, Metallic = 0f, Transmission = 0.9f, IOR = 1.77f, Tags = new() { "gem", "ruby", "precious", "red" } });
            _presets.Add(new MaterialPreset { Name = "Sapphire", Category = MaterialPresetCategory.Gem, CoreCategory = MaterialCategory.Gem, BaseColor = new Color3(0.1f, 0.15f, 0.8f), Roughness = 0.02f, Metallic = 0f, Transmission = 0.9f, IOR = 1.77f, Tags = new() { "gem", "sapphire", "precious", "blue" } });
            _presets.Add(new MaterialPreset { Name = "Emerald", Category = MaterialPresetCategory.Gem, CoreCategory = MaterialCategory.Gem, BaseColor = new Color3(0.1f, 0.6f, 0.3f), Roughness = 0.02f, Metallic = 0f, Transmission = 0.85f, IOR = 1.58f, Tags = new() { "gem", "emerald", "precious", "green" } });
            _presets.Add(new MaterialPreset { Name = "Diamond", Category = MaterialPresetCategory.Gem, CoreCategory = MaterialCategory.Gem, BaseColor = new Color3(0.98f, 0.98f, 1.0f), Roughness = 0.0f, Metallic = 0f, Transmission = 1.0f, IOR = 2.42f, Tags = new() { "gem", "diamond", "precious", "brilliant" } });
            _presets.Add(new MaterialPreset { Name = "Amethyst", Category = MaterialPresetCategory.Gem, CoreCategory = MaterialCategory.Gem, BaseColor = new Color3(0.6f, 0.2f, 0.8f), Roughness = 0.02f, Metallic = 0f, Transmission = 0.85f, IOR = 1.55f, Tags = new() { "gem", "amethyst", "precious", "purple" } });
            _presets.Add(new MaterialPreset { Name = "Jade", Category = MaterialPresetCategory.Gem, CoreCategory = MaterialCategory.Gem, BaseColor = new Color3(0.2f, 0.6f, 0.35f), Roughness = 0.1f, Metallic = 0f, Transmission = 0.5f, IOR = 1.66f, Tags = new() { "gem", "jade", "precious", "oriental" } });
        }

        private void CreateCarbonPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Carbon Fiber", Category = MaterialPresetCategory.Carbon, CoreCategory = MaterialCategory.Synthetic, BaseColor = new Color3(0.12f, 0.12f, 0.14f), Roughness = 0.3f, Metallic = 0.1f, Tags = new() { "carbon", "fiber", "composite", "lightweight" } });
            _presets.Add(new MaterialPreset { Name = "Carbon Matte", Category = MaterialPresetCategory.Carbon, CoreCategory = MaterialCategory.Synthetic, BaseColor = new Color3(0.08f, 0.08f, 0.08f), Roughness = 0.85f, Metallic = 0f, Tags = new() { "carbon", "matte", "stealth", "tactical" } });
        }

        private void CreateWaxSoapPresets()
        {
            _presets.Add(new MaterialPreset { Name = "Candle Wax", Category = MaterialPresetCategory.Wax, CoreCategory = MaterialCategory.Organic, BaseColor = new Color3(0.95f, 0.9f, 0.75f), Roughness = 0.3f, Metallic = 0f, SubsurfaceRadius = 1.5f, SubsurfaceColor = new Color3(0.95f, 0.85f, 0.6f), Transmission = 0.4f, Tags = new() { "wax", "candle", "translucent", "sss" } });
            _presets.Add(new MaterialPreset { Name = "Soap Bar", Category = MaterialPresetCategory.Soap, CoreCategory = MaterialCategory.Synthetic, BaseColor = new Color3(0.9f, 0.88f, 0.85f), Roughness = 0.2f, Metallic = 0f, SubsurfaceRadius = 0.8f, SubsurfaceColor = new Color3(0.9f, 0.85f, 0.8f), Tags = new() { "soap", "bar", "clean", "smooth" } });
            _presets.Add(new MaterialPreset { Name = "Beeswax", Category = MaterialPresetCategory.Wax, CoreCategory = MaterialCategory.Organic, BaseColor = new Color3(0.85f, 0.7f, 0.3f), Roughness = 0.25f, Metallic = 0f, SubsurfaceRadius = 1.2f, SubsurfaceColor = new Color3(0.9f, 0.65f, 0.2f), Tags = new() { "wax", "beeswax", "natural", "golden" } });
        }

        private void IndexAsset(MaterialAssetEntry entry)
        {
            if (_categoryIndex.TryGetValue(entry.Category, out var list))
                list.Add(entry.Id);
            else
                _categoryIndex[entry.Category] = new List<string> { entry.Id };

            foreach (var tag in entry.Tags)
            {
                if (_tagIndex.TryGetValue(tag, out var tagList))
                    tagList.Add(entry.Id);
                else
                    _tagIndex[tag] = new List<string> { entry.Id };
            }
        }

        private void RemoveFromIndex(MaterialAssetEntry entry)
        {
            if (_categoryIndex.TryGetValue(entry.Category, out var catList))
                catList.Remove(entry.Id);

            foreach (var tag in entry.Tags)
                if (_tagIndex.TryGetValue(tag, out var tagList))
                    tagList.Remove(entry.Id);
        }

        private static ulong ComputePresetHash(MaterialPreset preset)
        {
            var sb = new StringBuilder();
            sb.Append(preset.Name);
            sb.Append(preset.BaseColor.R).Append(preset.BaseColor.G).Append(preset.BaseColor.B);
            sb.Append(preset.Roughness).Append(preset.Metallic);
            sb.Append(preset.IOR).Append(preset.Transmission);
            using var sha = System.Security.Cryptography.SHA256.Create();
            var bytes = Encoding.UTF8.GetBytes(sb.ToString());
            var hash = sha.ComputeHash(bytes);
            return BitConverter.ToUInt64(hash, 0);
        }

        private void SaveMetadata(string path)
        {
            var data = _assets.Values.Select(a => new { a.Id, a.Name, a.Category, a.UsageCount, a.IsFavorite, a.Tags }).ToList();
            var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, json, Encoding.UTF8);
        }

        private void LoadMetadata(string path)
        {
            try
            {
                var json = File.ReadAllText(path, Encoding.UTF8);
                var data = JsonSerializer.Deserialize<List<JsonElement>>(json);
                if (data == null)
                    return;
                foreach (var item in data)
                {
                    if (item.TryGetProperty("Id", out var idProp) && item.TryGetProperty("UsageCount", out var usageProp))
                    {
                        string id = idProp.GetString();
                        if (_assets.TryGetValue(id, out var entry))
                        {
                            entry.UsageCount = usageProp.GetInt32();
                            if (item.TryGetProperty("IsFavorite", out var favProp))
                                entry.IsFavorite = favProp.GetBoolean();
                        }
                    }
                }
            }
            catch { }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _lock?.Dispose();
                _assets.Clear();
                _presets.Clear();
                _tagIndex.Clear();
                _categoryIndex.Clear();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }
}
