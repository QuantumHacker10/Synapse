using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using GDNN.Rendering.ArtPipeline;
using GDNN.Rendering.Bridge;
using Synapse.Infrastructure;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Plugins;
using Synapse.Runtime;
using Synapse.Simulation.DigitalTwins;
using Synapse.Studio.Contracts;
using EntityType = Synapse.Studio.Contracts.EntityType;
using SceneEntity = Synapse.Studio.Contracts.SceneEntity;

namespace Synapse.Studio.ViewModels
{
    /// <summary>
    /// Avalonia view model for Synapse Studio: scene editing, living laws, LLM console,
    /// evolution, blueprint/sculpt tools, and status HUD bound to <see cref="EngineHost"/>.
    /// </summary>
    public partial class MainWindowViewModel : ObservableObject, IDisposable
    {
        private readonly EngineHost _host;
        private readonly FrameOrchestrator _orchestrator;
        private readonly ISynapseLogger _logger;
        private readonly SynapseConfig _config;
        private readonly PluginHost? _pluginHost;
        private readonly DispatcherTimer _uiTimer;
        private readonly MegascansBridge _megascans = new();
        private readonly SculptSession _sculpt = new();
        private BlueprintDocument _blueprint = BlueprintDocument.CreateDefault();
        private string? _projectPath;
        private bool _disposed;

        public MainWindowViewModel(EngineHost host, FrameOrchestrator orchestrator, ISynapseLogger logger, SynapseConfig config, PluginHost? pluginHost = null)
        {
            _host = host;
            _orchestrator = orchestrator;
            _logger = logger;
            _config = config;
            _pluginHost = pluginHost;

            RefreshEntities();
            RefreshLaws();
            RefreshBlueprint();
            _host.ViewportEditor.ShowGrid = ShowViewportGrid;
            _host.ViewportEditor.ShowGizmos = ShowViewportGizmos;
            _host.ViewportEditor.ToolMode = ViewportToolMode.Translate;
            _host.ViewportEntitySelected += OnViewportEntitySelected;
            _host.InspectorFeedEntryAdded += OnInspectorFeedEntry;
            _host.CollaborationPatchApplied += OnCollaborationPatchApplied;

            WanSessionCode = config.WanSessionCode ?? "synapse-room";
            WanPort = config.WanPort;
            WanRendezvousPort = config.WanRendezvousPort;
            VrStatusText = _host.VrStatusText;
            WanStatusText = _host.WanStatusText;
            WebStatusText = _host.WebStatusText;

            _uiTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(200) };
            _uiTimer.Tick += (_, _) => RefreshStatus();
            _uiTimer.Start();

            ViewportHint = "Viewport Vulkan — Grille/Gizmos · Ctrl+clic = relier les nœuds blueprint · Espace = pause";
            SculptStatus = "Pinceau prêt";
            MegascansStatus = $"Bibliothèque : {_megascans.Config.LibraryRootPath}";
            LlmStatusText = _host.LlmProviderSummary;

            TryLoadMarketplaceCatalog();
            RefreshMarketplacePackages();
            RefreshTwins();
            RefreshPlugins();
            MarketplaceStatus = _host.MarketplaceStatusText;
            TwinStatusText = _host.TwinStatusText;
            AboutText =
                $"{SynapseProduct.Name} {SynapseProduct.Version} — Outil de simulation 3D\n\n" +
                "Pas un moteur de jeu : un monde numérique que l'on observe, modifie et fait évoluer.\n" +
                "Formes apprises (G-DNN), lois physiques réécrivables, évolution NEAT-G, " +
                "agents sentients (perception et décisions — pas des PNJ scriptés), " +
                "assistance créative (LLM) et rendu Vulkan temps réel.\n\n" +
                "Site : https://quantumhacker10.github.io/Synapse/\n" +
                "Licence MIT — open source.\n\n" +
                "Apprendre · Réécrire · Cultiver";

            foreach (EntityType t in Enum.GetValues<EntityType>())
            {
                if (t != EntityType.Unknown)
                    EntityTypes.Add(t);
            }
            NewEntityType = EntityType.Empty;
        }

        public ObservableCollection<SceneEntity> Entities { get; } = new();
        public ObservableCollection<LawCatalogEntry> FilteredLawCatalog { get; } = new();
        public ObservableCollection<EntityType> EntityTypes { get; } = new();
        public ObservableCollection<ChatMessageRecord> ChatMessages { get; } = new();
        public ObservableCollection<string> BlueprintNodes { get; } = new();
        public ObservableCollection<LiveInspectorEntry> InspectorFeed { get; } = new();
        public ObservableCollection<TwinListEntry> Twins { get; } = new();
        public ObservableCollection<PluginListEntry> Plugins { get; } = new();
        public ObservableCollection<LawCatalogEntry> MarketplacePackages { get; } = new();

        [ObservableProperty] private SceneEntity? selectedEntity;
        [ObservableProperty] private EntityType newEntityType = EntityType.Empty;
        [ObservableProperty] private string entityName = "";
        [ObservableProperty] private double entityPosX;
        [ObservableProperty] private double entityPosY;
        [ObservableProperty] private double entityPosZ;
        [ObservableProperty] private string inspectorMeta = "";
        [ObservableProperty] private bool hasSelectedEntity;
        [ObservableProperty] private string llmStatusText = "";
        [ObservableProperty] private bool showWelcomeTips = true;
        [ObservableProperty] private bool showAboutText;
        [ObservableProperty] private string aboutText = "";
        [ObservableProperty] private string? selectedLawId;
        [ObservableProperty] private LawCatalogEntry? selectedLawEntry;
        [ObservableProperty] private string lawSearchText = "";
        [ObservableProperty] private string lawCatalogSummary = "";
        [ObservableProperty] private string selectedLawSummary = "";
        [ObservableProperty] private string lawExpression = "∂T/∂t = α*∇²T";
        [ObservableProperty] private string lawStatus = "Prêt";
        [ObservableProperty] private string chatInput = "";
        [ObservableProperty] private string statusFps = "IPS : —";
        [ObservableProperty] private string statusPhysics = "Physique : —";
        [ObservableProperty] private string statusSimulation = "Sim : —";
        [ObservableProperty] private string statusQuality = "Qualité : —";
        [ObservableProperty] private string statusLaw = "Loi : —";
        [ObservableProperty] private string performanceReport = "En attente des premières images…";
        [ObservableProperty] private string simulationStatus = "En cours";
        [ObservableProperty] private string evolutionStatus = "Inactif";
        [ObservableProperty] private string viewportHint = "";
        [ObservableProperty] private string blueprintStatus = "Prêt";
        [ObservableProperty] private string sculptStatus = "";
        [ObservableProperty] private double sculptRadius = 0.5;
        [ObservableProperty] private double sculptStrength = 0.15;
        [ObservableProperty] private bool sculptInvert;
        [ObservableProperty] private string megascansPath = "";
        [ObservableProperty] private string megascansStatus = "";
        [ObservableProperty] private string llmApplyStatus = "Demandez un JSON d'éclairage ou un hint SDF, puis Appliquer.";
        [ObservableProperty] private bool showViewportGrid = true;
        [ObservableProperty] private bool showViewportGizmos = true;
        [ObservableProperty] private string giStatus = "GI : lecture GPU en attente";
        [ObservableProperty] private ViewportToolMode viewportTool = ViewportToolMode.Translate;
        [ObservableProperty] private string entityMeshPath = "";
        [ObservableProperty] private bool entityIsVehicle;
        [ObservableProperty] private bool entityBakeNeuralSdf;
        [ObservableProperty] private string physicsToolsStatus = "Physique Omnia : mesh · joints · véhicule";
        [ObservableProperty] private bool isInspectorModeEnabled;
        [ObservableProperty] private string inspectorFeedStatus = "Mode inspecteur inactif";
        [ObservableProperty] private string wanSessionCode = "synapse-room";
        [ObservableProperty] private int wanPort = 7777;
        [ObservableProperty] private int wanRendezvousPort;
        [ObservableProperty] private string vrStatusText = "VR : off";
        [ObservableProperty] private string wanStatusText = "WAN : off";
        [ObservableProperty] private string webStatusText = "Web : prêt";
        [ObservableProperty] private string collaborationStatus = "Collaboration inactive";
        [ObservableProperty] private string marketplaceStatus = "Marketplace : prêt";
        [ObservableProperty] private string twinStatusText = "Jumeaux : —";
        [ObservableProperty] private string pluginStatusText = "Plugins : aucun";
        [ObservableProperty] private string behaviorTreeText = "";
        [ObservableProperty] private bool hasAgentSelection;
        [ObservableProperty] private TwinListEntry? selectedTwin;
        private Guid? _jointPartnerId;
        private const int MaxInspectorFeedEntries = 500;

        public BlueprintDocument Blueprint => _blueprint;

        partial void OnSelectedEntityChanged(SceneEntity? value)
        {
            HasSelectedEntity = value != null;
            if (value != null)
                _host.SetViewportSelection(value.Id);
            else
                _host.SetViewportSelection(Guid.Empty);
            _host.SyncSceneToRenderer();
            if (value == null)
            {
                InspectorMeta = "";
                EntityName = "";
                EntityMeshPath = "";
                EntityIsVehicle = false;
                EntityBakeNeuralSdf = false;
                BehaviorTreeText = "";
                HasAgentSelection = false;
                return;
            }

            if (value.Type == EntityType.Agent)
            {
                BehaviorTreeText = _host.GetAgentBehaviorTreeText(value.Id) ?? "(aucun arbre de comportement)";
                HasAgentSelection = true;
            }
            else
            {
                BehaviorTreeText = "";
                HasAgentSelection = false;
            }

            EntityName = value.Name;
            EntityPosX = value.Position.X;
            EntityPosY = value.Position.Y;
            EntityPosZ = value.Position.Z;
            EntityMeshPath = value.MeshPath ?? "";
            EntityIsVehicle = value.IsVehicle;
            EntityBakeNeuralSdf = value.BakeNeuralSdf;
            InspectorMeta =
                $"Type : {value.Type}\nComportement : {value.BehaviorProfile ?? "—"}\n" +
                $"Génome : {value.GenomeId ?? "—"}\nLoi : {value.LawId ?? "—"}\n" +
                $"Mesh : {(string.IsNullOrWhiteSpace(value.MeshPath) ? "—" : value.MeshPath)}\n" +
                $"Véhicule : {(value.IsVehicle ? "oui" : "non")}";
        }

        partial void OnShowViewportGridChanged(bool value)
        {
            _host.ViewportEditor.ShowGrid = value;
            _host.SyncSceneToRenderer();
        }

        partial void OnShowViewportGizmosChanged(bool value)
        {
            _host.ViewportEditor.ShowGizmos = value;
            _host.SyncSceneToRenderer();
        }

        partial void OnViewportToolChanged(ViewportToolMode value)
        {
            _host.ViewportEditor.ToolMode = value;
        }

        public void NotifyBlueprintChanged() => RefreshBlueprint();

        public void SelectEntityById(Guid id)
        {
            SelectedEntity = Entities.FirstOrDefault(e => e.Id == id);
        }

        private void OnViewportEntitySelected(Guid id)
        {
            Avalonia.Threading.Dispatcher.UIThread.Post(() => SelectEntityById(id));
        }

        private void OnInspectorFeedEntry(InspectorFeedEntry entry)
        {
            if (!IsInspectorModeEnabled)
                return;

            Dispatcher.UIThread.Post(() =>
            {
                InspectorFeed.Insert(0, new LiveInspectorEntry
                {
                    Timestamp = entry.Timestamp,
                    Category = entry.Category,
                    Title = entry.Title,
                    Detail = entry.Detail
                });
                while (InspectorFeed.Count > MaxInspectorFeedEntries)
                    InspectorFeed.RemoveAt(InspectorFeed.Count - 1);
                InspectorFeedStatus = $"{InspectorFeed.Count} entrées · dernier : [{entry.Category}] {entry.Title}";
            });
        }

        partial void OnIsInspectorModeEnabledChanged(bool value)
        {
            InspectorFeedStatus = value
                ? "Mode inspecteur actif — NEAT-G et lois en direct"
                : "Mode inspecteur inactif";
            if (value && InspectorFeed.Count == 0)
                InspectorFeedStatus = "Mode inspecteur actif — en attente d'événements…";
        }

        [RelayCommand]
        private void ToggleInspectorMode() => IsInspectorModeEnabled = !IsInspectorModeEnabled;

        [RelayCommand]
        private void ClearInspectorFeed()
        {
            InspectorFeed.Clear();
            InspectorFeedStatus = IsInspectorModeEnabled
                ? "Flux vidé — en attente d'événements…"
                : "Mode inspecteur inactif";
        }

        [RelayCommand]
        private void SetViewportToolTranslate() => ViewportTool = ViewportToolMode.Translate;

        [RelayCommand]
        private void SetViewportToolRotate() => ViewportTool = ViewportToolMode.Rotate;

        [RelayCommand]
        private void SetViewportToolSelect() => ViewportTool = ViewportToolMode.Select;

        [RelayCommand]
        private void SetViewportToolScale() => ViewportTool = ViewportToolMode.Scale;

        partial void OnSelectedLawIdChanged(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return;
            var law = _host.ListLawCatalog().FirstOrDefault(l => l.Id == value);
            if (string.IsNullOrWhiteSpace(law.Id))
                return;

            LawExpression = law.Expression;
            var entry = FindCatalogEntry(value);
            if (entry != null && !ReferenceEquals(SelectedLawEntry, entry))
                SelectedLawEntry = entry;
            UpdateSelectedLawSummary();
        }

        partial void OnSelectedLawEntryChanged(LawCatalogEntry? value)
        {
            if (value == null)
                return;
            if (!string.Equals(SelectedLawId, value.Id, StringComparison.OrdinalIgnoreCase))
                SelectedLawId = value.Id;
            LawExpression = value.Expression;
            UpdateSelectedLawSummary();
        }

        partial void OnLawSearchTextChanged(string value) => ApplyLawCatalogFilter();

        private readonly List<LawCatalogEntry> _allLawCatalog = new();

        private void UpdateSelectedLawSummary()
        {
            if (SelectedLawEntry == null)
            {
                SelectedLawSummary = "";
                return;
            }

            SelectedLawSummary =
                $"{SelectedLawEntry.Name}\n" +
                $"Catégorie : {SelectedLawEntry.CategoryLabel}\n" +
                $"{SelectedLawEntry.Description}";
        }

        private LawCatalogEntry? FindCatalogEntry(string id) =>
            _allLawCatalog.FirstOrDefault(l => string.Equals(l.Id, id, StringComparison.OrdinalIgnoreCase));

        [RelayCommand]
        private void RefreshEntities()
        {
            Entities.Clear();
            foreach (var e in _host.Scene.Entities)
            {
                Entities.Add(new SceneEntity
                {
                    Id = e.Id,
                    Name = e.Name,
                    Type = ParseSceneEntityType(e.Type),
                    Position = e.Position.ToVector3(),
                    Scale = e.Scale.ToVector3(),
                    IsVisible = e.Visible,
                    BehaviorProfile = e.BehaviorProfile,
                    GenomeId = e.GenomeId,
                    LawId = e.LawId,
                    MeshPath = e.MeshPath,
                    IsVehicle = e.IsVehicle,
                    BakeNeuralSdf = e.BakeNeuralSdf
                });
            }
        }

        private static EntityType ParseSceneEntityType(string type)
        {
            if (SceneEntityKinds.IsSentientAgent(type))
                return EntityType.Agent;
            return Enum.TryParse<EntityType>(type, true, out var t) ? t : EntityType.Empty;
        }

        private void RefreshLaws()
        {
            _allLawCatalog.Clear();
            foreach (var law in _host.ListLawCatalog())
            {
                _allLawCatalog.Add(new LawCatalogEntry
                {
                    Id = law.Id,
                    Name = law.Name,
                    Category = law.Category,
                    Description = law.Description,
                    Expression = law.Expression
                });
            }

            LawCatalogSummary = $"{_allLawCatalog.Count} lois dans le catalogue · actif : {_host.ActiveLawId ?? "—"}";
            ApplyLawCatalogFilter();
            SelectedLawId = _host.ActiveLawId ?? _allLawCatalog.FirstOrDefault()?.Id;
            SelectedLawEntry = FindCatalogEntry(SelectedLawId ?? "") ?? _allLawCatalog.FirstOrDefault();
            UpdateSelectedLawSummary();
        }

        private void ApplyLawCatalogFilter()
        {
            FilteredLawCatalog.Clear();
            string q = LawSearchText.Trim();
            IEnumerable<LawCatalogEntry> query = _allLawCatalog;
            if (!string.IsNullOrWhiteSpace(q))
            {
                query = _allLawCatalog.Where(l =>
                    l.Id.Contains(q, StringComparison.OrdinalIgnoreCase) ||
                    l.Name.Contains(q, StringComparison.OrdinalIgnoreCase) ||
                    l.Category.Contains(q, StringComparison.OrdinalIgnoreCase) ||
                    l.Description.Contains(q, StringComparison.OrdinalIgnoreCase));
            }

            foreach (var entry in query)
                FilteredLawCatalog.Add(entry);
        }

        private void RefreshStatus()
        {
            var s = _orchestrator.LastStats;
            StatusFps = $"IPS : {s.Fps:F0}";
            StatusPhysics = $"Physique : {s.PhysicsMs:F1} ms  T≈{s.FieldTemperatureAvg:F1} K";
            StatusSimulation = $"Sim : {s.SimulationMs:F1} ms  entités={s.EntityCount}";
            StatusQuality = $"Qualité : {s.QualityPreset}";
            StatusLaw = $"Loi : {s.ActiveLawId}";
            LlmStatusText = _host.LlmProviderSummary;
            PerformanceReport =
                $"Image {s.RenderMs:F1} ms | Physique {s.PhysicsMs:F1} | Sim {s.SimulationMs:F1}\n" +
                $"Temps total {s.TotalTime:F1} s | Évolution gen={_host.EvolutionGeneration} fitness={_host.BestFitness:F3}";
            SimulationStatus = _host.SimulationPlaying
                ? $"En cours — {s.EntityCount} agents"
                : "En pause";
            if (_host.EvolutionRunning)
                EvolutionStatus = $"En cours gen {_host.EvolutionGeneration} best={_host.BestFitness:F3}";
            else if (!EvolutionStatus.StartsWith("Terminé", StringComparison.Ordinal))
                EvolutionStatus = "Inactif";

            var fillMode = _host.RenderEngine?.SceneRenderer?.LastGiFillMode ?? GiGBufferFillMode.None;
            GiStatus = fillMode switch
            {
                GiGBufferFillMode.GpuReadback => "GI : lecture G-buffer GPU active",
                GiGBufferFillMode.GpuResident => "GI : G-buffer GPU résident",
                GiGBufferFillMode.ProceduralPreview => "GI : preview procédurale L-DNN",
                GiGBufferFillMode.Constants => "GI : constantes de repli",
                _ => "GI : lecture GPU en attente"
            };

            VrStatusText = _host.VrStatusText;
            WanStatusText = _host.WanStatusText;
            WebStatusText = _host.WebStatusText;
            CollaborationStatus = _host.IsWanConnected
                ? $"WAN patches ↑{_host.WanPatchesSent} ↓{_host.WanPatchesReceived} | VR {(_host.IsVrActive ? "on" : "off")}"
                : (_host.IsVrActive ? $"VR actif ({s.VrMs:F1} ms)" : "Collaboration inactive");

            MarketplaceStatus = _host.MarketplaceStatusText;
            TwinStatusText = _host.TwinStatusText;
        }

        private void OnCollaborationPatchApplied(string peerId)
        {
            Dispatcher.UIThread.Post(() =>
            {
                RefreshEntities();
                CollaborationStatus = $"Patch reçu de {peerId[..Math.Min(8, peerId.Length)]}…";
            });
        }

        [RelayCommand]
        private void AddEntity()
        {
            var typeName = NewEntityType.ToString();
            var id = _host.CreateSceneEntity($"{typeName}_{Entities.Count + 1}", typeName);
            RefreshEntities();
            SelectedEntity = Entities.FirstOrDefault(e => e.Id == id);
            _logger.Info("Studio", $"Created entity {id} ({typeName})");
        }

        [RelayCommand]
        private void ApplyInspector()
        {
            if (SelectedEntity == null)
                return;

            var pos = new Vector3((float)EntityPosX, (float)EntityPosY, (float)EntityPosZ);
            var id = SelectedEntity.Id;
            _host.UpdateSceneEntity(
                id,
                EntityName,
                pos,
                SelectedEntity.Scale,
                meshPath: EntityMeshPath,
                isVehicle: EntityIsVehicle,
                bakeNeuralSdf: EntityBakeNeuralSdf,
                resyncPhysics: true);
            RefreshEntities();
            SelectedEntity = Entities.FirstOrDefault(e => e.Id == id);
            PhysicsToolsStatus = "Propriétés appliquées · physique resynchronisée";
        }

        [RelayCommand]
        private async Task BrowseMeshAsync()
        {
            if (SelectedEntity == null)
                return;

            var lifetime = Avalonia.Application.Current?.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime;
            var window = lifetime?.MainWindow;
            if (window?.StorageProvider == null)
                return;

            var files = await window.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title = "Importer un mesh (glTF / OBJ)",
                AllowMultiple = false,
                FileTypeFilter = new[]
                {
                    new FilePickerFileType("Mesh")
                    {
                        Patterns = new[] { "*.gltf", "*.glb", "*.obj" }
                    }
                }
            });
            var path = files.Count > 0 ? files[0].TryGetLocalPath() : null;
            if (path == null)
                return;

            EntityMeshPath = path;
            var id = SelectedEntity.Id;
            _host.SetEntityMeshPath(id, path, EntityBakeNeuralSdf);
            RefreshEntities();
            SelectedEntity = Entities.FirstOrDefault(e => e.Id == id);
            PhysicsToolsStatus = $"Mesh lié : {Path.GetFileName(path)}";
        }

        [RelayCommand]
        private void MarkAsVehicle()
        {
            if (SelectedEntity == null)
                return;
            EntityIsVehicle = true;
            var id = SelectedEntity.Id;
            _host.SetEntityAsVehicle(id, true);
            RefreshEntities();
            SelectedEntity = Entities.FirstOrDefault(e => e.Id == id);
            PhysicsToolsStatus = "Véhicule raycast activé sur la sélection";
        }

        [RelayCommand]
        private void AddHingeToWorld()
        {
            if (SelectedEntity == null)
                return;
            var jointId = _host.AddHingeToWorld(SelectedEntity.Id);
            PhysicsToolsStatus = jointId.HasValue
                ? $"Charnière monde ajoutée ({jointId.Value:N})"
                : "Impossible d'ajouter la charnière";
        }

        [RelayCommand]
        private void RememberJointPartner()
        {
            if (SelectedEntity == null)
                return;
            _jointPartnerId = SelectedEntity.Id;
            PhysicsToolsStatus = $"Partenaire joint mémorisé : {SelectedEntity.Name}";
        }

        [RelayCommand]
        private void AddDistanceToPartner()
        {
            if (SelectedEntity == null || !_jointPartnerId.HasValue)
            {
                PhysicsToolsStatus = "Mémorisez d'abord un partenaire (bouton Partner), puis liez.";
                return;
            }
            if (_jointPartnerId.Value == SelectedEntity.Id)
            {
                PhysicsToolsStatus = "Choisissez une autre entité pour le joint distance.";
                return;
            }

            var jointId = _host.AddDistanceJoint(_jointPartnerId.Value, SelectedEntity.Id, compliance: 0.02f);
            PhysicsToolsStatus = jointId.HasValue
                ? $"Joint distance soft ajouté ({jointId.Value:N})"
                : "Échec du joint distance";
        }

        [RelayCommand]
        private void ResyncPhysics()
        {
            _host.SyncSceneToPhysics();
            PhysicsToolsStatus =
                $"Physique resync — {_host.Scene.Joints.Count} joints, " +
                $"{_host.Scene.Entities.Count(e => e.IsVehicle)} véhicules";
        }

        [RelayCommand]
        private void DismissWelcome() => ShowWelcomeTips = false;

        [RelayCommand]
        private void DeleteEntity()
        {
            if (SelectedEntity == null)
                return;
            _host.DeleteSceneEntity(SelectedEntity.Id);
            RefreshEntities();
        }

        [RelayCommand]
        private void CompileLaw()
        {
            var id = SelectedLawId ?? "custom_law";
            var result = _host.CompileLaw(id, LawExpression);
            LawStatus = result.Success
                ? $"OK — {result.InstructionCount} ops, {result.CompilationTimeMs} ms"
                : $"Échec : {result.Message}";
            LawCatalogSummary = $"{_allLawCatalog.Count} lois · actif : {_host.ActiveLawId ?? id}";
        }

        [RelayCommand]
        private void ApplySelectedLaw()
        {
            if (SelectedLawEntry == null)
                return;
            var result = _host.ApplyLaw(SelectedLawEntry.Id);
            LawExpression = SelectedLawEntry.Expression;
            LawStatus = result.Success
                ? $"Loi active : {SelectedLawEntry.Id} ({result.InstructionCount} ops)"
                : $"Échec : {result.Message}";
            LawCatalogSummary = $"{_allLawCatalog.Count} lois · actif : {_host.ActiveLawId ?? SelectedLawEntry.Id}";
            RefreshStatus();
        }

        [RelayCommand]
        private async Task SendChatAsync()
        {
            if (string.IsNullOrWhiteSpace(ChatInput))
                return;
            var prompt = ChatInput.Trim();
            ChatInput = "";
            ChatMessages.Add(new ChatMessageRecord { Role = "Vous", Content = prompt });

            // Steer the model toward structured lighting/SDF JSON when relevant.
            string routedPrompt = LooksLikeSceneControlPrompt(prompt)
                ? prompt + "\n\nRespond with a JSON object including any of: " +
                  "directionalDirection [x,y,z], color (#RRGGBB), intensity, fogDensity, " +
                  "enableClouds, and/or primitive/center/radius for an SDF hint."
                : prompt;

            var response = await _host.ChatAsync(routedPrompt);
            string content = response == null
                ? "(aucune réponse)"
                : response.IsError
                    ? $"(erreur LLM) {response.ErrorMessage ?? "provider indisponible"}"
                    : response.Content;
            ChatMessages.Add(new ChatMessageRecord
            {
                Role = "Assistant",
                Content = content,
                Provider = response?.Provider ?? ""
            });

            // Auto-apply scene or behavior hints when parseable.
            LlmApplyStatus = _host.ApplyLlmSceneHints(content);
            if (!LlmApplyStatus.StartsWith("Applied:", StringComparison.Ordinal))
                LlmApplyStatus = _host.ApplyLlmBehaviorHints(content);
            if (LlmApplyStatus.Contains("Registered", StringComparison.Ordinal) ||
                LlmApplyStatus.StartsWith("Applied:", StringComparison.Ordinal))
                RefreshEntities();
        }

        [RelayCommand]
        private void ApplyLastLlmReply()
        {
            var last = ChatMessages.LastOrDefault(m =>
                string.Equals(m.Role, "Assistant", StringComparison.OrdinalIgnoreCase));
            if (last == null || string.IsNullOrWhiteSpace(last.Content))
            {
                LlmApplyStatus = "Aucune réponse de l'assistant à appliquer.";
                return;
            }

            LlmApplyStatus = _host.ApplyLlmSceneHints(last.Content);
            if (!LlmApplyStatus.StartsWith("Applied:", StringComparison.Ordinal))
                LlmApplyStatus = _host.ApplyLlmBehaviorHints(last.Content);
            if (LlmApplyStatus.Contains("Registered", StringComparison.Ordinal) ||
                LlmApplyStatus.StartsWith("Applied:", StringComparison.Ordinal))
                RefreshEntities();
        }

        [RelayCommand]
        private void InsertLightingPrompt()
        {
            ChatInput =
                "Suggest warm sunset lighting for the scene as JSON with " +
                "directionalDirection, color, intensity, fogDensity, enableClouds.";
        }

        [RelayCommand]
        private void InsertBehaviorPrompt()
        {
            ChatInput =
                "Design a sentient simulation agent behavior tree (not a game NPC). " +
                "When an observer is near, action: greet. Otherwise action: patrol. " +
                "Include conditions and LLM query nodes if needed.";
        }

        [RelayCommand]
        private void InsertSdfPrompt()
        {
            ChatInput =
                "Propose a neural SDF primitive as JSON with primitive (sphere|box), " +
                "center [x,y,z], and radius (or size [x,y,z] for a box).";
        }

        [RelayCommand]
        private void AddBlueprintLlmNode()
        {
            _blueprint.Nodes.Add(new BlueprintNode
            {
                Kind = BlueprintNodeKind.LlmQuery,
                Title = "LLM Query",
                Payload = "Decide next action based on nearby observer proximity.",
                X = 120 + _blueprint.Nodes.Count * 24,
                Y = 200,
                Inputs = { new BlueprintPin { Name = "Exec", IsInput = true } },
                Outputs = { new BlueprintPin { Name = "Then", IsInput = false } }
            });
            RefreshBlueprint();
        }

        private static bool LooksLikeSceneControlPrompt(string prompt)
        {
            return prompt.Contains("light", StringComparison.OrdinalIgnoreCase)
                || prompt.Contains("fog", StringComparison.OrdinalIgnoreCase)
                || prompt.Contains("cloud", StringComparison.OrdinalIgnoreCase)
                || prompt.Contains("sdf", StringComparison.OrdinalIgnoreCase)
                || prompt.Contains("sun", StringComparison.OrdinalIgnoreCase)
                || prompt.Contains("éclair", StringComparison.OrdinalIgnoreCase)
                || prompt.Contains("éclairage", StringComparison.OrdinalIgnoreCase);
        }

        [RelayCommand]
        private void TogglePlay()
        {
            _host.SimulationPlaying = !_host.SimulationPlaying;
            _orchestrator.IsPaused = !_host.SimulationPlaying;
            if (_host.RenderEngine != null)
                _host.RenderEngine.IsPaused = false; // keep rendering even if sim paused
            RefreshStatus();
        }

        [RelayCommand]
        private void SpawnAgent()
        {
            _host.SpawnAgent("patrol", new Vector3(Random.Shared.NextSingle() * 4 - 2, 0, Random.Shared.NextSingle() * 4 - 2));
            RefreshEntities();
        }

        [RelayCommand]
        private async Task StartEvolutionAsync()
        {
            try
            {
                if (!IsInspectorModeEnabled)
                    IsInspectorModeEnabled = true;
                EvolutionStatus = "Démarrage…";
                await Task.Run(async () => await _host.StartEvolutionAsync(20, 5));
                EvolutionStatus = $"Terminé — gen {_host.EvolutionGeneration} fitness={_host.BestFitness:F3} (volume mis à jour)";
                RefreshEntities();
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Evolution failed", ex);
                EvolutionStatus = $"Erreur — {ex.Message}";
            }
        }

        [RelayCommand]
        private void CancelEvolution() => _host.CancelEvolution();

        [RelayCommand]
        private async Task NewProjectAsync()
        {
            try
            {
                await _host.LoadSceneAsync(null);
                RefreshEntities();
                RefreshLaws();
                _projectPath = null;
                LawStatus = "Nouveau projet";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "New project failed", ex);
                LawStatus = $"Erreur nouveau projet — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task OpenProjectAsync()
        {
            try
            {
                var window = GetMainWindow();
                if (window?.StorageProvider == null)
                    return;
                var files = await window.StorageProvider.OpenFilePickerAsync(new Avalonia.Platform.Storage.FilePickerOpenOptions
                {
                    Title = "Ouvrir un projet Synapse",
                    AllowMultiple = false,
                    FileTypeFilter = new[]
                    {
                        new Avalonia.Platform.Storage.FilePickerFileType("Synapse")
                        {
                            Patterns = new[] { "*.synapse", "*.json" }
                        }
                    }
                });
                var path = files.Count > 0 ? files[0].TryGetLocalPath() : null;
                if (path == null)
                    return;
                await _host.LoadSceneAsync(path);
                GDNN.Streaming.AssetStreamer.AssetRootDirectory = Path.Combine(Path.GetDirectoryName(path)!, "assets");
                _projectPath = path;
                RefreshEntities();
                RefreshLaws();
                LawStatus = $"Ouvert — {Path.GetFileName(path)}";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Open project failed", ex);
                LawStatus = $"Erreur ouverture — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task SaveProjectAsync()
        {
            try
            {
                var path = _projectPath;
                if (string.IsNullOrWhiteSpace(path))
                {
                    var window = GetMainWindow();
                    if (window?.StorageProvider == null)
                        return;
                    var file = await window.StorageProvider.SaveFilePickerAsync(new Avalonia.Platform.Storage.FilePickerSaveOptions
                    {
                        Title = "Enregistrer le projet Synapse",
                        DefaultExtension = "synapse",
                        SuggestedFileName = "project.synapse",
                        FileTypeChoices = new[]
                        {
                            new Avalonia.Platform.Storage.FilePickerFileType("Synapse")
                            {
                                Patterns = new[] { "*.synapse" }
                            }
                        }
                    });
                    path = file?.TryGetLocalPath();
                }
                if (string.IsNullOrWhiteSpace(path))
                    return;

                Directory.CreateDirectory(_config.ProjectsDirectory);
                var assetsDir = Path.Combine(Path.GetDirectoryName(path)!, "assets");
                Directory.CreateDirectory(assetsDir);
                GDNN.Streaming.AssetStreamer.AssetRootDirectory = assetsDir;
                await _host.SaveSceneAsync(path);
                _projectPath = path;
                _logger.Info("Studio", $"Saved {path}");
                LawStatus = $"Enregistré — {Path.GetFileName(path)}";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Save project failed", ex);
                LawStatus = $"Erreur enregistrement — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task EnableVrAsync()
        {
            try
            {
                CollaborationStatus = "Démarrage OpenXR…";
                bool ok = await _host.EnableVrAsync();
                VrStatusText = _host.VrStatusText;
                CollaborationStatus = ok ? "VR activé" : "VR indisponible";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "VR enable failed", ex);
                CollaborationStatus = $"Erreur VR — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task DisableVrAsync()
        {
            await _host.DisableVrAsync();
            VrStatusText = _host.VrStatusText;
            CollaborationStatus = "VR arrêté";
        }

        [RelayCommand]
        private async Task StartWanHostAsync()
        {
            try
            {
                if (string.IsNullOrWhiteSpace(WanSessionCode))
                {
                    CollaborationStatus = "Code de session requis";
                    return;
                }

                CollaborationStatus = "Hébergement WAN…";
                await _host.StartWanHostAsync(WanSessionCode.Trim(), WanPort);
                WanStatusText = _host.WanStatusText;
                WanRendezvousPort = _host.WanHub?.RendezvousPort ?? 0;
                CollaborationStatus = $"Hôte prêt — rdv UDP {WanRendezvousPort}";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "WAN host failed", ex);
                CollaborationStatus = $"Erreur WAN host — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task JoinWanAsync()
        {
            try
            {
                if (string.IsNullOrWhiteSpace(WanSessionCode))
                {
                    CollaborationStatus = "Code de session requis";
                    return;
                }

                CollaborationStatus = "Connexion WAN…";
                await _host.JoinWanAsync(WanSessionCode.Trim(), WanRendezvousPort);
                WanStatusText = _host.WanStatusText;
                CollaborationStatus = "Connecté au pair WAN";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "WAN join failed", ex);
                CollaborationStatus = $"Erreur WAN join — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task StopWanAsync()
        {
            await _host.StopWanAsync();
            WanStatusText = _host.WanStatusText;
            CollaborationStatus = "WAN arrêté";
        }

        [RelayCommand]
        private async Task ExportWebStudioAsync()
        {
            try
            {
                var window = GetMainWindow();
                if (window?.StorageProvider == null)
                    return;
                var folder = await window.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
                {
                    Title = "Exporter Synapse Web Studio (WASM)",
                    AllowMultiple = false
                });
                var path = folder.Count > 0 ? folder[0].TryGetLocalPath() : null;
                if (string.IsNullOrWhiteSpace(path))
                    return;

                CollaborationStatus = "Publication Web Studio…";
                var result = await _host.ExportWebStudioAsync(path);
                WebStatusText = _host.WebStatusText;
                CollaborationStatus = result.UsedDotnetPublish
                    ? $"WASM publié — {result.OutputDirectory}"
                    : $"Site WebGPU — {result.OutputDirectory}";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Web export failed", ex);
                CollaborationStatus = $"Erreur export web — {ex.Message}";
            }
        }

        [RelayCommand]
        private void About() => ShowAboutText = true;

        [RelayCommand]
        private void Exit()
        {
            if (Avalonia.Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
                desktop.Shutdown();
        }

        private void RefreshBlueprint()
        {
            BlueprintNodes.Clear();
            foreach (var n in _blueprint.Nodes)
                BlueprintNodes.Add($"{n.Kind}: {n.Title}");
            var (ok, msg) = _blueprint.Validate();
            BlueprintStatus = msg;
        }

        [RelayCommand]
        private void NewBlueprint()
        {
            _blueprint = BlueprintDocument.CreateDefault();
            RefreshBlueprint();
        }

        [RelayCommand]
        private void AddBlueprintNode()
        {
            _blueprint.Nodes.Add(new BlueprintNode
            {
                Kind = BlueprintNodeKind.Action,
                Title = $"Action_{_blueprint.Nodes.Count}",
                Payload = "guard",
                X = 100 + _blueprint.Nodes.Count * 20,
                Y = 140,
                Inputs = { new BlueprintPin { Name = "Exec", IsInput = true } },
                Outputs = { new BlueprintPin { Name = "Then", IsInput = false } }
            });
            RefreshBlueprint();
        }

        [RelayCommand]
        private void CompileBlueprint()
        {
            try
            {
                _host.CompileAndSpawnBlueprint(_blueprint, Vector3.Zero);
                RefreshEntities();
                BlueprintStatus = $"Compilé → arbre '{_blueprint.Name}' enregistré + agent";
            }
            catch (Exception ex)
            {
                BlueprintStatus = $"Échec de compilation : {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task SaveBlueprintAsync()
        {
            try
            {
                var window = GetMainWindow();
                if (window?.StorageProvider == null)
                    return;
                var file = await window.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
                {
                    Title = "Enregistrer le blueprint",
                    DefaultExtension = "blueprint.json",
                    SuggestedFileName = "agent.blueprint.json",
                    FileTypeChoices = new[]
                    {
                        new FilePickerFileType("Blueprint") { Patterns = new[] { "*.blueprint.json", "*.json" } }
                    }
                });
                var path = file?.TryGetLocalPath();
                if (path == null)
                    return;
                var (ok, msg) = _blueprint.Validate();
                if (!ok)
                {
                    BlueprintStatus = $"Validation échouée — {msg}";
                    return;
                }
                await _blueprint.SaveAsync(path);
                BlueprintStatus = $"Enregistré : {path}";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Save blueprint failed", ex);
                BlueprintStatus = $"Erreur enregistrement — {ex.Message}";
            }
        }

        [RelayCommand]
        private void ApplySculptStroke()
        {
            _sculpt.BrushRadius = (float)SculptRadius;
            _sculpt.BrushStrength = (float)SculptStrength;
            _sculpt.Invert = SculptInvert;
            var pos = SelectedEntity?.Position ?? Vector3.Zero;
            _sculpt.ApplyStroke(pos.X, pos.Y, pos.Z);
            float disp = _sculpt.SampleDisplacement(pos.X, pos.Y, pos.Z);
            SculptStatus = $"Coups={_sculpt.Strokes.Count}  déplacement@sélection={disp:F3}";
        }

        [RelayCommand]
        private void ClearSculpt()
        {
            _sculpt.Clear();
            SculptStatus = "Effacé";
        }

        [RelayCommand]
        private async Task ImportMegascansAsync()
        {
            try
            {
                var window = GetMainWindow();
                if (window?.StorageProvider == null)
                    return;

                string? path = MegascansPath;
                if (string.IsNullOrWhiteSpace(path))
                {
                    var folders = await window.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
                    {
                        Title = "Dossier d'asset Megascans",
                        AllowMultiple = false
                    });
                    path = folders.Count > 0 ? folders[0].TryGetLocalPath() : null;
                }

                if (string.IsNullOrWhiteSpace(path) || !Directory.Exists(path))
                {
                    MegascansStatus = "Aucun dossier sélectionné";
                    return;
                }

                MegascansPath = path;
                MegascansStatus = "Import en cours…";
                var entry = await _megascans.ImportAssetAsync(path);
                MegascansStatus = entry.ImportSucceeded
                    ? $"OK — {entry.Asset?.Name} ({entry.ImportDuration.TotalMilliseconds:F0} ms)"
                    : $"Échec — {string.Join("; ", entry.Warnings)}";
                _logger.Info("Megascans", MegascansStatus);
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Megascans import failed", ex);
                MegascansStatus = $"Erreur import — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task ImportLawPackageAsync()
        {
            try
            {
                var window = GetMainWindow();
                if (window?.StorageProvider == null)
                    return;
                var files = await window.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
                {
                    Title = "Importer un package de loi (.synapse-law)",
                    AllowMultiple = false,
                    FileTypeFilter = new[]
                    {
                        new FilePickerFileType("Synapse Law") { Patterns = new[] { "*.synapse-law" } }
                    }
                });
                var path = files.Count > 0 ? files[0].TryGetLocalPath() : null;
                if (string.IsNullOrWhiteSpace(path))
                    return;

                var package = await _host.ImportLawPackageAsync(path);
                MarketplaceStatus = _host.MarketplaceStatusText;
                RefreshLaws();
                RefreshMarketplacePackages();
                _logger.Info("Studio", $"Imported law package {package.Id}");
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Law import failed", ex);
                MarketplaceStatus = $"Erreur import loi — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task ExportActiveLawPackageAsync()
        {
            try
            {
                var window = GetMainWindow();
                if (window?.StorageProvider == null)
                    return;
                var file = await window.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
                {
                    Title = "Exporter la loi active (.synapse-law)",
                    DefaultExtension = "synapse-law",
                    SuggestedFileName = $"{_host.ActiveLawId ?? "law"}.synapse-law",
                    FileTypeChoices = new[]
                    {
                        new FilePickerFileType("Synapse Law") { Patterns = new[] { "*.synapse-law" } }
                    }
                });
                var path = file?.TryGetLocalPath();
                if (string.IsNullOrWhiteSpace(path))
                    return;

                await _host.ExportActiveLawPackageAsync(path);
                MarketplaceStatus = _host.MarketplaceStatusText;
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Law export failed", ex);
                MarketplaceStatus = $"Erreur export loi — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task ExportSceneGlTFAsync()
        {
            try
            {
                var window = GetMainWindow();
                if (window?.StorageProvider == null)
                    return;
                var file = await window.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
                {
                    Title = "Exporter la scène en glTF",
                    DefaultExtension = "gltf",
                    SuggestedFileName = "scene.gltf",
                    FileTypeChoices = new[]
                    {
                        new FilePickerFileType("glTF") { Patterns = new[] { "*.gltf" } }
                    }
                });
                var path = file?.TryGetLocalPath();
                if (string.IsNullOrWhiteSpace(path))
                    return;

                var result = await _host.ExportSceneGlTFAsync(path);
                CollaborationStatus = result.Success
                    ? $"glTF exporté — {result.EntityCount} entités → {Path.GetFileName(path)}"
                    : $"Échec export glTF — {result.ErrorMessage}";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "glTF export failed", ex);
                CollaborationStatus = $"Erreur export glTF — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task ExportBestGenomeAsync()
        {
            try
            {
                var window = GetMainWindow();
                if (window?.StorageProvider == null)
                    return;
                var file = await window.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
                {
                    Title = "Exporter le meilleur génome (JSON)",
                    DefaultExtension = "json",
                    SuggestedFileName = "best-genome.json",
                    FileTypeChoices = new[]
                    {
                        new FilePickerFileType("JSON") { Patterns = new[] { "*.json" } }
                    }
                });
                var path = file?.TryGetLocalPath();
                if (string.IsNullOrWhiteSpace(path))
                    return;

                await _host.ExportBestGenomeAsync(path);
                EvolutionStatus = $"Génome exporté → {Path.GetFileName(path)}";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Genome export failed", ex);
                EvolutionStatus = $"Erreur export génome — {ex.Message}";
            }
        }

        [RelayCommand]
        private void RefreshBehaviorTree()
        {
            if (SelectedEntity == null || SelectedEntity.Type != EntityType.Agent)
            {
                BehaviorTreeText = "";
                HasAgentSelection = false;
                return;
            }

            BehaviorTreeText = _host.GetAgentBehaviorTreeText(SelectedEntity.Id) ?? "(aucun arbre de comportement)";
            HasAgentSelection = true;
        }

        [RelayCommand]
        private async Task SynchronizeTwinsAsync()
        {
            try
            {
                await _host.SynchronizeTwinsAsync();
                TwinStatusText = _host.TwinStatusText;
                RefreshTwins();
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Twin sync failed", ex);
                TwinStatusText = $"Erreur sync jumeaux — {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task ExportSelectedTwinAsync()
        {
            try
            {
                if (SelectedTwin == null)
                {
                    TwinStatusText = "Sélectionnez un jumeau à exporter.";
                    return;
                }

                var window = GetMainWindow();
                if (window?.StorageProvider == null)
                    return;
                var file = await window.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
                {
                    Title = "Exporter un snapshot de jumeau (JSON)",
                    DefaultExtension = "json",
                    SuggestedFileName = $"twin-{SelectedTwin.DisplayLine}.json",
                    FileTypeChoices = new[]
                    {
                        new FilePickerFileType("JSON") { Patterns = new[] { "*.json" } }
                    }
                });
                var path = file?.TryGetLocalPath();
                if (string.IsNullOrWhiteSpace(path))
                    return;

                await _host.ExportTwinSnapshotAsync(SelectedTwin.Id, path);
                TwinStatusText = _host.TwinStatusText;
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Twin export failed", ex);
                TwinStatusText = $"Erreur export jumeau — {ex.Message}";
            }
        }

        [RelayCommand]
        private void RegisterSceneTwin()
        {
            if (SelectedEntity == null)
            {
                TwinStatusText = "Sélectionnez une entité à jumeler.";
                return;
            }

            var twin = _host.RegisterTwin(SelectedEntity.Name);
            TwinStatusText = _host.TwinStatusText;
            RefreshTwins();
            _logger.Info("Studio", $"Registered twin {twin.Id} for {SelectedEntity.Name}");
        }

        [RelayCommand]
        private async Task LoadPluginsFromDirectoryAsync()
        {
            try
            {
                if (_pluginHost == null)
                {
                    PluginStatusText = "Plugins indisponibles (host non initialisé).";
                    return;
                }

                var window = GetMainWindow();
                if (window?.StorageProvider == null)
                    return;
                var folders = await window.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
                {
                    Title = "Dossier de plugins Synapse",
                    AllowMultiple = false
                });
                var path = folders.Count > 0 ? folders[0].TryGetLocalPath() : null;
                if (string.IsNullOrWhiteSpace(path))
                    return;

                int count = _pluginHost.LoadFromDirectory(path, _host);
                RefreshPlugins();
                PluginStatusText = $"{count} plugin(s) chargé(s) depuis {Path.GetFileName(path)} · total {_pluginHost.LoadedPlugins.Count}";
            }
            catch (Exception ex)
            {
                _logger.Error("Studio", "Plugin load failed", ex);
                PluginStatusText = $"Erreur chargement plugins — {ex.Message}";
            }
        }

        private void TryLoadMarketplaceCatalog()
        {
            try
            {
                var dir = ResolveSampleDirectory(Path.Combine("samples", "laws"));
                if (dir != null)
                {
                    // LoadMarketplaceCatalog is sync-over-async internally; run it off the UI
                    // SynchronizationContext to avoid a continuation deadlock at construction.
                    Task.Run(() => _host.LoadMarketplaceCatalog(dir)).GetAwaiter().GetResult();
                }
                MarketplaceStatus = _host.MarketplaceStatusText;
            }
            catch (Exception ex)
            {
                _logger.Warn("Studio", $"Marketplace catalog load failed: {ex.Message}");
            }
        }

        private void RefreshMarketplacePackages()
        {
            MarketplacePackages.Clear();
            foreach (var package in _host.ListMarketplaceLaws())
            {
                MarketplacePackages.Add(new LawCatalogEntry
                {
                    Id = package.Id,
                    Name = package.Name,
                    Category = package.Category,
                    Description = package.Description,
                    Expression = package.Expression
                });
            }
        }

        private void RefreshTwins()
        {
            var selectedId = SelectedTwin?.Id;
            Twins.Clear();
            foreach (var twin in _host.ListTwins())
            {
                Twins.Add(new TwinListEntry
                {
                    Id = twin.Id,
                    PhysicalId = twin.PhysicalId,
                    Status = twin.SynchronizationStatus.ToString()
                });
            }
            if (selectedId.HasValue)
                SelectedTwin = Twins.FirstOrDefault(t => t.Id == selectedId.Value);
        }

        private void RefreshPlugins()
        {
            Plugins.Clear();
            if (_pluginHost == null)
            {
                PluginStatusText = "Plugins indisponibles (host non initialisé).";
                return;
            }

            foreach (var meta in _pluginHost.LoadedPlugins)
            {
                Plugins.Add(new PluginListEntry
                {
                    Id = meta.Id,
                    Name = meta.Name,
                    Version = meta.Version
                });
            }
            PluginStatusText = Plugins.Count == 0
                ? "Plugins : aucun chargé"
                : $"Plugins : {Plugins.Count} chargé(s)";
        }

        private static string? ResolveSampleDirectory(string relative)
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null)
            {
                var candidate = Path.Combine(dir.FullName, relative);
                if (Directory.Exists(candidate))
                    return candidate;
                dir = dir.Parent;
            }
            return null;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _uiTimer.Stop();
            _host.ViewportEntitySelected -= OnViewportEntitySelected;
            _host.InspectorFeedEntryAdded -= OnInspectorFeedEntry;
            _host.CollaborationPatchApplied -= OnCollaborationPatchApplied;
            _megascans.Dispose();
            GC.SuppressFinalize(this);
        }

        private static Window? GetMainWindow() =>
            Avalonia.Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop
                ? desktop.MainWindow
                : null;
    }
}
