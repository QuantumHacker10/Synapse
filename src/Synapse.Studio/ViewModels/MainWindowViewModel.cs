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
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;
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
        private readonly DispatcherTimer _uiTimer;
        private readonly MegascansBridge _megascans = new();
        private readonly SculptSession _sculpt = new();
        private BlueprintDocument _blueprint = BlueprintDocument.CreateDefault();
        private string? _projectPath;
        private bool _disposed;

        public MainWindowViewModel(EngineHost host, FrameOrchestrator orchestrator, ISynapseLogger logger, SynapseConfig config)
        {
            _host = host;
            _orchestrator = orchestrator;
            _logger = logger;
            _config = config;

            RefreshEntities();
            RefreshLaws();
            RefreshBlueprint();
            _host.ViewportEditor.ShowGrid = ShowViewportGrid;
            _host.ViewportEditor.ShowGizmos = ShowViewportGizmos;
            _host.ViewportEditor.ToolMode = ViewportToolMode.Translate;
            _host.ViewportEntitySelected += OnViewportEntitySelected;

            _uiTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(200) };
            _uiTimer.Tick += (_, _) => RefreshStatus();
            _uiTimer.Start();

            ViewportHint = "Viewport Vulkan — Grille/Gizmos · Ctrl+clic = relier les nœuds blueprint · Espace = pause";
            SculptStatus = "Pinceau prêt";
            MegascansStatus = $"Bibliothèque : {_megascans.Config.LibraryRootPath}";
            LlmStatusText = _host.LlmProviderSummary;
            AboutText =
                "SYNAPSE OMNIA 1.2 — Moteur de simulation 3D\n\n" +
                "Un monde numérique que l'on observe, modifie et fait évoluer : formes apprises (G-DNN), " +
                "lois physiques réécrivables, évolution NEAT-G, habitants sentients, assistance créative " +
                "(LLM) et rendu Vulkan temps réel.\n\n" +
                "Site : https://quantumhacker10.github.io/Synapse/\n" +
                "Licence propriétaire — usage personnel non commercial.\n\n" +
                "Apprendre · Réécrire · Cultiver";

            foreach (EntityType t in Enum.GetValues<EntityType>())
            {
                if (t != EntityType.Unknown)
                    EntityTypes.Add(t);
            }
            NewEntityType = EntityType.Empty;
        }

        public ObservableCollection<SceneEntity> Entities { get; } = new();
        public ObservableCollection<string> LawIds { get; } = new();
        public ObservableCollection<EntityType> EntityTypes { get; } = new();
        public ObservableCollection<ChatMessageRecord> ChatMessages { get; } = new();
        public ObservableCollection<string> BlueprintNodes { get; } = new();

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
        private Guid? _jointPartnerId;

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
                return;
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

        [RelayCommand]
        private void SetViewportToolTranslate() => ViewportTool = ViewportToolMode.Translate;

        [RelayCommand]
        private void SetViewportToolRotate() => ViewportTool = ViewportToolMode.Rotate;

        [RelayCommand]
        private void SetViewportToolSelect() => ViewportTool = ViewportToolMode.Select;

        partial void OnSelectedLawIdChanged(string? value)
        {
            if (value == null)
                return;
            var law = _host.ListLaws().FirstOrDefault(l => l.Id == value);
            if (law.Id != null)
                LawExpression = law.Expression;
        }

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
                    Type = Enum.TryParse<EntityType>(e.Type, true, out var t) ? t : EntityType.Empty,
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

        private void RefreshLaws()
        {
            LawIds.Clear();
            foreach (var law in _host.ListLaws())
                LawIds.Add(law.Id);
            SelectedLawId = _host.ActiveLawId ?? LawIds.FirstOrDefault();
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

            var giGpu = _host.RenderEngine?.SceneRenderer?.GiUsesGpuReadback ?? false;
            GiStatus = giGpu ? "GI : lecture G-buffer GPU active" : "GI : constantes de repli";
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
            string content = response?.Content ?? "(aucune réponse)";
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
                "Design an NPC behavior tree. When the player is near, action: greet. " +
                "Otherwise action: patrol. Include conditions and LLM query nodes if needed.";
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
                Payload = "Decide next action based on player proximity.",
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
            EvolutionStatus = "Démarrage…";
            await Task.Run(async () => await _host.StartEvolutionAsync(20, 5));
            EvolutionStatus = $"Terminé — gen {_host.EvolutionGeneration} fitness={_host.BestFitness:F3} (volume mis à jour)";
            RefreshEntities();
        }

        [RelayCommand]
        private void CancelEvolution() => _host.CancelEvolution();

        [RelayCommand]
        private async Task NewProjectAsync()
        {
            await _host.LoadSceneAsync(null);
            RefreshEntities();
            RefreshLaws();
            _projectPath = null;
        }

        [RelayCommand]
        private async Task OpenProjectAsync()
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
        }

        [RelayCommand]
        private async Task SaveProjectAsync()
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
            await _blueprint.SaveAsync(path);
            BlueprintStatus = $"Enregistré : {path}";
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

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _uiTimer.Stop();
            _host.ViewportEntitySelected -= OnViewportEntitySelected;
            _megascans.Dispose();
            GC.SuppressFinalize(this);
        }

        private static Window? GetMainWindow() =>
            Avalonia.Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop
                ? desktop.MainWindow
                : null;
    }
}
