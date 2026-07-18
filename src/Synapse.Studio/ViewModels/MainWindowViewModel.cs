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
using GDNN.Studio.Contracts;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;
using SceneEntity = GDNN.Studio.Contracts.SceneEntity;
using EntityType = GDNN.Studio.Contracts.EntityType;

namespace GDNN.Studio.ViewModels
{
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

            _uiTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(200) };
            _uiTimer.Tick += (_, _) => RefreshStatus();
            _uiTimer.Start();

            ViewportHint = "Viewport Vulkan — WASD / souris (GLFW) ou panneau embarqué (Windows HWND)";
            PlayPauseLabel = "Pause";
            SculptStatus = "Brush ready";
            MegascansStatus = $"Library: {_megascans.Config.LibraryRootPath}";
        }

        public ObservableCollection<SceneEntity> Entities { get; } = new();
        public ObservableCollection<string> LawIds { get; } = new();
        public ObservableCollection<ChatMessageRecord> ChatMessages { get; } = new();
        public ObservableCollection<string> BlueprintNodes { get; } = new();

        [ObservableProperty] private SceneEntity? selectedEntity;
        [ObservableProperty] private string? selectedLawId;
        [ObservableProperty] private string lawExpression = "∂T/∂t = α*∇²T";
        [ObservableProperty] private string lawStatus = "Ready";
        [ObservableProperty] private string chatInput = "";
        [ObservableProperty] private string statusFps = "FPS: —";
        [ObservableProperty] private string statusPhysics = "Physics: —";
        [ObservableProperty] private string statusSimulation = "Sim: —";
        [ObservableProperty] private string statusQuality = "Quality: —";
        [ObservableProperty] private string statusLaw = "Law: —";
        [ObservableProperty] private string inspectorText = "Select an entity";
        [ObservableProperty] private string performanceReport = "Waiting for frames…";
        [ObservableProperty] private string simulationStatus = "Playing";
        [ObservableProperty] private string evolutionStatus = "Idle";
        [ObservableProperty] private string viewportHint = "";
        [ObservableProperty] private string playPauseLabel = "Pause";
        [ObservableProperty] private string blueprintStatus = "Ready";
        [ObservableProperty] private string sculptStatus = "";
        [ObservableProperty] private double sculptRadius = 0.5;
        [ObservableProperty] private double sculptStrength = 0.15;
        [ObservableProperty] private bool sculptInvert;
        [ObservableProperty] private string megascansPath = "";
        [ObservableProperty] private string megascansStatus = "";
        [ObservableProperty] private string llmApplyStatus = "Ask for lighting JSON or SDF hints, then Apply.";

        partial void OnSelectedEntityChanged(SceneEntity? value)
        {
            InspectorText = value == null
                ? "Select an entity"
                : $"{value.Name}\nType: {value.Type}\nPos: {value.Position}\nBehavior: {value.BehaviorProfile ?? "—"}\nGenome: {value.GenomeId ?? "—"}";
        }

        partial void OnSelectedLawIdChanged(string? value)
        {
            if (value == null) return;
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
                    LawId = e.LawId
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
            StatusFps = $"FPS: {s.Fps:F0}";
            StatusPhysics = $"Physics: {s.PhysicsMs:F1} ms  T≈{s.FieldTemperatureAvg:F1}K";
            StatusSimulation = $"Sim: {s.SimulationMs:F1} ms  entities={s.EntityCount}";
            StatusQuality = $"Quality: {s.QualityPreset}";
            StatusLaw = $"Law: {s.ActiveLawId}";
            PerformanceReport =
                $"Frame {s.RenderMs:F1} ms | Physics {s.PhysicsMs:F1} | Sim {s.SimulationMs:F1}\n" +
                $"Total time {s.TotalTime:F1}s | Evolution gen={_host.EvolutionGeneration} fitness={_host.BestFitness:F3}";
            SimulationStatus = _host.SimulationPlaying ? $"Playing — {s.EntityCount} agents" : "Paused";
            PlayPauseLabel = _host.SimulationPlaying ? "Pause" : "Play";
            if (_host.EvolutionRunning)
                EvolutionStatus = $"Running gen {_host.EvolutionGeneration} best={_host.BestFitness:F3}";
        }

        [RelayCommand]
        private void AddEntity()
        {
            var id = _host.CreateSceneEntity($"Entity_{Entities.Count + 1}", "Empty");
            RefreshEntities();
            SelectedEntity = Entities.FirstOrDefault(e => e.Id == id);
            _logger.Info("Studio", $"Created entity {id}");
        }

        [RelayCommand]
        private void DeleteEntity()
        {
            if (SelectedEntity == null) return;
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
                : $"Failed: {result.Message}";
        }

        [RelayCommand]
        private async Task SendChatAsync()
        {
            if (string.IsNullOrWhiteSpace(ChatInput)) return;
            var prompt = ChatInput.Trim();
            ChatInput = "";
            ChatMessages.Add(new ChatMessageRecord { Role = "user", Content = prompt });

            // Steer the model toward structured lighting/SDF JSON when relevant.
            string routedPrompt = LooksLikeSceneControlPrompt(prompt)
                ? prompt + "\n\nRespond with a JSON object including any of: " +
                  "directionalDirection [x,y,z], color (#RRGGBB), intensity, fogDensity, " +
                  "enableClouds, and/or primitive/center/radius for an SDF hint."
                : prompt;

            var response = await _host.ChatAsync(routedPrompt);
            string content = response?.Content ?? "(no response)";
            ChatMessages.Add(new ChatMessageRecord
            {
                Role = "assistant",
                Content = content,
                Provider = response?.Provider ?? ""
            });

            // Auto-apply when the reply contains parseable lighting/SDF params.
            LlmApplyStatus = _host.ApplyLlmSceneHints(content);
            if (LlmApplyStatus.StartsWith("Applied:", StringComparison.Ordinal))
                RefreshEntities();
        }

        [RelayCommand]
        private void ApplyLastLlmReply()
        {
            var last = ChatMessages.LastOrDefault(m =>
                string.Equals(m.Role, "assistant", StringComparison.OrdinalIgnoreCase));
            if (last == null || string.IsNullOrWhiteSpace(last.Content))
            {
                LlmApplyStatus = "No assistant reply to apply.";
                return;
            }

            LlmApplyStatus = _host.ApplyLlmSceneHints(last.Content);
            if (LlmApplyStatus.StartsWith("Applied:", StringComparison.Ordinal))
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
        private void InsertSdfPrompt()
        {
            ChatInput =
                "Propose a neural SDF primitive as JSON with primitive (sphere|box), " +
                "center [x,y,z], and radius (or size [x,y,z] for a box).";
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
            EvolutionStatus = "Starting…";
            await Task.Run(async () => await _host.StartEvolutionAsync(20, 5));
            EvolutionStatus = $"Done gen={_host.EvolutionGeneration} best={_host.BestFitness:F3}";
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
            if (window?.StorageProvider == null) return;
            var files = await window.StorageProvider.OpenFilePickerAsync(new Avalonia.Platform.Storage.FilePickerOpenOptions
            {
                Title = "Open Synapse Project",
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
            if (path == null) return;
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
                if (window?.StorageProvider == null) return;
                var file = await window.StorageProvider.SaveFilePickerAsync(new Avalonia.Platform.Storage.FilePickerSaveOptions
                {
                    Title = "Save Synapse Project",
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
            if (string.IsNullOrWhiteSpace(path)) return;

            Directory.CreateDirectory(_config.ProjectsDirectory);
            var assetsDir = Path.Combine(Path.GetDirectoryName(path)!, "assets");
            Directory.CreateDirectory(assetsDir);
            GDNN.Streaming.AssetStreamer.AssetRootDirectory = assetsDir;
            await _host.SaveSceneAsync(path);
            _projectPath = path;
            _logger.Info("Studio", $"Saved {path}");
        }

        [RelayCommand]
        private void About()
        {
            LawStatus =
                "SYNAPSE OMNIA 1.1 — G-DNN Studio. SDF neuronaux (G-DNN), illumination L-DNN " +
                "(teacher path tracing, ombres & reflets neuronaux, fog/nuages), NEAT-G, " +
                "lois physiques vivantes, console LLM → éclairage/SDF. " +
                "Les moteurs classiques assemblent ; Synapse apprend, réécrit et cultive.";
        }

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
                var profile = _blueprint.CompileToBehaviorTreeName();
                _host.SpawnAgent(profile, Vector3.Zero);
                RefreshEntities();
                BlueprintStatus = $"Compiled → agent '{profile}' spawned";
            }
            catch (Exception ex)
            {
                BlueprintStatus = $"Compile failed: {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task SaveBlueprintAsync()
        {
            var window = GetMainWindow();
            if (window?.StorageProvider == null) return;
            var file = await window.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
            {
                Title = "Save Blueprint",
                DefaultExtension = "blueprint.json",
                SuggestedFileName = "agent.blueprint.json",
                FileTypeChoices = new[]
                {
                    new FilePickerFileType("Blueprint") { Patterns = new[] { "*.blueprint.json", "*.json" } }
                }
            });
            var path = file?.TryGetLocalPath();
            if (path == null) return;
            await _blueprint.SaveAsync(path);
            BlueprintStatus = $"Saved {path}";
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
            SculptStatus = $"Strokes={_sculpt.Strokes.Count}  displacement@selection={disp:F3}";
        }

        [RelayCommand]
        private void ClearSculpt()
        {
            _sculpt.Clear();
            SculptStatus = "Cleared";
        }

        [RelayCommand]
        private async Task ImportMegascansAsync()
        {
            var window = GetMainWindow();
            if (window?.StorageProvider == null) return;

            string? path = MegascansPath;
            if (string.IsNullOrWhiteSpace(path))
            {
                var folders = await window.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
                {
                    Title = "Megascans asset folder",
                    AllowMultiple = false
                });
                path = folders.Count > 0 ? folders[0].TryGetLocalPath() : null;
            }

            if (string.IsNullOrWhiteSpace(path) || !Directory.Exists(path))
            {
                MegascansStatus = "No folder selected";
                return;
            }

            MegascansPath = path;
            MegascansStatus = "Importing…";
            var entry = await _megascans.ImportAssetAsync(path);
            MegascansStatus = entry.ImportSucceeded
                ? $"OK — {entry.Asset?.Name} ({entry.ImportDuration.TotalMilliseconds:F0} ms)"
                : $"Failed — {string.Join("; ", entry.Warnings)}";
            _logger.Info("Megascans", MegascansStatus);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _uiTimer.Stop();
            _megascans.Dispose();
            GC.SuppressFinalize(this);
        }

        private static Window? GetMainWindow() =>
            Avalonia.Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop
                ? desktop.MainWindow
                : null;
    }
}
