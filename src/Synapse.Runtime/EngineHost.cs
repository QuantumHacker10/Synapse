using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Core.NEAT;
using GDNN.Llm;
using GDNN.Rendering.Engine;
using GDNN.Rendering.Quality;
using GDNN.Scene;
using GDNN.Sentience;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Physics;
using Synapse.Simulation.DigitalTwins;

namespace Synapse.Runtime
{
    /// <summary>
    /// Central runtime facade for Synapse Studio: physics, simulation, LLM, rendering,
    /// and scene I/O. Call <see cref="InitializeModules"/> once, then either
    /// <see cref="InitializeRender"/> (GLFW) or <see cref="InitializeRenderFromHwnd"/>
    /// (embedded Windows viewport). Per-frame work is driven by <see cref="FrameOrchestrator"/>
    /// via <see cref="TickPhysics"/>, <see cref="TickSimulationAsync"/>, and
    /// <see cref="TickRender"/>.
    /// </summary>
    public sealed class EngineHost : IAsyncDisposable
    {
        private readonly SynapseConfig _config;
        private readonly ISynapseLogger _logger;
        private RenderEngine? _renderEngine;
        private LivingLawCompiler? _lawCompiler;
        private PhysicsField? _physicsField;
        private SentienceManager? _sentience;
        private HybridLlmRouter? _llmRouter;
        private RuntimeQualityManager? _quality;
        private NeatGEvolutionEngine? _evolution;
        private CancellationTokenSource? _evolutionCts;
        private readonly InMemoryDigitalTwinRegistry _twins = new();
        private SceneDocument _scene = SceneDocument.CreateDemo();
        private bool _renderInitialized;
        private bool _modulesInitialized;
        private string? _activeLawId;
        private int _evolutionGeneration;
        private double _bestFitness;
        private bool _simulationPlaying = true;
        private string _llmProviderSummary = "LLM not initialized";
        private readonly ViewportEditorState _viewportEditor = new();

        /// <summary>Viewport gizmo/grid/selection state for Studio.</summary>
        public ViewportEditorState ViewportEditor => _viewportEditor;

        /// <summary>Creates a host bound to application config and logging.</summary>
        public EngineHost(SynapseConfig config, ISynapseLogger logger)
        {
            _config = config;
            _logger = logger;
        }

        /// <summary>Vulkan render engine, or null before render init.</summary>
        public RenderEngine? RenderEngine => _renderEngine;

        /// <summary>Current in-memory scene document (entities, camera, active law).</summary>
        public SceneDocument Scene => _scene;

        /// <summary>Living-law compiler, available after <see cref="InitializeModules"/>.</summary>
        public LivingLawCompiler? LawCompiler => _lawCompiler;

        /// <summary>Multi-provider LLM router.</summary>
        public HybridLlmRouter? LlmRouter => _llmRouter;

        /// <summary>Sentient entity manager for simulation ticks.</summary>
        public SentienceManager? Sentience => _sentience;

        /// <summary>Registered digital twins (demo world by default).</summary>
        public IDigitalTwinRegistry Twins => _twins;

        /// <summary>Whether <see cref="InitializeRender"/> or <see cref="InitializeRenderFromHwnd"/> completed.</summary>
        public bool IsRenderInitialized => _renderInitialized;

        /// <summary>When false, <see cref="TickSimulationAsync"/> is skipped.</summary>
        public bool SimulationPlaying
        {
            get => _simulationPlaying;
            set => _simulationPlaying = value;
        }

        /// <summary>Id of the physics law currently applied to the field.</summary>
        public string? ActiveLawId => _activeLawId;

        /// <summary>Coarse average temperature (K) sampled from the physics field.</summary>
        public float AverageFieldTemperature { get; private set; } = 300f;

        /// <summary>Number of sentient entities in the simulation.</summary>
        public int EntityCount => _sentience?.EntityCount ?? 0;

        /// <summary>Human-readable quality preset from the adaptive quality manager.</summary>
        public string QualityPresetName => _quality?.CurrentLevel.Preset.ToString() ?? _config.QualityPreset;

        /// <summary>Latest NEAT-G generation index after evolution runs.</summary>
        public int EvolutionGeneration => _evolutionGeneration;

        /// <summary>Best fitness from the last evolution step.</summary>
        public double BestFitness => _bestFitness;

        /// <summary>True while a background NEAT-G run is in progress.</summary>
        public bool EvolutionRunning => _evolutionCts is { IsCancellationRequested: false } && _evolution != null;

        /// <summary>Human-readable list of registered LLM providers.</summary>
        public string LlmProviderSummary => _llmProviderSummary;

        /// <summary>
        /// Lazily constructs physics, simulation, LLM, quality, and twin subsystems.
        /// Safe to call multiple times.
        /// </summary>
        public void InitializeModules()
        {
            if (_modulesInitialized) return;

            _lawCompiler = new LivingLawCompiler();
            _physicsField = CreateSeedField(16);
            _sentience = new SentienceManager();
            _llmRouter = new HybridLlmRouter();
            _llmProviderSummary = LlmProviderBootstrap.Register(_llmRouter, _config, _logger).Summary;
            WireBehaviorLlmRouter();
            _quality = new RuntimeQualityManager(ParseQuality(_config.QualityPreset), AdaptationMode.Dynamic);

            _activeLawId = _scene.ActiveLawId ?? "heat_equation";
            EnsureLawCompiled(_activeLawId, _scene.ActiveLawExpression);
            ApplySceneToSimulation(_scene);

            var twin = new InMemoryDigitalTwin { PhysicalId = "demo-world" };
            twin.SetProperty("Name", _scene.Name);
            twin.SetProperty("EntityType", GDNN.Sentience.EntityType.Environmental.ToString());
            _twins.Register(twin);

            _modulesInitialized = true;
            _logger.Info("EngineHost", "Modules initialized (Physics, Simulation, LLM, Quality, Twins)");
        }

        /// <summary>Creates a standalone GLFW render surface.</summary>
        public void InitializeRender(int width, int height, bool enableValidation = true)
        {
            if (_renderInitialized) return;
            InitializeModules();

            _renderEngine = new RenderEngine();
            _renderEngine.Initialize(width, height, enableValidation);
            _renderInitialized = true;
            SyncSceneToRenderer();
            _logger.Info("EngineHost", $"RenderEngine initialized {width}x{height}");
        }

        /// <summary>Embeds Vulkan into a native window handle (Windows only; falls back to GLFW elsewhere).</summary>
        public void InitializeRenderFromHwnd(IntPtr hwnd, int width, int height, bool enableValidation = true)
        {
            if (_renderInitialized) return;
            if (!OperatingSystem.IsWindows())
            {
                _logger.Warn("EngineHost", "HWND embedding unsupported on this OS — using GLFW");
                InitializeRender(width, height, enableValidation);
                return;
            }

            InitializeModules();

            _renderEngine = new RenderEngine();
            _renderEngine.InitializeFromHwnd(hwnd, width, height, enableValidation);
            _renderInitialized = true;
            SyncSceneToRenderer();
            _logger.Info("EngineHost", $"RenderEngine initialized from HWND {width}x{height}");
        }

        /// <summary>Loads a <c>.synapse</c> project or resets to the built-in demo when <paramref name="path"/> is null.</summary>
        public async Task LoadSceneAsync(string? path, CancellationToken cancellationToken = default)
        {
            InitializeModules();
            _scene = string.IsNullOrWhiteSpace(path)
                ? SceneDocument.CreateDemo()
                : await SceneDocument.LoadAsync(path, cancellationToken).ConfigureAwait(false);

            _activeLawId = _scene.ActiveLawId;
            EnsureLawCompiled(_activeLawId, _scene.ActiveLawExpression);
            ApplySceneToSimulation(_scene);

            if (_renderEngine != null)
            {
                _renderEngine.LoadSceneName(_scene.Name);
                SyncSceneToRenderer();
            }

            if (!string.IsNullOrWhiteSpace(path))
            {
                var assets = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path))!, "assets");
                Directory.CreateDirectory(assets);
                GDNN.Streaming.AssetStreamer.AssetRootDirectory = assets;
            }

            _logger.Info("EngineHost", $"Scene loaded: {_scene.Name} ({_scene.Entities.Count} entities)");
        }

        /// <summary>Persists the scene and syncs the active camera from the render engine when available.</summary>
        public async Task SaveSceneAsync(string path, CancellationToken cancellationToken = default)
        {
            if (_renderEngine != null)
            {
                var cam = _renderEngine.GetCamera();
                _scene.Camera = new CameraData
                {
                    Position = Vec3.From(cam.Position),
                    Yaw = cam.Yaw,
                    Pitch = cam.Pitch,
                    Fov = cam.Fov
                };
            }

            _scene.ActiveLawId = _activeLawId;
            await _scene.SaveAsync(path, cancellationToken).ConfigureAwait(false);
            _logger.Info("EngineHost", $"Scene saved: {path}");
        }

        /// <summary>Applies the active living law to the physics field for one timestep.</summary>
        public void TickPhysics(float dt)
        {
            if (_lawCompiler == null || _physicsField == null || string.IsNullOrEmpty(_activeLawId)) return;

            var budget = TimeSpan.FromMilliseconds(_config.PhysicsBudgetMs);
            var start = Environment.TickCount64;
            try
            {
                _lawCompiler.ApplyLaw(_activeLawId, _physicsField, dt);
                _physicsField.Time += dt;
                AverageFieldTemperature = SampleAverageTemperature(_physicsField);
            }
            catch (Exception ex)
            {
                _logger.Warn("Physics", $"Law apply failed: {ex.Message}");
            }

            if (Environment.TickCount64 - start > budget.TotalMilliseconds)
                _logger.Debug("Physics", "Exceeded physics budget");
        }

        /// <summary>Advances sentient entities when <see cref="SimulationPlaying"/> is true.</summary>
        public async Task TickSimulationAsync(float dt, CancellationToken cancellationToken)
        {
            if (!_simulationPlaying || _sentience == null) return;
            try
            {
                await _sentience.UpdateAsync(dt).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.Warn("Simulation", $"Tick failed: {ex.Message}");
            }
        }

        /// <summary>Submits one Vulkan frame when the render engine is initialized and not paused.</summary>
        public void TickRender()
        {
            if (_renderEngine == null || !_renderInitialized) return;
            if (_renderEngine.IsPaused) return;
            _renderEngine.RenderFrame();
        }

        /// <summary>Feeds frame timing into the adaptive quality manager.</summary>
        public void TickQuality(float dt, float frameMs)
        {
            if (_quality == null) return;
            _quality.ReportFrame(frameMs, 0, 0, frameMs);
            _quality.Update(dt);
        }

        /// <summary>Hot-reloads or compiles a living law and marks it active on success.</summary>
        public CompilationResult CompileLaw(string lawId, string expression)
        {
            InitializeModules();
            var result = ActivateLaw(lawId, expression);
            if (result.Success)
            {
                _scene.ActiveLawId = lawId;
                _scene.ActiveLawExpression = expression;
                _logger.Info("Physics", $"Law '{lawId}' active ({result.InstructionCount} ops)");
            }
            else
            {
                _logger.Warn("Physics", $"Law compile failed: {result.Message}");
            }
            return result;
        }

        /// <summary>Returns all laws from the built-in library plus any user-defined entries.</summary>
        public IReadOnlyList<(string Id, string Name, string Expression)> ListLaws()
        {
            InitializeModules();
            return _lawCompiler!.Library.AllEntries
                .Select(e => (e.Id, e.Name, e.Expression))
                .ToList();
        }

        /// <summary>Routes a single user prompt through the hybrid LLM stack (local first, then cloud).</summary>
        public async Task<LlmResponse?> ChatAsync(string prompt, CancellationToken cancellationToken = default)
        {
            InitializeModules();
            try
            {
                var messages = new List<ChatMessage>
                {
                    new() { Role = MessageRole.User, Content = prompt }
                };
                var context = new PromptContext
                {
                    PreferredMode = LlmProviderMode.LocalOllama,
                    MaxLatencyMs = 30000,
                    TaskType = LlmTaskType.QueryAnswering
                };
                return await _llmRouter!.RouteChatAsync(messages, context, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.Warn("LLM", $"Chat failed (providers may be offline): {ex.Message}");
                return new LlmResponse
                {
                    Content = $"[Synapse offline reply] No LLM provider available. Configure Ollama at {_config.Llm.OllamaBaseUrl} or set API keys. Echo: {prompt}",
                    Provider = "Offline",
                    Model = "none"
                };
            }
        }

        /// <summary>Runs NEAT-G evolution asynchronously until completion or <see cref="CancelEvolution"/>.</summary>
        public async Task StartEvolutionAsync(int population, int generations, CancellationToken cancellationToken = default)
        {
            InitializeModules();
            _evolutionCts?.Cancel();
            _evolutionCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

            var config = new EvolutionConfig
            {
                PopulationSize = Math.Clamp(population, 10, 200),
                MaxGenerations = Math.Clamp(generations, 1, 100),
                RandomSeed = 42
            };
            _evolution = new NeatGEvolutionEngine(config);
            _evolution.InitializePopulation(3, 1);

            _logger.Info("Evolution", $"Starting NEAT-G pop={config.PopulationSize} gens={config.MaxGenerations}");

            try
            {
                var context = EvaluationContext.CreateDefault();
                for (int g = 0; g < config.MaxGenerations; g++)
                {
                    _evolutionCts.Token.ThrowIfCancellationRequested();
                    var metrics = await _evolution.StepAsync(context, _evolutionCts.Token).ConfigureAwait(false);
                    _evolutionGeneration = _evolution.CurrentGeneration;
                    _bestFitness = metrics.BestFitness;
                }

                ApplyEvolutionToScene();
            }
            catch (OperationCanceledException)
            {
                _logger.Info("Evolution", "Cancelled");
            }
            catch (Exception ex)
            {
                _logger.Warn("Evolution", $"Evolution step failed: {ex.Message}");
            }
        }

        /// <summary>Requests cancellation of an in-flight evolution run.</summary>
        public void CancelEvolution() => _evolutionCts?.Cancel();

        /// <summary>Spawns a sentient NPC and mirrors it in the scene document.</summary>
        public SentientEntity SpawnAgent(string profile, Vector3 position)
        {
            InitializeModules();
            var factory = new SentientEntityFactory(_sentience!);
            var entity = factory.CreateNPC(position, profile);
            _scene.Entities.Add(new SceneEntityData
            {
                Id = entity.EntityId,
                Name = $"Agent_{profile}_{entity.EntityId.ToString()[..8]}",
                Type = "Character",
                Position = Vec3.From(position),
                BehaviorProfile = profile
            });
            SyncSceneToRenderer();
            return entity;
        }

        /// <summary>Adds a scene entity; spawns a patrol agent when <paramref name="type"/> is Character.</summary>
        public Guid CreateSceneEntity(string name, string type)
        {
            InitializeModules();
            var id = Guid.NewGuid();
            _scene.Entities.Add(new SceneEntityData
            {
                Id = id,
                Name = name,
                Type = type,
                Position = new Vec3(0, 0, 0)
            });

            if (type.Equals("Character", StringComparison.OrdinalIgnoreCase))
                SpawnAgent("patrol", Vector3.Zero);

            SyncSceneToRenderer();
            return id;
        }

        /// <summary>Removes an entity from the scene and the simulation manager.</summary>
        public bool DeleteSceneEntity(Guid id)
        {
            InitializeModules();
            var removed = _scene.Entities.RemoveAll(e => e.Id == id) > 0;
            _sentience?.RemoveEntity(id);
            if (removed) SyncSceneToRenderer();
            return removed;
        }

        /// <summary>
        /// Parses lighting / SDF hints from an LLM reply and applies them to the
        /// renderer (when available) and the scene document.
        /// </summary>
        public string ApplyLlmSceneHints(string llmText)
        {
            if (string.IsNullOrWhiteSpace(llmText))
                return "Empty reply — nothing to apply.";

            InitializeModules();
            var applied = new List<string>();

            if (StructuredOutputParser.TryParseLightingParams(llmText, out LightingParams lighting))
            {
                _renderEngine?.SceneRenderer?.ApplyLlmLighting(lighting);

                var lightId = CreateSceneEntity("LLM_DirectionalLight", "Light");
                var lightEntity = _scene.Entities.FirstOrDefault(e => e.Id == lightId);
                if (lightEntity != null && lighting.DirectionalDirection is { } dir)
                {
                    // Place the light entity opposite the light direction (sun-like).
                    lightEntity.Position = new Vec3(-dir.X * 10f, -dir.Y * 10f, -dir.Z * 10f);
                }

                applied.Add(lighting.Intensity.HasValue
                    ? $"lighting(intensity={lighting.Intensity.Value:F2})"
                    : "lighting");
            }

            if (StructuredOutputParser.TryParseSdfHint(llmText, out SdfHint sdf))
            {
                string primitive = string.IsNullOrWhiteSpace(sdf.Primitive) ? "sphere" : sdf.Primitive;
                var sdfId = CreateSceneEntity($"LLM_SDF_{primitive}", "Volume");
                var sdfEntity = _scene.Entities.FirstOrDefault(e => e.Id == sdfId);
                if (sdfEntity != null)
                {
                    if (sdf.Center is { } c)
                        sdfEntity.Position = new Vec3(c.X, c.Y, c.Z);

                    if (sdf.Size is { } size)
                        sdfEntity.Scale = new Vec3(size.X, size.Y, size.Z);
                    else if (sdf.Radius is { } radius)
                        sdfEntity.Scale = new Vec3(radius, radius, radius);
                }

                applied.Add($"sdf:{primitive}");
            }

            if (applied.Count == 0)
                return "No lighting/SDF parameters found in the reply.";

            string summary = "Applied: " + string.Join(", ", applied);
            _logger.Info("LLM", summary);
            SyncSceneToRenderer();
            return summary;
        }

        /// <summary>Parses behavior-tree hints from LLM output, compiles, and registers the tree.</summary>
        public string ApplyLlmBehaviorHints(string llmText)
        {
            if (string.IsNullOrWhiteSpace(llmText))
                return "Empty reply — nothing to apply.";

            InitializeModules();
            var extracted = StructuredOutputParser.ExtractBehaviorTree(llmText);
            if (!extracted.Success || extracted.Data == null || extracted.Data.Count == 0)
                return "No behavior tree nodes found in the reply.";

            var blueprint = LlmBehaviorTreeConverter.ToBlueprint(extracted.Data);
            var tree = _sentience!.Compiler.CompileFromBlueprint("LLM_Generated", blueprint);
            _sentience.RegisterBehaviorTree(tree.Name, tree);

            var agent = _sentience.CreateEntity(EntityType.NPC, Vector3.Zero, tree.Name);
            _scene.Entities.Add(new SceneEntityData
            {
                Id = agent.EntityId,
                Name = tree.Name,
                Type = "Character",
                Position = Vec3.From(Vector3.Zero),
                BehaviorProfile = tree.Name
            });
            SyncSceneToRenderer();
            return $"Registered behavior tree '{tree.Name}' with {extracted.Data.Count} LLM node(s).";
        }

        /// <summary>Compiles a blueprint graph to a behavior tree and spawns an agent.</summary>
        public SentientEntity CompileAndSpawnBlueprint(BlueprintDocument document, Vector3 position)
        {
            InitializeModules();
            var blueprint = document.CompileToBehaviorTreeBlueprint();
            var tree = _sentience!.Compiler.CompileFromBlueprint(document.Name, blueprint);
            _sentience.RegisterBehaviorTree(document.Name, tree);
            var entity = _sentience.CreateEntity(EntityType.NPC, position, document.Name);
            _scene.Entities.Add(new SceneEntityData
            {
                Id = entity.EntityId,
                Name = $"Agent_{document.Name}",
                Type = "Character",
                Position = Vec3.From(position),
                BehaviorProfile = document.Name
            });
            SyncSceneToRenderer();
            return entity;
        }

        /// <summary>Sets the selected entity for viewport gizmos.</summary>
        public void SetViewportSelection(Guid entityId) => _viewportEditor.SelectedEntityId = entityId;

        /// <summary>Raised when the user picks a different entity in the viewport.</summary>
        public event Action<Guid>? ViewportEntitySelected;

        /// <summary>Left-click: pick entity or begin gizmo drag. Right-click: orbit camera.</summary>
        public void HandleViewportPointerDown(float x, float y, int width, int height, bool rightButton)
        {
            if (_renderEngine?.SceneRenderer == null || !_renderInitialized) return;

            var (camPos, camFront, camUp, yaw, pitch, fov) = _renderEngine.GetCamera();
            GetFramebufferSize(out int fbW, out int fbH, width, height);
            var aspect = (float)fbW / Math.Max(1, fbH);
            var ray = ViewportPickUtility.CreateRayFromScreen(camPos, camFront, camUp, fov, aspect, x, y, fbW, fbH);

            if (rightButton)
            {
                _viewportEditor.IsOrbitingCamera = true;
                _viewportEditor.OrbitStartYaw = yaw;
                _viewportEditor.OrbitStartPitch = pitch;
                _viewportEditor.DragStartMouseX = x;
                _viewportEditor.DragStartMouseY = y;
                return;
            }

            var selected = _scene.Entities.Find(e => e.Id == _viewportEditor.SelectedEntityId);
            if (selected != null && _viewportEditor.ShowGizmos &&
                (_viewportEditor.ToolMode == ViewportToolMode.Translate || _viewportEditor.ToolMode == ViewportToolMode.Rotate))
            {
                var axis = ViewportInteraction.PickGizmoAxis(ray, selected.Position.ToVector3(), selected.Scale.ToVector3());
                if (axis != GizmoAxis.None)
                {
                    _viewportEditor.ActiveGizmoAxis = axis;
                    _viewportEditor.IsDragging = true;
                    _viewportEditor.DragStartPosition = selected.Position.ToVector3();
                    _viewportEditor.DragStartRotation = selected.Rotation.ToVector3();
                    _viewportEditor.DragStartMouseX = x;
                    _viewportEditor.DragStartMouseY = y;
                    return;
                }
            }

            var picked = ViewportInteraction.PickEntity(_scene, _renderEngine.SceneRenderer, ray);
            if (picked.HasValue)
            {
                _viewportEditor.SelectedEntityId = picked.Value;
                ViewportEntitySelected?.Invoke(picked.Value);
                SyncSceneToRenderer();
            }
        }

        /// <summary>Drag gizmo or orbit camera.</summary>
        public void HandleViewportPointerMove(float x, float y, int width, int height)
        {
            if (_renderEngine == null || !_renderInitialized) return;

            if (_viewportEditor.IsOrbitingCamera)
            {
                float dx = x - _viewportEditor.DragStartMouseX;
                float dy = y - _viewportEditor.DragStartMouseY;
                _renderEngine.ApplyCameraDelta(dx * 0.15f, -dy * 0.15f, Vector3.Zero);
                _viewportEditor.DragStartMouseX = x;
                _viewportEditor.DragStartMouseY = y;
                return;
            }

            if (!_viewportEditor.IsDragging) return;
            var entity = _scene.Entities.Find(e => e.Id == _viewportEditor.SelectedEntityId);
            if (entity == null) return;

            GetFramebufferSize(out int fbW, out int fbH, width, height);
            var view = Matrix4x4.CreateLookAt(_renderEngine.GetCamera().Position,
                _renderEngine.GetCamera().Position + _renderEngine.GetCamera().Front,
                _renderEngine.GetCamera().Up);
            var proj = Matrix4x4.CreatePerspectiveFieldOfView(
                MathHelperDeg2Rad(_renderEngine.GetCamera().Fov),
                (float)fbW / Math.Max(1, fbH), 0.1f, 100f);
            proj.M11 *= -1;

            if (_viewportEditor.ToolMode == ViewportToolMode.Rotate)
                ViewportInteraction.ApplyRotateDrag(_viewportEditor, entity, x, y);
            else
                ViewportInteraction.ApplyTranslateDrag(_viewportEditor, entity, x, y, view, proj, fbW, fbH);

            SyncSceneToRenderer();
        }

        /// <summary>Ends gizmo drag or camera orbit.</summary>
        public void HandleViewportPointerUp()
        {
            _viewportEditor.IsDragging = false;
            _viewportEditor.IsOrbitingCamera = false;
            _viewportEditor.ActiveGizmoAxis = GizmoAxis.None;
        }

        private static void GetFramebufferSize(out int fbW, out int fbH, int controlW, int controlH)
        {
            fbW = Math.Max(1, controlW);
            fbH = Math.Max(1, controlH);
        }

        private static float MathHelperDeg2Rad(float deg) => deg * MathF.PI / 180f;

        /// <summary>Updates an entity in the scene document and mirrors it to the renderer.</summary>
        public bool UpdateSceneEntity(Guid id, string name, Vector3 position, Vector3 scale)
        {
            InitializeModules();
            var entity = _scene.Entities.Find(e => e.Id == id);
            if (entity == null) return false;

            entity.Name = name;
            entity.Position = Vec3.From(position);
            entity.Scale = Vec3.From(scale);
            SyncSceneToRenderer();
            return true;
        }

        /// <summary>Pushes scene entities, lights, and camera hints to the render pipeline.</summary>
        public void SyncSceneToRenderer()
        {
            if (_renderEngine?.SceneRenderer == null) return;
            SceneRenderBridge.SyncDocument(_renderEngine, _scene, _viewportEditor, _logger);

            if (_renderInitialized)
            {
                var cam = _scene.Camera;
                _renderEngine!.SetCamera(
                    cam.Position.ToVector3(),
                    cam.Yaw,
                    cam.Pitch,
                    cam.Fov);
            }
        }

        private void ApplyEvolutionToScene()
        {
            if (_evolution?.GetBestGenome() == null) return;
            SceneRenderBridge.ApplyEvolutionVisual(
                _renderEngine, _scene, _evolutionGeneration, _bestFitness, _logger);
        }

        private void ApplySceneToSimulation(SceneDocument scene)
        {
            if (_sentience == null) return;

            foreach (var existing in _sentience.GetAllEntities().ToList())
                _sentience.RemoveEntity(existing.EntityId);

            var factory = new SentientEntityFactory(_sentience);
            foreach (var e in scene.Entities)
            {
                if (e.Type.Equals("Character", StringComparison.OrdinalIgnoreCase))
                {
                    var agent = factory.CreateNPC(e.Position.ToVector3(), e.BehaviorProfile ?? "patrol");
                    // keep scene id mapping loosely via name
                    e.Id = agent.EntityId;
                }
            }
        }

        private void EnsureLawCompiled(string? lawId, string? expression)
        {
            if (_lawCompiler == null || string.IsNullOrWhiteSpace(lawId)) return;
            var result = ActivateLaw(lawId, expression);
            if (!result.Success)
                _logger.Warn("Physics", $"Unable to activate law '{lawId}': {result.Message}");
        }

        private CompilationResult ActivateLaw(string lawId, string? expression)
        {
            if (_lawCompiler == null)
                return CompilationResult.Fail("Compiler not ready", new[] { "null compiler" });

            CompilationResult result;
            if (!string.IsNullOrWhiteSpace(expression))
                result = _lawCompiler.HotReload(lawId, expression!);
            else
                result = _lawCompiler.CompileFromLibrary(lawId);

            // Built-in library strings are human-readable and often fail the bytecode parser.
            // Install a stable numeric form so applicators can still run under that law id.
            if (!result.Success)
                result = _lawCompiler.Compile("T", lawId);

            if (result.Success)
                _activeLawId = lawId;

            return result;
        }

        private static PhysicsField CreateSeedField(int size)
        {
            var field = new PhysicsField(size, "runtime");
            float cx = size / 2f;
            for (int z = 0; z < size; z++)
            for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                float dx = x - cx, dy = y - cx, dz = z - cx;
                float r = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                field.Temperature[x, y, z] = 300f + 80f * MathF.Exp(-r * r / (size * size * 0.25f));
                field.Density[x, y, z] = 1.225f;
                field.Pressure[x, y, z] = 101325f;
            }
            return field;
        }

        private static float SampleAverageTemperature(PhysicsField field)
        {
            float sum = 0;
            int n = 0;
            int step = Math.Max(1, field.GridSize / 4);
            for (int z = 0; z < field.GridSize; z += step)
            for (int y = 0; y < field.GridSize; y += step)
            for (int x = 0; x < field.GridSize; x += step)
            {
                sum += field.Temperature[x, y, z];
                n++;
            }
            return n == 0 ? 0 : sum / n;
        }

        private static QualityPreset ParseQuality(string name) =>
            Enum.TryParse<QualityPreset>(name, true, out var p) ? p : QualityPreset.High;

        private void WireBehaviorLlmRouter()
        {
            BehaviorLlmContext.QueryAsync = async (prompt, entity, context, ct) =>
            {
                var messages = new List<ChatMessage>
                {
                    new() { Role = MessageRole.System, Content = $"You are an NPC behavior assistant for entity {entity.EntityId}." },
                    new() { Role = MessageRole.User, Content = prompt }
                };
                var promptContext = new PromptContext
                {
                    TaskType = LlmTaskType.BehaviorGeneration,
                    PreferredMode = LlmProviderMode.LocalOllama,
                    MaxLatencyMs = 30000
                };
                var response = await _llmRouter!.RouteChatAsync(messages, promptContext, ct).ConfigureAwait(false);
                return response?.Content ?? "fail";
            };

            BehaviorLlmContext.ResponseHandler = (_, _, response) =>
            {
                if (response.Contains("fail", StringComparison.OrdinalIgnoreCase))
                    return GDNN.Sentience.TaskStatus.Failure;
                if (response.Contains("wait", StringComparison.OrdinalIgnoreCase))
                    return GDNN.Sentience.TaskStatus.Running;
                return GDNN.Sentience.TaskStatus.Success;
            };
        }

        /// <summary>Releases evolution, LLM, quality, and render resources.</summary>
        public async ValueTask DisposeAsync()
        {
            CancelEvolution();
            _evolutionCts?.Dispose();
            if (_evolution != null) await _evolution.DisposeAsync();
            _llmRouter?.Dispose();
            _quality?.Dispose();
            _renderEngine?.Dispose();
            _logger.Info("EngineHost", "Disposed");
        }
    }
}
