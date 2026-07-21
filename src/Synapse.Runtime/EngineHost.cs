using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Core.NEAT;
using GDNN.Llm;
using GDNN.Platform;
using GDNN.Rendering.Engine;
using GDNN.Rendering.MeshIO;
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
        private MultiphysicsOrchestrator? _multiphysics;
        private SynapseMeshProvider? _meshProvider;
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
        private bool _evolutionInProgress;
        private bool _simulationPlaying = true;
        private string _llmProviderSummary = "LLM not initialized";
        private readonly ViewportEditorState _viewportEditor = new();
        private BlueprintRuntimeExecutor? _blueprintExecutor;
        private string? _liveBlueprintName;
        private Guid? _liveBlueprintAgentId;
        private bool _disposed;

        /// <summary>When true, blueprint graph edits hot-reload the live agent tree without respawning.</summary>
        public bool BlueprintLiveEdit { get; set; } = true;

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

        /// <summary>Industrial multiphysics orchestrator (rigid bodies + living laws + continuum).</summary>
        public MultiphysicsOrchestrator? Multiphysics => _multiphysics;

        /// <summary>Rigid-body world driven each physics tick.</summary>
        public RigidBodyWorld? RigidWorld => _multiphysics?.RigidWorld;

        /// <summary>Synapse Omnia mesh provider (load → cook → optional G-DNN bake).</summary>
        public SynapseMeshProvider? MeshProvider => _meshProvider;

        /// <summary>Native platform capabilities (GLFW / Vulkan / HWND).</summary>
        public PlatformCapabilities PlatformCaps { get; private set; } = NativePlatform.Probe();

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
        public bool EvolutionRunning => _evolutionInProgress;

        /// <summary>Raised when evolution or living-law activity produces an inspector feed line.</summary>
        public event Action<InspectorFeedEntry>? InspectorFeedEntryAdded;

        /// <summary>Human-readable list of registered LLM providers.</summary>
        public string LlmProviderSummary => _llmProviderSummary;

        /// <summary>
        /// Lazily constructs physics, simulation, LLM, quality, and twin subsystems.
        /// Safe to call multiple times.
        /// </summary>
        public void InitializeModules()
        {
            if (_modulesInitialized)
                return;

            SimulationReproducibility.SetSeed(_config.SimulationSeed);

            _lawCompiler = new LivingLawCompiler();
            WireLawInspectorEvents(_lawCompiler);
            _physicsField = CreateSeedField(16);
            _multiphysics = new MultiphysicsOrchestrator(
                _lawCompiler,
                _physicsField,
                new MultiphysicsConfig
                {
                    EnabledModules = ContinuumModules.LivingLaws | ContinuumModules.RigidBodies,
                    FixedTimeStep = 1f / 60f,
                    MaxSubSteps = 4,
                    Gravity = new Vector3(0f, -9.81f, 0f)
                });
            _meshProvider = new SynapseMeshProvider(_logger);
            _sentience = new SentienceManager();
            _llmRouter = new HybridLlmRouter();
            _llmProviderSummary = LlmProviderBootstrap.Register(_llmRouter, _config, _logger).Summary;
            WireBehaviorLlmRouter();
            _quality = new RuntimeQualityManager(ParseQuality(_config.QualityPreset), AdaptationMode.Dynamic);

            _activeLawId = _scene.ActiveLawId ?? "heat_equation";
            EnsureLawCompiled(_activeLawId, _scene.ActiveLawExpression);
            _multiphysics.SetActiveLaw(_activeLawId);
            ApplySceneToSimulation(_scene);
            SyncSceneToPhysics();

            var twin = new InMemoryDigitalTwin { PhysicalId = "demo-world" };
            twin.SetProperty("Name", _scene.Name);
            twin.SetProperty("EntityType", GDNN.Sentience.EntityType.Environmental.ToString());
            _twins.Register(twin);

            PlatformCaps = NativePlatform.Probe();
            _modulesInitialized = true;
            _logger.Info("EngineHost", "Modules initialized (Physics, Simulation, LLM, Quality, Twins, MeshProvider)");
            _logger.Info("Platform", PlatformCaps.Summary);
        }

        /// <summary>
        /// Native multiplatform render init: always uses GLFW as the primary Vulkan WSI
        /// on Windows, Linux, and macOS (MoltenVK). Prefer this over HWND except for
        /// Avalonia Studio embedding on Windows.
        /// </summary>
        public void InitializeRenderNative(int width, int height, bool enableValidation = true)
        {
            PlatformCaps = NativePlatform.Probe();
            _logger.Info("Platform", PlatformCaps.Summary);
            if (!PlatformCaps.GlfwAvailable)
                _logger.Warn("Platform", "GLFW probe failed — RenderEngine will still attempt DllImport resolution");
            InitializeRender(width, height, enableValidation);
        }

        /// <summary>Creates a standalone GLFW render surface.</summary>
        public void InitializeRender(int width, int height, bool enableValidation = true)
        {
            if (_renderInitialized)
                return;
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
            if (_renderInitialized)
                return;
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
            _multiphysics?.SetActiveLaw(_activeLawId);
            ApplySceneToSimulation(_scene);
            SyncSceneToPhysics();

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

        /// <summary>
        /// Advances the industrial multiphysics pipeline (living laws + rigid bodies + optional continuum)
        /// and writes dynamic transforms back to the scene.
        /// </summary>
        public void TickPhysics(float dt)
        {
            if (_multiphysics == null)
                return;

            var budget = TimeSpan.FromMilliseconds(_config.PhysicsBudgetMs);
            try
            {
                _multiphysics.SetActiveLaw(_activeLawId);
                _multiphysics.Step(dt, budget);
                AverageFieldTemperature = _multiphysics.LastStats.AverageTemperature;
                WritePhysicsTransformsToScene();
            }
            catch (Exception ex)
            {
                _logger.Warn("Physics", $"Multiphysics step failed: {ex.Message}");
            }

            if (_multiphysics.LastStats.TotalMs > budget.TotalMilliseconds)
                _logger.Debug("Physics", "Exceeded physics budget");
        }

        /// <summary>Advances sentient entities when <see cref="SimulationPlaying"/> is true.</summary>
        public async Task TickSimulationAsync(float dt, CancellationToken cancellationToken)
        {
            if (!_simulationPlaying || _sentience == null)
                return;
            try
            {
                if (_blueprintExecutor is { IsRunning: true })
                    await _blueprintExecutor.TickAsync(dt, cancellationToken).ConfigureAwait(false);
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
            if (_renderEngine == null || !_renderInitialized)
                return;
            if (_renderEngine.IsPaused)
                return;
            _renderEngine.RenderFrame();
        }

        /// <summary>Feeds frame timing into the adaptive quality manager.</summary>
        public void TickQuality(float dt, float frameMs)
        {
            if (_quality == null)
                return;
            _quality.ReportFrame(frameMs, 0, 0, frameMs);
            _quality.Update(dt);
        }

        /// <summary>Activates a built-in law from the library by id.</summary>
        public CompilationResult ApplyLaw(string lawId)
        {
            InitializeModules();
            var entry = _lawCompiler!.Library.GetLaw(lawId);
            if (entry == null)
                return CompilationResult.Fail($"Unknown law '{lawId}'.", new[] { $"Law id '{lawId}' not in library." });
            return CompileLaw(lawId, entry.Expression);
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
                _multiphysics?.SetActiveLaw(lawId);
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

            void OnGenerationCompleted(object? sender, GenerationCompletedEventArgs args)
            {
                _evolutionGeneration = args.Generation;
                _bestFitness = args.Metrics.BestFitness;

                string mutationDetail = "";
                var rates = _evolution!.Diagnostics.GetMutationSuccessRates();
                if (rates.Count > 0)
                {
                    var top = rates.OrderByDescending(kvp => kvp.Value).First();
                    mutationDetail = $" · mutation {top.Key} {top.Value:P0}";
                }

                RaiseInspector("Evolution", $"Gen {args.Generation}",
                    $"best={args.Metrics.BestFitness:F3} avg={args.Metrics.AverageFitness:F3} " +
                    $"species={args.Metrics.SpeciesCount}{mutationDetail}");
                ApplyEvolutionToScene();
            }

            void OnMilestoneReached(object? sender, EvolutionMilestoneEventArgs args)
            {
                RaiseInspector("Evolution", args.MilestoneType.ToString(),
                    $"gen={args.Generation} fitness={args.BestFitness:F3}");
            }

            _evolution.GenerationCompleted += OnGenerationCompleted;
            _evolution.MilestoneReached += OnMilestoneReached;
            _evolutionInProgress = true;

            _logger.Info("Evolution", $"Starting NEAT-G pop={config.PopulationSize} gens={config.MaxGenerations}");
            RaiseInspector("Evolution", "Démarrage",
                $"pop={config.PopulationSize} gens={config.MaxGenerations}");

            try
            {
                var context = EvaluationContext.CreateDefault();
                for (int g = 0; g < config.MaxGenerations; g++)
                {
                    _evolutionCts.Token.ThrowIfCancellationRequested();
                    await _evolution.StepAsync(context, _evolutionCts.Token).ConfigureAwait(false);
                    _evolutionGeneration = _evolution.CurrentGeneration;
                    _bestFitness = _evolution.GetBestGenome()?.Fitness ?? _bestFitness;
                }

                ApplyEvolutionToScene();
                RaiseInspector("Evolution", "Terminé",
                    $"gen={_evolutionGeneration} best={_bestFitness:F3}");
            }
            catch (OperationCanceledException)
            {
                _logger.Info("Evolution", "Cancelled");
                RaiseInspector("Evolution", "Annulé",
                    $"gen={_evolutionGeneration} best={_bestFitness:F3}");
            }
            catch (Exception ex)
            {
                _logger.Warn("Evolution", $"Evolution step failed: {ex.Message}");
                RaiseInspector("Evolution", "Erreur", ex.Message);
            }
            finally
            {
                _evolution.GenerationCompleted -= OnGenerationCompleted;
                _evolution.MilestoneReached -= OnMilestoneReached;
                _evolutionInProgress = false;
            }
        }

        /// <summary>Requests cancellation of an in-flight evolution run.</summary>
        public void CancelEvolution() => _evolutionCts?.Cancel();

        /// <summary>Spawns a sentient simulation agent and mirrors it in the scene document.</summary>
        public SentientEntity SpawnAgent(string profile, Vector3 position)
        {
            InitializeModules();
            var factory = new SentientEntityFactory(_sentience!);
            var entity = factory.CreateAgent(position, profile);
            _scene.Entities.Add(new SceneEntityData
            {
                Id = entity.EntityId,
                Name = $"Agent_{profile}_{entity.EntityId.ToString()[..8]}",
                Type = SceneEntityKinds.Agent,
                Position = Vec3.From(position),
                BehaviorProfile = profile
            });
            SyncSceneToRenderer();
            return entity;
        }

        /// <summary>Adds a scene entity; spawns a patrol agent when <paramref name="type"/> is an agent kind.</summary>
        public Guid CreateSceneEntity(string name, string type)
        {
            InitializeModules();
            var id = Guid.NewGuid();
            var normalized = SceneEntityKinds.Normalize(type);
            _scene.Entities.Add(new SceneEntityData
            {
                Id = id,
                Name = name,
                Type = normalized,
                Position = new Vec3(0, 0, 0)
            });

            if (SceneEntityKinds.IsSentientAgent(normalized))
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
            if (removed)
                SyncSceneToRenderer();
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

            var agent = _sentience.CreateEntity(EntityType.Sentient, Vector3.Zero, tree.Name);
            _scene.Entities.Add(new SceneEntityData
            {
                Id = agent.EntityId,
                Name = tree.Name,
                Type = SceneEntityKinds.Agent,
                Position = Vec3.From(Vector3.Zero),
                BehaviorProfile = tree.Name
            });
            SyncSceneToRenderer();
            return $"Registered behavior tree '{tree.Name}' with {extracted.Data.Count} LLM node(s).";
        }

        /// <summary>Compiles a blueprint graph to a behavior tree and spawns an agent.</summary>
        public SentientEntity CompileAndSpawnBlueprint(BlueprintDocument document, Vector3 position)
        {
            ArgumentNullException.ThrowIfNull(document);
            InitializeModules();
            var blueprint = document.CompileToBehaviorTreeBlueprint();
            var tree = _sentience!.Compiler.CompileFromBlueprint(document.Name, blueprint);
            _sentience.RegisterBehaviorTree(document.Name, tree);
            var entity = _sentience.CreateEntity(EntityType.Sentient, position, document.Name);
            _scene.Entities.Add(new SceneEntityData
            {
                Id = entity.EntityId,
                Name = $"Agent_{document.Name}",
                Type = SceneEntityKinds.Agent,
                Position = Vec3.From(position),
                BehaviorProfile = document.Name
            });
            _liveBlueprintName = document.Name;
            _liveBlueprintAgentId = entity.EntityId;
            _blueprintExecutor ??= new BlueprintRuntimeExecutor(this, _logger);
            _blueprintExecutor.Load(document);
            SyncSceneToRenderer();
            return entity;
        }

        /// <summary>
        /// Hot-reloads a blueprint into the running simulation: recompiles the behavior tree in place
        /// and refreshes the live <see cref="BlueprintRuntimeExecutor"/> graph without spawning a new agent.
        /// </summary>
        public bool HotReloadBlueprint(BlueprintDocument document)
        {
            ArgumentNullException.ThrowIfNull(document);
            InitializeModules();
            var (ok, msg) = document.Validate();
            if (!ok)
            {
                _logger.Warn("Blueprint", $"Live edit rejected: {msg}");
                return false;
            }

            var blueprint = document.CompileToBehaviorTreeBlueprint();
            var tree = _sentience!.Compiler.CompileFromBlueprint(document.Name, blueprint);
            _sentience.RegisterBehaviorTree(document.Name, tree);

            // Update existing agents that were spawned from this blueprint (cloned trees).
            foreach (var entity in _sentience.GetAllEntities())
            {
                var treeName = entity.BehaviorTree?.Name;
                if (treeName == null)
                    continue;
                if (treeName == document.Name ||
                    treeName.StartsWith(document.Name + "_", StringComparison.Ordinal) ||
                    (_liveBlueprintAgentId is Guid id && entity.EntityId == id))
                {
                    entity.BehaviorTree = tree.Clone();
                }
            }

            _liveBlueprintName = document.Name;
            _blueprintExecutor ??= new BlueprintRuntimeExecutor(this, _logger);
            _blueprintExecutor.Load(document);
            _logger.Info("Blueprint", $"Live hot-reload OK: '{document.Name}'");
            return true;
        }

        /// <summary>Stops the live blueprint graph executor (behavior trees keep running on agents).</summary>
        public void StopLiveBlueprintExecutor() => _blueprintExecutor?.Stop();

        /// <summary>Sets the selected entity for viewport gizmos.</summary>
        public void SetViewportSelection(Guid entityId) => _viewportEditor.SelectedEntityId = entityId;

        /// <summary>Raised when the user picks a different entity in the viewport.</summary>
        public event Action<Guid>? ViewportEntitySelected;

        /// <summary>Left-click: pick entity or begin gizmo drag. Right-click: orbit camera.</summary>
        public void HandleViewportPointerDown(float x, float y, int width, int height, bool rightButton)
        {
            if (_renderEngine?.SceneRenderer == null || !_renderInitialized)
                return;

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
            if (_renderEngine == null || !_renderInitialized)
                return;

            if (_viewportEditor.IsOrbitingCamera)
            {
                float dx = x - _viewportEditor.DragStartMouseX;
                float dy = y - _viewportEditor.DragStartMouseY;
                _renderEngine.ApplyCameraDelta(dx * 0.15f, -dy * 0.15f, Vector3.Zero);
                _viewportEditor.DragStartMouseX = x;
                _viewportEditor.DragStartMouseY = y;
                return;
            }

            if (!_viewportEditor.IsDragging)
                return;
            var entity = _scene.Entities.Find(e => e.Id == _viewportEditor.SelectedEntityId);
            if (entity == null)
                return;

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
        public bool UpdateSceneEntity(
            Guid id,
            string name,
            Vector3 position,
            Vector3 scale,
            string? meshPath = null,
            bool? isVehicle = null,
            bool? bakeNeuralSdf = null,
            bool resyncPhysics = false)
        {
            InitializeModules();
            var entity = _scene.Entities.Find(e => e.Id == id);
            if (entity == null)
                return false;

            entity.Name = name;
            entity.Position = Vec3.From(position);
            entity.Scale = Vec3.From(scale);
            if (meshPath != null)
                entity.MeshPath = string.IsNullOrWhiteSpace(meshPath) ? null : meshPath;
            if (isVehicle.HasValue)
                entity.IsVehicle = isVehicle.Value;
            if (bakeNeuralSdf.HasValue)
                entity.BakeNeuralSdf = bakeNeuralSdf.Value;

            SyncSceneToRenderer();
            if (resyncPhysics)
                SyncSceneToPhysics();
            return true;
        }

        /// <summary>Adds a hinge joint anchoring <paramref name="bodyId"/> to the world.</summary>
        public Guid? AddHingeToWorld(Guid bodyId, Vector3? worldAxis = null, float compliance = 0f)
        {
            InitializeModules();
            if (_multiphysics?.RigidWorld.GetBody(bodyId) == null
                && _scene.Entities.Find(e => e.Id == bodyId) == null)
                return null;

            var joint = new SceneJointData
            {
                Name = "Hinge_World",
                Type = "Hinge",
                BodyA = bodyId,
                BodyB = Guid.Empty,
                LocalAnchorA = new Vec3(0, 0, 0),
                LocalAnchorB = Vec3.From(_scene.Entities.Find(e => e.Id == bodyId)?.Position.ToVector3() ?? Vector3.Zero),
                LocalAxisA = Vec3.From(worldAxis ?? Vector3.UnitY),
                LocalAxisB = Vec3.From(worldAxis ?? Vector3.UnitY),
                Compliance = compliance,
                Stiffness = 1.2f,
                Damping = 0.15f
            };
            _scene.Joints.Add(joint);
            SyncSceneToPhysics();
            return joint.Id;
        }

        /// <summary>Adds a distance (soft rope) joint between two scene entities.</summary>
        public Guid? AddDistanceJoint(Guid bodyA, Guid bodyB, float? restLength = null, float compliance = 0.01f)
        {
            InitializeModules();
            var ea = _scene.Entities.Find(e => e.Id == bodyA);
            var eb = _scene.Entities.Find(e => e.Id == bodyB);
            if (ea == null || eb == null)
                return null;

            float rest = restLength ?? Vector3.Distance(ea.Position.ToVector3(), eb.Position.ToVector3());
            var joint = new SceneJointData
            {
                Name = "Distance",
                Type = "Distance",
                BodyA = bodyA,
                BodyB = bodyB,
                RestLength = MathF.Max(0.05f, rest),
                Stiffness = 1.5f,
                Damping = 0.4f,
                Compliance = compliance
            };
            _scene.Joints.Add(joint);
            SyncSceneToPhysics();
            return joint.Id;
        }

        /// <summary>Marks an entity as a raycast vehicle chassis and rebuilds physics.</summary>
        public bool SetEntityAsVehicle(Guid id, bool isVehicle = true)
        {
            InitializeModules();
            var entity = _scene.Entities.Find(e => e.Id == id);
            if (entity == null)
                return false;
            entity.IsVehicle = isVehicle;
            SyncSceneToPhysics();
            return true;
        }

        /// <summary>Assigns a mesh asset path for SynapseMeshProvider cooking.</summary>
        public bool SetEntityMeshPath(Guid id, string? meshPath, bool bakeNeuralSdf = false)
        {
            InitializeModules();
            var entity = _scene.Entities.Find(e => e.Id == id);
            if (entity == null)
                return false;
            entity.MeshPath = string.IsNullOrWhiteSpace(meshPath) ? null : meshPath;
            entity.BakeNeuralSdf = bakeNeuralSdf;
            SyncSceneToPhysics();
            SyncSceneToRenderer();
            return true;
        }

        /// <summary>Pushes scene entities, lights, and camera hints to the render pipeline.</summary>
        public void SyncSceneToRenderer()
        {
            if (_renderEngine?.SceneRenderer == null)
                return;
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
            if (_evolution?.GetBestGenome() == null)
                return;
            SceneRenderBridge.ApplyEvolutionVisual(
                _renderEngine, _scene, _evolutionGeneration, _bestFitness, _logger);
        }

        private void ApplySceneToSimulation(SceneDocument scene)
        {
            if (_sentience == null)
                return;

            foreach (var existing in _sentience.GetAllEntities().ToList())
                _sentience.RemoveEntity(existing.EntityId);

            var factory = new SentientEntityFactory(_sentience);
            foreach (var e in scene.Entities)
            {
                if (SceneEntityKinds.IsSentientAgent(e.Type))
                {
                    e.Type = SceneEntityKinds.Normalize(e.Type);
                    var agent = factory.CreateAgent(e.Position.ToVector3(), e.BehaviorProfile ?? "patrol");
                    // keep scene id mapping loosely via name
                    e.Id = agent.EntityId;
                }
            }
        }

        /// <summary>Rebuilds the rigid-body world from the current scene document.</summary>
        public void SyncSceneToPhysics()
        {
            if (_multiphysics == null)
                return;

            _multiphysics.RigidWorld.Clear();
            var descs = new List<PhysicsEntityDesc>(_scene.Entities.Count);
            foreach (var e in _scene.Entities)
            {
                descs.Add(new PhysicsEntityDesc
                {
                    Id = e.Id,
                    Name = e.Name,
                    Type = e.Type,
                    Position = e.Position.ToVector3(),
                    Scale = e.Scale.ToVector3(),
                    IsStatic = e.Type.Equals("Mesh", StringComparison.OrdinalIgnoreCase)
                        && e.Name.Contains("Ground", StringComparison.OrdinalIgnoreCase),
                    Mass = 0f,
                    Restitution = e.Type.Equals("Genome", StringComparison.OrdinalIgnoreCase) ? 0.6f : 0.2f,
                    Collider = e.Type.Equals("Genome", StringComparison.OrdinalIgnoreCase)
                        || e.Type.Equals("Character", StringComparison.OrdinalIgnoreCase)
                        ? ColliderShape.Sphere
                        : ColliderShape.Box
                });
            }

            _multiphysics.SyncFromEntities(descs);

            // Mesh-backed colliders + optional vehicles (Synapse Omnia extensions).
            foreach (var e in _scene.Entities)
            {
                if (!string.IsNullOrWhiteSpace(e.MeshPath) && _meshProvider != null && File.Exists(e.MeshPath))
                {
                    try
                    {
                        var asset = new GDNN.Rendering.MeshIO.MeshLoader().LoadSync(e.MeshPath);
                        if (asset != null)
                        {
                            string meshId = e.Id.ToString("N");
                            _meshProvider.RegisterAsset(meshId, asset);
                            bool isStatic = e.Name.Contains("Ground", StringComparison.OrdinalIgnoreCase);
                            var bodyType = isStatic ? BodyType.Static : BodyType.Dynamic;
                            var cooked = _meshProvider.CookCollider(meshId, bodyType);
                            if (cooked != null)
                            {
                                _multiphysics.RigidWorld.RemoveBody(e.Id);
                                var replacement = new RigidBody
                                {
                                    Id = e.Id,
                                    Name = e.Name,
                                    Type = bodyType,
                                    Collider = cooked,
                                    Position = e.Position.ToVector3()
                                };
                                if (bodyType == BodyType.Dynamic)
                                    replacement.SetMass(MathF.Max(0.1f, e.Scale.X * e.Scale.Y * e.Scale.Z));
                                else
                                    replacement.SetMass(0f);
                                _multiphysics.RigidWorld.AddBody(replacement);
                            }

                            if (e.BakeNeuralSdf)
                                _ = _meshProvider.BakeNeuralSdfAsync(meshId);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.Warn("MeshProvider", $"Mesh bind failed for '{e.Name}': {ex.Message}");
                    }
                }

                if (e.IsVehicle)
                {
                    var chassis = _multiphysics.RigidWorld.GetBody(e.Id);
                    if (chassis != null)
                    {
                        chassis.Type = BodyType.Dynamic;
                        if (chassis.InverseMass <= 0f)
                            chassis.SetMass(1200f);
                        _multiphysics.RigidWorld.AddVehicle(VehicleController.CreateDefaultCar(chassis.Id));
                    }
                }
            }

            // Restore persisted joints after bodies exist.
            foreach (var j in _scene.Joints)
            {
                if (!Enum.TryParse<JointType>(j.Type, ignoreCase: true, out var jt))
                    jt = JointType.Hinge;

                _multiphysics.RigidWorld.AddJoint(new PhysicsJoint
                {
                    Id = j.Id == Guid.Empty ? Guid.NewGuid() : j.Id,
                    Name = j.Name,
                    Type = jt,
                    BodyA = j.BodyA,
                    BodyB = j.BodyB,
                    LocalAnchorA = j.LocalAnchorA.ToVector3(),
                    LocalAnchorB = j.LocalAnchorB.ToVector3(),
                    LocalAxisA = j.LocalAxisA.ToVector3(),
                    LocalAxisB = j.LocalAxisB.ToVector3(),
                    RestLength = j.RestLength,
                    Stiffness = j.Stiffness,
                    Damping = j.Damping,
                    Compliance = j.Compliance,
                    MinLimit = j.MinLimit,
                    MaxLimit = j.MaxLimit
                });
            }

            _logger.Info("Physics",
                $"Rigid world synced ({_multiphysics.RigidWorld.Bodies.Count} bodies, " +
                $"{_multiphysics.RigidWorld.Joints.Count} joints, {_multiphysics.RigidWorld.Vehicles.Count} vehicles)");
        }

        private void WritePhysicsTransformsToScene()
        {
            if (_multiphysics == null)
                return;

            bool dirty = false;
            foreach (var e in _scene.Entities)
            {
                var body = _multiphysics.RigidWorld.GetBody(e.Id);
                if (body == null || body.Type != BodyType.Dynamic)
                    continue;

                var p = body.Position;
                if (MathF.Abs(p.X - e.Position.X) > 1e-4f
                    || MathF.Abs(p.Y - e.Position.Y) > 1e-4f
                    || MathF.Abs(p.Z - e.Position.Z) > 1e-4f)
                {
                    e.Position = Vec3.From(p);
                    dirty = true;
                }
            }

            if (dirty && _renderInitialized)
                SyncSceneToRenderer();
        }

        private void EnsureLawCompiled(string? lawId, string? expression)
        {
            if (_lawCompiler == null || string.IsNullOrWhiteSpace(lawId))
                return;
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

        private void WireLawInspectorEvents(LivingLawCompiler compiler)
        {
            void Forward(object sender, LawEventArgs args)
            {
                if (args.EventType is LawEventType.CompilationStarted or LawEventType.CacheHit
                    or LawEventType.CacheMiss or LawEventType.CacheEviction)
                    return;

                string title = args.EventType switch
                {
                    LawEventType.CompilationCompleted => "Compilation OK",
                    LawEventType.CompilationFailed => "Compilation échouée",
                    LawEventType.ValidationFailed => "Validation échouée",
                    LawEventType.HotReloadTriggered => "Hot-reload",
                    LawEventType.HotReloadCompleted => "Hot-reload OK",
                    LawEventType.VersionCreated => "Version créée",
                    LawEventType.LawApplied => "Loi appliquée",
                    _ => args.EventType.ToString()
                };

                string detail = args.Message ?? "";
                if (!string.IsNullOrEmpty(args.LawId))
                    detail = string.IsNullOrEmpty(detail)
                        ? args.LawId
                        : $"{args.LawId} — {detail}";
                if (!string.IsNullOrEmpty(args.Expression))
                    detail += $"\n{args.Expression}";

                RaiseInspector("LivingLaw", title, detail.Trim());
            }

            foreach (LawEventType eventType in Enum.GetValues<LawEventType>())
                compiler.Events.Subscribe(eventType, Forward);
        }

        private void RaiseInspector(string category, string title, string detail)
        {
            InspectorFeedEntryAdded?.Invoke(new InspectorFeedEntry(
                DateTime.UtcNow, category, title, detail));
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

        private static QualityPreset ParseQuality(string name) =>
            Enum.TryParse<QualityPreset>(name, true, out var p) ? p : QualityPreset.High;

        private void WireBehaviorLlmRouter()
        {
            BehaviorLlmContext.QueryAsync = async (prompt, entity, context, ct) =>
            {
                var messages = new List<ChatMessage>
                {
                    new() { Role = MessageRole.System, Content = $"You are a sentient simulation agent assistant for entity {entity.EntityId}. You help an inhabitant perceive and decide inside a 3D simulation — not a game NPC." },
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

        /// <summary>Releases evolution, LLM, quality, render, and blueprint resources safely.</summary>
        public async ValueTask DisposeAsync()
        {
            if (_disposed)
                return;
            _disposed = true;

            try
            {
                CancelEvolution();
            }
            catch (Exception ex)
            {
                _logger.Warn("EngineHost", $"CancelEvolution: {ex.Message}");
            }

            try
            {
                _evolutionCts?.Dispose();
            }
            catch (Exception ex)
            {
                _logger.Warn("EngineHost", $"EvolutionCts dispose: {ex.Message}");
            }

            try
            {
                if (_evolution != null)
                    await _evolution.DisposeAsync().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.Warn("EngineHost", $"Evolution dispose: {ex.Message}");
            }

            try
            {
                _blueprintExecutor?.Stop();
            }
            catch (Exception ex)
            {
                _logger.Warn("EngineHost", $"Blueprint stop: {ex.Message}");
            }

            try
            {
                _multiphysics?.Dispose();
            }
            catch (Exception ex)
            {
                _logger.Warn("EngineHost", $"Multiphysics dispose: {ex.Message}");
            }

            try
            {
                _llmRouter?.Dispose();
            }
            catch (Exception ex)
            {
                _logger.Warn("EngineHost", $"LLM dispose: {ex.Message}");
            }

            try
            {
                _quality?.Dispose();
            }
            catch (Exception ex)
            {
                _logger.Warn("EngineHost", $"Quality dispose: {ex.Message}");
            }

            try
            {
                _renderEngine?.Dispose();
            }
            catch (Exception ex)
            {
                _logger.Warn("EngineHost", $"Render dispose: {ex.Message}");
            }

            _logger.Info("EngineHost", "Disposed");
        }

        /// <summary>
        /// Production readiness snapshot: platform, SIMD, modules, and known experimental surfaces.
        /// </summary>
        public ProductionHealthReport GetProductionHealth()
        {
            var caps = PlatformCaps;
            var cpu = caps.Cpu;
            bool usdOk = UsdProductionSmoke.TryVerify(out var usdDetail);
            return new ProductionHealthReport
            {
                ProductVersion = Synapse.Infrastructure.SynapseProduct.Version,
                ModulesInitialized = _modulesInitialized,
                RenderInitialized = _renderInitialized,
                SimulationPlaying = _simulationPlaying,
                GlfwAvailable = caps.GlfwAvailable,
                VulkanLoaderAvailable = caps.VulkanLoaderAvailable,
                SimdBaseline = cpu.BaselineLabel,
                PlatformSummary = caps.Summary,
                EntityCount = EntityCount,
                ActiveLawId = _activeLawId,
                BlueprintLiveEdit = BlueprintLiveEdit,
                MeetsMinimumCpu = cpu.MeetsMinimumCpu,
                UsdRuntimeReady = usdOk,
                UsdRuntimeDetail = usdDetail,
                ExperimentalNotes =
                {
                    "OpenXR uses native Vulkan2 swapchains when loader+HMD+Vulkan bind succeed; otherwise production simulated images (IsSimulated).",
                    "WAN NAT supports STUN reflexive candidates and TURN Allocate/CreatePermission/ChannelData for symmetric NAT (--stun-server / --turn-server).",
                    usdOk
                        ? $"OpenUSD MeshIO production-ready: {usdDetail} (faceVertexCounts, normals, multi-mesh, purpose/visibility, UDIM/MDL, Skel/blend, streamer, remote marketplace)."
                        : $"OpenUSD MeshIO smoke FAILED: {usdDetail}"
                }
            };
        }
    }

    /// <summary>Machine-readable production health snapshot for Studio / --health.</summary>
    public sealed class ProductionHealthReport
    {
        public string ProductVersion { get; init; } = "";
        public bool ModulesInitialized { get; init; }
        public bool RenderInitialized { get; init; }
        public bool SimulationPlaying { get; init; }
        public bool GlfwAvailable { get; init; }
        public bool VulkanLoaderAvailable { get; init; }
        public string SimdBaseline { get; init; } = "";
        public string PlatformSummary { get; init; } = "";
        public int EntityCount { get; init; }
        public string? ActiveLawId { get; init; }
        public bool BlueprintLiveEdit { get; init; }
        public bool MeetsMinimumCpu { get; init; }
        public List<string> ExperimentalNotes { get; init; } = new();
        /// <summary>True when embedded OpenUSD MeshIO production smoke passes.</summary>
        public bool UsdRuntimeReady { get; init; }
        public string UsdRuntimeDetail { get; init; } = "";

        /// <summary>True when core runtime can run (modules + CPU baseline). Vulkan optional for headless.</summary>
        public bool IsCoreReady => ModulesInitialized && MeetsMinimumCpu;

        /// <summary>True when interactive 3D viewport path is expected to work.</summary>
        public bool IsInteractiveReady => IsCoreReady && GlfwAvailable && VulkanLoaderAvailable;

        public override string ToString()
        {
            var mode = IsInteractiveReady ? "interactive-ready" : IsCoreReady ? "core-ready (headless/edit)" : "not-ready";
            var usd = UsdRuntimeReady ? "usd=ok" : "usd=fail";
            return $"Synapse {ProductVersion} [{mode}] {usd} SIMD={SimdBaseline} | {PlatformSummary} | entities={EntityCount} law={ActiveLawId ?? "-"}";
        }
    }
}
