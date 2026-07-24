using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using GDNN.Animation;
using GDNN.Core.DataStructures;
using GDNN.Core.Mathematics;
using GDNN.Core.NeuralNetwork;
using GDNN.Evaluation;
using GDNN.GPU;
using GDNN.Memory;
using GDNN.Polygonization;
using GDNN.Rendering.ArtPipeline;
using GDNN.Rendering.Compute;
using GDNN.Rendering.Geometry;
using GDNN.Rendering.MeshIO;
using GDNN.Rendering.Particles;
using GDNN.Rendering.Shadows;
using GDNN.Rendering.VirtualTextures;
using GDNN.Rendering.World;
using GDNN.RHI.Vulkan;
using GDNN.SIMD;
using GDNN.Streaming;
using GDNN.Threading;
using GDNN.Utilities;
using Synapse.Infrastructure.Logging;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>
    /// Owns and ticks Rendering + <b>all G-DNN</b> subsystems that feed the live FrameGraph.
    /// Heavy neural polygonization is lazy-built once with a tiny default SDF network.
    /// </summary>
    public sealed partial class RenderingAlgorithmHub : IDisposable
    {
        private readonly VulkanRhiDevice _rhi;
        private bool _disposed;
        private float _lastTime;
        private int _gdnnTick;
        private string? _meshletPagePath;
        private bool _gpuShaderResidencyDone;
        private bool _gdnnCoverageInitDone;
        private RasterTarget? _lastRasterTarget;
        private IReadOnlyList<NeuralMeshlet>? _lastMeshlets;
        private NeuralPolygonMesh? _lastRasterMesh;
        private int _lastCullWidth;
        private int _lastCullHeight;
        private NeuralPolygonMesh? _pendingPresentMesh;
        private long _presentMeshVersion = -1;
        private int _geometryMeshId = -1;
        private bool _gpuSdfInitLogged;
        private bool _gpuMeshletInitLogged;
        private string? _polyCacheDir;
        private string? _assetStreamRoot;
        private int _animClipIndex = -1;
        private float _animTime;

        // ── Non-G-DNN rendering algorithms ──────────────────────────────────
        public VirtualShadowMap VirtualShadows { get; }
        public ParticleSystem Particles { get; }
        public VirtualTextureSystem VirtualTextures { get; }
        public WorldPartitionSystem WorldPartition { get; }
        public ComputeDispatcher Compute { get; }
        public GeometryRenderer Geometry { get; }
        public MegascansBridge Megascans { get; }

        // ── G-DNN core networks ─────────────────────────────────────────────
        public HashEncodedDeepMLP DefaultSdfNetwork { get; }
        public DeepMicroMLP DeepMicroNetwork { get; }
        public QuantizedDeepMLP QuantizedNetwork { get; }
        public AABB DefaultBounds { get; }

        // ── G-DNN polygonization / meshlets ─────────────────────────────────
        public NeuralGeometryPipeline? NeuralGeometry { get; private set; }
        public NeuralPolygonizer Polygonizer { get; }
        public NeuralMeshletBuilder MeshletBuilder { get; }
        public SoftwareRasterizer SoftwareRasterizer { get; }
        public MeshletStreamer? MeshletStreamer { get; private set; }

        // ── G-DNN evaluation ────────────────────────────────────────────────
        public NeuralLodSelector NeuralLod { get; }
        public SceneEvaluator SceneEvaluator { get; }
        public SurfaceEvaluator SurfaceEvaluator => SceneEvaluator.SurfaceEvaluator;
        public RayMarcher RayMarcher => SceneEvaluator.RayMarcher;
        public HierarchicalSdfCache SdfCache { get; }
        public GradientCalculator Gradients { get; }
        public MicroMLP MicroNetwork { get; }
        public StochasticSphereTracer SphereTracer { get; }
        public WaveOptimizedBatchEvaluator WaveBatch { get; }
        public Profiler GdnnProfiler { get; }

        // ── G-DNN streaming / jobs / memory / animation / GPU residency ─────
        public AssetStreamer AssetStreamer { get; }
        public AsyncPipeline AsyncPipeline { get; }
        public JobSystem Jobs { get; }
        public ParallelEvaluator ParallelEval { get; }
        public StackAllocator StackMemory { get; }
        public ZeroCopyBuffer ZeroCopyScratch { get; }
        public AnimationBlender Animation { get; }
        public Octree<int> SpatialOctree { get; }
        public LooseOctree<int> LooseSpatial { get; }
        public SpatialHash<int> SpatialHash { get; }
        public CompressionUtils Compression { get; }
        public MemoryTracker MemoryTrack { get; }
        public WarpSpace Warp { get; }
        public Skeleton AnimSkeleton { get; }
        public GDNN.Animation.AnimationClip IdleClip { get; private set; } = null!;
        public SkinningWeights SkinWeights { get; private set; } = null!;
        public HyperNetwork HyperNet { get; }
        public MicroMLP? HyperGeneratedMlp { get; private set; }
        public NeuralAsset? StreamedNeuralAsset { get; private set; }
        public HashEncodedDeepMLP? MeshToSdfNetwork { get; private set; }
        public ReferenceMeshSdf? ReferenceSphere { get; private set; }
        public AABBTree<int> BroadphaseTree { get; }
        public ConcurrentSpatialHash<int> ConcurrentSpatial { get; }
        public StreamingBuffer<float> StreamRing { get; }
        public NativeBuffer<float> NativeScratch { get; }
        public SynchronizedBuffer<float> SyncScratch { get; }
        public WorkStealingPool StealPool { get; }
        public ShaderGenerator ShaderGen { get; }
        public ShaderCompiler ShaderCompile { get; }
        public ShaderVariantManager ShaderVariants { get; }
        public ConstantBufferLayout MicroMlpCbLayout { get; private set; } = null!;
        public string MeshletRasterGlsl { get; private set; } = "";
        public string NeuralComputeHlsl { get; private set; } = "";
        public string GeneratedShaderHlsl { get; private set; } = "";
        public string LastValidationSummary { get; private set; } = "";
        public float LastHyperSdfSample { get; private set; }
        public int LastStreamedAssetState { get; private set; }
        public ulong LastContentHash { get; private set; }

        /// <summary>
        /// Optional second Vulkan device used by <see cref="VulkanNeuralSdfDispatcher.Shared"/>
        /// for DeepMicroMLP compute (SPIR-V). Null when SPIR-V/Vulkan init fails (CPU fallback).
        /// </summary>
        public VulkanNeuralSdfDispatcher? GpuSdfDispatcher { get; private set; }

        /// <summary>
        /// Optional second Vulkan device used by <see cref="VulkanMeshletRasterizerDispatcher.Shared"/>
        /// for G-DNN meshlet compute raster (SPIR-V R32 visibility). Null → software fallback.
        /// </summary>
        public VulkanMeshletRasterizerDispatcher? GpuMeshletDispatcher { get; private set; }

        public int LastParticleCount { get; private set; }
        public int LastVsmCachedTiles { get; private set; }
        public int LastLoadedCells { get; private set; }
        public int LastVisibleMeshlets { get; private set; }
        public int LastNeuralLod { get; private set; }
        public int LastSoftwareRasterPixels { get; private set; }
        public int LastRasterCoveredPixels { get; private set; }
        public float LastBatchSdfMs { get; private set; }
        public float LastGpuSdfMs { get; private set; }
        public bool LastGpuSdfOk { get; private set; }
        public string GpuSdfStatus { get; private set; } = "not initialized";
        public float LastGpuMeshletMs { get; private set; }
        public bool LastGpuMeshletOk { get; private set; }
        public string GpuMeshletStatus { get; private set; } = "not initialized";
        /// <summary>Last batch SDF distances (used to paint contact AO into the present path).</summary>
        public float[] LastSdfDistances { get; private set; } = Array.Empty<float>();
        public Vector3 LastSdfSampleOrigin { get; private set; }
        public float LastNaniteLod { get; private set; }
        public int LastNanitePolyResolution { get; private set; }
        private Vector3 _sdfHintCenter;
        private float _sdfHintRadius = 1f;
        private string _sdfHintPrimitive = "sphere";
        private bool _sdfHintDirty;

        /// <summary>Notifies G-DNN Nanite Neural 3.0 of an LLM SDF primitive hint.</summary>
        public void NotifySdfHint(Vector3 center, float radius, string primitive)
        {
            _sdfHintCenter = center;
            _sdfHintRadius = MathF.Max(0.05f, radius);
            _sdfHintPrimitive = string.IsNullOrWhiteSpace(primitive) ? "sphere" : primitive;
            _sdfHintDirty = true;
            LastSdfSampleOrigin = center;
            // Bias default SDF network sample origin toward the hinted primitive.
            if (_sdfHintPrimitive.Contains("box", StringComparison.OrdinalIgnoreCase))
                LastNaniteLod = Math.Max(LastNaniteLod, 0.55f);
        }

        public bool CinematicNanite { get; set; }

        /// <summary>Active Nanite policy override from quality preset (null = Industrial / Cinematic flag).</summary>
        public NaniteNeural30.Policy? NanitePolicyOverride { get; set; }

        /// <summary>Applies AAA Nanite density from a quality preset name.</summary>
        public void ApplyAaaQuality(string presetName)
        {
            NanitePolicyOverride = NaniteNeural30.PolicyFromPreset(presetName);
            CinematicNanite = CinematicNanite
                || string.Equals(presetName, "Cinematic", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Full-res cinematic material resolve from the last visibility buffer into viewport MRTs.
        /// </summary>
        public void ResolveCinematicMaterials(
            int width,
            int height,
            Span<Vector3> albedo,
            Span<Vector3> normals,
            Span<float> roughness)
        {
            var policy = CinematicNanite ? NaniteCinematicResolve.Cinematic : NaniteNeural30.Industrial;
            float lod = LastNaniteLod;
            if (CinematicNanite)
                lod = Math.Max(lod, 0.75f);
            NaniteCinematicResolve.ResolveFullResMaterials(
                _lastRasterTarget,
                _lastMeshlets,
                lod,
                width,
                height,
                albedo,
                normals,
                roughness,
                _lastRasterMesh,
                policy);
        }

        public RenderingAlgorithmHub(VulkanRhiDevice rhi, int width, int height)
        {
            _rhi = rhi ?? throw new ArgumentNullException(nameof(rhi));
            _ = (width, height);

            VirtualShadows = new VirtualShadowMap(Math.Max(64, width / 8), Math.Max(64, height / 8), new VSMConfig
            {
                ClipmapLevels = 3,
                TileSize = 64,
                VirtualResolution = 512,
                ClipmapLevels = 4,
                TileSize = 64,
                VirtualResolution = 1024,
                PhysicalResolution = 512,
                FilterMode = VSMFilterMode.PCSS,
                CacheMode = VSMCacheMode.Software
            });

            Particles = new ParticleSystem(new ParticleSystemConfig
            {
                MaxParticles = 4096,
                MaxEmitters = 8,
                EnableGPU = true
            });
            Particles.AddEmitter(new ParticleEmitter
            {
                Name = "AmbientDust",
                Type = ParticleEmitterType.Box,
                Rate = 32f,
                Position = Vector3.Zero,
                EmitterShapeMin = new Vector3(-8f, 0.2f, -8f),
                EmitterShapeMax = new Vector3(8f, 4f, 8f),
                StartVelocityMin = new Vector3(-0.2f, 0.05f, -0.2f),
                StartVelocityMax = new Vector3(0.2f, 0.4f, 0.2f),
                LifetimeMin = 2f,
                LifetimeMax = 5f,
                StartSizeMin = 0.02f,
                StartSizeMax = 0.08f,
                StartColorMin = new Vector4(0.7f, 0.75f, 0.85f, 0.35f),
                StartColorMax = new Vector4(0.9f, 0.92f, 1f, 0.55f)
            });
            Particles.Emitters[0].Modules.Add(new ParticleModuleGravity { Gravity = new Vector3(0, -0.15f, 0) });

            VirtualTextures = new VirtualTextureSystem(new VTConfig
            {
                MaxResidentTiles = 512,
                MaxTilesPerFrame = 16,
                EnableStreaming = true
            });
            VirtualTextures.CreateLayer("WorldAlbedo", 4096, 4096, 6);

            WorldPartition = new WorldPartitionSystem(new WorldPartitionConfig
            {
                CellSize = 128f,
                MaxLoadedCells = 32,
                LoadingRange = 512f,
                UnloadingRange = 768f
            });

            Compute = new ComputeDispatcher(_rhi);
            Geometry = new GeometryRenderer();
            Megascans = new MegascansBridge(new MegascansConfig());

            // G-DNN default SDF (tiny) — enables polygonization / SIMD / cache ticks.
            DefaultSdfNetwork = new HashEncodedDeepMLP(new Random(42));
            DeepMicroNetwork = new DeepMicroMLP(new Random(7));
            DefaultBounds = new AABB(Vector3.Zero, new Vector3(2f, 2f, 2f));

            Polygonizer = new NeuralPolygonizer();
            MeshletBuilder = new NeuralMeshletBuilder();
            SoftwareRasterizer = new SoftwareRasterizer();

            NeuralLod = new NeuralLodSelector(new NeuralLodConfig { MaxLodLevel = 4 });
            NeuralLod.RegisterLod(0, DefaultSdfNetwork, Vector3.Zero, 2f, 40f);
            NeuralLod.RegisterLod(1, DefaultSdfNetwork, Vector3.Zero, 2f, 120f);
            NeuralLod.RegisterLod(2, DefaultSdfNetwork, Vector3.Zero, 2f, 400f);

            SceneEvaluator = new SceneEvaluator(new SceneEvaluatorConfig());
            SdfCache = new HierarchicalSdfCache(new HierarchicalCacheConfig());
            Gradients = new GradientCalculator();
            MicroNetwork = new MicroMLP(new Random(3));
            SphereTracer = new StochasticSphereTracer(new StochasticTracerConfig(), seed: 11);
            WaveBatch = new WaveOptimizedBatchEvaluator(new WaveBatchConfig
            {
                WaveSize = 32,
                WaveCount = 2,
                BatchSize = 64
            });
            GdnnProfiler = new Profiler();

            AssetStreamer = new AssetStreamer(new StreamerConfig
            {
                MemoryBudgetBytes = 32L * 1024 * 1024,
                MaxConcurrentDownloads = 2,
                MaxCachedAssets = 64
            });
            AsyncPipeline = new AsyncPipeline(new PipelineConfig
            {
                MaxConcurrentJobs = 2,
                EnableMonitoring = false
            });

            Jobs = new JobSystem(threadCount: Math.Clamp(Environment.ProcessorCount / 2, 1, 4));
            Jobs.Start();
            ParallelEval = new ParallelEvaluator(2);

            StackMemory = new StackAllocator(capacity: 64 * 1024);
            ZeroCopyScratch = new ZeroCopyBuffer(capacity: 64 * 1024);
            Animation = new AnimationBlender();
            SpatialOctree = new Octree<int>(DefaultBounds, maxItemsPerNode: 8, maxDepth: 4);
            LooseSpatial = new LooseOctree<int>(DefaultBounds, looseness: 1.5f, maxItemsPerNode: 8, maxDepth: 4);
            SpatialHash = new SpatialHash<int>(cellSize: 4f);
            Compression = new CompressionUtils();
            MemoryTrack = new MemoryTracker();
            Warp = new WarpSpace();
            AnimSkeleton = new Skeleton(maxJoints: 32);
            QuantizedNetwork = QuantizedDeepMLP.FromTrained(DeepMicroNetwork);

            HyperNet = new HyperNetwork(seed: 19); // not IDisposable
            BroadphaseTree = new AABBTree<int>(initialCapacity: 32);
            ConcurrentSpatial = new ConcurrentSpatialHash<int>(cellSize: 4f);
            StreamRing = new StreamingBuffer<float>(capacity: 256);
            NativeScratch = new NativeBuffer<float>(length: 64);
            SyncScratch = new SynchronizedBuffer<float>(capacity: 64);
            StealPool = new WorkStealingPool(threadCount: 2);
            StealPool.Start();
            ShaderGen = new ShaderGenerator(ShaderGeneratorConfig.ForMicroMLP(ShaderQualityLevel.Medium));
            ShaderCompile = new ShaderCompiler();
            ShaderVariants = new ShaderVariantManager();

            // Tiny default hierarchy so AnimationClip / skinning / WarpSpace have real joints.
            int root = AnimSkeleton.AddJoint("root", -1, Vector3.Zero, Quaternion.Identity, Vector3.One);
            int mid = AnimSkeleton.AddJoint("spine", root, new Vector3(0, 0.5f, 0), Quaternion.Identity, Vector3.One);
            AnimSkeleton.AddJoint("head", mid, new Vector3(0, 0.5f, 0), Quaternion.Identity, Vector3.One);
            AnimSkeleton.RebuildHierarchy();
            AnimSkeleton.ComputeBindPoseMatrices();

            IdleClip = new GDNN.Animation.AnimationClip("gdnn_idle", duration: 1f, frameRate: 30f) { Loop = true };
            IdleClip.AddPositionKeyframe(root, 0f, Vector3.Zero);
            IdleClip.AddPositionKeyframe(root, 0.5f, new Vector3(0, 0.05f, 0));
            IdleClip.AddPositionKeyframe(root, 1f, Vector3.Zero);
            IdleClip.AddRotationKeyframe(mid, 0f, Quaternion.Identity);
            IdleClip.AddRotationKeyframe(mid, 0.5f, Quaternion.CreateFromAxisAngle(Vector3.UnitY, 0.15f));
            IdleClip.AddRotationKeyframe(mid, 1f, Quaternion.Identity);
            IdleClip.ComputeAllTangents();
            _animClipIndex = AnimSkeleton.RegisterClip(IdleClip);
            Animation.SetOwnerSkeleton(AnimSkeleton);
            Animation.AddLayer(IdleClip, weight: 1f);
            Animation.AddState("idle", IdleClip);
            Animation.SetCurrentState("idle");

            SkinWeights = new SkinningWeights(vertexCount: 8, maxInfluences: 4);
            for (int v = 0; v < 8; v++)
            {
                SkinWeights.SetInfluences(v, new[]
                {
                    new SkinningWeights.Influence { BoneIndex = root, Weight = 0.6f },
                    new SkinningWeights.Influence { BoneIndex = mid, Weight = 0.4f }
                });
            }
            SkinWeights.NormalizeAllWeights();

            Warp.AddJoint(new Joint
            {
                Id = 0,
                Name = "root",
                ParentIndex = -1,
                BindPoseLocal = Matrix4x4.Identity,
                CurrentLocal = Matrix4x4.Identity
            });
            Warp.AddJoint(new Joint
            {
                Id = 1,
                Name = "spine",
                ParentIndex = 0,
                BindPoseLocal = Matrix4x4.CreateTranslation(0, 0.5f, 0),
                CurrentLocal = Matrix4x4.CreateTranslation(0, 0.5f, 0)
            });
            Warp.ComputeBindPoseWorldTransforms();
            Warp.AddVertexWeight(BoneWeight.Dual(0, 0.6f, 1, 0.4f));
            Warp.NormalizeAllWeights();
        }

        /// <summary>Replace / inject an external neural geometry pipeline (keeps ownership optional).</summary>
        public void BindNeuralGeometry(NeuralGeometryPipeline pipeline)
        {
            NeuralGeometry = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
        }

        /// <summary>Bind a meshlet page file for Nanite-style cluster streaming.</summary>
        public void BindMeshletPageFile(string path, long memoryBudgetBytes = 32L * 1024 * 1024)
        {
            MeshletStreamer?.Dispose();
            MeshletStreamer = new MeshletStreamer(MeshletPageFile.Open(path), memoryBudgetBytes);
            _meshletPagePath = path;
        }

        private void EnsureNeuralGeometry()
        {
            if (NeuralGeometry != null)
                return;

            try
            {
                NeuralGeometry = new NeuralGeometryPipeline(
                    DefaultSdfNetwork,
                    DefaultBounds,
                    new NeuralGeometryPipelineOptions
                    {
                        BaseResolution = 8,
                        LevelCount = 2,
                        PixelErrorBudget = 2.0f,
                        RebuildIntervalTicks = 32,
                        CacheDirectory = _polyCacheDir ??= Path.Combine(Path.GetTempPath(), "synapse_gdnn_poly_cache")
                    });
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AlgorithmHub", "NeuralGeometryPipeline init deferred.", ex);
            }
        }

        /// <summary>G-DNN + streaming cull tick — called from AlgorithmSystemsPass.</summary>
        public void TickCull(Vector3 cameraPos, Vector3 cameraForward, Matrix4x4 viewProj, int width, int height, float time)
        {
            float dt = ComputeDelta(time);
            _gdnnTick++;
            _lastCullWidth = Math.Max(1, width);
            _lastCullHeight = Math.Max(1, height);

            WorldPartition.Update(cameraPos, dt);
            LastLoadedCells = WorldPartition.LoadedCells;

            VirtualTextures.Update(cameraPos, viewProj, width, height);
            VirtualTextures.RequestTiles(0, new Vector2(0.25f, 0.25f), new Vector2(0.75f, 0.75f), 64);
            if ((_gdnnTick & 7) == 0)
                VirtualTextures.BlitResidentPagesToAtlas();

            EnsureNeuralGeometry();
            EnsureGpuShaderResidency();
            EnsureGdnnFullCoverage();

            var camState = new CameraState
            {
                Position = cameraPos,
                Forward = cameraForward,
                Up = Vector3.UnitY,
                Right = SafeRight(cameraForward),
                FieldOfView = MathF.PI / 3f,
                ScreenWidth = width,
                ScreenHeight = height,
                ViewProjection = viewProj
            };
            LastNeuralLod = NeuralLod.SelectLod(0, camState, cameraPos, 2f);

            SceneEvaluator.BeginFrame(viewProj, cameraPos, cameraForward);
            _ = SceneEvaluator.TraceRay(new TracingRay(cameraPos, cameraForward, 40f), cameraPos);
            _ = SceneEvaluator.EndFrame(0.01f);

            // SIMD batch SDF + hierarchical cache + wave evaluator (every frame, tiny batch).
            Span<Vector3> pts = stackalloc Vector3[8];
            Span<float> dists = stackalloc float[8];
            for (int i = 0; i < pts.Length; i++)
                pts[i] = cameraPos + cameraForward * (0.5f + i * 0.35f) + new Vector3(0.1f * i, 0, 0);
            BatchSdfEvaluator.EvaluateBatch(DefaultSdfNetwork, pts, dists);
            LastBatchSdfMs = (float)BatchSdfEvaluator.LastBatchTimeMs;
            if (LastSdfDistances.Length != dists.Length)
                LastSdfDistances = new float[dists.Length];
            dists.CopyTo(LastSdfDistances);
            LastSdfSampleOrigin = cameraPos;
            WaveBatch.EvaluateBatchWave(DeepMicroNetwork, pts, dists);
            WaveBatch.EvaluateBatchWave(MicroNetwork, pts, dists);
            _ = QuantizedNetwork.Evaluate(pts[0]);
            _ = SdfCache.TryLookupSimple(pts[0], out _);
            _ = Gradients.ComputeCentralDifference(MicroNetwork, pts[0]);
            TickGdnnInfrastructure(pts, dists, cameraPos, dt);

            // Staggered Vulkan SDF compute on the dedicated second device (when available).
            if ((_gdnnTick % 15) == 3)
                TickGpuSdf(pts);

            // Sphere tracer + surface evaluator (staggered).
            if ((_gdnnTick & 3) == 0)
            {
                var ray = new TracingRay(cameraPos, cameraForward, 50f);
                _ = SphereTracer.Trace(DeepMicroNetwork, ray);
                _ = SurfaceEvaluator.TraceRay(cameraPos, cameraForward, cameraPos);
                _ = Compression.Compress(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }, CompressionAlgorithm.LZ4);
                if (HyperGeneratedMlp != null)
                    LastHyperSdfSample = HyperGeneratedMlp.Evaluate(pts[0]);
                if (ReferenceSphere != null)
                    _ = ReferenceSphere.SignedDistance(pts[0]);
            }

            // JobSystem + ParallelEvaluator + work-stealing residency.
            float d0 = dists[0];
            var handle = Jobs.ScheduleParallelFor(i => { _ = d0 + i; }, 0, 8);
            Jobs.WaitForAll(new[] { handle });
            float[] sdfSnap = LastSdfDistances;
            unsafe
            {
                _ = ParallelEval.Evaluate(
                    (iteration, threadId, state) => sdfSnap[iteration & 7],
                    totalIterations: 8,
                    reduce: (a, b) => a + b,
                    seed: 0f);
            }
            StealPool.Submit(() => { _ = IntrinsicsHelper.IsAvx2Supported; });

            // Spatial structures + animation + memory pools.
            SpatialHash.Insert(0, cameraPos);
            ConcurrentSpatial.Insert(0, cameraPos);
            SpatialOctree.Insert(0, cameraPos);
            LooseSpatial.Insert(0, cameraPos);
            TickGdnnAnimation(dt);

            var view = CameraView.CreatePerspectiveLookAt(
                cameraPos,
                cameraPos + cameraForward,
                MathF.PI / 3f,
                height > 0 ? (float)width / height : 16f / 9f,
                0.1f,
                500f,
                height);

            if (NeuralGeometry != null)
            {
                var report = NeuralGeometry.Tick(view, trainBudgetMs: 0.25);
                LastVisibleMeshlets = report.VisibleClusters?.Count ?? 0;

                // Dense Nanite-like meshlets: high poly extract + screen-scale visibility raster.
                // Prefer GPU dispatcher; fall back to SoftwareRasterizer. Feeds G-buffer inject + GI/fog.
                if ((_gdnnTick % 3) == 1 && report.VisibleClusters is { Count: > 0 })
                {
                    try
                    {
                        // denser grid → many small clusters filling the frame (Nanite-like look)
                        int polyRes = Math.Clamp(Math.Max(_lastCullWidth, _lastCullHeight) / 48, 20, 36);
                // Dense Nanite Neural 3.0 meshlets: continuous LOD + screen-error density.
                // Prefer GPU dispatcher; fall back to SoftwareRasterizer. Feeds G-buffer inject + GI/fog.
                float projected = NaniteNeural30.ProjectedRadiusPx(
                    _sdfHintCenter,
                    _sdfHintRadius,
                    cameraPos,
                    MathF.PI / 3f,
                    Math.Max(1, _lastCullHeight));
                float lod01 = NaniteNeural30.ContinuousLod(
                    Vector3.Distance(cameraPos, _sdfHintCenter),
                    projected);
                LastNaniteLod = lod01;

                var policy = NanitePolicyOverride
                    ?? (CinematicNanite ? NaniteCinematicResolve.Cinematic : NaniteNeural30.Industrial);
                if (NaniteNeural30.ShouldRebuildMeshlets(_gdnnTick, _sdfHintDirty, policy) &&
                    report.VisibleClusters is { Count: > 0 })
                {
                    try
                    {
                        int polyRes = NaniteNeural30.PolyResolution(
                            lod01,
                            Math.Max(_lastCullWidth, _lastCullHeight),
                            policy);
                        LastNanitePolyResolution = polyRes;
                        var mesh = Polygonizer.Extract(DefaultSdfNetwork, DefaultBounds, resolution: polyRes);
                        var meshlets = MeshletBuilder.Build(mesh);
                        LastVisibleMeshlets = Math.Max(LastVisibleMeshlets, meshlets.Count);

                        // Visibility buffer at ~½ viewport (capped) so clusters read as tiny tiles on screen.
                        int rastW = Math.Clamp(_lastCullWidth / 2, 256, 512);
                        int rastH = Math.Clamp(_lastCullHeight / 2, 256, 512);
                        var (rastW, rastH) = NaniteNeural30.VisibilityBufferSize(
                            _lastCullWidth, _lastCullHeight, lod01, policy);
                        if (CinematicNanite)
                        {
                            // Full-res cinematic visibility (capped by policy max).
                            rastW = Math.Clamp(_lastCullWidth, policy.MinVisibilityWidth, policy.MaxVisibilityWidth) & ~1;
                            rastH = Math.Clamp(_lastCullHeight, policy.MinVisibilityWidth, policy.MaxVisibilityWidth) & ~1;
                        }
                        RasterTarget target = RasterizeMeshlets(mesh, meshlets, view, rastW, rastH);
                        LastRasterCoveredPixels = target.CountCoveredPixels();
                        _lastRasterTarget = target;
                        _lastMeshlets = meshlets;
                        _lastRasterMesh = mesh;
                        QueuePresentMesh(mesh, report.ExtractedGeometryVersion);
                        FeedGeometryRenderer(mesh, meshlets);
                        _sdfHintDirty = false;
                    }
                    catch (Exception ex)
                    {
                        SynapseLogger.Default.Warn("AlgorithmHub", "Meshlet/software raster tick skipped.", ex);
                    }
                }

                // Auto-bind meshlet page streamer once from the live chain.
                if (MeshletStreamer == null && (_gdnnTick % 60) == 2)
                    TryBuildMeshletPageStreamer();
            }

            if (MeshletStreamer != null)
            {
                var keys = MeshletStreamer.QueryVisible(view, level: LastNeuralLod >= 0 ? LastNeuralLod : 0);
                LastVisibleMeshlets = Math.Max(LastVisibleMeshlets, keys.Count);
                foreach (var key in keys)
                    _ = MeshletStreamer.GetOrLoad(key);

                // Feed resident streamed clusters into the present raster path on ticks where the
                // densify polygonizer did NOT already rebuild _lastRasterTarget (%3==1). Without
                // this, off-core meshlet streaming only warmed the LRU cache and never reached screen.
                if (keys.Count > 0 && (_gdnnTick % 3) != 1)
                    TryRasterizeStreamedClusters(view);
            }

            Geometry.ClearCommands();
            if (_geometryMeshId >= 0)
                Geometry.SubmitDraw(_geometryMeshId, Matrix4x4.Identity);
            _ = Geometry.FlushCommands();

            ProfilerGlobal.Current = GdnnProfiler;
            GdnnProfiler.BeginSection("GdnnCull");
            GdnnProfiler.EndSection();
        }

        public void TickShadows(Vector3 cameraPos, Matrix4x4 lightView, Matrix4x4 lightProj)
        {
            VirtualShadows.UpdateClipmaps(cameraPos, lightView, lightProj);
            VirtualShadows.GetStatistics(out _, out _, out var cached, out _);
            LastVsmCachedTiles = cached;
        }

        public float SampleVirtualShadow(Vector3 worldPos, Vector3 lightDir)
            => VirtualShadows.SampleShadow(worldPos, lightDir);

        public void TickPost(Vector3 cameraPos, float time)
        {
            float dt = ComputeDelta(time);
            Particles.Update(dt, cameraPos);
            LastParticleCount = Particles.ActiveParticles;

            try
            {
                var empty = Array.Empty<ComputeBuffer>();
                Compute.Dispatch("particle_update", 1, 1, 1, empty);
                Compute.Dispatch("taa_resolve", 1, 1, 1, empty);
                Compute.Dispatch("shadow_filter", 1, 1, 1, empty);
                Compute.Dispatch("bloom_downsample", 1, 1, 1, empty);
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AlgorithmHub", "Compute tick skipped.", ex);
            }
        }

        public void CompositeParticlesIntoFog(Vector3[,] fog, Matrix4x4 viewProj, int width, int height)
        {
            if (fog == null || Particles.ActiveParticles <= 0)
                return;

            int count = Math.Min(Particles.ActiveParticles, 256);
            for (int i = 0; i < count; i++)
            {
                Particles.GetParticleData(i, out var pos, out var color, out var size);
                var clip = Vector4.Transform(new Vector4(pos, 1f), viewProj);
                if (clip.W <= 1e-4f)
                    continue;
                float invW = 1f / clip.W;
                float ndcX = clip.X * invW;
                float ndcY = clip.Y * invW;
                if (ndcX < -1f || ndcX > 1f || ndcY < -1f || ndcY > 1f)
                    continue;

                int px = (int)((ndcX * 0.5f + 0.5f) * (width - 1));
                int py = (int)((ndcY * 0.5f + 0.5f) * (height - 1));
                int radius = Math.Max(1, (int)(size * 40f));
                for (int dy = -radius; dy <= radius; dy++)
                {
                    int y = py + dy;
                    if ((uint)y >= (uint)height)
                        continue;
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        int x = px + dx;
                        if ((uint)x >= (uint)width)
                            continue;
                        float w = 1f - MathF.Sqrt(dx * dx + dy * dy) / (radius + 1);
                        if (w <= 0f)
                            continue;
                        fog[x, y] += new Vector3(color.X, color.Y, color.Z) * (color.W * w * 0.08f);
                    }
                }
            }
        }

        /// <summary>
        /// Resolves the last meshlet visibility buffer into cluster-colored albedo (Nanite-like tiles)
        /// and composites into the L-DNN fog field for on-screen geometry presence.
        /// </summary>
        public void CompositeMeshletsIntoFog(Vector3[,] fog, int width, int height)
        {
            if (fog == null || _lastRasterTarget == null || width <= 0 || height <= 0)
                return;

            var target = _lastRasterTarget;
            for (int y = 0; y < height; y++)
            {
                int sy = Math.Min(target.Height - 1, y * target.Height / height);
                for (int x = 0; x < width; x++)
                {
                    int sx = Math.Min(target.Width - 1, x * target.Width / width);
                    if (!target.IsCovered(sx, sy))
                        continue;
                    float closeness = target.ClosenessAt(sx, sy);
                    var albedo = ResolveMeshletAlbedo(target, sx, sy);
                    // Soft in-scatter from cluster albedo — secondary to real G-buffer inject.
                    fog[x, y] += albedo * (closeness * 0.22f);
                }
            }
        }

        /// <summary>
        /// Paints resolved meshlet albedo / closeness into the GI irradiance field so cluster
        /// virtualization lights deferred shading (not just fog tint).
        /// </summary>
        public void CompositeMeshletsIntoIrradiance(Vector3[,] irradiance, int width, int height)
        {
            if (irradiance == null || _lastRasterTarget == null || width <= 0 || height <= 0)
                return;

            var target = _lastRasterTarget;
            for (int y = 0; y < height; y++)
            {
                int sy = Math.Min(target.Height - 1, y * target.Height / height);
                for (int x = 0; x < width; x++)
                {
                    int sx = Math.Min(target.Width - 1, x * target.Width / width);
                    if (!target.IsCovered(sx, sy))
                        continue;
                    float closeness = target.ClosenessAt(sx, sy);
                    var albedo = ResolveMeshletAlbedo(target, sx, sy);
                    // Cluster bounce proxy into GI — visible as Nanite-tile indirect light.
                    irradiance[x, y] += albedo * (0.55f + closeness * 0.65f);
                }
            }
        }

        /// <summary>
        /// Writes meshlet depth/normal/albedo into the L-DNN CPU G-buffer so Hybrid GI
        /// sees real cluster geometry (not constant fill).
        /// </summary>
        public void CompositeMeshletsIntoLdnnGBuffer(
            float[] depth, Vector3[] normals, Vector3[] albedo, int width, int height)
        {
            if (_lastRasterTarget == null || depth == null || normals == null || albedo == null)
                return;
            if (width <= 0 || height <= 0 || depth.Length < width * height)
                return;

            var target = _lastRasterTarget;
            var mesh = _lastRasterMesh;
            for (int y = 0; y < height; y++)
            {
                int sy = Math.Min(target.Height - 1, y * target.Height / height);
                for (int x = 0; x < width; x++)
                {
                    int sx = Math.Min(target.Width - 1, x * target.Width / width);
                    if (!target.IsCovered(sx, sy))
                        continue;

                    int idx = y * width + x;
                    float closeness = target.ClosenessAt(sx, sy);
                    // Closeness→linear depth proxy (near = small depth).
                    float d = Math.Clamp(1.5f - closeness * 1.2f, 0.15f, 80f);
                    depth[idx] = d;
                    albedo[idx] = ResolveMeshletAlbedo(target, sx, sy);

                    if (target.TryDecode(sx, sy, out int meshletIdx, out _) &&
                        mesh != null &&
                        _lastMeshlets != null &&
                        (uint)meshletIdx < (uint)_lastMeshlets.Count)
                    {
                        var ml = _lastMeshlets[meshletIdx];
                        var n = ml.ConeAxis;
                        if (n.LengthSquared() > 1e-8f)
                            normals[idx] = Vector3.Normalize(n);
                        else
                            normals[idx] = Vector3.UnitY;
                    }
                    else
                    {
                        normals[idx] = Vector3.UnitY;
                    }
                }
            }
        }

        /// <summary>Stable per-meshlet albedo so many small clusters read as Nanite tiles on screen.</summary>
        private Vector3 ResolveMeshletAlbedo(RasterTarget target, int sx, int sy)
        {
            if (!target.TryDecode(sx, sy, out int meshletIdx, out int triIdx))
                return new Vector3(0.55f, 0.58f, 0.62f);

            // Hash meshlet+triangle → distinct tile color (virtualized cluster look).
            uint h = unchecked((uint)(meshletIdx * 73856093) ^ (uint)(triIdx * 19349663));
            float r = ((h) & 255) / 255f;
            float g = ((h >> 8) & 255) / 255f;
            float b = ((h >> 16) & 255) / 255f;
            // Keep in a PBR-friendly mid range with slight cool bias for rock/metal feel.
            return new Vector3(
                0.28f + r * 0.55f,
                0.30f + g * 0.50f,
                0.32f + b * 0.48f);
            var mat = NaniteNeural30.ResolveClusterMaterial(meshletIdx, triIdx, LastNaniteLod);
            return new Vector3(mat.X, mat.Y, mat.Z);
        }

        /// <summary>
        /// Paints resident VT atlas colors into fog so virtual-texture streaming is visible
        /// even without Megascans disk assets.
        /// </summary>
        public void CompositeVirtualTexturesIntoFog(Vector3[,] fog, int width, int height)
        {
            if (fog == null || width <= 0 || height <= 0 || VirtualTextures.ResidentTiles <= 0)
                return;

            VirtualTextures.BlitResidentPagesToAtlas();
            int step = Math.Max(1, Math.Min(width, height) / 64);
            for (int y = 0; y < height; y += step)
            {
                float v = y / (float)Math.Max(1, height - 1);
                for (int x = 0; x < width; x += step)
                {
                    float u = x / (float)Math.Max(1, width - 1);
                    var c = VirtualTextures.SampleAtlas(u * 0.5f + 0.25f, v * 0.5f + 0.25f);
                    for (int dy = 0; dy < step && y + dy < height; dy++)
                    {
                        for (int dx = 0; dx < step && x + dx < width; dx++)
                            fog[x + dx, y + dy] += c * 0.08f;
                    }
                }
            }
        }

        /// <summary>
        /// Uses last SDF evaluation distances as soft contact AO so the 2nd-device / CPU SDF
        /// path darkens the present image near the camera sample ray.
        /// </summary>
        public void CompositeSdfAo(float[,] ao, int width, int height)
        {
            if (ao == null || width <= 0 || height <= 0 || LastSdfDistances.Length == 0)
                return;

            float mean = 0f;
            for (int i = 0; i < LastSdfDistances.Length; i++)
                mean += MathF.Abs(LastSdfDistances[i]);
            mean /= LastSdfDistances.Length;
            // Closer surface → stronger AO (darker).
            float contact = Math.Clamp(1f - mean * 0.35f, 0.55f, 1f);

            // Soft vignette toward screen center along the sample ray.
            for (int y = 0; y < height; y++)
            {
                float ny = (y / (float)Math.Max(1, height - 1)) * 2f - 1f;
                for (int x = 0; x < width; x++)
                {
                    float nx = (x / (float)Math.Max(1, width - 1)) * 2f - 1f;
                    float r = MathF.Sqrt(nx * nx + ny * ny);
                    float w = Math.Clamp(1f - r * 0.65f, 0f, 1f);
                    ao[x, y] *= Math.Clamp(1f - (1f - contact) * w, 0.4f, 1f);
                }
            }
        }

        /// <summary>
        /// Peek a polygonized G-DNN mesh waiting for SceneRenderer upload.
        /// Call <see cref="AcknowledgePresentMesh"/> after a successful inject.
        /// </summary>
        public NeuralPolygonMesh? PeekPendingPresentMesh() => _pendingPresentMesh;

        public void AcknowledgePresentMesh() => _pendingPresentMesh = null;

        private void TickGpuSdf(ReadOnlySpan<Vector3> pts)
        {
            var gpu = GpuSdfDispatcher ?? VulkanNeuralSdfDispatcher.Shared;
            GpuSdfDispatcher = gpu;
            GpuSdfStatus = VulkanNeuralSdfDispatcher.StatusLog;

            if (gpu == null)
            {
                LastGpuSdfOk = false;
                // CPU proxy keeps the same math as the SPIR-V path when the 2nd device is unavailable.
                var cpu = new HlslCompatibleEvaluator(DeepMicroNetwork);
                Span<float> tmp = stackalloc float[pts.Length];
                cpu.EvaluateBatch(pts, tmp);
                return;
            }

            try
            {
                var sw = Stopwatch.StartNew();
                var distances = gpu.Evaluate(DeepMicroNetwork, pts);
                sw.Stop();
                LastGpuSdfMs = (float)sw.Elapsed.TotalMilliseconds;
                LastGpuSdfOk = distances.Length == pts.Length;
                if (distances.Length > 0)
                {
                    LastSdfDistances = distances;
                    LastSdfSampleOrigin = pts.Length > 0 ? pts[0] : LastSdfSampleOrigin;
                }
                if (!_gpuSdfInitLogged)
                {
                    _gpuSdfInitLogged = true;
                    SynapseLogger.Default.Info("AlgorithmHub",
                        $"VulkanNeuralSdfDispatcher active (2nd device): {GpuSdfStatus}");
                }
            }
            catch (Exception ex)
            {
                LastGpuSdfOk = false;
                GpuSdfStatus = "dispatch failed: " + ex.Message;
                SynapseLogger.Default.Warn("AlgorithmHub", "GPU SDF dispatch failed; CPU path continues.", ex);
            }
        }

        private RasterTarget RasterizeMeshlets(
            NeuralPolygonMesh mesh,
            IReadOnlyList<NeuralMeshlet> meshlets,
            in CameraView view,
            int width,
            int height)
        {
            var gpu = GpuMeshletDispatcher ?? VulkanMeshletRasterizerDispatcher.Shared;
            GpuMeshletDispatcher = gpu;
            GpuMeshletStatus = VulkanMeshletRasterizerDispatcher.StatusLog;

            if (gpu != null)
            {
                try
                {
                    var sw = Stopwatch.StartNew();
                    var target = gpu.Rasterize(mesh, meshlets, view, width, height);
                    sw.Stop();
                    LastGpuMeshletMs = (float)sw.Elapsed.TotalMilliseconds;
                    LastGpuMeshletOk = true;
                    LastSoftwareRasterPixels = 0;
                    if (!_gpuMeshletInitLogged)
                    {
                        _gpuMeshletInitLogged = true;
                        SynapseLogger.Default.Info("AlgorithmHub",
                            $"VulkanMeshletRasterizerDispatcher active (2nd device): {GpuMeshletStatus}");
                    }
                    return target;
                }
                catch (Exception ex)
                {
                    LastGpuMeshletOk = false;
                    GpuMeshletStatus = "dispatch failed: " + ex.Message;
                    SynapseLogger.Default.Warn("AlgorithmHub",
                        "GPU meshlet raster failed; falling back to SoftwareRasterizer.", ex);
                }
            }
            else
            {
                LastGpuMeshletOk = false;
            }

            var cpuTarget = new RasterTarget(width, height);
            var stats = SoftwareRasterizer.Rasterize(cpuTarget, mesh, meshlets, view);
            LastSoftwareRasterPixels = stats.TrianglesRasterized;
            return cpuTarget;
        }

        /// <summary>
        /// Builds a composite <see cref="NeuralPolygonMesh"/> + <see cref="NeuralMeshlet"/> list from
        /// up to 48 resident streamed clusters and rasterizes them into <see cref="_lastRasterTarget"/>
        /// so off-core meshlet streaming feeds the present path (fog / GI / G-buffer inject).
        /// </summary>
        private void TryRasterizeStreamedClusters(in CameraView view)
        {
            if (MeshletStreamer == null)
                return;

            try
            {
                int level = LastNeuralLod >= 0 ? LastNeuralLod : 0;
                var keys = MeshletStreamer.QueryVisible(view, level);
                if (keys.Count == 0)
                    return;

                var positions = new System.Collections.Generic.List<Vector3>();
                var normals = new System.Collections.Generic.List<Vector3>();
                var indices = new System.Collections.Generic.List<int>();
                var meshlets = new System.Collections.Generic.List<NeuralMeshlet>();

                int taken = 0;
                foreach (var key in keys)
                {
                    if (taken >= 48)
                        break;
                    if (!MeshletStreamer.TryGetResident(key, out var cluster))
                        continue;
                    if (cluster.VertexCount == 0 || cluster.TriangleCount == 0)
                        continue;

                    int baseVertex = positions.Count;
                    for (int i = 0; i < cluster.Positions.Length; i++)
                    {
                        positions.Add(cluster.Positions[i]);
                        normals.Add(i < cluster.Normals.Length ? cluster.Normals[i] : Vector3.UnitY);
                    }

                    var vertexIndices = new int[cluster.VertexCount];
                    for (int i = 0; i < vertexIndices.Length; i++)
                        vertexIndices[i] = baseVertex + i;

                    var local = cluster.LocalIndices;
                    for (int i = 0; i < local.Length; i++)
                        indices.Add(baseVertex + local[i]);

                    meshlets.Add(new NeuralMeshlet
                    {
                        VertexIndices = vertexIndices,
                        LocalIndices = local,
                        Bounds = cluster.Bounds,
                        ConeAxis = cluster.ConeAxis,
                        ConeCutoff = cluster.ConeCutoff
                    });
                    taken++;
                }

                if (meshlets.Count == 0 || indices.Count == 0)
                    return;

                var mesh = new NeuralPolygonMesh
                {
                    Positions = positions.ToArray(),
                    Normals = normals.ToArray(),
                    Indices = indices.ToArray()
                };

                int rastW = Math.Clamp(_lastCullWidth / 2, 256, 512);
                int rastH = Math.Clamp(_lastCullHeight / 2, 256, 512);
                RasterTarget target = RasterizeMeshlets(mesh, meshlets, view, rastW, rastH);
                LastRasterCoveredPixels = target.CountCoveredPixels();
                _lastRasterTarget = target;
                _lastMeshlets = meshlets;
                _lastRasterMesh = mesh;
                LastVisibleMeshlets = Math.Max(LastVisibleMeshlets, meshlets.Count);
                QueuePresentMesh(mesh, unchecked((long)_gdnnTick * 2654435761L));
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AlgorithmHub", "Streamed cluster raster tick skipped.", ex);
            }
        }

        private void QueuePresentMesh(NeuralPolygonMesh mesh, long version)
        {
            if (mesh.TriangleCount <= 0)
                return;
            if (version == _presentMeshVersion && _pendingPresentMesh == null)
                return;
            _presentMeshVersion = version;
            _pendingPresentMesh = mesh;
        }

        private void FeedGeometryRenderer(NeuralPolygonMesh mesh, IReadOnlyList<NeuralMeshlet> meshlets)
        {
            if (mesh.VertexCount == 0 || mesh.TriangleCount == 0)
                return;

            // Pack pos+normal (6 floats) for GeometryRenderer residency / batch flush.
            var verts = new float[mesh.VertexCount * 6];
            for (int i = 0; i < mesh.VertexCount; i++)
            {
                var p = mesh.Positions[i];
                var n = i < mesh.Normals.Length ? mesh.Normals[i] : Vector3.UnitY;
                int o = i * 6;
                verts[o] = p.X;
                verts[o + 1] = p.Y;
                verts[o + 2] = p.Z;
                verts[o + 3] = n.X;
                verts[o + 4] = n.Y;
                verts[o + 5] = n.Z;
            }

            var indices = new uint[mesh.Indices.Length];
            for (int i = 0; i < mesh.Indices.Length; i++)
                indices[i] = (uint)mesh.Indices[i];

            if (_geometryMeshId < 0)
                _geometryMeshId = Geometry.CreateMesh("gdnn_meshlets");
            Geometry.SetMeshData(_geometryMeshId, verts, 6, indices);
            LastVisibleMeshlets = Math.Max(LastVisibleMeshlets, meshlets.Count);
        }

        private void EnsureGpuShaderResidency()
        {
            if (_gpuShaderResidencyDone)
                return;
            _gpuShaderResidencyDone = true;
            try
            {
                MeshletRasterGlsl = MeshletRasterizerShaderGenerator.GenerateGlslR32();
                NeuralComputeHlsl = NeuralComputeShaderGenerator.GenerateComputeShaderForSpirv();
                GeneratedShaderHlsl = ShaderGen.GenerateCompleteShader();
                MicroMlpCbLayout = ConstantBufferLayoutBuilder.ComputeMicroMLPLayout();
                _ = ConstantBufferLayoutBuilder.PackMicroMLP(MicroNetwork);
                _ = ConstantBufferLayoutBuilder.GenerateHLSL(MicroMlpCbLayout);
                var compile = ShaderCompile.Compile(GeneratedShaderHlsl, "PSMain", ShaderType.PixelShader);
                if (compile.Success && compile.Bytecode != null)
                {
                    var key = ShaderVariantKey.FullFeatured;
                    ShaderVariants.RegisterVariant(new ShaderVariant
                    {
                        Key = key,
                        Bytecode = compile.Bytecode,
                        EntryPoint = "PSMain"
                    });
                    ShaderVariants.SetActive(key);
                    _ = ShaderVariants.GetBestMatch(key);
                }
                _ = DeepMicroMLPSpirvEmitter.TryGetSpirv(out _, out _);
                _ = MeshletRasterizerShaderGenerator.TryGetSpirv(out _, out _, out _);
                _ = IntrinsicsHelper.GetFeatureSet();
                _ = SpirvToolchain.IsAvailable;

                // User-approved: Shared creates a dedicated second Vulkan device for SDF compute.
                GpuSdfDispatcher = VulkanNeuralSdfDispatcher.Shared;
                GpuSdfStatus = VulkanNeuralSdfDispatcher.StatusLog;
                if (GpuSdfDispatcher != null)
                {
                    SynapseLogger.Default.Info("AlgorithmHub",
                        $"G-DNN SDF 2nd Vulkan device ready: {GpuSdfStatus}");
                }
                else
                {
                    SynapseLogger.Default.Warn("AlgorithmHub",
                        $"G-DNN SDF GPU unavailable ({GpuSdfStatus}); CPU HlslCompatibleEvaluator fallback.");
                }

                GpuMeshletDispatcher = VulkanMeshletRasterizerDispatcher.Shared;
                GpuMeshletStatus = VulkanMeshletRasterizerDispatcher.StatusLog;
                if (GpuMeshletDispatcher != null)
                {
                    SynapseLogger.Default.Info("AlgorithmHub",
                        $"G-DNN meshlet GPU raster 2nd Vulkan device ready: {GpuMeshletStatus}");
                }
                else
                {
                    SynapseLogger.Default.Warn("AlgorithmHub",
                        $"G-DNN meshlet GPU unavailable ({GpuMeshletStatus}); SoftwareRasterizer fallback.");
                }
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AlgorithmHub", "G-DNN GPU shader residency skipped.", ex);
                GpuSdfStatus = "init exception: " + ex.Message;
            }
        }

        private void TryBuildMeshletPageStreamer()
        {
            if (NeuralGeometry?.Chain == null)
                return;
            try
            {
                string path = Path.Combine(Path.GetTempPath(), "synapse_gdnn_meshlets.glpn");
                MeshletPageFile.Build(NeuralGeometry.Chain, path);
                BindMeshletPageFile(path);
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AlgorithmHub", "Meshlet page streamer build skipped.", ex);
            }
        }

        private static Vector3 SafeRight(Vector3 forward)
        {
            var r = Vector3.Cross(forward, Vector3.UnitY);
            return r.LengthSquared() < 1e-6f ? Vector3.UnitX : Vector3.Normalize(r);
        }

        private float ComputeDelta(float time)
        {
            float dt = _lastTime <= 0f ? 1f / 60f : Math.Clamp(time - _lastTime, 1f / 240f, 0.1f);
            _lastTime = time;
            return dt;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;

            VirtualShadows.Dispose();
            Particles.Dispose();
            VirtualTextures.Dispose();
            WorldPartition.Dispose();
            Compute.Dispose();
            Geometry.Dispose();
            Megascans.Dispose();

            MeshletStreamer?.Dispose();
            MeshletStreamer = null;
            NeuralGeometry = null;
            // Do not Dispose GpuSdfDispatcher / GpuMeshletDispatcher — Shared owns process-lifetime 2nd devices.
            GpuSdfDispatcher = null;
            GpuMeshletDispatcher = null;
            DefaultSdfNetwork.Dispose();
            DeepMicroNetwork.Dispose();
            MicroNetwork.Dispose();
            HyperGeneratedMlp?.Dispose();
            HyperGeneratedMlp = null;
            MeshToSdfNetwork?.Dispose();
            MeshToSdfNetwork = null;
            SceneEvaluator.Dispose();
            SdfCache.Dispose();
            WaveBatch.Dispose();
            AssetStreamer.Dispose();
            AsyncPipeline.Dispose();
            Jobs.Dispose();
            ParallelEval.Dispose();
            StealPool.Dispose();
            StackMemory.Dispose();
            ZeroCopyScratch.Dispose();
            NativeScratch.Dispose();
            SyncScratch.Dispose();
            StreamRing.Dispose();
            Animation.Dispose();
            IdleClip?.Dispose();
            SkinWeights?.Dispose();
            SpatialOctree.Dispose();
            LooseSpatial.Dispose();
            SpatialHash.Dispose();
            ConcurrentSpatial.Dispose();
            BroadphaseTree.Dispose();
            NeuralLod.Dispose();
            GdnnProfiler.Dispose();
            MemoryTrack.Dispose();
            Warp.Dispose();
            AnimSkeleton.Dispose();
            ShaderCompile.Dispose();
            ShaderVariants.Clear();

            _lastRasterTarget = null;
            _lastMeshlets = null;
            _lastRasterMesh = null;
            _pendingPresentMesh = null;
            StreamedNeuralAsset = null;
            ReferenceSphere = null;

            if (_meshletPagePath != null)
            {
                try
                { File.Delete(_meshletPagePath); }
                catch { /* ignore */ }
            }
            if (_polyCacheDir != null)
            {
                try
                { Directory.Delete(_polyCacheDir, recursive: true); }
                catch { /* ignore */ }
            }
            if (_assetStreamRoot != null)
            {
                try
                { Directory.Delete(_assetStreamRoot, recursive: true); }
                catch { /* ignore */ }
            }
        }
    }
}
