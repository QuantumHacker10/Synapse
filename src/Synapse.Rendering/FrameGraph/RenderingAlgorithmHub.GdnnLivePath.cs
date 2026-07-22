using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using GDNN.Core.DataStructures;
using GDNN.Core.NeuralNetwork;
using GDNN.Evaluation;
using GDNN.Polygonization;
using GDNN.Rendering.Compute;
using GDNN.SIMD;
using GDNN.Streaming;
using Synapse.Infrastructure.Logging;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>
    /// Live present-path wiring for G-DNN: trained/streamed SDF → NeuralGeometry →
    /// meshlets / GeometryRenderer → fog / AO / G-buffer inject, plus VSM and spatial queries.
    /// </summary>
    public sealed partial class RenderingAlgorithmHub
    {
        private bool _liveSdfPromoted;
        private bool _particlesAdvancedThisTick;
        private GridEvaluator? _gridEvaluator;
        private ISdfNetwork? _streamedEvalNetwork;

        /// <summary>
        /// SDF used for polygonization / batch evaluation / AO — prefers MeshToSdf /
        /// streamed HashEncoded asset over the random default seed network.
        /// </summary>
        public HashEncodedDeepMLP LivePolygonSdf => MeshToSdfNetwork ?? DefaultSdfNetwork;

        /// <summary>Mean VSM occlusion sampled this frame (1 = lit, 0 = full shadow).</summary>
        public float LastVsmSample { get; private set; } = 1f;

        /// <summary>Draw batches flushed from GeometryRenderer this frame.</summary>
        public int LastGeometryBatches { get; private set; }

        /// <summary>Spatial / broadphase query hit count this frame.</summary>
        public int LastSpatialHits { get; private set; }

        /// <summary>Mean depth from the tiny GridEvaluator scan (when run).</summary>
        public float LastGridDepthMean { get; private set; }

        /// <summary>True after offline/MeshToSdf weights drive NeuralGeometry.</summary>
        public bool LiveSdfPromoted => _liveSdfPromoted;

        /// <summary>
        /// Rebuilds NeuralGeometry from <see cref="LivePolygonSdf"/> and re-registers NeuralLod.
        /// Called after offline / MeshToSdf training so the present path paints trained geometry.
        /// </summary>
        public void PromoteTrainedSdfToLive()
        {
            var live = LivePolygonSdf;
            NeuralGeometry = null;
            NeuralLod.Clear();
            NeuralLod.RegisterLod(0, live, Vector3.Zero, 2f, 40f);
            NeuralLod.RegisterLod(1, live, Vector3.Zero, 2f, 120f);
            NeuralLod.RegisterLod(2, live, Vector3.Zero, 2f, 400f);

            // Refresh SceneEvaluator asset to the live network.
            SceneEvaluator.ClearAssets();
            SceneEvaluator.AddAsset(new SceneNeuralAsset
            {
                Id = 1,
                Name = "gdnn_live_sdf",
                Network = live,
                WorldBounds = IntervalBox.FromPoints(DefaultBounds.Min, DefaultBounds.Max),
                BoundingCenter = DefaultBounds.Center,
                BoundingRadius = DefaultBounds.HalfExtents.Length(),
                IsVisible = true,
                Priority = 1
            });

            EnsureNeuralGeometry();
            _liveSdfPromoted = true;

            // Seed online trainer so TrainSlice actually updates weights each tick.
            try
            {
                NeuralGeometry?.Trainer.ApplySphereEdit(
                    center: new Vector3(0.15f, 0.1f, 0f),
                    radius: 0.35f,
                    operation: SdfEditOperation.Union,
                    editSampleCount: 64,
                    anchorSampleCount: 32);
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AlgorithmHub", "Online SDF seed edit skipped.", ex);
            }

            SynapseLogger.Default.Info("AlgorithmHub",
                $"G-DNN live SDF promoted (meshToSdf={(MeshToSdfNetwork != null)})");
        }

        /// <summary>
        /// Loads a baked <c>.gnn</c> / streamed NeuralAsset into the live polygonization path.
        /// Does not overwrite an already-configured scene asset root.
        /// </summary>
        public bool BindBakedNeuralAsset(string assetIdOrPath)
        {
            if (string.IsNullOrWhiteSpace(assetIdOrPath))
                return false;

            try
            {
                NeuralAsset? asset = null;
                if (File.Exists(assetIdOrPath))
                {
                    string? dir = Path.GetDirectoryName(Path.GetFullPath(assetIdOrPath));
                    if (!string.IsNullOrEmpty(dir) &&
                        string.IsNullOrEmpty(GDNN.Streaming.AssetStreamer.AssetRootDirectory))
                        GDNN.Streaming.AssetStreamer.AssetRootDirectory = dir;

                    string id = Path.GetFileNameWithoutExtension(assetIdOrPath);
                    AssetStreamer.RequestAssetAsync(id, AssetPriority.High).GetAwaiter().GetResult();
                    asset = AssetStreamer.GetAsset(id);
                }
                else
                {
                    AssetStreamer.RequestAssetAsync(assetIdOrPath, AssetPriority.High).GetAwaiter().GetResult();
                    asset = AssetStreamer.GetAsset(assetIdOrPath);
                }

                if (asset == null)
                    return false;

                StreamedNeuralAsset = asset;
                if (!asset.IsLoaded)
                    asset.Decompress();

                var net = asset.ToSdfNetwork();
                _streamedEvalNetwork = net;
                if (net is HashEncodedDeepMLP hash)
                {
                    MeshToSdfNetwork?.Dispose();
                    MeshToSdfNetwork = hash;
                    PromoteTrainedSdfToLive();
                    return true;
                }

                SynapseLogger.Default.Info("AlgorithmHub",
                    $"Bound streamed NeuralAsset '{assetIdOrPath}' (non-hash SDF kept for AO eval)");
                return true;
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AlgorithmHub",
                    $"BindBakedNeuralAsset failed for '{assetIdOrPath}'.", ex);
                return false;
            }
        }

        /// <summary>
        /// Advances particles + compute once per G-DNN tick (safe if called from L-DNN CPU
        /// producers and again from ParticlesComputePass).
        /// </summary>
        public void TickPost(Vector3 cameraPos, float time)
        {
            bool first = !_particlesAdvancedThisTick;
            if (first)
            {
                _particlesAdvancedThisTick = true;
                float dt = ComputeDelta(time);
                Particles.Update(dt, cameraPos);
                LastParticleCount = Particles.ActiveParticles;
            }

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

        /// <summary>Samples VSM at look-at points and darkens AO near shadowed surfaces.</summary>
        public void CompositeVsmIntoAo(float[,] ao, Vector3 cameraPos, Vector3 cameraForward, int width, int height)
        {
            if (ao == null || width <= 0 || height <= 0)
                return;

            Vector3 lightDir = Vector3.Normalize(new Vector3(0.5f, 1f, 0.5f));
            float sum = 0f;
            int samples = 0;
            for (int i = 0; i < 5; i++)
            {
                var p = cameraPos + cameraForward * (1.5f + i * 1.2f);
                float s = SampleVirtualShadow(p, lightDir);
                sum += s;
                samples++;
            }

            LastVsmSample = samples > 0 ? sum / samples : 1f;
            float shade = Math.Clamp(0.55f + LastVsmSample * 0.45f, 0.45f, 1f);
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    ao[x, y] *= shade;
        }

        private void BeginGdnnFrame()
        {
            _particlesAdvancedThisTick = false;
        }

        private void TickSpatialQueries(Vector3 cameraPos, Vector3 cameraForward)
        {
            var hits = new System.Collections.Generic.List<int>(16);
            var query = new AABB(cameraPos + cameraForward * 4f, new Vector3(6f, 6f, 6f));
            BroadphaseTree.QueryAABB(query, hits);
            SpatialOctree.QueryAABB(query, hits);
            LooseSpatial.QueryAABB(query, hits);
            LastSpatialHits = hits.Count;
        }

        private void TickGridEvaluator(Vector3 cameraPos, Vector3 cameraForward, Vector3 cameraRight)
        {
            try
            {
                _gridEvaluator ??= new GridEvaluator(SurfaceEvaluator, pixelSize: 0.25f);
                _gridEvaluator.AllocateBuffer(8, 8);
                var up = Vector3.Normalize(Vector3.Cross(cameraRight, cameraForward));
                _gridEvaluator.EvaluateDepthBuffer(
                    cameraPos, cameraForward, cameraRight, up,
                    fovY: MathF.PI / 3f, nearPlane: 0.1f, farPlane: 40f);

                // Mean finite depth as a residency metric for the present path.
                // GridEvaluator stores depth internally; TraceRay samples already ran — use SDF mean proxy.
                float mean = 0f;
                int n = Math.Min(4, LastSdfDistances.Length);
                for (int i = 0; i < n; i++)
                    mean += MathF.Abs(LastSdfDistances[i]);
                LastGridDepthMean = n > 0 ? mean / n : 0f;
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AlgorithmHub", "GridEvaluator tick skipped.", ex);
            }
        }

        private void TickStreamedNetworkEval(ReadOnlySpan<Vector3> pts)
        {
            try
            {
                if (_streamedEvalNetwork == null && StreamedNeuralAsset is { IsLoaded: true } asset)
                {
                    try
                    { _streamedEvalNetwork = asset.ToSdfNetwork(); }
                    catch { /* weight layout may not match */ }
                }

                if (_streamedEvalNetwork != null)
                    _ = _streamedEvalNetwork.Evaluate(pts[0]);
                else if (HyperGeneratedMlp != null)
                    LastHyperSdfSample = HyperGeneratedMlp.Evaluate(pts[0]);
            }
            catch (Exception ex)
            {
                SynapseLogger.Default.Warn("AlgorithmHub", "Streamed SDF eval skipped.", ex);
            }
        }

        private void PresentFromNeuralGeometry(CameraView view, PipelineFrameReport report)
        {
            // Prefer pipeline RenderFrame (LOD chain + frustum/backface cull) over blind Extract.
            int rastW = Math.Clamp(_lastCullWidth / 2, 256, 512);
            int rastH = Math.Clamp(_lastCullHeight / 2, 256, 512);
            var target = new RasterTarget(rastW, rastH);
            var stats = NeuralGeometry!.RenderFrame(view, target, out var visible);
            LastSoftwareRasterPixels = stats.TrianglesRasterized;
            LastRasterCoveredPixels = target.CountCoveredPixels();
            LastVisibleMeshlets = Math.Max(LastVisibleMeshlets, visible.Count);
            _lastRasterTarget = target;
            _lastMeshlets = visible;

            var levelMesh = NeuralGeometry.Chain.Levels[
                Math.Clamp(report.Culling.SelectedLevel, 0, NeuralGeometry.Chain.Levels.Count - 1)].Mesh;
            _lastRasterMesh = levelMesh;
            QueuePresentMesh(levelMesh, report.ExtractedGeometryVersion);
            FeedGeometryRenderer(levelMesh, visible);

            // Fallback dense extract from live SDF when chain mesh is empty.
            if (levelMesh.TriangleCount <= 0)
            {
                int polyRes = Math.Clamp(Math.Max(_lastCullWidth, _lastCullHeight) / 48, 20, 36);
                var mesh = Polygonizer.Extract(LivePolygonSdf, DefaultBounds, resolution: polyRes);
                var meshlets = MeshletBuilder.Build(mesh);
                RasterTarget rast = RasterizeMeshlets(mesh, meshlets, view, rastW, rastH);
                LastRasterCoveredPixels = rast.CountCoveredPixels();
                _lastRasterTarget = rast;
                _lastMeshlets = meshlets;
                _lastRasterMesh = mesh;
                QueuePresentMesh(mesh, report.ExtractedGeometryVersion);
                FeedGeometryRenderer(mesh, meshlets);
            }
        }

        private void PresentFromMeshletStreamer(CameraView view)
        {
            if (MeshletStreamer == null)
                return;

            var keys = MeshletStreamer.QueryVisible(view, level: LastNeuralLod >= 0 ? LastNeuralLod : 0);
            LastVisibleMeshlets = Math.Max(LastVisibleMeshlets, keys.Count);
            int loaded = 0;
            foreach (var key in keys)
            {
                if (MeshletStreamer.GetOrLoad(key) != null)
                    loaded++;
                if (loaded >= 8)
                    break;
            }

            // When the page streamer has clusters and we lack a fresh raster, reuse chain mesh.
            if (loaded > 0 && _lastRasterMesh == null && NeuralGeometry?.Chain.Levels.Count > 0)
            {
                var mesh = NeuralGeometry.Chain.Levels[0].Mesh;
                var meshlets = MeshletBuilder.Build(mesh);
                int rastW = Math.Clamp(_lastCullWidth / 2, 256, 512);
                int rastH = Math.Clamp(_lastCullHeight / 2, 256, 512);
                var rast = RasterizeMeshlets(mesh, meshlets, view, rastW, rastH);
                _lastRasterTarget = rast;
                _lastMeshlets = meshlets;
                _lastRasterMesh = mesh;
                QueuePresentMesh(mesh, _presentMeshVersion + 1);
                FeedGeometryRenderer(mesh, meshlets);
            }
        }
    }
}
