using System;
// ============================================================
// FILE: SceneEvaluator.cs
// PATH: Evaluation/SceneEvaluator.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks;
using GDNN.Core.DataStructures;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Evaluation
{
    /// <summary>
    /// Represents a neural asset in the scene (a mesh defined by a neural SDF).
    /// Renamed to SceneNeuralAsset to avoid collision with GDNN.Core.NeuralNetwork.NeuralAsset.
    /// </summary>
    public sealed class SceneNeuralAsset
    {
        /// <summary>Unique asset identifier.</summary>
        public int Id { get; set; }

        /// <summary>Human-readable name.</summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>The neural network defining this asset's surface.</summary>
        public ISdfNetwork Network { get; init; }

        /// <summary>World-space transform of the asset.</summary>
        public Matrix4x4 Transform { get; set; } = Matrix4x4.Identity;

        /// <summary>Inverse world transform.</summary>
        public Matrix4x4 InverseTransform { get; set; } = Matrix4x4.Identity;

        /// <summary>Axis-aligned bounding box in world space.</summary>
        public IntervalBox WorldBounds { get; set; }

        /// <summary>Bounding sphere radius.</summary>
        public float BoundingRadius { get; set; }

        /// <summary>Bounding sphere center in world space.</summary>
        public Vector3 BoundingCenter { get; set; }

        /// <summary>Is this asset visible in the current frame.</summary>
        public bool IsVisible { get; set; } = true;

        /// <summary>Is this asset currently being evaluated.</summary>
        public bool IsEvaluating { get; set; }

        /// <summary>Priority for evaluation (higher = more important).</summary>
        public int Priority { get; set; }

        /// <summary>Screen-space coverage in pixels (used for priority).</summary>
        public float ScreenCoverage { get; set; }

        /// <summary>LOD level currently in use.</summary>
        public int CurrentLod { get; set; }

        /// <summary>Evaluation time budget in microseconds.</summary>
        public float TimeBudgetMicroseconds { get; set; } = 100.0f;

        /// <summary>Actual evaluation time from last frame.</summary>
        public float LastEvalTimeMicroseconds { get; set; }

        /// <summary>Number of evaluations performed this frame.</summary>
        public int FrameEvalCount { get; set; }

        /// <summary>Custom user data.</summary>
        public object? UserData { get; set; }

        /// <summary>
        /// Updates the inverse transform after modifying the transform.
        /// </summary>
        public void UpdateInverseTransform()
        {
            if (Matrix4x4.Invert(Transform, out var inv))
                InverseTransform = inv;
        }

        /// <summary>
        /// Transforms a point from world space to local space.
        /// </summary>
        public Vector3 WorldToLocal(Vector3 worldPoint)
        {
            return Vector3.Transform(worldPoint, InverseTransform);
        }

        /// <summary>
        /// Transforms a point from local space to world space.
        /// </summary>
        public Vector3 LocalToWorld(Vector3 localPoint)
        {
            return Vector3.Transform(localPoint, Transform);
        }
    }

    /// <summary>
    /// Configuration for scene evaluation.
    /// </summary>
    public sealed class SceneEvaluatorConfig
    {
        /// <summary>Maximum evaluations per frame.</summary>
        public int MaxEvaluationsPerFrame { get; set; } = 10000;

        /// <summary>Frame time budget in milliseconds.</summary>
        public float FrameBudgetMs { get; set; } = 16.0f;

        /// <summary>Enable frustum culling.</summary>
        public bool EnableFrustumCulling { get; set; } = true;

        /// <summary>Enable BVH broad-phase culling for ray and point queries.</summary>
        public bool EnableBroadPhase { get; set; } = true;

        /// <summary>Enable priority-based ordering.</summary>
        public bool EnablePriorityOrdering { get; set; } = true;

        /// <summary>Number of evaluation worker threads.</summary>
        public int WorkerThreads { get; set; } = Environment.ProcessorCount;

        /// <summary>Evaluation batch size for parallel processing.</summary>
        public int BatchSize { get; set; } = 64;

        /// <summary>Screen coverage threshold for culling (fraction).</summary>
        public float MinScreenCoverage { get; set; } = 0.001f;

        /// <summary>Enable GPU/CPU workload splitting.</summary>
        public bool EnableWorkloadSplitting { get; set; }

        /// <summary>GPU workload fraction (0=CPU only, 1=GPU only).</summary>
        public float GpuWorkloadFraction { get; set; } = 0.5f;

        /// <summary>Maximum assets to evaluate in parallel.</summary>
        public int MaxParallelAssets { get; set; } = 16;

        /// <summary>Enable frame timing statistics.</summary>
        public bool EnableFrameTiming { get; set; } = true;
    }

    /// <summary>
    /// Represents a frustum for culling.
    /// </summary>
    public struct Frustum
    {
        /// <summary>Frustum planes (left, right, top, bottom, near, far).</summary>
        public Vector4[] Planes;

        /// <summary>Camera position.</summary>
        public Vector3 CameraPosition;

        /// <summary>Camera forward direction.</summary>
        public Vector3 CameraForward;

        /// <summary>Creates a frustum from view-projection matrix.</summary>
        public static Frustum FromViewProjection(Matrix4x4 viewProjection)
        {
            var frustum = new Frustum
            {
                Planes = new Vector4[6]
            };

            // Left
            frustum.Planes[0] = new Vector4(
                viewProjection.M14 + viewProjection.M11,
                viewProjection.M24 + viewProjection.M21,
                viewProjection.M34 + viewProjection.M31,
                viewProjection.M44 + viewProjection.M41);

            // Right
            frustum.Planes[1] = new Vector4(
                viewProjection.M14 - viewProjection.M11,
                viewProjection.M24 - viewProjection.M21,
                viewProjection.M34 - viewProjection.M31,
                viewProjection.M44 - viewProjection.M41);

            // Top
            frustum.Planes[2] = new Vector4(
                viewProjection.M14 - viewProjection.M12,
                viewProjection.M24 - viewProjection.M22,
                viewProjection.M34 - viewProjection.M32,
                viewProjection.M44 - viewProjection.M42);

            // Bottom
            frustum.Planes[3] = new Vector4(
                viewProjection.M14 + viewProjection.M12,
                viewProjection.M24 + viewProjection.M22,
                viewProjection.M34 + viewProjection.M32,
                viewProjection.M44 + viewProjection.M42);

            // Near
            frustum.Planes[4] = new Vector4(
                viewProjection.M13,
                viewProjection.M23,
                viewProjection.M33,
                viewProjection.M43);

            // Far
            frustum.Planes[5] = new Vector4(
                viewProjection.M14 - viewProjection.M13,
                viewProjection.M24 - viewProjection.M23,
                viewProjection.M34 - viewProjection.M33,
                viewProjection.M44 - viewProjection.M43);

            // Normalize planes
            for (int i = 0; i < 6; i++)
            {
                float len = MathF.Sqrt(
                    frustum.Planes[i].X * frustum.Planes[i].X +
                    frustum.Planes[i].Y * frustum.Planes[i].Y +
                    frustum.Planes[i].Z * frustum.Planes[i].Z);

                if (len > 1e-8f)
                    frustum.Planes[i] /= len;
            }

            return frustum;
        }

        /// <summary>
        /// Tests if a bounding sphere intersects the frustum.
        /// </summary>
        public bool TestSphere(Vector3 center, float radius)
        {
            for (int i = 0; i < 6; i++)
            {
                float dist = Vector3.Dot(new Vector3(Planes[i].X, Planes[i].Y, Planes[i].Z), center)
                    + Planes[i].W;

                if (dist < -radius)
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Tests if an AABB intersects the frustum.
        /// </summary>
        public bool TestAABB(IntervalBox box)
        {
            for (int i = 0; i < 6; i++)
            {
                Vector3 normal = new Vector3(Planes[i].X, Planes[i].Y, Planes[i].Z);

                Vector3 positiveVertex = new Vector3(
                    normal.X >= 0 ? box.Max.X : box.Min.X,
                    normal.Y >= 0 ? box.Max.Y : box.Min.Y,
                    normal.Z >= 0 ? box.Max.Z : box.Min.Z);

                float dist = Vector3.Dot(normal, positiveVertex) + Planes[i].W;
                if (dist < 0)
                    return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Result of a single asset evaluation.
    /// </summary>
    public sealed class AssetEvalResult
    {
        /// <summary>Asset that was evaluated.</summary>
        public SceneNeuralAsset Asset { get; init; }

        /// <summary>Evaluation results (one per query).</summary>
        public SurfaceHit[] Hits { get; set; } = Array.Empty<SurfaceHit>();

        /// <summary>Whether all queries hit.</summary>
        public bool AllHit { get; set; }

        /// <summary>Time taken for this asset's evaluation.</summary>
        public float EvalTimeMicroseconds { get; set; }

        /// <summary>Number of evaluations performed.</summary>
        public int EvalCount { get; set; }
    }

    /// <summary>
    /// Aggregate performance statistics for a frame.
    /// </summary>
    public sealed class FrameStats
    {
        /// <summary>Frame number.</summary>
        public long FrameNumber { get; set; }

        /// <summary>Total evaluation time in milliseconds.</summary>
        public float TotalEvalTimeMs { get; set; }

        /// <summary>Frustum culling time in milliseconds.</summary>
        public float CullingTimeMs { get; set; }

        /// <summary>Number of assets evaluated.</summary>
        public int AssetsEvaluated { get; set; }

        /// <summary>Number of assets culled.</summary>
        public int AssetsCulled { get; set; }

        /// <summary>Total surface evaluations.</summary>
        public int TotalEvaluations { get; set; }

        /// <summary>Asset queries skipped by the BVH broad-phase this frame.</summary>
        public int BroadPhaseCulled { get; set; }

        /// <summary>Total hits.</summary>
        public int TotalHits { get; set; }

        /// <summary>GPU evaluation time (if applicable).</summary>
        public float GpuTimeMs { get; set; }

        /// <summary>CPU evaluation time.</summary>
        public float CpuTimeMs { get; set; }

        /// <summary>Average evaluations per asset.</summary>
        public float AvgEvalsPerAsset => AssetsEvaluated > 0
            ? (float)TotalEvaluations / AssetsEvaluated : 0;

        /// <summary>Hit rate.</summary>
        public float HitRate => TotalEvaluations > 0
            ? (float)TotalHits / TotalEvaluations : 0;

        /// <summary>Whether the frame was within budget.</summary>
        public bool WithinBudget { get; set; }
    }

    /// <summary>
    /// Scene-level evaluation coordinator.
    /// Manages multiple SceneNeuralAsset instances, frustum culling, priority ordering,
    /// GPU/CPU workload distribution, and frame timing.
    /// </summary>
    public sealed class SceneEvaluator : IDisposable
    {
        private readonly SceneEvaluatorConfig _config;
        private readonly List<SceneNeuralAsset> _assets;
        private readonly ReaderWriterLockSlim _assetLock;
        private readonly ConcurrentBag<FrameStats> _frameStats;
        private readonly SurfaceEvaluator _surfaceEvaluator;
        private readonly RayMarcher _rayMarcher;
        private readonly AABBTree<SceneNeuralAsset> _broadPhase;
        private readonly System.Collections.Generic.List<SceneNeuralAsset> _unboundedAssets;
        private readonly object _broadPhaseLock = new();

        private long _frameNumber;
        private bool _disposed;
        private Frustum _currentFrustum;
        private int _totalFrames;
        private long _totalEvalTicks;
        private bool _broadPhaseDirty = true;
        private int _broadPhaseCulledThisFrame;

        /// <summary>Gets the configuration.</summary>
        public SceneEvaluatorConfig Config => _config;

        /// <summary>Gets the underlying surface evaluator.</summary>
        public SurfaceEvaluator SurfaceEvaluator => _surfaceEvaluator;

        /// <summary>Gets the underlying ray marcher.</summary>
        public RayMarcher RayMarcher => _rayMarcher;

        /// <summary>Gets the number of registered assets.</summary>
        public int AssetCount
        {
            get
            {
                _assetLock.EnterReadLock();
                try
                { return _assets.Count; }
                finally { _assetLock.ExitReadLock(); }
            }
        }

        /// <summary>Gets the current frame number.</summary>
        public long FrameNumber => Interlocked.CompareExchange(ref _frameNumber, 0, 0);

        /// <summary>Gets average frame evaluation time in milliseconds.</summary>
        public float AverageFrameTimeMs
        {
            get
            {
                long ticks = Interlocked.CompareExchange(ref _totalEvalTicks, 0, 0);
                int frames = Interlocked.CompareExchange(ref _totalFrames, 0, 0);
                return frames > 0 ? (float)(ticks * 1000.0 / (Stopwatch.Frequency * frames)) : 0;
            }
        }

        /// <summary>
        /// Initializes a new scene evaluator.
        /// </summary>
        /// <param name="config">Configuration. Uses defaults if null.</param>
        public SceneEvaluator(SceneEvaluatorConfig? config = null)
        {
            _config = config ?? new SceneEvaluatorConfig();
            _assets = new List<SceneNeuralAsset>();
            _assetLock = new ReaderWriterLockSlim();
            _frameStats = new ConcurrentBag<FrameStats>();
            _surfaceEvaluator = new SurfaceEvaluator();
            _rayMarcher = new RayMarcher();
            _broadPhase = new AABBTree<SceneNeuralAsset>();
            _unboundedAssets = new System.Collections.Generic.List<SceneNeuralAsset>();
        }

        /// <summary>
        /// Adds a neural asset to the scene.
        /// </summary>
        /// <param name="asset">The asset to add.</param>
        /// <returns>Index of the added asset.</returns>
        public int AddAsset(SceneNeuralAsset asset)
        {
            if (asset == null)
                throw new ArgumentNullException(nameof(asset));
            _assetLock.EnterWriteLock();
            try
            {
                int index = _assets.Count;
                asset.Id = index;
                _assets.Add(asset);
                _broadPhaseDirty = true;
                return index;
            }
            finally { _assetLock.ExitWriteLock(); }
        }

        /// <summary>
        /// Removes a neural asset by index.
        /// </summary>
        /// <param name="index">Asset index.</param>
        public void RemoveAsset(int index)
        {
            _assetLock.EnterWriteLock();
            try
            {
                if (index >= 0 && index < _assets.Count)
                {
                    _assets.RemoveAt(index);
                    _broadPhaseDirty = true;
                }
            }
            finally { _assetLock.ExitWriteLock(); }
        }

        /// <summary>
        /// Gets a neural asset by index.
        /// </summary>
        public SceneNeuralAsset GetAsset(int index)
        {
            _assetLock.EnterReadLock();
            try
            { return _assets[index]; }
            finally { _assetLock.ExitReadLock(); }
        }

        /// <summary>
        /// Clears all assets.
        /// </summary>
        public void ClearAssets()
        {
            _assetLock.EnterWriteLock();
            try
            {
                _assets.Clear();
                _broadPhaseDirty = true;
            }
            finally { _assetLock.ExitWriteLock(); }
        }

        /// <summary>
        /// Performs frustum culling on all assets.
        /// </summary>
        /// <param name="frustum">View frustum.</param>
        /// <returns>Number of visible assets.</returns>
        public int PerformFrustumCulling(Frustum frustum)
        {
            int visibleCount = 0;

            _assetLock.EnterReadLock();
            try
            {
                for (int i = 0; i < _assets.Count; i++)
                {
                    var asset = _assets[i];

                    if (!_config.EnableFrustumCulling)
                    {
                        asset.IsVisible = true;
                        visibleCount++;
                        continue;
                    }

                    bool visible = frustum.TestSphere(asset.BoundingCenter, asset.BoundingRadius);

                    if (!visible)
                    {
                        // Try tighter AABB test
                        visible = frustum.TestAABB(asset.WorldBounds);
                    }

                    asset.IsVisible = visible;
                    if (visible)
                        visibleCount++;
                }
            }
            finally { _assetLock.ExitReadLock(); }

            return visibleCount;
        }

        /// <summary>
        /// Computes screen-space coverage for priority ordering.
        /// </summary>
        /// <param name="cameraPosition">Camera position.</param>
        /// <param name="screenHeight">Screen height in pixels.</param>
        /// <param name="fovY">Vertical field of view.</param>
        public void ComputeScreenCoverage(Vector3 cameraPosition, float screenHeight, float fovY)
        {
            float halfFovTan = MathF.Tan(fovY * 0.5f);

            _assetLock.EnterReadLock();
            try
            {
                for (int i = 0; i < _assets.Count; i++)
                {
                    var asset = _assets[i];
                    float distance = Vector3.Distance(cameraPosition, asset.BoundingCenter);

                    if (distance < 1e-6f)
                    {
                        asset.ScreenCoverage = 1.0f;
                    }
                    else
                    {
                        // Approximate screen-space radius
                        float projectedRadius = asset.BoundingRadius / (distance * halfFovTan);
                        asset.ScreenCoverage = MathF.Min(1.0f, projectedRadius * projectedRadius);
                    }
                }
            }
            finally { _assetLock.ExitReadLock(); }
        }

        /// <summary>
        /// Gets assets sorted by evaluation priority.
        /// </summary>
        /// <returns>List of assets sorted by priority (highest first).</returns>
        public System.Collections.Generic.List<SceneNeuralAsset> GetAssetsByPriority()
        {
            _assetLock.EnterReadLock();
            try
            {
                var sorted = new System.Collections.Generic.List<SceneNeuralAsset>();
                for (int i = 0; i < _assets.Count; i++)
                {
                    if (_assets[i].IsVisible)
                        sorted.Add(_assets[i]);
                }

                if (_config.EnablePriorityOrdering)
                {
                    sorted.Sort((a, b) =>
                    {
                        // Primary sort by screen coverage, secondary by priority
                        int coverageComp = b.ScreenCoverage.CompareTo(a.ScreenCoverage);
                        if (coverageComp != 0)
                            return coverageComp;
                        return b.Priority.CompareTo(a.Priority);
                    });
                }

                return sorted;
            }
            finally { _assetLock.ExitReadLock(); }
        }

        /// <summary>
        /// Computes the distance from a point to an asset's broad-phase bounds
        /// (0 when inside). This is a conservative lower bound of the asset's SDF.
        /// </summary>
        private static float DistanceToAssetBounds(Vector3 point, SceneNeuralAsset asset)
        {
            var box = asset.WorldBounds;
            if (box.Size.LengthSquared() > 1e-12f)
            {
                Vector3 d = Vector3.Max(Vector3.Abs(point - box.Center) - box.HalfExtents, Vector3.Zero);
                return d.Length();
            }

            if (asset.BoundingRadius > 0)
                return MathF.Max(0, Vector3.Distance(point, asset.BoundingCenter) - asset.BoundingRadius);

            return 0; // No usable bounds: never skip this asset.
        }

        /// <summary>
        /// Evaluates a single point against all visible assets.
        /// </summary>
        /// <param name="point">World-space point.</param>
        /// <returns>Closest hit result.</returns>
        public SurfaceHit EvaluatePoint(Vector3 point)
        {
            var bestHit = new SurfaceHit { DidHit = false, Distance = float.MaxValue };
            bool broadPhase = _config.EnableBroadPhase;

            _assetLock.EnterReadLock();
            try
            {
                for (int i = 0; i < _assets.Count; i++)
                {
                    var asset = _assets[i];
                    if (!asset.IsVisible)
                        continue;

                    if (broadPhase && bestHit.DidHit &&
                        DistanceToAssetBounds(point, asset) > bestHit.SdfValue)
                    {
                        // The asset's surface cannot be closer than its bounds.
                        Interlocked.Increment(ref _broadPhaseCulledThisFrame);
                        continue;
                    }

                    Vector3 localPoint = asset.WorldToLocal(point);
                    float sdf = asset.Network.Evaluate(localPoint);

                    // Apply transform scale approximation
                    float scale = ExtractUniformScale(asset.Transform);
                    float worldSdf = sdf * scale;

                    if (worldSdf < bestHit.SdfValue || !bestHit.DidHit)
                    {
                        Vector3 localGrad = asset.Network.ComputeGradient(localPoint);
                        Vector3 worldNormal = Vector3.Normalize(Vector3.TransformNormal(localGrad, asset.Transform));

                        bestHit = new SurfaceHit
                        {
                            DidHit = true,
                            HitPoint = point,
                            Normal = worldNormal,
                            SdfValue = worldSdf,
                            Distance = 0,
                            LodLevel = asset.CurrentLod,
                            AssetId = asset.Id
                        };
                    }
                }
            }
            finally { _assetLock.ExitReadLock(); }

            return bestHit;
        }

        /// <summary>
        /// Traces a ray against all visible assets.
        /// </summary>
        /// <param name="ray">The ray to trace.</param>
        /// <param name="cameraPosition">Camera position for LOD selection.</param>
        /// <returns>Best (closest) hit result.</returns>
        public SurfaceHit TraceRay(TracingRay ray, Vector3 cameraPosition)
        {
            var bestHit = new SurfaceHit { DidHit = false, Distance = float.MaxValue };

            IReadOnlyList<(SceneNeuralAsset Asset, float Entry)> candidates =
                _config.EnableBroadPhase
                    ? GatherRayCandidates(ray)
                    : GatherAllVisible();

            foreach (var (asset, entry) in candidates)
            {
                // Assets are sorted by AABB entry distance: once the best hit is
                // closer than the next box entry, no remaining asset can win.
                if (bestHit.DidHit && bestHit.Distance < entry)
                    break;

                // Transform ray to local space
                Vector3 localOrigin = asset.WorldToLocal(ray.Origin);
                Vector3 localDir = Vector3.TransformNormal(ray.Direction, asset.InverseTransform);
                float localScale = ExtractUniformScale(asset.Transform);
                float localMaxDist = ray.MaxDistance / MathF.Max(localScale, 1e-6f);

                var localRay = new TracingRay(localOrigin, localDir, localMaxDist);
                var hit = _rayMarcher.Trace(asset.Network, localRay);

                if (hit.DidHit && hit.Distance < bestHit.Distance / localScale)
                {
                    // Transform back to world space
                    hit.HitPoint = asset.LocalToWorld(hit.HitPoint);
                    hit.Normal = Vector3.Normalize(Vector3.TransformNormal(hit.Normal, asset.Transform));
                    hit.Distance = Vector3.Distance(ray.Origin, hit.HitPoint);
                    hit.LodLevel = asset.CurrentLod;
                    hit.AssetId = asset.Id;
                    bestHit = hit;
                }

                asset.FrameEvalCount++;
            }

            return bestHit;
        }

        /// <summary>
        /// Broad-phase: queries the BVH for visible assets whose bounds the ray
        /// enters, sorted by entry distance. Assets without usable bounds are
        /// always included (entry 0).
        /// </summary>
        private System.Collections.Generic.List<(SceneNeuralAsset Asset, float Entry)> GatherRayCandidates(TracingRay ray)
        {
            var candidates = new System.Collections.Generic.List<(SceneNeuralAsset Asset, float Entry)>();
            int visibleTotal = 0;

            lock (_broadPhaseLock)
            {
                EnsureBroadPhaseLocked();

                _assetLock.EnterReadLock();
                try
                {
                    for (int i = 0; i < _assets.Count; i++)
                        if (_assets[i].IsVisible)
                            visibleTotal++;
                }
                finally { _assetLock.ExitReadLock(); }

                var rayHits = new System.Collections.Generic.List<RayHit<SceneNeuralAsset>>();
                _broadPhase.QueryRay(
                    new Ray(ray.Origin, ray.Direction), ray.MaxDistance, rayHits);

                foreach (var rayHit in rayHits)
                {
                    if (rayHit.Item.IsVisible)
                        candidates.Add((rayHit.Item, rayHit.Distance));
                }

                foreach (var asset in _unboundedAssets)
                {
                    if (asset.IsVisible)
                        candidates.Add((asset, 0f));
                }
            }

            candidates.Sort((a, b) => a.Item2.CompareTo(b.Item2));
            Interlocked.Add(ref _broadPhaseCulledThisFrame,
                Math.Max(0, visibleTotal - candidates.Count));
            return candidates;
        }

        /// <summary>Fallback path: all visible assets by priority, entry 0.</summary>
        private System.Collections.Generic.List<(SceneNeuralAsset Asset, float Entry)> GatherAllVisible()
        {
            var visibleAssets = GetAssetsByPriority();
            var candidates = new System.Collections.Generic.List<(SceneNeuralAsset Asset, float Entry)>(visibleAssets.Count);
            foreach (var asset in visibleAssets)
                candidates.Add((asset, 0f));
            return candidates;
        }

        /// <summary>
        /// Rebuilds the broad-phase BVH from asset bounds if it is out of date.
        /// Must be called while holding <see cref="_broadPhaseLock"/>.
        /// </summary>
        private void EnsureBroadPhaseLocked()
        {
            if (!_broadPhaseDirty)
                return;

            _assetLock.EnterReadLock();
            try
            {
                _unboundedAssets.Clear();
                var items = new System.Collections.Generic.List<(SceneNeuralAsset Item, AABB Bounds)>(_assets.Count);

                for (int i = 0; i < _assets.Count; i++)
                {
                    var asset = _assets[i];
                    if (asset.WorldBounds.Size.LengthSquared() > 1e-12f)
                    {
                        items.Add((asset, new AABB(asset.WorldBounds.Center, asset.WorldBounds.HalfExtents)));
                    }
                    else if (asset.BoundingRadius > 0)
                    {
                        items.Add((asset, new AABB(asset.BoundingCenter, new Vector3(asset.BoundingRadius))));
                    }
                    else
                    {
                        _unboundedAssets.Add(asset);
                    }
                }

                _broadPhase.Rebuild(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(items));
                _broadPhaseDirty = false;
            }
            finally { _assetLock.ExitReadLock(); }
        }

        /// <summary>
        /// Batch traces multiple rays against the scene.
        /// </summary>
        /// <param name="rays">Rays to trace.</param>
        /// <param name="hits">Output hit results.</param>
        /// <param name="cameraPosition">Camera position.</param>
        public void BatchTraceRays(ReadOnlySpan<TracingRay> rays, Span<SurfaceHit> hits,
            Vector3 cameraPosition)
        {
            if (rays.Length != hits.Length)
                throw new ArgumentException("Rays and hits must have the same length.");

            int batchSize = _config.BatchSize;

            int batchCount = (rays.Length + batchSize - 1) / batchSize;
            for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
            {
                int start = batchIdx * batchSize;
                int end = Math.Min(start + batchSize, rays.Length);

                for (int i = start; i < end; i++)
                {
                    hits[i] = TraceRay(rays[i], cameraPosition);
                }
            }
        }

        /// <summary>
        /// Begins a new frame. Updates timing and resets per-frame counters.
        /// </summary>
        /// <param name="viewProjection">View-projection matrix for frustum.</param>
        /// <param name="cameraPosition">Camera world position.</param>
        /// <param name="cameraForward">Camera forward direction.</param>
        public void BeginFrame(Matrix4x4 viewProjection, Vector3 cameraPosition, Vector3 cameraForward)
        {
            Interlocked.Increment(ref _frameNumber);
            Interlocked.Exchange(ref _broadPhaseCulledThisFrame, 0);

            _currentFrustum = Frustum.FromViewProjection(viewProjection);
            _currentFrustum.CameraPosition = cameraPosition;
            _currentFrustum.CameraForward = cameraForward;

            // Reset per-frame counters
            _assetLock.EnterReadLock();
            try
            {
                for (int i = 0; i < _assets.Count; i++)
                {
                    _assets[i].FrameEvalCount = 0;
                    _assets[i].IsEvaluating = false;
                }
            }
            finally { _assetLock.ExitReadLock(); }

            // Perform frustum culling
            PerformFrustumCulling(_currentFrustum);
        }

        /// <summary>
        /// Ends the frame and records statistics.
        /// </summary>
        /// <param name="totalEvalTimeMs">Total evaluation time this frame.</param>
        /// <returns>Frame statistics.</returns>
        public FrameStats EndFrame(float totalEvalTimeMs)
        {
            var stats = new FrameStats
            {
                FrameNumber = FrameNumber,
                TotalEvalTimeMs = totalEvalTimeMs,
                WithinBudget = totalEvalTimeMs <= _config.FrameBudgetMs,
                BroadPhaseCulled = Interlocked.CompareExchange(ref _broadPhaseCulledThisFrame, 0, 0)
            };

            _assetLock.EnterReadLock();
            try
            {
                int totalEvals = 0;
                int visibleAssets = 0;

                for (int i = 0; i < _assets.Count; i++)
                {
                    var asset = _assets[i];
                    totalEvals += asset.FrameEvalCount;
                    asset.LastEvalTimeMicroseconds = asset.TimeBudgetMicroseconds;

                    if (asset.IsVisible)
                    {
                        visibleAssets++;
                        stats.AssetsEvaluated++;
                    }
                    else
                    {
                        stats.AssetsCulled++;
                    }
                }

                stats.TotalEvaluations = totalEvals;
            }
            finally { _assetLock.ExitReadLock(); }

            _frameStats.Add(stats);
            Interlocked.Add(ref _totalEvalTicks,
                (long)(totalEvalTimeMs / 1000.0 * Stopwatch.Frequency));
            Interlocked.Increment(ref _totalFrames);

            return stats;
        }

        /// <summary>
        /// Gets recent frame statistics.
        /// </summary>
        /// <param name="count">Number of recent frames.</param>
        /// <returns>Array of frame stats.</returns>
        public FrameStats[] GetRecentStats(int count = 60)
        {
            var allStats = _frameStats.ToArray();
            int start = Math.Max(0, allStats.Length - count);
            var result = new FrameStats[allStats.Length - start];
            Array.Copy(allStats, start, result, 0, result.Length);
            return result;
        }

        /// <summary>
        /// Computes aggregate statistics across all frames.
        /// </summary>
        /// <returns>Aggregate stats string.</returns>
        public string GetAggregateStats()
        {
            var stats = GetRecentStats(1000);
            if (stats.Length == 0)
                return "No frame data available.";

            float totalTime = 0;
            float minTime = float.MaxValue;
            float maxTime = float.MinValue;
            int totalEvals = 0;
            int totalHits = 0;
            int withinBudget = 0;

            for (int i = 0; i < stats.Length; i++)
            {
                totalTime += stats[i].TotalEvalTimeMs;
                minTime = MathF.Min(minTime, stats[i].TotalEvalTimeMs);
                maxTime = MathF.Max(maxTime, stats[i].TotalEvalTimeMs);
                totalEvals += stats[i].TotalEvaluations;
                totalHits += stats[i].TotalHits;
                if (stats[i].WithinBudget)
                    withinBudget++;
            }

            return $"Frames={stats.Length} AvgMs={totalTime / stats.Length:F2} " +
                   $"MinMs={minTime:F2} MaxMs={maxTime:F2} " +
                   $"Evals={totalEvals} Hits={totalHits} " +
                   $"BudgetOk={100.0f * withinBudget / stats.Length:F1}%";
        }

        /// <summary>
        /// Adjusts asset priorities based on screen coverage and budget.
        /// </summary>
        /// <param name="cameraPosition">Camera position.</param>
        /// <param name="screenHeight">Screen height.</param>
        /// <param name="fovY">Field of view.</param>
        /// <param name="budgetMs">Time budget in milliseconds.</param>
        public void AdjustPriorities(Vector3 cameraPosition, float screenHeight, float fovY, float budgetMs)
        {
            ComputeScreenCoverage(cameraPosition, screenHeight, fovY);

            _assetLock.EnterReadLock();
            try
            {
                for (int i = 0; i < _assets.Count; i++)
                {
                    var asset = _assets[i];

                    // Compute priority based on screen coverage and distance
                    float distance = Vector3.Distance(cameraPosition, asset.BoundingCenter);
                    float coverageScore = asset.ScreenCoverage * 100.0f;
                    float distanceScore = 1.0f / (1.0f + distance * 0.01f);
                    float timePressure = asset.LastEvalTimeMicroseconds > asset.TimeBudgetMicroseconds
                        ? 0.5f : 1.0f;

                    asset.Priority = (int)(coverageScore * distanceScore * timePressure);

                    // Adjust LOD based on budget pressure
                    float budgetUsage = _surfaceEvaluator.AverageEvalTimeMicroseconds /
                        (budgetMs * 1000.0f / MathF.Max(1, _assets.Count));

                    if (budgetUsage > 1.0f && asset.CurrentLod > 0)
                    {
                        asset.CurrentLod--;
                    }
                    else if (budgetUsage < 0.5f && asset.CurrentLod < 3)
                    {
                        asset.CurrentLod++;
                    }
                }
            }
            finally { _assetLock.ExitReadLock(); }
        }

        /// <summary>
        /// Evaluates all visible assets against a grid of points.
        /// </summary>
        /// <param name="points">Query points.</param>
        /// <param name="results">Output SDF values.</param>
        /// <param name="cameraPosition">Camera position.</param>
        public void EvaluatePointGrid(ReadOnlySpan<Vector3> points, Span<float> results,
            Vector3 cameraPosition)
        {
            if (points.Length != results.Length)
                throw new ArgumentException("Points and results must have the same length.");

            bool broadPhase = _config.EnableBroadPhase;

            _assetLock.EnterReadLock();
            try
            {
                for (int p = 0; p < points.Length; p++)
                {
                    float minSdf = float.MaxValue;

                    for (int a = 0; a < _assets.Count; a++)
                    {
                        var asset = _assets[a];
                        if (!asset.IsVisible)
                            continue;

                        if (broadPhase)
                        {
                            // The bounds distance is a lower bound of this
                            // asset's SDF: skip evaluation when it cannot
                            // improve the running minimum, but keep it as a
                            // conservative far-field estimate.
                            float boundsDist = DistanceToAssetBounds(points[p], asset);
                            if (boundsDist > 0 && boundsDist >= minSdf)
                            {
                                Interlocked.Increment(ref _broadPhaseCulledThisFrame);
                                continue;
                            }
                        }

                        Vector3 localPoint = asset.WorldToLocal(points[p]);
                        float sdf = asset.Network.Evaluate(localPoint);
                        float scale = ExtractUniformScale(asset.Transform);
                        float worldSdf = sdf * scale;

                        if (worldSdf < minSdf)
                            minSdf = worldSdf;
                    }

                    results[p] = minSdf;
                }
            }
            finally { _assetLock.ExitReadLock(); }
        }

        /// <summary>
        /// Finds the closest asset to a world-space point.
        /// </summary>
        /// <param name="point">Query point.</param>
        /// <returns>Index of the closest asset, or -1 if none.</returns>
        public int FindClosestAsset(Vector3 point)
        {
            int closestIdx = -1;
            float closestDist = float.MaxValue;

            _assetLock.EnterReadLock();
            try
            {
                for (int i = 0; i < _assets.Count; i++)
                {
                    var asset = _assets[i];
                    if (!asset.IsVisible)
                        continue;

                    float dist = Vector3.Distance(point, asset.BoundingCenter);
                    if (dist < asset.BoundingRadius && dist < closestDist)
                    {
                        closestDist = dist;
                        closestIdx = i;
                    }
                }
            }
            finally { _assetLock.ExitReadLock(); }

            return closestIdx;
        }

        /// <summary>
        /// Computes workload distribution between CPU and GPU.
        /// </summary>
        /// <param name="totalAssets">Total visible assets.</param>
        /// <param name="cpuAssets">Output CPU asset count.</param>
        /// <param name="gpuAssets">Output GPU asset count.</param>
        public void ComputeWorkloadDistribution(int totalAssets, out int cpuAssets, out int gpuAssets)
        {
            if (!_config.EnableWorkloadSplitting)
            {
                cpuAssets = totalAssets;
                gpuAssets = 0;
                return;
            }

            gpuAssets = (int)(totalAssets * _config.GpuWorkloadFraction);
            cpuAssets = totalAssets - gpuAssets;

            // Clamp to reasonable ranges
            cpuAssets = Math.Max(0, Math.Min(cpuAssets, _config.MaxParallelAssets));
            gpuAssets = Math.Max(0, Math.Min(gpuAssets, _config.MaxParallelAssets));
        }

        /// <summary>
        /// Updates asset transforms and recomputes bounding volumes.
        /// </summary>
        public void UpdateAssetTransforms()
        {
            _assetLock.EnterReadLock();
            try
            {
                for (int i = 0; i < _assets.Count; i++)
                {
                    var asset = _assets[i];
                    asset.UpdateInverseTransform();

                    // Recompute bounding sphere from transform
                    Vector3 center = asset.LocalToWorld(Vector3.Zero);
                    asset.BoundingCenter = center;

                    // Approximate bounding radius from transform
                    Vector3 right = Vector3.TransformNormal(Vector3.UnitX, asset.Transform);
                    Vector3 up = Vector3.TransformNormal(Vector3.UnitY, asset.Transform);
                    Vector3 forward = Vector3.TransformNormal(Vector3.UnitZ, asset.Transform);
                    float maxScale = MathF.Max(right.Length(), MathF.Max(up.Length(), forward.Length()));
                    asset.BoundingRadius *= maxScale;
                }
            }
            finally { _assetLock.ExitReadLock(); }

            _broadPhaseDirty = true;
        }

        /// <summary>
        /// Extracts uniform scale from a transform matrix.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ExtractUniformScale(Matrix4x4 matrix)
        {
            float sx = new Vector3(matrix.M11, matrix.M12, matrix.M13).Length();
            float sy = new Vector3(matrix.M21, matrix.M22, matrix.M23).Length();
            float sz = new Vector3(matrix.M31, matrix.M32, matrix.M33).Length();
            return (sx + sy + sz) / 3.0f;
        }

        /// <summary>
        /// Disposes the scene evaluator.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _assetLock.Dispose();
            _surfaceEvaluator.Dispose();
            _rayMarcher.Dispose();
            _broadPhase.Dispose();
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// Extension for SurfaceHit to include asset ID.
    /// </summary>
    public static class SurfaceHitExtensions
    {
        /// <summary>
        /// Gets or sets the asset ID for a hit.
        /// Stored in the UserData field of the original struct.
        /// </summary>
        public static int GetAssetId(this SurfaceHit hit) => hit.LodLevel;

        /// <summary>
        /// Creates a copy of SurfaceHit with an asset ID.
        /// </summary>
        public static SurfaceHit WithAssetId(this SurfaceHit hit, int assetId)
        {
            hit.LodLevel = assetId;
            return hit;
        }
    }

    /// <summary>
    /// Represents an extended SurfaceHit with asset ID.
    /// </summary>
    public struct ExtendedSurfaceHit
    {
        /// <summary>Base hit result.</summary>
        public SurfaceHit Hit;

        /// <summary>ID of the asset that was hit.</summary>
        public int AssetId;

        /// <summary>World-space position.</summary>
        public Vector3 WorldPosition;

        /// <summary>Whether this is a valid hit.</summary>
        public bool Valid => Hit.DidHit;

        /// <summary>Creates an extended hit.</summary>
        public ExtendedSurfaceHit(SurfaceHit hit, int assetId)
        {
            Hit = hit;
            AssetId = assetId;
            WorldPosition = hit.HitPoint;
        }
    }
}
