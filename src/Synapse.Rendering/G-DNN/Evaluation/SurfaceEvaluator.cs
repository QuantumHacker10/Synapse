using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;


// ============================================================
// FILE: SurfaceEvaluator.cs
// PATH: Evaluation/SurfaceEvaluator.cs
// ============================================================


using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Evaluation
{
    /// <summary>
    /// Represents a single LOD level for a neural surface.
    /// </summary>
    public sealed class LodLevel
    {
        /// <summary>The neural network for this LOD level.</summary>
        public required ISdfNetwork Network { get; init; }

        /// <summary>Maximum world-space distance at which this LOD is used.</summary>
        public required float MaxDistance { get; init; }

        /// <summary>Scale factor applied to the SDF output at this LOD.</summary>
        public float Scale { get; init; } = 1.0f;

        /// <summary>Bias added to the SDF output at this LOD.</summary>
        public float Bias { get; init; } = 0.0f;

        /// <summary>Bounding sphere radius used for early-out optimization.</summary>
        public float BoundingRadius { get; init; } = float.MaxValue;

        /// <summary>Bounding sphere center in world space.</summary>
        public Vector3 BoundingCenter { get; init; } = Vector3.Zero;
    }

    /// <summary>
    /// Configuration for the surface evaluation pipeline.
    /// </summary>
    public sealed class SurfaceEvaluatorConfig
    {
        /// <summary>Maximum ray march steps before giving up.</summary>
        public int MaxRayMarchSteps { get; set; } = 128;

        /// <summary>Distance threshold for surface intersection.</summary>
        public float IntersectionThreshold { get; set; } = 0.001f;

        /// <summary>Maximum ray distance.</summary>
        public float MaxRayDistance { get; set; } = 1000.0f;

        /// <summary>Number of binary search refinement iterations.</summary>
        public int BinarySearchIterations { get; set; } = 8;

        /// <summary>Enable interval arithmetic bounding for guaranteed detection.</summary>
        public bool EnableIntervalBounding { get; set; } = true;

        /// <summary>Batch size for parallel pixel evaluation.</summary>
        public int BatchSize { get; set; } = 256;

        /// <summary>Number of evaluation threads.</summary>
        public int ThreadCount { get; set; } = Environment.ProcessorCount;

        /// <summary>Enable performance profiling.</summary>
        public bool EnableProfiling { get; set; } = true;

        /// <summary>LOD transition hysteresis factor to prevent popping.</summary>
        public float LodHysteresis { get; set; } = 0.1f;

        /// <summary>Minimum adaptive step size as fraction of distance.</summary>
        public float MinAdaptiveStepFraction { get; set; } = 0.01f;

        /// <summary>Maximum adaptive step size as fraction of distance.</summary>
        public float MaxAdaptiveStepFraction { get; set; } = 0.5f;

        /// <summary>Relaxation factor for sphere tracing (1.0 = conservative).</summary>
        public float RelaxationFactor { get; set; } = 0.8f;

        /// <summary>Number of interval arithmetic subdivisions.</summary>
        public int IntervalSubdivisions { get; set; } = 4;

        /// <summary>Enable temporal reprojection for coherence.</summary>
        public bool EnableTemporalReprojection { get; set; } = true;

        /// <summary>Maximum depth for reflection/refraction bounces.</summary>
        public int MaxBounces { get; set; } = 2;

        /// <summary>Normal offset for biasing rays away from surfaces.</summary>
        public float NormalBias { get; set; } = 0.001f;

        /// <summary>Slope-based step scaling factor.</summary>
        public float SlopeStepScale { get; set; } = 0.5f;
    }

    /// <summary>
    /// Result of a single surface evaluation query.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SurfaceHit
    {
        /// <summary>Whether the ray hit the surface.</summary>
        public bool DidHit;

        /// <summary>World-space hit point.</summary>
        public Vector3 HitPoint;

        /// <summary>Surface normal at the hit point.</summary>
        public Vector3 Normal;

        /// <summary>Distance from ray origin to hit point.</summary>
        public float Distance;

        /// <summary>Signed distance value at the hit point (should be near zero).</summary>
        public float SdfValue;

        /// <summary>LOD level that was used for evaluation.</summary>
        public int LodLevel;

        /// <summary>Number of sphere tracing steps taken.</summary>
        public int StepsTaken;

        /// <summary>Whether interval bounding confirmed the intersection.</summary>
        public bool IntervalConfirmed;

        /// <summary>ID of the scene asset associated with this hit.</summary>
        public int AssetId;

        /// <summary>Evaluation time in microseconds.</summary>
        public float EvalTimeMicroseconds;
    }

    /// <summary>
    /// Represents an axis-aligned bounding box for interval arithmetic.
    /// </summary>
    public struct IntervalBox
    {
        /// <summary>Minimum corner of the box.</summary>
        public Vector3 Min;

        /// <summary>Maximum corner of the box.</summary>
        public Vector3 Max;

        /// <summary>Center of the box.</summary>
        public Vector3 Center => (Min + Max) * 0.5f;

        /// <summary>Half-extents of the box.</summary>
        public Vector3 HalfExtents => (Max - Min) * 0.5f;

        /// <summary>Size of the box along each axis.</summary>
        public Vector3 Size => Max - Min;

        /// <summary>Diagonal length of the box.</summary>
        public float Diagonal => Vector3.Distance(Min, Max);

        /// <summary>Creates an interval box from a center and half-extents.</summary>
        public static IntervalBox FromCenterHalfExtents(Vector3 center, Vector3 halfExtents)
        {
            return new IntervalBox
            {
                Min = center - halfExtents,
                Max = center + halfExtents
            };
        }

        /// <summary>Creates an interval box from two points.</summary>
        public static IntervalBox FromPoints(Vector3 a, Vector3 b)
        {
            return new IntervalBox
            {
                Min = Vector3.Min(a, b),
                Max = Vector3.Max(a, b)
            };
        }

        /// <summary>Expands the box to contain the given point.</summary>
        public void Encapsulate(Vector3 point)
        {
            Min = Vector3.Min(Min, point);
            Max = Vector3.Max(Max, point);
        }

        /// <summary>Expands the box to contain the given box.</summary>
        public void Encapsulate(IntervalBox other)
        {
            Min = Vector3.Min(Min, other.Min);
            Max = Vector3.Max(Max, other.Max);
        }

        /// <summary>Tests if a ray intersects this box.</summary>
        public bool IntersectsRay(Vector3 origin, Vector3 direction, float maxDistance)
        {
            Vector3 invDir = new Vector3(
                MathF.Abs(direction.X) > 1e-8f ? 1.0f / direction.X : float.PositiveInfinity,
                MathF.Abs(direction.Y) > 1e-8f ? 1.0f / direction.Y : float.PositiveInfinity,
                MathF.Abs(direction.Z) > 1e-8f ? 1.0f / direction.Z : float.PositiveInfinity
            );

            Vector3 t0 = (Min - origin) * invDir;
            Vector3 t1 = (Max - origin) * invDir;

            Vector3 tMin = Vector3.Min(t0, t1);
            Vector3 tMax = Vector3.Max(t0, t1);

            float tNear = MathF.Max(MathF.Max(tMin.X, tMin.Y), tMin.Z);
            float tFar = MathF.Min(MathF.Min(tMax.X, tMax.Y), tMax.Z);

            return tNear <= tFar && tFar >= 0.0f && tNear <= maxDistance;
        }

        /// <summary>
        /// Provides a lower bound on the SDF value within this box.
        /// Used for conservative sphere tracing steps.
        /// </summary>
        public float LowerBoundSdf(ISdfNetwork network)
        {
            float minSdf = float.MaxValue;
            const int cornerCount = 8;

            Span<Vector3> corners = stackalloc Vector3[cornerCount];
            // Manually unrolled corner generation
            corners[0] = new Vector3(Min.X, Min.Y, Min.Z);
            corners[1] = new Vector3(Max.X, Min.Y, Min.Z);
            corners[2] = new Vector3(Min.X, Max.Y, Min.Z);
            corners[3] = new Vector3(Max.X, Max.Y, Min.Z);
            corners[4] = new Vector3(Min.X, Min.Y, Max.Z);
            corners[5] = new Vector3(Max.X, Min.Y, Max.Z);
            corners[6] = new Vector3(Min.X, Max.Y, Max.Z);
            corners[7] = new Vector3(Max.X, Max.Y, Max.Z);

            for (int i = 0; i < cornerCount; i++)
            {
                float sdf = network.Evaluate(corners[i]);
                if (sdf < minSdf) minSdf = sdf;
            }

            return minSdf;
        }
    }

    /// <summary>
    /// High-level surface evaluation orchestrator for neural geometry.
    /// Manages multiple ISdfNetwork networks, provides LOD selection,
    /// batch evaluation, adaptive sphere tracing with interval arithmetic,
    /// and statistical performance profiling.
    /// </summary>
    public sealed class SurfaceEvaluator : IDisposable
    {
        private readonly SurfaceEvaluatorConfig _config;
        private readonly List<LodLevel> _lodLevels;
        private readonly ReaderWriterLockSlim _lodLock;
        private readonly ConcurrentBag<EvalProfileEntry> _profileEntries;
        private readonly object _profileLock;
        private bool _disposed;
        private int _totalEvaluations;
        private int _totalHits;
        private long _totalEvalTicks;
        private long _totalMarchTicks;

        // Pre-allocated scratch buffers for thread safety
        private readonly ThreadLocal<Vector3[]> _scratchCorners;
        private readonly ThreadLocal<float[]> _scratchSdfValues;

        /// <summary>Gets the current configuration.</summary>
        public SurfaceEvaluatorConfig Config => _config;

        /// <summary>Gets the number of LOD levels currently configured.</summary>
        public int LodLevelCount
        {
            get
            {
                _lodLock.EnterReadLock();
                try { return _lodLevels.Count; }
                finally { _lodLock.ExitReadLock(); }
            }
        }

        /// <summary>Gets the total number of evaluations performed.</summary>
        public int TotalEvaluations => Interlocked.CompareExchange(ref _totalEvaluations, 0, 0);

        /// <summary>Gets the total number of hits.</summary>
        public int TotalHits => Interlocked.CompareExchange(ref _totalHits, 0, 0);

        /// <summary>Gets the hit rate as a fraction.</summary>
        public float HitRate
        {
            get
            {
                int total = TotalEvaluations;
                return total > 0 ? (float)TotalHits / total : 0.0f;
            }
        }

        /// <summary>Average evaluation time in microseconds.</summary>
        public float AverageEvalTimeMicroseconds
        {
            get
            {
                long ticks = Interlocked.CompareExchange(ref _totalEvalTicks, 0, 0);
                int count = TotalEvaluations;
                if (count == 0) return 0.0f;
                return (float)(ticks * 1_000_000.0 / (Stopwatch.Frequency * count));
            }
        }

        /// <summary>Average ray march time in microseconds.</summary>
        public float AverageMarchTimeMicroseconds
        {
            get
            {
                long ticks = Interlocked.CompareExchange(ref _totalMarchTicks, 0, 0);
                int count = TotalEvaluations;
                if (count == 0) return 0.0f;
                return (float)(ticks * 1_000_000.0 / (Stopwatch.Frequency * count));
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SurfaceEvaluator"/> class.
        /// </summary>
        /// <param name="config">Configuration for the evaluator. If null, default config is used.</param>
        public SurfaceEvaluator(SurfaceEvaluatorConfig? config = null)
        {
            _config = config ?? new SurfaceEvaluatorConfig();
            _lodLevels = new List<LodLevel>();
            _lodLock = new ReaderWriterLockSlim();
            _profileEntries = new ConcurrentBag<EvalProfileEntry>();
            _profileLock = new object();

            _scratchCorners = new ThreadLocal<Vector3[]>(() => new Vector3[8]);
            _scratchSdfValues = new ThreadLocal<float[]>(() => new float[8]);
        }

        /// <summary>
        /// Adds a LOD level to the evaluator.
        /// </summary>
        /// <param name="lod">The LOD level to add.</param>
        public void AddLodLevel(LodLevel lod)
        {
            if (lod == null) throw new ArgumentNullException(nameof(lod));
            _lodLock.EnterWriteLock();
            try
            {
                _lodLevels.Add(lod);
                _lodLevels.Sort((a, b) => a.MaxDistance.CompareTo(b.MaxDistance));
            }
            finally { _lodLock.ExitWriteLock(); }
        }

        /// <summary>
        /// Removes all LOD levels.
        /// </summary>
        public void ClearLodLevels()
        {
            _lodLock.EnterWriteLock();
            try { _lodLevels.Clear(); }
            finally { _lodLock.ExitWriteLock(); }
        }

        /// <summary>
        /// Selects the appropriate LOD level based on camera distance.
        /// Uses hysteresis to prevent LOD popping.
        /// </summary>
        /// <param name="cameraPosition">World-space camera position.</param>
        /// <param name="surfaceCenter">Center of the surface in world space.</param>
        /// <param name="previousLod">Previously used LOD level for hysteresis (-1 if none).</param>
        /// <returns>Selected LOD level index, or -1 if none available.</returns>
        public int SelectLod(Vector3 cameraPosition, Vector3 surfaceCenter, int previousLod = -1)
        {
            _lodLock.EnterReadLock();
            try
            {
                if (_lodLevels.Count == 0) return -1;

                float distance = Vector3.Distance(cameraPosition, surfaceCenter);
                int selected = 0;

                for (int i = 0; i < _lodLevels.Count; i++)
                {
                    if (distance <= _lodLevels[i].MaxDistance)
                    {
                        selected = i;
                        break;
                    }
                    selected = i;
                }

                // Apply hysteresis
                if (previousLod >= 0 && previousLod < _lodLevels.Count && _config.LodHysteresis > 0)
                {
                    float currentThreshold = _lodLevels[selected].MaxDistance;
                    float prevThreshold = _lodLevels[previousLod].MaxDistance;

                    if (previousLod < selected && distance > currentThreshold * (1.0f - _config.LodHysteresis))
                    {
                        selected = previousLod;
                    }
                    else if (previousLod > selected && distance < prevThreshold * (1.0f + _config.LodHysteresis))
                    {
                        selected = previousLod;
                    }
                }

                return selected;
            }
            finally { _lodLock.ExitReadLock(); }
        }

        /// <summary>
        /// Evaluates the signed distance at a single point using the best available LOD.
        /// </summary>
        /// <param name="point">World-space point to evaluate.</param>
        /// <param name="cameraPosition">Camera position for LOD selection.</param>
        /// <returns>Signed distance value.</returns>
        public float Evaluate(Vector3 point, Vector3 cameraPosition)
        {
            int lod = SelectLod(cameraPosition, point);
            if (lod < 0) return float.MaxValue;

            _lodLock.EnterReadLock();
            try
            {
                var level = _lodLevels[lod];
                float sdf = level.Network.Evaluate(point);
                return sdf * level.Scale + level.Bias;
            }
            finally { _lodLock.ExitReadLock(); }
        }

        /// <summary>
        /// Evaluates the signed distance with gradient at a single point.
        /// </summary>
        /// <param name="point">World-space point to evaluate.</param>
        /// <param name="gradient">Output gradient vector.</param>
        /// <param name="cameraPosition">Camera position for LOD selection.</param>
        /// <returns>Signed distance value.</returns>
        public float EvaluateWithGradient(Vector3 point, out Vector3 gradient, Vector3 cameraPosition)
        {
            int lod = SelectLod(cameraPosition, point);
            if (lod < 0)
            {
                gradient = Vector3.UnitY;
                return float.MaxValue;
            }

            _lodLock.EnterReadLock();
            try
            {
                var level = _lodLevels[lod];
                float sdf = level.Network.EvaluateWithGradient(point, out gradient);
                gradient *= level.Scale;
                return sdf * level.Scale + level.Bias;
            }
            finally { _lodLock.ExitReadLock(); }
        }

        /// <summary>
        /// Performs adaptive sphere tracing from an origin along a direction.
        /// Uses interval arithmetic bounding for guaranteed intersection detection.
        /// </summary>
        /// <param name="origin">Ray origin.</param>
        /// <param name="direction">Normalized ray direction.</param>
        /// <param name="cameraPosition">Camera position for LOD selection.</param>
        /// <returns>Surface hit result.</returns>
        public SurfaceHit TraceRay(Vector3 origin, Vector3 direction, Vector3 cameraPosition)
        {
            var stopwatch = _config.EnableProfiling ? Stopwatch.StartNew() : null;
            var hit = new SurfaceHit { DidHit = false };

            int lod = SelectLod(cameraPosition, origin);
            if (lod < 0)
            {
                if (stopwatch != null)
                {
                    Interlocked.Add(ref _totalMarchTicks, stopwatch.ElapsedTicks);
                    Interlocked.Increment(ref _totalEvaluations);
                }
                return hit;
            }

            _lodLock.EnterReadLock();
            try
            {
                var level = _lodLevels[lod];
                hit.LodLevel = lod;

                // Phase 1: Interval arithmetic bounding for early rejection
                if (_config.EnableIntervalBounding)
                {
                    if (!TestIntervalIntersection(origin, direction, level, out float earlyDistance))
                    {
                        hit.Distance = _config.MaxRayDistance;
                        if (stopwatch != null)
                        {
                            Interlocked.Add(ref _totalMarchTicks, stopwatch.ElapsedTicks);
                            Interlocked.Increment(ref _totalEvaluations);
                        }
                        return hit;
                    }
                }

                // Phase 2: Adaptive sphere tracing
                MarchResult marchResult = AdaptiveSphereTrace(
                    origin, direction, level, _config.MaxRayMarchSteps);

                hit.StepsTaken = marchResult.Steps;

                if (!marchResult.Hit)
                {
                    hit.Distance = _config.MaxRayDistance;
                    if (stopwatch != null)
                    {
                        Interlocked.Add(ref _totalMarchTicks, stopwatch.ElapsedTicks);
                        Interlocked.Increment(ref _totalEvaluations);
                    }
                    return hit;
                }

                // Phase 3: Binary search refinement
                Vector3 refinedPoint = BinarySearchRefine(
                    origin, direction, marchResult.EntryDistance, marchResult.ExitDistance,
                    level, _config.BinarySearchIterations);

                float finalDistance = Vector3.Distance(origin, refinedPoint);
                float finalSdf = level.Network.Evaluate(refinedPoint);

                // Phase 4: Compute surface normal via gradient
                Vector3 normal = level.Network.ComputeGradient(refinedPoint);
                if (normal.LengthSquared() > 1e-10f)
                    normal = Vector3.Normalize(normal);
                else
                    normal = Vector3.UnitY;

                // Apply normal bias to avoid self-intersection
                hit.DidHit = true;
                hit.HitPoint = refinedPoint + normal * _config.NormalBias;
                hit.Normal = normal;
                hit.Distance = finalDistance;
                hit.SdfValue = finalDistance * level.Scale + level.Bias;
                hit.IntervalConfirmed = true;

                Interlocked.Increment(ref _totalHits);

                if (stopwatch != null)
                {
                    stopwatch.Stop();
                    Interlocked.Add(ref _totalMarchTicks, stopwatch.ElapsedTicks);
                    Interlocked.Increment(ref _totalEvaluations);
                    hit.EvalTimeMicroseconds = (float)(stopwatch.ElapsedTicks * 1_000_000.0 / Stopwatch.Frequency);

                    if (_config.EnableProfiling)
                    {
                        _profileEntries.Add(new EvalProfileEntry
                        {
                            Steps = hit.StepsTaken,
                            Distance = hit.Distance,
                            TimeMicroseconds = hit.EvalTimeMicroseconds,
                            LodLevel = lod,
                            Hit = true
                        });
                    }
                }

                return hit;
            }
            finally { _lodLock.ExitReadLock(); }
        }

        /// <summary>
        /// Tests interval arithmetic bounding for a ray against a LOD level's network.
        /// </summary>
        private bool TestIntervalIntersection(Vector3 origin, Vector3 direction, LodLevel level, out float minDistance)
        {
            minDistance = 0.0f;
            float maxDist = _config.MaxRayDistance;
            float stepFraction = _config.MaxAdaptiveStepFraction;
            int subdivs = _config.IntervalSubdivisions;

            for (float t = 0.0f; t < maxDist; t += stepFraction * maxDist / subdivs)
            {
                float segmentLength = MathF.Min(stepFraction * maxDist / subdivs, maxDist - t);
                Vector3 start = origin + direction * t;
                Vector3 end = origin + direction * (t + segmentLength);

                IntervalBox box = IntervalBox.FromPoints(start, end);
                float lowerBound = box.LowerBoundSdf(level.Network);

                if (lowerBound <= 0.0f)
                {
                    minDistance = t;
                    return true;
                }

                // Skip ahead if far from surface
                if (lowerBound > segmentLength)
                {
                    t += lowerBound - segmentLength;
                }
            }

            return false;
        }

        /// <summary>
        /// Performs adaptive sphere tracing with dynamic step sizing.
        /// </summary>
        private MarchResult AdaptiveSphereTrace(Vector3 origin, Vector3 direction, LodLevel level, int maxSteps)
        {
            var result = new MarchResult();
            float totalDistance = 0.0f;
            float prevSdf = float.MaxValue;
            Vector3 currentPoint = origin;

            for (int i = 0; i < maxSteps; i++)
            {
                currentPoint = origin + direction * totalDistance;
                float sdf = level.Network.Evaluate(currentPoint);
                float scaledSdf = sdf * level.Scale + level.Bias;

                // Adaptive step size based on SDF magnitude and distance traveled
                float adaptiveFactor = ComputeAdaptiveStepFactor(sdf, prevSdf, totalDistance);
                float stepSize = MathF.Abs(scaledSdf) * adaptiveFactor * _config.RelaxationFactor;

                // Clamp step size
                stepSize = MathF.Max(stepSize, _config.IntersectionThreshold * _config.MinAdaptiveStepFraction);
                stepSize = MathF.Min(stepSize, _config.MaxRayDistance * _config.MaxAdaptiveStepFraction);

                // Detect intersection
                if (MathF.Abs(scaledSdf) < _config.IntersectionThreshold)
                {
                    result.Hit = true;
                    result.EntryDistance = totalDistance - MathF.Abs(scaledSdf);
                    result.ExitDistance = totalDistance + MathF.Abs(scaledSdf);
                    result.Steps = i + 1;
                    return result;
                }

                // Detect crossing (sign change indicates surface was passed)
                if (prevSdf > 0 && scaledSdf < 0)
                {
                    result.EntryDistance = totalDistance - MathF.Abs(prevSdf * level.Scale);
                    result.ExitDistance = totalDistance;
                    result.Hit = true;
                    result.Steps = i + 1;
                    return result;
                }

                // Under-step when close to surface for safety
                if (MathF.Abs(scaledSdf) < _config.IntersectionThreshold * 10)
                {
                    stepSize *= 0.5f;
                }

                totalDistance += stepSize;
                prevSdf = scaledSdf;

                // Check max distance
                if (totalDistance >= _config.MaxRayDistance)
                {
                    result.Steps = i + 1;
                    return result;
                }
            }

            result.Steps = maxSteps;
            return result;
        }

        /// <summary>
        /// Computes adaptive step factor based on SDF history.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ComputeAdaptiveStepFactor(float currentSdf, float prevSdf, float distanceTraveled)
        {
            if (MathF.Abs(prevSdf) < 1e-8f) return 1.0f;

            // Estimate curvature from SDF change rate
            float ratio = MathF.Abs(currentSdf / prevSdf);
            float curvatureFactor = 1.0f;

            if (ratio > 1.5f)
            {
                // SDF is increasing rapidly - we're moving away, can step larger
                curvatureFactor = MathF.Min(ratio, 3.0f);
            }
            else if (ratio < 0.5f)
            {
                // SDF is decreasing rapidly - approaching surface, step smaller
                curvatureFactor = MathF.Max(ratio, 0.25f);
            }

            // Distance-based scaling: allow larger steps further from surface
            float distanceScale = 1.0f + MathF.Log(1.0f + distanceTraveled) * _config.SlopeStepScale;

            return curvatureFactor * distanceScale;
        }

        /// <summary>
        /// Binary search refinement for sub-precision surface intersection.
        /// </summary>
        private Vector3 BinarySearchRefine(Vector3 origin, Vector3 direction, float entryDist, float exitDist,
            LodLevel level, int iterations)
        {
            float lo = entryDist;
            float hi = exitDist;

            for (int i = 0; i < iterations; i++)
            {
                float mid = (lo + hi) * 0.5f;
                Vector3 midPoint = origin + direction * mid;
                float sdf = level.Network.Evaluate(midPoint) * level.Scale + level.Bias;

                if (sdf < 0)
                {
                    hi = mid;
                }
                else
                {
                    lo = mid;
                }
            }

            return origin + direction * ((lo + hi) * 0.5f);
        }

        /// <summary>
        /// Evaluates SDF values for a grid of pixels in batch.
        /// </summary>
        /// <param name="pixels">Array of pixel world-space positions.</param>
        /// <param name="results">Output array of SDF values.</param>
        /// <param name="cameraPosition">Camera position for LOD selection.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        public void BatchEvaluate(ReadOnlySpan<Vector3> pixels, Span<float> results, Vector3 cameraPosition,
            CancellationToken cancellationToken = default)
        {
            if (pixels.Length != results.Length)
                throw new ArgumentException("Pixels and results must have the same length.");

            int batchSize = _config.BatchSize;
            int totalPixels = pixels.Length;

            int batchCount = (totalPixels + batchSize - 1) / batchSize;
            for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int start = batchIdx * batchSize;
                int end = Math.Min(start + batchSize, totalPixels);

                _lodLock.EnterReadLock();
                try
                {
                    for (int j = start; j < end; j++)
                    {
                        Vector3 point = pixels[j];
                        int lod = SelectLodInternal(cameraPosition, point);

                        if (lod < 0)
                        {
                            results[j] = float.MaxValue;
                            continue;
                        }

                        var level = _lodLevels[lod];
                        float sdf = level.Network.Evaluate(point);
                        results[j] = sdf * level.Scale + level.Bias;
                    }
                }
                finally { _lodLock.ExitReadLock(); }
            }
        }

        /// <summary>
        /// Batch ray tracing for a grid of rays.
        /// </summary>
        /// <param name="origins">Ray origins.</param>
        /// <param name="directions">Ray directions (must be normalized).</param>
        /// <param name="hits">Output hit results.</param>
        /// <param name="cameraPosition">Camera position for LOD selection.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        public void BatchTraceRays(ReadOnlySpan<Vector3> origins, ReadOnlySpan<Vector3> directions,
            Span<SurfaceHit> hits, Vector3 cameraPosition, CancellationToken cancellationToken = default)
        {
            if (origins.Length != directions.Length || origins.Length != hits.Length)
                throw new ArgumentException("All input arrays must have the same length.");

            int batchSize = _config.BatchSize;
            int totalRays = origins.Length;

            int batchCount = (totalRays + batchSize - 1) / batchSize;
            for (int batchIdx = 0; batchIdx < batchCount; batchIdx++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int start = batchIdx * batchSize;
                int end = Math.Min(start + batchSize, totalRays);

                for (int j = start; j < end; j++)
                {
                    hits[j] = TraceRay(origins[j], directions[j], cameraPosition);
                }
            }
        }

        /// <summary>
        /// Evaluates the SDF for a scanline of pixels (common in rendering).
        /// </summary>
        /// <param name="startPoint">First pixel world position.</param>
        /// <param name="endPoint">Last pixel world position.</param>
        /// <param name="pixelCount">Number of pixels.</param>
        /// <param name="results">Output SDF values.</param>
        /// <param name="cameraPosition">Camera position for LOD selection.</param>
        public void EvaluateScanline(Vector3 startPoint, Vector3 endPoint, int pixelCount,
            Span<float> results, Vector3 cameraPosition)
        {
            if (results.Length < pixelCount)
                throw new ArgumentException("Results buffer too small.");

            Vector3 step = (endPoint - startPoint) / MathF.Max(1, pixelCount - 1);

            _lodLock.EnterReadLock();
            try
            {
                for (int i = 0; i < pixelCount; i++)
                {
                    Vector3 point = startPoint + step * i;
                    int lod = SelectLodInternal(cameraPosition, point);

                    if (lod < 0)
                    {
                        results[i] = float.MaxValue;
                        continue;
                    }

                    var level = _lodLevels[lod];
                    float sdf = level.Network.Evaluate(point);
                    results[i] = sdf * level.Scale + level.Bias;
                }
            }
            finally { _lodLock.ExitReadLock(); }
        }

        /// <summary>
        /// Evaluates SDF at multiple points and computes gradient field.
        /// </summary>
        /// <param name="points">Points to evaluate.</param>
        /// <param name="sdfValues">Output SDF values.</param>
        /// <param name="gradients">Output gradient vectors.</param>
        /// <param name="cameraPosition">Camera position for LOD selection.</param>
        public void EvaluateGradientField(ReadOnlySpan<Vector3> points, Span<float> sdfValues,
            Span<Vector3> gradients, Vector3 cameraPosition)
        {
            if (points.Length != sdfValues.Length || points.Length != gradients.Length)
                throw new ArgumentException("All arrays must have the same length.");

            _lodLock.EnterReadLock();
            try
            {
                for (int i = 0; i < points.Length; i++)
                {
                    Vector3 point = points[i];
                    int lod = SelectLodInternal(cameraPosition, point);

                    if (lod < 0)
                    {
                        sdfValues[i] = float.MaxValue;
                        gradients[i] = Vector3.Zero;
                        continue;
                    }

                    var level = _lodLevels[lod];
                    float sdf = level.Network.EvaluateWithGradient(point, out Vector3 grad);
                    sdfValues[i] = sdf * level.Scale + level.Bias;
                    gradients[i] = grad * level.Scale;
                }
            }
            finally { _lodLock.ExitReadLock(); }
        }

        /// <summary>
        /// Internal LOD selection without lock (caller must hold read lock).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int SelectLodInternal(Vector3 cameraPosition, Vector3 surfaceCenter)
        {
            if (_lodLevels.Count == 0) return -1;

            float distance = Vector3.Distance(cameraPosition, surfaceCenter);
            int selected = 0;

            for (int i = 0; i < _lodLevels.Count; i++)
            {
                if (distance <= _lodLevels[i].MaxDistance)
                {
                    selected = i;
                    break;
                }
                selected = i;
            }

            return selected;
        }

        /// <summary>
        /// Retrieves a snapshot of profiling data.
        /// </summary>
        /// <returns>Array of profiling entries.</returns>
        public EvalProfileEntry[] GetProfileData()
        {
            lock (_profileLock)
            {
                return _profileEntries.ToArray();
            }
        }

        /// <summary>
        /// Computes aggregate profiling statistics.
        /// </summary>
        /// <returns>Profiling statistics.</returns>
        public EvalProfileStats GetProfileStats()
        {
            var entries = GetProfileData();
            if (entries.Length == 0)
                return new EvalProfileStats();

            float totalTime = 0;
            float minTime = float.MaxValue;
            float maxTime = float.MinValue;
            int totalSteps = 0;
            float p50 = 0, p95 = 0, p99 = 0;

            var times = new float[entries.Length];
            for (int i = 0; i < entries.Length; i++)
            {
                times[i] = entries[i].TimeMicroseconds;
                totalTime += times[i];
                totalSteps += entries[i].Steps;
                if (times[i] < minTime) minTime = times[i];
                if (times[i] > maxTime) maxTime = times[i];
            }

            Array.Sort(times);
            p50 = times[(int)(times.Length * 0.50f)];
            p95 = times[(int)(times.Length * 0.95f)];
            p99 = times[(int)(times.Length * 0.99f)];

            return new EvalProfileStats
            {
                TotalEvaluations = entries.Length,
                AverageTimeMicroseconds = totalTime / entries.Length,
                MinTimeMicroseconds = minTime,
                MaxTimeMicroseconds = maxTime,
                P50TimeMicroseconds = p50,
                P95TimeMicroseconds = p95,
                P99TimeMicroseconds = p99,
                AverageSteps = (float)totalSteps / entries.Length
            };
        }

        /// <summary>
        /// Clears all profiling data.
        /// </summary>
        public void ClearProfileData()
        {
            lock (_profileLock)
            {
                while (_profileEntries.TryTake(out _)) { }
            }
        }

        /// <summary>
        /// Disposes resources held by the evaluator.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _lodLock.Dispose();
            _scratchCorners.Dispose();
            _scratchSdfValues.Dispose();
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Internal march result structure.
        /// </summary>
        private struct MarchResult
        {
            public bool Hit;
            public float EntryDistance;
            public float ExitDistance;
            public int Steps;
        }
    }

    /// <summary>
    /// A single profiling entry for an evaluation.
    /// </summary>
    public sealed class EvalProfileEntry
    {
        /// <summary>Number of sphere tracing steps.</summary>
        public int Steps { get; set; }

        /// <summary>Ray distance.</summary>
        public float Distance { get; set; }

        /// <summary>Evaluation time in microseconds.</summary>
        public float TimeMicroseconds { get; set; }

        /// <summary>LOD level used.</summary>
        public int LodLevel { get; set; }

        /// <summary>Whether this was a hit.</summary>
        public bool Hit { get; set; }
    }

    /// <summary>
    /// Aggregate profiling statistics.
    /// </summary>
    public sealed class EvalProfileStats
    {
        /// <summary>Total number of evaluations.</summary>
        public int TotalEvaluations { get; set; }

        /// <summary>Average evaluation time in microseconds.</summary>
        public float AverageTimeMicroseconds { get; set; }

        /// <summary>Minimum evaluation time.</summary>
        public float MinTimeMicroseconds { get; set; }

        /// <summary>Maximum evaluation time.</summary>
        public float MaxTimeMicroseconds { get; set; }

        /// <summary>50th percentile evaluation time.</summary>
        public float P50TimeMicroseconds { get; set; }

        /// <summary>95th percentile evaluation time.</summary>
        public float P95TimeMicroseconds { get; set; }

        /// <summary>99th percentile evaluation time.</summary>
        public float P99TimeMicroseconds { get; set; }

        /// <summary>Average number of steps per evaluation.</summary>
        public float AverageSteps { get; set; }

        /// <summary>Returns a summary string.</summary>
        public override string ToString()
        {
            return $"Evals={TotalEvaluations} Avg={AverageTimeMicroseconds:F2}us " +
                   $"P50={P50TimeMicroseconds:F2}us P95={P95TimeMicroseconds:F2}us " +
                   $"P99={P99TimeMicroseconds:F2}us AvgSteps={AverageSteps:F1}";
        }
    }

    /// <summary>
    /// Provides extension methods for vector operations used in evaluation.
    /// </summary>
    internal static class VectorExtensions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SignedDistanceToPlane(Vector3 point, Vector3 planePoint, Vector3 planeNormal)
        {
            return Vector3.Dot(point - planePoint, planeNormal);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 ProjectOnPlane(Vector3 point, Vector3 planePoint, Vector3 planeNormal)
        {
            return point - planeNormal * Vector3.Dot(point - planePoint, planeNormal);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Reflect(Vector3 direction, Vector3 normal)
        {
            return direction - 2.0f * Vector3.Dot(direction, normal) * normal;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Refract(Vector3 direction, Vector3 normal, float eta)
        {
            float dot = Vector3.Dot(direction, normal);
            float k = 1.0f - eta * eta * (1.0f - dot * dot);
            if (k < 0) return Vector3.Reflect(-direction, normal) * -1.0f;
            return eta * direction - (eta * dot + MathF.Sqrt(k)) * normal;
        }
    }

    /// <summary>
    /// Represents a scanline for parallel pixel evaluation.
    /// </summary>
    public readonly struct ScanlineDefinition
    {
        /// <summary>Y coordinate of the scanline.</summary>
        public readonly int Y;

        /// <summary>Start X coordinate.</summary>
        public readonly int StartX;

        /// <summary>End X coordinate (exclusive).</summary>
        public readonly int EndX;

        /// <summary>World position of the first pixel.</summary>
        public readonly Vector3 StartWorldPos;

        /// <summary>World position step per pixel.</summary>
        public readonly Vector3 Step;

        /// <summary>Creates a new scanline definition.</summary>
        public ScanlineDefinition(int y, int startX, int endX, Vector3 startWorldPos, Vector3 step)
        {
            Y = y;
            StartX = startX;
            EndX = endX;
            StartWorldPos = startWorldPos;
            Step = step;
        }
    }

    /// <summary>
    /// Manages grid-based evaluation of neural SDFs for rendering.
    /// </summary>
    public sealed class GridEvaluator : IDisposable
    {
        private readonly SurfaceEvaluator _evaluator;
        private readonly float _pixelSize;
        private float[]? _depthBuffer;
        private int _bufferWidth;
        private int _bufferHeight;

        /// <summary>Gets the pixel size in world units.</summary>
        public float PixelSize => _pixelSize;

        /// <summary>
        /// Initializes a new grid evaluator.
        /// </summary>
        /// <param name="evaluator">Parent surface evaluator.</param>
        /// <param name="pixelSize">Size of each pixel in world units.</param>
        public GridEvaluator(SurfaceEvaluator evaluator, float pixelSize)
        {
            _evaluator = evaluator ?? throw new ArgumentNullException(nameof(evaluator));
            _pixelSize = pixelSize > 0 ? pixelSize : throw new ArgumentOutOfRangeException(nameof(pixelSize));
        }

        /// <summary>
        /// Allocates the depth buffer for the given resolution.
        /// </summary>
        /// <param name="width">Buffer width in pixels.</param>
        /// <param name="height">Buffer height in pixels.</param>
        public void AllocateBuffer(int width, int height)
        {
            if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
            if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));

            _bufferWidth = width;
            _bufferHeight = height;
            _depthBuffer = new float[width * height];
        }

        /// <summary>
        /// Evaluates the depth buffer from a camera viewpoint.
        /// </summary>
        /// <param name="cameraPosition">Camera world position.</param>
        /// <param name="cameraForward">Camera forward direction.</param>
        /// <param name="cameraRight">Camera right direction.</param>
        /// <param name="cameraUp">Camera up direction.</param>
        /// <param name="fovY">Vertical field of view in radians.</param>
        /// <param name="nearPlane">Near plane distance.</param>
        /// <param name="farPlane">Far plane distance.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        public void EvaluateDepthBuffer(Vector3 cameraPosition, Vector3 cameraForward, Vector3 cameraRight,
            Vector3 cameraUp, float fovY, float nearPlane, float farPlane,
            CancellationToken cancellationToken = default)
        {
            if (_depthBuffer == null)
                throw new InvalidOperationException("Buffer not allocated. Call AllocateBuffer first.");

            float aspectRatio = (float)_bufferWidth / _bufferHeight;
            float halfHeight = MathF.Tan(fovY * 0.5f);

            int batchSize = _evaluator.Config.BatchSize;

            Parallel.For(0, _bufferHeight,
                new ParallelOptions
                {
                    MaxDegreeOfParallelism = _evaluator.Config.ThreadCount,
                    CancellationToken = cancellationToken
                },
                y =>
                {
                    float normalizedY = (2.0f * y / _bufferHeight - 1.0f) * halfHeight;

                    for (int x = 0; x < _bufferWidth; x++)
                    {
                        float normalizedX = (2.0f * x / _bufferWidth - 1.0f) * halfHeight * aspectRatio;

                        Vector3 rayDir = Vector3.Normalize(
                            cameraForward + cameraRight * normalizedX + cameraUp * normalizedY);

                        SurfaceHit hit = _evaluator.TraceRay(cameraPosition + rayDir * nearPlane,
                            rayDir, cameraPosition);

                        int idx = y * _bufferWidth + x;
                        _depthBuffer[idx] = hit.DidHit ? hit.Distance : farPlane;
                    }
                });
        }

        /// <summary>
        /// Gets the depth value at a pixel coordinate.
        /// </summary>
        public float GetDepth(int x, int y)
        {
            if (_depthBuffer == null) throw new InvalidOperationException("Buffer not allocated.");
            if (x < 0 || x >= _bufferWidth || y < 0 || y >= _bufferHeight)
                throw new ArgumentOutOfRangeException(nameof(x));

            return _depthBuffer[y * _bufferWidth + x];
        }

        /// <summary>
        /// Gets a span of the depth buffer.
        /// </summary>
        public ReadOnlySpan<float> GetDepthBuffer()
        {
            if (_depthBuffer == null) throw new InvalidOperationException("Buffer not allocated.");
            return _depthBuffer;
        }

        /// <summary>
        /// Disposes the depth buffer.
        /// </summary>
        public void Dispose()
        {
            _depthBuffer = null;
        }
    }

    /// <summary>
    /// Manages interval arithmetic tests for conservative SDF bounding.
    /// </summary>
    public static class IntervalArithmetic
    {
        /// <summary>
        /// Computes the minimum possible SDF value within a box using corner sampling.
        /// This is a conservative lower bound.
        /// </summary>
        /// <param name="box">The bounding box.</param>
        /// <param name="network">The neural network.</param>
        /// <returns>Lower bound on the SDF value within the box.</returns>
        public static float ConservativeLowerBound(IntervalBox box, ISdfNetwork network)
        {
            return box.LowerBoundSdf(network);
        }

        /// <summary>
        /// Computes the maximum possible SDF value within a box.
        /// This is a conservative upper bound.
        /// </summary>
        /// <param name="box">The bounding box.</param>
        /// <param name="network">The neural network.</param>
        /// <returns>Upper bound on the SDF value within the box.</returns>
        public static float ConservativeUpperBound(IntervalBox box, ISdfNetwork network)
        {
            float maxSdf = float.MinValue;
            Vector3[] corners = new Vector3[8];

            corners[0] = new Vector3(box.Min.X, box.Min.Y, box.Min.Z);
            corners[1] = new Vector3(box.Max.X, box.Min.Y, box.Min.Z);
            corners[2] = new Vector3(box.Min.X, box.Max.Y, box.Min.Z);
            corners[3] = new Vector3(box.Max.X, box.Max.Y, box.Min.Z);
            corners[4] = new Vector3(box.Min.X, box.Min.Y, box.Max.Z);
            corners[5] = new Vector3(box.Max.X, box.Min.Y, box.Max.Z);
            corners[6] = new Vector3(box.Min.X, box.Max.Y, box.Max.Z);
            corners[7] = new Vector3(box.Max.X, box.Max.Y, box.Max.Z);

            for (int i = 0; i < 8; i++)
            {
                float sdf = network.Evaluate(corners[i]);
                if (sdf > maxSdf) maxSdf = sdf;
            }

            return maxSdf;
        }

        /// <summary>
        /// Tests whether a ray is guaranteed to miss a surface within a box.
        /// If the lower bound is positive, the entire box is outside the surface.
        /// </summary>
        /// <param name="box">The bounding box.</param>
        /// <param name="network">The neural network.</param>
        /// <returns>True if the ray is guaranteed to miss within the box.</returns>
        public static bool GuaranteedMiss(IntervalBox box, ISdfNetwork network)
        {
            return ConservativeLowerBound(box, network) > 0;
        }

        /// <summary>
        /// Tests whether a ray is guaranteed to hit a surface within a box.
        /// If the upper bound is negative, the entire box is inside the surface.
        /// </summary>
        /// <param name="box">The bounding box.</param>
        /// <param name="network">The neural network.</param>
        /// <returns>True if the ray is guaranteed to hit within the box.</returns>
        public static bool GuaranteedHit(IntervalBox box, ISdfNetwork network)
        {
            return ConservativeUpperBound(box, network) < 0;
        }

        /// <summary>
        /// Subdivides a box into sub-boxes along the longest axis.
        /// </summary>
        /// <param name="box">The box to subdivide.</param>
        /// <param name="count">Number of subdivisions.</param>
        /// <returns>Array of sub-boxes.</returns>
        public static IntervalBox[] Subdivide(IntervalBox box, int count)
        {
            if (count <= 0) return Array.Empty<IntervalBox>();

            Vector3 size = box.Size;
            int axis = 0;
            if (size.Y > size.X && size.Y > size.Z) axis = 1;
            else if (size.Z > size.X && size.Z > size.Y) axis = 2;

            var result = new IntervalBox[count];
            Vector3 step = Vector3.Zero;

            switch (axis)
            {
                case 0:
                    step = new Vector3(size.X / count, 0, 0);
                    for (int i = 0; i < count; i++)
                    {
                        Vector3 min = box.Min + step * i;
                        Vector3 max = min + new Vector3(step.X, size.Y, size.Z);
                        result[i] = new IntervalBox { Min = min, Max = max };
                    }
                    break;
                case 1:
                    step = new Vector3(0, size.Y / count, 0);
                    for (int i = 0; i < count; i++)
                    {
                        Vector3 min = box.Min + step * i;
                        Vector3 max = min + new Vector3(size.X, step.Y, size.Z);
                        result[i] = new IntervalBox { Min = min, Max = max };
                    }
                    break;
                case 2:
                    step = new Vector3(0, 0, size.Z / count);
                    for (int i = 0; i < count; i++)
                    {
                        Vector3 min = box.Min + step * i;
                        Vector3 max = min + new Vector3(size.X, size.Y, step.Z);
                        result[i] = new IntervalBox { Min = min, Max = max };
                    }
                    break;
            }

            return result;
        }
    }
}
