using System;
// ============================================================
// FILE: RayMarcher.cs
// PATH: Evaluation/RayMarcher.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Evaluation
{
    /// <summary>
    /// Configuration for the ray marcher.
    /// </summary>
    public sealed class RayMarcherConfig
    {
        /// <summary>Maximum number of sphere tracing steps per ray.</summary>
        public int MaxSteps { get; set; } = 128;

        /// <summary>Surface intersection threshold.</summary>
        public float SurfaceThreshold { get; set; } = 0.001f;

        /// <summary>Maximum trace distance.</summary>
        public float MaxDistance { get; set; } = 1000.0f;

        /// <summary>Number of binary search refinement iterations.</summary>
        public int BinarySearchIterations { get; set; } = 8;

        /// <summary>Relaxation factor for conservative stepping (0..1).</summary>
        public float Relaxation { get; set; } = 0.8f;

        /// <summary>Enable hierarchical sphere tracing.</summary>
        public bool EnableHierarchical { get; set; } = true;

        /// <summary>Number of mip levels for hierarchical tracing.</summary>
        public int HierarchicalLevels { get; set; } = 4;

        /// <summary>Enable temporal reprojection.</summary>
        public bool EnableReprojection { get; set; } = true;

        /// <summary>Maximum depth for multi-bounce rays.</summary>
        public int MaxBounces { get; set; } = 2;

        /// <summary>Bias applied along normals to avoid self-intersection.</summary>
        public float NormalBias { get; set; } = 0.002f;

        /// <summary>Slope-based step scaling.</summary>
        public float SlopeScale { get; set; } = 0.5f;

        /// <summary>Shadow ray offset.</summary>
        public float ShadowBias { get; set; } = 0.005f;

        /// <summary>Shadow ray maximum distance.</summary>
        public float ShadowMaxDistance { get; set; } = 50.0f;

        /// <summary>Shadow ray soft factor for penumbra estimation.</summary>
        public float ShadowSoftness { get; set; } = 1.0f;

        /// <summary>Refraction index of the material.</summary>
        public float RefractionIndex { get; set; } = 1.5f;

        /// <summary>Fresnel reflectance at normal incidence.</summary>
        public float FresnelF0 { get; set; } = 0.04f;

        /// <summary>Minimum step size to prevent infinite loops.</summary>
        public float MinStepSize { get; set; } = 1e-6f;

        /// <summary>Maximum step size.</summary>
        public float MaxStepSize { get; set; } = 100.0f;

        /// <summary>Sub-pixel jitter for anti-aliasing.</summary>
        public float JitterAmount { get; set; } = 0.0f;

        /// <summary>Enable interval bounding for early rejection.</summary>
        public bool EnableIntervalBounding { get; set; } = true;
    }

    /// <summary>
    /// Represents a ray for sphere tracing.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct TracingRay
    {
        /// <summary>Ray origin.</summary>
        public readonly Vector3 Origin;

        /// <summary>Normalized ray direction.</summary>
        public readonly Vector3 Direction;

        /// <summary>Maximum trace distance.</summary>
        public readonly float MaxDistance;

        /// <summary>Creates a new tracing ray.</summary>
        public TracingRay(Vector3 origin, Vector3 direction, float maxDistance)
        {
            Origin = origin;
            Direction = Vector3.Normalize(direction);
            MaxDistance = maxDistance;
        }

        /// <summary>Gets a point along the ray at distance t.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector3 GetPoint(float t) => Origin + Direction * t;

        /// <summary>Creates a default ray.</summary>
        public static TracingRay Default => new(Vector3.Zero, Vector3.UnitZ, 1000.0f);
    }

    /// <summary>
    /// Represents the result of a multi-bounce ray trace.
    /// </summary>
    public sealed class TraceResult
    {
        /// <summary>Whether any surface was hit.</summary>
        public bool DidHit { get; set; }

        /// <summary>Final color/intensity after all bounces.</summary>
        public Vector3 Contribution { get; set; }

        /// <summary>Accumulated path through the scene.</summary>
        public List<PathSegment> Segments { get; set; } = new();

        /// <summary>Total number of sphere tracing steps across all bounces.</summary>
        public int TotalSteps { get; set; }

        /// <summary>Number of bounces performed.</summary>
        public int BounceCount { get; set; }

        /// <summary>Whether the path was terminated early.</summary>
        public bool TerminatedEarly { get; set; }
    }

    /// <summary>
    /// A single segment of a traced path.
    /// </summary>
    public sealed class PathSegment
    {
        /// <summary>Ray origin for this segment.</summary>
        public Vector3 Origin { get; set; }

        /// <summary>Ray direction for this segment.</summary>
        public Vector3 Direction { get; set; }

        /// <summary>Hit point.</summary>
        public Vector3 HitPoint { get; set; }

        /// <summary>Surface normal at hit point.</summary>
        public Vector3 Normal { get; set; }

        /// <summary>Distance traveled in this segment.</summary>
        public float Distance { get; set; }

        /// <summary>Type of ray (primary, shadow, reflection, refraction).</summary>
        public RayType Type { get; set; }

        /// <summary>Whether this segment hit a surface.</summary>
        public bool Hit { get; set; }

        /// <summary>Transmittance through the medium.</summary>
        public float Transmittance { get; set; } = 1.0f;
    }

    /// <summary>
    /// Types of rays that can be traced.
    /// </summary>
    public enum RayType
    {
        /// <summary>Primary camera ray.</summary>
        Primary,

        /// <summary>Shadow ray toward light.</summary>
        Shadow,

        /// <summary>Reflection ray.</summary>
        Reflection,

        /// <summary>Refraction ray.</summary>
        Refraction
    }

    /// <summary>
    /// Represents a hierarchical mip level for sphere tracing acceleration.
    /// </summary>
    public sealed class HierarchyLevel
    {
        /// <summary>Downsampled SDF grid values.</summary>
        public float[] SdfValues { get; set; } = Array.Empty<float>();

        /// <summary>Grid resolution along each axis.</summary>
        public int Resolution { get; set; }

        /// <summary>World-space size of each cell.</summary>
        public float CellSize { get; set; }

        /// <summary>World-space origin of the grid.</summary>
        public Vector3 Origin { get; set; }

        /// <summary>Gets the minimum SDF value in the grid cell containing the given point.</summary>
        public float GetLowerBound(Vector3 point)
        {
            Vector3 local = point - Origin;
            int ix = Math.Clamp((int)(local.X / CellSize), 0, Resolution - 1);
            int iy = Math.Clamp((int)(local.Y / CellSize), 0, Resolution - 1);
            int iz = Math.Clamp((int)(local.Z / CellSize), 0, Resolution - 1);

            int idx = ix + iy * Resolution + iz * Resolution * Resolution;
            if (idx >= 0 && idx < SdfValues.Length)
                return SdfValues[idx];
            return float.MaxValue;
        }
    }

    /// <summary>
    /// Temporal reprojection data for coherent tracing.
    /// </summary>
    public sealed class TemporalData
    {
        /// <summary>Previous frame's camera position.</summary>
        public Vector3 PreviousCameraPosition { get; set; }

        /// <summary>Previous frame's camera rotation.</summary>
        public Quaternion PreviousCameraRotation { get; set; } = Quaternion.Identity;

        /// <summary>Previous frame's hit points (screen-space mapped).</summary>
        public Dictionary<(int, int), Vector3> PreviousHits { get; set; } = new();

        /// <summary>Velocity buffer for motion vectors.</summary>
        public Vector3[,]? VelocityBuffer { get; set; }

        /// <summary>Buffer width.</summary>
        public int Width { get; set; }

        /// <summary>Buffer height.</summary>
        public int Height { get; set; }

        /// <summary>
        /// Initializes the velocity buffer.
        /// </summary>
        public void Initialize(int width, int height)
        {
            Width = width;
            Height = height;
            VelocityBuffer = new Vector3[width, height];
            PreviousHits.Clear();
        }

        /// <summary>
        /// Reprojects a point from screen space to world space using motion vectors.
        /// </summary>
        public bool TryReproject(int screenX, int screenY, Vector3 currentCameraPos,
            Quaternion currentCameraRot, out Vector3 reprojectedPoint)
        {
            reprojectedPoint = Vector3.Zero;

            if (VelocityBuffer == null ||
                screenX < 0 || screenX >= Width ||
                screenY < 0 || screenY >= Height)
                return false;

            Vector3 velocity = VelocityBuffer[screenX, screenY];
            if (velocity.LengthSquared() < 1e-10f)
                return false;

            var key = (screenX, screenY);
            if (!PreviousHits.TryGetValue(key, out Vector3 prevHit))
                return false;

            // Transform previous hit to current frame
            Quaternion deltaRot = Quaternion.Inverse(PreviousCameraRotation) * currentCameraRot;
            Vector3 deltaPos = currentCameraPos - PreviousCameraPosition;

            reprojectedPoint = Vector3.Transform(prevHit - PreviousCameraPosition, deltaRot)
                + currentCameraPos + velocity;

            return true;
        }
    }

    /// <summary>
    /// Sphere tracing implementation for neural SDFs.
    /// Supports adaptive stepping, hierarchical acceleration, temporal reprojection,
    /// binary search refinement, multi-bounce, shadow, and reflection/refraction rays.
    /// </summary>
    public sealed class RayMarcher : IDisposable
    {
        private readonly RayMarcherConfig _config;
        private readonly List<HierarchyLevel> _hierarchyLevels;
        private readonly TemporalData _temporalData;
        private bool _disposed;

        private int _totalRaysTraced;
        private int _totalHits;
        private long _totalSteps;
        private int _shadowRaysEvaluated;
        private int _reflectionRaysEvaluated;

        /// <summary>Gets the configuration.</summary>
        public RayMarcherConfig Config => _config;

        /// <summary>Gets total rays traced.</summary>
        public int TotalRaysTraced => System.Threading.Interlocked.CompareExchange(ref _totalRaysTraced, 0, 0);

        /// <summary>Gets total hits.</summary>
        public int TotalHits => System.Threading.Interlocked.CompareExchange(ref _totalHits, 0, 0);

        /// <summary>Gets total steps across all rays.</summary>
        public long TotalSteps => System.Threading.Interlocked.CompareExchange(ref _totalSteps, 0, 0);

        /// <summary>Gets average steps per ray.</summary>
        public float AverageStepsPerRay
        {
            get
            {
                int rays = TotalRaysTraced;
                return rays > 0 ? (float)TotalSteps / rays : 0;
            }
        }

        /// <summary>
        /// Initializes a new ray marcher.
        /// </summary>
        /// <param name="config">Configuration. Uses defaults if null.</param>
        public RayMarcher(RayMarcherConfig? config = null)
        {
            _config = config ?? new RayMarcherConfig();
            _hierarchyLevels = new List<HierarchyLevel>();
            _temporalData = new TemporalData();
        }

        /// <summary>
        /// Builds hierarchical SDF bounds from a network.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="worldBounds">World-space bounding box of the scene.</param>
        /// <param name="baseResolution">Resolution of the finest level.</param>
        public void BuildHierarchy(ISdfNetwork network, IntervalBox worldBounds, int baseResolution = 32)
        {
            _hierarchyLevels.Clear();

            for (int level = 0; level < _config.HierarchicalLevels; level++)
            {
                int res = Math.Max(1, baseResolution >> level);
                float cellSize = worldBounds.Size.X / res;

                var hierarchyLevel = new HierarchyLevel
                {
                    Resolution = res,
                    CellSize = cellSize,
                    Origin = worldBounds.Min,
                    SdfValues = new float[res * res * res]
                };

                // Fill the SDF grid with minimum values per cell
                for (int z = 0; z < res; z++)
                    for (int y = 0; y < res; y++)
                        for (int x = 0; x < res; x++)
                        {
                            Vector3 cellCenter = worldBounds.Min + new Vector3(
                                (x + 0.5f) * cellSize,
                                (y + 0.5f) * cellSize,
                                (z + 0.5f) * cellSize);

                            float minSdf = float.MaxValue;

                            // Sample corners of the cell
                            for (int dz = 0; dz <= 1; dz++)
                                for (int dy = 0; dy <= 1; dy++)
                                    for (int dx = 0; dx <= 1; dx++)
                                    {
                                        Vector3 corner = worldBounds.Min + new Vector3(
                                            (x + dx) * cellSize,
                                            (y + dy) * cellSize,
                                            (z + dz) * cellSize);

                                        float sdf = network.Evaluate(corner);
                                        if (sdf < minSdf)
                                            minSdf = sdf;
                                    }

                            hierarchyLevel.SdfValues[x + y * res + z * res * res] = minSdf;
                        }

                _hierarchyLevels.Add(hierarchyLevel);
            }
        }

        /// <summary>
        /// Gets the hierarchical lower bound SDF for a point.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float GetHierarchyLowerBound(Vector3 point, int startLevel = 0)
        {
            if (!_config.EnableHierarchical || _hierarchyLevels.Count == 0)
                return float.MinValue;

            // Use the coarsest level that contains the point
            for (int i = startLevel; i < _hierarchyLevels.Count; i++)
            {
                var level = _hierarchyLevels[i];
                Vector3 local = point - level.Origin;

                if (local.X >= 0 && local.Y >= 0 && local.Z >= 0 &&
                    local.X < level.Resolution * level.CellSize &&
                    local.Y < level.Resolution * level.CellSize &&
                    local.Z < level.Resolution * level.CellSize)
                {
                    return level.GetLowerBound(point);
                }
            }

            return float.MinValue;
        }

        /// <summary>
        /// Traces a single ray against the neural SDF.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="ray">The ray to trace.</param>
        /// <returns>Surface hit result.</returns>
        public SurfaceHit Trace(ISdfNetwork network, TracingRay ray)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var hit = new SurfaceHit { DidHit = false };

            float totalDistance = 0.0f;
            float prevSdf = float.MaxValue;
            Vector3 currentPoint;

            for (int step = 0; step < _config.MaxSteps; step++)
            {
                currentPoint = ray.GetPoint(totalDistance);
                float sdf = network.Evaluate(currentPoint);

                // Check for intersection
                if (MathF.Abs(sdf) < _config.SurfaceThreshold)
                {
                    hit = CreateHit(network, ray, totalDistance, step + 1);
                    Interlocked.Increment(ref _totalHits);
                    break;
                }

                // Detect sign change (crossed surface)
                if (prevSdf > 0 && sdf < 0)
                {
                    float entryDist = totalDistance - MathF.Abs(prevSdf);
                    float exitDist = totalDistance;
                    float refined = BinarySearch(network, ray, entryDist, exitDist);
                    hit = CreateHit(network, ray, refined, step + 1);
                    Interlocked.Increment(ref _totalHits);
                    break;
                }

                // Hierarchical acceleration: use coarser level for larger steps
                float stepSize = MathF.Abs(sdf);
                if (_config.EnableHierarchical && _hierarchyLevels.Count > 0)
                {
                    float hierBound = GetHierarchyLowerBound(currentPoint);
                    if (hierBound > stepSize)
                    {
                        stepSize = hierBound;
                    }
                }

                // Adaptive step scaling
                stepSize *= ComputeAdaptiveScale(sdf, prevSdf, totalDistance);
                stepSize *= _config.Relaxation;

                // Clamp step size
                stepSize = MathF.Max(stepSize, _config.MinStepSize);
                stepSize = MathF.Min(stepSize, _config.MaxStepSize);

                totalDistance += stepSize;
                prevSdf = sdf;

                if (totalDistance >= ray.MaxDistance)
                    break;
            }

            stopwatch.Stop();
            System.Threading.Interlocked.Add(ref _totalSteps, stopwatch.ElapsedTicks);
            System.Threading.Interlocked.Increment(ref _totalRaysTraced);

            hit.EvalTimeMicroseconds = (float)(stopwatch.ElapsedTicks * 1_000_000.0 /
                System.Diagnostics.Stopwatch.Frequency);
            return hit;
        }

        /// <summary>
        /// Creates a hit result with computed normal.
        /// </summary>
        private SurfaceHit CreateHit(ISdfNetwork network, TracingRay ray, float distance, int steps)
        {
            Vector3 hitPoint = ray.GetPoint(distance);
            Vector3 normal = network.ComputeGradient(hitPoint);
            if (normal.LengthSquared() > 1e-10f)
                normal = Vector3.Normalize(normal);
            else
                normal = Vector3.UnitY;

            return new SurfaceHit
            {
                DidHit = true,
                HitPoint = hitPoint + normal * _config.NormalBias,
                Normal = normal,
                Distance = distance,
                SdfValue = network.Evaluate(hitPoint),
                StepsTaken = steps,
                LodLevel = 0
            };
        }

        /// <summary>
        /// Binary search refinement for sub-pixel accuracy.
        /// </summary>
        private float BinarySearch(ISdfNetwork network, TracingRay ray, float lo, float hi)
        {
            for (int i = 0; i < _config.BinarySearchIterations; i++)
            {
                float mid = (lo + hi) * 0.5f;
                float sdf = network.Evaluate(ray.GetPoint(mid));

                if (sdf < 0)
                    hi = mid;
                else
                    lo = mid;
            }

            return (lo + hi) * 0.5f;
        }

        /// <summary>
        /// Computes adaptive step scaling based on SDF gradient estimation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ComputeAdaptiveScale(float currentSdf, float prevSdf, float distance)
        {
            if (MathF.Abs(prevSdf) < 1e-8f)
                return 1.0f;

            float ratio = MathF.Abs(currentSdf / prevSdf);
            float scale = 1.0f;

            if (ratio > 1.5f)
            {
                // Moving away from surface - can step larger
                scale = MathF.Min(ratio, 3.0f);
            }
            else if (ratio < 0.5f)
            {
                // Approaching surface - step smaller
                scale = MathF.Max(ratio, 0.25f);
            }

            // Distance-based scaling
            float distScale = 1.0f + MathF.Log(1.0f + distance) * _config.SlopeScale;

            return scale * distScale;
        }

        /// <summary>
        /// Traces with multi-bounce support for reflections and refractions.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="ray">Initial ray.</param>
        /// <param name="lightDirection">Direction to the light source.</param>
        /// <param name="lightColor">Light color/intensity.</param>
        /// <param name="ambientColor">Ambient light color.</param>
        /// <returns>Full trace result with bounces.</returns>
        public TraceResult TraceMultiBounce(ISdfNetwork network, TracingRay ray,
            Vector3 lightDirection, Vector3 lightColor, Vector3 ambientColor)
        {
            var result = new TraceResult();
            Vector3 throughput = Vector3.One;
            Vector3 accumulated = Vector3.Zero;
            TracingRay currentRay = ray;

            for (int bounce = 0; bounce <= _config.MaxBounces; bounce++)
            {
                SurfaceHit hit = Trace(network, currentRay);
                result.TotalSteps += hit.StepsTaken;

                if (!hit.DidHit)
                {
                    result.DidHit = bounce > 0;
                    break;
                }

                var segment = new PathSegment
                {
                    Origin = currentRay.Origin,
                    Direction = currentRay.Direction,
                    HitPoint = hit.HitPoint,
                    Normal = hit.Normal,
                    Distance = hit.Distance,
                    Hit = true,
                    Type = bounce == 0 ? RayType.Primary : RayType.Reflection
                };

                result.Segments.Add(segment);

                // Compute direct lighting
                Vector3 toLight = -Vector3.Normalize(lightDirection);
                float nDotL = MathF.Max(0, Vector3.Dot(hit.Normal, toLight));

                // Shadow ray
                float shadow = EvaluateShadow(network, hit.HitPoint, toLight);
                Vector3 diffuse = lightColor * nDotL * shadow;
                Vector3 directLight = diffuse + ambientColor;

                accumulated += throughput * directLight;
                throughput *= ComputeFresnel(hit.Normal, -currentRay.Direction, _config.FresnelF0);

                // Generate reflection ray
                if (bounce < _config.MaxBounces && throughput.LengthSquared() > 0.01f)
                {
                    Vector3 reflDir = Vector3.Reflect(-currentRay.Direction, hit.Normal);
                    currentRay = new TracingRay(hit.HitPoint, reflDir, currentRay.MaxDistance * 0.5f);
                    result.BounceCount = bounce + 1;
                    System.Threading.Interlocked.Increment(ref _reflectionRaysEvaluated);
                }
                else
                {
                    result.TerminatedEarly = true;
                    break;
                }
            }

            result.Contribution = accumulated;
            return result;
        }

        /// <summary>
        /// Evaluates a shadow ray. Returns 0 for fully shadowed, 1 for fully lit.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="origin">Point on the surface.</param>
        /// <param name="direction">Direction toward the light.</param>
        /// <returns>Shadow factor [0,1].</returns>
        public float EvaluateShadow(ISdfNetwork network, Vector3 origin, Vector3 direction)
        {
            System.Threading.Interlocked.Increment(ref _shadowRaysEvaluated);

            Vector3 rayOrigin = origin + direction * _config.ShadowBias;
            float totalDistance = 0.0f;
            float occlusion = 0.0f;

            for (int step = 0; step < _config.MaxSteps / 2; step++)
            {
                Vector3 currentPoint = rayOrigin + direction * totalDistance;
                float sdf = network.Evaluate(currentPoint);

                if (MathF.Abs(sdf) < _config.SurfaceThreshold)
                {
                    // Smooth shadow penumbra
                    float penumbra = MathF.Max(0, totalDistance / _config.ShadowMaxDistance);
                    occlusion = 1.0f - penumbra * _config.ShadowSoftness;
                    return MathF.Max(0, occlusion);
                }

                // Accumulate soft shadow
                if (sdf < 0)
                {
                    occlusion = MathF.Max(occlusion, 0.5f + 0.5f * sdf / _config.ShadowBias);
                }

                totalDistance += MathF.Abs(sdf) * _config.Relaxation;

                if (totalDistance >= _config.ShadowMaxDistance)
                    break;
            }

            return MathF.Max(0, 1.0f - occlusion);
        }

        /// <summary>
        /// Traces a reflection ray.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="hitPoint">Point on the surface.</param>
        /// <param name="normal">Surface normal.</param>
        /// <param name="incidentDirection">Incoming ray direction.</param>
        /// <returns>Hit result for the reflection ray.</returns>
        public SurfaceHit TraceReflection(ISdfNetwork network, Vector3 hitPoint, Vector3 normal,
            Vector3 incidentDirection)
        {
            System.Threading.Interlocked.Increment(ref _reflectionRaysEvaluated);

            Vector3 reflDir = Vector3.Reflect(incidentDirection, normal);
            TracingRay reflRay = new TracingRay(hitPoint, reflDir, _config.MaxDistance * 0.5f);
            return Trace(network, reflRay);
        }

        /// <summary>
        /// Traces a refraction ray.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="hitPoint">Point on the surface.</param>
        /// <param name="normal">Surface normal (outward facing).</param>
        /// <param name="incidentDirection">Incoming ray direction.</param>
        /// <param name="etaRatio">Ratio of IOR (entering/exiting).</param>
        /// <returns>Hit result for the refraction ray.</returns>
        public SurfaceHit TraceRefraction(ISdfNetwork network, Vector3 hitPoint, Vector3 normal,
            Vector3 incidentDirection, float etaRatio)
        {
            Vector3 refracted = RaySurfaceUtils.Refract(incidentDirection, normal, etaRatio);

            if (refracted.LengthSquared() < 1e-8f)
            {
                // Total internal reflection
                return TraceReflection(network, hitPoint, normal, incidentDirection);
            }

            TracingRay refrRay = new TracingRay(hitPoint, refracted, _config.MaxDistance * 0.5f);
            return Trace(network, refrRay);
        }

        /// <summary>
        /// Traces a refraction ray using the configured refraction index.
        /// </summary>
        public SurfaceHit TraceRefraction(ISdfNetwork network, Vector3 hitPoint, Vector3 normal,
            Vector3 incidentDirection)
        {
            float eta = 1.0f / _config.RefractionIndex;
            float dot = Vector3.Dot(incidentDirection, normal);
            if (dot > 0)
                eta = _config.RefractionIndex;

            return TraceRefraction(network, hitPoint, normal, incidentDirection, eta);
        }

        /// <summary>
        /// Evaluates Fresnel reflectance using Schlick's approximation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ComputeFresnel(Vector3 normal, Vector3 viewDir, float f0)
        {
            float cosTheta = MathF.Max(0, Vector3.Dot(normal, viewDir));
            float oneMinusCos = 1.0f - cosTheta;
            float oneMinusCos2 = oneMinusCos * oneMinusCos;
            float oneMinusCos4 = oneMinusCos2 * oneMinusCos2;
            float oneMinusCos5 = oneMinusCos4 * oneMinusCos;

            return f0 + (1.0f - f0) * oneMinusCos5;
        }

        /// <summary>
        /// Evaluates soft shadow with variable penumbra.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="origin">Point on the surface.</param>
        /// <param name="direction">Direction toward light.</param>
        /// <param name="k">Penumbra softness parameter.</param>
        /// <returns>Soft shadow factor [0,1].</returns>
        public float EvaluateSoftShadow(ISdfNetwork network, Vector3 origin, Vector3 direction, float k)
        {
            System.Threading.Interlocked.Increment(ref _shadowRaysEvaluated);

            Vector3 rayOrigin = origin + direction * _config.ShadowBias;
            float totalDistance = 0.0f;
            float result = 1.0f;

            for (int step = 0; step < _config.MaxSteps / 2; step++)
            {
                Vector3 currentPoint = rayOrigin + direction * totalDistance;
                float sdf = network.Evaluate(currentPoint);

                if (MathF.Abs(sdf) < _config.SurfaceThreshold)
                    return 0.0f;

                // Soft shadow: accumulate minimum SDF ratio
                result = MathF.Min(result, k * sdf / totalDistance);

                totalDistance += MathF.Abs(sdf) * _config.Relaxation;

                if (totalDistance >= _config.ShadowMaxDistance || result < 0.001f)
                    break;
            }

            return MathF.Max(0, result);
        }

        /// <summary>
        /// Evaluates ambient occlusion at a point.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="point">Point on the surface.</param>
        /// <param name="normal">Surface normal.</param>
        /// <param name="samples">Number of AO samples.</param>
        /// <param name="radius">AO sampling radius.</param>
        /// <returns>Ambient occlusion factor [0,1].</returns>
        public float EvaluateAmbientOcclusion(ISdfNetwork network, Vector3 point, Vector3 normal,
            int samples = 8, float radius = 1.0f)
        {
            float occlusion = 0.0f;

            // Simple hemisphere sampling
            for (int i = 0; i < samples; i++)
            {
                float angle = (float)i / samples * MathF.PI;
                float phi = (float)((i * 137.508) % 360) / 360.0f * MathF.PI * 2.0f;

                Vector3 sampleDir = new Vector3(
                    MathF.Sin(angle) * MathF.Cos(phi),
                    MathF.Cos(angle),
                    MathF.Sin(angle) * MathF.Sin(phi));

                // Align to hemisphere around normal
                sampleDir = Vector3.Normalize(sampleDir + normal);

                float sampleDist = radius * (float)(i + 1) / samples;
                Vector3 samplePoint = point + sampleDir * sampleDist;
                float sdf = network.Evaluate(samplePoint);

                if (sdf < 0)
                {
                    occlusion += 1.0f;
                }
            }

            return 1.0f - occlusion / samples;
        }

        /// <summary>
        /// Evaluates curvature at a surface point using gradient finite differences.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="point">Point on the surface.</param>
        /// <param name="sampleDistance">Distance for finite differences.</param>
        /// <returns>Estimated mean curvature.</returns>
        public float EvaluateCurvature(ISdfNetwork network, Vector3 point, float sampleDistance = 0.01f)
        {
            Vector3 gradient = network.ComputeGradient(point);
            float gradLen = gradient.Length();

            if (gradLen < 1e-8f)
                return 0.0f;

            Vector3 normal = gradient / gradLen;

            // Sample along two tangent directions to estimate curvature
            Vector3 tangent1 = MathF.Abs(normal.X) < 0.9f
                ? Vector3.Cross(normal, Vector3.UnitX)
                : Vector3.Cross(normal, Vector3.UnitY);
            tangent1 = Vector3.Normalize(tangent1);
            Vector3 tangent2 = Vector3.Normalize(Vector3.Cross(normal, tangent1));

            // Compute second derivatives along tangent directions
            float d2f1 = SecondDerivative(network, point, tangent1, sampleDistance);
            float d2f2 = SecondDerivative(network, point, tangent2, sampleDistance);

            // Mean curvature approximation
            return (d2f1 + d2f2) * 0.5f;
        }

        /// <summary>
        /// Computes the second derivative of the SDF along a direction.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float SecondDerivative(ISdfNetwork network, Vector3 point, Vector3 direction, float h)
        {
            float fPlus = network.Evaluate(point + direction * h);
            float fCenter = network.Evaluate(point);
            float fMinus = network.Evaluate(point - direction * h);

            return (fPlus - 2.0f * fCenter + fMinus) / (h * h);
        }

        /// <summary>
        /// Initializes temporal reprojection data for a new frame.
        /// </summary>
        public void BeginFrame(int width, int height, Vector3 cameraPosition, Quaternion cameraRotation)
        {
            if (_config.EnableReprojection)
            {
                _temporalData.PreviousCameraPosition = _temporalData.PreviousCameraPosition == Vector3.Zero
                    ? cameraPosition
                    : _temporalData.PreviousHits.Count > 0 ? _temporalData.PreviousCameraPosition : cameraPosition;
                _temporalData.PreviousCameraRotation = cameraRotation;

                if (_temporalData.VelocityBuffer == null ||
                    _temporalData.Width != width || _temporalData.Height != height)
                {
                    _temporalData.Initialize(width, height);
                }
            }
        }

        /// <summary>
        /// Attempts to find a good starting point for sphere tracing via reprojection.
        /// </summary>
        /// <param name="screenX">Pixel X coordinate.</param>
        /// <param name="screenY">Pixel Y coordinate.</param>
        /// <param name="currentCameraPos">Current camera position.</param>
        /// <param name="currentCameraRot">Current camera rotation.</param>
        /// <param name="reprojectedOrigin">Output reprojected origin.</param>
        /// <returns>True if reprojection succeeded.</returns>
        public bool TryGetReprojectedOrigin(int screenX, int screenY, Vector3 currentCameraPos,
            Quaternion currentCameraRot, out Vector3 reprojectedOrigin)
        {
            reprojectedOrigin = Vector3.Zero;
            if (!_config.EnableReprojection)
                return false;
            return _temporalData.TryReproject(screenX, screenY, currentCameraPos, currentCameraRot,
                out reprojectedOrigin);
        }

        /// <summary>
        /// Updates the temporal velocity buffer after tracing.
        /// </summary>
        /// <param name="screenX">Pixel X coordinate.</param>
        /// <param name="screenY">Pixel Y coordinate.</param>
        /// <param name="hitPoint">World-space hit point.</param>
        public void UpdateTemporalData(int screenX, int screenY, Vector3 hitPoint)
        {
            if (!_config.EnableReprojection)
                return;

            _temporalData.PreviousHits[(screenX, screenY)] = hitPoint;

            if (_temporalData.VelocityBuffer != null &&
                screenX >= 0 && screenX < _temporalData.Width &&
                screenY >= 0 && screenY < _temporalData.Height)
            {
                _temporalData.VelocityBuffer[screenX, screenY] = hitPoint;
            }
        }

        /// <summary>
        /// Performs a full ray trace with reprojection optimization.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="ray">The ray.</param>
        /// <param name="screenX">Screen X for reprojection.</param>
        /// <param name="screenY">Screen Y for reprojection.</param>
        /// <param name="cameraPosition">Current camera position.</param>
        /// <param name="cameraRotation">Current camera rotation.</param>
        /// <returns>Surface hit result.</returns>
        public SurfaceHit TraceWithReprojection(ISdfNetwork network, TracingRay ray,
            int screenX, int screenY, Vector3 cameraPosition, Quaternion cameraRotation)
        {
            if (_config.EnableReprojection &&
                TryGetReprojectedOrigin(screenX, screenY, cameraPosition, cameraRotation,
                    out Vector3 reprojectedOrigin))
            {
                // Use reprojected point as initial estimate
                float reprojDist = Vector3.Distance(ray.Origin, reprojectedOrigin);
                float sdf = network.Evaluate(reprojectedOrigin);

                if (MathF.Abs(sdf) < _config.SurfaceThreshold * 10)
                {
                    // Reprojected point is close to surface - refine directly
                    var refinedHit = CreateHit(network, ray, reprojDist, 0);
                    UpdateTemporalData(screenX, screenY, refinedHit.HitPoint);
                    return refinedHit;
                }
            }

            // Fall back to full trace
            SurfaceHit hit = Trace(network, ray);
            if (hit.DidHit)
            {
                UpdateTemporalData(screenX, screenY, hit.HitPoint);
            }
            return hit;
        }

        /// <summary>
        /// Evaluates the SDF gradient at a point using central differences.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <param name="epsilon">Finite difference step.</param>
        /// <returns>Gradient vector.</returns>
        public Vector3 ComputeGradientCentral(ISdfNetwork network, Vector3 point, float epsilon = 0.001f)
        {
            float dx = network.Evaluate(point + new Vector3(epsilon, 0, 0))
                     - network.Evaluate(point - new Vector3(epsilon, 0, 0));
            float dy = network.Evaluate(point + new Vector3(0, epsilon, 0))
                     - network.Evaluate(point - new Vector3(0, epsilon, 0));
            float dz = network.Evaluate(point + new Vector3(0, 0, epsilon))
                     - network.Evaluate(point - new Vector3(0, 0, epsilon));

            return new Vector3(dx, dy, dz) / (2.0f * epsilon);
        }

        /// <summary>
        /// Returns shadow ray statistics.
        /// </summary>
        public int GetShadowRayCount() =>
            System.Threading.Interlocked.CompareExchange(ref _shadowRaysEvaluated, 0, 0);

        /// <summary>
        /// Returns reflection ray statistics.
        /// </summary>
        public int GetReflectionRayCount() =>
            System.Threading.Interlocked.CompareExchange(ref _reflectionRaysEvaluated, 0, 0);

        /// <summary>
        /// Resets all statistics.
        /// </summary>
        public void ResetStatistics()
        {
            System.Threading.Interlocked.Exchange(ref _totalRaysTraced, 0);
            System.Threading.Interlocked.Exchange(ref _totalHits, 0);
            System.Threading.Interlocked.Exchange(ref _totalSteps, 0);
            System.Threading.Interlocked.Exchange(ref _shadowRaysEvaluated, 0);
            System.Threading.Interlocked.Exchange(ref _reflectionRaysEvaluated, 0);
        }

        /// <summary>
        /// Disposes the ray marcher.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _hierarchyLevels.Clear();
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// Provides utilities for ray-surface interaction calculations.
    /// </summary>
    public static class RaySurfaceUtils
    {
        /// <summary>
        /// Computes the reflected ray direction.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Reflect(Vector3 incident, Vector3 normal)
        {
            return incident - 2.0f * Vector3.Dot(incident, normal) * normal;
        }

        /// <summary>
        /// Computes the refracted ray direction using Snell's law.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Refract(Vector3 incident, Vector3 normal, float etaRatio)
        {
            float cosI = -Vector3.Dot(incident, normal);
            float sinT2 = etaRatio * etaRatio * (1.0f - cosI * cosI);

            if (sinT2 > 1.0f)
                return Vector3.Zero; // Total internal reflection

            float cosT = MathF.Sqrt(1.0f - sinT2);
            return etaRatio * incident + (etaRatio * cosI - cosT) * normal;
        }

        /// <summary>
        /// Computes Fresnel reflectance using Schlick's approximation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float FresnelSchlick(float cosTheta, float f0)
        {
            float oneMinusCos = 1.0f - cosTheta;
            float oneMinusCos2 = oneMinusCos * oneMinusCos;
            return f0 + (1.0f - f0) * oneMinusCos2 * oneMinusCos2 * oneMinusCos;
        }

        /// <summary>
        /// Computes the view-dependent reflectance for a material.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 ComputeSpecular(Vector3 halfVector, Vector3 normal,
            Vector3 lightDir, float shininess)
        {
            float nDotH = MathF.Max(0, Vector3.Dot(normal, halfVector));
            float spec = MathF.Pow(nDotH, shininess);
            return Vector3.One * spec;
        }

        /// <summary>
        /// Computes half-vector between light and view directions.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 ComputeHalfVector(Vector3 lightDir, Vector3 viewDir)
        {
            return Vector3.Normalize(lightDir + viewDir);
        }

        /// <summary>
        /// Converts a hit result to a world-space position.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 HitToWorld(SurfaceHit hit, Matrix4x4 transform)
        {
            return Vector3.Transform(hit.HitPoint, transform);
        }

        /// <summary>
        /// Computes the depth from a camera to a hit point.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ComputeDepth(Vector3 cameraPos, Vector3 hitPoint, Vector3 cameraForward)
        {
            return Vector3.Dot(hitPoint - cameraPos, cameraForward);
        }

        /// <summary>
        /// Creates a ray from screen coordinates.
        /// </summary>
        public static TracingRay ScreenToRay(int screenX, int screenY, int screenWidth, int screenHeight,
            Vector3 cameraPosition, Vector3 cameraForward, Vector3 cameraRight, Vector3 cameraUp,
            float fovY, float maxDistance = 1000.0f)
        {
            float aspectRatio = (float)screenWidth / screenHeight;
            float halfHeight = MathF.Tan(fovY * 0.5f);

            float normalizedX = (2.0f * screenX / screenWidth - 1.0f) * halfHeight * aspectRatio;
            float normalizedY = (1.0f - 2.0f * screenY / screenHeight) * halfHeight;

            Vector3 direction = Vector3.Normalize(
                cameraForward + cameraRight * normalizedX + cameraUp * normalizedY);

            return new TracingRay(cameraPosition, direction, maxDistance);
        }
    }
}
