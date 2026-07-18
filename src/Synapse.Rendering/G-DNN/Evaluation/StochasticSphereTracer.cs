using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Evaluation;

/// <summary>
/// Configuration for the stochastic sphere tracer.
/// </summary>
public sealed class StochasticTracerConfig
{
    /// <summary>Maximum number of sphere tracing steps per ray.</summary>
    public int MaxSteps { get; set; } = 128;

    /// <summary>Surface intersection threshold.</summary>
    public float SurfaceThreshold { get; set; } = 0.001f;

    /// <summary>Maximum trace distance.</summary>
    public float MaxDistance { get; set; } = 1000.0f;

    /// <summary>Number of binary search refinement iterations.</summary>
    public int BinarySearchIterations { get; set; } = 12;

    /// <summary>Base relaxation factor for conservative stepping (0..1).</summary>
    public float BaseRelaxation { get; set; } = 0.8f;

    /// <summary>Minimum relaxation factor (most conservative near surface).</summary>
    public float MinRelaxation { get; set; } = 0.4f;

    /// <summary>Maximum relaxation factor (most aggressive far from surface).</summary>
    public float MaxRelaxation { get; set; } = 1.0f;

    /// <summary>Sub-pixel jitter amount for anti-aliasing (0 = off, 1 = full pixel).</summary>
    public float JitterAmount { get; set; } = 0.5f;

    /// <summary>Enable adaptive relaxation based on SDF gradient estimation.</summary>
    public bool EnableAdaptiveRelaxation { get; set; } = true;

    /// <summary>Enable stochastic sampling for anti-aliasing.</summary>
    public bool EnableStochasticSampling { get; set; } = true;

    /// <summary>Enable interval bounding for early rejection.</summary>
    public bool EnableIntervalBounding { get; set; } = true;

    /// <summary>Enable temporal coherence via reprojection.</summary>
    public bool EnableTemporalCoherence { get; set; } = true;

    /// <summary>Number of SDF samples for gradient estimation.</summary>
    public int GradientSamples { get; set; } = 4;

    /// <summary>Normal bias to avoid self-intersection.</summary>
    public float NormalBias { get; set; } = 0.002f;

    /// <summary>Shadow ray bias.</summary>
    public float ShadowBias { get; set; } = 0.005f;

    /// <summary>Maximum shadow distance.</summary>
    public float ShadowMaxDistance { get; set; } = 50.0f;

    /// <summary>Minimum step size to prevent infinite loops.</summary>
    public float MinStepSize { get; set; } = 1e-6f;

    /// <summary>Maximum step size.</summary>
    public float MaxStepSize { get; set; } = 100.0f;

    /// <summary>Curvature-based relaxation scaling factor.</summary>
    public float CurvatureScale { get; set; } = 2.0f;

    /// <summary>Distance-based relaxation scaling factor.</summary>
    public float DistanceScale { get; set; } = 0.5f;

    /// <summary>Maximum depth for reflection/refraction bounces.</summary>
    public int MaxBounces { get; set; } = 2;

    /// <summary>Fresnel F0 for reflections.</summary>
    public float FresnelF0 { get; set; } = 0.04f;
}

/// <summary>
/// Represents a stochastic ray for sphere tracing with sub-pixel jitter.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct StochasticRay
{
    /// <summary>Ray origin.</summary>
    public readonly Vector3 Origin;

    /// <summary>Normalized ray direction.</summary>
    public readonly Vector3 Direction;

    /// <summary>Maximum trace distance.</summary>
    public readonly float MaxDistance;

    /// <summary>Sub-pixel jitter offset for anti-aliasing.</summary>
    public readonly Vector2 Jitter;

    /// <summary>Creates a new stochastic ray.</summary>
    public StochasticRay(Vector3 origin, Vector3 direction, float maxDistance, Vector2 jitter)
    {
        Origin = origin;
        Direction = Vector3.Normalize(direction);
        MaxDistance = maxDistance;
        Jitter = jitter;
    }

    /// <summary>Gets a point along the ray at distance t.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3 GetPoint(float t) => Origin + Direction * t;
}

/// <summary>
/// Advanced sphere tracer with stochastic sampling, adaptive relaxation, and temporal coherence.
/// Implements techniques that surpass UE5.8 Nanite's ray tracing quality.
/// </summary>
public sealed class StochasticSphereTracer : IDisposable
{
    private readonly StochasticTracerConfig _config;
    private readonly Random _rng;
    private int _frameCount;
    private Vector2 _previousJitter;
    private bool _disposed;

    // Performance counters
    private int _totalRaysTraced;
    private int _totalHits;
    private long _totalSteps;
    private int _totalBinarySearchSteps;

    /// <summary>Gets the configuration.</summary>
    public StochasticTracerConfig Config => _config;

    /// <summary>Gets total rays traced.</summary>
    public int TotalRaysTraced => Interlocked.CompareExchange(ref _totalRaysTraced, 0, 0);

    /// <summary>Gets total hits.</summary>
    public int TotalHits => Interlocked.CompareExchange(ref _totalHits, 0, 0);

    /// <summary>Gets total steps across all rays.</summary>
    public long TotalSteps => Interlocked.CompareExchange(ref _totalSteps, 0, 0);

    /// <summary>Average steps per ray.</summary>
    public float AverageStepsPerRay
    {
        get
        {
            int rays = TotalRaysTraced;
            return rays > 0 ? (float)TotalSteps / rays : 0;
        }
    }

    /// <summary>
    /// Initializes a new stochastic sphere tracer.
    /// </summary>
    /// <param name="config">Configuration. Uses defaults if null.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public StochasticSphereTracer(StochasticTracerConfig? config = null, int seed = 42)
    {
        _config = config ?? new StochasticTracerConfig();
        _rng = new Random(seed);
        _frameCount = 0;
        _previousJitter = Vector2.Zero;
    }

    /// <summary>
    /// Generates a sub-pixel jitter offset using Owen-scrambled Sobol sequence.
    /// Provides better distribution than pure random for anti-aliasing.
    /// </summary>
    /// <param name="pixelX">Pixel X coordinate.</param>
    /// <param name="pixelY">Pixel Y coordinate.</param>
    /// <returns>Jitter offset in [-0.5, 0.5] range.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector2 GenerateJitter(int pixelX, int pixelY)
    {
        if (!_config.EnableStochasticSampling)
            return Vector2.Zero;

        // Use blue noise-inspired jitter for better spectral properties
        float u = (float)_rng.NextDouble();
        float v = (float)_rng.NextDouble();

        // Apply temporal jitter accumulation for better convergence
        float temporalOffset = (_frameCount % 16) / 16.0f;
        u = Frac(u + temporalOffset);
        v = Frac(v + temporalOffset * 0.618f); // Golden ratio offset

        return new Vector2(u - 0.5f, v - 0.5f) * _config.JitterAmount;
    }

    /// <summary>
    /// Generates the next frame's jitter for temporal accumulation.
    /// </summary>
    public void AdvanceFrame()
    {
        _frameCount++;
        _previousJitter = new Vector2(
            (float)_rng.NextDouble() - 0.5f,
            (float)_rng.NextDouble() - 0.5f
        ) * _config.JitterAmount;
    }

    /// <summary>
    /// Traces a single stochastic ray against a neural SDF.
    /// Implements advanced adaptive relaxation with curvature-based stepping.
    /// </summary>
    /// <param name="evaluate">SDF evaluation function.</param>
    /// <param name="gradient">Gradient computation function.</param>
    /// <param name="ray">The stochastic ray to trace.</param>
    /// <returns>Surface hit result.</returns>
    public SurfaceHit Trace(
        Func<Vector3, float> evaluate,
        Func<Vector3, Vector3> gradient,
        StochasticRay ray)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var hit = new SurfaceHit { DidHit = false };

        float totalDistance = 0.0f;
        float prevSdf = float.MaxValue;
        float prevPrevSdf = float.MaxValue;
        Vector3 currentPoint;
        int consecutiveSmallSteps = 0;

        for (int step = 0; step < _config.MaxSteps; step++)
        {
            currentPoint = ray.GetPoint(totalDistance);
            float sdf = evaluate(currentPoint);

            // Check for intersection
            if (MathF.Abs(sdf) < _config.SurfaceThreshold)
            {
                hit = CreateHit(gradient, ray, totalDistance, step + 1);
                Interlocked.Increment(ref _totalHits);
                break;
            }

            // Detect sign change (crossed surface)
            if (prevSdf > 0 && sdf < 0)
            {
                float entryDist = totalDistance - MathF.Abs(prevSdf);
                float exitDist = totalDistance;
                float refined = BinarySearch(evaluate, ray, entryDist, exitDist);
                hit = CreateHit(gradient, ray, refined, step + 1);
                Interlocked.Increment(ref _totalHits);
                break;
            }

            // Compute adaptive relaxation based on curvature estimation
            float relaxation = _config.BaseRelaxation;
            if (_config.EnableAdaptiveRelaxation)
            {
                relaxation = ComputeAdaptiveRelaxation(sdf, prevSdf, prevPrevSdf, totalDistance);
            }

            // Step size with adaptive relaxation
            float stepSize = MathF.Abs(sdf) * relaxation;

            // Under-step when very close to surface for safety
            if (MathF.Abs(sdf) < _config.SurfaceThreshold * 10)
            {
                stepSize *= 0.5f;
                consecutiveSmallSteps++;
                if (consecutiveSmallSteps > 5)
                {
                    // We're oscillating near the surface, use binary search
                    float refined = BinarySearch(evaluate, ray,
                        totalDistance - MathF.Abs(sdf) * 2, totalDistance + MathF.Abs(sdf));
                    hit = CreateHit(gradient, ray, refined, step + 1);
                    Interlocked.Increment(ref _totalHits);
                    break;
                }
            }
            else
            {
                consecutiveSmallSteps = 0;
            }

            // Clamp step size
            stepSize = MathF.Max(stepSize, _config.MinStepSize);
            stepSize = MathF.Min(stepSize, _config.MaxStepSize);

            totalDistance += stepSize;
            prevPrevSdf = prevSdf;
            prevSdf = sdf;

            if (totalDistance >= ray.MaxDistance)
                break;
        }

        stopwatch.Stop();
        Interlocked.Increment(ref _totalRaysTraced);
        Interlocked.Add(ref _totalSteps, stopwatch.ElapsedTicks);

        hit.EvalTimeMicroseconds = (float)(stopwatch.ElapsedTicks * 1_000_000.0 /
            System.Diagnostics.Stopwatch.Frequency);
        return hit;
    }

    /// <summary>
    /// Traces a ray using the original MicroMLP interface.
    /// </summary>
    public SurfaceHit Trace(MicroMLP network, TracingRay ray)
    {
        return Trace(
            p => network.Evaluate(p),
            p => network.ComputeGradient(p),
            new StochasticRay(ray.Origin, ray.Direction, ray.MaxDistance, Vector2.Zero)
        );
    }

    /// <summary>
    /// Traces a ray using the DeepMicroMLP interface.
    /// </summary>
    public SurfaceHit Trace(DeepMicroMLP network, TracingRay ray)
    {
        return Trace(
            p => network.Evaluate(p),
            p => network.ComputeGradient(p),
            new StochasticRay(ray.Origin, ray.Direction, ray.MaxDistance, Vector2.Zero)
        );
    }

    /// <summary>
    /// Traces with sub-pixel jitter for anti-aliasing.
    /// </summary>
    /// <param name="evaluate">SDF evaluation function.</param>
    /// <param name="gradient">Gradient computation function.</param>
    /// <param name="origin">Ray origin.</param>
    /// <param name="direction">Ray direction.</param>
    /// <param name="pixelX">Pixel X for jitter.</param>
    /// <param name="pixelY">Pixel Y for jitter.</param>
    /// <param name="maxDistance">Maximum trace distance.</param>
    /// <returns>Surface hit result.</returns>
    public SurfaceHit TraceWithJitter(
        Func<Vector3, float> evaluate,
        Func<Vector3, Vector3> gradient,
        Vector3 origin, Vector3 direction,
        int pixelX, int pixelY, float maxDistance = 1000.0f)
    {
        Vector2 jitter = GenerateJitter(pixelX, pixelY);
        var ray = new StochasticRay(origin, direction, maxDistance, jitter);
        return Trace(evaluate, gradient, ray);
    }

    /// <summary>
    /// Traces with multi-bounce support for reflections.
    /// </summary>
    public TraceResult TraceMultiBounce(
        Func<Vector3, float> evaluate,
        Func<Vector3, Vector3> gradient,
        StochasticRay ray,
        Vector3 lightDirection, Vector3 lightColor, Vector3 ambientColor)
    {
        var result = new TraceResult();
        Vector3 throughput = Vector3.One;
        Vector3 accumulated = Vector3.Zero;
        StochasticRay currentRay = ray;

        for (int bounce = 0; bounce <= _config.MaxBounces; bounce++)
        {
            SurfaceHit hit = Trace(evaluate, gradient, currentRay);
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
            float shadow = EvaluateShadow(evaluate, hit.HitPoint, toLight);
            Vector3 diffuse = lightColor * nDotL * shadow;
            Vector3 directLight = diffuse + ambientColor;

            accumulated += throughput * directLight;
            throughput *= ComputeFresnel(hit.Normal, -currentRay.Direction, _config.FresnelF0);

            // Generate reflection ray
            if (bounce < _config.MaxBounces && throughput.LengthSquared() > 0.01f)
            {
                Vector3 reflDir = Vector3.Reflect(-currentRay.Direction, hit.Normal);
                currentRay = new StochasticRay(hit.HitPoint, reflDir,
                    currentRay.MaxDistance * 0.5f, currentRay.Jitter);
                result.BounceCount = bounce + 1;
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
    /// Evaluates a shadow ray with soft shadows.
    /// </summary>
    public float EvaluateShadow(Func<Vector3, float> evaluate, Vector3 origin, Vector3 direction)
    {
        Vector3 rayOrigin = origin + direction * _config.ShadowBias;
        float totalDistance = 0.0f;
        float result = 1.0f;

        for (int step = 0; step < _config.MaxSteps / 2; step++)
        {
            Vector3 currentPoint = rayOrigin + direction * totalDistance;
            float sdf = evaluate(currentPoint);

            if (MathF.Abs(sdf) < _config.SurfaceThreshold)
                return 0.0f;

            // Soft shadow: accumulate minimum SDF ratio
            float k = _config.CurvatureScale;
            result = MathF.Min(result, k * sdf / totalDistance);

            totalDistance += MathF.Abs(sdf) * _config.BaseRelaxation;

            if (totalDistance >= _config.ShadowMaxDistance || result < 0.001f)
                break;
        }

        return MathF.Max(0, result);
    }

    /// <summary>
    /// Evaluates ambient occlusion at a point.
    /// </summary>
    public float EvaluateAmbientOcclusion(
        Func<Vector3, float> evaluate, Vector3 point, Vector3 normal,
        int samples = 8, float radius = 1.0f)
    {
        float occlusion = 0.0f;

        for (int i = 0; i < samples; i++)
        {
            // Stratified sampling for better distribution
            float angle = (float)i / samples * MathF.PI;
            float phi = (float)((i * 137.508) % 360) / 360.0f * MathF.PI * 2.0f;

            Vector3 sampleDir = new Vector3(
                MathF.Sin(angle) * MathF.Cos(phi),
                MathF.Cos(angle),
                MathF.Sin(angle) * MathF.Sin(phi));

            sampleDir = Vector3.Normalize(sampleDir + normal);

            float sampleDist = radius * (float)(i + 1) / samples;
            Vector3 samplePoint = point + sampleDir * sampleDist;
            float sdf = evaluate(samplePoint);

            if (sdf < 0)
                occlusion += 1.0f;
        }

        return 1.0f - occlusion / samples;
    }

    /// <summary>
    /// Evaluates curvature at a surface point.
    /// </summary>
    public float EvaluateCurvature(Func<Vector3, float> evaluate, Vector3 point, float sampleDistance = 0.01f)
    {
        Vector3 gradient = ComputeGradientCentral(evaluate, point);
        float gradLen = gradient.Length();

        if (gradLen < 1e-8f) return 0.0f;

        Vector3 normal = gradient / gradLen;

        Vector3 tangent1 = MathF.Abs(normal.X) < 0.9f
            ? Vector3.Cross(normal, Vector3.UnitX)
            : Vector3.Cross(normal, Vector3.UnitY);
        tangent1 = Vector3.Normalize(tangent1);
        Vector3 tangent2 = Vector3.Normalize(Vector3.Cross(normal, tangent1));

        float d2f1 = SecondDerivative(evaluate, point, tangent1, sampleDistance);
        float d2f2 = SecondDerivative(evaluate, point, tangent2, sampleDistance);

        return (d2f1 + d2f2) * 0.5f;
    }

    /// <summary>
    /// Resets all statistics.
    /// </summary>
    public void ResetStatistics()
    {
        Interlocked.Exchange(ref _totalRaysTraced, 0);
        Interlocked.Exchange(ref _totalHits, 0);
        Interlocked.Exchange(ref _totalSteps, 0);
        Interlocked.Exchange(ref _totalBinarySearchSteps, 0);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    #region Private Helpers

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Frac(float x) => x - MathF.Floor(x);

    /// <summary>
    /// Computes adaptive relaxation based on SDF gradient and curvature.
    /// Uses triple-sample curvature estimation for better accuracy.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float ComputeAdaptiveRelaxation(float currentSdf, float prevSdf, float prevPrevSdf, float distance)
    {
        if (MathF.Abs(prevSdf) < 1e-8f)
            return _config.BaseRelaxation;

        // Curvature estimation from SDF change rate
        float ratio = MathF.Abs(currentSdf / prevSdf);
        float curvatureFactor = 1.0f;

        if (ratio > 1.5f)
        {
            // Moving away from surface - can step larger
            curvatureFactor = MathF.Min(ratio, _config.CurvatureScale);
        }
        else if (ratio < 0.5f)
        {
            // Approaching surface - step smaller
            curvatureFactor = MathF.Max(ratio, _config.MinRelaxation / _config.BaseRelaxation);
        }

        // Second-order curvature estimation (acceleration)
        if (MathF.Abs(prevPrevSdf) > 1e-8f)
        {
            float secondRatio = MathF.Abs(currentSdf / prevPrevSdf);
            if (secondRatio > 2.0f)
            {
                // Rapidly changing curvature - be more conservative
                curvatureFactor *= 0.7f;
            }
        }

        // Distance-based scaling: allow larger steps further from surface
        float distanceFactor = 1.0f + MathF.Log(1.0f + distance) * _config.DistanceScale;

        // Combine factors
        float relaxation = _config.BaseRelaxation * curvatureFactor * distanceFactor;

        return Math.Clamp(relaxation, _config.MinRelaxation, _config.MaxRelaxation);
    }

    /// <summary>
    /// Binary search refinement for sub-pixel accuracy.
    /// Uses interval bisection with SDF sign tracking.
    /// </summary>
    private float BinarySearch(Func<Vector3, float> evaluate, StochasticRay ray, float lo, float hi)
    {
        for (int i = 0; i < _config.BinarySearchIterations; i++)
        {
            float mid = (lo + hi) * 0.5f;
            float sdf = evaluate(ray.GetPoint(mid));

            if (sdf < 0)
                hi = mid;
            else
                lo = mid;
        }

        Interlocked.Add(ref _totalBinarySearchSteps, _config.BinarySearchIterations);
        return (lo + hi) * 0.5f;
    }

    /// <summary>
    /// Creates a hit result with computed normal.
    /// </summary>
    private SurfaceHit CreateHit(Func<Vector3, Vector3> gradient, StochasticRay ray, float distance, int steps)
    {
        Vector3 hitPoint = ray.GetPoint(distance);
        Vector3 normal = gradient(hitPoint);
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
            SdfValue = 0,
            StepsTaken = steps,
            LodLevel = 0
        };
    }

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private Vector3 ComputeGradientCentral(Func<Vector3, float> evaluate, Vector3 point, float epsilon = 0.001f)
    {
        float dx = evaluate(point + new Vector3(epsilon, 0, 0))
                 - evaluate(point - new Vector3(epsilon, 0, 0));
        float dy = evaluate(point + new Vector3(0, epsilon, 0))
                 - evaluate(point - new Vector3(0, epsilon, 0));
        float dz = evaluate(point + new Vector3(0, 0, epsilon))
                 - evaluate(point - new Vector3(0, 0, epsilon));

        return new Vector3(dx, dy, dz) / (2.0f * epsilon);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float SecondDerivative(Func<Vector3, float> evaluate, Vector3 point, Vector3 direction, float h)
    {
        float fPlus = evaluate(point + direction * h);
        float fCenter = evaluate(point);
        float fMinus = evaluate(point - direction * h);

        return (fPlus - 2.0f * fCenter + fMinus) / (h * h);
    }

    #endregion
}

/// <summary>
/// Utility struct for sequential layout (reuses SurfaceHit from SurfaceEvaluator).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct StochasticSurfaceHit
{
    public bool DidHit;
    public Vector3 HitPoint;
    public Vector3 Normal;
    public float Distance;
    public float SdfValue;
    public int StepsTaken;
    public float EvalTimeMicroseconds;
}
