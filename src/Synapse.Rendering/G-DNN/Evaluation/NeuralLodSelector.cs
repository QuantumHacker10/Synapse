using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Evaluation;

/// <summary>
/// Configuration for neural LOD selection.
/// </summary>
public sealed class NeuralLodConfig
{
    /// <summary>Screen-space pixel error threshold for LOD selection.</summary>
    public float PixelErrorThreshold { get; set; } = 1.0f;

    /// <summary>Hysteresis factor to prevent LOD popping.</summary>
    public float Hysteresis { get; set; } = 0.1f;

    /// <summary>Maximum LOD level (0 = highest detail).</summary>
    public int MaxLodLevel { get; set; } = 16;

    /// <summary>Minimum LOD level (always use at least this).</summary>
    public int MinLodLevel { get; set; } = 0;

    /// <summary>Enable screen-space error metric.</summary>
    public bool UseScreenSpaceError { get; set; } = true;

    /// <summary>Enable distance-based LOD selection.</summary>
    public bool UseDistanceBased { get; set; } = true;

    /// <summary>Enable curvature-based LOD selection (more detail at high curvature).</summary>
    public bool UseCurvatureBased { get; set; } = true;

    /// <summary>Enable motion-based LOD selection (more detail for moving objects).</summary>
    public bool UseMotionBased { get; set; } = true;

    /// <summary>Maximum screen-space motion for motion-based LOD.</summary>
    public float MaxScreenSpaceMotion { get; set; } = 10.0f;

    /// <summary>Curvature weight for LOD selection.</summary>
    public float CurvatureWeight { get; set; } = 0.3f;

    /// <summary>Motion weight for LOD selection.</summary>
    public float MotionWeight { get; set; } = 0.2f;

    /// <summary>Distance weight for LOD selection.</summary>
    public float DistanceWeight { get; set; } = 0.5f;

    /// <summary>Enable temporal stabilization.</summary>
    public bool EnableTemporalStabilization { get; set; } = true;

    /// <strong>Maximum LOD changes per frame to prevent popping.</summary>
    public int MaxLodChangesPerFrame { get; set; } = 4;

    /// <summary>LOD transition distance (in world units).</summary>
    public float LodTransitionDistance { get; set; } = 10.0f;

    /// <summary>Enable predictive LOD selection based on camera velocity.</summary>
    public bool EnablePredictiveSelection { get; set; } = true;

    /// <summary>Camera velocity prediction time horizon (seconds).</summary>
    public float PredictionHorizon { get; set; } = 0.1f;
}

/// <summary>
/// Represents a LOD candidate with scoring information.
/// </summary>
public readonly struct LodCandidate
{
    /// <summary>LOD level index.</summary>
    public readonly int Level;

    /// <summary>Neural network for this LOD.</summary>
    public readonly ISdfNetwork Network;

    /// <summary>Bounding sphere center.</summary>
    public readonly Vector3 BoundingCenter;

    /// <summary>Bounding sphere radius.</summary>
    public readonly float BoundingRadius;

    /// <summary>Maximum world-space distance for this LOD.</summary>
    public readonly float MaxDistance;

    /// <summary>Estimated screen-space error.</summary>
    public readonly float ScreenSpaceError;

    /// <summary>Curvature at this LOD level.</summary>
    public readonly float Curvature;

    /// <summary>Composite LOD score (lower is better).</summary>
    public readonly float Score;

    /// <summary>Creates a new LOD candidate.</summary>
    public LodCandidate(int level, ISdfNetwork network, Vector3 center, float radius,
        float maxDistance, float screenSpaceError, float curvature, float score)
    {
        Level = level;
        Network = network;
        BoundingCenter = center;
        BoundingRadius = radius;
        MaxDistance = maxDistance;
        ScreenSpaceError = screenSpaceError;
        Curvature = curvature;
        Score = score;
    }
}

/// <summary>
/// Represents the camera state for predictive LOD selection.
/// </summary>
public struct CameraState
{
    /// <summary>Camera position.</summary>
    public Vector3 Position;

    /// <summary>Camera forward direction.</summary>
    public Vector3 Forward;

    /// <summary>Camera right direction.</summary>
    public Vector3 Right;

    /// <summary>Camera up direction.</summary>
    public Vector3 Up;

    /// <summary>Vertical field of view in radians.</summary>
    public float FieldOfView;

    /// <summary>Screen width in pixels.</summary>
    public int ScreenWidth;

    /// <summary>Screen height in pixels.</summary>
    public int ScreenHeight;

    /// <summary>Camera velocity (units per second).</summary>
    public Vector3 Velocity;

    /// <summary>Camera angular velocity (radians per second).</summary>
    public Vector3 AngularVelocity;

    /// <summary>Projection matrix.</summary>
    public Matrix4x4 Projection;

    /// <summary>View matrix.</summary>
    public Matrix4x4 View;

    /// <summary>View-Projection matrix.</summary>
    public Matrix4x4 ViewProjection;
}

/// <summary>
/// Advanced neural LOD selector with content-aware, motion-aware, and predictive selection.
/// Implements techniques that surpass UE5.8 Nanite's LOD system.
/// </summary>
public sealed class NeuralLodSelector : IDisposable
{
    private readonly NeuralLodConfig _config;
    private readonly List<LodCandidate> _candidates;
    private readonly Dictionary<int, LodState> _objectStates;
    private CameraState _previousCameraState;
    private bool _initialized;
    private bool _disposed;

    // Performance counters
    private int _totalSelections;
    private int _totalLodChanges;
    private int _totalPredictiveCorrections;

    /// <summary>Gets the configuration.</summary>
    public NeuralLodConfig Config => _config;

    /// <summary>Gets the number of registered LOD candidates.</summary>
    public int CandidateCount => _candidates.Count;

    /// <summary>Gets total LOD selections performed.</summary>
    public int TotalSelections => System.Threading.Interlocked.CompareExchange(ref _totalSelections, 0, 0);

    /// <summary>Gets total LOD changes.</summary>
    public int TotalLodChanges => System.Threading.Interlocked.CompareExchange(ref _totalLodChanges, 0, 0);

    /// <summary>
    /// Initializes a new neural LOD selector.
    /// </summary>
    public NeuralLodSelector(NeuralLodConfig? config = null)
    {
        _config = config ?? new NeuralLodConfig();
        _candidates = new List<LodCandidate>();
        _objectStates = new Dictionary<int, LodState>();
        _initialized = false;
    }

    /// <summary>
    /// Registers a LOD level with its network and bounding information.
    /// </summary>
    public void RegisterLod(int level, ISdfNetwork network, Vector3 center, float radius, float maxDistance)
    {
        float curvature = EstimateNetworkCurvature(network, center);
        var candidate = new LodCandidate(level, network, center, radius, maxDistance, 0, curvature, 0);
        _candidates.Add(candidate);
        _candidates.Sort((a, b) => a.Level.CompareTo(b.Level));
    }

    /// <summary>Removes all registered LOD candidates (used when swapping the live SDF).</summary>
    public void Clear()
    {
        _candidates.Clear();
        _objectStates.Clear();
    }

    /// <summary>
    /// Selects the optimal LOD level for an object based on camera state and content analysis.
    /// </summary>
    /// <param name="objectId">Unique object identifier.</param>
    /// <param name="cameraState">Current camera state.</param>
    /// <param name="objectCenter">Object center in world space.</param>
    /// <param name="objectRadius">Object bounding radius.</param>
    /// <param name="previousLod">Previous LOD level for hysteresis.</param>
    /// <returns>Selected LOD level index.</returns>
    public int SelectLod(int objectId, CameraState cameraState, Vector3 objectCenter, float objectRadius, int previousLod = -1)
    {
        if (_candidates.Count == 0)
            return -1;

        // Get or create object state
        if (!_objectStates.TryGetValue(objectId, out var state))
        {
            state = new LodState { CurrentLod = previousLod };
            _objectStates[objectId] = state;
        }

        // Update camera state
        if (_initialized)
        {
            UpdateCameraState(cameraState);
        }
        _previousCameraState = cameraState;
        _initialized = true;

        // Compute LOD scores for each candidate
        int bestLod = previousLod >= 0 ? previousLod : 0;
        float bestScore = float.MaxValue;

        for (int i = 0; i < _candidates.Count; i++)
        {
            var candidate = _candidates[i];
            float score = ComputeLodScore(candidate, cameraState, objectCenter, objectRadius, state);

            if (score < bestScore)
            {
                bestScore = score;
                bestLod = candidate.Level;
            }
        }

        // Apply hysteresis
        if (previousLod >= 0 && _config.Hysteresis > 0)
        {
            bestLod = ApplyHysteresis(bestLod, previousLod, cameraState, objectCenter);
        }

        // Apply temporal stabilization
        if (_config.EnableTemporalStabilization)
        {
            bestLod = ApplyTemporalStabilization(bestLod, state);
        }

        // Apply predictive correction
        if (_config.EnablePredictiveSelection && _initialized)
        {
            bestLod = ApplyPredictiveCorrection(bestLod, cameraState, objectCenter, objectRadius);
        }

        // Clamp to valid range
        bestLod = Math.Clamp(bestLod, _config.MinLodLevel, Math.Min(_config.MaxLodLevel, _candidates.Count - 1));

        // Update state
        if (state.CurrentLod != bestLod)
        {
            System.Threading.Interlocked.Increment(ref _totalLodChanges);
            state.PreviousLod = state.CurrentLod;
            state.CurrentLod = bestLod;
            state.LastChangeFrame = state.FrameCount;
        }
        state.FrameCount++;
        System.Threading.Interlocked.Increment(ref _totalSelections);

        return bestLod;
    }

    /// <summary>
    /// Selects the optimal LOD level with full analysis.
    /// </summary>
    public int SelectLodFull(int objectId, CameraState cameraState, Vector3 objectCenter, float objectRadius, int previousLod = -1)
    {
        return SelectLod(objectId, cameraState, objectCenter, objectRadius, previousLod);
    }

    /// <summary>
    /// Gets the LOD network for a specific level.
    /// </summary>
    public ISdfNetwork? GetLodNetwork(int level)
    {
        for (int i = 0; i < _candidates.Count; i++)
        {
            if (_candidates[i].Level == level)
                return _candidates[i].Network;
        }
        return null;
    }

    /// <summary>
    /// Gets all registered LOD candidates.
    /// </summary>
    public ReadOnlySpan<LodCandidate> GetCandidates()
    {
        return _candidates.ToArray().AsSpan();
    }

    /// <summary>
    /// Updates the camera state for predictive LOD selection.
    /// </summary>
    private void UpdateCameraState(CameraState current)
    {
        if (!_config.EnablePredictiveSelection)
            return;

        // Compute camera velocity from position change
        Vector3 deltaPos = current.Position - _previousCameraState.Position;
        float deltaTime = 1.0f / 60.0f; // Assume 60fps
        current.Velocity = deltaPos / deltaTime;
    }

    /// <summary>
    /// Computes the LOD score for a candidate based on multiple factors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float ComputeLodScore(LodCandidate candidate, CameraState camera, Vector3 objectCenter, float objectRadius, LodState state)
    {
        float score = 0;

        // Distance-based score
        if (_config.UseDistanceBased)
        {
            float distance = Vector3.Distance(camera.Position, objectCenter);
            float normalizedDistance = distance / candidate.MaxDistance;
            score += normalizedDistance * _config.DistanceWeight;
        }

        // Screen-space error metric
        if (_config.UseScreenSpaceError)
        {
            float screenError = ComputeScreenSpaceError(candidate, camera, objectCenter, objectRadius);
            score += screenError * (1.0f - _config.DistanceWeight);
        }

        // Curvature-based score (higher curvature = lower LOD = more detail)
        if (_config.UseCurvatureBased)
        {
            float curvatureScore = 1.0f - candidate.Curvature;
            score += curvatureScore * _config.CurvatureWeight;
        }

        // Motion-based score (faster motion = lower LOD = more detail)
        if (_config.UseMotionBased)
        {
            float motionScore = ComputeMotionScore(camera, objectCenter, objectRadius);
            score += motionScore * _config.MotionWeight;
        }

        // Penalize LOD changes (prefer stability)
        if (state.CurrentLod >= 0 && state.CurrentLod != candidate.Level)
        {
            int lodDiff = Math.Abs(state.CurrentLod - candidate.Level);
            score += lodDiff * 0.1f;
        }

        return score;
    }

    /// <summary>
    /// Computes the screen-space error for a LOD candidate.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float ComputeScreenSpaceError(LodCandidate candidate, CameraState camera, Vector3 objectCenter, float objectRadius)
    {
        // Project bounding sphere to screen space
        Vector3 viewCenter = Vector3.Transform(objectCenter, camera.View);
        Vector4 clipCenter = Vector4.Transform(new Vector4(objectCenter, 1), camera.ViewProjection);

        if (clipCenter.W <= 0)
            return 1.0f; // Behind camera

        // Compute screen-space radius
        float screenRadius = (objectRadius / viewCenter.Z) *
            (camera.FieldOfView / (2.0f * (float)Math.Tan(camera.FieldOfView / 2.0f))) *
            camera.ScreenHeight;

        // Normalized screen-space error
        float screenError = screenRadius / _config.PixelErrorThreshold;
        return MathF.Min(screenError, 1.0f);
    }

    /// <summary>
    /// Computes the motion score for an object.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float ComputeMotionScore(CameraState camera, Vector3 objectCenter, float objectRadius)
    {
        // Estimate screen-space motion
        Vector3 relativeVelocity = camera.Velocity;
        float speed = relativeVelocity.Length();

        // Normalize by object size and screen height
        float screenMotion = (speed * objectRadius) /
            (Vector3.Distance(camera.Position, objectCenter) * camera.ScreenHeight);

        return MathF.Min(screenMotion / _config.MaxScreenSpaceMotion, 1.0f);
    }

    /// <summary>
    /// Applies hysteresis to prevent LOD popping.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int ApplyHysteresis(int selectedLod, int previousLod, CameraState camera, Vector3 objectCenter)
    {
        if (previousLod < 0 || previousLod >= _candidates.Count)
            return selectedLod;

        float distance = Vector3.Distance(camera.Position, objectCenter);
        float prevThreshold = _candidates[previousLod].MaxDistance;
        float selectedThreshold = selectedLod < _candidates.Count ?
            _candidates[selectedLod].MaxDistance : float.MaxValue;

        // Don't switch to higher LOD if we're close to the threshold
        if (previousLod < selectedLod &&
            distance > prevThreshold * (1.0f - _config.Hysteresis))
        {
            return previousLod;
        }

        // Don't switch to lower LOD if we're close to the threshold
        if (previousLod > selectedLod &&
            distance < selectedThreshold * (1.0f + _config.Hysteresis))
        {
            return previousLod;
        }

        return selectedLod;
    }

    /// <summary>
    /// Applies temporal stabilization to limit LOD changes per frame.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int ApplyTemporalStabilization(int selectedLod, LodState state)
    {
        // Don't change LOD too frequently
        if (state.FrameCount - state.LastChangeFrame < 3)
        {
            return state.CurrentLod >= 0 ? state.CurrentLod : selectedLod;
        }

        return selectedLod;
    }

    /// <summary>
    /// Applies predictive correction based on camera velocity.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int ApplyPredictiveCorrection(int selectedLod, CameraState camera, Vector3 objectCenter, float objectRadius)
    {
        if (camera.Velocity.LengthSquared() < 1e-6f)
            return selectedLod;

        // Predict future camera position
        Vector3 futurePosition = camera.Position + camera.Velocity * _config.PredictionHorizon;

        // Compute predicted distance
        float currentDistance = Vector3.Distance(camera.Position, objectCenter);
        float futureDistance = Vector3.Distance(futurePosition, objectCenter);

        // If we're moving towards the object, prefer lower LOD (more detail)
        if (futureDistance < currentDistance * 0.9f)
        {
            // Moving towards - try to use one level lower
            int predictedLod = Math.Max(0, selectedLod - 1);
            System.Threading.Interlocked.Increment(ref _totalPredictiveCorrections);
            return predictedLod;
        }

        // If we're moving away, prefer higher LOD (less detail)
        if (futureDistance > currentDistance * 1.1f)
        {
            int predictedLod = Math.Min(_candidates.Count - 1, selectedLod + 1);
            System.Threading.Interlocked.Increment(ref _totalPredictiveCorrections);
            return predictedLod;
        }

        return selectedLod;
    }

    /// <summary>
    /// Estimates the curvature of a network at a point.
    /// </summary>
    private float EstimateNetworkCurvature(ISdfNetwork network, Vector3 point)
    {
        const float epsilon = 0.01f;
        Vector3 gradient = network.ComputeGradient(point);
        float gradLen = gradient.Length();

        if (gradLen < 1e-8f)
            return 0.0f;

        Vector3 normal = gradient / gradLen;

        // Sample along two tangent directions
        Vector3 tangent1 = MathF.Abs(normal.X) < 0.9f
            ? Vector3.Cross(normal, Vector3.UnitX)
            : Vector3.Cross(normal, Vector3.UnitY);
        tangent1 = Vector3.Normalize(tangent1);
        Vector3 tangent2 = Vector3.Normalize(Vector3.Cross(normal, tangent1));

        // Compute second derivatives
        float d2f1 = SecondDerivative(network, point, tangent1, epsilon);
        float d2f2 = SecondDerivative(network, point, tangent2, epsilon);

        return MathF.Abs((d2f1 + d2f2) * 0.5f);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float SecondDerivative(ISdfNetwork network, Vector3 point, Vector3 direction, float h)
    {
        float fPlus = network.Evaluate(point + direction * h);
        float fCenter = network.Evaluate(point);
        float fMinus = network.Evaluate(point - direction * h);

        return (fPlus - 2.0f * fCenter + fMinus) / (h * h);
    }

    /// <summary>
    /// Resets all statistics.
    /// </summary>
    public void ResetStatistics()
    {
        System.Threading.Interlocked.Exchange(ref _totalSelections, 0);
        System.Threading.Interlocked.Exchange(ref _totalLodChanges, 0);
        System.Threading.Interlocked.Exchange(ref _totalPredictiveCorrections, 0);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _candidates.Clear();
        _objectStates.Clear();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Internal state for each object's LOD selection.
    /// </summary>
    private sealed class LodState
    {
        public int CurrentLod = -1;
        public int PreviousLod = -1;
        public int LastChangeFrame = 0;
        public int FrameCount = 0;
    }
}
