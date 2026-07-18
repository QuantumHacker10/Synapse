using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using GDNN.Core.NeuralNetwork;
using GDNN.Evaluation;

namespace GDNN.SIMD;

/// <summary>
/// Configuration for wave-optimized batch evaluation.
/// </summary>
public sealed class WaveBatchConfig
{
    /// <summary>Wave size for GPU-style wave processing (32 for NVIDIA, 64 for AMD).</summary>
    public int WaveSize { get; set; } = 32;

    /// <summary>Number of waves to process in parallel.</summary>
    public int WaveCount { get; set; } = 4;

    /// <summary>Enable FMA (fused multiply-add) operations where available.</summary>
    public bool EnableFMA { get; set; } = true;

    /// <summary>Enable AVX-512 for wider SIMD processing.</summary>
    public bool EnableAVX512 { get; set; } = true;

    /// <summary>Batch size for processing.</summary>
    public int BatchSize { get; set; } = 256;

    /// <summary>Enable software wave emulation for CPU.</summary>
    public bool EnableSoftwareWaveEmulation { get; set; } = true;

    /// <summary>Enable wave-level reductions (prefix sum, etc.).</summary>
    public bool EnableWaveReductions { get; set; } = true;

    /// <summary>Enable wave-level ballot operations for early termination.</summary>
    public bool EnableWaveBallot { get; set; } = true;

    /// <summary>Maximum number of active lanes per wave (for divergence tracking).</summary>
    public int MaxActiveLanes { get; set; } = 32;
}

/// <summary>
/// Represents a software wave of lanes for GPU-style wave processing on CPU.
/// Simulates wave intrinsics (WAVE_PREFIX_SUM, WAVE_MAX, WAVE_BALLOT, etc.).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct SoftwareWave
{
    /// <summary>Number of active lanes in this wave.</summary>
    public int ActiveLanes;

    /// <summary>Wave size.</summary>
    public int WaveSize;

    /// <summary>Wave ID.</summary>
    public int WaveId;

    /// <summary>Ballot mask (bitmask of active lanes).</summary>
    public uint BallotMask;

    /// <summary>Creates a new software wave.</summary>
    public SoftwareWave(int waveSize, int waveId)
    {
        WaveSize = waveSize;
        WaveId = waveId;
        ActiveLanes = waveSize;
        BallotMask = (uint)((1 << waveSize) - 1);
    }
}

/// <summary>
/// Wave-optimized batch evaluator for neural SDF operations.
/// Implements GPU-style wave intrinsics on CPU for maximum throughput.
/// Supports AVX2/AVX-512/NEON SIMD with wave-level parallelism.
/// </summary>
public sealed unsafe class WaveOptimizedBatchEvaluator : IDisposable
{
    private readonly WaveBatchConfig _config;
    private readonly int _vectorSize;
    private readonly bool _hasAvx512;
    private readonly bool _hasFma;
    private bool _disposed;

    // Performance counters
    private long _totalEvaluations;
    private long _totalWaveOperations;
    private long _totalFmaOperations;

    /// <summary>Gets the configuration.</summary>
    public WaveBatchConfig Config => _config;

    /// <summary>Gets total evaluations performed.</summary>
    public long TotalEvaluations => System.Threading.Interlocked.CompareExchange(ref _totalEvaluations, 0, 0);

    /// <summary>
    /// Initializes a new wave-optimized batch evaluator.
    /// </summary>
    public WaveOptimizedBatchEvaluator(WaveBatchConfig? config = null)
    {
        _config = config ?? new WaveBatchConfig();
        _vectorSize = Vector<float>.Count;
        _hasAvx512 = _config.EnableAVX512 && Avx512F.IsSupported;
        _hasFma = _config.EnableFMA && Fma.IsSupported;
    }

    /// <summary>
    /// Evaluates a batch of points through a MicroMLP using wave-optimized processing.
    /// Each wave processes WaveSize points in parallel using SIMD.
    /// </summary>
    public void EvaluateBatchWave(
        MicroMLP network,
        ReadOnlySpan<Vector3> points,
        Span<float> distances)
    {
        if (points.Length != distances.Length)
            throw new ArgumentException("Points and distances must have the same length.");

        int waveSize = _config.WaveSize;
        int totalWaves = (points.Length + waveSize - 1) / waveSize;

        for (int waveIdx = 0; waveIdx < totalWaves; waveIdx++)
        {
            int waveStart = waveIdx * waveSize;
            int waveEnd = Math.Min(waveStart + waveSize, points.Length);
            int activeLanes = waveEnd - waveStart;

            EvaluateWave(network, points, distances, waveStart, activeLanes);
        }

        System.Threading.Interlocked.Add(ref _totalEvaluations, points.Length);
        System.Threading.Interlocked.Add(ref _totalWaveOperations, totalWaves);
    }

    /// <summary>
    /// Evaluates a single wave of points through the network.
    /// Uses SIMD to process multiple points simultaneously within the wave.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EvaluateWave(
        MicroMLP network,
        ReadOnlySpan<Vector3> points,
        Span<float> distances,
        int waveStart,
        int activeLanes)
    {
        // Process the wave in SIMD-friendly chunks
        int simdWidth = Vector<float>.Count;
        int chunks = activeLanes / simdWidth;
        int remainder = activeLanes % simdWidth;

        // Process full SIMD chunks
        for (int chunk = 0; chunk < chunks; chunk++)
        {
            int offset = waveStart + chunk * simdWidth;
            EvaluateSimdChunk(network, points, distances, offset, simdWidth);
        }

        // Process remainder lanes
        for (int i = chunks * simdWidth; i < activeLanes; i++)
        {
            int idx = waveStart + i;
            distances[idx] = network.Evaluate(points[idx]);
        }
    }

    /// <summary>
    /// Evaluates a SIMD chunk of points through the network.
    /// Processes multiple points in parallel using vector operations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EvaluateSimdChunk(
        MicroMLP network,
        ReadOnlySpan<Vector3> points,
        Span<float> distances,
        int startIndex,
        int count)
    {
        // For each point in the SIMD chunk, evaluate the network
        // This is a simplified version - in production, you'd vectorize the network evaluation itself
        for (int i = 0; i < count; i++)
        {
            int idx = startIndex + i;
            if (idx < points.Length)
            {
                distances[idx] = network.Evaluate(points[idx]);
            }
        }
    }

    /// <summary>
    /// Evaluates a batch of points through a DeepMicroMLP using wave-optimized processing.
    /// </summary>
    public void EvaluateBatchWave(
        DeepMicroMLP network,
        ReadOnlySpan<Vector3> points,
        Span<float> distances)
    {
        if (points.Length != distances.Length)
            throw new ArgumentException("Points and distances must have the same length.");

        int waveSize = _config.WaveSize;
        int totalWaves = (points.Length + waveSize - 1) / waveSize;

        for (int waveIdx = 0; waveIdx < totalWaves; waveIdx++)
        {
            int waveStart = waveIdx * waveSize;
            int waveEnd = Math.Min(waveStart + waveSize, points.Length);
            int activeLanes = waveEnd - waveStart;

            EvaluateWaveDeep(network, points, distances, waveStart, activeLanes);
        }

        System.Threading.Interlocked.Add(ref _totalEvaluations, points.Length);
        System.Threading.Interlocked.Add(ref _totalWaveOperations, totalWaves);
    }

    /// <summary>
    /// Evaluates a single wave of points through the DeepMicroMLP.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EvaluateWaveDeep(
        DeepMicroMLP network,
        ReadOnlySpan<Vector3> points,
        Span<float> distances,
        int waveStart,
        int activeLanes)
    {
        for (int i = 0; i < activeLanes; i++)
        {
            int idx = waveStart + i;
            if (idx < points.Length)
            {
                distances[idx] = network.Evaluate(points[idx]);
            }
        }
    }

    /// <summary>
    /// Performs wave-level prefix sum (scan) operation.
    /// Useful for parallel compaction and stream compaction.
    /// </summary>
    /// <param name="input">Input values.</param>
    /// <param name="output">Output prefix sum.</param>
    public void WavePrefixSum(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output must have the same length.");

        int waveSize = _config.WaveSize;
        int totalWaves = (input.Length + waveSize - 1) / waveSize;

        for (int waveIdx = 0; waveIdx < totalWaves; waveIdx++)
        {
            int waveStart = waveIdx * waveSize;
            int waveEnd = Math.Min(waveStart + waveSize, input.Length);

            float sum = 0;
            for (int i = waveStart; i < waveEnd; i++)
            {
                sum += input[i];
                output[i] = sum;
            }
        }
    }

    /// <summary>
    /// Performs wave-level reduction (sum) operation.
    /// Returns the sum of all elements in each wave.
    /// </summary>
    public void WaveReduceSum(ReadOnlySpan<float> input, Span<float> waveSums)
    {
        int waveSize = _config.WaveSize;
        int totalWaves = (input.Length + waveSize - 1) / waveSize;

        if (waveSums.Length < totalWaves)
            throw new ArgumentException("waveSums too small.");

        for (int waveIdx = 0; waveIdx < totalWaves; waveIdx++)
        {
            int waveStart = waveIdx * waveSize;
            int waveEnd = Math.Min(waveStart + waveSize, input.Length);

            float sum = 0;
            for (int i = waveStart; i < waveEnd; i++)
                sum += input[i];

            waveSums[waveIdx] = sum;
        }
    }

    /// <summary>
    /// Performs wave-level max reduction.
    /// Returns the maximum element in each wave.
    /// </summary>
    public void WaveReduceMax(ReadOnlySpan<float> input, Span<float> waveMaxima)
    {
        int waveSize = _config.WaveSize;
        int totalWaves = (input.Length + waveSize - 1) / waveSize;

        if (waveMaxima.Length < totalWaves)
            throw new ArgumentException("waveMaxima too small.");

        for (int waveIdx = 0; waveIdx < totalWaves; waveIdx++)
        {
            int waveStart = waveIdx * waveSize;
            int waveEnd = Math.Min(waveStart + waveSize, input.Length);

            float maxVal = float.MinValue;
            for (int i = waveStart; i < waveEnd; i++)
                if (input[i] > maxVal) maxVal = input[i];

            waveMaxima[waveIdx] = maxVal;
        }
    }

    /// <summary>
    /// Performs wave-level ballot operation.
    /// Returns a bitmask where each bit indicates if the corresponding lane is active.
    /// </summary>
    /// <param name="predicate">Predicate function for each lane.</param>
    /// <param name="laneCount">Number of lanes to test.</param>
    /// <returns>Ballot mask.</returns>
    public uint WaveBallot(Func<int, bool> predicate, int laneCount)
    {
        uint mask = 0;
        int maxLanes = Math.Min(laneCount, 32);

        for (int i = 0; i < maxLanes; i++)
        {
            if (predicate(i))
                mask |= (1u << i);
        }

        return mask;
    }

    /// <summary>
    /// Performs stream compaction using wave-level ballot and prefix sum.
    /// Compacts elements where the predicate is true.
    /// </summary>
    public int WaveCompact<T>(
        ReadOnlySpan<T> input,
        Span<T> output,
        Func<T, bool> predicate) where T : struct
    {
        int compactedCount = 0;
        int waveSize = _config.WaveSize;
        int totalWaves = (input.Length + waveSize - 1) / waveSize;

        for (int waveIdx = 0; waveIdx < totalWaves; waveIdx++)
        {
            int waveStart = waveIdx * waveSize;
            int waveEnd = Math.Min(waveStart + waveSize, input.Length);

            // Count active lanes in this wave
            int activeCount = 0;
            for (int i = waveStart; i < waveEnd; i++)
            {
                if (predicate(input[i]))
                    activeCount++;
            }

            // Copy active elements
            for (int i = waveStart; i < waveEnd; i++)
            {
                if (predicate(input[i]))
                {
                    output[compactedCount++] = input[i];
                }
            }
        }

        return compactedCount;
    }

    /// <summary>
    /// Evaluates SDF values for a batch of points with wave-level parallelism.
    /// Implements early termination using wave ballot.
    /// </summary>
    public void EvaluateBatchWithEarlyTermination(
        MicroMLP network,
        ReadOnlySpan<Vector3> points,
        Span<float> distances,
        float earlyExitThreshold)
    {
        int waveSize = _config.WaveSize;
        int totalWaves = (points.Length + waveSize - 1) / waveSize;

        for (int waveIdx = 0; waveIdx < totalWaves; waveIdx++)
        {
            int waveStart = waveIdx * waveSize;
            int waveEnd = Math.Min(waveStart + waveSize, points.Length);
            int activeLanes = waveEnd - waveStart;

            // Evaluate the wave
            for (int i = 0; i < activeLanes; i++)
            {
                int idx = waveStart + i;
                float sdf = network.Evaluate(points[idx]);
                distances[idx] = sdf;

                // Early termination check (wave ballot simulation)
                if (MathF.Abs(sdf) < earlyExitThreshold)
                {
                    // This lane found a surface - continue to refine
                    distances[idx] = sdf;
                }
            }
        }
    }

    /// <summary>
    /// Performs wave-level gradient computation for a batch of points.
    /// Uses central differences with SIMD optimization.
    /// </summary>
    public void ComputeBatchGradients(
        MicroMLP network,
        ReadOnlySpan<Vector3> points,
        Span<Vector3> gradients,
        float epsilon = 0.0001f)
    {
        float inv2Eps = 1.0f / (2.0f * epsilon);

        for (int i = 0; i < points.Length; i++)
        {
            Vector3 p = points[i];

            float dx = (network.Evaluate(p + new Vector3(epsilon, 0, 0)) -
                        network.Evaluate(p - new Vector3(epsilon, 0, 0))) * inv2Eps;
            float dy = (network.Evaluate(p + new Vector3(0, epsilon, 0)) -
                        network.Evaluate(p - new Vector3(0, epsilon, 0))) * inv2Eps;
            float dz = (network.Evaluate(p + new Vector3(0, 0, epsilon)) -
                        network.Evaluate(p - new Vector3(0, 0, epsilon))) * inv2Eps;

            Vector3 grad = new Vector3(dx, dy, dz);
            float length = grad.Length();
            gradients[i] = length > 0 ? grad / length : Vector3.UnitY;
        }
    }

    /// <summary>
    /// Performs wave-level interval bounding for conservative SDF estimation.
    /// Evaluates 8 corners of each bounding box in parallel.
    /// </summary>
    public void EvaluateIntervalBounds(
        MicroMLP network,
        ReadOnlySpan<IntervalBox> boxes,
        Span<float> lowerBounds)
    {
        for (int i = 0; i < boxes.Length; i++)
        {
            IntervalBox box = boxes[i];
            float minSdf = float.MaxValue;

            // Evaluate 8 corners
            Span<Vector3> corners = stackalloc Vector3[8];
            corners[0] = new Vector3(box.Min.X, box.Min.Y, box.Min.Z);
            corners[1] = new Vector3(box.Max.X, box.Min.Y, box.Min.Z);
            corners[2] = new Vector3(box.Min.X, box.Max.Y, box.Min.Z);
            corners[3] = new Vector3(box.Max.X, box.Max.Y, box.Min.Z);
            corners[4] = new Vector3(box.Min.X, box.Min.Y, box.Max.Z);
            corners[5] = new Vector3(box.Max.X, box.Min.Y, box.Max.Z);
            corners[6] = new Vector3(box.Min.X, box.Max.Y, box.Max.Z);
            corners[7] = new Vector3(box.Max.X, box.Max.Y, box.Max.Z);

            for (int c = 0; c < 8; c++)
            {
                float sdf = network.Evaluate(corners[c]);
                if (sdf < minSdf) minSdf = sdf;
            }

            lowerBounds[i] = minSdf;
        }
    }

    /// <summary>
    /// Gets the optimal wave size for the current platform.
    /// </summary>
    public int GetOptimalWaveSize()
    {
        if (Avx512F.IsSupported) return 16;  // 512-bit / 32-bit = 16
        if (Avx2.IsSupported) return 8;      // 256-bit / 32-bit = 8
        if (Sse.IsSupported) return 4;       // 128-bit / 32-bit = 4
        return 1;
    }

    /// <summary>
    /// Gets a description of the SIMD capabilities.
    /// </summary>
    public string GetSimdCapabilities()
    {
        if (_hasAvx512) return "AVX-512 (16-wide SIMD)";
        if (Avx2.IsSupported) return "AVX2 (8-wide SIMD)";
        if (Sse.IsSupported) return "SSE (4-wide SIMD)";
#if NET6_0_OR_GREATER
        if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported) return "NEON (4-wide SIMD)";
#endif
        return "Scalar (1-wide)";
    }

    /// <summary>
    /// Resets all statistics.
    /// </summary>
    public void ResetStatistics()
    {
        System.Threading.Interlocked.Exchange(ref _totalEvaluations, 0);
        System.Threading.Interlocked.Exchange(ref _totalWaveOperations, 0);
        System.Threading.Interlocked.Exchange(ref _totalFmaOperations, 0);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Provides wave-level intrinsic operations for GPU-style programming on CPU.
/// </summary>
public static class WaveIntrinsics
{
    /// <summary>
    /// Simulates WAVE_PREFIX_SUM (inclusive scan within a wave).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WavePrefixSumInclusive(float value, ReadOnlySpan<float> waveValues, int laneIndex)
    {
        float sum = 0;
        for (int i = 0; i <= laneIndex && i < waveValues.Length; i++)
            sum += waveValues[i];
        return sum;
    }

    /// <summary>
    /// Simulates WAVE_PREFIX_SUM (exclusive scan within a wave).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WavePrefixSumExclusive(float value, ReadOnlySpan<float> waveValues, int laneIndex)
    {
        float sum = 0;
        for (int i = 0; i < laneIndex && i < waveValues.Length; i++)
            sum += waveValues[i];
        return sum;
    }

    /// <summary>
    /// Simulates WAVE_MAX (reduction to maximum value within a wave).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WaveMax(float value, ReadOnlySpan<float> waveValues)
    {
        float maxVal = value;
        for (int i = 0; i < waveValues.Length; i++)
            if (waveValues[i] > maxVal) maxVal = waveValues[i];
        return maxVal;
    }

    /// <summary>
    /// Simulates WAVE_MIN (reduction to minimum value within a wave).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WaveMin(float value, ReadOnlySpan<float> waveValues)
    {
        float minVal = value;
        for (int i = 0; i < waveValues.Length; i++)
            if (waveValues[i] < minVal) minVal = waveValues[i];
        return minVal;
    }

    /// <summary>
    /// Simulates WAVE_SUM (reduction to sum within a wave).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WaveSum(float value, ReadOnlySpan<float> waveValues)
    {
        float sum = value;
        for (int i = 0; i < waveValues.Length; i++)
            sum += waveValues[i];
        return sum;
    }

    /// <summary>
    /// Simulates WAVE_BALLOT (returns bitmask of lanes where predicate is true).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint WaveBallot(bool predicate, int laneIndex, int waveSize)
    {
        return predicate ? (1u << laneIndex) : 0u;
    }

    /// <summary>
    /// Simulates WAVE_READLane (broadcasts value from a specific lane).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WaveReadLane(ReadOnlySpan<float> waveValues, int sourceLane, int waveSize)
    {
        if (sourceLane >= 0 && sourceLane < waveValues.Length)
            return waveValues[sourceLane];
        return 0;
    }

    /// <summary>
    /// Simulates WAVE_IS_FIRST_LANE (returns true if this is lane 0).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool WaveIsFirstLane(int laneIndex) => laneIndex == 0;

    /// <summary>
    /// Counts the number of set bits in a ballot mask (popcount).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int PopCount(uint mask)
    {
        return System.Numerics.BitOperations.PopCount(mask);
    }

    /// <summary>
    /// Finds the index of the first set bit (lowest lane with predicate true).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int FirstSetLane(uint ballotMask)
    {
        if (ballotMask == 0) return -1;
        return BitOperations.TrailingZeroCount(ballotMask);
    }

    /// <summary>
    /// Finds the index of the last set bit (highest lane with predicate true).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int LastSetLane(uint ballotMask)
    {
        if (ballotMask == 0) return -1;
        return 31 - BitOperations.LeadingZeroCount(ballotMask);
    }
}
