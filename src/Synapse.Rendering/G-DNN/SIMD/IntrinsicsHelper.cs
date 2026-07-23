using System;
// ============================================================
// FILE: IntrinsicsHelper.cs
// PATH: SIMD/IntrinsicsHelper.cs
// ============================================================


// SPDX-License-Identifier: MIT
// GDNN Neural Geometry Engine - SIMD Intrinsics Helper
// Runtime SIMD capability detection, platform dispatch, and benchmarking.

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.SIMD;

/// <summary>
/// Provides runtime detection of SIMD capabilities and platform-specific dispatch
/// for optimal code path selection across x86-64 (SSE/AVX) and ARM64 (NEON/SVE2).
/// </summary>
public static class IntrinsicsHelper
{
    /// <summary>Cached AVX2 support flag, initialized once.</summary>
    private static readonly Lazy<bool> s_avx2Supported = new(() => Avx2.IsSupported);

    /// <summary>Cached AVX-512 Foundation support flag.</summary>
    private static readonly Lazy<bool> s_avx512FSupported = new(() => Avx512F.IsSupported);

    /// <summary>Cached AVX-512 BW support flag.</summary>
    private static readonly Lazy<bool> s_avx512BWSupported = new(() => Avx512BW.IsSupported);

    /// <summary>Cached SSE4.2 support flag.</summary>
    private static readonly Lazy<bool> s_sse42Supported = new(() => Sse42.IsSupported);

    /// <summary>Cached SSE2 support flag.</summary>
    private static readonly Lazy<bool> s_sse2Supported = new(() => Sse2.IsSupported);

    /// <summary>Cached ARM AdvSimd (NEON) support flag.</summary>
    private static readonly Lazy<bool> s_armNeonSupported = new(() => AdvSimd.IsSupported);

    /// <summary>Cached ARM SVE2 support flag.</summary>
    private static readonly Lazy<bool> s_sve2Supported = new(() => Sve2.IsSupported);

    /// <summary>Cached ARM SVE support flag.</summary>
    private static readonly Lazy<bool> s_sveSupported = new(() => Sve.IsSupported);

    /// <summary>Cached Vector{T}.IsHardwareAccelerated flag.</summary>
    private static readonly Lazy<bool> s_vectorAccelerated = new(() => Vector.IsHardwareAccelerated);

    /// <summary>Cached Vector256 hardware acceleration flag.</summary>
    private static readonly Lazy<bool> s_vector256Accelerated = new(() => Vector256.IsHardwareAccelerated);

    /// <summary>Cached Vector128 hardware acceleration flag.</summary>
    private static readonly Lazy<bool> s_vector128Accelerated = new(() => Vector128.IsHardwareAccelerated);

    /// <summary>Cached preferred SIMD vector size in bytes.</summary>
    private static readonly Lazy<int> s_preferredVectorSize = new(DetectPreferredVectorSize);

    /// <summary>Cached SIMD vector width in float elements.</summary>
    private static readonly Lazy<int> s_vectorWidthFloat = new(DetectVectorWidthFloat);

    /// <summary>Cached FMA support flag.</summary>
    private static readonly Lazy<bool> s_fmaSupported = new(() => Fma.IsSupported);

    /// <summary>Cached BMI2 support flag.</summary>
    private static readonly Lazy<bool> s_bmi2Supported = new(() => Bmi2.X64.IsSupported);

    /// <summary>Cached LZCNT support flag.</summary>
    private static readonly Lazy<bool> s_lzcntSupported = new(() => Lzcnt.IsSupported);

    /// <summary>Cached POPCNT support flag.</summary>
    private static readonly Lazy<bool> s_popcntSupported = new(() => Popcnt.IsSupported);

    /// <summary>Cached PCLMULQDQ support flag.</summary>
    private static readonly Lazy<bool> s_pclmulqdqSupported = new(() => Pclmulqdq.IsSupported);

    /// <summary>The best SIMD platform detected for this runtime.</summary>
    private static readonly Lazy<SimdPlatform> s_detectedPlatform = new(DetectPlatform);

    /// <summary>
    /// Gets a value indicating whether AVX2 instructions are available at runtime.
    /// </summary>
    public static bool IsAvx2Supported => s_avx2Supported.Value;

    /// <summary>
    /// Gets a value indicating whether AVX-512 Foundation instructions are available.
    /// </summary>
    public static bool IsAvx512FSupported => s_avx512FSupported.Value;

    /// <summary>
    /// Gets a value indicating whether AVX-512 Byte/Word instructions are available.
    /// </summary>
    public static bool IsAvx512BWSupported => s_avx512BWSupported.Value;

    /// <summary>
    /// Gets a value indicating whether SSE4.2 instructions are available.
    /// </summary>
    public static bool IsSse42Supported => s_sse42Supported.Value;

    /// <summary>
    /// Gets a value indicating whether SSE2 instructions are available.
    /// </summary>
    public static bool IsSse2Supported => s_sse2Supported.Value;

    /// <summary>
    /// Gets a value indicating whether ARM NEON (AdvSimd) instructions are available.
    /// </summary>
    public static bool IsArmNeonSupported => s_armNeonSupported.Value;

    /// <summary>
    /// Gets a value indicating whether ARM SVE2 instructions are available.
    /// </summary>
    public static bool IsSve2Supported => s_sve2Supported.Value;

    /// <summary>
    /// Gets a value indicating whether ARM SVE instructions are available.
    /// </summary>
    public static bool IsSveSupported => s_sveSupported.Value;

    /// <summary>
    /// Gets a value indicating whether <see cref="Vector{T}"/> is hardware accelerated.
    /// </summary>
    public static bool IsVectorAccelerated => s_vectorAccelerated.Value;

    /// <summary>
    /// Gets a value indicating whether 256-bit SIMD vectors are hardware accelerated.
    /// </summary>
    public static bool IsVector256Accelerated => s_vector256Accelerated.Value;

    /// <summary>
    /// Gets a value indicating whether 128-bit SIMD vectors are hardware accelerated.
    /// </summary>
    public static bool IsVector128Accelerated => s_vector128Accelerated.Value;

    /// <summary>
    /// Gets a value indicating whether FMA (fused multiply-add) instructions are available.
    /// </summary>
    public static bool IsFmaSupported => s_fmaSupported.Value;

    /// <summary>
    /// Gets a value indicating whether BMI2 instructions are available.
    /// </summary>
    public static bool IsBmi2Supported => s_bmi2Supported.Value;

    /// <summary>
    /// Gets a value indicating whether LZCNT (leading zero count) instructions are available.
    /// </summary>
    public static bool IsLzcntSupported => s_lzcntSupported.Value;

    /// <summary>
    /// Gets a value indicating whether POPCNT (population count) instructions are available.
    /// </summary>
    public static bool IsPopcntSupported => s_popcntSupported.Value;

    /// <summary>
    /// Gets a value indicating whether PCLMULQDQ (carry-less multiply) instructions are available.
    /// </summary>
    public static bool IsPclmulqdqSupported => s_pclmulqdqSupported.Value;

    /// <summary>
    /// Gets the preferred SIMD vector size in bytes for the current platform.
    /// </summary>
    public static int PreferredVectorSize => s_preferredVectorSize.Value;

    /// <summary>
    /// Gets the number of float elements that fit in the preferred SIMD vector width.
    /// </summary>
    public static int VectorWidthFloat => s_vectorWidthFloat.Value;

    /// <summary>
    /// Gets the detected best SIMD platform for this runtime.
    /// </summary>
    public static SimdPlatform DetectedPlatform => s_detectedPlatform.Value;

    /// <summary>
    /// Gets a detailed report of all detected SIMD capabilities.
    /// </summary>
    public static string CapabilityReport
    {
        get
        {
            var sb = new System.Text.StringBuilder(1024);
            sb.AppendLine("=== GDNN SIMD Capability Report ===");
            sb.AppendLine($"  Platform:           {RuntimeInformation.ProcessArchitecture}");
            sb.AppendLine($"  OS:                 {RuntimeInformation.OSDescription}");
            sb.AppendLine($"  Detected SIMD:      {DetectedPlatform}");
            sb.AppendLine($"  Preferred Vec Size: {PreferredVectorSize} bytes");
            sb.AppendLine($"  Vec Width (float):  {VectorWidthFloat}");
            sb.AppendLine($"  Vector<T> HW Accel: {IsVectorAccelerated}");
            sb.AppendLine($"  Vector256 HW Accel: {IsVector256Accelerated}");
            sb.AppendLine($"  Vector128 HW Accel: {IsVector128Accelerated}");
            sb.AppendLine("  --- x86-64 ---");
            sb.AppendLine($"  SSE2:    {IsSse2Supported}");
            sb.AppendLine($"  SSE4.2:  {IsSse42Supported}");
            sb.AppendLine($"  AVX2:    {IsAvx2Supported}");
            sb.AppendLine($"  AVX-512F:{IsAvx512FSupported}");
            sb.AppendLine($"  AVX-512BW:{IsAvx512BWSupported}");
            sb.AppendLine($"  FMA:     {IsFmaSupported}");
            sb.AppendLine($"  BMI2:    {IsBmi2Supported}");
            sb.AppendLine($"  LZCNT:   {IsLzcntSupported}");
            sb.AppendLine($"  POPCNT:  {IsPopcntSupported}");
            sb.AppendLine($"  PCLMUL:  {IsPclmulqdqSupported}");
            sb.AppendLine("  --- ARM ---");
            sb.AppendLine($"  NEON:    {IsArmNeonSupported}");
            sb.AppendLine($"  SVE:     {IsSveSupported}");
            sb.AppendLine($"  SVE2:    {IsSve2Supported}");
            sb.AppendLine("======================================");
            return sb.ToString();
        }
    }

    /// <summary>
    /// Enumerates the detected SIMD platform tiers for dispatch logic.
    /// </summary>
    public enum SimdPlatform
    {
        /// <summary>No SIMD acceleration available; scalar fallback only.</summary>
        Scalar = 0,

        /// <summary>128-bit SIMD (SSE2 or ARM NEON).</summary>
        Simd128 = 1,

        /// <summary>256-bit SIMD (AVX2).</summary>
        Simd256 = 2,

        /// <summary>512-bit SIMD (AVX-512).</summary>
        Simd512 = 3,

        /// <summary>ARM SVE with configurable vector length.</summary>
        ArmSve = 4
    }

    /// <summary>
    /// Represents a SIMD feature set with capabilities and limitations.
    /// </summary>
    public readonly record struct SimdFeatureSet
    {
        /// <summary>The primary platform detected.</summary>
        public SimdPlatform Platform { get; init; }

        /// <summary>The native vector width in bytes.</summary>
        public int VectorBytes { get; init; }

        /// <summary>The number of 32-bit float elements per vector.</summary>
        public int FloatsPerVector { get; init; }

        /// <summary>Whether FMA operations are available.</summary>
        public bool HasFma { get; init; }

        /// <summary>Whether the platform supports unaligned loads efficiently.</summary>
        public bool HasFastUnaligned { get; init; }

        /// <summary>Whether the platform supports gather operations.</summary>
        public bool HasGather { get; init; }

        /// <summary>Whether the platform supports hardware population count.</summary>
        public bool HasPopcnt { get; init; }

        /// <summary>Default feature set for the current platform.</summary>
        public static SimdFeatureSet Current => new()
        {
            Platform = DetectedPlatform,
            VectorBytes = PreferredVectorSize,
            FloatsPerVector = VectorWidthFloat,
            HasFma = IsFmaSupported,
            HasFastUnaligned = IsAvx2Supported || IsArmNeonSupported,
            HasGather = IsAvx2Supported,
            HasPopcnt = IsPopcntSupported
        };
    }

    /// <summary>
    /// Detects the best SIMD platform available on the current runtime.
    /// </summary>
    /// <returns>The highest-tier SIMD platform detected.</returns>
    private static SimdPlatform DetectPlatform()
    {
        // Mid-range production baseline: honour CpuCapabilityProbe ceiling (AVX2 by default).
        var ceiling = GDNN.Platform.CpuCapabilityProbe.Probe().EffectiveCeiling;
        if (ceiling == GDNN.Platform.SimdCeiling.Avx512 && Avx512F.IsSupported)
            return SimdPlatform.Simd512;
        if (ceiling >= GDNN.Platform.SimdCeiling.Avx2 && Avx2.IsSupported)
            return SimdPlatform.Simd256;
        if (ceiling >= GDNN.Platform.SimdCeiling.Sse2OrNeon)
        {
            if (Sse2.IsSupported)
                return SimdPlatform.Simd128;
            if (Sve.IsSupported || Sve2.IsSupported)
                return SimdPlatform.ArmSve;
            if (AdvSimd.IsSupported)
                return SimdPlatform.Simd128;
        }
        if (ceiling == GDNN.Platform.SimdCeiling.Scalar)
            return SimdPlatform.Scalar;

        // Fallback when Auto path did not set a ceiling above (should not happen).
        if (Avx2.IsSupported)
            return SimdPlatform.Simd256;
        if (Sse2.IsSupported)
            return SimdPlatform.Simd128;
        if (Sve.IsSupported || Sve2.IsSupported)
            return SimdPlatform.ArmSve;
        if (AdvSimd.IsSupported)
            return SimdPlatform.Simd128;
        return SimdPlatform.Scalar;
    }

    /// <summary>
    /// Detects the preferred SIMD vector size in bytes.
    /// </summary>
    private static int DetectPreferredVectorSize()
    {
        return DetectPlatform() switch
        {
            SimdPlatform.Simd512 => 64,
            SimdPlatform.Simd256 => 32,
            SimdPlatform.Simd128 or SimdPlatform.ArmSve => 16,
            _ => sizeof(float)
        };
    }

    /// <summary>
    /// Detects the number of float elements per SIMD vector.
    /// </summary>
    private static int DetectVectorWidthFloat()
    {
        return PreferredVectorSize / sizeof(float);
    }

    /// <summary>
    /// Returns the feature set for the current platform.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static SimdFeatureSet GetFeatureSet() => SimdFeatureSet.Current;

    /// <summary>
    /// Executes the appropriate delegate for the detected platform tier.
    /// </summary>
    /// <typeparam name="T">Delegate type for the platform-specific implementations.</typeparam>
    /// <param name="scalar">Fallback scalar implementation.</param>
    /// <param name="sse2">SSE2/NEON 128-bit implementation, or null.</param>
    /// <param name="avx2">AVX2 256-bit implementation, or null.</param>
    /// <param name="avx512">AVX-512 512-bit implementation, or null.</param>
    /// <returns>The result of the selected implementation.</returns>
    public static T Dispatch<T>(
        T scalar,
        T? sse2 = default,
        T? avx2 = default,
        T? avx512 = default) where T : Delegate
    {
        if (avx512 is not null && Avx512F.IsSupported)
            return avx512;
        if (avx2 is not null && Avx2.IsSupported)
            return avx2;
        if (sse2 is not null && (Sse2.IsSupported || AdvSimd.IsSupported))
            return sse2;
        return scalar;
    }

    /// <summary>
    /// Executes the appropriate action for the detected platform tier.
    /// </summary>
    /// <param name="scalar">Fallback scalar action.</param>
    /// <param name="sse2">SSE2/NEON action, or null.</param>
    /// <param name="avx2">AVX2 action, or null.</param>
    /// <param name="avx512">AVX-512 action, or null.</param>
    public static void Dispatch(
        Action scalar,
        Action? sse2 = null,
        Action? avx2 = null,
        Action? avx512 = null)
    {
        if (avx512 is not null && Avx512F.IsSupported)
        { avx512(); return; }
        if (avx2 is not null && Avx2.IsSupported)
        { avx2(); return; }
        if (sse2 is not null && (Sse2.IsSupported || AdvSimd.IsSupported))
        { sse2(); return; }
        scalar();
    }

    /// <summary>
    /// Returns the alignment in bytes required for optimal SIMD performance on this platform.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetOptimalAlignment()
    {
        if (Avx512F.IsSupported)
            return 64;
        if (Avx2.IsSupported)
            return 32;
        if (Sse2.IsSupported || AdvSimd.IsSupported)
            return 16;
        return sizeof(float);
    }

    /// <summary>
    /// Rounds up a byte count to the next SIMD-aligned boundary.
    /// </summary>
    /// <param name="size">The size in bytes to align.</param>
    /// <returns>The aligned size, guaranteed to be a multiple of the optimal SIMD alignment.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int AlignUp(int size)
    {
        int alignment = GetOptimalAlignment();
        return (size + alignment - 1) & ~(alignment - 1);
    }

    /// <summary>
    /// Rounds up a byte count to the next alignment boundary.
    /// </summary>
    /// <param name="size">The size in bytes to align.</param>
    /// <param name="alignment">The alignment boundary in bytes (must be a power of 2).</param>
    /// <returns>The aligned size.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int AlignUp(int size, int alignment)
    {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    /// <summary>
    /// Computes the number of SIMD iterations needed to process a given element count,
    /// including scalar tail handling.
    /// </summary>
    /// <param name="elementCount">Total number of elements to process.</param>
    /// <param name="vectorLength">Number of elements per SIMD vector.</param>
    /// <param name="tailCount">Number of remaining elements that need scalar processing.</param>
    /// <returns>The number of full SIMD iterations.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetSimdIterationCount(int elementCount, int vectorLength, out int tailCount)
    {
        int simdCount = elementCount / vectorLength;
        tailCount = elementCount - (simdCount * vectorLength);
        return simdCount;
    }

    /// <summary>
    /// Determines whether a pointer is aligned to the specified boundary.
    /// </summary>
    /// <param name="pointer">The pointer to check.</param>
    /// <param name="alignment">The alignment boundary in bytes.</param>
    /// <returns>True if the pointer is aligned; otherwise false.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool IsAligned(void* pointer, int alignment)
    {
        return ((nuint)pointer % (nuint)alignment) == 0;
    }

    /// <summary>
    /// Determines whether a byte offset is aligned to the specified boundary.
    /// </summary>
    /// <param name="offset">The byte offset to check.</param>
    /// <param name="alignment">The alignment boundary in bytes.</param>
    /// <returns>True if the offset is aligned; otherwise false.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsAligned(nuint offset, int alignment)
    {
        return (offset % (nuint)alignment) == 0;
    }

    /// <summary>
    /// Gets the maximum number of elements that can be loaded without exceeding
    /// a span length, aligned to the SIMD vector width.
    /// </summary>
    /// <param name="totalLength">Total length of the data span.</param>
    /// <returns>The number of elements that can be safely processed with SIMD.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetSafeSimdCount(int totalLength)
    {
        int vectorSize = VectorWidthFloat;
        return totalLength - (totalLength % vectorSize);
    }

    /// <summary>
    /// Benchmarks a SIMD operation against a scalar reference and returns
    /// the speedup ratio.
    /// </summary>
    /// <param name="simdAction">The SIMD-optimized action to benchmark.</param>
    /// <param name="scalarAction">The scalar reference action to benchmark against.</param>
    /// <param name="iterationCount">Number of benchmark iterations.</param>
    /// <param name="warmupIterations">Number of warmup iterations before measurement.</param>
    /// <returns>A <see cref="BenchmarkResult"/> containing timing and speedup data.</returns>
    public static BenchmarkResult Benchmark(
        Action simdAction,
        Action scalarAction,
        int iterationCount = 1_000,
        int warmupIterations = 100)
    {
        if (simdAction is null)
            throw new ArgumentNullException(nameof(simdAction));
        if (scalarAction is null)
            throw new ArgumentNullException(nameof(scalarAction));
        if (iterationCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(iterationCount));
        if (warmupIterations < 0)
            throw new ArgumentOutOfRangeException(nameof(warmupIterations));

        for (int i = 0; i < warmupIterations; i++)
        {
            simdAction();
            scalarAction();
        }

        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterationCount; i++)
            simdAction();
        sw.Stop();
        long simdTicks = sw.ElapsedTicks;

        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();

        sw.Restart();
        for (int i = 0; i < iterationCount; i++)
            scalarAction();
        sw.Stop();
        long scalarTicks = sw.ElapsedTicks;

        double simdMs = (double)simdTicks / Stopwatch.Frequency * 1000.0;
        double scalarMs = (double)scalarTicks / Stopwatch.Frequency * 1000.0;
        double speedup = scalarMs > 0 ? simdMs > 0 ? scalarMs / simdMs : double.PositiveInfinity : 0;

        return new BenchmarkResult
        {
            SimdTimeMs = simdMs,
            ScalarTimeMs = scalarMs,
            SpeedupRatio = speedup,
            IterationCount = iterationCount,
            Platform = DetectedPlatform,
            VectorBytes = PreferredVectorSize
        };
    }

    /// <summary>
    /// Benchmarks a generic function that returns a value, comparing SIMD vs scalar.
    /// </summary>
    /// <typeparam name="T">The return type of the benchmarked functions.</typeparam>
    /// <param name="simdFunc">The SIMD function to benchmark.</param>
    /// <param name="scalarFunc">The scalar reference function.</param>
    /// <param name="validator">Optional validator to compare results for correctness.</param>
    /// <param name="iterationCount">Number of benchmark iterations.</param>
    /// <param name="warmupIterations">Number of warmup iterations.</param>
    /// <returns>A <see cref="BenchmarkResult"/> with timing data.</returns>
    public static BenchmarkResult Benchmark<T>(
        Func<T> simdFunc,
        Func<T> scalarFunc,
        Func<T, T, bool>? validator = null,
        int iterationCount = 1_000,
        int warmupIterations = 100)
    {
        for (int i = 0; i < warmupIterations; i++)
        {
            simdFunc();
            scalarFunc();
        }

        if (validator is not null)
        {
            T simdResult = simdFunc();
            T scalarResult = scalarFunc();
            if (!validator(simdResult, scalarResult))
                throw new InvalidOperationException(
                    "SIMD and scalar implementations produced different results.");
        }

        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterationCount; i++)
            simdFunc();
        sw.Stop();
        long simdTicks = sw.ElapsedTicks;

        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();

        sw.Restart();
        for (int i = 0; i < iterationCount; i++)
            scalarFunc();
        sw.Stop();
        long scalarTicks = sw.ElapsedTicks;

        double simdMs = (double)simdTicks / Stopwatch.Frequency * 1000.0;
        double scalarMs = (double)scalarTicks / Stopwatch.Frequency * 1000.0;
        double speedup = scalarMs > 0 ? simdMs > 0 ? scalarMs / simdMs : double.PositiveInfinity : 0;

        return new BenchmarkResult
        {
            SimdTimeMs = simdMs,
            ScalarTimeMs = scalarMs,
            SpeedupRatio = speedup,
            IterationCount = iterationCount,
            Platform = DetectedPlatform,
            VectorBytes = PreferredVectorSize
        };
    }

    /// <summary>
    /// Runs a full benchmark suite comparing SIMD vs scalar for multiple operations.
    /// </summary>
    /// <param name="benchmarks">Array of named benchmark pairs (name, simd, scalar).</param>
    /// <param name="iterationCount">Number of iterations per benchmark.</param>
    /// <returns>An array of named benchmark results.</returns>
    public static BenchmarkSuiteResult RunBenchmarkSuite(
        (string Name, Action Simd, Action Scalar)[] benchmarks,
        int iterationCount = 1_000)
    {
        var results = new BenchmarkResult[benchmarks.Length];
        for (int i = 0; i < benchmarks.Length; i++)
        {
            results[i] = Benchmark(
                benchmarks[i].Simd,
                benchmarks[i].Scalar,
                iterationCount);
            results[i] = results[i] with { OperationName = benchmarks[i].Name };
        }

        return new BenchmarkSuiteResult
        {
            Platform = DetectedPlatform,
            VectorBytes = PreferredVectorSize,
            Results = results
        };
    }

    /// <summary>
    /// Estimates the throughput (elements/second) of a SIMD operation.
    /// </summary>
    /// <param name="action">The operation to measure.</param>
    /// <param name="elementCount">Number of elements processed per iteration.</param>
    /// <param name="iterationCount">Number of iterations to run.</param>
    /// <returns>Estimated throughput in millions of elements per second.</returns>
    public static double EstimateThroughput(
        Action action,
        int elementCount,
        int iterationCount = 10_000)
    {
        for (int i = 0; i < 100; i++)
            action();

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterationCount; i++)
            action();
        sw.Stop();

        double totalElements = (double)elementCount * iterationCount;
        double elapsedSeconds = (double)sw.ElapsedTicks / Stopwatch.Frequency;
        return totalElements / elapsedSeconds / 1_000_000.0;
    }

    /// <summary>
    /// Validates that SIMD and scalar implementations produce equivalent results
    /// within a tolerance.
    /// </summary>
    /// <param name="simdValues">Results from the SIMD implementation.</param>
    /// <param name="scalarValues">Results from the scalar implementation.</param>
    /// <param name="tolerance">Maximum allowed per-element difference.</param>
    /// <returns>True if all elements match within tolerance.</returns>
    public static bool ValidateResults(ReadOnlySpan<float> simdValues, ReadOnlySpan<float> scalarValues, float tolerance = 1e-5f)
    {
        if (simdValues.Length != scalarValues.Length)
            return false;

        int vectorSize = Vector<float>.Count;
        int simdEnd = simdValues.Length - (simdValues.Length % vectorSize);

        for (int i = 0; i < simdEnd; i += vectorSize)
        {
            var simdVec = new Vector<float>(simdValues.Slice(i));
            var scalarVec = new Vector<float>(scalarValues.Slice(i));
            var diff = Vector.Abs(simdVec - scalarVec);
            for (int j = 0; j < vectorSize; j++)
            {
                if (diff[j] > tolerance)
                    return false;
            }
        }

        for (int i = simdEnd; i < simdValues.Length; i++)
        {
            if (MathF.Abs(simdValues[i] - scalarValues[i]) > tolerance)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Prints the full SIMD capability report to the console.
    /// </summary>
    public static void PrintCapabilityReport()
    {
        Console.Write(CapabilityReport);
    }

    /// <summary>
    /// Logs a performance comparison between SIMD and scalar for a given operation.
    /// </summary>
    /// <param name="operationName">Name of the operation being compared.</param>
    /// <param name="result">The benchmark result to log.</param>
    public static void LogPerformance(string operationName, BenchmarkResult result)
    {
        Console.WriteLine($"[SIMD Perf] {operationName}:");
        Console.WriteLine($"  Platform: {result.Platform} ({result.VectorBytes}-byte vectors)");
        Console.WriteLine($"  SIMD:     {result.SimdTimeMs:F4} ms");
        Console.WriteLine($"  Scalar:   {result.ScalarTimeMs:F4} ms");
        Console.WriteLine($"  Speedup:  {result.SpeedupRatio:F2}x");
        Console.WriteLine();
    }

    /// <summary>
    /// Returns the CPU cache line size in bytes for the current platform.
    /// Common values: 64 bytes for x86-64, 64 bytes for most ARM64.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetCacheLineSize()
    {
        return 64;
    }

    /// <summary>
    /// Returns the L1 data cache size in bytes for the current platform.
    /// Used to determine optimal tile sizes for compute-bound SIMD kernels.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetL1CacheSize()
    {
        return RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64 => 32 * 1024,
            Architecture.X86 => 32 * 1024,
            Architecture.Arm64 => 64 * 1024,
            _ => 32 * 1024
        };
    }

    /// <summary>
    /// Returns the L2 cache size in bytes for the current platform.
    /// Used for large batch processing decisions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetL2CacheSize()
    {
        return RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64 => 256 * 1024,
            Architecture.X86 => 256 * 1024,
            Architecture.Arm64 => 512 * 1024,
            _ => 256 * 1024
        };
    }

    /// <summary>
    /// Computes an optimal tile size for SIMD processing that fits within L1 cache.
    /// </summary>
    /// <param name="elementSize">Size of each element in bytes.</param>
    /// <param name="cacheUtilizationFraction">Fraction of L1 cache to target (0.0-1.0).</param>
    /// <returns>Optimal tile size in number of elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ComputeOptimalTileSize(int elementSize, float cacheUtilizationFraction = 0.75f)
    {
        int cacheBytes = (int)(GetL1CacheSize() * cacheUtilizationFraction);
        return cacheBytes / elementSize;
    }

    /// <summary>
    /// Determines whether a memory region should use streaming (non-temporal) stores
    /// based on its size relative to cache capacity.
    /// </summary>
    /// <param name="byteCount">Size of the memory region in bytes.</param>
    /// <returns>True if streaming stores are recommended.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool ShouldUseStreamingStore(int byteCount)
    {
        return byteCount > GetL2CacheSize();
    }

    /// <summary>
    /// Attempts to pin a span and execute an action with an unsafe pointer.
    /// Handles alignment checks and provides the raw pointer.
    /// </summary>
    /// <param name="span">The span to pin.</param>
    /// <param name="action">The action receiving the pinned pointer and length.</param>
    /// <typeparam name="T">Element type of the span.</typeparam>
    public static unsafe void WithPinnedSpan<T>(Span<T> span, delegate*<T*, int, void> action) where T : unmanaged
    {
        fixed (T* ptr = &MemoryMarshal.GetReference(span))
        {
            action(ptr, span.Length);
        }
    }

    /// <summary>
    /// Executes a function with a pinned span, returning a result.
    /// </summary>
    /// <typeparam name="T">Element type of the span.</typeparam>
    /// <typeparam name="TResult">Return type.</typeparam>
    /// <param name="span">The span to pin.</param>
    /// <param name="func">The function receiving the pinned pointer and length.</param>
    /// <returns>The result of the function.</returns>
    public static unsafe TResult WithPinnedSpan<T, TResult>(ReadOnlySpan<T> span, delegate*<T*, int, TResult> func) where T : unmanaged
    {
        fixed (T* ptr = &MemoryMarshal.GetReference(span))
        {
            return func(ptr, span.Length);
        }
    }

    /// <summary>
    /// Gets the current thread's processor core affinity hint.
    /// Useful for pinning SIMD-intensive workloads to specific cores.
    /// </summary>
    /// <returns>The current thread's processor affinity mask, or 0 if unavailable.</returns>
    public static long GetThreadAffinity()
    {
        try
        {
            return Thread.CurrentThread.ManagedThreadId;
        }
        catch
        {
            return 0;
        }
    }

    /// <summary>
    /// Checks if the given span length requires scalar tail processing
    /// after SIMD bulk operations.
    /// </summary>
    /// <param name="length">Total element count.</param>
    /// <param name="hasTail">Set to true if scalar tail processing is needed.</param>
    /// <returns>The number of elements that can be processed with SIMD.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int CheckAndComputeSimdLength(int length, out bool hasTail)
    {
        int vectorWidth = Vector<float>.Count;
        int simdLength = length - (length % vectorWidth);
        hasTail = simdLength < length;
        return simdLength;
    }

    /// <summary>
    /// Computes a recommended batch size for processing based on available SIMD width
    /// and target cache utilization.
    /// </summary>
    /// <param name="totalElements">Total number of elements to process.</param>
    /// <param name="elementSize">Size of each element in bytes.</param>
    /// <returns>Recommended batch size in number of elements.</returns>
    public static int ComputeBatchSize(int totalElements, int elementSize)
    {
        int optimalTile = ComputeOptimalTileSize(elementSize);
        int vectorWidth = Vector<float>.Count;

        int batchSize = optimalTile - (optimalTile % vectorWidth);
        if (batchSize > totalElements)
            batchSize = totalElements - (totalElements % vectorWidth);
        if (batchSize == 0)
            batchSize = Math.Min(vectorWidth, totalElements);

        return batchSize;
    }

    /// <summary>
    /// Gets the number of logical cores available on the system.
    /// </summary>
    public static int LogicalCoreCount => Environment.ProcessorCount;

    /// <summary>
    /// Determines if the workload is memory-bound or compute-bound
    /// based on the operation's arithmetic intensity.
    /// </summary>
    /// <param name="flopsPerElement">Floating-point operations per element.</param>
    /// <param name="bytesPerElement">Bytes loaded/stored per element.</param>
    /// <returns>True if the workload is likely compute-bound; false if memory-bound.</returns>
    public static bool IsComputeBound(int flopsPerElement, int bytesPerElement)
    {
        double arithmeticIntensity = (double)flopsPerElement / bytesPerElement;
        double rooflineRatio = arithmeticIntensity * VectorWidthFloat;
        return rooflineRatio > 16.0;
    }

    /// <summary>
    /// Records a benchmark result with timing and platform information.
    /// </summary>
    public readonly record struct BenchmarkResult
    {
        /// <summary>Name of the benchmarked operation.</summary>
        public string OperationName { get; init; }

        /// <summary>SIMD execution time in milliseconds.</summary>
        public double SimdTimeMs { get; init; }

        /// <summary>Scalar execution time in milliseconds.</summary>
        public double ScalarTimeMs { get; init; }

        /// <summary>Speedup ratio (scalar_time / simd_time). Values &gt; 1 indicate SIMD improvement.</summary>
        public double SpeedupRatio { get; init; }

        /// <summary>Number of iterations used for measurement.</summary>
        public int IterationCount { get; init; }

        /// <summary>The detected SIMD platform during the benchmark.</summary>
        public SimdPlatform Platform { get; init; }

        /// <summary>The SIMD vector width in bytes during the benchmark.</summary>
        public int VectorBytes { get; init; }

        /// <summary>Returns a human-readable summary of this result.</summary>
        public override string ToString() =>
            $"{OperationName}: SIMD {SimdTimeMs:F4}ms, Scalar {ScalarTimeMs:F4}ms, Speedup {SpeedupRatio:F2}x ({Platform} {VectorBytes}B)";
    }

    /// <summary>
    /// Contains results from a suite of benchmarks.
    /// </summary>
    public sealed class BenchmarkSuiteResult
    {
        /// <summary>The SIMD platform used for all benchmarks in this suite.</summary>
        public SimdPlatform Platform { get; init; }

        /// <summary>The vector width in bytes.</summary>
        public int VectorBytes { get; init; }

        /// <summary>Individual benchmark results.</summary>
        public BenchmarkResult[] Results { get; init; } = [];

        /// <summary>Returns the average speedup across all benchmarks.</summary>
        public double AverageSpeedup
        {
            get
            {
                if (Results.Length == 0)
                    return 0;
                double sum = 0;
                for (int i = 0; i < Results.Length; i++)
                    sum += Results[i].SpeedupRatio;
                return sum / Results.Length;
            }
        }

        /// <summary>Returns the minimum speedup across all benchmarks.</summary>
        public double MinSpeedup
        {
            get
            {
                if (Results.Length == 0)
                    return 0;
                double min = double.MaxValue;
                for (int i = 0; i < Results.Length; i++)
                    if (Results[i].SpeedupRatio < min)
                        min = Results[i].SpeedupRatio;
                return min;
            }
        }

        /// <summary>Returns the maximum speedup across all benchmarks.</summary>
        public double MaxSpeedup
        {
            get
            {
                if (Results.Length == 0)
                    return 0;
                double max = 0;
                for (int i = 0; i < Results.Length; i++)
                    if (Results[i].SpeedupRatio > max)
                        max = Results[i].SpeedupRatio;
                return max;
            }
        }

        /// <summary>Prints a formatted summary of all benchmark results.</summary>
        public void PrintSummary()
        {
            Console.WriteLine("=== Benchmark Suite Summary ===");
            Console.WriteLine($"Platform: {Platform} ({VectorBytes}-byte vectors)");
            Console.WriteLine($"Benchmarks: {Results.Length}");
            Console.WriteLine($"Average Speedup: {AverageSpeedup:F2}x");
            Console.WriteLine($"Min Speedup:     {MinSpeedup:F2}x");
            Console.WriteLine($"Max Speedup:     {MaxSpeedup:F2}x");
            Console.WriteLine();
            for (int i = 0; i < Results.Length; i++)
                Console.WriteLine($"  {Results[i]}");
            Console.WriteLine("===============================");
        }
    }
}
