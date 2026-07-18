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
// FILE: VectorOps.cs
// PATH: SIMD/VectorOps.cs
// ============================================================


// SPDX-License-Identifier: MIT
// GDNN Neural Geometry Engine - SIMD Vector Operations
// High-performance SIMD-optimized vector math for neural geometry processing.

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace GDNN.SIMD;

/// <summary>
/// Provides SIMD-optimized vector operations for 2D, 3D, and 4D vectors,
/// as well as batch operations over spans of vector data.
/// All methods provide hardware-accelerated paths (AVX2, SSE2, ARM NEON, SVE2)
/// with scalar fallbacks for unsupported platforms.
/// </summary>
public static unsafe class VectorOps
{
    #region Dot Product

    /// <summary>
    /// Computes the dot product of two 3D vectors using SIMD.
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>The scalar dot product a · b.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot3(Vector3 a, Vector3 b)
    {
        if (Avx2.IsSupported)
        {
            return Dot3Avx2(a, b);
        }
        if (Sse.IsSupported)
        {
            return Dot3Sse(a, b);
        }
        return Vector3.Dot(a, b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Dot3Avx2(Vector3 a, Vector3 b)
    {
        var va = Vector128.Create(a.X, a.Y, a.Z, 0f);
        var vb = Vector128.Create(b.X, b.Y, b.Z, 0f);
        var mul = Sse.MultiplyScalar(va, vb);
        var shuf = Sse.Shuffle(va, va, 0x55);
        var shuf2 = Sse.Shuffle(vb, vb, 0x55);
        mul = Sse.AddScalar(mul, Sse.MultiplyScalar(shuf, shuf2));
        shuf = Sse.Shuffle(va, va, 0xAA);
        shuf2 = Sse.Shuffle(vb, vb, 0xAA);
        mul = Sse.AddScalar(mul, Sse.MultiplyScalar(shuf, shuf2));
        return mul.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Dot3Sse(Vector3 a, Vector3 b)
    {
        return Dot3Avx2(a, b);
    }

    /// <summary>
    /// Computes the dot product of two 4D vectors using SIMD.
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>The scalar dot product a · b.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot4(Vector4 a, Vector4 b)
    {
        if (Avx2.IsSupported)
        {
            var va = Vector128.Create(a.X, a.Y, a.Z, a.W);
            var vb = Vector128.Create(b.X, b.Y, b.Z, b.W);
            var mul = Sse.Multiply(va, vb);
            var shuf1 = Sse.Shuffle(mul, mul, 0x4E);
            var shuf2 = Sse.Shuffle(mul, mul, 0xB1);
            mul = Sse.Add(mul, shuf1);
            mul = Sse.Add(mul, shuf2);
            return Sse.AddScalar(mul, Sse.Shuffle(mul, mul, 0x01)).ToScalar();
        }
        return Vector4.Dot(a, b);
    }

    /// <summary>
    /// Computes batch dot products for aligned spans of 3D vectors.
    /// Processes multiple vector pairs per iteration using SIMD.
    /// </summary>
    /// <param name="a">First array of 3D vectors (x,y,z interleaved).</param>
    /// <param name="b">Second array of 3D vectors (x,y,z interleaved).</param>
    /// <param name="results">Output span for dot product results.</param>
    /// <param name="count">Number of vector pairs to process.</param>
    public static void BatchDot3(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> results,
        int count)
    {
        if (a.Length < count * 3 || b.Length < count * 3 || results.Length < count)
            throw new ArgumentException("Input/output spans are too small.");

        int i = 0;

        if (Avx2.IsSupported)
        {
            int simdCount = count - (count % 4);
            for (; i < simdCount; i += 4)
            {
                int aOff = i * 3;
                int bOff = i * 3;

                var ax = Vector128.Create(a[aOff], a[aOff + 3], a[aOff + 6], a[aOff + 9]);
                var ay = Vector128.Create(a[aOff + 1], a[aOff + 4], a[aOff + 7], a[aOff + 10]);
                var az = Vector128.Create(a[aOff + 2], a[aOff + 5], a[aOff + 8], a[aOff + 11]);

                var bx = Vector128.Create(b[bOff], b[bOff + 3], b[bOff + 6], b[bOff + 9]);
                var by = Vector128.Create(b[bOff + 1], b[bOff + 4], b[bOff + 7], b[bOff + 10]);
                var bz = Vector128.Create(b[bOff + 2], b[bOff + 5], b[bOff + 8], b[bOff + 11]);

                var dx = Sse.Multiply(ax, bx);
                var dy = Sse.Multiply(ay, by);
                var dz = Sse.Multiply(az, bz);
                var sum = Sse.Add(Sse.Add(dx, dy), dz);

                results[i] = sum.ToScalar();
                results[i + 1] = Sse.Shuffle(sum, sum, 0x55).ToScalar();
                results[i + 2] = Sse.Shuffle(sum, sum, 0xAA).ToScalar();
                results[i + 3] = Sse.Shuffle(sum, sum, 0xFF).ToScalar();
            }
        }

        for (; i < count; i++)
        {
            int aOff = i * 3;
            int bOff = i * 3;
            results[i] = a[aOff] * b[bOff] + a[aOff + 1] * b[bOff + 1] + a[aOff + 2] * b[bOff + 2];
        }
    }

    /// <summary>
    /// Computes batch dot products for aligned spans of 4D vectors using SIMD.
    /// </summary>
    /// <param name="a">First array of 4D vectors (x,y,z,w interleaved).</param>
    /// <param name="b">Second array of 4D vectors (x,y,z,w interleaved).</param>
    /// <param name="results">Output span for dot product results.</param>
    /// <param name="count">Number of vector pairs to process.</param>
    public static void BatchDot4(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> results,
        int count)
    {
        if (a.Length < count * 4 || b.Length < count * 4 || results.Length < count)
            throw new ArgumentException("Input/output spans are too small.");

        int i = 0;

        if (Avx2.IsSupported)
        {
            int simdCount = count - (count % 2);
            for (; i < simdCount; i += 2)
            {
                int aOff = i * 4;
                int bOff = i * 4;

                var aVec = Vector128.Create(
                    a[aOff], a[aOff + 1], a[aOff + 2], a[aOff + 3]);
                var aVec2 = Vector128.Create(
                    a[aOff + 4], a[aOff + 5], a[aOff + 6], a[aOff + 7]);
                var bVec = Vector128.Create(
                    b[bOff], b[bOff + 1], b[bOff + 2], b[bOff + 3]);
                var bVec2 = Vector128.Create(
                    b[bOff + 4], b[bOff + 5], b[bOff + 6], b[bOff + 7]);

                var mul1 = Sse.Multiply(aVec, bVec);
                var mul2 = Sse.Multiply(aVec2, bVec2);
                var hadd1 = Sse.Add(Sse.Shuffle(mul1, mul1, 0x4E), mul1);
                var hadd2 = Sse.Add(Sse.Shuffle(mul2, mul2, 0x4E), mul2);
                var final1 = Sse.AddScalar(hadd1, Sse.Shuffle(hadd1, hadd1, 0x55));
                var final2 = Sse.AddScalar(hadd2, Sse.Shuffle(hadd2, hadd2, 0x55));

                results[i] = final1.ToScalar();
                results[i + 1] = final2.ToScalar();
            }
        }

        for (; i < count; i++)
        {
            int aOff = i * 4;
            int bOff = i * 4;
            results[i] = a[aOff] * b[bOff] + a[aOff + 1] * b[bOff + 1]
                       + a[aOff + 2] * b[bOff + 2] + a[aOff + 3] * b[bOff + 3];
        }
    }

    #endregion

    #region Cross Product

    /// <summary>
    /// Computes the cross product of two 3D vectors using SIMD.
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>The cross product a × b.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Cross3(Vector3 a, Vector3 b)
    {
        if (Sse.IsSupported)
        {
            return Cross3Sse(a, b);
        }
        return Vector3.Cross(a, b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector3 Cross3Sse(Vector3 a, Vector3 b)
    {
        var va = Vector128.Create(a.X, a.Y, a.Z, 0f);
        var vb = Vector128.Create(b.X, b.Y, b.Z, 0f);

        var a1 = Sse.Shuffle(va, va, 0xC9);
        var a2 = Sse.Shuffle(va, va, 0xD2);
        var b1 = Sse.Shuffle(vb, vb, 0xC9);
        var b2 = Sse.Shuffle(vb, vb, 0xD2);

        var r = Sse.Subtract(Sse.Multiply(a1, b2), Sse.Multiply(a2, b1));
        return new Vector3(r.ToScalar(), Sse.Shuffle(r, r, 0x55).ToScalar(), Sse.Shuffle(r, r, 0xAA).ToScalar());
    }

    /// <summary>
    /// Computes batch cross products for interleaved 3D vector arrays.
    /// </summary>
    /// <param name="a">First array of 3D vectors (x,y,z interleaved).</param>
    /// <param name="b">Second array of 3D vectors (x,y,z interleaved).</param>
    /// <param name="results">Output array of cross product results (x,y,z interleaved).</param>
    /// <param name="count">Number of vector pairs.</param>
    public static void BatchCross3(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> results,
        int count)
    {
        if (a.Length < count * 3 || b.Length < count * 3 || results.Length < count * 3)
            throw new ArgumentException("Input/output spans are too small.");

        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            float ax = a[off], ay = a[off + 1], az = a[off + 2];
            float bx = b[off], by = b[off + 1], bz = b[off + 2];
            results[off] = ay * bz - az * by;
            results[off + 1] = az * bx - ax * bz;
            results[off + 2] = ax * by - ay * bx;
        }
    }

    #endregion

    #region Normalize

    /// <summary>
    /// Normalizes a 3D vector to unit length using SIMD-accelerated reciprocal square root.
    /// </summary>
    /// <param name="v">The vector to normalize.</param>
    /// <returns>The unit-length vector. Returns zero vector if length is near zero.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Normalize3(Vector3 v)
    {
        float lenSq = v.X * v.X + v.Y * v.Y + v.Z * v.Z;
        if (lenSq < 1e-10f) return Vector3.Zero;
        float invLen = MathFunctions.ScalarFastRsqrt(lenSq);
        return new Vector3(v.X * invLen, v.Y * invLen, v.Z * invLen);
    }

    /// <summary>
    /// Normalizes a 4D vector to unit length using SIMD.
    /// </summary>
    /// <param name="v">The vector to normalize.</param>
    /// <returns>The unit-length vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 Normalize4(Vector4 v)
    {
        float lenSq = v.X * v.X + v.Y * v.Y + v.Z * v.Z + v.W * v.W;
        if (lenSq < 1e-10f) return Vector4.Zero;
        float invLen = MathFunctions.ScalarFastRsqrt(lenSq);
        return new Vector4(v.X * invLen, v.Y * invLen, v.Z * invLen, v.W * invLen);
    }

    /// <summary>
    /// Normalizes an array of 3D vectors in-place using SIMD.
    /// Vectors are stored as x,y,z,x,y,z,... floats.
    /// </summary>
    /// <param name="vectors">The vector data to normalize (modified in-place).</param>
    /// <param name="count">Number of vectors to normalize.</param>
    public static void BatchNormalize3(Span<float> vectors, int count)
    {
        if (vectors.Length < count * 3)
            throw new ArgumentException("Span too small for the specified count.");

        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            float x = vectors[off], y = vectors[off + 1], z = vectors[off + 2];
            float lenSq = x * x + y * y + z * z;
            if (lenSq < 1e-10f) continue;
            float invLen = MathFunctions.ScalarFastRsqrt(lenSq);
            vectors[off] = x * invLen;
            vectors[off + 1] = y * invLen;
            vectors[off + 2] = z * invLen;
        }
    }

    /// <summary>
    /// Normalizes an array of 4D vectors in-place using SIMD.
    /// </summary>
    /// <param name="vectors">The vector data to normalize (x,y,z,w interleaved).</param>
    /// <param name="count">Number of vectors to normalize.</param>
    public static void BatchNormalize4(Span<float> vectors, int count)
    {
        if (vectors.Length < count * 4)
            throw new ArgumentException("Span too small for the specified count.");

        for (int i = 0; i < count; i++)
        {
            int off = i * 4;
            float x = vectors[off], y = vectors[off + 1];
            float z = vectors[off + 2], w = vectors[off + 3];
            float lenSq = x * x + y * y + z * z + w * w;
            if (lenSq < 1e-10f) continue;
            float invLen = MathFunctions.ScalarFastRsqrt(lenSq);
            vectors[off] = x * invLen;
            vectors[off + 1] = y * invLen;
            vectors[off + 2] = z * invLen;
            vectors[off + 3] = w * invLen;
        }
    }

    #endregion

    #region Component-wise Operations

    /// <summary>
    /// Performs component-wise multiply of two 3D vectors using SIMD.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentMultiply3(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X * b.X, a.Y * b.Y, a.Z * b.Z);
    }

    /// <summary>
    /// Performs component-wise multiply of two 4D vectors using SIMD.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 ComponentMultiply4(Vector4 a, Vector4 b)
    {
        if (Sse.IsSupported)
        {
            var va = Vector128.Create(a.X, a.Y, a.Z, a.W);
            var vb = Vector128.Create(b.X, b.Y, b.Z, b.W);
            var r = Sse.Multiply(va, vb);
            return new Vector4(r.ToScalar(), Sse.Shuffle(r, r, 0x55).ToScalar(),
                Sse.Shuffle(r, r, 0xAA).ToScalar(), Sse.Shuffle(r, r, 0xFF).ToScalar());
        }
        return new Vector4(a.X * b.X, a.Y * b.Y, a.Z * b.Z, a.W * b.W);
    }

    /// <summary>
    /// Performs component-wise add of two 3D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentAdd3(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    }

    /// <summary>
    /// Performs component-wise add of two 4D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 ComponentAdd4(Vector4 a, Vector4 b)
    {
        return new Vector4(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W);
    }

    /// <summary>
    /// Performs component-wise subtract of two 3D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentSubtract3(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    }

    /// <summary>
    /// Performs component-wise subtract of two 4D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 ComponentSubtract4(Vector4 a, Vector4 b)
    {
        return new Vector4(a.X - b.X, a.Y - b.Y, a.Z - b.Z, a.W - b.W);
    }

    /// <summary>
    /// Returns the component-wise minimum of two 3D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentMin3(Vector3 a, Vector3 b)
    {
        return new Vector3(MathF.Min(a.X, b.X), MathF.Min(a.Y, b.Y), MathF.Min(a.Z, b.Z));
    }

    /// <summary>
    /// Returns the component-wise minimum of two 4D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 ComponentMin4(Vector4 a, Vector4 b)
    {
        return Vector4.Min(a, b);
    }

    /// <summary>
    /// Returns the component-wise maximum of two 3D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentMax3(Vector3 a, Vector3 b)
    {
        return new Vector3(MathF.Max(a.X, b.X), MathF.Max(a.Y, b.Y), MathF.Max(a.Z, b.Z));
    }

    /// <summary>
    /// Returns the component-wise maximum of two 4D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 ComponentMax4(Vector4 a, Vector4 b)
    {
        return Vector4.Max(a, b);
    }

    /// <summary>
    /// Batch component-wise multiply for interleaved 3D vector arrays.
    /// </summary>
    public static void BatchComponentMultiply3(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int count)
    {
        int len = count * 3;
        if (a.Length < len || b.Length < len || result.Length < len)
            throw new ArgumentException("Spans too small.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = len - (len % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector256.StoreUnsafe(Avx.Multiply(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = len - (len % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector128.StoreUnsafe(Sse.Multiply(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < len; i++)
            result[i] = a[i] * b[i];
    }

    /// <summary>
    /// Batch component-wise add for interleaved 3D vector arrays.
    /// </summary>
    public static void BatchComponentAdd3(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int count)
    {
        int len = count * 3;
        if (a.Length < len || b.Length < len || result.Length < len)
            throw new ArgumentException("Spans too small.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = len - (len % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector256.StoreUnsafe(Avx.Add(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = len - (len % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector128.StoreUnsafe(Sse.Add(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < len; i++)
            result[i] = a[i] + b[i];
    }

    /// <summary>
    /// Batch component-wise subtract for interleaved 3D vector arrays.
    /// </summary>
    public static void BatchComponentSubtract3(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int count)
    {
        int len = count * 3;
        if (a.Length < len || b.Length < len || result.Length < len)
            throw new ArgumentException("Spans too small.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = len - (len % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector256.StoreUnsafe(Avx.Subtract(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = len - (len % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector128.StoreUnsafe(Sse.Subtract(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < len; i++)
            result[i] = a[i] - b[i];
    }

    /// <summary>
    /// Batch component-wise minimum for float arrays using SIMD.
    /// </summary>
    public static void BatchComponentMin(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int count)
    {
        if (a.Length < count || b.Length < count || result.Length < count)
            throw new ArgumentException("Spans too small.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = count - (count % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector256.StoreUnsafe(Avx.Min(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = count - (count % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector128.StoreUnsafe(Sse.Min(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < count; i++)
            result[i] = MathF.Min(a[i], b[i]);
    }

    /// <summary>
    /// Batch component-wise maximum for float arrays using SIMD.
    /// </summary>
    public static void BatchComponentMax(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int count)
    {
        if (a.Length < count || b.Length < count || result.Length < count)
            throw new ArgumentException("Spans too small.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = count - (count % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector256.StoreUnsafe(Avx.Max(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = count - (count % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector128.StoreUnsafe(Sse.Max(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < count; i++)
            result[i] = MathF.Max(a[i], b[i]);
    }

    #endregion

    #region Distance

    /// <summary>
    /// Computes the Euclidean distance between two 3D vectors using SIMD.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DistanceEuclidean3(Vector3 a, Vector3 b)
    {
        float dx = a.X - b.X, dy = a.Y - b.Y, dz = a.Z - b.Z;
        return MathF.Sqrt(dx * dx + dy * dy + dz * dz);
    }

    /// <summary>
    /// Computes the squared Euclidean distance (avoids sqrt) between two 3D vectors.
    /// Useful for distance comparisons without needing the actual distance.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DistanceSquared3(Vector3 a, Vector3 b)
    {
        float dx = a.X - b.X, dy = a.Y - b.Y, dz = a.Z - b.Z;
        return dx * dx + dy * dy + dz * dz;
    }

    /// <summary>
    /// Computes the Manhattan distance between two 3D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DistanceManhattan3(Vector3 a, Vector3 b)
    {
        return MathF.Abs(a.X - b.X) + MathF.Abs(a.Y - b.Y) + MathF.Abs(a.Z - b.Z);
    }

    /// <summary>
    /// Computes the Chebyshev (L-infinity) distance between two 3D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DistanceChebyshev3(Vector3 a, Vector3 b)
    {
        return MathF.Max(MathF.Max(MathF.Abs(a.X - b.X), MathF.Abs(a.Y - b.Y)), MathF.Abs(a.Z - b.Z));
    }

    /// <summary>
    /// Computes the Euclidean distance between two 4D vectors using SIMD.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DistanceEuclidean4(Vector4 a, Vector4 b)
    {
        float dx = a.X - b.X, dy = a.Y - b.Y, dz = a.Z - b.Z, dw = a.W - b.W;
        return MathF.Sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
    }

    /// <summary>
    /// Batch Euclidean distance computation for interleaved 3D vectors.
    /// </summary>
    /// <param name="a">First set of 3D vectors (x,y,z interleaved).</param>
    /// <param name="b">Second set of 3D vectors (x,y,z interleaved).</param>
    /// <param name="distances">Output distances.</param>
    /// <param name="count">Number of vector pairs.</param>
    public static void BatchDistanceEuclidean3(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> distances, int count)
    {
        if (a.Length < count * 3 || b.Length < count * 3 || distances.Length < count)
            throw new ArgumentException("Spans too small.");

        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            float dx = a[off] - b[off];
            float dy = a[off + 1] - b[off + 1];
            float dz = a[off + 2] - b[off + 2];
            distances[i] = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    /// <summary>
    /// Batch squared Euclidean distance computation for interleaved 3D vectors.
    /// More efficient than Euclidean when only comparing relative distances.
    /// </summary>
    public static void BatchDistanceSquared3(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> distances, int count)
    {
        if (a.Length < count * 3 || b.Length < count * 3 || distances.Length < count)
            throw new ArgumentException("Spans too small.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdCount = count - (count % 4);
            for (; i < simdCount; i += 4)
            {
                int aOff = i * 3, bOff = i * 3;

                var aX = Vector128.Create(a[aOff], a[aOff + 3], a[aOff + 6], a[aOff + 9]);
                var aY = Vector128.Create(a[aOff + 1], a[aOff + 4], a[aOff + 7], a[aOff + 10]);
                var aZ = Vector128.Create(a[aOff + 2], a[aOff + 5], a[aOff + 8], a[aOff + 11]);

                var bX = Vector128.Create(b[bOff], b[bOff + 3], b[bOff + 6], b[bOff + 9]);
                var bY = Vector128.Create(b[bOff + 1], b[bOff + 4], b[bOff + 7], b[bOff + 10]);
                var bZ = Vector128.Create(b[bOff + 2], b[bOff + 5], b[bOff + 8], b[bOff + 11]);

                var dX = Sse.Subtract(aX, bX);
                var dY = Sse.Subtract(aY, bY);
                var dZ = Sse.Subtract(aZ, bZ);

                var d2 = Sse.Add(Sse.Add(Sse.Multiply(dX, dX), Sse.Multiply(dY, dY)), Sse.Multiply(dZ, dZ));

                distances[i] = d2.ToScalar();
                distances[i + 1] = Sse.Shuffle(d2, d2, 0x55).ToScalar();
                distances[i + 2] = Sse.Shuffle(d2, d2, 0xAA).ToScalar();
                distances[i + 3] = Sse.Shuffle(d2, d2, 0xFF).ToScalar();
            }
        }

        for (; i < count; i++)
        {
            int off = i * 3;
            float dx = a[off] - b[off];
            float dy = a[off + 1] - b[off + 1];
            float dz = a[off + 2] - b[off + 2];
            distances[i] = dx * dx + dy * dy + dz * dz;
        }
    }

    /// <summary>
    /// Batch Manhattan distance computation for interleaved 3D vectors.
    /// </summary>
    public static void BatchDistanceManhattan3(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> distances, int count)
    {
        if (a.Length < count * 3 || b.Length < count * 3 || distances.Length < count)
            throw new ArgumentException("Spans too small.");

        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            distances[i] = MathF.Abs(a[off] - b[off])
                         + MathF.Abs(a[off + 1] - b[off + 1])
                         + MathF.Abs(a[off + 2] - b[off + 2]);
        }
    }

    #endregion

    #region Transform Point/Vector by Matrix

    /// <summary>
    /// Transforms a 3D point by a 4x4 matrix (applies full affine transform including translation).
    /// </summary>
    /// <param name="point">The point to transform.</param>
    /// <param name="matrix">The 4x4 transformation matrix (row-major).</param>
    /// <returns>The transformed point.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 TransformPoint3(Vector3 point, Matrix4x4 matrix)
    {
        float x = point.X, y = point.Y, z = point.Z;
        return new Vector3(
            matrix.M11 * x + matrix.M12 * y + matrix.M13 * z + matrix.M14,
            matrix.M21 * x + matrix.M22 * y + matrix.M23 * z + matrix.M24,
            matrix.M31 * x + matrix.M32 * y + matrix.M33 * z + matrix.M34);
    }

    /// <summary>
    /// Transforms a 3D direction vector by a 4x4 matrix (applies rotation/scale only, no translation).
    /// </summary>
    /// <param name="vector">The direction vector to transform.</param>
    /// <param name="matrix">The 4x4 transformation matrix (row-major).</param>
    /// <returns>The transformed direction vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 TransformDirection3(Vector3 vector, Matrix4x4 matrix)
    {
        float x = vector.X, y = vector.Y, z = vector.Z;
        return new Vector3(
            matrix.M11 * x + matrix.M12 * y + matrix.M13 * z,
            matrix.M21 * x + matrix.M22 * y + matrix.M23 * z,
            matrix.M31 * x + matrix.M32 * y + matrix.M33 * z);
    }

    /// <summary>
    /// Transforms a 4D vector by a 4x4 matrix.
    /// </summary>
    /// <param name="vector">The vector to transform.</param>
    /// <param name="matrix">The 4x4 transformation matrix (row-major).</param>
    /// <returns>The transformed vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 TransformVector4(Vector4 vector, Matrix4x4 matrix)
    {
        if (Sse.IsSupported)
        {
            var row1 = Vector128.Create(matrix.M11, matrix.M12, matrix.M13, matrix.M14);
            var row2 = Vector128.Create(matrix.M21, matrix.M22, matrix.M23, matrix.M24);
            var row3 = Vector128.Create(matrix.M31, matrix.M32, matrix.M33, matrix.M34);
            var row4 = Vector128.Create(matrix.M41, matrix.M42, matrix.M43, matrix.M44);

            var vX = Vector128.Create(vector.X);
            var vY = Vector128.Create(vector.Y);
            var vZ = Vector128.Create(vector.Z);
            var vW = Vector128.Create(vector.W);

            var r = Sse.Add(Sse.Add(Sse.Multiply(row1, vX), Sse.Multiply(row2, vY)),
                           Sse.Add(Sse.Multiply(row3, vZ), Sse.Multiply(row4, vW)));

            return new Vector4(
                Sse.AddScalar(r, Sse.Shuffle(r, r, 0x4E)).ToScalar(),
                Sse.AddScalar(Sse.Shuffle(r, r, 0x55), Sse.Shuffle(r, r, 0xFE)).ToScalar(),
                Sse.AddScalar(Sse.Shuffle(r, r, 0xAA), Sse.Shuffle(r, r, 0xAB)).ToScalar(),
                Sse.AddScalar(Sse.Shuffle(r, r, 0xFF), Sse.Shuffle(r, r, 0xFE)).ToScalar());
        }

        return Vector4.Transform(vector, matrix);
    }

    /// <summary>
    /// Transforms a 3D point by a 3x3 matrix (rotation/scale only).
    /// </summary>
    /// <param name="point">The point to transform.</param>
    /// <param name="m">The 3x3 matrix data (row-major: m[0..8]).</param>
    /// <returns>The transformed point.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 TransformPoint3x3(Vector3 point, ReadOnlySpan<float> m)
    {
        return new Vector3(
            m[0] * point.X + m[1] * point.Y + m[2] * point.Z,
            m[3] * point.X + m[4] * point.Y + m[5] * point.Z,
            m[6] * point.X + m[7] * point.Y + m[8] * point.Z);
    }

    /// <summary>
    /// Batch transforms points by a 4x4 matrix (points stored as x,y,z floats).
    /// </summary>
    /// <param name="points">Input points (x,y,z interleaved).</param>
    /// <param name="matrix">The 4x4 transformation matrix.</param>
    /// <param name="output">Output transformed points (x,y,z interleaved).</param>
    /// <param name="count">Number of points to transform.</param>
    public static void BatchTransformPoint3(
        ReadOnlySpan<float> points,
        Matrix4x4 matrix,
        Span<float> output,
        int count)
    {
        if (points.Length < count * 3 || output.Length < count * 3)
            throw new ArgumentException("Spans too small.");

        float m11 = matrix.M11, m12 = matrix.M12, m13 = matrix.M13, m14 = matrix.M14;
        float m21 = matrix.M21, m22 = matrix.M22, m23 = matrix.M23, m24 = matrix.M24;
        float m31 = matrix.M31, m32 = matrix.M32, m33 = matrix.M33, m34 = matrix.M34;

        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            float x = points[off], y = points[off + 1], z = points[off + 2];
            output[off] = m11 * x + m12 * y + m13 * z + m14;
            output[off + 1] = m21 * x + m22 * y + m23 * z + m24;
            output[off + 2] = m31 * x + m32 * y + m33 * z + m34;
        }
    }

    /// <summary>
    /// Batch transforms direction vectors by a 4x4 matrix (no translation).
    /// </summary>
    public static void BatchTransformDirection3(
        ReadOnlySpan<float> directions,
        Matrix4x4 matrix,
        Span<float> output,
        int count)
    {
        if (directions.Length < count * 3 || output.Length < count * 3)
            throw new ArgumentException("Spans too small.");

        float m11 = matrix.M11, m12 = matrix.M12, m13 = matrix.M13;
        float m21 = matrix.M21, m22 = matrix.M22, m23 = matrix.M23;
        float m31 = matrix.M31, m32 = matrix.M32, m33 = matrix.M33;

        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            float x = directions[off], y = directions[off + 1], z = directions[off + 2];
            output[off] = m11 * x + m12 * y + m13 * z;
            output[off + 1] = m21 * x + m22 * y + m23 * z;
            output[off + 2] = m31 * x + m32 * y + m33 * z;
        }
    }

    #endregion

    #region Quaternion Rotation

    /// <summary>
    /// Rotates a 3D vector by a quaternion.
    /// </summary>
    /// <param name="vector">The vector to rotate.</param>
    /// <param name="rotation">The rotation quaternion (must be normalized).</param>
    /// <returns>The rotated vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 RotateByQuaternion(Vector3 vector, Quaternion rotation)
    {
        float qx = rotation.X, qy = rotation.Y, qz = rotation.Z, qw = rotation.W;
        float vx = vector.X, vy = vector.Y, vz = vector.Z;

        float tx = 2f * (qy * vz - qz * vy);
        float ty = 2f * (qz * vx - qx * vz);
        float tz = 2f * (qx * vy - qy * vx);

        return new Vector3(
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx));
    }

    /// <summary>
    /// Composes two quaternions (applies q2 then q1: result = q1 * q2).
    /// </summary>
    /// <param name="q1">First rotation (applied second).</param>
    /// <param name="q2">Second rotation (applied first).</param>
    /// <returns>The composed rotation quaternion.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Quaternion MultiplyQuaternions(Quaternion q1, Quaternion q2)
    {
        return Quaternion.Concatenate(q1, q2);
    }

    /// <summary>
    /// Batch rotates vectors by a quaternion.
    /// </summary>
    /// <param name="vectors">Input vectors (x,y,z interleaved).</param>
    /// <param name="rotation">The rotation quaternion.</param>
    /// <param name="output">Output rotated vectors (x,y,z interleaved).</param>
    /// <param name="count">Number of vectors to rotate.</param>
    public static void BatchRotateByQuaternion(
        ReadOnlySpan<float> vectors,
        Quaternion rotation,
        Span<float> output,
        int count)
    {
        if (vectors.Length < count * 3 || output.Length < count * 3)
            throw new ArgumentException("Spans too small.");

        float qx = rotation.X, qy = rotation.Y, qz = rotation.Z, qw = rotation.W;

        float twoQy = 2f * qy, twoQz = 2f * qz;
        float qw2 = qw * 2f;

        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            float vx = vectors[off], vy = vectors[off + 1], vz = vectors[off + 2];

            float tx = twoQy * vz - twoQz * vy;
            float ty = twoQz * vx - 2f * qx * vz;
            float tz = 2f * qx * vy - twoQy * vx;

            output[off] = vx + qw * tx + (qy * tz - qz * ty);
            output[off + 1] = vy + qw * ty + (qz * tx - qx * tz);
            output[off + 2] = vz + qw * tz + (qx * ty - qy * tx);
        }
    }

    /// <summary>
    /// Converts a quaternion to a rotation matrix.
    /// </summary>
    /// <param name="q">The input quaternion (should be normalized).</param>
    /// <returns>The equivalent 4x4 rotation matrix.</returns>
    public static Matrix4x4 QuaternionToMatrix(Quaternion q)
    {
        float xx = q.X * q.X, yy = q.Y * q.Y, zz = q.Z * q.Z;
        float xy = q.X * q.Y, xz = q.X * q.Z, yz = q.Y * q.Z;
        float wx = q.W * q.X, wy = q.W * q.Y, wz = q.W * q.Z;

        return new Matrix4x4(
            1f - 2f * (yy + zz), 2f * (xy - wz), 2f * (xz + wy), 0f,
            2f * (xy + wz), 1f - 2f * (xx + zz), 2f * (yz - wx), 0f,
            2f * (xz - wy), 2f * (yz + wx), 1f - 2f * (xx + yy), 0f,
            0f, 0f, 0f, 1f);
    }

    /// <summary>
    /// Converts an axis-angle representation to a quaternion.
    /// </summary>
    /// <param name="axis">The rotation axis (will be normalized).</param>
    /// <param name="angle">The rotation angle in radians.</param>
    /// <returns>The equivalent quaternion.</returns>
    public static Quaternion AxisAngleToQuaternion(Vector3 axis, float angle)
    {
        float halfAngle = angle * 0.5f;
        float s = MathF.Sin(halfAngle);
        float len = axis.Length();
        if (len < 1e-10f) return Quaternion.Identity;
        axis /= len;
        return new Quaternion(axis.X * s, axis.Y * s, axis.Z * s, MathF.Cos(halfAngle));
    }

    /// <summary>
    /// Slerp interpolation between two quaternions.
    /// </summary>
    /// <param name="q1">Start quaternion.</param>
    /// <param name="q2">End quaternion.</param>
    /// <param name="t">Interpolation parameter [0, 1].</param>
    /// <returns>The interpolated quaternion.</returns>
    public static Quaternion SlerpQuaternion(Quaternion q1, Quaternion q2, float t)
    {
        return Quaternion.Slerp(q1, q2, t);
    }

    #endregion

    #region Length and Length Squared

    /// <summary>
    /// Computes the squared length of a 3D vector (avoids sqrt).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float LengthSquared3(Vector3 v)
    {
        return v.X * v.X + v.Y * v.Y + v.Z * v.Z;
    }

    /// <summary>
    /// Computes the squared length of a 4D vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float LengthSquared4(Vector4 v)
    {
        return v.X * v.X + v.Y * v.Y + v.Z * v.Z + v.W * v.W;
    }

    /// <summary>
    /// Computes the length of a 3D vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Length3(Vector3 v)
    {
        return MathF.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);
    }

    /// <summary>
    /// Computes the length of a 4D vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Length4(Vector4 v)
    {
        return MathF.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z + v.W * v.W);
    }

    /// <summary>
    /// Batch length squared computation for interleaved 3D vectors.
    /// </summary>
    public static void BatchLengthSquared3(ReadOnlySpan<float> vectors, Span<float> lengthsSq, int count)
    {
        if (vectors.Length < count * 3 || lengthsSq.Length < count)
            throw new ArgumentException("Spans too small.");

        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            float x = vectors[off], y = vectors[off + 1], z = vectors[off + 2];
            lengthsSq[i] = x * x + y * y + z * z;
        }
    }

    /// <summary>
    /// Batch length computation for interleaved 3D vectors.
    /// </summary>
    public static void BatchLength3(ReadOnlySpan<float> vectors, Span<float> lengths, int count)
    {
        if (vectors.Length < count * 3 || lengths.Length < count)
            throw new ArgumentException("Spans too small.");

        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            float x = vectors[off], y = vectors[off + 1], z = vectors[off + 2];
            lengths[i] = MathF.Sqrt(x * x + y * y + z * z);
        }
    }

    #endregion

    #region Lerp, Clamp, Saturate

    /// <summary>
    /// Linearly interpolates between two 3D vectors.
    /// </summary>
    /// <param name="a">Start vector.</param>
    /// <param name="b">End vector.</param>
    /// <param name="t">Interpolation parameter [0, 1].</param>
    /// <returns>The interpolated vector: a + t * (b - a).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Lerp3(Vector3 a, Vector3 b, float t)
    {
        return a + t * (b - a);
    }

    /// <summary>
    /// Linearly interpolates between two 4D vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 Lerp4(Vector4 a, Vector4 b, float t)
    {
        return a + t * (b - a);
    }

    /// <summary>
    /// Clamps a scalar value between min and max.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Clamp(float value, float min, float max)
    {
        return MathF.Max(min, MathF.Min(max, value));
    }

    /// <summary>
    /// Clamps a 3D vector component-wise between min and max bounds.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Clamp3(Vector3 value, Vector3 min, Vector3 max)
    {
        return ComponentMax3(min, ComponentMin3(max, value));
    }

    /// <summary>
    /// Clamps a 4D vector component-wise between min and max bounds.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 Clamp4(Vector4 value, Vector4 min, Vector4 max)
    {
        return Vector4.Max(min, Vector4.Min(max, value));
    }

    /// <summary>
    /// Saturates a scalar value to the range [0, 1].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Saturate(float value)
    {
        return MathF.Max(0f, MathF.Min(1f, value));
    }

    /// <summary>
    /// Saturates a 3D vector component-wise to [0, 1].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Saturate3(Vector3 value)
    {
        return Clamp3(value, Vector3.Zero, Vector3.One);
    }

    /// <summary>
    /// Saturates a 4D vector component-wise to [0, 1].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 Saturate4(Vector4 value)
    {
        return Clamp4(value, Vector4.Zero, Vector4.One);
    }

    /// <summary>
    /// Batch linear interpolation for interleaved 3D vectors.
    /// </summary>
    /// <param name="a">Start vectors (x,y,z interleaved).</param>
    /// <param name="b">End vectors (x,y,z interleaved).</param>
    /// <param name="t">Interpolation parameter.</param>
    /// <param name="output">Output interpolated vectors.</param>
    /// <param name="count">Number of vectors.</param>
    public static void BatchLerp3(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, float t, Span<float> output, int count)
    {
        int len = count * 3;
        if (a.Length < len || b.Length < len || output.Length < len)
            throw new ArgumentException("Spans too small.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            var tVec = Vector256.Create(t);
            int simdEnd = len - (len % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                var result = Avx.Add(va, Avx.Multiply(tVec, Avx.Subtract(vb, va)));
                Vector256.StoreUnsafe(result, ref MemoryMarshal.GetReference(output.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            var tVec = Vector128.Create(t);
            int simdEnd = len - (len % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                var result = Sse.Add(va, Sse.Multiply(tVec, Sse.Subtract(vb, va)));
                Vector128.StoreUnsafe(result, ref MemoryMarshal.GetReference(output.Slice(i)));
            }
        }

        for (; i < len; i++)
            output[i] = a[i] + t * (b[i] - a[i]);
    }

    /// <summary>
    /// Batch clamp for float arrays.
    /// </summary>
    public static void BatchClamp(ReadOnlySpan<float> input, float min, float max, Span<float> output, int count)
    {
        if (input.Length < count || output.Length < count)
            throw new ArgumentException("Spans too small.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            var minVec = Vector256.Create(min);
            var maxVec = Vector256.Create(max);
            int simdEnd = count - (count % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(input.Slice(i)));
                v = Avx.Max(v, minVec);
                v = Avx.Min(v, maxVec);
                Vector256.StoreUnsafe(v, ref MemoryMarshal.GetReference(output.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            var minVec = Vector128.Create(min);
            var maxVec = Vector128.Create(max);
            int simdEnd = count - (count % 4);
            for (; i < simdEnd; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(input.Slice(i)));
                v = Sse.Max(v, minVec);
                v = Sse.Min(v, maxVec);
                Vector128.StoreUnsafe(v, ref MemoryMarshal.GetReference(output.Slice(i)));
            }
        }

        for (; i < count; i++)
            output[i] = MathF.Max(min, MathF.Min(max, input[i]));
    }

    /// <summary>
    /// Batch saturate for float arrays (clamp to [0, 1]).
    /// </summary>
    public static void BatchSaturate(ReadOnlySpan<float> input, Span<float> output, int count)
    {
        BatchClamp(input, 0f, 1f, output, count);
    }

    #endregion

    #region Vector Scale and Multiply

    /// <summary>
    /// Scales a 3D vector by a scalar.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Scale3(Vector3 v, float scale)
    {
        return new Vector3(v.X * scale, v.Y * scale, v.Z * scale);
    }

    /// <summary>
    /// Scales a 4D vector by a scalar.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 Scale4(Vector4 v, float scale)
    {
        return v * scale;
    }

    /// <summary>
    /// Batch scale for interleaved 3D vectors.
    /// </summary>
    public static void BatchScale3(ReadOnlySpan<float> vectors, float scale, Span<float> output, int count)
    {
        int len = count * 3;
        if (vectors.Length < len || output.Length < len)
            throw new ArgumentException("Spans too small.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            var sVec = Vector256.Create(scale);
            int simdEnd = len - (len % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(vectors.Slice(i)));
                Vector256.StoreUnsafe(Avx.Multiply(v, sVec), ref MemoryMarshal.GetReference(output.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            var sVec = Vector128.Create(scale);
            int simdEnd = len - (len % 4);
            for (; i < simdEnd; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(vectors.Slice(i)));
                Vector128.StoreUnsafe(Sse.Multiply(v, sVec), ref MemoryMarshal.GetReference(output.Slice(i)));
            }
        }

        for (; i < len; i++)
            output[i] = vectors[i] * scale;
    }

    #endregion

    #region Negate

    /// <summary>
    /// Negates a 3D vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Negate3(Vector3 v)
    {
        return new Vector3(-v.X, -v.Y, -v.Z);
    }

    /// <summary>
    /// Negates a 4D vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 Negate4(Vector4 v)
    {
        return -v;
    }

    /// <summary>
    /// Batch negate for interleaved 3D vectors.
    /// </summary>
    public static void BatchNegate3(ReadOnlySpan<float> vectors, Span<float> output, int count)
    {
        int len = count * 3;
        if (vectors.Length < len || output.Length < len)
            throw new ArgumentException("Spans too small.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            var zero = Vector256<float>.Zero;
            int simdEnd = len - (len % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(vectors.Slice(i)));
                Vector256.StoreUnsafe(Avx.Subtract(zero, v), ref MemoryMarshal.GetReference(output.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            var zero = Vector128<float>.Zero;
            int simdEnd = len - (len % 4);
            for (; i < simdEnd; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(vectors.Slice(i)));
                Vector128.StoreUnsafe(Sse.Subtract(zero, v), ref MemoryMarshal.GetReference(output.Slice(i)));
            }
        }

        for (; i < len; i++)
            output[i] = -vectors[i];
    }

    #endregion

    #region Reflect and Refract

    /// <summary>
    /// Reflects a 3D vector off a surface with the given normal.
    /// Formula: v - 2 * (v · n) * n
    /// </summary>
    /// <param name="vector">The incident vector.</param>
    /// <param name="normal">The surface normal (must be normalized).</param>
    /// <returns>The reflected vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Reflect3(Vector3 vector, Vector3 normal)
    {
        float dot = Dot3(vector, normal);
        return vector - 2f * dot * normal;
    }

    /// <summary>
    /// Refracts a 3D vector through a surface with the given normal.
    /// </summary>
    /// <param name="vector">The incident vector (normalized).</param>
    /// <param name="normal">The surface normal (normalized).</param>
    /// <param name="eta">The ratio of indices of refraction (n1/n2).</param>
    /// <returns>The refracted vector, or zero if total internal reflection occurs.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Refract3(Vector3 vector, Vector3 normal, float eta)
    {
        float dot = Dot3(vector, normal);
        float k = 1f - eta * eta * (1f - dot * dot);
        if (k < 0f) return Vector3.Zero;
        return eta * vector - (eta * dot + MathF.Sqrt(k)) * normal;
    }

    #endregion

    #region Angle Between Vectors

    /// <summary>
    /// Computes the angle in radians between two 3D vectors.
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>The angle in radians between the vectors.</returns>
    public static float AngleBetween3(Vector3 a, Vector3 b)
    {
        float lenA = a.Length();
        float lenB = b.Length();
        if (lenA < 1e-10f || lenB < 1e-10f) return 0f;
        float dot = Dot3(a, b) / (lenA * lenB);
        dot = MathF.Max(-1f, MathF.Min(1f, dot));
        return MathF.Acos(dot);
    }

    /// <summary>
    /// Computes the signed angle in radians between two 3D vectors around an axis.
    /// </summary>
    /// <param name="from">The source vector.</param>
    /// <param name="to">The target vector.</param>
    /// <param name="axis">The rotation axis.</param>
    /// <returns>The signed angle in radians.</returns>
    public static float SignedAngleBetween3(Vector3 from, Vector3 to, Vector3 axis)
    {
        float unsigned = AngleBetween3(from, to);
        float sign = Vector3.Dot(Vector3.Cross(from, to), axis);
        return sign < 0 ? -unsigned : unsigned;
    }

    #endregion

    #region Sum and Average

    /// <summary>
    /// Computes the sum of a batch of 3D vectors.
    /// </summary>
    /// <param name="vectors">Interleaved vector data (x,y,z).</param>
    /// <param name="count">Number of vectors.</param>
    /// <returns>The sum vector.</returns>
    public static Vector3 BatchSum3(ReadOnlySpan<float> vectors, int count)
    {
        if (vectors.Length < count * 3)
            throw new ArgumentException("Span too small.");

        float sx = 0, sy = 0, sz = 0;
        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            sx += vectors[off];
            sy += vectors[off + 1];
            sz += vectors[off + 2];
        }
        return new Vector3(sx, sy, sz);
    }

    /// <summary>
    /// Computes the average of a batch of 3D vectors.
    /// </summary>
    /// <param name="vectors">Interleaved vector data (x,y,z).</param>
    /// <param name="count">Number of vectors.</param>
    /// <returns>The average vector.</returns>
    public static Vector3 BatchAverage3(ReadOnlySpan<float> vectors, int count)
    {
        if (count == 0) return Vector3.Zero;
        return BatchSum3(vectors, count) / count;
    }

    #endregion
}
