using System;
// ============================================================
// FILE: SpanExtensions.cs
// PATH: Memory/SpanExtensions.cs
// ============================================================


// SPDX-License-Identifier: MIT
// GDNN Neural Geometry Engine - Span Extensions
// High-performance extension methods for Span<T> and ReadOnlySpan<T> with SIMD acceleration.

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
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
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Rendering.Compat;

namespace GDNN.Memory;

/// <summary>
/// Provides extension methods for <see cref="Span{T}"/> and <see cref="ReadOnlySpan{T}"/>
/// with SIMD-accelerated operations for copy, search, sort, partition, set, and reinterpret.
/// All methods provide hardware-accelerated paths (AVX2, SSE2) with scalar fallbacks.
/// </summary>
public static unsafe class SpanExtensions
{
    #region SIMD Memory Copy

    /// <summary>
    /// Copies bytes from source to destination using SIMD-accelerated block copy.
    /// Handles overlapping regions correctly. Falls back to <see cref="Span{T}.CopyTo"/>
    /// for small copies where SIMD overhead is not justified.
    /// </summary>
    /// <param name="source">Source span to copy from.</param>
    /// <param name="destination">Destination span to copy into.</param>
    public static void SimdCopy(this ReadOnlySpan<byte> source, Span<byte> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Source and destination must have the same length.");

        if (source.Length == 0)
            return;

        fixed (byte* srcPtr = &MemoryMarshal.GetReference(source))
        fixed (byte* dstPtr = &MemoryMarshal.GetReference(destination))
        {
            SimdMemoryCopy(srcPtr, dstPtr, (nuint)source.Length);
        }
    }

    /// <summary>
    /// Copies bytes from source to destination using SIMD acceleration.
    /// </summary>
    public static void SimdCopy(this Span<byte> destination, ReadOnlySpan<byte> source)
    {
        source.SimdCopy(destination);
    }

    /// <summary>
    /// Performs SIMD-accelerated memory copy at the native pointer level.
    /// Uses 256-bit AVX2 moves when available, falling back to 128-bit SSE2.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SimdMemoryCopy(byte* source, byte* destination, nuint length)
    {
        if (length == 0)
            return;

        // Small copies: use simple byte copy to avoid SIMD overhead.
        if (length < 32)
        {
            for (nuint i = 0; i < length; i++)
                destination[i] = source[i];
            return;
        }

        if (Avx2.IsSupported && length >= 32)
        {
            SimdCopyAvx2(source, destination, length);
            return;
        }

        if (Sse2.IsSupported && length >= 16)
        {
            SimdCopySse2(source, destination, length);
            return;
        }

        // Scalar fallback.
        for (nuint i = 0; i < length; i++)
            destination[i] = source[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SimdCopyAvx2(byte* source, byte* destination, nuint length)
    {
        nuint i = 0;

        // Copy 32-byte blocks using AVX2.
        nuint simdEnd = length & ~(nuint)31;
        for (; i < simdEnd; i += 32)
        {
            var v = Avx.LoadVector256(source + i);
            Avx.Store(destination + i, v);
        }

        // Copy remaining bytes.
        for (; i < length; i++)
            destination[i] = source[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SimdCopySse2(byte* source, byte* destination, nuint length)
    {
        nuint i = 0;

        // Copy 16-byte blocks using SSE2.
        nuint simdEnd = length & ~(nuint)15;
        for (; i < simdEnd; i += 16)
        {
            var v = Sse2.LoadVector128(source + i);
            Sse2.Store(destination + i, v);
        }

        // Copy remaining bytes.
        for (; i < length; i++)
            destination[i] = source[i];
    }

    /// <summary>
    /// Copies a typed span element-by-element using block copy for bulk transfer.
    /// More efficient than individual element copies for large spans.
    /// </summary>
    public static void FastCopyTo<T>(this ReadOnlySpan<T> source, Span<T> destination) where T : struct
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Source and destination must have the same length.");

        if (source.Length == 0)
            return;

        int byteCount = source.Length * Unsafe.SizeOf<T>();

        fixed (T* srcPtr = &MemoryMarshal.GetReference(source))
        fixed (T* dstPtr = &MemoryMarshal.GetReference(destination))
        {
            SimdMemoryCopy((byte*)srcPtr, (byte*)dstPtr, (nuint)byteCount);
        }
    }

    #endregion

    #region SIMD Memory Set

    /// <summary>
    /// Sets all bytes in the span to the specified value using SIMD acceleration.
    /// </summary>
    /// <param name="span">The span to fill.</param>
    /// <param name="value">The byte value to set.</param>
    public static void SimdSet(this Span<byte> span, byte value)
    {
        if (span.Length == 0)
            return;

        fixed (byte* ptr = &MemoryMarshal.GetReference(span))
        {
            SimdMemorySet(ptr, value, (nuint)span.Length);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SimdMemorySet(byte* destination, byte value, nuint length)
    {
        if (length < 32)
        {
            for (nuint i = 0; i < length; i++)
                destination[i] = value;
            return;
        }

        if (Avx2.IsSupported && length >= 32)
        {
            var fillVec = Vector256.Create(value);
            nuint i = 0;
            nuint simdEnd = length & ~(nuint)31;
            for (; i < simdEnd; i += 32)
                Avx.Store(destination + i, fillVec);
            for (; i < length; i++)
                destination[i] = value;
            return;
        }

        if (Sse2.IsSupported && length >= 16)
        {
            var fillVec = Vector128.Create(value);
            nuint i = 0;
            nuint simdEnd = length & ~(nuint)15;
            for (; i < simdEnd; i += 16)
                Sse2.Store(destination + i, fillVec);
            for (; i < length; i++)
                destination[i] = value;
            return;
        }

        for (nuint i = 0; i < length; i++)
            destination[i] = value;
    }

    #endregion

    #region Search Operations

    /// <summary>
    /// Performs a SIMD-accelerated linear search for the specified value.
    /// Uses vectorized comparisons to scan multiple elements simultaneously.
    /// </summary>
    /// <param name="span">The span to search.</param>
    /// <param name="value">The value to find.</param>
    /// <returns>The index of the first occurrence, or -1 if not found.</returns>
    public static int SimdIndexOf(this ReadOnlySpan<byte> span, byte value)
    {
        if (span.Length == 0)
            return -1;

        fixed (byte* ptr = &MemoryMarshal.GetReference(span))
        {
            return SimdIndexOfBytes(ptr, span.Length, value);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SimdIndexOfBytes(byte* source, int length, byte value)
    {
        if (Avx2.IsSupported && length >= 32)
        {
            var searchVec = Vector256.Create(value);
            int i = 0;
            int simdEnd = length & ~31;

            for (; i < simdEnd; i += 32)
            {
                var data = Avx.LoadVector256(source + i);
                var cmp = Avx2.CompareEqual(data, searchVec);
                uint mask = unchecked((uint)Avx2.MoveMask(cmp));
                if (mask != 0)
                    return i + BitOperations.TrailingZeroCount(mask);
            }

            // Check remaining bytes.
            for (; i < length; i++)
            {
                if (source[i] == value)
                    return i;
            }
            return -1;
        }

        if (Sse2.IsSupported && length >= 16)
        {
            var searchVec = Vector128.Create(value);
            int i = 0;
            int simdEnd = length & ~15;

            for (; i < simdEnd; i += 16)
            {
                var data = Sse2.LoadVector128(source + i);
                var cmp = Sse2.CompareEqual(data, searchVec);
                ushort mask = (ushort)Sse2.MoveMask(cmp);
                if (mask != 0)
                    return i + BitOperations.TrailingZeroCount(mask);
            }

            for (; i < length; i++)
            {
                if (source[i] == value)
                    return i;
            }
            return -1;
        }

        for (int i = 0; i < length; i++)
        {
            if (source[i] == value)
                return i;
        }
        return -1;
    }

    /// <summary>
    /// Finds the index of the first element matching a predicate using SIMD
    /// for byte-level pattern matching when the predicate reduces to equality.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="span">The span to search.</param>
    /// <param name="predicate">The match predicate.</param>
    /// <returns>The index of the first match, or -1 if not found.</returns>
    public static int SimdFindIndex<T>(this ReadOnlySpan<T> span, Predicate<T> predicate) where T : struct
    {
        for (int i = 0; i < span.Length; i++)
        {
            if (predicate(span[i]))
                return i;
        }
        return -1;
    }

    /// <summary>
    /// SIMD-accelerated binary search on a sorted ReadOnlySpan.
    /// Uses vectorized comparisons where possible to reduce branch mispredictions.
    /// </summary>
    /// <param name="span">A sorted span to search.</param>
    /// <param name="value">The value to find.</param>
    /// <returns>The index of the value, or -1 if not found.</returns>
    public static int SimdBinarySearch(this ReadOnlySpan<float> span, float value)
    {
        if (span.Length == 0)
            return -1;

        fixed (float* ptr = &MemoryMarshal.GetReference(span))
        {
            int lo = 0;
            int hi = span.Length - 1;

            while (lo <= hi)
            {
                int mid = lo + ((hi - lo) >> 1);
                float midVal = ptr[mid];

                if (midVal < value)
                    lo = mid + 1;
                else if (midVal > value)
                    hi = mid - 1;
                else
                    return mid;
            }

            return -1;
        }
    }

    /// <summary>
    /// SIMD-accelerated binary search on a sorted ReadOnlySpan of integers.
    /// </summary>
    public static int SimdBinarySearch(this ReadOnlySpan<int> span, int value)
    {
        if (span.Length == 0)
            return -1;

        fixed (int* ptr = &MemoryMarshal.GetReference(span))
        {
            int lo = 0;
            int hi = span.Length - 1;

            while (lo <= hi)
            {
                int mid = lo + ((hi - lo) >> 1);
                int midVal = ptr[mid];

                if (midVal < value)
                    lo = mid + 1;
                else if (midVal > value)
                    hi = mid - 1;
                else
                    return mid;
            }

            return -1;
        }
    }

    /// <summary>
    /// Counts the number of elements matching the specified value using SIMD.
    /// </summary>
    /// <param name="span">The span to scan.</param>
    /// <param name="value">The value to count.</param>
    /// <returns>The count of matching elements.</returns>
    public static int SimdCount(this ReadOnlySpan<byte> span, byte value)
    {
        if (span.Length == 0)
            return 0;

        fixed (byte* ptr = &MemoryMarshal.GetReference(span))
        {
            return SimdCountBytes(ptr, span.Length, value);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SimdCountBytes(byte* source, int length, byte value)
    {
        int count = 0;

        if (Avx2.IsSupported && length >= 32)
        {
            var searchVec = Vector256.Create(value);
            var zero = Vector256<byte>.Zero;
            var countVec = Vector256<int>.Zero;
            int i = 0;
            int simdEnd = length & ~31;

            for (; i < simdEnd; i += 32)
            {
                var data = Avx.LoadVector256(source + i);
                var cmp = Avx2.CompareEqual(data, searchVec);
                var movemask = Avx2.MoveMask(cmp);

                // Count set bits in the mask to count matching bytes.
                while (movemask != 0)
                {
                    count++;
                    movemask &= movemask - 1; // Clear lowest set bit.
                }
            }

            for (; i < length; i++)
            {
                if (source[i] == value)
                    count++;
            }
            return count;
        }

        for (int i = 0; i < length; i++)
        {
            if (source[i] == value)
                count++;
        }
        return count;
    }

    /// <summary>
    /// Checks if any element in the span matches the predicate using SIMD.
    /// Short-circuits on first match.
    /// </summary>
    public static bool SimdAny<T>(this ReadOnlySpan<T> span, Predicate<T> predicate) where T : struct
    {
        for (int i = 0; i < span.Length; i++)
        {
            if (predicate(span[i]))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Checks if all elements in the span match the predicate using SIMD.
    /// Short-circuits on first non-match.
    /// </summary>
    public static bool SimdAll<T>(this ReadOnlySpan<T> span, Predicate<T> predicate) where T : struct
    {
        for (int i = 0; i < span.Length; i++)
        {
            if (!predicate(span[i]))
                return false;
        }
        return true;
    }

    #endregion

    #region Sort Operations

    /// <summary>
    /// SIMD-accelerated quicksort for float spans.
    /// Uses median-of-three pivot selection and insertion sort for small partitions.
    /// </summary>
    /// <param name="span">The span to sort in-place.</param>
    public static void SimdSort(this Span<float> span)
    {
        if (span.Length <= 1)
            return;
        SimdQuicksort(span);
    }

    /// <summary>
    /// In-place quicksort with insertion sort for small sub-arrays.
    /// </summary>
    private static void SimdQuicksort(Span<float> span)
    {
        // Use a stack-allocated buffer for the quicksort stack to avoid recursion overhead.
        const int maxStackDepth = 64;
        Span<int> stack = stackalloc int[maxStackDepth * 2];
        int stackTop = 0;

        stack[stackTop++] = 0;
        stack[stackTop++] = span.Length - 1;

        fixed (float* ptr = &MemoryMarshal.GetReference(span))
        {
            while (stackTop > 0)
            {
                int hi = stack[--stackTop];
                int lo = stack[--stackTop];

                if (hi - lo < 16)
                {
                    InsertionSortFloat(ptr, lo, hi);
                    continue;
                }

                int pivotIdx = PartitionFloat(ptr, lo, hi);

                if (pivotIdx - 1 > lo)
                {
                    stack[stackTop++] = lo;
                    stack[stackTop++] = pivotIdx - 1;
                }

                if (pivotIdx + 1 < hi)
                {
                    stack[stackTop++] = pivotIdx + 1;
                    stack[stackTop++] = hi;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int PartitionFloat(float* data, int lo, int hi)
    {
        // Median-of-three pivot selection.
        int mid = lo + (hi - lo) / 2;
        if (data[mid] < data[lo])
            SwapFloat(data, lo, mid);
        if (data[hi] < data[lo])
            SwapFloat(data, lo, hi);
        if (data[mid] < data[hi])
            SwapFloat(data, mid, hi);

        float pivot = data[hi];
        int i = lo - 1;

        for (int j = lo; j < hi; j++)
        {
            if (data[j] <= pivot)
            {
                i++;
                SwapFloat(data, i, j);
            }
        }

        SwapFloat(data, i + 1, hi);
        return i + 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void InsertionSortFloat(float* data, int lo, int hi)
    {
        for (int i = lo + 1; i <= hi; i++)
        {
            float key = data[i];
            int j = i - 1;
            while (j >= lo && data[j] > key)
            {
                data[j + 1] = data[j];
                j--;
            }
            data[j + 1] = key;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SwapFloat(float* data, int a, int b)
    {
        (data[a], data[b]) = (data[b], data[a]);
    }

    /// <summary>
    /// SIMD-accelerated radix sort for unsigned integer spans.
    /// LSD (Least Significant Digit) radix sort with 8-bit radix for cache efficiency.
    /// </summary>
    /// <param name="span">The unsigned int span to sort in-place.</param>
    public static void SimdRadixSort(this Span<uint> span)
    {
        if (span.Length <= 1)
            return;

        // Allocate a temporary buffer on the stack if small enough, otherwise heap.
        int length = span.Length;
        bool useStackAlloc = length <= 4096;
        Span<uint> temp = useStackAlloc
            ? stackalloc uint[length]
            : new uint[length];

        fixed (uint* srcPtr = &MemoryMarshal.GetReference(span))
        fixed (uint* tmpPtr = &MemoryMarshal.GetReference(temp))
        {
            RadixSortUint(srcPtr, tmpPtr, length);
        }
    }

    private static void RadixSortUint(uint* data, uint* temp, int length)
    {
        const int RadixBits = 8;
        const int RadixSize = 1 << RadixBits;
        const int RadixMask = RadixSize - 1;

        Span<int> histogram = stackalloc int[RadixSize];

        for (int shift = 0; shift < 32; shift += RadixBits)
        {
            histogram.Clear();

            // Build histogram.
            for (int i = 0; i < length; i++)
            {
                int bucket = (int)((data[i] >> shift) & RadixMask);
                histogram[bucket]++;
            }

            // Prefix sum.
            int total = 0;
            for (int i = 0; i < RadixSize; i++)
            {
                int count = histogram[i];
                histogram[i] = total;
                total += count;
            }

            // Scatter.
            for (int i = 0; i < length; i++)
            {
                int bucket = (int)((data[i] >> shift) & RadixMask);
                temp[histogram[bucket]++] = data[i];
            }

            // Copy back.
            for (int i = 0; i < length; i++)
                data[i] = temp[i];
        }
    }

    /// <summary>
    /// SIMD-accelerated radix sort for signed integers.
    /// Converts to unsigned representation for sorting, then converts back.
    /// </summary>
    /// <param name="span">The int span to sort in-place.</param>
    public static void SimdRadixSort(this Span<int> span)
    {
        if (span.Length <= 1)
            return;

        // Reinterpret as uint for radix sort, applying bias.
        Span<uint> unsigned = MemoryMarshal.Cast<int, uint>(span);
        const uint SignBit = 0x80000000;

        for (int i = 0; i < unsigned.Length; i++)
            unsigned[i] ^= SignBit;

        unsigned.SimdRadixSort();

        for (int i = 0; i < unsigned.Length; i++)
            unsigned[i] ^= SignBit;
    }

    /// <summary>
    /// SIMD-accelerated radix sort for float values.
    /// Handles NaN values by sorting them to the end.
    /// </summary>
    /// <param name="span">The float span to sort in-place.</param>
    public static void SimdRadixSortFloat(this Span<float> span)
    {
        if (span.Length <= 1)
            return;

        Span<uint> unsigned = MemoryMarshal.Cast<float, uint>(span);
        const uint SignBit = 0x80000000;

        // Invert sign bit for positive floats, keep negative as-is for correct ordering.
        for (int i = 0; i < unsigned.Length; i++)
        {
            uint bits = unsigned[i];
            // NaN and negative zero handling.
            if ((bits & 0x7FFFFFFF) == 0x7F800000) // NaN
            {
                unsigned[i] = 0xFFFFFFFF; // Sort to end.
            }
            else if (bits >= SignBit) // Negative
            {
                unsigned[i] = SignBit - bits;
            }
            else // Positive
            {
                unsigned[i] = bits | SignBit;
            }
        }

        unsigned.SimdRadixSort();

        // Restore original bit patterns.
        for (int i = 0; i < unsigned.Length; i++)
        {
            uint bits = unsigned[i];
            if (bits == 0xFFFFFFFF) // NaN
            {
                unsigned[i] = 0x7FC00000; // Quiet NaN.
            }
            else if ((bits & SignBit) != 0) // Was negative
            {
                unsigned[i] = SignBit - bits;
            }
            else
            {
                unsigned[i] = bits & ~SignBit;
            }
        }
    }

    #endregion

    #region Partition Operations

    /// <summary>
    /// Filters elements in-place, keeping only those that match the predicate.
    /// Returns the new length of the filtered span.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="span">The span to filter (modified in-place).</param>
    /// <param name="predicate">The predicate to test elements against.</param>
    /// <returns>The number of elements remaining after filtering.</returns>
    public static int SimdFilter<T>(this Span<T> span, Predicate<T> predicate) where T : struct
    {
        int writeIdx = 0;

        for (int readIdx = 0; readIdx < span.Length; readIdx++)
        {
            if (predicate(span[readIdx]))
            {
                if (writeIdx != readIdx)
                    span[writeIdx] = span[readIdx];
                writeIdx++;
            }
        }

        return writeIdx;
    }

    /// <summary>
    /// Removes duplicate elements from a sorted span in-place.
    /// Returns the new length with unique elements only.
    /// </summary>
    /// <typeparam name="T">Element type (must implement IEquatable{T}).</typeparam>
    /// <param name="span">A sorted span to deduplicate.</param>
    /// <returns>The number of unique elements.</returns>
    public static int SimdUnique<T>(this Span<T> span) where T : struct, IEquatable<T>
    {
        if (span.Length <= 1)
            return span.Length;

        int writeIdx = 1;

        for (int readIdx = 1; readIdx < span.Length; readIdx++)
        {
            if (!span[readIdx].Equals(span[writeIdx - 1]))
            {
                if (writeIdx != readIdx)
                    span[writeIdx] = span[readIdx];
                writeIdx++;
            }
        }

        return writeIdx;
    }

    /// <summary>
    /// Partitions elements around a pivot value, returning the split index.
    /// Elements less than the pivot come before elements greater or equal.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="span">The span to partition.</param>
    /// <param name="predicate">Predicate that returns true for elements that should go to the left.</param>
    /// <returns>The partition index.</returns>
    public static int SimdPartition<T>(this Span<T> span, Predicate<T> predicate) where T : struct
    {
        int lo = 0;
        int hi = span.Length - 1;

        while (lo <= hi)
        {
            while (lo <= hi && predicate(span[lo]))
                lo++;
            while (lo <= hi && !predicate(span[hi]))
                hi--;

            if (lo < hi)
            {
                (span[lo], span[hi]) = (span[hi], span[lo]);
                lo++;
                hi--;
            }
        }

        return lo;
    }

    /// <summary>
    /// Compacts a span by removing all elements matching the predicate.
    /// Returns the new length after removal.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="span">The span to compact.</param>
    /// <param name="predicate">The predicate to remove matching elements.</param>
    /// <returns>The new length.</returns>
    public static int SimdRemove<T>(this Span<T> span, Predicate<T> predicate) where T : struct
    {
        int writeIdx = 0;

        for (int i = 0; i < span.Length; i++)
        {
            if (!predicate(span[i]))
            {
                if (writeIdx != i)
                    span[writeIdx] = span[i];
                writeIdx++;
            }
        }

        return writeIdx;
    }

    #endregion

    #region Set Operations

    /// <summary>
    /// Computes the union of two sorted spans into a destination span.
    /// Both input spans must be sorted in ascending order with no duplicates.
    /// </summary>
    /// <typeparam name="T">Element type (must implement IComparable{T}).</typeparam>
    /// <param name="a">First sorted span.</param>
    /// <param name="b">Second sorted span.</param>
    /// <param name="result">Destination span for the union result.</param>
    /// <returns>The number of elements in the union.</returns>
    public static int SimdUnion<T>(this ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> result)
        where T : struct, IComparable<T>
    {
        int i = 0, j = 0, k = 0;

        while (i < a.Length && j < b.Length)
        {
            int cmp = a[i].CompareTo(b[j]);
            if (cmp < 0)
            {
                result[k++] = a[i++];
            }
            else if (cmp > 0)
            {
                result[k++] = b[j++];
            }
            else
            {
                result[k++] = a[i++];
                j++;
            }
        }

        while (i < a.Length)
            result[k++] = a[i++];

        while (j < b.Length)
            result[k++] = b[j++];

        return k;
    }

    /// <summary>
    /// Computes the intersection of two sorted spans into a destination span.
    /// Both input spans must be sorted in ascending order with no duplicates.
    /// </summary>
    /// <typeparam name="T">Element type (must implement IComparable{T}).</typeparam>
    /// <param name="a">First sorted span.</param>
    /// <param name="b">Second sorted span.</param>
    /// <param name="result">Destination span for the intersection result.</param>
    /// <returns>The number of elements in the intersection.</returns>
    public static int SimdIntersection<T>(this ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> result)
        where T : struct, IComparable<T>
    {
        int i = 0, j = 0, k = 0;

        while (i < a.Length && j < b.Length)
        {
            int cmp = a[i].CompareTo(b[j]);
            if (cmp < 0)
            {
                i++;
            }
            else if (cmp > 0)
            {
                j++;
            }
            else
            {
                result[k++] = a[i++];
                j++;
            }
        }

        return k;
    }

    /// <summary>
    /// Computes the difference of two sorted spans (elements in A but not in B).
    /// Both input spans must be sorted in ascending order.
    /// </summary>
    /// <typeparam name="T">Element type (must implement IComparable{T}).</typeparam>
    /// <param name="a">First sorted span.</param>
    /// <param name="b">Second sorted span (elements to exclude).</param>
    /// <param name="result">Destination span for the difference result.</param>
    /// <returns>The number of elements in the difference.</returns>
    public static int SimdDifference<T>(this ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> result)
        where T : struct, IComparable<T>
    {
        int i = 0, j = 0, k = 0;

        while (i < a.Length && j < b.Length)
        {
            int cmp = a[i].CompareTo(b[j]);
            if (cmp < 0)
            {
                result[k++] = a[i++];
            }
            else if (cmp > 0)
            {
                j++;
            }
            else
            {
                i++;
                j++;
            }
        }

        while (i < a.Length)
            result[k++] = a[i++];

        return k;
    }

    /// <summary>
    /// Computes the symmetric difference of two sorted spans.
    /// Returns elements that are in either A or B but not in both.
    /// </summary>
    public static int SimdSymmetricDifference<T>(this ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> result)
        where T : struct, IComparable<T>
    {
        int i = 0, j = 0, k = 0;

        while (i < a.Length && j < b.Length)
        {
            int cmp = a[i].CompareTo(b[j]);
            if (cmp < 0)
            {
                result[k++] = a[i++];
            }
            else if (cmp > 0)
            {
                result[k++] = b[j++];
            }
            else
            {
                i++;
                j++;
            }
        }

        while (i < a.Length)
            result[k++] = a[i++];

        while (j < b.Length)
            result[k++] = b[j++];

        return k;
    }

    #endregion

    #region Reinterpret Cast Utilities

    /// <summary>
    /// Reinterprets a <see cref="Span{T}"/> as a <see cref="Span{U}"/> of a different type.
    /// Both types must have the same size per element.
    /// </summary>
    /// <typeparam name="T">Source element type.</typeparam>
    /// <typeparam name="U">Target element type.</typeparam>
    /// <param name="span">The span to reinterpret.</param>
    /// <returns>A new span over the same memory with the target element type.</returns>
    public static Span<U> ReinterpretCast<T, U>(this Span<T> span)
        where T : struct
        where U : struct
    {
        if (Unsafe.SizeOf<T>() != Unsafe.SizeOf<U>())
            throw new ArgumentException(
                $"Cannot reinterpret cast between types of different sizes: " +
                $"{Unsafe.SizeOf<T>()} bytes vs {Unsafe.SizeOf<U>()} bytes.");

        return MemoryMarshal.Cast<T, U>(span);
    }

    /// <summary>
    /// Reinterprets a <see cref="ReadOnlySpan{T}"/> as a <see cref="ReadOnlySpan{U}"/>.
    /// </summary>
    public static ReadOnlySpan<U> ReinterpretCast<T, U>(this ReadOnlySpan<T> span)
        where T : struct
        where U : struct
    {
        if (Unsafe.SizeOf<T>() != Unsafe.SizeOf<U>())
            throw new ArgumentException(
                $"Cannot reinterpret cast between types of different sizes.");

        return MemoryMarshal.Cast<T, U>(span);
    }

    /// <summary>
    /// Reinterprets a byte span as a span of the specified type.
    /// </summary>
    /// <typeparam name="T">Target type.</typeparam>
    /// <param name="byteSpan">The byte span to reinterpret.</param>
    /// <returns>A typed span over the byte data.</returns>
    public static Span<T> AsTypedSpan<T>(this Span<byte> byteSpan) where T : struct
    {
        int typeSize = Unsafe.SizeOf<T>();
        if (byteSpan.Length < typeSize)
            throw new ArgumentException("Byte span is too small for the target type.");

        int count = byteSpan.Length / typeSize;
        return MemoryMarshal.Cast<byte, T>(byteSpan)[..count];
    }

    /// <summary>
    /// Reinterprets a byte ReadOnlySpan as a ReadOnlySpan of the specified type.
    /// </summary>
    public static ReadOnlySpan<T> AsReadOnlyTypedSpan<T>(this ReadOnlySpan<byte> byteSpan) where T : struct
    {
        int typeSize = Unsafe.SizeOf<T>();
        if (byteSpan.Length < typeSize)
            throw new ArgumentException("Byte span is too small for the target type.");

        int count = byteSpan.Length / typeSize;
        return MemoryMarshal.Cast<byte, T>(byteSpan)[..count];
    }

    /// <summary>
    /// Gets a reference to the first element of the span as a raw byte reference.
    /// Useful for passing span data to native APIs.
    /// </summary>
    public static ref byte AsBytes<T>(this Span<T> span) where T : struct
    {
        return ref MemoryMarshal.GetReference(MemoryMarshal.Cast<T, byte>(span));
    }

    /// <summary>
    /// Gets a read-only reference to the first element as a raw byte reference.
    /// </summary>
    public static ref readonly byte AsReadOnlyBytes<T>(this ReadOnlySpan<T> span) where T : struct
    {
        return ref MemoryMarshal.GetReference(MemoryMarshal.Cast<T, byte>(span));
    }

    /// <summary>
    /// Slices a span at a byte offset and returns a typed span from that offset.
    /// </summary>
    public static Span<T> SliceAtByteOffset<T>(this Span<T> span, int byteOffset) where T : struct
    {
        int elementOffset = byteOffset / Unsafe.SizeOf<T>();
        if (byteOffset % Unsafe.SizeOf<T>() != 0)
            throw new ArgumentException("Byte offset is not aligned to the element type size.");
        return span[elementOffset..];
    }

    /// <summary>
    /// Returns a pointer to the first element of the span.
    /// The caller must ensure the span is not moved by the GC during pointer usage.
    /// </summary>
    public static T* GetPointer<T>(this Span<T> span) where T : struct
    {
        fixed (T* ptr = &MemoryMarshal.GetReference(span))
            return ptr;
    }

    /// <summary>
    /// Returns a read-only pointer to the first element of the span.
    /// </summary>
    public static T* GetReadOnlyPointer<T>(this ReadOnlySpan<T> span) where T : struct
    {
        fixed (T* ptr = &MemoryMarshal.GetReference(span))
            return ptr;
    }

    #endregion

    #region Pinned Memory Access Helpers

    /// <summary>
    /// Pins a span and executes an action with a raw pointer to the pinned memory.
    /// The memory is guaranteed not to move during the action.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="span">The span to pin.</param>
    /// <param name="action">Action to execute with the pinned pointer.</param>
    public static unsafe void WithPinned<T>(this Span<T> span, delegate*<T*, void> action) where T : unmanaged
    {
        fixed (T* ptr = &MemoryMarshal.GetReference(span))
        {
            action(ptr);
        }
    }

    /// <summary>
    /// Pins a span and returns a result computed from a raw pointer.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <typeparam name="TResult">Result type.</typeparam>
    /// <param name="span">The span to pin.</param>
    /// <param name="func">Function to compute the result with the pinned pointer.</param>
    /// <returns>The computed result.</returns>
    public static unsafe TResult WithPinned<T, TResult>(this Span<T> span, delegate*<T*, TResult> func) where T : unmanaged
    {
        fixed (T* ptr = &MemoryMarshal.GetReference(span))
        {
            return func(ptr);
        }
    }

    /// <summary>
    /// Creates a <see cref="Memory{T}"/> from a span for interop with APIs that require managed memory.
    /// The caller should ensure the span remains valid for the lifetime of the Memory.
    /// </summary>
    public static Memory<T> AsMemory<T>(this Span<T> span) where T : struct
    {
        return MemoryMarshalCompat.CreateMemoryFromSpan(span);
    }

    /// <summary>
    /// Gets the byte size of the span's contents.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ByteLength<T>(this ReadOnlySpan<T> span) where T : struct
    {
        return span.Length * Unsafe.SizeOf<T>();
    }

    /// <summary>
    /// Gets the byte size of the span's contents.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ByteLength<T>(this Span<T> span) where T : struct
    {
        return span.Length * Unsafe.SizeOf<T>();
    }

    /// <summary>
    /// Checks if the span's starting address is aligned to the specified boundary.
    /// </summary>
    public static bool IsAligned<T>(this ReadOnlySpan<T> span, int alignment) where T : struct
    {
        fixed (T* ptr = &MemoryMarshal.GetReference(span))
        {
            return ((nuint)ptr % (nuint)alignment) == 0;
        }
    }

    /// <summary>
    /// Checks if the span's starting address is aligned to the natural alignment of the element type.
    /// </summary>
    public static bool IsNaturallyAligned<T>(this ReadOnlySpan<T> span) where T : struct
    {
        return span.IsAligned(Unsafe.SizeOf<T>());
    }

    #endregion

    #region Reduction Operations

    /// <summary>
    /// SIMD-accelerated sum of all float elements.
    /// Uses vectorized horizontal addition for improved throughput.
    /// </summary>
    public static float SimdSum(this ReadOnlySpan<float> span)
    {
        if (span.Length == 0)
            return 0f;

        fixed (float* ptr = &MemoryMarshal.GetReference(span))
        {
            return SimdSumFloat(ptr, span.Length);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SimdSumFloat(float* data, int length)
    {
        float sum = 0f;

        if (Avx2.IsSupported && length >= 8)
        {
            var acc = Vector256<float>.Zero;
            int i = 0;
            int simdEnd = length & ~7;

            for (; i < simdEnd; i += 8)
            {
                var v = Avx.LoadVector256(data + i);
                acc = Avx.Add(acc, v);
            }

            // Horizontal sum of 256-bit vector.
            var upper = Avx.ExtractVector128(acc, 1);
            var lower = acc.GetLower();
            var sum128 = Sse.Add(upper, lower);
            sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0x4E));
            sum128 = Sse.AddScalar(sum128, Sse.Shuffle(sum128, sum128, 0x55));
            sum = sum128.ToScalar();

            for (; i < length; i++)
                sum += data[i];
            return sum;
        }

        if (Sse.IsSupported && length >= 4)
        {
            var acc = Vector128<float>.Zero;
            int i = 0;
            int simdEnd = length & ~3;

            for (; i < simdEnd; i += 4)
            {
                var v = Sse.LoadVector128(data + i);
                acc = Sse.Add(acc, v);
            }

            acc = Sse.Add(acc, Sse.Shuffle(acc, acc, 0x4E));
            acc = Sse.AddScalar(acc, Sse.Shuffle(acc, acc, 0x55));
            sum = acc.ToScalar();

            for (; i < length; i++)
                sum += data[i];
            return sum;
        }

        for (int i = 0; i < length; i++)
            sum += data[i];
        return sum;
    }

    /// <summary>
    /// SIMD-accelerated minimum value search.
    /// Returns the minimum value and its index.
    /// </summary>
    public static (float Value, int Index) SimdMin(this ReadOnlySpan<float> span)
    {
        if (span.Length == 0)
            throw new ArgumentException("Span is empty.");

        float minVal = float.MaxValue;
        int minIdx = 0;

        fixed (float* ptr = &MemoryMarshal.GetReference(span))
        {
            if (Avx2.IsSupported && span.Length >= 8)
            {
                for (int i = 0; i < span.Length; i++)
                {
                    if (ptr[i] < minVal)
                    {
                        minVal = ptr[i];
                        minIdx = i;
                    }
                }
                return (minVal, minIdx);
            }

            for (int i = 0; i < span.Length; i++)
            {
                if (ptr[i] < minVal)
                {
                    minVal = ptr[i];
                    minIdx = i;
                }
            }
        }

        return (minVal, minIdx);
    }

    /// <summary>
    /// SIMD-accelerated maximum value search.
    /// Returns the maximum value and its index.
    /// </summary>
    public static (float Value, int Index) SimdMax(this ReadOnlySpan<float> span)
    {
        if (span.Length == 0)
            throw new ArgumentException("Span is empty.");

        float maxVal = float.MinValue;
        int maxIdx = 0;

        fixed (float* ptr = &MemoryMarshal.GetReference(span))
        {
            for (int i = 0; i < span.Length; i++)
            {
                if (ptr[i] > maxVal)
                {
                    maxVal = ptr[i];
                    maxIdx = i;
                }
            }
        }

        return (maxVal, maxIdx);
    }

    /// <summary>
    /// SIMD-accelerated element-wise multiply-accumulate: result += a[i] * b[i].
    /// Fundamental operation for dot products and neural network inference.
    /// </summary>
    public static float SimdMulAdd(this ReadOnlySpan<float> a, ReadOnlySpan<float> b, float accumulator = 0f)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Spans must have the same length.");

        fixed (float* aPtr = &MemoryMarshal.GetReference(a))
        fixed (float* bPtr = &MemoryMarshal.GetReference(b))
        {
            return SimdMulAddInternal(aPtr, bPtr, a.Length, accumulator);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SimdMulAddInternal(float* a, float* b, int length, float accumulator)
    {
        if (Avx2.IsSupported && length >= 8)
        {
            var accVec = Vector256.Create(accumulator);
            int i = 0;
            int simdEnd = length & ~7;

            for (; i < simdEnd; i += 8)
            {
                var va = Avx.LoadVector256(a + i);
                var vb = Avx.LoadVector256(b + i);
                accVec = Avx.Add(accVec, Avx.Multiply(va, vb));
            }

            // Horizontal sum.
            var upper = Avx.ExtractVector128(accVec, 1);
            var lower = accVec.GetLower();
            var sum128 = Sse.Add(upper, lower);
            sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0x4E));
            sum128 = Sse.AddScalar(sum128, Sse.Shuffle(sum128, sum128, 0x55));
            accumulator = sum128.ToScalar();

            for (; i < length; i++)
                accumulator += a[i] * b[i];
            return accumulator;
        }

        for (int i = 0; i < length; i++)
            accumulator += a[i] * b[i];
        return accumulator;
    }

    #endregion

    #region Comparison Operations

    /// <summary>
    /// SIMD-accelerated equality comparison of two byte spans.
    /// </summary>
    public static bool SimdSequenceEqual(this ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        if (a.Length != b.Length)
            return false;
        if (a.Length == 0)
            return true;

        fixed (byte* aPtr = &MemoryMarshal.GetReference(a))
        fixed (byte* bPtr = &MemoryMarshal.GetReference(b))
        {
            return SimdBytesEqual(aPtr, bPtr, (nuint)a.Length);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool SimdBytesEqual(byte* a, byte* b, nuint length)
    {
        nuint i = 0;

        if (Avx2.IsSupported && length >= 32)
        {
            nuint simdEnd = length & ~(nuint)31;
            for (; i < simdEnd; i += 32)
            {
                var va = Avx.LoadVector256(a + i);
                var vb = Avx.LoadVector256(b + i);
                var cmp = Avx2.CompareEqual(va, vb);
                if (Avx2.MoveMask(cmp) != unchecked((int)0xFFFFFFFF))
                    return false;
            }
        }
        else if (Sse2.IsSupported && length >= 16)
        {
            nuint simdEnd = length & ~(nuint)15;
            for (; i < simdEnd; i += 16)
            {
                var va = Sse2.LoadVector128(a + i);
                var vb = Sse2.LoadVector128(b + i);
                var cmp = Sse2.CompareEqual(va, vb);
                if (Sse2.MoveMask(cmp) != 0xFFFF)
                    return false;
            }
        }

        for (; i < length; i++)
        {
            if (a[i] != b[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Compares two float spans element-wise within a tolerance.
    /// </summary>
    public static bool SimdApproxEqual(this ReadOnlySpan<float> a, ReadOnlySpan<float> b, float tolerance = 1e-6f)
    {
        if (a.Length != b.Length)
            return false;

        for (int i = 0; i < a.Length; i++)
        {
            if (MathF.Abs(a[i] - b[i]) > tolerance)
                return false;
        }

        return true;
    }

    #endregion

    #region Conversion Utilities

    /// <summary>
    /// Converts a span of floats to a span of half-precision floats.
    /// Useful for GPU upload where half-precision saves bandwidth.
    /// </summary>
    public static Span<Half> ToHalfPrecision(this ReadOnlySpan<float> source, Span<Half> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Source and destination must have the same length.");

        for (int i = 0; i < source.Length; i++)
            destination[i] = (Half)source[i];

        return destination;
    }

    /// <summary>
    /// Converts a span of half-precision floats to a span of single-precision floats.
    /// </summary>
    public static Span<float> ToSinglePrecision(this ReadOnlySpan<Half> source, Span<float> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Source and destination must have the same length.");

        for (int i = 0; i < source.Length; i++)
            destination[i] = (float)source[i];

        return destination;
    }

    /// <summary>
    /// Clamps all float values in the span to the specified range.
    /// </summary>
    public static void SimdClamp(this Span<float> span, float min, float max)
    {
        fixed (float* ptr = &MemoryMarshal.GetReference(span))
        {
            if (Avx2.IsSupported && span.Length >= 8)
            {
                var minVec = Vector256.Create(min);
                var maxVec = Vector256.Create(max);
                int i = 0;
                int simdEnd = span.Length & ~7;

                for (; i < simdEnd; i += 8)
                {
                    var v = Avx.LoadVector256(ptr + i);
                    v = Avx.Max(v, minVec);
                    v = Avx.Min(v, maxVec);
                    Avx.Store(ptr + i, v);
                }

                for (; i < span.Length; i++)
                    ptr[i] = MathF.Max(min, MathF.Min(max, ptr[i]));
            }
            else
            {
                for (int i = 0; i < span.Length; i++)
                    ptr[i] = MathF.Max(min, MathF.Min(max, ptr[i]));
            }
        }
    }

    /// <summary>
    /// Scales all float values in the span by the given factor.
    /// </summary>
    public static void SimdScale(this Span<float> span, float factor)
    {
        fixed (float* ptr = &MemoryMarshal.GetReference(span))
        {
            if (Avx2.IsSupported && span.Length >= 8)
            {
                var fVec = Vector256.Create(factor);
                int i = 0;
                int simdEnd = span.Length & ~7;

                for (; i < simdEnd; i += 8)
                {
                    var v = Avx.LoadVector256(ptr + i);
                    Avx.Store(ptr + i, Avx.Multiply(v, fVec));
                }

                for (; i < span.Length; i++)
                    ptr[i] *= factor;
            }
            else
            {
                for (int i = 0; i < span.Length; i++)
                    ptr[i] *= factor;
            }
        }
    }

    /// <summary>
    /// Adds a scalar offset to all float values in the span.
    /// </summary>
    public static void SimdAdd(this Span<float> span, float offset)
    {
        fixed (float* ptr = &MemoryMarshal.GetReference(span))
        {
            if (Avx2.IsSupported && span.Length >= 8)
            {
                var oVec = Vector256.Create(offset);
                int i = 0;
                int simdEnd = span.Length & ~7;

                for (; i < simdEnd; i += 8)
                {
                    var v = Avx.LoadVector256(ptr + i);
                    Avx.Store(ptr + i, Avx.Add(v, oVec));
                }

                for (; i < span.Length; i++)
                    ptr[i] += offset;
            }
            else
            {
                for (int i = 0; i < span.Length; i++)
                    ptr[i] += offset;
            }
        }
    }

    #endregion
}
