using System;
// ============================================================
// FILE: HashUtils.cs
// PATH: Utilities/HashUtils.cs
// ============================================================

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Security.Cryptography;
using System.Text;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Utilities;

/// <summary>
/// Fast hash functions, hash combining utilities, content-addressable storage keys,
/// and checksum computation for the G-DNN neural geometry engine.
/// </summary>
public static class HashUtils
{
    /// <summary>
    /// Computes FNV-1a 32-bit hash of a byte span.
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <returns>32-bit FNV-1a hash.</returns>
    public static uint Fnv1a32(ReadOnlySpan<byte> data)
    {
        const uint FnvOffset = 2166136261u;
        const uint FnvPrime = 16777619u;

        uint hash = FnvOffset;
        foreach (byte b in data)
        {
            hash ^= b;
            hash *= FnvPrime;
        }
        return hash;
    }

    /// <summary>
    /// Computes FNV-1a 64-bit hash of a byte span.
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <returns>64-bit FNV-1a hash.</returns>
    public static ulong Fnv1a64(ReadOnlySpan<byte> data)
    {
        const ulong FnvOffset = 14695981039346656037ul;
        const ulong FnvPrime = 1099511628211ul;

        ulong hash = FnvOffset;
        foreach (byte b in data)
        {
            hash ^= b;
            hash *= FnvPrime;
        }
        return hash;
    }

    /// <summary>
    /// Computes FNV-1a 32-bit hash of a string.
    /// </summary>
    public static uint Fnv1a32(string text) =>
        Fnv1a32(Encoding.UTF8.GetBytes(text));

    /// <summary>
    /// Computes FNV-1a 64-bit hash of a string.
    /// </summary>
    public static ulong Fnv1a64(string text) =>
        Fnv1a64(Encoding.UTF8.GetBytes(text));

    /// <summary>
    /// Computes xxHash32 of a byte span.
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <param name="seed">Seed value (default 0).</param>
    /// <returns>32-bit xxHash.</returns>
    public static uint XxHash32(ReadOnlySpan<byte> data, uint seed = 0)
    {
        const uint Prime1 = 2654435761u;
        const uint Prime2 = 2246822519u;
        const uint Prime3 = 3266489917u;
        const uint Prime4 = 668265263u;
        const uint Prime5 = 374761393u;

        uint h32;
        int index = 0;
        int length = data.Length;

        if (length >= 16)
        {
            uint v1 = seed + Prime1 + Prime2;
            uint v2 = seed + Prime2;
            uint v3 = seed;
            uint v4 = seed - Prime1;

            for (int limit = length - 16; index <= limit; index += 16)
            {
                uint k1 = ReadUInt32LittleEndian(data, index);
                uint k2 = ReadUInt32LittleEndian(data, index + 4);
                uint k3 = ReadUInt32LittleEndian(data, index + 8);
                uint k4 = ReadUInt32LittleEndian(data, index + 12);

                v1 = Round(v1, k1);
                v2 = Round(v2, k2);
                v3 = Round(v3, k3);
                v4 = Round(v4, k4);
            }

            h32 = BitOperations.RotateLeft(v1, 1) + BitOperations.RotateLeft(v2, 7) +
                   BitOperations.RotateLeft(v3, 12) + BitOperations.RotateLeft(v4, 18);
        }
        else
        {
            h32 = seed + Prime5;
        }

        h32 += (uint)length;

        for (; index + 4 <= length; index += 4)
        {
            h32 += ReadUInt32LittleEndian(data, index) * Prime3;
            h32 = BitOperations.RotateLeft(h32, 17) * Prime4;
            h32 ^= h32 >> 15;
            h32 *= Prime2;
            h32 ^= h32 >> 13;
            h32 *= Prime3;
            h32 ^= h32 >> 16;
        }

        for (; index < length; index++)
        {
            h32 += data[index] * Prime5;
            h32 = BitOperations.RotateLeft(h32, 11) * Prime1;
            h32 ^= h32 >> 15;
        }

        h32 ^= h32 >> 16;
        h32 *= Prime2;
        h32 ^= h32 >> 13;
        h32 *= Prime3;
        h32 ^= h32 >> 16;

        return h32;

        static uint ReadUInt32LittleEndian(ReadOnlySpan<byte> span, int offset)
        {
            return BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(offset));
        }

        static uint Round(uint acc, uint input)
        {
            acc += input * Prime2;
            acc = BitOperations.RotateLeft(acc, 13);
            acc *= Prime1;
            return acc;
        }
    }

    /// <summary>
    /// Computes xxHash32 of a string.
    /// </summary>
    public static uint XxHash32(string text, uint seed = 0) =>
        XxHash32(Encoding.UTF8.GetBytes(text), seed);

    /// <summary>
    /// Computes xxHash64 of a byte span.
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <param name="seed">Seed value (default 0).</param>
    /// <returns>64-bit xxHash.</returns>
    public static ulong XxHash64(ReadOnlySpan<byte> data, ulong seed = 0)
    {
        const ulong Prime1 = 11400714785074694791ul;
        const ulong Prime2 = 14029467366897019727ul;
        const ulong Prime3 = 16095879293928697453ul;
        const ulong Prime4 = 1099511628211ul;
        const ulong Prime5 = 14695981039346656037ul;

        ulong h64;
        int index = 0;
        int length = data.Length;

        if (length >= 32)
        {
            ulong v1 = seed + Prime1 + Prime2;
            ulong v2 = seed + Prime2;
            ulong v3 = seed;
            ulong v4 = seed - Prime1;

            for (int limit = length - 32; index <= limit; index += 32)
            {
                ulong k1 = ReadUInt64LittleEndian(data, index);
                ulong k2 = ReadUInt64LittleEndian(data, index + 8);
                ulong k3 = ReadUInt64LittleEndian(data, index + 16);
                ulong k4 = ReadUInt64LittleEndian(data, index + 24);

                v1 = Round64(v1, k1);
                v2 = Round64(v2, k2);
                v3 = Round64(v3, k3);
                v4 = Round64(v4, k4);
            }

            h64 = BitOperations.RotateLeft(v1, 1) + BitOperations.RotateLeft(v2, 7) +
                   BitOperations.RotateLeft(v3, 12) + BitOperations.RotateLeft(v4, 18);

            h64 = MergeRound64(h64, v1);
            h64 = MergeRound64(h64, v2);
            h64 = MergeRound64(h64, v3);
            h64 = MergeRound64(h64, v4);
        }
        else
        {
            h64 = seed + Prime5;
        }

        h64 += (ulong)length;

        for (; index + 8 <= length; index += 8)
        {
            ulong k1 = ReadUInt64LittleEndian(data, index) * Prime2;
            k1 = BitOperations.RotateLeft(k1, 31);
            k1 *= Prime1;
            h64 ^= k1;
            h64 = BitOperations.RotateLeft(h64, 27) * Prime1 + Prime4;
        }

        for (; index + 4 <= length; index += 4)
        {
            h64 ^= ReadUInt32LittleEndian(data, index) * Prime1;
            h64 = BitOperations.RotateLeft(h64, 23) * Prime2 + Prime3;
        }

        for (; index < length; index++)
        {
            h64 ^= data[index] * Prime5;
            h64 = BitOperations.RotateLeft(h64, 11) * Prime1;
        }

        h64 ^= h64 >> 33;
        h64 *= Prime2;
        h64 ^= h64 >> 29;
        h64 *= Prime3;
        h64 ^= h64 >> 32;

        return h64;

        static ulong ReadUInt64LittleEndian(ReadOnlySpan<byte> span, int offset)
        {
            return BinaryPrimitives.ReadUInt64LittleEndian(span.Slice(offset));
        }

        static uint ReadUInt32LittleEndian(ReadOnlySpan<byte> span, int offset)
        {
            return BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(offset));
        }

        static ulong Round64(ulong acc, ulong input)
        {
            acc += input * Prime2;
            acc = BitOperations.RotateLeft(acc, 31);
            acc *= Prime1;
            return acc;
        }

        static ulong MergeRound64(ulong acc, ulong val)
        {
            val = Round64(0, val);
            acc ^= val;
            acc = acc * Prime1 + Prime4;
            return acc;
        }
    }

    /// <summary>
    /// Computes xxHash64 of a string.
    /// </summary>
    public static ulong XxHash64(string text, ulong seed = 0) =>
        XxHash64(Encoding.UTF8.GetBytes(text), seed);

    /// <summary>
    /// Computes MurmurHash3 32-bit of a byte span.
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <param name="seed">Seed value (default 0).</param>
    /// <returns>32-bit MurmurHash3.</returns>
    public static uint MurmurHash3_32(ReadOnlySpan<byte> data, uint seed = 0)
    {
        const uint C1 = 0xcc9e2d51;
        const uint C2 = 0x1b873593;

        uint h1 = seed;
        int length = data.Length;
        int index = 0;

        for (int limit = length - 4; index <= limit; index += 4)
        {
            uint k1 = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(index));
            k1 *= C1;
            k1 = BitOperations.RotateLeft(k1, 15);
            k1 *= C2;
            h1 ^= k1;
            h1 = BitOperations.RotateLeft(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
        }

        uint k1Remaining = 0;
        int remaining = length - index;
        switch (remaining)
        {
            case 3:
                k1Remaining |= (uint)data[index + 2] << 16;
                goto case 2;
            case 2:
                k1Remaining |= (uint)data[index + 1] << 8;
                goto case 1;
            case 1:
                k1Remaining |= data[index];
                k1Remaining *= C1;
                k1Remaining = BitOperations.RotateLeft(k1Remaining, 15);
                k1Remaining *= C2;
                h1 ^= k1Remaining;
                break;
        }

        h1 ^= (uint)length;
        h1 ^= h1 >> 16;
        h1 *= 0x85ebca6b;
        h1 ^= h1 >> 13;
        h1 *= 0xc2b2ae35;
        h1 ^= h1 >> 16;

        return h1;
    }

    /// <summary>
    /// Computes MurmurHash3 32-bit of a string.
    /// </summary>
    public static uint MurmurHash3_32(string text, uint seed = 0) =>
        MurmurHash3_32(Encoding.UTF8.GetBytes(text), seed);

    /// <summary>
    /// Computes MurmurHash3 128-bit of a byte span (returns full 128-bit hash as two ulongs).
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <param name="seed">Seed value (default 0).</param>
    /// <returns>128-bit hash as (h1, h2).</returns>
    public static (ulong H1, ulong H2) MurmurHash3_128(ReadOnlySpan<byte> data, ulong seed = 0)
    {
        const ulong C1 = 0x87c37b91114253d5;
        const ulong C2 = 0x4cf5ad432745937f;

        ulong h1 = seed;
        ulong h2 = seed;
        int length = data.Length;
        int index = 0;

        for (int limit = length - 15; index <= limit; index += 16)
        {
            ulong blockK1 = BinaryPrimitives.ReadUInt64LittleEndian(data.Slice(index));
            ulong blockK2 = BinaryPrimitives.ReadUInt64LittleEndian(data.Slice(index + 8));

            blockK1 *= C1;
            blockK1 = BitOperations.RotateLeft(blockK1, 31);
            blockK1 *= C2;
            h1 ^= blockK1;
            h1 = BitOperations.RotateLeft(h1, 27);
            h1 += h2;
            h1 = h1 * 5 + 0x52dce729;

            blockK2 *= C2;
            blockK2 = BitOperations.RotateLeft(blockK2, 33);
            blockK2 *= C1;
            h2 ^= blockK2;
            h2 = BitOperations.RotateLeft(h2, 31);
            h2 += h1;
            h2 = h2 * 5 + 0x38495ab5;
        }

        ulong k1 = 0, k2 = 0;
        int remaining = length - index;

        if (remaining >= 8)
        {
            k1 = BinaryPrimitives.ReadUInt64LittleEndian(data.Slice(index));
            k1 *= C1;
            k1 = BitOperations.RotateLeft(k1, 31);
            k1 *= C2;
            h1 ^= k1;
            index += 8;
            remaining -= 8;
        }

        if (remaining >= 4)
        {
            k2 = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(index));
            k2 *= C2;
            k2 = BitOperations.RotateLeft(k2, 33);
            k2 *= C1;
            h2 ^= k2;
            index += 4;
            remaining -= 4;
        }

        k1 = 0;
        k2 = 0;
        switch (remaining)
        {
            case 7:
                k1 |= (ulong)data[index + 6] << 48;
                goto case 6;
            case 6:
                k1 |= (ulong)data[index + 5] << 40;
                goto case 5;
            case 5:
                k1 |= (ulong)data[index + 4] << 32;
                goto case 4;
            case 4:
                k1 |= (ulong)data[index + 3] << 24;
                goto case 3;
            case 3:
                k1 |= (ulong)data[index + 2] << 16;
                goto case 2;
            case 2:
                k1 |= (ulong)data[index + 1] << 8;
                goto case 1;
            case 1:
                k1 |= data[index];
                k1 *= C1;
                k1 = BitOperations.RotateLeft(k1, 31);
                k1 *= C2;
                h1 ^= k1;
                break;
        }

        h1 ^= (ulong)length;
        h2 ^= (ulong)length;

        h1 += h2;
        h2 += h1;

        h1 = Finalize128(h1);
        h2 = Finalize128(h2);

        h1 += h2;
        h2 += h1;

        return (h1, h2);

        static ulong Finalize128(ulong hash)
        {
            hash ^= hash >> 33;
            hash *= 0xff51afd7ed558ccd;
            hash ^= hash >> 33;
            hash *= 0xc4ceb9fe1a85ec53;
            hash ^= hash >> 33;
            return hash;
        }
    }

    /// <summary>
    /// Combines two hash values using the Cantor pairing function.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint CombineCantor(uint hash1, uint hash2)
    {
        uint sum = hash1 + hash2;
        return (sum * (sum + 1)) / 2 + hash2;
    }

    /// <summary>
    /// Combines two 64-bit hash values.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong CombineCantor64(ulong hash1, ulong hash2)
    {
        ulong sum = hash1 + hash2;
        return (sum * (sum + 1)) / 2 + hash2;
    }

    /// <summary>
    /// Combines multiple hash values using XOR folding.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint CombineXor(params uint[] hashes)
    {
        uint result = 0;
        foreach (uint h in hashes)
        {
            result ^= h;
            result = BitOperations.RotateLeft(result, 13);
        }
        return result;
    }

    /// <summary>
    /// Combines multiple 64-bit hash values using XOR folding.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong CombineXor64(params ulong[] hashes)
    {
        ulong result = 0;
        foreach (ulong h in hashes)
        {
            result ^= h;
            result = BitOperations.RotateLeft(result, 17);
        }
        return result;
    }

    /// <summary>
    /// Combines two hash values using a multiplicative scheme.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint CombineMultiply(uint hash1, uint hash2)
    {
        const uint Magic = 0x9e3779b9;
        uint result = hash1;
        result ^= hash2 + Magic + (result << 6) + (result >> 2);
        return result;
    }

    /// <summary>
    /// Combines two 64-bit hash values using a multiplicative scheme.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong CombineMultiply64(ulong hash1, ulong hash2)
    {
        const ulong Magic = 0x9e3779b97f4a7c15;
        ulong result = hash1;
        result ^= hash2 + Magic + (result << 12) + (result >> 4);
        return result;
    }

    /// <summary>
    /// Computes a hash for a Vector3.
    /// </summary>
    public static uint HashVector3(System.Numerics.Vector3 v)
    {
        Span<byte> data = stackalloc byte[12];
        MemoryMarshal.Write(data, ref Unsafe.As<System.Numerics.Vector3, (float, float, float)>(ref Unsafe.AsRef(in v)));
        return XxHash32(data);
    }

    /// <summary>
    /// Computes a hash for a Vector4.
    /// </summary>
    public static uint HashVector4(System.Numerics.Vector4 v)
    {
        Span<byte> data = stackalloc byte[16];
        MemoryMarshal.Write(data, ref Unsafe.As<System.Numerics.Vector4, (float, float, float, float)>(ref Unsafe.AsRef(in v)));
        return XxHash32(data);
    }

    /// <summary>
    /// Computes a hash for a Quaternion.
    /// </summary>
    public static uint HashQuaternion(System.Numerics.Quaternion q)
    {
        Span<byte> data = stackalloc byte[16];
        MemoryMarshal.Write(data, ref Unsafe.As<System.Numerics.Quaternion, (float, float, float, float)>(ref Unsafe.AsRef(in q)));
        return XxHash32(data);
    }

    /// <summary>
    /// Computes a hash for a Matrix4x4.
    /// </summary>
    public static uint HashMatrix4x4(System.Numerics.Matrix4x4 m)
    {
        Span<byte> data = stackalloc byte[64];
        MemoryMarshal.Write(data, ref Unsafe.As<System.Numerics.Matrix4x4, (float, float, float, float, float, float, float, float,
            float, float, float, float, float, float, float, float)>(ref Unsafe.AsRef(in m)));
        return XxHash32(data);
    }

    /// <summary>
    /// Creates a content-addressable storage key from data.
    /// </summary>
    /// <param name="data">The data to create a key for.</param>
    /// <returns>A 256-bit key (32 bytes) using SHA-256.</returns>
    public static byte[] CreateCasKey(ReadOnlySpan<byte> data)
    {
        return SHA256.HashData(data);
    }

    /// <summary>
    /// Creates a content-addressable storage key and returns it as a hex string.
    /// </summary>
    public static string CreateCasKeyHex(ReadOnlySpan<byte> data)
    {
        byte[] hash = SHA256.HashData(data);
        return Convert.ToHexString(hash);
    }

    /// <summary>
    /// Creates a short CAS key (first 8 bytes of SHA-256) for fast lookups.
    /// </summary>
    public static ulong CreateShortCasKey(ReadOnlySpan<byte> data)
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(data, hash);
        return BinaryPrimitives.ReadUInt64LittleEndian(hash);
    }

    /// <summary>
    /// Computes CRC32 checksum of a byte span.
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <returns>32-bit CRC.</returns>
    public static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFF;

        foreach (byte b in data)
        {
            crc ^= b;
            for (int i = 0; i < 8; i++)
            {
                crc = (crc >> 1) ^ (0xEDB88320 & (uint)(-(crc & 1)));
            }
        }

        return crc ^ 0xFFFFFFFF;
    }

    /// <summary>
    /// Computes CRC32 of a string.
    /// </summary>
    public static uint Crc32(string text) => Crc32(Encoding.UTF8.GetBytes(text));

    /// <summary>
    /// Computes Adler-32 checksum of a byte span.
    /// </summary>
    /// <param name="data">Input data.</param>
    /// <returns>32-bit Adler checksum.</returns>
    public static uint Adler32(ReadOnlySpan<byte> data)
    {
        const uint Modulus = 65521;
        uint a = 1, b = 0;

        foreach (byte d in data)
        {
            a = (a + d) % Modulus;
            b = (b + a) % Modulus;
        }

        return (b << 16) | a;
    }

    /// <summary>
    /// Computes a fast 64-bit identity hash for an integer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong IdentityHash64(int value)
    {
        ulong v = (ulong)value;
        v ^= v >> 33;
        v *= 0xff51afd7ed558ccd;
        v ^= v >> 33;
        v *= 0xc4ceb9fe1a85ec53;
        v ^= v >> 33;
        return v;
    }

    /// <summary>
    /// Computes a spatial hash for a 3D point.
    /// </summary>
    /// <param name="x">X coordinate.</param>
    /// <param name="y">Y coordinate.</param>
    /// <param name="z">Z coordinate.</param>
    /// <returns>64-bit spatial hash.</returns>
    public static ulong SpatialHash3D(int x, int y, int z)
    {
        ulong h = IdentityHash64(x);
        h = CombineMultiply64(h, IdentityHash64(y));
        h = CombineMultiply64(h, IdentityHash64(z));
        return h;
    }

    /// <summary>
    /// Computes a spatial hash for a 3D point with quantized coordinates.
    /// </summary>
    /// <param name="position">World-space position.</param>
    /// <param name="cellSize">Size of each hash cell.</param>
    /// <returns>64-bit spatial hash.</returns>
    public static ulong SpatialHash3D(System.Numerics.Vector3 position, float cellSize)
    {
        int x = (int)Math.Floor(position.X / cellSize);
        int y = (int)Math.Floor(position.Y / cellSize);
        int z = (int)Math.Floor(position.Z / cellSize);
        return SpatialHash3D(x, y, z);
    }

    /// <summary>
    /// Computes a spatial hash for a 2D point.
    /// </summary>
    public static ulong SpatialHash2D(int x, int y)
    {
        ulong h = IdentityHash64(x);
        h = CombineMultiply64(h, IdentityHash64(y));
        return h;
    }

    /// <summary>
    /// Computes a spatial hash for a 2D point with quantized coordinates.
    /// </summary>
    public static ulong SpatialHash2D(System.Numerics.Vector2 position, float cellSize)
    {
        int x = (int)Math.Floor(position.X / cellSize);
        int y = (int)Math.Floor(position.Y / cellSize);
        return SpatialHash2D(x, y);
    }

    /// <summary>
    /// MurmurHash3 mixer function for finalizing a hash.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint MurmurMix32(uint h)
    {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }

    /// <summary>
    /// MurmurHash3 mixer function for finalizing a 64-bit hash.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong MurmurMix64(ulong h)
    {
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccd;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53;
        h ^= h >> 33;
        return h;
    }

    /// <summary>
    /// Computes a hash of a struct using bit-level mixing.
    /// </summary>
    public static uint FastStructHash<T>(in T value) where T : struct
    {
        ReadOnlySpan<byte> data = MemoryMarshal.CreateReadOnlySpan(
            ref Unsafe.As<T, byte>(ref Unsafe.AsRef(in value)),
            Unsafe.SizeOf<T>());
        return XxHash32(data);
    }

    /// <summary>
    /// Computes a hash of a span of values using streaming.
    /// </summary>
    public static uint StreamingHash(ReadOnlySpan<uint> hashes)
    {
        uint result = 2166136261;
        foreach (uint h in hashes)
        {
            result = CombineMultiply(result, h);
        }
        return result;
    }

    /// <summary>
    /// Computes a deterministic hash for a floating-point value,
    /// treating +0 and -0 as equal and NaN as a specific value.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint HashFloat(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (bits == 0x80000000)
            bits = 0;
        return MurmurMix32(bits);
    }

    /// <summary>
    /// Computes a deterministic hash for a double.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong HashDouble(double value)
    {
        ulong bits = BitConverter.DoubleToUInt64Bits(value);
        return MurmurMix64(bits);
    }

    /// <summary>
    /// Computes xxHash32 of an integer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint HashInt32(int value)
    {
        Span<byte> data = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(data, value);
        return XxHash32(data);
    }

    /// <summary>
    /// Computes xxHash64 of a long.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong HashInt64(long value)
    {
        Span<byte> data = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(data, value);
        return XxHash64(data);
    }

    /// <summary>
    /// Computes a combined hash for multiple vectors.
    /// </summary>
    public static uint HashVectorArray(ReadOnlySpan<System.Numerics.Vector3> vectors)
    {
        uint hash = 2166136261;
        foreach (ref readonly System.Numerics.Vector3 v in vectors)
        {
            hash = CombineMultiply(hash, HashVector3(v));
        }
        return hash;
    }

    /// <summary>
    /// Computes a hash of a byte array with salt.
    /// </summary>
    public static uint HashWithSalt(ReadOnlySpan<byte> data, uint salt)
    {
        uint hash = XxHash32(data);
        return CombineMultiply(hash, salt);
    }

    /// <summary>
    /// Computes a bucket index for a hash value.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetBucketIndex(ulong hash, int bucketCount) =>
        (int)(hash % (ulong)bucketCount);

    /// <summary>
    /// Computes a bucket index for a hash value using fast modulo.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetBucketIndexFast(ulong hash, int bucketCount)
    {
        // bucketCount must be a power of 2
        return (int)(hash & (ulong)(bucketCount - 1));
    }
}
