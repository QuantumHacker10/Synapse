using System;
// ============================================================
// FILE: BinaryReaderWriterExtensions.cs
// PATH: Utilities/BinaryReaderWriterExtensions.cs
// ============================================================

using System;
using System.Buffers;
using System.Buffers.Binary;
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
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Utilities;

/// <summary>
/// Extension methods for BinaryReader/BinaryWriter providing serialization
/// of math types, string pooling, varint encoding, and bit packing.
/// </summary>
public static class BinaryReaderWriterExtensions
{
    #region Vector2

    /// <summary>
    /// Writes a Vector2 to the binary stream.
    /// </summary>
    public static void Write(this BinaryWriter writer, Vector2 value)
    {
        writer.Write(value.X);
        writer.Write(value.Y);
    }

    /// <summary>
    /// Reads a Vector2 from the binary stream.
    /// </summary>
    public static Vector2 ReadVector2(this BinaryReader reader)
    {
        float x = reader.ReadSingle();
        float y = reader.ReadSingle();
        return new Vector2(x, y);
    }

    #endregion

    #region Vector3

    /// <summary>
    /// Writes a Vector3 to the binary stream.
    /// </summary>
    public static void Write(this BinaryWriter writer, Vector3 value)
    {
        writer.Write(value.X);
        writer.Write(value.Y);
        writer.Write(value.Z);
    }

    /// <summary>
    /// Reads a Vector3 from the binary stream.
    /// </summary>
    public static Vector3 ReadVector3(this BinaryReader reader)
    {
        float x = reader.ReadSingle();
        float y = reader.ReadSingle();
        float z = reader.ReadSingle();
        return new Vector3(x, y, z);
    }

    #endregion

    #region Vector4

    /// <summary>
    /// Writes a Vector4 to the binary stream.
    /// </summary>
    public static void Write(this BinaryWriter writer, Vector4 value)
    {
        writer.Write(value.X);
        writer.Write(value.Y);
        writer.Write(value.Z);
        writer.Write(value.W);
    }

    /// <summary>
    /// Reads a Vector4 from the binary stream.
    /// </summary>
    public static Vector4 ReadVector4(this BinaryReader reader)
    {
        float x = reader.ReadSingle();
        float y = reader.ReadSingle();
        float z = reader.ReadSingle();
        float w = reader.ReadSingle();
        return new Vector4(x, y, z, w);
    }

    #endregion

    #region Quaternion

    /// <summary>
    /// Writes a Quaternion to the binary stream.
    /// </summary>
    public static void Write(this BinaryWriter writer, Quaternion value)
    {
        writer.Write(value.X);
        writer.Write(value.Y);
        writer.Write(value.Z);
        writer.Write(value.W);
    }

    /// <summary>
    /// Reads a Quaternion from the binary stream.
    /// </summary>
    public static Quaternion ReadQuaternion(this BinaryReader reader)
    {
        float x = reader.ReadSingle();
        float y = reader.ReadSingle();
        float z = reader.ReadSingle();
        float w = reader.ReadSingle();
        return new Quaternion(x, y, z, w);
    }

    #endregion

    #region Matrix4x4

    /// <summary>
    /// Writes a Matrix4x4 to the binary stream (row-major).
    /// </summary>
    public static void Write(this BinaryWriter writer, Matrix4x4 value)
    {
        writer.Write(value.M11);
        writer.Write(value.M12);
        writer.Write(value.M13);
        writer.Write(value.M14);
        writer.Write(value.M21);
        writer.Write(value.M22);
        writer.Write(value.M23);
        writer.Write(value.M24);
        writer.Write(value.M31);
        writer.Write(value.M32);
        writer.Write(value.M33);
        writer.Write(value.M34);
        writer.Write(value.M41);
        writer.Write(value.M42);
        writer.Write(value.M43);
        writer.Write(value.M44);
    }

    /// <summary>
    /// Reads a Matrix4x4 from the binary stream (row-major).
    /// </summary>
    public static Matrix4x4 ReadMatrix4x4(this BinaryReader reader)
    {
        Matrix4x4 m;
        m.M11 = reader.ReadSingle();
        m.M12 = reader.ReadSingle();
        m.M13 = reader.ReadSingle();
        m.M14 = reader.ReadSingle();
        m.M21 = reader.ReadSingle();
        m.M22 = reader.ReadSingle();
        m.M23 = reader.ReadSingle();
        m.M24 = reader.ReadSingle();
        m.M31 = reader.ReadSingle();
        m.M32 = reader.ReadSingle();
        m.M33 = reader.ReadSingle();
        m.M34 = reader.ReadSingle();
        m.M41 = reader.ReadSingle();
        m.M42 = reader.ReadSingle();
        m.M43 = reader.ReadSingle();
        m.M44 = reader.ReadSingle();
        return m;
    }

    /// <summary>
    /// Writes a Matrix4x4 using half-precision floats (16-bit) to save space.
    /// </summary>
    public static void WriteHalf(this BinaryWriter writer, Matrix4x4 value)
    {
        WriteHalf(writer, value.M11);
        WriteHalf(writer, value.M12);
        WriteHalf(writer, value.M13);
        WriteHalf(writer, value.M14);
        WriteHalf(writer, value.M21);
        WriteHalf(writer, value.M22);
        WriteHalf(writer, value.M23);
        WriteHalf(writer, value.M24);
        WriteHalf(writer, value.M31);
        WriteHalf(writer, value.M32);
        WriteHalf(writer, value.M33);
        WriteHalf(writer, value.M34);
        WriteHalf(writer, value.M41);
        WriteHalf(writer, value.M42);
        WriteHalf(writer, value.M43);
        WriteHalf(writer, value.M44);
    }

    /// <summary>
    /// Reads a Matrix4x4 using half-precision floats (16-bit).
    /// </summary>
    public static Matrix4x4 ReadMatrix4x4Half(this BinaryReader reader)
    {
        Matrix4x4 m;
        m.M11 = ReadHalf(reader);
        m.M12 = ReadHalf(reader);
        m.M13 = ReadHalf(reader);
        m.M14 = ReadHalf(reader);
        m.M21 = ReadHalf(reader);
        m.M22 = ReadHalf(reader);
        m.M23 = ReadHalf(reader);
        m.M24 = ReadHalf(reader);
        m.M31 = ReadHalf(reader);
        m.M32 = ReadHalf(reader);
        m.M33 = ReadHalf(reader);
        m.M34 = ReadHalf(reader);
        m.M41 = ReadHalf(reader);
        m.M42 = ReadHalf(reader);
        m.M43 = ReadHalf(reader);
        m.M44 = ReadHalf(reader);
        return m;
    }

    #endregion

    #region Half-precision Float

    /// <summary>
    /// Writes a half-precision (16-bit) float to the binary stream.
    /// </summary>
    public static void WriteHalf(this BinaryWriter writer, float value)
    {
        ushort half = FloatToHalf(value);
        writer.Write(half);
    }

    /// <summary>
    /// Reads a half-precision (16-bit) float from the binary stream.
    /// </summary>
    public static float ReadHalf(this BinaryReader reader)
    {
        ushort half = reader.ReadUInt16();
        return HalfToFloat(half);
    }

    /// <summary>
    /// Converts a float to half-precision (IEEE 754 binary16).
    /// </summary>
    public static ushort FloatToHalf(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint sign = (bits >> 16) & 0x8000;
        int exponent = (int)((bits >> 23) & 0xFF) - 127 + 15;
        uint mantissa = bits & 0x7FFFFF;

        if (exponent <= 0)
        {
            if (exponent < -10)
                return (ushort)sign;
            mantissa = (mantissa | 0x800000) >> (1 - exponent);
            return (ushort)(sign | (mantissa >> 13));
        }
        else if (exponent == 0xFF - 127 + 15)
        {
            return (ushort)(sign | 0x7C00 | (mantissa != 0 ? (mantissa >> 13) | 1 : 0));
        }
        else if (exponent > 30)
        {
            return (ushort)(sign | 0x7C00);
        }

        return (ushort)(sign | ((uint)exponent << 10) | (mantissa >> 13));
    }

    /// <summary>
    /// Converts a half-precision float (16-bit) to a single-precision float (32-bit).
    /// </summary>
    public static float HalfToFloat(ushort value)
    {
        uint sign = (uint)(value & 0x8000) << 16;
        int exponent = (value >> 10) & 0x1F;
        uint mantissa = (uint)(value & 0x3FF);

        if (exponent == 0)
        {
            if (mantissa == 0)
                return BitConverter.UInt32BitsToSingle(sign);
            exponent = 1;
            while ((mantissa & 0x400) == 0)
            {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
        }
        else if (exponent == 31)
        {
            return BitConverter.UInt32BitsToSingle(sign | 0x7F800000 | (mantissa << 13));
        }

        uint result = sign | (uint)((exponent - 15 + 127) << 23) | (mantissa << 13);
        return BitConverter.UInt32BitsToSingle(result);
    }

    #endregion

    #region Varint Encoding

    /// <summary>
    /// Writes a variable-length encoded unsigned 32-bit integer.
    /// Uses LEB128-style encoding (7 bits per byte, high bit for continuation).
    /// </summary>
    public static void WriteVaruint32(this BinaryWriter writer, uint value)
    {
        while (value > 0x7F)
        {
            writer.Write((byte)(value | 0x80));
            value >>= 7;
        }
        writer.Write((byte)value);
    }

    /// <summary>
    /// Reads a variable-length encoded unsigned 32-bit integer.
    /// </summary>
    public static uint ReadVaruint32(this BinaryReader reader)
    {
        uint result = 0;
        int shift = 0;
        byte b;
        do
        {
            b = reader.ReadByte();
            result |= (uint)(b & 0x7F) << shift;
            shift += 7;
        } while ((b & 0x80) != 0);
        return result;
    }

    /// <summary>
    /// Writes a variable-length encoded signed 32-bit integer (zigzag encoded).
    /// </summary>
    public static void WriteVarint32(this BinaryWriter writer, int value)
    {
        uint encoded = (uint)((value << 1) ^ (value >> 31));
        WriteVaruint32(writer, encoded);
    }

    /// <summary>
    /// Reads a variable-length encoded signed 32-bit integer (zigzag decoded).
    /// </summary>
    public static int ReadVarint32(this BinaryReader reader)
    {
        uint encoded = ReadVaruint32(reader);
        return (int)((encoded >> 1) ^ -(encoded & 1));
    }

    /// <summary>
    /// Writes a variable-length encoded unsigned 64-bit integer.
    /// </summary>
    public static void WriteVaruint64(this BinaryWriter writer, ulong value)
    {
        while (value > 0x7F)
        {
            writer.Write((byte)(value | 0x80));
            value >>= 7;
        }
        writer.Write((byte)value);
    }

    /// <summary>
    /// Reads a variable-length encoded unsigned 64-bit integer.
    /// </summary>
    public static ulong ReadVaruint64(this BinaryReader reader)
    {
        ulong result = 0;
        int shift = 0;
        byte b;
        do
        {
            b = reader.ReadByte();
            result |= (ulong)(b & 0x7F) << shift;
            shift += 7;
        } while ((b & 0x80) != 0);
        return result;
    }

    /// <summary>
    /// Writes a variable-length encoded signed 64-bit integer (zigzag encoded).
    /// </summary>
    public static void WriteVarint64(this BinaryWriter writer, long value)
    {
        ulong encoded = (ulong)((value << 1) ^ (value >> 63));
        WriteVaruint64(writer, encoded);
    }

    /// <summary>
    /// Reads a variable-length encoded signed 64-bit integer (zigzag decoded).
    /// </summary>
    public static long ReadVarint64(this BinaryReader reader)
    {
        ulong encoded = ReadVaruint64(reader);
        return (long)(encoded >> 1) ^ -(long)(encoded & 1);
    }

    /// <summary>
    /// Writes a variable-length encoded float.
    /// </summary>
    public static void WriteVarfloat(this BinaryWriter writer, float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        WriteVaruint32(writer, bits);
    }

    /// <summary>
    /// Reads a variable-length encoded float.
    /// </summary>
    public static float ReadVarfloat(this BinaryReader reader)
    {
        uint bits = ReadVaruint32(reader);
        return BitConverter.UInt32BitsToSingle(bits);
    }

    #endregion

    #region Bit Packing

    /// <summary>
    /// A bit writer for packing multiple values into bytes.
    /// </summary>
    public ref struct BitWriter
    {
        private readonly BinaryWriter _writer;
        private byte _currentByte;
        private int _bitCount;

        public BitWriter(BinaryWriter writer)
        {
            _writer = writer;
            _currentByte = 0;
            _bitCount = 0;
        }

        /// <summary>
        /// Writes a single bit.
        /// </summary>
        public void WriteBit(bool value)
        {
            if (value)
                _currentByte |= (byte)(1 << _bitCount);

            _bitCount++;
            if (_bitCount == 8)
            {
                Flush();
            }
        }

        /// <summary>
        /// Writes a single bit (0 or 1).
        /// </summary>
        public void WriteBit(int value) => WriteBit(value != 0);

        /// <summary>
        /// Writes multiple bits from an integer value.
        /// </summary>
        /// <param name="value">Value to write.</param>
        /// <param name="bitCount">Number of bits to write (1-32).</param>
        public void WriteBits(uint value, int bitCount)
        {
            for (int i = 0; i < bitCount; i++)
            {
                WriteBit(((value >> i) & 1) != 0);
            }
        }

        /// <summary>
        /// Writes a fixed number of bits from a uint.
        /// </summary>
        public void WriteBits(uint value, uint bitMask)
        {
            int bitCount = BitOperations.PopCount(bitMask);
            WriteBits(value, bitCount);
        }

        /// <summary>
        /// Flushes any remaining bits to the writer.
        /// </summary>
        public void Flush()
        {
            if (_bitCount > 0)
            {
                _writer.Write(_currentByte);
                _currentByte = 0;
                _bitCount = 0;
            }
        }

        /// <summary>
        /// Gets the number of bits written in the current byte.
        /// </summary>
        public int BitsInCurrentByte => _bitCount;

        /// <summary>
        /// Gets the total number of bits written.
        /// </summary>
        public long TotalBitsWritten { get; private set; }

        /// <summary>
        /// Gets the total number of bytes written.
        /// </summary>
        public long TotalBytesWritten => TotalBitsWritten / 8 + (_bitCount > 0 ? 1 : 0);
    }

    /// <summary>
    /// A bit reader for unpacking multiple values from bytes.
    /// </summary>
    public ref struct BitReader
    {
        private readonly BinaryReader _reader;
        private byte _currentByte;
        private int _bitsRemaining;

        public BitReader(BinaryReader reader)
        {
            _reader = reader;
            _currentByte = 0;
            _bitsRemaining = 0;
        }

        /// <summary>
        /// Reads a single bit.
        /// </summary>
        public bool ReadBit()
        {
            if (_bitsRemaining == 0)
            {
                _currentByte = _reader.ReadByte();
                _bitsRemaining = 8;
            }

            bool value = (_currentByte & (1 << (7 - _bitsRemaining + 1))) != 0;
            _bitsRemaining--;
            return value;
        }

        /// <summary>
        /// Reads a specified number of bits as a uint.
        /// </summary>
        public uint ReadBits(int bitCount)
        {
            uint result = 0;
            for (int i = 0; i < bitCount; i++)
            {
                if (ReadBit())
                    result |= 1u << i;
            }
            return result;
        }

        /// <summary>
        /// Skips to the next byte boundary.
        /// </summary>
        public void SkipToByteBoundary()
        {
            _bitsRemaining = 0;
        }
    }

    #endregion

    #region String Pool

    /// <summary>
    /// Writes a string using a string pool. Previously written strings are
    /// written as their pool index (varint), new strings are written with
    /// a length prefix followed by the string data.
    /// </summary>
    public static void WritePooled(this BinaryWriter writer, string value, Dictionary<string, int> pool)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        ArgumentNullException.ThrowIfNull(pool);

        if (pool.TryGetValue(value, out int index))
        {
            writer.WriteVaruint32((uint)((index << 1) | 1));
        }
        else
        {
            int newIndex = pool.Count;
            pool[value] = newIndex;
            writer.WriteVaruint32((uint)(newIndex << 1));
            writer.Write(value);
        }
    }

    /// <summary>
    /// Reads a string using a string pool.
    /// </summary>
    public static string ReadPooled(this BinaryReader reader, List<string> pool)
    {
        ArgumentNullException.ThrowIfNull(pool);

        uint encoded = reader.ReadVaruint32();
        bool isExisting = (encoded & 1) == 1;
        int index = (int)(encoded >> 1);

        if (isExisting)
        {
            if (index >= pool.Count)
                throw new InvalidDataException($"String pool index {index} out of range (pool size: {pool.Count}).");
            return pool[index];
        }

        string value = reader.ReadString();
        pool.Add(value);
        return value;
    }

    /// <summary>
    /// Writes a string using pooled encoding with a maximum length check.
    /// </summary>
    public static void WritePooledLimited(this BinaryWriter writer, string value,
        Dictionary<string, int> pool, int maxLength = 4096)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        if (value.Length > maxLength)
            throw new ArgumentException($"String length {value.Length} exceeds maximum {maxLength}.");

        WritePooled(writer, value, pool);
    }

    #endregion

    #region Array/Collection Helpers

    /// <summary>
    /// Writes a span of bytes with a length prefix.
    /// </summary>
    public static void WriteSpan(this BinaryWriter writer, ReadOnlySpan<byte> data)
    {
        writer.WriteVarint32(data.Length);
        writer.Write(data);
    }

    /// <summary>
    /// Reads a byte span with a length prefix.
    /// </summary>
    public static byte[] ReadSpan(this BinaryReader reader)
    {
        int length = reader.ReadInt32();
        return reader.ReadBytes(length);
    }

    /// <summary>
    /// Writes a span of Vector3 values.
    /// </summary>
    public static void WriteVector3Array(this BinaryWriter writer, ReadOnlySpan<Vector3> vectors)
    {
        writer.WriteVarint32(vectors.Length);
        foreach (ref readonly Vector3 v in vectors)
        {
            writer.Write(v);
        }
    }

    /// <summary>
    /// Reads an array of Vector3 values.
    /// </summary>
    public static Vector3[] ReadVector3Array(this BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var vectors = new Vector3[count];
        for (int i = 0; i < count; i++)
        {
            vectors[i] = reader.ReadVector3();
        }
        return vectors;
    }

    /// <summary>
    /// Writes a span of Vector4 values.
    /// </summary>
    public static void WriteVector4Array(this BinaryWriter writer, ReadOnlySpan<Vector4> vectors)
    {
        writer.WriteVarint32(vectors.Length);
        foreach (ref readonly Vector4 v in vectors)
        {
            writer.Write(v);
        }
    }

    /// <summary>
    /// Reads an array of Vector4 values.
    /// </summary>
    public static Vector4[] ReadVector4Array(this BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var vectors = new Vector4[count];
        for (int i = 0; i < count; i++)
        {
            vectors[i] = reader.ReadVector4();
        }
        return vectors;
    }

    /// <summary>
    /// Writes a span of float values.
    /// </summary>
    public static void WriteFloatArray(this BinaryWriter writer, ReadOnlySpan<float> values)
    {
        writer.WriteVarint32(values.Length);
        foreach (float v in values)
        {
            writer.Write(v);
        }
    }

    /// <summary>
    /// Reads an array of float values.
    /// </summary>
    public static float[] ReadFloatArray(this BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = reader.ReadSingle();
        }
        return values;
    }

    /// <summary>
    /// Writes a span of uint values.
    /// </summary>
    public static void WriteUInt32Array(this BinaryWriter writer, ReadOnlySpan<uint> values)
    {
        writer.WriteVarint32(values.Length);
        foreach (uint v in values)
        {
            writer.Write(v);
        }
    }

    /// <summary>
    /// Reads an array of uint values.
    /// </summary>
    public static uint[] ReadUInt32Array(this BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var values = new uint[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = reader.ReadUInt32();
        }
        return values;
    }

    /// <summary>
    /// Writes a span of int values.
    /// </summary>
    public static void WriteInt32Array(this BinaryWriter writer, ReadOnlySpan<int> values)
    {
        writer.WriteVarint32(values.Length);
        foreach (int v in values)
        {
            writer.Write(v);
        }
    }

    /// <summary>
    /// Reads an array of int values.
    /// </summary>
    public static int[] ReadInt32Array(this BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var values = new int[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = reader.ReadInt32();
        }
        return values;
    }

    /// <summary>
    /// Writes a span of Quaternion values.
    /// </summary>
    public static void WriteQuaternionArray(this BinaryWriter writer, ReadOnlySpan<Quaternion> quaternions)
    {
        writer.WriteVarint32(quaternions.Length);
        foreach (ref readonly Quaternion q in quaternions)
        {
            writer.Write(q);
        }
    }

    /// <summary>
    /// Reads an array of Quaternion values.
    /// </summary>
    public static Quaternion[] ReadQuaternionArray(this BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var quaternions = new Quaternion[count];
        for (int i = 0; i < count; i++)
        {
            quaternions[i] = reader.ReadQuaternion();
        }
        return quaternions;
    }

    /// <summary>
    /// Writes a span of Matrix4x4 values.
    /// </summary>
    public static void WriteMatrix4x4Array(this BinaryWriter writer, ReadOnlySpan<Matrix4x4> matrices)
    {
        writer.WriteVarint32(matrices.Length);
        foreach (ref readonly Matrix4x4 m in matrices)
        {
            writer.Write(m);
        }
    }

    /// <summary>
    /// Reads an array of Matrix4x4 values.
    /// </summary>
    public static Matrix4x4[] ReadMatrix4x4Array(this BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var matrices = new Matrix4x4[count];
        for (int i = 0; i < count; i++)
        {
            matrices[i] = reader.ReadMatrix4x4();
        }
        return matrices;
    }

    #endregion

    #region Compressed Writes

    /// <summary>
    /// Writes a normalized quaternion using 32 bits (smallest-three compression).
    /// </summary>
    public static void WriteQuaternionCompressed(this BinaryWriter writer, Quaternion q)
    {
        float absX = MathF.Abs(q.X), absY = MathF.Abs(q.Y);
        float absZ = MathF.Abs(q.Z), absW = MathF.Abs(q.W);

        byte index = 0;
        float maxVal = absX;
        if (absY > maxVal)
        { index = 1; maxVal = absY; }
        if (absZ > maxVal)
        { index = 2; maxVal = absZ; }
        if (absW > maxVal)
        { index = 3; }

        // Ensure the largest component is positive for unique representation
        if (((index == 0 ? q.X : index == 1 ? q.Y : index == 2 ? q.Z : q.W) < 0))
        {
            q.X = -q.X;
            q.Y = -q.Y;
            q.Z = -q.Z;
            q.W = -q.W;
        }

        Span<float> components = stackalloc float[3];
        int compIdx = 0;
        if (index != 0)
            components[compIdx++] = q.X;
        if (index != 1)
            components[compIdx++] = q.Y;
        if (index != 2)
            components[compIdx++] = q.Z;
        if (index != 3)
            components[compIdx++] = q.W;

        writer.Write(index);
        for (int i = 0; i < 3; i++)
        {
            ushort encoded = (ushort)Math.Clamp((int)((components[i] + 1f) * 0.5f * 65535f), 0, 65535);
            writer.Write(encoded);
        }
    }

    /// <summary>
    /// Reads a compressed quaternion (smallest-three, 32-bit index + 3x16-bit components).
    /// </summary>
    public static Quaternion ReadQuaternionCompressed(this BinaryReader reader)
    {
        byte index = reader.ReadByte();
        Span<float> components = stackalloc float[3];
        for (int i = 0; i < 3; i++)
        {
            ushort encoded = reader.ReadUInt16();
            components[i] = encoded / 65535f * 2f - 1f;
        }

        // Reconstruct the full quaternion
        float x = 0, y = 0, z = 0, w = 0;
        int compIdx = 0;
        if (index != 0)
            x = components[compIdx++];
        if (index != 1)
            y = components[compIdx++];
        if (index != 2)
            z = components[compIdx++];
        if (index != 3)
            w = components[compIdx++];

        // Compute the missing component
        float sum = 1f - (x * x + y * y + z * z + w * w);
        float missing = MathF.Sqrt(MathF.Max(0, sum));

        return index switch
        {
            0 => new Quaternion(missing, y, z, w),
            1 => new Quaternion(x, missing, z, w),
            2 => new Quaternion(x, y, missing, w),
            _ => new Quaternion(x, y, z, missing)
        };
    }

    /// <summary>
    /// Writes a Vector3 using half-precision floats (6 bytes total).
    /// </summary>
    public static void WriteVector3Half(this BinaryWriter writer, Vector3 value)
    {
        WriteHalf(writer, value.X);
        WriteHalf(writer, value.Y);
        WriteHalf(writer, value.Z);
    }

    /// <summary>
    /// Reads a Vector3 using half-precision floats.
    /// </summary>
    public static Vector3 ReadVector3Half(this BinaryReader reader)
    {
        float x = ReadHalf(reader);
        float y = ReadHalf(reader);
        float z = ReadHalf(reader);
        return new Vector3(x, y, z);
    }

    /// <summary>
    /// Writes a Vector4 using half-precision floats (8 bytes total).
    /// </summary>
    public static void WriteVector4Half(this BinaryWriter writer, Vector4 value)
    {
        WriteHalf(writer, value.X);
        WriteHalf(writer, value.Y);
        WriteHalf(writer, value.Z);
        WriteHalf(writer, value.W);
    }

    /// <summary>
    /// Reads a Vector4 using half-precision floats.
    /// </summary>
    public static Vector4 ReadVector4Half(this BinaryReader reader)
    {
        float x = ReadHalf(reader);
        float y = ReadHalf(reader);
        float z = ReadHalf(reader);
        float w = ReadHalf(reader);
        return new Vector4(x, y, z, w);
    }

    /// <summary>
    /// Writes an uncompressed normal as a packed uint (octahedral encoding).
    /// </summary>
    public static void WriteNormalPacked(this BinaryWriter writer, Vector3 normal)
    {
        // Octahedral encoding
        float invL1Norm = 1f / (MathF.Abs(normal.X) + MathF.Abs(normal.Y) + MathF.Abs(normal.Z));
        float nx = normal.X * invL1Norm;
        float ny = normal.Y * invL1Norm;

        if (normal.Z < 0)
        {
            float tx = (1f - MathF.Abs(ny)) * MathF.Sign(nx);
            float ty = (1f - MathF.Abs(nx)) * MathF.Sign(ny);
            nx = tx;
            ny = ty;
        }

        ushort encX = (ushort)Math.Clamp((int)((nx * 0.5f + 0.5f) * 65535f), 0, 65535);
        ushort encY = (ushort)Math.Clamp((int)((ny * 0.5f + 0.5f) * 65535f), 0, 65535);
        writer.Write(encX);
        writer.Write(encY);
    }

    /// <summary>
    /// Reads an octahedral-encoded normal.
    /// </summary>
    public static Vector3 ReadNormalPacked(this BinaryReader reader)
    {
        ushort encX = reader.ReadUInt16();
        ushort encY = reader.ReadUInt16();

        float nx = encX / 65535f * 2f - 1f;
        float ny = encY / 65535f * 2f - 1f;
        float nz = 1f - MathF.Abs(nx) - MathF.Abs(ny);

        if (nz < 0)
        {
            float tx = (1f - MathF.Abs(ny)) * MathF.Sign(nx);
            float ty = (1f - MathF.Abs(nx)) * MathF.Sign(ny);
            nx = tx;
            ny = ty;
        }

        Vector3 normal = new Vector3(nx, ny, nz);
        return Vector3.Normalize(normal);
    }

    #endregion

    #region Guid and Hash

    /// <summary>
    /// Writes a GUID (16 bytes).
    /// </summary>
    public static void WriteGuid(this BinaryWriter writer, Guid value)
    {
        Span<byte> bytes = stackalloc byte[16];
        value.TryWriteBytes(bytes);
        writer.Write(bytes);
    }

    /// <summary>
    /// Reads a GUID (16 bytes).
    /// </summary>
    public static Guid ReadGuid(this BinaryReader reader)
    {
        Span<byte> bytes = stackalloc byte[16];
        reader.Read(bytes);
        return new Guid(bytes);
    }

    /// <summary>
    /// Writes a hash128 (16 bytes).
    /// </summary>
    public static void WriteHash128(this BinaryWriter writer, ReadOnlySpan<byte> hash)
    {
        if (hash.Length != 16)
            throw new ArgumentException("Hash must be 16 bytes.");
        writer.Write(hash);
    }

    /// <summary>
    /// Reads a hash128 (16 bytes).
    /// </summary>
    public static byte[] ReadHash128(this BinaryReader reader) => reader.ReadBytes(16);

    /// <summary>
    /// Writes a hash256 (32 bytes).
    /// </summary>
    public static void WriteHash256(this BinaryWriter writer, ReadOnlySpan<byte> hash)
    {
        if (hash.Length != 32)
            throw new ArgumentException("Hash must be 32 bytes.");
        writer.Write(hash);
    }

    /// <summary>
    /// Reads a hash256 (32 bytes).
    /// </summary>
    public static byte[] ReadHash256(this BinaryReader reader) => reader.ReadBytes(32);

    #endregion

    #region Bounds

    /// <summary>
    /// Writes an axis-aligned bounding box (min + max).
    /// </summary>
    public static void WriteBounds(this BinaryWriter writer, Vector3 min, Vector3 max)
    {
        writer.Write(min);
        writer.Write(max);
    }

    /// <summary>
    /// Reads an axis-aligned bounding box.
    /// </summary>
    public static (Vector3 Min, Vector3 Max) ReadBounds(this BinaryReader reader)
    {
        Vector3 min = reader.ReadVector3();
        Vector3 max = reader.ReadVector3();
        return (min, max);
    }

    /// <summary>
    /// Writes a bounding sphere (center + radius).
    /// </summary>
    public static void WriteBoundingSphere(this BinaryWriter writer, Vector3 center, float radius)
    {
        writer.Write(center);
        writer.Write(radius);
    }

    /// <summary>
    /// Reads a bounding sphere.
    /// </summary>
    public static (Vector3 Center, float Radius) ReadBoundingSphere(this BinaryReader reader)
    {
        Vector3 center = reader.ReadVector3();
        float radius = reader.ReadSingle();
        return (center, radius);
    }

    #endregion

    #region Raw Memory

    /// <summary>
    /// Writes raw struct data directly to the binary writer.
    /// </summary>
    public static void WriteRaw<T>(this BinaryWriter writer, in T value) where T : struct
    {
        ReadOnlySpan<byte> data = MemoryMarshal.CreateReadOnlySpan(
            ref Unsafe.As<T, byte>(ref Unsafe.AsRef(in value)),
            Unsafe.SizeOf<T>());
        writer.Write(data);
    }

    /// <summary>
    /// Reads raw struct data directly from the binary reader.
    /// </summary>
    public static T ReadRaw<T>(this BinaryReader reader) where T : struct
    {
        Span<byte> data = stackalloc byte[Unsafe.SizeOf<T>()];
        reader.Read(data);
        return MemoryMarshal.Read<T>(data);
    }

    /// <summary>
    /// Writes raw span data (without length prefix).
    /// </summary>
    public static void WriteRawData(this BinaryWriter writer, ReadOnlySpan<byte> data)
    {
        writer.Write(data);
    }

    /// <summary>
    /// Reads raw data of a specified length (without length prefix).
    /// </summary>
    public static void ReadRawData(this BinaryReader reader, Span<byte> destination)
    {
        reader.Read(destination);
    }

    #endregion

    #region Length-Prefixed Sections

    /// <summary>
    /// Begins a length-prefixed section. Call EndSection after writing the section data.
    /// Returns the position of the length field (for patching later).
    /// </summary>
    public static long BeginSection(this BinaryWriter writer)
    {
        long position = writer.BaseStream.Position;
        writer.Write(0u); // placeholder for length
        return position;
    }

    /// <summary>
    /// Ends a length-prefixed section, patching the length field.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <param name="sectionStartPosition">Position returned by BeginSection.</param>
    public static void EndSection(this BinaryWriter writer, long sectionStartPosition)
    {
        long currentPosition = writer.BaseStream.Position;
        uint length = (uint)(currentPosition - sectionStartPosition - 4);
        long savedPosition = writer.BaseStream.Position;
        writer.BaseStream.Position = sectionStartPosition;
        writer.Write(length);
        writer.BaseStream.Position = savedPosition;
    }

    /// <summary>
    /// Reads a length-prefixed section's data size and prepares to read it.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <param name="sectionLength">The length of the section in bytes.</param>
    /// <returns>The end position of the section.</returns>
    public static long BeginReadSection(this BinaryReader reader, out uint sectionLength)
    {
        sectionLength = reader.ReadUInt32();
        return reader.BaseStream.Position + sectionLength;
    }

    /// <summary>
    /// Advances the reader to the end of a section if not already there.
    /// </summary>
    public static void EndReadSection(this BinaryReader reader, long sectionEndPosition)
    {
        reader.BaseStream.Position = sectionEndPosition;
    }

    #endregion

    #region Compressed Integers

    /// <summary>
    /// Writes a 32-bit integer using variable-length encoding, optimized for small positive values.
    /// </summary>
    public static void WritePackedInt32(this BinaryWriter writer, int value)
    {
        uint encoded = (uint)((value << 1) ^ (value >> 31));
        WritePackedUInt32(writer, encoded);
    }

    /// <summary>
    /// Reads a 32-bit integer using variable-length encoding.
    /// </summary>
    public static int ReadPackedInt32(this BinaryReader reader)
    {
        uint encoded = ReadPackedUInt32(reader);
        return (int)((encoded >> 1) ^ -(encoded & 1));
    }

    /// <summary>
    /// Writes a 32-bit unsigned integer using variable-length encoding, optimized for small values.
    /// </summary>
    public static void WritePackedUInt32(this BinaryWriter writer, uint value)
    {
        if (value < 128)
        {
            writer.Write((byte)value);
        }
        else if (value < 16384)
        {
            writer.Write((byte)((value & 0x7F) | 0x80));
            writer.Write((byte)(value >> 7));
        }
        else if (value < 2097152)
        {
            writer.Write((byte)((value & 0x7F) | 0x80));
            writer.Write((byte)(((value >> 7) & 0x7F) | 0x80));
            writer.Write((byte)(value >> 14));
        }
        else if (value < 268435456)
        {
            writer.Write((byte)((value & 0x7F) | 0x80));
            writer.Write((byte)(((value >> 7) & 0x7F) | 0x80));
            writer.Write((byte)(((value >> 14) & 0x7F) | 0x80));
            writer.Write((byte)(value >> 21));
        }
        else
        {
            writer.Write((byte)((value & 0x7F) | 0x80));
            writer.Write((byte)(((value >> 7) & 0x7F) | 0x80));
            writer.Write((byte)(((value >> 14) & 0x7F) | 0x80));
            writer.Write((byte)(((value >> 21) & 0x7F) | 0x80));
            writer.Write((byte)(value >> 28));
        }
    }

    /// <summary>
    /// Reads a 32-bit unsigned integer using variable-length encoding.
    /// </summary>
    public static uint ReadPackedUInt32(this BinaryReader reader)
    {
        uint result = 0;
        int shift = 0;
        byte b;
        do
        {
            b = reader.ReadByte();
            result |= (uint)(b & 0x7F) << shift;
            shift += 7;
        } while ((b & 0x80) != 0 && shift < 35);
        return result;
    }

    #endregion

    #region Fixed-Point

    /// <summary>
    /// Writes a fixed-point value (16.16 format) from a float.
    /// </summary>
    public static void WriteFixed16(this BinaryWriter writer, float value)
    {
        int fixedValue = (int)(value * 65536f);
        writer.Write(fixedValue);
    }

    /// <summary>
    /// Reads a fixed-point value (16.16 format) as a float.
    /// </summary>
    public static float ReadFixed16(this BinaryReader reader)
    {
        int fixedValue = reader.ReadInt32();
        return fixedValue / 65536f;
    }

    /// <summary>
    /// Writes a fixed-point value (24.8 format) from a float.
    /// </summary>
    public static void WriteFixed24(this BinaryWriter writer, float value)
    {
        int fixedValue = (int)(value * 256f);
        writer.Write(fixedValue);
    }

    /// <summary>
    /// Reads a fixed-point value (24.8 format) as a float.
    /// </summary>
    public static float ReadFixed24(this BinaryReader reader)
    {
        int fixedValue = reader.ReadInt32();
        return fixedValue / 256f;
    }

    #endregion
}
