using System;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Synapse.Core;

namespace GDNN.Rendering.Compat
{
    public static class RenderingMath
    {
        public static Vector3 Forward => -Vector3.UnitZ;
        public static Vector3 Right => Vector3.UnitX;

        public static Matrix4x4 ZeroMatrix => new(
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Rotate(Quaternion rotation, Vector3 vector) =>
            Vector3.Transform(vector, Matrix4x4.CreateFromQuaternion(rotation));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3D ToVector3D(Vector3 vector) => new(vector.X, vector.Y, vector.Z);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 ToVector3(Vector3D vector) => new((float)vector.X, (float)vector.Y, (float)vector.Z);
    }

    public static class Matrix4x4Compat
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Determinant(Matrix4x4 matrix)
        {
            float a = matrix.M11, b = matrix.M12, c = matrix.M13, d = matrix.M14;
            float e = matrix.M21, f = matrix.M22, g = matrix.M23, h = matrix.M24;
            float i = matrix.M31, j = matrix.M32, k = matrix.M33, l = matrix.M34;
            float m = matrix.M41, n = matrix.M42, o = matrix.M43, p = matrix.M44;

            float kpLo = k * p - l * o;
            float jpLn = j * p - l * n;
            float joKn = j * o - k * n;
            float ipLm = i * p - l * m;
            float ioKm = i * o - k * m;
            float inJm = i * n - j * m;

            return a * (f * kpLo - g * jpLn + h * joKn)
                 - b * (e * kpLo - g * ipLm + h * ioKm)
                 + c * (e * jpLn - f * ipLm + h * inJm)
                 - d * (e * joKn - f * ioKm + g * inJm);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 GetScale(Matrix4x4 matrix) => new(
            new Vector3(matrix.M11, matrix.M12, matrix.M13).Length(),
            new Vector3(matrix.M21, matrix.M22, matrix.M23).Length(),
            new Vector3(matrix.M31, matrix.M32, matrix.M33).Length());
    }

    public static class SpanCompat
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CopyFrom(this Span<byte> destination, ReadOnlySpan<byte> source) =>
            source.CopyTo(destination);
    }

    public static class BrotliCompat
    {
        public static byte[] Compress(ReadOnlySpan<byte> source, CompressionLevel level = CompressionLevel.Optimal)
        {
            using var output = new MemoryStream();
            using (var brotli = new BrotliStream(output, level, leaveOpen: true))
            {
                brotli.Write(source);
            }

            return output.ToArray();
        }

        public static byte[] Decompress(ReadOnlySpan<byte> source)
        {
            using var input = new MemoryStream(source.ToArray());
            using var brotli = new BrotliStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream();
            brotli.CopyTo(output);
            return output.ToArray();
        }
    }

    public static class GuidCompat
    {
        public static void WriteBytes(Span<byte> destination, Guid value) =>
            value.TryWriteBytes(destination);
    }

    public static class MemoryStreamExtensions
    {
        public static Span<byte> GetSpan(this MemoryStream stream, int byteCount)
        {
            var position = (int)stream.Position;
            var requiredLength = position + byteCount;
            if (stream.Length < requiredLength)
                stream.SetLength(requiredLength);

            return stream.GetBuffer().AsSpan(position, byteCount);
        }

        public static void Advance(this MemoryStream stream, int byteCount) =>
            stream.Position += byteCount;
    }

    public static class MemoryMarshalCompat
    {
        public static Memory<T> CreateMemory<T>(ref T reference, int length) where T : struct =>
            MemoryMarshal.CreateFromPinnedArray(new[] { reference }, 0, Math.Min(length, 1));

        public static Memory<T> CreateMemoryFromSpan<T>(Span<T> span) where T : struct
        {
            if (span.IsEmpty)
                return Memory<T>.Empty;

            var array = span.ToArray();
            return array.AsMemory(0, array.Length);
        }
    }

    public static class MarshalCompat
    {
        public static void WriteFloat(IntPtr ptr, float value) =>
            Marshal.WriteInt32(ptr, BitConverter.SingleToInt32Bits(value));
    }
}
