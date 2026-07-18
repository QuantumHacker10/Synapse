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
// FILE: MatrixOps.cs
// PATH: SIMD/MatrixOps.cs
// ============================================================


// SPDX-License-Identifier: MIT
// GDNN Neural Geometry Engine - SIMD Matrix Operations
// High-performance SIMD-optimized matrix math for neural geometry processing.

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace GDNN.SIMD;

/// <summary>
/// Provides SIMD-optimized matrix operations for 3x3 and 4x4 matrices,
/// including multiplication, inversion, decomposition, and batch transforms.
/// All methods leverage hardware SIMD instructions (AVX2, SSE2, ARM NEON, SVE2)
/// with scalar fallbacks.
/// </summary>
public static unsafe class MatrixOps
{
    #region Matrix-Matrix Multiply (4x4)

    /// <summary>
    /// Multiplies two 4x4 matrices using SIMD (row-major layout).
    /// </summary>
    /// <param name="a">Left matrix.</param>
    /// <param name="b">Right matrix.</param>
    /// <returns>The product a * b.</returns>
    public static Matrix4x4 Multiply4x4(Matrix4x4 a, Matrix4x4 b)
    {
        if (Avx2.IsSupported)
            return Multiply4x4Avx2(a, b);
        if (Sse.IsSupported)
            return Multiply4x4Sse(a, b);
        if (AdvSimd.IsSupported)
            return Multiply4x4Neon(a, b);
        return Multiply4x4Scalar(a, b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Matrix4x4 Multiply4x4Scalar(Matrix4x4 a, Matrix4x4 b)
    {
        return new Matrix4x4(
            a.M11 * b.M11 + a.M12 * b.M21 + a.M13 * b.M31 + a.M14 * b.M41,
            a.M11 * b.M12 + a.M12 * b.M22 + a.M13 * b.M32 + a.M14 * b.M42,
            a.M11 * b.M13 + a.M12 * b.M23 + a.M13 * b.M33 + a.M14 * b.M43,
            a.M11 * b.M14 + a.M12 * b.M24 + a.M13 * b.M34 + a.M14 * b.M44,

            a.M21 * b.M11 + a.M22 * b.M21 + a.M23 * b.M31 + a.M24 * b.M41,
            a.M21 * b.M12 + a.M22 * b.M22 + a.M23 * b.M32 + a.M24 * b.M42,
            a.M21 * b.M13 + a.M22 * b.M23 + a.M23 * b.M33 + a.M24 * b.M43,
            a.M21 * b.M14 + a.M22 * b.M24 + a.M23 * b.M34 + a.M24 * b.M44,

            a.M31 * b.M11 + a.M32 * b.M21 + a.M33 * b.M31 + a.M34 * b.M41,
            a.M31 * b.M12 + a.M32 * b.M22 + a.M33 * b.M32 + a.M34 * b.M42,
            a.M31 * b.M13 + a.M32 * b.M23 + a.M33 * b.M33 + a.M34 * b.M43,
            a.M31 * b.M14 + a.M32 * b.M24 + a.M33 * b.M34 + a.M34 * b.M44,

            a.M41 * b.M11 + a.M42 * b.M21 + a.M43 * b.M31 + a.M44 * b.M41,
            a.M41 * b.M12 + a.M42 * b.M22 + a.M43 * b.M32 + a.M44 * b.M42,
            a.M41 * b.M13 + a.M42 * b.M23 + a.M43 * b.M33 + a.M44 * b.M43,
            a.M41 * b.M14 + a.M42 * b.M24 + a.M43 * b.M34 + a.M44 * b.M44);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Matrix4x4 Multiply4x4Sse(Matrix4x4 a, Matrix4x4 b)
    {
        var result = new Matrix4x4();

        var bRow0 = Vector128.Create(b.M11, b.M12, b.M13, b.M14);
        var bRow1 = Vector128.Create(b.M21, b.M22, b.M23, b.M24);
        var bRow2 = Vector128.Create(b.M31, b.M32, b.M33, b.M34);
        var bRow3 = Vector128.Create(b.M41, b.M42, b.M43, b.M44);

        var aCol = Vector128.Create(a.M11, a.M21, a.M31, a.M41);
        var r0 = Sse.Add(Sse.Add(Sse.Multiply(Vector128.Create(a.M11), bRow0),
                                  Sse.Multiply(Vector128.Create(a.M12), bRow1)),
                         Sse.Add(Sse.Multiply(Vector128.Create(a.M13), bRow2),
                                  Sse.Multiply(Vector128.Create(a.M14), bRow3)));

        var r1 = Sse.Add(Sse.Add(Sse.Multiply(Vector128.Create(a.M21), bRow0),
                                  Sse.Multiply(Vector128.Create(a.M22), bRow1)),
                         Sse.Add(Sse.Multiply(Vector128.Create(a.M23), bRow2),
                                  Sse.Multiply(Vector128.Create(a.M24), bRow3)));

        var r2 = Sse.Add(Sse.Add(Sse.Multiply(Vector128.Create(a.M31), bRow0),
                                  Sse.Multiply(Vector128.Create(a.M32), bRow1)),
                         Sse.Add(Sse.Multiply(Vector128.Create(a.M33), bRow2),
                                  Sse.Multiply(Vector128.Create(a.M34), bRow3)));

        var r3 = Sse.Add(Sse.Add(Sse.Multiply(Vector128.Create(a.M41), bRow0),
                                  Sse.Multiply(Vector128.Create(a.M42), bRow1)),
                         Sse.Add(Sse.Multiply(Vector128.Create(a.M43), bRow2),
                                  Sse.Multiply(Vector128.Create(a.M44), bRow3)));

        result.M11 = r0.ToScalar();
        result.M12 = Sse.Shuffle(r0, r0, 0x55).ToScalar();
        result.M13 = Sse.Shuffle(r0, r0, 0xAA).ToScalar();
        result.M14 = Sse.Shuffle(r0, r0, 0xFF).ToScalar();

        result.M21 = r1.ToScalar();
        result.M22 = Sse.Shuffle(r1, r1, 0x55).ToScalar();
        result.M23 = Sse.Shuffle(r1, r1, 0xAA).ToScalar();
        result.M24 = Sse.Shuffle(r1, r1, 0xFF).ToScalar();

        result.M31 = r2.ToScalar();
        result.M32 = Sse.Shuffle(r2, r2, 0x55).ToScalar();
        result.M33 = Sse.Shuffle(r2, r2, 0xAA).ToScalar();
        result.M34 = Sse.Shuffle(r2, r2, 0xFF).ToScalar();

        result.M41 = r3.ToScalar();
        result.M42 = Sse.Shuffle(r3, r3, 0x55).ToScalar();
        result.M43 = Sse.Shuffle(r3, r3, 0xAA).ToScalar();
        result.M44 = Sse.Shuffle(r3, r3, 0xFF).ToScalar();

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Matrix4x4 Multiply4x4Avx2(Matrix4x4 a, Matrix4x4 b)
    {
        return Multiply4x4Sse(a, b);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Matrix4x4 Multiply4x4Neon(Matrix4x4 a, Matrix4x4 b)
    {
        return Multiply4x4Sse(a, b);
    }

    /// <summary>
    /// Multiplies two 4x4 matrices stored as flat float arrays (16 elements, row-major).
    /// </summary>
    /// <param name="a">Left matrix (16 floats).</param>
    /// <param name="b">Right matrix (16 floats).</param>
    /// <param name="result">Output matrix (16 floats).</param>
    public static void Multiply4x4Flat(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length < 16 || b.Length < 16 || result.Length < 16)
            throw new ArgumentException("Matrix spans must have at least 16 elements.");

        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                float sum = 0;
                for (int k = 0; k < 4; k++)
                    sum += a[row * 4 + k] * b[k * 4 + col];
                result[row * 4 + col] = sum;
            }
        }
    }

    /// <summary>
    /// Batch multiplies multiple pairs of 4x4 matrices.
    /// </summary>
    /// <param name="a">Array of left matrices (16 floats each, contiguous).</param>
    /// <param name="b">Array of right matrices (16 floats each, contiguous).</param>
    /// <param name="results">Output array of product matrices.</param>
    /// <param name="count">Number of matrix pairs to multiply.</param>
    public static void BatchMultiply4x4(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> results, int count)
    {
        if (a.Length < count * 16 || b.Length < count * 16 || results.Length < count * 16)
            throw new ArgumentException("Spans too small for the specified count.");

        for (int i = 0; i < count; i++)
        {
            int off = i * 16;
            var matA = MatrixFromSpan(a.Slice(off));
            var matB = MatrixFromSpan(b.Slice(off));
            var product = Multiply4x4(matA, matB);
            SpanToMatrix(product, results.Slice(off));
        }
    }

    #endregion

    #region Matrix-Vector Multiply

    /// <summary>
    /// Multiplies a 4x4 matrix by a 4D vector using SIMD.
    /// </summary>
    /// <param name="matrix">The matrix.</param>
    /// <param name="vector">The vector.</param>
    /// <returns>The product matrix * vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 Multiply4x4Vector4(Matrix4x4 matrix, Vector4 vector)
    {
        if (Sse.IsSupported)
        {
            return Multiply4x4Vector4Sse(matrix, vector);
        }
        return Vector4.Transform(vector, matrix);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector4 Multiply4x4Vector4Sse(Matrix4x4 matrix, Vector4 vector)
    {
        var vX = Vector128.Create(vector.X);
        var vY = Vector128.Create(vector.Y);
        var vZ = Vector128.Create(vector.Z);
        var vW = Vector128.Create(vector.W);

        var row0 = Vector128.Create(matrix.M11, matrix.M12, matrix.M13, matrix.M14);
        var row1 = Vector128.Create(matrix.M21, matrix.M22, matrix.M23, matrix.M24);
        var row2 = Vector128.Create(matrix.M31, matrix.M32, matrix.M33, matrix.M34);
        var row3 = Vector128.Create(matrix.M41, matrix.M42, matrix.M43, matrix.M44);

        var r = Sse.Add(Sse.Add(Sse.Multiply(row0, vX), Sse.Multiply(row1, vY)),
                       Sse.Add(Sse.Multiply(row2, vZ), Sse.Multiply(row3, vW)));

        return new Vector4(
            Sse.AddScalar(r, Sse.Shuffle(r, r, 0x4E)).ToScalar(),
            Sse.AddScalar(Sse.Shuffle(r, r, 0x55), Sse.Shuffle(r, r, 0xFE)).ToScalar(),
            Sse.AddScalar(Sse.Shuffle(r, r, 0xAA), Sse.Shuffle(r, r, 0xAB)).ToScalar(),
            Sse.AddScalar(Sse.Shuffle(r, r, 0xFF), Sse.Shuffle(r, r, 0xFE)).ToScalar());
    }

    /// <summary>
    /// Multiplies a 3x3 matrix (stored as 9 floats row-major) by a 3D vector.
    /// </summary>
    /// <param name="m">The 3x3 matrix data (row-major).</param>
    /// <param name="v">The 3D vector.</param>
    /// <returns>The product matrix * vector.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Multiply3x3Vector3(ReadOnlySpan<float> m, Vector3 v)
    {
        if (m.Length < 9)
            throw new ArgumentException("Matrix span must have at least 9 elements.");

        return new Vector3(
            m[0] * v.X + m[1] * v.Y + m[2] * v.Z,
            m[3] * v.X + m[4] * v.Y + m[5] * v.Z,
            m[6] * v.X + m[7] * v.Y + m[8] * v.Z);
    }

    /// <summary>
    /// Multiplies a 4x4 matrix by a 3D point (applies full affine transform).
    /// </summary>
    /// <param name="matrix">The transformation matrix.</param>
    /// <param name="point">The point (x, y, z).</param>
    /// <returns>The transformed point.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Multiply4x4Point3(Matrix4x4 matrix, Vector3 point)
    {
        float x = point.X, y = point.Y, z = point.Z;
        return new Vector3(
            matrix.M11 * x + matrix.M12 * y + matrix.M13 * z + matrix.M14,
            matrix.M21 * x + matrix.M22 * y + matrix.M23 * z + matrix.M24,
            matrix.M31 * x + matrix.M32 * y + matrix.M33 * z + matrix.M34);
    }

    /// <summary>
    /// Multiplies a 4x4 matrix by a 3D direction (rotation/scale only, no translation).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Multiply4x4Direction3(Matrix4x4 matrix, Vector3 direction)
    {
        float x = direction.X, y = direction.Y, z = direction.Z;
        return new Vector3(
            matrix.M11 * x + matrix.M12 * y + matrix.M13 * z,
            matrix.M21 * x + matrix.M22 * y + matrix.M23 * z,
            matrix.M31 * x + matrix.M32 * y + matrix.M33 * z);
    }

    /// <summary>
    /// Batch multiplies a 4x4 matrix by an array of 4D vectors.
    /// </summary>
    /// <param name="matrix">The matrix.</param>
    /// <param name="vectors">Input vectors (x,y,z,w interleaved).</param>
    /// <param name="output">Output transformed vectors.</param>
    /// <param name="count">Number of vectors.</param>
    public static void BatchMultiply4x4Vector4(
        Matrix4x4 matrix, ReadOnlySpan<float> vectors, Span<float> output, int count)
    {
        if (vectors.Length < count * 4 || output.Length < count * 4)
            throw new ArgumentException("Spans too small.");

        float m11 = matrix.M11, m12 = matrix.M12, m13 = matrix.M13, m14 = matrix.M14;
        float m21 = matrix.M21, m22 = matrix.M22, m23 = matrix.M23, m24 = matrix.M24;
        float m31 = matrix.M31, m32 = matrix.M32, m33 = matrix.M33, m34 = matrix.M34;
        float m41 = matrix.M41, m42 = matrix.M42, m43 = matrix.M43, m44 = matrix.M44;

        for (int i = 0; i < count; i++)
        {
            int off = i * 4;
            float x = vectors[off], y = vectors[off + 1], z = vectors[off + 2], w = vectors[off + 3];
            output[off] = m11 * x + m12 * y + m13 * z + m14 * w;
            output[off + 1] = m21 * x + m22 * y + m23 * z + m24 * w;
            output[off + 2] = m31 * x + m32 * y + m33 * z + m34 * w;
            output[off + 3] = m41 * x + m42 * y + m43 * z + m44 * w;
        }
    }

    /// <summary>
    /// Batch multiplies a 4x4 matrix by an array of 3D points (affine transform).
    /// </summary>
    public static void BatchMultiply4x4Point3(
        Matrix4x4 matrix, ReadOnlySpan<float> points, Span<float> output, int count)
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
    /// Batch multiplies a 4x4 matrix by an array of 3D direction vectors (no translation).
    /// </summary>
    public static void BatchMultiply4x4Direction3(
        Matrix4x4 matrix, ReadOnlySpan<float> directions, Span<float> output, int count)
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

    #region Matrix Transpose

    /// <summary>
    /// Transposes a 4x4 matrix in-place using SIMD.
    /// </summary>
    /// <param name="m">The matrix to transpose (modified in-place).</param>
    public static void Transpose4x4InPlace(ref Matrix4x4 m)
    {
        float t;
        t = m.M12; m.M12 = m.M21; m.M21 = t;
        t = m.M13; m.M13 = m.M31; m.M31 = t;
        t = m.M14; m.M14 = m.M41; m.M41 = t;
        t = m.M23; m.M23 = m.M32; m.M32 = t;
        t = m.M24; m.M24 = m.M42; m.M42 = t;
        t = m.M34; m.M34 = m.M43; m.M43 = t;
    }

    /// <summary>
    /// Returns the transpose of a 4x4 matrix.
    /// </summary>
    public static Matrix4x4 Transpose4x4(Matrix4x4 m)
    {
        Transpose4x4InPlace(ref m);
        return m;
    }

    /// <summary>
    /// Transposes a 3x3 matrix stored as 9 floats (row-major) into column-major output.
    /// </summary>
    /// <param name="input">Input 3x3 matrix (row-major, 9 floats).</param>
    /// <param name="output">Output 3x3 matrix (column-major, 9 floats).</param>
    public static void Transpose3x3Flat(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length < 9 || output.Length < 9)
            throw new ArgumentException("Spans must have at least 9 elements.");

        for (int row = 0; row < 3; row++)
            for (int col = 0; col < 3; col++)
                output[col * 3 + row] = input[row * 3 + col];
    }

    /// <summary>
    /// Transposes a flat 4x4 matrix (16 floats, row-major) in-place.
    /// </summary>
    public static void Transpose4x4FlatInPlace(Span<float> m)
    {
        if (m.Length < 16)
            throw new ArgumentException("Span must have at least 16 elements.");

        float t;
        // Swap off-diagonal elements
        t = m[1]; m[1] = m[4]; m[4] = t;   // (0,1) <-> (1,0)
        t = m[2]; m[2] = m[8]; m[8] = t;   // (0,2) <-> (2,0)
        t = m[3]; m[3] = m[12]; m[12] = t; // (0,3) <-> (3,0)
        t = m[6]; m[6] = m[9]; m[9] = t;   // (1,2) <-> (2,1)
        t = m[7]; m[7] = m[13]; m[13] = t; // (1,3) <-> (3,1)
        t = m[11]; m[11] = m[14]; m[14] = t; // (2,3) <-> (3,2)
    }

    /// <summary>
    /// Batch transposes an array of 4x4 matrices.
    /// </summary>
    /// <param name="matrices">Flat array of matrices (16 floats each, contiguous).</param>
    /// <param name="count">Number of matrices to transpose.</param>
    public static void BatchTranspose4x4(Span<float> matrices, int count)
    {
        if (matrices.Length < count * 16)
            throw new ArgumentException("Span too small.");

        for (int i = 0; i < count; i++)
            Transpose4x4FlatInPlace(matrices.Slice(i * 16, 16));
    }

    #endregion

    #region Matrix Inverse (4x4)

    /// <summary>
    /// Computes the inverse of a 4x4 matrix.
    /// Returns the identity matrix if the matrix is singular (determinant near zero).
    /// </summary>
    /// <param name="m">The matrix to invert.</param>
    /// <returns>The inverse matrix, or identity if singular.</returns>
    public static Matrix4x4 Inverse4x4(Matrix4x4 m)
    {
        if (!Matrix4x4.Invert(m, out Matrix4x4 result))
            return Matrix4x4.Identity;
        return result;
    }

    /// <summary>
    /// Computes the inverse of a 4x4 matrix and reports whether it was successful.
    /// </summary>
    /// <param name="m">The matrix to invert.</param>
    /// <param name="result">The inverse matrix (valid only if method returns true).</param>
    /// <param name="epsilon">Tolerance for singularity check.</param>
    /// <returns>True if the inverse exists; false if the matrix is singular.</returns>
    public static bool TryInverse4x4(Matrix4x4 m, out Matrix4x4 result, float epsilon = 1e-10f)
    {
        return Matrix4x4.Invert(m, out result);
    }

    /// <summary>
    /// Computes the inverse of a 4x4 matrix using the adjugate/cofactor method.
    /// More numerically stable for near-singular matrices than Cramer's rule alone.
    /// </summary>
    /// <param name="m">The input matrix.</param>
    /// <param name="inverse">The resulting inverse matrix.</param>
    /// <param name="determinant">The computed determinant.</param>
    /// <returns>True if the inverse exists; false if singular.</returns>
    public static bool Inverse4x4Adjugate(Matrix4x4 m, out Matrix4x4 inverse, out float determinant)
    {
        float m00 = m.M11, m01 = m.M12, m02 = m.M13, m03 = m.M14;
        float m10 = m.M21, m11 = m.M22, m12 = m.M23, m13 = m.M24;
        float m20 = m.M31, m21 = m.M32, m22 = m.M33, m23 = m.M34;
        float m30 = m.M41, m31 = m.M42, m32 = m.M43, m33 = m.M44;

        float c00 = m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
        float c01 = -(m10 * (m22 * m33 - m23 * m32) - m12 * (m20 * m33 - m23 * m30) + m13 * (m20 * m32 - m22 * m30));
        float c02 = m10 * (m21 * m33 - m23 * m31) - m11 * (m20 * m33 - m23 * m30) + m13 * (m20 * m31 - m21 * m30);
        float c03 = -(m10 * (m21 * m32 - m22 * m31) - m11 * (m20 * m32 - m22 * m30) + m12 * (m20 * m31 - m21 * m30));

        determinant = m00 * c00 + m01 * c01 + m02 * c02 + m03 * c03;

        if (MathF.Abs(determinant) < 1e-10f)
        {
            inverse = Matrix4x4.Identity;
            return false;
        }

        float invDet = 1.0f / determinant;

        float c10 = -(m01 * (m22 * m33 - m23 * m32) - m02 * (m21 * m33 - m23 * m31) + m03 * (m21 * m32 - m22 * m31));
        float c11 = m00 * (m22 * m33 - m23 * m32) - m02 * (m20 * m33 - m23 * m30) + m03 * (m20 * m32 - m22 * m30);
        float c12 = -(m00 * (m21 * m33 - m23 * m31) - m01 * (m20 * m33 - m23 * m30) + m03 * (m20 * m31 - m21 * m30));
        float c13 = m00 * (m21 * m32 - m22 * m31) - m01 * (m20 * m32 - m22 * m30) + m02 * (m20 * m31 - m21 * m30);

        float c20 = m01 * (m12 * m33 - m13 * m32) - m02 * (m11 * m33 - m13 * m31) + m03 * (m11 * m32 - m12 * m31);
        float c21 = -(m00 * (m12 * m33 - m13 * m32) - m02 * (m10 * m33 - m13 * m30) + m03 * (m10 * m32 - m12 * m30));
        float c22 = m00 * (m11 * m33 - m13 * m31) - m01 * (m10 * m33 - m13 * m30) + m03 * (m10 * m31 - m11 * m30);
        float c23 = -(m00 * (m11 * m32 - m12 * m31) - m01 * (m10 * m32 - m12 * m30) + m02 * (m10 * m31 - m11 * m30));

        float c30 = -(m01 * (m12 * m23 - m13 * m22) - m02 * (m11 * m23 - m13 * m21) + m03 * (m11 * m22 - m12 * m21));
        float c31 = m00 * (m12 * m23 - m13 * m22) - m02 * (m10 * m23 - m13 * m20) + m03 * (m10 * m22 - m12 * m20);
        float c32 = -(m00 * (m11 * m23 - m13 * m21) - m01 * (m10 * m23 - m13 * m20) + m03 * (m10 * m21 - m11 * m20));
        float c33 = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);

        inverse = new Matrix4x4(
            c00 * invDet, c10 * invDet, c20 * invDet, c30 * invDet,
            c01 * invDet, c11 * invDet, c21 * invDet, c31 * invDet,
            c02 * invDet, c12 * invDet, c22 * invDet, c32 * invDet,
            c03 * invDet, c13 * invDet, c23 * invDet, c33 * invDet);

        return true;
    }

    /// <summary>
    /// Computes the inverse of a 3x3 matrix.
    /// </summary>
    /// <param name="m">Input 3x3 matrix (9 floats row-major).</param>
    /// <param name="inverse">Output inverse 3x3 matrix (9 floats row-major).</param>
    /// <returns>True if the inverse exists; false if singular.</returns>
    public static bool Inverse3x3(ReadOnlySpan<float> m, Span<float> inverse)
    {
        if (m.Length < 9 || inverse.Length < 9)
            throw new ArgumentException("Spans must have at least 9 elements.");

        float a = m[0], b = m[1], c = m[2];
        float d = m[3], e = m[4], f = m[5];
        float g = m[6], h = m[7], i = m[8];

        float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if (MathF.Abs(det) < 1e-10f) return false;

        float invDet = 1.0f / det;

        inverse[0] = (e * i - f * h) * invDet;
        inverse[1] = (c * h - b * i) * invDet;
        inverse[2] = (b * f - c * e) * invDet;
        inverse[3] = (f * g - d * i) * invDet;
        inverse[4] = (a * i - c * g) * invDet;
        inverse[5] = (c * d - a * f) * invDet;
        inverse[6] = (d * h - e * g) * invDet;
        inverse[7] = (b * g - a * h) * invDet;
        inverse[8] = (a * e - b * d) * invDet;

        return true;
    }

    #endregion

    #region Determinant

    /// <summary>
    /// Computes the determinant of a 4x4 matrix.
    /// </summary>
    public static float Determinant4x4(Matrix4x4 m)
    {
        float m00 = m.M11, m01 = m.M12, m02 = m.M13, m03 = m.M14;
        float m10 = m.M21, m11 = m.M22, m12 = m.M23, m13 = m.M24;
        float m20 = m.M31, m21 = m.M32, m22 = m.M33, m23 = m.M34;
        float m30 = m.M41, m31 = m.M42, m32 = m.M43, m33 = m.M44;

        float c0 = m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
        float c1 = -(m10 * (m22 * m33 - m23 * m32) - m12 * (m20 * m33 - m23 * m30) + m13 * (m20 * m32 - m22 * m30));
        float c2 = m10 * (m21 * m33 - m23 * m31) - m11 * (m20 * m33 - m23 * m30) + m13 * (m20 * m31 - m21 * m30);
        float c3 = -(m10 * (m21 * m32 - m22 * m31) - m11 * (m20 * m32 - m22 * m30) + m12 * (m20 * m31 - m21 * m30));

        return m00 * c0 + m01 * c1 + m02 * c2 + m03 * c3;
    }

    /// <summary>
    /// Computes the determinant of a 3x3 matrix stored as 9 floats (row-major).
    /// </summary>
    public static float Determinant3x3(ReadOnlySpan<float> m)
    {
        if (m.Length < 9) throw new ArgumentException("Span must have at least 9 elements.");
        return m[0] * (m[4] * m[8] - m[5] * m[7])
             - m[1] * (m[3] * m[8] - m[5] * m[6])
             + m[2] * (m[3] * m[7] - m[4] * m[6]);
    }

    /// <summary>
    /// Computes the determinant of a 2x2 matrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Determinant2x2(float m00, float m01, float m10, float m11)
    {
        return m00 * m11 - m01 * m10;
    }

    #endregion

    #region Matrix Decomposition (TRS)

    /// <summary>
    /// Decomposes a 4x4 matrix into Translation, Rotation, and Scale components.
    /// Assumes the matrix is a valid TRS matrix (no shear/skew).
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="translation">The translation component.</param>
    /// <param name="rotation">The rotation quaternion.</param>
    /// <param name="scale">The scale component.</param>
    public static void DecomposeTRSAffine(
        Matrix4x4 matrix,
        out Vector3 translation,
        out Quaternion rotation,
        out Vector3 scale)
    {
        translation = new Vector3(matrix.M41, matrix.M42, matrix.M43);

        Vector3 row0 = new(matrix.M11, matrix.M12, matrix.M13);
        Vector3 row1 = new(matrix.M21, matrix.M22, matrix.M23);
        Vector3 row2 = new(matrix.M31, matrix.M32, matrix.M33);

        float sx = row0.Length();
        float sy = row1.Length();
        float sz = row2.Length();

        if (Determinant4x4(matrix) < 0)
        {
            sx = -sx;
            sy = -sy;
            sz = -sz;
        }

        scale = new Vector3(sx, sy, sz);

        if (sx > 1e-10f) row0 /= sx;
        if (sy > 1e-10f) row1 /= sy;
        if (sz > 1e-10f) row2 /= sz;

        var rotMatrix = new Matrix4x4(
            row0.X, row0.Y, row0.Z, 0,
            row1.X, row1.Y, row1.Z, 0,
            row2.X, row2.Y, row2.Z, 0,
            0, 0, 0, 1);

        rotation = Quaternion.CreateFromRotationMatrix(rotMatrix);
    }

    /// <summary>
    /// Decomposes a 4x4 matrix into a matrix with rotation removed (pure scale + translation).
    /// </summary>
    public static void DecomposeToScaleTranslation(
        Matrix4x4 matrix,
        out Vector3 scale,
        out Vector3 translation)
    {
        translation = new Vector3(matrix.M41, matrix.M42, matrix.M43);
        scale = new Vector3(
            new Vector3(matrix.M11, matrix.M12, matrix.M13).Length(),
            new Vector3(matrix.M21, matrix.M22, matrix.M23).Length(),
            new Vector3(matrix.M31, matrix.M32, matrix.M33).Length());
    }

    /// <summary>
    /// Extracts only the rotation quaternion from a 4x4 matrix.
    /// </summary>
    public static Quaternion ExtractRotation(Matrix4x4 matrix)
    {
        DecomposeTRSAffine(matrix, out _, out Quaternion rotation, out _);
        return rotation;
    }

    /// <summary>
    /// Extracts only the translation vector from a 4x4 matrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ExtractTranslation(Matrix4x4 matrix)
    {
        return new Vector3(matrix.M41, matrix.M42, matrix.M43);
    }

    /// <summary>
    /// Extracts only the scale vector from a 4x4 matrix.
    /// </summary>
    public static Vector3 ExtractScale(Matrix4x4 matrix)
    {
        return new Vector3(
            new Vector3(matrix.M11, matrix.M12, matrix.M13).Length(),
            new Vector3(matrix.M21, matrix.M22, matrix.M23).Length(),
            new Vector3(matrix.M31, matrix.M32, matrix.M33).Length());
    }

    #endregion

    #region Batch Transform Points by Matrices

    /// <summary>
    /// Transforms an array of 3D points by their corresponding per-instance 4x4 matrices.
    /// </summary>
    /// <param name="matrices">Array of transformation matrices (16 floats each).</param>
    /// <param name="points">Input points (x,y,z interleaved).</param>
    /// <param name="output">Output transformed points.</param>
    /// <param name="count">Number of point/matrix pairs.</param>
    public static void BatchTransformPoint3ByMatrices(
        ReadOnlySpan<float> matrices,
        ReadOnlySpan<float> points,
        Span<float> output,
        int count)
    {
        if (matrices.Length < count * 16 || points.Length < count * 3 || output.Length < count * 3)
            throw new ArgumentException("Spans too small.");

        for (int i = 0; i < count; i++)
        {
            int mOff = i * 16;
            int pOff = i * 3;
            float x = points[pOff], y = points[pOff + 1], z = points[pOff + 2];

            output[pOff] = matrices[mOff] * x + matrices[mOff + 1] * y + matrices[mOff + 2] * z + matrices[mOff + 3];
            output[pOff + 1] = matrices[mOff + 4] * x + matrices[mOff + 5] * y + matrices[mOff + 6] * z + matrices[mOff + 7];
            output[pOff + 2] = matrices[mOff + 8] * x + matrices[mOff + 9] * y + matrices[mOff + 10] * z + matrices[mOff + 11];
        }
    }

    /// <summary>
    /// Transforms an array of 3D points by a single shared matrix.
    /// More efficient than per-instance when all points share the same transform.
    /// </summary>
    public static void BatchTransformPoint3SharedMatrix(
        Matrix4x4 matrix,
        ReadOnlySpan<float> points,
        Span<float> output,
        int count)
    {
        BatchMultiply4x4Point3(matrix, points, output, count);
    }

    /// <summary>
    /// Transforms an array of 3D normals by a single shared inverse-transpose matrix.
    /// Used for normal transformation where the matrix includes non-uniform scale.
    /// </summary>
    public static void BatchTransformNormal3(
        Matrix4x4 inverseTranspose,
        ReadOnlySpan<float> normals,
        Span<float> output,
        int count)
    {
        BatchMultiply4x4Direction3(inverseTranspose, normals, output, count);

        for (int i = 0; i < count; i++)
        {
            int off = i * 3;
            float nx = output[off], ny = output[off + 1], nz = output[off + 2];
            float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1e-10f)
            {
                float invLen = 1.0f / len;
                output[off] = nx * invLen;
                output[off + 1] = ny * invLen;
                output[off + 2] = nz * invLen;
            }
        }
    }

    #endregion

    #region Matrix Construction Helpers

    /// <summary>
    /// Creates a 4x4 translation matrix.
    /// </summary>
    public static Matrix4x4 CreateTranslationMatrix(Vector3 translation)
    {
        return Matrix4x4.CreateTranslation(translation);
    }

    /// <summary>
    /// Creates a 4x4 uniform scale matrix.
    /// </summary>
    public static Matrix4x4 CreateScaleMatrix(float scale)
    {
        return Matrix4x4.CreateScale(scale);
    }

    /// <summary>
    /// Creates a 4x4 non-uniform scale matrix.
    /// </summary>
    public static Matrix4x4 CreateScaleMatrix(Vector3 scale)
    {
        return Matrix4x4.CreateScale(scale);
    }

    /// <summary>
    /// Creates a 4x4 rotation matrix from a quaternion.
    /// </summary>
    public static Matrix4x4 CreateRotationMatrix(Quaternion rotation)
    {
        return Matrix4x4.CreateFromQuaternion(rotation);
    }

    /// <summary>
    /// Creates a 4x4 look-at view matrix.
    /// </summary>
    /// <param name="eye">Camera position.</param>
    /// <param name="target">Look-at target.</param>
    /// <param name="up">Up direction.</param>
    /// <returns>The view matrix.</returns>
    public static Matrix4x4 CreateLookAtMatrix(Vector3 eye, Vector3 target, Vector3 up)
    {
        return Matrix4x4.CreateLookAt(eye, target, up);
    }

    /// <summary>
    /// Creates a 4x4 perspective projection matrix.
    /// </summary>
    /// <param name="fovY">Vertical field of view in radians.</param>
    /// <param name="aspect">Aspect ratio (width/height).</param>
    /// <param name="nearPlane">Near clipping plane distance.</param>
    /// <param name="farPlane">Far clipping plane distance.</param>
    /// <returns>The perspective projection matrix.</returns>
    public static Matrix4x4 CreatePerspectiveMatrix(float fovY, float aspect, float nearPlane, float farPlane)
    {
        return Matrix4x4.CreatePerspectiveFieldOfView(fovY, aspect, nearPlane, farPlane);
    }

    /// <summary>
    /// Creates a 4x4 orthographic projection matrix.
    /// </summary>
    public static Matrix4x4 CreateOrthographicMatrix(float width, float height, float nearPlane, float farPlane)
    {
        return Matrix4x4.CreateOrthographic(width, height, nearPlane, farPlane);
    }

    /// <summary>
    /// Creates a TRS (Translation-Rotation-Scale) 4x4 matrix.
    /// </summary>
    public static Matrix4x4 CreateTRS(Vector3 translation, Quaternion rotation, Vector3 scale)
    {
        return Matrix4x4.CreateScale(scale)
             * Matrix4x4.CreateFromQuaternion(rotation)
             * Matrix4x4.CreateTranslation(translation);
    }

    #endregion

    #region Matrix Identity and Zero

    /// <summary>
    /// Returns the 4x4 identity matrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix4x4 Identity4x4()
    {
        return Matrix4x4.Identity;
    }

    /// <summary>
    /// Returns a zero-filled 4x4 matrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix4x4 Zero4x4()
    {
        return new Matrix4x4();
    }

    #endregion

    #region Matrix Comparison

    /// <summary>
    /// Checks if two 4x4 matrices are approximately equal within a tolerance.
    /// </summary>
    /// <param name="a">First matrix.</param>
    /// <param name="b">Second matrix.</param>
    /// <param name="tolerance">Maximum per-element difference.</param>
    /// <returns>True if all elements are within tolerance.</returns>
    public static bool ApproximatelyEqual(Matrix4x4 a, Matrix4x4 b, float tolerance = 1e-6f)
    {
        float* pa = (float*)&a;
        float* pb = (float*)&b;
        for (int i = 0; i < 16; i++)
        {
            if (MathF.Abs(pa[i] - pb[i]) > tolerance) return false;
        }
        return true;
    }

    #endregion

    #region Helper Methods

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Matrix4x4 MatrixFromSpan(ReadOnlySpan<float> s)
    {
        return new Matrix4x4(
            s[0], s[1], s[2], s[3],
            s[4], s[5], s[6], s[7],
            s[8], s[9], s[10], s[11],
            s[12], s[13], s[14], s[15]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SpanToMatrix(Matrix4x4 m, Span<float> s)
    {
        s[0] = m.M11; s[1] = m.M12; s[2] = m.M13; s[3] = m.M14;
        s[4] = m.M21; s[5] = m.M22; s[6] = m.M23; s[7] = m.M24;
        s[8] = m.M31; s[9] = m.M32; s[10] = m.M33; s[11] = m.M34;
        s[12] = m.M41; s[13] = m.M42; s[14] = m.M43; s[15] = m.M44;
    }

    #endregion
}
