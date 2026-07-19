using System;
// ============================================================
// FILE: MathFunctions.cs
// PATH: SIMD/MathFunctions.cs
// ============================================================


// SPDX-License-Identifier: MIT
// GDNN Neural Geometry Engine - SIMD Math Functions
// Fast SIMD-approximated mathematical functions for neural network evaluation.

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
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.SIMD;

/// <summary>
/// Provides SIMD-approximated mathematical functions optimized for neural network
/// evaluation. These functions trade full IEEE 754 precision for significant
/// throughput gains, suitable for inference workloads where approximate results are acceptable.
/// </summary>
public static class MathFunctions
{
    /// <summary>Polynomial approximation coefficients for fast sin(x) on [-pi, pi].</summary>
    private static readonly Vector<float> SinCoeff1 = new(-1.6666654611e-1f);
    private static readonly Vector<float> SinCoeff2 = new(8.3333337673e-3f);
    private static readonly Vector<float> SinCoeff3 = new(-1.9841270114e-4f);

    /// <summary>Polynomial approximation coefficients for fast cos(x) on [-pi, pi].</summary>
    private static readonly Vector<float> CosCoeff1 = new(-4.9999990594e-1f);
    private static readonly Vector<float> CosCoeff2 = new(4.1666649020e-2f);
    private static readonly Vector<float> CosCoeff3 = new(-1.3888887545e-3f);

    /// <summary>Constants for fast exp approximation (Schraudolph's method).</summary>
    private static readonly Vector<float> ExpBase = new(2.0f);
    private static readonly Vector<float> ExpLn2 = new(0.6931471805599453f);
    private static readonly Vector<float> ExpP2 = new(1.4426950408889634f);

    /// <summary>Coefficients for the rational approximation of tanh.</summary>
    private static readonly Vector<float> TanhAlpha1 = new(-3.0f);
    private static readonly Vector<float> TanhAlpha2 = new(-3.0f);
    private static readonly Vector<float> TanhCoeff = new(1.0f / 3.0f);

    /// <summary>Approximate 1/sqrt(x) constant (magic number for fast rsqrt).</summary>
    private static readonly Vector<float> RsqrtMagic = new(0.5f);
    private static readonly Vector<float> RsqrtThreeHalves = new(1.5f);

    /// <summary>GELU approximation constant (sqrt(2/pi)).</summary>
    private const float GELUSqrt2OverPi = 0.7978845608028654f;

    /// <summary>GELU approximation polynomial coefficient.</summary>
    private const float GELUCoeff = 0.044715f;

    /// <summary>Log2(e) constant for base conversion.</summary>
    private const float Log2E = 1.4426950408889634f;

    /// <summary>Ln(2) constant for base conversion.</summary>
    private const float Ln2 = 0.6931471805599453f;

    /// <summary>LN2 polynomial approximation coefficients.</summary>
    private static readonly Vector<float> LnCoeff1 = new(0.66666668653f);
    private static readonly Vector<float> LnCoeff2 = new(-0.22221984321f);
    private static readonly Vector<float> LnCoeff3 = new(0.09876294159f);

    /// <summary>Scalar sin approximation coefficient set.</summary>
    private const float ScalarSinC1 = -1.6666654611e-1f;
    private const float ScalarSinC2 = 8.3333337673e-3f;
    private const float ScalarSinC3 = -1.9841270114e-4f;

    /// <summary>Scalar cos approximation coefficient set.</summary>
    private const float ScalarCosC1 = -4.9999990594e-1f;
    private const float ScalarCosC2 = 4.1666649020e-2f;
    private const float ScalarCosC3 = -1.3888887545e-3f;

    /// <summary>
    /// Computes the fast approximate reciprocal of a single-precision float.
    /// Uses the hardware rsqrt instruction where available, refined with Newton-Raphson.
    /// Maximum relative error: ~1e-6 after one refinement step.
    /// </summary>
    /// <param name="x">The input value. Must be positive and non-zero.</param>
    /// <returns>An approximation of 1/sqrt(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastRsqrt(float x)
    {
        if (Vector.IsHardwareAccelerated)
        {
            var v = new Vector<float>(x);
            var result = FastRsqrtVector(v);
            return result[0];
        }
        return ScalarFastRsqrt(x);
    }

    /// <summary>
    /// Computes the fast approximate reciprocal of a single-precision float using scalar operations.
    /// Uses the Quake III fast inverse square root technique with one Newton-Raphson iteration.
    /// </summary>
    /// <param name="x">The input value. Must be positive.</param>
    /// <returns>An approximation of 1/sqrt(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastRsqrt(float x)
    {
        float xHalf = 0.5f * x;
        int i = BitConverter.SingleToInt32Bits(x);
        i = 0x5F3759DF - (i >> 1);
        float y = BitConverter.Int32BitsToSingle(i);
        y = y * (1.5f - xHalf * y * y);
        return y;
    }

    /// <summary>
    /// Computes the fast approximate reciprocal square root for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector of approximate 1/sqrt(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastRsqrtVector(Vector<float> x) =>
        ScalarFastRsqrtVector(x);

    /// <summary>
    /// Scalar fallback for vector reciprocal square root.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> ScalarFastRsqrtVector(Vector<float> x)
    {
        Span<float> buffer = stackalloc float[Vector<float>.Count];
        for (int i = 0; i < Vector<float>.Count; i++)
            buffer[i] = ScalarFastRsqrt(x[i]);
        return new Vector<float>(buffer);
    }

    /// <summary>
    /// Computes the fast approximate reciprocal of a vector of floats.
    /// Uses IEEE 754 bit manipulation for the initial estimate, refined with Newton-Raphson.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector of approximate 1/x values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastReciprocal(Vector<float> x)
    {
        if (Sse.IsSupported || AdvSimd.IsSupported)
        {
            var one = Vector<float>.One;
            var two = one + one;
            var estimate = one / x;
            estimate = estimate * (two - x * estimate);
            return estimate;
        }

        Span<float> buffer = stackalloc float[Vector<float>.Count];
        for (int i = 0; i < Vector<float>.Count; i++)
            buffer[i] = ScalarFastReciprocal(x[i]);
        return new Vector<float>(buffer);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastSqrt(float x) => MathF.Sqrt(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastLn(float x) => MathF.Log(x);

    /// <summary>
    /// Computes the fast approximate reciprocal of a scalar float.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of 1/x.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastReciprocal(float x)
    {
        int i = BitConverter.SingleToInt32Bits(x);
        i = 0x7EF311C2 - i;
        float y = BitConverter.Int32BitsToSingle(i);
        y = y * (2.0f - x * y);
        return y;
    }

    /// <summary>
    /// Computes the fast approximate reciprocal for a Vector256 of floats (AVX2).
    /// </summary>
    /// <param name="x">The 256-bit input vector.</param>
    /// <returns>A Vector256 of approximate 1/x values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> FastReciprocal256(Vector256<float> x)
    {
        if (Avx2.IsSupported)
        {
            var estimate = Avx.Reciprocal(x);
            var two = Vector256.Create(2.0f);
            var refinement = Avx.Multiply(Avx.Multiply(x, estimate), estimate);
            return Avx.Multiply(estimate, Avx.Subtract(two, refinement));
        }

        return ScalarReciprocal256(x);
    }

    /// <summary>
    /// Scalar fallback for 256-bit reciprocal.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ScalarReciprocal256(Vector256<float> x)
    {
        var result = new float[8];
        var src = new float[8];
        x.CopyTo(src);
        for (int i = 0; i < 8; i++)
            result[i] = ScalarFastReciprocal(src[i]);
        return Vector256.Create(result[0], result[1], result[2], result[3],
                                result[4], result[5], result[6], result[7]);
    }

    /// <summary>
    /// Computes the fast approximate reciprocal square root for a Vector256 of floats (AVX2).
    /// </summary>
    /// <param name="x">The 256-bit input vector.</param>
    /// <returns>A Vector256 of approximate 1/sqrt(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> FastRsqrt256(Vector256<float> x)
    {
        if (Avx2.IsSupported)
        {
            return Avx.ReciprocalSqrt(x);
        }

        var result = new float[8];
        x.CopyTo(result);
        var output = new float[8];
        for (int i = 0; i < 8; i++)
            output[i] = ScalarFastRsqrt(result[i]);
        return Vector256.Create(output[0], output[1], output[2], output[3],
                                output[4], output[5], output[6], output[7]);
    }

    /// <summary>
    /// Computes the fast approximate sine of a value using a minimax polynomial.
    /// Valid input range: roughly [-2pi, 2pi] with reduced precision outside [-pi, pi].
    /// Maximum relative error: ~1e-4 in the primary range.
    /// </summary>
    /// <param name="x">The input angle in radians.</param>
    /// <returns>An approximation of sin(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastSin(float x)
    {
        if (Vector.IsHardwareAccelerated)
        {
            return FastSinVector(new Vector<float>(x))[0];
        }
        return ScalarFastSin(x);
    }

    /// <summary>
    /// Scalar fast sine approximation using a 5th-degree minimax polynomial.
    /// </summary>
    /// <param name="x">The input angle in radians.</param>
    /// <returns>An approximation of sin(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastSin(float x)
    {
        const float InvPi = 0.3183098861837907f;
        const float InvPiSq = InvPi * InvPi;

        float sign = 1.0f;
        if (x < 0)
        { x = -x; sign = -1.0f; }

        float n = (x + MathF.PI) * InvPi;
        float k = MathF.Round(n);
        float r = (x + MathF.PI) - k * MathF.PI;

        float r2 = r * r;
        float result = r + r * r2 * (ScalarSinC1 + r2 * (ScalarSinC2 + r2 * ScalarSinC3));

        float signK = k % 2.0f;
        if (signK < 0)
            signK += 2.0f;
        if (signK >= 1.0f)
            result = -result;

        return result * sign;
    }

    /// <summary>
    /// Computes the fast approximate sine for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector of angles in radians.</param>
    /// <returns>A vector of approximate sin(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastSinVector(Vector<float> x)
    {
        Vector<float> sign = Vector.ConditionalSelect(
            Vector.LessThan(x, Vector<float>.Zero),
            Vector.Negate(Vector<float>.One),
            Vector<float>.One);
        x = Vector.Abs(x);

        const float InvPi = 0.3183098861837907f;
        var invPi = new Vector<float>(InvPi);
        var pi = new Vector<float>(MathF.PI);

        var n = (x + pi) * invPi;
        var k = Vector.Round(n);
        var r = (x + pi) - k * pi;

        var r2 = r * r;
        var poly = r2 * SinCoeff3;
        poly = r2 * (poly + SinCoeff2);
        poly = r2 * (poly + SinCoeff1);
        var result = r + r * poly;

        var two = new Vector<float>(2.0f);
        var signK = k - Vector.Floor(k / two) * two;
        var flipMask = Vector.GreaterThanOrEqual(signK, Vector<float>.One);
        result = Vector.ConditionalSelect(flipMask, -result, result);

        return result * sign;
    }

    /// <summary>
    /// Computes the fast approximate cosine of a value using a minimax polynomial.
    /// </summary>
    /// <param name="x">The input angle in radians.</param>
    /// <returns>An approximation of cos(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastCos(float x)
    {
        if (Vector.IsHardwareAccelerated)
        {
            return FastCosVector(new Vector<float>(x))[0];
        }
        return ScalarFastCos(x);
    }

    /// <summary>
    /// Scalar fast cosine approximation using a 6th-degree minimax polynomial.
    /// </summary>
    /// <param name="x">The input angle in radians.</param>
    /// <returns>An approximation of cos(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastCos(float x)
    {
        const float InvPi = 0.3183098861837907f;

        x = MathF.Abs(x);

        float n = (x + MathF.PI) * InvPi;
        float k = MathF.Round(n);
        float r = (x + MathF.PI) - k * MathF.PI;

        float r2 = r * r;
        float result = 1.0f + r2 * (ScalarCosC1 + r2 * (ScalarCosC2 + r2 * ScalarCosC3));

        float signK = k % 2.0f;
        if (signK < 0)
            signK += 2.0f;
        if (signK >= 1.0f)
            result = -result;

        return result;
    }

    /// <summary>
    /// Computes the fast approximate cosine for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector of angles in radians.</param>
    /// <returns>A vector of approximate cos(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastCosVector(Vector<float> x)
    {
        x = Vector.Abs(x);

        const float InvPi = 0.3183098861837907f;
        var invPi = new Vector<float>(InvPi);
        var pi = new Vector<float>(MathF.PI);

        var n = (x + pi) * invPi;
        var k = Vector.Round(n);
        var r = (x + pi) - k * pi;

        var r2 = r * r;
        var poly = r2 * CosCoeff3;
        poly = r2 * (poly + CosCoeff2);
        poly = r2 * (poly + CosCoeff1);
        var result = Vector<float>.One + r2 * poly;

        var two = new Vector<float>(2.0f);
        var signK = k - Vector.Floor(k / two) * two;
        var flipMask = Vector.GreaterThanOrEqual(signK, Vector<float>.One);
        result = Vector.ConditionalSelect(flipMask, -result, result);

        return result;
    }

    /// <summary>
    /// Computes the fast approximate tangent using sin/cos division.
    /// </summary>
    /// <param name="x">The input angle in radians.</param>
    /// <returns>An approximation of tan(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastTan(float x)
    {
        float sin = FastSin(x);
        float cos = FastCos(x);
        if (MathF.Abs(cos) < 1e-10f)
            return sin > 0 ? float.PositiveInfinity : float.NegativeInfinity;
        return sin / cos;
    }

    /// <summary>
    /// Computes the fast approximate tangent for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector of angles in radians.</param>
    /// <returns>A vector of approximate tan(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastTanVector(Vector<float> x)
    {
        var sin = FastSinVector(x);
        var cos = FastCosVector(x);
        var epsilon = new Vector<float>(1e-10f);
        var safeCos = Vector.ConditionalSelect(
            Vector.LessThan(Vector.Abs(cos), epsilon),
            Vector<float>.One,
            cos);
        return sin / safeCos;
    }

    /// <summary>
    /// Computes the fast approximate exponential function (e^x) using Schraudolph's method.
    /// Maximum relative error: ~4% across the valid range [-87, 88].
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of e^x.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastExp(float x)
    {
        if (Vector.IsHardwareAccelerated)
        {
            return FastExpVector(new Vector<float>(x))[0];
        }
        return ScalarFastExp(x);
    }

    /// <summary>
    /// Scalar fast exponential using IEEE 754 bit manipulation (Schraudolph's method).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of e^x.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastExp(float x)
    {
        const float OneOverLn2 = 1.4426950408889634f;
        const float Bias = 127.0f;

        x *= OneOverLn2;
        float xf = MathF.Floor(x);
        float frac = x - xf;

        int i = (int)(Bias + xf);
        i = Math.Clamp(i, 0, 254);
        float f = BitConverter.Int32BitsToSingle(i << 23);

        float p = 1.0f + frac * (0.6931471805599453f + frac * 0.24022650695910071f);
        return f * p;
    }

    /// <summary>
    /// Computes the fast approximate exponential for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector of approximate e^x values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastExpVector(Vector<float> x)
    {
        const float OneOverLn2 = 1.4426950408889634f;
        const float Bias = 127.0f;

        var xScaled = x * new Vector<float>(OneOverLn2);
        var xFloor = Vector.Floor(xScaled);
        var frac = xScaled - xFloor;

        var bias = new Vector<float>(Bias);
        var i = Vector.ConvertToInt32(xFloor + bias);
        i = Vector.Max(i, Vector<int>.Zero);
        i = Vector.Min(i, new Vector<int>(254));

        Span<float> buffer = stackalloc float[Vector<float>.Count];
        for (int idx = 0; idx < Vector<float>.Count; idx++)
        {
            int bits = i[idx] << 23;
            buffer[idx] = BitConverter.Int32BitsToSingle(bits) *
                (1.0f + frac[idx] * (0.6931471805599453f + frac[idx] * 0.24022650695910071f));
        }

        return new Vector<float>(buffer);
    }

    /// <summary>
    /// Computes 2^x using fast approximation.
    /// </summary>
    /// <param name="x">The exponent.</param>
    /// <returns>An approximation of 2^x.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastPow2(float x)
    {
        return FastExp(x * Ln2);
    }

    /// <summary>
    /// Computes 2^x for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector of exponents.</param>
    /// <returns>A vector of approximate 2^x values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastPow2Vector(Vector<float> x)
    {
        return FastExpVector(x * new Vector<float>(Ln2));
    }

    /// <summary>
    /// Computes the fast approximate base-2 logarithm using IEEE 754 bit manipulation.
    /// Maximum relative error: ~2% in the normal range.
    /// </summary>
    /// <param name="x">The input value. Must be positive.</param>
    /// <returns>An approximation of log2(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastLog2(float x)
    {
        if (Vector.IsHardwareAccelerated)
        {
            return FastLog2Vector(new Vector<float>(x))[0];
        }
        return ScalarFastLog2(x);
    }

    /// <summary>
    /// Scalar fast base-2 logarithm using bit manipulation.
    /// </summary>
    /// <param name="x">The input value. Must be positive.</param>
    /// <returns>An approximation of log2(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastLog2(float x)
    {
        int i = BitConverter.SingleToInt32Bits(x);
        float exp = (float)((i >> 23) - 127);
        i &= 0x007FFFFF;
        i |= 0x3F800000;
        float mantissa = BitConverter.Int32BitsToSingle(i);
        return exp + mantissa - 1.0f;
    }

    /// <summary>
    /// Computes the fast approximate base-2 logarithm for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector of positive values.</param>
    /// <returns>A vector of approximate log2(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastLog2Vector(Vector<float> x) =>
        ScalarFastLog2Vector(x);

    /// <summary>
    /// Scalar fallback for vector log2.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> ScalarFastLog2Vector(Vector<float> x)
    {
        Span<float> buffer = stackalloc float[Vector<float>.Count];
        for (int i = 0; i < Vector<float>.Count; i++)
            buffer[i] = ScalarFastLog2(x[i]);
        return new Vector<float>(buffer);
    }

    /// <summary>
    /// Computes the fast approximate natural logarithm: ln(x) = log2(x) * ln(2).
    /// </summary>
    /// <param name="x">The input value. Must be positive.</param>
    /// <returns>An approximation of ln(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastLn(float x)
    {
        return FastLog2(x) * Ln2;
    }

    /// <summary>
    /// Computes the fast approximate natural logarithm for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector of positive values.</param>
    /// <returns>A vector of approximate ln(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastLnVector(Vector<float> x)
    {
        return FastLog2Vector(x) * new Vector<float>(Ln2);
    }

    /// <summary>
    /// Computes the fast approximate base-10 logarithm: log10(x) = log2(x) * log10(2).
    /// </summary>
    /// <param name="x">The input value. Must be positive.</param>
    /// <returns>An approximation of log10(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastLog10(float x)
    {
        return FastLog2(x) * 0.3010299956639812f;
    }

    /// <summary>
    /// Computes the fast approximate power function: x^y = exp(y * ln(x)).
    /// </summary>
    /// <param name="x">The base. Must be positive.</param>
    /// <param name="y">The exponent.</param>
    /// <returns>An approximation of x^y.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastPow(float x, float y)
    {
        if (x <= 0)
            return 0;
        return FastExp(y * FastLn(x));
    }

    /// <summary>
    /// Computes the fast approximate power function for SIMD vectors.
    /// </summary>
    /// <param name="x">The input vector of bases.</param>
    /// <param name="y">The exponent vector.</param>
    /// <returns>A vector of approximate x^y values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastPowVector(Vector<float> x, Vector<float> y)
    {
        var positive = Vector.GreaterThan(x, Vector<float>.Zero);
        var result = FastExpVector(y * FastLnVector(x));
        return Vector.ConditionalSelect(positive, result, Vector<float>.Zero);
    }

    /// <summary>
    /// Computes the fast approximate sigmoid function: 1 / (1 + exp(-x)).
    /// Uses a piecewise linear/quadratic approximation for maximum throughput.
    /// Maximum error: ~0.001 in the range [-5, 5].
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of sigmoid(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastSigmoid(float x)
    {
        if (Vector.IsHardwareAccelerated)
        {
            return FastSigmoidVector(new Vector<float>(x))[0];
        }
        return ScalarFastSigmoid(x);
    }

    /// <summary>
    /// Scalar fast sigmoid approximation.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of 1 / (1 + exp(-x)).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastSigmoid(float x)
    {
        if (x >= 4.0f)
            return 1.0f;
        if (x <= -4.0f)
            return 0.0f;

        if (x >= -2.0f && x <= 2.0f)
        {
            return 0.5f + x * (0.25f - x * x * 0.020833333f);
        }

        return 1.0f / (1.0f + ScalarFastExp(-x));
    }

    /// <summary>
    /// Computes the fast approximate sigmoid for a SIMD vector.
    /// Uses the piecewise approximation for values in [-4, 4] and saturates outside.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector of approximate sigmoid(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastSigmoidVector(Vector<float> x)
    {
        var zero = Vector<float>.Zero;
        var one = Vector<float>.One;
        var half = new Vector<float>(0.5f);
        var q1 = new Vector<float>(0.25f);
        var q2 = new Vector<float>(0.020833333f);

        var x2 = x * x;
        var poly = half + x * (q1 - x2 * q2);

        var lowMask = Vector.LessThan(x, new Vector<float>(-4.0f));
        var highMask = Vector.GreaterThan(x, new Vector<float>(4.0f));
        var result = Vector.ConditionalSelect(lowMask, zero, poly);
        result = Vector.ConditionalSelect(highMask, one, result);

        return result;
    }

    /// <summary>
    /// Computes the fast approximate SiLU (Swish) activation: x * sigmoid(x).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of x * sigmoid(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastSilu(float x)
    {
        return x * FastSigmoid(x);
    }

    /// <summary>
    /// Computes the fast approximate SiLU for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector of approximate SiLU(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastSiluVector(Vector<float> x)
    {
        return x * FastSigmoidVector(x);
    }

    /// <summary>
    /// Computes the fast approximate GELU (Gaussian Error Linear Unit).
    /// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of GELU(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastGelu(float x)
    {
        if (Vector.IsHardwareAccelerated)
        {
            return FastGeluVector(new Vector<float>(x))[0];
        }
        return ScalarFastGelu(x);
    }

    /// <summary>
    /// Scalar fast GELU approximation.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of GELU(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastGelu(float x)
    {
        float x3 = x * x * x;
        float inner = GELUSqrt2OverPi * (x + GELUCoeff * x3);
        float tanhVal = FastTanh(inner);
        return 0.5f * x * (1.0f + tanhVal);
    }

    /// <summary>
    /// Computes the fast approximate GELU for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector of approximate GELU(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastGeluVector(Vector<float> x)
    {
        var sqrt2opi = new Vector<float>(GELUSqrt2OverPi);
        var coeff = new Vector<float>(GELUCoeff);
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;

        var x3 = x * x * x;
        var inner = sqrt2opi * (x + coeff * x3);
        var tanhVal = FastTanhVector(inner);

        return half * x * (one + tanhVal);
    }

    /// <summary>
    /// Computes the fast approximate hyperbolic tangent using a rational polynomial.
    /// Maximum error: ~0.001 in the range [-4, 4].
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of tanh(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastTanh(float x)
    {
        if (Vector.IsHardwareAccelerated)
        {
            return FastTanhVector(new Vector<float>(x))[0];
        }
        return ScalarFastTanh(x);
    }

    /// <summary>
    /// Scalar fast tanh approximation using a Padé-like rational polynomial.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of tanh(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ScalarFastTanh(float x)
    {
        if (x >= 4.0f)
            return 1.0f;
        if (x <= -4.0f)
            return -1.0f;

        float x2 = x * x;
        float num = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return num / den;
    }

    /// <summary>
    /// Computes the fast approximate tanh for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector of approximate tanh(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastTanhVector(Vector<float> x)
    {
        var one = Vector<float>.One;
        var negOne = -one;

        var absX = Vector.Abs(x);
        var sign = Vector.ConditionalSelect(
            Vector.LessThan(x, Vector<float>.Zero),
            negOne, one);

        var x2 = absX * absX;
        var numCoeff3 = new Vector<float>(1.0f);
        var numCoeff2 = new Vector<float>(378.0f);
        var numCoeff1 = new Vector<float>(17325.0f);
        var numCoeff0 = new Vector<float>(135135.0f);

        var denCoeff3 = new Vector<float>(28.0f);
        var denCoeff2 = new Vector<float>(3150.0f);
        var denCoeff1 = new Vector<float>(62370.0f);
        var denCoeff0 = new Vector<float>(135135.0f);

        var num = absX * (numCoeff0 + x2 * (numCoeff1 + x2 * (numCoeff2 + x2 * numCoeff3)));
        var den = denCoeff0 + x2 * (denCoeff1 + x2 * (denCoeff2 + x2 * denCoeff3));

        var result = num / den;
        result = Vector.Min(result, one);
        result = Vector.Max(result, negOne);

        return result * sign;
    }

    /// <summary>
    /// Computes the fast approximate ReLU: max(0, x).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>max(0, x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastRelu(float x)
    {
        return MathF.Max(0, x);
    }

    /// <summary>
    /// Computes the fast approximate ReLU for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector of ReLU(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastReluVector(Vector<float> x)
    {
        return Vector.Max(Vector<float>.Zero, x);
    }

    /// <summary>
    /// Computes the fast approximate Leaky ReLU: x &gt; 0 ? x : alpha * x.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <param name="alpha">The slope for negative values (default 0.01).</param>
    /// <returns>The LeakyReLU output.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastLeakyRelu(float x, float alpha = 0.01f)
    {
        return x > 0 ? x : alpha * x;
    }

    /// <summary>
    /// Computes the fast approximate Leaky ReLU for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <param name="alpha">The slope for negative values (default 0.01).</param>
    /// <returns>A vector of LeakyReLU(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastLeakyReluVector(Vector<float> x, float alpha = 0.01f)
    {
        var a = new Vector<float>(alpha);
        var mask = Vector.GreaterThan(x, Vector<float>.Zero);
        return Vector.ConditionalSelect(mask, x, x * a);
    }

    /// <summary>
    /// Computes the fast approximate softplus: ln(1 + exp(x)).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of softplus(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastSoftplus(float x)
    {
        if (x > 20.0f)
            return x;
        if (x < -20.0f)
            return 0.0f;
        return ScalarFastLn(1.0f + ScalarFastExp(x));
    }

    /// <summary>
    /// Computes the fast approximate Mish activation: x * tanh(softplus(x)).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of Mish(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastMish(float x)
    {
        float sp = FastSoftplus(x);
        return x * FastTanh(sp);
    }

    /// <summary>
    /// Computes the fast approximate softsign: x / (1 + |x|).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of softsign(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastSoftsign(float x)
    {
        return x / (1.0f + MathF.Abs(x));
    }

    /// <summary>
    /// Computes the fast approximate softsign for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector of softsign(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastSoftsignVector(Vector<float> x)
    {
        return x / (Vector<float>.One + Vector.Abs(x));
    }

    /// <summary>
    /// Computes the fast approximate ELU (Exponential Linear Unit).
    /// Returns x if x &gt; 0, alpha * (exp(x) - 1) if x &lt;= 0.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <param name="alpha">Scale for negative inputs (default 1.0).</param>
    /// <returns>An approximation of ELU(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastElu(float x, float alpha = 1.0f)
    {
        return x > 0 ? x : alpha * (ScalarFastExp(x) - 1.0f);
    }

    /// <summary>
    /// Computes the fast approximate ELU for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <param name="alpha">Scale for negative inputs (default 1.0).</param>
    /// <returns>A vector of ELU(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastEluVector(Vector<float> x, float alpha = 1.0f)
    {
        var a = new Vector<float>(alpha);
        var one = Vector<float>.One;
        var mask = Vector.GreaterThan(x, Vector<float>.Zero);
        var expX = FastExpVector(x);
        var negResult = a * (expX - one);
        return Vector.ConditionalSelect(mask, x, negResult);
    }

    /// <summary>
    /// Computes the fast approximate SELU: scale * ELU(x) where scale ~1.0507 and alpha ~1.6733.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>An approximation of SELU(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastSelu(float x)
    {
        const float SeluAlpha = 1.6732632423543772f;
        const float SeluScale = 1.0507009873554805f;
        return SeluScale * FastElu(x, SeluAlpha);
    }

    /// <summary>
    /// Computes the fast approximate CELU (Continuously differentiable ELU).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <param name="alpha">Scale for negative inputs (default 1.0).</param>
    /// <returns>An approximation of CELU(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastCelu(float x, float alpha = 1.0f)
    {
        if (x > 0)
            return x;
        return alpha * (ScalarFastExp(x / alpha) - 1.0f);
    }

    /// <summary>
    /// Computes the fast approximate square root using hardware rsqrt with refinement.
    /// </summary>
    /// <param name="x">The input value. Must be non-negative.</param>
    /// <returns>An approximation of sqrt(x).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastSqrt(float x)
    {
        if (x <= 0)
            return 0;
        return x * ScalarFastRsqrt(x);
    }

    /// <summary>
    /// Computes the fast approximate square root for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector of non-negative values.</param>
    /// <returns>A vector of approximate sqrt(x) values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastSqrtVector(Vector<float> x)
    {
        var zero = Vector<float>.Zero;
        var mask = Vector.GreaterThan(x, zero);
        var result = x * FastRsqrtVector(x);
        return Vector.ConditionalSelect(mask, result, zero);
    }

    /// <summary>
    /// Computes the fast approximate absolute value for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A vector of |x| values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastAbsVector(Vector<float> x)
    {
        return Vector.Abs(x);
    }

    /// <summary>
    /// Computes the fast approximate clamp for a SIMD vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <param name="min">The minimum bound vector.</param>
    /// <param name="max">The maximum bound vector.</param>
    /// <returns>A vector of clamped values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> FastClampVector(Vector<float> x, Vector<float> min, Vector<float> max)
    {
        return Vector.Min(Vector.Max(x, min), max);
    }

    /// <summary>
    /// Computes the fast approximate softmax for a span of values.
    /// Uses the max-subtract trick for numerical stability.
    /// </summary>
    /// <param name="values">The input values. Overwritten with softmax output.</param>
    public static void FastSoftmaxInPlace(Span<float> values)
    {
        if (values.Length == 0)
            return;

        float maxVal = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > maxVal)
                maxVal = values[i];

        if (Vector.IsHardwareAccelerated && values.Length >= Vector<float>.Count)
        {
            int vecCount = Vector<float>.Count;
            int simdEnd = values.Length - (values.Length % vecCount);

            var sumVec = Vector<float>.Zero;
            var maxVec = new Vector<float>(maxVal);

            for (int i = 0; i < simdEnd; i += vecCount)
            {
                var v = new Vector<float>(values.Slice(i));
                var e = FastExpVector(v - maxVec);
                e.CopyTo(values.Slice(i));
                sumVec += e;
            }

            float sum = 0;
            for (int i = 0; i < vecCount; i++)
                sum += sumVec[i];

            for (int i = simdEnd; i < values.Length; i++)
            {
                float e = ScalarFastExp(values[i] - maxVal);
                values[i] = e;
                sum += e;
            }

            float invSum = ScalarFastReciprocal(sum);
            for (int i = 0; i < values.Length; i++)
                values[i] *= invSum;
        }
        else
        {
            float sum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                float e = ScalarFastExp(values[i] - maxVal);
                values[i] = e;
                sum += e;
            }

            float invSum = ScalarFastReciprocal(sum);
            for (int i = 0; i < values.Length; i++)
                values[i] *= invSum;
        }
    }

    /// <summary>
    /// Computes the fast approximate layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta.
    /// </summary>
    /// <param name="input">Input vector to normalize.</param>
    /// <param name="gamma">Scale parameter.</param>
    /// <param name="beta">Bias parameter.</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <param name="output">Output buffer for normalized values.</param>
    public static void FastLayerNorm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        Span<float> output,
        float epsilon = 1e-5f)
    {
        int length = input.Length;
        if (gamma.Length != length || beta.Length != length || output.Length != length)
            throw new ArgumentException("Input, gamma, beta, and output must have the same length.");

        float mean = 0;
        for (int i = 0; i < length; i++)
            mean += input[i];
        mean /= length;

        float variance = 0;
        for (int i = 0; i < length; i++)
        {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= length;

        float invStd = ScalarFastRsqrt(variance + epsilon);

        if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
        {
            int vecCount = Vector<float>.Count;
            int simdEnd = length - (length % vecCount);
            var meanVec = new Vector<float>(mean);
            var invStdVec = new Vector<float>(invStd);

            for (int i = 0; i < simdEnd; i += vecCount)
            {
                var x = new Vector<float>(input.Slice(i));
                var g = new Vector<float>(gamma.Slice(i));
                var b = new Vector<float>(beta.Slice(i));
                var normalized = (x - meanVec) * invStdVec;
                var result = normalized * g + b;
                result.CopyTo(output.Slice(i));
            }

            for (int i = simdEnd; i < length; i++)
            {
                float normalized = (input[i] - mean) * invStd;
                output[i] = normalized * gamma[i] + beta[i];
            }
        }
        else
        {
            for (int i = 0; i < length; i++)
            {
                float normalized = (input[i] - mean) * invStd;
                output[i] = normalized * gamma[i] + beta[i];
            }
        }
    }

    /// <summary>
    /// Computes the fast approximate batch normalization.
    /// </summary>
    /// <param name="input">Input values.</param>
    /// <param name="runningMean">Running mean.</param>
    /// <param name="runningVar">Running variance.</param>
    /// <param name="gamma">Scale parameter.</param>
    /// <param name="beta">Bias parameter.</param>
    /// <param name="output">Output buffer.</param>
    /// <param name="epsilon">Numerical stability constant.</param>
    public static void FastBatchNorm(
        ReadOnlySpan<float> input,
        float runningMean,
        float runningVar,
        float gamma,
        float beta,
        Span<float> output,
        float epsilon = 1e-5f)
    {
        int length = input.Length;
        if (output.Length != length)
            throw new ArgumentException("Output must have the same length as input.");

        float invStd = ScalarFastRsqrt(runningVar + epsilon);

        if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
        {
            int vecCount = Vector<float>.Count;
            int simdEnd = length - (length % vecCount);
            var meanVec = new Vector<float>(runningMean);
            var invStdVec = new Vector<float>(invStd);
            var gammaVec = new Vector<float>(gamma);
            var betaVec = new Vector<float>(beta);

            for (int i = 0; i < simdEnd; i += vecCount)
            {
                var x = new Vector<float>(input.Slice(i));
                var result = (x - meanVec) * invStdVec * gammaVec + betaVec;
                result.CopyTo(output.Slice(i));
            }

            for (int i = simdEnd; i < length; i++)
            {
                output[i] = (input[i] - runningMean) * invStd * gamma + beta;
            }
        }
        else
        {
            for (int i = 0; i < length; i++)
                output[i] = (input[i] - runningMean) * invStd * gamma + beta;
        }
    }

    /// <summary>
    /// Computes the dot product of two spans using SIMD reduction.
    /// </summary>
    /// <param name="a">First input vector.</param>
    /// <param name="b">Second input vector.</param>
    /// <returns>The dot product a · b.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastDotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Input spans must have the same length.");

        int length = a.Length;
        float sum = 0;

        if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
        {
            int vecCount = Vector<float>.Count;
            int simdEnd = length - (length % vecCount);
            var sumVec = Vector<float>.Zero;

            for (int i = 0; i < simdEnd; i += vecCount)
            {
                var va = new Vector<float>(a.Slice(i));
                var vb = new Vector<float>(b.Slice(i));
                sumVec += va * vb;
            }

            for (int i = 0; i < vecCount; i++)
                sum += sumVec[i];

            for (int i = simdEnd; i < length; i++)
                sum += a[i] * b[i];
        }
        else
        {
            for (int i = 0; i < length; i++)
                sum += a[i] * b[i];
        }

        return sum;
    }

    /// <summary>
    /// Computes the L2 norm (Euclidean length) of a span using SIMD.
    /// </summary>
    /// <param name="values">The input vector.</param>
    /// <returns>The L2 norm of the vector.</returns>
    public static float FastL2Norm(ReadOnlySpan<float> values)
    {
        float sumSq = 0;
        int length = values.Length;

        if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
        {
            int vecCount = Vector<float>.Count;
            int simdEnd = length - (length % vecCount);
            var sumVec = Vector<float>.Zero;

            for (int i = 0; i < simdEnd; i += vecCount)
            {
                var v = new Vector<float>(values.Slice(i));
                sumVec += v * v;
            }

            for (int i = 0; i < vecCount; i++)
                sumSq += sumVec[i];

            for (int i = simdEnd; i < length; i++)
                sumSq += values[i] * values[i];
        }
        else
        {
            for (int i = 0; i < length; i++)
                sumSq += values[i] * values[i];
        }

        return ScalarFastSqrt(sumSq);
    }

    /// <summary>
    /// Normalizes a span of values to unit length (L2 normalization) in-place using SIMD.
    /// </summary>
    /// <param name="values">The values to normalize. Overwritten with the unit vector.</param>
    /// <returns>The original L2 norm before normalization.</returns>
    public static float FastL2NormalizeInPlace(Span<float> values)
    {
        float norm = FastL2Norm(values);
        if (norm < 1e-10f)
            return 0;

        float invNorm = ScalarFastReciprocal(norm);
        int length = values.Length;

        if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
        {
            int vecCount = Vector<float>.Count;
            int simdEnd = length - (length % vecCount);
            var invNormVec = new Vector<float>(invNorm);

            for (int i = 0; i < simdEnd; i += vecCount)
            {
                var v = new Vector<float>(values.Slice(i));
                (v * invNormVec).CopyTo(values.Slice(i));
            }

            for (int i = simdEnd; i < length; i++)
                values[i] *= invNorm;
        }
        else
        {
            for (int i = 0; i < length; i++)
                values[i] *= invNorm;
        }

        return norm;
    }
}
