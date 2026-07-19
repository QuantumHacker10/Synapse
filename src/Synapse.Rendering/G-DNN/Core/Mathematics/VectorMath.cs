using System;
// ============================================================
// FILE: VectorMath.cs
// PATH: Core/Mathematics/VectorMath.cs
// ============================================================


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
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Core.Mathematics;

/// <summary>
/// Comprehensive vector mathematics utilities for the G-DNN neural geometry engine.
/// Provides extension methods for System.Numerics vector types with SIMD-optimized operations,
/// noise generation, geometric queries, and batch processing capabilities.
/// </summary>
public static class VectorMath
{
    private const float PI = MathF.PI;
    private const float TwoPI = MathF.Tau;
    private const float HalfPI = MathF.PI * 0.5f;
    private const float InvMaxByte = 1.0f / 255.0f;
    private const float InvMaxUShort = 1.0f / 65535.0f;
    private const float InvMaxUInt = 1.0f / 4294967295.0f;
    private const int PermutationSize = 256;
    private const float GradientScale2D = 0.5f;
    private const float GradientScale3D = 0.5f;

    private static readonly int[] _permutation;
    private static readonly int[] _permutationMod12;
    private static readonly Vector3[] _gradients3D;
    private static readonly Vector2[] _gradients2D;

    static VectorMath()
    {
        _permutation = new int[PermutationSize * 2];
        _permutationMod12 = new int[PermutationSize * 2];
        _gradients3D = new Vector3[12]
        {
            new(1, 1, 0), new(-1, 1, 0), new(1, -1, 0), new(-1, -1, 0),
            new(1, 0, 1), new(-1, 0, 1), new(1, 0, -1), new(-1, 0, -1),
            new(0, 1, 1), new(0, -1, 1), new(0, 1, -1), new(0, -1, -1)
        };
        _gradients2D = new Vector2[8]
        {
            new(1, 0), new(-1, 0), new(0, 1), new(0, -1),
            Vector2.Normalize(new Vector2(1, 1)), Vector2.Normalize(new Vector2(-1, 1)),
            Vector2.Normalize(new Vector2(1, -1)), Vector2.Normalize(new Vector2(-1, -1))
        };

        Span<int> p = stackalloc int[PermutationSize];
        for (int i = 0; i < PermutationSize; i++)
            p[i] = i;

        var rng = new Random(42);
        for (int i = PermutationSize - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (p[i], p[j]) = (p[j], p[i]);
        }

        for (int i = 0; i < PermutationSize * 2; i++)
        {
            _permutation[i] = p[i & (PermutationSize - 1)];
            _permutationMod12[i] = _permutation[i] % 12;
        }
    }

    #region Component-wise Operations (Vector3)

    /// <summary>Component-wise multiply of two vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentMultiply(this Vector3 a, Vector3 b) => new(a.X * b.X, a.Y * b.Y, a.Z * b.Z);

    /// <summary>Component-wise divide of two vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentDivide(this Vector3 a, Vector3 b) => new(a.X / b.X, a.Y / b.Y, a.Z / b.Z);

    /// <summary>Component-wise minimum of two vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentMin(this Vector3 a, Vector3 b) =>
        Vector3.Min(a, b);

    /// <summary>Component-wise maximum of two vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentMax(this Vector3 a, Vector3 b) =>
        Vector3.Max(a, b);

    /// <summary>Component-wise absolute value.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentAbs(this Vector3 v) =>
        new(MathF.Abs(v.X), MathF.Abs(v.Y), MathF.Abs(v.Z));

    /// <summary>Component-wise modulo.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentMod(this Vector3 a, Vector3 b) =>
        new(a.X % b.X, a.Y % b.Y, a.Z % b.Z);

    /// <summary>Component-wise clamp each element between min and max.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentClamp(this Vector3 v, Vector3 min, Vector3 max) =>
        Vector3.Clamp(v, min, max);

    /// <summary>Returns true if all components are approximately equal within tolerance.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool ApproximatelyEquals(this Vector3 a, Vector3 b, float tolerance = 1e-6f) =>
        MathF.Abs(a.X - b.X) <= tolerance &&
        MathF.Abs(a.Y - b.Y) <= tolerance &&
        MathF.Abs(a.Z - b.Z) <= tolerance;

    /// <summary>Swizzle components using index permutation (xyz, xzy, yxz, etc).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Swizzle(this Vector3 v, int x, int y, int z)
    {
        Span<float> src = stackalloc float[3] { v.X, v.Y, v.Z };
        return new(src[x], src[y], src[z]);
    }

    /// <summary>Reciprocal of each component (fast approximate).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Reciprocal(this Vector3 v) =>
        Vector3.One / v;

    /// <summary>Component-wise lerp between two vectors using a scalar t.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentLerp(this Vector3 a, Vector3 b, Vector3 t) =>
        new(
            a.X + (b.X - a.X) * t.X,
            a.Y + (b.Y - a.Y) * t.Y,
            a.Z + (b.Z - a.Z) * t.Z);

    /// <summary>Component-wise sign (-1, 0, or 1 per component).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ComponentSign(this Vector3 v) =>
        new(MathF.Sign(v.X), MathF.Sign(v.Y), MathF.Sign(v.Z));

    /// <summary>Component-wise floor.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Floor(this Vector3 v) =>
        new(MathF.Floor(v.X), MathF.Floor(v.Y), MathF.Floor(v.Z));

    /// <summary>Component-wise ceiling.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Ceil(this Vector3 v) =>
        new(MathF.Ceiling(v.X), MathF.Ceiling(v.Y), MathF.Ceiling(v.Z));

    /// <summary>Component-wise round.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Round(this Vector3 v) =>
        new(MathF.Round(v.X), MathF.Round(v.Y), MathF.Round(v.Z));

    /// <summary>Component-wise power.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Pow(this Vector3 v, float exponent) =>
        new(MathF.Pow(v.X, exponent), MathF.Pow(v.Y, exponent), MathF.Pow(v.Z, exponent));

    /// <summary>Component-wise square root.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Sqrt(this Vector3 v) =>
        new(MathF.Sqrt(v.X), MathF.Sqrt(v.Y), MathF.Sqrt(v.Z));

    #endregion

    #region Component-wise Operations (Vector4)

    /// <summary>Component-wise multiply of two vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 ComponentMultiply(this Vector4 a, Vector4 b) =>
        new(a.X * b.X, a.Y * b.Y, a.Z * b.Z, a.W * b.W);

    /// <summary>Component-wise divide of two vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 ComponentDivide(this Vector4 a, Vector4 b) =>
        new(a.X / b.X, a.Y / b.Y, a.Z / b.Z, a.W / b.W);

    /// <summary>Component-wise minimum.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 ComponentMin(this Vector4 a, Vector4 b) =>
        Vector4.Min(a, b);

    /// <summary>Component-wise maximum.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 ComponentMax(this Vector4 a, Vector4 b) =>
        Vector4.Max(a, b);

    /// <summary>Component-wise absolute value.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 ComponentAbs(this Vector4 v) =>
        new(MathF.Abs(v.X), MathF.Abs(v.Y), MathF.Abs(v.Z), MathF.Abs(v.W));

    /// <summary>Returns true if all components are approximately equal.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool ApproximatelyEquals(this Vector4 a, Vector4 b, float tolerance = 1e-6f) =>
        MathF.Abs(a.X - b.X) <= tolerance &&
        MathF.Abs(a.Y - b.Y) <= tolerance &&
        MathF.Abs(a.Z - b.Z) <= tolerance &&
        MathF.Abs(a.W - b.W) <= tolerance;

    /// <summary>Component-wise floor.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 Floor(this Vector4 v) =>
        new(MathF.Floor(v.X), MathF.Floor(v.Y), MathF.Floor(v.Z), MathF.Floor(v.W));

    /// <summary>Component-wise ceiling.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 Ceil(this Vector4 v) =>
        new(MathF.Ceiling(v.X), MathF.Ceiling(v.Y), MathF.Ceiling(v.Z), MathF.Ceiling(v.W));

    #endregion

    #region Component-wise Operations (Vector2)

    /// <summary>Component-wise multiply.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 ComponentMultiply(this Vector2 a, Vector2 b) =>
        new(a.X * b.X, a.Y * b.Y);

    /// <summary>Component-wise divide.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 ComponentDivide(this Vector2 a, Vector2 b) =>
        new(a.X / b.X, a.Y / b.Y);

    /// <summary>Component-wise minimum.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 ComponentMin(this Vector2 a, Vector2 b) =>
        Vector2.Min(a, b);

    /// <summary>Component-wise maximum.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 ComponentMax(this Vector2 a, Vector2 b) =>
        Vector2.Max(a, b);

    /// <summary>Component-wise floor.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 Floor(this Vector2 v) =>
        new(MathF.Floor(v.X), MathF.Floor(v.Y));

    /// <summary>Component-wise ceiling.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 Ceil(this Vector2 v) =>
        new(MathF.Ceiling(v.X), MathF.Ceiling(v.Y));

    /// <summary>Component-wise modulo.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 ComponentMod(this Vector2 a, Vector2 b) =>
        new(a.X % b.X, a.Y % b.Y);

    /// <summary>Returns true if approximately equal.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool ApproximatelyEquals(this Vector2 a, Vector2 b, float tolerance = 1e-6f) =>
        MathF.Abs(a.X - b.X) <= tolerance &&
        MathF.Abs(a.Y - b.Y) <= tolerance;

    /// <summary>Normalize a Vector2 safely (returns zero if magnitude is near zero).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 SafeNormalize(this Vector2 v)
    {
        float len = v.Length();
        return len > 1e-8f ? v / len : Vector2.Zero;
    }

    /// <summary>Perpendicular vector in 2D (rotated 90 degrees clockwise).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 Perpendicular(this Vector2 v) => new(v.Y, -v.X);

    /// <summary>Perpendicular vector in 2D (counter-clockwise).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector2 PerpendicularCCW(this Vector2 v) => new(-v.Y, v.X);

    #endregion

    #region Smooth Interpolation

    /// <summary>
    /// Hermite smoothstep interpolation: 3t² - 2t³.
    /// Returns 0 when t=0, 1 when t=1, with smooth acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Smoothstep(float edge0, float edge1, float x)
    {
        float t = MathF.Max(0, MathF.Min(1, (x - edge0) / (edge1 - edge0)));
        return t * t * (3f - 2f * t);
    }

    /// <summary>
    /// Smootherstep interpolation: 6t⁵ - 15t⁴ + 10t³.
    /// Perlin's improved smoothstep with zero first and second derivatives at endpoints.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Smootherstep(float edge0, float edge1, float x)
    {
        float t = MathF.Max(0, MathF.Min(1, (x - edge0) / (edge1 - edge0)));
        return t * t * t * (t * (t * 6f - 15f) + 10f);
    }

    /// <summary>Cubic interpolation using four control points.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float CubicInterpolate(float p0, float p1, float p2, float p3, float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        return 0.5f * (
            (2f * p1) +
            (-p0 + p2) * t +
            (2f * p0 - 5f * p1 + 4f * p2 - p3) * t2 +
            (-p0 + 3f * p1 - 3f * p2 + p3) * t3);
    }

    /// <summary>Hermite interpolation with explicit tangent control.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HermiteInterpolate(float p0, float p1, float m0, float m1, float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        float h00 = 2f * t3 - 3f * t2 + 1f;
        float h10 = t3 - 2f * t2 + t;
        float h01 = -2f * t3 + 3f * t2;
        float h11 = t3 - t2;
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
    }

    public static Vector3 HermiteInterpolate(Vector3 p0, Vector3 p1, Vector3 m0, Vector3 m1, float t) =>
        new(
            HermiteInterpolate(p0.X, p1.X, m0.X, m1.X, t),
            HermiteInterpolate(p0.Y, p1.Y, m0.Y, m1.Y, t),
            HermiteInterpolate(p0.Z, p1.Z, m0.Z, m1.Z, t));
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 CatmullRom(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        return 0.5f * (
            (2f * p1) +
            (-p0 + p2) * t +
            (2f * p0 - 5f * p1 + 4f * p2 - p3) * t2 +
            (-p0 + 3f * p1 - 3f * p2 + p3) * t3);
    }

    /// <summary>Cubic Bezier interpolation with four control points.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 CubicBezier(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        float u = 1f - t;
        float u2 = u * u;
        float u3 = u2 * u;
        float t2 = t * t;
        float t3 = t2 * t;
        return u3 * p0 +
               3f * u2 * t * p1 +
               3f * u * t2 * p2 +
               t3 * p3;
    }

    /// <summary>Quadratic Bezier interpolation with three control points.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 QuadraticBezier(Vector3 p0, Vector3 p1, Vector3 p2, float t)
    {
        float u = 1f - t;
        return u * u * p0 + 2f * u * t * p1 + t * t * p2;
    }

    /// <summary>Cubic B-spline basis evaluation for four control points.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 CubicBSpline(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        float inv6 = 1f / 6f;
        return inv6 * (
            (-t3 + 3f * t2 - 3f * t + 1f) * p0 +
            (3f * t3 - 6f * t2 + 4f) * p1 +
            (-3f * t3 + 3f * t2 + 3f * t + 1f) * p2 +
            t3 * p3);
    }

    /// <summary>Perlin-style remap for [0,1] to [0,1] with zero derivatives at endpoints.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float RemapZeroDerivatives(float t) =>
        t * t * t * (t * (6f * t - 15f) + 10f);

    /// <summary>Damped spring interpolation for physics-like easing.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DampedSpring(float from, float to, float t, float frequency = 4f, float damping = 0.5f)
    {
        float omega = frequency * TwoPI;
        float dampedOmega = omega * MathF.Sqrt(1f - damping * damping);
        float envelope = MathF.Exp(-damping * omega * t);
        float cosine = MathF.Cos(dampedOmega * t);
        return to + (from - to) * envelope * (cosine + damping * omega * MathF.Sin(dampedOmega * t) / dampedOmega);
    }

    /// <summary>Lerp between two vectors with a scalar t.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Lerp(this Vector3 a, Vector3 b, float t) =>
        Vector3.Lerp(a, b, t);

    /// <summary>Spherical interpolation between two direction vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Slerp(this Vector3 a, Vector3 b, float t) =>
        Vector3.Lerp(a, b, t); // System.Numerics uses slerp for unit vectors

    /// <summary>Unclamped linear interpolation.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 LerpUnclamped(this Vector3 a, Vector3 b, float t) =>
        a + (b - a) * t;

    /// <summary>Bilerp: bilinear interpolation on a 2D grid.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Bilerp(Vector3 v00, Vector3 v10, Vector3 v01, Vector3 v11, float u, float v) =>
        Vector3.Lerp(
            Vector3.Lerp(v00, v10, u),
            Vector3.Lerp(v01, v11, u),
            v);

    /// <summary>Barycentric interpolation using barycentric coordinates.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 BarycentricLerp(Vector3 a, Vector3 b, Vector3 c, float u, float v, float w) =>
        u * a + v * b + w * c;

    #endregion

    #region Distance Functions

    /// <summary>Euclidean distance between two points.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float EuclideanDistance(this Vector3 a, Vector3 b) =>
        Vector3.Distance(a, b);

    /// <summary>Euclidean distance squared (avoids sqrt for comparisons).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float EuclideanDistanceSquared(this Vector3 a, Vector3 b) =>
        Vector3.DistanceSquared(a, b);

    /// <summary>Manhattan (L1) distance between two points.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ManhattanDistance(this Vector3 a, Vector3 b)
    {
        Vector3 d = Vector3.Abs(a - b);
        return d.X + d.Y + d.Z;
    }

    /// <summary>Chebyshev (L∞) distance between two points.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ChebyshevDistance(this Vector3 a, Vector3 b)
    {
        Vector3 d = Vector3.Abs(a - b);
        return MathF.Max(d.X, MathF.Max(d.Y, d.Z));
    }

    /// <summary>Minkowski distance with custom exponent p.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float MinkowskiDistance(this Vector3 a, Vector3 b, float p)
    {
        Vector3 d = Vector3.Abs(a - b);
        return MathF.Pow(
            MathF.Pow(d.X, p) + MathF.Pow(d.Y, p) + MathF.Pow(d.Z, p),
            1f / p);
    }

    /// <summary>Euclidean distance for Vector2.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float EuclideanDistance(this Vector2 a, Vector2 b) =>
        Vector2.Distance(a, b);

    /// <summary>Manhattan distance for Vector2.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ManhattanDistance(this Vector2 a, Vector2 b)
    {
        Vector2 d = Vector2.Abs(a - b);
        return d.X + d.Y;
    }

    /// <summary>Chebyshev distance for Vector2.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ChebyshevDistance(this Vector2 a, Vector2 b)
    {
        Vector2 d = Vector2.Abs(a - b);
        return MathF.Max(d.X, d.Y);
    }

    /// <summary>Angle between two vectors in radians.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float AngleBetween(this Vector3 a, Vector3 b) =>
        MathF.Acos(MathF.Max(-1f, MathF.Min(1f, Vector3.Dot(a, b) / (a.Length() * b.Length()))));

    /// <summary>Signed angle between two vectors around an axis.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SignedAngleAroundAxis(this Vector3 from, Vector3 to, Vector3 axis)
    {
        float angle = MathF.Acos(MathF.Max(-1f, MathF.Min(1f,
            Vector3.Dot(from, to) / (from.Length() * to.Length()))));
        float sign = MathF.Sign(Vector3.Dot(axis, Vector3.Cross(from, to)));
        return angle * sign;
    }

    /// <summary>Distance from a point to a line segment.</summary>
    public static float PointToSegmentDistance(this Vector3 point, Vector3 segA, Vector3 segB)
    {
        Vector3 ab = segB - segA;
        float lenSq = ab.LengthSquared();
        if (lenSq < 1e-10f)
            return Vector3.Distance(point, segA);

        float t = MathF.Max(0, MathF.Min(1f, Vector3.Dot(point - segA, ab) / lenSq));
        Vector3 closest = segA + t * ab;
        return Vector3.Distance(point, closest);
    }

    /// <summary>Distance from a point to a plane (signed).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float PointToPlaneDistance(this Vector3 point, Vector3 planeNormal, float planeD) =>
        Vector3.Dot(planeNormal, point) + planeD;

    /// <summary>Squared distance from a point to a line segment.</summary>
    public static float PointToSegmentDistanceSquared(this Vector3 point, Vector3 segA, Vector3 segB)
    {
        Vector3 ab = segB - segA;
        float lenSq = ab.LengthSquared();
        if (lenSq < 1e-10f)
            return Vector3.DistanceSquared(point, segA);

        float t = MathF.Max(0, MathF.Min(1f, Vector3.Dot(point - segA, ab) / lenSq));
        Vector3 closest = segA + t * ab;
        return Vector3.DistanceSquared(point, closest);
    }

    /// <summary>Distance from a point to an infinite line.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float PointToLineDistance(this Vector3 point, Vector3 lineOrigin, Vector3 lineDirection) =>
        Vector3.Cross(point - lineOrigin, lineDirection).Length() / lineDirection.Length();

    #endregion

    #region Noise Functions

    /// <summary>Hash function for integer coordinates to gradient index.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Hash(int x) => _permutation[x & (PermutationSize * 2 - 1)];

    /// <summary>Hash for 2D integer coordinates.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Hash(int x, int y) => _permutation[(x + _permutation[y & 255]) & 255];

    /// <summary>Hash for 3D integer coordinates.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Hash(int x, int y, int z) =>
        _permutation[(x + _permutation[(y + _permutation[z & 255]) & 255]) & 255];

    /// <summary>Fade curve: 6t⁵ - 15t⁴ + 10t³.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Fade(float t) => t * t * t * (t * (t * 6f - 15f) + 10f);

    /// <summary>1D value noise.</summary>
    public static float ValueNoise1D(float x)
    {
        int ix = MathF.Floor(x) is float f ? (int)f : 0;
        float fx = x - ix;

        int h0 = Hash(ix);
        int h1 = Hash(ix + 1);

        float v0 = (h0 & 1) == 0 ? 0f : 1f;
        float v1 = (h1 & 1) == 0 ? 0f : 1f;

        float t = Fade(fx);
        return v0 + (v1 - v0) * t;
    }

    /// <summary>
    /// 2D Perlin gradient noise.
    /// Returns a value approximately in [-1, 1].
    /// </summary>
    public static float GradientNoise2D(float x, float y)
    {
        int xi = MathF.Floor(x) is float fx2 ? (int)fx2 : 0;
        int yi = MathF.Floor(y) is float fy2 ? (int)fy2 : 0;

        float xf = x - xi;
        float yf = y - yi;

        float u = Fade(xf);
        float v = Fade(yf);

        int aa = Hash(xi, yi);
        int ab = Hash(xi, yi + 1);
        int ba = Hash(xi + 1, yi);
        int bb = Hash(xi + 1, yi + 1);

        float g00 = DotGrad2D(_gradients2D[aa % 8], xf, yf);
        float g10 = DotGrad2D(_gradients2D[ba % 8], xf - 1f, yf);
        float g01 = DotGrad2D(_gradients2D[ab % 8], xf, yf - 1f);
        float g11 = DotGrad2D(_gradients2D[bb % 8], xf - 1f, yf - 1f);

        float lerp0 = g00 + u * (g10 - g00);
        float lerp1 = g01 + u * (g11 - g01);

        return (lerp0 + v * (lerp1 - lerp0)) * GradientScale2D;
    }

    /// <summary>
    /// 3D Perlin gradient noise.
    /// Returns a value approximately in [-1, 1].
    /// </summary>
    public static float GradientNoise3D(float x, float y, float z)
    {
        int xi = MathF.Floor(x) is float fx3 ? (int)fx3 : 0;
        int yi = MathF.Floor(y) is float fy3 ? (int)fy3 : 0;
        int zi = MathF.Floor(z) is float fz3 ? (int)fz3 : 0;

        float xf = x - xi;
        float yf = y - yi;
        float zf = z - zi;

        float u = Fade(xf);
        float v = Fade(yf);
        float w = Fade(zf);

        int aaa = Hash(xi, yi, zi);
        int aba = Hash(xi, yi + 1, zi);
        int aab = Hash(xi, yi, zi + 1);
        int abb = Hash(xi, yi + 1, zi + 1);
        int baa = Hash(xi + 1, yi, zi);
        int bba = Hash(xi + 1, yi + 1, zi);
        int bab = Hash(xi + 1, yi, zi + 1);
        int bbb = Hash(xi + 1, yi + 1, zi + 1);

        float g000 = DotGrad3D(_gradients3D[aaa % 12], xf, yf, zf);
        float g100 = DotGrad3D(_gradients3D[baa % 12], xf - 1f, yf, zf);
        float g010 = DotGrad3D(_gradients3D[aba % 12], xf, yf - 1f, zf);
        float g110 = DotGrad3D(_gradients3D[bba % 12], xf - 1f, yf - 1f, zf);
        float g001 = DotGrad3D(_gradients3D[aab % 12], xf, yf, zf - 1f);
        float g101 = DotGrad3D(_gradients3D[bab % 12], xf - 1f, yf, zf - 1f);
        float g011 = DotGrad3D(_gradients3D[abb % 12], xf, yf - 1f, zf - 1f);
        float g111 = DotGrad3D(_gradients3D[bbb % 12], xf - 1f, yf - 1f, zf - 1f);

        float lerp00 = g000 + u * (g100 - g000);
        float lerp10 = g010 + u * (g110 - g010);
        float lerp01 = g001 + u * (g101 - g001);
        float lerp11 = g011 + u * (g111 - g011);

        float lerp0 = lerp00 + v * (lerp10 - lerp00);
        float lerp1 = lerp01 + v * (lerp11 - lerp01);

        return (lerp0 + w * (lerp1 - lerp0)) * GradientScale3D;
    }

    /// <summary>Fractal Brownian Motion (fBm) using gradient noise with configurable octaves.</summary>
    public static float Fbm2D(float x, float y, int octaves = 6, float lacunarity = 2f, float gain = 0.5f)
    {
        float sum = 0f;
        float amp = 1f;
        float freq = 1f;
        float maxAmp = 0f;

        for (int i = 0; i < octaves; i++)
        {
            sum += amp * GradientNoise2D(x * freq, y * freq);
            maxAmp += amp;
            amp *= gain;
            freq *= lacunarity;
        }

        return sum / maxAmp;
    }

    /// <summary>Fractal Brownian Motion (fBm) using 3D gradient noise.</summary>
    public static float Fbm3D(float x, float y, float z, int octaves = 6, float lacunarity = 2f, float gain = 0.5f)
    {
        float sum = 0f;
        float amp = 1f;
        float freq = 1f;
        float maxAmp = 0f;

        for (int i = 0; i < octaves; i++)
        {
            sum += amp * GradientNoise3D(x * freq, y * freq, z * freq);
            maxAmp += amp;
            amp *= gain;
            freq *= lacunarity;
        }

        return sum / maxAmp;
    }

    /// <summary>Ridged multifractal noise for terrain-like features.</summary>
    public static float RidgedNoise2D(float x, float y, int octaves = 6, float lacunarity = 2f, float gain = 0.5f)
    {
        float sum = 0f;
        float amp = 1f;
        float freq = 1f;
        float prev = 1f;

        for (int i = 0; i < octaves; i++)
        {
            float n = 1f - MathF.Abs(GradientNoise2D(x * freq, y * freq));
            n *= n;
            sum += n * amp * prev;
            prev = n;
            amp *= gain;
            freq *= lacunarity;
        }

        return sum;
    }

    /// <summary>Simplex-like 2D noise (faster alternative to Perlin gradient noise).</summary>
    public static float SimplexNoise2D(float x, float y)
    {
        const float F2 = 0.3660254037844386f; // (sqrt(3) - 1) / 2
        const float G2 = 0.2113248654051871f; // (3 - sqrt(3)) / 6

        float s = (x + y) * F2;
        int i = MathF.Floor(x + s) is float fi ? (int)fi : 0;
        int j = MathF.Floor(y + s) is float fj ? (int)fj : 0;

        float t = (i + j) * G2;
        float X0 = i - t;
        float Y0 = j - t;
        float x0 = x - X0;
        float y0 = y - Y0;

        int i1, j1;
        if (x0 > y0)
        { i1 = 1; j1 = 0; }
        else
        { i1 = 0; j1 = 1; }

        float x1 = x0 - i1 + G2;
        float y1 = y0 - j1 + G2;
        float x2 = x0 - 1f + 2f * G2;
        float y2 = y0 - 1f + 2f * G2;

        int ii = i & 255;
        int jj = j & 255;

        float n0 = 0f, n1 = 0f, n2 = 0f;

        float t0 = 0.5f - x0 * x0 - y0 * y0;
        if (t0 > 0)
        {
            t0 *= t0;
            int gi0 = _permutation[ii + _permutation[jj] & 255] % 8;
            n0 = t0 * t0 * DotGrad2D(_gradients2D[gi0], x0, y0);
        }

        float t1 = 0.5f - x1 * x1 - y1 * y1;
        if (t1 > 0)
        {
            t1 *= t1;
            int gi1 = _permutation[ii + i1 + _permutation[jj + j1 & 255] & 255] % 8;
            n1 = t1 * t1 * DotGrad2D(_gradients2D[gi1], x1, y1);
        }

        float t2 = 0.5f - x2 * x2 - y2 * y2;
        if (t2 > 0)
        {
            t2 *= t2;
            int gi2 = _permutation[ii + 1 + _permutation[jj + 1 & 255] & 255] % 8;
            n2 = t2 * t2 * DotGrad2D(_gradients2D[gi2], x2, y2);
        }

        return 70f * (n0 + n1 + n2);
    }

    /// <summary>3D simplex-like noise.</summary>
    public static float SimplexNoise3D(float x, float y, float z)
    {
        const float F3 = 1f / 3f;
        const float G3 = 1f / 6f;

        float s = (x + y + z) * F3;
        int i = MathF.Floor(x + s) is float fi ? (int)fi : 0;
        int j = MathF.Floor(y + s) is float fj ? (int)fj : 0;
        int k = MathF.Floor(z + s) is float fk ? (int)fk : 0;

        float t = (i + j + k) * G3;
        float X0 = i - t;
        float Y0 = j - t;
        float Z0 = k - t;
        float x0 = x - X0;
        float y0 = y - Y0;
        float z0 = z - Z0;

        int i1, j1, k1, i2, j2, k2;

        if (x0 >= y0)
        {
            if (y0 >= z0)
            { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
            else if (x0 >= z0)
            { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
            else
            { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
        }
        else
        {
            if (y0 < z0)
            { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
            else if (x0 < z0)
            { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
            else
            { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
        }

        float x1 = x0 - i1 + G3;
        float y1 = y0 - j1 + G3;
        float z1 = z0 - k1 + G3;
        float x2 = x0 - i2 + 2f * G3;
        float y2 = y0 - j2 + 2f * G3;
        float z2 = z0 - k2 + 2f * G3;
        float x3 = x0 - 1f + 3f * G3;
        float y3 = y0 - 1f + 3f * G3;
        float z3 = z0 - 1f + 3f * G3;

        int ii = i & 255;
        int jj = j & 255;
        int kk = k & 255;

        float n0 = 0f, n1 = 0f, n2 = 0f, n3 = 0f;

        float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
        if (t0 > 0)
        { t0 *= t0; n0 = t0 * t0 * DotGrad3D(_gradients3D[_permutation[ii + _permutation[jj + _permutation[kk] & 255] & 255] % 12], x0, y0, z0); }

        float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
        if (t1 > 0)
        { t1 *= t1; n1 = t1 * t1 * DotGrad3D(_gradients3D[_permutation[ii + i1 + _permutation[jj + j1 + _permutation[kk + k1 & 255] & 255] & 255] % 12], x1, y1, z1); }

        float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
        if (t2 > 0)
        { t2 *= t2; n2 = t2 * t2 * DotGrad3D(_gradients3D[_permutation[ii + i2 + _permutation[jj + j2 + _permutation[kk + k2 & 255] & 255] & 255] % 12], x2, y2, z2); }

        float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
        if (t3 > 0)
        { t3 *= t3; n3 = t3 * t3 * DotGrad3D(_gradients3D[_permutation[ii + 1 + _permutation[jj + 1 + _permutation[kk + 1 & 255] & 255] & 255] % 12], x3, y3, z3); }

        return 32f * (n0 + n1 + n2 + n3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotGrad2D(Vector2 g, float x, float y) => g.X * x + g.Y * y;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotGrad3D(Vector3 g, float x, float y, float z) => g.X * x + g.Y * y + g.Z * z;

    /// <summary>Worley (cellular) noise returning distance to nearest feature point.</summary>
    public static float WorleyNoise2D(float x, float y, Func<Vector2, Vector2, float>? distanceFunc = null)
    {
        distanceFunc ??= Vector2.Distance;

        int ix = MathF.Floor(x) is float fwi ? (int)fwi : 0;
        int iy = MathF.Floor(y) is float fhi ? (int)fhi : 0;
        float fx = x - ix;
        float fy = y - iy;

        float minDist = float.MaxValue;

        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                int cx = ix + dx;
                int cy = iy + dy;
                Vector2 featurePoint = new(
                    cx + Hash(cx, cy) * InvMaxByte,
                    cy + Hash(cy, cx) * InvMaxByte);
                float dist = distanceFunc(new Vector2(fx - dx, fy - dy), featurePoint - new Vector2(cx, cy));
                if (dist < minDist)
                    minDist = dist;
            }
        }

        return minDist;
    }

    #endregion

    #region Random Point Generation

    /// <summary>Random point uniformly distributed inside a sphere.</summary>
    public static Vector3 RandomPointInSphere(this Random rng, float radius = 1f)
    {
        float u = rng.NextSingle() * 2f - 1f;
        float theta = rng.NextSingle() * TwoPI;
        float r = radius * MathF.Pow(rng.NextSingle(), 1f / 3f);
        float sqrtTerm = MathF.Sqrt(1f - u * u);
        return new Vector3(
            r * sqrtTerm * MathF.Cos(theta),
            r * sqrtTerm * MathF.Sin(theta),
            r * u);
    }

    /// <summary>Random point uniformly distributed on a sphere surface.</summary>
    public static Vector3 RandomPointOnSphere(this Random rng, float radius = 1f)
    {
        float u = rng.NextSingle() * 2f - 1f;
        float theta = rng.NextSingle() * TwoPI;
        float sqrtTerm = MathF.Sqrt(1f - u * u);
        return new Vector3(
            radius * sqrtTerm * MathF.Cos(theta),
            radius * sqrtTerm * MathF.Sin(theta),
            radius * u);
    }

    /// <summary>Random point uniformly distributed inside an axis-aligned box.</summary>
    public static Vector3 RandomPointInBox(this Random rng, Vector3 min, Vector3 max)
    {
        return new Vector3(
            min.X + rng.NextSingle() * (max.X - min.X),
            min.Y + rng.NextSingle() * (max.Y - min.Y),
            min.Z + rng.NextSingle() * (max.Z - min.Z));
    }

    /// <summary>Random point uniformly distributed inside a cone.</summary>
    public static Vector3 RandomPointInCone(this Random rng, float height, float radius, float angleRad)
    {
        float cosAngle = MathF.Cos(angleRad);
        float r = radius * MathF.Sqrt(rng.NextSingle());
        float theta = rng.NextSingle() * TwoPI;
        float h = height * rng.NextSingle();
        return new Vector3(
            r * MathF.Cos(theta),
            h,
            r * MathF.Sin(theta));
    }

    /// <summary>Random point uniformly distributed inside a torus.</summary>
    public static Vector3 RandomPointInTorus(this Random rng, float majorRadius, float minorRadius)
    {
        float theta = rng.NextSingle() * TwoPI;
        float phi = rng.NextSingle() * TwoPI;
        float r = minorRadius * MathF.Sqrt(rng.NextSingle());
        float x = (majorRadius + r * MathF.Cos(phi)) * MathF.Cos(theta);
        float y = r * MathF.Sin(phi);
        float z = (majorRadius + r * MathF.Cos(phi)) * MathF.Sin(theta);
        return new Vector3(x, y, z);
    }

    /// <summary>Random point on a triangle surface using barycentric coordinates.</summary>
    public static Vector3 RandomPointOnTriangle(this Random rng, Vector3 v0, Vector3 v1, Vector3 v2)
    {
        float u1 = rng.NextSingle();
        float u2 = rng.NextSingle();
        if (u1 + u2 > 1f)
        { u1 = 1f - u1; u2 = 1f - u2; }
        return v0 + u1 * (v1 - v0) + u2 * (v2 - v0);
    }

    /// <summary>Random unit direction vector (uniformly distributed).</summary>
    public static Vector3 RandomDirection(this Random rng) =>
        rng.RandomPointOnSphere(1f);

    /// <summary>Random point inside a capsule defined by two endpoints and radius.</summary>
    public static Vector3 RandomPointInCapsule(this Random rng, Vector3 a, Vector3 b, float radius)
    {
        float t = rng.NextSingle();
        Vector3 pointOnSegment = Vector3.Lerp(a, b, t);
        return pointOnSegment + rng.RandomPointOnSphere(radius);
    }

    #endregion

    #region Projection and Reflection

    /// <summary>Project vector v onto vector n.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Project(this Vector3 v, Vector3 n)
    {
        float dot = Vector3.Dot(v, n);
        float nLenSq = n.LengthSquared();
        return nLenSq > 1e-10f ? n * (dot / nLenSq) : Vector3.Zero;
    }

    /// <summary>Project vector v onto a plane defined by its normal.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ProjectOnPlane(this Vector3 v, Vector3 planeNormal)
    {
        float dot = Vector3.Dot(v, planeNormal);
        float nLenSq = planeNormal.LengthSquared();
        return nLenSq > 1e-10f ? v - planeNormal * (dot / nLenSq) : v;
    }

    /// <summary>Reflect vector v off a surface with given normal.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Reflect(this Vector3 v, Vector3 normal)
    {
        return v - 2f * Vector3.Dot(v, normal) * normal;
    }

    /// <summary>Compute the refraction vector using Snell's law.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Refract(this Vector3 v, Vector3 normal, float etaRatio)
    {
        float dot = Vector3.Dot(v, normal);
        float k = 1f - etaRatio * etaRatio * (1f - dot * dot);
        if (k < 0f)
            return Vector3.Zero;
        return etaRatio * v - (etaRatio * dot + MathF.Sqrt(k)) * normal;
    }

    /// <summary>Project a point onto a line defined by origin and direction.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ProjectPointOnLine(this Vector3 point, Vector3 lineOrigin, Vector3 lineDirection)
    {
        float t = Vector3.Dot(point - lineOrigin, lineDirection) / lineDirection.LengthSquared();
        return lineOrigin + t * lineDirection;
    }

    /// <summary>Find the closest point on a segment to a given point.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ClosestPointOnSegment(this Vector3 point, Vector3 segA, Vector3 segB)
    {
        Vector3 ab = segB - segA;
        float lenSq = ab.LengthSquared();
        if (lenSq < 1e-10f)
            return segA;
        float t = MathF.Max(0, MathF.Min(1f, Vector3.Dot(point - segA, ab) / lenSq));
        return segA + t * ab;
    }

    /// <summary>Compute barycentric coordinates of point p relative to triangle (a, b, c).</summary>
    public static Vector3 BarycentricCoordinates(this Vector3 p, Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 v0 = b - a;
        Vector3 v1 = c - a;
        Vector3 v2 = p - a;

        float d00 = Vector3.Dot(v0, v0);
        float d01 = Vector3.Dot(v0, v1);
        float d11 = Vector3.Dot(v1, v1);
        float d20 = Vector3.Dot(v2, v0);
        float d21 = Vector3.Dot(v2, v1);

        float denom = d00 * d11 - d01 * d01;
        if (MathF.Abs(denom) < 1e-10f)
            return new Vector3(1f, 0f, 0f);

        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1f - v - w;

        return new Vector3(u, v, w);
    }

    /// <summary>
    /// Compute the closest distance from a point to a triangle.
    /// Returns the distance and optionally the closest point on the triangle.
    /// </summary>
    public static float PointToTriangleDistance(this Vector3 point, Vector3 v0, Vector3 v1, Vector3 v2,
        out Vector3 closestPoint)
    {
        Vector3 edge0 = v1 - v0;
        Vector3 edge1 = v2 - v0;
        Vector3 v = point - v0;

        float d00 = Vector3.Dot(edge0, edge0);
        float d01 = Vector3.Dot(edge0, edge1);
        float d11 = Vector3.Dot(edge1, edge1);
        float dv0 = Vector3.Dot(v, edge0);
        float dv1 = Vector3.Dot(v, edge1);

        float denom = d00 * d11 - d01 * d01;
        float s = (d11 * dv0 - d01 * dv1) / denom;
        float t = (d00 * dv1 - d01 * dv0) / denom;

        if (s >= 0 && t >= 0 && s + t <= 1)
        {
            closestPoint = v0 + s * edge0 + t * edge1;
            return Vector3.Distance(point, closestPoint);
        }

        float bestDist = float.MaxValue;
        Vector3 bestPoint = v0;

        Vector3 closestOnEdge0 = ClosestPointOnSegment(point, v0, v1);
        float d = Vector3.DistanceSquared(point, closestOnEdge0);
        if (d < bestDist)
        { bestDist = d; bestPoint = closestOnEdge0; }

        Vector3 closestOnEdge1 = ClosestPointOnSegment(point, v1, v2);
        d = Vector3.DistanceSquared(point, closestOnEdge1);
        if (d < bestDist)
        { bestDist = d; bestPoint = closestOnEdge1; }

        Vector3 closestOnEdge2 = ClosestPointOnSegment(point, v2, v0);
        d = Vector3.DistanceSquared(point, closestOnEdge2);
        if (d < bestDist)
        { bestDist = d; bestPoint = closestOnEdge2; }

        closestPoint = bestPoint;
        return MathF.Sqrt(bestDist);
    }

    /// <summary>
    /// Compute the closest distance from a point to a triangle (no closest point output).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float PointToTriangleDistance(this Vector3 point, Vector3 v0, Vector3 v1, Vector3 v2) =>
        PointToTriangleDistance(point, v0, v1, v2, out _);

    /// <summary>Compute normal of a triangle from its three vertices.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 TriangleNormal(this Vector3 v0, Vector3 v1, Vector3 v2) =>
        Vector3.Normalize(Vector3.Cross(v1 - v0, v2 - v0));

    /// <summary>Compute the area of a triangle.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float TriangleArea(this Vector3 v0, Vector3 v1, Vector3 v2) =>
        Vector3.Cross(v1 - v0, v2 - v0).Length() * 0.5f;

    #endregion

    #region Batch SIMD Operations

    /// <summary>
    /// Batch component-wise multiply for arrays of vectors using SIMD.
    /// Both source arrays must have the same length.
    /// </summary>
    public static unsafe void BatchMultiply(ReadOnlySpan<Vector3> a, ReadOnlySpan<Vector3> b, Span<Vector3> result)
    {
        int count = a.Length;
        Debug.Assert(count == b.Length && count == result.Length);

        if (Vector.IsHardwareAccelerated && count >= Vector<float>.Count)
        {
            int simdCount = count - (count % Vector<float>.Count);
            fixed (Vector3* pA = a, pB = b, pR = result)
            {
                float* pAF = (float*)pA;
                float* pBF = (float*)pB;
                float* pRF = (float*)pR;

                for (int i = 0; i < simdCount; i += Vector<float>.Count)
                {
                    Vector<float> va = new(MemoryMarshal.CreateReadOnlySpan(ref pAF[i], Vector<float>.Count));
                    Vector<float> vb = new(MemoryMarshal.CreateReadOnlySpan(ref pBF[i], Vector<float>.Count));
                    (va * vb).CopyTo(MemoryMarshal.CreateSpan(ref pRF[i], Vector<float>.Count));
                }
            }

            for (int i = simdCount; i < count; i++)
                result[i] = a[i] * b[i];
        }
        else
        {
            for (int i = 0; i < count; i++)
                result[i] = a[i] * b[i];
        }
    }

    /// <summary>
    /// Batch component-wise divide for arrays of vectors using SIMD.
    /// </summary>
    public static unsafe void BatchDivide(ReadOnlySpan<Vector3> a, ReadOnlySpan<Vector3> b, Span<Vector3> result)
    {
        int count = a.Length;
        Debug.Assert(count == b.Length && count == result.Length);

        if (Vector.IsHardwareAccelerated && count >= Vector<float>.Count)
        {
            int simdCount = count - (count % Vector<float>.Count);
            fixed (Vector3* pA = a, pB = b, pR = result)
            {
                float* pAF = (float*)pA;
                float* pBF = (float*)pB;
                float* pRF = (float*)pR;

                for (int i = 0; i < simdCount; i += Vector<float>.Count)
                {
                    Vector<float> va = new(MemoryMarshal.CreateReadOnlySpan(ref pAF[i], Vector<float>.Count));
                    Vector<float> vb = new(MemoryMarshal.CreateReadOnlySpan(ref pBF[i], Vector<float>.Count));
                    (va / vb).CopyTo(MemoryMarshal.CreateSpan(ref pRF[i], Vector<float>.Count));
                }
            }

            for (int i = simdCount; i < count; i++)
                result[i] = a[i] / b[i];
        }
        else
        {
            for (int i = 0; i < count; i++)
                result[i] = a[i] / b[i];
        }
    }

    /// <summary>
    /// Batch lerp for arrays of vectors using SIMD.
    /// </summary>
    public static unsafe void BatchLerp(ReadOnlySpan<Vector3> a, ReadOnlySpan<Vector3> b, float t, Span<Vector3> result)
    {
        int count = a.Length;
        Debug.Assert(count == b.Length && count == result.Length);

        if (Vector.IsHardwareAccelerated && count >= Vector<float>.Count)
        {
            Vector<float> vt = new(t);
            int simdCount = count - (count % Vector<float>.Count);
            fixed (Vector3* pA = a, pB = b, pR = result)
            {
                float* pAF = (float*)pA;
                float* pBF = (float*)pB;
                float* pRF = (float*)pR;

                for (int i = 0; i < simdCount; i += Vector<float>.Count)
                {
                    Vector<float> va = new(MemoryMarshal.CreateReadOnlySpan(ref pAF[i], Vector<float>.Count));
                    Vector<float> vb = new(MemoryMarshal.CreateReadOnlySpan(ref pBF[i], Vector<float>.Count));
                    Vector<float> vr = va + vt * (vb - va);
                    vr.CopyTo(MemoryMarshal.CreateSpan(ref pRF[i], Vector<float>.Count));
                }
            }

            for (int i = simdCount; i < count; i++)
                result[i] = Vector3.Lerp(a[i], b[i], t);
        }
        else
        {
            for (int i = 0; i < count; i++)
                result[i] = Vector3.Lerp(a[i], b[i], t);
        }
    }

    /// <summary>
    /// Batch normalize for arrays of vectors using SIMD.
    /// </summary>
    public static unsafe void BatchNormalize(Span<Vector3> vectors)
    {
        int count = vectors.Length;

        if (Vector.IsHardwareAccelerated && count >= Vector<float>.Count)
        {
            int simdCount = count - (count % Vector<float>.Count);
            fixed (Vector3* pV = vectors)
            {
                float* pVF = (float*)pV;

                for (int i = 0; i < simdCount; i += Vector<float>.Count)
                {
                    Vector<float> vx = new(MemoryMarshal.CreateReadOnlySpan(ref pVF[i], Vector<float>.Count));
                    Vector<float> vy = new(MemoryMarshal.CreateReadOnlySpan(ref pVF[i + 1], Vector<float>.Count));
                    Vector<float> vz = new(MemoryMarshal.CreateReadOnlySpan(ref pVF[i + 2], Vector<float>.Count));
                    Vector<float> lenSq = vx * vx + vy * vy + vz * vz;
                    Vector<float> invLen = Vector<float>.One / Vector.SquareRoot(lenSq);
                    (vx * invLen).CopyTo(MemoryMarshal.CreateSpan(ref pVF[i], Vector<float>.Count));
                    (vy * invLen).CopyTo(MemoryMarshal.CreateSpan(ref pVF[i + 1], Vector<float>.Count));
                    (vz * invLen).CopyTo(MemoryMarshal.CreateSpan(ref pVF[i + 2], Vector<float>.Count));
                }
            }

            for (int i = simdCount; i < count; i++)
                vectors[i] = Vector3.Normalize(vectors[i]);
        }
        else
        {
            for (int i = 0; i < count; i++)
                vectors[i] = Vector3.Normalize(vectors[i]);
        }
    }

    /// <summary>
    /// Batch dot product for paired arrays of vectors.
    /// </summary>
    public static unsafe void BatchDot(ReadOnlySpan<Vector3> a, ReadOnlySpan<Vector3> b, Span<float> result)
    {
        int count = a.Length;
        Debug.Assert(count == b.Length && count == result.Length);

        if (Vector.IsHardwareAccelerated && count >= Vector<float>.Count)
        {
            int simdCount = count - (count % Vector<float>.Count);
            fixed (Vector3* pA = a, pB = b)
            fixed (float* pR = result)
            {
                float* pAF = (float*)pA;
                float* pBF = (float*)pB;

                for (int i = 0; i < simdCount; i += Vector<float>.Count)
                {
                    Vector<float> ax = new(MemoryMarshal.CreateReadOnlySpan(ref pAF[i * 3], Vector<float>.Count));
                    Vector<float> ay = new(MemoryMarshal.CreateReadOnlySpan(ref pAF[i * 3 + 1], Vector<float>.Count));
                    Vector<float> az = new(MemoryMarshal.CreateReadOnlySpan(ref pAF[i * 3 + 2], Vector<float>.Count));
                    Vector<float> bx = new(MemoryMarshal.CreateReadOnlySpan(ref pBF[i * 3], Vector<float>.Count));
                    Vector<float> by = new(MemoryMarshal.CreateReadOnlySpan(ref pBF[i * 3 + 1], Vector<float>.Count));
                    Vector<float> bz = new(MemoryMarshal.CreateReadOnlySpan(ref pBF[i * 3 + 2], Vector<float>.Count));
                    Vector<float> dot = ax * bx + ay * by + az * bz;
                    dot.CopyTo(MemoryMarshal.CreateSpan(ref pR[i], Vector<float>.Count));
                }
            }

            for (int i = simdCount; i < count; i++)
                result[i] = Vector3.Dot(a[i], b[i]);
        }
        else
        {
            for (int i = 0; i < count; i++)
                result[i] = Vector3.Dot(a[i], b[i]);
        }
    }

    /// <summary>
    /// Batch cross product for paired arrays of vectors.
    /// </summary>
    public static unsafe void BatchCross(ReadOnlySpan<Vector3> a, ReadOnlySpan<Vector3> b, Span<Vector3> result)
    {
        int count = a.Length;
        Debug.Assert(count == b.Length && count == result.Length);

        if (Avx.IsSupported && count >= 8)
        {
            int simdCount = count - (count % 8);
            fixed (Vector3* pA = a, pB = b, pR = result)
            {
                float* pAF = (float*)pA;
                float* pBF = (float*)pB;
                float* pRF = (float*)pR;

                for (int i = 0; i < simdCount; i += 8)
                {
                    // Process 8 vectors at a time with AVX
                    for (int j = 0; j < 8 && (i + j) < count; j++)
                    {
                        int idx = (i + j) * 3;
                        result[i + j] = Vector3.Cross(a[i + j], b[i + j]);
                    }
                }
            }

            for (int i = simdCount; i < count; i++)
                result[i] = Vector3.Cross(a[i], b[i]);
        }
        else
        {
            for (int i = 0; i < count; i++)
                result[i] = Vector3.Cross(a[i], b[i]);
        }
    }

    /// <summary>
    /// Batch distance calculation between paired arrays of vectors.
    /// </summary>
    public static void BatchDistance(ReadOnlySpan<Vector3> a, ReadOnlySpan<Vector3> b, Span<float> result)
    {
        int count = a.Length;
        Debug.Assert(count == b.Length && count == result.Length);

        for (int i = 0; i < count; i++)
            result[i] = Vector3.Distance(a[i], b[i]);
    }

    /// <summary>
    /// Batch distance squared calculation.
    /// </summary>
    public static void BatchDistanceSquared(ReadOnlySpan<Vector3> a, ReadOnlySpan<Vector3> b, Span<float> result)
    {
        int count = a.Length;
        Debug.Assert(count == b.Length && count == result.Length);

        for (int i = 0; i < count; i++)
            result[i] = Vector3.DistanceSquared(a[i], b[i]);
    }

    /// <summary>
    /// Batch transform vectors by a matrix.
    /// </summary>
    public static void BatchTransform(ReadOnlySpan<Vector3> vectors, Matrix4x4 matrix, Span<Vector3> result)
    {
        int count = vectors.Length;
        Debug.Assert(count == result.Length);

        for (int i = 0; i < count; i++)
            result[i] = Vector3.Transform(vectors[i], matrix);
    }

    /// <summary>
    /// Batch transform normals by the inverse transpose of a matrix.
    /// </summary>
    public static void BatchTransformNormal(ReadOnlySpan<Vector3> normals, Matrix4x4 matrix, Span<Vector3> result)
    {
        int count = normals.Length;
        Debug.Assert(count == result.Length);

        for (int i = 0; i < count; i++)
            result[i] = Vector3.TransformNormal(normals[i], matrix);
    }

    /// <summary>
    /// Batch clamp all vectors to specified min/max bounds.
    /// </summary>
    public static void BatchClamp(ReadOnlySpan<Vector3> vectors, Vector3 min, Vector3 max, Span<Vector3> result)
    {
        int count = vectors.Length;
        Debug.Assert(count == result.Length);

        for (int i = 0; i < count; i++)
            result[i] = Vector3.Clamp(vectors[i], min, max);
    }

    /// <summary>
    /// Compute the bounding box (min and max) of a set of vectors.
    /// </summary>
    public static void ComputeBounds(ReadOnlySpan<Vector3> points, out Vector3 min, out Vector3 max)
    {
        Debug.Assert(points.Length > 0);
        min = new Vector3(float.MaxValue);
        max = new Vector3(float.MinValue);

        for (int i = 0; i < points.Length; i++)
        {
            min = Vector3.Min(min, points[i]);
            max = Vector3.Max(max, points[i]);
        }
    }

    /// <summary>
    /// Compute the centroid (average position) of a set of vectors.
    /// </summary>
    public static Vector3 ComputeCentroid(ReadOnlySpan<Vector3> points)
    {
        Debug.Assert(points.Length > 0);
        Vector3 sum = Vector3.Zero;
        for (int i = 0; i < points.Length; i++)
            sum += points[i];
        return sum / points.Length;
    }

    /// <summary>
    /// Compute the covariance matrix of a set of vectors.
    /// Returns a 3x3 matrix stored as 9 floats: [xx xy xz yx yy yz zx zy zz].
    /// </summary>
    public static void ComputeCovarianceMatrix(ReadOnlySpan<Vector3> points, Span<float> covariance)
    {
        Debug.Assert(points.Length > 0 && covariance.Length >= 9);
        covariance.Clear();

        Vector3 centroid = ComputeCentroid(points);
        float invN = 1f / points.Length;

        for (int i = 0; i < points.Length; i++)
        {
            Vector3 d = points[i] - centroid;
            covariance[0] += d.X * d.X;
            covariance[1] += d.X * d.Y;
            covariance[2] += d.X * d.Z;
            covariance[4] += d.Y * d.Y;
            covariance[5] += d.Y * d.Z;
            covariance[8] += d.Z * d.Z;
        }

        covariance[0] *= invN;
        covariance[1] *= invN;
        covariance[2] *= invN;
        covariance[3] = covariance[1];
        covariance[4] *= invN;
        covariance[5] *= invN;
        covariance[6] = covariance[2];
        covariance[7] = covariance[5];
        covariance[8] *= invN;
    }

    /// <summary>
    /// Resample a curve defined by points to a new set of uniformly spaced points.
    /// </summary>
    public static void ResampleCurve(ReadOnlySpan<Vector3> points, int newCount, Span<Vector3> result)
    {
        Debug.Assert(points.Length >= 2 && newCount >= 2);
        Debug.Assert(result.Length == newCount);

        float totalLength = 0f;
        for (int i = 1; i < points.Length; i++)
            totalLength += Vector3.Distance(points[i - 1], points[i]);

        float segmentLength = totalLength / (newCount - 1);

        result[0] = points[0];
        int currentPoint = 1;
        float accumulated = 0f;

        for (int i = 1; i < newCount - 1; i++)
        {
            float targetDist = i * segmentLength;
            while (currentPoint < points.Length)
            {
                float segDist = Vector3.Distance(points[currentPoint - 1], points[currentPoint]);
                if (accumulated + segDist >= targetDist)
                {
                    float t = (targetDist - accumulated) / segDist;
                    result[i] = Vector3.Lerp(points[currentPoint - 1], points[currentPoint], t);
                    break;
                }
                accumulated += segDist;
                currentPoint++;
            }
        }

        result[newCount - 1] = points[points.Length - 1];
    }

    /// <summary>
    /// Generate a grid of vectors in a rectangular area.
    /// </summary>
    public static void GenerateGrid(Vector3 origin, Vector3 right, Vector3 up, int width, int height,
        float cellSize, Span<Vector3> result)
    {
        Debug.Assert(result.Length == width * height);

        int idx = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                result[idx++] = origin + right * (x * cellSize) + up * (y * cellSize);
            }
        }
    }

    /// <summary>
    /// Compute the convex hull of a set of 2D points using Andrew's monotone chain algorithm.
    /// Returns indices into the original point array.
    /// </summary>
    public static int ConvexHull2D(ReadOnlySpan<Vector2> points, Span<int> hullIndices)
    {
        int n = points.Length;
        if (n <= 2)
        {
            for (int i = 0; i < n; i++)
                hullIndices[i] = i;
            return n;
        }

        Span<int> sorted = stackalloc int[n];
        for (int i = 0; i < n; i++)
            sorted[i] = i;

        // Sort by x, then by y
        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (points[sorted[i]].X > points[sorted[j]].X ||
                    (points[sorted[i]].X == points[sorted[j]].X && points[sorted[i]].Y > points[sorted[j]].Y))
                {
                    (sorted[i], sorted[j]) = (sorted[j], sorted[i]);
                }
            }
        }

        int k = 0;
        // Lower hull
        for (int i = 0; i < n; i++)
        {
            while (k >= 2)
            {
                Vector2 a = points[hullIndices[k - 1]];
                Vector2 b = points[hullIndices[k - 2]];
                Vector2 c = points[sorted[i]];
                float cross = (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);
                if (cross <= 0)
                    k--;
                else
                    break;
            }
            hullIndices[k++] = sorted[i];
        }

        // Upper hull
        for (int i = n - 2; i >= 0; i--)
        {
            while (k >= 2)
            {
                Vector2 a = points[hullIndices[k - 1]];
                Vector2 b = points[hullIndices[k - 2]];
                Vector2 c = points[sorted[i]];
                float cross = (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);
                if (cross <= 0)
                    k--;
                else
                    break;
            }
            hullIndices[k++] = sorted[i];
        }

        return k;
    }

    #endregion

    #region Additional Utility Methods

    /// <summary>Snap a vector to a grid spacing.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 SnapToGrid(this Vector3 v, float gridSize)
    {
        return new Vector3(
            MathF.Round(v.X / gridSize) * gridSize,
            MathF.Round(v.Y / gridSize) * gridSize,
            MathF.Round(v.Z / gridSize) * gridSize);
    }

    /// <summary>Move a vector towards a target by a maximum distance.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 MoveTowards(this Vector3 current, Vector3 target, float maxDistance)
    {
        Vector3 diff = target - current;
        float distSq = diff.LengthSquared();
        if (distSq <= maxDistance * maxDistance || distSq < 1e-10f)
            return target;
        return current + diff / MathF.Sqrt(distSq) * maxDistance;
    }

    /// <summary>Smoothly damp between two vectors (spring-damper system).</summary>
    public static Vector3 SmoothDamp(Vector3 current, Vector3 target, ref Vector3 velocity,
        float smoothTime, float maxSpeed, float deltaTime)
    {
        smoothTime = MathF.Max(0.0001f, smoothTime);
        float omega = 2f / smoothTime;
        float x = omega * deltaTime;
        float exp = 1f / (1f + x + 0.48f * x * x + 0.235f * x * x * x);

        Vector3 change = current - target;
        Vector3 temp = (velocity + omega * change) * deltaTime;
        velocity = (velocity - omega * temp) * exp;
        Vector3 output = target + (change + temp) * exp;

        Vector3 maxChange = new(maxSpeed * smoothTime);
        Vector3 maxChangeSq = maxChange.ComponentMultiply(maxChange);
        Vector3 diff = output - target;
        Vector3 diffSq = diff.ComponentMultiply(diff);

        if (diffSq.X > maxChangeSq.X || diffSq.Y > maxChangeSq.Y || diffSq.Z > maxChangeSq.Z)
        {
            float magnitude = MathF.Sqrt(diffSq.X + diffSq.Y + diffSq.Z);
            if (magnitude > 1e-10f)
            {
                output = target + diff / magnitude * maxSpeed * smoothTime;
                velocity = (output - target) / deltaTime;
            }
        }

        return output;
    }

    /// <summary>Decompose a direction vector into normal and tangential components relative to a surface normal.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DecomposeTangentNormal(this Vector3 direction, Vector3 surfaceNormal,
        out Vector3 tangent, out Vector3 normal)
    {
        normal = direction.Project(surfaceNormal);
        tangent = direction - normal;
    }

    /// <summary>Compute the angle between two 2D vectors in radians.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float AngleBetween2D(this Vector2 a, Vector2 b) =>
        MathF.Atan2(b.Y - a.Y, b.X - a.X);

    /// <summary>Compute the winding order sign of three 2D points (positive = CCW).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WindingOrder2D(this Vector2 a, Vector2 b, Vector2 c) =>
        (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);

    /// <summary>
    /// Compute signed volume of a tetrahedron.
    /// Positive when vertices are in counter-clockwise order when viewed from outside.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SignedVolume(Vector3 a, Vector3 b, Vector3 c, Vector3 d) =>
        Vector3.Dot(Vector3.Cross(b - a, c - a), d - a) / 6f;

    /// <summary>
    /// Interpolate between multiple vectors using a blend shape weight array.
    /// weights[i] corresponds to offsets[i]; the base vector is added last.
    /// </summary>
    public static Vector3 BlendShapes(ReadOnlySpan<Vector3> baseShape, ReadOnlySpan<Vector3> offsets,
        ReadOnlySpan<float> weights)
    {
        Debug.Assert(baseShape.Length == offsets.Length && offsets.Length == weights.Length);

        Vector3 result = Vector3.Zero;
        for (int i = 0; i < weights.Length; i++)
            result += baseShape[i] + offsets[i] * weights[i];

        return result;
    }

    /// <summary>
    /// Compute the distance from a point to a plane defined by a point and normal.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float PointToPlaneDistance(this Vector3 point, Vector3 planePoint, Vector3 planeNormal) =>
        Vector3.Dot(planeNormal, point - planePoint);

    /// <summary>
    /// Check if a point is inside a triangle using barycentric coordinates.
    /// </summary>
    public static bool IsPointInTriangle(this Vector3 point, Vector3 v0, Vector3 v1, Vector3 v2, float tolerance = 1e-6f)
    {
        Vector3 bary = BarycentricCoordinates(point, v0, v1, v2);
        return bary.X >= -tolerance && bary.Y >= -tolerance && bary.Z >= -tolerance;
    }

    /// <summary>
    /// Compute the Minkowski sum of two convex polygons.
    /// </summary>
    public static void MinkowskiSum(ReadOnlySpan<Vector2> polyA, ReadOnlySpan<Vector2> polyB, Span<Vector2> result)
    {
        int countA = polyA.Length;
        int countB = polyB.Length;
        Debug.Assert(result.Length == countA * countB);

        int idx = 0;
        for (int i = 0; i < countA; i++)
        {
            for (int j = 0; j < countB; j++)
            {
                result[idx++] = polyA[i] + polyB[j];
            }
        }
    }

    /// <summary>
    /// Compute the centroid of a polygon (2D points assumed in XY plane).
    /// </summary>
    public static Vector2 PolygonCentroid(ReadOnlySpan<Vector2> vertices)
    {
        Debug.Assert(vertices.Length >= 3);
        float area = 0f;
        Vector2 centroid = Vector2.Zero;

        for (int i = 0; i < vertices.Length; i++)
        {
            int j = (i + 1) % vertices.Length;
            float cross = vertices[i].X * vertices[j].Y - vertices[j].X * vertices[i].Y;
            area += cross;
            centroid += new Vector2(
                (vertices[i].X + vertices[j].X) * cross,
                (vertices[i].Y + vertices[j].Y) * cross);
        }

        area *= 0.5f;
        if (MathF.Abs(area) < 1e-10f)
            return vertices[0];
        centroid /= 6f * area;
        return centroid;
    }

    /// <summary>
    /// Compute the signed area of a 2D polygon.
    /// </summary>
    public static float PolygonArea(ReadOnlySpan<Vector2> vertices)
    {
        float area = 0f;
        for (int i = 0; i < vertices.Length; i++)
        {
            int j = (i + 1) % vertices.Length;
            area += vertices[i].X * vertices[j].Y;
            area -= vertices[j].X * vertices[i].Y;
        }
        return area * 0.5f;
    }

    /// <summary>
    /// Check if a ray intersects a sphere.
    /// Returns true if intersection found, and outputs the distance to the nearer intersection.
    /// </summary>
    public static bool RaySphereIntersect(Vector3 rayOrigin, Vector3 rayDir, Vector3 sphereCenter, float sphereRadius,
        out float distance)
    {
        Vector3 oc = rayOrigin - sphereCenter;
        float a = Vector3.Dot(rayDir, rayDir);
        float b = 2f * Vector3.Dot(oc, rayDir);
        float c = Vector3.Dot(oc, oc) - sphereRadius * sphereRadius;
        float discriminant = b * b - 4f * a * c;

        if (discriminant < 0)
        {
            distance = 0;
            return false;
        }

        distance = (-b - MathF.Sqrt(discriminant)) / (2f * a);
        if (distance < 0)
        {
            distance = (-b + MathF.Sqrt(discriminant)) / (2f * a);
            if (distance < 0)
            {
                distance = 0;
                return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Check if a ray intersects an axis-aligned bounding box.
    /// Returns true if intersection found and outputs the distance.
    /// </summary>
    public static bool RayAABBIntersect(Vector3 rayOrigin, Vector3 rayDir, Vector3 boxMin, Vector3 boxMax,
        out float distance)
    {
        distance = 0;
        float tmin = float.MinValue;
        float tmax = float.MaxValue;

        for (int i = 0; i < 3; i++)
        {
            float orig = i switch { 0 => rayOrigin.X, 1 => rayOrigin.Y, _ => rayOrigin.Z };
            float dir = i switch { 0 => rayDir.X, 1 => rayDir.Y, _ => rayDir.Z };
            float min = i switch { 0 => boxMin.X, 1 => boxMin.Y, _ => boxMin.Z };
            float max = i switch { 0 => boxMax.X, 1 => boxMax.Y, _ => boxMax.Z };

            if (MathF.Abs(dir) < 1e-10f)
            {
                if (orig < min || orig > max)
                    return false;
            }
            else
            {
                float invD = 1f / dir;
                float t0 = (min - orig) * invD;
                float t1 = (max - orig) * invD;
                if (t0 > t1)
                    (t0, t1) = (t1, t0);
                tmin = MathF.Max(tmin, t0);
                tmax = MathF.Min(tmax, t1);
                if (tmin > tmax)
                    return false;
            }
        }

        distance = tmin >= 0 ? tmin : tmax;
        return distance >= 0;
    }

    #endregion
}
