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
// FILE: MathHelpers.cs
// PATH: Utilities/MathHelpers.cs
// ============================================================

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace GDNN.Utilities;

/// <summary>
/// Common math constants, angle conversions, interpolation utilities,
/// numerical methods, random number generation, and statistical functions.
/// </summary>
public static class MathHelpers
{
    /// <summary>Pi (3.14159...).</summary>
    public const float Pi = MathF.PI;

    /// <summary>Two times Pi (6.28318...).</summary>
    public const float TwoPi = MathF.Tau;

    /// <summary>Pi divided by 2 (1.57079...).</summary>
    public const float PiOver2 = MathF.PI * 0.5f;

    /// <summary>Pi divided by 4 (0.78539...).</summary>
    public const float PiOver4 = MathF.PI * 0.25f;

    /// <summary>Pi divided by 180 (conversion factor to radians).</summary>
    public const float Deg2Rad = MathF.PI / 180f;

    /// <summary>180 divided by Pi (conversion factor to degrees).</summary>
    public const float Rad2Deg = 180f / MathF.PI;

    /// <summary>Square root of 2.</summary>
    public const float Sqrt2 = 1.41421356f;

    /// <summary>Reciprocal of square root of 2.</summary>
    public const float InvSqrt2 = 0.70710678f;

    /// <summary>Euler's number (2.71828...).</summary>
    public const float E = MathF.E;

    /// <summary>Smallest positive float such that 1.0 + Epsilon != 1.0.</summary>
    public const float Epsilon = 1.19209290e-7f;

    /// <summary>Converts degrees to radians.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DegreesToRadians(float degrees) => degrees * Deg2Rad;

    /// <summary>Converts radians to degrees.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float RadiansToDegrees(float radians) => radians * Rad2Deg;

    /// <summary>Converts a Vector3 of angles (degrees) to radians.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 DegreesToRadians(Vector3 degrees) => degrees * Deg2Rad;

    /// <summary>Converts a Vector3 of angles (radians) to degrees.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 RadiansToDegrees(Vector3 radians) => radians * Rad2Deg;

    /// <summary>Clamps a value between min and max.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Clamp(float value, float min, float max) =>
        value < min ? min : value > max ? max : value;

    /// <summary>Clamps a value between 0 and 1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Clamp01(float value) =>
        value < 0f ? 0f : value > 1f ? 1f : value;

    /// <summary>Clamps an integer between min and max.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Clamp(int value, int min, int max) =>
        value < min ? min : value > max ? max : value;

    /// <summary>Clamps a value between min and max.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Clamp(double value, double min, double max) =>
        value < min ? min : value > max ? max : value;

    /// <summary>Linearly interpolates between a and b by t.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Lerp(float a, float b, float t) => a + (b - a) * Clamp01(t);

    /// <summary>Linearly interpolates between a and b by t (unclamped).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float LerpUnclamped(float a, float b, float t) => a + (b - a) * t;

    /// <summary>Linearly interpolates a Vector3.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Lerp(Vector3 a, Vector3 b, float t) =>
        Vector3.Lerp(a, b, Clamp01(t));

    /// <summary>Linearly interpolates a Quaternion using SLERP.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Quaternion Slerp(Quaternion a, Quaternion b, float t) =>
        Quaternion.Slerp(a, b, Clamp01(t));

    /// <summary>Smoothly interpolates between 0 and 1 using hermite interpolation (3t^2 - 2t^3).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SmoothStep(float edge0, float edge1, float x)
    {
        float t = Clamp01((x - edge0) / (edge1 - edge0));
        return t * t * (3f - 2f * t);
    }

    /// <summary>Smootherstep using Perlin's improved formula (6t^5 - 15t^4 + 10t^3).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SmootherStep(float edge0, float edge1, float x)
    {
        float t = Clamp01((x - edge0) / (edge1 - edge0));
        return t * t * t * (t * (t * 6f - 15f) + 10f);
    }

    /// <summary>Remaps a value from one range to another.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Remap(float value, float fromMin, float fromMax, float toMin, float toMax) =>
        toMin + (value - fromMin) * (toMax - toMin) / (fromMax - fromMin);

    /// <summary>Remaps a value from [0,1] to another range.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Remap01(float value, float toMin, float toMax) =>
        toMin + value * (toMax - toMin);

    /// <summary>Performs inverse linear interpolation: given a value between a and b, returns t.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float InverseLerp(float a, float b, float value)
    {
        if (MathF.Abs(b - a) < Epsilon) return 0;
        return Clamp01((value - a) / (b - a));
    }

    /// <summary>Returns the sign of a float (-1, 0, or 1).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sign(float value) => value > 0 ? 1 : value < 0 ? -1 : 0;

    /// <summary>Returns the next power of two greater than or equal to value.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int NextPowerOfTwo(int value)
    {
        value--;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        return value + 1;
    }

    /// <summary>Returns true if value is a power of two.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsPowerOfTwo(int value) => value > 0 && (value & (value - 1)) == 0;

    /// <summary>Returns the floor log base 2 of an integer.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Log2Floor(int value)
    {
        int result = 0;
        while ((value >>= 1) != 0) result++;
        return result;
    }

    /// <summary>Returns the ceiling log base 2 of an integer.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Log2Ceiling(int value) =>
        value <= 1 ? 0 : Log2Floor(value - 1) + 1;

    /// <summary>Wraps a value in the range [0, length).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Wrap(float value, float length)
    {
        value %= length;
        if (value < 0) value += length;
        return value;
    }

    /// <summary>Wraps an integer in the range [0, length).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Wrap(int value, int length)
    {
        value %= length;
        if (value < 0) value += length;
        return value;
    }

    /// <summary>Gets a perpendicular vector to the given vector.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 GetPerpendicular(Vector3 v)
    {
        Vector3 abs = Vector3.Abs(v);
        if (abs.X <= abs.Y && abs.X <= abs.Z)
            return Vector3.Cross(v, Vector3.UnitX);
        else if (abs.Y <= abs.Z)
            return Vector3.Cross(v, Vector3.UnitY);
        else
            return Vector3.Cross(v, Vector3.UnitZ);
    }

    /// <summary>Evaluates a polynomial using Horner's method.</summary>
    /// <param name="coefficients">Polynomial coefficients from lowest to highest degree.</param>
    /// <param name="x">The value to evaluate at.</param>
    /// <returns>The polynomial value at x.</returns>
    public static float EvaluatePolynomial(ReadOnlySpan<float> coefficients, float x)
    {
        if (coefficients.Length == 0) return 0;
        float result = coefficients[^1];
        for (int i = coefficients.Length - 2; i >= 0; i--)
        {
            result = result * x + coefficients[i];
        }
        return result;
    }

    /// <summary>Evaluates a polynomial using Horner's method (double precision).</summary>
    public static double EvaluatePolynomial(ReadOnlySpan<double> coefficients, double x)
    {
        if (coefficients.Length == 0) return 0;
        double result = coefficients[^1];
        for (int i = coefficients.Length - 2; i >= 0; i--)
        {
            result = result * x + coefficients[i];
        }
        return result;
    }

    /// <summary>
    /// Numerical integration using the trapezoidal rule.
    /// </summary>
    /// <param name="func">The function to integrate.</param>
    /// <param name="a">Lower bound.</param>
    /// <param name="b">Upper bound.</param>
    /// <param name="intervals">Number of intervals.</param>
    /// <returns>Approximate integral value.</returns>
    public static double IntegrateTrapezoidal(Func<double, double> func, double a, double b, int intervals = 1000)
    {
        if (intervals <= 0) throw new ArgumentException("Intervals must be positive.", nameof(intervals));
        if (MathF.Abs((float)(b - a)) < Epsilon) return 0;

        double h = (b - a) / intervals;
        double sum = 0.5 * (func(a) + func(b));

        for (int i = 1; i < intervals; i++)
        {
            sum += func(a + i * h);
        }

        return sum * h;
    }

    /// <summary>
    /// Numerical integration using Simpson's 1/3 rule.
    /// </summary>
    /// <param name="func">The function to integrate.</param>
    /// <param name="a">Lower bound.</param>
    /// <param name="b">Upper bound.</param>
    /// <param name="intervals">Number of intervals (must be even).</param>
    /// <returns>Approximate integral value.</returns>
    public static double IntegrateSimpson(Func<double, double> func, double a, double b, int intervals = 1000)
    {
        if (intervals <= 0) throw new ArgumentException("Intervals must be positive.", nameof(intervals));
        if (intervals % 2 != 0) intervals++;
        if (MathF.Abs((float)(b - a)) < Epsilon) return 0;

        double h = (b - a) / intervals;
        double sum = func(a) + func(b);

        for (int i = 1; i < intervals; i++)
        {
            double x = a + i * h;
            sum += (i % 2 == 0 ? 2 : 4) * func(x);
        }

        return sum * h / 3.0;
    }

    /// <summary>
    /// Numerical integration using adaptive Simpson's method.
    /// </summary>
    /// <param name="func">The function to integrate.</param>
    /// <param name="a">Lower bound.</param>
    /// <param name="b">Upper bound.</param>
    /// <param name="tolerance">Error tolerance.</param>
    /// <param name="maxDepth">Maximum recursion depth.</param>
    /// <returns>Approximate integral value.</returns>
    public static double IntegrateAdaptiveSimpson(Func<double, double> func, double a, double b,
        double tolerance = 1e-10, int maxDepth = 20)
    {
        double whole = SimpsonRule(func, a, b);
        return AdaptiveSimpsonRecursive(func, a, b, whole, tolerance, maxDepth, 0);
    }

    private static double SimpsonRule(Func<double, double> func, double a, double b)
    {
        double c = (a + b) * 0.5;
        double h = b - a;
        return h * (func(a) + 4 * func(c) + func(b)) / 6.0;
    }

    private static double AdaptiveSimpsonRecursive(Func<double, double> func, double a, double b,
        double whole, double tolerance, int maxDepth, int depth)
    {
        double c = (a + b) * 0.5;
        double left = SimpsonRule(func, a, c);
        double right = SimpsonRule(func, c, b);
        double combined = left + right;

        if (depth >= maxDepth || MathF.Abs((float)(combined - whole)) <= 15.0f * (float)tolerance)
        {
            return combined + (combined - whole) / 15.0;
        }

        return AdaptiveSimpsonRecursive(func, a, c, left, tolerance * 0.5, maxDepth, depth + 1) +
               AdaptiveSimpsonRecursive(func, c, b, right, tolerance * 0.5, maxDepth, depth + 1);
    }

    /// <summary>
    /// Finds a root using the bisection method.
    /// </summary>
    /// <param name="func">The function whose root to find.</param>
    /// <param name="a">Lower bound.</param>
    /// <param name="b">Upper bound.</param>
    /// <param name="tolerance">Convergence tolerance.</param>
    /// <param name="maxIterations">Maximum iterations.</param>
    /// <returns>Approximate root.</returns>
    public static double FindRootBisection(Func<double, double> func, double a, double b,
        double tolerance = 1e-10, int maxIterations = 1000)
    {
        double fa = func(a), fb = func(b);
        if (fa * fb > 0)
            throw new ArgumentException("Function must have different signs at a and b.");

        for (int i = 0; i < maxIterations; i++)
        {
            double c = (a + b) * 0.5;
            double fc = func(c);

            if (MathF.Abs((float)fc) < (float)tolerance || (b - a) * 0.5 < tolerance)
                return c;

            if (fa * fc < 0)
            {
                b = c;
                fb = fc;
            }
            else
            {
                a = c;
                fa = fc;
            }
        }

        return (a + b) * 0.5;
    }

    /// <summary>
    /// Finds a root using the Newton-Raphson method.
    /// </summary>
    /// <param name="func">The function whose root to find.</param>
    /// <param name="dfunc">Derivative of the function.</param>
    /// <param name="x0">Initial guess.</param>
    /// <param name="tolerance">Convergence tolerance.</param>
    /// <param name="maxIterations">Maximum iterations.</param>
    /// <returns>Approximate root.</returns>
    public static double FindRootNewtonRaphson(Func<double, double> func, Func<double, double> dfunc,
        double x0, double tolerance = 1e-10, int maxIterations = 100)
    {
        double x = x0;

        for (int i = 0; i < maxIterations; i++)
        {
            double fx = func(x);
            if (MathF.Abs((float)fx) < (float)tolerance)
                return x;

            double dfx = dfunc(x);
            if (MathF.Abs((float)dfx) < (float)Epsilon)
                throw new InvalidOperationException("Derivative is near zero; Newton-Raphson may not converge.");

            x = x - fx / dfx;
        }

        return x;
    }

    /// <summary>
    /// Finds a root using the secant method.
    /// </summary>
    public static double FindRootSecant(Func<double, double> func, double x0, double x1,
        double tolerance = 1e-10, int maxIterations = 100)
    {
        double f0 = func(x0), f1 = func(x1);

        for (int i = 0; i < maxIterations; i++)
        {
            if (MathF.Abs((float)f1) < (float)tolerance)
                return x1;

            double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
            x0 = x1; f0 = f1;
            x1 = x2; f1 = func(x1);
        }

        return x1;
    }

    /// <summary>
    /// Finds a minimum using golden section search.
    /// </summary>
    /// <param name="func">The function to minimize.</param>
    /// <param name="a">Lower bound.</param>
    /// <param name="b">Upper bound.</param>
    /// <param name="tolerance">Convergence tolerance.</param>
    /// <returns>Approximate minimum location.</returns>
    public static double FindMinimumGoldenSection(Func<double, double> func, double a, double b,
        double tolerance = 1e-10)
    {
        const double invPhi = 0.6180339887498948;
        double c = b - invPhi * (b - a);
        double d = a + invPhi * (b - a);
        double fc = func(c), fd = func(d);

        while (b - a > tolerance)
        {
            if (fc < fd)
            {
                b = d; d = c; fd = fc;
                c = b - invPhi * (b - a);
                fc = func(c);
            }
            else
            {
                a = c; c = d; fc = fd;
                d = a + invPhi * (b - a);
                fd = func(d);
            }
        }

        return (a + b) * 0.5;
    }

    /// <summary>
    /// Computes the mean of a span of values.
    /// </summary>
    public static double Mean(ReadOnlySpan<double> values)
    {
        if (values.Length == 0) return 0;
        double sum = 0;
        foreach (double v in values) sum += v;
        return sum / values.Length;
    }

    /// <summary>
    /// Computes the mean of a span of float values.
    /// </summary>
    public static float Mean(ReadOnlySpan<float> values)
    {
        if (values.Length == 0) return 0;
        double sum = 0;
        foreach (float v in values) sum += v;
        return (float)(sum / values.Length);
    }

    /// <summary>
    /// Computes the variance of a span of values.
    /// </summary>
    public static double Variance(ReadOnlySpan<double> values, bool populationVariance = true)
    {
        if (values.Length == 0) return 0;
        double mean = Mean(values);
        double sumSq = 0;
        foreach (double v in values)
        {
            double diff = v - mean;
            sumSq += diff * diff;
        }
        return populationVariance ? sumSq / values.Length : sumSq / (values.Length - 1);
    }

    /// <summary>
    /// Computes the variance of a span of float values.
    /// </summary>
    public static float Variance(ReadOnlySpan<float> values, bool populationVariance = true)
    {
        if (values.Length == 0) return 0;
        float mean = Mean(values);
        double sumSq = 0;
        foreach (float v in values)
        {
            double diff = v - mean;
            sumSq += diff * diff;
        }
        return (float)(populationVariance ? sumSq / values.Length : sumSq / (values.Length - 1));
    }

    /// <summary>
    /// Computes the standard deviation of a span of values.
    /// </summary>
    public static double StandardDeviation(ReadOnlySpan<double> values, bool population = true) =>
        Math.Sqrt(Variance(values, population));

    /// <summary>
    /// Computes the standard deviation of a span of float values.
    /// </summary>
    public static float StandardDeviation(ReadOnlySpan<float> values, bool population = true) =>
        MathF.Sqrt(Variance(values, population));

    /// <summary>
    /// Computes a percentile value from a sorted span.
    /// </summary>
    /// <param name="sortedValues">Sorted values in ascending order.</param>
    /// <param name="percentile">Percentile to compute (0-100).</param>
    /// <returns>The percentile value.</returns>
    public static double Percentile(ReadOnlySpan<double> sortedValues, double percentile)
    {
        if (sortedValues.Length == 0) return 0;
        if (sortedValues.Length == 1) return sortedValues[0];

        double index = (percentile / 100.0) * (sortedValues.Length - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper) return sortedValues[lower];

        double fraction = index - lower;
        return sortedValues[lower] * (1 - fraction) + sortedValues[upper] * fraction;
    }

    /// <summary>
    /// Computes a percentile value from an unsorted span (creates a copy for sorting).
    /// </summary>
    public static double PercentileUnsorted(ReadOnlySpan<double> values, double percentile)
    {
        if (values.Length == 0) return 0;
        double[] sorted = values.ToArray();
        Array.Sort(sorted);
        return Percentile(sorted, percentile);
    }

    /// <summary>
    /// Computes the median of a span of values.
    /// </summary>
    public static double Median(ReadOnlySpan<double> values) => PercentileUnsorted(values, 50);

    /// <summary>
    /// Computes the median of a span of float values.
    /// </summary>
    public static float Median(ReadOnlySpan<float> values)
    {
        if (values.Length == 0) return 0;
        float[] sorted = values.ToArray();
        Array.Sort(sorted);
        int mid = sorted.Length / 2;
        return sorted.Length % 2 == 0
            ? (sorted[mid - 1] + sorted[mid]) * 0.5f
            : sorted[mid];
    }

    /// <summary>
    /// Computes the weighted average of values.
    /// </summary>
    public static double WeightedMean(ReadOnlySpan<double> values, ReadOnlySpan<double> weights)
    {
        if (values.Length != weights.Length)
            throw new ArgumentException("Values and weights must have the same length.");
        if (values.Length == 0) return 0;

        double sumWeight = 0, sumValue = 0;
        for (int i = 0; i < values.Length; i++)
        {
            sumValue += values[i] * weights[i];
            sumWeight += weights[i];
        }

        return sumWeight < Epsilon ? 0 : sumValue / sumWeight;
    }

    /// <summary>
    /// Computes the moving average of a series.
    /// </summary>
    /// <param name="values">Input series.</param>
    /// <param name="windowSize">Moving average window size.</param>
    /// <returns>Moving average series (shorter than input by windowSize-1).</returns>
    public static double[] MovingAverage(ReadOnlySpan<double> values, int windowSize)
    {
        if (windowSize <= 0) throw new ArgumentException("Window size must be positive.");
        if (values.Length < windowSize) return Array.Empty<double>();

        int resultLength = values.Length - windowSize + 1;
        double[] result = new double[resultLength];

        double sum = 0;
        for (int i = 0; i < windowSize; i++)
            sum += values[i];
        result[0] = sum / windowSize;

        for (int i = 1; i < resultLength; i++)
        {
            sum += values[i + windowSize - 1] - values[i - 1];
            result[i] = sum / windowSize;
        }

        return result;
    }

    /// <summary>
    /// Computes the exponential moving average.
    /// </summary>
    public static double ExponentialMovingAverage(ReadOnlySpan<double> values, double alpha)
    {
        if (values.Length == 0) return 0;
        double ema = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            ema = alpha * values[i] + (1 - alpha) * ema;
        }
        return ema;
    }

    /// <summary>
    /// Generates a cryptographically strong random float in [0, 1).
    /// </summary>
    public static float RandomFloat() => Random.Shared.NextSingle();

    /// <summary>
    /// Generates a random float in [min, max).
    /// </summary>
    public static float RandomRange(float min, float max) =>
        min + Random.Shared.NextSingle() * (max - min);

    /// <summary>
    /// Generates a random double in [0, 1).
    /// </summary>
    public static double RandomDouble() => Random.Shared.NextDouble();

    /// <summary>
    /// Generates a random Vector3 within a unit cube.
    /// </summary>
    public static Vector3 RandomVector3() =>
        new(RandomFloat() - 0.5f, RandomFloat() - 0.5f, RandomFloat() - 0.5f);

    /// <summary>
    /// Generates a random direction vector (uniform on unit sphere).
    /// </summary>
    public static Vector3 RandomUnitVector()
    {
        float theta = RandomFloat() * TwoPi;
        float phi = MathF.Acos(2f * RandomFloat() - 1f);
        float sinPhi = MathF.Sin(phi);
        return new Vector3(
            sinPhi * MathF.Cos(theta),
            sinPhi * MathF.Sin(theta),
            MathF.Cos(phi)
        );
    }

    /// <summary>
    /// Generates a random point inside a unit sphere (uniform distribution).
    /// </summary>
    public static Vector3 RandomPointInSphere()
    {
        float u = RandomFloat();
        float r = MathF.Pow(u, 1f / 3f);
        return RandomUnitVector() * r;
    }

    /// <summary>
    /// Generates a random point on a unit sphere surface (uniform distribution).
    /// </summary>
    public static Vector3 RandomPointOnSphere() => RandomUnitVector();

    /// <summary>
    /// Generates a random point inside a unit disk.
    /// </summary>
    public static Vector2 RandomPointInDisk()
    {
        float angle = RandomFloat() * TwoPi;
        float r = MathF.Sqrt(RandomFloat());
        return new Vector2(MathF.Cos(angle) * r, MathF.Sin(angle) * r);
    }

    /// <summary>
    /// Generates a random Quaternion rotation.
    /// </summary>
    public static Quaternion RandomQuaternion()
    {
        float u1 = RandomFloat(), u2 = RandomFloat(), u3 = RandomFloat();
        float sqrtU1 = MathF.Sqrt(1 - u1);
        float sqrtU1Inv = MathF.Sqrt(u1);
        float theta1 = u2 * TwoPi;
        float theta2 = u3 * TwoPi;
        return new Quaternion(
            sqrtU1 * MathF.Sin(theta1),
            sqrtU1 * MathF.Cos(theta1),
            sqrtU1Inv * MathF.Sin(theta2),
            sqrtU1Inv * MathF.Cos(theta2)
        );
    }

    /// <summary>
    /// Generates Gaussian-distributed random number using Box-Muller transform.
    /// </summary>
    /// <param name="mean">Mean of the distribution.</param>
    /// <param name="stdDev">Standard deviation.</param>
    public static double RandomGaussian(double mean = 0, double stdDev = 1)
    {
        double u1 = 1.0 - Random.Shared.NextDouble();
        double u2 = 1.0 - Random.Shared.NextDouble();
        double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * normal;
    }

    /// <summary>
    /// Generates a random integer in [min, max).
    /// </summary>
    public static int RandomInt(int min, int max) => Random.Shared.Next(min, max);

    /// <summary>
    /// Shuffles an array in place using Fisher-Yates algorithm.
    /// </summary>
    public static void Shuffle<T>(Span<T> values)
    {
        for (int i = values.Length - 1; i > 0; i--)
        {
            int j = Random.Shared.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    /// <summary>
    /// Computes the dot product of two Vector3 values.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(Vector3 a, Vector3 b) => Vector3.Dot(a, b);

    /// <summary>
    /// Projects vector a onto vector b.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Project(Vector3 a, Vector3 b)
    {
        float sqrLen = b.LengthSquared();
        if (sqrLen < Epsilon) return Vector3.Zero;
        return b * (Vector3.Dot(a, b) / sqrLen);
    }

    /// <summary>
    /// Reflects a vector off a surface with the given normal.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Reflect(Vector3 direction, Vector3 normal) =>
        direction - 2f * Vector3.Dot(direction, normal) * normal;

    /// <summary>
    /// Refracts a vector through a surface with the given normal and ratio of indices.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 Refract(Vector3 direction, Vector3 normal, float eta)
    {
        float dot = Vector3.Dot(normal, direction);
        float k = 1f - eta * eta * (1f - dot * dot);
        if (k < 0) return Vector3.Zero;
        return eta * direction - (eta * dot + MathF.Sqrt(k)) * normal;
    }

    /// <summary>
    /// Computes the angle in radians between two vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float AngleBetween(Vector3 a, Vector3 b)
    {
        float lenA = a.Length(), lenB = b.Length();
        if (lenA < Epsilon || lenB < Epsilon) return 0;
        float dot = Math.Clamp(Vector3.Dot(a, b) / (lenA * lenB), -1f, 1f);
        return MathF.Acos(dot);
    }

    /// <summary>
    /// Computes the signed angle between two vectors around an axis.
    /// </summary>
    public static float SignedAngleBetween(Vector3 from, Vector3 to, Vector3 axis)
    {
        float unsignedAngle = AngleBetween(from, to);
        float sign = MathF.Sign(Vector3.Dot(axis, Vector3.Cross(from, to)));
        return unsignedAngle * sign;
    }

    /// <summary>
    /// Returns the closest point on a line segment to a point.
    /// </summary>
    public static Vector3 ClosestPointOnSegment(Vector3 point, Vector3 segmentStart, Vector3 segmentEnd)
    {
        Vector3 ab = segmentEnd - segmentStart;
        float t = Vector3.Dot(point - segmentStart, ab) / ab.LengthSquared();
        return segmentStart + ab * Clamp01(t);
    }

    /// <summary>
    /// Returns the distance from a point to a line segment.
    /// </summary>
    public static float DistanceToSegment(Vector3 point, Vector3 segmentStart, Vector3 segmentEnd) =>
        Vector3.Distance(point, ClosestPointOnSegment(point, segmentStart, segmentEnd));

    /// <summary>
    /// Returns the closest point on a triangle to a point.
    /// </summary>
    public static Vector3 ClosestPointOnTriangle(Vector3 point, Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 ab = b - a, ac = c - a, ap = point - a;
        float d1 = Vector3.Dot(ab, ap), d2 = Vector3.Dot(ac, ap);
        if (d1 <= 0 && d2 <= 0) return a;

        Vector3 bp = point - b;
        float d3 = Vector3.Dot(ab, bp), d4 = Vector3.Dot(ac, bp);
        if (d3 >= 0 && d4 <= d3) return b;

        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0)
        {
            float v = d1 / (d1 - d3);
            return a + v * ab;
        }

        Vector3 cp = point - c;
        float d5 = Vector3.Dot(ab, cp), d6 = Vector3.Dot(ac, cp);
        if (d6 >= 0 && d5 <= d6) return c;

        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0)
        {
            float w = d2 / (d2 - d6);
            return a + w * ac;
        }

        float va = d3 * d6 - d5 * d4;
        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + w * (c - b);
        }

        float denom = 1.0f / (va + vb + vc);
        float v2 = vb * denom;
        float w2 = vc * denom;
        return a + ab * v2 + ac * w2;
    }

    /// <summary>
    /// Checks if a point is inside a triangle using barycentric coordinates.
    /// </summary>
    public static bool IsPointInTriangle(Vector3 point, Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 v0 = c - a, v1 = b - a, v2 = point - a;
        float dot00 = Vector3.Dot(v0, v0), dot01 = Vector3.Dot(v0, v1);
        float dot02 = Vector3.Dot(v0, v2), dot11 = Vector3.Dot(v1, v1);
        float dot12 = Vector3.Dot(v1, v2);

        float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

        return u >= 0 && v >= 0 && (u + v) <= 1;
    }

    /// <summary>
    /// Computes the barycentric coordinates of a point on a triangle.
    /// </summary>
    public static Vector3 BarycentricCoordinates(Vector3 point, Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 v0 = b - a, v1 = c - a, v2 = point - a;
        float d00 = Vector3.Dot(v0, v0), d01 = Vector3.Dot(v0, v1);
        float d11 = Vector3.Dot(v1, v1), d20 = Vector3.Dot(v2, v0);
        float d21 = Vector3.Dot(v2, v1);
        float denom = d00 * d11 - d01 * d01;
        if (MathF.Abs(denom) < Epsilon) return new Vector3(1, 0, 0);

        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;
        return new Vector3(u, v, w);
    }

    /// <summary>
    /// Computes the area of a triangle.
    /// </summary>
    public static float TriangleArea(Vector3 a, Vector3 b, Vector3 c) =>
        Vector3.Cross(b - a, c - a).Length() * 0.5f;

    /// <summary>
    /// Computes the normal of a triangle.
    /// </summary>
    public static Vector3 TriangleNormal(Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 normal = Vector3.Cross(b - a, c - a);
        return Vector3.Normalize(normal);
    }

    /// <summary>
    /// Computes the centroid of a triangle.
    /// </summary>
    public static Vector3 TriangleCentroid(Vector3 a, Vector3 b, Vector3 c) =>
        (a + b + c) / 3f;

    /// <summary>
    /// Smoothly dampens a value toward a target (spring-damper system).
    /// </summary>
    /// <param name="current">Current value.</param>
    /// <param name="target">Target value.</param>
    /// <param name="currentVelocity">Current velocity (modified in-place).</param>
    /// <param name="smoothTime">Approximate time to reach the target.</param>
    /// <param name="maxSpeed">Maximum speed.</param>
    /// <param name="deltaTime">Time step.</param>
    /// <returns>The new value.</returns>
    public static float SmoothDamp(float current, float target, ref float currentVelocity,
        float smoothTime, float maxSpeed, float deltaTime)
    {
        smoothTime = MathF.Max(0.0001f, smoothTime);
        float omega = 2f / smoothTime;
        float x = omega * deltaTime;
        float exp = 1f / (1f + x + 0.48f * x * x + 0.235f * x * x * x);
        float change = current - target;
        float temp = (currentVelocity + omega * change) * deltaTime;
        currentVelocity = (currentVelocity - omega * temp) * exp;
        float output = target + (change + temp) * exp;

        if (target - current > 0 == output > target)
        {
            output = target;
            currentVelocity = (output - target) / deltaTime;
        }

        return output;
    }

    /// <summary>
    /// Moves a Vector3 toward a target at a given maximum distance.
    /// </summary>
    public static Vector3 MoveTowards(Vector3 current, Vector3 target, float maxDistance)
    {
        Vector3 diff = target - current;
        float dist = diff.Length();
        if (dist <= maxDistance || dist < Epsilon) return target;
        return current + diff / dist * maxDistance;
    }

    /// <summary>
    /// Rotates a vector around an axis by a given angle.
    /// </summary>
    public static Vector3 RotateAround(Vector3 point, Vector3 axis, float angle)
    {
        Quaternion rotation = Quaternion.CreateFromAxisAngle(axis, angle);
        return Vector3.Transform(point, rotation);
    }

    /// <summary>
    /// Computes the shortest rotation from one direction to another.
    /// </summary>
    public static Quaternion FromToRotation(Vector3 from, Vector3 to)
    {
        float dot = Vector3.Dot(from, to);
        if (dot >= 1f) return Quaternion.Identity;
        if (dot <= -1f)
        {
            Vector3 ortho = GetPerpendicular(from);
            return Quaternion.CreateFromAxisAngle(ortho, Pi);
        }
        Vector3 axis = Vector3.Cross(from, to);
        float w = MathF.Sqrt(from.LengthSquared() * to.LengthSquared()) + dot;
        return Quaternion.Normalize(new Quaternion(axis.X, axis.Y, axis.Z, w));
    }

    /// <summary>
    /// Computes the look-at rotation from direction to target direction.
    /// </summary>
    public static Quaternion LookRotation(Vector3 forward, Vector3 up)
    {
        forward = Vector3.Normalize(forward);
        Vector3 right = Vector3.Normalize(Vector3.Cross(up, forward));
        Vector3 correctedUp = Vector3.Cross(forward, right);

        Matrix4x4 m = Matrix4x4.Identity;
        m.M11 = right.X; m.M12 = right.Y; m.M13 = right.Z;
        m.M21 = correctedUp.X; m.M22 = correctedUp.Y; m.M23 = correctedUp.Z;
        m.M31 = forward.X; m.M32 = forward.Y; m.M33 = forward.Z;

        return Quaternion.CreateFromRotationMatrix(m);
    }

    /// <summary>
    /// Decomposes a matrix into scale, rotation, and translation.
    /// </summary>
    public static void DecomposeMatrix(Matrix4x4 matrix, out Vector3 scale,
        out Quaternion rotation, out Vector3 translation)
    {
        Matrix4x4.Decompose(matrix, out scale, out rotation, out translation);
    }

    /// <summary>
    /// Computes the frustum planes from a view-projection matrix.
    /// </summary>
    /// <param name="viewProjection">Combined view-projection matrix.</param>
    /// <returns>6 planes (left, right, bottom, top, near, far) as (normal, distance) pairs.</returns>
    public static (Vector3 Normal, float D)[] ExtractFrustumPlanes(Matrix4x4 viewProjection)
    {
        var planes = new (Vector3, float)[6];

        planes[0] = NormalizePlane(new Vector3(
            viewProjection.M14 + viewProjection.M11,
            viewProjection.M24 + viewProjection.M21,
            viewProjection.M34 + viewProjection.M31),
            viewProjection.M44 + viewProjection.M41);

        planes[1] = NormalizePlane(new Vector3(
            viewProjection.M14 - viewProjection.M11,
            viewProjection.M24 - viewProjection.M21,
            viewProjection.M34 - viewProjection.M31),
            viewProjection.M44 - viewProjection.M41);

        planes[2] = NormalizePlane(new Vector3(
            viewProjection.M14 + viewProjection.M12,
            viewProjection.M24 + viewProjection.M22,
            viewProjection.M34 + viewProjection.M32),
            viewProjection.M44 + viewProjection.M42);

        planes[3] = NormalizePlane(new Vector3(
            viewProjection.M14 - viewProjection.M12,
            viewProjection.M24 - viewProjection.M22,
            viewProjection.M34 - viewProjection.M32),
            viewProjection.M44 - viewProjection.M42);

        planes[4] = NormalizePlane(new Vector3(
            viewProjection.M13,
            viewProjection.M23,
            viewProjection.M33),
            viewProjection.M43);

        planes[5] = NormalizePlane(new Vector3(
            viewProjection.M14 - viewProjection.M13,
            viewProjection.M24 - viewProjection.M23,
            viewProjection.M34 - viewProjection.M33),
            viewProjection.M44 - viewProjection.M43);

        return planes;
    }

    private static (Vector3 Normal, float D) NormalizePlane(Vector3 normal, float d)
    {
        float length = normal.Length();
        if (length < Epsilon) return (Vector3.Zero, 0);
        return (normal / length, d / length);
    }

    /// <summary>
    /// Tests if a point is inside a sphere.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool PointInSphere(Vector3 point, Vector3 center, float radius) =>
        Vector3.DistanceSquared(point, center) <= radius * radius;

    /// <summary>
    /// Tests if a point is inside an axis-aligned bounding box.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool PointInAabb(Vector3 point, Vector3 min, Vector3 max) =>
        point.X >= min.X && point.X <= max.X &&
        point.Y >= min.Y && point.Y <= max.Y &&
        point.Z >= min.Z && point.Z <= max.Z;

    /// <summary>
    /// Tests ray-sphere intersection.
    /// </summary>
    public static bool RaySphereIntersect(Vector3 rayOrigin, Vector3 rayDirection,
        Vector3 sphereCenter, float sphereRadius, out float t)
    {
        t = 0;
        Vector3 oc = rayOrigin - sphereCenter;
        float a = Vector3.Dot(rayDirection, rayDirection);
        float b = 2.0f * Vector3.Dot(oc, rayDirection);
        float c = Vector3.Dot(oc, oc) - sphereRadius * sphereRadius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) return false;

        t = (-b - MathF.Sqrt(discriminant)) / (2 * a);
        if (t < 0)
        {
            t = (-b + MathF.Sqrt(discriminant)) / (2 * a);
            return t >= 0;
        }
        return true;
    }

    /// <summary>
    /// Tests ray-AABB intersection.
    /// </summary>
    public static bool RayAabbIntersect(Vector3 rayOrigin, Vector3 rayDirection,
        Vector3 boxMin, Vector3 boxMax, out float tMin, out float tMax)
    {
        tMin = 0; tMax = float.MaxValue;

        for (int i = 0; i < 3; i++)
        {
            float origin = i == 0 ? rayOrigin.X : i == 1 ? rayOrigin.Y : rayOrigin.Z;
            float dir = i == 0 ? rayDirection.X : i == 1 ? rayDirection.Y : rayDirection.Z;
            float min = i == 0 ? boxMin.X : i == 1 ? boxMin.Y : boxMin.Z;
            float max = i == 0 ? boxMax.X : i == 1 ? boxMax.Y : boxMax.Z;

            if (MathF.Abs(dir) < Epsilon)
            {
                if (origin < min || origin > max) return false;
            }
            else
            {
                float invDir = 1f / dir;
                float t1 = (min - origin) * invDir;
                float t2 = (max - origin) * invDir;

                if (t1 > t2) (t1, t2) = (t2, t1);
                tMin = MathF.Max(tMin, t1);
                tMax = MathF.Min(tMax, t2);

                if (tMin > tMax) return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Tests ray-triangle intersection.
    /// </summary>
    public static bool RayTriangleIntersect(Vector3 rayOrigin, Vector3 rayDirection,
        Vector3 v0, Vector3 v1, Vector3 v2, out float t, out float u, out float v)
    {
        t = u = v = 0;
        Vector3 edge1 = v1 - v0;
        Vector3 edge2 = v2 - v0;
        Vector3 h = Vector3.Cross(rayDirection, edge2);
        float a = Vector3.Dot(edge1, h);

        if (a > -Epsilon && a < Epsilon) return false;

        float f = 1f / a;
        Vector3 s = rayOrigin - v0;
        u = f * Vector3.Dot(s, h);
        if (u < 0f || u > 1f) return false;

        Vector3 q = Vector3.Cross(s, edge1);
        v = f * Vector3.Dot(rayDirection, q);
        if (v < 0f || u + v > 1f) return false;

        t = f * Vector3.Dot(edge2, q);
        return t > Epsilon;
    }

    /// <summary>
    /// Tests sphere-sphere intersection.
    /// </summary>
    public static bool SphereSphereIntersect(Vector3 centerA, float radiusA,
        Vector3 centerB, float radiusB)
    {
        float distSq = Vector3.DistanceSquared(centerA, centerB);
        float radiusSum = radiusA + radiusB;
        return distSq <= radiusSum * radiusSum;
    }

    /// <summary>
    /// Computes AABB-AABB intersection.
    /// </summary>
    public static bool AabbIntersect(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB) =>
        minA.X <= maxB.X && maxA.X >= minB.X &&
        minA.Y <= maxB.Y && maxA.Y >= minB.Y &&
        minA.Z <= maxB.Z && maxA.Z >= minB.Z;

    /// <summary>
    /// Computes the merged AABB of two AABBs.
    /// </summary>
    public static void MergeAabb(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB,
        out Vector3 mergedMin, out Vector3 mergedMax)
    {
        mergedMin = Vector3.Min(minA, minB);
        mergedMax = Vector3.Max(maxA, maxB);
    }

    /// <summary>
    /// Computes the surface area of an AABB.
    /// </summary>
    public static float AabbSurfaceArea(Vector3 min, Vector3 max)
    {
        Vector3 size = max - min;
        return 2f * (size.X * size.Y + size.Y * size.Z + size.Z * size.X);
    }

    /// <summary>
    /// Computes the volume of an AABB.
    /// </summary>
    public static float AabbVolume(Vector3 min, Vector3 max)
    {
        Vector3 size = max - min;
        return size.X * size.Y * size.Z;
    }

    /// <summary>
    /// Linearly interpolates between two matrices component-wise.
    /// </summary>
    public static Matrix4x4 Lerp(Matrix4x4 a, Matrix4x4 b, float t)
    {
        t = Clamp01(t);
        return new Matrix4x4(
            a.M11 + (b.M11 - a.M11) * t, a.M12 + (b.M12 - a.M12) * t,
            a.M13 + (b.M13 - a.M13) * t, a.M14 + (b.M14 - a.M14) * t,
            a.M21 + (b.M21 - a.M21) * t, a.M22 + (b.M22 - a.M22) * t,
            a.M23 + (b.M23 - a.M23) * t, a.M24 + (b.M24 - a.M24) * t,
            a.M31 + (b.M31 - a.M31) * t, a.M32 + (b.M32 - a.M32) * t,
            a.M33 + (b.M33 - a.M33) * t, a.M34 + (b.M34 - a.M34) * t,
            a.M41 + (b.M41 - a.M41) * t, a.M42 + (b.M42 - a.M42) * t,
            a.M43 + (b.M43 - a.M43) * t, a.M44 + (b.M44 - a.M44) * t);
    }

    /// <summary>
    /// Creates a look-at view matrix.
    /// </summary>
    public static Matrix4x4 CreateLookAt(Vector3 eye, Vector3 target, Vector3 up) =>
        Matrix4x4.CreateLookAt(eye, target, up);

    /// <summary>
    /// Creates a perspective projection matrix.
    /// </summary>
    public static Matrix4x4 CreatePerspective(float fovY, float aspect, float nearPlane, float farPlane) =>
        Matrix4x4.CreatePerspectiveFieldOfView(fovY, aspect, nearPlane, farPlane);

    /// <summary>
    /// Creates an orthographic projection matrix.
    /// </summary>
    public static Matrix4x4 CreateOrthographic(float width, float height, float nearPlane, float farPlane) =>
        Matrix4x4.CreateOrthographic(width, height, nearPlane, farPlane);

    /// <summary>
    /// Creates a scale matrix.
    /// </summary>
    public static Matrix4x4 CreateScale(Vector3 scale) => Matrix4x4.CreateScale(scale);

    /// <summary>
    /// Creates a translation matrix.
    /// </summary>
    public static Matrix4x4 CreateTranslation(Vector3 translation) =>
        Matrix4x4.CreateTranslation(translation);

    /// <summary>
    /// Creates a rotation matrix from an axis and angle.
    /// </summary>
    public static Matrix4x4 CreateRotation(Vector3 axis, float angle) =>
        Matrix4x4.CreateFromAxisAngle(axis, angle);

    /// <summary>
    /// Creates a rotation matrix from a quaternion.
    /// </summary>
    public static Matrix4x4 CreateRotation(Quaternion rotation) =>
        Matrix4x4.CreateFromQuaternion(rotation);
}
