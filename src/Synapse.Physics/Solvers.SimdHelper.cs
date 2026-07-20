// ============================================================================
// Synapse Omnia — Physics Solvers
// Complete implementations of electromagnetic, acoustic, thermodynamic,
// chemical, gravitational, lattice-Boltzmann, quantum, elastic, turbulent,
// and multiphysics solvers.
//
// C# 14 · unsafe · NativeAOT compatible
// ============================================================================

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Synapse.Physics;
internal static class SimdHelper
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ScaleAdd(ReadOnlySpan<double> src, double alpha, Span<double> dst)
    {
        int len = src.Length;
        int i = 0;
        for (; i + 3 < len; i += 4)
        {
            dst[i] += src[i] * alpha;
            dst[i + 1] += src[i + 1] * alpha;
            dst[i + 2] += src[i + 2] * alpha;
            dst[i + 3] += src[i + 3] * alpha;
        }
        for (; i < len; i++)
            dst[i] += src[i] * alpha;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double sum = 0.0;
        int len = Math.Min(a.Length, b.Length);
        int i = 0;
        for (; i + 3 < len; i += 4)
        {
            sum += a[i] * b[i] + a[i + 1] * b[i + 1] +
                   a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        }
        for (; i < len; i++)
            sum += a[i] * b[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Axpy(ReadOnlySpan<double> x, double a, ReadOnlySpan<double> y, Span<double> result)
    {
        int len = Math.Min(Math.Min(x.Length, y.Length), result.Length);
        for (int i = 0; i < len; i++)
            result[i] = x[i] * a + y[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Norm2(ReadOnlySpan<double> v)
    {
        double sum = 0.0;
        for (int i = 0; i < v.Length; i++)
            sum += v[i] * v[i];
        return Math.Sqrt(sum);
    }
}

// ============================================================================
//  1. MaxwellSolver — 3-D FDTD with Yee grid
// ============================================================================

/// <summary>Polarization type for dispersive material models.</summary>
