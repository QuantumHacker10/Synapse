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

public static class NumericalDifferentiation
{
    /// <summary>
    /// 2nd-order central first derivative: df/dx ≈ (f[i+1] − f[i−1]) / (2h)
    /// </summary>
    public static void FirstDerivative2nd(ReadOnlySpan<double> f, Span<double> dfdx, double h, int n)
    {
        dfdx[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * h);
        for (int i = 1; i < n - 1; i++)
            dfdx[i] = (f[i + 1] - f[i - 1]) / (2.0 * h);
        dfdx[n - 1] = (3.0 * f[n - 1] - 4.0 * f[n - 2] + f[n - 3]) / (2.0 * h);
    }

    /// <summary>
    /// 2nd-order central second derivative: d²f/dx² ≈ (f[i+1] − 2f[i] + f[i−1]) / h²
    /// </summary>
    public static void SecondDerivative2nd(ReadOnlySpan<double> f, Span<double> d2fdx2, double h, int n)
    {
        d2fdx2[0] = (2.0 * f[0] - 5.0 * f[1] + 4.0 * f[2] - f[3]) / (h * h);
        for (int i = 1; i < n - 1; i++)
            d2fdx2[i] = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / (h * h);
        d2fdx2[n - 1] = (2.0 * f[n - 1] - 5.0 * f[n - 2] + 4.0 * f[n - 3] - f[n - 4]) / (h * h);
    }

    /// <summary>
    /// 4th-order central first derivative.
    /// </summary>
    public static void FirstDerivative4th(ReadOnlySpan<double> f, Span<double> dfdx, double h, int n)
    {
        double invH = 1.0 / (12.0 * h);
        dfdx[0] = (-25.0 * f[0] + 48.0 * f[1] - 36.0 * f[2] + 16.0 * f[3] - 3.0 * f[4]) * invH;
        dfdx[1] = (-25.0 * f[1] + 48.0 * f[2] - 36.0 * f[3] + 16.0 * f[4] - 3.0 * f[5]) * invH;
        for (int i = 2; i < n - 2; i++)
            dfdx[i] = (f[i - 2] - 8.0 * f[i - 1] + 8.0 * f[i + 1] - f[i + 2]) * invH;
        dfdx[n - 2] = (25.0 * f[n - 2] - 48.0 * f[n - 3] + 36.0 * f[n - 4] - 16.0 * f[n - 5] + 3.0 * f[n - 6]) * (-invH);
        dfdx[n - 1] = (25.0 * f[n - 1] - 48.0 * f[n - 2] + 36.0 * f[n - 3] - 16.0 * f[n - 4] + 3.0 * f[n - 5]) * (-invH);
    }

    /// <summary>
    /// 4th-order central second derivative.
    /// </summary>
    public static void SecondDerivative4th(ReadOnlySpan<double> f, Span<double> d2fdx2, double h, int n)
    {
        double invH2 = 1.0 / (12.0 * h * h);
        d2fdx2[0] = (35.0 * f[0] - 104.0 * f[1] + 114.0 * f[2] - 56.0 * f[3] + 11.0 * f[4]) * invH2;
        d2fdx2[1] = (35.0 * f[1] - 104.0 * f[2] + 114.0 * f[3] - 56.0 * f[4] + 11.0 * f[5]) * invH2;
        for (int i = 2; i < n - 2; i++)
            d2fdx2[i] = (-f[i - 2] + 16.0 * f[i - 1] - 30.0 * f[i] + 16.0 * f[i + 1] - f[i + 2]) * invH2;
        d2fdx2[n - 2] = (35.0 * f[n - 2] - 104.0 * f[n - 3] + 114.0 * f[n - 4] - 56.0 * f[n - 5] + 11.0 * f[n - 6]) * invH2;
        d2fdx2[n - 1] = (35.0 * f[n - 1] - 104.0 * f[n - 2] + 114.0 * f[n - 3] - 56.0 * f[n - 4] + 11.0 * f[n - 5]) * invH2;
    }

    /// <summary>
    /// Gradient of a 3D scalar field. Returns (dF/dx, dF/dy, dF/dz).
    /// </summary>
    public static void Gradient3D(
        ReadOnlySpan<double> f,
        Span<double> dfdx, Span<double> dfdy, Span<double> dfdz,
        int nx, int ny, int nz, double dx)
    {
        double invDx2 = 0.5 / dx;
        for (int z = 1; z < nz - 1; z++)
            for (int y = 1; y < ny - 1; y++)
                for (int x = 1; x < nx - 1; x++)
                {
                    int idx = z * ny * nx + y * nx + x;
                    dfdx[idx] = (f[idx + 1] - f[idx - 1]) * invDx2;
                    dfdy[idx] = (f[idx + nx] - f[idx - nx]) * invDx2;
                    dfdz[idx] = (f[idx + ny * nx] - f[idx - ny * nx]) * invDx2;
                }
    }

    /// <summary>
    /// Divergence of a 3D vector field. Returns ∇·F = dFx/dx + dFy/dy + dFz/dz.
    /// </summary>
    public static double Divergence3D(
        ReadOnlySpan<double> fx, ReadOnlySpan<double> fy, ReadOnlySpan<double> fz,
        int nx, int ny, int nz, double dx)
    {
        double sum = 0;
        double invDx2 = 0.5 / dx;
        for (int z = 1; z < nz - 1; z++)
            for (int y = 1; y < ny - 1; y++)
                for (int x = 1; x < nx - 1; x++)
                {
                    int idx = z * ny * nx + y * nx + x;
                    double div = (fx[idx + 1] - fx[idx - 1]) * invDx2 +
                                 (fy[idx + nx] - fy[idx - nx]) * invDx2 +
                                 (fz[idx + ny * nx] - fz[idx - ny * nx]) * invDx2;
                    sum += div * div;
                }
        return Math.Sqrt(sum);
    }

    /// <summary>
    /// Curl of a 3D vector field. Returns (∇×F)x, (∇×F)y, (∇×F)z.
    /// </summary>
    public static void Curl3D(
        ReadOnlySpan<double> fx, ReadOnlySpan<double> fy, ReadOnlySpan<double> fz,
        Span<double> curlX, Span<double> curlY, Span<double> curlZ,
        int nx, int ny, int nz, double dx)
    {
        double invDx2 = 0.5 / dx;
        for (int z = 1; z < nz - 1; z++)
            for (int y = 1; y < ny - 1; y++)
                for (int x = 1; x < nx - 1; x++)
                {
                    int idx = z * ny * nx + y * nx + x;
                    int xp = idx + 1, xm = idx - 1;
                    int yp = idx + nx, ym = idx - nx;
                    int zp = idx + ny * nx, zm = idx - ny * nx;

                    curlX[idx] = (fz[yp] - fz[ym]) * invDx2 - (fy[zp] - fy[zm]) * invDx2;
                    curlY[idx] = (fx[zp] - fx[zm]) * invDx2 - (fz[xp] - fz[xm]) * invDx2;
                    curlZ[idx] = (fy[xp] - fy[xm]) * invDx2 - (fx[yp] - fx[ym]) * invDx2;
                }
    }
}

// ============================================================================
//  Additional Utility: Matrix Operations
// ============================================================================

/// <summary>
/// Lightweight matrix operations for physics solver internals.
/// Works with flattened row-major arrays.
/// </summary>
