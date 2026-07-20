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
public static class SpatialInterpolation
{
    /// <summary>Trilinear interpolation at position (x, y, z) on a regular grid.</summary>
    public static double TrilinearInterpolate(
        ReadOnlySpan<double> field, int nx, int ny, int nz,
        double dx, double dy, double dz,
        double x, double y, double z)
    {
        double xi = x / dx, yi = y / dy, zi = z / dz;
        int x0 = Math.Clamp((int)Math.Floor(xi), 0, nx - 2);
        int y0 = Math.Clamp((int)Math.Floor(yi), 0, ny - 2);
        int z0 = Math.Clamp((int)Math.Floor(zi), 0, nz - 2);
        double xf = xi - x0, yf = yi - y0, zf = zi - z0;

        double c000 = field[z0 * ny * nx + y0 * nx + x0];
        double c100 = field[z0 * ny * nx + y0 * nx + x0 + 1];
        double c010 = field[z0 * ny * nx + (y0 + 1) * nx + x0];
        double c110 = field[z0 * ny * nx + (y0 + 1) * nx + x0 + 1];
        double c001 = field[(z0 + 1) * ny * nx + y0 * nx + x0];
        double c101 = field[(z0 + 1) * ny * nx + y0 * nx + x0 + 1];
        double c011 = field[(z0 + 1) * ny * nx + (y0 + 1) * nx + x0];
        double c111 = field[(z0 + 1) * ny * nx + (y0 + 1) * nx + x0 + 1];

        double c00 = c000 * (1 - xf) + c100 * xf;
        double c10 = c010 * (1 - xf) + c110 * xf;
        double c01 = c001 * (1 - xf) + c101 * xf;
        double c11 = c011 * (1 - xf) + c111 * xf;
        double c0 = c00 * (1 - yf) + c10 * yf;
        double c1 = c01 * (1 - yf) + c11 * yf;
        return c0 * (1 - zf) + c1 * zf;
    }

    /// <summary>Nearest-neighbour interpolation.</summary>
    public static double NearestNeighbour(
        ReadOnlySpan<double> field, int nx, int ny, int nz,
        double dx, double dy, double dz,
        double x, double y, double z)
    {
        int xi = Math.Clamp((int)Math.Round(x / dx), 0, nx - 1);
        int yi = Math.Clamp((int)Math.Round(y / dy), 0, ny - 1);
        int zi = Math.Clamp((int)Math.Round(z / dz), 0, nz - 1);
        return field[zi * ny * nx + yi * nx + xi];
    }
}

// ============================================================================
//  Utility: FFT (Cooley-Tukey radix-2)
// ============================================================================

/// <summary>
/// In-place Cooley-Tukey radix-2 FFT for spectral analysis.
/// </summary>
