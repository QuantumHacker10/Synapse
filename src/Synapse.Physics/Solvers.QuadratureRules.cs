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
public static class QuadratureRules
{
    /// <summary>
    /// Get Gauss-Legendre quadrature points and weights on [−1, 1]
    /// for the specified order. Returns (points, weights) arrays.
    /// </summary>
    public static (double[] Points, double[] Weights) GaussLegendre(int order)
    {
        int n = order;
        double[] x = new double[n];
        double[] w = new double[n];

        for (int i = 0; i < n; i++)
        {
            double z = Math.Cos(Math.PI * (4.0 * i + 3.0) / (4.0 * n + 2.0));
            for (int iter = 0; iter < 100; iter++)
            {
                double pp0 = 1.0, pp1 = z;
                for (int j = 1; j < n; j++)
                {
                    double pp2 = ((2.0 * j + 1.0) * z * pp1 - j * pp0) / (j + 1.0);
                    pp0 = pp1;
                    pp1 = pp2;
                }
                double dp = n * (z * pp1 - pp0) / (z * z - 1.0);
                double dz = pp1 / dp;
                z -= dz;
                if (Math.Abs(dz) < 1e-15)
                    break;
            }
            x[i] = z;
            double p0 = 1.0, p1x = z;
            for (int j = 1; j < n; j++)
            {
                double p2 = ((2.0 * j + 1.0) * z * p1x - j * p0) / (j + 1.0);
                p0 = p1x;
                p1x = p2;
            }
            w[i] = 2.0 / ((1.0 - z * z) * p1x * p1x);
        }
        return (x, w);
    }

    /// <summary>
    /// Compute integral of f from a to b using Gauss-Legendre quadrature.
    /// </summary>
    public static double Integrate(Func<double, double> f, double a, double b, int order = 5)
    {
        var (points, weights) = GaussLegendre(order);
        double sum = 0;
        double mid = 0.5 * (a + b);
        double halfLen = 0.5 * (b - a);
        for (int i = 0; i < order; i++)
            sum += weights[i] * f(mid + halfLen * points[i]);
        return halfLen * sum;
    }

    /// <summary>
    /// Simpson's rule for uniform grids: ∫f dx ≈ h/3 [f₀ + 4f₁ + 2f₂ + 4f₃ + ... + fₙ]
    /// </summary>
    public static double Simpson(ReadOnlySpan<double> f, double h)
    {
        int n = f.Length;
        if (n < 3)
            return 0;
        double sum = f[0] + f[n - 1];
        for (int i = 1; i < n - 1; i++)
            sum += (i % 2 == 0 ? 2.0 : 4.0) * f[i];
        return sum * h / 3.0;
    }

    /// <summary>
    /// Trapezoidal rule for uniform grids.
    /// </summary>
    public static double Trapezoidal(ReadOnlySpan<double> f, double h)
    {
        int n = f.Length;
        if (n < 2)
            return 0;
        double sum = 0.5 * (f[0] + f[n - 1]);
        for (int i = 1; i < n - 1; i++)
            sum += f[i];
        return sum * h;
    }
}

// ============================================================================
//  Additional Utility: Coordinate Transforms
// ============================================================================

/// <summary>
/// Coordinate transformation utilities for the physics solvers.
/// Supports Cartesian, cylindrical, and spherical systems.
/// </summary>
