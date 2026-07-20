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
public static class ODEIntegrators
{
    /// <summary>4th-order Runge-Kutta step.</summary>
    public static double[] RK4(
        Func<double, double[], double[]> rhs,
        double t, ReadOnlySpan<double> y, double dt)
    {
        int n = y.Length;
        double[] yArr = y.ToArray();
        double[] k1 = rhs(t, yArr);

        double[] y2 = new double[n];
        for (int i = 0; i < n; i++)
            y2[i] = y[i] + 0.5 * dt * k1[i];
        double[] k2 = rhs(t + 0.5 * dt, y2);

        double[] y3 = new double[n];
        for (int i = 0; i < n; i++)
            y3[i] = y[i] + 0.5 * dt * k2[i];
        double[] k3 = rhs(t + 0.5 * dt, y3);

        double[] y4 = new double[n];
        for (int i = 0; i < n; i++)
            y4[i] = y[i] + dt * k3[i];
        double[] k4 = rhs(t + dt, y4);

        double[] result = new double[n];
        for (int i = 0; i < n; i++)
            result[i] = y[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        return result;
    }

    /// <summary>Crank-Nicolson step with fixed-point iteration.</summary>
    public static double[] CrankNicolson(
        Func<double, double[], double[]> rhs,
        double t, ReadOnlySpan<double> y, double dt, int iters = 10)
    {
        int n = y.Length;
        double[] yn = y.ToArray();
        double[] f0 = rhs(t, yn);
        double[] yn1 = new double[n];
        for (int i = 0; i < n; i++)
            yn1[i] = yn[i] + dt * f0[i];

        for (int iter = 0; iter < iters; iter++)
        {
            double[] f1 = rhs(t + dt, yn1);
            for (int i = 0; i < n; i++)
                yn1[i] = yn[i] + 0.5 * dt * (f0[i] + f1[i]);
        }
        return yn1;
    }
}

// ============================================================================
//  Utility: Spatial Interpolation
// ============================================================================

/// <summary>
/// Spatial interpolation utilities for non-matching grid data transfer.
/// </summary>
