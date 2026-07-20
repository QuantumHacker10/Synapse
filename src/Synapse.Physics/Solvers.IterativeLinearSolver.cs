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

public static class IterativeLinearSolver
{
    /// <summary>
    /// Solve Ax = b using the Conjugate Gradient method (SPD A).
    /// Matrix A applied via a function y = A*x.
    /// </summary>
    public static int ConjugateGradient(
        int n,
        Func<double[], double[]> matVec,
        ReadOnlySpan<double> b,
        Span<double> x,
        int maxIter = 1000,
        double tolerance = 1e-10)
    {
        double[] r = new double[n], p = new double[n], Ap = new double[n];
        double[] bArr = b.ToArray();
        double[] Ax0 = matVec(x.ToArray());
        for (int i = 0; i < n; i++)
        { r[i] = bArr[i] - Ax0[i]; p[i] = r[i]; }

        double rsOld = 0;
        for (int i = 0; i < n; i++)
            rsOld += r[i] * r[i];
        double bNorm = Math.Sqrt(rsOld);
        if (bNorm < 1e-30)
            bNorm = 1.0;

        int iter;
        for (iter = 0; iter < maxIter; iter++)
        {
            Ap = matVec(p);
            double pAp = 0;
            for (int i = 0; i < n; i++)
                pAp += p[i] * Ap[i];
            if (Math.Abs(pAp) < 1e-30)
                break;

            double alpha = rsOld / pAp;
            for (int i = 0; i < n; i++)
            { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }

            double rsNew = 0;
            for (int i = 0; i < n; i++)
                rsNew += r[i] * r[i];
            if (Math.Sqrt(rsNew) / bNorm < tolerance)
            { iter++; break; }

            double beta = rsNew / rsOld;
            for (int i = 0; i < n; i++)
                p[i] = r[i] + beta * p[i];
            rsOld = rsNew;
        }
        return iter;
    }

    /// <summary>
    /// SOR iteration: solve Ax = b with over-relaxation factor omega.
    /// </summary>
    public static int SOR(
        int n,
        Func<int, int, double> getElement,
        ReadOnlySpan<double> b,
        Span<double> x,
        double omega = 1.5,
        int maxIter = 10000,
        double tolerance = 1e-10)
    {
        double[] xOld = new double[n];
        for (int iter = 0; iter < maxIter; iter++)
        {
            for (int i = 0; i < n; i++)
                xOld[i] = x[i];
            for (int i = 0; i < n; i++)
            {
                double sigma = 0;
                double diag = getElement(i, i);
                for (int j = 0; j < n; j++)
                    if (j != i)
                        sigma += getElement(i, j) * x[j];
                double xGS = (b[i] - sigma) / diag;
                x[i] = (1.0 - omega) * x[i] + omega * xGS;
            }
            double diff = 0;
            for (int i = 0; i < n; i++)
                diff += (x[i] - xOld[i]) * (x[i] - xOld[i]);
            if (Math.Sqrt(diff) < tolerance)
                return iter + 1;
        }
        return maxIter;
    }
}

// ============================================================================
//  Utility: ODE Integrators
// ============================================================================

/// <summary>
/// Generic ODE time integrators for first-order systems dy/dt = f(t, y).
/// </summary>
