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

public static class RootFinding
{
    /// <summary>
    /// Brent's method for finding a root of f(x) = 0 in [a, b].
    /// Combines bisection, secant, and inverse quadratic interpolation.
    /// </summary>
    public static double Brent(Func<double, double> f, double a, double b,
        double tolerance = 1e-12, int maxIter = 100)
    {
        double fa = f(a), fb = f(b);
        if (fa * fb > 0)
            throw new ArgumentException("f(a) and f(b) must have opposite signs.");

        double c = a, fc = fa;
        double d = b - a, e = d;

        for (int iter = 0; iter < maxIter; iter++)
        {
            if (Math.Abs(fc) < Math.Abs(fb))
            {
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }

            double tol1 = 2.0 * 1e-12 * Math.Abs(b) + 0.5 * tolerance;
            double m = 0.5 * (c - b);

            if (Math.Abs(m) <= tol1 || fb == 0)
                return b;

            if (Math.Abs(e) >= tol1 && Math.Abs(fa) > Math.Abs(fb))
            {
                double s = fb / fa;
                double p, q;
                if (a == c)
                {
                    p = 2.0 * m * s;
                    q = 1.0 - s;
                }
                else
                {
                    q = fa / fc;
                    double r = fb / fc;
                    p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                }

                if (p > 0)
                    q = -q;
                else
                    p = -p;
                if (2.0 * p < 3.0 * m * q - Math.Abs(tol1 * q) &&
                    2.0 * p < Math.Abs(e * q))
                {
                    e = d;
                    d = p / q;
                }
                else
                {
                    d = m;
                    e = m;
                }
            }
            else
            {
                d = m;
                e = m;
            }

            a = b;
            fa = fb;
            if (Math.Abs(d) > tol1)
                b += d;
            else
                b += (m > 0 ? tol1 : -tol1);

            fb = f(b);
            if (fb * fc > 0)
            {
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }
        }
        return b;
    }

    /// <summary>
    /// Newton-Raphson method: x_{n+1} = x_n − f(x_n)/f'(x_n).
    /// </summary>
    public static double Newton(Func<double, double> f, Func<double, double> df,
        double x0, double tolerance = 1e-12, int maxIter = 50)
    {
        double x = x0;
        for (int i = 0; i < maxIter; i++)
        {
            double fx = f(x);
            double dfx = df(x);
            if (Math.Abs(dfx) < 1e-30)
                break;
            double dx = fx / dfx;
            x -= dx;
            if (Math.Abs(dx) < tolerance)
                break;
        }
        return x;
    }

    /// <summary>
    /// Secant method: x_{n+1} = x_n − f(x_n) * (x_n − x_{n−1}) / (f(x_n) − f(x_{n−1})).
    /// </summary>
    public static double Secant(Func<double, double> f,
        double x0, double x1, double tolerance = 1e-12, int maxIter = 50)
    {
        double f0 = f(x0), f1 = f(x1);
        for (int i = 0; i < maxIter; i++)
        {
            if (Math.Abs(f1 - f0) < 1e-30)
                break;
            double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
            x0 = x1;
            f0 = f1;
            x1 = x2;
            f1 = f(x1);
            if (Math.Abs(x1 - x0) < tolerance)
                break;
        }
        return x1;
    }
}

// ============================================================================
//  Additional Utility: Least-Squares Fitting
// ============================================================================

/// <summary>
/// Least-squares curve fitting utilities for physics data analysis.
/// </summary>
