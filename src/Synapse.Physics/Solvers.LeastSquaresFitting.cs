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

public static class LeastSquaresFitting
{
    /// <summary>
    /// Linear regression: y = a + b*x using least squares.
    /// Returns (intercept a, slope b, R²).
    /// </summary>
    public static (double Intercept, double Slope, double R2) Linear(
        ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        int n = Math.Min(x.Length, y.Length);
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        for (int i = 0; i < n; i++)
        {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }

        double denom = n * sumX2 - sumX * sumX;
        if (Math.Abs(denom) < 1e-30)
            return (sumY / n, 0, 0);

        double b = (n * sumXY - sumX * sumY) / denom;
        double a = (sumY - b * sumX) / n;

        double yMean = sumY / n;
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++)
        {
            double yPred = a + b * x[i];
            ssTot += (y[i] - yMean) * (y[i] - yMean);
            ssRes += (y[i] - yPred) * (y[i] - yPred);
        }
        double r2 = ssTot > 1e-30 ? 1.0 - ssRes / ssTot : 0;

        return (a, b, r2);
    }

    /// <summary>
    /// Polynomial least-squares fit of degree d: y = Σ c_k x^k.
    /// Uses normal equations with Vandermonde matrix.
    /// </summary>
    public static double[] PolynomialFit(ReadOnlySpan<double> x, ReadOnlySpan<double> y, int degree)
    {
        int n = Math.Min(x.Length, y.Length);
        int d = degree + 1;

        // Build normal equations: (V^T V) c = V^T y
        double[] A = new double[d * d];
        double[] rhs = new double[d];

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                    sum += Math.Pow(x[k], i + j);
                A[i * d + j] = sum;
            }
            double sumY = 0;
            for (int k = 0; k < n; k++)
                sumY += y[k] * Math.Pow(x[k], i);
            rhs[i] = sumY;
        }

        // Solve via Gaussian elimination with partial pivoting.
        double[] c = new double[d];
        for (int i = 0; i < d; i++)
            c[i] = rhs[i];

        for (int col = 0; col < d; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < d; row++)
                if (Math.Abs(A[row * d + col]) > Math.Abs(A[maxRow * d + col]))
                    maxRow = row;

            // Swap rows.
            for (int j = 0; j < d; j++)
            {
                (A[col * d + j], A[maxRow * d + j]) = (A[maxRow * d + j], A[col * d + j]);
            }
            (c[col], c[maxRow]) = (c[maxRow], c[col]);

            double diag = A[col * d + col];
            if (Math.Abs(diag) < 1e-30)
                continue;

            for (int row = col + 1; row < d; row++)
            {
                double factor = A[row * d + col] / diag;
                for (int j = col; j < d; j++)
                    A[row * d + j] -= factor * A[col * d + j];
                c[row] -= factor * c[col];
            }
        }

        // Back-substitute.
        for (int i = d - 1; i >= 0; i--)
        {
            for (int j = i + 1; j < d; j++)
                c[i] -= A[i * d + j] * c[j];
            c[i] /= A[i * d + i];
        }

        return c;
    }

    /// <summary>
    /// Evaluate polynomial at point x given coefficients.
    /// </summary>
    public static double PolynomialEvaluate(ReadOnlySpan<double> coefficients, double x)
    {
        double result = coefficients[coefficients.Length - 1];
        for (int i = coefficients.Length - 2; i >= 0; i--)
            result = result * x + coefficients[i];
        return result;
    }
}
