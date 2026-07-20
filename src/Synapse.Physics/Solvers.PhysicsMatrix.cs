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

public static class PhysicsMatrix
{
    /// <summary>
    /// Multiply matrix A (m×n) by vector x (n) yielding result (m).
    /// </summary>
    public static void MatVec(ReadOnlySpan<double> A, ReadOnlySpan<double> x,
        Span<double> result, int m, int n)
    {
        for (int i = 0; i < m; i++)
        {
            double sum = 0;
            int row = i * n;
            for (int j = 0; j < n; j++)
                sum += A[row + j] * x[j];
            result[i] = sum;
        }
    }

    /// <summary>
    /// Transpose matrix A (m×n) into AT (n×m).
    /// </summary>
    public static void Transpose(ReadOnlySpan<double> A, Span<double> AT, int m, int n)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                AT[j * m + i] = A[i * n + j];
    }

    /// <summary>
    /// Compute A^T A (n×n) from A (m×n).
    /// </summary>
    public static void GramMatrix(ReadOnlySpan<double> A, Span<double> AtA, int m, int n)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < m; k++)
                    sum += A[k * n + i] * A[k * n + j];
                AtA[i * n + j] = sum;
                AtA[j * n + i] = sum;
            }
    }

    /// <summary>
    /// Symmetric matrix-vector product for a banded system (used in FEM).
    /// Only lower band of width bw is stored.
    /// </summary>
    public static void BandedSymmetricMatVec(ReadOnlySpan<double> band, ReadOnlySpan<double> x,
        Span<double> result, int n, int bw)
    {
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            int rowStart = i * bw;
            int jStart = Math.Max(0, i - bw + 1);
            for (int j = jStart; j <= i; j++)
            {
                int k = rowStart + (i - j);
                if (k >= 0 && k < band.Length)
                    sum += band[k] * x[j];
            }
            // Upper part (symmetric).
            for (int j = i + 1; j < Math.Min(n, i + bw); j++)
            {
                int k = j * bw + (j - i);
                if (k >= 0 && k < band.Length)
                    sum += band[k] * x[j];
            }
            result[i] = sum;
        }
    }

    /// <summary>
    /// Compute trace of an n×n matrix stored in row-major format.
    /// </summary>
    public static double Trace(ReadOnlySpan<double> A, int n)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += A[i * n + i];
        return sum;
    }

    /// <summary>
    /// Compute the Frobenius norm of an m×n matrix.
    /// </summary>
    public static double FrobeniusNorm(ReadOnlySpan<double> A, int m, int n)
    {
        double sum = 0;
        for (int i = 0; i < m * n; i++)
            sum += A[i] * A[i];
        return Math.Sqrt(sum);
    }
}

// ============================================================================
//  Additional Utility: Quadrature Rules
// ============================================================================

/// <summary>
/// Gauss-Legendre quadrature points and weights for numerical integration.
/// </summary>
