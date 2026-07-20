// SYNAPSE OMNIA — Synapse.Core
// Split from PhysicsState.cs for maintainability.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Synapse.Core;

public class CholeskySolver
{
    private double[,] _L;
    private int _n;

    public CholeskySolver(double[,] matrix) { _n = matrix.GetLength(0); Decompose(matrix); }

    private void Decompose(double[,] A)
    {
        _L = new double[_n, _n];
        for (int i = 0; i < _n; i++)
        {
            double sum = 0;
            for (int k = 0; k < i; k++)
                sum += _L[i, k] * _L[i, k];
            double diag = A[i, i] - sum;
            _L[i, i] = diag > 0 ? Math.Sqrt(diag) : 1e-30;
            for (int j = i + 1; j < _n; j++)
            {
                sum = 0;
                for (int k = 0; k < i; k++)
                    sum += _L[j, k] * _L[i, k];
                _L[j, i] = (A[j, i] - sum) / _L[i, i];
            }
        }
    }

    public double[] Solve(double[] b)
    {
        var y = new double[_n];
        for (int i = 0; i < _n; i++)
        { double s = 0; for (int k = 0; k < i; k++) s += _L[i, k] * y[k]; y[i] = (b[i] - s) / _L[i, i]; }
        var x = new double[_n];
        for (int i = _n - 1; i >= 0; i--)
        { double s = 0; for (int k = i + 1; k < _n; k++) s += _L[k, i] * x[k]; x[i] = (y[i] - s) / _L[i, i]; }
        return x;
    }

    public double[,] Inverse()
    {
        var inv = new double[_n, _n];
        for (int i = 0; i < _n; i++)
        { var e = new double[_n]; e[i] = 1; var col = Solve(e); for (int j = 0; j < _n; j++) inv[j, i] = col[j]; }
        return inv;
    }

    public double Determinant() { double det = 1; for (int i = 0; i < _n; i++) det *= _L[i, i]; return det * det; }
}

/// <summary>Simple Kalman filter for state estimation with process and measurement noise.</summary>
