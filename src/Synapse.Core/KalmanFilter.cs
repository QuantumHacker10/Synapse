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

public class KalmanFilter
{
    private int _n; // state dimension
    private int _m; // measurement dimension
    private double[,] _x; // state estimate [n×1]
    private double[,] _P; // error covariance [n×n]
    private double[,] _Q; // process noise [n×n]
    private double[,] _R; // measurement noise [m×m]
    private double[,] _F; // state transition [n×n]
    private double[,] _H; // measurement model [m×n]

    public KalmanFilter(int stateDim, int measDim)
    {
        _n = stateDim;
        _m = measDim;
        _x = new double[_n, 1];
        _P = new double[_n, _n];
        _Q = new double[_n, _n];
        _R = new double[_m, _m];
        _F = new double[_n, _n];
        _H = new double[_m, _n];
        for (int i = 0; i < _n; i++)
            _F[i, i] = 1;
    }

    public void SetState(double[] state) { for (int i = 0; i < _n; i++) _x[i, 0] = state[i]; }
    public void SetCovariance(double[,] P) => _P = P;
    public void SetProcessNoise(double[,] Q) => _Q = Q;
    public void SetMeasurementNoise(double[,] R) => _R = R;
    public void SetTransition(double[,] F) => _F = F;
    public void SetMeasurementModel(double[,] H) => _H = H;

    public double[] Predict()
    {
        // x = F*x
        _x = MatMul(_F, _x);
        // P = F*P*F' + Q
        _P = MatAdd(MatMul(MatMul(_F, _P), Transpose(_F)), _Q);
        var result = new double[_n];
        for (int i = 0; i < _n; i++)
            result[i] = _x[i, 0];
        return result;
    }

    public double[] Update(double[] measurement)
    {
        var z = new double[_m, 1];
        for (int i = 0; i < _m; i++)
            z[i, 0] = measurement[i];
        // y = z - H*x
        var Hx = MatMul(_H, _x);
        var y = MatSub(z, Hx);
        // S = H*P*H' + R
        var S = MatAdd(MatMul(MatMul(_H, _P), Transpose(_H)), _R);
        // K = P*H'*S⁻¹
        var K = MatMul(MatMul(_P, Transpose(_H)), Inverse(S));
        // x = x + K*y
        _x = MatAdd(_x, MatMul(K, y));
        // P = (I - K*H)*P
        var I = Identity(_n);
        _P = MatMul(MatSub(I, MatMul(K, _H)), _P);
        var result = new double[_n];
        for (int i = 0; i < _n; i++)
            result[i] = _x[i, 0];
        return result;
    }

    public double[] GetState() { var r = new double[_n]; for (int i = 0; i < _n; i++) r[i] = _x[i, 0]; return r; }
    public double GetUncertainty(int i) => _P[i, i];

    private static double[,] MatMul(double[,] A, double[,] B)
    {
        int m = A.GetLength(0), n = B.GetLength(1), p = A.GetLength(1);
        var C = new double[m, n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            { double s = 0; for (int k = 0; k < p; k++) s += A[i, k] * B[k, j]; C[i, j] = s; }
        return C;
    }
    private static double[,] MatAdd(double[,] A, double[,] B)
    { int m = A.GetLength(0), n = A.GetLength(1); var C = new double[m, n]; for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) C[i, j] = A[i, j] + B[i, j]; return C; }
    private static double[,] MatSub(double[,] A, double[,] B)
    { int m = A.GetLength(0), n = A.GetLength(1); var C = new double[m, n]; for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) C[i, j] = A[i, j] - B[i, j]; return C; }
    private static double[,] Transpose(double[,] A)
    { int m = A.GetLength(0), n = A.GetLength(1); var T = new double[n, m]; for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) T[j, i] = A[i, j]; return T; }
    private static double[,] Identity(int n) { var I = new double[n, n]; for (int i = 0; i < n; i++) I[i, i] = 1; return I; }
    private static double[,] Inverse(double[,] A) { int n = A.GetLength(0); var aug = new double[n, 2 * n]; for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) aug[i, j] = A[i, j]; aug[i, n + i] = 1; } for (int i = 0; i < n; i++) { double pivot = aug[i, i]; if (Math.Abs(pivot) < 1e-30) pivot = 1e-30; for (int j = 0; j < 2 * n; j++) aug[i, j] /= pivot; for (int k = 0; k < n; k++) if (k != i) { double f = aug[k, i]; for (int j = 0; j < 2 * n; j++) aug[k, j] -= f * aug[i, j]; } } var inv = new double[n, n]; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) inv[i, j] = aug[i, n + j]; return inv; }
}

/// <summary>Discrete Fourier Transform and FFT utilities for signal processing in physics simulations.</summary>
