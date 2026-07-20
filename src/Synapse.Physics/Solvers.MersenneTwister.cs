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
public sealed class MersenneTwister
{
    private const int N = 624, M = 397;
    private const uint MatrixA = 0x9908b0df, UpperMask = 0x80000000, LowerMask = 0x7fffffff;
    private readonly uint[] _mt = new uint[N];
    private int _mti = N + 1;

    public MersenneTwister(uint seed = 5489)
    {
        _mt[0] = seed;
        for (int i = 1; i < N; i++)
            _mt[i] = 1812433253 * (_mt[i - 1] ^ (_mt[i - 1] >> 30)) + (uint)i;
    }

    private uint GenerateUInt()
    {
        uint[] mag01 = { 0, MatrixA };
        if (_mti >= N)
        {
            for (int k = 0; k < N - M; k++)
            { uint y = (_mt[k] & UpperMask) | (_mt[k + 1] & LowerMask); _mt[k] = _mt[k + M] ^ (y >> 1) ^ mag01[y & 1]; }
            for (int k = N - M; k < N - 1; k++)
            { uint y = (_mt[k] & UpperMask) | (_mt[k + 1] & LowerMask); _mt[k] = _mt[k + M - N] ^ (y >> 1) ^ mag01[y & 1]; }
            uint yb = (_mt[N - 1] & UpperMask) | (_mt[0] & LowerMask);
            _mt[N - 1] = _mt[M - 1] ^ (yb >> 1) ^ mag01[yb & 1];
            _mti = 0;
        }
        uint y2 = _mt[_mti++];
        y2 ^= y2 >> 11;
        y2 ^= (y2 << 7) & 0x9d2c5680;
        y2 ^= (y2 << 15) & 0xefc60000;
        y2 ^= y2 >> 18;
        return y2;
    }

    public double NextDouble() => GenerateUInt() * (1.0 / 4294967296.0);
    public int NextInt(int min, int max) => min + (int)(GenerateUInt() % (uint)(max - min));

    public double NextGaussian(double mean = 0, double stdDev = 1)
    {
        double u1 = 1.0 - NextDouble(), u2 = 1.0 - NextDouble();
        return mean + stdDev * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    public double NextExponential(double lambda = 1.0) => -Math.Log(1.0 - NextDouble()) / lambda;

    public int NextPoisson(double lambda)
    {
        double L = Math.Exp(-lambda), p = 1.0;
        int k = 0;
        do
        { k++; p *= NextDouble(); } while (p > L);
        return k - 1;
    }
}

// ============================================================================
//  End of Solvers.cs — Synapse Omonia Physics
// ============================================================================
// ============================================================================
//  Additional Utility: Numerical Differentiation
// ============================================================================

/// <summary>
/// High-order numerical differentiation on uniform grids using
/// central differences with boundary corrections.
/// </summary>
