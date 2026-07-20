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

public static class FFT
{
    public static void Forward(Span<double> real, Span<double> imag)
    {
        int n = real.Length;
        if ((n & (n - 1)) != 0)
            throw new ArgumentException("Length must be a power of 2.");

        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
            { (real[i], real[j]) = (real[j], real[i]); (imag[i], imag[j]) = (imag[j], imag[i]); }
        }

        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = -2.0 * Math.PI / len;
            double wr = Math.Cos(angle), wi = Math.Sin(angle);
            for (int i = 0; i < n; i += len)
            {
                double curRe = 1.0, curIm = 0.0;
                for (int j = 0; j < len / 2; j++)
                {
                    int a = i + j, b = i + j + len / 2;
                    double tRe = curRe * real[b] - curIm * imag[b];
                    double tIm = curRe * imag[b] + curIm * real[b];
                    real[b] = real[a] - tRe;
                    imag[b] = imag[a] - tIm;
                    real[a] += tRe;
                    imag[a] += tIm;
                    double newRe = curRe * wr - curIm * wi;
                    curIm = curRe * wi + curIm * wr;
                    curRe = newRe;
                }
            }
        }
    }

    public static void Inverse(Span<double> real, Span<double> imag)
    {
        for (int i = 0; i < imag.Length; i++)
            imag[i] = -imag[i];
        Forward(real, imag);
        double inv = 1.0 / real.Length;
        for (int i = 0; i < real.Length; i++)
        { real[i] *= inv; imag[i] = -imag[i] * inv; }
    }

    public static double[] PowerSpectrum(ReadOnlySpan<double> signal)
    {
        int n = signal.Length;
        int fftSize = 1;
        while (fftSize < n)
            fftSize <<= 1;
        double[] re = new double[fftSize], im = new double[fftSize];
        signal.CopyTo(re);
        Forward(re, im);
        int half = fftSize / 2;
        double[] psd = new double[half];
        for (int i = 0; i < half; i++)
            psd[i] = (re[i] * re[i] + im[i] * im[i]) / (fftSize * fftSize);
        return psd;
    }
}

// ============================================================================
//  Utility: Mersenne Twister RNG
// ============================================================================

/// <summary>
/// MT19937 Mersenne Twister random number generator.
/// </summary>
