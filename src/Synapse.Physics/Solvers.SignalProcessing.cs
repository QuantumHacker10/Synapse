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
public static class SignalProcessing
{
    /// <summary>
    /// Apply a simple moving average filter of given window size.
    /// </summary>
    public static void MovingAverage(ReadOnlySpan<double> input, Span<double> output, int windowSize)
    {
        int n = input.Length;
        int half = windowSize / 2;
        double invW = 1.0 / windowSize;

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            int count = 0;
            for (int j = i - half; j <= i + half; j++)
            {
                if (j >= 0 && j < n)
                { sum += input[j]; count++; }
            }
            output[i] = count > 0 ? sum / count : input[i];
        }
    }

    /// <summary>
    /// Apply a Butterworth low-pass filter (2nd order).
    /// </summary>
    public static void ButterworthLowPass(
        ReadOnlySpan<double> input, Span<double> output,
        double cutoffFreq, double sampleFreq)
    {
        double rc = 1.0 / (PhysicsConstants.TwoPi * cutoffFreq);
        double dt = 1.0 / sampleFreq;
        double alpha = dt / (rc + dt);

        double prevIn = input[0], prevOut = input[0];
        output[0] = input[0];

        for (int i = 1; i < input.Length; i++)
        {
            output[i] = alpha * input[i] + alpha * prevIn + (1.0 - 2.0 * alpha) * prevOut;
            prevIn = input[i];
            prevOut = output[i];
        }
    }

    /// <summary>
    /// Compute autocorrelation of a signal (normalised to [−1, 1]).
    /// </summary>
    public static double[] Autocorrelation(ReadOnlySpan<double> signal)
    {
        int n = signal.Length;
        double[] acf = new double[n];

        double mean = 0;
        for (int i = 0; i < n; i++)
            mean += signal[i];
        mean /= n;

        double variance = 0;
        for (int i = 0; i < n; i++)
        {
            double d = signal[i] - mean;
            variance += d * d;
        }
        variance /= n;
        if (variance < 1e-30)
            return acf;

        for (int lag = 0; lag < n; lag++)
        {
            double sum = 0;
            int count = n - lag;
            for (int i = 0; i < count; i++)
                sum += (signal[i] - mean) * (signal[i + lag] - mean);
            acf[lag] = sum / (count * variance);
        }
        return acf;
    }

    /// <summary>
    /// Hilbert transform approximation (for instantaneous amplitude/phase).
    /// </summary>
    public static void HilbertTransform(ReadOnlySpan<double> signal, Span<double> analytic)
    {
        int n = signal.Length;
        int fftSize = 1;
        while (fftSize < n)
            fftSize <<= 1;

        double[] re = new double[fftSize], im = new double[fftSize];
        signal.CopyTo(re);

        FFT.Forward(re, im);

        // Multiply positive frequencies by 2, zero negative frequencies.
        for (int i = 1; i < fftSize / 2; i++)
        {
            re[i] *= 2;
            im[i] *= 2;
            re[fftSize - i] = 0;
            im[fftSize - i] = 0;
        }
        re[0] *= 2;
        im[0] *= 2;

        FFT.Inverse(re, im);

        for (int i = 0; i < n; i++)
            analytic[i] = re[i];
    }

    /// <summary>
    /// Compute instantaneous amplitude (envelope) via Hilbert transform.
    /// </summary>
    public static double[] Envelope(ReadOnlySpan<double> signal)
    {
        int n = signal.Length;
        double[] analytic = new double[n];
        HilbertTransform(signal, analytic);

        double[] env = new double[n];
        for (int i = 0; i < n; i++)
            env[i] = Math.Sqrt(signal[i] * signal[i] + analytic[i] * analytic[i]);
        return env;
    }
}

// ============================================================================
//  Additional Utility: Adaptive Time Stepping
// ============================================================================

/// <summary>
/// Adaptive time-step controller based on embedded Runge-Kutta error estimates.
/// </summary>
