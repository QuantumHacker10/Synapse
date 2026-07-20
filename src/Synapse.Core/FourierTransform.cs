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

public static class FourierTransform
{
    /// <summary>Simple DFT — O(n²) for small datasets.</summary>
    public static (double[] real, double[] imag) DFT(double[] input)
    {
        int n = input.Length;
        var re = new double[n];
        var im = new double[n];
        for (int k = 0; k < n; k++)
        {
            double sumRe = 0, sumIm = 0;
            for (int j = 0; j < n; j++)
            {
                double angle = -2.0 * Math.PI * k * j / n;
                sumRe += input[j] * Math.Cos(angle);
                sumIm += input[j] * Math.Sin(angle);
            }
            re[k] = sumRe;
            im[k] = sumIm;
        }
        return (re, im);
    }

    /// <summary>Inverse DFT.</summary>
    public static double[] IDFT(double[] real, double[] imag)
    {
        int n = real.Length;
        var output = new double[n];
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            for (int k = 0; k < n; k++)
            {
                double angle = 2.0 * Math.PI * k * j / n;
                sum += real[k] * Math.Cos(angle) - imag[k] * Math.Sin(angle);
            }
            output[j] = sum / n;
        }
        return output;
    }

    /// <summary>Power spectrum: |X(k)|² = Re² + Im².</summary>
    public static double[] PowerSpectrum(double[] real, double[] imag)
    {
        var ps = new double[real.Length];
        for (int i = 0; i < real.Length; i++)
            ps[i] = real[i] * real[i] + imag[i] * imag[i];
        return ps;
    }

    /// <summary>Magnitude spectrum: |X(k)|.</summary>
    public static double[] MagnitudeSpectrum(double[] real, double[] imag)
    {
        var mag = new double[real.Length];
        for (int i = 0; i < real.Length; i++)
            mag[i] = Math.Sqrt(real[i] * real[i] + imag[i] * imag[i]);
        return mag;
    }

    /// <summary>Phase spectrum: angle(X(k)) = atan2(Im, Re).</summary>
    public static double[] PhaseSpectrum(double[] real, double[] imag)
    {
        var phase = new double[real.Length];
        for (int i = 0; i < real.Length; i++)
            phase[i] = Math.Atan2(imag[i], real[i]);
        return phase;
    }

    /// <summary>Convolution of two signals via direct multiplication (frequency domain).</summary>
    public static double[] Convolve(double[] a, double[] b)
    {
        int n = a.Length + b.Length - 1;
        var result = new double[n];
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
                result[i + j] += a[i] * b[j];
        return result;
    }

    /// <summary>Auto-correlation via direct computation.</summary>
    public static double[] AutoCorrelate(double[] input)
    {
        int n = input.Length;
        var result = new double[n];
        for (int lag = 0; lag < n; lag++)
        {
            double sum = 0;
            for (int i = 0; i < n - lag; i++)
                sum += input[i] * input[i + lag];
            result[lag] = sum / (n - lag);
        }
        return result;
    }

    /// <summary>Cross-correlation of two signals.</summary>
    public static double[] CrossCorrelate(double[] a, double[] b)
    {
        int n = a.Length + b.Length - 1;
        var result = new double[n];
        for (int lag = -(b.Length - 1); lag < a.Length; lag++)
        {
            double sum = 0;
            for (int i = Math.Max(0, lag); i < Math.Min(a.Length, lag + b.Length); i++)
                sum += a[i] * b[i - lag];
            result[lag + b.Length - 1] = sum;
        }
        return result;
    }

    /// <summary>Window functions for spectral analysis.</summary>
    public static double[] HanningWindow(int n) { var w = new double[n]; for (int i = 0; i < n; i++) w[i] = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (n - 1))); return w; }
    public static double[] HammingWindow(int n) { var w = new double[n]; for (int i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.Cos(2.0 * Math.PI * i / (n - 1)); return w; }
    public static double[] BlackmanWindow(int n) { var w = new double[n]; for (int i = 0; i < n; i++) w[i] = 0.42 - 0.5 * Math.Cos(2.0 * Math.PI * i / (n - 1)) + 0.08 * Math.Cos(4.0 * Math.PI * i / (n - 1)); return w; }
    public static double[] BartlettWindow(int n) { var w = new double[n]; for (int i = 0; i < n; i++) w[i] = 1.0 - Math.Abs((i - (n - 1.0) / 2.0) / ((n - 1.0) / 2.0)); return w; }
    public static double[] KaiserWindow(int n, double beta)
    {
        var w = new double[n];
        double a = (n - 1.0) / 2.0;
        for (int i = 0; i < n; i++)
        {
            double x = (i - a) / a;
            w[i] = BesselI0(beta * Math.Sqrt(1.0 - x * x)) / BesselI0(beta);
        }
        return w;
    }
    private static double BesselI0(double x)
    {
        double sum = 1, term = 1;
        for (int k = 1; k < 25; k++)
        { term *= (x * x) / (4.0 * k * k); sum += term; }
        return sum;
    }
}
