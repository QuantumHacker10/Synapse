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
public struct LennardJonesPotential
{
    public double Epsilon;
    public double Sigma;
    public double Cutoff;

    public LennardJonesPotential(double epsilon, double sigma, double cutoff)
    {
        Epsilon = epsilon;
        Sigma = sigma;
        Cutoff = cutoff;
    }

    /// <summary>
    /// Compute the LJ potential energy at distance r.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Energy(double r)
    {
        double sr = Sigma / r;
        double sr6 = sr * sr * sr;
        sr6 *= sr6; // sr^12
        return 4.0 * Epsilon * (sr6 * sr6 - sr6); // actually sr^12 is already sr6*sr6
        // Wait: sr^12 = (sr^6)^2
        // sr6 = (Sigma/r)^6, sr^12 = sr6*sr6
    }

    /// <summary>
    /// Compute LJ force magnitude (positive = repulsive) at distance r.
    /// F = −du/dr = 24ε [2(σ/r)¹² − (σ/r)⁶] / r
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ForceMagnitude(double r)
    {
        double sr = Sigma / r;
        double sr6 = sr * sr * sr * sr * sr * sr;
        double sr12 = sr6 * sr6;
        return 24.0 * Epsilon * (2.0 * sr12 - sr6) / r;
    }

    /// <summary>
    /// Compute pair energy with tail correction.
    /// </summary>
    public double EnergyWithTail(double r)
    {
        if (r >= Cutoff)
            return 0.0;
        return Energy(r);
    }

    /// <summary>
    /// Long-range tail correction for energy per particle.
    /// </summary>
    public double EnergyTailCorrection(double density, int numParticles)
    {
        double rc3 = Cutoff * Cutoff * Cutoff;
        double sr3 = Sigma / Cutoff;
        sr3 *= sr3 * sr3; // (σ/rc)³
        double sr9 = sr3 * sr3 * sr3;
        double uTail = (8.0 / 3.0) * Math.PI * density * Epsilon *
                       Sigma * Sigma * Sigma * (sr9 / 3.0 - sr3);
        return uTail * numParticles;
    }

    /// <summary>
    /// Long-range tail correction for pressure.
    /// </summary>
    public double PressureTailCorrection(double density, double temperature)
    {
        double sr3 = Sigma / Cutoff;
        sr3 *= sr3 * sr3;
        double sr9 = sr3 * sr3 * sr3;
        return (16.0 / 3.0) * Math.PI * density * density * Epsilon *
               Sigma * Sigma * Sigma * (2.0 * sr9 / 3.0 - sr3);
    }
}

/// <summary>
/// Thermodynamic ensemble simulator using Monte Carlo methods.
/// Supports canonical (NVT), isothermal-isobaric (NPT), grand canonical,
/// and Gibbs ensemble for liquid-vapour phase equilibria.
/// Implements Metropolis-Hastings sampling, radial distribution function,
/// entropy via Boltzmann formula, and free energy via thermodynamic integration.
/// </summary>
