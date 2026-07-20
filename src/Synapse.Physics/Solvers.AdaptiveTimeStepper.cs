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

public sealed class AdaptiveTimeStepper
{
    private readonly double _tolerance;
    private readonly double _minDt;
    private readonly double _maxDt;
    private readonly double _safetyFactor;
    private double _currentDt;
    private double _previousError;

    public double CurrentDt => _currentDt;

    public AdaptiveTimeStepper(
        double initialDt, double tolerance = 1e-6,
        double minDt = 1e-12, double maxDt = 1.0,
        double safetyFactor = 0.9)
    {
        _currentDt = initialDt;
        _tolerance = tolerance;
        _minDt = minDt;
        _maxDt = maxDt;
        _safetyFactor = safetyFactor;
    }

    /// <summary>
    /// Adjust time-step based on the ratio of tolerance to measured error.
    /// Uses the standard PI controller: dt_new = dt_old * (tol/err)^α * safety
    /// with order p from the embedded method (typically p = 4 for RK45).
    /// </summary>
    public double AdjustTimeStep(double error, double order = 4.0)
    {
        if (error < 1e-30)
            error = 1e-30;
        _previousError = error;

        double factor = _safetyFactor * Math.Pow(_tolerance / error, 1.0 / (order + 1.0));
        factor = Math.Clamp(factor, 0.2, 5.0); // limit step-size changes

        _currentDt *= factor;
        _currentDt = Math.Clamp(_currentDt, _minDt, _maxDt);

        return _currentDt;
    }

    /// <summary>
    /// Returns true if the current step should be rejected and retried.
    /// </summary>
    public bool ShouldReject(double error) => error > _tolerance * 10.0;
}

// ============================================================================
//  Additional Utility: Parallel Execution Helpers
// ============================================================================

/// <summary>
/// Partitioning helper for data-parallel loops.
/// </summary>
