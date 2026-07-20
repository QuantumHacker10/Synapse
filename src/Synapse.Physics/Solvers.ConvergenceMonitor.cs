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

public sealed class ConvergenceMonitor
{
    private readonly List<double> _residualHistory;
    private int _totalIterations;

    public double CurrentResidual { get; private set; }
    public bool Converged { get; private set; }
    public int Iterations => _totalIterations;
    public IReadOnlyList<double> ResidualHistory => _residualHistory;

    public ConvergenceMonitor()
    {
        _residualHistory = new List<double>();
    }

    public void Reset()
    {
        _residualHistory.Clear();
        _totalIterations = 0;
        CurrentResidual = double.MaxValue;
        Converged = false;
    }

    public bool CheckConvergence(double residual, double tolerance)
    {
        CurrentResidual = residual;
        _residualHistory.Add(residual);
        _totalIterations++;
        Converged = residual < tolerance;
        return Converged;
    }

    public static double ComputeResidual(double[] current, double[] previous)
    {
        double sum = 0;
        int n = Math.Min(current.Length, previous.Length);
        for (int i = 0; i < n; i++)
        {
            double diff = current[i] - previous[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum / Math.Max(n, 1));
    }
}
/// <summary>
/// Handles load balancing across coupled solvers.
/// </summary>
