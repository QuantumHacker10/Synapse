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
public static class ParallelPartitioner
{
    /// <summary>
    /// Execute a loop body in parallel over the range [0, count).
    /// Simple work-stealing partitioner for physics inner loops.
    /// </summary>
    public static void For(int count, Action<int> body, int maxDegreeOfParallelism = 0)
    {
        if (maxDegreeOfParallelism <= 0)
            maxDegreeOfParallelism = Environment.ProcessorCount;

        int batchSize = Math.Max(1, (count + maxDegreeOfParallelism - 1) / maxDegreeOfParallelism);
        int numBatches = (count + batchSize - 1) / batchSize;

        var tasks = new Task[numBatches];
        for (int b = 0; b < numBatches; b++)
        {
            int start = b * batchSize;
            int end = Math.Min(start + batchSize, count);
            tasks[b] = Task.Run(() =>
            {
                for (int i = start; i < end; i++)
                    body(i);
            });
        }
        Task.WaitAll(tasks);
    }

    /// <summary>
    /// Parallel reduction: sum over [0, count) with a value function.
    /// </summary>
    public static double ParallelSum(int count, Func<int, double> valueFunc,
        int maxDegreeOfParallelism = 0)
    {
        if (maxDegreeOfParallelism <= 0)
            maxDegreeOfParallelism = Environment.ProcessorCount;

        int batchSize = Math.Max(1, (count + maxDegreeOfParallelism - 1) / maxDegreeOfParallelism);
        int numBatches = (count + batchSize - 1) / batchSize;
        double[] partialSums = new double[numBatches];

        var tasks = new Task[numBatches];
        for (int b = 0; b < numBatches; b++)
        {
            int start = b * batchSize;
            int end = Math.Min(start + batchSize, count);
            int batchIdx = b;
            tasks[b] = Task.Run(() =>
            {
                double sum = 0;
                for (int i = start; i < end; i++)
                    sum += valueFunc(i);
                partialSums[batchIdx] = sum;
            });
        }
        Task.WaitAll(tasks);

        double total = 0;
        for (int b = 0; b < numBatches; b++)
            total += partialSums[b];
        return total;
    }
}

// ============================================================================
//  Additional Utility: File I/O Helpers for Physics Data
// ============================================================================

/// <summary>
/// Binary I/O helpers for writing physics field data in VTK-compatible format.
/// </summary>
