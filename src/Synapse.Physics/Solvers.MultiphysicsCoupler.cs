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

public sealed class MultiphysicsCoupler : IDisposable
{
    private readonly CouplingConfig _cfg;
    private readonly Dictionary<string, object> _solvers;
    private readonly Dictionary<string, InterfaceData> _interfaceData;
    private readonly List<CouplingLink> _links;
    private readonly ConvergenceMonitor _convergence;
    private readonly LoadBalancer _loadBalancer;
    private double _relaxation;
    private bool _disposed;

    public ConvergenceMonitor Convergence => _convergence;
    public LoadBalancer LoadBalancer => _loadBalancer;

    public MultiphysicsCoupler(CouplingConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        _solvers = new Dictionary<string, object>();
        _interfaceData = new Dictionary<string, InterfaceData>();
        _links = new List<CouplingLink>();
        _convergence = new ConvergenceMonitor();
        _loadBalancer = new LoadBalancer();
        _relaxation = config.Relaxation;
    }

    public void RegisterSolver(string name, object solver)
    {
        _solvers[name] = solver ?? throw new ArgumentNullException(nameof(solver));
    }

    public void RegisterInterface(string name, InterfaceData data)
    {
        _interfaceData[name] = data;
    }

    public void AddCouplingLink(CouplingLink link)
    {
        _links.Add(link ?? throw new ArgumentNullException(nameof(link)));
    }

    /// <summary>
    /// Perform one Gauss-Seidel coupling iteration.
    /// </summary>
    public bool IterateGaussSeidel()
    {
        _convergence.Reset();
        for (int iter = 0; iter < _cfg.MaxIterations; iter++)
        {
            bool anyChanged = false;
            foreach (var link in _links)
            {
                if (!_interfaceData.ContainsKey(link.SourceField) ||
                    !_interfaceData.ContainsKey(link.TargetField))
                    continue;

                var sourceData = _interfaceData[link.SourceField];
                var targetData = _interfaceData[link.TargetField];
                double[] prevTarget = (double[])targetData.FieldData.Clone();

                double[] transferred = link.TransferFunction != null
                    ? link.TransferFunction(sourceData.FieldData)
                    : (double[])sourceData.FieldData.Clone();

                for (int i = 0; i < Math.Min(transferred.Length, targetData.FieldData.Length); i++)
                    targetData.FieldData[i] = (1.0 - _relaxation) * targetData.FieldData[i] +
                                               _relaxation * transferred[i];

                targetData.Timestamp = DateTime.UtcNow;
                anyChanged = true;

                double residual = ConvergenceMonitor.ComputeResidual(targetData.FieldData, prevTarget);
                if (_convergence.CheckConvergence(residual, _cfg.ConvergenceTolerance))
                    return true;
            }

            if (_cfg.AdaptiveRelaxation && _convergence.ResidualHistory.Count >= 2)
            {
                double last = _convergence.ResidualHistory[^1];
                double prev = _convergence.ResidualHistory[^2];
                if (last > prev)
                    _relaxation = Math.Max(_cfg.MinRelaxation, _relaxation * 0.8);
                else
                    _relaxation = Math.Min(_cfg.MaxRelaxation, _relaxation * 1.05);
            }
        }
        return _convergence.Converged;
    }

    /// <summary>
    /// Perform one Jacobi coupling iteration.
    /// </summary>
    public bool IterateJacobi()
    {
        _convergence.Reset();
        var oldData = new Dictionary<string, double[]>();
        foreach (var kv in _interfaceData)
            oldData[kv.Key] = (double[])kv.Value.FieldData.Clone();

        for (int iter = 0; iter < _cfg.MaxIterations; iter++)
        {
            var newData = new Dictionary<string, double[]>();
            foreach (var link in _links)
            {
                if (!oldData.ContainsKey(link.SourceField))
                    continue;

                double[] transferred = link.TransferFunction != null
                    ? link.TransferFunction(oldData[link.SourceField])
                    : (double[])oldData[link.SourceField].Clone();

                var targetData = _interfaceData[link.TargetField];
                double[] mixed = new double[transferred.Length];
                for (int i = 0; i < Math.Min(transferred.Length, targetData.FieldData.Length); i++)
                    mixed[i] = (1.0 - _relaxation) * targetData.FieldData[i] +
                               _relaxation * transferred[i];

                newData[link.TargetField] = mixed;
            }

            foreach (var kv in newData)
            {
                double[] prev = _interfaceData[kv.Key].FieldData;
                _interfaceData[kv.Key].FieldData = kv.Value;
                _interfaceData[kv.Key].Timestamp = DateTime.UtcNow;

                double residual = ConvergenceMonitor.ComputeResidual(kv.Value, prev);
                if (_convergence.CheckConvergence(residual, _cfg.ConvergenceTolerance))
                    return true;
            }

            foreach (var kv in _interfaceData)
                oldData[kv.Key] = (double[])kv.Value.FieldData.Clone();
        }
        return _convergence.Converged;
    }

    /// <summary>
    /// Run the coupled simulation for the configured number of time-steps.
    /// </summary>
    public void Run()
    {
        for (int step = 0; step < _cfg.TimeSteps; step++)
        {
            foreach (var kv in _solvers)
            {
                var type = kv.Value.GetType();
                var stepMethod = type.GetMethod("Step");
                var sw = Stopwatch.StartNew();
                stepMethod?.Invoke(kv.Value, null);
                sw.Stop();
                _loadBalancer.RecordExecutionTime(kv.Key, sw.Elapsed);
            }

            if (step % _cfg.CouplingInterval == 0)
            {
                switch (_cfg.Scheme)
                {
                    case CouplingScheme.GaussSeidel:
                        IterateGaussSeidel();
                        break;
                    case CouplingScheme.Jacobi:
                        IterateJacobi();
                        break;
                    default:
                        IterateGaussSeidel();
                        break;
                }
            }
        }
    }

    public void SetupFSI(
        object fluidSolver, object structSolver,
        int interfaceNodes,
        Func<double[], double[]> pressureToForce,
        Func<double[], double[]> displacementToMesh)
    {
        RegisterSolver("fluid", fluidSolver);
        RegisterSolver("structure", structSolver);
        RegisterInterface("pressure", new InterfaceData("pressure", interfaceNodes));
        RegisterInterface("force", new InterfaceData("force", interfaceNodes));
        RegisterInterface("displacement", new InterfaceData("displacement", interfaceNodes));
        RegisterInterface("mesh", new InterfaceData("mesh", interfaceNodes));

        AddCouplingLink(new CouplingLink
        {
            SourceSolver = "fluid",
            TargetSolver = "structure",
            SourceField = "pressure",
            TargetField = "force",
            TransferFunction = pressureToForce
        });
        AddCouplingLink(new CouplingLink
        {
            SourceSolver = "structure",
            TargetSolver = "fluid",
            SourceField = "displacement",
            TargetField = "mesh",
            TransferFunction = displacementToMesh
        });
    }

    public void SetupCHT(
        object fluidSolver, object solidSolver,
        int interfaceNodes,
        Func<double[], double[]> tempToFlux,
        Func<double[], double[]> fluxToTemp)
    {
        RegisterSolver("fluid", fluidSolver);
        RegisterSolver("solid", solidSolver);
        RegisterInterface("temperature", new InterfaceData("temperature", interfaceNodes));
        RegisterInterface("heatflux", new InterfaceData("heatflux", interfaceNodes));
        RegisterInterface("solid_temperature", new InterfaceData("solid_temperature", interfaceNodes));
        RegisterInterface("solid_heatflux", new InterfaceData("solid_heatflux", interfaceNodes));

        AddCouplingLink(new CouplingLink
        {
            SourceSolver = "fluid",
            TargetSolver = "solid",
            SourceField = "temperature",
            TargetField = "solid_temperature",
            TransferFunction = tempToFlux
        });
        AddCouplingLink(new CouplingLink
        {
            SourceSolver = "solid",
            TargetSolver = "fluid",
            SourceField = "solid_heatflux",
            TargetField = "heatflux",
            TransferFunction = fluxToTemp
        });
    }

    public T? GetSolver<T>(string name) where T : class
    {
        if (_solvers.TryGetValue(name, out var solver))
            return solver as T;
        throw new KeyNotFoundException($"Solver '{name}' not registered.");
    }

    public InterfaceData GetInterfaceData(string name)
    {
        if (_interfaceData.TryGetValue(name, out var data))
            return data;
        throw new KeyNotFoundException($"Interface '{name}' not registered.");
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  Utility: Generic iterative solver for linear systems
// ============================================================================

/// <summary>
/// Generic iterative linear solver used across multiple physics solvers.
/// </summary>
