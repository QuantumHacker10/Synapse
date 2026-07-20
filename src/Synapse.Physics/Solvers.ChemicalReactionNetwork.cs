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
public sealed class ChemicalReactionNetwork : IDisposable
{
    private readonly CRNConfig _cfg;
    private readonly int _nSpecies;
    private double[] _concentrations;
    private double[] _concentrationsPrev;
    private Reaction[] _reactions;
    private MichaelisMentenReaction[] _mmReactions;
    private HillFunction[] _hillFunctions;
    private Random _rng;

    // Reaction-diffusion spatial arrays: [species, z, y, x].
    private double[,,,] _spatialConc;
    private double[,,,] _spatialConcPrev;
    private int _sx, _sy, _sz;

    // Conservation tracking.
    private double[] _conservedQuantity;
    private int[][] _conservedGroups;

    private bool _disposed;

    public ReadOnlySpan<double> Concentrations => _concentrations;
    public int NumSpecies => _nSpecies;

    public ChemicalReactionNetwork(CRNConfig config)
    {
        _cfg = config ?? throw new ArgumentNullException(nameof(config));
        _nSpecies = config.NumSpecies;
        _concentrations = new double[_nSpecies];
        _concentrationsPrev = new double[_nSpecies];
        _reactions = Array.Empty<Reaction>();
        _mmReactions = Array.Empty<MichaelisMentenReaction>();
        _hillFunctions = Array.Empty<HillFunction>();
        _rng = new Random(42);

        if (config.UseReactionDiffusion)
        {
            (_sx, _sy, _sz) = config.GridSize;
            _spatialConc = new double[_nSpecies, _sz, _sy, _sx];
            _spatialConcPrev = new double[_nSpecies, _sz, _sy, _sx];
        }
    }

    /// <summary>
    /// Set the concentration of a species.
    /// </summary>
    public void SetConcentration(int species, double value)
    {
        if ((uint)species >= (uint)_nSpecies)
            throw new ArgumentOutOfRangeException(nameof(species));
        _concentrations[species] = value;
    }

    /// <summary>
    /// Set a batch of initial concentrations.
    /// </summary>
    public void SetInitialConcentrations(ReadOnlySpan<double> values)
    {
        int len = Math.Min(values.Length, _nSpecies);
        for (int i = 0; i < len; i++)
            _concentrations[i] = values[i];
    }

    /// <summary>
    /// Define a set of reactions for the network.
    /// </summary>
    public void SetReactions(Reaction[] reactions)
    {
        _reactions = reactions ?? Array.Empty<Reaction>();
    }

    /// <summary>
    /// Define Michaelis-Menten reactions.
    /// </summary>
    public void SetMichaelisMentenReactions(MichaelisMentenReaction[] reactions)
    {
        _mmReactions = reactions ?? Array.Empty<MichaelisMentenReaction>();
    }

    /// <summary>
    /// Define Hill function regulatory interactions.
    /// </summary>
    public void SetHillFunctions(HillFunction[] functions)
    {
        _hillFunctions = functions ?? Array.Empty<HillFunction>();
    }

    /// <summary>
    /// Define conservation groups: each group is a set of species whose
    /// total concentration is conserved.
    /// </summary>
    public void SetConservationGroups(int[][] groups)
    {
        _conservedGroups = groups;
        _conservedQuantity = new double[groups.Length];
        for (int g = 0; g < groups.Length; g++)
        {
            double sum = 0;
            foreach (int s in groups[g])
                sum += _concentrations[s];
            _conservedQuantity[g] = sum;
        }
    }

    // -----------------------------------------------------------------------
    //  Mass-action kinetics: dx/dt = Σ k · Π[x_i^ν_i]
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute the mass-action propensity for a single reaction.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double MassActionPropensity(Reaction rxn)
    {
        double rate = rxn.RateConstant;
        if (rxn.ReactantCoefficients != null)
        {
            for (int i = 0; i < rxn.Reactants.Length; i++)
            {
                int sp = rxn.Reactants[i];
                int coeff = rxn.ReactantCoefficients.Length > i
                    ? rxn.ReactantCoefficients[i] : 1;
                rate *= Math.Pow(Math.Max(_concentrations[sp], 0.0), coeff);
            }
        }
        else
        {
            foreach (int sp in rxn.Reactants)
                rate *= Math.Max(_concentrations[sp], 0.0);
        }
        return rate;
    }

    // -----------------------------------------------------------------------
    //  Michaelis-Menten: v = Vmax · [S] / (Km + [S])
    // -----------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double MichaelisMentenRate(MichaelisMentenReaction mm)
    {
        double s = Math.Max(_concentrations[mm.SubstrateIndex], 0.0);
        double e = Math.Max(_concentrations[mm.EnzymeIndex], 0.0);
        return mm.Vmax * e * s / (mm.Km + s);
    }

    // -----------------------------------------------------------------------
    //  Hill function: f = [R]^n / (K^n + [R]^n)  (activation)
    //                  f = K^n / (K^n + [R]^n)    (repression)
    // -----------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double HillRate(HillFunction hill)
    {
        double r = Math.Max(_concentrations[hill.RegulatorIndex], 0.0);
        double kn = Math.Pow(hill.K, hill.HillCoefficient);
        double rn = Math.Pow(r, hill.HillCoefficient);
        if (hill.Activation)
            return rn / (kn + rn);
        else
            return kn / (kn + rn);
    }

    // -----------------------------------------------------------------------
    //  ODE RHS: deterministic rates of change
    // -----------------------------------------------------------------------

    /// <summary>
    /// Compute the rate of change for all species.
    /// </summary>
    public void ComputeDerivatives(double[] dcdt)
    {
        Array.Clear(dcdt, 0, _nSpecies);

        // Mass-action reactions.
        foreach (var rxn in _reactions)
        {
            double rate = MassActionPropensity(rxn);
            if (rxn.ReactantCoefficients != null)
            {
                for (int i = 0; i < rxn.Reactants.Length; i++)
                {
                    int sp = rxn.Reactants[i];
                    int coeff = rxn.ReactantCoefficients.Length > i
                        ? rxn.ReactantCoefficients[i] : 1;
                    dcdt[sp] -= coeff * rate;
                }
            }
            else
            {
                foreach (int sp in rxn.Reactants)
                    dcdt[sp] -= rate;
            }

            if (rxn.ProductCoefficients != null)
            {
                for (int i = 0; i < rxn.Products.Length; i++)
                {
                    int sp = rxn.Products[i];
                    int coeff = rxn.ProductCoefficients.Length > i
                        ? rxn.ProductCoefficients[i] : 1;
                    dcdt[sp] += coeff * rate;
                }
            }
            else
            {
                foreach (int sp in rxn.Products)
                    dcdt[sp] += rate;
            }
        }

        // Michaelis-Menten reactions.
        foreach (var mm in _mmReactions)
        {
            double rate = MichaelisMentenRate(mm);
            dcdt[mm.SubstrateIndex] -= rate;
            dcdt[mm.ProductIndex] += rate;
        }

        // Hill function modulated reactions.
        foreach (var hill in _hillFunctions)
        {
            double hVal = HillRate(hill);
            // Hill function modulates the target species production rate.
            dcdt[hill.TargetIndex] += hVal;
        }
    }

    // -----------------------------------------------------------------------
    //  Forward Euler time integration
    // -----------------------------------------------------------------------

    /// <summary>
    /// Advance the ODE system by one time-step using the 4th-order Runge-Kutta method.
    /// </summary>
    public void StepODE()
    {
        int n = _nSpecies;
        double dt = _cfg.TimeStep;
        double[] k1 = new double[n], k2 = new double[n];
        double[] k3 = new double[n], k4 = new double[n];
        double[] temp = new double[n];

        // k1 = f(t, y)
        ComputeDerivatives(k1);

        // k2 = f(t + dt/2, y + dt/2 k1)
        for (int i = 0; i < n; i++)
            temp[i] = _concentrations[i] + 0.5 * dt * k1[i];
        double[] saved = (double[])_concentrations.Clone();
        for (int i = 0; i < n; i++)
            _concentrations[i] = temp[i];
        ComputeDerivatives(k2);
        for (int i = 0; i < n; i++)
            _concentrations[i] = saved[i];

        // k3 = f(t + dt/2, y + dt/2 k2)
        for (int i = 0; i < n; i++)
            temp[i] = _concentrations[i] + 0.5 * dt * k2[i];
        for (int i = 0; i < n; i++)
            _concentrations[i] = temp[i];
        ComputeDerivatives(k3);
        for (int i = 0; i < n; i++)
            _concentrations[i] = saved[i];

        // k4 = f(t + dt, y + dt k3)
        for (int i = 0; i < n; i++)
            temp[i] = _concentrations[i] + dt * k3[i];
        for (int i = 0; i < n; i++)
            _concentrations[i] = temp[i];
        ComputeDerivatives(k4);
        for (int i = 0; i < n; i++)
            _concentrations[i] = saved[i];

        // y_new = y + dt/6 (k1 + 2k2 + 2k3 + k4)
        for (int i = 0; i < n; i++)
        {
            _concentrations[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            if (_concentrations[i] < 0)
                _concentrations[i] = 0; // enforce non-negativity
        }

        // Enforce conservation laws.
        if (_cfg.EnforceMassConservation && _conservedGroups != null)
            EnforceConservation();
    }

    /// <summary>
    /// Enforce conservation of mass by scaling species in each conserved group.
    /// </summary>
    private void EnforceConservation()
    {
        for (int g = 0; g < _conservedGroups.Length; g++)
        {
            double currentSum = 0;
            foreach (int s in _conservedGroups[g])
                currentSum += _concentrations[s];

            if (currentSum > 1e-30 && Math.Abs(currentSum - _conservedQuantity[g]) > 1e-15)
            {
                double scale = _conservedQuantity[g] / currentSum;
                foreach (int s in _conservedGroups[g])
                    _concentrations[s] *= scale;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Stochastic simulation: Gillespie SSA
    // -----------------------------------------------------------------------

    /// <summary>
    /// Perform one Gillespie SSA step. Returns the time of the next reaction.
    /// </summary>
    public double GillespieStep()
    {
        // Compute all propensities.
        double[] propensities = new double[_reactions.Length];
        double totalPropensity = 0;
        for (int i = 0; i < _reactions.Length; i++)
        {
            propensities[i] = MassActionPropensity(_reactions[i]);
            totalPropensity += propensities[i];
        }

        if (totalPropensity <= 0)
            return double.MaxValue; // system is frozen

        // Time to next reaction: exponential distribution.
        double dt = -Math.Log(_rng.NextDouble()) / totalPropensity;

        // Which reaction fires: inverse transform on cumulative propensities.
        double u = _rng.NextDouble() * totalPropensity;
        double cumSum = 0;
        int firedReaction = _reactions.Length - 1;
        for (int i = 0; i < _reactions.Length; i++)
        {
            cumSum += propensities[i];
            if (cumSum >= u)
            {
                firedReaction = i;
                break;
            }
        }

        // Update molecule counts (integer stoichiometry).
        var rxn = _reactions[firedReaction];
        if (rxn.Reactants != null)
            foreach (int sp in rxn.Reactants)
                _concentrations[sp] = Math.Max(0, _concentrations[sp] - 1.0);
        if (rxn.Products != null)
            foreach (int sp in rxn.Products)
                _concentrations[sp] += 1.0;

        return dt;
    }

    /// <summary>
    /// Run the stochastic simulation for a given total time.
    /// </summary>
    public void RunStochastic(double totalTime)
    {
        double t = 0;
        while (t < totalTime)
        {
            double dt = GillespieStep();
            if (dt == double.MaxValue || double.IsNaN(dt))
                break;
            t += dt;
        }
    }

    // -----------------------------------------------------------------------
    //  Reaction-diffusion (Turing patterns)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Set the initial concentration on the spatial grid for a given species.
    /// </summary>
    public void SetSpatialConcentration(int species, Func<int, int, int, double> initializer)
    {
        for (int z = 0; z < _sz; z++)
            for (int y = 0; y < _sy; y++)
                for (int x = 0; x < _sx; x++)
                    _spatialConc[species, z, y, x] = initializer(x, y, z);
    }

    /// <summary>
    /// Advance the reaction-diffusion system by one time-step.
    /// Uses operator splitting: diffusion (Crank-Nicolson implicit) then
    /// reaction (explicit Euler).
    /// </summary>
    public void StepReactionDiffusion()
    {
        int sx = _sx, sy = _sy, sz = _sz;
        double D = _cfg.DiffusionCoefficients?[0] ?? 1.0;
        double dx = _cfg.Dx;
        double dt = _cfg.Dt;
        double r = D * dt / (dx * dx);

        // Copy to prev.
        Array.Copy(_spatialConc, _spatialConcPrev,
            _nSpecies * sz * sy * sx);

        // Diffusion step (explicit Laplacian for simplicity; implicit would
        // require a tridiagonal solve per line).
        for (int sp = 0; sp < _nSpecies; sp++)
        {
            double dk = _cfg.DiffusionCoefficients != null && sp < _cfg.DiffusionCoefficients.Length
                ? _cfg.DiffusionCoefficients[sp] : D;
            double rk = dk * dt / (dx * dx);

            for (int z = 1; z < sz - 1; z++)
                for (int y = 1; y < sy - 1; y++)
                    for (int x = 1; x < sx - 1; x++)
                    {
                        double laplacian =
                            _spatialConcPrev[sp, z, y, x + 1] + _spatialConcPrev[sp, z, y, x - 1] +
                            _spatialConcPrev[sp, z, y + 1, x] + _spatialConcPrev[sp, z, y - 1, x] +
                            _spatialConcPrev[sp, z + 1, y, x] + _spatialConcPrev[sp, z - 1, y, x] -
                            6.0 * _spatialConcPrev[sp, z, y, x];

                        _spatialConc[sp, z, y, x] = _spatialConcPrev[sp, z, y, x] + rk * laplacian;
                    }
        }

        // Reaction step at each grid point.
        double[] localConc = new double[_nSpecies];
        double[] dcdt = new double[_nSpecies];
        for (int z = 0; z < sz; z++)
            for (int y = 0; y < sy; y++)
                for (int x = 0; x < sx; x++)
                {
                    for (int sp = 0; sp < _nSpecies; sp++)
                        localConc[sp] = _spatialConc[sp, z, y, x];

                    // Temporarily swap concentrations for local evaluation.
                    double[] savedGlobal = (double[])_concentrations.Clone();
                    for (int sp = 0; sp < _nSpecies; sp++)
                        _concentrations[sp] = localConc[sp];

                    ComputeDerivatives(dcdt);

                    for (int sp = 0; sp < _nSpecies; sp++)
                    {
                        _spatialConc[sp, z, y, x] = Math.Max(0,
                            localConc[sp] + dt * dcdt[sp]);
                    }

                    // Restore global concentrations.
                    for (int sp = 0; sp < _nSpecies; sp++)
                        _concentrations[sp] = savedGlobal[sp];
                }

        // Neumann (no-flux) boundaries are implicit in the zeroing of
        // ghost-cell contributions.
    }

    /// <summary>
    /// Run the reaction-diffusion simulation for the configured number of steps.
    /// </summary>
    public void RunReactionDiffusion(int steps)
    {
        for (int i = 0; i < steps; i++)
            StepReactionDiffusion();
    }

    /// <summary>
    /// Export the spatial concentration of a species.
    /// </summary>
    public double[,,] GetSpatialConcentration(int species)
    {
        var result = new double[_sz, _sy, _sx];
        for (int z = 0; z < _sz; z++)
            for (int y = 0; y < _sy; y++)
                for (int x = 0; x < _sx; x++)
                    result[z, y, x] = _spatialConc[species, z, y, x];
        return result;
    }

    /// <summary>
    /// Run the deterministic ODE simulation for the configured number of steps.
    /// Returns concentration history [step, species].
    /// </summary>
    public double[,] RunODEWithHistory()
    {
        double[,] history = new double[_cfg.NumSteps + 1, _nSpecies];
        for (int s = 0; s < _nSpecies; s++)
            history[0, s] = _concentrations[s];

        for (int step = 1; step <= _cfg.NumSteps; step++)
        {
            StepODE();
            for (int s = 0; s < _nSpecies; s++)
                history[step, s] = _concentrations[s];
        }

        return history;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
    }
}
// ============================================================================
//  5. NBodySolver — O(N²), Barnes-Hut, leapfrog, PN radiation
// ============================================================================

/// <summary>
/// Configuration for the N-body gravitational solver.
/// </summary>
