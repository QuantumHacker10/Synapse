// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    /// <summary>Runs simulations with compiled law bytecodes.</summary>
    public sealed class LawSimulationRunner
    {
        private readonly LivingLawCompiler _compiler;

        public LawSimulationRunner(LivingLawCompiler compiler)
        {
            _compiler = compiler;
        }

        /// <summary>Run a simulation with a single compiled law.</summary>
        public SimulationResult RunSimulation(string lawId, PhysicsField? initialField = null, SimulationConfig? config = null)
        {
            config ??= new SimulationConfig();
            var result = new SimulationResult();
            var field = initialField ?? CreateTestField(config.GridSize);

            if (config.RecordHistory)
            {
                result.Snapshots.Add(field.Clone());
                result.TimeSteps.Add(0f);
                result.EnergyHistory.Add(ComputeFieldEnergy(field));
            }

            var sw = Stopwatch.StartNew();
            int totalSteps = (int)(config.Duration / config.TimeStep);

            try
            {
                for (int step = 0; step < totalSteps; step++)
                {
                    _compiler.ApplyLaw(lawId, field, config.TimeStep);
                    field.Time += config.TimeStep;
                    result.Iterations = step + 1;

                    if (config.RecordHistory && step % config.HistoryInterval == 0)
                    {
                        result.Snapshots.Add(field.Clone());
                        result.TimeSteps.Add(field.Time);
                        result.EnergyHistory.Add(ComputeFieldEnergy(field));
                    }

                    float error = ComputeFieldEnergy(field);
                    result.ErrorHistory.Add(error);

                    if (config.StopCondition != null && config.StopCondition(step, field))
                    {
                        result.Converged = true;
                        break;
                    }

                    if (float.IsNaN(error) || float.IsInfinity(error))
                    {
                        result.ErrorMessage = $"Simulation diverged at step {step}";
                        break;
                    }
                }

                if (result.Converged == false && result.ErrorMessage == null)
                    result.Converged = true;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = ex.Message;
            }

            sw.Stop();
            result.TotalTime = (float)sw.Elapsed.TotalSeconds;
            if (config.RecordHistory)
            {
                result.Snapshots.Add(field.Clone());
                result.TimeSteps.Add(field.Time);
            }
            return result;
        }

        /// <summary>Run a simulation with multiple coupled laws.</summary>
        public SimulationResult RunCoupledSimulation(IReadOnlyList<string> lawIds, PhysicsField? initialField = null, SimulationConfig? config = null)
        {
            config ??= new SimulationConfig();
            var result = new SimulationResult();
            var field = initialField ?? CreateTestField(config.GridSize);

            if (config.RecordHistory)
            {
                result.Snapshots.Add(field.Clone());
                result.TimeSteps.Add(0f);
                result.EnergyHistory.Add(ComputeFieldEnergy(field));
            }

            var sw = Stopwatch.StartNew();
            int totalSteps = (int)(config.Duration / config.TimeStep);

            try
            {
                for (int step = 0; step < totalSteps; step++)
                {
                    foreach (var lawId in lawIds)
                    {
                        _compiler.ApplyLaw(lawId, field, config.TimeStep);
                    }
                    field.Time += config.TimeStep;
                    result.Iterations = step + 1;

                    if (config.RecordHistory && step % config.HistoryInterval == 0)
                    {
                        result.Snapshots.Add(field.Clone());
                        result.TimeSteps.Add(field.Time);
                        result.EnergyHistory.Add(ComputeFieldEnergy(field));
                    }

                    float error = ComputeFieldEnergy(field);
                    result.ErrorHistory.Add(error);

                    if (config.StopCondition != null && config.StopCondition(step, field))
                    {
                        result.Converged = true;
                        break;
                    }

                    if (float.IsNaN(error) || float.IsInfinity(error))
                    {
                        result.ErrorMessage = $"Simulation diverged at step {step}";
                        break;
                    }
                }

                if (result.Converged == false && result.ErrorMessage == null)
                    result.Converged = true;
            }
            catch (Exception ex)
            {
                result.ErrorMessage = ex.Message;
            }

            sw.Stop();
            result.TotalTime = (float)sw.Elapsed.TotalSeconds;
            if (config.RecordHistory)
            {
                result.Snapshots.Add(field.Clone());
                result.TimeSteps.Add(field.Time);
            }
            return result;
        }

        /// <summary>Compare two simulation runs.</summary>
        public ComparisonResult CompareSimulations(SimulationResult a, SimulationResult b)
        {
            int count = Math.Min(a.Snapshots.Count, b.Snapshots.Count);
            if (count == 0)
                return new ComparisonResult(0, 0, 0, 0, 0, 0, Array.Empty<string>(), true);

            float maxDiv = 0f, sumDiv = 0f;
            var diffs = new List<string>();

            for (int i = 0; i < count; i++)
            {
                var comp = LawComparison.ComparePhysicsFields(a.Snapshots[i], b.Snapshots[i]);
                if (comp.MaxDivergence > maxDiv)
                    maxDiv = comp.MaxDivergence;
                sumDiv += comp.MeanDivergence;
            }

            float meanDiv = sumDiv / count;
            float rmse = MathF.Sqrt(sumDiv * sumDiv / count);
            diffs.Add($"Compared {count} snapshots");
            if (MathF.Abs(a.TotalTime - b.TotalTime) > 0.01f)
                diffs.Add($"Total time difference: {a.TotalTime:F3}s vs {b.TotalTime:F3}s");

            return new ComparisonResult(maxDiv, meanDiv, rmse, 0, 0, 0, diffs.ToArray(),
                maxDiv < 0.01f && rmse < 0.01f);
        }

        private PhysicsField CreateTestField(int size)
        {
            var field = new PhysicsField(size, "simulation");
            float cx = size / 2f;
            for (int z = 0; z < size; z++)
                for (int y = 0; y < size; y++)
                    for (int x = 0; x < size; x++)
                    {
                        float dx = x - cx, dy = y - cx, dz = z - cx;
                        float r = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                        field.Temperature[x, y, z] = 300f + 50f * MathF.Exp(-r * r / (size * size));
                        field.Density[x, y, z] = 1.225f;
                        field.Pressure[x, y, z] = 101325f;
                    }
            return field;
        }

        private float ComputeFieldEnergy(PhysicsField field)
        {
            float energy = 0f;
            var temp = field.Temperature;
            for (int i = 0; i < temp.TotalCells; i++)
                energy += temp.Data[i] * temp.Data[i];
            return energy / temp.TotalCells;
        }
    }    // =========================================================================
}
