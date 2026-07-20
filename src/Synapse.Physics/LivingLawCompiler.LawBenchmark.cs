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

    /// <summary>Benchmarks compilation and execution performance.</summary>
    public sealed class LawBenchmark
    {
        private readonly LivingLawCompiler _compiler;
        private readonly List<BenchmarkResult> _results = new();

        public IReadOnlyList<BenchmarkResult> Results => _results;

        public LawBenchmark(LivingLawCompiler compiler)
        {
            _compiler = compiler;
        }

        /// <summary>Benchmark expression parsing.</summary>
        public BenchmarkResult BenchmarkParsing(string expression, int iterations = 1000)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var parser = new LawExpressionParser(expression);
                parser.Parse();
            }
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            var result = new BenchmarkResult
            {
                OperationName = "Parsing",
                ElapsedTicks = sw.ElapsedTicks,
                ElapsedMilliseconds = ms,
                Iterations = iterations,
                OpsPerSecond = iterations / (ms / 1000.0),
                AdditionalInfo = $"Expression: {expression}"
            };
            _results.Add(result);
            return result;
        }

        /// <summary>Benchmark bytecode compilation.</summary>
        public BenchmarkResult BenchmarkCompilation(string expression, int iterations = 1000)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                _compiler.Compile(expression);
            }
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            var result = new BenchmarkResult
            {
                OperationName = "Compilation",
                ElapsedTicks = sw.ElapsedTicks,
                ElapsedMilliseconds = ms,
                Iterations = iterations,
                OpsPerSecond = iterations / (ms / 1000.0),
                AdditionalInfo = $"Expression: {expression}"
            };
            _results.Add(result);
            return result;
        }

        /// <summary>Benchmark bytecode execution.</summary>
        public BenchmarkResult BenchmarkExecution(string expression, int iterations = 10000)
        {
            var compResult = _compiler.Compile(expression);
            if (!compResult.Success || compResult.Bytecode == null)
                return new BenchmarkResult { OperationName = "Execution (failed)", AdditionalInfo = compResult.Message };

            var interpreter = new BytecodeInterpreter();
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                interpreter.Execute(compResult.Bytecode, null, null, new Dictionary<string, float>
                {
                    ["x"] = 1f,
                    ["y"] = 2f,
                    ["z"] = 3f,
                    ["t"] = 0.5f,
                    ["T"] = 300f,
                    ["P"] = 101325f,
                    ["rho"] = 1.225f,
                    ["v"] = 10f,
                    ["c"] = 340f,
                    ["k"] = 100f
                });
            }
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            var result = new BenchmarkResult
            {
                OperationName = "Execution",
                ElapsedTicks = sw.ElapsedTicks,
                ElapsedMilliseconds = ms,
                Iterations = iterations,
                OpsPerSecond = iterations / (ms / 1000.0),
                AdditionalInfo = $"Instructions: {compResult.InstructionCount}"
            };
            _results.Add(result);
            return result;
        }

        /// <summary>Benchmark law application to a field.</summary>
        public BenchmarkResult BenchmarkApplication(string lawId, int gridSize = 32, int iterations = 10)
        {
            var field = new PhysicsField(gridSize, "benchmark");
            float cx = gridSize / 2f;
            for (int z = 0; z < gridSize; z++)
                for (int y = 0; y < gridSize; y++)
                    for (int x = 0; x < gridSize; x++)
                    {
                        float dx = x - cx, dy = y - cx, dz = z - cx;
                        float r = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                        field.Temperature[x, y, z] = 300f + 50f * MathF.Exp(-r * r / (gridSize * gridSize));
                        field.Density[x, y, z] = 1.225f;
                        field.Pressure[x, y, z] = 101325f;
                    }

            long memBefore = GC.GetTotalMemory(true);
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                _compiler.ApplyLaw(lawId, field);
            }
            sw.Stop();
            long memAfter = GC.GetTotalMemory(false);

            double ms = sw.Elapsed.TotalMilliseconds;
            var result = new BenchmarkResult
            {
                OperationName = "Application",
                ElapsedTicks = sw.ElapsedTicks,
                ElapsedMilliseconds = ms,
                Iterations = iterations,
                OpsPerSecond = iterations / (ms / 1000.0),
                MemoryBytes = memAfter - memBefore,
                AdditionalInfo = $"Grid: {gridSize}^3, Law: {lawId}"
            };
            _results.Add(result);
            return result;
        }

        /// <summary>Run a comprehensive benchmark suite.</summary>
        public List<BenchmarkResult> RunFullBenchmark()
        {
            _results.Clear();
            var expressions = new[]
            {
                "sin(x) + cos(y)", "exp(-x^2)", "sqrt(x^2 + y^2 + z^2)",
                "x*y + y*z + z*x", "log(abs(x) + 1)", "min(max(x, 0), 1)"
            };

            foreach (var expr in expressions)
            {
                BenchmarkParsing(expr, 1000);
                BenchmarkCompilation(expr, 1000);
                BenchmarkExecution(expr, 10000);
            }

            var lawIds = new[] { "heat_equation", "wave_equation", "hooke_law" };
            foreach (var id in lawIds)
            {
                var law = _compiler.LoadLaw(id);
                if (law != null)
                {
                    BenchmarkCompilation(law.Expression, 500);
                    BenchmarkApplication(id, 32, 5);
                }
            }

            return _results.ToList();
        }

        /// <summary>Generate a summary report.</summary>
        public string GenerateReport()
        {
            var sb = new StringBuilder();
            sb.AppendLine("=== Living Law Compiler Benchmark Report ===");
            sb.AppendLine($"Date: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss UTC}");
            sb.AppendLine($"Total benchmarks: {_results.Count}");
            sb.AppendLine();
            foreach (var r in _results)
            {
                sb.AppendLine($"--- {r.OperationName} ---");
                sb.AppendLine($"  Iterations: {r.Iterations}");
                sb.AppendLine($"  Total time: {r.ElapsedMilliseconds:F2} ms");
                sb.AppendLine($"  Per iteration: {r.ElapsedMilliseconds / r.Iterations:F4} ms");
                sb.AppendLine($"  Ops/sec: {r.OpsPerSecond:F0}");
                if (r.MemoryBytes != 0)
                    sb.AppendLine($"  Memory delta: {r.MemoryBytes:N0} bytes");
                if (r.AdditionalInfo != null)
                    sb.AppendLine($"  Info: {r.AdditionalInfo}");
                sb.AppendLine();
            }
            return sb.ToString();
        }
    }
}
