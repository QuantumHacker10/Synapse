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
    // LivingLawCompiler — core compiler orchestrating everything
    // =========================================================================

    /// <summary>The Living Law Compiler: loads, compiles, modifies, and applies physical laws.</summary>
    public sealed class LivingLawCompiler
    {
        private readonly LawLibrary _library;
        private readonly LawCompilerConfig _config;
        private readonly LawModificationEngine _modificationEngine;
        private readonly LawValidation _validation;
        private readonly LawInventor _inventor;
        private readonly LawSimulationRunner _simulationRunner;
        private readonly Dictionary<string, LawBytecode> _compiledCache = new();
        private readonly Dictionary<string, LawVersionTree> _versionTrees = new();
        private readonly Dictionary<string, LawApplicator> _applicators = new();
        private readonly ConcurrentDictionary<string, CompilationResult> _compilationResults = new();
        private long _totalCompilationTimeMs;
        private int _totalCompilations;

        public LawLibrary Library => _library;
        public LawCompilerConfig Config => _config;
        public LawModificationEngine ModificationEngine => _modificationEngine;
        public LawValidation Validation => _validation;
        public LawInventor Inventor => _inventor;
        public LawSimulationRunner SimulationRunner => _simulationRunner;
        public LawEventSystem Events { get; } = new();
        public long TotalCompilationTimeMs => Interlocked.Read(ref _totalCompilationTimeMs);
        public int TotalCompilations => Interlocked.CompareExchange(ref _totalCompilations, 0, 0);

        public LivingLawCompiler(LawCompilerConfig? config = null)
        {
            _config = config ?? new LawCompilerConfig();
            _library = LawLibrary.LoadBuiltIn();
            _modificationEngine = new LawModificationEngine(_library);
            _validation = new LawValidation(_library);
            _inventor = new LawInventor(_library);
            _simulationRunner = new LawSimulationRunner(this);
            RegisterDefaultApplicators();
        }

        public LivingLawCompiler(LawLibrary library, LawCompilerConfig? config = null)
        {
            _config = config ?? new LawCompilerConfig();
            _library = library;
            _modificationEngine = new LawModificationEngine(_library);
            _validation = new LawValidation(_library);
            _inventor = new LawInventor(_library);
            _simulationRunner = new LawSimulationRunner(this);
            RegisterDefaultApplicators();
        }

        private void RegisterDefaultApplicators()
        {
            _applicators["heat"] = new HeatApplicator();
            _applicators["wave"] = new WaveApplicator();
            _applicators["elasticity"] = new ElasticityApplicator();
            _applicators["advection"] = new AdvectionApplicator();
            _applicators["diffusion"] = new DiffusionApplicator();
            _applicators["incompressible_ns"] = new IncompressibleNSApplicator();
            _applicators["electromagnetic"] = new ElectromagneticApplicator();
            _applicators["gravity"] = new GravityApplicator();
            _applicators["generic"] = new GenericBytecodeApplicator("temperature");
        }

        /// <summary>Register a custom applicator.</summary>
        public void RegisterApplicator(string key, LawApplicator applicator)
        {
            _applicators[key] = applicator;
        }

        /// <summary>Load a law from the library by ID.</summary>
        public LawEntry? LoadLaw(string lawId) => _library.GetLaw(lawId);

        /// <summary>Compile a law expression string.</summary>
        public CompilationResult Compile(string expression, string? lawId = null)
        {
            var sw = Stopwatch.StartNew();
            Events.Raise(LawEventType.CompilationStarted, lawId, expression);
            try
            {
                var parser = new LawExpressionParser(expression);
                var ast = parser.Parse();

                if (parser.Errors.Count > 0)
                {
                    sw.Stop();
                    Interlocked.Add(ref _totalCompilationTimeMs, sw.ElapsedMilliseconds);
                    Interlocked.Increment(ref _totalCompilations);
                    var errors = parser.Errors.ToArray();
                    Events.Raise(LawEventType.CompilationFailed, lawId, expression,
                        string.Join("; ", errors));
                    return CompilationResult.Fail("Parse errors", errors);
                }

                var bytecode = parser.CompileToBytecode(ast);
                bytecode.OriginalExpression = expression;

                if (_config.EnableValidation)
                {
                    var valResult = LawValidation.ValidateDimensional(ast);
                    if (!valResult.IsValid)
                    {
                        sw.Stop();
                        Interlocked.Add(ref _totalCompilationTimeMs, sw.ElapsedMilliseconds);
                        Interlocked.Increment(ref _totalCompilations);
                        Events.Raise(LawEventType.ValidationFailed, lawId, expression,
                            string.Join("; ", valResult.Errors));
                        return CompilationResult.Fail("Validation errors", valResult.Errors);
                    }
                }

                sw.Stop();
                Interlocked.Add(ref _totalCompilationTimeMs, sw.ElapsedMilliseconds);
                Interlocked.Increment(ref _totalCompilations);

                string cacheKey = lawId ?? expression;
                _compiledCache[cacheKey] = bytecode;

                if (!string.IsNullOrEmpty(lawId))
                    CreateVersionTree(lawId, expression);

                var result = CompilationResult.Ok(
                    $"Compiled successfully in {sw.ElapsedMilliseconds}ms",
                    bytecode, bytecode.InstructionCount, sw.ElapsedMilliseconds);
                _compilationResults[cacheKey] = result;
                Events.Raise(LawEventType.CompilationCompleted, lawId, expression,
                    $"{result.InstructionCount} ops, {sw.ElapsedMilliseconds} ms");
                return result;
            }
            catch (Exception ex)
            {
                sw.Stop();
                Interlocked.Add(ref _totalCompilationTimeMs, sw.ElapsedMilliseconds);
                Interlocked.Increment(ref _totalCompilations);
                Events.Raise(LawEventType.CompilationFailed, lawId, expression, ex.Message);
                return CompilationResult.Fail($"Compilation failed: {ex.Message}", new[] { ex.Message });
            }
        }

        /// <summary>Load and compile a law from the library.</summary>
        public CompilationResult CompileFromLibrary(string lawId)
        {
            var probe = ProbeCompileFromLibrary(lawId);
            if (!probe.Success)
                return CompilationResult.Fail(probe.Error ?? "Compilation failed", new[] { probe.Error ?? "Compilation failed" });

            if (_compiledCache.TryGetValue(lawId, out var bytecode) && bytecode != null)
                return CompilationResult.Ok("Compiled", bytecode, bytecode.InstructionCount, 0);

            return CompilationResult.Fail("Bytecode cache miss after successful probe", new[] { "Bytecode cache miss" });
        }

        /// <summary>Probes direct vs fallback compilation for benchmarking and diagnostics.</summary>
        public LawCompilationProbe ProbeCompileFromLibrary(string lawId)
        {
            var law = _library.GetLaw(lawId);
            if (law == null)
                return new LawCompilationProbe(lawId, "", false, false, $"Law '{lawId}' not found in library");

            _compiledCache.Remove(lawId);
            string normalized = LawExpressionNormalizer.NormalizeForCompilation(law.Expression);
            var direct = Compile(normalized, lawId);
            if (direct.Success)
                return new LawCompilationProbe(lawId, law.Category, true, false);

            _compiledCache.Remove(lawId);
            if (!Enum.TryParse<LawCategory>(law.Category, ignoreCase: true, out var category))
                return new LawCompilationProbe(lawId, law.Category, false, false, direct.Message);

            string fallback = LawExpressionNormalizer.NormalizeForCompilation(
                LawApplicatorMapper.FallbackExpression(category));
            var fallbackResult = Compile(fallback, lawId);
            if (fallbackResult.Success)
                return new LawCompilationProbe(lawId, law.Category, true, true);

            return new LawCompilationProbe(lawId, law.Category, false, false, fallbackResult.Message);
        }

        /// <summary>Hot-reload: modify a law expression and recompile without stopping.</summary>
        public CompilationResult HotReload(string lawId, string newExpression)
        {
            Events.Raise(LawEventType.HotReloadTriggered, lawId, newExpression);
            var law = _library.GetLaw(lawId);
            if (law != null)
                law.Expression = newExpression;

            if (_versionTrees.TryGetValue(lawId, out var tree))
                tree.Commit(newExpression, $"Hot-reload at {DateTime.UtcNow:HH:mm:ss}");

            _compiledCache.Remove(lawId);
            var result = Compile(newExpression, lawId);
            Events.Raise(LawEventType.HotReloadCompleted, lawId, newExpression, result.Message);
            return result;
        }

        /// <summary>Create a version tree for a law expression.</summary>
        public LawVersionTree CreateVersionTree(string lawId, string? initialExpression = null)
        {
            if (_versionTrees.TryGetValue(lawId, out var existing))
                return existing;

            string expr = initialExpression ?? "";
            if (string.IsNullOrEmpty(expr))
            {
                var law = _library.GetLaw(lawId);
                if (law != null)
                    expr = law.Expression;
            }

            var tree = new LawVersionTree(expr);
            _versionTrees[lawId] = tree;
            return tree;
        }

        /// <summary>Get the version tree for a law.</summary>
        public LawVersionTree? GetVersionTree(string lawId) =>
            _versionTrees.TryGetValue(lawId, out var tree) ? tree : null;

        /// <summary>Apply a compiled law to a physics field.</summary>
        public void ApplyLaw(string lawId, PhysicsField field, float? dt = null)
        {
            if (!_compiledCache.TryGetValue(lawId, out var bytecode))
            {
                var result = CompileFromLibrary(lawId);
                if (!result.Success || result.Bytecode == null)
                    throw new InvalidOperationException($"Failed to compile law '{lawId}': {result.Message}");
                bytecode = result.Bytecode;
            }

            float timeStep = dt ?? _config.TimeStep;
            string applicatorKey = DetermineApplicatorKey(lawId);
            if (_applicators.TryGetValue(applicatorKey, out var applicator))
                applicator.Apply(bytecode, field, timeStep, _config);
            else
            {
                var generic = new GenericBytecodeApplicator("temperature");
                generic.Apply(bytecode, field, timeStep, _config);
            }
        }

        /// <summary>Apply a custom compiled bytecode to a field.</summary>
        public void ApplyBytecode(LawBytecode bytecode, PhysicsField field, string targetField = "temperature", float? dt = null)
        {
            float timeStep = dt ?? _config.TimeStep;
            var applicator = new GenericBytecodeApplicator(targetField);
            applicator.Apply(bytecode, field, timeStep, _config);
        }

        private string DetermineApplicatorKey(string lawId)
        {
            var law = _library.GetLaw(lawId);
            if (law == null)
                return "generic";

            Enum.TryParse<LawCategory>(law.Category, ignoreCase: true, out var category);
            return LawApplicatorMapper.Resolve(law.Category, category);
        }

        /// <summary>Number of laws available in the runtime library.</summary>
        public int CatalogLawCount => _library.AllEntries.Count;

        /// <summary>Validate a law expression.</summary>
        public ValidationResult Validate(string expression, string? lawId = null)
        {
            var parser = new LawExpressionParser(expression);
            var ast = parser.Parse();
            LawEntry? knownLaw = lawId != null ? _library.GetLaw(lawId) : null;
            return LawValidation.ValidateDimensional(ast, knownLaw);
        }

        /// <summary>Comprehensive validation of a law.</summary>
        public ValidationResult ComprehensiveValidate(string expression, string? lawId = null, PhysicsField? testField = null)
        {
            return _validation.ComprehensiveValidate(expression, lawId, testField);
        }

        /// <summary>Modify a law using the modification engine.</summary>
        public string ModifyLaw(string lawId, LawModification modification)
        {
            var law = _library.GetLaw(lawId) ?? throw new ArgumentException($"Law '{lawId}' not found");
            string newExpression = _modificationEngine.ApplyModification(law.Expression, modification);
            law.Expression = newExpression;
            return newExpression;
        }

        /// <summary>Modify a law using natural language instruction.</summary>
        public string ModifyLawNaturalLanguage(string lawId, string instruction)
        {
            var law = _library.GetLaw(lawId) ?? throw new ArgumentException($"Law '{lawId}' not found");
            string newExpression = _modificationEngine.ApplyNaturalLanguageModification(law.Expression, instruction);
            law.Expression = newExpression;
            return newExpression;
        }

        /// <summary>Compare two law versions applied to the same field.</summary>
        public ComparisonResult CompareLawVersions(string expressionA, string expressionB, PhysicsField? testField = null)
        {
            var field = testField ?? CreateTestField();
            var fieldB = field.Clone();

            var resultA = Compile(expressionA);
            var resultB = Compile(expressionB);

            if (resultA.Success && resultA.Bytecode != null)
                ApplyBytecode(resultA.Bytecode, field);
            if (resultB.Success && resultB.Bytecode != null)
                ApplyBytecode(resultB.Bytecode, fieldB);

            int editDist = LawVersionTree.ComputeEditDistance(expressionA, expressionB);
            return LawComparison.CompareFields(field.Temperature, fieldB.Temperature, editDist);
        }

        /// <summary>Compare two law library entries applied to a field.</summary>
        public ComparisonResult CompareLaws(string lawIdA, string lawIdB, PhysicsField? testField = null)
        {
            var lawA = _library.GetLaw(lawIdA) ?? throw new ArgumentException($"Law '{lawIdA}' not found");
            var lawB = _library.GetLaw(lawIdB) ?? throw new ArgumentException($"Law '{lawIdB}' not found");
            return CompareLawVersions(lawA.Expression, lawB.Expression, testField);
        }

        /// <summary>Run a simulation with a law.</summary>
        public SimulationResult RunSimulation(string lawId, PhysicsField? initialField = null, SimulationConfig? config = null)
        {
            return _simulationRunner.RunSimulation(lawId, initialField, config);
        }

        /// <summary>Run a coupled simulation with multiple laws.</summary>
        public SimulationResult RunCoupledSimulation(IReadOnlyList<string> lawIds, PhysicsField? initialField = null, SimulationConfig? config = null)
        {
            return _simulationRunner.RunCoupledSimulation(lawIds, initialField, config);
        }

        /// <summary>Create a test field with some initial conditions.</summary>
        private PhysicsField CreateTestField()
        {
            int size = _config.GridSize;
            var field = new PhysicsField(size, "test");
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

        /// <summary>Get compilation statistics.</summary>
        public (int TotalCompilations, long TotalTimeMs, float AvgTimeMs) GetStatistics()
        {
            int total = Interlocked.CompareExchange(ref _totalCompilations, 0, 0);
            long time = Interlocked.Read(ref _totalCompilationTimeMs);
            return (total, time, total > 0 ? (float)time / total : 0f);
        }

        /// <summary>Clear the compilation cache.</summary>
        public void ClearCache() { _compiledCache.Clear(); _compilationResults.Clear(); }

        /// <summary>Get all cached compiled bytecodes.</summary>
        public IReadOnlyDictionary<string, LawBytecode> GetCachedBytecodes() => _compiledCache;

        /// <summary>Compile and execute a law expression with given parameters.</summary>
        public float Evaluate(string expression, PhysicsField? field, Dictionary<string, float>? parameters = null)
        {
            var result = Compile(expression);
            if (!result.Success || result.Bytecode == null)
                throw new InvalidOperationException($"Failed to compile: {result.Message}");
            var interpreter = new BytecodeInterpreter(new GasMeter(_config.GasLimit));
            return interpreter.Execute(result.Bytecode, field, null, parameters);
        }

        /// <summary>Compile and evaluate a law from the library.</summary>
        public float EvaluateLaw(string lawId, PhysicsField? field, Dictionary<string, float>? parameters = null)
        {
            var law = _library.GetLaw(lawId) ?? throw new ArgumentException($"Law '{lawId}' not found");
            var allParams = new Dictionary<string, float>(law.Constants);
            if (parameters != null)
                foreach (var kv in parameters)
                    allParams[kv.Key] = kv.Value;
            return Evaluate(law.Expression, field, allParams);
        }

        /// <summary>Batch compile multiple expressions.</summary>
        public CompilationResult[] CompileBatch(string[] expressions)
        {
            var results = new CompilationResult[expressions.Length];
            for (int i = 0; i < expressions.Length; i++)
                results[i] = Compile(expressions[i]);
            return results;
        }

        /// <summary>Batch compile all laws in a category.</summary>
        public CompilationResult[] CompileCategory(string category)
        {
            var laws = _library.SearchByCategory(category);
            var results = new CompilationResult[laws.Count];
            for (int i = 0; i < laws.Count; i++)
                results[i] = Compile(laws[i].Expression, laws[i].Id);
            return results;
        }

        /// <summary>Get a list of all available applicator keys.</summary>
        public IReadOnlyCollection<string> GetApplicatorKeys() => _applicators.Keys;

        /// <summary>Export the compiler state as JSON.</summary>
        public string ExportState()
        {
            var state = new
            {
                Config = new
                {
                    _config.Tolerance,
                    _config.MaxIterations,
                    _config.TimeStep,
                    _config.CellSize,
                    BoundaryCondition = _config.BoundaryCondition.ToString(),
                    _config.BoundaryValue,
                    _config.GasLimit,
                    _config.CflLimit,
                    _config.GridSize,
                    _config.EnableHotReload,
                    _config.EnableValidation,
                    Solver = _config.Solver.ToString()
                },
                Statistics = GetStatistics(),
                CachedLaws = _compiledCache.Keys.ToArray(),
                VersionTrees = _versionTrees.Keys.ToArray(),
                ModificationHistory = _modificationEngine.History.Count,
                AvailableLaws = _library.AllEntries.Select(e => new { e.Id, e.Name, e.Category }).ToArray(),
                Applicators = _applicators.Keys.ToArray()
            };
            return JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true });
        }

        /// <summary>Invent a new law from a template.</summary>
        public LawEntry InventLaw(string templateName, Dictionary<string, float> parameters, string? lawId = null)
        {
            return _inventor.InventFromTemplate(templateName, parameters, lawId);
        }

        /// <summary>Generate variations of a law expression.</summary>
        public List<string> GenerateVariations(string expression, int count = 10)
        {
            return _inventor.GenerateVariations(expression, count);
        }

        /// <summary>Blend two law expressions.</summary>
        public string BlendLaws(string expressionA, string expressionB, float weightA = 0.5f)
        {
            return _inventor.BlendLaws(expressionA, expressionB, weightA);
        }

        /// <summary>Export all version trees to a dictionary.</summary>
        public Dictionary<string, List<(string Id, string Expression, DateTime Timestamp, string Description)>> ExportAllVersionHistories()
        {
            var result = new Dictionary<string, List<(string, string, DateTime, string)>>();
            foreach (var kv in _versionTrees)
                result[kv.Key] = kv.Value.ExportHistory();
            return result;
        }

        /// <summary>Import a law expression and create a new entry.</summary>
        public LawEntry ImportLaw(string id, string name, string category, string expression, string description = "")
        {
            var entry = new LawEntry
            {
                Id = id,
                Name = name,
                Category = category,
                Expression = expression,
                Description = description
            };
            _library.Register(entry);
            return entry;
        }

        /// <summary>Get all laws in the library.</summary>
        public IReadOnlyList<LawEntry> GetAllLaws() => _library.AllEntries;

        /// <summary>Search laws by name.</summary>
        public IReadOnlyList<LawEntry> SearchLaws(string query) => _library.SearchByName(query);

        /// <summary>Search laws by category.</summary>
        public IReadOnlyList<LawEntry> SearchLawsByCategory(string category) => _library.SearchByCategory(category);

        /// <summary>Remove a compiled law from the cache.</summary>
        public bool RemoveFromCache(string lawId) => _compiledCache.Remove(lawId);

        /// <summary>Check if a law is compiled and cached.</summary>
        public bool IsCompiled(string lawId) => _compiledCache.ContainsKey(lawId);

        /// <summary>Get the compiled bytecode for a law.</summary>
        public LawBytecode? GetCompiledBytecode(string lawId) =>
            _compiledCache.TryGetValue(lawId, out var bc) ? bc : null;

        /// <summary>Reset the compiler state (clear caches, reset statistics).</summary>
        public void Reset()
        {
            ClearCache();
            _versionTrees.Clear();
            Interlocked.Exchange(ref _totalCompilationTimeMs, 0);
            Interlocked.Exchange(ref _totalCompilations, 0);
        }
    }
}
