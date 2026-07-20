// =============================================================================
// NeatGEvolutionEngine.cs - NEAT-G Evolution Engine Core
// GDNN.Engine - Geometric Deep Neural Network Engine
// Copyright (c) 2024. All rights reserved.
// =============================================================================
// This file is the heart of the G-DNN Engine implementing the NEAT-G
// (NeuroEvolution of Augmented Topologies - Geometric) algorithm.
// It provides comprehensive evolutionary optimization for neural network
// architectures with geometric awareness, semantic crossover, manifold-based
// speciation, and swarm evolution capabilities.
// =============================================================================

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using GDNN.Core.NEAT.Models;
using Synapse.Infrastructure.Logging;

namespace GDNN.Core.NEAT
{

    /// <summary>
    /// The core NEAT-G (NeuroEvolution of Augmented Topologies - Geometric) evolution engine.
    /// This is the central orchestrator that coordinates all evolutionary processes:
    /// population initialization, fitness evaluation, speciation, selection, crossover,
    /// mutation, migration, and stagnation management.
    /// Implements the full evolutionary loop with support for adaptive parameters,
    /// parallel evaluation, and comprehensive diagnostics.
    /// </summary>
    public sealed class NeatGEvolutionEngine : IAsyncDisposable
    {
        private readonly EvolutionConfig _config;
        private readonly Random _rng;
        private readonly ICrossoverStrategy _crossoverStrategy;
        private readonly IMutationStrategy _mutationStrategy;
        private readonly ISpeciationStrategy _speciationStrategy;
        private readonly ISelectionStrategy _selectionStrategy;
        private readonly GenomePopulationManager _populationManager;
        private readonly EvolutionHistoryTracker _historyTracker;
        private readonly SpeciationAnalytics _speciationAnalytics;
        private readonly ParallelEvolutionScheduler _parallelScheduler;
        private readonly MigrationManager _migrationManager;
        private readonly EvolutionDiagnostics _diagnostics;
        private readonly AdaptiveMutationScheduler _adaptiveMutationScheduler;
        private readonly SwarmEvolutionScheduler _swarmScheduler;

        private GenomePopulation _currentPopulation;
        private ImmutableArray<SpeciesInfo> _currentSpecies;
        private EvolutionState _state;
        private int _currentGeneration;
        private long _totalEvaluations;
        private readonly Stopwatch _totalElapsed;
        private readonly CancellationTokenSource _engineCts;
        private readonly List<EvolutionMetrics> _metricsHistory;
        private bool _disposed;
        private int _speciesCreatedThisGeneration;
        private int _speciesEliminatedThisGeneration;

        /// <summary>
        /// Fired when a new generation is completed.
        /// </summary>
        public event EventHandler<GenerationCompletedEventArgs>? GenerationCompleted;

        /// <summary>
        /// Fired when evolution reaches a milestone (new best fitness, etc.).
        /// </summary>
        public event EventHandler<EvolutionMilestoneEventArgs>? MilestoneReached;

        /// <summary>
        /// Fired when the evolution state changes.
        /// </summary>
        public event EventHandler<EvolutionStateChangeEventArgs>? StateChanged;

        /// <summary>
        /// Fired when a migration event occurs.
        /// </summary>
        public event EventHandler<MigrationEventArgs>? MigrationOccurred;

        /// <summary>
        /// Initializes a new instance of the NeatGEvolutionEngine class with default strategies.
        /// </summary>
        /// <param name="config">Evolution configuration parameters.</param>
        public NeatGEvolutionEngine(EvolutionConfig config)
            : this(config, null, null, null, null)
        {
        }

        /// <summary>
        /// Initializes a new instance of the NeatGEvolutionEngine class with custom strategies.
        /// </summary>
        /// <param name="config">Evolution configuration parameters.</param>
        /// <param name="crossoverStrategy">Custom crossover strategy (null for default).</param>
        /// <param name="mutationStrategy">Custom mutation strategy (null for default).</param>
        /// <param name="speciationStrategy">Custom speciation strategy (null for default).</param>
        /// <param name="selectionStrategy">Custom selection strategy (null for default).</param>
        public NeatGEvolutionEngine(
            EvolutionConfig config,
            ICrossoverStrategy? crossoverStrategy,
            IMutationStrategy? mutationStrategy,
            ISpeciationStrategy? speciationStrategy,
            ISelectionStrategy? selectionStrategy)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _rng = config.RandomSeed.HasValue
                ? new Random(config.RandomSeed.Value)
                : new Random();

            _crossoverStrategy = crossoverStrategy ?? new SemanticCrossoverStrategy(config);
            _mutationStrategy = mutationStrategy ?? new ComprehensiveMutationStrategy(config);
            _speciationStrategy = speciationStrategy ?? new ManifoldSpeciationStrategy(config);
            _selectionStrategy = selectionStrategy ?? new SelectionStrategy(config);

            _populationManager = new GenomePopulationManager(config, _rng);
            _historyTracker = new EvolutionHistoryTracker(config);
            _speciationAnalytics = new SpeciationAnalytics(config);
            _parallelScheduler = new ParallelEvolutionScheduler(config);
            _migrationManager = new MigrationManager(config);
            _diagnostics = new EvolutionDiagnostics(config);
            _adaptiveMutationScheduler = new AdaptiveMutationScheduler(config, new MutationRate());
            _swarmScheduler = new SwarmEvolutionScheduler(config);

            _currentPopulation = new GenomePopulation();
            _currentSpecies = ImmutableArray<SpeciesInfo>.Empty;
            _state = EvolutionState.NotStarted;
            _totalElapsed = new Stopwatch();
            _engineCts = new CancellationTokenSource();
            _metricsHistory = new List<EvolutionMetrics>();
        }

        /// <summary>Gets the current evolution state.</summary>
        public EvolutionState State => _state;

        /// <summary>Gets the current generation number.</summary>
        public int CurrentGeneration => _currentGeneration;

        /// <summary>Gets the current population.</summary>
        public GenomePopulation CurrentPopulation => _currentPopulation;

        /// <summary>Gets the current species.</summary>
        public ImmutableArray<SpeciesInfo> CurrentSpecies => _currentSpecies;

        /// <summary>Gets total evaluations performed.</summary>
        public long TotalEvaluations => Interlocked.Read(ref _totalEvaluations);

        /// <summary>Gets the evolution configuration.</summary>
        public EvolutionConfig Configuration => _config;

        /// <summary>Gets the evolution history tracker.</summary>
        public EvolutionHistoryTracker History => _historyTracker;

        /// <summary>Gets the speciation analytics.</summary>
        public SpeciationAnalytics SpeciationAnalytics => _speciationAnalytics;

        /// <summary>Gets the evolution diagnostics.</summary>
        public EvolutionDiagnostics Diagnostics => _diagnostics;

        /// <summary>
        /// Main evolution loop. Runs the complete evolutionary process from initialization
        /// to completion or cancellation.
        /// </summary>
        /// <param name="inputCount">Number of input neurons.</param>
        /// <param name="outputCount">Number of output neurons.</param>
        /// <param name="context">Evaluation context for fitness assessment.</param>
        /// <param name="progressCallback">Optional callback for progress reporting.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>EvolutionResult containing the best genome and evolution statistics.</returns>
        public async Task<EvolutionResult> RunEvolutionAsync(
            int inputCount,
            int outputCount,
            EvaluationContext context,
            IProgress<EvolutionMetrics>? progressCallback = null,
            CancellationToken ct = default)
        {
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, _engineCts.Token);
            var cancellationToken = linkedCts.Token;

            SetState(EvolutionState.Initializing);
            _totalElapsed.Restart();

            _historyTracker.RecordEvent(EvolutionEventType.EvolutionStarted, 0, "Evolution started");

            var validationWarnings = _config.Validate();
            if (validationWarnings.Count > 0)
            {
                foreach (var warning in validationWarnings)
                {
                    _historyTracker.RecordEvent(EvolutionEventType.EvolutionStarted, 0, $"Warning: {warning}");
                }
            }

            InitializePopulation(inputCount, outputCount);

            SetState(EvolutionState.Evaluating);
            await EvaluatePopulationAsync(context, cancellationToken).ConfigureAwait(false);

            SetState(EvolutionState.Speciating);
            _currentSpecies = _speciationStrategy.Speciate(_currentPopulation);

            RegisterSpeciesWithSwarm();

            RecordGenerationMetrics();
            progressCallback?.Report(_metricsHistory.Last());

            double bestFitnessEver = _currentPopulation.BestGenome?.Fitness ?? double.MinValue;
            int generationsWithoutImprovement = 0;

            for (_currentGeneration = 1; _currentGeneration <= _config.MaxGenerations; _currentGeneration++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var generationStopwatch = Stopwatch.StartNew();

                if (_config.UseAdaptiveMutation)
                {
                    AdjustAdaptiveParameters();
                }

                SetState(EvolutionState.Selecting);
                var elites = _populationManager.GetElites(_currentPopulation, _currentSpecies);
                var parents = SelectParents(cancellationToken);

                SetState(EvolutionState.Evolving);
                var offspring = await ProduceOffspringAsync(parents, context, cancellationToken)
                    .ConfigureAwait(false);

                if (_config.EnableMigration && _currentGeneration % _config.MigrationInterval == 0)
                {
                    SetState(EvolutionState.Migrating);
                    await PerformMigrationAsync(cancellationToken).ConfigureAwait(false);
                }

                SetState(EvolutionState.Evaluating);
                var allCandidates = CombinePopulation(elites, offspring);
                await EvaluatePopulationAsync(allCandidates, context, cancellationToken)
                    .ConfigureAwait(false);

                SetState(EvolutionState.Speciating);
                _currentPopulation = CreateNewPopulation(allCandidates);
                _currentSpecies = _speciationStrategy.Speciate(_currentPopulation);

                RegisterSpeciesWithSwarm();

                _populationManager.DetectStagnation(_currentSpecies);
                HandleSpeciesExtinction();

                UpdateSpeciesStagnationCounters();

                generationStopwatch.Stop();

                RecordGenerationMetrics(generationStopwatch.Elapsed);

                var metrics = _metricsHistory.Last();
                progressCallback?.Report(metrics);
                _historyTracker.RecordMetrics(metrics);

                OnGenerationCompleted(new GenerationCompletedEventArgs
                {
                    Generation = _currentGeneration,
                    Metrics = metrics,
                    BestGenome = _currentPopulation.BestGenome
                });

                var currentBest = _currentPopulation.BestGenome?.Fitness ?? double.MinValue;
                if (currentBest > bestFitnessEver)
                {
                    bestFitnessEver = currentBest;
                    generationsWithoutImprovement = 0;
                    OnMilestoneReached(new EvolutionMilestoneEventArgs
                    {
                        Generation = _currentGeneration,
                        BestFitness = bestFitnessEver,
                        MilestoneType = EvolutionMilestoneType.NewBestFitness
                    });
                    _historyTracker.RecordEvent(EvolutionEventType.NewBestFitness, _currentGeneration,
                        $"New best fitness: {bestFitnessEver:F6}");
                }
                else
                {
                    generationsWithoutImprovement++;
                }

                if (_currentPopulation.BestGenome?.Fitness >= _config.TargetFitness)
                {
                    _historyTracker.RecordEvent(EvolutionEventType.EvolutionCompleted, _currentGeneration,
                        $"Target fitness {_config.TargetFitness} reached");
                    break;
                }

                if (generationsWithoutImprovement >= _config.MaxStagnationGenerations &&
                    _config.StagnationDetection == StagnationStrategy.BestFitness)
                {
                    _historyTracker.RecordEvent(EvolutionEventType.StagnationDetected, _currentGeneration,
                        $"Stagnation detected after {generationsWithoutImprovement} generations");
                    OnMilestoneReached(new EvolutionMilestoneEventArgs
                    {
                        Generation = _currentGeneration,
                        BestFitness = bestFitnessEver,
                        MilestoneType = EvolutionMilestoneType.StagnationDetected
                    });
                }

                if (_currentSpecies.All(s => s.IsMarkedForExtinction) && _currentSpecies.Length > 0)
                {
                    _historyTracker.RecordEvent(EvolutionEventType.StagnationDetected, _currentGeneration,
                        "All species marked for extinction - triggering rescue");
                    RescuePopulation(inputCount, outputCount);
                    generationsWithoutImprovement = 0;
                }
            }

            _totalElapsed.Stop();

            SetState(EvolutionState.Complete);

            bool targetReached = _currentPopulation.BestGenome?.Fitness >= _config.TargetFitness;
            string stopReason = targetReached
                ? "Target fitness reached"
                : _currentGeneration > _config.MaxGenerations
                    ? "Maximum generations reached"
                    : "Evolution stopped";

            _historyTracker.RecordEvent(EvolutionEventType.EvolutionCompleted, _currentGeneration, stopReason);

            var result = new EvolutionResult
            {
                FinalPopulation = _currentPopulation,
                BestGenome = _currentPopulation.BestGenome!,
                MetricsHistory = _metricsHistory.ToImmutableArray(),
                TotalGenerations = _currentGeneration,
                TotalEvaluations = TotalEvaluations,
                TotalElapsed = _totalElapsed.Elapsed,
                TargetReached = targetReached,
                StopReason = stopReason,
                Events = _historyTracker.GetEvents()
            };

            return result;
        }

        /// <summary>
        /// Performs a single step of evolution (one generation).
        /// Useful for manual control of the evolution process.
        /// </summary>
        /// <param name="context">Evaluation context.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Evolution metrics for this generation.</returns>
        public async Task<EvolutionMetrics> StepAsync(EvaluationContext context, CancellationToken ct = default)
        {
            if (_state == EvolutionState.NotStarted)
            {
                throw new InvalidOperationException("Evolution has not been initialized. Call RunEvolutionAsync first.");
            }

            _currentGeneration++;

            if (_config.UseAdaptiveMutation)
            {
                AdjustAdaptiveParameters();
            }

            var elites = _populationManager.GetElites(_currentPopulation, _currentSpecies);
            var parents = SelectParents(ct);

            var offspring = await ProduceOffspringAsync(parents, context, ct).ConfigureAwait(false);

            if (_config.EnableMigration && _currentGeneration % _config.MigrationInterval == 0)
            {
                await PerformMigrationAsync(ct).ConfigureAwait(false);
            }

            var allCandidates = CombinePopulation(elites, offspring);
            await EvaluatePopulationAsync(allCandidates, context, ct).ConfigureAwait(false);

            _currentPopulation = CreateNewPopulation(allCandidates);
            _currentSpecies = _speciationStrategy.Speciate(_currentPopulation);

            RegisterSpeciesWithSwarm();

            HandleSpeciesExtinction();
            UpdateSpeciesStagnationCounters();

            RecordGenerationMetrics();

            return _metricsHistory.Last();
        }

        /// <summary>
        /// Initializes the population with random genomes.
        /// </summary>
        /// <param name="inputCount">Number of input neurons.</param>
        /// <param name="outputCount">Number of output neurons.</param>
        public void InitializePopulation(int inputCount, int outputCount)
        {
            SetState(EvolutionState.Initializing);

            _currentPopulation = _populationManager.InitializePopulation(inputCount, outputCount);
            _currentGeneration = 0;

            _historyTracker.RecordEvent(EvolutionEventType.PopulationInitialized, 0,
                $"Initialized population of {_config.PopulationSize} genomes with {inputCount} inputs and {outputCount} outputs");

            OnMilestoneReached(new EvolutionMilestoneEventArgs
            {
                Generation = 0,
                BestFitness = 0,
                MilestoneType = EvolutionMilestoneType.PopulationInitialized
            });
        }

        /// <summary>
        /// Evaluates the fitness of all genomes in the population.
        /// </summary>
        /// <param name="context">Evaluation context.</param>
        /// <param name="ct">Cancellation token.</param>
        public async Task EvaluatePopulationAsync(EvaluationContext context, CancellationToken ct)
        {
            SetState(EvolutionState.Evaluating);

            var unevaluated = _currentPopulation.Genomes.Where(g => !g.IsFitnessValid).ToList();

            if (unevaluated.Count == 0)
                return;

            var resolvedContext = context.ApplyEvolutionConfig(_config);
            var evaluator = new FitnessEvaluator(resolvedContext);
            var results = await _parallelScheduler.EvaluatePopulationParallelAsync(
                unevaluated, evaluator, resolvedContext, ct).ConfigureAwait(false);

            var genomeMap = _currentPopulation.Genomes.ToDictionary(g => g.Id);
            foreach (var evaluated in results)
            {
                if (genomeMap.TryGetValue(evaluated.Id, out var original))
                {
                    genomeMap[evaluated.Id] = evaluated;
                }
            }

            Interlocked.Add(ref _totalEvaluations, unevaluated.Count);

            _currentPopulation = _currentPopulation with
            {
                Genomes = genomeMap.Values.ToImmutableArray()
            };

            _historyTracker.RecordEvent(EvolutionEventType.FitnessEvaluated, _currentGeneration,
                $"Evaluated {unevaluated.Count} genomes");
        }

        /// <summary>
        /// Evaluates the fitness of a specific set of genomes.
        /// </summary>
        /// <param name="genomes">Genomes to evaluate.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="ct">Cancellation token.</param>
        public async Task EvaluatePopulationAsync(
            IReadOnlyList<GeoGenome> genomes,
            EvaluationContext context,
            CancellationToken ct)
        {
            SetState(EvolutionState.Evaluating);

            var unevaluated = genomes.Where(g => !g.IsFitnessValid).ToList();
            if (unevaluated.Count == 0)
                return;

            var resolvedContext = context.ApplyEvolutionConfig(_config);
            var evaluator = new FitnessEvaluator(resolvedContext);
            var results = await _parallelScheduler.EvaluatePopulationParallelAsync(
                unevaluated, evaluator, resolvedContext, ct).ConfigureAwait(false);

            var genomeMap = genomes.ToDictionary(g => g.Id);
            foreach (var evaluated in results)
            {
                if (genomeMap.ContainsKey(evaluated.Id))
                {
                    genomeMap[evaluated.Id] = evaluated;
                }
            }

            Interlocked.Add(ref _totalEvaluations, unevaluated.Count);
        }

        /// <summary>
        /// Selects parent genomes for reproduction using the configured selection method.
        /// </summary>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>List of selected parent genomes.</returns>
        private IReadOnlyList<GeoGenome> SelectParents(CancellationToken ct)
        {
            int parentCount = _config.PopulationSize;
            return _selectionStrategy.SelectParents(_currentPopulation, parentCount, _rng);
        }

        /// <summary>
        /// Produces offspring through crossover and mutation.
        /// </summary>
        /// <param name="parents">Selected parent genomes.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>List of offspring genomes.</returns>
        private async Task<IReadOnlyList<GeoGenome>> ProduceOffspringAsync(
            IReadOnlyList<GeoGenome> parents,
            EvaluationContext context,
            CancellationToken ct)
        {
            var offspring = new List<GeoGenome>();
            var mutationRates = _adaptiveMutationScheduler.GetCurrentRates();
            int offspringNeeded = _config.PopulationSize;

            var tasks = new List<Task<GeoGenome>>();

            for (int i = 0; i < offspringNeeded; i++)
            {
                ct.ThrowIfCancellationRequested();

                int parentIdx1 = _rng.Next(parents.Count);
                int parentIdx2 = _rng.Next(parents.Count);

                GeoGenome parent1 = parents[parentIdx1];
                GeoGenome parent2 = parents[parentIdx2];

                GeoGenome child;

                double crossoverRoll = _rng.NextDouble();
                if (crossoverRoll < _config.CrossoverRate && parent1.Id != parent2.Id)
                {
                    float blendBias = (float)_rng.NextDouble();
                    var crossoverResult = _crossoverStrategy.Crossover(parent1, parent2, blendBias);

                    _diagnostics.RecordCrossover(crossoverResult.StrategyUsed, crossoverResult.IsSuccess);

                    if (crossoverResult.IsSuccess)
                    {
                        child = crossoverResult.Offspring;
                    }
                    else
                    {
                        child = parent1.Clone();
                    }

                    double mutationRoll = _rng.NextDouble();
                    if (mutationRoll < _config.MutationRate)
                    {
                        var mutationResult = _mutationStrategy.Mutate(child, mutationRates, _rng);
                        _diagnostics.RecordMutation(mutationResult.TypeApplied, mutationResult.IsSuccess);
                        child = mutationResult.MutatedGenome;
                    }
                }
                else
                {
                    child = parent1.Clone();

                    double mutationRoll = _rng.NextDouble();
                    if (mutationRoll < _config.MutationRate + _config.AsexualRate)
                    {
                        var mutationResult = _mutationStrategy.Mutate(child, mutationRates, _rng);
                        _diagnostics.RecordMutation(mutationResult.TypeApplied, mutationResult.IsSuccess);
                        child = mutationResult.MutatedGenome;
                    }
                }

                child.Generation = _currentGeneration + 1;
                child.Fitness = double.NaN;
                child.InvalidateFitness();
                child.Id = Guid.NewGuid();
                child.ParentIds = ImmutableArray.Create(parent1.Id, parent2.Id);

                offspring.Add(child);
            }

            return offspring;
        }

        /// <summary>
        /// Combines elites and offspring into a single candidate pool.
        /// </summary>
        private IReadOnlyList<GeoGenome> CombinePopulation(
            IReadOnlyList<GeoGenome> elites,
            IReadOnlyList<GeoGenome> offspring)
        {
            var combined = new List<GeoGenome>();
            combined.AddRange(elites);
            combined.AddRange(offspring);

            while (combined.Count < _config.PopulationSize)
            {
                if (offspring.Count > 0)
                {
                    var clone = offspring[_rng.Next(offspring.Count)].Clone();
                    combined.Add(clone);
                }
                else if (elites.Count > 0)
                {
                    var clone = elites[_rng.Next(elites.Count)].Clone();
                    combined.Add(clone);
                }
                else
                    break;
            }

            return combined.Take(_config.PopulationSize).ToList();
        }

        /// <summary>
        /// Creates a new population from candidates after survival selection.
        /// </summary>
        private GenomePopulation CreateNewPopulation(IReadOnlyList<GeoGenome> candidates)
        {
            var survivors = _selectionStrategy.SelectSurvivors(
                _currentPopulation.Genomes.ToList(),
                candidates.ToList(),
                _config.PopulationSize);

            var survivorGenomes = survivors.Select(g =>
            {
                var clone = g.Clone();
                clone.Age++;
                if (clone.Fitness > clone.BestFitness)
                {
                    clone.BestFitness = clone.Fitness;
                    clone.BestFitnessGeneration = clone.Generation;
                }
                return clone;
            }).ToImmutableArray();

            return new GenomePopulation
            {
                Genomes = survivorGenomes,
                GenerationNumber = _currentGeneration,
                Statistics = _populationManager.ComputePopulationStatistics(
                    new GenomePopulation { Genomes = survivorGenomes },
                    _currentSpecies)
            };
        }

        /// <summary>
        /// Performs migration between species.
        /// </summary>
        private async Task PerformMigrationAsync(CancellationToken ct)
        {
            var migrations = await _swarmScheduler.ScheduleMigrationsAsync(_currentSpecies, _rng)
                .ConfigureAwait(false);

            if (migrations.Count > 0)
            {
                var genomeMap = _currentPopulation.Genomes.ToDictionary(g => g.Id);
                _swarmScheduler.ExecuteMigrations(migrations, genomeMap);

                var updatedGenomes = genomeMap.Values.ToImmutableArray();
                _currentPopulation = _currentPopulation with { Genomes = updatedGenomes };

                foreach (var migration in migrations)
                {
                    _migrationManager.RecordMigration(migration);
                    _historyTracker.RecordEvent(EvolutionEventType.MigrationOccurred, _currentGeneration,
                        $"Migration from species {migration.SourceSpeciesId} to {migration.TargetSpeciesId}",
                        migration.MigratingGenomeId, migration.SourceSpeciesId);

                    OnMigrationOccurred(new MigrationEventArgs
                    {
                        Migration = migration
                    });
                }
            }
        }

        /// <summary>
        /// Handles species extinction and member reassignment.
        /// </summary>
        private void HandleSpeciesExtinction()
        {
            int beforeCount = _currentSpecies.Length;

            var (surviving, reassignments) = _populationManager.HandleExtinction(
                _currentSpecies, _currentPopulation);

            _currentSpecies = surviving;

            if (reassignments.Count > 0)
            {
                var updatedGenomes = _currentPopulation.Genomes.Select(g =>
                {
                    if (reassignments.TryGetValue(g.Id, out int newSpeciesId))
                    {
                        var clone = g.Clone();
                        clone.SpeciesId = newSpeciesId;
                        return clone;
                    }
                    return g;
                }).ToImmutableArray();

                _currentPopulation = _currentPopulation with { Genomes = updatedGenomes };
            }

            int afterCount = _currentSpecies.Length;
            _speciesEliminatedThisGeneration = Math.Max(0, beforeCount - afterCount);
            _speciesCreatedThisGeneration = Math.Max(0, afterCount - beforeCount);

            foreach (var extinct in _currentSpecies.Where(s => s.IsMarkedForExtinction))
            {
                _historyTracker.RecordEvent(EvolutionEventType.SpeciesEliminated, _currentGeneration,
                    $"Species {extinct.Id} marked for extinction (stagnant {extinct.StagnationCounter} generations)",
                    speciesId: extinct.Id);
            }
        }

        /// <summary>
        /// Updates stagnation counters for all species.
        /// </summary>
        private void UpdateSpeciesStagnationCounters()
        {
            _currentSpecies = _currentSpecies.Select(s =>
            {
                bool improved = s.FitnessHistory.Length >= 2 &&
                    s.FitnessHistory[^1] > s.FitnessHistory[^2];

                int newCounter = improved ? 0 : s.StagnationCounter + 1;

                return s with
                {
                    StagnationCounter = newCounter,
                    IsStagnant = newCounter >= _config.MaxStagnationGenerations
                };
            }).ToImmutableArray();
        }

        /// <summary>
        /// Adjusts adaptive mutation parameters based on population dynamics.
        /// </summary>
        private void AdjustAdaptiveParameters()
        {
            if (_metricsHistory.Count < 2)
                return;

            var current = _metricsHistory.Last();
            var previous = _metricsHistory[^2];

            double fitnessImprovement = current.BestFitness - previous.BestFitness;
            double diversity = current.DiversityMetric;

            var mutationSuccessRate = _diagnostics.GetOverallMutationSuccessRate();
            var newRates = _adaptiveMutationScheduler.Adjust(fitnessImprovement, diversity, mutationSuccessRate);

            if (_mutationStrategy is ComprehensiveMutationStrategy comprehensive)
            {
                comprehensive.CurrentPerturbationMagnitude = _adaptiveMutationScheduler.CurrentPerturbationMagnitude;
            }

            if (Math.Abs(_adaptiveMutationScheduler.CurrentMultiplier - 1.0) > 0.1)
            {
                _historyTracker.RecordEvent(EvolutionEventType.MutationRateAdjusted, _currentGeneration,
                    $"Mutation rate multiplier adjusted to {_adaptiveMutationScheduler.CurrentMultiplier:F3}");
            }
        }

        /// <summary>
        /// Registers current species with the swarm evolution scheduler.
        /// </summary>
        private void RegisterSpeciesWithSwarm()
        {
            foreach (var species in _currentSpecies)
            {
                var memberGenomes = _currentPopulation.Genomes
                    .Where(g => species.MemberIds.Contains(g.Id))
                    .ToImmutableArray();

                if (memberGenomes.Length > 0)
                {
                    _swarmScheduler.RegisterSpecies(species.Id, memberGenomes);
                }
            }

            var currentIds = _currentSpecies.Select(s => s.Id).ToHashSet();
            foreach (var registeredId in _swarmScheduler.GetSpeciesIds().ToList())
            {
                if (!currentIds.Contains(registeredId))
                {
                    _swarmScheduler.UnregisterSpecies(registeredId);
                }
            }
        }

        /// <summary>
        /// Rescues a population that has lost all species.
        /// </summary>
        private void RescuePopulation(int inputCount, int outputCount)
        {
            var bestGenome = _currentPopulation.BestGenome;
            var newGenomes = new List<GeoGenome>();

            if (bestGenome != null)
            {
                for (int i = 0; i < Math.Min(10, _config.PopulationSize); i++)
                {
                    var clone = bestGenome.Clone();
                    clone.Id = Guid.NewGuid();
                    clone.InvalidateFitness();
                    var rates = new MutationRate();
                    rates.ScaleAll(2.0);
                    _mutationStrategy.Mutate(clone, rates, _rng);
                    newGenomes.Add(clone);
                }
            }

            while (newGenomes.Count < _config.PopulationSize)
            {
                newGenomes.Add(_populationManager.CreateRandomGenome(inputCount, outputCount, _currentGeneration));
            }

            _currentPopulation = new GenomePopulation
            {
                Genomes = newGenomes.ToImmutableArray(),
                GenerationNumber = _currentGeneration
            };

            _historyTracker.RecordEvent(EvolutionEventType.PopulationInitialized, _currentGeneration,
                $"Population rescued with {newGenomes.Count} genomes");
        }

        /// <summary>
        /// Records metrics for the current generation.
        /// </summary>
        private void RecordGenerationMetrics(TimeSpan? generationTime = null)
        {
            var stats = _populationManager.ComputePopulationStatistics(_currentPopulation, _currentSpecies);

            double diversityIndex = _speciationAnalytics.ComputePopulationDiversity(_currentPopulation);

            foreach (var species in _currentSpecies)
            {
                _historyTracker.RecordSpeciesData(species.Id, species.BestFitness, species.MemberCount);
            }

            _diagnostics.RecordSnapshot(_currentGeneration, _currentPopulation, _currentSpecies);
            _speciationAnalytics.RecordSnapshot(_currentGeneration, _currentSpecies, _currentPopulation);

            double fitnessImprovement = 0;
            if (_metricsHistory.Count > 0)
            {
                fitnessImprovement = stats.BestFitness - _metricsHistory.Last().BestFitness;
            }

            var metrics = new EvolutionMetrics
            {
                Generation = _currentGeneration,
                BestFitness = stats.BestFitness,
                AverageFitness = stats.MeanFitness,
                MedianFitness = stats.MedianFitness,
                StdDevFitness = stats.StdDevFitness,
                SpeciesCount = _currentSpecies.Length,
                DiversityMetric = diversityIndex,
                TotalEvaluations = TotalEvaluations,
                EvaluationsThisGeneration = stats.EvaluationsThisGeneration,
                GenerationTime = generationTime ?? TimeSpan.Zero,
                TotalElapsed = _totalElapsed.Elapsed,
                MutationSuccessRate = _diagnostics.GetOverallMutationSuccessRate(),
                CrossoverSuccessRate = _diagnostics.GetOverallCrossoverSuccessRate(),
                MigrationEvents = _migrationManager.TotalMigrations,
                SpeciesCreated = _speciesCreatedThisGeneration,
                SpeciesEliminated = _speciesEliminatedThisGeneration,
                AverageComplexity = stats.AverageComplexity,
                MaxComplexity = _currentPopulation.Genomes.Length > 0
                    ? _currentPopulation.Genomes.Max(g => g.Complexity)
                    : 0,
                AdaptiveMutationRate = _adaptiveMutationScheduler.CurrentMultiplier,
                CurrentSpeciationThreshold = _speciationStrategy is ManifoldSpeciationStrategy manifold
                    ? manifold.CurrentThreshold
                    : _config.SpeciationThreshold,
                CurrentPerturbationMagnitude = _adaptiveMutationScheduler.CurrentPerturbationMagnitude
            };

            _metricsHistory.Add(metrics);
        }

        /// <summary>
        /// Sets the engine state and fires the StateChanged event.
        /// </summary>
        private void SetState(EvolutionState newState)
        {
            if (_state == newState)
                return;

            var oldState = _state;
            _state = newState;

            StateChanged?.Invoke(this, new EvolutionStateChangeEventArgs
            {
                OldState = oldState,
                NewState = newState,
                Generation = _currentGeneration
            });
        }

        /// <summary>
        /// Raises the GenerationCompleted event.
        /// </summary>
        private void OnGenerationCompleted(GenerationCompletedEventArgs args)
        {
            GenerationCompleted?.Invoke(this, args);
        }

        /// <summary>
        /// Raises the MilestoneReached event.
        /// </summary>
        private void OnMilestoneReached(EvolutionMilestoneEventArgs args)
        {
            MilestoneReached?.Invoke(this, args);
        }

        /// <summary>
        /// Raises the MigrationOccurred event.
        /// </summary>
        private void OnMigrationOccurred(MigrationEventArgs args)
        {
            MigrationOccurred?.Invoke(this, args);
        }

        /// <summary>
        /// Cancels the running evolution.
        /// </summary>
        public void Cancel()
        {
            _engineCts.Cancel();
            SetState(EvolutionState.Cancelled);
            _historyTracker.RecordEvent(EvolutionEventType.EvolutionCompleted, _currentGeneration, "Evolution cancelled by user");
        }

        /// <summary>
        /// Gets the best genome found so far.
        /// </summary>
        public GeoGenome? GetBestGenome()
        {
            return _currentPopulation.BestGenome;
        }

        /// <summary>
        /// Gets evolution metrics history.
        /// </summary>
        public IReadOnlyList<EvolutionMetrics> GetMetricsHistory()
        {
            return _metricsHistory.AsReadOnly();
        }

        /// <summary>
        /// Exports the current evolution state as JSON.
        /// </summary>
        /// <param name="includeEvents">Include detailed event log.</param>
        /// <returns>JSON string.</returns>
        public string ExportStateJson(bool includeEvents = false)
        {
            return _historyTracker.ExportJson(includeEvents);
        }

        /// <inheritdoc/>
        public async ValueTask DisposeAsync()
        {
            if (_disposed)
                return;
            _disposed = true;

            _engineCts.Cancel();
            await _swarmScheduler.DisposeAsync().ConfigureAwait(false);
            await _parallelScheduler.DisposeAsync().ConfigureAwait(false);
            _engineCts.Dispose();
        }
    }

}
