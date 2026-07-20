// =============================================================================
// NeatGEvolutionEngine.ParallelEvolutionScheduler.cs — NEAT-G partial module
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
    /// Manages parallel fitness evaluation across CPU cores with work-stealing
    /// for load balancing. Provides async evaluation pipeline with cancellation support.
    /// </summary>
    public sealed class ParallelEvolutionScheduler : IAsyncDisposable
    {
        private readonly EvolutionConfig _config;
        private readonly int _maxParallel;
        private long _totalEvaluations;
        private long _successfulEvaluations;
        private readonly ConcurrentQueue<(GeoGenome Genome, EvaluationContext Context, TaskCompletionSource<GeoGenome> Tcs)> _workQueue;
        private readonly List<Task> _workerTasks;
        private readonly CancellationTokenSource _cts;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the ParallelEvolutionScheduler class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public ParallelEvolutionScheduler(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _maxParallel = Math.Max(1, Math.Min(config.MaxParallelEvaluations, Environment.ProcessorCount));
            _workQueue = new ConcurrentQueue<(GeoGenome, EvaluationContext, TaskCompletionSource<GeoGenome>)>();
            _workerTasks = new List<Task>();
            _cts = new CancellationTokenSource();
        }

        /// <summary>Total evaluations performed.</summary>
        public long TotalEvaluations => Interlocked.Read(ref _totalEvaluations);

        /// <summary>Successful evaluations.</summary>
        public long SuccessfulEvaluations => Interlocked.Read(ref _successfulEvaluations);

        /// <summary>
        /// Evaluates a population of genomes in parallel.
        /// </summary>
        /// <param name="genomes">Genomes to evaluate.</param>
        /// <param name="evaluator">Fitness evaluator to use.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Evaluation results.</returns>
        public async Task<IReadOnlyList<GeoGenome>> EvaluatePopulationParallelAsync(
            IReadOnlyList<GeoGenome> genomes,
            IFitnessEvaluator evaluator,
            EvaluationContext context,
            CancellationToken ct)
        {
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, _cts.Token);

            var needsEval = genomes.Where(g => !g.IsFitnessValid).ToList();
            var alreadyValid = genomes.Where(g => g.IsFitnessValid).ToList();

            var results = new GeoGenome[genomes.Count];
            var options = new ParallelOptions
            {
                MaxDegreeOfParallelism = _maxParallel,
                CancellationToken = linkedCts.Token
            };

            var indexMap = new Dictionary<Guid, int>();
            for (int i = 0; i < genomes.Count; i++)
            {
                indexMap[genomes[i].Id] = i;
                results[i] = genomes[i];
            }

            try
            {
                await Parallel.ForEachAsync(needsEval, options, async (genome, token) =>
                {
                    Interlocked.Increment(ref _totalEvaluations);

                    try
                    {
                        var evaluated = await evaluator.EvaluateAsync(genome, context, token)
                            .ConfigureAwait(false);
                        Interlocked.Increment(ref _successfulEvaluations);

                        lock (results)
                        {
                            if (indexMap.TryGetValue(genome.Id, out int idx))
                            {
                                results[idx] = evaluated;
                            }
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        throw;
                    }
                    catch (Exception ex)
                    {
                        SynapseLogger.Default.Warn("NeatGEvolution", $"Genome evaluation failed for '{genome.Id}'; assigning minimum fitness.", ex);
                        genome.Fitness = double.MinValue;
                        lock (results)
                        {
                            if (indexMap.TryGetValue(genome.Id, out int idx))
                            {
                                results[idx] = genome;
                            }
                        }
                    }
                }).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
            }

            return results;
        }

        /// <summary>
        /// Evaluates genomes using a work-stealing queue for better load balancing.
        /// </summary>
        /// <param name="genomes">Genomes to evaluate.</param>
        /// <param name="evaluator">Fitness evaluator.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Evaluation results.</returns>
        public async Task<IReadOnlyList<GeoGenome>> EvaluatePopulationWorkStealingAsync(
            IReadOnlyList<GeoGenome> genomes,
            IFitnessEvaluator evaluator,
            EvaluationContext context,
            CancellationToken ct)
        {
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, _cts.Token);

            var results = new GeoGenome[genomes.Count];
            for (int i = 0; i < genomes.Count; i++)
                results[i] = genomes[i];

            var queue = new ConcurrentQueue<int>();
            for (int i = 0; i < genomes.Count; i++)
            {
                if (!genomes[i].IsFitnessValid)
                    queue.Enqueue(i);
            }

            var workers = new List<Task>();
            var workerLock = new object();
            int completedCount = 0;

            for (int w = 0; w < _maxParallel; w++)
            {
                workers.Add(Task.Run(async () =>
                {
                    while (!linkedCts.Token.IsCancellationRequested)
                    {
                        if (!queue.TryDequeue(out int idx))
                            break;

                        var genome = genomes[idx];
                        Interlocked.Increment(ref _totalEvaluations);

                        try
                        {
                            var evaluated = await evaluator.EvaluateAsync(genome, context, linkedCts.Token)
                                .ConfigureAwait(false);
                            Interlocked.Increment(ref _successfulEvaluations);
                            results[idx] = evaluated;
                        }
                        catch (OperationCanceledException) { throw; }
                        catch (Exception ex)
                        {
                            SynapseLogger.Default.Warn("NeatGEvolution", $"Genome evaluation failed for '{genome.Id}'; assigning minimum fitness.", ex);
                            genome.Fitness = double.MinValue;
                            results[idx] = genome;
                        }

                        lock (workerLock)
                        { completedCount++; }
                    }
                }, linkedCts.Token));
            }

            await Task.WhenAll(workers).ConfigureAwait(false);
            return results;
        }

        /// <inheritdoc/>
        public async ValueTask DisposeAsync()
        {
            if (_disposed)
                return;
            _disposed = true;

            _cts.Cancel();

            while (_workQueue.TryDequeue(out var item))
            {
                item.Tcs.TrySetCanceled();
            }

            try
            {
                await Task.WhenAll(_workerTasks).ConfigureAwait(false);
            }
            catch (OperationCanceledException) { }

            _cts.Dispose();
        }
    }

}
