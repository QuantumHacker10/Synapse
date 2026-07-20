// =============================================================================
// NeatGEvolutionEngine.SwarmEvolutionScheduler.cs — NEAT-G partial module
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
    /// Manages distributed evolution with gene migration between species pools.
    /// Uses Channel-based message passing for inter-species communication and
    /// ConcurrentDictionary-based pool management for thread-safe genome storage.
    /// Supports background evolution loops for parallel species evolution.
    /// </summary>
    public sealed class SwarmEvolutionScheduler : IAsyncDisposable
    {
        private readonly EvolutionConfig _config;
        private readonly ConcurrentDictionary<int, Channel<MigrationEvent>> _migrationChannels;
        private readonly ConcurrentDictionary<int, ImmutableArray<GeoGenome>> _speciesPools;
        private readonly ChannelWriter<MigrationEvent> _migrationBusWriter;
        private readonly ChannelReader<MigrationEvent> _migrationBusReader;
        private readonly CancellationTokenSource _backgroundCts;
        private readonly Task _backgroundLoop;
        private readonly object _poolLock = new();
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the SwarmEvolutionScheduler class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public SwarmEvolutionScheduler(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _migrationChannels = new ConcurrentDictionary<int, Channel<MigrationEvent>>();
            _speciesPools = new ConcurrentDictionary<int, ImmutableArray<GeoGenome>>();

            var busChannel = Channel.CreateBounded<MigrationEvent>(new BoundedChannelOptions(1000)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = false
            });
            _migrationBusWriter = busChannel.Writer;
            _migrationBusReader = busChannel.Reader;

            _backgroundCts = new CancellationTokenSource();
            _backgroundLoop = Task.Run(() => MigrationBusLoop(_backgroundCts.Token));
        }

        /// <summary>
        /// Registers a species pool for migration tracking.
        /// </summary>
        /// <param name="speciesId">The species identifier.</param>
        /// <param name="genomes">Initial genomes for this species.</param>
        public void RegisterSpecies(int speciesId, ImmutableArray<GeoGenome> genomes)
        {
            _speciesPools[speciesId] = genomes;

            var channel = Channel.CreateBounded<MigrationEvent>(new BoundedChannelOptions(100)
            {
                FullMode = BoundedChannelFullMode.DropOldest
            });
            _migrationChannels[speciesId] = channel;
        }

        /// <summary>
        /// Unregisters a species pool.
        /// </summary>
        /// <param name="speciesId">The species to unregister.</param>
        public void UnregisterSpecies(int speciesId)
        {
            _speciesPools.TryRemove(speciesId, out _);
            if (_migrationChannels.TryRemove(speciesId, out var channel))
            {
                channel.Writer.TryComplete();
            }
        }

        /// <summary>
        /// Schedules migrations between species pools based on fitness and semantic compatibility.
        /// </summary>
        /// <param name="speciesInfos">Current species information.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>A list of migration events to execute.</returns>
        public async Task<IReadOnlyList<MigrationEvent>> ScheduleMigrationsAsync(
            ImmutableArray<SpeciesInfo> speciesInfos,
            Random rng)
        {
            if (!_config.EnableMigration)
                return Array.Empty<MigrationEvent>();

            var migrations = new List<MigrationEvent>();
            var speciesIds = speciesInfos
                .Where(s => s.MemberCount >= _config.SpeciesMinimumSize)
                .Select(s => s.Id)
                .ToList();

            if (speciesIds.Count < 2)
                return migrations;

            int maxMigrations = Math.Min(_config.MaxMigrationsPerGeneration,
                (int)(speciesIds.Count * _config.MigrationRate));

            for (int i = 0; i < maxMigrations; i++)
            {
                int sourceIdx = rng.Next(speciesIds.Count);
                int targetIdx = rng.Next(speciesIds.Count);
                if (sourceIdx == targetIdx)
                    continue;

                int sourceId = speciesIds[sourceIdx];
                int targetId = speciesIds[targetIdx];

                if (!_speciesPools.TryGetValue(sourceId, out var sourcePool) || sourcePool.Length == 0)
                    continue;

                var candidates = sourcePool
                    .Where(g => g.Fitness > sourcePool.Average(gp => gp.Fitness))
                    .ToList();

                if (candidates.Count == 0)
                    continue;

                var migrant = candidates[rng.Next(candidates.Count)];

                var migrationType = (MigrationType)rng.Next(3);
                var migrationEvent = new MigrationEvent
                {
                    SourceSpeciesId = sourceId,
                    TargetSpeciesId = targetId,
                    MigratingGenomeId = migrant.Id,
                    MigrationType = migrationType,
                    CompatibilityScore = 0.5,
                    Generation = migrant.Generation
                };

                migrations.Add(migrationEvent);
                await _migrationBusWriter.WriteAsync(migrationEvent).ConfigureAwait(false);
            }

            return migrations;
        }

        /// <summary>
        /// Executes pending migrations by moving genomes between species pools.
        /// </summary>
        /// <param name="migrations">Migration events to execute.</param>
        /// <param name="allGenomes">All genomes in the population (by ID).</param>
        public void ExecuteMigrations(
            IReadOnlyList<MigrationEvent> migrations,
            Dictionary<Guid, GeoGenome> allGenomes)
        {
            lock (_poolLock)
            {
                foreach (var migration in migrations)
                {
                    if (!allGenomes.TryGetValue(migration.MigratingGenomeId, out var genome))
                        continue;

                    if (_speciesPools.TryGetValue(migration.SourceSpeciesId, out var sourcePool))
                    {
                        _speciesPools[migration.SourceSpeciesId] = sourcePool
                            .Where(g => g.Id != migration.MigratingGenomeId)
                            .ToImmutableArray();
                    }

                    if (_speciesPools.TryGetValue(migration.TargetSpeciesId, out var targetPool))
                    {
                        genome.SpeciesId = migration.TargetSpeciesId;
                        _speciesPools[migration.TargetSpeciesId] = targetPool.Add(genome);
                    }
                }
            }
        }

        /// <summary>
        /// Gets the genomes in a specific species pool.
        /// </summary>
        /// <param name="speciesId">The species ID.</param>
        /// <returns>Immutable array of genomes in the pool, or empty if species not found.</returns>
        public ImmutableArray<GeoGenome> GetSpeciesPool(int speciesId)
        {
            return _speciesPools.TryGetValue(speciesId, out var pool)
                ? pool
                : ImmutableArray<GeoGenome>.Empty;
        }

        /// <summary>
        /// Gets all registered species IDs.
        /// </summary>
        public IReadOnlyCollection<int> GetSpeciesIds()
        {
            return _speciesPools.Keys.ToImmutableArray();
        }

        /// <summary>
        /// Gets total migration events processed.
        /// </summary>
        private long _totalMigrationsProcessed;
        public long TotalMigrationsProcessed => _totalMigrationsProcessed;

        private async Task MigrationBusLoop(CancellationToken ct)
        {
            try
            {
                await foreach (var migration in _migrationBusReader.ReadAllAsync(ct).ConfigureAwait(false))
                {
                    if (_migrationChannels.TryGetValue(migration.TargetSpeciesId, out var channel))
                    {
                        await channel.Writer.WriteAsync(migration, ct).ConfigureAwait(false);
                    }
                    Interlocked.Increment(ref _totalMigrationsProcessed);
                }
            }
            catch (OperationCanceledException) { }
            catch (ChannelClosedException) { }
        }

        /// <inheritdoc/>
        public async ValueTask DisposeAsync()
        {
            if (_disposed)
                return;
            _disposed = true;

            _migrationBusWriter.TryComplete();
            _backgroundCts.Cancel();

            try
            {
                await _backgroundLoop.ConfigureAwait(false);
            }
            catch (OperationCanceledException) { }

            foreach (var channel in _migrationChannels.Values)
            {
                channel.Writer.TryComplete();
            }

            _backgroundCts.Dispose();
        }
    }

}
