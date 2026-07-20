// =============================================================================
// NeatGEvolutionEngine.GenomePopulationSnapshot.cs — NEAT-G partial module
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
    /// Provides point-in-time snapshots of the population state for debugging,
    /// replay, and analysis. Supports diffing between snapshots.
    /// </summary>
    public sealed class PopulationSnapshotManager
    {
        private readonly Queue<PopulationSnapshot> _snapshots;
        private readonly int _maxSnapshots;

        /// <summary>
        /// Initializes a new instance of the PopulationSnapshotManager class.
        /// </summary>
        /// <param name="maxSnapshots">Maximum number of snapshots to retain.</param>
        public PopulationSnapshotManager(int maxSnapshots = 50)
        {
            _maxSnapshots = maxSnapshots;
            _snapshots = new Queue<PopulationSnapshot>();
        }

        /// <summary>Number of stored snapshots.</summary>
        public int Count => _snapshots.Count;

        /// <summary>
        /// Captures a snapshot of the current population state.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <param name="species">Current species.</param>
        /// <param name="metrics">Current metrics.</param>
        /// <returns>The captured snapshot.</returns>
        public PopulationSnapshot CaptureSnapshot(
            GenomePopulation population,
            ImmutableArray<SpeciesInfo> species,
            EvolutionMetrics metrics)
        {
            var snapshot = new PopulationSnapshot
            {
                Timestamp = DateTime.UtcNow,
                Generation = metrics.Generation,
                Genomes = population.Genomes.Select(g => new GenomeSnapshot
                {
                    Id = g.Id,
                    Fitness = g.Fitness,
                    NeuronCount = g.ActiveNeuronCount,
                    SynapseCount = g.ActiveSynapseCount,
                    MaxLayer = g.MaxLayerDepth,
                    Complexity = g.Complexity,
                    SpeciesId = g.SpeciesId,
                    TopologyHash = g.ComputeTopologyHash()
                }).ToList().AsReadOnly(),
                SpeciesCount = species.Length,
                BestFitness = metrics.BestFitness,
                AverageFitness = metrics.AverageFitness,
                Diversity = metrics.DiversityMetric,
                SpeciesInfo = species.Select(s => new SpeciesSnapshotInfo
                {
                    Id = s.Id,
                    MemberCount = s.MemberCount,
                    BestFitness = s.BestFitness,
                    AverageFitness = s.AverageFitness,
                    StagnationCounter = s.StagnationCounter
                }).ToList().AsReadOnly()
            };

            _snapshots.Enqueue(snapshot);
            while (_snapshots.Count > _maxSnapshots)
            {
                _snapshots.Dequeue();
            }

            return snapshot;
        }

        /// <summary>
        /// Gets a specific snapshot by generation number.
        /// </summary>
        /// <param name="generation">Generation to retrieve.</param>
        /// <returns>The snapshot, or null if not found.</returns>
        public PopulationSnapshot? GetSnapshot(int generation)
        {
            return _snapshots.FirstOrDefault(s => s.Generation == generation);
        }

        /// <summary>
        /// Gets the most recent snapshot.
        /// </summary>
        public PopulationSnapshot? GetLatestSnapshot()
        {
            return _snapshots.Count > 0 ? _snapshots.Last() : null;
        }

        /// <summary>
        /// Gets all stored snapshots.
        /// </summary>
        public IReadOnlyList<PopulationSnapshot> GetAllSnapshots()
        {
            return _snapshots.ToList().AsReadOnly();
        }

        /// <summary>
        /// Computes the diff between two snapshots.
        /// </summary>
        /// <param name="before">Earlier snapshot.</param>
        /// <param name="after">Later snapshot.</param>
        /// <returns>Diff result.</returns>
        public SnapshotDiff ComputeDiff(PopulationSnapshot before, PopulationSnapshot after)
        {
            var beforeGenomeIds = before.Genomes.Select(g => g.Id).ToHashSet();
            var afterGenomeIds = after.Genomes.Select(g => g.Id).ToHashSet();

            var added = after.Genomes.Where(g => !beforeGenomeIds.Contains(g.Id)).ToList();
            var removed = before.Genomes.Where(g => !afterGenomeIds.Contains(g.Id)).ToList();
            var retained = after.Genomes.Where(g => beforeGenomeIds.Contains(g.Id)).ToList();

            var fitnessChanges = new List<(Guid Id, double Before, double After)>();
            foreach (var afterGenome in retained)
            {
                var beforeGenome = before.Genomes.FirstOrDefault(g => g.Id == afterGenome.Id);
                if (beforeGenome != null && Math.Abs(beforeGenome.Fitness - afterGenome.Fitness) > 1e-10)
                {
                    fitnessChanges.Add((afterGenome.Id, beforeGenome.Fitness, afterGenome.Fitness));
                }
            }

            var topologyChanges = new List<(Guid Id, long BeforeHash, long AfterHash)>();
            foreach (var afterGenome in retained)
            {
                var beforeGenome = before.Genomes.FirstOrDefault(g => g.Id == afterGenome.Id);
                if (beforeGenome != null && beforeGenome.TopologyHash != afterGenome.TopologyHash)
                {
                    topologyChanges.Add((afterGenome.Id, beforeGenome.TopologyHash, afterGenome.TopologyHash));
                }
            }

            return new SnapshotDiff
            {
                BeforeGeneration = before.Generation,
                AfterGeneration = after.Generation,
                AddedGenomes = added.Count,
                RemovedGenomes = removed.Count,
                RetainedGenomes = retained.Count,
                FitnessChanges = fitnessChanges.AsReadOnly(),
                TopologyChanges = topologyChanges.AsReadOnly(),
                FitnessImprovement = after.BestFitness - before.BestFitness,
                DiversityChange = after.Diversity - before.Diversity,
                SpeciesCountChange = after.SpeciesCount - before.SpeciesCount
            };
        }

        /// <summary>
        /// Clears all stored snapshots.
        /// </summary>
        public void Clear()
        {
            _snapshots.Clear();
        }
    }

    /// <summary>
    /// Point-in-time snapshot of the population.
    /// </summary>
    public sealed class PopulationSnapshot
    {
        public DateTime Timestamp { get; init; }
        public int Generation { get; init; }
        public IReadOnlyList<GenomeSnapshot> Genomes { get; init; } = Array.Empty<GenomeSnapshot>();
        public int SpeciesCount { get; init; }
        public double BestFitness { get; init; }
        public double AverageFitness { get; init; }
        public double Diversity { get; init; }
        public IReadOnlyList<SpeciesSnapshotInfo> SpeciesInfo { get; init; } = Array.Empty<SpeciesSnapshotInfo>();
    }

    /// <summary>
    /// Snapshot of a single genome.
    /// </summary>
    public sealed class GenomeSnapshot
    {
        public Guid Id { get; init; }
        public double Fitness { get; init; }
        public int NeuronCount { get; init; }
        public int SynapseCount { get; init; }
        public int MaxLayer { get; init; }
        public double Complexity { get; init; }
        public int SpeciesId { get; init; }
        public long TopologyHash { get; init; }
    }

    /// <summary>
    /// Snapshot of species info.
    /// </summary>
    public sealed class SpeciesSnapshotInfo
    {
        public int Id { get; init; }
        public int MemberCount { get; init; }
        public double BestFitness { get; init; }
        public double AverageFitness { get; init; }
        public int StagnationCounter { get; init; }
    }

    /// <summary>
    /// Diff between two population snapshots.
    /// </summary>
    public sealed class SnapshotDiff
    {
        public int BeforeGeneration { get; init; }
        public int AfterGeneration { get; init; }
        public int AddedGenomes { get; init; }
        public int RemovedGenomes { get; init; }
        public int RetainedGenomes { get; init; }
        public IReadOnlyList<(Guid Id, double Before, double After)> FitnessChanges { get; init; } = Array.Empty<(Guid, double, double)>();
        public IReadOnlyList<(Guid Id, long BeforeHash, long AfterHash)> TopologyChanges { get; init; } = Array.Empty<(Guid, long, long)>();
        public double FitnessImprovement { get; init; }
        public double DiversityChange { get; init; }
        public int SpeciesCountChange { get; init; }
    }

}
