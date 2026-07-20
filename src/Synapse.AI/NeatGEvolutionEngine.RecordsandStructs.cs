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
    /// Represents the complete state of a genome population at a given generation.
    /// Includes all genomes, species assignments, and statistical information.
    /// </summary>
    public record GenomePopulation
    {
        /// <summary>Immutable list of all genomes in the population.</summary>
        public ImmutableArray<GeoGenome> Genomes { get; init; } = ImmutableArray<GeoGenome>.Empty;

        /// <summary>Current generation number.</summary>
        public int GenerationNumber { get; init; }

        /// <summary>Species assignments (genome ID -> species ID).</summary>
        public ImmutableDictionary<Guid, int> SpeciesAssignments { get; init; } =
            ImmutableDictionary<Guid, int>.Empty;

        /// <summary>Population-level statistics.</summary>
        public PopulationStatistics Statistics { get; init; } = new();

        /// <summary>Number of genomes in the population.</summary>
        public int Count => Genomes.Length;

        /// <summary>Alias for GenerationNumber.</summary>
        public int Generation
        {
            get => GenerationNumber;
            init => GenerationNumber = value;
        }

        /// <summary>Number of species (initializer convenience; falls back to statistics).</summary>
        public int SpeciesCount
        {
            get => _speciesCount ?? Statistics.SpeciesCount;
            init => _speciesCount = value;
        }
        private int? _speciesCount;

        /// <summary>Best fitness convenience (falls back to best genome).</summary>
        public double BestFitness
        {
            get => _bestFitness ?? BestGenome?.Fitness ?? 0;
            init => _bestFitness = value;
        }
        private double? _bestFitness;

        /// <summary>Worst fitness convenience (falls back to worst genome).</summary>
        public double WorstFitness
        {
            get => _worstFitness ?? WorstGenome?.Fitness ?? 0;
            init => _worstFitness = value;
        }
        private double? _worstFitness;

        /// <summary>Timestamp convenience field for object initializers.</summary>
        public DateTime Timestamp { get; init; }

        /// <summary>Gets the best genome by fitness.</summary>
        public GeoGenome? BestGenome =>
            Genomes.Length > 0 ? Genomes.MaxBy(g => g.Fitness) : null;

        /// <summary>Gets the worst genome by fitness.</summary>
        public GeoGenome? WorstGenome =>
            Genomes.Length > 0 ? Genomes.MinBy(g => g.Fitness) : null;

        /// <summary>Gets the average fitness of the population.</summary>
        public double AverageFitness
        {
            get => _averageFitness ?? (Genomes.Length > 0 ? Genomes.Average(g => g.Fitness) : 0);
            init => _averageFitness = value;
        }
        private double? _averageFitness;

        /// <summary>Gets the median fitness of the population.</summary>
        public double MedianFitness
        {
            get
            {
                if (Genomes.Length == 0)
                    return 0;
                var sorted = Genomes.Select(g => g.Fitness).OrderBy(f => f).ToArray();
                int mid = sorted.Length / 2;
                return sorted.Length % 2 == 0
                    ? (sorted[mid - 1] + sorted[mid]) / 2.0
                    : sorted[mid];
            }
        }

        /// <summary>Gets the standard deviation of fitness.</summary>
        public double FitnessStandardDeviation
        {
            get
            {
                if (Genomes.Length <= 1)
                    return 0;
                double avg = AverageFitness;
                double sumSqDiff = Genomes.Sum(g => (g.Fitness - avg) * (g.Fitness - avg));
                return Math.Sqrt(sumSqDiff / (Genomes.Length - 1));
            }
        }
    }

    /// <summary>
    /// Population-level statistics computed at each generation.
    /// </summary>
    public record PopulationStatistics
    {
        /// <summary>Mean fitness of the population.</summary>
        public double MeanFitness { get; init; }

        /// <summary>Median fitness of the population.</summary>
        public double MedianFitness { get; init; }

        /// <summary>Standard deviation of fitness.</summary>
        public double StdDevFitness { get; init; }

        /// <summary>Best fitness in the population.</summary>
        public double BestFitness { get; init; }

        /// <summary>Worst fitness in the population.</summary>
        public double WorstFitness { get; init; }

        /// <summary>Number of unique topologies.</summary>
        public int UniqueTopologies { get; init; }

        /// <summary>Number of species.</summary>
        public int SpeciesCount { get; init; }

        /// <summary>Shannon diversity index of species.</summary>
        public double DiversityIndex { get; init; }

        /// <summary>Average genome complexity.</summary>
        public double AverageComplexity { get; init; }

        /// <summary>Number of evaluations performed this generation.</summary>
        public int EvaluationsThisGeneration { get; init; }

        /// <summary>Total evaluations across all generations.</summary>
        public long TotalEvaluations { get; init; }

        /// <summary>Time elapsed for this generation.</summary>
        public TimeSpan GenerationTime { get; init; }

        /// <summary>Best fitness improvement over previous generation.</summary>
        public double FitnessImprovement { get; init; }
    }

    /// <summary>
    /// Information about a single species in the population.
    /// Tracks species health, best fitness, age, and stagnation.
    /// </summary>
    public record SpeciesInfo
    {
        /// <summary>Unique species identifier.</summary>
        public int Id { get; init; }

        /// <summary>Innovation number of the representative genome.</summary>
        public Guid RepresentativeGenomeId { get; init; }

        /// <summary>Copy of the representative genome.</summary>
        public GeoGenome? Representative { get; init; }

        /// <summary>Immutable list of genome IDs that belong to this species.</summary>
        public ImmutableArray<Guid> MemberIds { get; init; } = ImmutableArray<Guid>.Empty;

        /// <summary>Number of members in this species.</summary>
        public int MemberCount => MemberIds.Length;

        /// <summary>Best fitness ever achieved by any member of this species.</summary>
        public double BestFitness { get; init; } = double.MinValue;

        /// <summary>Generation when best fitness was achieved.</summary>
        public int BestFitnessGeneration { get; init; }

        /// <summary>Current average fitness of the species.</summary>
        public double AverageFitness { get; init; }

        /// <summary>Number of generations since best fitness improved.</summary>
        public int StagnationCounter { get; init; }

        /// <summary>Alias for StagnationCounter.</summary>
        public int GenerationsWithoutImprovement
        {
            get => StagnationCounter;
            init => StagnationCounter = value;
        }

        /// <summary>Age of the species (generations since creation).</summary>
        public int Age { get; init; }

        /// <summary>Generation when this species was created.</summary>
        public int CreationGeneration { get; init; }

        /// <summary>Whether this species has been marked for extinction.</summary>
        public bool IsMarkedForExtinction { get; init; }

        /// <summary>Number of offspring allocated to this species for the next generation.</summary>
        public int OffspringAllocation { get; init; }

        /// <summary>Spawn sum (accumulated fitness for offspring allocation).</summary>
        public double SpawnSum { get; init; }

        /// <summary>Whether this species is stagnant (no improvement for MaxStagnationGenerations).</summary>
        public bool IsStagnant { get; init; }

        /// <summary>Fitness history of this species (last N generations).</summary>
        public ImmutableArray<double> FitnessHistory { get; init; } = ImmutableArray<double>.Empty;
    }

    /// <summary>
    /// Metrics about the evolution process at a specific generation.
    /// Used for tracking and diagnosing evolution progress.
    /// </summary>
    public record EvolutionMetrics
    {
        /// <summary>Current generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Best fitness in the population.</summary>
        public double BestFitness { get; init; }

        /// <summary>Fitness improvement relative to a prior generation.</summary>
        public double FitnessImprovement { get; init; }

        /// <summary>Average fitness in the population.</summary>
        public double AverageFitness { get; init; }

        /// <summary>Median fitness in the population.</summary>
        public double MedianFitness { get; init; }

        /// <summary>Standard deviation of fitness.</summary>
        public double StdDevFitness { get; init; }

        /// <summary>Number of species.</summary>
        public int SpeciesCount { get; init; }

        /// <summary>Diversity metric (0 = no diversity, 1 = maximum diversity).</summary>
        public double DiversityMetric { get; init; }

        /// <summary>Total fitness evaluations performed.</summary>
        public long TotalEvaluations { get; init; }

        /// <summary>Number of evaluations in this generation.</summary>
        public int EvaluationsThisGeneration { get; init; }

        /// <summary>Time elapsed for this generation.</summary>
        public TimeSpan GenerationTime { get; init; }

        /// <summary>Total elapsed time since evolution started.</summary>
        public TimeSpan TotalElapsed { get; init; }

        /// <summary>Mutation success rate in this generation.</summary>
        public double MutationSuccessRate { get; init; }

        /// <summary>Crossover success rate in this generation.</summary>
        public double CrossoverSuccessRate { get; init; }

        /// <summary>Number of migration events in this generation.</summary>
        public int MigrationEvents { get; init; }

        /// <summary>Number of species created this generation.</summary>
        public int SpeciesCreated { get; init; }

        /// <summary>Number of species eliminated this generation.</summary>
        public int SpeciesEliminated { get; init; }

        /// <summary>Average genome complexity.</summary>
        public double AverageComplexity { get; init; }

        /// <summary>Maximum genome complexity.</summary>
        public double MaxComplexity { get; init; }

        /// <summary>Current adaptive mutation rate multiplier.</summary>
        public double AdaptiveMutationRate { get; init; }

        /// <summary>Current speciation threshold.</summary>
        public double CurrentSpeciationThreshold { get; init; }

        /// <summary>Current perturbation magnitude.</summary>
        public double CurrentPerturbationMagnitude { get; init; }
    }

    /// <summary>
    /// Represents a single evolution event for history tracking.
    /// </summary>
    public record EvolutionEvent
    {
        /// <summary>Generation when this event occurred.</summary>
        public int Generation { get; init; }

        /// <summary>Timestamp of the event.</summary>
        public DateTime Timestamp { get; init; } = DateTime.UtcNow;

        /// <summary>Type of event.</summary>
        public EvolutionEventType EventType { get; init; }

        /// <summary>Description of the event.</summary>
        public string Description { get; init; } = string.Empty;

        /// <summary>Associated genome ID (if applicable).</summary>
        public Guid? GenomeId { get; init; }

        /// <summary>Associated species ID (if applicable).</summary>
        public int? SpeciesId { get; init; }

        /// <summary>Numeric value associated with the event.</summary>
        public double Value { get; init; }
    }

    /// <summary>
    /// Types of evolution events that can be tracked.
    /// </summary>
    public enum EvolutionEventType
    {
        /// <summary>Population was initialized.</summary>
        PopulationInitialized,

        /// <summary>Fitness evaluation completed.</summary>
        FitnessEvaluated,

        /// <summary>Speciation performed.</summary>
        SpeciationPerformed,

        /// <summary>Crossover occurred.</summary>
        CrossoverPerformed,

        /// <summary>Mutation occurred.</summary>
        MutationPerformed,

        /// <summary>Migration occurred.</summary>
        MigrationOccurred,

        /// <summary>Species created.</summary>
        SpeciesCreated,

        /// <summary>Species eliminated.</summary>
        SpeciesEliminated,

        /// <summary>New best fitness achieved.</summary>
        NewBestFitness,

        /// <summary>Stagnation detected.</summary>
        StagnationDetected,

        /// <summary>Generation completed.</summary>
        GenerationCompleted,

        /// <summary>Evolution started.</summary>
        EvolutionStarted,

        /// <summary>Evolution completed.</summary>
        EvolutionCompleted,

        /// <summary>Adaptive mutation rate adjusted.</summary>
        MutationRateAdjusted,

        /// <summary>Speciation threshold adjusted.</summary>
        ThresholdAdjusted
    }

    /// <summary>
    /// Result of a crossover operation.
    /// </summary>
    public record CrossoverResult
    {
        /// <summary>The offspring genome produced by crossover.</summary>
        public GeoGenome Offspring { get; init; } = null!;

        /// <summary>Whether the crossover was successful.</summary>
        public bool IsSuccess { get; init; }

        /// <summary>The crossover strategy used.</summary>
        public string StrategyUsed { get; init; } = string.Empty;

        /// <summary>Number of matching genes between parents.</summary>
        public int MatchingGenes { get; init; }

        /// <summary>Number of disjoint genes.</summary>
        public int DisjointGenes { get; init; }

        /// <summary>Average weight difference of matching genes.</summary>
        public double AverageWeightDifference { get; init; }
    }

    /// <summary>
    /// Result of a mutation operation.
    /// </summary>
    public record MutationResult
    {
        /// <summary>The mutated genome.</summary>
        public GeoGenome MutatedGenome { get; init; } = null!;

        /// <summary>Whether the mutation was successful.</summary>
        public bool IsSuccess { get; init; }

        /// <summary>The type of mutation applied.</summary>
        public MutationType TypeApplied { get; init; }

        /// <summary>Number of structural changes made.</summary>
        public int StructuralChanges { get; init; }

        /// <summary>Description of the mutation.</summary>
        public string Description { get; init; } = string.Empty;
    }

    /// <summary>
    /// Result of a selection operation.
    /// </summary>
    public record SelectionResult
    {
        /// <summary>Selected parent genomes.</summary>
        public ImmutableArray<GeoGenome> SelectedParents { get; init; } = ImmutableArray<GeoGenome>.Empty;

        /// <summary>Selection probabilities for each parent.</summary>
        public ImmutableArray<double> SelectionProbabilities { get; init; } = ImmutableArray<double>.Empty;

        /// <summary>The selection method used.</summary>
        public SelectionMethod Method { get; init; }
    }

    /// <summary>
    /// Represents a migration event between species.
    /// </summary>
    public record MigrationEvent
    {
        /// <summary>Source species ID.</summary>
        public int SourceSpeciesId { get; init; }

        /// <summary>Target species ID.</summary>
        public int TargetSpeciesId { get; init; }

        /// <summary>ID of the migrating genome.</summary>
        public Guid MigratingGenomeId { get; init; }

        /// <summary>Type of migration.</summary>
        public MigrationType MigrationType { get; init; }

        /// <summary>Semantic compatibility score (0-1).</summary>
        public double CompatibilityScore { get; init; }

        /// <summary>Generation when migration occurred.</summary>
        public int Generation { get; init; }
    }

    /// <summary>
    /// Internal helper struct for fast neuron lookup operations.
    /// </summary>
    internal readonly struct NeuronKey : IEquatable<NeuronKey>
    {
        public readonly long InnovationNumber;
        public readonly int LayerIndex;

        public NeuronKey(long innovationNumber, int layerIndex)
        {
            InnovationNumber = innovationNumber;
            LayerIndex = layerIndex;
        }

        public bool Equals(NeuronKey other) =>
            InnovationNumber == other.InnovationNumber && LayerIndex == other.LayerIndex;

        public override bool Equals(object? obj) => obj is NeuronKey k && Equals(k);

        public override int GetHashCode() => HashCode.Combine(InnovationNumber, LayerIndex);

        public static bool operator ==(NeuronKey left, NeuronKey right) => left.Equals(right);
        public static bool operator !=(NeuronKey left, NeuronKey right) => !left.Equals(right);
    }

    /// <summary>
    /// Internal helper struct for synapse lookup operations.
    /// </summary>
    internal readonly struct SynapseKey : IEquatable<SynapseKey>
    {
        public readonly long SourceId;
        public readonly long TargetId;

        public SynapseKey(long sourceId, long targetId)
        {
            SourceId = sourceId;
            TargetId = targetId;
        }

        public bool Equals(SynapseKey other) =>
            SourceId == other.SourceId && TargetId == other.TargetId;

        public override bool Equals(object? obj) => obj is SynapseKey k && Equals(k);

        public override int GetHashCode() => HashCode.Combine(SourceId, TargetId);

        public static bool operator ==(SynapseKey left, SynapseKey right) => left.Equals(right);
        public static bool operator !=(SynapseKey left, SynapseKey right) => !left.Equals(right);
    }

    /// <summary>
    /// Internal struct for distance computation caching.
    /// </summary>
    internal readonly struct DistanceCacheEntry : IEquatable<DistanceCacheEntry>
    {
        public readonly long GenomeAHash;
        public readonly long GenomeBHash;
        public readonly double Distance;

        public DistanceCacheEntry(long hashA, long hashB, double distance)
        {
            GenomeAHash = hashA;
            GenomeBHash = hashB;
            Distance = distance;
        }

        public bool Equals(DistanceCacheEntry other) =>
            GenomeAHash == other.GenomeAHash && GenomeBHash == other.GenomeBHash;

        public override bool Equals(object? obj) => obj is DistanceCacheEntry e && Equals(e);

        public override int GetHashCode() => HashCode.Combine(GenomeAHash, GenomeBHash);
    }

    /// <summary>
    /// Result of the evolution process.
    /// </summary>
    public record EvolutionResult
    {
        /// <summary>Final population state.</summary>
        public GenomePopulation FinalPopulation { get; init; } = null!;

        /// <summary>Best genome found during evolution.</summary>
        public GeoGenome BestGenome { get; init; } = null!;

        /// <summary>All metrics collected during evolution.</summary>
        public ImmutableArray<EvolutionMetrics> MetricsHistory { get; init; } = ImmutableArray<EvolutionMetrics>.Empty;

        /// <summary>Total generations executed.</summary>
        public int TotalGenerations { get; init; }

        /// <summary>Total fitness evaluations.</summary>
        public long TotalEvaluations { get; init; }

        /// <summary>Total elapsed time.</summary>
        public TimeSpan TotalElapsed { get; init; }

        /// <summary>Whether the target fitness was reached.</summary>
        public bool TargetReached { get; init; }

        /// <summary>Reason evolution stopped.</summary>
        public string StopReason { get; init; } = string.Empty;

        /// <summary>Evolution events log.</summary>
        public IReadOnlyList<EvolutionEvent> Events { get; init; } = Array.Empty<EvolutionEvent>();
    }

}
