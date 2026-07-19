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

namespace GDNN.Core.NEAT
{
    // =========================================================================
    #region Enums

    /// <summary>
    /// Specifies the type of mutation that can be applied to a genome.
    /// Each mutation type represents a distinct structural or parametric change
    /// to the neural network topology or weights.
    /// </summary>
    [Flags]
    public enum MutationType
    {
        /// <summary>No mutation applied.</summary>
        None = 0,

        /// <summary>Point mutation: modify a single weight or bias value.</summary>
        PointMutation = 1 << 0,

        /// <summary>Insertion mutation: add a new node into an existing connection.</summary>
        InsertionMutation = 1 << 1,

        /// <summary>Deletion mutation: remove a node and reconnect surrounding connections.</summary>
        DeletionMutation = 1 << 2,

        /// <summary>Duplication mutation: duplicate a node with its connection pattern.</summary>
        DuplicationMutation = 1 << 3,

        /// <summary>Inversion mutation: reverse the order of a segment of connections.</summary>
        InversionMutation = 1 << 4,

        /// <summary>Translocation mutation: move a segment of the genome to a different location.</summary>
        TranslocationMutation = 1 << 5,

        /// <summary>Semantic mutation: modify neuron activation based on semantic role.</summary>
        SemanticMutation = 1 << 6,

        /// <summary>Topology mutation: add or remove connections between existing nodes.</summary>
        TopologyMutation = 1 << 7,

        /// <summary>Weight perturbation: gaussian perturbation of weight values.</summary>
        WeightPerturbation = 1 << 8,

        /// <summary>Activation shift: change the activation function of a neuron.</summary>
        ActivationShift = 1 << 9,

        /// <summary>Bias drift: random walk perturbation of bias values.</summary>
        BiasDrift = 1 << 10,

        /// <summary>Synapse growth: create a new connection between existing neurons.</summary>
        SynapseGrowth = 1 << 11,

        /// <summary>Synapse pruning: remove low-magnitude connections.</summary>
        SynapsePruning = 1 << 12,

        /// <summary>Layer insertion: insert an entirely new hidden layer.</summary>
        LayerInsertion = 1 << 13,

        /// <summary>Layer removal: remove an entire hidden layer and reconnect.</summary>
        LayerRemoval = 1 << 14,

        /// <summary>Gene silencing: deactivate a neuron without removing it.</summary>
        GeneSilencing = 1 << 15,

        /// <summary>Gene activation: reactivate a previously silenced neuron.</summary>
        GeneActivation = 1 << 16,

        /// <summary>Regulatory mutation: modify gene regulatory interactions.</summary>
        RegulatoryMutation = 1 << 17,

        /// <summary>All mutation types combined.</summary>
        All = PointMutation | InsertionMutation | DeletionMutation | DuplicationMutation |
              InversionMutation | TranslocationMutation | SemanticMutation | TopologyMutation |
              WeightPerturbation | ActivationShift | BiasDrift | SynapseGrowth | SynapsePruning |
              LayerInsertion | LayerRemoval | GeneSilencing | GeneActivation | RegulatoryMutation
    }

    /// <summary>
    /// Specifies the selection method used during evolution.
    /// </summary>
    public enum SelectionMethod
    {
        /// <summary>Tournament selection: pick k random individuals and select the best.</summary>
        Tournament,

        /// <summary>Roulette wheel selection: probability proportional to fitness.</summary>
        RouletteWheel,

        /// <summary>Rank-based selection: probability based on fitness rank rather than magnitude.</summary>
        RankBased,

        /// <summary>Truncation selection: select top fraction of population.</summary>
        Truncation,

        /// <summary>Stochastic universal sampling: evenly spaced selection from ranked population.</summary>
        StochasticUniversal
    }

    /// <summary>
    /// Specifies the crossover strategy to use.
    /// </summary>
    public enum CrossoverStrategyType
    {
        /// <summary>Semantic crossover using neuron role alignment.</summary>
        Semantic,

        /// <summary>Topology crossover using innovation number alignment.</summary>
        Topology,

        /// <summary>Weight-only crossover without structural changes.</summary>
        Weight,

        /// <summary>Hybrid crossover combining multiple strategies.</summary>
        Hybrid
    }

    /// <summary>
    /// Specifies the speciation distance metric.
    /// </summary>
    public enum SpeciationMethod
    {
        /// <summary>No speciation (single species).</summary>
        None,

        /// <summary>Gromov-Hausdorff approximation using landmark embedding.</summary>
        GromovHausdorff,

        /// <summary>Standard NEAT compatibility distance.</summary>
        CompatibilityDistance,

        /// <summary>Manifold-aware distance using curvature information.</summary>
        ManifoldDistance,

        /// <summary>Hybrid distance combining multiple metrics.</summary>
        HybridDistance
    }

    /// <summary>
    /// Specifies the activation function type for neurons.
    /// </summary>
    public enum ActivationFunction
    {
        /// <summary>Hyperbolic tangent activation.</summary>
        Tanh,

        /// <summary>Sigmoid activation.</summary>
        Sigmoid,

        /// <summary>Rectified Linear Unit.</summary>
        ReLU,

        /// <summary>Leaky ReLU with alpha=0.01.</summary>
        LeakyReLU,

        /// <summary>Gaussian Error Linear Unit.</summary>
        GELU,

        /// <summary>Swish activation (x * sigmoid(x)).</summary>
        Swish,

        /// <summary>Sinusoidal activation.</summary>
        Sinusoidal,

        /// <summary>Linear (identity) activation.</summary>
        Linear,

        /// <summary>Abs activation.</summary>
        Abs,

        /// <summary>Step function (binary threshold).</summary>
        Step,

        /// <summary>Softplus activation.</summary>
        Softplus,

        /// <summary>Mish activation.</summary>
        Mish,

        /// <summary>Exponential activation.</summary>
        Exponential
    }

    /// <summary>
    /// Represents the current state of the evolution process.
    /// </summary>
    public enum EvolutionState
    {
        /// <summary>Evolution has not started.</summary>
        NotStarted,

        /// <summary>Evolution is initializing.</summary>
        Initializing,

        /// <summary>Population is being evaluated.</summary>
        Evaluating,

        /// <summary>Speciation is being performed.</summary>
        Speciating,

        /// <summary>Selection is being performed.</summary>
        Selecting,

        /// <summary>Crossover and mutation are being applied.</summary>
        Evolving,

        /// <summary>Migration between species is occurring.</summary>
        Migrating,

        /// <summary>Evolution is complete.</summary>
        Complete,

        /// <summary>Evolution was cancelled.</summary>
        Cancelled,

        /// <summary>Evolution encountered an error.</summary>
        Error
    }

    /// <summary>
    /// Specifies the objective of fitness evaluation.
    /// </summary>
    public enum FitnessObjective
    {
        /// <summary>Maximize the fitness value.</summary>
        Maximize,

        /// <summary>Minimize the fitness value.</summary>
        Minimize
    }

    /// <summary>
    /// Specifies the type of fitness component being measured.
    /// </summary>
    public enum FitnessComponent
    {
        /// <summary>Visual fidelity compared to reference.</summary>
        VisualFidelity,

        /// <summary>Inference performance (latency).</summary>
        Performance,

        /// <summary>Memory usage efficiency.</summary>
        MemoryEfficiency,

        /// <summary>Structural complexity penalty.</summary>
        StructuralComplexity,

        /// <summary>Perceptual quality metric.</summary>
        PerceptualQuality,

        /// <summary>SDF error metric for geometric accuracy.</summary>
        SDFError,

        /// <summary>
        /// Irradiance prediction error for L-DNN proxy heads evolved via NEAT-G.
        /// </summary>
        IrradianceError,

        /// <summary>Topological novelty.</summary>
        TopologicalNovelty,

        /// <summary>Generalization capability.</summary>
        Generalization
    }

    /// <summary>
    /// Specifies the type of migration event.
    /// </summary>
    public enum MigrationType
    {
        /// <summary>Random migration to a random target species.</summary>
        Random,

        /// <summary>Semantic migration to the most semantically similar species.</summary>
        Semantic,

        /// <summary>Fitness-based migration to higher-fitness species.</summary>
        FitnessBased,

        /// <summary>Diversity-driven migration to increase species diversity.</summary>
        DiversityDriven
    }

    /// <summary>
    /// Specifies the stagnation detection strategy.
    /// </summary>
    public enum StagnationStrategy
    {
        /// <summary>Detect stagnation based on no improvement in best fitness.</summary>
        BestFitness,

        /// <summary>Detect stagnation based on no improvement in average fitness.</summary>
        AverageFitness,

        /// <summary>Detect stagnation based on population diversity.</summary>
        DiversityBased,

        /// <summary>Combined stagnation detection using multiple signals.</summary>
        Combined
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Configuration Classes

    /// <summary>
    /// Configuration parameters for the NEAT-G evolution engine.
    /// Contains all tunable parameters that control the behavior of the
    /// evolutionary optimization process.
    /// </summary>
    public sealed class EvolutionConfig : ICloneable
    {
        /// <summary>Total population size across all species.</summary>
        public int PopulationSize { get; set; } = 300;

        /// <summary>Number of elite individuals preserved unchanged per species.</summary>
        public int EliteCount { get; set; } = 5;

        /// <summary>Alias for EliteCount used by some callers.</summary>
        public int ElitismCount { get => EliteCount; set => EliteCount = value; }

        /// <summary>Fraction of population designated as elite (alternative to EliteCount).</summary>
        public double EliteFraction { get; set; } = 0.05;

        /// <summary>Maximum number of generations before evolution stops.</summary>
        public int MaxGenerations { get; set; } = 1000;

        /// <summary>Target fitness value; evolution stops when reached.</summary>
        public double TargetFitness { get; set; } = double.MaxValue;

        /// <summary>Minimum fitness improvement to avoid stagnation counting.</summary>
        public double FitnessThreshold { get; set; } = 1e-6;

        /// <summary>Overall probability of crossover when producing offspring.</summary>
        public double CrossoverRate { get; set; } = 0.75;

        /// <summary>Probability of mutation per offspring after crossover.</summary>
        public double MutationRate { get; set; } = 0.25;

        /// <summary>Probability of asexual reproduction (mutation only, no crossover).</summary>
        public double AsexualRate { get; set; } = 0.1;

        /// <summary>Selection method for choosing parents.</summary>
        public SelectionMethod ParentSelection { get; set; } = SelectionMethod.Tournament;

        /// <summary>Tournament size for tournament selection.</summary>
        public int TournamentSize { get; set; } = 5;

        /// <summary>Selection method for survival selection.</summary>
        public SelectionMethod SurvivalSelection { get; set; } = SelectionMethod.Truncation;

        /// <summary>Crossover strategy type.</summary>
        public CrossoverStrategyType CrossoverStrategy { get; set; } = CrossoverStrategyType.Semantic;

        /// <summary>Speciation method used for grouping genomes.</summary>
        public SpeciationMethod SpeciationMethod { get; set; } = SpeciationMethod.GromovHausdorff;

        /// <summary>Initial speciation threshold; species are formed when distance below this.</summary>
        public double SpeciationThreshold { get; set; } = 3.0;

        /// <summary>Alias for SpeciationThreshold used by some callers.</summary>
        public double SpeciesCompatibilityThreshold
        {
            get => SpeciationThreshold;
            set => SpeciationThreshold = value;
        }

        /// <summary>Minimum speciation threshold to prevent species explosion.</summary>
        public double MinSpeciationThreshold { get; set; } = 0.5;

        /// <summary>Maximum speciation threshold to prevent species collapse.</summary>
        public double MaxSpeciationThreshold { get; set; } = 8.0;

        /// <summary>Target number of species; threshold adjusts to meet this target.</summary>
        public int TargetSpeciesCount { get; set; } = 10;

        /// <summary>Rate at which speciation threshold adjusts toward target species count.</summary>
        public double ThresholdAdjustmentRate { get; set; } = 0.1;

        /// <summary>Compatibility coefficient for excess genes.</summary>
        public double CompatibilityDisjointCoefficient { get; set; } = 1.0;

        /// <summary>Compatibility coefficient for weight differences.</summary>
        public double CompatibilityWeightCoefficient { get; set; } = 0.5;

        /// <summary>Maximum number of generations a species can exist without improvement.</summary>
        public int MaxStagnationGenerations { get; set; } = 20;

        /// <summary>Number of generations before species fitness is considered stagnant.</summary>
        public int StagnationPeriod { get; set; } = 15;

        /// <summary>Minimum number of individuals in a species before it can be removed.</summary>
        public int SpeciesMinimumSize { get; set; } = 2;

        /// <summary>Maximum number of species allowed.</summary>
        public int MaxSpeciesCount { get; set; } = 50;

        /// <summary>Whether to use fitness sharing within species.</summary>
        public bool UseFitnessSharing { get; set; } = true;

        /// <summary>Exponent for fitness sharing (controls sharing pressure).</summary>
        public double SharingExponent { get; set; } = 1.0;

        /// <summary>Species fitness adjustment method.</summary>
        public SpeciesFitnessAdjustment FitnessAdjustment { get; set; } = SpeciesFitnessAdjustment.Sharing;

        /// <summary>Stagnation detection strategy.</summary>
        public StagnationStrategy StagnationDetection { get; set; } = StagnationStrategy.Combined;

        /// <summary>Maximum number of parallel evaluation tasks.</summary>
        public int MaxParallelEvaluations { get; set; } = Environment.ProcessorCount;

        /// <summary>Timeout for individual fitness evaluations in milliseconds.</summary>
        public int EvaluationTimeoutMs { get; set; } = 30000;

        /// <summary>Maximum number of migration events per generation.</summary>
        public int MaxMigrationsPerGeneration { get; set; } = 5;

        /// <summary>Migration interval in generations.</summary>
        public int MigrationInterval { get; set; } = 10;

        /// <summary>Migration rate (fraction of species to migrate).</summary>
        public double MigrationRate { get; set; } = 0.05;

        /// <summary>Whether to enable migration between species.</summary>
        public bool EnableMigration { get; set; } = true;

        /// <summary>Whether to track evolution history for diagnostics.</summary>
        public bool EnableHistoryTracking { get; set; } = true;

        /// <summary>Maximum number of history entries to retain.</summary>
        public int MaxHistoryEntries { get; set; } = 10000;

        /// <summary>Number of landmark points for Gromov-Hausdorff approximation.</summary>
        public int LandmarkCount { get; set; } = 20;

        /// <summary>Curvature approximation resolution for manifold distance.</summary>
        public int CurvatureResolution { get; set; } = 10;

        /// <summary>Number of dimensions for semantic embedding.</summary>
        public int SemanticEmbeddingDimension { get; set; } = 32;

        /// <summary>Semantic similarity threshold for crossover alignment.</summary>
        public double SemanticAlignmentThreshold { get; set; } = 0.3;

        /// <summary>Weight initialization range [-range, +range].</summary>
        public double WeightInitRange { get; set; } = 1.0;

        /// <summary>Bias initialization range [-range, +range].</summary>
        public double BiasInitRange { get; set; } = 0.5;

        /// <summary>Default learning rate for weight perturbation magnitude.</summary>
        public double PerturbationMagnitude { get; set; } = 0.1;

        /// <summary>Rate of perturbation magnitude decay over generations.</summary>
        public double PerturbationDecayRate { get; set; } = 0.995;

        /// <summary>Minimum perturbation magnitude.</summary>
        public double MinPerturbationMagnitude { get; set; } = 0.001;

        /// <summary>Probability of uniform vs. point perturbation in weight mutation.</summary>
        public double UniformPerturbationProbability { get; set; } = 0.1;

        /// <summary>Whether to use adaptive mutation rates.</summary>
        public bool UseAdaptiveMutation { get; set; } = true;

        /// <summary>Window size for adaptive mutation rate calculation.</summary>
        public int AdaptiveWindow { get; set; } = 10;

        /// <summary>Maximum mutation rate for adaptive scheduling.</summary>
        public double MaxAdaptiveMutationRate { get; set; } = 0.5;

        /// <summary>Minimum mutation rate for adaptive scheduling.</summary>
        public double MinAdaptiveMutationRate { get; set; } = 0.01;

        /// <summary>Number of fitness evaluations for warm-up.</summary>
        public int WarmUpEvaluations { get; set; } = 5;

        /// <summary>Whether to use incremental evaluation (reuse previous fitness).</summary>
        public bool UseIncrementalEvaluation { get; set; } = true;

        /// <summary>Minimum number of generations before incremental evaluation is used.</summary>
        public int IncrementalEvaluationStart { get; set; } = 5;

        /// <summary>Random seed for reproducibility. Null for random seed.</summary>
        public int? RandomSeed { get; set; }

        /// <summary>Include IrradianceError in multi-objective fitness when enabled.</summary>
        public bool EnableIrradianceFitness { get; set; }

        /// <summary>Weight for IrradianceError when EnableIrradianceFitness is true.</summary>
        public double IrradianceFitnessWeight { get; set; } = 0.1;

        /// <summary>Objective for fitness optimization.</summary>
        public FitnessObjective Objective { get; set; } = FitnessObjective.Maximize;

        /// <summary>Creates a deep clone of this configuration.</summary>
        /// <returns>A new EvolutionConfig with identical parameter values.</returns>
        public EvolutionConfig Clone()
        {
            return (EvolutionConfig)MemberwiseClone();
        }

        object ICloneable.Clone() => Clone();

        /// <summary>
        /// Creates a default configuration tuned for general-purpose evolution.
        /// </summary>
        /// <returns>A configuration with sensible default values.</returns>
        public static EvolutionConfig CreateDefault()
        {
            return new EvolutionConfig
            {
                PopulationSize = 300,
                EliteCount = 5,
                EliteFraction = 0.05,
                MaxGenerations = 1000,
                TargetFitness = double.MaxValue,
                FitnessThreshold = 1e-6,
                CrossoverRate = 0.75,
                MutationRate = 0.25,
                AsexualRate = 0.1,
                ParentSelection = SelectionMethod.Tournament,
                TournamentSize = 5,
                SurvivalSelection = SelectionMethod.Truncation,
                CrossoverStrategy = CrossoverStrategyType.Semantic,
                SpeciationMethod = SpeciationMethod.GromovHausdorff,
                SpeciationThreshold = 3.0,
                MinSpeciationThreshold = 0.5,
                MaxSpeciationThreshold = 8.0,
                TargetSpeciesCount = 10,
                ThresholdAdjustmentRate = 0.1,
                CompatibilityDisjointCoefficient = 1.0,
                CompatibilityWeightCoefficient = 0.5,
                MaxStagnationGenerations = 20,
                StagnationPeriod = 15,
                SpeciesMinimumSize = 2,
                MaxSpeciesCount = 50,
                UseFitnessSharing = true,
                SharingExponent = 1.0,
                FitnessAdjustment = SpeciesFitnessAdjustment.Sharing,
                StagnationDetection = StagnationStrategy.Combined,
                MaxParallelEvaluations = Environment.ProcessorCount,
                EvaluationTimeoutMs = 30000,
                MaxMigrationsPerGeneration = 5,
                MigrationInterval = 10,
                MigrationRate = 0.05,
                EnableMigration = true,
                EnableHistoryTracking = true,
                MaxHistoryEntries = 10000,
                LandmarkCount = 20,
                CurvatureResolution = 10,
                SemanticEmbeddingDimension = 32,
                SemanticAlignmentThreshold = 0.3,
                WeightInitRange = 1.0,
                BiasInitRange = 0.5,
                PerturbationMagnitude = 0.1,
                PerturbationDecayRate = 0.995,
                MinPerturbationMagnitude = 0.001,
                UniformPerturbationProbability = 0.1,
                UseAdaptiveMutation = true,
                AdaptiveWindow = 10,
                MaxAdaptiveMutationRate = 0.5,
                MinAdaptiveMutationRate = 0.01,
                WarmUpEvaluations = 5,
                UseIncrementalEvaluation = true,
                IncrementalEvaluationStart = 5,
                RandomSeed = null,
                Objective = FitnessObjective.Maximize
            };
        }

        /// <summary>
        /// Creates a configuration optimized for complex topologies.
        /// </summary>
        /// <returns>A configuration with parameters favoring structural exploration.</returns>
        public static EvolutionConfig CreateComplexTopology()
        {
            var config = CreateDefault();
            config.PopulationSize = 500;
            config.CrossoverRate = 0.6;
            config.MutationRate = 0.4;
            config.SpeciationThreshold = 4.0;
            config.TargetSpeciesCount = 15;
            config.MaxStagnationGenerations = 30;
            config.PerturbationMagnitude = 0.15;
            config.EnableMigration = true;
            config.MigrationRate = 0.08;
            return config;
        }

        /// <summary>
        /// Creates a configuration optimized for fine-tuning existing solutions.
        /// </summary>
        /// <returns>A configuration with parameters favoring exploitation.</returns>
        public static EvolutionConfig CreateFineTuning()
        {
            var config = CreateDefault();
            config.PopulationSize = 200;
            config.CrossoverRate = 0.85;
            config.MutationRate = 0.15;
            config.SpeciationThreshold = 2.0;
            config.TargetSpeciesCount = 5;
            config.PerturbationMagnitude = 0.05;
            config.PerturbationDecayRate = 0.99;
            config.EnableMigration = false;
            return config;
        }

        /// <summary>
        /// Creates a configuration for quick prototyping with small populations.
        /// </summary>
        /// <returns>A configuration with small population and fast convergence.</returns>
        public static EvolutionConfig CreateQuickPrototype()
        {
            var config = CreateDefault();
            config.PopulationSize = 50;
            config.MaxGenerations = 100;
            config.EliteCount = 3;
            config.TournamentSize = 3;
            config.TargetSpeciesCount = 3;
            config.MaxParallelEvaluations = 4;
            config.EnableHistoryTracking = false;
            return config;
        }

        /// <summary>
        /// Validates the configuration parameters and returns any warnings.
        /// </summary>
        /// <returns>A list of validation warnings. Empty if configuration is valid.</returns>
        public IReadOnlyList<string> Validate()
        {
            var warnings = new List<string>();

            if (PopulationSize < 10)
                warnings.Add("PopulationSize < 10 may lead to insufficient genetic diversity.");
            if (PopulationSize > 10000)
                warnings.Add("PopulationSize > 10000 may cause excessive computational overhead.");
            if (EliteFraction < 0 || EliteFraction > 0.5)
                warnings.Add("EliteFraction should be between 0 and 0.5.");
            if (CrossoverRate < 0 || CrossoverRate > 1)
                warnings.Add("CrossoverRate should be between 0 and 1.");
            if (MutationRate < 0 || MutationRate > 1)
                warnings.Add("MutationRate should be between 0 and 1.");
            if (CrossoverRate + MutationRate + AsexualRate > 1.01)
                warnings.Add("CrossoverRate + MutationRate + AsexualRate should not exceed 1.0.");
            if (SpeciationThreshold <= 0)
                warnings.Add("SpeciationThreshold must be positive.");
            if (MaxStagnationGenerations < 1)
                warnings.Add("MaxStagnationGenerations must be at least 1.");
            if (SpeciesMinimumSize < 1)
                warnings.Add("SpeciesMinimumSize must be at least 1.");
            if (PerturbationMagnitude <= 0)
                warnings.Add("PerturbationMagnitude must be positive.");
            if (LandmarkCount < 3)
                warnings.Add("LandmarkCount should be at least 3 for meaningful GH distance approximation.");
            if (SemanticEmbeddingDimension < 4)
                warnings.Add("SemanticEmbeddingDimension should be at least 4 for meaningful embeddings.");

            return warnings;
        }
    }

    /// <summary>
    /// Specifies how species fitness is adjusted for selection pressure.
    /// </summary>
    public enum SpeciesFitnessAdjustment
    {
        /// <summary>No fitness adjustment.</summary>
        None,

        /// <summary>Fitness sharing within species.</summary>
        Sharing,

        /// <summary>Rank-based fitness scaling within species.</summary>
        RankBased,

        /// <summary>Boltzmann softmax selection.</summary>
        Boltzmann
    }

    /// <summary>
    /// Per-type mutation rate configuration.
    /// Contains individual mutation probabilities for each mutation type,
    /// allowing fine-grained control over the evolutionary search.
    /// </summary>
    public sealed class MutationRate : ICloneable
    {
        /// <summary>Probability of point mutation.</summary>
        public double PointMutation { get; set; } = 0.1;

        /// <summary>Probability of insertion mutation.</summary>
        public double InsertionMutation { get; set; } = 0.02;

        /// <summary>Probability of deletion mutation.</summary>
        public double DeletionMutation { get; set; } = 0.02;

        /// <summary>Probability of duplication mutation.</summary>
        public double DuplicationMutation { get; set; } = 0.01;

        /// <summary>Probability of inversion mutation.</summary>
        public double InversionMutation { get; set; } = 0.01;

        /// <summary>Probability of translocation mutation.</summary>
        public double TranslocationMutation { get; set; } = 0.01;

        /// <summary>Probability of semantic mutation.</summary>
        public double SemanticMutation { get; set; } = 0.05;

        /// <summary>Probability of topology mutation.</summary>
        public double TopologyMutation { get; set; } = 0.08;

        /// <summary>Probability of weight perturbation.</summary>
        public double WeightPerturbation { get; set; } = 0.3;

        /// <summary>Probability of activation function shift.</summary>
        public double ActivationShift { get; set; } = 0.02;

        /// <summary>Probability of bias drift.</summary>
        public double BiasDrift { get; set; } = 0.1;

        /// <summary>Probability of synapse growth (new connection).</summary>
        public double SynapseGrowth { get; set; } = 0.05;

        /// <summary>Probability of synapse pruning.</summary>
        public double SynapsePruning { get; set; } = 0.03;

        /// <summary>Probability of layer insertion.</summary>
        public double LayerInsertion { get; set; } = 0.01;

        /// <summary>Probability of layer removal.</summary>
        public double LayerRemoval { get; set; } = 0.005;

        /// <summary>Probability of gene silencing.</summary>
        public double GeneSilencing { get; set; } = 0.02;

        /// <summary>Probability of gene activation.</summary>
        public double GeneActivation { get; set; } = 0.02;

        /// <summary>Probability of regulatory mutation.</summary>
        public double RegulatoryMutation { get; set; } = 0.01;

        /// <summary>Global mutation rate multiplier applied to all rates.</summary>
        public double GlobalMultiplier { get; set; } = 1.0;

        /// <summary>
        /// Gets the effective rate for a specific mutation type, after applying the global multiplier.
        /// </summary>
        /// <param name="type">The mutation type to get the rate for.</param>
        /// <returns>The effective mutation rate.</returns>
        public double GetRate(MutationType type)
        {
            double raw = type switch
            {
                MutationType.PointMutation => PointMutation,
                MutationType.InsertionMutation => InsertionMutation,
                MutationType.DeletionMutation => DeletionMutation,
                MutationType.DuplicationMutation => DuplicationMutation,
                MutationType.InversionMutation => InversionMutation,
                MutationType.TranslocationMutation => TranslocationMutation,
                MutationType.SemanticMutation => SemanticMutation,
                MutationType.TopologyMutation => TopologyMutation,
                MutationType.WeightPerturbation => WeightPerturbation,
                MutationType.ActivationShift => ActivationShift,
                MutationType.BiasDrift => BiasDrift,
                MutationType.SynapseGrowth => SynapseGrowth,
                MutationType.SynapsePruning => SynapsePruning,
                MutationType.LayerInsertion => LayerInsertion,
                MutationType.LayerRemoval => LayerRemoval,
                MutationType.GeneSilencing => GeneSilencing,
                MutationType.GeneActivation => GeneActivation,
                MutationType.RegulatoryMutation => RegulatoryMutation,
                _ => 0.0
            };
            return Math.Clamp(raw * GlobalMultiplier, 0.0, 1.0);
        }

        /// <summary>
        /// Sets the rate for a specific mutation type.
        /// </summary>
        /// <param name="type">The mutation type to set.</param>
        /// <param name="rate">The rate value (will be clamped to [0,1]).</param>
        public void SetRate(MutationType type, double rate)
        {
            var clamped = Math.Clamp(rate, 0.0, 1.0);
            switch (type)
            {
                case MutationType.PointMutation:
                    PointMutation = clamped;
                    break;
                case MutationType.InsertionMutation:
                    InsertionMutation = clamped;
                    break;
                case MutationType.DeletionMutation:
                    DeletionMutation = clamped;
                    break;
                case MutationType.DuplicationMutation:
                    DuplicationMutation = clamped;
                    break;
                case MutationType.InversionMutation:
                    InversionMutation = clamped;
                    break;
                case MutationType.TranslocationMutation:
                    TranslocationMutation = clamped;
                    break;
                case MutationType.SemanticMutation:
                    SemanticMutation = clamped;
                    break;
                case MutationType.TopologyMutation:
                    TopologyMutation = clamped;
                    break;
                case MutationType.WeightPerturbation:
                    WeightPerturbation = clamped;
                    break;
                case MutationType.ActivationShift:
                    ActivationShift = clamped;
                    break;
                case MutationType.BiasDrift:
                    BiasDrift = clamped;
                    break;
                case MutationType.SynapseGrowth:
                    SynapseGrowth = clamped;
                    break;
                case MutationType.SynapsePruning:
                    SynapsePruning = clamped;
                    break;
                case MutationType.LayerInsertion:
                    LayerInsertion = clamped;
                    break;
                case MutationType.LayerRemoval:
                    LayerRemoval = clamped;
                    break;
                case MutationType.GeneSilencing:
                    GeneSilencing = clamped;
                    break;
                case MutationType.GeneActivation:
                    GeneActivation = clamped;
                    break;
                case MutationType.RegulatoryMutation:
                    RegulatoryMutation = clamped;
                    break;
            }
        }

        /// <summary>
        /// Creates a deep clone of this mutation rate configuration.
        /// </summary>
        /// <returns>A new MutationRate with identical values.</returns>
        public MutationRate Clone()
        {
            return (MutationRate)MemberwiseClone();
        }

        object ICloneable.Clone() => Clone();

        /// <summary>
        /// Scales all mutation rates by a factor.
        /// </summary>
        /// <param name="factor">The scaling factor to apply.</param>
        public void ScaleAll(double factor)
        {
            PointMutation = Math.Clamp(PointMutation * factor, 0.0, 1.0);
            InsertionMutation = Math.Clamp(InsertionMutation * factor, 0.0, 1.0);
            DeletionMutation = Math.Clamp(DeletionMutation * factor, 0.0, 1.0);
            DuplicationMutation = Math.Clamp(DuplicationMutation * factor, 0.0, 1.0);
            InversionMutation = Math.Clamp(InversionMutation * factor, 0.0, 1.0);
            TranslocationMutation = Math.Clamp(TranslocationMutation * factor, 0.0, 1.0);
            SemanticMutation = Math.Clamp(SemanticMutation * factor, 0.0, 1.0);
            TopologyMutation = Math.Clamp(TopologyMutation * factor, 0.0, 1.0);
            WeightPerturbation = Math.Clamp(WeightPerturbation * factor, 0.0, 1.0);
            ActivationShift = Math.Clamp(ActivationShift * factor, 0.0, 1.0);
            BiasDrift = Math.Clamp(BiasDrift * factor, 0.0, 1.0);
            SynapseGrowth = Math.Clamp(SynapseGrowth * factor, 0.0, 1.0);
            SynapsePruning = Math.Clamp(SynapsePruning * factor, 0.0, 1.0);
            LayerInsertion = Math.Clamp(LayerInsertion * factor, 0.0, 1.0);
            LayerRemoval = Math.Clamp(LayerRemoval * factor, 0.0, 1.0);
            GeneSilencing = Math.Clamp(GeneSilencing * factor, 0.0, 1.0);
            GeneActivation = Math.Clamp(GeneActivation * factor, 0.0, 1.0);
            RegulatoryMutation = Math.Clamp(RegulatoryMutation * factor, 0.0, 1.0);
        }

        /// <summary>
        /// Creates a MutationRate with high structural mutation rates for exploration.
        /// </summary>
        /// <returns>An exploration-focused MutationRate configuration.</returns>
        public static MutationRate CreateExploration()
        {
            return new MutationRate
            {
                PointMutation = 0.05,
                InsertionMutation = 0.05,
                DeletionMutation = 0.04,
                DuplicationMutation = 0.03,
                InversionMutation = 0.03,
                TranslocationMutation = 0.03,
                SemanticMutation = 0.1,
                TopologyMutation = 0.15,
                WeightPerturbation = 0.2,
                ActivationShift = 0.05,
                BiasDrift = 0.05,
                SynapseGrowth = 0.1,
                SynapsePruning = 0.05,
                LayerInsertion = 0.03,
                LayerRemoval = 0.02,
                GeneSilencing = 0.03,
                GeneActivation = 0.03,
                RegulatoryMutation = 0.02
            };
        }

        /// <summary>
        /// Creates a MutationRate with low structural mutation rates for exploitation.
        /// </summary>
        /// <returns>An exploitation-focused MutationRate configuration.</returns>
        public static MutationRate CreateExploitation()
        {
            return new MutationRate
            {
                PointMutation = 0.2,
                InsertionMutation = 0.005,
                DeletionMutation = 0.005,
                DuplicationMutation = 0.005,
                InversionMutation = 0.005,
                TranslocationMutation = 0.005,
                SemanticMutation = 0.02,
                TopologyMutation = 0.02,
                WeightPerturbation = 0.4,
                ActivationShift = 0.01,
                BiasDrift = 0.15,
                SynapseGrowth = 0.02,
                SynapsePruning = 0.02,
                LayerInsertion = 0.002,
                LayerRemoval = 0.001,
                GeneSilencing = 0.01,
                GeneActivation = 0.01,
                RegulatoryMutation = 0.005
            };
        }

        /// <summary>
        /// Gets the sum of all mutation rates (the total probability budget).
        /// </summary>
        /// <returns>The sum of all individual mutation rates.</returns>
        public double GetTotalRate()
        {
            return PointMutation + InsertionMutation + DeletionMutation + DuplicationMutation +
                   InversionMutation + TranslocationMutation + SemanticMutation + TopologyMutation +
                   WeightPerturbation + ActivationShift + BiasDrift + SynapseGrowth + SynapsePruning +
                   LayerInsertion + LayerRemoval + GeneSilencing + GeneActivation + RegulatoryMutation;
        }
    }

    /// <summary>
    /// Provides context for fitness evaluation including scene data,
    /// reference images, and rendering parameters.
    /// </summary>
    public sealed class EvaluationContext
    {
        /// <summary>Unique identifier for this evaluation context.</summary>
        public string Id { get; init; } = Guid.NewGuid().ToString("N");

        /// <summary>Target output values for supervised evaluation.</summary>
        public ImmutableArray<double> TargetOutput { get; init; } = ImmutableArray<double>.Empty;

        /// <summary>Input data for evaluation.</summary>
        public ImmutableArray<double> InputData { get; init; } = ImmutableArray<double>.Empty;

        /// <summary>Reference image data for visual fidelity comparison (serialized).</summary>
        public byte[]? ReferenceImageData { get; init; }

        /// <summary>Reference image dimensions (width, height).</summary>
        public Vector2 ReferenceImageDimensions { get; init; }

        /// <summary>Maximum allowed inference latency in milliseconds.</summary>
        public double MaxLatencyMs { get; init; } = 16.67;

        /// <summary>Maximum allowed memory usage in bytes.</summary>
        public long MaxMemoryBytes { get; init; } = 256 * 1024 * 1024;

        /// <summary>Scene-specific parameters.</summary>
        public ImmutableDictionary<string, double> SceneParameters { get; init; } =
            ImmutableDictionary<string, double>.Empty;

        /// <summary>Additional context data (arbitrary key-value pairs).</summary>
        public ImmutableDictionary<string, object> ContextData { get; init; } =
            ImmutableDictionary<string, object>.Empty;

        /// <summary>Weight for each fitness component in multi-objective evaluation.</summary>
        public ImmutableDictionary<FitnessComponent, double> ComponentWeights { get; init; } =
            ImmutableDictionary<FitnessComponent, double>.Empty;

        /// <summary>Number of evaluation samples (for batch evaluation).</summary>
        public int SampleCount { get; init; } = 1;

        /// <summary>Whether this is a regression task (vs classification).</summary>
        public bool IsRegression { get; init; } = true;

        /// <summary>Number of output neurons expected.</summary>
        public int ExpectedOutputSize { get; init; }

        /// <summary>Number of input neurons expected.</summary>
        public int ExpectedInputSize { get; init; }

        /// <summary>
        /// Creates a new EvaluationContext with default component weights.
        /// </summary>
        /// <returns>An EvaluationContext with balanced component weights.</returns>
        public static EvaluationContext CreateDefault()
        {
            return new EvaluationContext
            {
                ComponentWeights = ImmutableDictionary<FitnessComponent, double>.Empty
                    .Add(FitnessComponent.VisualFidelity, 0.3)
                    .Add(FitnessComponent.Performance, 0.2)
                    .Add(FitnessComponent.MemoryEfficiency, 0.1)
                    .Add(FitnessComponent.StructuralComplexity, 0.1)
                    .Add(FitnessComponent.PerceptualQuality, 0.2)
                    .Add(FitnessComponent.SDFError, 0.1),
                IsRegression = true,
                SampleCount = 100
            };
        }

        /// <summary>Empty evaluation context for lightweight sync evaluation.</summary>
        public static EvaluationContext Empty { get; } = new();

        /// <summary>
        /// Merges evolution-config flags into component weights (e.g. L-DNN irradiance fitness).
        /// </summary>
        public EvaluationContext ApplyEvolutionConfig(EvolutionConfig config)
        {
            if (!config.EnableIrradianceFitness)
                return this;

            var weights = ComponentWeights.Count > 0
                ? ComponentWeights
                : CreateDefault().ComponentWeights;

            if (weights.ContainsKey(FitnessComponent.IrradianceError))
                return this;

            return new EvaluationContext
            {
                Id = Id,
                TargetOutput = TargetOutput,
                InputData = InputData,
                ReferenceImageData = ReferenceImageData,
                ReferenceImageDimensions = ReferenceImageDimensions,
                MaxLatencyMs = MaxLatencyMs,
                MaxMemoryBytes = MaxMemoryBytes,
                SceneParameters = SceneParameters,
                ContextData = ContextData,
                ComponentWeights = weights.Add(FitnessComponent.IrradianceError, config.IrradianceFitnessWeight),
                SampleCount = SampleCount,
                IsRegression = IsRegression,
                ExpectedOutputSize = ExpectedOutputSize,
                ExpectedInputSize = ExpectedInputSize
            };
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Genome and Neural Network Models

    /// <summary>
    /// Represents a single neuron (node) in the neural network genome.
    /// Contains the neuron's activation function, bias, and metadata.
    /// </summary>
    public sealed class GeoNeuron : ICloneable, IEquatable<GeoNeuron>
    {
        /// <summary>Globally unique innovation number for this neuron.</summary>
        public long InnovationNumber { get; set; }

        /// <summary>Alias for InnovationNumber used by analysis tooling.</summary>
        public long Id { get => InnovationNumber; set => InnovationNumber = value; }

        /// <summary>Most recent activation value from a forward pass.</summary>
        public double LastActivation { get; set; }

        /// <summary>Layer index (depth) in the network topology.</summary>
        public int LayerIndex { get; set; }

        /// <summary>Position within the layer.</summary>
        public int PositionInLayer { get; set; }

        /// <summary>The activation function for this neuron.</summary>
        public ActivationFunction Activation { get; set; } = ActivationFunction.Tanh;

        /// <summary>Bias value for this neuron.</summary>
        public double Bias { get; set; }

        /// <summary>Whether this neuron is currently active (not silenced).</summary>
        public bool IsActive { get; set; } = true;

        /// <summary>Semantic role label for this neuron (used in semantic crossover).</summary>
        public string? SemanticRole { get; set; }

        /// <summary>Semantic embedding vector for manifold distance computation.</summary>
        public ImmutableArray<double> SemanticEmbedding { get; set; } = ImmutableArray<double>.Empty;

        /// <summary>Number of times this neuron has been expressed across generations.</summary>
        public int ExpressionCount { get; set; } = 1;

        /// <summary>Generation when this neuron was created.</summary>
        public int CreationGeneration { get; set; }

        /// <summary>Custom metadata tags.</summary>
        public ImmutableDictionary<string, string> Tags { get; set; } =
            ImmutableDictionary<string, string>.Empty;

        /// <summary>Applies the activation function to the given input.</summary>
        /// <param name="x">The weighted sum input to the neuron.</param>
        /// <returns>The activated output value.</returns>
        public double Activate(double x)
        {
            return Activation switch
            {
                ActivationFunction.Tanh => Math.Tanh(x),
                ActivationFunction.Sigmoid => 1.0 / (1.0 + Math.Exp(-x)),
                ActivationFunction.ReLU => Math.Max(0, x),
                ActivationFunction.LeakyReLU => x >= 0 ? x : 0.01 * x,
                ActivationFunction.GELU => 0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x))),
                ActivationFunction.Swish => x / (1.0 + Math.Exp(-x)),
                ActivationFunction.Sinusoidal => Math.Sin(x),
                ActivationFunction.Linear => x,
                ActivationFunction.Abs => Math.Abs(x),
                ActivationFunction.Step => x >= 0 ? 1.0 : 0.0,
                ActivationFunction.Softplus => Math.Log(1.0 + Math.Exp(x)),
                ActivationFunction.Mish => x * Math.Tanh(Math.Log(1.0 + Math.Exp(x))),
                ActivationFunction.Exponential => Math.Exp(Math.Clamp(x, -10, 10)),
                _ => x
            };
        }

        /// <summary>
        /// Computes the derivative of the activation function at the given input.
        /// Used for gradient approximation in hybrid learning.
        /// </summary>
        /// <param name="x">The weighted sum input to the neuron.</param>
        /// <returns>The derivative value.</returns>
        public double ActivateDerivative(double x)
        {
            switch (Activation)
            {
                case ActivationFunction.Tanh:
                    return 1.0 - Math.Tanh(x) * Math.Tanh(x);
                case ActivationFunction.Sigmoid:
                    {
                        var s = 1.0 / (1.0 + Math.Exp(-x));
                        return s * (1.0 - s);
                    }
                case ActivationFunction.ReLU:
                    return x >= 0 ? 1.0 : 0.0;
                case ActivationFunction.LeakyReLU:
                    return x >= 0 ? 1.0 : 0.01;
                case ActivationFunction.GELU:
                    {
                        var t = Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x));
                        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * Math.Sqrt(2.0 / Math.PI) * (1.0 + 3.0 * 0.044715 * x * x);
                    }
                case ActivationFunction.Swish:
                    {
                        var s = 1.0 / (1.0 + Math.Exp(-x));
                        return s + x * s * (1.0 - s);
                    }
                case ActivationFunction.Sinusoidal:
                    return Math.Cos(x);
                case ActivationFunction.Linear:
                    return 1.0;
                case ActivationFunction.Abs:
                    return x >= 0 ? 1.0 : -1.0;
                case ActivationFunction.Step:
                    return 0.0;
                case ActivationFunction.Softplus:
                    return 1.0 / (1.0 + Math.Exp(-x));
                case ActivationFunction.Mish:
                    {
                        var sp = Math.Log(1.0 + Math.Exp(x));
                        var tanhSp = Math.Tanh(sp);
                        var sigmoid = 1.0 / (1.0 + Math.Exp(-x));
                        return tanhSp + x * sigmoid * (1.0 - tanhSp * tanhSp) / (1.0 + Math.Exp(x));
                    }
                case ActivationFunction.Exponential:
                    return Math.Exp(Math.Clamp(x, -10, 10));
                default:
                    return 1.0;
            }
        }

        /// <summary>Creates a deep clone of this neuron.</summary>
        /// <returns>A new GeoNeuron with identical properties.</returns>
        public GeoNeuron Clone()
        {
            return new GeoNeuron
            {
                InnovationNumber = InnovationNumber,
                LayerIndex = LayerIndex,
                PositionInLayer = PositionInLayer,
                Activation = Activation,
                Bias = Bias,
                IsActive = IsActive,
                SemanticRole = SemanticRole,
                SemanticEmbedding = SemanticEmbedding,
                ExpressionCount = ExpressionCount,
                CreationGeneration = CreationGeneration,
                Tags = Tags
            };
        }

        object ICloneable.Clone() => Clone();

        /// <summary>Determines equality based on InnovationNumber.</summary>
        public bool Equals(GeoNeuron? other) =>
            other is not null && InnovationNumber == other.InnovationNumber;

        /// <summary>Gets the hash code based on InnovationNumber.</summary>
        public override int GetHashCode() => InnovationNumber.GetHashCode();

        /// <summary>Returns a string representation of this neuron.</summary>
        public override string ToString() =>
            $"GeoNeuron(Innovation={InnovationNumber}, Layer={LayerIndex}, Activation={Activation}, Active={IsActive})";

        public override bool Equals(object obj)
        {
            return Equals(obj as GeoNeuron);
        }
    }

    /// <summary>
    /// Represents a single connection (synapse/edge) in the neural network genome.
    /// Contains weight, source/target neuron references, and metadata.
    /// </summary>
    public sealed class GeoSynapse : ICloneable, IEquatable<GeoSynapse>
    {
        /// <summary>Globally unique innovation number for this synapse.</summary>
        public long InnovationNumber { get; set; }

        /// <summary>Alias for InnovationNumber used by analysis tooling.</summary>
        public long Id { get => InnovationNumber; set => InnovationNumber = value; }

        /// <summary>Innovation number of the source neuron.</summary>
        public long SourceNeuronId { get; set; }

        /// <summary>Innovation number of the target neuron.</summary>
        public long TargetNeuronId { get; set; }

        /// <summary>Connection weight.</summary>
        public double Weight { get; set; }

        /// <summary>Whether this synapse is currently active.</summary>
        public bool IsActive { get; set; } = true;

        /// <summary>Whether this is a recurrent connection.</summary>
        public bool IsRecurrent { get; set; }

        /// <summary>Temporal delay for recurrent connections (in time steps).</summary>
        public int RecurrentDelay { get; set; } = 1;

        /// <summary>Generation when this synapse was created.</summary>
        public int CreationGeneration { get; set; }

        /// <summary>Number of times this synapse has been expressed.</summary>
        public int ExpressionCount { get; set; } = 1;

        /// <summary>Semantic role label for weight alignment in crossover.</summary>
        public string? SemanticRole { get; set; }

        /// <summary>Confidence score for this connection (learned during evolution).</summary>
        public double Confidence { get; set; } = 1.0;

        /// <summary>Creates a deep clone of this synapse.</summary>
        /// <returns>A new GeoSynapse with identical properties.</returns>
        public GeoSynapse Clone()
        {
            return new GeoSynapse
            {
                InnovationNumber = InnovationNumber,
                SourceNeuronId = SourceNeuronId,
                TargetNeuronId = TargetNeuronId,
                Weight = Weight,
                IsActive = IsActive,
                IsRecurrent = IsRecurrent,
                RecurrentDelay = RecurrentDelay,
                CreationGeneration = CreationGeneration,
                ExpressionCount = ExpressionCount,
                SemanticRole = SemanticRole,
                Confidence = Confidence
            };
        }

        object ICloneable.Clone() => Clone();

        /// <summary>Determines equality based on InnovationNumber.</summary>
        public bool Equals(GeoSynapse? other) =>
            other is not null && InnovationNumber == other.InnovationNumber;

        /// <summary>Gets the hash code based on InnovationNumber.</summary>
        public override int GetHashCode() => InnovationNumber.GetHashCode();

        /// <summary>Returns a string representation of this synapse.</summary>
        public override string ToString() =>
            $"GeoSynapse(Innovation={InnovationNumber}, {SourceNeuronId}->{TargetNeuronId}, Weight={Weight:F4}, Active={IsActive})";

        public override bool Equals(object obj)
        {
            return Equals(obj as GeoSynapse);
        }
    }

    /// <summary>
    /// Represents a geometric genome - the complete genetic representation of a neural network.
    /// Contains neurons, synapses, and all metadata needed for evolution.
    /// This is the primary data structure evolved by the NEAT-G algorithm.
    /// </summary>
    public sealed class GeoGenome : ICloneable
    {
        private readonly object _lock = new();
        private double _cachedFitness = double.NaN;
        private bool _fitnessDirty = true;

        /// <summary>Globally unique identifier for this genome.</summary>
        public Guid Id { get; set; } = Guid.NewGuid();

        /// <summary>Ordered list of neurons in this genome.</summary>
        public List<GeoNeuron> Neurons { get; set; } = new();

        /// <summary>Ordered list of synapses in this genome.</summary>
        public List<GeoSynapse> Synapses { get; set; } = new();

        /// <summary>Fitness score (higher is better by convention).</summary>
        public double Fitness
        {
            get
            {
                lock (_lock)
                {
                    return _cachedFitness;
                }
            }
            set
            {
                lock (_lock)
                {
                    _cachedFitness = value;
                    _fitnessDirty = false;
                }
            }
        }

        /// <summary>Adjusted fitness after fitness sharing.</summary>
        public double AdjustedFitness { get; set; }

        /// <summary>Raw fitness before any adjustments.</summary>
        public double RawFitness { get; set; }

        /// <summary>Generation when this genome was created.</summary>
        public int Generation { get; set; }

        /// <summary>Species ID this genome belongs to.</summary>
        public int SpeciesId { get; set; } = -1;

        /// <summary>Number of generations this genome has survived.</summary>
        public int Age { get; set; }

        /// <summary>Best fitness ever achieved by this genome.</summary>
        public double BestFitness { get; set; } = double.MinValue;

        /// <summary>Generation when best fitness was achieved.</summary>
        public int BestFitnessGeneration { get; set; }

        /// <summary>Number of evaluations this genome has undergone.</summary>
        public int EvaluationCount { get; set; }

        /// <summary>Number of input neurons.</summary>
        public int InputCount { get; set; }

        /// <summary>Number of output neurons.</summary>
        public int OutputCount { get; set; }

        /// <summary>Parent genome IDs (for lineage tracking).</summary>
        public ImmutableArray<Guid> ParentIds { get; set; } = ImmutableArray<Guid>.Empty;

        /// <summary>Multi-objective fitness components.</summary>
        public ImmutableDictionary<FitnessComponent, double> FitnessComponents { get; set; } =
            ImmutableDictionary<FitnessComponent, double>.Empty;

        /// <summary>Semantic embedding vector for the entire genome.</summary>
        public ImmutableArray<double> SemanticEmbedding { get; set; } = ImmutableArray<double>.Empty;

        /// <summary>Complexity metric of this genome (used as regularization).</summary>
        public double Complexity { get; set; }

        /// <summary>Number of active neurons.</summary>
        public int ActiveNeuronCount => Neurons.Count(n => n.IsActive);

        /// <summary>Number of active synapses.</summary>
        public int ActiveSynapseCount => Synapses.Count(s => s.IsActive);

        /// <summary>Total number of neurons.</summary>
        public int TotalNeuronCount => Neurons.Count;

        /// <summary>Total number of synapses.</summary>
        public int TotalSynapseCount => Synapses.Count;

        /// <summary>Maximum layer depth in this genome.</summary>
        public int MaxLayerDepth => Neurons.Count > 0 ? Neurons.Max(n => n.LayerIndex) : 0;

        /// <summary>Connection density (active synapses / max possible connections).</summary>
        public double ConnectionDensity
        {
            get
            {
                var maxConn = (long)ActiveNeuronCount * (ActiveNeuronCount - 1);
                return maxConn > 0 ? (double)ActiveSynapseCount / maxConn : 0.0;
            }
        }

        /// <summary>
        /// Gets or sets the neuron at the specified index.
        /// </summary>
        /// <param name="index">The index of the neuron.</param>
        /// <returns>The neuron at the specified index.</returns>
        public GeoNeuron this[int index]
        {
            get
            {
                lock (_lock)
                {
                    return Neurons[index];
                }
            }
        }

        /// <summary>
        /// Invalidates the cached fitness, requiring re-evaluation.
        /// </summary>
        public void InvalidateFitness()
        {
            lock (_lock)
            {
                _fitnessDirty = true;
                _cachedFitness = double.NaN;
            }
        }

        /// <summary>
        /// Gets whether the fitness cache is valid.
        /// </summary>
        public bool IsFitnessValid
        {
            get
            {
                lock (_lock)
                {
                    return !_fitnessDirty && !double.IsNaN(_cachedFitness);
                }
            }
        }

        /// <summary>
        /// Computes the structural complexity of this genome.
        /// Complexity is measured as a weighted combination of neuron count,
        /// synapse count, and topological depth.
        /// </summary>
        /// <returns>The complexity metric value.</returns>
        public double ComputeComplexity()
        {
            double neuronContribution = ActiveNeuronCount * 1.0;
            double synapseContribution = ActiveSynapseCount * 0.5;
            double depthContribution = MaxLayerDepth * 2.0;
            double densityContribution = ConnectionDensity * 10.0;
            Complexity = neuronContribution + synapseContribution + depthContribution + densityContribution;
            return Complexity;
        }

        /// <summary>
        /// Computes a semantic embedding vector for this genome based on its topology
        /// and weight structure. Used for semantic crossover alignment and manifold distance.
        /// </summary>
        /// <param name="dimension">The dimensionality of the embedding.</param>
        /// <returns>An immutable array representing the embedding vector.</returns>
        public ImmutableArray<double> ComputeSemanticEmbedding(int dimension = 32)
        {
            var embedding = new double[dimension];
            var rng = new Random(Id.GetHashCode());

            double totalWeight = 0;
            int activeSynapseCount = 0;

            foreach (var synapse in Synapses)
            {
                if (!synapse.IsActive)
                    continue;
                totalWeight += Math.Abs(synapse.Weight);
                activeSynapseCount++;

                int srcHash = synapse.SourceNeuronId.GetHashCode();
                int tgtHash = synapse.TargetNeuronId.GetHashCode();

                for (int d = 0; d < dimension; d++)
                {
                    double phase = (srcHash * (d + 1) * 0.1 + tgtHash * (d + 1) * 0.07);
                    embedding[d] += synapse.Weight * Math.Sin(phase + d * 0.5);
                }
            }

            foreach (var neuron in Neurons)
            {
                if (!neuron.IsActive)
                    continue;

                int nHash = neuron.InnovationNumber.GetHashCode();
                for (int d = 0; d < dimension; d++)
                {
                    double phase = nHash * (d + 1) * 0.13;
                    embedding[d] += neuron.Bias * Math.Cos(phase + d * 0.3);
                    embedding[d] += (double)neuron.Activation / 13.0 * Math.Sin(phase);
                }
            }

            double norm = 0;
            for (int d = 0; d < dimension; d++)
                norm += embedding[d] * embedding[d];
            norm = Math.Sqrt(norm);

            if (norm > 1e-10)
            {
                for (int d = 0; d < dimension; d++)
                    embedding[d] /= norm;
            }

            SemanticEmbedding = embedding.ToImmutableArray();
            return SemanticEmbedding;
        }

        /// <summary>
        /// Computes the topological hash of this genome for fast equality checking.
        /// The hash encodes the structure (connections) rather than weights.
        /// </summary>
        /// <returns>A hash value representing the genome topology.</returns>
        public long ComputeTopologyHash()
        {
            long hash = 17;
            foreach (var synapse in Synapses.Where(s => s.IsActive).OrderBy(s => s.InnovationNumber))
            {
                hash = hash * 31 + synapse.SourceNeuronId;
                hash = hash * 31 + synapse.TargetNeuronId;
            }
            return hash;
        }

        /// <summary>
        /// Gets all neurons that are inputs to the specified neuron.
        /// </summary>
        /// <param name="neuronId">The innovation number of the target neuron.</param>
        /// <returns>A list of source neurons connected to the target.</returns>
        public IReadOnlyList<GeoNeuron> GetInputNeurons(long neuronId)
        {
            var sourceIds = Synapses
                .Where(s => s.IsActive && s.TargetNeuronId == neuronId)
                .Select(s => s.SourceNeuronId)
                .ToHashSet();
            return Neurons.Where(n => n.IsActive && sourceIds.Contains(n.InnovationNumber)).ToList();
        }

        /// <summary>
        /// Gets all neurons that are outputs from the specified neuron.
        /// </summary>
        /// <param name="neuronId">The innovation number of the source neuron.</param>
        /// <returns>A list of target neurons connected from the source.</returns>
        public IReadOnlyList<GeoNeuron> GetOutputNeurons(long neuronId)
        {
            var targetIds = Synapses
                .Where(s => s.IsActive && s.SourceNeuronId == neuronId)
                .Select(s => s.TargetNeuronId)
                .ToHashSet();
            return Neurons.Where(n => n.IsActive && targetIds.Contains(n.InnovationNumber)).ToList();
        }

        /// <summary>
        /// Checks whether adding a synapse between two neurons would create a cycle.
        /// Uses DFS to detect cycles in the directed graph.
        /// </summary>
        /// <param name="sourceId">Source neuron innovation number.</param>
        /// <param name="targetId">Target neuron innovation number.</param>
        /// <returns>True if adding the connection would create a cycle.</returns>
        public bool WouldCreateCycle(long sourceId, long targetId)
        {
            if (sourceId == targetId)
                return true;

            var visited = new HashSet<long>();
            var stack = new Stack<long>();
            stack.Push(targetId);

            while (stack.Count > 0)
            {
                var current = stack.Pop();
                if (current == sourceId)
                    return true;
                if (!visited.Add(current))
                    continue;

                foreach (var synapse in Synapses.Where(s => s.IsActive && s.SourceNeuronId == current))
                {
                    stack.Push(synapse.TargetNeuronId);
                }
            }

            return false;
        }

        /// <summary>
        /// Creates a deep clone of this genome, including all neurons, synapses, and metadata.
        /// </summary>
        /// <returns>A new GeoGenome that is a deep copy of this genome.</returns>
        public GeoGenome Clone()
        {
            var clone = new GeoGenome
            {
                Id = Guid.NewGuid(),
                Generation = Generation,
                SpeciesId = SpeciesId,
                Age = Age,
                BestFitness = BestFitness,
                BestFitnessGeneration = BestFitnessGeneration,
                EvaluationCount = EvaluationCount,
                InputCount = InputCount,
                OutputCount = OutputCount,
                ParentIds = ParentIds.Add(Id),
                FitnessComponents = FitnessComponents,
                SemanticEmbedding = SemanticEmbedding,
                Complexity = Complexity
            };

            foreach (var neuron in Neurons)
                clone.Neurons.Add(neuron.Clone());

            foreach (var synapse in Synapses)
                clone.Synapses.Add(synapse.Clone());

            return clone;
        }

        object ICloneable.Clone() => Clone();

        /// <summary>
        /// Returns a summary string representation of this genome.
        /// </summary>
        public override string ToString() =>
            $"GeoGenome(Id={Id:N8}, Neurons={ActiveNeuronCount}/{TotalNeuronCount}, " +
            $"Synapses={ActiveSynapseCount}/{TotalSynapseCount}, Fitness={Fitness:F4}, Gen={Generation})";

        /// <summary>Runs a forward pass through this genome's network.</summary>
        public double[] ForwardPass(ImmutableArray<double> input) =>
            FitnessEvaluator.ForwardPass(this, input);

        /// <summary>Runs a forward pass through this genome's network.</summary>
        public double[] ForwardPass(double[] input) =>
            ForwardPass(input?.ToImmutableArray() ?? ImmutableArray<double>.Empty);
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Records and Structs

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

    #endregion
    // =========================================================================

    // =========================================================================
    #region Interfaces

    /// <summary>
    /// Defines the strategy for evolving a population of genomes across generations.
    /// Implementations control the overall flow of the evolutionary process.
    /// </summary>
    public interface IEvolutionStrategy
    {
        /// <summary>
        /// Performs one generation of evolution on the given population.
        /// </summary>
        /// <param name="population">The current genome population to evolve.</param>
        /// <param name="ct">Cancellation token to stop evolution.</param>
        /// <returns>The evolved population after one generation.</returns>
        Task<GenomePopulation> EvolveAsync(GenomePopulation population, CancellationToken ct);

        /// <summary>
        /// Gets metrics about the most recent evolution step.
        /// </summary>
        /// <returns>Current evolution metrics.</returns>
        EvolutionMetrics GetEvolutionMetrics();
    }

    /// <summary>
    /// Defines the strategy for crossover (recombination) of two parent genomes.
    /// Supports both structural and parametric crossover with semantic alignment.
    /// </summary>
    public interface ICrossoverStrategy
    {
        /// <summary>
        /// Performs crossover between two parent genomes to produce offspring.
        /// </summary>
        /// <param name="parentA">The first parent genome.</param>
        /// <param name="parentB">The second parent genome.</param>
        /// <param name="blendBias">Bias toward parent A (0.0) or parent B (1.0). 0.5 = equal.</param>
        /// <returns>A crossover result containing the offspring and metadata.</returns>
        CrossoverResult Crossover(GeoGenome parentA, GeoGenome parentB, float blendBias);

        /// <summary>
        /// Aligns two parent genomes for crossover by matching corresponding genes.
        /// </summary>
        /// <param name="a">First parent genome.</param>
        /// <param name="b">Second parent genome.</param>
        /// <returns>A tuple of aligned genomes (potentially with gap-filling genes).</returns>
        (GeoGenome alignedA, GeoGenome alignedB) AlignParents(GeoGenome a, GeoGenome b);
    }

    /// <summary>
    /// Defines the strategy for mutating a genome.
    /// Implementations apply various types of mutations to evolve genome structure and parameters.
    /// </summary>
    public interface IMutationStrategy
    {
        /// <summary>
        /// Applies mutations to a genome based on the specified mutation rates.
        /// </summary>
        /// <param name="genome">The genome to mutate.</param>
        /// <param name="rates">The mutation rates to use.</param>
        /// <param name="rng">Random number generator for stochastic operations.</param>
        /// <returns>A mutation result containing the mutated genome and metadata.</returns>
        MutationResult Mutate(GeoGenome genome, MutationRate rates, Random rng);

        /// <summary>
        /// Gets the success rate of mutations over a recent window.
        /// </summary>
        /// <returns>The mutation success rate (0.0 to 1.0).</returns>
        double GetSuccessRate();
    }

    /// <summary>
    /// Defines the strategy for evaluating genome fitness.
    /// Implementations must provide both synchronous and asynchronous evaluation.
    /// </summary>
    public interface IFitnessEvaluator
    {
        /// <summary>
        /// Asynchronously evaluates the fitness of a genome in the given context.
        /// </summary>
        /// <param name="genome">The genome to evaluate.</param>
        /// <param name="context">The evaluation context with scene data and parameters.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>The evaluated genome with updated fitness scores.</returns>
        Task<GeoGenome> EvaluateAsync(GeoGenome genome, EvaluationContext context, CancellationToken ct);

        /// <summary>
        /// Synchronously evaluates genome fitness (defaults to empty context).
        /// </summary>
        double Evaluate(GeoGenome genome) =>
            EvaluateAsync(genome, EvaluationContext.Empty, CancellationToken.None)
                .GetAwaiter().GetResult().Fitness;

        /// <summary>
        /// Gets the per-component weights for multi-objective fitness evaluation.
        /// </summary>
        /// <returns>Immutable dictionary of component weights.</returns>
        ImmutableDictionary<FitnessComponent, double> GetComponentWeights();
    }

    /// <summary>
    /// Defines the strategy for speciation - grouping similar genomes into species.
    /// Uses distance metrics to determine genome similarity.
    /// </summary>
    public interface ISpeciationStrategy
    {
        /// <summary>
        /// Divides the population into species based on genome similarity.
        /// </summary>
        /// <param name="population">The population to speciate.</param>
        /// <returns>An immutable array of species information.</returns>
        ImmutableArray<SpeciesInfo> Speciate(GenomePopulation population);

        /// <summary>
        /// Computes the distance between two genomes using the speciation metric.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>The distance between the genomes (lower = more similar).</returns>
        double ComputeDistance(GeoGenome a, GeoGenome b);
    }

    /// <summary>
    /// Defines the selection strategy for choosing parents and survivors.
    /// </summary>
    public interface ISelectionStrategy
    {
        /// <summary>
        /// Selects parents from the population for reproduction.
        /// </summary>
        /// <param name="population">The population to select from.</param>
        /// <param name="count">Number of parents to select.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>The selected parent genomes.</returns>
        IReadOnlyList<GeoGenome> SelectParents(GenomePopulation population, int count, Random rng);

        /// <summary>
        /// Selects survivors from the current and offspring populations.
        /// </summary>
        /// <param name="current">Current population members.</param>
        /// <param name="offspring">Newly generated offspring.</param>
        /// <param name="targetSize">Target population size after selection.</param>
        /// <returns>The surviving genomes.</returns>
        IReadOnlyList<GeoGenome> SelectSurvivors(
            IReadOnlyList<GeoGenome> current,
            IReadOnlyList<GeoGenome> offspring,
            int targetSize);
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Strategy Implementations

    /// <summary>
    /// Semantic crossover strategy that aligns parent genomes based on
    /// the semantic roles of their neurons. Neurons with similar semantic
    /// roles are matched and blended, preserving functional building blocks.
    /// This strategy is particularly effective for preserving learned features
    /// across generations.
    /// </summary>
    public sealed class SemanticCrossoverStrategy : ICrossoverStrategy
    {
        private readonly EvolutionConfig _config;
        private int _crossoverCount;
        private int _successCount;

        /// <summary>
        /// Initializes a new instance of the SemanticCrossoverStrategy class.
        /// </summary>
        /// <param name="config">Evolution configuration parameters.</param>
        public SemanticCrossoverStrategy(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <inheritdoc/>
        public CrossoverResult Crossover(GeoGenome parentA, GeoGenome parentB, float blendBias)
        {
            Interlocked.Increment(ref _crossoverCount);

            if (parentA == null)
                throw new ArgumentNullException(nameof(parentA));
            if (parentB == null)
                throw new ArgumentNullException(nameof(parentB));

            var alignedA = parentA.Clone();
            var alignedB = parentB.Clone();

            if (alignedA.SemanticEmbedding.IsDefaultOrEmpty)
                alignedA.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);
            if (alignedB.SemanticEmbedding.IsDefaultOrEmpty)
                alignedB.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

            var (filledA, filledB) = AlignParents(parentA, parentB);
            var offspring = filledA.Clone();
            offspring.Id = Guid.NewGuid();
            offspring.ParentIds = ImmutableArray.Create(parentA.Id, parentB.Id);
            offspring.Generation = Math.Max(parentA.Generation, parentB.Generation) + 1;
            offspring.Fitness = double.NaN;
            offspring.InvalidateFitness();

            var rng = new Random(Guid.NewGuid().GetHashCode());
            int matchingGenes = 0;
            int disjointGenes = 0;
            double totalWeightDiff = 0;

            var bNeuronMap = new Dictionary<long, GeoNeuron>();
            foreach (var n in filledB.Neurons)
                bNeuronMap[n.InnovationNumber] = n;

            for (int i = 0; i < offspring.Neurons.Count; i++)
            {
                var neuronA = filledA.Neurons[i];
                if (bNeuronMap.TryGetValue(neuronA.InnovationNumber, out var neuronB))
                {
                    matchingGenes++;
                    double bias = neuronA.Bias * (1.0 - blendBias) + neuronB.Bias * blendBias;
                    offspring.Neurons[i].Bias = bias;
                    offspring.Neurons[i].Activation = rng.NextDouble() < 0.5
                        ? neuronA.Activation
                        : neuronB.Activation;
                    offspring.Neurons[i].IsActive = neuronA.IsActive && neuronB.IsActive
                        ? rng.NextDouble() < 0.5
                        : neuronA.IsActive || neuronB.IsActive;
                }
                else
                {
                    disjointGenes++;
                    offspring.Neurons[i].IsActive = rng.NextDouble() < 0.7;
                }
            }

            var bSynapseMap = new Dictionary<long, GeoSynapse>();
            foreach (var s in filledB.Synapses)
                bSynapseMap[s.InnovationNumber] = s;

            for (int i = 0; i < offspring.Synapses.Count; i++)
            {
                var synapseA = filledA.Synapses[i];
                if (bSynapseMap.TryGetValue(synapseA.InnovationNumber, out var synapseB))
                {
                    double weightDiff = Math.Abs(synapseA.Weight - synapseB.Weight);
                    totalWeightDiff += weightDiff;
                    offspring.Synapses[i].Weight = synapseA.Weight * (1.0 - blendBias)
                        + synapseB.Weight * blendBias;
                    offspring.Synapses[i].IsActive = synapseA.IsActive || synapseB.IsActive;
                }
                else
                {
                    disjointGenes++;
                    offspring.Synapses[i].IsActive = rng.NextDouble() < 0.7;
                }
            }

            bool success = offspring.ActiveNeuronCount >= parentA.InputCount + parentA.OutputCount;
            if (success)
                Interlocked.Increment(ref _successCount);

            return new CrossoverResult
            {
                Offspring = offspring,
                IsSuccess = success,
                StrategyUsed = nameof(SemanticCrossoverStrategy),
                MatchingGenes = matchingGenes,
                DisjointGenes = disjointGenes,
                AverageWeightDifference = matchingGenes > 0 ? totalWeightDiff / matchingGenes : 0
            };
        }

        /// <inheritdoc/>
        public (GeoGenome alignedA, GeoGenome alignedB) AlignParents(GeoGenome a, GeoGenome b)
        {
            var alignedA = a.Clone();
            var alignedB = b.Clone();

            var aNeuronIds = new HashSet<long>(a.Neurons.Select(n => n.InnovationNumber));
            var bNeuronIds = new HashSet<long>(b.Neurons.Select(n => n.InnovationNumber));

            long maxInnovation = 0;
            foreach (var n in a.Neurons)
                maxInnovation = Math.Max(maxInnovation, n.InnovationNumber);
            foreach (var n in b.Neurons)
                maxInnovation = Math.Max(maxInnovation, n.InnovationNumber);
            foreach (var s in a.Synapses)
                maxInnovation = Math.Max(maxInnovation, s.InnovationNumber);
            foreach (var s in b.Synapses)
                maxInnovation = Math.Max(maxInnovation, s.InnovationNumber);

            var missingInB = aNeuronIds.Except(bNeuronIds).ToList();
            foreach (var nId in missingInB)
            {
                var original = a.Neurons.First(n => n.InnovationNumber == nId);
                var gap = original.Clone();
                gap.IsActive = false;
                alignedB.Neurons.Add(gap);
            }

            var missingInA = bNeuronIds.Except(aNeuronIds).ToList();
            foreach (var nId in missingInA)
            {
                var original = b.Neurons.First(n => n.InnovationNumber == nId);
                var gap = original.Clone();
                gap.IsActive = false;
                alignedA.Neurons.Add(gap);
            }

            var aSynapseIds = new HashSet<long>(a.Synapses.Select(s => s.InnovationNumber));
            var bSynapseIds = new HashSet<long>(b.Synapses.Select(s => s.InnovationNumber));

            var missingSynapsesInB = aSynapseIds.Except(bSynapseIds).ToList();
            foreach (var sId in missingSynapsesInB)
            {
                var original = a.Synapses.First(s => s.InnovationNumber == sId);
                var gap = original.Clone();
                gap.IsActive = false;
                alignedB.Synapses.Add(gap);
            }

            var missingSynapsesInA = bSynapseIds.Except(aSynapseIds).ToList();
            foreach (var sId in missingSynapsesInA)
            {
                var original = b.Synapses.First(s => s.InnovationNumber == sId);
                var gap = original.Clone();
                gap.IsActive = false;
                alignedA.Synapses.Add(gap);
            }

            alignedA.Neurons.Sort((x, y) => x.InnovationNumber.CompareTo(y.InnovationNumber));
            alignedB.Neurons.Sort((x, y) => x.InnovationNumber.CompareTo(y.InnovationNumber));
            alignedA.Synapses.Sort((x, y) => x.InnovationNumber.CompareTo(y.InnovationNumber));
            alignedB.Synapses.Sort((x, y) => x.InnovationNumber.CompareTo(y.InnovationNumber));

            return (alignedA, alignedB);
        }

        /// <summary>
        /// Gets the crossover success rate.
        /// </summary>
        /// <returns>The success rate as a fraction.</returns>
        public double GetSuccessRate()
        {
            int total = Volatile.Read(ref _crossoverCount);
            int success = Volatile.Read(ref _successCount);
            return total > 0 ? (double)success / total : 0;
        }
    }

    /// <summary>
    /// Topology-based crossover strategy that aligns parent genomes using
    /// innovation numbers (chronological ordering of structural additions).
    /// Preserves schemata by matching structural features between parents.
    /// This is the classic NEAT crossover approach extended for geometric genomes.
    /// </summary>
    public sealed class TopologyCrossoverStrategy : ICrossoverStrategy
    {
        private readonly EvolutionConfig _config;
        private int _crossoverCount;
        private int _successCount;

        /// <summary>
        /// Initializes a new instance of the TopologyCrossoverStrategy class.
        /// </summary>
        /// <param name="config">Evolution configuration parameters.</param>
        public TopologyCrossoverStrategy(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <inheritdoc/>
        public CrossoverResult Crossover(GeoGenome parentA, GeoGenome parentB, float blendBias)
        {
            Interlocked.Increment(ref _crossoverCount);

            if (parentA == null)
                throw new ArgumentNullException(nameof(parentA));
            if (parentB == null)
                throw new ArgumentNullException(nameof(parentB));

            var (alignedA, alignedB) = AlignParents(parentA, parentB);
            var offspring = alignedA.Clone();
            offspring.Id = Guid.NewGuid();
            offspring.ParentIds = ImmutableArray.Create(parentA.Id, parentB.Id);
            offspring.Generation = Math.Max(parentA.Generation, parentB.Generation) + 1;
            offspring.Fitness = double.NaN;
            offspring.InvalidateFitness();

            var rng = new Random(Guid.NewGuid().GetHashCode());
            int matchingGenes = 0;
            int disjointGenes = 0;
            double totalWeightDiff = 0;

            for (int i = 0; i < Math.Min(alignedA.Neurons.Count, alignedB.Neurons.Count); i++)
            {
                var nA = alignedA.Neurons[i];
                var nB = alignedB.Neurons[i];

                if (nA.InnovationNumber == nB.InnovationNumber)
                {
                    matchingGenes++;
                    offspring.Neurons[i].Bias = nA.Bias * (1.0 - blendBias) + nB.Bias * blendBias;
                    offspring.Neurons[i].Activation = rng.NextDouble() < 0.5
                        ? nA.Activation
                        : nB.Activation;
                    offspring.Neurons[i].IsActive = nA.IsActive || nB.IsActive;
                    offspring.Neurons[i].SemanticRole = nA.SemanticRole ?? nB.SemanticRole;
                }
                else
                {
                    disjointGenes++;
                    offspring.Neurons[i].IsActive = false;
                }
            }

            for (int i = 0; i < Math.Min(alignedA.Synapses.Count, alignedB.Synapses.Count); i++)
            {
                var sA = alignedA.Synapses[i];
                var sB = alignedB.Synapses[i];

                if (sA.InnovationNumber == sB.InnovationNumber)
                {
                    matchingGenes++;
                    totalWeightDiff += Math.Abs(sA.Weight - sB.Weight);
                    offspring.Synapses[i].Weight = sA.Weight * (1.0 - blendBias)
                        + sB.Weight * blendBias;
                    offspring.Synapses[i].IsActive = sA.IsActive || sB.IsActive;
                }
                else
                {
                    disjointGenes++;
                    offspring.Synapses[i].IsActive = false;
                }
            }

            int activeRequired = parentA.InputCount + parentA.OutputCount;
            bool success = offspring.ActiveNeuronCount >= activeRequired;
            if (success)
                Interlocked.Increment(ref _successCount);

            return new CrossoverResult
            {
                Offspring = offspring,
                IsSuccess = success,
                StrategyUsed = nameof(TopologyCrossoverStrategy),
                MatchingGenes = matchingGenes,
                DisjointGenes = disjointGenes,
                AverageWeightDifference = matchingGenes > 0 ? totalWeightDiff / matchingGenes : 0
            };
        }

        /// <inheritdoc/>
        public (GeoGenome alignedA, GeoGenome alignedB) AlignParents(GeoGenome a, GeoGenome b)
        {
            var alignedA = a.Clone();
            var alignedB = b.Clone();

            var allInnovations = new SortedSet<long>();
            foreach (var n in a.Neurons)
                allInnovations.Add(n.InnovationNumber);
            foreach (var n in b.Neurons)
                allInnovations.Add(n.InnovationNumber);
            foreach (var s in a.Synapses)
                allInnovations.Add(s.InnovationNumber);
            foreach (var s in b.Synapses)
                allInnovations.Add(s.InnovationNumber);

            var aNeuronMap = a.Neurons.ToDictionary(n => n.InnovationNumber);
            var bNeuronMap = b.Neurons.ToDictionary(n => n.InnovationNumber);
            var aSynapseMap = a.Synapses.ToDictionary(s => s.InnovationNumber);
            var bSynapseMap = b.Synapses.ToDictionary(s => s.InnovationNumber);

            var resultA = new List<GeoNeuron>();
            var resultB = new List<GeoNeuron>();

            foreach (var innov in allInnovations.Where(id => !aSynapseMap.ContainsKey(id) && !bSynapseMap.ContainsKey(id)))
            {
                if (aNeuronMap.TryGetValue(innov, out var nA))
                    resultA.Add(nA.Clone());
                else
                {
                    var placeholder = CreatePlaceholderNeuron(innov, a.Generation);
                    resultA.Add(placeholder);
                }

                if (bNeuronMap.TryGetValue(innov, out var nB))
                    resultB.Add(nB.Clone());
                else
                {
                    var placeholder = CreatePlaceholderNeuron(innov, b.Generation);
                    resultB.Add(placeholder);
                }
            }

            alignedA.Neurons = resultA;
            alignedB.Neurons = resultB;

            var allSynapseInnovations = new SortedSet<long>();
            foreach (var s in a.Synapses)
                allSynapseInnovations.Add(s.InnovationNumber);
            foreach (var s in b.Synapses)
                allSynapseInnovations.Add(s.InnovationNumber);

            var resultSynA = new List<GeoSynapse>();
            var resultSynB = new List<GeoSynapse>();

            foreach (var innov in allSynapseInnovations)
            {
                if (aSynapseMap.TryGetValue(innov, out var sA))
                    resultSynA.Add(sA.Clone());
                else
                {
                    var placeholder = CreatePlaceholderSynapse(innov, a.Generation);
                    resultSynA.Add(placeholder);
                }

                if (bSynapseMap.TryGetValue(innov, out var sB))
                    resultSynB.Add(sB.Clone());
                else
                {
                    var placeholder = CreatePlaceholderSynapse(innov, b.Generation);
                    resultSynB.Add(placeholder);
                }
            }

            alignedA.Synapses = resultSynA;
            alignedB.Synapses = resultSynB;

            return (alignedA, alignedB);
        }

        private static GeoNeuron CreatePlaceholderNeuron(long innovationNumber, int generation)
        {
            return new GeoNeuron
            {
                InnovationNumber = innovationNumber,
                LayerIndex = -1,
                PositionInLayer = -1,
                Activation = ActivationFunction.Linear,
                Bias = 0,
                IsActive = false,
                CreationGeneration = generation
            };
        }

        private static GeoSynapse CreatePlaceholderSynapse(long innovationNumber, int generation)
        {
            return new GeoSynapse
            {
                InnovationNumber = innovationNumber,
                SourceNeuronId = -1,
                TargetNeuronId = -1,
                Weight = 0,
                IsActive = false,
                CreationGeneration = generation
            };
        }

        /// <summary>
        /// Gets the crossover success rate.
        /// </summary>
        public double GetSuccessRate()
        {
            int total = Volatile.Read(ref _crossoverCount);
            int success = Volatile.Read(ref _successCount);
            return total > 0 ? (double)success / total : 0;
        }
    }

    /// <summary>
    /// Weight-based crossover strategy that focuses on recombining weight values
    /// without structural changes. Supports uniform, blend, line, and arithmetic
    /// mean crossover methods.
    /// </summary>
    public sealed class WeightCrossoverStrategy : ICrossoverStrategy
    {
        private readonly EvolutionConfig _config;
        private readonly WeightCrossoverMethod _method;
        private int _crossoverCount;
        private int _successCount;

        /// <summary>
        /// Initializes a new instance of the WeightCrossoverStrategy class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="method">The weight crossover method to use.</param>
        public WeightCrossoverStrategy(EvolutionConfig config, WeightCrossoverMethod method = WeightCrossoverMethod.Blend)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _method = method;
        }

        /// <inheritdoc/>
        public CrossoverResult Crossover(GeoGenome parentA, GeoGenome parentB, float blendBias)
        {
            Interlocked.Increment(ref _crossoverCount);

            if (parentA == null)
                throw new ArgumentNullException(nameof(parentA));
            if (parentB == null)
                throw new ArgumentNullException(nameof(parentB));

            var offspring = parentA.Clone();
            offspring.Id = Guid.NewGuid();
            offspring.ParentIds = ImmutableArray.Create(parentA.Id, parentB.BestFitnessGeneration > 0 ? parentA.Id : parentB.Id);
            offspring.Generation = Math.Max(parentA.Generation, parentB.Generation) + 1;
            offspring.Fitness = double.NaN;
            offspring.InvalidateFitness();

            var rng = new Random(Guid.NewGuid().GetHashCode());
            var bSynapseMap = parentB.Synapses.ToDictionary(s => s.InnovationNumber);
            int matchingGenes = 0;
            double totalWeightDiff = 0;

            foreach (var synapse in offspring.Synapses)
            {
                if (bSynapseMap.TryGetValue(synapse.InnovationNumber, out var bSynapse))
                {
                    matchingGenes++;
                    double wA = synapse.Weight;
                    double wB = bSynapse.Weight;
                    totalWeightDiff += Math.Abs(wA - wB);

                    synapse.Weight = _method switch
                    {
                        WeightCrossoverMethod.Uniform => rng.NextDouble() < 0.5 ? wA : wB,
                        WeightCrossoverMethod.Blend => wA * (1.0 - blendBias) + wB * blendBias,
                        WeightCrossoverMethod.Line => ComputeLineCrossover(wA, wB, rng),
                        WeightCrossoverMethod.ArithmeticMean => (wA + wB) / 2.0,
                        _ => wA * (1.0 - blendBias) + wB * blendBias
                    };

                    synapse.IsActive = synapse.IsActive || bSynapse.IsActive;
                }
            }

            var bNeuronMap = parentB.Neurons.ToDictionary(n => n.InnovationNumber);
            foreach (var neuron in offspring.Neurons)
            {
                if (bNeuronMap.TryGetValue(neuron.InnovationNumber, out var bNeuron))
                {
                    neuron.Bias = neuron.Bias * (1.0 - blendBias) + bNeuron.Bias * blendBias;
                }
            }

            bool success = offspring.ActiveNeuronCount >= parentA.InputCount + parentA.OutputCount;
            if (success)
                Interlocked.Increment(ref _successCount);

            return new CrossoverResult
            {
                Offspring = offspring,
                IsSuccess = success,
                StrategyUsed = nameof(WeightCrossoverStrategy),
                MatchingGenes = matchingGenes,
                DisjointGenes = Math.Max(0, parentA.TotalSynapseCount - matchingGenes),
                AverageWeightDifference = matchingGenes > 0 ? totalWeightDiff / matchingGenes : 0
            };
        }

        /// <inheritdoc/>
        public (GeoGenome alignedA, GeoGenome alignedB) AlignParents(GeoGenome a, GeoGenome b)
        {
            return (a.Clone(), b.Clone());
        }

        private double ComputeLineCrossover(double wA, double wB, Random rng)
        {
            double alpha = rng.NextDouble();
            double child = wA + alpha * (wB - wA);
            double range = Math.Abs(wB - wA);
            double perturbation = (rng.NextDouble() - 0.5) * range * 0.1;
            return child + perturbation;
        }

        /// <summary>
        /// Gets the crossover success rate.
        /// </summary>
        public double GetSuccessRate()
        {
            int total = Volatile.Read(ref _crossoverCount);
            int success = Volatile.Read(ref _successCount);
            return total > 0 ? (double)success / total : 0;
        }
    }

    /// <summary>
    /// Methods for weight crossover.
    /// </summary>
    public enum WeightCrossoverMethod
    {
        /// <summary>Uniform: randomly pick weight from one parent.</summary>
        Uniform,

        /// <summary>Blend: weighted average of parent weights.</summary>
        Blend,

        /// <summary>Line: linear interpolation with perturbation.</summary>
        Line,

        /// <summary>Arithmetic mean: simple average.</summary>
        ArithmeticMean
    }

    /// <summary>
    /// Comprehensive mutation strategy implementing all mutation types for NEAT-G genomes.
    /// Each mutation type is mathematically defined and produces structurally valid offspring.
    /// </summary>
    public sealed class ComprehensiveMutationStrategy : IMutationStrategy
    {
        private readonly EvolutionConfig _config;
        private int _mutationCount;
        private int _successCount;
        private double _currentPerturbationMagnitude;

        /// <summary>
        /// Initializes a new instance of the ComprehensiveMutationStrategy class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public ComprehensiveMutationStrategy(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _currentPerturbationMagnitude = config.PerturbationMagnitude;
        }

        /// <summary>
        /// Gets or sets the current perturbation magnitude (can be adjusted by adaptive scheduler).
        /// </summary>
        public double CurrentPerturbationMagnitude
        {
            get => _currentPerturbationMagnitude;
            set => _currentPerturbationMagnitude = Math.Max(_config.MinPerturbationMagnitude, value);
        }

        /// <inheritdoc/>
        public MutationResult Mutate(GeoGenome genome, MutationRate rates, Random rng)
        {
            Interlocked.Increment(ref _mutationCount);

            if (genome == null)
                throw new ArgumentNullException(nameof(genome));
            if (rates == null)
                throw new ArgumentNullException(nameof(rates));
            if (rng == null)
                throw new ArgumentNullException(nameof(rng));

            var mutated = genome.Clone();
            mutated.InvalidateFitness();

            MutationType appliedType = MutationType.None;
            int structuralChanges = 0;
            var descriptions = new List<string>();

            if (rng.NextDouble() < rates.GetRate(MutationType.PointMutation))
            {
                ApplyPointMutation(mutated, rng);
                appliedType |= MutationType.PointMutation;
                descriptions.Add("PointMutation");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.BiasDrift))
            {
                ApplyBiasDrift(mutated, rng);
                appliedType |= MutationType.BiasDrift;
                descriptions.Add("BiasDrift");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.WeightPerturbation))
            {
                ApplyWeightPerturbation(mutated, rng);
                appliedType |= MutationType.WeightPerturbation;
                descriptions.Add("WeightPerturbation");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.ActivationShift))
            {
                ApplyActivationShift(mutated, rng);
                appliedType |= MutationType.ActivationShift;
                descriptions.Add("ActivationShift");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.SynapseGrowth))
            {
                structuralChanges += ApplySynapseGrowth(mutated, rng);
                appliedType |= MutationType.SynapseGrowth;
                descriptions.Add($"SynapseGrowth({structuralChanges})");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.SynapsePruning))
            {
                structuralChanges += ApplySynapsePruning(mutated, rng);
                appliedType |= MutationType.SynapsePruning;
                descriptions.Add("SynapsePruning");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.InsertionMutation))
            {
                structuralChanges += ApplyInsertionMutation(mutated, rng);
                appliedType |= MutationType.InsertionMutation;
                descriptions.Add("InsertionMutation");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.DeletionMutation))
            {
                structuralChanges += ApplyDeletionMutation(mutated, rng);
                appliedType |= MutationType.DeletionMutation;
                descriptions.Add("DeletionMutation");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.DuplicationMutation))
            {
                structuralChanges += ApplyDuplicationMutation(mutated, rng);
                appliedType |= MutationType.DuplicationMutation;
                descriptions.Add("DuplicationMutation");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.InversionMutation))
            {
                structuralChanges += ApplyInversionMutation(mutated, rng);
                appliedType |= MutationType.InversionMutation;
                descriptions.Add("InversionMutation");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.TranslocationMutation))
            {
                structuralChanges += ApplyTranslocationMutation(mutated, rng);
                appliedType |= MutationType.TranslocationMutation;
                descriptions.Add("TranslocationMutation");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.SemanticMutation))
            {
                ApplySemanticMutation(mutated, rng);
                appliedType |= MutationType.SemanticMutation;
                descriptions.Add("SemanticMutation");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.TopologyMutation))
            {
                structuralChanges += ApplyTopologyMutation(mutated, rng);
                appliedType |= MutationType.TopologyMutation;
                descriptions.Add("TopologyMutation");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.GeneSilencing))
            {
                structuralChanges += ApplyGeneSilencing(mutated, rng);
                appliedType |= MutationType.GeneSilencing;
                descriptions.Add("GeneSilencing");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.GeneActivation))
            {
                structuralChanges += ApplyGeneActivation(mutated, rng);
                appliedType |= MutationType.GeneActivation;
                descriptions.Add("GeneActivation");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.LayerInsertion))
            {
                structuralChanges += ApplyLayerInsertion(mutated, rng);
                appliedType |= MutationType.LayerInsertion;
                descriptions.Add("LayerInsertion");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.LayerRemoval))
            {
                structuralChanges += ApplyLayerRemoval(mutated, rng);
                appliedType |= MutationType.LayerRemoval;
                descriptions.Add("LayerRemoval");
            }

            if (rng.NextDouble() < rates.GetRate(MutationType.RegulatoryMutation))
            {
                ApplyRegulatoryMutation(mutated, rng);
                appliedType |= MutationType.RegulatoryMutation;
                descriptions.Add("RegulatoryMutation");
            }

            bool hasChanges = appliedType != MutationType.None;
            if (hasChanges)
            {
                mutated.ComputeComplexity();
                Interlocked.Increment(ref _successCount);
            }

            return new MutationResult
            {
                MutatedGenome = mutated,
                IsSuccess = hasChanges,
                TypeApplied = appliedType,
                StructuralChanges = structuralChanges,
                Description = string.Join(", ", descriptions)
            };
        }

        /// <inheritdoc/>
        public double GetSuccessRate()
        {
            int total = Volatile.Read(ref _mutationCount);
            int success = Volatile.Read(ref _successCount);
            return total > 0 ? (double)success / total : 0;
        }

        /// <summary>
        /// Applies point mutation: randomly modifies a single weight value.
        /// </summary>
        private void ApplyPointMutation(GeoGenome genome, Random rng)
        {
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();
            if (activeSynapses.Count == 0)
                return;

            var synapse = activeSynapses[rng.Next(activeSynapses.Count)];
            double perturbation = (rng.NextDouble() * 2.0 - 1.0) * _currentPerturbationMagnitude;
            synapse.Weight += perturbation;
            synapse.Weight = Math.Clamp(synapse.Weight, -10.0, 10.0);
        }

        /// <summary>
        /// Applies bias drift: random walk perturbation of bias values.
        /// </summary>
        private void ApplyBiasDrift(GeoGenome genome, Random rng)
        {
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            if (activeNeurons.Count == 0)
                return;

            int count = Math.Max(1, activeNeurons.Count / 10);
            for (int i = 0; i < count; i++)
            {
                var neuron = activeNeurons[rng.Next(activeNeurons.Count)];
                double drift = (rng.NextDouble() * 2.0 - 1.0) * _currentPerturbationMagnitude * 0.5;
                neuron.Bias += drift;
                neuron.Bias = Math.Clamp(neuron.Bias, -5.0, 5.0);
            }
        }

        /// <summary>
        /// Applies weight perturbation: Gaussian perturbation of weight values.
        /// With small probability, performs uniform perturbation of all weights.
        /// </summary>
        private void ApplyWeightPerturbation(GeoGenome genome, Random rng)
        {
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();
            if (activeSynapses.Count == 0)
                return;

            if (rng.NextDouble() < _config.UniformPerturbationProbability)
            {
                foreach (var synapse in activeSynapses)
                {
                    double noise = (rng.NextDouble() * 2.0 - 1.0) * _currentPerturbationMagnitude;
                    synapse.Weight += noise;
                    synapse.Weight = Math.Clamp(synapse.Weight, -10.0, 10.0);
                }
            }
            else
            {
                int perturbCount = Math.Max(1, (int)(activeSynapses.Count * 0.3));
                for (int i = 0; i < perturbCount; i++)
                {
                    var synapse = activeSynapses[rng.Next(activeSynapses.Count)];
                    double u1 = rng.NextDouble();
                    double u2 = rng.NextDouble();
                    double z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
                    double noise = z * _currentPerturbationMagnitude;
                    synapse.Weight += noise;
                    synapse.Weight = Math.Clamp(synapse.Weight, -10.0, 10.0);
                }
            }
        }

        /// <summary>
        /// Applies activation shift: changes the activation function of a random neuron.
        /// </summary>
        private void ApplyActivationShift(GeoGenome genome, Random rng)
        {
            var hiddenNeurons = genome.Neurons
                .Where(n => n.IsActive && n.LayerIndex > 0 && n.LayerIndex < genome.MaxLayerDepth)
                .ToList();
            if (hiddenNeurons.Count == 0)
                return;

            var neuron = hiddenNeurons[rng.Next(hiddenNeurons.Count)];
            var allActivations = Enum.GetValues<ActivationFunction>();
            neuron.Activation = allActivations[rng.Next(allActivations.Length)];
        }

        /// <summary>
        /// Applies synapse growth: creates a new connection between existing neurons.
        /// Ensures no cycles are created and no duplicate connections exist.
        /// </summary>
        private int ApplySynapseGrowth(GeoGenome genome, Random rng)
        {
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            if (activeNeurons.Count < 2)
                return 0;

            long maxInnovation = genome.Synapses.Count > 0
                ? genome.Synapses.Max(s => s.InnovationNumber)
                : 0;

            for (int attempt = 0; attempt < 10; attempt++)
            {
                var source = activeNeurons[rng.Next(activeNeurons.Count)];
                var target = activeNeurons[rng.Next(activeNeurons.Count)];

                if (source.InnovationNumber == target.InnovationNumber)
                    continue;
                if (source.LayerIndex >= target.LayerIndex && !genome.Synapses.Any(s => s.IsRecurrent))
                    continue;

                bool exists = genome.Synapses.Any(s =>
                    s.SourceNeuronId == source.InnovationNumber &&
                    s.TargetNeuronId == target.InnovationNumber);
                if (exists)
                    continue;

                if (!genome.WouldCreateCycle(source.InnovationNumber, target.InnovationNumber))
                {
                    var newSynapse = new GeoSynapse
                    {
                        InnovationNumber = maxInnovation + 1,
                        SourceNeuronId = source.InnovationNumber,
                        TargetNeuronId = target.InnovationNumber,
                        Weight = (rng.NextDouble() * 2.0 - 1.0) * _config.WeightInitRange,
                        IsActive = true,
                        IsRecurrent = source.LayerIndex >= target.LayerIndex,
                        CreationGeneration = genome.Generation
                    };
                    genome.Synapses.Add(newSynapse);
                    return 1;
                }
            }
            return 0;
        }

        /// <summary>
        /// Applies synapse pruning: removes low-magnitude connections.
        /// Uses soft thresholding based on weight magnitude.
        /// </summary>
        private int ApplySynapsePruning(GeoGenome genome, Random rng)
        {
            var activeSynapses = genome.Synapses
                .Where(s => s.IsActive)
                .OrderBy(s => Math.Abs(s.Weight))
                .ToList();

            if (activeSynapses.Count <= genome.InputCount + genome.OutputCount)
                return 0;

            double threshold = activeSynapses.Count > 0
                ? activeSynapses.Average(s => Math.Abs(s.Weight)) * 0.3
                : 0;

            int pruned = 0;
            foreach (var synapse in activeSynapses)
            {
                if (Math.Abs(synapse.Weight) < threshold && pruned < 3)
                {
                    synapse.IsActive = false;
                    pruned++;
                }
            }
            return pruned;
        }

        /// <summary>
        /// Applies insertion mutation: inserts a new node in the middle of an existing connection.
        /// The original connection is split, and the new node's activation is randomly assigned.
        /// </summary>
        private int ApplyInsertionMutation(GeoGenome genome, Random rng)
        {
            var activeSynapses = genome.Synapses
                .Where(s => s.IsActive && !s.IsRecurrent)
                .ToList();
            if (activeSynapses.Count == 0)
                return 0;

            var synapse = activeSynapses[rng.Next(activeSynapses.Count)];
            synapse.IsActive = false;

            long maxNeuronInnov = genome.Neurons.Count > 0
                ? genome.Neurons.Max(n => n.InnovationNumber)
                : 0;
            long maxSynapseInnov = genome.Synapses.Count > 0
                ? genome.Synapses.Max(s => s.InnovationNumber)
                : 0;

            var sourceNeuron = genome.Neurons.FirstOrDefault(n => n.InnovationNumber == synapse.SourceNeuronId);
            var targetNeuron = genome.Neurons.FirstOrDefault(n => n.InnovationNumber == synapse.TargetNeuronId);
            int newLayer = sourceNeuron != null && targetNeuron != null
                ? (sourceNeuron.LayerIndex + targetNeuron.LayerIndex) / 2 + 1
                : 1;

            var allActivations = Enum.GetValues<ActivationFunction>();
            var newNeuron = new GeoNeuron
            {
                InnovationNumber = maxNeuronInnov + 1,
                LayerIndex = newLayer,
                PositionInLayer = 0,
                Activation = allActivations[rng.Next(allActivations.Length)],
                Bias = (rng.NextDouble() * 2.0 - 1.0) * _config.BiasInitRange,
                IsActive = true,
                CreationGeneration = genome.Generation
            };
            genome.Neurons.Add(newNeuron);

            var synapseToSource = new GeoSynapse
            {
                InnovationNumber = maxSynapseInnov + 1,
                SourceNeuronId = synapse.SourceNeuronId,
                TargetNeuronId = newNeuron.InnovationNumber,
                Weight = 1.0,
                IsActive = true,
                CreationGeneration = genome.Generation
            };

            var synapseToTarget = new GeoSynapse
            {
                InnovationNumber = maxSynapseInnov + 2,
                SourceNeuronId = newNeuron.InnovationNumber,
                TargetNeuronId = synapse.TargetNeuronId,
                Weight = synapse.Weight,
                IsActive = true,
                CreationGeneration = genome.Generation
            };

            genome.Synapses.Add(synapseToSource);
            genome.Synapses.Add(synapseToTarget);

            return 1;
        }

        /// <summary>
        /// Applies deletion mutation: removes a node and reconnects its inputs to its outputs.
        /// Preserves overall signal flow by creating bypass connections.
        /// </summary>
        private int ApplyDeletionMutation(GeoGenome genome, Random rng)
        {
            var deletableNeurons = genome.Neurons
                .Where(n => n.IsActive && n.LayerIndex > 0 && n.LayerIndex < genome.MaxLayerDepth)
                .ToList();
            if (deletableNeurons.Count == 0)
                return 0;

            var target = deletableNeurons[rng.Next(deletableNeurons.Count)];
            var inputSynapses = genome.Synapses
                .Where(s => s.IsActive && s.TargetNeuronId == target.InnovationNumber)
                .ToList();
            var outputSynapses = genome.Synapses
                .Where(s => s.IsActive && s.SourceNeuronId == target.InnovationNumber)
                .ToList();

            if (inputSynapses.Count == 0 || outputSynapses.Count == 0)
            {
                target.IsActive = false;
                foreach (var s in genome.Synapses.Where(s =>
                    s.SourceNeuronId == target.InnovationNumber ||
                    s.TargetNeuronId == target.InnovationNumber))
                    s.IsActive = false;
                return 1;
            }

            long maxInnov = genome.Synapses.Count > 0
                ? genome.Synapses.Max(s => s.InnovationNumber)
                : 0;

            int newConnections = 0;
            foreach (var inputSyn in inputSynapses)
            {
                foreach (var outputSyn in outputSynapses)
                {
                    bool exists = genome.Synapses.Any(s =>
                        s.IsActive &&
                        s.SourceNeuronId == inputSyn.SourceNeuronId &&
                        s.TargetNeuronId == outputSyn.TargetNeuronId);
                    if (exists)
                        continue;

                    if (!genome.WouldCreateCycle(inputSyn.SourceNeuronId, outputSyn.TargetNeuronId))
                    {
                        var bypass = new GeoSynapse
                        {
                            InnovationNumber = maxInnov + 1 + newConnections,
                            SourceNeuronId = inputSyn.SourceNeuronId,
                            TargetNeuronId = outputSyn.TargetNeuronId,
                            Weight = inputSyn.Weight * outputSyn.Weight,
                            IsActive = true,
                            CreationGeneration = genome.Generation
                        };
                        genome.Synapses.Add(bypass);
                        newConnections++;
                    }
                }
            }

            target.IsActive = false;
            foreach (var s in genome.Synapses.Where(s =>
                s.SourceNeuronId == target.InnovationNumber ||
                s.TargetNeuronId == target.InnovationNumber))
                s.IsActive = false;

            return 1;
        }

        /// <summary>
        /// Applies duplication mutation: duplicates a random neuron along with its connection pattern.
        /// </summary>
        private int ApplyDuplicationMutation(GeoGenome genome, Random rng)
        {
            var duplicatable = genome.Neurons
                .Where(n => n.IsActive && n.LayerIndex > 0)
                .ToList();
            if (duplicatable.Count == 0)
                return 0;

            var source = duplicatable[rng.Next(duplicatable.Count)];
            long maxNeuronInnov = genome.Neurons.Max(n => n.InnovationNumber);
            long maxSynapseInnov = genome.Synapses.Count > 0
                ? genome.Synapses.Max(s => s.InnovationNumber)
                : 0;

            var duplicate = source.Clone();
            duplicate.InnovationNumber = maxNeuronInnov + 1;
            duplicate.PositionInLayer = source.PositionInLayer + 1;
            duplicate.Bias = source.Bias * (1.0 + (rng.NextDouble() * 0.2 - 0.1));
            duplicate.CreationGeneration = genome.Generation;
            genome.Neurons.Add(duplicate);

            var inputSynapses = genome.Synapses
                .Where(s => s.IsActive && s.TargetNeuronId == source.InnovationNumber)
                .ToList();
            var outputSynapses = genome.Synapses
                .Where(s => s.IsActive && s.SourceNeuronId == source.InnovationNumber)
                .ToList();

            int newSynapses = 0;
            foreach (var inputSyn in inputSynapses)
            {
                var newInput = new GeoSynapse
                {
                    InnovationNumber = maxSynapseInnov + 1 + newSynapses,
                    SourceNeuronId = inputSyn.SourceNeuronId,
                    TargetNeuronId = duplicate.InnovationNumber,
                    Weight = inputSyn.Weight * (1.0 + (rng.NextDouble() * 0.2 - 0.1)),
                    IsActive = true,
                    CreationGeneration = genome.Generation
                };
                genome.Synapses.Add(newInput);
                newSynapses++;
            }

            foreach (var outputSyn in outputSynapses)
            {
                var newOutput = new GeoSynapse
                {
                    InnovationNumber = maxSynapseInnov + 1 + newSynapses,
                    SourceNeuronId = duplicate.InnovationNumber,
                    TargetNeuronId = outputSyn.TargetNeuronId,
                    Weight = outputSyn.Weight * (1.0 + (rng.NextDouble() * 0.2 - 0.1)),
                    IsActive = true,
                    CreationGeneration = genome.Generation
                };
                genome.Synapses.Add(newOutput);
                newSynapses++;
            }

            return 1;
        }

        /// <summary>
        /// Applies inversion mutation: reverses the order of synapses in a randomly selected segment.
        /// </summary>
        private int ApplyInversionMutation(GeoGenome genome, Random rng)
        {
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();
            if (activeSynapses.Count < 3)
                return 0;

            int start = rng.Next(activeSynapses.Count - 2);
            int end = rng.Next(start + 2, activeSynapses.Count);
            int length = end - start;

            for (int i = 0; i < length / 2; i++)
            {
                var temp = activeSynapses[start + i];
                activeSynapses[start + i] = activeSynapses[end - 1 - i];
                activeSynapses[end - 1 - i] = temp;
            }

            return 1;
        }

        /// <summary>
        /// Applies translocation mutation: moves a segment of synapses to a different position.
        /// </summary>
        private int ApplyTranslocationMutation(GeoGenome genome, Random rng)
        {
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();
            if (activeSynapses.Count < 4)
                return 0;

            int segStart = rng.Next(activeSynapses.Count / 2);
            int segLen = rng.Next(2, Math.Max(3, activeSynapses.Count / 4));
            segLen = Math.Min(segLen, activeSynapses.Count - segStart);
            int insertPoint = rng.Next(activeSynapses.Count - segLen + 1);

            var segment = activeSynapses.GetRange(segStart, segLen);
            activeSynapses.RemoveRange(segStart, segLen);
            activeSynapses.InsertRange(insertPoint, segment);

            return 1;
        }

        /// <summary>
        /// Applies semantic mutation: modifies neuron activation and bias based on semantic role.
        /// Neurons with similar semantic roles are mutated in a correlated manner.
        /// </summary>
        private void ApplySemanticMutation(GeoGenome genome, Random rng)
        {
            var roleGroups = genome.Neurons
                .Where(n => n.IsActive && !string.IsNullOrEmpty(n.SemanticRole))
                .GroupBy(n => n.SemanticRole)
                .Where(g => g.Count() > 1)
                .ToList();

            if (roleGroups.Count == 0)
            {
                var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
                if (activeNeurons.Count == 0)
                    return;
                var neuron = activeNeurons[rng.Next(activeNeurons.Count)];
                double shift = (rng.NextDouble() * 2.0 - 1.0) * _currentPerturbationMagnitude;
                neuron.Bias += shift;
                return;
            }

            var group = roleGroups[rng.Next(roleGroups.Count)];
            double correlatedShift = (rng.NextDouble() * 2.0 - 1.0) * _currentPerturbationMagnitude;
            foreach (var neuron in group)
            {
                double noise = correlatedShift + (rng.NextDouble() * 2.0 - 1.0) * _currentPerturbationMagnitude * 0.3;
                neuron.Bias += noise;
                neuron.Bias = Math.Clamp(neuron.Bias, -5.0, 5.0);
            }
        }

        /// <summary>
        /// Applies topology mutation: adds or removes connections between existing neurons.
        /// </summary>
        private int ApplyTopologyMutation(GeoGenome genome, Random rng)
        {
            if (rng.NextDouble() < 0.6)
            {
                return ApplySynapseGrowth(genome, rng);
            }
            else
            {
                return ApplySynapsePruning(genome, rng);
            }
        }

        /// <summary>
        /// Applies gene silencing: deactivates a random neuron without removing it.
        /// </summary>
        private int ApplyGeneSilencing(GeoGenome genome, Random rng)
        {
            var activeNeurons = genome.Neurons
                .Where(n => n.IsActive && n.LayerIndex > 0 && n.LayerIndex < genome.MaxLayerDepth)
                .ToList();
            if (activeNeurons.Count == 0)
                return 0;

            var neuron = activeNeurons[rng.Next(activeNeurons.Count)];
            neuron.IsActive = false;
            return 1;
        }

        /// <summary>
        /// Applies gene activation: reactivates a previously silenced neuron.
        /// </summary>
        private int ApplyGeneActivation(GeoGenome genome, Random rng)
        {
            var inactiveNeurons = genome.Neurons
                .Where(n => !n.IsActive)
                .ToList();
            if (inactiveNeurons.Count == 0)
                return 0;

            var neuron = inactiveNeurons[rng.Next(inactiveNeurons.Count)];
            neuron.IsActive = true;
            return 1;
        }

        /// <summary>
        /// Applies layer insertion: inserts a new hidden layer between two existing layers.
        /// All connections crossing the new layer boundary are rerouted through the new layer.
        /// </summary>
        private int ApplyLayerInsertion(GeoGenome genome, Random rng)
        {
            int maxLayer = genome.Neurons.Count > 0
                ? genome.Neurons.Where(n => n.IsActive).Max(n => n.LayerIndex)
                : 0;
            int minLayer = genome.Neurons.Count > 0
                ? genome.Neurons.Where(n => n.IsActive).Min(n => n.LayerIndex)
                : 0;

            if (maxLayer - minLayer < 1)
                return 0;

            int insertAfter = rng.Next(minLayer, maxLayer);
            int newLayer = insertAfter + 1;

            foreach (var neuron in genome.Neurons.Where(n => n.IsActive && n.LayerIndex > insertAfter))
            {
                neuron.LayerIndex++;
            }

            long maxNeuronInnov = genome.Neurons.Count > 0
                ? genome.Neurons.Max(n => n.InnovationNumber)
                : 0;
            long maxSynapseInnov = genome.Synapses.Count > 0
                ? genome.Synapses.Max(s => s.InnovationNumber)
                : 0;

            var crossingSynapses = genome.Synapses
                .Where(s => s.IsActive && !s.IsRecurrent)
                .ToList();

            int newNeurons = 0;
            int newSynapses = 0;
            var allActivations = Enum.GetValues<ActivationFunction>();

            foreach (var synapse in crossingSynapses)
            {
                var srcNeuron = genome.Neurons.FirstOrDefault(n => n.InnovationNumber == synapse.SourceNeuronId);
                var tgtNeuron = genome.Neurons.FirstOrDefault(n => n.InnovationNumber == synapse.TargetNeuronId);

                if (srcNeuron != null && tgtNeuron != null &&
                    srcNeuron.LayerIndex <= insertAfter && tgtNeuron.LayerIndex > newLayer)
                {
                    long newId = maxNeuronInnov + 1 + newNeurons;
                    var intermediate = new GeoNeuron
                    {
                        InnovationNumber = newId,
                        LayerIndex = newLayer,
                        PositionInLayer = newNeurons,
                        Activation = allActivations[rng.Next(allActivations.Length)],
                        Bias = (rng.NextDouble() * 2.0 - 1.0) * _config.BiasInitRange,
                        IsActive = true,
                        CreationGeneration = genome.Generation
                    };
                    genome.Neurons.Add(intermediate);
                    newNeurons++;

                    synapse.TargetNeuronId = newId;
                    synapse.Weight = 1.0;

                    var newSynapse = new GeoSynapse
                    {
                        InnovationNumber = maxSynapseInnov + 1 + newSynapses,
                        SourceNeuronId = newId,
                        TargetNeuronId = tgtNeuron.InnovationNumber,
                        Weight = synapse.Weight,
                        IsActive = true,
                        CreationGeneration = genome.Generation
                    };
                    genome.Synapses.Add(newSynapse);
                    newSynapses++;
                }
            }

            return newNeurons > 0 ? 1 : 0;
        }

        /// <summary>
        /// Applies layer removal: removes a randomly selected hidden layer
        /// and reconnects its inputs to its outputs.
        /// </summary>
        private int ApplyLayerRemoval(GeoGenome genome, Random rng)
        {
            var layerGroups = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.LayerIndex)
                .Where(g => g.Key > 0)
                .ToList();

            if (layerGroups.Count <= 1)
                return 0;

            var targetLayer = layerGroups[rng.Next(layerGroups.Count)];
            int removedCount = 0;

            foreach (var neuron in targetLayer)
            {
                neuron.IsActive = false;
                removedCount++;
            }

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                var src = genome.Neurons.FirstOrDefault(n => n.InnovationNumber == synapse.SourceNeuronId);
                var tgt = genome.Neurons.FirstOrDefault(n => n.InnovationNumber == synapse.TargetNeuronId);

                if ((src != null && !src.IsActive) || (tgt != null && !tgt.IsActive))
                {
                    synapse.IsActive = false;
                }
            }

            return removedCount > 0 ? 1 : 0;
        }

        /// <summary>
        /// Applies regulatory mutation: modifies gene regulatory interactions
        /// by adjusting confidence scores and expression counts.
        /// </summary>
        private void ApplyRegulatoryMutation(GeoGenome genome, Random rng)
        {
            foreach (var neuron in genome.Neurons.Where(n => n.IsActive))
            {
                if (rng.NextDouble() < 0.1)
                {
                    double shift = (rng.NextDouble() * 2.0 - 1.0) * 0.1;
                    neuron.ExpressionCount = Math.Max(1, neuron.ExpressionCount + (int)(shift * 10));
                }
            }

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (rng.NextDouble() < 0.1)
                {
                    double perturbation = (rng.NextDouble() * 2.0 - 1.0) * 0.05;
                    synapse.Confidence = Math.Clamp(synapse.Confidence + perturbation, 0.0, 1.0);
                }
            }
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Adaptive Mutation Scheduler

    /// <summary>
    /// Dynamically adjusts mutation rates based on population fitness trends.
    /// When the population is stagnant, mutation rates are increased to encourage exploration.
    /// When the population is improving, mutation rates are decreased to allow exploitation.
    /// Uses a sliding window of fitness improvements to detect trends.
    /// </summary>
    public sealed class AdaptiveMutationScheduler
    {
        private readonly EvolutionConfig _config;
        private readonly MutationRate _baseRates;
        private readonly Queue<double> _fitnessImprovements;
        private readonly Queue<double> _diversityHistory;
        private double _currentMultiplier;
        private double _currentPerturbationMagnitude;
        private int _generationsSinceAdjustment;

        /// <summary>
        /// Initializes a new instance of the AdaptiveMutationScheduler class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="baseRates">Base mutation rates to scale from.</param>
        public AdaptiveMutationScheduler(EvolutionConfig config, MutationRate baseRates)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _baseRates = baseRates ?? throw new ArgumentNullException(nameof(baseRates));
            _fitnessImprovements = new Queue<double>(config.AdaptiveWindow + 1);
            _diversityHistory = new Queue<double>(config.AdaptiveWindow + 1);
            _currentMultiplier = 1.0;
            _currentPerturbationMagnitude = config.PerturbationMagnitude;
        }

        /// <summary>Gets the current adaptive mutation rate multiplier.</summary>
        public double CurrentMultiplier => _currentMultiplier;

        /// <summary>Gets the current perturbation magnitude.</summary>
        public double CurrentPerturbationMagnitude => _currentPerturbationMagnitude;

        /// <summary>
        /// Adjusts mutation rates based on the latest generation's fitness improvement and diversity.
        /// </summary>
        /// <param name="fitnessImprovement">The fitness improvement over the previous generation.</param>
        /// <param name="diversityMetric">The current population diversity metric (0-1).</param>
        /// <param name="mutationSuccessRate">The success rate of mutations in the last generation.</param>
        /// <returns>The adjusted mutation rates.</returns>
        public MutationRate Adjust(double fitnessImprovement, double diversityMetric, double mutationSuccessRate)
        {
            _fitnessImprovements.Enqueue(fitnessImprovement);
            _diversityHistory.Enqueue(diversityMetric);

            while (_fitnessImprovements.Count > _config.AdaptiveWindow)
                _fitnessImprovements.Dequeue();
            while (_diversityHistory.Count > _config.AdaptiveWindow)
                _diversityHistory.Dequeue();

            _generationsSinceAdjustment++;

            if (_generationsSinceAdjustment < 3)
                return CreateScaledRates();

            double avgImprovement = _fitnessImprovements.Count > 0
                ? _fitnessImprovements.Average()
                : 0;

            double avgDiversity = _diversityHistory.Count > 0
                ? _diversityHistory.Average()
                : 0;

            double targetMultiplier = ComputeTargetMultiplier(avgImprovement, avgDiversity, mutationSuccessRate);

            double smoothingFactor = 0.3;
            _currentMultiplier = _currentMultiplier * (1.0 - smoothingFactor) + targetMultiplier * smoothingFactor;
            _currentMultiplier = Math.Clamp(_currentMultiplier,
                _config.MinAdaptiveMutationRate / GetBaseAverageRate(),
                _config.MaxAdaptiveMutationRate / GetBaseAverageRate());

            _currentPerturbationMagnitude = _config.PerturbationMagnitude * _currentMultiplier;
            _currentPerturbationMagnitude = Math.Max(_config.MinPerturbationMagnitude, _currentPerturbationMagnitude);

            _generationsSinceAdjustment = 0;

            return CreateScaledRates();
        }

        /// <summary>
        /// Gets the adjusted mutation rates for the current state.
        /// </summary>
        /// <returns>The current scaled mutation rates.</returns>
        public MutationRate GetCurrentRates()
        {
            return CreateScaledRates();
        }

        /// <summary>
        /// Resets the scheduler to initial state.
        /// </summary>
        public void Reset()
        {
            _fitnessImprovements.Clear();
            _diversityHistory.Clear();
            _currentMultiplier = 1.0;
            _currentPerturbationMagnitude = _config.PerturbationMagnitude;
            _generationsSinceAdjustment = 0;
        }

        private double ComputeTargetMultiplier(double avgImprovement, double avgDiversity, double mutationSuccessRate)
        {
            double improvementScore = 0;
            if (Math.Abs(avgImprovement) < _config.FitnessThreshold)
            {
                improvementScore = 1.0;
            }
            else if (avgImprovement < 0)
            {
                improvementScore = 1.5;
            }
            else
            {
                improvementScore = Math.Max(0, 1.0 - avgImprovement * 10.0);
            }

            double diversityScore = 0;
            if (avgDiversity < 0.1)
            {
                diversityScore = 1.5;
            }
            else if (avgDiversity < 0.3)
            {
                diversityScore = 1.0;
            }
            else if (avgDiversity > 0.8)
            {
                diversityScore = 0.5;
            }
            else
            {
                diversityScore = 0.8;
            }

            double successScore = 0;
            if (mutationSuccessRate < 0.1)
            {
                successScore = 1.3;
            }
            else if (mutationSuccessRate > 0.8)
            {
                successScore = 0.7;
            }
            else
            {
                successScore = 1.0;
            }

            double combined = 0.4 * improvementScore + 0.35 * diversityScore + 0.25 * successScore;
            return Math.Clamp(combined, 0.3, 3.0);
        }

        private double GetBaseAverageRate()
        {
            return _baseRates.GetTotalRate() / 18.0;
        }

        private MutationRate CreateScaledRates()
        {
            var scaled = _baseRates.Clone();
            scaled.ScaleAll(_currentMultiplier);
            return scaled;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Manifold Speciation Strategy

    /// <summary>
    /// Implements speciation using Gromov-Hausdorff distance approximation
    /// with landmark-based embedding and curvature-aware manifold distance.
    /// This strategy provides more meaningful genome distance measurements than
    /// simple Euclidean distance by accounting for the geometric structure of
    /// the genome embedding space.
    /// </summary>
    public sealed class ManifoldSpeciationStrategy : ISpeciationStrategy
    {
        private readonly EvolutionConfig _config;
        private readonly ConcurrentDictionary<(Guid, Guid), double> _distanceCache;
        private readonly List<GeoGenome> _landmarks;
        private double _currentThreshold;
        private int _nextSpeciesId;

        /// <summary>
        /// Initializes a new instance of the ManifoldSpeciationStrategy class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public ManifoldSpeciationStrategy(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _distanceCache = new ConcurrentDictionary<(Guid, Guid), double>();
            _landmarks = new List<GeoGenome>();
            _currentThreshold = config.SpeciationThreshold;
            _nextSpeciesId = 1;
        }

        /// <summary>Gets the current speciation threshold.</summary>
        public double CurrentThreshold => _currentThreshold;

        /// <inheritdoc/>
        public ImmutableArray<SpeciesInfo> Speciate(GenomePopulation population)
        {
            if (population.Genomes.Length == 0)
                return ImmutableArray<SpeciesInfo>.Empty;

            SelectLandmarks(population.Genomes);

            var speciesMap = new Dictionary<int, List<GeoGenome>>();
            var genomeSpecies = new Dictionary<Guid, int>();
            var speciesRepresentatives = new Dictionary<int, GeoGenome>();

            foreach (var genome in population.Genomes)
            {
                if (genome.SemanticEmbedding.IsDefaultOrEmpty)
                    genome.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);
            }

            bool assigned = false;
            foreach (var existingSpecies in GetExistingSpecies(population))
            {
                if (existingSpecies.Representative == null)
                    continue;

                var representative = existingSpecies.Representative;
                if (representative.SemanticEmbedding.IsDefaultOrEmpty)
                    representative.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

                foreach (var genome in population.Genomes)
                {
                    if (genomeSpecies.ContainsKey(genome.Id))
                        continue;

                    double distance = ComputeDistance(representative, genome);
                    if (distance <= _currentThreshold)
                    {
                        if (!speciesMap.ContainsKey(existingSpecies.Id))
                        {
                            speciesMap[existingSpecies.Id] = new List<GeoGenome>();
                            speciesRepresentatives[existingSpecies.Id] = representative;
                        }
                        speciesMap[existingSpecies.Id].Add(genome);
                        genomeSpecies[genome.Id] = existingSpecies.Id;
                        assigned = true;
                    }
                }
            }

            foreach (var genome in population.Genomes)
            {
                if (genomeSpecies.ContainsKey(genome.Id))
                    continue;

                int speciesId = _nextSpeciesId++;
                speciesMap[speciesId] = new List<GeoGenome> { genome };
                speciesRepresentatives[speciesId] = genome;
                genomeSpecies[genome.Id] = speciesId;
            }

            AdjustThreshold(speciesMap.Count);

            var species = new List<SpeciesInfo>();
            foreach (var kvp in speciesMap)
            {
                var members = kvp.Value;
                double bestFitness = members.Max(g => g.Fitness);
                double avgFitness = members.Average(g => g.Fitness);
                int bestGen = members.Where(g => g.Fitness == bestFitness).First().Generation;

                var fitnessHistory = new List<double>();
                foreach (var member in members)
                {
                    fitnessHistory.Add(member.Fitness);
                }

                species.Add(new SpeciesInfo
                {
                    Id = kvp.Key,
                    Representative = speciesRepresentatives[kvp.Key],
                    RepresentativeGenomeId = speciesRepresentatives[kvp.Key].Id,
                    MemberIds = members.Select(g => g.Id).ToImmutableArray(),
                    BestFitness = bestFitness,
                    BestFitnessGeneration = bestGen,
                    AverageFitness = avgFitness,
                    Age = members.Count > 0 ? members.Max(g => g.Age) : 0,
                    CreationGeneration = members.Count > 0 ? members.Min(g => g.Generation) : 0,
                    FitnessHistory = fitnessHistory.ToImmutableArray()
                });
            }

            return species.ToImmutableArray();
        }

        /// <inheritdoc/>
        public double ComputeDistance(GeoGenome a, GeoGenome b)
        {
            if (a.Id == b.Id)
                return 0;

            var key = a.Id.CompareTo(b.Id) < 0
                ? (a.Id, b.Id)
                : (b.Id, a.Id);

            if (_distanceCache.TryGetValue(key, out var cached))
                return cached;

            double ghDistance = ComputeGromovHausdorffDistance(a, b);
            double manifoldDist = ComputeManifoldDistance(a, b);
            double weightDiff = ComputeWeightDistance(a, b);

            double combined = 0.4 * ghDistance + 0.35 * manifoldDist + 0.25 * weightDiff;

            _distanceCache[key] = combined;
            return combined;
        }

        /// <summary>
        /// Computes the Gromov-Hausdorff distance approximation between two genomes
        /// using landmark-based embedding. This measures the structural dissimilarity
        /// between the two genome topologies.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Approximated Gromov-Hausdorff distance.</returns>
        public double ComputeGromovHausdorffDistance(GeoGenome a, GeoGenome b)
        {
            if (a.SemanticEmbedding.IsDefaultOrEmpty)
                a.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);
            if (b.SemanticEmbedding.IsDefaultOrEmpty)
                b.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

            int dim = Math.Min(a.SemanticEmbedding.Length, b.SemanticEmbedding.Length);
            if (dim == 0)
                return 1.0;

            double maxMinDistAtoB = 0;
            foreach (var landmark in _landmarks)
            {
                if (landmark.SemanticEmbedding.IsDefaultOrEmpty || landmark.SemanticEmbedding.Length < dim)
                    continue;

                double distAtoLandmark = 0;
                double distBtoLandmark = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diffA = a.SemanticEmbedding[d] - landmark.SemanticEmbedding[d];
                    double diffB = b.SemanticEmbedding[d] - landmark.SemanticEmbedding[d];
                    distAtoLandmark += diffA * diffA;
                    distBtoLandmark += diffB * diffB;
                }
                distAtoLandmark = Math.Sqrt(distAtoLandmark);
                distBtoLandmark = Math.Sqrt(distBtoLandmark);

                double minDist = Math.Min(distAtoLandmark, distBtoLandmark);
                maxMinDistAtoB = Math.Max(maxMinDistAtoB, Math.Abs(distAtoLandmark - distBtoLandmark));
            }

            double structuralDist = ComputeStructuralDistance(a, b);
            double combined = 0.6 * maxMinDistAtoB + 0.4 * structuralDist;

            return combined;
        }

        /// <summary>
        /// Computes the manifold-aware distance between two genome embeddings,
        /// taking into account local curvature information.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Manifold-aware distance value.</returns>
        public double ComputeManifoldDistance(GeoGenome a, GeoGenome b)
        {
            if (a.SemanticEmbedding.IsDefaultOrEmpty || b.SemanticEmbedding.IsDefaultOrEmpty)
                return ComputeGromovHausdorffDistance(a, b);

            int dim = Math.Min(a.SemanticEmbedding.Length, b.SemanticEmbedding.Length);
            if (dim == 0)
                return 1.0;

            double euclideanDist = 0;
            double[] direction = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                double diff = a.SemanticEmbedding[d] - b.SemanticEmbedding[d];
                euclideanDist += diff * diff;
                direction[d] = diff;
            }
            euclideanDist = Math.Sqrt(euclideanDist);

            if (euclideanDist < 1e-10)
                return 0;

            double curvatureFactor = ComputeCurvatureFactor(a, b, direction, dim);
            double manifoldDist = euclideanDist * (1.0 + 0.5 * curvatureFactor);

            double geodesicEstimate = EstimateGeodesicDistance(a, b, dim);
            manifoldDist = 0.7 * manifoldDist + 0.3 * geodesicEstimate;

            return manifoldDist;
        }

        private double ComputeCurvatureFactor(GeoGenome a, GeoGenome b, double[] direction, int dim)
        {
            double[] midpoint = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                midpoint[d] = (a.SemanticEmbedding[d] + b.SemanticEmbedding[d]) / 2.0;
            }

            double curvature = 0;
            int resolution = _config.CurvatureResolution;
            for (int i = 0; i < resolution; i++)
            {
                double t = (double)i / (resolution - 1);
                double[] point = new double[dim];
                for (int d = 0; d < dim; d++)
                {
                    point[d] = a.SemanticEmbedding[d] + t * direction[d];
                }

                double localDeviation = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = point[d] - midpoint[d];
                    localDeviation += diff * diff;
                }
                localDeviation = Math.Sqrt(localDeviation);

                double expectedDist = t * Math.Sqrt(direction.Sum(d => d * d));
                curvature += Math.Abs(localDeviation - expectedDist);
            }

            return curvature / resolution;
        }

        private double EstimateGeodesicDistance(GeoGenome a, GeoGenome b, int dim)
        {
            if (_landmarks.Count < 2)
            {
                return ComputeEuclideanDistance(a, b);
            }

            double minDistToA = double.MaxValue;
            double minDistToB = double.MaxValue;
            double landmarkDistAB = 0;

            foreach (var landmark in _landmarks)
            {
                if (landmark.SemanticEmbedding.IsDefaultOrEmpty || landmark.SemanticEmbedding.Length < dim)
                    continue;

                double distToA = 0;
                double distToB = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diffA = a.SemanticEmbedding[d] - landmark.SemanticEmbedding[d];
                    double diffB = b.SemanticEmbedding[d] - landmark.SemanticEmbedding[d];
                    distToA += diffA * diffA;
                    distToB += diffB * diffB;
                }
                distToA = Math.Sqrt(distToA);
                distToB = Math.Sqrt(distToB);

                minDistToA = Math.Min(minDistToA, distToA);
                minDistToB = Math.Min(minDistToB, distToB);
            }

            double directDist = ComputeEuclideanDistance(a, b);
            double landmarkBased = Math.Abs(minDistToA - minDistToB) + 0.5 * (minDistToA + minDistToB);

            return 0.6 * directDist + 0.4 * landmarkBased;
        }

        private double ComputeStructuralDistance(GeoGenome a, GeoGenome b)
        {
            int neuronDiff = Math.Abs(a.ActiveNeuronCount - b.ActiveNeuronCount);
            int synapseDiff = Math.Abs(a.ActiveSynapseCount - b.ActiveSynapseCount);
            int layerDiff = Math.Abs(a.MaxLayerDepth - b.MaxLayerDepth);

            double maxNeurons = Math.Max(a.ActiveNeuronCount, b.ActiveNeuronCount);
            double maxSynapses = Math.Max(a.ActiveSynapseCount, b.ActiveSynapseCount);
            double maxLayers = Math.Max(a.MaxLayerDepth, b.MaxLayerDepth);

            double normNeuronDiff = maxNeurons > 0 ? (double)neuronDiff / maxNeurons : 0;
            double normSynapseDiff = maxSynapses > 0 ? (double)synapseDiff / maxSynapses : 0;
            double normLayerDiff = maxLayers > 0 ? (double)layerDiff / maxLayers : 0;

            double topologyHashDist = ComputeTopologyHashDistance(a, b);

            return 0.3 * normNeuronDiff + 0.3 * normSynapseDiff + 0.2 * normLayerDiff + 0.2 * topologyHashDist;
        }

        private double ComputeTopologyHashDistance(GeoGenome a, GeoGenome b)
        {
            long hashA = a.ComputeTopologyHash();
            long hashB = b.ComputeTopologyHash();

            if (hashA == hashB)
                return 0;

            long xor = hashA ^ hashB;
            int differingBits = 0;
            while (xor != 0)
            {
                differingBits++;
                xor &= xor - 1;
            }

            return Math.Min(1.0, (double)differingBits / 64.0);
        }

        private double ComputeWeightDistance(GeoGenome a, GeoGenome b)
        {
            var aWeights = a.Synapses
                .Where(s => s.IsActive)
                .OrderBy(s => s.InnovationNumber)
                .Select(s => s.Weight)
                .ToList();

            var bWeights = b.Synapses
                .Where(s => s.IsActive)
                .OrderBy(s => s.InnovationNumber)
                .Select(s => s.Weight)
                .ToList();

            if (aWeights.Count == 0 || bWeights.Count == 0)
                return 1.0;

            var commonInnovations = a.Synapses
                .Where(sA => sA.IsActive && b.Synapses.Any(sB => sB.IsActive && sB.InnovationNumber == sA.InnovationNumber))
                .Select(s => s.InnovationNumber)
                .ToList();

            if (commonInnovations.Count == 0)
                return 1.0;

            var aMap = a.Synapses.Where(s => s.IsActive).ToDictionary(s => s.InnovationNumber, s => s.Weight);
            var bMap = b.Synapses.Where(s => s.IsActive).ToDictionary(s => s.InnovationNumber, s => s.Weight);

            double totalDiff = 0;
            foreach (var innov in commonInnovations)
            {
                totalDiff += Math.Abs(aMap[innov] - bMap[innov]);
            }

            double avgDiff = totalDiff / commonInnovations.Count;
            double disjointFraction = 1.0 - (double)commonInnovations.Count /
                Math.Max(a.ActiveSynapseCount, b.ActiveSynapseCount);

            return 0.5 * Math.Min(1.0, avgDiff) + 0.5 * disjointFraction;
        }

        private double ComputeEuclideanDistance(GeoGenome a, GeoGenome b)
        {
            int dim = Math.Min(a.SemanticEmbedding.Length, b.SemanticEmbedding.Length);
            double dist = 0;
            for (int d = 0; d < dim; d++)
            {
                double diff = a.SemanticEmbedding[d] - b.SemanticEmbedding[d];
                dist += diff * diff;
            }
            return Math.Sqrt(dist);
        }

        private void SelectLandmarks(ImmutableArray<GeoGenome> genomes)
        {
            _landmarks.Clear();
            if (genomes.Length == 0)
                return;

            int landmarkCount = Math.Min(_config.LandmarkCount, genomes.Length);

            var rng = new Random(42);
            var shuffled = genomes.ToArray();
            for (int i = shuffled.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
            }

            _landmarks.Add(shuffled[0]);

            for (int l = 1; l < landmarkCount; l++)
            {
                GeoGenome? bestCandidate = null;
                double maxMinDist = -1;

                foreach (var candidate in shuffled)
                {
                    if (_landmarks.Any(lm => lm.Id == candidate.Id))
                        continue;

                    double minDist = double.MaxValue;
                    foreach (var landmark in _landmarks)
                    {
                        double dist = ComputeEuclideanDistance(candidate, landmark);
                        minDist = Math.Min(minDist, dist);
                    }

                    if (minDist > maxMinDist)
                    {
                        maxMinDist = minDist;
                        bestCandidate = candidate;
                    }
                }

                if (bestCandidate != null)
                    _landmarks.Add(bestCandidate);
            }
        }

        private void AdjustThreshold(int currentSpeciesCount)
        {
            double error = (double)(currentSpeciesCount - _config.TargetSpeciesCount) / _config.TargetSpeciesCount;
            _currentThreshold *= (1.0 - _config.ThresholdAdjustmentRate * error);
            _currentThreshold = Math.Clamp(_currentThreshold,
                _config.MinSpeciationThreshold, _config.MaxSpeciationThreshold);
        }

        private ImmutableArray<SpeciesInfo> GetExistingSpecies(GenomePopulation population)
        {
            var speciesDict = new Dictionary<int, SpeciesInfo>();

            foreach (var genome in population.Genomes)
            {
                if (genome.SpeciesId < 0)
                    continue;

                if (!speciesDict.TryGetValue(genome.SpeciesId, out var existing))
                {
                    speciesDict[genome.SpeciesId] = new SpeciesInfo
                    {
                        Id = genome.SpeciesId,
                        Representative = genome,
                        RepresentativeGenomeId = genome.Id,
                        MemberIds = ImmutableArray.Create(genome.Id),
                        BestFitness = genome.Fitness,
                        BestFitnessGeneration = genome.Generation,
                        Age = genome.Age
                    };
                }
                else
                {
                    var memberIds = existing.MemberIds.Add(genome.Id);
                    var bestFitness = Math.Max(existing.BestFitness, genome.Fitness);
                    speciesDict[genome.SpeciesId] = existing with
                    {
                        MemberIds = memberIds,
                        BestFitness = bestFitness
                    };
                }
            }

            return speciesDict.Values.ToImmutableArray();
        }

        /// <summary>
        /// Clears the distance cache to free memory.
        /// </summary>
        public void ClearCache()
        {
            _distanceCache.Clear();
        }

        /// <summary>
        /// Gets the number of cached distance entries.
        /// </summary>
        public int CacheSize => _distanceCache.Count;
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Swarm Evolution Scheduler

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

    #endregion
    // =========================================================================

    // =========================================================================
    #region Fitness Evaluator

    /// <summary>
    /// Multi-objective fitness evaluator for NEAT-G genomes.
    /// Evaluates visual fidelity, performance, memory efficiency, structural complexity,
    /// perceptual quality (JND distance, SSIM approximation), and SDF error metrics.
    /// Provides comprehensive fitness scoring for neural architecture optimization.
    /// </summary>
    public sealed class FitnessEvaluator : IFitnessEvaluator
    {
        private readonly EvaluationContext _context;
        private readonly ImmutableDictionary<FitnessComponent, double> _componentWeights;

        /// <summary>
        /// Initializes a new instance of the FitnessEvaluator class.
        /// </summary>
        /// <param name="context">The evaluation context with scene data and parameters.</param>
        public FitnessEvaluator(EvaluationContext context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _componentWeights = context.ComponentWeights.Count > 0
                ? context.ComponentWeights
                : EvaluationContext.CreateDefault().ComponentWeights;
        }

        /// <inheritdoc/>
        public async Task<GeoGenome> EvaluateAsync(GeoGenome genome, EvaluationContext context, CancellationToken ct)
        {
            if (genome == null)
                throw new ArgumentNullException(nameof(genome));
            if (genome.IsFitnessValid)
                return genome;

            ct.ThrowIfCancellationRequested();

            var components = new Dictionary<FitnessComponent, double>();

            double visualFidelity = await Task.Run(() => ComputeVisualFidelity(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.VisualFidelity] = visualFidelity;

            double performance = await Task.Run(() => ComputePerformanceScore(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.Performance] = performance;

            double memoryEfficiency = await Task.Run(() => ComputeMemoryEfficiency(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.MemoryEfficiency] = memoryEfficiency;

            double structuralComplexity = await Task.Run(() => ComputeStructuralComplexityScore(genome), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.StructuralComplexity] = structuralComplexity;

            double perceptualQuality = await Task.Run(() => ComputePerceptualQuality(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.PerceptualQuality] = perceptualQuality;

            double sdfError = await Task.Run(() => ComputeSDFError(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.SDFError] = sdfError;

            double irradianceError = await Task.Run(() => ComputeIrradianceError(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.IrradianceError] = irradianceError;

            double topoNovelty = ComputeTopologicalNovelty(genome);
            components[FitnessComponent.TopologicalNovelty] = topoNovelty;

            double generalization = ComputeGeneralizationScore(genome, context);
            components[FitnessComponent.Generalization] = generalization;

            double totalFitness = 0;
            double totalWeight = 0;
            foreach (var kvp in components)
            {
                if (_componentWeights.TryGetValue(kvp.Key, out double weight))
                {
                    totalFitness += kvp.Value * weight;
                    totalWeight += weight;
                }
            }

            if (totalWeight > 0)
                totalFitness /= totalWeight;

            genome.FitnessComponents = components.ToImmutableDictionary();
            genome.Fitness = totalFitness;
            genome.RawFitness = totalFitness;
            genome.EvaluationCount++;

            return genome;
        }

        /// <inheritdoc/>
        public ImmutableDictionary<FitnessComponent, double> GetComponentWeights()
        {
            return _componentWeights;
        }

        /// <summary>
        /// Computes visual fidelity score by comparing genome output to reference data.
        /// Uses pixel-wise comparison with perceptual weighting.
        /// </summary>
        private double ComputeVisualFidelity(GeoGenome genome, EvaluationContext context)
        {
            if (context.ReferenceImageData == null || context.ExpectedOutputSize == 0)
                return 0.5;

            double[] output = ForwardPass(genome, context.InputData);
            if (output.Length == 0)
                return 0;

            double mse = 0;
            double mae = 0;
            int compareLength = Math.Min(output.Length, context.ExpectedOutputSize);

            for (int i = 0; i < compareLength; i++)
            {
                double expected = i < context.TargetOutput.Length ? context.TargetOutput[i] : 0;
                double diff = output[i] - expected;
                mse += diff * diff;
                mae += Math.Abs(diff);
            }

            mse /= compareLength;
            mae /= compareLength;

            double mseScore = Math.Exp(-mse * 10.0);
            double maeScore = Math.Exp(-mae * 5.0);

            return 0.6 * mseScore + 0.4 * maeScore;
        }

        /// <summary>
        /// Computes performance score based on inference latency estimation.
        /// Estimates latency from genome complexity metrics.
        /// </summary>
        private double ComputePerformanceScore(GeoGenome genome, EvaluationContext context)
        {
            int activeNeurons = genome.ActiveNeuronCount;
            int activeSynapses = genome.ActiveSynapseCount;
            int depth = genome.MaxLayerDepth;

            double estimatedOps = activeSynapses * 2.0 + activeNeurons;
            double depthFactor = Math.Log2(Math.Max(2, depth));
            double parallelismFactor = activeNeurons > 0
                ? (double)activeNeurons / depth
                : 1.0;

            double estimatedLatencyMs = estimatedOps * 0.001 * depthFactor / Math.Max(1, parallelismFactor * 0.1);

            double score = context.MaxLatencyMs > 0
                ? Math.Exp(-Math.Max(0, estimatedLatencyMs - context.MaxLatencyMs * 0.5) / context.MaxLatencyMs)
                : 1.0;

            return Math.Clamp(score, 0, 1);
        }

        /// <summary>
        /// Computes memory efficiency score.
        /// </summary>
        private double ComputeMemoryEfficiency(GeoGenome genome, EvaluationContext context)
        {
            long estimatedMemory = (long)(genome.ActiveNeuronCount * 64 + genome.ActiveSynapseCount * 16);

            double score = context.MaxMemoryBytes > 0
                ? Math.Exp(-Math.Max(0, estimatedMemory - context.MaxMemoryBytes * 0.3) / context.MaxMemoryBytes)
                : 1.0;

            return Math.Clamp(score, 0, 1);
        }

        /// <summary>
        /// Computes structural complexity penalty score.
        /// Penalizes overly complex genomes to encourage parsimony.
        /// </summary>
        private double ComputeStructuralComplexityScore(GeoGenome genome)
        {
            double complexity = genome.ComputeComplexity();
            double parsimonyPressure = 0.01;
            double score = Math.Exp(-parsimonyPressure * complexity);
            return Math.Clamp(score, 0, 1);
        }

        /// <summary>
        /// Computes perceptual quality using JND (Just Noticeable Difference) distance
        /// and SSIM (Structural Similarity Index) approximation.
        /// </summary>
        private double ComputePerceptualQuality(GeoGenome genome, EvaluationContext context)
        {
            if (context.TargetOutput.Length == 0)
                return 0.5;

            double[] output = ForwardPass(genome, context.InputData);
            if (output.Length == 0)
                return 0;

            int length = Math.Min(output.Length, context.TargetOutput.Length);

            double jndScore = ComputeJNDDistance(output, context.TargetOutput, length);
            double ssimScore = ComputeSSIMApproximation(output, context.TargetOutput, length);

            return 0.5 * jndScore + 0.5 * ssimScore;
        }

        /// <summary>
        /// Computes JND (Just Noticeable Difference) distance between two signal arrays.
        /// Based on the Weber-Fechner law: perception is logarithmic.
        /// </summary>
        private double ComputeJNDDistance(double[] output, ImmutableArray<double> target, int length)
        {
            double jndThreshold = 0.02;
            double aboveJND = 0;

            for (int i = 0; i < length; i++)
            {
                double diff = Math.Abs(output[i] - target[i]);
                double reference = Math.Max(Math.Abs(target[i]), 1e-6);
                double weberRatio = diff / reference;

                if (weberRatio > jndThreshold)
                {
                    aboveJND += weberRatio - jndThreshold;
                }
            }

            double avgAboveJND = length > 0 ? aboveJND / length : 0;
            return Math.Exp(-avgAboveJND * 5.0);
        }

        /// <summary>
        /// Computes SSIM approximation between two signal arrays.
        /// SSIM considers luminance, contrast, and structure.
        /// </summary>
        private double ComputeSSIMApproximation(double[] output, ImmutableArray<double> target, int length)
        {
            if (length == 0)
                return 0;

            double muX = 0, muY = 0;
            for (int i = 0; i < length; i++)
            {
                muX += output[i];
                muY += target[i];
            }
            muX /= length;
            muY /= length;

            double sigmaX2 = 0, sigmaY2 = 0, sigmaXY = 0;
            for (int i = 0; i < length; i++)
            {
                double dx = output[i] - muX;
                double dy = target[i] - muY;
                sigmaX2 += dx * dx;
                sigmaY2 += dy * dy;
                sigmaXY += dx * dy;
            }
            sigmaX2 /= length - 1;
            sigmaY2 /= length - 1;
            sigmaXY /= length - 1;

            double C1 = 0.01 * 0.01;
            double C2 = 0.03 * 0.03;

            double luminance = (2 * muX * muY + C1) / (muX * muX + muY * muY + C1);
            double contrast = (2 * Math.Sqrt(Math.Max(0, sigmaX2)) * Math.Sqrt(Math.Max(0, sigmaY2)) + C2) /
                             (sigmaX2 + sigmaY2 + C2);
            double structure = (sigmaXY + C2 / 2) /
                              (Math.Sqrt(Math.Max(0, sigmaX2)) * Math.Sqrt(Math.Max(0, sigmaY2)) + C2 / 2);

            return Math.Clamp(luminance * contrast * structure, 0, 1);
        }

        /// <summary>
        /// Computes SDF (Signed Distance Function) error metrics for geometric accuracy.
        /// </summary>
        private double ComputeSDFError(GeoGenome genome, EvaluationContext context)
        {
            if (context.TargetOutput.Length == 0)
                return 0.5;

            double[] output = ForwardPass(genome, context.InputData);
            if (output.Length == 0)
                return 0;

            int length = Math.Min(output.Length, context.TargetOutput.Length);
            double maxError = 0;
            double mseError = 0;
            double chamferDistance = 0;

            for (int i = 0; i < length; i++)
            {
                double error = Math.Abs(output[i] - context.TargetOutput[i]);
                maxError = Math.Max(maxError, error);
                mseError += error * error;
                chamferDistance += error * error;
            }

            mseError /= length;
            chamferDistance = Math.Sqrt(chamferDistance / length);

            double maxErrorScore = Math.Exp(-maxError * 5.0);
            double mseScore = Math.Exp(-mseError * 10.0);
            double chamferScore = Math.Exp(-chamferDistance * 3.0);

            return 0.4 * maxErrorScore + 0.3 * mseScore + 0.3 * chamferScore;
        }

        /// <summary>
        /// Computes mean-squared irradiance error between the genome forward pass and
        /// <see cref="EvaluationContext.TargetOutput"/>. GeoGenome does not embed L-DNN
        /// weights directly; NEAT-G evolves a proxy irradiance predictor head whose outputs
        /// are compared against reference irradiance samples.
        /// </summary>
        private double ComputeIrradianceError(GeoGenome genome, EvaluationContext context)
        {
            if (context.TargetOutput.Length == 0)
                return 0.5;

            double[] output = ForwardPass(genome, context.InputData);
            if (output.Length == 0)
                return 0;

            int length = Math.Min(output.Length, context.TargetOutput.Length);
            double mse = 0;
            for (int i = 0; i < length; i++)
            {
                double diff = output[i] - context.TargetOutput[i];
                mse += diff * diff;
            }

            mse /= length;
            return Math.Exp(-mse * 10.0);
        }

        /// <summary>
        /// Computes topological novelty based on genome structure uniqueness.
        /// </summary>
        private double ComputeTopologicalNovelty(GeoGenome genome)
        {
            long hash = genome.ComputeTopologyHash();
            double hashNorm = (double)(hash & 0x7FFFFFFF) / int.MaxValue;
            return hashNorm;
        }

        /// <summary>
        /// Computes generalization score based on genome's expected ability to handle unseen data.
        /// </summary>
        private double ComputeGeneralizationScore(GeoGenome genome, EvaluationContext context)
        {
            double parsimony = Math.Exp(-0.01 * genome.ComputeComplexity());
            double dropout = (double)genome.Neurons.Count(n => !n.IsActive) /
                           Math.Max(1, genome.Neurons.Count);

            double implicitRegularization = 0.5 * parsimony + 0.5 * dropout;
            return Math.Clamp(implicitRegularization, 0, 1);
        }

        /// <summary>
        /// Performs a forward pass through the genome's neural network.
        /// </summary>
        /// <param name="genome">The genome to evaluate.</param>
        /// <param name="input">Input values.</param>
        /// <returns>Output values from the network.</returns>
        internal static double[] ForwardPass(GeoGenome genome, ImmutableArray<double> input)
        {
            if (genome == null || genome.Neurons.Count == 0)
                return Array.Empty<double>();

            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            if (activeNeurons.Count == 0)
                return Array.Empty<double>();

            var neuronValues = new Dictionary<long, double>();

            var inputNeurons = activeNeurons
                .Where(n => n.LayerIndex == 0)
                .OrderBy(n => n.PositionInLayer)
                .ToList();

            for (int i = 0; i < inputNeurons.Count; i++)
            {
                double val = i < input.Length ? input[i] : 0;
                neuronValues[inputNeurons[i].InnovationNumber] = val;
            }

            var layers = activeNeurons
                .GroupBy(n => n.LayerIndex)
                .OrderBy(g => g.Key)
                .ToList();

            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            foreach (var layer in layers)
            {
                if (layer.Key == 0)
                    continue;

                foreach (var neuron in layer)
                {
                    double weightedSum = neuron.Bias;

                    var inputs = activeSynapses
                        .Where(s => s.TargetNeuronId == neuron.InnovationNumber)
                        .ToList();

                    foreach (var synapse in inputs)
                    {
                        if (neuronValues.TryGetValue(synapse.SourceNeuronId, out double srcVal))
                        {
                            weightedSum += synapse.Weight * srcVal;
                        }
                    }

                    neuronValues[neuron.InnovationNumber] = neuron.Activate(weightedSum);
                }
            }

            var outputNeurons = activeNeurons
                .Where(n => n.LayerIndex == genome.MaxLayerDepth || n.LayerIndex == layers.Max(l => l.Key))
                .OrderBy(n => n.PositionInLayer)
                .ToList();

            if (outputNeurons.Count == 0)
            {
                outputNeurons = layers.Last().ToList();
            }

            var output = new double[outputNeurons.Count];
            for (int i = 0; i < outputNeurons.Count; i++)
            {
                if (neuronValues.TryGetValue(outputNeurons[i].InnovationNumber, out double val))
                    output[i] = val;
            }

            return output;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Genome Population Manager

    /// <summary>
    /// Manages the lifecycle of genome populations including initialization,
    /// diversity maintenance, elitism, age tracking, and stagnation detection.
    /// Provides utilities for population-level operations and statistics.
    /// </summary>
    public sealed class GenomePopulationManager
    {
        private readonly EvolutionConfig _config;
        private readonly Random _rng;
        private long _nextInnovationNumber;
        private int _nextSpeciesId;

        /// <summary>
        /// Initializes a new instance of the GenomePopulationManager class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="rng">Random number generator.</param>
        public GenomePopulationManager(EvolutionConfig config, Random rng)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _rng = rng ?? throw new ArgumentNullException(nameof(rng));
            _nextInnovationNumber = 1;
            _nextSpeciesId = 1;
        }

        /// <summary>Gets the next available innovation number.</summary>
        public long NextInnovationNumber => Interlocked.Increment(ref _nextInnovationNumber);

        /// <summary>Gets the next available species ID.</summary>
        public int NextSpeciesId => Interlocked.Increment(ref _nextSpeciesId);

        /// <summary>
        /// Initializes a new random population of genomes.
        /// </summary>
        /// <param name="inputCount">Number of input neurons.</param>
        /// <param name="outputCount">Number of output neurons.</param>
        /// <returns>An initialized genome population.</returns>
        public GenomePopulation InitializePopulation(int inputCount, int outputCount)
        {
            var genomes = new List<GeoGenome>();

            for (int i = 0; i < _config.PopulationSize; i++)
            {
                var genome = CreateRandomGenome(inputCount, outputCount, 0);
                genomes.Add(genome);
            }

            return new GenomePopulation
            {
                Genomes = genomes.ToImmutableArray(),
                GenerationNumber = 0,
                Statistics = new PopulationStatistics
                {
                    SpeciesCount = 0,
                    EvaluationsThisGeneration = 0
                }
            };
        }

        /// <summary>
        /// Creates a random genome with the specified architecture.
        /// </summary>
        /// <param name="inputCount">Number of input neurons.</param>
        /// <param name="outputCount">Number of output neurons.</param>
        /// <param name="generation">Generation number.</param>
        /// <returns>A new random genome.</returns>
        public GeoGenome CreateRandomGenome(int inputCount, int outputCount, int generation)
        {
            var genome = new GeoGenome
            {
                Id = Guid.NewGuid(),
                Generation = generation,
                InputCount = inputCount,
                OutputCount = outputCount,
                Age = 0,
                BestFitness = double.MinValue,
                EvaluationCount = 0
            };

            var allActivations = Enum.GetValues<ActivationFunction>();

            for (int i = 0; i < inputCount; i++)
            {
                genome.Neurons.Add(new GeoNeuron
                {
                    InnovationNumber = NextInnovationNumber,
                    LayerIndex = 0,
                    PositionInLayer = i,
                    Activation = ActivationFunction.Linear,
                    Bias = 0,
                    IsActive = true,
                    CreationGeneration = generation
                });
            }

            for (int i = 0; i < outputCount; i++)
            {
                genome.Neurons.Add(new GeoNeuron
                {
                    InnovationNumber = NextInnovationNumber,
                    LayerIndex = 1,
                    PositionInLayer = i,
                    Activation = allActivations[_rng.Next(allActivations.Length)],
                    Bias = (_rng.NextDouble() * 2.0 - 1.0) * _config.BiasInitRange,
                    IsActive = true,
                    CreationGeneration = generation
                });
            }

            for (int i = 0; i < inputCount; i++)
            {
                for (int j = 0; j < outputCount; j++)
                {
                    genome.Synapses.Add(new GeoSynapse
                    {
                        InnovationNumber = NextInnovationNumber,
                        SourceNeuronId = genome.Neurons[i].InnovationNumber,
                        TargetNeuronId = genome.Neurons[inputCount + j].InnovationNumber,
                        Weight = (_rng.NextDouble() * 2.0 - 1.0) * _config.WeightInitRange,
                        IsActive = true,
                        CreationGeneration = generation
                    });
                }
            }

            genome.ComputeComplexity();
            genome.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

            return genome;
        }

        /// <summary>
        /// Applies elitism by preserving the top individuals unchanged.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <param name="species">Species information.</param>
        /// <returns>List of elite genomes to preserve.</returns>
        public IReadOnlyList<GeoGenome> GetElites(GenomePopulation population, ImmutableArray<SpeciesInfo> species)
        {
            var elites = new List<GeoGenome>();

            foreach (var s in species)
            {
                if (s.MemberIds.Length == 0)
                    continue;

                var speciesMembers = population.Genomes
                    .Where(g => s.MemberIds.Contains(g.Id))
                    .OrderByDescending(g => g.Fitness)
                    .ToList();

                int eliteCount = Math.Min(_config.EliteCount, speciesMembers.Count);
                for (int i = 0; i < eliteCount; i++)
                {
                    var elite = speciesMembers[i].Clone();
                    elite.Age++;
                    elites.Add(elite);
                }
            }

            if (elites.Count == 0 && population.Genomes.Length > 0)
            {
                int globalEliteCount = Math.Min(_config.EliteCount,
                    (int)(_config.PopulationSize * _config.EliteFraction));
                var globalElites = population.Genomes
                    .OrderByDescending(g => g.Fitness)
                    .Take(globalEliteCount);

                foreach (var elite in globalElites)
                {
                    var clone = elite.Clone();
                    clone.Age++;
                    elites.Add(clone);
                }
            }

            return elites;
        }

        /// <summary>
        /// Updates age tracking for all genomes in the population.
        /// </summary>
        /// <param name="population">The population to update.</param>
        /// <returns>A new population with updated ages.</returns>
        public GenomePopulation UpdateAges(GenomePopulation population)
        {
            var updatedGenomes = population.Genomes.Select(g =>
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

            return population with
            {
                Genomes = updatedGenomes,
                GenerationNumber = population.GenerationNumber + 1
            };
        }

        /// <summary>
        /// Detects species stagnation and marks stagnant species for potential extinction.
        /// </summary>
        /// <param name="species">Current species information.</param>
        /// <returns>Updated species with stagnation status.</returns>
        public ImmutableArray<SpeciesInfo> DetectStagnation(ImmutableArray<SpeciesInfo> species)
        {
            return species.Select(s =>
            {
                bool isStagnant = s.StagnationCounter >= _config.MaxStagnationGenerations;

                bool markedForExtinction = isStagnant && s.MemberCount <= _config.SpeciesMinimumSize;

                return s with
                {
                    IsStagnant = isStagnant,
                    IsMarkedForExtinction = markedForExtinction
                };
            }).ToImmutableArray();
        }

        /// <summary>
        /// Removes extinct species and reassigns their members.
        /// </summary>
        /// <param name="species">Current species information.</param>
        /// <param name="population">Current population.</param>
        /// <returns>Tuple of surviving species and reassignment map.</returns>
        public (ImmutableArray<SpeciesInfo> survivingSpecies, Dictionary<Guid, int> reassignments)
            HandleExtinction(ImmutableArray<SpeciesInfo> species, GenomePopulation population)
        {
            var surviving = new List<SpeciesInfo>();
            var reassignments = new Dictionary<Guid, int>();
            var activeSpeciesIds = new List<int>();

            foreach (var s in species)
            {
                if (!s.IsMarkedForExtinction && s.MemberCount >= _config.SpeciesMinimumSize)
                {
                    surviving.Add(s);
                    activeSpeciesIds.Add(s.Id);
                }
            }

            if (activeSpeciesIds.Count == 0 && species.Length > 0)
            {
                surviving.Add(species[0] with { IsMarkedForExtinction = false });
                activeSpeciesIds.Add(species[0].Id);
            }

            foreach (var s in species.Where(s => s.IsMarkedForExtinction || s.MemberCount < _config.SpeciesMinimumSize))
            {
                foreach (var memberId in s.MemberIds)
                {
                    if (activeSpeciesIds.Count > 0)
                    {
                        reassignments[memberId] = activeSpeciesIds[_rng.Next(activeSpeciesIds.Count)];
                    }
                }
            }

            return (surviving.ToImmutableArray(), reassignments);
        }

        /// <summary>
        /// Computes population diversity metrics.
        /// </summary>
        /// <param name="population">The population to analyze.</param>
        /// <returns>Diversity metrics including topology diversity and species balance.</returns>
        public PopulationStatistics ComputePopulationStatistics(GenomePopulation population, ImmutableArray<SpeciesInfo> species)
        {
            if (population.Genomes.Length == 0)
                return new PopulationStatistics();

            var fitnesses = population.Genomes.Select(g => g.Fitness).ToArray();
            double mean = fitnesses.Average();
            double variance = fitnesses.Average(f => (f - mean) * (f - mean));
            double stdDev = Math.Sqrt(variance);

            var topologyHashes = population.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count();

            var speciesCounts = species.Select(s => s.MemberCount).ToArray();
            double shannonEntropy = 0;
            int totalMembers = speciesCounts.Sum();
            if (totalMembers > 0)
            {
                foreach (var count in speciesCounts)
                {
                    if (count > 0)
                    {
                        double p = (double)count / totalMembers;
                        shannonEntropy -= p * Math.Log2(p);
                    }
                }
            }
            double maxEntropy = speciesCounts.Length > 0 ? Math.Log2(speciesCounts.Length) : 1;
            double diversityIndex = maxEntropy > 0 ? shannonEntropy / maxEntropy : 0;

            return new PopulationStatistics
            {
                MeanFitness = mean,
                MedianFitness = population.MedianFitness,
                StdDevFitness = stdDev,
                BestFitness = fitnesses.Max(),
                WorstFitness = fitnesses.Min(),
                UniqueTopologies = topologyHashes,
                SpeciesCount = species.Length,
                DiversityIndex = diversityIndex,
                AverageComplexity = population.Genomes.Average(g => g.Complexity),
                EvaluationsThisGeneration = population.Genomes.Count(g => !g.IsFitnessValid)
            };
        }

        /// <summary>
        /// Computes fitness sharing within species.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <param name="species">Species information.</param>
        /// <returns>Population with adjusted fitness values.</returns>
        public GenomePopulation ApplyFitnessSharing(GenomePopulation population, ImmutableArray<SpeciesInfo> species)
        {
            var adjustedGenomes = population.Genomes.Select(genome =>
            {
                var clone = genome.Clone();

                var speciesInfo = species.FirstOrDefault(s => s.MemberIds.Contains(genome.Id));
                if (speciesInfo.MemberCount > 0)
                {
                    double sharedFitness = genome.Fitness / Math.Pow(speciesInfo.MemberCount, _config.SharingExponent);
                    clone.AdjustedFitness = sharedFitness;
                }
                else
                {
                    clone.AdjustedFitness = genome.Fitness;
                }

                return clone;
            }).ToImmutableArray();

            return population with { Genomes = adjustedGenomes };
        }

        /// <summary>
        /// Allocates offspring counts to each species based on adjusted fitness.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <param name="totalOffspring">Total offspring to allocate.</param>
        /// <returns>Species with offspring allocation.</returns>
        public ImmutableArray<SpeciesInfo> AllocateOffspring(ImmutableArray<SpeciesInfo> species, int totalOffspring)
        {
            double totalFitness = species.Sum(s => Math.Max(0, s.AverageFitness * s.MemberCount));
            if (totalFitness <= 0)
            {
                int perSpecies = totalOffspring / Math.Max(1, species.Length);
                return species.Select(s => s with { OffspringAllocation = perSpecies }).ToImmutableArray();
            }

            var allocated = new List<SpeciesInfo>();
            int allocated_total = 0;

            foreach (var s in species)
            {
                double proportion = (s.AverageFitness * s.MemberCount) / totalFitness;
                int offspring = Math.Max(_config.EliteCount, (int)(proportion * totalOffspring));
                offspring = Math.Min(offspring, totalOffspring - allocated_total);
                allocated_total += offspring;
                allocated.Add(s with { OffspringAllocation = offspring });
            }

            if (allocated_total < totalOffspring)
            {
                var last = allocated.Last();
                allocated[^1] = last with { OffspringAllocation = last.OffspringAllocation + (totalOffspring - allocated_total) };
            }

            return allocated.ToImmutableArray();
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution History Tracker

    /// <summary>
    /// Records and tracks all evolution events, statistics, and fitness history.
    /// Provides export capabilities and visualization data for evolution analysis.
    /// </summary>
    public sealed class EvolutionHistoryTracker
    {
        private readonly EvolutionConfig _config;
        private readonly List<EvolutionEvent> _events;
        private readonly List<EvolutionMetrics> _metricsHistory;
        private readonly Dictionary<int, List<double>> _speciesFitnessHistory;
        private readonly Dictionary<int, List<int>> _speciesSizeHistory;
        private readonly ConcurrentQueue<EvolutionEvent> _eventBuffer;
        private readonly object _lock = new();

        /// <summary>
        /// Initializes a new instance of the EvolutionHistoryTracker class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public EvolutionHistoryTracker(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _events = new List<EvolutionEvent>();
            _metricsHistory = new List<EvolutionMetrics>();
            _speciesFitnessHistory = new Dictionary<int, List<double>>();
            _speciesSizeHistory = new Dictionary<int, List<int>>();
            _eventBuffer = new ConcurrentQueue<EvolutionEvent>();
        }

        /// <summary>Gets the total number of recorded events.</summary>
        public int EventCount
        {
            get { lock (_lock) { return _events.Count; } }
        }

        /// <summary>Gets the number of recorded metrics entries.</summary>
        public int MetricsCount
        {
            get { lock (_lock) { return _metricsHistory.Count; } }
        }

        /// <summary>
        /// Records an evolution event.
        /// </summary>
        /// <param name="eventType">Type of event.</param>
        /// <param name="generation">Current generation.</param>
        /// <param name="description">Event description.</param>
        /// <param name="genomeId">Associated genome ID (optional).</param>
        /// <param name="speciesId">Associated species ID (optional).</param>
        /// <param name="value">Numeric value (optional).</param>
        public void RecordEvent(
            EvolutionEventType eventType,
            int generation,
            string description,
            Guid? genomeId = null,
            int? speciesId = null,
            double value = 0)
        {
            var evolutionEvent = new EvolutionEvent
            {
                EventType = eventType,
                Generation = generation,
                Description = description,
                GenomeId = genomeId,
                SpeciesId = speciesId,
                Value = value,
                Timestamp = DateTime.UtcNow
            };

            _eventBuffer.Enqueue(evolutionEvent);

            lock (_lock)
            {
                if (_config.EnableHistoryTracking && _events.Count < _config.MaxHistoryEntries)
                {
                    _events.Add(evolutionEvent);
                }
            }
        }

        /// <summary>
        /// Records evolution metrics for a generation.
        /// </summary>
        /// <param name="metrics">The metrics to record.</param>
        public void RecordMetrics(EvolutionMetrics metrics)
        {
            lock (_lock)
            {
                _metricsHistory.Add(metrics);
            }
        }

        /// <summary>
        /// Records species fitness history.
        /// </summary>
        /// <param name="speciesId">Species identifier.</param>
        /// <param name="fitness">Current best fitness.</param>
        /// <param name="size">Current species size.</param>
        public void RecordSpeciesData(int speciesId, double fitness, int size)
        {
            lock (_lock)
            {
                if (!_speciesFitnessHistory.ContainsKey(speciesId))
                    _speciesFitnessHistory[speciesId] = new List<double>();
                _speciesFitnessHistory[speciesId].Add(fitness);

                if (!_speciesSizeHistory.ContainsKey(speciesId))
                    _speciesSizeHistory[speciesId] = new List<int>();
                _speciesSizeHistory[speciesId].Add(size);
            }
        }

        /// <summary>
        /// Gets all recorded events.
        /// </summary>
        public IReadOnlyList<EvolutionEvent> GetEvents()
        {
            lock (_lock)
            {
                return _events.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets events filtered by type.
        /// </summary>
        /// <param name="eventType">The event type to filter by.</param>
        public IReadOnlyList<EvolutionEvent> GetEventsByType(EvolutionEventType eventType)
        {
            lock (_lock)
            {
                return _events.Where(e => e.EventType == eventType).ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets all recorded metrics.
        /// </summary>
        public IReadOnlyList<EvolutionMetrics> GetMetricsHistory()
        {
            lock (_lock)
            {
                return _metricsHistory.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets fitness over generations as a series of (generation, fitness) pairs.
        /// </summary>
        public IReadOnlyList<(int Generation, double BestFitness, double AverageFitness)> GetFitnessOverGenerations()
        {
            lock (_lock)
            {
                return _metricsHistory
                    .Select(m => (m.Generation, m.BestFitness, m.AverageFitness))
                    .ToList()
                    .AsReadOnly();
            }
        }

        /// <summary>
        /// Gets species fitness history for a specific species.
        /// </summary>
        /// <param name="speciesId">Species identifier.</param>
        public IReadOnlyList<(int Generation, double Fitness, int Size)> GetSpeciesHistory(int speciesId)
        {
            lock (_lock)
            {
                var result = new List<(int, double, int)>();
                if (_speciesFitnessHistory.TryGetValue(speciesId, out var fitnesses))
                {
                    if (_speciesSizeHistory.TryGetValue(speciesId, out var sizes))
                    {
                        for (int i = 0; i < fitnesses.Count; i++)
                        {
                            int size = i < sizes.Count ? sizes[i] : 0;
                            result.Add((i, fitnesses[i], size));
                        }
                    }
                }
                return result.AsReadOnly();
            }
        }

        /// <summary>
        /// Gets summary statistics for the evolution run.
        /// </summary>
        public EvolutionSummary GetSummary()
        {
            lock (_lock)
            {
                if (_metricsHistory.Count == 0)
                {
                    return new EvolutionSummary { TotalGenerations = 0 };
                }

                var bestEver = _metricsHistory.MaxBy(m => m.BestFitness);
                var first = _metricsHistory.First();
                var last = _metricsHistory.Last();

                return new EvolutionSummary
                {
                    TotalGenerations = _metricsHistory.Count,
                    BestFitnessEver = bestEver.BestFitness,
                    BestFitnessGeneration = bestEver.Generation,
                    InitialFitness = first.BestFitness,
                    FinalFitness = last.BestFitness,
                    TotalEvaluations = last.TotalEvaluations,
                    PeakSpeciesCount = _metricsHistory.Max(m => m.SpeciesCount),
                    FinalSpeciesCount = last.SpeciesCount,
                    AverageDiversity = _metricsHistory.Average(m => m.DiversityMetric),
                    FitnessImprovement = last.BestFitness - first.BestFitness,
                    TotalEvents = _events.Count
                };
            }
        }

        /// <summary>
        /// Exports evolution history as a JSON string.
        /// </summary>
        /// <param name="includeEvents">Whether to include detailed events.</param>
        /// <returns>JSON string representation of the history.</returns>
        public string ExportJson(bool includeEvents = false)
        {
            lock (_lock)
            {
                var data = new
                {
                    Summary = GetSummary(),
                    Metrics = _metricsHistory,
                    Events = includeEvents ? _events : new List<EvolutionEvent>()
                };

                return JsonSerializer.Serialize(data, new JsonSerializerOptions
                {
                    WriteIndented = true,
                    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
                });
            }
        }

        /// <summary>
        /// Exports evolution history as CSV-formatted metrics.
        /// </summary>
        /// <returns>CSV string of generation metrics.</returns>
        public string ExportCsv()
        {
            lock (_lock)
            {
                var sb = new StringBuilder();
                sb.AppendLine("Generation,BestFitness,AverageFitness,MedianFitness,StdDev,SpeciesCount,Diversity,Evaluations,GenTime");

                foreach (var m in _metricsHistory)
                {
                    sb.AppendLine($"{m.Generation},{m.BestFitness:F6},{m.AverageFitness:F6}," +
                                 $"{m.MedianFitness:F6},{m.StdDevFitness:F6},{m.SpeciesCount}," +
                                 $"{m.DiversityMetric:F4},{m.EvaluationsThisGeneration}," +
                                 $"{m.GenerationTime.TotalMilliseconds:F1}");
                }

                return sb.ToString();
            }
        }

        /// <summary>
        /// Clears all recorded history.
        /// </summary>
        public void Clear()
        {
            lock (_lock)
            {
                _events.Clear();
                _metricsHistory.Clear();
                _speciesFitnessHistory.Clear();
                _speciesSizeHistory.Clear();
            }

            while (_eventBuffer.TryDequeue(out _))
            { }
        }
    }

    /// <summary>
    /// Summary statistics for an evolution run.
    /// </summary>
    public record EvolutionSummary
    {
        /// <summary>Total generations executed.</summary>
        public int TotalGenerations { get; init; }

        /// <summary>Best fitness achieved across all generations.</summary>
        public double BestFitnessEver { get; init; }

        /// <summary>Generation when best fitness was achieved.</summary>
        public int BestFitnessGeneration { get; init; }

        /// <summary>Initial population best fitness.</summary>
        public double InitialFitness { get; init; }

        /// <summary>Final population best fitness.</summary>
        public double FinalFitness { get; init; }

        /// <summary>Total fitness evaluations.</summary>
        public long TotalEvaluations { get; init; }

        /// <summary>Peak species count during evolution.</summary>
        public int PeakSpeciesCount { get; init; }

        /// <summary>Final species count.</summary>
        public int FinalSpeciesCount { get; init; }

        /// <summary>Average diversity across all generations.</summary>
        public double AverageDiversity { get; init; }

        /// <summary>Total fitness improvement from first to last generation.</summary>
        public double FitnessImprovement { get; init; }

        /// <summary>Total events recorded.</summary>
        public int TotalEvents { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Selection Strategies

    /// <summary>
    /// Implements multiple selection strategies including tournament, roulette wheel,
    /// rank-based, truncation, and stochastic universal sampling.
    /// </summary>
    public sealed class SelectionStrategy : ISelectionStrategy
    {
        private readonly EvolutionConfig _config;
        private readonly FitnessObjective _objective;

        /// <summary>
        /// Initializes a new instance of the SelectionStrategy class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public SelectionStrategy(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _objective = config.Objective;
        }

        /// <inheritdoc/>
        public IReadOnlyList<GeoGenome> SelectParents(GenomePopulation population, int count, Random rng)
        {
            return _config.ParentSelection switch
            {
                SelectionMethod.Tournament => TournamentSelect(population, count, _config.TournamentSize, rng),
                SelectionMethod.RouletteWheel => RouletteWheelSelect(population, count, rng),
                SelectionMethod.RankBased => RankBasedSelect(population, count, rng),
                SelectionMethod.Truncation => TruncationSelect(population, count, rng),
                SelectionMethod.StochasticUniversal => StochasticUniversalSelect(population, count, rng),
                _ => TournamentSelect(population, count, _config.TournamentSize, rng)
            };
        }

        /// <inheritdoc/>
        public IReadOnlyList<GeoGenome> SelectSurvivors(
            IReadOnlyList<GeoGenome> current,
            IReadOnlyList<GeoGenome> offspring,
            int targetSize)
        {
            var combined = new List<GeoGenome>();
            combined.AddRange(current);
            combined.AddRange(offspring);

            return _config.SurvivalSelection switch
            {
                SelectionMethod.Truncation => combined
                    .OrderByDescending(g => GetFitnessForComparison(g))
                    .Take(targetSize)
                    .ToList(),
                SelectionMethod.Tournament => TournamentSelectCombined(combined, targetSize, _config.TournamentSize, new Random()),
                _ => combined
                    .OrderByDescending(g => GetFitnessForComparison(g))
                    .Take(targetSize)
                    .ToList()
            };
        }

        private IReadOnlyList<GeoGenome> TournamentSelect(GenomePopulation population, int count, int tournamentSize, Random rng)
        {
            var selected = new List<GeoGenome>();
            var genomes = population.Genomes.ToArray();

            for (int i = 0; i < count; i++)
            {
                GeoGenome? best = null;
                for (int t = 0; t < tournamentSize; t++)
                {
                    var candidate = genomes[rng.Next(genomes.Length)];
                    if (best == null || GetFitnessForComparison(candidate) > GetFitnessForComparison(best))
                    {
                        best = candidate;
                    }
                }
                if (best != null)
                    selected.Add(best);
            }

            return selected;
        }

        private IReadOnlyList<GeoGenome> TournamentSelectCombined(List<GeoGenome> genomes, int count, int tournamentSize, Random rng)
        {
            var selected = new List<GeoGenome>();
            var array = genomes.ToArray();

            for (int i = 0; i < count; i++)
            {
                GeoGenome? best = null;
                for (int t = 0; t < tournamentSize; t++)
                {
                    var candidate = array[rng.Next(array.Length)];
                    if (best == null || GetFitnessForComparison(candidate) > GetFitnessForComparison(best))
                    {
                        best = candidate;
                    }
                }
                if (best != null)
                    selected.Add(best);
            }

            return selected;
        }

        private IReadOnlyList<GeoGenome> RouletteWheelSelect(GenomePopulation population, int count, Random rng)
        {
            var genomes = population.Genomes.ToArray();
            var fitnesses = genomes.Select(g => GetFitnessForComparison(g)).ToArray();

            double minFitness = fitnesses.Min();
            double shifted = minFitness < 0 ? -minFitness + 1 : 0;
            double total = fitnesses.Sum(f => f + shifted);

            var selected = new List<GeoGenome>();
            for (int i = 0; i < count; i++)
            {
                double r = rng.NextDouble() * total;
                double cumulative = 0;
                for (int j = 0; j < genomes.Length; j++)
                {
                    cumulative += fitnesses[j] + shifted;
                    if (cumulative >= r)
                    {
                        selected.Add(genomes[j]);
                        break;
                    }
                }
            }

            return selected;
        }

        private IReadOnlyList<GeoGenome> RankBasedSelect(GenomePopulation population, int count, Random rng)
        {
            var ranked = population.Genomes
                .OrderByDescending(g => GetFitnessForComparison(g))
                .ToArray();

            int n = ranked.Length;
            var probabilities = new double[n];
            double totalProb = 0;

            for (int i = 0; i < n; i++)
            {
                probabilities[i] = (2.0 - _config.TournamentSize) / n +
                                   2.0 * (n - 1 - i) * (_config.TournamentSize - 1) / (n * (n - 1));
                totalProb += probabilities[i];
            }

            var selected = new List<GeoGenome>();
            for (int i = 0; i < count; i++)
            {
                double r = rng.NextDouble() * totalProb;
                double cumulative = 0;
                for (int j = 0; j < n; j++)
                {
                    cumulative += probabilities[j];
                    if (cumulative >= r)
                    {
                        selected.Add(ranked[j]);
                        break;
                    }
                }
            }

            return selected;
        }

        private IReadOnlyList<GeoGenome> TruncationSelect(GenomePopulation population, int count, Random rng)
        {
            int truncationSize = Math.Max(1, population.Genomes.Length / _config.TournamentSize);
            var top = population.Genomes
                .OrderByDescending(g => GetFitnessForComparison(g))
                .Take(truncationSize)
                .ToArray();

            var selected = new List<GeoGenome>();
            for (int i = 0; i < count; i++)
            {
                selected.Add(top[rng.Next(top.Length)]);
            }

            return selected;
        }

        private IReadOnlyList<GeoGenome> StochasticUniversalSelect(GenomePopulation population, int count, Random rng)
        {
            var ranked = population.Genomes
                .OrderByDescending(g => GetFitnessForComparison(g))
                .ToArray();

            int n = ranked.Length;
            double totalFitness = ranked.Sum(g => GetFitnessForComparison(g));
            if (totalFitness <= 0)
            {
                return ranked.Take(count).ToList();
            }

            double spacing = totalFitness / count;
            double start = rng.NextDouble() * spacing;

            var selected = new List<GeoGenome>();
            double cumulative = 0;
            int idx = 0;

            for (int i = 0; i < count; i++)
            {
                double pointer = start + i * spacing;
                while (idx < n - 1 && cumulative + GetFitnessForComparison(ranked[idx]) < pointer)
                {
                    cumulative += GetFitnessForComparison(ranked[idx]);
                    idx++;
                }
                selected.Add(ranked[Math.Min(idx, n - 1)]);
            }

            return selected;
        }

        private double GetFitnessForComparison(GeoGenome genome)
        {
            double fitness = genome.AdjustedFitness != 0 ? genome.AdjustedFitness : genome.Fitness;
            return _objective == FitnessObjective.Minimize ? -fitness : fitness;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Speciation Analytics

    /// <summary>
    /// Provides analytics and diagnostics for species dynamics during evolution.
    /// Tracks diversity indices, inter-species distance matrices, fitness distributions,
    /// and evolutionary lineage.
    /// </summary>
    public sealed class SpeciationAnalytics
    {
        private readonly EvolutionConfig _config;
        private readonly List<SpeciesSnapshot> _snapshots;

        /// <summary>
        /// Initializes a new instance of the SpeciationAnalytics class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public SpeciationAnalytics(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _snapshots = new List<SpeciesSnapshot>();
        }

        /// <summary>
        /// Records a snapshot of species state for analysis.
        /// </summary>
        /// <param name="generation">Current generation.</param>
        /// <param name="species">Current species information.</param>
        /// <param name="population">Current population.</param>
        public void RecordSnapshot(int generation, ImmutableArray<SpeciesInfo> species, GenomePopulation population)
        {
            var snapshot = new SpeciesSnapshot
            {
                Generation = generation,
                SpeciesCount = species.Length,
                SpeciesSizes = species.Select(s => s.MemberCount).ToImmutableArray(),
                SpeciesBestFitness = species.Select(s => s.BestFitness).ToImmutableArray(),
                SpeciesAverageFitness = species.Select(s => s.AverageFitness).ToImmutableArray(),
                ShannonDiversityIndex = ComputeShannonDiversity(species),
                SimpsonDiversityIndex = ComputeSimpsonDiversity(species),
                PopulationDiversity = ComputePopulationDiversity(population),
                InterSpeciesDistances = ComputeInterSpeciesDistanceMatrix(species, population)
            };

            _snapshots.Add(snapshot);
        }

        /// <summary>
        /// Computes the Shannon diversity index for species distribution.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <returns>Shannon diversity index (0 = no diversity, log(n) = max).</returns>
        public double ComputeShannonDiversity(ImmutableArray<SpeciesInfo> species)
        {
            int totalMembers = species.Sum(s => s.MemberCount);
            if (totalMembers == 0)
                return 0;

            double entropy = 0;
            foreach (var s in species)
            {
                if (s.MemberCount > 0)
                {
                    double p = (double)s.MemberCount / totalMembers;
                    entropy -= p * Math.Log2(p);
                }
            }

            return entropy;
        }

        /// <summary>
        /// Computes the Simpson diversity index for species distribution.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <returns>Simpson diversity index (0 = no diversity, 1 = max).</returns>
        public double ComputeSimpsonDiversity(ImmutableArray<SpeciesInfo> species)
        {
            int totalMembers = species.Sum(s => s.MemberCount);
            if (totalMembers <= 1)
                return 0;

            double sumP2 = 0;
            foreach (var s in species)
            {
                double p = (double)s.MemberCount / totalMembers;
                sumP2 += p * p;
            }

            return 1.0 - sumP2;
        }

        /// <summary>
        /// Computes population diversity based on genome feature distances.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <returns>Population diversity metric (0-1).</returns>
        public double ComputePopulationDiversity(GenomePopulation population)
        {
            if (population.Genomes.Length <= 1)
                return 0;

            var fitnesses = population.Genomes.Select(g => g.Fitness).ToArray();
            double range = fitnesses.Max() - fitnesses.Min();
            double stdDev = population.FitnessStandardDeviation;

            double normalizedRange = Math.Min(1.0, range / (Math.Abs(fitnesses.Average()) + 1e-10));
            double normalizedStdDev = Math.Min(1.0, stdDev / (Math.Abs(fitnesses.Average()) + 1e-10));

            return 0.5 * normalizedRange + 0.5 * normalizedStdDev;
        }

        /// <summary>
        /// Computes the inter-species distance matrix.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <param name="population">Current population.</param>
        /// <returns>Distance matrix as a 2D array.</returns>
        public double[,] ComputeInterSpeciesDistanceMatrix(
            ImmutableArray<SpeciesInfo> species,
            GenomePopulation population)
        {
            int n = species.Length;
            var matrix = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    var repA = species[i].Representative;
                    var repB = species[j].Representative;

                    if (repA != null && repB != null)
                    {
                        double dist = ComputeGenomeDistance(repA, repB);
                        matrix[i, j] = dist;
                        matrix[j, i] = dist;
                    }
                }
            }

            return matrix;
        }

        /// <summary>
        /// Gets species fitness distribution statistics.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <returns>Per-species fitness statistics.</returns>
        public IReadOnlyList<SpeciesFitnessStats> GetFitnessDistribution(ImmutableArray<SpeciesInfo> species)
        {
            return species.Select(s => new SpeciesFitnessStats
            {
                SpeciesId = s.Id,
                BestFitness = s.BestFitness,
                AverageFitness = s.AverageFitness,
                MemberCount = s.MemberCount,
                FitnessVariance = ComputeSpeciesVariance(s),
                StagnationCounter = s.StagnationCounter,
                Age = s.Age
            }).ToList().AsReadOnly();
        }

        /// <summary>
        /// Gets snapshots of species over time.
        /// </summary>
        public IReadOnlyList<SpeciesSnapshot> GetSnapshots()
        {
            return _snapshots.ToList().AsReadOnly();
        }

        /// <summary>
        /// Gets species count over generations.
        /// </summary>
        public IReadOnlyList<(int Generation, int SpeciesCount)> GetSpeciesCountOverTime()
        {
            return _snapshots.Select(s => (s.Generation, s.SpeciesCount)).ToList().AsReadOnly();
        }

        /// <summary>
        /// Computes the rate of speciation (new species per generation).
        /// </summary>
        /// <param name="windowSize">Number of generations to average over.</param>
        /// <returns>Average speciation rate.</returns>
        public double ComputeSpeciationRate(int windowSize = 10)
        {
            if (_snapshots.Count < windowSize)
                return 0;

            var recent = _snapshots.Skip(_snapshots.Count - windowSize).ToList();
            int totalNewSpecies = 0;

            for (int i = 1; i < recent.Count; i++)
            {
                totalNewSpecies += Math.Max(0, recent[i].SpeciesCount - recent[i - 1].SpeciesCount);
            }

            return (double)totalNewSpecies / (recent.Count - 1);
        }

        /// <summary>
        /// Computes the extinction rate (species lost per generation).
        /// </summary>
        /// <param name="windowSize">Number of generations to average over.</param>
        /// <returns>Average extinction rate.</returns>
        public double ComputeExtinctionRate(int windowSize = 10)
        {
            if (_snapshots.Count < windowSize)
                return 0;

            var recent = _snapshots.Skip(_snapshots.Count - windowSize).ToList();
            int totalExtinctions = 0;

            for (int i = 1; i < recent.Count; i++)
            {
                totalExtinctions += Math.Max(0, recent[i - 1].SpeciesCount - recent[i].SpeciesCount);
            }

            return (double)totalExtinctions / (recent.Count - 1);
        }

        /// <summary>
        /// Clears recorded snapshots.
        /// </summary>
        public void Clear()
        {
            _snapshots.Clear();
        }

        private double ComputeGenomeDistance(GeoGenome a, GeoGenome b)
        {
            if (a.SemanticEmbedding.IsDefaultOrEmpty || b.SemanticEmbedding.IsDefaultOrEmpty)
                return 1.0;

            int dim = Math.Min(a.SemanticEmbedding.Length, b.SemanticEmbedding.Length);
            double dist = 0;
            for (int d = 0; d < dim; d++)
            {
                double diff = a.SemanticEmbedding[d] - b.SemanticEmbedding[d];
                dist += diff * diff;
            }
            return Math.Sqrt(dist);
        }

        private double ComputeSpeciesVariance(SpeciesInfo species)
        {
            if (species.MemberCount <= 1)
                return 0;
            double mean = species.AverageFitness;
            return species.FitnessHistory.Length > 1
                ? species.FitnessHistory.Select(f => (f - mean) * (f - mean)).Average()
                : 0;
        }
    }

    /// <summary>
    /// Statistics for a single species' fitness distribution.
    /// </summary>
    public record SpeciesFitnessStats
    {
        /// <summary>Species identifier.</summary>
        public int SpeciesId { get; init; }

        /// <summary>Best fitness in the species.</summary>
        public double BestFitness { get; init; }

        /// <summary>Average fitness in the species.</summary>
        public double AverageFitness { get; init; }

        /// <summary>Number of members.</summary>
        public int MemberCount { get; init; }

        /// <summary>Fitness variance within the species.</summary>
        public double FitnessVariance { get; init; }

        /// <summary>Stagnation counter.</summary>
        public int StagnationCounter { get; init; }

        /// <summary>Species age.</summary>
        public int Age { get; init; }
    }

    /// <summary>
    /// Snapshot of species state at a specific generation.
    /// </summary>
    public record SpeciesSnapshot
    {
        /// <summary>Generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Number of species.</summary>
        public int SpeciesCount { get; init; }

        /// <summary>Sizes of each species.</summary>
        public ImmutableArray<int> SpeciesSizes { get; init; }

        /// <summary>Best fitness of each species.</summary>
        public ImmutableArray<double> SpeciesBestFitness { get; init; }

        /// <summary>Average fitness of each species.</summary>
        public ImmutableArray<double> SpeciesAverageFitness { get; init; }

        /// <summary>Shannon diversity index.</summary>
        public double ShannonDiversityIndex { get; init; }

        /// <summary>Simpson diversity index.</summary>
        public double SimpsonDiversityIndex { get; init; }

        /// <summary>Population diversity metric.</summary>
        public double PopulationDiversity { get; init; }

        /// <summary>Inter-species distance matrix.</summary>
        public double[,]? InterSpeciesDistances { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Parallel Evolution Scheduler

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
                    catch (Exception)
                    {
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
                        catch (Exception)
                        {
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

    #endregion
    // =========================================================================

    // =========================================================================
    #region Migration Manager

    /// <summary>
    /// Handles gene migration between species pools with semantic compatibility checking,
    /// migration rate control, and anti-premature-convergence measures.
    /// </summary>
    public sealed class MigrationManager
    {
        private readonly EvolutionConfig _config;
        private readonly ConcurrentDictionary<int, Channel<MigrationEvent>> _channels;
        private int _totalMigrations;
        private int _rejectedMigrations;
        private readonly Queue<MigrationEvent> _recentMigrations;

        /// <summary>
        /// Initializes a new instance of the MigrationManager class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public MigrationManager(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _channels = new ConcurrentDictionary<int, Channel<MigrationEvent>>();
            _recentMigrations = new Queue<MigrationEvent>();
        }

        /// <summary>Total migrations performed.</summary>
        public int TotalMigrations => Volatile.Read(ref _totalMigrations);

        /// <summary>Rejected migrations.</summary>
        public int RejectedMigrations => Volatile.Read(ref _rejectedMigrations);

        /// <summary>
        /// Evaluates whether a migration is semantically compatible.
        /// </summary>
        /// <param name="migrant">The genome to migrate.</param>
        /// <param name="targetSpecies">Target species information.</param>
        /// <param name="targetGenomes">Genomes in the target species.</param>
        /// <returns>Compatibility score (0-1) and whether migration is allowed.</returns>
        public (double CompatibilityScore, bool IsAllowed) EvaluateMigration(
            GeoGenome migrant,
            SpeciesInfo targetSpecies,
            IReadOnlyList<GeoGenome> targetGenomes)
        {
            if (targetGenomes.Count == 0)
                return (0.5, true);

            if (migrant.SemanticEmbedding.IsDefaultOrEmpty)
                migrant.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

            double totalDistance = 0;
            int count = 0;

            foreach (var target in targetGenomes.Take(Math.Min(10, targetGenomes.Count)))
            {
                if (target.SemanticEmbedding.IsDefaultOrEmpty)
                    target.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

                int dim = Math.Min(migrant.SemanticEmbedding.Length, target.SemanticEmbedding.Length);
                double dist = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = migrant.SemanticEmbedding[d] - target.SemanticEmbedding[d];
                    dist += diff * diff;
                }
                totalDistance += Math.Sqrt(dist);
                count++;
            }

            double avgDistance = count > 0 ? totalDistance / count : 1.0;
            double compatibilityScore = Math.Exp(-avgDistance * 2.0);

            bool isAllowed = compatibilityScore >= _config.SemanticAlignmentThreshold;

            if (!isAllowed)
            {
                Interlocked.Increment(ref _rejectedMigrations);
            }

            return (compatibilityScore, isAllowed);
        }

        /// <summary>
        /// Performs anti-premature-convergence checks.
        /// </summary>
        /// <param name="species">Current species information.</param>
        /// <param name="population">Current population.</param>
        /// <returns>Whether anti-convergence measures should be activated.</returns>
        public bool CheckAntiConvergence(ImmutableArray<SpeciesInfo> species, GenomePopulation population)
        {
            if (species.Length <= 1)
                return false;

            int totalMembers = species.Sum(s => s.MemberCount);
            if (totalMembers == 0)
                return false;

            double maxSpeciesFraction = species.Max(s => (double)s.MemberCount / totalMembers);
            if (maxSpeciesFraction > 0.6)
                return true;

            double fitnessVariance = population.FitnessStandardDeviation;
            double fitnessRange = population.Genomes.Length > 0
                ? population.Genomes.Max(g => g.Fitness) - population.Genomes.Min(g => g.Fitness)
                : 0;

            if (fitnessRange > 0 && fitnessVariance / fitnessRange < 0.01)
                return true;

            var topologyHashes = population.Genomes
                .Select(g => g.ComputeTopologyHash())
                .Distinct()
                .Count();
            double topologyDiversity = (double)topologyHashes / Math.Max(1, population.Genomes.Length);

            if (topologyDiversity < 0.1)
                return true;

            return false;
        }

        /// <summary>
        /// Gets the recommended migration rate based on current population dynamics.
        /// </summary>
        /// <param name="species">Current species.</param>
        /// <param name="population">Current population.</param>
        /// <returns>Recommended migration rate (0-1).</returns>
        public double GetRecommendedMigrationRate(ImmutableArray<SpeciesInfo> species, GenomePopulation population)
        {
            double baseRate = _config.MigrationRate;

            bool antiConverge = CheckAntiConvergence(species, population);
            if (antiConverge)
            {
                return Math.Min(baseRate * 3.0, 0.3);
            }

            double diversity = 0;
            if (population.Genomes.Length > 0)
            {
                var fitnesses = population.Genomes.Select(g => g.Fitness).ToArray();
                double mean = fitnesses.Average();
                double variance = fitnesses.Average(f => (f - mean) * (f - mean));
                diversity = Math.Min(1.0, Math.Sqrt(variance) / (Math.Abs(mean) + 1e-10));
            }

            if (diversity < 0.1)
                return Math.Min(baseRate * 2.0, 0.2);
            else if (diversity > 0.5)
                return baseRate * 0.5;

            return baseRate;
        }

        /// <summary>
        /// Records a completed migration event.
        /// </summary>
        /// <param name="migrationEvent">The migration event.</param>
        public void RecordMigration(MigrationEvent migrationEvent)
        {
            Interlocked.Increment(ref _totalMigrations);
            lock (_recentMigrations)
            {
                _recentMigrations.Enqueue(migrationEvent);
                while (_recentMigrations.Count > 100)
                    _recentMigrations.Dequeue();
            }
        }

        /// <summary>
        /// Gets recent migration events.
        /// </summary>
        public IReadOnlyList<MigrationEvent> GetRecentMigrations()
        {
            lock (_recentMigrations)
            {
                return _recentMigrations.ToList().AsReadOnly();
            }
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Diagnostics

    /// <summary>
    /// Tracks detailed diagnostics about the evolution process including
    /// mutation success rates, crossover success rates, species dynamics,
    /// and population diversity over time.
    /// </summary>
    public sealed class EvolutionDiagnostics
    {
        private readonly EvolutionConfig _config;
        private readonly List<DiagnosticsSnapshot> _snapshots;
        private readonly ConcurrentDictionary<MutationType, (int attempts, int successes)> _mutationStats;
        private readonly ConcurrentDictionary<string, (int attempts, int successes)> _crossoverStats;

        /// <summary>
        /// Initializes a new instance of the EvolutionDiagnostics class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public EvolutionDiagnostics(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _snapshots = new List<DiagnosticsSnapshot>();
            _mutationStats = new ConcurrentDictionary<MutationType, (int, int)>();
            _crossoverStats = new ConcurrentDictionary<string, (int, int)>();
        }

        /// <summary>
        /// Records a mutation attempt.
        /// </summary>
        /// <param name="type">Mutation type.</param>
        /// <param name="success">Whether the mutation was successful.</param>
        public void RecordMutation(MutationType type, bool success)
        {
            _mutationStats.AddOrUpdate(type,
                _ => success ? (1, 1) : (1, 0),
                (_, old) => (old.attempts + 1, old.successes + (success ? 1 : 0)));
        }

        /// <summary>
        /// Records a crossover attempt.
        /// </summary>
        /// <param name="strategy">Crossover strategy name.</param>
        /// <param name="success">Whether the crossover was successful.</param>
        public void RecordCrossover(string strategy, bool success)
        {
            _crossoverStats.AddOrUpdate(strategy,
                _ => success ? (1, 1) : (1, 0),
                (_, old) => (old.attempts + 1, old.successes + (success ? 1 : 0)));
        }

        /// <summary>
        /// Records a diagnostics snapshot.
        /// </summary>
        /// <param name="generation">Current generation.</param>
        /// <param name="population">Current population.</param>
        /// <param name="species">Current species.</param>
        public void RecordSnapshot(int generation, GenomePopulation population, ImmutableArray<SpeciesInfo> species)
        {
            var snapshot = new DiagnosticsSnapshot
            {
                Generation = generation,
                MutationSuccessRates = GetMutationSuccessRates(),
                CrossoverSuccessRates = GetCrossoverSuccessRates(),
                PopulationDiversity = ComputeDiversity(population),
                SpeciesCount = species.Length,
                SpeciesSizeVariance = species.Length > 1
                    ? species.Select(s => (double)s.MemberCount).ToArray().Select(m => { var avg = species.Average(s => (double)s.MemberCount); return (m - avg) * (m - avg); }).Average()
                    : 0,
                BestFitness = population.Genomes.Length > 0 ? population.Genomes.Max(g => g.Fitness) : 0,
                AverageFitness = population.AverageFitness,
                StructuralDiversity = ComputeStructuralDiversity(population),
                WeightDiversity = ComputeWeightDiversity(population),
                ActiveGeneRatio = ComputeActiveGeneRatio(population)
            };

            lock (_snapshots)
            {
                _snapshots.Add(snapshot);
            }
        }

        /// <summary>
        /// Gets mutation success rates per type.
        /// </summary>
        public IReadOnlyDictionary<MutationType, double> GetMutationSuccessRates()
        {
            return _mutationStats.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.attempts > 0
                    ? (double)kvp.Value.successes / kvp.Value.attempts
                    : 0);
        }

        /// <summary>
        /// Gets crossover success rates per strategy.
        /// </summary>
        public IReadOnlyDictionary<string, double> GetCrossoverSuccessRates()
        {
            return _crossoverStats.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.attempts > 0
                    ? (double)kvp.Value.successes / kvp.Value.attempts
                    : 0);
        }

        /// <summary>
        /// Gets all recorded snapshots.
        /// </summary>
        public IReadOnlyList<DiagnosticsSnapshot> GetSnapshots()
        {
            lock (_snapshots)
            {
                return _snapshots.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets overall mutation success rate across all types.
        /// </summary>
        public double GetOverallMutationSuccessRate()
        {
            int totalAttempts = _mutationStats.Values.Sum(v => v.attempts);
            int totalSuccesses = _mutationStats.Values.Sum(v => v.successes);
            return totalAttempts > 0 ? (double)totalSuccesses / totalAttempts : 0;
        }

        /// <summary>
        /// Gets overall crossover success rate across all strategies.
        /// </summary>
        public double GetOverallCrossoverSuccessRate()
        {
            int totalAttempts = _crossoverStats.Values.Sum(v => v.attempts);
            int totalSuccesses = _crossoverStats.Values.Sum(v => v.successes);
            return totalAttempts > 0 ? (double)totalSuccesses / totalAttempts : 0;
        }

        /// <summary>
        /// Clears all diagnostic data.
        /// </summary>
        public void Clear()
        {
            lock (_snapshots)
            {
                _snapshots.Clear();
            }
            _mutationStats.Clear();
            _crossoverStats.Clear();
        }

        private double ComputeDiversity(GenomePopulation population)
        {
            if (population.Genomes.Length <= 1)
                return 0;
            return population.FitnessStandardDeviation / (Math.Abs(population.AverageFitness) + 1e-10);
        }

        private double ComputeStructuralDiversity(GenomePopulation population)
        {
            if (population.Genomes.Length <= 1)
                return 0;
            var hashes = population.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count();
            return (double)hashes / population.Genomes.Length;
        }

        private double ComputeWeightDiversity(GenomePopulation population)
        {
            if (population.Genomes.Length <= 1)
                return 0;

            var allWeights = population.Genomes
                .SelectMany(g => g.Synapses.Where(s => s.IsActive).Select(s => s.Weight))
                .ToList();

            if (allWeights.Count == 0)
                return 0;

            double mean = allWeights.Average();
            double variance = allWeights.Average(w => (w - mean) * (w - mean));
            return Math.Sqrt(variance);
        }

        private double ComputeActiveGeneRatio(GenomePopulation population)
        {
            if (population.Genomes.Length == 0)
                return 0;

            double totalActive = population.Genomes.Sum(g => g.ActiveNeuronCount + g.ActiveSynapseCount);
            double totalPossible = population.Genomes.Sum(g => g.TotalNeuronCount + g.TotalSynapseCount);

            return totalPossible > 0 ? totalActive / totalPossible : 0;
        }
    }

    /// <summary>
    /// Snapshot of diagnostics data at a specific generation.
    /// </summary>
    public record DiagnosticsSnapshot
    {
        /// <summary>Generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Mutation success rates per type.</summary>
        public IReadOnlyDictionary<MutationType, double> MutationSuccessRates { get; init; } =
            new Dictionary<MutationType, double>();

        /// <summary>Crossover success rates per strategy.</summary>
        public IReadOnlyDictionary<string, double> CrossoverSuccessRates { get; init; } =
            new Dictionary<string, double>();

        /// <summary>Population diversity metric.</summary>
        public double PopulationDiversity { get; init; }

        /// <summary>Species count.</summary>
        public int SpeciesCount { get; init; }

        /// <summary>Variance of species sizes.</summary>
        public double SpeciesSizeVariance { get; init; }

        /// <summary>Best fitness.</summary>
        public double BestFitness { get; init; }

        /// <summary>Average fitness.</summary>
        public double AverageFitness { get; init; }

        /// <summary>Structural diversity (unique topologies / total).</summary>
        public double StructuralDiversity { get; init; }

        /// <summary>Weight diversity (std dev of weights).</summary>
        public double WeightDiversity { get; init; }

        /// <summary>Ratio of active genes to total genes.</summary>
        public double ActiveGeneRatio { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Core Engine

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

    #endregion
    // =========================================================================

    // =========================================================================
    #region Event Args

    /// <summary>
    /// Event arguments for generation completion.
    /// </summary>
    public sealed class GenerationCompletedEventArgs : EventArgs
    {
        /// <summary>Generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Metrics for this generation.</summary>
        public EvolutionMetrics Metrics { get; init; } = default!;

        /// <summary>Best genome in this generation.</summary>
        public GeoGenome? BestGenome { get; init; }
    }

    /// <summary>
    /// Event arguments for evolution milestones.
    /// </summary>
    public sealed class EvolutionMilestoneEventArgs : EventArgs
    {
        /// <summary>Generation when the milestone was reached.</summary>
        public int Generation { get; init; }

        /// <summary>Best fitness at the milestone.</summary>
        public double BestFitness { get; init; }

        /// <summary>Type of milestone.</summary>
        public EvolutionMilestoneType MilestoneType { get; init; }
    }

    /// <summary>
    /// Types of evolution milestones.
    /// </summary>
    public enum EvolutionMilestoneType
    {
        /// <summary>Population was initialized.</summary>
        PopulationInitialized,

        /// <summary>New best fitness achieved.</summary>
        NewBestFitness,

        /// <summary>Target fitness reached.</summary>
        TargetReached,

        /// <summary>Stagnation detected.</summary>
        StagnationDetected,

        /// <summary>Species extinction event.</summary>
        SpeciesExtinction,

        /// <summary>Significant migration event.</summary>
        SignificantMigration
    }

    /// <summary>
    /// Event arguments for evolution state changes.
    /// </summary>
    public sealed class EvolutionStateChangeEventArgs : EventArgs
    {
        /// <summary>Previous state.</summary>
        public EvolutionState OldState { get; init; }

        /// <summary>New state.</summary>
        public EvolutionState NewState { get; init; }

        /// <summary>Current generation.</summary>
        public int Generation { get; init; }
    }

    /// <summary>
    /// Event arguments for migration events.
    /// </summary>
    public sealed class MigrationEventArgs : EventArgs
    {
        /// <summary>The migration event details.</summary>
        public MigrationEvent Migration { get; init; } = default!;
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Spatial Models

    namespace Models
    {
        /// <summary>
        /// Represents a point in the semantic embedding space.
        /// Used for manifold distance computations and landmark selection.
        /// </summary>
        public readonly struct EmbeddingPoint : IEquatable<EmbeddingPoint>
        {
            /// <summary>The embedding coordinates.</summary>
            public readonly ImmutableArray<double> Coordinates;

            /// <summary>Optional identifier for this point.</summary>
            public readonly long Id;

            /// <summary>Dimensionality of the embedding.</summary>
            public int Dimension => Coordinates.Length;

            /// <summary>
            /// Initializes a new EmbeddingPoint.
            /// </summary>
            /// <param name="id">Identifier.</param>
            /// <param name="coordinates">Embedding coordinates.</param>
            public EmbeddingPoint(long id, ImmutableArray<double> coordinates)
            {
                Id = id;
                Coordinates = coordinates;
            }

            /// <summary>Computes Euclidean distance to another point.</summary>
            public double DistanceTo(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double sum = 0;
                for (int i = 0; i < dim; i++)
                {
                    double diff = Coordinates[i] - other.Coordinates[i];
                    sum += diff * diff;
                }
                return Math.Sqrt(sum);
            }

            /// <summary>Computes cosine similarity to another point.</summary>
            public double CosineSimilarity(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double dotProduct = 0, normA = 0, normB = 0;
                for (int i = 0; i < dim; i++)
                {
                    dotProduct += Coordinates[i] * other.Coordinates[i];
                    normA += Coordinates[i] * Coordinates[i];
                    normB += other.Coordinates[i] * other.Coordinates[i];
                }
                double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
                return denominator > 1e-10 ? dotProduct / denominator : 0;
            }

            /// <summary>Computes Manhattan distance to another point.</summary>
            public double ManhattanDistanceTo(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double sum = 0;
                for (int i = 0; i < dim; i++)
                {
                    sum += Math.Abs(Coordinates[i] - other.Coordinates[i]);
                }
                return sum;
            }

            /// <summary>Computes Chebyshev (L-infinity) distance to another point.</summary>
            public double ChebyshevDistanceTo(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double maxDiff = 0;
                for (int i = 0; i < dim; i++)
                {
                    maxDiff = Math.Max(maxDiff, Math.Abs(Coordinates[i] - other.Coordinates[i]));
                }
                return maxDiff;
            }

            /// <summary>Computes the Minkowski distance with given exponent.</summary>
            public double MinkowskiDistanceTo(EmbeddingPoint other, double p)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double sum = 0;
                for (int i = 0; i < dim; i++)
                {
                    sum += Math.Pow(Math.Abs(Coordinates[i] - other.Coordinates[i]), p);
                }
                return Math.Pow(sum, 1.0 / p);
            }

            /// <summary>Linearly interpolates between this point and another.</summary>
            public EmbeddingPoint Lerp(EmbeddingPoint other, double t)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                var coords = new double[dim];
                for (int i = 0; i < dim; i++)
                {
                    coords[i] = Coordinates[i] * (1 - t) + other.Coordinates[i] * t;
                }
                return new EmbeddingPoint(Id, coords.ToImmutableArray());
            }

            /// <summary>Normalizes the embedding to unit length.</summary>
            public EmbeddingPoint Normalize()
            {
                double norm = 0;
                for (int i = 0; i < Coordinates.Length; i++)
                    norm += Coordinates[i] * Coordinates[i];
                norm = Math.Sqrt(norm);

                if (norm < 1e-10)
                    return this;

                var normalized = new double[Coordinates.Length];
                for (int i = 0; i < Coordinates.Length; i++)
                    normalized[i] = Coordinates[i] / norm;
                return new EmbeddingPoint(Id, normalized.ToImmutableArray());
            }

            /// <summary>Subtracts two embedding points.</summary>
            public EmbeddingPoint Subtract(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                var result = new double[dim];
                for (int i = 0; i < dim; i++)
                    result[i] = Coordinates[i] - other.Coordinates[i];
                return new EmbeddingPoint(0, result.ToImmutableArray());
            }

            /// <summary>Adds two embedding points.</summary>
            public EmbeddingPoint Add(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                var result = new double[dim];
                for (int i = 0; i < dim; i++)
                    result[i] = Coordinates[i] + other.Coordinates[i];
                return new EmbeddingPoint(0, result.ToImmutableArray());
            }

            /// <summary>Scales the embedding by a scalar.</summary>
            public EmbeddingPoint Scale(double scalar)
            {
                var result = new double[Coordinates.Length];
                for (int i = 0; i < Coordinates.Length; i++)
                    result[i] = Coordinates[i] * scalar;
                return new EmbeddingPoint(Id, result.ToImmutableArray());
            }

            /// <summary>Computes the dot product with another point.</summary>
            public double DotProduct(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double sum = 0;
                for (int i = 0; i < dim; i++)
                    sum += Coordinates[i] * other.Coordinates[i];
                return sum;
            }

            /// <summary>Computes the L2 norm of this point.</summary>
            public double L2Norm()
            {
                double sum = 0;
                for (int i = 0; i < Coordinates.Length; i++)
                    sum += Coordinates[i] * Coordinates[i];
                return Math.Sqrt(sum);
            }

            /// <summary>Creates a zero vector of the given dimension.</summary>
            public static EmbeddingPoint Zero(int dimension) =>
                new(0, ImmutableArray.CreateRange(Enumerable.Repeat(0.0, dimension)));

            /// <summary>Creates a random embedding point.</summary>
            public static EmbeddingPoint Random(int dimension, Random rng) =>
                new(0, ImmutableArray.CreateRange(Enumerable.Range(0, dimension)
                    .Select(_ => rng.NextDouble() * 2 - 1)));

            /// <inheritdoc/>
            public bool Equals(EmbeddingPoint other) => Id == other.Id;

            /// <inheritdoc/>
            public override bool Equals(object? obj) => obj is EmbeddingPoint p && Equals(p);

            /// <inheritdoc/>
            public override int GetHashCode() => Id.GetHashCode();

            /// <inheritdoc/>
            public override string ToString() =>
                $"EmbeddingPoint(Id={Id}, Dim={Dimension}, Norm={L2Norm():F4})";
        }

        /// <summary>
        /// Represents the curvature of a manifold at a specific point.
        /// Used for manifold-aware distance computations.
        /// </summary>
        public readonly struct ManifoldCurvature
        {
            /// <summary>Ricci scalar curvature value.</summary>
            public readonly double ScalarCurvature;

            /// <summary>Sectional curvatures along principal directions.</summary>
            public readonly ImmutableArray<double> SectionalCurvatures;

            /// <summary>Principal directions of curvature.</summary>
            public readonly ImmutableArray<EmbeddingPoint> PrincipalDirections;

            /// <summary>Curvature rank (number of significant curvature components).</summary>
            public int Rank => SectionalCurvatures.Count(c => Math.Abs(c) > 1e-6);

            /// <summary>
            /// Initializes a new ManifoldCurvature.
            /// </summary>
            public ManifoldCurvature(
                double scalarCurvature,
                ImmutableArray<double> sectionalCurvatures,
                ImmutableArray<EmbeddingPoint> principalDirections)
            {
                ScalarCurvature = scalarCurvature;
                SectionalCurvatures = sectionalCurvatures;
                PrincipalDirections = principalDirections;
            }

            /// <summary>
            /// Computes the Gaussian curvature at this point.
            /// </summary>
            public double GaussianCurvature()
            {
                if (SectionalCurvatures.Length < 2)
                    return ScalarCurvature;
                return SectionalCurvatures[0] * SectionalCurvatures[1];
            }

            /// <summary>
            /// Computes the mean curvature at this point.
            /// </summary>
            public double MeanCurvature()
            {
                if (SectionalCurvatures.Length == 0)
                    return 0;
                return SectionalCurvatures.Average();
            }

            /// <summary>
            /// Computes the maximum principal curvature.
            /// </summary>
            public double MaxPrincipalCurvature()
            {
                return SectionalCurvatures.Length > 0
                    ? SectionalCurvatures.Max()
                    : 0;
            }

            /// <summary>
            /// Computes the minimum principal curvature.
            /// </summary>
            public double MinPrincipalCurvature()
            {
                return SectionalCurvatures.Length > 0
                    ? SectionalCurvatures.Min()
                    : 0;
            }

            /// <summary>
            /// Estimates geodesic deviation based on curvature.
            /// </summary>
            /// <param name="distance">Euclidean distance.</param>
            /// <returns>Estimated geodesic distance accounting for curvature.</returns>
            public double EstimateGeodesicDeviation(double distance)
            {
                double k = ScalarCurvature;
                if (Math.Abs(k) < 1e-10)
                    return distance;

                if (k > 0)
                {
                    double sqrtK = Math.Sqrt(k);
                    return Math.Sin(sqrtK * distance) / sqrtK;
                }
                else
                {
                    double sqrtAbsK = Math.Sqrt(-k);
                    return Math.Sinh(sqrtAbsK * distance) / sqrtAbsK;
                }
            }

            /// <summary>Creates a flat (zero curvature) instance.</summary>
            public static ManifoldCurvature Flat(int dimensions) =>
                new(0,
                    ImmutableArray.CreateRange(Enumerable.Repeat(0.0, dimensions)),
                    ImmutableArray<EmbeddingPoint>.Empty);
        }

        /// <summary>
        /// Represents a topological feature (connected component, cycle, etc.)
        /// found in a genome's graph structure.
        /// </summary>
        public readonly struct TopologicalFeature
        {
            /// <summary>Type of topological feature.</summary>
            public readonly TopologicalFeatureType Type;

            /// <summary>Innovation numbers of neurons in this feature.</summary>
            public readonly ImmutableArray<long> NeuronIds;

            /// <summary>Innovation numbers of synapses in this feature.</summary>
            public readonly ImmutableArray<long> SynapseIds;

            /// <summary>Persistence value (for persistent homology).</summary>
            public readonly double Persistence;

            /// <summary>Birth generation of this feature.</summary>
            public readonly int BirthGeneration;

            /// <summary>Death generation of this feature (-1 if still alive).</summary>
            public readonly int DeathGeneration;

            /// <summary>Lifetime of this feature.</summary>
            public int Lifetime =>
                DeathGeneration >= 0 ? DeathGeneration - BirthGeneration : -1;

            /// <summary>
            /// Initializes a new TopologicalFeature.
            /// </summary>
            public TopologicalFeature(
                TopologicalFeatureType type,
                ImmutableArray<long> neuronIds,
                ImmutableArray<long> synapseIds,
                double persistence,
                int birthGeneration,
                int deathGeneration = -1)
            {
                Type = type;
                NeuronIds = neuronIds;
                SynapseIds = synapseIds;
                Persistence = persistence;
                BirthGeneration = birthGeneration;
                DeathGeneration = deathGeneration;
            }

            /// <summary>
            /// Determines if this feature is more significant than another.
            /// </summary>
            public bool IsMoreSignificantThan(TopologicalFeature other)
            {
                return Persistence > other.Persistence;
            }

            /// <inheritdoc/>
            public override string ToString() =>
                $"TopoFeature({Type}, Persistence={Persistence:F4}, Life={Lifetime})";
        }

        /// <summary>
        /// Types of topological features.
        /// </summary>
        public enum TopologicalFeatureType
        {
            /// <summary>A connected component (0-dimensional hole).</summary>
            ConnectedComponent,

            /// <summary>A cycle or loop (1-dimensional hole).</summary>
            Cycle,

            /// <summary>A void or cavity (2-dimensional hole).</summary>
            Void,

            /// <summary>A saddle point in the topology.</summary>
            SaddlePoint,

            /// <summary>A local minimum in the topology.</summary>
            LocalMinimum,

            /// <summary>A local maximum in the topology.</summary>
            LocalMaximum
        }

        /// <summary>
        /// Represents a graph Laplacian eigenvalue spectrum for spectral analysis
        /// of genome topology.
        /// </summary>
        public readonly struct SpectralSignature
        {
            /// <summary>Sorted eigenvalues of the graph Laplacian.</summary>
            public readonly ImmutableArray<double> Eigenvalues;

            /// <summary>Fiedler value (second smallest eigenvalue, algebraic connectivity).</summary>
            public double FiedlerValue => Eigenvalues.Length >= 2 ? Eigenvalues[1] : 0;

            /// <summary>Spectral gap (difference between first two non-zero eigenvalues).</summary>
            public double SpectralGap => Eigenvalues.Length >= 3
                ? Eigenvalues[2] - Eigenvalues[1]
                : 0;

            /// <summary>Number of connected components (count of zero eigenvalues).</summary>
            public int ConnectedComponents
            {
                get
                {
                    int count = 0;
                    foreach (var ev in Eigenvalues)
                    {
                        if (Math.Abs(ev) < 1e-6)
                            count++;
                        else
                            break;
                    }
                    return count;
                }
            }

            /// <summary>
            /// Spectral trace (sum of eigenvalues).
            /// </summary>
            public double Trace => Eigenvalues.Sum();

            /// <summary>
            /// Spectral radius (maximum absolute eigenvalue).
            /// </summary>
            public double SpectralRadius => Eigenvalues.Length > 0 ? Eigenvalues.Max(e => Math.Abs(e)) : 0;

            /// <summary>
            /// Initializes a new SpectralSignature.
            /// </summary>
            public SpectralSignature(ImmutableArray<double> eigenvalues)
            {
                Eigenvalues = eigenvalues.Sort((a, b) => a.CompareTo(b));
            }

            /// <summary>
            /// Computes the spectral distance to another signature.
            /// </summary>
            /// <param name="other">Other spectral signature.</param>
            /// <returns>Spectral distance.</returns>
            public double SpectralDistance(SpectralSignature other)
            {
                int maxLen = Math.Max(Eigenvalues.Length, other.Eigenvalues.Length);
                double dist = 0;

                for (int i = 0; i < maxLen; i++)
                {
                    double a = i < Eigenvalues.Length ? Eigenvalues[i] : 0;
                    double b = i < other.Eigenvalues.Length ? other.Eigenvalues[i] : 0;
                    dist += (a - b) * (a - b);
                }

                return Math.Sqrt(dist);
            }

            /// <summary>
            /// Computes the spectral entropy.
            /// </summary>
            public double SpectralEntropy()
            {
                double trace = Trace;
                if (Math.Abs(trace) < 1e-10)
                    return 0;

                double entropy = 0;
                foreach (var ev in Eigenvalues)
                {
                    double p = Math.Abs(ev) / trace;
                    if (p > 1e-10)
                        entropy -= p * Math.Log2(p);
                }
                return entropy;
            }

            /// <summary>
            /// Computes the spectral norm of the Laplacian.
            /// </summary>
            public double SpectralNorm()
            {
                return Eigenvalues.Length > 0 ? Eigenvalues.Max(Math.Abs) : 0;
            }

            /// <inheritdoc/>
            public override string ToString() =>
                $"SpectralSig(Dim={Eigenvalues.Length}, Fiedler={FiedlerValue:F4}, Components={ConnectedComponents})";
        }

        /// <summary>
        /// Represents a persistent homology barcode for topological data analysis
        /// of genome structure.
        /// </summary>
        public readonly struct PersistentBarcode
        {
            /// <summary>Bars in the barcode (birth, death, dimension).</summary>
            public readonly ImmutableArray<PersistentBar> Bars;

            /// <summary>Number of bars (features).</summary>
            public int Count => Bars.Length;

            /// <summary>
            /// Initializes a new PersistentBarcode.
            /// </summary>
            public PersistentBarcode(ImmutableArray<PersistentBar> bars)
            {
                Bars = bars.Sort((a, b) => a.Birth.CompareTo(b.Birth));
            }

            /// <summary>
            /// Gets the total persistence (sum of lifetimes).
            /// </summary>
            public double TotalPersistence()
            {
                return Bars.Sum(b => b.Lifetime);
            }

            /// <summary>
            /// Gets the maximum persistence.
            /// </summary>
            public double MaxPersistence()
            {
                return Bars.Length > 0 ? Bars.Max(b => b.Lifetime) : 0;
            }

            /// <summary>
            /// Gets the persistence entropy.
            /// </summary>
            public double PersistenceEntropy()
            {
                double total = TotalPersistence();
                if (Math.Abs(total) < 1e-10)
                    return 0;

                double entropy = 0;
                foreach (var bar in Bars)
                {
                    double p = bar.Lifetime / total;
                    if (p > 1e-10)
                        entropy -= p * Math.Log2(p);
                }
                return entropy;
            }

            /// <summary>
            /// Computes the bottleneck distance to another barcode.
            /// </summary>
            /// <param name="other">Other barcode.</param>
            /// <returns>Bottleneck distance.</returns>
            public double BottleneckDistance(PersistentBarcode other)
            {
                double maxDist = 0;

                foreach (var barA in Bars)
                {
                    double minDist = double.MaxValue;
                    foreach (var barB in other.Bars)
                    {
                        double dist = Math.Max(
                            Math.Abs(barA.Birth - barB.Birth),
                            Math.Abs(barA.Death - barB.Death));
                        minDist = Math.Min(minDist, dist);
                    }
                    maxDist = Math.Max(maxDist, minDist);
                }

                foreach (var barB in other.Bars)
                {
                    double minDist = double.MaxValue;
                    foreach (var barA in Bars)
                    {
                        double dist = Math.Max(
                            Math.Abs(barA.Birth - barB.Birth),
                            Math.Abs(barA.Death - barB.Death));
                        minDist = Math.Min(minDist, dist);
                    }
                    maxDist = Math.Max(maxDist, minDist);
                }

                return maxDist;
            }

            /// <summary>
            /// Computes the Wasserstein distance to another barcode.
            /// </summary>
            /// <param name="other">Other barcode.</param>
            /// <param name="p">Exponent for the Wasserstein distance.</param>
            /// <returns>Wasserstein-p distance.</returns>
            public double WassersteinDistance(PersistentBarcode other, double p = 2.0)
            {
                if (Bars.Length == 0 && other.Bars.Length == 0)
                    return 0;

                var sortedA = Bars.OrderByDescending(b => b.Lifetime).ToList();
                var sortedB = other.Bars.OrderByDescending(b => b.Lifetime).ToList();

                int maxCount = Math.Max(sortedA.Count, sortedB.Count);
                double distance = 0;

                for (int i = 0; i < maxCount; i++)
                {
                    double birthA = i < sortedA.Count ? sortedA[i].Birth : 0;
                    double deathA = i < sortedA.Count ? sortedA[i].Death : 0;
                    double birthB = i < sortedB.Count ? sortedB[i].Birth : 0;
                    double deathB = i < sortedB.Count ? sortedB[i].Death : 0;

                    double barDist = Math.Max(
                        Math.Abs(birthA - birthB),
                        Math.Abs(deathA - deathB));

                    distance += Math.Pow(barDist, p);
                }

                return Math.Pow(distance, 1.0 / p);
            }

            /// <inheritdoc/>
            public override string ToString() =>
                $"PersistentBarcode(Bars={Count}, TotalPersistence={TotalPersistence():F4})";
        }

        /// <summary>
        /// A single bar in a persistent barcode.
        /// </summary>
        public readonly struct PersistentBar : IEquatable<PersistentBar>
        {
            /// <summary>Birth time.</summary>
            public readonly double Birth;

            /// <summary>Death time.</summary>
            public readonly double Death;

            /// <summary>Topological dimension (0=components, 1=loops, etc.).</summary>
            public readonly int Dimension;

            /// <summary>Lifetime of this bar.</summary>
            public double Lifetime => Death - Birth;

            /// <summary>
            /// Initializes a new PersistentBar.
            /// </summary>
            public PersistentBar(double birth, double death, int dimension = 0)
            {
                Birth = birth;
                Death = death;
                Dimension = dimension;
            }

            /// <summary>Determines if this bar is infinite (never dies).</summary>
            public bool IsInfinite => double.IsPositiveInfinity(Death);

            /// <inheritdoc/>
            public bool Equals(PersistentBar other) =>
                Birth == other.Birth && Death == other.Death && Dimension == other.Dimension;

            /// <inheritdoc/>
            public override bool Equals(object? obj) => obj is PersistentBar b && Equals(b);

            /// <inheritdoc/>
            public override int GetHashCode() => HashCode.Combine(Birth, Death, Dimension);

            /// <inheritdoc/>
            public override string ToString() =>
                $"Bar({Dimension}D: [{Birth:F4}, {Death:F4}], Life={Lifetime:F4})";
        }

        /// <summary>
        /// Represents the result of a topological analysis of a genome.
        /// </summary>
        public readonly struct TopologicalAnalysisResult
        {
            /// <summary>Spectral signature of the genome.</summary>
            public readonly SpectralSignature SpectralSignature;

            /// <summary>Persistent barcode of the genome.</summary>
            public readonly PersistentBarcode Barcode;

            /// <summary>Number of connected components.</summary>
            public readonly int ConnectedComponents;

            /// <summary>Number of cycles.</summary>
            public readonly int CycleCount;

            /// <summary>Betti numbers (beta_0, beta_1, ...).</summary>
            public readonly ImmutableArray<int> BettiNumbers;

            /// <summary>Euler characteristic.</summary>
            public double EulerCharacteristic =>
                BettiNumbers.Length > 0
                    ? BettiNumbers.Select((b, i) => i % 2 == 0 ? b : -b).Sum()
                    : 0;

            /// <summary>
            /// Initializes a new TopologicalAnalysisResult.
            /// </summary>
            public TopologicalAnalysisResult(
                SpectralSignature spectralSignature,
                PersistentBarcode barcode,
                int connectedComponents,
                int cycleCount,
                ImmutableArray<int> bettiNumbers)
            {
                SpectralSignature = spectralSignature;
                Barcode = barcode;
                ConnectedComponents = connectedComponents;
                CycleCount = cycleCount;
                BettiNumbers = bettiNumbers;
            }

            /// <inheritdoc/>
            public override string ToString() =>
                $"TopoAnalysis(Components={ConnectedComponents}, Cycles={CycleCount}, " +
                $"Euler={EulerCharacteristic}, Persistence={Barcode.TotalPersistence():F4})";
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Topology Analyzer

    /// <summary>
    /// Analyzes the topological structure of genomes using spectral graph theory
    /// and persistent homology. Provides deep structural insights for evolution.
    /// </summary>
    public sealed class TopologyAnalyzer
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the TopologyAnalyzer class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public TopologyAnalyzer(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Performs a complete topological analysis of a genome.
        /// </summary>
        /// <param name="genome">The genome to analyze.</param>
        /// <returns>A comprehensive topological analysis result.</returns>
        public TopologicalAnalysisResult Analyze(GeoGenome genome)
        {
            var adjMatrix = BuildAdjacencyMatrix(genome);
            int n = adjMatrix.GetLength(0);

            var degreeMatrix = BuildDegreeMatrix(adjMatrix);
            var laplacian = ComputeLaplacian(adjMatrix, degreeMatrix);
            var eigenvalues = ComputeEigenvalues(laplacian);
            var spectralSignature = new SpectralSignature(eigenvalues.ToImmutableArray());

            var barcode = ComputePersistentBarcode(genome);
            int connectedComponents = CountConnectedComponents(genome);
            int cycleCount = CountCycles(genome);
            var bettiNumbers = ComputeBettiNumbers(genome);

            return new TopologicalAnalysisResult(
                spectralSignature,
                barcode,
                connectedComponents,
                cycleCount,
                bettiNumbers);
        }

        /// <summary>
        /// Builds the adjacency matrix for a genome's graph structure.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>The adjacency matrix.</returns>
        public double[,] BuildAdjacencyMatrix(GeoGenome genome)
        {
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            int n = activeNeurons.Count;

            var indexMap = new Dictionary<long, int>();
            for (int i = 0; i < n; i++)
                indexMap[activeNeurons[i].InnovationNumber] = i;

            var matrix = new double[n, n];

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (indexMap.TryGetValue(synapse.SourceNeuronId, out int srcIdx) &&
                    indexMap.TryGetValue(synapse.TargetNeuronId, out int tgtIdx))
                {
                    matrix[srcIdx, tgtIdx] = Math.Abs(synapse.Weight);
                    matrix[tgtIdx, srcIdx] = Math.Abs(synapse.Weight);
                }
            }

            return matrix;
        }

        /// <summary>
        /// Builds the degree matrix from an adjacency matrix.
        /// </summary>
        /// <param name="adjMatrix">Adjacency matrix.</param>
        /// <returns>Degree matrix.</returns>
        public double[,] BuildDegreeMatrix(double[,] adjMatrix)
        {
            int n = adjMatrix.GetLength(0);
            var degreeMatrix = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                double degree = 0;
                for (int j = 0; j < n; j++)
                    degree += adjMatrix[i, j];
                degreeMatrix[i, i] = degree;
            }

            return degreeMatrix;
        }

        /// <summary>
        /// Computes the graph Laplacian matrix (L = D - A).
        /// </summary>
        /// <param name="adjMatrix">Adjacency matrix.</param>
        /// <param name="degreeMatrix">Degree matrix.</param>
        /// <returns>Laplacian matrix.</returns>
        public double[,] ComputeLaplacian(double[,] adjMatrix, double[,] degreeMatrix)
        {
            int n = adjMatrix.GetLength(0);
            var laplacian = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    laplacian[i, j] = degreeMatrix[i, j] - adjMatrix[i, j];
                }
            }

            return laplacian;
        }

        /// <summary>
        /// Computes eigenvalues of the Laplacian matrix using the power iteration method.
        /// </summary>
        /// <param name="laplacian">The Laplacian matrix.</param>
        /// <returns>Sorted eigenvalues.</returns>
        public double[] ComputeEigenvalues(double[,] laplacian)
        {
            int n = laplacian.GetLength(0);
            if (n == 0)
                return Array.Empty<double>();

            var eigenvalues = new List<double>();
            var matrix = (double[,])laplacian.Clone();

            for (int iter = 0; iter < Math.Min(100, n * 10); iter++)
            {
                double maxVal = 0;
                int maxRow = 0, maxCol = 0;

                for (int i = 0; i < n; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        if (Math.Abs(matrix[i, j]) > maxVal)
                        {
                            maxVal = Math.Abs(matrix[i, j]);
                            maxRow = i;
                            maxCol = j;
                        }
                    }
                }

                if (maxVal < 1e-10)
                    break;

                double theta = 0.5 * Math.Atan2(
                    2 * matrix[maxRow, maxCol],
                    matrix[maxRow, maxRow] - matrix[maxCol, maxCol]);

                double c = Math.Cos(theta);
                double s = Math.Sin(theta);

                var newMatrix = (double[,])matrix.Clone();

                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (i == maxRow || i == maxCol || j == maxRow || j == maxCol)
                        {
                            double a = matrix[maxRow, j];
                            double b = matrix[maxCol, j];
                            newMatrix[i, j] = c * a + s * b;
                            if (i == maxRow || i == maxCol)
                            {
                                double a2 = matrix[i, maxRow];
                                double b2 = matrix[i, maxCol];
                                newMatrix[i, j] = c * a2 + s * b2;
                                if (j == maxRow || j == maxCol)
                                {
                                    newMatrix[maxRow, j] = c * c * matrix[maxRow, j] +
                                        2 * s * c * matrix[maxCol, j] +
                                        s * s * matrix[maxCol, j];
                                }
                            }
                        }
                    }
                }

                matrix = newMatrix;
            }

            for (int i = 0; i < n; i++)
            {
                eigenvalues.Add(matrix[i, i]);
            }

            eigenvalues.Sort();
            return eigenvalues.ToArray();
        }

        /// <summary>
        /// Counts connected components in the genome's graph using BFS.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Number of connected components.</returns>
        public int CountConnectedComponents(GeoGenome genome)
        {
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            if (activeNeurons.Count == 0)
                return 0;

            var visited = new HashSet<long>();
            int components = 0;

            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            foreach (var neuron in activeNeurons)
            {
                if (visited.Contains(neuron.InnovationNumber))
                    continue;

                components++;
                var queue = new Queue<long>();
                queue.Enqueue(neuron.InnovationNumber);

                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (!visited.Add(current))
                        continue;

                    foreach (var synapse in activeSynapses)
                    {
                        if (synapse.SourceNeuronId == current && !visited.Contains(synapse.TargetNeuronId))
                            queue.Enqueue(synapse.TargetNeuronId);
                        if (synapse.TargetNeuronId == current && !visited.Contains(synapse.SourceNeuronId))
                            queue.Enqueue(synapse.SourceNeuronId);
                    }
                }
            }

            return components;
        }

        /// <summary>
        /// Counts the number of independent cycles in the genome's graph.
        /// Uses the formula: cycles = edges - vertices + components.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Number of independent cycles.</returns>
        public int CountCycles(GeoGenome genome)
        {
            int vertices = genome.ActiveNeuronCount;
            int edges = genome.ActiveSynapseCount;
            int components = CountConnectedComponents(genome);

            return Math.Max(0, edges - vertices + components);
        }

        /// <summary>
        /// Computes Betti numbers for the genome's topology.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Immutable array of Betti numbers.</returns>
        public ImmutableArray<int> ComputeBettiNumbers(GeoGenome genome)
        {
            int beta0 = CountConnectedComponents(genome);
            int beta1 = CountCycles(genome);

            var betti = ImmutableArray.CreateBuilder<int>();
            betti.Add(beta0);
            betti.Add(beta1);

            for (int i = 2; i < genome.MaxLayerDepth + 1; i++)
            {
                betti.Add(0);
            }

            return betti.MoveToImmutable();
        }

        /// <summary>
        /// Computes a persistent barcode for the genome using a filtration
        /// based on connection weight magnitudes.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Persistent barcode.</returns>
        public PersistentBarcode ComputePersistentBarcode(GeoGenome genome)
        {
            var bars = ImmutableArray.CreateBuilder<PersistentBar>();

            var components = FindConnectedComponentsWithThreshold(genome, 0);
            foreach (var component in components)
            {
                bars.Add(new PersistentBar(0, double.PositiveInfinity, 0));
            }

            var cycles = FindCyclesWithFiltration(genome);
            foreach (var cycle in cycles)
            {
                bars.Add(new PersistentBar(cycle.Birth, cycle.Death, 1));
            }

            return new PersistentBarcode(bars.MoveToImmutable());
        }

        private List<List<long>> FindConnectedComponentsWithThreshold(GeoGenome genome, double weightThreshold)
        {
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            var visited = new HashSet<long>();
            var components = new List<List<long>>();

            var activeSynapses = genome.Synapses
                .Where(s => s.IsActive && Math.Abs(s.Weight) >= weightThreshold)
                .ToList();

            foreach (var neuron in activeNeurons)
            {
                if (visited.Contains(neuron.InnovationNumber))
                    continue;

                var component = new List<long>();
                var queue = new Queue<long>();
                queue.Enqueue(neuron.InnovationNumber);

                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (!visited.Add(current))
                        continue;
                    component.Add(current);

                    foreach (var synapse in activeSynapses)
                    {
                        if (synapse.SourceNeuronId == current && !visited.Contains(synapse.TargetNeuronId))
                            queue.Enqueue(synapse.TargetNeuronId);
                        if (synapse.TargetNeuronId == current && !visited.Contains(synapse.SourceNeuronId))
                            queue.Enqueue(synapse.SourceNeuronId);
                    }
                }

                components.Add(component);
            }

            return components;
        }

        private List<(double Birth, double Death)> FindCyclesWithFiltration(GeoGenome genome)
        {
            var cycles = new List<(double Birth, double Death)>();
            var activeSynapses = genome.Synapses
                .Where(s => s.IsActive)
                .OrderBy(s => Math.Abs(s.Weight))
                .ToList();

            var unionFind = new Dictionary<long, long>();
            var rank = new Dictionary<long, int>();

            foreach (var neuron in genome.Neurons.Where(n => n.IsActive))
            {
                unionFind[neuron.InnovationNumber] = neuron.InnovationNumber;
                rank[neuron.InnovationNumber] = 0;
            }

            foreach (var synapse in activeSynapses)
            {
                long rootSource = FindRoot(unionFind, synapse.SourceNeuronId);
                long rootTarget = FindRoot(unionFind, synapse.TargetNeuronId);

                if (rootSource != rootTarget)
                {
                    if (rank[rootSource] < rank[rootTarget])
                        (rootSource, rootTarget) = (rootTarget, rootSource);
                    unionFind[rootTarget] = rootSource;
                    if (rank[rootSource] == rank[rootTarget])
                        rank[rootSource]++;
                }
                else
                {
                    double weight = Math.Abs(synapse.Weight);
                    cycles.Add((weight, weight * 2));
                }
            }

            return cycles;
        }

        private long FindRoot(Dictionary<long, long> unionFind, long node)
        {
            while (unionFind.TryGetValue(node, out var parent) && parent != node)
            {
                unionFind[node] = unionFind.TryGetValue(parent, out var grandParent) ? grandParent : parent;
                node = parent;
            }
            return node;
        }

        /// <summary>
        /// Computes the spectral distance between two genomes.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Spectral distance.</returns>
        public double ComputeSpectralDistance(GeoGenome a, GeoGenome b)
        {
            var analysisA = Analyze(a);
            var analysisB = Analyze(b);
            return analysisA.SpectralSignature.SpectralDistance(analysisB.SpectralSignature);
        }

        /// <summary>
        /// Computes the topological similarity between two genomes.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Topological similarity (0-1).</returns>
        public double ComputeTopologicalSimilarity(GeoGenome a, GeoGenome b)
        {
            var analysisA = Analyze(a);
            var analysisB = Analyze(b);

            double componentSimilarity = analysisA.ConnectedComponents == analysisB.ConnectedComponents
                ? 1.0
                : 1.0 / (1.0 + Math.Abs(analysisA.ConnectedComponents - analysisB.ConnectedComponents));

            double cycleSimilarity = analysisA.CycleCount == analysisB.CycleCount
                ? 1.0
                : 1.0 / (1.0 + Math.Abs(analysisA.CycleCount - analysisB.CycleCount));

            double spectralSim = Math.Exp(-analysisA.SpectralSignature.SpectralDistance(analysisB.SpectralSignature));

            double barcodeSim = Math.Exp(-analysisA.Barcode.WassersteinDistance(analysisB.Barcode));

            return 0.25 * componentSimilarity + 0.25 * cycleSimilarity +
                   0.25 * spectralSim + 0.25 * barcodeSim;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Genome Serializer

    /// <summary>
    /// Handles serialization and deserialization of genomes for persistence,
    /// transfer, and analysis. Supports JSON and binary formats.
    /// </summary>
    public sealed class GenomeSerializer
    {
        private readonly JsonSerializerOptions _jsonOptions;

        /// <summary>
        /// Initializes a new instance of the GenomeSerializer class.
        /// </summary>
        public GenomeSerializer()
        {
            _jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = false,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
                PropertyNameCaseInsensitive = true
            };
        }

        /// <summary>
        /// Serializes a genome to a JSON string.
        /// </summary>
        /// <param name="genome">The genome to serialize.</param>
        /// <returns>JSON string representation.</returns>
        public string SerializeToJson(GeoGenome genome)
        {
            var data = new GenomeData
            {
                Id = genome.Id.ToString(),
                Generation = genome.Generation,
                SpeciesId = genome.SpeciesId,
                Age = genome.Age,
                InputCount = genome.InputCount,
                OutputCount = genome.OutputCount,
                Fitness = genome.Fitness,
                BestFitness = genome.BestFitness,
                Neurons = genome.Neurons.Select(n => new NeuronData
                {
                    InnovationNumber = n.InnovationNumber,
                    LayerIndex = n.LayerIndex,
                    PositionInLayer = n.PositionInLayer,
                    Activation = (int)n.Activation,
                    Bias = n.Bias,
                    IsActive = n.IsActive,
                    SemanticRole = n.SemanticRole,
                    CreationGeneration = n.CreationGeneration,
                    ExpressionCount = n.ExpressionCount
                }).ToList(),
                Synapses = genome.Synapses.Select(s => new SynapseData
                {
                    InnovationNumber = s.InnovationNumber,
                    SourceNeuronId = s.SourceNeuronId,
                    TargetNeuronId = s.TargetNeuronId,
                    Weight = s.Weight,
                    IsActive = s.IsActive,
                    IsRecurrent = s.IsRecurrent,
                    RecurrentDelay = s.RecurrentDelay,
                    CreationGeneration = s.CreationGeneration,
                    Confidence = s.Confidence,
                    SemanticRole = s.SemanticRole
                }).ToList(),
                ParentIds = genome.ParentIds.Select(id => id.ToString()).ToList()
            };

            return JsonSerializer.Serialize(data, _jsonOptions);
        }

        /// <summary>
        /// Deserializes a genome from a JSON string.
        /// </summary>
        /// <param name="json">JSON string representation.</param>
        /// <returns>The deserialized genome.</returns>
        public GeoGenome DeserializeFromJson(string json)
        {
            var data = JsonSerializer.Deserialize<GenomeData>(json, _jsonOptions)
                ?? throw new ArgumentException("Invalid JSON data for genome deserialization.");

            var genome = new GeoGenome
            {
                Id = Guid.Parse(data.Id),
                Generation = data.Generation,
                SpeciesId = data.SpeciesId,
                Age = data.Age,
                InputCount = data.InputCount,
                OutputCount = data.OutputCount,
                Fitness = data.Fitness,
                BestFitness = data.BestFitness,
                ParentIds = data.ParentIds != null
                    ? data.ParentIds.Select(id => Guid.Parse(id)).ToImmutableArray()
                    : ImmutableArray<Guid>.Empty
            };

            foreach (var nData in data.Neurons)
            {
                genome.Neurons.Add(new GeoNeuron
                {
                    InnovationNumber = nData.InnovationNumber,
                    LayerIndex = nData.LayerIndex,
                    PositionInLayer = nData.PositionInLayer,
                    Activation = (ActivationFunction)nData.Activation,
                    Bias = nData.Bias,
                    IsActive = nData.IsActive,
                    SemanticRole = nData.SemanticRole,
                    CreationGeneration = nData.CreationGeneration,
                    ExpressionCount = nData.ExpressionCount
                });
            }

            foreach (var sData in data.Synapses)
            {
                genome.Synapses.Add(new GeoSynapse
                {
                    InnovationNumber = sData.InnovationNumber,
                    SourceNeuronId = sData.SourceNeuronId,
                    TargetNeuronId = sData.TargetNeuronId,
                    Weight = sData.Weight,
                    IsActive = sData.IsActive,
                    IsRecurrent = sData.IsRecurrent,
                    RecurrentDelay = sData.RecurrentDelay,
                    CreationGeneration = sData.CreationGeneration,
                    Confidence = sData.Confidence,
                    SemanticRole = sData.SemanticRole
                });
            }

            return genome;
        }

        /// <summary>
        /// Serializes a genome to a compact binary format.
        /// </summary>
        /// <param name="genome">The genome to serialize.</param>
        /// <returns>Binary data as byte array.</returns>
        public byte[] SerializeToBinary(GeoGenome genome)
        {
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            writer.Write(genome.Id.ToByteArray());
            writer.Write(genome.Generation);
            writer.Write(genome.SpeciesId);
            writer.Write(genome.Age);
            writer.Write(genome.InputCount);
            writer.Write(genome.OutputCount);
            writer.Write(genome.Fitness);

            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            writer.Write(activeNeurons.Count);
            foreach (var neuron in activeNeurons)
            {
                writer.Write(neuron.InnovationNumber);
                writer.Write(neuron.LayerIndex);
                writer.Write(neuron.PositionInLayer);
                writer.Write((int)neuron.Activation);
                writer.Write(neuron.Bias);
            }

            writer.Write(activeSynapses.Count);
            foreach (var synapse in activeSynapses)
            {
                writer.Write(synapse.InnovationNumber);
                writer.Write(synapse.SourceNeuronId);
                writer.Write(synapse.TargetNeuronId);
                writer.Write(synapse.Weight);
                writer.Write(synapse.IsRecurrent);
            }

            return ms.ToArray();
        }

        /// <summary>
        /// Deserializes a genome from binary data.
        /// </summary>
        /// <param name="data">Binary data.</param>
        /// <returns>The deserialized genome.</returns>
        public GeoGenome DeserializeFromBinary(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);

            var genome = new GeoGenome
            {
                Id = new Guid(reader.ReadBytes(16)),
                Generation = reader.ReadInt32(),
                SpeciesId = reader.ReadInt32(),
                Age = reader.ReadInt32(),
                InputCount = reader.ReadInt32(),
                OutputCount = reader.ReadInt32(),
                Fitness = reader.ReadDouble()
            };

            int neuronCount = reader.ReadInt32();
            for (int i = 0; i < neuronCount; i++)
            {
                genome.Neurons.Add(new GeoNeuron
                {
                    InnovationNumber = reader.ReadInt64(),
                    LayerIndex = reader.ReadInt32(),
                    PositionInLayer = reader.ReadInt32(),
                    Activation = (ActivationFunction)reader.ReadInt32(),
                    Bias = reader.ReadDouble(),
                    IsActive = true
                });
            }

            int synapseCount = reader.ReadInt32();
            for (int i = 0; i < synapseCount; i++)
            {
                genome.Synapses.Add(new GeoSynapse
                {
                    InnovationNumber = reader.ReadInt64(),
                    SourceNeuronId = reader.ReadInt64(),
                    TargetNeuronId = reader.ReadInt64(),
                    Weight = reader.ReadDouble(),
                    IsActive = true,
                    IsRecurrent = reader.ReadBoolean()
                });
            }

            return genome;
        }

        /// <summary>
        /// Serializes a population to JSON.
        /// </summary>
        /// <param name="population">The population.</param>
        /// <returns>JSON string.</returns>
        public string SerializePopulationToJson(GenomePopulation population)
        {
            var genomeJsons = population.Genomes.Select(SerializeToJson).ToList();
            return JsonSerializer.Serialize(new
            {
                Generation = population.GenerationNumber,
                Count = population.Genomes.Length,
                Genomes = genomeJsons
            }, _jsonOptions);
        }

        /// <summary>
        /// Computes a checksum for a genome for integrity verification.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Checksum as a hex string.</returns>
        public string ComputeChecksum(GeoGenome genome)
        {
            var json = SerializeToJson(genome);
            using var sha256 = System.Security.Cryptography.SHA256.Create();
            var hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(json));
            return Convert.ToHexString(hash);
        }

        /// <summary>
        /// Internal data class for JSON serialization.
        /// </summary>
        private sealed class GenomeData
        {
            public string Id { get; set; } = string.Empty;
            public int Generation { get; set; }
            public int SpeciesId { get; set; }
            public int Age { get; set; }
            public int InputCount { get; set; }
            public int OutputCount { get; set; }
            public double Fitness { get; set; }
            public double BestFitness { get; set; }
            public List<NeuronData> Neurons { get; set; } = new();
            public List<SynapseData> Synapses { get; set; } = new();
            public List<string> ParentIds { get; set; } = new();
        }

        private sealed class NeuronData
        {
            public long InnovationNumber { get; set; }
            public int LayerIndex { get; set; }
            public int PositionInLayer { get; set; }
            public int Activation { get; set; }
            public double Bias { get; set; }
            public bool IsActive { get; set; }
            public string? SemanticRole { get; set; }
            public int CreationGeneration { get; set; }
            public int ExpressionCount { get; set; }
        }

        private sealed class SynapseData
        {
            public long InnovationNumber { get; set; }
            public long SourceNeuronId { get; set; }
            public long TargetNeuronId { get; set; }
            public double Weight { get; set; }
            public bool IsActive { get; set; }
            public bool IsRecurrent { get; set; }
            public int RecurrentDelay { get; set; }
            public int CreationGeneration { get; set; }
            public double Confidence { get; set; }
            public string? SemanticRole { get; set; }
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Population Seed Strategies

    /// <summary>
    /// Provides different strategies for seeding initial populations.
    /// Each strategy biases the initial population differently to aid evolution.
    /// </summary>
    public sealed class PopulationSeedStrategy
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the PopulationSeedStrategy class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public PopulationSeedStrategy(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Creates a seed population with varying connectivity densities.
        /// Some genomes are fully connected, others are minimal.
        /// </summary>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="populationSize">Population size.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>List of seeded genomes.</returns>
        public IReadOnlyList<GeoGenome> CreateDensitySeededPopulation(
            int inputCount, int outputCount, int populationSize, Random rng)
        {
            var genomes = new List<GeoGenome>();
            var manager = new GenomePopulationManager(_config, rng);

            for (int i = 0; i < populationSize; i++)
            {
                double densityTarget = (double)i / populationSize;
                var genome = manager.CreateRandomGenome(inputCount, outputCount, 0);

                var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();
                int targetActive = Math.Max(1, (int)(activeSynapses.Count * densityTarget));

                var toDeactivate = activeSynapses
                    .OrderBy(_ => rng.NextDouble())
                    .Skip(targetActive)
                    .ToList();

                foreach (var synapse in toDeactivate)
                {
                    synapse.IsActive = false;
                }

                genome.ComputeComplexity();
                genomes.Add(genome);
            }

            return genomes;
        }

        /// <summary>
        /// Creates a seed population with varying activation functions.
        /// Each genome uses a different mix of activation functions.
        /// </summary>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="populationSize">Population size.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>List of seeded genomes.</returns>
        public IReadOnlyList<GeoGenome> CreateActivationSeededPopulation(
            int inputCount, int outputCount, int populationSize, Random rng)
        {
            var genomes = new List<GeoGenome>();
            var manager = new GenomePopulationManager(_config, rng);
            var allActivations = Enum.GetValues<ActivationFunction>();

            for (int i = 0; i < populationSize; i++)
            {
                var genome = manager.CreateRandomGenome(inputCount, outputCount, 0);

                double bias = (double)i / populationSize;
                var dominantActivation = allActivations[(int)(bias * (allActivations.Length - 1))];

                foreach (var neuron in genome.Neurons.Where(n => n.LayerIndex > 0))
                {
                    if (rng.NextDouble() < 0.7)
                    {
                        neuron.Activation = dominantActivation;
                    }
                }

                genome.ComputeComplexity();
                genomes.Add(genome);
            }

            return genomes;
        }

        /// <summary>
        /// Creates a seed population with a Lamarckian bias - some genomes are pre-optimized
        /// with heuristic weight initialization.
        /// </summary>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="populationSize">Population size.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>List of seeded genomes.</returns>
        public IReadOnlyList<GeoGenome> CreateLamarckianSeedPopulation(
            int inputCount, int outputCount, int populationSize, Random rng)
        {
            var genomes = new List<GeoGenome>();
            var manager = new GenomePopulationManager(_config, rng);

            int optimizedCount = populationSize / 10;

            for (int i = 0; i < populationSize; i++)
            {
                var genome = manager.CreateRandomGenome(inputCount, outputCount, 0);

                if (i < optimizedCount)
                {
                    foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
                    {
                        double XavierInit = rng.NextDouble() * 2.0 - 1.0;
                        XavierInit *= Math.Sqrt(2.0 / (inputCount + outputCount));
                        synapse.Weight = XavierInit;
                    }

                    foreach (var neuron in genome.Neurons.Where(n => n.LayerIndex > 0))
                    {
                        neuron.Bias = 0.01 * (rng.NextDouble() * 2 - 1);
                    }
                }

                genome.ComputeComplexity();
                genomes.Add(genome);
            }

            return genomes;
        }

        /// <summary>
        /// Creates a seed population with hierarchical structure.
        /// Genomes have varying numbers of hidden layers (0 to max).
        /// </summary>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="populationSize">Population size.</param>
        /// <param name="maxHiddenLayers">Maximum number of hidden layers.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>List of seeded genomes.</returns>
        public IReadOnlyList<GeoGenome> CreateHierarchicalSeedPopulation(
            int inputCount, int outputCount, int populationSize,
            int maxHiddenLayers, Random rng)
        {
            var genomes = new List<GeoGenome>();
            var allActivations = Enum.GetValues<ActivationFunction>();

            for (int i = 0; i < populationSize; i++)
            {
                int hiddenLayers = (int)((double)i / populationSize * maxHiddenLayers) + 1;
                hiddenLayers = Math.Min(hiddenLayers, maxHiddenLayers);

                int neuronsPerLayer = Math.Max(2, (inputCount + outputCount) / 2);

                var genome = new GeoGenome
                {
                    Id = Guid.NewGuid(),
                    Generation = 0,
                    InputCount = inputCount,
                    OutputCount = outputCount
                };

                long nextInnov = 1;

                for (int inp = 0; inp < inputCount; inp++)
                {
                    genome.Neurons.Add(new GeoNeuron
                    {
                        InnovationNumber = nextInnov++,
                        LayerIndex = 0,
                        PositionInLayer = inp,
                        Activation = ActivationFunction.Linear,
                        Bias = 0,
                        IsActive = true
                    });
                }

                int currentLayer = 1;
                var prevLayerNeurons = genome.Neurons.Where(n => n.LayerIndex == 0).ToList();

                for (int hl = 0; hl < hiddenLayers; hl++)
                {
                    var layerNeurons = new List<GeoNeuron>();
                    int layerSize = Math.Max(2, neuronsPerLayer - hl);

                    for (int n = 0; n < layerSize; n++)
                    {
                        var neuron = new GeoNeuron
                        {
                            InnovationNumber = nextInnov++,
                            LayerIndex = currentLayer,
                            PositionInLayer = n,
                            Activation = allActivations[rng.Next(allActivations.Length)],
                            Bias = (rng.NextDouble() * 2 - 1) * _config.BiasInitRange,
                            IsActive = true
                        };
                        genome.Neurons.Add(neuron);
                        layerNeurons.Add(neuron);
                    }

                    long nextSynInnov = genome.Synapses.Count > 0
                        ? genome.Synapses.Max(s => s.InnovationNumber) + 1
                        : nextInnov;

                    foreach (var prev in prevLayerNeurons)
                    {
                        foreach (var curr in layerNeurons)
                        {
                            genome.Synapses.Add(new GeoSynapse
                            {
                                InnovationNumber = nextSynInnov++,
                                SourceNeuronId = prev.InnovationNumber,
                                TargetNeuronId = curr.InnovationNumber,
                                Weight = (rng.NextDouble() * 2 - 1) * _config.WeightInitRange,
                                IsActive = true
                            });
                        }
                    }

                    nextInnov = nextSynInnov;
                    prevLayerNeurons = layerNeurons;
                    currentLayer++;
                }

                var outputNeurons = new List<GeoNeuron>();
                for (int outp = 0; outp < outputCount; outp++)
                {
                    var neuron = new GeoNeuron
                    {
                        InnovationNumber = nextInnov++,
                        LayerIndex = currentLayer,
                        PositionInLayer = outp,
                        Activation = allActivations[rng.Next(allActivations.Length)],
                        Bias = (rng.NextDouble() * 2 - 1) * _config.BiasInitRange,
                        IsActive = true
                    };
                    genome.Neurons.Add(neuron);
                    outputNeurons.Add(neuron);
                }

                long finalSynInnov = genome.Synapses.Count > 0
                    ? genome.Synapses.Max(s => s.InnovationNumber) + 1
                    : nextInnov;

                foreach (var prev in prevLayerNeurons)
                {
                    foreach (var curr in outputNeurons)
                    {
                        genome.Synapses.Add(new GeoSynapse
                        {
                            InnovationNumber = finalSynInnov++,
                            SourceNeuronId = prev.InnovationNumber,
                            TargetNeuronId = curr.InnovationNumber,
                            Weight = (rng.NextDouble() * 2 - 1) * _config.WeightInitRange,
                            IsActive = true
                        });
                    }
                }

                genome.ComputeComplexity();
                genomes.Add(genome);
            }

            return genomes;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Fitness Function Composables

    /// <summary>
    /// Provides composable fitness function building blocks for creating
    /// custom multi-objective fitness evaluations.
    /// </summary>
    public static class FitnessFunctions
    {
        /// <summary>
        /// Computes mean squared error between output and target.
        /// </summary>
        /// <param name="output">Output values.</param>
        /// <param name="target">Target values.</param>
        /// <returns>MSE value (lower is better).</returns>
        public static double MeanSquaredError(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < length; i++)
            {
                double diff = output[i] - target[i];
                sum += diff * diff;
            }
            return sum / length;
        }

        /// <summary>
        /// Computes root mean squared error.
        /// </summary>
        public static double RootMeanSquaredError(double[] output, ImmutableArray<double> target)
        {
            return Math.Sqrt(MeanSquaredError(output, target));
        }

        /// <summary>
        /// Computes mean absolute error.
        /// </summary>
        public static double MeanAbsoluteError(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < length; i++)
            {
                sum += Math.Abs(output[i] - target[i]);
            }
            return sum / length;
        }

        /// <summary>
        /// Computes negative log likelihood loss.
        /// </summary>
        public static double NegativeLogLikelihood(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < length; i++)
            {
                double pred = Math.Clamp(output[i], 1e-7, 1 - 1e-7);
                double t = i < target.Length ? target[i] : 0;
                sum -= t * Math.Log(pred) + (1 - t) * Math.Log(1 - pred);
            }
            return sum / length;
        }

        /// <summary>
        /// Computes cross-entropy loss for multi-class output.
        /// </summary>
        public static double CrossEntropyLoss(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < length; i++)
            {
                double pred = Math.Max(output[i], 1e-7);
                double t = i < target.Length ? target[i] : 0;
                sum -= t * Math.Log(pred);
            }
            return sum / length;
        }

        /// <summary>
        /// Computes cosine similarity between output and target.
        /// </summary>
        public static double CosineSimilarity(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length == 0)
                return 0;

            double dotProduct = 0, normA = 0, normB = 0;
            for (int i = 0; i < length; i++)
            {
                dotProduct += output[i] * target[i];
                normA += output[i] * output[i];
                normB += target[i] * target[i];
            }

            double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
            return denominator > 1e-10 ? dotProduct / denominator : 0;
        }

        /// <summary>
        /// Computes signal-to-noise ratio of the output.
        /// </summary>
        public static double SignalToNoiseRatio(double[] output)
        {
            if (output.Length == 0)
                return 0;

            double mean = output.Average();
            double signalPower = mean * mean;
            double noisePower = output.Average(v => (v - mean) * (v - mean));

            return noisePower > 1e-10 ? 10 * Math.Log10(signalPower / noisePower) : 0;
        }

        /// <summary>
        /// Computes structural similarity (SSIM) between output and target signals.
        /// </summary>
        public static double StructuralSimilarity(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length <= 1)
                return 0;

            double muX = output.Average();
            double muY = 0;
            for (int i = 0; i < length; i++)
                muY += target[i];
            muY /= length;

            double sigmaX2 = output.Average(v => (v - muX) * (v - muX));
            double sigmaY2 = 0;
            for (int i = 0; i < length; i++)
                sigmaY2 += (target[i] - muY) * (target[i] - muY);
            sigmaY2 /= length;

            double sigmaXY = 0;
            for (int i = 0; i < length; i++)
                sigmaXY += (output[i] - muX) * (target[i] - muY);
            sigmaXY /= length;

            double C1 = 0.01 * 0.01;
            double C2 = 0.03 * 0.03;

            double numerator = (2 * muX * muY + C1) * (2 * sigmaXY + C2);
            double denominator = (muX * muX + muY * muY + C1) * (sigmaX2 + sigmaY2 + C2);

            return denominator > 0 ? numerator / denominator : 0;
        }

        /// <summary>
        /// Computes the peak signal-to-noise ratio (PSNR).
        /// </summary>
        public static double PeakSignalToNoiseRatio(double[] output, ImmutableArray<double> target, double maxSignal = 1.0)
        {
            double mse = MeanSquaredError(output, target);
            if (mse < 1e-10)
                return 100.0;
            return 20 * Math.Log10(maxSignal) - 10 * Math.Log10(mse);
        }

        /// <summary>
        /// Computes the L1 regularization penalty on genome weights.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="lambda">Regularization strength.</param>
        /// <returns>L1 penalty value.</returns>
        public static double L1Regularization(GeoGenome genome, double lambda = 0.001)
        {
            double sum = genome.Synapses
                .Where(s => s.IsActive)
                .Sum(s => Math.Abs(s.Weight));
            return lambda * sum;
        }

        /// <summary>
        /// Computes the L2 regularization penalty on genome weights.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="lambda">Regularization strength.</param>
        /// <returns>L2 penalty value.</returns>
        public static double L2Regularization(GeoGenome genome, double lambda = 0.001)
        {
            double sum = genome.Synapses
                .Where(s => s.IsActive)
                .Sum(s => s.Weight * s.Weight);
            return lambda * Math.Sqrt(sum);
        }

        /// <summary>
        /// Computes elastic net regularization combining L1 and L2.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="l1Lambda">L1 regularization strength.</param>
        /// <param name="l2Lambda">L2 regularization strength.</param>
        /// <returns>Elastic net penalty value.</returns>
        public static double ElasticNetRegularization(GeoGenome genome, double l1Lambda = 0.001, double l2Lambda = 0.001)
        {
            return L1Regularization(genome, l1Lambda) + L2Regularization(genome, l2Lambda);
        }

        /// <summary>
        /// Computes a complexity penalty based on genome structure.
        /// Encourages parsimonious solutions.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="complexityWeight">Weight for complexity penalty.</param>
        /// <returns>Complexity penalty value (higher = more complex).</returns>
        public static double ComplexityPenalty(GeoGenome genome, double complexityWeight = 0.01)
        {
            double neuronPenalty = genome.ActiveNeuronCount * 1.0;
            double synapsePenalty = genome.ActiveSynapseCount * 0.5;
            double depthPenalty = genome.MaxLayerDepth * 2.0;
            double densityPenalty = genome.ConnectionDensity * 10.0;

            return complexityWeight * (neuronPenalty + synapsePenalty + depthPenalty + densityPenalty);
        }

        /// <summary>
        /// Computes novelty score based on distance to nearest neighbors in feature space.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="archive">Archive of previously seen genomes.</param>
        /// <param name="k">Number of nearest neighbors.</param>
        /// <returns>Novelty score (higher = more novel).</returns>
        public static double NoveltyScore(GeoGenome genome, IReadOnlyList<GeoGenome> archive, int k = 5)
        {
            if (archive.Count == 0)
                return 1.0;

            var distances = archive
                .Select(other => ComputeGenomeFeatureDistance(genome, other))
                .OrderBy(d => d)
                .Take(Math.Min(k, archive.Count))
                .ToList();

            return distances.Count > 0 ? distances.Average() : 0;
        }

        /// <summary>
        /// Computes feature distance between two genomes based on structural features.
        /// </summary>
        private static double ComputeGenomeFeatureDistance(GeoGenome a, GeoGenome b)
        {
            double neuronDiff = Math.Abs(a.ActiveNeuronCount - b.ActiveNeuronCount);
            double synapseDiff = Math.Abs(a.ActiveSynapseCount - b.ActiveSynapseCount);
            double depthDiff = Math.Abs(a.MaxLayerDepth - b.MaxLayerDepth);
            double densityDiff = Math.Abs(a.ConnectionDensity - b.ConnectionDensity);

            double maxNeurons = Math.Max(a.ActiveNeuronCount, b.ActiveNeuronCount);
            double maxSynapses = Math.Max(a.ActiveSynapseCount, b.ActiveSynapseCount);
            double maxDepth = Math.Max(a.MaxLayerDepth, b.MaxLayerDepth);

            double normNeuron = maxNeurons > 0 ? neuronDiff / maxNeurons : 0;
            double normSynapse = maxSynapses > 0 ? synapseDiff / maxSynapses : 0;
            double normDepth = maxDepth > 0 ? depthDiff / maxDepth : 0;

            return Math.Sqrt(normNeuron * normNeuron + normSynapse * normSynapse +
                           normDepth * normDepth + densityDiff * densityDiff);
        }

        /// <summary>
        /// Creates a fitness function that combines multiple objectives with configurable weights.
        /// </summary>
        /// <param name="components">Component functions and their weights.</param>
        /// <returns>A combined fitness function.</returns>
        public static Func<GeoGenome, double> CreateWeightedCombine(
            params (Func<GeoGenome, double> Function, double Weight)[] components)
        {
            return genome =>
            {
                double totalWeight = components.Sum(c => c.Weight);
                if (totalWeight <= 0)
                    return 0;

                double totalFitness = 0;
                foreach (var (function, weight) in components)
                {
                    totalFitness += function(genome) * weight;
                }
                return totalFitness / totalWeight;
            };
        }

        /// <summary>
        /// Creates a penalized fitness function that subtracts penalties from a base fitness.
        /// </summary>
        /// <param name="baseFitness">Base fitness function.</param>
        /// <param name="penalties">Penalty functions and their weights.</param>
        /// <returns>A penalized fitness function.</returns>
        public static Func<GeoGenome, double> CreatePenalized(
            Func<GeoGenome, double> baseFitness,
            params (Func<GeoGenome, double> Penalty, double Weight)[] penalties)
        {
            return genome =>
            {
                double fitness = baseFitness(genome);
                foreach (var (penalty, weight) in penalties)
                {
                    fitness -= penalty(genome) * weight;
                }
                return fitness;
            };
        }

        /// <summary>
        /// Creates a fitness function that applies a threshold transformation.
        /// </summary>
        /// <param name="inner">Inner fitness function.</param>
        /// <param name="threshold">Minimum threshold value.</param>
        /// <returns>Thresholded fitness function.</returns>
        public static Func<GeoGenome, double> WithThreshold(
            Func<GeoGenome, double> inner,
            double threshold)
        {
            return genome =>
            {
                double fitness = inner(genome);
                return fitness >= threshold ? fitness : fitness - (threshold - fitness) * 10;
            };
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Innovation Number Generator

    /// <summary>
    /// Thread-safe global innovation number generator for NEAT-G.
    /// Ensures unique innovation numbers across all genomes and evolution runs.
    /// </summary>
    public sealed class InnovationNumberGenerator
    {
        private long _nextNeuronInnovation;
        private long _nextSynapseInnovation;
        private readonly ConcurrentDictionary<(long, long), long> _synapseInnovationCache;
        private readonly ConcurrentDictionary<long, long> _neuronInnovationCache;

        /// <summary>
        /// Initializes a new instance of the InnovationNumberGenerator class.
        /// </summary>
        /// <param name="startNeuronInnovation">Starting neuron innovation number.</param>
        /// <param name="startSynapseInnovation">Starting synapse innovation number.</param>
        public InnovationNumberGenerator(long startNeuronInnovation = 1, long startSynapseInnovation = 1000000)
        {
            _nextNeuronInnovation = startNeuronInnovation;
            _nextSynapseInnovation = startSynapseInnovation;
            _synapseInnovationCache = new ConcurrentDictionary<(long, long), long>();
            _neuronInnovationCache = new ConcurrentDictionary<long, long>();
        }

        /// <summary>Gets the next available neuron innovation number.</summary>
        public long GetNextNeuronInnovation()
        {
            return Interlocked.Increment(ref _nextNeuronInnovation);
        }

        /// <summary>
        /// Gets the innovation number for a synapse between two neurons.
        /// If the synapse already has an innovation number, returns it;
        /// otherwise, assigns a new one.
        /// </summary>
        /// <param name="sourceNeuronId">Source neuron innovation number.</param>
        /// <param name="targetNeuronId">Target neuron innovation number.</param>
        /// <returns>The innovation number for this synapse.</returns>
        public long GetSynapseInnovation(long sourceNeuronId, long targetNeuronId)
        {
            var key = sourceNeuronId < targetNeuronId
                ? (sourceNeuronId, targetNeuronId)
                : (targetNeuronId, sourceNeuronId);

            return _synapseInnovationCache.GetOrAdd(key, _ => Interlocked.Increment(ref _nextSynapseInnovation));
        }

        /// <summary>
        /// Checks if a synapse innovation already exists.
        /// </summary>
        /// <param name="sourceNeuronId">Source neuron ID.</param>
        /// <param name="targetNeuronId">Target neuron ID.</param>
        /// <param name="innovationNumber">The existing innovation number, if found.</param>
        /// <returns>True if the synapse innovation exists.</returns>
        public bool TryGetSynapseInnovation(long sourceNeuronId, long targetNeuronId, out long innovationNumber)
        {
            var key = sourceNeuronId < targetNeuronId
                ? (sourceNeuronId, targetNeuronId)
                : (targetNeuronId, sourceNeuronId);

            return _synapseInnovationCache.TryGetValue(key, out innovationNumber);
        }

        /// <summary>
        /// Gets the total number of neuron innovations issued.
        /// </summary>
        public long TotalNeuronInnovations => Volatile.Read(ref _nextNeuronInnovation);

        /// <summary>
        /// Gets the total number of synapse innovations issued.
        /// </summary>
        public long TotalSynapseInnovations => Volatile.Read(ref _nextSynapseInnovation) - 1000000;

        /// <summary>
        /// Resets the generator to initial state.
        /// </summary>
        public void Reset()
        {
            Interlocked.Exchange(ref _nextNeuronInnovation, 1);
            Interlocked.Exchange(ref _nextSynapseInnovation, 1000000);
            _synapseInnovationCache.Clear();
            _neuronInnovationCache.Clear();
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Advanced Genetic Operators

    /// <summary>
    /// Implements advanced genetic operators for NEAT-G evolution including
    /// speculative crossover, adaptive mutation scheduling, and topological
    /// search operators. These operators extend the basic crossover and mutation
    /// with more sophisticated search strategies.
    /// </summary>
    public sealed class AdvancedGeneticOperators
    {
        private readonly EvolutionConfig _config;
        private readonly InnovationNumberGenerator _innovationGenerator;
        private readonly Random _rng;

        /// <summary>
        /// Initializes a new instance of the AdvancedGeneticOperators class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="innovationGenerator">Innovation number generator.</param>
        /// <param name="rng">Random number generator.</param>
        public AdvancedGeneticOperators(
            EvolutionConfig config,
            InnovationNumberGenerator innovationGenerator,
            Random rng)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _innovationGenerator = innovationGenerator ?? throw new ArgumentNullException(nameof(innovationGenerator));
            _rng = rng ?? throw new ArgumentNullException(nameof(rng));
        }

        /// <summary>
        /// Performs speculative crossover: tries multiple crossover strategies and
        /// picks the offspring that maximizes estimated fitness improvement.
        /// </summary>
        /// <param name="parentA">First parent.</param>
        /// <param name="parentB">Second parent.</param>
        /// <param name="evaluator">Fitness evaluator for estimation.</param>
        /// <param name="context">Evaluation context.</param>
        /// <returns>The best offspring from multiple crossover attempts.</returns>
        public async Task<GeoGenome> SpeculativeCrossoverAsync(
            GeoGenome parentA,
            GeoGenome parentB,
            IFitnessEvaluator evaluator,
            EvaluationContext context)
        {
            var strategies = new ICrossoverStrategy[]
            {
                new SemanticCrossoverStrategy(_config),
                new TopologyCrossoverStrategy(_config),
                new WeightCrossoverStrategy(_config)
            };

            var candidates = new List<GeoGenome>();

            foreach (var strategy in strategies)
            {
                for (int attempt = 0; attempt < 3; attempt++)
                {
                    float blendBias = (float)_rng.NextDouble();
                    var result = strategy.Crossover(parentA, parentB, blendBias);

                    if (result.IsSuccess)
                    {
                        var evaluated = await evaluator.EvaluateAsync(result.Offspring, context, CancellationToken.None)
                            .ConfigureAwait(false);
                        candidates.Add(evaluated);
                    }
                }
            }

            if (candidates.Count == 0)
            {
                return parentA.Clone();
            }

            return candidates.OrderByDescending(g => g.Fitness).First();
        }

        /// <summary>
        /// Performs differential evolution-style crossover between multiple parents.
        /// Uses the DE/rand/1/bin strategy adapted for neural network genomes.
        /// </summary>
        /// <param name="parents">Pool of parent genomes (minimum 3).</param>
        /// <param name="crossoverProbability">Probability of crossover for each gene.</param>
        /// <returns>Offspring genome.</returns>
        public GeoGenome DifferentialEvolutionCrossover(
            IReadOnlyList<GeoGenome> parents,
            double crossoverProbability = 0.7)
        {
            if (parents.Count < 3)
                throw new ArgumentException("Differential evolution requires at least 3 parents.");

            int idx1 = _rng.Next(parents.Count);
            int idx2 = _rng.Next(parents.Count);
            int idx3 = _rng.Next(parents.Count);

            while (idx2 == idx1)
                idx2 = _rng.Next(parents.Count);
            while (idx3 == idx1 || idx3 == idx2)
                idx3 = _rng.Next(parents.Count);

            var base_genome = parents[idx1].Clone();
            var diffA = parents[idx2];
            var diffB = parents[idx3];

            double scalingFactor = 0.5 + _rng.NextDouble() * 0.5;

            var diffSynapses = new Dictionary<long, double>();
            foreach (var synapse in diffA.Synapses.Where(s => s.IsActive))
            {
                var matchingB = diffB.Synapses.FirstOrDefault(s =>
                    s.IsActive && s.InnovationNumber == synapse.InnovationNumber);

                if (matchingB != null)
                {
                    diffSynapses[synapse.InnovationNumber] = synapse.Weight - matchingB.Weight;
                }
            }

            foreach (var synapse in base_genome.Synapses.Where(s => s.IsActive))
            {
                if (_rng.NextDouble() < crossoverProbability && diffSynapses.TryGetValue(synapse.InnovationNumber, out var diff))
                {
                    synapse.Weight += scalingFactor * diff;
                    synapse.Weight = Math.Clamp(synapse.Weight, -10, 10);
                }
            }

            var diffNeurons = new Dictionary<long, double>();
            foreach (var neuron in diffA.Neurons.Where(n => n.IsActive))
            {
                var matchingB = diffB.Neurons.FirstOrDefault(n =>
                    n.IsActive && n.InnovationNumber == neuron.InnovationNumber);

                if (matchingB != null)
                {
                    diffNeurons[neuron.InnovationNumber] = neuron.Bias - matchingB.Bias;
                }
            }

            foreach (var neuron in base_genome.Neurons.Where(n => n.IsActive))
            {
                if (_rng.NextDouble() < crossoverProbability && diffNeurons.TryGetValue(neuron.InnovationNumber, out var diff))
                {
                    neuron.Bias += scalingFactor * diff;
                    neuron.Bias = Math.Clamp(neuron.Bias, -5, 5);
                }
            }

            base_genome.InvalidateFitness();
            base_genome.ComputeComplexity();
            return base_genome;
        }

        /// <summary>
        /// Performs polynomial mutation: generates offspring with polynomial probability
        /// distribution for continuous parameter optimization.
        /// </summary>
        /// <param name="genome">Genome to mutate.</param>
        /// <param name="eta">Distribution index (higher = more concentrated around parent).</param>
        /// <param name="mutationProbability">Probability of mutating each parameter.</param>
        /// <returns>Mutated genome.</returns>
        public GeoGenome PolynomialMutation(GeoGenome genome, double eta = 20.0, double mutationProbability = 0.1)
        {
            var mutated = genome.Clone();

            foreach (var synapse in mutated.Synapses.Where(s => s.IsActive))
            {
                if (_rng.NextDouble() < mutationProbability)
                {
                    double u = _rng.NextDouble();
                    double delta = u < 0.5
                        ? Math.Pow(2.0 * u, 1.0 / (eta + 1.0)) - 1.0
                        : 1.0 - Math.Pow(2.0 * (1.0 - u), 1.0 / (eta + 1.0));

                    synapse.Weight += delta * 2.0;
                    synapse.Weight = Math.Clamp(synapse.Weight, -10, 10);
                }
            }

            foreach (var neuron in mutated.Neurons.Where(n => n.IsActive && n.LayerIndex > 0))
            {
                if (_rng.NextDouble() < mutationProbability)
                {
                    double u = _rng.NextDouble();
                    double delta = u < 0.5
                        ? Math.Pow(2.0 * u, 1.0 / (eta + 1.0)) - 1.0
                        : 1.0 - Math.Pow(2.0 * (1.0 - u), 1.0 / (eta + 1.0));

                    neuron.Bias += delta;
                    neuron.Bias = Math.Clamp(neuron.Bias, -5, 5);
                }
            }

            mutated.InvalidateFitness();
            return mutated;
        }

        /// <summary>
        /// Performs simulated binary crossover (SBX) between two parent genomes.
        /// Creates offspring with a probability distribution centered around the parents.
        /// </summary>
        /// <param name="parentA">First parent.</param>
        /// <param name="parentB">Second parent.</param>
        /// <param name="eta">Distribution index.</param>
        /// <returns>Two offspring genomes.</returns>
        public (GeoGenome offspring1, GeoGenome offspring2) SimulatedBinaryCrossover(
            GeoGenome parentA,
            GeoGenome parentB,
            double eta = 20.0)
        {
            var child1 = parentA.Clone();
            var child2 = parentB.Clone();
            child1.Id = Guid.NewGuid();
            child2.Id = Guid.NewGuid();
            child1.InvalidateFitness();
            child2.InvalidateFitness();

            var aSynapseMap = parentA.Synapses.Where(s => s.IsActive).ToDictionary(s => s.InnovationNumber);
            var bSynapseMap = parentB.Synapses.Where(s => s.IsActive).ToDictionary(s => s.InnovationNumber);

            var commonInnovations = aSynapseMap.Keys.Intersect(bSynapseMap.Keys).ToList();

            foreach (var innov in commonInnovations)
            {
                double w1 = aSynapseMap[innov].Weight;
                double w2 = bSynapseMap[innov].Weight;

                if (_rng.NextDouble() <= 0.5)
                {
                    double u = _rng.NextDouble();
                    double beta = u <= 0.5
                        ? Math.Pow(2.0 * u, 1.0 / (eta + 1.0))
                        : Math.Pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta + 1.0));

                    double childW1 = 0.5 * ((1 + beta) * w1 + (1 - beta) * w2);
                    double childW2 = 0.5 * ((1 - beta) * w1 + (1 + beta) * w2);

                    var s1 = child1.Synapses.FirstOrDefault(s => s.InnovationNumber == innov);
                    var s2 = child2.Synapses.FirstOrDefault(s => s.InnovationNumber == innov);
                    if (s1 != null)
                        s1.Weight = childW1;
                    if (s2 != null)
                        s2.Weight = childW2;
                }
            }

            var aNeuronMap = parentA.Neurons.Where(n => n.IsActive).ToDictionary(n => n.InnovationNumber);
            var bNeuronMap = parentB.Neurons.Where(n => n.IsActive).ToDictionary(n => n.InnovationNumber);

            var commonNeurons = aNeuronMap.Keys.Intersect(bNeuronMap.Keys).ToList();

            foreach (var innov in commonNeurons)
            {
                double b1 = aNeuronMap[innov].Bias;
                double b2 = bNeuronMap[innov].Bias;

                if (_rng.NextDouble() <= 0.5)
                {
                    double u = _rng.NextDouble();
                    double beta = u <= 0.5
                        ? Math.Pow(2.0 * u, 1.0 / (eta + 1.0))
                        : Math.Pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta + 1.0));

                    double childB1 = 0.5 * ((1 + beta) * b1 + (1 - beta) * b2);
                    double childB2 = 0.5 * ((1 - beta) * b1 + (1 + beta) * b2);

                    var n1 = child1.Neurons.FirstOrDefault(n => n.InnovationNumber == innov);
                    var n2 = child2.Neurons.FirstOrDefault(n => n.InnovationNumber == innov);
                    if (n1 != null)
                        n1.Bias = childB1;
                    if (n2 != null)
                        n2.Bias = childB2;
                }
            }

            child1.ComputeComplexity();
            child2.ComputeComplexity();

            return (child1, child2);
        }

        /// <summary>
        /// Performs an adaptive window mutation where the mutation magnitude
        /// adapts based on the local fitness landscape gradient.
        /// </summary>
        /// <param name="genome">Genome to mutate.</param>
        /// <param name="windowSize">Number of mutations to try in the local search window.</param>
        /// <param name="evaluator">Fitness evaluator.</param>
        /// <param name="context">Evaluation context.</param>
        /// <returns>The best genome from the local search.</returns>
        public async Task<GeoGenome> AdaptiveWindowMutationAsync(
            GeoGenome genome,
            int windowSize,
            IFitnessEvaluator evaluator,
            EvaluationContext context)
        {
            var baseFitness = genome.Fitness;
            GeoGenome bestGenome = genome;
            double bestFitness = baseFitness;

            double[] perturbationMagnitudes = { 0.01, 0.05, 0.1, 0.2, 0.5 };

            for (int w = 0; w < windowSize; w++)
            {
                var candidate = genome.Clone();
                candidate.Id = Guid.NewGuid();
                candidate.InvalidateFitness();

                double magnitude = perturbationMagnitudes[_rng.Next(perturbationMagnitudes.Length)];
                int perturbCount = _rng.Next(1, Math.Max(2, candidate.ActiveSynapseCount / 5));

                var activeSynapses = candidate.Synapses.Where(s => s.IsActive).ToList();
                for (int p = 0; p < perturbCount && activeSynapses.Count > 0; p++)
                {
                    var synapse = activeSynapses[_rng.Next(activeSynapses.Count)];
                    double u1 = _rng.NextDouble();
                    double u2 = _rng.NextDouble();
                    double z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
                    synapse.Weight += z * magnitude;
                    synapse.Weight = Math.Clamp(synapse.Weight, -10, 10);
                }

                var evaluated = await evaluator.EvaluateAsync(candidate, context, CancellationToken.None)
                    .ConfigureAwait(false);

                if (evaluated.Fitness > bestFitness)
                {
                    bestFitness = evaluated.Fitness;
                    bestGenome = evaluated;
                }
            }

            return bestGenome;
        }

        /// <summary>
        /// Performs a hill-climbing local search around a genome.
        /// Attempts single-weight perturbations and keeps improvements.
        /// </summary>
        /// <param name="genome">Starting genome.</param>
        /// <param name="maxIterations">Maximum hill-climbing iterations.</param>
        /// <param name="stepSize">Step size for weight perturbation.</param>
        /// <param name="evaluator">Fitness evaluator.</param>
        /// <param name="context">Evaluation context.</param>
        /// <returns>Optimized genome.</returns>
        public async Task<GeoGenome> HillClimbingAsync(
            GeoGenome genome,
            int maxIterations,
            double stepSize,
            IFitnessEvaluator evaluator,
            EvaluationContext context)
        {
            var current = genome.Clone();
            current = await evaluator.EvaluateAsync(current, context, CancellationToken.None)
                .ConfigureAwait(false);
            double currentFitness = current.Fitness;

            var activeSynapses = current.Synapses.Where(s => s.IsActive).ToList();

            for (int iter = 0; iter < maxIterations; iter++)
            {
                if (activeSynapses.Count == 0)
                    break;

                var candidate = current.Clone();
                candidate.Id = Guid.NewGuid();
                candidate.InvalidateFitness();

                var candidateSynapse = candidate.Synapses
                    .First(s => s.InnovationNumber == activeSynapses[_rng.Next(activeSynapses.Count)].InnovationNumber);

                candidateSynapse.Weight += (_rng.NextDouble() * 2 - 1) * stepSize;
                candidateSynapse.Weight = Math.Clamp(candidateSynapse.Weight, -10, 10);

                var evaluated = await evaluator.EvaluateAsync(candidate, context, CancellationToken.None)
                    .ConfigureAwait(false);

                if (evaluated.Fitness > currentFitness)
                {
                    current = evaluated;
                    currentFitness = evaluated.Fitness;
                }
            }

            return current;
        }

        /// <summary>
        /// Performs crossover with topological schema preservation.
        /// Identifies and preserves important substructures (schemas) during recombination.
        /// </summary>
        /// <param name="parentA">First parent.</param>
        /// <param name="parentB">Second parent.</param>
        /// <returns>Offspring with preserved schemas.</returns>
        public GeoGenome SchemaPreservingCrossover(GeoGenome parentA, GeoGenome parentB)
        {
            var child = parentA.Clone();
            child.Id = Guid.NewGuid();
            child.InvalidateFitness();

            var schemasA = IdentifySchemas(parentA);
            var schemasB = IdentifySchemas(parentB);

            var selectedSchemas = new List<TopologicalSchema>();
            foreach (var schemaA in schemasA)
            {
                bool dominated = schemasB.Any(sB =>
                    sB.Size >= schemaA.Size &&
                    sB.AverageWeight > schemaA.AverageWeight &&
                    sB.Connectivity >= schemaA.Connectivity);

                if (!dominated || _rng.NextDouble() < 0.3)
                {
                    selectedSchemas.Add(schemaA);
                }
            }

            foreach (var schemaB in schemasB)
            {
                bool alreadyCovered = selectedSchemas.Any(sA =>
                    sA.Size >= schemaB.Size &&
                    sA.AverageWeight > schemaB.AverageWeight);

                if (!alreadyCovered && _rng.NextDouble() < 0.5)
                {
                    selectedSchemas.Add(schemaB);
                }
            }

            foreach (var schema in selectedSchemas)
            {
                foreach (var neuronId in schema.NeuronIds)
                {
                    var neuron = child.Neurons.FirstOrDefault(n => n.InnovationNumber == neuronId);
                    if (neuron != null)
                    {
                        neuron.IsActive = true;
                        var sourceNeuron = schema.IsFromParentA
                            ? parentA.Neurons.FirstOrDefault(n => n.InnovationNumber == neuronId)
                            : parentB.Neurons.FirstOrDefault(n => n.InnovationNumber == neuronId);

                        if (sourceNeuron != null)
                        {
                            neuron.Bias = sourceNeuron.Bias;
                            neuron.Activation = sourceNeuron.Activation;
                        }
                    }
                }

                foreach (var synapseId in schema.SynapseIds)
                {
                    var synapse = child.Synapses.FirstOrDefault(s => s.InnovationNumber == synapseId);
                    if (synapse != null)
                    {
                        synapse.IsActive = true;
                        var sourceSynapse = schema.IsFromParentA
                            ? parentA.Synapses.FirstOrDefault(s => s.InnovationNumber == synapseId)
                            : parentB.Synapses.FirstOrDefault(s => s.InnovationNumber == synapseId);

                        if (sourceSynapse != null)
                        {
                            synapse.Weight = sourceSynapse.Weight;
                        }
                    }
                }
            }

            child.ComputeComplexity();
            return child;
        }

        /// <summary>
        /// Identifies topological schemas (important substructures) in a genome.
        /// </summary>
        /// <param name="genome">The genome to analyze.</param>
        /// <returns>List of identified schemas.</returns>
        public IReadOnlyList<TopologicalSchema> IdentifySchemas(GeoGenome genome)
        {
            var schemas = new List<TopologicalSchema>();
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            var visited = new HashSet<long>();
            foreach (var neuron in activeNeurons)
            {
                if (visited.Contains(neuron.InnovationNumber))
                    continue;

                var connected = new List<long> { neuron.InnovationNumber };
                var connectedSynapses = new List<long>();
                var queue = new Queue<long>();
                queue.Enqueue(neuron.InnovationNumber);

                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (!visited.Contains(current))
                    {
                        visited.Add(current);

                        foreach (var synapse in activeSynapses.Where(s =>
                            s.SourceNeuronId == current || s.TargetNeuronId == current))
                        {
                            long neighbor = synapse.SourceNeuronId == current
                                ? synapse.TargetNeuronId
                                : synapse.SourceNeuronId;

                            if (!visited.Contains(neighbor))
                            {
                                queue.Enqueue(neighbor);
                                connected.Add(neighbor);
                                connectedSynapses.Add(synapse.InnovationNumber);
                            }
                        }
                    }
                }

                if (connected.Count >= 2)
                {
                    double avgWeight = connectedSynapses.Count > 0
                        ? connectedSynapses
                            .Select(id => activeSynapses.FirstOrDefault(s => s.InnovationNumber == id))
                            .Where(s => s != null)
                            .Average(s => s!.Weight)
                        : 0;

                    int maxConnections = connected.Count * (connected.Count - 1);
                    double connectivity = maxConnections > 0
                        ? (double)connectedSynapses.Count / maxConnections
                        : 0;

                    schemas.Add(new TopologicalSchema
                    {
                        NeuronIds = connected.ToImmutableArray(),
                        SynapseIds = connectedSynapses.ToImmutableArray(),
                        Size = connected.Count,
                        AverageWeight = avgWeight,
                        Connectivity = connectivity,
                        IsFromParentA = true
                    });
                }
            }

            return schemas;
        }

        /// <summary>
        /// Performs gene transfer between non-homologous genomes by identifying
        /// functionally equivalent neurons based on their connectivity patterns.
        /// </summary>
        /// <param name="donor">Donor genome.</param>
        /// <param name="recipient">Recipient genome.</param>
        /// <returns>Recipient genome with transferred genes.</returns>
        public GeoGenome TransferHomologousGenes(GeoGenome donor, GeoGenome recipient)
        {
            var result = recipient.Clone();
            result.InvalidateFitness();

            var donorSignatures = ComputeNeuronSignatures(donor);
            var recipientSignatures = ComputeNeuronSignatures(recipient);

            foreach (var donorSig in donorSignatures)
            {
                var bestMatch = recipientSignatures
                    .OrderBy(rSig => ComputeSignatureDistance(donorSig.Value, rSig.Value))
                    .FirstOrDefault();

                if (bestMatch.Value != null)
                {
                    double distance = ComputeSignatureDistance(donorSig.Value, bestMatch.Value);

                    if (distance < 0.3)
                    {
                        var resultNeuron = result.Neurons
                            .FirstOrDefault(n => n.InnovationNumber == bestMatch.Key);

                        if (resultNeuron != null)
                        {
                            double blendFactor = 1.0 - distance;
                            resultNeuron.Bias = resultNeuron.Bias * (1 - blendFactor) +
                                               donor.Neurons.First(n => n.InnovationNumber == donorSig.Key).Bias * blendFactor;
                        }
                    }
                }
            }

            return result;
        }

        private Dictionary<long, double[]> ComputeNeuronSignatures(GeoGenome genome)
        {
            var signatures = new Dictionary<long, double[]>();
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            foreach (var neuron in genome.Neurons.Where(n => n.IsActive))
            {
                var inputWeights = activeSynapses
                    .Where(s => s.TargetNeuronId == neuron.InnovationNumber)
                    .Select(s => s.Weight)
                    .OrderBy(w => w)
                    .Take(10)
                    .ToArray();

                var outputWeights = activeSynapses
                    .Where(s => s.SourceNeuronId == neuron.InnovationNumber)
                    .Select(s => s.Weight)
                    .OrderBy(w => w)
                    .Take(10)
                    .ToArray();

                var signature = new double[22];
                signature[0] = neuron.LayerIndex;
                signature[1] = (int)neuron.Activation;

                for (int i = 0; i < Math.Min(inputWeights.Length, 10); i++)
                    signature[2 + i] = inputWeights[i];

                for (int i = 0; i < Math.Min(outputWeights.Length, 10); i++)
                    signature[12 + i] = outputWeights[i];

                signatures[neuron.InnovationNumber] = signature;
            }

            return signatures;
        }

        private double ComputeSignatureDistance(double[] sigA, double[] sigB)
        {
            int dim = Math.Min(sigA.Length, sigB.Length);
            double dist = 0;
            for (int i = 0; i < dim; i++)
            {
                double diff = sigA[i] - sigB[i];
                dist += diff * diff;
            }
            return Math.Sqrt(dist) / Math.Sqrt(dim);
        }
    }

    /// <summary>
    /// Represents a topological schema - an important substructure in a genome.
    /// </summary>
    public sealed class TopologicalSchema
    {
        /// <summary>Innovation numbers of neurons in this schema.</summary>
        public ImmutableArray<long> NeuronIds { get; init; }

        /// <summary>Innovation numbers of synapses in this schema.</summary>
        public ImmutableArray<long> SynapseIds { get; init; }

        /// <summary>Size of the schema (number of neurons).</summary>
        public int Size { get; init; }

        /// <summary>Average weight of synapses in this schema.</summary>
        public double AverageWeight { get; init; }

        /// <summary>Connectivity ratio of this schema.</summary>
        public double Connectivity { get; init; }

        /// <summary>Whether this schema came from parent A.</summary>
        public bool IsFromParentA { get; init; }

        /// <summary>Schema fitness (if evaluated).</summary>
        public double Fitness { get; set; }

        /// <inheritdoc/>
        public override string ToString() =>
            $"Schema(Size={Size}, Connectivity={Connectivity:F3}, AvgWeight={AverageWeight:F3})";
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Genome Visualization Data

    /// <summary>
    /// Provides data structures for visualizing genome structure and evolution progress.
    /// Can be used to generate graph layouts, fitness plots, and species diagrams.
    /// </summary>
    public sealed class GenomeVisualizationData
    {
        /// <summary>
        /// Node data for graph visualization.
        /// </summary>
        public sealed class GraphNode
        {
            /// <summary>Node identifier.</summary>
            public long Id { get; init; }

            /// <summary>Layer index (for layered layout).</summary>
            public int Layer { get; init; }

            /// <summary>Position within layer.</summary>
            public int Position { get; init; }

            /// <summary>Activation function name.</summary>
            public string Activation { get; init; } = string.Empty;

            /// <summary>Bias value.</summary>
            public double Bias { get; init; }

            /// <summary>Whether the node is active.</summary>
            public bool IsActive { get; init; }

            /// <summary>X coordinate for layout.</summary>
            public double X { get; set; }

            /// <summary>Y coordinate for layout.</summary>
            public double Y { get; set; }

            /// <summary>Node importance score.</summary>
            public double Importance { get; init; }

            /// <summary>Color for visualization.</summary>
            public string Color { get; init; } = "#4A90D9";
        }

        /// <summary>
        /// Edge data for graph visualization.
        /// </summary>
        public sealed class GraphEdge
        {
            /// <summary>Source node identifier.</summary>
            public long SourceId { get; init; }

            /// <summary>Target node identifier.</summary>
            public long TargetId { get; init; }

            /// <summary>Connection weight.</summary>
            public double Weight { get; init; }

            /// <summary>Whether the edge is active.</summary>
            public bool IsActive { get; init; }

            /// <summary>Edge thickness for visualization (based on weight magnitude).</summary>
            public double Thickness => Math.Max(0.5, Math.Min(5.0, Math.Abs(Weight) * 2));

            /// <summary>Edge color (green for positive, red for negative).</summary>
            public string Color => Weight >= 0 ? "#27AE60" : "#E74C3C";

            /// <summary>Edge opacity based on confidence.</summary>
            public double Opacity { get; init; } = 1.0;
        }

        /// <summary>
        /// Evolution timeline data point.
        /// </summary>
        public sealed class TimelinePoint
        {
            /// <summary>Generation number.</summary>
            public int Generation { get; init; }

            /// <summary>Best fitness at this generation.</summary>
            public double BestFitness { get; init; }

            /// <summary>Average fitness at this generation.</summary>
            public double AverageFitness { get; init; }

            /// <summary>Number of species at this generation.</summary>
            public int SpeciesCount { get; init; }

            /// <summary>Diversity metric at this generation.</summary>
            public double Diversity { get; init; }

            /// <summary>Mutation rate at this generation.</summary>
            public double MutationRate { get; init; }

            /// <summary>Number of evaluations at this generation.</summary>
            public long Evaluations { get; init; }
        }

        /// <summary>
        /// Species cluster data for species visualization.
        /// </summary>
        public sealed class SpeciesCluster
        {
            /// <summary>Species identifier.</summary>
            public int SpeciesId { get; init; }

            /// <summary>Centroid position in embedding space.</summary>
            public (double X, double Y) Centroid { get; init; }

            /// <summary>Radius of the species cluster.</summary>
            public double Radius { get; init; }

            /// <summary>Number of members.</summary>
            public int MemberCount { get; init; }

            /// <summary>Best fitness in the species.</summary>
            public double BestFitness { get; init; }

            /// <summary>Species color.</summary>
            public string Color { get; init; } = "#4A90D9";

            /// <summary>Member positions in embedding space.</summary>
            public IReadOnlyList<(double X, double Y)> MemberPositions { get; init; } =
                Array.Empty<(double, double)>();
        }

        /// <summary>
        /// Generates graph layout data for a genome.
        /// </summary>
        /// <param name="genome">The genome to visualize.</param>
        /// <returns>Tuple of nodes and edges for graph rendering.</returns>
        public static (IReadOnlyList<GraphNode> Nodes, IReadOnlyList<GraphEdge> Edges) GenerateGraphData(GeoGenome genome)
        {
            var nodes = new List<GraphNode>();
            var edges = new List<GraphEdge>();

            int maxLayer = genome.MaxLayerDepth;
            var layerCounts = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.LayerIndex)
                .ToDictionary(g => g.Key, g => g.Count());

            foreach (var neuron in genome.Neurons)
            {
                int layerSize = layerCounts.TryGetValue(neuron.LayerIndex, out var count) ? count : 1;
                int positionInLayer = neuron.PositionInLayer;

                double x = neuron.LayerIndex * 100.0;
                double y = (positionInLayer - layerSize / 2.0) * 60.0;

                int inputCount = genome.Synapses.Count(s => s.IsActive && s.TargetNeuronId == neuron.InnovationNumber);
                int outputCount = genome.Synapses.Count(s => s.IsActive && s.SourceNeuronId == neuron.InnovationNumber);
                double importance = Math.Log(1 + inputCount + outputCount);

                string color = neuron.LayerIndex == 0 ? "#3498DB" :
                              neuron.LayerIndex == maxLayer ? "#E67E22" :
                              neuron.IsActive ? "#2ECC71" : "#95A5A6";

                nodes.Add(new GraphNode
                {
                    Id = neuron.InnovationNumber,
                    Layer = neuron.LayerIndex,
                    Position = positionInLayer,
                    Activation = neuron.Activation.ToString(),
                    Bias = neuron.Bias,
                    IsActive = neuron.IsActive,
                    X = x,
                    Y = y,
                    Importance = importance,
                    Color = color
                });
            }

            foreach (var synapse in genome.Synapses)
            {
                double opacity = synapse.IsActive ? Math.Min(1.0, Math.Abs(synapse.Weight) + 0.2) : 0.2;

                edges.Add(new GraphEdge
                {
                    SourceId = synapse.SourceNeuronId,
                    TargetId = synapse.TargetNeuronId,
                    Weight = synapse.Weight,
                    IsActive = synapse.IsActive,
                    Opacity = opacity
                });
            }

            return (nodes.AsReadOnly(), edges.AsReadOnly());
        }

        /// <summary>
        /// Generates timeline data from evolution metrics history.
        /// </summary>
        /// <param name="metricsHistory">History of evolution metrics.</param>
        /// <returns>List of timeline points.</returns>
        public static IReadOnlyList<TimelinePoint> GenerateTimelineData(IReadOnlyList<EvolutionMetrics> metricsHistory)
        {
            return metricsHistory.Select(m => new TimelinePoint
            {
                Generation = m.Generation,
                BestFitness = m.BestFitness,
                AverageFitness = m.AverageFitness,
                SpeciesCount = m.SpeciesCount,
                Diversity = m.DiversityMetric,
                MutationRate = m.AdaptiveMutationRate,
                Evaluations = m.TotalEvaluations
            }).ToList().AsReadOnly();
        }

        /// <summary>
        /// Generates species cluster data from species information using t-SNE-like
        /// dimensionality reduction.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <param name="population">Current population.</param>
        /// <returns>List of species clusters for visualization.</returns>
        public static IReadOnlyList<SpeciesCluster> GenerateSpeciesClusters(
            ImmutableArray<SpeciesInfo> species,
            GenomePopulation population)
        {
            var clusters = new List<SpeciesCluster>();
            string[] palette = {
                "#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6",
                "#1ABC9C", "#E67E22", "#34495E", "#16A085", "#C0392B",
                "#27AE60", "#2980B9", "#8E44AD", "#D35400", "#1F77B4"
            };

            var rng = new Random(42);

            foreach (var s in species)
            {
                var memberGenomes = population.Genomes
                    .Where(g => s.MemberIds.Contains(g.Id))
                    .ToList();

                var positions = new List<(double X, double Y)>();
                double sumX = 0, sumY = 0;

                foreach (var genome in memberGenomes)
                {
                    double x = rng.NextDouble() * 200 - 100;
                    double y = rng.NextDouble() * 200 - 100;

                    if (genome.SemanticEmbedding.Length >= 2)
                    {
                        x = genome.SemanticEmbedding[0] * 100;
                        y = genome.SemanticEmbedding[1] * 100;
                    }

                    positions.Add((x, y));
                    sumX += x;
                    sumY += y;
                }

                double centroidX = positions.Count > 0 ? sumX / positions.Count : 0;
                double centroidY = positions.Count > 0 ? sumY / positions.Count : 0;

                double radius = 0;
                foreach (var (px, py) in positions)
                {
                    double dist = Math.Sqrt((px - centroidX) * (px - centroidX) + (py - centroidY) * (py - centroidY));
                    radius = Math.Max(radius, dist);
                }

                clusters.Add(new SpeciesCluster
                {
                    SpeciesId = s.Id,
                    Centroid = (centroidX, centroidY),
                    Radius = Math.Max(20, radius),
                    MemberCount = s.MemberCount,
                    BestFitness = s.BestFitness,
                    Color = palette[s.Id % palette.Length],
                    MemberPositions = positions.AsReadOnly()
                });
            }

            return clusters.AsReadOnly();
        }

        /// <summary>
        /// Generates an adjacency list representation of the genome for external visualization tools.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Adjacency list as a dictionary.</returns>
        public static IReadOnlyDictionary<long, IReadOnlyList<(long TargetId, double Weight)>> GenerateAdjacencyList(GeoGenome genome)
        {
            var adjacencyList = new Dictionary<long, List<(long, double)>>();

            foreach (var neuron in genome.Neurons.Where(n => n.IsActive))
            {
                adjacencyList[neuron.InnovationNumber] = new List<(long, double)>();
            }

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (adjacencyList.TryGetValue(synapse.SourceNeuronId, out var neighbors))
                {
                    neighbors.Add((synapse.TargetNeuronId, synapse.Weight));
                }
            }

            return adjacencyList.ToDictionary(
                kvp => kvp.Key,
                kvp => (IReadOnlyList<(long, double)>)kvp.Value.AsReadOnly());
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Event Pipeline

    /// <summary>
    /// Provides a pipeline-based approach to evolution where each step is
    /// a composable transformation. Allows custom evolution workflows.
    /// </summary>
    public sealed class EvolutionPipeline
    {
        private readonly List<IEvolutionStep> _steps;
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the EvolutionPipeline class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public EvolutionPipeline(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _steps = new List<IEvolutionStep>();
        }

        /// <summary>
        /// Adds a step to the pipeline.
        /// </summary>
        /// <param name="step">The evolution step to add.</param>
        /// <returns>This pipeline for chaining.</returns>
        public EvolutionPipeline AddStep(IEvolutionStep step)
        {
            _steps.Add(step);
            return this;
        }

        /// <summary>
        /// Executes the complete pipeline.
        /// </summary>
        /// <param name="population">Initial population.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Final population after pipeline execution.</returns>
        public async Task<GenomePopulation> ExecuteAsync(
            GenomePopulation population,
            EvaluationContext context,
            CancellationToken ct)
        {
            var current = population;

            foreach (var step in _steps)
            {
                ct.ThrowIfCancellationRequested();
                current = await step.ExecuteAsync(current, context, ct).ConfigureAwait(false);
            }

            return current;
        }

        /// <summary>
        /// Creates a standard NEAT-G evolution pipeline with all default steps.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <returns>A configured pipeline.</returns>
        public static EvolutionPipeline CreateStandardPipeline(EvolutionConfig config)
        {
            return new EvolutionPipeline(config)
                .AddStep(new InitializationStep())
                .AddStep(new EvaluationStep())
                .AddStep(new SpeciationStep())
                .AddStep(new SelectionStep())
                .AddStep(new CrossoverStep())
                .AddStep(new MutationStep())
                .AddStep(new SurvivalStep());
        }
    }

    /// <summary>
    /// Interface for an evolution pipeline step.
    /// </summary>
    public interface IEvolutionStep
    {
        /// <summary>
        /// Executes this evolution step.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Modified population after this step.</returns>
        Task<GenomePopulation> ExecuteAsync(GenomePopulation population, EvaluationContext context, CancellationToken ct);

        /// <summary>Gets the name of this step.</summary>
        string Name { get; }
    }

    /// <summary>
    /// Pipeline step for population initialization.
    /// </summary>
    public sealed class InitializationStep : IEvolutionStep
    {
        /// <inheritdoc/>
        public string Name => "Initialization";

        /// <inheritdoc/>
        public Task<GenomePopulation> ExecuteAsync(
            GenomePopulation population,
            EvaluationContext context,
            CancellationToken ct)
        {
            return Task.FromResult(population);
        }
    }

    /// <summary>
    /// Pipeline step for fitness evaluation.
    /// </summary>
    public sealed class EvaluationStep : IEvolutionStep
    {
        /// <inheritdoc/>
        public string Name => "Evaluation";

        /// <inheritdoc/>
        public async Task<GenomePopulation> ExecuteAsync(
            GenomePopulation population,
            EvaluationContext context,
            CancellationToken ct)
        {
            var evaluator = new FitnessEvaluator(context);
            var scheduler = new ParallelEvolutionScheduler(new EvolutionConfig());

            var results = await scheduler.EvaluatePopulationParallelAsync(
                population.Genomes.ToList(), evaluator, context, ct).ConfigureAwait(false);

            return population with { Genomes = results.ToImmutableArray() };
        }
    }

    /// <summary>
    /// Pipeline step for speciation.
    /// </summary>
    public sealed class SpeciationStep : IEvolutionStep
    {
        /// <inheritdoc/>
        public string Name => "Speciation";

        /// <inheritdoc/>
        public Task<GenomePopulation> ExecuteAsync(
            GenomePopulation population,
            EvaluationContext context,
            CancellationToken ct)
        {
            var config = new EvolutionConfig();
            var speciation = new ManifoldSpeciationStrategy(config);
            var species = speciation.Speciate(population);

            var assignments = new Dictionary<Guid, int>();
            foreach (var s in species)
            {
                foreach (var memberId in s.MemberIds)
                {
                    assignments[memberId] = s.Id;
                }
            }

            return Task.FromResult(population with
            {
                SpeciesAssignments = assignments.ToImmutableDictionary()
            });
        }
    }

    /// <summary>
    /// Pipeline step for parent selection.
    /// </summary>
    public sealed class SelectionStep : IEvolutionStep
    {
        /// <inheritdoc/>
        public string Name => "Selection";

        /// <inheritdoc/>
        public Task<GenomePopulation> ExecuteAsync(
            GenomePopulation population,
            EvaluationContext context,
            CancellationToken ct)
        {
            return Task.FromResult(population);
        }
    }

    /// <summary>
    /// Pipeline step for crossover.
    /// </summary>
    public sealed class CrossoverStep : IEvolutionStep
    {
        /// <inheritdoc/>
        public string Name => "Crossover";

        /// <inheritdoc/>
        public Task<GenomePopulation> ExecuteAsync(
            GenomePopulation population,
            EvaluationContext context,
            CancellationToken ct)
        {
            return Task.FromResult(population);
        }
    }

    /// <summary>
    /// Pipeline step for mutation.
    /// </summary>
    public sealed class MutationStep : IEvolutionStep
    {
        /// <inheritdoc/>
        public string Name => "Mutation";

        /// <inheritdoc/>
        public Task<GenomePopulation> ExecuteAsync(
            GenomePopulation population,
            EvaluationContext context,
            CancellationToken ct)
        {
            return Task.FromResult(population);
        }
    }

    /// <summary>
    /// Pipeline step for survival selection.
    /// </summary>
    public sealed class SurvivalStep : IEvolutionStep
    {
        /// <inheritdoc/>
        public string Name => "Survival";

        /// <inheritdoc/>
        public Task<GenomePopulation> ExecuteAsync(
            GenomePopulation population,
            EvaluationContext context,
            CancellationToken ct)
        {
            return Task.FromResult(population);
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Validation Utilities

    /// <summary>
    /// Provides validation utilities for genomes and population data.
    /// Ensures structural integrity and consistency of evolutionary data.
    /// </summary>
    public static class GenomeValidator
    {
        /// <summary>
        /// Validates a genome for structural integrity.
        /// </summary>
        /// <param name="genome">The genome to validate.</param>
        /// <returns>A list of validation errors. Empty if valid.</returns>
        public static IReadOnlyList<string> Validate(GeoGenome genome)
        {
            var errors = new List<string>();

            if (genome == null)
            {
                errors.Add("Genome is null.");
                return errors;
            }

            if (genome.Neurons.Count == 0)
                errors.Add("Genome has no neurons.");

            if (genome.Synapses.Count == 0)
                errors.Add("Genome has no synapses.");

            var neuronIds = new HashSet<long>();
            foreach (var neuron in genome.Neurons)
            {
                if (!neuronIds.Add(neuron.InnovationNumber))
                    errors.Add($"Duplicate neuron innovation number: {neuron.InnovationNumber}");

                if (double.IsNaN(neuron.Bias) || double.IsInfinity(neuron.Bias))
                    errors.Add($"Invalid bias for neuron {neuron.InnovationNumber}: {neuron.Bias}");
            }

            foreach (var synapse in genome.Synapses)
            {
                if (!neuronIds.Contains(synapse.SourceNeuronId))
                    errors.Add($"Synapse {synapse.InnovationNumber} references non-existent source neuron {synapse.SourceNeuronId}");

                if (!neuronIds.Contains(synapse.TargetNeuronId))
                    errors.Add($"Synapse {synapse.InnovationNumber} references non-existent target neuron {synapse.TargetNeuronId}");

                if (double.IsNaN(synapse.Weight) || double.IsInfinity(synapse.Weight))
                    errors.Add($"Invalid weight for synapse {synapse.InnovationNumber}: {synapse.Weight}");

                if (synapse.SourceNeuronId == synapse.TargetNeuronId)
                    errors.Add($"Self-loop detected in synapse {synapse.InnovationNumber}");
            }

            if (genome.InputCount <= 0)
                errors.Add($"Invalid input count: {genome.InputCount}");

            if (genome.OutputCount <= 0)
                errors.Add($"Invalid output count: {genome.OutputCount}");

            int inputNeurons = genome.Neurons.Count(n => n.LayerIndex == 0 && n.IsActive);
            if (inputNeurons != genome.InputCount)
                errors.Add($"Input neuron count mismatch: expected {genome.InputCount}, found {inputNeurons}");

            int outputNeurons = genome.Neurons.Count(n => n.LayerIndex == genome.MaxLayerDepth && n.IsActive);
            if (outputNeurons != genome.OutputCount)
                errors.Add($"Output neuron count mismatch: expected {genome.OutputCount}, found {outputNeurons}");

            if (double.IsNaN(genome.Fitness) && genome.EvaluationCount > 0)
                errors.Add("Fitness is NaN after evaluation.");

            return errors;
        }

        /// <summary>
        /// Validates a population for consistency.
        /// </summary>
        /// <param name="population">The population to validate.</param>
        /// <returns>A list of validation errors. Empty if valid.</returns>
        public static IReadOnlyList<string> ValidatePopulation(GenomePopulation population)
        {
            var errors = new List<string>();

            if (population == null)
            {
                errors.Add("Population is null.");
                return errors;
            }

            if (population.Genomes.Length == 0)
                errors.Add("Population is empty.");

            var ids = new HashSet<Guid>();
            foreach (var genome in population.Genomes)
            {
                if (!ids.Add(genome.Id))
                    errors.Add($"Duplicate genome ID: {genome.Id}");

                var genomeErrors = Validate(genome);
                foreach (var error in genomeErrors)
                {
                    errors.Add($"Genome {genome.Id}: {error}");
                }
            }

            return errors;
        }

        /// <summary>
        /// Validates species information for consistency.
        /// </summary>
        /// <param name="species">Species to validate.</param>
        /// <param name="population">Population the species belong to.</param>
        /// <returns>A list of validation errors.</returns>
        public static IReadOnlyList<string> ValidateSpecies(
            ImmutableArray<SpeciesInfo> species,
            GenomePopulation population)
        {
            var errors = new List<string>();
            var genomeIds = population.Genomes.Select(g => g.Id).ToHashSet();

            foreach (var s in species)
            {
                foreach (var memberId in s.MemberIds)
                {
                    if (!genomeIds.Contains(memberId))
                        errors.Add($"Species {s.Id} references non-existent genome {memberId}");
                }

                if (s.Representative != null && !s.MemberIds.Contains(s.Representative.Id))
                    errors.Add($"Species {s.Id} representative is not a member of the species");
            }

            var allMemberIds = species.SelectMany(s => s.MemberIds).ToList();
            var duplicates = allMemberIds.GroupBy(id => id).Where(g => g.Count() > 1);
            foreach (var dup in duplicates)
            {
                errors.Add($"Genome {dup.Key} belongs to multiple species");
            }

            return errors;
        }

        /// <summary>
        /// Checks if a genome is structurally valid (has required input/output connectivity).
        /// </summary>
        /// <param name="genome">The genome to check.</param>
        /// <returns>True if the genome is structurally valid.</returns>
        public static bool IsStructurallyValid(GeoGenome genome)
        {
            if (genome == null)
                return false;

            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            if (activeNeurons.Count < genome.InputCount + genome.OutputCount)
                return false;

            var inputNeurons = activeNeurons.Where(n => n.LayerIndex == 0).ToList();
            var outputNeurons = activeNeurons.Where(n => n.LayerIndex >= genome.MaxLayerDepth).ToList();

            if (inputNeurons.Count < genome.InputCount)
                return false;

            if (outputNeurons.Count < genome.OutputCount)
                return false;

            foreach (var outputNeuron in outputNeurons)
            {
                bool hasInput = activeSynapses.Any(s => s.TargetNeuronId == outputNeuron.InnovationNumber);
                if (!hasInput)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Computes a health score for a genome (0 = unhealthy, 1 = perfectly healthy).
        /// </summary>
        /// <param name="genome">The genome to assess.</param>
        /// <returns>Health score between 0 and 1.</returns>
        public static double ComputeHealthScore(GeoGenome genome)
        {
            if (genome == null)
                return 0;

            var errors = Validate(genome);
            double errorPenalty = errors.Count * 0.1;

            bool structurallyValid = IsStructurallyValid(genome);
            double structureBonus = structurallyValid ? 0.3 : 0;

            double connectivityScore = genome.ConnectionDensity > 0
                ? Math.Min(1.0, genome.ConnectionDensity * 5)
                : 0;

            double balanceScore = 0;
            var layerSizes = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.LayerIndex)
                .Select(g => g.Count())
                .ToList();

            if (layerSizes.Count > 1)
            {
                double avgSize = layerSizes.Average();
                double variance = layerSizes.Average(s => (s - avgSize) * (s - avgSize));
                double cv = avgSize > 0 ? Math.Sqrt(variance) / avgSize : 0;
                balanceScore = Math.Max(0, 1.0 - cv);
            }

            double health = Math.Clamp(
                0.4 * (1 - Math.Min(1, errorPenalty)) +
                structureBonus +
                0.15 * connectivityScore +
                0.15 * balanceScore,
                0, 1);

            return health;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Analytics Dashboard

    /// <summary>
    /// Provides comprehensive analytics and reporting for evolution runs.
    /// Generates detailed reports, statistics, and performance metrics.
    /// </summary>
    public sealed class EvolutionAnalyticsDashboard
    {
        private readonly EvolutionConfig _config;
        private readonly EvolutionHistoryTracker _history;
        private readonly SpeciationAnalytics _speciationAnalytics;
        private readonly EvolutionDiagnostics _diagnostics;

        /// <summary>
        /// Initializes a new instance of the EvolutionAnalyticsDashboard class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="history">Evolution history tracker.</param>
        /// <param name="speciationAnalytics">Speciation analytics.</param>
        /// <param name="diagnostics">Evolution diagnostics.</param>
        public EvolutionAnalyticsDashboard(
            EvolutionConfig config,
            EvolutionHistoryTracker history,
            SpeciationAnalytics speciationAnalytics,
            EvolutionDiagnostics diagnostics)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _history = history ?? throw new ArgumentNullException(nameof(history));
            _speciationAnalytics = speciationAnalytics ?? throw new ArgumentNullException(nameof(speciationAnalytics));
            _diagnostics = diagnostics ?? throw new ArgumentNullException(nameof(diagnostics));
        }

        /// <summary>
        /// Generates a comprehensive evolution report.
        /// </summary>
        /// <returns>A detailed report string.</returns>
        public string GenerateReport()
        {
            var summary = _history.GetSummary();
            var metrics = _history.GetMetricsHistory();
            var mutationRates = _diagnostics.GetMutationSuccessRates();
            var crossoverRates = _diagnostics.GetCrossoverSuccessRates();

            var sb = new StringBuilder();
            sb.AppendLine("=== NEAT-G Evolution Report ===");
            sb.AppendLine($"Total Generations: {summary.TotalGenerations}");
            sb.AppendLine($"Total Evaluations: {summary.TotalEvaluations:N0}");
            sb.AppendLine($"Best Fitness Ever: {summary.BestFitnessEver:F6}");
            sb.AppendLine($"Best Fitness Generation: {summary.BestFitnessGeneration}");
            sb.AppendLine($"Initial Fitness: {summary.InitialFitness:F6}");
            sb.AppendLine($"Final Fitness: {summary.FinalFitness:F6}");
            sb.AppendLine($"Fitness Improvement: {summary.FitnessImprovement:F6}");
            sb.AppendLine($"Peak Species Count: {summary.PeakSpeciesCount}");
            sb.AppendLine($"Final Species Count: {summary.FinalSpeciesCount}");
            sb.AppendLine($"Average Diversity: {summary.AverageDiversity:F4}");
            sb.AppendLine($"Total Events: {summary.TotalEvents}");
            sb.AppendLine();

            if (metrics.Count > 0)
            {
                sb.AppendLine("--- Fitness Progression ---");
                int reportInterval = Math.Max(1, metrics.Count / 20);
                for (int i = 0; i < metrics.Count; i += reportInterval)
                {
                    var m = metrics[i];
                    sb.AppendLine($"Gen {m.Generation,5}: Best={m.BestFitness:F4}, Avg={m.AverageFitness:F4}, " +
                                 $"Species={m.SpeciesCount}, Diversity={m.DiversityMetric:F3}, " +
                                 $"Evals={m.EvaluationsThisGeneration}");
                }
                sb.AppendLine();
            }

            if (mutationRates.Count > 0)
            {
                sb.AppendLine("--- Mutation Success Rates ---");
                foreach (var kvp in mutationRates.OrderByDescending(k => k.Value))
                {
                    sb.AppendLine($"  {kvp.Key}: {kvp.Value:P1}");
                }
                sb.AppendLine();
            }

            if (crossoverRates.Count > 0)
            {
                sb.AppendLine("--- Crossover Success Rates ---");
                foreach (var kvp in crossoverRates.OrderByDescending(k => k.Value))
                {
                    sb.AppendLine($"  {kvp.Key}: {kvp.Value:P1}");
                }
                sb.AppendLine();
            }

            sb.AppendLine("--- Configuration ---");
            sb.AppendLine($"  Population Size: {_config.PopulationSize}");
            sb.AppendLine($"  Max Generations: {_config.MaxGenerations}");
            sb.AppendLine($"  Crossover Rate: {_config.CrossoverRate}");
            sb.AppendLine($"  Mutation Rate: {_config.MutationRate}");
            sb.AppendLine($"  Speciation Threshold: {_config.SpeciationThreshold}");
            sb.AppendLine($"  Target Species: {_config.TargetSpeciesCount}");
            sb.AppendLine($"  Selection Method: {_config.ParentSelection}");
            sb.AppendLine($"  Speciation Method: {_config.SpeciationMethod}");

            return sb.ToString();
        }

        /// <summary>
        /// Generates convergence analysis data.
        /// </summary>
        /// <returns>Convergence analysis results.</returns>
        public ConvergenceAnalysis AnalyzeConvergence()
        {
            var metrics = _history.GetMetricsHistory();
            if (metrics.Count < 2)
            {
                return new ConvergenceAnalysis { HasConverged = false };
            }

            var fitnesses = metrics.Select(m => m.BestFitness).ToList();
            var recentWindow = fitnesses.Skip(Math.Max(0, fitnesses.Count - 20)).ToList();

            double recentMean = recentWindow.Average();
            double recentVariance = recentWindow.Average(f => (f - recentMean) * (f - recentMean));
            double recentStdDev = Math.Sqrt(recentVariance);

            double overallRange = fitnesses.Max() - fitnesses.Min();
            double convergenceRatio = overallRange > 0 ? recentStdDev / overallRange : 0;

            bool hasConverged = convergenceRatio < 0.01 && recentWindow.Count >= 10;

            int stagnationStart = -1;
            double plateauFitness = 0;
            for (int i = fitnesses.Count - 1; i >= 1; i--)
            {
                if (Math.Abs(fitnesses[i] - fitnesses[i - 1]) < _config.FitnessThreshold)
                {
                    if (stagnationStart < 0)
                        stagnationStart = i;
                    plateauFitness = fitnesses[i];
                }
                else
                {
                    break;
                }
            }

            double fitnessGrowthRate = 0;
            if (metrics.Count >= 10)
            {
                var first10 = metrics.Take(10).Select(m => m.BestFitness).ToList();
                var last10 = metrics.TakeLast(10).Select(m => m.BestFitness).ToList();
                fitnessGrowthRate = (last10.Average() - first10.Average()) / Math.Max(1, first10.Average());
            }

            return new ConvergenceAnalysis
            {
                HasConverged = hasConverged,
                ConvergenceRatio = convergenceRatio,
                RecentStdDev = recentStdDev,
                OverallRange = overallRange,
                StagnationStart = stagnationStart,
                PlateauFitness = plateauFitness,
                FitnessGrowthRate = fitnessGrowthRate,
                EstimatedRemainingGenerations = EstimateRemainingGenerations(metrics)
            };
        }

        /// <summary>
        /// Generates species dynamics analysis.
        /// </summary>
        /// <returns>Species dynamics data.</returns>
        public SpeciesDynamicsAnalysis AnalyzeSpeciesDynamics()
        {
            var snapshots = _speciationAnalytics.GetSnapshots();
            var speciesOverTime = _speciationAnalytics.GetSpeciesCountOverTime();

            double speciationRate = _speciationAnalytics.ComputeSpeciationRate();
            double extinctionRate = _speciationAnalytics.ComputeExtinctionRate();

            int peakSpecies = speciesOverTime.Count > 0 ? speciesOverTime.Max(s => s.SpeciesCount) : 0;
            int minSpecies = speciesOverTime.Count > 0 ? speciesOverTime.Min(s => s.SpeciesCount) : 0;
            double avgSpecies = speciesOverTime.Count > 0 ? speciesOverTime.Average(s => s.SpeciesCount) : 0;

            double speciesStability = 0;
            if (speciesOverTime.Count > 1)
            {
                var counts = speciesOverTime.Select(s => s.SpeciesCount).ToList();
                double mean = counts.Average();
                double variance = counts.Average(c => (c - mean) * (c - mean));
                speciesStability = mean > 0 ? 1.0 - Math.Sqrt(variance) / mean : 0;
            }

            var fitnessBySpecies = new Dictionary<int, List<double>>();
            foreach (var snapshot in snapshots)
            {
                for (int i = 0; i < snapshot.SpeciesBestFitness.Length; i++)
                {
                    int speciesId = i;
                    if (!fitnessBySpecies.ContainsKey(speciesId))
                        fitnessBySpecies[speciesId] = new List<double>();
                    fitnessBySpecies[speciesId].Add(snapshot.SpeciesBestFitness[i]);
                }
            }

            var speciesDominance = fitnessBySpecies
                .Select(kvp => new
                {
                    SpeciesId = kvp.Key,
                    MaxFitness = kvp.Value.Max(),
                    AvgFitness = kvp.Value.Average(),
                    Longevity = kvp.Value.Count
                })
                .OrderByDescending(s => s.MaxFitness)
                .ToList();

            return new SpeciesDynamicsAnalysis
            {
                SpeciationRate = speciationRate,
                ExtinctionRate = extinctionRate,
                PeakSpeciesCount = peakSpecies,
                MinSpeciesCount = minSpecies,
                AverageSpeciesCount = avgSpecies,
                SpeciesStability = speciesStability,
                SpeciesDominanceRanking = speciesDominance.Select(s => s.SpeciesId).ToList().AsReadOnly()
            };
        }

        /// <summary>
        /// Generates mutation effectiveness analysis.
        /// </summary>
        /// <returns>Mutation effectiveness data.</returns>
        public MutationEffectivenessAnalysis AnalyzeMutationEffectiveness()
        {
            var mutationRates = _diagnostics.GetMutationSuccessRates();
            var crossoverRates = _diagnostics.GetCrossoverSuccessRates();

            var snapshots = _diagnostics.GetSnapshots();
            var diversityOverTime = snapshots.Select(s => s.PopulationDiversity).ToList();
            var structuralOverTime = snapshots.Select(s => s.StructuralDiversity).ToList();

            double avgMutationRate = mutationRates.Count > 0 ? mutationRates.Values.Average() : 0;
            double avgCrossoverRate = crossoverRates.Count > 0 ? crossoverRates.Values.Average() : 0;

            var mostEffectiveMutation = mutationRates.Count > 0
                ? mutationRates.OrderByDescending(k => k.Value).First()
                : new KeyValuePair<MutationType, double>(MutationType.None, 0);

            var leastEffectiveMutation = mutationRates.Count > 0
                ? mutationRates.OrderBy(k => k.Value).First()
                : new KeyValuePair<MutationType, double>(MutationType.None, 0);

            double diversityTrend = 0;
            if (diversityOverTime.Count >= 10)
            {
                var firstHalf = diversityOverTime.Take(diversityOverTime.Count / 2).Average();
                var secondHalf = diversityOverTime.Skip(diversityOverTime.Count / 2).Average();
                diversityTrend = secondHalf - firstHalf;
            }

            double weightDiversity = 0;
            if (snapshots.Count > 0)
            {
                weightDiversity = snapshots.Average(s => s.WeightDiversity);
            }

            return new MutationEffectivenessAnalysis
            {
                AverageMutationSuccessRate = avgMutationRate,
                AverageCrossoverSuccessRate = avgCrossoverRate,
                MostEffectiveMutation = mostEffectiveMutation.Key,
                MostEffectiveMutationRate = mostEffectiveMutation.Value,
                LeastEffectiveMutation = leastEffectiveMutation.Key,
                LeastEffectiveMutationRate = leastEffectiveMutation.Value,
                DiversityTrend = diversityTrend,
                AverageWeightDiversity = weightDiversity,
                MutationTypeRates = mutationRates.ToImmutableDictionary(),
                CrossoverStrategyRates = crossoverRates.ToImmutableDictionary()
            };
        }

        /// <summary>
        /// Generates a fitness landscape summary.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <returns>Fitness landscape analysis.</returns>
        public FitnessLandscapeAnalysis AnalyzeFitnessLandscape(GenomePopulation population)
        {
            if (population.Genomes.Length == 0)
            {
                return new FitnessLandscapeAnalysis();
            }

            var fitnesses = population.Genomes.Select(g => g.Fitness).ToArray();
            double mean = fitnesses.Average();
            double variance = fitnesses.Average(f => (f - mean) * (f - mean));
            double stdDev = Math.Sqrt(variance);

            double skewness = 0;
            double kurtosis = 0;
            if (stdDev > 0)
            {
                skewness = fitnesses.Average(f => Math.Pow((f - mean) / stdDev, 3));
                kurtosis = fitnesses.Average(f => Math.Pow((f - mean) / stdDev, 4)) - 3;
            }

            var sorted = fitnesses.OrderBy(f => f).ToArray();
            double q1 = sorted[sorted.Length / 4];
            double median = sorted[sorted.Length / 2];
            double q3 = sorted[3 * sorted.Length / 4];
            double iqr = q3 - q1;

            int outlierCount = fitnesses.Count(f => f < q1 - 1.5 * iqr || f > q3 + 1.5 * iqr);

            double fitnessEntropy = ComputeFitnessEntropy(fitnesses, 20);

            bool isMultimodal = ComputeModesCount(fitnesses) > 1;

            return new FitnessLandscapeAnalysis
            {
                Mean = mean,
                Variance = variance,
                StandardDeviation = stdDev,
                Skewness = skewness,
                Kurtosis = kurtosis,
                Q1 = q1,
                Median = median,
                Q3 = q3,
                IQR = iqr,
                OutlierCount = outlierCount,
                FitnessEntropy = fitnessEntropy,
                IsMultimodal = isMultimodal,
                ModeCount = ComputeModesCount(fitnesses)
            };
        }

        private int EstimateRemainingGenerations(IReadOnlyList<EvolutionMetrics> metrics)
        {
            if (metrics.Count < 10)
                return -1;

            var recent = metrics.TakeLast(10).ToList();
            double avgImprovement = 0;
            for (int i = 1; i < recent.Count; i++)
            {
                avgImprovement += recent[i].BestFitness - recent[i - 1].BestFitness;
            }
            avgImprovement /= (recent.Count - 1);

            if (Math.Abs(avgImprovement) < _config.FitnessThreshold)
                return 0;

            double targetGap = _config.TargetFitness - recent.Last().BestFitness;
            if (targetGap <= 0)
                return 0;

            return avgImprovement > 0 ? (int)Math.Ceiling(targetGap / avgImprovement) : -1;
        }

        private double ComputeFitnessEntropy(double[] fitnesses, int bins)
        {
            if (fitnesses.Length == 0 || bins <= 0)
                return 0;

            double min = fitnesses.Min();
            double max = fitnesses.Max();
            double range = max - min;

            if (range < 1e-10)
                return 0;

            var counts = new int[bins];
            foreach (var f in fitnesses)
            {
                int bin = Math.Min(bins - 1, (int)((f - min) / range * bins));
                counts[bin]++;
            }

            double entropy = 0;
            foreach (var count in counts)
            {
                if (count > 0)
                {
                    double p = (double)count / fitnesses.Length;
                    entropy -= p * Math.Log2(p);
                }
            }

            return entropy / Math.Log2(bins);
        }

        private int ComputeModesCount(double[] fitnesses)
        {
            if (fitnesses.Length < 5)
                return 1;

            double min = fitnesses.Min();
            double max = fitnesses.Max();
            double range = max - min;

            if (range < 1e-10)
                return 1;

            int bins = Math.Max(5, (int)Math.Sqrt(fitnesses.Length));
            var counts = new int[bins];

            foreach (var f in fitnesses)
            {
                int bin = Math.Min(bins - 1, (int)((f - min) / range * bins));
                counts[bin]++;
            }

            int modes = 0;
            for (int i = 1; i < counts.Length - 1; i++)
            {
                if (counts[i] > counts[i - 1] && counts[i] > counts[i + 1] && counts[i] >= 3)
                {
                    modes++;
                }
            }

            return Math.Max(1, modes);
        }
    }

    /// <summary>
    /// Convergence analysis results.
    /// </summary>
    public record ConvergenceAnalysis
    {
        /// <summary>Whether the evolution has converged.</summary>
        public bool HasConverged { get; init; }

        /// <summary>Convergence ratio (lower = more converged).</summary>
        public double ConvergenceRatio { get; init; }

        /// <summary>Recent fitness standard deviation.</summary>
        public double RecentStdDev { get; init; }

        /// <summary>Overall fitness range.</summary>
        public double OverallRange { get; init; }

        /// <summary>Generation when stagnation started (-1 if not stagnant).</summary>
        public int StagnationStart { get; init; }

        /// <summary>Fitness at the plateau.</summary>
        public double PlateauFitness { get; init; }

        /// <summary>Rate of fitness growth.</summary>
        public double FitnessGrowthRate { get; init; }

        /// <summary>Estimated remaining generations to target (-1 if unknown).</summary>
        public int EstimatedRemainingGenerations { get; init; }
    }

    /// <summary>
    /// Species dynamics analysis results.
    /// </summary>
    public record SpeciesDynamicsAnalysis
    {
        /// <summary>Rate of new species formation.</summary>
        public double SpeciationRate { get; init; }

        /// <summary>Rate of species extinction.</summary>
        public double ExtinctionRate { get; init; }

        /// <summary>Peak species count.</summary>
        public int PeakSpeciesCount { get; init; }

        /// <summary>Minimum species count.</summary>
        public int MinSpeciesCount { get; init; }

        /// <summary>Average species count.</summary>
        public double AverageSpeciesCount { get; init; }

        /// <summary>Species count stability (0-1).</summary>
        public double SpeciesStability { get; init; }

        /// <summary>Species ranked by fitness dominance.</summary>
        public IReadOnlyList<int> SpeciesDominanceRanking { get; init; } = Array.Empty<int>();
    }

    /// <summary>
    /// Mutation effectiveness analysis results.
    /// </summary>
    public record MutationEffectivenessAnalysis
    {
        /// <summary>Average mutation success rate.</summary>
        public double AverageMutationSuccessRate { get; init; }

        /// <summary>Average crossover success rate.</summary>
        public double AverageCrossoverSuccessRate { get; init; }

        /// <summary>Most effective mutation type.</summary>
        public MutationType MostEffectiveMutation { get; init; }

        /// <summary>Success rate of most effective mutation.</summary>
        public double MostEffectiveMutationRate { get; init; }

        /// <summary>Least effective mutation type.</summary>
        public MutationType LeastEffectiveMutation { get; init; }

        /// <summary>Success rate of least effective mutation.</summary>
        public double LeastEffectiveMutationRate { get; init; }

        /// <summary>Trend in population diversity (positive = increasing).</summary>
        public double DiversityTrend { get; init; }

        /// <summary>Average weight diversity across snapshots.</summary>
        public double AverageWeightDiversity { get; init; }

        /// <summary>Per-type mutation success rates.</summary>
        public ImmutableDictionary<MutationType, double> MutationTypeRates { get; init; } =
            ImmutableDictionary<MutationType, double>.Empty;

        /// <summary>Per-strategy crossover success rates.</summary>
        public ImmutableDictionary<string, double> CrossoverStrategyRates { get; init; } =
            ImmutableDictionary<string, double>.Empty;
    }

    /// <summary>
    /// Fitness landscape analysis results.
    /// </summary>
    public record FitnessLandscapeAnalysis
    {
        /// <summary>Mean fitness.</summary>
        public double Mean { get; init; }

        /// <summary>Fitness variance.</summary>
        public double Variance { get; init; }

        /// <summary>Fitness standard deviation.</summary>
        public double StandardDeviation { get; init; }

        /// <summary>Fitness skewness.</summary>
        public double Skewness { get; init; }

        /// <summary>Fitness kurtosis.</summary>
        public double Kurtosis { get; init; }

        /// <summary>First quartile.</summary>
        public double Q1 { get; init; }

        /// <summary>Median fitness.</summary>
        public double Median { get; init; }

        /// <summary>Third quartile.</summary>
        public double Q3 { get; init; }

        /// <summary>Interquartile range.</summary>
        public double IQR { get; init; }

        /// <summary>Number of outlier genomes.</summary>
        public int OutlierCount { get; init; }

        /// <summary>Fitness entropy (0-1, higher = more diverse).</summary>
        public double FitnessEntropy { get; init; }

        /// <summary>Whether the fitness landscape is multimodal.</summary>
        public bool IsMultimodal { get; init; }

        /// <summary>Estimated number of fitness modes.</summary>
        public int ModeCount { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Thread-Safe Genome Pool

    /// <summary>
    /// Thread-safe genome pool for concurrent access during parallel evolution.
    /// Uses lock-free data structures for high-performance concurrent operations.
    /// </summary>
    public sealed class ConcurrentGenomePool
    {
        private readonly ConcurrentDictionary<Guid, GeoGenome> _pool;
        private readonly ConcurrentQueue<Guid> _availableIds;
        private readonly ConcurrentBag<Guid> _recycledIds;
        private long _totalCount;
        private long _activeCount;

        /// <summary>
        /// Initializes a new instance of the ConcurrentGenomePool class.
        /// </summary>
        public ConcurrentGenomePool()
        {
            _pool = new ConcurrentDictionary<Guid, GeoGenome>();
            _availableIds = new ConcurrentQueue<Guid>();
            _recycledIds = new ConcurrentBag<Guid>();
        }

        /// <summary>Total genomes ever added to the pool.</summary>
        public long TotalCount => Interlocked.Read(ref _totalCount);

        /// <summary>Currently active genomes in the pool.</summary>
        public long ActiveCount => Interlocked.Read(ref _activeCount);

        /// <summary>Number of available (unclaimed) genomes.</summary>
        public int AvailableCount => _availableIds.Count;

        /// <summary>
        /// Adds a genome to the pool.
        /// </summary>
        /// <param name="genome">The genome to add.</param>
        public void Add(GeoGenome genome)
        {
            if (_pool.TryAdd(genome.Id, genome))
            {
                _availableIds.Enqueue(genome.Id);
                Interlocked.Increment(ref _totalCount);
                Interlocked.Increment(ref _activeCount);
            }
        }

        /// <summary>
        /// Adds multiple genomes to the pool.
        /// </summary>
        /// <param name="genomes">Genomes to add.</param>
        public void AddRange(IEnumerable<GeoGenome> genomes)
        {
            foreach (var genome in genomes)
            {
                Add(genome);
            }
        }

        /// <summary>
        /// Tries to claim a genome from the pool.
        /// </summary>
        /// <param name="genome">The claimed genome, or default if none available.</param>
        /// <returns>True if a genome was successfully claimed.</returns>
        public bool TryClaim(out GeoGenome? genome)
        {
            genome = null;

            while (_availableIds.TryDequeue(out var id))
            {
                if (_pool.TryGetValue(id, out var g))
                {
                    genome = g;
                    Interlocked.Decrement(ref _activeCount);
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Claims a specific genome by ID.
        /// </summary>
        /// <param name="id">Genome ID to claim.</param>
        /// <returns>The claimed genome, or null if not found.</returns>
        public GeoGenome? ClaimById(Guid id)
        {
            if (_pool.TryRemove(id, out var genome))
            {
                Interlocked.Decrement(ref _activeCount);
                return genome;
            }
            return null;
        }

        /// <summary>
        /// Returns a genome to the pool for reuse.
        /// </summary>
        /// <param name="genome">The genome to return.</param>
        public void Return(GeoGenome genome)
        {
            genome.InvalidateFitness();
            if (_pool.TryAdd(genome.Id, genome))
            {
                _availableIds.Enqueue(genome.Id);
                Interlocked.Increment(ref _activeCount);
            }
        }

        /// <summary>
        /// Removes a genome from the pool permanently.
        /// </summary>
        /// <param name="id">Genome ID to remove.</param>
        /// <returns>True if the genome was removed.</returns>
        public bool Remove(Guid id)
        {
            if (_pool.TryRemove(id, out _))
            {
                Interlocked.Decrement(ref _activeCount);
                _recycledIds.Add(id);
                return true;
            }
            return false;
        }

        /// <summary>
        /// Gets a genome by ID without claiming it.
        /// </summary>
        /// <param name="id">Genome ID.</param>
        /// <returns>The genome, or null if not found.</returns>
        public GeoGenome? Peek(Guid id)
        {
            return _pool.TryGetValue(id, out var genome) ? genome : null;
        }

        /// <summary>
        /// Gets all genomes in the pool.
        /// </summary>
        public IReadOnlyList<GeoGenome> GetAll()
        {
            return _pool.Values.ToList().AsReadOnly();
        }

        /// <summary>
        /// Clears the pool.
        /// </summary>
        public void Clear()
        {
            _pool.Clear();
            while (_availableIds.TryDequeue(out _))
            { }
            Interlocked.Exchange(ref _activeCount, 0);
        }

        /// <summary>
        /// Gets the top N genomes by fitness.
        /// </summary>
        /// <param name="count">Number of top genomes to retrieve.</param>
        public IReadOnlyList<GeoGenome> GetTop(int count)
        {
            return _pool.Values
                .OrderByDescending(g => g.Fitness)
                .Take(count)
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Gets genomes matching a predicate.
        /// </summary>
        /// <param name="predicate">Filter predicate.</param>
        public IReadOnlyList<GeoGenome> Where(Func<GeoGenome, bool> predicate)
        {
            return _pool.Values.Where(predicate).ToList().AsReadOnly();
        }

        /// <summary>
        /// Performs a bulk operation on all genomes in the pool.
        /// </summary>
        /// <param name="operation">Operation to perform on each genome.</param>
        public void ForEach(Action<GeoGenome> operation)
        {
            foreach (var genome in _pool.Values)
            {
                operation(genome);
            }
        }

        /// <summary>
        /// Performs a parallel bulk operation on all genomes in the pool.
        /// </summary>
        /// <param name="operation">Operation to perform on each genome.</param>
        /// <param name="maxDegreeOfParallelism">Maximum parallel threads.</param>
        public void ParallelForEach(Action<GeoGenome> operation, int maxDegreeOfParallelism = -1)
        {
            var options = new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism
            };
            Parallel.ForEach(_pool.Values, options, operation);
        }

        /// <summary>
        /// Computes aggregate statistics for all genomes in the pool.
        /// </summary>
        public PoolStatistics GetStatistics()
        {
            var genomes = _pool.Values.ToList();
            if (genomes.Count == 0)
                return new PoolStatistics();

            var fitnesses = genomes.Select(g => g.Fitness).ToArray();
            return new PoolStatistics
            {
                Count = genomes.Count,
                MeanFitness = fitnesses.Average(),
                BestFitness = fitnesses.Max(),
                WorstFitness = fitnesses.Min(),
                AvgNeuronCount = genomes.Average(g => g.ActiveNeuronCount),
                AvgSynapseCount = genomes.Average(g => g.ActiveSynapseCount),
                AvgComplexity = genomes.Average(g => g.Complexity),
                UniqueTopologies = genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count()
            };
        }
    }

    /// <summary>
    /// Statistics for a genome pool.
    /// </summary>
    public record PoolStatistics
    {
        /// <summary>Number of genomes in the pool.</summary>
        public int Count { get; init; }

        /// <summary>Mean fitness.</summary>
        public double MeanFitness { get; init; }

        /// <summary>Best fitness.</summary>
        public double BestFitness { get; init; }

        /// <summary>Worst fitness.</summary>
        public double WorstFitness { get; init; }

        /// <summary>Average neuron count.</summary>
        public double AvgNeuronCount { get; init; }

        /// <summary>Average synapse count.</summary>
        public double AvgSynapseCount { get; init; }

        /// <summary>Average complexity.</summary>
        public double AvgComplexity { get; init; }

        /// <summary>Number of unique topologies.</summary>
        public int UniqueTopologies { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Multi-Objective Optimization

    /// <summary>
    /// Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) for
    /// multi-objective optimization of genome fitness.
    /// Provides Pareto-optimal solutions for conflicting objectives.
    /// </summary>
    public sealed class NSGAIISelector
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the NSGAIISelector class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public NSGAIISelector(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Performs non-dominated sorting of the population.
        /// </summary>
        /// <param name="population">Population to sort.</param>
        /// <returns>List of fronts, each containing genome indices.</returns>
        public IReadOnlyList<IReadOnlyList<int>> NonDominatedSorting(GenomePopulation population)
        {
            var genomes = population.Genomes.ToArray();
            int n = genomes.Length;
            var dominationCount = new int[n];
            var dominatedSet = new List<int>[n];
            var fronts = new List<IReadOnlyList<int>>();

            for (int i = 0; i < n; i++)
            {
                dominatedSet[i] = new List<int>();
                dominationCount[i] = 0;
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (Domominates(genomes[i], genomes[j]))
                    {
                        dominatedSet[i].Add(j);
                        dominationCount[j]++;
                    }
                    else if (Domominates(genomes[j], genomes[i]))
                    {
                        dominatedSet[j].Add(i);
                        dominationCount[i]++;
                    }
                }
            }

            var currentFront = new List<int>();
            for (int i = 0; i < n; i++)
            {
                if (dominationCount[i] == 0)
                {
                    currentFront.Add(i);
                }
            }

            while (currentFront.Count > 0)
            {
                fronts.Add(currentFront.AsReadOnly());

                var nextFront = new List<int>();
                foreach (int i in currentFront)
                {
                    foreach (int j in dominatedSet[i])
                    {
                        dominationCount[j]--;
                        if (dominationCount[j] == 0)
                        {
                            nextFront.Add(j);
                        }
                    }
                }
                currentFront = nextFront;
            }

            return fronts;
        }

        /// <summary>
        /// Computes crowding distances for a front.
        /// </summary>
        /// <param name="front">Indices of genomes in the front.</param>
        /// <param name="population">The population.</param>
        /// <returns>Crowding distance for each genome in the front.</returns>
        public IReadOnlyList<double> ComputeCrowdingDistances(
            IReadOnlyList<int> front,
            GenomePopulation population)
        {
            int frontSize = front.Count;
            var distances = new double[frontSize];

            if (frontSize <= 2)
            {
                for (int i = 0; i < frontSize; i++)
                    distances[i] = double.PositiveInfinity;
                return distances;
            }

            var objectives = Enum.GetValues<FitnessComponent>();

            foreach (var objective in objectives)
            {
                var genomeObjectives = front
                    .Select((idx, pos) => (Index: pos, Value: GetObjectiveValue(population.Genomes[idx], objective)))
                    .OrderBy(x => x.Value)
                    .ToList();

                distances[genomeObjectives[0].Index] = double.PositiveInfinity;
                distances[genomeObjectives[^1].Index] = double.PositiveInfinity;

                double range = genomeObjectives[^1].Value - genomeObjectives[0].Value;
                if (range < 1e-10)
                    continue;

                for (int i = 1; i < frontSize - 1; i++)
                {
                    double gap = genomeObjectives[i + 1].Value - genomeObjectives[i - 1].Value;
                    distances[genomeObjectives[i].Index] += gap / range;
                }
            }

            return distances.ToList().AsReadOnly();
        }

        /// <summary>
        /// Selects parents using NSGA-II tournament selection.
        /// </summary>
        /// <param name="population">Population to select from.</param>
        /// <param name="count">Number of parents to select.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>Selected parent genomes.</returns>
        public IReadOnlyList<GeoGenome> Select(GenomePopulation population, int count, Random rng)
        {
            var fronts = NonDominatedSorting(population);
            var selected = new List<GeoGenome>();

            var frontRanks = new int[population.Genomes.Length];
            for (int f = 0; f < fronts.Count; f++)
            {
                foreach (int idx in fronts[f])
                {
                    frontRanks[idx] = f;
                }
            }

            var crowdingDistances = new double[population.Genomes.Length];
            for (int f = 0; f < fronts.Count; f++)
            {
                var distances = ComputeCrowdingDistances(fronts[f], population);
                for (int i = 0; i < fronts[f].Count; i++)
                {
                    crowdingDistances[fronts[f][i]] = distances[i];
                }
            }

            for (int i = 0; i < count; i++)
            {
                int candidate1 = rng.Next(population.Genomes.Length);
                int candidate2 = rng.Next(population.Genomes.Length);

                bool candidate1Better = fronts.Count == 0 ||
                    frontRanks[candidate1] < frontRanks[candidate2] ||
                    (frontRanks[candidate1] == frontRanks[candidate2] &&
                     crowdingDistances[candidate1] > crowdingDistances[candidate2]);

                int winner = candidate1Better ? candidate1 : candidate2;
                selected.Add(population.Genomes[winner]);
            }

            return selected;
        }

        private bool Domominates(GeoGenome a, GeoGenome b)
        {
            bool atLeastOneBetter = false;

            foreach (var component in Enum.GetValues<FitnessComponent>())
            {
                double valA = GetObjectiveValue(a, component);
                double valB = GetObjectiveValue(b, component);

                if (valA < valB)
                    return false;
                if (valA > valB)
                    atLeastOneBetter = true;
            }

            return atLeastOneBetter;
        }

        private double GetObjectiveValue(GeoGenome genome, FitnessComponent component)
        {
            if (genome.FitnessComponents.TryGetValue(component, out double value))
                return value;
            return genome.Fitness;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Genome Repair

    /// <summary>
    /// Provides genome repair utilities to fix structural issues and ensure validity.
    /// </summary>
    public sealed class GenomeRepairer
    {
        private readonly EvolutionConfig _config;
        private readonly InnovationNumberGenerator _innovationGenerator;

        /// <summary>
        /// Initializes a new instance of the GenomeRepairer class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="innovationGenerator">Innovation number generator.</param>
        public GenomeRepairer(EvolutionConfig config, InnovationNumberGenerator innovationGenerator)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _innovationGenerator = innovationGenerator ?? throw new ArgumentNullException(nameof(innovationGenerator));
        }

        /// <summary>
        /// Repairs a genome to ensure structural integrity.
        /// </summary>
        /// <param name="genome">Genome to repair.</param>
        /// <returns>Repaired genome.</returns>
        public GeoGenome Repair(GeoGenome genome)
        {
            var repaired = genome.Clone();
            repaired.InvalidateFitness();

            RemoveDanglingSynapses(repaired);
            RepairDisconnectedOutputs(repaired);
            RemoveOrphanedNeurons(repaired);
            FixLayerIndices(repaired);
            RepairDuplicateConnections(repaired);
            EnsureMinimumConnectivity(repaired);

            repaired.ComputeComplexity();
            return repaired;
        }

        /// <summary>
        /// Removes synapses that reference non-existent or inactive neurons.
        /// </summary>
        private void RemoveDanglingSynapses(GeoGenome genome)
        {
            var activeNeuronIds = genome.Neurons
                .Where(n => n.IsActive)
                .Select(n => n.InnovationNumber)
                .ToHashSet();

            foreach (var synapse in genome.Synapses)
            {
                if (synapse.IsActive)
                {
                    if (!activeNeuronIds.Contains(synapse.SourceNeuronId) ||
                        !activeNeuronIds.Contains(synapse.TargetNeuronId))
                    {
                        synapse.IsActive = false;
                    }
                }
            }
        }

        /// <summary>
        /// Ensures all output neurons have at least one active input connection.
        /// </summary>
        private void RepairDisconnectedOutputs(GeoGenome genome)
        {
            var outputNeurons = genome.Neurons
                .Where(n => n.IsActive && n.LayerIndex >= genome.MaxLayerDepth)
                .ToList();

            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            foreach (var outputNeuron in outputNeurons)
            {
                bool hasInput = activeSynapses.Any(s => s.TargetNeuronId == outputNeuron.InnovationNumber);
                if (!hasInput)
                {
                    var inputNeurons = genome.Neurons
                        .Where(n => n.IsActive && n.LayerIndex < outputNeuron.LayerIndex)
                        .ToList();

                    if (inputNeurons.Count > 0)
                    {
                        var source = inputNeurons[new Random().Next(inputNeurons.Count)];
                        var newSynapse = new GeoSynapse
                        {
                            InnovationNumber = _innovationGenerator.GetSynapseInnovation(
                                source.InnovationNumber, outputNeuron.InnovationNumber),
                            SourceNeuronId = source.InnovationNumber,
                            TargetNeuronId = outputNeuron.InnovationNumber,
                            Weight = (_config.WeightInitRange * 0.1),
                            IsActive = true,
                            CreationGeneration = genome.Generation
                        };
                        genome.Synapses.Add(newSynapse);
                        activeSynapses.Add(newSynapse);
                    }
                }
            }
        }

        /// <summary>
        /// Removes neurons that have no connections (orphaned).
        /// </summary>
        private void RemoveOrphanedNeurons(GeoGenome genome)
        {
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();
            var connectedNeuronIds = new HashSet<long>();

            foreach (var synapse in activeSynapses)
            {
                connectedNeuronIds.Add(synapse.SourceNeuronId);
                connectedNeuronIds.Add(synapse.TargetNeuronId);
            }

            foreach (var neuron in genome.Neurons)
            {
                if (neuron.IsActive && neuron.LayerIndex > 0 && neuron.LayerIndex < genome.MaxLayerDepth)
                {
                    if (!connectedNeuronIds.Contains(neuron.InnovationNumber))
                    {
                        neuron.IsActive = false;
                    }
                }
            }
        }

        /// <summary>
        /// Fixes layer indices to ensure proper ordering.
        /// </summary>
        private void FixLayerIndices(GeoGenome genome)
        {
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();

            bool changed = true;
            int iterations = 0;

            while (changed && iterations < 100)
            {
                changed = false;
                iterations++;

                foreach (var synapse in activeSynapses)
                {
                    var source = activeNeurons.FirstOrDefault(n => n.InnovationNumber == synapse.SourceNeuronId);
                    var target = activeNeurons.FirstOrDefault(n => n.InnovationNumber == synapse.TargetNeuronId);

                    if (source != null && target != null && !synapse.IsRecurrent)
                    {
                        if (target.LayerIndex <= source.LayerIndex)
                        {
                            target.LayerIndex = source.LayerIndex + 1;
                            changed = true;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Removes duplicate connections between same neuron pairs.
        /// </summary>
        private void RepairDuplicateConnections(GeoGenome genome)
        {
            var seenConnections = new HashSet<(long, long)>();
            var toDeactivate = new List<GeoSynapse>();

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                var key = (synapse.SourceNeuronId, synapse.TargetNeuronId);
                if (!seenConnections.Add(key))
                {
                    toDeactivate.Add(synapse);
                }
            }

            foreach (var synapse in toDeactivate)
            {
                synapse.IsActive = false;
            }
        }

        /// <summary>
        /// Ensures minimum connectivity in the genome.
        /// </summary>
        private void EnsureMinimumConnectivity(GeoGenome genome)
        {
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();

            foreach (var neuron in activeNeurons)
            {
                if (neuron.LayerIndex == 0)
                    continue;

                int inputCount = activeSynapses.Count(s => s.TargetNeuronId == neuron.InnovationNumber);
                int outputCount = activeSynapses.Count(s => s.SourceNeuronId == neuron.InnovationNumber);

                if (inputCount == 0 && outputCount == 0 && neuron.LayerIndex < genome.MaxLayerDepth)
                {
                    var inputCandidates = activeNeurons
                        .Where(n => n.LayerIndex < neuron.LayerIndex)
                        .ToList();

                    if (inputCandidates.Count > 0)
                    {
                        var source = inputCandidates[new Random().Next(inputCandidates.Count)];
                        var newSynapse = new GeoSynapse
                        {
                            InnovationNumber = _innovationGenerator.GetSynapseInnovation(
                                source.InnovationNumber, neuron.InnovationNumber),
                            SourceNeuronId = source.InnovationNumber,
                            TargetNeuronId = neuron.InnovationNumber,
                            Weight = _config.WeightInitRange * 0.1,
                            IsActive = true,
                            CreationGeneration = genome.Generation
                        };
                        genome.Synapses.Add(newSynapse);
                    }
                }
            }
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Event Bus

    /// <summary>
    /// Central event bus for the NEAT-G evolution engine.
    /// Provides publish-subscribe pattern for decoupled communication between
    /// evolution components. Supports typed event handlers and async subscriptions.
    /// </summary>
    public sealed class EvolutionEventBus : IAsyncDisposable
    {
        private readonly ConcurrentDictionary<Type, List<Delegate>> _handlers;
        private readonly ConcurrentQueue<(Type EventType, object EventData, DateTime Timestamp)> _eventLog;
        private readonly int _maxLogSize;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the EvolutionEventBus class.
        /// </summary>
        /// <param name="maxLogSize">Maximum number of events to retain in the log.</param>
        public EvolutionEventBus(int maxLogSize = 10000)
        {
            _handlers = new ConcurrentDictionary<Type, List<Delegate>>();
            _eventLog = new ConcurrentQueue<(Type, object, DateTime)>();
            _maxLogSize = maxLogSize;
        }

        /// <summary>Total events published.</summary>
        private long _totalEventsPublished;
        public long TotalEventsPublished => _totalEventsPublished;

        /// <summary>
        /// Subscribes a handler for a specific event type.
        /// </summary>
        /// <typeparam name="TEvent">The event type.</typeparam>
        /// <param name="handler">The event handler.</param>
        public void Subscribe<TEvent>(Action<TEvent> handler) where TEvent : class
        {
            var eventType = typeof(TEvent);
            _handlers.AddOrUpdate(eventType,
                _ => new List<Delegate> { handler },
                (_, existing) =>
                {
                    lock (existing)
                    {
                        existing.Add(handler);
                    }
                    return existing;
                });
        }

        /// <summary>
        /// Subscribes an async handler for a specific event type.
        /// </summary>
        /// <typeparam name="TEvent">The event type.</typeparam>
        /// <param name="handler">The async event handler.</param>
        public void SubscribeAsync<TEvent>(Func<TEvent, Task> handler) where TEvent : class
        {
            var eventType = typeof(TEvent);
            _handlers.AddOrUpdate(eventType,
                _ => new List<Delegate> { handler },
                (_, existing) =>
                {
                    lock (existing)
                    {
                        existing.Add(handler);
                    }
                    return existing;
                });
        }

        /// <summary>
        /// Unsubscribes a handler for a specific event type.
        /// </summary>
        /// <typeparam name="TEvent">The event type.</typeparam>
        /// <param name="handler">The handler to unsubscribe.</param>
        public void Unsubscribe<TEvent>(Action<TEvent> handler) where TEvent : class
        {
            var eventType = typeof(TEvent);
            if (_handlers.TryGetValue(eventType, out var handlers))
            {
                lock (handlers)
                {
                    handlers.Remove(handler);
                }
            }
        }

        /// <summary>
        /// Publishes an event to all subscribed handlers.
        /// </summary>
        /// <typeparam name="TEvent">The event type.</typeparam>
        /// <param name="eventData">The event data.</param>
        public async Task PublishAsync<TEvent>(TEvent eventData) where TEvent : class
        {
            Interlocked.Increment(ref _totalEventsPublished);

            _eventLog.Enqueue((typeof(TEvent), eventData, DateTime.UtcNow));
            while (_eventLog.Count > _maxLogSize)
            {
                _eventLog.TryDequeue(out _);
            }

            if (!_handlers.TryGetValue(typeof(TEvent), out var handlers))
                return;

            List<Delegate> snapshot;
            lock (handlers)
            {
                snapshot = handlers.ToList();
            }

            foreach (var handler in snapshot)
            {
                try
                {
                    if (handler is Action<TEvent> syncHandler)
                    {
                        syncHandler(eventData);
                    }
                    else if (handler is Func<TEvent, Task> asyncHandler)
                    {
                        await asyncHandler(eventData).ConfigureAwait(false);
                    }
                }
                catch (Exception)
                {
                    // Event handler exceptions are swallowed to prevent cascading failures
                }
            }
        }

        /// <summary>
        /// Publishes an event synchronously (non-async handlers only).
        /// </summary>
        /// <typeparam name="TEvent">The event type.</typeparam>
        /// <param name="eventData">The event data.</param>
        public void Publish<TEvent>(TEvent eventData) where TEvent : class
        {
            Interlocked.Increment(ref _totalEventsPublished);

            _eventLog.Enqueue((typeof(TEvent), eventData, DateTime.UtcNow));
            while (_eventLog.Count > _maxLogSize)
            {
                _eventLog.TryDequeue(out _);
            }

            if (!_handlers.TryGetValue(typeof(TEvent), out var handlers))
                return;

            List<Delegate> snapshot;
            lock (handlers)
            {
                snapshot = handlers.ToList();
            }

            foreach (var handler in snapshot)
            {
                try
                {
                    if (handler is Action<TEvent> syncHandler)
                    {
                        syncHandler(eventData);
                    }
                }
                catch (Exception)
                {
                    // Event handler exceptions are swallowed
                }
            }
        }

        /// <summary>
        /// Gets the event log filtered by event type.
        /// </summary>
        /// <typeparam name="TEvent">Event type to filter by.</typeparam>
        /// <param name="count">Maximum number of events to return.</param>
        /// <returns>Filtered events.</returns>
        public IReadOnlyList<(TEvent Data, DateTime Timestamp)> GetEventLog<TEvent>(int count = 100) where TEvent : class
        {
            return _eventLog
                .Where(e => e.EventType == typeof(TEvent))
                .Take(count)
                .Select(e => ((TEvent)e.EventData, e.Timestamp))
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Clears the event log.
        /// </summary>
        public void ClearLog()
        {
            while (_eventLog.TryDequeue(out _))
            { }
        }

        /// <summary>
        /// Gets the number of handlers for a specific event type.
        /// </summary>
        /// <typeparam name="TEvent">Event type.</typeparam>
        public int GetHandlerCount<TEvent>() where TEvent : class
        {
            if (_handlers.TryGetValue(typeof(TEvent), out var handlers))
            {
                lock (handlers)
                {
                    return handlers.Count;
                }
            }
            return 0;
        }

        /// <inheritdoc/>
        public ValueTask DisposeAsync()
        {
            if (_disposed)
                return ValueTask.CompletedTask;
            _disposed = true;
            _handlers.Clear();
            ClearLog();
            return ValueTask.CompletedTask;
        }
    }

    /// <summary>
    /// Event data for genome evaluation completion.
    /// </summary>
    public sealed class GenomeEvaluatedEvent
    {
        /// <summary>The evaluated genome.</summary>
        public GeoGenome Genome { get; init; } = null!;

        /// <summary>Time taken for evaluation.</summary>
        public TimeSpan EvaluationTime { get; init; }

        /// <summary>Generation when evaluation occurred.</summary>
        public int Generation { get; init; }
    }

    /// <summary>
    /// Event data for species creation.
    /// </summary>
    public sealed class SpeciesCreatedEvent
    {
        /// <summary>New species information.</summary>
        public SpeciesInfo Species { get; init; } = default!;

        /// <summary>Generation when created.</summary>
        public int Generation { get; init; }

        /// <summary>Number of initial members.</summary>
        public int InitialMemberCount { get; init; }
    }

    /// <summary>
    /// Event data for species extinction.
    /// </summary>
    public sealed class SpeciesExtinctEvent
    {
        /// <summary>Extinct species information.</summary>
        public SpeciesInfo Species { get; init; } = default!;

        /// <summary>Generation when extinct.</summary>
        public int Generation { get; init; }

        /// <summary>Reason for extinction.</summary>
        public string Reason { get; init; } = string.Empty;

        /// <summary>Number of members reassignment.</summary>
        public int MembersReassigned { get; init; }
    }

    /// <summary>
    /// Event data for evolution phase transition.
    /// </summary>
    public sealed class PhaseTransitionEvent
    {
        /// <summary>Previous evolution phase.</summary>
        public string FromPhase { get; init; } = string.Empty;

        /// <summary>New evolution phase.</summary>
        public string ToPhase { get; init; } = string.Empty;

        /// <summary>Generation of transition.</summary>
        public int Generation { get; init; }

        /// <summary>Duration of previous phase.</summary>
        public TimeSpan PhaseDuration { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Genome Cache

    /// <summary>
    /// LRU (Least Recently Used) cache for genome fitness evaluations.
    /// Prevents redundant evaluations by caching previously computed fitness values.
    /// Thread-safe for concurrent access during parallel evaluation.
    /// </summary>
    public sealed class GenomeFitnessCache
    {
        private readonly ConcurrentDictionary<string, CacheEntry> _cache;
        private readonly int _maxSize;
        private readonly object _evictionLock = new();
        private long _hits;
        private long _misses;

        /// <summary>
        /// Initializes a new instance of the GenomeFitnessCache class.
        /// </summary>
        /// <param name="maxSize">Maximum number of cache entries.</param>
        public GenomeFitnessCache(int maxSize = 10000)
        {
            _maxSize = maxSize;
            _cache = new ConcurrentDictionary<string, CacheEntry>();
        }

        /// <summary>Cache hit count.</summary>
        public long Hits => Interlocked.Read(ref _hits);

        /// <summary>Cache miss count.</summary>
        public long Misses => Interlocked.Read(ref _misses);

        /// <summary>Cache hit rate.</summary>
        public double HitRate
        {
            get
            {
                long total = Hits + Misses;
                return total > 0 ? (double)Hits / total : 0;
            }
        }

        /// <summary>Current cache size.</summary>
        public int Size => _cache.Count;

        /// <summary>
        /// Tries to get a cached fitness value for a genome.
        /// </summary>
        /// <param name="genome">The genome to look up.</param>
        /// <param name="fitness">The cached fitness value, if found.</param>
        /// <returns>True if found in cache.</returns>
        public bool TryGetFitness(GeoGenome genome, out double fitness)
        {
            string key = ComputeCacheKey(genome);
            if (_cache.TryGetValue(key, out var entry))
            {
                entry.LastAccessTime = DateTime.UtcNow;
                Interlocked.Increment(ref _hits);
                fitness = entry.Fitness;
                return true;
            }

            Interlocked.Increment(ref _misses);
            fitness = 0;
            return false;
        }

        /// <summary>
        /// Adds a fitness value to the cache.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="fitness">The fitness value.</param>
        public void AddFitness(GeoGenome genome, double fitness)
        {
            string key = ComputeCacheKey(genome);
            var entry = new CacheEntry
            {
                Fitness = fitness,
                LastAccessTime = DateTime.UtcNow,
                InsertionTime = DateTime.UtcNow
            };

            _cache[key] = entry;

            if (_cache.Count > _maxSize)
            {
                EvictOldest();
            }
        }

        /// <summary>
        /// Removes a genome from the cache.
        /// </summary>
        /// <param name="genome">The genome to remove.</param>
        /// <returns>True if removed.</returns>
        public bool Remove(GeoGenome genome)
        {
            string key = ComputeCacheKey(genome);
            return _cache.TryRemove(key, out _);
        }

        /// <summary>
        /// Clears the entire cache.
        /// </summary>
        public void Clear()
        {
            _cache.Clear();
        }

        /// <summary>
        /// Gets cache statistics.
        /// </summary>
        public CacheStatistics GetStatistics()
        {
            return new CacheStatistics
            {
                Size = Size,
                MaxSize = _maxSize,
                Hits = Hits,
                Misses = Misses,
                HitRate = HitRate,
                OldestEntry = _cache.Values
                    .OrderBy(e => e.InsertionTime)
                    .FirstOrDefault()?.InsertionTime ?? DateTime.MinValue,
                NewestEntry = _cache.Values
                    .OrderByDescending(e => e.InsertionTime)
                    .FirstOrDefault()?.InsertionTime ?? DateTime.MinValue
            };
        }

        private string ComputeCacheKey(GeoGenome genome)
        {
            long topologyHash = genome.ComputeTopologyHash();
            long weightHash = 0;
            foreach (var synapse in genome.Synapses.Where(s => s.IsActive).OrderBy(s => s.InnovationNumber))
            {
                weightHash = HashCode.Combine(weightHash, synapse.Weight.GetHashCode());
            }
            return $"{topologyHash}_{weightHash}";
        }

        private void EvictOldest()
        {
            lock (_evictionLock)
            {
                if (_cache.Count <= _maxSize)
                    return;

                var toRemove = _cache
                    .OrderBy(kvp => kvp.Value.LastAccessTime)
                    .Take(_cache.Count - _maxSize + 100)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var key in toRemove)
                {
                    _cache.TryRemove(key, out _);
                }
            }
        }

        private sealed class CacheEntry
        {
            public double Fitness { get; set; }
            public DateTime LastAccessTime { get; set; }
            public DateTime InsertionTime { get; set; }
        }
    }

    /// <summary>
    /// Cache statistics.
    /// </summary>
    public record CacheStatistics
    {
        /// <summary>Current cache size.</summary>
        public int Size { get; init; }

        /// <summary>Maximum cache size.</summary>
        public int MaxSize { get; init; }

        /// <summary>Total cache hits.</summary>
        public long Hits { get; init; }

        /// <summary>Total cache misses.</summary>
        public long Misses { get; init; }

        /// <summary>Cache hit rate.</summary>
        public double HitRate { get; init; }

        /// <summary>Timestamp of oldest entry.</summary>
        public DateTime OldestEntry { get; init; }

        /// <summary>Timestamp of newest entry.</summary>
        public DateTime NewestEntry { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Configuration Presets

    /// <summary>
    /// Provides pre-configured evolution settings for common use cases.
    /// </summary>
    public static class EvolutionPresets
    {
        /// <summary>
        /// Configuration for image classification tasks.
        /// Optimized for discovering efficient CNN-like architectures.
        /// </summary>
        public static EvolutionConfig ImageClassification()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 400;
            config.CrossoverRate = 0.7;
            config.MutationRate = 0.3;
            config.SpeciationThreshold = 3.5;
            config.TargetSpeciesCount = 12;
            config.MaxStagnationGenerations = 25;
            config.PerturbationMagnitude = 0.12;
            config.LandmarkCount = 25;
            config.SemanticEmbeddingDimension = 48;
            config.EnableMigration = true;
            config.MigrationRate = 0.06;
            config.MigrationInterval = 8;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 15;
            config.ParentSelection = SelectionMethod.Tournament;
            config.TournamentSize = 6;
            config.SurvivalSelection = SelectionMethod.Truncation;
            config.Objective = FitnessObjective.Maximize;
            return config;
        }

        /// <summary>
        /// Configuration for regression tasks.
        /// Balanced exploration and exploitation.
        /// </summary>
        public static EvolutionConfig Regression()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 300;
            config.CrossoverRate = 0.75;
            config.MutationRate = 0.25;
            config.SpeciationThreshold = 3.0;
            config.TargetSpeciesCount = 10;
            config.MaxStagnationGenerations = 20;
            config.PerturbationMagnitude = 0.1;
            config.LandmarkCount = 20;
            config.SemanticEmbeddingDimension = 32;
            config.EnableMigration = true;
            config.MigrationRate = 0.05;
            config.MigrationInterval = 10;
            config.UseAdaptiveMutation = true;
            config.Objective = FitnessObjective.Maximize;
            return config;
        }

        /// <summary>
        /// Configuration for time series prediction.
        /// Emphasizes recurrent structures and temporal patterns.
        /// </summary>
        public static EvolutionConfig TimeSeries()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 350;
            config.CrossoverRate = 0.65;
            config.MutationRate = 0.35;
            config.SpeciationThreshold = 4.0;
            config.TargetSpeciesCount = 8;
            config.MaxStagnationGenerations = 30;
            config.PerturbationMagnitude = 0.15;
            config.LandmarkCount = 15;
            config.SemanticEmbeddingDimension = 40;
            config.EnableMigration = true;
            config.MigrationRate = 0.08;
            config.MigrationInterval = 7;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 12;
            config.ParentSelection = SelectionMethod.RankBased;
            return config;
        }

        /// <summary>
        /// Configuration for reinforcement learning policy evolution.
        /// High mutation rate for exploration of action spaces.
        /// </summary>
        public static EvolutionConfig ReinforcementLearning()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 500;
            config.CrossoverRate = 0.6;
            config.MutationRate = 0.4;
            config.SpeciationThreshold = 5.0;
            config.TargetSpeciesCount = 15;
            config.MaxStagnationGenerations = 40;
            config.PerturbationMagnitude = 0.2;
            config.LandmarkCount = 30;
            config.SemanticEmbeddingDimension = 64;
            config.EnableMigration = true;
            config.MigrationRate = 0.1;
            config.MigrationInterval = 5;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 20;
            config.MaxAdaptiveMutationRate = 0.6;
            config.ParentSelection = SelectionMethod.Tournament;
            config.TournamentSize = 7;
            return config;
        }

        /// <summary>
        /// Configuration for neural architecture search (NAS).
        /// Focuses on discovering novel topologies.
        /// </summary>
        public static EvolutionConfig NeuralArchitectureSearch()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 600;
            config.CrossoverRate = 0.5;
            config.MutationRate = 0.5;
            config.SpeciationThreshold = 6.0;
            config.TargetSpeciesCount = 20;
            config.MaxStagnationGenerations = 50;
            config.PerturbationMagnitude = 0.25;
            config.LandmarkCount = 40;
            config.SemanticEmbeddingDimension = 64;
            config.EnableMigration = true;
            config.MigrationRate = 0.12;
            config.MigrationInterval = 5;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 25;
            config.MaxAdaptiveMutationRate = 0.7;
            config.ParentSelection = SelectionMethod.StochasticUniversal;
            config.SurvivalSelection = SelectionMethod.Truncation;
            config.EliteFraction = 0.03;
            return config;
        }

        /// <summary>
        /// Configuration for compact/mobile model evolution.
        /// Aggressive parsimony pressure for small models.
        /// </summary>
        public static EvolutionConfig CompactModel()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 200;
            config.CrossoverRate = 0.8;
            config.MutationRate = 0.2;
            config.SpeciationThreshold = 2.5;
            config.TargetSpeciesCount = 6;
            config.MaxStagnationGenerations = 15;
            config.PerturbationMagnitude = 0.08;
            config.LandmarkCount = 15;
            config.SemanticEmbeddingDimension = 24;
            config.EnableMigration = false;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 8;
            config.ParentSelection = SelectionMethod.Tournament;
            config.TournamentSize = 4;
            return config;
        }

        /// <summary>
        /// Configuration for real-time/low-latency evolution.
        /// Small population for fast iteration.
        /// </summary>
        public static EvolutionConfig RealTime()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 100;
            config.MaxGenerations = 500;
            config.CrossoverRate = 0.75;
            config.MutationRate = 0.25;
            config.SpeciationThreshold = 2.0;
            config.TargetSpeciesCount = 5;
            config.MaxStagnationGenerations = 10;
            config.PerturbationMagnitude = 0.08;
            config.LandmarkCount = 10;
            config.SemanticEmbeddingDimension = 16;
            config.EnableMigration = false;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 5;
            config.EvaluationTimeoutMs = 5000;
            config.EnableHistoryTracking = false;
            config.ParentSelection = SelectionMethod.Tournament;
            config.TournamentSize = 3;
            return config;
        }

        /// <summary>
        /// Configuration for exploring completely unknown fitness landscapes.
        /// Maximum diversity and exploration.
        /// </summary>
        public static EvolutionConfig ExploreUnknown()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 800;
            config.CrossoverRate = 0.5;
            config.MutationRate = 0.5;
            config.SpeciationThreshold = 7.0;
            config.TargetSpeciesCount = 25;
            config.MaxStagnationGenerations = 60;
            config.PerturbationMagnitude = 0.3;
            config.LandmarkCount = 50;
            config.SemanticEmbeddingDimension = 64;
            config.EnableMigration = true;
            config.MigrationRate = 0.15;
            config.MigrationInterval = 3;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 30;
            config.MaxAdaptiveMutationRate = 0.8;
            config.MaxSpeciesCount = 100;
            config.ParentSelection = SelectionMethod.StochasticUniversal;
            config.SurvivalSelection = SelectionMethod.Truncation;
            config.EliteFraction = 0.02;
            config.RandomSeed = 42;
            return config;
        }

        /// <summary>
        /// Configuration for fine-tuning a pre-evolved genome.
        /// Very conservative mutations around the existing solution.
        /// </summary>
        public static EvolutionConfig FineTune()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 150;
            config.CrossoverRate = 0.85;
            config.MutationRate = 0.15;
            config.SpeciationThreshold = 1.5;
            config.TargetSpeciesCount = 3;
            config.MaxStagnationGenerations = 10;
            config.PerturbationMagnitude = 0.03;
            config.PerturbationDecayRate = 0.98;
            config.MinPerturbationMagnitude = 0.0005;
            config.LandmarkCount = 10;
            config.SemanticEmbeddingDimension = 24;
            config.EnableMigration = false;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 5;
            config.MaxAdaptiveMutationRate = 0.2;
            config.ParentSelection = SelectionMethod.Tournament;
            config.TournamentSize = 3;
            return config;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Timers

    /// <summary>
    /// High-resolution timer for tracking evolution phase durations.
    /// Uses Stopwatch for precise timing of evolution operations.
    /// </summary>
    public sealed class EvolutionTimers
    {
        private readonly ConcurrentDictionary<string, PhaseTimer> _timers;
        private readonly List<PhaseTimingRecord> _records;

        /// <summary>
        /// Initializes a new instance of the EvolutionTimers class.
        /// </summary>
        public EvolutionTimers()
        {
            _timers = new ConcurrentDictionary<string, PhaseTimer>();
            _records = new List<PhaseTimingRecord>();
        }

        /// <summary>
        /// Starts timing a phase.
        /// </summary>
        /// <param name="phaseName">Name of the phase.</param>
        public void StartPhase(string phaseName)
        {
            var timer = _timers.GetOrAdd(phaseName, _ => new PhaseTimer());
            timer.Restart();
        }

        /// <summary>
        /// Stops timing a phase and records the result.
        /// </summary>
        /// <param name="phaseName">Name of the phase.</param>
        /// <param name="generation">Current generation.</param>
        public void StopPhase(string phaseName, int generation)
        {
            if (_timers.TryGetValue(phaseName, out var timer))
            {
                timer.Stop();
                lock (_records)
                {
                    _records.Add(new PhaseTimingRecord
                    {
                        PhaseName = phaseName,
                        Duration = timer.Elapsed,
                        Generation = generation,
                        Timestamp = DateTime.UtcNow
                    });
                }
            }
        }

        /// <summary>
        /// Gets the total time spent in a specific phase.
        /// </summary>
        /// <param name="phaseName">Phase name.</param>
        /// <returns>Total time spent.</returns>
        public TimeSpan GetTotalTime(string phaseName)
        {
            lock (_records)
            {
                return _records
                    .Where(r => r.PhaseName == phaseName)
                    .Aggregate(TimeSpan.Zero, (sum, r) => sum + r.Duration);
            }
        }

        /// <summary>
        /// Gets the average time for a specific phase.
        /// </summary>
        /// <param name="phaseName">Phase name.</param>
        /// <returns>Average time per occurrence.</returns>
        public TimeSpan GetAverageTime(string phaseName)
        {
            lock (_records)
            {
                var phaseRecords = _records.Where(r => r.PhaseName == phaseName).ToList();
                return phaseRecords.Count > 0
                    ? TimeSpan.FromTicks((long)phaseRecords.Average(r => r.Duration.Ticks))
                    : TimeSpan.Zero;
            }
        }

        /// <summary>
        /// Gets all timing records.
        /// </summary>
        public IReadOnlyList<PhaseTimingRecord> GetRecords()
        {
            lock (_records)
            {
                return _records.ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Gets timing summary for all phases.
        /// </summary>
        public IReadOnlyDictionary<string, (TimeSpan Total, TimeSpan Average, int Count)> GetSummary()
        {
            lock (_records)
            {
                return _records
                    .GroupBy(r => r.PhaseName)
                    .ToDictionary(
                        g => g.Key,
                        g => (
                            Total: TimeSpan.FromTicks(g.Sum(r => r.Duration.Ticks)),
                            Average: TimeSpan.FromTicks((long)g.Average(r => r.Duration.Ticks)),
                            Count: g.Count()
                        ));
            }
        }

        /// <summary>
        /// Clears all timing records.
        /// </summary>
        public void Clear()
        {
            lock (_records)
            {
                _records.Clear();
            }
            _timers.Clear();
        }

        private sealed class PhaseTimer
        {
            private readonly Stopwatch _stopwatch = new();

            public TimeSpan Elapsed => _stopwatch.Elapsed;

            public void Restart()
            {
                _stopwatch.Restart();
            }

            public void Stop()
            {
                _stopwatch.Stop();
            }
        }
    }

    /// <summary>
    /// Record of a phase timing measurement.
    /// </summary>
    public record PhaseTimingRecord
    {
        /// <summary>Name of the phase.</summary>
        public string PhaseName { get; init; } = string.Empty;

        /// <summary>Duration of the phase.</summary>
        public TimeSpan Duration { get; init; }

        /// <summary>Generation when timing occurred.</summary>
        public int Generation { get; init; }

        /// <summary>Timestamp of the measurement.</summary>
        public DateTime Timestamp { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Callbacks

    /// <summary>
    /// Provides callback hooks for monitoring and controlling the evolution process.
    /// Allows injection of custom logic at various points in the evolution loop.
    /// </summary>
    public sealed class EvolutionCallbacks
    {
        /// <summary>
        /// Callback invoked before each generation starts.
        /// Return false to skip this generation.
        /// </summary>
        public Func<int, GenomePopulation, bool>? OnBeforeGeneration { get; set; }

        /// <summary>
        /// Callback invoked after each generation completes.
        /// </summary>
        public Action<int, GenomePopulation, EvolutionMetrics>? OnAfterGeneration { get; set; }

        /// <summary>
        /// Callback invoked when new offspring are produced.
        /// Can modify offspring before evaluation.
        /// </summary>
        public Func<IReadOnlyList<GeoGenome>, IReadOnlyList<GeoGenome>>? OnOffspringProduced { get; set; }

        /// <summary>
        /// Callback invoked before fitness evaluation.
        /// Can modify the evaluation context.
        /// </summary>
        public Func<EvaluationContext, EvaluationContext>? OnBeforeEvaluation { get; set; }

        /// <summary>
        /// Callback invoked after fitness evaluation.
        /// Can modify genome fitness values.
        /// </summary>
        public Action<GeoGenome>? OnAfterEvaluation { get; set; }

        /// <summary>
        /// Callback invoked when speciation occurs.
        /// Can modify species assignments.
        /// </summary>
        public Func<ImmutableArray<SpeciesInfo>, ImmutableArray<SpeciesInfo>>? OnAfterSpeciation { get; set; }

        /// <summary>
        /// Callback invoked when a new best fitness is achieved.
        /// </summary>
        public Action<int, GeoGenome>? OnNewBestFitness { get; set; }

        /// <summary>
        /// Callback invoked when stagnation is detected.
        /// Can trigger custom recovery mechanisms.
        /// </summary>
        public Func<int, bool>? OnStagnationDetected { get; set; }

        /// <summary>
        /// Callback invoked when mutation rates are adjusted.
        /// </summary>
        public Action<double, double>? OnMutationRateAdjusted { get; set; }

        /// <summary>
        /// Callback invoked when migration occurs.
        /// </summary>
        public Action<MigrationEvent>? OnMigration { get; set; }

        /// <summary>
        /// Callback invoked when a species is created.
        /// </summary>
        public Action<SpeciesInfo>? OnSpeciesCreated { get; set; }

        /// <summary>
        /// Callback invoked when a species goes extinct.
        /// </summary>
        public Action<SpeciesInfo>? OnSpeciesExtinct { get; set; }

        /// <summary>
        /// Callback for custom genome initialization.
        /// Can provide domain-specific initial genomes.
        /// </summary>
        public Func<int, int, GeoGenome>? OnInitializeGenome { get; set; }

        /// <summary>
        /// Callback invoked for progress reporting.
        /// </summary>
        public IProgress<EvolutionMetrics>? ProgressReporter { get; set; }

        /// <summary>
        /// Callback for custom fitness post-processing.
        /// </summary>
        public Func<GeoGenome, GeoGenome>? OnFitnessPostProcess { get; set; }

        /// <summary>
        /// Callback for monitoring resource usage.
        /// </summary>
        public Action<long, TimeSpan>? OnResourceCheck { get; set; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Control

    /// <summary>
    /// Provides external control over a running evolution process.
    /// Supports pausing, resuming, speed adjustment, and parameter modification.
    /// </summary>
    public sealed class EvolutionController : IDisposable
    {
        private readonly ManualResetEventSlim _pauseEvent;
        private readonly object _parameterLock = new();
        private volatile bool _isPaused;
        private volatile bool _shouldStop;
        private double _speedMultiplier;
        private int _maxGenerationsOverride;
        private double _targetFitnessOverride;

        /// <summary>
        /// Initializes a new instance of the EvolutionController class.
        /// </summary>
        public EvolutionController()
        {
            _pauseEvent = new ManualResetEventSlim(true);
            _isPaused = false;
            _shouldStop = false;
            _speedMultiplier = 1.0;
            _maxGenerationsOverride = -1;
            _targetFitnessOverride = double.MaxValue;
        }

        /// <summary>Whether evolution is currently paused.</summary>
        public bool IsPaused => _isPaused;

        /// <summary>Whether evolution should stop.</summary>
        public bool ShouldStop => _shouldStop;

        /// <summary>Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed).</summary>
        public double SpeedMultiplier
        {
            get => Volatile.Read(ref _speedMultiplier);
            set
            {
                lock (_parameterLock)
                {
                    _speedMultiplier = Math.Max(0.1, Math.Min(10.0, value));
                }
            }
        }

        /// <summary>
        /// Pauses the evolution process.
        /// </summary>
        public void Pause()
        {
            _isPaused = true;
            _pauseEvent.Reset();
        }

        /// <summary>
        /// Resumes the evolution process.
        /// </summary>
        public void Resume()
        {
            _isPaused = false;
            _pauseEvent.Set();
        }

        /// <summary>
        /// Signals evolution to stop gracefully.
        /// </summary>
        public void Stop()
        {
            _shouldStop = true;
            _pauseEvent.Set();
        }

        /// <summary>
        /// Waits if paused. Call this at the start of each generation.
        /// </summary>
        public void WaitIfPaused()
        {
            _pauseEvent.Wait();
        }

        /// <summary>
        /// Gets the delay to apply based on speed multiplier.
        /// </summary>
        /// <returns>Delay in milliseconds.</returns>
        public int GetSpeedDelay()
        {
            double speed = Volatile.Read(ref _speedMultiplier);
            if (speed >= 1.0)
                return 0;
            return (int)(100.0 / speed);
        }

        /// <summary>
        /// Overrides the maximum generations for the current run.
        /// </summary>
        /// <param name="maxGenerations">New maximum (-1 to use config default).</param>
        public void OverrideMaxGenerations(int maxGenerations)
        {
            Interlocked.Exchange(ref _maxGenerationsOverride, maxGenerations);
        }

        /// <summary>
        /// Overrides the target fitness for the current run.
        /// </summary>
        /// <param name="targetFitness">New target fitness.</param>
        public void OverrideTargetFitness(double targetFitness)
        {
            lock (_parameterLock)
            {
                _targetFitnessOverride = targetFitness;
            }
        }

        /// <summary>
        /// Gets the current max generations override.
        /// </summary>
        public int GetMaxGenerationsOverride()
        {
            return Volatile.Read(ref _maxGenerationsOverride);
        }

        /// <summary>
        /// Gets the current target fitness override.
        /// </summary>
        public double GetTargetFitnessOverride()
        {
            lock (_parameterLock)
            {
                return _targetFitnessOverride;
            }
        }

        /// <summary>
        /// Resets the controller to initial state.
        /// </summary>
        public void Reset()
        {
            _shouldStop = false;
            _isPaused = false;
            _pauseEvent.Set();
            SpeedMultiplier = 1.0;
            _maxGenerationsOverride = -1;
            _targetFitnessOverride = double.MaxValue;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            _pauseEvent.Dispose();
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Advanced Fitness Evaluation Strategies

    /// <summary>
    /// Provides advanced multi-objective fitness evaluation strategies including
    /// Pareto optimization, lexicographic ordering, and goal programming approaches.
    /// </summary>
    public sealed class AdvancedFitnessStrategies
    {
        private readonly EvaluationContext _context;

        /// <summary>
        /// Initializes a new instance of the AdvancedFitnessStrategies class.
        /// </summary>
        /// <param name="context">Evaluation context.</param>
        public AdvancedFitnessStrategies(EvaluationContext context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
        }

        /// <summary>
        /// Evaluates genome fitness using weighted Tchebycheff scalarization.
        /// This approach minimizes the maximum weighted deviation from ideal points.
        /// </summary>
        /// <param name="genome">Genome to evaluate.</param>
        /// <param name="idealPoint">Ideal point for each objective.</param>
        /// <param name="weights">Weights for each objective.</param>
        /// <returns>Scalarized fitness value.</returns>
        public double TchebycheffScalarization(
            GeoGenome genome,
            ImmutableDictionary<FitnessComponent, double> idealPoint,
            ImmutableDictionary<FitnessComponent, double> weights)
        {
            double maxWeightedDeviation = 0;

            foreach (var component in genome.FitnessComponents)
            {
                if (!idealPoint.TryGetValue(component.Key, out double ideal))
                    ideal = 1.0;

                if (!weights.TryGetValue(component.Key, out double weight))
                    weight = 1.0;

                double deviation = Math.Abs(component.Value - ideal);
                double weightedDeviation = weight * deviation;
                maxWeightedDeviation = Math.Max(maxWeightedDeviation, weightedDeviation);
            }

            return -maxWeightedDeviation;
        }

        /// <summary>
        /// Evaluates genome fitness using the weighted sum approach.
        /// </summary>
        /// <param name="genome">Genome to evaluate.</param>
        /// <param name="weights">Objective weights.</param>
        /// <returns>Weighted sum fitness.</returns>
        public double WeightedSumFitness(
            GeoGenome genome,
            ImmutableDictionary<FitnessComponent, double> weights)
        {
            double totalFitness = 0;
            double totalWeight = 0;

            foreach (var component in genome.FitnessComponents)
            {
                if (weights.TryGetValue(component.Key, out double weight))
                {
                    totalFitness += component.Value * weight;
                    totalWeight += weight;
                }
            }

            return totalWeight > 0 ? totalFitness / totalWeight : 0;
        }

        /// <summary>
        /// Evaluates genome fitness using epsilon-constraint method.
        /// Primary objective is optimized while other objectives are treated as constraints.
        /// </summary>
        /// <param name="genome">Genome to evaluate.</param>
        /// <param name="primaryObjective">Primary objective to optimize.</param>
        /// <param name="constraints">Constraint bounds for other objectives (minimum acceptable values).</param>
        /// <returns>Constrained fitness value.</returns>
        public double EpsilonConstraintFitness(
            GeoGenome genome,
            FitnessComponent primaryObjective,
            ImmutableDictionary<FitnessComponent, double> constraints)
        {
            if (!genome.FitnessComponents.TryGetValue(primaryObjective, out double primaryValue))
                return 0;

            double penalty = 0;
            foreach (var constraint in constraints)
            {
                if (constraint.Key == primaryObjective)
                    continue;

                if (genome.FitnessComponents.TryGetValue(constraint.Key, out double value))
                {
                    if (value < constraint.Value)
                    {
                        penalty += (constraint.Value - value) * 10.0;
                    }
                }
            }

            return primaryValue - penalty;
        }

        /// <summary>
        /// Evaluates genome fitness using achievement scalarizing function.
        /// Balances proximity to reference point and distribution.
        /// </summary>
        /// <param name="genome">Genome to evaluate.</param>
        /// <param name="referencePoint">Reference point for each objective.</param>
        /// <param name="weights">Objective weights.</param>
        /// <param name="rho">Small positive constant to avoid division by zero.</param>
        /// <returns>Achievement scalarized fitness.</returns>
        public double AchievementScalarization(
            GeoGenome genome,
            ImmutableDictionary<FitnessComponent, double> referencePoint,
            ImmutableDictionary<FitnessComponent, double> weights,
            double rho = 0.001)
        {
            double maxAchievement = double.MinValue;

            foreach (var component in genome.FitnessComponents)
            {
                if (!referencePoint.TryGetValue(component.Key, out double reference))
                    reference = 0;

                if (!weights.TryGetValue(component.Key, out double weight))
                    weight = 1.0;

                double achievement = weight * (component.Value - reference);
                maxAchievement = Math.Max(maxAchievement, achievement);
            }

            double sumWeightedDeviations = 0;
            foreach (var component in genome.FitnessComponents)
            {
                if (referencePoint.TryGetValue(component.Key, out double reference) &&
                    weights.TryGetValue(component.Key, out double weight))
                {
                    sumWeightedDeviations += weight * (component.Value - reference);
                }
            }

            return maxAchievement + rho * sumWeightedDeviations;
        }

        /// <summary>
        /// Computes the hypervolume indicator for a set of genomes.
        /// Measures the volume of objective space dominated by the population.
        /// </summary>
        /// <param name="genomes">Population of genomes.</param>
        /// <param name="referencePoint">Reference point for hypervolume computation.</param>
        /// <returns>Hypervolume indicator value.</returns>
        public double ComputeHypervolume(
            IReadOnlyList<GeoGenome> genomes,
            ImmutableDictionary<FitnessComponent, double> referencePoint)
        {
            if (genomes.Count == 0)
                return 0;

            var objectives = referencePoint.Keys.ToList();
            if (objectives.Count == 0)
                return 0;

            var nonDominated = GetNonDominatedSet(genomes, objectives);
            if (nonDominated.Count == 0)
                return 0;

            if (objectives.Count == 1)
            {
                double minVal = nonDominated.Min(g =>
                    g.FitnessComponents.TryGetValue(objectives[0], out double v) ? v : 0);
                double refVal = referencePoint[objectives[0]];
                return Math.Max(0, refVal - minVal);
            }

            if (objectives.Count == 2)
            {
                return ComputeHypervolume2D(nonDominated, objectives, referencePoint);
            }

            return ComputeHypervolumeApproximation(nonDominated, objectives, referencePoint);
        }

        /// <summary>
        /// Computes the spacing metric for a set of genomes.
        /// Measures how evenly distributed the Pareto front is.
        /// </summary>
        /// <param name="genomes">Population of genomes.</param>
        /// <returns>Spacing metric (lower = more evenly distributed).</returns>
        public double ComputeSpacing(IReadOnlyList<GeoGenome> genomes)
        {
            if (genomes.Count <= 1)
                return 0;

            var objectives = genomes[0].FitnessComponents.Keys.ToList();
            if (objectives.Count == 0)
                return 0;

            var distances = new List<double>();

            for (int i = 0; i < genomes.Count; i++)
            {
                double minDist = double.MaxValue;
                for (int j = 0; j < genomes.Count; j++)
                {
                    if (i == j)
                        continue;
                    double dist = EuclideanDistance(genomes[i], genomes[j], objectives);
                    if (dist < minDist)
                        minDist = dist;
                }
                distances.Add(minDist);
            }

            double meanDist = distances.Average();
            double variance = distances.Average(d => (d - meanDist) * (d - meanDist));
            return Math.Sqrt(variance);
        }

        /// <summary>
        /// Computes the spread (delta) metric for a set of genomes.
        /// Measures the extent of the Pareto front coverage.
        /// </summary>
        /// <param name="genomes">Population of genomes.</param>
        /// <returns>Spread metric (lower = better spread).</returns>
        public double ComputeSpread(IReadOnlyList<GeoGenome> genomes)
        {
            if (genomes.Count <= 1)
                return 1;

            var objectives = genomes[0].FitnessComponents.Keys.ToList();
            if (objectives.Count == 0)
                return 1;

            var distances = new List<double>();
            for (int i = 0; i < genomes.Count; i++)
            {
                double minDist = double.MaxValue;
                for (int j = 0; j < genomes.Count; j++)
                {
                    if (i == j)
                        continue;
                    double dist = EuclideanDistance(genomes[i], genomes[j], objectives);
                    if (dist < minDist)
                        minDist = dist;
                }
                distances.Add(minDist);
            }

            double meanDist = distances.Average();
            if (meanDist < 1e-10)
                return 0;

            double df = ComputeBoundaryDistances(genomes, objectives);
            double dl = distances.Sum(d => Math.Abs(d - meanDist));

            return (df + dl) / (df + distances.Count * meanDist);
        }

        private double ComputeHypervolume2D(
            IReadOnlyList<GeoGenome> genomes,
            List<FitnessComponent> objectives,
            ImmutableDictionary<FitnessComponent, double> referencePoint)
        {
            var sorted = genomes
                .OrderBy(g => g.FitnessComponents.TryGetValue(objectives[0], out double v) ? v : 0)
                .ToList();

            double hypervolume = 0;
            double prevX = referencePoint[objectives[0]];

            foreach (var genome in sorted)
            {
                double x = genome.FitnessComponents.TryGetValue(objectives[0], out double xv) ? xv : 0;
                double y = genome.FitnessComponents.TryGetValue(objectives[1], out double yv) ? yv : 0;
                double refY = referencePoint[objectives[1]];

                double width = Math.Max(0, prevX - x);
                double height = Math.Max(0, refY - y);
                hypervolume += width * height;
                prevX = x;
            }

            return hypervolume;
        }

        private double ComputeHypervolumeApproximation(
            IReadOnlyList<GeoGenome> genomes,
            List<FitnessComponent> objectives,
            ImmutableDictionary<FitnessComponent, double> referencePoint)
        {
            double volume = 0;
            int samples = 1000;
            var rng = new Random(42);

            for (int s = 0; s < samples; s++)
            {
                var point = new Dictionary<FitnessComponent, double>();
                foreach (var obj in objectives)
                {
                    double refVal = referencePoint[obj];
                    point[obj] = rng.NextDouble() * refVal;
                }

                bool dominated = false;
                foreach (var genome in genomes)
                {
                    bool allBetter = true;
                    foreach (var obj in objectives)
                    {
                        double gv = genome.FitnessComponents.TryGetValue(obj, out double v) ? v : 0;
                        if (gv < point[obj])
                        {
                            allBetter = false;
                            break;
                        }
                    }
                    if (allBetter)
                    {
                        dominated = true;
                        break;
                    }
                }

                if (dominated)
                    volume++;
            }

            double totalVolume = 1;
            foreach (var obj in objectives)
            {
                totalVolume *= referencePoint[obj];
            }

            return (double)volume / samples * totalVolume;
        }

        private List<GeoGenome> GetNonDominatedSet(
            IReadOnlyList<GeoGenome> genomes,
            List<FitnessComponent> objectives)
        {
            var nonDominated = new List<GeoGenome>();

            foreach (var candidate in genomes)
            {
                bool isDominated = false;
                foreach (var other in genomes)
                {
                    if (candidate == other)
                        continue;

                    bool allOthersBetterOrEqual = true;
                    bool atLeastOneBetter = false;

                    foreach (var obj in objectives)
                    {
                        double candVal = candidate.FitnessComponents.TryGetValue(obj, out double cv) ? cv : 0;
                        double otherVal = other.FitnessComponents.TryGetValue(obj, out double ov) ? ov : 0;

                        if (otherVal < candVal)
                        {
                            allOthersBetterOrEqual = false;
                            break;
                        }
                        if (otherVal > candVal)
                        {
                            atLeastOneBetter = true;
                        }
                    }

                    if (allOthersBetterOrEqual && atLeastOneBetter)
                    {
                        isDominated = true;
                        break;
                    }
                }

                if (!isDominated)
                    nonDominated.Add(candidate);
            }

            return nonDominated;
        }

        private double EuclideanDistance(GeoGenome a, GeoGenome b, List<FitnessComponent> objectives)
        {
            double dist = 0;
            foreach (var obj in objectives)
            {
                double aVal = a.FitnessComponents.TryGetValue(obj, out double av) ? av : 0;
                double bVal = b.FitnessComponents.TryGetValue(obj, out double bv) ? bv : 0;
                dist += (aVal - bVal) * (aVal - bVal);
            }
            return Math.Sqrt(dist);
        }

        private double ComputeBoundaryDistances(
            IReadOnlyList<GeoGenome> genomes,
            List<FitnessComponent> objectives)
        {
            if (genomes.Count < 2)
                return 0;

            double totalDist = 0;
            foreach (var obj in objectives)
            {
                var values = genomes
                    .Select(g => g.FitnessComponents.TryGetValue(obj, out double v) ? v : 0)
                    .OrderBy(v => v)
                    .ToList();

                if (values.Count >= 2)
                {
                    totalDist += values[^1] - values[0];
                }
            }

            return totalDist;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution State Machine

    /// <summary>
    /// Implements a formal state machine for the NEAT-G evolution process.
    /// Ensures valid state transitions and provides deterministic behavior.
    /// </summary>
    public sealed class EvolutionStateMachine
    {
        private EvolutionState _currentState;
        private readonly Dictionary<(EvolutionState From, string Trigger), EvolutionState> _transitions;
        private readonly List<(EvolutionState From, EvolutionState To, string Trigger, DateTime Timestamp)> _transitionLog;

        /// <summary>
        /// Initializes a new instance of the EvolutionStateMachine class.
        /// </summary>
        public EvolutionStateMachine()
        {
            _currentState = EvolutionState.NotStarted;
            _transitions = new Dictionary<(EvolutionState, string), EvolutionState>();
            _transitionLog = new List<(EvolutionState, EvolutionState, string, DateTime)>();

            DefineTransitions();
        }

        /// <summary>Gets the current state.</summary>
        public EvolutionState CurrentState => _currentState;

        /// <summary>Gets the transition log.</summary>
        public IReadOnlyList<(EvolutionState From, EvolutionState To, string Trigger, DateTime Timestamp)> TransitionLog =>
            _transitionLog.AsReadOnly();

        /// <summary>
        /// Attempts a state transition.
        /// </summary>
        /// <param name="trigger">The trigger causing the transition.</param>
        /// <returns>True if transition was successful.</returns>
        public bool TryTransition(string trigger)
        {
            var key = (_currentState, trigger);
            if (_transitions.TryGetValue(key, out var newState))
            {
                var oldState = _currentState;
                _currentState = newState;
                _transitionLog.Add((oldState, newState, trigger, DateTime.UtcNow));
                return true;
            }
            return false;
        }

        /// <summary>
        /// Forces a state transition without checking validity.
        /// </summary>
        /// <param name="newState">The new state.</param>
        public void ForceTransition(EvolutionState newState)
        {
            var oldState = _currentState;
            _currentState = newState;
            _transitionLog.Add((oldState, newState, "Force", DateTime.UtcNow));
        }

        /// <summary>
        /// Checks if a transition is valid from the current state.
        /// </summary>
        /// <param name="trigger">The trigger to check.</param>
        /// <returns>True if the transition is valid.</summary>
        public bool CanTransition(string trigger)
        {
            return _transitions.ContainsKey((_currentState, trigger));
        }

        /// <summary>
        /// Gets all valid triggers from the current state.
        /// </summary>
        public IReadOnlyList<string> GetValidTriggers()
        {
            return _transitions
                .Where(kvp => kvp.Key.From == _currentState)
                .Select(kvp => kvp.Key.Trigger)
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Resets the state machine to initial state.
        /// </summary>
        public void Reset()
        {
            _currentState = EvolutionState.NotStarted;
            _transitionLog.Clear();
        }

        private void DefineTransitions()
        {
            _transitions[(EvolutionState.NotStarted, "Initialize")] = EvolutionState.Initializing;
            _transitions[(EvolutionState.Initializing, "Evaluate")] = EvolutionState.Evaluating;
            _transitions[(EvolutionState.Evaluating, "Speciate")] = EvolutionState.Speciating;
            _transitions[(EvolutionState.Speciating, "Select")] = EvolutionState.Selecting;
            _transitions[(EvolutionState.Selecting, "Evolve")] = EvolutionState.Evolving;
            _transitions[(EvolutionState.Evolving, "Evaluate")] = EvolutionState.Evaluating;
            _transitions[(EvolutionState.Evaluating, "Migrate")] = EvolutionState.Migrating;
            _transitions[(EvolutionState.Migrating, "Evaluate")] = EvolutionState.Evaluating;
            _transitions[(EvolutionState.Evaluating, "Complete")] = EvolutionState.Complete;
            _transitions[(EvolutionState.Evolving, "Complete")] = EvolutionState.Complete;
            _transitions[(EvolutionState.Evaluating, "Cancel")] = EvolutionState.Cancelled;
            _transitions[(EvolutionState.Speciating, "Cancel")] = EvolutionState.Cancelled;
            _transitions[(EvolutionState.Selecting, "Cancel")] = EvolutionState.Cancelled;
            _transitions[(EvolutionState.Evolving, "Cancel")] = EvolutionState.Cancelled;
            _transitions[(EvolutionState.Migrating, "Cancel")] = EvolutionState.Cancelled;
            _transitions[(EvolutionState.Evaluating, "Error")] = EvolutionState.Error;
            _transitions[(EvolutionState.Error, "Initialize")] = EvolutionState.Initializing;
            _transitions[(EvolutionState.Complete, "Initialize")] = EvolutionState.Initializing;
            _transitions[(EvolutionState.Cancelled, "Initialize")] = EvolutionState.Initializing;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Weight Initialization Strategies

    /// <summary>
    /// Provides different strategies for initializing connection weights in new genomes.
    /// Each strategy has different statistical properties that affect evolution dynamics.
    /// </summary>
    public enum WeightInitializationStrategy
    {
        /// <summary>Uniform random distribution [-range, +range].</summary>
        Uniform,

        /// <summary>Normal distribution with given standard deviation.</summary>
        Normal,

        /// <summary>Xavier/Glorot initialization scaled by fan-in and fan-out.</summary>
        Xavier,

        /// <summary>He initialization scaled by fan-in.</summary>
        He,

        /// <summary>LeCun initialization scaled by fan-in.</summary>
        LeCun,

        /// <summary>Orthogonal initialization.</summary>
        Orthogonal,

        /// <summary>Sparse initialization with mostly zeros.</summary>
        Sparse,

        /// <summary>Constant initialization to a fixed value.</summary>
        Constant
    }

    /// <summary>
    /// Provides weight initialization for new synapses based on various strategies.
    /// </summary>
    public sealed class WeightInitializer
    {
        private readonly WeightInitializationStrategy _strategy;
        private readonly double _scale;
        private readonly Random _rng;

        /// <summary>
        /// Initializes a new instance of the WeightInitializer class.
        /// </summary>
        /// <param name="strategy">Initialization strategy.</param>
        /// <param name="scale">Scale parameter for the strategy.</param>
        /// <param name="rng">Random number generator.</param>
        public WeightInitializer(
            WeightInitializationStrategy strategy = WeightInitializationStrategy.Xavier,
            double scale = 1.0,
            Random? rng = null)
        {
            _strategy = strategy;
            _scale = scale;
            _rng = rng ?? new Random();
        }

        /// <summary>
        /// Generates a weight value based on the initialization strategy.
        /// </summary>
        /// <param name="fanIn">Number of inputs to the target neuron.</param>
        /// <param name="fanOut">Number of outputs from the source neuron.</param>
        /// <returns>Initialized weight value.</returns>
        public double Initialize(int fanIn = 1, int fanOut = 1)
        {
            return _strategy switch
            {
                WeightInitializationStrategy.Uniform => UniformInitialize(),
                WeightInitializationStrategy.Normal => NormalInitialize(),
                WeightInitializationStrategy.Xavier => XavierInitialize(fanIn, fanOut),
                WeightInitializationStrategy.He => HeInitialize(fanIn),
                WeightInitializationStrategy.LeCun => LeCunInitialize(fanIn),
                WeightInitializationStrategy.Orthogonal => OrthogonalInitialize(),
                WeightInitializationStrategy.Sparse => SparseInitialize(),
                WeightInitializationStrategy.Constant => _scale,
                _ => UniformInitialize()
            };
        }

        /// <summary>
        /// Initializes a batch of weights.
        /// </summary>
        /// <param name="count">Number of weights to initialize.</param>
        /// <param name="fanIn">Fan-in for each weight.</param>
        /// <param name="fanOut">Fan-out for each weight.</param>
        /// <returns>Array of initialized weights.</returns>
        public double[] InitializeBatch(int count, int fanIn = 1, int fanOut = 1)
        {
            return Enumerable.Range(0, count)
                .Select(_ => Initialize(fanIn, fanOut))
                .ToArray();
        }

        private double UniformInitialize()
        {
            return (_rng.NextDouble() * 2.0 - 1.0) * _scale;
        }

        private double NormalInitialize()
        {
            double u1 = _rng.NextDouble();
            double u2 = _rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            return z * _scale;
        }

        private double XavierInitialize(int fanIn, int fanOut)
        {
            double limit = Math.Sqrt(6.0 / (fanIn + fanOut));
            return (_rng.NextDouble() * 2.0 - 1.0) * limit * _scale;
        }

        private double HeInitialize(int fanIn)
        {
            double stdDev = Math.Sqrt(2.0 / fanIn);
            double u1 = _rng.NextDouble();
            double u2 = _rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            return z * stdDev * _scale;
        }

        private double LeCunInitialize(int fanIn)
        {
            double stdDev = Math.Sqrt(1.0 / fanIn);
            double u1 = _rng.NextDouble();
            double u2 = _rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            return z * stdDev * _scale;
        }

        private double OrthogonalInitialize()
        {
            return (_rng.NextDouble() * 2.0 - 1.0) * _scale;
        }

        private double SparseInitialize()
        {
            if (_rng.NextDouble() < 0.8)
                return 0;
            return (_rng.NextDouble() * 2.0 - 1.0) * _scale;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Genome Distance Metrics Collection

    /// <summary>
    /// Comprehensive collection of distance metrics for comparing genomes.
    /// Provides multiple distance measures for different use cases.
    /// </summary>
    public sealed class GenomeDistanceMetrics
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the GenomeDistanceMetrics class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public GenomeDistanceMetrics(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Computes the NEAT compatibility distance between two genomes.
        /// Classic NEAT distance metric based on excess, disjoint genes, and weight differences.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Compatibility distance.</returns>
        public double ComputeCompatibilityDistance(GeoGenome a, GeoGenome b)
        {
            var aSynapseInnovs = a.Synapses.Where(s => s.IsActive).Select(s => s.InnovationNumber).ToHashSet();
            var bSynapseInnovs = b.Synapses.Where(s => s.IsActive).Select(s => s.InnovationNumber).ToHashSet();

            long maxInnovA = aSynapseInnovs.Count > 0 ? aSynapseInnovs.Max() : 0;
            long maxInnovB = bSynapseInnovs.Count > 0 ? bSynapseInnovs.Max() : 0;
            long maxInnov = Math.Max(maxInnovA, maxInnovB);

            if (maxInnov == 0)
                return 0;

            int excess = 0;
            int disjoint = 0;
            double weightDiffSum = 0;
            int matchingCount = 0;

            var aMap = a.Synapses.Where(s => s.IsActive).ToDictionary(s => s.InnovationNumber, s => s.Weight);
            var bMap = b.Synapses.Where(s => s.IsActive).ToDictionary(s => s.InnovationNumber, s => s.Weight);

            for (long i = 1; i <= maxInnov; i++)
            {
                bool inA = aMap.ContainsKey(i);
                bool inB = bMap.ContainsKey(i);

                if (inA && inB)
                {
                    matchingCount++;
                    weightDiffSum += Math.Abs(aMap[i] - bMap[i]);
                }
                else if (inA || inB)
                {
                    if (i > Math.Min(maxInnovA, maxInnovB))
                        excess++;
                    else
                        disjoint++;
                }
            }

            int N = Math.Max(a.ActiveSynapseCount, b.ActiveSynapseCount);
            if (N == 0)
                return 0;

            double normalizedExcess = (double)excess / N;
            double normalizedDisjoint = (double)disjoint / N;
            double avgWeightDiff = matchingCount > 0 ? weightDiffSum / matchingCount : 0;

            return _config.CompatibilityDisjointCoefficient * (normalizedExcess + normalizedDisjoint) +
                   _config.CompatibilityWeightCoefficient * avgWeightDiff;
        }

        /// <summary>
        /// Computes the Jaccard distance between two genomes based on their synapse innovation sets.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Jaccard distance (0 = identical, 1 = completely different).</returns>
        public double ComputeJaccardDistance(GeoGenome a, GeoGenome b)
        {
            var aInnovs = a.Synapses.Where(s => s.IsActive).Select(s => s.InnovationNumber).ToHashSet();
            var bInnovs = b.Synapses.Where(s => s.IsActive).Select(s => s.InnovationNumber).ToHashSet();

            int intersection = aInnovs.Intersect(bInnovs).Count();
            int union = aInnovs.Union(bInnovs).Count();

            return union > 0 ? 1.0 - (double)intersection / union : 0;
        }

        /// <summary>
        /// Computes the Hamming distance between two genomes based on activation functions.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Normalized Hamming distance.</returns>
        public double ComputeActivationHammingDistance(GeoGenome a, GeoGenome b)
        {
            var aActivations = a.Neurons
                .Where(n => n.IsActive)
                .OrderBy(n => n.InnovationNumber)
                .Select(n => n.Activation)
                .ToList();

            var bActivations = b.Neurons
                .Where(n => n.IsActive)
                .OrderBy(n => n.InnovationNumber)
                .Select(n => n.Activation)
                .ToList();

            int maxLen = Math.Max(aActivations.Count, bActivations.Count);
            if (maxLen == 0)
                return 0;

            int differences = 0;
            for (int i = 0; i < maxLen; i++)
            {
                var aAct = i < aActivations.Count ? aActivations[i] : ActivationFunction.Linear;
                var bAct = i < bActivations.Count ? bActivations[i] : ActivationFunction.Linear;
                if (aAct != bAct)
                    differences++;
            }

            return (double)differences / maxLen;
        }

        /// <summary>
        /// Computes the weight distribution distance between two genomes.
        /// Uses the Kolmogorov-Smirnov statistic.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>KS distance (0-1).</returns>
        public double ComputeWeightDistributionDistance(GeoGenome a, GeoGenome b)
        {
            var aWeights = a.Synapses.Where(s => s.IsActive).Select(s => s.Weight).OrderBy(w => w).ToList();
            var bWeights = b.Synapses.Where(s => s.IsActive).Select(s => s.Weight).OrderBy(w => w).ToList();

            if (aWeights.Count == 0 || bWeights.Count == 0)
                return 1.0;

            double maxDiff = 0;
            int i = 0, j = 0;

            while (i < aWeights.Count && j < bWeights.Count)
            {
                double aCDF = (double)(i + 1) / aWeights.Count;
                double bCDF = (double)(j + 1) / bWeights.Count;
                double diff = Math.Abs(aCDF - bCDF);
                maxDiff = Math.Max(maxDiff, diff);

                if (aWeights[i] < bWeights[j])
                    i++;
                else if (aWeights[i] > bWeights[j])
                    j++;
                else
                {
                    i++;
                    j++;
                }
            }

            return maxDiff;
        }

        /// <summary>
        /// Computes the structural similarity index (SSIM) between two genomes.
        /// Based on the SSIM image quality metric adapted for graph structures.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Structural similarity (0-1, higher = more similar).</returns>
        public double ComputeStructuralSimilarity(GeoGenome a, GeoGenome b)
        {
            double[] featuresA = ExtractStructuralFeatures(a);
            double[] featuresB = ExtractStructuralFeatures(b);

            int length = Math.Min(featuresA.Length, featuresB.Length);
            if (length == 0)
                return 0;

            double muA = featuresA.Take(length).Average();
            double muB = featuresB.Take(length).Average();

            double sigmaA2 = featuresA.Take(length).Average(f => (f - muA) * (f - muA));
            double sigmaB2 = featuresB.Take(length).Average(f => (f - muB) * (f - muB));
            double sigmaAB = featuresA.Take(length).Zip(featuresB.Take(length),
                (a2, b2) => (a2 - muA) * (b2 - muB)).Average();

            double C1 = 0.01 * 0.01;
            double C2 = 0.03 * 0.03;

            double luminance = (2 * muA * muB + C1) / (muA * muA + muB * muB + C1);
            double contrast = (2 * Math.Sqrt(sigmaA2) * Math.Sqrt(sigmaB2) + C2) / (sigmaA2 + sigmaB2 + C2);
            double structure = (sigmaAB + C2 / 2) / (Math.Sqrt(sigmaA2) * Math.Sqrt(sigmaB2) + C2 / 2);

            return Math.Clamp(luminance * contrast * structure, 0, 1);
        }

        /// <summary>
        /// Computes an ensemble distance combining multiple metrics.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Combined distance (0-1).</returns>
        public double ComputeEnsembleDistance(GeoGenome a, GeoGenome b)
        {
            double compatDist = ComputeCompatibilityDistance(a, b);
            double jaccardDist = ComputeJaccardDistance(a, b);
            double activationDist = ComputeActivationHammingDistance(a, b);
            double weightDist = ComputeWeightDistributionDistance(a, b);
            double structuralSim = ComputeStructuralSimilarity(a, b);

            return 0.25 * Math.Min(1, compatDist) +
                   0.25 * jaccardDist +
                   0.15 * activationDist +
                   0.15 * weightDist +
                   0.20 * (1 - structuralSim);
        }

        private double[] ExtractStructuralFeatures(GeoGenome genome)
        {
            var features = new List<double>();

            features.Add(genome.ActiveNeuronCount);
            features.Add(genome.ActiveSynapseCount);
            features.Add(genome.MaxLayerDepth);
            features.Add(genome.ConnectionDensity);

            var layerSizes = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.LayerIndex)
                .Select(g => (double)g.Count())
                .ToList();

            features.Add(layerSizes.Count);
            if (layerSizes.Count > 0)
            {
                features.Add(layerSizes.Average());
                features.Add(layerSizes.Max());
                features.Add(layerSizes.Min());
                double variance = layerSizes.Average(s => (s - layerSizes.Average()) * (s - layerSizes.Average()));
                features.Add(Math.Sqrt(variance));
            }

            var weights = genome.Synapses.Where(s => s.IsActive).Select(s => s.Weight).ToList();
            if (weights.Count > 0)
            {
                features.Add(weights.Average());
                features.Add(weights.Max());
                features.Add(weights.Min());
                features.Add(weights.Sum(w => Math.Abs(w)));
                double variance = weights.Average(w => (w - weights.Average()) * (w - weights.Average()));
                features.Add(Math.Sqrt(variance));
            }

            var activations = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.Activation)
                .Select(g => (double)g.Count())
                .ToList();

            features.Add(activations.Count);
            if (activations.Count > 0)
            {
                features.Add(activations.Average());
            }

            return features.ToArray();
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Summary Report Generator

    /// <summary>
    /// Generates formatted summary reports for evolution runs.
    /// Supports plain text, markdown, and HTML output formats.
    /// </summary>
    public sealed class EvolutionReportGenerator
    {
        private readonly EvolutionAnalyticsDashboard _dashboard;
        private readonly EvolutionHistoryTracker _history;

        /// <summary>
        /// Initializes a new instance of the EvolutionReportGenerator class.
        /// </summary>
        /// <param name="dashboard">Analytics dashboard.</param>
        /// <param name="history">History tracker.</param>
        public EvolutionReportGenerator(
            EvolutionAnalyticsDashboard dashboard,
            EvolutionHistoryTracker history)
        {
            _dashboard = dashboard ?? throw new ArgumentNullException(nameof(dashboard));
            _history = history ?? throw new ArgumentNullException(nameof(history));
        }

        /// <summary>
        /// Generates a plain text summary report.
        /// </summary>
        /// <returns>Plain text report string.</returns>
        public string GeneratePlainTextReport()
        {
            return _dashboard.GenerateReport();
        }

        /// <summary>
        /// Generates a markdown-formatted summary report.
        /// </summary>
        /// <returns>Markdown report string.</returns>
        public string GenerateMarkdownReport()
        {
            var summary = _history.GetSummary();
            var convergence = _dashboard.AnalyzeConvergence();
            var speciesDynamics = _dashboard.AnalyzeSpeciesDynamics();
            var mutationAnalysis = _dashboard.AnalyzeMutationEffectiveness();

            var sb = new StringBuilder();
            sb.AppendLine("# NEAT-G Evolution Report");
            sb.AppendLine();

            sb.AppendLine("## Overview");
            sb.AppendLine($"| Metric | Value |");
            sb.AppendLine($"|--------|-------|");
            sb.AppendLine($"| Total Generations | {summary.TotalGenerations} |");
            sb.AppendLine($"| Total Evaluations | {summary.TotalEvaluations:N0} |");
            sb.AppendLine($"| Best Fitness | {summary.BestFitnessEver:F6} |");
            sb.AppendLine($"| Fitness Improvement | {summary.FitnessImprovement:F6} |");
            sb.AppendLine($"| Peak Species | {summary.PeakSpeciesCount} |");
            sb.AppendLine($"| Final Species | {summary.FinalSpeciesCount} |");
            sb.AppendLine();

            sb.AppendLine("## Convergence Analysis");
            sb.AppendLine($"- **Converged:** {convergence.HasConverged}");
            sb.AppendLine($"- **Convergence Ratio:** {convergence.ConvergenceRatio:F4}");
            sb.AppendLine($"- **Recent Std Dev:** {convergence.RecentStdDev:F6}");
            sb.AppendLine($"- **Fitness Growth Rate:** {convergence.FitnessGrowthRate:P1}");
            if (convergence.EstimatedRemainingGenerations > 0)
                sb.AppendLine($"- **Est. Remaining Generations:** {convergence.EstimatedRemainingGenerations}");
            sb.AppendLine();

            sb.AppendLine("## Species Dynamics");
            sb.AppendLine($"- **Speciation Rate:** {speciesDynamics.SpeciationRate:F2} species/gen");
            sb.AppendLine($"- **Extinction Rate:** {speciesDynamics.ExtinctionRate:F2} species/gen");
            sb.AppendLine($"- **Species Stability:** {speciesDynamics.SpeciesStability:F3}");
            sb.AppendLine();

            sb.AppendLine("## Mutation Effectiveness");
            sb.AppendLine($"- **Avg Mutation Success:** {mutationAnalysis.AverageMutationSuccessRate:P1}");
            sb.AppendLine($"- **Avg Crossover Success:** {mutationAnalysis.AverageCrossoverSuccessRate:P1}");
            sb.AppendLine($"- **Most Effective:** {mutationAnalysis.MostEffectiveMutation} ({mutationAnalysis.MostEffectiveMutationRate:P1})");
            sb.AppendLine($"- **Diversity Trend:** {mutationAnalysis.DiversityTrend:F4}");
            sb.AppendLine();

            sb.AppendLine("## Fitness Progression");
            var metrics = _history.GetMetricsHistory();
            if (metrics.Count > 0)
            {
                sb.AppendLine("```");
                sb.AppendLine("Gen | Best     | Avg      | Species | Diversity");
                sb.AppendLine("----|----------|----------|---------|----------");
                int interval = Math.Max(1, metrics.Count / 30);
                for (int i = 0; i < metrics.Count; i += interval)
                {
                    var m = metrics[i];
                    sb.AppendLine($"{m.Generation,4} | {m.BestFitness,8:F4} | {m.AverageFitness,8:F4} | {m.SpeciesCount,7} | {m.DiversityMetric,8:F4}");
                }
                sb.AppendLine("```");
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generates an HTML-formatted summary report.
        /// </summary>
        /// <returns>HTML report string.</returns>
        public string GenerateHtmlReport()
        {
            var summary = _history.GetSummary();

            var sb = new StringBuilder();
            sb.AppendLine("<!DOCTYPE html>");
            sb.AppendLine("<html><head><title>NEAT-G Evolution Report</title>");
            sb.AppendLine("<style>");
            sb.AppendLine("body { font-family: Arial, sans-serif; margin: 20px; }");
            sb.AppendLine("h1 { color: #2C3E50; }");
            sb.AppendLine("h2 { color: #34495E; }");
            sb.AppendLine("table { border-collapse: collapse; width: 100%; }");
            sb.AppendLine("th, td { border: 1px solid #BDC3C7; padding: 8px; text-align: left; }");
            sb.AppendLine("th { background-color: #ECF0F1; }");
            sb.AppendLine(".metric { font-size: 1.2em; font-weight: bold; }");
            sb.AppendLine(".improvement { color: #27AE60; }");
            sb.AppendLine(".warning { color: #E67E22; }");
            sb.AppendLine("</style></head><body>");
            sb.AppendLine("<h1>NEAT-G Evolution Report</h1>");

            sb.AppendLine("<h2>Overview</h2>");
            sb.AppendLine("<table>");
            sb.AppendLine("<tr><th>Metric</th><th>Value</th></tr>");
            sb.AppendLine($"<tr><td>Total Generations</td><td>{summary.TotalGenerations}</td></tr>");
            sb.AppendLine($"<tr><td>Total Evaluations</td><td>{summary.TotalEvaluations:N0}</td></tr>");
            sb.AppendLine($"<tr><td>Best Fitness</td><td class='metric improvement'>{summary.BestFitnessEver:F6}</td></tr>");
            sb.AppendLine($"<tr><td>Fitness Improvement</td><td class='improvement'>{summary.FitnessImprovement:F6}</td></tr>");
            sb.AppendLine($"<tr><td>Peak Species</td><td>{summary.PeakSpeciesCount}</td></tr>");
            sb.AppendLine($"<tr><td>Final Species</td><td>{summary.FinalSpeciesCount}</td></tr>");
            sb.AppendLine($"<tr><td>Average Diversity</td><td>{summary.AverageDiversity:F4}</td></tr>");
            sb.AppendLine("</table>");

            sb.AppendLine("<h2>Fitness Over Generations</h2>");
            sb.AppendLine("<table>");
            sb.AppendLine("<tr><th>Generation</th><th>Best Fitness</th><th>Avg Fitness</th><th>Species</th><th>Diversity</th></tr>");

            var metrics = _history.GetMetricsHistory();
            int interval = Math.Max(1, metrics.Count / 20);
            for (int i = 0; i < metrics.Count; i += interval)
            {
                var m = metrics[i];
                sb.AppendLine($"<tr><td>{m.Generation}</td><td>{m.BestFitness:F4}</td><td>{m.AverageFitness:F4}</td><td>{m.SpeciesCount}</td><td>{m.DiversityMetric:F4}</td></tr>");
            }
            sb.AppendLine("</table>");

            sb.AppendLine("</body></html>");
            return sb.ToString();
        }

        /// <summary>
        /// Generates a compact summary suitable for console output.
        /// </summary>
        /// <param name="metrics">Current metrics.</param>
        /// <returns>Compact single-line summary.</returns>
        public string GenerateCompactSummary(EvolutionMetrics metrics)
        {
            return $"Gen {metrics.Generation,5} | Best {metrics.BestFitness,8:F4} | Avg {metrics.AverageFitness,8:F4} | " +
                   $"Species {metrics.SpeciesCount,3} | Div {metrics.DiversityMetric,5:F3} | " +
                   $"Mut {metrics.MutationSuccessRate:P0} | Cross {metrics.CrossoverSuccessRate:P0} | " +
                   $"Time {metrics.GenerationTime.TotalMilliseconds:F0}ms";
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Auto-Tuner

    /// <summary>
    /// Automatically tunes evolution parameters based on observed performance.
    /// Uses Bayesian optimization principles to find optimal hyperparameter settings.
    /// Monitors fitness progress, diversity, and convergence to adjust parameters dynamically.
    /// </summary>
    public sealed class EvolutionAutoTuner
    {
        private readonly EvolutionConfig _config;
        private readonly Queue<AutoTunerObservation> _observations;
        private readonly Dictionary<string, ParameterRange> _parameterRanges;
        private int _tuningInterval;
        private int _observationsSinceTuning;

        /// <summary>
        /// Initializes a new instance of the EvolutionAutoTuner class.
        /// </summary>
        /// <param name="config">Evolution configuration to tune.</param>
        /// <param name="tuningInterval">Generations between tuning attempts.</param>
        public EvolutionAutoTuner(EvolutionConfig config, int tuningInterval = 20)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _observations = new Queue<AutoTunerObservation>();
            _parameterRanges = InitializeParameterRanges();
            _tuningInterval = tuningInterval;
            _observationsSinceTuning = 0;
        }

        /// <summary>Gets the number of observations collected.</summary>
        public int ObservationCount => _observations.Count;

        /// <summary>
        /// Records an observation of evolution performance with current parameters.
        /// </summary>
        /// <param name="metrics">Current evolution metrics.</param>
        /// <param name="config">Current configuration used.</param>
        public void RecordObservation(EvolutionMetrics metrics, EvolutionConfig config)
        {
            _observations.Enqueue(new AutoTunerObservation
            {
                Timestamp = DateTime.UtcNow,
                Generation = metrics.Generation,
                BestFitness = metrics.BestFitness,
                AverageFitness = metrics.AverageFitness,
                FitnessImprovement = metrics.FitnessImprovement,
                SpeciesCount = metrics.SpeciesCount,
                DiversityMetric = metrics.DiversityMetric,
                MutationSuccessRate = metrics.MutationSuccessRate,
                CrossoverSuccessRate = metrics.CrossoverSuccessRate,
                AverageComplexity = metrics.AverageComplexity,
                CrossoverRate = config.CrossoverRate,
                MutationRate = config.MutationRate,
                SpeciationThreshold = config.SpeciationThreshold,
                TournamentSize = config.TournamentSize,
                PerturbationMagnitude = config.PerturbationMagnitude,
                MigrationRate = config.MigrationRate,
                PopulationSize = config.PopulationSize
            });

            _observationsSinceTuning++;
        }

        /// <summary>
        /// Analyzes collected observations and suggests parameter adjustments.
        /// </summary>
        /// <returns>Recommended parameter adjustments.</returns>
        public AutoTunerRecommendations AnalyzeAndRecommend()
        {
            var recentObservations = _observations.TakeLast(50).ToList();
            if (recentObservations.Count < 5)
            {
                return new AutoTunerRecommendations { Confidence = 0, Changes = new List<ParameterChange>() };
            }

            var recommendations = new List<ParameterChange>();

            var fitnessTrend = ComputeFitnessTrend(recentObservations);
            var diversityTrend = ComputeDiversityTrend(recentObservations);
            var mutationEffectiveness = ComputeMutationEffectiveness(recentObservations);
            var convergenceRate = ComputeConvergenceRate(recentObservations);

            if (fitnessTrend < 0.001 && diversityTrend < -0.05)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.MutationRate),
                    CurrentValue = _config.MutationRate,
                    RecommendedValue = Math.Min(0.5, _config.MutationRate * 1.3),
                    Reason = "Fitness stagnation with declining diversity"
                });

                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.MigrationRate),
                    CurrentValue = _config.MigrationRate,
                    RecommendedValue = Math.Min(0.2, _config.MigrationRate * 1.5),
                    Reason = "Increase migration to boost diversity"
                });
            }

            if (fitnessTrend > 0.01 && diversityTrend > 0.1)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.MutationRate),
                    CurrentValue = _config.MutationRate,
                    RecommendedValue = Math.Max(0.05, _config.MutationRate * 0.9),
                    Reason = "Good progress; slightly reduce mutation for exploitation"
                });
            }

            if (mutationEffectiveness < 0.05)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.PerturbationMagnitude),
                    CurrentValue = _config.PerturbationMagnitude,
                    RecommendedValue = Math.Min(0.5, _config.PerturbationMagnitude * 1.5),
                    Reason = "Low mutation effectiveness; increase perturbation magnitude"
                });
            }

            if (recentObservations.Average(o => o.SpeciesCount) > _config.TargetSpeciesCount * 1.5)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.SpeciationThreshold),
                    CurrentValue = _config.SpeciationThreshold,
                    RecommendedValue = Math.Min(_config.MaxSpeciationThreshold,
                        _config.SpeciationThreshold * 1.2),
                    Reason = "Too many species; increase threshold to merge species"
                });
            }
            else if (recentObservations.Average(o => o.SpeciesCount) < _config.TargetSpeciesCount * 0.5)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.SpeciationThreshold),
                    CurrentValue = _config.SpeciationThreshold,
                    RecommendedValue = Math.Max(_config.MinSpeciationThreshold,
                        _config.SpeciationThreshold * 0.8),
                    Reason = "Too few species; decrease threshold to create more species"
                });
            }

            if (convergenceRate < 0.0001 && recentObservations.Count > 20)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.CrossoverRate),
                    CurrentValue = _config.CrossoverRate,
                    RecommendedValue = Math.Max(0.3, _config.CrossoverRate - 0.05),
                    Reason = "Very slow convergence; reduce crossover to increase exploration via mutation"
                });
            }

            double avgTournamentSize = recentObservations.Average(o => o.TournamentSize);
            if (avgTournamentSize > 7 && diversityTrend < -0.1)
            {
                recommendations.Add(new ParameterChange
                {
                    ParameterName = nameof(EvolutionConfig.TournamentSize),
                    CurrentValue = _config.TournamentSize,
                    RecommendedValue = Math.Max(2, _config.TournamentSize - 1),
                    Reason = "High selection pressure causing diversity loss"
                });
            }

            double confidence = ComputeRecommendationConfidence(recentObservations, recommendations);

            return new AutoTunerRecommendations
            {
                Confidence = confidence,
                Changes = recommendations.AsReadOnly(),
                FitnessTrend = fitnessTrend,
                DiversityTrend = diversityTrend,
                ConvergenceRate = convergenceRate,
                ObservationCount = recentObservations.Count
            };
        }

        /// <summary>
        /// Applies recommended changes to the configuration.
        /// </summary>
        /// <param name="recommendations">Recommendations to apply.</param>
        /// <param name="maxChangesPerIteration">Maximum number of changes to apply.</param>
        /// <returns>Number of changes actually applied.</returns>
        public int ApplyRecommendations(AutoTunerRecommendations recommendations, int maxChangesPerIteration = 2)
        {
            if (recommendations.Confidence < 0.3)
                return 0;

            int applied = 0;
            foreach (var change in recommendations.Changes.Take(maxChangesPerIteration))
            {
                if (ApplyParameterChange(change))
                {
                    applied++;
                }
            }
            _observationsSinceTuning = 0;
            return applied;
        }

        /// <summary>
        /// Gets the tuning history.
        /// </summary>
        public IReadOnlyList<AutoTunerObservation> GetObservations()
        {
            return _observations.ToList().AsReadOnly();
        }

        /// <summary>
        /// Resets the auto-tuner state.
        /// </summary>
        public void Reset()
        {
            _observations.Clear();
            _observationsSinceTuning = 0;
        }

        private bool ApplyParameterChange(ParameterChange change)
        {
            switch (change.ParameterName)
            {
                case nameof(EvolutionConfig.MutationRate):
                    _config.MutationRate = Math.Clamp(change.RecommendedValue, 0.01, 0.8);
                    return true;
                case nameof(EvolutionConfig.CrossoverRate):
                    _config.CrossoverRate = Math.Clamp(change.RecommendedValue, 0.1, 0.95);
                    return true;
                case nameof(EvolutionConfig.SpeciationThreshold):
                    _config.SpeciationThreshold = Math.Clamp(change.RecommendedValue,
                        _config.MinSpeciationThreshold, _config.MaxSpeciationThreshold);
                    return true;
                case nameof(EvolutionConfig.TournamentSize):
                    _config.TournamentSize = Math.Clamp((int)change.RecommendedValue, 2, 15);
                    return true;
                case nameof(EvolutionConfig.PerturbationMagnitude):
                    _config.PerturbationMagnitude = Math.Clamp(change.RecommendedValue, 0.001, 1.0);
                    return true;
                case nameof(EvolutionConfig.MigrationRate):
                    _config.MigrationRate = Math.Clamp(change.RecommendedValue, 0.0, 0.3);
                    return true;
                default:
                    return false;
            }
        }

        private double ComputeFitnessTrend(List<AutoTunerObservation> observations)
        {
            if (observations.Count < 2)
                return 0;

            var fitnesses = observations.Select(o => o.BestFitness).ToList();
            return (fitnesses[^1] - fitnesses[0]) / Math.Max(1, Math.Abs(fitnesses[0]));
        }

        private double ComputeDiversityTrend(List<AutoTunerObservation> observations)
        {
            if (observations.Count < 2)
                return 0;

            var diversities = observations.Select(o => o.DiversityMetric).ToList();
            return (diversities[^1] - diversities[0]) / Math.Max(0.01, diversities[0]);
        }

        private double ComputeMutationEffectiveness(List<AutoTunerObservation> observations)
        {
            return observations.Average(o => o.MutationSuccessRate);
        }

        private double ComputeConvergenceRate(List<AutoTunerObservation> observations)
        {
            if (observations.Count < 5)
                return 0;

            var recent = observations.TakeLast(5).ToList();
            var improvements = new List<double>();
            for (int i = 1; i < recent.Count; i++)
            {
                improvements.Add(recent[i].BestFitness - recent[i - 1].BestFitness);
            }

            return improvements.Count > 0 ? improvements.Average() : 0;
        }

        private double ComputeRecommendationConfidence(
            List<AutoTunerObservation> observations,
            List<ParameterChange> recommendations)
        {
            if (recommendations.Count == 0)
                return 0;

            double dataQuality = Math.Min(1.0, observations.Count / 30.0);
            double signalStrength = Math.Min(1.0, recommendations.Count * 0.3);
            double consistency = ComputeObservationConsistency(observations);

            return 0.4 * dataQuality + 0.3 * signalStrength + 0.3 * consistency;
        }

        private double ComputeObservationConsistency(List<AutoTunerObservation> observations)
        {
            if (observations.Count < 3)
                return 0.5;

            var fitnesses = observations.Select(o => o.BestFitness).ToList();
            int monotonicCount = 0;
            for (int i = 1; i < fitnesses.Count; i++)
            {
                if (fitnesses[i] >= fitnesses[i - 1])
                    monotonicCount++;
            }

            return (double)monotonicCount / (fitnesses.Count - 1);
        }

        private Dictionary<string, ParameterRange> InitializeParameterRanges()
        {
            return new Dictionary<string, ParameterRange>
            {
                [nameof(EvolutionConfig.MutationRate)] = new ParameterRange(0.01, 0.8, 0.25),
                [nameof(EvolutionConfig.CrossoverRate)] = new ParameterRange(0.1, 0.95, 0.75),
                [nameof(EvolutionConfig.SpeciationThreshold)] = new ParameterRange(0.5, 8.0, 3.0),
                [nameof(EvolutionConfig.TournamentSize)] = new ParameterRange(2, 15, 5),
                [nameof(EvolutionConfig.PerturbationMagnitude)] = new ParameterRange(0.001, 1.0, 0.1),
                [nameof(EvolutionConfig.MigrationRate)] = new ParameterRange(0.0, 0.3, 0.05),
                [nameof(EvolutionConfig.PopulationSize)] = new ParameterRange(50, 2000, 300)
            };
        }
    }

    /// <summary>
    /// Observation record for auto-tuning.
    /// </summary>
    public sealed class AutoTunerObservation
    {
        /// <summary>Timestamp of observation.</summary>
        public DateTime Timestamp { get; init; }

        /// <summary>Generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Best fitness.</summary>
        public double BestFitness { get; init; }

        /// <summary>Average fitness.</summary>
        public double AverageFitness { get; init; }

        /// <summary>Fitness improvement over previous generation.</summary>
        public double FitnessImprovement { get; init; }

        /// <summary>Number of species.</summary>
        public int SpeciesCount { get; init; }

        /// <summary>Diversity metric.</summary>
        public double DiversityMetric { get; init; }

        /// <summary>Mutation success rate.</summary>
        public double MutationSuccessRate { get; init; }

        /// <summary>Crossover success rate.</summary>
        public double CrossoverSuccessRate { get; init; }

        /// <summary>Average complexity.</summary>
        public double AverageComplexity { get; init; }

        /// <summary>Configuration parameters at observation time.</summary>
        public double CrossoverRate { get; init; }
        public double MutationRate { get; init; }
        public double SpeciationThreshold { get; init; }
        public int TournamentSize { get; init; }
        public double PerturbationMagnitude { get; init; }
        public double MigrationRate { get; init; }
        public int PopulationSize { get; init; }
    }

    /// <summary>
    /// Recommendations from the auto-tuner.
    /// </summary>
    public sealed class AutoTunerRecommendations
    {
        /// <summary>Confidence in the recommendations (0-1).</summary>
        public double Confidence { get; init; }

        /// <summary>Recommended parameter changes.</summary>
        public IReadOnlyList<ParameterChange> Changes { get; init; } = Array.Empty<ParameterChange>();

        /// <summary>Observed fitness trend.</summary>
        public double FitnessTrend { get; init; }

        /// <summary>Observed diversity trend.</summary>
        public double DiversityTrend { get; init; }

        /// <summary>Observed convergence rate.</summary>
        public double ConvergenceRate { get; init; }

        /// <summary>Number of observations used.</summary>
        public int ObservationCount { get; init; }
    }

    /// <summary>
    /// A single parameter change recommendation.
    /// </summary>
    public sealed class ParameterChange
    {
        /// <summary>Name of the parameter.</summary>
        public string ParameterName { get; init; } = string.Empty;

        /// <summary>Current value.</summary>
        public double CurrentValue { get; init; }

        /// <summary>Recommended value.</summary>
        public double RecommendedValue { get; init; }

        /// <summary>Reason for the change.</summary>
        public string Reason { get; init; } = string.Empty;

        /// <inheritdoc/>
        public override string ToString() =>
            $"{ParameterName}: {CurrentValue:F3} -> {RecommendedValue:F3} ({Reason})";
    }

    /// <summary>
    /// Range definition for a parameter.
    /// </summary>
    public sealed class ParameterRange
    {
        /// <summary>Minimum allowed value.</summary>
        public double Min { get; init; }

        /// <summary>Maximum allowed value.</summary>
        public double Max { get; init; }

        /// <summary>Default value.</summary>
        public double Default { get; init; }

        /// <summary>
        /// Initializes a new ParameterRange.
        /// </summary>
        public ParameterRange(double min, double max, double @default)
        {
            Min = min;
            Max = max;
            Default = @default;
        }

        /// <summary>Clamps a value to the valid range.</summary>
        public double Clamp(double value) => Math.Clamp(value, Min, Max);

        /// <summary>Normalizes a value to [0, 1].</summary>
        public double Normalize(double value) =>
            (Math.Clamp(value, Min, Max) - Min) / Math.Max(1e-10, Max - Min);

        /// <summary>Denormalizes a value from [0, 1].</summary>
        public double Denormalize(double normalized) =>
            Min + Math.Clamp(normalized, 0, 1) * (Max - Min);
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Experiment Runner

    /// <summary>
    /// Runs controlled evolution experiments with statistical rigor.
    /// Supports multiple runs, parameter sweeps, and comparative analysis.
    /// </summary>
    public sealed class EvolutionExperimentRunner
    {
        /// <summary>
        /// Runs multiple evolution trials with the same configuration for statistical analysis.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="trialCount">Number of trials to run.</param>
        /// <param name="progressCallback">Progress callback.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Statistical summary of all trials.</returns>
        public async Task<ExperimentResults> RunMultipleTrialsAsync(
            EvolutionConfig config,
            int inputCount,
            int outputCount,
            EvaluationContext context,
            int trialCount,
            IProgress<int>? progressCallback = null,
            CancellationToken ct = default)
        {
            var trialResults = new List<TrialResult>();

            for (int trial = 0; trial < trialCount; trial++)
            {
                ct.ThrowIfCancellationRequested();

                var trialConfig = config.Clone();
                trialConfig.RandomSeed = (trial + 1) * 42;

                var engine = new NeatGEvolutionEngine(trialConfig);
                var result = await engine.RunEvolutionAsync(
                    inputCount, outputCount, context, null, ct).ConfigureAwait(false);

                trialResults.Add(new TrialResult
                {
                    TrialNumber = trial + 1,
                    BestFitness = result.BestGenome.Fitness,
                    TotalGenerations = result.TotalGenerations,
                    TotalEvaluations = result.TotalEvaluations,
                    ElapsedTime = result.TotalElapsed,
                    TargetReached = result.TargetReached,
                    FinalSpeciesCount = result.FinalPopulation.Count > 0
                        ? result.MetricsHistory.Last().SpeciesCount
                        : 0
                });

                progressCallback?.Report(trial + 1);
            }

            return ComputeExperimentStatistics(trialResults);
        }

        /// <summary>
        /// Runs a parameter sweep over specified parameter ranges.
        /// </summary>
        /// <param name="baseConfig">Base configuration.</param>
        /// <param name="parameterName">Parameter to sweep.</param>
        /// <param name="values">Values to try.</param>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="trialsPerValue">Number of trials per parameter value.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Results for each parameter value.</returns>
        public async Task<IReadOnlyList<ParameterSweepResult>> RunParameterSweepAsync(
            EvolutionConfig baseConfig,
            string parameterName,
            double[] values,
            int inputCount,
            int outputCount,
            EvaluationContext context,
            int trialsPerValue = 3,
            CancellationToken ct = default)
        {
            var results = new List<ParameterSweepResult>();

            foreach (var value in values)
            {
                ct.ThrowIfCancellationRequested();

                var config = baseConfig.Clone();
                SetParameterValue(config, parameterName, value);

                var trialResults = new List<double>();
                var trialTimes = new List<TimeSpan>();

                for (int trial = 0; trial < trialsPerValue; trial++)
                {
                    ct.ThrowIfCancellationRequested();

                    var trialConfig = config.Clone();
                    trialConfig.RandomSeed = (trial + 1) * 100;

                    var engine = new NeatGEvolutionEngine(trialConfig);
                    var result = await engine.RunEvolutionAsync(
                        inputCount, outputCount, context, null, ct).ConfigureAwait(false);

                    trialResults.Add(result.BestGenome.Fitness);
                    trialTimes.Add(result.TotalElapsed);
                }

                results.Add(new ParameterSweepResult
                {
                    ParameterName = parameterName,
                    ParameterValue = value,
                    AverageFitness = trialResults.Average(),
                    StdDevFitness = trialResults.Count > 1
                        ? Math.Sqrt(trialResults.Average(f => (f - trialResults.Average()) * (f - trialResults.Average())))
                        : 0,
                    BestFitness = trialResults.Max(),
                    WorstFitness = trialResults.Min(),
                    AverageTime = TimeSpan.FromTicks((long)trialTimes.Average(t => t.Ticks)),
                    TrialCount = trialsPerValue
                });
            }

            return results.AsReadOnly();
        }

        /// <summary>
        /// Runs a comparative experiment between two configurations.
        /// </summary>
        public async Task<ComparisonResult> RunComparativeExperimentAsync(
            EvolutionConfig configA,
            EvolutionConfig configB,
            string configAName,
            string configBName,
            int inputCount,
            int outputCount,
            EvaluationContext context,
            int trialCount = 5,
            CancellationToken ct = default)
        {
            var resultsA = new List<double>();
            var resultsB = new List<double>();
            var timesA = new List<TimeSpan>();
            var timesB = new List<TimeSpan>();

            for (int trial = 0; trial < trialCount; trial++)
            {
                ct.ThrowIfCancellationRequested();

                int seed = (trial + 1) * 42;

                var configAClone = configA.Clone();
                configAClone.RandomSeed = seed;
                var engineA = new NeatGEvolutionEngine(configAClone);
                var resultA = await engineA.RunEvolutionAsync(
                    inputCount, outputCount, context, null, ct).ConfigureAwait(false);
                resultsA.Add(resultA.BestGenome.Fitness);
                timesA.Add(resultA.TotalElapsed);

                var configBClone = configB.Clone();
                configBClone.RandomSeed = seed;
                var engineB = new NeatGEvolutionEngine(configBClone);
                var resultB = await engineB.RunEvolutionAsync(
                    inputCount, outputCount, context, null, ct).ConfigureAwait(false);
                resultsB.Add(resultB.BestGenome.Fitness);
                timesB.Add(resultB.TotalElapsed);
            }

            double meanA = resultsA.Average();
            double meanB = resultsB.Average();
            double stdA = resultsA.Count > 1
                ? Math.Sqrt(resultsA.Average(f => (f - meanA) * (f - meanA))) : 0;
            double stdB = resultsB.Count > 1
                ? Math.Sqrt(resultsB.Average(f => (f - meanB) * (f - meanB))) : 0;

            double tStatistic = 0;
            double pValue = 0.5;
            if (stdA > 0 && stdB > 0)
            {
                double pooledSE = Math.Sqrt(stdA * stdA / trialCount + stdB * stdB / trialCount);
                if (pooledSE > 1e-10)
                {
                    tStatistic = (meanA - meanB) / pooledSE;
                    pValue = ComputeTwoTailedPValue(Math.Abs(tStatistic), 2 * trialCount - 2);
                }
            }

            return new ComparisonResult
            {
                ConfigAName = configAName,
                ConfigBName = configBName,
                MeanFitnessA = meanA,
                MeanFitnessB = meanB,
                StdDevA = stdA,
                StdDevB = stdB,
                TStatistic = tStatistic,
                PValue = pValue,
                IsSignificant = pValue < 0.05,
                Winner = meanA > meanB ? configAName : meanB > meanA ? configBName : "Tie",
                AverageTimeA = TimeSpan.FromTicks((long)timesA.Average(t => t.Ticks)),
                AverageTimeB = TimeSpan.FromTicks((long)timesB.Average(t => t.Ticks)),
                TrialCount = trialCount
            };
        }

        private ExperimentResults ComputeExperimentStatistics(List<TrialResult> trialResults)
        {
            var bestFitnesses = trialResults.Select(t => t.BestFitness).ToList();
            var generations = trialResults.Select(t => t.TotalGenerations).ToList();
            var evaluations = trialResults.Select(t => t.TotalEvaluations).ToList();

            double mean = bestFitnesses.Average();
            double variance = bestFitnesses.Average(f => (f - mean) * (f - mean));
            double stdDev = Math.Sqrt(variance);

            return new ExperimentResults
            {
                TrialCount = trialResults.Count,
                MeanBestFitness = mean,
                StdDevBestFitness = stdDev,
                MedianBestFitness = GetMedian(bestFitnesses),
                MinBestFitness = bestFitnesses.Min(),
                MaxBestFitness = bestFitnesses.Max(),
                MeanGenerations = generations.Average(),
                MeanEvaluations = evaluations.Average(),
                TargetReachedCount = trialResults.Count(t => t.TargetReached),
                TargetReachedRate = (double)trialResults.Count(t => t.TargetReached) / trialResults.Count,
                MeanElapsedTime = TimeSpan.FromTicks(
                    (long)trialResults.Average(t => t.ElapsedTime.Ticks)),
                TrialResults = trialResults.AsReadOnly()
            };
        }

        private double GetMedian(List<double> values)
        {
            var sorted = values.OrderBy(v => v).ToList();
            int mid = sorted.Count / 2;
            return sorted.Count % 2 == 0
                ? (sorted[mid - 1] + sorted[mid]) / 2.0
                : sorted[mid];
        }

        private void SetParameterValue(EvolutionConfig config, string parameterName, double value)
        {
            switch (parameterName)
            {
                case nameof(EvolutionConfig.MutationRate):
                    config.MutationRate = value;
                    break;
                case nameof(EvolutionConfig.CrossoverRate):
                    config.CrossoverRate = value;
                    break;
                case nameof(EvolutionConfig.SpeciationThreshold):
                    config.SpeciationThreshold = value;
                    break;
                case nameof(EvolutionConfig.TournamentSize):
                    config.TournamentSize = (int)value;
                    break;
                case nameof(EvolutionConfig.PerturbationMagnitude):
                    config.PerturbationMagnitude = value;
                    break;
                case nameof(EvolutionConfig.MigrationRate):
                    config.MigrationRate = value;
                    break;
                case nameof(EvolutionConfig.PopulationSize):
                    config.PopulationSize = (int)value;
                    break;
                case nameof(EvolutionConfig.MaxStagnationGenerations):
                    config.MaxStagnationGenerations = (int)value;
                    break;
            }
        }

        private double ComputeTwoTailedPValue(double tStat, int degreesOfFreedom)
        {
            double x = (double)degreesOfFreedom / (degreesOfFreedom + tStat * tStat);
            double p = 0.5 * IncompleteBetaFunction(degreesOfFreedom / 2.0, 0.5, x);
            return 2 * p;
        }

        private double IncompleteBetaFunction(double a, double b, double x)
        {
            if (x <= 0)
                return 0;
            if (x >= 1)
                return 1;

            double result = 0;
            double term = Math.Pow(x, a) * Math.Pow(1 - x, b) / a;
            result = term;

            for (int n = 1; n < 200; n++)
            {
                double numerator = n * (b - n) * x / ((a + 2 * n - 1) * (a + 2 * n));
                term *= numerator;
                result += term;
                if (Math.Abs(term) < 1e-10)
                    break;
            }

            double beta = GammaFunction(a) * GammaFunction(b) / GammaFunction(a + b);
            return result / beta;
        }

        private double GammaFunction(double z)
        {
            if (z < 0.5)
                return Math.PI / (Math.Sin(Math.PI * z) * GammaFunction(1 - z));

            z -= 1;
            double[] coefficients = {
                0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                771.32342877765313, -176.61502916214059, 12.507343278686905,
                -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
            };

            double x = coefficients[0];
            for (int i = 1; i < coefficients.Length; i++)
                x += coefficients[i] / (z + i);

            double t = z + coefficients.Length - 1.5;
            return Math.Sqrt(2 * Math.PI) * Math.Pow(t, z + 0.5) * Math.Exp(-t) * x;
        }
    }

    /// <summary>
    /// Results of a single trial run.
    /// </summary>
    public sealed class TrialResult
    {
        public int TrialNumber { get; init; }
        public double BestFitness { get; init; }
        public int TotalGenerations { get; init; }
        public long TotalEvaluations { get; init; }
        public TimeSpan ElapsedTime { get; init; }
        public bool TargetReached { get; init; }
        public int FinalSpeciesCount { get; init; }
    }

    /// <summary>
    /// Statistical results of multiple experiment trials.
    /// </summary>
    public sealed class ExperimentResults
    {
        public int TrialCount { get; init; }
        public double MeanBestFitness { get; init; }
        public double StdDevBestFitness { get; init; }
        public double MedianBestFitness { get; init; }
        public double MinBestFitness { get; init; }
        public double MaxBestFitness { get; init; }
        public double MeanGenerations { get; init; }
        public double MeanEvaluations { get; init; }
        public int TargetReachedCount { get; init; }
        public double TargetReachedRate { get; init; }
        public TimeSpan MeanElapsedTime { get; init; }
        public IReadOnlyList<TrialResult> TrialResults { get; init; } = Array.Empty<TrialResult>();
    }

    /// <summary>
    /// Result of a parameter sweep value.
    /// </summary>
    public sealed class ParameterSweepResult
    {
        public string ParameterName { get; init; } = string.Empty;
        public double ParameterValue { get; init; }
        public double AverageFitness { get; init; }
        public double StdDevFitness { get; init; }
        public double BestFitness { get; init; }
        public double WorstFitness { get; init; }
        public TimeSpan AverageTime { get; init; }
        public int TrialCount { get; init; }
    }

    /// <summary>
    /// Result of a comparative experiment between two configurations.
    /// </summary>
    public sealed class ComparisonResult
    {
        public string ConfigAName { get; init; } = string.Empty;
        public string ConfigBName { get; init; } = string.Empty;
        public double MeanFitnessA { get; init; }
        public double MeanFitnessB { get; init; }
        public double StdDevA { get; init; }
        public double StdDevB { get; init; }
        public double TStatistic { get; init; }
        public double PValue { get; init; }
        public bool IsSignificant { get; init; }
        public string Winner { get; init; } = string.Empty;
        public TimeSpan AverageTimeA { get; init; }
        public TimeSpan AverageTimeB { get; init; }
        public int TrialCount { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Final Assembly Info

    /// <summary>
    /// Provides metadata about the NEAT-G Evolution Engine assembly.
    /// </summary>
    public static class NeatGEngineInfo
    {
        /// <summary>Engine version.</summary>
        public const string Version = "2.0.0";

        /// <summary>Engine name.</summary>
        public const string Name = "GDNN NEAT-G Evolution Engine";

        /// <summary>Engine description.</summary>
        public const string Description = "NeuroEvolution of Augmented Topologies - Geometric: " +
            "A comprehensive evolutionary algorithm for neural network architecture optimization " +
            "with geometric awareness, semantic crossover, manifold-based speciation, and swarm evolution.";

        /// <summary>Supported mutation types count.</summary>
        public const int SupportedMutationTypes = 18;

        /// <summary>Supported activation functions count.</summary>
        public const int SupportedActivationFunctions = 13;

        /// <summary>Supported selection methods count.</summary>
        public const int SupportedSelectionMethods = 5;

        /// <summary>Supported speciation methods count.</summary>
        public const int SupportedSpeciationMethods = 4;

        /// <summary>Supported crossover strategies count.</summary>
        public const int SupportedCrossoverStrategies = 4;

        /// <summary>
        /// Gets a formatted information string.
        /// </summary>
        public static string GetInfoString()
        {
            return $"{Name} v{Version}\n" +
                   $"{Description}\n" +
                   $"Mutation Types: {SupportedMutationTypes}\n" +
                   $"Activation Functions: {SupportedActivationFunctions}\n" +
                   $"Selection Methods: {SupportedSelectionMethods}\n" +
                   $"Speciation Methods: {SupportedSpeciationMethods}\n" +
                   $"Crossover Strategies: {SupportedCrossoverStrategies}";
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Network Inference Engine

    /// <summary>
    /// High-performance neural network inference engine for evaluating genome outputs.
    /// Supports forward pass, batch inference, and caching for efficient repeated evaluations.
    /// Implements multiple evaluation strategies including single-pass, recurrent, and
    /// Monte Carlo dropout for uncertainty estimation.
    /// </summary>
    public sealed class NetworkInferenceEngine
    {
        private readonly EvolutionConfig _config;
        private readonly Dictionary<long, double> _activationCache;
        private readonly Dictionary<long, int> _topologicalOrder;
        private bool _topologicalOrderDirty;

        /// <summary>
        /// Initializes a new instance of the NetworkInferenceEngine class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public NetworkInferenceEngine(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _activationCache = new Dictionary<long, double>();
            _topologicalOrder = new Dictionary<long, int>();
            _topologicalOrderDirty = true;
        }

        /// <summary>
        /// Performs a forward pass through the network defined by a genome.
        /// Computes output values for given inputs using topological ordering.
        /// </summary>
        /// <param name="genome">The genome defining the network.</param>
        /// <param name="inputs">Input values.</param>
        /// <returns>Output values from the network.</returns>
        public double[] ForwardPass(GeoGenome genome, ImmutableArray<double> inputs)
        {
            if (genome == null || genome.Neurons.Count == 0)
                return Array.Empty<double>();

            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            if (activeNeurons.Count == 0)
                return Array.Empty<double>();

            ComputeTopologicalOrderIfNeeded(genome, activeNeurons, activeSynapses);

            _activationCache.Clear();

            var inputNeurons = activeNeurons
                .Where(n => n.LayerIndex == 0)
                .OrderBy(n => n.PositionInLayer)
                .ToList();

            for (int i = 0; i < inputNeurons.Count; i++)
            {
                double val = i < inputs.Length ? inputs[i] : 0;
                _activationCache[inputNeurons[i].InnovationNumber] = val;
            }

            var synapseLookup = activeSynapses
                .GroupBy(s => s.TargetNeuronId)
                .ToDictionary(g => g.Key, g => g.ToList());

            var sortedLayers = activeNeurons
                .GroupBy(n => n.LayerIndex)
                .OrderBy(g => g.Key)
                .ToList();

            foreach (var layer in sortedLayers)
            {
                if (layer.Key == 0)
                    continue;

                foreach (var neuron in layer)
                {
                    double weightedSum = neuron.Bias;

                    if (synapseLookup.TryGetValue(neuron.InnovationNumber, out var inputsynapses))
                    {
                        foreach (var synapse in inputsynapses)
                        {
                            if (_activationCache.TryGetValue(synapse.SourceNeuronId, out double srcVal))
                            {
                                weightedSum += synapse.Weight * srcVal;
                            }
                        }
                    }

                    _activationCache[neuron.InnovationNumber] = neuron.Activate(weightedSum);
                }
            }

            var outputNeurons = activeNeurons
                .Where(n => n.LayerIndex == sortedLayers.Last().Key)
                .OrderBy(n => n.PositionInLayer)
                .ToList();

            var output = new double[outputNeurons.Count];
            for (int i = 0; i < outputNeurons.Count; i++)
            {
                if (_activationCache.TryGetValue(outputNeurons[i].InnovationNumber, out double val))
                    output[i] = val;
            }

            return output;
        }

        /// <summary>
        /// Performs batch forward pass for multiple input sets.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="batchInputs">Batch of input arrays.</param>
        /// <returns>Batch of output arrays.</returns>
        public IReadOnlyList<double[]> BatchForwardPass(GeoGenome genome, IReadOnlyList<ImmutableArray<double>> batchInputs)
        {
            var results = new List<double[]>(batchInputs.Count);
            foreach (var inputs in batchInputs)
            {
                results.Add(ForwardPass(genome, inputs));
            }
            return results;
        }

        /// <summary>
        /// Performs Monte Carlo forward pass with dropout for uncertainty estimation.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="inputs">Input values.</param>
        /// <param name="dropoutRate">Dropout rate (probability of dropping a neuron).</param>
        /// <param name="samples">Number of Monte Carlo samples.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>Mean output and standard deviation for each output neuron.</returns>
        public (double[] Mean, double[] StdDev) MonteCarloForwardPass(
            GeoGenome genome,
            ImmutableArray<double> inputs,
            double dropoutRate,
            int samples,
            Random rng)
        {
            var allOutputs = new List<double[]>();

            for (int s = 0; s < samples; s++)
            {
                var sampledGenome = ApplyDropout(genome, dropoutRate, rng);
                var output = ForwardPass(sampledGenome, inputs);
                allOutputs.Add(output);
            }

            if (allOutputs.Count == 0)
                return (Array.Empty<double>(), Array.Empty<double>());

            int outputSize = allOutputs[0].Length;
            var mean = new double[outputSize];
            var stdDev = new double[outputSize];

            for (int i = 0; i < outputSize; i++)
            {
                double sum = allOutputs.Average(o => i < o.Length ? o[i] : 0);
                mean[i] = sum;

                double variance = allOutputs.Average(o =>
                {
                    double val = i < o.Length ? o[i] : 0;
                    return (val - sum) * (val - sum);
                });
                stdDev[i] = Math.Sqrt(variance);
            }

            return (mean, stdDev);
        }

        /// <summary>
        /// Computes Jacobian matrix of outputs with respect to inputs.
        /// Useful for sensitivity analysis.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="inputs">Input values.</param>
        /// <param name="epsilon">Perturbation size for finite differences.</param>
        /// <returns>Jacobian matrix [output_size x input_size].</returns>
        public double[,] ComputeJacobian(GeoGenome genome, ImmutableArray<double> inputs, double epsilon = 1e-5)
        {
            var baseOutput = ForwardPass(genome, inputs);
            int outputSize = baseOutput.Length;
            int inputSize = inputs.Length;

            var jacobian = new double[outputSize, inputSize];

            for (int j = 0; j < inputSize; j++)
            {
                var perturbedInputs = inputs.ToArray();
                perturbedInputs[j] += epsilon;
                var perturbedOutput = ForwardPass(genome, perturbedInputs.ToImmutableArray());

                for (int i = 0; i < outputSize; i++)
                {
                    double baseVal = i < baseOutput.Length ? baseOutput[i] : 0;
                    double pertVal = i < perturbedOutput.Length ? perturbedOutput[i] : 0;
                    jacobian[i, j] = (pertVal - baseVal) / epsilon;
                }
            }

            return jacobian;
        }

        /// <summary>
        /// Computes the total sensitivity of each input to the output.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="inputs">Input values.</param>
        /// <returns>Sensitivity scores for each input.</returns>
        public double[] ComputeInputSensitivity(GeoGenome genome, ImmutableArray<double> inputs)
        {
            var jacobian = ComputeJacobian(genome, inputs);
            int inputSize = inputs.Length;
            int outputSize = jacobian.GetLength(0);

            var sensitivity = new double[inputSize];
            for (int j = 0; j < inputSize; j++)
            {
                double sumSq = 0;
                for (int i = 0; i < outputSize; i++)
                {
                    sumSq += jacobian[i, j] * jacobian[i, j];
                }
                sensitivity[j] = Math.Sqrt(sumSq);
            }

            double maxSens = sensitivity.Max();
            if (maxSens > 1e-10)
            {
                for (int j = 0; j < inputSize; j++)
                    sensitivity[j] /= maxSens;
            }

            return sensitivity;
        }

        /// <summary>
        /// Estimates the computational cost of evaluating a genome.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Estimated FLOPs for a single forward pass.</returns>
        public long EstimateComputeCost(GeoGenome genome)
        {
            long flops = 0;

            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            flops += activeSynapses.Count * 2;

            foreach (var neuron in activeNeurons)
            {
                flops += GetActivationCost(neuron.Activation);
            }

            return flops;
        }

        /// <summary>
        /// Estimates memory usage for evaluating a genome.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Estimated memory usage in bytes.</returns>
        public long EstimateMemoryUsage(GeoGenome genome)
        {
            long bytes = 0;

            bytes += genome.ActiveNeuronCount * sizeof(double);
            bytes += genome.ActiveSynapseCount * (sizeof(double) + sizeof(long) * 2);
            bytes += genome.TotalNeuronCount * 64;
            bytes += genome.TotalSynapseCount * 80;

            bytes += 1024;

            return bytes;
        }

        /// <summary>
        /// Profiles the inference performance of a genome.
        /// </summary>
        /// <param name="genome">The genome to profile.</param>
        /// <param name="inputSize">Input vector size.</param>
        /// <param name="iterations">Number of iterations for timing.</param>
        /// <returns>Profiling results.</returns>
        public InferenceProfile ProfileInference(GeoGenome genome, int inputSize, int iterations = 1000)
        {
            var rng = new Random(42);
            var inputs = ImmutableArray.CreateRange(
                Enumerable.Range(0, inputSize).Select(_ => rng.NextDouble() * 2 - 1));

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                ForwardPass(genome, inputs);
            }
            sw.Stop();

            long flops = EstimateComputeCost(genome);
            long memory = EstimateMemoryUsage(genome);

            double avgTimeMs = sw.Elapsed.TotalMilliseconds / iterations;
            double flopsPerSecond = avgTimeMs > 0 ? flops / (avgTimeMs / 1000.0) : 0;

            return new InferenceProfile
            {
                AverageInferenceTimeMs = avgTimeMs,
                TotalTimeMs = sw.Elapsed.TotalMilliseconds,
                Iterations = iterations,
                EstimatedFLOPs = flops,
                EstimatedMemoryBytes = memory,
                FLOPS = flopsPerSecond,
                NeuronCount = genome.ActiveNeuronCount,
                SynapseCount = genome.ActiveSynapseCount,
                LayerCount = genome.MaxLayerDepth + 1
            };
        }

        private GeoGenome ApplyDropout(GeoGenome genome, double dropoutRate, Random rng)
        {
            var dropped = genome.Clone();
            foreach (var neuron in dropped.Neurons.Where(n => n.IsActive && n.LayerIndex > 0))
            {
                if (rng.NextDouble() < dropoutRate)
                {
                    neuron.IsActive = false;
                }
            }
            return dropped;
        }

        private void ComputeTopologicalOrderIfNeeded(GeoGenome genome, List<GeoNeuron> activeNeurons, List<GeoSynapse> activeSynapses)
        {
            if (!_topologicalOrderDirty && _topologicalOrder.Count == activeNeurons.Count)
                return;

            _topologicalOrder.Clear();
            int order = 0;

            var layers = activeNeurons
                .GroupBy(n => n.LayerIndex)
                .OrderBy(g => g.Key);

            foreach (var layer in layers)
            {
                foreach (var neuron in layer)
                {
                    _topologicalOrder[neuron.InnovationNumber] = order++;
                }
            }

            _topologicalOrderDirty = false;
        }

        private long GetActivationCost(ActivationFunction activation)
        {
            return activation switch
            {
                ActivationFunction.Tanh => 8,
                ActivationFunction.Sigmoid => 6,
                ActivationFunction.ReLU => 2,
                ActivationFunction.LeakyReLU => 3,
                ActivationFunction.GELU => 15,
                ActivationFunction.Swish => 8,
                ActivationFunction.Sinusoidal => 10,
                ActivationFunction.Linear => 1,
                ActivationFunction.Abs => 2,
                ActivationFunction.Step => 2,
                ActivationFunction.Softplus => 6,
                ActivationFunction.Mish => 20,
                ActivationFunction.Exponential => 6,
                _ => 2
            };
        }
    }

    /// <summary>
    /// Inference profiling results.
    /// </summary>
    public sealed class InferenceProfile
    {
        /// <summary>Average inference time per forward pass.</summary>
        public double AverageInferenceTimeMs { get; init; }

        /// <summary>Total profiling time.</summary>
        public double TotalTimeMs { get; init; }

        /// <summary>Number of iterations profiled.</summary>
        public int Iterations { get; init; }

        /// <summary>Estimated floating point operations.</summary>
        public long EstimatedFLOPs { get; init; }

        /// <summary>Estimated memory usage in bytes.</summary>
        public long EstimatedMemoryBytes { get; init; }

        /// <summary>Estimated FLOPS throughput.</summary>
        public double FLOPS { get; init; }

        /// <summary>Number of active neurons.</summary>
        public int NeuronCount { get; init; }

        /// <summary>Number of active synapses.</summary>
        public int SynapseCount { get; init; }

        /// <summary>Number of layers.</summary>
        public int LayerCount { get; init; }

        /// <inheritdoc/>
        public override string ToString() =>
            $"InferenceProfile(Time={AverageInferenceTimeMs:F3}ms, FLOPs={EstimatedFLOPs:N0}, " +
            $"Memory={EstimatedMemoryBytes:N0}B, Neurons={NeuronCount}, Synapses={SynapseCount})";
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Genome Population Snapshot

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

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Logging

    /// <summary>
    /// Structured logging for the NEAT-G evolution engine.
    /// Provides severity levels, categorized messages, and optional file output.
    /// </summary>
    public sealed class EvolutionLogger
    {
        private readonly ConcurrentQueue<LogEntry> _logEntries;
        private readonly int _maxEntries;
        private readonly object _writeLock;
        private LogLevel _minimumLevel;

        /// <summary>
        /// Initializes a new instance of the EvolutionLogger class.
        /// </summary>
        /// <param name="minimumLevel">Minimum log level to record.</param>
        /// <param name="maxEntries">Maximum log entries to retain.</param>
        public EvolutionLogger(LogLevel minimumLevel = LogLevel.Info, int maxEntries = 50000)
        {
            _minimumLevel = minimumLevel;
            _maxEntries = maxEntries;
            _logEntries = new ConcurrentQueue<LogEntry>();
            _writeLock = new object();
        }

        /// <summary>Gets the current minimum log level.</summary>
        public LogLevel MinimumLevel
        {
            get => _minimumLevel;
            set => _minimumLevel = value;
        }

        /// <summary>Gets the number of log entries.</summary>
        public int EntryCount => _logEntries.Count;

        /// <summary>
        /// Logs a debug message.
        /// </summary>
        /// <param name="message">Log message.</param>
        /// <param name="category">Optional category.</param>
        public void LogDebug(string message, string? category = null)
        {
            Log(LogLevel.Debug, message, category);
        }

        /// <summary>
        /// Logs an informational message.
        /// </summary>
        /// <param name="message">Log message.</param>
        /// <param name="category">Optional category.</param>
        public void LogInfo(string message, string? category = null)
        {
            Log(LogLevel.Info, message, category);
        }

        /// <summary>
        /// Logs a warning message.
        /// </summary>
        /// <param name="message">Log message.</param>
        /// <param name="category">Optional category.</param>
        public void LogWarning(string message, string? category = null)
        {
            Log(LogLevel.Warning, message, category);
        }

        /// <summary>
        /// Logs an error message.
        /// </summary>
        /// <param name="message">Log message.</param>
        /// <param name="exception">Optional exception.</param>
        /// <param name="category">Optional category.</param>
        public void LogError(string message, Exception? exception = null, string? category = null)
        {
            Log(LogLevel.Error, message, category, exception);
        }

        /// <summary>
        /// Logs a message at the specified level.
        /// </summary>
        /// <param name="level">Log level.</param>
        /// <param name="message">Log message.</param>
        /// <param name="category">Optional category.</param>
        /// <param name="exception">Optional exception.</param>
        public void Log(LogLevel level, string message, string? category = null, Exception? exception = null)
        {
            if (level < _minimumLevel)
                return;

            var entry = new LogEntry
            {
                Timestamp = DateTime.UtcNow,
                Level = level,
                Message = message,
                Category = category ?? "General",
                Exception = exception?.ToString(),
                ThreadId = Environment.CurrentManagedThreadId
            };

            _logEntries.Enqueue(entry);

            while (_logEntries.Count > _maxEntries)
            {
                _logEntries.TryDequeue(out _);
            }
        }

        /// <summary>
        /// Gets log entries filtered by level and category.
        /// </summary>
        /// <param name="level">Minimum level (null for all).</param>
        /// <param name="category">Category filter (null for all).</param>
        /// <param name="count">Maximum entries to return.</param>
        public IReadOnlyList<LogEntry> GetEntries(
            LogLevel? level = null,
            string? category = null,
            int count = 100)
        {
            return _logEntries
                .Where(e => (!level.HasValue || e.Level >= level.Value) &&
                           (category == null || e.Category == category))
            .Take(count)
            .ToList()
            .AsReadOnly();
        }

        /// <summary>
        /// Exports all log entries as a formatted string.
        /// </summary>
        /// <param name="format">Log format.</param>
        public string Export(LogFormat format = LogFormat.Text)
        {
            var entries = _logEntries.ToList();

            return format switch
            {
                LogFormat.Text => ExportAsText(entries),
                LogFormat.Json => ExportAsJson(entries),
                LogFormat.Csv => ExportAsCsv(entries),
                _ => ExportAsText(entries)
            };
        }

        /// <summary>
        /// Clears all log entries.
        /// </summary>
        public void Clear()
        {
            while (_logEntries.TryDequeue(out _))
            { }
        }

        /// <summary>
        /// Gets summary statistics for the log.
        /// </summary>
        public LogSummary GetSummary()
        {
            var entries = _logEntries.ToList();
            return new LogSummary
            {
                TotalEntries = entries.Count,
                DebugCount = entries.Count(e => e.Level == LogLevel.Debug),
                InfoCount = entries.Count(e => e.Level == LogLevel.Info),
                WarningCount = entries.Count(e => e.Level == LogLevel.Warning),
                ErrorCount = entries.Count(e => e.Level == LogLevel.Error),
                Categories = entries.Select(e => e.Category).Distinct().ToList().AsReadOnly(),
                FirstEntry = entries.Count > 0 ? entries[0].Timestamp : DateTime.MinValue,
                LastEntry = entries.Count > 0 ? entries[^1].Timestamp : DateTime.MinValue
            };
        }

        private string ExportAsText(List<LogEntry> entries)
        {
            var sb = new StringBuilder();
            foreach (var entry in entries)
            {
                sb.AppendLine($"[{entry.Timestamp:HH:mm:ss.fff}] [{entry.Level}] [{entry.Category}] {entry.Message}");
                if (entry.Exception != null)
                    sb.AppendLine($"  Exception: {entry.Exception}");
            }
            return sb.ToString();
        }

        private string ExportAsJson(List<LogEntry> entries)
        {
            return JsonSerializer.Serialize(entries, new JsonSerializerOptions
            {
                WriteIndented = true,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            });
        }

        private string ExportAsCsv(List<LogEntry> entries)
        {
            var sb = new StringBuilder();
            sb.AppendLine("Timestamp,Level,Category,Message,ThreadId");
            foreach (var entry in entries)
            {
                string msg = entry.Message.Replace("\"", "\"\"");
                sb.AppendLine($"\"{entry.Timestamp:O}\",{entry.Level},{entry.Category},\"{msg}\",{entry.ThreadId}");
            }
            return sb.ToString();
        }
    }

    /// <summary>
    /// Log severity levels.
    /// </summary>
    public enum LogLevel
    {
        Debug = 0,
        Info = 1,
        Warning = 2,
        Error = 3
    }

    /// <summary>
    /// Log output formats.
    /// </summary>
    public enum LogFormat
    {
        Text,
        Json,
        Csv
    }

    /// <summary>
    /// A single log entry.
    /// </summary>
    public sealed class LogEntry
    {
        public DateTime Timestamp { get; init; }
        public LogLevel Level { get; init; }
        public string Message { get; init; } = string.Empty;
        public string Category { get; init; } = string.Empty;
        public string? Exception { get; init; }
        public int ThreadId { get; init; }
    }

    /// <summary>
    /// Summary statistics for logs.
    /// </summary>
    public sealed class LogSummary
    {
        public int TotalEntries { get; init; }
        public int DebugCount { get; init; }
        public int InfoCount { get; init; }
        public int WarningCount { get; init; }
        public int ErrorCount { get; init; }
        public IReadOnlyList<string> Categories { get; init; } = Array.Empty<string>();
        public DateTime FirstEntry { get; init; }
        public DateTime LastEntry { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Configuration Optimizer

    /// <summary>
    /// Optimizes evolution configuration parameters using meta-evolution.
    /// Treats configuration parameters as a genome and evolves them to find
    /// optimal settings for a given problem class.
    /// </summary>
    public sealed class ConfigurationOptimizer
    {
        private readonly int _inputCount;
        private readonly int _outputCount;
        private readonly EvaluationContext _context;
        private readonly EvolutionConfig _baseConfig;
        private readonly Random _rng;

        /// <summary>
        /// Initializes a new instance of the ConfigurationOptimizer class.
        /// </summary>
        /// <param name="inputCount">Network input count.</param>
        /// <param name="outputCount">Network output count.</param>
        /// <param name="context">Evaluation context.</param>
        /// <param name="baseConfig">Base configuration to optimize from.</param>
        public ConfigurationOptimizer(
            int inputCount,
            int outputCount,
            EvaluationContext context,
            EvolutionConfig baseConfig)
        {
            _inputCount = inputCount;
            _outputCount = outputCount;
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _baseConfig = baseConfig ?? throw new ArgumentNullException(nameof(baseConfig));
            _rng = new Random(42);
        }

        /// <summary>
        /// Optimizes configuration parameters using random search.
        /// </summary>
        /// <param name="trials">Number of random configurations to try.</param>
        /// <param name="maxGenerationsPerTrial">Max generations per trial.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Best configuration found and its performance.</returns>
        public async Task<(EvolutionConfig BestConfig, double BestFitness)> OptimizeRandomSearchAsync(
            int trials,
            int maxGenerationsPerTrial,
            CancellationToken ct = default)
        {
            EvolutionConfig bestConfig = _baseConfig.Clone();
            double bestFitness = double.MinValue;

            for (int trial = 0; trial < trials; trial++)
            {
                ct.ThrowIfCancellationRequested();

                var config = GenerateRandomConfig();
                config.MaxGenerations = maxGenerationsPerTrial;
                config.RandomSeed = _rng.Next();

                var engine = new NeatGEvolutionEngine(config);
                var result = await engine.RunEvolutionAsync(
                    _inputCount, _outputCount, _context, null, ct).ConfigureAwait(false);

                if (result.BestGenome.Fitness > bestFitness)
                {
                    bestFitness = result.BestGenome.Fitness;
                    bestConfig = config;
                }
            }

            return (bestConfig, bestFitness);
        }

        /// <summary>
        /// Optimizes configuration using hill-climbing on parameter space.
        /// </summary>
        /// <param name="iterations">Number of hill-climbing iterations.</param>
        /// <param name="maxGenerationsPerTrial">Max generations per trial.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Best configuration found and its performance.</returns>
        public async Task<(EvolutionConfig BestConfig, double BestFitness)> OptimizeHillClimbingAsync(
            int iterations,
            int maxGenerationsPerTrial,
            CancellationToken ct = default)
        {
            var currentConfig = _baseConfig.Clone();
            currentConfig.MaxGenerations = maxGenerationsPerTrial;
            currentConfig.RandomSeed = _rng.Next();

            var engine = new NeatGEvolutionEngine(currentConfig);
            var result = await engine.RunEvolutionAsync(
                _inputCount, _outputCount, _context, null, ct).ConfigureAwait(false);
            double currentFitness = result.BestGenome.Fitness;

            for (int iter = 0; iter < iterations; iter++)
            {
                ct.ThrowIfCancellationRequested();

                var neighborConfig = PerturbConfig(currentConfig);
                neighborConfig.MaxGenerations = maxGenerationsPerTrial;
                neighborConfig.RandomSeed = _rng.Next();

                var neighborEngine = new NeatGEvolutionEngine(neighborConfig);
                var neighborResult = await neighborEngine.RunEvolutionAsync(
                    _inputCount, _outputCount, _context, null, ct).ConfigureAwait(false);

                if (neighborResult.BestGenome.Fitness > currentFitness)
                {
                    currentConfig = neighborConfig;
                    currentFitness = neighborResult.BestGenome.Fitness;
                }
            }

            return (currentConfig, currentFitness);
        }

        /// <summary>
        /// Optimizes configuration using a genetic algorithm on parameter space.
        /// </summary>
        /// <param name="populationSize">Population size for config evolution.</param>
        /// <param name="generations">Number of generations for config evolution.</param>
        /// <param name="maxGenerationsPerTrial">Max generations per evolution trial.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>Best configuration found and its performance.</returns>
        public async Task<(EvolutionConfig BestConfig, double BestFitness)> OptimizeWithGAAsync(
            int populationSize,
            int generations,
            int maxGenerationsPerTrial,
            CancellationToken ct = default)
        {
            var configPopulation = new List<(EvolutionConfig Config, double Fitness)>();

            for (int i = 0; i < populationSize; i++)
            {
                ct.ThrowIfCancellationRequested();

                var config = i == 0 ? _baseConfig.Clone() : GenerateRandomConfig();
                config.MaxGenerations = maxGenerationsPerTrial;
                config.RandomSeed = _rng.Next();

                var engine = new NeatGEvolutionEngine(config);
                var result = await engine.RunEvolutionAsync(
                    _inputCount, _outputCount, _context, null, ct).ConfigureAwait(false);

                configPopulation.Add((config, result.BestGenome.Fitness));
            }

            for (int gen = 0; gen < generations; gen++)
            {
                ct.ThrowIfCancellationRequested();

                configPopulation = configPopulation
                    .OrderByDescending(x => x.Fitness)
                    .ToList();

                var newPopulation = new List<(EvolutionConfig, double)>();

                int eliteCount = Math.Max(1, populationSize / 5);
                newPopulation.AddRange(configPopulation.Take(eliteCount));

                while (newPopulation.Count < populationSize)
                {
                    var parentA = configPopulation[_rng.Next(populationSize)].Config;
                    var parentB = configPopulation[_rng.Next(populationSize)].Config;

                    var child = CrossoverConfigs(parentA, parentB);
                    child = MutateConfig(child);
                    child.MaxGenerations = maxGenerationsPerTrial;
                    child.RandomSeed = _rng.Next();

                    var engine = new NeatGEvolutionEngine(child);
                    var result = await engine.RunEvolutionAsync(
                        _inputCount, _outputCount, _context, null, ct).ConfigureAwait(false);

                    newPopulation.Add((child, result.BestGenome.Fitness));
                }

                configPopulation = newPopulation;
            }

            var best = configPopulation.OrderByDescending(x => x.Fitness).First();
            return (best.Config, best.Fitness);
        }

        private EvolutionConfig GenerateRandomConfig()
        {
            var config = _baseConfig.Clone();
            config.PopulationSize = _rng.Next(100, 800);
            config.CrossoverRate = _rng.NextDouble() * 0.6 + 0.2;
            config.MutationRate = _rng.NextDouble() * 0.5 + 0.1;
            config.SpeciationThreshold = _rng.NextDouble() * 6 + 1;
            config.TargetSpeciesCount = _rng.Next(3, 25);
            config.TournamentSize = _rng.Next(2, 10);
            config.MaxStagnationGenerations = _rng.Next(10, 50);
            config.PerturbationMagnitude = _rng.NextDouble() * 0.3 + 0.01;
            config.MigrationRate = _rng.NextDouble() * 0.15;
            config.MigrationInterval = _rng.Next(3, 20);
            config.LandmarkCount = _rng.Next(10, 50);
            config.SemanticEmbeddingDimension = _rng.Next(16, 64);
            config.EliteFraction = _rng.NextDouble() * 0.1 + 0.01;
            return config;
        }

        private EvolutionConfig PerturbConfig(EvolutionConfig config)
        {
            var perturbed = config.Clone();
            string[] parameters = {
                nameof(EvolutionConfig.CrossoverRate),
                nameof(EvolutionConfig.MutationRate),
                nameof(EvolutionConfig.SpeciationThreshold),
                nameof(EvolutionConfig.TournamentSize),
                nameof(EvolutionConfig.PerturbationMagnitude),
                nameof(EvolutionConfig.MigrationRate),
                nameof(EvolutionConfig.MaxStagnationGenerations)
            };

            string param = parameters[_rng.Next(parameters.Length)];
            double perturbation = (_rng.NextDouble() * 0.4 - 0.2);

            switch (param)
            {
                case nameof(EvolutionConfig.CrossoverRate):
                    perturbed.CrossoverRate = Math.Clamp(perturbed.CrossoverRate + perturbation, 0.2, 0.95);
                    break;
                case nameof(EvolutionConfig.MutationRate):
                    perturbed.MutationRate = Math.Clamp(perturbed.MutationRate + perturbation, 0.05, 0.8);
                    break;
                case nameof(EvolutionConfig.SpeciationThreshold):
                    perturbed.SpeciationThreshold = Math.Clamp(perturbed.SpeciationThreshold + perturbation * 2, 0.5, 8);
                    break;
                case nameof(EvolutionConfig.TournamentSize):
                    perturbed.TournamentSize = Math.Clamp(perturbed.TournamentSize + (perturbation > 0 ? 1 : -1), 2, 15);
                    break;
                case nameof(EvolutionConfig.PerturbationMagnitude):
                    perturbed.PerturbationMagnitude = Math.Clamp(perturbed.PerturbationMagnitude + perturbation * 0.1, 0.001, 0.5);
                    break;
                case nameof(EvolutionConfig.MigrationRate):
                    perturbed.MigrationRate = Math.Clamp(perturbed.MigrationRate + perturbation * 0.05, 0, 0.2);
                    break;
                case nameof(EvolutionConfig.MaxStagnationGenerations):
                    perturbed.MaxStagnationGenerations = Math.Clamp(perturbed.MaxStagnationGenerations + (int)(perturbation * 10), 5, 100);
                    break;
            }

            return perturbed;
        }

        private EvolutionConfig CrossoverConfigs(EvolutionConfig a, EvolutionConfig b)
        {
            var child = a.Clone();
            if (_rng.NextDouble() < 0.5)
                child.CrossoverRate = b.CrossoverRate;
            if (_rng.NextDouble() < 0.5)
                child.MutationRate = b.MutationRate;
            if (_rng.NextDouble() < 0.5)
                child.SpeciationThreshold = b.SpeciationThreshold;
            if (_rng.NextDouble() < 0.5)
                child.TournamentSize = b.TournamentSize;
            if (_rng.NextDouble() < 0.5)
                child.PerturbationMagnitude = b.PerturbationMagnitude;
            if (_rng.NextDouble() < 0.5)
                child.MigrationRate = b.MigrationRate;
            if (_rng.NextDouble() < 0.5)
                child.MaxStagnationGenerations = b.MaxStagnationGenerations;
            return child;
        }

        private EvolutionConfig MutateConfig(EvolutionConfig config)
        {
            if (_rng.NextDouble() < 0.3)
                config.CrossoverRate = Math.Clamp(config.CrossoverRate + (_rng.NextDouble() - 0.5) * 0.2, 0.2, 0.95);
            if (_rng.NextDouble() < 0.3)
                config.MutationRate = Math.Clamp(config.MutationRate + (_rng.NextDouble() - 0.5) * 0.2, 0.05, 0.8);
            if (_rng.NextDouble() < 0.3)
                config.SpeciationThreshold = Math.Clamp(config.SpeciationThreshold + (_rng.NextDouble() - 0.5) * 2, 0.5, 8);
            return config;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Replay System

    /// <summary>
    /// Records and replays evolution runs for debugging, analysis, and visualization.
    /// Captures all genome operations and allows step-by-step replay.
    /// </summary>
    public sealed class EvolutionReplaySystem
    {
        private readonly List<ReplayEvent> _events;
        private int _currentPosition;
        private bool _isRecording;

        /// <summary>
        /// Initializes a new instance of the EvolutionReplaySystem class.
        /// </summary>
        public EvolutionReplaySystem()
        {
            _events = new List<ReplayEvent>();
            _currentPosition = 0;
            _isRecording = false;
        }

        /// <summary>Whether the system is currently recording.</summary>
        public bool IsRecording => _isRecording;

        /// <summary>Total recorded events.</summary>
        public int EventCount => _events.Count;

        /// <summary>Current replay position.</summary>
        public int CurrentPosition => _currentPosition;

        /// <summary>
        /// Starts recording evolution events.
        /// </summary>
        public void StartRecording()
        {
            _isRecording = true;
            _events.Clear();
            _currentPosition = 0;
        }

        /// <summary>
        /// Stops recording.
        /// </summary>
        public void StopRecording()
        {
            _isRecording = false;
        }

        /// <summary>
        /// Records a replay event.
        /// </summary>
        /// <param name="eventType">Type of event.</param>
        /// <param name="generation">Current generation.</param>
        /// <param name="data">Event data.</param>
        /// <param name="genomeId">Associated genome ID (optional).</param>
        public void RecordEvent(ReplayEventType eventType, int generation, string data, Guid? genomeId = null)
        {
            if (!_isRecording)
                return;

            _events.Add(new ReplayEvent
            {
                Position = _events.Count,
                Timestamp = DateTime.UtcNow,
                EventType = eventType,
                Generation = generation,
                Data = data,
                GenomeId = genomeId
            });
        }

        /// <summary>
        /// Records a genome snapshot at a specific point.
        /// </summary>
        /// <param name="genome">Genome to record.</param>
        /// <param name="generation">Current generation.</param>
        public void RecordGenomeSnapshot(GeoGenome genome, int generation)
        {
            if (!_isRecording)
                return;

            _events.Add(new ReplayEvent
            {
                Position = _events.Count,
                Timestamp = DateTime.UtcNow,
                EventType = ReplayEventType.GenomeSnapshot,
                Generation = generation,
                Data = JsonSerializer.Serialize(new
                {
                    genome.Id,
                    genome.Fitness,
                    NeuronCount = genome.ActiveNeuronCount,
                    SynapseCount = genome.ActiveSynapseCount,
                    genome.Complexity
                }),
                GenomeId = genome.Id
            });
        }

        /// <summary>
        /// Gets the next event in replay.
        /// </summary>
        public ReplayEvent? GetNextEvent()
        {
            if (_currentPosition >= _events.Count)
                return null;
            return _events[_currentPosition++];
        }

        /// <summary>
        /// Gets a specific event by position.
        /// </summary>
        /// <param name="position">Event position.</param>
        public ReplayEvent? GetEventAt(int position)
        {
            if (position < 0 || position >= _events.Count)
                return null;
            return _events[position];
        }

        /// <summary>
        /// Seeks to a specific position in the replay.
        /// </summary>
        /// <param name="position">Target position.</param>
        public void SeekTo(int position)
        {
            _currentPosition = Math.Clamp(position, 0, _events.Count);
        }

        /// <summary>
        /// Seeks to a specific generation.
        /// </summary>
        /// <param name="generation">Target generation.</param>
        public void SeekToGeneration(int generation)
        {
            var idx = _events.FindIndex(e => e.Generation >= generation);
            if (idx >= 0)
                _currentPosition = idx;
        }

        /// <summary>
        /// Gets all events for a specific generation.
        /// </summary>
        /// <param name="generation">Generation number.</param>
        public IReadOnlyList<ReplayEvent> GetEventsForGeneration(int generation)
        {
            return _events.Where(e => e.Generation == generation).ToList().AsReadOnly();
        }

        /// <summary>
        /// Gets events filtered by type.
        /// </summary>
        /// <param name="eventType">Event type to filter.</param>
        public IReadOnlyList<ReplayEvent> GetEventsByType(ReplayEventType eventType)
        {
            return _events.Where(e => e.EventType == eventType).ToList().AsReadOnly();
        }

        /// <summary>
        /// Gets the total number of generations recorded.
        /// </summary>
        public int GetTotalGenerations()
        {
            return _events.Count > 0 ? _events.Max(e => e.Generation) + 1 : 0;
        }

        /// <summary>
        /// Exports the replay data.
        /// </summary>
        public string ExportJson()
        {
            return JsonSerializer.Serialize(_events, new JsonSerializerOptions
            {
                WriteIndented = false
            });
        }

        /// <summary>
        /// Imports replay data from JSON.
        /// </summary>
        /// <param name="json">JSON string.</param>
        public void ImportJson(string json)
        {
            var imported = JsonSerializer.Deserialize<List<ReplayEvent>>(json);
            if (imported != null)
            {
                _events.Clear();
                _events.AddRange(imported);
                _currentPosition = 0;
            }
        }

        /// <summary>
        /// Resets the replay system.
        /// </summary>
        public void Reset()
        {
            _events.Clear();
            _currentPosition = 0;
            _isRecording = false;
        }
    }

    /// <summary>
    /// Types of replay events.
    /// </summary>
    public enum ReplayEventType
    {
        GenerationStart,
        GenerationEnd,
        GenomeCreated,
        GenomeEvaluated,
        GenomeMutated,
        GenomeCrossover,
        GenomeSnapshot,
        SpeciesCreated,
        SpeciesEliminated,
        Migration,
        Selection,
        ParameterChange
    }

    /// <summary>
    /// A single replay event.
    /// </summary>
    public sealed class ReplayEvent
    {
        public int Position { get; init; }
        public DateTime Timestamp { get; init; }
        public ReplayEventType EventType { get; init; }
        public int Generation { get; init; }
        public string Data { get; init; } = string.Empty;
        public Guid? GenomeId { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Migration Strategies

    /// <summary>
    /// Advanced migration strategies for inter-species gene flow.
    /// Implements island model migration with various topologies.
    /// </summary>
    public sealed class IslandModelMigrationManager
    {
        private readonly EvolutionConfig _config;
        private readonly MigrationTopology _topology;
        private readonly Dictionary<int, List<int>> _neighborMap;
        private readonly Random _rng;

        /// <summary>
        /// Initializes a new instance of the IslandModelMigrationManager class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="topology">Migration topology.</param>
        /// <param name="rng">Random number generator.</param>
        public IslandModelMigrationManager(
            EvolutionConfig config,
            MigrationTopology topology = MigrationTopology.Ring,
            Random? rng = null)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _topology = topology;
            _rng = rng ?? new Random();
            _neighborMap = new Dictionary<int, List<int>>();
        }

        /// <summary>
        /// Configures the neighbor map based on species IDs and topology.
        /// </summary>
        /// <param name="speciesIds">List of species IDs.</param>
        public void ConfigureNeighbors(IReadOnlyList<int> speciesIds)
        {
            _neighborMap.Clear();

            switch (_topology)
            {
                case MigrationTopology.Ring:
                    ConfigureRing(speciesIds);
                    break;
                case MigrationTopology.FullMesh:
                    ConfigureFullMesh(speciesIds);
                    break;
                case MigrationTopology.Random:
                    ConfigureRandom(speciesIds);
                    break;
                case MigrationTopology.Hierarchical:
                    ConfigureHierarchical(speciesIds);
                    break;
                case MigrationTopology.SmallWorld:
                    ConfigureSmallWorld(speciesIds);
                    break;
            }
        }

        /// <summary>
        /// Selects migration partners for each species based on the topology.
        /// </summary>
        /// <param name="species">Current species.</param>
        /// <returns>Migration pairs (source, target).</returns>
        public IReadOnlyList<(int SourceSpeciesId, int TargetSpeciesId)> SelectMigrationPartners(
            ImmutableArray<SpeciesInfo> species)
        {
            var speciesIds = species.Select(s => s.Id).ToList();
            ConfigureNeighbors(speciesIds);

            var pairs = new List<(int, int)>();
            int maxMigrations = Math.Min(_config.MaxMigrationsPerGeneration, speciesIds.Count);

            foreach (var sourceId in speciesIds)
            {
                if (!_neighborMap.TryGetValue(sourceId, out var neighbors) || neighbors.Count == 0)
                    continue;

                int targetId = neighbors[_rng.Next(neighbors.Count)];
                if (sourceId != targetId)
                {
                    pairs.Add((sourceId, targetId));
                }

                if (pairs.Count >= maxMigrations)
                    break;
            }

            return pairs;
        }

        /// <summary>
        /// Selects the best migrants from a species for emigration.
        /// </summary>
        /// <param name="species">Source species info.</param>
        /// <param name="genomes">All genomes.</param>
        /// <param name="count">Number of migrants to select.</param>
        /// <returns>Selected migrant genomes.</returns>
        public IReadOnlyList<GeoGenome> SelectMigrants(
            SpeciesInfo species,
            IReadOnlyList<GeoGenome> genomes,
            int count)
        {
            var speciesGenomes = genomes
                .Where(g => species.MemberIds.Contains(g.Id))
                .ToList();

            if (speciesGenomes.Count == 0)
                return Array.Empty<GeoGenome>();

            var ranked = speciesGenomes
                .OrderByDescending(g => g.Fitness)
                .ToList();

            int eliteBoundary = Math.Max(1, ranked.Count / 3);
            var candidates = ranked.Skip(eliteBoundary).ToList();

            if (candidates.Count == 0)
                candidates = ranked;

            var migrants = new List<GeoGenome>();
            for (int i = 0; i < Math.Min(count, candidates.Count); i++)
            {
                migrants.Add(candidates[i].Clone());
            }

            return migrants;
        }

        /// <summary>
        /// Computes migration acceptance probability based on fitness and diversity.
        /// </summary>
        /// <param name="migrant">The migrant genome.</param>
        /// <param name="targetSpecies">Target species info.</param>
        /// <param name="targetGenomes">Target species genomes.</param>
        /// <returns>Acceptance probability (0-1).</returns>
        public double ComputeAcceptanceProbability(
            GeoGenome migrant,
            SpeciesInfo targetSpecies,
            IReadOnlyList<GeoGenome> targetGenomes)
        {
            if (targetGenomes.Count == 0)
                return 1.0;

            double fitnessRatio = targetSpecies.AverageFitness > 0
                ? migrant.Fitness / targetSpecies.AverageFitness
                : 1.0;

            double fitnessScore = Math.Clamp(fitnessRatio, 0.1, 2.0) / 2.0;

            double diversityBenefit = 0;
            if (targetGenomes.Count > 1)
            {
                var targetHashes = targetGenomes.Select(g => g.ComputeTopologyHash()).ToHashSet();
                long migrantHash = migrant.ComputeTopologyHash();
                bool isNewTopology = !targetHashes.Contains(migrantHash);
                diversityBenefit = isNewTopology ? 0.3 : 0;
            }

            double acceptance = 0.5 * fitnessScore + 0.2 + diversityBenefit;
            return Math.Clamp(acceptance, 0.1, 0.9);
        }

        private void ConfigureRing(IReadOnlyList<int> speciesIds)
        {
            for (int i = 0; i < speciesIds.Count; i++)
            {
                int nextIdx = (i + 1) % speciesIds.Count;
                _neighborMap[speciesIds[i]] = new List<int> { speciesIds[nextIdx] };
            }
        }

        private void ConfigureFullMesh(IReadOnlyList<int> speciesIds)
        {
            foreach (var id in speciesIds)
            {
                _neighborMap[id] = speciesIds.Where(other => other != id).ToList();
            }
        }

        private void ConfigureRandom(IReadOnlyList<int> speciesIds)
        {
            foreach (var id in speciesIds)
            {
                int neighborCount = Math.Max(1, speciesIds.Count / 3);
                var neighbors = speciesIds
                    .Where(other => other != id)
                    .OrderBy(_ => _rng.Next())
                    .Take(neighborCount)
                    .ToList();
                _neighborMap[id] = neighbors;
            }
        }

        private void ConfigureHierarchical(IReadOnlyList<int> speciesIds)
        {
            if (speciesIds.Count <= 2)
            {
                ConfigureFullMesh(speciesIds);
                return;
            }

            int mid = speciesIds.Count / 2;
            var left = speciesIds.Take(mid).ToList();
            var right = speciesIds.Skip(mid).ToList();

            foreach (var id in left)
            {
                _neighborMap[id] = left.Where(other => other != id).ToList();
                if (right.Count > 0)
                    _neighborMap[id].Add(right[_rng.Next(right.Count)]);
            }

            foreach (var id in right)
            {
                _neighborMap[id] = right.Where(other => other != id).ToList();
                if (left.Count > 0)
                    _neighborMap[id].Add(left[_rng.Next(left.Count)]);
            }
        }

        private void ConfigureSmallWorld(IReadOnlyList<int> speciesIds)
        {
            ConfigureRing(speciesIds);

            int rewiredCount = Math.Max(1, speciesIds.Count / 5);
            for (int i = 0; i < rewiredCount; i++)
            {
                int sourceIdx = _rng.Next(speciesIds.Count);
                int sourceId = speciesIds[sourceIdx];

                if (_neighborMap.TryGetValue(sourceId, out var neighbors) && neighbors.Count > 0)
                {
                    int targetIdx = _rng.Next(neighbors.Count);
                    int newTarget = speciesIds[_rng.Next(speciesIds.Count)];
                    if (newTarget != sourceId)
                    {
                        neighbors[targetIdx] = newTarget;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Migration network topologies.
    /// </summary>
    public enum MigrationTopology
    {
        /// <summary>Each species migrates to the next in a ring.</summary>
        Ring,

        /// <summary>Every species can migrate to every other species.</summary>
        FullMesh,

        /// <summary>Random neighbor selection.</summary>
        Random,

        /// <summary>Hierarchical tree structure.</summary>
        Hierarchical,

        /// <summary>Ring with random shortcuts (small-world network).</summary>
        SmallWorld
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Resource Monitor

    /// <summary>
    /// Monitors resource usage during evolution including CPU, memory, and time.
    /// Provides alerts when resource limits are approached.
    /// </summary>
    public sealed class EvolutionResourceMonitor
    {
        private readonly long _maxMemoryBytes;
        private readonly TimeSpan _maxTotalTime;
        private readonly TimeSpan _maxGenerationTime;
        private readonly Queue<ResourceSnapshot> _snapshots;
        private readonly List<ResourceAlert> _alerts;

        /// <summary>
        /// Initializes a new instance of the EvolutionResourceMonitor class.
        /// </summary>
        /// <param name="maxMemoryBytes">Maximum allowed memory usage.</param>
        /// <param name="maxTotalTime">Maximum total evolution time.</param>
        /// <param name="maxGenerationTime">Maximum time per generation.</param>
        public EvolutionResourceMonitor(
            long maxMemoryBytes = 2L * 1024 * 1024 * 1024,
            TimeSpan? maxTotalTime = null,
            TimeSpan? maxGenerationTime = null)
        {
            _maxMemoryBytes = maxMemoryBytes;
            _maxTotalTime = maxTotalTime ?? TimeSpan.FromHours(1);
            _maxGenerationTime = maxGenerationTime ?? TimeSpan.FromMinutes(5);
            _snapshots = new Queue<ResourceSnapshot>();
            _alerts = new List<ResourceAlert>();
        }

        /// <summary>Gets all triggered alerts.</summary>
        public IReadOnlyList<ResourceAlert> Alerts => _alerts.AsReadOnly();

        /// <summary>
        /// Takes a resource snapshot.
        /// </summary>
        /// <param name="generation">Current generation.</param>
        /// <returns>Resource snapshot.</returns>
        public ResourceSnapshot TakeSnapshot(int generation)
        {
            var process = Process.GetCurrentProcess();
            var snapshot = new ResourceSnapshot
            {
                Timestamp = DateTime.UtcNow,
                Generation = generation,
                MemoryUsedBytes = process.WorkingSet64,
                MemoryAvailableBytes = GC.GetTotalMemory(false),
                Gen0Collections = GC.CollectionCount(0),
                Gen1Collections = GC.CollectionCount(1),
                Gen2Collections = GC.CollectionCount(2),
                ThreadCount = Environment.ProcessorCount,
                CpuTimeMs = process.TotalProcessorTime.TotalMilliseconds
            };

            _snapshots.Enqueue(snapshot);
            while (_snapshots.Count > 1000)
                _snapshots.Dequeue();

            CheckLimits(snapshot);
            return snapshot;
        }

        /// <summary>
        /// Gets the current memory usage.
        /// </summary>
        public long GetCurrentMemoryUsage()
        {
            return Process.GetCurrentProcess().WorkingSet64;
        }

        /// <summary>
        /// Gets the memory usage trend.
        /// </summary>
        public double GetMemoryTrend()
        {
            var recent = _snapshots.TakeLast(10).ToList();
            if (recent.Count < 2)
                return 0;

            return (recent[^1].MemoryUsedBytes - recent[0].MemoryUsedBytes) /
                   Math.Max(1, recent[0].MemoryUsedBytes);
        }

        /// <summary>
        /// Forces a garbage collection if memory usage is high.
        /// </summary>
        /// <returns>Memory freed in bytes.</returns>
        public long ForceCollectionIfNeeded()
        {
            long before = GC.GetTotalMemory(false);
            if (before > _maxMemoryBytes * 0.8)
            {
                GC.Collect(2, GCCollectionMode.Forced, true);
                GC.WaitForPendingFinalizers();
            }
            long after = GC.GetTotalMemory(false);
            return Math.Max(0, before - after);
        }

        /// <summary>
        /// Gets a summary of resource usage.
        /// </summary>
        public ResourceSummary GetSummary()
        {
            var allSnapshots = _snapshots.ToList();
            if (allSnapshots.Count == 0)
                return new ResourceSummary();

            return new ResourceSummary
            {
                PeakMemoryBytes = allSnapshots.Max(s => s.MemoryUsedBytes),
                AverageMemoryBytes = (long)allSnapshots.Average(s => s.MemoryUsedBytes),
                CurrentMemoryBytes = allSnapshots.Last().MemoryUsedBytes,
                TotalCpuTimeMs = allSnapshots.Last().CpuTimeMs,
                SnapshotCount = allSnapshots.Count,
                AlertCount = _alerts.Count,
                MemoryUtilization = (double)allSnapshots.Last().MemoryUsedBytes / _maxMemoryBytes
            };
        }

        /// <summary>
        /// Clears all snapshots and alerts.
        /// </summary>
        public void Clear()
        {
            _snapshots.Clear();
            _alerts.Clear();
        }

        private void CheckLimits(ResourceSnapshot snapshot)
        {
            double memoryRatio = (double)snapshot.MemoryUsedBytes / _maxMemoryBytes;

            if (memoryRatio > 0.9)
            {
                _alerts.Add(new ResourceAlert
                {
                    Timestamp = DateTime.UtcNow,
                    AlertType = ResourceAlertType.MemoryCritical,
                    Message = $"Memory usage at {memoryRatio:P0} of limit",
                    Severity = AlertSeverity.Critical
                });
            }
            else if (memoryRatio > 0.75)
            {
                _alerts.Add(new ResourceAlert
                {
                    Timestamp = DateTime.UtcNow,
                    AlertType = ResourceAlertType.MemoryWarning,
                    Message = $"Memory usage at {memoryRatio:P0} of limit",
                    Severity = AlertSeverity.Warning
                });
            }

            var process = Process.GetCurrentProcess();
            if (process.TotalProcessorTime > _maxTotalTime)
            {
                _alerts.Add(new ResourceAlert
                {
                    Timestamp = DateTime.UtcNow,
                    AlertType = ResourceAlertType.TimeLimit,
                    Message = $"Total CPU time exceeded limit",
                    Severity = AlertSeverity.Critical
                });
            }
        }
    }

    /// <summary>
    /// Resource usage snapshot.
    /// </summary>
    public sealed class ResourceSnapshot
    {
        public DateTime Timestamp { get; init; }
        public int Generation { get; init; }
        public long MemoryUsedBytes { get; init; }
        public long MemoryAvailableBytes { get; init; }
        public int Gen0Collections { get; init; }
        public int Gen1Collections { get; init; }
        public int Gen2Collections { get; init; }
        public int ThreadCount { get; init; }
        public double CpuTimeMs { get; init; }
    }

    /// <summary>
    /// Resource usage summary.
    /// </summary>
    public sealed class ResourceSummary
    {
        public long PeakMemoryBytes { get; init; }
        public long AverageMemoryBytes { get; init; }
        public long CurrentMemoryBytes { get; init; }
        public double TotalCpuTimeMs { get; init; }
        public int SnapshotCount { get; init; }
        public int AlertCount { get; init; }
        public double MemoryUtilization { get; init; }
    }

    /// <summary>
    /// Resource usage alert.
    /// </summary>
    public sealed class ResourceAlert
    {
        public DateTime Timestamp { get; init; }
        public ResourceAlertType AlertType { get; init; }
        public string Message { get; init; } = string.Empty;
        public AlertSeverity Severity { get; init; }
    }

    /// <summary>
    /// Types of resource alerts.
    /// </summary>
    public enum ResourceAlertType
    {
        MemoryWarning,
        MemoryCritical,
        TimeLimit,
        GenerationTimeout,
        HighCPU
    }

    /// <summary>
    /// Alert severity levels.
    /// </summary>
    public enum AlertSeverity
    {
        Info,
        Warning,
        Critical
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Pipeline Compositor

    /// <summary>
    /// Composable evolution pipeline that chains multiple evolution operators
    /// in a functional style. Supports filtering, mapping, and reducing
    /// over genome populations.
    /// </summary>
    public sealed class EvolutionPipelineCompositor
    {
        private readonly List<Func<ImmutableArray<GeoGenome>, ImmutableArray<GeoGenome>>> _transforms;
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the EvolutionPipelineCompositor class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public EvolutionPipelineCompositor(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _transforms = new List<Func<ImmutableArray<GeoGenome>, ImmutableArray<GeoGenome>>>();
        }

        /// <summary>
        /// Adds a filter transform to the pipeline.
        /// </summary>
        /// <param name="predicate">Filter predicate.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor Filter(Func<GeoGenome, bool> predicate)
        {
            _transforms.Add(genomes => genomes.Where(predicate).ToImmutableArray());
            return this;
        }

        /// <summary>
        /// Adds a map transform to the pipeline.
        /// </summary>
        /// <param name="mapper">Map function.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor Map(Func<GeoGenome, GeoGenome> mapper)
        {
            _transforms.Add(genomes => genomes.Select(mapper).ToImmutableArray());
            return this;
        }

        /// <summary>
        /// Adds a sort transform to the pipeline.
        /// </summary>
        /// <param name="comparer">Sort comparer.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor Sort(Func<GeoGenome, GeoGenome, int> comparer)
        {
            _transforms.Add(genomes =>
            {
                var sorted = genomes.ToArray();
                Array.Sort(sorted, (a, b) => comparer(a, b));
                return sorted.ToImmutableArray();
            });
            return this;
        }

        /// <summary>
        /// Adds a take (limit) transform to the pipeline.
        /// </summary>
        /// <param name="count">Number of genomes to take.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor Take(int count)
        {
            _transforms.Add(genomes => genomes.Take(count).ToImmutableArray());
            return this;
        }

        /// <summary>
        /// Adds a skip transform to the pipeline.
        /// </summary>
        /// <param name="count">Number of genomes to skip.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor Skip(int count)
        {
            _transforms.Add(genomes => genomes.Skip(count).ToImmutableArray());
            return this;
        }

        /// <summary>
        /// Adds a distinct by topology transform.
        /// </summary>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor DistinctByTopology()
        {
            _transforms.Add(genomes =>
            {
                var seen = new HashSet<long>();
                var result = new List<GeoGenome>();
                foreach (var g in genomes)
                {
                    long hash = g.ComputeTopologyHash();
                    if (seen.Add(hash))
                        result.Add(g);
                }
                return result.ToImmutableArray();
            });
            return this;
        }

        /// <summary>
        /// Adds a diversity-preserving selection transform.
        /// Ensures a minimum number of distinct topologies are retained.
        /// </summary>
        /// <param name="minDiversity">Minimum distinct topologies to retain.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor EnsureDiversity(int minDiversity)
        {
            _transforms.Add(genomes =>
            {
                var result = new List<GeoGenome>();
                var seenHashes = new HashSet<long>();

                var sortedByFitness = genomes.OrderByDescending(g => g.Fitness).ToList();

                foreach (var genome in sortedByFitness)
                {
                    long hash = genome.ComputeTopologyHash();
                    if (seenHashes.Contains(hash) && result.Count >= minDiversity)
                        continue;

                    result.Add(genome);
                    seenHashes.Add(hash);
                }

                return result.ToImmutableArray();
            });
            return this;
        }

        /// <summary>
        /// Adds an elitism transform that preserves top N genomes.
        /// </summary>
        /// <param name="count">Number of elites to preserve.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor PreserveElites(int count)
        {
            _transforms.Add(genomes =>
            {
                var elites = genomes.OrderByDescending(g => g.Fitness).Take(count).ToList();
                var rest = genomes.Where(g => !elites.Any(e => e.Id == g.Id)).ToList();
                return elites.Concat(rest).ToImmutableArray();
            });
            return this;
        }

        /// <summary>
        /// Adds a complexity regularization transform that penalizes complex genomes.
        /// </summary>
        /// <param name="lambda">Regularization strength.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor RegularizeComplexity(double lambda = 0.01)
        {
            _transforms.Add(genomes =>
            {
                foreach (var genome in genomes)
                {
                    double penalty = lambda * genome.ComputeComplexity();
                    genome.Fitness -= penalty;
                }
                return genomes;
            });
            return this;
        }

        /// <summary>
        /// Executes the pipeline on a population.
        /// </summary>
        /// <param name="population">Input population.</param>
        /// <returns>Transformed population.</returns>
        public ImmutableArray<GeoGenome> Execute(ImmutableArray<GeoGenome> population)
        {
            var result = population;
            foreach (var transform in _transforms)
            {
                result = transform(result);
            }
            return result;
        }

        /// <summary>
        /// Executes the pipeline on a population and updates the population record.
        /// </summary>
        /// <param name="population">Input population.</param>
        /// <returns>Updated population record.</returns>
        public GenomePopulation ExecuteOnPopulation(GenomePopulation population)
        {
            var result = Execute(population.Genomes);
            return population with { Genomes = result };
        }

        /// <summary>
        /// Gets the number of transforms in the pipeline.
        /// </summary>
        public int TransformCount => _transforms.Count;

        /// <summary>
        /// Clears all transforms from the pipeline.
        /// </summary>
        public void Clear()
        {
            _transforms.Clear();
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Genome Feature Extractor

    /// <summary>
    /// Extracts statistical and structural features from genomes for analysis,
    /// comparison, and machine learning applications.
    /// </summary>
    public sealed class GenomeFeatureExtractor
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the GenomeFeatureExtractor class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public GenomeFeatureExtractor(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Extracts a comprehensive feature vector from a genome.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Feature vector.</returns>
        public double[] ExtractFeatures(GeoGenome genome)
        {
            var features = new List<double>();

            features.Add(genome.ActiveNeuronCount);
            features.Add(genome.ActiveSynapseCount);
            features.Add(genome.MaxLayerDepth);
            features.Add(genome.ConnectionDensity);
            features.Add(genome.Complexity);
            features.Add(genome.Fitness);

            var layerSizes = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.LayerIndex)
                .Select(g => (double)g.Count())
                .ToList();

            features.Add(layerSizes.Count);
            if (layerSizes.Count > 0)
            {
                features.Add(layerSizes.Average());
                features.Add(layerSizes.Max());
                features.Add(layerSizes.Min());
                features.Add(layerSizes.Count > 1 ? layerSizes.StandardDeviation() : 0);
            }
            else
            {
                features.AddRange(new double[] { 0, 0, 0, 0, 0 });
            }

            var weights = genome.Synapses.Where(s => s.IsActive).Select(s => s.Weight).ToList();
            if (weights.Count > 0)
            {
                features.Add(weights.Average());
                features.Add(weights.Max());
                features.Add(weights.Min());
                features.Add(weights.Sum(w => Math.Abs(w)));
                features.Add(weights.StandardDeviation());
                features.Add(weights.Median());
            }
            else
            {
                features.AddRange(new double[] { 0, 0, 0, 0, 0, 0 });
            }

            var activations = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.Activation)
                .ToList();

            features.Add(activations.Count);
            foreach (var activation in Enum.GetValues<ActivationFunction>())
            {
                var group = activations.FirstOrDefault(g => g.Key == activation);
                features.Add(group != null ? (double)group.Count() / genome.ActiveNeuronCount : 0);
            }

            int inputNeurons = genome.Neurons.Count(n => n.IsActive && n.LayerIndex == 0);
            int outputNeurons = genome.Neurons.Count(n => n.IsActive && n.LayerIndex == genome.MaxLayerDepth);
            int hiddenNeurons = genome.ActiveNeuronCount - inputNeurons - outputNeurons;

            features.Add(inputNeurons);
            features.Add(outputNeurons);
            features.Add(hiddenNeurons);
            features.Add(hiddenNeurons > 0 ? (double)genome.ActiveSynapseCount / hiddenNeurons : 0);

            long topologyHash = genome.ComputeTopologyHash();
            features.Add((topologyHash & 0xFFFF) / (double)0xFFFF);
            features.Add(((topologyHash >> 16) & 0xFFFF) / (double)0xFFFF);

            return features.ToArray();
        }

        /// <summary>
        /// Extracts features from a population.
        /// </summary>
        /// <param name="population">The population.</param>
        /// <returns>Feature matrix [genome_index, feature_index].</returns>
        public double[,] ExtractPopulationFeatures(GenomePopulation population)
        {
            if (population.Genomes.Length == 0)
                return new double[0, 0];

            var firstFeatures = ExtractFeatures(population.Genomes[0]);
            var matrix = new double[population.Genomes.Length, firstFeatures.Length];

            for (int i = 0; i < population.Genomes.Length; i++)
            {
                var features = ExtractFeatures(population.Genomes[i]);
                for (int j = 0; j < Math.Min(features.Length, firstFeatures.Length); j++)
                {
                    matrix[i, j] = features[j];
                }
            }

            return matrix;
        }

        /// <summary>
        /// Gets feature names corresponding to the feature vector.
        /// </summary>
        public IReadOnlyList<string> GetFeatureNames()
        {
            var names = new List<string>
            {
                "ActiveNeuronCount", "ActiveSynapseCount", "MaxLayerDepth",
                "ConnectionDensity", "Complexity", "Fitness",
                "LayerCount", "AvgLayerSize", "MaxLayerSize", "MinLayerSize", "LayerSizeStdDev",
                "AvgWeight", "MaxWeight", "MinWeight", "TotalAbsWeight", "WeightStdDev", "WeightMedian",
                "ActivationDiversity"
            };

            foreach (var activation in Enum.GetValues<ActivationFunction>())
            {
                names.Add($"Fraction_{activation}");
            }

            names.AddRange(new[]
            {
                "InputNeuronCount", "OutputNeuronCount", "HiddenNeuronCount",
                "SynapsesPerHidden", "TopologyHashLow", "TopologyHashHigh"
            });

            return names.AsReadOnly();
        }

        /// <summary>
        /// Normalizes features using min-max scaling.
        /// </summary>
        /// <param name="features">Feature matrix.</param>
        /// <returns>Normalized feature matrix.</returns>
        public double[,] NormalizeFeatures(double[,] features)
        {
            int rows = features.GetLength(0);
            int cols = features.GetLength(1);
            var normalized = new double[rows, cols];

            for (int j = 0; j < cols; j++)
            {
                double min = double.MaxValue;
                double max = double.MinValue;

                for (int i = 0; i < rows; i++)
                {
                    min = Math.Min(min, features[i, j]);
                    max = Math.Max(max, features[i, j]);
                }

                double range = max - min;
                for (int i = 0; i < rows; i++)
                {
                    normalized[i, j] = range > 1e-10
                        ? (features[i, j] - min) / range
                        : 0.5;
                }
            }

            return normalized;
        }

        /// <summary>
        /// Standardizes features using z-score normalization.
        /// </summary>
        /// <param name="features">Feature matrix.</param>
        /// <returns>Standardized feature matrix.</returns>
        public double[,] StandardizeFeatures(double[,] features)
        {
            int rows = features.GetLength(0);
            int cols = features.GetLength(1);
            var standardized = new double[rows, cols];

            for (int j = 0; j < cols; j++)
            {
                double sum = 0;
                for (int i = 0; i < rows; i++)
                    sum += features[i, j];
                double mean = sum / rows;

                double sumSq = 0;
                for (int i = 0; i < rows; i++)
                    sumSq += (features[i, j] - mean) * (features[i, j] - mean);
                double stdDev = Math.Sqrt(sumSq / Math.Max(1, rows - 1));

                for (int i = 0; i < rows; i++)
                {
                    standardized[i, j] = stdDev > 1e-10
                        ? (features[i, j] - mean) / stdDev
                        : 0;
                }
            }

            return standardized;
        }

        /// <summary>
        /// Computes pairwise cosine similarity between genomes.
        /// </summary>
        /// <param name="genomes">List of genomes.</param>
        /// <returns>Cosine similarity matrix.</returns>
        public double[,] ComputeCosineSimilarityMatrix(IReadOnlyList<GeoGenome> genomes)
        {
            int n = genomes.Count;
            var matrix = new double[n, n];

            var featureVectors = genomes.Select(ExtractFeatures).ToList();

            for (int i = 0; i < n; i++)
            {
                matrix[i, i] = 1.0;
                for (int j = i + 1; j < n; j++)
                {
                    double sim = CosineSimilarity(featureVectors[i], featureVectors[j]);
                    matrix[i, j] = sim;
                    matrix[j, i] = sim;
                }
            }

            return matrix;
        }

        private double CosineSimilarity(double[] a, double[] b)
        {
            int dim = Math.Min(a.Length, b.Length);
            double dotProduct = 0, normA = 0, normB = 0;

            for (int i = 0; i < dim; i++)
            {
                dotProduct += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
            return denominator > 1e-10 ? dotProduct / denominator : 0;
        }
    }

    /// <summary>
    /// Extension methods for collections used in evolution.
    /// </summary>
    public static class EvolutionExtensions
    {
        /// <summary>
        /// Computes the standard deviation of a collection of doubles.
        /// </summary>
        public static double StandardDeviation(this IEnumerable<double> source)
        {
            var list = source.ToList();
            if (list.Count <= 1)
                return 0;

            double mean = list.Average();
            double variance = list.Average(v => (v - mean) * (v - mean));
            return Math.Sqrt(variance);
        }

        /// <summary>
        /// Computes the median of a collection of doubles.
        /// </summary>
        public static double Median(this IEnumerable<double> source)
        {
            var sorted = source.OrderBy(v => v).ToList();
            if (sorted.Count == 0)
                return 0;

            int mid = sorted.Count / 2;
            return sorted.Count % 2 == 0
                ? (sorted[mid - 1] + sorted[mid]) / 2.0
                : sorted[mid];
        }

        /// <summary>
        /// Computes the coefficient of variation.
        /// </summary>
        public static double CoefficientOfVariation(this IEnumerable<double> source)
        {
            var list = source.ToList();
            if (list.Count == 0)
                return 0;

            double mean = list.Average();
            if (Math.Abs(mean) < 1e-10)
                return 0;

            double stdDev = list.StandardDeviation();
            return stdDev / Math.Abs(mean);
        }

        /// <summary>
        /// Computes the entropy of a probability distribution.
        /// </summary>
        public static double Entropy(this IEnumerable<double> probabilities)
        {
            return probabilities
                .Where(p => p > 1e-10)
                .Sum(p => -p * Math.Log2(p));
        }

        /// <summary>
        /// Shuffles a list in place using Fisher-Yates algorithm.
        /// </summary>
        public static void Shuffle<T>(this IList<T> list, Random rng)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }

        /// <summary>
        /// Returns a random element from the collection.
        /// </summary>
        public static T RandomElement<T>(this IReadOnlyList<T> list, Random rng)
        {
            if (list.Count == 0)
                throw new InvalidOperationException("Collection is empty.");
            return list[rng.Next(list.Count)];
        }

        /// <summary>
        /// Returns N random elements from the collection without replacement.
        /// </summary>
        public static IReadOnlyList<T> RandomElements<T>(this IReadOnlyList<T> list, int count, Random rng)
        {
            if (count >= list.Count)
                return list.ToList().AsReadOnly();

            var indices = new HashSet<int>();
            var result = new List<T>();

            while (result.Count < count)
            {
                int idx = rng.Next(list.Count);
                if (indices.Add(idx))
                    result.Add(list[idx]);
            }

            return result.AsReadOnly();
        }

        /// <summary>
        /// Partitions a list into batches of a given size.
        /// </summary>
        public static IEnumerable<IReadOnlyList<T>> Batch<T>(this IReadOnlyList<T> list, int batchSize)
        {
            for (int i = 0; i < list.Count; i += batchSize)
            {
                yield return list.Skip(i).Take(batchSize).ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Returns the element that maximizes the given key selector.
        /// </summary>
        public static T MaxBy<T, TKey>(this IReadOnlyList<T> list, Func<T, TKey> keySelector)
            where TKey : IComparable<TKey>
        {
            if (list.Count == 0)
                throw new InvalidOperationException("Collection is empty.");

            T best = list[0];
            TKey bestKey = keySelector(best);

            for (int i = 1; i < list.Count; i++)
            {
                TKey key = keySelector(list[i]);
                if (key.CompareTo(bestKey) > 0)
                {
                    best = list[i];
                    bestKey = key;
                }
            }

            return best;
        }

        /// <summary>
        /// Returns the element that minimizes the given key selector.
        /// </summary>
        public static T MinBy<T, TKey>(this IReadOnlyList<T> list, Func<T, TKey> keySelector)
            where TKey : IComparable<TKey>
        {
            if (list.Count == 0)
                throw new InvalidOperationException("Collection is empty.");

            T best = list[0];
            TKey bestKey = keySelector(best);

            for (int i = 1; i < list.Count; i++)
            {
                TKey key = keySelector(list[i]);
                if (key.CompareTo(bestKey) < 0)
                {
                    best = list[i];
                    bestKey = key;
                }
            }

            return best;
        }

        /// <summary>
        /// Weighted random selection from a collection.
        /// </summary>
        public static T WeightedRandom<T>(this IReadOnlyList<T> list, IReadOnlyList<double> weights, Random rng)
        {
            if (list.Count == 0)
                throw new InvalidOperationException("Collection is empty.");

            double totalWeight = weights.Sum();
            double r = rng.NextDouble() * totalWeight;
            double cumulative = 0;

            for (int i = 0; i < list.Count; i++)
            {
                cumulative += i < weights.Count ? weights[i] : 0;
                if (cumulative >= r)
                    return list[i];
            }

            return list[^1];
        }

        /// <summary>
        /// Computes pairwise distances between all elements.
        /// </summary>
        public static double[,] PairwiseDistances<T>(this IReadOnlyList<T> list, Func<T, T, double> distanceFunc)
        {
            int n = list.Count;
            var matrix = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                matrix[i, i] = 0;
                for (int j = i + 1; j < n; j++)
                {
                    double dist = distanceFunc(list[i], list[j]);
                    matrix[i, j] = dist;
                    matrix[j, i] = dist;
                }
            }

            return matrix;
        }

        /// <summary>
        /// Softmax normalization of a collection of values.
        /// </summary>
        public static double[] Softmax(this IEnumerable<double> source)
        {
            var values = source.ToArray();
            double maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal)).ToArray();
            double sum = exp.Sum();
            return exp.Select(e => e / sum).ToArray();
        }

        /// <summary>
        /// Converts an array to an ImmutableArray.
        /// </summary>
        public static ImmutableArray<T> ToImmutableArray<T>(this T[] array)
        {
            return ImmutableArray.Create(array);
        }

        /// <summary>
        /// Converts a list to an ImmutableArray.
        /// </summary>
        public static ImmutableArray<T> ToImmutableArray<T>(this List<T> list)
        {
            return ImmutableArray.CreateRange(list);
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Hyper-Heuristic Evolution Controller

    /// <summary>
    /// Hyper-heuristic controller that selects and tunes evolution operators
    /// dynamically based on problem characteristics and performance feedback.
    /// Uses a multi-armed bandit approach for operator selection.
    /// </summary>
    public sealed class HyperHeuristicController : IDisposable
    {
        private readonly ConcurrentDictionary<string, OperatorStatistics> _operatorStats;
        private readonly Timer _rebalanceTimer;
        private readonly object _lock = new();
        private readonly Random _rng;
        private bool _disposed;

        private const double ExplorationRate = 0.15;
        private const double LearningRate = 0.1;
        private const double DiscountFactor = 0.95;
        private const int MinSamplesBeforeSelection = 5;

        /// <summary>
        /// Occurs when the operator selection strategy changes.
        /// </summary>
        public event EventHandler<HyperHeuristicEventArgs>? StrategyChanged;

        /// <summary>
        /// Initializes a new instance of the HyperHeuristicController class.
        /// </summary>
        public HyperHeuristicController()
        {
            _operatorStats = new ConcurrentDictionary<string, OperatorStatistics>();
            _rng = Random.Shared;
            _rebalanceTimer = new Timer(RebalanceCallback, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
        }

        /// <summary>
        /// Records the result of an operator execution.
        /// </summary>
        /// <param name="operatorName">Name of the operator.</param>
        /// <param name="improvement">Fitness improvement achieved.</param>
        /// <param name="executionTime">Time taken.</param>
        public void RecordResult(string operatorName, double improvement, TimeSpan executionTime)
        {
            var stats = _operatorStats.GetOrAdd(operatorName, _ => new OperatorStatistics(operatorName));

            lock (stats.Lock)
            {
                stats.TotalExecutions++;
                stats.TotalImprovement += improvement;
                stats.TotalExecutionTime += executionTime;
                stats.RecentImprovements.Add(improvement);

                if (stats.RecentImprovements.Count > 100)
                    stats.RecentImprovements.RemoveAt(0);

                double avgImprovement = stats.RecentImprovements.Average();
                stats.QValue = (1 - LearningRate) * stats.QValue + LearningRate * improvement;
                stats.AverageImprovement = avgImprovement;
                stats.LastUpdate = DateTime.UtcNow;
            }
        }

        /// <summary>
        /// Selects the next operator using the UCB1 (Upper Confidence Bound) algorithm.
        /// </summary>
        /// <param name="availableOperators">Available operator names.</param>
        /// <returns>Selected operator name.</returns>
        public string SelectOperator(IReadOnlyList<string> availableOperators)
        {
            if (availableOperators.Count == 0)
                throw new ArgumentException("No operators available.", nameof(availableOperators));

            if (availableOperators.Count == 1)
                return availableOperators[0];

            if (_rng.NextDouble() < ExplorationRate)
            {
                return availableOperators[_rng.Next(availableOperators.Count)];
            }

            int totalExecutions = _operatorStats.Values.Sum(s =>
            {
                lock (s.Lock)
                    return s.TotalExecutions;
            });

            string bestOperator = availableOperators[0];
            double bestScore = double.MinValue;

            foreach (var opName in availableOperators)
            {
                var stats = _operatorStats.GetOrAdd(opName, _ => new OperatorStatistics(opName));

                double qValue;
                int executions;
                lock (stats.Lock)
                {
                    qValue = stats.QValue;
                    executions = stats.TotalExecutions;
                }

                double explorationBonus = executions > 0
                    ? Math.Sqrt(2 * Math.Log(Math.Max(1, totalExecutions)) / executions)
                    : double.MaxValue;

                double score = qValue + explorationBonus;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestOperator = opName;
                }
            }

            return bestOperator;
        }

        /// <summary>
        /// Gets the best performing operators ranked by Q-value.
        /// </summary>
        /// <param name="count">Number of operators to return.</param>
        /// <returns>Ranked list of operator statistics.</returns>
        public IReadOnlyList<OperatorRanking> GetRankedOperators(int count = 10)
        {
            var rankings = new List<OperatorRanking>();

            foreach (var kvp in _operatorStats)
            {
                lock (kvp.Value.Lock)
                {
                    rankings.Add(new OperatorRanking
                    {
                        OperatorName = kvp.Key,
                        QValue = kvp.Value.QValue,
                        AverageImprovement = kvp.Value.AverageImprovement,
                        TotalExecutions = kvp.Value.TotalExecutions,
                        AverageExecutionTime = kvp.Value.TotalExecutions > 0
                            ? kvp.Value.TotalExecutionTime / kvp.Value.TotalExecutions
                            : TimeSpan.Zero,
                        Rank = 0
                    });
                }
            }

            rankings = rankings.OrderByDescending(r => r.QValue).Take(count).ToList();
            for (int i = 0; i < rankings.Count; i++)
                rankings[i].Rank = i + 1;

            return rankings.AsReadOnly();
        }

        /// <summary>
        /// Gets statistics for a specific operator.
        /// </summary>
        /// <param name="operatorName">Operator name.</param>
        /// <returns>Statistics or null if not found.</returns>
        public OperatorStatistics? GetStatistics(string operatorName)
        {
            return _operatorStats.TryGetValue(operatorName, out var stats) ? stats.Clone() : null;
        }

        /// <summary>
        /// Resets all operator statistics.
        /// </summary>
        public void Reset()
        {
            _operatorStats.Clear();
        }

        private void RebalanceCallback(object? state)
        {
            var staleOperators = _operatorStats
                .Where(kvp =>
                {
                    lock (kvp.Value.Lock)
                        return (DateTime.UtcNow - kvp.Value.LastUpdate).TotalMinutes > 5;
                })
                .Select(kvp => kvp.Key)
                .ToList();

            foreach (var opName in staleOperators)
            {
                if (_operatorStats.TryRemove(opName, out var stats))
                {
                    lock (stats.Lock)
                    {
                        stats.QValue *= 0.9;
                    }
                    _operatorStats[opName] = stats;
                }
            }

            StrategyChanged?.Invoke(this, new HyperHeuristicEventArgs(staleOperators));
        }

        /// <summary>
        /// Disposes of the controller.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _rebalanceTimer?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Statistics for an evolution operator in the hyper-heuristic.
    /// </summary>
    public sealed class OperatorStatistics
    {
        internal readonly object Lock = new();

        /// <summary>
        /// Operator name.
        /// </summary>
        public string OperatorName { get; }

        /// <summary>
        /// Q-value (expected reward).
        /// </summary>
        public double QValue { get; internal set; }

        /// <summary>
        /// Average fitness improvement.
        /// </summary>
        public double AverageImprovement { get; internal set; }

        /// <summary>
        /// Total number of executions.
        /// </summary>
        public int TotalExecutions { get; internal set; }

        /// <summary>
        /// Total improvement across all executions.
        /// </summary>
        public double TotalImprovement { get; internal set; }

        /// <summary>
        /// Total execution time.
        /// </summary>
        public TimeSpan TotalExecutionTime { get; internal set; }

        /// <summary>
        /// Last update timestamp.
        /// </summary>
        public DateTime LastUpdate { get; internal set; }

        /// <summary>
        /// Recent improvements for rolling average.
        /// </summary>
        internal List<double> RecentImprovements { get; } = new();

        /// <summary>
        /// Initializes a new instance of the OperatorStatistics class.
        /// </summary>
        public OperatorStatistics(string operatorName)
        {
            OperatorName = operatorName;
            LastUpdate = DateTime.UtcNow;
        }

        /// <summary>
        /// Creates a deep copy of this statistics object.
        /// </summary>
        public OperatorStatistics Clone()
        {
            lock (Lock)
            {
                var clone = new OperatorStatistics(OperatorName)
                {
                    QValue = QValue,
                    AverageImprovement = AverageImprovement,
                    TotalExecutions = TotalExecutions,
                    TotalImprovement = TotalImprovement,
                    TotalExecutionTime = TotalExecutionTime,
                    LastUpdate = LastUpdate
                };
                clone.RecentImprovements.AddRange(RecentImprovements);
                return clone;
            }
        }

        /// <summary>
        /// Gets the success rate (fraction of improvements).
        /// </summary>
        public double GetSuccessRate()
        {
            lock (Lock)
            {
                return TotalExecutions > 0
                    ? RecentImprovements.Count(v => v > 0) / (double)RecentImprovements.Count
                    : 0;
            }
        }
    }

    /// <summary>
    /// Ranked operator in the hyper-heuristic.
    /// </summary>
    public sealed class OperatorRanking
    {
        /// <summary>Operator name.</summary>
        public required string OperatorName { get; init; }
        /// <summary>Q-value.</summary>
        public double QValue { get; init; }
        /// <summary>Average improvement.</summary>
        public double AverageImprovement { get; init; }
        /// <summary>Total executions.</summary>
        public int TotalExecutions { get; init; }
        /// <summary>Average execution time.</summary>
        public TimeSpan AverageExecutionTime { get; init; }
        /// <summary>Rank position (1-based).</summary>
        public int Rank { get; set; }
    }

    /// <summary>
    /// Event args for hyper-heuristic strategy changes.
    /// </summary>
    public sealed class HyperHeuristicEventArgs : EventArgs
    {
        /// <summary>Operators that were affected by rebalancing.</summary>
        public IReadOnlyList<string> AffectedOperators { get; }

        /// <summary>
        /// Initializes a new instance of the HyperHeuristicEventArgs class.
        /// </summary>
        public HyperHeuristicEventArgs(IReadOnlyList<string> affectedOperators)
        {
            AffectedOperators = affectedOperators;
        }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Adaptive Genome Decomposer

    /// <summary>
    /// Decomposes complex genomes into functional modules using community
    /// detection algorithms on the neural network graph. Enables modular
    /// evolution where independent subnetworks can be evolved separately.
    /// </summary>
    public sealed class AdaptiveGenomeDecomposer
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the AdaptiveGenomeDecomposer class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public AdaptiveGenomeDecomposer(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Decomposes a genome into functional modules using Louvain community detection.
        /// </summary>
        /// <param name="genome">The genome to decompose.</param>
        /// <returns>List of genome modules.</returns>
        public IReadOnlyList<GenomeModule> Decompose(GeoGenome genome)
        {
            var adjacency = BuildAdjacencyMatrix(genome);
            var communities = LouvainCommunityDetection(adjacency);

            var modules = new List<GenomeModule>();

            foreach (var community in communities)
            {
                var moduleNeurons = community.Select(idx => genome.Neurons[idx]).ToList();
                var moduleNeuronIds = new HashSet<long>(moduleNeurons.Select(n => n.Id));
                var moduleSynapses = genome.Synapses
                    .Where(s => moduleNeuronIds.Contains(s.SourceNeuronId) &&
                                moduleNeuronIds.Contains(s.TargetNeuronId))
                    .ToList();

                var inputs = FindExternalInputs(genome, moduleNeuronIds, community);
                var outputs = FindExternalOutputs(genome, moduleNeuronIds, community);

                modules.Add(new GenomeModule
                {
                    Id = modules.Count,
                    Neurons = moduleNeurons.AsReadOnly(),
                    Synapses = moduleSynapses.AsReadOnly(),
                    ExternalInputs = inputs.AsReadOnly(),
                    ExternalOutputs = outputs.AsReadOnly(),
                    Modularity = ComputeModuleModularity(genome, community, moduleNeuronIds),
                    ModuleSize = moduleNeurons.Count,
                    ConnectionDensity = ComputeModuleDensity(genome, moduleNeuronIds),
                    IsInterfacing = inputs.Count > 0 || outputs.Count > 0
                });
            }

            return modules.AsReadOnly();
        }

        /// <summary>
        /// Identifies critical modules whose modification would significantly
        /// impact overall genome fitness.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="modules">Decomposed modules.</param>
        /// <param name="fitnessEvaluator">Fitness evaluator for ablation testing.</param>
        /// <returns>Criticality scores for each module.</returns>
        public IReadOnlyList<ModuleCriticality> IdentifyCriticalModules(
            GeoGenome genome,
            IReadOnlyList<GenomeModule> modules,
            IFitnessEvaluator fitnessEvaluator)
        {
            double baselineFitness = genome.Fitness;
            var criticalities = new List<ModuleCriticality>();

            foreach (var module in modules)
            {
                var ablatedGenome = CreateAblatedGenome(genome, module);
                double ablatedFitness = fitnessEvaluator.Evaluate(ablatedGenome);

                double fitnessDrop = baselineFitness - ablatedFitness;
                double relativeDrop = baselineFitness > 0 ? fitnessDrop / baselineFitness : 0;

                criticalities.Add(new ModuleCriticality
                {
                    ModuleId = module.Id,
                    FitnessDrop = fitnessDrop,
                    RelativeFitnessDrop = relativeDrop,
                    IsCritical = relativeDrop > 0.1,
                    IsRedundant = relativeDrop < 0.01,
                    CriticalityScore = Math.Tanh(relativeDrop * 5.0)
                });
            }

            return criticalities.AsReadOnly();
        }

        /// <summary>
        /// Merges similar modules that have high structural overlap.
        /// </summary>
        /// <param name="modules">List of modules.</param>
        /// <param name="threshold">Similarity threshold for merging.</param>
        /// <returns>Merged module list.</returns>
        public IReadOnlyList<GenomeModule> MergeSimilarModules(
            IReadOnlyList<GenomeModule> modules,
            double threshold = 0.7)
        {
            var merged = new List<GenomeModule>(modules);
            var mergeMap = new Dictionary<int, int>();

            for (int i = 0; i < merged.Count; i++)
            {
                if (mergeMap.ContainsKey(i))
                    continue;

                for (int j = i + 1; j < merged.Count; j++)
                {
                    if (mergeMap.ContainsKey(j))
                        continue;

                    double similarity = ComputeModuleSimilarity(merged[i], merged[j]);
                    if (similarity >= threshold)
                    {
                        int targetIdx = mergeMap.ContainsKey(i) ? mergeMap[i] : i;
                        mergeMap[j] = targetIdx;
                    }
                }
            }

            if (mergeMap.Count == 0)
                return modules;

            var mergedModules = new Dictionary<int, List<GenomeModule>>();
            for (int i = 0; i < merged.Count; i++)
            {
                int target = mergeMap.ContainsKey(i) ? mergeMap[i] : i;
                if (!mergedModules.ContainsKey(target))
                    mergedModules[target] = new List<GenomeModule>();
                mergedModules[target].Add(merged[i]);
            }

            var result = new List<GenomeModule>();
            foreach (var kvp in mergedModules)
            {
                if (kvp.Value.Count == 1)
                {
                    result.Add(kvp.Value[0]);
                }
                else
                {
                    result.Add(MergeModules(kvp.Value));
                }
            }

            return result.AsReadOnly();
        }

        /// <summary>
        /// Extracts a sub-genome containing only the specified module.
        /// </summary>
        /// <param name="genome">Source genome.</param>
        /// <param name="module">Module to extract.</param>
        /// <returns>Sub-genome with module neurons and synapses.</returns>
        public GeoGenome ExtractModuleAsGenome(GeoGenome genome, GenomeModule module)
        {
            var moduleNeuronIds = new HashSet<long>(module.Neurons.Select(n => n.Id));

            var subNeurons = module.Neurons.Select(n =>
            {
                var clone = n.Clone();
                if (moduleNeuronIds.Contains(n.Id))
                {
                    clone.LayerIndex = 0;
                }
                return clone;
            }).ToList();

            var subSynapses = module.Synapses.Select(s =>
            {
                var clone = s.Clone();
                int srcIdx = subNeurons.FindIndex(n => n.Id == clone.SourceNeuronId);
                int tgtIdx = subNeurons.FindIndex(n => n.Id == clone.TargetNeuronId);
                if (srcIdx >= 0 && tgtIdx >= 0)
                {
                    clone.SourceNeuronId = srcIdx;
                    clone.TargetNeuronId = tgtIdx;
                }
                return clone;
            }).ToList();

            return new GeoGenome
            {
                Id = Guid.NewGuid(),
                Neurons = subNeurons,
                Synapses = subSynapses,
                Fitness = genome.Fitness * (module.ModuleSize / (double)Math.Max(1, genome.ActiveNeuronCount))
            };
        }

        private int[,] BuildAdjacencyMatrix(GeoGenome genome)
        {
            int n = genome.Neurons.Count;
            var adjacency = new int[n, n];
            var idToIndex = new Dictionary<long, int>(n);
            for (int i = 0; i < n; i++)
                idToIndex[genome.Neurons[i].Id] = i;

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (idToIndex.TryGetValue(synapse.SourceNeuronId, out int si) &&
                    idToIndex.TryGetValue(synapse.TargetNeuronId, out int ti))
                {
                    adjacency[si, ti] = 1;
                    adjacency[ti, si] = 1;
                }
            }

            return adjacency;
        }

        private List<List<int>> LouvainCommunityDetection(int[,] adjacency)
        {
            int n = adjacency.GetLength(0);
            var communities = Enumerable.Range(0, n).Select(i => new List<int> { i }).ToList();
            var membership = Enumerable.Range(0, n).ToArray();
            int totalEdges = 0;

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    totalEdges += adjacency[i, j];

            if (totalEdges == 0)
            {
                return Enumerable.Range(0, n)
                    .Select(i => new List<int> { i })
                    .ToList();
            }

            bool improved = true;
            int maxIterations = 100;
            int iteration = 0;

            while (improved && iteration < maxIterations)
            {
                improved = false;
                iteration++;

                for (int i = 0; i < n; i++)
                {
                    int currentCommunity = membership[i];
                    int bestCommunity = currentCommunity;
                    double bestGain = 0;

                    var neighborCommunities = new HashSet<int>();
                    for (int j = 0; j < n; j++)
                    {
                        if (adjacency[i, j] > 0)
                            neighborCommunities.Add(membership[j]);
                    }

                    foreach (int neighborComm in neighborCommunities)
                    {
                        if (neighborComm == currentCommunity)
                            continue;

                        double gain = ComputeModularityGain(adjacency, membership, i, currentCommunity, neighborComm, n, totalEdges);
                        if (gain > bestGain)
                        {
                            bestGain = gain;
                            bestCommunity = neighborComm;
                        }
                    }

                    if (bestCommunity != currentCommunity)
                    {
                        membership[i] = bestCommunity;
                        improved = true;
                    }
                }
            }

            var result = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                int comm = membership[i];
                if (!result.ContainsKey(comm))
                    result[comm] = new List<int>();
                result[comm].Add(i);
            }

            return result.Values.ToList();
        }

        private double ComputeModularityGain(int[,] adjacency, int[] membership, int node, int fromComm, int toComm, int n, int totalEdges)
        {
            double m2 = 2.0 * totalEdges;
            if (m2 == 0)
                return 0;

            double ki = 0;
            double kiTo = 0;
            double sigmaTo = 0;
            double sigmaIn = 0;

            for (int j = 0; j < n; j++)
            {
                ki += adjacency[node, j];
                if (membership[j] == toComm)
                {
                    kiTo += adjacency[node, j];
                    for (int k = 0; k < n; k++)
                        sigmaTo += adjacency[j, k];
                }
                if (membership[j] == fromComm)
                {
                    for (int k = 0; k < n; k++)
                        sigmaIn += adjacency[j, k];
                }
            }

            double gain = (kiTo - sigmaTo * ki / m2) - (adjacency[node, node] - sigmaIn * ki / m2);
            return gain / m2;
        }

        private List<long> FindExternalInputs(GeoGenome genome, HashSet<long> moduleNeuronIds, List<int> communityIndices)
        {
            var inputs = new HashSet<long>();

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (moduleNeuronIds.Contains(synapse.TargetNeuronId) &&
                    !moduleNeuronIds.Contains(synapse.SourceNeuronId))
                {
                    inputs.Add(synapse.SourceNeuronId);
                }
            }

            return inputs.ToList();
        }

        private List<long> FindExternalOutputs(GeoGenome genome, HashSet<long> moduleNeuronIds, List<int> communityIndices)
        {
            var outputs = new HashSet<long>();

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (moduleNeuronIds.Contains(synapse.SourceNeuronId) &&
                    !moduleNeuronIds.Contains(synapse.TargetNeuronId))
                {
                    outputs.Add(synapse.TargetNeuronId);
                }
            }

            return outputs.ToList();
        }

        private double ComputeModuleModularity(GeoGenome genome, List<int> communityIndices, HashSet<long> moduleNeuronIds)
        {
            int internalEdges = 0;
            int totalEdges = genome.Synapses.Count(s => s.IsActive);

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (moduleNeuronIds.Contains(synapse.SourceNeuronId) &&
                    moduleNeuronIds.Contains(synapse.TargetNeuronId))
                {
                    internalEdges++;
                }
            }

            return totalEdges > 0 ? (double)internalEdges / totalEdges : 0;
        }

        private double ComputeModuleDensity(GeoGenome genome, HashSet<long> moduleNeuronIds)
        {
            int n = moduleNeuronIds.Count;
            if (n <= 1)
                return 0;

            int internalEdges = genome.Synapses.Count(s =>
                s.IsActive &&
                moduleNeuronIds.Contains(s.SourceNeuronId) &&
                moduleNeuronIds.Contains(s.TargetNeuronId));

            int maxPossible = n * (n - 1) / 2;
            return maxPossible > 0 ? (double)internalEdges / maxPossible : 0;
        }

        private double ComputeModuleSimilarity(GenomeModule a, GenomeModule b)
        {
            var aNeuronIds = new HashSet<long>(a.Neurons.Select(n => n.Id));
            var bNeuronIds = new HashSet<long>(b.Neurons.Select(n => n.Id));

            int intersection = aNeuronIds.Intersect(bNeuronIds).Count();
            int union = aNeuronIds.Union(bNeuronIds).Count();

            return union > 0 ? (double)intersection / union : 0;
        }

        private GenomeModule MergeModules(List<GenomeModule> modules)
        {
            var allNeurons = modules.SelectMany(m => m.Neurons).ToList();
            var allSynapses = modules.SelectMany(m => m.Synapses).ToList();
            var allInputs = modules.SelectMany(m => m.ExternalInputs).Distinct().ToList();
            var allOutputs = modules.SelectMany(m => m.ExternalOutputs).Distinct().ToList();

            int totalNeuronCount = allNeurons.Count;
            int uniqueNeuronCount = allNeurons.Select(n => n.Id).Distinct().Count();

            return new GenomeModule
            {
                Id = modules.Min(m => m.Id),
                Neurons = allNeurons.DistinctBy(n => n.Id).ToList().AsReadOnly(),
                Synapses = allSynapses.DistinctBy(s => s.Id).ToList().AsReadOnly(),
                ExternalInputs = allInputs.AsReadOnly(),
                ExternalOutputs = allOutputs.AsReadOnly(),
                Modularity = modules.Average(m => m.Modularity),
                ModuleSize = uniqueNeuronCount,
                ConnectionDensity = modules.Average(m => m.ConnectionDensity),
                IsInterfacing = allInputs.Count > 0 || allOutputs.Count > 0
            };
        }

        private GeoGenome CreateAblatedGenome(GeoGenome genome, GenomeModule module)
        {
            var moduleNeuronIds = new HashSet<long>(module.Neurons.Select(n => n.Id));

            var ablatedNeurons = genome.Neurons.Select(n =>
            {
                if (moduleNeuronIds.Contains(n.Id))
                {
                    var clone = n.Clone();
                    clone.IsActive = false;
                    return clone;
                }
                return n;
            }).ToList();

            var ablatedSynapses = genome.Synapses.Select(s =>
            {
                if (moduleNeuronIds.Contains(s.SourceNeuronId) ||
                    moduleNeuronIds.Contains(s.TargetNeuronId))
                {
                    var clone = s.Clone();
                    clone.IsActive = false;
                    return clone;
                }
                return s;
            }).ToList();

            return new GeoGenome
            {
                Id = genome.Id,
                Neurons = ablatedNeurons,
                Synapses = ablatedSynapses,
                Fitness = genome.Fitness
            };
        }
    }

    /// <summary>
    /// Represents a functional module within a genome.
    /// </summary>
    public sealed class GenomeModule
    {
        /// <summary>Module ID.</summary>
        public int Id { get; init; }
        /// <summary>Neurons in this module.</summary>
        public IReadOnlyList<GeoNeuron> Neurons { get; init; } = Array.Empty<GeoNeuron>();
        /// <summary>Synapses within this module.</summary>
        public IReadOnlyList<GeoSynapse> Synapses { get; init; } = Array.Empty<GeoSynapse>();
        /// <summary>External input neuron IDs.</summary>
        public IReadOnlyList<long> ExternalInputs { get; init; } = Array.Empty<long>();
        /// <summary>External output neuron IDs.</summary>
        public IReadOnlyList<long> ExternalOutputs { get; init; } = Array.Empty<long>();
        /// <summary>Module modularity score.</summary>
        public double Modularity { get; init; }
        /// <summary>Number of neurons in module.</summary>
        public int ModuleSize { get; init; }
        /// <summary>Internal connection density.</summary>
        public double ConnectionDensity { get; init; }
        /// <summary>Whether this module interfaces with other modules.</summary>
        public bool IsInterfacing { get; init; }
    }

    /// <summary>
    /// Criticality analysis result for a genome module.
    /// </summary>
    public sealed class ModuleCriticality
    {
        /// <summary>Module ID.</summary>
        public int ModuleId { get; init; }
        /// <summary>Absolute fitness drop when module is ablated.</summary>
        public double FitnessDrop { get; init; }
        /// <summary>Relative fitness drop as fraction of baseline.</summary>
        public double RelativeFitnessDrop { get; init; }
        /// <summary>Whether the module is critical (>10% fitness drop).</summary>
        public bool IsCritical { get; init; }
        /// <summary>Whether the module is redundant (<1% fitness drop).</summary>
        public bool IsRedundant { get; init; }
        /// <summary>Normalized criticality score [0, 1].</summary>
        public double CriticalityScore { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Quality Monitor

    /// <summary>
    /// Monitors the quality of the evolution process and detects issues
    /// like premature convergence, fitness stagnation, and genetic drift.
    /// Provides actionable recommendations for parameter tuning.
    /// </summary>
    public sealed class EvolutionQualityMonitor
    {
        private readonly Queue<EvolutionQualitySnapshot> _snapshots;
        private readonly int _maxSnapshots;
        private readonly object _lock = new();

        /// <summary>
        /// Occurs when a quality issue is detected.
        /// </summary>
        public event EventHandler<EvolutionQualityIssueEventArgs>? IssueDetected;

        /// <summary>
        /// Initializes a new instance of the EvolutionQualityMonitor class.
        /// </summary>
        /// <param name="maxSnapshots">Maximum number of snapshots to retain.</param>
        public EvolutionQualityMonitor(int maxSnapshots = 100)
        {
            _maxSnapshots = maxSnapshots;
            _snapshots = new Queue<EvolutionQualitySnapshot>(maxSnapshots);
        }

        /// <summary>
        /// Records an evolution quality snapshot.
        /// </summary>
        /// <param name="generation">Current generation.</param>
        /// <param name="population">Current population.</param>
        /// <param name="species">Current species.</param>
        /// <param name="config">Evolution configuration.</param>
        public void RecordSnapshot(
            int generation,
            GenomePopulation population,
            IReadOnlyList<SpeciesInfo> species,
            EvolutionConfig config)
        {
            var snapshot = CreateSnapshot(generation, population, species, config);

            lock (_lock)
            {
                _snapshots.Enqueue(snapshot);
                while (_snapshots.Count > _maxSnapshots)
                    _snapshots.Dequeue();
            }

            AnalyzeQuality(snapshot, config);
        }

        /// <summary>
        /// Analyzes the current quality of evolution and provides diagnostics.
        /// </summary>
        /// <returns>Quality analysis result.</returns>
        public EvolutionQualityReport AnalyzeCurrentQuality()
        {
            EvolutionQualitySnapshot[] snapshots;
            lock (_lock)
            {
                snapshots = _snapshots.ToArray();
            }

            if (snapshots.Length == 0)
            {
                return new EvolutionQualityReport
                {
                    OverallScore = 0,
                    Status = EvolutionQualityStatus.InsufficientData,
                    Issues = new List<QualityIssue>().AsReadOnly(),
                    Recommendations = new List<string> { "Run more generations for quality analysis." }.AsReadOnly()
                };
            }

            var issues = new List<QualityIssue>();
            var recommendations = new List<string>();

            AnalyzeConvergence(snapshots, issues, recommendations);
            AnalyzeDiversity(snapshots, issues, recommendations);
            AnalyzeStagnation(snapshots, issues, recommendations);
            AnalyzeFitnessProgression(snapshots, issues, recommendations);
            AnalyzeSpeciesHealth(snapshots, issues, recommendations);

            double overallScore = ComputeOverallScore(snapshots, issues);

            return new EvolutionQualityReport
            {
                OverallScore = overallScore,
                Status = DetermineStatus(overallScore, issues),
                Issues = issues.AsReadOnly(),
                Recommendations = recommendations.AsReadOnly(),
                LatestSnapshot = snapshots[^1],
                TrendAnalysis = ComputeTrend(snapshots)
            };
        }

        /// <summary>
        /// Gets the recommended parameter adjustments based on quality analysis.
        /// </summary>
        /// <returns>Recommended configuration adjustments.</returns>
        public IReadOnlyDictionary<string, double> GetRecommendedAdjustments()
        {
            var report = AnalyzeCurrentQuality();
            var adjustments = new Dictionary<string, double>();

            foreach (var issue in report.Issues)
            {
                switch (issue.Type)
                {
                    case QualityIssueType.PrematureConvergence:
                        adjustments["MutationRate"] = Math.Min(1.0, adjustments.GetValueOrDefault("MutationRate", 0.1) * 1.5);
                        adjustments["CrossoverRate"] = Math.Max(0.1, adjustments.GetValueOrDefault("CrossoverRate", 0.8) * 0.8);
                        break;

                    case QualityIssueType.FitnessStagnation:
                        adjustments["MutationRate"] = Math.Min(1.0, adjustments.GetValueOrDefault("MutationRate", 0.1) * 2.0);
                        break;

                    case QualityIssueType.LowDiversity:
                        adjustments["SpeciesCompatibilityThreshold"] =
                            Math.Max(0.1, adjustments.GetValueOrDefault("SpeciesCompatibilityThreshold", 0.5) * 0.7);
                        break;

                    case QualityIssueType.GeneticDrift:
                        adjustments["PopulationSize"] = adjustments.GetValueOrDefault("PopulationSize", 100) * 1.3;
                        break;

                    case QualityIssueType.UnbalancedSpecies:
                        adjustments["ElitismCount"] = Math.Min(20, adjustments.GetValueOrDefault("ElitismCount", 5) + 2);
                        break;
                }
            }

            return adjustments;
        }

        /// <summary>
        /// Gets all recorded snapshots.
        /// </summary>
        public IReadOnlyList<EvolutionQualitySnapshot> GetSnapshots()
        {
            lock (_lock)
            {
                return _snapshots.ToArray().AsReadOnly();
            }
        }

        /// <summary>
        /// Clears all recorded snapshots.
        /// </summary>
        public void Clear()
        {
            lock (_lock)
            {
                _snapshots.Clear();
            }
        }

        private EvolutionQualitySnapshot CreateSnapshot(
            int generation,
            GenomePopulation population,
            IReadOnlyList<SpeciesInfo> species,
            EvolutionConfig config)
        {
            var fitnesses = population.Genomes.Select(g => g.Fitness).ToList();
            var complexities = population.Genomes.Select(g => g.Complexity).ToList();

            return new EvolutionQualitySnapshot
            {
                Timestamp = DateTime.UtcNow,
                Generation = generation,
                PopulationSize = population.Genomes.Length,
                BestFitness = fitnesses.Max(),
                WorstFitness = fitnesses.Min(),
                AverageFitness = fitnesses.Average(),
                FitnessStdDev = fitnesses.StandardDeviation(),
                FitnessMedian = fitnesses.Median(),
                AverageComplexity = complexities.Average(),
                ComplexityStdDev = complexities.StandardDeviation(),
                SpeciesCount = species.Count,
                AverageSpeciesSize = species.Count > 0 ? species.Average(s => s.MemberCount) : 0,
                LargestSpeciesSize = species.Count > 0 ? species.Max(s => s.MemberCount) : 0,
                SmallestSpeciesSize = species.Count > 0 ? species.Min(s => s.MemberCount) : 0,
                SpeciesSizeStdDev = species.Count > 1
                    ? species.Select(s => (double)s.MemberCount).StandardDeviation()
                    : 0,
                UniqueTopologies = population.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count(),
                TopologyDiversityRatio = (double)population.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count() / Math.Max(1, population.Genomes.Length)
            };
        }

        private void AnalyzeQuality(EvolutionQualitySnapshot snapshot, EvolutionConfig config)
        {
            lock (_lock)
            {
                if (_snapshots.Count < 5)
                    return;
            }

            var allSnapshots = GetSnapshots();
            if (allSnapshots.Count < 5)
                return;

            var issues = new List<QualityIssue>();
            var recommendations = new List<string>();

            AnalyzeConvergence(allSnapshots.ToArray(), issues, recommendations);
            AnalyzeDiversity(allSnapshots.ToArray(), issues, recommendations);
            AnalyzeStagnation(allSnapshots.ToArray(), issues, recommendations);

            foreach (var issue in issues)
            {
                IssueDetected?.Invoke(this, new EvolutionQualityIssueEventArgs(issue, recommendations));
            }
        }

        private void AnalyzeConvergence(
            EvolutionQualitySnapshot[] snapshots,
            List<QualityIssue> issues,
            List<string> recommendations)
        {
            if (snapshots.Length < 10)
                return;

            var recentFitnesses = snapshots.Skip(snapshots.Length - 10).Select(s => s.BestFitness).ToList();
            double variance = recentFitnesses.StandardDeviation();

            if (variance < 1e-6)
            {
                issues.Add(new QualityIssue
                {
                    Type = QualityIssueType.PrematureConvergence,
                    Severity = QualityIssueSeverity.High,
                    Message = "Best fitness has converged to a narrow range, suggesting premature convergence.",
                    DetectedAt = DateTime.UtcNow
                });
                recommendations.Add("Increase mutation rate or reduce selection pressure to escape local optima.");
            }
        }

        private void AnalyzeDiversity(
            EvolutionQualitySnapshot[] snapshots,
            List<QualityIssue> issues,
            List<string> recommendations)
        {
            if (snapshots.Length < 5)
                return;

            var recentDiversity = snapshots.Skip(snapshots.Length - 5).Select(s => s.TopologyDiversityRatio).ToList();
            double avgDiversity = recentDiversity.Average();

            if (avgDiversity < 0.3)
            {
                issues.Add(new QualityIssue
                {
                    Type = QualityIssueType.LowDiversity,
                    Severity = QualityIssueSeverity.Medium,
                    Message = $"Topology diversity ratio is low ({avgDiversity:F3}), limiting evolutionary exploration.",
                    DetectedAt = DateTime.UtcNow
                });
                recommendations.Add("Reduce species compatibility threshold or increase mutation rates.");
            }

            if (snapshots.Length >= 3)
            {
                double diversityTrend = snapshots[^1].TopologyDiversityRatio - snapshots[^3].TopologyDiversityRatio;
                if (diversityTrend < -0.2)
                {
                    issues.Add(new QualityIssue
                    {
                        Type = QualityIssueType.GeneticDrift,
                        Severity = QualityIssueSeverity.Medium,
                        Message = "Topology diversity is declining rapidly, indicating genetic drift.",
                        DetectedAt = DateTime.UtcNow
                    });
                    recommendations.Add("Introduce immigration or increase crossover rate.");
                }
            }
        }

        private void AnalyzeStagnation(
            EvolutionQualitySnapshot[] snapshots,
            List<QualityIssue> issues,
            List<string> recommendations)
        {
            if (snapshots.Length < 15)
                return;

            var recentAvgFitness = snapshots.Skip(snapshots.Length - 15).Select(s => s.AverageFitness).ToList();
            double slope = ComputeLinearSlope(recentAvgFitness);

            if (Math.Abs(slope) < 1e-8)
            {
                issues.Add(new QualityIssue
                {
                    Type = QualityIssueType.FitnessStagnation,
                    Severity = QualityIssueSeverity.High,
                    Message = "Average fitness has stagnated over the last 15 generations.",
                    DetectedAt = DateTime.UtcNow
                });
                recommendations.Add("Significantly increase mutation rate or introduce novel genetic operators.");
            }
        }

        private void AnalyzeFitnessProgression(
            EvolutionQualitySnapshot[] snapshots,
            List<QualityIssue> issues,
            List<string> recommendations)
        {
            if (snapshots.Length < 20)
                return;

            var allBestFitness = snapshots.Select(s => s.BestFitness).ToList();
            double overallSlope = ComputeLinearSlope(allBestFitness);

            if (overallSlope < 0)
            {
                issues.Add(new QualityIssue
                {
                    Type = QualityIssueType.FitnessRegression,
                    Severity = QualityIssueSeverity.Critical,
                    Message = "Overall fitness trend is negative, indicating regression.",
                    DetectedAt = DateTime.UtcNow
                });
                recommendations.Add("Review mutation operators for destructive changes. Consider reducing mutation strength.");
            }
        }

        private void AnalyzeSpeciesHealth(
            EvolutionQualitySnapshot[] snapshots,
            List<QualityIssue> issues,
            List<string> recommendations)
        {
            if (snapshots.Length == 0)
                return;

            var latest = snapshots[^1];
            if (latest.SpeciesCount > 1 && latest.SpeciesSizeStdDev > latest.AverageSpeciesSize * 0.5)
            {
                issues.Add(new QualityIssue
                {
                    Type = QualityIssueType.UnbalancedSpecies,
                    Severity = QualityIssueSeverity.Medium,
                    Message = "Species sizes are highly unbalanced, with dominant species emerging.",
                    DetectedAt = DateTime.UtcNow
                });
                recommendations.Add("Increase diversity preservation mechanisms or adjust fitness sharing.");
            }
        }

        private double ComputeLinearSlope(List<double> values)
        {
            int n = values.Count;
            if (n < 2)
                return 0;

            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            for (int i = 0; i < n; i++)
            {
                sumX += i;
                sumY += values[i];
                sumXY += i * values[i];
                sumX2 += i * i;
            }

            double denominator = n * sumX2 - sumX * sumX;
            return Math.Abs(denominator) > 1e-10
                ? (n * sumXY - sumX * sumY) / denominator
                : 0;
        }

        private double ComputeOverallScore(EvolutionQualitySnapshot[] snapshots, List<QualityIssue> issues)
        {
            double score = 1.0;

            foreach (var issue in issues)
            {
                switch (issue.Severity)
                {
                    case QualityIssueSeverity.Critical:
                        score -= 0.3;
                        break;
                    case QualityIssueSeverity.High:
                        score -= 0.2;
                        break;
                    case QualityIssueSeverity.Medium:
                        score -= 0.1;
                        break;
                    case QualityIssueSeverity.Low:
                        score -= 0.05;
                        break;
                }
            }

            if (snapshots.Length > 0)
            {
                score *= snapshots[^1].TopologyDiversityRatio;
            }

            return Math.Max(0, Math.Min(1, score));
        }

        private EvolutionQualityStatus DetermineStatus(double score, List<QualityIssue> issues)
        {
            if (issues.Any(i => i.Severity == QualityIssueSeverity.Critical))
                return EvolutionQualityStatus.Critical;

            if (score > 0.8)
                return EvolutionQualityStatus.Excellent;
            if (score > 0.6)
                return EvolutionQualityStatus.Good;
            if (score > 0.4)
                return EvolutionQualityStatus.Fair;
            if (score > 0.2)
                return EvolutionQualityStatus.Poor;
            return EvolutionQualityStatus.Critical;
        }

        private TrendAnalysis ComputeTrend(EvolutionQualitySnapshot[] snapshots)
        {
            if (snapshots.Length < 3)
            {
                return new TrendAnalysis { FitnessTrend = TrendDirection.Stable, DiversityTrend = TrendDirection.Stable };
            }

            var recentBestFitness = snapshots.Skip(snapshots.Length - 5).Select(s => s.BestFitness).ToList();
            var recentDiversity = snapshots.Skip(snapshots.Length - 5).Select(s => s.TopologyDiversityRatio).ToList();

            return new TrendAnalysis
            {
                FitnessTrend = ComputeTrendDirection(recentBestFitness),
                DiversityTrend = ComputeTrendDirection(recentDiversity),
                FitnessSlope = ComputeLinearSlope(recentBestFitness),
                DiversitySlope = ComputeLinearSlope(recentDiversity),
                GenerationsAnalyzed = Math.Min(5, snapshots.Length)
            };
        }

        private TrendDirection ComputeTrendDirection(List<double> values)
        {
            double slope = ComputeLinearSlope(values);
            if (slope > 0.01)
                return TrendDirection.Improving;
            if (slope < -0.01)
                return TrendDirection.Declining;
            return TrendDirection.Stable;
        }
    }

    /// <summary>
    /// Snapshot of evolution quality metrics at a specific point.
    /// </summary>
    public sealed class EvolutionQualitySnapshot
    {
        /// <summary>Timestamp.</summary>
        public DateTime Timestamp { get; init; }
        /// <summary>Generation number.</summary>
        public int Generation { get; init; }
        /// <summary>Population size.</summary>
        public int PopulationSize { get; init; }
        /// <summary>Best fitness in population.</summary>
        public double BestFitness { get; init; }
        /// <summary>Worst fitness in population.</summary>
        public double WorstFitness { get; init; }
        /// <summary>Average fitness.</summary>
        public double AverageFitness { get; init; }
        /// <summary>Standard deviation of fitness.</summary>
        public double FitnessStdDev { get; init; }
        /// <summary>Median fitness.</summary>
        public double FitnessMedian { get; init; }
        /// <summary>Average complexity.</summary>
        public double AverageComplexity { get; init; }
        /// <summary>Standard deviation of complexity.</summary>
        public double ComplexityStdDev { get; init; }
        /// <summary>Number of species.</summary>
        public int SpeciesCount { get; init; }
        /// <summary>Average species size.</summary>
        public double AverageSpeciesSize { get; init; }
        /// <summary>Largest species size.</summary>
        public int LargestSpeciesSize { get; init; }
        /// <summary>Smallest species size.</summary>
        public int SmallestSpeciesSize { get; init; }
        /// <summary>Standard deviation of species sizes.</summary>
        public double SpeciesSizeStdDev { get; init; }
        /// <summary>Number of unique topologies.</summary>
        public int UniqueTopologies { get; init; }
        /// <summary>Ratio of unique topologies to population size.</summary>
        public double TopologyDiversityRatio { get; init; }
    }

    /// <summary>
    /// Quality issue detected in the evolution process.
    /// </summary>
    public sealed class QualityIssue
    {
        /// <summary>Issue type.</summary>
        public QualityIssueType Type { get; init; }
        /// <summary>Issue severity.</summary>
        public QualityIssueSeverity Severity { get; init; }
        /// <summary>Human-readable message.</summary>
        public string Message { get; init; } = string.Empty;
        /// <summary>When the issue was detected.</summary>
        public DateTime DetectedAt { get; init; }
    }

    /// <summary>
    /// Quality analysis report.
    /// </summary>
    public sealed class EvolutionQualityReport
    {
        /// <summary>Overall quality score [0, 1].</summary>
        public double OverallScore { get; init; }
        /// <summary>Overall quality status.</summary>
        public EvolutionQualityStatus Status { get; init; }
        /// <summary>Detected issues.</summary>
        public IReadOnlyList<QualityIssue> Issues { get; init; } = Array.Empty<QualityIssue>();
        /// <summary>Recommendations for improvement.</summary>
        public IReadOnlyList<string> Recommendations { get; init; } = Array.Empty<string>();
        /// <summary>Latest quality snapshot.</summary>
        public EvolutionQualitySnapshot? LatestSnapshot { get; init; }
        /// <summary>Trend analysis.</summary>
        public TrendAnalysis? TrendAnalysis { get; init; }
    }

    /// <summary>
    /// Trend analysis result.
    /// </summary>
    public sealed class TrendAnalysis
    {
        /// <summary>Fitness trend direction.</summary>
        public TrendDirection FitnessTrend { get; init; }
        /// <summary>Diversity trend direction.</summary>
        public TrendDirection DiversityTrend { get; init; }
        /// <summary>Fitness slope value.</summary>
        public double FitnessSlope { get; init; }
        /// <summary>Diversity slope value.</summary>
        public double DiversitySlope { get; init; }
        /// <summary>Number of generations analyzed.</summary>
        public int GenerationsAnalyzed { get; init; }
    }

    /// <summary>
    /// Event args for detected quality issues.
    /// </summary>
    public sealed class EvolutionQualityIssueEventArgs : EventArgs
    {
        /// <summary>The detected issue.</summary>
        public QualityIssue Issue { get; }
        /// <summary>Recommendations.</summary>
        public IReadOnlyList<string> Recommendations { get; }

        /// <summary>
        /// Initializes a new instance of the EvolutionQualityIssueEventArgs class.
        /// </summary>
        public EvolutionQualityIssueEventArgs(QualityIssue issue, IReadOnlyList<string> recommendations)
        {
            Issue = issue;
            Recommendations = recommendations;
        }
    }

    /// <summary>
    /// Types of quality issues in evolution.
    /// </summary>
    public enum QualityIssueType
    {
        /// <summary>Premature convergence detected.</summary>
        PrematureConvergence,
        /// <summary>Fitness stagnation.</summary>
        FitnessStagnation,
        /// <summary>Low population diversity.</summary>
        LowDiversity,
        /// <summary>Genetic drift.</summary>
        GeneticDrift,
        /// <summary>Fitness regression.</summary>
        FitnessRegression,
        /// <summary>Unbalanced species sizes.</summary>
        UnbalancedSpecies,
        /// <summary>Excessive complexity growth.</summary>
        ComplexityGrowth,
        /// <summary>Computation inefficiency.</summary>
        ComputationInefficiency
    }

    /// <summary>
    /// Severity levels for quality issues.
    /// </summary>
    public enum QualityIssueSeverity
    {
        /// <summary>Low severity.</summary>
        Low,
        /// <summary>Medium severity.</summary>
        Medium,
        /// <summary>High severity.</summary>
        High,
        /// <summary>Critical severity.</summary>
        Critical
    }

    /// <summary>
    /// Overall evolution quality status.
    /// </summary>
    public enum EvolutionQualityStatus
    {
        /// <summary>Insufficient data for analysis.</summary>
        InsufficientData,
        /// <summary>Critical issues detected.</summary>
        Critical,
        /// <summary>Poor quality.</summary>
        Poor,
        /// <summary>Fair quality.</summary>
        Fair,
        /// <summary>Good quality.</summary>
        Good,
        /// <summary>Excellent quality.</summary>
        Excellent
    }

    /// <summary>
    /// Trend direction.
    /// </summary>
    public enum TrendDirection
    {
        /// <summary>Metric is declining.</summary>
        Declining,
        /// <summary>Metric is stable.</summary>
        Stable,
        /// <summary>Metric is improving.</summary>
        Improving
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Genome Expression Profiler

    /// <summary>
    /// Profiles genome expression patterns across different input conditions
    /// to understand network behavior and classify functional motifs.
    /// </summary>
    public sealed class GenomeExpressionProfiler
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the GenomeExpressionProfiler class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public GenomeExpressionProfiler(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Profiles genome activation patterns across multiple input samples.
        /// </summary>
        /// <param name="genome">The genome to profile.</param>
        /// <param name="inputSamples">Input samples for profiling.</param>
        /// <returns>Expression profile.</returns>
        public GenomeExpressionProfile Profile(
            GeoGenome genome,
            IReadOnlyList<double[]> inputSamples)
        {
            var neuronActivations = new Dictionary<long, List<double>>();
            foreach (var neuron in genome.Neurons.Where(n => n.IsActive))
            {
                neuronActivations[neuron.Id] = new List<double>();
            }

            var outputPatterns = new List<double[]>();

            foreach (var input in inputSamples)
            {
                var output = genome.ForwardPass(input);
                outputPatterns.Add(output.ToArray());

                foreach (var neuron in genome.Neurons.Where(n => n.IsActive))
                {
                    neuronActivations[neuron.Id].Add(neuron.LastActivation);
                }
            }

            var neuronStats = new Dictionary<long, NeuronActivationStats>();
            foreach (var kvp in neuronActivations)
            {
                var values = kvp.Value;
                double mean = values.Count > 0 ? values.Average() : 0;
                double variance = values.Count > 1
                    ? values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1)
                    : 0;
                neuronStats[kvp.Key] = new NeuronActivationStats
                {
                    NeuronId = kvp.Key,
                    MeanActivation = mean,
                    Variance = variance,
                    MinActivation = values.Count > 0 ? values.Min() : 0,
                    MaxActivation = values.Count > 0 ? values.Max() : 0,
                    ActivationRange = values.Count > 0 ? values.Max() - values.Min() : 0,
                    SpikeFrequency = values.Count > 0 ? values.Count(v => Math.Abs(v) > 0.5) / (double)values.Count : 0,
                    IsSilent = values.Count == 0 || values.All(v => Math.Abs(v) < 1e-6),
                    IsHyperactive = values.Count > 0 && values.All(v => Math.Abs(v) > 0.9),
                    ActivationEntropy = ComputeActivationEntropy(values)
                };
            }

            var functionalMotifs = IdentifyFunctionalMotifs(genome, neuronStats);
            var sensitivityMatrix = ComputeNeuronSensitivity(genome, inputSamples);

            return new GenomeExpressionProfile
            {
                GenomeId = genome.Id,
                SampleCount = inputSamples.Count,
                NeuronStats = neuronStats,
                OutputPatterns = outputPatterns.Select(p => p.AsReadOnly()).ToList().AsReadOnly(),
                FunctionalMotifs = functionalMotifs.AsReadOnly(),
                SensitivityMatrix = sensitivityMatrix,
                SilenceRatio = (double)neuronStats.Values.Count(s => s.IsSilent) / Math.Max(1, neuronStats.Count),
                HyperactivityRatio = (double)neuronStats.Values.Count(s => s.IsHyperactive) / Math.Max(1, neuronStats.Count),
                AverageActivationEntropy = neuronStats.Values.Average(s => s.ActivationEntropy)
            };
        }

        /// <summary>
        /// Compares expression profiles between two genomes.
        /// </summary>
        public double CompareProfiles(GenomeExpressionProfile profile1, GenomeExpressionProfile profile2)
        {
            var allNeuronIds = profile1.NeuronStats.Keys
                .Union(profile2.NeuronStats.Keys)
                .ToList();

            double totalSimilarity = 0;

            foreach (var neuronId in allNeuronIds)
            {
                if (profile1.NeuronStats.TryGetValue(neuronId, out var stats1) &&
                    profile2.NeuronStats.TryGetValue(neuronId, out var stats2))
                {
                    double sim = 1.0 - Math.Abs(stats1.MeanActivation - stats2.MeanActivation);
                    sim *= 1.0 - Math.Abs(stats1.SpikeFrequency - stats2.SpikeFrequency);
                    totalSimilarity += Math.Max(0, sim);
                }
            }

            return allNeuronIds.Count > 0 ? totalSimilarity / allNeuronIds.Count : 0;
        }

        /// <summary>
        /// Identifies neurons that are critical for specific output behaviors.
        /// </summary>
        public IReadOnlyDictionary<long, double> IdentifyCriticalNeurons(
            GeoGenome genome,
            int targetOutputIndex,
            IReadOnlyList<double[]> inputSamples)
        {
            var baselineOutputs = new List<double[]>();
            foreach (var input in inputSamples)
            {
                baselineOutputs.Add(genome.ForwardPass(input).ToArray());
            }

            var criticalityScores = new Dictionary<long, double>();

            foreach (var neuron in genome.Neurons.Where(n => n.IsActive))
            {
                var modifiedGenome = CreateSilencedGenome(genome, neuron.Id);

                double baselineVariance = 0;
                double modifiedVariance = 0;

                for (int i = 0; i < inputSamples.Count; i++)
                {
                    var modifiedOutput = modifiedGenome.ForwardPass(inputSamples[i]).ToArray();
                    baselineVariance += Math.Abs(baselineOutputs[i][targetOutputIndex]);
                    modifiedVariance += Math.Abs(modifiedOutput[targetOutputIndex]);
                }

                double impact = Math.Abs(baselineVariance - modifiedVariance);
                double normalizedImpact = baselineVariance > 0 ? impact / baselineVariance : 0;

                criticalityScores[neuron.Id] = Math.Tanh(normalizedImpact);
            }

            return criticalityScores;
        }

        private double ComputeActivationEntropy(List<double> activations)
        {
            int bins = 10;
            double min = activations.Min();
            double max = activations.Max();
            double range = max - min;
            if (range < 1e-10)
                return 0;

            var histogram = new int[bins];
            foreach (var val in activations)
            {
                int bin = Math.Min(bins - 1, (int)((val - min) / range * bins));
                histogram[bin]++;
            }

            double entropy = 0;
            double total = activations.Count;
            foreach (var count in histogram)
            {
                if (count > 0)
                {
                    double p = count / total;
                    entropy -= p * Math.Log2(p);
                }
            }

            return entropy / Math.Log2(bins);
        }

        private List<FunctionalMotif> IdentifyFunctionalMotifs(
            GeoGenome genome,
            Dictionary<long, NeuronActivationStats> neuronStats)
        {
            var motifs = new List<FunctionalMotif>();

            var silentNeurons = neuronStats.Values.Where(s => s.IsSilent).Select(s => s.NeuronId).ToList();
            if (silentNeurons.Count > 0)
            {
                motifs.Add(new FunctionalMotif
                {
                    Type = MotifType.SilentSubnetwork,
                    NeuronIds = silentNeurons.AsReadOnly(),
                    Description = $"Subnetwork of {silentNeurons.Count} silent neurons."
                });
            }

            var highVarianceNeurons = neuronStats.Values
                .Where(s => s.Variance > 0.1)
                .OrderByDescending(s => s.Variance)
                .Take(10)
                .Select(s => s.NeuronId)
                .ToList();

            if (highVarianceNeurons.Count > 2)
            {
                motifs.Add(new FunctionalMotif
                {
                    Type = MotifType.VarianceAmplifier,
                    NeuronIds = highVarianceNeurons.AsReadOnly(),
                    Description = $"Set of {highVarianceNeurons.Count} high-variance neurons."
                });
            }

            var bottleneckCandidates = new List<long>();
            var synapseCounts = new Dictionary<long, int>();
            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (!synapseCounts.ContainsKey(synapse.TargetNeuronId))
                    synapseCounts[synapse.TargetNeuronId] = 0;
                synapseCounts[synapse.TargetNeuronId]++;
            }

            foreach (var kvp in synapseCounts)
            {
                if (kvp.Value > genome.ActiveNeuronCount * 0.3)
                    bottleneckCandidates.Add(kvp.Key);
            }

            if (bottleneckCandidates.Count > 0)
            {
                motifs.Add(new FunctionalMotif
                {
                    Type = MotifType.InformationBottleneck,
                    NeuronIds = bottleneckCandidates.AsReadOnly(),
                    Description = $"Potential information bottleneck neurons."
                });
            }

            return motifs;
        }

        private double[,] ComputeNeuronSensitivity(GeoGenome genome, IReadOnlyList<double[]> inputSamples)
        {
            int neuronCount = genome.Neurons.Count;
            int sampleCount = inputSamples.Count;
            var sensitivity = new double[neuronCount, sampleCount];

            var baselineOutputs = new List<double[]>();
            foreach (var input in inputSamples)
            {
                baselineOutputs.Add(genome.ForwardPass(input).ToArray());
            }

            for (int n = 0; n < neuronCount; n++)
            {
                if (!genome.Neurons[n].IsActive)
                    continue;
                var silenced = CreateSilencedGenome(genome, genome.Neurons[n].Id);

                for (int s = 0; s < sampleCount; s++)
                {
                    var modifiedOutput = silenced.ForwardPass(inputSamples[s]).ToArray();
                    double diff = 0;
                    for (int o = 0; o < Math.Min(baselineOutputs[s].Length, modifiedOutput.Length); o++)
                        diff += Math.Abs(baselineOutputs[s][o] - modifiedOutput[o]);
                    sensitivity[n, s] = diff;
                }
            }

            return sensitivity;
        }

        private GeoGenome CreateSilencedGenome(GeoGenome genome, long neuronId)
        {
            var neurons = genome.Neurons.Select(n =>
            {
                if (n.Id == neuronId)
                {
                    var clone = n.Clone();
                    clone.IsActive = false;
                    return clone;
                }
                return n;
            }).ToList();

            var synapses = genome.Synapses.Select(s =>
            {
                if (s.SourceNeuronId == neuronId || s.TargetNeuronId == neuronId)
                {
                    var clone = s.Clone();
                    clone.IsActive = false;
                    return clone;
                }
                return s;
            }).ToList();

            return new GeoGenome
            {
                Id = genome.Id,
                Neurons = neurons,
                Synapses = synapses,
                Fitness = genome.Fitness
            };
        }
    }

    /// <summary>
    /// Expression profile for a genome.
    /// </summary>
    public sealed class GenomeExpressionProfile
    {
        public Guid GenomeId { get; init; }
        public int SampleCount { get; init; }
        public IReadOnlyDictionary<long, NeuronActivationStats> NeuronStats { get; init; } =
            new Dictionary<long, NeuronActivationStats>();
        public IReadOnlyList<IReadOnlyList<double>> OutputPatterns { get; init; } =
            Array.Empty<IReadOnlyList<double>>();
        public IReadOnlyList<FunctionalMotif> FunctionalMotifs { get; init; } =
            Array.Empty<FunctionalMotif>();
        public double[,]? SensitivityMatrix { get; init; }
        public double SilenceRatio { get; init; }
        public double HyperactivityRatio { get; init; }
        public double AverageActivationEntropy { get; init; }
    }

    /// <summary>
    /// Activation statistics for a single neuron.
    /// </summary>
    public sealed class NeuronActivationStats
    {
        public long NeuronId { get; init; }
        public double MeanActivation { get; init; }
        public double Variance { get; init; }
        public double MinActivation { get; init; }
        public double MaxActivation { get; init; }
        public double ActivationRange { get; init; }
        public double SpikeFrequency { get; init; }
        public bool IsSilent { get; init; }
        public bool IsHyperactive { get; init; }
        public double ActivationEntropy { get; init; }
    }

    /// <summary>
    /// Functional motif identified in genome expression.
    /// </summary>
    public sealed class FunctionalMotif
    {
        public MotifType Type { get; init; }
        public IReadOnlyList<long> NeuronIds { get; init; } = Array.Empty<long>();
        public string Description { get; init; } = string.Empty;
    }

    /// <summary>
    /// Types of functional motifs in neural networks.
    /// </summary>
    public enum MotifType
    {
        SilentSubnetwork,
        VarianceAmplifier,
        DirectPathway,
        InformationBottleneck,
        FeedforwardLoop,
        RecurrentLoop,
        ConvergenceHub,
        DivergenceHub
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Population Topology Analyzer

    /// <summary>
    /// Analyzes the topology of the entire population as a meta-graph,
    /// where genomes are nodes and edges represent similarity.
    /// </summary>
    public sealed class PopulationTopologyAnalyzer
    {
        private readonly EvolutionConfig _config;

        public PopulationTopologyAnalyzer(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Builds a similarity graph of the population.
        /// </summary>
        public IReadOnlyDictionary<int, IReadOnlyList<int>> BuildSimilarityGraph(
            GenomePopulation population,
            double similarityThreshold = 0.5)
        {
            var graph = new Dictionary<int, List<int>>();
            int n = population.Genomes.Length;

            for (int i = 0; i < n; i++)
                graph[i] = new List<int>();

            var featureExtractor = new GenomeFeatureExtractor(_config);
            var featureVectors = population.Genomes
                .Select(g => featureExtractor.ExtractFeatures(g))
                .ToArray();

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double similarity = ComputeFeatureSimilarity(featureVectors[i], featureVectors[j]);
                    if (similarity >= similarityThreshold)
                    {
                        graph[i].Add(j);
                        graph[j].Add(i);
                    }
                }
            }

            return graph.ToDictionary(
                kvp => kvp.Key,
                kvp => (IReadOnlyList<int>)kvp.Value.AsReadOnly());
        }

        /// <summary>
        /// Detects clusters in the population similarity graph.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<int>> DetectClusters(
            GenomePopulation population,
            double resolution = 1.0)
        {
            var graph = BuildSimilarityGraph(population, 0.3);
            var communities = new List<List<int>>();
            var membership = new Dictionary<int, int>();
            int communityId = 0;

            foreach (var nodeId in graph.Keys)
            {
                if (!membership.ContainsKey(nodeId))
                {
                    var community = new List<int>();
                    var queue = new Queue<int>();
                    queue.Enqueue(nodeId);

                    while (queue.Count > 0)
                    {
                        int current = queue.Dequeue();
                        if (membership.ContainsKey(current))
                            continue;

                        membership[current] = communityId;
                        community.Add(current);

                        if (graph.TryGetValue(current, out var neighbors))
                        {
                            foreach (var neighbor in neighbors)
                            {
                                if (!membership.ContainsKey(neighbor))
                                    queue.Enqueue(neighbor);
                            }
                        }
                    }

                    communities.Add(community);
                    communityId++;
                }
            }

            return communities.Select(c => (IReadOnlyList<int>)c.AsReadOnly()).ToList().AsReadOnly();
        }

        /// <summary>
        /// Computes population-level network metrics.
        /// </summary>
        public PopulationNetworkMetrics ComputeMetrics(GenomePopulation population)
        {
            var graph = BuildSimilarityGraph(population, 0.3);
            int nodeCount = graph.Count;
            int edgeCount = graph.Values.Sum(n => n.Count) / 2;
            double density = nodeCount > 1 ? (2.0 * edgeCount) / (nodeCount * (nodeCount - 1)) : 0;
            var degrees = graph.Values.Select(n => n.Count).ToList();
            double averageDegree = degrees.Count > 0 ? degrees.Average() : 0;
            double clusteringCoefficient = ComputeClusteringCoefficient(graph);
            var components = FindConnectedComponents(graph);

            return new PopulationNetworkMetrics
            {
                NodeCount = nodeCount,
                EdgeCount = edgeCount,
                Density = density,
                AverageDegree = averageDegree,
                ClusteringCoefficient = clusteringCoefficient,
                ConnectedComponentCount = components.Count,
                LargestComponentRatio = components.Count > 0
                    ? (double)components.Max(c => c.Count) / nodeCount : 0,
                AveragePathLength = ComputeAveragePathLength(graph)
            };
        }

        /// <summary>
        /// Identifies outlier genomes far from the population center.
        /// </summary>
        public IReadOnlyList<int> IdentifyOutliers(GenomePopulation population, double zScoreThreshold = 2.0)
        {
            var featureExtractor = new GenomeFeatureExtractor(_config);
            var features = population.Genomes
                .Select(g => featureExtractor.ExtractFeatures(g))
                .ToArray();

            if (features.Length == 0)
                return Array.Empty<int>().AsReadOnly();

            int featureCount = features[0].Length;
            var mean = new double[featureCount];
            var stdDev = new double[featureCount];

            for (int j = 0; j < featureCount; j++)
            {
                var column = features.Select(f => f[j]).ToList();
                mean[j] = column.Average();
                stdDev[j] = column.StandardDeviation();
            }

            var outlierIds = new List<int>();

            for (int i = 0; i < features.Length; i++)
            {
                double maxZScore = 0;
                for (int j = 0; j < featureCount; j++)
                {
                    if (stdDev[j] > 1e-10)
                    {
                        double zScore = Math.Abs((features[i][j] - mean[j]) / stdDev[j]);
                        maxZScore = Math.Max(maxZScore, zScore);
                    }
                }

                if (maxZScore > zScoreThreshold)
                    outlierIds.Add(i);
            }

            return outlierIds.AsReadOnly();
        }

        private double ComputeFeatureSimilarity(double[] a, double[] b)
        {
            int dim = Math.Min(a.Length, b.Length);
            double dotProduct = 0, normA = 0, normB = 0;
            for (int i = 0; i < dim; i++)
            {
                dotProduct += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
            return denominator > 1e-10 ? dotProduct / denominator : 0;
        }

        private double ComputeClusteringCoefficient(IReadOnlyDictionary<int, IReadOnlyList<int>> graph)
        {
            double totalCoefficient = 0;
            int nodeCount = 0;

            foreach (var kvp in graph)
            {
                var neighbors = kvp.Value;
                int k = neighbors.Count;
                if (k < 2)
                    continue;

                int triangles = 0;
                for (int i = 0; i < k; i++)
                {
                    for (int j = i + 1; j < k; j++)
                    {
                        if (graph.TryGetValue(neighbors[i], out var nn) && nn.Contains(neighbors[j]))
                            triangles++;
                    }
                }

                totalCoefficient += (2.0 * triangles) / (k * (k - 1));
                nodeCount++;
            }

            return nodeCount > 0 ? totalCoefficient / nodeCount : 0;
        }

        private List<List<int>> FindConnectedComponents(IReadOnlyDictionary<int, IReadOnlyList<int>> graph)
        {
            var visited = new HashSet<int>();
            var components = new List<List<int>>();

            foreach (var nodeId in graph.Keys)
            {
                if (visited.Contains(nodeId))
                    continue;

                var component = new List<int>();
                var queue = new Queue<int>();
                queue.Enqueue(nodeId);

                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    if (!visited.Add(current))
                        continue;
                    component.Add(current);

                    if (graph.TryGetValue(current, out var neighbors))
                    {
                        foreach (var neighbor in neighbors)
                        {
                            if (!visited.Contains(neighbor))
                                queue.Enqueue(neighbor);
                        }
                    }
                }

                components.Add(component);
            }

            return components;
        }

        private double ComputeAveragePathLength(IReadOnlyDictionary<int, IReadOnlyList<int>> graph)
        {
            var nodes = graph.Keys.ToList();
            if (nodes.Count <= 1)
                return 0;

            double totalPathLength = 0;
            int pathCount = 0;

            foreach (var source in nodes)
            {
                var distances = new Dictionary<int, int>();
                var queue = new Queue<int>();
                queue.Enqueue(source);
                distances[source] = 0;

                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    int currentDist = distances[current];

                    if (graph.TryGetValue(current, out var neighbors))
                    {
                        foreach (var neighbor in neighbors)
                        {
                            if (!distances.ContainsKey(neighbor))
                            {
                                distances[neighbor] = currentDist + 1;
                                queue.Enqueue(neighbor);
                            }
                        }
                    }
                }

                foreach (var dist in distances.Values)
                {
                    if (dist > 0)
                    {
                        totalPathLength += dist;
                        pathCount++;
                    }
                }
            }

            return pathCount > 0 ? totalPathLength / pathCount : 0;
        }
    }

    /// <summary>
    /// Population-level network metrics.
    /// </summary>
    public sealed class PopulationNetworkMetrics
    {
        public int NodeCount { get; init; }
        public int EdgeCount { get; init; }
        public double Density { get; init; }
        public double AverageDegree { get; init; }
        public double ClusteringCoefficient { get; init; }
        public int ConnectedComponentCount { get; init; }
        public double LargestComponentRatio { get; init; }
        public double AveragePathLength { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Budget Manager

    /// <summary>
    /// Manages computational budgets for evolution runs, including time limits,
    /// generation limits, fitness targets, and adaptive budget allocation.
    /// </summary>
    public sealed class EvolutionBudgetManager
    {
        private long _totalFitnessEvaluations;
        private long _totalMutations;
        private long _totalCrossovers;
        private long _totalSpeciesComputations;
        private readonly Stopwatch _elapsedTimer;
        private readonly object _lock = new();

        /// <summary>Budget configuration.</summary>
        public EvolutionBudgetConfig Config { get; }

        /// <summary>Gets the elapsed evolution time.</summary>
        public TimeSpan Elapsed => _elapsedTimer.Elapsed;

        /// <summary>Gets total fitness evaluations performed.</summary>
        public long TotalFitnessEvaluations => Interlocked.Read(ref _totalFitnessEvaluations);

        /// <summary>Gets total mutations performed.</summary>
        public long TotalMutations => Interlocked.Read(ref _totalMutations);

        /// <summary>Gets total crossovers performed.</summary>
        public long TotalCrossovers => Interlocked.Read(ref _totalCrossovers);

        /// <summary>Gets total species computations performed.</summary>
        public long TotalSpeciesComputations => Interlocked.Read(ref _totalSpeciesComputations);

        /// <summary>
        /// Initializes a new instance of the EvolutionBudgetManager class.
        /// </summary>
        /// <param name="config">Budget configuration.</param>
        public EvolutionBudgetManager(EvolutionBudgetConfig config)
        {
            Config = config ?? throw new ArgumentNullException(nameof(config));
            _elapsedTimer = new Stopwatch();
            _elapsedTimer.Start();
        }

        /// <summary>Records a fitness evaluation.</summary>
        public void RecordFitnessEvaluation(long count = 1)
        {
            Interlocked.Add(ref _totalFitnessEvaluations, count);
        }

        /// <summary>Records a mutation operation.</summary>
        public void RecordMutation(long count = 1)
        {
            Interlocked.Add(ref _totalMutations, count);
        }

        /// <summary>Records a crossover operation.</summary>
        public void RecordCrossover(long count = 1)
        {
            Interlocked.Add(ref _totalCrossovers, count);
        }

        /// <summary>Records a species computation.</summary>
        public void RecordSpeciesComputation(long count = 1)
        {
            Interlocked.Add(ref _totalSpeciesComputations, count);
        }

        /// <summary>
        /// Checks if any budget limit has been exceeded.
        /// </summary>
        public bool IsBudgetExceeded()
        {
            if (Config.MaxTime.HasValue && _elapsedTimer.Elapsed >= Config.MaxTime.Value)
                return true;
            if (Config.MaxFitnessEvaluations.HasValue &&
                _totalFitnessEvaluations >= Config.MaxFitnessEvaluations.Value)
                return true;
            if (Config.MaxGenerations.HasValue)
                return true;
            if (Config.TargetFitness.HasValue)
                return true;
            return false;
        }

        /// <summary>
        /// Checks if a specific budget limit has been exceeded.
        /// </summary>
        public bool IsLimitExceeded(BudgetLimitType type)
        {
            return type switch
            {
                BudgetLimitType.Time => Config.MaxTime.HasValue && _elapsedTimer.Elapsed >= Config.MaxTime.Value,
                BudgetLimitType.FitnessEvaluations => Config.MaxFitnessEvaluations.HasValue &&
                    _totalFitnessEvaluations >= Config.MaxFitnessEvaluations.Value,
                BudgetLimitType.Mutations => Config.MaxMutations.HasValue &&
                    _totalMutations >= Config.MaxMutations.Value,
                BudgetLimitType.Crossovers => Config.MaxCrossovers.HasValue &&
                    _totalCrossovers >= Config.MaxCrossovers.Value,
                _ => false
            };
        }

        /// <summary>
        /// Gets the remaining budget for each limit type.
        /// </summary>
        public BudgetRemaining GetRemainingBudget()
        {
            return new BudgetRemaining
            {
                RemainingTime = Config.MaxTime.HasValue
                    ? Config.MaxTime.Value - _elapsedTimer.Elapsed
                    : (TimeSpan?)null,
                RemainingFitnessEvaluations = Config.MaxFitnessEvaluations.HasValue
                    ? Math.Max(0, Config.MaxFitnessEvaluations.Value - _totalFitnessEvaluations)
                    : (long?)null,
                RemainingMutations = Config.MaxMutations.HasValue
                    ? Math.Max(0, Config.MaxMutations.Value - _totalMutations)
                    : (long?)null,
                RemainingCrossovers = Config.MaxCrossovers.HasValue
                    ? Math.Max(0, Config.MaxCrossovers.Value - _totalCrossovers)
                    : (long?)null
            };
        }

        /// <summary>
        /// Gets the completion percentage for each budget limit.
        /// </summary>
        public IReadOnlyDictionary<BudgetLimitType, double> GetCompletionPercentages()
        {
            var percentages = new Dictionary<BudgetLimitType, double>();

            if (Config.MaxTime.HasValue)
            {
                percentages[BudgetLimitType.Time] = Math.Min(1.0,
                    _elapsedTimer.Elapsed.TotalMilliseconds / Config.MaxTime.Value.TotalMilliseconds);
            }

            if (Config.MaxFitnessEvaluations.HasValue)
            {
                percentages[BudgetLimitType.FitnessEvaluations] = Math.Min(1.0,
                    (double)_totalFitnessEvaluations / Config.MaxFitnessEvaluations.Value);
            }

            if (Config.MaxMutations.HasValue)
            {
                percentages[BudgetLimitType.Mutations] = Math.Min(1.0,
                    (double)_totalMutations / Config.MaxMutations.Value);
            }

            if (Config.MaxCrossovers.HasValue)
            {
                percentages[BudgetLimitType.Crossovers] = Math.Min(1.0,
                    (double)_totalCrossovers / Config.MaxCrossovers.Value);
            }

            return percentages;
        }

        /// <summary>
        /// Gets the most constraining budget limit.
        /// </summary>
        public BudgetLimitType? GetMostConstrainingLimit()
        {
            var percentages = GetCompletionPercentages();
            if (percentages.Count == 0)
                return null;
            return percentages.OrderByDescending(kvp => kvp.Value).First().Key;
        }

        /// <summary>Resets all budget counters.</summary>
        public void Reset()
        {
            Interlocked.Exchange(ref _totalFitnessEvaluations, 0);
            Interlocked.Exchange(ref _totalMutations, 0);
            Interlocked.Exchange(ref _totalCrossovers, 0);
            Interlocked.Exchange(ref _totalSpeciesComputations, 0);
            lock (_lock)
            { _elapsedTimer.Restart(); }
        }

        /// <summary>Stops the elapsed timer.</summary>
        public void Stop()
        {
            lock (_lock)
            { _elapsedTimer.Stop(); }
        }

        /// <summary>Creates a budget report.</summary>
        public BudgetReport CreateReport()
        {
            return new BudgetReport
            {
                ElapsedTime = _elapsedTimer.Elapsed,
                TotalFitnessEvaluations = _totalFitnessEvaluations,
                TotalMutations = _totalMutations,
                TotalCrossovers = _totalCrossovers,
                TotalSpeciesComputations = _totalSpeciesComputations,
                RemainingBudget = GetRemainingBudget(),
                CompletionPercentages = GetCompletionPercentages(),
                MostConstrainingLimit = GetMostConstrainingLimit(),
                IsBudgetExceeded = IsBudgetExceeded()
            };
        }
    }

    /// <summary>
    /// Budget configuration for evolution.
    /// </summary>
    public sealed class EvolutionBudgetConfig
    {
        /// <summary>Maximum time allowed.</summary>
        public TimeSpan? MaxTime { get; init; }
        /// <summary>Maximum fitness evaluations.</summary>
        public long? MaxFitnessEvaluations { get; init; }
        /// <summary>Maximum number of generations.</summary>
        public int? MaxGenerations { get; init; }
        /// <summary>Target fitness value.</summary>
        public double? TargetFitness { get; init; }
        /// <summary>Maximum mutation operations.</summary>
        public long? MaxMutations { get; init; }
        /// <summary>Maximum crossover operations.</summary>
        public long? MaxCrossovers { get; init; }
        /// <summary>Maximum memory usage in bytes.</summary>
        public long? MaxMemoryBytes { get; init; }
        /// <summary>Maximum species computations.</summary>
        public long? MaxSpeciesComputations { get; init; }
    }

    /// <summary>Types of budget limits.</summary>
    public enum BudgetLimitType
    {
        Time,
        FitnessEvaluations,
        Mutations,
        Crossovers,
        Memory,
        SpeciesComputations
    }

    /// <summary>Remaining budget information.</summary>
    public sealed class BudgetRemaining
    {
        public TimeSpan? RemainingTime { get; init; }
        public long? RemainingFitnessEvaluations { get; init; }
        public long? RemainingMutations { get; init; }
        public long? RemainingCrossovers { get; init; }
    }

    /// <summary>Budget status report.</summary>
    public sealed class BudgetReport
    {
        public TimeSpan ElapsedTime { get; init; }
        public long TotalFitnessEvaluations { get; init; }
        public long TotalMutations { get; init; }
        public long TotalCrossovers { get; init; }
        public long TotalSpeciesComputations { get; init; }
        public BudgetRemaining RemainingBudget { get; init; } = new();
        public IReadOnlyDictionary<BudgetLimitType, double> CompletionPercentages { get; init; } =
            new Dictionary<BudgetLimitType, double>();
        public BudgetLimitType? MostConstrainingLimit { get; init; }
        public bool IsBudgetExceeded { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Multi-Objective Optimization Support

    /// <summary>
    /// Supports multi-objective optimization using NSGA-II and NSGA-III
    /// algorithms for Pareto-optimal genome evolution.
    /// </summary>
    public sealed class MultiObjectiveOptimizer
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the MultiObjectiveOptimizer class.
        /// </summary>
        public MultiObjectiveOptimizer(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Computes Pareto fronts using NSGA-II non-dominated sorting.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<GeoGenome>> ComputeParetoFronts(
            GenomePopulation population,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions)
        {
            var objectives = new List<double[]>();
            foreach (var genome in population.Genomes)
            {
                var objValues = new double[objectiveCount];
                for (int i = 0; i < objectiveCount; i++)
                    objValues[i] = objectiveFunctions[i](genome);
                objectives.Add(objValues);
            }

            var dominatedCount = new int[population.Genomes.Length];
            var dominatesSet = new List<int>[population.Genomes.Length];

            for (int i = 0; i < population.Genomes.Length; i++)
            {
                dominatesSet[i] = new List<int>();
                dominatedCount[i] = 0;
            }

            for (int i = 0; i < population.Genomes.Length; i++)
            {
                for (int j = i + 1; j < population.Genomes.Length; j++)
                {
                    if (Dominates(objectives[i], objectives[j]))
                    {
                        dominatesSet[i].Add(j);
                        dominatedCount[j]++;
                    }
                    else if (Dominates(objectives[j], objectives[i]))
                    {
                        dominatesSet[j].Add(i);
                        dominatedCount[i]++;
                    }
                }
            }

            var currentFront = new List<int>();
            for (int i = 0; i < population.Genomes.Length; i++)
            {
                if (dominatedCount[i] == 0)
                    currentFront.Add(i);
            }

            var fronts = new List<List<int>>();
            while (currentFront.Count > 0)
            {
                fronts.Add(currentFront);
                var nextFront = new List<int>();

                foreach (int i in currentFront)
                {
                    foreach (int j in dominatesSet[i])
                    {
                        dominatedCount[j]--;
                        if (dominatedCount[j] == 0)
                            nextFront.Add(j);
                    }
                }

                currentFront = nextFront;
            }

            return fronts.Select(front =>
                (IReadOnlyList<GeoGenome>)front.Select(idx => population.Genomes[idx]).ToList().AsReadOnly()
            ).ToList().AsReadOnly();
        }

        /// <summary>
        /// Computes crowding distances for diversity preservation.
        /// </summary>
        public double[] ComputeCrowdingDistances(
            IReadOnlyList<GeoGenome> front,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions)
        {
            int n = front.Count;
            if (n <= 2)
                return Enumerable.Repeat(double.MaxValue, n).ToArray();

            var distances = new double[n];

            for (int m = 0; m < objectiveCount; m++)
            {
                var objectiveValues = front.Select(g => objectiveFunctions[m](g)).ToArray();
                var sortedIndices = Enumerable.Range(0, n)
                    .OrderBy(i => objectiveValues[i])
                    .ToArray();

                distances[sortedIndices[0]] = double.MaxValue;
                distances[sortedIndices[^1]] = double.MaxValue;

                double range = objectiveValues[sortedIndices[^1]] - objectiveValues[sortedIndices[0]];
                if (range < 1e-10)
                    continue;

                for (int i = 1; i < n - 1; i++)
                {
                    double spread = objectiveValues[sortedIndices[i + 1]] - objectiveValues[sortedIndices[i - 1]];
                    distances[sortedIndices[i]] += spread / range;
                }
            }

            return distances;
        }

        /// <summary>
        /// Performs NSGA-II selection for the next generation.
        /// </summary>
        public IReadOnlyList<GeoGenome> SelectNextGeneration(
            GenomePopulation population,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions,
            int targetSize)
        {
            var fronts = ComputeParetoFronts(population, objectiveCount, objectiveFunctions);
            var selected = new List<GeoGenome>();
            int frontIndex = 0;

            while (selected.Count < targetSize && frontIndex < fronts.Count)
            {
                var currentFront = fronts[frontIndex];

                if (selected.Count + currentFront.Count <= targetSize)
                {
                    selected.AddRange(currentFront);
                }
                else
                {
                    var crowdingDistances = ComputeCrowdingDistances(
                        currentFront, objectiveCount, objectiveFunctions);

                    var sortedFront = currentFront
                        .Select((g, i) => new { Genome = g, Index = i })
                        .OrderByDescending(x => crowdingDistances[x.Index])
                        .ToList();

                    int remaining = targetSize - selected.Count;
                    selected.AddRange(sortedFront.Take(remaining).Select(x => x.Genome));
                }

                frontIndex++;
            }

            return selected.AsReadOnly();
        }

        /// <summary>
        /// Finds the knee point on a Pareto front (best trade-off).
        /// </summary>
        public GeoGenome? FindKneePoint(
            IReadOnlyList<GeoGenome> front,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions)
        {
            if (front.Count < 3)
                return front.FirstOrDefault();

            var objectives = front.Select(g =>
            {
                var obj = new double[objectiveCount];
                for (int i = 0; i < objectiveCount; i++)
                    obj[i] = objectiveFunctions[i](g);
                return obj;
            }).ToList();

            var minValues = new double[objectiveCount];
            var maxValues = new double[objectiveCount];
            for (int j = 0; j < objectiveCount; j++)
            {
                minValues[j] = objectives.Min(o => o[j]);
                maxValues[j] = objectives.Max(o => o[j]);
            }

            var normalizedObjectives = objectives.Select(o =>
            {
                var normalized = new double[objectiveCount];
                for (int j = 0; j < objectiveCount; j++)
                {
                    double range = maxValues[j] - minValues[j];
                    normalized[j] = range > 1e-10 ? (o[j] - minValues[j]) / range : 0;
                }
                return normalized;
            }).ToList();

            var referencePoint = new double[objectiveCount];
            for (int j = 0; j < objectiveCount; j++)
                referencePoint[j] = 0;

            double bestDistance = double.MinValue;
            int bestIndex = 0;

            for (int i = 0; i < normalizedObjectives.Count; i++)
            {
                double distance = 0;
                for (int j = 0; j < objectiveCount; j++)
                {
                    double diff = normalizedObjectives[i][j] - referencePoint[j];
                    distance += diff * diff;
                }
                distance = Math.Sqrt(distance);

                double angleBonus = ComputeAngleBonus(normalizedObjectives[i], referencePoint);
                double score = distance + angleBonus * 0.1;

                if (score > bestDistance)
                {
                    bestDistance = score;
                    bestIndex = i;
                }
            }

            return front[bestIndex];
        }

        /// <summary>
        /// Computes the hypervolume indicator for a Pareto front.
        /// </summary>
        public double ComputeHypervolume(
            IReadOnlyList<GeoGenome> front,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions,
            double[] referencePoint)
        {
            if (front.Count == 0)
                return 0;

            var objectives = front.Select(g =>
            {
                var obj = new double[objectiveCount];
                for (int i = 0; i < objectiveCount; i++)
                    obj[i] = objectiveFunctions[i](g);
                return obj;
            }).ToList();

            var filteredObjectives = new List<double[]>();
            foreach (var obj in objectives)
            {
                bool dominated = false;
                for (int j = 0; j < objectiveCount; j++)
                {
                    if (obj[j] >= referencePoint[j])
                    {
                        dominated = true;
                        break;
                    }
                }
                if (!dominated)
                    filteredObjectives.Add(obj);
            }

            if (filteredObjectives.Count == 0)
                return 0;

            return ComputeHypervolumeRecursive(filteredObjectives, referencePoint, objectiveCount);
        }

        /// <summary>
        /// Computes the inverted generational distance (IGD) for a Pareto front approximation.
        /// </summary>
        public double ComputeIGD(
            IReadOnlyList<GeoGenome> approximation,
            IReadOnlyList<double[]> referenceFront,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions)
        {
            if (referenceFront.Count == 0)
                return 0;

            var approxObjectives = approximation.Select(g =>
            {
                var obj = new double[objectiveCount];
                for (int i = 0; i < objectiveCount; i++)
                    obj[i] = objectiveFunctions[i](g);
                return obj;
            }).ToList();

            double totalMinDistance = 0;

            foreach (var refPoint in referenceFront)
            {
                double minDistance = double.MaxValue;

                foreach (var approxPoint in approxObjectives)
                {
                    double distance = EuclideanDistance(refPoint, approxPoint);
                    minDistance = Math.Min(minDistance, distance);
                }

                totalMinDistance += minDistance;
            }

            return totalMinDistance / referenceFront.Count;
        }

        /// <summary>
        /// Merges two Pareto fronts and returns the non-dominated subset.
        /// </summary>
        public IReadOnlyList<GeoGenome> MergeParetoFronts(
            IReadOnlyList<GeoGenome> front1,
            IReadOnlyList<GeoGenome> front2,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions)
        {
            var merged = front1.Concat(front2).ToList();
            var population = new GenomePopulation
            {
                Genomes = merged.ToImmutableArray(),
                Generation = 0,
                SpeciesCount = 0,
                AverageFitness = merged.Average(g => g.Fitness),
                BestFitness = merged.Max(g => g.Fitness),
                WorstFitness = merged.Min(g => g.Fitness),
                Timestamp = DateTime.UtcNow
            };

            var fronts = ComputeParetoFronts(population, objectiveCount, objectiveFunctions);
            return fronts.Count > 0 ? fronts[0] : Array.Empty<GeoGenome>().AsReadOnly();
        }

        private bool Dominates(double[] a, double[] b)
        {
            bool atLeastOneBetter = false;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] < b[i])
                    return false;
                if (a[i] > b[i])
                    atLeastOneBetter = true;
            }
            return atLeastOneBetter;
        }

        private double ComputeAngleBonus(double[] point, double[] reference)
        {
            double dotProduct = 0;
            double normA = 0;
            for (int i = 0; i < point.Length; i++)
            {
                double diff = point[i] - reference[i];
                dotProduct += diff;
                normA += diff * diff;
            }
            return normA > 1e-10 ? dotProduct / Math.Sqrt(normA) : 0;
        }

        private double ComputeHypervolumeRecursive(List<double[]> points, double[] reference, int objectiveCount)
        {
            if (objectiveCount == 1)
            {
                double volume = 0;
                var sortedPoints = points.OrderBy(p => p[0]).ToList();

                double prevX = reference[0];
                foreach (var point in sortedPoints)
                {
                    if (point[0] < prevX)
                    {
                        volume += prevX - point[0];
                        prevX = point[0];
                    }
                }

                return volume;
            }

            var sortedByLastDim = points.OrderBy(p => p[objectiveCount - 1]).ToList();
            double hypervolume = 0;
            var processedPoints = new List<double[]>();

            double prevValue = reference[objectiveCount - 1];
            foreach (var point in sortedByLastDim)
            {
                if (point[objectiveCount - 1] >= prevValue)
                    continue;

                double sliceHeight = prevValue - point[objectiveCount - 1];
                prevValue = point[objectiveCount - 1];

                var projectedPoint = new double[objectiveCount - 1];
                for (int i = 0; i < objectiveCount - 1; i++)
                    projectedPoint[i] = point[i];

                processedPoints.Add(projectedPoint);

                var filteredPoints = processedPoints
                    .Where(p => p.All(v => v < reference[Array.IndexOf(p, p.Min())]))
                    .ToList();

                if (filteredPoints.Count > 0)
                {
                    hypervolume += sliceHeight * ComputeHypervolumeRecursive(
                        filteredPoints,
                        reference.Take(objectiveCount - 1).ToArray(),
                        objectiveCount - 1);
                }
            }

            return hypervolume;
        }

        private double EuclideanDistance(double[] a, double[] b)
        {
            double sum = 0;
            int dim = Math.Min(a.Length, b.Length);
            for (int i = 0; i < dim; i++)
            {
                double diff = a[i] - b[i];
                sum += diff * diff;
            }
            return Math.Sqrt(sum);
        }
    }

    /// <summary>
    /// Result of a multi-objective optimization run.
    /// </summary>
    public sealed class MultiObjectiveResult
    {
        /// <summary>Pareto front genomes.</summary>
        public IReadOnlyList<GeoGenome> ParetoFront { get; init; } = Array.Empty<GeoGenome>();
        /// <summary>Hypervolume indicator.</summary>
        public double Hypervolume { get; init; }
        /// <summary>Inverted generational distance.</summary>
        public double IGD { get; init; }
        /// <summary>Knee point genome.</summary>
        public GeoGenome? KneePoint { get; init; }
        /// <summary>Number of Pareto fronts.</summary>
        public int FrontCount { get; init; }
        /// <summary>Number of generations used.</summary>
        public int GenerationsUsed { get; init; }
        /// <summary>Total evaluation time.</summary>
        public TimeSpan TotalTime { get; init; }
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Experiment Framework

    /// <summary>
    /// Framework for running controlled evolution experiments with
    /// parameter sweeps, ablation studies, and statistical analysis.
    /// </summary>
    public sealed class EvolutionExperimentFramework
    {
        private readonly EvolutionConfig _baseConfig;
        private readonly Random _rng;

        /// <summary>
        /// Initializes a new instance of the EvolutionExperimentFramework class.
        /// </summary>
        public EvolutionExperimentFramework(EvolutionConfig baseConfig)
        {
            _baseConfig = baseConfig ?? throw new ArgumentNullException(nameof(baseConfig));
            _rng = Random.Shared;
        }

        /// <summary>
        /// Runs a parameter sweep experiment.
        /// </summary>
        public async Task<IReadOnlyList<ExperimentResult>> RunParameterSweepAsync(
            string parameterName,
            IReadOnlyList<double> parameterValues,
            Func<EvolutionConfig, Task<GeoGenome>> evolutionRunner,
            CancellationToken cancellationToken = default)
        {
            var results = new List<ExperimentResult>();

            foreach (var value in parameterValues)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var config = ApplyParameter(_baseConfig, parameterName, value);
                var stopwatch = Stopwatch.StartNew();

                try
                {
                    var bestGenome = await evolutionRunner(config);
                    stopwatch.Stop();

                    results.Add(new ExperimentResult
                    {
                        ParameterName = parameterName,
                        ParameterValue = value,
                        BestFitness = bestGenome.Fitness,
                        BestGenome = bestGenome,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = true,
                        Error = null
                    });
                }
                catch (Exception ex)
                {
                    stopwatch.Stop();
                    results.Add(new ExperimentResult
                    {
                        ParameterName = parameterName,
                        ParameterValue = value,
                        BestFitness = double.MinValue,
                        BestGenome = null,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = false,
                        Error = ex.Message
                    });
                }
            }

            return results.AsReadOnly();
        }

        /// <summary>
        /// Runs an ablation study by removing one component at a time.
        /// </summary>
        public async Task<IReadOnlyList<ExperimentResult>> RunAblationStudyAsync(
            IReadOnlyList<string> componentNames,
            Func<EvolutionConfig, Task<GeoGenome>> evolutionRunner,
            CancellationToken cancellationToken = default)
        {
            var baselineResult = await evolutionRunner(_baseConfig);
            double baselineFitness = baselineResult.Fitness;

            var results = new List<ExperimentResult>();

            results.Add(new ExperimentResult
            {
                ParameterName = "Baseline",
                ParameterValue = 1.0,
                BestFitness = baselineFitness,
                BestGenome = baselineResult,
                ExecutionTime = TimeSpan.Zero,
                IsSuccess = true,
                Error = null
            });

            foreach (var component in componentNames)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var config = RemoveComponent(_baseConfig, component);
                var stopwatch = Stopwatch.StartNew();

                try
                {
                    var bestGenome = await evolutionRunner(config);
                    stopwatch.Stop();

                    double fitnessDrop = baselineFitness - bestGenome.Fitness;

                    results.Add(new ExperimentResult
                    {
                        ParameterName = $"Ablated: {component}",
                        ParameterValue = fitnessDrop,
                        BestFitness = bestGenome.Fitness,
                        BestGenome = bestGenome,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = true,
                        Error = null
                    });
                }
                catch (Exception ex)
                {
                    stopwatch.Stop();
                    results.Add(new ExperimentResult
                    {
                        ParameterName = $"Ablated: {component}",
                        ParameterValue = double.MinValue,
                        BestFitness = double.MinValue,
                        BestGenome = null,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = false,
                        Error = ex.Message
                    });
                }
            }

            return results.AsReadOnly();
        }

        /// <summary>
        /// Runs multiple trials of the same configuration for statistical analysis.
        /// </summary>
        public async Task<StatisticalAnalysisResult> RunStatisticalAnalysisAsync(
            Func<EvolutionConfig, Task<GeoGenome>> evolutionRunner,
            int trialCount = 10,
            CancellationToken cancellationToken = default)
        {
            var fitnesses = new List<double>();
            var times = new List<TimeSpan>();
            var genomes = new List<GeoGenome>();
            var errors = new List<string>();

            for (int i = 0; i < trialCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var stopwatch = Stopwatch.StartNew();
                try
                {
                    var genome = await evolutionRunner(_baseConfig);
                    stopwatch.Stop();

                    fitnesses.Add(genome.Fitness);
                    times.Add(stopwatch.Elapsed);
                    genomes.Add(genome);
                }
                catch (Exception ex)
                {
                    stopwatch.Stop();
                    errors.Add(ex.Message);
                    times.Add(stopwatch.Elapsed);
                }
            }

            return new StatisticalAnalysisResult
            {
                TrialCount = trialCount,
                SuccessfulTrials = fitnesses.Count,
                FailedTrials = errors.Count,
                MeanFitness = fitnesses.Count > 0 ? fitnesses.Average() : double.MinValue,
                MedianFitness = fitnesses.Count > 0 ? fitnesses.Median() : double.MinValue,
                StdDevFitness = fitnesses.Count > 1 ? fitnesses.StandardDeviation() : 0,
                MinFitness = fitnesses.Count > 0 ? fitnesses.Min() : double.MinValue,
                MaxFitness = fitnesses.Count > 0 ? fitnesses.Max() : double.MinValue,
                MeanTime = times.Count > 0 ? TimeSpan.FromMilliseconds(times.Average(t => t.TotalMilliseconds)) : TimeSpan.Zero,
                BestGenome = fitnesses.Count > 0 ? genomes[fitnesses.IndexOf(fitnesses.Max())] : null,
                Errors = errors.AsReadOnly(),
                FitnessValues = fitnesses.AsReadOnly()
            };
        }

        /// <summary>
        /// Runs a grid search over multiple parameters.
        /// </summary>
        public async Task<IReadOnlyList<ExperimentResult>> RunGridSearchAsync(
            IReadOnlyDictionary<string, IReadOnlyList<double>> parameterGrid,
            Func<EvolutionConfig, Task<GeoGenome>> evolutionRunner,
            CancellationToken cancellationToken = default)
        {
            var parameterNames = parameterGrid.Keys.ToList();
            var results = new List<ExperimentResult>();

            var combinations = GenerateCombinations(parameterGrid);

            foreach (var combination in combinations)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var config = _baseConfig;
                foreach (var kvp in combination)
                {
                    config = ApplyParameter(config, kvp.Key, kvp.Value);
                }

                var paramDescription = string.Join(", ", combination.Select(kvp => $"{kvp.Key}={kvp.Value}"));
                var stopwatch = Stopwatch.StartNew();

                try
                {
                    var bestGenome = await evolutionRunner(config);
                    stopwatch.Stop();

                    results.Add(new ExperimentResult
                    {
                        ParameterName = paramDescription,
                        ParameterValue = bestGenome.Fitness,
                        BestFitness = bestGenome.Fitness,
                        BestGenome = bestGenome,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = true,
                        Error = null
                    });
                }
                catch (Exception ex)
                {
                    stopwatch.Stop();
                    results.Add(new ExperimentResult
                    {
                        ParameterName = paramDescription,
                        ParameterValue = double.MinValue,
                        BestFitness = double.MinValue,
                        BestGenome = null,
                        ExecutionTime = stopwatch.Elapsed,
                        IsSuccess = false,
                        Error = ex.Message
                    });
                }
            }

            return results.OrderByDescending(r => r.BestFitness).ToList().AsReadOnly();
        }

        private EvolutionConfig ApplyParameter(EvolutionConfig config, string parameterName, double value)
        {
            var clone = config.Clone();
            switch (parameterName.ToLowerInvariant())
            {
                case "populationsize":
                    clone.PopulationSize = (int)value;
                    break;
                case "mutationrate":
                    clone.MutationRate = value;
                    break;
                case "crossoverrate":
                    clone.CrossoverRate = value;
                    break;
                case "maxgenerations":
                    clone.MaxGenerations = (int)value;
                    break;
                case "elitismcount":
                    clone.ElitismCount = (int)value;
                    break;
                case "speciationthreshold":
                    clone.SpeciesCompatibilityThreshold = value;
                    break;
            }
            return clone;
        }

        private EvolutionConfig RemoveComponent(EvolutionConfig config, string componentName)
        {
            var clone = config.Clone();
            switch (componentName.ToLowerInvariant())
            {
                case "speciation":
                    clone.SpeciationMethod = SpeciationMethod.None;
                    break;
                case "crossover":
                    clone.CrossoverRate = 0;
                    break;
            }
            return clone;
        }

        private IReadOnlyList<IReadOnlyDictionary<string, double>> GenerateCombinations(
            IReadOnlyDictionary<string, IReadOnlyList<double>> parameterGrid)
        {
            var keys = parameterGrid.Keys.ToList();
            var combinations = new List<IReadOnlyDictionary<string, double>>();

            if (keys.Count == 0)
            {
                return new List<IReadOnlyDictionary<string, double>> { new Dictionary<string, double>() };
            }

            var current = new Dictionary<string, double>();
            GenerateCombinationsRecursive(parameterGrid, keys, 0, current, combinations);

            return combinations;
        }

        private void GenerateCombinationsRecursive(
            IReadOnlyDictionary<string, IReadOnlyList<double>> parameterGrid,
            List<string> keys,
            int index,
            Dictionary<string, double> current,
            List<IReadOnlyDictionary<string, double>> results)
        {
            if (index == keys.Count)
            {
                results.Add(new Dictionary<string, double>(current));
                return;
            }

            string key = keys[index];
            foreach (double value in parameterGrid[key])
            {
                current[key] = value;
                GenerateCombinationsRecursive(parameterGrid, keys, index + 1, current, results);
            }
        }
    }

    /// <summary>
    /// Result of an evolution experiment.
    /// </summary>
    public sealed class ExperimentResult
    {
        /// <summary>Parameter name tested.</summary>
        public string ParameterName { get; init; } = string.Empty;
        /// <summary>Parameter value used.</summary>
        public double ParameterValue { get; init; }
        /// <summary>Best fitness achieved.</summary>
        public double BestFitness { get; init; }
        /// <summary>Best genome found.</summary>
        public GeoGenome? BestGenome { get; init; }
        /// <summary>Execution time.</summary>
        public TimeSpan ExecutionTime { get; init; }
        /// <summary>Whether the experiment succeeded.</summary>
        public bool IsSuccess { get; init; }
        /// <summary>Error message if failed.</summary>
        public string? Error { get; init; }
    }

    /// <summary>
    /// Result of statistical analysis over multiple trials.
    /// </summary>
    public sealed class StatisticalAnalysisResult
    {
        /// <summary>Total number of trials.</summary>
        public int TrialCount { get; init; }
        /// <summary>Number of successful trials.</summary>
        public int SuccessfulTrials { get; init; }
        /// <summary>Number of failed trials.</summary>
        public int FailedTrials { get; init; }
        /// <summary>Mean fitness across trials.</summary>
        public double MeanFitness { get; init; }
        /// <summary>Median fitness across trials.</summary>
        public double MedianFitness { get; init; }
        /// <summary>Standard deviation of fitness.</summary>
        public double StdDevFitness { get; init; }
        /// <summary>Minimum fitness.</summary>
        public double MinFitness { get; init; }
        /// <summary>Maximum fitness.</summary>
        public double MaxFitness { get; init; }
        /// <summary>Mean execution time.</summary>
        public TimeSpan MeanTime { get; init; }
        /// <summary>Best genome across trials.</summary>
        public GeoGenome? BestGenome { get; init; }
        /// <summary>Error messages.</summary>
        public IReadOnlyList<string> Errors { get; init; } = Array.Empty<string>();
        /// <summary>All fitness values.</summary>
        public IReadOnlyList<double> FitnessValues { get; init; } = Array.Empty<double>();
    }

    #endregion
    // =========================================================================

    // =========================================================================
    #region Evolution Convergence Criteria

    /// <summary>
    /// Provides pluggable convergence criteria for stopping evolution based on
    /// fitness plateaus, diversity thresholds, or custom predicates.
    /// </summary>
    public sealed class ConvergenceCriteria
    {
        private readonly List<Func<GenomePopulation, IReadOnlyList<SpeciesInfo>, int, bool>> _criteria;
        private readonly List<string> _criterionNames;
        private readonly Queue<double> _fitnessHistory;
        private readonly int _historyWindowSize;

        /// <summary>
        /// Initializes a new instance of the ConvergenceCriteria class.
        /// </summary>
        /// <param name="historyWindowSize">Number of generations to track for plateau detection.</param>
        public ConvergenceCriteria(int historyWindowSize = 50)
        {
            _historyWindowSize = historyWindowSize;
            _fitnessHistory = new Queue<double>(historyWindowSize);
            _criteria = new List<Func<GenomePopulation, IReadOnlyList<SpeciesInfo>, int, bool>>();
            _criterionNames = new List<string>();
        }

        /// <summary>
        /// Adds a fitness plateau criterion. Stops when best fitness hasn't improved
        /// for the specified number of generations.
        /// </summary>
        /// <param name="generationsWithoutImprovement">Number of stagnant generations.</param>
        /// <param name="tolerance">Minimum improvement to consider as progress.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddFitnessPlateau(int generationsWithoutImprovement, double tolerance = 1e-8)
        {
            _criteria.Add((pop, species, gen) =>
            {
                if (_fitnessHistory.Count < generationsWithoutImprovement)
                    return false;

                var recent = _fitnessHistory.TakeLast(generationsWithoutImprovement).ToList();
                double improvement = recent[^1] - recent[0];
                return improvement <= tolerance;
            });
            _criterionNames.Add($"FitnessPlateau(gens={generationsWithoutImprovement}, tol={tolerance})");
            return this;
        }

        /// <summary>
        /// Adds a maximum generation limit criterion.
        /// </summary>
        /// <param name="maxGenerations">Maximum number of generations.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddMaxGenerations(int maxGenerations)
        {
            _criteria.Add((pop, species, gen) => gen >= maxGenerations);
            _criterionNames.Add($"MaxGenerations({maxGenerations})");
            return this;
        }

        /// <summary>
        /// Adds a minimum diversity criterion. Stops when topology diversity drops below threshold.
        /// </summary>
        /// <param name="minimumDiversityRatio">Minimum fraction of unique topologies.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddMinimumDiversity(double minimumDiversityRatio)
        {
            _criteria.Add((pop, species, gen) =>
            {
                if (pop.Genomes.Length == 0)
                    return true;
                var uniqueHashes = pop.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count();
                double diversity = (double)uniqueHashes / pop.Genomes.Length;
                return diversity < minimumDiversityRatio;
            });
            _criterionNames.Add($"MinimumDiversity({minimumDiversityRatio})");
            return this;
        }

        /// <summary>
        /// Adds a target fitness criterion. Stops when any genome reaches the target.
        /// </summary>
        /// <param name="targetFitness">Target fitness value.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddTargetFitness(double targetFitness)
        {
            _criteria.Add((pop, species, gen) =>
                pop.Genomes.Any(g => g.Fitness >= targetFitness));
            _criterionNames.Add($"TargetFitness({targetFitness})");
            return this;
        }

        /// <summary>
        /// Adds a species stagnation criterion. Stops when all species are stagnant.
        /// </summary>
        /// <param name="stagnantGenerations">Number of generations without improvement per species.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddSpeciesStagnation(int stagnantGenerations)
        {
            _criteria.Add((pop, species, gen) =>
            {
                if (species.Count == 0)
                    return false;
                return species.All(s => s.GenerationsWithoutImprovement >= stagnantGenerations);
            });
            _criterionNames.Add($"SpeciesStagnation({stagnantGenerations})");
            return this;
        }

        /// <summary>
        /// Adds a custom convergence criterion.
        /// </summary>
        /// <param name="name">Name of the criterion.</param>
        /// <param name="predicate">Predicate that returns true when evolution should stop.</param>
        /// <returns>This instance for chaining.</returns>
        public ConvergenceCriteria AddCustom(string name, Func<GenomePopulation, IReadOnlyList<SpeciesInfo>, int, bool> predicate)
        {
            _criteria.Add(predicate);
            _criterionNames.Add(name);
            return this;
        }

        /// <summary>
        /// Evaluates all convergence criteria against the current state.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <param name="species">Current species.</param>
        /// <param name="generation">Current generation.</param>
        /// <returns>Convergence result with details.</returns>
        public ConvergenceResult Evaluate(GenomePopulation population, IReadOnlyList<SpeciesInfo> species, int generation)
        {
            if (population.Genomes.Length > 0)
            {
                double bestFitness = population.Genomes.Max(g => g.Fitness);
                _fitnessHistory.Enqueue(bestFitness);

                while (_fitnessHistory.Count > _historyWindowSize)
                    _fitnessHistory.Dequeue();
            }

            var triggeredCriteria = new List<string>();

            for (int i = 0; i < _criteria.Count; i++)
            {
                if (_criteria[i](population, species, generation))
                {
                    triggeredCriteria.Add(_criterionNames[i]);
                }
            }

            bool hasConverged = triggeredCriteria.Count > 0;

            double bestFitnessValue = population.Genomes.Length > 0
                ? population.Genomes.Max(g => g.Fitness)
                : double.MinValue;

            double diversityRatio = population.Genomes.Length > 0
                ? (double)population.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count() / population.Genomes.Length
                : 0;

            return new ConvergenceResult
            {
                HasConverged = hasConverged,
                TriggeredCriteria = triggeredCriteria.AsReadOnly(),
                Generation = generation,
                BestFitness = bestFitnessValue,
                TopologyDiversity = diversityRatio,
                SpeciesCount = species.Count,
                FitnessHistorySize = _fitnessHistory.Count
            };
        }

        /// <summary>
        /// Resets the internal state (fitness history).
        /// </summary>
        public void Reset()
        {
            _fitnessHistory.Clear();
        }

        /// <summary>
        /// Gets the names of all registered criteria.
        /// </summary>
        public IReadOnlyList<string> GetCriterionNames()
        {
            return _criterionNames.AsReadOnly();
        }
    }

    /// <summary>
    /// Result of convergence evaluation.
    /// </summary>
    public sealed class ConvergenceResult
    {
        /// <summary>Whether convergence was detected.</summary>
        public bool HasConverged { get; init; }
        /// <summary>Names of triggered criteria.</summary>
        public IReadOnlyList<string> TriggeredCriteria { get; init; } = Array.Empty<string>();
        /// <summary>Current generation.</summary>
        public int Generation { get; init; }
        /// <summary>Current best fitness.</summary>
        public double BestFitness { get; init; }
        /// <summary>Current topology diversity ratio.</summary>
        public double TopologyDiversity { get; init; }
        /// <summary>Number of species.</summary>
        public int SpeciesCount { get; init; }
        /// <summary>Size of fitness history buffer.</summary>
        public int FitnessHistorySize { get; init; }
    }

    #endregion
    // =========================================================================
}
