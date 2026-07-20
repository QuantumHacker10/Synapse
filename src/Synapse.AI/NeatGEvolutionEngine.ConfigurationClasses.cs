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

}
