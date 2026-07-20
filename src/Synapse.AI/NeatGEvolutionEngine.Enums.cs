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

}
