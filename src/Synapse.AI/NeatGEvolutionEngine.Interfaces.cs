// =============================================================================
// NeatGEvolutionEngine.Interfaces.cs — NEAT-G partial module
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

}
