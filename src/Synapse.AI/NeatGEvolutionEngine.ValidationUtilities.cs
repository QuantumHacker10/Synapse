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

}
