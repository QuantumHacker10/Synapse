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

}
