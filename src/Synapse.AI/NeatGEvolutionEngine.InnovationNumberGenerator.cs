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

}
