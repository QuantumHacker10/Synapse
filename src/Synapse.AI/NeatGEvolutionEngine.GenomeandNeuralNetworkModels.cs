// =============================================================================
// NeatGEvolutionEngine.GenomeandNeuralNetworkModels.cs — NEAT-G partial module
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

}
