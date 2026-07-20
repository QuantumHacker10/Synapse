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

}
