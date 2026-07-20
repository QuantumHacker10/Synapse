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

}
