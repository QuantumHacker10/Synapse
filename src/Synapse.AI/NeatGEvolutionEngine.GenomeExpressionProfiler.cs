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

}
