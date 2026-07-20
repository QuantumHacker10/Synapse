// =============================================================================
// NeatGEvolutionEngine.NetworkInferenceEngine.cs — NEAT-G partial module
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
    /// High-performance neural network inference engine for evaluating genome outputs.
    /// Supports forward pass, batch inference, and caching for efficient repeated evaluations.
    /// Implements multiple evaluation strategies including single-pass, recurrent, and
    /// Monte Carlo dropout for uncertainty estimation.
    /// </summary>
    public sealed class NetworkInferenceEngine
    {
        private readonly EvolutionConfig _config;
        private readonly Dictionary<long, double> _activationCache;
        private readonly Dictionary<long, int> _topologicalOrder;
        private bool _topologicalOrderDirty;

        /// <summary>
        /// Initializes a new instance of the NetworkInferenceEngine class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public NetworkInferenceEngine(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _activationCache = new Dictionary<long, double>();
            _topologicalOrder = new Dictionary<long, int>();
            _topologicalOrderDirty = true;
        }

        /// <summary>
        /// Performs a forward pass through the network defined by a genome.
        /// Computes output values for given inputs using topological ordering.
        /// </summary>
        /// <param name="genome">The genome defining the network.</param>
        /// <param name="inputs">Input values.</param>
        /// <returns>Output values from the network.</returns>
        public double[] ForwardPass(GeoGenome genome, ImmutableArray<double> inputs)
        {
            if (genome == null || genome.Neurons.Count == 0)
                return Array.Empty<double>();

            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            if (activeNeurons.Count == 0)
                return Array.Empty<double>();

            ComputeTopologicalOrderIfNeeded(genome, activeNeurons, activeSynapses);

            _activationCache.Clear();

            var inputNeurons = activeNeurons
                .Where(n => n.LayerIndex == 0)
                .OrderBy(n => n.PositionInLayer)
                .ToList();

            for (int i = 0; i < inputNeurons.Count; i++)
            {
                double val = i < inputs.Length ? inputs[i] : 0;
                _activationCache[inputNeurons[i].InnovationNumber] = val;
            }

            var synapseLookup = activeSynapses
                .GroupBy(s => s.TargetNeuronId)
                .ToDictionary(g => g.Key, g => g.ToList());

            var sortedLayers = activeNeurons
                .GroupBy(n => n.LayerIndex)
                .OrderBy(g => g.Key)
                .ToList();

            foreach (var layer in sortedLayers)
            {
                if (layer.Key == 0)
                    continue;

                foreach (var neuron in layer)
                {
                    double weightedSum = neuron.Bias;

                    if (synapseLookup.TryGetValue(neuron.InnovationNumber, out var inputsynapses))
                    {
                        foreach (var synapse in inputsynapses)
                        {
                            if (_activationCache.TryGetValue(synapse.SourceNeuronId, out double srcVal))
                            {
                                weightedSum += synapse.Weight * srcVal;
                            }
                        }
                    }

                    _activationCache[neuron.InnovationNumber] = neuron.Activate(weightedSum);
                }
            }

            var outputNeurons = activeNeurons
                .Where(n => n.LayerIndex == sortedLayers.Last().Key)
                .OrderBy(n => n.PositionInLayer)
                .ToList();

            var output = new double[outputNeurons.Count];
            for (int i = 0; i < outputNeurons.Count; i++)
            {
                if (_activationCache.TryGetValue(outputNeurons[i].InnovationNumber, out double val))
                    output[i] = val;
            }

            return output;
        }

        /// <summary>
        /// Performs batch forward pass for multiple input sets.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="batchInputs">Batch of input arrays.</param>
        /// <returns>Batch of output arrays.</returns>
        public IReadOnlyList<double[]> BatchForwardPass(GeoGenome genome, IReadOnlyList<ImmutableArray<double>> batchInputs)
        {
            var results = new List<double[]>(batchInputs.Count);
            foreach (var inputs in batchInputs)
            {
                results.Add(ForwardPass(genome, inputs));
            }
            return results;
        }

        /// <summary>
        /// Performs Monte Carlo forward pass with dropout for uncertainty estimation.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="inputs">Input values.</param>
        /// <param name="dropoutRate">Dropout rate (probability of dropping a neuron).</param>
        /// <param name="samples">Number of Monte Carlo samples.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>Mean output and standard deviation for each output neuron.</returns>
        public (double[] Mean, double[] StdDev) MonteCarloForwardPass(
            GeoGenome genome,
            ImmutableArray<double> inputs,
            double dropoutRate,
            int samples,
            Random rng)
        {
            var allOutputs = new List<double[]>();

            for (int s = 0; s < samples; s++)
            {
                var sampledGenome = ApplyDropout(genome, dropoutRate, rng);
                var output = ForwardPass(sampledGenome, inputs);
                allOutputs.Add(output);
            }

            if (allOutputs.Count == 0)
                return (Array.Empty<double>(), Array.Empty<double>());

            int outputSize = allOutputs[0].Length;
            var mean = new double[outputSize];
            var stdDev = new double[outputSize];

            for (int i = 0; i < outputSize; i++)
            {
                double sum = allOutputs.Average(o => i < o.Length ? o[i] : 0);
                mean[i] = sum;

                double variance = allOutputs.Average(o =>
                {
                    double val = i < o.Length ? o[i] : 0;
                    return (val - sum) * (val - sum);
                });
                stdDev[i] = Math.Sqrt(variance);
            }

            return (mean, stdDev);
        }

        /// <summary>
        /// Computes Jacobian matrix of outputs with respect to inputs.
        /// Useful for sensitivity analysis.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="inputs">Input values.</param>
        /// <param name="epsilon">Perturbation size for finite differences.</param>
        /// <returns>Jacobian matrix [output_size x input_size].</returns>
        public double[,] ComputeJacobian(GeoGenome genome, ImmutableArray<double> inputs, double epsilon = 1e-5)
        {
            var baseOutput = ForwardPass(genome, inputs);
            int outputSize = baseOutput.Length;
            int inputSize = inputs.Length;

            var jacobian = new double[outputSize, inputSize];

            for (int j = 0; j < inputSize; j++)
            {
                var perturbedInputs = inputs.ToArray();
                perturbedInputs[j] += epsilon;
                var perturbedOutput = ForwardPass(genome, perturbedInputs.ToImmutableArray());

                for (int i = 0; i < outputSize; i++)
                {
                    double baseVal = i < baseOutput.Length ? baseOutput[i] : 0;
                    double pertVal = i < perturbedOutput.Length ? perturbedOutput[i] : 0;
                    jacobian[i, j] = (pertVal - baseVal) / epsilon;
                }
            }

            return jacobian;
        }

        /// <summary>
        /// Computes the total sensitivity of each input to the output.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="inputs">Input values.</param>
        /// <returns>Sensitivity scores for each input.</returns>
        public double[] ComputeInputSensitivity(GeoGenome genome, ImmutableArray<double> inputs)
        {
            var jacobian = ComputeJacobian(genome, inputs);
            int inputSize = inputs.Length;
            int outputSize = jacobian.GetLength(0);

            var sensitivity = new double[inputSize];
            for (int j = 0; j < inputSize; j++)
            {
                double sumSq = 0;
                for (int i = 0; i < outputSize; i++)
                {
                    sumSq += jacobian[i, j] * jacobian[i, j];
                }
                sensitivity[j] = Math.Sqrt(sumSq);
            }

            double maxSens = sensitivity.Max();
            if (maxSens > 1e-10)
            {
                for (int j = 0; j < inputSize; j++)
                    sensitivity[j] /= maxSens;
            }

            return sensitivity;
        }

        /// <summary>
        /// Estimates the computational cost of evaluating a genome.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Estimated FLOPs for a single forward pass.</returns>
        public long EstimateComputeCost(GeoGenome genome)
        {
            long flops = 0;

            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            flops += activeSynapses.Count * 2;

            foreach (var neuron in activeNeurons)
            {
                flops += GetActivationCost(neuron.Activation);
            }

            return flops;
        }

        /// <summary>
        /// Estimates memory usage for evaluating a genome.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Estimated memory usage in bytes.</returns>
        public long EstimateMemoryUsage(GeoGenome genome)
        {
            long bytes = 0;

            bytes += genome.ActiveNeuronCount * sizeof(double);
            bytes += genome.ActiveSynapseCount * (sizeof(double) + sizeof(long) * 2);
            bytes += genome.TotalNeuronCount * 64;
            bytes += genome.TotalSynapseCount * 80;

            bytes += 1024;

            return bytes;
        }

        /// <summary>
        /// Profiles the inference performance of a genome.
        /// </summary>
        /// <param name="genome">The genome to profile.</param>
        /// <param name="inputSize">Input vector size.</param>
        /// <param name="iterations">Number of iterations for timing.</param>
        /// <returns>Profiling results.</returns>
        public InferenceProfile ProfileInference(GeoGenome genome, int inputSize, int iterations = 1000)
        {
            var rng = new Random(42);
            var inputs = ImmutableArray.CreateRange(
                Enumerable.Range(0, inputSize).Select(_ => rng.NextDouble() * 2 - 1));

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                ForwardPass(genome, inputs);
            }
            sw.Stop();

            long flops = EstimateComputeCost(genome);
            long memory = EstimateMemoryUsage(genome);

            double avgTimeMs = sw.Elapsed.TotalMilliseconds / iterations;
            double flopsPerSecond = avgTimeMs > 0 ? flops / (avgTimeMs / 1000.0) : 0;

            return new InferenceProfile
            {
                AverageInferenceTimeMs = avgTimeMs,
                TotalTimeMs = sw.Elapsed.TotalMilliseconds,
                Iterations = iterations,
                EstimatedFLOPs = flops,
                EstimatedMemoryBytes = memory,
                FLOPS = flopsPerSecond,
                NeuronCount = genome.ActiveNeuronCount,
                SynapseCount = genome.ActiveSynapseCount,
                LayerCount = genome.MaxLayerDepth + 1
            };
        }

        private GeoGenome ApplyDropout(GeoGenome genome, double dropoutRate, Random rng)
        {
            var dropped = genome.Clone();
            foreach (var neuron in dropped.Neurons.Where(n => n.IsActive && n.LayerIndex > 0))
            {
                if (rng.NextDouble() < dropoutRate)
                {
                    neuron.IsActive = false;
                }
            }
            return dropped;
        }

        private void ComputeTopologicalOrderIfNeeded(GeoGenome genome, List<GeoNeuron> activeNeurons, List<GeoSynapse> activeSynapses)
        {
            if (!_topologicalOrderDirty && _topologicalOrder.Count == activeNeurons.Count)
                return;

            _topologicalOrder.Clear();
            int order = 0;

            var layers = activeNeurons
                .GroupBy(n => n.LayerIndex)
                .OrderBy(g => g.Key);

            foreach (var layer in layers)
            {
                foreach (var neuron in layer)
                {
                    _topologicalOrder[neuron.InnovationNumber] = order++;
                }
            }

            _topologicalOrderDirty = false;
        }

        private long GetActivationCost(ActivationFunction activation)
        {
            return activation switch
            {
                ActivationFunction.Tanh => 8,
                ActivationFunction.Sigmoid => 6,
                ActivationFunction.ReLU => 2,
                ActivationFunction.LeakyReLU => 3,
                ActivationFunction.GELU => 15,
                ActivationFunction.Swish => 8,
                ActivationFunction.Sinusoidal => 10,
                ActivationFunction.Linear => 1,
                ActivationFunction.Abs => 2,
                ActivationFunction.Step => 2,
                ActivationFunction.Softplus => 6,
                ActivationFunction.Mish => 20,
                ActivationFunction.Exponential => 6,
                _ => 2
            };
        }
    }

    /// <summary>
    /// Inference profiling results.
    /// </summary>
    public sealed class InferenceProfile
    {
        /// <summary>Average inference time per forward pass.</summary>
        public double AverageInferenceTimeMs { get; init; }

        /// <summary>Total profiling time.</summary>
        public double TotalTimeMs { get; init; }

        /// <summary>Number of iterations profiled.</summary>
        public int Iterations { get; init; }

        /// <summary>Estimated floating point operations.</summary>
        public long EstimatedFLOPs { get; init; }

        /// <summary>Estimated memory usage in bytes.</summary>
        public long EstimatedMemoryBytes { get; init; }

        /// <summary>Estimated FLOPS throughput.</summary>
        public double FLOPS { get; init; }

        /// <summary>Number of active neurons.</summary>
        public int NeuronCount { get; init; }

        /// <summary>Number of active synapses.</summary>
        public int SynapseCount { get; init; }

        /// <summary>Number of layers.</summary>
        public int LayerCount { get; init; }

        /// <inheritdoc/>
        public override string ToString() =>
            $"InferenceProfile(Time={AverageInferenceTimeMs:F3}ms, FLOPs={EstimatedFLOPs:N0}, " +
            $"Memory={EstimatedMemoryBytes:N0}B, Neurons={NeuronCount}, Synapses={SynapseCount})";
    }

}
