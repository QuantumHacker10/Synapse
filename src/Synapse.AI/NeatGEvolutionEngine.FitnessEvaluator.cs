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
    /// Multi-objective fitness evaluator for NEAT-G genomes.
    /// Evaluates visual fidelity, performance, memory efficiency, structural complexity,
    /// perceptual quality (JND distance, SSIM approximation), and SDF error metrics.
    /// Provides comprehensive fitness scoring for neural architecture optimization.
    /// </summary>
    public sealed class FitnessEvaluator : IFitnessEvaluator
    {
        private readonly EvaluationContext _context;
        private readonly ImmutableDictionary<FitnessComponent, double> _componentWeights;

        /// <summary>
        /// Initializes a new instance of the FitnessEvaluator class.
        /// </summary>
        /// <param name="context">The evaluation context with scene data and parameters.</param>
        public FitnessEvaluator(EvaluationContext context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _componentWeights = context.ComponentWeights.Count > 0
                ? context.ComponentWeights
                : EvaluationContext.CreateDefault().ComponentWeights;
        }

        /// <inheritdoc/>
        public async Task<GeoGenome> EvaluateAsync(GeoGenome genome, EvaluationContext context, CancellationToken ct)
        {
            if (genome == null)
                throw new ArgumentNullException(nameof(genome));
            if (genome.IsFitnessValid)
                return genome;

            ct.ThrowIfCancellationRequested();

            var components = new Dictionary<FitnessComponent, double>();

            double visualFidelity = await Task.Run(() => ComputeVisualFidelity(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.VisualFidelity] = visualFidelity;

            double performance = await Task.Run(() => ComputePerformanceScore(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.Performance] = performance;

            double memoryEfficiency = await Task.Run(() => ComputeMemoryEfficiency(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.MemoryEfficiency] = memoryEfficiency;

            double structuralComplexity = await Task.Run(() => ComputeStructuralComplexityScore(genome), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.StructuralComplexity] = structuralComplexity;

            double perceptualQuality = await Task.Run(() => ComputePerceptualQuality(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.PerceptualQuality] = perceptualQuality;

            double sdfError = await Task.Run(() => ComputeSDFError(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.SDFError] = sdfError;

            double irradianceError = await Task.Run(() => ComputeIrradianceError(genome, context), ct)
                .ConfigureAwait(false);
            components[FitnessComponent.IrradianceError] = irradianceError;

            double topoNovelty = ComputeTopologicalNovelty(genome);
            components[FitnessComponent.TopologicalNovelty] = topoNovelty;

            double generalization = ComputeGeneralizationScore(genome, context);
            components[FitnessComponent.Generalization] = generalization;

            double totalFitness = 0;
            double totalWeight = 0;
            foreach (var kvp in components)
            {
                if (_componentWeights.TryGetValue(kvp.Key, out double weight))
                {
                    totalFitness += kvp.Value * weight;
                    totalWeight += weight;
                }
            }

            if (totalWeight > 0)
                totalFitness /= totalWeight;

            genome.FitnessComponents = components.ToImmutableDictionary();
            genome.Fitness = totalFitness;
            genome.RawFitness = totalFitness;
            genome.EvaluationCount++;

            return genome;
        }

        /// <inheritdoc/>
        public ImmutableDictionary<FitnessComponent, double> GetComponentWeights()
        {
            return _componentWeights;
        }

        /// <summary>
        /// Computes visual fidelity score by comparing genome output to reference data.
        /// Uses pixel-wise comparison with perceptual weighting.
        /// </summary>
        private double ComputeVisualFidelity(GeoGenome genome, EvaluationContext context)
        {
            if (context.ReferenceImageData == null || context.ExpectedOutputSize == 0)
                return 0.5;

            double[] output = ForwardPass(genome, context.InputData);
            if (output.Length == 0)
                return 0;

            double mse = 0;
            double mae = 0;
            int compareLength = Math.Min(output.Length, context.ExpectedOutputSize);

            for (int i = 0; i < compareLength; i++)
            {
                double expected = i < context.TargetOutput.Length ? context.TargetOutput[i] : 0;
                double diff = output[i] - expected;
                mse += diff * diff;
                mae += Math.Abs(diff);
            }

            mse /= compareLength;
            mae /= compareLength;

            double mseScore = Math.Exp(-mse * 10.0);
            double maeScore = Math.Exp(-mae * 5.0);

            return 0.6 * mseScore + 0.4 * maeScore;
        }

        /// <summary>
        /// Computes performance score based on inference latency estimation.
        /// Estimates latency from genome complexity metrics.
        /// </summary>
        private double ComputePerformanceScore(GeoGenome genome, EvaluationContext context)
        {
            int activeNeurons = genome.ActiveNeuronCount;
            int activeSynapses = genome.ActiveSynapseCount;
            int depth = genome.MaxLayerDepth;

            double estimatedOps = activeSynapses * 2.0 + activeNeurons;
            double depthFactor = Math.Log2(Math.Max(2, depth));
            double parallelismFactor = activeNeurons > 0
                ? (double)activeNeurons / depth
                : 1.0;

            double estimatedLatencyMs = estimatedOps * 0.001 * depthFactor / Math.Max(1, parallelismFactor * 0.1);

            double score = context.MaxLatencyMs > 0
                ? Math.Exp(-Math.Max(0, estimatedLatencyMs - context.MaxLatencyMs * 0.5) / context.MaxLatencyMs)
                : 1.0;

            return Math.Clamp(score, 0, 1);
        }

        /// <summary>
        /// Computes memory efficiency score.
        /// </summary>
        private double ComputeMemoryEfficiency(GeoGenome genome, EvaluationContext context)
        {
            long estimatedMemory = (long)(genome.ActiveNeuronCount * 64 + genome.ActiveSynapseCount * 16);

            double score = context.MaxMemoryBytes > 0
                ? Math.Exp(-Math.Max(0, estimatedMemory - context.MaxMemoryBytes * 0.3) / context.MaxMemoryBytes)
                : 1.0;

            return Math.Clamp(score, 0, 1);
        }

        /// <summary>
        /// Computes structural complexity penalty score.
        /// Penalizes overly complex genomes to encourage parsimony.
        /// </summary>
        private double ComputeStructuralComplexityScore(GeoGenome genome)
        {
            double complexity = genome.ComputeComplexity();
            double parsimonyPressure = 0.01;
            double score = Math.Exp(-parsimonyPressure * complexity);
            return Math.Clamp(score, 0, 1);
        }

        /// <summary>
        /// Computes perceptual quality using JND (Just Noticeable Difference) distance
        /// and SSIM (Structural Similarity Index) approximation.
        /// </summary>
        private double ComputePerceptualQuality(GeoGenome genome, EvaluationContext context)
        {
            if (context.TargetOutput.Length == 0)
                return 0.5;

            double[] output = ForwardPass(genome, context.InputData);
            if (output.Length == 0)
                return 0;

            int length = Math.Min(output.Length, context.TargetOutput.Length);

            double jndScore = ComputeJNDDistance(output, context.TargetOutput, length);
            double ssimScore = ComputeSSIMApproximation(output, context.TargetOutput, length);

            return 0.5 * jndScore + 0.5 * ssimScore;
        }

        /// <summary>
        /// Computes JND (Just Noticeable Difference) distance between two signal arrays.
        /// Based on the Weber-Fechner law: perception is logarithmic.
        /// </summary>
        private double ComputeJNDDistance(double[] output, ImmutableArray<double> target, int length)
        {
            double jndThreshold = 0.02;
            double aboveJND = 0;

            for (int i = 0; i < length; i++)
            {
                double diff = Math.Abs(output[i] - target[i]);
                double reference = Math.Max(Math.Abs(target[i]), 1e-6);
                double weberRatio = diff / reference;

                if (weberRatio > jndThreshold)
                {
                    aboveJND += weberRatio - jndThreshold;
                }
            }

            double avgAboveJND = length > 0 ? aboveJND / length : 0;
            return Math.Exp(-avgAboveJND * 5.0);
        }

        /// <summary>
        /// Computes SSIM approximation between two signal arrays.
        /// SSIM considers luminance, contrast, and structure.
        /// </summary>
        private double ComputeSSIMApproximation(double[] output, ImmutableArray<double> target, int length)
        {
            if (length == 0)
                return 0;

            double muX = 0, muY = 0;
            for (int i = 0; i < length; i++)
            {
                muX += output[i];
                muY += target[i];
            }
            muX /= length;
            muY /= length;

            double sigmaX2 = 0, sigmaY2 = 0, sigmaXY = 0;
            for (int i = 0; i < length; i++)
            {
                double dx = output[i] - muX;
                double dy = target[i] - muY;
                sigmaX2 += dx * dx;
                sigmaY2 += dy * dy;
                sigmaXY += dx * dy;
            }
            sigmaX2 /= length - 1;
            sigmaY2 /= length - 1;
            sigmaXY /= length - 1;

            double C1 = 0.01 * 0.01;
            double C2 = 0.03 * 0.03;

            double luminance = (2 * muX * muY + C1) / (muX * muX + muY * muY + C1);
            double contrast = (2 * Math.Sqrt(Math.Max(0, sigmaX2)) * Math.Sqrt(Math.Max(0, sigmaY2)) + C2) /
                             (sigmaX2 + sigmaY2 + C2);
            double structure = (sigmaXY + C2 / 2) /
                              (Math.Sqrt(Math.Max(0, sigmaX2)) * Math.Sqrt(Math.Max(0, sigmaY2)) + C2 / 2);

            return Math.Clamp(luminance * contrast * structure, 0, 1);
        }

        /// <summary>
        /// Computes SDF (Signed Distance Function) error metrics for geometric accuracy.
        /// </summary>
        private double ComputeSDFError(GeoGenome genome, EvaluationContext context)
        {
            if (context.TargetOutput.Length == 0)
                return 0.5;

            double[] output = ForwardPass(genome, context.InputData);
            if (output.Length == 0)
                return 0;

            int length = Math.Min(output.Length, context.TargetOutput.Length);
            double maxError = 0;
            double mseError = 0;
            double chamferDistance = 0;

            for (int i = 0; i < length; i++)
            {
                double error = Math.Abs(output[i] - context.TargetOutput[i]);
                maxError = Math.Max(maxError, error);
                mseError += error * error;
                chamferDistance += error * error;
            }

            mseError /= length;
            chamferDistance = Math.Sqrt(chamferDistance / length);

            double maxErrorScore = Math.Exp(-maxError * 5.0);
            double mseScore = Math.Exp(-mseError * 10.0);
            double chamferScore = Math.Exp(-chamferDistance * 3.0);

            return 0.4 * maxErrorScore + 0.3 * mseScore + 0.3 * chamferScore;
        }

        /// <summary>
        /// Computes mean-squared irradiance error between the genome forward pass and
        /// <see cref="EvaluationContext.TargetOutput"/>. GeoGenome does not embed L-DNN
        /// weights directly; NEAT-G evolves a proxy irradiance predictor head whose outputs
        /// are compared against reference irradiance samples.
        /// </summary>
        private double ComputeIrradianceError(GeoGenome genome, EvaluationContext context)
        {
            if (context.TargetOutput.Length == 0)
                return 0.5;

            double[] output = ForwardPass(genome, context.InputData);
            if (output.Length == 0)
                return 0;

            int length = Math.Min(output.Length, context.TargetOutput.Length);
            double mse = 0;
            for (int i = 0; i < length; i++)
            {
                double diff = output[i] - context.TargetOutput[i];
                mse += diff * diff;
            }

            mse /= length;
            return Math.Exp(-mse * 10.0);
        }

        /// <summary>
        /// Computes topological novelty based on genome structure uniqueness.
        /// </summary>
        private double ComputeTopologicalNovelty(GeoGenome genome)
        {
            long hash = genome.ComputeTopologyHash();
            double hashNorm = (double)(hash & 0x7FFFFFFF) / int.MaxValue;
            return hashNorm;
        }

        /// <summary>
        /// Computes generalization score based on genome's expected ability to handle unseen data.
        /// </summary>
        private double ComputeGeneralizationScore(GeoGenome genome, EvaluationContext context)
        {
            double parsimony = Math.Exp(-0.01 * genome.ComputeComplexity());
            double dropout = (double)genome.Neurons.Count(n => !n.IsActive) /
                           Math.Max(1, genome.Neurons.Count);

            double implicitRegularization = 0.5 * parsimony + 0.5 * dropout;
            return Math.Clamp(implicitRegularization, 0, 1);
        }

        /// <summary>
        /// Performs a forward pass through the genome's neural network.
        /// </summary>
        /// <param name="genome">The genome to evaluate.</param>
        /// <param name="input">Input values.</param>
        /// <returns>Output values from the network.</returns>
        internal static double[] ForwardPass(GeoGenome genome, ImmutableArray<double> input)
        {
            if (genome == null || genome.Neurons.Count == 0)
                return Array.Empty<double>();

            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            if (activeNeurons.Count == 0)
                return Array.Empty<double>();

            var neuronValues = new Dictionary<long, double>();

            var inputNeurons = activeNeurons
                .Where(n => n.LayerIndex == 0)
                .OrderBy(n => n.PositionInLayer)
                .ToList();

            for (int i = 0; i < inputNeurons.Count; i++)
            {
                double val = i < input.Length ? input[i] : 0;
                neuronValues[inputNeurons[i].InnovationNumber] = val;
            }

            var layers = activeNeurons
                .GroupBy(n => n.LayerIndex)
                .OrderBy(g => g.Key)
                .ToList();

            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            foreach (var layer in layers)
            {
                if (layer.Key == 0)
                    continue;

                foreach (var neuron in layer)
                {
                    double weightedSum = neuron.Bias;

                    var inputs = activeSynapses
                        .Where(s => s.TargetNeuronId == neuron.InnovationNumber)
                        .ToList();

                    foreach (var synapse in inputs)
                    {
                        if (neuronValues.TryGetValue(synapse.SourceNeuronId, out double srcVal))
                        {
                            weightedSum += synapse.Weight * srcVal;
                        }
                    }

                    neuronValues[neuron.InnovationNumber] = neuron.Activate(weightedSum);
                }
            }

            var outputNeurons = activeNeurons
                .Where(n => n.LayerIndex == genome.MaxLayerDepth || n.LayerIndex == layers.Max(l => l.Key))
                .OrderBy(n => n.PositionInLayer)
                .ToList();

            if (outputNeurons.Count == 0)
            {
                outputNeurons = layers.Last().ToList();
            }

            var output = new double[outputNeurons.Count];
            for (int i = 0; i < outputNeurons.Count; i++)
            {
                if (neuronValues.TryGetValue(outputNeurons[i].InnovationNumber, out double val))
                    output[i] = val;
            }

            return output;
        }
    }

}
