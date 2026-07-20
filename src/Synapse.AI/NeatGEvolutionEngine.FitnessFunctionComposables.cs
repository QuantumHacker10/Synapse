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
    /// Provides composable fitness function building blocks for creating
    /// custom multi-objective fitness evaluations.
    /// </summary>
    public static class FitnessFunctions
    {
        /// <summary>
        /// Computes mean squared error between output and target.
        /// </summary>
        /// <param name="output">Output values.</param>
        /// <param name="target">Target values.</param>
        /// <returns>MSE value (lower is better).</returns>
        public static double MeanSquaredError(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < length; i++)
            {
                double diff = output[i] - target[i];
                sum += diff * diff;
            }
            return sum / length;
        }

        /// <summary>
        /// Computes root mean squared error.
        /// </summary>
        public static double RootMeanSquaredError(double[] output, ImmutableArray<double> target)
        {
            return Math.Sqrt(MeanSquaredError(output, target));
        }

        /// <summary>
        /// Computes mean absolute error.
        /// </summary>
        public static double MeanAbsoluteError(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < length; i++)
            {
                sum += Math.Abs(output[i] - target[i]);
            }
            return sum / length;
        }

        /// <summary>
        /// Computes negative log likelihood loss.
        /// </summary>
        public static double NegativeLogLikelihood(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < length; i++)
            {
                double pred = Math.Clamp(output[i], 1e-7, 1 - 1e-7);
                double t = i < target.Length ? target[i] : 0;
                sum -= t * Math.Log(pred) + (1 - t) * Math.Log(1 - pred);
            }
            return sum / length;
        }

        /// <summary>
        /// Computes cross-entropy loss for multi-class output.
        /// </summary>
        public static double CrossEntropyLoss(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < length; i++)
            {
                double pred = Math.Max(output[i], 1e-7);
                double t = i < target.Length ? target[i] : 0;
                sum -= t * Math.Log(pred);
            }
            return sum / length;
        }

        /// <summary>
        /// Computes cosine similarity between output and target.
        /// </summary>
        public static double CosineSimilarity(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length == 0)
                return 0;

            double dotProduct = 0, normA = 0, normB = 0;
            for (int i = 0; i < length; i++)
            {
                dotProduct += output[i] * target[i];
                normA += output[i] * output[i];
                normB += target[i] * target[i];
            }

            double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
            return denominator > 1e-10 ? dotProduct / denominator : 0;
        }

        /// <summary>
        /// Computes signal-to-noise ratio of the output.
        /// </summary>
        public static double SignalToNoiseRatio(double[] output)
        {
            if (output.Length == 0)
                return 0;

            double mean = output.Average();
            double signalPower = mean * mean;
            double noisePower = output.Average(v => (v - mean) * (v - mean));

            return noisePower > 1e-10 ? 10 * Math.Log10(signalPower / noisePower) : 0;
        }

        /// <summary>
        /// Computes structural similarity (SSIM) between output and target signals.
        /// </summary>
        public static double StructuralSimilarity(double[] output, ImmutableArray<double> target)
        {
            int length = Math.Min(output.Length, target.Length);
            if (length <= 1)
                return 0;

            double muX = output.Average();
            double muY = 0;
            for (int i = 0; i < length; i++)
                muY += target[i];
            muY /= length;

            double sigmaX2 = output.Average(v => (v - muX) * (v - muX));
            double sigmaY2 = 0;
            for (int i = 0; i < length; i++)
                sigmaY2 += (target[i] - muY) * (target[i] - muY);
            sigmaY2 /= length;

            double sigmaXY = 0;
            for (int i = 0; i < length; i++)
                sigmaXY += (output[i] - muX) * (target[i] - muY);
            sigmaXY /= length;

            double C1 = 0.01 * 0.01;
            double C2 = 0.03 * 0.03;

            double numerator = (2 * muX * muY + C1) * (2 * sigmaXY + C2);
            double denominator = (muX * muX + muY * muY + C1) * (sigmaX2 + sigmaY2 + C2);

            return denominator > 0 ? numerator / denominator : 0;
        }

        /// <summary>
        /// Computes the peak signal-to-noise ratio (PSNR).
        /// </summary>
        public static double PeakSignalToNoiseRatio(double[] output, ImmutableArray<double> target, double maxSignal = 1.0)
        {
            double mse = MeanSquaredError(output, target);
            if (mse < 1e-10)
                return 100.0;
            return 20 * Math.Log10(maxSignal) - 10 * Math.Log10(mse);
        }

        /// <summary>
        /// Computes the L1 regularization penalty on genome weights.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="lambda">Regularization strength.</param>
        /// <returns>L1 penalty value.</returns>
        public static double L1Regularization(GeoGenome genome, double lambda = 0.001)
        {
            double sum = genome.Synapses
                .Where(s => s.IsActive)
                .Sum(s => Math.Abs(s.Weight));
            return lambda * sum;
        }

        /// <summary>
        /// Computes the L2 regularization penalty on genome weights.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="lambda">Regularization strength.</param>
        /// <returns>L2 penalty value.</returns>
        public static double L2Regularization(GeoGenome genome, double lambda = 0.001)
        {
            double sum = genome.Synapses
                .Where(s => s.IsActive)
                .Sum(s => s.Weight * s.Weight);
            return lambda * Math.Sqrt(sum);
        }

        /// <summary>
        /// Computes elastic net regularization combining L1 and L2.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="l1Lambda">L1 regularization strength.</param>
        /// <param name="l2Lambda">L2 regularization strength.</param>
        /// <returns>Elastic net penalty value.</returns>
        public static double ElasticNetRegularization(GeoGenome genome, double l1Lambda = 0.001, double l2Lambda = 0.001)
        {
            return L1Regularization(genome, l1Lambda) + L2Regularization(genome, l2Lambda);
        }

        /// <summary>
        /// Computes a complexity penalty based on genome structure.
        /// Encourages parsimonious solutions.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="complexityWeight">Weight for complexity penalty.</param>
        /// <returns>Complexity penalty value (higher = more complex).</returns>
        public static double ComplexityPenalty(GeoGenome genome, double complexityWeight = 0.01)
        {
            double neuronPenalty = genome.ActiveNeuronCount * 1.0;
            double synapsePenalty = genome.ActiveSynapseCount * 0.5;
            double depthPenalty = genome.MaxLayerDepth * 2.0;
            double densityPenalty = genome.ConnectionDensity * 10.0;

            return complexityWeight * (neuronPenalty + synapsePenalty + depthPenalty + densityPenalty);
        }

        /// <summary>
        /// Computes novelty score based on distance to nearest neighbors in feature space.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="archive">Archive of previously seen genomes.</param>
        /// <param name="k">Number of nearest neighbors.</param>
        /// <returns>Novelty score (higher = more novel).</returns>
        public static double NoveltyScore(GeoGenome genome, IReadOnlyList<GeoGenome> archive, int k = 5)
        {
            if (archive.Count == 0)
                return 1.0;

            var distances = archive
                .Select(other => ComputeGenomeFeatureDistance(genome, other))
                .OrderBy(d => d)
                .Take(Math.Min(k, archive.Count))
                .ToList();

            return distances.Count > 0 ? distances.Average() : 0;
        }

        /// <summary>
        /// Computes feature distance between two genomes based on structural features.
        /// </summary>
        private static double ComputeGenomeFeatureDistance(GeoGenome a, GeoGenome b)
        {
            double neuronDiff = Math.Abs(a.ActiveNeuronCount - b.ActiveNeuronCount);
            double synapseDiff = Math.Abs(a.ActiveSynapseCount - b.ActiveSynapseCount);
            double depthDiff = Math.Abs(a.MaxLayerDepth - b.MaxLayerDepth);
            double densityDiff = Math.Abs(a.ConnectionDensity - b.ConnectionDensity);

            double maxNeurons = Math.Max(a.ActiveNeuronCount, b.ActiveNeuronCount);
            double maxSynapses = Math.Max(a.ActiveSynapseCount, b.ActiveSynapseCount);
            double maxDepth = Math.Max(a.MaxLayerDepth, b.MaxLayerDepth);

            double normNeuron = maxNeurons > 0 ? neuronDiff / maxNeurons : 0;
            double normSynapse = maxSynapses > 0 ? synapseDiff / maxSynapses : 0;
            double normDepth = maxDepth > 0 ? depthDiff / maxDepth : 0;

            return Math.Sqrt(normNeuron * normNeuron + normSynapse * normSynapse +
                           normDepth * normDepth + densityDiff * densityDiff);
        }

        /// <summary>
        /// Creates a fitness function that combines multiple objectives with configurable weights.
        /// </summary>
        /// <param name="components">Component functions and their weights.</param>
        /// <returns>A combined fitness function.</returns>
        public static Func<GeoGenome, double> CreateWeightedCombine(
            params (Func<GeoGenome, double> Function, double Weight)[] components)
        {
            return genome =>
            {
                double totalWeight = components.Sum(c => c.Weight);
                if (totalWeight <= 0)
                    return 0;

                double totalFitness = 0;
                foreach (var (function, weight) in components)
                {
                    totalFitness += function(genome) * weight;
                }
                return totalFitness / totalWeight;
            };
        }

        /// <summary>
        /// Creates a penalized fitness function that subtracts penalties from a base fitness.
        /// </summary>
        /// <param name="baseFitness">Base fitness function.</param>
        /// <param name="penalties">Penalty functions and their weights.</param>
        /// <returns>A penalized fitness function.</returns>
        public static Func<GeoGenome, double> CreatePenalized(
            Func<GeoGenome, double> baseFitness,
            params (Func<GeoGenome, double> Penalty, double Weight)[] penalties)
        {
            return genome =>
            {
                double fitness = baseFitness(genome);
                foreach (var (penalty, weight) in penalties)
                {
                    fitness -= penalty(genome) * weight;
                }
                return fitness;
            };
        }

        /// <summary>
        /// Creates a fitness function that applies a threshold transformation.
        /// </summary>
        /// <param name="inner">Inner fitness function.</param>
        /// <param name="threshold">Minimum threshold value.</param>
        /// <returns>Thresholded fitness function.</returns>
        public static Func<GeoGenome, double> WithThreshold(
            Func<GeoGenome, double> inner,
            double threshold)
        {
            return genome =>
            {
                double fitness = inner(genome);
                return fitness >= threshold ? fitness : fitness - (threshold - fitness) * 10;
            };
        }
    }

}
