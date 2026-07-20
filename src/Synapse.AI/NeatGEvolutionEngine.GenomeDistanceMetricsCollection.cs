// =============================================================================
// NeatGEvolutionEngine.GenomeDistanceMetricsCollection.cs — NEAT-G partial module
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
    /// Comprehensive collection of distance metrics for comparing genomes.
    /// Provides multiple distance measures for different use cases.
    /// </summary>
    public sealed class GenomeDistanceMetrics
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the GenomeDistanceMetrics class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public GenomeDistanceMetrics(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Computes the NEAT compatibility distance between two genomes.
        /// Classic NEAT distance metric based on excess, disjoint genes, and weight differences.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Compatibility distance.</returns>
        public double ComputeCompatibilityDistance(GeoGenome a, GeoGenome b)
        {
            var aSynapseInnovs = a.Synapses.Where(s => s.IsActive).Select(s => s.InnovationNumber).ToHashSet();
            var bSynapseInnovs = b.Synapses.Where(s => s.IsActive).Select(s => s.InnovationNumber).ToHashSet();

            long maxInnovA = aSynapseInnovs.Count > 0 ? aSynapseInnovs.Max() : 0;
            long maxInnovB = bSynapseInnovs.Count > 0 ? bSynapseInnovs.Max() : 0;
            long maxInnov = Math.Max(maxInnovA, maxInnovB);

            if (maxInnov == 0)
                return 0;

            int excess = 0;
            int disjoint = 0;
            double weightDiffSum = 0;
            int matchingCount = 0;

            var aMap = a.Synapses.Where(s => s.IsActive).ToDictionary(s => s.InnovationNumber, s => s.Weight);
            var bMap = b.Synapses.Where(s => s.IsActive).ToDictionary(s => s.InnovationNumber, s => s.Weight);

            for (long i = 1; i <= maxInnov; i++)
            {
                bool inA = aMap.ContainsKey(i);
                bool inB = bMap.ContainsKey(i);

                if (inA && inB)
                {
                    matchingCount++;
                    weightDiffSum += Math.Abs(aMap[i] - bMap[i]);
                }
                else if (inA || inB)
                {
                    if (i > Math.Min(maxInnovA, maxInnovB))
                        excess++;
                    else
                        disjoint++;
                }
            }

            int N = Math.Max(a.ActiveSynapseCount, b.ActiveSynapseCount);
            if (N == 0)
                return 0;

            double normalizedExcess = (double)excess / N;
            double normalizedDisjoint = (double)disjoint / N;
            double avgWeightDiff = matchingCount > 0 ? weightDiffSum / matchingCount : 0;

            return _config.CompatibilityDisjointCoefficient * (normalizedExcess + normalizedDisjoint) +
                   _config.CompatibilityWeightCoefficient * avgWeightDiff;
        }

        /// <summary>
        /// Computes the Jaccard distance between two genomes based on their synapse innovation sets.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Jaccard distance (0 = identical, 1 = completely different).</returns>
        public double ComputeJaccardDistance(GeoGenome a, GeoGenome b)
        {
            var aInnovs = a.Synapses.Where(s => s.IsActive).Select(s => s.InnovationNumber).ToHashSet();
            var bInnovs = b.Synapses.Where(s => s.IsActive).Select(s => s.InnovationNumber).ToHashSet();

            int intersection = aInnovs.Intersect(bInnovs).Count();
            int union = aInnovs.Union(bInnovs).Count();

            return union > 0 ? 1.0 - (double)intersection / union : 0;
        }

        /// <summary>
        /// Computes the Hamming distance between two genomes based on activation functions.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Normalized Hamming distance.</returns>
        public double ComputeActivationHammingDistance(GeoGenome a, GeoGenome b)
        {
            var aActivations = a.Neurons
                .Where(n => n.IsActive)
                .OrderBy(n => n.InnovationNumber)
                .Select(n => n.Activation)
                .ToList();

            var bActivations = b.Neurons
                .Where(n => n.IsActive)
                .OrderBy(n => n.InnovationNumber)
                .Select(n => n.Activation)
                .ToList();

            int maxLen = Math.Max(aActivations.Count, bActivations.Count);
            if (maxLen == 0)
                return 0;

            int differences = 0;
            for (int i = 0; i < maxLen; i++)
            {
                var aAct = i < aActivations.Count ? aActivations[i] : ActivationFunction.Linear;
                var bAct = i < bActivations.Count ? bActivations[i] : ActivationFunction.Linear;
                if (aAct != bAct)
                    differences++;
            }

            return (double)differences / maxLen;
        }

        /// <summary>
        /// Computes the weight distribution distance between two genomes.
        /// Uses the Kolmogorov-Smirnov statistic.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>KS distance (0-1).</returns>
        public double ComputeWeightDistributionDistance(GeoGenome a, GeoGenome b)
        {
            var aWeights = a.Synapses.Where(s => s.IsActive).Select(s => s.Weight).OrderBy(w => w).ToList();
            var bWeights = b.Synapses.Where(s => s.IsActive).Select(s => s.Weight).OrderBy(w => w).ToList();

            if (aWeights.Count == 0 || bWeights.Count == 0)
                return 1.0;

            double maxDiff = 0;
            int i = 0, j = 0;

            while (i < aWeights.Count && j < bWeights.Count)
            {
                double aCDF = (double)(i + 1) / aWeights.Count;
                double bCDF = (double)(j + 1) / bWeights.Count;
                double diff = Math.Abs(aCDF - bCDF);
                maxDiff = Math.Max(maxDiff, diff);

                if (aWeights[i] < bWeights[j])
                    i++;
                else if (aWeights[i] > bWeights[j])
                    j++;
                else
                {
                    i++;
                    j++;
                }
            }

            return maxDiff;
        }

        /// <summary>
        /// Computes the structural similarity index (SSIM) between two genomes.
        /// Based on the SSIM image quality metric adapted for graph structures.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Structural similarity (0-1, higher = more similar).</returns>
        public double ComputeStructuralSimilarity(GeoGenome a, GeoGenome b)
        {
            double[] featuresA = ExtractStructuralFeatures(a);
            double[] featuresB = ExtractStructuralFeatures(b);

            int length = Math.Min(featuresA.Length, featuresB.Length);
            if (length == 0)
                return 0;

            double muA = featuresA.Take(length).Average();
            double muB = featuresB.Take(length).Average();

            double sigmaA2 = featuresA.Take(length).Average(f => (f - muA) * (f - muA));
            double sigmaB2 = featuresB.Take(length).Average(f => (f - muB) * (f - muB));
            double sigmaAB = featuresA.Take(length).Zip(featuresB.Take(length),
                (a2, b2) => (a2 - muA) * (b2 - muB)).Average();

            double C1 = 0.01 * 0.01;
            double C2 = 0.03 * 0.03;

            double luminance = (2 * muA * muB + C1) / (muA * muA + muB * muB + C1);
            double contrast = (2 * Math.Sqrt(sigmaA2) * Math.Sqrt(sigmaB2) + C2) / (sigmaA2 + sigmaB2 + C2);
            double structure = (sigmaAB + C2 / 2) / (Math.Sqrt(sigmaA2) * Math.Sqrt(sigmaB2) + C2 / 2);

            return Math.Clamp(luminance * contrast * structure, 0, 1);
        }

        /// <summary>
        /// Computes an ensemble distance combining multiple metrics.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Combined distance (0-1).</returns>
        public double ComputeEnsembleDistance(GeoGenome a, GeoGenome b)
        {
            double compatDist = ComputeCompatibilityDistance(a, b);
            double jaccardDist = ComputeJaccardDistance(a, b);
            double activationDist = ComputeActivationHammingDistance(a, b);
            double weightDist = ComputeWeightDistributionDistance(a, b);
            double structuralSim = ComputeStructuralSimilarity(a, b);

            return 0.25 * Math.Min(1, compatDist) +
                   0.25 * jaccardDist +
                   0.15 * activationDist +
                   0.15 * weightDist +
                   0.20 * (1 - structuralSim);
        }

        private double[] ExtractStructuralFeatures(GeoGenome genome)
        {
            var features = new List<double>();

            features.Add(genome.ActiveNeuronCount);
            features.Add(genome.ActiveSynapseCount);
            features.Add(genome.MaxLayerDepth);
            features.Add(genome.ConnectionDensity);

            var layerSizes = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.LayerIndex)
                .Select(g => (double)g.Count())
                .ToList();

            features.Add(layerSizes.Count);
            if (layerSizes.Count > 0)
            {
                features.Add(layerSizes.Average());
                features.Add(layerSizes.Max());
                features.Add(layerSizes.Min());
                double variance = layerSizes.Average(s => (s - layerSizes.Average()) * (s - layerSizes.Average()));
                features.Add(Math.Sqrt(variance));
            }

            var weights = genome.Synapses.Where(s => s.IsActive).Select(s => s.Weight).ToList();
            if (weights.Count > 0)
            {
                features.Add(weights.Average());
                features.Add(weights.Max());
                features.Add(weights.Min());
                features.Add(weights.Sum(w => Math.Abs(w)));
                double variance = weights.Average(w => (w - weights.Average()) * (w - weights.Average()));
                features.Add(Math.Sqrt(variance));
            }

            var activations = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.Activation)
                .Select(g => (double)g.Count())
                .ToList();

            features.Add(activations.Count);
            if (activations.Count > 0)
            {
                features.Add(activations.Average());
            }

            return features.ToArray();
        }
    }

}
