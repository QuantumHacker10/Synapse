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
    /// Extracts statistical and structural features from genomes for analysis,
    /// comparison, and machine learning applications.
    /// </summary>
    public sealed class GenomeFeatureExtractor
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the GenomeFeatureExtractor class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public GenomeFeatureExtractor(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Extracts a comprehensive feature vector from a genome.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Feature vector.</returns>
        public double[] ExtractFeatures(GeoGenome genome)
        {
            var features = new List<double>();

            features.Add(genome.ActiveNeuronCount);
            features.Add(genome.ActiveSynapseCount);
            features.Add(genome.MaxLayerDepth);
            features.Add(genome.ConnectionDensity);
            features.Add(genome.Complexity);
            features.Add(genome.Fitness);

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
                features.Add(layerSizes.Count > 1 ? layerSizes.StandardDeviation() : 0);
            }
            else
            {
                features.AddRange(new double[] { 0, 0, 0, 0, 0 });
            }

            var weights = genome.Synapses.Where(s => s.IsActive).Select(s => s.Weight).ToList();
            if (weights.Count > 0)
            {
                features.Add(weights.Average());
                features.Add(weights.Max());
                features.Add(weights.Min());
                features.Add(weights.Sum(w => Math.Abs(w)));
                features.Add(weights.StandardDeviation());
                features.Add(weights.Median());
            }
            else
            {
                features.AddRange(new double[] { 0, 0, 0, 0, 0, 0 });
            }

            var activations = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.Activation)
                .ToList();

            features.Add(activations.Count);
            foreach (var activation in Enum.GetValues<ActivationFunction>())
            {
                var group = activations.FirstOrDefault(g => g.Key == activation);
                features.Add(group != null ? (double)group.Count() / genome.ActiveNeuronCount : 0);
            }

            int inputNeurons = genome.Neurons.Count(n => n.IsActive && n.LayerIndex == 0);
            int outputNeurons = genome.Neurons.Count(n => n.IsActive && n.LayerIndex == genome.MaxLayerDepth);
            int hiddenNeurons = genome.ActiveNeuronCount - inputNeurons - outputNeurons;

            features.Add(inputNeurons);
            features.Add(outputNeurons);
            features.Add(hiddenNeurons);
            features.Add(hiddenNeurons > 0 ? (double)genome.ActiveSynapseCount / hiddenNeurons : 0);

            long topologyHash = genome.ComputeTopologyHash();
            features.Add((topologyHash & 0xFFFF) / (double)0xFFFF);
            features.Add(((topologyHash >> 16) & 0xFFFF) / (double)0xFFFF);

            return features.ToArray();
        }

        /// <summary>
        /// Extracts features from a population.
        /// </summary>
        /// <param name="population">The population.</param>
        /// <returns>Feature matrix [genome_index, feature_index].</returns>
        public double[,] ExtractPopulationFeatures(GenomePopulation population)
        {
            if (population.Genomes.Length == 0)
                return new double[0, 0];

            var firstFeatures = ExtractFeatures(population.Genomes[0]);
            var matrix = new double[population.Genomes.Length, firstFeatures.Length];

            for (int i = 0; i < population.Genomes.Length; i++)
            {
                var features = ExtractFeatures(population.Genomes[i]);
                for (int j = 0; j < Math.Min(features.Length, firstFeatures.Length); j++)
                {
                    matrix[i, j] = features[j];
                }
            }

            return matrix;
        }

        /// <summary>
        /// Gets feature names corresponding to the feature vector.
        /// </summary>
        public IReadOnlyList<string> GetFeatureNames()
        {
            var names = new List<string>
            {
                "ActiveNeuronCount", "ActiveSynapseCount", "MaxLayerDepth",
                "ConnectionDensity", "Complexity", "Fitness",
                "LayerCount", "AvgLayerSize", "MaxLayerSize", "MinLayerSize", "LayerSizeStdDev",
                "AvgWeight", "MaxWeight", "MinWeight", "TotalAbsWeight", "WeightStdDev", "WeightMedian",
                "ActivationDiversity"
            };

            foreach (var activation in Enum.GetValues<ActivationFunction>())
            {
                names.Add($"Fraction_{activation}");
            }

            names.AddRange(new[]
            {
                "InputNeuronCount", "OutputNeuronCount", "HiddenNeuronCount",
                "SynapsesPerHidden", "TopologyHashLow", "TopologyHashHigh"
            });

            return names.AsReadOnly();
        }

        /// <summary>
        /// Normalizes features using min-max scaling.
        /// </summary>
        /// <param name="features">Feature matrix.</param>
        /// <returns>Normalized feature matrix.</returns>
        public double[,] NormalizeFeatures(double[,] features)
        {
            int rows = features.GetLength(0);
            int cols = features.GetLength(1);
            var normalized = new double[rows, cols];

            for (int j = 0; j < cols; j++)
            {
                double min = double.MaxValue;
                double max = double.MinValue;

                for (int i = 0; i < rows; i++)
                {
                    min = Math.Min(min, features[i, j]);
                    max = Math.Max(max, features[i, j]);
                }

                double range = max - min;
                for (int i = 0; i < rows; i++)
                {
                    normalized[i, j] = range > 1e-10
                        ? (features[i, j] - min) / range
                        : 0.5;
                }
            }

            return normalized;
        }

        /// <summary>
        /// Standardizes features using z-score normalization.
        /// </summary>
        /// <param name="features">Feature matrix.</param>
        /// <returns>Standardized feature matrix.</returns>
        public double[,] StandardizeFeatures(double[,] features)
        {
            int rows = features.GetLength(0);
            int cols = features.GetLength(1);
            var standardized = new double[rows, cols];

            for (int j = 0; j < cols; j++)
            {
                double sum = 0;
                for (int i = 0; i < rows; i++)
                    sum += features[i, j];
                double mean = sum / rows;

                double sumSq = 0;
                for (int i = 0; i < rows; i++)
                    sumSq += (features[i, j] - mean) * (features[i, j] - mean);
                double stdDev = Math.Sqrt(sumSq / Math.Max(1, rows - 1));

                for (int i = 0; i < rows; i++)
                {
                    standardized[i, j] = stdDev > 1e-10
                        ? (features[i, j] - mean) / stdDev
                        : 0;
                }
            }

            return standardized;
        }

        /// <summary>
        /// Computes pairwise cosine similarity between genomes.
        /// </summary>
        /// <param name="genomes">List of genomes.</param>
        /// <returns>Cosine similarity matrix.</returns>
        public double[,] ComputeCosineSimilarityMatrix(IReadOnlyList<GeoGenome> genomes)
        {
            int n = genomes.Count;
            var matrix = new double[n, n];

            var featureVectors = genomes.Select(ExtractFeatures).ToList();

            for (int i = 0; i < n; i++)
            {
                matrix[i, i] = 1.0;
                for (int j = i + 1; j < n; j++)
                {
                    double sim = CosineSimilarity(featureVectors[i], featureVectors[j]);
                    matrix[i, j] = sim;
                    matrix[j, i] = sim;
                }
            }

            return matrix;
        }

        private double CosineSimilarity(double[] a, double[] b)
        {
            int dim = Math.Min(a.Length, b.Length);
            double dotProduct = 0, normA = 0, normB = 0;

            for (int i = 0; i < dim; i++)
            {
                dotProduct += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
            return denominator > 1e-10 ? dotProduct / denominator : 0;
        }
    }

    /// <summary>
    /// Extension methods for collections used in evolution.
    /// </summary>
    public static class EvolutionExtensions
    {
        /// <summary>
        /// Computes the standard deviation of a collection of doubles.
        /// </summary>
        public static double StandardDeviation(this IEnumerable<double> source)
        {
            var list = source.ToList();
            if (list.Count <= 1)
                return 0;

            double mean = list.Average();
            double variance = list.Average(v => (v - mean) * (v - mean));
            return Math.Sqrt(variance);
        }

        /// <summary>
        /// Computes the median of a collection of doubles.
        /// </summary>
        public static double Median(this IEnumerable<double> source)
        {
            var sorted = source.OrderBy(v => v).ToList();
            if (sorted.Count == 0)
                return 0;

            int mid = sorted.Count / 2;
            return sorted.Count % 2 == 0
                ? (sorted[mid - 1] + sorted[mid]) / 2.0
                : sorted[mid];
        }

        /// <summary>
        /// Computes the coefficient of variation.
        /// </summary>
        public static double CoefficientOfVariation(this IEnumerable<double> source)
        {
            var list = source.ToList();
            if (list.Count == 0)
                return 0;

            double mean = list.Average();
            if (Math.Abs(mean) < 1e-10)
                return 0;

            double stdDev = list.StandardDeviation();
            return stdDev / Math.Abs(mean);
        }

        /// <summary>
        /// Computes the entropy of a probability distribution.
        /// </summary>
        public static double Entropy(this IEnumerable<double> probabilities)
        {
            return probabilities
                .Where(p => p > 1e-10)
                .Sum(p => -p * Math.Log2(p));
        }

        /// <summary>
        /// Shuffles a list in place using Fisher-Yates algorithm.
        /// </summary>
        public static void Shuffle<T>(this IList<T> list, Random rng)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }

        /// <summary>
        /// Returns a random element from the collection.
        /// </summary>
        public static T RandomElement<T>(this IReadOnlyList<T> list, Random rng)
        {
            if (list.Count == 0)
                throw new InvalidOperationException("Collection is empty.");
            return list[rng.Next(list.Count)];
        }

        /// <summary>
        /// Returns N random elements from the collection without replacement.
        /// </summary>
        public static IReadOnlyList<T> RandomElements<T>(this IReadOnlyList<T> list, int count, Random rng)
        {
            if (count >= list.Count)
                return list.ToList().AsReadOnly();

            var indices = new HashSet<int>();
            var result = new List<T>();

            while (result.Count < count)
            {
                int idx = rng.Next(list.Count);
                if (indices.Add(idx))
                    result.Add(list[idx]);
            }

            return result.AsReadOnly();
        }

        /// <summary>
        /// Partitions a list into batches of a given size.
        /// </summary>
        public static IEnumerable<IReadOnlyList<T>> Batch<T>(this IReadOnlyList<T> list, int batchSize)
        {
            for (int i = 0; i < list.Count; i += batchSize)
            {
                yield return list.Skip(i).Take(batchSize).ToList().AsReadOnly();
            }
        }

        /// <summary>
        /// Returns the element that maximizes the given key selector.
        /// </summary>
        public static T MaxBy<T, TKey>(this IReadOnlyList<T> list, Func<T, TKey> keySelector)
            where TKey : IComparable<TKey>
        {
            if (list.Count == 0)
                throw new InvalidOperationException("Collection is empty.");

            T best = list[0];
            TKey bestKey = keySelector(best);

            for (int i = 1; i < list.Count; i++)
            {
                TKey key = keySelector(list[i]);
                if (key.CompareTo(bestKey) > 0)
                {
                    best = list[i];
                    bestKey = key;
                }
            }

            return best;
        }

        /// <summary>
        /// Returns the element that minimizes the given key selector.
        /// </summary>
        public static T MinBy<T, TKey>(this IReadOnlyList<T> list, Func<T, TKey> keySelector)
            where TKey : IComparable<TKey>
        {
            if (list.Count == 0)
                throw new InvalidOperationException("Collection is empty.");

            T best = list[0];
            TKey bestKey = keySelector(best);

            for (int i = 1; i < list.Count; i++)
            {
                TKey key = keySelector(list[i]);
                if (key.CompareTo(bestKey) < 0)
                {
                    best = list[i];
                    bestKey = key;
                }
            }

            return best;
        }

        /// <summary>
        /// Weighted random selection from a collection.
        /// </summary>
        public static T WeightedRandom<T>(this IReadOnlyList<T> list, IReadOnlyList<double> weights, Random rng)
        {
            if (list.Count == 0)
                throw new InvalidOperationException("Collection is empty.");

            double totalWeight = weights.Sum();
            double r = rng.NextDouble() * totalWeight;
            double cumulative = 0;

            for (int i = 0; i < list.Count; i++)
            {
                cumulative += i < weights.Count ? weights[i] : 0;
                if (cumulative >= r)
                    return list[i];
            }

            return list[^1];
        }

        /// <summary>
        /// Computes pairwise distances between all elements.
        /// </summary>
        public static double[,] PairwiseDistances<T>(this IReadOnlyList<T> list, Func<T, T, double> distanceFunc)
        {
            int n = list.Count;
            var matrix = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                matrix[i, i] = 0;
                for (int j = i + 1; j < n; j++)
                {
                    double dist = distanceFunc(list[i], list[j]);
                    matrix[i, j] = dist;
                    matrix[j, i] = dist;
                }
            }

            return matrix;
        }

        /// <summary>
        /// Softmax normalization of a collection of values.
        /// </summary>
        public static double[] Softmax(this IEnumerable<double> source)
        {
            var values = source.ToArray();
            double maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal)).ToArray();
            double sum = exp.Sum();
            return exp.Select(e => e / sum).ToArray();
        }

        /// <summary>
        /// Converts an array to an ImmutableArray.
        /// </summary>
        public static ImmutableArray<T> ToImmutableArray<T>(this T[] array)
        {
            return ImmutableArray.Create(array);
        }

        /// <summary>
        /// Converts a list to an ImmutableArray.
        /// </summary>
        public static ImmutableArray<T> ToImmutableArray<T>(this List<T> list)
        {
            return ImmutableArray.CreateRange(list);
        }
    }

}
