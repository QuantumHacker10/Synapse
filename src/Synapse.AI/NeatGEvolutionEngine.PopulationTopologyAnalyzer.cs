// =============================================================================
// NeatGEvolutionEngine.PopulationTopologyAnalyzer.cs — NEAT-G partial module
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
    /// Analyzes the topology of the entire population as a meta-graph,
    /// where genomes are nodes and edges represent similarity.
    /// </summary>
    public sealed class PopulationTopologyAnalyzer
    {
        private readonly EvolutionConfig _config;

        public PopulationTopologyAnalyzer(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Builds a similarity graph of the population.
        /// </summary>
        public IReadOnlyDictionary<int, IReadOnlyList<int>> BuildSimilarityGraph(
            GenomePopulation population,
            double similarityThreshold = 0.5)
        {
            var graph = new Dictionary<int, List<int>>();
            int n = population.Genomes.Length;

            for (int i = 0; i < n; i++)
                graph[i] = new List<int>();

            var featureExtractor = new GenomeFeatureExtractor(_config);
            var featureVectors = population.Genomes
                .Select(g => featureExtractor.ExtractFeatures(g))
                .ToArray();

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double similarity = ComputeFeatureSimilarity(featureVectors[i], featureVectors[j]);
                    if (similarity >= similarityThreshold)
                    {
                        graph[i].Add(j);
                        graph[j].Add(i);
                    }
                }
            }

            return graph.ToDictionary(
                kvp => kvp.Key,
                kvp => (IReadOnlyList<int>)kvp.Value.AsReadOnly());
        }

        /// <summary>
        /// Detects clusters in the population similarity graph.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<int>> DetectClusters(
            GenomePopulation population,
            double resolution = 1.0)
        {
            var graph = BuildSimilarityGraph(population, 0.3);
            var communities = new List<List<int>>();
            var membership = new Dictionary<int, int>();
            int communityId = 0;

            foreach (var nodeId in graph.Keys)
            {
                if (!membership.ContainsKey(nodeId))
                {
                    var community = new List<int>();
                    var queue = new Queue<int>();
                    queue.Enqueue(nodeId);

                    while (queue.Count > 0)
                    {
                        int current = queue.Dequeue();
                        if (membership.ContainsKey(current))
                            continue;

                        membership[current] = communityId;
                        community.Add(current);

                        if (graph.TryGetValue(current, out var neighbors))
                        {
                            foreach (var neighbor in neighbors)
                            {
                                if (!membership.ContainsKey(neighbor))
                                    queue.Enqueue(neighbor);
                            }
                        }
                    }

                    communities.Add(community);
                    communityId++;
                }
            }

            return communities.Select(c => (IReadOnlyList<int>)c.AsReadOnly()).ToList().AsReadOnly();
        }

        /// <summary>
        /// Computes population-level network metrics.
        /// </summary>
        public PopulationNetworkMetrics ComputeMetrics(GenomePopulation population)
        {
            var graph = BuildSimilarityGraph(population, 0.3);
            int nodeCount = graph.Count;
            int edgeCount = graph.Values.Sum(n => n.Count) / 2;
            double density = nodeCount > 1 ? (2.0 * edgeCount) / (nodeCount * (nodeCount - 1)) : 0;
            var degrees = graph.Values.Select(n => n.Count).ToList();
            double averageDegree = degrees.Count > 0 ? degrees.Average() : 0;
            double clusteringCoefficient = ComputeClusteringCoefficient(graph);
            var components = FindConnectedComponents(graph);

            return new PopulationNetworkMetrics
            {
                NodeCount = nodeCount,
                EdgeCount = edgeCount,
                Density = density,
                AverageDegree = averageDegree,
                ClusteringCoefficient = clusteringCoefficient,
                ConnectedComponentCount = components.Count,
                LargestComponentRatio = components.Count > 0
                    ? (double)components.Max(c => c.Count) / nodeCount : 0,
                AveragePathLength = ComputeAveragePathLength(graph)
            };
        }

        /// <summary>
        /// Identifies outlier genomes far from the population center.
        /// </summary>
        public IReadOnlyList<int> IdentifyOutliers(GenomePopulation population, double zScoreThreshold = 2.0)
        {
            var featureExtractor = new GenomeFeatureExtractor(_config);
            var features = population.Genomes
                .Select(g => featureExtractor.ExtractFeatures(g))
                .ToArray();

            if (features.Length == 0)
                return Array.Empty<int>().AsReadOnly();

            int featureCount = features[0].Length;
            var mean = new double[featureCount];
            var stdDev = new double[featureCount];

            for (int j = 0; j < featureCount; j++)
            {
                var column = features.Select(f => f[j]).ToList();
                mean[j] = column.Average();
                stdDev[j] = column.StandardDeviation();
            }

            var outlierIds = new List<int>();

            for (int i = 0; i < features.Length; i++)
            {
                double maxZScore = 0;
                for (int j = 0; j < featureCount; j++)
                {
                    if (stdDev[j] > 1e-10)
                    {
                        double zScore = Math.Abs((features[i][j] - mean[j]) / stdDev[j]);
                        maxZScore = Math.Max(maxZScore, zScore);
                    }
                }

                if (maxZScore > zScoreThreshold)
                    outlierIds.Add(i);
            }

            return outlierIds.AsReadOnly();
        }

        private double ComputeFeatureSimilarity(double[] a, double[] b)
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

        private double ComputeClusteringCoefficient(IReadOnlyDictionary<int, IReadOnlyList<int>> graph)
        {
            double totalCoefficient = 0;
            int nodeCount = 0;

            foreach (var kvp in graph)
            {
                var neighbors = kvp.Value;
                int k = neighbors.Count;
                if (k < 2)
                    continue;

                int triangles = 0;
                for (int i = 0; i < k; i++)
                {
                    for (int j = i + 1; j < k; j++)
                    {
                        if (graph.TryGetValue(neighbors[i], out var nn) && nn.Contains(neighbors[j]))
                            triangles++;
                    }
                }

                totalCoefficient += (2.0 * triangles) / (k * (k - 1));
                nodeCount++;
            }

            return nodeCount > 0 ? totalCoefficient / nodeCount : 0;
        }

        private List<List<int>> FindConnectedComponents(IReadOnlyDictionary<int, IReadOnlyList<int>> graph)
        {
            var visited = new HashSet<int>();
            var components = new List<List<int>>();

            foreach (var nodeId in graph.Keys)
            {
                if (visited.Contains(nodeId))
                    continue;

                var component = new List<int>();
                var queue = new Queue<int>();
                queue.Enqueue(nodeId);

                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    if (!visited.Add(current))
                        continue;
                    component.Add(current);

                    if (graph.TryGetValue(current, out var neighbors))
                    {
                        foreach (var neighbor in neighbors)
                        {
                            if (!visited.Contains(neighbor))
                                queue.Enqueue(neighbor);
                        }
                    }
                }

                components.Add(component);
            }

            return components;
        }

        private double ComputeAveragePathLength(IReadOnlyDictionary<int, IReadOnlyList<int>> graph)
        {
            var nodes = graph.Keys.ToList();
            if (nodes.Count <= 1)
                return 0;

            double totalPathLength = 0;
            int pathCount = 0;

            foreach (var source in nodes)
            {
                var distances = new Dictionary<int, int>();
                var queue = new Queue<int>();
                queue.Enqueue(source);
                distances[source] = 0;

                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    int currentDist = distances[current];

                    if (graph.TryGetValue(current, out var neighbors))
                    {
                        foreach (var neighbor in neighbors)
                        {
                            if (!distances.ContainsKey(neighbor))
                            {
                                distances[neighbor] = currentDist + 1;
                                queue.Enqueue(neighbor);
                            }
                        }
                    }
                }

                foreach (var dist in distances.Values)
                {
                    if (dist > 0)
                    {
                        totalPathLength += dist;
                        pathCount++;
                    }
                }
            }

            return pathCount > 0 ? totalPathLength / pathCount : 0;
        }
    }

    /// <summary>
    /// Population-level network metrics.
    /// </summary>
    public sealed class PopulationNetworkMetrics
    {
        public int NodeCount { get; init; }
        public int EdgeCount { get; init; }
        public double Density { get; init; }
        public double AverageDegree { get; init; }
        public double ClusteringCoefficient { get; init; }
        public int ConnectedComponentCount { get; init; }
        public double LargestComponentRatio { get; init; }
        public double AveragePathLength { get; init; }
    }

}
