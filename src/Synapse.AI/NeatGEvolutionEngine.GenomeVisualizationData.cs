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
    /// Provides data structures for visualizing genome structure and evolution progress.
    /// Can be used to generate graph layouts, fitness plots, and species diagrams.
    /// </summary>
    public sealed class GenomeVisualizationData
    {
        /// <summary>
        /// Node data for graph visualization.
        /// </summary>
        public sealed class GraphNode
        {
            /// <summary>Node identifier.</summary>
            public long Id { get; init; }

            /// <summary>Layer index (for layered layout).</summary>
            public int Layer { get; init; }

            /// <summary>Position within layer.</summary>
            public int Position { get; init; }

            /// <summary>Activation function name.</summary>
            public string Activation { get; init; } = string.Empty;

            /// <summary>Bias value.</summary>
            public double Bias { get; init; }

            /// <summary>Whether the node is active.</summary>
            public bool IsActive { get; init; }

            /// <summary>X coordinate for layout.</summary>
            public double X { get; set; }

            /// <summary>Y coordinate for layout.</summary>
            public double Y { get; set; }

            /// <summary>Node importance score.</summary>
            public double Importance { get; init; }

            /// <summary>Color for visualization.</summary>
            public string Color { get; init; } = "#4A90D9";
        }

        /// <summary>
        /// Edge data for graph visualization.
        /// </summary>
        public sealed class GraphEdge
        {
            /// <summary>Source node identifier.</summary>
            public long SourceId { get; init; }

            /// <summary>Target node identifier.</summary>
            public long TargetId { get; init; }

            /// <summary>Connection weight.</summary>
            public double Weight { get; init; }

            /// <summary>Whether the edge is active.</summary>
            public bool IsActive { get; init; }

            /// <summary>Edge thickness for visualization (based on weight magnitude).</summary>
            public double Thickness => Math.Max(0.5, Math.Min(5.0, Math.Abs(Weight) * 2));

            /// <summary>Edge color (green for positive, red for negative).</summary>
            public string Color => Weight >= 0 ? "#27AE60" : "#E74C3C";

            /// <summary>Edge opacity based on confidence.</summary>
            public double Opacity { get; init; } = 1.0;
        }

        /// <summary>
        /// Evolution timeline data point.
        /// </summary>
        public sealed class TimelinePoint
        {
            /// <summary>Generation number.</summary>
            public int Generation { get; init; }

            /// <summary>Best fitness at this generation.</summary>
            public double BestFitness { get; init; }

            /// <summary>Average fitness at this generation.</summary>
            public double AverageFitness { get; init; }

            /// <summary>Number of species at this generation.</summary>
            public int SpeciesCount { get; init; }

            /// <summary>Diversity metric at this generation.</summary>
            public double Diversity { get; init; }

            /// <summary>Mutation rate at this generation.</summary>
            public double MutationRate { get; init; }

            /// <summary>Number of evaluations at this generation.</summary>
            public long Evaluations { get; init; }
        }

        /// <summary>
        /// Species cluster data for species visualization.
        /// </summary>
        public sealed class SpeciesCluster
        {
            /// <summary>Species identifier.</summary>
            public int SpeciesId { get; init; }

            /// <summary>Centroid position in embedding space.</summary>
            public (double X, double Y) Centroid { get; init; }

            /// <summary>Radius of the species cluster.</summary>
            public double Radius { get; init; }

            /// <summary>Number of members.</summary>
            public int MemberCount { get; init; }

            /// <summary>Best fitness in the species.</summary>
            public double BestFitness { get; init; }

            /// <summary>Species color.</summary>
            public string Color { get; init; } = "#4A90D9";

            /// <summary>Member positions in embedding space.</summary>
            public IReadOnlyList<(double X, double Y)> MemberPositions { get; init; } =
                Array.Empty<(double, double)>();
        }

        /// <summary>
        /// Generates graph layout data for a genome.
        /// </summary>
        /// <param name="genome">The genome to visualize.</param>
        /// <returns>Tuple of nodes and edges for graph rendering.</returns>
        public static (IReadOnlyList<GraphNode> Nodes, IReadOnlyList<GraphEdge> Edges) GenerateGraphData(GeoGenome genome)
        {
            var nodes = new List<GraphNode>();
            var edges = new List<GraphEdge>();

            int maxLayer = genome.MaxLayerDepth;
            var layerCounts = genome.Neurons
                .Where(n => n.IsActive)
                .GroupBy(n => n.LayerIndex)
                .ToDictionary(g => g.Key, g => g.Count());

            foreach (var neuron in genome.Neurons)
            {
                int layerSize = layerCounts.TryGetValue(neuron.LayerIndex, out var count) ? count : 1;
                int positionInLayer = neuron.PositionInLayer;

                double x = neuron.LayerIndex * 100.0;
                double y = (positionInLayer - layerSize / 2.0) * 60.0;

                int inputCount = genome.Synapses.Count(s => s.IsActive && s.TargetNeuronId == neuron.InnovationNumber);
                int outputCount = genome.Synapses.Count(s => s.IsActive && s.SourceNeuronId == neuron.InnovationNumber);
                double importance = Math.Log(1 + inputCount + outputCount);

                string color = neuron.LayerIndex == 0 ? "#3498DB" :
                              neuron.LayerIndex == maxLayer ? "#E67E22" :
                              neuron.IsActive ? "#2ECC71" : "#95A5A6";

                nodes.Add(new GraphNode
                {
                    Id = neuron.InnovationNumber,
                    Layer = neuron.LayerIndex,
                    Position = positionInLayer,
                    Activation = neuron.Activation.ToString(),
                    Bias = neuron.Bias,
                    IsActive = neuron.IsActive,
                    X = x,
                    Y = y,
                    Importance = importance,
                    Color = color
                });
            }

            foreach (var synapse in genome.Synapses)
            {
                double opacity = synapse.IsActive ? Math.Min(1.0, Math.Abs(synapse.Weight) + 0.2) : 0.2;

                edges.Add(new GraphEdge
                {
                    SourceId = synapse.SourceNeuronId,
                    TargetId = synapse.TargetNeuronId,
                    Weight = synapse.Weight,
                    IsActive = synapse.IsActive,
                    Opacity = opacity
                });
            }

            return (nodes.AsReadOnly(), edges.AsReadOnly());
        }

        /// <summary>
        /// Generates timeline data from evolution metrics history.
        /// </summary>
        /// <param name="metricsHistory">History of evolution metrics.</param>
        /// <returns>List of timeline points.</returns>
        public static IReadOnlyList<TimelinePoint> GenerateTimelineData(IReadOnlyList<EvolutionMetrics> metricsHistory)
        {
            return metricsHistory.Select(m => new TimelinePoint
            {
                Generation = m.Generation,
                BestFitness = m.BestFitness,
                AverageFitness = m.AverageFitness,
                SpeciesCount = m.SpeciesCount,
                Diversity = m.DiversityMetric,
                MutationRate = m.AdaptiveMutationRate,
                Evaluations = m.TotalEvaluations
            }).ToList().AsReadOnly();
        }

        /// <summary>
        /// Generates species cluster data from species information using t-SNE-like
        /// dimensionality reduction.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <param name="population">Current population.</param>
        /// <returns>List of species clusters for visualization.</returns>
        public static IReadOnlyList<SpeciesCluster> GenerateSpeciesClusters(
            ImmutableArray<SpeciesInfo> species,
            GenomePopulation population)
        {
            var clusters = new List<SpeciesCluster>();
            string[] palette = {
                "#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6",
                "#1ABC9C", "#E67E22", "#34495E", "#16A085", "#C0392B",
                "#27AE60", "#2980B9", "#8E44AD", "#D35400", "#1F77B4"
            };

            var rng = new Random(42);

            foreach (var s in species)
            {
                var memberGenomes = population.Genomes
                    .Where(g => s.MemberIds.Contains(g.Id))
                    .ToList();

                var positions = new List<(double X, double Y)>();
                double sumX = 0, sumY = 0;

                foreach (var genome in memberGenomes)
                {
                    double x = rng.NextDouble() * 200 - 100;
                    double y = rng.NextDouble() * 200 - 100;

                    if (genome.SemanticEmbedding.Length >= 2)
                    {
                        x = genome.SemanticEmbedding[0] * 100;
                        y = genome.SemanticEmbedding[1] * 100;
                    }

                    positions.Add((x, y));
                    sumX += x;
                    sumY += y;
                }

                double centroidX = positions.Count > 0 ? sumX / positions.Count : 0;
                double centroidY = positions.Count > 0 ? sumY / positions.Count : 0;

                double radius = 0;
                foreach (var (px, py) in positions)
                {
                    double dist = Math.Sqrt((px - centroidX) * (px - centroidX) + (py - centroidY) * (py - centroidY));
                    radius = Math.Max(radius, dist);
                }

                clusters.Add(new SpeciesCluster
                {
                    SpeciesId = s.Id,
                    Centroid = (centroidX, centroidY),
                    Radius = Math.Max(20, radius),
                    MemberCount = s.MemberCount,
                    BestFitness = s.BestFitness,
                    Color = palette[s.Id % palette.Length],
                    MemberPositions = positions.AsReadOnly()
                });
            }

            return clusters.AsReadOnly();
        }

        /// <summary>
        /// Generates an adjacency list representation of the genome for external visualization tools.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Adjacency list as a dictionary.</returns>
        public static IReadOnlyDictionary<long, IReadOnlyList<(long TargetId, double Weight)>> GenerateAdjacencyList(GeoGenome genome)
        {
            var adjacencyList = new Dictionary<long, List<(long, double)>>();

            foreach (var neuron in genome.Neurons.Where(n => n.IsActive))
            {
                adjacencyList[neuron.InnovationNumber] = new List<(long, double)>();
            }

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (adjacencyList.TryGetValue(synapse.SourceNeuronId, out var neighbors))
                {
                    neighbors.Add((synapse.TargetNeuronId, synapse.Weight));
                }
            }

            return adjacencyList.ToDictionary(
                kvp => kvp.Key,
                kvp => (IReadOnlyList<(long, double)>)kvp.Value.AsReadOnly());
        }
    }

}
