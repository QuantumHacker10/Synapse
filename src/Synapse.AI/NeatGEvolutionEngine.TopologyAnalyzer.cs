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
    /// Analyzes the topological structure of genomes using spectral graph theory
    /// and persistent homology. Provides deep structural insights for evolution.
    /// </summary>
    public sealed class TopologyAnalyzer
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the TopologyAnalyzer class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public TopologyAnalyzer(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Performs a complete topological analysis of a genome.
        /// </summary>
        /// <param name="genome">The genome to analyze.</param>
        /// <returns>A comprehensive topological analysis result.</returns>
        public TopologicalAnalysisResult Analyze(GeoGenome genome)
        {
            var adjMatrix = BuildAdjacencyMatrix(genome);
            int n = adjMatrix.GetLength(0);

            var degreeMatrix = BuildDegreeMatrix(adjMatrix);
            var laplacian = ComputeLaplacian(adjMatrix, degreeMatrix);
            var eigenvalues = ComputeEigenvalues(laplacian);
            var spectralSignature = new SpectralSignature(eigenvalues.ToImmutableArray());

            var barcode = ComputePersistentBarcode(genome);
            int connectedComponents = CountConnectedComponents(genome);
            int cycleCount = CountCycles(genome);
            var bettiNumbers = ComputeBettiNumbers(genome);

            return new TopologicalAnalysisResult(
                spectralSignature,
                barcode,
                connectedComponents,
                cycleCount,
                bettiNumbers);
        }

        /// <summary>
        /// Builds the adjacency matrix for a genome's graph structure.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>The adjacency matrix.</returns>
        public double[,] BuildAdjacencyMatrix(GeoGenome genome)
        {
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            int n = activeNeurons.Count;

            var indexMap = new Dictionary<long, int>();
            for (int i = 0; i < n; i++)
                indexMap[activeNeurons[i].InnovationNumber] = i;

            var matrix = new double[n, n];

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (indexMap.TryGetValue(synapse.SourceNeuronId, out int srcIdx) &&
                    indexMap.TryGetValue(synapse.TargetNeuronId, out int tgtIdx))
                {
                    matrix[srcIdx, tgtIdx] = Math.Abs(synapse.Weight);
                    matrix[tgtIdx, srcIdx] = Math.Abs(synapse.Weight);
                }
            }

            return matrix;
        }

        /// <summary>
        /// Builds the degree matrix from an adjacency matrix.
        /// </summary>
        /// <param name="adjMatrix">Adjacency matrix.</param>
        /// <returns>Degree matrix.</returns>
        public double[,] BuildDegreeMatrix(double[,] adjMatrix)
        {
            int n = adjMatrix.GetLength(0);
            var degreeMatrix = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                double degree = 0;
                for (int j = 0; j < n; j++)
                    degree += adjMatrix[i, j];
                degreeMatrix[i, i] = degree;
            }

            return degreeMatrix;
        }

        /// <summary>
        /// Computes the graph Laplacian matrix (L = D - A).
        /// </summary>
        /// <param name="adjMatrix">Adjacency matrix.</param>
        /// <param name="degreeMatrix">Degree matrix.</param>
        /// <returns>Laplacian matrix.</returns>
        public double[,] ComputeLaplacian(double[,] adjMatrix, double[,] degreeMatrix)
        {
            int n = adjMatrix.GetLength(0);
            var laplacian = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    laplacian[i, j] = degreeMatrix[i, j] - adjMatrix[i, j];
                }
            }

            return laplacian;
        }

        /// <summary>
        /// Computes eigenvalues of the Laplacian matrix using the power iteration method.
        /// </summary>
        /// <param name="laplacian">The Laplacian matrix.</param>
        /// <returns>Sorted eigenvalues.</returns>
        public double[] ComputeEigenvalues(double[,] laplacian)
        {
            int n = laplacian.GetLength(0);
            if (n == 0)
                return Array.Empty<double>();

            var eigenvalues = new List<double>();
            var matrix = (double[,])laplacian.Clone();

            for (int iter = 0; iter < Math.Min(100, n * 10); iter++)
            {
                double maxVal = 0;
                int maxRow = 0, maxCol = 0;

                for (int i = 0; i < n; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        if (Math.Abs(matrix[i, j]) > maxVal)
                        {
                            maxVal = Math.Abs(matrix[i, j]);
                            maxRow = i;
                            maxCol = j;
                        }
                    }
                }

                if (maxVal < 1e-10)
                    break;

                double theta = 0.5 * Math.Atan2(
                    2 * matrix[maxRow, maxCol],
                    matrix[maxRow, maxRow] - matrix[maxCol, maxCol]);

                double c = Math.Cos(theta);
                double s = Math.Sin(theta);

                var newMatrix = (double[,])matrix.Clone();

                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (i == maxRow || i == maxCol || j == maxRow || j == maxCol)
                        {
                            double a = matrix[maxRow, j];
                            double b = matrix[maxCol, j];
                            newMatrix[i, j] = c * a + s * b;
                            if (i == maxRow || i == maxCol)
                            {
                                double a2 = matrix[i, maxRow];
                                double b2 = matrix[i, maxCol];
                                newMatrix[i, j] = c * a2 + s * b2;
                                if (j == maxRow || j == maxCol)
                                {
                                    newMatrix[maxRow, j] = c * c * matrix[maxRow, j] +
                                        2 * s * c * matrix[maxCol, j] +
                                        s * s * matrix[maxCol, j];
                                }
                            }
                        }
                    }
                }

                matrix = newMatrix;
            }

            for (int i = 0; i < n; i++)
            {
                eigenvalues.Add(matrix[i, i]);
            }

            eigenvalues.Sort();
            return eigenvalues.ToArray();
        }

        /// <summary>
        /// Counts connected components in the genome's graph using BFS.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Number of connected components.</returns>
        public int CountConnectedComponents(GeoGenome genome)
        {
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            if (activeNeurons.Count == 0)
                return 0;

            var visited = new HashSet<long>();
            int components = 0;

            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            foreach (var neuron in activeNeurons)
            {
                if (visited.Contains(neuron.InnovationNumber))
                    continue;

                components++;
                var queue = new Queue<long>();
                queue.Enqueue(neuron.InnovationNumber);

                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (!visited.Add(current))
                        continue;

                    foreach (var synapse in activeSynapses)
                    {
                        if (synapse.SourceNeuronId == current && !visited.Contains(synapse.TargetNeuronId))
                            queue.Enqueue(synapse.TargetNeuronId);
                        if (synapse.TargetNeuronId == current && !visited.Contains(synapse.SourceNeuronId))
                            queue.Enqueue(synapse.SourceNeuronId);
                    }
                }
            }

            return components;
        }

        /// <summary>
        /// Counts the number of independent cycles in the genome's graph.
        /// Uses the formula: cycles = edges - vertices + components.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Number of independent cycles.</returns>
        public int CountCycles(GeoGenome genome)
        {
            int vertices = genome.ActiveNeuronCount;
            int edges = genome.ActiveSynapseCount;
            int components = CountConnectedComponents(genome);

            return Math.Max(0, edges - vertices + components);
        }

        /// <summary>
        /// Computes Betti numbers for the genome's topology.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Immutable array of Betti numbers.</returns>
        public ImmutableArray<int> ComputeBettiNumbers(GeoGenome genome)
        {
            int beta0 = CountConnectedComponents(genome);
            int beta1 = CountCycles(genome);

            var betti = ImmutableArray.CreateBuilder<int>();
            betti.Add(beta0);
            betti.Add(beta1);

            for (int i = 2; i < genome.MaxLayerDepth + 1; i++)
            {
                betti.Add(0);
            }

            return betti.MoveToImmutable();
        }

        /// <summary>
        /// Computes a persistent barcode for the genome using a filtration
        /// based on connection weight magnitudes.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Persistent barcode.</returns>
        public PersistentBarcode ComputePersistentBarcode(GeoGenome genome)
        {
            var bars = ImmutableArray.CreateBuilder<PersistentBar>();

            var components = FindConnectedComponentsWithThreshold(genome, 0);
            foreach (var component in components)
            {
                bars.Add(new PersistentBar(0, double.PositiveInfinity, 0));
            }

            var cycles = FindCyclesWithFiltration(genome);
            foreach (var cycle in cycles)
            {
                bars.Add(new PersistentBar(cycle.Birth, cycle.Death, 1));
            }

            return new PersistentBarcode(bars.MoveToImmutable());
        }

        private List<List<long>> FindConnectedComponentsWithThreshold(GeoGenome genome, double weightThreshold)
        {
            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            var visited = new HashSet<long>();
            var components = new List<List<long>>();

            var activeSynapses = genome.Synapses
                .Where(s => s.IsActive && Math.Abs(s.Weight) >= weightThreshold)
                .ToList();

            foreach (var neuron in activeNeurons)
            {
                if (visited.Contains(neuron.InnovationNumber))
                    continue;

                var component = new List<long>();
                var queue = new Queue<long>();
                queue.Enqueue(neuron.InnovationNumber);

                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (!visited.Add(current))
                        continue;
                    component.Add(current);

                    foreach (var synapse in activeSynapses)
                    {
                        if (synapse.SourceNeuronId == current && !visited.Contains(synapse.TargetNeuronId))
                            queue.Enqueue(synapse.TargetNeuronId);
                        if (synapse.TargetNeuronId == current && !visited.Contains(synapse.SourceNeuronId))
                            queue.Enqueue(synapse.SourceNeuronId);
                    }
                }

                components.Add(component);
            }

            return components;
        }

        private List<(double Birth, double Death)> FindCyclesWithFiltration(GeoGenome genome)
        {
            var cycles = new List<(double Birth, double Death)>();
            var activeSynapses = genome.Synapses
                .Where(s => s.IsActive)
                .OrderBy(s => Math.Abs(s.Weight))
                .ToList();

            var unionFind = new Dictionary<long, long>();
            var rank = new Dictionary<long, int>();

            foreach (var neuron in genome.Neurons.Where(n => n.IsActive))
            {
                unionFind[neuron.InnovationNumber] = neuron.InnovationNumber;
                rank[neuron.InnovationNumber] = 0;
            }

            foreach (var synapse in activeSynapses)
            {
                long rootSource = FindRoot(unionFind, synapse.SourceNeuronId);
                long rootTarget = FindRoot(unionFind, synapse.TargetNeuronId);

                if (rootSource != rootTarget)
                {
                    if (rank[rootSource] < rank[rootTarget])
                        (rootSource, rootTarget) = (rootTarget, rootSource);
                    unionFind[rootTarget] = rootSource;
                    if (rank[rootSource] == rank[rootTarget])
                        rank[rootSource]++;
                }
                else
                {
                    double weight = Math.Abs(synapse.Weight);
                    cycles.Add((weight, weight * 2));
                }
            }

            return cycles;
        }

        private long FindRoot(Dictionary<long, long> unionFind, long node)
        {
            while (unionFind.TryGetValue(node, out var parent) && parent != node)
            {
                unionFind[node] = unionFind.TryGetValue(parent, out var grandParent) ? grandParent : parent;
                node = parent;
            }
            return node;
        }

        /// <summary>
        /// Computes the spectral distance between two genomes.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Spectral distance.</returns>
        public double ComputeSpectralDistance(GeoGenome a, GeoGenome b)
        {
            var analysisA = Analyze(a);
            var analysisB = Analyze(b);
            return analysisA.SpectralSignature.SpectralDistance(analysisB.SpectralSignature);
        }

        /// <summary>
        /// Computes the topological similarity between two genomes.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Topological similarity (0-1).</returns>
        public double ComputeTopologicalSimilarity(GeoGenome a, GeoGenome b)
        {
            var analysisA = Analyze(a);
            var analysisB = Analyze(b);

            double componentSimilarity = analysisA.ConnectedComponents == analysisB.ConnectedComponents
                ? 1.0
                : 1.0 / (1.0 + Math.Abs(analysisA.ConnectedComponents - analysisB.ConnectedComponents));

            double cycleSimilarity = analysisA.CycleCount == analysisB.CycleCount
                ? 1.0
                : 1.0 / (1.0 + Math.Abs(analysisA.CycleCount - analysisB.CycleCount));

            double spectralSim = Math.Exp(-analysisA.SpectralSignature.SpectralDistance(analysisB.SpectralSignature));

            double barcodeSim = Math.Exp(-analysisA.Barcode.WassersteinDistance(analysisB.Barcode));

            return 0.25 * componentSimilarity + 0.25 * cycleSimilarity +
                   0.25 * spectralSim + 0.25 * barcodeSim;
        }
    }

}
