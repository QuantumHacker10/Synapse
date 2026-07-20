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
    /// Supports multi-objective optimization using NSGA-II and NSGA-III
    /// algorithms for Pareto-optimal genome evolution.
    /// </summary>
    public sealed class MultiObjectiveOptimizer
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the MultiObjectiveOptimizer class.
        /// </summary>
        public MultiObjectiveOptimizer(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Computes Pareto fronts using NSGA-II non-dominated sorting.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<GeoGenome>> ComputeParetoFronts(
            GenomePopulation population,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions)
        {
            var objectives = new List<double[]>();
            foreach (var genome in population.Genomes)
            {
                var objValues = new double[objectiveCount];
                for (int i = 0; i < objectiveCount; i++)
                    objValues[i] = objectiveFunctions[i](genome);
                objectives.Add(objValues);
            }

            var dominatedCount = new int[population.Genomes.Length];
            var dominatesSet = new List<int>[population.Genomes.Length];

            for (int i = 0; i < population.Genomes.Length; i++)
            {
                dominatesSet[i] = new List<int>();
                dominatedCount[i] = 0;
            }

            for (int i = 0; i < population.Genomes.Length; i++)
            {
                for (int j = i + 1; j < population.Genomes.Length; j++)
                {
                    if (Dominates(objectives[i], objectives[j]))
                    {
                        dominatesSet[i].Add(j);
                        dominatedCount[j]++;
                    }
                    else if (Dominates(objectives[j], objectives[i]))
                    {
                        dominatesSet[j].Add(i);
                        dominatedCount[i]++;
                    }
                }
            }

            var currentFront = new List<int>();
            for (int i = 0; i < population.Genomes.Length; i++)
            {
                if (dominatedCount[i] == 0)
                    currentFront.Add(i);
            }

            var fronts = new List<List<int>>();
            while (currentFront.Count > 0)
            {
                fronts.Add(currentFront);
                var nextFront = new List<int>();

                foreach (int i in currentFront)
                {
                    foreach (int j in dominatesSet[i])
                    {
                        dominatedCount[j]--;
                        if (dominatedCount[j] == 0)
                            nextFront.Add(j);
                    }
                }

                currentFront = nextFront;
            }

            return fronts.Select(front =>
                (IReadOnlyList<GeoGenome>)front.Select(idx => population.Genomes[idx]).ToList().AsReadOnly()
            ).ToList().AsReadOnly();
        }

        /// <summary>
        /// Computes crowding distances for diversity preservation.
        /// </summary>
        public double[] ComputeCrowdingDistances(
            IReadOnlyList<GeoGenome> front,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions)
        {
            int n = front.Count;
            if (n <= 2)
                return Enumerable.Repeat(double.MaxValue, n).ToArray();

            var distances = new double[n];

            for (int m = 0; m < objectiveCount; m++)
            {
                var objectiveValues = front.Select(g => objectiveFunctions[m](g)).ToArray();
                var sortedIndices = Enumerable.Range(0, n)
                    .OrderBy(i => objectiveValues[i])
                    .ToArray();

                distances[sortedIndices[0]] = double.MaxValue;
                distances[sortedIndices[^1]] = double.MaxValue;

                double range = objectiveValues[sortedIndices[^1]] - objectiveValues[sortedIndices[0]];
                if (range < 1e-10)
                    continue;

                for (int i = 1; i < n - 1; i++)
                {
                    double spread = objectiveValues[sortedIndices[i + 1]] - objectiveValues[sortedIndices[i - 1]];
                    distances[sortedIndices[i]] += spread / range;
                }
            }

            return distances;
        }

        /// <summary>
        /// Performs NSGA-II selection for the next generation.
        /// </summary>
        public IReadOnlyList<GeoGenome> SelectNextGeneration(
            GenomePopulation population,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions,
            int targetSize)
        {
            var fronts = ComputeParetoFronts(population, objectiveCount, objectiveFunctions);
            var selected = new List<GeoGenome>();
            int frontIndex = 0;

            while (selected.Count < targetSize && frontIndex < fronts.Count)
            {
                var currentFront = fronts[frontIndex];

                if (selected.Count + currentFront.Count <= targetSize)
                {
                    selected.AddRange(currentFront);
                }
                else
                {
                    var crowdingDistances = ComputeCrowdingDistances(
                        currentFront, objectiveCount, objectiveFunctions);

                    var sortedFront = currentFront
                        .Select((g, i) => new { Genome = g, Index = i })
                        .OrderByDescending(x => crowdingDistances[x.Index])
                        .ToList();

                    int remaining = targetSize - selected.Count;
                    selected.AddRange(sortedFront.Take(remaining).Select(x => x.Genome));
                }

                frontIndex++;
            }

            return selected.AsReadOnly();
        }

        /// <summary>
        /// Finds the knee point on a Pareto front (best trade-off).
        /// </summary>
        public GeoGenome? FindKneePoint(
            IReadOnlyList<GeoGenome> front,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions)
        {
            if (front.Count < 3)
                return front.FirstOrDefault();

            var objectives = front.Select(g =>
            {
                var obj = new double[objectiveCount];
                for (int i = 0; i < objectiveCount; i++)
                    obj[i] = objectiveFunctions[i](g);
                return obj;
            }).ToList();

            var minValues = new double[objectiveCount];
            var maxValues = new double[objectiveCount];
            for (int j = 0; j < objectiveCount; j++)
            {
                minValues[j] = objectives.Min(o => o[j]);
                maxValues[j] = objectives.Max(o => o[j]);
            }

            var normalizedObjectives = objectives.Select(o =>
            {
                var normalized = new double[objectiveCount];
                for (int j = 0; j < objectiveCount; j++)
                {
                    double range = maxValues[j] - minValues[j];
                    normalized[j] = range > 1e-10 ? (o[j] - minValues[j]) / range : 0;
                }
                return normalized;
            }).ToList();

            var referencePoint = new double[objectiveCount];
            for (int j = 0; j < objectiveCount; j++)
                referencePoint[j] = 0;

            double bestDistance = double.MinValue;
            int bestIndex = 0;

            for (int i = 0; i < normalizedObjectives.Count; i++)
            {
                double distance = 0;
                for (int j = 0; j < objectiveCount; j++)
                {
                    double diff = normalizedObjectives[i][j] - referencePoint[j];
                    distance += diff * diff;
                }
                distance = Math.Sqrt(distance);

                double angleBonus = ComputeAngleBonus(normalizedObjectives[i], referencePoint);
                double score = distance + angleBonus * 0.1;

                if (score > bestDistance)
                {
                    bestDistance = score;
                    bestIndex = i;
                }
            }

            return front[bestIndex];
        }

        /// <summary>
        /// Computes the hypervolume indicator for a Pareto front.
        /// </summary>
        public double ComputeHypervolume(
            IReadOnlyList<GeoGenome> front,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions,
            double[] referencePoint)
        {
            if (front.Count == 0)
                return 0;

            var objectives = front.Select(g =>
            {
                var obj = new double[objectiveCount];
                for (int i = 0; i < objectiveCount; i++)
                    obj[i] = objectiveFunctions[i](g);
                return obj;
            }).ToList();

            var filteredObjectives = new List<double[]>();
            foreach (var obj in objectives)
            {
                bool dominated = false;
                for (int j = 0; j < objectiveCount; j++)
                {
                    if (obj[j] >= referencePoint[j])
                    {
                        dominated = true;
                        break;
                    }
                }
                if (!dominated)
                    filteredObjectives.Add(obj);
            }

            if (filteredObjectives.Count == 0)
                return 0;

            return ComputeHypervolumeRecursive(filteredObjectives, referencePoint, objectiveCount);
        }

        /// <summary>
        /// Computes the inverted generational distance (IGD) for a Pareto front approximation.
        /// </summary>
        public double ComputeIGD(
            IReadOnlyList<GeoGenome> approximation,
            IReadOnlyList<double[]> referenceFront,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions)
        {
            if (referenceFront.Count == 0)
                return 0;

            var approxObjectives = approximation.Select(g =>
            {
                var obj = new double[objectiveCount];
                for (int i = 0; i < objectiveCount; i++)
                    obj[i] = objectiveFunctions[i](g);
                return obj;
            }).ToList();

            double totalMinDistance = 0;

            foreach (var refPoint in referenceFront)
            {
                double minDistance = double.MaxValue;

                foreach (var approxPoint in approxObjectives)
                {
                    double distance = EuclideanDistance(refPoint, approxPoint);
                    minDistance = Math.Min(minDistance, distance);
                }

                totalMinDistance += minDistance;
            }

            return totalMinDistance / referenceFront.Count;
        }

        /// <summary>
        /// Merges two Pareto fronts and returns the non-dominated subset.
        /// </summary>
        public IReadOnlyList<GeoGenome> MergeParetoFronts(
            IReadOnlyList<GeoGenome> front1,
            IReadOnlyList<GeoGenome> front2,
            int objectiveCount,
            IReadOnlyList<Func<GeoGenome, double>> objectiveFunctions)
        {
            var merged = front1.Concat(front2).ToList();
            var population = new GenomePopulation
            {
                Genomes = merged.ToImmutableArray(),
                Generation = 0,
                SpeciesCount = 0,
                AverageFitness = merged.Average(g => g.Fitness),
                BestFitness = merged.Max(g => g.Fitness),
                WorstFitness = merged.Min(g => g.Fitness),
                Timestamp = DateTime.UtcNow
            };

            var fronts = ComputeParetoFronts(population, objectiveCount, objectiveFunctions);
            return fronts.Count > 0 ? fronts[0] : Array.Empty<GeoGenome>().AsReadOnly();
        }

        private bool Dominates(double[] a, double[] b)
        {
            bool atLeastOneBetter = false;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] < b[i])
                    return false;
                if (a[i] > b[i])
                    atLeastOneBetter = true;
            }
            return atLeastOneBetter;
        }

        private double ComputeAngleBonus(double[] point, double[] reference)
        {
            double dotProduct = 0;
            double normA = 0;
            for (int i = 0; i < point.Length; i++)
            {
                double diff = point[i] - reference[i];
                dotProduct += diff;
                normA += diff * diff;
            }
            return normA > 1e-10 ? dotProduct / Math.Sqrt(normA) : 0;
        }

        private double ComputeHypervolumeRecursive(List<double[]> points, double[] reference, int objectiveCount)
        {
            if (objectiveCount == 1)
            {
                double volume = 0;
                var sortedPoints = points.OrderBy(p => p[0]).ToList();

                double prevX = reference[0];
                foreach (var point in sortedPoints)
                {
                    if (point[0] < prevX)
                    {
                        volume += prevX - point[0];
                        prevX = point[0];
                    }
                }

                return volume;
            }

            var sortedByLastDim = points.OrderBy(p => p[objectiveCount - 1]).ToList();
            double hypervolume = 0;
            var processedPoints = new List<double[]>();

            double prevValue = reference[objectiveCount - 1];
            foreach (var point in sortedByLastDim)
            {
                if (point[objectiveCount - 1] >= prevValue)
                    continue;

                double sliceHeight = prevValue - point[objectiveCount - 1];
                prevValue = point[objectiveCount - 1];

                var projectedPoint = new double[objectiveCount - 1];
                for (int i = 0; i < objectiveCount - 1; i++)
                    projectedPoint[i] = point[i];

                processedPoints.Add(projectedPoint);

                var filteredPoints = processedPoints
                    .Where(p => p.All(v => v < reference[Array.IndexOf(p, p.Min())]))
                    .ToList();

                if (filteredPoints.Count > 0)
                {
                    hypervolume += sliceHeight * ComputeHypervolumeRecursive(
                        filteredPoints,
                        reference.Take(objectiveCount - 1).ToArray(),
                        objectiveCount - 1);
                }
            }

            return hypervolume;
        }

        private double EuclideanDistance(double[] a, double[] b)
        {
            double sum = 0;
            int dim = Math.Min(a.Length, b.Length);
            for (int i = 0; i < dim; i++)
            {
                double diff = a[i] - b[i];
                sum += diff * diff;
            }
            return Math.Sqrt(sum);
        }
    }

    /// <summary>
    /// Result of a multi-objective optimization run.
    /// </summary>
    public sealed class MultiObjectiveResult
    {
        /// <summary>Pareto front genomes.</summary>
        public IReadOnlyList<GeoGenome> ParetoFront { get; init; } = Array.Empty<GeoGenome>();
        /// <summary>Hypervolume indicator.</summary>
        public double Hypervolume { get; init; }
        /// <summary>Inverted generational distance.</summary>
        public double IGD { get; init; }
        /// <summary>Knee point genome.</summary>
        public GeoGenome? KneePoint { get; init; }
        /// <summary>Number of Pareto fronts.</summary>
        public int FrontCount { get; init; }
        /// <summary>Number of generations used.</summary>
        public int GenerationsUsed { get; init; }
        /// <summary>Total evaluation time.</summary>
        public TimeSpan TotalTime { get; init; }
    }

}
