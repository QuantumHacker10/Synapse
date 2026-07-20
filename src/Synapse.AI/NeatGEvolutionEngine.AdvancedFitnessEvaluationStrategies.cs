// =============================================================================
// NeatGEvolutionEngine.AdvancedFitnessEvaluationStrategies.cs — NEAT-G partial module
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
    /// Provides advanced multi-objective fitness evaluation strategies including
    /// Pareto optimization, lexicographic ordering, and goal programming approaches.
    /// </summary>
    public sealed class AdvancedFitnessStrategies
    {
        private readonly EvaluationContext _context;

        /// <summary>
        /// Initializes a new instance of the AdvancedFitnessStrategies class.
        /// </summary>
        /// <param name="context">Evaluation context.</param>
        public AdvancedFitnessStrategies(EvaluationContext context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
        }

        /// <summary>
        /// Evaluates genome fitness using weighted Tchebycheff scalarization.
        /// This approach minimizes the maximum weighted deviation from ideal points.
        /// </summary>
        /// <param name="genome">Genome to evaluate.</param>
        /// <param name="idealPoint">Ideal point for each objective.</param>
        /// <param name="weights">Weights for each objective.</param>
        /// <returns>Scalarized fitness value.</returns>
        public double TchebycheffScalarization(
            GeoGenome genome,
            ImmutableDictionary<FitnessComponent, double> idealPoint,
            ImmutableDictionary<FitnessComponent, double> weights)
        {
            double maxWeightedDeviation = 0;

            foreach (var component in genome.FitnessComponents)
            {
                if (!idealPoint.TryGetValue(component.Key, out double ideal))
                    ideal = 1.0;

                if (!weights.TryGetValue(component.Key, out double weight))
                    weight = 1.0;

                double deviation = Math.Abs(component.Value - ideal);
                double weightedDeviation = weight * deviation;
                maxWeightedDeviation = Math.Max(maxWeightedDeviation, weightedDeviation);
            }

            return -maxWeightedDeviation;
        }

        /// <summary>
        /// Evaluates genome fitness using the weighted sum approach.
        /// </summary>
        /// <param name="genome">Genome to evaluate.</param>
        /// <param name="weights">Objective weights.</param>
        /// <returns>Weighted sum fitness.</returns>
        public double WeightedSumFitness(
            GeoGenome genome,
            ImmutableDictionary<FitnessComponent, double> weights)
        {
            double totalFitness = 0;
            double totalWeight = 0;

            foreach (var component in genome.FitnessComponents)
            {
                if (weights.TryGetValue(component.Key, out double weight))
                {
                    totalFitness += component.Value * weight;
                    totalWeight += weight;
                }
            }

            return totalWeight > 0 ? totalFitness / totalWeight : 0;
        }

        /// <summary>
        /// Evaluates genome fitness using epsilon-constraint method.
        /// Primary objective is optimized while other objectives are treated as constraints.
        /// </summary>
        /// <param name="genome">Genome to evaluate.</param>
        /// <param name="primaryObjective">Primary objective to optimize.</param>
        /// <param name="constraints">Constraint bounds for other objectives (minimum acceptable values).</param>
        /// <returns>Constrained fitness value.</returns>
        public double EpsilonConstraintFitness(
            GeoGenome genome,
            FitnessComponent primaryObjective,
            ImmutableDictionary<FitnessComponent, double> constraints)
        {
            if (!genome.FitnessComponents.TryGetValue(primaryObjective, out double primaryValue))
                return 0;

            double penalty = 0;
            foreach (var constraint in constraints)
            {
                if (constraint.Key == primaryObjective)
                    continue;

                if (genome.FitnessComponents.TryGetValue(constraint.Key, out double value))
                {
                    if (value < constraint.Value)
                    {
                        penalty += (constraint.Value - value) * 10.0;
                    }
                }
            }

            return primaryValue - penalty;
        }

        /// <summary>
        /// Evaluates genome fitness using achievement scalarizing function.
        /// Balances proximity to reference point and distribution.
        /// </summary>
        /// <param name="genome">Genome to evaluate.</param>
        /// <param name="referencePoint">Reference point for each objective.</param>
        /// <param name="weights">Objective weights.</param>
        /// <param name="rho">Small positive constant to avoid division by zero.</param>
        /// <returns>Achievement scalarized fitness.</returns>
        public double AchievementScalarization(
            GeoGenome genome,
            ImmutableDictionary<FitnessComponent, double> referencePoint,
            ImmutableDictionary<FitnessComponent, double> weights,
            double rho = 0.001)
        {
            double maxAchievement = double.MinValue;

            foreach (var component in genome.FitnessComponents)
            {
                if (!referencePoint.TryGetValue(component.Key, out double reference))
                    reference = 0;

                if (!weights.TryGetValue(component.Key, out double weight))
                    weight = 1.0;

                double achievement = weight * (component.Value - reference);
                maxAchievement = Math.Max(maxAchievement, achievement);
            }

            double sumWeightedDeviations = 0;
            foreach (var component in genome.FitnessComponents)
            {
                if (referencePoint.TryGetValue(component.Key, out double reference) &&
                    weights.TryGetValue(component.Key, out double weight))
                {
                    sumWeightedDeviations += weight * (component.Value - reference);
                }
            }

            return maxAchievement + rho * sumWeightedDeviations;
        }

        /// <summary>
        /// Computes the hypervolume indicator for a set of genomes.
        /// Measures the volume of objective space dominated by the population.
        /// </summary>
        /// <param name="genomes">Population of genomes.</param>
        /// <param name="referencePoint">Reference point for hypervolume computation.</param>
        /// <returns>Hypervolume indicator value.</returns>
        public double ComputeHypervolume(
            IReadOnlyList<GeoGenome> genomes,
            ImmutableDictionary<FitnessComponent, double> referencePoint)
        {
            if (genomes.Count == 0)
                return 0;

            var objectives = referencePoint.Keys.ToList();
            if (objectives.Count == 0)
                return 0;

            var nonDominated = GetNonDominatedSet(genomes, objectives);
            if (nonDominated.Count == 0)
                return 0;

            if (objectives.Count == 1)
            {
                double minVal = nonDominated.Min(g =>
                    g.FitnessComponents.TryGetValue(objectives[0], out double v) ? v : 0);
                double refVal = referencePoint[objectives[0]];
                return Math.Max(0, refVal - minVal);
            }

            if (objectives.Count == 2)
            {
                return ComputeHypervolume2D(nonDominated, objectives, referencePoint);
            }

            return ComputeHypervolumeApproximation(nonDominated, objectives, referencePoint);
        }

        /// <summary>
        /// Computes the spacing metric for a set of genomes.
        /// Measures how evenly distributed the Pareto front is.
        /// </summary>
        /// <param name="genomes">Population of genomes.</param>
        /// <returns>Spacing metric (lower = more evenly distributed).</returns>
        public double ComputeSpacing(IReadOnlyList<GeoGenome> genomes)
        {
            if (genomes.Count <= 1)
                return 0;

            var objectives = genomes[0].FitnessComponents.Keys.ToList();
            if (objectives.Count == 0)
                return 0;

            var distances = new List<double>();

            for (int i = 0; i < genomes.Count; i++)
            {
                double minDist = double.MaxValue;
                for (int j = 0; j < genomes.Count; j++)
                {
                    if (i == j)
                        continue;
                    double dist = EuclideanDistance(genomes[i], genomes[j], objectives);
                    if (dist < minDist)
                        minDist = dist;
                }
                distances.Add(minDist);
            }

            double meanDist = distances.Average();
            double variance = distances.Average(d => (d - meanDist) * (d - meanDist));
            return Math.Sqrt(variance);
        }

        /// <summary>
        /// Computes the spread (delta) metric for a set of genomes.
        /// Measures the extent of the Pareto front coverage.
        /// </summary>
        /// <param name="genomes">Population of genomes.</param>
        /// <returns>Spread metric (lower = better spread).</returns>
        public double ComputeSpread(IReadOnlyList<GeoGenome> genomes)
        {
            if (genomes.Count <= 1)
                return 1;

            var objectives = genomes[0].FitnessComponents.Keys.ToList();
            if (objectives.Count == 0)
                return 1;

            var distances = new List<double>();
            for (int i = 0; i < genomes.Count; i++)
            {
                double minDist = double.MaxValue;
                for (int j = 0; j < genomes.Count; j++)
                {
                    if (i == j)
                        continue;
                    double dist = EuclideanDistance(genomes[i], genomes[j], objectives);
                    if (dist < minDist)
                        minDist = dist;
                }
                distances.Add(minDist);
            }

            double meanDist = distances.Average();
            if (meanDist < 1e-10)
                return 0;

            double df = ComputeBoundaryDistances(genomes, objectives);
            double dl = distances.Sum(d => Math.Abs(d - meanDist));

            return (df + dl) / (df + distances.Count * meanDist);
        }

        private double ComputeHypervolume2D(
            IReadOnlyList<GeoGenome> genomes,
            List<FitnessComponent> objectives,
            ImmutableDictionary<FitnessComponent, double> referencePoint)
        {
            var sorted = genomes
                .OrderBy(g => g.FitnessComponents.TryGetValue(objectives[0], out double v) ? v : 0)
                .ToList();

            double hypervolume = 0;
            double prevX = referencePoint[objectives[0]];

            foreach (var genome in sorted)
            {
                double x = genome.FitnessComponents.TryGetValue(objectives[0], out double xv) ? xv : 0;
                double y = genome.FitnessComponents.TryGetValue(objectives[1], out double yv) ? yv : 0;
                double refY = referencePoint[objectives[1]];

                double width = Math.Max(0, prevX - x);
                double height = Math.Max(0, refY - y);
                hypervolume += width * height;
                prevX = x;
            }

            return hypervolume;
        }

        private double ComputeHypervolumeApproximation(
            IReadOnlyList<GeoGenome> genomes,
            List<FitnessComponent> objectives,
            ImmutableDictionary<FitnessComponent, double> referencePoint)
        {
            double volume = 0;
            int samples = 1000;
            var rng = new Random(42);

            for (int s = 0; s < samples; s++)
            {
                var point = new Dictionary<FitnessComponent, double>();
                foreach (var obj in objectives)
                {
                    double refVal = referencePoint[obj];
                    point[obj] = rng.NextDouble() * refVal;
                }

                bool dominated = false;
                foreach (var genome in genomes)
                {
                    bool allBetter = true;
                    foreach (var obj in objectives)
                    {
                        double gv = genome.FitnessComponents.TryGetValue(obj, out double v) ? v : 0;
                        if (gv < point[obj])
                        {
                            allBetter = false;
                            break;
                        }
                    }
                    if (allBetter)
                    {
                        dominated = true;
                        break;
                    }
                }

                if (dominated)
                    volume++;
            }

            double totalVolume = 1;
            foreach (var obj in objectives)
            {
                totalVolume *= referencePoint[obj];
            }

            return (double)volume / samples * totalVolume;
        }

        private List<GeoGenome> GetNonDominatedSet(
            IReadOnlyList<GeoGenome> genomes,
            List<FitnessComponent> objectives)
        {
            var nonDominated = new List<GeoGenome>();

            foreach (var candidate in genomes)
            {
                bool isDominated = false;
                foreach (var other in genomes)
                {
                    if (candidate == other)
                        continue;

                    bool allOthersBetterOrEqual = true;
                    bool atLeastOneBetter = false;

                    foreach (var obj in objectives)
                    {
                        double candVal = candidate.FitnessComponents.TryGetValue(obj, out double cv) ? cv : 0;
                        double otherVal = other.FitnessComponents.TryGetValue(obj, out double ov) ? ov : 0;

                        if (otherVal < candVal)
                        {
                            allOthersBetterOrEqual = false;
                            break;
                        }
                        if (otherVal > candVal)
                        {
                            atLeastOneBetter = true;
                        }
                    }

                    if (allOthersBetterOrEqual && atLeastOneBetter)
                    {
                        isDominated = true;
                        break;
                    }
                }

                if (!isDominated)
                    nonDominated.Add(candidate);
            }

            return nonDominated;
        }

        private double EuclideanDistance(GeoGenome a, GeoGenome b, List<FitnessComponent> objectives)
        {
            double dist = 0;
            foreach (var obj in objectives)
            {
                double aVal = a.FitnessComponents.TryGetValue(obj, out double av) ? av : 0;
                double bVal = b.FitnessComponents.TryGetValue(obj, out double bv) ? bv : 0;
                dist += (aVal - bVal) * (aVal - bVal);
            }
            return Math.Sqrt(dist);
        }

        private double ComputeBoundaryDistances(
            IReadOnlyList<GeoGenome> genomes,
            List<FitnessComponent> objectives)
        {
            if (genomes.Count < 2)
                return 0;

            double totalDist = 0;
            foreach (var obj in objectives)
            {
                var values = genomes
                    .Select(g => g.FitnessComponents.TryGetValue(obj, out double v) ? v : 0)
                    .OrderBy(v => v)
                    .ToList();

                if (values.Count >= 2)
                {
                    totalDist += values[^1] - values[0];
                }
            }

            return totalDist;
        }
    }

}
