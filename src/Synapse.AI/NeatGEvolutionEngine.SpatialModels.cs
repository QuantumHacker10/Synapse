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

    namespace Models
    {
        /// <summary>
        /// Represents a point in the semantic embedding space.
        /// Used for manifold distance computations and landmark selection.
        /// </summary>
        public readonly struct EmbeddingPoint : IEquatable<EmbeddingPoint>
        {
            /// <summary>The embedding coordinates.</summary>
            public readonly ImmutableArray<double> Coordinates;

            /// <summary>Optional identifier for this point.</summary>
            public readonly long Id;

            /// <summary>Dimensionality of the embedding.</summary>
            public int Dimension => Coordinates.Length;

            /// <summary>
            /// Initializes a new EmbeddingPoint.
            /// </summary>
            /// <param name="id">Identifier.</param>
            /// <param name="coordinates">Embedding coordinates.</param>
            public EmbeddingPoint(long id, ImmutableArray<double> coordinates)
            {
                Id = id;
                Coordinates = coordinates;
            }

            /// <summary>Computes Euclidean distance to another point.</summary>
            public double DistanceTo(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double sum = 0;
                for (int i = 0; i < dim; i++)
                {
                    double diff = Coordinates[i] - other.Coordinates[i];
                    sum += diff * diff;
                }
                return Math.Sqrt(sum);
            }

            /// <summary>Computes cosine similarity to another point.</summary>
            public double CosineSimilarity(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double dotProduct = 0, normA = 0, normB = 0;
                for (int i = 0; i < dim; i++)
                {
                    dotProduct += Coordinates[i] * other.Coordinates[i];
                    normA += Coordinates[i] * Coordinates[i];
                    normB += other.Coordinates[i] * other.Coordinates[i];
                }
                double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
                return denominator > 1e-10 ? dotProduct / denominator : 0;
            }

            /// <summary>Computes Manhattan distance to another point.</summary>
            public double ManhattanDistanceTo(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double sum = 0;
                for (int i = 0; i < dim; i++)
                {
                    sum += Math.Abs(Coordinates[i] - other.Coordinates[i]);
                }
                return sum;
            }

            /// <summary>Computes Chebyshev (L-infinity) distance to another point.</summary>
            public double ChebyshevDistanceTo(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double maxDiff = 0;
                for (int i = 0; i < dim; i++)
                {
                    maxDiff = Math.Max(maxDiff, Math.Abs(Coordinates[i] - other.Coordinates[i]));
                }
                return maxDiff;
            }

            /// <summary>Computes the Minkowski distance with given exponent.</summary>
            public double MinkowskiDistanceTo(EmbeddingPoint other, double p)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double sum = 0;
                for (int i = 0; i < dim; i++)
                {
                    sum += Math.Pow(Math.Abs(Coordinates[i] - other.Coordinates[i]), p);
                }
                return Math.Pow(sum, 1.0 / p);
            }

            /// <summary>Linearly interpolates between this point and another.</summary>
            public EmbeddingPoint Lerp(EmbeddingPoint other, double t)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                var coords = new double[dim];
                for (int i = 0; i < dim; i++)
                {
                    coords[i] = Coordinates[i] * (1 - t) + other.Coordinates[i] * t;
                }
                return new EmbeddingPoint(Id, coords.ToImmutableArray());
            }

            /// <summary>Normalizes the embedding to unit length.</summary>
            public EmbeddingPoint Normalize()
            {
                double norm = 0;
                for (int i = 0; i < Coordinates.Length; i++)
                    norm += Coordinates[i] * Coordinates[i];
                norm = Math.Sqrt(norm);

                if (norm < 1e-10)
                    return this;

                var normalized = new double[Coordinates.Length];
                for (int i = 0; i < Coordinates.Length; i++)
                    normalized[i] = Coordinates[i] / norm;
                return new EmbeddingPoint(Id, normalized.ToImmutableArray());
            }

            /// <summary>Subtracts two embedding points.</summary>
            public EmbeddingPoint Subtract(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                var result = new double[dim];
                for (int i = 0; i < dim; i++)
                    result[i] = Coordinates[i] - other.Coordinates[i];
                return new EmbeddingPoint(0, result.ToImmutableArray());
            }

            /// <summary>Adds two embedding points.</summary>
            public EmbeddingPoint Add(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                var result = new double[dim];
                for (int i = 0; i < dim; i++)
                    result[i] = Coordinates[i] + other.Coordinates[i];
                return new EmbeddingPoint(0, result.ToImmutableArray());
            }

            /// <summary>Scales the embedding by a scalar.</summary>
            public EmbeddingPoint Scale(double scalar)
            {
                var result = new double[Coordinates.Length];
                for (int i = 0; i < Coordinates.Length; i++)
                    result[i] = Coordinates[i] * scalar;
                return new EmbeddingPoint(Id, result.ToImmutableArray());
            }

            /// <summary>Computes the dot product with another point.</summary>
            public double DotProduct(EmbeddingPoint other)
            {
                int dim = Math.Min(Coordinates.Length, other.Coordinates.Length);
                double sum = 0;
                for (int i = 0; i < dim; i++)
                    sum += Coordinates[i] * other.Coordinates[i];
                return sum;
            }

            /// <summary>Computes the L2 norm of this point.</summary>
            public double L2Norm()
            {
                double sum = 0;
                for (int i = 0; i < Coordinates.Length; i++)
                    sum += Coordinates[i] * Coordinates[i];
                return Math.Sqrt(sum);
            }

            /// <summary>Creates a zero vector of the given dimension.</summary>
            public static EmbeddingPoint Zero(int dimension) =>
                new(0, ImmutableArray.CreateRange(Enumerable.Repeat(0.0, dimension)));

            /// <summary>Creates a random embedding point.</summary>
            public static EmbeddingPoint Random(int dimension, Random rng) =>
                new(0, ImmutableArray.CreateRange(Enumerable.Range(0, dimension)
                    .Select(_ => rng.NextDouble() * 2 - 1)));

            /// <inheritdoc/>
            public bool Equals(EmbeddingPoint other) => Id == other.Id;

            /// <inheritdoc/>
            public override bool Equals(object? obj) => obj is EmbeddingPoint p && Equals(p);

            /// <inheritdoc/>
            public override int GetHashCode() => Id.GetHashCode();

            /// <inheritdoc/>
            public override string ToString() =>
                $"EmbeddingPoint(Id={Id}, Dim={Dimension}, Norm={L2Norm():F4})";
        }

        /// <summary>
        /// Represents the curvature of a manifold at a specific point.
        /// Used for manifold-aware distance computations.
        /// </summary>
        public readonly struct ManifoldCurvature
        {
            /// <summary>Ricci scalar curvature value.</summary>
            public readonly double ScalarCurvature;

            /// <summary>Sectional curvatures along principal directions.</summary>
            public readonly ImmutableArray<double> SectionalCurvatures;

            /// <summary>Principal directions of curvature.</summary>
            public readonly ImmutableArray<EmbeddingPoint> PrincipalDirections;

            /// <summary>Curvature rank (number of significant curvature components).</summary>
            public int Rank => SectionalCurvatures.Count(c => Math.Abs(c) > 1e-6);

            /// <summary>
            /// Initializes a new ManifoldCurvature.
            /// </summary>
            public ManifoldCurvature(
                double scalarCurvature,
                ImmutableArray<double> sectionalCurvatures,
                ImmutableArray<EmbeddingPoint> principalDirections)
            {
                ScalarCurvature = scalarCurvature;
                SectionalCurvatures = sectionalCurvatures;
                PrincipalDirections = principalDirections;
            }

            /// <summary>
            /// Computes the Gaussian curvature at this point.
            /// </summary>
            public double GaussianCurvature()
            {
                if (SectionalCurvatures.Length < 2)
                    return ScalarCurvature;
                return SectionalCurvatures[0] * SectionalCurvatures[1];
            }

            /// <summary>
            /// Computes the mean curvature at this point.
            /// </summary>
            public double MeanCurvature()
            {
                if (SectionalCurvatures.Length == 0)
                    return 0;
                return SectionalCurvatures.Average();
            }

            /// <summary>
            /// Computes the maximum principal curvature.
            /// </summary>
            public double MaxPrincipalCurvature()
            {
                return SectionalCurvatures.Length > 0
                    ? SectionalCurvatures.Max()
                    : 0;
            }

            /// <summary>
            /// Computes the minimum principal curvature.
            /// </summary>
            public double MinPrincipalCurvature()
            {
                return SectionalCurvatures.Length > 0
                    ? SectionalCurvatures.Min()
                    : 0;
            }

            /// <summary>
            /// Estimates geodesic deviation based on curvature.
            /// </summary>
            /// <param name="distance">Euclidean distance.</param>
            /// <returns>Estimated geodesic distance accounting for curvature.</returns>
            public double EstimateGeodesicDeviation(double distance)
            {
                double k = ScalarCurvature;
                if (Math.Abs(k) < 1e-10)
                    return distance;

                if (k > 0)
                {
                    double sqrtK = Math.Sqrt(k);
                    return Math.Sin(sqrtK * distance) / sqrtK;
                }
                else
                {
                    double sqrtAbsK = Math.Sqrt(-k);
                    return Math.Sinh(sqrtAbsK * distance) / sqrtAbsK;
                }
            }

            /// <summary>Creates a flat (zero curvature) instance.</summary>
            public static ManifoldCurvature Flat(int dimensions) =>
                new(0,
                    ImmutableArray.CreateRange(Enumerable.Repeat(0.0, dimensions)),
                    ImmutableArray<EmbeddingPoint>.Empty);
        }

        /// <summary>
        /// Represents a topological feature (connected component, cycle, etc.)
        /// found in a genome's graph structure.
        /// </summary>
        public readonly struct TopologicalFeature
        {
            /// <summary>Type of topological feature.</summary>
            public readonly TopologicalFeatureType Type;

            /// <summary>Innovation numbers of neurons in this feature.</summary>
            public readonly ImmutableArray<long> NeuronIds;

            /// <summary>Innovation numbers of synapses in this feature.</summary>
            public readonly ImmutableArray<long> SynapseIds;

            /// <summary>Persistence value (for persistent homology).</summary>
            public readonly double Persistence;

            /// <summary>Birth generation of this feature.</summary>
            public readonly int BirthGeneration;

            /// <summary>Death generation of this feature (-1 if still alive).</summary>
            public readonly int DeathGeneration;

            /// <summary>Lifetime of this feature.</summary>
            public int Lifetime =>
                DeathGeneration >= 0 ? DeathGeneration - BirthGeneration : -1;

            /// <summary>
            /// Initializes a new TopologicalFeature.
            /// </summary>
            public TopologicalFeature(
                TopologicalFeatureType type,
                ImmutableArray<long> neuronIds,
                ImmutableArray<long> synapseIds,
                double persistence,
                int birthGeneration,
                int deathGeneration = -1)
            {
                Type = type;
                NeuronIds = neuronIds;
                SynapseIds = synapseIds;
                Persistence = persistence;
                BirthGeneration = birthGeneration;
                DeathGeneration = deathGeneration;
            }

            /// <summary>
            /// Determines if this feature is more significant than another.
            /// </summary>
            public bool IsMoreSignificantThan(TopologicalFeature other)
            {
                return Persistence > other.Persistence;
            }

            /// <inheritdoc/>
            public override string ToString() =>
                $"TopoFeature({Type}, Persistence={Persistence:F4}, Life={Lifetime})";
        }

        /// <summary>
        /// Types of topological features.
        /// </summary>
        public enum TopologicalFeatureType
        {
            /// <summary>A connected component (0-dimensional hole).</summary>
            ConnectedComponent,

            /// <summary>A cycle or loop (1-dimensional hole).</summary>
            Cycle,

            /// <summary>A void or cavity (2-dimensional hole).</summary>
            Void,

            /// <summary>A saddle point in the topology.</summary>
            SaddlePoint,

            /// <summary>A local minimum in the topology.</summary>
            LocalMinimum,

            /// <summary>A local maximum in the topology.</summary>
            LocalMaximum
        }

        /// <summary>
        /// Represents a graph Laplacian eigenvalue spectrum for spectral analysis
        /// of genome topology.
        /// </summary>
        public readonly struct SpectralSignature
        {
            /// <summary>Sorted eigenvalues of the graph Laplacian.</summary>
            public readonly ImmutableArray<double> Eigenvalues;

            /// <summary>Fiedler value (second smallest eigenvalue, algebraic connectivity).</summary>
            public double FiedlerValue => Eigenvalues.Length >= 2 ? Eigenvalues[1] : 0;

            /// <summary>Spectral gap (difference between first two non-zero eigenvalues).</summary>
            public double SpectralGap => Eigenvalues.Length >= 3
                ? Eigenvalues[2] - Eigenvalues[1]
                : 0;

            /// <summary>Number of connected components (count of zero eigenvalues).</summary>
            public int ConnectedComponents
            {
                get
                {
                    int count = 0;
                    foreach (var ev in Eigenvalues)
                    {
                        if (Math.Abs(ev) < 1e-6)
                            count++;
                        else
                            break;
                    }
                    return count;
                }
            }

            /// <summary>
            /// Spectral trace (sum of eigenvalues).
            /// </summary>
            public double Trace => Eigenvalues.Sum();

            /// <summary>
            /// Spectral radius (maximum absolute eigenvalue).
            /// </summary>
            public double SpectralRadius => Eigenvalues.Length > 0 ? Eigenvalues.Max(e => Math.Abs(e)) : 0;

            /// <summary>
            /// Initializes a new SpectralSignature.
            /// </summary>
            public SpectralSignature(ImmutableArray<double> eigenvalues)
            {
                Eigenvalues = eigenvalues.Sort((a, b) => a.CompareTo(b));
            }

            /// <summary>
            /// Computes the spectral distance to another signature.
            /// </summary>
            /// <param name="other">Other spectral signature.</param>
            /// <returns>Spectral distance.</returns>
            public double SpectralDistance(SpectralSignature other)
            {
                int maxLen = Math.Max(Eigenvalues.Length, other.Eigenvalues.Length);
                double dist = 0;

                for (int i = 0; i < maxLen; i++)
                {
                    double a = i < Eigenvalues.Length ? Eigenvalues[i] : 0;
                    double b = i < other.Eigenvalues.Length ? other.Eigenvalues[i] : 0;
                    dist += (a - b) * (a - b);
                }

                return Math.Sqrt(dist);
            }

            /// <summary>
            /// Computes the spectral entropy.
            /// </summary>
            public double SpectralEntropy()
            {
                double trace = Trace;
                if (Math.Abs(trace) < 1e-10)
                    return 0;

                double entropy = 0;
                foreach (var ev in Eigenvalues)
                {
                    double p = Math.Abs(ev) / trace;
                    if (p > 1e-10)
                        entropy -= p * Math.Log2(p);
                }
                return entropy;
            }

            /// <summary>
            /// Computes the spectral norm of the Laplacian.
            /// </summary>
            public double SpectralNorm()
            {
                return Eigenvalues.Length > 0 ? Eigenvalues.Max(Math.Abs) : 0;
            }

            /// <inheritdoc/>
            public override string ToString() =>
                $"SpectralSig(Dim={Eigenvalues.Length}, Fiedler={FiedlerValue:F4}, Components={ConnectedComponents})";
        }

        /// <summary>
        /// Represents a persistent homology barcode for topological data analysis
        /// of genome structure.
        /// </summary>
        public readonly struct PersistentBarcode
        {
            /// <summary>Bars in the barcode (birth, death, dimension).</summary>
            public readonly ImmutableArray<PersistentBar> Bars;

            /// <summary>Number of bars (features).</summary>
            public int Count => Bars.Length;

            /// <summary>
            /// Initializes a new PersistentBarcode.
            /// </summary>
            public PersistentBarcode(ImmutableArray<PersistentBar> bars)
            {
                Bars = bars.Sort((a, b) => a.Birth.CompareTo(b.Birth));
            }

            /// <summary>
            /// Gets the total persistence (sum of lifetimes).
            /// </summary>
            public double TotalPersistence()
            {
                return Bars.Sum(b => b.Lifetime);
            }

            /// <summary>
            /// Gets the maximum persistence.
            /// </summary>
            public double MaxPersistence()
            {
                return Bars.Length > 0 ? Bars.Max(b => b.Lifetime) : 0;
            }

            /// <summary>
            /// Gets the persistence entropy.
            /// </summary>
            public double PersistenceEntropy()
            {
                double total = TotalPersistence();
                if (Math.Abs(total) < 1e-10)
                    return 0;

                double entropy = 0;
                foreach (var bar in Bars)
                {
                    double p = bar.Lifetime / total;
                    if (p > 1e-10)
                        entropy -= p * Math.Log2(p);
                }
                return entropy;
            }

            /// <summary>
            /// Computes the bottleneck distance to another barcode.
            /// </summary>
            /// <param name="other">Other barcode.</param>
            /// <returns>Bottleneck distance.</returns>
            public double BottleneckDistance(PersistentBarcode other)
            {
                double maxDist = 0;

                foreach (var barA in Bars)
                {
                    double minDist = double.MaxValue;
                    foreach (var barB in other.Bars)
                    {
                        double dist = Math.Max(
                            Math.Abs(barA.Birth - barB.Birth),
                            Math.Abs(barA.Death - barB.Death));
                        minDist = Math.Min(minDist, dist);
                    }
                    maxDist = Math.Max(maxDist, minDist);
                }

                foreach (var barB in other.Bars)
                {
                    double minDist = double.MaxValue;
                    foreach (var barA in Bars)
                    {
                        double dist = Math.Max(
                            Math.Abs(barA.Birth - barB.Birth),
                            Math.Abs(barA.Death - barB.Death));
                        minDist = Math.Min(minDist, dist);
                    }
                    maxDist = Math.Max(maxDist, minDist);
                }

                return maxDist;
            }

            /// <summary>
            /// Computes the Wasserstein distance to another barcode.
            /// </summary>
            /// <param name="other">Other barcode.</param>
            /// <param name="p">Exponent for the Wasserstein distance.</param>
            /// <returns>Wasserstein-p distance.</returns>
            public double WassersteinDistance(PersistentBarcode other, double p = 2.0)
            {
                if (Bars.Length == 0 && other.Bars.Length == 0)
                    return 0;

                var sortedA = Bars.OrderByDescending(b => b.Lifetime).ToList();
                var sortedB = other.Bars.OrderByDescending(b => b.Lifetime).ToList();

                int maxCount = Math.Max(sortedA.Count, sortedB.Count);
                double distance = 0;

                for (int i = 0; i < maxCount; i++)
                {
                    double birthA = i < sortedA.Count ? sortedA[i].Birth : 0;
                    double deathA = i < sortedA.Count ? sortedA[i].Death : 0;
                    double birthB = i < sortedB.Count ? sortedB[i].Birth : 0;
                    double deathB = i < sortedB.Count ? sortedB[i].Death : 0;

                    double barDist = Math.Max(
                        Math.Abs(birthA - birthB),
                        Math.Abs(deathA - deathB));

                    distance += Math.Pow(barDist, p);
                }

                return Math.Pow(distance, 1.0 / p);
            }

            /// <inheritdoc/>
            public override string ToString() =>
                $"PersistentBarcode(Bars={Count}, TotalPersistence={TotalPersistence():F4})";
        }

        /// <summary>
        /// A single bar in a persistent barcode.
        /// </summary>
        public readonly struct PersistentBar : IEquatable<PersistentBar>
        {
            /// <summary>Birth time.</summary>
            public readonly double Birth;

            /// <summary>Death time.</summary>
            public readonly double Death;

            /// <summary>Topological dimension (0=components, 1=loops, etc.).</summary>
            public readonly int Dimension;

            /// <summary>Lifetime of this bar.</summary>
            public double Lifetime => Death - Birth;

            /// <summary>
            /// Initializes a new PersistentBar.
            /// </summary>
            public PersistentBar(double birth, double death, int dimension = 0)
            {
                Birth = birth;
                Death = death;
                Dimension = dimension;
            }

            /// <summary>Determines if this bar is infinite (never dies).</summary>
            public bool IsInfinite => double.IsPositiveInfinity(Death);

            /// <inheritdoc/>
            public bool Equals(PersistentBar other) =>
                Birth == other.Birth && Death == other.Death && Dimension == other.Dimension;

            /// <inheritdoc/>
            public override bool Equals(object? obj) => obj is PersistentBar b && Equals(b);

            /// <inheritdoc/>
            public override int GetHashCode() => HashCode.Combine(Birth, Death, Dimension);

            /// <inheritdoc/>
            public override string ToString() =>
                $"Bar({Dimension}D: [{Birth:F4}, {Death:F4}], Life={Lifetime:F4})";
        }

        /// <summary>
        /// Represents the result of a topological analysis of a genome.
        /// </summary>
        public readonly struct TopologicalAnalysisResult
        {
            /// <summary>Spectral signature of the genome.</summary>
            public readonly SpectralSignature SpectralSignature;

            /// <summary>Persistent barcode of the genome.</summary>
            public readonly PersistentBarcode Barcode;

            /// <summary>Number of connected components.</summary>
            public readonly int ConnectedComponents;

            /// <summary>Number of cycles.</summary>
            public readonly int CycleCount;

            /// <summary>Betti numbers (beta_0, beta_1, ...).</summary>
            public readonly ImmutableArray<int> BettiNumbers;

            /// <summary>Euler characteristic.</summary>
            public double EulerCharacteristic =>
                BettiNumbers.Length > 0
                    ? BettiNumbers.Select((b, i) => i % 2 == 0 ? b : -b).Sum()
                    : 0;

            /// <summary>
            /// Initializes a new TopologicalAnalysisResult.
            /// </summary>
            public TopologicalAnalysisResult(
                SpectralSignature spectralSignature,
                PersistentBarcode barcode,
                int connectedComponents,
                int cycleCount,
                ImmutableArray<int> bettiNumbers)
            {
                SpectralSignature = spectralSignature;
                Barcode = barcode;
                ConnectedComponents = connectedComponents;
                CycleCount = cycleCount;
                BettiNumbers = bettiNumbers;
            }

            /// <inheritdoc/>
            public override string ToString() =>
                $"TopoAnalysis(Components={ConnectedComponents}, Cycles={CycleCount}, " +
                $"Euler={EulerCharacteristic}, Persistence={Barcode.TotalPersistence():F4})";
        }
    }

}
