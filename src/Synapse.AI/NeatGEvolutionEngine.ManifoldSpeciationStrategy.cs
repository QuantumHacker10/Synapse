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
    /// Implements speciation using Gromov-Hausdorff distance approximation
    /// with landmark-based embedding and curvature-aware manifold distance.
    /// This strategy provides more meaningful genome distance measurements than
    /// simple Euclidean distance by accounting for the geometric structure of
    /// the genome embedding space.
    /// </summary>
    public sealed class ManifoldSpeciationStrategy : ISpeciationStrategy
    {
        private readonly EvolutionConfig _config;
        private readonly ConcurrentDictionary<(Guid, Guid), double> _distanceCache;
        private readonly List<GeoGenome> _landmarks;
        private double _currentThreshold;
        private int _nextSpeciesId;

        /// <summary>
        /// Initializes a new instance of the ManifoldSpeciationStrategy class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public ManifoldSpeciationStrategy(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _distanceCache = new ConcurrentDictionary<(Guid, Guid), double>();
            _landmarks = new List<GeoGenome>();
            _currentThreshold = config.SpeciationThreshold;
            _nextSpeciesId = 1;
        }

        /// <summary>Gets the current speciation threshold.</summary>
        public double CurrentThreshold => _currentThreshold;

        /// <inheritdoc/>
        public ImmutableArray<SpeciesInfo> Speciate(GenomePopulation population)
        {
            if (population.Genomes.Length == 0)
                return ImmutableArray<SpeciesInfo>.Empty;

            SelectLandmarks(population.Genomes);

            var speciesMap = new Dictionary<int, List<GeoGenome>>();
            var genomeSpecies = new Dictionary<Guid, int>();
            var speciesRepresentatives = new Dictionary<int, GeoGenome>();

            foreach (var genome in population.Genomes)
            {
                if (genome.SemanticEmbedding.IsDefaultOrEmpty)
                    genome.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);
            }

            bool assigned = false;
            foreach (var existingSpecies in GetExistingSpecies(population))
            {
                if (existingSpecies.Representative == null)
                    continue;

                var representative = existingSpecies.Representative;
                if (representative.SemanticEmbedding.IsDefaultOrEmpty)
                    representative.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

                foreach (var genome in population.Genomes)
                {
                    if (genomeSpecies.ContainsKey(genome.Id))
                        continue;

                    double distance = ComputeDistance(representative, genome);
                    if (distance <= _currentThreshold)
                    {
                        if (!speciesMap.ContainsKey(existingSpecies.Id))
                        {
                            speciesMap[existingSpecies.Id] = new List<GeoGenome>();
                            speciesRepresentatives[existingSpecies.Id] = representative;
                        }
                        speciesMap[existingSpecies.Id].Add(genome);
                        genomeSpecies[genome.Id] = existingSpecies.Id;
                        assigned = true;
                    }
                }
            }

            foreach (var genome in population.Genomes)
            {
                if (genomeSpecies.ContainsKey(genome.Id))
                    continue;

                int speciesId = _nextSpeciesId++;
                speciesMap[speciesId] = new List<GeoGenome> { genome };
                speciesRepresentatives[speciesId] = genome;
                genomeSpecies[genome.Id] = speciesId;
            }

            AdjustThreshold(speciesMap.Count);

            var species = new List<SpeciesInfo>();
            foreach (var kvp in speciesMap)
            {
                var members = kvp.Value;
                double bestFitness = members.Max(g => g.Fitness);
                double avgFitness = members.Average(g => g.Fitness);
                int bestGen = members.Where(g => g.Fitness == bestFitness).First().Generation;

                var fitnessHistory = new List<double>();
                foreach (var member in members)
                {
                    fitnessHistory.Add(member.Fitness);
                }

                species.Add(new SpeciesInfo
                {
                    Id = kvp.Key,
                    Representative = speciesRepresentatives[kvp.Key],
                    RepresentativeGenomeId = speciesRepresentatives[kvp.Key].Id,
                    MemberIds = members.Select(g => g.Id).ToImmutableArray(),
                    BestFitness = bestFitness,
                    BestFitnessGeneration = bestGen,
                    AverageFitness = avgFitness,
                    Age = members.Count > 0 ? members.Max(g => g.Age) : 0,
                    CreationGeneration = members.Count > 0 ? members.Min(g => g.Generation) : 0,
                    FitnessHistory = fitnessHistory.ToImmutableArray()
                });
            }

            return species.ToImmutableArray();
        }

        /// <inheritdoc/>
        public double ComputeDistance(GeoGenome a, GeoGenome b)
        {
            if (a.Id == b.Id)
                return 0;

            var key = a.Id.CompareTo(b.Id) < 0
                ? (a.Id, b.Id)
                : (b.Id, a.Id);

            if (_distanceCache.TryGetValue(key, out var cached))
                return cached;

            double ghDistance = ComputeGromovHausdorffDistance(a, b);
            double manifoldDist = ComputeManifoldDistance(a, b);
            double weightDiff = ComputeWeightDistance(a, b);

            double combined = 0.4 * ghDistance + 0.35 * manifoldDist + 0.25 * weightDiff;

            _distanceCache[key] = combined;
            return combined;
        }

        /// <summary>
        /// Computes the Gromov-Hausdorff distance approximation between two genomes
        /// using landmark-based embedding. This measures the structural dissimilarity
        /// between the two genome topologies.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Approximated Gromov-Hausdorff distance.</returns>
        public double ComputeGromovHausdorffDistance(GeoGenome a, GeoGenome b)
        {
            if (a.SemanticEmbedding.IsDefaultOrEmpty)
                a.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);
            if (b.SemanticEmbedding.IsDefaultOrEmpty)
                b.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

            int dim = Math.Min(a.SemanticEmbedding.Length, b.SemanticEmbedding.Length);
            if (dim == 0)
                return 1.0;

            double maxMinDistAtoB = 0;
            foreach (var landmark in _landmarks)
            {
                if (landmark.SemanticEmbedding.IsDefaultOrEmpty || landmark.SemanticEmbedding.Length < dim)
                    continue;

                double distAtoLandmark = 0;
                double distBtoLandmark = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diffA = a.SemanticEmbedding[d] - landmark.SemanticEmbedding[d];
                    double diffB = b.SemanticEmbedding[d] - landmark.SemanticEmbedding[d];
                    distAtoLandmark += diffA * diffA;
                    distBtoLandmark += diffB * diffB;
                }
                distAtoLandmark = Math.Sqrt(distAtoLandmark);
                distBtoLandmark = Math.Sqrt(distBtoLandmark);

                double minDist = Math.Min(distAtoLandmark, distBtoLandmark);
                maxMinDistAtoB = Math.Max(maxMinDistAtoB, Math.Abs(distAtoLandmark - distBtoLandmark));
            }

            double structuralDist = ComputeStructuralDistance(a, b);
            double combined = 0.6 * maxMinDistAtoB + 0.4 * structuralDist;

            return combined;
        }

        /// <summary>
        /// Computes the manifold-aware distance between two genome embeddings,
        /// taking into account local curvature information.
        /// </summary>
        /// <param name="a">First genome.</param>
        /// <param name="b">Second genome.</param>
        /// <returns>Manifold-aware distance value.</returns>
        public double ComputeManifoldDistance(GeoGenome a, GeoGenome b)
        {
            if (a.SemanticEmbedding.IsDefaultOrEmpty || b.SemanticEmbedding.IsDefaultOrEmpty)
                return ComputeGromovHausdorffDistance(a, b);

            int dim = Math.Min(a.SemanticEmbedding.Length, b.SemanticEmbedding.Length);
            if (dim == 0)
                return 1.0;

            double euclideanDist = 0;
            double[] direction = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                double diff = a.SemanticEmbedding[d] - b.SemanticEmbedding[d];
                euclideanDist += diff * diff;
                direction[d] = diff;
            }
            euclideanDist = Math.Sqrt(euclideanDist);

            if (euclideanDist < 1e-10)
                return 0;

            double curvatureFactor = ComputeCurvatureFactor(a, b, direction, dim);
            double manifoldDist = euclideanDist * (1.0 + 0.5 * curvatureFactor);

            double geodesicEstimate = EstimateGeodesicDistance(a, b, dim);
            manifoldDist = 0.7 * manifoldDist + 0.3 * geodesicEstimate;

            return manifoldDist;
        }

        private double ComputeCurvatureFactor(GeoGenome a, GeoGenome b, double[] direction, int dim)
        {
            double[] midpoint = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                midpoint[d] = (a.SemanticEmbedding[d] + b.SemanticEmbedding[d]) / 2.0;
            }

            double curvature = 0;
            int resolution = _config.CurvatureResolution;
            for (int i = 0; i < resolution; i++)
            {
                double t = (double)i / (resolution - 1);
                double[] point = new double[dim];
                for (int d = 0; d < dim; d++)
                {
                    point[d] = a.SemanticEmbedding[d] + t * direction[d];
                }

                double localDeviation = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = point[d] - midpoint[d];
                    localDeviation += diff * diff;
                }
                localDeviation = Math.Sqrt(localDeviation);

                double expectedDist = t * Math.Sqrt(direction.Sum(d => d * d));
                curvature += Math.Abs(localDeviation - expectedDist);
            }

            return curvature / resolution;
        }

        private double EstimateGeodesicDistance(GeoGenome a, GeoGenome b, int dim)
        {
            if (_landmarks.Count < 2)
            {
                return ComputeEuclideanDistance(a, b);
            }

            double minDistToA = double.MaxValue;
            double minDistToB = double.MaxValue;
            double landmarkDistAB = 0;

            foreach (var landmark in _landmarks)
            {
                if (landmark.SemanticEmbedding.IsDefaultOrEmpty || landmark.SemanticEmbedding.Length < dim)
                    continue;

                double distToA = 0;
                double distToB = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diffA = a.SemanticEmbedding[d] - landmark.SemanticEmbedding[d];
                    double diffB = b.SemanticEmbedding[d] - landmark.SemanticEmbedding[d];
                    distToA += diffA * diffA;
                    distToB += diffB * diffB;
                }
                distToA = Math.Sqrt(distToA);
                distToB = Math.Sqrt(distToB);

                minDistToA = Math.Min(minDistToA, distToA);
                minDistToB = Math.Min(minDistToB, distToB);
            }

            double directDist = ComputeEuclideanDistance(a, b);
            double landmarkBased = Math.Abs(minDistToA - minDistToB) + 0.5 * (minDistToA + minDistToB);

            return 0.6 * directDist + 0.4 * landmarkBased;
        }

        private double ComputeStructuralDistance(GeoGenome a, GeoGenome b)
        {
            int neuronDiff = Math.Abs(a.ActiveNeuronCount - b.ActiveNeuronCount);
            int synapseDiff = Math.Abs(a.ActiveSynapseCount - b.ActiveSynapseCount);
            int layerDiff = Math.Abs(a.MaxLayerDepth - b.MaxLayerDepth);

            double maxNeurons = Math.Max(a.ActiveNeuronCount, b.ActiveNeuronCount);
            double maxSynapses = Math.Max(a.ActiveSynapseCount, b.ActiveSynapseCount);
            double maxLayers = Math.Max(a.MaxLayerDepth, b.MaxLayerDepth);

            double normNeuronDiff = maxNeurons > 0 ? (double)neuronDiff / maxNeurons : 0;
            double normSynapseDiff = maxSynapses > 0 ? (double)synapseDiff / maxSynapses : 0;
            double normLayerDiff = maxLayers > 0 ? (double)layerDiff / maxLayers : 0;

            double topologyHashDist = ComputeTopologyHashDistance(a, b);

            return 0.3 * normNeuronDiff + 0.3 * normSynapseDiff + 0.2 * normLayerDiff + 0.2 * topologyHashDist;
        }

        private double ComputeTopologyHashDistance(GeoGenome a, GeoGenome b)
        {
            long hashA = a.ComputeTopologyHash();
            long hashB = b.ComputeTopologyHash();

            if (hashA == hashB)
                return 0;

            long xor = hashA ^ hashB;
            int differingBits = 0;
            while (xor != 0)
            {
                differingBits++;
                xor &= xor - 1;
            }

            return Math.Min(1.0, (double)differingBits / 64.0);
        }

        private double ComputeWeightDistance(GeoGenome a, GeoGenome b)
        {
            var aWeights = a.Synapses
                .Where(s => s.IsActive)
                .OrderBy(s => s.InnovationNumber)
                .Select(s => s.Weight)
                .ToList();

            var bWeights = b.Synapses
                .Where(s => s.IsActive)
                .OrderBy(s => s.InnovationNumber)
                .Select(s => s.Weight)
                .ToList();

            if (aWeights.Count == 0 || bWeights.Count == 0)
                return 1.0;

            var commonInnovations = a.Synapses
                .Where(sA => sA.IsActive && b.Synapses.Any(sB => sB.IsActive && sB.InnovationNumber == sA.InnovationNumber))
                .Select(s => s.InnovationNumber)
                .ToList();

            if (commonInnovations.Count == 0)
                return 1.0;

            var aMap = a.Synapses.Where(s => s.IsActive).ToDictionary(s => s.InnovationNumber, s => s.Weight);
            var bMap = b.Synapses.Where(s => s.IsActive).ToDictionary(s => s.InnovationNumber, s => s.Weight);

            double totalDiff = 0;
            foreach (var innov in commonInnovations)
            {
                totalDiff += Math.Abs(aMap[innov] - bMap[innov]);
            }

            double avgDiff = totalDiff / commonInnovations.Count;
            double disjointFraction = 1.0 - (double)commonInnovations.Count /
                Math.Max(a.ActiveSynapseCount, b.ActiveSynapseCount);

            return 0.5 * Math.Min(1.0, avgDiff) + 0.5 * disjointFraction;
        }

        private double ComputeEuclideanDistance(GeoGenome a, GeoGenome b)
        {
            int dim = Math.Min(a.SemanticEmbedding.Length, b.SemanticEmbedding.Length);
            double dist = 0;
            for (int d = 0; d < dim; d++)
            {
                double diff = a.SemanticEmbedding[d] - b.SemanticEmbedding[d];
                dist += diff * diff;
            }
            return Math.Sqrt(dist);
        }

        private void SelectLandmarks(ImmutableArray<GeoGenome> genomes)
        {
            _landmarks.Clear();
            if (genomes.Length == 0)
                return;

            int landmarkCount = Math.Min(_config.LandmarkCount, genomes.Length);

            var rng = new Random(42);
            var shuffled = genomes.ToArray();
            for (int i = shuffled.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
            }

            _landmarks.Add(shuffled[0]);

            for (int l = 1; l < landmarkCount; l++)
            {
                GeoGenome? bestCandidate = null;
                double maxMinDist = -1;

                foreach (var candidate in shuffled)
                {
                    if (_landmarks.Any(lm => lm.Id == candidate.Id))
                        continue;

                    double minDist = double.MaxValue;
                    foreach (var landmark in _landmarks)
                    {
                        double dist = ComputeEuclideanDistance(candidate, landmark);
                        minDist = Math.Min(minDist, dist);
                    }

                    if (minDist > maxMinDist)
                    {
                        maxMinDist = minDist;
                        bestCandidate = candidate;
                    }
                }

                if (bestCandidate != null)
                    _landmarks.Add(bestCandidate);
            }
        }

        private void AdjustThreshold(int currentSpeciesCount)
        {
            double error = (double)(currentSpeciesCount - _config.TargetSpeciesCount) / _config.TargetSpeciesCount;
            _currentThreshold *= (1.0 - _config.ThresholdAdjustmentRate * error);
            _currentThreshold = Math.Clamp(_currentThreshold,
                _config.MinSpeciationThreshold, _config.MaxSpeciationThreshold);
        }

        private ImmutableArray<SpeciesInfo> GetExistingSpecies(GenomePopulation population)
        {
            var speciesDict = new Dictionary<int, SpeciesInfo>();

            foreach (var genome in population.Genomes)
            {
                if (genome.SpeciesId < 0)
                    continue;

                if (!speciesDict.TryGetValue(genome.SpeciesId, out var existing))
                {
                    speciesDict[genome.SpeciesId] = new SpeciesInfo
                    {
                        Id = genome.SpeciesId,
                        Representative = genome,
                        RepresentativeGenomeId = genome.Id,
                        MemberIds = ImmutableArray.Create(genome.Id),
                        BestFitness = genome.Fitness,
                        BestFitnessGeneration = genome.Generation,
                        Age = genome.Age
                    };
                }
                else
                {
                    var memberIds = existing.MemberIds.Add(genome.Id);
                    var bestFitness = Math.Max(existing.BestFitness, genome.Fitness);
                    speciesDict[genome.SpeciesId] = existing with
                    {
                        MemberIds = memberIds,
                        BestFitness = bestFitness
                    };
                }
            }

            return speciesDict.Values.ToImmutableArray();
        }

        /// <summary>
        /// Clears the distance cache to free memory.
        /// </summary>
        public void ClearCache()
        {
            _distanceCache.Clear();
        }

        /// <summary>
        /// Gets the number of cached distance entries.
        /// </summary>
        public int CacheSize => _distanceCache.Count;
    }

}
