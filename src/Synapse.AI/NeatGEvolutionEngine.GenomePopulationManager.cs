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
    /// Manages the lifecycle of genome populations including initialization,
    /// diversity maintenance, elitism, age tracking, and stagnation detection.
    /// Provides utilities for population-level operations and statistics.
    /// </summary>
    public sealed class GenomePopulationManager
    {
        private readonly EvolutionConfig _config;
        private readonly Random _rng;
        private long _nextInnovationNumber;
        private int _nextSpeciesId;

        /// <summary>
        /// Initializes a new instance of the GenomePopulationManager class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="rng">Random number generator.</param>
        public GenomePopulationManager(EvolutionConfig config, Random rng)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _rng = rng ?? throw new ArgumentNullException(nameof(rng));
            _nextInnovationNumber = 1;
            _nextSpeciesId = 1;
        }

        /// <summary>Gets the next available innovation number.</summary>
        public long NextInnovationNumber => Interlocked.Increment(ref _nextInnovationNumber);

        /// <summary>Gets the next available species ID.</summary>
        public int NextSpeciesId => Interlocked.Increment(ref _nextSpeciesId);

        /// <summary>
        /// Initializes a new random population of genomes.
        /// </summary>
        /// <param name="inputCount">Number of input neurons.</param>
        /// <param name="outputCount">Number of output neurons.</param>
        /// <returns>An initialized genome population.</returns>
        public GenomePopulation InitializePopulation(int inputCount, int outputCount)
        {
            var genomes = new List<GeoGenome>();

            for (int i = 0; i < _config.PopulationSize; i++)
            {
                var genome = CreateRandomGenome(inputCount, outputCount, 0);
                genomes.Add(genome);
            }

            return new GenomePopulation
            {
                Genomes = genomes.ToImmutableArray(),
                GenerationNumber = 0,
                Statistics = new PopulationStatistics
                {
                    SpeciesCount = 0,
                    EvaluationsThisGeneration = 0
                }
            };
        }

        /// <summary>
        /// Creates a random genome with the specified architecture.
        /// </summary>
        /// <param name="inputCount">Number of input neurons.</param>
        /// <param name="outputCount">Number of output neurons.</param>
        /// <param name="generation">Generation number.</param>
        /// <returns>A new random genome.</returns>
        public GeoGenome CreateRandomGenome(int inputCount, int outputCount, int generation)
        {
            var genome = new GeoGenome
            {
                Id = Guid.NewGuid(),
                Generation = generation,
                InputCount = inputCount,
                OutputCount = outputCount,
                Age = 0,
                BestFitness = double.MinValue,
                EvaluationCount = 0
            };

            var allActivations = Enum.GetValues<ActivationFunction>();

            for (int i = 0; i < inputCount; i++)
            {
                genome.Neurons.Add(new GeoNeuron
                {
                    InnovationNumber = NextInnovationNumber,
                    LayerIndex = 0,
                    PositionInLayer = i,
                    Activation = ActivationFunction.Linear,
                    Bias = 0,
                    IsActive = true,
                    CreationGeneration = generation
                });
            }

            for (int i = 0; i < outputCount; i++)
            {
                genome.Neurons.Add(new GeoNeuron
                {
                    InnovationNumber = NextInnovationNumber,
                    LayerIndex = 1,
                    PositionInLayer = i,
                    Activation = allActivations[_rng.Next(allActivations.Length)],
                    Bias = (_rng.NextDouble() * 2.0 - 1.0) * _config.BiasInitRange,
                    IsActive = true,
                    CreationGeneration = generation
                });
            }

            for (int i = 0; i < inputCount; i++)
            {
                for (int j = 0; j < outputCount; j++)
                {
                    genome.Synapses.Add(new GeoSynapse
                    {
                        InnovationNumber = NextInnovationNumber,
                        SourceNeuronId = genome.Neurons[i].InnovationNumber,
                        TargetNeuronId = genome.Neurons[inputCount + j].InnovationNumber,
                        Weight = (_rng.NextDouble() * 2.0 - 1.0) * _config.WeightInitRange,
                        IsActive = true,
                        CreationGeneration = generation
                    });
                }
            }

            genome.ComputeComplexity();
            genome.ComputeSemanticEmbedding(_config.SemanticEmbeddingDimension);

            return genome;
        }

        /// <summary>
        /// Applies elitism by preserving the top individuals unchanged.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <param name="species">Species information.</param>
        /// <returns>List of elite genomes to preserve.</returns>
        public IReadOnlyList<GeoGenome> GetElites(GenomePopulation population, ImmutableArray<SpeciesInfo> species)
        {
            var elites = new List<GeoGenome>();

            foreach (var s in species)
            {
                if (s.MemberIds.Length == 0)
                    continue;

                var speciesMembers = population.Genomes
                    .Where(g => s.MemberIds.Contains(g.Id))
                    .OrderByDescending(g => g.Fitness)
                    .ToList();

                int eliteCount = Math.Min(_config.EliteCount, speciesMembers.Count);
                for (int i = 0; i < eliteCount; i++)
                {
                    var elite = speciesMembers[i].Clone();
                    elite.Age++;
                    elites.Add(elite);
                }
            }

            if (elites.Count == 0 && population.Genomes.Length > 0)
            {
                int globalEliteCount = Math.Min(_config.EliteCount,
                    (int)(_config.PopulationSize * _config.EliteFraction));
                var globalElites = population.Genomes
                    .OrderByDescending(g => g.Fitness)
                    .Take(globalEliteCount);

                foreach (var elite in globalElites)
                {
                    var clone = elite.Clone();
                    clone.Age++;
                    elites.Add(clone);
                }
            }

            return elites;
        }

        /// <summary>
        /// Updates age tracking for all genomes in the population.
        /// </summary>
        /// <param name="population">The population to update.</param>
        /// <returns>A new population with updated ages.</returns>
        public GenomePopulation UpdateAges(GenomePopulation population)
        {
            var updatedGenomes = population.Genomes.Select(g =>
            {
                var clone = g.Clone();
                clone.Age++;
                if (clone.Fitness > clone.BestFitness)
                {
                    clone.BestFitness = clone.Fitness;
                    clone.BestFitnessGeneration = clone.Generation;
                }
                return clone;
            }).ToImmutableArray();

            return population with
            {
                Genomes = updatedGenomes,
                GenerationNumber = population.GenerationNumber + 1
            };
        }

        /// <summary>
        /// Detects species stagnation and marks stagnant species for potential extinction.
        /// </summary>
        /// <param name="species">Current species information.</param>
        /// <returns>Updated species with stagnation status.</returns>
        public ImmutableArray<SpeciesInfo> DetectStagnation(ImmutableArray<SpeciesInfo> species)
        {
            return species.Select(s =>
            {
                bool isStagnant = s.StagnationCounter >= _config.MaxStagnationGenerations;

                bool markedForExtinction = isStagnant && s.MemberCount <= _config.SpeciesMinimumSize;

                return s with
                {
                    IsStagnant = isStagnant,
                    IsMarkedForExtinction = markedForExtinction
                };
            }).ToImmutableArray();
        }

        /// <summary>
        /// Removes extinct species and reassigns their members.
        /// </summary>
        /// <param name="species">Current species information.</param>
        /// <param name="population">Current population.</param>
        /// <returns>Tuple of surviving species and reassignment map.</returns>
        public (ImmutableArray<SpeciesInfo> survivingSpecies, Dictionary<Guid, int> reassignments)
            HandleExtinction(ImmutableArray<SpeciesInfo> species, GenomePopulation population)
        {
            var surviving = new List<SpeciesInfo>();
            var reassignments = new Dictionary<Guid, int>();
            var activeSpeciesIds = new List<int>();

            foreach (var s in species)
            {
                if (!s.IsMarkedForExtinction && s.MemberCount >= _config.SpeciesMinimumSize)
                {
                    surviving.Add(s);
                    activeSpeciesIds.Add(s.Id);
                }
            }

            if (activeSpeciesIds.Count == 0 && species.Length > 0)
            {
                surviving.Add(species[0] with { IsMarkedForExtinction = false });
                activeSpeciesIds.Add(species[0].Id);
            }

            foreach (var s in species.Where(s => s.IsMarkedForExtinction || s.MemberCount < _config.SpeciesMinimumSize))
            {
                foreach (var memberId in s.MemberIds)
                {
                    if (activeSpeciesIds.Count > 0)
                    {
                        reassignments[memberId] = activeSpeciesIds[_rng.Next(activeSpeciesIds.Count)];
                    }
                }
            }

            return (surviving.ToImmutableArray(), reassignments);
        }

        /// <summary>
        /// Computes population diversity metrics.
        /// </summary>
        /// <param name="population">The population to analyze.</param>
        /// <returns>Diversity metrics including topology diversity and species balance.</returns>
        public PopulationStatistics ComputePopulationStatistics(GenomePopulation population, ImmutableArray<SpeciesInfo> species)
        {
            if (population.Genomes.Length == 0)
                return new PopulationStatistics();

            var fitnesses = population.Genomes.Select(g => g.Fitness).ToArray();
            double mean = fitnesses.Average();
            double variance = fitnesses.Average(f => (f - mean) * (f - mean));
            double stdDev = Math.Sqrt(variance);

            var topologyHashes = population.Genomes.Select(g => g.ComputeTopologyHash()).Distinct().Count();

            var speciesCounts = species.Select(s => s.MemberCount).ToArray();
            double shannonEntropy = 0;
            int totalMembers = speciesCounts.Sum();
            if (totalMembers > 0)
            {
                foreach (var count in speciesCounts)
                {
                    if (count > 0)
                    {
                        double p = (double)count / totalMembers;
                        shannonEntropy -= p * Math.Log2(p);
                    }
                }
            }
            double maxEntropy = speciesCounts.Length > 0 ? Math.Log2(speciesCounts.Length) : 1;
            double diversityIndex = maxEntropy > 0 ? shannonEntropy / maxEntropy : 0;

            return new PopulationStatistics
            {
                MeanFitness = mean,
                MedianFitness = population.MedianFitness,
                StdDevFitness = stdDev,
                BestFitness = fitnesses.Max(),
                WorstFitness = fitnesses.Min(),
                UniqueTopologies = topologyHashes,
                SpeciesCount = species.Length,
                DiversityIndex = diversityIndex,
                AverageComplexity = population.Genomes.Average(g => g.Complexity),
                EvaluationsThisGeneration = population.Genomes.Count(g => !g.IsFitnessValid)
            };
        }

        /// <summary>
        /// Computes fitness sharing within species.
        /// </summary>
        /// <param name="population">Current population.</param>
        /// <param name="species">Species information.</param>
        /// <returns>Population with adjusted fitness values.</returns>
        public GenomePopulation ApplyFitnessSharing(GenomePopulation population, ImmutableArray<SpeciesInfo> species)
        {
            var adjustedGenomes = population.Genomes.Select(genome =>
            {
                var clone = genome.Clone();

                var speciesInfo = species.FirstOrDefault(s => s.MemberIds.Contains(genome.Id));
                if (speciesInfo.MemberCount > 0)
                {
                    double sharedFitness = genome.Fitness / Math.Pow(speciesInfo.MemberCount, _config.SharingExponent);
                    clone.AdjustedFitness = sharedFitness;
                }
                else
                {
                    clone.AdjustedFitness = genome.Fitness;
                }

                return clone;
            }).ToImmutableArray();

            return population with { Genomes = adjustedGenomes };
        }

        /// <summary>
        /// Allocates offspring counts to each species based on adjusted fitness.
        /// </summary>
        /// <param name="species">Species information.</param>
        /// <param name="totalOffspring">Total offspring to allocate.</param>
        /// <returns>Species with offspring allocation.</returns>
        public ImmutableArray<SpeciesInfo> AllocateOffspring(ImmutableArray<SpeciesInfo> species, int totalOffspring)
        {
            double totalFitness = species.Sum(s => Math.Max(0, s.AverageFitness * s.MemberCount));
            if (totalFitness <= 0)
            {
                int perSpecies = totalOffspring / Math.Max(1, species.Length);
                return species.Select(s => s with { OffspringAllocation = perSpecies }).ToImmutableArray();
            }

            var allocated = new List<SpeciesInfo>();
            int allocated_total = 0;

            foreach (var s in species)
            {
                double proportion = (s.AverageFitness * s.MemberCount) / totalFitness;
                int offspring = Math.Max(_config.EliteCount, (int)(proportion * totalOffspring));
                offspring = Math.Min(offspring, totalOffspring - allocated_total);
                allocated_total += offspring;
                allocated.Add(s with { OffspringAllocation = offspring });
            }

            if (allocated_total < totalOffspring)
            {
                var last = allocated.Last();
                allocated[^1] = last with { OffspringAllocation = last.OffspringAllocation + (totalOffspring - allocated_total) };
            }

            return allocated.ToImmutableArray();
        }
    }

}
