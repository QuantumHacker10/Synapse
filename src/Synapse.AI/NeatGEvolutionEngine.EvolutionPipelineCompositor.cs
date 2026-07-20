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
    /// Composable evolution pipeline that chains multiple evolution operators
    /// in a functional style. Supports filtering, mapping, and reducing
    /// over genome populations.
    /// </summary>
    public sealed class EvolutionPipelineCompositor
    {
        private readonly List<Func<ImmutableArray<GeoGenome>, ImmutableArray<GeoGenome>>> _transforms;
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the EvolutionPipelineCompositor class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public EvolutionPipelineCompositor(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _transforms = new List<Func<ImmutableArray<GeoGenome>, ImmutableArray<GeoGenome>>>();
        }

        /// <summary>
        /// Adds a filter transform to the pipeline.
        /// </summary>
        /// <param name="predicate">Filter predicate.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor Filter(Func<GeoGenome, bool> predicate)
        {
            _transforms.Add(genomes => genomes.Where(predicate).ToImmutableArray());
            return this;
        }

        /// <summary>
        /// Adds a map transform to the pipeline.
        /// </summary>
        /// <param name="mapper">Map function.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor Map(Func<GeoGenome, GeoGenome> mapper)
        {
            _transforms.Add(genomes => genomes.Select(mapper).ToImmutableArray());
            return this;
        }

        /// <summary>
        /// Adds a sort transform to the pipeline.
        /// </summary>
        /// <param name="comparer">Sort comparer.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor Sort(Func<GeoGenome, GeoGenome, int> comparer)
        {
            _transforms.Add(genomes =>
            {
                var sorted = genomes.ToArray();
                Array.Sort(sorted, (a, b) => comparer(a, b));
                return sorted.ToImmutableArray();
            });
            return this;
        }

        /// <summary>
        /// Adds a take (limit) transform to the pipeline.
        /// </summary>
        /// <param name="count">Number of genomes to take.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor Take(int count)
        {
            _transforms.Add(genomes => genomes.Take(count).ToImmutableArray());
            return this;
        }

        /// <summary>
        /// Adds a skip transform to the pipeline.
        /// </summary>
        /// <param name="count">Number of genomes to skip.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor Skip(int count)
        {
            _transforms.Add(genomes => genomes.Skip(count).ToImmutableArray());
            return this;
        }

        /// <summary>
        /// Adds a distinct by topology transform.
        /// </summary>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor DistinctByTopology()
        {
            _transforms.Add(genomes =>
            {
                var seen = new HashSet<long>();
                var result = new List<GeoGenome>();
                foreach (var g in genomes)
                {
                    long hash = g.ComputeTopologyHash();
                    if (seen.Add(hash))
                        result.Add(g);
                }
                return result.ToImmutableArray();
            });
            return this;
        }

        /// <summary>
        /// Adds a diversity-preserving selection transform.
        /// Ensures a minimum number of distinct topologies are retained.
        /// </summary>
        /// <param name="minDiversity">Minimum distinct topologies to retain.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor EnsureDiversity(int minDiversity)
        {
            _transforms.Add(genomes =>
            {
                var result = new List<GeoGenome>();
                var seenHashes = new HashSet<long>();

                var sortedByFitness = genomes.OrderByDescending(g => g.Fitness).ToList();

                foreach (var genome in sortedByFitness)
                {
                    long hash = genome.ComputeTopologyHash();
                    if (seenHashes.Contains(hash) && result.Count >= minDiversity)
                        continue;

                    result.Add(genome);
                    seenHashes.Add(hash);
                }

                return result.ToImmutableArray();
            });
            return this;
        }

        /// <summary>
        /// Adds an elitism transform that preserves top N genomes.
        /// </summary>
        /// <param name="count">Number of elites to preserve.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor PreserveElites(int count)
        {
            _transforms.Add(genomes =>
            {
                var elites = genomes.OrderByDescending(g => g.Fitness).Take(count).ToList();
                var rest = genomes.Where(g => !elites.Any(e => e.Id == g.Id)).ToList();
                return elites.Concat(rest).ToImmutableArray();
            });
            return this;
        }

        /// <summary>
        /// Adds a complexity regularization transform that penalizes complex genomes.
        /// </summary>
        /// <param name="lambda">Regularization strength.</param>
        /// <returns>This compositor for chaining.</returns>
        public EvolutionPipelineCompositor RegularizeComplexity(double lambda = 0.01)
        {
            _transforms.Add(genomes =>
            {
                foreach (var genome in genomes)
                {
                    double penalty = lambda * genome.ComputeComplexity();
                    genome.Fitness -= penalty;
                }
                return genomes;
            });
            return this;
        }

        /// <summary>
        /// Executes the pipeline on a population.
        /// </summary>
        /// <param name="population">Input population.</param>
        /// <returns>Transformed population.</returns>
        public ImmutableArray<GeoGenome> Execute(ImmutableArray<GeoGenome> population)
        {
            var result = population;
            foreach (var transform in _transforms)
            {
                result = transform(result);
            }
            return result;
        }

        /// <summary>
        /// Executes the pipeline on a population and updates the population record.
        /// </summary>
        /// <param name="population">Input population.</param>
        /// <returns>Updated population record.</returns>
        public GenomePopulation ExecuteOnPopulation(GenomePopulation population)
        {
            var result = Execute(population.Genomes);
            return population with { Genomes = result };
        }

        /// <summary>
        /// Gets the number of transforms in the pipeline.
        /// </summary>
        public int TransformCount => _transforms.Count;

        /// <summary>
        /// Clears all transforms from the pipeline.
        /// </summary>
        public void Clear()
        {
            _transforms.Clear();
        }
    }

}
