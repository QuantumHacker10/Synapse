// =============================================================================
// NeatGEvolutionEngine.PopulationSeedStrategies.cs — NEAT-G partial module
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
    /// Provides different strategies for seeding initial populations.
    /// Each strategy biases the initial population differently to aid evolution.
    /// </summary>
    public sealed class PopulationSeedStrategy
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the PopulationSeedStrategy class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public PopulationSeedStrategy(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Creates a seed population with varying connectivity densities.
        /// Some genomes are fully connected, others are minimal.
        /// </summary>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="populationSize">Population size.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>List of seeded genomes.</returns>
        public IReadOnlyList<GeoGenome> CreateDensitySeededPopulation(
            int inputCount, int outputCount, int populationSize, Random rng)
        {
            var genomes = new List<GeoGenome>();
            var manager = new GenomePopulationManager(_config, rng);

            for (int i = 0; i < populationSize; i++)
            {
                double densityTarget = (double)i / populationSize;
                var genome = manager.CreateRandomGenome(inputCount, outputCount, 0);

                var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();
                int targetActive = Math.Max(1, (int)(activeSynapses.Count * densityTarget));

                var toDeactivate = activeSynapses
                    .OrderBy(_ => rng.NextDouble())
                    .Skip(targetActive)
                    .ToList();

                foreach (var synapse in toDeactivate)
                {
                    synapse.IsActive = false;
                }

                genome.ComputeComplexity();
                genomes.Add(genome);
            }

            return genomes;
        }

        /// <summary>
        /// Creates a seed population with varying activation functions.
        /// Each genome uses a different mix of activation functions.
        /// </summary>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="populationSize">Population size.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>List of seeded genomes.</returns>
        public IReadOnlyList<GeoGenome> CreateActivationSeededPopulation(
            int inputCount, int outputCount, int populationSize, Random rng)
        {
            var genomes = new List<GeoGenome>();
            var manager = new GenomePopulationManager(_config, rng);
            var allActivations = Enum.GetValues<ActivationFunction>();

            for (int i = 0; i < populationSize; i++)
            {
                var genome = manager.CreateRandomGenome(inputCount, outputCount, 0);

                double bias = (double)i / populationSize;
                var dominantActivation = allActivations[(int)(bias * (allActivations.Length - 1))];

                foreach (var neuron in genome.Neurons.Where(n => n.LayerIndex > 0))
                {
                    if (rng.NextDouble() < 0.7)
                    {
                        neuron.Activation = dominantActivation;
                    }
                }

                genome.ComputeComplexity();
                genomes.Add(genome);
            }

            return genomes;
        }

        /// <summary>
        /// Creates a seed population with a Lamarckian bias - some genomes are pre-optimized
        /// with heuristic weight initialization.
        /// </summary>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="populationSize">Population size.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>List of seeded genomes.</returns>
        public IReadOnlyList<GeoGenome> CreateLamarckianSeedPopulation(
            int inputCount, int outputCount, int populationSize, Random rng)
        {
            var genomes = new List<GeoGenome>();
            var manager = new GenomePopulationManager(_config, rng);

            int optimizedCount = populationSize / 10;

            for (int i = 0; i < populationSize; i++)
            {
                var genome = manager.CreateRandomGenome(inputCount, outputCount, 0);

                if (i < optimizedCount)
                {
                    foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
                    {
                        double XavierInit = rng.NextDouble() * 2.0 - 1.0;
                        XavierInit *= Math.Sqrt(2.0 / (inputCount + outputCount));
                        synapse.Weight = XavierInit;
                    }

                    foreach (var neuron in genome.Neurons.Where(n => n.LayerIndex > 0))
                    {
                        neuron.Bias = 0.01 * (rng.NextDouble() * 2 - 1);
                    }
                }

                genome.ComputeComplexity();
                genomes.Add(genome);
            }

            return genomes;
        }

        /// <summary>
        /// Creates a seed population with hierarchical structure.
        /// Genomes have varying numbers of hidden layers (0 to max).
        /// </summary>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputCount">Number of outputs.</param>
        /// <param name="populationSize">Population size.</param>
        /// <param name="maxHiddenLayers">Maximum number of hidden layers.</param>
        /// <param name="rng">Random number generator.</param>
        /// <returns>List of seeded genomes.</returns>
        public IReadOnlyList<GeoGenome> CreateHierarchicalSeedPopulation(
            int inputCount, int outputCount, int populationSize,
            int maxHiddenLayers, Random rng)
        {
            var genomes = new List<GeoGenome>();
            var allActivations = Enum.GetValues<ActivationFunction>();

            for (int i = 0; i < populationSize; i++)
            {
                int hiddenLayers = (int)((double)i / populationSize * maxHiddenLayers) + 1;
                hiddenLayers = Math.Min(hiddenLayers, maxHiddenLayers);

                int neuronsPerLayer = Math.Max(2, (inputCount + outputCount) / 2);

                var genome = new GeoGenome
                {
                    Id = Guid.NewGuid(),
                    Generation = 0,
                    InputCount = inputCount,
                    OutputCount = outputCount
                };

                long nextInnov = 1;

                for (int inp = 0; inp < inputCount; inp++)
                {
                    genome.Neurons.Add(new GeoNeuron
                    {
                        InnovationNumber = nextInnov++,
                        LayerIndex = 0,
                        PositionInLayer = inp,
                        Activation = ActivationFunction.Linear,
                        Bias = 0,
                        IsActive = true
                    });
                }

                int currentLayer = 1;
                var prevLayerNeurons = genome.Neurons.Where(n => n.LayerIndex == 0).ToList();

                for (int hl = 0; hl < hiddenLayers; hl++)
                {
                    var layerNeurons = new List<GeoNeuron>();
                    int layerSize = Math.Max(2, neuronsPerLayer - hl);

                    for (int n = 0; n < layerSize; n++)
                    {
                        var neuron = new GeoNeuron
                        {
                            InnovationNumber = nextInnov++,
                            LayerIndex = currentLayer,
                            PositionInLayer = n,
                            Activation = allActivations[rng.Next(allActivations.Length)],
                            Bias = (rng.NextDouble() * 2 - 1) * _config.BiasInitRange,
                            IsActive = true
                        };
                        genome.Neurons.Add(neuron);
                        layerNeurons.Add(neuron);
                    }

                    long nextSynInnov = genome.Synapses.Count > 0
                        ? genome.Synapses.Max(s => s.InnovationNumber) + 1
                        : nextInnov;

                    foreach (var prev in prevLayerNeurons)
                    {
                        foreach (var curr in layerNeurons)
                        {
                            genome.Synapses.Add(new GeoSynapse
                            {
                                InnovationNumber = nextSynInnov++,
                                SourceNeuronId = prev.InnovationNumber,
                                TargetNeuronId = curr.InnovationNumber,
                                Weight = (rng.NextDouble() * 2 - 1) * _config.WeightInitRange,
                                IsActive = true
                            });
                        }
                    }

                    nextInnov = nextSynInnov;
                    prevLayerNeurons = layerNeurons;
                    currentLayer++;
                }

                var outputNeurons = new List<GeoNeuron>();
                for (int outp = 0; outp < outputCount; outp++)
                {
                    var neuron = new GeoNeuron
                    {
                        InnovationNumber = nextInnov++,
                        LayerIndex = currentLayer,
                        PositionInLayer = outp,
                        Activation = allActivations[rng.Next(allActivations.Length)],
                        Bias = (rng.NextDouble() * 2 - 1) * _config.BiasInitRange,
                        IsActive = true
                    };
                    genome.Neurons.Add(neuron);
                    outputNeurons.Add(neuron);
                }

                long finalSynInnov = genome.Synapses.Count > 0
                    ? genome.Synapses.Max(s => s.InnovationNumber) + 1
                    : nextInnov;

                foreach (var prev in prevLayerNeurons)
                {
                    foreach (var curr in outputNeurons)
                    {
                        genome.Synapses.Add(new GeoSynapse
                        {
                            InnovationNumber = finalSynInnov++,
                            SourceNeuronId = prev.InnovationNumber,
                            TargetNeuronId = curr.InnovationNumber,
                            Weight = (rng.NextDouble() * 2 - 1) * _config.WeightInitRange,
                            IsActive = true
                        });
                    }
                }

                genome.ComputeComplexity();
                genomes.Add(genome);
            }

            return genomes;
        }
    }

}
