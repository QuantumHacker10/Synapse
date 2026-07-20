// =============================================================================
// NeatGEvolutionEngine.WeightInitializationStrategies.cs — NEAT-G partial module
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
    /// Provides different strategies for initializing connection weights in new genomes.
    /// Each strategy has different statistical properties that affect evolution dynamics.
    /// </summary>
    public enum WeightInitializationStrategy
    {
        /// <summary>Uniform random distribution [-range, +range].</summary>
        Uniform,

        /// <summary>Normal distribution with given standard deviation.</summary>
        Normal,

        /// <summary>Xavier/Glorot initialization scaled by fan-in and fan-out.</summary>
        Xavier,

        /// <summary>He initialization scaled by fan-in.</summary>
        He,

        /// <summary>LeCun initialization scaled by fan-in.</summary>
        LeCun,

        /// <summary>Orthogonal initialization.</summary>
        Orthogonal,

        /// <summary>Sparse initialization with mostly zeros.</summary>
        Sparse,

        /// <summary>Constant initialization to a fixed value.</summary>
        Constant
    }

    /// <summary>
    /// Provides weight initialization for new synapses based on various strategies.
    /// </summary>
    public sealed class WeightInitializer
    {
        private readonly WeightInitializationStrategy _strategy;
        private readonly double _scale;
        private readonly Random _rng;

        /// <summary>
        /// Initializes a new instance of the WeightInitializer class.
        /// </summary>
        /// <param name="strategy">Initialization strategy.</param>
        /// <param name="scale">Scale parameter for the strategy.</param>
        /// <param name="rng">Random number generator.</param>
        public WeightInitializer(
            WeightInitializationStrategy strategy = WeightInitializationStrategy.Xavier,
            double scale = 1.0,
            Random? rng = null)
        {
            _strategy = strategy;
            _scale = scale;
            _rng = rng ?? new Random();
        }

        /// <summary>
        /// Generates a weight value based on the initialization strategy.
        /// </summary>
        /// <param name="fanIn">Number of inputs to the target neuron.</param>
        /// <param name="fanOut">Number of outputs from the source neuron.</param>
        /// <returns>Initialized weight value.</returns>
        public double Initialize(int fanIn = 1, int fanOut = 1)
        {
            return _strategy switch
            {
                WeightInitializationStrategy.Uniform => UniformInitialize(),
                WeightInitializationStrategy.Normal => NormalInitialize(),
                WeightInitializationStrategy.Xavier => XavierInitialize(fanIn, fanOut),
                WeightInitializationStrategy.He => HeInitialize(fanIn),
                WeightInitializationStrategy.LeCun => LeCunInitialize(fanIn),
                WeightInitializationStrategy.Orthogonal => OrthogonalInitialize(),
                WeightInitializationStrategy.Sparse => SparseInitialize(),
                WeightInitializationStrategy.Constant => _scale,
                _ => UniformInitialize()
            };
        }

        /// <summary>
        /// Initializes a batch of weights.
        /// </summary>
        /// <param name="count">Number of weights to initialize.</param>
        /// <param name="fanIn">Fan-in for each weight.</param>
        /// <param name="fanOut">Fan-out for each weight.</param>
        /// <returns>Array of initialized weights.</returns>
        public double[] InitializeBatch(int count, int fanIn = 1, int fanOut = 1)
        {
            return Enumerable.Range(0, count)
                .Select(_ => Initialize(fanIn, fanOut))
                .ToArray();
        }

        private double UniformInitialize()
        {
            return (_rng.NextDouble() * 2.0 - 1.0) * _scale;
        }

        private double NormalInitialize()
        {
            double u1 = _rng.NextDouble();
            double u2 = _rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            return z * _scale;
        }

        private double XavierInitialize(int fanIn, int fanOut)
        {
            double limit = Math.Sqrt(6.0 / (fanIn + fanOut));
            return (_rng.NextDouble() * 2.0 - 1.0) * limit * _scale;
        }

        private double HeInitialize(int fanIn)
        {
            double stdDev = Math.Sqrt(2.0 / fanIn);
            double u1 = _rng.NextDouble();
            double u2 = _rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            return z * stdDev * _scale;
        }

        private double LeCunInitialize(int fanIn)
        {
            double stdDev = Math.Sqrt(1.0 / fanIn);
            double u1 = _rng.NextDouble();
            double u2 = _rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            return z * stdDev * _scale;
        }

        private double OrthogonalInitialize()
        {
            return (_rng.NextDouble() * 2.0 - 1.0) * _scale;
        }

        private double SparseInitialize()
        {
            if (_rng.NextDouble() < 0.8)
                return 0;
            return (_rng.NextDouble() * 2.0 - 1.0) * _scale;
        }
    }

}
