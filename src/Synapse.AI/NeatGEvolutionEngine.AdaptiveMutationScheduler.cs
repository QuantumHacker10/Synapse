// =============================================================================
// NeatGEvolutionEngine.AdaptiveMutationScheduler.cs — NEAT-G partial module
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
    /// Dynamically adjusts mutation rates based on population fitness trends.
    /// When the population is stagnant, mutation rates are increased to encourage exploration.
    /// When the population is improving, mutation rates are decreased to allow exploitation.
    /// Uses a sliding window of fitness improvements to detect trends.
    /// </summary>
    public sealed class AdaptiveMutationScheduler
    {
        private readonly EvolutionConfig _config;
        private readonly MutationRate _baseRates;
        private readonly Queue<double> _fitnessImprovements;
        private readonly Queue<double> _diversityHistory;
        private double _currentMultiplier;
        private double _currentPerturbationMagnitude;
        private int _generationsSinceAdjustment;

        /// <summary>
        /// Initializes a new instance of the AdaptiveMutationScheduler class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        /// <param name="baseRates">Base mutation rates to scale from.</param>
        public AdaptiveMutationScheduler(EvolutionConfig config, MutationRate baseRates)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _baseRates = baseRates ?? throw new ArgumentNullException(nameof(baseRates));
            _fitnessImprovements = new Queue<double>(config.AdaptiveWindow + 1);
            _diversityHistory = new Queue<double>(config.AdaptiveWindow + 1);
            _currentMultiplier = 1.0;
            _currentPerturbationMagnitude = config.PerturbationMagnitude;
        }

        /// <summary>Gets the current adaptive mutation rate multiplier.</summary>
        public double CurrentMultiplier => _currentMultiplier;

        /// <summary>Gets the current perturbation magnitude.</summary>
        public double CurrentPerturbationMagnitude => _currentPerturbationMagnitude;

        /// <summary>
        /// Adjusts mutation rates based on the latest generation's fitness improvement and diversity.
        /// </summary>
        /// <param name="fitnessImprovement">The fitness improvement over the previous generation.</param>
        /// <param name="diversityMetric">The current population diversity metric (0-1).</param>
        /// <param name="mutationSuccessRate">The success rate of mutations in the last generation.</param>
        /// <returns>The adjusted mutation rates.</returns>
        public MutationRate Adjust(double fitnessImprovement, double diversityMetric, double mutationSuccessRate)
        {
            _fitnessImprovements.Enqueue(fitnessImprovement);
            _diversityHistory.Enqueue(diversityMetric);

            while (_fitnessImprovements.Count > _config.AdaptiveWindow)
                _fitnessImprovements.Dequeue();
            while (_diversityHistory.Count > _config.AdaptiveWindow)
                _diversityHistory.Dequeue();

            _generationsSinceAdjustment++;

            if (_generationsSinceAdjustment < 3)
                return CreateScaledRates();

            double avgImprovement = _fitnessImprovements.Count > 0
                ? _fitnessImprovements.Average()
                : 0;

            double avgDiversity = _diversityHistory.Count > 0
                ? _diversityHistory.Average()
                : 0;

            double targetMultiplier = ComputeTargetMultiplier(avgImprovement, avgDiversity, mutationSuccessRate);

            double smoothingFactor = 0.3;
            _currentMultiplier = _currentMultiplier * (1.0 - smoothingFactor) + targetMultiplier * smoothingFactor;
            _currentMultiplier = Math.Clamp(_currentMultiplier,
                _config.MinAdaptiveMutationRate / GetBaseAverageRate(),
                _config.MaxAdaptiveMutationRate / GetBaseAverageRate());

            _currentPerturbationMagnitude = _config.PerturbationMagnitude * _currentMultiplier;
            _currentPerturbationMagnitude = Math.Max(_config.MinPerturbationMagnitude, _currentPerturbationMagnitude);

            _generationsSinceAdjustment = 0;

            return CreateScaledRates();
        }

        /// <summary>
        /// Gets the adjusted mutation rates for the current state.
        /// </summary>
        /// <returns>The current scaled mutation rates.</returns>
        public MutationRate GetCurrentRates()
        {
            return CreateScaledRates();
        }

        /// <summary>
        /// Resets the scheduler to initial state.
        /// </summary>
        public void Reset()
        {
            _fitnessImprovements.Clear();
            _diversityHistory.Clear();
            _currentMultiplier = 1.0;
            _currentPerturbationMagnitude = _config.PerturbationMagnitude;
            _generationsSinceAdjustment = 0;
        }

        private double ComputeTargetMultiplier(double avgImprovement, double avgDiversity, double mutationSuccessRate)
        {
            double improvementScore = 0;
            if (Math.Abs(avgImprovement) < _config.FitnessThreshold)
            {
                improvementScore = 1.0;
            }
            else if (avgImprovement < 0)
            {
                improvementScore = 1.5;
            }
            else
            {
                improvementScore = Math.Max(0, 1.0 - avgImprovement * 10.0);
            }

            double diversityScore = 0;
            if (avgDiversity < 0.1)
            {
                diversityScore = 1.5;
            }
            else if (avgDiversity < 0.3)
            {
                diversityScore = 1.0;
            }
            else if (avgDiversity > 0.8)
            {
                diversityScore = 0.5;
            }
            else
            {
                diversityScore = 0.8;
            }

            double successScore = 0;
            if (mutationSuccessRate < 0.1)
            {
                successScore = 1.3;
            }
            else if (mutationSuccessRate > 0.8)
            {
                successScore = 0.7;
            }
            else
            {
                successScore = 1.0;
            }

            double combined = 0.4 * improvementScore + 0.35 * diversityScore + 0.25 * successScore;
            return Math.Clamp(combined, 0.3, 3.0);
        }

        private double GetBaseAverageRate()
        {
            return _baseRates.GetTotalRate() / 18.0;
        }

        private MutationRate CreateScaledRates()
        {
            var scaled = _baseRates.Clone();
            scaled.ScaleAll(_currentMultiplier);
            return scaled;
        }
    }

}
