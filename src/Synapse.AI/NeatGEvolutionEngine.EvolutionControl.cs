// =============================================================================
// NeatGEvolutionEngine.EvolutionControl.cs — NEAT-G partial module
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
    /// Provides external control over a running evolution process.
    /// Supports pausing, resuming, speed adjustment, and parameter modification.
    /// </summary>
    public sealed class EvolutionController : IDisposable
    {
        private readonly ManualResetEventSlim _pauseEvent;
        private readonly object _parameterLock = new();
        private volatile bool _isPaused;
        private volatile bool _shouldStop;
        private double _speedMultiplier;
        private int _maxGenerationsOverride;
        private double _targetFitnessOverride;

        /// <summary>
        /// Initializes a new instance of the EvolutionController class.
        /// </summary>
        public EvolutionController()
        {
            _pauseEvent = new ManualResetEventSlim(true);
            _isPaused = false;
            _shouldStop = false;
            _speedMultiplier = 1.0;
            _maxGenerationsOverride = -1;
            _targetFitnessOverride = double.MaxValue;
        }

        /// <summary>Whether evolution is currently paused.</summary>
        public bool IsPaused => _isPaused;

        /// <summary>Whether evolution should stop.</summary>
        public bool ShouldStop => _shouldStop;

        /// <summary>Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed).</summary>
        public double SpeedMultiplier
        {
            get => Volatile.Read(ref _speedMultiplier);
            set
            {
                lock (_parameterLock)
                {
                    _speedMultiplier = Math.Max(0.1, Math.Min(10.0, value));
                }
            }
        }

        /// <summary>
        /// Pauses the evolution process.
        /// </summary>
        public void Pause()
        {
            _isPaused = true;
            _pauseEvent.Reset();
        }

        /// <summary>
        /// Resumes the evolution process.
        /// </summary>
        public void Resume()
        {
            _isPaused = false;
            _pauseEvent.Set();
        }

        /// <summary>
        /// Signals evolution to stop gracefully.
        /// </summary>
        public void Stop()
        {
            _shouldStop = true;
            _pauseEvent.Set();
        }

        /// <summary>
        /// Waits if paused. Call this at the start of each generation.
        /// </summary>
        public void WaitIfPaused()
        {
            _pauseEvent.Wait();
        }

        /// <summary>
        /// Gets the delay to apply based on speed multiplier.
        /// </summary>
        /// <returns>Delay in milliseconds.</returns>
        public int GetSpeedDelay()
        {
            double speed = Volatile.Read(ref _speedMultiplier);
            if (speed >= 1.0)
                return 0;
            return (int)(100.0 / speed);
        }

        /// <summary>
        /// Overrides the maximum generations for the current run.
        /// </summary>
        /// <param name="maxGenerations">New maximum (-1 to use config default).</param>
        public void OverrideMaxGenerations(int maxGenerations)
        {
            Interlocked.Exchange(ref _maxGenerationsOverride, maxGenerations);
        }

        /// <summary>
        /// Overrides the target fitness for the current run.
        /// </summary>
        /// <param name="targetFitness">New target fitness.</param>
        public void OverrideTargetFitness(double targetFitness)
        {
            lock (_parameterLock)
            {
                _targetFitnessOverride = targetFitness;
            }
        }

        /// <summary>
        /// Gets the current max generations override.
        /// </summary>
        public int GetMaxGenerationsOverride()
        {
            return Volatile.Read(ref _maxGenerationsOverride);
        }

        /// <summary>
        /// Gets the current target fitness override.
        /// </summary>
        public double GetTargetFitnessOverride()
        {
            lock (_parameterLock)
            {
                return _targetFitnessOverride;
            }
        }

        /// <summary>
        /// Resets the controller to initial state.
        /// </summary>
        public void Reset()
        {
            _shouldStop = false;
            _isPaused = false;
            _pauseEvent.Set();
            SpeedMultiplier = 1.0;
            _maxGenerationsOverride = -1;
            _targetFitnessOverride = double.MaxValue;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            _pauseEvent.Dispose();
        }
    }

}
