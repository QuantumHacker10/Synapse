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
    /// Implements a formal state machine for the NEAT-G evolution process.
    /// Ensures valid state transitions and provides deterministic behavior.
    /// </summary>
    public sealed class EvolutionStateMachine
    {
        private EvolutionState _currentState;
        private readonly Dictionary<(EvolutionState From, string Trigger), EvolutionState> _transitions;
        private readonly List<(EvolutionState From, EvolutionState To, string Trigger, DateTime Timestamp)> _transitionLog;

        /// <summary>
        /// Initializes a new instance of the EvolutionStateMachine class.
        /// </summary>
        public EvolutionStateMachine()
        {
            _currentState = EvolutionState.NotStarted;
            _transitions = new Dictionary<(EvolutionState, string), EvolutionState>();
            _transitionLog = new List<(EvolutionState, EvolutionState, string, DateTime)>();

            DefineTransitions();
        }

        /// <summary>Gets the current state.</summary>
        public EvolutionState CurrentState => _currentState;

        /// <summary>Gets the transition log.</summary>
        public IReadOnlyList<(EvolutionState From, EvolutionState To, string Trigger, DateTime Timestamp)> TransitionLog =>
            _transitionLog.AsReadOnly();

        /// <summary>
        /// Attempts a state transition.
        /// </summary>
        /// <param name="trigger">The trigger causing the transition.</param>
        /// <returns>True if transition was successful.</returns>
        public bool TryTransition(string trigger)
        {
            var key = (_currentState, trigger);
            if (_transitions.TryGetValue(key, out var newState))
            {
                var oldState = _currentState;
                _currentState = newState;
                _transitionLog.Add((oldState, newState, trigger, DateTime.UtcNow));
                return true;
            }
            return false;
        }

        /// <summary>
        /// Forces a state transition without checking validity.
        /// </summary>
        /// <param name="newState">The new state.</param>
        public void ForceTransition(EvolutionState newState)
        {
            var oldState = _currentState;
            _currentState = newState;
            _transitionLog.Add((oldState, newState, "Force", DateTime.UtcNow));
        }

        /// <summary>
        /// Checks if a transition is valid from the current state.
        /// </summary>
        /// <param name="trigger">The trigger to check.</param>
        /// <returns>True if the transition is valid.</summary>
        public bool CanTransition(string trigger)
        {
            return _transitions.ContainsKey((_currentState, trigger));
        }

        /// <summary>
        /// Gets all valid triggers from the current state.
        /// </summary>
        public IReadOnlyList<string> GetValidTriggers()
        {
            return _transitions
                .Where(kvp => kvp.Key.From == _currentState)
                .Select(kvp => kvp.Key.Trigger)
                .ToList()
                .AsReadOnly();
        }

        /// <summary>
        /// Resets the state machine to initial state.
        /// </summary>
        public void Reset()
        {
            _currentState = EvolutionState.NotStarted;
            _transitionLog.Clear();
        }

        private void DefineTransitions()
        {
            _transitions[(EvolutionState.NotStarted, "Initialize")] = EvolutionState.Initializing;
            _transitions[(EvolutionState.Initializing, "Evaluate")] = EvolutionState.Evaluating;
            _transitions[(EvolutionState.Evaluating, "Speciate")] = EvolutionState.Speciating;
            _transitions[(EvolutionState.Speciating, "Select")] = EvolutionState.Selecting;
            _transitions[(EvolutionState.Selecting, "Evolve")] = EvolutionState.Evolving;
            _transitions[(EvolutionState.Evolving, "Evaluate")] = EvolutionState.Evaluating;
            _transitions[(EvolutionState.Evaluating, "Migrate")] = EvolutionState.Migrating;
            _transitions[(EvolutionState.Migrating, "Evaluate")] = EvolutionState.Evaluating;
            _transitions[(EvolutionState.Evaluating, "Complete")] = EvolutionState.Complete;
            _transitions[(EvolutionState.Evolving, "Complete")] = EvolutionState.Complete;
            _transitions[(EvolutionState.Evaluating, "Cancel")] = EvolutionState.Cancelled;
            _transitions[(EvolutionState.Speciating, "Cancel")] = EvolutionState.Cancelled;
            _transitions[(EvolutionState.Selecting, "Cancel")] = EvolutionState.Cancelled;
            _transitions[(EvolutionState.Evolving, "Cancel")] = EvolutionState.Cancelled;
            _transitions[(EvolutionState.Migrating, "Cancel")] = EvolutionState.Cancelled;
            _transitions[(EvolutionState.Evaluating, "Error")] = EvolutionState.Error;
            _transitions[(EvolutionState.Error, "Initialize")] = EvolutionState.Initializing;
            _transitions[(EvolutionState.Complete, "Initialize")] = EvolutionState.Initializing;
            _transitions[(EvolutionState.Cancelled, "Initialize")] = EvolutionState.Initializing;
        }
    }

}
