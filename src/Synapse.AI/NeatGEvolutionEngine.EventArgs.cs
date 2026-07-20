// =============================================================================
// NeatGEvolutionEngine.EventArgs.cs — NEAT-G partial module
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
    /// Event arguments for generation completion.
    /// </summary>
    public sealed class GenerationCompletedEventArgs : EventArgs
    {
        /// <summary>Generation number.</summary>
        public int Generation { get; init; }

        /// <summary>Metrics for this generation.</summary>
        public EvolutionMetrics Metrics { get; init; } = default!;

        /// <summary>Best genome in this generation.</summary>
        public GeoGenome? BestGenome { get; init; }
    }

    /// <summary>
    /// Event arguments for evolution milestones.
    /// </summary>
    public sealed class EvolutionMilestoneEventArgs : EventArgs
    {
        /// <summary>Generation when the milestone was reached.</summary>
        public int Generation { get; init; }

        /// <summary>Best fitness at the milestone.</summary>
        public double BestFitness { get; init; }

        /// <summary>Type of milestone.</summary>
        public EvolutionMilestoneType MilestoneType { get; init; }
    }

    /// <summary>
    /// Types of evolution milestones.
    /// </summary>
    public enum EvolutionMilestoneType
    {
        /// <summary>Population was initialized.</summary>
        PopulationInitialized,

        /// <summary>New best fitness achieved.</summary>
        NewBestFitness,

        /// <summary>Target fitness reached.</summary>
        TargetReached,

        /// <summary>Stagnation detected.</summary>
        StagnationDetected,

        /// <summary>Species extinction event.</summary>
        SpeciesExtinction,

        /// <summary>Significant migration event.</summary>
        SignificantMigration
    }

    /// <summary>
    /// Event arguments for evolution state changes.
    /// </summary>
    public sealed class EvolutionStateChangeEventArgs : EventArgs
    {
        /// <summary>Previous state.</summary>
        public EvolutionState OldState { get; init; }

        /// <summary>New state.</summary>
        public EvolutionState NewState { get; init; }

        /// <summary>Current generation.</summary>
        public int Generation { get; init; }
    }

    /// <summary>
    /// Event arguments for migration events.
    /// </summary>
    public sealed class MigrationEventArgs : EventArgs
    {
        /// <summary>The migration event details.</summary>
        public MigrationEvent Migration { get; init; } = default!;
    }

}
