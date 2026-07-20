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
    /// Provides callback hooks for monitoring and controlling the evolution process.
    /// Allows injection of custom logic at various points in the evolution loop.
    /// </summary>
    public sealed class EvolutionCallbacks
    {
        /// <summary>
        /// Callback invoked before each generation starts.
        /// Return false to skip this generation.
        /// </summary>
        public Func<int, GenomePopulation, bool>? OnBeforeGeneration { get; set; }

        /// <summary>
        /// Callback invoked after each generation completes.
        /// </summary>
        public Action<int, GenomePopulation, EvolutionMetrics>? OnAfterGeneration { get; set; }

        /// <summary>
        /// Callback invoked when new offspring are produced.
        /// Can modify offspring before evaluation.
        /// </summary>
        public Func<IReadOnlyList<GeoGenome>, IReadOnlyList<GeoGenome>>? OnOffspringProduced { get; set; }

        /// <summary>
        /// Callback invoked before fitness evaluation.
        /// Can modify the evaluation context.
        /// </summary>
        public Func<EvaluationContext, EvaluationContext>? OnBeforeEvaluation { get; set; }

        /// <summary>
        /// Callback invoked after fitness evaluation.
        /// Can modify genome fitness values.
        /// </summary>
        public Action<GeoGenome>? OnAfterEvaluation { get; set; }

        /// <summary>
        /// Callback invoked when speciation occurs.
        /// Can modify species assignments.
        /// </summary>
        public Func<ImmutableArray<SpeciesInfo>, ImmutableArray<SpeciesInfo>>? OnAfterSpeciation { get; set; }

        /// <summary>
        /// Callback invoked when a new best fitness is achieved.
        /// </summary>
        public Action<int, GeoGenome>? OnNewBestFitness { get; set; }

        /// <summary>
        /// Callback invoked when stagnation is detected.
        /// Can trigger custom recovery mechanisms.
        /// </summary>
        public Func<int, bool>? OnStagnationDetected { get; set; }

        /// <summary>
        /// Callback invoked when mutation rates are adjusted.
        /// </summary>
        public Action<double, double>? OnMutationRateAdjusted { get; set; }

        /// <summary>
        /// Callback invoked when migration occurs.
        /// </summary>
        public Action<MigrationEvent>? OnMigration { get; set; }

        /// <summary>
        /// Callback invoked when a species is created.
        /// </summary>
        public Action<SpeciesInfo>? OnSpeciesCreated { get; set; }

        /// <summary>
        /// Callback invoked when a species goes extinct.
        /// </summary>
        public Action<SpeciesInfo>? OnSpeciesExtinct { get; set; }

        /// <summary>
        /// Callback for custom genome initialization.
        /// Can provide domain-specific initial genomes.
        /// </summary>
        public Func<int, int, GeoGenome>? OnInitializeGenome { get; set; }

        /// <summary>
        /// Callback invoked for progress reporting.
        /// </summary>
        public IProgress<EvolutionMetrics>? ProgressReporter { get; set; }

        /// <summary>
        /// Callback for custom fitness post-processing.
        /// </summary>
        public Func<GeoGenome, GeoGenome>? OnFitnessPostProcess { get; set; }

        /// <summary>
        /// Callback for monitoring resource usage.
        /// </summary>
        public Action<long, TimeSpan>? OnResourceCheck { get; set; }
    }

}
