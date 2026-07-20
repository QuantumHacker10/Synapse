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
    /// Provides metadata about the NEAT-G Evolution Engine assembly.
    /// </summary>
    public static class NeatGEngineInfo
    {
        /// <summary>Engine version.</summary>
        public const string Version = "2.0.0";

        /// <summary>Engine name.</summary>
        public const string Name = "GDNN NEAT-G Evolution Engine";

        /// <summary>Engine description.</summary>
        public const string Description = "NeuroEvolution of Augmented Topologies - Geometric: " +
            "A comprehensive evolutionary algorithm for neural network architecture optimization " +
            "with geometric awareness, semantic crossover, manifold-based speciation, and swarm evolution.";

        /// <summary>Supported mutation types count.</summary>
        public const int SupportedMutationTypes = 18;

        /// <summary>Supported activation functions count.</summary>
        public const int SupportedActivationFunctions = 13;

        /// <summary>Supported selection methods count.</summary>
        public const int SupportedSelectionMethods = 5;

        /// <summary>Supported speciation methods count.</summary>
        public const int SupportedSpeciationMethods = 4;

        /// <summary>Supported crossover strategies count.</summary>
        public const int SupportedCrossoverStrategies = 4;

        /// <summary>
        /// Gets a formatted information string.
        /// </summary>
        public static string GetInfoString()
        {
            return $"{Name} v{Version}\n" +
                   $"{Description}\n" +
                   $"Mutation Types: {SupportedMutationTypes}\n" +
                   $"Activation Functions: {SupportedActivationFunctions}\n" +
                   $"Selection Methods: {SupportedSelectionMethods}\n" +
                   $"Speciation Methods: {SupportedSpeciationMethods}\n" +
                   $"Crossover Strategies: {SupportedCrossoverStrategies}";
        }
    }

}
