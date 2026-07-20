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
    /// Provides pre-configured evolution settings for common use cases.
    /// </summary>
    public static class EvolutionPresets
    {
        /// <summary>
        /// Configuration for image classification tasks.
        /// Optimized for discovering efficient CNN-like architectures.
        /// </summary>
        public static EvolutionConfig ImageClassification()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 400;
            config.CrossoverRate = 0.7;
            config.MutationRate = 0.3;
            config.SpeciationThreshold = 3.5;
            config.TargetSpeciesCount = 12;
            config.MaxStagnationGenerations = 25;
            config.PerturbationMagnitude = 0.12;
            config.LandmarkCount = 25;
            config.SemanticEmbeddingDimension = 48;
            config.EnableMigration = true;
            config.MigrationRate = 0.06;
            config.MigrationInterval = 8;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 15;
            config.ParentSelection = SelectionMethod.Tournament;
            config.TournamentSize = 6;
            config.SurvivalSelection = SelectionMethod.Truncation;
            config.Objective = FitnessObjective.Maximize;
            return config;
        }

        /// <summary>
        /// Configuration for regression tasks.
        /// Balanced exploration and exploitation.
        /// </summary>
        public static EvolutionConfig Regression()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 300;
            config.CrossoverRate = 0.75;
            config.MutationRate = 0.25;
            config.SpeciationThreshold = 3.0;
            config.TargetSpeciesCount = 10;
            config.MaxStagnationGenerations = 20;
            config.PerturbationMagnitude = 0.1;
            config.LandmarkCount = 20;
            config.SemanticEmbeddingDimension = 32;
            config.EnableMigration = true;
            config.MigrationRate = 0.05;
            config.MigrationInterval = 10;
            config.UseAdaptiveMutation = true;
            config.Objective = FitnessObjective.Maximize;
            return config;
        }

        /// <summary>
        /// Configuration for time series prediction.
        /// Emphasizes recurrent structures and temporal patterns.
        /// </summary>
        public static EvolutionConfig TimeSeries()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 350;
            config.CrossoverRate = 0.65;
            config.MutationRate = 0.35;
            config.SpeciationThreshold = 4.0;
            config.TargetSpeciesCount = 8;
            config.MaxStagnationGenerations = 30;
            config.PerturbationMagnitude = 0.15;
            config.LandmarkCount = 15;
            config.SemanticEmbeddingDimension = 40;
            config.EnableMigration = true;
            config.MigrationRate = 0.08;
            config.MigrationInterval = 7;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 12;
            config.ParentSelection = SelectionMethod.RankBased;
            return config;
        }

        /// <summary>
        /// Configuration for reinforcement learning policy evolution.
        /// High mutation rate for exploration of action spaces.
        /// </summary>
        public static EvolutionConfig ReinforcementLearning()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 500;
            config.CrossoverRate = 0.6;
            config.MutationRate = 0.4;
            config.SpeciationThreshold = 5.0;
            config.TargetSpeciesCount = 15;
            config.MaxStagnationGenerations = 40;
            config.PerturbationMagnitude = 0.2;
            config.LandmarkCount = 30;
            config.SemanticEmbeddingDimension = 64;
            config.EnableMigration = true;
            config.MigrationRate = 0.1;
            config.MigrationInterval = 5;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 20;
            config.MaxAdaptiveMutationRate = 0.6;
            config.ParentSelection = SelectionMethod.Tournament;
            config.TournamentSize = 7;
            return config;
        }

        /// <summary>
        /// Configuration for neural architecture search (NAS).
        /// Focuses on discovering novel topologies.
        /// </summary>
        public static EvolutionConfig NeuralArchitectureSearch()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 600;
            config.CrossoverRate = 0.5;
            config.MutationRate = 0.5;
            config.SpeciationThreshold = 6.0;
            config.TargetSpeciesCount = 20;
            config.MaxStagnationGenerations = 50;
            config.PerturbationMagnitude = 0.25;
            config.LandmarkCount = 40;
            config.SemanticEmbeddingDimension = 64;
            config.EnableMigration = true;
            config.MigrationRate = 0.12;
            config.MigrationInterval = 5;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 25;
            config.MaxAdaptiveMutationRate = 0.7;
            config.ParentSelection = SelectionMethod.StochasticUniversal;
            config.SurvivalSelection = SelectionMethod.Truncation;
            config.EliteFraction = 0.03;
            return config;
        }

        /// <summary>
        /// Configuration for compact/mobile model evolution.
        /// Aggressive parsimony pressure for small models.
        /// </summary>
        public static EvolutionConfig CompactModel()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 200;
            config.CrossoverRate = 0.8;
            config.MutationRate = 0.2;
            config.SpeciationThreshold = 2.5;
            config.TargetSpeciesCount = 6;
            config.MaxStagnationGenerations = 15;
            config.PerturbationMagnitude = 0.08;
            config.LandmarkCount = 15;
            config.SemanticEmbeddingDimension = 24;
            config.EnableMigration = false;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 8;
            config.ParentSelection = SelectionMethod.Tournament;
            config.TournamentSize = 4;
            return config;
        }

        /// <summary>
        /// Configuration for real-time/low-latency evolution.
        /// Small population for fast iteration.
        /// </summary>
        public static EvolutionConfig RealTime()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 100;
            config.MaxGenerations = 500;
            config.CrossoverRate = 0.75;
            config.MutationRate = 0.25;
            config.SpeciationThreshold = 2.0;
            config.TargetSpeciesCount = 5;
            config.MaxStagnationGenerations = 10;
            config.PerturbationMagnitude = 0.08;
            config.LandmarkCount = 10;
            config.SemanticEmbeddingDimension = 16;
            config.EnableMigration = false;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 5;
            config.EvaluationTimeoutMs = 5000;
            config.EnableHistoryTracking = false;
            config.ParentSelection = SelectionMethod.Tournament;
            config.TournamentSize = 3;
            return config;
        }

        /// <summary>
        /// Configuration for exploring completely unknown fitness landscapes.
        /// Maximum diversity and exploration.
        /// </summary>
        public static EvolutionConfig ExploreUnknown()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 800;
            config.CrossoverRate = 0.5;
            config.MutationRate = 0.5;
            config.SpeciationThreshold = 7.0;
            config.TargetSpeciesCount = 25;
            config.MaxStagnationGenerations = 60;
            config.PerturbationMagnitude = 0.3;
            config.LandmarkCount = 50;
            config.SemanticEmbeddingDimension = 64;
            config.EnableMigration = true;
            config.MigrationRate = 0.15;
            config.MigrationInterval = 3;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 30;
            config.MaxAdaptiveMutationRate = 0.8;
            config.MaxSpeciesCount = 100;
            config.ParentSelection = SelectionMethod.StochasticUniversal;
            config.SurvivalSelection = SelectionMethod.Truncation;
            config.EliteFraction = 0.02;
            config.RandomSeed = 42;
            return config;
        }

        /// <summary>
        /// Configuration for fine-tuning a pre-evolved genome.
        /// Very conservative mutations around the existing solution.
        /// </summary>
        public static EvolutionConfig FineTune()
        {
            var config = EvolutionConfig.CreateDefault();
            config.PopulationSize = 150;
            config.CrossoverRate = 0.85;
            config.MutationRate = 0.15;
            config.SpeciationThreshold = 1.5;
            config.TargetSpeciesCount = 3;
            config.MaxStagnationGenerations = 10;
            config.PerturbationMagnitude = 0.03;
            config.PerturbationDecayRate = 0.98;
            config.MinPerturbationMagnitude = 0.0005;
            config.LandmarkCount = 10;
            config.SemanticEmbeddingDimension = 24;
            config.EnableMigration = false;
            config.UseAdaptiveMutation = true;
            config.AdaptiveWindow = 5;
            config.MaxAdaptiveMutationRate = 0.2;
            config.ParentSelection = SelectionMethod.Tournament;
            config.TournamentSize = 3;
            return config;
        }
    }

}
