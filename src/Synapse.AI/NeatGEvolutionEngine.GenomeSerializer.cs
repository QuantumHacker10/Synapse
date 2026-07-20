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
    /// Handles serialization and deserialization of genomes for persistence,
    /// transfer, and analysis. Supports JSON and binary formats.
    /// </summary>
    public sealed class GenomeSerializer
    {
        private readonly JsonSerializerOptions _jsonOptions;

        /// <summary>
        /// Initializes a new instance of the GenomeSerializer class.
        /// </summary>
        public GenomeSerializer()
        {
            _jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = false,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
                PropertyNameCaseInsensitive = true
            };
        }

        /// <summary>
        /// Serializes a genome to a JSON string.
        /// </summary>
        /// <param name="genome">The genome to serialize.</param>
        /// <returns>JSON string representation.</returns>
        public string SerializeToJson(GeoGenome genome)
        {
            var data = new GenomeData
            {
                Id = genome.Id.ToString(),
                Generation = genome.Generation,
                SpeciesId = genome.SpeciesId,
                Age = genome.Age,
                InputCount = genome.InputCount,
                OutputCount = genome.OutputCount,
                Fitness = genome.Fitness,
                BestFitness = genome.BestFitness,
                Neurons = genome.Neurons.Select(n => new NeuronData
                {
                    InnovationNumber = n.InnovationNumber,
                    LayerIndex = n.LayerIndex,
                    PositionInLayer = n.PositionInLayer,
                    Activation = (int)n.Activation,
                    Bias = n.Bias,
                    IsActive = n.IsActive,
                    SemanticRole = n.SemanticRole,
                    CreationGeneration = n.CreationGeneration,
                    ExpressionCount = n.ExpressionCount
                }).ToList(),
                Synapses = genome.Synapses.Select(s => new SynapseData
                {
                    InnovationNumber = s.InnovationNumber,
                    SourceNeuronId = s.SourceNeuronId,
                    TargetNeuronId = s.TargetNeuronId,
                    Weight = s.Weight,
                    IsActive = s.IsActive,
                    IsRecurrent = s.IsRecurrent,
                    RecurrentDelay = s.RecurrentDelay,
                    CreationGeneration = s.CreationGeneration,
                    Confidence = s.Confidence,
                    SemanticRole = s.SemanticRole
                }).ToList(),
                ParentIds = genome.ParentIds.Select(id => id.ToString()).ToList()
            };

            return JsonSerializer.Serialize(data, _jsonOptions);
        }

        /// <summary>
        /// Deserializes a genome from a JSON string.
        /// </summary>
        /// <param name="json">JSON string representation.</param>
        /// <returns>The deserialized genome.</returns>
        public GeoGenome DeserializeFromJson(string json)
        {
            var data = JsonSerializer.Deserialize<GenomeData>(json, _jsonOptions)
                ?? throw new ArgumentException("Invalid JSON data for genome deserialization.");

            var genome = new GeoGenome
            {
                Id = Guid.Parse(data.Id),
                Generation = data.Generation,
                SpeciesId = data.SpeciesId,
                Age = data.Age,
                InputCount = data.InputCount,
                OutputCount = data.OutputCount,
                Fitness = data.Fitness,
                BestFitness = data.BestFitness,
                ParentIds = data.ParentIds != null
                    ? data.ParentIds.Select(id => Guid.Parse(id)).ToImmutableArray()
                    : ImmutableArray<Guid>.Empty
            };

            foreach (var nData in data.Neurons)
            {
                genome.Neurons.Add(new GeoNeuron
                {
                    InnovationNumber = nData.InnovationNumber,
                    LayerIndex = nData.LayerIndex,
                    PositionInLayer = nData.PositionInLayer,
                    Activation = (ActivationFunction)nData.Activation,
                    Bias = nData.Bias,
                    IsActive = nData.IsActive,
                    SemanticRole = nData.SemanticRole,
                    CreationGeneration = nData.CreationGeneration,
                    ExpressionCount = nData.ExpressionCount
                });
            }

            foreach (var sData in data.Synapses)
            {
                genome.Synapses.Add(new GeoSynapse
                {
                    InnovationNumber = sData.InnovationNumber,
                    SourceNeuronId = sData.SourceNeuronId,
                    TargetNeuronId = sData.TargetNeuronId,
                    Weight = sData.Weight,
                    IsActive = sData.IsActive,
                    IsRecurrent = sData.IsRecurrent,
                    RecurrentDelay = sData.RecurrentDelay,
                    CreationGeneration = sData.CreationGeneration,
                    Confidence = sData.Confidence,
                    SemanticRole = sData.SemanticRole
                });
            }

            return genome;
        }

        /// <summary>
        /// Serializes a genome to a compact binary format.
        /// </summary>
        /// <param name="genome">The genome to serialize.</param>
        /// <returns>Binary data as byte array.</returns>
        public byte[] SerializeToBinary(GeoGenome genome)
        {
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            writer.Write(genome.Id.ToByteArray());
            writer.Write(genome.Generation);
            writer.Write(genome.SpeciesId);
            writer.Write(genome.Age);
            writer.Write(genome.InputCount);
            writer.Write(genome.OutputCount);
            writer.Write(genome.Fitness);

            var activeNeurons = genome.Neurons.Where(n => n.IsActive).ToList();
            var activeSynapses = genome.Synapses.Where(s => s.IsActive).ToList();

            writer.Write(activeNeurons.Count);
            foreach (var neuron in activeNeurons)
            {
                writer.Write(neuron.InnovationNumber);
                writer.Write(neuron.LayerIndex);
                writer.Write(neuron.PositionInLayer);
                writer.Write((int)neuron.Activation);
                writer.Write(neuron.Bias);
            }

            writer.Write(activeSynapses.Count);
            foreach (var synapse in activeSynapses)
            {
                writer.Write(synapse.InnovationNumber);
                writer.Write(synapse.SourceNeuronId);
                writer.Write(synapse.TargetNeuronId);
                writer.Write(synapse.Weight);
                writer.Write(synapse.IsRecurrent);
            }

            return ms.ToArray();
        }

        /// <summary>
        /// Deserializes a genome from binary data.
        /// </summary>
        /// <param name="data">Binary data.</param>
        /// <returns>The deserialized genome.</returns>
        public GeoGenome DeserializeFromBinary(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);

            var genome = new GeoGenome
            {
                Id = new Guid(reader.ReadBytes(16)),
                Generation = reader.ReadInt32(),
                SpeciesId = reader.ReadInt32(),
                Age = reader.ReadInt32(),
                InputCount = reader.ReadInt32(),
                OutputCount = reader.ReadInt32(),
                Fitness = reader.ReadDouble()
            };

            int neuronCount = reader.ReadInt32();
            for (int i = 0; i < neuronCount; i++)
            {
                genome.Neurons.Add(new GeoNeuron
                {
                    InnovationNumber = reader.ReadInt64(),
                    LayerIndex = reader.ReadInt32(),
                    PositionInLayer = reader.ReadInt32(),
                    Activation = (ActivationFunction)reader.ReadInt32(),
                    Bias = reader.ReadDouble(),
                    IsActive = true
                });
            }

            int synapseCount = reader.ReadInt32();
            for (int i = 0; i < synapseCount; i++)
            {
                genome.Synapses.Add(new GeoSynapse
                {
                    InnovationNumber = reader.ReadInt64(),
                    SourceNeuronId = reader.ReadInt64(),
                    TargetNeuronId = reader.ReadInt64(),
                    Weight = reader.ReadDouble(),
                    IsActive = true,
                    IsRecurrent = reader.ReadBoolean()
                });
            }

            return genome;
        }

        /// <summary>
        /// Serializes a population to JSON.
        /// </summary>
        /// <param name="population">The population.</param>
        /// <returns>JSON string.</returns>
        public string SerializePopulationToJson(GenomePopulation population)
        {
            var genomeJsons = population.Genomes.Select(SerializeToJson).ToList();
            return JsonSerializer.Serialize(new
            {
                Generation = population.GenerationNumber,
                Count = population.Genomes.Length,
                Genomes = genomeJsons
            }, _jsonOptions);
        }

        /// <summary>
        /// Computes a checksum for a genome for integrity verification.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <returns>Checksum as a hex string.</returns>
        public string ComputeChecksum(GeoGenome genome)
        {
            var json = SerializeToJson(genome);
            using var sha256 = System.Security.Cryptography.SHA256.Create();
            var hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(json));
            return Convert.ToHexString(hash);
        }

        /// <summary>
        /// Internal data class for JSON serialization.
        /// </summary>
        private sealed class GenomeData
        {
            public string Id { get; set; } = string.Empty;
            public int Generation { get; set; }
            public int SpeciesId { get; set; }
            public int Age { get; set; }
            public int InputCount { get; set; }
            public int OutputCount { get; set; }
            public double Fitness { get; set; }
            public double BestFitness { get; set; }
            public List<NeuronData> Neurons { get; set; } = new();
            public List<SynapseData> Synapses { get; set; } = new();
            public List<string> ParentIds { get; set; } = new();
        }

        private sealed class NeuronData
        {
            public long InnovationNumber { get; set; }
            public int LayerIndex { get; set; }
            public int PositionInLayer { get; set; }
            public int Activation { get; set; }
            public double Bias { get; set; }
            public bool IsActive { get; set; }
            public string? SemanticRole { get; set; }
            public int CreationGeneration { get; set; }
            public int ExpressionCount { get; set; }
        }

        private sealed class SynapseData
        {
            public long InnovationNumber { get; set; }
            public long SourceNeuronId { get; set; }
            public long TargetNeuronId { get; set; }
            public double Weight { get; set; }
            public bool IsActive { get; set; }
            public bool IsRecurrent { get; set; }
            public int RecurrentDelay { get; set; }
            public int CreationGeneration { get; set; }
            public double Confidence { get; set; }
            public string? SemanticRole { get; set; }
        }
    }

}
