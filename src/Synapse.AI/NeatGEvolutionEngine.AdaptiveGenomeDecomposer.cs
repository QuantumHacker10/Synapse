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
    /// Decomposes complex genomes into functional modules using community
    /// detection algorithms on the neural network graph. Enables modular
    /// evolution where independent subnetworks can be evolved separately.
    /// </summary>
    public sealed class AdaptiveGenomeDecomposer
    {
        private readonly EvolutionConfig _config;

        /// <summary>
        /// Initializes a new instance of the AdaptiveGenomeDecomposer class.
        /// </summary>
        /// <param name="config">Evolution configuration.</param>
        public AdaptiveGenomeDecomposer(EvolutionConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Decomposes a genome into functional modules using Louvain community detection.
        /// </summary>
        /// <param name="genome">The genome to decompose.</param>
        /// <returns>List of genome modules.</returns>
        public IReadOnlyList<GenomeModule> Decompose(GeoGenome genome)
        {
            var adjacency = BuildAdjacencyMatrix(genome);
            var communities = LouvainCommunityDetection(adjacency);

            var modules = new List<GenomeModule>();

            foreach (var community in communities)
            {
                var moduleNeurons = community.Select(idx => genome.Neurons[idx]).ToList();
                var moduleNeuronIds = new HashSet<long>(moduleNeurons.Select(n => n.Id));
                var moduleSynapses = genome.Synapses
                    .Where(s => moduleNeuronIds.Contains(s.SourceNeuronId) &&
                                moduleNeuronIds.Contains(s.TargetNeuronId))
                    .ToList();

                var inputs = FindExternalInputs(genome, moduleNeuronIds, community);
                var outputs = FindExternalOutputs(genome, moduleNeuronIds, community);

                modules.Add(new GenomeModule
                {
                    Id = modules.Count,
                    Neurons = moduleNeurons.AsReadOnly(),
                    Synapses = moduleSynapses.AsReadOnly(),
                    ExternalInputs = inputs.AsReadOnly(),
                    ExternalOutputs = outputs.AsReadOnly(),
                    Modularity = ComputeModuleModularity(genome, community, moduleNeuronIds),
                    ModuleSize = moduleNeurons.Count,
                    ConnectionDensity = ComputeModuleDensity(genome, moduleNeuronIds),
                    IsInterfacing = inputs.Count > 0 || outputs.Count > 0
                });
            }

            return modules.AsReadOnly();
        }

        /// <summary>
        /// Identifies critical modules whose modification would significantly
        /// impact overall genome fitness.
        /// </summary>
        /// <param name="genome">The genome.</param>
        /// <param name="modules">Decomposed modules.</param>
        /// <param name="fitnessEvaluator">Fitness evaluator for ablation testing.</param>
        /// <returns>Criticality scores for each module.</returns>
        public IReadOnlyList<ModuleCriticality> IdentifyCriticalModules(
            GeoGenome genome,
            IReadOnlyList<GenomeModule> modules,
            IFitnessEvaluator fitnessEvaluator)
        {
            double baselineFitness = genome.Fitness;
            var criticalities = new List<ModuleCriticality>();

            foreach (var module in modules)
            {
                var ablatedGenome = CreateAblatedGenome(genome, module);
                double ablatedFitness = fitnessEvaluator.Evaluate(ablatedGenome);

                double fitnessDrop = baselineFitness - ablatedFitness;
                double relativeDrop = baselineFitness > 0 ? fitnessDrop / baselineFitness : 0;

                criticalities.Add(new ModuleCriticality
                {
                    ModuleId = module.Id,
                    FitnessDrop = fitnessDrop,
                    RelativeFitnessDrop = relativeDrop,
                    IsCritical = relativeDrop > 0.1,
                    IsRedundant = relativeDrop < 0.01,
                    CriticalityScore = Math.Tanh(relativeDrop * 5.0)
                });
            }

            return criticalities.AsReadOnly();
        }

        /// <summary>
        /// Merges similar modules that have high structural overlap.
        /// </summary>
        /// <param name="modules">List of modules.</param>
        /// <param name="threshold">Similarity threshold for merging.</param>
        /// <returns>Merged module list.</returns>
        public IReadOnlyList<GenomeModule> MergeSimilarModules(
            IReadOnlyList<GenomeModule> modules,
            double threshold = 0.7)
        {
            var merged = new List<GenomeModule>(modules);
            var mergeMap = new Dictionary<int, int>();

            for (int i = 0; i < merged.Count; i++)
            {
                if (mergeMap.ContainsKey(i))
                    continue;

                for (int j = i + 1; j < merged.Count; j++)
                {
                    if (mergeMap.ContainsKey(j))
                        continue;

                    double similarity = ComputeModuleSimilarity(merged[i], merged[j]);
                    if (similarity >= threshold)
                    {
                        int targetIdx = mergeMap.ContainsKey(i) ? mergeMap[i] : i;
                        mergeMap[j] = targetIdx;
                    }
                }
            }

            if (mergeMap.Count == 0)
                return modules;

            var mergedModules = new Dictionary<int, List<GenomeModule>>();
            for (int i = 0; i < merged.Count; i++)
            {
                int target = mergeMap.ContainsKey(i) ? mergeMap[i] : i;
                if (!mergedModules.ContainsKey(target))
                    mergedModules[target] = new List<GenomeModule>();
                mergedModules[target].Add(merged[i]);
            }

            var result = new List<GenomeModule>();
            foreach (var kvp in mergedModules)
            {
                if (kvp.Value.Count == 1)
                {
                    result.Add(kvp.Value[0]);
                }
                else
                {
                    result.Add(MergeModules(kvp.Value));
                }
            }

            return result.AsReadOnly();
        }

        /// <summary>
        /// Extracts a sub-genome containing only the specified module.
        /// </summary>
        /// <param name="genome">Source genome.</param>
        /// <param name="module">Module to extract.</param>
        /// <returns>Sub-genome with module neurons and synapses.</returns>
        public GeoGenome ExtractModuleAsGenome(GeoGenome genome, GenomeModule module)
        {
            var moduleNeuronIds = new HashSet<long>(module.Neurons.Select(n => n.Id));

            var subNeurons = module.Neurons.Select(n =>
            {
                var clone = n.Clone();
                if (moduleNeuronIds.Contains(n.Id))
                {
                    clone.LayerIndex = 0;
                }
                return clone;
            }).ToList();

            var subSynapses = module.Synapses.Select(s =>
            {
                var clone = s.Clone();
                int srcIdx = subNeurons.FindIndex(n => n.Id == clone.SourceNeuronId);
                int tgtIdx = subNeurons.FindIndex(n => n.Id == clone.TargetNeuronId);
                if (srcIdx >= 0 && tgtIdx >= 0)
                {
                    clone.SourceNeuronId = srcIdx;
                    clone.TargetNeuronId = tgtIdx;
                }
                return clone;
            }).ToList();

            return new GeoGenome
            {
                Id = Guid.NewGuid(),
                Neurons = subNeurons,
                Synapses = subSynapses,
                Fitness = genome.Fitness * (module.ModuleSize / (double)Math.Max(1, genome.ActiveNeuronCount))
            };
        }

        private int[,] BuildAdjacencyMatrix(GeoGenome genome)
        {
            int n = genome.Neurons.Count;
            var adjacency = new int[n, n];
            var idToIndex = new Dictionary<long, int>(n);
            for (int i = 0; i < n; i++)
                idToIndex[genome.Neurons[i].Id] = i;

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (idToIndex.TryGetValue(synapse.SourceNeuronId, out int si) &&
                    idToIndex.TryGetValue(synapse.TargetNeuronId, out int ti))
                {
                    adjacency[si, ti] = 1;
                    adjacency[ti, si] = 1;
                }
            }

            return adjacency;
        }

        private List<List<int>> LouvainCommunityDetection(int[,] adjacency)
        {
            int n = adjacency.GetLength(0);
            var communities = Enumerable.Range(0, n).Select(i => new List<int> { i }).ToList();
            var membership = Enumerable.Range(0, n).ToArray();
            int totalEdges = 0;

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    totalEdges += adjacency[i, j];

            if (totalEdges == 0)
            {
                return Enumerable.Range(0, n)
                    .Select(i => new List<int> { i })
                    .ToList();
            }

            bool improved = true;
            int maxIterations = 100;
            int iteration = 0;

            while (improved && iteration < maxIterations)
            {
                improved = false;
                iteration++;

                for (int i = 0; i < n; i++)
                {
                    int currentCommunity = membership[i];
                    int bestCommunity = currentCommunity;
                    double bestGain = 0;

                    var neighborCommunities = new HashSet<int>();
                    for (int j = 0; j < n; j++)
                    {
                        if (adjacency[i, j] > 0)
                            neighborCommunities.Add(membership[j]);
                    }

                    foreach (int neighborComm in neighborCommunities)
                    {
                        if (neighborComm == currentCommunity)
                            continue;

                        double gain = ComputeModularityGain(adjacency, membership, i, currentCommunity, neighborComm, n, totalEdges);
                        if (gain > bestGain)
                        {
                            bestGain = gain;
                            bestCommunity = neighborComm;
                        }
                    }

                    if (bestCommunity != currentCommunity)
                    {
                        membership[i] = bestCommunity;
                        improved = true;
                    }
                }
            }

            var result = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                int comm = membership[i];
                if (!result.ContainsKey(comm))
                    result[comm] = new List<int>();
                result[comm].Add(i);
            }

            return result.Values.ToList();
        }

        private double ComputeModularityGain(int[,] adjacency, int[] membership, int node, int fromComm, int toComm, int n, int totalEdges)
        {
            double m2 = 2.0 * totalEdges;
            if (m2 == 0)
                return 0;

            double ki = 0;
            double kiTo = 0;
            double sigmaTo = 0;
            double sigmaIn = 0;

            for (int j = 0; j < n; j++)
            {
                ki += adjacency[node, j];
                if (membership[j] == toComm)
                {
                    kiTo += adjacency[node, j];
                    for (int k = 0; k < n; k++)
                        sigmaTo += adjacency[j, k];
                }
                if (membership[j] == fromComm)
                {
                    for (int k = 0; k < n; k++)
                        sigmaIn += adjacency[j, k];
                }
            }

            double gain = (kiTo - sigmaTo * ki / m2) - (adjacency[node, node] - sigmaIn * ki / m2);
            return gain / m2;
        }

        private List<long> FindExternalInputs(GeoGenome genome, HashSet<long> moduleNeuronIds, List<int> communityIndices)
        {
            var inputs = new HashSet<long>();

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (moduleNeuronIds.Contains(synapse.TargetNeuronId) &&
                    !moduleNeuronIds.Contains(synapse.SourceNeuronId))
                {
                    inputs.Add(synapse.SourceNeuronId);
                }
            }

            return inputs.ToList();
        }

        private List<long> FindExternalOutputs(GeoGenome genome, HashSet<long> moduleNeuronIds, List<int> communityIndices)
        {
            var outputs = new HashSet<long>();

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (moduleNeuronIds.Contains(synapse.SourceNeuronId) &&
                    !moduleNeuronIds.Contains(synapse.TargetNeuronId))
                {
                    outputs.Add(synapse.TargetNeuronId);
                }
            }

            return outputs.ToList();
        }

        private double ComputeModuleModularity(GeoGenome genome, List<int> communityIndices, HashSet<long> moduleNeuronIds)
        {
            int internalEdges = 0;
            int totalEdges = genome.Synapses.Count(s => s.IsActive);

            foreach (var synapse in genome.Synapses.Where(s => s.IsActive))
            {
                if (moduleNeuronIds.Contains(synapse.SourceNeuronId) &&
                    moduleNeuronIds.Contains(synapse.TargetNeuronId))
                {
                    internalEdges++;
                }
            }

            return totalEdges > 0 ? (double)internalEdges / totalEdges : 0;
        }

        private double ComputeModuleDensity(GeoGenome genome, HashSet<long> moduleNeuronIds)
        {
            int n = moduleNeuronIds.Count;
            if (n <= 1)
                return 0;

            int internalEdges = genome.Synapses.Count(s =>
                s.IsActive &&
                moduleNeuronIds.Contains(s.SourceNeuronId) &&
                moduleNeuronIds.Contains(s.TargetNeuronId));

            int maxPossible = n * (n - 1) / 2;
            return maxPossible > 0 ? (double)internalEdges / maxPossible : 0;
        }

        private double ComputeModuleSimilarity(GenomeModule a, GenomeModule b)
        {
            var aNeuronIds = new HashSet<long>(a.Neurons.Select(n => n.Id));
            var bNeuronIds = new HashSet<long>(b.Neurons.Select(n => n.Id));

            int intersection = aNeuronIds.Intersect(bNeuronIds).Count();
            int union = aNeuronIds.Union(bNeuronIds).Count();

            return union > 0 ? (double)intersection / union : 0;
        }

        private GenomeModule MergeModules(List<GenomeModule> modules)
        {
            var allNeurons = modules.SelectMany(m => m.Neurons).ToList();
            var allSynapses = modules.SelectMany(m => m.Synapses).ToList();
            var allInputs = modules.SelectMany(m => m.ExternalInputs).Distinct().ToList();
            var allOutputs = modules.SelectMany(m => m.ExternalOutputs).Distinct().ToList();

            int totalNeuronCount = allNeurons.Count;
            int uniqueNeuronCount = allNeurons.Select(n => n.Id).Distinct().Count();

            return new GenomeModule
            {
                Id = modules.Min(m => m.Id),
                Neurons = allNeurons.DistinctBy(n => n.Id).ToList().AsReadOnly(),
                Synapses = allSynapses.DistinctBy(s => s.Id).ToList().AsReadOnly(),
                ExternalInputs = allInputs.AsReadOnly(),
                ExternalOutputs = allOutputs.AsReadOnly(),
                Modularity = modules.Average(m => m.Modularity),
                ModuleSize = uniqueNeuronCount,
                ConnectionDensity = modules.Average(m => m.ConnectionDensity),
                IsInterfacing = allInputs.Count > 0 || allOutputs.Count > 0
            };
        }

        private GeoGenome CreateAblatedGenome(GeoGenome genome, GenomeModule module)
        {
            var moduleNeuronIds = new HashSet<long>(module.Neurons.Select(n => n.Id));

            var ablatedNeurons = genome.Neurons.Select(n =>
            {
                if (moduleNeuronIds.Contains(n.Id))
                {
                    var clone = n.Clone();
                    clone.IsActive = false;
                    return clone;
                }
                return n;
            }).ToList();

            var ablatedSynapses = genome.Synapses.Select(s =>
            {
                if (moduleNeuronIds.Contains(s.SourceNeuronId) ||
                    moduleNeuronIds.Contains(s.TargetNeuronId))
                {
                    var clone = s.Clone();
                    clone.IsActive = false;
                    return clone;
                }
                return s;
            }).ToList();

            return new GeoGenome
            {
                Id = genome.Id,
                Neurons = ablatedNeurons,
                Synapses = ablatedSynapses,
                Fitness = genome.Fitness
            };
        }
    }

    /// <summary>
    /// Represents a functional module within a genome.
    /// </summary>
    public sealed class GenomeModule
    {
        /// <summary>Module ID.</summary>
        public int Id { get; init; }
        /// <summary>Neurons in this module.</summary>
        public IReadOnlyList<GeoNeuron> Neurons { get; init; } = Array.Empty<GeoNeuron>();
        /// <summary>Synapses within this module.</summary>
        public IReadOnlyList<GeoSynapse> Synapses { get; init; } = Array.Empty<GeoSynapse>();
        /// <summary>External input neuron IDs.</summary>
        public IReadOnlyList<long> ExternalInputs { get; init; } = Array.Empty<long>();
        /// <summary>External output neuron IDs.</summary>
        public IReadOnlyList<long> ExternalOutputs { get; init; } = Array.Empty<long>();
        /// <summary>Module modularity score.</summary>
        public double Modularity { get; init; }
        /// <summary>Number of neurons in module.</summary>
        public int ModuleSize { get; init; }
        /// <summary>Internal connection density.</summary>
        public double ConnectionDensity { get; init; }
        /// <summary>Whether this module interfaces with other modules.</summary>
        public bool IsInterfacing { get; init; }
    }

    /// <summary>
    /// Criticality analysis result for a genome module.
    /// </summary>
    public sealed class ModuleCriticality
    {
        /// <summary>Module ID.</summary>
        public int ModuleId { get; init; }
        /// <summary>Absolute fitness drop when module is ablated.</summary>
        public double FitnessDrop { get; init; }
        /// <summary>Relative fitness drop as fraction of baseline.</summary>
        public double RelativeFitnessDrop { get; init; }
        /// <summary>Whether the module is critical (>10% fitness drop).</summary>
        public bool IsCritical { get; init; }
        /// <summary>Whether the module is redundant (<1% fitness drop).</summary>
        public bool IsRedundant { get; init; }
        /// <summary>Normalized criticality score [0, 1].</summary>
        public double CriticalityScore { get; init; }
    }

}
