// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    /// <summary>Directed graph of law dependencies and couplings.</summary>
    public sealed class LawGraph
    {
        private readonly Dictionary<string, List<LawDependency>> _adjacency = new();
        private readonly Dictionary<string, List<LawDependency>> _reverseAdjacency = new();

        /// <summary>Add a dependency between two laws.</summary>
        public void AddDependency(LawDependency dependency)
        {
            if (!_adjacency.TryGetValue(dependency.SourceLawId, out var list))
            {
                list = new List<LawDependency>();
                _adjacency[dependency.SourceLawId] = list;
            }
            list.Add(dependency);

            if (!_reverseAdjacency.TryGetValue(dependency.TargetLawId, out var rList))
            {
                rList = new List<LawDependency>();
                _reverseAdjacency[dependency.TargetLawId] = rList;
            }
            rList.Add(dependency);
        }

        /// <summary>Get all dependencies of a law (laws it depends on).</summary>
        public IReadOnlyList<LawDependency> GetDependencies(string lawId) =>
            _adjacency.TryGetValue(lawId, out var list) ? list : Array.Empty<LawDependency>();

        /// <summary>Get all dependents of a law (laws that depend on it).</summary>
        public IReadOnlyList<LawDependency> GetDependents(string lawId) =>
            _reverseAdjacency.TryGetValue(lawId, out var list) ? list : Array.Empty<LawDependency>();

        /// <summary>Check if a dependency cycle exists.</summary>
        public bool HasCycle()
        {
            var visited = new HashSet<string>();
            var inStack = new HashSet<string>();

            foreach (var node in _adjacency.Keys)
            {
                if (!visited.Contains(node) && HasCycleDFS(node, visited, inStack))
                    return true;
            }
            return false;
        }

        private bool HasCycleDFS(string node, HashSet<string> visited, HashSet<string> inStack)
        {
            visited.Add(node);
            inStack.Add(node);

            if (_adjacency.TryGetValue(node, out var deps))
            {
                foreach (var dep in deps)
                {
                    if (!visited.Contains(dep.TargetLawId))
                    {
                        if (HasCycleDFS(dep.TargetLawId, visited, inStack))
                            return true;
                    }
                    else if (inStack.Contains(dep.TargetLawId))
                        return true;
                }
            }

            inStack.Remove(node);
            return false;
        }

        /// <summary>Topological sort of laws.</summary>
        public List<string> TopologicalSort()
        {
            var result = new List<string>();
            var visited = new HashSet<string>();

            foreach (var node in _adjacency.Keys)
            {
                if (!visited.Contains(node))
                    TopologicalSortDFS(node, visited, result);
            }

            result.Reverse();
            return result;
        }

        private void TopologicalSortDFS(string node, HashSet<string> visited, List<string> result)
        {
            visited.Add(node);
            if (_adjacency.TryGetValue(node, out var deps))
            {
                foreach (var dep in deps)
                {
                    if (!visited.Contains(dep.TargetLawId))
                        TopologicalSortDFS(dep.TargetLawId, visited, result);
                }
            }
            result.Add(node);
        }

        /// <summary>Find all laws in a connected component.</summary>
        public List<string> GetConnectedComponent(string startLawId)
        {
            var component = new HashSet<string>();
            var queue = new Queue<string>();
            queue.Enqueue(startLawId);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                if (!component.Add(current))
                    continue;

                if (_adjacency.TryGetValue(current, out var deps))
                    foreach (var dep in deps)
                        if (!component.Contains(dep.TargetLawId))
                            queue.Enqueue(dep.TargetLawId);

                if (_reverseAdjacency.TryGetValue(current, out var rDeps))
                    foreach (var dep in rDeps)
                        if (!component.Contains(dep.SourceLawId))
                            queue.Enqueue(dep.SourceLawId);
            }

            return component.ToList();
        }

        /// <summary>Get the longest dependency chain from a given law.</summary>
        public int GetLongestChain(string lawId)
        {
            var visited = new HashSet<string>();
            return GetLongestChainDFS(lawId, visited);
        }

        private int GetLongestChainDFS(string node, HashSet<string> visited)
        {
            if (!visited.Add(node))
                return 0;
            int maxDepth = 0;

            if (_adjacency.TryGetValue(node, out var deps))
            {
                foreach (var dep in deps)
                {
                    int depth = GetLongestChainDFS(dep.TargetLawId, visited);
                    if (depth > maxDepth)
                        maxDepth = depth;
                }
            }

            return maxDepth + 1;
        }

        /// <summary>Export the graph as adjacency list.</summary>
        public Dictionary<string, List<string>> ExportAdjacencyList()
        {
            var result = new Dictionary<string, List<string>>();
            foreach (var kv in _adjacency)
            {
                result[kv.Key] = kv.Value.Select(d => d.TargetLawId).ToList();
            }
            return result;
        }

        /// <summary>Count total edges in the graph.</summary>
        public int EdgeCount => _adjacency.Values.Sum(list => list.Count);

        /// <summary>Count total nodes in the graph.</summary>
        public int NodeCount => _adjacency.Keys.Count;
    }
}
