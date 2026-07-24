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

    /// <summary>Tree structure for managing law versions, forks, merges, and diffs.</summary>
    public sealed class LawVersionTree
    {
        private readonly LawVersionNode _root;
        private readonly Dictionary<string, LawVersionNode> _nodes = new();
        private LawVersionNode _current;
        private int _versionCounter;

        public LawVersionNode Root => _root;
        public LawVersionNode Current => _current;
        public int VersionCount => _nodes.Count;

        public LawVersionTree(string initialExpression)
        {
            _root = new LawVersionNode
            {
                VersionId = "v0",
                Expression = initialExpression,
                Description = "Initial version",
                Timestamp = DateTime.UtcNow,
                IsActive = true
            };
            _nodes[_root.VersionId] = _root;
            _current = _root;
            _versionCounter = 1;
        }

        /// <summary>Create a new version from a modified expression.</summary>
        public LawVersionNode Commit(string expression, string description = "")
        {
            var node = new LawVersionNode
            {
                VersionId = $"v{_versionCounter++}",
                Expression = expression,
                Description = description,
                Timestamp = DateTime.UtcNow,
                Parent = _current
            };
            _current.Children.Add(node);
            _nodes[node.VersionId] = node;
            _current = node;
            return node;
        }

        /// <summary>Fork from the current version to create a branch.</summary>
        public LawVersionNode Fork(string expression, string branchName = "")
        {
            var node = new LawVersionNode
            {
                VersionId = $"v{_versionCounter++}_fork",
                Expression = expression,
                Description = $"Fork: {branchName}",
                Timestamp = DateTime.UtcNow,
                Parent = _current
            };
            _current.Children.Add(node);
            _nodes[node.VersionId] = node;
            return node;
        }

        /// <summary>Merge two version branches.</summary>
        public LawVersionNode Merge(string sourceId, string targetId, string mergedExpression)
        {
            if (!_nodes.TryGetValue(sourceId, out var source))
                throw new ArgumentException($"Source version {sourceId} not found");
            if (!_nodes.TryGetValue(targetId, out var target))
                throw new ArgumentException($"Target version {targetId} not found");

            var mergeNode = new LawVersionNode
            {
                VersionId = $"v{_versionCounter++}_merge",
                Expression = mergedExpression,
                Description = $"Merge {sourceId} into {targetId}",
                Timestamp = DateTime.UtcNow,
                Parent = target
            };
            target.Children.Add(mergeNode);
            _nodes[mergeNode.VersionId] = mergeNode;
            _current = mergeNode;
            return mergeNode;
        }

        /// <summary>Rollback to a specific version.</summary>
        public LawVersionNode Rollback(string versionId)
        {
            if (!_nodes.TryGetValue(versionId, out var node))
                throw new ArgumentException($"Version {versionId} not found");
            _current = node;
            return node;
        }

        /// <summary>Rollback to a specific version by index in the path from root.</summary>
        public LawVersionNode RollbackToIndex(int index)
        {
            var path = GetPath(_current);
            if (index < 0 || index >= path.Count)
                throw new ArgumentOutOfRangeException(nameof(index));
            _current = path[index];
            return _current;
        }

        /// <summary>Get the path from root to a specific node.</summary>
        public List<LawVersionNode> GetPath(LawVersionNode? target = null)
        {
            target ??= _current;
            var path = new List<LawVersionNode>();
            var node = target;
            while (node != null)
            { path.Add(node); node = node.Parent; }
            path.Reverse();
            return path;
        }

        /// <summary>Get all versions in the tree (breadth-first).</summary>
        public List<LawVersionNode> GetAllVersions()
        {
            var result = new List<LawVersionNode>();
            var queue = new Queue<LawVersionNode>();
            queue.Enqueue(_root);
            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                result.Add(node);
                foreach (var child in node.Children)
                    queue.Enqueue(child);
            }
            return result;
        }

        /// <summary>Compute Levenshtein edit distance between two strings.</summary>
        public static int ComputeEditDistance(string a, string b)
        {
            int m = a.Length, n = b.Length;
            var dp = new int[m + 1, n + 1];
            for (int i = 0; i <= m; i++)
                dp[i, 0] = i;
            for (int j = 0; j <= n; j++)
                dp[0, j] = j;
            for (int i = 1; i <= m; i++)
                for (int j = 1; j <= n; j++)
                {
                    int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                    dp[i, j] = Math.Min(Math.Min(dp[i - 1, j] + 1, dp[i, j - 1] + 1), dp[i - 1, j - 1] + cost);
                }
            return dp[m, n];
        }

        /// <summary>Compute structural similarity (Jaccard) between two expressions.</summary>
        public static float ComputeStructuralSimilarity(string a, string b)
        {
            var tokensA = TokenizeExpression(a);
            var tokensB = TokenizeExpression(b);
            var setA = new HashSet<string>(tokensA);
            var setB = new HashSet<string>(tokensB);
            if (setA.Count == 0 && setB.Count == 0)
                return 1f;
            int intersection = setA.Intersect(setB).Count();
            int union = setA.Union(setB).Count();
            return union > 0 ? (float)intersection / union : 0f;
        }

        private static List<string> TokenizeExpression(string expr)
        {
            var tokens = new List<string>();
            var sb = new StringBuilder();
            foreach (char c in expr)
            {
                if (char.IsLetterOrDigit(c) || c == '_' || c == '.')
                    sb.Append(c);
                else if (sb.Length > 0)
                { tokens.Add(sb.ToString()); sb.Clear(); }
            }
            if (sb.Length > 0)
                tokens.Add(sb.ToString());
            return tokens;
        }

        /// <summary>Compute edit distance between two version nodes.</summary>
        public int CompareVersions(string versionIdA, string versionIdB)
        {
            if (!_nodes.TryGetValue(versionIdA, out var a))
                throw new ArgumentException($"Version {versionIdA} not found");
            if (!_nodes.TryGetValue(versionIdB, out var b))
                throw new ArgumentException($"Version {versionIdB} not found");
            return ComputeEditDistance(a.Expression, b.Expression);
        }

        /// <summary>Export version history.</summary>
        public List<(string Id, string Expression, DateTime Timestamp, string Description)> ExportHistory()
        {
            return GetAllVersions().OrderBy(n => n.Timestamp)
                .Select(n => (n.VersionId, n.Expression, n.Timestamp, n.Description)).ToList();
        }

        /// <summary>Find the most recent common ancestor of two versions.</summary>
        public LawVersionNode? FindCommonAncestor(string versionIdA, string versionIdB)
        {
            if (!_nodes.TryGetValue(versionIdA, out var a))
                return null;
            if (!_nodes.TryGetValue(versionIdB, out var b))
                return null;
            var ancestorsA = new HashSet<string>();
            var node = a;
            while (node != null)
            { ancestorsA.Add(node.VersionId); node = node.Parent; }
            node = b;
            while (node != null)
            {
                if (ancestorsA.Contains(node.VersionId))
                    return node;
                node = node.Parent;
            }
            return null;
        }

        /// <summary>Get a version node by ID.</summary>
        public LawVersionNode? GetVersion(string versionId)
        {
            return _nodes.TryGetValue(versionId, out var node) ? node : null;
        }

        /// <summary>List all branch tips (leaves).</summary>
        public List<LawVersionNode> GetBranchTips()
        {
            return GetAllVersions().Where(n => n.Children.Count == 0).ToList();
        }

        /// <summary>Get depth of a version from root.</summary>
        public int GetDepth(string versionId)
        {
            return GetPath(GetVersion(versionId)).Count;
        }
    }
}
