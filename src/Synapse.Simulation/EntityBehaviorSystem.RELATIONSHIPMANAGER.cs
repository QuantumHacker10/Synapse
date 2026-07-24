// =============================================================================
// EntityBehaviorSystem.cs
// GDNN.Sentience - Complete Entity Behavior System for G-DNN Engine
// =============================================================================

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace GDNN.Sentience
{

    public class RelationshipManager
    {
        private readonly Dictionary<(Guid, Guid), Relationship> _relationships = new();
        private readonly Dictionary<string, HashSet<Guid>> _groups = new();
        private readonly object _lock = new();

        public void AddRelationship(Relationship relationship)
        {
            lock (_lock)
            {
                var key = GetKey(relationship.EntityA, relationship.EntityB);
                _relationships[key] = relationship;
            }
        }

        public void RemoveRelationship(Guid entityA, Guid entityB)
        {
            lock (_lock)
            { _relationships.Remove(GetKey(entityA, entityB)); }
        }

        public Relationship? GetRelationship(Guid entityA, Guid entityB)
        {
            lock (_lock)
            { return _relationships.TryGetValue(GetKey(entityA, entityB), out var r) ? r : null; }
        }

        public List<Relationship> GetRelationshipsFor(Guid entityId)
        {
            lock (_lock)
            { return _relationships.Values.Where(r => r.EntityA == entityId || r.EntityB == entityId).ToList(); }
        }

        public List<Relationship> GetRelationshipsOfType(RelationshipType type)
        {
            lock (_lock)
            { return _relationships.Values.Where(r => r.Type == type).ToList(); }
        }

        public void UpdateRelationshipStrength(Guid entityA, Guid entityB, float newStrength, string reason = "")
        {
            lock (_lock)
            {
                if (_relationships.TryGetValue(GetKey(entityA, entityB), out var r))
                {
                    var prev = r.Type;
                    r.History.Add(new RelationshipEvent(prev, r.Type, r.Strength, newStrength, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, reason));
                    r.Strength = Math.Clamp(newStrength, 0, 1);
                    if (r.Strength <= 0.01f)
                        r.Type = RelationshipType.Neutral;
                }
            }
        }

        public void ChangeRelationshipType(Guid entityA, Guid entityB, RelationshipType newType, string reason = "")
        {
            lock (_lock)
            {
                if (_relationships.TryGetValue(GetKey(entityA, entityB), out var r))
                {
                    r.History.Add(new RelationshipEvent(r.Type, newType, r.Strength, r.Strength, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond, reason));
                    r.Type = newType;
                }
            }
        }

        public float CalculateInfluencePropagation(Guid source, Guid target, int maxDepth = 3)
        {
            lock (_lock)
            {
                var visited = new HashSet<Guid>();
                return PropagateHelper(source, target, maxDepth, visited);
            }
        }

        private float PropagateHelper(Guid current, Guid target, int depth, HashSet<Guid> visited)
        {
            if (depth <= 0 || !visited.Add(current))
                return 0;
            if (current == target)
                return 1f;

            float maxInfluence = 0;
            foreach (var r in _relationships.Values.Where(r => r.EntityA == current || r.EntityB == current))
            {
                var next = r.EntityA == current ? r.EntityB : r.EntityA;
                float childInfluence = PropagateHelper(next, target, depth - 1, visited) * r.Strength;
                maxInfluence = Math.Max(maxInfluence, childInfluence);
            }
            return maxInfluence;
        }

        public void AddToGroup(Guid entityId, string groupName)
        {
            lock (_lock)
            {
                if (!_groups.TryGetValue(groupName, out var members))
                { members = new HashSet<Guid>(); _groups[groupName] = members; }
                members.Add(entityId);
            }
        }

        public void RemoveFromGroup(Guid entityId, string groupName)
        {
            lock (_lock)
            { if (_groups.TryGetValue(groupName, out var m)) m.Remove(entityId); }
        }

        public List<Guid> GetGroupMembers(string groupName)
        {
            lock (_lock)
            { return _groups.TryGetValue(groupName, out var m) ? m.ToList() : new List<Guid>(); }
        }

        public List<string> GetGroupsFor(Guid entityId)
        {
            lock (_lock)
            { return _groups.Where(kv => kv.Value.Contains(entityId)).Select(kv => kv.Key).ToList(); }
        }

        public bool AreInSameGroup(Guid entityA, Guid entityB)
        {
            lock (_lock)
            { return _groups.Values.Any(g => g.Contains(entityA) && g.Contains(entityB)); }
        }

        public Dictionary<string, float> AnalyzeSocialNetwork(Guid entityId)
        {
            lock (_lock)
            {
                var relationships = GetRelationshipsFor(entityId);
                var analysis = new Dictionary<string, float>
                {
                    { "ConnectionCount", relationships.Count },
                    { "AverageStrength", relationships.Count > 0 ? relationships.Average(r => r.Strength) : 0 },
                    { "StrongestConnection", relationships.Count > 0 ? relationships.Max(r => r.Strength) : 0 },
                    { "WeakestConnection", relationships.Count > 0 ? relationships.Min(r => r.Strength) : 0 },
                    { "GroupCount", GetGroupsFor(entityId).Count },
                    { "CentralityScore", CalculateCentrality(entityId) }
                };
                return analysis;
            }
        }

        private float CalculateCentrality(Guid entityId)
        {
            int totalPairs = _relationships.Count;
            if (totalPairs == 0)
                return 0;
            int involvingEntity = _relationships.Values.Count(r => r.EntityA == entityId || r.EntityB == entityId);
            return (float)involvingEntity / totalPairs;
        }

        public void EvolveRelationships(float deltaTime, float evolutionRate = 0.001f)
        {
            lock (_lock)
            {
                foreach (var r in _relationships.Values)
                {
                    float change = (Random.Shared.NextSingle() - 0.5f) * evolutionRate * deltaTime;
                    r.Strength = Math.Clamp(r.Strength + change, 0, 1);
                    if (r.Strength <= 0.01f)
                        r.Type = RelationshipType.Neutral;
                }
            }
        }

        private static (Guid, Guid) GetKey(Guid a, Guid b) => a < b ? (a, b) : (b, a);
    }

}
