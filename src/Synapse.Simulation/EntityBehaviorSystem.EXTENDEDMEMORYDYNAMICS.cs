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

    public class MemoryConsolidationEngine
    {
        private readonly Dictionary<Guid, MemoryConsolidationState> _consolidationStates = new();

        public float ConsolidationThreshold { get; set; } = 0.5f;
        public float RehearsalBoost { get; set; } = 0.15f;
        public float InterferenceDecay { get; set; } = 0.05f;
        public float EmotionalAmplification { get; set; } = 0.3f;

        public void ProcessConsolidation(SentientEntity entity, IMemorySystem memorySystem, float deltaTime)
        {
            var state = GetOrCreate(entity.EntityId);
            state.TimeSinceLastRehearsal += deltaTime;
            if (state.TimeSinceLastRehearsal >= state.NextRehearsalInterval)
            {
                state.NeedsRehearsal = true;
                state.TimeSinceLastRehearsal = 0;
                state.RehearsalCount++;
                state.NextRehearsalInterval = Math.Max(5f, 30f * (float)Math.Pow(2.5, state.RehearsalCount));
            }
            if (state.NeedsRehearsal)
            {
                state.NeedsRehearsal = false;
                var query = new MemoryQuery { MaxResults = 10, SortByEmotionalIntensity = true, MinImportance = 0.3f };
                var memories = memorySystem.Retrieve(entity, query);
                foreach (var m in memories)
                {
                    m.ConsolidationStrength = Math.Min(1f, m.ConsolidationStrength + RehearsalBoost);
                    m.RetrievalCount++;
                    m.LastRetrieved = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
                    if (m.ConsolidationStrength >= ConsolidationThreshold && !m.IsConsolidated)
                    { m.IsConsolidated = true; state.ConsolidatedMemoryCount++; }
                }
            }
            state.GlobalConsolidationRate = Math.Clamp(state.GlobalConsolidationRate + (state.ConsolidatedMemoryCount - state.ForgottenMemoryCount) * 0.01f * deltaTime, 0.01f, 1f);
            foreach (var entry in state.SpacedRepetitionSchedule.ToList())
            {
                entry.ElapsedTime += deltaTime;
                if (entry.ElapsedTime >= entry.NextReviewTime)
                {
                    entry.ReviewCount++;
                    entry.ElapsedTime = 0;
                    entry.NextReviewTime *= entry.EaseFactor;
                    entry.EaseFactor = Math.Max(1.3f, entry.EaseFactor + 0.1f);
                }
            }
        }

        public MemoryConsolidationState? GetState(Guid eid) => _consolidationStates.TryGetValue(eid, out var s) ? s : null;

        private MemoryConsolidationState GetOrCreate(Guid eid)
        {
            if (!_consolidationStates.TryGetValue(eid, out var s))
            { s = new MemoryConsolidationState(); _consolidationStates[eid] = s; }
            return s;
        }
    }

    public class MemoryConsolidationState
    {
        public float TimeSinceLastRehearsal { get; set; }
        public float NextRehearsalInterval { get; set; } = 30f;
        public bool NeedsRehearsal { get; set; }
        public int RehearsalCount { get; set; }
        public int ConsolidatedMemoryCount { get; set; }
        public int ForgottenMemoryCount { get; set; }
        public float GlobalConsolidationRate { get; set; } = 0.5f;
        public List<SpacedRepetitionEntry> SpacedRepetitionSchedule { get; set; } = new();
    }

    public class SpacedRepetitionEntry
    {
        public Guid MemoryId { get; set; }
        public int ReviewCount { get; set; }
        public float EaseFactor { get; set; } = 2.5f;
        public float NextReviewTime { get; set; } = 60f;
        public float ElapsedTime { get; set; }
    }

    public class SemanticNetworkBuilder
    {
        private readonly Dictionary<string, List<string>> _graph = new();

        public void AddConcept(string concept, List<string> related)
        {
            if (!_graph.TryGetValue(concept, out var e))
            { e = new List<string>(); _graph[concept] = e; }
            foreach (var r in related)
            {
                if (!e.Contains(r))
                    e.Add(r);
                if (!_graph.TryGetValue(r, out var rev))
                { rev = new List<string>(); _graph[r] = rev; }
                if (!rev.Contains(concept))
                    rev.Add(concept);
            }
        }

        public List<string> FindRelated(string concept, int depth = 2)
        {
            var result = new HashSet<string>();
            var queue = new Queue<(string, int)>();
            queue.Enqueue((concept, 0));
            while (queue.Count > 0)
            {
                var (cur, d) = queue.Dequeue();
                if (d >= depth)
                    continue;
                if (_graph.TryGetValue(cur, out var neighbors))
                    foreach (var n in neighbors)
                        if (result.Add(n) && n != concept)
                            queue.Enqueue((n, d + 1));
            }
            return result.ToList();
        }

        public float CalculateDistance(string a, string b)
        {
            var relatedA = FindRelated(a, 3);
            if (relatedA.Contains(b))
                return 0.5f;
            var relatedB = FindRelated(b, 3);
            int shared = relatedA.Intersect(relatedB).Count();
            int total = relatedA.Union(relatedB).Count();
            return total > 0 ? 1f - (float)shared / total : 1f;
        }

        public List<string> GetConcepts() => _graph.Keys.ToList();
        public int ConceptCount => _graph.Count;
        public int LinkCount => _graph.Values.Sum(v => v.Count);
    }

}
