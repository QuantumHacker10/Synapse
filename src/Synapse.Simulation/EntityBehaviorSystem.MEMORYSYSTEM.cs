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

    public class MemorySystem : IMemorySystem
    {
        private readonly Dictionary<Guid, EntityMemoryBank> _entityMemories = new();

        public int ShortTermCapacity { get; set; } = 7;
        public float ShortTermDuration { get; set; } = 30f;
        public float ConsolidationThreshold { get; set; } = 0.3f;
        public float DecayRate { get; set; } = 0.01f;

        private EntityMemoryBank GetOrCreate(Guid eid)
        {
            if (!_entityMemories.TryGetValue(eid, out var b))
            { b = new EntityMemoryBank(); _entityMemories[eid] = b; }
            return b;
        }

        public void Store(SentientEntity entity, MemoryEntry memory)
        {
            var bank = GetOrCreate(entity.EntityId);
            bank.ShortTerm.Add(memory);
            if (bank.ShortTerm.Count > ShortTermCapacity)
            {
                var oldest = bank.ShortTerm.OrderBy(m => m.Timestamp).First();
                if (oldest.Importance >= ConsolidationThreshold)
                    ConsolidateMemory(bank, oldest);
                else
                    bank.ShortTerm.Remove(oldest);
            }
            bank.Episodic.Add(memory);
            if (memory.Type == MemoryType.Semantic)
                UpdateSemantic(bank, memory);
            if (memory.Type == MemoryType.Procedural)
                UpdateProcedural(bank, memory);
        }

        public IReadOnlyList<MemoryEntry> Retrieve(SentientEntity entity, MemoryQuery query)
        {
            var bank = GetOrCreate(entity.EntityId);
            var results = new List<MemoryEntry>();
            results.AddRange(Search(bank.ShortTerm, query));
            results.AddRange(Search(bank.LongTerm, query));
            results.AddRange(Search(bank.Episodic, query));
            results.AddRange(Search(bank.Semantic, query));
            results.AddRange(Search(bank.Procedural, query));
            results = results.DistinctBy(m => m.Id).ToList();
            if (query.SortByEmotionalIntensity)
                results = results.OrderByDescending(m => m.EmotionalIntensity).ToList();
            else if (query.SortByImportance)
                results = results.OrderByDescending(m => m.Importance).ToList();
            else if (query.SortByRecency)
                results = results.OrderByDescending(m => m.Timestamp).ToList();
            if (results.Count > query.MaxResults)
                results = results.Take(query.MaxResults).ToList();
            foreach (var m in results)
            { m.RetrievalCount++; m.LastRetrieved = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond; }
            return results.AsReadOnly();
        }

        public int Forget(SentientEntity entity, Func<MemoryEntry, bool> criteria)
        {
            var bank = GetOrCreate(entity.EntityId);
            int count = 0;
            count += bank.ShortTerm.RemoveAll(m => criteria(m));
            count += bank.LongTerm.RemoveAll(m => criteria(m));
            count += bank.Episodic.RemoveAll(m => criteria(m));
            count += bank.Semantic.RemoveAll(m => criteria(m));
            count += bank.Procedural.RemoveAll(m => criteria(m));
            return count;
        }

        public void Consolidate(SentientEntity entity)
        {
            var bank = GetOrCreate(entity.EntityId);
            var toCon = bank.ShortTerm.Where(m => m.Importance >= ConsolidationThreshold || m.EmotionalIntensity > 0.7f || m.RetrievalCount > 3).ToList();
            foreach (var m in toCon)
                ConsolidateMemory(bank, m);
            var rehearsed = bank.Episodic.Where(m => m.RetrievalCount > 2 && !m.IsConsolidated).ToList();
            foreach (var m in rehearsed)
            {
                m.ConsolidationStrength += 0.1f * m.RetrievalCount;
                if (m.ConsolidationStrength >= 1f)
                { m.IsConsolidated = true; bank.LongTerm.Add(m); }
            }
            ApplyDecay(bank);
        }

        public IReadOnlyDictionary<MemoryType, int> GetMemoryCounts(SentientEntity entity)
        {
            var bank = GetOrCreate(entity.EntityId);
            return new Dictionary<MemoryType, int>
            {
                { MemoryType.ShortTerm, bank.ShortTerm.Count }, { MemoryType.LongTerm, bank.LongTerm.Count },
                { MemoryType.Episodic, bank.Episodic.Count }, { MemoryType.Semantic, bank.Semantic.Count },
                { MemoryType.Procedural, bank.Procedural.Count }
            };
        }

        private void ConsolidateMemory(EntityMemoryBank bank, MemoryEntry m)
        {
            if (!m.IsConsolidated)
            { m.IsConsolidated = true; m.ConsolidationStrength += 0.2f; bank.LongTerm.Add(m); }
            bank.ShortTerm.Remove(m);
        }

        private static List<MemoryEntry> Search(List<MemoryEntry> memories, MemoryQuery q)
        {
            return memories.Where(m =>
                (string.IsNullOrEmpty(q.SearchTerms) || m.Content.Contains(q.SearchTerms, StringComparison.OrdinalIgnoreCase)) &&
                (!q.Type.HasValue || m.Type == q.Type.Value) &&
                m.EmotionalIntensity >= q.MinEmotionalIntensity &&
                m.Importance >= q.MinImportance &&
                (Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond - m.Timestamp) <= q.MaxAge &&
                (!q.AssociatedEntity.HasValue || m.AssociatedEntities.Contains(q.AssociatedEntity.Value)) &&
                (!q.LocationFilter.HasValue || Vector3.Distance(m.Location, q.LocationFilter.Value.Center) <= q.LocationFilter.Value.Radius) &&
                (q.Tags.Count == 0 || q.Tags.Any(t => m.Tags.Contains(t)))
            ).ToList();
        }

        private void UpdateSemantic(EntityMemoryBank bank, MemoryEntry m)
        {
            var existing = bank.Semantic.FirstOrDefault(s => s.Content == m.Content);
            if (existing != null)
            { existing.RetrievalCount++; existing.Importance = Math.Min(1f, existing.Importance + 0.1f); }
            else
                bank.Semantic.Add(m);
        }

        private void UpdateProcedural(EntityMemoryBank bank, MemoryEntry m)
        {
            var existing = bank.Procedural.FirstOrDefault(p => p.Content == m.Content);
            if (existing != null)
            { existing.RetrievalCount++; existing.ConsolidationStrength = Math.Min(1f, existing.ConsolidationStrength + 0.1f); }
            else
                bank.Procedural.Add(m);
        }

        private void ApplyDecay(EntityMemoryBank bank)
        {
            var now = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
            foreach (var list in new[] { bank.ShortTerm, bank.LongTerm, bank.Episodic, bank.Semantic, bank.Procedural })
            {
                foreach (var m in list.Where(m => m.DecayFactor > 0).ToList())
                {
                    float age = (float)(now - m.Timestamp);
                    m.DecayFactor = Math.Max(0, m.DecayFactor - age * 0.0001f * m.DecayFactor);
                    if (m.DecayFactor <= 0.01f && m.Type != MemoryType.Procedural)
                        list.Remove(m);
                }
            }
        }
    }

    public class EntityMemoryBank
    {
        public List<MemoryEntry> ShortTerm { get; set; } = new();
        public List<MemoryEntry> LongTerm { get; set; } = new();
        public List<MemoryEntry> Episodic { get; set; } = new();
        public List<MemoryEntry> Semantic { get; set; } = new();
        public List<MemoryEntry> Procedural { get; set; } = new();
    }

}
