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

    public interface IBehaviorNode
    {
        string Id { get; }
        string Name { get; }
        BehaviorNodeType NodeType { get; }
        TaskStatus Tick(SentientEntity entity, EntityContext context, float deltaTime);
        IReadOnlyList<IBehaviorNode> GetChildren();
        void Reset();
        IBehaviorNode Clone();
        bool Validate(List<string> errors);
    }

    public interface IPerceptionSystem
    {
        void Update(SentientEntity entity, WorldStateData worldState);
        IReadOnlyList<PerceptionEvent> GetPerceptions(SentientEntity entity);
        IReadOnlyList<PerceptionEvent> FilterPerceptions(SentientEntity entity, PerceptionFilter filters);
    }

    public interface IEmotionalModel
    {
        void Update(SentientEntity entity, IReadOnlyList<PerceptionEvent> events, float deltaTime);
        EmotionalState GetCurrentState(SentientEntity entity);
        EmotionalResponse ReactToEvent(SentientEntity entity, PerceptionEvent perceptionEvent);
        IReadOnlyDictionary<EmotionalState, float> GetEmotionalProfile(SentientEntity entity);
    }

    public interface IMemorySystem
    {
        void Store(SentientEntity entity, MemoryEntry memory);
        IReadOnlyList<MemoryEntry> Retrieve(SentientEntity entity, MemoryQuery query);
        int Forget(SentientEntity entity, Func<MemoryEntry, bool> criteria);
        void Consolidate(SentientEntity entity);
        IReadOnlyDictionary<MemoryType, int> GetMemoryCounts(SentientEntity entity);
    }

}
