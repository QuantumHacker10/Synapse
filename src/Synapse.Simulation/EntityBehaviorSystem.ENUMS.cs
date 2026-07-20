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

    public enum BehaviorState { Idle, Active, Dormant, Transitioning, Terminated }
    /// <summary>
    /// Simulation entity roles — not game-engine actor kinds.
    /// <see cref="Sentient"/> is an adaptive inhabitant (perception + decisions), not an NPC script.
    /// <see cref="Observer"/> is the operator/camera presence in the simulation, not a player avatar.
    /// </summary>
    public enum EntityType { Static, Dynamic, Kinematic, Observer, Sentient, Environmental, Trigger, Sensor }
    public enum PerceptionType { Visual, Auditory, Tactile, Proximity, Semantic, Emotional }
    public enum RelationshipType { Neutral, Friendly, Hostile, Parent, Child, Sibling, Partner, Servant, Master }
    public enum EmotionalState { Neutral, Happy, Sad, Angry, Fearful, Surprised, Disgusted, Trusting, Anticipating, Calm, Excited, Bored, Curious, Confused, Determined, Anxious, Frustrated }
    public enum ScheduleMode { Fixed, Variable, EventDriven, Priority, Adaptive }
    public enum BehaviorNodeType { Sequence, Selector, Parallel, Condition, Action, Decorator, RandomSelector, Loop, Inverter, Repeater, Succeeder, Failer, Wait, Cooldown, Guard, LLMQuery, WeightedRandom }
    public enum TaskStatus { Success, Failure, Running, Pending }
    public enum WeatherType { Clear, Cloudy, Rain, HeavyRain, Snow, Fog, Storm, Sandstorm }
    public enum MemoryType { ShortTerm, LongTerm, Episodic, Semantic, Procedural }
    public enum NodeType { Root, Branch, Leaf }

}
