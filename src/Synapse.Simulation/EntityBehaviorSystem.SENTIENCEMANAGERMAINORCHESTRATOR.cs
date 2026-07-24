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

    public class SentienceManager
    {
        private readonly WorldStateManager _worldState;
        private readonly PerceptionSystem _perceptionSystem;
        private readonly EmotionalModel _emotionalModel;
        private readonly MemorySystem _memorySystem;
        private readonly RelationshipManager _relationshipManager;
        private readonly SpatialReasoningSystem _spatialReasoning;
        private readonly GroupBehaviorSystem _groupBehavior;
        private readonly EntityTickScheduler _tickScheduler;
        private readonly BehaviorDebugger _debugger;
        private readonly BehaviorAnalytics _analytics;
        private readonly BehaviorCompiler _compiler;
        private readonly EmotionalDynamicsEngine _emotionalDynamics;
        private readonly AdvancedPerceptionProcessor _advancedPerception;
        private readonly MemoryConsolidationEngine _memoryConsolidation;
        private readonly SemanticNetworkBuilder _semanticNetwork;
        private readonly PredictionEngine _predictionEngine;
        private readonly ConcurrentDictionary<Guid, SentientEntity> _entities = new();
        private readonly ConcurrentDictionary<string, BehaviorTree> _sharedBehaviorTrees = new();
        private bool _isRunning;
        private CancellationTokenSource? _cts;
        private float _globalTimeScale = 1f;

        public SentienceManager()
        {
            _worldState = new WorldStateManager();
            _perceptionSystem = new PerceptionSystem();
            _emotionalModel = new EmotionalModel();
            _memorySystem = new MemorySystem();
            _relationshipManager = new RelationshipManager();
            _spatialReasoning = new SpatialReasoningSystem();
            _groupBehavior = new GroupBehaviorSystem();
            _debugger = new BehaviorDebugger();
            _analytics = new BehaviorAnalytics();
            _compiler = new BehaviorCompiler(_debugger);
            _emotionalDynamics = new EmotionalDynamicsEngine();
            _advancedPerception = new AdvancedPerceptionProcessor();
            _memoryConsolidation = new MemoryConsolidationEngine();
            _semanticNetwork = new SemanticNetworkBuilder();
            _predictionEngine = new PredictionEngine();
            _tickScheduler = new EntityTickScheduler(_worldState, _perceptionSystem, _emotionalModel, _debugger, _analytics);
        }

        public WorldStateManager WorldState => _worldState;
        public PerceptionSystem PerceptionSystem => _perceptionSystem;
        public EmotionalModel EmotionalModel => _emotionalModel;
        public MemorySystem MemorySystem => _memorySystem;
        public RelationshipManager RelationshipManager => _relationshipManager;
        public SpatialReasoningSystem SpatialReasoning => _spatialReasoning;
        public GroupBehaviorSystem GroupBehavior => _groupBehavior;
        public BehaviorDebugger Debugger => _debugger;
        public BehaviorAnalytics Analytics => _analytics;
        public BehaviorCompiler Compiler => _compiler;
        public float GlobalTimeScale { get => _globalTimeScale; set => _globalTimeScale = Math.Max(0, value); }
        public bool IsRunning => _isRunning;
        public int EntityCount => _entities.Count;

        public SentientEntity CreateEntity(EntityType type, Vector3 position, string? behaviorTreeName = null)
        {
            var entity = new SentientEntity(Guid.NewGuid(), type)
            {
                Position = position,
                EmotionalModel = _emotionalModel,
                MemorySystem = _memorySystem
            };

            if (behaviorTreeName != null && _sharedBehaviorTrees.TryGetValue(behaviorTreeName, out var tree))
                entity.BehaviorTree = tree.Clone();

            _entities[entity.EntityId] = entity;
            _worldState.AddEntity(entity);
            _tickScheduler.RegisterEntity(entity);
            return entity;
        }

        public void RemoveEntity(Guid entityId)
        {
            if (_entities.TryRemove(entityId, out var entity))
            {
                _worldState.RemoveEntity(entityId);
                _tickScheduler.UnregisterEntity(entityId);
                _perceptionSystem.RemoveEntity(entityId);
                _emotionalModel.RemoveEntity(entityId);
                _groupBehavior.UnregisterEntity(entityId);
            }
        }

        public SentientEntity? GetEntity(Guid entityId) => _entities.TryGetValue(entityId, out var e) ? e : null;
        public IReadOnlyCollection<SentientEntity> GetAllEntities() => _entities.Values.ToList().AsReadOnly();

        public void RegisterBehaviorTree(string name, BehaviorTree tree) => _sharedBehaviorTrees[name] = tree;
        public BehaviorTree? GetBehaviorTree(string name) => _sharedBehaviorTrees.TryGetValue(name, out var t) ? t : null;

        public void SetRelationship(Guid entityA, Guid entityB, RelationshipType type, float strength = 0.5f)
        {
            _relationshipManager.AddRelationship(new Relationship
            {
                EntityA = entityA,
                EntityB = entityB,
                Type = type,
                Strength = strength,
                EstablishedTime = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond
            });
            if (_entities.TryGetValue(entityA, out var eA))
                eA.Relationships[entityB] = new Relationship { EntityA = entityA, EntityB = entityB, Type = type, Strength = strength };
            if (_entities.TryGetValue(entityB, out var eB))
                eB.Relationships[entityA] = new Relationship { EntityA = entityA, EntityB = entityB, Type = type, Strength = strength };
        }

        public void AddGroupMember(string groupName, Guid entityId)
        {
            _groupBehavior.RegisterEntity(entityId, groupName);
            if (_entities.TryGetValue(entityId, out var e))
                e.EntityGroup = groupName;
        }

        public async Task RunAsync(float deltaTime, CancellationToken ct)
        {
            _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            _isRunning = true;
            try
            {
                while (!_cts.Token.IsCancellationRequested)
                {
                    await UpdateAsync(deltaTime * _globalTimeScale);
                    await Task.Delay((int)(deltaTime * 1000), _cts.Token);
                }
            }
            catch (OperationCanceledException) { }
            finally { _isRunning = false; }
        }

        public void Stop() { _cts?.Cancel(); _isRunning = false; }

        public async Task UpdateAsync(float deltaTime)
        {
            _worldState.CurrentTime += deltaTime;
            _worldState.ProcessEvents();
            _perceptionSystem.UpdateSpatialIndex(_entities.Values);
            await _tickScheduler.TickAllEntitiesAsync(deltaTime);

            foreach (var entity in _entities.Values)
            {
                if (entity.CurrentState == BehaviorState.Terminated)
                    continue;
                _predictionEngine.UpdateHistory(entity.EntityId, entity.Position, entity.Velocity);
                _emotionalDynamics.UpdateDynamics(entity.EntityId, _emotionalModel.GetStateData(entity.EntityId) ?? new EmotionalStateData(), deltaTime);
                _memoryConsolidation.ProcessConsolidation(entity, _memorySystem, deltaTime);
                _relationshipManager.EvolveRelationships(deltaTime, 0.001f);
            }
        }

        public Dictionary<string, object> GetSystemReport()
        {
            var report = new Dictionary<string, object>
            {
                { "EntityCount", _entities.Count },
                { "IsRunning", _isRunning },
                { "GlobalTimeScale", _globalTimeScale },
                { "WorldTime", _worldState.CurrentTime },
                { "Weather", _worldState.Weather.Type.ToString() },
                { "Analytics", _analytics.GetComprehensiveReport() },
                { "WorldStats", _worldState.GetWorldStats() }
            };
            return report;
        }
    }

}
