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

    public static class WorldStateExtensions
    {
        public static List<SentientEntity> FindEntitiesByType(this WorldStateManager wsm, EntityType type)
        {
            return wsm.GetAllEntities().Values.Where(e => e.EntityType == type).ToList();
        }

        public static List<SentientEntity> FindEntitiesByEmotion(this WorldStateManager wsm, EmotionalState emotion)
        {
            return wsm.GetAllEntities().Values.Where(e => e.CurrentEmotion == emotion).ToList();
        }

        public static List<SentientEntity> FindEntitiesByState(this WorldStateManager wsm, BehaviorState state)
        {
            return wsm.GetAllEntities().Values.Where(e => e.CurrentState == state).ToList();
        }

        public static SentientEntity? FindNearestEnemy(this WorldStateManager wsm, SentientEntity entity)
        {
            var enemies = wsm.GetAllEntities().Values
                .Where(e => e.EntityId != entity.EntityId && entity.Relationships.TryGetValue(e.EntityId, out var r) && r.Type == RelationshipType.Hostile)
                .OrderBy(e => entity.DistanceTo(e))
                .ToList();
            return enemies.Count > 0 ? enemies[0] : null;
        }

        public static SentientEntity? FindNearestAlly(this WorldStateManager wsm, SentientEntity entity)
        {
            var allies = wsm.GetAllEntities().Values
                .Where(e => e.EntityId != entity.EntityId && entity.Relationships.TryGetValue(e.EntityId, out var r) && (r.Type == RelationshipType.Friendly || r.Type == RelationshipType.Parent))
                .OrderBy(e => entity.DistanceTo(e))
                .ToList();
            return allies.Count > 0 ? allies[0] : null;
        }

        public static int CountEnemiesInRange(this WorldStateManager wsm, SentientEntity entity, float range)
        {
            return wsm.GetAllEntities().Values
                .Count(e => e.EntityId != entity.EntityId && entity.DistanceTo(e) <= range && entity.Relationships.TryGetValue(e.EntityId, out var r) && r.Type == RelationshipType.Hostile);
        }

        public static float CalculateThreatLevel(this WorldStateManager wsm, SentientEntity entity)
        {
            float threat = 0;
            foreach (var e in wsm.GetAllEntities().Values)
            {
                if (e.EntityId == entity.EntityId)
                    continue;
                if (!entity.Relationships.TryGetValue(e.EntityId, out var r) || r.Type != RelationshipType.Hostile)
                    continue;
                float dist = entity.DistanceTo(e);
                float distFactor = Math.Max(0, 1f - dist / 50f);
                threat += distFactor * r.Strength * (e.Health / e.MaxHealth);
            }
            return Math.Clamp(threat, 0, 1);
        }

        public static WeatherConditions LerpWeather(this WeatherConditions a, WeatherConditions b, float t)
        {
            return new WeatherConditions(
                a.Temperature + (b.Temperature - a.Temperature) * t,
                a.Humidity + (b.Humidity - a.Humidity) * t,
                a.WindSpeed + (b.WindSpeed - a.WindSpeed) * t,
                a.WindDirection + (b.WindDirection - a.WindDirection) * t,
                a.Visibility + (b.Visibility - a.Visibility) * t,
                a.AmbientLight + (b.AmbientLight - a.AmbientLight) * t,
                t > 0.5f ? b.Type : a.Type);
        }
    }

}
