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

    public class EnvironmentalResponseSystem
    {
        private readonly Dictionary<Guid, EnvironmentalState> _entityEnvStates = new();

        public float WeatherResponseFactor { get; set; } = 0.3f;
        public float TimeOfDayResponseFactor { get; set; } = 0.2f;
        public float DangerResponseFactor { get; set; } = 0.5f;

        public void UpdateResponses(SentientEntity entity, WorldStateData worldState, float deltaTime)
        {
            var state = GetOrCreate(entity.EntityId);
            UpdateWeatherResponse(entity, worldState.Weather, state, deltaTime);
            UpdateTimeResponse(entity, worldState.Time, state, deltaTime);
            UpdateDangerResponse(entity, worldState, state, deltaTime);
            UpdateComfortResponse(entity, state, deltaTime);
            ApplyBehaviorModifiers(entity, state);
        }

        private void UpdateWeatherResponse(SentientEntity entity, WeatherConditions weather, EnvironmentalState state, float deltaTime)
        {
            var prevComfort = state.WeatherComfort;
            state.WeatherComfort = weather.Type switch
            {
                WeatherType.Clear => 1.0f,
                WeatherType.Cloudy => 0.8f,
                WeatherType.Rain => 0.4f,
                WeatherType.HeavyRain => 0.2f,
                WeatherType.Snow => 0.5f,
                WeatherType.Fog => 0.6f,
                WeatherType.Storm => 0.1f,
                WeatherType.Sandstorm => 0.15f,
                _ => 0.7f
            };
            state.WeatherComfort *= weather.Visibility;
            float change = state.WeatherComfort - prevComfort;
            if (Math.Abs(change) > 0.1f)
            {
                entity.AddPerception(new PerceptionEvent(
                    PerceptionType.Proximity, Guid.Empty, Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond,
                    Math.Abs(change), entity.Position, Vector3.Zero,
                    $"Weather changed to {weather.Type}", Math.Abs(change)));
            }
        }

        private void UpdateTimeResponse(SentientEntity entity, double worldTime, EnvironmentalState state, float deltaTime)
        {
            double hourOfDay = (worldTime / 3600.0) % 24.0;
            state.TimeOfDayFactor = hourOfDay switch
            {
                >= 6 and < 8 => 0.8f,
                >= 8 and < 17 => 1.0f,
                >= 17 and < 20 => 0.7f,
                >= 20 or < 6 => 0.3f,
                _ => 0.5f
            };
        }

        private void UpdateDangerResponse(SentientEntity entity, WorldStateData worldState, EnvironmentalState state, float deltaTime)
        {
            float dangerLevel = 0;
            foreach (var rel in entity.Relationships.Values)
            {
                if (rel.Type == RelationshipType.Hostile)
                    dangerLevel += rel.Strength * 0.3f;
            }
            var nearbyHostiles = worldState.Entities.Values
                .Where(e => e.EntityId != entity.EntityId && entity.Relationships.TryGetValue(e.EntityId, out var r) && r.Type == RelationshipType.Hostile)
                .ToList();
            foreach (var hostile in nearbyHostiles)
            {
                float dist = entity.DistanceTo(hostile);
                dangerLevel += Math.Max(0, 1f - dist / 30f) * 0.5f;
            }
            state.DangerLevel = Math.Clamp(state.DangerLevel + (dangerLevel - state.DangerLevel) * deltaTime * 2f, 0, 1);
        }

        private void UpdateComfortResponse(SentientEntity entity, EnvironmentalState state, float deltaTime)
        {
            float overallComfort = (state.WeatherComfort * 0.4f + state.TimeOfDayFactor * 0.3f + (1f - state.DangerLevel) * 0.3f);
            state.OverallComfort = Math.Clamp(state.OverallComfort + (overallComfort - state.OverallComfort) * deltaTime, 0, 1);
        }

        private void ApplyBehaviorModifiers(SentientEntity entity, EnvironmentalState state)
        {
            entity.SetProperty("WeatherComfort", state.WeatherComfort);
            entity.SetProperty("TimeFactor", state.TimeOfDayFactor);
            entity.SetProperty("DangerLevel", state.DangerLevel);
            entity.SetProperty("OverallComfort", state.OverallComfort);
        }

        private EnvironmentalState GetOrCreate(Guid entityId)
        {
            if (!_entityEnvStates.TryGetValue(entityId, out var s))
            { s = new EnvironmentalState(); _entityEnvStates[entityId] = s; }
            return s;
        }
    }

    public class EnvironmentalState
    {
        public float WeatherComfort { get; set; } = 0.7f;
        public float TimeOfDayFactor { get; set; } = 1.0f;
        public float DangerLevel { get; set; }
        public float OverallComfort { get; set; } = 0.7f;
    }

}
