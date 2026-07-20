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

    public class PersonalitySystem
    {
        private readonly Dictionary<Guid, PersonalityProfile> _profiles = new();
        private static readonly string[] CoreTraits = { "Aggression", "Courage", "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism", "Intelligence", "Empathy", "Patience" };

        public void InitializePersonality(Guid entityId, float variance = 0.2f)
        {
            var profile = new PersonalityProfile();
            var rng = new Random(entityId.GetHashCode());
            foreach (var trait in CoreTraits)
            {
                profile.Traits[trait] = Math.Clamp(0.5f + (float)(rng.NextDouble() * 2 - 1) * variance, 0, 1);
            }
            var temperament = DetermineTemperament(profile);
            profile.Temperament = temperament;
            _profiles[entityId] = profile;
        }

        public float GetTrait(Guid entityId, string trait)
        {
            if (_profiles.TryGetValue(entityId, out var p) && p.Traits.TryGetValue(trait, out var v))
                return v;
            return 0.5f;
        }

        public void ModifyTrait(Guid entityId, string trait, float delta)
        {
            if (_profiles.TryGetValue(entityId, out var p))
            {
                p.Traits.TryGetValue(trait, out float current);
                p.Traits[trait] = Math.Clamp(current + delta, 0, 1);
            }
        }

        public string GetTemperament(Guid entityId)
        {
            return _profiles.TryGetValue(entityId, out var p) ? p.Temperament : "Unknown";
        }

        public PersonalityProfile? GetProfile(Guid entityId)
        {
            return _profiles.TryGetValue(entityId, out var p) ? p : null;
        }

        public float CalculateBehaviorWeight(Guid entityId, string behavior)
        {
            if (!_profiles.TryGetValue(entityId, out var profile))
                return 0.5f;
            return behavior.ToLower() switch
            {
                "attack" or "combat" or "fight" => (profile.Traits.GetValueOrDefault("Aggression", 0.5f) + profile.Traits.GetValueOrDefault("Courage", 0.5f)) / 2f,
                "flee" or "escape" or "hide" => 1f - (profile.Traits.GetValueOrDefault("Courage", 0.5f) + profile.Traits.GetValueOrDefault("Aggression", 0.5f)) / 2f,
                "explore" or "investigate" => profile.Traits.GetValueOrDefault("Openness", 0.5f),
                "socialize" or "communicate" => profile.Traits.GetValueOrDefault("Extraversion", 0.5f),
                "help" or "protect" or "heal" => profile.Traits.GetValueOrDefault("Agreeableness", 0.5f) * profile.Traits.GetValueOrDefault("Empathy", 0.5f),
                "wait" or "idle" or "rest" => profile.Traits.GetValueOrDefault("Patience", 0.5f),
                "patrol" or "guard" => profile.Traits.GetValueOrDefault("Conscientiousness", 0.5f),
                _ => 0.5f
            };
        }

        public EmotionalState GetEmotionalTendency(Guid entityId)
        {
            if (!_profiles.TryGetValue(entityId, out var profile))
                return EmotionalState.Neutral;
            float neuroticism = profile.Traits.GetValueOrDefault("Neuroticism", 0.5f);
            float extraversion = profile.Traits.GetValueOrDefault("Extraversion", 0.5f);

            if (neuroticism > 0.7f)
                return EmotionalState.Anxious;
            if (extraversion > 0.7f)
                return EmotionalState.Excited;
            if (neuroticism < 0.3f && extraversion < 0.3f)
                return EmotionalState.Calm;
            if (extraversion > 0.5f)
                return EmotionalState.Happy;
            return EmotionalState.Neutral;
        }

        public Dictionary<string, float> GenerateCompatibility(Guid entityA, Guid entityB)
        {
            var profileA = _profiles.TryGetValue(entityA, out var pa) ? pa : null;
            var profileB = _profiles.TryGetValue(entityB, out var pb) ? pb : null;
            if (profileA == null || profileB == null)
                return new Dictionary<string, float> { { "Overall", 0.5f } };

            var compatibility = new Dictionary<string, float>();
            float totalCompat = 0;
            int traitCount = 0;

            foreach (var trait in CoreTraits)
            {
                float valA = profileA.Traits.GetValueOrDefault(trait, 0.5f);
                float valB = profileB.Traits.GetValueOrDefault(trait, 0.5f);
                float traitCompat = 1f - Math.Abs(valA - valB);

                if (trait == "Aggression" || trait == "Neuroticism")
                    traitCompat = 1f - (valA + valB) / 2f;

                compatibility[trait] = traitCompat;
                totalCompat += traitCompat;
                traitCount++;
            }

            compatibility["Overall"] = traitCount > 0 ? totalCompat / traitCount : 0.5f;
            return compatibility;
        }

        private static string DetermineTemperament(PersonalityProfile profile)
        {
            float aggression = profile.Traits.GetValueOrDefault("Aggression", 0.5f);
            float extraversion = profile.Traits.GetValueOrDefault("Extraversion", 0.5f);
            float neuroticism = profile.Traits.GetValueOrDefault("Neuroticism", 0.5f);
            float agreeableness = profile.Traits.GetValueOrDefault("Agreeableness", 0.5f);

            if (aggression > 0.7f && extraversion > 0.6f)
                return "Choleric";
            if (extraversion > 0.7f && agreeableness > 0.6f)
                return "Sanguine";
            if (neuroticism > 0.6f && extraversion < 0.4f)
                return "Melancholic";
            if (agreeableness > 0.7f && extraversion < 0.4f)
                return "Phlegmatic";
            return "Mixed";
        }
    }

    public class PersonalityProfile
    {
        public Dictionary<string, float> Traits { get; set; } = new();
        public string Temperament { get; set; } = "Mixed";
        public double LastUpdated { get; set; } = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;
        public int TraitUpdateCount { get; set; }
    }

}
