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

    public static class BehaviorRandom
    {
        [ThreadStatic] private static Random? _threadRandom;

        public static Random Instance => _threadRandom ??= new Random(Guid.NewGuid().GetHashCode());

        public static float NextFloat() => (float)Instance.NextDouble();
        public static float NextFloat(float min, float max) => min + (float)Instance.NextDouble() * (max - min);
        public static int NextInt(int min, int max) => Instance.Next(min, max);
        public static bool NextBool(float probability = 0.5f) => Instance.NextDouble() < probability;
        public static T PickRandom<T>(IReadOnlyList<T> items) => items[Instance.Next(items.Count)];

        public static T WeightedPick<T>(IReadOnlyList<T> items, IReadOnlyList<float> weights)
        {
            float total = weights.Sum();
            float r = (float)(Instance.NextDouble() * total);
            float cum = 0;
            for (int i = 0; i < items.Count; i++)
            {
                cum += weights[i];
                if (r <= cum)
                    return items[i];
            }
            return items[^1];
        }

        public static Vector3 InsideUnitSphere() => VectorMath.RandomPointInSphere(1f);
        public static Vector3 OnUnitCircle() => VectorMath.RandomPointOnCircle(1f);
        public static Vector3 InsideCircle(float radius) => VectorMath.RandomPointOnCircle(radius * (float)Math.Sqrt(Instance.NextDouble()));
    }

}
