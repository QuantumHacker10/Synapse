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

    public static class VectorMath
    {
        public static float AngleBetween(Vector3 a, Vector3 b)
        {
            float dot = Vector3.Dot(Vector3.Normalize(a), Vector3.Normalize(b));
            return (float)Math.Acos(Math.Clamp(dot, -1f, 1f));
        }

        public static Vector3 Slerp(Vector3 a, Vector3 b, float t)
        {
            float angle = AngleBetween(a, b);
            if (angle < 0.001f)
                return Vector3.Lerp(a, b, t);
            float sinAngle = (float)Math.Sin(angle);
            float factorA = (float)Math.Sin((1 - t) * angle) / sinAngle;
            float factorB = (float)Math.Sin(t * angle) / sinAngle;
            return a * factorA + b * factorB;
        }

        public static Vector3 ClampMagnitude(Vector3 v, float maxLength)
        {
            if (v.LengthSquared() > maxLength * maxLength)
                return Vector3.Normalize(v) * maxLength;
            return v;
        }

        public static Vector3 RandomPointInSphere(float radius)
        {
            var rng = new Random();
            float theta = (float)(rng.NextDouble() * 2 * Math.PI);
            float phi = (float)(Math.Acos(2 * rng.NextDouble() - 1));
            float r = radius * (float)Math.Pow(rng.NextDouble(), 1.0 / 3.0);
            return new Vector3(
                r * (float)Math.Sin(phi) * (float)Math.Cos(theta),
                r * (float)Math.Sin(phi) * (float)Math.Sin(theta),
                r * (float)Math.Cos(phi));
        }

        public static Vector3 RandomPointOnCircle(float radius)
        {
            float angle = (float)(Random.Shared.NextDouble() * 2 * Math.PI);
            return new Vector3(MathF.Cos(angle) * radius, 0, MathF.Sin(angle) * radius);
        }

        public static float Lerp(float a, float b, float t) => a + (b - a) * Math.Clamp(t, 0, 1);
        public static Vector3 Lerp(Vector3 a, Vector3 b, float t) => Vector3.Lerp(a, b, Math.Clamp(t, 0, 1));
        public static float SmoothStep(float edge0, float edge1, float x)
        {
            float t = Math.Clamp((x - edge0) / (edge1 - edge0), 0, 1);
            return t * t * (3 - 2 * t);
        }
    }

}
