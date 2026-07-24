// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    // =========================================================================
    // LawMathUtils — mathematical utility functions
    // =========================================================================

    /// <summary>Mathematical utility functions for law computations.</summary>
    public static class LawMathUtils
    {
        /// <summary>Clamp a value between min and max.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Clamp(float value, float min, float max) =>
            MathF.Max(min, MathF.Min(max, value));

        /// <summary>Linear interpolation between a and b.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Lerp(float a, float b, float t) => a + t * (b - a);

        /// <summary>Inverse linear interpolation: find t such that Lerp(a, b, t) = value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float InverseLerp(float a, float b, float value)
        {
            if (MathF.Abs(b - a) < float.Epsilon)
                return 0f;
            return (value - a) / (b - a);
        }

        /// <summary>Smooth step (Hermite interpolation).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmoothStep(float edge0, float edge1, float x)
        {
            float t = Clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
            return t * t * (3f - 2f * t);
        }

        /// <summary>Smoother step (Ken Perlin's version).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmootherStep(float edge0, float edge1, float x)
        {
            float t = Clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
            return t * t * t * (t * (t * 6f - 15f) + 10f);
        }

        /// <summary>Remap a value from one range to another.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Remap(float value, float fromMin, float fromMax, float toMin, float toMax)
        {
            float t = InverseLerp(fromMin, fromMax, value);
            return Lerp(toMin, toMax, t);
        }

        /// <summary>Compute the Euclidean distance between two 3D points.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Distance3D(float x1, float y1, float z1, float x2, float y2, float z2)
        {
            float dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
            return MathF.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        /// <summary>Compute the dot product of two 3D vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Dot3D(float ax, float ay, float az, float bx, float by, float bz) =>
            ax * bx + ay * by + az * bz;

        /// <summary>Compute the cross product Z component of two 2D vectors.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Cross2D(float ax, float ay, float bx, float by) =>
            ax * by - ay * bx;

        /// <summary>Normalize a 3D vector.</summary>
        public static (float X, float Y, float Z) Normalize3D(float x, float y, float z)
        {
            float len = MathF.Sqrt(x * x + y * y + z * z);
            if (len < float.Epsilon)
                return (0f, 0f, 0f);
            return (x / len, y / len, z / len);
        }

        /// <summary>Compute the magnitude of a 3D vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Magnitude3D(float x, float y, float z) =>
            MathF.Sqrt(x * x + y * y + z * z);

        /// <summary>Compute the Laplacian of a scalar field at a point using 7-point stencil.</summary>
        public static float Laplacian7Point(FieldGrid field, int x, int y, int z, float dx)
        {
            int sx = field.SizeX, sy = field.SizeY, sz = field.SizeZ;
            float c = field[x, y, z];
            float left = x > 0 ? field[x - 1, y, z] : c;
            float right = x < sx - 1 ? field[x + 1, y, z] : c;
            float bottom = y > 0 ? field[x, y - 1, z] : c;
            float top = y < sy - 1 ? field[x, y + 1, z] : c;
            float back = z > 0 ? field[x, y, z - 1] : c;
            float front = z < sz - 1 ? field[x, y, z + 1] : c;
            return (left + right + bottom + top + back + front - 6f * c) / (dx * dx);
        }

        /// <summary>Compute the gradient X component using central differences.</summary>
        public static float GradientX(FieldGrid field, int x, int y, int z, float dx)
        {
            int sx = field.SizeX;
            float left = x > 0 ? field[x - 1, y, z] : field[0, y, z];
            float right = x < sx - 1 ? field[x + 1, y, z] : field[sx - 1, y, z];
            return (right - left) / (2f * dx);
        }

        /// <summary>Compute the gradient Y component using central differences.</summary>
        public static float GradientY(FieldGrid field, int x, int y, int z, float dy)
        {
            int sy = field.SizeY;
            float bottom = y > 0 ? field[x, y - 1, z] : field[x, 0, z];
            float top = y < sy - 1 ? field[x, y + 1, z] : field[x, sy - 1, z];
            return (top - bottom) / (2f * dy);
        }

        /// <summary>Compute the gradient Z component using central differences.</summary>
        public static float GradientZ(FieldGrid field, int x, int y, int z, float dz)
        {
            int sz = field.SizeZ;
            float back = z > 0 ? field[x, y, z - 1] : field[x, y, 0];
            float front = z < sz - 1 ? field[x, y, z + 1] : field[x, y, sz - 1];
            return (front - back) / (2f * dz);
        }

        /// <summary>Compute the divergence of a vector field at a point.</summary>
        public static float Divergence(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return GradientX(vx, x, y, z, dx) + GradientY(vy, x, y, z, dx) + GradientZ(vz, x, y, z, dx);
        }

        /// <summary>Compute the curl X component at a point.</summary>
        public static float CurlX(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return GradientY(vz, x, y, z, dx) - GradientZ(vy, x, y, z, dx);
        }

        /// <summary>Compute the curl Y component at a point.</summary>
        public static float CurlY(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return GradientZ(vx, x, y, z, dx) - GradientX(vz, x, y, z, dx);
        }

        /// <summary>Compute the curl Z component at a point.</summary>
        public static float CurlZ(FieldGrid vx, FieldGrid vy, FieldGrid vz, int x, int y, int z, float dx)
        {
            return GradientX(vy, x, y, z, dx) - GradientY(vx, x, y, z, dx);
        }

        /// <summary>Compute the total kinetic energy of a velocity field.</summary>
        public static float KineticEnergy(FieldGrid vx, FieldGrid vy, FieldGrid vz, float density)
        {
            float energy = 0f;
            int total = vx.TotalCells;
            for (int i = 0; i < total; i++)
            {
                float v2 = vx.Data[i] * vx.Data[i] + vy.Data[i] * vy.Data[i] + vz.Data[i] * vz.Data[i];
                energy += 0.5f * density * v2;
            }
            return energy;
        }

        /// <summary>Compute the total thermal energy of a temperature field.</summary>
        public static float ThermalEnergy(FieldGrid temperature, float specificHeat, float density)
        {
            float energy = 0f;
            int total = temperature.TotalCells;
            for (int i = 0; i < total; i++)
                energy += specificHeat * density * temperature.Data[i];
            return energy;
        }

        /// <summary>Compute the L2 norm of a field.</summary>
        public static float L2Norm(FieldGrid field)
        {
            double sum = 0;
            for (int i = 0; i < field.TotalCells; i++)
                sum += (double)field.Data[i] * field.Data[i];
            return MathF.Sqrt((float)(sum / field.TotalCells));
        }

        /// <summary>Compute the maximum absolute value in a field.</summary>
        public static float InfinityNorm(FieldGrid field)
        {
            float max = 0f;
            for (int i = 0; i < field.TotalCells; i++)
            {
                float abs = MathF.Abs(field.Data[i]);
                if (abs > max)
                    max = abs;
            }
            return max;
        }

        /// <summary>Compute the field integral (sum * volume element).</summary>
        public static float Integral(FieldGrid field, float dx)
        {
            float volume = dx * dx * dx;
            float sum = 0f;
            for (int i = 0; i < field.TotalCells; i++)
                sum += field.Data[i];
            return sum * volume;
        }

        /// <summary>Compute the field average over interior cells only.</summary>
        public static float InteriorAverage(FieldGrid field)
        {
            float sum = 0f;
            int count = 0;
            for (int z = 1; z < field.SizeZ - 1; z++)
                for (int y = 1; y < field.SizeY - 1; y++)
                    for (int x = 1; x < field.SizeX - 1; x++)
                    {
                        sum += field[x, y, z];
                        count++;
                    }
            return count > 0 ? sum / count : 0f;
        }

        /// <summary>Compute the maximum gradient magnitude in a field.</summary>
        public static float MaxGradientMagnitude(FieldGrid field, float dx)
        {
            float maxGrad = 0f;
            for (int z = 1; z < field.SizeZ - 1; z++)
                for (int y = 1; y < field.SizeY - 1; y++)
                    for (int x = 1; x < field.SizeX - 1; x++)
                    {
                        float gx = GradientX(field, x, y, z, dx);
                        float gy = GradientY(field, x, y, z, dx);
                        float gz = GradientZ(field, x, y, z, dx);
                        float mag = MathF.Sqrt(gx * gx + gy * gy + gz * gz);
                        if (mag > maxGrad)
                            maxGrad = mag;
                    }
            return maxGrad;
        }
    }
}
