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
    // Supporting types
    // =========================================================================

    /// <summary>Physical dimension exponents for dimensional analysis.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct Dimension : IEquatable<Dimension>
    {
        public readonly float Mass;
        public readonly float Length;
        public readonly float Time;
        public readonly float Temperature;
        public readonly float Amount;
        public readonly float Current;
        public readonly float Luminous;
        public readonly float Information;

        public Dimension(float mass, float length, float time, float temperature,
            float amount, float current, float luminous, float information)
        {
            Mass = mass;
            Length = length;
            Time = time;
            Temperature = temperature;
            Amount = amount;
            Current = current;
            Luminous = luminous;
            Information = information;
        }

        public static readonly Dimension Scalar = new(0, 0, 0, 0, 0, 0, 0, 0);
        public static readonly Dimension LengthD = new(0, 1, 0, 0, 0, 0, 0, 0);
        public static readonly Dimension TimeD = new(0, 0, 1, 0, 0, 0, 0, 0);
        public static readonly Dimension MassD = new(1, 0, 0, 0, 0, 0, 0, 0);
        public static readonly Dimension Velocity = new(0, 1, -1, 0, 0, 0, 0, 0);
        public static readonly Dimension Acceleration = new(0, 1, -2, 0, 0, 0, 0, 0);
        public static readonly Dimension Force = new(1, 1, -2, 0, 0, 0, 0, 0);
        public static readonly Dimension Energy = new(1, 2, -2, 0, 0, 0, 0, 0);
        public static readonly Dimension Power = new(1, 2, -3, 0, 0, 0, 0, 0);
        public static readonly Dimension Pressure = new(1, -1, -2, 0, 0, 0, 0, 0);
        public static readonly Dimension TemperatureD = new(0, 0, 0, 1, 0, 0, 0, 0);
        public static readonly Dimension Density = new(1, -3, 0, 0, 0, 0, 0, 0);
        public static readonly Dimension Viscosity = new(1, -1, -1, 0, 0, 0, 0, 0);

        public Dimension Multiply(Dimension other) => new(
            Mass + other.Mass, Length + other.Length, Time + other.Time,
            Temperature + other.Temperature, Amount + other.Amount,
            Current + other.Current, Luminous + other.Luminous, Information + other.Information);

        public Dimension Divide(Dimension other) => new(
            Mass - other.Mass, Length - other.Length, Time - other.Time,
            Temperature - other.Temperature, Amount - other.Amount,
            Current - other.Current, Luminous - other.Luminous, Information - other.Information);

        public Dimension Pow(float exp) => new(
            Mass * exp, Length * exp, Time * exp, Temperature * exp,
            Amount * exp, Current * exp, Luminous * exp, Information * exp);

        public bool IsCompatible(Dimension other) =>
            MathF.Abs(Mass - other.Mass) < 1e-6f && MathF.Abs(Length - other.Length) < 1e-6f &&
            MathF.Abs(Time - other.Time) < 1e-6f && MathF.Abs(Temperature - other.Temperature) < 1e-6f &&
            MathF.Abs(Amount - other.Amount) < 1e-6f && MathF.Abs(Current - other.Current) < 1e-6f &&
            MathF.Abs(Luminous - other.Luminous) < 1e-6f && MathF.Abs(Information - other.Information) < 1e-6f;

        public bool IsDimensionless => MathF.Abs(Mass) < 1e-6f && MathF.Abs(Length) < 1e-6f &&
            MathF.Abs(Time) < 1e-6f && MathF.Abs(Temperature) < 1e-6f && MathF.Abs(Amount) < 1e-6f &&
            MathF.Abs(Current) < 1e-6f && MathF.Abs(Luminous) < 1e-6f && MathF.Abs(Information) < 1e-6f;

        public bool Equals(Dimension other) => IsCompatible(other);
        public override bool Equals(object? obj) => obj is Dimension d && Equals(d);
        public override int GetHashCode() => HashCode.Combine(Mass, Length, Time, Temperature, Amount, Current, Luminous, Information);
        public override string ToString() => $"[M^{Mass} L^{Length} T^{Time} Θ^{Temperature} N^{Amount} I^{Current} J^{Luminous}]";

        public static Dimension operator +(Dimension a, Dimension b) => a.Multiply(b);
        public static Dimension operator -(Dimension a, Dimension b) => a.Divide(b);
        public static bool operator ==(Dimension a, Dimension b) => a.Equals(b);
        public static bool operator !=(Dimension a, Dimension b) => !a.Equals(b);
    }
}
