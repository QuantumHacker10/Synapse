// ============================================================
// LawLibraryRegistry.cs - Synapse Omnia Reference Physics Law Library
// The canonical registry of physical laws consumed by LivingLawCompiler.
// C# 14, unsafe code, NativeAOT compatible.
// ============================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;

namespace Synapse.Physics;


/// <summary>Represents a single parameter of a physical law.</summary>
public readonly record struct LawParameter(
    string Name,
    double Value,
    string Unit,
    double MinValue,
    double MaxValue,
    string Description)
{
    public bool IsValid(double value) => value >= MinValue && value <= MaxValue;

    public double Normalize(double value) =>
        Math.Abs(MaxValue - MinValue) < 1e-15
            ? 1.0
            : (value - MinValue) / (MaxValue - MinValue);

    public override string ToString() => $"{Name} = {Value} {Unit} [{MinValue}..{MaxValue}]";
}
