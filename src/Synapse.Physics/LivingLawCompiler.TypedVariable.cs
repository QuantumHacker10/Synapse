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

    /// <summary>Represents a typed variable in the expression system.</summary>
    public readonly struct TypedVariable : IEquatable<TypedVariable>
    {
        public readonly string Name;
        public readonly VariableType Type;
        public readonly Dimension Dim;

        public TypedVariable(string name, VariableType type, Dimension dim)
        {
            Name = name;
            Type = type;
            Dim = dim;
        }

        public bool Equals(TypedVariable other) => Name == other.Name && Type == other.Type && Dim.Equals(other.Dim);
        public override bool Equals(object? obj) => obj is TypedVariable tv && Equals(tv);
        public override int GetHashCode() => HashCode.Combine(Name, Type, Dim);
        public override string ToString() => $"{Type} {Name} {Dim}";
    }
}
