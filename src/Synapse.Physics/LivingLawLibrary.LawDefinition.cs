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


/// <summary>The immutable, complete definition of a physical law.</summary>
public sealed record LawDefinition
{
    /// <summary>Unique identifier (e.g. "mechanics.newton2").</summary>
    public string Id { get; init; }

    /// <summary>Human-readable name.</summary>
    public string Name { get; init; }

    /// <summary>Broad scientific category.</summary>
    public LawCategory Category { get; init; }

    /// <summary>Mathematical expression (LaTeX or plain-text).</summary>
    public string Expression { get; init; }

    /// <summary>Plain-text description of the law.</summary>
    public string Description { get; init; }

    /// <summary>Ordered list of named parameters.</summary>
    public IReadOnlyList<LawParameter> Parameters { get; init; }

    /// <summary>Variables that appear in the expression.</summary>
    public IReadOnlyList<LawVariable> Variables { get; init; }

    /// <summary>Boundary / applicability conditions.</summary>
    public IReadOnlyList<BoundaryConditionDef> BoundaryConditionDefs { get; init; }

    /// <summary>Specific domains where the law is valid.</summary>
    public IReadOnlyList<string> ApplicableDomains { get; init; }

    /// <summary>Academic reference (paper, book, DOI).</summary>
    public string Reference { get; init; }

    /// <summary>Semantic version string.</summary>
    public string Version { get; init; }

    /// <summary>UTC creation timestamp.</summary>
    public DateTimeOffset CreatedAt { get; init; }

    /// <summary>Severity of applicability restrictions.</summary>
    public DomainRestriction Restriction { get; init; } = DomainRestriction.None;

    /// <summary>Whether the law is exact (not empirical).</summary>
    public bool IsExact { get; init; } = true;

    /// <summary>Optional source code for compiled evaluation.</summary>
    public string? CompiledSource { get; init; }

    /// <summary>Tags for fuzzy search.</summary>
    public IReadOnlyList<string> Tags { get; init; } = [];

    public bool HasParameter(string paramName) =>
        Parameters.Any(p => string.Equals(p.Name, paramName, StringComparison.OrdinalIgnoreCase));

    public bool TryGetParameter(string paramName, out LawParameter param)
    {
        foreach (var p in Parameters)
        {
            if (string.Equals(p.Name, paramName, StringComparison.OrdinalIgnoreCase))
            {
                param = p;
                return true;
            }
        }
        param = default;
        return false;
    }

    public override string ToString() => $"[{Category}] {Name}: {Expression}";
}
