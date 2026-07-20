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


//  SECTION 3 — LawFactory

/// <summary>
/// Factory for creating and discovering laws at runtime.
/// </summary>
public static class LawFactory
{
    /// <summary>Creates a well-known law by its identifier.</summary>
    public static LawDefinition CreateLaw(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        if (LawLibraryRegistry.TryGet(id, out var existing))
            return existing;
        throw new KeyNotFoundException($"No built-in law with id '{id}'. Use CreateCustomLaw for custom definitions.");
    }

    /// <summary>Creates a custom law with the given expression and category.</summary>
    public static LawDefinition CreateCustomLaw(string expression, LawCategory category, string? name = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(expression);
        var id = $"custom.{category.ToString().ToLowerInvariant()}.{GetDeterministicHash(expression)}";
        return new LawDefinition
        {
            Id = id,
            Name = name ?? $"Custom {category} Law",
            Category = category,
            Expression = expression,
            Description = $"User-defined law: {expression}",
            Parameters = [],
            Variables = ExtractVariables(expression),
            BoundaryConditionDefs = [],
            ApplicableDomains = [category.ToString()],
            Reference = "User-defined",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            IsExact = false,
            Restriction = DomainRestriction.Empirical,
            Tags = ["custom", category.ToString().ToLowerInvariant()]
        };
    }

    /// <summary>Auto-detects category from an expression string and creates the law.</summary>
    public static LawDefinition FromExpression(string expr)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(expr);
        var category = DetectCategory(expr);
        return CreateCustomLaw(expr, category);
    }

    /// <summary>Attempts to auto-detect the category from expression tokens.</summary>
    public static LawCategory DetectCategory(string expression)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(expression);
        var lower = expression.ToLowerInvariant();

        if (ContainsAny(lower, "force", "mass", "acceleration", "momentum", "torque", "inertia"))
            return LawCategory.Mechanics;
        if (ContainsAny(lower, "temperature", "heat", "entropy", "enthalpy", "thermal"))
            return LawCategory.Thermodynamics;
        if (ContainsAny(lower, "electric", "magnetic", "voltage", "current", "charge", "capacitance", "inductance"))
            return LawCategory.Electromagnetism;
        if (ContainsAny(lower, "velocity", "pressure", "viscosity", "reynolds", "navier"))
            return LawCategory.FluidDynamics;
        if (ContainsAny(lower, "wavefunction", "hamiltonian", "eigenvalue", "planck", "quantum"))
            return LawCategory.QuantumMechanics;
        if (ContainsAny(lower, "refractive", "wavelength", "lens", "diffraction", "intensity", "polariz"))
            return LawCategory.Optics;
        if (ContainsAny(lower, "gravitational", "spacetime", "curvature", "metric", "cosmolog"))
            return LawCategory.Gravitation;
        if (ContainsAny(lower, "reaction", "diffusion", "concentration", "catalyst", "ph", "buffer"))
            return LawCategory.Chemistry;
        if (ContainsAny(lower, "population", "growth", "prey", "predator", "enzyme", "neuron"))
            return LawCategory.Biology;
        if (ContainsAny(lower, "option", "interest rate", "volatility", "portfolio", "betting"))
            return LawCategory.Finance;
        if (ContainsAny(lower, "epidemic", "infection", "susceptible", "recovered", "r0"))
            return LawCategory.Epidemiology;
        if (ContainsAny(lower, "stress", "strain", "fatigue", "hardness", "creep"))
            return LawCategory.MaterialScience;
        if (ContainsAny(lower, "sound", "acoustic", "frequency", "decibel", "doppler"))
            return LawCategory.Acoustics;
        if (ContainsAny(lower, "plasma", "debye", "alfven", "cyclotron", "ionized"))
            return LawCategory.Plasma;
        if (ContainsAny(lower, "nuclear", "decay", "fission", "fusion", "radioactive"))
            return LawCategory.Nuclear;
        if (ContainsAny(lower, "star", "galaxy", "luminosity", "cosmic", "hubble"))
            return LawCategory.Astrophysics;
        if (ContainsAny(lower, "climate", "greenhouse", "albedo", "radiative", "moisture"))
            return LawCategory.Climate;
        if (ContainsAny(lower, "membrane", "synaptic", "action potential", "cortex"))
            return LawCategory.Neuroscience;

        return LawCategory.Mechanics; // default
    }

    private static bool ContainsAny(string source, params string[] values)
    {
        foreach (var v in values)
        {
            if (source.Contains(v, StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    private static IReadOnlyList<LawVariable> ExtractVariables(string expression)
    {
        var variables = new List<LawVariable>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Simple heuristic: extract single-letter and underscore identifiers
        // that look like mathematical variables
        var tokens = expression.Split(new[] { ' ', '(', ')', '+', '-', '*', '/', '^', '=', ',', ';', '[', ']', '{', '}', '<', '>', '~', '|', '!' },
            StringSplitOptions.RemoveEmptyEntries);

        foreach (var token in tokens)
        {
            var clean = token.Trim();
            if (clean.Length == 0 || seen.Contains(clean))
                continue;

            // Skip known constants and keywords
            if (IsKnownConstant(clean))
                continue;
            if (clean.Length <= 3 && !clean.Any(char.IsDigit))
            {
                seen.Add(clean);
                variables.Add(new LawVariable(clean, LawVariableType.Scalar, "varies", "Extracted variable"));
            }
        }

        return variables;
    }

    private static bool IsKnownConstant(string token)
    {
        var known = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "pi", "e", "i", "exp", "ln", "log", "sin", "cos", "tan", "sqrt",
            "del", "grad", "div", "curl", "oint", "integral", "delta",
            "sigma", "mu", "epsilon", "alpha", "beta", "gamma", "omega",
            "phi", "theta", "psi", "rho", "tau", "lambda", "kappa",
        };
        return known.Contains(token);
    }

    private static string GetDeterministicHash(string input)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(input);
        unchecked
        {
            int hash = 23;
            foreach (var b in bytes)
                hash = hash * 31 + b;
            return Math.Abs(hash).ToString("x8");
        }
    }
}
