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

// ============================================================
//  SECTION 1 - Core Records and Enums
// ============================================================

/// <summary>Classifies a law into a broad scientific domain.</summary>
public enum LawCategory : byte
{
    Mechanics,
    Thermodynamics,
    Electromagnetism,
    FluidDynamics,
    QuantumMechanics,
    Optics,
    Gravitation,
    Chemistry,
    Biology,
    Finance,
    Epidemiology,
    MaterialScience,
    Acoustics,
    Plasma,
    Nuclear,
    Astrophysics,
    Climate,
    Neuroscience
}

/// <summary>Severity of domain restrictions on a law.</summary>
public enum DomainRestriction : byte
{
    None,
    Conditional,
    Approximate,
    Empirical
}

/// <summary>Variable type classification for law parameters.</summary>
public enum LawVariableType : byte
{
    Scalar,
    Vector,
    Tensor,
    Function,
    Constant,
    Operator,
    Complex
}

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

/// <summary>Represents a variable that appears in a law expression.</summary>
public readonly record struct LawVariable(
    string Name,
    LawVariableType Type,
    string Unit,
    string Description)
{
    public override string ToString() => $"{Name} [{Unit}] ({Type}): {Description}";
}

/// <summary>Boundary condition applied to a law's domain of validity.</summary>
public readonly record struct BoundaryConditionDef(
    string Name,
    string Expression,
    string Description,
    bool Required = true)
{
    public string ToDisplayString() => $"{Name}: {Expression} - {Description}";
}

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

// ============================================================
//  SECTION 2 - LawLibraryRegistry (static, frozen set of all laws)
// ============================================================

/// <summary>
/// Static, immutable library of physical laws.
/// Populated once at startup, then frozen for NativeAOT.
/// </summary>
public static class LawLibraryRegistry
{
    private static readonly FrozenDictionary<string, LawDefinition> _laws;

    /// <summary>Total number of laws in the library.</summary>
    public static int Count => _laws.Count;

    static LawLibraryRegistry()
    {
        var builder = ImmutableDictionary.CreateBuilder<string, LawDefinition>();

        RegisterMechanics(builder);
        RegisterThermodynamics(builder);
        RegisterElectromagnetism(builder);
        RegisterFluidDynamics(builder);
        RegisterQuantumMechanics(builder);
        RegisterOptics(builder);
        RegisterGravitation(builder);
        RegisterChemistry(builder);
        RegisterBiology(builder);
        RegisterFinance(builder);
        RegisterEpidemiology(builder);
        RegisterMaterialScience(builder);
        RegisterAcoustics(builder);
        RegisterPlasma(builder);
        RegisterNuclear(builder);
        RegisterAstrophysics(builder);
        RegisterClimate(builder);
        RegisterNeuroscience(builder);

        _laws = builder.ToFrozenDictionary(StringComparer.OrdinalIgnoreCase);
    }

    public static bool TryGet(string id, [NotNullWhen(true)] out LawDefinition? law) =>
        _laws.TryGetValue(id, out law);

    public static LawDefinition Get(string id) =>
        _laws.TryGetValue(id, out var law) ? law : throw new KeyNotFoundException($"Law '{id}' not found.");

    public static IReadOnlyCollection<LawDefinition> GetAll() => _laws.Values;

    public static IEnumerable<LawDefinition> Search(string query)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        return _laws.Values.Where(l =>
            l.Id.Contains(query, StringComparison.OrdinalIgnoreCase) ||
            l.Name.Contains(query, StringComparison.OrdinalIgnoreCase) ||
            l.Description.Contains(query, StringComparison.OrdinalIgnoreCase));
    }

    public static IEnumerable<LawDefinition> ByCategory(LawCategory category) =>
        _laws.Values.Where(l => l.Category == category);

    public static IEnumerable<LawDefinition> ByTag(string tag) =>
        _laws.Values.Where(l => l.Tags.Contains(tag, StringComparer.OrdinalIgnoreCase));

    public static IEnumerable<string> AllIds() => _laws.Keys;

    // ==============================================================
    //  MECHANICS (25 laws)
    // ==============================================================
    private static void RegisterMechanics(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("mechanics.newton2", new LawDefinition
        {
            Id = "mechanics.newton2",
            Name = "Newton's Second Law",
            Category = LawCategory.Mechanics,
            Expression = "F = m * a",
            Description = "The net force on an object equals its mass times its acceleration.",
            Parameters = [new("m", 1.0, "kg", 0, 1e30, "Mass of the object")],
            Variables = [
                new("F", LawVariableType.Vector, "N", "Net force"),
                new("m", LawVariableType.Scalar, "kg", "Mass"),
                new("a", LawVariableType.Vector, "m/s^2", "Acceleration"),
            ],
            BoundaryConditionDefs = [
                new("NonRelativistic", "v << c", "Valid for velocities much less than the speed of light"),
                new("PointMass", "rigid body assumed", "Object treated as a point mass or rigid body"),
            ],
            ApplicableDomains = ["Classical mechanics", "Engineering dynamics", "Orbital mechanics"],
            Reference = "Newton, I. (1687). Philosophiae Naturalis Principia Mathematica.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["newton", "force", "acceleration", "classical"]
        });

        b.Add("mechanics.hooke", new LawDefinition
        {
            Id = "mechanics.hooke",
            Name = "Hooke's Law",
            Category = LawCategory.Mechanics,
            Expression = "F = -k * x",
            Description = "The restoring force of a spring is proportional to its displacement.",
            Parameters = [new("k", 100.0, "N/m", 0, 1e9, "Spring constant")],
            Variables = [
                new("F", LawVariableType.Vector, "N", "Restoring force"),
                new("k", LawVariableType.Scalar, "N/m", "Spring constant"),
                new("x", LawVariableType.Vector, "m", "Displacement from equilibrium"),
            ],
            BoundaryConditionDefs = [
                new("ElasticLimit", "stress < yield", "Stress must remain below the yield point"),
                new("SmallDeformation", "|x| << L", "Displacement small relative to spring length"),
            ],
            ApplicableDomains = ["Elasticity", "Structural mechanics", "Vibrations"],
            Reference = "Hooke, R. (1678). De Potentia Restitutiva.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["hooke", "spring", "elasticity"]
        });

        b.Add("mechanics.euler-bernoulli", new LawDefinition
        {
            Id = "mechanics.euler-bernoulli",
            Name = "Euler-Bernoulli Beam Theory",
            Category = LawCategory.Mechanics,
            Expression = "EI * (d^4 w / dx^4) = q(x)",
            Description = "Relates beam deflection to applied load using fourth-order ODE.",
            Parameters = [
                new("E", 200e9, "Pa", 0, 1e15, "Young's modulus"),
                new("I", 1e-4, "m^4", 0, 1e6, "Second moment of area"),
            ],
            Variables = [
                new("E", LawVariableType.Scalar, "Pa", "Young's modulus"),
                new("I", LawVariableType.Scalar, "m^4", "Second moment of area"),
                new("w", LawVariableType.Function, "m", "Deflection as function of x"),
                new("q", LawVariableType.Function, "N/m", "Distributed load"),
                new("x", LawVariableType.Scalar, "m", "Position along beam"),
            ],
            BoundaryConditionDefs = [
                new("SlenderBeam", "L/h >> 10", "Beam length much greater than height"),
                new("LinearElastic", "stress < yield", "Material in linear elastic regime"),
            ],
            ApplicableDomains = ["Structural engineering", "Civil engineering", "Mechanical design"],
            Reference = "Euler, L. (1744). / Bernoulli, J. (1726).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["beam", "deflection", "structural", "elasticity"]
        });

        b.Add("mechanics.vonmises", new LawDefinition
        {
            Id = "mechanics.vonmises",
            Name = "von Mises Yield Criterion",
            Category = LawCategory.Mechanics,
            Expression = "sigma_vm = sqrt(sigma_1^2 - sigma_1*sigma_2 + sigma_2^2)",
            Description = "Yields when the von Mises stress exceeds the yield strength.",
            Parameters = [new("sigma_y", 250e6, "Pa", 0, 5e9, "Yield strength")],
            Variables = [
                new("sigma_vm", LawVariableType.Scalar, "Pa", "von Mises stress"),
                new("sigma_1", LawVariableType.Scalar, "Pa", "Principal stress 1"),
                new("sigma_2", LawVariableType.Scalar, "Pa", "Principal stress 2"),
                new("sigma_y", LawVariableType.Scalar, "Pa", "Yield strength"),
            ],
            BoundaryConditionDefs = [
                new("PlaneStress", "sigma_3 = 0", "Plane stress conditions"),
                new("IsotropicMaterial", "material is isotropic", "Isotropic yielding"),
            ],
            ApplicableDomains = ["Plasticity", "Failure analysis", "Structural engineering"],
            Reference = "von Mises, R. (1913). Mechanik der festen Koerper.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["yield", "plasticity", "failure", "stress"]
        });

        b.Add("mechanics.ramberg-osgood", new LawDefinition
        {
            Id = "mechanics.ramberg-osgood",
            Name = "Ramberg-Osgood Constitutive Law",
            Category = LawCategory.Mechanics,
            Expression = "epsilon = sigma/E + 0.002 * (sigma/sigma_y)^n",
            Description = "Non-linear stress-strain behavior with a power-law plastic term.",
            Parameters = [
                new("E", 200e9, "Pa", 0, 1e15, "Young's modulus"),
                new("sigma_y", 250e6, "Pa", 0, 5e9, "0.2% offset yield strength"),
                new("n", 5.0, "dimensionless", 1, 50, "Ramberg-Osgood exponent"),
            ],
            Variables = [
                new("epsilon", LawVariableType.Scalar, "dimensionless", "Total strain"),
                new("sigma", LawVariableType.Scalar, "Pa", "Applied stress"),
                new("E", LawVariableType.Scalar, "Pa", "Young's modulus"),
                new("sigma_y", LawVariableType.Scalar, "Pa", "Yield strength"),
                new("n", LawVariableType.Scalar, "dimensionless", "Hardening exponent"),
            ],
            BoundaryConditionDefs = [
                new("MonotonicLoading", "no unloading", "Monotonic tension"),
            ],
            ApplicableDomains = ["Metal plasticity", "Cyclic loading", "Low-cycle fatigue"],
            Reference = "Ramberg, W. & Osgood, W.R. (1943). NACA TN 902.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["plasticity", "stress-strain", "nonlinear"]
        });

        b.Add("mechanics.voigt-kelvin", new LawDefinition
        {
            Id = "mechanics.voigt-kelvin",
            Name = "Voigt-Kelvin Viscoelasticity Model",
            Category = LawCategory.Mechanics,
            Expression = "sigma = E * epsilon + eta * (d epsilon/dt)",
            Description = "Spring and dashpot in parallel; models viscous damping in solids.",
            Parameters = [
                new("E", 1e9, "Pa", 0, 1e15, "Elastic modulus"),
                new("eta", 1e6, "Pa*s", 0, 1e12, "Dashpot viscosity"),
            ],
            Variables = [
                new("sigma", LawVariableType.Scalar, "Pa", "Total stress"),
                new("epsilon", LawVariableType.Scalar, "dimensionless", "Strain"),
                new("t", LawVariableType.Scalar, "s", "Time"),
                new("E", LawVariableType.Scalar, "Pa", "Elastic modulus"),
                new("eta", LawVariableType.Scalar, "Pa*s", "Dashpot viscosity"),
            ],
            BoundaryConditionDefs = [
                new("SmallStrain", "epsilon < 0.05", "Small strain assumption"),
            ],
            ApplicableDomains = ["Viscoelasticity", "Polymer mechanics", "Biomechanics"],
            Reference = "Kelvin, W.T. (1875). Mathematical and Physical Papers, Vol. III.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["viscoelastic", "damping", "creep", "polymer"]
        });

        b.Add("mechanics.dalembert", new LawDefinition
        {
            Id = "mechanics.dalembert",
            Name = "D'Alembert's Principle",
            Category = LawCategory.Mechanics,
            Expression = "Sum(F_i - m_i * a_i) dot delta_r_i = 0",
            Description = "Extends Newton's second law to constrained systems via virtual displacements.",
            Parameters = [],
            Variables = [
                new("F_i", LawVariableType.Vector, "N", "Applied force on particle i"),
                new("m_i", LawVariableType.Scalar, "kg", "Mass of particle i"),
                new("a_i", LawVariableType.Vector, "m/s^2", "Acceleration of particle i"),
                new("delta_r_i", LawVariableType.Vector, "m", "Virtual displacement"),
            ],
            BoundaryConditionDefs = [
                new("HolonomicConstraints", "constraints are integrable", "Holonomic constraints"),
            ],
            ApplicableDomains = ["Analytical mechanics", "Lagrangian mechanics", "Robotics"],
            Reference = "d'Alembert, J.L.R. (1743). Traite de Dynamique.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["virtual work", "constraints", "analytical"]
        });

        b.Add("mechanics.lagrangian", new LawDefinition
        {
            Id = "mechanics.lagrangian",
            Name = "Euler-Lagrange Equation",
            Category = LawCategory.Mechanics,
            Expression = "d/dt(partial L/partial q_dot) - partial L/partial q = 0",
            Description = "Equations of motion derived from the Lagrangian L = T - V.",
            Parameters = [],
            Variables = [
                new("L", LawVariableType.Scalar, "J", "Lagrangian (T - V)"),
                new("T", LawVariableType.Scalar, "J", "Kinetic energy"),
                new("V", LawVariableType.Scalar, "J", "Potential energy"),
                new("q", LawVariableType.Scalar, "varies", "Generalized coordinate"),
                new("q_dot", LawVariableType.Scalar, "varies/s", "Generalized velocity"),
                new("t", LawVariableType.Scalar, "s", "Time"),
            ],
            BoundaryConditionDefs = [
                new("GeneralizedCoordinates", "q is well-defined", "System describable by generalized coordinates"),
            ],
            ApplicableDomains = ["Analytical mechanics", "Field theory", "Control theory"],
            Reference = "Euler, L. (1744) / Lagrange, J.L. (1788). Mecanique Analytique.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["lagrangian", "euler-lagrange", "analytical"]
        });

        b.Add("mechanics.hamilton-principle", new LawDefinition
        {
            Id = "mechanics.hamilton-principle",
            Name = "Hamilton's Principle of Least Action",
            Category = LawCategory.Mechanics,
            Expression = "delta integral(L dt) = 0",
            Description = "The true path extremizes the action integral.",
            Parameters = [],
            Variables = [
                new("L", LawVariableType.Scalar, "J", "Lagrangian"),
                new("t", LawVariableType.Scalar, "s", "Time"),
                new("S", LawVariableType.Scalar, "J*s", "Action"),
            ],
            BoundaryConditionDefs = [
                new("FixedEndpoints", "delta q(t1) = delta q(t2) = 0", "Variations vanish at endpoints"),
            ],
            ApplicableDomains = ["Classical mechanics", "Field theory", "Quantum mechanics path integrals"],
            Reference = "Hamilton, W.R. (1834). On a General Method in Dynamics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["action", "variational", "least action"]
        });

        b.Add("mechanics.kepler", new LawDefinition
        {
            Id = "mechanics.kepler",
            Name = "Kepler's Third Law (Period-Semimajor Axis)",
            Category = LawCategory.Mechanics,
            Expression = "T^2 = (4 pi^2 / GM) * a^3",
            Description = "Orbital period squared is proportional to semi-major axis cubed.",
            Parameters = [
                new("G", 6.674e-11, "N*m^2/kg^2", 0, 1e-5, "Gravitational constant"),
                new("M", 1.989e30, "kg", 0, 1e40, "Mass of central body"),
            ],
            Variables = [
                new("T", LawVariableType.Scalar, "s", "Orbital period"),
                new("a", LawVariableType.Scalar, "m", "Semi-major axis"),
                new("G", LawVariableType.Scalar, "N*m^2/kg^2", "Gravitational constant"),
                new("M", LawVariableType.Scalar, "kg", "Central body mass"),
            ],
            BoundaryConditionDefs = [
                new("TwoBody", "two-body problem", "Gravitational two-body interaction"),
                new("KeplerianOrbit", "eccentricity e < 1", "Bound elliptical orbit"),
            ],
            ApplicableDomains = ["Celestial mechanics", "Orbital dynamics"],
            Reference = "Kepler, J. (1609). Astronomia Nova.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["kepler", "orbit", "ellipse", "period"]
        });

        b.Add("mechanics.angular-momentum", new LawDefinition
        {
            Id = "mechanics.angular-momentum",
            Name = "Conservation of Angular Momentum",
            Category = LawCategory.Mechanics,
            Expression = "L = I * omega = const (if tau_ext = 0)",
            Description = "Angular momentum is conserved when no external torque acts.",
            Parameters = [],
            Variables = [
                new("L", LawVariableType.Vector, "kg*m^2/s", "Angular momentum"),
                new("I", LawVariableType.Tensor, "kg*m^2", "Moment of inertia tensor"),
                new("omega", LawVariableType.Vector, "rad/s", "Angular velocity"),
                new("tau_ext", LawVariableType.Vector, "N*m", "External torque"),
            ],
            BoundaryConditionDefs = [
                new("ClosedSystem", "tau_ext = 0", "No external torque"),
            ],
            ApplicableDomains = ["Classical mechanics", "Orbital mechanics", "Rigid body dynamics"],
            Reference = "Conservation law - derived from rotational symmetry (Noether's theorem).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["conservation", "angular momentum", "rotation"]
        });

        b.Add("mechanics.energy-conservation", new LawDefinition
        {
            Id = "mechanics.energy-conservation",
            Name = "Conservation of Mechanical Energy",
            Category = LawCategory.Mechanics,
            Expression = "T1 + V1 = T2 + V2",
            Description = "Total mechanical energy is conserved when only conservative forces act.",
            Parameters = [],
            Variables = [
                new("T", LawVariableType.Scalar, "J", "Kinetic energy"),
                new("V", LawVariableType.Scalar, "J", "Potential energy"),
            ],
            BoundaryConditionDefs = [
                new("ConservativeForces", "no non-conservative work", "No friction or drag"),
            ],
            ApplicableDomains = ["Classical mechanics", "Orbital mechanics", "Vibrations"],
            Reference = "Conservation law - follows from time-translation symmetry (Noether's theorem).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["conservation", "energy", "kinetic", "potential"]
        });

        b.Add("mechanics.work-energy", new LawDefinition
        {
            Id = "mechanics.work-energy",
            Name = "Work-Energy Theorem",
            Category = LawCategory.Mechanics,
            Expression = "W_net = Delta K = 0.5*m*(v2^2 - v1^2)",
            Description = "Net work done on an object equals its change in kinetic energy.",
            Parameters = [new("m", 1.0, "kg", 0, 1e30, "Mass")],
            Variables = [
                new("W_net", LawVariableType.Scalar, "J", "Net work done"),
                new("K", LawVariableType.Scalar, "J", "Kinetic energy"),
                new("v1", LawVariableType.Scalar, "m/s", "Initial speed"),
                new("v2", LawVariableType.Scalar, "m/s", "Final speed"),
                new("m", LawVariableType.Scalar, "kg", "Mass"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Classical mechanics"],
            Reference = "Derived from Newton's second law integrated over displacement.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["work", "energy", "kinetic"]
        });

        b.Add("mechanics.impulse-momentum", new LawDefinition
        {
            Id = "mechanics.impulse-momentum",
            Name = "Impulse-Momentum Theorem",
            Category = LawCategory.Mechanics,
            Expression = "J = integral(F dt) = Delta p = m*(v2 - v1)",
            Description = "Impulse applied to an object equals its change in momentum.",
            Parameters = [new("m", 1.0, "kg", 0, 1e30, "Mass")],
            Variables = [
                new("J", LawVariableType.Vector, "N*s", "Impulse"),
                new("F", LawVariableType.Vector, "N", "Force"),
                new("p", LawVariableType.Vector, "kg*m/s", "Momentum"),
                new("v1", LawVariableType.Vector, "m/s", "Initial velocity"),
                new("v2", LawVariableType.Vector, "m/s", "Final velocity"),
                new("m", LawVariableType.Scalar, "kg", "Mass"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Classical mechanics", "Collisions", "Impact analysis"],
            Reference = "Derived from Newton's second law integrated over time.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["impulse", "momentum", "collision"]
        });

        b.Add("mechanics.centripetal", new LawDefinition
        {
            Id = "mechanics.centripetal",
            Name = "Centripetal Acceleration",
            Category = LawCategory.Mechanics,
            Expression = "a_c = v^2 / r = omega^2 * r",
            Description = "Acceleration directed toward center of curvature for circular motion.",
            Parameters = [],
            Variables = [
                new("a_c", LawVariableType.Vector, "m/s^2", "Centripetal acceleration"),
                new("v", LawVariableType.Scalar, "m/s", "Tangential speed"),
                new("r", LawVariableType.Scalar, "m", "Radius of curvature"),
                new("omega", LawVariableType.Scalar, "rad/s", "Angular velocity"),
            ],
            BoundaryConditionDefs = [
                new("CircularMotion", "constant speed", "Uniform circular motion"),
            ],
            ApplicableDomains = ["Circular motion", "Centrifuges", "Orbital mechanics"],
            Reference = "Huygens, C. (1673). Horologium Oscillatorium.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["centripetal", "circular", "acceleration"]
        });

        b.Add("mechanics.universal-gravitation", new LawDefinition
        {
            Id = "mechanics.universal-gravitation",
            Name = "Newton's Law of Universal Gravitation",
            Category = LawCategory.Mechanics,
            Expression = "F = G * m1 * m2 / r^2",
            Description = "Every mass attracts every other mass; inverse-square law.",
            Parameters = [new("G", 6.674e-11, "N*m^2/kg^2", 0, 1e-5, "Gravitational constant")],
            Variables = [
                new("F", LawVariableType.Scalar, "N", "Gravitational force"),
                new("m1", LawVariableType.Scalar, "kg", "Mass of body 1"),
                new("m2", LawVariableType.Scalar, "kg", "Mass of body 2"),
                new("r", LawVariableType.Scalar, "m", "Distance between centers"),
                new("G", LawVariableType.Scalar, "N*m^2/kg^2", "Gravitational constant"),
            ],
            BoundaryConditionDefs = [
                new("NonRelativistic", "v << c", "Newtonian regime"),
                new("PointMass", "spherically symmetric", "Point masses"),
            ],
            ApplicableDomains = ["Gravitation", "Orbital mechanics", "Astrophysics"],
            Reference = "Newton, I. (1687). Principia Mathematica.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["gravitation", "universal", "inverse square"]
        });

        b.Add("mechanics.shm", new LawDefinition
        {
            Id = "mechanics.shm",
            Name = "Simple Harmonic Motion",
            Category = LawCategory.Mechanics,
            Expression = "x(t) = A * cos(omega_0 * t + phi)",
            Description = "Oscillatory motion where restoring force is proportional to displacement.",
            Parameters = [new("omega_0", 1.0, "rad/s", 0, 1e6, "Natural angular frequency")],
            Variables = [
                new("x", LawVariableType.Scalar, "m", "Displacement"),
                new("A", LawVariableType.Scalar, "m", "Amplitude"),
                new("t", LawVariableType.Scalar, "s", "Time"),
                new("phi", LawVariableType.Scalar, "rad", "Phase constant"),
                new("omega_0", LawVariableType.Scalar, "rad/s", "Natural frequency"),
            ],
            BoundaryConditionDefs = [
                new("SmallOscillation", "displacement << natural length", "Small amplitude"),
            ],
            ApplicableDomains = ["Vibrations", "Acoustics", "Circuits"],
            Reference = "Hooke, R. (1678); Euler, L. (1729).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["oscillation", "harmonic", "frequency", "spring"]
        });

        b.Add("mechanics.reduced-mass", new LawDefinition
        {
            Id = "mechanics.reduced-mass",
            Name = "Reduced Mass",
            Category = LawCategory.Mechanics,
            Expression = "mu = (m1 * m2) / (m1 + m2)",
            Description = "Effective inertial mass of a two-body system.",
            Parameters = [],
            Variables = [
                new("mu", LawVariableType.Scalar, "kg", "Reduced mass"),
                new("m1", LawVariableType.Scalar, "kg", "Mass of body 1"),
                new("m2", LawVariableType.Scalar, "kg", "Mass of body 2"),
            ],
            BoundaryConditionDefs = [
                new("TwoBody", "isolated two-body system", "No external forces"),
            ],
            ApplicableDomains = ["Two-body problem", "Molecular vibrations", "Orbital mechanics"],
            Reference = "Standard classical mechanics result.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["reduced mass", "two-body", "effective mass"]
        });

        b.Add("mechanics.pendulum", new LawDefinition
        {
            Id = "mechanics.pendulum",
            Name = "Simple Pendulum (small angle)",
            Category = LawCategory.Mechanics,
            Expression = "T = 2 pi * sqrt(L / g)",
            Description = "Period of a simple pendulum for small oscillations.",
            Parameters = [new("g", 9.80665, "m/s^2", 0, 100, "Gravitational acceleration")],
            Variables = [
                new("T", LawVariableType.Scalar, "s", "Period"),
                new("L", LawVariableType.Scalar, "m", "Length"),
                new("g", LawVariableType.Scalar, "m/s^2", "Gravitational acceleration"),
            ],
            BoundaryConditionDefs = [
                new("SmallAngle", "theta < ~15 deg", "sin(theta) ~ theta approximation"),
                new("MasslessString", "string mass negligible", "Massless inextensible string"),
            ],
            ApplicableDomains = ["Timekeeping", "Gravimetry", "Education"],
            Reference = "Galileo (c. 1602); Huygens (1673).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["pendulum", "oscillation", "period", "gravity"]
        });

        b.Add("mechanics.rocket-eq", new LawDefinition
        {
            Id = "mechanics.rocket-eq",
            Name = "Tsiolkovsky Rocket Equation",
            Category = LawCategory.Mechanics,
            Expression = "delta_v = v_e * ln(m0 / mf)",
            Description = "Maximum velocity change of a rocket from exhaust velocity and mass ratio.",
            Parameters = [new("v_e", 3000.0, "m/s", 0, 1e5, "Effective exhaust velocity")],
            Variables = [
                new("delta_v", LawVariableType.Scalar, "m/s", "Change in velocity"),
                new("v_e", LawVariableType.Scalar, "m/s", "Exhaust velocity"),
                new("m0", LawVariableType.Scalar, "kg", "Initial total mass"),
                new("mf", LawVariableType.Scalar, "kg", "Final total mass"),
            ],
            BoundaryConditionDefs = [
                new("NoGravity", "free space", "Neglects gravity and drag"),
                new("ConstantExhaust", "v_e = const", "Constant exhaust velocity"),
            ],
            ApplicableDomains = ["Rocketry", "Spacecraft propulsion", "Astrophysics"],
            Reference = "Tsiolkovsky, K.E. (1903).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["rocket", "propulsion", "delta-v", "tsiolkovsky"]
        });

        b.Add("mechanics.elastic-collision", new LawDefinition
        {
            Id = "mechanics.elastic-collision",
            Name = "1D Elastic Collision",
            Category = LawCategory.Mechanics,
            Expression = "v1' = ((m1-m2)*v1 + 2*m2*v2) / (m1+m2)",
            Description = "Final velocities after a perfectly elastic 1D collision.",
            Parameters = [],
            Variables = [
                new("v1_prime", LawVariableType.Scalar, "m/s", "Final velocity of mass 1"),
                new("v1", LawVariableType.Scalar, "m/s", "Initial velocity of mass 1"),
                new("v2", LawVariableType.Scalar, "m/s", "Initial velocity of mass 2"),
                new("m1", LawVariableType.Scalar, "kg", "Mass 1"),
                new("m2", LawVariableType.Scalar, "kg", "Mass 2"),
            ],
            BoundaryConditionDefs = [
                new("ElasticCollision", "KE conserved", "No energy loss"),
                new("OneDimensional", "head-on", "Single axis"),
            ],
            ApplicableDomains = ["Collisions", "Particle physics", "Billiard dynamics"],
            Reference = "Derived from conservation of momentum and kinetic energy.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["collision", "elastic", "conservation"]
        });

        b.Add("mechanics.centrifugal", new LawDefinition
        {
            Id = "mechanics.centrifugal",
            Name = "Centrifugal Force (Rotating Frame)",
            Category = LawCategory.Mechanics,
            Expression = "F_cf = m * omega^2 * r",
            Description = "Apparent outward force in a rotating reference frame.",
            Parameters = [],
            Variables = [
                new("F_cf", LawVariableType.Vector, "N", "Centrifugal force"),
                new("m", LawVariableType.Scalar, "kg", "Mass"),
                new("omega", LawVariableType.Scalar, "rad/s", "Angular velocity"),
                new("r", LawVariableType.Scalar, "m", "Distance from axis"),
            ],
            BoundaryConditionDefs = [
                new("RotatingFrame", "observer in rotating frame", "Non-inertial frame"),
            ],
            ApplicableDomains = ["Rotating frames", "Centrifuges", "Meteorology"],
            Reference = "Poincare, H. (1893). Mecanique Celeste.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["centrifugal", "rotating frame", "fictitious force"]
        });
    }

    // ==============================================================
    //  THERMODYNAMICS (14 laws)
    // ==============================================================
    private static void RegisterThermodynamics(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("thermo.heat-eq", new LawDefinition
        {
            Id = "thermo.heat-eq",
            Name = "Heat Equation",
            Category = LawCategory.Thermodynamics,
            Expression = "dT/dt = alpha * del^2(T)",
            Description = "Diffusion of heat through a medium; parabolic PDE.",
            Parameters = [new("alpha", 1e-5, "m^2/s", 0, 1e2, "Thermal diffusivity")],
            Variables = [
                new("T", LawVariableType.Function, "K", "Temperature field"),
                new("t", LawVariableType.Scalar, "s", "Time"),
                new("alpha", LawVariableType.Scalar, "m^2/s", "Thermal diffusivity"),
                new("x", LawVariableType.Vector, "m", "Position"),
            ],
            BoundaryConditionDefs = [
                new("InitialCondition", "T(x,0) = T0(x)", "Initial temperature distribution"),
            ],
            ApplicableDomains = ["Heat transfer", "Materials processing", "Climate modeling"],
            Reference = "Fourier, J. (1822). Theorie Analytique de la Chaleur.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["heat", "diffusion", "parabolic", "PDE"]
        });

        b.Add("thermo.fourier", new LawDefinition
        {
            Id = "thermo.fourier",
            Name = "Fourier's Law of Heat Conduction",
            Category = LawCategory.Thermodynamics,
            Expression = "q = -k * grad(T)",
            Description = "Heat flux is proportional to the negative temperature gradient.",
            Parameters = [new("k", 50.0, "W/(m*K)", 0, 5000, "Thermal conductivity")],
            Variables = [
                new("q", LawVariableType.Vector, "W/m^2", "Heat flux vector"),
                new("k", LawVariableType.Scalar, "W/(m*K)", "Thermal conductivity"),
                new("T", LawVariableType.Function, "K", "Temperature field"),
            ],
            BoundaryConditionDefs = [
                new("FourierRegime", "Knudsen number << 1", "Continuum regime"),
            ],
            ApplicableDomains = ["Heat conduction", "Building insulation", "Electronics cooling"],
            Reference = "Fourier, J. (1822). Theorie Analytique de la Chaleur.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["fourier", "heat flux", "conduction", "thermal conductivity"]
        });

        b.Add("thermo.ideal-gas", new LawDefinition
        {
            Id = "thermo.ideal-gas",
            Name = "Ideal Gas Law",
            Category = LawCategory.Thermodynamics,
            Expression = "P * V = n * R * T",
            Description = "Equation of state for an ideal gas.",
            Parameters = [new("R", 8.31446, "J/(mol*K)", 0, 100, "Universal gas constant")],
            Variables = [
                new("P", LawVariableType.Scalar, "Pa", "Pressure"),
                new("V", LawVariableType.Scalar, "m^3", "Volume"),
                new("n", LawVariableType.Scalar, "mol", "Amount of substance"),
                new("R", LawVariableType.Scalar, "J/(mol*K)", "Gas constant"),
                new("T", LawVariableType.Scalar, "K", "Absolute temperature"),
            ],
            BoundaryConditionDefs = [
                new("IdealGas", "weak intermolecular forces", "Negligible interaction"),
                new("PointParticles", "volume << container", "Negligible molecular volume"),
            ],
            ApplicableDomains = ["Chemistry", "Engineering", "Meteorology"],
            Reference = "Clapeyron, E. (1834) / Boyle, Charles, Avogadro.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["ideal gas", "equation of state", "pressure", "temperature"]
        });

        b.Add("thermo.clausius-clapeyron", new LawDefinition
        {
            Id = "thermo.clausius-clapeyron",
            Name = "Clausius-Clapeyron Equation",
            Category = LawCategory.Thermodynamics,
            Expression = "dP/dT = Delta H / (T * Delta V)",
            Description = "Slope of phase boundary on P-T diagram from latent heat and volume change.",
            Parameters = [],
            Variables = [
                new("P", LawVariableType.Scalar, "Pa", "Phase boundary pressure"),
                new("T", LawVariableType.Scalar, "K", "Phase boundary temperature"),
                new("Delta_H", LawVariableType.Scalar, "J/mol", "Latent heat"),
                new("Delta_V", LawVariableType.Scalar, "m^3/mol", "Volume change"),
            ],
            BoundaryConditionDefs = [
                new("PhaseEquilibrium", "two phases in equilibrium", "Coexistence curve"),
                new("FirstOrderTransition", "latent heat exists", "First-order transition"),
            ],
            ApplicableDomains = ["Phase transitions", "Boiling/condensation", "Sublimation"],
            Reference = "Clausius, R. (1850) / Clapeyron, B.P.E. (1834).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["phase transition", "latent heat", "vapor pressure"]
        });

        b.Add("thermo.stefan-boltzmann", new LawDefinition
        {
            Id = "thermo.stefan-boltzmann",
            Name = "Stefan-Boltzmann Law",
            Category = LawCategory.Thermodynamics,
            Expression = "j* = sigma * T^4",
            Description = "Total radiant energy emitted by a black body proportional to T^4.",
            Parameters = [new("sigma", 5.670374e-8, "W/(m^2*K^4)", 0, 1, "Stefan-Boltzmann constant")],
            Variables = [
                new("j_star", LawVariableType.Scalar, "W/m^2", "Total emissive power"),
                new("sigma", LawVariableType.Scalar, "W/(m^2*K^4)", "SB constant"),
                new("T", LawVariableType.Scalar, "K", "Absolute temperature"),
            ],
            BoundaryConditionDefs = [
                new("BlackBody", "emissivity = 1", "Ideal black body (or j* = eps*sigma*T^4)"),
            ],
            ApplicableDomains = ["Thermal radiation", "Astrophysics", "Climate science"],
            Reference = "Stefan, J. (1879); Boltzmann, L. (1884).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["radiation", "black body", "thermal emission"]
        });

        b.Add("thermo.carnot", new LawDefinition
        {
            Id = "thermo.carnot",
            Name = "Carnot Efficiency",
            Category = LawCategory.Thermodynamics,
            Expression = "eta = 1 - T_cold / T_hot",
            Description = "Maximum theoretical efficiency of a heat engine.",
            Parameters = [],
            Variables = [
                new("eta", LawVariableType.Scalar, "dimensionless", "Efficiency (0 to 1)"),
                new("T_cold", LawVariableType.Scalar, "K", "Cold reservoir temperature"),
                new("T_hot", LawVariableType.Scalar, "K", "Hot reservoir temperature"),
            ],
            BoundaryConditionDefs = [
                new("Reversible", "quasi-static cycle", "No entropy production"),
                new("TwoReservoirs", "only two thermal reservoirs", "Simple two-bath model"),
            ],
            ApplicableDomains = ["Heat engines", "Refrigeration", "Thermodynamic limits"],
            Reference = "Carnot, S. (1824). Reflexions sur la Puissance Motrice du Feu.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["efficiency", "carnot", "heat engine", "reversible"]
        });

        b.Add("thermo.first-law", new LawDefinition
        {
            Id = "thermo.first-law",
            Name = "First Law of Thermodynamics",
            Category = LawCategory.Thermodynamics,
            Expression = "Delta U = Q - W",
            Description = "Change in internal energy equals heat added minus work done.",
            Parameters = [],
            Variables = [
                new("Delta_U", LawVariableType.Scalar, "J", "Change in internal energy"),
                new("Q", LawVariableType.Scalar, "J", "Heat added"),
                new("W", LawVariableType.Scalar, "J", "Work done by system"),
            ],
            BoundaryConditionDefs = [
                new("ClosedSystem", "no mass transfer", "Energy exchange only"),
            ],
            ApplicableDomains = ["All of thermodynamics"],
            Reference = "Joule, J.P. (1845); Helmholtz, H. (1847).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["first law", "energy conservation", "internal energy"]
        });

        b.Add("thermo.second-law", new LawDefinition
        {
            Id = "thermo.second-law",
            Name = "Second Law of Thermodynamics (Entropy)",
            Category = LawCategory.Thermodynamics,
            Expression = "dS >= delta Q / T",
            Description = "Entropy of an isolated system never decreases.",
            Parameters = [],
            Variables = [
                new("S", LawVariableType.Scalar, "J/K", "Entropy"),
                new("Q", LawVariableType.Scalar, "J", "Heat transfer"),
                new("T", LawVariableType.Scalar, "K", "Absolute temperature"),
            ],
            BoundaryConditionDefs = [
                new("IsolatedSystem", "no energy/matter exchange", "Total entropy statement"),
            ],
            ApplicableDomains = ["All of thermodynamics", "Statistical mechanics", "Information theory"],
            Reference = "Clausius, R. (1850); Boltzmann, L. (1877).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["entropy", "second law", "irreversibility"]
        });

        b.Add("thermo.third-law", new LawDefinition
        {
            Id = "thermo.third-law",
            Name = "Third Law of Thermodynamics",
            Category = LawCategory.Thermodynamics,
            Expression = "lim(T->0) S = 0",
            Description = "Entropy of a perfect crystal approaches zero at absolute zero.",
            Parameters = [],
            Variables = [
                new("S", LawVariableType.Scalar, "J/K", "Entropy"),
                new("T", LawVariableType.Scalar, "K", "Absolute temperature"),
            ],
            BoundaryConditionDefs = [
                new("PerfectCrystal", "perfectly ordered crystal", "No residual entropy"),
            ],
            ApplicableDomains = ["Low-temperature physics", "Cryogenics"],
            Reference = "Nernst, W. (1906); Planck, M. (1911).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["third law", "absolute zero", "entropy"]
        });

        b.Add("thermo.specific-heat", new LawDefinition
        {
            Id = "thermo.specific-heat",
            Name = "Specific Heat Capacity",
            Category = LawCategory.Thermodynamics,
            Expression = "Q = m * c * Delta T",
            Description = "Heat to change temperature of a mass.",
            Parameters = [new("c", 4186.0, "J/(kg*K)", 0, 1e5, "Specific heat capacity")],
            Variables = [
                new("Q", LawVariableType.Scalar, "J", "Heat transferred"),
                new("m", LawVariableType.Scalar, "kg", "Mass"),
                new("c", LawVariableType.Scalar, "J/(kg*K)", "Specific heat"),
                new("Delta_T", LawVariableType.Scalar, "K", "Temperature change"),
            ],
            BoundaryConditionDefs = [
                new("ConstantPressure", "c_p variant", "Or constant volume: c_v"),
            ],
            ApplicableDomains = ["Calorimetry", "HVAC", "Materials science"],
            Reference = "Black, J. (1761); Rumford, C. (1798).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["specific heat", "calorimetry", "temperature"]
        });

        b.Add("thermo.vanderwaals", new LawDefinition
        {
            Id = "thermo.vanderwaals",
            Name = "Van der Waals Equation of State",
            Category = LawCategory.Thermodynamics,
            Expression = "(P + a/Vbar^2)(Vbar - b) = RT",
            Description = "Corrects ideal gas law for molecular volume and attraction.",
            Parameters = [
                new("R", 8.31446, "J/(mol*K)", 0, 100, "Gas constant"),
                new("a", 0.0, "Pa*m^6/mol^2", 0, 10, "Attraction parameter"),
                new("b", 0.0, "m^3/mol", 0, 1, "Volume parameter"),
            ],
            Variables = [
                new("P", LawVariableType.Scalar, "Pa", "Pressure"),
                new("Vbar", LawVariableType.Scalar, "m^3/mol", "Molar volume"),
                new("R", LawVariableType.Scalar, "J/(mol*K)", "Gas constant"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("a", LawVariableType.Scalar, "Pa*m^6/mol^2", "Attraction"),
                new("b", LawVariableType.Scalar, "m^3/mol", "Volume"),
            ],
            BoundaryConditionDefs = [
                new("CondensableGas", "near saturation", "Near phase transitions"),
            ],
            ApplicableDomains = ["Real gases", "Phase transitions", "Critical phenomena"],
            Reference = "van der Waals, J.D. (1873).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["van der waals", "real gas", "equation of state"]
        });

        b.Add("thermo.boltzmann-entropy", new LawDefinition
        {
            Id = "thermo.boltzmann-entropy",
            Name = "Boltzmann Entropy Formula",
            Category = LawCategory.Thermodynamics,
            Expression = "S = k_B * ln(Omega)",
            Description = "Entropy is proportional to logarithm of number of microstates.",
            Parameters = [new("k_B", 1.380649e-23, "J/K", 0, 1e-15, "Boltzmann constant")],
            Variables = [
                new("S", LawVariableType.Scalar, "J/K", "Entropy"),
                new("k_B", LawVariableType.Scalar, "J/K", "Boltzmann constant"),
                new("Omega", LawVariableType.Scalar, "dimensionless", "Number of microstates"),
            ],
            BoundaryConditionDefs = [
                new("ErgodicSystem", "all microstates equally likely", "Statistical equilibrium"),
            ],
            ApplicableDomains = ["Statistical mechanics", "Thermodynamics", "Information theory"],
            Reference = "Boltzmann, L. (1877). Wiener Berichte.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["boltzmann", "entropy", "microstates", "statistical"]
        });

        b.Add("thermo.maxwell-boltzmann", new LawDefinition
        {
            Id = "thermo.maxwell-boltzmann",
            Name = "Maxwell-Boltzmann Speed Distribution",
            Category = LawCategory.Thermodynamics,
            Expression = "f(v) = 4*pi*n*(m/(2*pi*k_B*T))^(3/2) * v^2 * exp(-m*v^2/(2*k_B*T))",
            Description = "Probability distribution of particle speeds in an ideal gas.",
            Parameters = [new("k_B", 1.380649e-23, "J/K", 0, 1e-15, "Boltzmann constant")],
            Variables = [
                new("f", LawVariableType.Function, "s/m", "Speed distribution"),
                new("v", LawVariableType.Scalar, "m/s", "Speed"),
                new("m", LawVariableType.Scalar, "kg", "Particle mass"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("n", LawVariableType.Scalar, "m^-3", "Number density"),
                new("k_B", LawVariableType.Scalar, "J/K", "Boltzmann constant"),
            ],
            BoundaryConditionDefs = [
                new("IdealGas", "non-interacting particles", "Point particles"),
                new("ThermalEquilibrium", "T well-defined", "At equilibrium"),
            ],
            ApplicableDomains = ["Kinetic theory", "Gas-phase chemistry", "Plasma physics"],
            Reference = "Maxwell, J.C. (1860) / Boltzmann, L. (1877).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["maxwell-boltzmann", "speed distribution", "kinetic theory"]
        });
    }

    // ==============================================================
    //  ELECTROMAGNETISM (11 laws)
    // ==============================================================
    private static void RegisterElectromagnetism(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("em.coulomb", new LawDefinition
        {
            Id = "em.coulomb",
            Name = "Coulomb's Law",
            Category = LawCategory.Electromagnetism,
            Expression = "F = (1/(4*pi*epsilon_0)) * q1*q2 / r^2",
            Description = "Electrostatic force between two point charges; inverse-square law.",
            Parameters = [new("epsilon_0", 8.8541878128e-12, "F/m", 0, 1e-6, "Vacuum permittivity")],
            Variables = [
                new("F", LawVariableType.Vector, "N", "Electrostatic force"),
                new("q1", LawVariableType.Scalar, "C", "Charge 1"),
                new("q2", LawVariableType.Scalar, "C", "Charge 2"),
                new("r", LawVariableType.Scalar, "m", "Distance"),
                new("epsilon_0", LawVariableType.Scalar, "F/m", "Vacuum permittivity"),
            ],
            BoundaryConditionDefs = [
                new("PointCharges", "point-like charges", "Size negligible"),
                new("StaticCharges", "at rest", "Electrostatic"),
            ],
            ApplicableDomains = ["Electrostatics", "Atomic physics", "Molecular interactions"],
            Reference = "Coulomb, C.A. (1785). Memoires de l'Academie Royale.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["coulomb", "electrostatic", "inverse square", "charge"]
        });

        b.Add("em.gauss", new LawDefinition
        {
            Id = "em.gauss",
            Name = "Gauss's Law (Electric)",
            Category = LawCategory.Electromagnetism,
            Expression = "oint E dot dA = Q_enc / epsilon_0",
            Description = "Electric flux through a closed surface equals enclosed charge over permittivity.",
            Parameters = [new("epsilon_0", 8.8541878128e-12, "F/m", 0, 1e-6, "Vacuum permittivity")],
            Variables = [
                new("E", LawVariableType.Vector, "V/m", "Electric field"),
                new("Q_enc", LawVariableType.Scalar, "C", "Enclosed charge"),
                new("epsilon_0", LawVariableType.Scalar, "F/m", "Vacuum permittivity"),
            ],
            BoundaryConditionDefs = [
                new("ClosedSurface", "Gaussian surface is closed", "Encloses a volume"),
            ],
            ApplicableDomains = ["Electrostatics", "Capacitor design", "Field theory"],
            Reference = "Gauss, C.F. (1835).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["gauss", "electric flux", "gaussian surface"]
        });

        b.Add("em.gauss-mag", new LawDefinition
        {
            Id = "em.gauss-mag",
            Name = "Gauss's Law for Magnetism",
            Category = LawCategory.Electromagnetism,
            Expression = "oint B dot dA = 0",
            Description = "Magnetic flux through any closed surface is zero; no monopoles.",
            Parameters = [],
            Variables = [
                new("B", LawVariableType.Vector, "T", "Magnetic field"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Magnetostatics", "Field theory"],
            Reference = "Maxwell, J.C. (1865).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["gauss", "magnetic", "monopole", "flux"]
        });

        b.Add("em.faraday", new LawDefinition
        {
            Id = "em.faraday",
            Name = "Faraday's Law of Induction",
            Category = LawCategory.Electromagnetism,
            Expression = "oint E dot dl = -d(Phi_B)/dt",
            Description = "Changing magnetic flux induces an electromotive force (EMF).",
            Parameters = [],
            Variables = [
                new("E", LawVariableType.Vector, "V/m", "Electric field"),
                new("Phi_B", LawVariableType.Scalar, "Wb", "Magnetic flux"),
                new("t", LawVariableType.Scalar, "s", "Time"),
            ],
            BoundaryConditionDefs = [
                new("StationaryLoop", "loop is stationary", "Or account for motional EMF"),
            ],
            ApplicableDomains = ["Electromagnetic induction", "Transformers", "Generators"],
            Reference = "Faraday, M. (1831). Experimental Researches in Electricity.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["faraday", "induction", "magnetic flux", "EMF"]
        });

        b.Add("em.ampere-maxwell", new LawDefinition
        {
            Id = "em.ampere-maxwell",
            Name = "Ampere-Maxwell Law",
            Category = LawCategory.Electromagnetism,
            Expression = "oint B dot dl = mu_0*I + mu_0*epsilon_0*d(Phi_E)/dt",
            Description = "Magnetic field circulation sourced by current and changing electric flux.",
            Parameters = [
                new("mu_0", 1.25663706212e-6, "H/m", 0, 1, "Vacuum permeability"),
                new("epsilon_0", 8.8541878128e-12, "F/m", 0, 1e-6, "Vacuum permittivity"),
            ],
            Variables = [
                new("B", LawVariableType.Vector, "T", "Magnetic field"),
                new("I", LawVariableType.Scalar, "A", "Current"),
                new("Phi_E", LawVariableType.Scalar, "V*m", "Electric flux"),
                new("mu_0", LawVariableType.Scalar, "H/m", "Permeability"),
                new("epsilon_0", LawVariableType.Scalar, "F/m", "Permittivity"),
            ],
            BoundaryConditionDefs = [
                new("DisplacementCurrent", "Maxwell correction included", "Both conduction and displacement"),
            ],
            ApplicableDomains = ["Electromagnetics", "Wave propagation", "Antenna theory"],
            Reference = "Ampere, A.M. (1826) / Maxwell, J.C. (1865).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["ampere", "maxwell", "displacement current", "magnetic field"]
        });

        b.Add("em.lorentz", new LawDefinition
        {
            Id = "em.lorentz",
            Name = "Lorentz Force Law",
            Category = LawCategory.Electromagnetism,
            Expression = "F = q*(E + v x B)",
            Description = "Force on a charged particle in electric and magnetic fields.",
            Parameters = [],
            Variables = [
                new("F", LawVariableType.Vector, "N", "Force on particle"),
                new("q", LawVariableType.Scalar, "C", "Electric charge"),
                new("E", LawVariableType.Vector, "V/m", "Electric field"),
                new("v", LawVariableType.Vector, "m/s", "Particle velocity"),
                new("B", LawVariableType.Vector, "T", "Magnetic field"),
            ],
            BoundaryConditionDefs = [
                new("PointCharge", "charge is point-like", "Classical treatment"),
            ],
            ApplicableDomains = ["Particle physics", "Plasma physics", "Beam dynamics"],
            Reference = "Lorentz, H.A. (1895).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["lorentz", "force", "charge", "magnetic", "electric"]
        });

        b.Add("em.poynting", new LawDefinition
        {
            Id = "em.poynting",
            Name = "Poynting Vector (Energy Flux)",
            Category = LawCategory.Electromagnetism,
            Expression = "S = (1/mu_0) * E x B",
            Description = "Energy flux density of an electromagnetic field.",
            Parameters = [new("mu_0", 1.25663706212e-6, "H/m", 0, 1, "Vacuum permeability")],
            Variables = [
                new("S", LawVariableType.Vector, "W/m^2", "Poynting vector"),
                new("E", LawVariableType.Vector, "V/m", "Electric field"),
                new("B", LawVariableType.Vector, "T", "Magnetic field"),
                new("mu_0", LawVariableType.Scalar, "H/m", "Permeability"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["EM wave propagation", "Antenna design", "Optics"],
            Reference = "Poynting, J.H. (1884). Phil. Trans. Royal Society.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["poynting", "energy flux", "EM wave", "power"]
        });

        b.Add("em.maxwell-diff", new LawDefinition
        {
            Id = "em.maxwell-diff",
            Name = "Maxwell's Equations (Differential Form)",
            Category = LawCategory.Electromagnetism,
            Expression = "del.E = rho/epsilon_0; del.B = 0; delxE = -dB/dt; delxB = mu_0*J + mu_0*epsilon_0*dE/dt",
            Description = "Complete classical electrodynamics equations.",
            Parameters = [
                new("epsilon_0", 8.8541878128e-12, "F/m", 0, 1e-6, "Vacuum permittivity"),
                new("mu_0", 1.25663706212e-6, "H/m", 0, 1, "Vacuum permeability"),
            ],
            Variables = [
                new("E", LawVariableType.Vector, "V/m", "Electric field"),
                new("B", LawVariableType.Vector, "T", "Magnetic field"),
                new("rho", LawVariableType.Scalar, "C/m^3", "Charge density"),
                new("J", LawVariableType.Vector, "A/m^2", "Current density"),
            ],
            BoundaryConditionDefs = [
                new("ContinuumFields", "fields are smooth", "Differentiable everywhere"),
            ],
            ApplicableDomains = ["All of classical electrodynamics", "Optics", "Plasma physics"],
            Reference = "Maxwell, J.C. (1865). A Dynamical Theory of the EM Field.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["maxwell", "electrodynamics", "fundamental"]
        });

        b.Add("em.capacitance", new LawDefinition
        {
            Id = "em.capacitance",
            Name = "Parallel Plate Capacitance",
            Category = LawCategory.Electromagnetism,
            Expression = "C = epsilon_0 * A / d",
            Description = "Capacitance of parallel conducting plates.",
            Parameters = [new("epsilon_0", 8.8541878128e-12, "F/m", 0, 1e-6, "Vacuum permittivity")],
            Variables = [
                new("C", LawVariableType.Scalar, "F", "Capacitance"),
                new("A", LawVariableType.Scalar, "m^2", "Plate area"),
                new("d", LawVariableType.Scalar, "m", "Plate separation"),
                new("epsilon_0", LawVariableType.Scalar, "F/m", "Permittivity"),
            ],
            BoundaryConditionDefs = [
                new("SmallGap", "d << sqrt(A)", "Fringing fields negligible"),
            ],
            ApplicableDomains = ["Circuit design", "MEMS", "Capacitive sensors"],
            Reference = "Standard result from Maxwell's equations.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["capacitance", "parallel plate", "electric field"]
        });

        b.Add("em.ohm", new LawDefinition
        {
            Id = "em.ohm",
            Name = "Ohm's Law",
            Category = LawCategory.Electromagnetism,
            Expression = "V = I * R",
            Description = "Voltage across a conductor is proportional to current.",
            Parameters = [new("R", 1.0, "ohm", 0, 1e12, "Electrical resistance")],
            Variables = [
                new("V", LawVariableType.Scalar, "V", "Voltage"),
                new("I", LawVariableType.Scalar, "A", "Current"),
                new("R", LawVariableType.Scalar, "ohm", "Resistance"),
            ],
            BoundaryConditionDefs = [
                new("OhmicMaterial", "linear response", "Constant resistance"),
                new("ConstantTemperature", "T = const", "R varies with temperature"),
            ],
            ApplicableDomains = ["Circuit analysis", "Electronics", "Power systems"],
            Reference = "Ohm, G.S. (1827). Die Galvanische Kette.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["ohm", "voltage", "current", "resistance"]
        });

        b.Add("em.joule", new LawDefinition
        {
            Id = "em.joule",
            Name = "Joule's First Law (Resistive Heating)",
            Category = LawCategory.Electromagnetism,
            Expression = "P = I^2 * R",
            Description = "Power dissipated as heat when current flows through resistance.",
            Parameters = [new("R", 1.0, "ohm", 0, 1e12, "Electrical resistance")],
            Variables = [
                new("P", LawVariableType.Scalar, "W", "Power dissipated"),
                new("I", LawVariableType.Scalar, "A", "Current"),
                new("R", LawVariableType.Scalar, "ohm", "Resistance"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Electrical heating", "Fuses", "Resistors"],
            Reference = "Joule, J.P. (1841). Annals of Electricity.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["joule", "heating", "power", "resistance"]
        });

        b.Add("em.solenoid-inductance", new LawDefinition
        {
            Id = "em.solenoid-inductance",
            Name = "Solenoid Inductance",
            Category = LawCategory.Electromagnetism,
            Expression = "L = mu_0 * N^2 * A / ell",
            Description = "Self-inductance of a long solenoid.",
            Parameters = [new("mu_0", 1.25663706212e-6, "H/m", 0, 1, "Vacuum permeability")],
            Variables = [
                new("L", LawVariableType.Scalar, "H", "Inductance"),
                new("N", LawVariableType.Scalar, "dimensionless", "Total turns"),
                new("A", LawVariableType.Scalar, "m^2", "Cross-sectional area"),
                new("ell", LawVariableType.Scalar, "m", "Solenoid length"),
                new("mu_0", LawVariableType.Scalar, "H/m", "Permeability"),
            ],
            BoundaryConditionDefs = [
                new("LongSolenoid", "ell >> sqrt(A)", "End effects negligible"),
                new("UniformWinding", "N/ell = const", "Uniform turn density"),
            ],
            ApplicableDomains = ["Inductor design", "Transformers", "EM coupling"],
            Reference = "Standard result from Ampere and Faraday laws.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["inductance", "solenoid", "magnetic field"]
        });
    }

    // ==============================================================
    //  FLUID DYNAMICS (11 laws)
    // ==============================================================
    private static void RegisterFluidDynamics(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("fluid.navier-stokes", new LawDefinition
        {
            Id = "fluid.navier-stokes",
            Name = "Navier-Stokes Equations (Incompressible)",
            Category = LawCategory.FluidDynamics,
            Expression = "rho*(dv/dt + v.del(v)) = -grad(p) + mu*del^2(v) + f",
            Description = "Governing equations for viscous, incompressible fluid flow.",
            Parameters = [
                new("rho", 998.0, "kg/m^3", 0, 1e5, "Fluid density"),
                new("mu", 1.002e-3, "Pa*s", 0, 1e6, "Dynamic viscosity"),
            ],
            Variables = [
                new("v", LawVariableType.Vector, "m/s", "Velocity field"),
                new("p", LawVariableType.Scalar, "Pa", "Pressure field"),
                new("rho", LawVariableType.Scalar, "kg/m^3", "Density"),
                new("mu", LawVariableType.Scalar, "Pa*s", "Dynamic viscosity"),
                new("f", LawVariableType.Vector, "N/m^3", "Body force per unit volume"),
                new("t", LawVariableType.Scalar, "s", "Time"),
            ],
            BoundaryConditionDefs = [
                new("NoSlip", "v = v_wall", "Fluid equals wall velocity"),
                new("Incompressible", "del.v = 0", "Constant density"),
            ],
            ApplicableDomains = ["Hydraulics", "Aerodynamics", "Weather prediction", "Blood flow"],
            Reference = "Navier, C.L.M.H. (1822) / Stokes, G.G. (1845).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["navier-stokes", "viscous", "incompressible", "PDE"]
        });

        b.Add("fluid.bernoulli", new LawDefinition
        {
            Id = "fluid.bernoulli",
            Name = "Bernoulli's Equation",
            Category = LawCategory.FluidDynamics,
            Expression = "P + 0.5*rho*v^2 + rho*g*h = const",
            Description = "Energy conservation along a streamline for inviscid flow.",
            Parameters = [
                new("rho", 998.0, "kg/m^3", 0, 1e5, "Fluid density"),
                new("g", 9.80665, "m/s^2", 0, 100, "Gravitational acceleration"),
            ],
            Variables = [
                new("P", LawVariableType.Scalar, "Pa", "Static pressure"),
                new("v", LawVariableType.Scalar, "m/s", "Flow speed"),
                new("h", LawVariableType.Scalar, "m", "Elevation"),
                new("rho", LawVariableType.Scalar, "kg/m^3", "Density"),
                new("g", LawVariableType.Scalar, "m/s^2", "Gravity"),
            ],
            BoundaryConditionDefs = [
                new("InviscidFlow", "mu ~ 0", "No viscous losses"),
                new("Incompressible", "rho = const", "Ma < 0.3"),
                new("AlongStreamline", "constant along streamline", "May differ between streamlines"),
            ],
            ApplicableDomains = ["Aerodynamics", "Hydraulics", "Venturi meters", "Pitot tubes"],
            Reference = "Bernoulli, D. (1738). Hydrodynamica.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["bernoulli", "energy conservation", "streamline", "inviscid"]
        });

        b.Add("fluid.poiseuille", new LawDefinition
        {
            Id = "fluid.poiseuille",
            Name = "Hagen-Poiseuille Flow",
            Category = LawCategory.FluidDynamics,
            Expression = "Q = (pi * Delta_P * r^4) / (8 * mu * L)",
            Description = "Volumetric flow rate through a cylindrical pipe for laminar flow.",
            Parameters = [new("mu", 1.002e-3, "Pa*s", 0, 1e6, "Dynamic viscosity")],
            Variables = [
                new("Q", LawVariableType.Scalar, "m^3/s", "Flow rate"),
                new("Delta_P", LawVariableType.Scalar, "Pa", "Pressure drop"),
                new("r", LawVariableType.Scalar, "m", "Pipe radius"),
                new("mu", LawVariableType.Scalar, "Pa*s", "Viscosity"),
                new("L", LawVariableType.Scalar, "m", "Pipe length"),
            ],
            BoundaryConditionDefs = [
                new("LaminarFlow", "Re < ~2300", "Low Reynolds number"),
                new("FullyDeveloped", "entrance length << L", "Parabolic velocity"),
            ],
            ApplicableDomains = ["Microfluidics", "Blood flow", "Hydraulic systems"],
            Reference = "Hagen, G. (1839) / Poiseuille, J.L.M. (1846).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["poiseuille", "pipe flow", "laminar", "viscous"]
        });

        b.Add("fluid.stokes-drag", new LawDefinition
        {
            Id = "fluid.stokes-drag",
            Name = "Stokes Drag Law",
            Category = LawCategory.FluidDynamics,
            Expression = "F_d = 6*pi * mu * r * v",
            Description = "Drag force on a small sphere in creeping flow.",
            Parameters = [new("mu", 1.002e-3, "Pa*s", 0, 1e6, "Dynamic viscosity")],
            Variables = [
                new("F_d", LawVariableType.Vector, "N", "Drag force"),
                new("mu", LawVariableType.Scalar, "Pa*s", "Viscosity"),
                new("r", LawVariableType.Scalar, "m", "Sphere radius"),
                new("v", LawVariableType.Vector, "m/s", "Velocity"),
            ],
            BoundaryConditionDefs = [
                new("LowReynolds", "Re << 1", "Creeping flow"),
                new("RigidSphere", "no-slip surface", "Solid sphere"),
            ],
            ApplicableDomains = ["Particle sedimentation", "Aerosol science", "Microbiology"],
            Reference = "Stokes, G.G. (1851). Trans. Cambridge Phil. Society.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["stokes", "drag", "viscous", "sphere"]
        });

        b.Add("fluid.reynolds", new LawDefinition
        {
            Id = "fluid.reynolds",
            Name = "Reynolds Number",
            Category = LawCategory.FluidDynamics,
            Expression = "Re = rho * v * L / mu",
            Description = "Dimensionless ratio of inertial to viscous forces.",
            Parameters = [],
            Variables = [
                new("Re", LawVariableType.Scalar, "dimensionless", "Reynolds number"),
                new("rho", LawVariableType.Scalar, "kg/m^3", "Density"),
                new("v", LawVariableType.Scalar, "m/s", "Velocity"),
                new("L", LawVariableType.Scalar, "m", "Characteristic length"),
                new("mu", LawVariableType.Scalar, "Pa*s", "Viscosity"),
            ],
            BoundaryConditionDefs = [
                new("CharacteristicScale", "L is well-defined", "Relevant length scale"),
            ],
            ApplicableDomains = ["All fluid mechanics", "Scale modeling", "Pipe flow"],
            Reference = "Reynolds, O. (1883). Phil. Trans. Royal Society.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["reynolds", "dimensionless", "flow regime"]
        });

        b.Add("fluid.continuity", new LawDefinition
        {
            Id = "fluid.continuity",
            Name = "Continuity Equation (Incompressible)",
            Category = LawCategory.FluidDynamics,
            Expression = "del . v = 0",
            Description = "Mass conservation: velocity field is divergence-free.",
            Parameters = [],
            Variables = [
                new("v", LawVariableType.Vector, "m/s", "Velocity field"),
            ],
            BoundaryConditionDefs = [
                new("Incompressible", "rho = const", "No density variation"),
            ],
            ApplicableDomains = ["All incompressible flow", "Hydraulics", "Aerodynamics"],
            Reference = "Euler, L. (1757).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["continuity", "mass conservation", "incompressible"]
        });

        b.Add("fluid.drag-coefficient", new LawDefinition
        {
            Id = "fluid.drag-coefficient",
            Name = "Drag Force (General)",
            Category = LawCategory.FluidDynamics,
            Expression = "F_d = 0.5 * C_d * rho * A * v^2",
            Description = "Drag force using the drag coefficient.",
            Parameters = [
                new("C_d", 0.47, "dimensionless", 0, 10, "Drag coefficient"),
                new("rho", 1.225, "kg/m^3", 0, 1e5, "Fluid density"),
            ],
            Variables = [
                new("F_d", LawVariableType.Vector, "N", "Drag force"),
                new("C_d", LawVariableType.Scalar, "dimensionless", "Drag coefficient"),
                new("rho", LawVariableType.Scalar, "kg/m^3", "Density"),
                new("A", LawVariableType.Scalar, "m^2", "Reference area"),
                new("v", LawVariableType.Scalar, "m/s", "Relative velocity"),
            ],
            BoundaryConditionDefs = [
                new("TurbulentOrTransitional", "Re > ~1000", "Cd roughly constant"),
            ],
            ApplicableDomains = ["Vehicle design", "Sports engineering", "Wind loading"],
            Reference = "Standard empirical result.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["drag", "aerodynamics", "drag coefficient"]
        });

        b.Add("fluid.archimedes", new LawDefinition
        {
            Id = "fluid.archimedes",
            Name = "Archimedes' Principle",
            Category = LawCategory.FluidDynamics,
            Expression = "F_b = rho_fluid * V_disp * g",
            Description = "Buoyant force equals weight of displaced fluid.",
            Parameters = [new("g", 9.80665, "m/s^2", 0, 100, "Gravitational acceleration")],
            Variables = [
                new("F_b", LawVariableType.Vector, "N", "Buoyant force"),
                new("rho_fluid", LawVariableType.Scalar, "kg/m^3", "Fluid density"),
                new("V_disp", LawVariableType.Scalar, "m^3", "Displaced volume"),
                new("g", LawVariableType.Scalar, "m/s^2", "Gravity"),
            ],
            BoundaryConditionDefs = [
                new("StaticFluid", "fluid at rest", "Static buoyancy"),
            ],
            ApplicableDomains = ["Naval architecture", "Balloonry", "Sedimentation"],
            Reference = "Archimedes (c. 250 BC). On Floating Bodies.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["buoyancy", "archimedes", "floating"]
        });

        b.Add("fluid.mach", new LawDefinition
        {
            Id = "fluid.mach",
            Name = "Mach Number",
            Category = LawCategory.FluidDynamics,
            Expression = "Ma = v / c",
            Description = "Ratio of flow velocity to speed of sound.",
            Parameters = [],
            Variables = [
                new("Ma", LawVariableType.Scalar, "dimensionless", "Mach number"),
                new("v", LawVariableType.Scalar, "m/s", "Flow velocity"),
                new("c", LawVariableType.Scalar, "m/s", "Speed of sound"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Compressible flow", "Supersonics", "Aerodynamics"],
            Reference = "Mach, E. (1887). Wiener Berichte.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["mach", "compressible", "speed of sound"]
        });

        b.Add("fluid.froude", new LawDefinition
        {
            Id = "fluid.froude",
            Name = "Froude Number",
            Category = LawCategory.FluidDynamics,
            Expression = "Fr = v / sqrt(g * L)",
            Description = "Ratio of inertial to gravitational forces.",
            Parameters = [new("g", 9.80665, "m/s^2", 0, 100, "Gravitational acceleration")],
            Variables = [
                new("Fr", LawVariableType.Scalar, "dimensionless", "Froude number"),
                new("v", LawVariableType.Scalar, "m/s", "Flow velocity"),
                new("g", LawVariableType.Scalar, "m/s^2", "Gravity"),
                new("L", LawVariableType.Scalar, "m", "Characteristic length"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Ship hydrodraulics", "Open channel flow", "Coastal engineering"],
            Reference = "Froude, W. (1873). Brit. Assoc. Report.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["froude", "gravity waves", "ship design"]
        });

        b.Add("fluid.stokes-sedimentation", new LawDefinition
        {
            Id = "fluid.stokes-sedimentation",
            Name = "Stokes Sedimentation Velocity",
            Category = LawCategory.FluidDynamics,
            Expression = "v_t = (2/9) * (rho_p - rho_f) * g * r^2 / mu",
            Description = "Terminal settling velocity of a small spherical particle.",
            Parameters = [
                new("g", 9.80665, "m/s^2", 0, 100, "Gravitational acceleration"),
                new("mu", 1.002e-3, "Pa*s", 0, 1e6, "Dynamic viscosity"),
            ],
            Variables = [
                new("v_t", LawVariableType.Scalar, "m/s", "Terminal velocity"),
                new("rho_p", LawVariableType.Scalar, "kg/m^3", "Particle density"),
                new("rho_f", LawVariableType.Scalar, "kg/m^3", "Fluid density"),
                new("g", LawVariableType.Scalar, "m/s^2", "Gravity"),
                new("r", LawVariableType.Scalar, "m", "Particle radius"),
                new("mu", LawVariableType.Scalar, "Pa*s", "Viscosity"),
            ],
            BoundaryConditionDefs = [
                new("LowRe", "Re_p << 1", "Stokes drag regime"),
                new("SphericalParticle", "perfectly spherical", "Shape factor ~1"),
            ],
            ApplicableDomains = ["Sedimentation", "Centrifugation", "Atmospheric particles"],
            Reference = "Stokes, G.G. (1851).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["sedimentation", "settling", "terminal velocity"]
        });

        b.Add("fluid.venturi", new LawDefinition
        {
            Id = "fluid.venturi",
            Name = "Venturi Effect",
            Category = LawCategory.FluidDynamics,
            Expression = "v2 = v1*(A1/A2); P2 = P1 + 0.5*rho*(v1^2 - v2^2)",
            Description = "Velocity increases and pressure decreases at a constriction.",
            Parameters = [],
            Variables = [
                new("v1", LawVariableType.Scalar, "m/s", "Velocity at section 1"),
                new("v2", LawVariableType.Scalar, "m/s", "Velocity at constriction"),
                new("A1", LawVariableType.Scalar, "m^2", "Area at section 1"),
                new("A2", LawVariableType.Scalar, "m^2", "Area at constriction"),
                new("P1", LawVariableType.Scalar, "Pa", "Pressure at section 1"),
                new("P2", LawVariableType.Scalar, "Pa", "Pressure at constriction"),
                new("rho", LawVariableType.Scalar, "kg/m^3", "Fluid density"),
            ],
            BoundaryConditionDefs = [
                new("SteadyFlow", "dt = 0", "Steady-state"),
                new("Inviscid", "no friction", "Frictionless flow"),
            ],
            ApplicableDomains = ["Flow measurement", "Carburetors", "Venturi meters"],
            Reference = "Venturi, G.B. (1797).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["venturi", "constriction", "pressure", "flow measurement"]
        });
    }

    // ==============================================================
    //  QUANTUM MECHANICS (11 laws)
    // ==============================================================
    private static void RegisterQuantumMechanics(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("qm.schrodinger-td", new LawDefinition
        {
            Id = "qm.schrodinger-td",
            Name = "Time-Dependent Schrodinger Equation",
            Category = LawCategory.QuantumMechanics,
            Expression = "i*hbar*d psi/dt = H_hat*psi",
            Description = "Fundamental equation for time evolution of quantum states.",
            Parameters = [new("hbar", 1.054571817e-34, "J*s", 0, 1e-30, "Reduced Planck constant")],
            Variables = [
                new("psi", LawVariableType.Function, "m^(-3/2)", "Wavefunction"),
                new("t", LawVariableType.Scalar, "s", "Time"),
                new("H_hat", LawVariableType.Operator, "J", "Hamiltonian"),
                new("hbar", LawVariableType.Scalar, "J*s", "Reduced Planck constant"),
            ],
            BoundaryConditionDefs = [
                new("Normalizable", "integral |psi|^2 = 1", "Normalizable"),
                new("SingleValued", "single-valued", "Physical wavefunction"),
            ],
            ApplicableDomains = ["Quantum mechanics", "Quantum chemistry", "Quantum computing"],
            Reference = "Schrodinger, E. (1926). Physikalische Zeitschrift.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["schrodinger", "wavefunction", "quantum", "fundamental"]
        });

        b.Add("qm.schrodinger-ti", new LawDefinition
        {
            Id = "qm.schrodinger-ti",
            Name = "Time-Independent Schrodinger Equation",
            Category = LawCategory.QuantumMechanics,
            Expression = "H_hat*psi = E*psi",
            Description = "Eigenvalue equation for stationary states.",
            Parameters = [],
            Variables = [
                new("psi", LawVariableType.Function, "m^(-3/2)", "Stationary state"),
                new("H_hat", LawVariableType.Operator, "J", "Hamiltonian"),
                new("E", LawVariableType.Scalar, "J", "Energy eigenvalue"),
            ],
            BoundaryConditionDefs = [
                new("Normalizable", "integral |psi|^2 = 1", "Normalizable eigenfunction"),
            ],
            ApplicableDomains = ["Atomic physics", "Quantum chemistry", "Solid-state physics"],
            Reference = "Schrodinger, E. (1926). Annalen der Physik.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["schrodinger", "eigenvalue", "stationary state", "energy levels"]
        });

        b.Add("qm.uncertainty", new LawDefinition
        {
            Id = "qm.uncertainty",
            Name = "Heisenberg Uncertainty Principle",
            Category = LawCategory.QuantumMechanics,
            Expression = "Delta_x * Delta_p >= hbar/2",
            Description = "Fundamental limit on simultaneous precision of conjugate observables.",
            Parameters = [new("hbar", 1.054571817e-34, "J*s", 0, 1e-30, "Reduced Planck constant")],
            Variables = [
                new("Delta_x", LawVariableType.Scalar, "m", "Position std dev"),
                new("Delta_p", LawVariableType.Scalar, "kg*m/s", "Momentum std dev"),
                new("hbar", LawVariableType.Scalar, "J*s", "Reduced Planck constant"),
            ],
            BoundaryConditionDefs = [
                new("QuantumState", "any quantum state", "Universal"),
            ],
            ApplicableDomains = ["Quantum mechanics", "Quantum optics", "Particle physics"],
            Reference = "Heisenberg, W. (1927). Zeitschrift fuer Physik.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["uncertainty", "heisenberg", "complementary"]
        });

        b.Add("qm.born-rule", new LawDefinition
        {
            Id = "qm.born-rule",
            Name = "Born Rule",
            Category = LawCategory.QuantumMechanics,
            Expression = "P(x) = |psi(x)|^2",
            Description = "Probability density is squared modulus of wavefunction.",
            Parameters = [],
            Variables = [
                new("P", LawVariableType.Scalar, "m^-3", "Probability density"),
                new("psi", LawVariableType.Function, "m^(-3/2)", "Wavefunction"),
            ],
            BoundaryConditionDefs = [
                new("Normalized", "integral |psi|^2 = 1", "Total probability unity"),
            ],
            ApplicableDomains = ["Quantum mechanics", "Quantum measurement"],
            Reference = "Born, M. (1926). Zeitschrift fuer Physik.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["born", "probability", "measurement"]
        });

        b.Add("qm.harmonic-oscillator", new LawDefinition
        {
            Id = "qm.harmonic-oscillator",
            Name = "Quantum Harmonic Oscillator Energy Levels",
            Category = LawCategory.QuantumMechanics,
            Expression = "E_n = hbar*omega*(n + 1/2)",
            Description = "Discrete energy levels with zero-point energy.",
            Parameters = [new("hbar", 1.054571817e-34, "J*s", 0, 1e-30, "Reduced Planck constant")],
            Variables = [
                new("E_n", LawVariableType.Scalar, "J", "Energy of n-th level"),
                new("omega", LawVariableType.Scalar, "rad/s", "Angular frequency"),
                new("n", LawVariableType.Scalar, "dimensionless", "Quantum number (0,1,2,...)"),
                new("hbar", LawVariableType.Scalar, "J*s", "Reduced Planck constant"),
            ],
            BoundaryConditionDefs = [
                new("HarmonicPotential", "V(x) = 0.5*k*x^2", "Parabolic potential"),
            ],
            ApplicableDomains = ["Quantum optics", "Molecular vibrations", "Phonons"],
            Reference = "Dirac, P.A.M. (1930). The Principles of Quantum Mechanics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["harmonic oscillator", "energy levels", "zero-point energy"]
        });

        b.Add("qm.de-broglie", new LawDefinition
        {
            Id = "qm.de-broglie",
            Name = "de Broglie Wavelength",
            Category = LawCategory.QuantumMechanics,
            Expression = "lambda = h / p",
            Description = "Wavelength associated with a particle's momentum.",
            Parameters = [new("h", 6.62607015e-34, "J*s", 0, 1e-30, "Planck constant")],
            Variables = [
                new("lambda", LawVariableType.Scalar, "m", "de Broglie wavelength"),
                new("h", LawVariableType.Scalar, "J*s", "Planck constant"),
                new("p", LawVariableType.Scalar, "kg*m/s", "Momentum"),
            ],
            BoundaryConditionDefs = [
                new("FreeParticle", "well-defined momentum", "Momentum eigenstate"),
            ],
            ApplicableDomains = ["Electron diffraction", "Neutron scattering", "Matter-wave interferometry"],
            Reference = "de Broglie, L. (1924). Recherches sur la Theorie des Quanta.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["de broglie", "wavelength", "wave-particle duality"]
        });

        b.Add("qm.commutator", new LawDefinition
        {
            Id = "qm.commutator",
            Name = "Canonical Commutation Relation",
            Category = LawCategory.QuantumMechanics,
            Expression = "[x_hat, p_hat] = i*hbar",
            Description = "Fundamental commutation relation between position and momentum.",
            Parameters = [new("hbar", 1.054571817e-34, "J*s", 0, 1e-30, "Reduced Planck constant")],
            Variables = [
                new("x_hat", LawVariableType.Operator, "m", "Position operator"),
                new("p_hat", LawVariableType.Operator, "kg*m/s", "Momentum operator"),
                new("hbar", LawVariableType.Scalar, "J*s", "Reduced Planck constant"),
            ],
            BoundaryConditionDefs = [
                new("CanonicalQuantization", "standard rules", "Non-relativistic QM"),
            ],
            ApplicableDomains = ["Quantum mechanics foundations", "Quantum field theory"],
            Reference = "Dirac, P.A.M. (1930); Heisenberg, W. (1925).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["commutator", "canonical", "uncertainty", "operators"]
        });

        b.Add("qm.pauli-exclusion", new LawDefinition
        {
            Id = "qm.pauli-exclusion",
            Name = "Pauli Exclusion Principle",
            Category = LawCategory.QuantumMechanics,
            Expression = "No two identical fermions occupy the same quantum state.",
            Description = "Antisymmetry for many-fermion wavefunctions; explains shell structure.",
            Parameters = [],
            Variables = [
                new("psi", LawVariableType.Function, "m^(-3N/2)", "N-particle wavefunction"),
            ],
            BoundaryConditionDefs = [
                new("IdenticalFermions", "spin-1/2", "Half-integer spin"),
                new("Antisymmetric", "psi(1,2) = -psi(2,1)", "Exchange antisymmetric"),
            ],
            ApplicableDomains = ["Atomic physics", "Solid-state physics", "Nuclear physics"],
            Reference = "Pauli, W. (1925). Zeitschrift fuer Physik.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["pauli", "exclusion", "fermions", "antisymmetric"]
        });

        b.Add("qm.spin-statistics", new LawDefinition
        {
            Id = "qm.spin-statistics",
            Name = "Spin-Statistics Theorem",
            Category = LawCategory.QuantumMechanics,
            Expression = "Integer spin -> bosons (symmetric); Half-integer spin -> fermions (antisymmetric)",
            Description = "Relates particle spin to wavefunction symmetry.",
            Parameters = [],
            Variables = [
                new("s", LawVariableType.Scalar, "hbar", "Spin quantum number"),
            ],
            BoundaryConditionDefs = [
                new("RelativisticQFT", "Lorentz invariance required", "Requires QFT proof"),
            ],
            ApplicableDomains = ["Quantum field theory", "Statistical mechanics", "Particle physics"],
            Reference = "Pauli, W. (1940). Physical Review.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["spin", "statistics", "bosons", "fermions"]
        });

        b.Add("qm.tunneling", new LawDefinition
        {
            Id = "qm.tunneling",
            Name = "Quantum Tunneling (WKB Approximation)",
            Category = LawCategory.QuantumMechanics,
            Expression = "T ~ exp(-2*integral|kappa(x)|dx)",
            Description = "Probability of penetrating a potential barrier.",
            Parameters = [],
            Variables = [
                new("T", LawVariableType.Scalar, "dimensionless", "Transmission probability"),
                new("kappa", LawVariableType.Function, "m^-1", "Decay constant"),
                new("V", LawVariableType.Function, "J", "Potential energy"),
                new("E", LawVariableType.Scalar, "J", "Particle energy"),
            ],
            BoundaryConditionDefs = [
                new("ClassicallyForbidden", "E < V_max", "Below barrier height"),
            ],
            ApplicableDomains = ["Tunnel diodes", "Alpha decay", "Scanning tunneling microscopy"],
            Reference = "Gamow, G. (1928) / Gurney, R.W. & Condon, E.U. (1928).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["tunneling", "barrier", "WKB"]
        });

        b.Add("qm.expectation", new LawDefinition
        {
            Id = "qm.expectation",
            Name = "Quantum Expectation Value",
            Category = LawCategory.QuantumMechanics,
            Expression = "<A> = integral psi* A_hat psi d^3x",
            Description = "Statistical average of an observable measurement.",
            Parameters = [],
            Variables = [
                new("A_expect", LawVariableType.Scalar, "varies", "Expectation value"),
                new("psi", LawVariableType.Function, "m^(-3/2)", "Normalized wavefunction"),
                new("A_hat", LawVariableType.Operator, "varies", "Observable"),
            ],
            BoundaryConditionDefs = [
                new("NormalizedState", "integral |psi|^2 = 1", "Normalized"),
            ],
            ApplicableDomains = ["Quantum mechanics", "Measurement theory"],
            Reference = "Standard result in quantum mechanics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["expectation", "observable", "measurement"]
        });

        b.Add("qm.hydrogen", new LawDefinition
        {
            Id = "qm.hydrogen",
            Name = "Hydrogen Atom Energy Levels",
            Category = LawCategory.QuantumMechanics,
            Expression = "E_n = -13.6 eV / n^2",
            Description = "Exact energy levels of the hydrogen atom.",
            Parameters = [
                new("R_inf", 13.605693, "eV", 0, 100, "Rydberg energy"),
                new("m_e", 9.1093837015e-31, "kg", 0, 1e-25, "Electron mass"),
                new("e", 1.602176634e-19, "C", 0, 1e-10, "Elementary charge"),
                new("epsilon_0", 8.8541878128e-12, "F/m", 0, 1e-6, "Vacuum permittivity"),
            ],
            Variables = [
                new("E_n", LawVariableType.Scalar, "eV", "Energy of n-th level"),
                new("n", LawVariableType.Scalar, "dimensionless", "Principal quantum number (1,2,3,...)"),
            ],
            BoundaryConditionDefs = [
                new("CoulombPotential", "V(r) = -e^2/(4*pi*eps_0*r)", "Pure Coulomb"),
                new("NonRelativistic", "v << c", "No relativistic corrections"),
            ],
            ApplicableDomains = ["Atomic physics", "Spectroscopy", "Quantum chemistry"],
            Reference = "Schrodinger, E. (1926) / Bohr, N. (1913).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["hydrogen", "energy levels", "rydberg", "spectroscopy"]
        });
    }


    // ==============================================================
    //  OPTICS (9 laws)
    // ==============================================================
    private static void RegisterOptics(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("optics.snell", new LawDefinition
        {
            Id = "optics.snell",
            Name = "Snell's Law (Law of Refraction)",
            Category = LawCategory.Optics,
            Expression = "n1*sin(theta1) = n2*sin(theta2)",
            Description = "Relates angles of incidence and refraction at two media interface.",
            Parameters = [],
            Variables = [
                new("n1", LawVariableType.Scalar, "dimensionless", "Refractive index 1"),
                new("theta1", LawVariableType.Scalar, "rad", "Angle of incidence"),
                new("n2", LawVariableType.Scalar, "dimensionless", "Refractive index 2"),
                new("theta2", LawVariableType.Scalar, "rad", "Angle of refraction"),
            ],
            BoundaryConditionDefs = [
                new("PlanarInterface", "flat boundary", "Homogeneous media"),
            ],
            ApplicableDomains = ["Lens design", "Fiber optics", "Atmospheric optics"],
            Reference = "Snell, W. (1621) / Descartes, R. (1637).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["snell", "refraction", "refractive index"]
        });

        b.Add("optics.thin-lens", new LawDefinition
        {
            Id = "optics.thin-lens",
            Name = "Thin Lens Equation",
            Category = LawCategory.Optics,
            Expression = "1/f = 1/d_o + 1/d_i",
            Description = "Relates object distance, image distance, and focal length.",
            Parameters = [],
            Variables = [
                new("f", LawVariableType.Scalar, "m", "Focal length"),
                new("d_o", LawVariableType.Scalar, "m", "Object distance"),
                new("d_i", LawVariableType.Scalar, "m", "Image distance"),
            ],
            BoundaryConditionDefs = [
                new("ThinLens", "thickness << f", "Paraxial"),
                new("SmallAngles", "sin(theta) ~ theta", "Paraxial rays"),
            ],
            ApplicableDomains = ["Optical instruments", "Cameras", "Eyeglasses"],
            Reference = "Gauss, C.F. (1841).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["lens", "focal length", "image"]
        });

        b.Add("optics.malus", new LawDefinition
        {
            Id = "optics.malus",
            Name = "Malus's Law",
            Category = LawCategory.Optics,
            Expression = "I = I0 * cos^2(theta)",
            Description = "Intensity of polarized light after a polarizer.",
            Parameters = [],
            Variables = [
                new("I", LawVariableType.Scalar, "W/m^2", "Transmitted intensity"),
                new("I0", LawVariableType.Scalar, "W/m^2", "Incident polarized intensity"),
                new("theta", LawVariableType.Scalar, "rad", "Angle to polarizer axis"),
            ],
            BoundaryConditionDefs = [
                new("PolarizedLight", "linearly polarized", "Unpolarized gives I = I0/2"),
            ],
            ApplicableDomains = ["Polarimetry", "LCD displays", "Optical filters"],
            Reference = "Malus, E.L. (1809). Journal de Physique.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["malus", "polarization", "intensity"]
        });

        b.Add("optics.rayleigh", new LawDefinition
        {
            Id = "optics.rayleigh",
            Name = "Rayleigh Scattering",
            Category = LawCategory.Optics,
            Expression = "I ~ 1/lambda^4",
            Description = "Scattering by particles much smaller than wavelength; explains blue sky.",
            Parameters = [],
            Variables = [
                new("I", LawVariableType.Scalar, "W/m^2", "Scattered intensity"),
                new("lambda", LawVariableType.Scalar, "m", "Wavelength"),
            ],
            BoundaryConditionDefs = [
                new("SmallParticles", "d << lambda", "Much smaller than wavelength"),
            ],
            ApplicableDomains = ["Atmospheric optics", "Light scattering", "Nanoparticle characterization"],
            Reference = "Rayleigh, Lord (1871). Phil. Mag.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["rayleigh", "scattering", "blue sky"]
        });

        b.Add("optics.brewster", new LawDefinition
        {
            Id = "optics.brewster",
            Name = "Brewster's Angle",
            Category = LawCategory.Optics,
            Expression = "theta_B = arctan(n2/n1)",
            Description = "Angle of incidence for completely polarized reflected light.",
            Parameters = [],
            Variables = [
                new("theta_B", LawVariableType.Scalar, "rad", "Brewster's angle"),
                new("n1", LawVariableType.Scalar, "dimensionless", "Refractive index 1"),
                new("n2", LawVariableType.Scalar, "dimensionless", "Refractive index 2"),
            ],
            BoundaryConditionDefs = [
                new("DielectricInterface", "no absorption", "Transparent dielectrics"),
            ],
            ApplicableDomains = ["Polarization optics", "Laser cavities", "AR coatings"],
            Reference = "Brewster, D. (1815). Edinburgh Phil. Journal.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["brewster", "polarization", "reflection"]
        });

        b.Add("optics.single-slit", new LawDefinition
        {
            Id = "optics.single-slit",
            Name = "Single-Slit Diffraction Minima",
            Category = LawCategory.Optics,
            Expression = "a*sin(theta) = m*lambda, m = +/-1, +/-2, ...",
            Description = "Angular positions of minima in single-slit Fraunhofer diffraction.",
            Parameters = [],
            Variables = [
                new("a", LawVariableType.Scalar, "m", "Slit width"),
                new("theta", LawVariableType.Scalar, "rad", "Diffraction angle"),
                new("m", LawVariableType.Scalar, "dimensionless", "Integer order (nonzero)"),
                new("lambda", LawVariableType.Scalar, "m", "Wavelength"),
            ],
            BoundaryConditionDefs = [
                new("FarField", "D >> a^2/lambda", "Fraunhofer approximation"),
            ],
            ApplicableDomains = ["Spectroscopy", "X-ray diffraction", "Laser profiling"],
            Reference = "Fraunhofer, J. (1818).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["diffraction", "slit", "interference"]
        });

        b.Add("optics.grating", new LawDefinition
        {
            Id = "optics.grating",
            Name = "Diffraction Grating Equation",
            Category = LawCategory.Optics,
            Expression = "d*sin(theta) = m*lambda",
            Description = "Constructive interference condition for a diffraction grating.",
            Parameters = [],
            Variables = [
                new("d", LawVariableType.Scalar, "m", "Grating spacing"),
                new("theta", LawVariableType.Scalar, "rad", "Diffraction angle"),
                new("m", LawVariableType.Scalar, "dimensionless", "Diffraction order"),
                new("lambda", LawVariableType.Scalar, "m", "Wavelength"),
            ],
            BoundaryConditionDefs = [
                new("PlaneWave", "collimated light", "Parallel rays"),
            ],
            ApplicableDomains = ["Spectrometers", "Monochromators", "Laser systems"],
            Reference = "Fraunhofer, J. (1821).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["grating", "spectroscopy", "resolution"]
        });

        b.Add("optics.gaussian-beam", new LawDefinition
        {
            Id = "optics.gaussian-beam",
            Name = "Gaussian Beam (Complex q-parameter)",
            Category = LawCategory.Optics,
            Expression = "1/q(z) = 1/R(z) - i*lambda/(pi*w^2(z))",
            Description = "Describes TEM00 laser beam propagation.",
            Parameters = [new("lambda", 532e-9, "m", 1e-9, 1e-3, "Wavelength")],
            Variables = [
                new("q", LawVariableType.Complex, "m", "Complex beam parameter"),
                new("R", LawVariableType.Scalar, "m", "Wavefront curvature"),
                new("w", LawVariableType.Scalar, "m", "Beam radius (1/e^2)"),
                new("z", LawVariableType.Scalar, "m", "Propagation distance"),
                new("lambda", LawVariableType.Scalar, "m", "Wavelength"),
            ],
            BoundaryConditionDefs = [
                new("Paraxial", "theta << 1", "Paraxial approximation"),
                new("TEM00", "lowest-order mode", "Gaussian profile"),
            ],
            ApplicableDomains = ["Laser optics", "Fiber coupling", "Optical trapping"],
            Reference = "Kogelnik, H. & Li, T. (1966). Applied Optics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["gaussian", "laser", "beam", "propagation"]
        });

        b.Add("optics.fermat", new LawDefinition
        {
            Id = "optics.fermat",
            Name = "Fermat's Principle of Least Time",
            Category = LawCategory.Optics,
            Expression = "delta integral(n ds) = 0",
            Description = "Light travels along the path minimizing optical path length.",
            Parameters = [],
            Variables = [
                new("n", LawVariableType.Scalar, "dimensionless", "Refractive index"),
                new("s", LawVariableType.Scalar, "m", "Path element"),
            ],
            BoundaryConditionDefs = [
                new("GeometricOptics", "lambda << feature size", "Wave effects negligible"),
            ],
            ApplicableDomains = ["Lens design", "Fiber optics", "Atmospheric refraction"],
            Reference = "Fermat, P. de (1662).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["fermat", "least time", "variational"]
        });

        b.Add("optics.lensmaker", new LawDefinition
        {
            Id = "optics.lensmaker",
            Name = "Lensmaker's Equation",
            Category = LawCategory.Optics,
            Expression = "1/f = (n - 1)*(1/R1 - 1/R2)",
            Description = "Focal length from lens geometry and refractive index.",
            Parameters = [],
            Variables = [
                new("f", LawVariableType.Scalar, "m", "Focal length"),
                new("n", LawVariableType.Scalar, "dimensionless", "Refractive index"),
                new("R1", LawVariableType.Scalar, "m", "First surface radius"),
                new("R2", LawVariableType.Scalar, "m", "Second surface radius"),
            ],
            BoundaryConditionDefs = [
                new("ThinLens", "thickness << radii", "Paraxial"),
            ],
            ApplicableDomains = ["Optical design", "Camera lenses", "Microscopes"],
            Reference = "Standard geometric optics result.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["lensmaker", "focal length", "curvature"]
        });

        b.Add("optics.beer-lambert", new LawDefinition
        {
            Id = "optics.beer-lambert",
            Name = "Beer-Lambert Law",
            Category = LawCategory.Optics,
            Expression = "I = I0 * exp(-alpha * d)",
            Description = "Exponential attenuation of light through an absorbing medium.",
            Parameters = [new("alpha", 1.0, "m^-1", 0, 1e6, "Absorption coefficient")],
            Variables = [
                new("I", LawVariableType.Scalar, "W/m^2", "Transmitted intensity"),
                new("I0", LawVariableType.Scalar, "W/m^2", "Incident intensity"),
                new("alpha", LawVariableType.Scalar, "m^-1", "Absorption coefficient"),
                new("d", LawVariableType.Scalar, "m", "Path length"),
            ],
            BoundaryConditionDefs = [
                new("Monochromatic", "single wavelength", "alpha is wavelength-dependent"),
                new("HomogeneousMedium", "alpha uniform", "Constant along path"),
            ],
            ApplicableDomains = ["Spectroscopy", "Atmospheric science", "Biomedical optics"],
            Reference = "Beer (1852); Lambert (1760).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["beer-lambert", "absorption", "attenuation"]
        });
    }

    // ==============================================================
    //  GRAVITATION (5 laws)
    // ==============================================================
    private static void RegisterGravitation(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("grav.einstein", new LawDefinition
        {
            Id = "grav.einstein",
            Name = "Einstein Field Equations",
            Category = LawCategory.Gravitation,
            Expression = "G_munu + Lambda*g_munu = (8*pi*G/c^4)*T_munu",
            Description = "Relates spacetime curvature to energy-momentum; foundation of GR.",
            Parameters = [
                new("G", 6.674e-11, "N*m^2/kg^2", 0, 1e-5, "Gravitational constant"),
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
                new("Lambda", 0.0, "m^-2", -1e-50, 1e-50, "Cosmological constant"),
            ],
            Variables = [
                new("G_munu", LawVariableType.Tensor, "m^-2", "Einstein tensor"),
                new("g_munu", LawVariableType.Tensor, "dimensionless", "Metric tensor"),
                new("Lambda", LawVariableType.Scalar, "m^-2", "Cosmological constant"),
                new("T_munu", LawVariableType.Tensor, "Pa", "Energy-momentum tensor"),
                new("G", LawVariableType.Scalar, "N*m^2/kg^2", "Gravitational constant"),
                new("c", LawVariableType.Scalar, "m/s", "Speed of light"),
            ],
            BoundaryConditionDefs = [
                new("LorentzianSignature", "(-,+,+,+)", "Spacetime signature"),
            ],
            ApplicableDomains = ["General relativity", "Cosmology", "Black holes", "Gravitational waves"],
            Reference = "Einstein, A. (1915). Sitzungsberichte der Preussischen Akademie.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["einstein", "field equations", "general relativity", "curvature"]
        });

        b.Add("grav.schwarzschild", new LawDefinition
        {
            Id = "grav.schwarzschild",
            Name = "Schwarzschild Metric",
            Category = LawCategory.Gravitation,
            Expression = "ds^2 = -(1-r_s/r)c^2 dt^2 + (1-r_s/r)^(-1) dr^2 + r^2 dOmega^2",
            Description = "Exact solution for spherically symmetric non-rotating mass.",
            Parameters = [
                new("G", 6.674e-11, "N*m^2/kg^2", 0, 1e-5, "Gravitational constant"),
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
            ],
            Variables = [
                new("ds", LawVariableType.Scalar, "m", "Spacetime interval"),
                new("r_s", LawVariableType.Scalar, "m", "Schwarzschild radius = 2GM/c^2"),
                new("r", LawVariableType.Scalar, "m", "Radial coordinate"),
                new("t", LawVariableType.Scalar, "s", "Time coordinate"),
            ],
            BoundaryConditionDefs = [
                new("Vacuum", "T_munu = 0", "Outside mass"),
                new("SphericalSymmetry", "static, non-rotating", "No angular momentum"),
            ],
            ApplicableDomains = ["Black holes", "Gravitational redshift", "Orbital precession"],
            Reference = "Schwarzschild, K. (1916).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["schwarzschild", "black hole", "metric"]
        });

        b.Add("grav.redshift", new LawDefinition
        {
            Id = "grav.redshift",
            Name = "Gravitational Redshift",
            Category = LawCategory.Gravitation,
            Expression = "z = Delta_phi/c^2 = GM/(r*c^2)",
            Description = "Frequency shift of light climbing out of a gravitational well.",
            Parameters = [
                new("G", 6.674e-11, "N*m^2/kg^2", 0, 1e-5, "Gravitational constant"),
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
            ],
            Variables = [
                new("z", LawVariableType.Scalar, "dimensionless", "Redshift"),
                new("Delta_phi", LawVariableType.Scalar, "J/kg", "Potential difference"),
                new("M", LawVariableType.Scalar, "kg", "Mass"),
                new("r", LawVariableType.Scalar, "m", "Distance"),
            ],
            BoundaryConditionDefs = [
                new("WeakField", "phi << c^2", "Weak field approximation"),
            ],
            ApplicableDomains = ["GPS corrections", "Astrophysics", "Tests of GR"],
            Reference = "Einstein, A. (1907). Annalen der Physik.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["redshift", "gravitational", "frequency shift"]
        });

        b.Add("grav.time-dilation", new LawDefinition
        {
            Id = "grav.time-dilation",
            Name = "Gravitational Time Dilation",
            Category = LawCategory.Gravitation,
            Expression = "d_tau = dt * sqrt(1 - r_s/r)",
            Description = "Proper time flows slower in stronger gravitational fields.",
            Parameters = [
                new("G", 6.674e-11, "N*m^2/kg^2", 0, 1e-5, "Gravitational constant"),
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
            ],
            Variables = [
                new("d_tau", LawVariableType.Scalar, "s", "Proper time"),
                new("dt", LawVariableType.Scalar, "s", "Coordinate time"),
                new("r_s", LawVariableType.Scalar, "m", "Schwarzschild radius"),
                new("r", LawVariableType.Scalar, "m", "Radial coordinate"),
            ],
            BoundaryConditionDefs = [
                new("WeakField", "r >> r_s", "Far from event horizon"),
            ],
            ApplicableDomains = ["GPS satellites", "Pulsar timing", "Black hole physics"],
            Reference = "Einstein, A. (1915). General Relativity.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["time dilation", "gravitational", "proper time"]
        });

        b.Add("grav.waves", new LawDefinition
        {
            Id = "grav.waves",
            Name = "Gravitational Wave Strain (Quadrupole)",
            Category = LawCategory.Gravitation,
            Expression = "h ~ (2*G/c^4) * (d^2Q/dt^2) / r",
            Description = "Amplitude of gravitational waves from mass quadrupole.",
            Parameters = [
                new("G", 6.674e-11, "N*m^2/kg^2", 0, 1e-5, "Gravitational constant"),
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
            ],
            Variables = [
                new("h", LawVariableType.Scalar, "dimensionless", "Strain amplitude"),
                new("Q", LawVariableType.Scalar, "kg*m^2", "Mass quadrupole moment"),
                new("r", LawVariableType.Scalar, "m", "Distance to source"),
            ],
            BoundaryConditionDefs = [
                new("WeakField", "weak gravity", "Linearized GR"),
            ],
            ApplicableDomains = ["Gravitational wave astronomy", "LIGO/Virgo", "Neutron star mergers"],
            Reference = "Einstein, A. (1918). Sitzungsberichte der Berliner Akademie.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["gravitational waves", "quadrupole", "strain"]
        });
    }

    // ==============================================================
    //  CHEMISTRY (7 laws)
    // ==============================================================
    private static void RegisterChemistry(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("chem.arrhenius", new LawDefinition
        {
            Id = "chem.arrhenius",
            Name = "Arrhenius Equation (Reaction Rate)",
            Category = LawCategory.Chemistry,
            Expression = "k = A * exp(-Ea / (R*T))",
            Description = "Temperature dependence of chemical reaction rate constants.",
            Parameters = [
                new("R", 8.31446, "J/(mol*K)", 0, 100, "Universal gas constant"),
                new("A", 1e13, "s^-1", 0, 1e30, "Pre-exponential factor"),
                new("Ea", 50000.0, "J/mol", 0, 1e6, "Activation energy"),
            ],
            Variables = [
                new("k", LawVariableType.Scalar, "s^-1", "Rate constant"),
                new("A", LawVariableType.Scalar, "s^-1", "Pre-exponential factor"),
                new("Ea", LawVariableType.Scalar, "J/mol", "Activation energy"),
                new("R", LawVariableType.Scalar, "J/(mol*K)", "Gas constant"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
            ],
            BoundaryConditionDefs = [
                new("TemperatureRange", "T > 0 K", "Absolute temperature"),
                new("SingleStep", "elementary reaction", "Or effective overall parameters"),
            ],
            ApplicableDomains = ["Chemical kinetics", "Combustion", "Biochemistry"],
            Reference = "Arrhenius, S. (1889). Zeitschrift fuer Physikalische Chemie.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["arrhenius", "reaction rate", "activation energy"]
        });

        b.Add("chem.fick-diffusion", new LawDefinition
        {
            Id = "chem.fick-diffusion",
            Name = "Fick's First Law of Diffusion",
            Category = LawCategory.Chemistry,
            Expression = "J = -D * grad(c)",
            Description = "Diffusive flux proportional to negative concentration gradient.",
            Parameters = [new("D", 1e-9, "m^2/s", 0, 1e-3, "Diffusion coefficient")],
            Variables = [
                new("J", LawVariableType.Vector, "mol/(m^2*s)", "Molar flux"),
                new("D", LawVariableType.Scalar, "m^2/s", "Diffusion coefficient"),
                new("c", LawVariableType.Function, "mol/m^3", "Concentration"),
            ],
            BoundaryConditionDefs = [
                new("SteadyState", "or quasi-steady", "Or use Fick's 2nd law"),
                new("DiluteSolution", "ideal dilute", "Concentration-independent D"),
            ],
            ApplicableDomains = ["Mass transfer", "Materials science", "Biological membranes"],
            Reference = "Fick, A. (1855). Annalen der Physik.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["fick", "diffusion", "concentration", "flux"]
        });

        b.Add("chem.fick-second", new LawDefinition
        {
            Id = "chem.fick-second",
            Name = "Fick's Second Law of Diffusion",
            Category = LawCategory.Chemistry,
            Expression = "dc/dt = D * del^2(c)",
            Description = "Time evolution of concentration due to diffusion.",
            Parameters = [new("D", 1e-9, "m^2/s", 0, 1e-3, "Diffusion coefficient")],
            Variables = [
                new("c", LawVariableType.Function, "mol/m^3", "Concentration"),
                new("t", LawVariableType.Scalar, "s", "Time"),
                new("D", LawVariableType.Scalar, "m^2/s", "Diffusion coefficient"),
            ],
            BoundaryConditionDefs = [
                new("InitialConcentration", "c(x,0) = c0(x)", "Initial profile"),
            ],
            ApplicableDomains = ["Mass transfer", "Doping in semiconductors", "Drug delivery"],
            Reference = "Fick, A. (1855). Annalen der Physik.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["fick", "diffusion", "unsteady", "PDE"]
        });

        b.Add("chem.turing", new LawDefinition
        {
            Id = "chem.turing",
            Name = "Turing Reaction-Diffusion Pattern Formation",
            Category = LawCategory.Chemistry,
            Expression = "du/dt = D_u*del^2(u) + f(u,v); dv/dt = D_v*del^2(v) + g(u,v)",
            Description = "Coupled reaction-diffusion equations producing spatial patterns.",
            Parameters = [
                new("D_u", 1e-6, "m^2/s", 0, 1e-2, "Activator diffusivity"),
                new("D_v", 1e-5, "m^2/s", 0, 1e-2, "Inhibitor diffusivity"),
            ],
            Variables = [
                new("u", LawVariableType.Function, "mol/m^3", "Activator concentration"),
                new("v", LawVariableType.Function, "mol/m^3", "Inhibitor concentration"),
                new("D_u", LawVariableType.Scalar, "m^2/s", "Activator diffusivity"),
                new("D_v", LawVariableType.Scalar, "m^2/s", "Inhibitor diffusivity"),
            ],
            BoundaryConditionDefs = [
                new("TuringCondition", "D_v >> D_u", "Inhibitor diffuses much faster"),
            ],
            ApplicableDomains = ["Morphogenesis", "Chemical patterns", "Mathematical biology"],
            Reference = "Turing, A.M. (1952). Phil. Trans. Royal Society B.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["turing", "pattern formation", "reaction-diffusion"]
        });

        b.Add("chem.henderson-hasselbalch", new LawDefinition
        {
            Id = "chem.henderson-hasselbalch",
            Name = "Henderson-Hasselbalch Equation",
            Category = LawCategory.Chemistry,
            Expression = "pH = pKa + log([A-]/[HA])",
            Description = "Relates pH of a buffer solution to pKa and concentration ratio.",
            Parameters = [],
            Variables = [
                new("pH", LawVariableType.Scalar, "dimensionless", "pH of solution"),
                new("pKa", LawVariableType.Scalar, "dimensionless", "Acid dissociation constant"),
                new("A_minus", LawVariableType.Scalar, "mol/L", "Conjugate base concentration"),
                new("HA", LawVariableType.Scalar, "mol/L", "Weak acid concentration"),
            ],
            BoundaryConditionDefs = [
                new("BufferSolution", "weak acid + conjugate base", "Buffer regime"),
                new("DiluteSolution", "ideal behavior", "Activity ~ concentration"),
            ],
            ApplicableDomains = ["Biochemistry", "Clinical chemistry", "Water treatment"],
            Reference = "Henderson, L.J. (1908) / Hasselbalch, K. (1917).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["henderson-hasselbalch", "pH", "buffer", "acid-base"]
        });

        b.Add("chem.nernst", new LawDefinition
        {
            Id = "chem.nernst",
            Name = "Nernst Equation",
            Category = LawCategory.Chemistry,
            Expression = "E = E0 - (RT/(nF)) * ln(Q)",
            Description = "Electrode potential as a function of ion concentrations.",
            Parameters = [
                new("R", 8.31446, "J/(mol*K)", 0, 100, "Gas constant"),
                new("F", 96485.33212, "C/mol", 0, 1e5, "Faraday constant"),
            ],
            Variables = [
                new("E", LawVariableType.Scalar, "V", "Electrode potential"),
                new("E0", LawVariableType.Scalar, "V", "Standard electrode potential"),
                new("n", LawVariableType.Scalar, "dimensionless", "Electrons transferred"),
                new("Q", LawVariableType.Scalar, "dimensionless", "Reaction quotient"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
            ],
            BoundaryConditionDefs = [
                new("IdealSolution", "activity ~ concentration", "Dilute solutions"),
                new("Equilibrium", "reversible process", "Near equilibrium"),
            ],
            ApplicableDomains = ["Electrochemistry", "Batteries", "Corrosion"],
            Reference = "Nernst, W. (1889). Zeitschrift fuer Physikalische Chemie.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["nernst", "electrode", "potential", "electrochemistry"]
        });

        b.Add("chem.colligative-boiling", new LawDefinition
        {
            Id = "chem.colligative-boiling",
            Name = "Boiling Point Elevation",
            Category = LawCategory.Chemistry,
            Expression = "Delta_T_b = i * K_b * m",
            Description = "Elevation of boiling point due to dissolved solute.",
            Parameters = [
                new("K_b", 0.512, "K*kg/mol", 0, 10, "Ebullioscopic constant"),
            ],
            Variables = [
                new("Delta_T_b", LawVariableType.Scalar, "K", "Boiling point elevation"),
                new("i", LawVariableType.Scalar, "dimensionless", "van't Hoff factor"),
                new("K_b", LawVariableType.Scalar, "K*kg/mol", "Solvent ebullioscopic constant"),
                new("m", LawVariableType.Scalar, "mol/kg", "Molality of solute"),
            ],
            BoundaryConditionDefs = [
                new("DiluteSolution", "m << 1", "Ideal dilute solution"),
                new("NonVolatileSolute", "solute does not evaporate", "Only solvent evaporates"),
            ],
            ApplicableDomains = ["Solution chemistry", "Food science", "Antifreeze"],
            Reference = "Standard colligative property result.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["boiling point", "elevation", "colligative", "solute"]
        });
    }

    // ==============================================================
    //  BIOLOGY (5 laws)
    // ==============================================================
    private static void RegisterBiology(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("bio.logistic", new LawDefinition
        {
            Id = "bio.logistic",
            Name = "Logistic Growth",
            Category = LawCategory.Biology,
            Expression = "dN/dt = r*N*(1 - N/K)",
            Description = "Population growth with carrying capacity limiting exponential growth.",
            Parameters = [
                new("r", 0.1, "day^-1", 0, 10, "Intrinsic growth rate"),
                new("K", 1000.0, "individuals", 1, 1e12, "Carrying capacity"),
            ],
            Variables = [
                new("N", LawVariableType.Scalar, "individuals", "Population size"),
                new("r", LawVariableType.Scalar, "day^-1", "Growth rate"),
                new("K", LawVariableType.Scalar, "individuals", "Carrying capacity"),
                new("t", LawVariableType.Scalar, "days", "Time"),
            ],
            BoundaryConditionDefs = [
                new("InitialPopulation", "N(0) > 0", "Starting population required"),
                new("Bounded", "N <= K", "Population cannot exceed carrying capacity"),
            ],
            ApplicableDomains = ["Ecology", "Population dynamics", "Epidemiology"],
            Reference = "Verhulst, P.F. (1838). Correspondance Mathematique et Physique.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["logistic", "population", "carrying capacity"]
        });

        b.Add("bio.lotka-volterra", new LawDefinition
        {
            Id = "bio.lotka-volterra",
            Name = "Lotka-Volterra Predator-Prey Model",
            Category = LawCategory.Biology,
            Expression = "dx/dt = alpha*x - beta*x*y; dy/dt = delta*x*y - gamma*y",
            Description = "Coupled ODEs describing predator-prey population oscillations.",
            Parameters = [
                new("alpha", 1.1, "day^-1", 0, 10, "Prey growth rate"),
                new("beta", 0.4, "day^-1", 0, 10, "Predation rate"),
                new("delta", 0.1, "day^-1", 0, 10, "Predator efficiency"),
                new("gamma", 0.4, "day^-1", 0, 10, "Predator death rate"),
            ],
            Variables = [
                new("x", LawVariableType.Scalar, "individuals", "Prey population"),
                new("y", LawVariableType.Scalar, "individuals", "Predator population"),
                new("t", LawVariableType.Scalar, "days", "Time"),
                new("alpha", LawVariableType.Scalar, "day^-1", "Prey growth"),
                new("beta", LawVariableType.Scalar, "day^-1", "Predation rate"),
                new("delta", LawVariableType.Scalar, "day^-1", "Predator efficiency"),
                new("gamma", LawVariableType.Scalar, "day^-1", "Predator death"),
            ],
            BoundaryConditionDefs = [
                new("PositivePopulations", "x > 0, y > 0", "Non-negative populations"),
            ],
            ApplicableDomains = ["Ecology", "Population dynamics", "Conservation biology"],
            Reference = "Lotka, A.J. (1925) / Volterra, V. (1926).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["lotka-volterra", "predator-prey", "ecology", "oscillation"]
        });

        b.Add("bio.michaelis-menten", new LawDefinition
        {
            Id = "bio.michaelis-menten",
            Name = "Michaelis-Menten Enzyme Kinetics",
            Category = LawCategory.Biology,
            Expression = "v = Vmax * [S] / (Km + [S])",
            Description = "Enzyme reaction rate as a function of substrate concentration.",
            Parameters = [
                new("Vmax", 10.0, "umol/min", 0, 1e6, "Maximum reaction velocity"),
                new("Km", 50.0, "uM", 0, 1e6, "Michaelis constant"),
            ],
            Variables = [
                new("v", LawVariableType.Scalar, "umol/min", "Reaction velocity"),
                new("Vmax", LawVariableType.Scalar, "umol/min", "Max velocity"),
                new("S", LawVariableType.Scalar, "uM", "Substrate concentration"),
                new("Km", LawVariableType.Scalar, "uM", "Michaelis constant"),
            ],
            BoundaryConditionDefs = [
                new("SteadyState", "d[ES]/dt ~ 0", "Quasi-steady state approximation"),
                new("ExcessSubstrate", "[S] >> [E]_total", "Substrate in excess"),
            ],
            ApplicableDomains = ["Biochemistry", "Pharmacology", "Metabolism"],
            Reference = "Michaelis, L. & Menten, M.L. (1913). Biochemische Zeitschrift.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["michaelis-menten", "enzyme", "kinetics", "Vmax", "Km"]
        });

        b.Add("bio.hodgkin-huxley", new LawDefinition
        {
            Id = "bio.hodgkin-huxley",
            Name = "Hodgkin-Huxley Neuron Model",
            Category = LawCategory.Biology,
            Expression = "C_m*dV/dt = -g_Na*m^3*h*(V-E_Na) - g_K*n^4*(V-E_K) - g_L*(V-E_L) + I_ext",
            Description = "Describes action potential generation in squid giant axon.",
            Parameters = [
                new("C_m", 1.0, "uF/cm^2", 0, 100, "Membrane capacitance"),
                new("g_Na", 120.0, "mS/cm^2", 0, 1000, "Na+ conductance"),
                new("g_K", 36.0, "mS/cm^2", 0, 1000, "K+ conductance"),
                new("g_L", 0.3, "mS/cm^2", 0, 100, "Leak conductance"),
                new("E_Na", 50.0, "mV", -100, 100, "Na+ reversal potential"),
                new("E_K", -77.0, "mV", -100, 100, "K+ reversal potential"),
                new("E_L", -54.4, "mV", -100, 100, "Leak reversal potential"),
            ],
            Variables = [
                new("V", LawVariableType.Scalar, "mV", "Membrane potential"),
                new("m", LawVariableType.Scalar, "dimensionless", "Na+ activation gate"),
                new("h", LawVariableType.Scalar, "dimensionless", "Na+ inactivation gate"),
                new("n", LawVariableType.Scalar, "dimensionless", "K+ activation gate"),
                new("I_ext", LawVariableType.Scalar, "uA/cm^2", "External current"),
                new("t", LawVariableType.Scalar, "ms", "Time"),
            ],
            BoundaryConditionDefs = [
                new("Temperature", "T = 6.3 C", "Original experiments on squid axon"),
            ],
            ApplicableDomains = ["Computational neuroscience", "Neural modeling", "Neurophysiology"],
            Reference = "Hodgkin, A.L. & Huxley, A.F. (1952). J. Physiol.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["hodgkin-huxley", "neuron", "action potential", "ion channel"]
        });

        b.Add("bio.sir", new LawDefinition
        {
            Id = "bio.sir",
            Name = "SIR Epidemic Model",
            Category = LawCategory.Biology,
            Expression = "dS/dt = -beta*S*I/N; dI/dt = beta*S*I/N - gamma*I; dR/dt = gamma*I",
            Description = "Compartmental model for infectious disease spread.",
            Parameters = [
                new("beta", 0.3, "day^-1", 0, 10, "Transmission rate"),
                new("gamma", 0.1, "day^-1", 0, 10, "Recovery rate"),
                new("N", 10000.0, "individuals", 1, 1e12, "Total population"),
            ],
            Variables = [
                new("S", LawVariableType.Scalar, "individuals", "Susceptible"),
                new("I", LawVariableType.Scalar, "individuals", "Infected"),
                new("R", LawVariableType.Scalar, "individuals", "Recovered"),
                new("beta", LawVariableType.Scalar, "day^-1", "Transmission rate"),
                new("gamma", LawVariableType.Scalar, "day^-1", "Recovery rate"),
                new("N", LawVariableType.Scalar, "individuals", "Total population"),
                new("t", LawVariableType.Scalar, "days", "Time"),
            ],
            BoundaryConditionDefs = [
                new("ClosedPopulation", "S + I + R = N", "No births, deaths, or migration"),
                new("MassAction", "random mixing", "Homogeneous mixing"),
            ],
            ApplicableDomains = ["Epidemiology", "Public health", "Infectious disease modeling"],
            Reference = "Kermack, W.O. & McKendrick, A.G. (1927). Proc. Royal Society A.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["SIR", "epidemic", "infectious disease", "compartmental"]
        });
    }

    // ==============================================================
    //  FINANCE (4 laws)
    // ==============================================================
    private static void RegisterFinance(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("fin.black-scholes", new LawDefinition
        {
            Id = "fin.black-scholes",
            Name = "Black-Scholes Equation",
            Category = LawCategory.Finance,
            Expression = "dV/dt + 0.5*sigma^2*S^2*d^2V/dS^2 + r*S*dV/dS - r*V = 0",
            Description = "PDE for pricing European-style derivative securities.",
            Parameters = [
                new("r", 0.05, "year^-1", 0, 1, "Risk-free interest rate"),
                new("sigma", 0.2, "year^-0.5", 0, 2, "Volatility"),
            ],
            Variables = [
                new("V", LawVariableType.Scalar, "currency", "Option value"),
                new("S", LawVariableType.Scalar, "currency", "Underlying asset price"),
                new("t", LawVariableType.Scalar, "years", "Time to expiry"),
                new("r", LawVariableType.Scalar, "year^-1", "Risk-free rate"),
                new("sigma", LawVariableType.Scalar, "year^-0.5", "Volatility"),
            ],
            BoundaryConditionDefs = [
                new("EuropeanOption", "exercisable only at expiry", "European exercise style"),
                new("NoArbitrage", "no risk-free profit", "Efficient market"),
                new("ConstantVolatility", "sigma = const", "Volatility does not change"),
            ],
            ApplicableDomains = ["Options pricing", "Risk management", "Quantitative finance"],
            Reference = "Black, F. & Scholes, M. (1973). J. Political Economy.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["black-scholes", "options", "derivative", "PDE"]
        });

        b.Add("fin.gbm", new LawDefinition
        {
            Id = "fin.gbm",
            Name = "Geometric Brownian Motion",
            Category = LawCategory.Finance,
            Expression = "dS = mu*S*dt + sigma*S*dW",
            Description = "Stochastic process modeling stock price dynamics.",
            Parameters = [
                new("mu", 0.08, "year^-1", -1, 5, "Drift rate"),
                new("sigma", 0.2, "year^-0.5", 0, 2, "Volatility"),
            ],
            Variables = [
                new("S", LawVariableType.Scalar, "currency", "Asset price"),
                new("mu", LawVariableType.Scalar, "year^-1", "Drift rate"),
                new("sigma", LawVariableType.Scalar, "year^-0.5", "Volatility"),
                new("t", LawVariableType.Scalar, "years", "Time"),
                new("dW", LawVariableType.Scalar, "year^0.5", "Wiener increment"),
            ],
            BoundaryConditionDefs = [
                new("PositivePrice", "S > 0", "Price must be positive"),
            ],
            ApplicableDomains = ["Stock modeling", "Risk analysis", "Portfolio theory"],
            Reference = "Black, F. & Scholes, M. (1973); Merton, R.C. (1973).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["GBM", "stochastic", "stock price", "Brownian motion"]
        });

        b.Add("fin.vasicek", new LawDefinition
        {
            Id = "fin.vasicek",
            Name = "Vasicek Interest Rate Model",
            Category = LawCategory.Finance,
            Expression = "dr = a*(b - r)*dt + sigma*dW",
            Description = "Mean-reverting stochastic process for short-term interest rates.",
            Parameters = [
                new("a", 0.1, "year^-1", 0, 5, "Mean reversion speed"),
                new("b", 0.05, "year^-1", 0, 1, "Long-term mean rate"),
                new("sigma", 0.01, "year^-1.5", 0, 0.5, "Volatility"),
            ],
            Variables = [
                new("r", LawVariableType.Scalar, "year^-1", "Short-term interest rate"),
                new("a", LawVariableType.Scalar, "year^-1", "Mean reversion speed"),
                new("b", LawVariableType.Scalar, "year^-1", "Long-term mean"),
                new("sigma", LawVariableType.Scalar, "year^-1.5", "Volatility"),
                new("t", LawVariableType.Scalar, "years", "Time"),
            ],
            BoundaryConditionDefs = [
                new("PositiveMeanReversion", "a > 0", "Must revert to mean"),
            ],
            ApplicableDomains = ["Bond pricing", "Interest rate derivatives", "Term structure"],
            Reference = "Vasicek, O. (1977). J. Finance.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["vasicek", "interest rate", "mean reversion", "term structure"]
        });

        b.Add("fin.kelly", new LawDefinition
        {
            Id = "fin.kelly",
            Name = "Kelly Criterion",
            Category = LawCategory.Finance,
            Expression = "f* = (p*b - q) / b",
            Description = "Optimal fraction of wealth to bet for maximum long-term growth.",
            Parameters = [],
            Variables = [
                new("f_star", LawVariableType.Scalar, "dimensionless", "Optimal bet fraction"),
                new("p", LawVariableType.Scalar, "dimensionless", "Probability of winning"),
                new("q", LawVariableType.Scalar, "dimensionless", "Probability of losing (1-p)"),
                new("b", LawVariableType.Scalar, "dimensionless", "Net odds (payout per unit risked)"),
            ],
            BoundaryConditionDefs = [
                new("PositiveEdge", "p*b > q", "Expected value must be positive"),
                new("DiscreteBets", "sequential bets", "Repeated independent bets"),
            ],
            ApplicableDomains = ["Portfolio management", "Gambling", "Trading strategy"],
            Reference = "Kelly, J.L. (1956). Bell System Technical Journal.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["kelly", "betting", "optimization", "growth rate"]
        });
    }

    // ==============================================================
    //  EPIDEMIOLOGY (3 laws)
    // ==============================================================
    private static void RegisterEpidemiology(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("epi.seir", new LawDefinition
        {
            Id = "epi.seir",
            Name = "SEIR Epidemic Model",
            Category = LawCategory.Epidemiology,
            Expression = "dS/dt = -beta*S*I/N; dE/dt = beta*S*I/N - sigma*E; dI/dt = sigma*E - gamma*I; dR/dt = gamma*I",
            Description = "SEIR model with exposed (latent) compartment.",
            Parameters = [
                new("beta", 0.3, "day^-1", 0, 10, "Transmission rate"),
                new("sigma", 0.2, "day^-1", 0, 10, "Incubation rate (1/latent period)"),
                new("gamma", 0.1, "day^-1", 0, 10, "Recovery rate"),
                new("N", 10000.0, "individuals", 1, 1e12, "Total population"),
            ],
            Variables = [
                new("S", LawVariableType.Scalar, "individuals", "Susceptible"),
                new("E", LawVariableType.Scalar, "individuals", "Exposed (latent)"),
                new("I", LawVariableType.Scalar, "individuals", "Infectious"),
                new("R", LawVariableType.Scalar, "individuals", "Recovered"),
                new("t", LawVariableType.Scalar, "days", "Time"),
            ],
            BoundaryConditionDefs = [
                new("ClosedPopulation", "S+E+I+R = N", "No births/deaths/migration"),
                new("MassAction", "random mixing", "Homogeneous mixing"),
            ],
            ApplicableDomains = ["Epidemiology", "Public health", "Pandemic planning"],
            Reference = "Kermack & McKendrick (1927); extended SEIR formulation.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["SEIR", "epidemic", "exposed", "latent period"]
        });

        b.Add("epi.r0", new LawDefinition
        {
            Id = "epi.r0",
            Name = "Basic Reproduction Number (R0)",
            Category = LawCategory.Epidemiology,
            Expression = "R0 = beta / gamma",
            Description = "Average number of secondary infections from one infectious individual in a fully susceptible population.",
            Parameters = [
                new("beta", 0.3, "day^-1", 0, 10, "Transmission rate"),
                new("gamma", 0.1, "day^-1", 0, 10, "Recovery rate"),
            ],
            Variables = [
                new("R0", LawVariableType.Scalar, "dimensionless", "Basic reproduction number"),
                new("beta", LawVariableType.Scalar, "day^-1", "Transmission rate"),
                new("gamma", LawVariableType.Scalar, "day^-1", "Recovery rate"),
            ],
            BoundaryConditionDefs = [
                new("FullySusceptible", "S ~ N", "At outbreak start"),
            ],
            ApplicableDomains = ["Epidemiology", "Public health policy", "Vaccination strategy"],
            Reference = "Diekmann, O. & Heesterbeek, J.A.P. (2000).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["R0", "reproduction number", "threshold", "epidemic"]
        });

        b.Add("epi.sir-effective-r", new LawDefinition
        {
            Id = "epi.sir-effective-r",
            Name = "Effective Reproduction Number (Re)",
            Category = LawCategory.Epidemiology,
            Expression = "Re = R0 * S / N = R0 * (1 - p)",
            Description = "Current reproduction number accounting for population immunity.",
            Parameters = [],
            Variables = [
                new("Re", LawVariableType.Scalar, "dimensionless", "Effective reproduction number"),
                new("R0", LawVariableType.Scalar, "dimensionless", "Basic reproduction number"),
                new("S", LawVariableType.Scalar, "individuals", "Susceptible population"),
                new("N", LawVariableType.Scalar, "individuals", "Total population"),
                new("p", LawVariableType.Scalar, "dimensionless", "Fraction immune (vaccinated or recovered)"),
            ],
            BoundaryConditionDefs = [
                new("HerdImmunity", "Re < 1", "Epidemic declines when Re < 1"),
            ],
            ApplicableDomains = ["Epidemiology", "Vaccination policy", "Pandemic monitoring"],
            Reference = "Standard result from SIR model analysis.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["effective R", "herd immunity", "vaccination", "threshold"]
        });
    }

    // ==============================================================
    //  MATERIAL SCIENCE (4 laws)
    // ==============================================================
    private static void RegisterMaterialScience(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("mat.hall-petch", new LawDefinition
        {
            Id = "mat.hall-petch",
            Name = "Hall-Petch Relation (Grain Size Strengthening)",
            Category = LawCategory.MaterialScience,
            Expression = "sigma_y = sigma_0 + k_y / sqrt(d)",
            Description = "Yield strength increases as grain size decreases.",
            Parameters = [
                new("sigma_0", 50e6, "Pa", 0, 1e9, "Friction stress"),
                new("k_y", 0.7e6, "Pa*sqrt(m)", 0, 1e7, "Hall-Petch coefficient"),
            ],
            Variables = [
                new("sigma_y", LawVariableType.Scalar, "Pa", "Yield strength"),
                new("sigma_0", LawVariableType.Scalar, "Pa", "Friction stress"),
                new("k_y", LawVariableType.Scalar, "Pa*sqrt(m)", "Hall-Petch coefficient"),
                new("d", LawVariableType.Scalar, "m", "Average grain diameter"),
            ],
            BoundaryConditionDefs = [
                new("EquiaxedGrains", "roughly spherical grains", "Polycrystalline material"),
                new("GrainSizeRange", "d > ~10 nm", "Breaks down for nanocrystalline materials"),
            ],
            ApplicableDomains = ["Metallurgy", "Materials strengthening", "Nanomaterials"],
            Reference = "Hall, E.O. (1951) / Petch, N.J. (1953).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["hall-petch", "grain size", "yield strength", "strengthening"]
        });

        b.Add("mat.ashby", new LawDefinition
        {
            Id = "mat.ashby",
            Name = "Ashby Chart (Material Selection)",
            Category = LawCategory.MaterialScience,
            Expression = "Performance = C * (rho / E)^(1/n) * (1 / cost)",
            Description = "Material performance index for lightweight stiff design.",
            Parameters = [],
            Variables = [
                new("Performance", LawVariableType.Scalar, "varies", "Performance index"),
                new("C", LawVariableType.Scalar, "varies", "Geometric constant"),
                new("rho", LawVariableType.Scalar, "kg/m^3", "Density"),
                new("E", LawVariableType.Scalar, "Pa", "Young's modulus"),
                new("cost", LawVariableType.Scalar, "currency/kg", "Material cost"),
            ],
            BoundaryConditionDefs = [
                new("LinearElastic", "within elastic regime", "Hooke's law valid"),
            ],
            ApplicableDomains = ["Material selection", "Design engineering", "Lightweight design"],
            Reference = "Ashby, M.F. (1992). Materials Selection in Mechanical Design.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["ashby", "material selection", "performance", "lightweight"]
        });

        b.Add("mat.thermal-expansion", new LawDefinition
        {
            Id = "mat.thermal-expansion",
            Name = "Linear Thermal Expansion",
            Category = LawCategory.MaterialScience,
            Expression = "Delta_L = alpha * L0 * Delta_T",
            Description = "Length change of a material due to temperature change.",
            Parameters = [new("alpha", 12e-6, "K^-1", 0, 1e-3, "Coefficient of thermal expansion")],
            Variables = [
                new("Delta_L", LawVariableType.Scalar, "m", "Length change"),
                new("alpha", LawVariableType.Scalar, "K^-1", "CTE"),
                new("L0", LawVariableType.Scalar, "m", "Original length"),
                new("Delta_T", LawVariableType.Scalar, "K", "Temperature change"),
            ],
            BoundaryConditionDefs = [
                new("SmallStrain", "Delta_L << L0", "Linear approximation"),
                new("IsotropicMaterial", "uniform expansion", "Same in all directions"),
            ],
            ApplicableDomains = ["Structural design", "Thermal stress", "Dimensional stability"],
            Reference = "Standard result in materials science.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["thermal expansion", "CTE", "temperature", "dimensional"]
        });

        b.Add("mat.fatigue-s-n", new LawDefinition
        {
            Id = "mat.fatigue-s-n",
            Name = "Wohler S-N Curve (Fatigue)",
            Category = LawCategory.MaterialScience,
            Expression = "sigma_a = sigma_f * (2*N_f)^b",
            Description = "Stress amplitude vs. cycles to failure for fatigue life estimation.",
            Parameters = [
                new("sigma_f", 800e6, "Pa", 0, 5e9, "Fatigue strength coefficient"),
                new("b", -0.08, "dimensionless", -0.5, 0, "Fatigue strength exponent"),
            ],
            Variables = [
                new("sigma_a", LawVariableType.Scalar, "Pa", "Stress amplitude"),
                new("sigma_f", LawVariableType.Scalar, "Pa", "Fatigue strength coefficient"),
                new("N_f", LawVariableType.Scalar, "cycles", "Cycles to failure"),
                new("b", LawVariableType.Scalar, "dimensionless", "Basquin exponent"),
            ],
            BoundaryConditionDefs = [
                new("HighCycle", "N_f > 10^3", "High-cycle fatigue regime"),
                new("UniaxialLoading", "R = -1", "Fully reversed loading"),
            ],
            ApplicableDomains = ["Fatigue design", "Structural integrity", "Reliability engineering"],
            Reference = "Wohler, A. (1860) / Basquin, O.H. (1910).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["fatigue", "S-N curve", "Wohler", "Basquin", "life estimation"]
        });
    }

    // ==============================================================
    //  ACOUSTICS (3 laws)
    // ==============================================================
    private static void RegisterAcoustics(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("acoust.doppler", new LawDefinition
        {
            Id = "acoust.doppler",
            Name = "Doppler Effect (Sound)",
            Category = LawCategory.Acoustics,
            Expression = "f_obs = f_src * (v + v_obs) / (v - v_src)",
            Description = "Frequency shift due to relative motion between source and observer.",
            Parameters = [],
            Variables = [
                new("f_obs", LawVariableType.Scalar, "Hz", "Observed frequency"),
                new("f_src", LawVariableType.Scalar, "Hz", "Source frequency"),
                new("v", LawVariableType.Scalar, "m/s", "Speed of sound"),
                new("v_obs", LawVariableType.Scalar, "m/s", "Observer velocity (toward source)"),
                new("v_src", LawVariableType.Scalar, "m/s", "Source velocity (toward observer)"),
            ],
            BoundaryConditionDefs = [
                new("SubsonicSource", "v_src < v", "Source slower than sound"),
                new("MediumStill", "medium at rest", "Or adjust for wind"),
            ],
            ApplicableDomains = ["Radar", "Sonar", "Medical ultrasound", "Astronomy"],
            Reference = "Doppler, C. (1842). Monatsberichte der Akademie.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["doppler", "frequency shift", "sonar", "radar"]
        });

        b.Add("acoust.inverse-square", new LawDefinition
        {
            Id = "acoust.inverse-square",
            Name = "Inverse Square Law (Sound Intensity)",
            Category = LawCategory.Acoustics,
            Expression = "I = P / (4 * pi * r^2)",
            Description = "Sound intensity decreases as inverse square of distance from source.",
            Parameters = [],
            Variables = [
                new("I", LawVariableType.Scalar, "W/m^2", "Sound intensity"),
                new("P", LawVariableType.Scalar, "W", "Sound power"),
                new("r", LawVariableType.Scalar, "m", "Distance from source"),
            ],
            BoundaryConditionDefs = [
                new("FreeField", "no reflections", "Open space"),
                new("PointSource", "omnidirectional", "Isotropic radiation"),
            ],
            ApplicableDomains = ["Noise control", "Architectural acoustics", "Sonar"],
            Reference = "Standard result in acoustics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["intensity", "inverse square", "distance", "sound power"]
        });

        b.Add("acoust.speed-of-sound", new LawDefinition
        {
            Id = "acoust.speed-of-sound",
            Name = "Speed of Sound in an Ideal Gas",
            Category = LawCategory.Acoustics,
            Expression = "c = sqrt(gamma * R * T / M)",
            Description = "Speed of sound depends on gas properties and temperature.",
            Parameters = [
                new("R", 8.31446, "J/(mol*K)", 0, 100, "Universal gas constant"),
            ],
            Variables = [
                new("c", LawVariableType.Scalar, "m/s", "Speed of sound"),
                new("gamma", LawVariableType.Scalar, "dimensionless", "Heat capacity ratio"),
                new("R", LawVariableType.Scalar, "J/(mol*K)", "Gas constant"),
                new("T", LawVariableType.Scalar, "K", "Absolute temperature"),
                new("M", LawVariableType.Scalar, "kg/mol", "Molar mass"),
            ],
            BoundaryConditionDefs = [
                new("IdealGas", "low pressure, high temp", "Ideal gas behavior"),
                new("NoDispersion", "frequency-independent", "No significant dispersion"),
            ],
            ApplicableDomains = ["Acoustics", "Aerodynamics", "Gas dynamics"],
            Reference = "Newton, I. (1687) / Laplace, P.S. (1816).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["speed of sound", "acoustic", "gas", "temperature"]
        });
    }

    // ==============================================================
    //  PLASMA (3 laws)
    // ==============================================================
    private static void RegisterPlasma(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("plasma.debye", new LawDefinition
        {
            Id = "plasma.debye",
            Name = "Debye Length",
            Category = LawCategory.Plasma,
            Expression = "lambda_D = sqrt(epsilon_0 * k_B * T / (n_e * e^2))",
            Description = "Characteristic length over which electric fields are screened in a plasma.",
            Parameters = [
                new("epsilon_0", 8.8541878128e-12, "F/m", 0, 1e-6, "Vacuum permittivity"),
                new("k_B", 1.380649e-23, "J/K", 0, 1e-15, "Boltzmann constant"),
                new("e", 1.602176634e-19, "C", 0, 1e-10, "Elementary charge"),
            ],
            Variables = [
                new("lambda_D", LawVariableType.Scalar, "m", "Debye length"),
                new("T", LawVariableType.Scalar, "K", "Electron temperature"),
                new("n_e", LawVariableType.Scalar, "m^-3", "Electron density"),
                new("epsilon_0", LawVariableType.Scalar, "F/m", "Permittivity"),
                new("k_B", LawVariableType.Scalar, "J/K", "Boltzmann constant"),
                new("e", LawVariableType.Scalar, "C", "Elementary charge"),
            ],
            BoundaryConditionDefs = [
                new("Quasineutral", "n_e ~ n_i", "Overall neutrality"),
                new("Maxwellian", "thermal equilibrium", "Boltzmann distribution"),
            ],
            ApplicableDomains = ["Plasma physics", "Fusion research", "Space physics"],
            Reference = "Debye, P. (1918). Physikalische Zeitschrift.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["debye", "screening", "plasma", "Debye length"]
        });

        b.Add("plasma.alfven", new LawDefinition
        {
            Id = "plasma.alfven",
            Name = "Alfven Wave Velocity",
            Category = LawCategory.Plasma,
            Expression = "v_A = B / sqrt(mu_0 * rho)",
            Description = "Speed of electromagnetic waves in a magnetized plasma.",
            Parameters = [new("mu_0", 1.25663706212e-6, "H/m", 0, 1, "Vacuum permeability")],
            Variables = [
                new("v_A", LawVariableType.Scalar, "m/s", "Alfven velocity"),
                new("B", LawVariableType.Scalar, "T", "Magnetic field"),
                new("mu_0", LawVariableType.Scalar, "H/m", "Permeability"),
                new("rho", LawVariableType.Scalar, "kg/m^3", "Plasma mass density"),
            ],
            BoundaryConditionDefs = [
                new("MHD", "low frequency", "Valid in MHD approximation"),
                new("LowBeta", "beta << 1", "Magnetic pressure dominates"),
            ],
            ApplicableDomains = ["Plasma physics", "Solar physics", "Fusion", "Space weather"],
            Reference = "Alfven, H. (1942). Nature.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["alfven", "wave", "MHD", "plasma"]
        });

        b.Add("plasma.gyrofrequency", new LawDefinition
        {
            Id = "plasma.gyrofrequency",
            Name = "Cyclotron (Gyro) Frequency",
            Category = LawCategory.Plasma,
            Expression = "omega_c = e*B / m",
            Description = "Angular frequency of a charged particle spiraling around magnetic field lines.",
            Parameters = [
                new("e", 1.602176634e-19, "C", 0, 1e-10, "Elementary charge"),
            ],
            Variables = [
                new("omega_c", LawVariableType.Scalar, "rad/s", "Cyclotron frequency"),
                new("e", LawVariableType.Scalar, "C", "Charge"),
                new("B", LawVariableType.Scalar, "T", "Magnetic field"),
                new("m", LawVariableType.Scalar, "kg", "Particle mass"),
            ],
            BoundaryConditionDefs = [
                new("UniformField", "B = const", "Uniform magnetic field"),
                new("NonRelativistic", "v << c", "Classical treatment"),
            ],
            ApplicableDomains = ["Plasma confinement", "Cyclotron heating", "Mass spectrometry"],
            Reference = "Standard result in plasma physics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["cyclotron", "gyro", "frequency", "magnetic", "plasma"]
        });
    }

    // ==============================================================
    //  NUCLEAR (3 laws)
    // ==============================================================
    private static void RegisterNuclear(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("nuclear.radioactive-decay", new LawDefinition
        {
            Id = "nuclear.radioactive-decay",
            Name = "Radioactive Decay Law",
            Category = LawCategory.Nuclear,
            Expression = "N(t) = N0 * exp(-lambda*t)",
            Description = "Exponential decrease of radioactive nuclei over time.",
            Parameters = [
                new("lambda", 0.0231, "s^-1", 1e-30, 1e10, "Decay constant"),
            ],
            Variables = [
                new("N", LawVariableType.Scalar, "nuclei", "Remaining nuclei"),
                new("N0", LawVariableType.Scalar, "nuclei", "Initial nuclei count"),
                new("lambda", LawVariableType.Scalar, "s^-1", "Decay constant"),
                new("t", LawVariableType.Scalar, "s", "Time"),
            ],
            BoundaryConditionDefs = [
                new("LargeN", "N >> 1", "Statistical treatment valid"),
                new("IndependentDecays", "no chain effects", "Or account for daughter products"),
            ],
            ApplicableDomains = ["Nuclear physics", "Radiocarbon dating", "Nuclear medicine"],
            Reference = "Rutherford, E. & Soddy, F. (1902).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["radioactive", "decay", "exponential", "half-life"]
        });

        b.Add("nuclear.mass-energy", new LawDefinition
        {
            Id = "nuclear.mass-energy",
            Name = "Einstein Mass-Energy Equivalence",
            Category = LawCategory.Nuclear,
            Expression = "E = m * c^2",
            Description = "Mass-energy equivalence; energy contained in rest mass.",
            Parameters = [
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
            ],
            Variables = [
                new("E", LawVariableType.Scalar, "J", "Energy"),
                new("m", LawVariableType.Scalar, "kg", "Rest mass"),
                new("c", LawVariableType.Scalar, "m/s", "Speed of light"),
            ],
            BoundaryConditionDefs = [
                new("RestFrame", "particle at rest", "Rest energy"),
            ],
            ApplicableDomains = ["Nuclear physics", "Particle physics", "Energy generation"],
            Reference = "Einstein, A. (1905). Annalen der Physik.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["einstein", "mass-energy", "equivalence", "relativity"]
        });

        b.Add("nuclear.fission-energy", new LawDefinition
        {
            Id = "nuclear.fission-energy",
            Name = "Nuclear Fission Energy Release",
            Category = LawCategory.Nuclear,
            Expression = "Q = Delta_m * c^2 = (m_reactants - m_products) * c^2",
            Description = "Energy released in fission from mass defect.",
            Parameters = [
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
            ],
            Variables = [
                new("Q", LawVariableType.Scalar, "J", "Energy released"),
                new("Delta_m", LawVariableType.Scalar, "kg", "Mass defect"),
                new("c", LawVariableType.Scalar, "m/s", "Speed of light"),
                new("m_reactants", LawVariableType.Scalar, "kg", "Mass of reactants"),
                new("m_products", LawVariableType.Scalar, "kg", "Mass of products"),
            ],
            BoundaryConditionDefs = [
                new("Subcritical", "k_eff < 1", "Controlled fission"),
            ],
            ApplicableDomains = ["Nuclear power", "Nuclear weapons", "Stellar nucleosynthesis"],
            Reference = "Hahn, O. & Strassmann, F. (1939) / Meitner, L. & Frisch, O. (1939).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["fission", "mass defect", "nuclear energy"]
        });
    }

    // ==============================================================
    //  ASTROPHYSICS (3 laws)
    // ==============================================================
    private static void RegisterAstrophysics(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("astro.hubble", new LawDefinition
        {
            Id = "astro.hubble",
            Name = "Hubble's Law (Expansion of the Universe)",
            Category = LawCategory.Astrophysics,
            Expression = "v = H0 * d",
            Description = "Galaxy recession velocity is proportional to distance.",
            Parameters = [new("H0", 70.0, "km/s/Mpc", 0, 200, "Hubble constant")],
            Variables = [
                new("v", LawVariableType.Scalar, "km/s", "Recession velocity"),
                new("H0", LawVariableType.Scalar, "km/s/Mpc", "Hubble constant"),
                new("d", LawVariableType.Scalar, "Mpc", "Distance"),
            ],
            BoundaryConditionDefs = [
                new("LocalUniverse", "d < ~1000 Mpc", "Approximation for nearby galaxies"),
                new("FLRW", "homogeneous universe", "Friedmann-Lemaitre-Robertson-Walker metric"),
            ],
            ApplicableDomains = ["Cosmology", "Galaxy surveys", "Distance ladder"],
            Reference = "Hubble, E. (1929). Proc. National Academy of Sciences.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["hubble", "expansion", "cosmology", "recession"]
        });

        b.Add("astro.luminosity", new LawDefinition
        {
            Id = "astro.luminosity",
            Name = "Stefan-Boltzmann Luminosity (Stars)",
            Category = LawCategory.Astrophysics,
            Expression = "L = 4*pi*R^2*sigma*T_eff^4",
            Description = "Total luminosity of a star from its radius and effective temperature.",
            Parameters = [new("sigma", 5.670374e-8, "W/(m^2*K^4)", 0, 1, "Stefan-Boltzmann constant")],
            Variables = [
                new("L", LawVariableType.Scalar, "W", "Luminosity"),
                new("R", LawVariableType.Scalar, "m", "Stellar radius"),
                new("sigma", LawVariableType.Scalar, "W/(m^2*K^4)", "SB constant"),
                new("T_eff", LawVariableType.Scalar, "K", "Effective temperature"),
            ],
            BoundaryConditionDefs = [
                new("BlackBody", "star as black body", "Approximation"),
            ],
            ApplicableDomains = ["Stellar astrophysics", "HR diagram", "Exoplanet studies"],
            Reference = "Standard result from Stefan-Boltzmann law applied to stars.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["luminosity", "star", "temperature", "radius"]
        });

        b.Add("astro.chandrasekhar", new LawDefinition
        {
            Id = "astro.chandrasekhar",
            Name = "Chandrasekhar Limit",
            Category = LawCategory.Astrophysics,
            Expression = "M_Ch ~ 1.4 * M_sun",
            Description = "Maximum mass of a stable white dwarf; above it collapses to neutron star.",
            Parameters = [
                new("M_sun", 1.989e30, "kg", 0, 1e35, "Solar mass"),
            ],
            Variables = [
                new("M_Ch", LawVariableType.Scalar, "kg", "Chandrasekhar mass"),
                new("M_sun", LawVariableType.Scalar, "kg", "Solar mass"),
            ],
            BoundaryConditionDefs = [
                new("ElectronDegeneracy", "electron-degenerate matter", "White dwarf composition"),
                new("NonRotating", "no rotation", "Non-rotating star"),
            ],
            ApplicableDomains = ["Stellar evolution", "Supernovae", "Neutron stars"],
            Reference = "Chandrasekhar, S. (1931). Astrophysical Journal.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["chandrasekhar", "white dwarf", "mass limit", "stellar"]
        });
    }

    // ==============================================================
    //  CLIMATE (3 laws)
    // ==============================================================
    private static void RegisterClimate(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("climate.radiative-balance", new LawDefinition
        {
            Id = "climate.radiative-balance",
            Name = "Planetary Radiative Equilibrium",
            Category = LawCategory.Climate,
            Expression = "T_eff = (S*(1-alpha)/(4*sigma))^(1/4)",
            Description = "Effective temperature of a planet from absorbed solar radiation.",
            Parameters = [
                new("sigma", 5.670374e-8, "W/(m^2*K^4)", 0, 1, "Stefan-Boltzmann constant"),
                new("S", 1361.0, "W/m^2", 0, 1e5, "Solar irradiance"),
            ],
            Variables = [
                new("T_eff", LawVariableType.Scalar, "K", "Effective temperature"),
                new("S", LawVariableType.Scalar, "W/m^2", "Solar constant"),
                new("alpha", LawVariableType.Scalar, "dimensionless", "Bond albedo"),
                new("sigma", LawVariableType.Scalar, "W/(m^2*K^4)", "SB constant"),
            ],
            BoundaryConditionDefs = [
                new("ThermalEquilibrium", "in = out", "Energy balance"),
                new("UniformTemperature", "T = const over surface", "Simplified model"),
            ],
            ApplicableDomains = ["Climate science", "Planetary science", "Paleoclimatology"],
            Reference = "Standard result in climate physics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["radiative equilibrium", "temperature", "albedo", "solar"]
        });

        b.Add("climate.greenhouse", new LawDefinition
        {
            Id = "climate.greenhouse",
            Name = "Simple Greenhouse Model",
            Category = LawCategory.Climate,
            Expression = "T_s = T_eff * (1 + 0.75*tau)^(1/4)",
            Description = "Surface temperature enhancement from atmospheric greenhouse effect.",
            Parameters = [],
            Variables = [
                new("T_s", LawVariableType.Scalar, "K", "Surface temperature"),
                new("T_eff", LawVariableType.Scalar, "K", "Effective temperature"),
                new("tau", LawVariableType.Scalar, "dimensionless", "Infrared optical depth"),
            ],
            BoundaryConditionDefs = [
                new("GrayAtmosphere", "single tau value", "Grey gas approximation"),
                new("SingleLayer", "one atmospheric layer", "Simplified model"),
            ],
            ApplicableDomains = ["Climate science", "Atmospheric physics", "Global warming"],
            Reference = "Standard single-layer greenhouse model.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["greenhouse", "atmosphere", "optical depth", "warming"]
        });

        b.Add("climate.clausius-clapeyron-climate", new LawDefinition
        {
            Id = "climate.clausius-clapeyron-climate",
            Name = "Clausius-Clapeyron (Atmospheric Moisture)",
            Category = LawCategory.Climate,
            Expression = "de_s/dT = L_v * e_s / (R_v * T^2)",
            Description = "Saturation vapor pressure increases exponentially with temperature.",
            Parameters = [
                new("L_v", 2.5e6, "J/kg", 0, 1e7, "Latent heat of vaporization"),
                new("R_v", 461.5, "J/(kg*K)", 0, 1e3, "Gas constant for water vapor"),
            ],
            Variables = [
                new("e_s", LawVariableType.Scalar, "Pa", "Saturation vapor pressure"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("L_v", LawVariableType.Scalar, "J/kg", "Latent heat"),
                new("R_v", LawVariableType.Scalar, "J/(kg*K)", "Water vapor gas constant"),
            ],
            BoundaryConditionDefs = [
                new("Saturation", "relative humidity = 100%", "At saturation"),
            ],
            ApplicableDomains = ["Climate science", "Meteorology", "Hydrology"],
            Reference = "Clausius, R. (1850) / applied to atmospheric moisture.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["clausius-clapeyron", "moisture", "vapor pressure", "climate"]
        });
    }

    // ==============================================================
    //  NEUROSCIENCE (3 laws)
    // ==============================================================
    private static void RegisterNeuroscience(ImmutableDictionary<string, LawDefinition>.Builder b)
    {
        b.Add("neuro.nernst-potential", new LawDefinition
        {
            Id = "neuro.nernst-potential",
            Name = "Nernst Equation (Neural Membrane)",
            Category = LawCategory.Neuroscience,
            Expression = "E_ion = (RT/(z*F)) * ln([ion]_out / [ion]_in)",
            Description = "Equilibrium potential for a single ion across a neural membrane.",
            Parameters = [
                new("R", 8.31446, "J/(mol*K)", 0, 100, "Gas constant"),
                new("F", 96485.33212, "C/mol", 0, 1e5, "Faraday constant"),
            ],
            Variables = [
                new("E_ion", LawVariableType.Scalar, "V", "Equilibrium potential"),
                new("z", LawVariableType.Scalar, "dimensionless", "Ion charge (e.g., +1 for Na+)"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("out", LawVariableType.Scalar, "mol/L", "Extracellular concentration"),
                new("in", LawVariableType.Scalar, "mol/L", "Intracellular concentration"),
            ],
            BoundaryConditionDefs = [
                new("SingleIon", "only one ion species", "Or use GHK for multiple"),
                new("Equilibrium", "net current = 0", "At equilibrium"),
            ],
            ApplicableDomains = ["Neuroscience", "Electrophysiology", "Ion channels"],
            Reference = "Nernst, W. (1889) / applied to neural membranes.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["nernst", "membrane potential", "ion", "equilibrium"]
        });

        b.Add("neuro.goldman", new LawDefinition
        {
            Id = "neuro.goldman",
            Name = "Goldman-Hodgkin-Katz (GHK) Voltage Equation",
            Category = LawCategory.Neuroscience,
            Expression = "V_m = (RT/F) * ln((P_K[K]_out + P_Na[Na]_out + P_Cl[Cl]_in) / (P_K[K]_in + P_Na[Na]_in + P_Cl[Cl]_out))",
            Description = "Membrane potential from weighted permeabilities and concentrations of major ions.",
            Parameters = [
                new("R", 8.31446, "J/(mol*K)", 0, 100, "Gas constant"),
                new("F", 96485.33212, "C/mol", 0, 1e5, "Faraday constant"),
            ],
            Variables = [
                new("V_m", LawVariableType.Scalar, "V", "Membrane potential"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("P_K", LawVariableType.Scalar, "cm/s", "Potassium permeability"),
                new("P_Na", LawVariableType.Scalar, "cm/s", "Sodium permeability"),
                new("P_Cl", LawVariableType.Scalar, "cm/s", "Chloride permeability"),
            ],
            BoundaryConditionDefs = [
                new("SteadyState", "constant permeabilities", "Or use GHK current equation"),
                new("PlanarMembrane", "flat membrane", "No curvature effects"),
            ],
            ApplicableDomains = ["Neuroscience", "Electrophysiology", "Cardiac modeling"],
            Reference = "Goldman, D.E. (1943) / Hodgkin, A.L. & Katz, B. (1949).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["goldman", "GHK", "membrane potential", "permeability"]
        });

        b.Add("neuro.fitts-law", new LawDefinition
        {
            Id = "neuro.fitts-law",
            Name = "Fitts' Law (Motor Control)",
            Category = LawCategory.Neuroscience,
            Expression = "MT = a + b * log2(2*D/W)",
            Description = "Movement time to a target depends on distance and target width.",
            Parameters = [
                new("a", 0.05, "s", 0, 1, "Intercept (reaction time component)"),
                new("b", 0.1, "s/bit", 0, 1, "Slope (information processing rate)"),
            ],
            Variables = [
                new("MT", LawVariableType.Scalar, "s", "Movement time"),
                new("a", LawVariableType.Scalar, "s", "Intercept"),
                new("b", LawVariableType.Scalar, "s/bit", "Slope"),
                new("D", LawVariableType.Scalar, "m", "Distance to target"),
                new("W", LawVariableType.Scalar, "m", "Target width"),
            ],
            BoundaryConditionDefs = [
                new("DiscreteMovement", "point-to-point", "Not continuous tracking"),
                new("VisualTarget", "visually identified", "Or proprioceptive"),
            ],
            ApplicableDomains = ["Motor control", "Ergonomics", "User interface design"],
            Reference = "Fitts, P.M. (1954). J. Experimental Psychology.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["fitts", "motor control", "speed-accuracy", "HCI"]
        });
    }
}

// ============================================================
//  SECTION 3 — LawFactory
// ============================================================

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

// ============================================================
//  SECTION 4 — LawRegistry (thread-safe, versioned, importable)
// ============================================================

/// <summary>
/// Thread-safe, versioned registry for runtime law management.
/// Supports registration, unregistration, versioning, search, and import/export.
/// </summary>
public sealed class LawRegistry
{
    private readonly ConcurrentDictionary<string, LawDefinition> _registry = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, int> _versionMap = new(StringComparer.OrdinalIgnoreCase);
    private int _nextVersion;
    private readonly object _lock = new();

    /// <summary>Total laws in this registry instance.</summary>
    public int Count => _registry.Count;

    /// <summary>Raised when a law is registered or unregistered.</summary>
    public event Action<string, LawChangeType>? LawChanged;

    /// <summary>Type of change event.</summary>
    public enum LawChangeType { Registered, Unregistered, Updated }

    public LawRegistry()
    {
        _nextVersion = 1;
    }

    /// <summary>Loads all built-in laws from the static library into this registry.</summary>
    public void LoadDefaults()
    {
        foreach (var law in LawLibraryRegistry.GetAll())
        {
            _registry[law.Id] = law;
            _versionMap[law.Id] = _nextVersion++;
        }
    }

    /// <summary>Registers a new law. Throws if id already exists.</summary>
    public LawDefinition Register(LawDefinition law)
    {
        ArgumentNullException.ThrowIfNull(law);
        ArgumentException.ThrowIfNullOrWhiteSpace(law.Id);

        var registered = law with { Version = $"1.{Interlocked.Increment(ref _nextVersion)}.0" };
        if (!_registry.TryAdd(law.Id, registered))
            throw new InvalidOperationException($"Law '{law.Id}' is already registered. Use Update instead.");

        _versionMap[law.Id] = _nextVersion;
        LawChanged?.Invoke(law.Id, LawChangeType.Registered);
        return registered;
    }

    /// <summary>Updates an existing law, incrementing its version.</summary>
    public LawDefinition Update(LawDefinition law)
    {
        ArgumentNullException.ThrowIfNull(law);
        ArgumentException.ThrowIfNullOrWhiteSpace(law.Id);

        var existing = _registry.GetValueOrDefault(law.Id)
            ?? throw new KeyNotFoundException($"Law '{law.Id}' not found.");

        var parts = existing.Version.Split('.');
        var major = int.TryParse(parts[0], out var m) ? m : 1;
        var updated = law with { Version = $"{major}.{Interlocked.Increment(ref _nextVersion)}.0" };

        _registry[law.Id] = updated;
        _versionMap[law.Id] = _nextVersion;
        LawChanged?.Invoke(law.Id, LawChangeType.Updated);
        return updated;
    }

    /// <summary>Unregisters a law by id. Returns true if it existed.</summary>
    public bool Unregister(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        if (_registry.TryRemove(id, out _))
        {
            _versionMap.TryRemove(id, out _);
            LawChanged?.Invoke(id, LawChangeType.Unregistered);
            return true;
        }
        return false;
    }

    /// <summary>Gets a law by id, or null.</summary>
    public LawDefinition? Get(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        return _registry.GetValueOrDefault(id);
    }

    /// <summary>Returns all registered laws.</summary>
    public IReadOnlyCollection<LawDefinition> GetAll() => _registry.Values.ToList().AsReadOnly();

    /// <summary>Searches by name, id, expression, or description.</summary>
    public IEnumerable<LawDefinition> Search(string query)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        return _registry.Values.Where(l =>
            l.Id.Contains(query, StringComparison.OrdinalIgnoreCase) ||
            l.Name.Contains(query, StringComparison.OrdinalIgnoreCase) ||
            l.Expression.Contains(query, StringComparison.OrdinalIgnoreCase) ||
            l.Description.Contains(query, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>Returns laws matching the given category.</summary>
    public IEnumerable<LawDefinition> ByCategory(LawCategory category) =>
        _registry.Values.Where(l => l.Category == category);

    /// <summary>Returns laws containing the given tag.</summary>
    public IEnumerable<LawDefinition> ByTag(string tag) =>
        _registry.Values.Where(l => l.Tags.Contains(tag, StringComparer.OrdinalIgnoreCase));

    /// <summary>Returns the version number for a registered law.</summary>
    public int GetVersion(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        return _versionMap.GetValueOrDefault(id);
    }

    /// <summary>Checks if a law id is registered.</summary>
    public bool Contains(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        return _registry.ContainsKey(id);
    }

    /// <summary>Exports all laws as a JSON string.</summary>
    public string ExportJson()
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };
        var laws = _registry.Values.OrderBy(l => l.Id).ToList();
        return JsonSerializer.Serialize(laws, options);
    }

    /// <summary>Imports laws from a JSON string. Returns the count of laws imported.</summary>
    public int ImportJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        var laws = JsonSerializer.Deserialize<List<LawDefinition>>(json);
        if (laws == null)
            return 0;

        int count = 0;
        foreach (var law in laws)
        {
            if (string.IsNullOrWhiteSpace(law.Id))
                continue;
            if (_registry.TryAdd(law.Id, law))
            {
                _versionMap[law.Id] = _nextVersion;
                count++;
                LawChanged?.Invoke(law.Id, LawChangeType.Registered);
            }
        }
        return count;
    }

    /// <summary>Merges laws from another registry, skipping duplicates.</summary>
    public int MergeFrom(LawRegistry other)
    {
        ArgumentNullException.ThrowIfNull(other);
        int count = 0;
        foreach (var kvp in other._registry)
        {
            if (_registry.TryAdd(kvp.Key, kvp.Value))
            {
                _versionMap[kvp.Key] = _nextVersion;
                count++;
                LawChanged?.Invoke(kvp.Key, LawChangeType.Registered);
            }
        }
        return count;
    }

    /// <summary>Removes all laws from the registry.</summary>
    public void Clear()
    {
        var ids = _registry.Keys.ToList();
        _registry.Clear();
        _versionMap.Clear();
        foreach (var id in ids)
            LawChanged?.Invoke(id, LawChangeType.Unregistered);
    }

    /// <summary>Returns all registered law ids.</summary>
    public IEnumerable<string> AllIds() => _registry.Keys;
}

// ============================================================
//  SECTION 5 — Additional Law Definitions (Extended Library)
// ============================================================

/// <summary>
/// Extended law catalog: additional laws across all categories
/// to reach comprehensive coverage.
/// </summary>
public static class LawLibraryRegistryExtended
{
    /// <summary>Returns all additional laws for registration.</summary>
    public static IEnumerable<LawDefinition> GetAdditionalLaws()
    {
        // ── ADDITIONAL MECHANICS ──────────────────────────────────
        yield return new LawDefinition
        {
            Id = "mechanics.euler-equations",
            Name = "Euler's Equations (Rigid Body Rotation)",
            Category = LawCategory.Mechanics,
            Expression = "I1*d(omega1)/dt = (I2-I3)*omega2*omega3 + T1",
            Description = "Rotational equations of motion for a rigid body with principal moments of inertia.",
            Parameters = [],
            Variables = [
                new("I1", LawVariableType.Scalar, "kg*m^2", "Principal moment of inertia 1"),
                new("I2", LawVariableType.Scalar, "kg*m^2", "Principal moment of inertia 2"),
                new("I3", LawVariableType.Scalar, "kg*m^2", "Principal moment of inertia 3"),
                new("omega1", LawVariableType.Scalar, "rad/s", "Angular velocity component 1"),
                new("omega2", LawVariableType.Scalar, "rad/s", "Angular velocity component 2"),
                new("omega3", LawVariableType.Scalar, "rad/s", "Angular velocity component 3"),
                new("T1", LawVariableType.Scalar, "N*m", "Torque component 1"),
            ],
            BoundaryConditionDefs = [
                new("RigidBody", "body is rigid", "No deformation"),
                new("PrincipalAxes", "diagonal inertia tensor", "Aligned with principal axes"),
            ],
            ApplicableDomains = ["Rigid body dynamics", "Spacecraft attitude", "Gyroscopes"],
            Reference = "Euler, L. (1758). Theoria Motus Corporum Solidorum.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["euler", "rigid body", "rotation", "angular velocity"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.kinetic-energy-rotational",
            Name = "Rotational Kinetic Energy",
            Category = LawCategory.Mechanics,
            Expression = "K_rot = 0.5 * I * omega^2",
            Description = "Kinetic energy of a rotating rigid body.",
            Parameters = [],
            Variables = [
                new("K_rot", LawVariableType.Scalar, "J", "Rotational kinetic energy"),
                new("I", LawVariableType.Scalar, "kg*m^2", "Moment of inertia"),
                new("omega", LawVariableType.Scalar, "rad/s", "Angular velocity"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Rotational mechanics", "Flywheel design", "Sports physics"],
            Reference = "Standard result in classical mechanics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["kinetic energy", "rotation", "moment of inertia"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.parallel-axis",
            Name = "Parallel Axis Theorem",
            Category = LawCategory.Mechanics,
            Expression = "I = I_cm + m * d^2",
            Description = "Moment of inertia about any axis from the center-of-mass value.",
            Parameters = [],
            Variables = [
                new("I", LawVariableType.Scalar, "kg*m^2", "Moment of inertia about new axis"),
                new("I_cm", LawVariableType.Scalar, "kg*m^2", "Moment about center of mass"),
                new("m", LawVariableType.Scalar, "kg", "Total mass"),
                new("d", LawVariableType.Scalar, "m", "Distance between axes"),
            ],
            BoundaryConditionDefs = [
                new("PlanarObject", "flat body in plane", "Or 3D generalization"),
            ],
            ApplicableDomains = ["Rigid body mechanics", "Structural engineering"],
            Reference = "Steiner, J. (1834).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["parallel axis", "moment of inertia", "Steiner"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.perpendicular-axis",
            Name = "Perpendicular Axis Theorem (Lamina)",
            Category = LawCategory.Mechanics,
            Expression = "I_z = I_x + I_y",
            Description = "For a planar lamina, the perpendicular moment equals the sum of two in-plane moments.",
            Parameters = [],
            Variables = [
                new("I_z", LawVariableType.Scalar, "kg*m^2", "Moment about axis perpendicular to plane"),
                new("I_x", LawVariableType.Scalar, "kg*m^2", "Moment about x-axis"),
                new("I_y", LawVariableType.Scalar, "kg*m^2", "Moment about y-axis"),
            ],
            BoundaryConditionDefs = [
                new("PlanarLamina", "flat object", "Object must be planar"),
            ],
            ApplicableDomains = ["Rigid body mechanics", "Thin plate analysis"],
            Reference = "Standard result in classical mechanics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["perpendicular axis", "lamina", "moment of inertia"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.gravitational-pe-universal",
            Name = "Universal Gravitational Potential Energy",
            Category = LawCategory.Mechanics,
            Expression = "U = -G*m1*m2/r",
            Description = "Potential energy of two point masses separated by distance r.",
            Parameters = [
                new("G", 6.674e-11, "N*m^2/kg^2", 0, 1e-5, "Gravitational constant"),
            ],
            Variables = [
                new("U", LawVariableType.Scalar, "J", "Gravitational potential energy"),
                new("G", LawVariableType.Scalar, "N*m^2/kg^2", "Gravitational constant"),
                new("m1", LawVariableType.Scalar, "kg", "Mass 1"),
                new("m2", LawVariableType.Scalar, "kg", "Mass 2"),
                new("r", LawVariableType.Scalar, "m", "Distance between centers"),
            ],
            BoundaryConditionDefs = [
                new("PointMasses", "spherically symmetric", "Point masses"),
                new("ZeroAtInfinity", "U -> 0 as r -> inf", "Reference at infinity"),
            ],
            ApplicableDomains = ["Orbital mechanics", "Astrophysics", "Gravitational binding"],
            Reference = "Newton, I. (1687). Principia Mathematica.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["gravitational", "potential energy", "universal", "binding energy"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.kinetic-energy-translation",
            Name = "Translational Kinetic Energy",
            Category = LawCategory.Mechanics,
            Expression = "K = 0.5 * m * v^2",
            Description = "Kinetic energy of a mass moving with velocity v.",
            Parameters = [],
            Variables = [
                new("K", LawVariableType.Scalar, "J", "Kinetic energy"),
                new("m", LawVariableType.Scalar, "kg", "Mass"),
                new("v", LawVariableType.Scalar, "m/s", "Speed"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Classical mechanics", "Engineering", "Collisions"],
            Reference = "Leibniz, G.W. (1686). Vis Viva.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["kinetic energy", "translation", "velocity"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.power-mechanical",
            Name = "Mechanical Power",
            Category = LawCategory.Mechanics,
            Expression = "P = F * v = dW/dt",
            Description = "Rate of doing work; force times velocity.",
            Parameters = [],
            Variables = [
                new("P", LawVariableType.Scalar, "W", "Power"),
                new("F", LawVariableType.Vector, "N", "Force"),
                new("v", LawVariableType.Vector, "m/s", "Velocity"),
                new("W", LawVariableType.Scalar, "J", "Work"),
                new("t", LawVariableType.Scalar, "s", "Time"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Mechanical engineering", "Motors", "Vehicle dynamics"],
            Reference = "Standard result in mechanics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["power", "work", "force", "velocity"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.stress-strain",
            Name = "Engineering Stress-Strain (Hooke's Law, 3D)",
            Category = LawCategory.Mechanics,
            Expression = "sigma_ij = C_ijkl * epsilon_kl",
            Description = "Generalized Hooke's law relating stress and strain tensors via stiffness tensor.",
            Parameters = [],
            Variables = [
                new("sigma_ij", LawVariableType.Tensor, "Pa", "Stress tensor"),
                new("C_ijkl", LawVariableType.Tensor, "Pa", "Stiffness tensor"),
                new("epsilon_kl", LawVariableType.Tensor, "dimensionless", "Strain tensor"),
            ],
            BoundaryConditionDefs = [
                new("LinearElastic", "small strains", "Within elastic regime"),
                new("IsotropicMaterial", "C reduces to 2 constants", "Or use full anisotropic C"),
            ],
            ApplicableDomains = ["Elasticity", "Structural mechanics", "Materials science"],
            Reference = "Cauchy, A.L. (1828) / generalized by Green, G. (1839).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["stress", "strain", "elasticity", "tensor", "Hooke"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.poisson-ratio",
            Name = "Poisson's Ratio",
            Category = LawCategory.Mechanics,
            Expression = "nu = -epsilon_lat / epsilon_long",
            Description = "Ratio of lateral to axial strain under uniaxial loading.",
            Parameters = [],
            Variables = [
                new("nu", LawVariableType.Scalar, "dimensionless", "Poisson's ratio"),
                new("epsilon_lat", LawVariableType.Scalar, "dimensionless", "Lateral strain"),
                new("epsilon_long", LawVariableType.Scalar, "dimensionless", "Axial strain"),
            ],
            BoundaryConditionDefs = [
                new("LinearElastic", "within elastic range", "Small deformation"),
                new("IsotropicMaterial", "isotropic material", "Different for anisotropic"),
            ],
            ApplicableDomains = ["Materials science", "Structural engineering", "Geomechanics"],
            Reference = "Poisson, S.D. (1827).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["poisson", "ratio", "strain", "lateral", "elasticity"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.bulk-modulus",
            Name = "Bulk Modulus",
            Category = LawCategory.Mechanics,
            Expression = "K = -V * dP/dV",
            Description = "Resistance of a substance to uniform compression.",
            Parameters = [],
            Variables = [
                new("K", LawVariableType.Scalar, "Pa", "Bulk modulus"),
                new("V", LawVariableType.Scalar, "m^3", "Volume"),
                new("P", LawVariableType.Scalar, "Pa", "Pressure"),
            ],
            BoundaryConditionDefs = [
                new("HydrostaticStress", "equal pressure all sides", "Uniform compression"),
            ],
            ApplicableDomains = ["Materials science", "Geophysics", "Acoustics"],
            Reference = "Standard elasticity constant.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["bulk modulus", "compressibility", "pressure", "volume"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.shear-modulus",
            Name = "Shear Modulus",
            Category = LawCategory.Mechanics,
            Expression = "G = tau / gamma",
            Description = "Ratio of shear stress to shear strain.",
            Parameters = [],
            Variables = [
                new("G", LawVariableType.Scalar, "Pa", "Shear modulus"),
                new("tau", LawVariableType.Scalar, "Pa", "Shear stress"),
                new("gamma", LawVariableType.Scalar, "dimensionless", "Shear strain"),
            ],
            BoundaryConditionDefs = [
                new("SmallStrain", "gamma << 1", "Linear elastic regime"),
            ],
            ApplicableDomains = ["Structural engineering", "Materials science", "Seismology"],
            Reference = "Standard elasticity constant.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["shear modulus", "shear stress", "rigidity"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.youngs-modulus",
            Name = "Young's Modulus (Tensile)",
            Category = LawCategory.Mechanics,
            Expression = "E = sigma / epsilon",
            Description = "Ratio of tensile stress to tensile strain; measure of stiffness.",
            Parameters = [],
            Variables = [
                new("E", LawVariableType.Scalar, "Pa", "Young's modulus"),
                new("sigma", LawVariableType.Scalar, "Pa", "Tensile stress"),
                new("epsilon", LawVariableType.Scalar, "dimensionless", "Tensile strain"),
            ],
            BoundaryConditionDefs = [
                new("LinearElastic", "within proportional limit", "Hooke's law applies"),
                new("UniaxialLoading", "stress along one axis", "Simple tension"),
            ],
            ApplicableDomains = ["Materials science", "Structural engineering", "Biomechanics"],
            Reference = "Young, T. (1807). A Course of Lectures on Natural Philosophy.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["young's modulus", "stiffness", "tension", "elasticity"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.strain-energy-density",
            Name = "Elastic Strain Energy Density",
            Category = LawCategory.Mechanics,
            Expression = "u = 0.5 * sigma * epsilon = 0.5 * E * epsilon^2",
            Description = "Energy stored per unit volume in a linear elastic material.",
            Parameters = [],
            Variables = [
                new("u", LawVariableType.Scalar, "J/m^3", "Strain energy density"),
                new("sigma", LawVariableType.Scalar, "Pa", "Stress"),
                new("epsilon", LawVariableType.Scalar, "dimensionless", "Strain"),
                new("E", LawVariableType.Scalar, "Pa", "Young's modulus"),
            ],
            BoundaryConditionDefs = [
                new("LinearElastic", "Hooke's law valid", "Within elastic limit"),
            ],
            ApplicableDomains = ["Elasticity", "Structural mechanics", "Energy methods"],
            Reference = "Standard result in elasticity theory.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["strain energy", "elastic energy", "energy density"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.viscosity-newtonian",
            Name = "Newtonian Viscosity (Newton's Law of Viscosity)",
            Category = LawCategory.Mechanics,
            Expression = "tau = mu * du/dy",
            Description = "Shear stress proportional to velocity gradient in a Newtonian fluid.",
            Parameters = [new("mu", 1.0e-3, "Pa*s", 0, 1e6, "Dynamic viscosity")],
            Variables = [
                new("tau", LawVariableType.Scalar, "Pa", "Shear stress"),
                new("mu", LawVariableType.Scalar, "Pa*s", "Dynamic viscosity"),
                new("du/dy", LawVariableType.Scalar, "s^-1", "Velocity gradient"),
            ],
            BoundaryConditionDefs = [
                new("NewtonianFluid", "constant viscosity", "Viscosity independent of shear rate"),
            ],
            ApplicableDomains = ["Fluid mechanics", "Lubrication", "Polymer processing"],
            Reference = "Newton, I. (1687). Principia Mathematica.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["viscosity", "Newtonian", "shear stress", "velocity gradient"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.kinetic-friction",
            Name = "Coulomb Friction (Kinetic)",
            Category = LawCategory.Mechanics,
            Expression = "F_f = mu_k * N",
            Description = "Kinetic friction force proportional to normal force.",
            Parameters = [
                new("mu_k", 0.3, "dimensionless", 0, 2, "Coefficient of kinetic friction"),
            ],
            Variables = [
                new("F_f", LawVariableType.Scalar, "N", "Friction force"),
                new("mu_k", LawVariableType.Scalar, "dimensionless", "Kinetic friction coefficient"),
                new("N", LawVariableType.Scalar, "N", "Normal force"),
            ],
            BoundaryConditionDefs = [
                new("SlidingMotion", "surfaces in relative motion", "Kinetic friction"),
                new("DryFriction", "no lubrication", "Solid-solid contact"),
            ],
            ApplicableDomains = ["Tribology", "Braking systems", "Machinery"],
            Reference = "Coulomb, C.A. (1785). Theorie des Machines Simples.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["friction", "Coulomb", "kinetic", "normal force"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.stiction",
            Name = "Static Friction (Stiction)",
            Category = LawCategory.Mechanics,
            Expression = "F_f <= mu_s * N",
            Description = "Maximum static friction force before motion begins.",
            Parameters = [
                new("mu_s", 0.5, "dimensionless", 0, 2, "Coefficient of static friction"),
            ],
            Variables = [
                new("F_f", LawVariableType.Scalar, "N", "Static friction force"),
                new("mu_s", LawVariableType.Scalar, "dimensionless", "Static friction coefficient"),
                new("N", LawVariableType.Scalar, "N", "Normal force"),
            ],
            BoundaryConditionDefs = [
                new("NoSlip", "surfaces at rest relative", "Static contact"),
            ],
            ApplicableDomains = ["Tribology", "Braking", "Robotics", "Geophysics"],
            Reference = "Coulomb, C.A. (1785).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["static friction", "stiction", "slip", "adhesion"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.wind-loading",
            Name = "Wind Loading (Dynamic Pressure)",
            Category = LawCategory.Mechanics,
            Expression = "q = 0.5 * rho * v^2",
            Description = "Dynamic pressure exerted by wind on a structure.",
            Parameters = [
                new("rho", 1.225, "kg/m^3", 0, 2, "Air density at sea level"),
            ],
            Variables = [
                new("q", LawVariableType.Scalar, "Pa", "Dynamic pressure"),
                new("rho", LawVariableType.Scalar, "kg/m^3", "Air density"),
                new("v", LawVariableType.Scalar, "m/s", "Wind speed"),
            ],
            BoundaryConditionDefs = [
                new("Incompressible", "Ma < 0.3", "Subsonic wind"),
            ],
            ApplicableDomains = ["Structural engineering", "Building codes", "Wind energy"],
            Reference = "Standard aerodynamic result.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["wind", "dynamic pressure", "aerodynamic load"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.buckling-euler",
            Name = "Euler Column Buckling Load",
            Category = LawCategory.Mechanics,
            Expression = "P_cr = pi^2 * E * I / (K*L)^2",
            Description = "Critical load at which a slender column buckles under compression.",
            Parameters = [
                new("E", 200e9, "Pa", 0, 1e15, "Young's modulus"),
                new("I", 1e-4, "m^4", 0, 1e6, "Second moment of area"),
                new("K", 1.0, "dimensionless", 0.5, 2, "Effective length factor"),
            ],
            Variables = [
                new("P_cr", LawVariableType.Scalar, "N", "Critical buckling load"),
                new("E", LawVariableType.Scalar, "Pa", "Young's modulus"),
                new("I", LawVariableType.Scalar, "m^4", "Second moment of area"),
                new("K", LawVariableType.Scalar, "dimensionless", "Effective length factor"),
                new("L", LawVariableType.Scalar, "m", "Column length"),
            ],
            BoundaryConditionDefs = [
                new("SlenderColumn", "L/r >> 50", "Euler buckling valid"),
                new("CentricLoading", "no eccentricity", "Load through centroid"),
                new("LinearElastic", "sigma < sigma_y", "Elastic buckling"),
            ],
            ApplicableDomains = ["Structural engineering", "Civil engineering", "Mechanical design"],
            Reference = "Euler, L. (1744). Methodus Inveniendi Lineas Curvas.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["buckling", "Euler", "column", "critical load", "stability"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.bending-stress",
            Name = "Flexural Formula (Bending Stress)",
            Category = LawCategory.Mechanics,
            Expression = "sigma = -M*y / I",
            Description = "Bending stress at a distance y from the neutral axis.",
            Parameters = [],
            Variables = [
                new("sigma", LawVariableType.Scalar, "Pa", "Bending stress"),
                new("M", LawVariableType.Scalar, "N*m", "Bending moment"),
                new("y", LawVariableType.Scalar, "m", "Distance from neutral axis"),
                new("I", LawVariableType.Scalar, "m^4", "Second moment of area"),
            ],
            BoundaryConditionDefs = [
                new("PlaneSections", "sections remain plane", "Bernoulli-Navier hypothesis"),
                new("LinearElastic", "within elastic range", "Hooke's law"),
            ],
            ApplicableDomains = ["Structural engineering", "Beam design", "Bridge engineering"],
            Reference = "Standard result from Euler-Bernoulli beam theory.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["bending", "stress", "moment", "neutral axis", "beam"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.shear-stress-beam",
            Name = "Transverse Shear Stress in Beams",
            Category = LawCategory.Mechanics,
            Expression = "tau = V*Q / (I*t)",
            Description = "Shear stress distribution in a beam cross-section.",
            Parameters = [],
            Variables = [
                new("tau", LawVariableType.Scalar, "Pa", "Shear stress"),
                new("V", LawVariableType.Scalar, "N", "Shear force"),
                new("Q", LawVariableType.Scalar, "m^3", "First moment of area above point"),
                new("I", LawVariableType.Scalar, "m^4", "Second moment of area"),
                new("t", LawVariableType.Scalar, "m", "Width at the point"),
            ],
            BoundaryConditionDefs = [
                new("BeamTheory", "slender beam", "Euler-Bernoulli assumptions"),
            ],
            ApplicableDomains = ["Structural engineering", "Beam design"],
            Reference = "Standard result in strength of materials.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["shear stress", "beam", "transverse", "VQ/It"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.deflection-cantilever",
            Name = "Cantilever Beam Tip Deflection (Point Load)",
            Category = LawCategory.Mechanics,
            Expression = "delta = P*L^3 / (3*E*I)",
            Description = "Maximum deflection at the free end of a cantilever beam.",
            Parameters = [
                new("E", 200e9, "Pa", 0, 1e15, "Young's modulus"),
                new("I", 1e-4, "m^4", 0, 1e6, "Second moment of area"),
            ],
            Variables = [
                new("delta", LawVariableType.Scalar, "m", "Tip deflection"),
                new("P", LawVariableType.Scalar, "N", "Applied load"),
                new("L", LawVariableType.Scalar, "m", "Beam length"),
                new("E", LawVariableType.Scalar, "Pa", "Young's modulus"),
                new("I", LawVariableType.Scalar, "m^4", "Second moment of area"),
            ],
            BoundaryConditionDefs = [
                new("Cantilever", "fixed at one end", "Clamped boundary"),
                new("PointLoad", "single load at tip", "Concentrated load"),
                new("LinearElastic", "small deflection", "Linear theory"),
            ],
            ApplicableDomains = ["Structural engineering", "MEMS cantilevers", "Probes"],
            Reference = "Standard beam deflection formula.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["cantilever", "deflection", "beam", "tip load"]
        };

        yield return new LawDefinition
        {
            Id = "mechanics.deflection-simply-supported",
            Name = "Simply Supported Beam Center Deflection",
            Category = LawCategory.Mechanics,
            Expression = "delta = 5*P*L^3 / (384*E*I)",
            Description = "Maximum deflection at midspan of a simply supported beam under uniform load.",
            Parameters = [
                new("E", 200e9, "Pa", 0, 1e15, "Young's modulus"),
                new("I", 1e-4, "m^4", 0, 1e6, "Second moment of area"),
            ],
            Variables = [
                new("delta", LawVariableType.Scalar, "m", "Center deflection"),
                new("P", LawVariableType.Scalar, "N", "Total applied load"),
                new("L", LawVariableType.Scalar, "m", "Beam length"),
                new("E", LawVariableType.Scalar, "Pa", "Young's modulus"),
                new("I", LawVariableType.Scalar, "m^4", "Second moment of area"),
            ],
            BoundaryConditionDefs = [
                new("SimplySupported", "pinned at both ends", "Hinged supports"),
                new("UniformLoad", "distributed load", "wL = P"),
                new("LinearElastic", "small deflection", "Linear theory"),
            ],
            ApplicableDomains = ["Structural engineering", "Floor design", "Bridges"],
            Reference = "Standard beam deflection formula.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["simply supported", "deflection", "beam", "uniform load"]
        };
    }

    // ── ADDITIONAL THERMODYNAMICS ────────────────────────────────
    /// <summary>Returns additional thermodynamics laws.</summary>
    public static IEnumerable<LawDefinition> GetAdditionalThermodynamicsLaws()
    {
        yield return new LawDefinition
        {
            Id = "thermo.entropy-change-ideal-gas",
            Name = "Entropy Change of Ideal Gas",
            Category = LawCategory.Thermodynamics,
            Expression = "Delta_S = n*Cv*ln(T2/T1) + n*R*ln(V2/V1)",
            Description = "Entropy change for an ideal gas process.",
            Parameters = [
                new("R", 8.31446, "J/(mol*K)", 0, 100, "Gas constant"),
            ],
            Variables = [
                new("Delta_S", LawVariableType.Scalar, "J/K", "Entropy change"),
                new("n", LawVariableType.Scalar, "mol", "Amount of substance"),
                new("Cv", LawVariableType.Scalar, "J/(mol*K)", "Molar heat capacity at constant volume"),
                new("T1", LawVariableType.Scalar, "K", "Initial temperature"),
                new("T2", LawVariableType.Scalar, "K", "Final temperature"),
                new("V1", LawVariableType.Scalar, "m^3", "Initial volume"),
                new("V2", LawVariableType.Scalar, "m^3", "Final volume"),
            ],
            BoundaryConditionDefs = [
                new("IdealGas", "ideal gas law valid", "PV = nRT"),
            ],
            ApplicableDomains = ["Thermodynamics", "Chemical engineering", "Refrigeration"],
            Reference = "Standard result from combining first and second laws.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["entropy", "ideal gas", "process", "temperature", "volume"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.entropy-irreversibility",
            Name = "Clausius Inequality",
            Category = LawCategory.Thermodynamics,
            Expression = "oint (delta Q / T) <= 0",
            Description = "Integral of heat over temperature around a cycle is non-positive.",
            Parameters = [],
            Variables = [
                new("Q", LawVariableType.Scalar, "J", "Heat transfer"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
            ],
            BoundaryConditionDefs = [
                new("ClosedCycle", "complete cycle", "Start and end at same state"),
            ],
            ApplicableDomains = ["Thermodynamics", "Heat engines", "Exergy analysis"],
            Reference = "Clausius, R. (1854).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Clausius", "inequality", "entropy", "cycle"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.efficiency-heat-engine",
            Name = "Heat Engine Efficiency",
            Category = LawCategory.Thermodynamics,
            Expression = "eta = W_net / Q_in = 1 - Q_out / Q_in",
            Description = "Efficiency of any heat engine from heat and work transfers.",
            Parameters = [],
            Variables = [
                new("eta", LawVariableType.Scalar, "dimensionless", "Efficiency"),
                new("W_net", LawVariableType.Scalar, "J", "Net work output"),
                new("Q_in", LawVariableType.Scalar, "J", "Heat input"),
                new("Q_out", LawVariableType.Scalar, "J", "Heat rejected"),
            ],
            BoundaryConditionDefs = [
                new("SteadyState", "continuous operation", "Cyclic process"),
            ],
            ApplicableDomains = ["Heat engines", "Power plants", "Automotive"],
            Reference = "Standard thermodynamic result.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["efficiency", "heat engine", "work", "heat input"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.cop-refrigerator",
            Name = "Coefficient of Performance (Refrigerator)",
            Category = LawCategory.Thermodynamics,
            Expression = "COP_R = Q_c / W_in = Q_c / (Q_h - Q_c)",
            Description = "Performance measure for a refrigeration cycle.",
            Parameters = [],
            Variables = [
                new("COP_R", LawVariableType.Scalar, "dimensionless", "Coefficient of performance"),
                new("Q_c", LawVariableType.Scalar, "J", "Heat removed from cold reservoir"),
                new("Q_h", LawVariableType.Scalar, "J", "Heat rejected to hot reservoir"),
                new("W_in", LawVariableType.Scalar, "J", "Work input"),
            ],
            BoundaryConditionDefs = [
                new("CyclicOperation", "steady-state cycle", "Continuous refrigeration"),
            ],
            ApplicableDomains = ["Refrigeration", "Air conditioning", "Heat pumps"],
            Reference = "Standard thermodynamic result.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["COP", "refrigerator", "heat pump", "cooling"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.cop-heat-pump",
            Name = "Coefficient of Performance (Heat Pump)",
            Category = LawCategory.Thermodynamics,
            Expression = "COP_HP = Q_h / W_in = Q_h / (Q_h - Q_c)",
            Description = "Performance measure for a heat pump cycle.",
            Parameters = [],
            Variables = [
                new("COP_HP", LawVariableType.Scalar, "dimensionless", "Coefficient of performance"),
                new("Q_h", LawVariableType.Scalar, "J", "Heat delivered to hot space"),
                new("Q_c", LawVariableType.Scalar, "J", "Heat absorbed from cold source"),
                new("W_in", LawVariableType.Scalar, "J", "Work input"),
            ],
            BoundaryConditionDefs = [
                new("CyclicOperation", "steady-state cycle", "Continuous heating"),
                new("Relation", "COP_HP = COP_R + 1", "Linked to refrigerator COP"),
            ],
            ApplicableDomains = ["Heat pumps", "HVAC", "Geothermal heating"],
            Reference = "Standard thermodynamic result.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["COP", "heat pump", "heating", "efficiency"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.joule-thomson",
            Name = "Joule-Thomson Coefficient",
            Category = LawCategory.Thermodynamics,
            Expression = "mu_JT = (dT/dP)_H = (1/C_p) * [T*(dV/dT)_P - V]",
            Description = "Temperature change of a gas upon throttling at constant enthalpy.",
            Parameters = [],
            Variables = [
                new("mu_JT", LawVariableType.Scalar, "K/Pa", "Joule-Thomson coefficient"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("P", LawVariableType.Scalar, "Pa", "Pressure"),
                new("V", LawVariableType.Scalar, "m^3", "Volume"),
                new("C_p", LawVariableType.Scalar, "J/(mol*K)", "Molar heat capacity at constant P"),
            ],
            BoundaryConditionDefs = [
                new("Isenthalpic", "H = const", "Throttling process"),
                new("SteadyFlow", "open system", "Flow through valve or porous plug"),
            ],
            ApplicableDomains = ["Refrigeration", "Gas liquefaction", "Cryogenics"],
            Reference = "Joule, J.P. & Thomson, W. (1852).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Joule-Thomson", "throttling", "enthalpy", "gas cooling"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.isentropic-process",
            Name = "Isentropic Process Relations (Ideal Gas)",
            Category = LawCategory.Thermodynamics,
            Expression = "T*V^(gamma-1) = const; P*V^gamma = const; T*P^((1-gamma)/gamma) = const",
            Description = "State relations for reversible adiabatic (isentropic) processes.",
            Parameters = [],
            Variables = [
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("V", LawVariableType.Scalar, "m^3", "Volume"),
                new("P", LawVariableType.Scalar, "Pa", "Pressure"),
                new("gamma", LawVariableType.Scalar, "dimensionless", "Heat capacity ratio Cp/Cv"),
            ],
            BoundaryConditionDefs = [
                new("Reversible", "no entropy generation", "Quasi-static"),
                new("Adiabatic", "Q = 0", "No heat transfer"),
            ],
            ApplicableDomains = ["Compressor design", "Turbine analysis", "Gas dynamics"],
            Reference = "Standard result for isentropic ideal gas processes.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["isentropic", "adiabatic", "reversible", "compression"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.polytropic-process",
            Name = "Polytropic Process",
            Category = LawCategory.Thermodynamics,
            Expression = "P*V^n = const",
            Description = "Generalized gas process with polytropic index n.",
            Parameters = [],
            Variables = [
                new("P", LawVariableType.Scalar, "Pa", "Pressure"),
                new("V", LawVariableType.Scalar, "m^3", "Volume"),
                new("n", LawVariableType.Scalar, "dimensionless", "Polytropic index"),
            ],
            BoundaryConditionDefs = [
                new("PolytropicIndex", "1 < n < gamma", "For real processes"),
                new("SpecialCases", "n=0 (const P), n=1 (isothermal), n=gamma (isentropic)", "Special cases"),
            ],
            ApplicableDomains = ["Compressor design", "Engine cycles", "Gas dynamics"],
            Reference = "Standard result in thermodynamics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["polytropic", "process", "compression", "expansion"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.dew-point",
            Name = "Dew Point Temperature",
            Category = LawCategory.Thermodynamics,
            Expression = "T_dew = T - (100 - RH) / 5  (approximate for air)",
            Description = "Temperature at which air becomes saturated and water condenses.",
            Parameters = [],
            Variables = [
                new("T_dew", LawVariableType.Scalar, "C", "Dew point temperature"),
                new("T", LawVariableType.Scalar, "C", "Air temperature"),
                new("RH", LawVariableType.Scalar, "%", "Relative humidity"),
            ],
            BoundaryConditionDefs = [
                new("AtmosphericAir", "low pressure", "Approximate formula"),
                new("LowHumidity", "RH < 100%", "Not saturated"),
            ],
            ApplicableDomains = ["Meteorology", "HVAC", "Condensation prevention"],
            Reference = "Magnus formula and approximations.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["dew point", "condensation", "humidity", "atmosphere"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.wien-displacement",
            Name = "Wien's Displacement Law",
            Category = LawCategory.Thermodynamics,
            Expression = "lambda_max * T = b = 2.898e-3 m*K",
            Description = "Wavelength of peak emission from a black body is inversely proportional to temperature.",
            Parameters = [
                new("b", 2.897771955e-3, "m*K", 0, 1, "Wien's displacement constant"),
            ],
            Variables = [
                new("lambda_max", LawVariableType.Scalar, "m", "Wavelength of peak emission"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("b", LawVariableType.Scalar, "m*K", "Wien's constant"),
            ],
            BoundaryConditionDefs = [
                new("BlackBody", "ideal black body", "Perfect emitter"),
            ],
            ApplicableDomains = ["Thermal radiation", "Astrophysics", "Pyrometry"],
            Reference = "Wien, W. (1893).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Wien", "displacement", "peak wavelength", "black body"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.kirchhoff-law",
            Name = "Kirchhoff's Law of Thermal Radiation",
            Category = LawCategory.Thermodynamics,
            Expression = "epsilon_lambda = alpha_lambda  (at thermal equilibrium)",
            Description = "At thermal equilibrium, emissivity equals absorptivity at each wavelength.",
            Parameters = [],
            Variables = [
                new("epsilon", LawVariableType.Scalar, "dimensionless", "Emissivity"),
                new("alpha", LawVariableType.Scalar, "dimensionless", "Absorptivity"),
                new("lambda", LawVariableType.Scalar, "m", "Wavelength"),
            ],
            BoundaryConditionDefs = [
                new("ThermalEquilibrium", "T = const", "Local thermodynamic equilibrium"),
            ],
            ApplicableDomains = ["Thermal radiation", "Solar energy", "Remote sensing"],
            Reference = "Kirchhoff, G. (1860). Monatsberichte der Berliner Akademie.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Kirchhoff", "emissivity", "absorptivity", "radiation"]
        };

        yield return new LawDefinition
        {
            Id = "thermo.rayleigh-jeans",
            Name = "Rayleigh-Jeans Law (Classical Radiation)",
            Category = LawCategory.Thermodynamics,
            Expression = "u(nu) = 8*pi*nu^2*k_B*T / c^3",
            Description = "Classical spectral energy density of black-body radiation; fails at high frequencies (ultraviolet catastrophe).",
            Parameters = [
                new("k_B", 1.380649e-23, "J/K", 0, 1e-15, "Boltzmann constant"),
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
            ],
            Variables = [
                new("u", LawVariableType.Scalar, "J/m^3", "Energy density per frequency"),
                new("nu", LawVariableType.Scalar, "Hz", "Frequency"),
                new("k_B", LawVariableType.Scalar, "J/K", "Boltzmann constant"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("c", LawVariableType.Scalar, "m/s", "Speed of light"),
            ],
            BoundaryConditionDefs = [
                new("ClassicalLimit", "h*nu << k_B*T", "Low frequency or high T"),
            ],
            ApplicableDomains = ["Thermal radiation", "Historical physics", "Planck derivation"],
            Reference = "Rayleigh, Lord (1900) / Jeans, J.H. (1905).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Rayleigh-Jeans", "classical", "ultraviolet catastrophe", "radiation"]
        };
    }

    // ── ADDITIONAL ELECTROMAGNETISM ──────────────────────────────
    /// <summary>Returns additional electromagnetism laws.</summary>
    public static IEnumerable<LawDefinition> GetAdditionalElectromagnetismLaws()
    {
        yield return new LawDefinition
        {
            Id = "em.inductance-mutual",
            Name = "Mutual Inductance",
            Category = LawCategory.Electromagnetism,
            Expression = "M_21 = Phi_21 / I_1 = L_12",
            Description = "Magnetic flux through coil 2 per unit current in coil 1.",
            Parameters = [],
            Variables = [
                new("M_21", LawVariableType.Scalar, "H", "Mutual inductance"),
                new("Phi_21", LawVariableType.Scalar, "Wb", "Flux through coil 2"),
                new("I_1", LawVariableType.Scalar, "A", "Current in coil 1"),
            ],
            BoundaryConditionDefs = [
                new("Reciprocity", "M_12 = M_21", "For linear media"),
            ],
            ApplicableDomains = ["Transformers", "Wireless charging", "Coupled circuits"],
            Reference = "Standard result from Faraday's law.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["mutual inductance", "transformer", "coupling"]
        };

        yield return new LawDefinition
        {
            Id = "em.capacitor-energy",
            Name = "Energy Stored in a Capacitor",
            Category = LawCategory.Electromagnetism,
            Expression = "U = 0.5 * C * V^2 = 0.5 * Q^2 / C",
            Description = "Electrostatic energy stored in a charged capacitor.",
            Parameters = [],
            Variables = [
                new("U", LawVariableType.Scalar, "J", "Stored energy"),
                new("C", LawVariableType.Scalar, "F", "Capacitance"),
                new("V", LawVariableType.Scalar, "V", "Voltage"),
                new("Q", LawVariableType.Scalar, "C", "Charge"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Circuit design", "Energy storage", "Pulse power"],
            Reference = "Standard result in electrostatics.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["capacitor", "energy", "stored", "electrostatic"]
        };

        yield return new LawDefinition
        {
            Id = "em.inductor-energy",
            Name = "Energy Stored in an Inductor",
            Category = LawCategory.Electromagnetism,
            Expression = "U = 0.5 * L * I^2",
            Description = "Magnetic energy stored in the field of an inductor.",
            Parameters = [],
            Variables = [
                new("U", LawVariableType.Scalar, "J", "Stored energy"),
                new("L", LawVariableType.Scalar, "H", "Inductance"),
                new("I", LawVariableType.Scalar, "A", "Current"),
            ],
            BoundaryConditionDefs = [],
            ApplicableDomains = ["Circuit design", "Energy storage", "Magnetic systems"],
            Reference = "Standard result in electromagnetism.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["inductor", "energy", "stored", "magnetic"]
        };

        yield return new LawDefinition
        {
            Id = "em.rc-time-constant",
            Name = "RC Time Constant",
            Category = LawCategory.Electromagnetism,
            Expression = "tau = R * C",
            Description = "Time constant for charging/discharging a capacitor through a resistor.",
            Parameters = [],
            Variables = [
                new("tau", LawVariableType.Scalar, "s", "Time constant"),
                new("R", LawVariableType.Scalar, "ohm", "Resistance"),
                new("C", LawVariableType.Scalar, "F", "Capacitance"),
            ],
            BoundaryConditionDefs = [
                new("FirstOrder", "single R and C", "Simple RC circuit"),
            ],
            ApplicableDomains = ["Circuit design", "Timing circuits", "Filters"],
            Reference = "Standard result in circuit theory.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["RC", "time constant", "capacitor", "resistor"]
        };

        yield return new LawDefinition
        {
            Id = "em.rl-time-constant",
            Name = "RL Time Constant",
            Category = LawCategory.Electromagnetism,
            Expression = "tau = L / R",
            Description = "Time constant for current buildup/decay in an inductor-resistor circuit.",
            Parameters = [],
            Variables = [
                new("tau", LawVariableType.Scalar, "s", "Time constant"),
                new("L", LawVariableType.Scalar, "H", "Inductance"),
                new("R", LawVariableType.Scalar, "ohm", "Resistance"),
            ],
            BoundaryConditionDefs = [
                new("FirstOrder", "single L and R", "Simple RL circuit"),
            ],
            ApplicableDomains = ["Circuit design", "Motor control", "Relay circuits"],
            Reference = "Standard result in circuit theory.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["RL", "time constant", "inductor", "resistor"]
        };

        yield return new LawDefinition
        {
            Id = "em.biot-savart",
            Name = "Biot-Savart Law",
            Category = LawCategory.Electromagnetism,
            Expression = "dB = (mu_0 / 4*pi) * (I dl x r_hat) / r^2",
            Description = "Magnetic field produced by a current-carrying wire element.",
            Parameters = [
                new("mu_0", 1.25663706212e-6, "H/m", 0, 1, "Vacuum permeability"),
            ],
            Variables = [
                new("dB", LawVariableType.Vector, "T", "Magnetic field element"),
                new("I", LawVariableType.Scalar, "A", "Current"),
                new("dl", LawVariableType.Vector, "m", "Wire element"),
                new("r", LawVariableType.Scalar, "m", "Distance"),
                new("r_hat", LawVariableType.Vector, "dimensionless", "Unit vector to point"),
            ],
            BoundaryConditionDefs = [
                new("Magnetostatics", "steady currents", "No time-varying fields"),
            ],
            ApplicableDomains = ["Electromagnetism", "Magnet design", "MRI systems"],
            Reference = "Biot, J.B. & Savart, F. (1820).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Biot-Savart", "magnetic field", "current", "wire"]
        };

        yield return new LawDefinition
        {
            Id = "em.leyden-jar",
            Name = "Capacitor Charging (Exponential)",
            Category = LawCategory.Electromagnetism,
            Expression = "V(t) = V0 * (1 - exp(-t/(R*C)))",
            Description = "Voltage across a capacitor charging through a resistor.",
            Parameters = [],
            Variables = [
                new("V", LawVariableType.Scalar, "V", "Voltage at time t"),
                new("V0", LawVariableType.Scalar, "V", "Supply voltage"),
                new("t", LawVariableType.Scalar, "s", "Time"),
                new("R", LawVariableType.Scalar, "ohm", "Resistance"),
                new("C", LawVariableType.Scalar, "F", "Capacitance"),
            ],
            BoundaryConditionDefs = [
                new("InitialCondition", "V(0) = 0", "Capacitor initially uncharged"),
            ],
            ApplicableDomains = ["Circuit analysis", "Timing circuits", "Pulse shaping"],
            Reference = "Standard first-order circuit result.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["capacitor", "charging", "exponential", "RC"]
        };

        yield return new LawDefinition
        {
            Id = "em.impedance-series-rlc",
            Name = "Impedance of Series RLC Circuit",
            Category = LawCategory.Electromagnetism,
            Expression = "Z = R + j*(omega*L - 1/(omega*C))",
            Description = "Complex impedance of a series RLC circuit.",
            Parameters = [],
            Variables = [
                new("Z", LawVariableType.Complex, "ohm", "Impedance"),
                new("R", LawVariableType.Scalar, "ohm", "Resistance"),
                new("omega", LawVariableType.Scalar, "rad/s", "Angular frequency"),
                new("L", LawVariableType.Scalar, "H", "Inductance"),
                new("C", LawVariableType.Scalar, "F", "Capacitance"),
            ],
            BoundaryConditionDefs = [
                new("LinearCircuit", "linear components", "No nonlinear effects"),
                new("SteadyStateAC", "sinusoidal steady state", "Phasor analysis"),
            ],
            ApplicableDomains = ["AC circuit analysis", "Filter design", "Resonance"],
            Reference = "Standard AC circuit theory.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["impedance", "RLC", "resonance", "AC circuit"]
        };

        yield return new LawDefinition
        {
            Id = "em.resonant-frequency-rlc",
            Name = "Resonant Frequency of RLC Circuit",
            Category = LawCategory.Electromagnetism,
            Expression = "f_0 = 1 / (2*pi*sqrt(L*C))",
            Description = "Frequency at which impedance is purely resistive (resonance).",
            Parameters = [],
            Variables = [
                new("f_0", LawVariableType.Scalar, "Hz", "Resonant frequency"),
                new("L", LawVariableType.Scalar, "H", "Inductance"),
                new("C", LawVariableType.Scalar, "F", "Capacitance"),
            ],
            BoundaryConditionDefs = [
                new("SeriesResonance", "X_L = X_C", "Inductive and capacitive reactances cancel"),
            ],
            ApplicableDomains = ["Radio tuning", "Filters", "Wireless communication"],
            Reference = "Standard result in AC circuit theory.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["resonance", "frequency", "RLC", "tuning"]
        };
    }
}

// ============================================================
//  SECTION 6 — Additional Extended Laws (reaching 250KB+)
// ============================================================

/// <summary>
/// Additional law definitions to extend the library.
/// </summary>
public static class LawLibraryRegistryExtended2
{
    /// <summary>Returns all additional laws.</summary>
    public static IEnumerable<LawDefinition> GetAllAdditionalLaws()
    {
        yield return new LawDefinition
        {
            Id = "fluid.bernoulli-compressible",
            Name = "Bernoulli's Equation (Compressible, Isentropic)",
            Category = LawCategory.FluidDynamics,
            Expression = "v^2/2 + gamma/(gamma-1)*P/rho = const",
            Description = "Bernoulli equation extended for compressible isentropic flow.",
            Parameters = [],
            Variables = [
                new("v", LawVariableType.Scalar, "m/s", "Flow velocity"),
                new("gamma", LawVariableType.Scalar, "dimensionless", "Heat capacity ratio"),
                new("P", LawVariableType.Scalar, "Pa", "Pressure"),
                new("rho", LawVariableType.Scalar, "kg/m^3", "Density"),
            ],
            BoundaryConditionDefs = [
                new("Isentropic", "s = const", "Reversible adiabatic"),
                new("SteadyFlow", "d/dt = 0", "Steady-state"),
            ],
            ApplicableDomains = ["Compressible flow", "Gas dynamics", "Nozzle design"],
            Reference = "Bernoulli, D. (1738); extended for compressibility.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Bernoulli", "compressible", "isentropic"]
        };

        yield return new LawDefinition
        {
            Id = "fluid.kelvin-helmholtz",
            Name = "Kelvin-Helmholtz Instability Criterion",
            Category = LawCategory.FluidDynamics,
            Expression = "g*(rho2-rho1)/(rho1+rho2) < 0.5*(v1-v2)^2",
            Description = "Condition for shear instability at the interface between two fluid layers.",
            Parameters = [
                new("g", 9.80665, "m/s^2", 0, 100, "Gravitational acceleration"),
            ],
            Variables = [
                new("rho1", LawVariableType.Scalar, "kg/m^3", "Density of fluid 1"),
                new("rho2", LawVariableType.Scalar, "kg/m^3", "Density of fluid 2"),
                new("v1", LawVariableType.Scalar, "m/s", "Velocity of fluid 1"),
                new("v2", LawVariableType.Scalar, "m/s", "Velocity of fluid 2"),
                new("g", LawVariableType.Scalar, "m/s^2", "Gravity"),
            ],
            BoundaryConditionDefs = [
                new("DensityInterface", "two distinct layers", "Sharp density gradient"),
                new("ShearFlow", "velocity difference", "Parallel flow"),
            ],
            ApplicableDomains = ["Atmospheric science", "Oceanography", "Astrophysics"],
            Reference = "Kelvin, Lord (1871); Helmholtz, H. (1868).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Kelvin-Helmholtz", "instability", "shear"]
        };

        yield return new LawDefinition
        {
            Id = "fluid.newtons-cooling",
            Name = "Newton's Law of Cooling (Convection)",
            Category = LawCategory.FluidDynamics,
            Expression = "q = h * (T_s - T_inf)",
            Description = "Convective heat transfer rate from a surface to a fluid.",
            Parameters = [new("h", 25.0, "W/(m^2*K)", 0, 1e5, "Heat transfer coefficient")],
            Variables = [
                new("q", LawVariableType.Scalar, "W/m^2", "Heat flux"),
                new("h", LawVariableType.Scalar, "W/(m^2*K)", "Heat transfer coefficient"),
                new("T_s", LawVariableType.Scalar, "K", "Surface temperature"),
                new("T_inf", LawVariableType.Scalar, "K", "Fluid temperature"),
            ],
            BoundaryConditionDefs = [
                new("LumpedAnalysis", "Bi < 0.1", "Uniform surface temperature"),
            ],
            ApplicableDomains = ["Heat transfer", "HVAC", "Electronics cooling"],
            Reference = "Newton, I. (1701). Scale of the Degrees of Heat.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["convection", "heat transfer", "Newton", "cooling"]
        };

        yield return new LawDefinition
        {
            Id = "qm.hydrogen-radius",
            Name = "Hydrogen Atom Bohr Radius",
            Category = LawCategory.QuantumMechanics,
            Expression = "a_0 = 4*pi*epsilon_0*hbar^2 / (m_e*e^2)",
            Description = "Most probable distance between the proton and electron in ground state.",
            Parameters = [
                new("epsilon_0", 8.8541878128e-12, "F/m", 0, 1e-6, "Vacuum permittivity"),
                new("hbar", 1.054571817e-34, "J*s", 0, 1e-30, "Reduced Planck constant"),
                new("m_e", 9.1093837015e-31, "kg", 0, 1e-25, "Electron mass"),
                new("e", 1.602176634e-19, "C", 0, 1e-10, "Elementary charge"),
            ],
            Variables = [
                new("a_0", LawVariableType.Scalar, "m", "Bohr radius"),
            ],
            BoundaryConditionDefs = [
                new("GroundState", "n = 1", "Lowest energy state"),
            ],
            ApplicableDomains = ["Atomic physics", "Quantum chemistry", "Spectroscopy"],
            Reference = "Bohr, N. (1913).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Bohr radius", "hydrogen", "atomic scale"]
        };

        yield return new LawDefinition
        {
            Id = "qm.fine-structure",
            Name = "Fine-Structure Constant",
            Category = LawCategory.QuantumMechanics,
            Expression = "alpha = e^2 / (4*pi*epsilon_0*hbar*c) ~ 1/137",
            Description = "Dimensionless coupling constant for electromagnetic interaction.",
            Parameters = [
                new("e", 1.602176634e-19, "C", 0, 1e-10, "Elementary charge"),
                new("epsilon_0", 8.8541878128e-12, "F/m", 0, 1e-6, "Vacuum permittivity"),
                new("hbar", 1.054571817e-34, "J*s", 0, 1e-30, "Reduced Planck constant"),
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
            ],
            Variables = [
                new("alpha", LawVariableType.Scalar, "dimensionless", "Fine-structure constant"),
            ],
            BoundaryConditionDefs = [
                new("FundamentalConstant", "no dependence on units", "Dimensionless"),
            ],
            ApplicableDomains = ["Quantum electrodynamics", "Atomic physics"],
            Reference = "Sommerfeld, A. (1916).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["fine structure", "alpha", "coupling constant"]
        };

        yield return new LawDefinition
        {
            Id = "qm.rydberg-formula",
            Name = "Rydberg Formula (Hydrogen Spectral Lines)",
            Category = LawCategory.QuantumMechanics,
            Expression = "1/lambda = R_inf * (1/n1^2 - 1/n2^2)",
            Description = "Wavelengths of spectral lines of the hydrogen atom.",
            Parameters = [
                new("R_inf", 1.0973731568e7, "m^-1", 0, 1e8, "Rydberg constant"),
            ],
            Variables = [
                new("lambda", LawVariableType.Scalar, "m", "Wavelength"),
                new("R_inf", LawVariableType.Scalar, "m^-1", "Rydberg constant"),
                new("n1", LawVariableType.Scalar, "dimensionless", "Lower energy level"),
                new("n2", LawVariableType.Scalar, "dimensionless", "Upper energy level"),
            ],
            BoundaryConditionDefs = [
                new("HydrogenLike", "single electron atoms", "Or hydrogen-like ions"),
            ],
            ApplicableDomains = ["Spectroscopy", "Astronomy", "Plasma diagnostics"],
            Reference = "Rydberg, J.R. (1888).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Rydberg", "spectral lines", "hydrogen", "wavelength"]
        };

        yield return new LawDefinition
        {
            Id = "qm.compton-scattering",
            Name = "Compton Scattering Wavelength Shift",
            Category = LawCategory.QuantumMechanics,
            Expression = "Delta_lambda = (h / (m_e * c)) * (1 - cos(theta))",
            Description = "Wavelength shift of X-ray photons scattered by electrons.",
            Parameters = [
                new("h", 6.62607015e-34, "J*s", 0, 1e-30, "Planck constant"),
                new("m_e", 9.1093837015e-31, "kg", 0, 1e-25, "Electron mass"),
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
            ],
            Variables = [
                new("Delta_lambda", LawVariableType.Scalar, "m", "Wavelength shift"),
                new("theta", LawVariableType.Scalar, "rad", "Scattering angle"),
            ],
            BoundaryConditionDefs = [
                new("FreeElectron", "electron initially at rest", "Stationary target"),
            ],
            ApplicableDomains = ["X-ray physics", "Medical imaging", "Material characterization"],
            Reference = "Compton, A.H. (1923). Physical Review.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Compton", "scattering", "X-ray", "wavelength shift"]
        };

        yield return new LawDefinition
        {
            Id = "chem.gibbs-free-energy",
            Name = "Gibbs Free Energy (Reaction Spontaneity)",
            Category = LawCategory.Chemistry,
            Expression = "Delta_G = Delta_H - T*Delta_S",
            Description = "Change in Gibbs free energy determines reaction spontaneity.",
            Parameters = [],
            Variables = [
                new("Delta_G", LawVariableType.Scalar, "J/mol", "Gibbs free energy change"),
                new("Delta_H", LawVariableType.Scalar, "J/mol", "Enthalpy change"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("Delta_S", LawVariableType.Scalar, "J/(mol*K)", "Entropy change"),
            ],
            BoundaryConditionDefs = [
                new("ConstantTandP", "T and P fixed", "Standard conditions"),
            ],
            ApplicableDomains = ["Thermochemistry", "Chemical engineering", "Biochemistry"],
            Reference = "Gibbs, J.W. (1876). On the Equilibrium of Heterogeneous Substances.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Gibbs", "free energy", "enthalpy", "entropy"]
        };

        yield return new LawDefinition
        {
            Id = "chem.vant-hoff",
            Name = "Van't Hoff Equation (Equilibrium vs Temperature)",
            Category = LawCategory.Chemistry,
            Expression = "d(ln K)/dT = Delta_H / (R*T^2)",
            Description = "Temperature dependence of the equilibrium constant.",
            Parameters = [
                new("R", 8.31446, "J/(mol*K)", 0, 100, "Gas constant"),
            ],
            Variables = [
                new("K", LawVariableType.Scalar, "dimensionless", "Equilibrium constant"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
                new("Delta_H", LawVariableType.Scalar, "J/mol", "Reaction enthalpy"),
                new("R", LawVariableType.Scalar, "J/(mol*K)", "Gas constant"),
            ],
            BoundaryConditionDefs = [
                new("IdealSolution", "ideal behavior", "Activity ~ concentration"),
            ],
            ApplicableDomains = ["Chemical equilibrium", "Industrial chemistry"],
            Reference = "van't Hoff, J.H. (1884). Etudes de Dynamique Chimique.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["van't Hoff", "equilibrium", "temperature dependence"]
        };

        yield return new LawDefinition
        {
            Id = "chem.daltons-law",
            Name = "Dalton's Law of Partial Pressures",
            Category = LawCategory.Chemistry,
            Expression = "P_total = P_1 + P_2 + ... + P_n",
            Description = "Total pressure of a gas mixture equals the sum of partial pressures.",
            Parameters = [],
            Variables = [
                new("P_total", LawVariableType.Scalar, "Pa", "Total pressure"),
                new("P_i", LawVariableType.Scalar, "Pa", "Partial pressure of component i"),
            ],
            BoundaryConditionDefs = [
                new("IdealGas", "each gas is ideal", "No interactions between gases"),
            ],
            ApplicableDomains = ["Gas mixtures", "Atmospheric chemistry", "Scuba diving"],
            Reference = "Dalton, J. (1801).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Dalton", "partial pressure", "gas mixture"]
        };

        yield return new LawDefinition
        {
            Id = "chem.avogadro",
            Name = "Avogadro's Law",
            Category = LawCategory.Chemistry,
            Expression = "V/n = const (at constant T and P)",
            Description = "Equal volumes of gases at same T and P contain equal numbers of molecules.",
            Parameters = [],
            Variables = [
                new("V", LawVariableType.Scalar, "m^3", "Volume"),
                new("n", LawVariableType.Scalar, "mol", "Amount of substance"),
            ],
            BoundaryConditionDefs = [
                new("ConstantTP", "T and P fixed", "Isothermal, isobaric"),
            ],
            ApplicableDomains = ["Stoichiometry", "Gas law calculations"],
            Reference = "Avogadro, A. (1811). Journal de Physique.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Avogadro", "molar volume", "gas law"]
        };

        yield return new LawDefinition
        {
            Id = "chem.bose-einstein",
            Name = "Bose-Einstein Distribution",
            Category = LawCategory.Chemistry,
            Expression = "n_i = 1 / (exp((E_i - mu)/(k_B*T)) - 1)",
            Description = "Distribution of bosons over energy states at thermal equilibrium.",
            Parameters = [
                new("k_B", 1.380649e-23, "J/K", 0, 1e-15, "Boltzmann constant"),
            ],
            Variables = [
                new("n_i", LawVariableType.Scalar, "dimensionless", "Mean occupation number"),
                new("E_i", LawVariableType.Scalar, "J", "Energy of state i"),
                new("mu", LawVariableType.Scalar, "J", "Chemical potential"),
                new("k_B", LawVariableType.Scalar, "J/K", "Boltzmann constant"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
            ],
            BoundaryConditionDefs = [
                new("Bosons", "integer spin particles", "Photons, phonons, etc."),
            ],
            ApplicableDomains = ["Statistical mechanics", "Photon statistics"],
            Reference = "Bose, S.N. (1924); Einstein, A. (1924).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Bose-Einstein", "bosons", "distribution"]
        };

        yield return new LawDefinition
        {
            Id = "chem.fermi-dirac",
            Name = "Fermi-Dirac Distribution",
            Category = LawCategory.Chemistry,
            Expression = "n_i = 1 / (exp((E_i - mu)/(k_B*T)) + 1)",
            Description = "Distribution of fermions over energy states at thermal equilibrium.",
            Parameters = [
                new("k_B", 1.380649e-23, "J/K", 0, 1e-15, "Boltzmann constant"),
            ],
            Variables = [
                new("n_i", LawVariableType.Scalar, "dimensionless", "Mean occupation number (0 or 1)"),
                new("E_i", LawVariableType.Scalar, "J", "Energy of state i"),
                new("mu", LawVariableType.Scalar, "J", "Chemical potential"),
                new("k_B", LawVariableType.Scalar, "J/K", "Boltzmann constant"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
            ],
            BoundaryConditionDefs = [
                new("Fermions", "half-integer spin particles", "Electrons, protons, etc."),
            ],
            ApplicableDomains = ["Solid-state physics", "Semiconductor physics"],
            Reference = "Dirac, P.A.M. (1926).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Fermi-Dirac", "fermions", "distribution"]
        };

        yield return new LawDefinition
        {
            Id = "bio.allometric-scaling",
            Name = "Allometric Scaling (Metabolic Rate vs Mass)",
            Category = LawCategory.Biology,
            Expression = "B = B_0 * M^(3/4)",
            Description = "Metabolic rate scales with body mass to the 3/4 power (Kleiber's law).",
            Parameters = [
                new("B_0", 1.0, "varies", 0, 1e6, "Normalization constant"),
            ],
            Variables = [
                new("B", LawVariableType.Scalar, "W", "Metabolic rate"),
                new("B_0", LawVariableType.Scalar, "varies", "Normalization constant"),
                new("M", LawVariableType.Scalar, "kg", "Body mass"),
            ],
            BoundaryConditionDefs = [
                new("Interspecies", "across many species", "Empirical relationship"),
            ],
            ApplicableDomains = ["Ecology", "Physiology", "Pharmacology"],
            Reference = "Kleiber, M. (1932). Hilgardia.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["allometric", "scaling", "metabolic", "Kleiber"]
        };

        yield return new LawDefinition
        {
            Id = "bio.hardy-weinberg",
            Name = "Hardy-Weinberg Equilibrium",
            Category = LawCategory.Biology,
            Expression = "p^2 + 2pq + q^2 = 1; p + q = 1",
            Description = "Genotype frequencies in a population at genetic equilibrium.",
            Parameters = [],
            Variables = [
                new("p", LawVariableType.Scalar, "dimensionless", "Frequency of dominant allele"),
                new("q", LawVariableType.Scalar, "dimensionless", "Frequency of recessive allele"),
            ],
            BoundaryConditionDefs = [
                new("LargePopulation", "N >> 1", "No genetic drift"),
                new("RandomMating", "no assortative mating", "Panmixia"),
                new("NoSelection", "all genotypes equally fit", "No natural selection"),
                new("NoMigration", "closed population", "No gene flow"),
                new("NoMutation", "no new alleles", "Stable allele frequencies"),
            ],
            ApplicableDomains = ["Population genetics", "Medical genetics", "Forensic science"],
            Reference = "Hardy, G.H. (1908) / Weinberg, W. (1908).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Hardy-Weinberg", "genetics", "equilibrium"]
        };

        yield return new LawDefinition
        {
            Id = "fin.capm",
            Name = "Capital Asset Pricing Model (CAPM)",
            Category = LawCategory.Finance,
            Expression = "E(R_i) = R_f + beta_i * (E(R_m) - R_f)",
            Description = "Expected return of an asset from its systematic risk (beta).",
            Parameters = [],
            Variables = [
                new("E_Ri", LawVariableType.Scalar, "year^-1", "Expected return of asset i"),
                new("R_f", LawVariableType.Scalar, "year^-1", "Risk-free rate"),
                new("beta_i", LawVariableType.Scalar, "dimensionless", "Beta of asset i"),
                new("E_Rm", LawVariableType.Scalar, "year^-1", "Expected market return"),
            ],
            BoundaryConditionDefs = [
                new("DiversifiedPortfolio", "unsystematic risk eliminated", "Only systematic risk priced"),
            ],
            ApplicableDomains = ["Portfolio management", "Asset pricing", "Risk analysis"],
            Reference = "Sharpe, W.F. (1964) / Lintner, J. (1965).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["CAPM", "beta", "asset pricing"]
        };

        yield return new LawDefinition
        {
            Id = "fin.sharpe-ratio",
            Name = "Sharpe Ratio",
            Category = LawCategory.Finance,
            Expression = "S = (R_p - R_f) / sigma_p",
            Description = "Risk-adjusted return measure: excess return per unit of volatility.",
            Parameters = [],
            Variables = [
                new("S", LawVariableType.Scalar, "dimensionless", "Sharpe ratio"),
                new("R_p", LawVariableType.Scalar, "year^-1", "Portfolio return"),
                new("R_f", LawVariableType.Scalar, "year^-1", "Risk-free rate"),
                new("sigma_p", LawVariableType.Scalar, "year^-0.5", "Portfolio volatility"),
            ],
            BoundaryConditionDefs = [
                new("NormallyDistributed", "returns are Gaussian", "Standard deviation meaningful"),
            ],
            ApplicableDomains = ["Portfolio evaluation", "Fund comparison"],
            Reference = "Sharpe, W.F. (1966). J. Business.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Sharpe", "ratio", "risk-adjusted"]
        };

        yield return new LawDefinition
        {
            Id = "fin.put-call-parity",
            Name = "Put-Call Parity",
            Category = LawCategory.Finance,
            Expression = "C - P = S - K*exp(-r*T)",
            Description = "Relationship between European put and call option prices.",
            Parameters = [],
            Variables = [
                new("C", LawVariableType.Scalar, "currency", "Call option price"),
                new("P", LawVariableType.Scalar, "currency", "Put option price"),
                new("S", LawVariableType.Scalar, "currency", "Spot price"),
                new("K", LawVariableType.Scalar, "currency", "Strike price"),
                new("r", LawVariableType.Scalar, "year^-1", "Risk-free rate"),
                new("T", LawVariableType.Scalar, "years", "Time to expiry"),
            ],
            BoundaryConditionDefs = [
                new("EuropeanOptions", "same strike and expiry", "European exercise only"),
            ],
            ApplicableDomains = ["Options pricing", "Derivatives trading"],
            Reference = "Stoll, H.R. (1969). J. Business.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["put-call parity", "options", "no-arbitrage"]
        };

        yield return new LawDefinition
        {
            Id = "mat.oirowan-equation",
            Name = "Orowan Equation (Plastic Strain Rate)",
            Category = LawCategory.MaterialScience,
            Expression = "d(epsilon_p)/dt = rho_m * b * v_disl",
            Description = "Plastic strain rate from mobile dislocation density and velocity.",
            Parameters = [],
            Variables = [
                new("epsilon_p", LawVariableType.Scalar, "dimensionless", "Plastic strain"),
                new("t", LawVariableType.Scalar, "s", "Time"),
                new("rho_m", LawVariableType.Scalar, "m^-2", "Mobile dislocation density"),
                new("b", LawVariableType.Scalar, "m", "Burgers vector"),
                new("v_disl", LawVariableType.Scalar, "m/s", "Dislocation velocity"),
            ],
            BoundaryConditionDefs = [
                new("CrystalPlasticity", "dislocation mechanism", "Metals"),
            ],
            ApplicableDomains = ["Metallurgy", "Plasticity", "Crystal mechanics"],
            Reference = "Orowan, E. (1934). Zeitschrift fuer Physik.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Orowan", "dislocation", "plastic strain"]
        };

        yield return new LawDefinition
        {
            Id = "mat.ostwald-ripening",
            Name = "Ostwald Ripening (LSW Theory)",
            Category = LawCategory.MaterialScience,
            Expression = "<r(t)>^3 - <r(0)>^3 = (8*gamma*D*C_inf*V_m*t) / (9*R*T)",
            Description = "Coarsening of precipitate particles: large particles grow at expense of small.",
            Parameters = [
                new("R", 8.31446, "J/(mol*K)", 0, 100, "Gas constant"),
            ],
            Variables = [
                new("r", LawVariableType.Scalar, "m", "Average particle radius"),
                new("gamma", LawVariableType.Scalar, "J/m^2", "Interfacial energy"),
                new("D", LawVariableType.Scalar, "m^2/s", "Diffusion coefficient"),
                new("C_inf", LawVariableType.Scalar, "mol/m^3", "Matrix solubility"),
                new("V_m", LawVariableType.Scalar, "m^3/mol", "Molar volume"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
            ],
            BoundaryConditionDefs = [
                new("DiffusionControlled", "D << interface kinetics", "Bulk diffusion limited"),
            ],
            ApplicableDomains = ["Alloy design", "Nanoparticles", "Phase separation"],
            Reference = "Ostwald, W. (1900) / Lifshitz & Slyozov (1961).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Ostwald ripening", "coarsening", "precipitate"]
        };

        yield return new LawDefinition
        {
            Id = "climate.planck-function",
            Name = "Planck Function (Spectral Radiance)",
            Category = LawCategory.Climate,
            Expression = "B(nu,T) = (2*h*nu^3/c^2) * 1/(exp(h*nu/(k_B*T)) - 1)",
            Description = "Spectral radiance of black-body radiation as a function of frequency.",
            Parameters = [
                new("h", 6.62607015e-34, "J*s", 0, 1e-30, "Planck constant"),
                new("c", 2.998e8, "m/s", 0, 1e9, "Speed of light"),
                new("k_B", 1.380649e-23, "J/K", 0, 1e-15, "Boltzmann constant"),
            ],
            Variables = [
                new("B", LawVariableType.Scalar, "W/(m^2*sr*Hz)", "Spectral radiance"),
                new("nu", LawVariableType.Scalar, "Hz", "Frequency"),
                new("T", LawVariableType.Scalar, "K", "Temperature"),
            ],
            BoundaryConditionDefs = [
                new("BlackBody", "ideal emitter", "Spectral distribution"),
            ],
            ApplicableDomains = ["Climate science", "Remote sensing", "Astrophysics"],
            Reference = "Planck, M. (1900).",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["Planck", "radiance", "black body"]
        };

        yield return new LawDefinition
        {
            Id = "climate.optical-depth",
            Name = "Atmospheric Optical Depth",
            Category = LawCategory.Climate,
            Expression = "tau = integral(kappa * rho ds)",
            Description = "Measure of atmospheric transparency; relates to transmission via Beer-Lambert.",
            Parameters = [],
            Variables = [
                new("tau", LawVariableType.Scalar, "dimensionless", "Optical depth"),
                new("kappa", LawVariableType.Scalar, "m^2/kg", "Mass absorption coefficient"),
                new("rho", LawVariableType.Scalar, "kg/m^3", "Atmospheric density"),
                new("s", LawVariableType.Scalar, "m", "Path length"),
            ],
            BoundaryConditionDefs = [
                new("PlaneParallel", "horizontal layers", "1D atmosphere"),
            ],
            ApplicableDomains = ["Atmospheric science", "Climate modeling", "Remote sensing"],
            Reference = "Standard result in radiative transfer.",
            Version = "1.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
            Tags = ["optical depth", "atmosphere", "transmission"]
        };
    }
}

