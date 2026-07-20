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


//  SECTION 2 - LawLibraryRegistry (static, frozen set of all laws)

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
