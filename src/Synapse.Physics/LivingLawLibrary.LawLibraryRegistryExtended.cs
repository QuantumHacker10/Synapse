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


//  SECTION 5 — Additional Law Definitions (Extended Library)

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
