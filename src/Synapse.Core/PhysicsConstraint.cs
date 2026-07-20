// SYNAPSE OMNIA — Synapse.Core
// Split from PhysicsState.cs for maintainability.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Synapse.Core;

// SECTION 10: PHYSICS CONSTRAINT — CONTRAINTES PINN
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Contrainte physique appliquee a un point du champ. Le PINN (Physics-Informed
/// Neural Network) utilise ces contraintes pour garantir la coherence physique
/// de la solution apprise. La perte totale est la somme des residus de contraintes.
///
/// Types de contraintes :
/// - EDP : equation differentielle partielle (Navier-Stokes, diffusion, etc.)
/// - Conditions aux limites (Dirichlet, Neumann, Robin)
/// - Conditions initiales
/// - Inegalites physiques (positivite, bornes)
/// </summary>
public sealed class PhysicsConstraint
{
    /// <summary>Nom descriptif de la contrainte.</summary>
    public string Name { get; init; }
    /// <summary>Fonction residuelle : retourne 0 si la contrainte est satisfaite.</summary>
    public Func<PhysicsState, FieldGradient, double> Residual { get; init; }
    /// <summary>Couche cible de la contrainte.</summary>
    public FieldLayer TargetLayer { get; init; }
    /// <summary>Poids dans la fonction de perte (importance relative).</summary>
    public double Weight { get; set; } = 1.0;
    /// <summary>Si vrai, la contrainte est verifiee exactement (penalite dure).</summary>
    public bool IsHardConstraint { get; set; } = false;
    /// <summary>Tolerance d'erreur acceptee (pour les contraintes douces).</summary>
    public double Tolerance { get; set; } = 1e-6;
    /// <summary>Type de condition aux limites associee.</summary>
    public BoundaryConditionKind BoundaryType { get; set; } = BoundaryConditionKind.Dirichlet;
    /// <summary>Description de la contrainte.</summary>
    public string? Description { get; init; }
    /// <summary>Poids adaptatif (ajuste pendant l'entrainement).</summary>
    public double AdaptiveWeight { get; set; } = 1.0;
    /// <summary>Frequence de mise a jour du poids adaptatif.</summary>
    public int UpdateFrequency { get; set; } = 100;

    /// <summary>Calcule le residu et le pese pour la perte totale.</summary>
    public double ComputeLoss(PhysicsState state, FieldGradient gradient) => Weight * AdaptiveWeight * Residual(state, gradient) * Residual(state, gradient);

    /// <summary>Factory : cree une contrainte de conservation de la masse.</summary>
    public static PhysicsConstraint ConservationOfMass(double tolerance = 1e-6) => new() { Name = "MassConservation", TargetLayer = FieldLayer.Fluid, Tolerance = tolerance, Residual = (s, g) => g.VelocityDivergence * s.Density + s.Density * g.VelocityDivergence, Description = "d(rho)/dt + div(rho*v) = 0" };

    /// <summary>Factory : cree une contrainte de conservation de la quantite de mouvement (Navier-Stokes).</summary>
    public static PhysicsConstraint MomentumConservation(double viscosity) => new() { Name = "MomentumConservation", TargetLayer = FieldLayer.Fluid, Weight = viscosity, Residual = (s, g) => s.Density * g.VelocityDivergence + viscosity * g.VelocityTensor.Trace, Description = "rho*(dv/dt + v.grad(v)) = -grad(p) + mu*laplacian(v)" };

    /// <summary>Factory : cree une contrainte de conservation de l'energie.</summary>
    public static PhysicsConstraint EnergyConservation(double thermalDiffusivity) => new() { Name = "EnergyConservation", TargetLayer = FieldLayer.Thermal, Weight = thermalDiffusivity, Residual = (s, g) => g.TemperatureLaplacian * thermalDiffusivity, Description = "dT/dt + v.grad(T) = alpha*laplacian(T)" };

    public override string ToString() => $"Constraint[{Name}] weight={Weight:F2} (hard={IsHardConstraint})";
}

// ═══════════════════════════════════════════════════════════════════════════════
