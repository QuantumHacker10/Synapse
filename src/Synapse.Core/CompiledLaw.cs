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

// SECTION 7: COMPILED LAW — LOIS PHYSIQUES COMPPILEES
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Represente une loi physique appliquee au champ — version runtime du LivingLawCompiler.
/// Chaque loi possede une expression mathematique, un evaluateur compile,
/// et des parametres modifiables en temps reel. Les lois peuvent etre coupees
/// entre elles pour modeliser les interactions multi-physiques.
///
/// Le systeme de couplage supporte :
/// - Couplage lineaire : F = k * G
/// - Couplage bilineaire : F = k * G * H
/// - Couplage non-lineaire : F = f(G)
/// - Couplage bidirectionnel : F <-> G
/// - Couplage en cascade : F -> G -> H
/// - Couplage avec retroaction : F -> G -> F
/// </summary>
public sealed class CompiledLaw
{
    /// <summary>Identifiant unique de la loi.</summary>
    public string Id { get; init; }
    /// <summary>Nom descriptif de la loi.</summary>
    public string Name { get; init; }
    /// <summary>Expression mathematique source (pour reference).</summary>
    public string Expression { get; init; }
    /// <summary>Couche cible : quels champs cette loi affecte.</summary>
    public FieldLayer TargetLayer { get; init; }
    /// <summary>Evaluateur compile pour un point unique (hot-path).</summary>
    public Func<PhysicsState, FieldGradient, double, PhysicsState> Evaluate { get; init; }
    /// <summary>Evaluation en lot (bulk) pour des tableaux de points.</summary>
    public Action<nint, int, double> BulkEvaluate { get; init; }
    /// <summary>Numero de version (incremente a chaque modification).</summary>
    public int Version { get; init; } = 1;
    /// <summary>ID de la version parente (pour le historique).</summary>
    public string? ParentVersionId { get; init; }
    /// <summary>Date de creation (UTC).</summary>
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
    /// <summary>Parametres modifiables en temps reel.</summary>
    public Dictionary<string, double> Parameters { get; init; } = new();
    /// <summary>La loi est-elle active dans la simulation.</summary>
    public bool IsActive { get; set; } = true;
    /// <summary>Force du couplage avec les autres lois.</summary>
    public double CouplingStrength { get; set; } = 1.0;
    /// <summary>Mode de couplage.</summary>
    public CouplingType CouplingMode { get; set; } = CouplingType.Linear;
    /// <summary>IDs des lois coupees.</summary>
    public List<string> CoupledLawIds { get; init; } = new();
    /// <summary>Espace de noms pour l'organisation.</summary>
    public string? Namespace { get; init; }
    /// <summary>Description detaillee de la loi.</summary>
    public string? Description { get; init; }
    /// <references>References bibliographiques.</summary>
    public List<string> References { get; init; } = new();

    public override string ToString() => $"CompiledLaw[{Id}] {Name} v{Version} (active={IsActive}, coupling={CouplingMode})";
}

// ═══════════════════════════════════════════════════════════════════════════════
