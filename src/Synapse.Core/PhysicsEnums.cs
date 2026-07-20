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

// SECTION 2: ENUMERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Type d'unite de champ physique — identifie quelle couche du PhysicsField
/// est active pour un point donne. Les drapeaux sont combinables par OR bit,
/// permettant un systeme multi-physique ou chaque point peut porter
/// simultanement des champs thermiques, mecaniques, fluides, etc.
/// </summary>
[Flags]
public enum FieldLayer : uint
{
    /// <summary>Aucun champ actif.</summary>
    None = 0,
    /// <summary>Champ thermique : temperature, flux de chaleur, conductivite thermique.</summary>
    Thermal = 1u << 0,
    /// <summary>Champ mecanique : contraintes, deformations, vitesse, densite.</summary>
    Mechanical = 1u << 1,
    /// <summary>Champ fluide : Navier-Stokes, viscosite, turbulence, pression.</summary>
    Fluid = 1u << 2,
    /// <summary>Champ electromagnetique : champ E, champ B, courant, polarisation.</summary>
    Electromagnetic = 1u << 3,
    /// <summary>Champ chimique : concentrations, cinetique reactionnelle, equilibre.</summary>
    Chemical = 1u << 4,
    /// <summary>Champ biologique : croissance cellulaire, diffusion biochimique.</summary>
    Biological = 1u << 5,
    /// <summary>Champ epidemiologique : modele SIR/SEIR, taux de contagion, R0.</summary>
    Epidemiological = 1u << 6,
    /// <summary>Champ financier : prix, volatilite, flux de capital, options.</summary>
    Financial = 1u << 7,
    /// <summary>Champ quantique : fonctions d'onde, densite de probabilite.</summary>
    Quantum = 1u << 8,
    /// <summary>Champ acoustique : pression acoustique, intensite sonore, attenuation.</summary>
    Acoustic = 1u << 9,
    /// <summary>Champ gravitationnel : potentiel gravitationnel, marree, force de marree.</summary>
    Gravitational = 1u << 10,
    /// <summary>Tous les champs actifs simultanement (0x7FFFFFFF).</summary>
    All = 0x7FFFFFFF
}

/// <summary>
/// Processus stochastique sous-jacent d'un champ. Determine la nature
/// du bruit et la dynamique aleatoire appliquee aux champs physiques.
/// Chaque processus a ses propres proprietes mathematiques : stationnarite,
/// ergodicite, independance des incrementes, etc.
/// </summary>
public enum StochasticProcess
{
    /// <summary>Pas de composante aleatoire. Evolution purement deterministe.</summary>
    Deterministic,
    /// <summary>Mouvement brownien geometrique : dS = mu*S*dt + sigma*S*dW. Modelise les prix d'actifs financiers.</summary>
    GeometricBrownianMotion,
    /// <summary>Processus d'Ornstein-Uhlenbeck : dx = theta*(mu-x)*dt + sigma*dW. Retour a la moyenne avec bruit.</summary>
    OrnsteinUhlenbeck,
    /// <summary>Processus de Poisson : arrivees a taux lambda. Evenements discrets aleatoires.</summary>
    PoissonArrival,
    /// <summary>Modele de saut-diffusion : Brownien + sauts ponctuels (Merton, Kou).</summary>
    JumpDiffusion,
    /// <summary>Chaine de Markov a temps discret ou continu. Transitions probabilistes.</summary>
    MarkovChain,
    /// <summary>Processus de Wiener standard : dW ~ N(0, dt). Base du calcul stochastique.</summary>
    WienerProcess,
    /// <summary>Mouvement brownien fractionnaire : H > 0.5 persistant, H < 0.5 anti-persistant.</summary>
    FractionalBrownian
}

/// <summary>
/// Condition aux limites pour les solveurs PDE. Chaque type impose
/// une contrainte differente sur la frontiere du domaine de simulation.
/// Le choix de la condition aux limites affecte la stabilite et la precision
/// du schema numerique utilise.
/// </summary>
public enum BoundaryConditionKind
{
    /// <summary>Valeur fixe sur la frontiere : u = g(x,t). Impose la valeur de la solution.</summary>
    Dirichlet,
    /// <summary>Flux fixe sur la frontiere : du/dn = h(x,t). Impose la derivee normale.</summary>
    Neumann,
    /// <summary>Combinaison lineaire : alpha*u + beta*du/dn = gamma. Condition la plus generale.</summary>
    Robin,
    /// <summary>Conditions periodiques : u(0,t) = u(L,t). Pour domaines periodiques.</summary>
    Periodic,
    /// <summary>Conditions absorbantes (PML, Mur). Empeche les reflexions aux frontieres.</summary>
    Absorbing,
    /// <summary>Conditions reflechissantes : du/dn = 0. Symetrie ou mur rigide.</summary>
    Reflecting,
    /// <summary>Conditions mixtes : differentes conditions selon les zones de la frontiere.</summary>
    Mixed
}

/// <summary>
/// Type de couplage entre champs physiques. Determine comment
/// l'evolution d'un champ affecte (et est affectee par) un autre champ.
/// Le couplage bidirectionnel est courant en physique multi-echelle.
/// </summary>
public enum CouplingType
{
    /// <summary>Pas de couplage entre les champs.</summary>
    None,
    /// <summary>Couplage lineaire : F = k * G. Proportionnel.</summary>
    Linear,
    /// <summary>Couplage bilineaire : F = k * G * H. Interaction croisee.</summary>
    Bilinear,
    /// <summary>Couplage non-lineaire : F = f(G) avec f non-linéaire.</summary>
    Nonlinear,
    /// <summary>Couplage bidirectionnel : F influence G et G influence F.</summary>
    Bidirectional,
    /// <summary>Couplage en cascade : F -> G -> H (chaine unidirectionnelle).</summary>
    Cascaded,
    /// <summary>Couplage avec retroaction : F -> G -> F (boucle de retroaction).</summary>
    Feedback
}

/// <summary>
/// Systeme d'unites pour les conversions et l'analyse dimensionnelle.
/// Le systeme SI est le standard international. Les autres systemes
/// sont maintenus pour la compatibilite avec des modeles existants.
/// </summary>
public enum UnitSystem
{
    /// <summary>Systeme International (m, kg, s, K, A, mol, cd). Standard scientifique.</summary>
    SI,
    /// <summary>Systeme CGS (cm, g, s). Utilise en astrophysique et electromagnetisme.</summary>
    CGS,
    /// <summary>Systeme Imperial (ft, lb, s, deg R). Utilise aux USA et en Grande-Bretagne.</summary>
    Imperial,
    /// <summary>Unites naturelles (hbar = c = 1). Utilisee en physique theorique.</summary>
    Natural,
    /// <summary>Unites atomiques (m_e, e, hbar, a_0). Utilisee en chimie quantique.</summary>
    Atomic
}

/// <summary>
/// Dimension physique pour l'analyse dimensionnelle (systeme MLTQThetaNJ).
/// Chaque dimension est representee par un bit dans un long 64 bits,
/// permettant des operations de verification et de combinaison par bit.
/// </summary>
[Flags]
public enum Dimension : long
{
    /// <summary>Nombre sans dimension.</summary>
    Dimensionless = 0,
    /// <summary>Longueur [L]. Unite de base du systeme SI.</summary>
    Length = 1L << 0,
    /// <summary>Masse [M]. Unite de base du systeme SI.</summary>
    Mass = 1L << 1,
    /// <summary>Temps [T]. Unite de base du systeme SI.</summary>
    Time = 1L << 2,
    /// <summary>Temperature [Theta]. Unite de base du systeme SI.</summary>
    Temperature = 1L << 3,
    /// <summary>Charge electrique [Q]. Unite de base du systeme SI.</summary>
    ElectricCharge = 1L << 4,
    /// <summary>Substance [N]. Unite de base du systeme SI (mole).</summary>
    AmountOfSubstance = 1L << 5,
    /// <summary>Luminosite [J]. Unite de base du systeme SI (candela).</summary>
    LuminousIntensity = 1L << 6,
    /// <summary>Vitesse [L T^-1]. Dimension derivee : longueur par unite de temps.</summary>
    Velocity = Length | Time,
    /// <summary>Acceleration [L T^-2]. Derivee de la vitesse par rapport au temps.</summary>
    Acceleration = Length | (Time << 1),
    /// <summary>Force [M L T^-2]. Deuxieme loi de Newton : F = ma.</summary>
    Force = Mass | Length | (Time << 1),
    /// <summary>Energie [M L^2 T^-2]. Travail = force x distance.</summary>
    Energy = Mass | (Length << 1) | (Time << 1),
    /// <summary>Puissance [M L^2 T^-3]. Energie par unite de temps.</summary>
    Power = Mass | (Length << 1) | (Time << 2),
    /// <summary>Pression [M L^-1 T^-2]. Force par unite de surface.</summary>
    Pressure = Mass | (Time << 1),
    /// <summary>Densite [M L^-3]. Masse par unite de volume.</summary>
    Density = Mass,
    /// <summary>Viscosite dynamique [M L^-1 T^-1]. Resistance a l'ecoulement.</summary>
    Viscosity = Mass | Time,
    /// <summary>Conductivite thermique [M L T^-3 Theta^-1]. Capacite de conduction.</summary>
    ThermalConductivity = Mass | Length | (Time << 2),
    /// <summary>Capacite thermique [M L^2 T^-2 Theta^-1]. Energie pour changer la temperature.</summary>
    HeatCapacity = Mass | (Length << 1) | (Time << 1),
    /// <summary>Champ electrique [M L T^-3 Q^-1]. Force par unite de charge.</summary>
    ElectricField = Mass | Length | (Time << 2),
    /// <summary>Champ magnetique [M T^-2 Q^-1]. Force de Lorentz sur charges en mouvement.</summary>
    MagneticField = Mass | (Time << 1)
}

/// <summary>
/// Type de materiau pour la classification et la selection automatique
/// de modeles constitutifs. Chaque categorie a des comportements
/// mecaniques, thermiques et optiques caracteristiques.
/// </summary>
public enum MaterialCategory
{
    /// <summary>Metal pur (fer, cuivre, aluminium, etc.). Conducteur, ductile.</summary>
    Metal,
    /// <summary>Alliage (acier, bronze, superalliage). Proprietes combinees.</summary>
    Alloy,
    /// <summary>Ceramique (alumine, zircone). Dur, fragile, isolant thermique.</summary>
    Ceramic,
    /// <summary>Polymere (plastique, caoutchouc). Flexible, isolant.</summary>
    Polymer,
    /// <summary>Composite (fibre de carbone, Kevlar). Anisotrope, legere.</summary>
    Composite,
    /// <summary>Liquide (eau, huile, mercure). Incompressible, visqueux.</summary>
    Liquid,
    /// <summary>Gaz (air, helium, CO2). Compressible, diffusion rapide.</summary>
    Gas,
    /// <summary>Plasma (ionise). Conducteur, champs electrique et magnetique.</summary>
    Plasma,
    /// <summary>Materiau biologique (tissu, os, sang). Visco-elastique, vivant.</summary>
    Biological,
    /// <summary>Semiconducteur (silicium, GaAs). Proprietes electriques tunables.</summary>
    Semiconductor,
    /// <summary>Isolant electrique (verre, caoutchouc). Non-conducteur.</summary>
    Insulator,
    /// <summary>Cristal (quartz, diamant). Anisotrope, structure periodique.</summary>
    Crystal,
    /// <summary>Materiau amorphe (verre, polymere). Isotrope, sans structure a longue distance.</summary>
    Amorphous,
    /// <summary>Mousse (polystyrene, metal). Faible densite, isolation.</summary>
    Foam,
    /// <summary>Gel (silice, polymerique). Semi-solide, retention d'eau.</summary>
    Gel,
    /// <summary>Materiau non classifie.</summary>
    Unknown
}

/// <summary>
/// Methodes d'interpolation entre deux materiaux pour la transition
/// progressive dans les zones de melange ou de gradient de propriete.
/// </summary>
public enum InterpolationMethod
{
    /// <summary>Interpolation lineaire simple : m = (1-t)*m1 + t*m2.</summary>
    Linear,
    /// <summary>Interpolation spherique (SLERP) dans l'espace des proprietes normalisees.</summary>
    Spherical,
    /// <summary>Interpolation perceptive (espace Lab ou Oklab) pour des transitions visuelles.</summary>
    Perceptual,
    /// <summary>Interpolation cubique par morceaux pour des transitions plus douces.</summary>
    PiecewiseCubic
}

// ═══════════════════════════════════════════════════════════════════════════════
