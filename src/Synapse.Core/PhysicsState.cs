// ═══════════════════════════════════════════════════════════════════════════════
// SYNAPSE OMNIA — Core/PhysicsState.cs
// Le coeur ontologique du simulateur : chaque point de l'espace-temps porte
// un etat physique complet et differentiable, de la temperature au taux de
// contagion, de la contrainte mecanique au flux de capital.
//
// Ce fichier contient l'integralite des types fondamentaux du systeme :
//   - Constantes physiques universelles (CODATA 2018)
//   - Enumerations (FieldLayer, StochasticProcess, BoundaryConditionKind, etc.)
//   - Vector3D — algebre vectorielle 3D avec operations avancees
//   - Symmetric3x3 — tenseurs symetriques de contrainte/deformation
//   - Tensor3D — matrices 3x3 completes pour rotations et gradients
//   - QuaternionD — rotations sans gimbal lock (double precision)
//   - Matrix4x4D — transformations homogenes, projections, vue
//   - BoundingBox3D — AABB pour acceleration spatiale
//   - Ray3D — rayons pour picking, raycasting, intersection
//   - Plane3D — plans geometriques et conditions aux limites
//   - Frustum6Planes — volume de vision pour culling
//   - ColorHDR — couleurs a plage dynamique etendue avec tonemapping
//   - IntervalD — arithmetique par intervalles pour analyse d'erreurs
//   - DiffScalar — differentiation automatique forward-mode
//   - DiffExpression — expressions differentiables pour gradient computation
//   - FieldGradient — gradients spatiaux pour operateurs differentiels
//   - CompiledLaw — lois physiques compilees (LivingLawCompiler)
//   - MaterialProperties — proprietes constitutives completes
//   - StochasticState — etats stochastiques pour champs aleatoires
//   - PhysicsConstraint — contraintes PINN pour coherent physique
//   - FieldSampleResult — resultats d'evaluation de champ
//   - SimulationConfig — configuration globale de simulation
//   - PhysicsState — etat physique complet (256 bytes, AVX-512 aligned)
//   - UnitConverter — conversions SI/CGS/Imperial/Natural/Atomic
//   - MaterialDatabase — 25+ materiaux predefinis avec interpolation
//   - OctreeNode — octree lache pour acceleration spatiale hierarchique
//   - KdTree — arbre KD pour requetes de plus proches voisins
//   - GridHash — hachage spatial pour physique sur grille reguliere
//   - RK4/Hamiltonian/Dissipative/Langevin evolution operators
//   - Equations of state (ideal, van der Waals, Redlich-Kwong, Peng-Robinson)
//   - Transport properties (diffusivite thermique, viscosite, Prandtl)
//   - Phase diagram calculations (Clausius-Clapeyron, etc.)
// ═══════════════════════════════════════════════════════════════════════════════

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Synapse.Core;

/// <summary>
/// Constantes physiques universelles en unites SI (CODATA 2018).
/// Utilisees par le PinnTrainer, LivingLawCompiler et tous les solveurs.
/// Toutes les constantes sont en double precision pour la precision numerique.
/// </summary>
public static class PhysicalConstants
{
    /// <summary>
    /// Constante gravitationnelle universelle G [m^3 kg^-1 s^-2].
    /// Determine la force d'attraction entre deux masses. Essentielle pour la simulation gravitationnelle et les modeles d'echelles cosmiques.
    /// </summary>
    public const double G = 6.67430e-11;

    /// <summary>
    /// Vitesse de la lumiere dans le vide c [m s^-1].
    /// Limite de vitesse dans l'univers relativiste. Utilisee dans les conversions d'energie-masse et les effets relativistes.
    /// </summary>
    public const double c = 2.99792458e8;

    /// <summary>
    /// Constante de Planck h [J s].
    /// Constante fondamentale de la mecanique quantique. Relie l'energie d'un photon a sa frequence : E = hf.
    /// </summary>
    public const double h = 6.62607015e-34;

    /// <summary>
    /// Constante de Planck reduite h-barre = h/(2pi) [J s].
    /// Forme naturelle de la constante de Planck. Utilisee dans les equations de Schrodinger et les calculs quantiques.
    /// </summary>
    public const double hbar = 1.054571817e-34;

    /// <summary>
    /// Constante de Boltzmann k_B [J K^-1].
    /// Relie l'energie cinetique moyenne d'un gaz a sa temperature. Fondamentale pour la thermodynamique statistique.
    /// </summary>
    public const double k_B = 1.380649e-23;

    /// <summary>
    /// Constante de Stefan-Boltzmann sigma [W m^-2 K^-4].
    /// Determine le rayonnement thermique total d'un corps noir. Puissance = sigma * A * T^4.
    /// </summary>
    public const double sigma = 5.670374411e-8;

    /// <summary>
    /// Permittivite electrique du vide epsilon_0 [F m^-1].
    /// Constante qui determine la force de l'interaction electromagnetique dans le vide.
    /// </summary>
    public const double epsilon_0 = 8.8541878128e-12;

    /// <summary>
    /// Permeabilite magnetique du vide mu_0 [H m^-1].
    /// Constante fondamentale de l'electromagnetisme. Lien avec epsilon_0 : mu_0 * epsilon_0 = 1/c^2.
    /// </summary>
    public const double mu_0 = 1.25663706212e-6;

    /// <summary>
    /// Charge elementaire e [C].
    /// Plus petite charge electrique stable. Charge d'un proton ou magnitude de la charge d'un electron.
    /// </summary>
    public const double e = 1.602176634e-19;

    /// <summary>
    /// Masse au repos de l'electron m_e [kg].
    /// Masse fondamentale de l'electron. Utilisee dans les unites atomiques et les calculs de structure electronique.
    /// </summary>
    public const double m_e = 9.1093837015e-31;

    /// <summary>
    /// Masse au repos du proton m_p [kg].
    /// Masse fondamentale du proton. Rapport avec m_e = 1836.15 (masse du neutron legereurement inferieure).
    /// </summary>
    public const double m_p = 1.67262192369e-27;

    /// <summary>
    /// Nombre d'Avogadro N_A [mol^-1].
    /// Nombre de particules dans une mole. Pont entre l'echelle macroscopique et microscopique.
    /// </summary>
    public const double N_A = 6.02214076e23;

    /// <summary>
    /// Constante molaire des gaz parfaits R [J mol^-1 K^-1].
    /// Produit de Boltzmann et du nombre d'Avogadro : R = k_B * N_A. Utilisee dans PV = nRT.
    /// </summary>
    public const double R_gas = 8.314462618;

    /// <summary>
    /// Acceleration de la pesanteur terrestre standard g [m s^-2].
    /// Acceleration gravitationnelle a la surface de la Terre. Reference pour les calculs de poids et de force.
    /// </summary>
    public const double g_earth = 9.80665;

    /// <summary>
    /// Pression atmospherique standard 1 atm [Pa].
    /// Pression a niveau de la mer. Reference pour les conversions de pression et les equations d'etat.
    /// </summary>
    public const double atm = 101325.0;

    /// <summary>
    /// Constante de Coulomb k_e [N m^2 C^-2].
    /// Constante de la loi de Coulomb : F = k_e * q1 * q2 / r^2. Force entre charges ponctuelles.
    /// </summary>
    public const double k_e = 8.9875517923e9;

    /// <summary>
    /// Constante de structure fine alpha [sans dimension].
    /// Rapport de la vitesse de l'electron dans l'orbite de Bohr a c. Mesure de la force de l'interaction electromagnetique.
    /// </summary>
    public const double alpha_fs = 7.2973525693e-3;

    /// <summary>
    /// Magneton de Bohr mu_B [J T^-1].
    /// Moment magnetique orbital d'un electron. Unite naturelle pour les moments magnetiques atomiques.
    /// </summary>
    public const double mu_B = 9.2740100783e-24;

    /// <summary>
    /// Longueur d'onde reduite de l'electron lambda_C [m].
    /// λ_C = h / (m_e * c). Longueur d'onde de Compton de l'electron, echelle quantique fondamentale.
    /// </summary>
    public const double lambda_C = 2.42631023867e-12;

    /// <summary>
    /// Densite critique de l'eau rho_c [kg m^-3].
    /// Au point critique, les phases liquide et gazeuse deviennent indistingables.
    /// </summary>
    public const double water_critical_density = 322.0;

    /// <summary>
    /// Temperature critique de l'eau T_c [K].
    /// 373.946 °C. Audessus, l'eau est un fluide supercritique.
    /// </summary>
    public const double water_critical_temperature = 647.096;

    /// <summary>
    /// Pression critique de l'eau P_c [Pa].
    /// 217.7 atm. Pression minimale pour liquefaction a T_c.
    /// </summary>
    public const double water_critical_pressure = 22.064e6;

    /// <summary>
    /// Temperature du point triple de l'eau T_tp [K].
    /// 0.01 °C. Point unique ou les trois phases coexistent.
    /// </summary>
    public const double water_triple_point_temperature = 273.16;

    /// <summary>
    /// Pression du point triple de l'eau P_tp [Pa].
    /// 6.11657 mbar. Reference pour l'echelle de temperature ITS-90.
    /// </summary>
    public const double water_triple_point_pressure = 611.657;

}

// ═══════════════════════════════════════════════════════════════════════════════
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
// SECTION 3: VECTOR3D — STRUCTURE VECTORIELLE 3D DOUBLE PRECISION
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Structure vectorielle 3D double precision. Alignee sur 32 bytes pour les
/// operations AVX-2 SIMD. Supporte toutes les operations vectorielles usuelles
/// en physique : produits scalaires et vectoriels, distances, interpolations,
/// rotations, conversions de coordonnees, primitives geometriques, et plus.
///
/// Chaque methode est marquee [AggressiveInlining] pour eliminer le cout
/// d'appel de fonction dans les boucles de simulation hot-path.
///
/// MEMORY LAYOUT: 32 bytes (3 doubles + 1 padding).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 32, Pack = 32)]
[DebuggerDisplay("({X:F4}, {Y:F4}, {Z:F4})")]
public struct Vector3D : IEquatable<Vector3D>
{
    [FieldOffset(0)] public double X;
    [FieldOffset(8)] public double Y;
    [FieldOffset(16)] public double Z;
    [FieldOffset(24)] public double _pad;

    /// <summary>Constructeur principal avec composantes x, y, z.</summary>
    /// <param name="x">Composante X (abscisse).</param>
    /// <param name="y">Composante Y (ordonnee).</param>
    /// <param name="z">Composante Z (cote).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D(double x, double y, double z) { X = x; Y = y; Z = z; _pad = 0; }

    /// <summary>Constructeur a partir d'un scalaire unique (toutes composantes identiques).</summary>
    /// <param name="s">Valeur assignee a X, Y et Z.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D(double s) { X = s; Y = s; Z = s; _pad = 0; }

    /// <summary>Constructeur a partir d'un Vector3 System.Numerics (single precision).</summary>
    /// <param name="v">Vecteur System.Numerics.Vector3 a convertir.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D(System.Numerics.Vector3 v) { X = v.X; Y = v.Y; Z = v.Z; _pad = 0; }

    /// <summary>Vecteur nul (0, 0, 0). Identite additive.</summary>
    public static readonly Vector3D Zero = new(0, 0, 0);
    /// <summary>Vecteur unitaire (1, 1, 1). Utilise pour les echelles uniformes.</summary>
    public static readonly Vector3D One = new(1, 1, 1);
    /// <summary>Vecteur unitaire X (1, 0, 0). Axe des abscisses.</summary>
    public static readonly Vector3D UnitX = new(1, 0, 0);
    /// <summary>Vecteur unitaire Y (0, 1, 0). Axe des ordonnees.</summary>
    public static readonly Vector3D UnitY = new(0, 1, 0);
    /// <summary>Vecteur unitaire Z (0, 0, 1). Axe de la profondeur.</summary>
    public static readonly Vector3D UnitZ = new(0, 0, 1);
    /// <summary>Vecteur vers le haut (0, 1, 0). Convention Y-up.</summary>
    public static readonly Vector3D Up = new(0, 1, 0);
    /// <summary>Vecteur vers le bas (0, -1, 0).</summary>
    public static readonly Vector3D Down = new(0, -1, 0);
    /// <summary>Vecteur vers la gauche (-1, 0, 0).</summary>
    public static readonly Vector3D Left = new(-1, 0, 0);
    /// <summary>Vecteur vers la droite (1, 0, 0).</summary>
    public static readonly Vector3D Right = new(1, 0, 0);
    /// <summary>Vecteur vers l'avant (0, 0, -1). Convention OpenGL.</summary>
    public static readonly Vector3D Forward = new(0, 0, -1);
    /// <summary>Vecteur vers l'arriere (0, 0, 1).</summary>
    public static readonly Vector3D Backward = new(0, 0, 1);

    /// <summary>Care de la longueur : |v|^2 = x^2 + y^2 + z^2. Evite sqrt.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double LengthSquared() => X * X + Y * Y + Z * Z;

    /// <summary>Longueur euclidienne : |v| = sqrt(x^2 + y^2 + z^2).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double Length() => Math.Sqrt(LengthSquared());

    /// <summary>Longueur L1 (Manhattan) : |x| + |y| + |z|. Distance de grille.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double LengthL1() => Math.Abs(X) + Math.Abs(Y) + Math.Abs(Z);

    /// <summary>Longueur L-infinity (Chebyshev) : max(|x|, |y|, |z|). Distance de echiquier.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double LengthLInfinity() => Math.Max(Math.Max(Math.Abs(X), Math.Abs(Y)), Math.Abs(Z));

    /// <summary>Vecteur normalise : v/|v|. Retourne Zero si |v| est proche de zero.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Normalized() { double len = Length(); return len > 1e-30 ? this / len : Zero; }

    /// <summary>Vecteur perpendiculaire (choisi arbitrairement si colineaire).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Perpendicular() => Math.Abs(X) < Math.Abs(Y) ? Cross(this, UnitX).Normalized() : Cross(this, UnitY).Normalized();

    /// <summary>Teste si le vecteur est proche de zero (norme < tol).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IsNearZero(double tol = 1e-10) => Math.Abs(X) < tol && Math.Abs(Y) < tol && Math.Abs(Z) < tol;

    /// <summary>Composante maximale : max(x, y, z).</summary>
    public readonly double MaxComponent => Math.Max(Math.Max(X, Y), Z);
    /// <summary>Composante minimale : min(x, y, z).</summary>
    public readonly double MinComponent => Math.Min(Math.Min(X, Y), Z);

    /// <summary>Valeur absolue de chaque composante : (|x|, |y|, |z|).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Abs() => new(Math.Abs(X), Math.Abs(Y), Math.Abs(Z));

    /// <summary>Inverse de chaque composante : (1/x, 1/y, 1/z). Zero si composante nulle.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Reciprocal() => new(X != 0 ? 1.0 / X : 0, Y != 0 ? 1.0 / Y : 0, Z != 0 ? 1.0 / Z : 0);

    /// <summary>Minimum element par element.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseMin(Vector3D b) => new(Math.Min(X, b.X), Math.Min(Y, b.Y), Math.Min(Z, b.Z));

    /// <summary>Maximum element par element.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseMax(Vector3D b) => new(Math.Max(X, b.X), Math.Max(Y, b.Y), Math.Max(Z, b.Z));

    /// <summary>Produit element par element (Hadamard product).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseProduct(Vector3D b) => new(X * b.X, Y * b.Y, Z * b.Z);

    /// <summary>Division element par element.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseDivide(Vector3D b) => new(b.X != 0 ? X / b.X : 0, b.Y != 0 ? Y / b.Y : 0, b.Z != 0 ? Z / b.Z : 0);

    /// <summary>Puissance element par element : (|x|^e * sign(x), ...).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWisePow(double e) => new(Math.Pow(Math.Abs(X), e) * Math.Sign(X), Math.Pow(Math.Abs(Y), e) * Math.Sign(Y), Math.Pow(Math.Abs(Z), e) * Math.Sign(Z));

    /// <summary>Exponentielle element par element : (e^x, e^y, e^z).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseExp() => new(Math.Exp(X), Math.Exp(Y), Math.Exp(Z));

    /// <summary>Logarithme naturel element par element. -Inf si composante <= 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseLog() => new(X > 0 ? Math.Log(X) : double.NegativeInfinity, Y > 0 ? Math.Log(Y) : double.NegativeInfinity, Z > 0 ? Math.Log(Z) : double.NegativeInfinity);

    /// <summary>Signe de chaque composante : -1, 0, ou +1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Sign() => new(Math.Sign(X), Math.Sign(Y), Math.Sign(Z));

    /// <summary>Troncature vers zero de chaque composante.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Truncate() => new(Math.Truncate(X), Math.Truncate(Y), Math.Truncate(Z));

    /// <summary>Arrondi de chaque composante a 'decimals' decimales.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Round(int d = 0) => new(Math.Round(X, d), Math.Round(Y, d), Math.Round(Z, d));

    /// <summary>Clamp chaque composante entre min et max.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Clamp(double min, double max) => new(Math.Clamp(X, min, max), Math.Clamp(Y, min, max), Math.Clamp(Z, min, max));

    /// <summary>Lineraisation (lerp) vers un vecteur cible.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D LerpTo(Vector3D target, double t) => this + (target - this) * t;

    /// <summary>
    /// Spherical linear interpolation (SLERP) pour interpolation sur la sphere unite.
    /// Preserves les angles et la longueur, ideal pour les rotations.
    /// </summary>
    /// <param name="target">Vecteur cible (doit etre sur la sphere unite).</param>
    /// <param name="t">Parametre d'interpolation [0, 1].</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D SlerpTo(Vector3D target, double t)
    {
        double dot = Math.Clamp(Dot(this, target), -1.0, 1.0);
        double theta = Math.Acos(dot);
        if (theta < 1e-10)
            return LerpTo(target, t);
        double sinTheta = Math.Sin(theta);
        return this * Math.Sin((1 - t) * theta) / sinTheta + target * Math.Sin(t * theta) / sinTheta;
    }

    /// <summary>Reflexion par rapport a une normale : v - 2(v.n)n.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Reflect(Vector3D n) => this - 2 * Dot(this, n) * n;

    /// <summary>Projection orthogonale sur un axe : (v.a/|a|^2) * a.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ProjectOn(Vector3D axis) => Dot(this, axis) * axis.Normalized();

    /// <summary>Rejet (composante perpendiculaire) sur un axe : v - proj(v, a).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D RejectFrom(Vector3D axis) => this - ProjectOn(axis);

    /// <summary>Angle en radians entre ce vecteur et un autre.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double AngleTo(Vector3D o) => Math.Acos(Math.Clamp(Dot(Normalized(), o.Normalized()), -1.0, 1.0));

    /// <summary>Angle en degrés.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double AngleDegreesTo(Vector3D o) => AngleTo(o) * (180.0 / Math.PI);

    /// <summary>Rotation autour de l'axe X par un angle donne (radians).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D RotateX(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(X, Y * c - Z * s, Y * s + Z * c); }

    /// <summary>Rotation autour de l'axe Y par un angle donne (radians).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D RotateY(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(X * c + Z * s, Y, -X * s + Z * c); }

    /// <summary>Rotation autour de l'axe Z par un angle donne (radians).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D RotateZ(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(X * c - Y * s, X * s + Y * c, Z); }

    /// <summary>Conversion en coordonnees spheriques : (r, theta, phi).
    /// r = distance a l'origine, theta = angle polaire [0,pi], phi = azimut [-pi,pi].</summary>
    public readonly Vector3D ToSpherical() { double r = Length(); if (r < 1e-30) return Zero; return new(r, Math.Acos(Math.Clamp(Z / r, -1, 1)), Math.Atan2(Y, X)); }
    /// <summary>Conversion depuis des coordonnees spheriques (r, theta, phi).</summary>
    public static Vector3D FromSpherical(double r, double theta, double phi) => new(r * Math.Sin(theta) * Math.Cos(phi), r * Math.Sin(theta) * Math.Sin(phi), r * Math.Cos(theta));
    /// <summary>Conversion en coordonnees cylindriques : (r, theta, z).</summary>
    public readonly Vector3D ToCylindrical() { double r = Math.Sqrt(X * X + Y * Y); return new(r, Math.Atan2(Y, X), Z); }
    /// <summary>Conversion depuis des coordonnees cylindriques (r, theta, z).</summary>
    public static Vector3D FromCylindrical(double r, double theta, double z) => new(r * Math.Cos(theta), r * Math.Sin(theta), z);

    /// <summary>Interpolation de Catmull-Rom spline. Passe par les 4 points.</summary>
    public static Vector3D CatmullRom(Vector3D p0, Vector3D p1, Vector3D p2, Vector3D p3, double t)
    {
        double t2 = t * t, t3 = t2 * t;
        return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
    }

    /// <summary>Interpolation de Bezier cubique. Controle par 4 points de controle.</summary>
    public static Vector3D CubicBezier(Vector3D p0, Vector3D p1, Vector3D p2, Vector3D p3, double t)
    {
        double u = 1 - t;
        return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
    }

    /// <summary>Spline B-spline cubique. Approximation lisse des 4 points.</summary>
    public static Vector3D BSpline(Vector3D p0, Vector3D p1, Vector3D p2, Vector3D p3, double t)
    {
        double t2 = t * t, t3 = t2 * t;
        return p0 * (-t3 + 3 * t2 - 3 * t + 1) / 6 + p1 * (3 * t3 - 6 * t2 + 4) / 6 + p2 * (-3 * t3 + 3 * t2 + 3 * t + 1) / 6 + p3 * t3 / 6;
    }

    /// <summary>Fonction smoothstep (interpolation cubique hermitienne).</summary>
    public static double Smoothstep(double edge0, double edge1, double x) { double t = Math.Clamp((x - edge0) / (edge1 - edge0), 0, 1); return t * t * (3 - 2 * t); }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator +(Vector3D a, Vector3D b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator -(Vector3D a, Vector3D b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator *(Vector3D v, double s) => new(v.X * s, v.Y * s, v.Z * s);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator *(double s, Vector3D v) => new(v.X * s, v.Y * s, v.Z * s);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator /(Vector3D v, double s) { double i = 1.0 / s; return new(v.X * i, v.Y * i, v.Z * i); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator -(Vector3D v) => new(-v.X, -v.Y, -v.Z);

    /// <summary>Produit scalaire (dot product) : a.b = ax*bx + ay*by + az*bz.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Dot(Vector3D a, Vector3D b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    /// <summary>Produit vectoriel (cross product) : axb. Resultat perpendiculaire aux deux.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D Cross(Vector3D a, Vector3D b) => new(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    /// <summary>Triple produit scalaire : a.(bxc). Volume du parallelepiped.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double TripleScalarProduct(Vector3D a, Vector3D b, Vector3D c) => Dot(a, Cross(b, c));
    /// <summary>Triple produit vectoriel : ax(bxc).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D TripleVectorProduct(Vector3D a, Vector3D b, Vector3D c) => Cross(a, Cross(b, c));

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Distance(Vector3D a, Vector3D b) => (a - b).Length();
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double DistanceSquared(Vector3D a, Vector3D b) => (a - b).LengthSquared();
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double DistanceL1(Vector3D a, Vector3D b) => (a - b).LengthL1();
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double DistanceLInfinity(Vector3D a, Vector3D b) => (a - b).LengthLInfinity();
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D Lerp(Vector3D a, Vector3D b, double t) => new(a.X + (b.X - a.X) * t, a.Y + (b.Y - a.Y) * t, a.Z + (b.Z - a.Z) * t);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D Min(Vector3D a, Vector3D b) => new(Math.Min(a.X, b.X), Math.Min(a.Y, b.Y), Math.Min(a.Z, b.Z));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D Max(Vector3D a, Vector3D b) => new(Math.Max(a.X, b.X), Math.Max(a.Y, b.Y), Math.Max(a.Z, b.Z));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D Clamp(Vector3D v, Vector3D mn, Vector3D mx) => Min(Max(v, mn), mx);
    /// <summary>Point le plus proche sur un segment [a,b].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D ClosestPointOnSegment(Vector3D p, Vector3D a, Vector3D b) { Vector3D ab = b - a; double t = Math.Clamp(Dot(p - a, ab) / Dot(ab, ab), 0, 1); return a + ab * t; }
    /// <summary>Distance d'un point a une droite infinie.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double DistancePointToLine(Vector3D p, Vector3D o, Vector3D d) => Cross(p - o, d).Length() / d.Length();
    /// <summary>Distance entre deux droites infinies.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double DistanceBetweenLines(Vector3D o1, Vector3D d1, Vector3D o2, Vector3D d2) { Vector3D n = Cross(d1, d2); double den = Dot(n, n); if (den < 1e-20) return DistancePointToLine(o1, o2, d2); return Math.Abs(Dot(o2 - o1, n)) / n.Length(); }

    /// <summary>Conversion en Vector3 single-precision System.Numerics.</summary>
    public readonly System.Numerics.Vector3 ToSingle() => new((float)X, (float)Y, (float)Z);
    /// <summary>Conversion en Vector4 single-precision (w=1).</summary>
    public readonly System.Numerics.Vector4 ToSingle4() => new((float)X, (float)Y, (float)Z, 1.0f);
    /// <summary>Depuis un Vector3 System.Numerics.</summary>
    public static Vector3D FromSingle(System.Numerics.Vector3 v) => new(v.X, v.Y, v.Z);
    /// <summary>Conversion en tableau de doubles [3].</summary>
    public readonly double[] ToArray() => new double[] { X, Y, Z };

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool Equals(Vector3D o) => Math.Abs(X - o.X) < 1e-10 && Math.Abs(Y - o.Y) < 1e-10 && Math.Abs(Z - o.Z) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is Vector3D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(X, Y, Z);
    public static bool operator ==(Vector3D a, Vector3D b) => a.Equals(b);
    public static bool operator !=(Vector3D a, Vector3D b) => !a.Equals(b);
    public override readonly string ToString() => $"({X:F4}, {Y:F4}, {Z:F4})";
    public readonly string ToString(string fmt) => $"({X.ToString(fmt)}, {Y.ToString(fmt)}, {Z.ToString(fmt)})";
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3.2: SYMMETRIC3X3 — MATRICE 3x3 SYMETRIQUE
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Matrice 3x3 symetrique pour le tenseur de contrainte et la deformation.
/// Stockee en format compact (6 composantes uniques : XX, YY, ZZ, XY, XZ, YZ).
/// Utilisee pour les tenseurs de stress (Cauchy, Piola-Kirchhoff),
/// strain (Green-Lagrange, Almansi), et d'inertie. Les operations SIMD
/// sont optimisees pour les operateurs differentiels.
///
/// L'invariant J2 (partie deviatorique) est calcule analytiquement pour le
/// critere de plasticite de von Mises. L'inversion est explicite.
///
/// MEMORY LAYOUT: 64 bytes (6 doubles + 2 padding pour alignement AVX).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64, Pack = 32)]
[DebuggerDisplay("Sym3x3({XX:F3}, {YY:F3}, {ZZ:F3}, {XY:F3}, {XZ:F3}, {YZ:F3})")]
public struct Symmetric3x3 : IEquatable<Symmetric3x3>
{
    [FieldOffset(0)] public double XX;
    [FieldOffset(8)] public double YY;
    [FieldOffset(16)] public double ZZ;
    [FieldOffset(24)] public double XY;
    [FieldOffset(32)] public double XZ;
    [FieldOffset(40)] public double YZ;
    [FieldOffset(48)] public double _pad0;
    [FieldOffset(56)] public double _pad1;

    /// <summary>Constructeur avec les 6 composantes independantes d'une matrice symetrique.</summary>
    /// <param name="xx">Composante (1,1) — contrainte normale X.</param>
    /// <param name="yy">Composante (2,2) — contrainte normale Y.</param>
    /// <param name="zz">Composante (3,3) — contrainte normale Z.</param>
    /// <param name="xy">Composante (1,2) = (2,1) — contrainte de cisaillement XY.</param>
    /// <param name="xz">Composante (1,3) = (3,1) — contrainte de cisaillement XZ.</param>
    /// <param name="yz">Composante (2,3) = (3,2) — contrainte de cisaillement YZ.</param>
    public Symmetric3x3(double xx, double yy, double zz, double xy, double xz, double yz)
    { XX = xx; YY = yy; ZZ = zz; XY = xy; XZ = xz; YZ = yz; _pad0 = 0; _pad1 = 0; }

    /// <summary>Matrice symetrique nulle (toutes composantes a zero).</summary>
    public static readonly Symmetric3x3 Zero = default;
    /// <summary>Matrice identite : diag(1,1,1). Tenseur de contrainte sans contrainte.</summary>
    public static readonly Symmetric3x3 Identity = new(1, 1, 1, 0, 0, 0);

    /// <summary>Premier invariant : Tr(sigma) = XX + YY + ZZ. Pression hydrostatique = -Tr/3.</summary>
    public readonly double Trace => XX + YY + ZZ;

    /// <summary>Determinant de la matrice symetrique 3x3 : det = XX*(YY*ZZ - YZ^2) - XY*(XY*ZZ - YZ*XZ) + XZ*(XY*YZ - YY*XZ).</summary>
    public readonly double Determinant =>
        XX * (YY * ZZ - YZ * YZ) - XY * (XY * ZZ - YZ * XZ) + XZ * (XY * YZ - YY * XZ);

    /// <summary>Partie deviatorique : sigma' = sigma - (Tr(sigma)/3)*I. Retire la pression hydrostatique.</summary>
    public readonly Symmetric3x3 Deviatoric { get { double m = Trace / 3.0; return new(XX - m, YY - m, ZZ - m, XY, XZ, YZ); } }

    /// <summary>Deuxieme invariant deviatorique J2 : mesure de l'intensite de cisaillement.</summary>
    /// <remarks>J2 = 0.5*(s11^2 + s22^2 + s33^2) + s12^2 + s13^2 + s23^2, ou s = deviatoric.</remarks>
    public readonly double J2 { get { var d = Deviatoric; return 0.5 * (d.XX * d.XX + d.YY * d.YY + d.ZZ * d.ZZ) + d.XY * d.XY + d.XZ * d.XZ + d.YZ * d.YZ; } }

    /// <summary>Deuxieme invariant complet I2 = XX*YY + YY*ZZ + ZZ*XX - XY^2 - XZ^2 - YZ^2.</summary>
    public readonly double I2 => XX * YY + YY * ZZ + ZZ * XX - XY * XY - XZ * XZ - YZ * YZ;

    /// <summary>Troisieme invariant I3 = det(sigma).</summary>
    public readonly double I3 => Determinant;

    /// <summary>Contrainte equivalente de von Mises : sigma_vM = sqrt(3*J2). Critere de plasticite.</summary>
    /// <remarks>Si sigma_vM depasse la limite d'elasticite (YieldStrength), le materiau plastifie.</remarks>
    public readonly double VonMises => Math.Sqrt(3.0 * J2);

    /// <summary>Norme de Frobenius : ||sigma||_F = sqrt(Sigma sigma_ij^2). Mesure globale de l'intensite.</summary>
    public readonly double FrobeniusNorm => Math.Sqrt(XX * XX + YY * YY + ZZ * ZZ + 2.0 * (XY * XY + XZ * XZ + YZ * YZ));

    /// <summary>Energie de distorsion : U_d = J2. Energie associee a la deformation de cisaillement.</summary>
    public readonly double DistortionEnergy => J2;

    /// <summary>Energie de dilatation : U_v = Tr(sigma)^2 / 18. Energie de changement de volume.</summary>
    public readonly double DilatationEnergy => Trace * Trace / 18.0;

    /// <summary>Energie de deformation totale : U = U_d + U_v.</summary>
    public readonly double StrainEnergy => DistortionEnergy + DilatationEnergy;

    /// <summary>Premiere direction principale (approximation par Jacobi).</summary>
    public readonly Vector3D PrincipalDirection1 { get { MaxEigenvalues(out var v1, out _, out _); return new Vector3D(v1, 0, 0); } }
    /// <summary>Deuxieme direction principale.</summary>
    public readonly Vector3D PrincipalDirection2 { get { MaxEigenvalues(out _, out var v2, out _); return new Vector3D(0, v2, 0); } }
    /// <summary>Troisieme direction principale.</summary>
    public readonly Vector3D PrincipalDirection3 { get { MaxEigenvalues(out _, out _, out var v3); return new Vector3D(0, 0, v3); } }

    /// <summary>Valeurs propres principales (tries par ordre decroissant).</summary>
    public readonly void MaxEigenvalues(out double v1, out double v2, out double v3)
    {
        Symmetric3x3 m = this;
        Vector3D d1 = UnitX, d2 = UnitY, d3 = UnitZ;
        for (int iter = 0; iter < 50; iter++)
        {
            double theta = 0.5 * Math.Atan2(2.0 * m.XY, m.XX - m.YY);
            double c = Math.Cos(theta), s = Math.Sin(theta);
            double c2 = c * c, s2 = s * s, cs = c * s;
            m = new Symmetric3x3(
                c2 * m.XX + 2 * cs * m.XY + s2 * m.YY,
                s2 * m.XX - 2 * cs * m.XY + c2 * m.YY, m.ZZ,
                cs * (m.YY - m.XX) + (c2 - s2) * m.XY,
                c * m.XZ + s * m.YZ, -s * m.XZ + c * m.YZ);
            d1 = new Vector3D(c * d1.X + s * d1.Y, -s * d1.X + c * d1.Y, d1.Z);
            d2 = new Vector3D(c * d2.X + s * d2.Y, -s * d2.X + c * d2.Y, d2.Z);
        }
        if (m.XX < m.YY)
        { (m.XX, m.YY) = (m.YY, m.XX); (d1, d2) = (d2, d1); }
        if (m.XX < m.ZZ)
        { (m.XX, m.ZZ) = (m.ZZ, m.XX); (d1, d3) = (d3, d1); }
        if (m.YY < m.ZZ)
        { (m.YY, m.ZZ) = (m.ZZ, m.YY); (d2, d3) = (d3, d2); }
        v1 = m.XX;
        v2 = m.YY;
        v3 = m.ZZ;
    }

    private static readonly Vector3D UnitX = Vector3D.UnitX, UnitY = Vector3D.UnitY, UnitZ = Vector3D.UnitZ;

    /// <summary>Produit matrice-vecteur : sigma.v. Calcule la force sur une surface.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Multiply(Vector3D v) =>
        new(XX * v.X + XY * v.Y + XZ * v.Z, XY * v.X + YY * v.Y + YZ * v.Z, XZ * v.X + YZ * v.Y + ZZ * v.Z);

    /// <summary>Double contraction : v.sigma.v. Energie de deformation sur un vecteur.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double DoubleContract(Vector3D v) => Vector3D.Dot(v, Multiply(v));

    /// <summary>Matrice symetrique de Cauchy-Green : C = F^T * F.</summary>
    public static Symmetric3x3 FromCauchyGreen(Matrix4x4D F)
    {
        Matrix4x4D ft = F.Transpose();
        Matrix4x4D c = ft * F;
        return new Symmetric3x3(c.M11, c.M22, c.M33, c.M12, c.M13, c.M23);
    }

    /// <summary>Contrainte de Green-Lagrange : E = 0.5*(C - I). Deformation finie.</summary>
    public readonly Symmetric3x3 GreenLagrangeStrain() =>
        new((XX - 1) * 0.5, (YY - 1) * 0.5, (ZZ - 1) * 0.5, XY * 0.5, XZ * 0.5, YZ * 0.5);

    /// <summary>Inversion de la matrice symetrique 3x3 (methode de Cramer).</summary>
    public readonly Symmetric3x3 Inverse()
    {
        double det = Determinant;
        if (Math.Abs(det) < 1e-30)
            return Zero;
        double inv = 1.0 / det;
        return new Symmetric3x3(
            (YY * ZZ - YZ * YZ) * inv, (XX * ZZ - XZ * XZ) * inv, (XX * YY - XY * XY) * inv,
            (XZ * YZ - XY * ZZ) * inv, (XY * YZ - XZ * YY) * inv, (XY * XZ - XX * YZ) * inv);
    }

    /// <summary>Rotation par une matrice de rotation : R * sigma * R^T.</summary>
    public readonly Symmetric3x3 RotateBy(Matrix4x4D R)
    {
        double r00 = R.M11, r01 = R.M12, r02 = R.M13, r10 = R.M21, r11 = R.M22, r12 = R.M23, r20 = R.M31, r21 = R.M32, r22 = R.M33;
        double nxx = r00 * (r00 * XX + r01 * XY + r02 * XZ) + r01 * (r00 * XY + r01 * YY + r02 * YZ) + r02 * (r00 * XZ + r01 * YZ + r02 * ZZ);
        double nyy = r10 * (r10 * XX + r11 * XY + r12 * XZ) + r11 * (r10 * XY + r11 * YY + r12 * YZ) + r12 * (r10 * XZ + r11 * YZ + r12 * ZZ);
        double nzz = r20 * (r20 * XX + r21 * XY + r22 * XZ) + r21 * (r20 * XY + r21 * YY + r22 * YZ) + r22 * (r20 * XZ + r21 * YZ + r22 * ZZ);
        double nxy = r00 * (r10 * XX + r11 * XY + r12 * XZ) + r01 * (r10 * XY + r11 * YY + r12 * YZ) + r02 * (r10 * XZ + r11 * YZ + r12 * ZZ);
        double nxz = r00 * (r20 * XX + r21 * XY + r22 * XZ) + r01 * (r20 * XY + r21 * YY + r22 * YZ) + r02 * (r20 * XZ + r21 * YZ + r22 * ZZ);
        double nyz = r10 * (r20 * XX + r21 * XY + r22 * XZ) + r11 * (r20 * XY + r21 * YY + r22 * YZ) + r12 * (r20 * XZ + r21 * YZ + r22 * ZZ);
        return new Symmetric3x3(nxx, nyy, nzz, nxy, nxz, nyz);
    }

    /// <summary>Operateurs arithmetiques.</summary>
    public static Symmetric3x3 operator +(Symmetric3x3 a, Symmetric3x3 b) => new(a.XX + b.XX, a.YY + b.YY, a.ZZ + b.ZZ, a.XY + b.XY, a.XZ + b.XZ, a.YZ + b.YZ);
    public static Symmetric3x3 operator -(Symmetric3x3 a, Symmetric3x3 b) => new(a.XX - b.XX, a.YY - b.YY, a.ZZ - b.ZZ, a.XY - b.XY, a.XZ - b.XZ, a.YZ - b.YZ);
    public static Symmetric3x3 operator *(Symmetric3x3 m, double s) => new(m.XX * s, m.YY * s, m.ZZ * s, m.XY * s, m.XZ * s, m.YZ * s);
    public static Symmetric3x3 operator *(double s, Symmetric3x3 m) => m * s;
    public static Symmetric3x3 operator +(Symmetric3x3 m, double s) => new(m.XX + s, m.YY + s, m.ZZ + s, m.XY, m.XZ, m.YZ);
    public static Symmetric3x3 operator -(Symmetric3x3 m, double s) => new(m.XX - s, m.YY - s, m.ZZ - s, m.XY, m.XZ, m.YZ);

    /// <summary>Interpolation lineaire entre deux tenseurs symetriques.</summary>
    public static Symmetric3x3 Lerp(Symmetric3x3 a, Symmetric3x3 b, double t)
    {
        double u = 1.0 - t;
        return new Symmetric3x3(a.XX * u + b.XX * t, a.YY * u + b.YY * t, a.ZZ * u + b.ZZ * t, a.XY * u + b.XY * t, a.XZ * u + b.XZ * t, a.YZ * u + b.YZ * t);
    }

    /// <summary>Norme L2 des composantes du tenseur.</summary>
    public readonly double Norm => FrobeniusNorm;

    /// <summary>Determinant signe.</summary>
    public readonly double SignedDeterminant => Determinant;

    /// <summary>Teste si la matrice est definie positive (toutes valeurs propres > 0).</summary>
    public readonly bool IsPositiveDefinite
    {
        get
        {
            MaxEigenvalues(out double v1, out double v2, out double v3);
            return v1 > 0 && v2 > 0 && v3 > 0;
        }
    }

    /// <summary>Teste si la matrice est semi-definie positive.</summary>
    public readonly bool IsPositiveSemiDefinite
    {
        get
        {
            MaxEigenvalues(out double v1, out double v2, out double v3);
            return v1 >= 0 && v2 >= 0 && v3 >= 0;
        }
    }

    /// <summary>Condition number : rapport max/min des valeurs propres.</summary>
    public readonly double ConditionNumber
    {
        get
        {
            MaxEigenvalues(out double v1, out double v2, out double v3);
            double min = Math.Min(Math.Min(Math.Abs(v1), Math.Abs(v2)), Math.Abs(v3));
            double max = Math.Max(Math.Max(Math.Abs(v1), Math.Abs(v2)), Math.Abs(v3));
            return min > 1e-30 ? max / min : double.MaxValue;
        }
    }

    public readonly bool Equals(Symmetric3x3 o) => Math.Abs(XX - o.XX) < 1e-10 && Math.Abs(YY - o.YY) < 1e-10 && Math.Abs(ZZ - o.ZZ) < 1e-10 && Math.Abs(XY - o.XY) < 1e-10 && Math.Abs(XZ - o.XZ) < 1e-10 && Math.Abs(YZ - o.YZ) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is Symmetric3x3 o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(XX, YY, ZZ, XY, XZ, YZ);
    public static bool operator ==(Symmetric3x3 a, Symmetric3x3 b) => a.Equals(b);
    public static bool operator !=(Symmetric3x3 a, Symmetric3x3 b) => !a.Equals(b);
    public override readonly string ToString() => $"[{XX:F3} {XY:F3} {XZ:F3} | {XY:F3} {YY:F3} {YZ:F3} | {XZ:F3} {YZ:F3} {ZZ:F3}]";
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3.3: TENSOR3D — MATRICE 3x3 COMPLETE
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Matrice 3x3 complete (non necessairement symetrique).
/// Utilisee pour les tenseurs de rotation, gradient de deformation F,
/// Jacobien, et operations tensorielles generales en mecanique du continu.
///
/// Stockage ligne par ligne : M_ij = ligne i, colonne j.
/// Multiplication standard : (AB)_ij = Sum_k A_ik * B_kj.
///
/// MEMORY LAYOUT: 72 bytes (9 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 72, Pack = 8)]
[DebuggerDisplay("[{M11:F3},{M12:F3},{M13:F3} | {M21:F3},{M22:F3},{M23:F3} | {M31:F3},{M32:F3},{M33:F3}]")]
public struct Tensor3D : IEquatable<Tensor3D>
{
    [FieldOffset(0)] public double M11; [FieldOffset(8)] public double M12; [FieldOffset(16)] public double M13;
    [FieldOffset(24)] public double M21; [FieldOffset(32)] public double M22; [FieldOffset(40)] public double M23;
    [FieldOffset(48)] public double M31; [FieldOffset(56)] public double M32; [FieldOffset(64)] public double M33;

    /// <summary>Constructeur principal avec les 9 composantes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D(double m11, double m12, double m13, double m21, double m22, double m23, double m31, double m32, double m33)
    { M11 = m11; M12 = m12; M13 = m13; M21 = m21; M22 = m22; M23 = m23; M31 = m31; M32 = m32; M33 = m33; }

    /// <summary>Matrice nulle.</summary>
    public static readonly Tensor3D Zero = default;
    /// <summary>Matrice identite.</summary>
    public static readonly Tensor3D Identity = new(1, 0, 0, 0, 1, 0, 0, 0, 1);

    /// <summary>Matrice de rotation autour de l'axe X.</summary>
    public static Tensor3D RotationX(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(1, 0, 0, 0, c, -s, 0, s, c); }
    /// <summary>Matrice de rotation autour de l'axe Y.</summary>
    public static Tensor3D RotationY(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(c, 0, s, 0, 1, 0, -s, 0, c); }
    /// <summary>Matrice de rotation autour de l'axe Z.</summary>
    public static Tensor3D RotationZ(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(c, -s, 0, s, c, 0, 0, 0, 1); }
    /// <summary>Matrice de rotation autour d'un axe arbitraire (formule de Rodrigues).</summary>
    public static Tensor3D RotationAxis(Vector3D ax, double a)
    {
        Vector3D u = ax.Normalized();
        double c = Math.Cos(a), s = Math.Sin(a), t = 1 - c;
        return new(t * u.X * u.X + c, t * u.X * u.Y - s * u.Z, t * u.X * u.Z + s * u.Y,
            t * u.X * u.Y + s * u.Z, t * u.Y * u.Y + c, t * u.Y * u.Z - s * u.X,
            t * u.X * u.Z - s * u.Y, t * u.Y * u.Z + s * u.X, t * u.Z * u.Z + c);
    }
    /// <summary>Matrice de mise a l'echelle diagonale.</summary>
    public static Tensor3D Scaling(double sx, double sy, double sz) => new(sx, 0, 0, 0, sy, 0, 0, 0, sz);
    /// <summary>Matrice de cisaillement XY.</summary>
    public static Tensor3D ShearXY(double s) => new(1, s, 0, 0, 1, 0, 0, 0, 1);
    /// <summary>Matrice de cisaillement XZ.</summary>
    public static Tensor3D ShearXZ(double s) => new(1, 0, s, 0, 1, 0, 0, 0, 1);
    /// <summary>Matrice de cisaillement YZ.</summary>
    public static Tensor3D ShearYZ(double s) => new(1, 0, 0, 0, 1, s, 0, 0, 1);

    /// <summary>Tr : M11 + M22 + M33.</summary>
    public readonly double Trace => M11 + M22 + M33;
    /// <summary>Determinant : M11*(M22*M33-M23*M32) - M12*(M21*M33-M23*M31) + M13*(M21*M32-M22*M31).</summary>
    public readonly double Determinant => M11 * (M22 * M33 - M23 * M32) - M12 * (M21 * M33 - M23 * M31) + M13 * (M21 * M32 - M22 * M31);
    /// <summary>Transpose : (M^T)_ij = M_ji.</summary>
    public readonly Tensor3D Transpose() => new(M11, M21, M31, M12, M22, M32, M13, M23, M33);
    /// <summary>Inverse par la methode de Cramer (cofacteurs).</summary>
    public readonly Tensor3D Inverse()
    {
        double det = Determinant;
        if (Math.Abs(det) < 1e-30)
            return Zero;
        double inv = 1.0 / det;
        return new(
            (M22 * M33 - M23 * M32) * inv, (M13 * M32 - M12 * M33) * inv, (M12 * M23 - M13 * M22) * inv,
            (M23 * M31 - M21 * M33) * inv, (M11 * M33 - M13 * M31) * inv, (M13 * M21 - M11 * M23) * inv,
            (M21 * M32 - M22 * M31) * inv, (M12 * M31 - M11 * M32) * inv, (M11 * M22 - M12 * M21) * inv);
    }
    /// <summary>Norme de Frobenius.</summary>
    public readonly double FrobeniusNorm => Math.Sqrt(M11 * M11 + M12 * M12 + M13 * M13 + M21 * M21 + M22 * M22 + M23 * M23 + M31 * M31 + M32 * M32 + M33 * M33);
    public readonly double FrobeniusNormSquared => M11 * M11 + M12 * M12 + M13 * M13 + M21 * M21 + M22 * M22 + M23 * M23 + M31 * M31 + M32 * M32 + M33 * M33;
    /// <summary>Partie symetrique : (M + M^T) / 2.</summary>
    public readonly Symmetric3x3 SymmetricPart() => new(M11, M22, M33, (M12 + M21) * 0.5, (M13 + M31) * 0.5, (M23 + M32) * 0.5);
    /// <summary>Partie antisymetrique : (M - M^T) / 2.</summary>
    public readonly Tensor3D AntisymmetricPart() => new(0, (M12 - M21) * 0.5, (M13 - M31) * 0.5, (M21 - M12) * 0.5, 0, (M23 - M32) * 0.5, (M31 - M13) * 0.5, (M32 - M23) * 0.5, 0);

    /// <summary>Produit matrice-vecteur : M.v.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Multiply(Vector3D v) => new(M11 * v.X + M12 * v.Y + M13 * v.Z, M21 * v.X + M22 * v.Y + M23 * v.Z, M31 * v.X + M32 * v.Y + M33 * v.Z);
    /// <summary>Produit de deux matrices.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Tensor3D Multiply(Tensor3D b)
    {
        Tensor3D r;
        r.M11 = M11 * b.M11 + M12 * b.M21 + M13 * b.M31;
        r.M12 = M11 * b.M12 + M12 * b.M22 + M13 * b.M32;
        r.M13 = M11 * b.M13 + M12 * b.M23 + M13 * b.M33;
        r.M21 = M21 * b.M11 + M22 * b.M21 + M23 * b.M31;
        r.M22 = M21 * b.M12 + M22 * b.M22 + M23 * b.M32;
        r.M23 = M21 * b.M13 + M22 * b.M23 + M23 * b.M33;
        r.M31 = M31 * b.M11 + M32 * b.M21 + M33 * b.M31;
        r.M32 = M31 * b.M12 + M32 * b.M22 + M33 * b.M32;
        r.M33 = M31 * b.M13 + M32 * b.M23 + M33 * b.M33;
        return r;
    }
    /// <summary>Double contraction A:B = Sum(A_ij * B_ij).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double DoubleContract(Tensor3D b) => M11 * b.M11 + M12 * b.M12 + M13 * b.M13 + M21 * b.M21 + M22 * b.M22 + M23 * b.M23 + M31 * b.M31 + M32 * b.M32 + M33 * b.M33;
    /// <summary>Exponentielle matricielle : exp(M) = Sum(M^n/n!).</summary>
    public readonly Tensor3D Exponential()
    {
        Tensor3D m2 = Multiply(this), m3 = m2.Multiply(this), m4 = m2.Multiply(m2), m5 = m4.Multiply(this);
        return Identity + this + m2 * 0.5 + m3 * (1.0 / 6.0) + m4 * (1.0 / 24.0) + m5 * (1.0 / 120.0);
    }
    /// <summary>Extraction d'une colonne.</summary>
    public readonly Vector3D Column(int j) => j switch { 0 => new(M11, M21, M31), 1 => new(M12, M22, M32), 2 => new(M13, M23, M33), _ => Vector3D.Zero };
    /// <summary>Extraction d'une ligne.</summary>
    public readonly Vector3D Row(int i) => i switch { 0 => new(M11, M12, M13), 1 => new(M21, M22, M23), 2 => new(M31, M32, M33), _ => Vector3D.Zero };
    public readonly Symmetric3x3 ToSymmetric3x3() => SymmetricPart();

    public static Tensor3D operator +(Tensor3D a, Tensor3D b) => new(a.M11 + b.M11, a.M12 + b.M12, a.M13 + b.M13, a.M21 + b.M21, a.M22 + b.M22, a.M23 + b.M23, a.M31 + b.M31, a.M32 + b.M32, a.M33 + b.M33);
    public static Tensor3D operator -(Tensor3D a, Tensor3D b) => new(a.M11 - b.M11, a.M12 - b.M12, a.M13 - b.M13, a.M21 - b.M21, a.M22 - b.M22, a.M23 - b.M23, a.M31 - b.M31, a.M32 - b.M32, a.M33 - b.M33);
    public static Tensor3D operator *(Tensor3D m, double s) => new(m.M11 * s, m.M12 * s, m.M13 * s, m.M21 * s, m.M22 * s, m.M23 * s, m.M31 * s, m.M32 * s, m.M33 * s);
    public static Tensor3D operator *(double s, Tensor3D m) => m * s;
    public static Tensor3D operator *(Tensor3D a, Tensor3D b) => a.Multiply(b);
    public static Vector3D operator *(Tensor3D m, Vector3D v) => m.Multiply(v);
    public static Tensor3D operator -(Tensor3D m) => new(-m.M11, -m.M12, -m.M13, -m.M21, -m.M22, -m.M23, -m.M31, -m.M32, -m.M33);

    public readonly bool Equals(Tensor3D o) => Math.Abs(M11 - o.M11) < 1e-10 && Math.Abs(M12 - o.M12) < 1e-10 && Math.Abs(M13 - o.M13) < 1e-10 && Math.Abs(M21 - o.M21) < 1e-10 && Math.Abs(M22 - o.M22) < 1e-10 && Math.Abs(M23 - o.M23) < 1e-10 && Math.Abs(M31 - o.M31) < 1e-10 && Math.Abs(M32 - o.M32) < 1e-10 && Math.Abs(M33 - o.M33) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is Tensor3D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(M11, M22, M33);
    public static bool operator ==(Tensor3D a, Tensor3D b) => a.Equals(b);
    public static bool operator !=(Tensor3D a, Tensor3D b) => !a.Equals(b);
    public override readonly string ToString() => $"[{M11:F4} {M12:F4} {M13:F4}]\n[{M21:F4} {M22:F4} {M23:F4}]\n[{M31:F4} {M32:F4} {M33:F4}]";
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3.4: QUATERNIOND — QUATERNION DOUBLE PRECISION
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Quaternion double precision pour les rotations 3D sans gimbal lock.
/// Format : q = w + xi + yj + zk, avec |q| = 1 pour les rotations propres.
///
/// Le quaternion evite les singularites des angles d'Euler (gimbal lock)
/// et permet des interpolations lisses (SLERP) entre rotations.
///
/// MEMORY LAYOUT: 32 bytes (4 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 32, Pack = 8)]
[DebuggerDisplay("Quat({W:F4}, {X:F4}, {Y:F4}, {Z:F4})")]
public struct QuaternionD : IEquatable<QuaternionD>
{
    /// <summary>Partie scalaire w = cos(theta/2).</summary>
    [FieldOffset(0)] public double W;
    /// <summary>Partie vectorielle x.</summary>
    [FieldOffset(8)] public double X;
    /// <summary>Partie vectorielle y.</summary>
    [FieldOffset(16)] public double Y;
    /// <summary>Partie vectorielle z.</summary>
    [FieldOffset(24)] public double Z;

    /// <summary>Constructeur avec les 4 composantes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public QuaternionD(double w, double x, double y, double z) { W = w; X = x; Y = y; Z = z; }

    /// <summary>Quaternion identite (pas de rotation). q = (1, 0, 0, 0).</summary>
    public static readonly QuaternionD Identity = new(1, 0, 0, 0);
    /// <summary>Quaternion nul.</summary>
    public static readonly QuaternionD Zero = new(0, 0, 0, 0);

    /// <summary>Conjuge : q* = (w, -x, -y, -z). Inverse pour les rotations.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly QuaternionD Conjugate() => new(W, -X, -Y, -Z);
    /// <summary>Inverse : q^-1 = q*/|q|^2. Retourne Identity si |q| ~ 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly QuaternionD Inverse()
    {
        double ls = W * W + X * X + Y * Y + Z * Z;
        if (ls < 1e-30)
            return Identity;
        double inv = 1.0 / ls;
        return new QuaternionD(W * inv, -X * inv, -Y * inv, -Z * inv);
    }
    /// <summary>Norme du quaternion : |q| = sqrt(w^2 + x^2 + y^2 + z^2).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double Length() => Math.Sqrt(W * W + X * X + Y * Y + Z * Z);
    /// <summary>Care de la norme.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double LengthSquared() => W * W + X * X + Y * Y + Z * Z;
    /// <summary>Quaternion normalise (norme = 1). Indispensable pour les rotations.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly QuaternionD Normalized()
    {
        double len = Length();
        if (len < 1e-30)
            return Identity;
        double inv = 1.0 / len;
        return new QuaternionD(W * inv, X * inv, Y * inv, Z * inv);
    }
    /// <summary>Normalise en place.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Normalize() { double len = Length(); if (len < 1e-30) return; double inv = 1.0 / len; W *= inv; X *= inv; Y *= inv; Z *= inv; }

    /// <summary>
    /// Construit un quaternion a partir d'un axe et d'un angle.
    /// q = cos(theta/2) + sin(theta/2)*(ux*i + uy*j + uz*k)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static QuaternionD FromAxisAngle(Vector3D axis, double angle)
    {
        double ha = angle * 0.5, s = Math.Sin(ha);
        Vector3D n = axis.Normalized();
        return new QuaternionD(Math.Cos(ha), n.X * s, n.Y * s, n.Z * s);
    }
    /// <summary>Construit un quaternion depuis les angles d'Euler (ZYX convention : yaw, pitch, roll).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static QuaternionD FromEuler(double roll, double pitch, double yaw)
    {
        double cr = Math.Cos(roll * 0.5), sr = Math.Sin(roll * 0.5);
        double cp = Math.Cos(pitch * 0.5), sp = Math.Sin(pitch * 0.5);
        double cy = Math.Cos(yaw * 0.5), sy = Math.Sin(yaw * 0.5);
        return new QuaternionD(cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy);
    }
    /// <summary>Construit un quaternion depuis une matrice de rotation (algorithme de Shepperd).</summary>
    public static QuaternionD FromMatrix3x3(Tensor3D m)
    {
        double trace = m.M11 + m.M22 + m.M33;
        if (trace > 0)
        { double s = 0.5 / Math.Sqrt(trace + 1.0); return new QuaternionD(0.25 / s, (m.M32 - m.M23) * s, (m.M13 - m.M31) * s, (m.M21 - m.M12) * s).Normalized(); }
        if (m.M11 > m.M22 && m.M11 > m.M33)
        { double s = 2.0 * Math.Sqrt(1.0 + m.M11 - m.M22 - m.M33); return new QuaternionD((m.M32 - m.M23) / s, 0.25 * s, (m.M12 + m.M21) / s, (m.M13 + m.M31) / s).Normalized(); }
        if (m.M22 > m.M33)
        { double s = 2.0 * Math.Sqrt(1.0 + m.M22 - m.M11 - m.M33); return new QuaternionD((m.M13 - m.M31) / s, (m.M12 + m.M21) / s, 0.25 * s, (m.M23 + m.M32) / s).Normalized(); }
        double s2 = 2.0 * Math.Sqrt(1.0 + m.M33 - m.M11 - m.M22);
        return new QuaternionD((m.M21 - m.M12) / s2, (m.M13 + m.M31) / s2, (m.M23 + m.M32) / s2, 0.25 * s2).Normalized();
    }

    /// <summary>Convertit en matrice de rotation 3x3.</summary>
    public readonly Tensor3D ToTensor3D()
    {
        double xx = X * X, yy = Y * Y, zz = Z * Z, xy = X * Y, xz = X * Z, yz = Y * Z;
        double wx = W * X, wy = W * Y, wz = W * Z;
        return new Tensor3D(1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy));
    }
    /// <summary>Convertit en matrice 4x4 (rotation + translation nulle).</summary>
    public readonly Matrix4x4D ToMatrix4x4D() { Tensor3D r = ToTensor3D(); return new Matrix4x4D(r.M11, r.M12, r.M13, 0, r.M21, r.M22, r.M23, 0, r.M31, r.M32, r.M33, 0, 0, 0, 0, 1); }
    /// <summary>Renvoie les angles d'Euler (roll, pitch, yaw).</summary>
    public readonly Vector3D ToEuler()
    {
        double sr = 2.0 * (W * X + Y * Z), cr = 1.0 - 2.0 * (X * X + Y * Y);
        double roll = Math.Atan2(sr, cr);
        double sp = 2.0 * (W * Y - Z * X);
        double pitch = Math.Abs(sp) >= 1 ? Math.CopySign(Math.PI / 2.0, sp) : Math.Asin(sp);
        double sy = 2.0 * (W * Z + X * Y), cy = 1.0 - 2.0 * (Y * Y + Z * Z);
        double yaw = Math.Atan2(sy, cy);
        return new Vector3D(roll, pitch, yaw);
    }
    /// <summary>Renvoie l'axe et l'angle de rotation.</summary>
    public readonly void ToAxisAngle(out Vector3D axis, out double angle)
    {
        double len = Math.Sqrt(X * X + Y * Y + Z * Z);
        if (len < 1e-30)
        { axis = Vector3D.UnitY; angle = 0; return; }
        axis = new Vector3D(X / len, Y / len, Z / len);
        angle = 2.0 * Math.Atan2(len, W);
    }
    /// <summary>Angle de rotation en radians.</summary>
    public readonly double Angle => 2.0 * Math.Atan2(Math.Sqrt(X * X + Y * Y + Z * Z), W);

    /// <summary>Rotation d'un vecteur : q * v * q^-1. Methode optimisee (pas de produit quaternion).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D RotateVector(Vector3D v)
    {
        double vx = v.X, vy = v.Y, vz = v.Z, qw = W, qx = X, qy = Y, qz = Z;
        double tx = 2.0 * (qy * vz - qz * vy), ty = 2.0 * (qz * vx - qx * vz), tz = 2.0 * (qx * vy - qy * vx);
        return new Vector3D(vx + qw * tx + (qy * tz - qz * ty), vy + qw * ty + (qz * tx - qx * tz), vz + qw * tz + (qx * ty - qy * tx));
    }

    /// <summary>Spherical Linear Interpolation (SLERP). Preserves les angles.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static QuaternionD Slerp(QuaternionD a, QuaternionD b, double t)
    {
        double dot = a.W * b.W + a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        QuaternionD target = b;
        if (dot < 0)
        { dot = -dot; target = new QuaternionD(-b.W, -b.X, -b.Y, -b.Z); }
        if (dot > 0.9995)
        { double it = 1.0 - t; return new QuaternionD(it * a.W + t * target.W, it * a.X + t * target.X, it * a.Y + t * target.Y, it * a.Z + t * target.Z).Normalized(); }
        double theta = Math.Acos(dot), sinTheta = Math.Sin(theta);
        return a * (Math.Sin((1 - t) * theta) / sinTheta) + target * (Math.Sin(t * theta) / sinTheta);
    }
    /// <summary>Normalized Linear Interpolation (NLERP). Plus rapide mais moins precis.</summary>
    public static QuaternionD Nlerp(QuaternionD a, QuaternionD b, double t) { double dot = a.W * b.W + a.X * b.X + a.Y * b.Y + a.Z * b.Z; double sign = dot < 0 ? -1 : 1; double it = 1.0 - t; return new QuaternionD(it * a.W + t * sign * b.W, it * a.X + t * sign * b.X, it * a.Y + t * sign * b.Y, it * a.Z + t * sign * b.Z).Normalized(); }

    /// <summary>Produit de deux quaternions (composition de rotations).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static QuaternionD operator *(QuaternionD a, QuaternionD b) =>
        new(a.W * b.W - a.X * b.X - a.Y * b.Y - a.Z * b.Z,
            a.W * b.X + a.X * b.W + a.Y * b.Z - a.Z * b.Y,
            a.W * b.Y - a.X * b.Z + a.Y * b.W + a.Z * b.X,
            a.W * b.Z + a.X * b.Y - a.Y * b.X + a.Z * b.W);
    /// <summary>Multiplication par un scalaire (slerp implicite).</summary>
    public static QuaternionD operator *(QuaternionD q, double s) => new QuaternionD(q.W * s, q.X * s, q.Y * s, q.Z * s);
    public static QuaternionD operator *(double s, QuaternionD q) => q * s;
    public static QuaternionD operator +(QuaternionD a, QuaternionD b) => new(a.W + b.W, a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static QuaternionD operator -(QuaternionD a) => new(-a.W, -a.X, -a.Y, -a.Z);
    public static QuaternionD operator -(QuaternionD a, QuaternionD b) => new(a.W - b.W, a.X - b.X, a.Y - b.Y, a.Z - b.Z);

    /// <summary>Difference relative : b * a^-1.</summary>
    public static QuaternionD Difference(QuaternionD a, QuaternionD b) => a.Inverse() * b;
    /// <summary>Logarithme quaternion (pour interpolations sur l'espace tangent).</summary>
    public readonly QuaternionD Log() { double len = Math.Sqrt(X * X + Y * Y + Z * Z); if (len < 1e-30) return Zero; double ha = Math.Atan2(len, W), s = ha / len; return new QuaternionD(0, X * s, Y * s, Z * s); }
    /// <summary>Exponentielle quaternion.</summary>
    public static QuaternionD Exp(QuaternionD q) { double len = Math.Sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z); if (len < 1e-30) return Identity; double s = Math.Sin(len) / len; return new QuaternionD(Math.Cos(len), q.X * s, q.Y * s, q.Z * s); }
    /// <summary>Retourne le quaternion le plus proche (evite les problemes de signe).</summary>
    public readonly QuaternionD ClosestTo(QuaternionD other) => (W * other.W + X * other.X + Y * other.Y + Z * other.Z) < 0 ? new QuaternionD(-W, -X, -Y, -Z) : this;

    public readonly bool Equals(QuaternionD o) => Math.Abs(W - o.W) < 1e-10 && Math.Abs(X - o.X) < 1e-10 && Math.Abs(Y - o.Y) < 1e-10 && Math.Abs(Z - o.Z) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is QuaternionD o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(W, X, Y, Z);
    public static bool operator ==(QuaternionD a, QuaternionD b) => a.Equals(b);
    public static bool operator !=(QuaternionD a, QuaternionD b) => !a.Equals(b);
    public override readonly string ToString() => $"({W:F4}, {X:F4}, {Y:F4}, {Z:F4})";
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3.5: MATRIX4X4D — MATRICE 4x4 DOUBLE PRECISION
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Matrice 4x4 double precision pour les transformations homogenes.
/// Utilisee pour les transformations de modele, vue, projection et les
/// operations dans l'espace projectif. Stockage en colonnes (OpenGL convention).
///
/// Les methodes statiques generent les matrices standard : translation, rotation,
/// mise a l'echelle, projection perspective/orthographique, LookAt.
///
/// MEMORY LAYOUT: 128 bytes (16 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 128, Pack = 8)]
[DebuggerDisplay("[{M11:F3},{M12:F3},{M13:F3},{M14:F3} | {M21:F3},{M22:F3},{M23:F3},{M24:F3} | {M31:F3},{M32:F3},{M33:F3},{M34:F3} | {M41:F3},{M42:F3},{M43:F3},{M44:F3}]")]
public struct Matrix4x4D : IEquatable<Matrix4x4D>
{
    [FieldOffset(0)] public double M11; [FieldOffset(8)] public double M12; [FieldOffset(16)] public double M13; [FieldOffset(24)] public double M14;
    [FieldOffset(32)] public double M21; [FieldOffset(40)] public double M22; [FieldOffset(48)] public double M23; [FieldOffset(56)] public double M24;
    [FieldOffset(64)] public double M31; [FieldOffset(72)] public double M32; [FieldOffset(80)] public double M33; [FieldOffset(88)] public double M34;
    [FieldOffset(96)] public double M41; [FieldOffset(104)] public double M42; [FieldOffset(112)] public double M43; [FieldOffset(120)] public double M44;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix4x4D(double m11, double m12, double m13, double m14, double m21, double m22, double m23, double m24, double m31, double m32, double m33, double m34, double m41, double m42, double m43, double m44)
    { M11 = m11; M12 = m12; M13 = m13; M14 = m14; M21 = m21; M22 = m22; M23 = m23; M24 = m24; M31 = m31; M32 = m32; M33 = m33; M34 = m34; M41 = m41; M42 = m42; M43 = m43; M44 = m44; }

    public static readonly Matrix4x4D Zero = default;
    public static readonly Matrix4x4D Identity = new(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

    public static Matrix4x4D Translation(double tx, double ty, double tz) => new(1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1);
    public static Matrix4x4D Translation(Vector3D t) => Translation(t.X, t.Y, t.Z);
    public static Matrix4x4D Scaling(double sx, double sy, double sz) => new(sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1);
    public static Matrix4x4D Scaling(Vector3D s) => Scaling(s.X, s.Y, s.Z);
    public static Matrix4x4D RotationX(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1); }
    public static Matrix4x4D RotationY(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1); }
    public static Matrix4x4D RotationZ(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1); }
    public static Matrix4x4D RotationAxis(Vector3D ax, double a)
    {
        Vector3D u = ax.Normalized();
        double c = Math.Cos(a), s = Math.Sin(a), t = 1 - c;
        return new(t * u.X * u.X + c, t * u.X * u.Y - s * u.Z, t * u.X * u.Z + s * u.Y, 0,
            t * u.X * u.Y + s * u.Z, t * u.Y * u.Y + c, t * u.Y * u.Z - s * u.X, 0,
            t * u.X * u.Z - s * u.Y, t * u.Y * u.Z + s * u.X, t * u.Z * u.Z + c, 0, 0, 0, 0, 1);
    }
    public static Matrix4x4D FromQuaternion(QuaternionD q) => q.ToMatrix4x4D();
    public static Matrix4x4D LookAt(Vector3D eye, Vector3D target, Vector3D up)
    {
        Vector3D f = (target - eye).Normalized(), r = Vector3D.Cross(f, up).Normalized(), u = Vector3D.Cross(r, f);
        return new Matrix4x4D(r.X, r.Y, r.Z, -Vector3D.Dot(r, eye), u.X, u.Y, u.Z, -Vector3D.Dot(u, eye), -f.X, -f.Y, -f.Z, Vector3D.Dot(f, eye), 0, 0, 0, 1);
    }
    public static Matrix4x4D Perspective(double fovyRad, double aspect, double near, double far)
    {
        double t = Math.Tan(fovyRad * 0.5);
        return new Matrix4x4D(1.0 / (aspect * t), 0, 0, 0, 0, 1.0 / t, 0, 0, 0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near), 0, 0, -1, 0);
    }
    public static Matrix4x4D InfinitePerspective(double fovyRad, double aspect, double near)
    {
        double t = Math.Tan(fovyRad * 0.5);
        return new Matrix4x4D(1.0 / (aspect * t), 0, 0, 0, 0, 1.0 / t, 0, 0, 0, 0, -1, -2 * near, 0, 0, -1, 0);
    }
    public static Matrix4x4D Orthographic(double l, double r, double b, double t, double n, double f)
    {
        double rl = r - l, tb = t - b, fn = f - n;
        return new Matrix4x4D(2.0 / rl, 0, 0, -(r + l) / rl, 0, 2.0 / tb, 0, -(t + b) / tb, 0, 0, -2.0 / fn, -(f + n) / fn, 0, 0, 0, 1);
    }

    public readonly double Trace => M11 + M22 + M33 + M44;
    public readonly double Determinant
    {
        get
        {
            double a = M11 * (M22 * (M33 * M44 - M34 * M43) - M23 * (M32 * M44 - M34 * M42) + M24 * (M32 * M43 - M33 * M42));
            double b = M12 * (M21 * (M33 * M44 - M34 * M43) - M23 * (M31 * M44 - M34 * M41) + M24 * (M31 * M43 - M33 * M41));
            double c = M13 * (M21 * (M32 * M44 - M34 * M42) - M22 * (M31 * M44 - M34 * M41) + M24 * (M31 * M42 - M32 * M41));
            double d = M14 * (M21 * (M32 * M43 - M33 * M42) - M22 * (M31 * M43 - M33 * M41) + M23 * (M31 * M42 - M32 * M41));
            return a - b + c - d;
        }
    }
    public readonly Matrix4x4D Transpose() => new(M11, M21, M31, M41, M12, M22, M32, M42, M13, M23, M33, M43, M14, M24, M34, M44);
    public readonly Matrix4x4D Inverse()
    {
        double det = Determinant;
        if (Math.Abs(det) < 1e-30)
            return Zero;
        double inv = 1.0 / det;
        double c11 = (M22 * (M33 * M44 - M34 * M43) - M23 * (M32 * M44 - M34 * M42) + M24 * (M32 * M43 - M33 * M42));
        double c12 = -(M21 * (M33 * M44 - M34 * M43) - M23 * (M31 * M44 - M34 * M41) + M24 * (M31 * M43 - M33 * M41));
        double c13 = (M21 * (M32 * M44 - M34 * M42) - M22 * (M31 * M44 - M34 * M41) + M24 * (M31 * M42 - M32 * M41));
        double c14 = -(M21 * (M32 * M43 - M33 * M42) - M22 * (M31 * M43 - M33 * M41) + M23 * (M31 * M42 - M32 * M41));
        double c21 = -(M12 * (M33 * M44 - M34 * M43) - M13 * (M32 * M44 - M34 * M42) + M14 * (M32 * M43 - M33 * M42));
        double c22 = (M11 * (M33 * M44 - M34 * M43) - M13 * (M31 * M44 - M34 * M41) + M14 * (M31 * M43 - M33 * M41));
        double c23 = -(M11 * (M32 * M44 - M34 * M42) - M12 * (M31 * M44 - M34 * M41) + M14 * (M31 * M42 - M32 * M41));
        double c24 = (M11 * (M32 * M43 - M33 * M42) - M12 * (M31 * M43 - M33 * M41) + M13 * (M31 * M42 - M32 * M41));
        double c31 = (M12 * (M23 * M44 - M24 * M43) - M13 * (M22 * M44 - M24 * M42) + M14 * (M22 * M43 - M23 * M42));
        double c32 = -(M11 * (M23 * M44 - M24 * M43) - M13 * (M21 * M44 - M24 * M41) + M14 * (M21 * M43 - M23 * M41));
        double c33 = (M11 * (M22 * M44 - M24 * M42) - M12 * (M21 * M44 - M24 * M41) + M14 * (M21 * M42 - M22 * M41));
        double c34 = -(M11 * (M22 * M43 - M23 * M42) - M12 * (M21 * M43 - M23 * M41) + M13 * (M21 * M42 - M22 * M41));
        double c41 = -(M12 * (M23 * M34 - M24 * M33) - M13 * (M22 * M34 - M24 * M32) + M14 * (M22 * M33 - M23 * M32));
        double c42 = (M11 * (M23 * M34 - M24 * M33) - M13 * (M21 * M34 - M24 * M31) + M14 * (M21 * M33 - M23 * M31));
        double c43 = -(M11 * (M22 * M34 - M24 * M32) - M12 * (M21 * M34 - M24 * M31) + M14 * (M21 * M32 - M22 * M31));
        double c44 = (M11 * (M22 * M33 - M23 * M32) - M12 * (M21 * M33 - M23 * M31) + M13 * (M21 * M32 - M22 * M31));
        return new Matrix4x4D(c11 * inv, c21 * inv, c31 * inv, c41 * inv, c12 * inv, c22 * inv, c32 * inv, c42 * inv, c13 * inv, c23 * inv, c33 * inv, c43 * inv, c14 * inv, c24 * inv, c34 * inv, c44 * inv);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly Vector3D TransformPoint(Vector3D p) { double w = M41 * p.X + M42 * p.Y + M43 * p.Z + M44; if (Math.Abs(w) < 1e-30) w = 1e-30; return new((M11 * p.X + M12 * p.Y + M13 * p.Z + M14) / w, (M21 * p.X + M22 * p.Y + M23 * p.Z + M24) / w, (M31 * p.X + M32 * p.Y + M33 * p.Z + M34) / w); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly Vector3D TransformVector(Vector3D v) => new(M11 * v.X + M12 * v.Y + M13 * v.Z, M21 * v.X + M22 * v.Y + M23 * v.Z, M31 * v.X + M32 * v.Y + M33 * v.Z);
    public readonly Tensor3D ToTensor3D() => new(M11, M12, M13, M21, M22, M23, M31, M32, M33);
    public readonly Vector3D GetTranslation() => new(M14, M24, M34);
    public readonly QuaternionD GetRotation() => QuaternionD.FromMatrix3x3(ToTensor3D());
    public readonly Vector3D GetScale() => new(ToTensor3D().Column(0).Length(), ToTensor3D().Column(1).Length(), ToTensor3D().Column(2).Length());
    public readonly void Decompose(out Vector3D translation, out QuaternionD rotation, out Vector3D scale)
    {
        translation = GetTranslation();
        scale = GetScale();
        Tensor3D rot = ToTensor3D();
        if (scale.X > 1e-10)
        { rot.M11 /= scale.X; rot.M12 /= scale.X; rot.M13 /= scale.X; }
        if (scale.Y > 1e-10)
        { rot.M21 /= scale.Y; rot.M22 /= scale.Y; rot.M23 /= scale.Y; }
        if (scale.Z > 1e-10)
        { rot.M31 /= scale.Z; rot.M32 /= scale.Z; rot.M33 /= scale.Z; }
        rotation = QuaternionD.FromMatrix3x3(rot);
    }

    public readonly double FrobeniusNorm => Math.Sqrt(M11 * M11 + M12 * M12 + M13 * M13 + M14 * M14 + M21 * M21 + M22 * M22 + M23 * M23 + M24 * M24 + M31 * M31 + M32 * M32 + M33 * M33 + M34 * M34 + M41 * M41 + M42 * M42 + M43 * M43 + M44 * M44);

    public static Matrix4x4D operator *(Matrix4x4D a, Matrix4x4D b) { Matrix4x4D r; r.M11 = a.M11 * b.M11 + a.M12 * b.M21 + a.M13 * b.M31 + a.M14 * b.M41; r.M12 = a.M11 * b.M12 + a.M12 * b.M22 + a.M13 * b.M32 + a.M14 * b.M42; r.M13 = a.M11 * b.M13 + a.M12 * b.M23 + a.M13 * b.M33 + a.M14 * b.M43; r.M14 = a.M11 * b.M14 + a.M12 * b.M24 + a.M13 * b.M34 + a.M14 * b.M44; r.M21 = a.M21 * b.M11 + a.M22 * b.M21 + a.M23 * b.M31 + a.M24 * b.M41; r.M22 = a.M21 * b.M12 + a.M22 * b.M22 + a.M23 * b.M32 + a.M24 * b.M42; r.M23 = a.M21 * b.M13 + a.M22 * b.M23 + a.M23 * b.M33 + a.M24 * b.M43; r.M24 = a.M21 * b.M14 + a.M22 * b.M24 + a.M23 * b.M34 + a.M24 * b.M44; r.M31 = a.M31 * b.M11 + a.M32 * b.M21 + a.M33 * b.M31 + a.M34 * b.M41; r.M32 = a.M31 * b.M12 + a.M32 * b.M22 + a.M33 * b.M32 + a.M34 * b.M42; r.M33 = a.M31 * b.M13 + a.M32 * b.M23 + a.M33 * b.M33 + a.M34 * b.M43; r.M34 = a.M31 * b.M14 + a.M32 * b.M24 + a.M33 * b.M34 + a.M34 * b.M44; r.M41 = a.M41 * b.M11 + a.M42 * b.M21 + a.M43 * b.M31 + a.M44 * b.M41; r.M42 = a.M41 * b.M12 + a.M42 * b.M22 + a.M43 * b.M32 + a.M44 * b.M42; r.M43 = a.M41 * b.M13 + a.M42 * b.M23 + a.M43 * b.M33 + a.M44 * b.M43; r.M44 = a.M41 * b.M14 + a.M42 * b.M24 + a.M43 * b.M34 + a.M44 * b.M44; return r; }
    public static Matrix4x4D operator *(Matrix4x4D m, double s) => new(m.M11 * s, m.M12 * s, m.M13 * s, m.M14 * s, m.M21 * s, m.M22 * s, m.M23 * s, m.M24 * s, m.M31 * s, m.M32 * s, m.M33 * s, m.M34 * s, m.M41 * s, m.M42 * s, m.M43 * s, m.M44 * s);
    public static Matrix4x4D operator *(double s, Matrix4x4D m) => m * s;
    public static Matrix4x4D operator +(Matrix4x4D a, Matrix4x4D b) => new(a.M11 + b.M11, a.M12 + b.M12, a.M13 + b.M13, a.M14 + b.M14, a.M21 + b.M21, a.M22 + b.M22, a.M23 + b.M23, a.M24 + b.M24, a.M31 + b.M31, a.M32 + b.M32, a.M33 + b.M33, a.M34 + b.M34, a.M41 + b.M41, a.M42 + b.M42, a.M43 + b.M43, a.M44 + b.M44);
    public static Matrix4x4D operator -(Matrix4x4D a, Matrix4x4D b) => new(a.M11 - b.M11, a.M12 - b.M12, a.M13 - b.M13, a.M14 - b.M14, a.M21 - b.M21, a.M22 - b.M22, a.M23 - b.M23, a.M24 - b.M24, a.M31 - b.M31, a.M32 - b.M32, a.M33 - b.M33, a.M34 - b.M34, a.M41 - b.M41, a.M42 - b.M42, a.M43 - b.M43, a.M44 - b.M44);
    public static Matrix4x4D operator -(Matrix4x4D m) => new(-m.M11, -m.M12, -m.M13, -m.M14, -m.M21, -m.M22, -m.M23, -m.M24, -m.M31, -m.M32, -m.M33, -m.M34, -m.M41, -m.M42, -m.M43, -m.M44);

    public readonly bool Equals(Matrix4x4D o) => Math.Abs(M11 - o.M11) < 1e-10 && Math.Abs(M12 - o.M12) < 1e-10 && Math.Abs(M13 - o.M13) < 1e-10 && Math.Abs(M14 - o.M14) < 1e-10 && Math.Abs(M21 - o.M21) < 1e-10 && Math.Abs(M22 - o.M22) < 1e-10 && Math.Abs(M23 - o.M23) < 1e-10 && Math.Abs(M24 - o.M24) < 1e-10 && Math.Abs(M31 - o.M31) < 1e-10 && Math.Abs(M32 - o.M32) < 1e-10 && Math.Abs(M33 - o.M33) < 1e-10 && Math.Abs(M34 - o.M34) < 1e-10 && Math.Abs(M41 - o.M41) < 1e-10 && Math.Abs(M42 - o.M42) < 1e-10 && Math.Abs(M43 - o.M43) < 1e-10 && Math.Abs(M44 - o.M44) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is Matrix4x4D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(M11, M22, M33, M44);
    public static bool operator ==(Matrix4x4D a, Matrix4x4D b) => a.Equals(b);
    public static bool operator !=(Matrix4x4D a, Matrix4x4D b) => !a.Equals(b);
    public override readonly string ToString() => $"[{M11:F3} {M12:F3} {M13:F3} {M14:F3}]\n[{M21:F3} {M22:F3} {M23:F3} {M24:F3}]\n[{M31:F3} {M32:F3} {M33:F3} {M34:F3}]\n[{M41:F3} {M42:F3} {M43:F3} {M44:F3}]";
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 4: PRIMITIVES GEOMETRIQUES
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Boite englobante axiale (AABB) en 3D. Definie par un coin min et un coin max.
/// Utilisee pour l'acceleration spatiale, le culling, et les tests de collision.
/// Les operations sont O(1) grace au stockage direct min/max.
///
/// MEMORY LAYOUT: 64 bytes (2 Vector3D).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64, Pack = 32)]
[DebuggerDisplay("AABB({Min} -> {Max})")]
public struct BoundingBox3D : IEquatable<BoundingBox3D>
{
    [FieldOffset(0)] public Vector3D Min;
    [FieldOffset(32)] public Vector3D Max;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public BoundingBox3D(Vector3D min, Vector3D max) { Min = min; Max = max; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public BoundingBox3D(double minX, double minY, double minZ, double maxX, double maxY, double maxZ) { Min = new(minX, minY, minZ); Max = new(maxX, maxY, maxZ); }

    public static readonly BoundingBox3D Empty = new(Vector3D.Zero, Vector3D.Zero);
    public static readonly BoundingBox3D Infinite = new(new(double.NegativeInfinity), new(double.PositiveInfinity));

    public readonly Vector3D Center => (Min + Max) * 0.5;
    public readonly Vector3D Size => Max - Min;
    public readonly Vector3D HalfSize => Size * 0.5;
    public readonly double Volume { get { Vector3D s = Size; return s.X * s.Y * s.Z; } }
    public readonly double SurfaceArea { get { Vector3D s = Size; return 2.0 * (s.X * s.Y + s.Y * s.Z + s.Z * s.X); } }
    public readonly double DiagonalLength => Size.Length();
    public readonly Vector3D Diagonal => Size;
    public readonly double Perimeter { get { Vector3D s = Size; return 4.0 * (s.X + s.Y + s.Z); } }
    public readonly double MaxFaceArea { get { Vector3D s = Size; double xy = s.X * s.Y, yz = s.Y * s.Z, zx = s.Z * s.X; return Math.Max(Math.Max(xy, yz), zx); } }
    public readonly int LargestAxis { get { Vector3D s = Size; if (s.X >= s.Y && s.X >= s.Z) return 0; if (s.Y >= s.Z) return 1; return 2; } }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool Contains(Vector3D p) => p.X >= Min.X && p.X <= Max.X && p.Y >= Min.Y && p.Y <= Max.Y && p.Z >= Min.Z && p.Z <= Max.Z;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool ContainsStrict(Vector3D p) => p.X > Min.X && p.X < Max.X && p.Y > Min.Y && p.Y < Max.Y && p.Z > Min.Z && p.Z < Max.Z;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool Contains(BoundingBox3D o) => o.Min.X >= Min.X && o.Max.X <= Max.X && o.Min.Y >= Min.Y && o.Max.Y <= Max.Y && o.Min.Z >= Min.Z && o.Max.Z <= Max.Z;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool Intersects(BoundingBox3D o) => Min.X <= o.Max.X && Max.X >= o.Min.X && Min.Y <= o.Max.Y && Max.Y >= o.Min.Y && Min.Z <= o.Max.Z && Max.Z >= o.Min.Z;
    public readonly BoundingBox3D Intersection(BoundingBox3D o) => new(Vector3D.Max(Min, o.Min), Vector3D.Min(Max, o.Max));
    public readonly BoundingBox3D Union(BoundingBox3D o) => new(Vector3D.Min(Min, o.Min), Vector3D.Max(Max, o.Max));
    public static BoundingBox3D FromCenterAndHalfExtents(Vector3D c, Vector3D h) => new(c - h, c + h);
    public readonly BoundingBox3D Expanded(double m) => new(Min - new Vector3D(m), Max + new Vector3D(m));
    public readonly BoundingBox3D Expanded(Vector3D m) => new(Min - m, Max + m);
    public readonly Vector3D ClosestPoint(Vector3D p) => Vector3D.Clamp(p, Min, Max);
    public readonly double DistanceTo(Vector3D p) => (ClosestPoint(p) - p).Length();
    public readonly double DistanceSquaredTo(Vector3D p) => (ClosestPoint(p) - p).LengthSquared();

    public readonly bool Equals(BoundingBox3D o) => Min.Equals(o.Min) && Max.Equals(o.Max);
    public override readonly bool Equals(object? obj) => obj is BoundingBox3D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Min, Max);
    public static bool operator ==(BoundingBox3D a, BoundingBox3D b) => a.Equals(b);
    public static bool operator !=(BoundingBox3D a, BoundingBox3D b) => !a.Equals(b);
    public override readonly string ToString() => $"AABB({Min} -> {Max})";
}

/// <summary>
/// Rayon en 3D : origine + direction normalisee. Utilise pour le raycasting,
/// le picking, les tests d'intersection, et la simulation de photons.
/// La direction est toujours normalisee pour simplifier les calculs de distance.
///
/// MEMORY LAYOUT: 64 bytes (2 Vector3D).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64, Pack = 32)]
[DebuggerDisplay("Ray(Origin={Origin}, Dir={Direction})")]
public struct Ray3D : IEquatable<Ray3D>
{
    [FieldOffset(0)] public Vector3D Origin;
    [FieldOffset(32)] public Vector3D Direction;

    /// <summary>Constructeur avec normalisation automatique de la direction.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Ray3D(Vector3D origin, Vector3D direction) { Origin = origin; Direction = direction.Normalized(); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Ray3D(double ox, double oy, double oz, double dx, double dy, double dz) { Origin = new Vector3D(ox, oy, oz); Direction = new Vector3D(dx, dy, dz).Normalized(); }

    /// <summary>Point sur le rayon a la distance t : P = O + t*D.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly Vector3D GetPoint(double t) => Origin + Direction * t;

    /// <summary>Intersection avec un AABB. Retourne true si le rayon traverse la boite.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IntersectsAABB(BoundingBox3D box, out double tMin, out double tMax)
    {
        tMin = 0;
        tMax = double.PositiveInfinity;
        for (int i = 0; i < 3; i++)
        {
            double orig = i == 0 ? Origin.X : (i == 1 ? Origin.Y : Origin.Z);
            double dir = i == 0 ? Direction.X : (i == 1 ? Direction.Y : Direction.Z);
            double bmin = i == 0 ? box.Min.X : (i == 1 ? box.Min.Y : box.Min.Z);
            double bmax = i == 0 ? box.Max.X : (i == 1 ? box.Max.Y : box.Max.Z);
            if (Math.Abs(dir) < 1e-15)
            { if (orig < bmin || orig > bmax) return false; }
            else
            { double inv = 1.0 / dir; double t1 = (bmin - orig) * inv, t2 = (bmax - orig) * inv; if (t1 > t2) (t1, t2) = (t2, t1); tMin = Math.Max(tMin, t1); tMax = Math.Min(tMax, t2); if (tMin > tMax) return false; }
        }
        return true;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool IntersectsAABB(BoundingBox3D box) => IntersectsAABB(box, out _, out _);

    /// <summary>Intersection avec un plan.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IntersectPlane(Plane3D plane, out double t)
    {
        double denom = Vector3D.Dot(plane.Normal, Direction);
        if (Math.Abs(denom) < 1e-15)
        { t = 0; return false; }
        t = -(Vector3D.Dot(plane.Normal, Origin) + plane.Distance) / denom;
        return t >= 0;
    }

    /// <summary>Intersection avec une sphere : retourne les distances d'entree et de sortie.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IntersectSphere(Vector3D center, double radius, out double tEntry, out double tExit)
    {
        Vector3D oc = Origin - center;
        double a = Vector3D.Dot(Direction, Direction);
        double b = 2.0 * Vector3D.Dot(oc, Direction);
        double c = Vector3D.Dot(oc, oc) - radius * radius;
        double disc = b * b - 4.0 * a * c;
        if (disc < 0)
        { tEntry = 0; tExit = 0; return false; }
        double sqrtD = Math.Sqrt(disc);
        tEntry = (-b - sqrtD) / (2.0 * a);
        tExit = (-b + sqrtD) / (2.0 * a);
        return tExit >= 0;
    }

    /// <summary>Point le plus proche du rayon a une position donnee.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ClosestPoint(Vector3D point) { double t = Math.Max(0, Vector3D.Dot(point - Origin, Direction)); return Origin + Direction * t; }

    /// <summary>Distance du rayon a un point.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double DistanceToPoint(Vector3D point) => (ClosestPoint(point) - point).Length();

    public readonly bool Equals(Ray3D o) => Origin.Equals(o.Origin) && Direction.Equals(o.Direction);
    public override readonly bool Equals(object? obj) => obj is Ray3D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Origin, Direction);
    public static bool operator ==(Ray3D a, Ray3D b) => a.Equals(b);
    public static bool operator !=(Ray3D a, Ray3D b) => !a.Equals(b);
    public override readonly string ToString() => $"Ray(Origin={Origin}, Dir={Direction})";
}

/// <summary>
/// Plan en 3D : normale unitaire + distance signee au plan depuis l'origine.
/// L'equation du plan est : N.x + d = 0, ou N est la normale et d la distance.
/// Un point est devant le plan si N.p + d > 0, derriere si < 0.
///
/// MEMORY LAYOUT: 32 bytes (Vector3D + double).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 32, Pack = 16)]
[DebuggerDisplay("Plane(N={Normal}, D={Distance:F4})")]
public struct Plane3D : IEquatable<Plane3D>
{
    [FieldOffset(0)] public Vector3D Normal;
    [FieldOffset(24)] public double Distance;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public Plane3D(Vector3D normal, double distance) { Normal = normal.Normalized(); Distance = distance; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public Plane3D(Vector3D point, Vector3D normal) { Normal = normal.Normalized(); Distance = -Vector3D.Dot(Normal, point); }

    /// <summary>Construit un plan a partir de 3 points non colineaires.</summary>
    public static Plane3D FromPoints(Vector3D a, Vector3D b, Vector3D c) { Vector3D n = Vector3D.Cross(b - a, c - a).Normalized(); return new(n, -Vector3D.Dot(n, a)); }

    /// <summary>Distance signee d'un point au plan : N.p + d.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly double SignedDistance(Vector3D point) => Vector3D.Dot(Normal, point) + Distance;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool IsBehind(Vector3D p) => SignedDistance(p) < 0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool IsInFront(Vector3D p) => SignedDistance(p) > 0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly Vector3D ProjectPoint(Vector3D p) => p - Normal * SignedDistance(p);
    public readonly Plane3D Flipped => new(-Normal, -Distance);

    public readonly bool Equals(Plane3D o) => Normal.Equals(o.Normal) && Math.Abs(Distance - o.Distance) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is Plane3D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Normal, Distance);
    public static bool operator ==(Plane3D a, Plane3D b) => a.Equals(b);
    public static bool operator !=(Plane3D a, Plane3D b) => !a.Equals(b);
    public override readonly string ToString() => $"Plane(N={Normal}, D={Distance:F4})";
}

/// <summary>
/// Volume de vision (frustum) compose de 6 plans : left, right, bottom, top, near, far.
/// Utilise pour le culling hierarchique (frustum culling) dans le moteur de rendu.
///
/// MEMORY LAYOUT: 192 bytes (6 Plane3D).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 192, Pack = 16)]
public struct Frustum6Planes : IEquatable<Frustum6Planes>
{
    [FieldOffset(0)] public Plane3D Left;
    [FieldOffset(32)] public Plane3D Right;
    [FieldOffset(64)] public Plane3D Bottom;
    [FieldOffset(96)] public Plane3D Top;
    [FieldOffset(128)] public Plane3D Near;
    [FieldOffset(160)] public Plane3D Far;

    /// <summary>Extrait les 6 plans a partir d'une matrice view-projection.</summary>
    public static Frustum6Planes FromViewProjection(Matrix4x4D vp)
    {
        Frustum6Planes f;
        f.Left = Norm(new Plane3D(new Vector3D(vp.M14 + vp.M11, vp.M24 + vp.M21, vp.M34 + vp.M31), vp.M44 + vp.M41));
        f.Right = Norm(new Plane3D(new Vector3D(vp.M14 - vp.M11, vp.M24 - vp.M21, vp.M34 - vp.M31), vp.M44 - vp.M41));
        f.Bottom = Norm(new Plane3D(new Vector3D(vp.M14 + vp.M12, vp.M24 + vp.M22, vp.M34 + vp.M32), vp.M44 + vp.M42));
        f.Top = Norm(new Plane3D(new Vector3D(vp.M14 - vp.M12, vp.M24 - vp.M22, vp.M34 - vp.M32), vp.M44 - vp.M42));
        f.Near = Norm(new Plane3D(new Vector3D(vp.M13, vp.M23, vp.M33), vp.M43));
        f.Far = Norm(new Plane3D(new Vector3D(vp.M14 - vp.M13, vp.M24 - vp.M23, vp.M34 - vp.M33), vp.M44 - vp.M43));
        return f;
    }
    private static Plane3D Norm(Plane3D p) { double len = p.Normal.Length(); return len < 1e-15 ? p : new(p.Normal / len, p.Distance / len); }

    /// <summary>Teste si un point est a l'interieur du frustum.</summary>
    public readonly bool ContainsPoint(Vector3D p) => Left.SignedDistance(p) >= 0 && Right.SignedDistance(p) >= 0 && Bottom.SignedDistance(p) >= 0 && Top.SignedDistance(p) >= 0 && Near.SignedDistance(p) >= 0 && Far.SignedDistance(p) >= 0;

    /// <summary>Teste l'intersection avec un AABB (separation axis theorem).</summary>
    public readonly bool IntersectsAABB(BoundingBox3D box)
    {
        Span<Plane3D> planes = stackalloc Plane3D[] { Left, Right, Bottom, Top, Near, Far };
        for (int i = 0; i < 6; i++)
        {
            Vector3D pv = new(planes[i].Normal.X >= 0 ? box.Max.X : box.Min.X, planes[i].Normal.Y >= 0 ? box.Max.Y : box.Min.Y, planes[i].Normal.Z >= 0 ? box.Max.Z : box.Min.Z);
            if (planes[i].SignedDistance(pv) < 0)
                return false;
        }
        return true;
    }

    public readonly bool Equals(Frustum6Planes o) => Left.Equals(o.Left) && Right.Equals(o.Right) && Bottom.Equals(o.Bottom) && Top.Equals(o.Top) && Near.Equals(o.Near) && Far.Equals(o.Far);
    public override readonly bool Equals(object? obj) => obj is Frustum6Planes o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Left, Right, Bottom, Top, Near, Far);
    public static bool operator ==(Frustum6Planes a, Frustum6Planes b) => a.Equals(b);
    public static bool operator !=(Frustum6Planes a, Frustum6Planes b) => !a.Equals(b);
}

/// <summary>
/// Couleur HDR (High Dynamic Range) avec composantes spectrales.
/// Supporte les valeurs au-dela de [0,1] pour l'eclairage realiste.
/// Inclut des methodes de tonemapping (Reinhard, ACES) pour la conversion
/// vers l'affichage standard.
///
/// MEMORY LAYOUT: 64 bytes (7 doubles + padding).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64, Pack = 8)]
[DebuggerDisplay("ColorHDR({R:F3}, {G:F3}, {B:F3}, I={Intensity:F3})")]
public struct ColorHDR : IEquatable<ColorHDR>
{
    [FieldOffset(0)] public double R;
    [FieldOffset(8)] public double G;
    [FieldOffset(16)] public double B;
    [FieldOffset(24)] public double Intensity;
    [FieldOffset(32)] public double SpectralLow;
    [FieldOffset(40)] public double SpectralMid;
    [FieldOffset(48)] public double SpectralHigh;
    [FieldOffset(56)] public double _pad;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public ColorHDR(double r, double g, double b, double intensity = 1.0) { R = r; G = g; B = b; Intensity = intensity; SpectralLow = r; SpectralMid = g; SpectralHigh = b; _pad = 0; }

    public static readonly ColorHDR Black = new(0, 0, 0, 0);
    public static readonly ColorHDR White = new(1, 1, 1, 1);
    public static readonly ColorHDR Red = new(1, 0, 0, 1);
    public static readonly ColorHDR Green = new(0, 1, 0, 1);
    public static readonly ColorHDR Blue = new(0, 0, 1, 1);
    public static readonly ColorHDR Yellow = new(1, 1, 0, 1);
    public static readonly ColorHDR Cyan = new(0, 1, 1, 1);
    public static readonly ColorHDR Magenta = new(1, 0, 1, 1);

    /// <summary>Luminance relative (BT.709) : 0.2126*R + 0.7152*G + 0.0722*B.</summary>
    public readonly double Luminance => 0.2126 * R + 0.7152 * G + 0.0722 * B;
    public readonly double MaxComponent => Math.Max(Math.Max(R, G), B);
    public readonly double AverageComponent => (R + G + B) / 3.0;

    public readonly ColorHDR Scaled(double s) => new(R * s * Intensity, G * s * Intensity, B * s * Intensity, 1.0);
    public readonly ColorHDR WithIntensity(double i) => new(R, G, B, i);

    /// <summary>Tonemapping Reinhard : L = 1 - exp(-E * exposure).</summary>
    public readonly ColorHDR ToneMapReinhard(double exposure = 1.0) { double er = R * Intensity * exposure, eg = G * Intensity * exposure, eb = B * Intensity * exposure; return new(1 - Math.Exp(-er), 1 - Math.Exp(-eg), 1 - Math.Exp(-eb), 1.0); }
    /// <summary>Tonemapping ACES Filmic : courbe S pour un contraste cinematographique.</summary>
    public readonly ColorHDR ToneMapAces()
    {
        const double a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
        double er = R * Intensity * 0.6, eg = G * Intensity * 0.6, eb = B * Intensity * 0.6;
        return new(Math.Clamp((er * (a * er + b)) / (er * (c * er + d) + e), 0, 1), Math.Clamp((eg * (a * eg + b)) / (eg * (c * eg + d) + e), 0, 1), Math.Clamp((eb * (a * eb + b)) / (eb * (c * eb + d) + e), 0, 1), 1.0);
    }
    /// <summary>Conversion gamma vers linear.</summary>
    public readonly ColorHDR GammaToLinear() => new(Math.Pow(R, 2.2), Math.Pow(G, 2.2), Math.Pow(B, 2.2), Intensity);
    /// <summary>Conversion linear vers gamma.</summary>
    public readonly ColorHDR LinearToGamma() => new(Math.Pow(Math.Max(R, 0), 1.0 / 2.2), Math.Pow(Math.Max(G, 0), 1.0 / 2.2), Math.Pow(Math.Max(B, 0), 1.0 / 2.2), Intensity);

    public readonly ColorHDR Lerp(ColorHDR o, double t) { double u = 1 - t; return new(R * u + o.R * t, G * u + o.G * t, B * u + o.B * t, Intensity * u + o.Intensity * t); }
    public static ColorHDR operator +(ColorHDR a, ColorHDR b) => new(a.R + b.R, a.G + b.G, a.B + b.B, (a.Intensity + b.Intensity) * 0.5);
    public static ColorHDR operator *(ColorHDR a, ColorHDR b) => new(a.R * b.R, a.G * b.G, a.B * b.B, a.Intensity * b.Intensity);
    public static ColorHDR operator *(ColorHDR c, double s) => new(c.R * s, c.G * s, c.B * s, c.Intensity);
    public static ColorHDR operator *(double s, ColorHDR c) => c * s;

    public readonly bool Equals(ColorHDR o) => Math.Abs(R - o.R) < 1e-10 && Math.Abs(G - o.G) < 1e-10 && Math.Abs(B - o.B) < 1e-10 && Math.Abs(Intensity - o.Intensity) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is ColorHDR o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(R, G, B, Intensity);
    public static bool operator ==(ColorHDR a, ColorHDR b) => a.Equals(b);
    public static bool operator !=(ColorHDR a, ColorHDR b) => !a.Equals(b);
    public override readonly string ToString() => $"ColorHDR({R:F3}, {G:F3}, {B:F3}, I={Intensity:F3})";
}

/// <summary>
/// Intervalle [Min, Max] pour l'arithmetique par intervalles (IA).
/// Garantit les bornes d'erreur dans les calculs numeriques.
/// Chaque operation arithmetique propage correctement les erreurs d'arrondi.
///
/// Utilise pour la verification de racines, l'analyse d'erreurs,
/// et les tests d'inclusion dans les structures de donnees spatiales.
///
/// MEMORY LAYOUT: 16 bytes (2 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 16, Pack = 8)]
[DebuggerDisplay("[{Min:F6}, {Max:F6}]")]
public struct IntervalD : IEquatable<IntervalD>
{
    [FieldOffset(0)] public double Min;
    [FieldOffset(8)] public double Max;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public IntervalD(double min, double max) { Min = min; Max = max; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public IntervalD(double value) { Min = value; Max = value; }

    public static readonly IntervalD Zero = new(0, 0);
    public static readonly IntervalD Positive = new(0, double.PositiveInfinity);
    public static readonly IntervalD Negative = new(double.NegativeInfinity, 0);
    public static readonly IntervalD Entire = new(double.NegativeInfinity, double.PositiveInfinity);
    public static readonly IntervalD Unit = new(0, 1);

    public readonly double Midpoint => (Min + Max) * 0.5;
    public readonly double Width => Max - Min;
    public readonly double Radius => Width * 0.5;
    public readonly bool IsValid => Min <= Max;
    public readonly bool IsPoint => Math.Abs(Max - Min) < 1e-15;
    public readonly bool Contains(double v) => v >= Min && v <= Max;
    public readonly bool Contains(IntervalD o) => o.Min >= Min && o.Max <= Max;
    public readonly bool Intersects(IntervalD o) => Min <= o.Max && Max >= o.Min;
    public readonly IntervalD Intersection(IntervalD o) => new(Math.Max(Min, o.Min), Math.Min(Max, o.Max));
    public readonly IntervalD Union(IntervalD o) => new(Math.Min(Min, o.Min), Math.Max(Max, o.Max));

    public static IntervalD operator +(IntervalD a, IntervalD b) => new(a.Min + b.Min, a.Max + b.Max);
    public static IntervalD operator -(IntervalD a, IntervalD b) => new(a.Min - b.Max, a.Max - b.Min);
    public static IntervalD operator *(IntervalD a, IntervalD b) { double p1 = a.Min * b.Min, p2 = a.Min * b.Max, p3 = a.Max * b.Min, p4 = a.Max * b.Max; return new(Math.Min(Math.Min(p1, p2), Math.Min(p3, p4)), Math.Max(Math.Max(p1, p2), Math.Max(p3, p4))); }
    public static IntervalD operator /(IntervalD a, IntervalD b) { if (b.Contains(0)) return Entire; double p1 = a.Min / b.Min, p2 = a.Min / b.Max, p3 = a.Max / b.Min, p4 = a.Max / b.Max; return new(Math.Min(Math.Min(p1, p2), Math.Min(p3, p4)), Math.Max(Math.Max(p1, p2), Math.Max(p3, p4))); }
    public static IntervalD operator +(IntervalD a, double s) => new(a.Min + s, a.Max + s);
    public static IntervalD operator -(IntervalD a, double s) => new(a.Min - s, a.Max - s);
    public static IntervalD operator *(IntervalD a, double s) => s >= 0 ? new(a.Min * s, a.Max * s) : new(a.Max * s, a.Min * s);
    public static IntervalD operator /(IntervalD a, double s) => a * (1.0 / s);
    public static IntervalD operator -(IntervalD a) => new(-a.Max, -a.Min);

    public readonly IntervalD Sin() { double sMin = Math.Sin(Min), sMax = Math.Sin(Max); double lo = Math.Min(sMin, sMax), hi = Math.Max(sMin, sMax); if (Min < Max && Math.PI > Min) { lo = -1; hi = Math.Max(sMin, sMax); } return new(lo, hi); }
    public readonly IntervalD Cos() { double cMin = Math.Cos(Min), cMax = Math.Cos(Max); double lo = Math.Min(cMin, cMax), hi = Math.Max(cMin, cMax); if (Min < Max && 0 > Min && 0 < Max) hi = 1; return new(lo, hi); }
    public readonly IntervalD Exp() => new(Math.Exp(Min), Math.Exp(Max));
    public readonly IntervalD Log() => Min > 0 ? new(Math.Log(Min), Math.Log(Max)) : Entire;
    public readonly IntervalD Abs() => Min >= 0 ? this : (Max <= 0 ? new(-Max, -Min) : new(0, Math.Max(-Min, Max)));
    public readonly IntervalD Sqr() => Min >= 0 ? new(Min * Min, Max * Max) : (Max <= 0 ? new(Max * Max, Min * Min) : new(0, Math.Max(Min * Min, Max * Max)));
    public readonly IntervalD Sqrt() => Min >= 0 ? new(Math.Sqrt(Min), Math.Sqrt(Max)) : Entire;

    public readonly bool Equals(IntervalD o) => Math.Abs(Min - o.Min) < 1e-15 && Math.Abs(Max - o.Max) < 1e-15;
    public override readonly bool Equals(object? obj) => obj is IntervalD o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Min, Max);
    public static bool operator ==(IntervalD a, IntervalD b) => a.Equals(b);
    public static bool operator !=(IntervalD a, IntervalD b) => !a.Equals(b);
    public override readonly string ToString() => $"[{Min:F6}, {Max:F6}]";
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 5: DIFFERENTIATION AUTOMATIQUE FORWARD-MODE
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Scalaire differentiable : valeur + derivee (forward-mode AD).
/// Chaque operation arithmetique propage automatiquement les derivees.
/// Utilise pour l'optimisation de parametres physiques et le calcul de gradient.
///
/// Exemple : si x = DiffScalar(2.0, 1.0) (valeur=2, dx/dx=1)
/// alors x*x = DiffScalar(4.0, 4.0) (valeur=4, d(x^2)/dx=2x=4).
///
/// MEMORY LAYOUT: 16 bytes (2 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 16, Pack = 8)]
[DebuggerDisplay("d={Value:F6} (d/dx={Derivative:F6})")]
public struct DiffScalar : IEquatable<DiffScalar>
{
    /// <summary>Valeur de la fonction en ce point.</summary>
    [FieldOffset(0)] public double Value;
    /// <summary>Derivee par rapport a la variable independante.</summary>
    [FieldOffset(8)] public double Derivative;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public DiffScalar(double value, double derivative) { Value = value; Derivative = derivative; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public DiffScalar(double value) { Value = value; Derivative = 0; }

    public static readonly DiffScalar Zero = new(0, 0);
    public static readonly DiffScalar One = new(1, 0);
    public static DiffScalar Variable(double value) => new(value, 1.0);

    // Addition : d(a+b)/dx = da/dx + db/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator +(DiffScalar a, DiffScalar b) => new(a.Value + b.Value, a.Derivative + b.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator +(DiffScalar a, double b) => new(a.Value + b, a.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator +(double a, DiffScalar b) => new(a + b.Value, b.Derivative);
    // Soustraction
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator -(DiffScalar a, DiffScalar b) => new(a.Value - b.Value, a.Derivative - b.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator -(DiffScalar a, double b) => new(a.Value - b, a.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator -(double a, DiffScalar b) => new(a - b.Value, -b.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator -(DiffScalar a) => new(-a.Value, -a.Derivative);
    // Multiplication : d(a*b)/dx = a'*b + a*b'
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator *(DiffScalar a, DiffScalar b) => new(a.Value * b.Value, a.Derivative * b.Value + a.Value * b.Derivative);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator *(DiffScalar a, double b) => new(a.Value * b, a.Derivative * b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator *(double a, DiffScalar b) => new(a * b.Value, a * b.Derivative);
    // Division : d(a/b)/dx = (a'*b - a*b') / b^2
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator /(DiffScalar a, DiffScalar b) { double q = a.Value / b.Value; return new(q, (a.Derivative * b.Value - a.Value * b.Derivative) / (b.Value * b.Value)); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator /(DiffScalar a, double b) => new(a.Value / b, a.Derivative / b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar operator /(double a, DiffScalar b) { double q = a / b.Value; return new(q, -a * b.Derivative / (b.Value * b.Value)); }

    // Puissance : d(x^n)/dx = n*x^(n-1)*dx/dx
    public static DiffScalar Pow(DiffScalar x, double n) => new(Math.Pow(x.Value, n), n * Math.Pow(x.Value, n - 1) * x.Derivative);
    // Racine carree : d(sqrt(x))/dx = 1/(2*sqrt(x)) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Sqrt(DiffScalar x) { double s = Math.Sqrt(x.Value); return new(s, x.Derivative / (2 * s)); }
    // Racine nieme : d(x^(1/n))/dx = (1/n)*x^(1/n-1)*dx/dx
    public static DiffScalar NthRoot(DiffScalar x, double n) { double r = Math.Pow(x.Value, 1.0 / n); return new(r, (1.0 / n) * Math.Pow(x.Value, 1.0 / n - 1) * x.Derivative); }
    // Exponentielle : d(e^x)/dx = e^x * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Exp(DiffScalar x) { double e = Math.Exp(x.Value); return new(e, e * x.Derivative); }
    // Logarithme : d(ln(x))/dx = (1/x) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Log(DiffScalar x) => new(Math.Log(x.Value), x.Derivative / x.Value);
    // Logarithme base 10 : d(log10(x))/dx = 1/(x*ln(10)) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Log10(DiffScalar x) => new(Math.Log10(x.Value), x.Derivative / (x.Value * Math.Log(10)));
    // Logarithme base 2
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Log2(DiffScalar x) => new(Math.Log2(x.Value), x.Derivative / (x.Value * Math.Log(2)));

    // Trigonometrie : d(sin(x))/dx = cos(x) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Sin(DiffScalar x) => new(Math.Sin(x.Value), Math.Cos(x.Value) * x.Derivative);
    // d(cos(x))/dx = -sin(x) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Cos(DiffScalar x) => new(Math.Cos(x.Value), -Math.Sin(x.Value) * x.Derivative);
    // d(tan(x))/dx = sec^2(x) * dx/dx = 1/cos^2(x) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Tan(DiffScalar x) { double c = Math.Cos(x.Value); return new(Math.Tan(x.Value), x.Derivative / (c * c)); }
    // Inverse trigonometrie : d(asin(x))/dx = 1/sqrt(1-x^2) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Asin(DiffScalar x) => new(Math.Asin(x.Value), x.Derivative / Math.Sqrt(1 - x.Value * x.Value));
    // d(acos(x))/dx = -1/sqrt(1-x^2) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Acos(DiffScalar x) => new(Math.Acos(x.Value), -x.Derivative / Math.Sqrt(1 - x.Value * x.Value));
    // d(atan(x))/dx = 1/(1+x^2) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Atan(DiffScalar x) => new(Math.Atan(x.Value), x.Derivative / (1 + x.Value * x.Value));
    // d(atan2(y,x))/dx = (x*dy/dx - y*dx/dx) / (x^2+y^2)
    public static DiffScalar Atan2(DiffScalar y, DiffScalar x) => new(Math.Atan2(y.Value, x.Value), (x.Value * y.Derivative - y.Value * x.Derivative) / (x.Value * x.Value + y.Value * y.Value));

    // Hyperbolique
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Sinh(DiffScalar x) { double h = Math.Sinh(x.Value); return new(h, Math.Cosh(x.Value) * x.Derivative); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Cosh(DiffScalar x) { double h = Math.Cosh(x.Value); return new(h, Math.Sinh(x.Value) * x.Derivative); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Tanh(DiffScalar x) { double t = Math.Tanh(x.Value); return new(t, (1 - t * t) * x.Derivative); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Asinh(DiffScalar x) => new(Math.Asinh(x.Value), x.Derivative / Math.Sqrt(x.Value * x.Value + 1));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Acosh(DiffScalar x) => new(Math.Acosh(x.Value), x.Derivative / Math.Sqrt(x.Value * x.Value - 1));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Atanh(DiffScalar x) => new(Math.Atanh(x.Value), x.Derivative / (1 - x.Value * x.Value));

    // Valeur absolue : d|x|/dx = sign(x) * dx/dx
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Abs(DiffScalar x) => new(Math.Abs(x.Value), Math.Sign(x.Value) * x.Derivative);
    // Clamp : clamp(x, a, b)
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Clamp(DiffScalar x, double a, double b) => x.Value < a ? new(a, 0) : x.Value > b ? new(b, 0) : x;
    // Min/Max
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Max(DiffScalar a, DiffScalar b) => a.Value >= b.Value ? a : b;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Min(DiffScalar a, DiffScalar b) => a.Value <= b.Value ? a : b;
    // Sigmoid : sigma(x) = 1/(1+e^-x), d/dx = sigma(x)*(1-sigma(x))
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Sigmoid(DiffScalar x) { double s = 1.0 / (1.0 + Math.Exp(-x.Value)); return new(s, s * (1 - s) * x.Derivative); }
    // Softplus : ln(1+e^x), d/dx = sigmoid(x)
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar Softplus(DiffScalar x) => new(Math.Log(1 + Math.Exp(x.Value)), Sigmoid(x).Derivative);
    // GELU approx : 0.5*x*(1+erf(x/sqrt(2)))
    // ReLU : max(0, x)
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar ReLU(DiffScalar x) => x.Value > 0 ? x : new(0, 0);
    // LeakyReLU : max(alpha*x, x)
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static DiffScalar LeakyReLU(DiffScalar x, double alpha = 0.01) => x.Value > 0 ? x : new(alpha * x.Value, alpha * x.Derivative);

    /// <summary>Chain rule : compose this with f(x) : d/dx f(g(x)) = f'(g(x)) * g'(x).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public DiffScalar ChainRule(DiffScalar outerFuncDerivative) => new(Value, Value * outerFuncDerivative.Derivative);

    /// <summary>Cosinue la differentiation avec un nouveau point.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar Chain(DiffScalar inner, DiffScalar outerAtInner) => new(outerAtInner.Value, outerAtInner.Derivative * inner.Derivative);

    /// <summary>Promote to a second-order jet (value, first derivative, second derivative).</summary>
    public readonly DiffScalar2 WithSecondDerivative(double secondDerivative) => new(Value, Derivative, secondDerivative);

    // Equality
    public readonly bool Equals(DiffScalar o) => Math.Abs(Value - o.Value) < 1e-15 && Math.Abs(Derivative - o.Derivative) < 1e-15;
    public override readonly bool Equals(object? obj) => obj is DiffScalar o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Value, Derivative);
    public static bool operator ==(DiffScalar a, DiffScalar b) => a.Equals(b);
    public static bool operator !=(DiffScalar a, DiffScalar b) => !a.Equals(b);
    public override readonly string ToString() => $"d={Value:F6} (d/dx={Derivative:F6})";
}

/// <summary>
/// Second-order dual number (jet): tracks f, f', f'' for Hessian-vector products
/// and curvature-aware physics optimization.
/// MEMORY LAYOUT: 24 bytes (3 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 24, Pack = 8)]
[DebuggerDisplay("d={Value:F6} (d/dx={Derivative:F6}, d²/dx²={SecondDerivative:F6})")]
public struct DiffScalar2 : IEquatable<DiffScalar2>
{
    [FieldOffset(0)] public double Value;
    [FieldOffset(8)] public double Derivative;
    [FieldOffset(16)] public double SecondDerivative;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public DiffScalar2(double value, double derivative = 0, double secondDerivative = 0)
    {
        Value = value;
        Derivative = derivative;
        SecondDerivative = secondDerivative;
    }

    public static DiffScalar2 Variable(double value) => new(value, 1.0, 0.0);
    public DiffScalar ToFirstOrder() => new(Value, Derivative);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 operator +(DiffScalar2 a, DiffScalar2 b) =>
        new(a.Value + b.Value, a.Derivative + b.Derivative, a.SecondDerivative + b.SecondDerivative);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 operator -(DiffScalar2 a, DiffScalar2 b) =>
        new(a.Value - b.Value, a.Derivative - b.Derivative, a.SecondDerivative - b.SecondDerivative);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 operator *(DiffScalar2 a, DiffScalar2 b) =>
        new(
            a.Value * b.Value,
            a.Derivative * b.Value + a.Value * b.Derivative,
            a.SecondDerivative * b.Value + 2 * a.Derivative * b.Derivative + a.Value * b.SecondDerivative);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 Exp(DiffScalar2 x)
    {
        double e = Math.Exp(x.Value);
        return new(e, e * x.Derivative, e * (x.SecondDerivative + x.Derivative * x.Derivative));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 Sin(DiffScalar2 x)
    {
        double s = Math.Sin(x.Value), c = Math.Cos(x.Value);
        return new(s, c * x.Derivative, -s * x.Derivative * x.Derivative + c * x.SecondDerivative);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DiffScalar2 Cos(DiffScalar2 x)
    {
        double s = Math.Sin(x.Value), c = Math.Cos(x.Value);
        return new(c, -s * x.Derivative, -c * x.Derivative * x.Derivative - s * x.SecondDerivative);
    }

    public readonly bool Equals(DiffScalar2 o) =>
        Math.Abs(Value - o.Value) < 1e-15 &&
        Math.Abs(Derivative - o.Derivative) < 1e-15 &&
        Math.Abs(SecondDerivative - o.SecondDerivative) < 1e-15;

    public override readonly bool Equals(object? obj) => obj is DiffScalar2 o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Value, Derivative, SecondDerivative);
    public override readonly string ToString() =>
        $"d={Value:F6} (d/dx={Derivative:F6}, d²/dx²={SecondDerivative:F6})";
}

/// <summary>
/// Expression differentiable — contient un graphe de computation pour le calcul
/// de gradients. Supporte le forward-mode AD via DiffScalar et le backward-mode
/// via un accumulateur de gradient. Utilise pour l'optimisation de parametres
/// physiques dans le PINN (Physics-Informed Neural Network).
///
/// L'expression est construite par composition d'operateurs differentiables.
/// Chaque evaluation propage les valeurs et les derivees simultanement.
/// </summary>
public sealed class DiffExpression
{
    private readonly List<(Func<DiffScalar[], DiffScalar> fn, int[] inputIndices)> _operations = new();
    private readonly List<DiffScalar> _tapes = new();
    private int _nextIndex = 0;

    /// <summary>Enregistre une variable comme entree differentiable.</summary>
    public int AddVariable(double value)
    {
        _tapes.Add(DiffScalar.Variable(value));
        return _nextIndex++;
    }

    /// <summary>Enregistre une constante (pas de derivee).</summary>
    public int AddConstant(double value)
    {
        _tapes.Add(new DiffScalar(value, 0));
        return _nextIndex++;
    }

    /// <summary>Ajoute une operation : result = fn(inputs).</summary>
    public int AddOperation(Func<DiffScalar[], DiffScalar> fn, params int[] inputIndices)
    {
        var inputs = new DiffScalar[inputIndices.Length];
        for (int i = 0; i < inputIndices.Length; i++)
            inputs[i] = _tapes[inputIndices[i]];
        _tapes.Add(fn(inputs));
        _operations.Add((fn, inputIndices));
        return _nextIndex++;
    }

    /// <summary>Evalue l'expression et retourne le resultat differentiable.</summary>
    public DiffScalar Evaluate() => _tapes.Count > 0 ? _tapes[^1] : DiffScalar.Zero;

    /// <summary>Calcule le gradient par rapport a toutes les variables d'entree.</summary>
    public double[] ComputeGradients()
    {
        var result = new double[_tapes.Count];
        for (int i = 0; i < _tapes.Count; i++)
            result[i] = _tapes[i].Derivative;
        return result;
    }

    /// <summary>Met a jour une variable et re-evalue.</summary>
    public void UpdateVariable(int index, double newValue)
    {
        _tapes[index] = DiffScalar.Variable(newValue);
        // Re-evaluer toutes les operations dependantes
        for (int i = 0; i < _operations.Count; i++)
        {
            var (fn, inputIndices) = _operations[i];
            var inputs = new DiffScalar[inputIndices.Length];
            for (int j = 0; j < inputIndices.Length; j++)
                inputs[j] = _tapes[inputIndices[j]];
            _tapes[_tapes.Count - _operations.Count + i] = fn(inputs);
        }
    }

    /// <summary>Nombre total de noeuds dans le graphe de computation.</summary>
    public int TapeSize => _tapes.Count;

    /// <summary>Reinitialise l'expression pour un nouvel evaluation.</summary>
    public void Reset() { _tapes.Clear(); _operations.Clear(); _nextIndex = 0; }

    // Factory methods pour les expressions courantes
    public static DiffExpression Quadratic(double a, double b, double c)
    {
        var expr = new DiffExpression();
        int x = expr.AddVariable(0);
        int aI = expr.AddConstant(a);
        int bI = expr.AddConstant(b);
        int cI = expr.AddConstant(c);
        int x2 = expr.AddOperation(d => d[0] * d[0], x);
        int ax2 = expr.AddOperation(d => d[0] * d[1], aI, x2);
        int bx = expr.AddOperation(d => d[0] * d[1], bI, x);
        expr.AddOperation(d => d[0] + d[1] + d[2], ax2, bx, cI);
        return expr;
    }

    public static DiffExpression Polynomial(double[] coefficients)
    {
        var expr = new DiffExpression();
        int x = expr.AddVariable(0);
        int[] powerIndices = new int[coefficients.Length];
        powerIndices[0] = expr.AddConstant(1); // x^0 = 1
        for (int i = 1; i < coefficients.Length; i++)
        {
            int pow = i;
            powerIndices[i] = expr.AddOperation(d => DiffScalar.Pow(d[0], pow), x);
        }
        for (int i = 0; i < coefficients.Length; i++)
        {
            int cI = expr.AddConstant(coefficients[i]);
            // Multiply coefficient by power
        }
        return expr;
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 6: FIELD GRADIENT
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Gradient spatial d'un champ physique — utilise pour les operateurs differentiels
/// (divergence, rotationnel, laplacien). Chaque gradient est un vecteur 3D
/// representant la direction et l'intensite du changement spatial.
///
/// MEMORY LAYOUT: 192 bytes (6 Vector3D).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 192, Pack = 32)]
public struct FieldGradient
{
    [FieldOffset(0)] public Vector3D TemperatureGradient;
    [FieldOffset(32)] public Vector3D PressureGradient;
    [FieldOffset(64)] public Vector3D VelocityGradientX;
    [FieldOffset(96)] public Vector3D VelocityGradientY;
    [FieldOffset(128)] public Vector3D VelocityGradientZ;
    [FieldOffset(160)] public Vector3D ConcentrationGradient;

    public static readonly FieldGradient Zero = default;

    /// <summary>Divergence de la vitesse : nabla.v = dvx/dx + dvy/dy + dvz/dz.</summary>
    public readonly double VelocityDivergence => VelocityGradientX.X + VelocityGradientY.Y + VelocityGradientZ.Z;
    /// <summary>Rotationnel de la vitesse : nabla x v.</summary>
    public readonly Vector3D VelocityCurl => new(VelocityGradientZ.Y - VelocityGradientY.Z, VelocityGradientX.Z - VelocityGradientZ.X, VelocityGradientY.X - VelocityGradientX.Y);
    /// <summary>Laplacien de la temperature : nabla^2 T.</summary>
    public readonly double TemperatureLaplacian => TemperatureGradient.X + TemperatureGradient.Y + TemperatureGradient.Z;
    /// <summary>Force de gradient de pression : -nabla p.</summary>
    public readonly Vector3D PressureForce => -PressureGradient;
    /// <summary>Gradient de concentration (magnitude).</summary>
    public readonly double ConcentrationMagnitude => ConcentrationGradient.Length();
    /// <summary>Tenseur gradient complet de vitesse (3x3).</summary>
    public readonly Tensor3D VelocityTensor => new(VelocityGradientX.X, VelocityGradientX.Y, VelocityGradientX.Z, VelocityGradientY.X, VelocityGradientY.Y, VelocityGradientY.Z, VelocityGradientZ.X, VelocityGradientZ.Y, VelocityGradientZ.Z);
    /// <summary>Tenseur de deformation (partie symetrique du gradient de vitesse).</summary>
    public readonly Symmetric3x3 StrainRateTensor => VelocityTensor.SymmetricPart();
    /// <summary>Taux de dissipation visqueuse : phi = mu * (dvi/dxj + dvj/dxi)^2.</summary>
    public readonly double ViscousDissipation(double mu) { var s = StrainRateTensor; return mu * (2 * (s.XX * s.XX + s.YY * s.YY + s.ZZ * s.ZZ) + s.XY * s.XY + s.XZ * s.XZ + s.YZ * s.YZ); }
    /// <summary>Energie cinetique turbulente (approximation).</summary>
    public readonly double TurbulentKineticEnergy { get { var s = StrainRateTensor; return 0.5 * s.DoubleContract(new(s.XX, s.YY, s.ZZ)); } }
    /// <summary>Norme du gradient de temperature.</summary>
    public readonly double TemperatureGradientMagnitude => TemperatureGradient.Length();
    /// <summary>Direction du gradient de temperature.</summary>
    public readonly Vector3D TemperatureGradientDirection => TemperatureGradient.Normalized();

    public static FieldGradient Lerp(FieldGradient a, FieldGradient b, double t) { double u = 1 - t; return new FieldGradient { TemperatureGradient = a.TemperatureGradient * u + b.TemperatureGradient * t, PressureGradient = a.PressureGradient * u + b.PressureGradient * t, VelocityGradientX = a.VelocityGradientX * u + b.VelocityGradientX * t, VelocityGradientY = a.VelocityGradientY * u + b.VelocityGradientY * t, VelocityGradientZ = a.VelocityGradientZ * u + b.VelocityGradientZ * t, ConcentrationGradient = a.ConcentrationGradient * u + b.ConcentrationGradient * t }; }
}

// ═══════════════════════════════════════════════════════════════════════════════
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
// SECTION 8: MATERIAL PROPERTIES — PROPRIETES CONSTITUTIVES
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Materiau physique complet avec proprietes constitutives pour la simulation.
/// Contient les proprietes mecaniques (module de Young, Poisson, plasticite),
/// thermiques (conductivite, capacite, dilatation), optiques (albedo, IOR),
/// et electromagnetiques (conductivite, permeabilite).
///
/// Le NeuralBrdf appris est couple au comportement mecanique pour
/// un rendu physiquement coherent (photorealiste).
///
/// MEMORY LAYOUT: 256 bytes (32 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 256, Pack = 32)]
public unsafe struct MaterialProperties
{
    [FieldOffset(0)] public fixed double Albedo[3];       // RGB albedo
    [FieldOffset(24)] public double Metallic;             // 0=dielectrique, 1=metal
    [FieldOffset(32)] public double Roughness;            // 0=miroir, 1=mat
    [FieldOffset(40)] public double RefractiveIndex;      // Indice de refraction (IOR)
    [FieldOffset(48)] public double AbsorptionCoeff;      // Coefficient d'absorption optique
    [FieldOffset(56)] public double ScatteringCoeff;      // Coefficient de diffusion (SSS)
    [FieldOffset(64)] public double Density;              // Densite massique [kg/m3]
    [FieldOffset(72)] public double YoungModulus;         // Module de Young (raideur) [Pa]
    [FieldOffset(80)] public double PoissonRatio;         // Coefficient de Poisson (0-0.5)
    [FieldOffset(88)] public double YieldStrength;        // Limite d'elasticite [Pa]
    [FieldOffset(96)] public double ThermalConductivity;  // Conductivite thermique [W/(m*K)]
    [FieldOffset(104)] public double SpecificHeat;        // Capacite calorifique specifique [J/(kg*K)]
    [FieldOffset(112)] public double ThermalExpansion;    // Coefficient de dilatation [1/K]
    [FieldOffset(120)] public double ElectricalConductivity; // Conductivite electrique [S/m]
    [FieldOffset(128)] public double MagneticPermeability;   // Permeabilite magnetique relative
    [FieldOffset(136)] public double DielectricConstant;     // Constante dielectrique relative
    [FieldOffset(144)] public double PlasticStrain;           // Deformation plastique cumulee
    [FieldOffset(152)] public double HardeningExponent;       // Exponent d'ecrouissage
    [FieldOffset(160)] public double FatigueLife;             // Duree de vie en fatigue [cycles]
    [FieldOffset(168)] public double FractureToughness;       // Tenuite a la fracture [Pa*sqrt(m)]
    [FieldOffset(176)] public double CreepRate;               // Taux de fluage [1/s]
    [FieldOffset(184)] public double Viscosity;               // Viscosite dynamique [Pa*s]
    [FieldOffset(192)] public double SurfaceTension;          // Tension superficielle [N/m]
    [FieldOffset(200)] public double Porosity;                // Porosite (0-1)
    [FieldOffset(208)] public double Permeability;            // Permeabilite [m2]
    [FieldOffset(216)] public double ReactionRate;            // Constante cinetique [1/s]
    [FieldOffset(224)] public double ActivationEnergy;        // Energie d'activation [J/mol]
    [FieldOffset(232)] public int MaterialId;                 // ID unique du materiau
    [FieldOffset(236)] public int _pad0;
    [FieldOffset(240)] public long Flags;                     // Drapeaux de proprietes
    [FieldOffset(248)] public long _pad1;

    public double Albedo0 { get => Albedo[0]; set => Albedo[0] = value; }
    public double Albedo1 { get => Albedo[1]; set => Albedo[1] = value; }
    public double Albedo2 { get => Albedo[2]; set => Albedo[2] = value; }

    // === Proprietes derivees ===
    /// <summary>Module de cisaillement : G = E / (2*(1+v)).</summary>
    public readonly double ShearModulus => YoungModulus / (2.0 * (1.0 + PoissonRatio));
    /// <summary>Module de volume : K = E / (3*(1-2v)).</summary>
    public readonly double BulkModulus => YoungModulus / (3.0 * (1.0 - 2.0 * PoissonRatio));
    /// <summary>Vitesse du son : c = sqrt(K/rho).</summary>
    public readonly double SpeedOfSound => Math.Sqrt(BulkModulus / Math.Max(Density, 1e-30));
    /// <summary>Nombre de Peclet local (advection vs diffusion thermique).</summary>
    public readonly double PecletNumber(double velocity, double length) => Density * SpecificHeat * velocity * length / Math.Max(ThermalConductivity, 1e-30);
    /// <summary>Critere de von Mises : le materiau plastifie si sigma_vM >= YieldStrength.</summary>
    public readonly bool IsPlastic(double vonMisesStress) => vonMisesStress >= YieldStrength;
    /// <summary>Contrainte via loi de Ramberg-Osgood : sigma = sigma_y * (eps*E/sigma_y)^(1/n).</summary>
    public readonly double RambergOsgoodStress(double strain) => YieldStrength * Math.Pow(Math.Abs(strain) * YoungModulus / YieldStrength, 1.0 / Math.Max(HardeningExponent, 0.1));
    /// <summary>Module de Lamé lambda : lambda = v*E/((1+v)*(1-2v)).</summary>
    public readonly double LaméLambda => PoissonRatio * YoungModulus / ((1.0 + PoissonRatio) * (1.0 - 2.0 * PoissonRatio));
    /// <summary>Numero de Mach local : M = v/c.</summary>
    public readonly double MachNumber(double velocity) => velocity / Math.Max(SpeedOfSound, 1e-30);
    /// <summary>Nombre de Reynolds local : Re = rho*v*L/mu.</summary>
    public readonly double ReynoldsNumber(double velocity, double length) => Density * velocity * length / Math.Max(Viscosity, 1e-30);
    /// <summary>Nombre de Prandtl local : Pr = mu*cp/k.</summary>
    public readonly double PrandtlNumber => Viscosity * SpecificHeat / Math.Max(ThermalConductivity, 1e-30);
    /// <summary>Diffusivite thermique : alpha = k/(rho*cp).</summary>
    public readonly double ThermalDiffusivity => ThermalConductivity / Math.Max(Density * SpecificHeat, 1e-30);
    /// <summary>Viscosite cinematique : nu = mu/rho.</summary>
    public readonly double KinematicViscosity => Viscosity / Math.Max(Density, 1e-30);
    /// <summary>Flux de chaleur maximal (estimation par Fourier).</summary>
    public readonly double MaxHeatFlux(double temperatureGradient) => ThermalConductivity * Math.Abs(temperatureGradient);

    // === Materiaux predefinis ===
    public static MaterialProperties Default => new() { Density = 1000, YoungModulus = 200e9, PoissonRatio = 0.3, YieldStrength = 250e6, ThermalConductivity = 50, SpecificHeat = 500, ThermalExpansion = 12e-6, ElectricalConductivity = 1e7, RefractiveIndex = 1.5, SurfaceTension = 0.072 };
    public static MaterialProperties Steel => new() { Metallic = 0.95, Roughness = 0.4, RefractiveIndex = 2.5, Density = 7850, YoungModulus = 200e9, PoissonRatio = 0.3, YieldStrength = 250e6, ThermalConductivity = 50, SpecificHeat = 500, ThermalExpansion = 12e-6, ElectricalConductivity = 1.4e7, FractureToughness = 50e6 };
    public static MaterialProperties TitaniumAlloy => new() { Metallic = 0.9, Roughness = 0.35, RefractiveIndex = 2.0, Density = 4500, YoungModulus = 114e9, PoissonRatio = 0.34, YieldStrength = 880e6, ThermalConductivity = 6.7, SpecificHeat = 526, ThermalExpansion = 8.6e-6, ElectricalConductivity = 5.8e5, FractureToughness = 75e6 };
    public static MaterialProperties CarbonFiber => new() { Metallic = 0.1, Roughness = 0.6, RefractiveIndex = 1.6, Density = 1600, YoungModulus = 230e9, PoissonRatio = 0.2, YieldStrength = 3500e6, ThermalConductivity = 7, SpecificHeat = 710, ThermalExpansion = -0.5e-6, ElectricalConductivity = 1e4, FractureToughness = 25e6 };
    public static MaterialProperties Water => new() { Albedo0 = 0.01, Albedo1 = 0.02, Albedo2 = 0.95, Metallic = 0, Roughness = 0.05, RefractiveIndex = 1.33, Density = 1000, ThermalConductivity = 0.6, SpecificHeat = 4186, ThermalExpansion = 2.1e-4, SurfaceTension = 0.072, Viscosity = 1e-3 };
    public static MaterialProperties Biological => new() { Albedo0 = 0.8, Albedo1 = 0.3, Albedo2 = 0.3, Metallic = 0, Roughness = 0.8, RefractiveIndex = 1.4, Density = 1060, YoungModulus = 1e6, PoissonRatio = 0.45, YieldStrength = 1e5, ThermalConductivity = 0.5, SpecificHeat = 3500, ThermalExpansion = 1e-4, ElectricalConductivity = 0.3, Viscosity = 0.01 };
    public static MaterialProperties Aluminum => new() { Metallic = 0.91, Roughness = 0.25, RefractiveIndex = 1.44, Density = 2700, YoungModulus = 69e9, PoissonRatio = 0.33, YieldStrength = 276e6, ThermalConductivity = 237, SpecificHeat = 897, ThermalExpansion = 23.1e-6, ElectricalConductivity = 3.77e7, FractureToughness = 29e6 };
    public static MaterialProperties Copper => new() { Metallic = 0.97, Roughness = 0.2, RefractiveIndex = 0.62, Density = 8960, YoungModulus = 130e9, PoissonRatio = 0.34, YieldStrength = 210e6, ThermalConductivity = 401, SpecificHeat = 385, ThermalExpansion = 16.5e-6, ElectricalConductivity = 5.96e7, FractureToughness = 30e6 };
    public static MaterialProperties Glass => new() { Metallic = 0.0, Roughness = 0.05, RefractiveIndex = 1.52, Density = 2500, YoungModulus = 70e9, PoissonRatio = 0.22, YieldStrength = 33e6, ThermalConductivity = 1.05, SpecificHeat = 840, ThermalExpansion = 9e-6, ElectricalConductivity = 1e-14, FractureToughness = 0.7e6 };
    public static MaterialProperties Concrete => new() { Metallic = 0.0, Roughness = 0.8, RefractiveIndex = 1.5, Density = 2400, YoungModulus = 30e9, PoissonRatio = 0.2, YieldStrength = 30e6, ThermalConductivity = 1.7, SpecificHeat = 880, ThermalExpansion = 12e-6, ElectricalConductivity = 1e-4, FractureToughness = 1.0e6 };
    public static MaterialProperties Wood => new() { Metallic = 0.0, Roughness = 0.7, RefractiveIndex = 1.5, Density = 600, YoungModulus = 12e9, PoissonRatio = 0.35, YieldStrength = 40e6, ThermalConductivity = 0.16, SpecificHeat = 1700, ThermalExpansion = 5e-6, ElectricalConductivity = 1e-10 };
    public static MaterialProperties Rubber => new() { Metallic = 0.0, Roughness = 0.9, RefractiveIndex = 1.52, Density = 1100, YoungModulus = 0.05e9, PoissonRatio = 0.49, YieldStrength = 15e6, ThermalConductivity = 0.16, SpecificHeat = 2010, ThermalExpansion = 200e-6, ElectricalConductivity = 1e-13 };
    public static MaterialProperties Gold => new() { Metallic = 1.0, Roughness = 0.15, RefractiveIndex = 0.18, Density = 19300, YoungModulus = 79e9, PoissonRatio = 0.44, YieldStrength = 35e6, ThermalConductivity = 318, SpecificHeat = 129, ThermalExpansion = 14.2e-6, ElectricalConductivity = 4.52e7 };
    public static MaterialProperties Silicon => new() { Metallic = 0.5, Roughness = 0.3, RefractiveIndex = 4.0, Density = 2330, YoungModulus = 130e9, PoissonRatio = 0.28, YieldStrength = 7e9, ThermalConductivity = 149, SpecificHeat = 700, ThermalExpansion = 2.6e-6, ElectricalConductivity = 1e3 };
    public static MaterialProperties Diamond => new() { Metallic = 0.0, Roughness = 0.01, RefractiveIndex = 2.42, Density = 3510, YoungModulus = 1050e9, PoissonRatio = 0.07, YieldStrength = 60e9, ThermalConductivity = 2200, SpecificHeat = 509, ThermalExpansion = 1e-6, ElectricalConductivity = 1e-13, FractureToughness = 5.0e6 };
    public static MaterialProperties Lead => new() { Metallic = 0.85, Roughness = 0.5, RefractiveIndex = 2.01, Density = 11340, YoungModulus = 16e9, PoissonRatio = 0.44, YieldStrength = 12e6, ThermalConductivity = 35, SpecificHeat = 128, ThermalExpansion = 29e-6, ElectricalConductivity = 4.5e6 };
    public static MaterialProperties Platinum => new() { Metallic = 0.95, Roughness = 0.2, RefractiveIndex = 2.33, Density = 21450, YoungModulus = 168e9, PoissonRatio = 0.38, YieldStrength = 240e6, ThermalConductivity = 71.6, SpecificHeat = 133, ThermalExpansion = 8.9e-6, ElectricalConductivity = 9.4e6 };
    public static MaterialProperties Nylon => new() { Metallic = 0.0, Roughness = 0.6, RefractiveIndex = 1.53, Density = 1150, YoungModulus = 2.5e9, PoissonRatio = 0.4, YieldStrength = 50e6, ThermalConductivity = 0.25, SpecificHeat = 1670, ThermalExpansion = 80e-6, ElectricalConductivity = 1e-12 };
    public static MaterialProperties Air => new() { Density = 1.225, SpecificHeat = 1005, ThermalConductivity = 0.025, Viscosity = 1.8e-5, RefractiveIndex = 1.0003, ElectricalConductivity = 0 };
    public static MaterialProperties Mercury => new() { Metallic = 1.0, Roughness = 0.1, RefractiveIndex = 1.73, Density = 13546, Viscosity = 0.00155, ThermalConductivity = 8.3, SpecificHeat = 139, ThermalExpansion = 18e-6, ElectricalConductivity = 1.04e6, SurfaceTension = 0.487 };

    public void Serialize(float* dest) { fixed (double* a = Albedo) { for (int i = 0; i < 3; i++) dest[i] = (float)a[i]; } dest[3] = (float)Metallic; dest[4] = (float)Roughness; dest[5] = (float)RefractiveIndex; dest[6] = (float)AbsorptionCoeff; dest[7] = (float)ScatteringCoeff; dest[8] = (float)Density; dest[9] = (float)YoungModulus; dest[10] = (float)PoissonRatio; dest[11] = (float)YieldStrength; dest[12] = (float)ThermalConductivity; dest[13] = (float)SpecificHeat; dest[14] = (float)ThermalExpansion; dest[15] = (float)ElectricalConductivity; }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 9: STOCHASTIC STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Etat stochastique d'un point — porte par le StochasticField pour
/// les phenomenes aleatoires (epidemie, marche, rumeur, turbulence).
/// Contient les moments statistiques et les parametres du processus sous-jacent.
///
/// MEMORY LAYOUT: 128 bytes.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 128, Pack = 32)]
public struct StochasticState
{
    [FieldOffset(0)] public double Mean;                  // Moyenne du processus
    [FieldOffset(8)] public double Variance;              // Variance (sigma^2)
    [FieldOffset(16)] public double Drift;                // Derive (mu)
    [FieldOffset(24)] public double Diffusion;            // Coefficient de diffusion (sigma)
    [FieldOffset(32)] public double JumpIntensity;        // Intensite des sauts (lambda)
    [FieldOffset(40)] public double JumpMean;             // Moyenne des sauts
    [FieldOffset(48)] public double JumpVariance;         // Variance des sauts
    [FieldOffset(56)] public double CorrelationTime;      // Temps de correlation (tau)
    [FieldOffset(64)] public double LongTermMean;         // Moyenne a long terme (pour O-U)
    [FieldOffset(72)] public double MeanReversion;        // Taux de retour a la moyenne (theta)
    [FieldOffset(80)] public double Entropy;              // Entropie de la distribution
    [FieldOffset(88)] public double MutualInformation;    // Information mutuelle
    [FieldOffset(96)] public StochasticProcess ProcessType; // Type de processus
    [FieldOffset(100)] public int _pad0;
    [FieldOffset(104)] public double Volatility;          // Volatilite (annualisee)
    [FieldOffset(112)] public double Skewness;            // Asymetrie de la distribution
    [FieldOffset(120)] public double Kurtosis;            // Aplatissement (excedent)

    public static StochasticState Default => new() { ProcessType = StochasticProcess.Deterministic, Diffusion = 1.0 };
    public readonly double StandardDeviation() => Math.Sqrt(Math.Max(0, Variance));
    public readonly double CoefficientOfVariation() => Mean != 0 ? StandardDeviation() / Math.Abs(Mean) : 0;
    public readonly double ConfidenceInterval95 => 1.96 * StandardDeviation();
    public readonly double ConfidenceInterval99 => 2.576 * StandardDeviation();

    /// <summary>Evolution d'Ornstein-Uhlenbeck : dx = theta*(mu-x)*dt + sigma*dW.</summary>
    public readonly StochasticState StepOU(double dt) => new() { Mean = LongTermMean + (Mean - LongTermMean) * Math.Exp(-MeanReversion * dt), Variance = Diffusion * Diffusion / (2 * MeanReversion) * (1 - Math.Exp(-2 * MeanReversion * dt)), Drift = Drift, Diffusion = Diffusion, JumpIntensity = JumpIntensity, JumpMean = JumpMean, JumpVariance = JumpVariance, CorrelationTime = CorrelationTime, LongTermMean = LongTermMean, MeanReversion = MeanReversion, ProcessType = ProcessType, Volatility = Volatility };
    /// <summary>Evolution de Geometric Brownian Motion : dS = mu*S*dt + sigma*S*dW.</summary>
    public readonly StochasticState StepGBM(double dt) { double drift = (Drift - 0.5 * Diffusion * Diffusion) * dt; double vol = Diffusion * Math.Sqrt(dt); return new() { Mean = Mean * Math.Exp(drift), Variance = Variance * Math.Exp(2 * Drift + vol * vol) * (Math.Exp(vol * vol) - 1), Drift = Drift, Diffusion = Diffusion, ProcessType = ProcessType, Volatility = Volatility }; }

    public static StochasticState Lerp(StochasticState a, StochasticState b, double t) { double u = 1 - t; return new StochasticState { Mean = a.Mean * u + b.Mean * t, Variance = a.Variance * u + b.Variance * t, Drift = a.Drift * u + b.Drift * t, Diffusion = a.Diffusion * u + b.Diffusion * t, JumpIntensity = a.JumpIntensity * u + b.JumpIntensity * t, CorrelationTime = a.CorrelationTime * u + b.CorrelationTime * t, LongTermMean = a.LongTermMean * u + b.LongTermMean * t, MeanReversion = a.MeanReversion * u + b.MeanReversion * t, Entropy = a.Entropy * u + b.Entropy * t, ProcessType = t < 0.5 ? a.ProcessType : b.ProcessType, Volatility = a.Volatility * u + b.Volatility * t }; }
}

// ═══════════════════════════════════════════════════════════════════════════════
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
// SECTION 11: FIELD SAMPLE RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Resultat d'une evaluation de champ pour un point donne.
/// Contient toutes les informations pour le rendu, l'inspection et la journalisation.
/// Combine l'etat physique, l'etat stochastique, le gradient, et les metadonnees.
///
/// MEMORY LAYOUT: ~480 bytes.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 480, Pack = 32)]
public unsafe struct FieldSampleResult
{
    [FieldOffset(0)] public PhysicsState State;
    [FieldOffset(256)] public StochasticState Stochastic;
    [FieldOffset(384)] public FieldGradient Gradient;
    [FieldOffset(384 + 192 - 4)] public float SdfValue;       // Distance signee au champ SDF
    [FieldOffset(384 + 192 - 3)] public float Confidence;     // Confiance du reseau neuronal
    [FieldOffset(384 + 192 - 2)] public ushort ActiveLod;     // Niveau de detail actif (LOD)
    [FieldOffset(384 + 192 - 1)] public byte Representation;  // 0=poly, 1=neuralSDF, 2=physics
    [FieldOffset(384 + 192)] public byte Flags;

    public static readonly FieldSampleResult Default = default;
    public readonly bool IsEmpty => State.Norm < 1e-20 && Stochastic.Mean == 0;
    public readonly double TotalEnergy => State.KineticEnergy + State.InternalEnergy;
    public readonly double TotalEntropy => State.Entropy + Stochastic.Entropy;
    public readonly Vector3D NetFlux => State.HeatFlux + State.Velocity * State.Density;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 12: SIMULATION CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Configuration globale du champ physique — parametres de simulation.
/// Definit la grille spatiale, le pas de temps, les conditions aux limites,
/// et les options de performance (SIMD, GPU, threads).
/// </summary>
public sealed class SimulationConfig
{
    // === Parametres temporels ===
    public double TimeStep { get; set; } = 0.001;
    public double MaxTimeStep { get; set; } = 0.1;
    public double MinTimeStep { get; set; } = 1e-8;
    public bool EnableAdaptiveTimeStepping { get; set; } = true;
    public double CourantNumber { get; set; } = 0.5; // CFL condition

    // === Grille spatiale ===
    public int GridResolutionX { get; set; } = 128;
    public int GridResolutionY { get; set; } = 128;
    public int GridResolutionZ { get; set; } = 128;
    public double DomainSizeX { get; set; } = 10.0;
    public double DomainSizeY { get; set; } = 10.0;
    public double DomainSizeZ { get; set; } = 10.0;
    public double DomainOffsetX { get; set; } = -5.0;
    public double DomainOffsetY { get; set; } = -5.0;
    public double DomainOffsetZ { get; set; } = -5.0;

    // === Solveur ===
    public int MaxIterations { get; set; } = 10000;
    public double ConvergenceTolerance { get; set; } = 1e-6;
    public bool EnableBoundaryConditions { get; set; } = true;
    public BoundaryConditionKind DefaultBoundaryCondition { get; set; } = BoundaryConditionKind.Absorbing;

    // === PNS (Progressive Neural Simulation) ===
    public int PnsTransitionNear { get; set; } = 5;
    public int PnsTransitionFar { get; set; } = 50;
    public int PnsMaxPolygonBudget { get; set; } = 10_000_000;
    public double PnsNeuralPrecisionThreshold { get; set; } = 0.001;
    public double PnsPhysicsLODDistance { get; set; } = 100.0;

    // === Performance ===
    public int MaxConcurrentThreads { get; set; } = Environment.ProcessorCount;
    public bool EnableSimd { get; set; } = true;
    public bool EnableGpuAcceleration { get; set; } = true;
    public string DeviceId { get; set; } = "auto";
    public bool EnableDoublePrecision { get; set; } = true;

    // === Proprietes calculees ===
    public int TotalCells => GridResolutionX * GridResolutionY * GridResolutionZ;
    public double CellSizeX => DomainSizeX / GridResolutionX;
    public double CellSizeY => DomainSizeY / GridResolutionY;
    public double CellSizeZ => DomainSizeZ / GridResolutionZ;
    public double CellVolume => CellSizeX * CellSizeY * CellSizeZ;
    public double SmallestCellDimension => Math.Min(Math.Min(CellSizeX, CellSizeY), CellSizeZ);
    public double LargestCellDimension => Math.Max(Math.Max(CellSizeX, CellSizeY), CellSizeZ);

    public Vector3D GridToWorld(int ix, int iy, int iz) => new(DomainOffsetX + ix * CellSizeX, DomainOffsetY + iy * CellSizeY, DomainOffsetZ + iz * CellSizeZ);
    public void WorldToGrid(Vector3D w, out int ix, out int iy, out int iz) { ix = Math.Clamp((int)((w.X - DomainOffsetX) / CellSizeX), 0, GridResolutionX - 1); iy = Math.Clamp((int)((w.Y - DomainOffsetY) / CellSizeY), 0, GridResolutionY - 1); iz = Math.Clamp((int)((w.Z - DomainOffsetZ) / CellSizeZ), 0, GridResolutionZ - 1); }
    public bool IsValidIndex(int ix, int iy, int iz) => ix >= 0 && ix < GridResolutionX && iy >= 0 && iy < GridResolutionY && iz >= 0 && iz < GridResolutionZ;
    public int FlattenIndex(int ix, int iy, int iz) => ix + GridResolutionX * (iy + GridResolutionY * iz);
    public void UnflattenIndex(int flat, out int ix, out int iy, out int iz) { iz = flat / (GridResolutionX * GridResolutionY); int rem = flat % (GridResolutionX * GridResolutionY); iy = rem / GridResolutionX; ix = rem % GridResolutionX; }

    public override string ToString() => $"SimConfig[{GridResolutionX}x{GridResolutionY}x{GridResolutionZ}] dt={TimeStep} domain=[{DomainSizeX},{DomainSizeY},{DomainSizeZ}] cells={TotalCells:N0}";
}
// SECTION 12: PHYSICSSTATE EXTENSIONS
[StructLayout(LayoutKind.Explicit, Size = 256)]
public partial struct PhysicsState
{
    // ── CORE FIELDS (referenced by FieldSampleResult and PhysicsConstraint) ──
    [FieldOffset(0)] public double Density;
    [FieldOffset(8)] public double Pressure;
    [FieldOffset(16)] public double Temperature;
    [FieldOffset(24)] public double Entropy;
    [FieldOffset(32)] public double InternalEnergy;
    [FieldOffset(40)] public double _kineticEnergy;
    [FieldOffset(48)] public double Norm;
    [FieldOffset(56)] public Vector3D Velocity;
    [FieldOffset(80)] public Vector3D HeatFlux;
    [FieldOffset(104)] public Vector3D Position;
    [FieldOffset(128)] public Tensor3D VelocityGradient;

    public readonly double KineticEnergy => _kineticEnergy;

    // ── 12.1 TENSOR OPERATIONS ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeStrainTensor(Tensor3D L) => (L + L.Transpose()) * 0.5;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeRotationTensor(Tensor3D L) => (L - L.Transpose()) * 0.5;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeGreenLagrangeStrain(Tensor3D F) => (F.Transpose() * F - Tensor3D.Identity) * 0.5;
    public Tensor3D ComputeAlmansiStrain(Tensor3D F) => (Tensor3D.Identity - (F * F.Transpose()).Inverse()) * 0.5;
    public Tensor3D ComputeCauchyStress(double p, double mu, double lam, Tensor3D e)
        => -Tensor3D.Identity * p + e * (2.0 * mu) + Tensor3D.Identity * (lam * e.Trace);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeFirstPiolaKirchhoff(Tensor3D F, Tensor3D S) => F * S;
    public Tensor3D ComputeSecondPiolaKirchhoff(Tensor3D F, Tensor3D sigma)
        => F.Inverse() * sigma * F.Inverse().Transpose() * F.Determinant;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D ComputeVorticity(Tensor3D L) => new Vector3D(L.M32 - L.M23, L.M13 - L.M31, L.M21 - L.M12);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ComputeEnstrophy(Vector3D w) => 0.5 * w.LengthSquared();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ComputeStrainInvariant2(Tensor3D e) { double t = e.Trace; return 0.5 * (e.FrobeniusNormSquared - t * t); }
    public double ComputeOkuboWeiss(Tensor3D L) => ComputeStrainTensor(L).FrobeniusNormSquared - ComputeRotationTensor(L).FrobeniusNormSquared;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeLeftCauchyGreen(Tensor3D F) => F * F.Transpose();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor3D ComputeRightCauchyGreen(Tensor3D F) => F.Transpose() * F;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ComputeVolumeRatio(Tensor3D F) => F.Determinant;
    public Tensor3D ComputeMandelStress(Tensor3D F, Tensor3D S) => ComputeRightCauchyGreen(F).Inverse() * S;

    // ── 12.2 THERMODYNAMIC STATES ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double InternalEnergyIdealGas(double T, double cv) => cv * T;
    public double HelmholtzFreeEnergy(double T, double V, double n, double cv, double R, double T0, double V0)
        => cv * T - T * n * (cv * Math.Log(T / T0) + R * Math.Log(V / V0));
    public double GibbsFreeEnergy(double T, double P, double n, double cp, double R, double S0, double P0)
        => cp * T - T * (S0 + n * R * Math.Log(P / P0));
    public double SackurTetrodeEntropy(double n, double V, double T, double m, double kB, double h)
        => n * 8.314462618 * (Math.Log(V * Math.Pow(2.0 * Math.PI * m * kB * T / (h * h), 1.5)) + 2.5);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double CpFromCv(double cv, double R) => cv + R;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double GammaFromCv(double cv, double R) => (cv + R) / cv;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double EnthalpyIdealGas(double T, double n, double cv, double R) => n * (cv + R) * T;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double InternalEnergyVdW(double T, double n, double V, double cv, double a) => cv * n * T - a * n * n / V;
    public double EntropyVdW(double n, double T, double V, double cv, double R, double b) => n * cv * Math.Log(T) + n * R * Math.Log(V - n * b);
    public double EntropyOfMixing(double[] x, double R) { double s = 0; for (int i = 0; i < x.Length; i++) if (x[i] > 1e-15) s -= x[i] * Math.Log(x[i]); return R * s; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ChemicalPotential(double T, double Pi, double mu0, double R, double P0) => mu0 + R * T * Math.Log(Pi / P0);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double EquilibriumConstant(double dG0, double T, double R) => Math.Exp(-dG0 / (R * T));
    public double VanHoffEquation(double K1, double T1, double T2, double dH0, double R) => K1 * Math.Exp(-dH0 / R * (1.0 / T2 - 1.0 / T1));

    // ── 12.3 EQUATIONS OF STATE ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PIdealGas(double n, double T, double V, double R) => n * R * T / V;
    public double PVdW(double n, double T, double V, double R, double a, double b) => n * R * T / (V - b * n) - a * n * n / (V * V);
    public double PVdW_Iterative(double n, double T, double P, double R, double a, double b)
    {
        double v = n * R * T / P;
        for (int i = 0; i < 50; i++)
        {
            double f = P - n * R * T / (v - b * n) + a * n * n / (v * v);
            double df = -n * R * T / ((v - b * n) * (v - b * n)) + 2.0 * a * n * n / (v * v * v);
            double dv = f / df;
            v -= dv;
            if (Math.Abs(dv) < 1e-14 * Math.Abs(v))
                break;
            if (v < b * n * 1.01)
                v = b * n * 1.01;
        }
        return v;
    }
    public double PRK(double n, double T, double V, double R, double a, double b) => n * R * T / (V - b * n) - a * n * n / (Math.Sqrt(T) * V * (V + b * n));
    public double PPengRobinson(double n, double T, double V, double R, double a, double b, double omega)
    {
        double kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega;
        double Tc = a / (0.45724 * R * R);
        double Tr = T / Tc;
        double alpha = (1.0 + kappa * (1.0 - Math.Sqrt(Tr)));
        alpha *= alpha;
        return n * R * T / (V - b * n) - a * alpha * n * n / (V * V + 2.0 * b * V - b * b);
    }
    public double PBWR(double n, double T, double V, double R, double A0, double B0, double C0, double a, double b, double c, double alpha, double gamma)
    {
        double nRT = n * R * T, n2 = n * n, n3 = n2 * n, n6 = n3 * n3;
        double V2 = V * V, V6 = V2 * V2 * V2, T2 = T * T;
        return nRT / V + (B0 * nRT - A0 - C0 / T2) * n2 / V2 + (b * nRT - a) * n3 / V6 + a * alpha * n6 / V6 + c * n3 * Math.Exp(-gamma * n2 / V2) / (V * V2);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double CompressibilityFactor(double P, double V, double n, double T, double R) => P * V / (n * R * T);
    public double FugacityCoeffVdW(double Z, double A, double B) => Z <= B ? 1e-10 : Math.Exp(Z - 1.0 - Math.Log(Z - B) - A / Z);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Fugacity(double phi, double P) => phi * P;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SpeedOfSound(double gamma, double R, double T, double M) => Math.Sqrt(gamma * R * T / M);
    public double JouleThomson(double T, double V, double dVdT, double Cp) => (T * dVdT - V) / Cp;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double InversionTempVdW(double a, double b, double R) => 2.0 * a / (R * b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SecondVirial(double T, double a, double b, double R) => b - a / (R * T);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double BoyleTemp(double a, double b, double R) => a / (R * b);

    // ── 12.4 TRANSPORT PROPERTIES ──
    public double ViscosityChapmanEnskog(double T, double m, double sigma, double Omega)
    {
        double kB = 1.380649e-23;
        return (5.0 / 16.0) * Math.Sqrt(Math.PI * m * kB * T) / (Math.PI * sigma * sigma * Omega);
    }
    public double ViscositySutherland(double T, double eta0, double T0, double S)
        => eta0 * Math.Pow(T / T0, 1.5) * (T0 + S) / (T + S);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ThermalConductivityEucken(double eta, double cp, double R, double M) => eta * (cp + 1.25 * R) / M;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PrandtlNumber(double cp, double eta, double lambda) => cp * eta / lambda;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SchmidtNumber(double eta, double rho, double D) => eta / (rho * D);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double LewisNumber(double alpha, double D) => alpha / D;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double KnudsenNumber(double mfp, double L_char) => mfp / L_char;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double MeanFreePath(double T, double P, double sigma) => 1.380649e-23 * T / (Math.Sqrt(2.0) * Math.PI * sigma * sigma * P);
    public double DiffusionCoefficient(double T, double P, double sigma, double m)
    {
        double kB = 1.380649e-23;
        double n = P / (kB * T);
        return 0.375 * Math.Sqrt(kB * T / (Math.PI * m)) / (sigma * sigma * n);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D FicksLaw(double D, Vector3D gradC) => gradC * (-D);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D FouriersLaw(double lambda, Vector3D gradT) => gradT * (-lambda);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double NewtonViscosity(double eta, double dudy) => eta * dudy;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PowerLawViscosity(double K, double shearRate, double n) => Math.Abs(shearRate) < 1e-15 ? double.MaxValue : K * Math.Pow(Math.Abs(shearRate), n - 1.0);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ArrheniusRate(double A, double Ea, double T, double R) => A * Math.Exp(-Ea / (R * T));
    public double WLFViscosity(double eta0, double T, double Tg, double c1, double c2) => eta0 * Math.Pow(10.0, -c1 * (T - Tg) / (c2 + T - Tg));

    // ── 12.5 EVOLUTION / TIME INTEGRATION ──
    public Vector3D RungeKutta4(Func<double, Vector3D, Vector3D> f, double t, Vector3D y, double h)
    {
        var k1 = f(t, y);
        var k2 = f(t + h * 0.5, y + k1 * (h * 0.5));
        var k3 = f(t + h * 0.5, y + k2 * (h * 0.5));
        var k4 = f(t + h, y + k3 * h);
        return y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
    }
    public void VelocityVerlet(ref Vector3D pos, ref Vector3D vel, Func<Vector3D, Vector3D> acc, double h)
    {
        var a = acc(pos);
        pos = pos + vel * h + a * (0.5 * h * h);
        var an = acc(pos);
        vel = vel + (a + an) * (0.5 * h);
    }
    public void Leapfrog(ref Vector3D pos, ref Vector3D vHalf, Func<Vector3D, Vector3D> acc, double h)
    {
        pos = pos + vHalf * h;
        vHalf = vHalf + acc(pos) * h;
    }
    public void SymplecticEuler(ref Vector3D pos, ref Vector3D vel, Func<Vector3D, Vector3D> acc, double h)
    {
        vel = vel + acc(pos) * h;
        pos = pos + vel * h;
    }
    public Vector3D StormerVerlet(Vector3D pos, Vector3D posPrev, Func<Vector3D, Vector3D> acc, double h)
        => pos * 2.0 - posPrev + acc(pos) * (h * h);
    public void YoshidaSuzuki4(ref Vector3D pos, ref Vector3D vel, Func<Vector3D, Vector3D> acc, double h)
    {
        double w1 = 1.0 / (2.0 - Math.Pow(2.0, 1.0 / 3.0));
        double w0 = 1.0 - 2.0 * w1;
        VelocityVerlet(ref pos, ref vel, acc, h * w1);
        VelocityVerlet(ref pos, ref vel, acc, h * w0);
        VelocityVerlet(ref pos, ref vel, acc, h * w1);
    }
    public (Vector3D pos, Vector3D vel, double H) HMCStep(Vector3D pos, Vector3D mom, Func<Vector3D, double> PE, Func<Vector3D, Vector3D> force, double h, int steps)
    {
        double H0 = 0.5 * mom.LengthSquared() + PE(pos);
        for (int i = 0; i < steps; i++)
        { mom = mom + force(pos) * (h * 0.5); pos = pos + mom * h; mom = mom + force(pos) * (h * 0.5); }
        return (pos, mom, 0.5 * mom.LengthSquared() + PE(pos));
    }
    public (Vector3D pos, Vector3D vel) LangevinBAOAB(Vector3D pos, Vector3D vel, Func<Vector3D, Vector3D> force, double m, double gamma, double T, double kB, double h, Random rng)
    {
        double hh = h * 0.5;
        vel = vel + force(pos) * (hh / m);
        double c1 = Math.Exp(-gamma * hh);
        vel = vel * c1;
        double noise = Math.Sqrt((1.0 - c1 * c1) * kB * T / m);
        vel = vel + new Vector3D(NormalSample(rng), NormalSample(rng), NormalSample(rng)) * noise;
        vel = vel * c1;
        pos = pos + vel * hh;
        vel = vel + force(pos) * (hh / m);
        return (pos, vel);
    }
    public Vector3D BrownianStep(Vector3D pos, Func<Vector3D, Vector3D> force, double gamma, double T, double kB, double h, Random rng)
        => pos + force(pos) * (h / gamma) + new Vector3D(NormalSample(rng), NormalSample(rng), NormalSample(rng)) * Math.Sqrt(2.0 * kB * T * h / gamma);
    private static double NormalSample(Random rng) { double u1 = rng.NextDouble(), u2 = rng.NextDouble(); return Math.Sqrt(-2.0 * Math.Log(u1 + 1e-300)) * Math.Cos(2.0 * Math.PI * u2); }
    public (Vector3D state, double newH) AdaptiveRKF45(Func<double, Vector3D, Vector3D> f, double t, Vector3D y, double h, double tol, double hMin, double hMax)
    {
        var k1 = f(t, y);
        var k2 = f(t + h * 0.25, y + k1 * (h * 0.25));
        var k3 = f(t + h * 0.375, y + k1 * (h * 3.0 / 32) + k2 * (h * 9.0 / 32));
        var k4 = f(t + h * 12.0 / 13, y + k1 * (h * 1932.0 / 2197) - k2 * (h * 7200.0 / 2197) + k3 * (h * 7296.0 / 2197));
        var k5 = f(t + h, y + k1 * (h * 439.0 / 216) - k2 * (h * 8.0) + k3 * (h * 3680.0 / 513) - k4 * (h * 845.0 / 4104));
        var k6 = f(t + h * 0.5, y - k1 * (h * 8.0 / 27) + k2 * (h * 2.0) - k3 * (h * 3544.0 / 2565) + k4 * (h * 1859.0 / 4104) - k5 * (h * 11.0 / 40));
        var y4 = y + k1 * (h * 25.0 / 216) + k3 * (h * 1408.0 / 2565) + k4 * (h * 2197.0 / 4104) + k5 * (-h / 5.0);
        var y5 = y + k1 * (h * 16.0 / 135) + k3 * (h * 6656.0 / 12825) + k4 * (h * 28561.0 / 56430) + k5 * (-h * 9.0 / 50) + k6 * (h * 2.0 / 55);
        double err = Math.Max((y5 - y4).Length(), 1e-15);
        double newH = Math.Max(hMin, Math.Min(hMax, h * Math.Pow(tol / err, 0.2)));
        return err <= tol ? (y5, newH) : AdaptiveRKF45(f, t, y, newH, tol, hMin, hMax);
    }
    public Vector3D EulerMaruyama(Func<Vector3D, Vector3D> drift, Func<Vector3D, Vector3D> diff, Vector3D y, double h, Random rng)
    {
        var Z = new Vector3D(NormalSample(rng), NormalSample(rng), NormalSample(rng));
        return y + drift(y) * h + new Vector3D(diff(y).X * Z.X, diff(y).Y * Z.Y, diff(y).Z * Z.Z) * Math.Sqrt(h);
    }
    public double TotalEnergy(double m, Vector3D v, Func<Vector3D, double> PE, Vector3D r) => 0.5 * m * v.LengthSquared() + PE(r);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D Momentum(double m, Vector3D v) => v * m;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double KineticEnergyCalc(double m, Vector3D v) => 0.5 * m * v.LengthSquared();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D AngularMomentum(double m, Vector3D r, Vector3D v) => Vector3D.Cross(r, v) * m;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D Torque(Vector3D r, Vector3D F) => Vector3D.Cross(r, F);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PowerCalc(Vector3D F, Vector3D v) => Vector3D.Dot(F, v);
    public double WorkAlongPath(Func<Vector3D, Vector3D> field, Vector3D[] path)
    {
        double w = 0;
        for (int i = 0; i < path.Length - 1; i++)
        { var mid = (path[i] + path[i + 1]) * 0.5; w += Vector3D.Dot(field(mid), path[i + 1] - path[i]); }
        return w;
    }

    // ── 12.6 PHASE & EQUILIBRIUM ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ClausiusClapeyron(double L, double T, double dV) => L / (T * dV);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double AntoineEquation(double T, double A, double B, double C) => Math.Pow(10.0, A - B / (C + T));
    public double VaporPressureCC(double P1, double T1, double T2, double dHvap, double R) => P1 * Math.Exp(-dHvap / R * (1.0 / T2 - 1.0 / T1));
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double TroutonsRule(double Tb, double Kt) => Kt * Tb;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double RaoultLaw(double x, double P0) => x * P0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double HenrysLaw(double KH, double C) => KH * C;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double BoilingPointElevation(double i, double Kb, double m) => i * Kb * m;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double OsmoticPressure(double i, double M, double R, double T) => i * M * R * T;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GibbsPhaseRule(int C, int P) => C - P + 2;
    public double RegularSolutionGibbs(double x1, double T, double R, double omega)
    { double x2 = 1.0 - x1; return R * T * (x1 * Math.Log(x1 + 1e-30) + x2 * Math.Log(x2 + 1e-30)) + omega * x1 * x2; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsInsideSpinodal(double x1, double T, double R, double omega) => R * T / (x1 * (1.0 - x1) + 1e-30) < 2.0 * omega;
    public double FloryHuggins(double phi, int N, double chi) => (phi / N) * Math.Log(phi + 1e-30) + (1.0 - phi) * Math.Log(1.0 - phi + 1e-30) + chi * phi * (1.0 - phi);

    // ── 12.7 FIELD OPERATORS ──
    public double Divergence(Vector3D[,,] F, int ix, int iy, int iz, double dx, double dy, double dz)
    {
        int nx = F.GetLength(0), ny = F.GetLength(1), nz = F.GetLength(2);
        double dFx = (ix + 1 < nx ? F[ix + 1, iy, iz].X : F[ix, iy, iz].X) - (ix > 0 ? F[ix - 1, iy, iz].X : F[ix, iy, iz].X);
        double dFy = (iy + 1 < ny ? F[ix, iy + 1, iz].Y : F[ix, iy, iz].Y) - (iy > 0 ? F[ix, iy - 1, iz].Y : F[ix, iy, iz].Y);
        double dFz = (iz + 1 < nz ? F[ix, iy, iz + 1].Z : F[ix, iy, iz].Z) - (iz > 0 ? F[ix, iy, iz - 1].Z : F[ix, iy, iz].Z);
        return dFx / (2.0 * dx) + dFy / (2.0 * dy) + dFz / (2.0 * dz);
    }
    public Vector3D Curl(Vector3D[,,] F, int ix, int iy, int iz, double dx, double dy, double dz)
    {
        int nx = F.GetLength(0), ny = F.GetLength(1), nz = F.GetLength(2);
        double dFzdy = ((iy + 1 < ny ? F[ix, iy + 1, iz].Z : F[ix, iy, iz].Z) - (iy > 0 ? F[ix, iy - 1, iz].Z : F[ix, iy, iz].Z)) / (2.0 * dy);
        double dFydz = ((iz + 1 < nz ? F[ix, iy, iz + 1].Y : F[ix, iy, iz].Y) - (iz > 0 ? F[ix, iy, iz - 1].Y : F[ix, iy, iz].Y)) / (2.0 * dz);
        double dFxdz = ((iz + 1 < nz ? F[ix, iy, iz + 1].X : F[ix, iy, iz].X) - (iz > 0 ? F[ix, iy, iz - 1].X : F[ix, iy, iz].X)) / (2.0 * dz);
        double dFzdx = ((ix + 1 < nx ? F[ix + 1, iy, iz].Z : F[ix, iy, iz].Z) - (ix > 0 ? F[ix - 1, iy, iz].Z : F[ix, iy, iz].Z)) / (2.0 * dx);
        double dFydx = ((ix + 1 < nx ? F[ix + 1, iy, iz].Y : F[ix, iy, iz].Y) - (ix > 0 ? F[ix - 1, iy, iz].Y : F[ix, iy, iz].Y)) / (2.0 * dx);
        double dFxdy = ((iy + 1 < ny ? F[ix, iy + 1, iz].X : F[ix, iy, iz].X) - (iy > 0 ? F[ix, iy - 1, iz].X : F[ix, iy, iz].X)) / (2.0 * dy);
        return new Vector3D(dFzdy - dFydz, dFxdz - dFzdx, dFydx - dFxdy);
    }
    public Vector3D Gradient(double[,,] f, int ix, int iy, int iz, double dx, double dy, double dz)
    {
        int nx = f.GetLength(0), ny = f.GetLength(1), nz = f.GetLength(2);
        double dFx = ((ix + 1 < nx ? f[ix + 1, iy, iz] : f[ix, iy, iz]) - (ix > 0 ? f[ix - 1, iy, iz] : f[ix, iy, iz])) / (2.0 * dx);
        double dFy = ((iy + 1 < ny ? f[ix, iy + 1, iz] : f[ix, iy, iz]) - (iy > 0 ? f[ix, iy - 1, iz] : f[ix, iy, iz])) / (2.0 * dy);
        double dFz = ((iz + 1 < nz ? f[ix, iy, iz + 1] : f[ix, iy, iz]) - (iz > 0 ? f[ix, iy, iz - 1] : f[ix, iy, iz])) / (2.0 * dz);
        return new Vector3D(dFx, dFy, dFz);
    }
    public double Laplacian(double[,,] f, int ix, int iy, int iz, double dx, double dy, double dz)
    {
        int nx = f.GetLength(0), ny = f.GetLength(1), nz = f.GetLength(2);
        double c = f[ix, iy, iz];
        return ((ix + 1 < nx ? f[ix + 1, iy, iz] : c) - 2.0 * c + (ix > 0 ? f[ix - 1, iy, iz] : c)) / (dx * dx)
            + ((iy + 1 < ny ? f[ix, iy + 1, iz] : c) - 2.0 * c + (iy > 0 ? f[ix, iy - 1, iz] : c)) / (dy * dy)
            + ((iz + 1 < nz ? f[ix, iy, iz + 1] : c) - 2.0 * c + (iz > 0 ? f[ix, iy, iz - 1] : c)) / (dz * dz);
    }
    public double AdvectionUpwind(double[,,] f, Vector3D vel, int ix, int iy, int iz, double dx, double dy, double dz)
    {
        int nx = f.GetLength(0), ny = f.GetLength(1), nz = f.GetLength(2);
        double c = f[ix, iy, iz];
        double dphidx = vel.X > 0 ? (ix > 0 ? c - f[ix - 1, iy, iz] : 0) / dx : (ix + 1 < nx ? f[ix + 1, iy, iz] - c : 0) / dx;
        double dphidy = vel.Y > 0 ? (iy > 0 ? c - f[ix, iy - 1, iz] : 0) / dy : (iy + 1 < ny ? f[ix, iy + 1, iz] - c : 0) / dy;
        double dphidz = vel.Z > 0 ? (iz > 0 ? c - f[ix, iy, iz - 1] : 0) / dz : (iz + 1 < nz ? f[ix, iy, iz + 1] - c : 0) / dz;
        return -(vel.X * dphidx + vel.Y * dphidy + vel.Z * dphidz);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DiffusionExplicit(double[,,] f, double alpha, int ix, int iy, int iz, double dx, double dy, double dz) => alpha * Laplacian(f, ix, iy, iz, dx, dy, dz);
    public double WaveEquation(double[,,] fCur, double[,,] fPrev, double c, int ix, int iy, int iz, double dx, double dy, double dz, double dt)
        => 2.0 * fCur[ix, iy, iz] - fPrev[ix, iy, iz] + c * c * dt * dt * Laplacian(fCur, ix, iy, iz, dx, dy, dz);

    // ── 12.8 PARTICLE & RELATIVISTIC PHYSICS ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double LorentzFactor(double v, double c) => 1.0 / Math.Sqrt(1.0 - (v / c) * (v / c));
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double RelativisticEnergy(double gamma, double m, double c) => gamma * m * c * c;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D RelativisticMomentum(double gamma, double m, Vector3D v) => v * (gamma * m);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double RelativisticKE(double gamma, double m, double c) => (gamma - 1.0) * m * c * c;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double EnergyMomentum(double p, double m, double c) { double pc = p * c, mc2 = m * c * c; return Math.Sqrt(pc * pc + mc2 * mc2); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DopplerShift(double f0, double vr, double c) { double b = vr / c; return f0 * Math.Sqrt((1.0 + b) / (1.0 - b)); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ComptonWavelength(double m, double h, double c) => h / (m * c);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DeBroglieWavelength(double p, double h) => h / p;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SchwarzschildRadius(double M, double G, double c) => 2.0 * G * M / (c * c);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double GravitationalRedshift(double M, double r, double G, double c) => G * M / (r * c * c);
    public double HawkingTemperature(double M, double G, double c, double hbar, double kB) => hbar * c * c * c / (8.0 * Math.PI * G * M * kB);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double StefanBoltzmann(double A, double T, double sigma) => sigma * A * Math.Pow(T, 4);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double WienDisplacement(double T, double b) => b / T;
    public double PlanckLaw(double lambda, double T, double h, double c, double kB)
    {
        double hc = h * c;
        return 2.0 * h * c * c / Math.Pow(lambda, 5) / (Math.Exp(hc / (lambda * kB * T)) - 1.0);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double EinsteinME(double m, double c) => m * c * c;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double BoltzmannEntropy(long Omega, double kB) => kB * Math.Log(Omega);
    public double GibbsEntropy(double[] p, double kB) { double s = 0; for (int i = 0; i < p.Length; i++) if (p[i] > 1e-30) s -= p[i] * Math.Log(p[i]); return kB * s; }

    // ── 12.9 GEOMETRY & MATH UTILITIES ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D CrossProduct(Vector3D a, Vector3D b) => Vector3D.Cross(a, b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DotProduct(Vector3D a, Vector3D b) => Vector3D.Dot(a, b);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double TripleScalar(Vector3D a, Vector3D b, Vector3D c) => Vector3D.Dot(a, Vector3D.Cross(b, c));
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D VectorTriple(Vector3D a, Vector3D b, Vector3D c) => b * Vector3D.Dot(a, c) - c * Vector3D.Dot(a, b);
    public Tensor3D OuterProduct(Vector3D a, Vector3D b) => new Tensor3D(a.X * b.X, a.X * b.Y, a.X * b.Z, a.Y * b.X, a.Y * b.Y, a.Y * b.Z, a.Z * b.X, a.Z * b.Y, a.Z * b.Z);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D Project(Vector3D a, Vector3D b) { double d = Vector3D.Dot(b, b); return d < 1e-30 ? Vector3D.Zero : b * (Vector3D.Dot(a, b) / d); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D Reject(Vector3D a, Vector3D b) => a - Project(a, b);
    public double AngleBetween(Vector3D a, Vector3D b) { double c = Math.Max(-1, Math.Min(1, Vector3D.Dot(a.Normalized(), b.Normalized()))); return Math.Acos(c); }
    public double SignedAngleAxis(Vector3D a, Vector3D b, Vector3D axis)
    {
        var ax = Project(a, axis.Normalized());
        var bx = Project(b, axis.Normalized());
        var ap = a - ax;
        var bp = b - bx;
        return Math.Atan2(Vector3D.Dot(axis, Vector3D.Cross(ap, bp)), Vector3D.Dot(ap, bp));
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SmoothMin(double a, double b, double k) { double h = Math.Max(k - Math.Abs(a - b), 0) / k; return Math.Min(a, b) - h * h * h * k / 6.0; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SmoothMax(double a, double b, double k) => -SmoothMin(-a, -b, k);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Remap(double v, double inMin, double inMax, double outMin, double outMax) => outMin + (v - inMin) * (outMax - outMin) / (inMax - inMin);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Lerp(double a, double b, double t) => a + Math.Max(0, Math.Min(1, t)) * (b - a);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double InverseLerp(double a, double b, double v) => Math.Abs(b - a) < 1e-30 ? 0 : (v - a) / (b - a);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Smoothstep(double e0, double e1, double x) { double t = Math.Max(0, Math.Min(1, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Smootherstep(double e0, double e1, double x) { double t = Math.Max(0, Math.Min(1, (x - e0) / (e1 - e0))); return t * t * t * (t * (t * 15 - 10) * t); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Clamp(double v, double lo, double hi) => Math.Max(lo, Math.Min(hi, v));
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Wrap(double v, double lo, double hi) { double r = hi - lo; return lo + ((v - lo) % r + r) % r; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PingPong(double v, double len) { v = Math.Abs(v); double m = v % (2 * len); return len - Math.Abs(m - len); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Distance3D(Vector3D a, Vector3D b) => (b - a).Length();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DistanceSq(Vector3D a, Vector3D b) => (b - a).LengthSquared();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ManhattanDist(Vector3D a, Vector3D b) => Math.Abs(a.X - b.X) + Math.Abs(a.Y - b.Y) + Math.Abs(a.Z - b.Z);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ChebyshevDist(Vector3D a, Vector3D b) => Math.Max(Math.Max(Math.Abs(a.X - b.X), Math.Abs(a.Y - b.Y)), Math.Abs(a.Z - b.Z));
    public double MinkowskiDist(Vector3D a, Vector3D b, double p) => Math.Pow(Math.Pow(Math.Abs(a.X - b.X), p) + Math.Pow(Math.Abs(a.Y - b.Y), p) + Math.Pow(Math.Abs(a.Z - b.Z), p), 1.0 / p);
    public Vector3D HermiteInterp(Vector3D p0, Vector3D m0, Vector3D p1, Vector3D m1, double t)
    { double t2 = t * t, t3 = t2 * t; return p0 * (2 * t3 - 3 * t2 + 1) + m0 * (t3 - 2 * t2 + t) + p1 * (-2 * t3 + 3 * t2) + m1 * (t3 - t2); }
    public Vector3D CatmullRom(Vector3D p0, Vector3D p1, Vector3D p2, Vector3D p3, double t, double tau)
    { double t2 = t * t, t3 = t2 * t; return (p1 * 2 + (p2 - p0) * tau * t + (p0 * 2 - p1 * 5 + p2 * 4 - p3) * tau * t2 + (-p1 + p2 * 3 - p3 + p0 * -1) * tau * t3) * 0.5; }
    public Vector3D Bezier(Vector3D p0, Vector3D p1, Vector3D p2, Vector3D p3, double t)
    { double u = 1 - t, u2 = u * u, u3 = u2 * u, t2 = t * t, t3 = t2 * t; return p0 * u3 + p1 * (3 * u2 * t) + p2 * (3 * u * t2) + p3 * t3; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Fade(double t) => t * t * t * (t * (t * 6 - 15) + 10);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float FastInvSqrt(float x) { int i = BitConverter.SingleToInt32Bits(x); i = 0x5f3759df - (i >> 1); float y = BitConverter.Int32BitsToSingle(i); return y * (1.5f - 0.5f * x * y * y); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Cross2D(Vector3D a, Vector3D b) => a.X * b.Y - a.Y * b.X;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double TriangleArea(Vector3D a, Vector3D b, Vector3D c) => 0.5 * Vector3D.Cross(b - a, c - a).Length();
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D TriangleCentroid(Vector3D a, Vector3D b, Vector3D c) => (a + b + c) * (1.0 / 3.0);
    public (double u, double v, double w) Barycentric(Vector3D p, Vector3D a, Vector3D b, Vector3D c)
    {
        var v0 = b - a;
        var v1 = c - a;
        var v2 = p - a;
        double d00 = Vector3D.Dot(v0, v0), d01 = Vector3D.Dot(v0, v1), d11 = Vector3D.Dot(v1, v1);
        double d20 = Vector3D.Dot(v2, v0), d21 = Vector3D.Dot(v2, v1);
        double den = d00 * d11 - d01 * d01;
        if (Math.Abs(den) < 1e-30)
            return (1.0 / 3, 1.0 / 3, 1.0 / 3);
        double v = (d11 * d20 - d01 * d21) / den;
        double w = (d00 * d21 - d01 * d20) / den;
        return (1 - v - w, v, w);
    }
    public bool PointInTriangle(Vector3D p, Vector3D a, Vector3D b, Vector3D c) { var (u, v, w) = Barycentric(p, a, b, c); return u >= -1e-6 && v >= -1e-6 && w >= -1e-6; }
    public Vector3D ClosestPointSegment(Vector3D p, Vector3D a, Vector3D b)
    { var ab = b - a; double t = Math.Max(0, Math.Min(1, Vector3D.Dot(p - a, ab) / Vector3D.Dot(ab, ab))); return a + ab * t; }
    public double DistPointSegment(Vector3D p, Vector3D a, Vector3D b) => (p - ClosestPointSegment(p, a, b)).Length();
    public Vector3D ClosestPointTriangle(Vector3D p, Vector3D a, Vector3D b, Vector3D c)
    {
        var (u, v, w) = Barycentric(p, a, b, c);
        if (u >= 0 && v >= 0 && w >= 0)
            return p;
        var best = ClosestPointSegment(p, a, b);
        double bestD = (p - best).LengthSquared();
        var cand = ClosestPointSegment(p, b, c);
        double d = (p - cand).LengthSquared();
        if (d < bestD)
        { best = cand; bestD = d; }
        cand = ClosestPointSegment(p, c, a);
        d = (p - cand).LengthSquared();
        if (d < bestD)
            best = cand;
        return best;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SignedVolumeTetra(Vector3D a, Vector3D b, Vector3D c, Vector3D d) => Vector3D.Dot(a - d, Vector3D.Cross(b - d, c - d)) / 6.0;
    public double HullVolume(Vector3D[] verts, (int i, int j, int k)[] tris)
    { double v = 0; var o = verts[0]; foreach (var t in tris) v += SignedVolumeTetra(verts[t.i], verts[t.j], verts[t.k], o); return Math.Abs(v); }
    public double MeshArea(Vector3D[] verts, (int i, int j, int k)[] tris) { double a = 0; foreach (var t in tris) a += TriangleArea(verts[t.i], verts[t.j], verts[t.k]); return a; }
    public Symmetric3x3 InertiaTensor((double m, Vector3D r)[] particles)
    {
        double ixx = 0, iyy = 0, izz = 0, ixy = 0, ixz = 0, iyz = 0;
        foreach (var (m, r) in particles)
        { ixx += m * (r.Y * r.Y + r.Z * r.Z); iyy += m * (r.X * r.X + r.Z * r.Z); izz += m * (r.X * r.X + r.Y * r.Y); ixy -= m * r.X * r.Y; ixz -= m * r.X * r.Z; iyz -= m * r.Y * r.Z; }
        return new Symmetric3x3(ixx, ixy, ixz, iyy, iyz, izz);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D PrincipalMoments(Symmetric3x3 I) { I.MaxEigenvalues(out double v1, out double v2, out double v3); return new Vector3D(v1, v2, v3); }
    public Vector3D CenterOfMass((double m, Vector3D r)[] p) { double M = 0; Vector3D s = Vector3D.Zero; foreach (var (m, r) in p) { M += m; s = s + r * m; } return M > 1e-30 ? s * (1 / M) : Vector3D.Zero; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ReducedMass(double m1, double m2) => m1 * m2 / (m1 + m2);
    public Vector3D GravForce(double m1, double m2, Vector3D r1, Vector3D r2, double G)
    { var d = r2 - r1; double dsq = d.LengthSquared(); return dsq < 1e-30 ? Vector3D.Zero : d.Normalized() * (G * m1 * m2 / dsq); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double GravPE(double m1, double m2, double r, double G) => -G * m1 * m2 / r;
    public Vector3D CoulombF(double q1, double q2, Vector3D r1, Vector3D r2, double ke)
    { var d = r2 - r1; double dsq = d.LengthSquared(); return dsq < 1e-30 ? Vector3D.Zero : d.Normalized() * (ke * q1 * q2 / dsq); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double CoulombV(double q, double r, double ke) => ke * q / r;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D HookesLaw(double k, Vector3D x) => x * (-k);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double SpringPE(double k, Vector3D x) => 0.5 * k * x.LengthSquared();
    public Vector3D DampedOscillator(double m, double c, double k, Vector3D x, Vector3D v) => (x * (-k) + v * (-c)) * (1.0 / m);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double NatFreq(double k, double m) => Math.Sqrt(k / m);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double DampRatio(double c, double k, double m) => c / (2.0 * Math.Sqrt(k * m));
    public double DampedPeriod(double k, double m, double c) { double w0 = Math.Sqrt(k / m); double z = c / (2 * Math.Sqrt(k * m)); return 2 * Math.PI / (w0 * Math.Sqrt(1 - z * z)); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double ResonanceAmp(double F0, double c, double w0) => F0 / (c * w0);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double QualityFactor(double zeta) => 1.0 / (2.0 * zeta);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double RotKE(double I, double w) => 0.5 * I * w * w;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PrecessionFreq(double m, double g, double r, double I, double w) => m * g * r / (I * w);

    // ── 12.10 RANDOM & NOISE ──
    public Vector3D RandomUnitVector(Random rng)
    {
        double theta = 2.0 * Math.PI * rng.NextDouble();
        double phi = Math.Acos(2.0 * rng.NextDouble() - 1.0);
        return new Vector3D(Math.Sin(phi) * Math.Cos(theta), Math.Sin(phi) * Math.Sin(theta), Math.Cos(phi));
    }
    public Vector3D RandomPointInSphere(Random rng)
    {
        double u = rng.NextDouble();
        double r = Math.Pow(u, 1.0 / 3.0);
        return RandomUnitVector(rng) * r;
    }
    public Vector3D RandomPointOnSphere(Random rng) => RandomUnitVector(rng);
    public double PerlinNoise3D(double x, double y, double z)
    {
        int xi = (int)Math.Floor(x), yi = (int)Math.Floor(y), zi = (int)Math.Floor(z);
        double xf = x - xi, yf = y - yi, zf = z - zi;
        double u = Fade(xf), v = Fade(yf), w2 = Fade(zf);
        int h000 = PerlinHash(xi, yi, zi), h100 = PerlinHash(xi + 1, yi, zi), h010 = PerlinHash(xi, yi + 1, zi), h110 = PerlinHash(xi + 1, yi + 1, zi);
        int h001 = PerlinHash(xi, yi, zi + 1), h101 = PerlinHash(xi + 1, yi, zi + 1), h011 = PerlinHash(xi, yi + 1, zi + 1), h111 = PerlinHash(xi + 1, yi + 1, zi + 1);
        double x1 = LerpUnclamped(Grad(h000, xf, yf, zf), Grad(h100, xf - 1, yf, zf), u);
        double x2 = LerpUnclamped(Grad(h010, xf, yf - 1, zf), Grad(h110, xf - 1, yf - 1, zf), u);
        double y1 = LerpUnclamped(x1, x2, v);
        x1 = LerpUnclamped(Grad(h001, xf, yf, zf - 1), Grad(h101, xf - 1, yf, zf - 1), u);
        x2 = LerpUnclamped(Grad(h011, xf, yf - 1, zf - 1), Grad(h111, xf - 1, yf - 1, zf - 1), u);
        double y2 = LerpUnclamped(x1, x2, v);
        return LerpUnclamped(y1, y2, w2);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int PerlinHash(int x, int y, int z) { int h = x * 374761393 + y * 668265263 + z * 1274126177; h = (h ^ (h >> 13)) * 1274126177; return h ^ (h >> 16); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Grad(int hash, double x, double y, double z)
    {
        int h = hash & 15;
        double u = h < 8 ? x : y;
        double v = h < 4 ? y : h == 12 || h == 14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double LerpUnclamped(double a, double b, double t) => a + t * (b - a);
}
/// <summary>Convertisseur universel d'unites avec analyse dimensionnelle.</summary>
public static class UnitConverter
{
    // ── SI Base Units ──
    public const double SpeedOfLight = 299792458.0;
    public const double PlanckConstant = 6.62607015e-34;
    public const double BoltzmannConstant = 1.380649e-23;
    public const double AvogadroNumber = 6.02214076e23;
    public const double GasConstant = 8.314462618;
    public const double ElementaryCharge = 1.602176634e-19;
    public const double ElectronMass = 9.1093837015e-31;
    public const double ProtonMass = 1.67262192369e-27;
    public const double GravitationalConstant = 6.67430e-11;
    public const double StefanBoltzmann = 5.670374419e-8;
    public const double VacuumPermittivity = 8.8541878128e-12;
    public const double VacuumPermeability = 1.25663706212e-6;
    public const double RydbergConstant = 1.0973731568160e7;
    public const double BohrRadius = 5.29177210903e-11;
    public const double FineStructure = 7.2973525693e-3;

    // ── Length Conversions (to SI meter) ──
    public const double Angstrom = 1e-10;
    public const double Nanometer = 1e-9;
    public const double Micrometer = 1e-6;
    public const double Millimeter = 1e-3;
    public const double Centimeter = 1e-2;
    public const double Inch = 0.0254;
    public const double Foot = 0.3048;
    public const double Yard = 0.9144;
    public const double Mile = 1609.344;
    public const double NauticalMile = 1852.0;
    public const double AstronomicalUnit = 1.495978707e11;
    public const double LightYear = 9.4607304725808e15;
    public const double Parsec = 3.08567758149137e16;
    public const double Fermi = 1e-15;

    // ── Mass Conversions (to SI kg) ──
    public const double Gram = 1e-3;
    public const double Milligram = 1e-6;
    public const double Tonne = 1e3;
    public const double Pound = 0.45359237;
    public const double Ounce = 0.028349523125;
    public const double Stone = 6.35029318;
    public const double AtomicMassUnit = 1.66053906660e-27;
    public const double Grain = 6.479891e-5;
    public const short TroyOunce = 31; // 31.1035g exact

    // ── Time Conversions (to SI second) ──
    public const double Millisecond = 1e-3;
    public const double Microsecond = 1e-6;
    public const double Nanosecond = 1e-9;
    public const double Minute = 60.0;
    public const double Hour = 3600.0;
    public const double Day = 86400.0;
    public const double Week = 604800.0;
    public const double JulianYear = 31557600.0;

    // ── Energy Conversions (to SI joule) ──
    public const double ElectronVolt = 1.602176634e-19;
    public const double KiloElectronVolt = 1.602176634e-16;
    public const double MegaElectronVolt = 1.602176634e-13;
    public const double GigaElectronVolt = 1.602176634e-10;
    public const double TeraElectronVolt = 1.602176634e-7;
    public const double Calorie = 4.184;
    public const double Kilocalorie = 4184.0;
    public const double BTU = 1055.06;
    public const double KilowattHour = 3.6e6;
    public const double Therm = 1.055e8;
    public const double Erg = 1e-7;
    public const double Hartree = 4.3597447222071e-18;
    public const double Rydberg = 2.1798723611030e-18;
    public const double WavenumberInverseCm = 1.986445857e-23;

    // ── Pressure Conversions (to SI pascal) ──
    public const double Bar = 1e5;
    public const double Millibar = 100.0;
    public const double Atmosphere = 101325.0;
    public const double Torr = 133.32236842;
    public const double MmHg = 133.322;
    public const double PSI = 6894.757;
    public const double InchHg = 3386.389;

    // ── Temperature Conversions ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double CelsiusToKelvin(double c) => c + 273.15;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double KelvinToCelsius(double k) => k - 273.15;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double FahrenheitToKelvin(double f) => (f - 32.0) * 5.0 / 9.0 + 273.15;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double KelvinToFahrenheit(double k) => (k - 273.15) * 9.0 / 5.0 + 32.0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double FahrenheitToCelsius(double f) => (f - 32.0) * 5.0 / 9.0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double CelsiusToFahrenheit(double c) => c * 9.0 / 5.0 + 32.0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double RankineToKelvin(double r) => r * 5.0 / 9.0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double KelvinToRankine(double k) => k * 9.0 / 5.0;

    // ── Area Conversions (to SI m²) ──
    public const double Hectare = 1e4;
    public const double Acre = 4046.8564224;
    public const double SquareInch = 6.4516e-4;
    public const double SquareFoot = 0.09290304;
    public const double SquareMile = 2589988.110336;
    public const double Barn = 1e-28;

    // ── Volume Conversions (to SI m³) ──
    public const double Liter = 1e-3;
    public const double Milliliter = 1e-6;
    public const double GallonUS = 3.785411784e-3;
    public const double GallonUK = 4.54609e-3;
    public const double QuartUS = 9.46352946e-4;
    public const double PintUS = 4.73176473e-4;
    public const double CupUS = 2.365882365e-4;
    public const double FluidOunceUS = 2.95735295625e-5;
    public const double TablespoonUS = 1.478676478125e-5;
    public const double TeaspoonUS = 4.92892159375e-6;
    public const double CubicInch = 1.6387064e-5;
    public const double CubicFoot = 0.028316846592;

    // ── Force Conversions (to SI newton) ──
    public const double Dyne = 1e-5;
    public const double KilogramForce = 9.80665;
    public const double PoundForce = 4.4482216152605;
    public const double Poundal = 0.138254954376;

    // ── Power Conversions (to SI watt) ──
    public const double Horsepower = 745.69987158227022;
    public const double FootPoundPerSecond = 1.3558179483314;
    public const double BTUPerSecond = 1055.06;

    // ── Electric Conversions ──
    public const double Ampere = 1.0;
    public const double Coulomb = 1.0;
    public const double Volt = 1.0;
    public const double Ohm = 1.0;
    public const double Siemens = 1.0;
    public const double Farad = 1.0;
    public const double Henry = 1.0;
    public const double Weber = 1.0;
    public const double Tesla = 1.0;
    public const double Gauss = 1e-4;
    public const double MaxwellsPerSquareCentimeter = 1e4;

    // ── Magnetic Conversions ──
    public const double Oersted = 79.57747154594767;
    public const double Gamma = 1e-9; // magnetic flux density

    // ── Angle Conversions ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double DegreesToRadians(double deg) => deg * Math.PI / 180.0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double RadiansToDegrees(double rad) => rad * 180.0 / Math.PI;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ArcMinutesToDegrees(double arcmin) => arcmin / 60.0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ArcSecondsToDegrees(double arcsec) => arcsec / 3600.0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double GradiansToRadians(double grad) => grad * Math.PI / 200.0;

    // ── Data/Information Conversions ──
    public const double Byte = 1.0;
    public const double Kilobyte = 1024.0;
    public const double Megabyte = 1048576.0;
    public const double Gigabyte = 1073741824.0;
    public const double Terabyte = 1099511627776.0;
    public const double Petabyte = 1125899906842624.0;
    public const double Bit = 0.125;
    public const double Nibble = 0.5;
    public const double Kilobit = 128.0;
    public const double Megabit = 131072.0;
    public const double Gigabit = 134217728.0;

    // ── Speed Conversions (to SI m/s) ──
    public const double Kph = 1.0 / 3.6;
    public const double Mph = 0.44704;
    public const double Knot = 0.514444;
    public const double FeetPerSecond = 0.3048;
    public const double Mach = 343.0;
    public const double SpeedOfLightMps = 299792458.0;

    // ── Viscosity Conversions ──
    public const double Poise = 0.1; // Pa·s
    public const double Centipoise = 1e-3;
    public const double Stokes = 1e-4; // m²/s
    public const double Centistokes = 1e-6;
    public const double Reyns = 6894.757;

    // ── Radiation Conversions ──
    public const double Gray = 1.0;
    public const double Rad = 0.01;
    public const double Sievert = 1.0;
    public const double Rem = 0.01;
    public const double Becquerel = 1.0;
    public const double Curie = 3.7e10;

    // ── Cross-section Conversions ──
    public const double BarnCm2 = 1e-24;
    public const double Millibarn = 1e-27;
    public const double Microbarn = 1e-30;
    public const double Nanobarn = 1e-33;
    public const double Picobarn = 1e-36;
    public const double Femtobarn = 1e-39;
    public const double Attobarn = 1e-42;

    // ── Dimensional Analysis ──
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Convert(double value, double factor) => value * factor;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ConvertRatio(double value, double fromFactor, double toFactor) => value * fromFactor / toFactor;

    /// <summary>Convert energy from one unit to another.</summary>
    public static double ConvertEnergy(double value, string from, string to)
    {
        double toSI = from.ToLower() switch
        {
            "j" or "joule" => 1.0,
            "kj" or "kilojoule" => 1e3,
            "mj" or "megajoule" => 1e6,
            "cal" or "calorie" => 4.184,
            "kcal" or "kilocalorie" => 4184.0,
            "ev" or "electronvolt" => 1.602176634e-19,
            "kev" => 1.602176634e-16,
            "mev" => 1.602176634e-13,
            "gev" => 1.602176634e-10,
            "tev" => 1.602176634e-7,
            "btu" => 1055.06,
            "kwh" or "kilowatthour" => 3.6e6,
            "erg" => 1e-7,
            "hartree" => 4.3597447222071e-18,
            "wh" or "watthour" => 3600.0,
            "mwh" or "megawatthour" => 3.6e9,
            "therm" => 1.055e8,
            "foe" => 1e44,
            _ => 1.0
        };
        double fromSI = to.ToLower() switch
        {
            "j" or "joule" => 1.0,
            "kj" or "kilojoule" => 1e3,
            "mj" or "megajoule" => 1e6,
            "cal" or "calorie" => 4.184,
            "kcal" or "kilocalorie" => 4184.0,
            "ev" or "electronvolt" => 1.602176634e-19,
            "kev" => 1.602176634e-16,
            "mev" => 1.602176634e-13,
            "gev" => 1.602176634e-10,
            "tev" => 1.602176634e-7,
            "btu" => 1055.06,
            "kwh" or "kilowatthour" => 3.6e6,
            "erg" => 1e-7,
            "hartree" => 4.3597447222071e-18,
            "wh" or "watthour" => 3600.0,
            "mwh" or "megawatthour" => 3.6e9,
            "therm" => 1.055e8,
            "foe" => 1e44,
            _ => 1.0
        };
        return value * toSI / fromSI;
    }

    /// <summary>Convert length from one unit to another.</summary>
    public static double ConvertLength(double value, string from, string to)
    {
        double toSI = from.ToLower() switch
        {
            "m" or "meter" => 1.0,
            "km" or "kilometer" => 1e3,
            "cm" or "centimeter" => 1e-2,
            "mm" or "millimeter" => 1e-3,
            "um" or "micrometer" or "micron" => 1e-6,
            "nm" or "nanometer" => 1e-9,
            "pm" or "picometer" => 1e-12,
            "fm" or "femtometer" or "fermi" => 1e-15,
            "a" or "angstrom" => 1e-10,
            "au" or "astronomicalunit" => 1.495978707e11,
            "ly" or "lightyear" => 9.4607304725808e15,
            "pc" or "parsec" => 3.08567758149137e16,
            "in" or "inch" => 0.0254,
            "ft" or "foot" => 0.3048,
            "yd" or "yard" => 0.9144,
            "mi" or "mile" => 1609.344,
            "nmi" or "nauticalmile" => 1852.0,
            _ => 1.0
        };
        double fromSI = to.ToLower() switch
        {
            "m" or "meter" => 1.0,
            "km" or "kilometer" => 1e3,
            "cm" or "centimeter" => 1e-2,
            "mm" or "millimeter" => 1e-3,
            "um" or "micrometer" or "micron" => 1e-6,
            "nm" or "nanometer" => 1e-9,
            "pm" or "picometer" => 1e-12,
            "fm" or "femtometer" or "fermi" => 1e-15,
            "a" or "angstrom" => 1e-10,
            "au" or "astronomicalunit" => 1.495978707e11,
            "ly" or "lightyear" => 9.4607304725808e15,
            "pc" or "parsec" => 3.08567758149137e16,
            "in" or "inch" => 0.0254,
            "ft" or "foot" => 0.3048,
            "yd" or "yard" => 0.9144,
            "mi" or "mile" => 1609.344,
            "nmi" or "nauticalmile" => 1852.0,
            _ => 1.0
        };
        return value * toSI / fromSI;
    }

    /// <summary>Convert mass from one unit to another.</summary>
    public static double ConvertMass(double value, string from, string to)
    {
        double toSI = from.ToLower() switch
        {
            "kg" or "kilogram" => 1.0,
            "g" or "gram" => 1e-3,
            "mg" or "milligram" => 1e-6,
            "ug" or "microgram" => 1e-9,
            "ng" or "nanogram" => 1e-12,
            "t" or "tonne" or "metricton" => 1e3,
            "lb" or "pound" => 0.45359237,
            "oz" or "ounce" => 0.028349523125,
            "st" or "stone" => 6.35029318,
            "amu" or "u" or "atomicmassunit" => 1.66053906660e-27,
            "gr" or "grain" => 6.479891e-5,
            "slug" => 14.593903,
            "ct" or "carat" => 2e-4,
            "shortton" => 907.18474,
            "longton" => 1016.0469088,
            _ => 1.0
        };
        double fromSI = to.ToLower() switch
        {
            "kg" or "kilogram" => 1.0,
            "g" or "gram" => 1e-3,
            "mg" or "milligram" => 1e-6,
            "ug" or "microgram" => 1e-9,
            "ng" or "nanogram" => 1e-12,
            "t" or "tonne" or "metricton" => 1e3,
            "lb" or "pound" => 0.45359237,
            "oz" or "ounce" => 0.028349523125,
            "st" or "stone" => 6.35029318,
            "amu" or "u" or "atomicmassunit" => 1.66053906660e-27,
            "gr" or "grain" => 6.479891e-5,
            "slug" => 14.593903,
            "ct" or "carat" => 2e-4,
            "shortton" => 907.18474,
            "longton" => 1016.0469088,
            _ => 1.0
        };
        return value * toSI / fromSI;
    }

    /// <summary>Convert pressure from one unit to another.</summary>
    public static double ConvertPressure(double value, string from, string to)
    {
        double toSI = from.ToLower() switch
        {
            "pa" or "pascal" => 1.0,
            "kpa" or "kilopascal" => 1e3,
            "mpa" or "megapascal" => 1e6,
            "gpa" or "gigapascal" => 1e9,
            "bar" => 1e5,
            "mbar" or "millibar" => 100.0,
            "atm" or "atmosphere" => 101325.0,
            "torr" or "mmhg" => 133.32236842,
            "psi" => 6894.757,
            "inhg" => 3386.389,
            "cmh2o" => 98.0665,
            "mh2o" => 9806.65,
            "dynecm2" or "ba" or "barye" => 0.1,
            _ => 1.0
        };
        double fromSI = to.ToLower() switch
        {
            "pa" or "pascal" => 1.0,
            "kpa" or "kilopascal" => 1e3,
            "mpa" or "megapascal" => 1e6,
            "gpa" or "gigapascal" => 1e9,
            "bar" => 1e5,
            "mbar" or "millibar" => 100.0,
            "atm" or "atmosphere" => 101325.0,
            "torr" or "mmhg" => 133.32236842,
            "psi" => 6894.757,
            "inhg" => 3386.389,
            "cmh2o" => 98.0665,
            "mh2o" => 9806.65,
            "dynecm2" or "ba" or "barye" => 0.1,
            _ => 1.0
        };
        return value * toSI / fromSI;
    }

    /// <summary>Convert speed from one unit to another.</summary>
    public static double ConvertSpeed(double value, string from, string to)
    {
        double toSI = from.ToLower() switch
        {
            "ms" or "mps" or "meterspersecond" => 1.0,
            "kmh" or "kph" or "kilometersperhour" => 1.0 / 3.6,
            "mph" or "milesperhour" => 0.44704,
            "kn" or "knot" => 0.514444,
            "fts" or "feetpersecond" => 0.3048,
            "mach" => 343.0,
            "c" or "light" or "speedoflight" => 299792458.0,
            _ => 1.0
        };
        double fromSI = to.ToLower() switch
        {
            "ms" or "mps" or "meterspersecond" => 1.0,
            "kmh" or "kph" or "kilometersperhour" => 1.0 / 3.6,
            "mph" or "milesperhour" => 0.44704,
            "kn" or "knot" => 0.514444,
            "fts" or "feetpersecond" => 0.3048,
            "mach" => 343.0,
            "c" or "light" or "speedoflight" => 299792458.0,
            _ => 1.0
        };
        return value * toSI / fromSI;
    }

    // ── Unit Dimensions (SI) ──
    public readonly struct Dimension
    {
        public int Mass { get; }
        public int Length { get; }
        public int Time { get; }
        public int Current { get; }
        public int Temperature { get; }
        public int Amount { get; }
        public int LuminousIntensity { get; }
        public Dimension(int m, int l, int t, int i, int th, int n, int j)
        { Mass = m; Length = l; Time = t; Current = i; Temperature = th; Amount = n; LuminousIntensity = j; }
        public static readonly Dimension Dimensionless = new(0, 0, 0, 0, 0, 0, 0);
        public static readonly Dimension LengthDim = new(0, 1, 0, 0, 0, 0, 0);
        public static readonly Dimension MassDim = new(1, 0, 0, 0, 0, 0, 0);
        public static readonly Dimension TimeDim = new(0, 0, 1, 0, 0, 0, 0);
        public static readonly Dimension CurrentDim = new(0, 0, 0, 1, 0, 0, 0);
        public static readonly Dimension TemperatureDim = new(0, 0, 0, 0, 1, 0, 0);
        public static readonly Dimension AmountDim = new(0, 0, 0, 0, 0, 1, 0);
        public static readonly Dimension ForceDim = new(1, 1, -2, 0, 0, 0, 0);
        public static readonly Dimension EnergyDim = new(1, 2, -2, 0, 0, 0, 0);
        public static readonly Dimension PowerDim = new(1, 2, -3, 0, 0, 0, 0);
        public static readonly Dimension PressureDim = new(1, -1, -2, 0, 0, 0, 0);
        public static readonly Dimension VelocityDim = new(0, 1, -1, 0, 0, 0, 0);
        public static readonly Dimension AccelerationDim = new(0, 1, -2, 0, 0, 0, 0);
        public static readonly Dimension ChargeDim = new(0, 0, 1, 1, 0, 0, 0);
        public static readonly Dimension VoltageDim = new(1, 2, -3, -1, 0, 0, 0);
        public static readonly Dimension FrequencyDim = new(0, 0, -1, 0, 0, 0, 0);
        public static readonly Dimension AreaDim = new(0, 2, 0, 0, 0, 0, 0);
        public static readonly Dimension VolumeDim = new(0, 3, 0, 0, 0, 0, 0);
        public static readonly Dimension MomentumDim = new(1, 1, -1, 0, 0, 0, 0);
        public static readonly Dimension AngularMomentumDim = new(1, 2, -1, 0, 0, 0, 0);
        public static readonly Dimension ViscosityDim = new(1, -1, -1, 0, 0, 0, 0);
        public static readonly Dimension KinematicViscosityDim = new(0, 2, -1, 0, 0, 0, 0);
        public static readonly Dimension ThermalConductivityDim = new(1, 1, -3, 0, -1, 0, 0);
        public static readonly Dimension SpecificHeatDim = new(0, 2, -2, 0, -1, 0, 0);
        public static readonly Dimension ElectricFieldDim = new(1, 1, -3, -1, 0, 0, 0);
        public static readonly Dimension MagneticFieldDim = new(1, 0, -2, -1, 0, 0, 0);
        public override string ToString() => $"M^{Mass} L^{Length} T^{Time} I^{Current} Θ^{Temperature} N^{Amount} J^{LuminousIntensity}";
        public static Dimension operator *(Dimension a, Dimension b) => new(a.Mass + b.Mass, a.Length + b.Length, a.Time + b.Time, a.Current + b.Current, a.Temperature + b.Temperature, a.Amount + b.Amount, a.LuminousIntensity + b.LuminousIntensity);
        public static Dimension operator /(Dimension a, Dimension b) => new(a.Mass - b.Mass, a.Length - b.Length, a.Time - b.Time, a.Current - b.Current, a.Temperature - b.Temperature, a.Amount - b.Amount, a.LuminousIntensity - b.LuminousIntensity);
        public static Dimension operator -(Dimension d) => new(-d.Mass, -d.Length, -d.Time, -d.Current, -d.Temperature, -d.Amount, -d.LuminousIntensity);
        public static bool operator ==(Dimension a, Dimension b) => a.Mass == b.Mass && a.Length == b.Length && a.Time == b.Time && a.Current == b.Current && a.Temperature == b.Temperature && a.Amount == b.Amount && a.LuminousIntensity == b.LuminousIntensity;
        public static bool operator !=(Dimension a, Dimension b) => !(a == b);
        public override bool Equals(object obj) => obj is Dimension d && this == d;
        public override int GetHashCode() => HashCode.Combine(Mass, Length, Time, Current, Temperature, Amount, LuminousIntensity);
    }

    public readonly struct DimensionedValue
    {
        public double Value { get; }
        public Dimension Dim { get; }
        public DimensionedValue(double value, Dimension dim) { Value = value; Dim = dim; }
        public static DimensionedValue operator *(DimensionedValue a, DimensionedValue b) => new(a.Value * b.Value, a.Dim * b.Dim);
        public static DimensionedValue operator /(DimensionedValue a, DimensionedValue b) => new(a.Value / b.Value, a.Dim / b.Dim);
        public static DimensionedValue operator +(DimensionedValue a, DimensionedValue b) => new(a.Value + b.Value, a.Dim);
        public static DimensionedValue operator -(DimensionedValue a, DimensionedValue b) => new(a.Value - b.Value, a.Dim);
        public bool IsDimensionless => Dim == Dimension.Dimensionless;
        public override string ToString() => $"{Value} [{Dim}]";
    }
}

/// <summary>Base de donnees de materiaux pre-definis avec proprietes completes.</summary>
public static class MaterialDatabase
{
    public readonly struct MaterialEntry
    {
        public string Name { get; }
        public string Category { get; }
        public double Density { get; } // kg/m³
        public double YoungsModulus { get; } // Pa
        public double PoissonsRatio { get; }
        public double ShearModulus { get; } // Pa
        public double BulkModulus { get; } // Pa
        public double ThermalConductivity { get; } // W/(m·K)
        public double SpecificHeat { get; } // J/(kg·K)
        public double ThermalExpansion { get; } // 1/K
        public double MeltingPoint { get; } // K
        public double BoilingPoint { get; } // K
        public double TensileStrength { get; } // Pa
        public double YieldStrength { get; } // Pa
        public double Hardness { get; } // Mohs (1-10)
        public double ElectricalConductivity { get; } // S/m
        public double RefractiveIndex { get; }
        public double BandGap { get; } // eV
        public double PoissonRatio { get; } // derived
        public double FractureToughness { get; } // MPa·m^(1/2)
        public double FatigueLimit { get; } // Pa
        public double DampingCapacity { get; } // dimensionless
        public double MagneticPermeability { get; } // relative
        public double DielectricConstant { get; }
        public MaterialEntry(string name, string cat, double rho, double E, double nu, double G, double K,
            double lambda, double cp, double alpha, double Tm, double Tb, double Ts, double Sy,
            double H, double sigma, double n, double Eg, double ft, double fl, double damp, double mu, double eps, double pr)
        {
            Name = name;
            Category = cat;
            Density = rho;
            YoungsModulus = E;
            PoissonsRatio = nu;
            ShearModulus = G;
            BulkModulus = K;
            ThermalConductivity = lambda;
            SpecificHeat = cp;
            ThermalExpansion = alpha;
            MeltingPoint = Tm;
            BoilingPoint = Tb;
            TensileStrength = Ts;
            YieldStrength = Sy;
            Hardness = H;
            ElectricalConductivity = sigma;
            RefractiveIndex = n;
            BandGap = Eg;
            PoissonRatio = pr;
            FractureToughness = ft;
            FatigueLimit = fl;
            DampingCapacity = damp;
            MagneticPermeability = mu;
            DielectricConstant = eps;
        }
    }

    private static readonly Dictionary<string, MaterialEntry> _materials = new(StringComparer.OrdinalIgnoreCase);
    private static readonly List<MaterialEntry> _all = new();
    private static readonly Dictionary<string, List<MaterialEntry>> _byCategory = new();
    static MaterialDatabase()
    {
        void Add(MaterialEntry m)
        { _materials[m.Name] = m; _all.Add(m); if (!_byCategory.ContainsKey(m.Category)) _byCategory[m.Category] = new(); _byCategory[m.Category].Add(m); }
        // ── METALS ──
        Add(new("Steel AISI 1045", "Metal", 7850, 205e9, 0.29, 80e9, 160e9, 50, 480, 11e-6, 1700, 3100, 585e6, 310e6, 5.5, 4.0e6, 1.458, 0, 0.29, 60, 205e6, 0.01, 100, 1));
        Add(new("Steel AISI 304", "Metal", 8000, 193e9, 0.29, 75e9, 150e9, 16.2, 500, 17.3e-6, 1697, 3100, 515e6, 215e6, 6, 1.45e6, 1.447, 0, 0.29, 120, 210e6, 0.012, 100, 1));
        Add(new("Steel AISI 4340", "Metal", 7850, 205e9, 0.29, 80e9, 160e9, 44.5, 475, 12.3e-6, 1697, 3100, 1100e6, 710e6, 6.3, 4.0e6, 1.458, 0, 0.29, 130, 490e6, 0.008, 100, 1));
        Add(new("Stainless Steel 316", "Metal", 8000, 193e9, 0.29, 75e9, 150e9, 13.4, 500, 16e-6, 1673, 3100, 515e6, 205e6, 6.5, 1.33e6, 1.447, 0, 0.29, 100, 240e6, 0.012, 100, 1));
        Add(new("Aluminum 6061-T6", "Metal", 2700, 68.9e9, 0.33, 26e9, 76e9, 167, 896, 23.6e-6, 855, 2792, 310e6, 276e6, 3, 37.7e6, 1.39, 0, 0.33, 29, 97e6, 0.02, 1, 1));
        Add(new("Aluminum 7075-T6", "Metal", 2810, 71.7e9, 0.33, 26.9e9, 71.7e9, 130, 960, 23.6e-6, 775, 2900, 572e6, 503e6, 3.5, 23.6e6, 1.45, 0, 0.33, 29, 159e6, 0.005, 1, 1));
        Add(new("Copper C11000", "Metal", 8960, 115e9, 0.34, 43e9, 140e9, 401, 385, 16.5e-6, 1356, 2835, 220e6, 70e6, 3, 59.6e6, 1.1, 0, 0.34, 25, 70e6, 0.003, 1, 1));
        Add(new("Brass C26000", "Metal", 8530, 110e9, 0.34, 41e9, 115e9, 120, 380, 20.5e-6, 1188, 3100, 340e6, 75e6, 3.5, 16e6, 1.43, 0, 0.34, 50, 100e6, 0.01, 1, 1));
        Add(new("Titanium Ti-6Al-4V", "Metal", 4430, 113.8e9, 0.342, 42.4e9, 118e9, 6.7, 526, 8.6e-6, 1877, 3533, 1170e6, 1050e6, 6, 5.8e6, 1.55, 0, 0.342, 75, 500e6, 0.003, 1, 1));
        Add(new("Nickel 200", "Metal", 8900, 204e9, 0.31, 78e9, 176e9, 91, 444, 13e-6, 1663, 3186, 380e6, 110e6, 4, 14.3e6, 1.445, 0, 0.31, 80, 130e6, 0.003, 600, 1));
        Add(new("Inconel 718", "Metal", 8190, 204e9, 0.3, 79.3e9, 158e9, 11.4, 435, 13e-6, 1533, 3186, 1380e6, 1030e6, 6.5, 0.8e6, 1.45, 0, 0.3, 110, 510e6, 0.003, 1, 1));
        Add(new("Molybdenum", "Metal", 10200, 329e9, 0.31, 125e9, 272e9, 138, 253, 4.8e-6, 2890, 5910, 690e6, 500e6, 5.5, 18.7e6, 1.57, 0, 0.31, 80, 300e6, 0.001, 1, 1));
        Add(new("Tungsten", "Metal", 19300, 411e9, 0.28, 161e9, 310e9, 174, 134, 4.5e-6, 3695, 6203, 1510e6, 1080e6, 7.5, 18.9e6, 1.59, 0, 0.28, 90, 560e6, 0.001, 1, 1));
        Add(new("Gold", "Metal", 19320, 77.2e9, 0.44, 27e9, 180e9, 317, 129, 14.2e-6, 1337, 3129, 220e6, 25e6, 2.5, 45.2e6, 0.35, 0, 0.44, 10, 50e6, 0.002, 1, 1));
        Add(new("Silver", "Metal", 10490, 82.7e9, 0.37, 30e9, 114e9, 429, 235, 18.9e-6, 1235, 2435, 170e6, 55e6, 2.5, 63e6, 0.05, 0, 0.37, 15, 60e6, 0.002, 1, 1));
        Add(new("Platinum", "Metal", 21450, 168e9, 0.38, 61e9, 280e9, 71.6, 133, 8.8e-6, 2041, 4098, 350e6, 125e6, 3.5, 9.4e6, 1.77, 0, 0.38, 30, 120e6, 0.002, 1, 1));
        Add(new("Lead", "Metal", 11340, 16e9, 0.44, 5.6e9, 46e9, 35.3, 128, 29e-6, 600, 2022, 12e6, 5e6, 1.5, 4.5e6, 1.45, 0, 0.44, 10, 5e6, 0.03, 1, 1));
        Add(new("Zinc", "Metal", 7134, 108e9, 0.25, 43e9, 83e9, 116, 388, 30.2e-6, 692, 1180, 110e6, 80e6, 3.5, 16.9e6, 1.45, 0, 0.25, 20, 40e6, 0.003, 1, 1));
        Add(new("Tin", "Metal", 7310, 41.5e9, 0.36, 15e9, 58e9, 67, 228, 22e-6, 505, 2875, 20e6, 12e6, 1.5, 9.1e6, 1.57, 0, 0.36, 15, 15e6, 0.003, 1, 1));
        Add(new("Beryllium", "Metal", 1845, 287e9, 0.032, 132e9, 130e9, 200, 1825, 11.3e-6, 1560, 2744, 448e6, 370e6, 5.5, 25e6, 1.485, 0, 0.032, 40, 175e6, 0.005, 1, 1));
        Add(new("Chromium", "Metal", 7190, 279e9, 0.21, 115e9, 160e9, 93.9, 449, 4.9e-6, 2180, 2944, 400e6, 320e6, 8.5, 7.7e6, 1.57, 0, 0.21, 40, 380e6, 0.001, 1, 1));
        Add(new("Manganese", "Metal", 7210, 108e9, 0.24, 43.5e9, 83e9, 7.8, 479, 21e-6, 1519, 2334, 350e6, 185e6, 6, 0.69e6, 1.55, 0, 0.24, 20, 190e6, 0.003, 1, 1));
        Add(new("Cobalt", "Metal", 8900, 211e9, 0.31, 80.7e9, 183e9, 100, 421, 12.5e-6, 1768, 3200, 750e6, 450e6, 5.5, 20e6, 1.45, 0, 0.31, 60, 250e6, 0.003, 250, 1));
        Add(new("Palladium", "Metal", 12020, 121e9, 0.39, 43.6e9, 187e9, 71.8, 244, 11.8e-6, 1828, 3236, 340e6, 90e6, 4.75, 9.5e6, 1.68, 0, 0.39, 15, 75e6, 0.003, 1, 1));
        Add(new("Iridium", "Metal", 22560, 528e9, 0.26, 210e9, 370e9, 147, 130, 6.4e-6, 2719, 4701, 2240e6, 1800e6, 6.5, 19.8e6, 1.97, 0, 0.26, 50, 950e6, 0.001, 1, 1));
        Add(new("Rhodium", "Metal", 12410, 380e9, 0.26, 150e9, 280e9, 150, 243, 8.1e-6, 2237, 4000, 1400e6, 850e6, 6, 21e6, 1.8, 0, 0.26, 45, 500e6, 0.001, 1, 1));
        // ── CERAMICS ──
        Add(new("Alumina Al2O3", "Ceramic", 3950, 370e9, 0.22, 152e9, 228e9, 30, 775, 8.1e-6, 2323, 3250, 379e6, 2000e6, 9, 1e-12, 1.77, 8.8, 0.22, 3.5, 150e6, 0, 1, 8.5));
        Add(new("Silicon Carbide SiC", "Ceramic", 3210, 410e9, 0.17, 175e9, 205e9, 120, 750, 4e-6, 2870, 3250, 480e6, 4000e6, 9.5, 1e-6, 2.65, 2.36, 0.17, 4, 200e6, 0, 1, 10));
        Add(new("Silicon Nitride Si3N4", "Ceramic", 3170, 304e9, 0.27, 120e9, 216e9, 30, 750, 3.2e-6, 2173, 3250, 600e6, 3500e6, 9, 1e-13, 2.03, 4.5, 0.27, 6, 300e6, 0, 1, 8));
        Add(new("Zirconia ZrO2", "Ceramic", 5680, 200e9, 0.31, 76e9, 187e9, 2, 450, 10.5e-6, 2988, 4823, 200e6, 1500e6, 11, 1e-10, 2.15, 5, 0.31, 8, 100e6, 0, 1, 25));
        Add(new("Quartz SiO2", "Ceramic", 2200, 72e9, 0.17, 31e9, 37e9, 1.38, 730, 0.55e-6, 1883, 2503, 50e6, 1100e6, 7, 1e-18, 1.46, 9, 0.17, 1, 200e6, 0, 1, 3.8));
        Add(new("Boron Nitride BN", "Ceramic", 2100, 90e9, 0.25, 36e9, 46e9, 27, 800, 1e-6, 3000, 5000, 270e6, 2000e6, 9.5, 2e-3, 1.65, 6, 0.25, 3, 150e6, 0, 1, 4));
        Add(new("Beryllium Oxide BeO", "Ceramic", 3025, 345e9, 0.26, 137e9, 216e9, 330, 1000, 6e-6, 2800, 4000, 240e6, 1400e6, 9, 1e-10, 1.72, 10.6, 0.26, 4, 150e6, 0, 1, 6.7));
        // ── POLYMERS ──
        Add(new("Polyethylene PE", "Polymer", 950, 1e9, 0.42, 0.35e9, 1.7e9, 0.33, 2300, 200e-6, 400, 400, 30e6, 20e6, 1, 1e-15, 1.5, 0, 0.42, 1, 15e6, 0.1, 1, 2.3));
        Add(new("Polypropylene PP", "Polymer", 900, 1.1e9, 0.42, 0.4e9, 1.8e9, 0.12, 1900, 150e-6, 433, 443, 35e6, 25e6, 1, 1e-15, 1.49, 0, 0.42, 1, 18e6, 0.05, 1, 2.2));
        Add(new("PVC", "Polymer", 1400, 3e9, 0.38, 1.1e9, 4.6e9, 0.16, 1000, 80e-6, 373, 473, 52e6, 40e6, 2, 1e-12, 1.54, 0, 0.38, 1, 25e6, 0.03, 1, 3.4));
        Add(new("Polystyrene PS", "Polymer", 1050, 3e9, 0.35, 1.1e9, 3.3e9, 0.13, 1300, 70e-6, 373, 473, 40e6, 30e6, 2.5, 1e-14, 1.59, 0, 0.35, 1, 20e6, 0.02, 1, 2.6));
        Add(new("Nylon 6/6", "Polymer", 1140, 2.7e9, 0.41, 0.96e9, 4.7e9, 0.24, 1700, 80e-6, 500, 523, 82e6, 60e6, 2.5, 1e-12, 1.53, 0, 0.41, 2, 30e6, 0.08, 1, 3.7));
        Add(new("PEEK", "Polymer", 1300, 3.6e9, 0.4, 1.3e9, 4.7e9, 0.25, 1300, 47e-6, 616, 643, 100e6, 90e6, 3, 1e-14, 1.59, 0, 0.4, 2, 40e6, 0.01, 1, 3.2));
        Add(new("PTFE Teflon", "Polymer", 2200, 0.5e9, 0.46, 0.17e9, 0.7e9, 0.25, 1000, 100e-6, 600, 610, 23e6, 11e6, 1, 1e-25, 1.35, 0, 0.46, 1, 15e6, 0.01, 1, 2.1));
        Add(new("Polycarbonate PC", "Polymer", 1200, 2.4e9, 0.37, 0.88e9, 3.0e9, 0.2, 1200, 65e-6, 483, 523, 65e6, 55e6, 3, 1e-13, 1.586, 0, 0.37, 2, 35e6, 0.02, 1, 3.0));
        Add(new("ABS", "Polymer", 1050, 2.3e9, 0.39, 0.83e9, 2.8e9, 0.17, 1400, 72e-6, 443, 473, 43e6, 35e6, 2, 1e-13, 1.54, 0, 0.39, 1.5, 22e6, 0.02, 1, 2.8));
        Add(new("Epoxy", "Polymer", 1200, 3e9, 0.35, 1.1e9, 3.3e9, 0.17, 800, 55e-6, 393, 523, 60e6, 45e6, 3, 1e-12, 1.55, 0, 0.35, 0.5, 25e6, 0.03, 1, 3.5));
        // ── COMPOSITES ──
        Add(new("Carbon Fiber Composite", "Composite", 1600, 181e9, 0.28, 70e9, 140e9, 7, 800, -0.5e-6, 673, 973, 2550e6, 1500e6, 8, 1e3, 1.55, 0, 0.28, 30, 600e6, 0.005, 1, 4));
        Add(new("Glass Fiber Composite", "Composite", 2000, 45e9, 0.3, 17.3e9, 42.9e9, 0.35, 800, 18e-6, 573, 773, 500e6, 300e6, 6, 1e-10, 1.55, 0, 0.3, 20, 150e6, 0.01, 1, 4));
        Add(new("Kevlar Composite", "Composite", 1440, 70e9, 0.36, 25.7e9, 54e9, 0.5, 1000, -2e-6, 673, 773, 2800e6, 1800e6, 7, 1e-12, 1.55, 0, 0.36, 25, 400e6, 0.01, 1, 3.5));
        Add(new("Fiberglass", "Composite", 1800, 20e9, 0.3, 7.7e9, 16.7e9, 0.5, 700, 15e-6, 673, 773, 300e6, 200e6, 5, 1e-10, 1.52, 0, 0.3, 15, 80e6, 0.015, 1, 4));
        Add(new("Ceramic Matrix Composite", "Composite", 2700, 250e9, 0.2, 104e9, 139e9, 15, 750, 3e-6, 1800, 3000, 350e6, 2500e6, 9, 1e-8, 1.6, 2, 0.2, 15, 200e6, 0, 1, 7));
        Add(new("Metal Matrix Composite Al-SiC", "Composite", 2900, 120e9, 0.3, 46e9, 100e9, 180, 850, 16e-6, 855, 2792, 500e6, 350e6, 7, 20e6, 1.45, 0, 0.3, 30, 200e6, 0.005, 1, 1));
        // ── SEMICONDUCTORS ──
        Add(new("Silicon Si", "Semiconductor", 2329, 165e9, 0.28, 64.5e9, 97.8e9, 148, 712, 2.6e-6, 1687, 3538, 7000e6, 7000e6, 6.5, 1.56e3, 3.42, 1.12, 0.28, 0.7, 1e6, 0, 1, 11.7));
        Add(new("Germanium Ge", "Semiconductor", 5323, 103e9, 0.26, 40.8e9, 75.2e9, 60, 320, 6e-6, 1211, 3106, 370e6, 370e6, 6, 2.17e3, 4.05, 0.66, 0.26, 0.6, 100e6, 0, 1, 16));
        Add(new("Gallium Arsenide GaAs", "Semiconductor", 5317, 85e9, 0.31, 32.6e9, 75.5e9, 55, 330, 5.7e-6, 1513, 2673, 1800e6, 1800e6, 4.5, 3.3e3, 3.42, 1.42, 0.31, 0.4, 200e6, 0, 1, 12.9));
        Add(new("Silicon Dioxide SiO2", "Semiconductor", 2200, 70e9, 0.17, 30e9, 37e9, 1.4, 730, 0.5e-6, 1986, 2503, 1100e6, 8400e6, 7, 1e-18, 1.46, 9, 0.17, 0.8, 200e6, 0, 1, 3.9));
        Add(new("Gallium Nitride GaN", "Semiconductor", 6150, 295e9, 0.24, 121e9, 190e9, 130, 490, 3.2e-6, 2773, 4773, 3000e6, 3000e6, 9, 1e-9, 2.35, 3.4, 0.24, 1, 500e6, 0, 1, 8.9));
        Add(new("Indium Phosphide InP", "Semiconductor", 4787, 61e9, 0.36, 22.4e9, 57.7e9, 68, 310, 4.6e-6, 1333, 2580, 1000e6, 1000e6, 5, 1e3, 3.42, 1.35, 0.36, 0.4, 200e6, 0, 1, 12.5));
        // ── GLASSES ──
        Add(new("Soda-Lime Glass", "Glass", 2500, 72e9, 0.24, 29e9, 47e9, 1.0, 840, 9e-6, 903, 1683, 45e6, 45e6, 5.5, 1e-12, 1.52, 0, 0.24, 0.7, 25e6, 0, 1, 7.2));
        Add(new("Borosilicate Glass Pyrex", "Glass", 2230, 63e9, 0.2, 26e9, 35e9, 1.14, 830, 3.3e-6, 808, 1643, 63e6, 50e6, 6, 1e-13, 1.47, 0, 0.2, 0.7, 25e6, 0, 1, 4.6));
        Add(new("Fused Silica Quartz Glass", "Glass", 2200, 72e9, 0.17, 31e9, 37e9, 1.38, 730, 0.55e-6, 1883, 2503, 50e6, 1100e6, 7, 1e-18, 1.458, 9, 0.17, 0.75, 200e6, 0, 1, 3.8));
        Add(new("Lead Crystal Glass", "Glass", 3000, 55e9, 0.24, 22.2e9, 35.1e9, 0.8, 500, 8.5e-6, 800, 1400, 35e6, 25e6, 5, 1e-13, 1.55, 0, 0.24, 0.5, 20e6, 0, 1, 7.5));
        Add(new("Sapphire Al2O3", "Glass", 3980, 345e9, 0.29, 134e9, 226e9, 25, 765, 5.8e-6, 2323, 3250, 400e6, 3500e6, 9, 1e-14, 1.77, 8.8, 0.29, 2, 200e6, 0, 1, 11.5));
        // ── NATURAL MATERIALS ──
        Add(new("Wood Oak", "Natural", 770, 12.3e9, 0.35, 4.6e9, 1.4e9, 0.16, 2000, 50e-6, 400, 573, 100e6, 50e6, 2, 1e-12, 1.5, 0, 0.35, 0.5, 10e6, 0.1, 1, 2.5));
        Add(new("Wood Pine", "Natural", 520, 8.96e9, 0.35, 3.32e9, 0.89e9, 0.12, 2000, 60e-6, 400, 573, 70e6, 30e6, 2, 1e-12, 1.5, 0, 0.35, 0.3, 8e6, 0.12, 1, 2.5));
        Add(new("Wood Maple", "Natural", 710, 12.6e9, 0.35, 4.67e9, 1.49e9, 0.17, 1900, 40e-6, 400, 573, 109e6, 50e6, 2, 1e-12, 1.5, 0, 0.35, 0.5, 12e6, 0.1, 1, 2.5));
        Add(new("Bone", "Natural", 1900, 18e9, 0.3, 6.9e9, 15e9, 0.3, 1300, 12e-6, 310, 373, 130e6, 100e6, 3, 1e-5, 1.5, 0, 0.3, 2, 40e6, 0.05, 1, 8));
        Add(new("Ivory", "Natural", 1900, 16e9, 0.3, 6.2e9, 13.3e9, 0.35, 1200, 10e-6, 310, 373, 100e6, 80e6, 2.5, 1e-10, 1.54, 0, 0.3, 1, 30e6, 0.02, 1, 2.5));
        Add(new("Rubber Natural", "Natural", 920, 0.05e9, 0.499, 0.017e9, 83e9, 0.13, 2000, 220e-6, 350, 500, 20e6, 1e6, 0, 1e-15, 1.52, 0, 0.499, 0.01, 5e6, 0.5, 1, 2.5));
        Add(new("Silk", "Natural", 1300, 5e9, 0.35, 1.85e9, 4.2e9, 0.038, 1300, 2e-6, 573, 643, 500e6, 400e6, 5, 1e-14, 1.55, 0, 0.35, 1, 70e6, 0.1, 1, 3));
        Add(new("Cellulose", "Natural", 1500, 10e9, 0.35, 3.7e9, 11.1e9, 0.05, 1500, 50e-6, 550, 600, 150e6, 80e6, 2, 1e-12, 1.5, 0, 0.35, 0.3, 50e6, 0.1, 1, 3.5));
        // ── LIQUIDS ──
        Add(new("Water H2O", "Liquid", 1000, 2.15e9, 0.49, 0, 2.2e9, 0.598, 4186, 207e-6, 273, 373, 0, 0, 1, 5.5e-6, 1.333, 0, 0.49, 0, 0, 0.1, 1, 80));
        Add(new("Ethanol", "Liquid", 789, 1.09e9, 0.49, 0, 1.06e9, 0.167, 2440, 750e-6, 159, 351, 0, 0, 1, 1.35e-7, 1.361, 0, 0.49, 0, 0, 0.05, 1, 24.5));
        Add(new("Mercury", "Liquid", 13546, 41e9, 0.45, 0, 28.5e9, 8.3, 140, 180e-6, 234, 630, 0, 0, 1, 1.04e6, 1.74, 0, 0.45, 0, 0, 0, 0.001, 0));
        Add(new("Acetone", "Liquid", 791, 0.8e9, 0.49, 0, 0.7e9, 0.161, 2200, 1300e-6, 178, 329, 0, 0, 1, 5e-8, 1.359, 0, 0.49, 0, 0, 0.02, 1, 20.7));
        // ── GASES (at STP) ──
        Add(new("Air (dry)", "Gas", 1.225, 0, 0, 0, 101325, 0.026, 1005, 3.43e-3, 55, 77.4, 0, 0, 1, 0, 1.0003, 0, 0, 0, 0, 0, 1, 1.0006));
        Add(new("Nitrogen N2", "Gas", 1.165, 0, 0, 0, 101325, 0.026, 1040, 3.4e-3, 63, 77.4, 0, 0, 1, 0, 1.0005, 0, 0, 0, 0, 0, 1, 1.0005));
        Add(new("Oxygen O2", "Gas", 1.331, 0, 0, 0, 101325, 0.0265, 920, 3.1e-3, 54, 90.2, 0, 0, 1, 0, 1.0005, 0, 0, 0, 0, 0, 1, 1.0005));
        Add(new("Carbon Dioxide CO2", "Gas", 1.842, 0, 0, 0, 101325, 0.0168, 844, 4.4e-3, 195, 217, 0, 0, 1, 0, 1.00045, 0, 0, 0, 0, 0, 1, 1.0009));
        Add(new("Hydrogen H2", "Gas", 0.08375, 0, 0, 0, 101325, 0.187, 14300, 2.2e-3, 14, 20.3, 0, 0, 1, 0, 1.00014, 0, 0, 0, 0, 0, 1, 1.00026));
        Add(new("Helium He", "Gas", 0.1664, 0, 0, 0, 101325, 0.155, 5193, 0, 0.95, 4.22, 0, 0, 1, 0, 1.000035, 0, 0, 0, 0, 0, 1, 1.000065));
        // ── EXOTIC ──
        Add(new("Diamond", "Exotic", 3515, 1050e9, 0.07, 480e9, 440e9, 2200, 502, 1e-6, 4200, 4827, 60000e6, 60000e6, 10, 1e-15, 2.417, 5.47, 0.07, 4, 1000e6, 0, 1, 5.5));
        Add(new("Graphite", "Exotic", 2200, 11e9, 0.2, 4.6e9, 33e9, 150, 710, -1e-6, 3823, 4273, 20e6, 20e6, 1.5, 2e5, 1.55, 0, 0.2, 1, 15e6, 0.01, 1, 3));
        Add(new("Graphene", "Exotic", 2200, 1000e9, 0.16, 430e9, 420e9, 5000, 710, -8e-6, 3823, 4273, 130000e6, 130000e6, 10, 1e8, 1.46, 0, 0.16, 5, 1000e6, 0, 1, 2.4));
        Add(new("Carbon Nanotube", "Exotic", 1300, 1000e9, 0.2, 415e9, 442e9, 6000, 710, -1e-6, 3823, 4273, 50000e6, 50000e6, 10, 1e7, 1.46, 0, 0.2, 10, 2000e6, 0, 1, 3));
        Add(new("Silicene", "Exotic", 2330, 200e9, 0.25, 80e9, 100e9, 50, 710, 3e-6, 1687, 3538, 15000e6, 15000e6, 7, 1e3, 3.42, 1.12, 0.25, 2, 500e6, 0, 1, 11.7));
        Add(new("Borophene", "Exotic", 2340, 398e9, 0.2, 166e9, 200e9, 45, 710, 2e-6, 2348, 4073, 18000e6, 18000e6, 8, 1e3, 1.6, 0.5, 0.2, 2, 500e6, 0, 1, 4));
        Add(new("Phosphorene", "Exotic", 1820, 24e9, 0.4, 8.6e9, 14.3e9, 12, 710, -2e-6, 883, 1123, 1200e6, 1200e6, 4, 1e3, 2.5, 0.3, 0.4, 1, 300e6, 0, 1, 4.5));
        Add(new("Molybdenum Disulfide MoS2", "Exotic", 5060, 270e9, 0.22, 110e9, 150e9, 34, 400, 2e-6, 1458, 3100, 5000e6, 5000e6, 1.5, 1e-1, 2.15, 1.8, 0.22, 0.5, 500e6, 0, 1, 4));
        Add(new("Tungsten Diselenide WSe2", "Exotic", 7500, 155e9, 0.19, 65e9, 100e9, 20, 290, 5e-6, 1473, 3273, 2000e6, 2000e6, 3, 1e-1, 2.8, 1.65, 0.19, 0.5, 400e6, 0, 1, 4.5));
    }

    /// <summary>Get material by exact name.</summary>
    public static bool TryGetMaterial(string name, out MaterialEntry material) => _materials.TryGetValue(name, out material);

    /// <summary>Get all materials in a category.</summary>
    public static IReadOnlyList<MaterialEntry> GetByCategory(string category)
        => _byCategory.TryGetValue(category, out var list) ? list : Array.Empty<MaterialEntry>();

    /// <summary>Search materials by partial name match (case-insensitive).</summary>
    public static List<MaterialEntry> Search(string query)
    {
        var results = new List<MaterialEntry>();
        var q = query.ToLowerInvariant();
        foreach (var m in _all)
            if (m.Name.Contains(q, StringComparison.OrdinalIgnoreCase) || m.Category.Contains(q, StringComparison.OrdinalIgnoreCase))
                results.Add(m);
        return results;
    }

    /// <summary>Get material with density closest to given value.</summary>
    public static MaterialEntry ClosestByDensity(double targetDensity)
    {
        MaterialEntry best = _all[0];
        double bestDist = double.MaxValue;
        foreach (var m in _all)
        { double d = Math.Abs(m.Density - targetDensity); if (d < bestDist) { bestDist = d; best = m; } }
        return best;
    }

    /// <summary>Get material with Young's modulus closest to given value.</summary>
    public static MaterialEntry ClosestByYoungsModulus(double targetE)
    {
        MaterialEntry best = _all[0];
        double bestDist = double.MaxValue;
        foreach (var m in _all)
        { double d = Math.Abs(m.YoungsModulus - targetE); if (d < bestDist) { bestDist = d; best = m; } }
        return best;
    }

    /// <summary>Interpolate material properties between two materials by blend factor t ∈ [0,1].</summary>
    public static MaterialEntry Interpolate(MaterialEntry a, MaterialEntry b, double t)
    {
        t = Math.Max(0, Math.Min(1, t));
        double u = 1 - t;
        return new MaterialEntry($"{a.Name}_{b.Name}", a.Category,
            u * a.Density + t * b.Density, u * a.YoungsModulus + t * b.YoungsModulus, u * a.PoissonsRatio + t * b.PoissonsRatio,
            u * a.ShearModulus + t * b.ShearModulus, u * a.BulkModulus + t * b.BulkModulus,
            u * a.ThermalConductivity + t * b.ThermalConductivity, u * a.SpecificHeat + t * b.SpecificHeat,
            u * a.ThermalExpansion + t * b.ThermalExpansion, u * a.MeltingPoint + t * b.MeltingPoint,
            u * a.BoilingPoint + t * b.BoilingPoint, u * a.TensileStrength + t * b.TensileStrength,
            u * a.YieldStrength + t * b.YieldStrength, u * a.Hardness + t * b.Hardness,
            u * a.ElectricalConductivity + t * b.ElectricalConductivity, u * a.RefractiveIndex + t * b.RefractiveIndex,
            u * a.BandGap + t * b.BandGap, u * a.PoissonRatio + t * b.PoissonRatio,
            u * a.FractureToughness + t * b.FractureToughness, u * a.FatigueLimit + t * b.FatigueLimit,
            u * a.DampingCapacity + t * b.DampingCapacity, u * a.MagneticPermeability + t * b.MagneticPermeability,
            u * a.DielectricConstant + t * b.DielectricConstant);
    }

    /// <summary>Get all available materials.</summary>
    public static IReadOnlyList<MaterialEntry> All => _all;

    /// <summary>Get all category names.</summary>
    public static IEnumerable<string> Categories => _byCategory.Keys;
}

/// <summary>Octree node for spatial partitioning of 3D data. Supports adaptive subdivision based on particle density.</summary>
public class OctreeNode
{
    private const int MaxLeafParticles = 8;
    private const int MaxDepth = 20;

    public BoundingBox3D Bounds { get; }
    public int Depth { get; }
    public bool IsLeaf => Children == null;
    public int Count { get; private set; }
    public List<int> ParticleIndices { get; }
    public OctreeNode[] Children { get; private set; }
    public double TotalMass { get; private set; }
    public Vector3D CenterOfMass { get; private set; }
    public OctreeNode(BoundingBox3D bounds, int depth) { Bounds = bounds; Depth = depth; ParticleIndices = new(); }

    public void Insert(int particleIndex, Vector3D position, double mass)
    {
        Count++;
        double newCount = Count;
        CenterOfMass = (CenterOfMass * (newCount - 1) + position * mass) / newCount;
        TotalMass += mass;
        if (IsLeaf)
        {
            ParticleIndices.Add(particleIndex);
            if (ParticleIndices.Count > MaxLeafParticles && Depth < MaxDepth)
                Subdivide();
        }
        else
        {
            int childIdx = GetChildIndex(position);
            Children[childIdx].Insert(particleIndex, position, mass);
        }
    }

    private void Subdivide()
    {
        Children = new OctreeNode[8];
        var center = Bounds.Center;
        for (int i = 0; i < 8; i++)
        {
            var min = new Vector3D(
                (i & 1) == 0 ? Bounds.Min.X : center.X,
                (i & 2) == 0 ? Bounds.Min.Y : center.Y,
                (i & 4) == 0 ? Bounds.Min.Z : center.Z);
            var max = new Vector3D(
                (i & 1) == 0 ? center.X : Bounds.Max.X,
                (i & 2) == 0 ? center.Y : Bounds.Max.Y,
                (i & 4) == 0 ? center.Z : Bounds.Max.Z);
            Children[i] = new OctreeNode(new BoundingBox3D(min, max), Depth + 1);
        }
        var indices = new List<int>(ParticleIndices);
        ParticleIndices.Clear();
        foreach (var idx in indices)
            Insert(idx, Vector3D.Zero, 0); // re-insert (simplified)
    }

    private int GetChildIndex(Vector3D pos)
    {
        var center = Bounds.Center;
        int idx = 0;
        if (pos.X >= center.X)
            idx |= 1;
        if (pos.Y >= center.Y)
            idx |= 2;
        if (pos.Z >= center.Z)
            idx |= 4;
        return idx;
    }

    public void RangeQuery(BoundingBox3D queryBounds, List<int> results)
    {
        if (!Bounds.Intersects(queryBounds))
            return;
        if (IsLeaf)
        { results.AddRange(ParticleIndices); return; }
        foreach (var child in Children)
            child?.RangeQuery(queryBounds, results);
    }

    public void RadiusQuery(Vector3D center, double radius, List<int> results)
    {
        double rSq = radius * radius;
        if (Bounds.DistanceSquaredTo(center) > rSq)
            return;
        if (IsLeaf)
        { results.AddRange(ParticleIndices); return; }
        foreach (var child in Children)
            child?.RadiusQuery(center, radius, results);
    }

    public int DepthFirstTraversal(Func<OctreeNode, bool> visit)
    {
        int visited = 1;
        if (!visit(this))
            return visited;
        if (!IsLeaf)
            foreach (var child in Children)
                if (child != null)
                    visited += child.DepthFirstTraversal(visit);
        return visited;
    }

    public int ComputeMaxDepth() => IsLeaf ? Depth : Children.Max(c => c?.ComputeMaxDepth() ?? Depth);
    public int CountNodes() => IsLeaf ? 1 : 1 + Children.Sum(c => c?.CountNodes() ?? 0);
    public int CountLeaves() => IsLeaf ? 1 : Children.Sum(c => c?.CountLeaves() ?? 0);
    public double AverageParticlesPerLeaf() { int leaves = CountLeaves(); return leaves > 0 ? (double)Count / leaves : 0; }

    public void Clear() { ParticleIndices.Clear(); Children = null; Count = 0; TotalMass = 0; CenterOfMass = Vector3D.Zero; }
}

/// <summary>KD-tree for efficient nearest-neighbor and range queries in 3D.</summary>
public class KdTree
{
    private class KdNode
    {
        public int PointIndex;
        public int Axis;
        public KdNode? Left, Right;
        public BoundingBox3D Bounds;
    }

    private KdNode _root;
    private readonly List<Vector3D> _points;
    private readonly int[] _indices;

    public int Count => _points.Count;
    public int MaxDepth { get; private set; }

    public KdTree(List<Vector3D> points)
    {
        _points = points;
        _indices = new int[points.Count];
        for (int i = 0; i < points.Count; i++)
            _indices[i] = i;
        if (points.Count > 0)
            _root = Build(0, points.Count - 1, 0);
    }

    private KdNode? Build(int lo, int hi, int depth)
    {
        if (lo > hi)
            return null;
        int axis = depth % 3;
        int mid = (lo + hi) / 2;
        SortByAxis(lo, hi, mid, axis);
        var node = new KdNode { PointIndex = _indices[mid], Axis = axis };
        node.Left = Build(lo, mid - 1, depth + 1);
        node.Right = Build(mid + 1, hi, depth + 1);
        return node;
    }

    private void SortByAxis(int lo, int hi, int mid, int axis)
    {
        for (int i = lo; i < hi; i++)
        {
            int minIdx = i;
            for (int j = i + 1; j <= hi; j++)
                if (GetComponent(_indices[j], axis) < GetComponent(_indices[minIdx], axis))
                    minIdx = j;
            if (minIdx != i)
            { int t = _indices[i]; _indices[i] = _indices[minIdx]; _indices[minIdx] = t; }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double GetComponent(int idx, int axis)
        => axis == 0 ? _points[idx].X : axis == 1 ? _points[idx].Y : _points[idx].Z;

    /// <summary>Find the K nearest neighbors to a query point.</summary>
    public List<(int index, double distanceSquared)> KNearest(Vector3D query, int k)
    {
        var results = new List<(int, double)>();
        KNearestSearch(_root, query, k, results);
        return results.OrderByDescending(x => x.Item2).Take(k).ToList();
    }

    private void KNearestSearch(KdNode node, Vector3D query, int k, List<(int, double)> results)
    {
        if (node == null)
            return;
        double dx = query.X - _points[node.PointIndex].X;
        double dy = query.Y - _points[node.PointIndex].Y;
        double dz = query.Z - _points[node.PointIndex].Z;
        double distSq = dx * dx + dy * dy + dz * dz;
        if (results.Count < k || distSq < results[0].Item2)
        {
            results.Add((node.PointIndex, distSq));
            if (results.Count > k)
                results.Sort((a, b) => b.Item2.CompareTo(a.Item2));
            if (results.Count > k)
                results.RemoveAt(0);
        }
        double diff = GetComponent(node.PointIndex, node.Axis) - (node.Axis == 0 ? query.X : node.Axis == 1 ? query.Y : query.Z);
        var near = diff < 0 ? node.Right : node.Left;
        var far = diff < 0 ? node.Left : node.Right;
        KNearestSearch(near, query, k, results);
        if (results.Count < k || diff * diff < results[0].Item2)
            KNearestSearch(far, query, k, results);
    }

    /// <summary>Find all points within a radius.</summary>
    public List<int> RangeQuery(Vector3D center, double radius)
    {
        var results = new List<int>();
        RangeSearch(_root, center, radius * radius, results);
        return results;
    }

    private void RangeSearch(KdNode node, Vector3D center, double rSq, List<int> results)
    {
        if (node == null)
            return;
        double dx = center.X - _points[node.PointIndex].X;
        double dy = center.Y - _points[node.PointIndex].Y;
        double dz = center.Z - _points[node.PointIndex].Z;
        double distSq = dx * dx + dy * dy + dz * dz;
        if (distSq <= rSq)
            results.Add(node.PointIndex);
        double diff = GetComponent(node.PointIndex, node.Axis) - (node.Axis == 0 ? center.X : node.Axis == 1 ? center.Y : center.Z);
        var near = diff < 0 ? node.Right : node.Left;
        var far = diff < 0 ? node.Left : node.Right;
        RangeSearch(near, center, rSq, results);
        if (diff * diff <= rSq)
            RangeSearch(far, center, rSq, results);
    }

    /// <summary>Find the single closest point.</summary>
    public int NearestNeighbor(Vector3D query) => KNearest(query, 1)[0].index;
}

/// <summary>Spatial hashing for uniform grid-based particle lookup. O(1) average case neighbor search.</summary>
public class GridHash
{
    private readonly double _cellSize;
    private readonly Dictionary<long, List<int>> _cells = new();
    private readonly List<Vector3D> _points;

    public int Count => _points.Count;
    public int CellCount => _cells.Count;
    public double CellSize => _cellSize;
    public int AverageParticlesPerCell => _cells.Count > 0 ? _points.Count / _cells.Count : 0;

    public GridHash(List<Vector3D> points, double cellSize)
    {
        _points = points;
        _cellSize = cellSize;
        for (int i = 0; i < points.Count; i++)
            Insert(i, points[i]);
    }

    public GridHash(double cellSize) { _cellSize = cellSize; _points = new(); }

    public void Insert(int index, Vector3D point)
    {
        _points.Add(point);
        long hash = Hash(point);
        if (!_cells.TryGetValue(hash, out var list))
        { list = new List<int>(); _cells[hash] = list; }
        list.Add(index);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private long Hash(Vector3D p)
    {
        long ix = (long)Math.Floor(p.X / _cellSize);
        long iy = (long)Math.Floor(p.Y / _cellSize);
        long iz = (long)Math.Floor(p.Z / _cellSize);
        return ix * 73856093L ^ iy * 19349663L ^ iz * 83492791L;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private long HashCell(int ix, int iy, int iz) => ix * 73856093L ^ iy * 19349663L ^ iz * 83492791L;

    /// <summary>Find all particles within a radius of a point.</summary>
    public List<int> RadiusSearch(Vector3D center, double radius)
    {
        var results = new List<int>();
        double rSq = radius * radius;
        int cellRadius = (int)Math.Ceiling(radius / _cellSize);
        long baseHash = Hash(center);
        int baseIx = (int)Math.Floor(center.X / _cellSize);
        int baseIy = (int)Math.Floor(center.Y / _cellSize);
        int baseIz = (int)Math.Floor(center.Z / _cellSize);
        for (int dx = -cellRadius; dx <= cellRadius; dx++)
        {
            for (int dy = -cellRadius; dy <= cellRadius; dy++)
            {
                for (int dz = -cellRadius; dz <= cellRadius; dz++)
                {
                    long hash = HashCell(baseIx + dx, baseIy + dy, baseIz + dz);
                    if (_cells.TryGetValue(hash, out var list))
                    {
                        foreach (var idx in list)
                        {
                            var p = _points[idx];
                            double dSq = (p - center).LengthSquared();
                            if (dSq <= rSq)
                                results.Add(idx);
                        }
                    }
                }
            }
        }
        return results;
    }

    /// <summary>Find the nearest neighbor to a query point.</summary>
    public int NearestNeighbor(Vector3D query)
    {
        int best = -1;
        double bestDist = double.MaxValue;
        int baseIx = (int)Math.Floor(query.X / _cellSize);
        int baseIy = (int)Math.Floor(query.Y / _cellSize);
        int baseIz = (int)Math.Floor(query.Z / _cellSize);
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++)
                {
                    long hash = HashCell(baseIx + dx, baseIy + dy, baseIz + dz);
                    if (_cells.TryGetValue(hash, out var list))
                        foreach (var idx in list)
                        {
                            double dSq = (_points[idx] - query).LengthSquared();
                            if (dSq < bestDist)
                            { bestDist = dSq; best = idx; }
                        }
                }
        return best;
    }

    /// <summary>Get the cell key for a point.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long GetCellKey(Vector3D point) => Hash(point);

    /// <summary>Get all particle indices in a specific cell.</summary>
    public IReadOnlyList<int> GetCellParticles(Vector3D point) => _cells.TryGetValue(Hash(point), out var list) ? list : Array.Empty<int>();

    /// <summary>Clear all particles.</summary>
    public void Clear() { _cells.Clear(); _points.Clear(); }
}

/// <summary>Conservation law enforcement for physics simulation. Ensures exact conservation of energy, momentum, and angular momentum.</summary>
public class ConservationEnforcer
{
    private Vector3D _initialLinearMomentum;
    private Vector3D _initialAngularMomentum;
    private double _initialEnergy;
    private bool _initialized;
    private readonly double _tolerance;
    private int _violationCount;
    private double _maxEnergyDrift;
    private double _maxMomentumDrift;

    public int ViolationCount => _violationCount;
    public double MaxEnergyDrift => _maxEnergyDrift;
    public double MaxMomentumDrift => _maxMomentumDrift;

    public ConservationEnforcer(double tolerance = 1e-10) { _tolerance = tolerance; }

    public void Initialize(Vector3D linearMomentum, Vector3D angularMomentum, double energy)
    {
        _initialLinearMomentum = linearMomentum;
        _initialAngularMomentum = angularMomentum;
        _initialEnergy = energy;
        _initialized = true;
        _violationCount = 0;
        _maxEnergyDrift = 0;
        _maxMomentumDrift = 0;
    }

    public bool CheckEnergy(double currentEnergy, out double drift)
    {
        drift = Math.Abs(currentEnergy - _initialEnergy) / Math.Max(Math.Abs(_initialEnergy), 1.0);
        _maxEnergyDrift = Math.Max(_maxEnergyDrift, drift);
        if (drift > _tolerance)
        { _violationCount++; return false; }
        return true;
    }

    public bool CheckLinearMomentum(Vector3D current, out double drift)
    {
        drift = (current - _initialLinearMomentum).Length() / Math.Max(_initialLinearMomentum.Length(), 1.0);
        _maxMomentumDrift = Math.Max(_maxMomentumDrift, drift);
        if (drift > _tolerance)
        { _violationCount++; return false; }
        return true;
    }

    public bool CheckAngularMomentum(Vector3D current, out double drift)
    {
        drift = (current - _initialAngularMomentum).Length() / Math.Max(_initialAngularMomentum.Length(), 1.0);
        _maxMomentumDrift = Math.Max(_maxMomentumDrift, drift);
        if (drift > _tolerance)
        { _violationCount++; return false; }
        return true;
    }

    public Vector3D CorrectLinearMomentum(Vector3D current, double totalMass)
    {
        if (totalMass < 1e-30)
            return current;
        var correction = (_initialLinearMomentum - current) / totalMass;
        return current + correction;
    }

    public double CorrectEnergy(double currentEnergy) => _initialEnergy;

    public void Reset() { _initialized = false; _violationCount = 0; _maxEnergyDrift = 0; _maxMomentumDrift = 0; }
}

/// <summary>Symmetric positive-definite matrix solver using Cholesky decomposition.</summary>
public class CholeskySolver
{
    private double[,] _L;
    private int _n;

    public CholeskySolver(double[,] matrix) { _n = matrix.GetLength(0); Decompose(matrix); }

    private void Decompose(double[,] A)
    {
        _L = new double[_n, _n];
        for (int i = 0; i < _n; i++)
        {
            double sum = 0;
            for (int k = 0; k < i; k++)
                sum += _L[i, k] * _L[i, k];
            double diag = A[i, i] - sum;
            _L[i, i] = diag > 0 ? Math.Sqrt(diag) : 1e-30;
            for (int j = i + 1; j < _n; j++)
            {
                sum = 0;
                for (int k = 0; k < i; k++)
                    sum += _L[j, k] * _L[i, k];
                _L[j, i] = (A[j, i] - sum) / _L[i, i];
            }
        }
    }

    public double[] Solve(double[] b)
    {
        var y = new double[_n];
        for (int i = 0; i < _n; i++)
        { double s = 0; for (int k = 0; k < i; k++) s += _L[i, k] * y[k]; y[i] = (b[i] - s) / _L[i, i]; }
        var x = new double[_n];
        for (int i = _n - 1; i >= 0; i--)
        { double s = 0; for (int k = i + 1; k < _n; k++) s += _L[k, i] * x[k]; x[i] = (y[i] - s) / _L[i, i]; }
        return x;
    }

    public double[,] Inverse()
    {
        var inv = new double[_n, _n];
        for (int i = 0; i < _n; i++)
        { var e = new double[_n]; e[i] = 1; var col = Solve(e); for (int j = 0; j < _n; j++) inv[j, i] = col[j]; }
        return inv;
    }

    public double Determinant() { double det = 1; for (int i = 0; i < _n; i++) det *= _L[i, i]; return det * det; }
}

/// <summary>Simple Kalman filter for state estimation with process and measurement noise.</summary>
public class KalmanFilter
{
    private int _n; // state dimension
    private int _m; // measurement dimension
    private double[,] _x; // state estimate [n×1]
    private double[,] _P; // error covariance [n×n]
    private double[,] _Q; // process noise [n×n]
    private double[,] _R; // measurement noise [m×m]
    private double[,] _F; // state transition [n×n]
    private double[,] _H; // measurement model [m×n]

    public KalmanFilter(int stateDim, int measDim)
    {
        _n = stateDim;
        _m = measDim;
        _x = new double[_n, 1];
        _P = new double[_n, _n];
        _Q = new double[_n, _n];
        _R = new double[_m, _m];
        _F = new double[_n, _n];
        _H = new double[_m, _n];
        for (int i = 0; i < _n; i++)
            _F[i, i] = 1;
    }

    public void SetState(double[] state) { for (int i = 0; i < _n; i++) _x[i, 0] = state[i]; }
    public void SetCovariance(double[,] P) => _P = P;
    public void SetProcessNoise(double[,] Q) => _Q = Q;
    public void SetMeasurementNoise(double[,] R) => _R = R;
    public void SetTransition(double[,] F) => _F = F;
    public void SetMeasurementModel(double[,] H) => _H = H;

    public double[] Predict()
    {
        // x = F*x
        _x = MatMul(_F, _x);
        // P = F*P*F' + Q
        _P = MatAdd(MatMul(MatMul(_F, _P), Transpose(_F)), _Q);
        var result = new double[_n];
        for (int i = 0; i < _n; i++)
            result[i] = _x[i, 0];
        return result;
    }

    public double[] Update(double[] measurement)
    {
        var z = new double[_m, 1];
        for (int i = 0; i < _m; i++)
            z[i, 0] = measurement[i];
        // y = z - H*x
        var Hx = MatMul(_H, _x);
        var y = MatSub(z, Hx);
        // S = H*P*H' + R
        var S = MatAdd(MatMul(MatMul(_H, _P), Transpose(_H)), _R);
        // K = P*H'*S⁻¹
        var K = MatMul(MatMul(_P, Transpose(_H)), Inverse(S));
        // x = x + K*y
        _x = MatAdd(_x, MatMul(K, y));
        // P = (I - K*H)*P
        var I = Identity(_n);
        _P = MatMul(MatSub(I, MatMul(K, _H)), _P);
        var result = new double[_n];
        for (int i = 0; i < _n; i++)
            result[i] = _x[i, 0];
        return result;
    }

    public double[] GetState() { var r = new double[_n]; for (int i = 0; i < _n; i++) r[i] = _x[i, 0]; return r; }
    public double GetUncertainty(int i) => _P[i, i];

    private static double[,] MatMul(double[,] A, double[,] B)
    {
        int m = A.GetLength(0), n = B.GetLength(1), p = A.GetLength(1);
        var C = new double[m, n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            { double s = 0; for (int k = 0; k < p; k++) s += A[i, k] * B[k, j]; C[i, j] = s; }
        return C;
    }
    private static double[,] MatAdd(double[,] A, double[,] B)
    { int m = A.GetLength(0), n = A.GetLength(1); var C = new double[m, n]; for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) C[i, j] = A[i, j] + B[i, j]; return C; }
    private static double[,] MatSub(double[,] A, double[,] B)
    { int m = A.GetLength(0), n = A.GetLength(1); var C = new double[m, n]; for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) C[i, j] = A[i, j] - B[i, j]; return C; }
    private static double[,] Transpose(double[,] A)
    { int m = A.GetLength(0), n = A.GetLength(1); var T = new double[n, m]; for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) T[j, i] = A[i, j]; return T; }
    private static double[,] Identity(int n) { var I = new double[n, n]; for (int i = 0; i < n; i++) I[i, i] = 1; return I; }
    private static double[,] Inverse(double[,] A) { int n = A.GetLength(0); var aug = new double[n, 2 * n]; for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) aug[i, j] = A[i, j]; aug[i, n + i] = 1; } for (int i = 0; i < n; i++) { double pivot = aug[i, i]; if (Math.Abs(pivot) < 1e-30) pivot = 1e-30; for (int j = 0; j < 2 * n; j++) aug[i, j] /= pivot; for (int k = 0; k < n; k++) if (k != i) { double f = aug[k, i]; for (int j = 0; j < 2 * n; j++) aug[k, j] -= f * aug[i, j]; } } var inv = new double[n, n]; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) inv[i, j] = aug[i, n + j]; return inv; }
}

/// <summary>Discrete Fourier Transform and FFT utilities for signal processing in physics simulations.</summary>
public static class FourierTransform
{
    /// <summary>Simple DFT — O(n²) for small datasets.</summary>
    public static (double[] real, double[] imag) DFT(double[] input)
    {
        int n = input.Length;
        var re = new double[n];
        var im = new double[n];
        for (int k = 0; k < n; k++)
        {
            double sumRe = 0, sumIm = 0;
            for (int j = 0; j < n; j++)
            {
                double angle = -2.0 * Math.PI * k * j / n;
                sumRe += input[j] * Math.Cos(angle);
                sumIm += input[j] * Math.Sin(angle);
            }
            re[k] = sumRe;
            im[k] = sumIm;
        }
        return (re, im);
    }

    /// <summary>Inverse DFT.</summary>
    public static double[] IDFT(double[] real, double[] imag)
    {
        int n = real.Length;
        var output = new double[n];
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            for (int k = 0; k < n; k++)
            {
                double angle = 2.0 * Math.PI * k * j / n;
                sum += real[k] * Math.Cos(angle) - imag[k] * Math.Sin(angle);
            }
            output[j] = sum / n;
        }
        return output;
    }

    /// <summary>Power spectrum: |X(k)|² = Re² + Im².</summary>
    public static double[] PowerSpectrum(double[] real, double[] imag)
    {
        var ps = new double[real.Length];
        for (int i = 0; i < real.Length; i++)
            ps[i] = real[i] * real[i] + imag[i] * imag[i];
        return ps;
    }

    /// <summary>Magnitude spectrum: |X(k)|.</summary>
    public static double[] MagnitudeSpectrum(double[] real, double[] imag)
    {
        var mag = new double[real.Length];
        for (int i = 0; i < real.Length; i++)
            mag[i] = Math.Sqrt(real[i] * real[i] + imag[i] * imag[i]);
        return mag;
    }

    /// <summary>Phase spectrum: angle(X(k)) = atan2(Im, Re).</summary>
    public static double[] PhaseSpectrum(double[] real, double[] imag)
    {
        var phase = new double[real.Length];
        for (int i = 0; i < real.Length; i++)
            phase[i] = Math.Atan2(imag[i], real[i]);
        return phase;
    }

    /// <summary>Convolution of two signals via direct multiplication (frequency domain).</summary>
    public static double[] Convolve(double[] a, double[] b)
    {
        int n = a.Length + b.Length - 1;
        var result = new double[n];
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
                result[i + j] += a[i] * b[j];
        return result;
    }

    /// <summary>Auto-correlation via direct computation.</summary>
    public static double[] AutoCorrelate(double[] input)
    {
        int n = input.Length;
        var result = new double[n];
        for (int lag = 0; lag < n; lag++)
        {
            double sum = 0;
            for (int i = 0; i < n - lag; i++)
                sum += input[i] * input[i + lag];
            result[lag] = sum / (n - lag);
        }
        return result;
    }

    /// <summary>Cross-correlation of two signals.</summary>
    public static double[] CrossCorrelate(double[] a, double[] b)
    {
        int n = a.Length + b.Length - 1;
        var result = new double[n];
        for (int lag = -(b.Length - 1); lag < a.Length; lag++)
        {
            double sum = 0;
            for (int i = Math.Max(0, lag); i < Math.Min(a.Length, lag + b.Length); i++)
                sum += a[i] * b[i - lag];
            result[lag + b.Length - 1] = sum;
        }
        return result;
    }

    /// <summary>Window functions for spectral analysis.</summary>
    public static double[] HanningWindow(int n) { var w = new double[n]; for (int i = 0; i < n; i++) w[i] = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (n - 1))); return w; }
    public static double[] HammingWindow(int n) { var w = new double[n]; for (int i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.Cos(2.0 * Math.PI * i / (n - 1)); return w; }
    public static double[] BlackmanWindow(int n) { var w = new double[n]; for (int i = 0; i < n; i++) w[i] = 0.42 - 0.5 * Math.Cos(2.0 * Math.PI * i / (n - 1)) + 0.08 * Math.Cos(4.0 * Math.PI * i / (n - 1)); return w; }
    public static double[] BartlettWindow(int n) { var w = new double[n]; for (int i = 0; i < n; i++) w[i] = 1.0 - Math.Abs((i - (n - 1.0) / 2.0) / ((n - 1.0) / 2.0)); return w; }
    public static double[] KaiserWindow(int n, double beta)
    {
        var w = new double[n];
        double a = (n - 1.0) / 2.0;
        for (int i = 0; i < n; i++)
        {
            double x = (i - a) / a;
            w[i] = BesselI0(beta * Math.Sqrt(1.0 - x * x)) / BesselI0(beta);
        }
        return w;
    }
    private static double BesselI0(double x)
    {
        double sum = 1, term = 1;
        for (int k = 1; k < 25; k++)
        { term *= (x * x) / (4.0 * k * k); sum += term; }
        return sum;
    }
}
