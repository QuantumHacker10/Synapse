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
