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
