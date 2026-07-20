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


//  SECTION 6 — Additional Extended Laws (reaching 250KB+)

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
