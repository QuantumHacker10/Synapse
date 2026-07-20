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
