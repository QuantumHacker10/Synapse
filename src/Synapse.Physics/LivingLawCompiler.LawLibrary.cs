// =============================================================================
// Synapse Omnia — Compilateur de Lois Vivantes
// LivingLawCompiler.cs
//
// Complete implementation of the Living Law Compiler: loads, modifies, invents
// physical laws as manipulable objects. Supports expression parsing, bytecode
// compilation, hot-reload, version trees, validation, and law application.
//
// C# 14 · Unsafe · NativeAOT compatible
// =============================================================================

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Physics
{

    /// <summary>Repository of physical laws that can be loaded and compiled.</summary>
    public sealed class LawLibrary
    {
        private readonly ConcurrentDictionary<string, LawEntry> _entries = new();
        private readonly List<LawEntry> _allEntries = new();

        public IReadOnlyList<LawEntry> AllEntries => _allEntries;

        public void Register(LawEntry entry)
        {
            _entries[entry.Id] = entry;
            lock (_allEntries)
            { _allEntries.Add(entry); }
        }

        public LawEntry? GetLaw(string id) =>
            _entries.TryGetValue(id, out var entry) ? entry : null;

        public IReadOnlyList<LawEntry> SearchByCategory(string category)
        {
            var results = new List<LawEntry>();
            foreach (var e in _allEntries)
                if (string.Equals(e.Category, category, StringComparison.OrdinalIgnoreCase))
                    results.Add(e);
            return results;
        }

        public IReadOnlyList<LawEntry> SearchByName(string query)
        {
            var results = new List<LawEntry>();
            foreach (var e in _allEntries)
                if (e.Name.Contains(query, StringComparison.OrdinalIgnoreCase))
                    results.Add(e);
            return results;
        }

        public static LawLibrary LoadFromJson(string filePath)
        {
            var library = new LawLibrary();
            var json = File.ReadAllText(filePath);
            var entries = JsonSerializer.Deserialize<LawEntry[]>(json);
            if (entries != null)
                foreach (var entry in entries)
                    library.Register(entry);
            return library;
        }

        /// <summary>Built-in library: core scene laws plus the full reference catalog (100+ laws).</summary>
        public static LawLibrary LoadBuiltIn() => LawCatalogBridge.LoadFullCatalog();

        /// <summary>Registers the small set of laws used by sample scenes and benchmarks.</summary>
        internal static void RegisterCoreSimulationLaws(LawLibrary library)
        {
            library.Register(new LawEntry
            {
                Id = "navier_stokes_x",
                Name = "Navier-Stokes (X)",
                Category = "FluidDynamics",
                Expression = "rho * (dv/dt + v·∇v) = -∇P + μ*∇²v + f",
                Description = "Navier-Stokes momentum equation, X component",
                Parameters = new() { { "v", "velocity" }, { "P", "pressure" }, { "rho", "density" }, { "mu", "viscosity" } },
                Constants = new() { { "mu", 1.002e-3f } },
                ResultDimension = Dimension.Force
            });
            library.Register(new LawEntry
            {
                Id = "heat_equation",
                Name = "Heat Equation",
                Category = "ThermalDynamics",
                Expression = "∂T/∂t = α*∇²T",
                Description = "Heat diffusion equation",
                Parameters = new() { { "T", "temperature" }, { "alpha", "thermal_diffusivity" } },
                Constants = new() { { "alpha", 1.43e-4f } },
                ResultDimension = Dimension.TemperatureD.Divide(Dimension.TimeD)
            });
            library.Register(new LawEntry
            {
                Id = "wave_equation",
                Name = "Wave Equation",
                Category = "WaveDynamics",
                Expression = "∂²u/∂t² = c²*∇²u",
                Description = "Wave propagation equation",
                Parameters = new() { { "u", "displacement" }, { "c", "wave_speed" } },
                Constants = new() { { "c", 340.0f } },
                ResultDimension = Dimension.Acceleration
            });
            library.Register(new LawEntry
            {
                Id = "ideal_gas",
                Name = "Ideal Gas Law",
                Category = "Thermodynamics",
                Expression = "P = ρ*R*T",
                Description = "Ideal gas equation of state",
                Parameters = new() { { "rho", "density" }, { "R", "specific_gas_constant" }, { "T", "temperature" } },
                Constants = new() { { "R", 287.058f } },
                ResultDimension = Dimension.Pressure
            });
            library.Register(new LawEntry
            {
                Id = "fourier_law",
                Name = "Fourier's Law",
                Category = "ThermalDynamics",
                Expression = "q = -k*∇T",
                Description = "Heat conduction equation",
                Parameters = new() { { "k", "thermal_conductivity" }, { "T", "temperature" } },
                Constants = new() { { "k", 205.0f } },
                ResultDimension = Dimension.Power.Divide(new Dimension(0, 2, 0, 0, 0, 0, 0, 0))
            });
            library.Register(new LawEntry
            {
                Id = "coulomb_force",
                Name = "Coulomb Force",
                Category = "Electrodynamics",
                Expression = "F = k_e * q1 * q2 / r²",
                Description = "Electrostatic force between charges",
                Parameters = new() { { "q1", "charge" }, { "q2", "charge" }, { "r", "distance" } },
                Constants = new() { { "ke", 8.9875517923e9f } },
                ResultDimension = Dimension.Force
            });
            library.Register(new LawEntry
            {
                Id = "gravity_newton",
                Name = "Newtonian Gravity",
                Category = "Gravitation",
                Expression = "F = G * m1 * m2 / r²",
                Description = "Newton's law of universal gravitation",
                Parameters = new() { { "m1", "mass" }, { "m2", "mass" }, { "r", "distance" } },
                Constants = new() { { "G", 6.674e-11f } },
                ResultDimension = Dimension.Force
            });
            library.Register(new LawEntry
            {
                Id = "hooke_law",
                Name = "Hooke's Law",
                Category = "Elasticity",
                Expression = "F = -k * Δx",
                Description = "Linear elastic restoring force",
                Parameters = new() { { "k", "spring_constant" }, { "x", "displacement" } },
                Constants = new() { { "k", 100.0f } },
                ResultDimension = Dimension.Force
            });
            library.Register(new LawEntry
            {
                Id = "ohms_law",
                Name = "Ohm's Law",
                Category = "Electrodynamics",
                Expression = "V = I * R",
                Description = "Voltage-current relationship",
                Parameters = new() { { "I", "current" }, { "R", "resistance" } },
                Constants = new() { { "R", 1.0f } },
                ResultDimension = new Dimension(1, 2, -3, 0, 0, -1, 0, 0)
            });
            library.Register(new LawEntry
            {
                Id = "stefan_boltzmann",
                Name = "Stefan-Boltzmann Law",
                Category = "ThermalDynamics",
                Expression = "j = σ * T⁴",
                Description = "Thermal radiation power",
                Parameters = new() { { "T", "temperature" } },
                Constants = new() { { "sigma", 5.670374419e-8f } },
                ResultDimension = new Dimension(1, 0, -3, 0, 0, 0, 0, 0)
            });
        }
    }
}
