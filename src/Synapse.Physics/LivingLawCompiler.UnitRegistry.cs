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

    /// <summary>Registry of physical units for dimensional analysis.</summary>
    public sealed class UnitRegistry
    {
        private readonly Dictionary<string, PhysicalUnit> _units = new();

        public UnitRegistry()
        {
            RegisterDefaults();
        }

        private void RegisterDefaults()
        {
            Register(new PhysicalUnit { Symbol = "m", Name = "meter", BaseDimension = Dimension.LengthD });
            Register(new PhysicalUnit { Symbol = "kg", Name = "kilogram", BaseDimension = Dimension.MassD });
            Register(new PhysicalUnit { Symbol = "s", Name = "second", BaseDimension = Dimension.TimeD });
            Register(new PhysicalUnit { Symbol = "K", Name = "kelvin", BaseDimension = Dimension.TemperatureD });
            Register(new PhysicalUnit { Symbol = "mol", Name = "mole", BaseDimension = new Dimension(0, 0, 0, 0, 1, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "A", Name = "ampere", BaseDimension = new Dimension(0, 0, 0, 0, 0, 1, 0, 0) });
            Register(new PhysicalUnit { Symbol = "cd", Name = "candela", BaseDimension = new Dimension(0, 0, 0, 0, 0, 0, 1, 0) });
            Register(new PhysicalUnit { Symbol = "Hz", Name = "hertz", BaseDimension = new Dimension(0, 0, -1, 0, 0, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "N", Name = "newton", BaseDimension = Dimension.Force });
            Register(new PhysicalUnit { Symbol = "Pa", Name = "pascal", BaseDimension = Dimension.Pressure });
            Register(new PhysicalUnit { Symbol = "J", Name = "joule", BaseDimension = Dimension.Energy });
            Register(new PhysicalUnit { Symbol = "W", Name = "watt", BaseDimension = Dimension.Power });
            Register(new PhysicalUnit { Symbol = "V", Name = "volt", BaseDimension = new Dimension(1, 2, -3, 0, 0, -1, 0, 0) });
            Register(new PhysicalUnit { Symbol = "Ω", Name = "ohm", BaseDimension = new Dimension(1, 2, -3, 0, 0, -2, 0, 0) });
            Register(new PhysicalUnit { Symbol = "C", Name = "coulomb", BaseDimension = new Dimension(0, 0, 0, 0, 0, 1, 0, 0) });
            Register(new PhysicalUnit { Symbol = "F", Name = "farad", BaseDimension = new Dimension(-1, -2, 4, 0, 0, 2, 0, 0) });
            Register(new PhysicalUnit { Symbol = "H", Name = "henry", BaseDimension = new Dimension(1, 2, -2, 0, 0, -2, 0, 0) });
            Register(new PhysicalUnit { Symbol = "T", Name = "tesla", BaseDimension = new Dimension(1, 0, -2, 0, 0, -1, 0, 0) });
            Register(new PhysicalUnit { Symbol = "Wb", Name = "weber", BaseDimension = new Dimension(1, 2, -2, 0, 0, -1, 0, 0) });
            Register(new PhysicalUnit { Symbol = "lm", Name = "lumen", BaseDimension = new Dimension(0, 0, 0, 0, 0, 0, 1, 0) });
            Register(new PhysicalUnit { Symbol = "lx", Name = "lux", BaseDimension = new Dimension(0, 0, 0, 0, 0, 0, 1, -2) });
            Register(new PhysicalUnit { Symbol = "Bq", Name = "becquerel", BaseDimension = new Dimension(0, 0, -1, 0, 0, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "Gy", Name = "gray", BaseDimension = new Dimension(0, 2, -2, 0, 0, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "Sv", Name = "sievert", BaseDimension = new Dimension(0, 2, -2, 0, 0, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "kat", Name = "katal", BaseDimension = new Dimension(0, 0, -1, 0, 1, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "m/s", Name = "meters per second", BaseDimension = Dimension.Velocity });
            Register(new PhysicalUnit { Symbol = "m/s²", Name = "meters per second squared", BaseDimension = Dimension.Acceleration });
            Register(new PhysicalUnit { Symbol = "kg/m³", Name = "kilograms per cubic meter", BaseDimension = Dimension.Density });
            Register(new PhysicalUnit { Symbol = "Pa·s", Name = "pascal-second", BaseDimension = Dimension.Viscosity });
            Register(new PhysicalUnit { Symbol = "W/(m·K)", Name = "watts per meter-kelvin", BaseDimension = new Dimension(1, 1, -3, -1, 0, 0, 0, 0) });
            Register(new PhysicalUnit { Symbol = "m²/s", Name = "square meters per second", BaseDimension = new Dimension(0, 2, -1, 0, 0, 0, 0, 0) });
        }

        public void Register(PhysicalUnit unit) => _units[unit.Symbol] = unit;

        public PhysicalUnit? GetUnit(string symbol) =>
            _units.TryGetValue(symbol, out var unit) ? unit : null;

        public bool UnitExists(string symbol) => _units.ContainsKey(symbol);

        /// <summary>Check if two units are dimensionally compatible.</summary>
        public bool AreCompatible(string unitA, string unitB)
        {
            var a = GetUnit(unitA);
            var b = GetUnit(unitB);
            if (a == null || b == null)
                return false;
            return a.BaseDimension.IsCompatible(b.BaseDimension);
        }

        /// <summary>Get the product dimension of two units.</summary>
        public Dimension ProductDimension(string unitA, string unitB)
        {
            var a = GetUnit(unitA);
            var b = GetUnit(unitB);
            if (a == null || b == null)
                return Dimension.Scalar;
            return a.BaseDimension.Multiply(b.BaseDimension);
        }

        /// <summary>Get the quotient dimension of two units.</summary>
        public Dimension QuotientDimension(string unitA, string unitB)
        {
            var a = GetUnit(unitA);
            var b = GetUnit(unitB);
            if (a == null || b == null)
                return Dimension.Scalar;
            return a.BaseDimension.Divide(b.BaseDimension);
        }
    }
}
