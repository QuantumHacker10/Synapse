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
