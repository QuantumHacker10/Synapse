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

    // =========================================================================
    // LawPhysicsUtils — physics-specific utility functions
    // =========================================================================

    /// <summary>Physics-specific utility functions for law computations.</summary>
    public static class LawPhysicsUtils
    {
        /// <summary>Compute the speed of sound for an ideal gas.</summary>
        public static float SpeedOfSound(float temperature, float gamma = 1.4f, float R = 287.058f)
        {
            return MathF.Sqrt(gamma * R * temperature);
        }

        /// <summary>Compute the Mach number.</summary>
        public static float MachNumber(float velocity, float soundSpeed)
        {
            return MathF.Abs(soundSpeed) > float.Epsilon ? MathF.Abs(velocity) / soundSpeed : 0f;
        }

        /// <summary>Compute the Reynolds number.</summary>
        public static float ReynoldsNumber(float density, float velocity, float length, float viscosity)
        {
            return MathF.Abs(viscosity) > float.Epsilon ? density * MathF.Abs(velocity) * length / viscosity : float.MaxValue;
        }

        /// <summary>Compute the Prandtl number.</summary>
        public static float PrandtlNumber(float viscosity, float specificHeat, float thermalConductivity)
        {
            return MathF.Abs(thermalConductivity) > float.Epsilon ? viscosity * specificHeat / thermalConductivity : 0f;
        }

        /// <summary>Compute the Nusselt number from Rayleigh and Prandtl numbers.</summary>
        public static float NusseltNumber(float rayleigh, float prandtl, string correlation = "churchill")
        {
            return correlation.ToLowerInvariant() switch
            {
                "churchill" => 0.68f + 0.670f * MathF.Pow(rayleigh * prandtl, 0.25f) /
                    MathF.Pow(1f + MathF.Pow(0.492f / prandtl, 9f / 16f), 4f / 9f),
                "rayleigh_benard" => MathF.Pow(rayleigh * prandtl, 0.25f) * 0.069f,
                _ => MathF.Pow(rayleigh, 0.25f)
            };
        }

        /// <summary>Compute the Biot number.</summary>
        public static float BiotNumber(float heatTransferCoeff, float length, float thermalConductivity)
        {
            return MathF.Abs(thermalConductivity) > float.Epsilon ? heatTransferCoeff * length / thermalConductivity : float.MaxValue;
        }

        /// <summary>Compute the Fourier number.</summary>
        public static float FourierNumber(float diffusivity, float time, float length)
        {
            return MathF.Abs(length * length) > float.Epsilon ? diffusivity * time / (length * length) : 0f;
        }

        /// <summary>Compute the Peclet number.</summary>
        public static float PecletNumber(float velocity, float length, float diffusivity)
        {
            return MathF.Abs(diffusivity) > float.Epsilon ? velocity * length / diffusivity : float.MaxValue;
        }

        /// <summary>Compute the Grashof number.</summary>
        public static float GrashofNumber(float beta, float deltaT, float length, float gravity, float viscosity, float kinematicViscosity)
        {
            return MathF.Abs(kinematicViscosity * kinematicViscosity) > float.Epsilon
                ? beta * deltaT * length * length * length * gravity / (kinematicViscosity * kinematicViscosity)
                : 0f;
        }

        /// <summary>Compute the Rayleigh number.</summary>
        public static float RayleighNumber(float grashof, float prandtl) => grashof * prandtl;

        /// <summary>Compute the Strouhal number.</summary>
        public static float StrouhalNumber(float frequency, float length, float velocity)
        {
            return MathF.Abs(velocity) > float.Epsilon ? frequency * length / velocity : 0f;
        }

        /// <summary>Compute the Froude number.</summary>
        public static float FroudeNumber(float velocity, float gravity, float length)
        {
            return MathF.Abs(gravity * length) > float.Epsilon ? velocity / MathF.Sqrt(gravity * length) : float.MaxValue;
        }

        /// <summary>Compute the Weber number.</summary>
        public static float WeberNumber(float density, float velocity, float length, float surfaceTension)
        {
            return MathF.Abs(surfaceTension) > float.Epsilon ? density * velocity * velocity * length / surfaceTension : float.MaxValue;
        }

        /// <summary>Compute the Knudsen number.</summary>
        public static float KnudsenNumber(float meanFreePath, float characteristicLength)
        {
            return MathF.Abs(characteristicLength) > float.Epsilon ? meanFreePath / characteristicLength : float.MaxValue;
        }

        /// <summary>Compute the Mach cone half-angle.</summary>
        public static float MachConeAngle(float machNumber)
        {
            return machNumber > 1f ? MathF.Asin(1f / machNumber) : MathF.PI / 2f;
        }

        /// <summary>Compute the Doppler shift for a moving source.</summary>
        public static float DopplerShift(float sourceFreq, float sourceVelocity, float observerVelocity, float soundSpeed)
        {
            float denominator = soundSpeed - sourceVelocity;
            return MathF.Abs(denominator) > float.Epsilon
                ? sourceFreq * (soundSpeed + observerVelocity) / denominator
                : sourceFreq;
        }

        /// <summary>Compute the adiabatic temperature lapse rate.</summary>
        public static float AdiabaticLapseRate(float gravity, float specificHeat) =>
            MathF.Abs(specificHeat) > float.Epsilon ? gravity / specificHeat : 0f;

        /// <summary>Compute the saturation vapor pressure (Magnus formula).</summary>
        public static float SaturationVaporPressure(float temperatureCelsius)
        {
            return 610.78f * MathF.Exp(17.27f * temperatureCelsius / (temperatureCelsius + 237.3f));
        }

        /// <summary>Compute the heat transfer rate using Newton's law of cooling.</summary>
        public static float NewtonsCooling(float surfaceTemp, float fluidTemp, float heatTransferCoeff, float area)
        {
            return heatTransferCoeff * area * (surfaceTemp - fluidTemp);
        }

        /// <summary>Compute the Stefan-Boltzmann radiative heat flux.</summary>
        public static float StefanBoltzmannFlux(float temperature, float emissivity = 1f)
        {
            const float sigma = 5.670374419e-8f;
            return emissivity * sigma * MathF.Pow(temperature, 4);
        }

        /// <summary>Compute the Planck distribution at a given wavelength and temperature.</summary>
        public static float PlanckDistribution(float wavelength, float temperature)
        {
            const float h = 6.62607015e-34f;
            const float c = 299792458f;
            const float kB = 1.380649e-23f;
            float numerator = 2f * h * c * c / MathF.Pow(wavelength, 5);
            float exponent = h * c / (wavelength * kB * temperature);
            if (exponent > 100f)
                return 0f;
            return numerator / (MathF.Exp(exponent) - 1f);
        }

        /// <summary>Compute the gravitational potential energy.</summary>
        public static float GravitationalPotentialEnergy(float mass, float height, float g = 9.81f) => mass * g * height;

        /// <summary>Compute the kinetic energy of a particle.</summary>
        public static float KineticEnergy(float mass, float velocity) => 0.5f * mass * velocity * velocity;

        /// <summary>Compute the orbital velocity for a circular orbit.</summary>
        public static float OrbitalVelocity(float centralMass, float radius, float G = 6.674e-11f)
        {
            return MathF.Abs(radius) > float.Epsilon ? MathF.Sqrt(G * centralMass / radius) : 0f;
        }

        /// <summary>Compute the escape velocity.</summary>
        public static float EscapeVelocity(float mass, float radius, float G = 6.674e-11f)
        {
            return MathF.Abs(radius) > float.Epsilon ? MathF.Sqrt(2f * G * mass / radius) : 0f;
        }

        /// <summary>Compute the Schwarzschild radius.</summary>
        public static float SchwarzschildRadius(float mass, float c = 299792458f, float G = 6.674e-11f)
        {
            return 2f * G * mass / (c * c);
        }
    }
}
