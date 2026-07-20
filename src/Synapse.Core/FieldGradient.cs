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
