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
