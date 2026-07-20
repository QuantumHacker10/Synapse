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

// SECTION 3.2: SYMMETRIC3X3 — MATRICE 3x3 SYMETRIQUE
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Matrice 3x3 symetrique pour le tenseur de contrainte et la deformation.
/// Stockee en format compact (6 composantes uniques : XX, YY, ZZ, XY, XZ, YZ).
/// Utilisee pour les tenseurs de stress (Cauchy, Piola-Kirchhoff),
/// strain (Green-Lagrange, Almansi), et d'inertie. Les operations SIMD
/// sont optimisees pour les operateurs differentiels.
///
/// L'invariant J2 (partie deviatorique) est calcule analytiquement pour le
/// critere de plasticite de von Mises. L'inversion est explicite.
///
/// MEMORY LAYOUT: 64 bytes (6 doubles + 2 padding pour alignement AVX).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64, Pack = 32)]
[DebuggerDisplay("Sym3x3({XX:F3}, {YY:F3}, {ZZ:F3}, {XY:F3}, {XZ:F3}, {YZ:F3})")]
public struct Symmetric3x3 : IEquatable<Symmetric3x3>
{
    [FieldOffset(0)] public double XX;
    [FieldOffset(8)] public double YY;
    [FieldOffset(16)] public double ZZ;
    [FieldOffset(24)] public double XY;
    [FieldOffset(32)] public double XZ;
    [FieldOffset(40)] public double YZ;
    [FieldOffset(48)] public double _pad0;
    [FieldOffset(56)] public double _pad1;

    /// <summary>Constructeur avec les 6 composantes independantes d'une matrice symetrique.</summary>
    /// <param name="xx">Composante (1,1) — contrainte normale X.</param>
    /// <param name="yy">Composante (2,2) — contrainte normale Y.</param>
    /// <param name="zz">Composante (3,3) — contrainte normale Z.</param>
    /// <param name="xy">Composante (1,2) = (2,1) — contrainte de cisaillement XY.</param>
    /// <param name="xz">Composante (1,3) = (3,1) — contrainte de cisaillement XZ.</param>
    /// <param name="yz">Composante (2,3) = (3,2) — contrainte de cisaillement YZ.</param>
    public Symmetric3x3(double xx, double yy, double zz, double xy, double xz, double yz)
    { XX = xx; YY = yy; ZZ = zz; XY = xy; XZ = xz; YZ = yz; _pad0 = 0; _pad1 = 0; }

    /// <summary>Matrice symetrique nulle (toutes composantes a zero).</summary>
    public static readonly Symmetric3x3 Zero = default;
    /// <summary>Matrice identite : diag(1,1,1). Tenseur de contrainte sans contrainte.</summary>
    public static readonly Symmetric3x3 Identity = new(1, 1, 1, 0, 0, 0);

    /// <summary>Premier invariant : Tr(sigma) = XX + YY + ZZ. Pression hydrostatique = -Tr/3.</summary>
    public readonly double Trace => XX + YY + ZZ;

    /// <summary>Determinant de la matrice symetrique 3x3 : det = XX*(YY*ZZ - YZ^2) - XY*(XY*ZZ - YZ*XZ) + XZ*(XY*YZ - YY*XZ).</summary>
    public readonly double Determinant =>
        XX * (YY * ZZ - YZ * YZ) - XY * (XY * ZZ - YZ * XZ) + XZ * (XY * YZ - YY * XZ);

    /// <summary>Partie deviatorique : sigma' = sigma - (Tr(sigma)/3)*I. Retire la pression hydrostatique.</summary>
    public readonly Symmetric3x3 Deviatoric { get { double m = Trace / 3.0; return new(XX - m, YY - m, ZZ - m, XY, XZ, YZ); } }

    /// <summary>Deuxieme invariant deviatorique J2 : mesure de l'intensite de cisaillement.</summary>
    /// <remarks>J2 = 0.5*(s11^2 + s22^2 + s33^2) + s12^2 + s13^2 + s23^2, ou s = deviatoric.</remarks>
    public readonly double J2 { get { var d = Deviatoric; return 0.5 * (d.XX * d.XX + d.YY * d.YY + d.ZZ * d.ZZ) + d.XY * d.XY + d.XZ * d.XZ + d.YZ * d.YZ; } }

    /// <summary>Deuxieme invariant complet I2 = XX*YY + YY*ZZ + ZZ*XX - XY^2 - XZ^2 - YZ^2.</summary>
    public readonly double I2 => XX * YY + YY * ZZ + ZZ * XX - XY * XY - XZ * XZ - YZ * YZ;

    /// <summary>Troisieme invariant I3 = det(sigma).</summary>
    public readonly double I3 => Determinant;

    /// <summary>Contrainte equivalente de von Mises : sigma_vM = sqrt(3*J2). Critere de plasticite.</summary>
    /// <remarks>Si sigma_vM depasse la limite d'elasticite (YieldStrength), le materiau plastifie.</remarks>
    public readonly double VonMises => Math.Sqrt(3.0 * J2);

    /// <summary>Norme de Frobenius : ||sigma||_F = sqrt(Sigma sigma_ij^2). Mesure globale de l'intensite.</summary>
    public readonly double FrobeniusNorm => Math.Sqrt(XX * XX + YY * YY + ZZ * ZZ + 2.0 * (XY * XY + XZ * XZ + YZ * YZ));

    /// <summary>Energie de distorsion : U_d = J2. Energie associee a la deformation de cisaillement.</summary>
    public readonly double DistortionEnergy => J2;

    /// <summary>Energie de dilatation : U_v = Tr(sigma)^2 / 18. Energie de changement de volume.</summary>
    public readonly double DilatationEnergy => Trace * Trace / 18.0;

    /// <summary>Energie de deformation totale : U = U_d + U_v.</summary>
    public readonly double StrainEnergy => DistortionEnergy + DilatationEnergy;

    /// <summary>Premiere direction principale (approximation par Jacobi).</summary>
    public readonly Vector3D PrincipalDirection1 { get { MaxEigenvalues(out var v1, out _, out _); return new Vector3D(v1, 0, 0); } }
    /// <summary>Deuxieme direction principale.</summary>
    public readonly Vector3D PrincipalDirection2 { get { MaxEigenvalues(out _, out var v2, out _); return new Vector3D(0, v2, 0); } }
    /// <summary>Troisieme direction principale.</summary>
    public readonly Vector3D PrincipalDirection3 { get { MaxEigenvalues(out _, out _, out var v3); return new Vector3D(0, 0, v3); } }

    /// <summary>Valeurs propres principales (tries par ordre decroissant).</summary>
    public readonly void MaxEigenvalues(out double v1, out double v2, out double v3)
    {
        Symmetric3x3 m = this;
        Vector3D d1 = UnitX, d2 = UnitY, d3 = UnitZ;
        for (int iter = 0; iter < 50; iter++)
        {
            double theta = 0.5 * Math.Atan2(2.0 * m.XY, m.XX - m.YY);
            double c = Math.Cos(theta), s = Math.Sin(theta);
            double c2 = c * c, s2 = s * s, cs = c * s;
            m = new Symmetric3x3(
                c2 * m.XX + 2 * cs * m.XY + s2 * m.YY,
                s2 * m.XX - 2 * cs * m.XY + c2 * m.YY, m.ZZ,
                cs * (m.YY - m.XX) + (c2 - s2) * m.XY,
                c * m.XZ + s * m.YZ, -s * m.XZ + c * m.YZ);
            d1 = new Vector3D(c * d1.X + s * d1.Y, -s * d1.X + c * d1.Y, d1.Z);
            d2 = new Vector3D(c * d2.X + s * d2.Y, -s * d2.X + c * d2.Y, d2.Z);
        }
        if (m.XX < m.YY)
        { (m.XX, m.YY) = (m.YY, m.XX); (d1, d2) = (d2, d1); }
        if (m.XX < m.ZZ)
        { (m.XX, m.ZZ) = (m.ZZ, m.XX); (d1, d3) = (d3, d1); }
        if (m.YY < m.ZZ)
        { (m.YY, m.ZZ) = (m.ZZ, m.YY); (d2, d3) = (d3, d2); }
        v1 = m.XX;
        v2 = m.YY;
        v3 = m.ZZ;
    }

    private static readonly Vector3D UnitX = Vector3D.UnitX, UnitY = Vector3D.UnitY, UnitZ = Vector3D.UnitZ;

    /// <summary>Produit matrice-vecteur : sigma.v. Calcule la force sur une surface.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Multiply(Vector3D v) =>
        new(XX * v.X + XY * v.Y + XZ * v.Z, XY * v.X + YY * v.Y + YZ * v.Z, XZ * v.X + YZ * v.Y + ZZ * v.Z);

    /// <summary>Double contraction : v.sigma.v. Energie de deformation sur un vecteur.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double DoubleContract(Vector3D v) => Vector3D.Dot(v, Multiply(v));

    /// <summary>Matrice symetrique de Cauchy-Green : C = F^T * F.</summary>
    public static Symmetric3x3 FromCauchyGreen(Matrix4x4D F)
    {
        Matrix4x4D ft = F.Transpose();
        Matrix4x4D c = ft * F;
        return new Symmetric3x3(c.M11, c.M22, c.M33, c.M12, c.M13, c.M23);
    }

    /// <summary>Contrainte de Green-Lagrange : E = 0.5*(C - I). Deformation finie.</summary>
    public readonly Symmetric3x3 GreenLagrangeStrain() =>
        new((XX - 1) * 0.5, (YY - 1) * 0.5, (ZZ - 1) * 0.5, XY * 0.5, XZ * 0.5, YZ * 0.5);

    /// <summary>Inversion de la matrice symetrique 3x3 (methode de Cramer).</summary>
    public readonly Symmetric3x3 Inverse()
    {
        double det = Determinant;
        if (Math.Abs(det) < 1e-30)
            return Zero;
        double inv = 1.0 / det;
        return new Symmetric3x3(
            (YY * ZZ - YZ * YZ) * inv, (XX * ZZ - XZ * XZ) * inv, (XX * YY - XY * XY) * inv,
            (XZ * YZ - XY * ZZ) * inv, (XY * YZ - XZ * YY) * inv, (XY * XZ - XX * YZ) * inv);
    }

    /// <summary>Rotation par une matrice de rotation : R * sigma * R^T.</summary>
    public readonly Symmetric3x3 RotateBy(Matrix4x4D R)
    {
        double r00 = R.M11, r01 = R.M12, r02 = R.M13, r10 = R.M21, r11 = R.M22, r12 = R.M23, r20 = R.M31, r21 = R.M32, r22 = R.M33;
        double nxx = r00 * (r00 * XX + r01 * XY + r02 * XZ) + r01 * (r00 * XY + r01 * YY + r02 * YZ) + r02 * (r00 * XZ + r01 * YZ + r02 * ZZ);
        double nyy = r10 * (r10 * XX + r11 * XY + r12 * XZ) + r11 * (r10 * XY + r11 * YY + r12 * YZ) + r12 * (r10 * XZ + r11 * YZ + r12 * ZZ);
        double nzz = r20 * (r20 * XX + r21 * XY + r22 * XZ) + r21 * (r20 * XY + r21 * YY + r22 * YZ) + r22 * (r20 * XZ + r21 * YZ + r22 * ZZ);
        double nxy = r00 * (r10 * XX + r11 * XY + r12 * XZ) + r01 * (r10 * XY + r11 * YY + r12 * YZ) + r02 * (r10 * XZ + r11 * YZ + r12 * ZZ);
        double nxz = r00 * (r20 * XX + r21 * XY + r22 * XZ) + r01 * (r20 * XY + r21 * YY + r22 * YZ) + r02 * (r20 * XZ + r21 * YZ + r22 * ZZ);
        double nyz = r10 * (r20 * XX + r21 * XY + r22 * XZ) + r11 * (r20 * XY + r21 * YY + r22 * YZ) + r12 * (r20 * XZ + r21 * YZ + r22 * ZZ);
        return new Symmetric3x3(nxx, nyy, nzz, nxy, nxz, nyz);
    }

    /// <summary>Operateurs arithmetiques.</summary>
    public static Symmetric3x3 operator +(Symmetric3x3 a, Symmetric3x3 b) => new(a.XX + b.XX, a.YY + b.YY, a.ZZ + b.ZZ, a.XY + b.XY, a.XZ + b.XZ, a.YZ + b.YZ);
    public static Symmetric3x3 operator -(Symmetric3x3 a, Symmetric3x3 b) => new(a.XX - b.XX, a.YY - b.YY, a.ZZ - b.ZZ, a.XY - b.XY, a.XZ - b.XZ, a.YZ - b.YZ);
    public static Symmetric3x3 operator *(Symmetric3x3 m, double s) => new(m.XX * s, m.YY * s, m.ZZ * s, m.XY * s, m.XZ * s, m.YZ * s);
    public static Symmetric3x3 operator *(double s, Symmetric3x3 m) => m * s;
    public static Symmetric3x3 operator +(Symmetric3x3 m, double s) => new(m.XX + s, m.YY + s, m.ZZ + s, m.XY, m.XZ, m.YZ);
    public static Symmetric3x3 operator -(Symmetric3x3 m, double s) => new(m.XX - s, m.YY - s, m.ZZ - s, m.XY, m.XZ, m.YZ);

    /// <summary>Interpolation lineaire entre deux tenseurs symetriques.</summary>
    public static Symmetric3x3 Lerp(Symmetric3x3 a, Symmetric3x3 b, double t)
    {
        double u = 1.0 - t;
        return new Symmetric3x3(a.XX * u + b.XX * t, a.YY * u + b.YY * t, a.ZZ * u + b.ZZ * t, a.XY * u + b.XY * t, a.XZ * u + b.XZ * t, a.YZ * u + b.YZ * t);
    }

    /// <summary>Norme L2 des composantes du tenseur.</summary>
    public readonly double Norm => FrobeniusNorm;

    /// <summary>Determinant signe.</summary>
    public readonly double SignedDeterminant => Determinant;

    /// <summary>Teste si la matrice est definie positive (toutes valeurs propres > 0).</summary>
    public readonly bool IsPositiveDefinite
    {
        get
        {
            MaxEigenvalues(out double v1, out double v2, out double v3);
            return v1 > 0 && v2 > 0 && v3 > 0;
        }
    }

    /// <summary>Teste si la matrice est semi-definie positive.</summary>
    public readonly bool IsPositiveSemiDefinite
    {
        get
        {
            MaxEigenvalues(out double v1, out double v2, out double v3);
            return v1 >= 0 && v2 >= 0 && v3 >= 0;
        }
    }

    /// <summary>Condition number : rapport max/min des valeurs propres.</summary>
    public readonly double ConditionNumber
    {
        get
        {
            MaxEigenvalues(out double v1, out double v2, out double v3);
            double min = Math.Min(Math.Min(Math.Abs(v1), Math.Abs(v2)), Math.Abs(v3));
            double max = Math.Max(Math.Max(Math.Abs(v1), Math.Abs(v2)), Math.Abs(v3));
            return min > 1e-30 ? max / min : double.MaxValue;
        }
    }

    public readonly bool Equals(Symmetric3x3 o) => Math.Abs(XX - o.XX) < 1e-10 && Math.Abs(YY - o.YY) < 1e-10 && Math.Abs(ZZ - o.ZZ) < 1e-10 && Math.Abs(XY - o.XY) < 1e-10 && Math.Abs(XZ - o.XZ) < 1e-10 && Math.Abs(YZ - o.YZ) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is Symmetric3x3 o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(XX, YY, ZZ, XY, XZ, YZ);
    public static bool operator ==(Symmetric3x3 a, Symmetric3x3 b) => a.Equals(b);
    public static bool operator !=(Symmetric3x3 a, Symmetric3x3 b) => !a.Equals(b);
    public override readonly string ToString() => $"[{XX:F3} {XY:F3} {XZ:F3} | {XY:F3} {YY:F3} {YZ:F3} | {XZ:F3} {YZ:F3} {ZZ:F3}]";
}

// ═══════════════════════════════════════════════════════════════════════════════
