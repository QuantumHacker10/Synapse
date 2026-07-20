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

// SECTION 3: VECTOR3D — STRUCTURE VECTORIELLE 3D DOUBLE PRECISION
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Structure vectorielle 3D double precision. Alignee sur 32 bytes pour les
/// operations AVX-2 SIMD. Supporte toutes les operations vectorielles usuelles
/// en physique : produits scalaires et vectoriels, distances, interpolations,
/// rotations, conversions de coordonnees, primitives geometriques, et plus.
///
/// Chaque methode est marquee [AggressiveInlining] pour eliminer le cout
/// d'appel de fonction dans les boucles de simulation hot-path.
///
/// MEMORY LAYOUT: 32 bytes (3 doubles + 1 padding).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 32, Pack = 32)]
[DebuggerDisplay("({X:F4}, {Y:F4}, {Z:F4})")]
public struct Vector3D : IEquatable<Vector3D>
{
    [FieldOffset(0)] public double X;
    [FieldOffset(8)] public double Y;
    [FieldOffset(16)] public double Z;
    [FieldOffset(24)] public double _pad;

    /// <summary>Constructeur principal avec composantes x, y, z.</summary>
    /// <param name="x">Composante X (abscisse).</param>
    /// <param name="y">Composante Y (ordonnee).</param>
    /// <param name="z">Composante Z (cote).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D(double x, double y, double z) { X = x; Y = y; Z = z; _pad = 0; }

    /// <summary>Constructeur a partir d'un scalaire unique (toutes composantes identiques).</summary>
    /// <param name="s">Valeur assignee a X, Y et Z.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D(double s) { X = s; Y = s; Z = s; _pad = 0; }

    /// <summary>Constructeur a partir d'un Vector3 System.Numerics (single precision).</summary>
    /// <param name="v">Vecteur System.Numerics.Vector3 a convertir.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3D(System.Numerics.Vector3 v) { X = v.X; Y = v.Y; Z = v.Z; _pad = 0; }

    /// <summary>Vecteur nul (0, 0, 0). Identite additive.</summary>
    public static readonly Vector3D Zero = new(0, 0, 0);
    /// <summary>Vecteur unitaire (1, 1, 1). Utilise pour les echelles uniformes.</summary>
    public static readonly Vector3D One = new(1, 1, 1);
    /// <summary>Vecteur unitaire X (1, 0, 0). Axe des abscisses.</summary>
    public static readonly Vector3D UnitX = new(1, 0, 0);
    /// <summary>Vecteur unitaire Y (0, 1, 0). Axe des ordonnees.</summary>
    public static readonly Vector3D UnitY = new(0, 1, 0);
    /// <summary>Vecteur unitaire Z (0, 0, 1). Axe de la profondeur.</summary>
    public static readonly Vector3D UnitZ = new(0, 0, 1);
    /// <summary>Vecteur vers le haut (0, 1, 0). Convention Y-up.</summary>
    public static readonly Vector3D Up = new(0, 1, 0);
    /// <summary>Vecteur vers le bas (0, -1, 0).</summary>
    public static readonly Vector3D Down = new(0, -1, 0);
    /// <summary>Vecteur vers la gauche (-1, 0, 0).</summary>
    public static readonly Vector3D Left = new(-1, 0, 0);
    /// <summary>Vecteur vers la droite (1, 0, 0).</summary>
    public static readonly Vector3D Right = new(1, 0, 0);
    /// <summary>Vecteur vers l'avant (0, 0, -1). Convention OpenGL.</summary>
    public static readonly Vector3D Forward = new(0, 0, -1);
    /// <summary>Vecteur vers l'arriere (0, 0, 1).</summary>
    public static readonly Vector3D Backward = new(0, 0, 1);

    /// <summary>Care de la longueur : |v|^2 = x^2 + y^2 + z^2. Evite sqrt.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double LengthSquared() => X * X + Y * Y + Z * Z;

    /// <summary>Longueur euclidienne : |v| = sqrt(x^2 + y^2 + z^2).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double Length() => Math.Sqrt(LengthSquared());

    /// <summary>Longueur L1 (Manhattan) : |x| + |y| + |z|. Distance de grille.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double LengthL1() => Math.Abs(X) + Math.Abs(Y) + Math.Abs(Z);

    /// <summary>Longueur L-infinity (Chebyshev) : max(|x|, |y|, |z|). Distance de echiquier.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double LengthLInfinity() => Math.Max(Math.Max(Math.Abs(X), Math.Abs(Y)), Math.Abs(Z));

    /// <summary>Vecteur normalise : v/|v|. Retourne Zero si |v| est proche de zero.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Normalized() { double len = Length(); return len > 1e-30 ? this / len : Zero; }

    /// <summary>Vecteur perpendiculaire (choisi arbitrairement si colineaire).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Perpendicular() => Math.Abs(X) < Math.Abs(Y) ? Cross(this, UnitX).Normalized() : Cross(this, UnitY).Normalized();

    /// <summary>Teste si le vecteur est proche de zero (norme < tol).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IsNearZero(double tol = 1e-10) => Math.Abs(X) < tol && Math.Abs(Y) < tol && Math.Abs(Z) < tol;

    /// <summary>Composante maximale : max(x, y, z).</summary>
    public readonly double MaxComponent => Math.Max(Math.Max(X, Y), Z);
    /// <summary>Composante minimale : min(x, y, z).</summary>
    public readonly double MinComponent => Math.Min(Math.Min(X, Y), Z);

    /// <summary>Valeur absolue de chaque composante : (|x|, |y|, |z|).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Abs() => new(Math.Abs(X), Math.Abs(Y), Math.Abs(Z));

    /// <summary>Inverse de chaque composante : (1/x, 1/y, 1/z). Zero si composante nulle.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Reciprocal() => new(X != 0 ? 1.0 / X : 0, Y != 0 ? 1.0 / Y : 0, Z != 0 ? 1.0 / Z : 0);

    /// <summary>Minimum element par element.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseMin(Vector3D b) => new(Math.Min(X, b.X), Math.Min(Y, b.Y), Math.Min(Z, b.Z));

    /// <summary>Maximum element par element.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseMax(Vector3D b) => new(Math.Max(X, b.X), Math.Max(Y, b.Y), Math.Max(Z, b.Z));

    /// <summary>Produit element par element (Hadamard product).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseProduct(Vector3D b) => new(X * b.X, Y * b.Y, Z * b.Z);

    /// <summary>Division element par element.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseDivide(Vector3D b) => new(b.X != 0 ? X / b.X : 0, b.Y != 0 ? Y / b.Y : 0, b.Z != 0 ? Z / b.Z : 0);

    /// <summary>Puissance element par element : (|x|^e * sign(x), ...).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWisePow(double e) => new(Math.Pow(Math.Abs(X), e) * Math.Sign(X), Math.Pow(Math.Abs(Y), e) * Math.Sign(Y), Math.Pow(Math.Abs(Z), e) * Math.Sign(Z));

    /// <summary>Exponentielle element par element : (e^x, e^y, e^z).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseExp() => new(Math.Exp(X), Math.Exp(Y), Math.Exp(Z));

    /// <summary>Logarithme naturel element par element. -Inf si composante <= 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ElementWiseLog() => new(X > 0 ? Math.Log(X) : double.NegativeInfinity, Y > 0 ? Math.Log(Y) : double.NegativeInfinity, Z > 0 ? Math.Log(Z) : double.NegativeInfinity);

    /// <summary>Signe de chaque composante : -1, 0, ou +1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Sign() => new(Math.Sign(X), Math.Sign(Y), Math.Sign(Z));

    /// <summary>Troncature vers zero de chaque composante.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Truncate() => new(Math.Truncate(X), Math.Truncate(Y), Math.Truncate(Z));

    /// <summary>Arrondi de chaque composante a 'decimals' decimales.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Round(int d = 0) => new(Math.Round(X, d), Math.Round(Y, d), Math.Round(Z, d));

    /// <summary>Clamp chaque composante entre min et max.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Clamp(double min, double max) => new(Math.Clamp(X, min, max), Math.Clamp(Y, min, max), Math.Clamp(Z, min, max));

    /// <summary>Lineraisation (lerp) vers un vecteur cible.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D LerpTo(Vector3D target, double t) => this + (target - this) * t;

    /// <summary>
    /// Spherical linear interpolation (SLERP) pour interpolation sur la sphere unite.
    /// Preserves les angles et la longueur, ideal pour les rotations.
    /// </summary>
    /// <param name="target">Vecteur cible (doit etre sur la sphere unite).</param>
    /// <param name="t">Parametre d'interpolation [0, 1].</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D SlerpTo(Vector3D target, double t)
    {
        double dot = Math.Clamp(Dot(this, target), -1.0, 1.0);
        double theta = Math.Acos(dot);
        if (theta < 1e-10)
            return LerpTo(target, t);
        double sinTheta = Math.Sin(theta);
        return this * Math.Sin((1 - t) * theta) / sinTheta + target * Math.Sin(t * theta) / sinTheta;
    }

    /// <summary>Reflexion par rapport a une normale : v - 2(v.n)n.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D Reflect(Vector3D n) => this - 2 * Dot(this, n) * n;

    /// <summary>Projection orthogonale sur un axe : (v.a/|a|^2) * a.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ProjectOn(Vector3D axis) => Dot(this, axis) * axis.Normalized();

    /// <summary>Rejet (composante perpendiculaire) sur un axe : v - proj(v, a).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D RejectFrom(Vector3D axis) => this - ProjectOn(axis);

    /// <summary>Angle en radians entre ce vecteur et un autre.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double AngleTo(Vector3D o) => Math.Acos(Math.Clamp(Dot(Normalized(), o.Normalized()), -1.0, 1.0));

    /// <summary>Angle en degrés.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double AngleDegreesTo(Vector3D o) => AngleTo(o) * (180.0 / Math.PI);

    /// <summary>Rotation autour de l'axe X par un angle donne (radians).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D RotateX(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(X, Y * c - Z * s, Y * s + Z * c); }

    /// <summary>Rotation autour de l'axe Y par un angle donne (radians).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D RotateY(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(X * c + Z * s, Y, -X * s + Z * c); }

    /// <summary>Rotation autour de l'axe Z par un angle donne (radians).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D RotateZ(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(X * c - Y * s, X * s + Y * c, Z); }

    /// <summary>Conversion en coordonnees spheriques : (r, theta, phi).
    /// r = distance a l'origine, theta = angle polaire [0,pi], phi = azimut [-pi,pi].</summary>
    public readonly Vector3D ToSpherical() { double r = Length(); if (r < 1e-30) return Zero; return new(r, Math.Acos(Math.Clamp(Z / r, -1, 1)), Math.Atan2(Y, X)); }
    /// <summary>Conversion depuis des coordonnees spheriques (r, theta, phi).</summary>
    public static Vector3D FromSpherical(double r, double theta, double phi) => new(r * Math.Sin(theta) * Math.Cos(phi), r * Math.Sin(theta) * Math.Sin(phi), r * Math.Cos(theta));
    /// <summary>Conversion en coordonnees cylindriques : (r, theta, z).</summary>
    public readonly Vector3D ToCylindrical() { double r = Math.Sqrt(X * X + Y * Y); return new(r, Math.Atan2(Y, X), Z); }
    /// <summary>Conversion depuis des coordonnees cylindriques (r, theta, z).</summary>
    public static Vector3D FromCylindrical(double r, double theta, double z) => new(r * Math.Cos(theta), r * Math.Sin(theta), z);

    /// <summary>Interpolation de Catmull-Rom spline. Passe par les 4 points.</summary>
    public static Vector3D CatmullRom(Vector3D p0, Vector3D p1, Vector3D p2, Vector3D p3, double t)
    {
        double t2 = t * t, t3 = t2 * t;
        return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
    }

    /// <summary>Interpolation de Bezier cubique. Controle par 4 points de controle.</summary>
    public static Vector3D CubicBezier(Vector3D p0, Vector3D p1, Vector3D p2, Vector3D p3, double t)
    {
        double u = 1 - t;
        return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
    }

    /// <summary>Spline B-spline cubique. Approximation lisse des 4 points.</summary>
    public static Vector3D BSpline(Vector3D p0, Vector3D p1, Vector3D p2, Vector3D p3, double t)
    {
        double t2 = t * t, t3 = t2 * t;
        return p0 * (-t3 + 3 * t2 - 3 * t + 1) / 6 + p1 * (3 * t3 - 6 * t2 + 4) / 6 + p2 * (-3 * t3 + 3 * t2 + 3 * t + 1) / 6 + p3 * t3 / 6;
    }

    /// <summary>Fonction smoothstep (interpolation cubique hermitienne).</summary>
    public static double Smoothstep(double edge0, double edge1, double x) { double t = Math.Clamp((x - edge0) / (edge1 - edge0), 0, 1); return t * t * (3 - 2 * t); }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator +(Vector3D a, Vector3D b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator -(Vector3D a, Vector3D b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator *(Vector3D v, double s) => new(v.X * s, v.Y * s, v.Z * s);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator *(double s, Vector3D v) => new(v.X * s, v.Y * s, v.Z * s);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator /(Vector3D v, double s) { double i = 1.0 / s; return new(v.X * i, v.Y * i, v.Z * i); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D operator -(Vector3D v) => new(-v.X, -v.Y, -v.Z);

    /// <summary>Produit scalaire (dot product) : a.b = ax*bx + ay*by + az*bz.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Dot(Vector3D a, Vector3D b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    /// <summary>Produit vectoriel (cross product) : axb. Resultat perpendiculaire aux deux.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D Cross(Vector3D a, Vector3D b) => new(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    /// <summary>Triple produit scalaire : a.(bxc). Volume du parallelepiped.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double TripleScalarProduct(Vector3D a, Vector3D b, Vector3D c) => Dot(a, Cross(b, c));
    /// <summary>Triple produit vectoriel : ax(bxc).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D TripleVectorProduct(Vector3D a, Vector3D b, Vector3D c) => Cross(a, Cross(b, c));

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double Distance(Vector3D a, Vector3D b) => (a - b).Length();
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double DistanceSquared(Vector3D a, Vector3D b) => (a - b).LengthSquared();
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double DistanceL1(Vector3D a, Vector3D b) => (a - b).LengthL1();
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double DistanceLInfinity(Vector3D a, Vector3D b) => (a - b).LengthLInfinity();
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D Lerp(Vector3D a, Vector3D b, double t) => new(a.X + (b.X - a.X) * t, a.Y + (b.Y - a.Y) * t, a.Z + (b.Z - a.Z) * t);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D Min(Vector3D a, Vector3D b) => new(Math.Min(a.X, b.X), Math.Min(a.Y, b.Y), Math.Min(a.Z, b.Z));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D Max(Vector3D a, Vector3D b) => new(Math.Max(a.X, b.X), Math.Max(a.Y, b.Y), Math.Max(a.Z, b.Z));
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D Clamp(Vector3D v, Vector3D mn, Vector3D mx) => Min(Max(v, mn), mx);
    /// <summary>Point le plus proche sur un segment [a,b].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static Vector3D ClosestPointOnSegment(Vector3D p, Vector3D a, Vector3D b) { Vector3D ab = b - a; double t = Math.Clamp(Dot(p - a, ab) / Dot(ab, ab), 0, 1); return a + ab * t; }
    /// <summary>Distance d'un point a une droite infinie.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double DistancePointToLine(Vector3D p, Vector3D o, Vector3D d) => Cross(p - o, d).Length() / d.Length();
    /// <summary>Distance entre deux droites infinies.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double DistanceBetweenLines(Vector3D o1, Vector3D d1, Vector3D o2, Vector3D d2) { Vector3D n = Cross(d1, d2); double den = Dot(n, n); if (den < 1e-20) return DistancePointToLine(o1, o2, d2); return Math.Abs(Dot(o2 - o1, n)) / n.Length(); }

    /// <summary>Conversion en Vector3 single-precision System.Numerics.</summary>
    public readonly System.Numerics.Vector3 ToSingle() => new((float)X, (float)Y, (float)Z);
    /// <summary>Conversion en Vector4 single-precision (w=1).</summary>
    public readonly System.Numerics.Vector4 ToSingle4() => new((float)X, (float)Y, (float)Z, 1.0f);
    /// <summary>Depuis un Vector3 System.Numerics.</summary>
    public static Vector3D FromSingle(System.Numerics.Vector3 v) => new(v.X, v.Y, v.Z);
    /// <summary>Conversion en tableau de doubles [3].</summary>
    public readonly double[] ToArray() => new double[] { X, Y, Z };

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool Equals(Vector3D o) => Math.Abs(X - o.X) < 1e-10 && Math.Abs(Y - o.Y) < 1e-10 && Math.Abs(Z - o.Z) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is Vector3D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(X, Y, Z);
    public static bool operator ==(Vector3D a, Vector3D b) => a.Equals(b);
    public static bool operator !=(Vector3D a, Vector3D b) => !a.Equals(b);
    public override readonly string ToString() => $"({X:F4}, {Y:F4}, {Z:F4})";
    public readonly string ToString(string fmt) => $"({X.ToString(fmt)}, {Y.ToString(fmt)}, {Z.ToString(fmt)})";
}


// ═══════════════════════════════════════════════════════════════════════════════
