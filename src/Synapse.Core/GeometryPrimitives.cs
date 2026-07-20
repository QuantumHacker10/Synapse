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

// SECTION 4: PRIMITIVES GEOMETRIQUES
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Boite englobante axiale (AABB) en 3D. Definie par un coin min et un coin max.
/// Utilisee pour l'acceleration spatiale, le culling, et les tests de collision.
/// Les operations sont O(1) grace au stockage direct min/max.
///
/// MEMORY LAYOUT: 64 bytes (2 Vector3D).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64, Pack = 32)]
[DebuggerDisplay("AABB({Min} -> {Max})")]
public struct BoundingBox3D : IEquatable<BoundingBox3D>
{
    [FieldOffset(0)] public Vector3D Min;
    [FieldOffset(32)] public Vector3D Max;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public BoundingBox3D(Vector3D min, Vector3D max) { Min = min; Max = max; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public BoundingBox3D(double minX, double minY, double minZ, double maxX, double maxY, double maxZ) { Min = new(minX, minY, minZ); Max = new(maxX, maxY, maxZ); }

    public static readonly BoundingBox3D Empty = new(Vector3D.Zero, Vector3D.Zero);
    public static readonly BoundingBox3D Infinite = new(new(double.NegativeInfinity), new(double.PositiveInfinity));

    public readonly Vector3D Center => (Min + Max) * 0.5;
    public readonly Vector3D Size => Max - Min;
    public readonly Vector3D HalfSize => Size * 0.5;
    public readonly double Volume { get { Vector3D s = Size; return s.X * s.Y * s.Z; } }
    public readonly double SurfaceArea { get { Vector3D s = Size; return 2.0 * (s.X * s.Y + s.Y * s.Z + s.Z * s.X); } }
    public readonly double DiagonalLength => Size.Length();
    public readonly Vector3D Diagonal => Size;
    public readonly double Perimeter { get { Vector3D s = Size; return 4.0 * (s.X + s.Y + s.Z); } }
    public readonly double MaxFaceArea { get { Vector3D s = Size; double xy = s.X * s.Y, yz = s.Y * s.Z, zx = s.Z * s.X; return Math.Max(Math.Max(xy, yz), zx); } }
    public readonly int LargestAxis { get { Vector3D s = Size; if (s.X >= s.Y && s.X >= s.Z) return 0; if (s.Y >= s.Z) return 1; return 2; } }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool Contains(Vector3D p) => p.X >= Min.X && p.X <= Max.X && p.Y >= Min.Y && p.Y <= Max.Y && p.Z >= Min.Z && p.Z <= Max.Z;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool ContainsStrict(Vector3D p) => p.X > Min.X && p.X < Max.X && p.Y > Min.Y && p.Y < Max.Y && p.Z > Min.Z && p.Z < Max.Z;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool Contains(BoundingBox3D o) => o.Min.X >= Min.X && o.Max.X <= Max.X && o.Min.Y >= Min.Y && o.Max.Y <= Max.Y && o.Min.Z >= Min.Z && o.Max.Z <= Max.Z;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool Intersects(BoundingBox3D o) => Min.X <= o.Max.X && Max.X >= o.Min.X && Min.Y <= o.Max.Y && Max.Y >= o.Min.Y && Min.Z <= o.Max.Z && Max.Z >= o.Min.Z;
    public readonly BoundingBox3D Intersection(BoundingBox3D o) => new(Vector3D.Max(Min, o.Min), Vector3D.Min(Max, o.Max));
    public readonly BoundingBox3D Union(BoundingBox3D o) => new(Vector3D.Min(Min, o.Min), Vector3D.Max(Max, o.Max));
    public static BoundingBox3D FromCenterAndHalfExtents(Vector3D c, Vector3D h) => new(c - h, c + h);
    public readonly BoundingBox3D Expanded(double m) => new(Min - new Vector3D(m), Max + new Vector3D(m));
    public readonly BoundingBox3D Expanded(Vector3D m) => new(Min - m, Max + m);
    public readonly Vector3D ClosestPoint(Vector3D p) => Vector3D.Clamp(p, Min, Max);
    public readonly double DistanceTo(Vector3D p) => (ClosestPoint(p) - p).Length();
    public readonly double DistanceSquaredTo(Vector3D p) => (ClosestPoint(p) - p).LengthSquared();

    public readonly bool Equals(BoundingBox3D o) => Min.Equals(o.Min) && Max.Equals(o.Max);
    public override readonly bool Equals(object? obj) => obj is BoundingBox3D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Min, Max);
    public static bool operator ==(BoundingBox3D a, BoundingBox3D b) => a.Equals(b);
    public static bool operator !=(BoundingBox3D a, BoundingBox3D b) => !a.Equals(b);
    public override readonly string ToString() => $"AABB({Min} -> {Max})";
}

/// <summary>
/// Rayon en 3D : origine + direction normalisee. Utilise pour le raycasting,
/// le picking, les tests d'intersection, et la simulation de photons.
/// La direction est toujours normalisee pour simplifier les calculs de distance.
///
/// MEMORY LAYOUT: 64 bytes (2 Vector3D).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64, Pack = 32)]
[DebuggerDisplay("Ray(Origin={Origin}, Dir={Direction})")]
public struct Ray3D : IEquatable<Ray3D>
{
    [FieldOffset(0)] public Vector3D Origin;
    [FieldOffset(32)] public Vector3D Direction;

    /// <summary>Constructeur avec normalisation automatique de la direction.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Ray3D(Vector3D origin, Vector3D direction) { Origin = origin; Direction = direction.Normalized(); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Ray3D(double ox, double oy, double oz, double dx, double dy, double dz) { Origin = new Vector3D(ox, oy, oz); Direction = new Vector3D(dx, dy, dz).Normalized(); }

    /// <summary>Point sur le rayon a la distance t : P = O + t*D.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly Vector3D GetPoint(double t) => Origin + Direction * t;

    /// <summary>Intersection avec un AABB. Retourne true si le rayon traverse la boite.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IntersectsAABB(BoundingBox3D box, out double tMin, out double tMax)
    {
        tMin = 0;
        tMax = double.PositiveInfinity;
        for (int i = 0; i < 3; i++)
        {
            double orig = i == 0 ? Origin.X : (i == 1 ? Origin.Y : Origin.Z);
            double dir = i == 0 ? Direction.X : (i == 1 ? Direction.Y : Direction.Z);
            double bmin = i == 0 ? box.Min.X : (i == 1 ? box.Min.Y : box.Min.Z);
            double bmax = i == 0 ? box.Max.X : (i == 1 ? box.Max.Y : box.Max.Z);
            if (Math.Abs(dir) < 1e-15)
            { if (orig < bmin || orig > bmax) return false; }
            else
            { double inv = 1.0 / dir; double t1 = (bmin - orig) * inv, t2 = (bmax - orig) * inv; if (t1 > t2) (t1, t2) = (t2, t1); tMin = Math.Max(tMin, t1); tMax = Math.Min(tMax, t2); if (tMin > tMax) return false; }
        }
        return true;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool IntersectsAABB(BoundingBox3D box) => IntersectsAABB(box, out _, out _);

    /// <summary>Intersection avec un plan.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IntersectPlane(Plane3D plane, out double t)
    {
        double denom = Vector3D.Dot(plane.Normal, Direction);
        if (Math.Abs(denom) < 1e-15)
        { t = 0; return false; }
        t = -(Vector3D.Dot(plane.Normal, Origin) + plane.Distance) / denom;
        return t >= 0;
    }

    /// <summary>Intersection avec une sphere : retourne les distances d'entree et de sortie.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool IntersectSphere(Vector3D center, double radius, out double tEntry, out double tExit)
    {
        Vector3D oc = Origin - center;
        double a = Vector3D.Dot(Direction, Direction);
        double b = 2.0 * Vector3D.Dot(oc, Direction);
        double c = Vector3D.Dot(oc, oc) - radius * radius;
        double disc = b * b - 4.0 * a * c;
        if (disc < 0)
        { tEntry = 0; tExit = 0; return false; }
        double sqrtD = Math.Sqrt(disc);
        tEntry = (-b - sqrtD) / (2.0 * a);
        tExit = (-b + sqrtD) / (2.0 * a);
        return tExit >= 0;
    }

    /// <summary>Point le plus proche du rayon a une position donnee.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D ClosestPoint(Vector3D point) { double t = Math.Max(0, Vector3D.Dot(point - Origin, Direction)); return Origin + Direction * t; }

    /// <summary>Distance du rayon a un point.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double DistanceToPoint(Vector3D point) => (ClosestPoint(point) - point).Length();

    public readonly bool Equals(Ray3D o) => Origin.Equals(o.Origin) && Direction.Equals(o.Direction);
    public override readonly bool Equals(object? obj) => obj is Ray3D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Origin, Direction);
    public static bool operator ==(Ray3D a, Ray3D b) => a.Equals(b);
    public static bool operator !=(Ray3D a, Ray3D b) => !a.Equals(b);
    public override readonly string ToString() => $"Ray(Origin={Origin}, Dir={Direction})";
}

/// <summary>
/// Plan en 3D : normale unitaire + distance signee au plan depuis l'origine.
/// L'equation du plan est : N.x + d = 0, ou N est la normale et d la distance.
/// Un point est devant le plan si N.p + d > 0, derriere si < 0.
///
/// MEMORY LAYOUT: 32 bytes (Vector3D + double).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 32, Pack = 16)]
[DebuggerDisplay("Plane(N={Normal}, D={Distance:F4})")]
public struct Plane3D : IEquatable<Plane3D>
{
    [FieldOffset(0)] public Vector3D Normal;
    [FieldOffset(24)] public double Distance;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public Plane3D(Vector3D normal, double distance) { Normal = normal.Normalized(); Distance = distance; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public Plane3D(Vector3D point, Vector3D normal) { Normal = normal.Normalized(); Distance = -Vector3D.Dot(Normal, point); }

    /// <summary>Construit un plan a partir de 3 points non colineaires.</summary>
    public static Plane3D FromPoints(Vector3D a, Vector3D b, Vector3D c) { Vector3D n = Vector3D.Cross(b - a, c - a).Normalized(); return new(n, -Vector3D.Dot(n, a)); }

    /// <summary>Distance signee d'un point au plan : N.p + d.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly double SignedDistance(Vector3D point) => Vector3D.Dot(Normal, point) + Distance;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool IsBehind(Vector3D p) => SignedDistance(p) < 0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly bool IsInFront(Vector3D p) => SignedDistance(p) > 0;
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly Vector3D ProjectPoint(Vector3D p) => p - Normal * SignedDistance(p);
    public readonly Plane3D Flipped => new(-Normal, -Distance);

    public readonly bool Equals(Plane3D o) => Normal.Equals(o.Normal) && Math.Abs(Distance - o.Distance) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is Plane3D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Normal, Distance);
    public static bool operator ==(Plane3D a, Plane3D b) => a.Equals(b);
    public static bool operator !=(Plane3D a, Plane3D b) => !a.Equals(b);
    public override readonly string ToString() => $"Plane(N={Normal}, D={Distance:F4})";
}

/// <summary>
/// Volume de vision (frustum) compose de 6 plans : left, right, bottom, top, near, far.
/// Utilise pour le culling hierarchique (frustum culling) dans le moteur de rendu.
///
/// MEMORY LAYOUT: 192 bytes (6 Plane3D).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 192, Pack = 16)]
public struct Frustum6Planes : IEquatable<Frustum6Planes>
{
    [FieldOffset(0)] public Plane3D Left;
    [FieldOffset(32)] public Plane3D Right;
    [FieldOffset(64)] public Plane3D Bottom;
    [FieldOffset(96)] public Plane3D Top;
    [FieldOffset(128)] public Plane3D Near;
    [FieldOffset(160)] public Plane3D Far;

    /// <summary>Extrait les 6 plans a partir d'une matrice view-projection.</summary>
    public static Frustum6Planes FromViewProjection(Matrix4x4D vp)
    {
        Frustum6Planes f;
        f.Left = Norm(new Plane3D(new Vector3D(vp.M14 + vp.M11, vp.M24 + vp.M21, vp.M34 + vp.M31), vp.M44 + vp.M41));
        f.Right = Norm(new Plane3D(new Vector3D(vp.M14 - vp.M11, vp.M24 - vp.M21, vp.M34 - vp.M31), vp.M44 - vp.M41));
        f.Bottom = Norm(new Plane3D(new Vector3D(vp.M14 + vp.M12, vp.M24 + vp.M22, vp.M34 + vp.M32), vp.M44 + vp.M42));
        f.Top = Norm(new Plane3D(new Vector3D(vp.M14 - vp.M12, vp.M24 - vp.M22, vp.M34 - vp.M32), vp.M44 - vp.M42));
        f.Near = Norm(new Plane3D(new Vector3D(vp.M13, vp.M23, vp.M33), vp.M43));
        f.Far = Norm(new Plane3D(new Vector3D(vp.M14 - vp.M13, vp.M24 - vp.M23, vp.M34 - vp.M33), vp.M44 - vp.M43));
        return f;
    }
    private static Plane3D Norm(Plane3D p) { double len = p.Normal.Length(); return len < 1e-15 ? p : new(p.Normal / len, p.Distance / len); }

    /// <summary>Teste si un point est a l'interieur du frustum.</summary>
    public readonly bool ContainsPoint(Vector3D p) => Left.SignedDistance(p) >= 0 && Right.SignedDistance(p) >= 0 && Bottom.SignedDistance(p) >= 0 && Top.SignedDistance(p) >= 0 && Near.SignedDistance(p) >= 0 && Far.SignedDistance(p) >= 0;

    /// <summary>Teste l'intersection avec un AABB (separation axis theorem).</summary>
    public readonly bool IntersectsAABB(BoundingBox3D box)
    {
        Span<Plane3D> planes = stackalloc Plane3D[] { Left, Right, Bottom, Top, Near, Far };
        for (int i = 0; i < 6; i++)
        {
            Vector3D pv = new(planes[i].Normal.X >= 0 ? box.Max.X : box.Min.X, planes[i].Normal.Y >= 0 ? box.Max.Y : box.Min.Y, planes[i].Normal.Z >= 0 ? box.Max.Z : box.Min.Z);
            if (planes[i].SignedDistance(pv) < 0)
                return false;
        }
        return true;
    }

    public readonly bool Equals(Frustum6Planes o) => Left.Equals(o.Left) && Right.Equals(o.Right) && Bottom.Equals(o.Bottom) && Top.Equals(o.Top) && Near.Equals(o.Near) && Far.Equals(o.Far);
    public override readonly bool Equals(object? obj) => obj is Frustum6Planes o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Left, Right, Bottom, Top, Near, Far);
    public static bool operator ==(Frustum6Planes a, Frustum6Planes b) => a.Equals(b);
    public static bool operator !=(Frustum6Planes a, Frustum6Planes b) => !a.Equals(b);
}

/// <summary>
/// Couleur HDR (High Dynamic Range) avec composantes spectrales.
/// Supporte les valeurs au-dela de [0,1] pour l'eclairage realiste.
/// Inclut des methodes de tonemapping (Reinhard, ACES) pour la conversion
/// vers l'affichage standard.
///
/// MEMORY LAYOUT: 64 bytes (7 doubles + padding).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64, Pack = 8)]
[DebuggerDisplay("ColorHDR({R:F3}, {G:F3}, {B:F3}, I={Intensity:F3})")]
public struct ColorHDR : IEquatable<ColorHDR>
{
    [FieldOffset(0)] public double R;
    [FieldOffset(8)] public double G;
    [FieldOffset(16)] public double B;
    [FieldOffset(24)] public double Intensity;
    [FieldOffset(32)] public double SpectralLow;
    [FieldOffset(40)] public double SpectralMid;
    [FieldOffset(48)] public double SpectralHigh;
    [FieldOffset(56)] public double _pad;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public ColorHDR(double r, double g, double b, double intensity = 1.0) { R = r; G = g; B = b; Intensity = intensity; SpectralLow = r; SpectralMid = g; SpectralHigh = b; _pad = 0; }

    public static readonly ColorHDR Black = new(0, 0, 0, 0);
    public static readonly ColorHDR White = new(1, 1, 1, 1);
    public static readonly ColorHDR Red = new(1, 0, 0, 1);
    public static readonly ColorHDR Green = new(0, 1, 0, 1);
    public static readonly ColorHDR Blue = new(0, 0, 1, 1);
    public static readonly ColorHDR Yellow = new(1, 1, 0, 1);
    public static readonly ColorHDR Cyan = new(0, 1, 1, 1);
    public static readonly ColorHDR Magenta = new(1, 0, 1, 1);

    /// <summary>Luminance relative (BT.709) : 0.2126*R + 0.7152*G + 0.0722*B.</summary>
    public readonly double Luminance => 0.2126 * R + 0.7152 * G + 0.0722 * B;
    public readonly double MaxComponent => Math.Max(Math.Max(R, G), B);
    public readonly double AverageComponent => (R + G + B) / 3.0;

    public readonly ColorHDR Scaled(double s) => new(R * s * Intensity, G * s * Intensity, B * s * Intensity, 1.0);
    public readonly ColorHDR WithIntensity(double i) => new(R, G, B, i);

    /// <summary>Tonemapping Reinhard : L = 1 - exp(-E * exposure).</summary>
    public readonly ColorHDR ToneMapReinhard(double exposure = 1.0) { double er = R * Intensity * exposure, eg = G * Intensity * exposure, eb = B * Intensity * exposure; return new(1 - Math.Exp(-er), 1 - Math.Exp(-eg), 1 - Math.Exp(-eb), 1.0); }
    /// <summary>Tonemapping ACES Filmic : courbe S pour un contraste cinematographique.</summary>
    public readonly ColorHDR ToneMapAces()
    {
        const double a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
        double er = R * Intensity * 0.6, eg = G * Intensity * 0.6, eb = B * Intensity * 0.6;
        return new(Math.Clamp((er * (a * er + b)) / (er * (c * er + d) + e), 0, 1), Math.Clamp((eg * (a * eg + b)) / (eg * (c * eg + d) + e), 0, 1), Math.Clamp((eb * (a * eb + b)) / (eb * (c * eb + d) + e), 0, 1), 1.0);
    }
    /// <summary>Conversion gamma vers linear.</summary>
    public readonly ColorHDR GammaToLinear() => new(Math.Pow(R, 2.2), Math.Pow(G, 2.2), Math.Pow(B, 2.2), Intensity);
    /// <summary>Conversion linear vers gamma.</summary>
    public readonly ColorHDR LinearToGamma() => new(Math.Pow(Math.Max(R, 0), 1.0 / 2.2), Math.Pow(Math.Max(G, 0), 1.0 / 2.2), Math.Pow(Math.Max(B, 0), 1.0 / 2.2), Intensity);

    public readonly ColorHDR Lerp(ColorHDR o, double t) { double u = 1 - t; return new(R * u + o.R * t, G * u + o.G * t, B * u + o.B * t, Intensity * u + o.Intensity * t); }
    public static ColorHDR operator +(ColorHDR a, ColorHDR b) => new(a.R + b.R, a.G + b.G, a.B + b.B, (a.Intensity + b.Intensity) * 0.5);
    public static ColorHDR operator *(ColorHDR a, ColorHDR b) => new(a.R * b.R, a.G * b.G, a.B * b.B, a.Intensity * b.Intensity);
    public static ColorHDR operator *(ColorHDR c, double s) => new(c.R * s, c.G * s, c.B * s, c.Intensity);
    public static ColorHDR operator *(double s, ColorHDR c) => c * s;

    public readonly bool Equals(ColorHDR o) => Math.Abs(R - o.R) < 1e-10 && Math.Abs(G - o.G) < 1e-10 && Math.Abs(B - o.B) < 1e-10 && Math.Abs(Intensity - o.Intensity) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is ColorHDR o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(R, G, B, Intensity);
    public static bool operator ==(ColorHDR a, ColorHDR b) => a.Equals(b);
    public static bool operator !=(ColorHDR a, ColorHDR b) => !a.Equals(b);
    public override readonly string ToString() => $"ColorHDR({R:F3}, {G:F3}, {B:F3}, I={Intensity:F3})";
}

/// <summary>
/// Intervalle [Min, Max] pour l'arithmetique par intervalles (IA).
/// Garantit les bornes d'erreur dans les calculs numeriques.
/// Chaque operation arithmetique propage correctement les erreurs d'arrondi.
///
/// Utilise pour la verification de racines, l'analyse d'erreurs,
/// et les tests d'inclusion dans les structures de donnees spatiales.
///
/// MEMORY LAYOUT: 16 bytes (2 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 16, Pack = 8)]
[DebuggerDisplay("[{Min:F6}, {Max:F6}]")]
public struct IntervalD : IEquatable<IntervalD>
{
    [FieldOffset(0)] public double Min;
    [FieldOffset(8)] public double Max;

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public IntervalD(double min, double max) { Min = min; Max = max; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public IntervalD(double value) { Min = value; Max = value; }

    public static readonly IntervalD Zero = new(0, 0);
    public static readonly IntervalD Positive = new(0, double.PositiveInfinity);
    public static readonly IntervalD Negative = new(double.NegativeInfinity, 0);
    public static readonly IntervalD Entire = new(double.NegativeInfinity, double.PositiveInfinity);
    public static readonly IntervalD Unit = new(0, 1);

    public readonly double Midpoint => (Min + Max) * 0.5;
    public readonly double Width => Max - Min;
    public readonly double Radius => Width * 0.5;
    public readonly bool IsValid => Min <= Max;
    public readonly bool IsPoint => Math.Abs(Max - Min) < 1e-15;
    public readonly bool Contains(double v) => v >= Min && v <= Max;
    public readonly bool Contains(IntervalD o) => o.Min >= Min && o.Max <= Max;
    public readonly bool Intersects(IntervalD o) => Min <= o.Max && Max >= o.Min;
    public readonly IntervalD Intersection(IntervalD o) => new(Math.Max(Min, o.Min), Math.Min(Max, o.Max));
    public readonly IntervalD Union(IntervalD o) => new(Math.Min(Min, o.Min), Math.Max(Max, o.Max));

    public static IntervalD operator +(IntervalD a, IntervalD b) => new(a.Min + b.Min, a.Max + b.Max);
    public static IntervalD operator -(IntervalD a, IntervalD b) => new(a.Min - b.Max, a.Max - b.Min);
    public static IntervalD operator *(IntervalD a, IntervalD b) { double p1 = a.Min * b.Min, p2 = a.Min * b.Max, p3 = a.Max * b.Min, p4 = a.Max * b.Max; return new(Math.Min(Math.Min(p1, p2), Math.Min(p3, p4)), Math.Max(Math.Max(p1, p2), Math.Max(p3, p4))); }
    public static IntervalD operator /(IntervalD a, IntervalD b) { if (b.Contains(0)) return Entire; double p1 = a.Min / b.Min, p2 = a.Min / b.Max, p3 = a.Max / b.Min, p4 = a.Max / b.Max; return new(Math.Min(Math.Min(p1, p2), Math.Min(p3, p4)), Math.Max(Math.Max(p1, p2), Math.Max(p3, p4))); }
    public static IntervalD operator +(IntervalD a, double s) => new(a.Min + s, a.Max + s);
    public static IntervalD operator -(IntervalD a, double s) => new(a.Min - s, a.Max - s);
    public static IntervalD operator *(IntervalD a, double s) => s >= 0 ? new(a.Min * s, a.Max * s) : new(a.Max * s, a.Min * s);
    public static IntervalD operator /(IntervalD a, double s) => a * (1.0 / s);
    public static IntervalD operator -(IntervalD a) => new(-a.Max, -a.Min);

    public readonly IntervalD Sin() { double sMin = Math.Sin(Min), sMax = Math.Sin(Max); double lo = Math.Min(sMin, sMax), hi = Math.Max(sMin, sMax); if (Min < Max && Math.PI > Min) { lo = -1; hi = Math.Max(sMin, sMax); } return new(lo, hi); }
    public readonly IntervalD Cos() { double cMin = Math.Cos(Min), cMax = Math.Cos(Max); double lo = Math.Min(cMin, cMax), hi = Math.Max(cMin, cMax); if (Min < Max && 0 > Min && 0 < Max) hi = 1; return new(lo, hi); }
    public readonly IntervalD Exp() => new(Math.Exp(Min), Math.Exp(Max));
    public readonly IntervalD Log() => Min > 0 ? new(Math.Log(Min), Math.Log(Max)) : Entire;
    public readonly IntervalD Abs() => Min >= 0 ? this : (Max <= 0 ? new(-Max, -Min) : new(0, Math.Max(-Min, Max)));
    public readonly IntervalD Sqr() => Min >= 0 ? new(Min * Min, Max * Max) : (Max <= 0 ? new(Max * Max, Min * Min) : new(0, Math.Max(Min * Min, Max * Max)));
    public readonly IntervalD Sqrt() => Min >= 0 ? new(Math.Sqrt(Min), Math.Sqrt(Max)) : Entire;

    public readonly bool Equals(IntervalD o) => Math.Abs(Min - o.Min) < 1e-15 && Math.Abs(Max - o.Max) < 1e-15;
    public override readonly bool Equals(object? obj) => obj is IntervalD o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(Min, Max);
    public static bool operator ==(IntervalD a, IntervalD b) => a.Equals(b);
    public static bool operator !=(IntervalD a, IntervalD b) => !a.Equals(b);
    public override readonly string ToString() => $"[{Min:F6}, {Max:F6}]";
}

// ═══════════════════════════════════════════════════════════════════════════════
