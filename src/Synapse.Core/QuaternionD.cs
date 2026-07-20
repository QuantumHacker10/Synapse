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

// SECTION 3.4: QUATERNIOND — QUATERNION DOUBLE PRECISION
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Quaternion double precision pour les rotations 3D sans gimbal lock.
/// Format : q = w + xi + yj + zk, avec |q| = 1 pour les rotations propres.
///
/// Le quaternion evite les singularites des angles d'Euler (gimbal lock)
/// et permet des interpolations lisses (SLERP) entre rotations.
///
/// MEMORY LAYOUT: 32 bytes (4 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 32, Pack = 8)]
[DebuggerDisplay("Quat({W:F4}, {X:F4}, {Y:F4}, {Z:F4})")]
public struct QuaternionD : IEquatable<QuaternionD>
{
    /// <summary>Partie scalaire w = cos(theta/2).</summary>
    [FieldOffset(0)] public double W;
    /// <summary>Partie vectorielle x.</summary>
    [FieldOffset(8)] public double X;
    /// <summary>Partie vectorielle y.</summary>
    [FieldOffset(16)] public double Y;
    /// <summary>Partie vectorielle z.</summary>
    [FieldOffset(24)] public double Z;

    /// <summary>Constructeur avec les 4 composantes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public QuaternionD(double w, double x, double y, double z) { W = w; X = x; Y = y; Z = z; }

    /// <summary>Quaternion identite (pas de rotation). q = (1, 0, 0, 0).</summary>
    public static readonly QuaternionD Identity = new(1, 0, 0, 0);
    /// <summary>Quaternion nul.</summary>
    public static readonly QuaternionD Zero = new(0, 0, 0, 0);

    /// <summary>Conjuge : q* = (w, -x, -y, -z). Inverse pour les rotations.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly QuaternionD Conjugate() => new(W, -X, -Y, -Z);
    /// <summary>Inverse : q^-1 = q*/|q|^2. Retourne Identity si |q| ~ 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly QuaternionD Inverse()
    {
        double ls = W * W + X * X + Y * Y + Z * Z;
        if (ls < 1e-30)
            return Identity;
        double inv = 1.0 / ls;
        return new QuaternionD(W * inv, -X * inv, -Y * inv, -Z * inv);
    }
    /// <summary>Norme du quaternion : |q| = sqrt(w^2 + x^2 + y^2 + z^2).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double Length() => Math.Sqrt(W * W + X * X + Y * Y + Z * Z);
    /// <summary>Care de la norme.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly double LengthSquared() => W * W + X * X + Y * Y + Z * Z;
    /// <summary>Quaternion normalise (norme = 1). Indispensable pour les rotations.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly QuaternionD Normalized()
    {
        double len = Length();
        if (len < 1e-30)
            return Identity;
        double inv = 1.0 / len;
        return new QuaternionD(W * inv, X * inv, Y * inv, Z * inv);
    }
    /// <summary>Normalise en place.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Normalize() { double len = Length(); if (len < 1e-30) return; double inv = 1.0 / len; W *= inv; X *= inv; Y *= inv; Z *= inv; }

    /// <summary>
    /// Construit un quaternion a partir d'un axe et d'un angle.
    /// q = cos(theta/2) + sin(theta/2)*(ux*i + uy*j + uz*k)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static QuaternionD FromAxisAngle(Vector3D axis, double angle)
    {
        double ha = angle * 0.5, s = Math.Sin(ha);
        Vector3D n = axis.Normalized();
        return new QuaternionD(Math.Cos(ha), n.X * s, n.Y * s, n.Z * s);
    }
    /// <summary>Construit un quaternion depuis les angles d'Euler (ZYX convention : yaw, pitch, roll).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static QuaternionD FromEuler(double roll, double pitch, double yaw)
    {
        double cr = Math.Cos(roll * 0.5), sr = Math.Sin(roll * 0.5);
        double cp = Math.Cos(pitch * 0.5), sp = Math.Sin(pitch * 0.5);
        double cy = Math.Cos(yaw * 0.5), sy = Math.Sin(yaw * 0.5);
        return new QuaternionD(cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy);
    }
    /// <summary>Construit un quaternion depuis une matrice de rotation (algorithme de Shepperd).</summary>
    public static QuaternionD FromMatrix3x3(Tensor3D m)
    {
        double trace = m.M11 + m.M22 + m.M33;
        if (trace > 0)
        { double s = 0.5 / Math.Sqrt(trace + 1.0); return new QuaternionD(0.25 / s, (m.M32 - m.M23) * s, (m.M13 - m.M31) * s, (m.M21 - m.M12) * s).Normalized(); }
        if (m.M11 > m.M22 && m.M11 > m.M33)
        { double s = 2.0 * Math.Sqrt(1.0 + m.M11 - m.M22 - m.M33); return new QuaternionD((m.M32 - m.M23) / s, 0.25 * s, (m.M12 + m.M21) / s, (m.M13 + m.M31) / s).Normalized(); }
        if (m.M22 > m.M33)
        { double s = 2.0 * Math.Sqrt(1.0 + m.M22 - m.M11 - m.M33); return new QuaternionD((m.M13 - m.M31) / s, (m.M12 + m.M21) / s, 0.25 * s, (m.M23 + m.M32) / s).Normalized(); }
        double s2 = 2.0 * Math.Sqrt(1.0 + m.M33 - m.M11 - m.M22);
        return new QuaternionD((m.M21 - m.M12) / s2, (m.M13 + m.M31) / s2, (m.M23 + m.M32) / s2, 0.25 * s2).Normalized();
    }

    /// <summary>Convertit en matrice de rotation 3x3.</summary>
    public readonly Tensor3D ToTensor3D()
    {
        double xx = X * X, yy = Y * Y, zz = Z * Z, xy = X * Y, xz = X * Z, yz = Y * Z;
        double wx = W * X, wy = W * Y, wz = W * Z;
        return new Tensor3D(1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy));
    }
    /// <summary>Convertit en matrice 4x4 (rotation + translation nulle).</summary>
    public readonly Matrix4x4D ToMatrix4x4D() { Tensor3D r = ToTensor3D(); return new Matrix4x4D(r.M11, r.M12, r.M13, 0, r.M21, r.M22, r.M23, 0, r.M31, r.M32, r.M33, 0, 0, 0, 0, 1); }
    /// <summary>Renvoie les angles d'Euler (roll, pitch, yaw).</summary>
    public readonly Vector3D ToEuler()
    {
        double sr = 2.0 * (W * X + Y * Z), cr = 1.0 - 2.0 * (X * X + Y * Y);
        double roll = Math.Atan2(sr, cr);
        double sp = 2.0 * (W * Y - Z * X);
        double pitch = Math.Abs(sp) >= 1 ? Math.CopySign(Math.PI / 2.0, sp) : Math.Asin(sp);
        double sy = 2.0 * (W * Z + X * Y), cy = 1.0 - 2.0 * (Y * Y + Z * Z);
        double yaw = Math.Atan2(sy, cy);
        return new Vector3D(roll, pitch, yaw);
    }
    /// <summary>Renvoie l'axe et l'angle de rotation.</summary>
    public readonly void ToAxisAngle(out Vector3D axis, out double angle)
    {
        double len = Math.Sqrt(X * X + Y * Y + Z * Z);
        if (len < 1e-30)
        { axis = Vector3D.UnitY; angle = 0; return; }
        axis = new Vector3D(X / len, Y / len, Z / len);
        angle = 2.0 * Math.Atan2(len, W);
    }
    /// <summary>Angle de rotation en radians.</summary>
    public readonly double Angle => 2.0 * Math.Atan2(Math.Sqrt(X * X + Y * Y + Z * Z), W);

    /// <summary>Rotation d'un vecteur : q * v * q^-1. Methode optimisee (pas de produit quaternion).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector3D RotateVector(Vector3D v)
    {
        double vx = v.X, vy = v.Y, vz = v.Z, qw = W, qx = X, qy = Y, qz = Z;
        double tx = 2.0 * (qy * vz - qz * vy), ty = 2.0 * (qz * vx - qx * vz), tz = 2.0 * (qx * vy - qy * vx);
        return new Vector3D(vx + qw * tx + (qy * tz - qz * ty), vy + qw * ty + (qz * tx - qx * tz), vz + qw * tz + (qx * ty - qy * tx));
    }

    /// <summary>Spherical Linear Interpolation (SLERP). Preserves les angles.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static QuaternionD Slerp(QuaternionD a, QuaternionD b, double t)
    {
        double dot = a.W * b.W + a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        QuaternionD target = b;
        if (dot < 0)
        { dot = -dot; target = new QuaternionD(-b.W, -b.X, -b.Y, -b.Z); }
        if (dot > 0.9995)
        { double it = 1.0 - t; return new QuaternionD(it * a.W + t * target.W, it * a.X + t * target.X, it * a.Y + t * target.Y, it * a.Z + t * target.Z).Normalized(); }
        double theta = Math.Acos(dot), sinTheta = Math.Sin(theta);
        return a * (Math.Sin((1 - t) * theta) / sinTheta) + target * (Math.Sin(t * theta) / sinTheta);
    }
    /// <summary>Normalized Linear Interpolation (NLERP). Plus rapide mais moins precis.</summary>
    public static QuaternionD Nlerp(QuaternionD a, QuaternionD b, double t) { double dot = a.W * b.W + a.X * b.X + a.Y * b.Y + a.Z * b.Z; double sign = dot < 0 ? -1 : 1; double it = 1.0 - t; return new QuaternionD(it * a.W + t * sign * b.W, it * a.X + t * sign * b.X, it * a.Y + t * sign * b.Y, it * a.Z + t * sign * b.Z).Normalized(); }

    /// <summary>Produit de deux quaternions (composition de rotations).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static QuaternionD operator *(QuaternionD a, QuaternionD b) =>
        new(a.W * b.W - a.X * b.X - a.Y * b.Y - a.Z * b.Z,
            a.W * b.X + a.X * b.W + a.Y * b.Z - a.Z * b.Y,
            a.W * b.Y - a.X * b.Z + a.Y * b.W + a.Z * b.X,
            a.W * b.Z + a.X * b.Y - a.Y * b.X + a.Z * b.W);
    /// <summary>Multiplication par un scalaire (slerp implicite).</summary>
    public static QuaternionD operator *(QuaternionD q, double s) => new QuaternionD(q.W * s, q.X * s, q.Y * s, q.Z * s);
    public static QuaternionD operator *(double s, QuaternionD q) => q * s;
    public static QuaternionD operator +(QuaternionD a, QuaternionD b) => new(a.W + b.W, a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static QuaternionD operator -(QuaternionD a) => new(-a.W, -a.X, -a.Y, -a.Z);
    public static QuaternionD operator -(QuaternionD a, QuaternionD b) => new(a.W - b.W, a.X - b.X, a.Y - b.Y, a.Z - b.Z);

    /// <summary>Difference relative : b * a^-1.</summary>
    public static QuaternionD Difference(QuaternionD a, QuaternionD b) => a.Inverse() * b;
    /// <summary>Logarithme quaternion (pour interpolations sur l'espace tangent).</summary>
    public readonly QuaternionD Log() { double len = Math.Sqrt(X * X + Y * Y + Z * Z); if (len < 1e-30) return Zero; double ha = Math.Atan2(len, W), s = ha / len; return new QuaternionD(0, X * s, Y * s, Z * s); }
    /// <summary>Exponentielle quaternion.</summary>
    public static QuaternionD Exp(QuaternionD q) { double len = Math.Sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z); if (len < 1e-30) return Identity; double s = Math.Sin(len) / len; return new QuaternionD(Math.Cos(len), q.X * s, q.Y * s, q.Z * s); }
    /// <summary>Retourne le quaternion le plus proche (evite les problemes de signe).</summary>
    public readonly QuaternionD ClosestTo(QuaternionD other) => (W * other.W + X * other.X + Y * other.Y + Z * other.Z) < 0 ? new QuaternionD(-W, -X, -Y, -Z) : this;

    public readonly bool Equals(QuaternionD o) => Math.Abs(W - o.W) < 1e-10 && Math.Abs(X - o.X) < 1e-10 && Math.Abs(Y - o.Y) < 1e-10 && Math.Abs(Z - o.Z) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is QuaternionD o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(W, X, Y, Z);
    public static bool operator ==(QuaternionD a, QuaternionD b) => a.Equals(b);
    public static bool operator !=(QuaternionD a, QuaternionD b) => !a.Equals(b);
    public override readonly string ToString() => $"({W:F4}, {X:F4}, {Y:F4}, {Z:F4})";
}

// ═══════════════════════════════════════════════════════════════════════════════
