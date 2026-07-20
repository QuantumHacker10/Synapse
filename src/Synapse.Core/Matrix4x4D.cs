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

// SECTION 3.5: MATRIX4X4D — MATRICE 4x4 DOUBLE PRECISION
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Matrice 4x4 double precision pour les transformations homogenes.
/// Utilisee pour les transformations de modele, vue, projection et les
/// operations dans l'espace projectif. Stockage en colonnes (OpenGL convention).
///
/// Les methodes statiques generent les matrices standard : translation, rotation,
/// mise a l'echelle, projection perspective/orthographique, LookAt.
///
/// MEMORY LAYOUT: 128 bytes (16 doubles).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 128, Pack = 8)]
[DebuggerDisplay("[{M11:F3},{M12:F3},{M13:F3},{M14:F3} | {M21:F3},{M22:F3},{M23:F3},{M24:F3} | {M31:F3},{M32:F3},{M33:F3},{M34:F3} | {M41:F3},{M42:F3},{M43:F3},{M44:F3}]")]
public struct Matrix4x4D : IEquatable<Matrix4x4D>
{
    [FieldOffset(0)] public double M11; [FieldOffset(8)] public double M12; [FieldOffset(16)] public double M13; [FieldOffset(24)] public double M14;
    [FieldOffset(32)] public double M21; [FieldOffset(40)] public double M22; [FieldOffset(48)] public double M23; [FieldOffset(56)] public double M24;
    [FieldOffset(64)] public double M31; [FieldOffset(72)] public double M32; [FieldOffset(80)] public double M33; [FieldOffset(88)] public double M34;
    [FieldOffset(96)] public double M41; [FieldOffset(104)] public double M42; [FieldOffset(112)] public double M43; [FieldOffset(120)] public double M44;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix4x4D(double m11, double m12, double m13, double m14, double m21, double m22, double m23, double m24, double m31, double m32, double m33, double m34, double m41, double m42, double m43, double m44)
    { M11 = m11; M12 = m12; M13 = m13; M14 = m14; M21 = m21; M22 = m22; M23 = m23; M24 = m24; M31 = m31; M32 = m32; M33 = m33; M34 = m34; M41 = m41; M42 = m42; M43 = m43; M44 = m44; }

    public static readonly Matrix4x4D Zero = default;
    public static readonly Matrix4x4D Identity = new(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

    public static Matrix4x4D Translation(double tx, double ty, double tz) => new(1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1);
    public static Matrix4x4D Translation(Vector3D t) => Translation(t.X, t.Y, t.Z);
    public static Matrix4x4D Scaling(double sx, double sy, double sz) => new(sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1);
    public static Matrix4x4D Scaling(Vector3D s) => Scaling(s.X, s.Y, s.Z);
    public static Matrix4x4D RotationX(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1); }
    public static Matrix4x4D RotationY(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1); }
    public static Matrix4x4D RotationZ(double a) { double c = Math.Cos(a), s = Math.Sin(a); return new(c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1); }
    public static Matrix4x4D RotationAxis(Vector3D ax, double a)
    {
        Vector3D u = ax.Normalized();
        double c = Math.Cos(a), s = Math.Sin(a), t = 1 - c;
        return new(t * u.X * u.X + c, t * u.X * u.Y - s * u.Z, t * u.X * u.Z + s * u.Y, 0,
            t * u.X * u.Y + s * u.Z, t * u.Y * u.Y + c, t * u.Y * u.Z - s * u.X, 0,
            t * u.X * u.Z - s * u.Y, t * u.Y * u.Z + s * u.X, t * u.Z * u.Z + c, 0, 0, 0, 0, 1);
    }
    public static Matrix4x4D FromQuaternion(QuaternionD q) => q.ToMatrix4x4D();
    public static Matrix4x4D LookAt(Vector3D eye, Vector3D target, Vector3D up)
    {
        Vector3D f = (target - eye).Normalized(), r = Vector3D.Cross(f, up).Normalized(), u = Vector3D.Cross(r, f);
        return new Matrix4x4D(r.X, r.Y, r.Z, -Vector3D.Dot(r, eye), u.X, u.Y, u.Z, -Vector3D.Dot(u, eye), -f.X, -f.Y, -f.Z, Vector3D.Dot(f, eye), 0, 0, 0, 1);
    }
    public static Matrix4x4D Perspective(double fovyRad, double aspect, double near, double far)
    {
        double t = Math.Tan(fovyRad * 0.5);
        return new Matrix4x4D(1.0 / (aspect * t), 0, 0, 0, 0, 1.0 / t, 0, 0, 0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near), 0, 0, -1, 0);
    }
    public static Matrix4x4D InfinitePerspective(double fovyRad, double aspect, double near)
    {
        double t = Math.Tan(fovyRad * 0.5);
        return new Matrix4x4D(1.0 / (aspect * t), 0, 0, 0, 0, 1.0 / t, 0, 0, 0, 0, -1, -2 * near, 0, 0, -1, 0);
    }
    public static Matrix4x4D Orthographic(double l, double r, double b, double t, double n, double f)
    {
        double rl = r - l, tb = t - b, fn = f - n;
        return new Matrix4x4D(2.0 / rl, 0, 0, -(r + l) / rl, 0, 2.0 / tb, 0, -(t + b) / tb, 0, 0, -2.0 / fn, -(f + n) / fn, 0, 0, 0, 1);
    }

    public readonly double Trace => M11 + M22 + M33 + M44;
    public readonly double Determinant
    {
        get
        {
            double a = M11 * (M22 * (M33 * M44 - M34 * M43) - M23 * (M32 * M44 - M34 * M42) + M24 * (M32 * M43 - M33 * M42));
            double b = M12 * (M21 * (M33 * M44 - M34 * M43) - M23 * (M31 * M44 - M34 * M41) + M24 * (M31 * M43 - M33 * M41));
            double c = M13 * (M21 * (M32 * M44 - M34 * M42) - M22 * (M31 * M44 - M34 * M41) + M24 * (M31 * M42 - M32 * M41));
            double d = M14 * (M21 * (M32 * M43 - M33 * M42) - M22 * (M31 * M43 - M33 * M41) + M23 * (M31 * M42 - M32 * M41));
            return a - b + c - d;
        }
    }
    public readonly Matrix4x4D Transpose() => new(M11, M21, M31, M41, M12, M22, M32, M42, M13, M23, M33, M43, M14, M24, M34, M44);
    public readonly Matrix4x4D Inverse()
    {
        double det = Determinant;
        if (Math.Abs(det) < 1e-30)
            return Zero;
        double inv = 1.0 / det;
        double c11 = (M22 * (M33 * M44 - M34 * M43) - M23 * (M32 * M44 - M34 * M42) + M24 * (M32 * M43 - M33 * M42));
        double c12 = -(M21 * (M33 * M44 - M34 * M43) - M23 * (M31 * M44 - M34 * M41) + M24 * (M31 * M43 - M33 * M41));
        double c13 = (M21 * (M32 * M44 - M34 * M42) - M22 * (M31 * M44 - M34 * M41) + M24 * (M31 * M42 - M32 * M41));
        double c14 = -(M21 * (M32 * M43 - M33 * M42) - M22 * (M31 * M43 - M33 * M41) + M23 * (M31 * M42 - M32 * M41));
        double c21 = -(M12 * (M33 * M44 - M34 * M43) - M13 * (M32 * M44 - M34 * M42) + M14 * (M32 * M43 - M33 * M42));
        double c22 = (M11 * (M33 * M44 - M34 * M43) - M13 * (M31 * M44 - M34 * M41) + M14 * (M31 * M43 - M33 * M41));
        double c23 = -(M11 * (M32 * M44 - M34 * M42) - M12 * (M31 * M44 - M34 * M41) + M14 * (M31 * M42 - M32 * M41));
        double c24 = (M11 * (M32 * M43 - M33 * M42) - M12 * (M31 * M43 - M33 * M41) + M13 * (M31 * M42 - M32 * M41));
        double c31 = (M12 * (M23 * M44 - M24 * M43) - M13 * (M22 * M44 - M24 * M42) + M14 * (M22 * M43 - M23 * M42));
        double c32 = -(M11 * (M23 * M44 - M24 * M43) - M13 * (M21 * M44 - M24 * M41) + M14 * (M21 * M43 - M23 * M41));
        double c33 = (M11 * (M22 * M44 - M24 * M42) - M12 * (M21 * M44 - M24 * M41) + M14 * (M21 * M42 - M22 * M41));
        double c34 = -(M11 * (M22 * M43 - M23 * M42) - M12 * (M21 * M43 - M23 * M41) + M13 * (M21 * M42 - M22 * M41));
        double c41 = -(M12 * (M23 * M34 - M24 * M33) - M13 * (M22 * M34 - M24 * M32) + M14 * (M22 * M33 - M23 * M32));
        double c42 = (M11 * (M23 * M34 - M24 * M33) - M13 * (M21 * M34 - M24 * M31) + M14 * (M21 * M33 - M23 * M31));
        double c43 = -(M11 * (M22 * M34 - M24 * M32) - M12 * (M21 * M34 - M24 * M31) + M14 * (M21 * M32 - M22 * M31));
        double c44 = (M11 * (M22 * M33 - M23 * M32) - M12 * (M21 * M33 - M23 * M31) + M13 * (M21 * M32 - M22 * M31));
        return new Matrix4x4D(c11 * inv, c21 * inv, c31 * inv, c41 * inv, c12 * inv, c22 * inv, c32 * inv, c42 * inv, c13 * inv, c23 * inv, c33 * inv, c43 * inv, c14 * inv, c24 * inv, c34 * inv, c44 * inv);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly Vector3D TransformPoint(Vector3D p) { double w = M41 * p.X + M42 * p.Y + M43 * p.Z + M44; if (Math.Abs(w) < 1e-30) w = 1e-30; return new((M11 * p.X + M12 * p.Y + M13 * p.Z + M14) / w, (M21 * p.X + M22 * p.Y + M23 * p.Z + M24) / w, (M31 * p.X + M32 * p.Y + M33 * p.Z + M34) / w); }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] public readonly Vector3D TransformVector(Vector3D v) => new(M11 * v.X + M12 * v.Y + M13 * v.Z, M21 * v.X + M22 * v.Y + M23 * v.Z, M31 * v.X + M32 * v.Y + M33 * v.Z);
    public readonly Tensor3D ToTensor3D() => new(M11, M12, M13, M21, M22, M23, M31, M32, M33);
    public readonly Vector3D GetTranslation() => new(M14, M24, M34);
    public readonly QuaternionD GetRotation() => QuaternionD.FromMatrix3x3(ToTensor3D());
    public readonly Vector3D GetScale() => new(ToTensor3D().Column(0).Length(), ToTensor3D().Column(1).Length(), ToTensor3D().Column(2).Length());
    public readonly void Decompose(out Vector3D translation, out QuaternionD rotation, out Vector3D scale)
    {
        translation = GetTranslation();
        scale = GetScale();
        Tensor3D rot = ToTensor3D();
        if (scale.X > 1e-10)
        { rot.M11 /= scale.X; rot.M12 /= scale.X; rot.M13 /= scale.X; }
        if (scale.Y > 1e-10)
        { rot.M21 /= scale.Y; rot.M22 /= scale.Y; rot.M23 /= scale.Y; }
        if (scale.Z > 1e-10)
        { rot.M31 /= scale.Z; rot.M32 /= scale.Z; rot.M33 /= scale.Z; }
        rotation = QuaternionD.FromMatrix3x3(rot);
    }

    public readonly double FrobeniusNorm => Math.Sqrt(M11 * M11 + M12 * M12 + M13 * M13 + M14 * M14 + M21 * M21 + M22 * M22 + M23 * M23 + M24 * M24 + M31 * M31 + M32 * M32 + M33 * M33 + M34 * M34 + M41 * M41 + M42 * M42 + M43 * M43 + M44 * M44);

    public static Matrix4x4D operator *(Matrix4x4D a, Matrix4x4D b) { Matrix4x4D r; r.M11 = a.M11 * b.M11 + a.M12 * b.M21 + a.M13 * b.M31 + a.M14 * b.M41; r.M12 = a.M11 * b.M12 + a.M12 * b.M22 + a.M13 * b.M32 + a.M14 * b.M42; r.M13 = a.M11 * b.M13 + a.M12 * b.M23 + a.M13 * b.M33 + a.M14 * b.M43; r.M14 = a.M11 * b.M14 + a.M12 * b.M24 + a.M13 * b.M34 + a.M14 * b.M44; r.M21 = a.M21 * b.M11 + a.M22 * b.M21 + a.M23 * b.M31 + a.M24 * b.M41; r.M22 = a.M21 * b.M12 + a.M22 * b.M22 + a.M23 * b.M32 + a.M24 * b.M42; r.M23 = a.M21 * b.M13 + a.M22 * b.M23 + a.M23 * b.M33 + a.M24 * b.M43; r.M24 = a.M21 * b.M14 + a.M22 * b.M24 + a.M23 * b.M34 + a.M24 * b.M44; r.M31 = a.M31 * b.M11 + a.M32 * b.M21 + a.M33 * b.M31 + a.M34 * b.M41; r.M32 = a.M31 * b.M12 + a.M32 * b.M22 + a.M33 * b.M32 + a.M34 * b.M42; r.M33 = a.M31 * b.M13 + a.M32 * b.M23 + a.M33 * b.M33 + a.M34 * b.M43; r.M34 = a.M31 * b.M14 + a.M32 * b.M24 + a.M33 * b.M34 + a.M34 * b.M44; r.M41 = a.M41 * b.M11 + a.M42 * b.M21 + a.M43 * b.M31 + a.M44 * b.M41; r.M42 = a.M41 * b.M12 + a.M42 * b.M22 + a.M43 * b.M32 + a.M44 * b.M42; r.M43 = a.M41 * b.M13 + a.M42 * b.M23 + a.M43 * b.M33 + a.M44 * b.M43; r.M44 = a.M41 * b.M14 + a.M42 * b.M24 + a.M43 * b.M34 + a.M44 * b.M44; return r; }
    public static Matrix4x4D operator *(Matrix4x4D m, double s) => new(m.M11 * s, m.M12 * s, m.M13 * s, m.M14 * s, m.M21 * s, m.M22 * s, m.M23 * s, m.M24 * s, m.M31 * s, m.M32 * s, m.M33 * s, m.M34 * s, m.M41 * s, m.M42 * s, m.M43 * s, m.M44 * s);
    public static Matrix4x4D operator *(double s, Matrix4x4D m) => m * s;
    public static Matrix4x4D operator +(Matrix4x4D a, Matrix4x4D b) => new(a.M11 + b.M11, a.M12 + b.M12, a.M13 + b.M13, a.M14 + b.M14, a.M21 + b.M21, a.M22 + b.M22, a.M23 + b.M23, a.M24 + b.M24, a.M31 + b.M31, a.M32 + b.M32, a.M33 + b.M33, a.M34 + b.M34, a.M41 + b.M41, a.M42 + b.M42, a.M43 + b.M43, a.M44 + b.M44);
    public static Matrix4x4D operator -(Matrix4x4D a, Matrix4x4D b) => new(a.M11 - b.M11, a.M12 - b.M12, a.M13 - b.M13, a.M14 - b.M14, a.M21 - b.M21, a.M22 - b.M22, a.M23 - b.M23, a.M24 - b.M24, a.M31 - b.M31, a.M32 - b.M32, a.M33 - b.M33, a.M34 - b.M34, a.M41 - b.M41, a.M42 - b.M42, a.M43 - b.M43, a.M44 - b.M44);
    public static Matrix4x4D operator -(Matrix4x4D m) => new(-m.M11, -m.M12, -m.M13, -m.M14, -m.M21, -m.M22, -m.M23, -m.M24, -m.M31, -m.M32, -m.M33, -m.M34, -m.M41, -m.M42, -m.M43, -m.M44);

    public readonly bool Equals(Matrix4x4D o) => Math.Abs(M11 - o.M11) < 1e-10 && Math.Abs(M12 - o.M12) < 1e-10 && Math.Abs(M13 - o.M13) < 1e-10 && Math.Abs(M14 - o.M14) < 1e-10 && Math.Abs(M21 - o.M21) < 1e-10 && Math.Abs(M22 - o.M22) < 1e-10 && Math.Abs(M23 - o.M23) < 1e-10 && Math.Abs(M24 - o.M24) < 1e-10 && Math.Abs(M31 - o.M31) < 1e-10 && Math.Abs(M32 - o.M32) < 1e-10 && Math.Abs(M33 - o.M33) < 1e-10 && Math.Abs(M34 - o.M34) < 1e-10 && Math.Abs(M41 - o.M41) < 1e-10 && Math.Abs(M42 - o.M42) < 1e-10 && Math.Abs(M43 - o.M43) < 1e-10 && Math.Abs(M44 - o.M44) < 1e-10;
    public override readonly bool Equals(object? obj) => obj is Matrix4x4D o && Equals(o);
    public override readonly int GetHashCode() => HashCode.Combine(M11, M22, M33, M44);
    public static bool operator ==(Matrix4x4D a, Matrix4x4D b) => a.Equals(b);
    public static bool operator !=(Matrix4x4D a, Matrix4x4D b) => !a.Equals(b);
    public override readonly string ToString() => $"[{M11:F3} {M12:F3} {M13:F3} {M14:F3}]\n[{M21:F3} {M22:F3} {M23:F3} {M24:F3}]\n[{M31:F3} {M32:F3} {M33:F3} {M34:F3}]\n[{M41:F3} {M42:F3} {M43:F3} {M44:F3}]";
}

// ═══════════════════════════════════════════════════════════════════════════════
