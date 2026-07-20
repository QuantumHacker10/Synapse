// ============================================================================
// Synapse Omnia — Physics Solvers
// Complete implementations of electromagnetic, acoustic, thermodynamic,
// chemical, gravitational, lattice-Boltzmann, quantum, elastic, turbulent,
// and multiphysics solvers.
//
// C# 14 · unsafe · NativeAOT compatible
// ============================================================================

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace Synapse.Physics;

public static class CoordinateTransforms
{
    /// <summary>Cylindrical (r, θ, z) → Cartesian (x, y, z).</summary>
    public static void CylindricalToCartesian(double r, double theta, double z,
        out double x, out double y, out double cz)
    {
        x = r * Math.Cos(theta);
        y = r * Math.Sin(theta);
        cz = z;
    }

    /// <summary>Cartesian (x, y, z) → Cylindrical (r, θ, z).</summary>
    public static void CartesianToCylindrical(double x, double y, double z,
        out double r, out double theta, out double cz)
    {
        r = Math.Sqrt(x * x + y * y);
        theta = Math.Atan2(y, x);
        cz = z;
    }

    /// <summary>Spherical (r, θ, φ) → Cartesian (x, y, z). θ = polar, φ = azimuthal.</summary>
    public static void SphericalToCartesian(double r, double theta, double phi,
        out double x, out double y, out double z)
    {
        x = r * Math.Sin(theta) * Math.Cos(phi);
        y = r * Math.Sin(theta) * Math.Sin(phi);
        z = r * Math.Cos(theta);
    }

    /// <summary>Cartesian (x, y, z) → Spherical (r, θ, φ).</summary>
    public static void CartesianToSpherical(double x, double y, double z,
        out double r, out double theta, out double phi)
    {
        r = Math.Sqrt(x * x + y * y + z * z);
        theta = Math.Acos(Math.Clamp(z / Math.Max(r, 1e-30), -1.0, 1.0));
        phi = Math.Atan2(y, x);
    }

    /// <summary>
    /// Rotate a vector (vx, vy, vz) around axis (ax, ay, az) by angle θ radians.
    /// Uses Rodrigues' rotation formula.
    /// </summary>
    public static void RotateVector(
        double vx, double vy, double vz,
        double ax, double ay, double az, double theta,
        out double rx, out double ry, out double rz)
    {
        double len = Math.Sqrt(ax * ax + ay * ay + az * az);
        if (len < 1e-30)
        { rx = vx; ry = vy; rz = vz; return; }
        ax /= len;
        ay /= len;
        az /= len;

        double cosT = Math.Cos(theta), sinT = Math.Sin(theta);
        double dot = ax * vx + ay * vy + az * vz;
        double crossX = ay * vz - az * vy;
        double crossY = az * vx - ax * vz;
        double crossZ = ax * vy - ay * vx;

        rx = vx * cosT + crossX * sinT + ax * dot * (1.0 - cosT);
        ry = vy * cosT + crossY * sinT + ay * dot * (1.0 - cosT);
        rz = vz * cosT + crossZ * sinT + az * dot * (1.0 - cosT);
    }

    /// <summary>
    /// Compute a rotation matrix for rotation around an arbitrary axis.
    /// Returns 3×3 matrix in row-major order.
    /// </summary>
    public static double[] RotationMatrix(double ax, double ay, double az, double theta)
    {
        double len = Math.Sqrt(ax * ax + ay * ay + az * az);
        if (len < 1e-30)
            return new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        ax /= len;
        ay /= len;
        az /= len;

        double c = Math.Cos(theta), s = Math.Sin(theta), t = 1.0 - c;
        return new double[] {
            t * ax * ax + c,       t * ax * ay - s * az,  t * ax * az + s * ay,
            t * ax * ay + s * az,  t * ay * ay + c,       t * ay * az - s * ax,
            t * ax * az - s * ay,  t * ay * az + s * ax,  t * az * az + c
        };
    }
}

// ============================================================================
//  Additional Utility: Signal Processing
// ============================================================================

/// <summary>
/// Digital signal processing utilities for physics post-processing.
/// </summary>
