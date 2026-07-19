using System;
// ============================================================
// FILE: QuaternionMath.cs
// PATH: Core/Mathematics/QuaternionMath.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Core.Mathematics;

/// <summary>
/// Comprehensive quaternion mathematics utilities for the G-DNN neural geometry engine.
/// Provides extension methods for System.Numerics.Quaternion with interpolation, decomposition,
/// averaging, exponential/logarithm operations, and batch processing capabilities.
/// </summary>
public static class QuaternionMath
{
    private const float PI = MathF.PI;
    private const float TwoPI = MathF.Tau;
    private const float HalfPI = MathF.PI * 0.5f;
    private const float Epsilon = 1e-6f;

    #region Spherical Interpolation

    /// <summary>
    /// Spherical linear interpolation between two quaternions.
    /// Always takes the shortest path.
    /// </summary>
    public static Quaternion SlerpShortest(this Quaternion a, Quaternion b, float t)
    {
        float dot = Quaternion.Dot(a, b);

        if (dot < 0)
        {
            b = -b;
            dot = -dot;
        }

        if (dot > 0.9995f)
            return Quaternion.Normalize(Quaternion.Lerp(a, b, t));

        float theta = MathF.Acos(Math.Clamp(dot, -1f, 1f));
        float sinTheta = MathF.Sin(theta);

        float wa = MathF.Sin((1f - t) * theta) / sinTheta;
        float wb = MathF.Sin(t * theta) / sinTheta;

        return Quaternion.Normalize(new Quaternion(
            wa * a.X + wb * b.X,
            wa * a.Y + wb * b.Y,
            wa * a.Z + wb * b.Z,
            wa * a.W + wb * b.W));
    }

    /// <summary>
    /// Normalized linear interpolation between two quaternions.
    /// Faster than SLERP but less accurate for large angles.
    /// Always takes the shortest path.
    /// </summary>
    public static Quaternion NlerpShortest(this Quaternion a, Quaternion b, float t)
    {
        if (Quaternion.Dot(a, b) < 0)
            b = -b;

        return Quaternion.Normalize(Quaternion.Lerp(a, b, t));
    }

    /// <summary>
    /// Spherical quadratic interpolation using an intermediate quaternion.
    /// </summary>
    public static Quaternion Squad(this Quaternion a, Quaternion b, Quaternion c, Quaternion d, float t)
    {
        Quaternion ab = SlerpShortest(a, b, t);
        Quaternion bc = SlerpShortest(b, c, t);
        Quaternion cd = SlerpShortest(c, d, t);
        Quaternion abc = SlerpShortest(ab, bc, t);
        Quaternion bcd = SlerpShortest(bc, cd, t);
        return SlerpShortest(abc, bcd, t);
    }

    /// <summary>
    /// Spherical cubic interpolation between four quaternions.
    /// </summary>
    public static Quaternion Slerp4(ReadOnlySpan<Quaternion> quats, ReadOnlySpan<float> weights)
    {
        Debug.Assert(quats.Length == weights.Length && quats.Length >= 2);

        Quaternion result = quats[0] * weights[0];
        for (int i = 1; i < quats.Length; i++)
            result = SlerpShortest(result, quats[i], weights[i]);

        return Quaternion.Normalize(result);
    }

    /// <summary>
    /// Interpolate between two rotations with constant angular velocity.
    /// </summary>
    public static Quaternion ConstantVelocitySlerp(this Quaternion from, Quaternion to, float t, float angularVelocity)
    {
        float dot = Quaternion.Dot(from, to);
        if (dot < 0)
        { to = -to; dot = -dot; }

        float angle = MathF.Acos(Math.Clamp(dot, -1f, 1f));
        if (angle < Epsilon)
            return from;

        float totalAngle = angle;
        float adjustedT = MathF.Min(1f, angularVelocity * t / totalAngle);

        return SlerpShortest(from, to, adjustedT);
    }

    #endregion

    #region Rotation Construction

    /// <summary>
    /// Create a quaternion from axis-angle representation.
    /// </summary>
    public static Quaternion FromAxisAngle(Vector3 axis, float angleRad)
    {
        return Quaternion.CreateFromAxisAngle(Vector3.Normalize(axis), angleRad);
    }

    /// <summary>
    /// Create a quaternion from Euler angles in radians (pitch, yaw, roll).
    /// </summary>
    public static Quaternion FromEulerAngles(float pitch, float yaw, float roll) =>
        Quaternion.CreateFromYawPitchRoll(yaw, pitch, roll);

    /// <summary>
    /// Create a quaternion from Euler angles vector in radians.
    /// </summary>
    public static Quaternion FromEulerVector(Vector3 eulerAngles) =>
        Quaternion.CreateFromYawPitchRoll(eulerAngles.Y, eulerAngles.X, eulerAngles.Z);

    /// <summary>
    /// Create a quaternion from a rotation matrix.
    /// </summary>
    public static Quaternion FromRotationMatrix(Matrix4x4 matrix) =>
        Quaternion.CreateFromRotationMatrix(matrix);

    /// <summary>
    /// Create a quaternion that looks in the 'forward' direction with a given 'up' vector.
    /// </summary>
    public static Quaternion LookRotation(Vector3 forward, Vector3? up = null)
    {
        Vector3 upVec = up ?? Vector3.UnitY;
        Vector3 zAxis = Vector3.Normalize(forward);
        Vector3 xAxis = Vector3.Normalize(Vector3.Cross(upVec, zAxis));
        Vector3 yAxis = Vector3.Cross(zAxis, xAxis);

        Matrix4x4 m = new Matrix4x4(
            xAxis.X, xAxis.Y, xAxis.Z, 0,
            yAxis.X, yAxis.Y, yAxis.Z, 0,
            zAxis.X, zAxis.Y, zAxis.Z, 0,
            0, 0, 0, 1);

        return Quaternion.CreateFromRotationMatrix(m);
    }

    /// <summary>
    /// Create a quaternion from two direction vectors (rotation from 'from' to 'to').
    /// </summary>
    public static Quaternion FromToRotation(Vector3 from, Vector3 to)
    {
        from = Vector3.Normalize(from);
        to = Vector3.Normalize(to);

        float dot = Vector3.Dot(from, to);

        if (dot > 1f - Epsilon)
            return Quaternion.Identity;

        if (dot < -1f + Epsilon)
        {
            Vector3 ortho = MathF.Abs(from.X) < 0.9f
                ? Vector3.Cross(from, Vector3.UnitX)
                : Vector3.Cross(from, Vector3.UnitY);
            return Quaternion.CreateFromAxisAngle(Vector3.Normalize(ortho), PI);
        }

        Vector3 axis = Vector3.Cross(from, to);
        float angle = MathF.Acos(Math.Clamp(dot, -1f, 1f));

        return Quaternion.CreateFromAxisAngle(axis, angle);
    }

    /// <summary>
    /// Create a rotation that aligns 'source' direction to 'target' direction,
    /// constrained to rotate around the specified 'axis'.
    /// </summary>
    public static Quaternion RotateAroundAxis(Vector3 source, Vector3 target, Vector3 axis)
    {
        source = Vector3.Normalize(source);
        target = Vector3.Normalize(target);
        axis = Vector3.Normalize(axis);

        Vector3 projSource = source - axis * Vector3.Dot(axis, source);
        Vector3 projTarget = target - axis * Vector3.Dot(axis, target);

        projSource = Vector3.Normalize(projSource);
        projTarget = Vector3.Normalize(projTarget);

        float dot = Vector3.Dot(projSource, projTarget);
        if (dot > 1f - Epsilon)
            return Quaternion.Identity;
        if (dot < -1f + Epsilon)
            return Quaternion.CreateFromAxisAngle(axis, PI);

        Vector3 cross = Vector3.Cross(projSource, projTarget);
        float angle = MathF.Acos(Math.Clamp(dot, -1f, 1f));

        return Quaternion.CreateFromAxisAngle(axis, angle);
    }

    #endregion

    #region Decomposition

    /// <summary>
    /// Decompose a quaternion into axis and angle.
    /// </summary>
    public static void ToAxisAngle(this Quaternion q, out Vector3 axis, out float angle)
    {
        q = Quaternion.Normalize(q);
        angle = 2f * MathF.Acos(Math.Clamp(MathF.Abs(q.W), 0f, 1f));

        float s = MathF.Sqrt(1f - q.W * q.W);
        if (s < Epsilon)
        {
            axis = Vector3.UnitX;
            return;
        }

        axis = new Vector3(q.X, q.Y, q.Z) / s;
    }

    /// <summary>
    /// Convert a quaternion to Euler angles in radians (pitch, yaw, roll).
    /// </summary>
    public static Vector3 ToEulerAngles(this Quaternion q)
    {
        float sinr_cosp = 2f * (q.W * q.X + q.Y * q.Z);
        float cosr_cosp = 1f - 2f * (q.X * q.X + q.Y * q.Y);
        float roll = MathF.Atan2(sinr_cosp, cosr_cosp);

        float sinp = 2f * (q.W * q.Y - q.Z * q.X);
        float pitch = MathF.Abs(sinp) >= 1f
            ? MathF.CopySign(PI / 2f, sinp)
            : MathF.Asin(sinp);

        float siny_cosp = 2f * (q.W * q.Z + q.X * q.Y);
        float cosy_cosp = 1f - 2f * (q.Y * q.Y + q.Z * q.Z);
        float yaw = MathF.Atan2(siny_cosp, cosy_cosp);

        return new Vector3(pitch, yaw, roll);
    }

    /// <summary>
    /// Convert a quaternion to a rotation matrix.
    /// </summary>
    public static Matrix4x4 ToRotationMatrix(this Quaternion q) =>
        Matrix4x4.CreateFromQuaternion(Quaternion.Normalize(q));

    /// <summary>
    /// Decompose a rotation into swing and twist components around a given twist axis.
    /// Swing handles the off-axis rotation, twist handles the around-axis rotation.
    /// </summary>
    public static void SwingTwistDecompose(this Quaternion rotation, Vector3 twistAxis,
        out Quaternion swing, out Quaternion twist)
    {
        Vector3 rAxis = new(rotation.X, rotation.Y, rotation.Z);
        float rW = rotation.W;

        Vector3 projection = twistAxis * Vector3.Dot(rAxis, twistAxis);
        twist = Quaternion.Normalize(new Quaternion(projection.X, projection.Y, projection.Z, rW));
        swing = rotation * Quaternion.Inverse(twist);
    }

    /// <summary>
    /// Extract the rotation component from a matrix and decompose into swing-twist.
    /// </summary>
    public static void SwingTwistFromMatrix(Matrix4x4 matrix, Vector3 twistAxis,
        out Quaternion swing, out Quaternion twist)
    {
        Quaternion rotation = Quaternion.CreateFromRotationMatrix(matrix);
        rotation.SwingTwistDecompose(twistAxis, out swing, out twist);
    }

    /// <summary>
    /// Decompose a quaternion into a local rotation and a twist around a given axis.
    /// Useful for separating head tilt from turn in character animation.
    /// </summary>
    public static void DecomposeLocalWorld(this Quaternion worldRotation, Quaternion parentRotation,
        out Quaternion localRotation, out Quaternion worldTwist, out Quaternion worldSwing)
    {
        localRotation = Quaternion.Normalize(Quaternion.Inverse(parentRotation) * worldRotation);
        localRotation.SwingTwistDecompose(Vector3.UnitY, out worldSwing, out worldTwist);
    }

    /// <summary>
    /// Extract the twist component of a rotation around an axis.
    /// </summary>
    public static Quaternion ExtractTwist(this Quaternion rotation, Vector3 twistAxis)
    {
        Vector3 rAxis = new(rotation.X, rotation.Y, rotation.Z);
        Vector3 projection = twistAxis * Vector3.Dot(rAxis, twistAxis);
        return Quaternion.Normalize(new Quaternion(projection.X, projection.Y, projection.Z, rotation.W));
    }

    /// <summary>
    /// Extract the swing component of a rotation around an axis.
    /// </summary>
    public static Quaternion ExtractSwing(this Quaternion rotation, Vector3 twistAxis)
    {
        Quaternion twist = rotation.ExtractTwist(twistAxis);
        return Quaternion.Normalize(rotation * Quaternion.Inverse(twist));
    }

    /// <summary>
    /// Get the angle of rotation in radians.
    /// </summary>
    public static float GetAngle(this Quaternion q)
    {
        float dot = MathF.Min(MathF.Abs(q.W), 1f);
        return 2f * MathF.Acos(dot);
    }

    /// <summary>
    /// Get the rotation axis (normalized).
    /// </summary>
    public static Vector3 GetAxis(this Quaternion q)
    {
        float s = MathF.Sqrt(1f - q.W * q.W);
        if (s < Epsilon)
            return Vector3.UnitX;
        return new Vector3(q.X, q.Y, q.Z) / s;
    }

    #endregion

    #region Quaternion Averaging

    /// <summary>
    /// Average multiple quaternions using the iterative method.
    /// Suitable for animation blending.
    /// </summary>
    public static Quaternion Average(ReadOnlySpan<Quaternion> quaternions, ReadOnlySpan<float> weights)
    {
        Debug.Assert(quaternions.Length == weights.Length);
        Debug.Assert(quaternions.Length > 0);

        Quaternion result = quaternions[0] * weights[0];

        for (int iter = 0; iter < 16; iter++)
        {
            Quaternion avg = Quaternion.Zero;
            for (int i = 0; i < quaternions.Length; i++)
            {
                Quaternion diff = Quaternion.Inverse(result) * quaternions[i];
                avg += diff * weights[i];
            }
            result = Quaternion.Normalize(result * avg);
        }

        return result;
    }

    /// <summary>
    /// Average multiple quaternions with equal weights.
    /// </summary>
    public static Quaternion Average(ReadOnlySpan<Quaternion> quaternions)
    {
        Debug.Assert(quaternions.Length > 0);

        if (quaternions.Length == 1)
            return quaternions[0];

        Span<float> weights = stackalloc float[quaternions.Length];
        weights.Fill(1f / quaternions.Length);

        return Average(quaternions, weights);
    }

    /// <summary>
    /// Weighted average using the sum-of-quaternions method (fast but less accurate for large spreads).
    /// </summary>
    public static Quaternion WeightedAverageFast(ReadOnlySpan<Quaternion> quaternions, ReadOnlySpan<float> weights)
    {
        Debug.Assert(quaternions.Length == weights.Length);
        Debug.Assert(quaternions.Length > 0);

        Quaternion sum = Quaternion.Identity;
        for (int i = 0; i < quaternions.Length; i++)
        {
            Quaternion q = quaternions[i];
            if (Quaternion.Dot(q, sum) < 0)
                q = -q;
            sum += q * weights[i];
        }

        return Quaternion.Normalize(sum);
    }

    /// <summary>
    /// Blend multiple rotations using log-space interpolation.
    /// More accurate than naive averaging for large angular spreads.
    /// </summary>
    public static Quaternion LogAverage(ReadOnlySpan<Quaternion> quaternions, ReadOnlySpan<float> weights)
    {
        Debug.Assert(quaternions.Length == weights.Length);
        Debug.Assert(quaternions.Length > 0);

        Vector3 axisSum = Vector3.Zero;
        float angleSum = 0f;

        for (int i = 0; i < quaternions.Length; i++)
        {
            quaternions[i].ToAxisAngle(out Vector3 axis, out float angle);
            axisSum += axis * angle * weights[i];
            angleSum += angle * weights[i];
        }

        float avgAngle = angleSum;
        Vector3 avgAxis = axisSum.Length() > Epsilon ? Vector3.Normalize(axisSum) : Vector3.UnitY;

        return Quaternion.CreateFromAxisAngle(avgAxis, avgAngle);
    }

    #endregion

    #region Shortest Path Rotation

    /// <summary>
    /// Ensure the rotation takes the shortest path by flipping the quaternion if necessary.
    /// </summary>
    public static Quaternion ShortestPath(this Quaternion current, Quaternion target)
    {
        if (Quaternion.Dot(current, target) < 0)
            return -target;
        return target;
    }

    /// <summary>
    /// Ensure a sequence of quaternions takes the shortest path between consecutive frames.
    /// Modifies the array in-place.
    /// </summary>
    public static void EnsureShortestPath(Span<Quaternion> quaternions)
    {
        for (int i = 1; i < quaternions.Length; i++)
        {
            if (Quaternion.Dot(quaternions[i - 1], quaternions[i]) < 0)
                quaternions[i] = -quaternions[i];
        }
    }

    /// <summary>
    /// Find the nearest quaternion to 'target' from the set of equivalent representations.
    /// Quaternions q and -q represent the same rotation.
    /// </summary>
    public static Quaternion NearestEquivalent(this Quaternion current, Quaternion target)
    {
        float d1 = Quaternion.Dot(current, target);
        float d2 = Quaternion.Dot(current, -target);
        return d1 >= d2 ? target : -target;
    }

    #endregion

    #region Exponential and Logarithm

    /// <summary>
    /// Quaternion exponential: maps a pure quaternion (w=0) to a unit quaternion rotation.
    /// </summary>
    public static Quaternion QuaternionExp(this Quaternion q)
    {
        float angle = new Vector3(q.X, q.Y, q.Z).Length();
        if (angle < Epsilon)
            return Quaternion.Identity;

        float halfAngle = angle * 0.5f;
        float s = MathF.Sin(halfAngle) / angle;
        return new Quaternion(q.X * s, q.Y * s, q.Z * s, MathF.Cos(halfAngle));
    }

    /// <summary>
    /// Quaternion logarithm: maps a unit quaternion to a pure quaternion (w=0).
    /// </summary>
    public static Quaternion QuaternionLog(this Quaternion q)
    {
        q = Quaternion.Normalize(q);
        float angle = 2f * MathF.Acos(Math.Clamp(MathF.Abs(q.W), 0f, 1f));
        float s = MathF.Sqrt(1f - q.W * q.W);

        if (s < Epsilon)
            return Quaternion.Identity;

        return new Quaternion(q.X / s * angle, q.Y / s * angle, q.Z / s * angle, 0);
    }

    /// <summary>
    /// Quaternion power: raises a unit quaternion to a scalar power.
    /// Useful for scaling rotation speed.
    /// </summary>
    public static Quaternion QuaternionPower(this Quaternion q, float exponent)
    {
        q = Quaternion.Normalize(q);
        float angle = 2f * MathF.Acos(Math.Clamp(MathF.Abs(q.W), 0f, 1f));
        float newAngle = angle * exponent;

        float s = MathF.Sqrt(1f - q.W * q.W);
        if (s < Epsilon)
            return Quaternion.Identity;

        float newS = MathF.Sin(newAngle * 0.5f) / s;
        return new Quaternion(q.X * newS, q.Y * newS, q.Z * newS, MathF.Cos(newAngle * 0.5f));
    }

    /// <summary>
    /// Square root of a quaternion.
    /// </summary>
    public static Quaternion QuaternionSqrt(this Quaternion q)
    {
        q = Quaternion.Normalize(q);
        float angle = MathF.Acos(Math.Clamp(MathF.Abs(q.W), 0f, 1f));
        float halfAngle = angle * 0.5f;

        float s = MathF.Sqrt(1f - q.W * q.W);
        if (s < Epsilon)
            return Quaternion.Identity;

        float newS = MathF.Sin(halfAngle) / s;
        return new Quaternion(q.X * newS, q.Y * newS, q.Z * newS, MathF.Cos(halfAngle));
    }

    #endregion

    #region Additional Quaternion Utilities

    /// <summary>
    /// Invert a quaternion (conjugate for unit quaternions).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Quaternion InverseRotation(this Quaternion q) =>
        Quaternion.Inverse(q);

    /// <summary>
    /// Get the conjugate of a quaternion.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Quaternion Conjugate(this Quaternion q) =>
        new(-q.X, -q.Y, -q.Z, q.W);

    /// <summary>
    /// Compute the angular distance between two quaternions in radians.
    /// </summary>
    public static float AngularDistance(this Quaternion a, Quaternion b)
    {
        float dot = MathF.Abs(Quaternion.Dot(a, b));
        dot = MathF.Min(dot, 1f);
        return 2f * MathF.Acos(dot);
    }

    /// <summary>
    /// Scale a quaternion (non-uniform scaling not supported; returns normalized result).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Quaternion Scale(this Quaternion q, float scale) =>
        Quaternion.Normalize(new Quaternion(q.X * scale, q.Y * scale, q.Z * scale, q.W * scale));

    /// <summary>
    /// Clamp a quaternion's rotation angle to a maximum value.
    /// </summary>
    public static Quaternion ClampAngle(this Quaternion q, float maxAngle)
    {
        q = Quaternion.Normalize(q);
        float angle = q.GetAngle();
        if (angle <= maxAngle)
            return q;

        Vector3 axis = q.GetAxis();
        return Quaternion.CreateFromAxisAngle(axis, maxAngle);
    }

    /// <summary>
    /// Check if two quaternions are approximately equal.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool ApproximatelyEquals(this Quaternion a, Quaternion b, float tolerance = Epsilon)
    {
        float dot = MathF.Abs(Quaternion.Dot(a, b));
        return dot >= 1f - tolerance;
    }

    /// <summary>
    /// Normalize a quaternion safely (returns identity if magnitude is near zero).
    /// </summary>
    public static Quaternion SafeNormalize(this Quaternion q)
    {
        float mag = MathF.Sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W);
        if (mag < Epsilon)
            return Quaternion.Identity;
        return new Quaternion(q.X / mag, q.Y / mag, q.Z / mag, q.W / mag);
    }

    /// <summary>
    /// Compute the relative rotation from 'from' to 'to'.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Quaternion RelativeRotation(this Quaternion from, Quaternion to) =>
        Quaternion.Normalize(Quaternion.Inverse(from) * to);

    /// <summary>
    /// Unwarp a quaternion angle to the range [-PI, PI].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float UnwrapAngle(float angle)
    {
        angle = MathF.IEEERemainder(angle, TwoPI);
        if (angle < -PI)
            angle += TwoPI;
        if (angle > PI)
            angle -= TwoPI;
        return angle;
    }

    /// <summary>
    /// Compute the torque (angular velocity vector) needed to rotate from one quaternion to another.
    /// </summary>
    public static Vector3 ComputeTorque(this Quaternion from, Quaternion to, float dt)
    {
        Quaternion diff = Quaternion.Normalize(Quaternion.Inverse(from) * to);

        if (diff.W < 0)
            diff = -diff;

        float angle = 2f * MathF.Acos(Math.Clamp(MathF.Abs(diff.W), 0f, 1f));
        float s = MathF.Sqrt(1f - diff.W * diff.W);

        if (s < Epsilon || dt < Epsilon)
            return Vector3.Zero;

        Vector3 axis = new Vector3(diff.X, diff.Y, diff.Z) / s;
        return axis * (angle / dt);
    }

    /// <summary>
    /// Blend two rotations with a weight. The result is normalized.
    /// </summary>
    public static Quaternion Blend(this Quaternion a, Quaternion b, float weightB)
    {
        if (Quaternion.Dot(a, b) < 0)
            b = -b;

        float weightA = 1f - weightB;
        Quaternion result = new(
            weightA * a.X + weightB * b.X,
            weightA * a.Y + weightB * b.Y,
            weightA * a.Z + weightB * b.Z,
            weightA * a.W + weightB * b.W);

        return Quaternion.Normalize(result);
    }

    /// <summary>
    /// Compute the great circle distance between two quaternion orientations.
    /// </summary>
    public static float GreatCircleDistance(this Quaternion a, Quaternion b) =>
        MathF.Acos(Math.Clamp(MathF.Abs(Quaternion.Dot(Quaternion.Normalize(a), Quaternion.Normalize(b))), 0f, 1f));

    /// <summary>
    /// Create a rotation that rotates around a specific point (pivot) in 3D space.
    /// </summary>
    public static Matrix4x4 RotateAroundPoint(Quaternion rotation, Vector3 pivot, Vector3 point)
    {
        Matrix4x4 toOrigin = Matrix4x4.CreateTranslation(-pivot);
        Matrix4x4 rotMatrix = Matrix4x4.CreateFromQuaternion(rotation);
        Matrix4x4 fromOrigin = Matrix4x4.CreateTranslation(pivot);

        return toOrigin * rotMatrix * fromOrigin;
    }

    #endregion

    #region Batch Quaternion Operations

    /// <summary>
    /// Batch normalize an array of quaternions.
    /// </summary>
    public static void BatchNormalize(Span<Quaternion> quaternions)
    {
        for (int i = 0; i < quaternions.Length; i++)
            quaternions[i] = Quaternion.Normalize(quaternions[i]);
    }

    /// <summary>
    /// Batch SLERP between paired arrays of quaternions with a scalar t.
    /// </summary>
    public static void BatchSlerp(ReadOnlySpan<Quaternion> a, ReadOnlySpan<Quaternion> b, float t,
        Span<Quaternion> result)
    {
        int count = Math.Min(a.Length, Math.Min(b.Length, result.Length));

        for (int i = 0; i < count; i++)
            result[i] = SlerpShortest(a[i], b[i], t);
    }

    /// <summary>
    /// Batch NLERP between paired arrays of quaternions.
    /// </summary>
    public static void BatchNlerp(ReadOnlySpan<Quaternion> a, ReadOnlySpan<Quaternion> b, float t,
        Span<Quaternion> result)
    {
        int count = Math.Min(a.Length, Math.Min(b.Length, result.Length));

        for (int i = 0; i < count; i++)
            result[i] = NlerpShortest(a[i], b[i], t);
    }

    /// <summary>
    /// Batch multiply quaternions: result[i] = a[i] * b[i].
    /// </summary>
    public static void BatchMultiply(ReadOnlySpan<Quaternion> a, ReadOnlySpan<Quaternion> b,
        Span<Quaternion> result)
    {
        int count = Math.Min(a.Length, Math.Min(b.Length, result.Length));

        for (int i = 0; i < count; i++)
            result[i] = Quaternion.Normalize(a[i] * b[i]);
    }

    /// <summary>
    /// Batch inverse: result[i] = Inverse(quaternions[i]).
    /// </summary>
    public static void BatchInverse(ReadOnlySpan<Quaternion> quaternions, Span<Quaternion> result)
    {
        int count = Math.Min(quaternions.Length, result.Length);

        for (int i = 0; i < count; i++)
            result[i] = Quaternion.Inverse(quaternions[i]);
    }

    /// <summary>
    /// Batch compute angular distances between paired quaternions.
    /// </summary>
    public static void BatchAngularDistance(ReadOnlySpan<Quaternion> a, ReadOnlySpan<Quaternion> b,
        Span<float> result)
    {
        int count = Math.Min(a.Length, Math.Min(b.Length, result.Length));

        for (int i = 0; i < count; i++)
            result[i] = AngularDistance(a[i], b[i]);
    }

    /// <summary>
    /// Batch convert quaternions to rotation matrices.
    /// </summary>
    public static void BatchToMatrix(ReadOnlySpan<Quaternion> quaternions, Span<Matrix4x4> matrices)
    {
        int count = Math.Min(quaternions.Length, matrices.Length);

        for (int i = 0; i < count; i++)
            matrices[i] = Matrix4x4.CreateFromQuaternion(Quaternion.Normalize(quaternions[i]));
    }

    /// <summary>
    /// Batch convert quaternions to Euler angles.
    /// </summary>
    public static void BatchToEuler(ReadOnlySpan<Quaternion> quaternions, Span<Vector3> eulerAngles)
    {
        int count = Math.Min(quaternions.Length, eulerAngles.Length);

        for (int i = 0; i < count; i++)
            eulerAngles[i] = ToEulerAngles(quaternions[i]);
    }

    /// <summary>
    /// Batch extract axis-angle from quaternions.
    /// </summary>
    public static void BatchToAxisAngle(ReadOnlySpan<Quaternion> quaternions, Span<Vector3> axes,
        Span<float> angles)
    {
        int count = quaternions.Length;
        Debug.Assert(axes.Length >= count && angles.Length >= count);

        for (int i = 0; i < count; i++)
            quaternions[i].ToAxisAngle(out axes[i], out angles[i]);
    }

    /// <summary>
    /// Batch apply shortest-path fix to a sequence of quaternions.
    /// </summary>
    public static void BatchShortestPath(Span<Quaternion> quaternions)
    {
        for (int i = 1; i < quaternions.Length; i++)
        {
            if (Quaternion.Dot(quaternions[i - 1], quaternions[i]) < 0)
                quaternions[i] = -quaternions[i];
        }
    }

    /// <summary>
    /// Batch blend with a single reference quaternion.
    /// </summary>
    public static void BatchBlendWithReference(ReadOnlySpan<Quaternion> quaternions, Quaternion reference,
        float weight, Span<Quaternion> result)
    {
        int count = Math.Min(quaternions.Length, result.Length);

        for (int i = 0; i < count; i++)
            result[i] = Blend(quaternions[i], reference, weight);
    }

    /// <summary>
    /// Compute the mean rotation of a set of quaternions using the quaternion logarithm method.
    /// </summary>
    public static Quaternion MeanRotation(ReadOnlySpan<Quaternion> quaternions)
    {
        Debug.Assert(quaternions.Length > 0);

        if (quaternions.Length == 1)
            return quaternions[0];

        Quaternion sum = Quaternion.Zero;
        for (int i = 0; i < quaternions.Length; i++)
        {
            Quaternion q = quaternions[i];
            if (Quaternion.Dot(q, quaternions[0]) < 0)
                q = -q;
            sum += q;
        }

        return Quaternion.Normalize(sum);
    }

    /// <summary>
    /// Compute a smooth rotation curve through keyframes.
    /// </summary>
    public static Quaternion SmoothRotationCurve(ReadOnlySpan<Quaternion> keyframes, ReadOnlySpan<float> keyTimes,
        float time)
    {
        Debug.Assert(keyframes.Length >= 2);
        Debug.Assert(keyframes.Length == keyTimes.Length);

        if (time <= keyTimes[0])
            return keyframes[0];
        if (time >= keyTimes[^1])
            return keyframes[^1];

        int segment = 0;
        for (int i = 0; i < keyTimes.Length - 1; i++)
        {
            if (time >= keyTimes[i] && time <= keyTimes[i + 1])
            {
                segment = i;
                break;
            }
        }

        float segmentDuration = keyTimes[segment + 1] - keyTimes[segment];
        float t = segmentDuration > Epsilon
            ? (time - keyTimes[segment]) / segmentDuration
            : 0f;

        return SlerpShortest(keyframes[segment], keyframes[segment + 1], t);
    }

    #endregion
}
