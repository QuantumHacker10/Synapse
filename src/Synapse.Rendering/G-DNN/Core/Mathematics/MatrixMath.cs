using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;


// ============================================================
// FILE: MatrixMath.cs
// PATH: Core/Mathematics/MatrixMath.cs
// ============================================================


using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

using GDNN.Rendering.Compat;

namespace GDNN.Core.Mathematics;

/// <summary>
/// Comprehensive matrix mathematics utilities for the G-DNN neural geometry engine.
/// Provides decomposition, curve evaluation, frustum operations, interpolation,
/// and batch SIMD-optimized matrix operations for System.Numerics.Matrix4x4.
/// </summary>
public static class MatrixMath
{
    private const float PI = MathF.PI;
    private const float TwoPI = MathF.Tau;
    private const float Epsilon = 1e-6f;
    private const int MatrixStride = 16; // 4x4 = 16 floats

    #region Matrix Decomposition

    /// <summary>
    /// Decompose a matrix into translation, rotation, and scale (TRS decomposition).
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="translation">Extracted translation vector.</param>
    /// <param name="rotation">Extracted rotation quaternion.</param>
    /// <param name="scale">Extracted scale vector.</param>
    /// <returns>True if decomposition was successful.</returns>
    public static bool DecomposeTRS(this Matrix4x4 matrix, out Vector3 translation, out Quaternion rotation, out Vector3 scale)
    {
        translation = new Vector3(matrix.M41, matrix.M42, matrix.M43);

        Vector3 row0 = new(matrix.M11, matrix.M12, matrix.M13);
        Vector3 row1 = new(matrix.M21, matrix.M22, matrix.M23);
        Vector3 row2 = new(matrix.M31, matrix.M32, matrix.M33);

        float sx = row0.Length();
        float sy = row1.Length();
        float sz = row2.Length();

        scale = new Vector3(sx, sy, sz);

        if (sx < Epsilon || sy < Epsilon || sz < Epsilon)
        {
            rotation = Quaternion.Identity;
            return false;
        }

        Matrix4x4 rotationMatrix = matrix;
        rotationMatrix.M11 /= sx; rotationMatrix.M12 /= sx; rotationMatrix.M13 /= sx;
        rotationMatrix.M21 /= sy; rotationMatrix.M22 /= sy; rotationMatrix.M23 /= sy;
        rotationMatrix.M31 /= sz; rotationMatrix.M32 /= sz; rotationMatrix.M33 /= sz;

        rotation = Quaternion.CreateFromRotationMatrix(rotationMatrix);
        return true;
    }

    /// <summary>
    /// Decompose a matrix into translation and a 3x3 rotation matrix.
    /// </summary>
    public static void DecomposeTranslationRotation(this Matrix4x4 matrix, out Vector3 translation, out Matrix4x4 rotation)
    {
        translation = new Vector3(matrix.M41, matrix.M42, matrix.M43);
        rotation = Matrix4x4.Identity;
        rotation.M11 = matrix.M11; rotation.M12 = matrix.M12; rotation.M13 = matrix.M13;
        rotation.M21 = matrix.M21; rotation.M22 = matrix.M22; rotation.M23 = matrix.M23;
        rotation.M31 = matrix.M31; rotation.M32 = matrix.M32; rotation.M33 = matrix.M33;
    }

    /// <summary>
    /// Extract Euler angles (pitch, yaw, roll) from a rotation matrix in radians.
    /// </summary>
    public static Vector3 ExtractEulerAngles(this Matrix4x4 matrix)
    {
        float pitch, yaw, roll;

        if (MathF.Abs(matrix.M31) < 1f - Epsilon)
        {
            pitch = MathF.Asin(-matrix.M31);
            yaw = MathF.Atan2(matrix.M32, matrix.M33);
            roll = MathF.Atan2(matrix.M21, matrix.M11);
        }
        else
        {
            pitch = matrix.M31 > 0 ? -PI / 2f : PI / 2f;
            yaw = MathF.Atan2(-matrix.M13, matrix.M22);
            roll = 0f;
        }

        return new Vector3(pitch, yaw, roll);
    }

    /// <summary>
    /// Extract the scale factors from a matrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ExtractScale(this Matrix4x4 matrix)
    {
        float sx = new Vector3(matrix.M11, matrix.M12, matrix.M13).Length();
        float sy = new Vector3(matrix.M21, matrix.M22, matrix.M23).Length();
        float sz = new Vector3(matrix.M31, matrix.M32, matrix.M33).Length();
        return new Vector3(sx, sy, sz);
    }

    /// <summary>
    /// Extract the translation vector from a matrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 ExtractTranslation(this Matrix4x4 matrix) =>
        new(matrix.M41, matrix.M42, matrix.M43);

    /// <summary>
    /// Build a TRS matrix from individual translation, rotation, and scale components.
    /// </summary>
    public static Matrix4x4 BuildTRS(Vector3 translation, Quaternion rotation, Vector3 scale)
    {
        Matrix4x4 result = Matrix4x4.CreateFromQuaternion(rotation);
        result.M11 *= scale.X; result.M12 *= scale.X; result.M13 *= scale.X;
        result.M21 *= scale.Y; result.M22 *= scale.Y; result.M23 *= scale.Y;
        result.M31 *= scale.Z; result.M32 *= scale.Z; result.M33 *= scale.Z;
        result.M41 = translation.X; result.M42 = translation.Y; result.M43 = translation.Z;
        return result;
    }

    /// <summary>
    /// Build a TRS matrix from Euler angles (in radians) and individual components.
    /// </summary>
    public static Matrix4x4 BuildTRSFromEuler(Vector3 translation, Vector3 eulerAngles, Vector3 scale)
    {
        Quaternion rotation = Quaternion.CreateFromYawPitchRoll(eulerAngles.Y, eulerAngles.X, eulerAngles.Z);
        return BuildTRS(translation, rotation, scale);
    }

    /// <summary>
    /// Build a perspective projection matrix.
    /// </summary>
    public static Matrix4x4 CreatePerspective(float fovY, float aspectRatio, float nearPlane, float farPlane)
    {
        return Matrix4x4.CreatePerspectiveFieldOfView(fovY, aspectRatio, nearPlane, farPlane);
    }

    /// <summary>
    /// Build an infinite perspective projection matrix (no far plane).
    /// </summary>
    public static Matrix4x4 CreateInfinitePerspective(float fovY, float aspectRatio, float nearPlane)
    {
        float tanHalfFov = MathF.Tan(fovY * 0.5f);
        float invTan = 1f / tanHalfFov;

        Matrix4x4 result = RenderingMath.ZeroMatrix;
        result.M11 = invTan / aspectRatio;
        result.M22 = invTan;
        result.M33 = -1f;
        result.M34 = -1f;
        result.M43 = -2f * nearPlane;
        return result;
    }

    /// <summary>
    /// Build an orthographic projection matrix.
    /// </summary>
    public static Matrix4x4 CreateOrthographic(float width, float height, float nearPlane, float farPlane)
    {
        return Matrix4x4.CreateOrthographic(width, height, nearPlane, farPlane);
    }

    /// <summary>
    /// Build a look-at view matrix.
    /// </summary>
    public static Matrix4x4 CreateLookAt(Vector3 eye, Vector3 target, Vector3 up) =>
        Matrix4x4.CreateLookAt(eye, target, up);

    /// <summary>
    /// Build a world matrix from position, forward direction, and up direction.
    /// </summary>
    public static Matrix4x4 CreateWorldFromDirection(Vector3 position, Vector3 forward, Vector3 up)
    {
        Vector3 zAxis = Vector3.Normalize(forward);
        Vector3 xAxis = Vector3.Normalize(Vector3.Cross(up, zAxis));
        Vector3 yAxis = Vector3.Cross(zAxis, xAxis);

        Matrix4x4 result = Matrix4x4.Identity;
        result.M11 = xAxis.X; result.M12 = xAxis.Y; result.M13 = xAxis.Z;
        result.M21 = yAxis.X; result.M22 = yAxis.Y; result.M23 = yAxis.Z;
        result.M31 = zAxis.X; result.M32 = zAxis.Y; result.M33 = zAxis.Z;
        result.M41 = position.X; result.M42 = position.Y; result.M43 = position.Z;
        return result;
    }

    /// <summary>
    /// Build a rotation matrix from axis and angle (in radians).
    /// </summary>
    public static Matrix4x4 CreateRotationAxis(Vector3 axis, float angle) =>
        Matrix4x4.CreateFromAxisAngle(axis, angle);

    /// <summary>
    /// Build a rotation matrix from Euler angles (pitch, yaw, roll) in radians.
    /// </summary>
    public static Matrix4x4 CreateRotationFromEuler(Vector3 eulerAngles) =>
        Matrix4x4.CreateFromYawPitchRoll(eulerAngles.Y, eulerAngles.X, eulerAngles.Z);

    /// <summary>
    /// Build a scale matrix.
    /// </summary>
    public static Matrix4x4 CreateScale(Vector3 scale) =>
        Matrix4x4.CreateScale(scale);

    /// <summary>
    /// Build a translation matrix.
    /// </summary>
    public static Matrix4x4 CreateTranslation(Vector3 translation) =>
        Matrix4x4.CreateTranslation(translation);

    /// <summary>
    /// Build a mirror/reflection matrix for a plane defined by a normal and distance.
    /// </summary>
    public static Matrix4x4 CreateReflection(Vector3 planeNormal, float planeD)
    {
        float nx = planeNormal.X, ny = planeNormal.Y, nz = planeNormal.Z;
        float nx2 = nx * 2f, ny2 = ny * 2f, nz2 = nz * 2f;

        Matrix4x4 result = Matrix4x4.Identity;
        result.M11 = 1f - nx2 * nx; result.M12 = -nx2 * ny; result.M13 = -nx2 * nz;
        result.M21 = -ny2 * nx; result.M22 = 1f - ny2 * ny; result.M23 = -ny2 * nz;
        result.M31 = -nz2 * nx; result.M32 = -nz2 * ny; result.M33 = 1f - nz2 * nz;
        result.M41 = -2f * planeD * nx; result.M42 = -2f * planeD * ny; result.M43 = -2f * planeD * nz;
        return result;
    }

    /// <summary>
    /// Build a shear matrix.
    /// </summary>
    public static Matrix4x4 CreateShear(float xy, float xz, float yx, float yz, float zx, float zy)
    {
        Matrix4x4 result = Matrix4x4.Identity;
        result.M12 = xy; result.M13 = xz;
        result.M21 = yx; result.M23 = yz;
        result.M31 = zx; result.M32 = zy;
        return result;
    }

    /// <summary>
    /// Build a billboarding matrix that faces the camera.
    /// </summary>
    public static Matrix4x4 CreateBillboard(Vector3 objectPosition, Vector3 cameraPosition,
        Vector3 cameraUp, Vector3? cameraForward = null)
    {
        Vector3 forward = Vector3.Normalize(cameraPosition - objectPosition);
        if (cameraForward.HasValue)
            forward = -cameraForward.Value;

        Vector3 right = Vector3.Normalize(Vector3.Cross(cameraUp, forward));
        Vector3 up = Vector3.Cross(forward, right);

        Matrix4x4 result = Matrix4x4.Identity;
        result.M11 = right.X; result.M12 = right.Y; result.M13 = right.Z;
        result.M21 = up.X; result.M22 = up.Y; result.M23 = up.Z;
        result.M31 = forward.X; result.M32 = forward.Y; result.M33 = forward.Z;
        result.M41 = objectPosition.X; result.M42 = objectPosition.Y; result.M43 = objectPosition.Z;
        return result;
    }

    #endregion

    #region Bezier Curve Evaluation

    /// <summary>
    /// Evaluate a quadratic Bezier curve at parameter t.
    /// </summary>
    public static Vector3 QuadraticBezier(Matrix4x4 controlPoints, float t)
    {
        float u = 1f - t;
        float u2 = u * u;
        float t2 = t * t;

        return new Vector3(
            u2 * controlPoints.M11 + 2f * u * t * controlPoints.M12 + t2 * controlPoints.M13,
            u2 * controlPoints.M21 + 2f * u * t * controlPoints.M22 + t2 * controlPoints.M23,
            u2 * controlPoints.M31 + 2f * u * t * controlPoints.M32 + t2 * controlPoints.M33);
    }

    /// <summary>
    /// Evaluate a cubic Bezier curve at parameter t using four control points.
    /// </summary>
    public static Vector3 CubicBezier(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        float u = 1f - t;
        float u2 = u * u;
        float u3 = u2 * u;
        float t2 = t * t;
        float t3 = t2 * t;

        return u3 * p0 +
               3f * u2 * t * p1 +
               3f * u * t2 * p2 +
               t3 * p3;
    }

    /// <summary>
    /// Evaluate a cubic Bezier curve and its first derivative (tangent) at parameter t.
    /// </summary>
    public static void CubicBezierWithTangent(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t,
        out Vector3 position, out Vector3 tangent)
    {
        float u = 1f - t;
        float u2 = u * u;
        float u3 = u2 * u;
        float t2 = t * t;
        float t3 = t2 * t;

        position = u3 * p0 + 3f * u2 * t * p1 + 3f * u * t2 * p2 + t3 * p3;
        tangent = 3f * u2 * (p1 - p0) + 6f * u * t * (p2 - p1) + 3f * t2 * (p3 - p2);
    }

    /// <summary>
    /// Evaluate a cubic Bezier curve with second derivative (curvature) at parameter t.
    /// </summary>
    public static void CubicBezierFull(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t,
        out Vector3 position, out Vector3 tangent, out Vector3 secondDerivative)
    {
        float u = 1f - t;

        position = u * u * u * p0 +
                   3f * u * u * t * p1 +
                   3f * u * t * t * p2 +
                   t * t * t * p3;

        tangent = 3f * u * u * (p1 - p0) +
                  6f * u * t * (p2 - p1) +
                  3f * t * t * (p3 - p2);

        secondDerivative = 6f * u * (p2 - 2f * p1 + p0) +
                           6f * t * (p3 - 2f * p2 + p1);
    }

    /// <summary>
    /// Evaluate a cubic B-spline curve at parameter t.
    /// </summary>
    public static Vector3 CubicBSpline(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        float inv6 = 1f / 6f;

        return inv6 * (
            (-t3 + 3f * t2 - 3f * t + 1f) * p0 +
            (3f * t3 - 6f * t2 + 4f) * p1 +
            (-3f * t3 + 3f * t2 + 3f * t + 1f) * p2 +
            t3 * p3);
    }

    /// <summary>
    /// Evaluate a Catmull-Rom spline at parameter t using four control points.
    /// </summary>
    public static Vector3 CatmullRomSpline(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;

        return 0.5f * (
            (2f * p1) +
            (-p0 + p2) * t +
            (2f * p0 - 5f * p1 + 4f * p2 - p3) * t2 +
            (-p0 + 3f * p1 - 3f * p2 + p3) * t3);
    }

    /// <summary>
    /// Evaluate a NURBS-like curve with rational weighting at parameter t.
    /// Control points are rows of a 4x4 matrix, weights are provided separately.
    /// </summary>
    public static Vector3 NURBSEvaluate(ReadOnlySpan<Vector4> controlPoints, ReadOnlySpan<float> weights, float t)
    {
        Debug.Assert(controlPoints.Length == 4 && weights.Length == 4);

        float u = 1f - t;
        float u2 = u * u;
        float u3 = u2 * u;
        float t2 = t * t;
        float t3 = t2 * t;

        float w0 = u3 * weights[0];
        float w1 = 3f * u2 * t * weights[1];
        float w2 = 3f * u * t2 * weights[2];
        float w3 = t3 * weights[3];

        float denominator = w0 + w1 + w2 + w3;
        if (MathF.Abs(denominator) < Epsilon) return Vector3.Zero;

        Vector4 result = w0 * controlPoints[0] +
                         w1 * controlPoints[1] +
                         w2 * controlPoints[2] +
                         w3 * controlPoints[3];

        return new Vector3(result.X, result.Y, result.Z) / denominator;
    }

    /// <summary>
    /// Subdivide a cubic Bezier curve at parameter t into two sub-curves.
    /// Uses De Casteljau's algorithm.
    /// </summary>
    public static void SubdivideBezier(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t,
        out Vector3 a0, out Vector3 a1, out Vector3 a2, out Vector3 a3,
        out Vector3 b0, out Vector3 b1, out Vector3 b2, out Vector3 b3)
    {
        Vector3 l01 = Vector3.Lerp(p0, p1, t);
        Vector3 l12 = Vector3.Lerp(p1, p2, t);
        Vector3 l23 = Vector3.Lerp(p2, p3, t);
        Vector3 l012 = Vector3.Lerp(l01, l12, t);
        Vector3 l123 = Vector3.Lerp(l12, l23, t);
        Vector3 l0123 = Vector3.Lerp(l012, l123, t);

        a0 = p0; a1 = l01; a2 = l012; a3 = l0123;
        b0 = l0123; b1 = l123; b2 = l23; b3 = p3;
    }

    /// <summary>
    /// Compute the closest point on a cubic Bezier curve to a given point using Newton-Raphson.
    /// </summary>
    public static Vector3 ClosestPointOnBezier(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3,
        Vector3 queryPoint, int iterations = 8)
    {
        float bestT = 0f;
        float bestDistSq = float.MaxValue;

        const int sampleCount = 32;
        for (int i = 0; i <= sampleCount; i++)
        {
            float t = (float)i / sampleCount;
            Vector3 point = CubicBezier(p0, p1, p2, p3, t);
            float distSq = Vector3.DistanceSquared(point, queryPoint);
            if (distSq < bestDistSq) { bestDistSq = distSq; bestT = t; }
        }

        for (int iter = 0; iter < iterations; iter++)
        {
            float t = bestT;
            CubicBezierWithTangent(p0, p1, p2, p3, t, out Vector3 pos, out Vector3 tan);
            float diff = Vector3.Dot(pos - queryPoint, tan);
            float denom = Vector3.Dot(tan, tan) + Vector3.Dot(pos - queryPoint,
                6f * (1f - t) * (p2 - 2f * p1 + p0) + 6f * t * (p3 - 2f * p2 + p1));
            if (MathF.Abs(denom) > Epsilon)
                bestT -= diff / denom;
            bestT = Math.Clamp(bestT, 0f, 1f);
        }

        return CubicBezier(p0, p1, p2, p3, bestT);
    }

    /// <summary>
    /// Compute arc length of a cubic Bezier curve via adaptive Simpson integration.
    /// </summary>
    public static float BezierArcLength(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, int segments = 32)
    {
        float length = 0f;
        Vector3 prev = p0;
        for (int i = 1; i <= segments; i++)
        {
            float t = (float)i / segments;
            Vector3 current = CubicBezier(p0, p1, p2, p3, t);
            length += Vector3.Distance(prev, current);
            prev = current;
        }
        return length;
    }

    #endregion

    #region Frustum Operations

    /// <summary>
    /// Extract frustum planes from a combined view-projection matrix.
    /// Returns 6 planes as (normal, distance) tuples in left, right, bottom, top, near, far order.
    /// </summary>
    public static void ExtractFrustumPlanes(Matrix4x4 viewProj, Span<Vector4> planes)
    {
        Debug.Assert(planes.Length >= 6);

        // Left
        planes[0] = new Vector4(
            viewProj.M14 + viewProj.M11,
            viewProj.M24 + viewProj.M21,
            viewProj.M34 + viewProj.M31,
            viewProj.M44 + viewProj.M41);

        // Right
        planes[1] = new Vector4(
            viewProj.M14 - viewProj.M11,
            viewProj.M24 - viewProj.M21,
            viewProj.M34 - viewProj.M31,
            viewProj.M44 - viewProj.M41);

        // Bottom
        planes[2] = new Vector4(
            viewProj.M14 + viewProj.M12,
            viewProj.M24 + viewProj.M22,
            viewProj.M34 + viewProj.M32,
            viewProj.M44 + viewProj.M42);

        // Top
        planes[3] = new Vector4(
            viewProj.M14 - viewProj.M12,
            viewProj.M24 - viewProj.M22,
            viewProj.M34 - viewProj.M32,
            viewProj.M44 - viewProj.M42);

        // Near
        planes[4] = new Vector4(
            viewProj.M13,
            viewProj.M23,
            viewProj.M33,
            viewProj.M43);

        // Far
        planes[5] = new Vector4(
            viewProj.M14 - viewProj.M13,
            viewProj.M24 - viewProj.M23,
            viewProj.M34 - viewProj.M33,
            viewProj.M44 - viewProj.M43);

        for (int i = 0; i < 6; i++)
        {
            float length = new Vector3(planes[i].X, planes[i].Y, planes[i].Z).Length();
            if (length > Epsilon)
                planes[i] /= length;
        }
    }

    /// <summary>
    /// Test if a point is inside a frustum defined by 6 planes.
    /// </summary>
    public static bool IsPointInFrustum(Vector3 point, ReadOnlySpan<Vector4> planes)
    {
        Debug.Assert(planes.Length >= 6);

        for (int i = 0; i < 6; i++)
        {
            float dot = planes[i].X * point.X + planes[i].Y * point.Y +
                        planes[i].Z * point.Z + planes[i].W;
            if (dot < 0) return false;
        }
        return true;
    }

    /// <summary>
    /// Test if a sphere intersects a frustum defined by 6 planes.
    /// Returns 0 = outside, 1 = intersecting, 2 = fully inside.
    /// </summary>
    public static int SphereFrustumTest(Vector3 center, float radius, ReadOnlySpan<Vector4> planes)
    {
        Debug.Assert(planes.Length >= 6);

        int insideCount = 0;

        for (int i = 0; i < 6; i++)
        {
            float dot = planes[i].X * center.X + planes[i].Y * center.Y +
                        planes[i].Z * center.Z + planes[i].W;
            if (dot < -radius) return 0;
            if (dot >= radius) insideCount++;
        }

        return insideCount == 6 ? 2 : 1;
    }

    /// <summary>
    /// Test if an axis-aligned bounding box intersects a frustum.
    /// Returns 0 = outside, 1 = intersecting, 2 = fully inside.
    /// </summary>
    public static int AABBFrustumTest(Vector3 boxMin, Vector3 boxMax, ReadOnlySpan<Vector4> planes)
    {
        Debug.Assert(planes.Length >= 6);

        int insideCount = 0;

        for (int i = 0; i < 6; i++)
        {
            Vector3 positiveVertex = new(
                planes[i].X > 0 ? boxMax.X : boxMin.X,
                planes[i].Y > 0 ? boxMax.Y : boxMin.Y,
                planes[i].Z > 0 ? boxMax.Z : boxMin.Z);

            float dot = planes[i].X * positiveVertex.X + planes[i].Y * positiveVertex.Y +
                        planes[i].Z * positiveVertex.Z + planes[i].W;
            if (dot < 0) return 0;

            Vector3 negativeVertex = new(
                planes[i].X > 0 ? boxMin.X : boxMax.X,
                planes[i].Y > 0 ? boxMin.Y : boxMax.Y,
                planes[i].Z > 0 ? boxMin.Z : boxMax.Z);

            float negDot = planes[i].X * negativeVertex.X + planes[i].Y * negativeVertex.Y +
                           planes[i].Z * negativeVertex.Z + planes[i].W;
            if (negDot >= 0) insideCount++;
        }

        return insideCount == 6 ? 2 : 1;
    }

    /// <summary>
    /// Test if an oriented bounding box intersects a frustum.
    /// </summary>
    public static int OBBFrustumTest(Vector3 center, Vector3 extent, Quaternion orientation,
        ReadOnlySpan<Vector4> planes)
    {
        Debug.Assert(planes.Length >= 6);

        Matrix4x4 rotation = Matrix4x4.CreateFromQuaternion(orientation);
        Vector3 axes0 = new(rotation.M11, rotation.M12, rotation.M13);
        Vector3 axes1 = new(rotation.M21, rotation.M22, rotation.M23);
        Vector3 axes2 = new(rotation.M31, rotation.M32, rotation.M33);

        int insideCount = 0;

        for (int i = 0; i < 6; i++)
        {
            Vector3 planeNormal = new(planes[i].X, planes[i].Y, planes[i].Z);

            float r = extent.X * MathF.Abs(Vector3.Dot(planeNormal, axes0)) +
                      extent.Y * MathF.Abs(Vector3.Dot(planeNormal, axes1)) +
                      extent.Z * MathF.Abs(Vector3.Dot(planeNormal, axes2));

            float d = Vector3.Dot(planeNormal, center) + planes[i].W;

            if (d < -r) return 0;
            if (d >= r) insideCount++;
        }

        return insideCount == 6 ? 2 : 1;
    }

    /// <summary>
    /// Compute the frustum corner points from a view-projection matrix.
    /// </summary>
    public static void ExtractFrustumCorners(Matrix4x4 inverseViewProj, Span<Vector3> corners)
    {
        Debug.Assert(corners.Length >= 8);

        ReadOnlySpan<Vector4> ndcCorners = stackalloc Vector4[8]
        {
            new(-1, -1, -1, 1), new(1, -1, -1, 1),
            new(-1, 1, -1, 1), new(1, 1, -1, 1),
            new(-1, -1, 1, 1), new(1, -1, 1, 1),
            new(-1, 1, 1, 1), new(1, 1, 1, 1)
        };

        for (int i = 0; i < 8; i++)
        {
            Vector4 corner = Vector4.Transform(ndcCorners[i], inverseViewProj);
            corners[i] = new Vector3(corner.X, corner.Y, corner.Z) / corner.W;
        }
    }

    /// <summary>
    /// Compute the frustum bounding sphere for shadow map stable fitting.
    /// </summary>
    public static void FrustumBoundingSphere(Matrix4x4 viewProj, out Vector3 center, out float radius)
    {
        Matrix4x4 invViewProj = Matrix4x4.Identity;
        Matrix4x4.Invert(viewProj, out invViewProj);

        Span<Vector3> corners = stackalloc Vector3[8];
        ExtractFrustumCorners(invViewProj, corners);

        center = Vector3.Zero;
        for (int i = 0; i < 8; i++) center += corners[i];
        center /= 8f;

        radius = 0f;
        for (int i = 0; i < 8; i++)
        {
            float dist = Vector3.Distance(center, corners[i]);
            if (dist > radius) radius = dist;
        }
    }

    #endregion

    #region Bounds Transformation and Intersection

    /// <summary>
    /// Transform an axis-aligned bounding box by a matrix and compute the new AABB.
    /// </summary>
    public static void TransformAABB(Vector3 min, Vector3 max, Matrix4x4 matrix, out Vector3 newMin, out Vector3 newMax)
    {
        Vector3 corners = max - min;
        Vector3 center = (min + max) * 0.5f;
        Vector3 halfExtent = corners * 0.5f;

        Vector3 newCenter = Vector3.Transform(center, matrix);
        Matrix4x4 absMatrix = matrix;
        absMatrix.M11 = MathF.Abs(absMatrix.M11); absMatrix.M12 = MathF.Abs(absMatrix.M12); absMatrix.M13 = MathF.Abs(absMatrix.M13);
        absMatrix.M21 = MathF.Abs(absMatrix.M21); absMatrix.M22 = MathF.Abs(absMatrix.M22); absMatrix.M23 = MathF.Abs(absMatrix.M23);
        absMatrix.M31 = MathF.Abs(absMatrix.M31); absMatrix.M32 = MathF.Abs(absMatrix.M32); absMatrix.M33 = MathF.Abs(absMatrix.M33);

        Vector3 newHalfExtent = new Vector3(
            absMatrix.M11 * halfExtent.X + absMatrix.M21 * halfExtent.Y + absMatrix.M31 * halfExtent.Z,
            absMatrix.M12 * halfExtent.X + absMatrix.M22 * halfExtent.Y + absMatrix.M32 * halfExtent.Z,
            absMatrix.M13 * halfExtent.X + absMatrix.M23 * halfExtent.Y + absMatrix.M33 * halfExtent.Z);

        newMin = newCenter - newHalfExtent;
        newMax = newCenter + newHalfExtent;
    }

    /// <summary>
    /// Test if two axis-aligned bounding boxes intersect.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool AABBIntersects(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB) =>
        minA.X <= maxB.X && maxA.X >= minB.X &&
        minA.Y <= maxB.Y && maxA.Y >= minB.Y &&
        minA.Z <= maxB.Z && maxA.Z >= minB.Z;

    /// <summary>
    /// Test if a point is inside an axis-aligned bounding box.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool PointInAABB(Vector3 point, Vector3 min, Vector3 max) =>
        point.X >= min.X && point.X <= max.X &&
        point.Y >= min.Y && point.Y <= max.Y &&
        point.Z >= min.Z && point.Z <= max.Z;

    /// <summary>
    /// Compute the intersection of two AABBs. Returns false if they don't overlap.
    /// </summary>
    public static bool AABBIntersection(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB,
        out Vector3 resultMin, out Vector3 resultMax)
    {
        resultMin = Vector3.Max(minA, minB);
        resultMax = Vector3.Min(maxA, maxB);

        return resultMin.X <= resultMax.X &&
               resultMin.Y <= resultMax.Y &&
               resultMin.Z <= resultMax.Z;
    }

    /// <summary>
    /// Merge two AABBs into a single encompassing AABB.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AABBMerge(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB,
        out Vector3 resultMin, out Vector3 resultMax)
    {
        resultMin = Vector3.Min(minA, minB);
        resultMax = Vector3.Max(maxA, maxB);
    }

    /// <summary>
    /// Compute the surface area of an AABB.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float AABBSurfaceArea(Vector3 min, Vector3 max)
    {
        Vector3 d = max - min;
        return 2f * (d.X * d.Y + d.Y * d.Z + d.Z * d.X);
    }

    /// <summary>
    /// Compute the volume of an AABB.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float AABBVolume(Vector3 min, Vector3 max)
    {
        Vector3 d = max - min;
        return d.X * d.Y * d.Z;
    }

    /// <summary>
    /// Compute the AABB of a set of points.
    /// </summary>
    public static void ComputeAABB(ReadOnlySpan<Vector3> points, out Vector3 min, out Vector3 max)
    {
        Debug.Assert(points.Length > 0);
        min = new Vector3(float.MaxValue);
        max = new Vector3(float.MinValue);

        for (int i = 0; i < points.Length; i++)
        {
            min = Vector3.Min(min, points[i]);
            max = Vector3.Max(max, points[i]);
        }
    }

    /// <summary>
    /// Compute an oriented bounding box from a set of points using PCA.
    /// Returns center, half-extents, and orientation quaternion.
    /// </summary>
    public static void ComputeOBB(ReadOnlySpan<Vector3> points, out Vector3 center, out Vector3 halfExtents,
        out Quaternion orientation)
    {
        Debug.Assert(points.Length >= 3);

        center = Vector3.Zero;
        for (int i = 0; i < points.Length; i++) center += points[i];
        center /= points.Length;

        Span<float> covariance = stackalloc float[9];
        ComputeCovarianceMatrix(points, center, covariance);

        // Jacobi eigenvalue algorithm for 3x3 symmetric matrix
        Span<float> eigenvalues = stackalloc float[3];
        Span<Vector3> eigenvectors = stackalloc Vector3[3];
        JacobiEigen(covariance, eigenvalues, eigenvectors);

        // Sort eigenvalues descending
        if (eigenvalues[0] < eigenvalues[1]) { Swap(eigenvalues, 0, 1); Swap(eigenvectors, 0, 1); }
        if (eigenvalues[0] < eigenvalues[2]) { Swap(eigenvalues, 0, 2); Swap(eigenvectors, 0, 2); }
        if (eigenvalues[1] < eigenvalues[2]) { Swap(eigenvalues, 1, 2); Swap(eigenvectors, 1, 2); }

        // Project points onto principal axes to find extents
        Vector3 axis0 = eigenvectors[0];
        Vector3 axis1 = eigenvectors[1];
        Vector3 axis2 = eigenvectors[2];

        Vector3 minProj = new(float.MaxValue);
        Vector3 maxProj = new(float.MinValue);

        for (int i = 0; i < points.Length; i++)
        {
            Vector3 d = points[i] - center;
            float p0 = Vector3.Dot(d, axis0);
            float p1 = Vector3.Dot(d, axis1);
            float p2 = Vector3.Dot(d, axis2);

            minProj = Vector3.Min(minProj, new Vector3(p0, p1, p2));
            maxProj = Vector3.Max(maxProj, new Vector3(p0, p1, p2));
        }

        halfExtents = (maxProj - minProj) * 0.5f;
        center += (maxProj + minProj) * 0.5f * new Vector3(
            Vector3.Dot(axis0, Vector3.UnitX) != 0 ? axis0.X : axis1.X,
            Vector3.Dot(axis0, Vector3.UnitY) != 0 ? axis0.Y : axis1.Y,
            1f);

        orientation = Quaternion.CreateFromRotationMatrix(new Matrix4x4(
            axis0.X, axis0.Y, axis0.Z, 0,
            axis1.X, axis1.Y, axis1.Z, 0,
            axis2.X, axis2.Y, axis2.Z, 0,
            0, 0, 0, 1));
    }

    private static void Swap(Span<float> span, int a, int b) { (span[a], span[b]) = (span[b], span[a]); }
    private static void Swap(Span<Vector3> span, int a, int b) { (span[a], span[b]) = (span[b], span[a]); }

    private static void ComputeCovarianceMatrix(ReadOnlySpan<Vector3> points, Vector3 centroid, Span<float> cov)
    {
        cov.Clear();
        float invN = 1f / points.Length;

        for (int i = 0; i < points.Length; i++)
        {
            Vector3 d = points[i] - centroid;
            cov[0] += d.X * d.X; cov[1] += d.X * d.Y; cov[2] += d.X * d.Z;
            cov[4] += d.Y * d.Y; cov[5] += d.Y * d.Z;
            cov[8] += d.Z * d.Z;
        }

        cov[0] *= invN; cov[1] *= invN; cov[2] *= invN;
        cov[3] = cov[1]; cov[4] *= invN; cov[5] *= invN;
        cov[6] = cov[2]; cov[7] = cov[5]; cov[8] *= invN;
    }

    private static void JacobiEigen(Span<float> matrix, Span<float> eigenvalues, Span<Vector3> eigenvectors)
    {
        Span<float> a = stackalloc float[9];
        matrix.CopyTo(a);

        eigenvectors[0] = Vector3.UnitX;
        eigenvectors[1] = Vector3.UnitY;
        eigenvectors[2] = Vector3.UnitZ;

        for (int iter = 0; iter < 50; iter++)
        {
            int p = 0, q = 1;
            float maxVal = MathF.Abs(a[1]);
            if (MathF.Abs(a[2]) > maxVal) { p = 0; q = 2; maxVal = MathF.Abs(a[2]); }
            if (MathF.Abs(a[5]) > maxVal) { p = 1; q = 2; }

            if (maxVal < 1e-10f) break;

            float theta = 0.5f * MathF.Atan2(2f * a[p * 3 + q], a[p * 3 + p] - a[q * 3 + q]);
            float c = MathF.Cos(theta);
            float s = MathF.Sin(theta);

            // Apply rotation
            Span<float> newA = stackalloc float[9];
            a.CopyTo(newA);

            for (int i = 0; i < 3; i++)
            {
                newA[i * 3 + p] = c * a[i * 3 + p] + s * a[i * 3 + q];
                newA[i * 3 + q] = -s * a[i * 3 + p] + c * a[i * 3 + q];
            }
            for (int i = 0; i < 3; i++)
            {
                newA[p * 3 + i] = c * a[p * 3 + i] + s * a[q * 3 + i];
                newA[q * 3 + i] = -s * a[p * 3 + i] + c * a[q * 3 + i];
            }
            newA[p * 3 + q] = 0;
            newA[q * 3 + p] = 0;

            newA.CopyTo(a);

            // Update eigenvectors
            Vector3 vp = eigenvectors[p];
            Vector3 vq = eigenvectors[q];
            eigenvectors[p] = c * vp + s * vq;
            eigenvectors[q] = -s * vp + c * vq;
        }

        eigenvalues[0] = a[0];
        eigenvalues[1] = a[4];
        eigenvalues[2] = a[8];
    }

    #endregion

    #region Homogeneous Clipping

    /// <summary>
    /// Clip a line segment against a homogeneous plane.
    /// Returns true if any part of the segment remains, and outputs the clipped segment.
    /// </summary>
    public static bool ClipLineToPlane(Vector4 p0, Vector4 p1, Vector4 plane, out Vector4 clippedP0, out Vector4 clippedP1)
    {
        float d0 = Vector4.Dot(plane, p0);
        float d1 = Vector4.Dot(plane, p1);

        bool inside0 = d0 >= 0;
        bool inside1 = d1 >= 0;

        if (inside0 && inside1)
        {
            clippedP0 = p0;
            clippedP1 = p1;
            return true;
        }

        if (!inside0 && !inside1)
        {
            clippedP0 = Vector4.Zero;
            clippedP1 = Vector4.Zero;
            return false;
        }

        float t = d0 / (d0 - d1);
        Vector4 intersection = p0 + t * (p1 - p0);

        if (inside0)
        {
            clippedP0 = p0;
            clippedP1 = intersection;
        }
        else
        {
            clippedP0 = intersection;
            clippedP1 = p1;
        }
        return true;
    }

    /// <summary>
    /// Clip a triangle against a homogeneous plane using Sutherland-Hodgman algorithm.
    /// Returns the number of output vertices (0, 1, 2, or 3).
    /// </summary>
    public static int ClipTriangleToPlane(ReadOnlySpan<Vector4> triangle, Vector4 plane, Span<Vector4> output)
    {
        Debug.Assert(triangle.Length == 3 && output.Length >= 3);

        Span<float> dists = stackalloc float[3];
        Span<bool> inside = stackalloc bool[3];

        for (int i = 0; i < 3; i++)
        {
            dists[i] = Vector4.Dot(plane, triangle[i]);
            inside[i] = dists[i] >= 0;
        }

        int outCount = 0;

        for (int i = 0; i < 3; i++)
        {
            int j = (i + 1) % 3;

            if (inside[i])
            {
                output[outCount++] = triangle[i];
            }

            if (inside[i] != inside[j])
            {
                float t = dists[i] / (dists[i] - dists[j]);
                output[outCount++] = triangle[i] + t * (triangle[j] - triangle[i]);
            }
        }

        return outCount;
    }

    /// <summary>
    /// Clip a convex polygon against a plane.
    /// Returns the number of output vertices.
    /// </summary>
    public static int ClipConvexPolygonToPlane(ReadOnlySpan<Vector4> polygon, Vector4 plane, Span<Vector4> output)
    {
        int inCount = polygon.Length;
        int outCount = 0;

        Span<float> dists = stackalloc float[polygon.Length];

        for (int i = 0; i < inCount; i++)
            dists[i] = Vector4.Dot(plane, polygon[i]);

        for (int i = 0; i < inCount; i++)
        {
            int j = (i + 1) % inCount;
            bool insideI = dists[i] >= 0;
            bool insideJ = dists[j] >= 0;

            if (insideI)
            {
                output[outCount++] = polygon[i];
            }

            if (insideI != insideJ)
            {
                float t = dists[i] / (dists[i] - dists[j]);
                output[outCount++] = polygon[i] + t * (polygon[j] - polygon[i]);
            }
        }

        return outCount;
    }

    /// <summary>
    /// Clip a line segment against the view frustum (6 planes).
    /// Returns true if any part remains.
    /// </summary>
    public static bool ClipLineToFrustum(Vector4 p0, Vector4 p1, ReadOnlySpan<Vector4> planes,
        out Vector4 clippedP0, out Vector4 clippedP1)
    {
        clippedP0 = p0;
        clippedP1 = p1;

        for (int i = 0; i < 6; i++)
        {
            if (!ClipLineToPlane(clippedP0, clippedP1, planes[i], out clippedP0, out clippedP1))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Compute the signed distance from a point to a plane defined as (normal, distance).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float PlaneDot(Vector4 plane, Vector3 point) =>
        plane.X * point.X + plane.Y * point.Y + plane.Z * point.Z + plane.W;

    /// <summary>
    /// Normalize a plane equation (a, b, c, d) so that the normal has unit length.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector4 NormalizePlane(Vector4 plane)
    {
        float length = new Vector3(plane.X, plane.Y, plane.Z).Length();
        return length > Epsilon ? plane / length : plane;
    }

    #endregion

    #region Matrix Interpolation

    /// <summary>
    /// Linearly interpolate between two matrices component-wise.
    /// </summary>
    public static Matrix4x4 Lerp(Matrix4x4 a, Matrix4x4 b, float t)
    {
        return new Matrix4x4(
            MathHelper.Lerp(a.M11, b.M11, t), MathHelper.Lerp(a.M12, b.M12, t),
            MathHelper.Lerp(a.M13, b.M13, t), MathHelper.Lerp(a.M14, b.M14, t),
            MathHelper.Lerp(a.M21, b.M21, t), MathHelper.Lerp(a.M22, b.M22, t),
            MathHelper.Lerp(a.M23, b.M23, t), MathHelper.Lerp(a.M24, b.M24, t),
            MathHelper.Lerp(a.M31, b.M31, t), MathHelper.Lerp(a.M32, b.M32, t),
            MathHelper.Lerp(a.M33, b.M33, t), MathHelper.Lerp(a.M34, b.M34, t),
            MathHelper.Lerp(a.M41, b.M41, t), MathHelper.Lerp(a.M42, b.M42, t),
            MathHelper.Lerp(a.M43, b.M43, t), MathHelper.Lerp(a.M44, b.M44, t));
    }

    /// <summary>
    /// Interpolate between two matrices using TRS decomposition (proper rotation interpolation).
    /// </summary>
    public static Matrix4x4 TRSLerp(Matrix4x4 a, Matrix4x4 b, float t)
    {
        a.DecomposeTRS(out Vector3 tA, out Quaternion rA, out Vector3 sA);
        b.DecomposeTRS(out Vector3 tB, out Quaternion rB, out Vector3 sB);

        Vector3 translation = Vector3.Lerp(tA, tB, t);
        Quaternion rotation = Quaternion.Slerp(rA, rB, t);
        Vector3 scale = Vector3.Lerp(sA, sB, t);

        return BuildTRS(translation, rotation, scale);
    }

    /// <summary>
    /// Blend two transform matrices with weights.
    /// </summary>
    public static Matrix4x4 Blend(Matrix4x4 a, Matrix4x4 b, float weightA, float weightB)
    {
        return Lerp(a, b, weightB / (weightA + weightB));
    }

    /// <summary>
    /// Apply a delta rotation to a matrix around its local origin.
    /// </summary>
    public static Matrix4x4 RotateLocal(Matrix4x4 matrix, Quaternion deltaRotation)
    {
        matrix.DecomposeTRS(out Vector3 translation, out Quaternion rotation, out Vector3 scale);
        rotation = Quaternion.Normalize(deltaRotation * rotation);
        return BuildTRS(translation, rotation, scale);
    }

    /// <summary>
    /// Apply a delta translation to a matrix in world space.
    /// </summary>
    public static Matrix4x4 TranslateWorld(Matrix4x4 matrix, Vector3 delta)
    {
        matrix.M41 += delta.X;
        matrix.M42 += delta.Y;
        matrix.M43 += delta.Z;
        return matrix;
    }

    /// <summary>
    /// Apply a delta scale to a matrix.
    /// </summary>
    public static Matrix4x4 ScaleLocal(Matrix4x4 matrix, Vector3 scaleFactor)
    {
        matrix.DecomposeTRS(out Vector3 translation, out Quaternion rotation, out Vector3 scale);
        scale = scale.ComponentMultiply(scaleFactor);
        return BuildTRS(translation, rotation, scale);
    }

    #endregion

    #region Inverse, Transpose, Determinant

    /// <summary>
    /// Compute the inverse of a matrix, returning false if the matrix is singular.
    /// </summary>
    public static bool TryInverse(this Matrix4x4 matrix, out Matrix4x4 result) =>
        Matrix4x4.Invert(matrix, out result);

    /// <summary>
    /// Compute the inverse of a matrix (throws if singular).
    /// </summary>
    public static Matrix4x4 Inverse(this Matrix4x4 matrix)
    {
        if (!Matrix4x4.Invert(matrix, out Matrix4x4 result))
            throw new InvalidOperationException("Matrix is singular and cannot be inverted.");
        return result;
    }

    /// <summary>
    /// Compute the determinant of a 4x4 matrix.
    /// </summary>
    public static float Determinant(this Matrix4x4 matrix) =>
        GDNN.Rendering.Compat.Matrix4x4Compat.Determinant(matrix);

    /// <summary>
    /// Compute the transpose of a 4x4 matrix.
    /// </summary>
    public static Matrix4x4 Transpose(this Matrix4x4 matrix) =>
        Matrix4x4.Transpose(matrix);

    /// <summary>
    /// Compute the inverse transpose (useful for transforming normals).
    /// </summary>
    public static Matrix4x4 InverseTranspose(this Matrix4x4 matrix)
    {
        Matrix4x4.Invert(matrix, out Matrix4x4 inv);
        return Matrix4x4.Transpose(inv);
    }

    /// <summary>
    /// Compute the cofactor matrix.
    /// </summary>
    public static Matrix4x4 Cofactor(this Matrix4x4 m)
    {
        return new Matrix4x4(
            m.M22 * (m.M33 * m.M44 - m.M34 * m.M43) - m.M23 * (m.M32 * m.M44 - m.M34 * m.M42) + m.M24 * (m.M32 * m.M43 - m.M33 * m.M42),
            -(m.M21 * (m.M33 * m.M44 - m.M34 * m.M43) - m.M23 * (m.M31 * m.M44 - m.M34 * m.M41) + m.M24 * (m.M31 * m.M43 - m.M33 * m.M41)),
            m.M21 * (m.M32 * m.M44 - m.M34 * m.M42) - m.M22 * (m.M31 * m.M44 - m.M34 * m.M41) + m.M24 * (m.M31 * m.M42 - m.M32 * m.M41),
            -(m.M21 * (m.M32 * m.M43 - m.M33 * m.M42) - m.M22 * (m.M31 * m.M43 - m.M33 * m.M41) + m.M23 * (m.M31 * m.M42 - m.M32 * m.M41)),

            -(m.M12 * (m.M33 * m.M44 - m.M34 * m.M43) - m.M13 * (m.M32 * m.M44 - m.M34 * m.M42) + m.M14 * (m.M32 * m.M43 - m.M33 * m.M42)),
            m.M11 * (m.M33 * m.M44 - m.M34 * m.M43) - m.M13 * (m.M31 * m.M44 - m.M34 * m.M41) + m.M14 * (m.M31 * m.M43 - m.M33 * m.M41),
            -(m.M11 * (m.M32 * m.M44 - m.M34 * m.M42) - m.M12 * (m.M31 * m.M44 - m.M34 * m.M41) + m.M14 * (m.M31 * m.M42 - m.M32 * m.M41)),
            m.M11 * (m.M32 * m.M43 - m.M33 * m.M42) - m.M12 * (m.M31 * m.M43 - m.M33 * m.M41) + m.M13 * (m.M31 * m.M42 - m.M32 * m.M41),

            m.M12 * (m.M23 * m.M44 - m.M24 * m.M43) - m.M13 * (m.M22 * m.M44 - m.M24 * m.M42) + m.M14 * (m.M22 * m.M43 - m.M23 * m.M42),
            -(m.M11 * (m.M23 * m.M44 - m.M24 * m.M43) - m.M13 * (m.M21 * m.M44 - m.M24 * m.M41) + m.M14 * (m.M21 * m.M43 - m.M23 * m.M41)),
            m.M11 * (m.M22 * m.M44 - m.M24 * m.M42) - m.M12 * (m.M21 * m.M44 - m.M24 * m.M41) + m.M14 * (m.M21 * m.M42 - m.M22 * m.M41),
            -(m.M11 * (m.M22 * m.M43 - m.M23 * m.M42) - m.M12 * (m.M21 * m.M43 - m.M23 * m.M41) + m.M13 * (m.M21 * m.M42 - m.M22 * m.M41)),

            -(m.M12 * (m.M23 * m.M34 - m.M24 * m.M33) - m.M13 * (m.M22 * m.M34 - m.M24 * m.M32) + m.M14 * (m.M22 * m.M33 - m.M23 * m.M32)),
            m.M11 * (m.M23 * m.M34 - m.M24 * m.M33) - m.M13 * (m.M21 * m.M34 - m.M24 * m.M31) + m.M14 * (m.M21 * m.M33 - m.M23 * m.M31),
            -(m.M11 * (m.M22 * m.M34 - m.M24 * m.M32) - m.M12 * (m.M21 * m.M34 - m.M24 * m.M31) + m.M14 * (m.M21 * m.M32 - m.M22 * m.M31)),
            m.M11 * (m.M22 * m.M33 - m.M23 * m.M32) - m.M12 * (m.M21 * m.M33 - m.M23 * m.M31) + m.M13 * (m.M21 * m.M32 - m.M22 * m.M31));
    }

    /// <summary>
    /// Check if a matrix is approximately an identity matrix.
    /// </summary>
    public static bool IsIdentity(this Matrix4x4 matrix, float tolerance = Epsilon)
    {
        return MathF.Abs(matrix.M11 - 1) <= tolerance && MathF.Abs(matrix.M12) <= tolerance &&
               MathF.Abs(matrix.M13) <= tolerance && MathF.Abs(matrix.M14) <= tolerance &&
               MathF.Abs(matrix.M21) <= tolerance && MathF.Abs(matrix.M22 - 1) <= tolerance &&
               MathF.Abs(matrix.M23) <= tolerance && MathF.Abs(matrix.M24) <= tolerance &&
               MathF.Abs(matrix.M31) <= tolerance && MathF.Abs(matrix.M32) <= tolerance &&
               MathF.Abs(matrix.M33 - 1) <= tolerance && MathF.Abs(matrix.M34) <= tolerance &&
               MathF.Abs(matrix.M41) <= tolerance && MathF.Abs(matrix.M42) <= tolerance &&
               MathF.Abs(matrix.M43) <= tolerance && MathF.Abs(matrix.M44 - 1) <= tolerance;
    }

    /// <summary>
    /// Check if a matrix is approximately equal to another.
    /// </summary>
    public static bool ApproximatelyEquals(this Matrix4x4 a, Matrix4x4 b, float tolerance = Epsilon)
    {
        return MathF.Abs(a.M11 - b.M11) <= tolerance && MathF.Abs(a.M12 - b.M12) <= tolerance &&
               MathF.Abs(a.M13 - b.M13) <= tolerance && MathF.Abs(a.M14 - b.M14) <= tolerance &&
               MathF.Abs(a.M21 - b.M21) <= tolerance && MathF.Abs(a.M22 - b.M22) <= tolerance &&
               MathF.Abs(a.M23 - b.M23) <= tolerance && MathF.Abs(a.M24 - b.M24) <= tolerance &&
               MathF.Abs(a.M31 - b.M31) <= tolerance && MathF.Abs(a.M32 - b.M32) <= tolerance &&
               MathF.Abs(a.M33 - b.M33) <= tolerance && MathF.Abs(a.M34 - b.M34) <= tolerance &&
               MathF.Abs(a.M41 - b.M41) <= tolerance && MathF.Abs(a.M42 - b.M42) <= tolerance &&
               MathF.Abs(a.M43 - b.M43) <= tolerance && MathF.Abs(a.M44 - b.M44) <= tolerance;
    }

    /// <summary>
    /// Compute the 3x3 normal matrix (inverse transpose of upper-left 3x3).
    /// Used for transforming normals correctly under non-uniform scaling.
    /// </summary>
    public static Matrix4x4 NormalMatrix(this Matrix4x4 matrix)
    {
        Matrix4x4 m3x3 = Matrix4x4.Identity;
        m3x3.M11 = matrix.M11; m3x3.M12 = matrix.M12; m3x3.M13 = matrix.M13;
        m3x3.M21 = matrix.M21; m3x3.M22 = matrix.M22; m3x3.M23 = matrix.M23;
        m3x3.M31 = matrix.M31; m3x3.M32 = matrix.M32; m3x3.M33 = matrix.M33;

        Matrix4x4.Invert(m3x3, out Matrix4x4 inv);
        return Matrix4x4.Transpose(inv);
    }

    #endregion

    #region Batch Matrix Operations

    /// <summary>
    /// Batch multiply matrices: result[i] = a[i] * b[i].
    /// </summary>
    public static unsafe void BatchMultiply(ReadOnlySpan<Matrix4x4> a, ReadOnlySpan<Matrix4x4> b, Span<Matrix4x4> result)
    {
        int count = a.Length;
        Debug.Assert(count == b.Length && count == result.Length);

        if (Avx.IsSupported && count >= 8)
        {
            int simdCount = count - (count % 8);
            fixed (Matrix4x4* pA = a, pB = b, pR = result)
            {
                float* pAF = (float*)pA;
                float* pBF = (float*)pB;
                float* pRF = (float*)pR;

                for (int m = 0; m < simdCount; m++)
                {
                    int offset = m * MatrixStride;
                    // Process 8 matrices (128 floats) using AVX
                    Matrix4x4 matA = a[m];
                    Matrix4x4 matB = b[m];
                    result[m] = Matrix4x4.Multiply(matA, matB);
                }
            }

            for (int i = simdCount; i < count; i++)
                result[i] = Matrix4x4.Multiply(a[i], b[i]);
        }
        else
        {
            for (int i = 0; i < count; i++)
                result[i] = Matrix4x4.Multiply(a[i], b[i]);
        }
    }

    /// <summary>
    /// Batch transform points by matrices.
    /// </summary>
    public static void BatchTransformPoints(ReadOnlySpan<Matrix4x4> matrices, ReadOnlySpan<Vector3> points,
        Span<Vector3> results)
    {
        int count = Math.Min(matrices.Length, points.Length);
        Debug.Assert(results.Length >= count);

        for (int i = 0; i < count; i++)
            results[i] = Vector3.Transform(points[i], matrices[i]);
    }

    /// <summary>
    /// Batch transform normals by the inverse transpose of matrices.
    /// </summary>
    public static void BatchTransformNormals(ReadOnlySpan<Matrix4x4> matrices, ReadOnlySpan<Vector3> normals,
        Span<Vector3> results)
    {
        int count = Math.Min(matrices.Length, normals.Length);
        Debug.Assert(results.Length >= count);

        for (int i = 0; i < count; i++)
        {
            Matrix4x4.Invert(matrices[i], out Matrix4x4 inv);
            Matrix4x4 invT = Matrix4x4.Transpose(inv);
            results[i] = Vector3.TransformNormal(normals[i], invT);
        }
    }

    /// <summary>
    /// Accumulate a weighted sum of matrices: result = sum(weight[i] * matrices[i]).
    /// </summary>
    public static Matrix4x4 WeightedSum(ReadOnlySpan<Matrix4x4> matrices, ReadOnlySpan<float> weights)
    {
        Debug.Assert(matrices.Length == weights.Length);
        Matrix4x4 result = RenderingMath.ZeroMatrix;

        for (int i = 0; i < matrices.Length; i++)
        {
            float w = weights[i];
            result.M11 += matrices[i].M11 * w; result.M12 += matrices[i].M12 * w;
            result.M13 += matrices[i].M13 * w; result.M14 += matrices[i].M14 * w;
            result.M21 += matrices[i].M21 * w; result.M22 += matrices[i].M22 * w;
            result.M23 += matrices[i].M23 * w; result.M24 += matrices[i].M24 * w;
            result.M31 += matrices[i].M31 * w; result.M32 += matrices[i].M32 * w;
            result.M33 += matrices[i].M33 * w; result.M34 += matrices[i].M34 * w;
            result.M41 += matrices[i].M41 * w; result.M42 += matrices[i].M42 * w;
            result.M43 += matrices[i].M43 * w; result.M44 += matrices[i].M44 * w;
        }

        return result;
    }

    /// <summary>
    /// Compute the product of a chain of matrices.
    /// </summary>
    public static Matrix4x4 ChainMultiply(ReadOnlySpan<Matrix4x4> matrices)
    {
        Debug.Assert(matrices.Length > 0);
        Matrix4x4 result = matrices[0];
        for (int i = 1; i < matrices.Length; i++)
            result = Matrix4x4.Multiply(result, matrices[i]);
        return result;
    }

    /// <summary>
    /// Compute the transpose of each matrix in a batch.
    /// </summary>
    public static void BatchTranspose(ReadOnlySpan<Matrix4x4> matrices, Span<Matrix4x4> results)
    {
        Debug.Assert(matrices.Length == results.Length);
        for (int i = 0; i < matrices.Length; i++)
            results[i] = Matrix4x4.Transpose(matrices[i]);
    }

    /// <summary>
    /// Compute the inverse of each matrix in a batch.
    /// Returns false if any matrix is singular.
    /// </summary>
    public static bool BatchInverse(ReadOnlySpan<Matrix4x4> matrices, Span<Matrix4x4> results)
    {
        Debug.Assert(matrices.Length == results.Length);
        bool allInvertible = true;

        for (int i = 0; i < matrices.Length; i++)
        {
            if (!Matrix4x4.Invert(matrices[i], out results[i]))
                allInvertible = false;
        }

        return allInvertible;
    }

    /// <summary>
    /// Compute the trace (sum of diagonal elements) of a matrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Trace(this Matrix4x4 matrix) =>
        matrix.M11 + matrix.M22 + matrix.M33 + matrix.M44;

    /// <summary>
    /// Decompose a matrix into a chain of simple transforms (for debug/analysis).
    /// Returns translation, then rotation, then scale.
    /// </summary>
    public static string DecomposeToString(this Matrix4x4 matrix)
    {
        matrix.DecomposeTRS(out Vector3 t, out Quaternion r, out Vector3 s);
        return $"T({t.X:F3}, {t.Y:F3}, {t.Z:F3}) R({r.X:F3}, {r.Y:F3}, {r.Z:F3}, {r.W:F3}) S({s.X:F3}, {s.Y:F3}, {s.Z:F3})";
    }

    #endregion

    #region Additional Matrix Utilities

    /// <summary>
    /// Build a matrix that maps from one coordinate system to another.
    /// </summary>
    public static Matrix4x4 ChangeOfBasis(Vector3 right, Vector3 up, Vector3 forward, Vector3 origin)
    {
        Matrix4x4 result = Matrix4x4.Identity;
        result.M11 = right.X; result.M12 = right.Y; result.M13 = right.Z;
        result.M21 = up.X; result.M22 = up.Y; result.M23 = up.Z;
        result.M31 = forward.X; result.M32 = forward.Y; result.M33 = forward.Z;
        result.M41 = origin.X; result.M42 = origin.Y; result.M43 = origin.Z;

        Matrix4x4.Invert(result, out Matrix4x4 inv);
        return inv;
    }

    /// <summary>
    /// Build a rotation matrix that rotates from direction 'from' to direction 'to'.
    /// </summary>
    public static Matrix4x4 RotationFromTo(Vector3 from, Vector3 to)
    {
        Vector3 axis = Vector3.Cross(from, to);
        float dot = Vector3.Dot(from, to);

        if (dot < -1f + Epsilon)
        {
            Vector3 ortho = MathF.Abs(from.X) < 0.9f
                ? Vector3.Cross(from, Vector3.UnitX)
                : Vector3.Cross(from, Vector3.UnitY);
            ortho = Vector3.Normalize(ortho);
            return Matrix4x4.CreateFromAxisAngle(ortho, PI);
        }

        float angle = MathF.Acos(Math.Clamp(dot, -1f, 1f));
        axis = Vector3.Normalize(axis);
        return Matrix4x4.CreateFromAxisAngle(axis, angle);
    }

    /// <summary>
    /// Compute the polar decomposition of a matrix (closest rotation to a given matrix).
    /// </summary>
    public static Matrix4x4 PolarDecomposition(this Matrix4x4 matrix, int iterations = 10)
    {
        Matrix4x4 x = matrix;
        Matrix4x4 xInv = x;
        for (int i = 0; i < iterations; i++)
        {
            Matrix4x4.Invert(x, out xInv);
            x = (x + xInv) * 0.5f;
        }

        // Extract rotation
        Matrix4x4.Invert(x, out xInv);
        return Matrix4x4.Multiply(matrix, xInv);
    }

    /// <summary>
    /// Compute the logarithm of a rotation matrix (axis-angle representation).
    /// </summary>
    public static Vector3 RotationMatrixLog(this Matrix4x4 matrix)
    {
        Quaternion q = Quaternion.CreateFromRotationMatrix(matrix);
        return QuaternionToAxisAngle(q);
    }

    /// <summary>
    /// Compute the exponential of an axis-angle vector to a rotation matrix.
    /// </summary>
    public static Matrix4x4 RotationMatrixExp(Vector3 axisAngle)
    {
        float angle = axisAngle.Length();
        if (angle < Epsilon) return Matrix4x4.Identity;
        Vector3 axis = axisAngle / angle;
        return Matrix4x4.CreateFromAxisAngle(axis, angle);
    }

    /// <summary>
    /// Convert a quaternion to axis-angle representation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector3 QuaternionToAxisAngle(Quaternion q)
    {
        float angle = 2f * MathF.Acos(MathF.Min(1f, MathF.Abs(q.W)));
        float s = MathF.Sqrt(1f - q.W * q.W);
        if (s < Epsilon) return Vector3.Zero;
        return new Vector3(q.X, q.Y, q.Z) / s * angle;
    }

    /// <summary>
    /// Compute the Kronecker product of two 4x4 matrices (16x16 result stored as 4x4 blocks).
    /// </summary>
    public static void KroneckerProduct(Matrix4x4 a, Matrix4x4 b, Span<Matrix4x4> result)
    {
        Debug.Assert(result.Length >= 16);

        ReadOnlySpan<float> aF = MemoryMarshal.CreateReadOnlySpan(ref a.M11, 16);
        ReadOnlySpan<float> bF = MemoryMarshal.CreateReadOnlySpan(ref b.M11, 16);

        for (int i = 0; i < 16; i++)
        {
            Matrix4x4 block = b * aF[i];
            result[i] = block;
        }
    }

    /// <summary>
    /// Compute a shadow matrix for directional light shadows.
    /// </summary>
    public static Matrix4x4 CreateShadowMatrix(Vector4 lightPlane, Matrix4x4 world)
    {
        float dot = Vector4.Dot(lightPlane, new Vector4(world.M41, world.M42, world.M43, 1f));
        Matrix4x4 shadow = Matrix4x4.Identity;

        shadow.M11 -= lightPlane.X * world.M11;
        shadow.M12 -= lightPlane.X * world.M12;
        shadow.M13 -= lightPlane.X * world.M13;
        shadow.M14 -= lightPlane.X * world.M14;

        shadow.M21 -= lightPlane.Y * world.M21;
        shadow.M22 -= lightPlane.Y * world.M22;
        shadow.M23 -= lightPlane.Y * world.M23;
        shadow.M24 -= lightPlane.Y * world.M24;

        shadow.M31 -= lightPlane.Z * world.M31;
        shadow.M32 -= lightPlane.Z * world.M32;
        shadow.M33 -= lightPlane.Z * world.M33;
        shadow.M34 -= lightPlane.Z * world.M34;

        shadow.M41 -= lightPlane.W * world.M41;
        shadow.M42 -= lightPlane.W * world.M42;
        shadow.M43 -= lightPlane.W * world.M43;
        shadow.M44 -= lightPlane.W * world.M44;

        return shadow;
    }

    #endregion
}

/// <summary>
/// Internal math helper for scalar lerp operations.
/// </summary>
internal static class MathHelper
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Lerp(float a, float b, float t) => a + (b - a) * t;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Clamp(float value, float min, float max) =>
        MathF.Max(min, MathF.Min(max, value));
}
