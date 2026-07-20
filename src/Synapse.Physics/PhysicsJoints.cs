// ============================================================================
// Synapse Omnia — Physics/PhysicsJoints.cs
// Advanced bilateral joints solved with the same PGS loop as contacts.
// Hinge · Ball-socket · Slider · Fixed · Distance
// ============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;

namespace Synapse.Physics;

/// <summary>Industrial joint kinds for articulated Synapse assemblies.</summary>
public enum JointType : byte
{
    /// <summary>Revolute about a shared axis (doors, wheels hubs).</summary>
    Hinge = 0,
    /// <summary>Spherical — 3 translational constraints.</summary>
    BallSocket = 1,
    /// <summary>Prismatic along an axis with optional limits.</summary>
    Slider = 2,
    /// <summary>Weld — lock relative pose.</summary>
    Fixed = 3,
    /// <summary>Keep two anchors at a rest length (springy rope / shock).</summary>
    Distance = 4
}

/// <summary>Bilateral constraint between two rigid bodies (or body ↔ world).</summary>
public sealed class PhysicsJoint
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public string Name { get; set; } = "Joint";
    public JointType Type { get; set; } = JointType.BallSocket;
    public Guid BodyA { get; set; }
    /// <summary>Guid.Empty means anchored to the world (static).</summary>
    public Guid BodyB { get; set; }

    /// <summary>Anchor in local space of body A.</summary>
    public Vector3 LocalAnchorA { get; set; }
    /// <summary>Anchor in local space of body B (or world if BodyB empty).</summary>
    public Vector3 LocalAnchorB { get; set; }

    /// <summary>Hinge/slider axis in local space of A.</summary>
    public Vector3 LocalAxisA { get; set; } = Vector3.UnitY;
    /// <summary>Matching axis in local space of B.</summary>
    public Vector3 LocalAxisB { get; set; } = Vector3.UnitY;

    public float MinLimit { get; set; } = float.NegativeInfinity;
    public float MaxLimit { get; set; } = float.PositiveInfinity;
    public float RestLength { get; set; } = 1f;
    public float Stiffness { get; set; } = 1f;
    public float Damping { get; set; } = 0.1f;
    /// <summary>
    /// Soft-constraint compliance (CFM). 0 = hard bilateral; larger values yield springier joints.
    /// Units approximate m/N scaled for the velocity-level PGS denominator.
    /// </summary>
    public float Compliance { get; set; }
    public bool CollideConnected { get; set; }

    // Accumulated impulses for warm-starting.
    public Vector3 AccumulatedImpulse;
    public float AccumulatedMotor;
}

/// <summary>Joint solver mixed into <see cref="RigidBodyWorld"/> PGS iterations.</summary>
public static class JointSolver
{
    public static void SolveVelocity(
        IReadOnlyList<PhysicsJoint> joints,
        Func<Guid, RigidBody?> getBody,
        int iterations,
        float dt)
    {
        if (joints.Count == 0 || dt <= 0f) return;
        float invDt = 1f / dt;

        for (int iter = 0; iter < iterations; iter++)
        {
            for (int ji = 0; ji < joints.Count; ji++)
            {
                var j = joints[ji];
                var a = getBody(j.BodyA);
                if (a == null) continue;
                var b = j.BodyB == Guid.Empty ? null : getBody(j.BodyB);

                Vector3 worldAnchorA = a.Position + Vector3.Transform(j.LocalAnchorA, a.Orientation);
                Vector3 worldAnchorB = b != null
                    ? b.Position + Vector3.Transform(j.LocalAnchorB, b.Orientation)
                    : j.LocalAnchorB;

                switch (j.Type)
                {
                    case JointType.Distance:
                        SolveDistance(j, a, b, worldAnchorA, worldAnchorB, invDt);
                        break;
                    case JointType.BallSocket:
                    case JointType.Fixed:
                        SolvePointToPoint(j, a, b, worldAnchorA, worldAnchorB, invDt, lockOrientation: j.Type == JointType.Fixed);
                        break;
                    case JointType.Hinge:
                        SolveHinge(j, a, b, worldAnchorA, worldAnchorB, invDt);
                        break;
                    case JointType.Slider:
                        SolveSlider(j, a, b, worldAnchorA, worldAnchorB, invDt);
                        break;
                }
            }
        }
    }

    public static void SolvePosition(
        IReadOnlyList<PhysicsJoint> joints,
        Func<Guid, RigidBody?> getBody,
        int iterations,
        float baumgarte)
    {
        for (int iter = 0; iter < iterations; iter++)
        {
            for (int ji = 0; ji < joints.Count; ji++)
            {
                var j = joints[ji];
                var a = getBody(j.BodyA);
                if (a == null) continue;
                var b = j.BodyB == Guid.Empty ? null : getBody(j.BodyB);

                Vector3 wa = a.Position + Vector3.Transform(j.LocalAnchorA, a.Orientation);
                Vector3 wb = b != null
                    ? b.Position + Vector3.Transform(j.LocalAnchorB, b.Orientation)
                    : j.LocalAnchorB;

                if (j.Type == JointType.Distance)
                {
                    Vector3 d = wb - wa;
                    float len = d.Length();
                    if (len < 1e-6f) continue;
                    float err = len - j.RestLength;
                    Vector3 n = d / len;
                    float soft = 1f / (1f + MathF.Max(0f, j.Compliance) * 80f);
                    ApplyPositionalCorrection(a, b, wa, wb, n * (err * baumgarte * soft));
                }
                else
                {
                    Vector3 err = wb - wa;
                    float soft = 1f / (1f + MathF.Max(0f, j.Compliance) * 80f);
                    ApplyPositionalCorrection(a, b, wa, wb, err * (baumgarte * soft));
                }
            }
        }
    }

    private static void SolveDistance(
        PhysicsJoint j, RigidBody a, RigidBody? b,
        Vector3 wa, Vector3 wb, float invDt)
    {
        Vector3 d = wb - wa;
        float len = d.Length();
        if (len < 1e-6f) return;
        Vector3 n = d / len;
        float C = len - j.RestLength;

        Vector3 ra = wa - a.Position;
        Vector3 rb = b != null ? wb - b.Position : Vector3.Zero;
        Vector3 va = a.LinearVelocity + Vector3.Cross(a.AngularVelocity, ra);
        Vector3 vb = b != null ? b.LinearVelocity + Vector3.Cross(b.AngularVelocity, rb) : Vector3.Zero;
        float vn = Vector3.Dot(vb - va, n);

        float kn = SoftDenominator(EffectiveMass(a, b, ra, rb, n), j.Compliance, invDt);
        if (kn < 1e-8f) return;

        float compliance = MathF.Max(0f, j.Compliance);
        float bias;
        if (compliance < 1e-8f)
        {
            // Hard bilateral spring (legacy industrial defaults).
            bias = j.Stiffness * C * invDt + j.Damping * vn;
        }
        else
        {
            // Soft spring: ERP scaled by compliance; CFM already in the denominator.
            float erp = Math.Clamp(0.2f * MathF.Max(0.05f, j.Stiffness), 0.02f, 0.9f);
            erp /= 1f + compliance * 12f;
            bias = erp * C * invDt + j.Damping * vn;
        }

        float lambda = -(vn + bias) / kn;
        Vector3 impulse = n * lambda;
        ApplyImpulse(a, -impulse, ra);
        if (b != null) ApplyImpulse(b, impulse, rb);
        j.AccumulatedImpulse += impulse;
    }

    private static void SolvePointToPoint(
        PhysicsJoint j, RigidBody a, RigidBody? b,
        Vector3 wa, Vector3 wb, float invDt, bool lockOrientation)
    {
        Vector3 err = wb - wa;
        Vector3 ra = wa - a.Position;
        Vector3 rb = b != null ? wb - b.Position : Vector3.Zero;
        Vector3 va = a.LinearVelocity + Vector3.Cross(a.AngularVelocity, ra);
        Vector3 vb = b != null ? b.LinearVelocity + Vector3.Cross(b.AngularVelocity, rb) : Vector3.Zero;
        Vector3 dv = vb - va;
        float erp = Math.Clamp(0.2f + j.Stiffness * 0.05f, 0.05f, 0.8f);

        // Solve each axis independently (stable industrial subset).
        for (int axis = 0; axis < 3; axis++)
        {
            Vector3 n = axis switch
            {
                0 => Vector3.UnitX,
                1 => Vector3.UnitY,
                _ => Vector3.UnitZ
            };
            float kn = SoftDenominator(EffectiveMass(a, b, ra, rb, n), j.Compliance, invDt);
            if (kn < 1e-8f) continue;
            float bias = Vector3.Dot(err, n) * erp * invDt + j.Damping * Vector3.Dot(dv, n);
            float lambda = -(Vector3.Dot(dv, n) + bias) / kn;
            Vector3 impulse = n * lambda;
            ApplyImpulse(a, -impulse, ra);
            if (b != null) ApplyImpulse(b, impulse, rb);
            j.AccumulatedImpulse += impulse;
        }

        if (lockOrientation && b != null)
        {
            // Soft angular lock: damp relative angular velocity (stronger when compliance is low).
            float soft = 1f / (1f + MathF.Max(0f, j.Compliance) * 10f);
            Vector3 wRel = b.AngularVelocity - a.AngularVelocity;
            a.AngularVelocity += wRel * (0.25f * soft);
            b.AngularVelocity -= wRel * (0.25f * soft);
        }
    }

    /// <summary>PGS denominator with optional CFM compliance term.</summary>
    private static float SoftDenominator(float effectiveMass, float compliance, float invDt)
        => effectiveMass + MathF.Max(0f, compliance) * invDt;

    private static void SolveHinge(
        PhysicsJoint j, RigidBody a, RigidBody? b,
        Vector3 wa, Vector3 wb, float invDt)
    {
        // Point-to-point + keep axes aligned (two orthogonal constraints).
        SolvePointToPoint(j, a, b, wa, wb, invDt, lockOrientation: false);
        Vector3 axisA = Vector3.Normalize(Vector3.Transform(j.LocalAxisA, a.Orientation));
        Vector3 axisB = b != null
            ? Vector3.Normalize(Vector3.Transform(j.LocalAxisB, b.Orientation))
            : Vector3.Normalize(j.LocalAxisB);
        Vector3 cross = Vector3.Cross(axisA, axisB);
        if (cross.LengthSquared() < 1e-8f) return;

        // Torque impulses to align axes.
        if (b != null)
        {
            a.AngularVelocity -= cross * 0.5f;
            b.AngularVelocity += cross * 0.5f;
        }
        else
        {
            a.AngularVelocity -= cross;
        }
    }

    private static void SolveSlider(
        PhysicsJoint j, RigidBody a, RigidBody? b,
        Vector3 wa, Vector3 wb, float invDt)
    {
        Vector3 axis = Vector3.Normalize(Vector3.Transform(j.LocalAxisA, a.Orientation));
        Vector3 d = wb - wa;
        // Constrain the two axes orthogonal to the slider direction.
        Vector3 t1 = Vector3.Normalize(Vector3.Cross(axis, MathF.Abs(axis.Y) < 0.9f ? Vector3.UnitY : Vector3.UnitX));
        Vector3 t2 = Vector3.Cross(axis, t1);
        Vector3 ra = wa - a.Position;
        Vector3 rb = b != null ? wb - b.Position : Vector3.Zero;

        foreach (var n in new[] { t1, t2 })
        {
            float kn = SoftDenominator(EffectiveMass(a, b, ra, rb, n), j.Compliance, invDt);
            if (kn < 1e-8f) continue;
            float bias = Vector3.Dot(d, n) * 0.2f * invDt;
            Vector3 va = a.LinearVelocity + Vector3.Cross(a.AngularVelocity, ra);
            Vector3 vb = b != null ? b.LinearVelocity + Vector3.Cross(b.AngularVelocity, rb) : Vector3.Zero;
            float lambda = -(Vector3.Dot(vb - va, n) + bias) / kn;
            Vector3 impulse = n * lambda;
            ApplyImpulse(a, -impulse, ra);
            if (b != null) ApplyImpulse(b, impulse, rb);
        }

        // Optional travel limits along axis.
        float travel = Vector3.Dot(d, axis);
        if (travel < j.MinLimit || travel > j.MaxLimit)
        {
            float target = Math.Clamp(travel, j.MinLimit, j.MaxLimit);
            float err = travel - target;
            float kn = SoftDenominator(EffectiveMass(a, b, ra, rb, axis), j.Compliance, invDt);
            if (kn > 1e-8f)
            {
                float lambda = -(err * 0.2f * invDt) / kn;
                Vector3 impulse = axis * lambda;
                ApplyImpulse(a, -impulse, ra);
                if (b != null) ApplyImpulse(b, impulse, rb);
            }
        }
    }

    private static float EffectiveMass(RigidBody a, RigidBody? b, Vector3 ra, Vector3 rb, Vector3 n)
    {
        float kn = a.InverseMass
            + Vector3.Dot(Vector3.Cross(ra, n) * a.InverseInertiaDiagonal, Vector3.Cross(ra, n));
        if (b != null)
        {
            kn += b.InverseMass
                + Vector3.Dot(Vector3.Cross(rb, n) * b.InverseInertiaDiagonal, Vector3.Cross(rb, n));
        }
        return kn;
    }

    private static void ApplyImpulse(RigidBody body, Vector3 impulse, Vector3 r)
    {
        if (body.InverseMass <= 0f) return;
        body.IsSleeping = false;
        body.LinearVelocity += impulse * body.InverseMass;
        Vector3 ang = Vector3.Cross(r, impulse);
        body.AngularVelocity += new Vector3(
            ang.X * body.InverseInertiaDiagonal.X,
            ang.Y * body.InverseInertiaDiagonal.Y,
            ang.Z * body.InverseInertiaDiagonal.Z);
    }

    private static void ApplyPositionalCorrection(
        RigidBody a, RigidBody? b, Vector3 wa, Vector3 wb, Vector3 correction)
    {
        float imSum = a.InverseMass + (b?.InverseMass ?? 0f);
        if (imSum < 1e-8f) return;
        if (a.InverseMass > 0f)
        {
            a.Position -= correction * (a.InverseMass / imSum);
            a.UpdateAabb();
        }
        if (b != null && b.InverseMass > 0f)
        {
            b.Position += correction * (b.InverseMass / imSum);
            b.UpdateAabb();
        }
    }
}
