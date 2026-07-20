// ============================================================================
// Synapse Omnia — Physics/VehicleSystem.cs
// Raycast vehicle: chassis + 4 suspension rays + drive/steer/brake.
// Adapted to Synapse living-law worlds (no PhysX dependency).
// ============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;

namespace Synapse.Physics;

/// <summary>One wheel of a raycast vehicle.</summary>
public sealed class VehicleWheel
{
    public string Name { get; set; } = "Wheel";
    /// <summary>Wheel hard-point in chassis local space.</summary>
    public Vector3 LocalHardPoint { get; set; }
    public float Radius { get; set; } = 0.35f;
    public float RestLength { get; set; } = 0.4f;
    public float MaxTravel { get; set; } = 0.25f;
    public float Stiffness { get; set; } = 35000f;
    public float Damping { get; set; } = 4500f;
    public float Friction { get; set; } = 1.2f;
    public bool IsSteered { get; set; }
    public bool IsDriven { get; set; } = true;

    // Runtime
    public float Compression { get; set; }
    public bool InContact { get; set; }
    public Vector3 ContactPoint { get; set; }
    public Vector3 ContactNormal { get; set; } = Vector3.UnitY;
    public float SteerAngle { get; set; }
}

/// <summary>Industrial raycast vehicle controller for Synapse Omnia.</summary>
public sealed class VehicleController
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public string Name { get; set; } = "Vehicle";
    public Guid ChassisBodyId { get; set; }
    public List<VehicleWheel> Wheels { get; } = new();

    /// <summary>Throttle in [-1, 1].</summary>
    public float Throttle { get; set; }
    /// <summary>Steering in [-1, 1].</summary>
    public float Steer { get; set; }
    /// <summary>Brake in [0, 1].</summary>
    public float Brake { get; set; }

    public float MaxSteerAngle { get; set; } = 0.55f;
    public float MaxDriveForce { get; set; } = 8000f;
    public float MaxBrakeForce { get; set; } = 12000f;
    public Vector3 DownDirection { get; set; } = -Vector3.UnitY;

    public static VehicleController CreateDefaultCar(Guid chassisId)
    {
        var v = new VehicleController { ChassisBodyId = chassisId, Name = "Car" };
        float track = 0.75f;
        float wheelBase = 1.2f;
        float y = -0.2f;
        v.Wheels.Add(new VehicleWheel { Name = "FL", LocalHardPoint = new Vector3(-track, y, wheelBase), IsSteered = true, IsDriven = true });
        v.Wheels.Add(new VehicleWheel { Name = "FR", LocalHardPoint = new Vector3(track, y, wheelBase), IsSteered = true, IsDriven = true });
        v.Wheels.Add(new VehicleWheel { Name = "RL", LocalHardPoint = new Vector3(-track, y, -wheelBase), IsSteered = false, IsDriven = true });
        v.Wheels.Add(new VehicleWheel { Name = "RR", LocalHardPoint = new Vector3(track, y, -wheelBase), IsSteered = false, IsDriven = true });
        return v;
    }

    /// <summary>
    /// Steps suspension rays against the world's static geometry and applies
    /// spring-damper + lateral friction + drive/brake impulses to the chassis.
    /// </summary>
    public void Step(RigidBodyWorld world, float dt)
    {
        if (dt <= 0f)
            return;
        var chassis = world.GetBody(ChassisBodyId);
        if (chassis == null || chassis.Type != BodyType.Dynamic)
            return;

        float steerAngle = Steer * MaxSteerAngle;
        Vector3 down = Vector3.Normalize(Vector3.Transform(DownDirection, chassis.Orientation));

        for (int i = 0; i < Wheels.Count; i++)
        {
            var w = Wheels[i];
            w.SteerAngle = w.IsSteered ? steerAngle : 0f;

            Vector3 hardPoint = chassis.Position + Vector3.Transform(w.LocalHardPoint, chassis.Orientation);
            float rayLen = w.RestLength + w.MaxTravel + w.Radius;
            Vector3 rayEnd = hardPoint + down * rayLen;

            if (!TryRaycastStatic(world, hardPoint, rayEnd, out float toi, out Vector3 hitPos, out Vector3 hitNormal))
            {
                w.InContact = false;
                w.Compression = 0f;
                continue;
            }

            float hitDist = toi * rayLen;
            float suspensionLength = MathF.Max(0f, hitDist - w.Radius);
            float compression = w.RestLength - suspensionLength;
            compression = Math.Clamp(compression, 0f, w.RestLength + w.MaxTravel);
            w.Compression = compression;
            w.InContact = compression > 0f;
            w.ContactPoint = hitPos;
            w.ContactNormal = hitNormal;

            if (!w.InContact)
                continue;

            // Spring-damper along suspension.
            Vector3 relVel = chassis.LinearVelocity + Vector3.Cross(chassis.AngularVelocity, hardPoint - chassis.Position);
            float suspSpeed = Vector3.Dot(relVel, hitNormal);
            float force = w.Stiffness * compression - w.Damping * suspSpeed;
            Vector3 impulse = hitNormal * (force * dt);
            chassis.ApplyImpulse(impulse, hardPoint);

            // Wheel forward / side in world (steered).
            Quaternion steerRot = Quaternion.CreateFromAxisAngle(hitNormal, w.SteerAngle);
            Vector3 forward = Vector3.Normalize(Vector3.Transform(Vector3.UnitZ, chassis.Orientation * steerRot));
            Vector3 side = Vector3.Normalize(Vector3.Cross(hitNormal, forward));
            if (side.LengthSquared() < 1e-6f)
                side = Vector3.Normalize(Vector3.Transform(Vector3.UnitX, chassis.Orientation));

            // Lateral friction (kill side slip).
            float sideSpeed = Vector3.Dot(relVel, side);
            Vector3 latImpulse = -side * sideSpeed * chassis.Mass * 0.25f * w.Friction;
            chassis.ApplyImpulse(latImpulse, hardPoint);

            // Drive / brake along forward.
            float drive = 0f;
            if (w.IsDriven)
                drive += Throttle * MaxDriveForce * dt;
            drive -= MathF.Sign(Vector3.Dot(relVel, forward)) * Brake * MaxBrakeForce * dt;
            chassis.ApplyImpulse(forward * drive, hardPoint);
        }

        chassis.IsSleeping = false;
    }

    private static bool TryRaycastStatic(
        RigidBodyWorld world,
        Vector3 start,
        Vector3 end,
        out float toi,
        out Vector3 hitPos,
        out Vector3 hitNormal)
    {
        toi = 1f;
        hitPos = end;
        hitNormal = Vector3.UnitY;
        Vector3 delta = end - start;
        float best = 1f;
        bool hit = false;

        for (int i = 0; i < world.Bodies.Count; i++)
        {
            var b = world.Bodies[i];
            if (b.Type == BodyType.Dynamic)
                continue;

            if (b.Collider.Shape == ColliderShape.Plane)
            {
                float planeY = b.Position.Y;
                if (MathF.Abs(delta.Y) < 1e-8f)
                    continue;
                float t = (planeY - start.Y) / delta.Y;
                if (t < 0f || t > 1f || t >= best)
                    continue;
                best = t;
                hitPos = start + delta * t;
                hitNormal = Vector3.UnitY;
                hit = true;
                continue;
            }

            // AABB slab against box / cooked mesh AABB / sphere bounds.
            Vector3 center = b.Position + b.Collider.LocalOffset;
            Vector3 he = b.Collider.Shape switch
            {
                ColliderShape.Sphere => new Vector3(b.Collider.Size.X),
                ColliderShape.Capsule => new Vector3(b.Collider.Size.X, b.Collider.Size.Y + b.Collider.Size.X, b.Collider.Size.X),
                ColliderShape.ConvexHull or ColliderShape.TriangleMesh =>
                    (b.WorldAabb.Max - b.WorldAabb.Min) * 0.5f,
                _ => b.Collider.Size
            };
            if (TryRayAabb(start, delta, center - he, center + he, out float tEnter, out Vector3 n)
                && tEnter >= 0f && tEnter < best)
            {
                best = tEnter;
                hitPos = start + delta * tEnter;
                hitNormal = n;
                hit = true;
            }
        }

        toi = best;
        return hit;
    }

    private static bool TryRayAabb(Vector3 origin, Vector3 dir, Vector3 min, Vector3 max, out float tEnter, out Vector3 normal)
    {
        tEnter = 0f;
        normal = Vector3.UnitY;
        float tMin = 0f, tMax = 1f;
        int hitAxis = 1;
        float hitSign = 1f;

        for (int axis = 0; axis < 3; axis++)
        {
            float o = axis == 0 ? origin.X : axis == 1 ? origin.Y : origin.Z;
            float d = axis == 0 ? dir.X : axis == 1 ? dir.Y : dir.Z;
            float mn = axis == 0 ? min.X : axis == 1 ? min.Y : min.Z;
            float mx = axis == 0 ? max.X : axis == 1 ? max.Y : max.Z;
            if (MathF.Abs(d) < 1e-8f)
            {
                if (o < mn || o > mx)
                    return false;
                continue;
            }
            float inv = 1f / d;
            float t1 = (mn - o) * inv;
            float t2 = (mx - o) * inv;
            float sign = -1f;
            if (t1 > t2)
            { (t1, t2) = (t2, t1); sign = 1f; }
            if (t1 > tMin)
            { tMin = t1; hitAxis = axis; hitSign = sign; }
            tMax = MathF.Min(tMax, t2);
            if (tMin > tMax)
                return false;
        }
        if (tMin < 0f || tMin > 1f)
            return false;
        tEnter = tMin;
        normal = hitAxis switch
        {
            0 => new Vector3(hitSign, 0, 0),
            2 => new Vector3(0, 0, hitSign),
            _ => new Vector3(0, hitSign, 0)
        };
        return true;
    }
}
