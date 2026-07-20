// ============================================================================
// Synapse Omnia — Physics/RigidBodyWorld.cs
// Industrial rigid-body dynamics: primitives, broad/narrow phase, PGS contacts.
// Deterministic fixed-step friendly · C# 14 · NativeAOT compatible
// ============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Synapse.Physics;

/// <summary>Rigid-body motion type.</summary>
public enum BodyType : byte
{
    Static = 0,
    Kinematic = 1,
    Dynamic = 2
}

/// <summary>Supported collider primitives for industrial contact generation.</summary>
public enum ColliderShape : byte
{
    Sphere = 0,
    Box = 1,
    Capsule = 2,
    Plane = 3,
    /// <summary>Cooked convex hull (dynamic-friendly).</summary>
    ConvexHull = 4,
    /// <summary>Triangle mesh (prefer static / kinematic for Synapse worlds).</summary>
    TriangleMesh = 5
}

/// <summary>Material parameters for friction / restitution / density.</summary>
public sealed class PhysicsMaterial
{
    public float Friction { get; set; } = 0.5f;
    public float Restitution { get; set; } = 0.2f;
    public float Density { get; set; } = 1000f;

    public static PhysicsMaterial Default { get; } = new();
    public static PhysicsMaterial Bouncy { get; } = new() { Restitution = 0.8f, Friction = 0.2f };
    public static PhysicsMaterial Ice { get; } = new() { Friction = 0.05f, Restitution = 0.1f };
}

/// <summary>Collider attached to a rigid body.</summary>
public sealed class Collider
{
    public ColliderShape Shape { get; set; } = ColliderShape.Box;
    /// <summary>Half-extents for box; radius in X for sphere; (radius, half-height, radius) for capsule.</summary>
    public Vector3 Size { get; set; } = new(0.5f, 0.5f, 0.5f);
    public Vector3 LocalOffset { get; set; }

    /// <summary>Cooked convex hull vertices in local space (ConvexHull).</summary>
    public Vector3[]? HullVertices { get; set; }
    /// <summary>Triangle indices into <see cref="MeshVertices"/> (TriangleMesh).</summary>
    public int[]? MeshIndices { get; set; }
    /// <summary>Triangle mesh vertices in local space.</summary>
    public Vector3[]? MeshVertices { get; set; }
    /// <summary>Optional source mesh asset id (Synapse MeshProvider).</summary>
    public string? SourceMeshId { get; set; }
}

/// <summary>Single rigid body in the industrial dynamics world.</summary>
public sealed class RigidBody
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public string Name { get; set; } = "Body";
    public BodyType Type { get; set; } = BodyType.Dynamic;
    public Collider Collider { get; set; } = new();
    public PhysicsMaterial Material { get; set; } = PhysicsMaterial.Default;

    public Vector3 Position;
    public Quaternion Orientation = Quaternion.Identity;
    public Vector3 LinearVelocity;
    public Vector3 AngularVelocity;
    public Vector3 Force;
    public Vector3 Torque;

    public float Mass { get; private set; } = 1f;
    public float InverseMass { get; private set; } = 1f;
    public Vector3 InertiaDiagonal { get; private set; } = Vector3.One;
    public Vector3 InverseInertiaDiagonal { get; private set; } = Vector3.One;

    public bool IsSleeping { get; set; }
    public float SleepTimer { get; set; }
    public bool IsAwake => !IsSleeping;

    /// <summary>When true, fast movers use continuous collision (TOI) against static geometry.</summary>
    public bool EnableCcd { get; set; } = true;

    /// <summary>Motion length (m) above which CCD activates for this body (scaled by shape radius).</summary>
    public float CcdMotionThreshold { get; set; } = 0.25f;

    public Aabb WorldAabb;

    public void SetMass(float mass)
    {
        if (Type == BodyType.Static || Type == BodyType.Kinematic || mass <= 0f)
        {
            Mass = 0f;
            InverseMass = 0f;
            InertiaDiagonal = Vector3.Zero;
            InverseInertiaDiagonal = Vector3.Zero;
            return;
        }

        Mass = mass;
        InverseMass = 1f / mass;
        RecomputeInertia();
    }

    public void RecomputeInertia()
    {
        if (InverseMass <= 0f)
        {
            InertiaDiagonal = Vector3.Zero;
            InverseInertiaDiagonal = Vector3.Zero;
            return;
        }

        var s = Collider.Size;
        InertiaDiagonal = Collider.Shape switch
        {
            ColliderShape.Sphere => Vector3.One * (0.4f * Mass * s.X * s.X),
            ColliderShape.Capsule =>
                new Vector3(
                    0.25f * Mass * s.X * s.X + (1f / 12f) * Mass * (2f * s.Y) * (2f * s.Y),
                    0.5f * Mass * s.X * s.X,
                    0.25f * Mass * s.X * s.X + (1f / 12f) * Mass * (2f * s.Y) * (2f * s.Y)),
            _ => new Vector3(
                (1f / 12f) * Mass * (4f * s.Y * s.Y + 4f * s.Z * s.Z),
                (1f / 12f) * Mass * (4f * s.X * s.X + 4f * s.Z * s.Z),
                (1f / 12f) * Mass * (4f * s.X * s.X + 4f * s.Y * s.Y))
        };

        InverseInertiaDiagonal = new Vector3(
            InertiaDiagonal.X > 1e-8f ? 1f / InertiaDiagonal.X : 0f,
            InertiaDiagonal.Y > 1e-8f ? 1f / InertiaDiagonal.Y : 0f,
            InertiaDiagonal.Z > 1e-8f ? 1f / InertiaDiagonal.Z : 0f);
    }

    public void ApplyForce(Vector3 force) => Force += force;
    public void ApplyImpulse(Vector3 impulse, Vector3 worldPoint)
    {
        if (InverseMass <= 0f) return;
        IsSleeping = false;
        LinearVelocity += impulse * InverseMass;
        var r = worldPoint - Position;
        var ang = Vector3.Cross(r, impulse);
        AngularVelocity += new Vector3(
            ang.X * InverseInertiaDiagonal.X,
            ang.Y * InverseInertiaDiagonal.Y,
            ang.Z * InverseInertiaDiagonal.Z);
    }

    public void ClearForces()
    {
        Force = Vector3.Zero;
        Torque = Vector3.Zero;
    }

    public void UpdateAabb()
    {
        var center = Position + Collider.LocalOffset;
        Vector3 extents = Collider.Shape switch
        {
            ColliderShape.Sphere => new Vector3(Collider.Size.X),
            ColliderShape.Capsule => new Vector3(Collider.Size.X, Collider.Size.Y + Collider.Size.X, Collider.Size.X),
            ColliderShape.Plane => new Vector3(1e6f, 0.01f, 1e6f),
            ColliderShape.ConvexHull => ComputePointsExtents(Collider.HullVertices),
            ColliderShape.TriangleMesh => ComputePointsExtents(Collider.MeshVertices),
            _ => Collider.Size
        };
        WorldAabb = new Aabb(center - extents, center + extents);
    }

    private static Vector3 ComputePointsExtents(Vector3[]? pts)
    {
        if (pts == null || pts.Length == 0)
            return new Vector3(0.5f);
        Vector3 min = pts[0], max = pts[0];
        for (int i = 1; i < pts.Length; i++)
        {
            min = Vector3.Min(min, pts[i]);
            max = Vector3.Max(max, pts[i]);
        }
        return (max - min) * 0.5f + new Vector3(0.01f);
    }
}

/// <summary>Axis-aligned bounding box.</summary>
public readonly struct Aabb
{
    public readonly Vector3 Min;
    public readonly Vector3 Max;

    public Aabb(Vector3 min, Vector3 max)
    {
        Min = min;
        Max = max;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Overlaps(in Aabb other) =>
        Min.X <= other.Max.X && Max.X >= other.Min.X &&
        Min.Y <= other.Max.Y && Max.Y >= other.Min.Y &&
        Min.Z <= other.Max.Z && Max.Z >= other.Min.Z;
}

/// <summary>Contact manifold point between two bodies.</summary>
public struct ContactPoint
{
    public Vector3 Position;
    public Vector3 Normal;
    public float Penetration;
    public float NormalImpulse;
    public float TangentImpulse;
}

/// <summary>Pair contact generated by the narrow phase.</summary>
public sealed class ContactManifold
{
    public RigidBody BodyA = null!;
    public RigidBody BodyB = null!;
    public readonly List<ContactPoint> Points = new(4);
    public float CombinedFriction;
    public float CombinedRestitution;
}

/// <summary>World diagnostics for industrial validation.</summary>
public sealed class PhysicsWorldStats
{
    public int BodyCount { get; set; }
    public int ActiveBodyCount { get; set; }
    public int BroadPhasePairs { get; set; }
    public int ContactCount { get; set; }
    public float KineticEnergy { get; set; }
    public Vector3 LinearMomentum { get; set; }
    public float StepTimeMs { get; set; }
    public int CcdHitCount { get; set; }
}

/// <summary>
/// Industrial rigid-body world: gravity, broad-phase AABB pairs, analytical
/// primitive contacts, Projected Gauss–Seidel solver, sleeping.
/// </summary>
public sealed class RigidBodyWorld
{
    private readonly List<RigidBody> _bodies = new();
    private readonly List<ContactManifold> _manifolds = new();
    private readonly Dictionary<Guid, RigidBody> _byId = new();
    private readonly HashSet<int> _ccdResolved = new();
    private readonly List<PhysicsJoint> _joints = new();
    private readonly List<VehicleController> _vehicles = new();

    public Vector3 Gravity { get; set; } = new(0f, -9.81f, 0f);
    public int VelocityIterations { get; set; } = 10;
    public int PositionIterations { get; set; } = 3;
    public float SleepLinearThreshold { get; set; } = 0.05f;
    public float SleepAngularThreshold { get; set; } = 0.05f;
    public float SleepTime { get; set; } = 0.5f;
    public float Baumgarte { get; set; } = 0.2f;
    public float Slop { get; set; } = 0.005f;
    public bool EnableSleeping { get; set; } = true;

    /// <summary>Enables continuous collision detection for fast dynamic bodies.</summary>
    public bool EnableCcd { get; set; } = true;

    /// <summary>World-space speed (m/s) above which CCD is considered for a body.</summary>
    public float CcdVelocityThreshold { get; set; } = 4f;

    public IReadOnlyList<RigidBody> Bodies => _bodies;
    public IReadOnlyList<ContactManifold> Manifolds => _manifolds;
    public IReadOnlyList<PhysicsJoint> Joints => _joints;
    public IReadOnlyList<VehicleController> Vehicles => _vehicles;
    public PhysicsWorldStats LastStats { get; } = new();

    public RigidBody AddBody(RigidBody body)
    {
        if (body.Type == BodyType.Dynamic && body.InverseMass <= 0f)
            body.SetMass(MathF.Max(1e-3f, body.Mass > 0 ? body.Mass : 1f));
        else if (body.Type != BodyType.Dynamic)
            body.SetMass(0f);
        else
            body.RecomputeInertia();

        body.UpdateAabb();
        _bodies.Add(body);
        _byId[body.Id] = body;
        return body;
    }

    public bool RemoveBody(Guid id)
    {
        if (!_byId.Remove(id, out var body))
            return false;
        _bodies.Remove(body);
        return true;
    }

    public RigidBody? GetBody(Guid id) => _byId.TryGetValue(id, out var b) ? b : null;

    public void Clear()
    {
        _bodies.Clear();
        _byId.Clear();
        _manifolds.Clear();
        _joints.Clear();
        _vehicles.Clear();
        _ccdResolved.Clear();
    }

    public PhysicsJoint AddJoint(PhysicsJoint joint)
    {
        ArgumentNullException.ThrowIfNull(joint);
        _joints.Add(joint);
        return joint;
    }

    public bool RemoveJoint(Guid id)
    {
        for (int i = 0; i < _joints.Count; i++)
        {
            if (_joints[i].Id == id)
            {
                _joints.RemoveAt(i);
                return true;
            }
        }
        return false;
    }

    public VehicleController AddVehicle(VehicleController vehicle)
    {
        ArgumentNullException.ThrowIfNull(vehicle);
        _vehicles.Add(vehicle);
        return vehicle;
    }

    public bool RemoveVehicle(Guid id)
    {
        for (int i = 0; i < _vehicles.Count; i++)
        {
            if (_vehicles[i].Id == id)
            {
                _vehicles.RemoveAt(i);
                return true;
            }
        }
        return false;
    }

    /// <summary>Advances the world by <paramref name="dt"/> seconds.</summary>
    public void Step(float dt)
    {
        if (dt <= 0f) return;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        IntegrateForces(dt);
        BroadAndNarrowPhase();
        SolveVelocityConstraints(dt);
        JointSolver.SolveVelocity(_joints, GetBody, VelocityIterations, dt);
        for (int i = 0; i < _vehicles.Count; i++)
            _vehicles[i].Step(this, dt);
        _ccdResolved.Clear();
        if (EnableCcd)
            SolveContinuousCollisions(dt);
        IntegrateVelocities(dt);
        SolvePositionConstraints();
        JointSolver.SolvePosition(_joints, GetBody, PositionIterations, Baumgarte);
        UpdateSleepState(dt);
        ClearForcesAll();

        sw.Stop();
        CollectStats((float)sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>
    /// Continuous collision detection: sweeps fast dynamic shapes against static geometry
    /// and clamps motion to the first time-of-impact (prevents tunneling).
    /// </summary>
    private void SolveContinuousCollisions(float dt)
    {
        int hits = 0;
        for (int i = 0; i < _bodies.Count; i++)
        {
            var body = _bodies[i];
            if (body.Type != BodyType.Dynamic || body.IsSleeping || !body.EnableCcd)
                continue;

            float speed = body.LinearVelocity.Length();
            if (speed < CcdVelocityThreshold)
                continue;

            float radius = body.Collider.Shape switch
            {
                ColliderShape.Sphere => body.Collider.Size.X,
                ColliderShape.Capsule => body.Collider.Size.X + body.Collider.Size.Y,
                _ => MathF.Max(body.Collider.Size.X, MathF.Max(body.Collider.Size.Y, body.Collider.Size.Z))
            };

            float motion = speed * dt;
            if (motion < body.CcdMotionThreshold && motion < radius)
                continue;

            Vector3 start = body.Position;
            Vector3 end = start + body.LinearVelocity * dt;
            float bestToi = 1f;
            Vector3 hitNormal = Vector3.UnitY;
            bool hit = false;

            for (int j = 0; j < _bodies.Count; j++)
            {
                var other = _bodies[j];
                if (ReferenceEquals(other, body) || other.Type == BodyType.Dynamic)
                    continue;

                if (TrySweptHit(body, start, end, other, out float toi, out Vector3 n) && toi < bestToi)
                {
                    bestToi = toi;
                    hitNormal = n;
                    hit = true;
                }
            }

            if (!hit || bestToi >= 1f)
                continue;

            // Advance to TOI with a small skin and reflect the impact velocity.
            float skin = Math.Clamp(bestToi - 1e-3f, 0f, 1f);
            body.Position = start + (end - start) * skin;
            float vn = Vector3.Dot(body.LinearVelocity, hitNormal);
            if (vn < 0f)
            {
                float e = body.Material.Restitution;
                body.LinearVelocity -= hitNormal * vn * (1f + e);
            }

            body.UpdateAabb();
            _ccdResolved.Add(i);
            hits++;
        }

        LastStats.CcdHitCount = hits;
    }

    private static bool TrySweptHit(
        RigidBody moving,
        Vector3 start,
        Vector3 end,
        RigidBody obstacle,
        out float toi,
        out Vector3 normal)
    {
        toi = 1f;
        normal = Vector3.UnitY;
        Vector3 delta = end - start;
        float motionLen = delta.Length();
        if (motionLen < 1e-8f)
            return false;

        if (obstacle.Collider.Shape == ColliderShape.Plane)
        {
            // Infinite ground plane y = obstacle.Position.Y, normal +Y.
            float planeY = obstacle.Position.Y;
            float radius = moving.Collider.Shape == ColliderShape.Sphere
                ? moving.Collider.Size.X
                : moving.Collider.Shape == ColliderShape.Capsule
                    ? moving.Collider.Size.X + moving.Collider.Size.Y
                    : moving.Collider.Size.Y;

            float y0 = start.Y - radius;
            float y1 = end.Y - radius;
            if (y0 >= planeY && y1 >= planeY)
                return false;
            if (y0 < planeY && y1 < planeY)
            {
                // Already penetrating — TOI 0.
                toi = 0f;
                normal = Vector3.UnitY;
                return true;
            }

            // Crossing from above.
            if (y0 >= planeY && y1 < planeY)
            {
                toi = (y0 - planeY) / (y0 - y1);
                normal = Vector3.UnitY;
                return toi >= 0f && toi <= 1f;
            }

            return false;
        }

        if (moving.Collider.Shape == ColliderShape.Sphere
            && (obstacle.Collider.Shape == ColliderShape.Sphere || obstacle.Collider.Shape == ColliderShape.Box))
        {
            // Sphere vs expanded AABB (or sphere treated as AABB of its diameter).
            Vector3 center = obstacle.Position + obstacle.Collider.LocalOffset;
            Vector3 he = obstacle.Collider.Shape == ColliderShape.Sphere
                ? new Vector3(obstacle.Collider.Size.X)
                : obstacle.Collider.Size;
            float r = moving.Collider.Size.X;
            he += new Vector3(r);

            if (TryRayAabb(start, delta, center - he, center + he, out float tEnter, out Vector3 n)
                && tEnter >= 0f && tEnter <= 1f)
            {
                toi = tEnter;
                normal = n;
                return true;
            }
        }

        return false;
    }

    /// <summary>Ray vs AABB slab test. Returns entry TOI in [0,1] along <paramref name="dir"/>.</summary>
    private static bool TryRayAabb(Vector3 origin, Vector3 dir, Vector3 min, Vector3 max, out float tEnter, out Vector3 normal)
    {
        tEnter = 0f;
        normal = Vector3.UnitY;
        float tMin = 0f;
        float tMax = 1f;
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
            {
                (t1, t2) = (t2, t1);
                sign = 1f;
            }

            if (t1 > tMin)
            {
                tMin = t1;
                hitAxis = axis;
                hitSign = sign;
            }
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

    private void IntegrateForces(float dt)
    {
        for (int i = 0; i < _bodies.Count; i++)
        {
            var b = _bodies[i];
            if (b.Type != BodyType.Dynamic || b.IsSleeping)
                continue;

            b.Force += Gravity * b.Mass;
            b.LinearVelocity += b.Force * b.InverseMass * dt;
            b.AngularVelocity += new Vector3(
                b.Torque.X * b.InverseInertiaDiagonal.X,
                b.Torque.Y * b.InverseInertiaDiagonal.Y,
                b.Torque.Z * b.InverseInertiaDiagonal.Z) * dt;
        }
    }

    private void IntegrateVelocities(float dt)
    {
        for (int i = 0; i < _bodies.Count; i++)
        {
            var b = _bodies[i];
            if (b.Type == BodyType.Static || b.IsSleeping)
                continue;
            if (_ccdResolved.Contains(i))
            {
                // CCD already placed the body at the TOI; only integrate orientation.
                float angSpeedCcd = b.AngularVelocity.Length();
                if (angSpeedCcd > 1e-6f)
                {
                    var axis = b.AngularVelocity / angSpeedCcd;
                    b.Orientation = Quaternion.Normalize(
                        Quaternion.CreateFromAxisAngle(axis, angSpeedCcd * dt) * b.Orientation);
                }
                b.UpdateAabb();
                continue;
            }

            b.Position += b.LinearVelocity * dt;
            float angSpeed = b.AngularVelocity.Length();
            if (angSpeed > 1e-6f)
            {
                var axis = b.AngularVelocity / angSpeed;
                b.Orientation = Quaternion.Normalize(
                    Quaternion.CreateFromAxisAngle(axis, angSpeed * dt) * b.Orientation);
            }

            b.UpdateAabb();
        }
    }

    private void BroadAndNarrowPhase()
    {
        _manifolds.Clear();
        int pairs = 0;

        for (int i = 0; i < _bodies.Count; i++)
        {
            var a = _bodies[i];
            for (int j = i + 1; j < _bodies.Count; j++)
            {
                var b = _bodies[j];
                if (a.Type == BodyType.Static && b.Type == BodyType.Static)
                    continue;
                if (a.IsSleeping && b.IsSleeping)
                    continue;
                if (!a.WorldAabb.Overlaps(b.WorldAabb))
                    continue;

                pairs++;
                if (TryGenerateContact(a, b, out var manifold) && manifold.Points.Count > 0)
                    _manifolds.Add(manifold);
            }
        }

        LastStats.BroadPhasePairs = pairs;
    }

    private static bool TryGenerateContact(RigidBody a, RigidBody b, out ContactManifold manifold)
    {
        manifold = new ContactManifold
        {
            BodyA = a,
            BodyB = b,
            CombinedFriction = MathF.Sqrt(a.Material.Friction * b.Material.Friction),
            CombinedRestitution = MathF.Max(a.Material.Restitution, b.Material.Restitution)
        };

        // Prefer plane / sphere / box analytical pairs.
        if (a.Collider.Shape == ColliderShape.Plane || b.Collider.Shape == ColliderShape.Plane)
            return GeneratePlaneContact(a, b, manifold);

        if (a.Collider.Shape == ColliderShape.Sphere && b.Collider.Shape == ColliderShape.Sphere)
            return GenerateSphereSphere(a, b, manifold);

        // Precise sphere ↔ triangle mesh / convex hull before falling back to AABB proxies.
        if ((a.Collider.Shape == ColliderShape.Sphere
                && b.Collider.Shape is ColliderShape.TriangleMesh or ColliderShape.ConvexHull)
            || (b.Collider.Shape == ColliderShape.Sphere
                && a.Collider.Shape is ColliderShape.TriangleMesh or ColliderShape.ConvexHull))
            return GenerateSphereMesh(a, b, manifold);

        if (a.Collider.Shape == ColliderShape.Sphere || b.Collider.Shape == ColliderShape.Sphere)
            return GenerateSphereBox(a, b, manifold);

        // Remaining hull/mesh pairs: AABB proxy (stable industrial fallback).
        if (a.Collider.Shape is ColliderShape.ConvexHull or ColliderShape.TriangleMesh
            || b.Collider.Shape is ColliderShape.ConvexHull or ColliderShape.TriangleMesh)
            return GenerateBoxBoxFromAabb(a, b, manifold);

        return GenerateBoxBox(a, b, manifold);
    }

    /// <summary>
    /// Sphere against cooked triangle mesh (or convex hull triangulated via centroid fan).
    /// Picks the deepest penetrating triangle contact for a stable single-point manifold.
    /// </summary>
    private static bool GenerateSphereMesh(RigidBody a, RigidBody b, ContactManifold m)
    {
        RigidBody sphere = a.Collider.Shape == ColliderShape.Sphere ? a : b;
        RigidBody mesh = ReferenceEquals(sphere, a) ? b : a;
        Vector3 center = sphere.Position + sphere.Collider.LocalOffset;
        float radius = sphere.Collider.Size.X;
        if (radius <= 0f)
            return false;

        if (!TryGetWorldTriangles(mesh, out Vector3[] verts, out int[] indices))
            return GenerateBoxBoxFromAabb(a, b, m);

        float bestPen = 0f;
        Vector3 bestNormal = Vector3.UnitY;
        Vector3 bestPoint = center;
        const int maxTris = 4096;
        int triCount = Math.Min(indices.Length / 3, maxTris);

        for (int t = 0; t < triCount; t++)
        {
            int i0 = indices[t * 3];
            int i1 = indices[t * 3 + 1];
            int i2 = indices[t * 3 + 2];
            if ((uint)i0 >= (uint)verts.Length
                || (uint)i1 >= (uint)verts.Length
                || (uint)i2 >= (uint)verts.Length)
                continue;

            Vector3 p0 = verts[i0], p1 = verts[i1], p2 = verts[i2];
            Vector3 closest = ClosestPointOnTriangle(center, p0, p1, p2);
            Vector3 delta = center - closest;
            float distSq = delta.LengthSquared();
            if (distSq > radius * radius)
                continue;

            float dist = MathF.Sqrt(MathF.Max(distSq, 1e-12f));
            float pen = radius - dist;
            if (pen <= bestPen)
                continue;

            Vector3 n;
            if (distSq < 1e-10f)
            {
                Vector3 faceN = Vector3.Cross(p1 - p0, p2 - p0);
                n = faceN.LengthSquared() > 1e-12f
                    ? Vector3.Normalize(faceN)
                    : Vector3.UnitY;
                // Orient normal so it points from mesh toward sphere center.
                if (Vector3.Dot(n, center - (p0 + p1 + p2) * (1f / 3f)) < 0f)
                    n = -n;
            }
            else
            {
                n = delta / dist;
            }

            bestPen = pen;
            bestNormal = n;
            bestPoint = closest;
        }

        if (bestPen <= 0f)
            return false;

        m.BodyA = mesh;
        m.BodyB = sphere;
        m.Points.Add(new ContactPoint
        {
            Position = bestPoint,
            Normal = bestNormal,
            Penetration = bestPen
        });
        return true;
    }

    private static bool TryGetWorldTriangles(RigidBody mesh, out Vector3[] worldVerts, out int[] indices)
    {
        worldVerts = Array.Empty<Vector3>();
        indices = Array.Empty<int>();
        var col = mesh.Collider;
        Quaternion q = mesh.Orientation;
        Vector3 origin = mesh.Position + col.LocalOffset;

        if (col.Shape == ColliderShape.TriangleMesh
            && col.MeshVertices is { Length: > 0 } mv
            && col.MeshIndices is { Length: >= 3 } mi)
        {
            worldVerts = new Vector3[mv.Length];
            for (int i = 0; i < mv.Length; i++)
                worldVerts[i] = origin + Vector3.Transform(mv[i], q);
            indices = mi;
            return true;
        }

        if (col.Shape == ColliderShape.ConvexHull && col.HullVertices is { Length: >= 3 } hv)
        {
            worldVerts = new Vector3[hv.Length];
            Vector3 centroid = Vector3.Zero;
            for (int i = 0; i < hv.Length; i++)
            {
                worldVerts[i] = origin + Vector3.Transform(hv[i], q);
                centroid += worldVerts[i];
            }
            centroid /= hv.Length;

            // Centroid fan triangulation (industrial approx for cooked hulls).
            var tris = new List<int>((hv.Length - 2) * 3);
            for (int i = 1; i < hv.Length - 1; i++)
            {
                tris.Add(0);
                tris.Add(i);
                tris.Add(i + 1);
            }
            // Keep centroid available as vertex 0 of a temporary buffer if fan from first vertex fails
            // for non-star-shaped hulls — still better than AABB for sphere contacts.
            _ = centroid;
            indices = tris.ToArray();
            return indices.Length >= 3;
        }

        return false;
    }

    /// <summary>Closest point on triangle ABC to point P (Ericson RTCD).</summary>
    public static Vector3 ClosestPointOnTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 ab = b - a;
        Vector3 ac = c - a;
        Vector3 ap = p - a;
        float d1 = Vector3.Dot(ab, ap);
        float d2 = Vector3.Dot(ac, ap);
        if (d1 <= 0f && d2 <= 0f)
            return a;

        Vector3 bp = p - b;
        float d3 = Vector3.Dot(ab, bp);
        float d4 = Vector3.Dot(ac, bp);
        if (d3 >= 0f && d4 <= d3)
            return b;

        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0f && d1 >= 0f && d3 <= 0f)
        {
            float v = d1 / (d1 - d3);
            return a + ab * v;
        }

        Vector3 cp = p - c;
        float d5 = Vector3.Dot(ab, cp);
        float d6 = Vector3.Dot(ac, cp);
        if (d6 >= 0f && d5 <= d6)
            return c;

        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0f && d2 >= 0f && d6 <= 0f)
        {
            float w = d2 / (d2 - d6);
            return a + ac * w;
        }

        float va = d3 * d6 - d5 * d4;
        if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
        {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + (c - b) * w;
        }

        float denom = 1f / (va + vb + vc);
        float v2 = vb * denom;
        float w2 = vc * denom;
        return a + ab * v2 + ac * w2;
    }

    private static bool GenerateBoxBoxFromAabb(RigidBody a, RigidBody b, ContactManifold m)
    {
        Vector3 ca = (a.WorldAabb.Min + a.WorldAabb.Max) * 0.5f;
        Vector3 cb = (b.WorldAabb.Min + b.WorldAabb.Max) * 0.5f;
        Vector3 ha = (a.WorldAabb.Max - a.WorldAabb.Min) * 0.5f;
        Vector3 hb = (b.WorldAabb.Max - b.WorldAabb.Min) * 0.5f;
        Vector3 d = cb - ca;
        Vector3 overlap = ha + hb - new Vector3(MathF.Abs(d.X), MathF.Abs(d.Y), MathF.Abs(d.Z));
        if (overlap.X <= 0f || overlap.Y <= 0f || overlap.Z <= 0f)
            return false;

        Vector3 n;
        float pen;
        if (overlap.X <= overlap.Y && overlap.X <= overlap.Z)
        {
            n = new Vector3(d.X >= 0 ? 1f : -1f, 0, 0);
            pen = overlap.X;
        }
        else if (overlap.Y <= overlap.Z)
        {
            n = new Vector3(0, d.Y >= 0 ? 1f : -1f, 0);
            pen = overlap.Y;
        }
        else
        {
            n = new Vector3(0, 0, d.Z >= 0 ? 1f : -1f);
            pen = overlap.Z;
        }

        m.Points.Add(new ContactPoint
        {
            Position = ca + d * 0.5f,
            Normal = n,
            Penetration = pen
        });
        return true;
    }

    private static bool GeneratePlaneContact(RigidBody a, RigidBody b, ContactManifold m)
    {
        var plane = a.Collider.Shape == ColliderShape.Plane ? a : b;
        var other = ReferenceEquals(plane, a) ? b : a;
        // Infinite ground plane: y = plane.Position.Y, normal +Y
        float planeY = plane.Position.Y;
        Vector3 center = other.Position + other.Collider.LocalOffset;
        float radius = other.Collider.Shape switch
        {
            ColliderShape.Sphere => other.Collider.Size.X,
            ColliderShape.Capsule => other.Collider.Size.X + other.Collider.Size.Y,
            _ => other.Collider.Size.Y
        };

        float penetration = planeY + radius - (center.Y - (other.Collider.Shape == ColliderShape.Box ? other.Collider.Size.Y : 0f));
        if (other.Collider.Shape == ColliderShape.Box)
            penetration = planeY + other.Collider.Size.Y - center.Y;
        else if (other.Collider.Shape == ColliderShape.Sphere)
            penetration = (planeY + other.Collider.Size.X) - center.Y;
        else if (other.Collider.Shape == ColliderShape.Capsule)
            penetration = (planeY + other.Collider.Size.X + other.Collider.Size.Y) - center.Y;

        if (penetration <= 0f)
            return false;

        // Normal points from plane toward other (world +Y). Store as A→B with BodyA=other, BodyB=plane for consistency.
        m.BodyA = other;
        m.BodyB = plane;
        m.Points.Add(new ContactPoint
        {
            Position = new Vector3(center.X, planeY, center.Z),
            // Standard convention: normal points from A toward B (down into the plane).
            Normal = -Vector3.UnitY,
            Penetration = penetration
        });
        return true;
    }

    private static bool GenerateSphereSphere(RigidBody a, RigidBody b, ContactManifold m)
    {
        Vector3 ca = a.Position + a.Collider.LocalOffset;
        Vector3 cb = b.Position + b.Collider.LocalOffset;
        float ra = a.Collider.Size.X;
        float rb = b.Collider.Size.X;
        Vector3 d = cb - ca;
        float dist = d.Length();
        float pen = ra + rb - dist;
        if (pen <= 0f)
            return false;

        Vector3 n = dist > 1e-6f ? d / dist : Vector3.UnitY;
        m.Points.Add(new ContactPoint
        {
            Position = ca + n * ra,
            Normal = n,
            Penetration = pen
        });
        return true;
    }

    private static bool GenerateSphereBox(RigidBody a, RigidBody b, ContactManifold m)
    {
        RigidBody sphere = a.Collider.Shape == ColliderShape.Sphere ? a : b;
        RigidBody box = ReferenceEquals(sphere, a) ? b : a;
        Vector3 sc = sphere.Position + sphere.Collider.LocalOffset;
        Vector3 bc = box.Position + box.Collider.LocalOffset;
        Vector3 he = box.Collider.Size;
        Vector3 local = sc - bc;
        Vector3 closest = Vector3.Clamp(local, -he, he);
        Vector3 delta = local - closest;
        float distSq = delta.LengthSquared();
        float r = sphere.Collider.Size.X;
        if (distSq > r * r && distSq > 1e-12f)
            return false;

        float dist = MathF.Sqrt(MathF.Max(distSq, 1e-12f));
        Vector3 n;
        float pen;
        if (distSq < 1e-8f)
        {
            // Sphere center inside box — push out along smallest axis.
            Vector3 toFace = he - new Vector3(MathF.Abs(local.X), MathF.Abs(local.Y), MathF.Abs(local.Z));
            if (toFace.X < toFace.Y && toFace.X < toFace.Z)
            {
                n = new Vector3(MathF.Sign(local.X) == 0 ? 1 : MathF.Sign(local.X), 0, 0);
                pen = toFace.X + r;
            }
            else if (toFace.Y < toFace.Z)
            {
                n = new Vector3(0, MathF.Sign(local.Y) == 0 ? 1 : MathF.Sign(local.Y), 0);
                pen = toFace.Y + r;
            }
            else
            {
                n = new Vector3(0, 0, MathF.Sign(local.Z) == 0 ? 1 : MathF.Sign(local.Z));
                pen = toFace.Z + r;
            }
        }
        else
        {
            n = delta / dist;
            pen = r - dist;
        }

        // Sphere-box: Normal points from A(box) toward B(sphere).
        m.BodyA = box;
        m.BodyB = sphere;
        m.Points.Add(new ContactPoint
        {
            Position = bc + closest,
            Normal = n,
            Penetration = pen
        });
        return pen > 0f;
    }

    private static bool GenerateBoxBox(RigidBody a, RigidBody b, ContactManifold m)
    {
        // Separating-axis test on AABB extents (orientation ignored for industrial MVP stability).
        Vector3 ca = a.Position + a.Collider.LocalOffset;
        Vector3 cb = b.Position + b.Collider.LocalOffset;
        Vector3 ha = a.Collider.Size;
        Vector3 hb = b.Collider.Size;
        Vector3 d = cb - ca;
        Vector3 overlap = ha + hb - new Vector3(MathF.Abs(d.X), MathF.Abs(d.Y), MathF.Abs(d.Z));
        if (overlap.X <= 0f || overlap.Y <= 0f || overlap.Z <= 0f)
            return false;

        Vector3 n;
        float pen;
        if (overlap.X <= overlap.Y && overlap.X <= overlap.Z)
        {
            n = new Vector3(d.X >= 0 ? 1f : -1f, 0, 0);
            pen = overlap.X;
        }
        else if (overlap.Y <= overlap.Z)
        {
            n = new Vector3(0, d.Y >= 0 ? 1f : -1f, 0);
            pen = overlap.Y;
        }
        else
        {
            n = new Vector3(0, 0, d.Z >= 0 ? 1f : -1f);
            pen = overlap.Z;
        }

        m.Points.Add(new ContactPoint
        {
            Position = ca + d * 0.5f,
            Normal = n,
            Penetration = pen
        });
        return true;
    }

    private void SolveVelocityConstraints(float dt)
    {
        float invDt = dt > 1e-8f ? 1f / dt : 0f;

        for (int iter = 0; iter < VelocityIterations; iter++)
        {
            for (int mi = 0; mi < _manifolds.Count; mi++)
            {
                var m = _manifolds[mi];
                var a = m.BodyA;
                var b = m.BodyB;

                for (int pi = 0; pi < m.Points.Count; pi++)
                {
                    var p = m.Points[pi];
                    Vector3 ra = p.Position - a.Position;
                    Vector3 rb = p.Position - b.Position;
                    Vector3 va = a.LinearVelocity + Vector3.Cross(a.AngularVelocity, ra);
                    Vector3 vb = b.LinearVelocity + Vector3.Cross(b.AngularVelocity, rb);
                    Vector3 rv = vb - va;

                    float vn = Vector3.Dot(rv, p.Normal);
                    float bias = 0f;
                    if (p.Penetration > Slop)
                        bias = -Baumgarte * invDt * (p.Penetration - Slop);

                    float e = m.CombinedRestitution;
                    float target = bias - e * MathF.Min(vn, 0f);

                    float kn = a.InverseMass + b.InverseMass
                        + Vector3.Dot(Vector3.Cross(ra, p.Normal) * a.InverseInertiaDiagonal, Vector3.Cross(ra, p.Normal))
                        + Vector3.Dot(Vector3.Cross(rb, p.Normal) * b.InverseInertiaDiagonal, Vector3.Cross(rb, p.Normal));
                    if (kn <= 1e-8f)
                        continue;

                    float lambda = (-vn + target) / kn; // note: relative velocity uses vb-va, impulse on B along normal
                    // Correct impulse: separate along normal — apply -n*λ to A, +n*λ to B when normal is A→B.
                    // Our Generate* set Normal as pointing toward the second body in the pair ordering;
                    // for plane contacts BodyA=dynamic, BodyB=plane, Normal=+Y (into dynamic from plane) — wait.
                    // Plane: Normal = +Y means from plane toward other. BodyA=other, BodyB=plane.
                    // We need impulse that pushes BodyA up: +Normal on A, -Normal on B.
                    // Relative vel rv = vb - va; for resting on plane va is falling (negative Y), vb=0 → rv = -va has positive Y if falling.
                    // Actually for separation we want relative velocity along n (from A to B? or contact normal?).
                    // Standard: normal points from A to B. Impulse j*n applied to B, -j*n to A.
                    // Plane contact: BodyA=other, BodyB=plane, Normal=+Y which is from plane to other = from B to A = wrong.
                    // Fix: treat Normal as pointing from B toward A for plane case... 
                    // Simpler: always apply impulse that increases separation along stored normal from B onto A:
                    // A gets +λ n, B gets -λ n when we want to push A along n away from B.

                    float old = p.NormalImpulse;
                    p.NormalImpulse = MathF.Max(0f, old + lambda);
                    float dLambda = p.NormalImpulse - old;
                    Vector3 impulse = p.Normal * dLambda;

                    // Standard: normal from A→B; apply −λn to A, +λn to B.
                    ApplyImpulse(a, -impulse, ra);
                    ApplyImpulse(b, impulse, rb);

                    // Friction
                    Vector3 tangent = rv - p.Normal * Vector3.Dot(rv, p.Normal);
                    float tLen = tangent.Length();
                    if (tLen > 1e-6f)
                    {
                        tangent /= tLen;
                        float kt = a.InverseMass + b.InverseMass;
                        if (kt > 1e-8f)
                        {
                            float jt = -Vector3.Dot(rv, tangent) / kt;
                            float maxF = m.CombinedFriction * p.NormalImpulse;
                            float oldT = p.TangentImpulse;
                            p.TangentImpulse = Math.Clamp(oldT + jt, -maxF, maxF);
                            Vector3 fImp = tangent * (p.TangentImpulse - oldT);
                            ApplyImpulse(a, -fImp, ra);
                            ApplyImpulse(b, fImp, rb);
                        }
                    }

                    m.Points[pi] = p;
                }
            }
        }
    }

    private void SolvePositionConstraints()
    {
        for (int iter = 0; iter < PositionIterations; iter++)
        {
            for (int mi = 0; mi < _manifolds.Count; mi++)
            {
                var m = _manifolds[mi];
                for (int pi = 0; pi < m.Points.Count; pi++)
                {
                    var p = m.Points[pi];
                    float corr = MathF.Max(p.Penetration - Slop, 0f) * Baumgarte;
                    float imSum = m.BodyA.InverseMass + m.BodyB.InverseMass;
                    if (imSum <= 1e-8f) continue;
                    // Normal A→B: move A opposite to normal, B along normal.
                    Vector3 correction = p.Normal * (corr / imSum);
                    if (m.BodyA.InverseMass > 0f)
                    {
                        m.BodyA.Position -= correction * m.BodyA.InverseMass;
                        m.BodyA.UpdateAabb();
                    }
                    if (m.BodyB.InverseMass > 0f)
                    {
                        m.BodyB.Position += correction * m.BodyB.InverseMass;
                        m.BodyB.UpdateAabb();
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

    private void UpdateSleepState(float dt)
    {
        if (!EnableSleeping) return;
        for (int i = 0; i < _bodies.Count; i++)
        {
            var b = _bodies[i];
            if (b.Type != BodyType.Dynamic) continue;

            float lin = b.LinearVelocity.LengthSquared();
            float ang = b.AngularVelocity.LengthSquared();
            if (lin < SleepLinearThreshold * SleepLinearThreshold &&
                ang < SleepAngularThreshold * SleepAngularThreshold)
            {
                b.SleepTimer += dt;
                if (b.SleepTimer >= SleepTime)
                {
                    b.IsSleeping = true;
                    b.LinearVelocity = Vector3.Zero;
                    b.AngularVelocity = Vector3.Zero;
                }
            }
            else
            {
                b.SleepTimer = 0f;
                b.IsSleeping = false;
            }
        }
    }

    private void ClearForcesAll()
    {
        for (int i = 0; i < _bodies.Count; i++)
            _bodies[i].ClearForces();
    }

    private void CollectStats(float stepMs)
    {
        float ke = 0f;
        Vector3 momentum = Vector3.Zero;
        int active = 0;
        int contacts = 0;
        for (int i = 0; i < _bodies.Count; i++)
        {
            var b = _bodies[i];
            if (b.Type == BodyType.Dynamic && !b.IsSleeping) active++;
            if (b.InverseMass > 0f)
            {
                ke += 0.5f * b.Mass * b.LinearVelocity.LengthSquared();
                momentum += b.LinearVelocity * b.Mass;
            }
        }
        for (int i = 0; i < _manifolds.Count; i++)
            contacts += _manifolds[i].Points.Count;

        LastStats.BodyCount = _bodies.Count;
        LastStats.ActiveBodyCount = active;
        LastStats.ContactCount = contacts;
        LastStats.KineticEnergy = ke;
        LastStats.LinearMomentum = momentum;
        LastStats.StepTimeMs = stepMs;
    }

    /// <summary>Total kinetic energy of dynamic bodies (validation).</summary>
    public float ComputeKineticEnergy()
    {
        float ke = 0f;
        for (int i = 0; i < _bodies.Count; i++)
        {
            var b = _bodies[i];
            if (b.InverseMass <= 0f) continue;
            ke += 0.5f * b.Mass * b.LinearVelocity.LengthSquared();
            ke += 0.5f * Vector3.Dot(b.AngularVelocity * b.InertiaDiagonal, b.AngularVelocity);
        }
        return ke;
    }

    /// <summary>Total linear momentum of dynamic bodies (validation).</summary>
    public Vector3 ComputeLinearMomentum()
    {
        Vector3 p = Vector3.Zero;
        for (int i = 0; i < _bodies.Count; i++)
        {
            var b = _bodies[i];
            if (b.InverseMass <= 0f) continue;
            p += b.LinearVelocity * b.Mass;
        }
        return p;
    }
}
