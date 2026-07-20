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
    Plane = 3
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
            _ => Collider.Size
        };
        WorldAabb = new Aabb(center - extents, center + extents);
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

    public Vector3 Gravity { get; set; } = new(0f, -9.81f, 0f);
    public int VelocityIterations { get; set; } = 10;
    public int PositionIterations { get; set; } = 3;
    public float SleepLinearThreshold { get; set; } = 0.05f;
    public float SleepAngularThreshold { get; set; } = 0.05f;
    public float SleepTime { get; set; } = 0.5f;
    public float Baumgarte { get; set; } = 0.2f;
    public float Slop { get; set; } = 0.005f;
    public bool EnableSleeping { get; set; } = true;

    public IReadOnlyList<RigidBody> Bodies => _bodies;
    public IReadOnlyList<ContactManifold> Manifolds => _manifolds;
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
    }

    /// <summary>Advances the world by <paramref name="dt"/> seconds.</summary>
    public void Step(float dt)
    {
        if (dt <= 0f) return;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        IntegrateForces(dt);
        BroadAndNarrowPhase();
        SolveVelocityConstraints(dt);
        IntegrateVelocities(dt);
        SolvePositionConstraints();
        UpdateSleepState(dt);
        ClearForcesAll();

        sw.Stop();
        CollectStats((float)sw.Elapsed.TotalMilliseconds);
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

        if (a.Collider.Shape == ColliderShape.Sphere || b.Collider.Shape == ColliderShape.Sphere)
            return GenerateSphereBox(a, b, manifold);

        return GenerateBoxBox(a, b, manifold);
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
