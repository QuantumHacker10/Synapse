using System;
using System.Collections.Generic;
using System.Numerics;
using FluentAssertions;
using GDNN.Platform;
using GDNN.Rendering.MeshIO;
using Synapse.Physics;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Industrial;

public class NativePlatformTests
{
    [Fact]
    public void Probe_ShouldReportPreferredGlfwBackend()
    {
        NativePlatform.InvalidateProbeCache();
        var caps = NativePlatform.Probe();
        caps.Rid.Should().NotBeNullOrWhiteSpace();
        caps.PreferredBackend.Should().Be(NativeSurfaceBackend.Glfw);
        caps.Summary.Should().Contain("GLFW");
    }

    [Fact]
    public void CreatePrimarySurfaceFactory_ShouldReturnGlfw()
    {
        // May throw if GLFW is truly absent — then skip via exception type check.
        try
        {
            var factory = NativePlatform.CreatePrimarySurfaceFactory();
            factory.Backend.Should().Be(NativeSurfaceBackend.Glfw);
        }
        catch (DllNotFoundException)
        {
            // Headless CI without GLFW is acceptable; probe still documents the gap.
            NativePlatform.Probe().GlfwAvailable.Should().BeFalse();
        }
    }
}

public class JointAndVehicleTests
{
    [Fact]
    public void DistanceJoint_ShouldKeepBodiesNearRestLength()
    {
        var world = new RigidBodyWorld { Gravity = Vector3.Zero, EnableSleeping = false, EnableCcd = false };
        var a = world.AddBody(new RigidBody
        {
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(0.2f) },
            Position = new Vector3(-1, 0, 0)
        });
        a.SetMass(1f);
        var b = world.AddBody(new RigidBody
        {
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(0.2f) },
            Position = new Vector3(1, 0, 0)
        });
        b.SetMass(1f);

        world.AddJoint(new PhysicsJoint
        {
            Type = JointType.Distance,
            BodyA = a.Id,
            BodyB = b.Id,
            RestLength = 1.0f,
            Stiffness = 2f,
            Damping = 0.5f
        });

        for (int i = 0; i < 120; i++)
            world.Step(1f / 60f);

        float dist = Vector3.Distance(a.Position, b.Position);
        dist.Should().BeInRange(0.7f, 1.4f);
    }

    [Fact]
    public void HingeJoint_ShouldBeAddableAndStepStable()
    {
        var world = new RigidBodyWorld { Gravity = new Vector3(0, -9.81f, 0), EnableSleeping = false };
        var anchor = world.AddBody(new RigidBody
        {
            Type = BodyType.Static,
            Collider = new Collider { Shape = ColliderShape.Box, Size = new Vector3(0.1f) },
            Position = Vector3.Zero
        });
        var door = world.AddBody(new RigidBody
        {
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Box, Size = new Vector3(0.5f, 1f, 0.05f) },
            Position = new Vector3(0.5f, 0, 0)
        });
        door.SetMass(20f);

        world.AddJoint(new PhysicsJoint
        {
            Type = JointType.Hinge,
            BodyA = door.Id,
            BodyB = Guid.Empty,
            LocalAnchorA = new Vector3(-0.5f, 0, 0),
            LocalAnchorB = Vector3.Zero,
            LocalAxisA = Vector3.UnitY,
            LocalAxisB = Vector3.UnitY
        });

        for (int i = 0; i < 60; i++)
            world.Step(1f / 60f);

        float.IsFinite(door.Position.Y).Should().BeTrue();
        world.Joints.Should().HaveCount(1);
        _ = anchor;
    }

    [Fact]
    public void Vehicle_OnPlane_ShouldReceiveSuspensionSupport()
    {
        var world = new RigidBodyWorld { Gravity = new Vector3(0, -9.81f, 0), EnableSleeping = false };
        world.AddBody(new RigidBody
        {
            Type = BodyType.Static,
            Collider = new Collider { Shape = ColliderShape.Plane },
            Position = Vector3.Zero
        });
        var chassis = world.AddBody(new RigidBody
        {
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Box, Size = new Vector3(1f, 0.3f, 2f) },
            Position = new Vector3(0, 1.0f, 0)
        });
        chassis.SetMass(1200f);

        var vehicle = VehicleController.CreateDefaultCar(chassis.Id);
        vehicle.Throttle = 0.2f;
        world.AddVehicle(vehicle);

        for (int i = 0; i < 90; i++)
            world.Step(1f / 60f);

        chassis.Position.Y.Should().BeGreaterThan(0.2f);
        vehicle.Wheels.Should().Contain(w => w.InContact || w.Compression >= 0f);
    }
}

public class MeshProviderTests
{
    [Fact]
    public void CookConvexHull_FromCube_ShouldProduceHullCollider()
    {
        var cube = MeshAsset.CreateUnitCube(1f);
        var source = SynapseMeshProvider.ToCollisionSource("cube", cube);
        var collider = MeshCollisionCooker.CookConvexHull(source);

        collider.Shape.Should().Be(ColliderShape.ConvexHull);
        collider.HullVertices.Should().NotBeNull();
        collider.HullVertices!.Length.Should().BeGreaterThanOrEqualTo(4);
        collider.SourceMeshId.Should().Be("cube");
    }

    [Fact]
    public void CookTriangleMesh_FromCube_ShouldProduceMeshCollider()
    {
        var cube = MeshAsset.CreateUnitCube(1f);
        var source = SynapseMeshProvider.ToCollisionSource("cube", cube);
        var collider = MeshCollisionCooker.CookTriangleMesh(source);

        collider.Shape.Should().Be(ColliderShape.TriangleMesh);
        collider.MeshVertices.Should().NotBeNull();
        collider.MeshIndices.Should().NotBeNull();
        collider.MeshIndices!.Length.Should().BeGreaterThanOrEqualTo(12);
    }

    [Fact]
    public void SynapseMeshProvider_RegisterAndCreateBody()
    {
        var provider = new SynapseMeshProvider();
        var cube = MeshAsset.CreateUnitCube(2f);
        provider.RegisterAsset("prop", cube);

        var body = provider.CreateBodyFromMesh("prop", BodyType.Dynamic, new Vector3(0, 2, 0), mass: 5f, name: "Prop");
        body.Should().NotBeNull();
        body!.Collider.Shape.Should().Be(ColliderShape.ConvexHull);
        body.Mass.Should().BeApproximately(5f, 0.01f);
    }
}

public class SoftConstraintAndMeshContactTests
{
    [Fact]
    public void SoftDistanceJoint_UnderGravity_ShouldHangLowerThanHard()
    {
        float HangHeight(float compliance)
        {
            var world = new RigidBodyWorld
            {
                Gravity = new Vector3(0, -9.81f, 0),
                EnableSleeping = false,
                EnableCcd = false
            };
            var anchor = world.AddBody(new RigidBody
            {
                Type = BodyType.Static,
                Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(0.05f) },
                Position = new Vector3(0, 2, 0)
            });
            var bob = world.AddBody(new RigidBody
            {
                Type = BodyType.Dynamic,
                Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(0.15f) },
                Position = new Vector3(0, 1, 0)
            });
            bob.SetMass(1f);

            world.AddJoint(new PhysicsJoint
            {
                Type = JointType.Distance,
                BodyA = bob.Id,
                BodyB = Guid.Empty,
                LocalAnchorA = Vector3.Zero,
                LocalAnchorB = anchor.Position,
                RestLength = 1.0f,
                Stiffness = 1.2f,
                Damping = 0.3f,
                Compliance = compliance
            });

            for (int i = 0; i < 240; i++)
                world.Step(1f / 60f);

            return bob.Position.Y;
        }

        float hardY = HangHeight(0f);
        float softY = HangHeight(0.35f);
        // Soft CFM yields more stretch under gravity → bob hangs lower.
        softY.Should().BeLessThan(hardY - 0.03f);
        hardY.Should().BeGreaterThan(0.7f);
    }

    [Fact]
    public void SphereTriangleMesh_ShouldGeneratePreciseContact()
    {
        var world = new RigidBodyWorld { Gravity = Vector3.Zero, EnableSleeping = false, EnableCcd = false };
        // Flat floor triangle pair in XZ, y=0.
        var floor = world.AddBody(new RigidBody
        {
            Type = BodyType.Static,
            Position = Vector3.Zero,
            Collider = new Collider
            {
                Shape = ColliderShape.TriangleMesh,
                MeshVertices =
                [
                    new Vector3(-2, 0, -2),
                    new Vector3(2, 0, -2),
                    new Vector3(2, 0, 2),
                    new Vector3(-2, 0, 2)
                ],
                MeshIndices = [0, 1, 2, 0, 2, 3],
                Size = new Vector3(2, 0.01f, 2)
            }
        });
        floor.SetMass(0f);

        var ball = world.AddBody(new RigidBody
        {
            Type = BodyType.Dynamic,
            Position = new Vector3(0, 0.3f, 0),
            Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(0.5f) }
        });
        ball.SetMass(1f);

        world.Step(1f / 60f);
        world.Manifolds.Should().NotBeEmpty();
        world.Manifolds[0].Points.Should().NotBeEmpty();
        world.Manifolds[0].Points[0].Penetration.Should().BeGreaterThan(0.1f);
        // After a few steps the sphere should rest above the mesh plane.
        for (int i = 0; i < 90; i++)
            world.Step(1f / 60f);
        ball.Position.Y.Should().BeGreaterThan(0.35f);
    }

    [Fact]
    public void ClosestPointOnTriangle_AtVertex_ReturnsVertex()
    {
        var a = new Vector3(0, 0, 0);
        var b = new Vector3(1, 0, 0);
        var c = new Vector3(0, 1, 0);
        var p = new Vector3(-1, -1, 0);
        var closest = RigidBodyWorld.ClosestPointOnTriangle(p, a, b, c);
        closest.Should().Be(a);
    }
}
