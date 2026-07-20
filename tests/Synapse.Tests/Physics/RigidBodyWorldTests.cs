using System;
using System.Collections.Generic;
using System.Numerics;
using FluentAssertions;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Physics;

public class RigidBodyWorldTests
{
    [Fact]
    public void Sphere_DroppedOnPlane_ShouldComeToRestAboveGround()
    {
        var world = new RigidBodyWorld
        {
            Gravity = new Vector3(0, -9.81f, 0),
            VelocityIterations = 12,
            EnableSleeping = true
        };

        world.AddBody(new RigidBody
        {
            Name = "Ground",
            Type = BodyType.Static,
            Collider = new Collider { Shape = ColliderShape.Plane },
            Position = Vector3.Zero
        });

        var ball = world.AddBody(new RigidBody
        {
            Name = "Ball",
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(0.5f) },
            Position = new Vector3(0, 4f, 0),
            Material = new PhysicsMaterial { Restitution = 0.1f, Friction = 0.5f }
        });
        ball.SetMass(1f);

        for (int i = 0; i < 300; i++)
            world.Step(1f / 60f);

        ball.Position.Y.Should().BeGreaterThan(0.4f);
        ball.Position.Y.Should().BeLessThan(1.2f);
        MathF.Abs(ball.LinearVelocity.Y).Should().BeLessThan(0.5f);
    }

    [Fact]
    public void TwoFreeBodies_NoGravity_ShouldConserveMomentum()
    {
        var world = new RigidBodyWorld
        {
            Gravity = Vector3.Zero,
            EnableSleeping = false,
            VelocityIterations = 8
        };

        var a = world.AddBody(new RigidBody
        {
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(0.5f) },
            Position = new Vector3(-2, 0, 0),
            LinearVelocity = new Vector3(2f, 0, 0),
            Material = new PhysicsMaterial { Restitution = 1f, Friction = 0f }
        });
        a.SetMass(1f);

        var b = world.AddBody(new RigidBody
        {
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Sphere, Size = new Vector3(0.5f) },
            Position = new Vector3(2, 0, 0),
            LinearVelocity = new Vector3(-1f, 0, 0),
            Material = new PhysicsMaterial { Restitution = 1f, Friction = 0f }
        });
        b.SetMass(1f);

        Vector3 p0 = world.ComputeLinearMomentum();

        for (int i = 0; i < 120; i++)
            world.Step(1f / 120f);

        Vector3 p1 = world.ComputeLinearMomentum();
        (p1 - p0).Length().Should().BeLessThan(0.05f);
    }

    [Fact]
    public void BoxBox_Overlapping_ShouldGenerateContact()
    {
        var world = new RigidBodyWorld { Gravity = Vector3.Zero, EnableSleeping = false };

        var a = world.AddBody(new RigidBody
        {
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Box, Size = new Vector3(0.5f) },
            Position = new Vector3(0, 0, 0)
        });
        a.SetMass(1f);

        var b = world.AddBody(new RigidBody
        {
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Box, Size = new Vector3(0.5f) },
            Position = new Vector3(0.6f, 0, 0)
        });
        b.SetMass(1f);

        world.Step(1f / 60f);

        world.LastStats.ContactCount.Should().BeGreaterThan(0);
        world.LastStats.BroadPhasePairs.Should().BeGreaterThan(0);
    }
}

public class MultiphysicsOrchestratorTests
{
    [Fact]
    public void Step_ShouldAdvanceLivingLawAndRigidBodies()
    {
        var compiler = new LivingLawCompiler();
        var field = new PhysicsField(8, "test");
        for (int z = 0; z < 8; z++)
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 8; x++)
                    field.Temperature[x, y, z] = 300f;

        var result = compiler.Compile("T", "heat");
        result.Success.Should().BeTrue();

        var orch = new MultiphysicsOrchestrator(compiler, field);
        orch.SetActiveLaw("heat");
        orch.SeedDemoBodies();

        float y0 = orch.RigidWorld.Bodies[1].Position.Y;
        orch.Step(1f / 30f).Should().BeTrue();
        orch.LastStats.SubSteps.Should().BeGreaterThan(0);
        orch.RigidWorld.Bodies[1].Position.Y.Should().BeLessThan(y0);
        orch.LastStats.AverageTemperature.Should().BeGreaterThan(0f);
    }

    [Fact]
    public void SyncFromEntities_ShouldCreateGroundAndDynamicBodies()
    {
        var compiler = new LivingLawCompiler();
        var field = new PhysicsField(4, "sync");
        var orch = new MultiphysicsOrchestrator(compiler, field);

        orch.SyncFromEntities(new[]
        {
            new PhysicsEntityDesc
            {
                Id = Guid.Parse("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                Name = "Ground",
                Type = "Mesh",
                Position = Vector3.Zero,
                Scale = new Vector3(10, 0.2f, 10),
                IsStatic = true
            },
            new PhysicsEntityDesc
            {
                Id = Guid.Parse("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                Name = "Crate",
                Type = "Mesh",
                Position = new Vector3(0, 2, 0),
                Scale = new Vector3(1, 1, 1),
                Mass = 5f
            }
        });

        orch.RigidWorld.Bodies.Should().HaveCountGreaterOrEqualTo(2);
        orch.RigidWorld.Bodies.Should().Contain(b => b.Collider.Shape == ColliderShape.Plane);
        orch.RigidWorld.Bodies.Should().Contain(b => b.Type == BodyType.Dynamic);
    }
}
