using System.Numerics;
using FluentAssertions;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Physics;

public sealed class RigidBodyWorldLifecycleTests
{
    [Fact]
    public void Capsule_OnPlane_AndRemoveClear()
    {
        var world = new RigidBodyWorld
        {
            Gravity = new Vector3(0, -9.81f, 0),
            EnableSleeping = true
        };

        world.AddBody(new RigidBody
        {
            Name = "Ground",
            Type = BodyType.Static,
            Collider = new Collider { Shape = ColliderShape.Plane },
            Position = Vector3.Zero
        });

        var capsule = world.AddBody(new RigidBody
        {
            Name = "Capsule",
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Capsule, Size = new Vector3(0.3f, 1f, 0.3f) },
            Position = new Vector3(0, 3f, 0),
            Material = new PhysicsMaterial { Restitution = 0.05f, Friction = 0.4f }
        });
        capsule.SetMass(2f);

        for (int i = 0; i < 120; i++)
            world.Step(1f / 60f);

        capsule.Position.Y.Should().BeLessThan(3f);

        var id = capsule.Id;
        world.RemoveBody(id).Should().BeTrue();
        world.GetBody(id).Should().BeNull();
        world.Clear();
        world.GetBody(id).Should().BeNull();
    }

    [Fact]
    public void DynamicBox_ReceivesImpulse()
    {
        var world = new RigidBodyWorld { Gravity = Vector3.Zero };
        var box = world.AddBody(new RigidBody
        {
            Name = "Box",
            Type = BodyType.Dynamic,
            Collider = new Collider { Shape = ColliderShape.Box, Size = new Vector3(1, 1, 1) },
            Position = Vector3.Zero
        });
        box.SetMass(1f);
        box.ApplyImpulse(new Vector3(5, 0, 0), Vector3.Zero);
        world.Step(1f / 60f);
        box.LinearVelocity.X.Should().BeGreaterThan(0);
    }
}
