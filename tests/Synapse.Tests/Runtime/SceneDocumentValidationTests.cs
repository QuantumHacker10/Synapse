using System;
using System.IO;
using System.Threading.Tasks;
using FluentAssertions;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class SceneDocumentValidationTests
{
    [Fact]
    public void ToJson_RoundTripsDemoScene()
    {
        var scene = SceneDocument.CreateDemo();
        var json = scene.ToJson();
        json.Should().Contain(scene.Name);
        json.Should().Contain("entities");
        json.Length.Should().BeGreaterThan(100);
    }

    [Fact]
    public void ToJson_ThrowsOnNonPositiveScale()
    {
        var scene = SceneDocument.CreateDemo();
        scene.Entities[0].Scale = new Vec3(1, 0, 1);
        var act = () => scene.ToJson();
        act.Should().Throw<InvalidDataException>().WithMessage("*Scale*");
    }

    [Fact]
    public void ToJson_ThrowsOnNaNPosition()
    {
        var scene = SceneDocument.CreateDemo();
        scene.Entities[0].Position = new Vec3(float.NaN, 0, 0);
        var act = () => scene.ToJson();
        act.Should().Throw<InvalidDataException>().WithMessage("*finite*");
    }

    [Fact]
    public void Validate_RejectsBadCameraFov()
    {
        var scene = SceneDocument.CreateDemo();
        scene.Camera.Fov = 0.5f;
        var act = () => scene.Validate();
        act.Should().Throw<InvalidDataException>().WithMessage("*Fov*");
    }

    [Fact]
    public void Validate_RejectsBrokenJointBodyA()
    {
        var scene = SceneDocument.CreateDemo();
        scene.Joints.Add(new SceneJointData
        {
            Name = "bad-joint",
            Type = "Hinge",
            BodyA = Guid.NewGuid(),
            BodyB = Guid.Empty
        });
        var act = () => scene.Validate();
        act.Should().Throw<InvalidDataException>().WithMessage("*BodyA*");
    }

    [Fact]
    public void Validate_RejectsDuplicateEntityIds()
    {
        var scene = SceneDocument.CreateDemo();
        var dup = scene.Entities[0].Id;
        scene.Entities.Add(new SceneEntityData
        {
            Id = dup,
            Name = "Dup",
            Type = "Mesh",
            Scale = new Vec3(1, 1, 1)
        });
        var act = () => scene.Validate();
        act.Should().Throw<InvalidDataException>().WithMessage("*Duplicate*");
    }

    [Fact]
    public void Validate_RejectsMeshPathTraversal()
    {
        var scene = SceneDocument.CreateDemo();
        scene.Entities[0].MeshPath = "../secret.glb";
        var act = () => scene.Validate();
        act.Should().Throw<InvalidDataException>().WithMessage("*..*");
    }

    [Fact]
    public async Task SaveAndLoad_RoundTripsThroughTempFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-doc-{Guid.NewGuid():N}.synapse");
        try
        {
            var original = SceneDocument.CreateDemo();
            original.Name = "RoundTrip";
            await original.SaveAsync(path);
            var loaded = await SceneDocument.LoadAsync(path);
            loaded.Name.Should().Be("RoundTrip");
            loaded.Entities.Should().HaveCount(original.Entities.Count);
            loaded.ActiveLawId.Should().Be(original.ActiveLawId);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}
