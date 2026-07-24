using System;
using System.Text;
using FluentAssertions;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class ScenePatchCodecTests
{
    [Fact]
    public void EncodeApply_PreservesPositionsAndScale()
    {
        var scene = SceneDocument.CreateDemo();
        scene.Entities[0].Position = new Vec3(1.5f, 2.5f, 3.5f);
        scene.Entities[0].Scale = new Vec3(2, 3, 4);

        var bytes = ScenePatchCodec.Encode(scene, 42);
        ScenePatchCodec.TryDecode(bytes, out var patch).Should().BeTrue();
        var target = new SceneDocument { Name = "dst", Camera = scene.Camera };
        // Seed camera FOV valid for later validation if needed
        target.Camera.Fov = 60;
        ScenePatchCodec.Apply(target, patch!).Should().Be(scene.Entities.Count);

        var e = target.Entities.Find(x => x.Id == scene.Entities[0].Id)!;
        e.Position.X.Should().BeApproximately(1.5f, 1e-5f);
        e.Position.Y.Should().BeApproximately(2.5f, 1e-5f);
        e.Position.Z.Should().BeApproximately(3.5f, 1e-5f);
        e.Scale.X.Should().BeApproximately(2f, 1e-5f);
        e.Scale.Y.Should().BeApproximately(3f, 1e-5f);
        e.Scale.Z.Should().BeApproximately(4f, 1e-5f);
    }

    [Fact]
    public void Apply_RemovedEntity_DeletesFromTarget()
    {
        var scene = SceneDocument.CreateDemo();
        var id = scene.Entities[1].Id;
        var patch = ScenePatchCodec.FromScene(scene, 1);
        patch.Entities.Clear();
        patch.Entities.Add(new ScenePatchCodec.EntityTransformPatch
        {
            Id = id.ToString("N"),
            Removed = true
        });

        var before = scene.Entities.Count;
        ScenePatchCodec.Apply(scene, patch).Should().Be(1);
        scene.Entities.Should().HaveCount(before - 1);
        scene.Entities.Exists(e => e.Id == id).Should().BeFalse();
    }

    [Fact]
    public void Apply_ReplaceMissingFalse_SkipsUnknownIds()
    {
        var target = new SceneDocument { Name = "empty", Camera = { Fov = 60 } };
        var patch = new ScenePatchCodec.ScenePatch
        {
            Sequence = 1,
            Entities =
            [
                new ScenePatchCodec.EntityTransformPatch
                {
                    Id = Guid.NewGuid().ToString("N"),
                    Name = "ghost",
                    X = 1, Y = 2, Z = 3
                }
            ]
        };

        ScenePatchCodec.Apply(target, patch, replaceMissing: false).Should().Be(0);
        target.Entities.Should().BeEmpty();
    }

    [Fact]
    public void TryDecode_RejectsGarbageAndVersionZero()
    {
        ScenePatchCodec.TryDecode(Encoding.UTF8.GetBytes("not-json"), out _).Should().BeFalse();
        ScenePatchCodec.TryDecode(Encoding.UTF8.GetBytes("""{"version":0,"entities":[]}"""), out _).Should().BeFalse();
    }

    [Fact]
    public void Apply_SkipsInvalidGuidIds()
    {
        var target = new SceneDocument { Name = "t", Camera = { Fov = 60 } };
        var patch = new ScenePatchCodec.ScenePatch
        {
            Entities =
            [
                new ScenePatchCodec.EntityTransformPatch { Id = "not-a-guid", Name = "bad" }
            ]
        };
        ScenePatchCodec.Apply(target, patch).Should().Be(0);
        target.Entities.Should().BeEmpty();
    }
}
