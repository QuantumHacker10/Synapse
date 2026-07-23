using System;
using System.IO;
using System.Numerics;
using FluentAssertions;
using GDNN.Rendering.MeshIO;
using Xunit;

namespace Synapse.Tests.Runtime;

public sealed class UsdSkelAnimationTests
{
    [Fact]
    public void SkelAnimation_ParsesTimeSamplesAndEvaluates()
    {
        var path = Resolve("samples/meshes/skel_anim_wave.usda");
        var result = new UsdAsciiLoader().LoadLeafMeshAsync(path, null, default).GetAwaiter().GetResult();
        result.Success.Should().BeTrue(result.ErrorMessage);
        result.Asset!.Skeleton.Should().NotBeNull();
        result.Asset.AnimationClips.Should().ContainSingle(c => c.Name == "Wave");

        var clip = result.Asset.AnimationClips[0];
        clip.Duration.Should().BeApproximately(1f, 0.001f);
        clip.JointNames.Should().Equal("Root", "Root/Arm");
        clip.Curves.Should().HaveCount(2);

        var at0 = clip.Evaluate(0f);
        at0[1].Translation.Y.Should().BeApproximately(0f, 0.01f);

        var atMid = clip.Evaluate(0.5f);
        atMid[1].Translation.Y.Should().BeApproximately(0.5f, 0.01f);

        var mats = clip.EvaluateLocalMatrices(0.5f);
        mats.Should().HaveCount(2);
        mats[1].M41.Should().BeApproximately(1f, 0.05f); // translation X in row-major M41
    }

    [Fact]
    public void SkelAnimation_FindsAnimationSource()
    {
        var text = File.ReadAllText(Resolve("samples/meshes/skel_anim_wave.usda"));
        UsdSkelAnimationParser.ParseAnimationSourcePaths(text)
            .Should().Contain("/Wave");
        var clips = UsdSkelAnimationParser.ParseClips(text);
        UsdSkelAnimationParser.FindClip(clips, "/Wave")!.Name.Should().Be("Wave");
    }

    [Fact]
    public void SkelAnimation_StaticPoseClip()
    {
        const string usda = """
            def SkelAnimation "Bind"
            {
                uniform token[] joints = ["A", "B"]
                float3[] translations = [(0,0,0), (2,0,0)]
                quatf[] rotations = [(0,0,0,1), (0,0,0,1)]
                float3[] scales = [(1,1,1), (1,1,1)]
            }
            """;
        var clips = UsdSkelAnimationParser.ParseClips(usda);
        clips.Should().ContainSingle();
        clips[0].Duration.Should().Be(0f);
        clips[0].Evaluate(0)[1].Translation.X.Should().BeApproximately(2f, 0.001f);
    }

    [Fact]
    public void SkelAnimation_LerpBetweenKeys()
    {
        const string usda = """
            def SkelAnimation "Lerp"
            {
                uniform token[] joints = ["J"]
                float3[] translations.timeSamples = {
                    0: [(0, 0, 0)],
                    2: [(0, 10, 0)],
                }
            }
            """;
        var clip = UsdSkelAnimationParser.ParseClips(usda)[0];
        clip.Evaluate(1f)[0].Translation.Y.Should().BeApproximately(5f, 0.01f);
    }

    private static string Resolve(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }

        throw new FileNotFoundException(relative);
    }
}
