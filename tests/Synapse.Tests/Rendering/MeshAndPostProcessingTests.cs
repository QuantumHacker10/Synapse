using System.Collections.Generic;
using System.Numerics;
using FluentAssertions;
using GDNN.Rendering.LOD;
using GDNN.Rendering.MeshIO;
using GDNN.Rendering.PostProcess;
using Xunit;

namespace Synapse.Tests.Rendering;

public class MeshLoaderTests
{
    [Fact]
    public void MeshLoader_SupportedFormats_ShouldIncludeGltf()
    {
        var loader = new MeshLoader();

        loader.SupportedFormats.Should().Contain(".gltf");
    }

    [Fact]
    public void MeshLoader_SupportedFormats_ShouldIncludeFbx()
    {
        var loader = new MeshLoader();

        loader.SupportedFormats.Should().Contain(".fbx");
    }

    [Fact]
    public void MeshLoader_SupportedFormats_ShouldIncludeObj()
    {
        var loader = new MeshLoader();

        loader.SupportedFormats.Should().Contain(".obj");
    }
}

public class MeshLODSystemTests
{
    [Fact]
    public void LodGenerator_GenerateLevels_ShouldProduceMultipleLevels()
    {
        var generator = new LodGenerator();
        var vertices = new List<Vector3> { Vector3.Zero, Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ };
        var indices = new List<uint> { 0, 1, 2, 0, 2, 3 };

        var levels = generator.GenerateLevels(vertices, indices, targetLevelCount: 3);

        levels.Should().HaveCountGreaterOrEqualTo(3);
    }

    [Fact]
    public void LodGroup_ComputeScreenCoverage_ShouldReturnValidPercentage()
    {
        var group = new LodGroup { ReferenceRadius = 1.0f };
        var cameraPosition = new Vector3(0f, 0f, 10f);

        var coverage = group.ComputeScreenCoverage(
            Vector3.Zero,
            group.ReferenceRadius,
            cameraPosition,
            fovRadians: MathF.PI / 4f,
            viewportHeight: 1080f);

        coverage.Should().BeInRange(0.0f, 1.0f);
    }
}

public class PostProcessingPipelineTests
{
    [Fact]
    public void PostProcessingPipeline_DefaultConfig_ShouldHaveBloomSection()
    {
        var pipeline = new PostProcessingPipeline();

        pipeline.Config.Bloom.Should().NotBeNull();
    }

    [Fact]
    public void PostProcessingPipeline_Bloom_ShouldBeDisabledByDefault()
    {
        var pipeline = new PostProcessingPipeline();

        pipeline.Config.Bloom.Enabled.Should().BeFalse();
    }
}
