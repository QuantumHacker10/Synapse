using System;
using System.Numerics;
using FluentAssertions;
using GDNN.Lighting.LDNN;
using GDNN.Polygonization;
using GDNN.Rendering.FrameGraph;
using GDNN.Rendering.Upscaling;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Physics;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Runtime;

public class CinematicNativeStackTests
{
    [Fact]
    public void FrameGraph_IncludesMeshletResolveAndUpscalePasses()
    {
        var graph = FrameGraphFactory.CreateDefault();
        graph.Passes.Should().Contain(p => p.Name == "MeshletMaterialResolve");
        graph.Passes.Should().Contain(p => p.Name == "TemporalUpscale");
        // Order: GBuffer → MeshletResolve → … → PostTonemap → Upscale
        int gbuffer = -1, resolve = -1, post = -1, upscale = -1;
        for (int i = 0; i < graph.Passes.Count; i++)
        {
            if (graph.Passes[i].Name == "GBuffer")
                gbuffer = i;
            if (graph.Passes[i].Name == "MeshletMaterialResolve")
                resolve = i;
            if (graph.Passes[i].Name == "PostTonemap")
                post = i;
            if (graph.Passes[i].Name == "TemporalUpscale")
                upscale = i;
        }
        resolve.Should().BeGreaterThan(gbuffer);
        upscale.Should().BeGreaterThan(post);
    }

    [Fact]
    public void NaniteCinematic_FullResResolve_WritesAlbedo()
    {
        // Minimal empty resolve should clear buffers without throwing.
        var albedo = new Vector3[16];
        var normals = new Vector3[16];
        var roughness = new float[16];
        NaniteCinematicResolve.ResolveFullResMaterials(
            null, null, 0.8f, 4, 4, albedo, normals, roughness);
        albedo.Should().OnlyContain(v => v == Vector3.Zero);
    }

    [Fact]
    public void FsrSpatialUpscaler_DoublesResolution()
    {
        var up = new FsrSpatialUpscaler();
        up.Configure(2, 2, 4, 4);
        var src = new Vector3[]
        {
            new(1, 0, 0), new(0, 1, 0),
            new(0, 0, 1), new(1, 1, 0)
        };
        var dst = new Vector3[16];
        up.Upscale(src, dst, ReadOnlySpan<Vector2>.Empty);
        dst[0].Length().Should().BeGreaterThan(0f);
        dst[15].Length().Should().BeGreaterThanOrEqualTo(0f);
    }

    [Fact]
    public void UpscalerFactory_Auto_IsAvailable()
    {
        var u = UpscalerFactory.Create(UpscalerBackend.Auto);
        u.IsAvailable.Should().BeTrue();
        u.Name.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void GpuContinuumScheduler_Cinematic_ScalesParticles()
    {
        using var sched = new GpuContinuumScheduler();
        sched.Configure(ContinuumScale.Cinematic, 1f / 60f);
        sched.SphParticleCount.Should().Be(8192);
        sched.LbmGrid.X.Should().Be(96);
        sched.Step(1f / 60f);
        sched.LastSphMs.Should().BeGreaterThanOrEqualTo(0f);
        sched.TryGetSphPositions(out var xyz).Should().BeTrue();
        xyz.Length.Should().Be(8192 * 3);
    }

    [Fact]
    public void EnableCinematicStack_ArmsContinuumAndModules()
    {
        using var logger = new SynapseLogger(null, LogLevel.Warning, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { QualityPreset = "High" }, logger);
        host.InitializeModules();
        host.EnableCinematicStack(ContinuumScale.Industrial);

        host.Multiphysics.Should().NotBeNull();
        host.Multiphysics!.GpuContinuum.Should().NotBeNull();
        (host.Multiphysics.Config.EnabledModules & ContinuumModules.Lbm)
            .Should().Be(ContinuumModules.Lbm);
        host.QualityPresetName.Should().Be("Cinematic");
    }

    [Fact]
    public void LumenCinematicGi_HybridRefine_ProducesEnergy()
    {
        var gi = new LumenCinematicGi(32);
        var irr = new Vector3[8, 8];
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                irr[x, y] = new Vector3(0.2f);
        var gb = new GBuffer { Width = 8, Height = 8 };
        gb.Depth = new float[64];
        gb.Normals = new Vector3[64];
        gb.Albedo = new Vector3[64];
        gb.Velocity = new Vector2[64];
        gb.MaterialProps = new Vector4[64];
        gb.Specular = new Vector3[64];
        gb.Emissive = new Vector3[64];
        for (int i = 0; i < 64; i++)
        {
            gb.Depth[i] = 2f;
            gb.Normals[i] = Vector3.UnitY;
            gb.Albedo[i] = new Vector3(0.6f);
            gb.Specular[i] = new Vector3(0.2f);
        }
        var cam = new CameraState { Position = new Vector3(0, 1, 3), Forward = -Vector3.UnitZ };
        var result = gi.Refine(irr, gb, cam, new System.Collections.Generic.List<LightConfig>(),
            LumenCinematicGi.Mode.GpuSurfaceCache);
        result[4, 4].Length().Should().BeGreaterThan(0.01f);
    }

    [Fact]
    public void LumenCinematicGi_FullPathTrace_ScalesSamplesWithSpp()
    {
        var gi = new LumenCinematicGi(24);
        var irr = new Vector3[12, 12];
        for (int y = 0; y < 12; y++)
            for (int x = 0; x < 12; x++)
                irr[x, y] = new Vector3(0.12f);
        var gb = new GBuffer { Width = 12, Height = 12 };
        int n = 144;
        gb.Depth = new float[n];
        gb.Normals = new Vector3[n];
        gb.Albedo = new Vector3[n];
        gb.Velocity = new Vector2[n];
        gb.MaterialProps = new Vector4[n];
        gb.Specular = new Vector3[n];
        gb.Emissive = new Vector3[n];
        for (int i = 0; i < n; i++)
        {
            gb.Depth[i] = 2f;
            gb.Normals[i] = Vector3.UnitY;
            gb.Albedo[i] = new Vector3(0.5f);
        }
        var cam = new CameraState { Position = new Vector3(0, 1, 3), Forward = -Vector3.UnitZ, Up = Vector3.UnitY, Right = Vector3.UnitX };
        var lights = new System.Collections.Generic.List<LightConfig>
        {
            new() { Type = LightType.Directional, Direction = -Vector3.UnitY, Color = Vector3.One, Intensity = 2f }
        };
        gi.Refine(irr, gb, cam, lights, LumenCinematicGi.Mode.FullPathTrace, pathTraceSpp: 3);
        gi.UsedRealPathTracer.Should().BeTrue();
        gi.LastPathTraceSamples.Should().BeGreaterThan(3);
    }

    [Fact]
    public void NaniteCinematic_Policy_ExceedsIndustrialVisibility()
    {
        NaniteCinematicResolve.Cinematic.MaxVisibilityWidth
            .Should().BeGreaterThan(NaniteNeural30.Industrial.MaxVisibilityWidth);
        NaniteCinematicResolve.Cinematic.MaxPolyResolution
            .Should().BeGreaterThan(NaniteNeural30.Industrial.MaxPolyResolution);
    }

    [Fact]
    public void MeshShaderCompatGenerator_EmitsGlsl()
    {
        string glsl = global::GDNN.GPU.MeshShaderCompatGenerator.GenerateMaterialResolveGlsl();
        glsl.Should().Contain("local_size_x");
        glsl.Should().Contain("materialBuffer");
    }
}
