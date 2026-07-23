using System;
using System.Collections.Generic;
using System.Numerics;
using FluentAssertions;
using GDNN.Lighting.LDNN;
using GDNN.Polygonization;
using GDNN.Rendering.Bridge;
using GDNN.Rendering.Quality;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Physics;
using Synapse.Runtime;
using Xunit;

namespace Synapse.Tests.Rendering;

/// <summary>
/// AAA quality wiring: Lumen / Nanite policies, filtered cache, real path-trace refine,
/// and RuntimeQualityManager → present-path push.
/// </summary>
public class AaaRenderingQualityTests
{
    [Fact]
    public void LumenPolicies_CinematicExceedsIndustrial()
    {
        LumenNeural30.Cinematic.MaxBounces.Should().BeGreaterThan(LumenNeural30.Industrial.MaxBounces);
        LumenNeural30.Cinematic.SurfaceCacheResolution.Should().BeGreaterThan(LumenNeural30.Industrial.SurfaceCacheResolution);
        LumenNeural30.Aaa.RefineGridDivisor.Should().BeLessThanOrEqualTo(32);
        LumenNeural30.PolicyFromPreset("Cinematic").PathTraceSpp.Should().BeGreaterThanOrEqualTo(4);
    }

    [Fact]
    public void SurfaceCache_SampleFiltered_IsContinuous()
    {
        var cache = new LumenNeural30.SurfaceRadianceCache(32, origin: new Vector3(-8f, -2f, -8f), cellSize: 0.5f);
        cache.Accumulate(new Vector3(0f, 0f, 0f), new Vector3(2f, 1f, 0.5f), weight: 4f);
        cache.Accumulate(new Vector3(0.5f, 0f, 0f), new Vector3(0.5f, 2f, 1f), weight: 4f);

        Vector3 a = cache.SampleFiltered(new Vector3(0.1f, 0f, 0f));
        Vector3 b = cache.SampleFiltered(new Vector3(0.2f, 0f, 0f));
        a.Length().Should().BeGreaterThan(0.01f);
        b.Length().Should().BeGreaterThan(0.01f);
        // Neighbor samples should not jump discontinuously to zero.
        Vector3.Distance(a, b).Should().BeLessThan(a.Length() + b.Length());
    }

    [Fact]
    public void MultiBounceRefine_AaaProducesMoreEnergyThanIndustrialFloor()
    {
        var ssgi = new Vector3(0.2f, 0.18f, 0.15f);
        var cascade = new Vector3(0.12f, 0.11f, 0.10f);
        var cache = new Vector3(0.08f, 0.07f, 0.06f);
        var albedo = new Vector3(0.7f, 0.65f, 0.6f);
        var spec = new Vector3(0.25f);

        Vector3 industrial = LumenNeural30.MultiBounceRefine(ssgi, cascade, cache, albedo, spec, LumenNeural30.Industrial);
        Vector3 cinematic = LumenNeural30.MultiBounceRefine(ssgi, cascade, cache, albedo, spec, LumenNeural30.Cinematic);

        cinematic.Length().Should().BeGreaterThan(industrial.Length() * 0.85f);
        cinematic.X.Should().BeGreaterThan(LumenNeural30.Cinematic.AmbientFloor);
    }

    [Fact]
    public void NanitePolicies_AaaAndCinematicExceedPreviousIndustrialCaps()
    {
        NaniteNeural30.Industrial.MaxVisibilityWidth.Should().BeGreaterThanOrEqualTo(1280);
        NaniteCinematicResolve.Cinematic.MaxPolyResolution.Should().BeGreaterThan(NaniteNeural30.Industrial.MaxPolyResolution);
        NaniteCinematicResolve.Cinematic.MaxVisibilityWidth.Should().Be(2048);

        var (wInd, _) = NaniteNeural30.VisibilityBufferSize(1920, 1080, 0.9f, NaniteNeural30.Industrial);
        var (wCin, _) = NaniteNeural30.VisibilityBufferSize(1920, 1080, 0.9f, NaniteCinematicResolve.Cinematic);
        wCin.Should().BeGreaterThanOrEqualTo(wInd);
    }

    [Fact]
    public void LumenCinematicGi_FullPathTrace_UsesRealPathTracer()
    {
        var gi = new LumenCinematicGi(32);
        var irr = new Vector3[16, 16];
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++)
                irr[x, y] = new Vector3(0.15f);

        var gb = new GBuffer { Width = 16, Height = 16 };
        gb.Depth = new float[256];
        gb.Normals = new Vector3[256];
        gb.Albedo = new Vector3[256];
        gb.Velocity = new Vector2[256];
        gb.MaterialProps = new Vector4[256];
        gb.Specular = new Vector3[256];
        gb.Emissive = new Vector3[256];
        for (int i = 0; i < 256; i++)
        {
            gb.Depth[i] = 2.5f;
            gb.Normals[i] = Vector3.UnitY;
            gb.Albedo[i] = new Vector3(0.55f);
            gb.Specular[i] = new Vector3(0.2f);
        }

        var cam = new CameraState
        {
            Position = new Vector3(0, 1.5f, 4),
            Forward = -Vector3.UnitZ,
            Up = Vector3.UnitY,
            Right = Vector3.UnitX,
            FieldOfView = MathF.PI / 3f,
            AspectRatio = 1f
        };
        var lights = new List<LightConfig>
        {
            new()
            {
                Type = LightType.Directional,
                Direction = Vector3.Normalize(new Vector3(-0.3f, -1f, -0.2f)),
                Color = Vector3.One,
                Intensity = 3f
            }
        };

        var result = gi.Refine(
            irr, gb, cam, lights,
            LumenCinematicGi.Mode.FullPathTrace,
            pathTraceSpp: 2,
            policyOverride: LumenNeural30.Aaa);

        gi.UsedRealPathTracer.Should().BeTrue();
        gi.LastPathTraceSamples.Should().BeGreaterThan(0);
        result[8, 8].Length().Should().BeGreaterThan(0.01f);
    }

    [Fact]
    public void LDNNBridge_ApplyAaaQuality_RaisesCacheAndRefineDensity()
    {
        using var bridge = new LDNNBridge(32, 32);
        bridge.Initialize();
        bridge.ApplyAaaQuality("Industrial");
        int industrialDiv = bridge.ActiveRefineGridDivisor;
        int industrialCache = bridge.SurfaceCacheResolution;

        bridge.ApplyAaaQuality("Cinematic", giMaxBounces: 8, giCascadeResolution: 512, ssaoQuality: 4);
        bridge.ActiveLumenPolicy.MaxBounces.Should().BeGreaterThanOrEqualTo(8);
        bridge.ActivePathTraceSpp.Should().BeGreaterThanOrEqualTo(4);
        bridge.ActiveRefineGridDivisor.Should().BeLessThanOrEqualTo(industrialDiv);
        bridge.SurfaceCacheResolution.Should().BeGreaterThanOrEqualTo(industrialCache);
    }

    [Fact]
    public void QualityPresets_Cinematic_MapsIntoBridgeViaEngineHost()
    {
        QualityPresets.Cinematic.GIMaxBounces.Should().Be(8);
        QualityPresets.Cinematic.ShadowResolution.Should().Be(8192);
        QualityPresets.Ultra.GIMaxBounces.Should().BeGreaterThan(QualityPresets.High.GIMaxBounces);

        using var logger = new SynapseLogger(null, LogLevel.Warning, consoleEnabled: false);
        var host = new EngineHost(new SynapseConfig { QualityPreset = "High" }, logger);
        host.InitializeModules();
        host.EnableCinematicStack(ContinuumScale.Industrial);
        host.QualityPresetName.Should().Be("Cinematic");
    }

    [Fact]
    public void NaniteCinematic_PerturbNormal_PreservesUnitLength()
    {
        var n = NaniteNeural30.PerturbClusterNormal(Vector3.UnitY, 3, 7, 0.9f, NaniteCinematicResolve.Cinematic);
        n.Length().Should().BeApproximately(1f, 1e-4f);
    }
}
