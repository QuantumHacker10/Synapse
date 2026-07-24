using System.Collections.Generic;
using System.Numerics;
using FluentAssertions;
using GDNN.Lighting.LDNN;
using GDNN.Rendering.Bridge;
using Xunit;

namespace Synapse.Tests.Rendering;

/// <summary>
/// Verifies the present-path wiring of the L-DNN radiance cascades: after a Hybrid GI frame on a
/// lit procedural G-buffer, the cascade buffers are actually filled and consumable through
/// <see cref="RadianceCascadesManager.SampleScreenCascade"/> / <c>SampleHemisphereIrradiance</c>.
/// </summary>
public sealed class LdnnCascadeSampleTests
{
    private const int Size = 32;

    private static LDNNBridge BuildLitBridge()
    {
        var bridge = new LDNNBridge(Size, Size);
        bridge.Initialize();
        bridge.UpdateCamera(
            Matrix4x4.Identity,
            Matrix4x4.Identity,
            new Vector3(0, 1.5f, 3f),
            Vector3.Normalize(new Vector3(0, -0.2f, -1f)),
            Vector3.UnitX,
            Vector3.UnitY,
            60f,
            1f,
            0.1f,
            100f);

        // A bright key light + fill so the cascade ray tracer accumulates non-zero radiance.
        bridge.AddDirectionalLight(new Vector3(-0.4f, -1f, -0.3f), new Vector3(1f, 0.95f, 0.85f), 3.0f);
        bridge.AddPointLight(new Vector3(0f, 2.5f, 1.5f), new Vector3(0.8f, 0.85f, 1f), 6.0f, 25f);

        bridge.FillGBufferProceduralPreview();
        return bridge;
    }

    [Fact]
    public void SampleScreenCascade_ReturnsNonZero_AfterHybridFrame()
    {
        using var bridge = BuildLitBridge();
        _ = bridge.RenderGI();

        var cascades = bridge.Renderer.RadianceCascades;
        cascades.Should().NotBeNull();
        cascades.IsInitialized.Should().BeTrue();

        float maxScreen = 0f;
        var normal = Vector3.UnitY;
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                float u = (x + 0.5f) / Size;
                float v = (y + 0.5f) / Size;
                var s = cascades.SampleScreenCascade(u, v, normal);
                s.X.Should().BeGreaterThanOrEqualTo(0f);
                maxScreen = System.MathF.Max(maxScreen, s.Length());
            }
        }

        maxScreen.Should().BeGreaterThan(0f, "Hybrid GI must fill cascade buffers that SampleScreenCascade can read");

        // Hemisphere sampling must be a valid consumer of the same buffers: for every basis normal
        // it returns a finite, non-negative-weighted blend (it reads the face-center texels).
        foreach (var n in new[] { Vector3.UnitY, Vector3.UnitX, -Vector3.UnitZ, Vector3.UnitZ })
        {
            var h = cascades.SampleHemisphereIrradiance(n);
            float.IsNaN(h.Length()).Should().BeFalse();
        }
    }

    [Fact]
    public void SampleScreenCascade_IsDeterministic()
    {
        using var bridge = BuildLitBridge();
        _ = bridge.RenderGI();
        var cascades = bridge.Renderer.RadianceCascades;

        var a = cascades.SampleScreenCascade(0.5f, 0.5f, Vector3.UnitY);
        var b = cascades.SampleScreenCascade(0.5f, 0.5f, Vector3.UnitY);
        a.Should().Be(b);
    }

    [Fact]
    public void SampleScreenCascade_RejectsInvalidNormal()
    {
        using var bridge = BuildLitBridge();
        _ = bridge.RenderGI();
        var cascades = bridge.Renderer.RadianceCascades;

        cascades.SampleScreenCascade(0.5f, 0.5f, Vector3.Zero).Should().Be(Vector3.Zero);
        cascades.SampleHemisphereIrradiance(Vector3.Zero).Should().Be(Vector3.Zero);
    }

    [Fact]
    public void UpsertProbe_MakesProbeCacheReturnNonEmptyIrradiance()
    {
        var cache = new IrradianceCacheManager();
        cache.Initialize(IrradianceCacheType.SphericalHarmonics, 64, 2.0f, ProbeUpdateMode.OnDemand);

        cache.ActiveProbeCount.Should().Be(0);
        cache.TrilinearInterpolate(Vector3.Zero, Vector3.UnitY).Should().Be(Vector3.Zero);

        var irr = new Vector3(0.4f, 0.5f, 0.6f);
        cache.UpsertProbe(Vector3.Zero, irr, Vector3.UnitY);
        cache.ActiveProbeCount.Should().Be(1);

        var sampled = cache.TrilinearInterpolate(Vector3.Zero, Vector3.UnitY);
        sampled.Length().Should().BeGreaterThan(0f);

        // Upserting again within probe spacing refreshes the same probe rather than adding one.
        cache.UpsertProbe(new Vector3(0.5f, 0f, 0f), irr, Vector3.UnitY);
        cache.ActiveProbeCount.Should().Be(1);
    }
}
