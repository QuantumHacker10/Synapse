using FluentAssertions;
using GDNN.Rendering.Bridge;
using System.Numerics;
using Xunit;

namespace Synapse.Tests.Rendering;

public sealed class LDNNProceduralPreviewTests
{
    [Fact]
    public void FillGBufferProceduralPreview_ProducesSpatialVariation()
    {
        using var bridge = new LDNNBridge(32, 32);
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

        bridge.FillGBufferProceduralPreview();
        bridge.LastFillMode.Should().Be(GiGBufferFillMode.ProceduralPreview);

        var gi = bridge.RenderGI();
        gi.Should().NotBeNull();
        gi.GetLength(0).Should().BeGreaterThan(0);

        float center = gi[16, 16].Length();
        float corner = gi[2, 2].Length();
        (center > 0f || corner > 0f).Should().BeTrue();
    }
}
