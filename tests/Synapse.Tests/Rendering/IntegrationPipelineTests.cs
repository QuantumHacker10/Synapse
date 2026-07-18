using FluentAssertions;
using GDNN.Lighting.LDNN;
using GDNN.Rendering.Bridge;
using Xunit;

namespace Synapse.Tests.Rendering;

/// <summary>
/// Lot Intégration : G-Buffer étendu (velocity / material ID côté L-DNN),
/// configuration HybridRT, cache bridge.
/// </summary>
public class IntegrationPipelineTests
{
    [Fact]
    public void GBuffer_HoldsVelocityAndMaterialIdSlots()
    {
        const int w = 4, h = 4;
        var gbuffer = new GBuffer
        {
            Width = w,
            Height = h,
            Depth = new float[w * h],
            Normals = new System.Numerics.Vector3[w * h],
            Albedo = new System.Numerics.Vector3[w * h],
            Velocity = new System.Numerics.Vector2[w * h],
            MaterialProps = new System.Numerics.Vector4[w * h],
            Specular = new System.Numerics.Vector3[w * h],
            Emissive = new System.Numerics.Vector3[w * h]
        };

        gbuffer.Velocity[0] = new System.Numerics.Vector2(0.1f, -0.2f);
        gbuffer.MaterialProps[0] = new System.Numerics.Vector4(0.5f, 0.0f, 3f, 0f);

        var sample = gbuffer.GetSample(0, 0);
        sample.Velocity.Should().Be(new System.Numerics.Vector2(0.1f, -0.2f));
        sample.MaterialID.Should().Be(3);
    }

    [Fact]
    public void Bridge_HybridRT_QualityModeIsDefault()
    {
        using var bridge = new LDNNBridge(8, 8);
        bridge.Initialize();
        bridge.Renderer.Config.QualityMode.Should().Be(LDNNQualityMode.HybridRT);
        bridge.Renderer.Config.GIComputationMode.Should().Be(GIComputationMode.Hybrid);
    }

    [Fact]
    public void Bridge_StaticCache_WorksAlongsideTeacherConfig()
    {
        using var bridge = new LDNNBridge(4, 4);
        bridge.Initialize();
        bridge.Resize(4, 4);
        bridge.FillGBufferFromConstants(5f, System.Numerics.Vector3.UnitY, System.Numerics.Vector3.One);
        bridge.EnableStaticSceneCache = true;

        bridge.RenderGI();
        bridge.RenderGI();
        bridge.StaticCacheHits.Should().BeGreaterThan(0);
    }
}
