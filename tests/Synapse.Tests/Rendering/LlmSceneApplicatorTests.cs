using FluentAssertions;
using GDNN.Lighting.LDNN;
using GDNN.Llm;
using GDNN.Rendering;
using GDNN.Scene;
using Xunit;

namespace Synapse.Tests.Rendering;

public class LlmSceneApplicatorTests
{
    [Fact]
    public void ApplyLighting_ProducesDirectionalLight()
    {
        var parameters = new LightingParams
        {
            DirectionalDirection = (0.2f, -0.9f, 0.1f),
            Color = "#FFFFFF",
            Intensity = 3f
        };

        var lights = LlmSceneApplicator.ApplyLighting(parameters);

        lights.Should().ContainSingle();
        lights[0].Type.Should().Be(LightType.Directional);
        lights[0].Intensity.Should().Be(3f);
        lights[0].Direction.Length().Should().BeApproximately(1f, 1e-4f);
    }

    [Fact]
    public void ApplyLighting_FromParsedLlmText_ProducesDirectionalLight()
    {
        const string llmText = """
            {
              "directionalDirection": [0.0, -1.0, 0.0],
              "color": "#FFAA88",
              "intensity": 1.5
            }
            """;

        StructuredOutputParser.TryParseLightingParams(llmText, out var parameters).Should().BeTrue();

        var lights = LlmSceneApplicator.ApplyLighting(parameters);

        lights.Should().ContainSingle(light => light.Type == LightType.Directional);
    }
}
