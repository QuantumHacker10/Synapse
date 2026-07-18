using FluentAssertions;
using GDNN.Llm;
using Xunit;

namespace Synapse.Tests.LLM;

public class HybridLlmRouterTests
{
    [Fact]
    public void HybridLlmRouter_Create_ShouldHaveDefaultProviders()
    {
        var router = new HybridLlmRouter();

        router.FallbackChain.Should().NotBeEmpty();
        router.RegisteredProviders.Should().BeEmpty();
        router.ProviderCount.Should().Be(0);
    }

    [Fact]
    public void HybridLlmRouter_Route_ShouldReturnValidProvider()
    {
        var router = new HybridLlmRouter();

        var providerName = router.FallbackChain[0];

        providerName.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void HybridLlmRouter_CostTracker_ShouldStartAtZero()
    {
        var router = new HybridLlmRouter();

        router.CurrentBudget.CurrentDailyCostUsd.Should().Be(0.0m);
    }

    [Fact]
    public void TryParseLightingParams_ParsesJsonSnippet()
    {
        const string llmText = """
            {
              "directionalDirection": [0.3, -0.8, 0.5],
              "color": "#FFEECC",
              "intensity": 2.5,
              "fogDensity": 0.02,
              "enableClouds": true
            }
            """;

        var parsed = StructuredOutputParser.TryParseLightingParams(llmText, out var parameters);

        parsed.Should().BeTrue();
        parameters.DirectionalDirection.Should().Be((0.3f, -0.8f, 0.5f));
        parameters.Color.Should().Be("#FFEECC");
        parameters.Intensity.Should().Be(2.5f);
        parameters.FogDensity.Should().Be(0.02f);
        parameters.EnableClouds.Should().BeTrue();
    }

    [Fact]
    public void TryParseLightingParams_ParsesKeyValueSnippet()
    {
        const string llmText = """
            sunDirection: 0.1, -1.0, 0.2
            color: #AABBCC
            intensity: 1.75
            fogDensity: 0.015
            enableClouds: yes
            """;

        var parsed = StructuredOutputParser.TryParseLightingParams(llmText, out var parameters);

        parsed.Should().BeTrue();
        parameters.DirectionalDirection.Should().Be((0.1f, -1.0f, 0.2f));
        parameters.Color.Should().Be("#AABBCC");
        parameters.Intensity.Should().Be(1.75f);
        parameters.FogDensity.Should().Be(0.015f);
        parameters.EnableClouds.Should().BeTrue();
    }
}
