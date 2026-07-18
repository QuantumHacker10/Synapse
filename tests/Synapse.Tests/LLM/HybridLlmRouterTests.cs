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
}
