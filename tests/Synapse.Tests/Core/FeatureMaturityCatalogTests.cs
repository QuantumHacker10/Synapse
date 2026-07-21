using System;
using System.Linq;
using System.Reflection;
using FluentAssertions;
using Synapse.Core.Maturity;
using Synapse.Network;
using Synapse.Plugins;
using Synapse.VR;
using Synapse.Web;
using Xunit;

namespace Synapse.Tests.Core;

public sealed class FeatureMaturityCatalogTests
{
    [Fact]
    public void Catalog_HasNoSupportedSurfacesYet()
    {
        FeatureMaturityCatalog.OfTier(MaturityTier.Supported).Should().BeEmpty(
            "v2.2 is early-access; Supported is reserved for validated Windows+GPU surfaces");
    }

    [Fact]
    public void Catalog_MarksVrWanAndWebAsEarlyAccess()
    {
        FeatureMaturityCatalog.Find("VR.OpenXR")!.Tier.Should().Be(MaturityTier.EarlyAccess);
        FeatureMaturityCatalog.Find("Network.WAN")!.Tier.Should().Be(MaturityTier.EarlyAccess);
        FeatureMaturityCatalog.Find("Web.Editor")!.Tier.Should().Be(MaturityTier.EarlyAccess);
        FeatureMaturityCatalog.OfTier(MaturityTier.Experimental).Should().HaveCountGreaterThanOrEqualTo(1);
    }

    [Fact]
    public void FirstClassSurfaces_AreNotMarkedExperimental()
    {
        typeof(OpenXrVulkanSwapchain).GetCustomAttribute<SynapseExperimentalAttribute>().Should().BeNull();
        typeof(OpenXrVulkanSession).GetCustomAttribute<SynapseExperimentalAttribute>().Should().BeNull();
        typeof(NatTraversalCoordinator).GetCustomAttribute<SynapseExperimentalAttribute>().Should().BeNull();
        typeof(WanSimulationPeerHub).GetCustomAttribute<SynapseExperimentalAttribute>().Should().BeNull();
        typeof(WebEditorBuilder).GetCustomAttribute<SynapseExperimentalAttribute>().Should().BeNull();
        typeof(WasmStudioPublisher).GetCustomAttribute<SynapseExperimentalAttribute>().Should().BeNull();
    }

    [Fact]
    public void RemainingExperimentalTypes_CarrySynapseExperimentalAttribute()
    {
        AssertExperimental(typeof(MultiPeerSimulationHub), "Network.P2P");
        AssertExperimental(typeof(PluginHost), "Plugins.CSharp");
    }

    [Fact]
    public void OpenXrSwapchain_DeclaresSyntheticHandles()
    {
        using var swap = new OpenXrVulkanSwapchain(2, 64, 64);
        swap.UsesSyntheticImageHandles.Should().BeTrue();
    }

    [Fact]
    public void NatCoordinator_DeclaresLoopbackOnlyUntilStun()
    {
        using var logger = new Synapse.Infrastructure.Logging.SynapseLogger(null, Synapse.Infrastructure.Logging.LogLevel.Error, consoleEnabled: false);
        using var nat = new NatTraversalCoordinator(logger, "maturity-test");
        nat.IsLoopbackOnly.Should().BeTrue();
    }

    private static void AssertExperimental(Type type, string featureId)
    {
        var attr = type.GetCustomAttribute<SynapseExperimentalAttribute>();
        attr.Should().NotBeNull($"{type.Name} must be marked experimental");
        attr!.FeatureId.Should().Be(featureId);
        attr.Reason.Should().NotBeNullOrWhiteSpace();
    }
}
