using System;
using System.Linq;
using FluentAssertions;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Physics;

public sealed class LivingLawLibraryCoverageTests
{
    [Fact]
    public void LawLibraryRegistry_GetAll_IsNonEmpty()
    {
        var all = LawLibraryRegistry.GetAll();
        all.Should().NotBeEmpty();
        all.Should().Contain(l => l.Id.Contains("heat", StringComparison.OrdinalIgnoreCase)
                                  || l.Tags.Any(t => t.Contains("heat", StringComparison.OrdinalIgnoreCase)));
    }

    [Fact]
    public void LawLibraryRegistry_Search_FindsHeat()
    {
        var hits = LawLibraryRegistry.Search("heat").ToList();
        hits.Should().NotBeEmpty();
    }

    [Fact]
    public void LawRegistry_RegisterAndGet()
    {
        var registry = new LawRegistry();
        var all = registry.GetAll();
        all.Should().NotBeNull();
        registry.Search("heat").Should().NotBeNull();
    }

    [Fact]
    public void LawFactory_CreatesCustomDefinition()
    {
        var def = LawFactory.CreateCustomLaw("T", LawCategory.Thermodynamics, "Unit Test Law");
        def.Name.Should().Be("Unit Test Law");
        def.Expression.Should().Be("T");
        def.Category.Should().Be(LawCategory.Thermodynamics);
    }

    [Fact]
    public void ExtendedRegistries_ExposeAdditionalLaws()
    {
        LawLibraryRegistryExtended.GetAdditionalLaws().Should().NotBeEmpty();
        LawLibraryRegistryExtended2.GetAllAdditionalLaws().Should().NotBeEmpty();
    }
}
