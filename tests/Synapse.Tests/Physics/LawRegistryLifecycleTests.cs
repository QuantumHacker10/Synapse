using System;
using System.Linq;
using FluentAssertions;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Physics;

public sealed class LawRegistryLifecycleTests
{
    [Fact]
    public void LoadDefaults_RegisterUpdateUnregister_ExportImport()
    {
        var registry = new LawRegistry();
        registry.LoadDefaults();
        registry.Count.Should().BeGreaterThan(0);

        var custom = LawFactory.CreateCustomLaw("x+1", LawCategory.Mechanics, "Custom Mech");
        var registered = registry.Register(custom);
        registry.Contains(registered.Id).Should().BeTrue();
        registry.Get(registered.Id).Should().NotBeNull();

        var updated = registry.Update(registered with { Description = "updated" });
        updated.Description.Should().Be("updated");
        registry.GetVersion(registered.Id).Should().BeGreaterThan(0);

        registry.ByCategory(LawCategory.Mechanics).Should().NotBeEmpty();
        registry.Search("Custom").Should().NotBeEmpty();

        var json = registry.ExportJson();
        json.Should().Contain(registered.Id);

        registry.Unregister(registered.Id).Should().BeTrue();
        registry.Contains(registered.Id).Should().BeFalse();

        var other = new LawRegistry();
        other.ImportJson(json);
        other.Count.Should().BeGreaterThan(0);
    }

    [Fact]
    public void RegisterDuplicate_Throws()
    {
        var registry = new LawRegistry();
        var law = LawFactory.CreateCustomLaw("y", LawCategory.Thermodynamics, "Dup");
        registry.Register(law);
        var act = () => registry.Register(law);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void MergeFrom_AddsLaws()
    {
        var a = new LawRegistry();
        a.Register(LawFactory.CreateCustomLaw("a", LawCategory.Optics, "A"));
        var b = new LawRegistry();
        b.Register(LawFactory.CreateCustomLaw("b", LawCategory.Optics, "B"));
        var merged = a.MergeFrom(b);
        merged.Should().BeGreaterThan(0);
        a.Search("B").Should().NotBeEmpty();
    }
}
