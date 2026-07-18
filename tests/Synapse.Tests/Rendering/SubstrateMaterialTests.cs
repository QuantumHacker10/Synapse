using FluentAssertions;
using GDNN.Materials.SubstrateOmega;
using Xunit;

namespace Synapse.Tests.Rendering;

public class SubstrateMaterialSystemTests
{
    [Fact]
    public void Material_Create_ShouldHaveDefaultProperties()
    {
        var material = new SubstrateMaterial();
        material.InitializeDefaults();

        material.GetFloat("Metallic").Should().Be(0.0f);
        material.GetFloat("Roughness").Should().Be(0.5f);
        material.GetProperty("BaseColor").Should().NotBeNull();
    }

    [Fact]
    public void Material_BSDF_ShouldReturnValidColor()
    {
        var material = new SubstrateMaterial();
        material.InitializeDefaults();
        material.SetProperty("BaseColor", new Color3(1.0f, 0.0f, 0.0f));
        material.SetProperty("Metallic", 0.0f);
        material.SetProperty("Roughness", 0.5f);

        var evaluator = new BsdfEvaluator();
        var color = evaluator.EvaluateBxDF(Vec3.UnitY, Vec3.UnitX, material, Vec3.UnitY);

        float.IsNaN(color.R).Should().BeFalse();
        float.IsNaN(color.G).Should().BeFalse();
        float.IsNaN(color.B).Should().BeFalse();
    }

    [Fact]
    public void Material_HasSubsurface_ShouldBeFalseByDefault()
    {
        var material = new SubstrateMaterial();
        material.InitializeDefaults();

        material.FeatureFlags.HasFlag(MaterialFeatureFlags.SubsurfaceScattering).Should().BeFalse();
    }

    [Fact]
    public void Material_HasClearcoat_ShouldBeFalseByDefault()
    {
        var material = new SubstrateMaterial();
        material.InitializeDefaults();

        material.GetFloat("ClearCoat").Should().Be(0.0f);
        material.FeatureFlags.HasFlag(MaterialFeatureFlags.ClearCoat).Should().BeFalse();
    }
}
