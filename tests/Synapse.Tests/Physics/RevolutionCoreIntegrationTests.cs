using FluentAssertions;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Physics;

/// <summary>
/// Integration tests for the unified law catalog and applicator routing fixes.
/// </summary>
public sealed class RevolutionCoreIntegrationTests
{
    [Fact]
    public void LoadBuiltIn_ExposesFullReferenceCatalog()
    {
        var library = LawLibrary.LoadBuiltIn();
        library.AllEntries.Count.Should().BeGreaterThanOrEqualTo(100);
        library.GetLaw("heat_equation").Should().NotBeNull();
        library.GetLaw("thermo.heat-eq").Should().NotBeNull();
    }

    [Fact]
    public void LawApplicatorMapper_MapsThermodynamicsToHeat()
    {
        LawApplicatorMapper.Resolve("Thermodynamics", LawCategory.Thermodynamics).Should().Be("heat");
        LawApplicatorMapper.Resolve("ThermalDynamics").Should().Be("heat");
        LawApplicatorMapper.Resolve("Electromagnetism", LawCategory.Electromagnetism).Should().Be("electromagnetic");
    }

    [Fact]
    public void LawExpressionNormalizer_ExtractsRhsAndLaplacian()
    {
        var normalized = LawExpressionNormalizer.NormalizeForCompilation("dT/dt = alpha * del^2(T)");
        normalized.Should().Be("alpha * laplacian(T)");
    }

    [Fact]
    public void LivingLawCompiler_CompilesCatalogHeatLaw()
    {
        var field = CreateField();
        var compiler = new LivingLawCompiler();
        compiler.CatalogLawCount.Should().BeGreaterThanOrEqualTo(100);

        var result = compiler.CompileFromLibrary("thermo.heat-eq");
        result.Success.Should().BeTrue(result.Message);

        var act = () => compiler.ApplyLaw("thermo.heat-eq", field, 0.01f);
        act.Should().NotThrow();
    }

    [Fact]
    public void LivingLawCompiler_UsesHeatApplicatorForThermodynamics()
    {
        var compiler = new LivingLawCompiler();
        var result = compiler.CompileFromLibrary("thermo.heat-eq");
        result.Success.Should().BeTrue();

        var field = CreateField();
        var before = field.Temperature[8, 8, 8];
        compiler.ApplyLaw("thermo.heat-eq", field, 0.05f);
        field.Temperature[8, 8, 8].Should().NotBe(before);
    }

    private static PhysicsField CreateField()
    {
        var field = new PhysicsField(16, "revolution-test");
        for (int z = 0; z < 16; z++)
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++)
                {
                    float dx = x - 8f;
                    float dy = y - 8f;
                    float dz = z - 8f;
                    float r2 = dx * dx + dy * dy + dz * dz;
                    field.Temperature[x, y, z] = 300f + 80f * MathF.Exp(-r2 / 8f);
                }
        return field;
    }
}
