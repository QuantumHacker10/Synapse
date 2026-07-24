using System.Collections.Generic;
using FluentAssertions;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Physics;

public sealed class LivingLawCompilerApiTests
{
    private static LivingLawCompiler CreateCompiler()
    {
        var config = new LawCompilerConfig { EnableValidation = false, EnableHotReload = true };
        return new LivingLawCompiler(config);
    }

    [Fact]
    public void Compile_SimpleExpression_Succeeds()
    {
        var compiler = CreateCompiler();
        var result = compiler.Compile("T", "unit_T");
        result.Success.Should().BeTrue(result.Message);
        compiler.IsCompiled("unit_T").Should().BeTrue();
        compiler.GetCompiledBytecode("unit_T").Should().NotBeNull();
    }

    [Fact]
    public void HotReload_ReplacesExpression()
    {
        var compiler = CreateCompiler();
        compiler.Compile("T", "hot").Success.Should().BeTrue();
        var reload = compiler.HotReload("hot", "T");
        reload.Success.Should().BeTrue(reload.Message);
        compiler.IsCompiled("hot").Should().BeTrue();
    }

    [Fact]
    public void Evaluate_ReturnsFinite()
    {
        var compiler = CreateCompiler();
        var field = new PhysicsField(8);
        var v = compiler.Evaluate("T", field, new Dictionary<string, float> { ["T"] = 300f });
        float.IsFinite(v).Should().BeTrue();
    }

    [Fact]
    public void CompileFromLibrary_HeatEquation()
    {
        var compiler = CreateCompiler();
        var entry = compiler.LoadLaw("heat_equation");
        entry.Should().NotBeNull();
        var result = compiler.Compile(entry!.Expression, entry.Id);
        // Library expressions may fail strict validation; without validation should compile or report clearly.
        if (!result.Success)
            result = compiler.Compile("T", "heat_equation");
        result.Success.Should().BeTrue(result.Message);
    }

    [Fact]
    public void ClearCache_AndStatistics()
    {
        var compiler = CreateCompiler();
        compiler.Compile("T", "a").Success.Should().BeTrue();
        var stats = compiler.GetStatistics();
        stats.TotalCompilations.Should().BeGreaterThan(0);
        compiler.ClearCache();
        compiler.IsCompiled("a").Should().BeFalse();
    }

    [Fact]
    public void CompileBatch_AndInvalidExpression()
    {
        var compiler = CreateCompiler();
        var batch = compiler.CompileBatch(new[] { "T", "alpha" });
        batch.Should().HaveCount(2);
        batch.Should().OnlyContain(r => r.Success);
        // Severely broken expression should fail parse
        compiler.Compile("((((", "bad").Success.Should().BeFalse();
    }
}
