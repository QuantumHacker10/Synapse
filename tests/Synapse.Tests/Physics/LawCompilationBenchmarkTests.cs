using FluentAssertions;
using Synapse.Physics;
using Xunit;

namespace Synapse.Tests.Physics;

public sealed class LawCompilationBenchmarkTests
{
    [Fact]
    public void Run_CatalogHasAtLeast100Laws()
    {
        var report = LawCompilationBenchmarkRunner.Run();
        report.TotalLaws.Should().BeGreaterThanOrEqualTo(100);
    }

    [Fact]
    public void Run_MajorityCompileDirectOrWithFallback()
    {
        var report = LawCompilationBenchmarkRunner.Run();
        (report.CompiledDirect + report.CompiledWithFallback).Should().BeGreaterThan(report.TotalLaws / 2);
        report.Failed.Should().BeLessThan(report.TotalLaws);
    }

    [Fact]
    public void ProbeCompile_HeatCatalogLaw_CompilesDirect()
    {
        var compiler = new LivingLawCompiler();
        var probe = compiler.ProbeCompileFromLibrary("thermo.heat-eq");
        probe.Success.Should().BeTrue();
        probe.UsedFallback.Should().BeFalse();
    }
}
