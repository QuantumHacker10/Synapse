namespace Synapse.Physics;

/// <summary>Benchmarks how many catalog laws compile without category fallback.</summary>
public static class LawCompilationBenchmarkRunner
{
    public static LawCompilationBenchmarkReport Run(LivingLawCompiler? compiler = null)
    {
        compiler ??= new LivingLawCompiler();
        var report = new LawCompilationBenchmarkReport
        {
            SynapseVersion = typeof(LivingLawCompiler).Assembly.GetName().Version?.ToString() ?? "unknown"
        };

        foreach (var entry in compiler.Library.AllEntries)
        {
            var probe = compiler.ProbeCompileFromLibrary(entry.Id);
            report.Entries.Add(probe);
            report.TotalLaws++;
            if (!probe.Success)
                report.Failed++;
            else if (probe.UsedFallback)
                report.CompiledWithFallback++;
            else
                report.CompiledDirect++;
        }

        return report;
    }
}
