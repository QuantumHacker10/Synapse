namespace Synapse.Physics;

/// <summary>Result of probing whether a catalog law compiles directly or via category fallback.</summary>
public readonly struct LawCompilationProbe
{
    public LawCompilationProbe(
        string lawId,
        string category,
        bool success,
        bool usedFallback,
        string? error = null)
    {
        LawId = lawId;
        Category = category;
        Success = success;
        UsedFallback = usedFallback;
        Error = error;
    }

    public string LawId { get; }
    public string Category { get; }
    public bool Success { get; }
    public bool UsedFallback { get; }
    public string? Error { get; }
}

/// <summary>Aggregate report for law compilation benchmarking.</summary>
public sealed class LawCompilationBenchmarkReport
{
    public int TotalLaws { get; set; }
    public int CompiledDirect { get; set; }
    public int CompiledWithFallback { get; set; }
    public int Failed { get; set; }
    public string SynapseVersion { get; set; } = "";
    public DateTimeOffset CompletedAt { get; set; } = DateTimeOffset.UtcNow;
    public List<LawCompilationProbe> Entries { get; set; } = new();

    public double DirectCompileRate =>
        TotalLaws == 0 ? 0 : (double)CompiledDirect / TotalLaws;
}
