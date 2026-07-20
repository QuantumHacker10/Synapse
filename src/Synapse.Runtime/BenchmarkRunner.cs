using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;

namespace Synapse.Runtime;

public sealed class BenchmarkSuiteConfig
{
    public string Name { get; set; } = "default";
    public int WarmupFrames { get; set; } = 30;
    public int MeasureFrames { get; set; } = 300;
    public int SimulationSeed { get; set; } = 42;
    public string? ScenePath { get; set; }
    public string ActiveLawId { get; set; } = "heat_equation";
}

public sealed class BenchmarkReport
{
    public string SuiteName { get; set; } = "";
    public int SimulationSeed { get; set; }
    public int WarmupFrames { get; set; }
    public int MeasureFrames { get; set; }
    public double PhysicsMsAvg { get; set; }
    public double SimulationMsAvg { get; set; }
    public double TotalMsAvg { get; set; }
    public double PhysicsMsP95 { get; set; }
    public int EntityCount { get; set; }
    public string ActiveLawId { get; set; } = "";
    public string SynapseVersion { get; set; } = "";
    public DateTimeOffset CompletedAt { get; set; } = DateTimeOffset.UtcNow;
}

/// <summary>Headless benchmark runner for CI and reproducible performance reports.</summary>
public sealed class BenchmarkRunner
{
    private readonly ISynapseLogger _logger;

    public BenchmarkRunner(ISynapseLogger logger) => _logger = logger;

    public static BenchmarkSuiteConfig LoadConfig(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize(json, BenchmarkJsonContext.Default.BenchmarkSuiteConfig)
               ?? new BenchmarkSuiteConfig();
    }

    public async Task<BenchmarkReport> RunAsync(
        BenchmarkSuiteConfig config,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(config);
        SimulationReproducibility.SetSeed(config.SimulationSeed);

        var synapseConfig = new SynapseConfig
        {
            Headless = true,
            ScenePath = config.ScenePath,
            EnableValidation = false
        };

        await using var host = new EngineHost(synapseConfig, _logger);
        host.InitializeModules();

        if (!string.IsNullOrWhiteSpace(config.ScenePath))
            await host.LoadSceneAsync(config.ScenePath, cancellationToken).ConfigureAwait(false);

        if (!string.IsNullOrWhiteSpace(config.ActiveLawId))
            host.ApplyLaw(config.ActiveLawId);

        var orchestrator = new FrameOrchestrator(host, _logger);
        var physicsSamples = new List<double>();
        var simSamples = new List<double>();

        for (int i = 0; i < config.WarmupFrames + config.MeasureFrames; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var stats = await orchestrator.TickAsync(cancellationToken).ConfigureAwait(false);
            if (i >= config.WarmupFrames)
            {
                physicsSamples.Add(stats.PhysicsMs);
                simSamples.Add(stats.SimulationMs);
            }
        }

        var report = new BenchmarkReport
        {
            SuiteName = config.Name,
            SimulationSeed = config.SimulationSeed,
            WarmupFrames = config.WarmupFrames,
            MeasureFrames = config.MeasureFrames,
            PhysicsMsAvg = Average(physicsSamples),
            SimulationMsAvg = Average(simSamples),
            TotalMsAvg = Average(physicsSamples) + Average(simSamples),
            PhysicsMsP95 = Percentile(physicsSamples, 0.95),
            EntityCount = host.EntityCount,
            ActiveLawId = host.ActiveLawId ?? config.ActiveLawId,
            SynapseVersion = Synapse.Infrastructure.SynapseProduct.Version
        };

        _logger.Info("Benchmark", $"{report.SuiteName}: physics={report.PhysicsMsAvg:F2}ms sim={report.SimulationMsAvg:F2}ms");
        return report;
    }

    public async Task SaveReportAsync(BenchmarkReport report, string path, CancellationToken ct = default)
    {
        var full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        await using var stream = File.Create(full);
        await JsonSerializer.SerializeAsync(stream, report, BenchmarkJsonContext.Default.BenchmarkReport, ct);
    }

    private static double Average(IReadOnlyList<double> values)
    {
        if (values.Count == 0)
            return 0;
        double sum = 0;
        foreach (var v in values)
            sum += v;
        return sum / values.Count;
    }

    private static double Percentile(IReadOnlyList<double> values, double p)
    {
        if (values.Count == 0)
            return 0;
        var sorted = values.ToArray();
        Array.Sort(sorted);
        int idx = (int)Math.Clamp(Math.Ceiling(p * sorted.Length) - 1, 0, sorted.Length - 1);
        return sorted[idx];
    }
}

[JsonSerializable(typeof(BenchmarkSuiteConfig))]
[JsonSerializable(typeof(BenchmarkReport))]
[JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class BenchmarkJsonContext : JsonSerializerContext;
