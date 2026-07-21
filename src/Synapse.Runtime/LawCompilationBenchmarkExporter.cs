using System.Text.Json;
using System.Text.Json.Serialization;
using Synapse.Physics;

namespace Synapse.Runtime;

/// <summary>Serializes <see cref="LawCompilationBenchmarkReport"/> to JSON.</summary>
public static class LawCompilationBenchmarkExporter
{
    public static string ToJson(LawCompilationBenchmarkReport report) =>
        JsonSerializer.Serialize(report, LawCompilationBenchmarkJsonContext.Default.LawCompilationBenchmarkReport);

    public static async Task SaveAsync(LawCompilationBenchmarkReport report, string path, CancellationToken ct = default)
    {
        var full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        await File.WriteAllTextAsync(full, ToJson(report), ct).ConfigureAwait(false);
    }
}

[JsonSerializable(typeof(LawCompilationBenchmarkReport))]
[JsonSerializable(typeof(LawCompilationProbe))]
[JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class LawCompilationBenchmarkJsonContext : JsonSerializerContext;
