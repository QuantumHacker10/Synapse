using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Core.Maturity;
using Synapse.Infrastructure;

namespace Synapse.Web;

/// <summary>
/// Publishes the Blazor WASM Studio and writes a scene bundle for static hosting.
/// </summary>
[SynapseExperimental("Web.Editor", "Blazor WASM Studio publish + static scene bundle.")]
public static class WasmStudioPublisher
{
    private static readonly JsonSerializerOptions SceneJsonOptions = new() { WriteIndented = true };

    public sealed class PublishResult
    {
        public required string OutputDirectory { get; init; }
        public required string IndexPath { get; init; }
        public required string SceneJsonPath { get; init; }
        public bool UsedDotnetPublish { get; init; }
        public string? PublishLog { get; init; }
    }

    /// <summary>
    /// Writes a WASM Studio host page + scene JSON. When <paramref name="projectDirectory"/>
    /// points at <c>Synapse.Web.Studio</c> and <c>dotnet publish</c> succeeds, the full Blazor
    /// output is copied into <paramref name="outputDirectory"/>; otherwise a self-contained
    /// static fallback (compatible with site/editor) is emitted.
    /// </summary>
    public static async Task<PublishResult> PublishAsync(
        string outputDirectory,
        string sceneName,
        string? activeLawId,
        int entityCount,
        string? sceneJson = null,
        string? webStudioProjectDirectory = null,
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);
        var outDir = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(outDir);

        var scenePath = Path.Combine(outDir, "scene.synapse.json");
        var json = sceneJson ?? JsonSerializer.Serialize(new
        {
            name = sceneName,
            version = "2.2",
            activeLawId,
            entities = Array.Empty<object>(),
            entityCount
        }, SceneJsonOptions);
        await File.WriteAllTextAsync(scenePath, json, Encoding.UTF8, ct).ConfigureAwait(false);

        bool published = false;
        string? log = null;
        var projectDir = webStudioProjectDirectory ?? FindWebStudioProject();
        if (!string.IsNullOrWhiteSpace(projectDir) && File.Exists(Path.Combine(projectDir, "Synapse.Web.Studio.csproj")))
        {
            var publishDir = Path.Combine(Path.GetTempPath(), "synapse-wasm-publish-" + Guid.NewGuid().ToString("N"));
            try
            {
                Directory.CreateDirectory(publishDir);
                var psi = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    ArgumentList =
                    {
                        "publish",
                        Path.Combine(projectDir, "Synapse.Web.Studio.csproj"),
                        "-c", "Release",
                        "-o", publishDir,
                        "/p:WasmFingerprintAssets=false"
                    },
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false
                };
                using var proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to start dotnet publish.");
                var stdout = await proc.StandardOutput.ReadToEndAsync(ct).ConfigureAwait(false);
                var stderr = await proc.StandardError.ReadToEndAsync(ct).ConfigureAwait(false);
                await proc.WaitForExitAsync(ct).ConfigureAwait(false);
                log = stdout + stderr;
                if (proc.ExitCode == 0)
                {
                    CopyDirectory(publishDir, outDir);
                    // Blazor may place index.html under wwwroot — normalize to output root.
                    var nestedIndex = Path.Combine(outDir, "wwwroot", "index.html");
                    var rootIndex = Path.Combine(outDir, "index.html");
                    if (!File.Exists(rootIndex) && File.Exists(nestedIndex))
                        File.Copy(nestedIndex, rootIndex, overwrite: true);
                    // Ensure scene JSON sits next to the published app.
                    await File.WriteAllTextAsync(scenePath, json, Encoding.UTF8, ct).ConfigureAwait(false);
                    published = File.Exists(rootIndex) || File.Exists(nestedIndex);
                }
            }
            catch (Exception ex)
            {
                log = (log ?? "") + "\n" + ex.Message;
            }
            finally
            {
                try
                { if (Directory.Exists(publishDir)) Directory.Delete(publishDir, true); }
                catch { /* ignore */ }
            }
        }

        if (!published)
        {
            var bundle = WebEditorBuilder.FromScene(sceneName, "demo.gltf", activeLawId, entityCount);
            await WebEditorBuilder.WriteSiteAsync(outDir, bundle, ct).ConfigureAwait(false);
            await File.WriteAllTextAsync(
                Path.Combine(outDir, "studio-fallback.txt"),
                $"WASM publish unavailable — static WebGPU editor written. {SynapseProduct.Name}",
                ct).ConfigureAwait(false);
        }

        // Guarantee index.html for consumers/tests even if publish layout was unexpected.
        var indexPath = Path.Combine(outDir, "index.html");
        if (!File.Exists(indexPath))
        {
            var nested = Path.Combine(outDir, "wwwroot", "index.html");
            if (File.Exists(nested))
                File.Copy(nested, indexPath, overwrite: true);
            else
            {
                var bundle = WebEditorBuilder.FromScene(sceneName, "demo.gltf", activeLawId, entityCount);
                await File.WriteAllTextAsync(indexPath, WebEditorBuilder.ToHtml(bundle), ct).ConfigureAwait(false);
            }
        }

        return new PublishResult
        {
            OutputDirectory = outDir,
            IndexPath = indexPath,
            SceneJsonPath = scenePath,
            UsedDotnetPublish = published,
            PublishLog = log
        };
    }

    private static string? FindWebStudioProject()
    {
        // Walk up from cwd / base directory looking for src/Synapse.Web.Studio
        foreach (var start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            var dir = new DirectoryInfo(start);
            for (int i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
            {
                var candidate = Path.Combine(dir.FullName, "src", "Synapse.Web.Studio");
                if (File.Exists(Path.Combine(candidate, "Synapse.Web.Studio.csproj")))
                    return candidate;
            }
        }

        return null;
    }

    private static void CopyDirectory(string source, string target)
    {
        foreach (var dir in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
        {
            var rel = Path.GetRelativePath(source, dir);
            Directory.CreateDirectory(Path.Combine(target, rel));
        }

        foreach (var file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
        {
            var rel = Path.GetRelativePath(source, file);
            var dest = Path.Combine(target, rel);
            Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
            File.Copy(file, dest, overwrite: true);
        }
    }
}
