using System;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure;

namespace Synapse.Web;

/// <summary>
/// WebGPU / glTF editor preview bundle + WASM Studio publish path.
/// Bridged by <see cref="Synapse.Runtime.EngineHost.ExportWebStudioAsync"/>.
/// </summary>
public sealed class WebEditorBundle
{
    public required string SceneName { get; init; }
    public required string GlbUrl { get; init; }
    public string GltfUrl { get; init; } = "demo.gltf";
    public string? ActiveLawId { get; init; }
    public int EntityCount { get; init; }
    public string EditorVersion { get; init; } = "2.2.0";
}

/// <summary>Builder for the web glTF preview site and WASM export fallback.</summary>
public static class WebEditorBuilder
{
    public static WebEditorBundle FromScene(string sceneName, string glbUrl, string? lawId, int entityCount, string? gltfUrl = null)
    {
        ArgumentNullException.ThrowIfNull(glbUrl);
        return new()
        {
            SceneName = sceneName,
            GlbUrl = glbUrl,
            GltfUrl = gltfUrl ?? glbUrl.Replace(".glb", ".gltf"),
            ActiveLawId = lawId,
            EntityCount = entityCount
        };
    }

    public static string ToHtml(WebEditorBundle bundle)
    {
        ArgumentNullException.ThrowIfNull(bundle);
        var meta = JsonSerializer.Serialize(new
        {
            bundle.SceneName,
            bundle.GlbUrl,
            bundle.GltfUrl,
            bundle.ActiveLawId,
            bundle.EntityCount,
            bundle.EditorVersion,
            generator = SynapseProduct.Name
        });

        return $$"""
            <!DOCTYPE html>
            <html lang="fr">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1" />
              <title>{{bundle.SceneName}} — Synapse Web Editor</title>
              <link rel="stylesheet" href="editor.css" />
            </head>
            <body>
              <header class="toolbar">
                <strong>Synapse Web Editor v{{bundle.EditorVersion}}</strong>
                <span id="law-label">Loi : {{bundle.ActiveLawId ?? "—"}}</span>
                <span id="entity-label">{{bundle.EntityCount}} entité(s)</span>
              </header>
              <main class="layout">
                <aside id="hierarchy" class="panel"></aside>
                <canvas id="viewport" width="960" height="540"></canvas>
                <aside id="inspector" class="panel"></aside>
              </main>
              <script id="synapse-scene" type="application/json">{{meta}}</script>
              <script src="editor.js"></script>
            </body>
            </html>
            """;
    }

    public static async Task WriteSiteAsync(string outputDirectory, WebEditorBundle bundle, CancellationToken ct = default)
    {
        var dir = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(dir);
        await File.WriteAllTextAsync(Path.Combine(dir, "index.html"), ToHtml(bundle), ct);
        await File.WriteAllTextAsync(Path.Combine(dir, "editor.css"), EditorCss, ct);
        await File.WriteAllTextAsync(Path.Combine(dir, "editor.js"), EditorJs, ct);
    }

    private const string EditorCss = """
        body { margin: 0; font-family: Inter, system-ui, sans-serif; background: #0e1218; color: #c9d1d9; }
        .toolbar { display: flex; gap: 16px; padding: 10px 16px; background: #161b22; border-bottom: 1px solid #1e2633; }
        .layout { display: grid; grid-template-columns: 240px 1fr 260px; height: calc(100vh - 44px); }
        .panel { background: #12171f; border-right: 1px solid #1e2633; padding: 12px; overflow: auto; }
        #inspector { border-right: none; border-left: 1px solid #1e2633; }
        #viewport { width: 100%; height: 100%; background: #05070a; display: block; }
        """;

    private const string EditorJs = """
        (function () {
          const meta = JSON.parse(document.getElementById('synapse-scene').textContent);
          const canvas = document.getElementById('viewport');
          const hierarchy = document.getElementById('hierarchy');
          const inspector = document.getElementById('inspector');
          hierarchy.innerHTML = '<h3>Scène</h3><p>' + meta.sceneName + '</p>';
          inspector.innerHTML = '<h3>Loi active</h3><p>' + (meta.activeLawId || '—') + '</p>';

          async function init() {
            if (!navigator.gpu) {
              canvas.insertAdjacentHTML('afterend', '<p style="color:#f85149">WebGPU non disponible — utilisez Chrome/Edge récent.</p>');
              return;
            }
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return;
            const device = await adapter.requestDevice();
            const context = canvas.getContext('webgpu');
            const format = navigator.gpu.getPreferredCanvasFormat();
            context.configure({ device, format, alphaMode: 'opaque' });
            const encoder = device.createCommandEncoder();
            const view = context.getCurrentTexture().createView();
            const pass = encoder.beginRenderPass({
              colorAttachments: [{ view, clearValue: { r: 0.04, g: 0.06, b: 0.1, a: 1 }, loadOp: 'clear', storeOp: 'store' }]
            });
            pass.end();
            device.queue.submit([encoder.finish()]);
          }
          init();
        })();
        """;
}

/// <summary>Legacy preview helper kept for compatibility.</summary>
public sealed class WebPreviewDescriptor
{
    public required string SceneName { get; init; }
    public required string GlbUrl { get; init; }
    public string? ActiveLawId { get; init; }
    public int EntityCount { get; init; }
}

public static class WebPreviewBuilder
{
    public static WebPreviewDescriptor FromScene(string sceneName, string glbUrl, string? lawId, int entityCount) =>
        new() { SceneName = sceneName, GlbUrl = glbUrl, ActiveLawId = lawId, EntityCount = entityCount };

    public static string ToHtml(WebPreviewDescriptor descriptor)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        var bundle = WebEditorBuilder.FromScene(descriptor.SceneName, descriptor.GlbUrl, descriptor.ActiveLawId, descriptor.EntityCount);
        return WebEditorBuilder.ToHtml(bundle);
    }
}
