namespace Synapse.Web;

/// <summary>WebGPU scene preview descriptor for WASM editor (v2 foundation).</summary>
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
        new()
        {
            SceneName = sceneName,
            GlbUrl = glbUrl,
            ActiveLawId = lawId,
            EntityCount = entityCount
        };

    public static string ToHtml(WebPreviewDescriptor descriptor)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        return $"""
            <!DOCTYPE html>
            <html lang="fr">
            <head><meta charset="utf-8"><title>{descriptor.SceneName} — Synapse Preview</title></head>
            <body>
            <h1>{descriptor.SceneName}</h1>
            <p>Loi active : {descriptor.ActiveLawId ?? "—"} · {descriptor.EntityCount} entité(s)</p>
            <model-viewer src="{descriptor.GlbUrl}" auto-rotate camera-controls style="width:100%;height:480px"></model-viewer>
            <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
            </body>
            </html>
            """;
    }
}
