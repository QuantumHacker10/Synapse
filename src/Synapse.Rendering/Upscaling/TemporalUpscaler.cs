// =============================================================================
// Temporal upscaling — FSR-style spatial + DLSS/MetalFX-compatible interface
// =============================================================================

using System;
using System.Numerics;

namespace GDNN.Rendering.Upscaling;

/// <summary>Vendor-neutral temporal upscaler contract for the FrameGraph.</summary>
public interface ITemporalUpscaler
{
    string Name { get; }
    bool IsAvailable { get; }
    void Configure(int renderWidth, int renderHeight, int displayWidth, int displayHeight);
    /// <summary>Upscales HDR/LDR color from render res to display res.</summary>
    void Upscale(ReadOnlySpan<Vector3> sourceRgb, Span<Vector3> destRgb, ReadOnlySpan<Vector2> velocity);
}

public enum UpscalerBackend : byte
{
    Auto = 0,
    FsrSpatial = 1,
    DlssCompatible = 2,
    MetalFxCompatible = 3,
    NativeCopy = 4
}

/// <summary>
/// AMD FSR-style EASU spatial upscaler (Lanczos-ish) — always available CPU path,
/// SPIR-V compute can replace later. Industrial cinematic baseline.
/// </summary>
public sealed class FsrSpatialUpscaler : ITemporalUpscaler
{
    private int _srcW, _srcH, _dstW, _dstH;

    public string Name => "FSR-Spatial";
    public bool IsAvailable => true;

    public void Configure(int renderWidth, int renderHeight, int displayWidth, int displayHeight)
    {
        _srcW = Math.Max(1, renderWidth);
        _srcH = Math.Max(1, renderHeight);
        _dstW = Math.Max(1, displayWidth);
        _dstH = Math.Max(1, displayHeight);
    }

    public void Upscale(ReadOnlySpan<Vector3> sourceRgb, Span<Vector3> destRgb, ReadOnlySpan<Vector2> velocity)
    {
        _ = velocity;
        if (sourceRgb.Length < _srcW * _srcH || destRgb.Length < _dstW * _dstH)
            return;

        if (_srcW == _dstW && _srcH == _dstH)
        {
            sourceRgb.Slice(0, _dstW * _dstH).CopyTo(destRgb);
            return;
        }

        for (int y = 0; y < _dstH; y++)
        {
            float v = (y + 0.5f) * _srcH / _dstH - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(v), 0, _srcH - 1);
            int y1 = Math.Min(y0 + 1, _srcH - 1);
            float fy = v - y0;
            for (int x = 0; x < _dstW; x++)
            {
                float u = (x + 0.5f) * _srcW / _dstW - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(u), 0, _srcW - 1);
                int x1 = Math.Min(x0 + 1, _srcW - 1);
                float fx = u - x0;

                // EASU-inspired: bicubic-ish sharpen via 4-tap + high-pass
                Vector3 c00 = sourceRgb[y0 * _srcW + x0];
                Vector3 c10 = sourceRgb[y0 * _srcW + x1];
                Vector3 c01 = sourceRgb[y1 * _srcW + x0];
                Vector3 c11 = sourceRgb[y1 * _srcW + x1];
                Vector3 bilinear = Vector3.Lerp(
                    Vector3.Lerp(c00, c10, fx),
                    Vector3.Lerp(c01, c11, fx),
                    fy);
                Vector3 avg = (c00 + c10 + c01 + c11) * 0.25f;
                Vector3 sharpened = bilinear + (bilinear - avg) * 0.35f;
                destRgb[y * _dstW + x] = Vector3.Clamp(sharpened, Vector3.Zero, new Vector3(8f));
            }
        }
    }
}

/// <summary>
/// NVIDIA DLSS-compatible wrapper: uses FSR spatial when NGX is absent,
/// preserves the same API so a native DLSS plugin can replace the backend.
/// </summary>
public sealed class DlssCompatibleUpscaler : ITemporalUpscaler
{
    private readonly FsrSpatialUpscaler _fallback = new();
    public string Name => IsNativeDlss ? "DLSS" : "DLSS-Compatible(FSR)";
    public bool IsAvailable => true;
    public bool IsNativeDlss { get; private set; }

    public void Configure(int renderWidth, int renderHeight, int displayWidth, int displayHeight)
    {
        // Native NGX DLSS would be probed here (Windows + NVIDIA). Absent → FSR path.
        IsNativeDlss = false;
        _fallback.Configure(renderWidth, renderHeight, displayWidth, displayHeight);
    }

    public void Upscale(ReadOnlySpan<Vector3> sourceRgb, Span<Vector3> destRgb, ReadOnlySpan<Vector2> velocity) =>
        _fallback.Upscale(sourceRgb, destRgb, velocity);
}

/// <summary>
/// Apple MetalFX-compatible wrapper (macOS). Falls back to FSR spatial via MoltenVK path.
/// </summary>
public sealed class MetalFxCompatibleUpscaler : ITemporalUpscaler
{
    private readonly FsrSpatialUpscaler _fallback = new();
    public string Name => OperatingSystem.IsMacOS() ? "MetalFX-Compatible(FSR)" : "MetalFX-Unavailable→FSR";
    public bool IsAvailable => true;

    public void Configure(int renderWidth, int renderHeight, int displayWidth, int displayHeight) =>
        _fallback.Configure(renderWidth, renderHeight, displayWidth, displayHeight);

    public void Upscale(ReadOnlySpan<Vector3> sourceRgb, Span<Vector3> destRgb, ReadOnlySpan<Vector2> velocity) =>
        _fallback.Upscale(sourceRgb, destRgb, velocity);
}

/// <summary>Selects the best available upscaler for the host platform.</summary>
public static class UpscalerFactory
{
    public static ITemporalUpscaler Create(UpscalerBackend backend = UpscalerBackend.Auto)
    {
        return backend switch
        {
            UpscalerBackend.FsrSpatial => new FsrSpatialUpscaler(),
            UpscalerBackend.DlssCompatible => new DlssCompatibleUpscaler(),
            UpscalerBackend.MetalFxCompatible => new MetalFxCompatibleUpscaler(),
            UpscalerBackend.NativeCopy => new FsrSpatialUpscaler(),
            _ => OperatingSystem.IsMacOS()
                ? new MetalFxCompatibleUpscaler()
                : OperatingSystem.IsWindows()
                    ? new DlssCompatibleUpscaler()
                    : new FsrSpatialUpscaler()
        };
    }
}
