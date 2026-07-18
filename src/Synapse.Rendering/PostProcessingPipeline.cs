// =============================================================================
// PostProcessingPipeline.cs - G-DNN Engine: Post-Processing Effects
// GDNN.Engine - GDNN.Rendering.PostProcess
// Complete post-processing pipeline: Bloom, DOF, Motion Blur, Tonemapping
// =============================================================================

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace GDNN.Rendering.PostProcess
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    /// <summary>Tonemapping operator types.</summary>
    public enum TonemapOperator
    {
        Linear,
        Reinhard,
        ReinhardExtended,
        ACES,
        ACESApproximation,
        Uncharted2,
        KhronosPBR,
        Filmic,
        Lottes,
        Neutral
    }

    /// <summary>Bloom quality presets.</summary>
    public enum BloomQuality
    {
        Low,
        Medium,
        High,
        Ultra
    }

    /// <summary>DOF quality presets.</summary>
    public enum DOFQuality
    {
        Low,
        Medium,
        High,
        Ultra
    }

    /// <summary>Motion blur quality presets.</summary>
    public enum MotionBlurQuality
    {
        Low,
        Medium,
        High,
        Ultra
    }

    /// <summary>Final-screen artistic look presets.</summary>
    public enum ArtisticStyle
    {
        None,
        Cartoon,
        Grayscale,
        Noir
    }

    // =========================================================================
    // CONFIGURATION RECORDS
    // =========================================================================

    /// <summary>Bloom configuration.</summary>
    public record BloomConfig
    {
        public bool Enabled { get; init; }
        public float Threshold { get; init; } = 0.8f;
        public float Knee { get; init; } = 0.5f;
        public float Intensity { get; init; } = 0.5f;
        public float Radius { get; init; } = 0.8f;
        public BloomQuality Quality { get; init; } = BloomQuality.Medium;
        public int MaxIterations { get; init; } = 6;
        public bool HighPrecision { get; init; }
    }

    /// <summary>Depth of Field configuration.</summary>
    public record DOFConfig
    {
        public bool Enabled { get; init; }
        public float FocusDistance { get; init; } = 10.0f;
        public float Aperture { get; init; } = 1.4f;
        public float FocalLength { get; init; } = 50.0f;
        public float NearBlurStart { get; init; } = 1.0f;
        public float NearBlurEnd { get; init; } = 5.0f;
        public float FarBlurStart { get; init; } = 15.0f;
        public float FarBlurEnd { get; init; } = 50.0f;
        public DOFQuality Quality { get; init; } = DOFQuality.Medium;
        public bool EnableBokeh { get; init; }
        public float MaxBokehSize { get; init; } = 16.0f;
    }

    /// <summary>Motion blur configuration.</summary>
    public record MotionBlurConfig
    {
        public bool Enabled { get; init; }
        public float Intensity { get; init; } = 1.0f;
        public int SampleCount { get; init; } = 8;
        public MotionBlurQuality Quality { get; init; } = MotionBlurQuality.Medium;
        public float MaxVelocityClamp { get; init; } = 100.0f;
        public bool PerObjectBlur { get; init; } = true;
        public float DepthFactor { get; init; } = 0.5f;
    }

    /// <summary>Tonemapping configuration.</summary>
    public record TonemapConfig
    {
        public bool Enabled { get; init; } = true;
        public TonemapOperator Operator { get; init; } = TonemapOperator.ACES;
        public float Exposure { get; init; } = 1.0f;
        public float Gamma { get; init; } = 2.2f;
        public float WhitePoint { get; init; } = 4.0f;
        public bool EnableAutoExposure { get; init; }
        public float AutoExposureSpeed { get; init; } = 1.0f;
        public float MinExposure { get; init; } = 0.1f;
        public float MaxExposure { get; init; } = 10.0f;
    }

    /// <summary>Artistic post-process configuration.</summary>
    public record ArtisticStyleConfig
    {
        public bool Enabled { get; init; }
        public ArtisticStyle Style { get; init; } = ArtisticStyle.None;
        public int CartoonColorLevels { get; init; } = 6;
        public float CartoonEdgeStrength { get; init; } = 0.35f;
        public float NoirContrast { get; init; } = 1.8f;
        public float NoirCrush { get; init; } = 0.15f;
    }

    /// <summary>Global post-processing configuration.</summary>
    public record PostProcessConfig
    {
        public BloomConfig Bloom { get; init; } = new();
        public DOFConfig DOF { get; init; } = new();
        public MotionBlurConfig MotionBlur { get; init; } = new();
        public TonemapConfig Tonemap { get; init; } = new();
        public TAAConfig TAA { get; init; } = new();
        public SSRConfig SSR { get; init; } = new();
        public ArtisticStyleConfig ArtisticStyle { get; init; } = new();
    }

    // =========================================================================
    // FRAME BUFFER (RGBA float for HDR)
    // =========================================================================

    /// <summary>
    /// HDR frame buffer storing RGBA float data for post-processing passes.
    /// </summary>
    public class HDRFrameBuffer : IDisposable
    {
        public int Width { get; }
        public int Height { get; }
        public float[] Data { get; }
        public int Stride => Width * 4;
        public bool IsDisposed { get; private set; }

        public HDRFrameBuffer(int width, int height)
        {
            Width = width;
            Height = height;
            Data = new float[width * height * 4];
        }

        public Span<Vector4> AsVector4Span()
        {
            return MemoryMarshal.Cast<float, Vector4>(Data.AsSpan());
        }

        public Vector4 GetPixel(int x, int y)
        {
            int idx = (y * Width + x) * 4;
            return new Vector4(Data[idx], Data[idx + 1], Data[idx + 2], Data[idx + 3]);
        }

        public void SetPixel(int x, int y, Vector4 color)
        {
            int idx = (y * Width + x) * 4;
            Data[idx] = color.X;
            Data[idx + 1] = color.Y;
            Data[idx + 2] = color.Z;
            Data[idx + 3] = color.W;
        }

        public void Clear()
        {
            Array.Clear(Data);
        }

        public void CopyTo(HDRFrameBuffer target)
        {
            int len = Math.Min(Data.Length, target.Data.Length);
            Buffer.BlockCopy(Data, 0, target.Data, 0, len * sizeof(float));
        }

        public void Dispose()
        {
            IsDisposed = true;
        }
    }

    /// <summary>Depth buffer storing float depth per pixel.</summary>
    public class DepthBuffer : IDisposable
    {
        public int Width { get; }
        public int Height { get; }
        public float[] Data { get; }
        public bool IsDisposed { get; private set; }

        public DepthBuffer(int width, int height)
        {
            Width = width;
            Height = height;
            Data = new float[width * height];
        }

        public float GetDepth(int x, int y) => Data[y * Width + x];
        public void SetDepth(int x, int y, float d) => Data[y * Width + x] = d;
        public void Clear() => Array.Clear(Data);
        public void Dispose() => IsDisposed = true;
    }

    /// <summary>Velocity buffer for motion blur.</summary>
    public class VelocityBuffer : IDisposable
    {
        public int Width { get; }
        public int Height { get; }
        public Vector2[] Data { get; }
        public bool IsDisposed { get; private set; }

        public VelocityBuffer(int width, int height)
        {
            Width = width;
            Height = height;
            Data = new Vector2[width * height];
        }

        public Vector2 GetVelocity(int x, int y) => Data[y * Width + x];
        public void SetVelocity(int x, int y, Vector2 v) => Data[y * Width + x] = v;
        public void Clear() => Array.Clear(Data);
        public void Dispose() => IsDisposed = true;
    }

    // =========================================================================
    // POST-PROCESSING PASSES
    // =========================================================================

    /// <summary>
    /// Performs tonemapping on an HDR frame buffer, converting HDR to LDR.
    /// Supports multiple industry-standard tonemapping operators.
    /// </summary>
    public static class TonemapPass
    {
        public static void Apply(HDRFrameBuffer input, HDRFrameBuffer output, TonemapConfig config)
        {
            float exposure = config.Exposure;
            float gamma = config.Gamma;
            float wp = config.WhitePoint;

            for (int i = 0; i < input.Data.Length; i += 4)
            {
                Vector4 pixel = new(input.Data[i], input.Data[i + 1], input.Data[i + 2], input.Data[i + 3]);
                Vector3 hdr = new(pixel.X * exposure, pixel.Y * exposure, pixel.Z * exposure);

                Vector3 mapped = config.Operator switch
                {
                    TonemapOperator.Linear => hdr,
                    TonemapOperator.Reinhard => Reinhard(hdr),
                    TonemapOperator.ReinhardExtended => ReinhardExtended(hdr, wp),
                    TonemapOperator.ACES => ACESTonemap(hdr),
                    TonemapOperator.ACESApproximation => ACESApproximate(hdr),
                    TonemapOperator.Uncharted2 => Uncharted2(hdr),
                    TonemapOperator.Filmic => Filmic(hdr),
                    TonemapOperator.Neutral => Neutral(hdr),
                    _ => hdr
                };

                mapped = Vector3.Clamp(mapped, Vector3.Zero, Vector3.One);
                mapped = new Vector3(MathF.Pow(mapped.X, 1.0f / gamma), MathF.Pow(mapped.Y, 1.0f / gamma), MathF.Pow(mapped.Z, 1.0f / gamma));

                output.Data[i] = mapped.X;
                output.Data[i + 1] = mapped.Y;
                output.Data[i + 2] = mapped.Z;
                output.Data[i + 3] = pixel.W;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector3 Reinhard(Vector3 c) => c / (c + Vector3.One);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector3 ReinhardExtended(Vector3 c, float white)
        {
            float w2 = white * white;
            return c * (Vector3.One + c / w2) / (c + Vector3.One);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector3 ACESFilm(Vector3 x, float a, float b, float c, float d, float e, float f)
        {
            return Vector3.Clamp((x * (a * x + new Vector3(b))) / (x * (c * x + new Vector3(d)) + new Vector3(e)), Vector3.Zero, Vector3.One);
        }

        private static Vector3 ACESTonemap(Vector3 c)
        {
            c *= 0.6f;
            return ACESFilm(c, 2.51f, 0.03f, 2.43f, 0.59f, 0.14f, 0.0f);
        }

        private static Vector3 ACESApproximate(Vector3 c)
        {
            c *= 0.6f;
            Vector3 a = c * (c * new Vector3(2.51f) + new Vector3(0.03f));
            Vector3 b = c * (c * new Vector3(2.43f) + new Vector3(0.59f)) + new Vector3(0.14f);
            return Vector3.Clamp(a / b, Vector3.Zero, Vector3.One);
        }

        private static Vector3 Uncharted2(Vector3 c)
        {
            Vector3 A = new Vector3(0.15f);
            Vector3 B = new Vector3(0.50f);
            Vector3 C = new Vector3(0.10f);
            Vector3 D = new Vector3(0.20f);
            Vector3 E = new Vector3(0.02f);
            Vector3 F = new Vector3(0.30f);

            Vector3 white = new Vector3(11.2f);
            Vector3 exposureBias = new Vector3(2.0f);
            c *= exposureBias;

            Vector3 curr = (c * (c * A + C * B) + D * E) / (c * (c * A + B) + D * F) - E / F;
            Vector3 whiteScale = new Vector3(1.0f) / (((white * (white * A + C * B) + D * E) / (white * (white * A + B) + D * F) - E / F));
            return curr * whiteScale;
        }

        private static Vector3 Filmic(Vector3 c)
        {
            c = new Vector3(MathF.Max(0, c.X - 0.004f), MathF.Max(0, c.Y - 0.004f), MathF.Max(0, c.Z - 0.004f));
            Vector3 num = c * (6.2f * c + new Vector3(0.5f));
            Vector3 den = c * (6.2f * c + new Vector3(1.7f)) + new Vector3(0.06f);
            return Vector3.Clamp(new Vector3(num.X / den.X, num.Y / den.Y, num.Z / den.Z), Vector3.Zero, Vector3.One);
        }

        private static Vector3 Neutral(Vector3 c)
        {
            float startCompression = 0.8f - 0.04f;
            float desaturation = 0.15f;
            float x = MathF.Min(c.X, MathF.Min(c.Y, c.Z));
            float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
            c -= new Vector3(offset);

            float peak = MathF.Max(c.X, MathF.Max(c.Y, c.Z));
            if (peak > startCompression)
            {
                float d = 1.0f - startCompression;
                float newPeak = 1.0f - d * d / (peak + d - startCompression);
                c *= newPeak / peak;
            }

            float f = MathF.Max(0, MathF.Max(c.X, MathF.Max(c.Y, c.Z)) - 0.3f);
            f = MathF.Min(f, 0.64f);
            f /= 0.64f;
            float g = 1.0f - desaturation * (1.0f - f);
            float luminance = 0.2126f * c.X + 0.7152f * c.Y + 0.0722f * c.Z;
            c = luminance * (Vector3.One - new Vector3(g)) + c * new Vector3(g);
            return c;
        }
    }

    /// <summary>
    /// Bloom post-processing pass implementing dual Kawase blur with thresholding.
    /// Extracts bright areas, downsamples with progressive blur, then composites.
    /// </summary>
    public static class BloomPass
    {
        private static readonly int[] _qualitySamples = { 3, 5, 7, 9 };

        public static void Apply(HDRFrameBuffer input, HDRFrameBuffer output, BloomConfig config)
        {
            if (!config.Enabled) return;

            int w = input.Width;
            int h = input.Height;
            int iterations = Math.Min(config.MaxIterations, w >= 1920 ? 7 : w >= 1280 ? 6 : 5);

            using var thresholdBuffer = new HDRFrameBuffer(w, h);
            ExtractBrightPixels(input, thresholdBuffer, config.Threshold, config.Knee);

            int mipW = w / 2;
            int mipH = h / 2;
            var mipChain = new List<HDRFrameBuffer>();

            for (int i = 0; i < iterations && mipW >= 2 && mipH >= 2; i++)
            {
                var mip = new HDRFrameBuffer(mipW, mipH);
                if (i == 0)
                    Downsample(thresholdBuffer, mip);
                else
                    Downsample(mipChain[i - 1], mip);

                ApplyDualKawaseBlur(mip, mip, config.Radius * (i + 1) / iterations);
                mipChain.Add(mip);
                mipW /= 2;
                mipH /= 2;
            }

            for (int i = mipChain.Count - 2; i >= 0; i--)
            {
                UpsampleAndAdd(mipChain[i + 1], mipChain[i]);
            }

            CompositeBloom(input, mipChain.Count > 0 ? mipChain[0] : thresholdBuffer, output, config.Intensity);

            foreach (var mip in mipChain)
                mip.Dispose();
        }

        private static void ExtractBrightPixels(HDRFrameBuffer src, HDRFrameBuffer dst, float threshold, float knee)
        {
            float softThreshold = threshold + knee;
            for (int y = 0; y < src.Height; y++)
            {
                for (int x = 0; x < src.Width; x++)
                {
                    Vector4 p = src.GetPixel(x, y);
                    float brightness = 0.2126f * p.X + 0.7152f * p.Y + 0.0722f * p.Z;

                    float contribution = 0;
                    if (brightness > threshold)
                    {
                        contribution = brightness > softThreshold ? 1.0f : (brightness - threshold) / knee;
                        contribution = MathF.Max(0, MathF.Min(1, contribution));
                    }

                    dst.SetPixel(x, y, p * contribution);
                }
            }
        }

        private static void Downsample(HDRFrameBuffer src, HDRFrameBuffer dst)
        {
            for (int y = 0; y < dst.Height; y++)
            {
                for (int x = 0; x < dst.Width; x++)
                {
                    int sx = x * 2;
                    int sy = y * 2;
                    Vector4 sum = Vector4.Zero;
                    sum += src.GetPixel(sx, sy);
                    sum += src.GetPixel(Math.Min(sx + 1, src.Width - 1), sy);
                    sum += src.GetPixel(sx, Math.Min(sy + 1, src.Height - 1));
                    sum += src.GetPixel(Math.Min(sx + 1, src.Width - 1), Math.Min(sy + 1, src.Height - 1));
                    dst.SetPixel(x, y, sum * 0.25f);
                }
            }
        }

        private static void UpsampleAndAdd(HDRFrameBuffer src, HDRFrameBuffer dst)
        {
            for (int y = 0; y < dst.Height; y++)
            {
                for (int x = 0; x < dst.Width; x++)
                {
                    int sx = x / 2;
                    int sy = y / 2;
                    Vector4 upsampled = src.GetPixel(sx, sy);
                    Vector4 existing = dst.GetPixel(x, y);
                    dst.SetPixel(x, y, existing + upsampled);
                }
            }
        }

        private static void ApplyDualKawaseBlur(HDRFrameBuffer src, HDRFrameBuffer dst, float offset)
        {
            int w = dst.Width;
            int h = dst.Height;
            int o = Math.Max(1, (int)(offset + 0.5f));

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    Vector4 color = Vector4.Zero;
                    color += src.GetPixel(x, y) * 4.0f;
                    color += src.GetPixel(Math.Clamp(x - o, 0, w - 1), y);
                    color += src.GetPixel(Math.Clamp(x + o, 0, w - 1), y);
                    color += src.GetPixel(x, Math.Clamp(y - o, 0, h - 1));
                    color += src.GetPixel(x, Math.Clamp(y + o, 0, h - 1));
                    dst.SetPixel(x, y, color / 8.0f);
                }
            }
        }

        private static void CompositeBloom(HDRFrameBuffer scene, HDRFrameBuffer bloom, HDRFrameBuffer output, float intensity)
        {
            for (int i = 0; i < scene.Data.Length; i += 4)
            {
                output.Data[i] = scene.Data[i] + bloom.Data[i] * intensity;
                output.Data[i + 1] = scene.Data[i + 1] + bloom.Data[i + 1] * intensity;
                output.Data[i + 2] = scene.Data[i + 2] + bloom.Data[i + 2] * intensity;
                output.Data[i + 3] = scene.Data[i + 3];
            }
        }

        private class BloomData : HDRFrameBuffer { public BloomData(int w, int h) : base(w, h) { } }
    }

    /// <summary>
    /// Depth of Field post-processing pass implementing Circle of Confusion (CoC)
    /// based blur with near/far field separation and optional bokeh.
    /// </summary>
    public static class DOFPass
    {
        public static void Apply(HDRFrameBuffer colorBuffer, DepthBuffer depthBuffer, HDRFrameBuffer output, DOFConfig config, float aspectRatio)
        {
            if (!config.Enabled) return;

            int w = colorBuffer.Width;
            int h = colorBuffer.Height;
            int sampleCount = config.Quality switch
            {
                DOFQuality.Low => 4,
                DOFQuality.Medium => 8,
                DOFQuality.High => 12,
                DOFQuality.Ultra => 16,
                _ => 8
            };

            using var cocBuffer = new HDRFrameBuffer(w, h);
            ComputeCoC(depthBuffer, cocBuffer, config, aspectRatio);

            using var nearField = new HDRFrameBuffer(w, h);
            using var farField = new HDRFrameBuffer(w, h);

            SplitFields(colorBuffer, cocBuffer, nearField, farField);

            ApplyGaussianBlur(nearField, config.MaxBokehSize * 0.5f, sampleCount);
            ApplyGaussianBlur(farField, config.MaxBokehSize, sampleCount);

            if (config.EnableBokeh)
            {
                ApplyBokehDisc(nearField, config.MaxBokehSize * 0.3f, sampleCount);
            }

            RecombineFields(colorBuffer, nearField, farField, cocBuffer, output);
        }

        private static void ComputeCoC(DepthBuffer depth, HDRFrameBuffer cocBuffer, DOFConfig config, float aspectRatio)
        {
            float focalLength = config.FocalLength / 1000.0f;
            float maxCoC = 1.0f;

            for (int y = 0; y < depth.Height; y++)
            {
                for (int x = 0; x < depth.Width; x++)
                {
                    float d = depth.GetDepth(x, y);
                    float focusPlane = config.FocusDistance;
                    float blurFar = MathF.Max(0, (d - focusPlane) / (config.FarBlurEnd - focusPlane));
                    float blurNear = MathF.Max(0, (focusPlane - d) / (focusPlane - config.NearBlurStart));
                    float coc = MathF.Min(MathF.Max(blurFar, blurNear), maxCoC);

                    cocBuffer.SetPixel(x, y, new Vector4(coc, coc, coc, 1.0f));
                }
            }
        }

        private static void SplitFields(HDRFrameBuffer color, HDRFrameBuffer coc, HDRFrameBuffer near, HDRFrameBuffer far)
        {
            for (int y = 0; y < color.Height; y++)
            {
                for (int x = 0; x < color.Width; x++)
                {
                    Vector4 p = color.GetPixel(x, y);
                    float cocValue = coc.GetPixel(x, y).X;

                    if (cocValue > 0.0f)
                        far.SetPixel(x, y, p);
                    else
                        near.SetPixel(x, y, p);
                }
            }
        }

        private static void ApplyGaussianBlur(HDRFrameBuffer buffer, float radius, int samples)
        {
            using var temp = new HDRFrameBuffer(buffer.Width, buffer.Height);
            buffer.CopyTo(temp);

            int r = Math.Max(1, (int)radius);
            float[] kernel = GenerateGaussianKernel(r, r / 3.0f);

            for (int y = 0; y < buffer.Height; y++)
            {
                for (int x = 0; x < buffer.Width; x++)
                {
                    Vector4 sum = Vector4.Zero;
                    float weightSum = 0;

                    for (int i = -r; i <= r; i++)
                    {
                        float weight = kernel[i + r];
                        int sx = Math.Clamp(x + i, 0, buffer.Width - 1);
                        sum += temp.GetPixel(sx, y) * weight;
                        weightSum += weight;
                    }

                    buffer.SetPixel(x, y, sum / weightSum);
                }
            }

            buffer.CopyTo(temp);

            for (int y = 0; y < buffer.Height; y++)
            {
                for (int x = 0; x < buffer.Width; x++)
                {
                    Vector4 sum = Vector4.Zero;
                    float weightSum = 0;

                    for (int i = -r; i <= r; i++)
                    {
                        float weight = kernel[i + r];
                        int sy = Math.Clamp(y + i, 0, buffer.Height - 1);
                        sum += temp.GetPixel(x, sy) * weight;
                        weightSum += weight;
                    }

                    buffer.SetPixel(x, y, sum / weightSum);
                }
            }
        }

        private static void ApplyBokehDisc(HDRFrameBuffer buffer, float maxRadius, int sampleCount)
        {
            int r = Math.Max(1, (int)maxRadius);
            using var temp = new HDRFrameBuffer(buffer.Width, buffer.Height);
            buffer.CopyTo(temp);

            Random rng = new(42);
            for (int y = 0; y < buffer.Height; y++)
            {
                for (int x = 0; x < buffer.Width; x++)
                {
                    Vector4 center = temp.GetPixel(x, y);
                    if (center.X + center.Y + center.Z < 0.01f) continue;

                    Vector4 sum = center;
                    float weightSum = 1.0f;

                    for (int s = 0; s < sampleCount; s++)
                    {
                        float angle = (float)(s * Math.PI * 2.0 / sampleCount);
                        float dist = r * (0.3f + 0.7f * (float)rng.NextDouble());
                        int sx = Math.Clamp(x + (int)(MathF.Cos(angle) * dist), 0, buffer.Width - 1);
                        int sy = Math.Clamp(y + (int)(MathF.Sin(angle) * dist), 0, buffer.Height - 1);
                        sum += temp.GetPixel(sx, sy);
                        weightSum += 1.0f;
                    }

                    buffer.SetPixel(x, y, sum / weightSum);
                }
            }
        }

        private static void RecombineFields(HDRFrameBuffer original, HDRFrameBuffer near, HDRFrameBuffer far, HDRFrameBuffer coc, HDRFrameBuffer output)
        {
            for (int y = 0; y < original.Height; y++)
            {
                for (int x = 0; x < original.Width; x++)
                {
                    Vector4 p = original.GetPixel(x, y);
                    float cocValue = coc.GetPixel(x, y).X;
                    Vector4 nearP = near.GetPixel(x, y);
                    Vector4 farP = far.GetPixel(x, y);

                    Vector4 result;
                    if (cocValue > 0)
                        result = Vector4.Lerp(p, farP, MathF.Min(cocValue, 1.0f));
                    else
                        result = p + nearP * MathF.Min(MathF.Abs(cocValue), 1.0f) * 0.5f;

                    output.SetPixel(x, y, result);
                }
            }
        }

        private static float[] GenerateGaussianKernel(int radius, float sigma)
        {
            float[] kernel = new float[radius * 2 + 1];
            float sum = 0;
            for (int i = -radius; i <= radius; i++)
            {
                float val = MathF.Exp(-(i * i) / (2 * sigma * sigma));
                kernel[i + radius] = val;
                sum += val;
            }
            for (int i = 0; i < kernel.Length; i++) kernel[i] /= sum;
            return kernel;
        }
    }

    /// <summary>
    /// Motion blur post-processing pass using per-pixel velocity buffers.
    /// Implements scattered and gathered motion blur with depth-aware sampling.
    /// </summary>
    public static class MotionBlurPass
    {
        public static void Apply(HDRFrameBuffer colorBuffer, VelocityBuffer velocityBuffer, DepthBuffer depthBuffer, HDRFrameBuffer output, MotionBlurConfig config)
        {
            if (!config.Enabled) return;

            int w = colorBuffer.Width;
            int h = colorBuffer.Height;
            int sampleCount = config.Quality switch
            {
                MotionBlurQuality.Low => 4,
                MotionBlurQuality.Medium => 8,
                MotionBlurQuality.High => 12,
                MotionBlurQuality.Ultra => 16,
                _ => 8
            };

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    Vector2 vel = velocityBuffer.GetVelocity(x, y);
                    float velLength = vel.Length();

                    if (velLength < 0.001f)
                    {
                        output.SetPixel(x, y, colorBuffer.GetPixel(x, y));
                        continue;
                    }

                    Vector2 velDir = vel / velLength;
                    float clampedLength = MathF.Min(velLength * config.Intensity, config.MaxVelocityClamp);

                    float centerDepth = depthBuffer.GetDepth(x, y);
                    float depthFactor = float.Lerp(1.0f, config.DepthFactor, centerDepth);

                    Vector4 sum = colorBuffer.GetPixel(x, y);
                    float weightSum = 1.0f;

                    for (int s = 1; s <= sampleCount; s++)
                    {
                        float t = (float)s / sampleCount;
                        Vector2 offset = velDir * clampedLength * t * depthFactor;
                        int sx = Math.Clamp(x + (int)offset.X, 0, w - 1);
                        int sy = Math.Clamp(y + (int)offset.Y, 0, h - 1);

                        float sampleDepth = depthBuffer.GetDepth(sx, sy);
                        float depthWeight = MathF.Abs(centerDepth - sampleDepth) < 0.1f ? 1.0f : 0.5f;

                        sum += colorBuffer.GetPixel(sx, sy) * depthWeight;
                        weightSum += depthWeight;

                        Vector2 backOffset = -velDir * clampedLength * t * depthFactor;
                        int bx = Math.Clamp(x + (int)backOffset.X, 0, w - 1);
                        int by = Math.Clamp(y + (int)backOffset.Y, 0, h - 1);

                        sum += colorBuffer.GetPixel(bx, by) * depthWeight;
                        weightSum += depthWeight;
                    }

                    output.SetPixel(x, y, sum / weightSum);
                }
            }
        }
    }

    // =========================================================================
    // TAA (TEMPORAL ANTI-ALIASING)
    // =========================================================================

    public enum TAAResolveMode { Standard = 0, Responsive = 1, Ultra = 2 }

    public record TAAConfig
    {
        public bool Enabled { get; init; } = true;
        public TAAResolveMode Mode { get; init; } = TAAResolveMode.Standard;
        public float BlendFactor { get; init; } = 0.9f;
        public float VarianceClipping { get; init; } = 0.75f;
        public int HistoryLength { get; init; } = 8;
        public bool EnableYCoCgColorSpace { get; init; } = true;
        public bool EnableMotionBlurReject { get; init; } = true;
    }

    public static class TAAPass
    {
        private static float[]? _historyBuffer;
        private static int _historyWidth, _historyHeight;
        private static int _frameIndex;

        public static void Apply(HDRFrameBuffer current, HDRFrameBuffer output, VelocityBuffer velocity,
                                  DepthBuffer depth, TAAConfig config, Matrix4x4 prevViewProj, Matrix4x4 viewProj,
                                  int width, int height)
        {
            if (!config.Enabled) return;

            EnsureHistory(width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Vector4 currentColor = current.GetPixel(x, y);
                    Vector2 vel = velocity.GetVelocity(x, y);

                    float prevU = (x + 0.5f) / width - vel.X;
                    float prevV = (y + 0.5f) / height - vel.Y;

                    int hx = Math.Clamp((int)(prevU * _historyWidth), 0, _historyWidth - 1);
                    int hy = Math.Clamp((int)(prevV * _historyHeight), 0, _historyHeight - 1);

                    int hidx = (hy * _historyWidth + hx) * 4;
                    Vector4 historyColor = new Vector4(
                        _historyBuffer[hidx], _historyBuffer[hidx + 1],
                        _historyBuffer[hidx + 2], _historyBuffer[hidx + 3]);

                    Vector3 neighborhoodMin = new Vector3(float.MaxValue);
                    Vector3 neighborhoodMax = new Vector3(float.MinValue);
                    Vector3 neighborhoodSum = Vector3.Zero;
                    int sampleCount = 0;

                    for (int ky = -1; ky <= 1; ky++)
                    {
                        for (int kx = -1; kx <= 1; kx++)
                        {
                            int sx = Math.Clamp(x + kx, 0, width - 1);
                            int sy = Math.Clamp(y + ky, 0, height - 1);
                            Vector4 s = current.GetPixel(sx, sy);
                            Vector3 sv = new Vector3(s.X, s.Y, s.Z);
                            neighborhoodMin = Vector3.Min(neighborhoodMin, sv);
                            neighborhoodMax = Vector3.Max(neighborhoodMax, sv);
                            neighborhoodSum += sv;
                            sampleCount++;
                        }
                    }

                    Vector3 neighborhoodAvg = neighborhoodSum / sampleCount;

                    Vector3 histRGB = new Vector3(historyColor.X, historyColor.Y, historyColor.Z);
                    histRGB = Vector3.Clamp(histRGB, neighborhoodMin - config.VarianceClipping * (neighborhoodMax - neighborhoodMin),
                                                  neighborhoodMax + config.VarianceClipping * (neighborhoodMax - neighborhoodMin));

                    Vector3 currentRGB = new Vector3(currentColor.X, currentColor.Y, currentColor.Z);

                    float blendWeight = config.Mode switch
                    {
                        TAAResolveMode.Responsive => 0.5f,
                        TAAResolveMode.Ultra => 0.85f,
                        _ => config.BlendFactor
                    };

                    bool motionValid = vel.Length() < 0.1f;
                    if (!motionValid && config.EnableMotionBlurReject)
                        blendWeight *= 0.5f;

                    if (_frameIndex < config.HistoryLength)
                        blendWeight = float.Lerp(0.1f, blendWeight, (float)_frameIndex / config.HistoryLength);

                    Vector3 result = Vector3.Lerp(histRGB, currentRGB, 1.0f - blendWeight);

                    int oidx = (y * width + x) * 4;
                    output.Data[oidx] = result.X;
                    output.Data[oidx + 1] = result.Y;
                    output.Data[oidx + 2] = result.Z;
                    output.Data[oidx + 3] = currentColor.W;
                }
            }

            Buffer.BlockCopy(output.Data, 0, _historyBuffer, 0, Math.Min(output.Data.Length, _historyBuffer.Length));
            _frameIndex++;
        }

        private static void EnsureHistory(int w, int h)
        {
            if (_historyBuffer == null || _historyWidth != w || _historyHeight != h)
            {
                _historyWidth = w;
                _historyHeight = h;
                _historyBuffer = new float[w * h * 4];
                _frameIndex = 0;
            }
        }

        public static void ResetHistory() { _historyBuffer = null; _frameIndex = 0; }
    }

    // =========================================================================
    // SSR (SCREEN-SPACE REFLECTIONS)
    // =========================================================================

    public enum SSRQuality { Low = 0, Medium = 1, High = 2, Ultra = 3 }

    public record SSRConfig
    {
        public bool Enabled { get; init; } = true;
        public SSRQuality Quality { get; init; } = SSRQuality.Medium;
        public int MaxSteps { get; init; } = 32;
        public float MaxDistance { get; init; } = 100.0f;
        public float Thickness { get; init; } = 0.5f;
        public float Intensity { get; init; } = 1.0f;
        public float EdgeFade { get; init; } = 0.1f;
        public bool EnableHalfRes { get; init; }
        public bool EnableTemporalFilter { get; init; } = true;
    }

    public static class SSRPass
    {
        private static float[]? _prevFrame;
        private static int _prevWidth, _prevHeight;

        public static void Apply(HDRFrameBuffer colorBuffer, DepthBuffer depthBuffer, HDRFrameBuffer output,
                                  SSRConfig config, Matrix4x4 view, Matrix4x4 proj, int width, int height)
        {
            if (!config.Enabled) return;

            int stepMultiplier = config.Quality switch
            {
                SSRQuality.Low => 2,
                SSRQuality.Medium => 1,
                SSRQuality.High => 1,
                SSRQuality.Ultra => 1,
                _ => 1
            };
            int maxSteps = config.MaxSteps / stepMultiplier;
            int stepSize = stepMultiplier;

            Matrix4x4 invView = Matrix4x4.Identity;
            Matrix4x4 invProj = Matrix4x4.Identity;
            Matrix4x4.Invert(view, out invView);
            Matrix4x4.Invert(proj, out invProj);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Vector4 original = colorBuffer.GetPixel(x, y);
                    float sceneDepth = depthBuffer.GetDepth(x, y);

                    if (sceneDepth >= 1.0f || sceneDepth <= 0)
                    {
                        output.SetPixel(x, y, original);
                        continue;
                    }

                    Vector3 viewPos = ReconstructViewPos(x, y, sceneDepth, width, height, invProj);
                    Vector3 viewNormal = ReconstructViewNormal(x, y, depthBuffer, width, height, invView, proj);

                    float roughness = 0.5f;
                    float reflectivity = 0.5f;

                    Vector3 V = -Vector3.Normalize(viewPos);
                    Vector3 R = Vector3.Reflect(-V, viewNormal);

                    Vector2 hitUV = Vector2.Zero;
                    bool hit = TraceScreenSpaceRay(viewPos, R, depthBuffer, view, proj, width, height,
                        maxSteps, config.MaxDistance, config.Thickness, stepSize, out hitUV);

                    if (hit)
                    {
                        float edgeFade = 1.0f;
                        float edgeX = MathF.Min(hitUV.X, 1.0f - hitUV.X);
                        float edgeY = MathF.Min(hitUV.Y, 1.0f - hitUV.Y);
                        edgeFade = MathF.Min(edgeX, edgeY) / config.EdgeFade;
                        edgeFade = Math.Clamp(edgeFade, 0, 1);

                        int hx = Math.Clamp((int)(hitUV.X * width), 0, width - 1);
                        int hy = Math.Clamp((int)(hitUV.Y * height), 0, height - 1);
                        Vector4 reflectedColor = colorBuffer.GetPixel(hx, hy);

                        if (config.EnableTemporalFilter && _prevFrame != null && _prevWidth == width && _prevHeight == height)
                        {
                            int pidx = (hy * width + hx) * 4;
                            if (pidx + 3 < _prevFrame.Length)
                            {
                                Vector4 prev = new Vector4(_prevFrame[pidx], _prevFrame[pidx + 1], _prevFrame[pidx + 2], _prevFrame[pidx + 3]);
                                reflectedColor = Vector4.Lerp(reflectedColor, prev, 0.3f);
                            }
                        }

                        float fresnel = MathF.Pow(1.0f - MathF.Max(0, Vector3.Dot(V, viewNormal)), 5.0f);
                        float ssrIntensity = reflectivity * config.Intensity * edgeFade * fresnel;

                        output.SetPixel(x, y, new Vector4(
                            original.X + reflectedColor.X * ssrIntensity,
                            original.Y + reflectedColor.Y * ssrIntensity,
                            original.Z + reflectedColor.Z * ssrIntensity,
                            original.W));
                    }
                    else
                    {
                        output.SetPixel(x, y, original);
                    }
                }
            }

            EnsurePrevFrame(width, height);
            Buffer.BlockCopy(colorBuffer.Data, 0, _prevFrame, 0, Math.Min(colorBuffer.Data.Length, _prevFrame.Length));
        }

        private static bool TraceScreenSpaceRay(Vector3 origin, Vector3 direction, DepthBuffer depthBuffer,
            Matrix4x4 view, Matrix4x4 proj, int width, int height,
            int maxSteps, float maxDistance, float thickness, int stepSize, out Vector2 hitUV)
        {
            hitUV = Vector2.Zero;
            Matrix4x4 viewProj = view * proj;

            Vector3 step = direction * (maxDistance / maxSteps);
            Vector3 currentPos = origin;

            float prevDepth = 0;
            for (int i = 0; i < maxSteps; i += stepSize)
            {
                currentPos = origin + direction * (maxDistance * i / maxSteps);
                Vector4 clipPos = Vector4.Transform(new Vector4(currentPos, 1.0f), viewProj);

                if (clipPos.W <= 0) return false;

                float ndcX = clipPos.X / clipPos.W;
                float ndcY = clipPos.Y / clipPos.W;
                float ndcZ = clipPos.Z / clipPos.W;

                int sx = (int)((ndcX * 0.5f + 0.5f) * width);
                int sy = (int)((ndcY * 0.5f + 0.5f) * height);

                if (sx < 0 || sx >= width || sy < 0 || sy >= height) return false;

                float sceneDepth = depthBuffer.GetDepth(sx, sy);
                float rayDepth = ndcZ;

                if (rayDepth > 0 && rayDepth < 1 && sceneDepth > 0 && sceneDepth < 1)
                {
                    float depthDiff = sceneDepth - rayDepth;
                    if (depthDiff > 0 && depthDiff < thickness / clipPos.W)
                    {
                        hitUV = new Vector2((float)sx / width, (float)sy / height);
                        return true;
                    }
                }

                prevDepth = sceneDepth;
            }

            return false;
        }

        private static Vector3 ReconstructViewPos(int x, int y, float depth, int w, int h, Matrix4x4 invProj)
        {
            float ndcX = (x + 0.5f) / w * 2.0f - 1.0f;
            float ndcY = (y + 0.5f) / h * 2.0f - 1.0f;
            Vector4 clipPos = new Vector4(ndcX, ndcY, depth, 1.0f);
            Vector4 viewPos = Vector4.Transform(clipPos, invProj);
            return new Vector3(viewPos.X, viewPos.Y, viewPos.Z) / viewPos.W;
        }

        private static Vector3 ReconstructViewNormal(int x, int y, DepthBuffer depthBuffer, int w, int h, Matrix4x4 invView, Matrix4x4 proj)
        {
            float dx1 = depthBuffer.GetDepth(Math.Clamp(x - 1, 0, w - 1), y);
            float dx2 = depthBuffer.GetDepth(Math.Clamp(x + 1, 0, w - 1), y);
            float dy1 = depthBuffer.GetDepth(x, Math.Clamp(y - 1, 0, h - 1));
            float dy2 = depthBuffer.GetDepth(x, Math.Clamp(y + 1, 0, h - 1));

            Vector3 ddx = new Vector3(dx2 - dx1, 0, 0);
            Vector3 ddy = new Vector3(0, dy2 - dy1, 0);
            Vector3 normal = Vector3.Cross(ddy, ddx);
            if (normal.LengthSquared() > 0.0001f)
                normal = Vector3.Normalize(normal);
            else
                normal = Vector3.UnitY;
            return normal;
        }

        private static void EnsurePrevFrame(int w, int h)
        {
            if (_prevFrame == null || _prevWidth != w || _prevHeight != h)
            {
                _prevWidth = w;
                _prevHeight = h;
                _prevFrame = new float[w * h * 4];
            }
        }

        public static void ResetHistory() { _prevFrame = null; }
    }

    /// <summary>
    /// Applies optional final-screen artistic looks after tonemapping.
    /// </summary>
    public static class ArtisticStylePass
    {
        public static void Apply(HDRFrameBuffer input, HDRFrameBuffer output, ArtisticStyleConfig config)
        {
            if (!config.Enabled || config.Style == ArtisticStyle.None)
            {
                if (!ReferenceEquals(input, output))
                    input.CopyTo(output);
                return;
            }

            int width = input.Width;
            int height = input.Height;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Vector4 pixel = input.GetPixel(x, y);
                    Vector3 color = new(pixel.X, pixel.Y, pixel.Z);

                    color = config.Style switch
                    {
                        ArtisticStyle.Grayscale => ApplyGrayscale(color),
                        ArtisticStyle.Cartoon => ApplyCartoon(color, input, x, y, width, height, config),
                        ArtisticStyle.Noir => ApplyNoir(color, config),
                        _ => color
                    };

                    output.SetPixel(x, y, new Vector4(color, pixel.W));
                }
            }
        }

        public static Vector3 ApplyGrayscale(Vector3 color)
        {
            float lum = Luminance(color);
            return new Vector3(lum);
        }

        public static Vector3 ApplyCartoon(
            Vector3 color,
            HDRFrameBuffer input,
            int x,
            int y,
            int width,
            int height,
            ArtisticStyleConfig config)
        {
            int levels = Math.Max(2, config.CartoonColorLevels);
            float step = 1.0f / (levels - 1);
            color = new Vector3(
                MathF.Round(color.X / step) * step,
                MathF.Round(color.Y / step) * step,
                MathF.Round(color.Z / step) * step);

            if (config.CartoonEdgeStrength <= 0f)
                return color;

            float center = Luminance(input.GetPixel(x, y));
            float right = Luminance(input.GetPixel(Math.Min(x + 1, width - 1), y));
            float down = Luminance(input.GetPixel(x, Math.Min(y + 1, height - 1)));
            float edge = MathF.Abs(center - right) + MathF.Abs(center - down);
            float edgeFactor = Math.Clamp(1.0f - edge * config.CartoonEdgeStrength * 4.0f, 0.35f, 1.0f);
            return color * edgeFactor;
        }

        public static Vector3 ApplyNoir(Vector3 color, ArtisticStyleConfig config)
        {
            float lum = Luminance(color);
            lum = (lum - 0.5f) * config.NoirContrast + 0.5f;
            lum = MathF.Max(lum, config.NoirCrush);
            lum = Math.Clamp(lum, 0.0f, 1.0f);
            return new Vector3(lum);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Luminance(Vector4 pixel) => Luminance(new Vector3(pixel.X, pixel.Y, pixel.Z));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Luminance(Vector3 color) =>
            0.2126f * color.X + 0.7152f * color.Y + 0.0722f * color.Z;
    }

    // =========================================================================
    // POST-PROCESSING PIPELINE
    // =========================================================================

    /// <summary>
    /// Complete post-processing pipeline orchestrating Bloom, DOF, Motion Blur,
    /// and Tonemapping in the correct order with shared resources.
    /// </summary>
    public class PostProcessingPipeline : IDisposable
    {
        private PostProcessConfig _config;
        private HDRFrameBuffer? _tempBuffer0;
        private HDRFrameBuffer? _tempBuffer1;
        private int _lastWidth, _lastHeight;
        private readonly object _lock = new();
        private float _autoExposure = 1.0f;
        private float[] _luminanceHistory = new float[64];
        private int _luminanceIndex;

        public PostProcessConfig Config
        {
            get => _config;
            set => _config = value;
        }

        public float CurrentExposure => _autoExposure;

        public PostProcessingPipeline(PostProcessConfig? config = null)
        {
            _config = config ?? new PostProcessConfig();
        }

        public void Process(
            HDRFrameBuffer colorBuffer,
            DepthBuffer? depthBuffer,
            VelocityBuffer? velocityBuffer,
            HDRFrameBuffer outputBuffer,
            float aspectRatio = 16.0f / 9.0f)
        {
            lock (_lock)
            {
                EnsureBuffers(colorBuffer.Width, colorBuffer.Height);

                if (_config.Tonemap.EnableAutoExposure && depthBuffer != null)
                {
                    UpdateAutoExposure(colorBuffer);
                }

                var current = colorBuffer;

                if (_config.MotionBlur.Enabled && velocityBuffer != null)
                {
                    MotionBlurPass.Apply(current, velocityBuffer, depthBuffer ?? CreateDummyDepth(current.Width, current.Height), _tempBuffer0!, _config.MotionBlur);
                    Swap(ref current, ref _tempBuffer0!);
                }

                if (_config.DOF.Enabled && depthBuffer != null)
                {
                    DOFPass.Apply(current, depthBuffer, _tempBuffer0!, _config.DOF, aspectRatio);
                    Swap(ref current, ref _tempBuffer0!);
                }

                if (_config.Bloom.Enabled)
                {
                    BloomPass.Apply(current, _tempBuffer0!, _config.Bloom);
                    Swap(ref current, ref _tempBuffer0!);
                }

                if (_config.SSR.Enabled && depthBuffer != null)
                {
                    SSRPass.Apply(current, depthBuffer, _tempBuffer0!, _config.SSR, Matrix4x4.Identity, Matrix4x4.Identity, colorBuffer.Width, colorBuffer.Height);
                    Swap(ref current, ref _tempBuffer0!);
                }

                if (_config.TAA.Enabled && velocityBuffer != null && depthBuffer != null)
                {
                    TAAPass.Apply(current, _tempBuffer0!, velocityBuffer, depthBuffer, _config.TAA, Matrix4x4.Identity, Matrix4x4.Identity, colorBuffer.Width, colorBuffer.Height);
                    Swap(ref current, ref _tempBuffer0!);
                }

                TonemapPass.Apply(current, _tempBuffer0!, _config.Tonemap);

                if (_config.ArtisticStyle.Enabled && _config.ArtisticStyle.Style != ArtisticStyle.None)
                    ArtisticStylePass.Apply(_tempBuffer0!, outputBuffer, _config.ArtisticStyle);
                else
                    _tempBuffer0!.CopyTo(outputBuffer);
            }
        }

        private void UpdateAutoExposure(HDRFrameBuffer buffer)
        {
            float totalLuminance = 0;
            int sampleCount = 0;
            int step = Math.Max(1, buffer.Width * buffer.Height / 1024);

            for (int i = 0; i < buffer.Data.Length; i += 4 * step)
            {
                float lum = 0.2126f * buffer.Data[i] + 0.7152f * buffer.Data[i + 1] + 0.0722f * buffer.Data[i + 2];
                totalLuminance += MathF.Log(MathF.Max(lum, 0.0001f));
                sampleCount++;
            }

            float avgLuminance = sampleCount > 0 ? MathF.Exp(totalLuminance / sampleCount) : 0.5f;
            _luminanceHistory[_luminanceIndex % _luminanceHistory.Length] = avgLuminance;
            _luminanceIndex++;

            float smoothLum = _luminanceHistory.Average();
            float targetExposure = Math.Clamp(0.5f / MathF.Max(smoothLum, 0.0001f), _config.Tonemap.MinExposure, _config.Tonemap.MaxExposure);
            _autoExposure += (targetExposure - _autoExposure) * _config.Tonemap.AutoExposureSpeed * 0.016f;
        }

        private void EnsureBuffers(int w, int h)
        {
            if (_tempBuffer0 == null || _lastWidth != w || _lastHeight != h)
            {
                _tempBuffer0?.Dispose();
                _tempBuffer1?.Dispose();
                _tempBuffer0 = new HDRFrameBuffer(w, h);
                _tempBuffer1 = new HDRFrameBuffer(w, h);
                _lastWidth = w;
                _lastHeight = h;
            }
        }

        private static void Swap(ref HDRFrameBuffer a, ref HDRFrameBuffer b)
        {
            (a, b) = (b, a);
        }

        private static DepthBuffer CreateDummyDepth(int w, int h) => new(w, h);

        public void Dispose()
        {
            _tempBuffer0?.Dispose();
            _tempBuffer1?.Dispose();
        }
    }
}