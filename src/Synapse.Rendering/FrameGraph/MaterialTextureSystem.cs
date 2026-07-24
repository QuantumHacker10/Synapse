using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using GDNN.Materials.SubstrateOmega;
using GDNN.RHI.Vulkan;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>
    /// Material texture cache for G-buffer PBR sampling (Megascans / Substrate channels).
    /// Decodes on-disk PNG/JPEG/BMP/TGA via <see cref="MegascansImageDecoder"/> into Vulkan
    /// RGBA16F textures; falls back to procedural patterned textures when files are missing/corrupt.
    /// </summary>
    public sealed class MaterialTextureSystem : IDisposable
    {
        private const int ProcSize = 64;

        private readonly VulkanRhiDevice _rhi;
        private readonly Dictionary<string, VulkanTexture> _cache = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _failedPaths = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<int, VulkanTexture> _procAlbedo = new();
        private readonly Dictionary<int, VulkanTexture> _procNormal = new();
        private readonly Dictionary<int, VulkanTexture> _procOrm = new();
        private VulkanTexture? _white;
        private VulkanTexture? _flatNormal;
        private VulkanTexture? _ormDefault;
        private VulkanTexture? _vtAtlas;
        private Sampler? _sampler;
        private bool _disposed;

        public MaterialTextureSystem(VulkanRhiDevice rhi)
        {
            _rhi = rhi ?? throw new ArgumentNullException(nameof(rhi));
        }

        public Sampler Sampler => _sampler ??= _rhi.CreateSampler(new SamplerDescription
        {
            MagFilter = Filter.Linear,
            MinFilter = Filter.Linear,
            AddressModeU = SamplerAddressMode.Repeat,
            AddressModeV = SamplerAddressMode.Repeat,
            AddressModeW = SamplerAddressMode.Repeat,
            AnisotropyEnable = false,
            MaxAnisotropy = 1f,
            CompareEnable = false,
            CompareOp = CompareOp.Always,
            MinLod = 0f,
            MaxLod = 1f,
        });

        public VulkanTexture White => _white ??= CreateSolid(1f, 1f, 1f, 1f);
        public VulkanTexture FlatNormal => _flatNormal ??= CreateSolid(0.5f, 0.5f, 1f, 1f);
        public VulkanTexture OrmDefault => _ormDefault ??= CreateSolid(1f, 0.5f, 0f, 1f);

        /// <summary>Optional VT physical atlas sampled as world albedo when bound.</summary>
        public VulkanTexture? VirtualTextureAtlas => _vtAtlas;

        public VulkanTexture ResolveAlbedo(SubstrateMaterial? material)
        {
            var path = material?.GetTexture(TextureChannel.Albedo)?.Path;
            if (TryResolvePath(path, out var tex))
                return tex;
            return ResolveProcedural(material, TextureKind.Albedo);
        }

        public VulkanTexture ResolveNormal(SubstrateMaterial? material)
        {
            var path = material?.GetTexture(TextureChannel.Normal)?.Path;
            if (TryResolvePath(path, out var tex))
                return tex;
            return ResolveProcedural(material, TextureKind.Normal);
        }

        public VulkanTexture ResolveOrm(SubstrateMaterial? material)
        {
            var path = material?.GetTexture(TextureChannel.AO)?.Path
                       ?? material?.GetTexture(TextureChannel.Roughness)?.Path
                       ?? material?.GetTexture(TextureChannel.Metallic)?.Path;
            if (TryResolvePath(path, out var tex))
                return tex;
            return ResolveProcedural(material, TextureKind.Orm);
        }

        /// <summary>
        /// Uploads a resident VT physical color page grid into a sampled atlas (64×64 tiles packed).
        /// </summary>
        public unsafe void UploadVirtualTextureAtlas(float[]? physicalRgba, int physicalW, int physicalH)
        {
            if (physicalRgba == null || physicalW <= 0 || physicalH <= 0)
                return;

            int atlasW = Math.Min(physicalW, 256);
            int atlasH = Math.Min(physicalH, 256);
            int pixelCount = atlasW * atlasH;
            var bytes = new byte[pixelCount * 8];
            for (int y = 0; y < atlasH; y++)
            {
                for (int x = 0; x < atlasW; x++)
                {
                    int src = (y * physicalW + x) * 4;
                    int dst = (y * atlasW + x) * 8;
                    float r = src + 3 < physicalRgba.Length ? physicalRgba[src] : 0.4f;
                    float g = src + 3 < physicalRgba.Length ? physicalRgba[src + 1] : 0.35f;
                    float b = src + 3 < physicalRgba.Length ? physicalRgba[src + 2] : 0.3f;
                    BitConverter.TryWriteBytes(bytes.AsSpan(dst, 2), (Half)r);
                    BitConverter.TryWriteBytes(bytes.AsSpan(dst + 2, 2), (Half)g);
                    BitConverter.TryWriteBytes(bytes.AsSpan(dst + 4, 2), (Half)b);
                    BitConverter.TryWriteBytes(bytes.AsSpan(dst + 6, 2), (Half)1f);
                }
            }

            _vtAtlas?.Dispose();
            _vtAtlas = CreateFromHalfRgba(bytes, atlasW, atlasH);
        }

        private enum TextureKind { Albedo, Normal, Orm }

        private bool TryResolvePath(string? path, out VulkanTexture tex)
        {
            tex = White;
            if (string.IsNullOrWhiteSpace(path))
                return false;
            // procedural:// and other non-disk schemes stay on the procedural path.
            if (path.Contains("://", StringComparison.Ordinal))
                return false;
            if (_cache.TryGetValue(path, out tex!))
                return true;
            if (_failedPaths.Contains(path))
                return false;

            if (!File.Exists(path))
            {
                _failedPaths.Add(path);
                return false;
            }

            if (MegascansImageDecoder.TryDecodeFile(path, out var decoded))
            {
                tex = CreateFromDecoded(decoded);
                _cache[path] = tex;
                return true;
            }

            // Corrupt / unsupported (EXR/TIFF/etc.): keep procedural fallback for this material.
            _failedPaths.Add(path);
            return false;
        }

        private unsafe VulkanTexture CreateFromDecoded(DecodedImage img)
        {
            int w = img.Width;
            int h = img.Height;
            var bytes = new byte[w * h * 8];
            var src = img.Rgba8;
            for (int i = 0; i < w * h; i++)
            {
                int si = i * 4;
                int di = i * 8;
                BitConverter.TryWriteBytes(bytes.AsSpan(di, 2), (Half)(src[si] / 255f));
                BitConverter.TryWriteBytes(bytes.AsSpan(di + 2, 2), (Half)(src[si + 1] / 255f));
                BitConverter.TryWriteBytes(bytes.AsSpan(di + 4, 2), (Half)(src[si + 2] / 255f));
                BitConverter.TryWriteBytes(bytes.AsSpan(di + 6, 2), (Half)(src[si + 3] / 255f));
            }
            return CreateFromHalfRgba(bytes, w, h);
        }

        private VulkanTexture ResolveProcedural(SubstrateMaterial? material, TextureKind kind)
        {
            int seed = material != null
                ? HashString(material.Id ?? material.Name ?? "mat")
                : kind switch { TextureKind.Normal => 1, TextureKind.Orm => 2, _ => 0 };

            var map = kind switch
            {
                TextureKind.Normal => _procNormal,
                TextureKind.Orm => _procOrm,
                _ => _procAlbedo
            };
            if (map.TryGetValue(seed, out var existing))
                return existing;

            var tex = CreateProcedural(seed, kind);
            map[seed] = tex;
            return tex;
        }

        private VulkanTexture CreateProcedural(int seed, TextureKind kind)
        {
            var bytes = new byte[ProcSize * ProcSize * 8];
            var rng = new Random(seed);
            float hue = (float)rng.NextDouble();
            float sat = 0.35f + (float)rng.NextDouble() * 0.45f;
            float roughBase = 0.2f + (float)rng.NextDouble() * 0.7f;
            float metalBase = (float)rng.NextDouble() > 0.7f ? 0.85f : 0.05f;

            for (int y = 0; y < ProcSize; y++)
            {
                for (int x = 0; x < ProcSize; x++)
                {
                    float u = x / (float)ProcSize;
                    float v = y / (float)ProcSize;
                    int checker = ((x / 8) ^ (y / 8)) & 1;
                    float stripes = MathF.Sin(u * 18f + seed) * 0.5f + 0.5f;
                    float noise = Fract(MathF.Sin(x * 12.9898f + y * 78.233f + seed) * 43758.5453f);
                    int o = (y * ProcSize + x) * 8;

                    float r, g, b, a = 1f;
                    switch (kind)
                    {
                        case TextureKind.Normal:
                            float nx = (stripes - 0.5f) * 0.6f + 0.5f;
                            float ny = (noise - 0.5f) * 0.6f + 0.5f;
                            r = nx;
                            g = ny;
                            b = 1f;
                            r = nx; g = ny; b = 1f;
                            break;
                        case TextureKind.Orm:
                            r = 0.75f + noise * 0.25f;           // AO
                            g = Math.Clamp(roughBase + (checker * 0.15f) - 0.05f, 0.04f, 1f);
                            b = metalBase * (0.7f + stripes * 0.3f);
                            break;
                        default:
                            var rgb = HsvToRgb(hue + checker * 0.08f, sat, 0.45f + stripes * 0.35f + noise * 0.1f);
                            r = rgb.X;
                            g = rgb.Y;
                            b = rgb.Z;
                            r = rgb.X; g = rgb.Y; b = rgb.Z;
                            break;
                    }

                    BitConverter.TryWriteBytes(bytes.AsSpan(o, 2), (Half)r);
                    BitConverter.TryWriteBytes(bytes.AsSpan(o + 2, 2), (Half)g);
                    BitConverter.TryWriteBytes(bytes.AsSpan(o + 4, 2), (Half)b);
                    BitConverter.TryWriteBytes(bytes.AsSpan(o + 6, 2), (Half)a);
                }
            }

            return CreateFromHalfRgba(bytes, ProcSize, ProcSize);
        }

        private static float Fract(float x) => x - MathF.Floor(x);

        private static int HashString(string s)
        {
            unchecked
            {
                int h = 23;
                foreach (char c in s)
                    h = h * 31 + c;
                return h == 0 ? 1 : h;
            }
        }

        private static Vector3 HsvToRgb(float h, float s, float v)
        {
            h = Fract(h);
            float c = v * s;
            float x = c * (1f - MathF.Abs(Fract(h * 6f) * 2f - 1f));
            float m = v - c;
            Vector3 rgb = (h * 6f) switch
            {
                < 1f => new Vector3(c, x, 0),
                < 2f => new Vector3(x, c, 0),
                < 3f => new Vector3(0, c, x),
                < 4f => new Vector3(0, x, c),
                < 5f => new Vector3(x, 0, c),
                _ => new Vector3(c, 0, x)
            };
            return rgb + new Vector3(m);
        }

        private unsafe VulkanTexture CreateSolid(float r, float g, float b, float a)
        {
            var bytes = new byte[8];
            BitConverter.TryWriteBytes(bytes.AsSpan(0, 2), (Half)r);
            BitConverter.TryWriteBytes(bytes.AsSpan(2, 2), (Half)g);
            BitConverter.TryWriteBytes(bytes.AsSpan(4, 2), (Half)b);
            BitConverter.TryWriteBytes(bytes.AsSpan(6, 2), (Half)a);
            return CreateFromHalfRgba(bytes, 1, 1);
        }

        private unsafe VulkanTexture CreateFromHalfRgba(byte[] bytes, int width, int height)
        {
            var tex = _rhi.CreateTexture(new TextureDescription
            {
                Width = (uint)width,
                Height = (uint)height,
                Format = VulkanFormat.R16G16B16A16Sfloat,
                Usage = ImageUsageFlag.Sampled | ImageUsageFlag.TransferDst,
                Tiling = ImageTiling.Optimal,
                InitialLayout = ImageLayout.Undefined,
                Samples = SampleCountFlag.Count1,
            });

            var staging = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)bytes.Length,
                Usage = BufferUsageFlag.TransferSrc,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });
            var mapped = staging.Map();
            Marshal.Copy(bytes, 0, mapped, bytes.Length);
            staging.Unmap();

            var cmd = _rhi.CreateCommandBuffer();
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            cmd.PipelineBarrier(
                PipelineStageFlag.TopOfPipe,
                PipelineStageFlag.Transfer,
                imageBarriers: new[]
                {
                    new ImageMemoryBarrier
                    {
                        Image = tex.Handle,
                        OldLayout = ImageLayout.Undefined,
                        NewLayout = ImageLayout.TransferDstOptimal,
                        DstAccessMask = AccessFlag.TransferWrite,
                        SubresourceRange = new ImageSubresourceRange
                        {
                            AspectMask = ImageAspectFlag.Color,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    }
                });
            cmd.CopyBufferToImage(staging, tex, ImageLayout.TransferDstOptimal, new[]
            {
                new BufferImageCopy
                {
                    BufferOffset = 0,
                    ImageSubresource = new ImageSubresourceLayers
                    {
                        AspectMask = ImageAspectFlag.Color,
                        MipLevel = 0,
                        BaseArrayLayer = 0,
                        LayerCount = 1
                    },
                    ImageExtent = new Extent3D((uint)width, (uint)height, 1)
                }
            });
            cmd.PipelineBarrier(
                PipelineStageFlag.Transfer,
                PipelineStageFlag.FragmentShader,
                imageBarriers: new[]
                {
                    new ImageMemoryBarrier
                    {
                        Image = tex.Handle,
                        OldLayout = ImageLayout.TransferDstOptimal,
                        NewLayout = ImageLayout.ShaderReadOnlyOptimal,
                        SrcAccessMask = AccessFlag.TransferWrite,
                        DstAccessMask = AccessFlag.ShaderRead,
                        SubresourceRange = new ImageSubresourceRange
                        {
                            AspectMask = ImageAspectFlag.Color,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    }
                });
            cmd.End();
            _rhi.SubmitCommandBuffer(cmd, _rhi.GraphicsQueue, null);
            _rhi.WaitForIdle();
            staging.Dispose();
            return tex;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            if (_disposed) return;
            _disposed = true;
            foreach (var t in _cache.Values)
            {
                if (t != _white && t != _flatNormal && t != _ormDefault)
                    t.Dispose();
            }
            foreach (var t in _procAlbedo.Values)
                t.Dispose();
            foreach (var t in _procNormal.Values)
                t.Dispose();
            foreach (var t in _procOrm.Values)
                t.Dispose();
            foreach (var t in _procAlbedo.Values) t.Dispose();
            foreach (var t in _procNormal.Values) t.Dispose();
            foreach (var t in _procOrm.Values) t.Dispose();
            _vtAtlas?.Dispose();
            _white?.Dispose();
            _flatNormal?.Dispose();
            _ormDefault?.Dispose();
            _sampler?.Dispose();
            _cache.Clear();
            _failedPaths.Clear();
            _procAlbedo.Clear();
            _procNormal.Clear();
            _procOrm.Clear();
        }
    }
}
