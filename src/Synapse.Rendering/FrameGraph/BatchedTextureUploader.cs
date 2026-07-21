using System;
using System.Numerics;
using System.Runtime.InteropServices;
using GDNN.RHI.Vulkan;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>
    /// Batches GI / AO / fog Half-RGBA uploads into a single command buffer + one WaitForIdle.
    /// Replaces per-texture Submit/Wait stalls on the L-DNN present path.
    /// </summary>
    public sealed class BatchedTextureUploader : IDisposable
    {
        private readonly VulkanRhiDevice _rhi;
        private VulkanBuffer? _staging;
        private int _capacityBytes;
        private bool _disposed;

        public BatchedTextureUploader(VulkanRhiDevice rhi)
        {
            _rhi = rhi ?? throw new ArgumentNullException(nameof(rhi));
        }

        public void Upload(
            VulkanTexture? gi, Vector3[,]? giField,
            VulkanTexture? ao, float[,]? aoField,
            VulkanTexture? fog, Vector3[,]? fogField,
            int width, int height)
        {
            if (width <= 0 || height <= 0)
                return;

            int pixelCount = width * height;
            int stride = pixelCount * 8;
            int total = 0;
            if (gi != null && giField != null)
                total += stride;
            if (ao != null && aoField != null)
                total += stride;
            if (fog != null && fogField != null)
                total += stride;
            if (total == 0)
                return;

            EnsureStaging(total);
            var bytes = new byte[total];
            int offset = 0;

            void PackRgb(Vector3[,] field)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        var c = field[x, y];
                        int idx = offset + (y * width + x) * 8;
                        BitConverter.TryWriteBytes(bytes.AsSpan(idx, 2), (Half)c.X);
                        BitConverter.TryWriteBytes(bytes.AsSpan(idx + 2, 2), (Half)c.Y);
                        BitConverter.TryWriteBytes(bytes.AsSpan(idx + 4, 2), (Half)c.Z);
                        BitConverter.TryWriteBytes(bytes.AsSpan(idx + 6, 2), (Half)1f);
                    }
                }
                offset += stride;
            }

            void PackAo(float[,] field)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        float v = field[x, y];
                        int idx = offset + (y * width + x) * 8;
                        BitConverter.TryWriteBytes(bytes.AsSpan(idx, 2), (Half)v);
                        BitConverter.TryWriteBytes(bytes.AsSpan(idx + 2, 2), (Half)v);
                        BitConverter.TryWriteBytes(bytes.AsSpan(idx + 4, 2), (Half)v);
                        BitConverter.TryWriteBytes(bytes.AsSpan(idx + 6, 2), (Half)1f);
                    }
                }
                offset += stride;
            }

            int giOff = -1, aoOff = -1, fogOff = -1;
            if (gi != null && giField != null)
            { giOff = offset; PackRgb(giField); }
            if (ao != null && aoField != null)
            { aoOff = offset; PackAo(aoField); }
            if (fog != null && fogField != null)
            { fogOff = offset; PackRgb(fogField); }

            var mapped = _staging!.Map();
            Marshal.Copy(bytes, 0, mapped, bytes.Length);
            _staging.Unmap();

            var cmd = _rhi.CreateCommandBuffer();
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);

            void CopyTo(VulkanTexture tex, int srcOffset)
            {
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
                cmd.CopyBufferToImage(_staging, tex, ImageLayout.TransferDstOptimal, new[]
                {
                    new BufferImageCopy
                    {
                        BufferOffset = (ulong)srcOffset,
                        BufferRowLength = 0,
                        BufferImageHeight = 0,
                        ImageSubresource = new ImageSubresourceLayers
                        {
                            AspectMask = ImageAspectFlag.Color,
                            MipLevel = 0,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        },
                        ImageOffset = new Offset3D(),
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
            }

            if (giOff >= 0)
                CopyTo(gi!, giOff);
            if (aoOff >= 0)
                CopyTo(ao!, aoOff);
            if (fogOff >= 0)
                CopyTo(fog!, fogOff);

            cmd.End();
            _rhi.SubmitCommandBuffer(cmd, _rhi.GraphicsQueue, null);
            _rhi.WaitForIdle();
        }

        private void EnsureStaging(int bytes)
        {
            if (_staging != null && _capacityBytes >= bytes)
                return;
            _staging?.Dispose();
            _capacityBytes = bytes;
            _staging = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)bytes,
                Usage = BufferUsageFlag.TransferSrc,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _staging?.Dispose();
        }
    }
}
