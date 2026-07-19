using System;
using System.Numerics;
using System.Runtime.InteropServices;
using GDNN.RHI.Vulkan;

namespace GDNN.Rendering.Engine
{
    /// <summary>CPU-side snapshot of all deferred G-buffer attachments after GPU readback.</summary>
    public sealed class GBufferSnapshot
    {
        public int Width { get; init; }
        public int Height { get; init; }
        public float[] Depth = Array.Empty<float>();
        public Vector3[] Normals = Array.Empty<Vector3>();
        public Vector3[] Albedo = Array.Empty<Vector3>();
        public Vector2[] Velocity = Array.Empty<Vector2>();
        public Vector4[] MaterialProps = Array.Empty<Vector4>();
        public Vector3[] Specular = Array.Empty<Vector3>();
        public Vector3[] Emissive = Array.Empty<Vector3>();
    }

    /// <summary>Reads Vulkan G-buffer color attachments into CPU arrays for L-DNN GI.</summary>
    public sealed class GBufferReadback : IDisposable
    {
        private readonly VulkanRhiDevice _rhi;
        private VulkanBuffer? _staging;
        private int _width;
        private int _height;
        private bool _disposed;

        public GBufferReadback(VulkanRhiDevice rhi) => _rhi = rhi;

        public bool TryRead(
            VulkanTexture albedo,
            VulkanTexture normals,
            VulkanTexture depthColor,
            VulkanTexture material,
            VulkanTexture velocity,
            int width,
            int height,
            out GBufferSnapshot snapshot)
        {
            snapshot = new GBufferSnapshot { Width = width, Height = height };
            if (width <= 0 || height <= 0)
                return false;

            EnsureStaging(width, height);
            int pixelCount = width * height;
            snapshot.Depth = new float[pixelCount];
            snapshot.Normals = new Vector3[pixelCount];
            snapshot.Albedo = new Vector3[pixelCount];
            snapshot.Velocity = new Vector2[pixelCount];
            snapshot.MaterialProps = new Vector4[pixelCount];
            snapshot.Specular = new Vector3[pixelCount];
            snapshot.Emissive = new Vector3[pixelCount];

            var albedoRaw = ReadColorAttachment(albedo, width, height);
            var normalRaw = ReadColorAttachment(normals, width, height);
            var depthRaw = ReadColorAttachment(depthColor, width, height);
            var materialRaw = ReadColorAttachment(material, width, height);
            var velocityRaw = ReadColorAttachment(velocity, width, height);

            if (albedoRaw == null || normalRaw == null || depthRaw == null || materialRaw == null || velocityRaw == null)
                return false;

            for (int i = 0; i < pixelCount; i++)
            {
                int o = i * 4;
                snapshot.Albedo[i] = new Vector3(albedoRaw[o], albedoRaw[o + 1], albedoRaw[o + 2]);
                snapshot.Normals[i] = Vector3.Normalize(new Vector3(normalRaw[o], normalRaw[o + 1], normalRaw[o + 2]));
                snapshot.Depth[i] = depthRaw[o + 2];
                snapshot.MaterialProps[i] = new Vector4(materialRaw[o], materialRaw[o + 1], materialRaw[o + 2], materialRaw[o + 3]);
                snapshot.Velocity[i] = new Vector2(velocityRaw[o], velocityRaw[o + 1]);
                float metallic = materialRaw[o + 1];
                float roughness = MathF.Max(materialRaw[o], 0.04f);
                snapshot.Specular[i] = new Vector3(0.04f * (1f - metallic) + metallic, 0.04f * (1f - metallic) + metallic, 0.04f * (1f - metallic) + metallic);
                snapshot.Emissive[i] = snapshot.Albedo[i] * (1f - roughness) * 0.02f;
            }

            return true;
        }

        private float[]? ReadColorAttachment(VulkanTexture texture, int width, int height)
        {
            if (_staging == null)
                return null;
            ulong rowPitch = (ulong)width * 8;
            ulong imageSize = rowPitch * (ulong)height;

            var cmd = _rhi.CreateCommandBuffer();
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);

            cmd.PipelineBarrier(
                PipelineStageFlag.FragmentShader,
                PipelineStageFlag.Transfer,
                imageBarriers: new[]
                {
                    new ImageMemoryBarrier
                    {
                        Image = texture.Handle,
                        OldLayout = ImageLayout.ShaderReadOnlyOptimal,
                        NewLayout = ImageLayout.TransferSrcOptimal,
                        SrcAccessMask = AccessFlag.ShaderRead,
                        DstAccessMask = AccessFlag.TransferRead,
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

            cmd.CopyImageToBuffer(texture, ImageLayout.TransferSrcOptimal, _staging, new[]
            {
                new BufferImageCopy
                {
                    BufferOffset = 0,
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
                        Image = texture.Handle,
                        OldLayout = ImageLayout.TransferSrcOptimal,
                        NewLayout = ImageLayout.ShaderReadOnlyOptimal,
                        SrcAccessMask = AccessFlag.TransferRead,
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

            var mapped = _staging.Map(0, imageSize);
            if (mapped == IntPtr.Zero)
                return null;

            var result = new float[width * height * 4];
            unsafe
            {
                var src = (Half*)mapped;
                for (int i = 0; i < result.Length; i++)
                    result[i] = (float)src[i];
            }
            _staging.Unmap();
            return result;
        }

        private void EnsureStaging(int width, int height)
        {
            if (_staging != null && _width == width && _height == height)
                return;
            _staging?.Dispose();
            _width = width;
            _height = height;
            ulong size = (ulong)width * 8 * (ulong)height;
            _staging = _rhi.CreateBuffer(new BufferDescription
            {
                Size = size,
                Usage = BufferUsageFlag.TransferDst,
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
