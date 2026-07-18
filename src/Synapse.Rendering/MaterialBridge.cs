using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using GDNN.Materials.SubstrateOmega;
using GDNN.RHI.Vulkan;

namespace GDNN.Rendering.Bridge
{
    [StructLayout(LayoutKind.Sequential)]
    public struct MaterialUBO
    {
        public Vector4 BaseColor;
        public Vector4 Emissive;
        public float Roughness;
        public float Metallic;
        public float AO;
        public float NormalScale;
        public float Opacity;
        public float Specular;
        public float Subsurface;
        public float ClearCoat;
        public float Sheen;
        public float Padding0;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CameraUBO
    {
        public Matrix4x4 View;
        public Matrix4x4 Projection;
        public Matrix4x4 ViewProjection;
        public Matrix4x4 InverseView;
        public Matrix4x4 InverseProjection;
        public Vector4 CameraPosition;
        public Vector4 ScreenParams;
        public Vector4 TimeParams;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LightUBO
    {
        public Vector4 Position;
        public Vector4 Direction;
        public Vector4 Color;
        public float Intensity;
        public float Range;
        public float InnerConeAngle;
        public float OuterConeAngle;
    }

    public class MaterialBridge : IDisposable
    {
        private readonly VulkanRhiDevice _rhi;
        private readonly Dictionary<string, VulkanBuffer> _materialBuffers = new();
        private bool _disposed;

        public MaterialBridge(VulkanRhiDevice rhi)
        {
            _rhi = rhi;
        }

        public MaterialUBO ExtractProperties(SubstrateMaterial material)
        {
            var baseColor = material.GetColor("BaseColor", new Color3(0.8f, 0.8f, 0.8f));
            var emissive = material.GetColor("Emissive", new Color3(0, 0, 0));

            return new MaterialUBO
            {
                BaseColor = new Vector4(baseColor.R, baseColor.G, baseColor.B, 1.0f),
                Emissive = new Vector4(emissive.R, emissive.G, emissive.B, material.GetFloat("EmissiveIntensity", 1.0f)),
                Roughness = material.GetFloat("Roughness", 0.5f),
                Metallic = material.GetFloat("Metallic", 0.0f),
                AO = material.GetFloat("AO", 1.0f),
                NormalScale = material.GetFloat("NormalScale", 1.0f),
                Opacity = material.GetFloat("Opacity", 1.0f),
                Specular = material.GetFloat("Specular", 0.5f),
                Subsurface = material.GetFloat("Subsurface", 0.0f),
                ClearCoat = material.GetFloat("ClearCoat", 0.0f),
                Sheen = material.GetFloat("Sheen", 0.0f),
            };
        }

        public VulkanBuffer UploadMaterial(SubstrateMaterial material)
        {
            var key = material.Id;
            if (_materialBuffers.TryGetValue(key, out var existing))
                return existing;

            var ubo = ExtractProperties(material);
            var size = (ulong)Marshal.SizeOf<MaterialUBO>();
            var buffer = _rhi.CreateBuffer(new BufferDescription
            {
                Size = size,
                Usage = BufferUsageFlag.UniformBuffer,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });

            var mapped = buffer.Map();
            Marshal.StructureToPtr(ubo, mapped, false);
            buffer.Unmap();

            _materialBuffers[key] = buffer;
            return buffer;
        }

        public void UpdateMaterial(SubstrateMaterial material)
        {
            var key = material.Id;
            if (!_materialBuffers.TryGetValue(key, out var buffer)) return;

            var ubo = ExtractProperties(material);
            var mapped = buffer.Map();
            Marshal.StructureToPtr(ubo, mapped, false);
            buffer.Unmap();
        }

        public CameraUBO BuildCameraUBO(
            Matrix4x4 view, Matrix4x4 projection,
            Vector3 cameraPos, float time,
            float width, float height)
        {
            var vp = view * projection;
            Matrix4x4.Invert(view, out var invView);
            Matrix4x4.Invert(projection, out var invProj);

            return new CameraUBO
            {
                View = view,
                Projection = projection,
                ViewProjection = vp,
                InverseView = invView,
                InverseProjection = invProj,
                CameraPosition = new Vector4(cameraPos.X, cameraPos.Y, cameraPos.Z, 1.0f),
                ScreenParams = new Vector4(width, height, 1.0f / width, 1.0f / height),
                TimeParams = new Vector4(time, 1.0f / time, 0, 0),
            };
        }

        public LightUBO BuildLightData(
            Vector3 position, Vector3 direction, Vector3 color,
            float intensity, float range, float innerCone, float outerCone)
        {
            return new LightUBO
            {
                Position = new Vector4(position.X, position.Y, position.Z, 1.0f),
                Direction = new Vector4(direction.X, direction.Y, direction.Z, 0.0f),
                Color = new Vector4(color.X, color.Y, color.Z, 1.0f),
                Intensity = intensity,
                Range = range,
                InnerConeAngle = innerCone,
                OuterConeAngle = outerCone,
            };
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var buf in _materialBuffers.Values)
                buf?.Dispose();
            _materialBuffers.Clear();
        }
    }
}
