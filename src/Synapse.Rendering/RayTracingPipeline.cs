// =============================================================================
// RayTracingPipeline.cs - GDNN Engine: Hardware Ray Tracing
// Vulkan RT extensions: BLAS, TLAS, ray tracing shaders, denoiser
// =============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using GDNN.RHI.Vulkan;

namespace GDNN.Rendering.RayTracing
{
    public enum RTAccelerationStructureType { BottomLevel = 0, TopLevel = 1 }
    public enum RTGeometryType { Triangles = 0, AABBs = 1 }
    public enum RTHitGroupType { Triangles = 0, Procedural = 1 }
    public enum RTDenoiserMode { None = 0, Temporal = 1, Spatial = 2, Joint = 3 }

    public record RTGeometryDesc
    {
        public RTGeometryType GeometryType { get; init; } = RTGeometryType.Triangles;
        public IntPtr VertexBuffer { get; init; }
        public ulong VertexOffset { get; init; }
        public uint VertexCount { get; init; }
        public uint VertexStride { get; init; }
        public VulkanFormat VertexFormat { get; init; } = VulkanFormat.R32G32B32Sfloat;
        public IntPtr IndexBuffer { get; init; }
        public ulong IndexOffset { get; init; }
        public uint IndexCount { get; init; }
        public VulkanFormat IndexFormat { get; init; } = VulkanFormat.R32Uint;
        public IntPtr TransformBuffer { get; init; }
        public ulong TransformOffset { get; init; }
        public bool Opaque { get; init; } = true;
        public bool NoDuplicateAnyHitInvocation { get; init; }
    }

    public record RTInstanceDesc
    {
        public Matrix4x4 Transform { get; init; } = Matrix4x4.Identity;
        public uint InstanceCustomIndex { get; init; }
        public uint Mask { get; init; } = 0xFF;
        public uint ShaderBindingTableOffset { get; init; }
        public uint Flags { get; init; }
        public IntPtr AccelerationStructureHandle { get; init; }
    }

    public record RTPipelineConfig
    {
        public uint MaxRecursionDepth { get; init; } = 2;
        public uint MaxPayloadSize { get; init; } = 64;
        public uint MaxAttributeSize { get; init; } = 32;
        public bool EnableTraversalShaders { get; init; }
    }

    public record RTDenoiserConfig
    {
        public RTDenoiserMode Mode { get; init; } = RTDenoiserMode.Temporal;
        public float Alpha { get; init; } = 0.2f;
        public float ZThreshold { get; init; } = 0.1f;
        public float NormalThreshold { get; init; } = 0.5f;
        public int SpatialRadius { get; init; } = 3;
        public int TemporalFrames { get; init; } = 8;
        public bool EnableEdgeStopping { get; init; } = true;
    }

    public class RTAccelerationStructure : IDisposable
    {
        public IntPtr Handle { get; set; }
        public RTAccelerationStructureType Type { get; set; }
        public VulkanBuffer Buffer { get; set; }
        public ulong CompactSize { get; set; }
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Buffer?.Dispose();
        }
    }

    public class RShaderBindingTable : IDisposable
    {
        public IntPtr Handle { get; set; }
        public VulkanBuffer Buffer { get; set; }
        public ulong RayGenOffset { get; set; }
        public ulong MissOffset { get; set; }
        public ulong HitGroupOffset { get; set; }
        public uint RayGenStride { get; set; }
        public uint MissStride { get; set; }
        public uint HitGroupStride { get; set; }
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Buffer?.Dispose();
        }
    }

    public class RTDenoiser : IDisposable
    {
        private RTDenoiserConfig _config;
        private float[] _temporalHistory;
        private float[] _velocityHistory;
        private int _frameIndex;
        private int _width, _height;
        private readonly object _lock = new();

        public RTDenoiser(int width, int height, RTDenoiserConfig? config = null)
        {
            _width = width;
            _height = height;
            _config = config ?? new RTDenoiserConfig();
            _temporalHistory = new float[width * height * 4];
            _velocityHistory = new float[width * height * 2];
        }

        public void Denoise(float[] inputColor, float[] inputNormal, float[] inputDepth,
                            float[] outputColor, int width, int height, float deltaTime)
        {
            lock (_lock)
            {
                if (_config.Mode == RTDenoiserMode.None)
                {
                    Buffer.BlockCopy(inputColor, 0, outputColor, 0, Math.Min(inputColor.Length, outputColor.Length));
                    return;
                }

                if (_config.Mode == RTDenoiserMode.Temporal || _config.Mode == RTDenoiserMode.Joint)
                    ApplyTemporalFilter(inputColor, inputDepth, outputColor, width, height, deltaTime);

                if (_config.Mode == RTDenoiserMode.Spatial || _config.Mode == RTDenoiserMode.Joint)
                    ApplySpatialFilter(outputColor, inputNormal, inputDepth, width, height);

                if (_config.EnableEdgeStopping)
                    ApplyEdgeStoppingFilter(outputColor, inputNormal, inputDepth, width, height);

                _frameIndex++;
            }
        }

        private void ApplyTemporalFilter(float[] input, float[] depth, float[] output, int w, int h, float dt)
        {
            float alpha = MathF.Min(_config.Alpha, 1.0f - MathF.Exp(-dt * 3.0f));
            int pixelCount = w * h;

            for (int i = 0; i < pixelCount; i++)
            {
                int idx = i * 4;
                int didx = i;
                float d = depth[didx];

                float depthDiff = MathF.Abs(d - _temporalHistory[didx * 4 + 3]);
                float motionFactor = MathF.Exp(-depthDiff * _config.ZThreshold * 100.0f);
                float adaptiveAlpha = alpha * motionFactor;

                for (int c = 0; c < 3; c++)
                {
                    float history = _temporalHistory[idx + c];
                    output[idx + c] = float.Lerp(input[idx + c], history, adaptiveAlpha);
                }
                output[idx + 3] = input[idx + 3];

                _temporalHistory[idx] = output[idx];
                _temporalHistory[idx + 1] = output[idx + 1];
                _temporalHistory[idx + 2] = output[idx + 2];
                _temporalHistory[idx + 3] = d;
            }
        }

        private void ApplySpatialFilter(float[] color, float[] normal, float[] depth, int w, int h)
        {
            int radius = _config.SpatialRadius;
            float[] temp = new float[color.Length];
            Buffer.BlockCopy(color, 0, temp, 0, color.Length * sizeof(float));

            for (int y = radius; y < h - radius; y++)
            {
                for (int x = radius; x < w - radius; x++)
                {
                    int idx = (y * w + x) * 4;
                    Vector3 sum = new(temp[idx], temp[idx + 1], temp[idx + 2]);
                    float weightSum = 1.0f;

                    Vector3 centerNormal = new(normal[idx], normal[idx + 1], normal[idx + 2]);
                    float centerDepth = depth[y * w + x];

                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            if (kx == 0 && ky == 0) continue;
                            int sx = x + kx;
                            int sy = y + ky;
                            int sidx = (sy * w + sx) * 4;

                            float spatialWeight = 1.0f / (1.0f + (kx * kx + ky * ky));

                            Vector3 sNormal = new(normal[sidx], normal[sidx + 1], normal[sidx + 2]);
                            float normalDot = Vector3.Dot(centerNormal, sNormal);
                            float normalWeight = MathF.Pow(MathF.Max(0, normalDot), 16.0f);

                            float sDepth = depth[sy * w + sx];
                            float depthDiff = MathF.Abs(centerDepth - sDepth) / MathF.Max(centerDepth, 0.001f);
                            float depthWeight = MathF.Exp(-depthDiff * _config.ZThreshold * 100.0f);

                            float weight = spatialWeight * normalWeight * depthWeight;
                            sum += new Vector3(temp[sidx], temp[sidx + 1], temp[sidx + 2]) * weight;
                            weightSum += weight;
                        }
                    }

                    color[idx] = sum.X / weightSum;
                    color[idx + 1] = sum.Y / weightSum;
                    color[idx + 2] = sum.Z / weightSum;
                }
            }
        }

        private void ApplyEdgeStoppingFilter(float[] color, float[] normal, float[] depth, int w, int h)
        {
            for (int y = 1; y < h - 1; y++)
            {
                for (int x = 1; x < w - 1; x++)
                {
                    int idx = (y * w + x) * 4;
                    Vector3 center = new(color[idx], color[idx + 1], color[idx + 2]);
                    float cd = depth[y * w + x];
                    Vector3 cn = new(normal[idx], normal[idx + 1], normal[idx + 2]);

                    float maxDiff = 0;
                    int dirs = 0;
                    for (int d = 0; d < 4; d++)
                    {
                        int nx = x + (d == 0 ? 1 : d == 2 ? -1 : 0);
                        int ny = y + (d == 1 ? 1 : d == 3 ? -1 : 0);
                        float nd = depth[ny * w + nx];
                        Vector3 nn = new(normal[(ny * w + nx) * 4], normal[(ny * w + nx) * 4 + 1], normal[(ny * w + nx) * 4 + 2]);
                        float dd = MathF.Abs(cd - nd);
                        float dn = 1.0f - Vector3.Dot(cn, nn);
                        maxDiff = MathF.Max(maxDiff, dd + dn * 0.5f);
                        if (dd < 0.01f && dn < 0.1f) dirs++;
                    }

                    if (dirs < 2)
                    {
                        float luma = 0.2126f * center.X + 0.7152f * center.Y + 0.0722f * center.Z;
                        float edge = MathF.Min(maxDiff * 2.0f, 1.0f);
                        color[idx] = float.Lerp(center.X, luma, edge * 0.3f);
                        color[idx + 1] = float.Lerp(center.Y, luma, edge * 0.3f);
                        color[idx + 2] = float.Lerp(center.Z, luma, edge * 0.3f);
                    }
                }
            }
        }

        public void Resize(int width, int height)
        {
            _width = width;
            _height = height;
            _temporalHistory = new float[width * height * 4];
            _velocityHistory = new float[width * height * 2];
        }

        public void ResetHistory()
        {
            Array.Clear(_temporalHistory);
            Array.Clear(_velocityHistory);
            _frameIndex = 0;
        }

        public void Dispose()
        {
            _temporalHistory = null;
            _velocityHistory = null;
        }
    }

    public class RayTracingPipeline : IDisposable
    {
        private VulkanRhiDevice _rhi;
        private RTAccelerationStructure _bottomLevelAS;
        private RTAccelerationStructure _topLevelAS;
        private RShaderBindingTable _shaderBindingTable;
        private RTDenoiser _denoiser;
        private RTPipelineConfig _config;
        private bool _disposed;
        private bool _rtSupported;

        private VulkanBuffer _scratchBuffer;
        private VulkanBuffer _instanceBuffer;
        private VulkanBuffer _sbtBuffer;

        private List<RTGeometryDesc> _geometries = new();
        private List<RTInstanceDesc> _instances = new();
        private Matrix4x4[] _instanceTransforms = new Matrix4x4[0];

        public bool IsSupported => _rtSupported;
        public RTAccelerationStructure BottomLevelAS => _bottomLevelAS;
        public RTAccelerationStructure TopLevelAS => _topLevelAS;
        public RShaderBindingTable ShaderBindingTable => _shaderBindingTable;
        public RTDenoiser Denoiser => _denoiser;

        public RayTracingPipeline(VulkanRhiDevice rhi, RTPipelineConfig? config = null)
        {
            _rhi = rhi;
            _config = config ?? new RTPipelineConfig();
            CheckRTSupport();
        }

        private void CheckRTSupport()
        {
            try
            {
                var caps = _rhi.QueryCapabilities();
                var extensions = caps.SupportedExtensions;
                if (extensions != null)
                {
                    _rtSupported = Array.Exists(extensions, e => e == "VK_KHR_ray_tracing_pipeline" || e == "VK_KHR_acceleration_structure");
                }
            }
            catch
            {
                _rtSupported = false;
            }
        }

        public void AddGeometry(RTGeometryDesc desc)
        {
            _geometries.Add(desc);
        }

        public void AddInstance(RTInstanceDesc desc)
        {
            _instances.Add(desc);
        }

        public void BuildBottomLevelAS()
        {
            if (!_rtSupported || _geometries.Count == 0) return;

            _bottomLevelAS = new RTAccelerationStructure
            {
                Type = RTAccelerationStructureType.BottomLevel
            };

            ulong totalGeometrySize = 0;
            foreach (var geo in _geometries)
            {
                if (geo.GeometryType == RTGeometryType.Triangles)
                {
                    totalGeometrySize += (ulong)(geo.VertexCount * geo.VertexStride);
                    totalGeometrySize += (ulong)(geo.IndexCount * (geo.IndexFormat == VulkanFormat.R32Uint ? 4 : 2));
                }
                else
                {
                    totalGeometrySize += 32;
                }
            }

            _bottomLevelAS.Buffer = _rhi.CreateBuffer(new BufferDescription
            {
                Size = Math.Max(totalGeometrySize, 256),
                Usage = BufferUsageFlag.TransferDst | BufferUsageFlag.StorageBuffer,
                MemoryProperties = MemoryPropertyFlag.DeviceLocal
            });
        }

        public void BuildTopLevelAS()
        {
            if (!_rtSupported || _bottomLevelAS == null || _instances.Count == 0) return;

            _topLevelAS = new RTAccelerationStructure
            {
                Type = RTAccelerationStructureType.TopLevel
            };

            int instanceSize = 64;
            ulong instanceBufferSize = (ulong)(_instances.Count * instanceSize);

            _instanceBuffer = _rhi.CreateBuffer(new BufferDescription
            {
                Size = instanceBufferSize,
                Usage = BufferUsageFlag.TransferDst | BufferUsageFlag.AccelerationStructureBuildInputReadOnly,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });

            _topLevelAS.Buffer = _rhi.CreateBuffer(new BufferDescription
            {
                Size = Math.Max(instanceBufferSize * 4, 4096),
                Usage = BufferUsageFlag.TransferDst | BufferUsageFlag.StorageBuffer | BufferUsageFlag.ShaderDeviceAddress,
                MemoryProperties = MemoryPropertyFlag.DeviceLocal
            });

            _instanceTransforms = new Matrix4x4[_instances.Count];
            for (int i = 0; i < _instances.Count; i++)
                _instanceTransforms[i] = _instances[i].Transform;
        }

        public void BuildShaderBindingTable()
        {
            if (!_rtSupported) return;

            uint handleSize = 64;
            uint alignment = 64;

            uint rayGenRegionSize = alignment;
            uint missRegionSize = alignment * 2;
            uint hitGroupRegionSize = alignment * Math.Max((uint)_geometries.Count, 1);
            uint totalSize = rayGenRegionSize + missRegionSize + hitGroupRegionSize;

            _sbtBuffer = _rhi.CreateBuffer(new BufferDescription
            {
                Size = totalSize,
                Usage = BufferUsageFlag.TransferDst | BufferUsageFlag.ShaderBindingTableKHR,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });

            _shaderBindingTable = new RShaderBindingTable
            {
                Buffer = _sbtBuffer,
                RayGenOffset = 0,
                MissOffset = rayGenRegionSize,
                HitGroupOffset = rayGenRegionSize + missRegionSize,
                RayGenStride = handleSize,
                MissStride = handleSize,
                HitGroupStride = handleSize
            };
        }

        public void CreateDenoiser(int width, int height, RTDenoiserConfig? config = null)
        {
            _denoiser = new RTDenoiser(width, height, config);
        }

        public void TraceRays(float[] colorBuffer, float[] normalBuffer, float[] depthBuffer,
                             float[] velocityBuffer, int width, int height,
                             Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
                             float[] lightPosition, float[] lightColor, float[] lightIntensity,
                             int lightCount, int maxBounces = 3, int samplesPerPixel = 1)
        {
            if (!_rtSupported || colorBuffer == null) return;

            int pixelCount = width * height;
            float[] irradiance = new float[pixelCount * 4];

            Random rng = new(Environment.TickCount);

            for (int py = 0; py < height; py++)
            {
                for (int px = 0; px < width; px++)
                {
                    int pixelIdx = py * width + px;
                    int idx = pixelIdx * 4;

                    Vector3 finalColor = Vector3.Zero;

                    for (int s = 0; s < samplesPerPixel; s++)
                    {
                        float u = (px + (samplesPerPixel > 1 ? (float)rng.NextDouble() : 0.5f)) / width;
                        float v = (py + (samplesPerPixel > 1 ? (float)rng.NextDouble() : 0.5f)) / height;

                        Vector3 rayDir = ReconstructRayDirection(u, v, view, proj, width, height);
                        Vector3 rayOrigin = cameraPos;

                        Vector3 throughput = Vector3.One;
                        Vector3 radiance = Vector3.Zero;

                        for (int bounce = 0; bounce < maxBounces; bounce++)
                        {
                            float t;
                            Vector3 hitNormal;
                            Vector3 hitAlbedo;
                            float hitRoughness;
                            float hitMetallic;

                            if (!IntersectScene(rayOrigin, rayDir, out t, out hitNormal, out hitAlbedo, out hitRoughness, out hitMetallic, depthBuffer, normalBuffer, colorBuffer, width, height, px, py))
                                break;

                            Vector3 hitPos = rayOrigin + rayDir * t;

                            for (int l = 0; l < lightCount; l++)
                            {
                                Vector3 lPos = new Vector3(lightPosition[l * 3], lightPosition[l * 3 + 1], lightPosition[l * 3 + 2]);
                                Vector3 lCol = new Vector3(lightColor[l * 3], lightColor[l * 3 + 1], lightColor[l * 3 + 2]);
                                float lInt = lightIntensity[l];

                                Vector3 toLight = lPos - hitPos;
                                float lightDist = toLight.Length();
                                toLight /= lightDist;

                                float ndotl = MathF.Max(0, Vector3.Dot(hitNormal, toLight));
                                if (ndotl <= 0) continue;

                                float attenuation = lInt / (1.0f + lightDist * lightDist * 0.01f);
                                Vector3 brdf = EvaluatePBR(hitAlbedo, hitRoughness, hitMetallic, hitNormal, -rayDir, toLight);
                                radiance += throughput * lCol * brdf * ndotl * attenuation;
                            }

                            Vector3 diffuse = hitAlbedo / MathF.PI;
                            throughput *= diffuse;

                            Vector3 reflDir = Reflect(rayDir, hitNormal);
                            float pdf = MathF.Max(0, Vector3.Dot(hitNormal, reflDir));
                            if (pdf > 0 && hitMetallic > 0.1f)
                            {
                                rayDir = Vector3.Lerp(reflDir, CosineWeightedHemisphere(hitNormal, rng), hitRoughness);
                            }
                            else
                            {
                                rayDir = CosineWeightedHemisphere(hitNormal, rng);
                            }
                            rayDir = Vector3.Normalize(rayDir);
                            rayOrigin = hitPos + hitNormal * 0.001f;
                        }

                        finalColor += radiance;
                    }

                    finalColor /= samplesPerPixel;

                    irradiance[idx] = finalColor.X;
                    irradiance[idx + 1] = finalColor.Y;
                    irradiance[idx + 2] = finalColor.Z;
                    irradiance[idx + 3] = 1.0f;
                }
            }

            if (_denoiser != null)
                _denoiser.Denoise(irradiance, normalBuffer, depthBuffer, colorBuffer, width, height, 0.016f);
            else
                Buffer.BlockCopy(irradiance, 0, colorBuffer, 0, Math.Min(irradiance.Length, colorBuffer.Length));
        }

        private bool IntersectScene(Vector3 origin, Vector3 dir, out float t, out Vector3 normal, out Vector3 albedo, out float roughness, out float metallic, float[] depthBuffer, float[] normalBuffer, float[] colorBuffer, int w, int h, int px, int py, float maxDist = 1000.0f)
        {
            t = 0;
            normal = Vector3.UnitY;
            albedo = new Vector3(0.5f);
            roughness = 0.5f;
            metallic = 0.0f;

            if (px < 0 || px >= w || py < 0 || py >= h) return false;

            int idx = py * w + px;
            float sceneDepth = depthBuffer[idx];

            if (sceneDepth <= 0 || sceneDepth >= 1.0f) return false;

            float linearDepth = 0.1f * 100.0f / (100.0f - sceneDepth * (100.0f - 0.1f));
            t = linearDepth;

            int nidx = idx * 4;
            normal = Vector3.Normalize(new Vector3(normalBuffer[nidx], normalBuffer[nidx + 1], normalBuffer[nidx + 2]));
            albedo = new Vector3(colorBuffer[nidx], colorBuffer[nidx + 1], colorBuffer[nidx + 2]);

            return true;
        }

        private Vector3 ReconstructRayDirection(float u, float v, Matrix4x4 view, Matrix4x4 proj, int w, int h)
        {
            Vector4 clipPos = new Vector4(u * 2.0f - 1.0f, v * 2.0f - 1.0f, 1.0f, 1.0f);
            var invProj = Matrix4x4.Identity;
            if (Matrix4x4.Invert(proj, out var inv))
                invProj = inv;
            var invView = Matrix4x4.Identity;
            if (Matrix4x4.Invert(view, out var invV))
                invView = invV;

            Vector4 viewPos = Vector4.Transform(clipPos, invProj);
            viewPos.W = 0;
            Vector4 worldDir = Vector4.Transform(viewPos, invView);
            return Vector3.Normalize(new Vector3(worldDir.X, worldDir.Y, worldDir.Z));
        }

        private Vector3 EvaluatePBR(Vector3 albedo, float roughness, float metallic, Vector3 N, Vector3 V, Vector3 L)
        {
            Vector3 H = Vector3.Normalize(V + L);
            float NdotH = MathF.Max(0, Vector3.Dot(N, H));
            float NdotL = MathF.Max(0, Vector3.Dot(N, L));
            float NdotV = MathF.Max(0, Vector3.Dot(N, V));
            float HdotV = MathF.Max(0, Vector3.Dot(H, V));

            float a = roughness * roughness;
            float a2 = a * a;
            float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
            float D = a2 / (MathF.PI * denom * denom);

            Vector3 F0 = Vector3.Lerp(new Vector3(0.04f), albedo, metallic);
            Vector3 F = F0 + (Vector3.One - F0) * MathF.Pow(1.0f - HdotV, 5.0f);

            float k = (roughness + 1.0f);
            k = k * k / 8.0f;
            float G1L = NdotL / (NdotL * (1.0f - k) + k);
            float G1V = NdotV / (NdotV * (1.0f - k) + k);
            float G = G1L * G1V;

            Vector3 kD = (Vector3.One - F) * (1.0f - metallic);
            Vector3 diffuse = kD * albedo / MathF.PI;
            Vector3 specular = (D * F * G) / MathF.Max(4.0f * NdotL * NdotV, 0.001f);

            return diffuse + specular;
        }

        private Vector3 Reflect(Vector3 I, Vector3 N) => I - 2.0f * Vector3.Dot(N, I) * N;

        private Vector3 CosineWeightedHemisphere(Vector3 normal, Random rng)
        {
            float r1 = (float)rng.NextDouble();
            float r2 = (float)rng.NextDouble();
            float phi = 2.0f * MathF.PI * r1;
            float cosTheta = MathF.Sqrt(r2);
            float sinTheta = MathF.Sqrt(1.0f - r2);

            Vector3 tangent = MathF.Abs(normal.Y) < 0.99f
                ? Vector3.Normalize(Vector3.Cross(normal, Vector3.UnitY))
                : Vector3.Normalize(Vector3.Cross(normal, Vector3.UnitX));
            Vector3 bitangent = Vector3.Cross(normal, tangent);

            return Vector3.Normalize(tangent * MathF.Cos(phi) * sinTheta + bitangent * MathF.Sin(phi) * sinTheta + normal * cosTheta);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _bottomLevelAS?.Dispose();
            _topLevelAS?.Dispose();
            _shaderBindingTable?.Dispose();
            _denoiser?.Dispose();
            _scratchBuffer?.Dispose();
            _instanceBuffer?.Dispose();
            _sbtBuffer?.Dispose();
        }
    }
}
