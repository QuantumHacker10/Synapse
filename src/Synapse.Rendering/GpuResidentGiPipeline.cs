using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using GDNN.Lighting.LDNN;
using GDNN.Rendering.Engine;

namespace GDNN.Rendering.Bridge
{
    /// <summary>Which industrial GI path produced the last irradiance field.</summary>
    public enum GiComputePath : byte
    {
        /// <summary>No GI yet.</summary>
        None = 0,
        /// <summary>Fresh Vulkan G-buffer readback → L-DNN.</summary>
        GpuReadback = 1,
        /// <summary>Resident G-buffer (previous GPU frame) → compute SSGI, no new readback.</summary>
        GpuResidentCompute = 2,
        /// <summary>Constant-filled G-buffer fallback (no GPU geometry yet).</summary>
        ConstantFallback = 3
    }

    /// <summary>
    /// Keeps a GPU-origin G-buffer resident on the CPU staging side and runs
    /// screen-space GI compute without requiring a fresh Vulkan readback every frame.
    /// </summary>
    public sealed class GpuResidentGiPipeline
    {
        private GBuffer? _resident;
        private int _width;
        private int _height;
        private int _version;
        private Vector3[,]? _lastIrradiance;

        public bool HasResidentGBuffer => _resident?.Depth != null && _resident.Depth.Length == _width * _height;
        public int ResidentVersion => _version;
        public GiComputePath LastPath { get; private set; } = GiComputePath.None;
        public long ResidentComputeFrames { get; private set; }
        public long ReadbackFrames { get; private set; }

        /// <summary>Stores a GPU readback snapshot as the resident G-buffer.</summary>
        public void UpdateFromSnapshot(GBufferSnapshot snapshot)
        {
            ArgumentNullException.ThrowIfNull(snapshot);
            _width = snapshot.Width;
            _height = snapshot.Height;
            EnsureResident();
            Array.Copy(snapshot.Depth, _resident!.Depth, snapshot.Depth.Length);
            Array.Copy(snapshot.Normals, _resident.Normals, snapshot.Normals.Length);
            Array.Copy(snapshot.Albedo, _resident.Albedo, snapshot.Albedo.Length);
            Array.Copy(snapshot.Velocity, _resident.Velocity, snapshot.Velocity.Length);
            Array.Copy(snapshot.MaterialProps, _resident.MaterialProps, snapshot.MaterialProps.Length);
            Array.Copy(snapshot.Specular, _resident.Specular, snapshot.Specular.Length);
            Array.Copy(snapshot.Emissive, _resident.Emissive, snapshot.Emissive.Length);
            _version++;
            LastPath = GiComputePath.GpuReadback;
            ReadbackFrames++;
        }

        /// <summary>Copies an already-filled bridge G-buffer into the resident store.</summary>
        public void UpdateFromGBuffer(GBuffer gbuffer)
        {
            ArgumentNullException.ThrowIfNull(gbuffer);
            if (gbuffer.Depth == null)
                return;
            _width = gbuffer.Width;
            _height = gbuffer.Height;
            EnsureResident();
            int n = _width * _height;
            Array.Copy(gbuffer.Depth, _resident!.Depth, n);
            Array.Copy(gbuffer.Normals, _resident.Normals, n);
            Array.Copy(gbuffer.Albedo, _resident.Albedo, n);
            if (gbuffer.Velocity != null)
                Array.Copy(gbuffer.Velocity, _resident.Velocity, n);
            if (gbuffer.MaterialProps != null)
                Array.Copy(gbuffer.MaterialProps, _resident.MaterialProps, n);
            if (gbuffer.Specular != null)
                Array.Copy(gbuffer.Specular, _resident.Specular, n);
            if (gbuffer.Emissive != null)
                Array.Copy(gbuffer.Emissive, _resident.Emissive, n);
            _version++;
        }

        /// <summary>
        /// Runs industrial screen-space irradiance compute on the resident G-buffer
        /// (no Vulkan readback). Used when the previous GPU frame is still valid.
        /// </summary>
        public Vector3[,] ComputeResidentIrradiance(
            LDNNRenderer renderer,
            CameraState camera,
            IReadOnlyList<LightConfig> lights)
        {
            ArgumentNullException.ThrowIfNull(renderer);
            ArgumentNullException.ThrowIfNull(camera);
            if (!HasResidentGBuffer)
                return new Vector3[Math.Max(1, _width), Math.Max(1, _height)];

            var irradiance = new Vector3[_width, _height];
            var flat = new Vector3[_width * _height];

            renderer.DispatchComputeShaders(
                "ssgi_irradiance",
                (_width + 7) / 8,
                (_height + 7) / 8,
                1,
                new Dictionary<string, object>
                {
                    ["gbuffer"] = _resident!,
                    ["camera"] = camera,
                    ["lights"] = lights ?? Array.Empty<LightConfig>(),
                    ["dest"] = flat,
                    ["width"] = _width,
                    ["height"] = _height
                });

            for (int y = 0; y < _height; y++)
                for (int x = 0; x < _width; x++)
                    irradiance[x, y] = flat[y * _width + x];

            _lastIrradiance = irradiance;
            LastPath = GiComputePath.GpuResidentCompute;
            ResidentComputeFrames++;
            return irradiance;
        }

        public Vector3[,]? LastIrradiance => _lastIrradiance;

        public void MarkConstantFallback() => LastPath = GiComputePath.ConstantFallback;

        public void CopyResidentTo(GBuffer target)
        {
            if (!HasResidentGBuffer || target == null)
                return;
            int n = _width * _height;
            if (target.Depth == null || target.Depth.Length != n)
                return;
            Array.Copy(_resident!.Depth, target.Depth, n);
            Array.Copy(_resident.Normals, target.Normals, n);
            Array.Copy(_resident.Albedo, target.Albedo, n);
            Array.Copy(_resident.Velocity, target.Velocity, n);
            Array.Copy(_resident.MaterialProps, target.MaterialProps, n);
            Array.Copy(_resident.Specular, target.Specular, n);
            Array.Copy(_resident.Emissive, target.Emissive, n);
            target.Width = _width;
            target.Height = _height;
        }

        private void EnsureResident()
        {
            int n = Math.Max(1, _width * _height);
            if (_resident != null && _resident.Depth?.Length == n)
                return;
            _resident = new GBuffer
            {
                Width = _width,
                Height = _height,
                Depth = new float[n],
                Normals = new Vector3[n],
                Albedo = new Vector3[n],
                Velocity = new Vector2[n],
                MaterialProps = new Vector4[n],
                Specular = new Vector3[n],
                Emissive = new Vector3[n]
            };
        }
    }
}
