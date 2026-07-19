// L-DNN neural global illumination subsystem (split from LDNNRenderer.cs).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Lighting.LDNN
{

    /// <summary>
    /// Manages the irradiance cache with various placement strategies and probe management.
    /// </summary>
    public class IrradianceCacheManager
    {
        private IrradianceCacheType _cacheType;
        private List<IrradianceProbe> _probes;
        private int _maxProbes;
        private float _probeSpacing;
        private int _frameIndex;
        private bool _isInitialized;
        private ProbeUpdateMode _updateMode;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>Number of active probes.</summary>
        public int ActiveProbeCount => _probes?.Count(p => p.IsValid) ?? 0;
        /// <summary>Total probe capacity.</summary>
        public int MaxProbes => _maxProbes;
        /// <summary>Current frame index.</summary>
        public int FrameIndex => _frameIndex;

        /// <summary>
        /// Initializes the irradiance cache manager.
        /// </summary>
        public void Initialize(IrradianceCacheType cacheType, int maxProbes, float probeSpacing,
            ProbeUpdateMode updateMode)
        {
            _cacheType = cacheType;
            _maxProbes = maxProbes;
            _probeSpacing = probeSpacing;
            _updateMode = updateMode;
            _probes = new List<IrradianceProbe>(maxProbes);
            _frameIndex = 0;
            _isInitialized = true;
        }

        /// <summary>
        /// Places probes using octahedral mapping.
        /// </summary>
        public void PlaceOctahedralProbes(Vector3 center, float radius, int probesPerAxis)
        {
            float step = radius * 2.0f / probesPerAxis;
            for (int x = 0; x < probesPerAxis; x++)
            {
                for (int y = 0; y < probesPerAxis; y++)
                {
                    for (int z = 0; z < probesPerAxis; z++)
                    {
                        Vector3 pos = center + new Vector3(
                            (x - probesPerAxis / 2.0f) * step,
                            (y - probesPerAxis / 2.0f) * step,
                            (z - probesPerAxis / 2.0f) * step);

                        if (_probes.Count >= _maxProbes)
                            return;

                        _probes.Add(new IrradianceProbe
                        {
                            Position = pos,
                            SHCoefficients = new Vector4[9],
                            IsValid = false,
                            LastUpdateFrame = -1,
                            Importance = 1.0f,
                            ReferenceDepth = 0,
                            ReferenceNormal = Vector3.UnitY,
                            Variance = 0,
                            SampleCount = 0
                        });
                    }
                }
            }
        }

        /// <summary>
        /// Places probes using spherical harmonics layout.
        /// </summary>
        public void PlaceSphericalHarmonicsProbes(Vector3 center, float radius, int numRings)
        {
            for (int ring = 0; ring < numRings; ring++)
            {
                float phi = PI * (ring + 0.5f) / numRings;
                int probesInRing = (int)(numRings * MathF.Sin(phi) * 4);
                probesInRing = Math.Max(1, probesInRing);

                for (int i = 0; i < probesInRing; i++)
                {
                    float theta = TWO_PI * i / probesInRing;
                    Vector3 pos = center + new Vector3(
                        radius * MathF.Sin(phi) * MathF.Cos(theta),
                        radius * MathF.Cos(phi),
                        radius * MathF.Sin(phi) * MathF.Sin(theta));

                    if (_probes.Count >= _maxProbes)
                        return;

                    _probes.Add(new IrradianceProbe
                    {
                        Position = pos,
                        SHCoefficients = new Vector4[9],
                        IsValid = false,
                        LastUpdateFrame = -1,
                        Importance = 1.0f,
                        ReferenceDepth = 0,
                        ReferenceNormal = Vector3.UnitY,
                        Variance = 0,
                        SampleCount = 0
                    });
                }
            }
        }

        /// <summary>
        /// Places probes in a regular radiance grid.
        /// </summary>
        public void PlaceRadianceGrid(Vector3 origin, Vector3Int gridSize, float cellSize)
        {
            for (int x = 0; x < gridSize.X; x++)
            {
                for (int y = 0; y < gridSize.Y; y++)
                {
                    for (int z = 0; z < gridSize.Z; z++)
                    {
                        Vector3 pos = origin + new Vector3(x, y, z) * cellSize;

                        if (_probes.Count >= _maxProbes)
                            return;

                        _probes.Add(new IrradianceProbe
                        {
                            Position = pos,
                            SHCoefficients = new Vector4[9],
                            IsValid = false,
                            LastUpdateFrame = -1,
                            Importance = 1.0f,
                            ReferenceDepth = 0,
                            ReferenceNormal = Vector3.UnitY,
                            Variance = 0,
                            SampleCount = 0
                        });
                    }
                }
            }
        }

        /// <summary>
        /// Interpolates irradiance from nearby probes using trilinear interpolation.
        /// </summary>
        public Vector3 TrilinearInterpolate(Vector3 worldPos, Vector3 normal)
        {
            if (_probes == null || _probes.Count == 0)
                return Vector3.Zero;

            float totalWeight = 0;
            Vector3 totalIrradiance = Vector3.Zero;

            foreach (var probe in _probes)
            {
                if (!probe.IsValid)
                    continue;
                float dist = (probe.Position - worldPos).Length();
                float weight = MathF.Exp(-dist / _probeSpacing);
                if (weight < 0.001f)
                    continue;

                Vector3 irradiance = ComputeProbeIrradiance(probe, normal);
                totalIrradiance += irradiance * weight;
                totalWeight += weight;
            }

            return totalWeight > 0 ? totalIrradiance / totalWeight : Vector3.Zero;
        }

        /// <summary>
        /// Interpolates irradiance using tetrahedral interpolation.
        /// </summary>
        public Vector3 TetrahedralInterpolate(Vector3 worldPos, Vector3 normal)
        {
            var nearest = FindNearestProbes(worldPos, 4);
            if (nearest.Count < 4)
                return TrilinearInterpolate(worldPos, normal);

            Vector3 p0 = nearest[0].Position;
            Vector3 p1 = nearest[1].Position;
            Vector3 p2 = nearest[2].Position;
            Vector3 p3 = nearest[3].Position;

            Vector4 barycentrics = ComputeBarycentricCoordinates(worldPos, p0, p1, p2, p3);

            Vector3 irr0 = ComputeProbeIrradiance(nearest[0], normal);
            Vector3 irr1 = ComputeProbeIrradiance(nearest[1], normal);
            Vector3 irr2 = ComputeProbeIrradiance(nearest[2], normal);
            Vector3 irr3 = ComputeProbeIrradiance(nearest[3], normal);

            return irr0 * barycentrics.X + irr1 * barycentrics.Y +
                   irr2 * barycentrics.Z + irr3 * barycentrics.W;
        }

        /// <summary>
        /// Schedules probe updates based on importance.
        /// </summary>
        public void ScheduleUpdates(CameraState camera, float timeBudget)
        {
            _frameIndex++;

            for (int probeIndex = 0; probeIndex < _probes.Count; probeIndex++)
            {
                var probe = _probes[probeIndex];
                float dist = (probe.Position - camera.Position).Length();
                float importance = 1.0f / (1.0f + dist * 0.01f);

                bool needsUpdate = false;
                switch (_updateMode)
                {
                    case ProbeUpdateMode.OnDemand:
                        needsUpdate = !probe.IsValid;
                        break;
                    case ProbeUpdateMode.Periodic:
                        needsUpdate = (_frameIndex - probe.LastUpdateFrame) > 60;
                        break;
                    case ProbeUpdateMode.DistanceBased:
                        needsUpdate = dist < 50.0f && (_frameIndex - probe.LastUpdateFrame) > (int)(dist * 0.5f);
                        break;
                    case ProbeUpdateMode.ImportanceDriven:
                        needsUpdate = importance > 0.5f && (_frameIndex - probe.LastUpdateFrame) > 10;
                        break;
                    case ProbeUpdateMode.BudgetLimited:
                        needsUpdate = importance > 0.3f;
                        break;
                }

                if (needsUpdate)
                {
                    _probes[probeIndex] = probe with
                    {
                        Importance = importance,
                        LastUpdateFrame = _frameIndex
                    };
                }
            }
        }

        /// <summary>
        /// Manages probe budget by removing least important probes.
        /// </summary>
        public void ManageBudget(int targetCount)
        {
            if (_probes.Count <= targetCount)
                return;

            var sorted = _probes.OrderByDescending(p => p.Importance).Take(targetCount).ToList();
            _probes = sorted;
        }

        /// <summary>
        /// Tracks cache coherence across frames.
        /// </summary>
        public float TrackCacheCoherence(CameraState camera)
        {
            if (_probes.Count == 0)
                return 0;

            int reusedCount = 0;
            foreach (var probe in _probes)
            {
                if (probe.IsValid && (_frameIndex - probe.LastUpdateFrame) < 5)
                    reusedCount++;
            }

            return (float)reusedCount / _probes.Count;
        }

        /// <summary>
        /// Checks probe validity by comparing depth with current G-Buffer.
        /// </summary>
        public void CheckProbeValidity(GBuffer gbuffer, CameraState camera)
        {
            for (int i = 0; i < _probes.Count; i++)
            {
                var probe = _probes[i];
                Vector3 screenPos = camera.ProjectToScreen(probe.Position);

                if (screenPos.X < 0 || screenPos.X >= 1 || screenPos.Y < 0 || screenPos.Y >= 1)
                {
                    _probes[i] = probe with { IsValid = false };
                    continue;
                }

                int pixX = (int)(screenPos.X * gbuffer.Width);
                int pixY = (int)(screenPos.Y * gbuffer.Height);
                pixX = Math.Clamp(pixX, 0, gbuffer.Width - 1);
                pixY = Math.Clamp(pixY, 0, gbuffer.Height - 1);

                float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                float probeDepth = screenPos.Z;

                float depthError = MathF.Abs(sceneDepth - probeDepth) / MathF.Max(0.001f, sceneDepth);
                if (depthError > 0.1f)
                    _probes[i] = probe with { IsValid = false };
            }
        }

        /// <summary>
        /// Compresses probe data using SH compression.
        /// </summary>
        public byte[] CompressProbes()
        {
            using var ms = new System.IO.MemoryStream();
            using var bw = new System.IO.BinaryWriter(ms);

            bw.Write(_probes.Count);
            foreach (var probe in _probes)
            {
                bw.Write(probe.Position.X);
                bw.Write(probe.Position.Y);
                bw.Write(probe.Position.Z);
                bw.Write(probe.IsValid);
                bw.Write(probe.Importance);

                for (int i = 0; i < 9; i++)
                {
                    bw.Write(probe.SHCoefficients[i].X);
                    bw.Write(probe.SHCoefficients[i].Y);
                    bw.Write(probe.SHCoefficients[i].Z);
                    bw.Write(probe.SHCoefficients[i].W);
                }
            }

            return ms.ToArray();
        }

        /// <summary>
        /// Decompresses probe data from a byte array.
        /// </summary>
        public void DecompressProbes(byte[] data)
        {
            using var ms = new System.IO.MemoryStream(data);
            using var br = new System.IO.BinaryReader(ms);

            int count = br.ReadInt32();
            _probes.Clear();

            for (int i = 0; i < count; i++)
            {
                float px = br.ReadSingle();
                float py = br.ReadSingle();
                float pz = br.ReadSingle();
                bool valid = br.ReadBoolean();
                float importance = br.ReadSingle();

                var shCoeffs = new Vector4[9];
                for (int j = 0; j < 9; j++)
                    shCoeffs[j] = new Vector4(br.ReadSingle(), br.ReadSingle(), br.ReadSingle(), br.ReadSingle());

                _probes.Add(new IrradianceProbe
                {
                    Position = new Vector3(px, py, pz),
                    SHCoefficients = shCoeffs,
                    IsValid = valid,
                    Importance = importance
                });
            }
        }

        /// <summary>
        /// Streams probes to disk for persistence.
        /// </summary>
        public void StreamToDisk(string filePath)
        {
            byte[] data = CompressProbes();
            System.IO.File.WriteAllBytes(filePath, data);
        }

        /// <summary>
        /// Loads probes from disk.
        /// </summary>
        public void StreamFromDisk(string filePath)
        {
            if (System.IO.File.Exists(filePath))
            {
                byte[] data = System.IO.File.ReadAllBytes(filePath);
                DecompressProbes(data);
            }
        }

        /// <summary>
        /// Gets nearby probes for a given position and normal.
        /// </summary>
        public List<IrradianceProbe> GetNearbyProbes(Vector3 worldPos, Vector3 normal, int maxCount)
        {
            return _probes
                .Where(p => p.IsValid)
                .OrderBy(p => (p.Position - worldPos).LengthSquared())
                .Take(maxCount)
                .ToList();
        }

        private List<IrradianceProbe> FindNearestProbes(Vector3 worldPos, int count)
        {
            return _probes
                .OrderBy(p => (p.Position - worldPos).LengthSquared())
                .Take(count)
                .ToList();
        }

        private Vector3 ComputeProbeIrradiance(IrradianceProbe probe, Vector3 direction)
        {
            if (probe.SHCoefficients == null || probe.SHCoefficients.Length < 9)
                return Vector3.Zero;

            float x = direction.X;
            float y = direction.Y;
            float z = direction.Z;

            Vector3 result = Vector3.Zero;
            result += new Vector3(probe.SHCoefficients[0].X, probe.SHCoefficients[0].Y, probe.SHCoefficients[0].Z) * 0.282095f;
            result += new Vector3(probe.SHCoefficients[1].X, probe.SHCoefficients[1].Y, probe.SHCoefficients[1].Z) * 0.488603f * y;
            result += new Vector3(probe.SHCoefficients[2].X, probe.SHCoefficients[2].Y, probe.SHCoefficients[2].Z) * 0.488603f * z;
            result += new Vector3(probe.SHCoefficients[3].X, probe.SHCoefficients[3].Y, probe.SHCoefficients[3].Z) * 0.488603f * x;
            result += new Vector3(probe.SHCoefficients[4].X, probe.SHCoefficients[4].Y, probe.SHCoefficients[4].Z) * 1.092548f * x * y;
            result += new Vector3(probe.SHCoefficients[5].X, probe.SHCoefficients[5].Y, probe.SHCoefficients[5].Z) * 1.092548f * y * z;
            result += new Vector3(probe.SHCoefficients[6].X, probe.SHCoefficients[6].Y, probe.SHCoefficients[6].Z) * 0.315392f * (3.0f * z * z - 1.0f);
            result += new Vector3(probe.SHCoefficients[7].X, probe.SHCoefficients[7].Y, probe.SHCoefficients[7].Z) * 1.092548f * x * z;
            result += new Vector3(probe.SHCoefficients[8].X, probe.SHCoefficients[8].Y, probe.SHCoefficients[8].Z) * 0.546274f * (x * x - y * y);

            return Vector3.Max(result, Vector3.Zero);
        }

        private Vector4 ComputeBarycentricCoordinates(Vector3 point, Vector3 a, Vector3 b, Vector3 c, Vector3 d)
        {
            Vector3 v0 = b - a, v1 = c - a, v2 = d - a, v3 = point - a;
            float d00 = Vector3.Dot(v0, v0);
            float d01 = Vector3.Dot(v0, v1);
            float d02 = Vector3.Dot(v0, v2);
            float d11 = Vector3.Dot(v1, v1);
            float d12 = Vector3.Dot(v1, v2);
            float d22 = Vector3.Dot(v2, v2);
            float d30 = Vector3.Dot(v3, v0);
            float d31 = Vector3.Dot(v3, v1);
            float d32 = Vector3.Dot(v3, v2);

            float denom = d00 * (d11 * d22 - d12 * d12) - d01 * (d01 * d22 - d12 * d02) + d02 * (d01 * d12 - d11 * d02);
            if (MathF.Abs(denom) < 0.0001f)
                return new Vector4(0.25f, 0.25f, 0.25f, 0.25f);

            float w = (d30 * (d11 * d22 - d12 * d12) - d01 * (d31 * d22 - d12 * d32) + d02 * (d31 * d12 - d11 * d32)) / denom;
            float u = (d00 * (d31 * d22 - d12 * d32) - d30 * (d01 * d22 - d12 * d02) + d02 * (d01 * d32 - d31 * d02)) / denom;
            float v = (d00 * (d11 * d32 - d31 * d12) - d01 * (d01 * d32 - d31 * d02) + d30 * (d01 * d12 - d11 * d02)) / denom;
            float t = 1.0f - u - v - w;

            return new Vector4(t, u, v, w);
        }
    }
}
