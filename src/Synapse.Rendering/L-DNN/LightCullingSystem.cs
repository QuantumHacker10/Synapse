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
    /// Tiled/clustered light culling system for efficient light evaluation.
    /// </summary>
    public class LightCullingSystem
    {
        private const int TILE_SIZE = 16;
        private const int MAX_LIGHTS_PER_TILE = 256;
        private const int CLUSTER_Z_SLICES = 16;

        private List<LightConfig> _allLights = new();
        private List<int>[,] _tileLightLists;
        private List<int>[][][] _clusterLightLists;
        private int _tileCountX;
        private int _tileCountY;
        private int _screenWidth;
        private int _screenHeight;
        private float[] _clusterDepths;
        private bool _isInitialized;

        /// <summary>Number of tiles in the X direction.</summary>
        public int TileCountX => _tileCountX;
        /// <summary>Number of tiles in the Y direction.</summary>
        public int TileCountY => _tileCountY;
        /// <summary>Total number of tiles.</summary>
        public int TotalTiles => _tileCountX * _tileCountY;
        /// <summary>Total number of clusters.</summary>
        public int TotalClusters => _tileCountX * _tileCountY * CLUSTER_Z_SLICES;

        /// <summary>
        /// Initializes the light culling system for a given screen resolution.
        /// </summary>
        public void Initialize(int screenWidth, int screenHeight)
        {
            _screenWidth = screenWidth;
            _screenHeight = screenHeight;
            _tileCountX = (screenWidth + TILE_SIZE - 1) / TILE_SIZE;
            _tileCountY = (screenHeight + TILE_SIZE - 1) / TILE_SIZE;

            _tileLightLists = new List<int>[_tileCountX, _tileCountY];
            for (int x = 0; x < _tileCountX; x++)
                for (int y = 0; y < _tileCountY; y++)
                    _tileLightLists[x, y] = new List<int>(MAX_LIGHTS_PER_TILE);

            _clusterLightLists = new List<int>[_tileCountX][][];
            for (int x = 0; x < _tileCountX; x++)
            {
                _clusterLightLists[x] = new List<int>[_tileCountY][];
                for (int y = 0; y < _tileCountY; y++)
                {
                    _clusterLightLists[x][y] = new List<int>[CLUSTER_Z_SLICES];
                    for (int z = 0; z < CLUSTER_Z_SLICES; z++)
                        _clusterLightLists[x][y][z] = new List<int>(MAX_LIGHTS_PER_TILE);
                }
            }

            _clusterDepths = new float[CLUSTER_Z_SLICES + 1];
            _isInitialized = true;
        }

        /// <summary>
        /// Updates the light list and performs culling for the current frame.
        /// </summary>
        public void CullLights(List<LightConfig> lights, CameraState camera)
        {
            if (!_isInitialized) return;
            _allLights = lights;
            ComputeClusterDepths(camera.NearPlane, camera.FarPlane);

            Parallel.For(0, _tileCountX, tx =>
            {
                for (int ty = 0; ty < _tileCountY; ty++)
                {
                    _tileLightLists[tx, ty].Clear();
                    for (int i = 0; i < lights.Count; i++)
                    {
                        if (IsLightRelevant(lights[i], tx, ty, camera))
                        {
                            _tileLightLists[tx, ty].Add(i);
                        }
                    }
                }
            });

            Parallel.For(0, _tileCountX, tx =>
            {
                for (int ty = 0; ty < _tileCountY; ty++)
                {
                    for (int tz = 0; tz < CLUSTER_Z_SLICES; tz++)
                    {
                        _clusterLightLists[tx][ty][tz].Clear();
                        float zMin = _clusterDepths[tz];
                        float zMax = _clusterDepths[tz + 1];

                        for (int li = 0; li < _tileLightLists[tx, ty].Count; li++)
                        {
                            int lightIdx = _tileLightLists[tx, ty][li];
                            if (IsLightInCluster(lights[lightIdx], tx, ty, zMin, zMax, camera))
                            {
                                _clusterLightLists[tx][ty][tz].Add(lightIdx);
                            }
                        }
                    }
                }
            });
        }

        private void ComputeClusterDepths(float nearPlane, float farPlane)
        {
            for (int i = 0; i <= CLUSTER_Z_SLICES; i++)
            {
                float t = (float)i / CLUSTER_Z_SLICES;
                _clusterDepths[i] = nearPlane * MathF.Pow(farPlane / nearPlane, t);
            }
        }

        private bool IsLightRelevant(LightConfig light, int tx, int ty, CameraState camera)
        {
            if (light.Type == LightType.Directional) return true;
            float range = light.Range;
            if (range <= 0) range = 100.0f;
            Vector3 viewPos = Vector3.Transform(light.Position, camera.ViewMatrix);
            if (viewPos.Z > -camera.NearPlane) return false;
            float tileXMin = (float)(tx * TILE_SIZE) / _screenWidth * 2.0f - 1.0f;
            float tileXMax = (float)((tx + 1) * TILE_SIZE) / _screenWidth * 2.0f - 1.0f;
            float tileYMin = 1.0f - (float)((ty + 1) * TILE_SIZE) / _screenHeight * 2.0f;
            float tileYMax = 1.0f - (float)(ty * TILE_SIZE) / _screenHeight * 2.0f;
            float absZ = MathF.Abs(viewPos.Z);
            if (absZ < 0.001f) return true;
            float projRadius = range / absZ * MathF.Max(_screenWidth, _screenHeight) * 0.5f / MathF.Tan(camera.FieldOfView * 0.5f);
            float screenX = (viewPos.X / absZ * 0.5f + 0.5f) * 2.0f - 1.0f;
            float screenY = 1.0f - (viewPos.Y / absZ * 0.5f + 0.5f) * 2.0f;
            return screenX + projRadius / _screenWidth >= tileXMin &&
                   screenX - projRadius / _screenWidth <= tileXMax &&
                   screenY + projRadius / _screenHeight >= tileYMin &&
                   screenY - projRadius / _screenHeight <= tileYMax;
        }

        private bool IsLightInCluster(LightConfig light, int tx, int ty, float zMin, float zMax, CameraState camera)
        {
            if (light.Type == LightType.Directional) return true;
            float range = light.Range;
            if (range <= 0) range = 100.0f;
            Vector3 viewPos = Vector3.Transform(light.Position, camera.ViewMatrix);
            if (viewPos.Z + range < -zMax || viewPos.Z - range > -zMin) return false;
            return true;
        }

        /// <summary>
        /// Gets the list of light indices affecting a specific tile.
        /// </summary>
        public List<int> GetTileLights(int tileX, int tileY)
        {
            if (tileX < 0 || tileX >= _tileCountX || tileY < 0 || tileY >= _tileCountY)
                return new List<int>();
            return _tileLightLists[tileX, tileY];
        }

        /// <summary>
        /// Gets the list of light indices in a specific cluster.
        /// </summary>
        public List<int> GetClusterLights(int tileX, int tileY, int clusterZ)
        {
            if (tileX < 0 || tileX >= _tileCountX || tileY < 0 || tileY >= _tileCountY ||
                clusterZ < 0 || clusterZ >= CLUSTER_Z_SLICES)
                return new List<int>();
            return _clusterLightLists[tileX][tileY][clusterZ];
        }

        /// <summary>
        /// Gets the depth range for a cluster slice.
        /// </summary>
        public (float Near, float Far) GetClusterDepthRange(int clusterZ)
        {
            if (clusterZ < 0 || clusterZ >= CLUSTER_Z_SLICES) return (0, 0);
            return (_clusterDepths[clusterZ], _clusterDepths[clusterZ + 1]);
        }

        /// <summary>
        /// Builds the light grid data for GPU consumption.
        /// </summary>
        public (int[] LightIndices, int[] TileOffsets, int[] TileCounts) BuildLightGridData()
        {
            var allIndices = new List<int>();
            var offsets = new int[_tileCountX * _tileCountY * CLUSTER_Z_SLICES];
            var counts = new int[_tileCountX * _tileCountY * CLUSTER_Z_SLICES];

            for (int tx = 0; tx < _tileCountX; tx++)
            {
                for (int ty = 0; ty < _tileCountY; ty++)
                {
                    for (int tz = 0; tz < CLUSTER_Z_SLICES; tz++)
                    {
                        int clusterIdx = (tx * _tileCountY + ty) * CLUSTER_Z_SLICES + tz;
                        offsets[clusterIdx] = allIndices.Count;
                        var lights = _clusterLightLists[tx][ty][tz];
                        counts[clusterIdx] = lights.Count;
                        allIndices.AddRange(lights);
                    }
                }
            }

            return (allIndices.ToArray(), offsets, counts);
        }
    }

    /// <summary>
    /// Extension methods for Vector4.
    /// </summary>
    public static class Vector4Extensions
    {
        /// <summary>Extracts XYZ components as a Vector3.</summary>
        public static Vector3 XYZ(this Vector4 v) => new Vector3(v.X, v.Y, v.Z);
    }
}
