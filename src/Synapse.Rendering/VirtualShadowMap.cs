// =============================================================================
// VirtualShadowMap.cs - GDNN Engine: Virtual Shadow Maps
// Clipmap-based virtual shadow mapping with caching and streaming
// =============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace GDNN.Rendering.Shadows
{
    public enum VSMFilterMode { None = 0, PCF = 1, PCSS = 2, VSM = 3 }
    public enum VSMCacheMode { Disabled = 0, Hardware = 1, Software = 2 }

    public record VSMConfig
    {
        public int ClipmapLevels { get; init; } = 6;
        public int TileSize { get; init; } = 128;
        public int VirtualResolution { get; init; } = 16384;
        public int PhysicalResolution { get; init; } = 4096;
        public VSMFilterMode FilterMode { get; init; } = VSMFilterMode.PCSS;
        public VSMCacheMode CacheMode { get; init; } = VSMCacheMode.Software;
        public float DepthBias { get; init; } = 1.25f;
        public float NormalBias { get; init; } = 2.0f;
        public int PCFFilterSize { get; init; } = 5;
        public float PCSSBlockerSearchSize { get; init; } = 16.0f;
        public bool EnableRayTraceCulling { get; init; }
    }

    public class VSMTile
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int ClipmapLevel { get; set; }
        public float MinDepth { get; set; } = float.MaxValue;
        public float MaxDepth { get; set; } = float.MinValue;
        public bool IsDirty { get; set; } = true;
        public bool IsCached { get; set; }
        public int PhysicalTileIndex { get; set; } = -1;
        public uint CacheHash { get; set; }
        public int LastUsedFrame { get; set; }
    }

    public class VSMClipmap
    {
        public int Level { get; set; }
        public float Scale { get; set; }
        public Vector3 Offset { get; set; }
        public Matrix4x4 ViewProjection { get; set; }
        public float NearPlane { get; set; }
        public float FarPlane { get; set; }
        public VSMTile[] Tiles { get; set; }
        public int TileCountX { get; set; }
        public int TileCountY { get; set; }
    }

    public class VSMPhysicalPage
    {
        public int PageIndex { get; set; }
        public bool IsAllocated { get; set; }
        public int VirtualX { get; set; }
        public int VirtualY { get; set; }
        public int ClipmapLevel { get; set; }
        public uint Hash { get; set; }
        public int LastAccessFrame { get; set; }
        public float[] DepthData { get; set; }
        public float[] NormalData { get; set; }
    }

    public class VirtualShadowMap : IDisposable
    {
        private VSMConfig _config;
        private VSMClipmap[] _clipmaps;
        private VSMPhysicalPage[] _physicalPages;
        private Queue<int> _freePages;
        private int _frameIndex;
        private int _width, _height;
        private bool _disposed;
        private readonly object _lock = new();

        private float[] _shadowCache;
        private float[] _normalCache;
        private float[] _tempDepth;
        private float[] _tempNormal;
        private int _cacheWidth, _cacheHeight;

        private int _totalTiles;
        private int _renderedTiles;
        private int _cachedTiles;

        public int TotalTiles => _totalTiles;
        public int RenderedTiles => _renderedTiles;
        public int CachedTiles => _cachedTiles;
        public VSMConfig Config => _config;

        public VirtualShadowMap(int width, int height, VSMConfig? config = null)
        {
            _width = width;
            _height = height;
            _config = config ?? new VSMConfig();

            InitializeClipmaps();
            InitializePhysicalPages();
            InitializeCaches();
        }

        private void InitializeClipmaps()
        {
            _clipmaps = new VSMClipmap[_config.ClipmapLevels];

            for (int level = 0; level < _config.ClipmapLevels; level++)
            {
                float scale = MathF.Pow(2.0f, level);
                int tilesX = _config.VirtualResolution / _config.TileSize;
                int tilesY = _config.VirtualResolution / _config.TileSize;

                _clipmaps[level] = new VSMClipmap
                {
                    Level = level,
                    Scale = scale,
                    TileCountX = tilesX,
                    TileCountY = tilesY,
                    Tiles = new VSMTile[tilesX * tilesY]
                };

                for (int y = 0; y < tilesY; y++)
                {
                    for (int x = 0; x < tilesX; x++)
                    {
                        int idx = y * tilesX + x;
                        _clipmaps[level].Tiles[idx] = new VSMTile
                        {
                            X = x,
                            Y = y,
                            ClipmapLevel = level
                        };
                    }
                }

                _totalTiles += tilesX * tilesY;
            }
        }

        private void InitializePhysicalPages()
        {
            int maxPages = (_config.PhysicalResolution * _config.PhysicalResolution) / (_config.TileSize * _config.TileSize);
            _physicalPages = new VSMPhysicalPage[maxPages];
            _freePages = new Queue<int>();

            for (int i = 0; i < maxPages; i++)
            {
                _physicalPages[i] = new VSMPhysicalPage { PageIndex = i };
                _freePages.Enqueue(i);
            }
        }

        private void InitializeCaches()
        {
            _cacheWidth = _config.PhysicalResolution;
            _cacheHeight = _config.PhysicalResolution;
            _shadowCache = new float[_cacheWidth * _cacheHeight];
            _normalCache = new float[_cacheWidth * _cacheHeight * 3];
            _tempDepth = new float[_config.TileSize * _config.TileSize];
            _tempNormal = new float[_config.TileSize * _config.TileSize * 3];
        }

        public void UpdateClipmaps(Vector3 cameraPosition, Matrix4x4 lightView, Matrix4x4 lightProj)
        {
            lock (_lock)
            {
                for (int level = 0; level < _config.ClipmapLevels; level++)
                {
                    var clipmap = _clipmaps[level];
                    float scale = clipmap.Scale;
                    float tileSizeWorld = _config.TileSize * scale;

                    clipmap.Offset = new Vector3(
                        MathF.Floor(cameraPosition.X / tileSizeWorld) * tileSizeWorld,
                        0,
                        MathF.Floor(cameraPosition.Z / tileSizeWorld) * tileSizeWorld
                    );

                    clipmap.ViewProjection = lightView * lightProj;
                    clipmap.NearPlane = 0.1f;
                    clipmap.FarPlane = 100.0f * scale;
                }
            }
        }

        public void CullAndRender(Vector3 cameraPosition, Matrix4x4 viewProj, float[] depthBuffer, float[] normalBuffer, int screenW, int screenH)
        {
            lock (_lock)
            {
                _renderedTiles = 0;
                _cachedTiles = 0;

                for (int level = 0; level < _config.ClipmapLevels; level++)
                {
                    var clipmap = _clipmaps[level];

                    for (int y = 0; y < clipmap.TileCountY; y++)
                    {
                        for (int x = 0; x < clipmap.TileCountX; x++)
                        {
                            int tileIdx = y * clipmap.TileCountX + x;
                            var tile = clipmap.Tiles[tileIdx];

                            if (!IsTileVisible(tile, clipmap, cameraPosition, viewProj))
                            {
                                tile.IsDirty = false;
                                continue;
                            }

                            uint hash = ComputeTileHash(tile, clipmap);

                            if (_config.CacheMode != VSMCacheMode.Disabled && tile.CacheHash == hash && tile.IsCached)
                            {
                                tile.LastUsedFrame = _frameIndex;
                                _cachedTiles++;
                                continue;
                            }

                            RenderTile(tile, clipmap, depthBuffer, normalBuffer, screenW, screenH);
                            tile.CacheHash = hash;
                            tile.IsCached = true;
                            tile.IsDirty = false;
                            tile.LastUsedFrame = _frameIndex;
                            _renderedTiles++;
                        }
                    }
                }

                EvictOldPages();
                _frameIndex++;
            }
        }

        private bool IsTileVisible(VSMTile tile, VSMClipmap clipmap, Vector3 camPos, Matrix4x4 viewProj)
        {
            float tileSizeWorld = _config.TileSize * clipmap.Scale;
            Vector3 tileMin = new Vector3(
                clipmap.Offset.X + tile.X * tileSizeWorld,
                -50,
                clipmap.Offset.Z + tile.Y * tileSizeWorld
            );
            Vector3 tileMax = tileMin + new Vector3(tileSizeWorld, 100, tileSizeWorld);

            Vector3 center = (tileMin + tileMax) * 0.5f;
            Vector3 extents = (tileMax - tileMin) * 0.5f;

            Vector4 viewCenter = Vector4.Transform(new Vector4(center, 1f), viewProj);

            if (viewCenter.Z < -viewCenter.W || viewCenter.Z > viewCenter.W) return false;
            if (viewCenter.X < -viewCenter.W || viewCenter.X > viewCenter.W) return false;
            if (viewCenter.Y < -viewCenter.W || viewCenter.Y > viewCenter.W) return false;

            float distToCam = Vector3.Distance(center, camPos);
            if (distToCam > 200.0f * clipmap.Scale) return false;

            return true;
        }

        private uint ComputeTileHash(VSMTile tile, VSMClipmap clipmap)
        {
            uint hash = (uint)(tile.X * 73856093 ^ tile.Y * 19349663 ^ tile.ClipmapLevel * 83492791);
            hash ^= (uint)(clipmap.Offset.X.GetHashCode() * 2654435761);
            hash ^= (uint)(clipmap.Offset.Z.GetHashCode() * 2246822519);
            return hash;
        }

        private void RenderTile(VSMTile tile, VSMClipmap clipmap, float[] depthBuffer, float[] normalBuffer, int screenW, int screenH)
        {
            int physicalPage = AllocatePage();
            if (physicalPage < 0) return;

            var page = _physicalPages[physicalPage];
            page.VirtualX = tile.X;
            page.VirtualY = tile.Y;
            page.ClipmapLevel = tile.ClipmapLevel;
            page.Hash = tile.CacheHash;
            page.LastAccessFrame = _frameIndex;
            page.IsAllocated = true;

            tile.PhysicalTileIndex = physicalPage;

            if (page.DepthData == null)
            {
                page.DepthData = new float[_config.TileSize * _config.TileSize];
                page.NormalData = new float[_config.TileSize * _config.TileSize * 3];
            }

            float tileSizeWorld = _config.TileSize * clipmap.Scale;
            Matrix4x4 tileViewProj = clipmap.ViewProjection;

            for (int py = 0; py < _config.TileSize; py++)
            {
                for (int px = 0; px < _config.TileSize; px++)
                {
                    float u = (float)px / _config.TileSize;
                    float v = (float)py / _config.TileSize;

                    Vector3 worldPos = new Vector3(
                        clipmap.Offset.X + (tile.X + u) * tileSizeWorld,
                        0,
                        clipmap.Offset.Z + (tile.Y + v) * tileSizeWorld
                    );

                    Vector4 clipPos = Vector4.Transform(new Vector4(worldPos, 1.0f), tileViewProj);
                    if (clipPos.W <= 0) continue;

                    float ndcX = clipPos.X / clipPos.W;
                    float ndcY = clipPos.Y / clipPos.W;
                    float ndcZ = clipPos.Z / clipPos.W;

                    int screenX = (int)((ndcX * 0.5f + 0.5f) * screenW);
                    int screenY = (int)((ndcY * 0.5f + 0.5f) * screenH);

                    if (screenX >= 0 && screenX < screenW && screenY >= 0 && screenY < screenH)
                    {
                        int screenIdx = screenY * screenW + screenX;
                        int tileIdx = py * _config.TileSize + px;

                        float depth = depthBuffer[screenIdx];
                        page.DepthData[tileIdx] = depth;

                        int nIdx = screenIdx * 3;
                        page.NormalData[tileIdx * 3] = normalBuffer[nIdx];
                        page.NormalData[tileIdx * 3 + 1] = normalBuffer[nIdx + 1];
                        page.NormalData[tileIdx * 3 + 2] = normalBuffer[nIdx + 2];

                        tile.MinDepth = MathF.Min(tile.MinDepth, depth);
                        tile.MaxDepth = MathF.Max(tile.MaxDepth, depth);
                    }
                }
            }
        }

        private int AllocatePage()
        {
            if (_freePages.Count > 0)
                return _freePages.Dequeue();

            int oldestFrame = int.MaxValue;
            int oldestIndex = -1;

            for (int i = 0; i < _physicalPages.Length; i++)
            {
                if (_physicalPages[i].IsAllocated && _physicalPages[i].LastAccessFrame < oldestFrame)
                {
                    oldestFrame = _physicalPages[i].LastAccessFrame;
                    oldestIndex = i;
                }
            }

            if (oldestIndex >= 0)
            {
                _physicalPages[oldestIndex].IsAllocated = false;
                return oldestIndex;
            }

            return -1;
        }

        private void EvictOldPages()
        {
            int evictionThreshold = _frameIndex - 120;

            for (int i = 0; i < _physicalPages.Length; i++)
            {
                var page = _physicalPages[i];
                if (page.IsAllocated && page.LastAccessFrame < evictionThreshold)
                {
                    page.IsAllocated = false;
                    _freePages.Enqueue(i);
                }
            }
        }

        public float SampleShadow(Vector3 worldPos, Vector3 lightDir, int clipmapLevel = 0)
        {
            lock (_lock)
            {
                if (clipmapLevel >= _config.ClipmapLevels) return 1.0f;

                var clipmap = _clipmaps[clipmapLevel];
                float tileSizeWorld = _config.TileSize * clipmap.Scale;

                int tileX = (int)((worldPos.X - clipmap.Offset.X) / tileSizeWorld);
                int tileY = (int)((worldPos.Z - clipmap.Offset.Z) / tileSizeWorld);

                if (tileX < 0 || tileX >= clipmap.TileCountX || tileY < 0 || tileY >= clipmap.TileCountY)
                    return SampleShadow(worldPos, lightDir, clipmapLevel + 1);

                int tileIdx = tileY * clipmap.TileCountX + tileX;
                var tile = clipmap.Tiles[tileIdx];

                if (!tile.IsCached || tile.PhysicalTileIndex < 0)
                    return 0.0f;

                var page = _physicalPages[tile.PhysicalTileIndex];
                if (page.DepthData == null) return 0.0f;

                float localX = ((worldPos.X - clipmap.Offset.X) / tileSizeWorld - tile.X) * _config.TileSize;
                float localY = ((worldPos.Z - clipmap.Offset.Z) / tileSizeWorld - tile.Y) * _config.TileSize;

                int px = Math.Clamp((int)localX, 0, _config.TileSize - 1);
                int py = Math.Clamp((int)localY, 0, _config.TileSize - 1);

                return _config.FilterMode switch
                {
                    VSMFilterMode.None => SampleBilinear(page.DepthData, localX, localY),
                    VSMFilterMode.PCF => SamplePCF(page.DepthData, localX, localY, worldPos),
                    VSMFilterMode.PCSS => SamplePCSS(page.DepthData, localX, localY, worldPos, lightDir),
                    VSMFilterMode.VSM => SampleVSM(page.DepthData, localX, localY, worldPos),
                    _ => 1.0f
                };
            }
        }

        private float SampleBilinear(float[] depthData, float x, float y)
        {
            int x0 = Math.Clamp((int)x, 0, _config.TileSize - 1);
            int y0 = Math.Clamp((int)y, 0, _config.TileSize - 1);
            int x1 = Math.Clamp(x0 + 1, 0, _config.TileSize - 1);
            int y1 = Math.Clamp(y0 + 1, 0, _config.TileSize - 1);

            float fx = x - x0;
            float fy = y - y0;

            float d00 = depthData[y0 * _config.TileSize + x0];
            float d10 = depthData[y0 * _config.TileSize + x1];
            float d01 = depthData[y1 * _config.TileSize + x0];
            float d11 = depthData[y1 * _config.TileSize + x1];

            return float.Lerp(float.Lerp(d00, d10, fx), float.Lerp(d01, d11, fx), fy);
        }

        private float SamplePCF(float[] depthData, float x, float y, Vector3 worldPos)
        {
            float shadow = 0;
            int radius = _config.PCFFilterSize / 2;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    float sx = x + kx;
                    float sy = y + ky;
                    float depth = SampleBilinear(depthData, sx, sy);
                    shadow += depth < worldPos.Z ? 0.0f : 1.0f;
                }
            }

            return shadow / ((_config.PCFFilterSize) * (_config.PCFFilterSize));
        }

        private float SamplePCSS(float[] depthData, float x, float y, Vector3 worldPos, Vector3 lightDir)
        {
            int searchRadius = (int)(_config.PCSSBlockerSearchSize / 2);

            float avgBlockerDepth = 0;
            int blockerCount = 0;

            for (int ky = -searchRadius; ky <= searchRadius; ky++)
            {
                for (int kx = -searchRadius; kx <= searchRadius; kx++)
                {
                    float depth = SampleBilinear(depthData, x + kx, y + ky);
                    if (depth < worldPos.Z)
                    {
                        avgBlockerDepth += depth;
                        blockerCount++;
                    }
                }
            }

            if (blockerCount == 0) return 1.0f;
            avgBlockerDepth /= blockerCount;

            float penumbraWidth = MathF.Max(0, (worldPos.Z - avgBlockerDepth) / avgBlockerDepth) * 10.0f;
            int filterRadius = Math.Clamp((int)(penumbraWidth + 0.5f), 1, _config.PCFFilterSize / 2);

            float shadow = 0;
            int samples = 0;
            for (int ky = -filterRadius; ky <= filterRadius; ky++)
            {
                for (int kx = -filterRadius; kx <= filterRadius; kx++)
                {
                    float depth = SampleBilinear(depthData, x + kx, y + ky);
                    shadow += depth < worldPos.Z ? 0.0f : 1.0f;
                    samples++;
                }
            }

            return samples > 0 ? shadow / samples : 1.0f;
        }

        private float SampleVSM(float[] depthData, float x, float y, Vector3 worldPos)
        {
            float meanDepth = 0;
            float meanDepthSq = 0;
            int radius = _config.PCFFilterSize / 2;
            int samples = 0;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    float d = SampleBilinear(depthData, x + kx, y + ky);
                    meanDepth += d;
                    meanDepthSq += d * d;
                    samples++;
                }
            }

            meanDepth /= samples;
            meanDepthSq /= samples;
            float variance = MathF.Max(0, meanDepthSq - meanDepth * meanDepth);
            float chebyshev = variance / (variance + MathF.Max(0, worldPos.Z - meanDepth) * MathF.Max(0, worldPos.Z - meanDepth));

            return MathF.Max(chebyshev, worldPos.Z <= meanDepth ? 1.0f : 0.0f);
        }

        public void GetStatistics(out int totalTiles, out int renderedTiles, out int cachedTiles, out int allocatedPages)
        {
            lock (_lock)
            {
                totalTiles = _totalTiles;
                renderedTiles = _renderedTiles;
                cachedTiles = _cachedTiles;
                allocatedPages = 0;
                for (int i = 0; i < _physicalPages.Length; i++)
                    if (_physicalPages[i].IsAllocated) allocatedPages++;
            }
        }

        public void Resize(int width, int height)
        {
            lock (_lock)
            {
                _width = width;
                _height = height;
                InitializeCaches();
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _shadowCache = null;
            _normalCache = null;
            _tempDepth = null;
            _tempNormal = null;
        }
    }
}
