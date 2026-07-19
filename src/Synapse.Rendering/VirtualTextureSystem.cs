// =============================================================================
// VirtualTextureSystem.cs - GDNN Engine: Virtual Textures
// Sparse/resident texture streaming with tile-based management
// =============================================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Rendering.VirtualTextures
{
    public enum VTFeedbackMode { None = 0, Coarse = 1, Fine = 2 }
    public enum VTBorderColor { Black = 0, White = 1, Transparent = 2 }
    public enum VTTileState { Unloaded = 0, Loading = 1, Resident = 2, Evicting = 3 }

    public record VTConfig
    {
        public int VirtualTextureSize { get; init; } = 16384;
        public int PhysicalTileSize { get; init; } = 256;
        public int PhysicalTextureCount { get; init; } = 4;
        public int MaxResidentTiles { get; init; } = 4096;
        public int MaxTilesPerFrame { get; init; } = 32;
        public int MipBias { get; init; } = 2;
        public VTBorderColor BorderColor { get; init; } = VTBorderColor.Black;
        public bool EnableStreaming { get; init; } = true;
        public bool EnableCompression { get; init; } = true;
    }

    public class VTTile
    {
        public int TileX { get; set; }
        public int TileY { get; set; }
        public int MipLevel { get; set; }
        public VTTileState State { get; set; } = VTTileState.Unloaded;
        public int PhysicalPageIndex { get; set; } = -1;
        public uint ContentHash { get; set; }
        public int LastAccessFrame { get; set; }
        public int Priority { get; set; }
        public required string SourcePath { get; set; }
    }

    public class VTPhysicalPage
    {
        public int PageIndex { get; set; }
        public bool IsAllocated { get; set; }
        public int VirtualTileX { get; set; }
        public int VirtualTileY { get; set; }
        public int MipLevel { get; set; }
        public uint Hash { get; set; }
        public int LastAccessFrame { get; set; }
        public required float[] ColorData { get; set; }
        public required float[] NormalData { get; set; }
    }

    public class VTLayer
    {
        public required string Name { get; set; }
        public int VirtualWidth { get; set; }
        public int VirtualHeight { get; set; }
        public int TileCountX { get; set; }
        public int TileCountY { get; set; }
        public int MipLevels { get; set; }
        public required VTTile[,] Tiles { get; set; }
        public bool IsDirty { get; set; }
    }

    public class VTTileData
    {
        public int Width { get; set; }
        public int Height { get; set; }
        public int MipLevel { get; set; }
        public required float[] ColorData { get; set; }
        public required float[] NormalData { get; set; }
        public required float[] RoughnessData { get; set; }
    }

    public class VTStreamingQueue
    {
        private ConcurrentQueue<VTTile> _loadQueue;
        private ConcurrentQueue<VTTile> _unloadQueue;
        private ConcurrentDictionary<uint, VTTileData> _dataCache;
        private int _maxCacheSize;

        public int PendingLoads => _loadQueue.Count;
        public int PendingUnloads => _unloadQueue.Count;
        public int CachedTiles => _dataCache.Count;

        public VTStreamingQueue(int maxCacheSize = 1024)
        {
            _loadQueue = new ConcurrentQueue<VTTile>();
            _unloadQueue = new ConcurrentQueue<VTTile>();
            _dataCache = new ConcurrentDictionary<uint, VTTileData>();
            _maxCacheSize = maxCacheSize;
        }

        public void EnqueueLoad(VTTile tile) => _loadQueue.Enqueue(tile);
        public void EnqueueUnload(VTTile tile) => _unloadQueue.Enqueue(tile);

        public bool TryDequeueLoad(out VTTile tile) => _loadQueue.TryDequeue(out tile);
        public bool TryDequeueUnload(out VTTile tile) => _unloadQueue.TryDequeue(out tile);

        public void CacheTileData(uint hash, VTTileData data)
        {
            if (_dataCache.Count >= _maxCacheSize)
            {
                var keys = _dataCache.Keys.ToArray();
                if (keys.Length > 0)
                    _dataCache.TryRemove(keys[0], out _);
            }
            _dataCache[hash] = data;
        }

        public bool TryGetCachedData(uint hash, out VTTileData data) => _dataCache.TryGetValue(hash, out data);
    }

    public class VirtualTextureSystem : IDisposable
    {
        private VTConfig _config;
        private VTLayer[] _layers;
        private VTPhysicalPage[] _physicalPages;
        private Queue<int> _freePages;
        private VTStreamingQueue _streamingQueue;
        private int _frameIndex;
        private bool _disposed;
        private readonly object _lock = new();

        private float[] _physicalColorData;
        private float[] _physicalNormalData;
        private int _physicalTextureWidth;
        private int _physicalTextureHeight;

        private Thread _streamingThread;
        private bool _streamingRunning;
        private ManualResetEventSlim _streamingSignal;

        private int _totalTiles;
        private int _residentTiles;
        private int _streamingTiles;

        public int TotalTiles => _totalTiles;
        public int ResidentTiles => _residentTiles;
        public int StreamingTiles => _streamingTiles;
        public VTConfig Config => _config;
        public VTLayer[] Layers => _layers;
        public float[] PhysicalColorData => _physicalColorData;

        public VirtualTextureSystem(VTConfig? config = null)
        {
            _config = config ?? new VTConfig();
            _streamingQueue = new VTStreamingQueue();
            _streamingSignal = new ManualResetEventSlim(false);
            _streamingRunning = true;

            InitializePhysicalTexture();
            StartStreamingThread();
        }

        private void InitializePhysicalTexture()
        {
            int tilesPerAxis = _config.PhysicalTextureCount > 0
                ? (int)MathF.Sqrt(_config.MaxResidentTiles)
                : 64;

            _physicalTextureWidth = tilesPerAxis * _config.PhysicalTileSize;
            _physicalTextureHeight = tilesPerAxis * _config.PhysicalTileSize;

            int totalPages = tilesPerAxis * tilesPerAxis;
            _physicalPages = new VTPhysicalPage[totalPages];
            _freePages = new Queue<int>();

            for (int i = 0; i < totalPages; i++)
            {
                _physicalPages[i] = new VTPhysicalPage { PageIndex = i };
                _freePages.Enqueue(i);
            }

            _physicalColorData = new float[_physicalTextureWidth * _physicalTextureHeight * 4];
            _physicalNormalData = new float[_physicalTextureWidth * _physicalTextureHeight * 3];
        }

        public void CreateLayer(string name, int virtualWidth, int virtualHeight, int mipLevels = 8)
        {
            int tileCountX = virtualWidth / _config.PhysicalTileSize;
            int tileCountY = virtualHeight / _config.PhysicalTileSize;

            var layer = new VTLayer
            {
                Name = name,
                VirtualWidth = virtualWidth,
                VirtualHeight = virtualHeight,
                TileCountX = tileCountX,
                TileCountY = tileCountY,
                MipLevels = mipLevels,
                Tiles = new VTTile[tileCountX, tileCountY]
            };

            for (int y = 0; y < tileCountY; y++)
            {
                for (int x = 0; x < tileCountX; x++)
                {
                    layer.Tiles[x, y] = new VTTile
                    {
                        TileX = x,
                        TileY = y,
                        MipLevel = 0,
                        SourcePath = $"{name}_{x}_{y}.vt"
                    };
                    _totalTiles++;
                }
            }

            if (_layers == null)
                _layers = new[] { layer };
            else
            {
                var newLayers = new VTLayer[_layers.Length + 1];
                Array.Copy(_layers, newLayers, _layers.Length);
                newLayers[_layers.Length] = layer;
                _layers = newLayers;
            }
        }

        public void RequestTiles(int layerIndex, Vector2 uvMin, Vector2 uvMax, int screenTileSize)
        {
            if (_layers == null || layerIndex >= _layers.Length)
                return;

            var layer = _layers[layerIndex];
            int mipLevel = CalculateMipLevel(screenTileSize);

            int minTileX = Math.Clamp((int)(uvMin.X * layer.TileCountX), 0, layer.TileCountX - 1);
            int minTileY = Math.Clamp((int)(uvMin.Y * layer.TileCountY), 0, layer.TileCountY - 1);
            int maxTileX = Math.Clamp((int)(uvMax.X * layer.TileCountX), 0, layer.TileCountX - 1);
            int maxTileY = Math.Clamp((int)(uvMax.Y * layer.TileCountY), 0, layer.TileCountY - 1);

            int tilesRequested = 0;
            for (int y = minTileY; y <= maxTileY && tilesRequested < _config.MaxTilesPerFrame; y++)
            {
                for (int x = minTileX; x <= maxTileX && tilesRequested < _config.MaxTilesPerFrame; x++)
                {
                    var tile = layer.Tiles[x, y];
                    if (tile.State == VTTileState.Unloaded)
                    {
                        tile.MipLevel = mipLevel;
                        tile.Priority = CalculatePriority(x, y, minTileX, minTileY, maxTileX, maxTileY);
                        _streamingQueue.EnqueueLoad(tile);
                        tile.State = VTTileState.Loading;
                        tilesRequested++;
                    }
                }
            }
        }

        public void Update(Vector3 cameraPosition, Matrix4x4 viewProj, int screenW, int screenH)
        {
            lock (_lock)
            {
                _residentTiles = 0;
                _streamingTiles = 0;

                if (_layers == null)
                    return;

                for (int l = 0; l < _layers.Length; l++)
                {
                    var layer = _layers[l];
                    for (int y = 0; y < layer.TileCountY; y++)
                    {
                        for (int x = 0; x < layer.TileCountX; x++)
                        {
                            var tile = layer.Tiles[x, y];
                            tile.LastAccessFrame = _frameIndex;

                            if (tile.State == VTTileState.Resident)
                            {
                                _residentTiles++;
                                EvictTileIfNeeded(tile);
                            }
                            else if (tile.State == VTTileState.Loading)
                            {
                                _streamingTiles++;
                            }
                        }
                    }
                }

                ProcessStreamingQueue();
                _frameIndex++;
            }
        }

        private void ProcessStreamingQueue()
        {
            int processed = 0;
            while (_streamingQueue.TryDequeueLoad(out var tile) && processed < _config.MaxTilesPerFrame)
            {
                int pageIndex = AllocatePage();
                if (pageIndex < 0)
                    break;

                var page = _physicalPages[pageIndex];
                page.VirtualTileX = tile.TileX;
                page.VirtualTileY = tile.TileY;
                page.MipLevel = tile.MipLevel;
                page.LastAccessFrame = _frameIndex;
                page.IsAllocated = true;

                GenerateTileData(tile, page);

                tile.PhysicalPageIndex = pageIndex;
                tile.State = VTTileState.Resident;
                tile.ContentHash = ComputeTileHash(tile);
                processed++;
            }

            while (_streamingQueue.TryDequeueUnload(out var tile))
            {
                if (tile.PhysicalPageIndex >= 0 && tile.PhysicalPageIndex < _physicalPages.Length)
                {
                    _physicalPages[tile.PhysicalPageIndex].IsAllocated = false;
                    _freePages.Enqueue(tile.PhysicalPageIndex);
                }
                tile.State = VTTileState.Unloaded;
                tile.PhysicalPageIndex = -1;
            }
        }

        private void GenerateTileData(VTTile tile, VTPhysicalPage page)
        {
            int tileSize = _config.PhysicalTileSize;
            page.ColorData = new float[tileSize * tileSize * 4];
            page.NormalData = new float[tileSize * tileSize * 3];

            uint hash = ComputeTileHash(tile);
            if (_streamingQueue.TryGetCachedData(hash, out var cached))
            {
                if (cached.ColorData != null)
                    Array.Copy(cached.ColorData, page.ColorData, Math.Min(cached.ColorData.Length, page.ColorData.Length));
                if (cached.NormalData != null)
                    Array.Copy(cached.NormalData, page.NormalData, Math.Min(cached.NormalData.Length, page.NormalData.Length));
                return;
            }

            Random rng = new((int)(hash & 0x7FFFFFFF));
            float baseHue = (float)rng.NextDouble();
            float baseSat = 0.3f + (float)rng.NextDouble() * 0.5f;

            for (int y = 0; y < tileSize; y++)
            {
                for (int x = 0; x < tileSize; x++)
                {
                    float u = (float)x / tileSize;
                    float v = (float)y / tileSize;

                    float pattern = MathF.Sin(u * 4 + baseHue * 6.28f) * 0.5f + 0.5f;
                    pattern *= MathF.Cos(v * 3 + baseHue * 3.14f) * 0.5f + 0.5f;

                    Vector3 color = HSVToRGB(baseHue, baseSat, 0.3f + pattern * 0.5f);

                    int idx = (y * tileSize + x) * 4;
                    page.ColorData[idx] = color.X;
                    page.ColorData[idx + 1] = color.Y;
                    page.ColorData[idx + 2] = color.Z;
                    page.ColorData[idx + 3] = 1.0f;

                    int nidx = (y * tileSize + x) * 3;
                    float nx = MathF.Sin(u * 10 + baseHue) * 0.5f;
                    float ny = MathF.Cos(v * 10 + baseHue) * 0.5f;
                    float nz = MathF.Sqrt(MathF.Max(0, 1.0f - nx * nx - ny * ny));
                    page.NormalData[nidx] = nx;
                    page.NormalData[nidx + 1] = ny;
                    page.NormalData[nidx + 2] = nz;
                }
            }

            var tileData = new VTTileData
            {
                Width = tileSize,
                Height = tileSize,
                MipLevel = tile.MipLevel,
                ColorData = page.ColorData,
                NormalData = page.NormalData
            };
            _streamingQueue.CacheTileData(hash, tileData);
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

        private void EvictTileIfNeeded(VTTile tile)
        {
            int evictionThreshold = _frameIndex - 300;
            if (tile.LastAccessFrame < evictionThreshold && _residentTiles > _config.MaxResidentTiles * 0.9f)
            {
                tile.State = VTTileState.Evicting;
                _streamingQueue.EnqueueUnload(tile);
            }
        }

        private int CalculateMipLevel(int screenTileSize)
        {
            int mip = 0;
            int tileSize = _config.PhysicalTileSize;
            while (tileSize > screenTileSize && mip < 8)
            {
                tileSize /= 2;
                mip++;
            }
            return mip + _config.MipBias;
        }

        private int CalculatePriority(int x, int y, int minX, int minY, int maxX, int maxY)
        {
            int cx = (minX + maxX) / 2;
            int cy = (minY + maxY) / 2;
            int dx = x - cx;
            int dy = y - cy;
            return -(dx * dx + dy * dy);
        }

        private uint ComputeTileHash(VTTile tile)
        {
            return (uint)(tile.TileX * 73856093 ^ tile.TileY * 19349663 ^ tile.MipLevel * 83492791);
        }

        private Vector3 HSVToRGB(float h, float s, float v)
        {
            int i = (int)(h * 6);
            float f = h * 6 - i;
            float p = v * (1 - s);
            float q = v * (1 - f * s);
            float t = v * (1 - (1 - f) * s);

            return (i % 6) switch
            {
                0 => new Vector3(v, t, p),
                1 => new Vector3(q, v, p),
                2 => new Vector3(p, v, t),
                3 => new Vector3(p, q, v),
                4 => new Vector3(t, p, v),
                _ => new Vector3(v, p, q)
            };
        }

        public VTTileData? SampleVirtualTexture(int layerIndex, float u, float v)
        {
            lock (_lock)
            {
                if (_layers == null || layerIndex >= _layers.Length)
                    return null;

                var layer = _layers[layerIndex];
                int tileX = Math.Clamp((int)(u * layer.TileCountX), 0, layer.TileCountX - 1);
                int tileY = Math.Clamp((int)(v * layer.TileCountY), 0, layer.TileCountY - 1);

                var tile = layer.Tiles[tileX, tileY];
                if (tile.State != VTTileState.Resident || tile.PhysicalPageIndex < 0)
                    return null;

                var page = _physicalPages[tile.PhysicalPageIndex];
                return new VTTileData
                {
                    Width = _config.PhysicalTileSize,
                    Height = _config.PhysicalTileSize,
                    MipLevel = tile.MipLevel,
                    ColorData = page.ColorData,
                    NormalData = page.NormalData
                };
            }
        }

        private void StartStreamingThread()
        {
            _streamingThread = new Thread(() =>
            {
                while (_streamingRunning)
                {
                    _streamingSignal.Wait(16);
                    _streamingSignal.Reset();

                    int processed = 0;
                    while (_streamingQueue.TryDequeueLoad(out var tile) && processed < 4)
                    {
                        GenerateTileDataForStreaming(tile);
                        processed++;
                    }
                }
            })
            {
                IsBackground = true,
                Priority = ThreadPriority.BelowNormal
            };
            _streamingThread.Start();
        }

        private void GenerateTileDataForStreaming(VTTile tile)
        {
            int tileSize = _config.PhysicalTileSize;
            float[] colorData = new float[tileSize * tileSize * 4];

            Random rng = new((int)(ComputeTileHash(tile) & 0x7FFFFFFF));
            float hue = (float)rng.NextDouble();

            for (int y = 0; y < tileSize; y++)
            {
                for (int x = 0; x < tileSize; x++)
                {
                    float u = (float)x / tileSize;
                    float v = (float)y / tileSize;
                    Vector3 color = HSVToRGB(hue, 0.5f, 0.3f + u * 0.4f);
                    int idx = (y * tileSize + x) * 4;
                    colorData[idx] = color.X;
                    colorData[idx + 1] = color.Y;
                    colorData[idx + 2] = color.Z;
                    colorData[idx + 3] = 1.0f;
                }
            }

            var data = new VTTileData
            {
                Width = tileSize,
                Height = tileSize,
                MipLevel = tile.MipLevel,
                ColorData = colorData
            };
            _streamingQueue.CacheTileData(ComputeTileHash(tile), data);
        }

        public void GetStatistics(out int totalTiles, out int residentTiles, out int streamingTiles, out int physicalPages)
        {
            lock (_lock)
            {
                totalTiles = _totalTiles;
                residentTiles = _residentTiles;
                streamingTiles = _streamingTiles;
                physicalPages = 0;
                for (int i = 0; i < _physicalPages.Length; i++)
                    if (_physicalPages[i].IsAllocated)
                        physicalPages++;
            }
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _streamingRunning = false;
            _streamingSignal?.Set();
            _streamingThread?.Join(2000);
            _physicalColorData = null;
            _physicalNormalData = null;
        }
    }
}
