using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;
using GDNN.Rendering.Shaders;

namespace GDNN.Rendering.World
{
    public enum CellState { Unloaded = 0, Loading = 1, Loaded = 2, Unloading = 3 }
    public enum StreamingPriority { Critical = 0, High = 1, Medium = 2, Low = 3, Background = 4 }
    public enum CellLOD { LOD0 = 0, LOD1 = 1, LOD2 = 2, LOD3 = 3, HLOD = 4 }

    public record WorldPartitionConfig
    {
        public float CellSize { get; init; } = 256.0f;
        public int MaxLoadedCells { get; init; } = 64;
        public int MaxLoadingCells { get; init; } = 8;
        public float LoadingRange { get; init; } = 1024.0f;
        public float UnloadingRange { get; init; } = 1536.0f;
        public float LOD0Range { get; init; } = 256.0f;
        public float LOD1Range { get; init; } = 512.0f;
        public float LOD2Range { get; init; } = 1024.0f;
        public float LOD3Range { get; init; } = 2048.0f;
        public int StreamingBatchSize { get; init; } = 4;
        public bool EnableHLOD { get; init; } = true;
        public bool EnableAsyncLoading { get; init; } = true;
    }

    public struct WorldBounds
    {
        public Vector3 Min;
        public Vector3 Max;
        public Vector3 Center => (Min + Max) * 0.5f;
        public float DistanceTo(Vector3 point)
        {
            Vector3 closest = Vector3.Clamp(point, Min, Max);
            return Vector3.Distance(closest, point);
        }
    }

    public struct CellData
    {
        public int CellX;
        public int CellZ;
        public CellState State;
        public CellLOD LOD;
        public StreamingPriority Priority;
        public float DistanceToViewer;
        public float LastAccessTime;
        public float LoadProgress;
        public WorldBounds Bounds;
        public Vector3 WorldMin;
        public Vector3 WorldMax;
    }

    public struct StreamingRequest
    {
        public int CellX;
        public int CellZ;
        public StreamingPriority Priority;
        public float RequestTime;
        public bool IsCancellation;
    }

    public struct HLODCluster
    {
        public int ClusterId;
        public int CellX;
        public int CellZ;
        public Vector3 Center;
        public float Radius;
        public bool IsBuilt;
        public float[] MeshData;
    }

    public class StreamingCellManager
    {
        private ConcurrentDictionary<long, CellData> _cells;
        private ConcurrentQueue<StreamingRequest> _loadQueue;
        private ConcurrentQueue<StreamingRequest> _unloadQueue;
        private Dictionary<long, float[]> _cellMeshData;
        private Dictionary<long, HLODCluster[]> _cellHLODs;

        public int CellCount => _cells.Count;
        public int PendingLoads => _loadQueue.Count;
        public int PendingUnloads => _unloadQueue.Count;

        public StreamingCellManager()
        {
            _cells = new ConcurrentDictionary<long, CellData>();
            _loadQueue = new ConcurrentQueue<StreamingRequest>();
            _unloadQueue = new ConcurrentQueue<StreamingRequest>();
            _cellMeshData = new Dictionary<long, float[]>();
            _cellHLODs = new Dictionary<long, HLODCluster[]>();
        }

        public static long CellKey(int x, int z) => ((long)x << 32) | (long)(uint)z;

        public CellData GetOrCreateCell(int x, int z, float cellSize)
        {
            long key = CellKey(x, z);
            return _cells.GetOrAdd(key, _ =>
            {
                float minX = x * cellSize;
                float minZ = z * cellSize;
                return new CellData
                {
                    CellX = x, CellZ = z,
                    State = CellState.Unloaded, LOD = CellLOD.LOD3,
                    WorldMin = new Vector3(minX, -100, minZ),
                    WorldMax = new Vector3(minX + cellSize, 500, minZ + cellSize)
                };
            });
        }

        public void EnqueueLoad(StreamingRequest req) => _loadQueue.Enqueue(req);
        public void EnqueueUnload(StreamingRequest req) => _unloadQueue.Enqueue(req);
        public bool TryDequeueLoad(out StreamingRequest req) => _loadQueue.TryDequeue(out req);
        public bool TryDequeueUnload(out StreamingRequest req) => _unloadQueue.TryDequeue(out req);

        public void SetCellMeshData(long key, float[] data) => _cellMeshData[key] = data;
        public float[] GetCellMeshData(long key) => _cellMeshData.TryGetValue(key, out var d) ? d : null;
        public void SetCellHLODs(long key, HLODCluster[] h) => _cellHLODs[key] = h;

        public void UpdateCellState(long key, CellState state, float progress = 0)
        {
            if (_cells.TryGetValue(key, out var cell))
            {
                cell.State = state;
                cell.LoadProgress = progress;
                cell.LastAccessTime = Environment.TickCount / 1000.0f;
                _cells[key] = cell;
            }
        }

        public void UpdateCellLOD(long key, CellLOD lod)
        {
            if (_cells.TryGetValue(key, out var cell))
            {
                cell.LOD = lod;
                _cells[key] = cell;
            }
        }

        public void GetCellsInRange(Vector3 center, float range, List<CellData> result)
        {
            result.Clear();
            foreach (var kvp in _cells)
                if (kvp.Value.Bounds.DistanceTo(center) <= range)
                    result.Add(kvp.Value);
        }

        public void GetAllCells(List<CellData> output)
        {
            output.Clear();
            foreach (var kvp in _cells)
                output.Add(kvp.Value);
        }
    }

    public class WorldPartitionSystem : IDisposable
    {
        private WorldPartitionConfig _config;
        private StreamingCellManager _cellManager;
        private Vector3 _viewerPosition;
        private int _viewerCellX, _viewerCellZ;
        private bool _disposed;
        private readonly object _lock = new();
        private List<CellData> _tempCells = new();
        private Thread _streamingThread;
        private bool _streamingRunning;
        private ManualResetEventSlim _streamingSignal;
        private Queue<float> _recentLoadTimes = new();
        private int _totalCells, _loadedCells, _loadingCells;

        public int LoadedCells => _loadedCells;
        public int LoadingCells => _loadingCells;
        public int TotalCells => _totalCells;
        public StreamingCellManager CellManager => _cellManager;

        public WorldPartitionSystem(WorldPartitionConfig? config = null)
        {
            _config = config ?? new WorldPartitionConfig();
            _cellManager = new StreamingCellManager();
            _streamingSignal = new ManualResetEventSlim(false);
            _streamingRunning = true;
            StartStreamingThread();
        }

        private void StartStreamingThread()
        {
            _streamingThread = new Thread(() =>
            {
                while (_streamingRunning)
                {
                    _streamingSignal.Wait(16);
                    _streamingSignal.Reset();
                    ProcessStreamingQueue();
                }
            }) { IsBackground = true, Priority = ThreadPriority.BelowNormal };
            _streamingThread.Start();
        }

        public void Update(Vector3 viewerPosition, float deltaTime)
        {
            lock (_lock)
            {
                _viewerPosition = viewerPosition;
                _viewerCellX = (int)MathF.Floor(viewerPosition.X / _config.CellSize);
                _viewerCellZ = (int)MathF.Floor(viewerPosition.Z / _config.CellSize);

                UpdateVisibility();
                UpdateStreaming();
                UpdateLODs();
                UpdateStats();
            }
        }

        private void UpdateVisibility()
        {
            int rangeCells = (int)MathF.Ceiling(_config.LoadingRange / _config.CellSize) + 1;

            for (int z = _viewerCellZ - rangeCells; z <= _viewerCellZ + rangeCells; z++)
            {
                for (int x = _viewerCellX - rangeCells; x <= _viewerCellX + rangeCells; x++)
                {
                    var cell = _cellManager.GetOrCreateCell(x, z, _config.CellSize);
                    float dist = cell.Bounds.DistanceTo(_viewerPosition);
                    cell.DistanceToViewer = dist;

                    long key = StreamingCellManager.CellKey(x, z);

                    if (dist <= _config.LoadingRange && cell.State == CellState.Unloaded)
                    {
                        cell.Priority = dist <= _config.LOD0Range ? StreamingPriority.Critical
                            : dist <= _config.LOD1Range ? StreamingPriority.High
                            : dist <= _config.LOD2Range ? StreamingPriority.Medium
                            : StreamingPriority.Low;

                        _cellManager.EnqueueLoad(new StreamingRequest
                        {
                            CellX = x, CellZ = z,
                            Priority = cell.Priority,
                            RequestTime = Environment.TickCount / 1000.0f
                        });
                        cell.State = CellState.Loading;
                    }
                    else if (dist > _config.UnloadingRange && cell.State == CellState.Loaded)
                    {
                        _cellManager.EnqueueUnload(new StreamingRequest
                        {
                            CellX = x, CellZ = z,
                            RequestTime = Environment.TickCount / 1000.0f
                        });
                        cell.State = CellState.Unloading;
                    }

                    _cellManager.UpdateCellState(key, cell.State);
                }
            }
        }

        private void ProcessStreamingQueue()
        {
            lock (_lock)
            {
                UpdateStreaming();
            }
        }

        private void UpdateStreaming()
        {
            int loaded = 0, loading = 0;

            while (_cellManager.TryDequeueLoad(out var req) && loading < _config.MaxLoadingCells)
            {
                long key = StreamingCellManager.CellKey(req.CellX, req.CellZ);
                GenerateCellData(req.CellX, req.CellZ);
                _cellManager.UpdateCellState(key, CellState.Loaded, 1.0f);

                float loadTime = 0.01f;
                _recentLoadTimes.Enqueue(loadTime);
                if (_recentLoadTimes.Count > 100) _recentLoadTimes.Dequeue();
                loading++;
            }

            while (_cellManager.TryDequeueUnload(out var req))
            {
                long key = StreamingCellManager.CellKey(req.CellX, req.CellZ);
                _cellManager.UpdateCellState(key, CellState.Unloaded);
            }

            _tempCells.Clear();
            _cellManager.GetAllCells(_tempCells);
            _loadedCells = 0;
            _loadingCells = 0;
            foreach (var c in _tempCells)
            {
                if (c.State == CellState.Loaded) _loadedCells++;
                else if (c.State == CellState.Loading) _loadingCells++;
            }
        }

        private void GenerateCellData(int cellX, int cellZ)
        {
            long key = StreamingCellManager.CellKey(cellX, cellZ);
            Random rng = new(cellX * 73856093 ^ cellZ * 19349663);
            int objCount = rng.Next(5, 20);
            float[] meshData = new float[objCount * 36];

            float baseX = cellX * _config.CellSize;
            float baseZ = cellZ * _config.CellSize;

            for (int i = 0; i < objCount; i++)
            {
                float px = baseX + (float)rng.NextDouble() * _config.CellSize;
                float pz = baseZ + (float)rng.NextDouble() * _config.CellSize;
                float py = (float)rng.NextDouble() * 10;
                float s = 0.5f + (float)rng.NextDouble() * 2.0f;

                int vOff = i * 36;
                meshData[vOff] = px - s; meshData[vOff + 1] = py; meshData[vOff + 2] = pz - s;
                meshData[vOff + 3] = px + s; meshData[vOff + 4] = py; meshData[vOff + 5] = pz - s;
                meshData[vOff + 6] = px + s; meshData[vOff + 7] = py + s * 2; meshData[vOff + 8] = pz - s;
                meshData[vOff + 9] = px - s; meshData[vOff + 10] = py; meshData[vOff + 11] = pz + s;
                meshData[vOff + 12] = px + s; meshData[vOff + 13] = py; meshData[vOff + 14] = pz + s;
                meshData[vOff + 15] = px + s; meshData[vOff + 16] = py + s * 2; meshData[vOff + 17] = pz + s;
                meshData[vOff + 18] = px - s; meshData[vOff + 19] = py + s * 2; meshData[vOff + 20] = pz - s;
                meshData[vOff + 21] = px - s; meshData[vOff + 22] = py + s * 2; meshData[vOff + 23] = pz + s;
                meshData[vOff + 24] = px - s; meshData[vOff + 25] = py; meshData[vOff + 26] = pz - s;
                meshData[vOff + 27] = px - s; meshData[vOff + 28] = py; meshData[vOff + 29] = pz + s;
                meshData[vOff + 30] = px + s; meshData[vOff + 31] = py; meshData[vOff + 32] = pz - s;
                meshData[vOff + 33] = px + s; meshData[vOff + 34] = py; meshData[vOff + 35] = pz + s;
            }

            _cellManager.SetCellMeshData(key, meshData);

            if (_config.EnableHLOD)
            {
                var hloads = new HLODCluster[]
                {
                    new HLODCluster
                    {
                        ClusterId = 0, CellX = cellX, CellZ = cellZ,
                        Center = new Vector3(baseX + _config.CellSize * 0.5f, 5, baseZ + _config.CellSize * 0.5f),
                        Radius = _config.CellSize * 0.7f, IsBuilt = true
                    }
                };
                _cellManager.SetCellHLODs(key, hloads);
            }
        }

        private void UpdateLODs()
        {
            _tempCells.Clear();
            _cellManager.GetAllCells(_tempCells);
            foreach (var cell in _tempCells)
            {
                if (cell.State != CellState.Loaded) continue;
                long key = StreamingCellManager.CellKey(cell.CellX, cell.CellZ);

                CellLOD newLOD = cell.DistanceToViewer <= _config.LOD0Range ? CellLOD.LOD0
                    : cell.DistanceToViewer <= _config.LOD1Range ? CellLOD.LOD1
                    : cell.DistanceToViewer <= _config.LOD2Range ? CellLOD.LOD2
                    : CellLOD.LOD3;

                if (newLOD != cell.LOD)
                {
                    _cellManager.UpdateCellLOD(key, newLOD);
                }
            }
        }

        private void UpdateStats()
        {
            _totalCells = _cellManager.CellCount;
        }

        public float GetAverageLoadTime()
        {
            if (_recentLoadTimes.Count == 0) return 0;
            float sum = 0;
            foreach (var t in _recentLoadTimes) sum += t;
            return sum / _recentLoadTimes.Count;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _streamingRunning = false;
            _streamingSignal?.Set();
            _streamingThread?.Join(2000);
        }
    }
}
