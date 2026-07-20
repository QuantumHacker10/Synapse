// =============================================================================
// EntityBehaviorSystem.cs
// GDNN.Sentience - Complete Entity Behavior System for G-DNN Engine
// =============================================================================

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace GDNN.Sentience
{

    public class SpatialIndex
    {
        private readonly Dictionary<(int, int, int), HashSet<Guid>> _cells;
        private readonly Dictionary<Guid, (int, int, int)> _entityCells;
        private readonly float _cellSize;
        private readonly object _lock = new();

        public SpatialIndex(float cellSize = 10.0f)
        {
            _cellSize = cellSize;
            _cells = new Dictionary<(int, int, int), HashSet<Guid>>();
            _entityCells = new Dictionary<Guid, (int, int, int)>();
        }

        public void Insert(Guid entityId, Vector3 position)
        {
            var cell = GetCell(position);
            lock (_lock)
            {
                if (_entityCells.TryGetValue(entityId, out var oldCell) && _cells.TryGetValue(oldCell, out var oldSet))
                {
                    oldSet.Remove(entityId);
                    if (oldSet.Count == 0)
                        _cells.Remove(oldCell);
                }
                if (!_cells.TryGetValue(cell, out var cellSet))
                { cellSet = new HashSet<Guid>(); _cells[cell] = cellSet; }
                cellSet.Add(entityId);
                _entityCells[entityId] = cell;
            }
        }

        public void Remove(Guid entityId)
        {
            lock (_lock)
            {
                if (_entityCells.TryGetValue(entityId, out var cell) && _cells.TryGetValue(cell, out var cellSet))
                {
                    cellSet.Remove(entityId);
                    if (cellSet.Count == 0)
                        _cells.Remove(cell);
                }
                _entityCells.Remove(entityId);
            }
        }

        public List<Guid> QueryRadius(Vector3 center, float radius)
        {
            var results = new List<Guid>();
            var minCell = GetCell(center - new Vector3(radius));
            var maxCell = GetCell(center + new Vector3(radius));
            lock (_lock)
            {
                for (int x = minCell.Item1; x <= maxCell.Item1; x++)
                    for (int y = minCell.Item2; y <= maxCell.Item2; y++)
                        for (int z = minCell.Item3; z <= maxCell.Item3; z++)
                            if (_cells.TryGetValue((x, y, z), out var s))
                                results.AddRange(s);
            }
            return results;
        }

        public List<Guid> QueryBox(Vector3 min, Vector3 max)
        {
            var minCell = GetCell(min);
            var maxCell = GetCell(max);
            var results = new List<Guid>();
            lock (_lock)
            {
                for (int x = minCell.Item1; x <= maxCell.Item1; x++)
                    for (int y = minCell.Item2; y <= maxCell.Item2; y++)
                        for (int z = minCell.Item3; z <= maxCell.Item3; z++)
                            if (_cells.TryGetValue((x, y, z), out var s))
                                results.AddRange(s);
            }
            return results;
        }

        public Guid? QueryNearest(Vector3 point, Dictionary<Guid, Vector3> positions, float maxDistance = float.MaxValue)
        {
            var candidates = QueryRadius(point, maxDistance);
            Guid? nearest = null;
            float bestDistSq = maxDistance * maxDistance;
            foreach (var id in candidates)
            {
                if (positions.TryGetValue(id, out var pos))
                {
                    float d = Vector3.DistanceSquared(point, pos);
                    if (d < bestDistSq)
                    { bestDistSq = d; nearest = id; }
                }
            }
            return nearest;
        }

        public void Clear() { lock (_lock) { _cells.Clear(); _entityCells.Clear(); } }
        public int Count { get { lock (_lock) return _entityCells.Count; } }

        private (int, int, int) GetCell(Vector3 p) =>
            ((int)Math.Floor(p.X / _cellSize), (int)Math.Floor(p.Y / _cellSize), (int)Math.Floor(p.Z / _cellSize));
    }

}
