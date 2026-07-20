// SYNAPSE OMNIA — Synapse.Core
// Split from PhysicsState.cs for maintainability.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Synapse.Core;

public class GridHash
{
    private readonly double _cellSize;
    private readonly Dictionary<long, List<int>> _cells = new();
    private readonly List<Vector3D> _points;

    public int Count => _points.Count;
    public int CellCount => _cells.Count;
    public double CellSize => _cellSize;
    public int AverageParticlesPerCell => _cells.Count > 0 ? _points.Count / _cells.Count : 0;

    public GridHash(List<Vector3D> points, double cellSize)
    {
        _points = points;
        _cellSize = cellSize;
        for (int i = 0; i < points.Count; i++)
            Insert(i, points[i]);
    }

    public GridHash(double cellSize) { _cellSize = cellSize; _points = new(); }

    public void Insert(int index, Vector3D point)
    {
        _points.Add(point);
        long hash = Hash(point);
        if (!_cells.TryGetValue(hash, out var list))
        { list = new List<int>(); _cells[hash] = list; }
        list.Add(index);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private long Hash(Vector3D p)
    {
        long ix = (long)Math.Floor(p.X / _cellSize);
        long iy = (long)Math.Floor(p.Y / _cellSize);
        long iz = (long)Math.Floor(p.Z / _cellSize);
        return ix * 73856093L ^ iy * 19349663L ^ iz * 83492791L;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private long HashCell(int ix, int iy, int iz) => ix * 73856093L ^ iy * 19349663L ^ iz * 83492791L;

    /// <summary>Find all particles within a radius of a point.</summary>
    public List<int> RadiusSearch(Vector3D center, double radius)
    {
        var results = new List<int>();
        double rSq = radius * radius;
        int cellRadius = (int)Math.Ceiling(radius / _cellSize);
        long baseHash = Hash(center);
        int baseIx = (int)Math.Floor(center.X / _cellSize);
        int baseIy = (int)Math.Floor(center.Y / _cellSize);
        int baseIz = (int)Math.Floor(center.Z / _cellSize);
        for (int dx = -cellRadius; dx <= cellRadius; dx++)
        {
            for (int dy = -cellRadius; dy <= cellRadius; dy++)
            {
                for (int dz = -cellRadius; dz <= cellRadius; dz++)
                {
                    long hash = HashCell(baseIx + dx, baseIy + dy, baseIz + dz);
                    if (_cells.TryGetValue(hash, out var list))
                    {
                        foreach (var idx in list)
                        {
                            var p = _points[idx];
                            double dSq = (p - center).LengthSquared();
                            if (dSq <= rSq)
                                results.Add(idx);
                        }
                    }
                }
            }
        }
        return results;
    }

    /// <summary>Find the nearest neighbor to a query point.</summary>
    public int NearestNeighbor(Vector3D query)
    {
        int best = -1;
        double bestDist = double.MaxValue;
        int baseIx = (int)Math.Floor(query.X / _cellSize);
        int baseIy = (int)Math.Floor(query.Y / _cellSize);
        int baseIz = (int)Math.Floor(query.Z / _cellSize);
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++)
                {
                    long hash = HashCell(baseIx + dx, baseIy + dy, baseIz + dz);
                    if (_cells.TryGetValue(hash, out var list))
                        foreach (var idx in list)
                        {
                            double dSq = (_points[idx] - query).LengthSquared();
                            if (dSq < bestDist)
                            { bestDist = dSq; best = idx; }
                        }
                }
        return best;
    }

    /// <summary>Get the cell key for a point.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long GetCellKey(Vector3D point) => Hash(point);

    /// <summary>Get all particle indices in a specific cell.</summary>
    public IReadOnlyList<int> GetCellParticles(Vector3D point) => _cells.TryGetValue(Hash(point), out var list) ? list : Array.Empty<int>();

    /// <summary>Clear all particles.</summary>
    public void Clear() { _cells.Clear(); _points.Clear(); }
}

/// <summary>Conservation law enforcement for physics simulation. Ensures exact conservation of energy, momentum, and angular momentum.</summary>
