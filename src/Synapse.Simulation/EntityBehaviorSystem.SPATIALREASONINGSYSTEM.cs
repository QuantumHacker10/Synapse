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

    public class SpatialReasoningSystem
    {
        private readonly SpatialIndex _spatialIndex;
        private readonly Dictionary<Guid, List<Vector3>> _pathCache = new();
        private readonly Dictionary<Guid, Territory> _territories = new();

        public SpatialReasoningSystem() { _spatialIndex = new SpatialIndex(15f); }

        public float CellSize { get; set; } = 15f;

        public List<SentientEntity> FindNearestEntities(SentientEntity entity, WorldStateData worldState, int count = 5)
        {
            var candidates = _spatialIndex.QueryRadius(entity.Position, entity.PerceptionRadius);
            return candidates
                .Where(id => id != entity.EntityId && worldState.Entities.ContainsKey(id))
                .Select(id => worldState.Entities[id])
                .OrderBy(e => Vector3.Distance(entity.Position, e.Position))
                .Take(count)
                .ToList();
        }

        public List<SentientEntity> FindEntitiesInRadius(SentientEntity entity, WorldStateData worldState, float radius)
        {
            var candidates = _spatialIndex.QueryRadius(entity.Position, radius);
            return candidates
                .Where(id => id != entity.EntityId && worldState.Entities.ContainsKey(id))
                .Select(id => worldState.Entities[id])
                .ToList();
        }

        public bool HasLineOfSight(Vector3 from, Vector3 to, List<Vector3> obstacles = null)
        {
            obstacles ??= new List<Vector3>();
            var dir = to - from;
            float dist = dir.Length();
            var norm = dir / dist;
            int steps = (int)(dist / 0.5f);
            for (int i = 0; i <= steps; i++)
            {
                var point = from + norm * (i * 0.5f);
                if (obstacles.Any(o => Vector3.Distance(point, o) < 0.5f))
                    return false;
            }
            return true;
        }

        public List<Vector3> FindPath(Vector3 start, Vector3 goal, List<Vector3> obstacles = null, int maxIterations = 1000)
        {
            obstacles ??= new List<Vector3>();
            var openSet = new SortedSet<(float F, Vector3 Pos)>(Comparer<(float, Vector3)>.Create((a, b) => a.Item1.CompareTo(b.Item1)));
            var cameFrom = new Dictionary<Vector3, Vector3>();
            var gScore = new Dictionary<Vector3, float> { [start] = 0 };
            var closedSet = new HashSet<Vector3>();
            var snapStart = SnapToGrid(start);
            var snapGoal = SnapToGrid(goal);

            openSet.Add((Heuristic(snapStart, snapGoal), snapStart));
            int iter = 0;

            while (openSet.Count > 0 && iter < maxIterations)
            {
                iter++;
                var current = openSet.Min;
                openSet.Remove(current);

                if (Vector3.Distance(current.Pos, snapGoal) < 1f)
                    return ReconstructPath(cameFrom, current.Pos);

                closedSet.Add(current.Pos);
                float currentG = gScore.GetValueOrDefault(current.Pos, float.MaxValue);

                foreach (var neighbor in GetNeighbors(current.Pos))
                {
                    if (closedSet.Contains(neighbor))
                        continue;
                    if (obstacles.Any(o => Vector3.Distance(neighbor, o) < 1f))
                        continue;

                    float tentativeG = currentG + Vector3.Distance(current.Pos, neighbor);
                    if (tentativeG < gScore.GetValueOrDefault(neighbor, float.MaxValue))
                    {
                        cameFrom[neighbor] = current.Pos;
                        gScore[neighbor] = tentativeG;
                        float f = tentativeG + Heuristic(neighbor, snapGoal);
                        openSet.Add((f, neighbor));
                    }
                }
            }
            return new List<Vector3> { start, goal };
        }

        public Vector3 CalculateAvoidance(Vector3 position, List<SentientEntity> nearbyEntities, float avoidRadius = 3f)
        {
            var avoidance = Vector3.Zero;
            foreach (var e in nearbyEntities)
            {
                float dist = Vector3.Distance(position, e.Position);
                if (dist < avoidRadius && dist > 0.01f)
                {
                    var dir = Vector3.Normalize(position - e.Position);
                    float strength = (avoidRadius - dist) / avoidRadius;
                    avoidance += dir * strength;
                }
            }
            return avoidance;
        }

        public Vector3 CalculateFormationPosition(Vector3 leaderPos, int index, int total, float spacing = 2f, FormationType formation = FormationType.Circle)
        {
            switch (formation)
            {
                case FormationType.Line:
                    return leaderPos + new Vector3(index * spacing, 0, 0);
                case FormationType.Circle:
                    {
                        float angle = (2 * MathF.PI * index) / Math.Max(1, total);
                        return leaderPos + new Vector3(MathF.Cos(angle) * spacing, 0, MathF.Sin(angle) * spacing);
                    }
                case FormationType.VShape:
                    {
                        int side = index % 2 == 0 ? 1 : -1;
                        int row = (index / 2) + 1;
                        return leaderPos + new Vector3(side * row * spacing * 0.5f, 0, -row * spacing);
                    }
                case FormationType.Grid:
                    {
                        int cols = (int)Math.Ceiling(Math.Sqrt(total));
                        int row = index / cols;
                        int col = index % cols;
                        return leaderPos + new Vector3((col - cols / 2f) * spacing, 0, row * spacing);
                    }
                default:
                    return leaderPos;
            }
        }

        public void RegisterTerritory(Guid ownerId, Vector3 center, float radius)
        {
            _territories[ownerId] = new Territory { OwnerId = ownerId, Center = center, Radius = radius };
        }

        public bool IsInTerritory(Vector3 position, out Guid? ownerId)
        {
            foreach (var t in _territories.Values)
            {
                if (Vector3.Distance(position, t.Center) <= t.Radius)
                { ownerId = t.OwnerId; return true; }
            }
            ownerId = null;
            return false;
        }

        public List<Vector3> FindPatrolPoints(Vector3 center, float radius, int count = 4)
        {
            var points = new List<Vector3>();
            for (int i = 0; i < count; i++)
            {
                float angle = (2 * MathF.PI * i) / count;
                points.Add(center + new Vector3(MathF.Cos(angle) * radius, 0, MathF.Sin(angle) * radius));
            }
            return points;
        }

        public Vector3 PredictPosition(SentientEntity entity, float timeAhead)
        {
            return entity.Position + entity.Velocity * timeAhead;
        }

        public Vector3 InterceptPosition(SentientEntity pursuer, SentientEntity target, float pursuerSpeed)
        {
            var toTarget = target.Position - pursuer.Position;
            float dist = toTarget.Length();
            float targetSpeed = target.Velocity.Length();
            if (targetSpeed < 0.01f)
                return target.Position;
            float t = dist / Math.Max(0.01f, pursuerSpeed + targetSpeed);
            return target.Position + target.Velocity * t;
        }

        public void UpdateSpatialIndex(IEnumerable<SentientEntity> entities) { _spatialIndex.Clear(); foreach (var e in entities) _spatialIndex.Insert(e.EntityId, e.Position); }

        private Vector3 SnapToGrid(Vector3 p) => new((float)Math.Round(p.X), (float)Math.Round(p.Y), (float)Math.Round(p.Z));

        private float Heuristic(Vector3 a, Vector3 b) => Vector3.Distance(a, b);

        private List<Vector3> GetNeighbors(Vector3 pos)
        {
            var neighbors = new List<Vector3>();
            for (int x = -1; x <= 1; x++)
                for (int y = -1; y <= 1; y++)
                    for (int z = -1; z <= 1; z++)
                        if (x != 0 || y != 0 || z != 0)
                            neighbors.Add(pos + new Vector3(x, y, z));
            return neighbors;
        }

        private List<Vector3> ReconstructPath(Dictionary<Vector3, Vector3> cameFrom, Vector3 current)
        {
            var path = new List<Vector3> { current };
            while (cameFrom.TryGetValue(current, out var prev))
            { current = prev; path.Insert(0, current); }
            return path;
        }
    }

    public enum FormationType { Line, Circle, VShape, Grid }

    public class Territory
    {
        public Guid OwnerId { get; set; }
        public Vector3 Center { get; set; }
        public float Radius { get; set; }
    }

}
