// =============================================================================
// MeshLODSystem.cs - G-DNN Engine: Level of Detail Management
// GDNN.Engine - GDNN.Rendering.LOD
// Automatic LOD generation, selection, and management for mesh rendering
// =============================================================================

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;

namespace GDNN.Rendering.LOD
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    /// <summary>LOD transition strategy.</summary>
    public enum LodTransitionMode
    {
        Immediate,
        Dithered,
        CrossFade,
        Smooth
    }

    /// <summary>LOD calculation method.</summary>
    public enum LodCalculationMode
    {
        ScreenCoverage,
        Distance,
        ScreenCoverageWithHysteresis
    }

    /// <summary>Quality of the LOD generation algorithm.</summary>
    public enum LodGenerationQuality
    {
        Fast,
        Balanced,
        HighQuality
    }

    // =========================================================================
    // LOD LEVEL DEFINITION
    // =========================================================================

    /// <summary>
    /// Defines a single LOD level with its mesh data and transition parameters.
    /// </summary>
    [DebuggerDisplay("LOD Level {Level}: Screen={ScreenCoverageThreshold:F2}, Tris={TriangleCount}")]
    public class LodLevel
    {
        public int Level { get; set; }
        public float ScreenCoverageThreshold { get; set; }
        public float MinDistance { get; set; }
        public float MaxDistance { get; set; }
        public float FadeInDistance { get; set; }
        public float FadeOutDistance { get; set; }
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public float QualityFactor { get; set; } = 1.0f;
        public bool IsImpostor { get; set; }
        public object? MeshData { get; set; }
    }

    // =========================================================================
    // LOD GROUP
    // =========================================================================

    /// <summary>
    /// Groups multiple LOD levels for a single mesh, managing transitions
    /// and distance-based selection.
    /// </summary>
    [DebuggerDisplay("LODGroup({Name}): Levels={Levels.Count}, Mode={CalculationMode}")]
    public class LodGroup
    {
        private static int _nextId;
        public int Id { get; } = System.Threading.Interlocked.Increment(ref _nextId);
        public string Name { get; set; } = "";
        public Vector3 LocalReferencePoint { get; set; }
        public float ReferenceRadius { get; set; }
        public LodCalculationMode CalculationMode { get; set; } = LodCalculationMode.ScreenCoverageWithHysteresis;
        public LodTransitionMode TransitionMode { get; set; } = LodTransitionMode.Dithered;
        public List<LodLevel> Levels { get; } = new();
        public bool ForceHighestLOD { get; set; }
        public bool ForceLowestLOD { get; set; }
        public float Hysteresis { get; set; } = 0.05f;
        public float LastSelectedFactor { get; private set; }
        public int LastSelectedLevel { get; private set; }

        public LodLevel? SelectLevel(float screenCoverage, float distance)
        {
            if (ForceHighestLOD)
                return Levels.Count > 0 ? Levels[0] : null;
            if (ForceLowestLOD)
                return Levels.Count > 0 ? Levels[^1] : null;
            if (Levels.Count == 0)
                return null;

            float factor = CalculationMode switch
            {
                LodCalculationMode.ScreenCoverage => screenCoverage,
                LodCalculationMode.Distance => distance,
                LodCalculationMode.ScreenCoverageWithHysteresis => screenCoverage,
                _ => screenCoverage
            };

            LodLevel? selected = null;

            for (int i = 0; i < Levels.Count; i++)
            {
                var level = Levels[i];
                float threshold = level.ScreenCoverageThreshold;

                if (CalculationMode == LodCalculationMode.Distance)
                {
                    if (distance >= level.MinDistance && distance < level.MaxDistance)
                    {
                        selected = level;
                        break;
                    }
                    continue;
                }

                if (factor >= threshold * (1.0f - Hysteresis))
                {
                    selected = level;
                    break;
                }
            }

            selected ??= Levels[^1];

            float transitionFactor = 0;
            if (LastSelectedLevel != selected.Level && Levels.Count > 1)
            {
                int prevIdx = Levels.FindIndex(l => l.Level == LastSelectedLevel);
                int newIdx = Levels.IndexOf(selected);

                if (prevIdx >= 0 && newIdx >= 0)
                {
                    float prevThreshold = Levels[prevIdx].ScreenCoverageThreshold;
                    float newThreshold = selected.ScreenCoverageThreshold;
                    float range = MathF.Abs(prevThreshold - newThreshold);
                    if (range > 0.001f)
                        transitionFactor = MathF.Abs(factor - prevThreshold) / range;
                }
            }

            LastSelectedFactor = factor;
            LastSelectedLevel = selected.Level;
            return selected;
        }

        public float ComputeScreenCoverage(Vector3 objectCenter, float objectRadius, Vector3 cameraPosition, float fovRadians, float viewportHeight)
        {
            float distance = Vector3.Distance(cameraPosition, objectCenter);
            if (distance < 0.001f)
                return 1.0f;

            float apparentSize = objectRadius / distance;
            float screenSize = apparentSize / MathF.Tan(fovRadians * 0.5f) * viewportHeight;
            return Math.Clamp(screenSize / viewportHeight, 0, 1);
        }

        public float ComputeScreenCoverageFromBounds(BoundingBox3D bounds, Vector3 cameraPosition, float fovRadians, float viewportHeight)
        {
            float distance = Vector3.Distance(cameraPosition, bounds.Center);
            if (distance < 0.001f)
                return 1.0f;

            float radius = bounds.Extents.Length();
            return ComputeScreenCoverage(bounds.Center, radius, cameraPosition, fovRadians, viewportHeight);
        }
    }

    // =========================================================================
    // LOD GENERATOR
    // =========================================================================

    /// <summary>
    /// Generates LOD levels from a base mesh by reducing triangle count per level
    /// using Garland–Heckbert quadric error metrics (QEM) edge collapse.
    /// </summary>
    public class LodGenerator
    {
        /// <summary>
        /// Produces <paramref name="targetLevelCount"/> LOD levels with decreasing triangle budgets.
        /// Simplified index buffers are stored in <see cref="LodLevel.MeshData"/> as <c>List&lt;uint&gt;</c>.
        /// </summary>
        public List<LodLevel> GenerateLevels(
            List<Vector3> vertices,
            List<uint> indices,
            int targetLevelCount = 4,
            LodGenerationQuality quality = LodGenerationQuality.Balanced)
        {
            ArgumentNullException.ThrowIfNull(vertices);
            ArgumentNullException.ThrowIfNull(indices);

            float[] reductionFactors = quality switch
            {
                LodGenerationQuality.Fast => new float[] { 1.0f, 0.5f, 0.25f, 0.125f },
                LodGenerationQuality.Balanced => new float[] { 1.0f, 0.65f, 0.4f, 0.2f, 0.1f },
                LodGenerationQuality.HighQuality => new float[] { 1.0f, 0.75f, 0.55f, 0.35f, 0.2f, 0.1f },
                _ => new float[] { 1.0f, 0.5f, 0.25f, 0.125f }
            };

            int maxLevels = Math.Min(targetLevelCount, reductionFactors.Length);
            var levels = new List<LodLevel>();
            float thresholdStep = 1.0f / maxLevels;

            var currentIndices = new List<uint>(indices);
            var workingVertices = new List<Vector3>(vertices);

            for (int i = 0; i < maxLevels; i++)
            {
                float factor = reductionFactors[i];
                int targetTriangles = Math.Max(1, (int)(indices.Count / 3 * factor));

                if (i > 0)
                    currentIndices = QuadricMeshSimplifier.Simplify(workingVertices, currentIndices, targetTriangles, quality);

                levels.Add(new LodLevel
                {
                    Level = i,
                    ScreenCoverageThreshold = 1.0f - (i * thresholdStep),
                    TriangleCount = currentIndices.Count / 3,
                    VertexCount = workingVertices.Count,
                    QualityFactor = factor,
                    MeshData = new List<uint>(currentIndices)
                });
            }

            ComputeDistances(levels);
            return levels;
        }

        private void ComputeDistances(List<LodLevel> levels)
        {
            if (levels.Count == 0)
                return;

            for (int i = 0; i < levels.Count; i++)
            {
                float factor = levels[i].QualityFactor;
                levels[i].MinDistance = factor > 0.5f ? 0 : (1.0f - factor) * 100;
                levels[i].MaxDistance = factor < 0.5f ? float.MaxValue : (1.0f - factor + 0.3f) * 100;
                levels[i].FadeInDistance = levels[i].MinDistance * 0.9f;
                levels[i].FadeOutDistance = levels[i].MaxDistance * 1.1f;
            }
        }
    }

    /// <summary>
    /// Industrial mesh simplification via quadric error metrics (Garland &amp; Heckbert).
    /// Collapses lowest-cost edges until the triangle budget is met.
    /// </summary>
    public static class QuadricMeshSimplifier
    {
        /// <summary>
        /// Returns a new index buffer with at most <paramref name="targetTriangles"/> triangles.
        /// Vertex positions may be updated in-place when pairs collapse to an optimal point.
        /// </summary>
        public static List<uint> Simplify(
            List<Vector3> vertices,
            List<uint> indices,
            int targetTriangles,
            LodGenerationQuality quality = LodGenerationQuality.Balanced)
        {
            ArgumentNullException.ThrowIfNull(vertices);
            ArgumentNullException.ThrowIfNull(indices);

            int currentTriangles = indices.Count / 3;
            if (currentTriangles <= targetTriangles || vertices.Count < 3)
                return new List<uint>(indices);

            var quadrics = BuildVertexQuadrics(vertices, indices);
            var faces = new List<(int A, int B, int C)>(currentTriangles);
            for (int i = 0; i + 2 < indices.Count; i += 3)
                faces.Add(((int)indices[i], (int)indices[i + 1], (int)indices[i + 2]));

            int maxCollapses = quality switch
            {
                LodGenerationQuality.Fast => (currentTriangles - targetTriangles) / 2 + 1,
                LodGenerationQuality.HighQuality => (currentTriangles - targetTriangles) * 2 + 1,
                _ => currentTriangles - targetTriangles + 1
            };

            for (int iter = 0; iter < maxCollapses && faces.Count > targetTriangles; iter++)
            {
                if (!TryFindBestEdge(vertices, faces, quadrics, out int vKeep, out int vRemove, out Vector3 optimal))
                    break;

                // Collapse vRemove → vKeep at optimal position.
                vertices[vKeep] = optimal;
                AddQuadric(quadrics[vKeep], quadrics[vRemove]);

                for (int f = faces.Count - 1; f >= 0; f--)
                {
                    var (a, b, c) = faces[f];
                    if (a == vRemove)
                        a = vKeep;
                    if (b == vRemove)
                        b = vKeep;
                    if (c == vRemove)
                        c = vKeep;

                    if (a == b || b == c || a == c)
                    {
                        faces.RemoveAt(f);
                        continue;
                    }

                    faces[f] = (a, b, c);
                }

                // Recompute quadric for the kept vertex from remaining adjacent faces.
                ClearQuadric(quadrics[vKeep]);
                for (int f = 0; f < faces.Count; f++)
                {
                    var (a, b, c) = faces[f];
                    if (a != vKeep && b != vKeep && c != vKeep)
                        continue;
                    AccumulateFaceQuadric(vertices, a, b, c, quadrics);
                }
            }

            var result = new List<uint>(faces.Count * 3);
            for (int i = 0; i < faces.Count; i++)
            {
                result.Add((uint)faces[i].A);
                result.Add((uint)faces[i].B);
                result.Add((uint)faces[i].C);
            }
            return result;
        }

        private static float[][] BuildVertexQuadrics(List<Vector3> vertices, List<uint> indices)
        {
            var q = new float[vertices.Count][];
            for (int i = 0; i < vertices.Count; i++)
                q[i] = new float[10]; // symmetric 4x4 packed: q11,q12,q13,q14,q22,q23,q24,q33,q34,q44

            for (int i = 0; i + 2 < indices.Count; i += 3)
                AccumulateFaceQuadric(vertices, (int)indices[i], (int)indices[i + 1], (int)indices[i + 2], q);

            return q;
        }

        private static void AccumulateFaceQuadric(List<Vector3> vertices, int a, int b, int c, float[][] quadrics)
        {
            if ((uint)a >= (uint)vertices.Count || (uint)b >= (uint)vertices.Count || (uint)c >= (uint)vertices.Count)
                return;

            Vector3 n = Vector3.Cross(vertices[b] - vertices[a], vertices[c] - vertices[a]);
            float len = n.Length();
            if (len < 1e-8f)
                return;
            n /= len;
            float d = -Vector3.Dot(n, vertices[a]);

            // Plane (n, d): quadric = p * p^T
            AddPlane(quadrics[a], n.X, n.Y, n.Z, d);
            AddPlane(quadrics[b], n.X, n.Y, n.Z, d);
            AddPlane(quadrics[c], n.X, n.Y, n.Z, d);
        }

        private static void AddPlane(float[] q, float a, float b, float c, float d)
        {
            q[0] += a * a;
            q[1] += a * b;
            q[2] += a * c;
            q[3] += a * d;
            q[4] += b * b;
            q[5] += b * c;
            q[6] += b * d;
            q[7] += c * c;
            q[8] += c * d;
            q[9] += d * d;
        }

        private static void AddQuadric(float[] dst, float[] src)
        {
            for (int i = 0; i < 10; i++)
                dst[i] += src[i];
        }

        private static void ClearQuadric(float[] q)
        {
            Array.Clear(q);
        }

        private static float VertexError(float[] q, Vector3 v)
        {
            float x = v.X, y = v.Y, z = v.Z;
            // v^T Q v for symmetric Q
            return q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x
                 + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y
                 + q[7] * z * z + 2 * q[8] * z
                 + q[9];
        }

        private static bool TryFindBestEdge(
            List<Vector3> vertices,
            List<(int A, int B, int C)> faces,
            float[][] quadrics,
            out int vKeep,
            out int vRemove,
            out Vector3 optimal)
        {
            vKeep = vRemove = -1;
            optimal = Vector3.Zero;
            float bestCost = float.MaxValue;

            var edges = new HashSet<long>();
            for (int i = 0; i < faces.Count; i++)
            {
                var (a, b, c) = faces[i];
                edges.Add(EdgeKey(a, b));
                edges.Add(EdgeKey(b, c));
                edges.Add(EdgeKey(c, a));
            }

            foreach (long key in edges)
            {
                int i = (int)(key >> 32);
                int j = (int)(key & 0xffffffff);
                if ((uint)i >= (uint)vertices.Count || (uint)j >= (uint)vertices.Count)
                    continue;

                float[] q = (float[])quadrics[i].Clone();
                AddQuadric(q, quadrics[j]);

                // Candidate contraction points: endpoints + midpoint (stable industrial subset).
                Vector3 mid = (vertices[i] + vertices[j]) * 0.5f;
                Span<Vector3> candidates = stackalloc Vector3[3];
                candidates[0] = vertices[i];
                candidates[1] = vertices[j];
                candidates[2] = mid;

                for (int c = 0; c < candidates.Length; c++)
                {
                    float cost = VertexError(q, candidates[c]);
                    // Prefer shorter edges slightly for numerical stability.
                    cost += Vector3.DistanceSquared(vertices[i], vertices[j]) * 1e-4f;
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        // Keep the endpoint closer to the optimal candidate when possible.
                        float di = Vector3.DistanceSquared(candidates[c], vertices[i]);
                        float dj = Vector3.DistanceSquared(candidates[c], vertices[j]);
                        if (di <= dj)
                        {
                            vKeep = i;
                            vRemove = j;
                        }
                        else
                        {
                            vKeep = j;
                            vRemove = i;
                        }
                        optimal = candidates[c];
                    }
                }
            }

            return vKeep >= 0 && vRemove >= 0 && vKeep != vRemove;
        }

        private static long EdgeKey(int a, int b)
        {
            if (a > b)
                (a, b) = (b, a);
            return ((long)a << 32) | (uint)b;
        }
    }

    // =========================================================================
    // LOD MANAGER
    // =========================================================================

    /// <summary>
    /// Manages all LOD groups in a scene, performing per-frame LOD selection
    /// based on camera position and viewport parameters.
    /// </summary>
    public class LodManager
    {
        private readonly Dictionary<int, LodGroup> _groups = new();
        private readonly List<LodGroup> _allGroups = new();
        private readonly object _lock = new();

        public int GroupCount { get { lock (_lock) { return _groups.Count; } } }
        public IReadOnlyList<LodGroup> Groups { get { lock (_lock) { return _allGroups.AsReadOnly(); } } }

        public LodGroup RegisterGroup(string name, float referenceRadius, LodCalculationMode mode = LodCalculationMode.ScreenCoverageWithHysteresis)
        {
            var group = new LodGroup
            {
                Name = name,
                ReferenceRadius = referenceRadius,
                CalculationMode = mode
            };

            lock (_lock)
            {
                _groups[group.Id] = group;
                _allGroups.Add(group);
            }

            return group;
        }

        public void UnregisterGroup(int groupId)
        {
            lock (_lock)
            {
                if (_groups.TryGetValue(groupId, out var group))
                {
                    _groups.Remove(groupId);
                    _allGroups.Remove(group);
                }
            }
        }

        public LodGroup? GetGroup(int groupId)
        {
            lock (_lock)
            { return _groups.TryGetValue(groupId, out var g) ? g : null; }
        }

        public void UpdateAll(Vector3 cameraPosition, float fovRadians, float viewportHeight)
        {
            lock (_lock)
            {
                foreach (var group in _allGroups)
                {
                    float distance = Vector3.Distance(cameraPosition, group.LocalReferencePoint);
                    float coverage = group.ComputeScreenCoverage(group.LocalReferencePoint, group.ReferenceRadius, cameraPosition, fovRadians, viewportHeight);
                    group.SelectLevel(coverage, distance);
                }
            }
        }

        public List<(LodGroup Group, LodLevel SelectedLevel)> GetAllSelectedLevels()
        {
            var results = new List<(LodGroup, LodLevel)>();
            lock (_lock)
            {
                foreach (var group in _allGroups)
                {
                    var level = group.Levels.FirstOrDefault(l => l.Level == group.LastSelectedLevel);
                    if (level != null)
                        results.Add((group, level));
                }
            }
            return results;
        }

        public LodStatistics GetStatistics()
        {
            lock (_lock)
            {
                int totalVertices = 0;
                int totalTriangles = 0;
                int totalLevels = 0;
                int activeGroups = 0;

                foreach (var group in _allGroups)
                {
                    activeGroups++;
                    totalLevels += group.Levels.Count;
                    var selected = group.Levels.FirstOrDefault(l => l.Level == group.LastSelectedLevel);
                    if (selected != null)
                    {
                        totalVertices += selected.VertexCount;
                        totalTriangles += selected.TriangleCount;
                    }
                }

                return new LodStatistics
                {
                    TotalGroups = activeGroups,
                    TotalLevels = totalLevels,
                    TotalVertices = totalVertices,
                    TotalTriangles = totalTriangles,
                    AverageTrianglesPerGroup = activeGroups > 0 ? totalTriangles / (float)activeGroups : 0
                };
            }
        }

        public void Clear()
        {
            lock (_lock)
            {
                _groups.Clear();
                _allGroups.Clear();
            }
        }
    }

    /// <summary>LOD system statistics.</summary>
    public class LodStatistics
    {
        public int TotalGroups { get; set; }
        public int TotalLevels { get; set; }
        public int TotalVertices { get; set; }
        public int TotalTriangles { get; set; }
        public float AverageTrianglesPerGroup { get; set; }
    }

    /// <summary>Simple bounding box for LOD calculations (single precision).</summary>
    public struct BoundingBox3D
    {
        public Vector3 Min;
        public Vector3 Max;
        public BoundingBox3D(Vector3 min, Vector3 max) { Min = min; Max = max; }
        public Vector3 Center => (Min + Max) * 0.5f;
        public Vector3 Extents => (Max - Min) * 0.5f;
    }
}
