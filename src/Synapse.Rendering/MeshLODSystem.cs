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
            if (ForceHighestLOD) return Levels.Count > 0 ? Levels[0] : null;
            if (ForceLowestLOD) return Levels.Count > 0 ? Levels[^1] : null;
            if (Levels.Count == 0) return null;

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
            if (distance < 0.001f) return 1.0f;

            float apparentSize = objectRadius / distance;
            float screenSize = apparentSize / MathF.Tan(fovRadians * 0.5f) * viewportHeight;
            return Math.Clamp(screenSize / viewportHeight, 0, 1);
        }

        public float ComputeScreenCoverageFromBounds(BoundingBox3D bounds, Vector3 cameraPosition, float fovRadians, float viewportHeight)
        {
            float distance = Vector3.Distance(cameraPosition, bounds.Center);
            if (distance < 0.001f) return 1.0f;

            float radius = bounds.Extents.Length();
            return ComputeScreenCoverage(bounds.Center, radius, cameraPosition, fovRadians, viewportHeight);
        }
    }

    // =========================================================================
    // LOD GENERATOR
    // =========================================================================

    /// <summary>
    /// Generates LOD levels from a base mesh by reducing triangle count per level.
    /// Uses a lightweight stochastic decimation placeholder (not full QEM/edge collapse).
    /// </summary>
    public class LodGenerator
    {
        private readonly Random _rng = new(42);

        /// <summary>
        /// Produces <paramref name="targetLevelCount"/> LOD levels with decreasing triangle budgets.
        /// </summary>
        public List<LodLevel> GenerateLevels(
            List<Vector3> vertices,
            List<uint> indices,
            int targetLevelCount = 4,
            LodGenerationQuality quality = LodGenerationQuality.Balanced)
        {
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

            for (int i = 0; i < maxLevels; i++)
            {
                float factor = reductionFactors[i];
                int targetTriangles = (int)(indices.Count / 3 * factor);

                List<uint> simplifiedIndices = SimplifyMesh(indices, targetTriangles, quality);

                levels.Add(new LodLevel
                {
                    Level = i,
                    ScreenCoverageThreshold = 1.0f - (i * thresholdStep),
                    TriangleCount = simplifiedIndices.Count / 3,
                    VertexCount = vertices.Count,
                    QualityFactor = factor
                });

                indices = simplifiedIndices;
            }

            ComputeDistances(levels);
            return levels;
        }

        /// <summary>
        /// Stochastic triangle removal until the target count is reached.
        /// Deterministic seed for reproducible tests; replace with QEM when production mesh IO is wired.
        /// </summary>
        private List<uint> SimplifyMesh(List<uint> indices, int targetTriangles, LodGenerationQuality quality)
        {
            int currentTriangles = indices.Count / 3;
            if (currentTriangles <= targetTriangles)
                return new List<uint>(indices);

            int trianglesToRemove = currentTriangles - targetTriangles;
            var result = new List<uint>(indices);
            int maxIterations = quality switch
            {
                LodGenerationQuality.Fast => trianglesToRemove / 2,
                LodGenerationQuality.Balanced => trianglesToRemove,
                LodGenerationQuality.HighQuality => trianglesToRemove * 2,
                _ => trianglesToRemove
            };

            for (int iter = 0; iter < maxIterations && result.Count / 3 > targetTriangles; iter++)
            {
                int triIdx = _rng.Next(0, result.Count / 3) * 3;
                result.RemoveRange(triIdx, 3);
            }

            return result;
        }

        private void ComputeDistances(List<LodLevel> levels)
        {
            if (levels.Count == 0) return;

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
            lock (_lock) { return _groups.TryGetValue(groupId, out var g) ? g : null; }
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