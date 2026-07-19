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
    /// Manages the radiance cascade hierarchy for world-space global illumination.
    /// Implements multi-level cascades with temporal accumulation, spatial filtering,
    /// and importance-driven adaptive allocation.
    /// </summary>
    public class RadianceCascadesManager
    {
        private CascadeConfig _config;
        private CascadeLevelConfig[] _levels;
        private Vector3[][] _cascadeData;
        private Vector3[][] _temporalHistory;
        private float[][] _varianceData;
        private int[][] _sampleCounts;
        private int[] _resolutionPerLevel;
        private float[] _importancePerLevel;
        private float[] _timeAllocationPerLevel;
        private int _totalLevels;
        private int _frameIndex;
        private bool _isInitialized;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>Current cascade configuration.</summary>
        public CascadeConfig Config => _config;
        /// <summary>Number of active cascade levels.</summary>
        public int ActiveLevelCount => _totalLevels;
        /// <summary>Is the manager initialized.</summary>
        public bool IsInitialized => _isInitialized;

        /// <summary>
        /// Initializes the cascade hierarchy with configurable levels.
        /// </summary>
        public void Initialize(CascadeConfig config)
        {
            _config = config;
            _totalLevels = config.NumLevels;
            _levels = new CascadeLevelConfig[_totalLevels];
            _cascadeData = new Vector3[_totalLevels][];
            _temporalHistory = new Vector3[_totalLevels][];
            _varianceData = new float[_totalLevels][];
            _sampleCounts = new int[_totalLevels][];
            _resolutionPerLevel = new int[_totalLevels];
            _importancePerLevel = new float[_totalLevels];
            _timeAllocationPerLevel = new float[_totalLevels];

            for (int i = 0; i < _totalLevels; i++)
            {
                int resolution = ComputeLevelResolution(i);
                _resolutionPerLevel[i] = resolution;
                int pixelCount = resolution * resolution * 6;

                _levels[i] = new CascadeLevelConfig
                {
                    LevelIndex = i,
                    Resolution = resolution,
                    MaxTraceDistance = ComputeMaxTraceDistance(i),
                    MinTraceDistance = ComputeMinTraceDistance(i),
                    RaysPerTexel = ComputeRaysPerTexel(i),
                    AngularResolution = ComputeAngularResolution(i),
                    IsActive = true,
                    ImportanceWeight = 1.0f,
                    UpdateFrequency = ComputeUpdateFrequency(i),
                    FilterKernelSize = config.SpatialFilterRadius
                };

                _cascadeData[i] = new Vector3[pixelCount];
                _temporalHistory[i] = new Vector3[pixelCount];
                _varianceData[i] = new float[pixelCount];
                _sampleCounts[i] = new int[pixelCount];
                _importancePerLevel[i] = 1.0f;
            }

            ComputeBudgetAllocations();
            _isInitialized = true;
        }

        private int ComputeLevelResolution(int level)
        {
            int baseRes = (int)_config.BaseResolution;
            return Math.Max(16, baseRes >> level);
        }

        private float ComputeMaxTraceDistance(int level)
        {
            return _config.DistanceScale * MathF.Pow(2.0f, level + 1);
        }

        private float ComputeMinTraceDistance(int level)
        {
            if (level == 0)
                return 0.0f;
            return _config.DistanceScale * MathF.Pow(2.0f, level);
        }

        private int ComputeRaysPerTexel(int level)
        {
            return Math.Max(1, 8 >> level);
        }

        private int ComputeAngularResolution(int level)
        {
            return MathMax(4, 64 >> level);
        }

        private int MathMax(int a, int b) => a > b ? a : b;

        private int ComputeUpdateFrequency(int level)
        {
            return Math.Max(1, 1 << level);
        }

        /// <summary>
        /// Allocates cascade resources based on budget constraints.
        /// </summary>
        public CascadeBudget AllocateCascadeResources(long totalMemoryBudget, float totalTimeBudget)
        {
            var timePerLevel = new float[_totalLevels];
            var memoryPerLevel = new long[_totalLevels];
            var raysPerLevel = new int[_totalLevels];
            var resolutionPerLevel = new int[_totalLevels];

            switch (_config.AllocationStrategy)
            {
                case CascadeAllocationStrategy.Uniform:
                    AllocateUniform(ref timePerLevel, ref memoryPerLevel,
                        ref raysPerLevel, ref resolutionPerLevel,
                        totalMemoryBudget, totalTimeBudget);
                    break;
                case CascadeAllocationStrategy.Logarithmic:
                    AllocateLogarithmic(ref timePerLevel, ref memoryPerLevel,
                        ref raysPerLevel, ref resolutionPerLevel,
                        totalMemoryBudget, totalTimeBudget);
                    break;
                case CascadeAllocationStrategy.ImportanceDriven:
                    AllocateImportanceDriven(ref timePerLevel, ref memoryPerLevel,
                        ref raysPerLevel, ref resolutionPerLevel,
                        totalMemoryBudget, totalTimeBudget);
                    break;
                case CascadeAllocationStrategy.Adaptive:
                    AllocateAdaptive(ref timePerLevel, ref memoryPerLevel,
                        ref raysPerLevel, ref resolutionPerLevel,
                        totalMemoryBudget, totalTimeBudget);
                    break;
                case CascadeAllocationStrategy.Fixed:
                    AllocateFixed(ref timePerLevel, ref memoryPerLevel,
                        ref raysPerLevel, ref resolutionPerLevel,
                        totalMemoryBudget, totalTimeBudget);
                    break;
            }

            long totalMemoryUsed = 0;
            float totalTimeUsed = 0;
            for (int i = 0; i < _totalLevels; i++)
            {
                totalMemoryUsed += memoryPerLevel[i];
                totalTimeUsed += timePerLevel[i];
            }

            return new CascadeBudget
            {
                TotalTimeBudgetMs = totalTimeBudget,
                TotalMemoryBudgetBytes = totalMemoryBudget,
                TimePerLevelMs = timePerLevel,
                MemoryPerLevelBytes = memoryPerLevel,
                RaysPerLevel = raysPerLevel,
                ResolutionPerLevel = resolutionPerLevel,
                WithinBudget = totalMemoryUsed <= totalMemoryBudget && totalTimeUsed <= totalTimeBudget
            };
        }

        private void AllocateUniform(ref float[] timePerLevel, ref long[] memoryPerLevel,
            ref int[] raysPerLevel, ref int[] resolutionPerLevel,
            long totalMemory, float totalTime)
        {
            float timePer = totalTime / _totalLevels;
            long memPer = totalMemory / _totalLevels;
            for (int i = 0; i < _totalLevels; i++)
            {
                timePerLevel[i] = timePer;
                memoryPerLevel[i] = memPer;
                raysPerLevel[i] = _levels[i].RaysPerTexel;
                resolutionPerLevel[i] = _levels[i].Resolution;
            }
        }

        private void AllocateLogarithmic(ref float[] timePerLevel, ref long[] memoryPerLevel,
            ref int[] raysPerLevel, ref int[] resolutionPerLevel,
            long totalMemory, float totalTime)
        {
            float totalWeight = 0;
            for (int i = 0; i < _totalLevels; i++)
                totalWeight += 1.0f / (i + 1);
            for (int i = 0; i < _totalLevels; i++)
            {
                float weight = (1.0f / (i + 1)) / totalWeight;
                timePerLevel[i] = totalTime * weight;
                memoryPerLevel[i] = (long)(totalMemory * weight);
                raysPerLevel[i] = Math.Max(1, _levels[i].RaysPerTexel);
                resolutionPerLevel[i] = _levels[i].Resolution;
            }
        }

        private void AllocateImportanceDriven(ref float[] timePerLevel, ref long[] memoryPerLevel,
            ref int[] raysPerLevel, ref int[] resolutionPerLevel,
            long totalMemory, float totalTime)
        {
            float totalImportance = 0;
            for (int i = 0; i < _totalLevels; i++)
                totalImportance += _importancePerLevel[i];
            for (int i = 0; i < _totalLevels; i++)
            {
                float weight = _importancePerLevel[i] / MathF.Max(0.001f, totalImportance);
                timePerLevel[i] = totalTime * weight;
                memoryPerLevel[i] = (long)(totalMemory * weight);
                raysPerLevel[i] = Math.Max(1, (int)(_levels[i].RaysPerTexel * weight * _totalLevels));
                resolutionPerLevel[i] = _levels[i].Resolution;
            }
        }

        private void AllocateAdaptive(ref float[] timePerLevel, ref long[] memoryPerLevel,
            ref int[] raysPerLevel, ref int[] resolutionPerLevel,
            long totalMemory, float totalTime)
        {
            for (int i = 0; i < _totalLevels; i++)
            {
                float levelWeight = (1.0f - (float)i / _totalLevels) * _importancePerLevel[i];
                timePerLevel[i] = totalTime * levelWeight / _totalLevels;
                memoryPerLevel[i] = (long)(totalMemory * levelWeight / _totalLevels);
                raysPerLevel[i] = Math.Max(1, (int)(_levels[i].RaysPerTexel * levelWeight));
                resolutionPerLevel[i] = (int)(_levels[i].Resolution * MathF.Sqrt(levelWeight));
            }
        }

        private void AllocateFixed(ref float[] timePerLevel, ref long[] memoryPerLevel,
            ref int[] raysPerLevel, ref int[] resolutionPerLevel,
            long totalMemory, float totalTime)
        {
            int[] fixedResolutions = { 256, 128, 64, 32, 16, 8, 4, 2 };
            for (int i = 0; i < _totalLevels; i++)
            {
                int resIdx = Math.Min(i, fixedResolutions.Length - 1);
                resolutionPerLevel[i] = fixedResolutions[resIdx];
                raysPerLevel[i] = Math.Max(1, 8 >> i);
                timePerLevel[i] = totalTime / _totalLevels;
                memoryPerLevel[i] = totalMemory / _totalLevels;
            }
        }

        /// <summary>
        /// Renders a single cascade level by dispatching compute shaders.
        /// </summary>
        public void RenderCascadesLevel(int level, GBuffer gbuffer, CameraState camera,
            List<LightConfig> lights, RandomNumberGenerator rng)
        {
            if (level < 0 || level >= _totalLevels)
                return;
            if (!_levels[level].IsActive)
                return;

            int resolution = _resolutionPerLevel[level];
            float maxDist = _levels[level].MaxTraceDistance;
            float minDist = _levels[level].MinTraceDistance;
            int raysPerTexel = _levels[level].RaysPerTexel;

            for (int face = 0; face < 6; face++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    for (int y = 0; y < resolution; y++)
                    {
                        Vector3 radiance = Vector3.Zero;
                        Vector3 direction = ComputeCascadeDirection(x, y, face, resolution);

                        for (int r = 0; r < raysPerTexel; r++)
                        {
                            float jitter = rng.NextFloat();
                            float rayAngle = (float)r / raysPerTexel + jitter / raysPerTexel;
                            Vector3 rayDir = RotateDirection(direction, rayAngle, face);

                            Vector3 rayRadiance = TraceCascadeRay(
                                direction, rayDir, minDist, maxDist, gbuffer, lights, rng);
                            radiance += rayRadiance;
                        }

                        radiance /= raysPerTexel;
                        int idx = ComputeCascadeIndex(x, y, face, resolution);
                        _cascadeData[level][idx] = radiance;
                    }
                }
            }
        }

        private Vector3 ComputeCascadeDirection(int x, int y, int face, int resolution)
        {
            float u = (x + 0.5f) / resolution * 2.0f - 1.0f;
            float v = (y + 0.5f) / resolution * 2.0f - 1.0f;
            return face switch
            {
                0 => Vector3.Normalize(new Vector3(1, v, -u)),
                1 => Vector3.Normalize(new Vector3(-1, v, u)),
                2 => Vector3.Normalize(new Vector3(u, 1, -v)),
                3 => Vector3.Normalize(new Vector3(u, -1, v)),
                4 => Vector3.Normalize(new Vector3(u, v, 1)),
                5 => Vector3.Normalize(new Vector3(-u, v, -1)),
                _ => Vector3.UnitY
            };
        }

        private Vector3 RotateDirection(Vector3 direction, float angle, int face)
        {
            float cosA = MathF.Cos(angle * TWO_PI);
            float sinA = MathF.Sin(angle * TWO_PI);
            Vector3 tangent = GetTangentForFace(face);
            Vector3 bitangent = Vector3.Cross(direction, tangent);
            return Vector3.Normalize(direction * cosA + tangent * sinA + bitangent * cosA * 0.5f);
        }

        private Vector3 GetTangentForFace(int face) => face switch
        {
            0 => Vector3.UnitY,
            1 => Vector3.UnitY,
            2 => Vector3.UnitX,
            3 => Vector3.UnitX,
            4 => Vector3.UnitY,
            5 => Vector3.UnitY,
            _ => Vector3.UnitX
        };

        private int ComputeCascadeIndex(int x, int y, int face, int resolution)
        {
            return (face * resolution * resolution) + (y * resolution + x);
        }

        private Vector3 TraceCascadeRay(Vector3 origin, Vector3 direction, float minDist,
            float maxDist, GBuffer gbuffer, List<LightConfig> lights, RandomNumberGenerator rng)
        {
            Vector3 totalRadiance = Vector3.Zero;
            float stepSize = (maxDist - minDist) / 32.0f;
            Vector3 currentPos = origin + direction * minDist;

            for (int step = 0; step < 32; step++)
            {
                Vector3 samplePos = currentPos + direction * stepSize * 0.5f;
                int pixX = (int)(MathF.Atan2(direction.X, -direction.Z) * INV_PI * 0.5f * gbuffer.Width);
                int pixY = (int)(MathF.Acos(Math.Clamp(direction.Y, -1, 1)) * INV_PI * gbuffer.Height);
                pixX = Math.Abs(pixX) % gbuffer.Width;
                pixY = Math.Abs(pixY) % gbuffer.Height;

                if (pixX >= 0 && pixX < gbuffer.Width && pixY >= 0 && pixY < gbuffer.Height)
                {
                    int idx = gbuffer.GetIndex(pixX, pixY);
                    float geoDepth = gbuffer.Depth[idx];
                    float sampleDepth = samplePos.Length() * 0.1f;

                    if (geoDepth > 0 && MathF.Abs(geoDepth - sampleDepth) < stepSize)
                    {
                        Vector3 albedo = gbuffer.Albedo[idx];
                        Vector3 normal = gbuffer.Normals[idx];
                        foreach (var light in lights)
                        {
                            Vector3 lightDir = Vector3.Normalize(light.Position - samplePos);
                            float NdotL = MathF.Max(0, Vector3.Dot(normal, lightDir));
                            totalRadiance += albedo * light.Color * light.Intensity * NdotL;
                        }
                        break;
                    }
                }

                currentPos += direction * stepSize;
            }

            return totalRadiance;
        }

        /// <summary>
        /// Propagates cascades bottom-up by merging adjacent levels.
        /// </summary>
        public void PropagateCascades()
        {
            for (int level = _totalLevels - 1; level > 0; level--)
            {
                int childRes = _resolutionPerLevel[level];
                int parentRes = _resolutionPerLevel[level - 1];

                for (int face = 0; face < 6; face++)
                {
                    for (int px = 0; px < parentRes; px++)
                    {
                        for (int py = 0; py < parentRes; py++)
                        {
                            Vector3 mergedRadiance = Vector3.Zero;
                            int samples = 0;

                            for (int dx = 0; dx < 2; dx++)
                            {
                                for (int dy = 0; dy < 2; dy++)
                                {
                                    int cx = px * 2 + dx;
                                    int cy = py * 2 + dy;
                                    if (cx < childRes && cy < childRes)
                                    {
                                        int childIdx = ComputeCascadeIndex(cx, cy, face, childRes);
                                        mergedRadiance += _cascadeData[level][childIdx];
                                        samples++;
                                    }
                                }
                            }

                            if (samples > 0)
                            {
                                mergedRadiance /= samples;
                                int parentIdx = ComputeCascadeIndex(px, py, face, parentRes);
                                _cascadeData[level - 1][parentIdx] = Vector3.Lerp(
                                    _cascadeData[level - 1][parentIdx],
                                    mergedRadiance,
                                    0.5f);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Merges adjacent cascades for seamless transitions.
        /// </summary>
        public void MergeAdjacentCascades()
        {
            for (int level = 0; level < _totalLevels - 1; level++)
            {
                int currentRes = _resolutionPerLevel[level];
                int nextRes = _resolutionPerLevel[level + 1];

                for (int face = 0; face < 6; face++)
                {
                    for (int x = 0; x < currentRes; x++)
                    {
                        for (int y = 0; y < currentRes; y++)
                        {
                            int idx = ComputeCascadeIndex(x, y, face, currentRes);
                            int nx = x / 2;
                            int ny = y / 2;
                            if (nx < nextRes && ny < nextRes)
                            {
                                int nextIdx = ComputeCascadeIndex(nx, ny, face, nextRes);
                                _cascadeData[level][idx] = Vector3.Lerp(
                                    _cascadeData[level][idx],
                                    _cascadeData[level + 1][nextIdx],
                                    0.3f);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Performs temporal accumulation across frames.
        /// </summary>
        public void TemporalAccumulation(float blendFactor)
        {
            for (int level = 0; level < _totalLevels; level++)
            {
                int pixelCount = _cascadeData[level].Length;
                for (int i = 0; i < pixelCount; i++)
                {
                    Vector3 current = _cascadeData[level][i];
                    Vector3 history = _temporalHistory[level][i];

                    float historyWeight = MathF.Min(1.0f, blendFactor);
                    if (_sampleCounts[level][i] < 2)
                        historyWeight = 0;

                    _cascadeData[level][i] = Vector3.Lerp(current, history, historyWeight);
                    _temporalHistory[level][i] = _cascadeData[level][i];
                    _sampleCounts[level][i] = Math.Min(_sampleCounts[level][i] + 1, 256);

                    Vector3 diff = current - _temporalHistory[level][i];
                    _varianceData[level][i] = _varianceData[level][i] * 0.95f + diff.LengthSquared() * 0.05f;
                }
            }
        }

        /// <summary>
        /// Applies spatial bilateral filtering within cascades.
        /// </summary>
        public void SpatialFilter(int radius)
        {
            for (int level = 0; level < _totalLevels; level++)
            {
                int res = _resolutionPerLevel[level];
                Vector3[] filtered = new Vector3[_cascadeData[level].Length];
                Array.Copy(_cascadeData[level], filtered, _cascadeData[level].Length);

                for (int face = 0; face < 6; face++)
                {
                    for (int x = radius; x < res - radius; x++)
                    {
                        for (int y = radius; y < res - radius; y++)
                        {
                            Vector3 sum = Vector3.Zero;
                            float weightSum = 0;
                            int idx = ComputeCascadeIndex(x, y, face, res);
                            Vector3 centerColor = _cascadeData[level][idx];

                            for (int dx = -radius; dx <= radius; dx++)
                            {
                                for (int dy = -radius; dy <= radius; dy++)
                                {
                                    int nx = x + dx;
                                    int ny = y + dy;
                                    int nIdx = ComputeCascadeIndex(nx, ny, face, res);
                                    Vector3 neighborColor = _cascadeData[level][nIdx];

                                    float spatialWeight = MathF.Exp(-(dx * dx + dy * dy) / (2.0f * radius * radius));
                                    float colorDist = (neighborColor - centerColor).LengthSquared();
                                    float colorWeight = MathF.Exp(-colorDist * 10.0f);
                                    float weight = spatialWeight * colorWeight;

                                    sum += neighborColor * weight;
                                    weightSum += weight;
                                }
                            }

                            filtered[idx] = weightSum > 0 ? sum / weightSum : centerColor;
                        }
                    }
                }

                Array.Copy(filtered, _cascadeData[level], filtered.Length);
            }
        }

        /// <summary>
        /// Computes the temporal blend factor based on motion and history confidence.
        /// </summary>
        public float ComputeTemporalBlendFactor(int level, Vector2 velocity)
        {
            float motionMagnitude = velocity.Length();
            float motionFactor = MathF.Exp(-motionMagnitude * 10.0f);
            float historyFactor = MathF.Min(1.0f, (float)_sampleCounts[level][0] / 16.0f);
            return _config.TemporalBlendFactor * motionFactor * historyFactor;
        }

        /// <summary>
        /// Handles disocclusion by detecting large depth changes.
        /// </summary>
        public void HandleDisocclusion(GBuffer currentGBuffer, GBuffer previousGBuffer, float threshold)
        {
            if (previousGBuffer == null)
                return;

            for (int level = 0; level < _totalLevels; level++)
            {
                int res = _resolutionPerLevel[level];
                for (int face = 0; face < 6; face++)
                {
                    for (int x = 0; x < res; x++)
                    {
                        for (int y = 0; y < res; y++)
                        {
                            int idx = ComputeCascadeIndex(x, y, face, res);
                            int gbufX = Math.Clamp((int)((float)x / res * currentGBuffer.Width), 0, currentGBuffer.Width - 1);
                            int gbufY = Math.Clamp((int)((float)y / res * currentGBuffer.Height), 0, currentGBuffer.Height - 1);

                            float currentDepth = currentGBuffer.Depth[currentGBuffer.GetIndex(gbufX, gbufY)];
                            float prevDepth = previousGBuffer.Depth[previousGBuffer.GetIndex(gbufX, gbufY)];

                            if (MathF.Abs(currentDepth - prevDepth) > threshold * currentDepth)
                            {
                                _temporalHistory[level][idx] = Vector3.Zero;
                                _sampleCounts[level][idx] = 0;
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Adaptive level allocation based on importance metrics.
        /// </summary>
        public void AdaptiveLevelAllocation(float[] importanceScores)
        {
            if (importanceScores == null || importanceScores.Length < _totalLevels)
                return;

            float totalImportance = 0;
            for (int i = 0; i < _totalLevels; i++)
            {
                _importancePerLevel[i] = importanceScores[i];
                totalImportance += importanceScores[i];
            }

            for (int i = 0; i < _totalLevels; i++)
            {
                float normalizedImportance = _importancePerLevel[i] / MathF.Max(0.001f, totalImportance);
                _levels[i] = _levels[i] with
                {
                    ImportanceWeight = normalizedImportance,
                    IsActive = normalizedImportance > 0.05f
                };
            }
        }

        /// <summary>
        /// Computes the cascade budget allocation.
        /// </summary>
        public CascadeBudget ComputeCascadeBudget(float totalTimeBudget, long totalMemoryBudget)
        {
            return AllocateCascadeResources(totalMemoryBudget, totalTimeBudget);
        }

        private void ComputeBudgetAllocations()
        {
            float totalTime = _config.TimeBudgetMs;
            for (int i = 0; i < _totalLevels; i++)
            {
                float weight = 1.0f / (i + 1);
                _timeAllocationPerLevel[i] = totalTime * weight;
            }
        }

        /// <summary>
        /// Generates debug visualization of the cascade levels.
        /// </summary>
        public Vector3[,] GenerateCascadeDebugVisualization(int level, int face)
        {
            if (level < 0 || level >= _totalLevels)
                return new Vector3[0, 0];

            int res = _resolutionPerLevel[level];
            var visualization = new Vector3[res, res];

            for (int x = 0; x < res; x++)
                for (int y = 0; y < res; y++)
                {
                    int idx = ComputeCascadeIndex(x, y, face, res);
                    visualization[x, y] = _cascadeData[level][idx];
                }

            return visualization;
        }

        /// <summary>
        /// Returns statistics about the cascade system.
        /// </summary>
        public CascadeStatistics CascadeStatistics()
        {
            long totalMemory = 0;
            int totalActiveLevels = 0;
            float totalImportance = 0;

            for (int i = 0; i < _totalLevels; i++)
            {
                totalMemory += _cascadeData[i].Length * 12;
                if (_levels[i].IsActive)
                    totalActiveLevels++;
                totalImportance += _importancePerLevel[i];
            }

            return new CascadeStatistics
            {
                TotalLevels = _totalLevels,
                ActiveLevels = totalActiveLevels,
                TotalMemoryBytes = totalMemory,
                AverageImportance = totalImportance / _totalLevels,
                TotalTexels = totalMemory / 12,
                FrameIndex = _frameIndex
            };
        }

        /// <summary>
        /// Gets the cascade data for a specific level.
        /// </summary>
        public Vector3[] GetCascadeData(int level)
        {
            if (level < 0 || level >= _totalLevels)
                return Array.Empty<Vector3>();
            return _cascadeData[level];
        }

        /// <summary>
        /// Gets the resolution for a specific cascade level.
        /// </summary>
        public int GetLevelResolution(int level)
        {
            if (level < 0 || level >= _totalLevels)
                return 0;
            return _resolutionPerLevel[level];
        }
    }

    /// <summary>
    /// Statistics about the radiance cascade system.
    /// </summary>
    public record CascadeStatistics
    {
        /// <summary>Total cascade levels configured.</summary>
        public int TotalLevels { get; init; }
        /// <summary>Number of active cascade levels.</summary>
        public int ActiveLevels { get; init; }
        /// <summary>Total memory used by cascades.</summary>
        public long TotalMemoryBytes { get; init; }
        /// <summary>Average importance across all levels.</summary>
        public float AverageImportance { get; init; }
        /// <summary>Total number of texels across all levels.</summary>
        public long TotalTexels { get; init; }
        /// <summary>Current frame index.</summary>
        public int FrameIndex { get; init; }
    }
}
