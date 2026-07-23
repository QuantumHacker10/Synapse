// =============================================================================
// RuntimeQualityManager.cs - G-DNN Engine: Adaptive Quality System
// GDNN.Engine - GDNN.Rendering.Quality
// Runtime quality adaptation based on performance targets and GPU capabilities
// =============================================================================

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;

namespace GDNN.Rendering.Quality
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    /// <summary>Quality preset levels.</summary>
    public enum QualityPreset
    {
        Low,
        Medium,
        High,
        Ultra,
        Cinematic,
        Custom
    }

    /// <summary>Quality adaptation mode.</summary>
    public enum AdaptationMode
    {
        Disabled,
        Static,
        Dynamic,
        AggressiveDynamic
    }

    /// <summary>Quality category.</summary>
    public enum QualityCategory
    {
        Shadows,
        Lighting,
        PostProcessing,
        Geometry,
        Textures,
        Effects,
        GlobalIllumination,
        AntiAliasing
    }

    /// <summary>Performance state detected by the system.</summary>
    public enum PerformanceState
    {
        UnderBudget,
        OnBudget,
        NearBudget,
        OverBudget,
        Critical
    }

    // =========================================================================
    // QUALITY LEVEL
    // =========================================================================

    /// <summary>
    /// Defines a specific quality level with all its associated settings.
    /// </summary>
    [DebuggerDisplay("QualityLevel {Preset}: Scale={ResolutionScale:F2}, Shadows={ShadowQuality}")]
    public class QualityLevel
    {
        public QualityPreset Preset { get; set; }

        // Geometry
        public float ResolutionScale { get; set; } = 1.0f;
        public int MaxLODLevel { get; set; }
        public bool EnableTessellation { get; set; }
        public float TessellationFactor { get; set; } = 1.0f;

        // Shadows
        public int ShadowQuality { get; set; } = 2;
        public int ShadowResolution { get; set; } = 1024;
        public int ShadowCascades { get; set; } = 4;
        public float ShadowDistance { get; set; } = 100.0f;
        public bool EnableSoftShadows { get; set; } = true;
        public int ShadowFilterSamples { get; set; } = 4;

        // Lighting
        public int MaxDynamicLights { get; set; } = 32;
        public int MaxDecals { get; set; } = 64;
        public bool EnableVolumetricLighting { get; set; } = true;
        public bool EnableLightProbes { get; set; } = true;

        // GI
        public bool EnableGlobalIllumination { get; set; } = true;
        public int GICascadeResolution { get; set; } = 128;
        public int GIMaxBounces { get; set; } = 2;
        public bool EnableScreenSpaceGI { get; set; } = true;

        // Post-Processing
        public bool EnableBloom { get; set; } = true;
        public int BloomQuality { get; set; } = 2;
        public bool EnableDOF { get; set; } = true;
        public int DOFQuality { get; set; } = 2;
        public bool EnableMotionBlur { get; set; } = true;
        public int MotionBlurSamples { get; set; } = 8;
        public bool EnableSSAO { get; set; } = true;
        public int SSAOQuality { get; set; } = 2;
        public bool EnableSSR { get; set; } = true;
        public int SSRQuality { get; set; } = 2;
        public bool EnableToneMapping { get; set; } = true;

        // Anti-Aliasing
        public int AAMode { get; set; } = 2;
        public int AASamples { get; set; } = 4;
        public bool EnableTAA { get; set; } = true;

        // Textures
        public int TextureQuality { get; set; } = 2;
        public bool EnableMipmaps { get; set; } = true;
        public int MaxAnisotropy { get; set; } = 8;

        // Effects
        public int ParticleQuality { get; set; } = 2;
        public bool EnableScreenSpaceParticles { get; set; } = true;
        public bool EnableWeatherEffects { get; set; } = true;
    }

    // =========================================================================
    // QUALITY PRESETS
    // =========================================================================

    /// <summary>
    /// Provides predefined quality presets for different hardware tiers.
    /// </summary>
    public static class QualityPresets
    {
        public static QualityLevel Low => new()
        {
            Preset = QualityPreset.Low,
            ResolutionScale = 0.75f,
            MaxLODLevel = 3,
            EnableTessellation = false,
            ShadowQuality = 0,
            ShadowResolution = 512,
            ShadowCascades = 1,
            ShadowDistance = 50.0f,
            EnableSoftShadows = false,
            MaxDynamicLights = 8,
            MaxDecals = 16,
            EnableVolumetricLighting = false,
            EnableGlobalIllumination = false,
            EnableScreenSpaceGI = false,
            EnableBloom = false,
            EnableDOF = false,
            EnableMotionBlur = false,
            EnableSSAO = false,
            EnableSSR = false,
            AAMode = 0,
            AASamples = 1,
            EnableTAA = false,
            TextureQuality = 0,
            MaxAnisotropy = 1,
            ParticleQuality = 0,
            EnableScreenSpaceParticles = false,
            EnableWeatherEffects = false
        };

        public static QualityLevel Medium => new()
        {
            Preset = QualityPreset.Medium,
            ResolutionScale = 0.85f,
            MaxLODLevel = 2,
            EnableTessellation = false,
            ShadowQuality = 1,
            ShadowResolution = 1024,
            ShadowCascades = 2,
            ShadowDistance = 80.0f,
            EnableSoftShadows = true,
            MaxDynamicLights = 16,
            MaxDecals = 32,
            EnableVolumetricLighting = false,
            EnableGlobalIllumination = false,
            EnableScreenSpaceGI = true,
            GICascadeResolution = 64,
            GIMaxBounces = 1,
            EnableBloom = true,
            BloomQuality = 1,
            EnableDOF = false,
            EnableMotionBlur = true,
            MotionBlurSamples = 4,
            EnableSSAO = true,
            SSAOQuality = 1,
            EnableSSR = false,
            AAMode = 2,
            AASamples = 2,
            EnableTAA = true,
            TextureQuality = 1,
            MaxAnisotropy = 4,
            ParticleQuality = 1,
            EnableScreenSpaceParticles = false,
            EnableWeatherEffects = false
        };

        public static QualityLevel High => new()
        {
            Preset = QualityPreset.High,
            ResolutionScale = 1.0f,
            MaxLODLevel = 1,
            EnableTessellation = true,
            TessellationFactor = 0.5f,
            ShadowQuality = 2,
            ShadowResolution = 2048,
            ShadowCascades = 4,
            ShadowDistance = 120.0f,
            EnableSoftShadows = true,
            ShadowFilterSamples = 8,
            MaxDynamicLights = 32,
            MaxDecals = 64,
            EnableVolumetricLighting = true,
            EnableGlobalIllumination = true,
            EnableScreenSpaceGI = true,
            GICascadeResolution = 128,
            GIMaxBounces = 2,
            EnableBloom = true,
            BloomQuality = 2,
            EnableDOF = true,
            DOFQuality = 2,
            EnableMotionBlur = true,
            MotionBlurSamples = 8,
            EnableSSAO = true,
            SSAOQuality = 2,
            EnableSSR = true,
            SSRQuality = 1,
            AAMode = 2,
            AASamples = 4,
            EnableTAA = true,
            TextureQuality = 2,
            MaxAnisotropy = 8,
            ParticleQuality = 2,
            EnableScreenSpaceParticles = true,
            EnableWeatherEffects = true
        };

        public static QualityLevel Ultra => new()
        {
            Preset = QualityPreset.Ultra,
            ResolutionScale = 1.0f,
            MaxLODLevel = 0,
            EnableTessellation = true,
            TessellationFactor = 1.0f,
            ShadowQuality = 3,
            ShadowResolution = 4096,
            ShadowCascades = 4,
            ShadowDistance = 200.0f,
            EnableSoftShadows = true,
            ShadowFilterSamples = 16,
            MaxDynamicLights = 64,
            MaxDecals = 128,
            EnableVolumetricLighting = true,
            EnableGlobalIllumination = true,
            EnableScreenSpaceGI = true,
            GICascadeResolution = 256,
            GIMaxBounces = 4,
            EnableBloom = true,
            BloomQuality = 3,
            EnableDOF = true,
            DOFQuality = 3,
            EnableMotionBlur = true,
            MotionBlurSamples = 12,
            EnableSSAO = true,
            SSAOQuality = 3,
            EnableSSR = true,
            SSRQuality = 3,
            AAMode = 4,
            AASamples = 8,
            EnableTAA = true,
            TextureQuality = 3,
            MaxAnisotropy = 16,
            ParticleQuality = 3,
            EnableScreenSpaceParticles = true,
            EnableWeatherEffects = true
        };

        public static QualityLevel Cinematic => new()
        {
            Preset = QualityPreset.Cinematic,
            ResolutionScale = 1.0f,
            MaxLODLevel = 0,
            EnableTessellation = true,
            TessellationFactor = 2.0f,
            ShadowQuality = 4,
            ShadowResolution = 8192,
            ShadowCascades = 8,
            ShadowDistance = 500.0f,
            EnableSoftShadows = true,
            ShadowFilterSamples = 32,
            MaxDynamicLights = 128,
            MaxDecals = 256,
            EnableVolumetricLighting = true,
            EnableGlobalIllumination = true,
            EnableScreenSpaceGI = true,
            GICascadeResolution = 512,
            GIMaxBounces = 8,
            EnableBloom = true,
            BloomQuality = 4,
            EnableDOF = true,
            DOFQuality = 4,
            EnableMotionBlur = true,
            MotionBlurSamples = 16,
            EnableSSAO = true,
            SSAOQuality = 4,
            EnableSSR = true,
            SSRQuality = 4,
            AAMode = 4,
            AASamples = 16,
            EnableTAA = true,
            TextureQuality = 4,
            MaxAnisotropy = 16,
            ParticleQuality = 4,
            EnableScreenSpaceParticles = true,
            EnableWeatherEffects = true
        };

        public static QualityLevel GetPreset(QualityPreset preset) => preset switch
        {
            QualityPreset.Low => Low,
            QualityPreset.Medium => Medium,
            QualityPreset.High => High,
            QualityPreset.Ultra => Ultra,
            QualityPreset.Cinematic => Cinematic,
            _ => Medium
        };
    }

    // =========================================================================
    // PERFORMANCE MONITOR
    // =========================================================================

    /// <summary>
    /// Monitors frame performance metrics and detects budget violations.
    /// </summary>
    public class PerformanceMonitor
    {
        private readonly Queue<FramePerformanceData> _history = new(300);
        private readonly object _lock = new();

        public float TargetFrameTimeMs { get; set; } = 16.67f;
        public float BudgetWarningThreshold { get; set; } = 0.85f;
        public float BudgetCriticalThreshold { get; set; } = 0.95f;
        public int HistorySize { get; set; } = 300;

        public PerformanceState CurrentState { get; private set; } = PerformanceState.UnderBudget;
        public float AverageFrameTime { get; private set; }
        public float AverageFps { get; private set; }
        public float FrameTimeVariance { get; private set; }
        public float BudgetUsage => TargetFrameTimeMs > 0 ? AverageFrameTime / TargetFrameTimeMs : 0;

        public void ReportFrame(float frameTimeMs, int drawCalls, int triangles, float gpuTimeMs)
        {
            lock (_lock)
            {
                _history.Enqueue(new FramePerformanceData
                {
                    FrameTimeMs = frameTimeMs,
                    DrawCalls = drawCalls,
                    Triangles = triangles,
                    GpuTimeMs = gpuTimeMs,
                    Timestamp = Stopwatch.GetTimestamp()
                });

                while (_history.Count > HistorySize)
                    _history.Dequeue();

                Recalculate();
            }
        }

        private void Recalculate()
        {
            if (_history.Count == 0)
                return;

            var data = _history.ToArray();
            AverageFrameTime = data.Average(d => d.FrameTimeMs);
            AverageFps = AverageFrameTime > 0 ? 1000.0f / AverageFrameTime : 0;
            FrameTimeVariance = data.Sum(d => (d.FrameTimeMs - AverageFrameTime) * (d.FrameTimeMs - AverageFrameTime)) / data.Length;

            float budget = BudgetUsage;
            CurrentState = budget < BudgetWarningThreshold ? PerformanceState.UnderBudget :
                          budget < 1.0f ? PerformanceState.OnBudget :
                          budget < BudgetCriticalThreshold ? PerformanceState.NearBudget :
                          budget < 1.5f ? PerformanceState.OverBudget :
                          PerformanceState.Critical;
        }

        public FramePerformanceData GetLatest()
        {
            lock (_lock)
            { return _history.Count > 0 ? _history.Last() : new FramePerformanceData(); }
        }

        public FramePerformanceData[] GetHistory()
        {
            lock (_lock)
            { return _history.ToArray(); }
        }

        public float GetFrameTimePercentile(float percentile)
        {
            lock (_lock)
            {
                if (_history.Count == 0)
                    return 0;
                var sorted = _history.Select(d => d.FrameTimeMs).OrderBy(x => x).ToArray();
                int idx = (int)(sorted.Length * percentile / 100.0f);
                return sorted[Math.Min(idx, sorted.Length - 1)];
            }
        }
    }

    public class FramePerformanceData
    {
        public float FrameTimeMs { get; set; }
        public int DrawCalls { get; set; }
        public int Triangles { get; set; }
        public float GpuTimeMs { get; set; }
        public long Timestamp { get; set; }
    }

    // =========================================================================
    // QUALITY ADAPTER
    // =========================================================================

    /// <summary>
    /// Adapts quality settings dynamically based on performance metrics.
    /// Supports smooth transitions between quality levels.
    /// </summary>
    public class QualityAdapter
    {
        private readonly PerformanceMonitor _monitor;
        private QualityLevel _current;
        private QualityLevel _target;
        private float _transitionProgress = 1.0f;
        private float _transitionSpeed = 0.5f;
        private AdaptationMode _mode;
        private readonly Stopwatch _adaptTimer = new();
        private int _adaptCooldownMs = 5000;
        private int _overshootCount;
        private int _undershootCount;

        public QualityLevel CurrentLevel => _current;
        public QualityLevel TargetLevel => _target;
        public float TransitionProgress => _transitionProgress;
        public AdaptationMode Mode => _mode;
        public bool IsTransitioning => _transitionProgress < 1.0f;

        public event Action<QualityLevel, QualityLevel>? OnQualityChanged;

        public QualityAdapter(PerformanceMonitor monitor, QualityPreset initial = QualityPreset.Medium, AdaptationMode mode = AdaptationMode.Dynamic)
        {
            _monitor = monitor;
            _current = QualityPresets.GetPreset(initial);
            _target = _current;
            _mode = mode;
        }

        public void SetMode(AdaptationMode mode) => _mode = mode;

        public void SetTarget(QualityPreset preset)
        {
            _target = QualityPresets.GetPreset(preset);
            _transitionProgress = 0;
        }

        /// <summary>Immediately applies the target preset (no blend).</summary>
        public void SnapToTarget()
        {
            _current = _target;
            _transitionProgress = 1f;
            OnQualityChanged?.Invoke(_current, _target);
        }

        public void Update(float deltaTime)
        {
            if (_mode == AdaptationMode.Disabled)
                return;

            if (_mode == AdaptationMode.Dynamic || _mode == AdaptationMode.AggressiveDynamic)
            {
                AdaptBasedOnPerformance(deltaTime);
            }

            if (_transitionProgress < 1.0f)
            {
                _transitionProgress = MathF.Min(1.0f, _transitionProgress + _transitionSpeed * deltaTime);
                if (_transitionProgress >= 1.0f)
                {
                    _current = _target;
                    OnQualityChanged?.Invoke(_current, _target);
                }
            }
        }

        private void AdaptBasedOnPerformance(float deltaTime)
        {
            if (!_adaptTimer.IsRunning || _adaptTimer.ElapsedMilliseconds >= _adaptCooldownMs)
            {
                var state = _monitor.CurrentState;
                float budget = _monitor.BudgetUsage;

                float adaptThreshold = _mode == AdaptationMode.AggressiveDynamic ? 0.7f : 0.85f;
                float degradeThreshold = _mode == AdaptationMode.AggressiveDynamic ? 0.95f : 1.1f;

                if (state == PerformanceState.Critical || budget > degradeThreshold)
                {
                    _overshootCount++;
                    _undershootCount = 0;
                    if (_overshootCount >= 3 || _mode == AdaptationMode.AggressiveDynamic)
                    {
                        DegradeQuality();
                        _adaptTimer.Restart();
                        _overshootCount = 0;
                    }
                }
                else if (state == PerformanceState.UnderBudget && budget < adaptThreshold)
                {
                    _undershootCount++;
                    _overshootCount = 0;
                    if (_undershootCount >= 5)
                    {
                        UpgradeQuality();
                        _adaptTimer.Restart();
                        _undershootCount = 0;
                    }
                }
                else
                {
                    _overshootCount = Math.Max(0, _overshootCount - 1);
                    _undershootCount = Math.Max(0, _undershootCount - 1);
                }
            }
        }

        private void DegradeQuality()
        {
            var current = _current.Preset;
            var next = current switch
            {
                QualityPreset.Cinematic => QualityPreset.Ultra,
                QualityPreset.Ultra => QualityPreset.High,
                QualityPreset.High => QualityPreset.Medium,
                QualityPreset.Medium => QualityPreset.Low,
                _ => QualityPreset.Low
            };

            if (next != current)
            {
                _target = QualityPresets.GetPreset(next);
                _transitionProgress = 0;
            }
        }

        private void UpgradeQuality()
        {
            var current = _current.Preset;
            var next = current switch
            {
                QualityPreset.Low => QualityPreset.Medium,
                QualityPreset.Medium => QualityPreset.High,
                QualityPreset.High => QualityPreset.Ultra,
                QualityPreset.Ultra => QualityPreset.Cinematic,
                _ => QualityPreset.Cinematic
            };

            if (next != current)
            {
                _target = QualityPresets.GetPreset(next);
                _transitionProgress = 0;
            }
        }

        public QualityLevel InterpolateLevels(float t)
        {
            if (_transitionProgress >= 1.0f)
                return _current;
            float blend = _transitionProgress * _transitionProgress * (3f - 2f * _transitionProgress);

            return new QualityLevel
            {
                Preset = _current.Preset,
                ResolutionScale = Lerp(_current.ResolutionScale, _target.ResolutionScale, blend),
                ShadowResolution = LerpInt(_current.ShadowResolution, _target.ShadowResolution, blend),
                ShadowCascades = LerpInt(_current.ShadowCascades, _target.ShadowCascades, blend),
                ShadowDistance = Lerp(_current.ShadowDistance, _target.ShadowDistance, blend),
                MaxDynamicLights = LerpInt(_current.MaxDynamicLights, _target.MaxDynamicLights, blend),
                EnableBloom = blend > 0.5f ? _target.EnableBloom : _current.EnableBloom,
                EnableDOF = blend > 0.5f ? _target.EnableDOF : _current.EnableDOF,
                EnableMotionBlur = blend > 0.5f ? _target.EnableMotionBlur : _current.EnableMotionBlur,
                EnableSSAO = blend > 0.5f ? _target.EnableSSAO : _current.EnableSSAO,
                EnableSSR = blend > 0.5f ? _target.EnableSSR : _current.EnableSSR,
                EnableGlobalIllumination = blend > 0.5f ? _target.EnableGlobalIllumination : _current.EnableGlobalIllumination,
                TextureQuality = LerpInt(_current.TextureQuality, _target.TextureQuality, blend),
                MaxAnisotropy = LerpInt(_current.MaxAnisotropy, _target.MaxAnisotropy, blend),
                AASamples = LerpInt(_current.AASamples, _target.AASamples, blend),
                ParticleQuality = LerpInt(_current.ParticleQuality, _target.ParticleQuality, blend)
            };
        }

        private static float Lerp(float a, float b, float t) => a + (b - a) * t;
        private static int LerpInt(int a, int b, float t) => (int)(a + (b - a) * t);
    }

    // =========================================================================
    // RUNTIME QUALITY MANAGER
    // =========================================================================

    /// <summary>
    /// High-level quality manager coordinating monitoring, adaptation, and
    /// quality settings across all rendering subsystems.
    /// </summary>
    public class RuntimeQualityManager : IDisposable
    {
        private readonly PerformanceMonitor _monitor;
        private readonly QualityAdapter _adapter;
        private readonly Dictionary<QualityCategory, QualityLevel> _categoryOverrides = new();
        private QualityPreset _basePreset;
        private bool _disposed;

        public PerformanceMonitor Monitor => _monitor;
        public QualityAdapter Adapter => _adapter;
        public QualityLevel CurrentLevel => _adapter.CurrentLevel;
        public float TargetFrameTimeMs { get; set; } = 16.67f;

        public event Action<QualityLevel, QualityLevel>? OnQualityChanged
        {
            add => _adapter.OnQualityChanged += value;
            remove => _adapter.OnQualityChanged -= value;
        }

        public RuntimeQualityManager(QualityPreset initial = QualityPreset.Medium, AdaptationMode mode = AdaptationMode.Dynamic)
        {
            _basePreset = initial;
            _monitor = new PerformanceMonitor { TargetFrameTimeMs = TargetFrameTimeMs };
            _adapter = new QualityAdapter(_monitor, initial, mode);
        }

        public void Update(float deltaTime)
        {
            _adapter.Update(deltaTime);
        }

        public void ReportFrame(float frameTimeMs, int drawCalls, int triangles, float gpuTimeMs)
        {
            _monitor.ReportFrame(frameTimeMs, drawCalls, triangles, gpuTimeMs);
        }

        public void SetQualityPreset(QualityPreset preset)
        {
            _basePreset = preset;
            _adapter.SetTarget(preset);
            _adapter.SnapToTarget();
        }

        public void SetAdaptationMode(AdaptationMode mode)
        {
            _adapter.SetMode(mode);
        }

        public QualityLevel GetEffectiveQuality()
        {
            var baseLevel = _adapter.IsTransitioning
                ? _adapter.InterpolateLevels(_adapter.TransitionProgress)
                : _adapter.CurrentLevel;

            foreach (var overrideKvp in _categoryOverrides)
            {
                ApplyCategoryOverride(baseLevel, overrideKvp.Key, overrideKvp.Value);
            }

            return baseLevel;
        }

        public void SetCategoryOverride(QualityCategory category, QualityLevel level)
        {
            _categoryOverrides[category] = level;
        }

        public void ClearCategoryOverride(QualityCategory category)
        {
            _categoryOverrides.Remove(category);
        }

        public void ClearAllOverrides()
        {
            _categoryOverrides.Clear();
        }

        public PerformanceReport GetPerformanceReport()
        {
            return new PerformanceReport
            {
                CurrentQuality = _adapter.CurrentLevel.Preset,
                TargetQuality = _adapter.TargetLevel.Preset,
                IsTransitioning = _adapter.IsTransitioning,
                TransitionProgress = _adapter.TransitionProgress,
                PerformanceState = _monitor.CurrentState,
                AverageFrameTime = _monitor.AverageFrameTime,
                AverageFps = _monitor.AverageFps,
                BudgetUsage = _monitor.BudgetUsage,
                FrameTimeVariance = _monitor.FrameTimeVariance,
                P95FrameTime = _monitor.GetFrameTimePercentile(95),
                P99FrameTime = _monitor.GetFrameTimePercentile(99)
            };
        }

        private void ApplyCategoryOverride(QualityLevel level, QualityCategory category, QualityLevel overrideLevel)
        {
            switch (category)
            {
                case QualityCategory.Shadows:
                    level.ShadowQuality = overrideLevel.ShadowQuality;
                    level.ShadowResolution = overrideLevel.ShadowResolution;
                    level.ShadowCascades = overrideLevel.ShadowCascades;
                    level.ShadowDistance = overrideLevel.ShadowDistance;
                    level.EnableSoftShadows = overrideLevel.EnableSoftShadows;
                    break;
                case QualityCategory.Lighting:
                    level.MaxDynamicLights = overrideLevel.MaxDynamicLights;
                    level.EnableVolumetricLighting = overrideLevel.EnableVolumetricLighting;
                    level.EnableLightProbes = overrideLevel.EnableLightProbes;
                    break;
                case QualityCategory.PostProcessing:
                    level.EnableBloom = overrideLevel.EnableBloom;
                    level.EnableDOF = overrideLevel.EnableDOF;
                    level.EnableMotionBlur = overrideLevel.EnableMotionBlur;
                    level.EnableSSAO = overrideLevel.EnableSSAO;
                    level.EnableSSR = overrideLevel.EnableSSR;
                    break;
                case QualityCategory.Geometry:
                    level.ResolutionScale = overrideLevel.ResolutionScale;
                    level.MaxLODLevel = overrideLevel.MaxLODLevel;
                    level.EnableTessellation = overrideLevel.EnableTessellation;
                    break;
                case QualityCategory.Textures:
                    level.TextureQuality = overrideLevel.TextureQuality;
                    level.MaxAnisotropy = overrideLevel.MaxAnisotropy;
                    break;
                case QualityCategory.Effects:
                    level.ParticleQuality = overrideLevel.ParticleQuality;
                    level.EnableScreenSpaceParticles = overrideLevel.EnableScreenSpaceParticles;
                    level.EnableWeatherEffects = overrideLevel.EnableWeatherEffects;
                    break;
                case QualityCategory.GlobalIllumination:
                    level.EnableGlobalIllumination = overrideLevel.EnableGlobalIllumination;
                    level.GICascadeResolution = overrideLevel.GICascadeResolution;
                    level.GIMaxBounces = overrideLevel.GIMaxBounces;
                    level.EnableScreenSpaceGI = overrideLevel.EnableScreenSpaceGI;
                    break;
                case QualityCategory.AntiAliasing:
                    level.AAMode = overrideLevel.AAMode;
                    level.AASamples = overrideLevel.AASamples;
                    level.EnableTAA = overrideLevel.EnableTAA;
                    break;
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
            }
        }
    }

    /// <summary>Summary of current performance state.</summary>
    public class PerformanceReport
    {
        public QualityPreset CurrentQuality { get; set; }
        public QualityPreset TargetQuality { get; set; }
        public bool IsTransitioning { get; set; }
        public float TransitionProgress { get; set; }
        public PerformanceState PerformanceState { get; set; }
        public float AverageFrameTime { get; set; }
        public float AverageFps { get; set; }
        public float BudgetUsage { get; set; }
        public float FrameTimeVariance { get; set; }
        public float P95FrameTime { get; set; }
        public float P99FrameTime { get; set; }
    }
}
