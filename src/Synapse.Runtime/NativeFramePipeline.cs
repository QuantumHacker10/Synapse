using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Rendering.Quality;
using Synapse.Infrastructure.Logging;
using Synapse.Physics;

namespace Synapse.Runtime
{
    /// <summary>
    /// Native multi-module frame pipeline.
    /// Explicitly integrates Runtime orchestration with Physics, Simulation,
    /// Rendering (FrameGraph), and Infrastructure quality — one authority per frame.
    /// </summary>
    public sealed class NativeFramePipeline
    {
        private readonly EngineHost _host;
        private readonly ISynapseLogger _logger;

        public NativeFramePipeline(EngineHost host, ISynapseLogger logger)
        {
            _host = host ?? throw new ArgumentNullException(nameof(host));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Runs one native frame:
        /// Physics → Simulation → scene sync → Physics→Render field → Render FrameGraph → Quality sink.
        /// </summary>
        public async Task<NativeFrameResult> ExecuteAsync(
            float dt,
            bool paused,
            CancellationToken cancellationToken = default)
        {
            var result = new NativeFrameResult { DeltaTime = dt, IsPaused = paused };
            var sw = Stopwatch.StartNew();

            if (!paused)
            {
                sw.Restart();
                _host.TickPhysics(dt);
                result.PhysicsMs = (float)sw.Elapsed.TotalMilliseconds;
                result.ContinuumMs = (float)(_host.Multiphysics?.LastStats.ContinuumMs ?? 0);
                result.LivingLawMs = (float)(_host.Multiphysics?.LastStats.LivingLawMs ?? 0);
                result.RigidBodyMs = (float)(_host.Multiphysics?.LastStats.RigidBodyMs ?? 0);

                sw.Restart();
                await _host.TickSimulationAsync(dt, cancellationToken).ConfigureAwait(false);
                result.SimulationMs = (float)sw.Elapsed.TotalMilliseconds;

                sw.Restart();
                _host.SyncSimulationTransformsToScene();
                _host.PushPhysicsFieldToRenderer();
                result.SyncMs = (float)sw.Elapsed.TotalMilliseconds;
            }
            else
            {
                // Keep editor gizmos / field viz coherent while paused.
                _host.PushPhysicsFieldToRenderer();
            }

            sw.Restart();
            _host.TickRender();
            result.RenderMs = (float)sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            _host.TickQuality(dt, result.PhysicsMs, result.SimulationMs, result.RenderMs);
            result.QualityMs = (float)sw.Elapsed.TotalMilliseconds;

            result.QualityPreset = _host.QualityPresetName;
            result.EntityCount = _host.SceneEntityCount;
            result.AgentCount = _host.EntityCount;
            result.ActiveLawId = _host.ActiveLawId ?? "";
            result.FieldTemperatureAvg = _host.AverageFieldTemperature;
            result.EvolutionGeneration = _host.EvolutionGeneration;
            result.BestFitness = _host.BestFitness;
            result.EnabledContinuumModules = _host.Multiphysics?.Config.EnabledModules ?? ContinuumModules.None;
            result.RenderReady = _host.IsRenderInitialized;

            return result;
        }
    }

    /// <summary>Telemetry for one native pipeline frame.</summary>
    public sealed class NativeFrameResult
    {
        public float DeltaTime { get; set; }
        public bool IsPaused { get; set; }
        public float PhysicsMs { get; set; }
        public float SimulationMs { get; set; }
        public float SyncMs { get; set; }
        public float RenderMs { get; set; }
        public float QualityMs { get; set; }
        public float ContinuumMs { get; set; }
        public float LivingLawMs { get; set; }
        public float RigidBodyMs { get; set; }
        public string QualityPreset { get; set; } = "High";
        public int EntityCount { get; set; }
        public int AgentCount { get; set; }
        public string ActiveLawId { get; set; } = "";
        public float FieldTemperatureAvg { get; set; }
        public int EvolutionGeneration { get; set; }
        public double BestFitness { get; set; }
        public ContinuumModules EnabledContinuumModules { get; set; }
        public bool RenderReady { get; set; }
    }

    /// <summary>Maps Infrastructure quality levels onto the Rendering FrameGraph sink.</summary>
    public static class RuntimeQualityMapper
    {
        public static RuntimeRenderQuality FromLevel(QualityLevel level)
        {
            ArgumentNullException.ThrowIfNull(level);
            return new RuntimeRenderQuality
            {
                PresetName = level.Preset.ToString(),
                ResolutionScale = level.ResolutionScale,
                MaxLodLevel = level.MaxLODLevel,
                ShadowCascades = level.ShadowCascades,
                ShadowQuality = level.ShadowQuality,
                EnableGlobalIllumination = level.EnableGlobalIllumination,
                EnableScreenSpaceGi = level.EnableScreenSpaceGI,
                EnableSsao = level.EnableSSAO,
                EnableBloom = level.EnableBloom,
                EnableTaa = level.EnableTAA,
                EnableVolumetricLighting = level.EnableVolumetricLighting,
                ParticleQuality = level.ParticleQuality,
                GiMaxBounces = level.GIMaxBounces
            };
        }
    }
}
