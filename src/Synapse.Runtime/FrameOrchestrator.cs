using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Synapse.Infrastructure.Logging;

namespace Synapse.Runtime
{
    public sealed class FrameStats
    {
        public float DeltaTime { get; set; }
        public float Fps { get; set; }
        public float TotalTime { get; set; }
        public float PhysicsMs { get; set; }
        public float SimulationMs { get; set; }
        public float SyncMs { get; set; }
        public float RenderMs { get; set; }
        public float QualityMs { get; set; }
        public float ContinuumMs { get; set; }
        public bool IsPaused { get; set; }
        public string QualityPreset { get; set; } = "High";
        public int EntityCount { get; set; }
        public int AgentCount { get; set; }
        public string ActiveLawId { get; set; } = "";
        public float FieldTemperatureAvg { get; set; }
        public int EvolutionGeneration { get; set; }
        public double BestFitness { get; set; }
        public bool RenderReady { get; set; }
        public string ContinuumModules { get; set; } = "";
    }

    /// <summary>
    /// Studio / engine frame authority. Delegates to <see cref="NativeFramePipeline"/>
    /// so Physics, Simulation, Rendering and Quality advance as one native loop —
    /// independent of whether Vulkan init succeeded.
    /// </summary>
    public sealed class FrameOrchestrator
    {
        private readonly EngineHost _host;
        private readonly ISynapseLogger _logger;
        private readonly NativeFramePipeline _pipeline;
        private readonly Stopwatch _frameTimer = Stopwatch.StartNew();
        private readonly Stopwatch _fpsTimer = Stopwatch.StartNew();
        private int _fpsFrames;
        private float _fps;
        private float _totalTime;
        private bool _paused;

        public FrameOrchestrator(EngineHost host, ISynapseLogger logger)
        {
            _host = host;
            _logger = logger;
            _pipeline = new NativeFramePipeline(host, logger);
        }

        public bool IsPaused
        {
            get => _paused;
            set => _paused = value;
        }

        public NativeFramePipeline Pipeline => _pipeline;

        public FrameStats LastStats { get; private set; } = new();

        public async Task<FrameStats> TickAsync(CancellationToken cancellationToken = default)
        {
            float dt = (float)_frameTimer.Elapsed.TotalSeconds;
            _frameTimer.Restart();
            if (dt <= 0 || dt > 0.25f)
                dt = 1f / 60f;
            if (!_paused)
                _totalTime += dt;

            _fpsFrames++;
            if (_fpsTimer.Elapsed.TotalSeconds >= 1.0)
            {
                _fps = _fpsFrames / (float)_fpsTimer.Elapsed.TotalSeconds;
                _fpsFrames = 0;
                _fpsTimer.Restart();
            }

            var result = await _pipeline.ExecuteAsync(dt, _paused, cancellationToken).ConfigureAwait(false);

            LastStats = new FrameStats
            {
                DeltaTime = result.DeltaTime,
                Fps = _fps,
                TotalTime = _totalTime,
                PhysicsMs = result.PhysicsMs,
                SimulationMs = result.SimulationMs,
                SyncMs = result.SyncMs,
                RenderMs = result.RenderMs,
                QualityMs = result.QualityMs,
                ContinuumMs = result.ContinuumMs,
                IsPaused = result.IsPaused,
                QualityPreset = result.QualityPreset,
                EntityCount = result.EntityCount > 0 ? result.EntityCount : result.AgentCount,
                AgentCount = result.AgentCount,
                ActiveLawId = result.ActiveLawId,
                FieldTemperatureAvg = result.FieldTemperatureAvg,
                EvolutionGeneration = result.EvolutionGeneration,
                BestFitness = result.BestFitness,
                RenderReady = result.RenderReady,
                ContinuumModules = result.EnabledContinuumModules.ToString()
            };

            return LastStats;
        }
    }
}
