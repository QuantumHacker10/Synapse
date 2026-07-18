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
        public float RenderMs { get; set; }
        public float QualityMs { get; set; }
        public bool IsPaused { get; set; }
        public string QualityPreset { get; set; } = "High";
        public int EntityCount { get; set; }
        public string ActiveLawId { get; set; } = "";
        public float FieldTemperatureAvg { get; set; }
        public int EvolutionGeneration { get; set; }
        public double BestFitness { get; set; }
    }

    public sealed class FrameOrchestrator
    {
        private readonly EngineHost _host;
        private readonly ISynapseLogger _logger;
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
        }

        public bool IsPaused
        {
            get => _paused;
            set => _paused = value;
        }

        public FrameStats LastStats { get; private set; } = new();

        public async Task<FrameStats> TickAsync(CancellationToken cancellationToken = default)
        {
            float dt = (float)_frameTimer.Elapsed.TotalSeconds;
            _frameTimer.Restart();
            if (dt <= 0 || dt > 0.25f) dt = 1f / 60f;
            if (!_paused) _totalTime += dt;

            _fpsFrames++;
            if (_fpsTimer.Elapsed.TotalSeconds >= 1.0)
            {
                _fps = _fpsFrames / (float)_fpsTimer.Elapsed.TotalSeconds;
                _fpsFrames = 0;
                _fpsTimer.Restart();
            }

            float physicsMs = 0, simMs = 0, renderMs = 0, qualityMs = 0;

            if (!_paused)
            {
                var sw = Stopwatch.StartNew();
                _host.TickPhysics(dt);
                physicsMs = (float)sw.Elapsed.TotalMilliseconds;

                sw.Restart();
                await _host.TickSimulationAsync(dt, cancellationToken).ConfigureAwait(false);
                simMs = (float)sw.Elapsed.TotalMilliseconds;
            }

            {
                var sw = Stopwatch.StartNew();
                _host.TickRender();
                renderMs = (float)sw.Elapsed.TotalMilliseconds;
            }

            {
                var sw = Stopwatch.StartNew();
                _host.TickQuality(dt, renderMs);
                qualityMs = (float)sw.Elapsed.TotalMilliseconds;
            }

            LastStats = new FrameStats
            {
                DeltaTime = dt,
                Fps = _fps,
                TotalTime = _totalTime,
                PhysicsMs = physicsMs,
                SimulationMs = simMs,
                RenderMs = renderMs,
                QualityMs = qualityMs,
                IsPaused = _paused,
                QualityPreset = _host.QualityPresetName,
                EntityCount = _host.EntityCount,
                ActiveLawId = _host.ActiveLawId ?? "",
                FieldTemperatureAvg = _host.AverageFieldTemperature,
                EvolutionGeneration = _host.EvolutionGeneration,
                BestFitness = _host.BestFitness
            };

            return LastStats;
        }
    }
}
