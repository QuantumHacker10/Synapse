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
        public float CouplingMs { get; set; }
        public float RenderMs { get; set; }
        public float QualityMs { get; set; }
        public float VrMs { get; set; }
        public float CollaborationMs { get; set; }
        public bool IsPaused { get; set; }
        public string QualityPreset { get; set; } = "High";
        public int EntityCount { get; set; }
        public string ActiveLawId { get; set; } = "";
        public float FieldTemperatureAvg { get; set; }
        public int EvolutionGeneration { get; set; }
        public double BestFitness { get; set; }
        public string? LastRuntimeError { get; set; }
        public bool VrActive { get; set; }
        public bool WanActive { get; set; }
        public string VrStatus { get; set; } = "";
        public string WanStatus { get; set; } = "";
    }

    public sealed class FrameOrchestrator : IDisposable
    /// <summary>
    /// Per-frame industrial cascade:
    /// <b>Physics → Simulation → Coupling (Physics↔Render↔Sim) → Rendering → Quality</b>.
    /// LLM world-deltas are applied asynchronously via <see cref="EngineHost.ApplyLlmWorldDelta"/>.
    /// </summary>
    public sealed class FrameOrchestrator
    {
        private readonly EngineHost _host;
        private readonly ISynapseLogger _logger;
        private readonly Stopwatch _frameTimer = Stopwatch.StartNew();
        private readonly Stopwatch _fpsTimer = Stopwatch.StartNew();
        private readonly SemaphoreSlim _tickGate = new(1, 1);
        private int _fpsFrames;
        private float _fps;
        private float _totalTime;
        private bool _paused;
        private bool _disposed;

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
            if (!await _tickGate.WaitAsync(0, cancellationToken).ConfigureAwait(false))
            {
                // Drop overlapping ticks rather than re-entering shared engine state.
                return LastStats;
            }

            try
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

                float physicsMs = 0, simMs = 0, renderMs = 0, qualityMs = 0, vrMs = 0, collabMs = 0;

                {
                    var sw = Stopwatch.StartNew();
                    await _host.TickVrBeginAsync(cancellationToken).ConfigureAwait(false);
                    vrMs += (float)sw.Elapsed.TotalMilliseconds;
                }

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
            float physicsMs = 0, simMs = 0, couplingMs = 0, renderMs = 0, qualityMs = 0;

            if (!_paused)
            {
                // 1) Physics (living laws + rigid + optional continuum)
                var sw = Stopwatch.StartNew();
                _host.TickPhysics(dt);
                physicsMs = (float)sw.Elapsed.TotalMilliseconds;

                // 2) Simulation (sentience / behavior trees — may queue physics actuators)
                sw.Restart();
                await _host.TickSimulationAsync(dt, cancellationToken).ConfigureAwait(false);
                simMs = (float)sw.Elapsed.TotalMilliseconds;

                // 3) Coupling: Physics → Rendering + Simulation → Physics drain
                sw.Restart();
                _host.TickIndustrialCoupling(dt);
                couplingMs = (float)sw.Elapsed.TotalMilliseconds;
            }

            {
                // 4) Rendering (G-DNN Nanite Neural 3.0 + L-DNN Lumen Neural 3.0)
                var sw = Stopwatch.StartNew();
                _host.TickRender();
                renderMs = (float)sw.Elapsed.TotalMilliseconds;
            }

                {
                    var sw = Stopwatch.StartNew();
                    await _host.TickVrEndAsync(cancellationToken).ConfigureAwait(false);
                    vrMs += (float)sw.Elapsed.TotalMilliseconds;
                }

                if (!_paused)
                {
                    var sw = Stopwatch.StartNew();
                    await _host.TickCollaborationAsync(cancellationToken).ConfigureAwait(false);
                    collabMs = (float)sw.Elapsed.TotalMilliseconds;
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
                    VrMs = vrMs,
                    CollaborationMs = collabMs,
                    IsPaused = _paused,
                    QualityPreset = _host.QualityPresetName,
                    EntityCount = _host.EntityCount,
                    ActiveLawId = _host.ActiveLawId ?? "",
                    FieldTemperatureAvg = _host.AverageFieldTemperature,
                    EvolutionGeneration = _host.EvolutionGeneration,
                    BestFitness = _host.BestFitness,
                    LastRuntimeError = _host.LastRuntimeError,
                    VrActive = _host.IsVrActive,
                    WanActive = _host.IsWanConnected,
                    VrStatus = _host.VrStatusText,
                    WanStatus = _host.WanStatusText
                };

                return LastStats;
            }
            finally
            {
                _tickGate.Release();
            }
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _tickGate.Dispose();
            LastStats = new FrameStats
            {
                DeltaTime = dt,
                Fps = _fps,
                TotalTime = _totalTime,
                PhysicsMs = physicsMs,
                SimulationMs = simMs,
                CouplingMs = couplingMs,
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
