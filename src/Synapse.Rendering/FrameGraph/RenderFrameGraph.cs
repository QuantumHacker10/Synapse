using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>
    /// Native GPU-first frame graph: setup → ordered execute of registered passes.
    /// Supports a two-phase present path (CPU producers → GPU recording) so L-DNN,
    /// G-DNN cull/algorithms, shadows, deferred, particles and post all share one executor.
    /// </summary>
    public sealed class RenderFrameGraph
    {
        private readonly List<IRenderPass> _passes = new();
        private readonly FrameGraphBuilder _builder = new();
        private readonly Dictionary<string, long> _passTimingsMs = new();

        public IReadOnlyList<IRenderPass> Passes => _passes;
        public IReadOnlyDictionary<string, long> LastPassTimingsMs => _passTimingsMs;

        public void ClearPasses() => _passes.Clear();

        public void AddPass(IRenderPass pass)
        {
            ArgumentNullException.ThrowIfNull(pass);
            _passes.Add(pass);
        }

        /// <summary>Executes every registered pass in order (legacy single-shot).</summary>
        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            ExecutePhase(context, phaseFilter: null);
        }

        /// <summary>Runs only CPU-producer passes (before <c>cmd.Begin</c>).</summary>
        public void ExecuteCpuProducers(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            ExecutePhase(context, RenderPassPhase.CpuProducer);
        }

        /// <summary>Runs only GPU recording passes (inside an active command buffer).</summary>
        public void ExecuteGpuPasses(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            ExecutePhase(context, RenderPassPhase.Gpu);
        }

        private void ExecutePhase(FrameGraphContext context, RenderPassPhase? phaseFilter)
        {
            _builder.Reset();
            if (phaseFilter is null)
                _passTimingsMs.Clear();

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < _passes.Count; i++)
            {
                var pass = _passes[i];
                if (phaseFilter is { } phase && pass.Phase != phase)
                    continue;

                pass.Setup(_builder);
                sw.Restart();
                pass.Execute(context);
                _passTimingsMs[pass.Name] = sw.ElapsedMilliseconds;
            }
        }
    }
}
