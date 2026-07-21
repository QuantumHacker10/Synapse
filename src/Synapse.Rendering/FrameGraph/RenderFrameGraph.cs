using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>
    /// GPU-first frame graph: setup → ordered execute of registered passes.
    /// Replaces the ad-hoc RecordCommandBuffer sequence in SceneRenderer.
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

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);

            _builder.Reset();
            _passTimingsMs.Clear();

            for (int i = 0; i < _passes.Count; i++)
                _passes[i].Setup(_builder);

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < _passes.Count; i++)
            {
                sw.Restart();
                _passes[i].Execute(context);
                _passTimingsMs[_passes[i].Name] = sw.ElapsedMilliseconds;
            }
        }
    }
}
