using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>
    /// Ticks Rendering-module systems (world partition, VT, neural LOD, geometry batching)
    /// before shadow / G-buffer submission.
    /// </summary>
    public sealed class AlgorithmSystemsPass : IRenderPass
    {
        public string Name => "AlgorithmSystems";
        public RenderPassPhase Phase => RenderPassPhase.Gpu;

        public void Setup(FrameGraphBuilder builder)
        {
            builder.CreateTexture("AlgorithmFeedback", 1, 1, FrameGraphResourceUsage.Write);
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            context.Backend.OnFrameGraphAlgorithms(context);
        }
    }
}
