using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>HDR → LDR tonemap/bloom into the swapchain.</summary>
    public sealed class PostTonemapPass : IRenderPass
    {
        public string Name => "PostTonemap";
        public RenderPassPhase Phase => RenderPassPhase.Gpu;

        public void Setup(FrameGraphBuilder builder)
        {
            builder.ImportTexture("HdrColor");
            builder.ImportTexture("GBufferVelocity");
            builder.CreateTexture("Swapchain", 0, 0, FrameGraphResourceUsage.Import);
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            context.Backend.OnFrameGraphPost(context);
        }
    }
}
