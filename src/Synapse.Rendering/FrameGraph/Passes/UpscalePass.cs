using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>
    /// Temporal upscaling after tonemap: FSR / DLSS-compatible / MetalFX-compatible.
    /// Renders at internal resolution then upscales to display/swapchain size.
    /// </summary>
    public sealed class UpscalePass : IRenderPass
    {
        public string Name => "TemporalUpscale";

        public void Setup(FrameGraphBuilder builder)
        {
            builder.ImportTexture("Swapchain");
            builder.ImportTexture("GBufferVelocity");
            builder.CreateTexture("DisplayColor", 0, 0, FrameGraphResourceUsage.Write);
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            context.Backend.OnFrameGraphUpscale(context);
        }
    }
}
