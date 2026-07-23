using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>Deferred G-buffer MRT pass (albedo/N/world/mat/velocity).</summary>
    public sealed class GBufferPass : IRenderPass
    {
        public string Name => "GBuffer";

        public void Setup(FrameGraphBuilder builder)
        {
            builder.CreateTexture("GBufferAlbedo", 0, 0, FrameGraphResourceUsage.Write);
            builder.CreateTexture("GBufferNormals", 0, 0, FrameGraphResourceUsage.Write);
            builder.CreateTexture("GBufferWorld", 0, 0, FrameGraphResourceUsage.Write);
            builder.CreateTexture("GBufferMaterial", 0, 0, FrameGraphResourceUsage.Write);
            builder.CreateTexture("GBufferVelocity", 0, 0, FrameGraphResourceUsage.Write);
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            context.Backend.OnFrameGraphGBuffer(context);
        }
    }
}
