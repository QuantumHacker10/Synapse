using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>Directional shadow map / cascades recorded into the backend shadow resources.</summary>
    public sealed class ShadowCascadesPass : IRenderPass
    {
        public string Name => "ShadowCascades";

        public void Setup(FrameGraphBuilder builder)
        {
            builder.CreateTexture("ShadowMap", 2048, 2048, FrameGraphResourceUsage.Write);
            builder.CreateTexture("ShadowMapCascade1", 2048, 2048, FrameGraphResourceUsage.Write);
            builder.CreateTexture("ShadowMapCascade2", 2048, 2048, FrameGraphResourceUsage.Write);
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            context.Backend.OnFrameGraphShadow(context);
        }
    }
}
