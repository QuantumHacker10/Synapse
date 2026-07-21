using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>
    /// L-DNN GI producer: Hybrid SSGI + Radiance Cascades + AO + fog → GPU textures.
    /// Phase 1–2: CPU algorithms with batched upload; Phase 2+: compute kernels.
    /// </summary>
    public sealed class LdnnGiPass : IRenderPass
    {
        public string Name => "LdnnGi";

        public void Setup(FrameGraphBuilder builder)
        {
            builder.CreateTexture("GiIrradiance", 0, 0, FrameGraphResourceUsage.Write);
            builder.CreateTexture("Ssao", 0, 0, FrameGraphResourceUsage.Write);
            builder.CreateTexture("VolumetricFog", 0, 0, FrameGraphResourceUsage.Write);
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            if (context.RunLdnnCpuProducers)
                context.Backend.OnFrameGraphLdnn(context);
        }
    }
}
