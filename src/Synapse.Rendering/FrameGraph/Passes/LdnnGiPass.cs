using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>
    /// L-DNN GI producer: Hybrid SSGI + Radiance Cascades + AO + fog → GPU textures.
    /// CPU phase uploads irradiance/AO/fog; GPU phase may dispatch resident compute (SSAO).
    /// </summary>
    public sealed class LdnnGiPass : IRenderPass
    {
        public string Name => "LdnnGi";
        public RenderPassPhase Phase => RenderPassPhase.CpuProducer;

        public void Setup(FrameGraphBuilder builder)
        {
            builder.CreateTexture("GiIrradiance", 0, 0, FrameGraphResourceUsage.Write);
            builder.CreateTexture("Ssao", 0, 0, FrameGraphResourceUsage.Write);
            builder.CreateTexture("VolumetricFog", 0, 0, FrameGraphResourceUsage.Write);
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            context.Backend.OnFrameGraphLdnn(context);
        }
    }
}
