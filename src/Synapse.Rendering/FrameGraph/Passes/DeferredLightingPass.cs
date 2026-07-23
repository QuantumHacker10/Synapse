using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>Fullscreen deferred lighting into HDR target.</summary>
    public sealed class DeferredLightingPass : IRenderPass
    {
        public string Name => "DeferredLighting";

        public void Setup(FrameGraphBuilder builder)
        {
            builder.ImportTexture("GBufferAlbedo");
            builder.ImportTexture("GBufferNormals");
            builder.ImportTexture("GBufferWorld");
            builder.ImportTexture("GBufferMaterial");
            builder.ImportTexture("GiIrradiance");
            builder.ImportTexture("ShadowMap");
            builder.ImportTexture("Ssao");
            builder.ImportTexture("VolumetricFog");
            builder.CreateTexture("HdrColor", 0, 0, FrameGraphResourceUsage.Write);
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            context.Backend.OnFrameGraphLighting(context);
        }
    }
}
