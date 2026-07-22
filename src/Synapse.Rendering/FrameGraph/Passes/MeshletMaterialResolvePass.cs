using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>
    /// Full-res G-DNN Nanite material resolve into G-buffer MRT (cinematic path).
    /// Runs after GBuffer so deferred lighting samples cluster materials at viewport scale.
    /// </summary>
    public sealed class MeshletMaterialResolvePass : IRenderPass
    {
        public string Name => "MeshletMaterialResolve";

        public void Setup(FrameGraphBuilder builder)
        {
            builder.ImportTexture("GBufferAlbedo");
            builder.ImportTexture("GBufferNormal");
            builder.CreateTexture("NaniteMaterialAlbedo", 0, 0, FrameGraphResourceUsage.Write);
            builder.CreateTexture("NaniteMaterialNormal", 0, 0, FrameGraphResourceUsage.Write);
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            context.Backend.OnFrameGraphMeshletResolve(context);
        }
    }
}
