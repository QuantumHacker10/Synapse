using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>G-DNN LOD / cull pass — updates LodManager before geometry submission.</summary>
    public sealed class SceneCullPass : IRenderPass
    {
        public string Name => "SceneCull_GDNN";
        public RenderPassPhase Phase => RenderPassPhase.Gpu;

        public void Setup(FrameGraphBuilder builder)
        {
            builder.CreateTexture("VisibleDrawList", 1, 1, FrameGraphResourceUsage.Write);
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            context.World.CameraPosition = context.CameraPos;
            context.World.CameraForward = context.CameraForward;
            context.World.CameraRight = context.CameraRight;
            context.World.ViewportHeight = context.Height;
            context.World.UpdateLod();
            context.Backend.OnFrameGraphCull(context);
        }
    }
}
