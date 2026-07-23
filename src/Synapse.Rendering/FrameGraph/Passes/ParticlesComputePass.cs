using System;

namespace GDNN.Rendering.FrameGraph.Passes
{
    /// <summary>
    /// Particle simulation + compute dispatcher residency after lighting, before tonemap.
    /// Particle glow is composited into L-DNN fog on the CPU producer path.
    /// </summary>
    public sealed class ParticlesComputePass : IRenderPass
    {
        public string Name => "ParticlesCompute";

        public void Setup(FrameGraphBuilder builder)
        {
            builder.CreateTexture("ParticleField", 1, 1, FrameGraphResourceUsage.Write);
            builder.ImportTexture("HdrColor");
        }

        public void Execute(FrameGraphContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            context.Backend.OnFrameGraphParticles(context);
        }
    }
}
