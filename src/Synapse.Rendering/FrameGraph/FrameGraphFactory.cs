using GDNN.Rendering.FrameGraph.Passes;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>
    /// Builds the canonical Synapse native present-path FrameGraph.
    /// Integrates L-DNN (GI), G-DNN (cull + algorithm hub), shadows, deferred, particles, post.
    /// </summary>
    public static class FrameGraphFactory
    {
        public static RenderFrameGraph CreateDefault()
        {
            var graph = new RenderFrameGraph();
            // CPU producers (before cmd.Begin)
            graph.AddPass(new LdnnGiPass());
            // GPU recording (inside command buffer)
            graph.AddPass(new SceneCullPass());
            graph.AddPass(new AlgorithmSystemsPass());
            graph.AddPass(new ShadowCascadesPass());
            graph.AddPass(new GBufferPass());
            graph.AddPass(new DeferredLightingPass());
            graph.AddPass(new ParticlesComputePass());
            graph.AddPass(new PostTonemapPass());
            return graph;
        }
    }
}
