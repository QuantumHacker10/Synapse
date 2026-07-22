using GDNN.Rendering.FrameGraph.Passes;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>Builds the canonical Synapse present-path FrameGraph (G-DNN → L-DNN → deferred → post → upscale).</summary>
    public static class FrameGraphFactory
    {
        public static RenderFrameGraph CreateDefault()
        {
            var graph = new RenderFrameGraph();
            // L-DNN CPU producers run before GPU recording so textures are ready for lighting.
            graph.AddPass(new LdnnGiPass());
            graph.AddPass(new SceneCullPass());
            graph.AddPass(new AlgorithmSystemsPass());
            graph.AddPass(new ShadowCascadesPass());
            graph.AddPass(new GBufferPass());
            // Cinematic: full-res Nanite material resolve into G-buffer MRTs.
            graph.AddPass(new MeshletMaterialResolvePass());
            graph.AddPass(new DeferredLightingPass());
            graph.AddPass(new ParticlesComputePass());
            graph.AddPass(new PostTonemapPass());
            // Cinematic: FSR / DLSS-compatible / MetalFX upscale to display.
            graph.AddPass(new UpscalePass());
            return graph;
        }
    }
}
