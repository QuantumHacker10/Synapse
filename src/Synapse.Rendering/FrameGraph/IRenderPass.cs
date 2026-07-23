namespace GDNN.Rendering.FrameGraph
{
    /// <summary>A single named pass in the GPU-first render graph.</summary>
    public interface IRenderPass
    {
        string Name { get; }

        /// <summary>Declare resource reads/writes for this frame.</summary>
        void Setup(FrameGraphBuilder builder);

        /// <summary>Record or run the pass (CPU producers allowed before GPU work).</summary>
        void Execute(FrameGraphContext context);
    }
}
