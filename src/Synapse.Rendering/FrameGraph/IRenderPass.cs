namespace GDNN.Rendering.FrameGraph
{
    /// <summary>
    /// When a pass runs relative to Vulkan command-buffer recording.
    /// CPU producers (e.g. L-DNN Hybrid) must finish before <c>cmd.Begin</c>;
    /// GPU passes record into the active buffer.
    /// </summary>
    public enum RenderPassPhase
    {
        /// <summary>Runs before command-buffer recording (CPU / upload).</summary>
        CpuProducer = 0,
        /// <summary>Records GPU work into the active command buffer.</summary>
        Gpu = 1
    }

    /// <summary>A single named pass in the GPU-first render graph.</summary>
    public interface IRenderPass
    {
        string Name { get; }

        /// <summary>Execution phase for the native FrameGraph pipeline.</summary>
        RenderPassPhase Phase { get; }

        /// <summary>Declare resource reads/writes for this frame.</summary>
        void Setup(FrameGraphBuilder builder);

        /// <summary>Record or run the pass (CPU producers allowed before GPU work).</summary>
        void Execute(FrameGraphContext context);
    }
}
