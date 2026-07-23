using System.Numerics;
using GDNN.Rendering.Engine;
using GDNN.RHI.Vulkan;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>Per-frame execution context shared by all FrameGraph passes.</summary>
    public sealed class FrameGraphContext
    {
        public required VulkanRhiDevice Rhi { get; init; }
        public required VulkanCommandBuffer Cmd { get; init; }
        public required uint ImageIndex { get; init; }
        public required int FrameIndex { get; init; }
        public required int Width { get; init; }
        public required int Height { get; init; }
        public required SceneWorld World { get; init; }

        /// <summary>Legacy backend used during Phase 1–3 migration (GPU resource owner).</summary>
        public required SceneRenderer Backend { get; init; }

        public Matrix4x4 View { get; set; }
        public Matrix4x4 Projection { get; set; }
        public Vector3 CameraPos { get; set; }
        public Vector3 CameraForward { get; set; }
        public Vector3 CameraRight { get; set; }
        public float Time { get; set; }
        public bool RunLdnnCpuProducers { get; set; } = true;
    }
}
