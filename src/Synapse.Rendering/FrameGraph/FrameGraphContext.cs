using System.Numerics;
using GDNN.Rendering.Engine;
using GDNN.Rendering.Quality;
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

        /// <summary>Deferred renderer owning Vulkan resources and pass backends.</summary>
        public required SceneRenderer Backend { get; init; }

        public Matrix4x4 View { get; set; }
        public Matrix4x4 Projection { get; set; }
        public Vector3 CameraPos { get; set; }
        public Vector3 CameraForward { get; set; }
        public Vector3 CameraRight { get; set; }
        public float Time { get; set; }

        /// <summary>
        /// When true, <see cref="Passes.LdnnGiPass"/> runs Hybrid CPU producers + texture upload.
        /// Cleared for the GPU phase so the same pass can optionally dispatch compute.
        /// </summary>
        public bool RunLdnnCpuProducers { get; set; } = true;

        /// <summary>Average living-law field temperature from Physics (drives fog / GI warmth).</summary>
        public float PhysicsFieldTemperature { get; set; }

        /// <summary>Adaptive quality snapshot applied before GPU passes (may be null).</summary>
        public RuntimeRenderQuality? Quality { get; set; }
    }
}
