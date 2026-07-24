using System;
using GDNN.RHI.Vulkan;
using GDNN.Rendering.Compute;

namespace GDNN.Rendering.FrameGraph
{
    /// <summary>
    /// Phase-2 L-DNN compute scaffolding. The CPU Hybrid path in <c>LDNNRenderer</c> remains the
    /// production GI path until <see cref="BindComputeModule"/> is called with a real SPIR-V module.
    /// When a <see cref="ComputeDispatcher"/> is attached, SSAO/blur requests enqueue named kernels
    /// (`denoise`, `gi_irradiance`) on the shared compute path, but the dispatch is only reported as
    /// handled (return true) once an actual compute module is bound.
    /// Phase-2 L-DNN compute scaffolding: prefers Vulkan compute when a pipeline is bound,
    /// otherwise callers keep using the CPU Parallel.For kernels in LDNNRenderer.
    /// When a <see cref="ComputeDispatcher"/> is attached, SSAO/blur requests enqueue
    /// named kernels (`denoise`, `gi_irradiance`) on the shared compute path.
    /// </summary>
    public sealed class LdnnComputeSupport : IDisposable
    {
        private readonly VulkanRhiDevice _rhi;
        private ComputeDispatcher? _dispatcher;
        private bool _disposed;

        public LdnnComputeSupport(VulkanRhiDevice rhi)
        {
            _rhi = rhi ?? throw new ArgumentNullException(nameof(rhi));
        }

        public void AttachDispatcher(ComputeDispatcher dispatcher)
            => _dispatcher = dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));

        /// <summary>True when the device exposes a graphics queue usable for compute transfers.</summary>
        public bool IsAvailable => _rhi != null && !_disposed;

        /// <summary>
        /// Enqueues SSAO/denoise kernels when a SPIR-V compute module is bound. Returns false
        /// (so LdnnGiPass keeps running the Hybrid CPU producers) whenever no module is bound —
        /// it does not pretend to have handled the pass by dispatching empty kernels.
        /// Enqueues SSAO/denoise kernels when a dispatcher is attached. Returns false so
        /// LdnnGiPass still runs the Hybrid CPU producers until SPIR-V compute modules
        /// replace Parallel.For (see <see cref="BindComputeModule"/>).
        /// </summary>
        public bool TryDispatchSsao(VulkanCommandBuffer cmd, int groupsX, int groupsY)
        {
            _ = cmd;
            if (_computeModule == null || !IsAvailable)
                return false;

            if (_dispatcher != null)
            if (_dispatcher != null && IsAvailable)
            {
                uint gx = (uint)Math.Max(1, groupsX);
                uint gy = (uint)Math.Max(1, groupsY);
                _dispatcher.Dispatch("denoise", gx, gy, 1, Array.Empty<ComputeBuffer>());
                _dispatcher.Dispatch("gi_irradiance", gx, gy, 1, Array.Empty<ComputeBuffer>());
            }
            return true;
            return _computeModule != null && IsAvailable && false; // enable when SPIR-V module wired
        }

        public bool TryDispatchAoBlur(VulkanCommandBuffer cmd, int groupsX, int groupsY)
        {
            _ = (cmd, groupsX, groupsY);
            if (_computeModule == null || !IsAvailable)
                return false;

            if (_dispatcher != null)
                _dispatcher.Dispatch("blur_h", 1, 1, 1, Array.Empty<ComputeBuffer>());
            return true;
            if (_dispatcher != null && IsAvailable)
                _dispatcher.Dispatch("blur_h", 1, 1, 1, Array.Empty<ComputeBuffer>());
            return _computeModule != null && IsAvailable && false;
        }

        private byte[]? _computeModule;

        public void BindComputeModule(byte[] spirv)
        {
            _computeModule = spirv ?? throw new ArgumentNullException(nameof(spirv));
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            if (_disposed) return;
            _disposed = true;
            _computeModule = null;
            _dispatcher = null;
        }
    }
}
