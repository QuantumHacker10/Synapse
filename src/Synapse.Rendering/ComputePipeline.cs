// =============================================================================
// ComputePipeline.cs - GDNN Engine: GPU Compute Dispatch
// Compute shader pipeline for GI, post-process, and general GPU compute
// =============================================================================

using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using GDNN.RHI.Vulkan;

namespace GDNN.Rendering.Compute
{
    public enum ComputeKernelType { Generic = 0, Blur = 1, Tonemap = 2, Bloom = 3, Convolution = 4, Reduction = 5 }

    public record ComputePipelineDesc
    {
        public string Name { get; init; } = "";
        public uint GroupCountX { get; init; } = 1;
        public uint GroupCountY { get; init; } = 1;
        public uint GroupCountZ { get; init; } = 1;
        public uint LocalSizeX { get; init; } = 16;
        public uint LocalSizeY { get; init; } = 16;
        public uint LocalSizeZ { get; init; } = 1;
        public ComputeKernelType KernelType { get; init; }
    }

    public class ComputeBuffer : IDisposable
    {
        public IntPtr Handle { get; set; }
        public VulkanBuffer Buffer { get; set; }
        public uint ElementCount { get; set; }
        public uint Stride { get; set; }
        public bool IsStorageBuffer { get; set; }
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Buffer?.Dispose();
        }
    }

    public class ComputeTexture : IDisposable
    {
        public IntPtr Handle { get; set; }
        public VulkanTexture Texture { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Texture?.Dispose();
        }
    }

    public class ComputeJob
    {
        public int JobId { get; set; }
        public string KernelName { get; set; }
        public uint GroupCountX { get; set; }
        public uint GroupCountY { get; set; }
        public uint GroupCountZ { get; set; }
        public float[] PushConstants { get; set; }
        public ComputeBuffer[] Buffers { get; set; }
        public ComputeTexture[] Textures { get; set; }
        public ManualResetEventSlim CompletionEvent { get; set; }
        public bool IsCompleted { get; set; }
        public Exception Error { get; set; }
    }

    public class ComputeDispatcher : IDisposable
    {
        private VulkanRhiDevice _rhi;
        private bool _disposed;
        private ConcurrentQueue<ComputeJob> _pendingJobs;
        private ConcurrentQueue<ComputeJob> _completedJobs;
        private int _nextJobId;
        private Thread _computeThread;
        private bool _running;
        private readonly object _lock = new();

        private Dictionary<string, ComputePipelineDesc> _kernels;
        private Dictionary<string, float[]> _kernelCache;

        public int PendingJobCount => _pendingJobs.Count;
        public int CompletedJobCount => _completedJobs.Count;

        public ComputeDispatcher(VulkanRhiDevice rhi)
        {
            _rhi = rhi;
            _pendingJobs = new ConcurrentQueue<ComputeJob>();
            _completedJobs = new ConcurrentQueue<ComputeJob>();
            _kernels = new Dictionary<string, ComputePipelineDesc>();
            _kernelCache = new Dictionary<string, float[]>();
            _running = true;

            RegisterBuiltinKernels();
            StartComputeThread();
        }

        private void RegisterBuiltinKernels()
        {
            _kernels["blur_h"] = new ComputePipelineDesc { Name = "blur_h", LocalSizeX = 256, LocalSizeY = 1, KernelType = ComputeKernelType.Blur };
            _kernels["blur_v"] = new ComputePipelineDesc { Name = "blur_v", LocalSizeX = 1, LocalSizeY = 256, KernelType = ComputeKernelType.Blur };
            _kernels["bloom_downsample"] = new ComputePipelineDesc { Name = "bloom_downsample", LocalSizeX = 8, LocalSizeY = 8, KernelType = ComputeKernelType.Bloom };
            _kernels["bloom_upsample"] = new ComputePipelineDesc { Name = "bloom_upsample", LocalSizeX = 8, LocalSizeY = 8, KernelType = ComputeKernelType.Bloom };
            _kernels["tonemap"] = new ComputePipelineDesc { Name = "tonemap", LocalSizeX = 16, LocalSizeY = 16, KernelType = ComputeKernelType.Tonemap };
            _kernels["convolution"] = new ComputePipelineDesc { Name = "convolution", LocalSizeX = 16, LocalSizeY = 16, KernelType = ComputeKernelType.Convolution };
            _kernels["reduction"] = new ComputePipelineDesc { Name = "reduction", LocalSizeX = 256, LocalSizeY = 1, KernelType = ComputeKernelType.Reduction };
            _kernels["gi_radiance"] = new ComputePipelineDesc { Name = "gi_radiance", LocalSizeX = 8, LocalSizeY = 8, KernelType = ComputeKernelType.Generic };
            _kernels["gi_irradiance"] = new ComputePipelineDesc { Name = "gi_irradiance", LocalSizeX = 8, LocalSizeY = 8, KernelType = ComputeKernelType.Generic };
            _kernels["denoise"] = new ComputePipelineDesc { Name = "denoise", LocalSizeX = 16, LocalSizeY = 16, KernelType = ComputeKernelType.Generic };
            _kernels["ssr_trace"] = new ComputePipelineDesc { Name = "ssr_trace", LocalSizeX = 8, LocalSizeY = 8, KernelType = ComputeKernelType.Generic };
            _kernels["shadow_filter"] = new ComputePipelineDesc { Name = "shadow_filter", LocalSizeX = 16, LocalSizeY = 16, KernelType = ComputeKernelType.Generic };
            _kernels["velocity_reproject"] = new ComputePipelineDesc { Name = "velocity_reproject", LocalSizeX = 8, LocalSizeY = 8, KernelType = ComputeKernelType.Generic };
            _kernels["taa_resolve"] = new ComputePipelineDesc { Name = "taa_resolve", LocalSizeX = 8, LocalSizeY = 8, KernelType = ComputeKernelType.Generic };
            _kernels["particle_update"] = new ComputePipelineDesc { Name = "particle_update", LocalSizeX = 256, LocalSizeY = 1, KernelType = ComputeKernelType.Generic };
        }

        private void StartComputeThread()
        {
            _computeThread = new Thread(ComputeThreadLoop)
            {
                IsBackground = true,
                Priority = ThreadPriority.AboveNormal
            };
            _computeThread.Start();
        }

        private void ComputeThreadLoop()
        {
            while (_running)
            {
                if (_pendingJobs.TryDequeue(out var job))
                {
                    try
                    {
                        ExecuteJobCPU(job);
                        job.IsCompleted = true;
                    }
                    catch (Exception ex)
                    {
                        job.Error = ex;
                        job.IsCompleted = true;
                    }

                    _completedJobs.Enqueue(job);
                    job.CompletionEvent?.Set();
                }
                else
                {
                    Thread.Sleep(1);
                }
            }
        }

        private void ExecuteJobCPU(ComputeJob job)
        {
            if (job.Buffers == null || job.Buffers.Length == 0) return;

            var kernel = _kernels.GetValueOrDefault(job.KernelName);
            if (kernel == null) return;

            switch (kernel.KernelType)
            {
                case ComputeKernelType.Blur:
                    ExecuteBlurJob(job, kernel);
                    break;
                case ComputeKernelType.Tonemap:
                    ExecuteTonemapJob(job);
                    break;
                case ComputeKernelType.Bloom:
                    ExecuteBloomJob(job, kernel);
                    break;
                case ComputeKernelType.Convolution:
                    ExecuteConvolutionJob(job);
                    break;
                case ComputeKernelType.Reduction:
                    ExecuteReductionJob(job);
                    break;
                default:
                    ExecuteGenericJob(job);
                    break;
            }
        }

        private void ExecuteBlurJob(ComputeJob job, ComputePipelineDesc kernel)
        {
            if (job.Buffers.Length < 2) return;
            var input = job.Buffers[0];
            var output = job.Buffers[1];

            float[] inputData = ReadBuffer(input);
            float[] outputData = new float[inputData.Length];

            bool horizontal = kernel.Name.EndsWith("_h");
            int width = (int)job.GroupCountX * (int)kernel.LocalSizeX;
            int height = (int)job.GroupCountY * (int)kernel.LocalSizeY;

            if (width <= 0) width = 1;
            if (height <= 0) height = 1;

            float sigma = job.PushConstants?.Length > 0 ? job.PushConstants[0] : 3.0f;
            int radius = Math.Max(1, (int)(sigma * 2));
            float[] kernelWeights = GenerateGaussian1D(radius, sigma);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Vector4 sum = Vector4.Zero;
                    float weightSum = 0;

                    for (int i = -radius; i <= radius; i++)
                    {
                        float weight = kernelWeights[i + radius];
                        int sx = horizontal ? Math.Clamp(x + i, 0, width - 1) : x;
                        int sy = horizontal ? y : Math.Clamp(y + i, 0, height - 1);
                        int idx = (sy * width + sx) * 4;

                        if (idx + 3 < inputData.Length)
                        {
                            sum += new Vector4(inputData[idx], inputData[idx + 1], inputData[idx + 2], inputData[idx + 3]) * weight;
                            weightSum += weight;
                        }
                    }

                    int oidx = (y * width + x) * 4;
                    if (oidx + 3 < outputData.Length)
                    {
                        outputData[oidx] = sum.X / weightSum;
                        outputData[oidx + 1] = sum.Y / weightSum;
                        outputData[oidx + 2] = sum.Z / weightSum;
                        outputData[oidx + 3] = sum.W / weightSum;
                    }
                }
            }

            WriteBuffer(output, outputData);
        }

        private void ExecuteTonemapJob(ComputeJob job)
        {
            if (job.Buffers.Length < 2) return;
            var input = job.Buffers[0];
            var output = job.Buffers[1];

            float[] inputData = ReadBuffer(input);
            float[] outputData = new float[inputData.Length];

            float exposure = job.PushConstants?.Length > 0 ? job.PushConstants[0] : 1.0f;
            float gamma = job.PushConstants?.Length > 1 ? job.PushConstants[1] : 2.2f;

            for (int i = 0; i < inputData.Length; i += 4)
            {
                Vector3 hdr = new Vector3(inputData[i] * exposure, inputData[i + 1] * exposure, inputData[i + 2] * exposure);
                Vector3 mapped = ACESFilm(hdr);
                mapped = new Vector3(MathF.Pow(mapped.X, 1.0f / gamma), MathF.Pow(mapped.Y, 1.0f / gamma), MathF.Pow(mapped.Z, 1.0f / gamma));
                outputData[i] = Math.Clamp(mapped.X, 0, 1);
                outputData[i + 1] = Math.Clamp(mapped.Y, 0, 1);
                outputData[i + 2] = Math.Clamp(mapped.Z, 0, 1);
                outputData[i + 3] = inputData[i + 3];
            }

            WriteBuffer(output, outputData);
        }

        private void ExecuteBloomJob(ComputeJob job, ComputePipelineDesc kernel)
        {
            if (job.Buffers.Length < 2) return;
            bool downsample = kernel.Name.Contains("down");
            var input = job.Buffers[0];
            var output = job.Buffers[1];

            float[] inputData = ReadBuffer(input);
            int srcW = (int)job.GroupCountX * (int)kernel.LocalSizeX;
            int srcH = (int)job.GroupCountY * (int)kernel.LocalSizeY;
            if (srcW <= 0) srcW = 1;
            if (srcH <= 0) srcH = 1;

            int dstW = downsample ? srcW / 2 : srcW * 2;
            int dstH = downsample ? srcH / 2 : srcH * 2;
            if (dstW <= 0) dstW = 1;
            if (dstH <= 0) dstH = 1;

            float[] outputData = new float[dstW * dstH * 4];

            for (int y = 0; y < dstH; y++)
            {
                for (int x = 0; x < dstW; x++)
                {
                    Vector4 sum = Vector4.Zero;
                    if (downsample)
                    {
                        int sx = x * 2;
                        int sy = y * 2;
                        sum += SamplePixel(inputData, srcW, srcH, sx, sy);
                        sum += SamplePixel(inputData, srcW, srcH, sx + 1, sy);
                        sum += SamplePixel(inputData, srcW, srcH, sx, sy + 1);
                        sum += SamplePixel(inputData, srcW, srcH, sx + 1, sy + 1);
                        sum *= 0.25f;
                    }
                    else
                    {
                        int sx = x / 2;
                        int sy = y / 2;
                        sum = SamplePixel(inputData, srcW, srcH, sx, sy);
                    }

                    int oidx = (y * dstW + x) * 4;
                    outputData[oidx] = sum.X;
                    outputData[oidx + 1] = sum.Y;
                    outputData[oidx + 2] = sum.Z;
                    outputData[oidx + 3] = sum.W;
                }
            }

            WriteBuffer(output, outputData);
        }

        private void ExecuteConvolutionJob(ComputeJob job)
        {
            if (job.Buffers.Length < 3) return;
            var input = job.Buffers[0];
            var kernelBuf = job.Buffers[1];
            var output = job.Buffers[2];

            float[] inputData = ReadBuffer(input);
            float[] kernelData = ReadBuffer(kernelBuf);
            float[] outputData = new float[inputData.Length];

            int width = (int)MathF.Sqrt(inputData.Length / 4);
            int kSize = (int)MathF.Sqrt(kernelData.Length);
            int kRadius = kSize / 2;

            for (int y = 0; y < width; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Vector4 sum = Vector4.Zero;
                    for (int ky = -kRadius; ky <= kRadius; ky++)
                    {
                        for (int kx = -kRadius; kx <= kRadius; kx++)
                        {
                            int sx = Math.Clamp(x + kx, 0, width - 1);
                            int sy = Math.Clamp(y + ky, 0, width - 1);
                            float w = kernelData[(ky + kRadius) * kSize + (kx + kRadius)];
                            sum += SamplePixel(inputData, width, width, sx, sy) * w;
                        }
                    }
                    int oidx = (y * width + x) * 4;
                    outputData[oidx] = sum.X;
                    outputData[oidx + 1] = sum.Y;
                    outputData[oidx + 2] = sum.Z;
                    outputData[oidx + 3] = sum.W;
                }
            }

            WriteBuffer(output, outputData);
        }

        private void ExecuteReductionJob(ComputeJob job)
        {
            if (job.Buffers.Length < 1) return;
            var input = job.Buffers[0];
            float[] data = ReadBuffer(input);

            float sum = 0;
            for (int i = 0; i < data.Length; i += 4)
                sum += 0.2126f * data[i] + 0.7152f * data[i + 1] + 0.0722f * data[i + 2];

            if (job.Buffers.Length > 1)
                WriteBuffer(job.Buffers[1], new float[] { sum / (data.Length / 4) });
        }

        private void ExecuteGenericJob(ComputeJob job)
        {
            if (job.Buffers.Length < 2) return;
            var input = job.Buffers[0];
            var output = job.Buffers[1];

            float[] inputData = ReadBuffer(input);
            float[] outputData = new float[inputData.Length];
            Buffer.BlockCopy(inputData, 0, outputData, 0, inputData.Length * sizeof(float));

            if (job.Textures != null)
            {
                foreach (var tex in job.Textures)
                {
                    if (tex?.Texture != null)
                        outputData[0] = tex.Width;
                }
            }

            WriteBuffer(output, outputData);
        }

        private Vector4 SamplePixel(float[] data, int width, int height, int x, int y)
        {
            x = Math.Clamp(x, 0, width - 1);
            y = Math.Clamp(y, 0, height - 1);
            int idx = (y * width + x) * 4;
            if (idx + 3 >= data.Length) return Vector4.Zero;
            return new Vector4(data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);
        }

        private float[] GenerateGaussian1D(int radius, float sigma)
        {
            float[] kernel = new float[radius * 2 + 1];
            float sum = 0;
            for (int i = -radius; i <= radius; i++)
            {
                float val = MathF.Exp(-(i * i) / (2 * sigma * sigma));
                kernel[i + radius] = val;
                sum += val;
            }
            for (int i = 0; i < kernel.Length; i++) kernel[i] /= sum;
            return kernel;
        }

        private Vector3 ACESFilm(Vector3 x)
        {
            float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
            return Vector3.Clamp((x * (a * x + new Vector3(b))) / (x * (c * x + new Vector3(d)) + new Vector3(e)), Vector3.Zero, Vector3.One);
        }

        private float[] ReadBuffer(ComputeBuffer buf)
        {
            if (buf.Buffer == null) return Array.Empty<float>();
            try
            {
                var mapped = buf.Buffer.Map();
                int byteCount = (int)(buf.ElementCount * buf.Stride);
                float[] result = new float[byteCount / 4];
                Marshal.Copy(mapped, MemoryMarshal.AsBytes(result.AsSpan()).ToArray(), 0, Math.Min(byteCount, result.Length * 4));
                buf.Buffer.Unmap();
                return result;
            }
            catch { return new float[buf.ElementCount * buf.Stride / 4]; }
        }

        private void WriteBuffer(ComputeBuffer buf, float[] data)
        {
            if (buf.Buffer == null) return;
            try
            {
                var mapped = buf.Buffer.Map();
                int byteCount = Math.Min(data.Length * 4, (int)(buf.ElementCount * buf.Stride));
                var bytes = MemoryMarshal.AsBytes(data.AsSpan().Slice(0, data.Length));
                unsafe
                {
                    fixed (byte* src = bytes)
                        System.Buffer.MemoryCopy(src, (void*)mapped, byteCount, byteCount);
                }
                buf.Buffer.Unmap();
            }
            catch { }
        }

        public int Dispatch(string kernelName, uint groupsX, uint groupsY, uint groupsZ, ComputeBuffer[] buffers, ComputeTexture[] textures = null, float[] pushConstants = null)
        {
            int jobId = Interlocked.Increment(ref _nextJobId);
            var job = new ComputeJob
            {
                JobId = jobId,
                KernelName = kernelName,
                GroupCountX = groupsX,
                GroupCountY = groupsY,
                GroupCountZ = groupsZ,
                Buffers = buffers,
                Textures = textures,
                PushConstants = pushConstants,
                CompletionEvent = new ManualResetEventSlim(false)
            };

            _pendingJobs.Enqueue(job);
            return jobId;
        }

        public bool IsJobComplete(int jobId)
        {
            foreach (var job in _completedJobs)
                if (job.JobId == jobId) return true;
            return false;
        }

        public ComputeJob WaitForJob(int jobId, int timeoutMs = 5000)
        {
            foreach (var job in _completedJobs)
                if (job.JobId == jobId) return job;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                foreach (var job in _completedJobs)
                    if (job.JobId == jobId) return job;
                Thread.Sleep(1);
            }

            return null;
        }

        public void DispatchBlur(ComputeBuffer input, ComputeBuffer output, float sigma, bool horizontal)
        {
            string kernel = horizontal ? "blur_h" : "blur_v";
            if (!_kernels.ContainsKey(kernel)) return;

            var desc = _kernels[kernel];
            uint groupsX = (uint)MathF.Ceiling(1.0f / desc.LocalSizeX);
            uint groupsY = (uint)MathF.Ceiling(1.0f / desc.LocalSizeY);
            Dispatch(kernel, groupsX, groupsY, 1, new[] { input, output }, pushConstants: new[] { sigma });
        }

        public void DispatchTonemap(ComputeBuffer input, ComputeBuffer output, float exposure, float gamma)
        {
            if (!_kernels.ContainsKey("tonemap")) return;
            var desc = _kernels["tonemap"];
            uint groupsX = (uint)MathF.Ceiling(1.0f / desc.LocalSizeX);
            uint groupsY = (uint)MathF.Ceiling(1.0f / desc.LocalSizeY);
            Dispatch("tonemap", groupsX, groupsY, 1, new[] { input, output }, pushConstants: new[] { exposure, gamma });
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _running = false;
            _computeThread?.Join(1000);
        }
    }
}
