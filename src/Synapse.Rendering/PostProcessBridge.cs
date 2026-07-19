using System;
using System.Numerics;
using GDNN.Rendering.PostProcess;
using GDNN.RHI.Vulkan;

namespace GDNN.Rendering.Bridge
{
    public class PostProcessBridge : IDisposable
    {
        private PostProcessingPipeline _pipeline;
        private HDRFrameBuffer _hdrBuffer;
        private DepthBuffer _depthBuffer;
        private HDRFrameBuffer _outputBuffer;
        private int _width;
        private int _height;
        private bool _disposed;

        public PostProcessConfig Config
        {
            get => _pipeline?.Config;
            set { if (_pipeline != null) _pipeline.Config = value; }
        }

        public float CurrentExposure => _pipeline?.CurrentExposure ?? 1.0f;

        public PostProcessBridge(int width, int height)
        {
            _width = width;
            _height = height;

            _pipeline = new PostProcessingPipeline(new PostProcessConfig
            {
                Bloom = new BloomConfig
                {
                    Enabled = true,
                    Threshold = 1.0f,
                    Knee = 0.5f,
                    Intensity = 0.4f,
                    Radius = 0.8f,
                    Quality = BloomQuality.High,
                    MaxIterations = 6,
                    HighPrecision = true
                },
                DOF = new DOFConfig
                {
                    Enabled = false,
                    FocusDistance = 10.0f,
                    Aperture = 1.4f,
                    FocalLength = 50.0f
                },
                MotionBlur = new MotionBlurConfig
                {
                    Enabled = false,
                    Intensity = 1.0f,
                    SampleCount = 8,
                    Quality = MotionBlurQuality.Medium
                },
                Tonemap = new TonemapConfig
                {
                    Enabled = true,
                    Operator = TonemapOperator.ACES,
                    Exposure = 1.2f,
                    Gamma = 2.2f,
                    WhitePoint = 4.0f
                }
            });

            _hdrBuffer = new HDRFrameBuffer(width, height);
            _depthBuffer = new DepthBuffer(width, height);
            _outputBuffer = new HDRFrameBuffer(width, height);
        }

        public void Resize(int width, int height)
        {
            _width = width;
            _height = height;
            _hdrBuffer?.Dispose();
            _depthBuffer?.Dispose();
            _outputBuffer?.Dispose();
            _hdrBuffer = new HDRFrameBuffer(width, height);
            _depthBuffer = new DepthBuffer(width, height);
            _outputBuffer = new HDRFrameBuffer(width, height);
        }

        public unsafe void UploadHDRFromBuffer(byte* srcData, int srcStride, int width, int height)
        {
            if (width != _width || height != _height)
                Resize(width, height);

            _hdrBuffer.Clear();
            var dst = _hdrBuffer.AsVector4Span();
            int dstIdx = 0;

            for (int y = 0; y < height; y++)
            {
                var srcRow = srcData + y * srcStride;
                for (int x = 0; x < width; x++)
                {
                    var pixel = (Vector4*)(srcRow + x * sizeof(Vector4));
                    dst[dstIdx++] = *pixel;
                }
            }
        }

        public unsafe void UploadDepthFromBuffer(float* srcData, int width, int height)
        {
            if (width != _width || height != _height)
                Resize(width, height);

            _depthBuffer.Clear();
            var dst = _depthBuffer.Data;
            int count = width * height;
            fixed (float* dstPtr = dst)
            {
                System.Buffer.MemoryCopy(srcData, dstPtr, count * sizeof(float), count * sizeof(float));
            }
        }

        public void UploadHDRFromColor3(Vector3[,] colorData, int width, int height)
        {
            if (width != _width || height != _height)
                Resize(width, height);

            _hdrBuffer.Clear();
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    _hdrBuffer.SetPixel(x, y, new Vector4(colorData[x, y].X, colorData[x, y].Y, colorData[x, y].Z, 1.0f));
        }

        public void UploadHDRFromFloats(float[] rgbaData, int width, int height)
        {
            if (width != _width || height != _height)
                Resize(width, height);

            _hdrBuffer.Clear();
            var span = _hdrBuffer.AsVector4Span();
            int pixelCount = width * height;
            for (int i = 0; i < pixelCount && i * 4 + 3 < rgbaData.Length; i++)
            {
                span[i] = new Vector4(rgbaData[i * 4], rgbaData[i * 4 + 1], rgbaData[i * 4 + 2], rgbaData[i * 4 + 3]);
            }
        }

        public void UploadDepth(float[] depthData, int width, int height)
        {
            if (width != _width || height != _height)
                Resize(width, height);

            _depthBuffer.Clear();
            var dst = _depthBuffer.Data;
            int count = Math.Min(depthData.Length, width * height);
            Array.Copy(depthData, dst, count);
        }

        public void Process(float aspectRatio = 16.0f / 9.0f)
        {
            _outputBuffer.Clear();
            _pipeline.Process(_hdrBuffer, _depthBuffer, null, _outputBuffer, aspectRatio);
        }

        public float GetOutputPixelR(int x, int y)
        {
            if (x < 0 || x >= _width || y < 0 || y >= _height)
                return 0;
            var pixel = _outputBuffer.GetPixel(x, y);
            return pixel.X;
        }

        public float GetOutputPixelG(int x, int y)
        {
            if (x < 0 || x >= _width || y < 0 || y >= _height)
                return 0;
            var pixel = _outputBuffer.GetPixel(x, y);
            return pixel.Y;
        }

        public float GetOutputPixelB(int x, int y)
        {
            if (x < 0 || x >= _width || y < 0 || y >= _height)
                return 0;
            var pixel = _outputBuffer.GetPixel(x, y);
            return pixel.Z;
        }

        public HDRFrameBuffer GetOutputBuffer() => _outputBuffer;
        public HDRFrameBuffer GetHDRBuffer() => _hdrBuffer;
        public DepthBuffer GetDepthBuffer() => _depthBuffer;

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _pipeline?.Dispose();
            _hdrBuffer?.Dispose();
            _depthBuffer?.Dispose();
            _outputBuffer?.Dispose();
        }
    }
}
