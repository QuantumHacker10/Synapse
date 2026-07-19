// L-DNN neural global illumination subsystem (split from LDNNRenderer.cs).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace GDNN.Lighting.LDNN
{

    /// <summary>
    /// Analytics system for tracking performance and quality metrics.
    /// </summary>
    public class LDNNAnalytics
    {
        private readonly Queue<float> _frameTimeHistory = new(120);
        private readonly Queue<float> _cascadeTimeHistory = new(120);
        private readonly Queue<float> _neuralTimeHistory = new(120);
        private readonly Queue<int> _raysTracedHistory = new(120);
        private readonly Queue<float> _qualityMetricsHistory = new(120);
        private readonly object _lock = new();
        private int _totalFramesTraced;
        private long _totalRaysTraced;
        private float _averagePSNR;
        private float _averageSSIM;
        private int _convergenceFrameCount;
        private float _lastQualityScore;

        /// <summary>Average frame time over the last 120 frames.</summary>
        public float AverageFrameTimeMs { get; private set; }
        /// <summary>Average cascade rendering time.</summary>
        public float AverageCascadeTimeMs { get; private set; }
        /// <summary>Average neural prediction time.</summary>
        public float AverageNeuralTimeMs { get; private set; }
        /// <summary>Average rays traced per frame.</summary>
        public float AverageRaysPerFrame { get; private set; }
        /// <summary>Current quality score (0-1).</summary>
        public float CurrentQualityScore { get; private set; }
        /// <summary>Performance budget utilization (0-1).</summary>
        public float BudgetUtilization { get; private set; }
        /// <summary>Total frames rendered.</summary>
        public int TotalFramesRendered => _totalFramesTraced;
        /// <summary>Total rays traced across all frames.</summary>
        public long TotalRaysTraced => _totalRaysTraced;

        /// <summary>
        /// Records a frame's telemetry data.
        /// </summary>
        public void RecordFrame(FrameTelemetry telemetry)
        {
            lock (_lock)
            {
                _frameTimeHistory.Enqueue(telemetry.TotalFrameTimeMs);
                if (_frameTimeHistory.Count > 120) _frameTimeHistory.Dequeue();

                _cascadeTimeHistory.Enqueue(telemetry.CascadeRenderTimeMs);
                if (_cascadeTimeHistory.Count > 120) _cascadeTimeHistory.Dequeue();

                _neuralTimeHistory.Enqueue(telemetry.NeuralPredictionTimeMs);
                if (_neuralTimeHistory.Count > 120) _neuralTimeHistory.Dequeue();

                _raysTracedHistory.Enqueue(telemetry.RaysTraced);
                if (_raysTracedHistory.Count > 120) _raysTracedHistory.Dequeue();

                _totalFramesTraced++;
                _totalRaysTraced += telemetry.RaysTraced;

                RecomputeAverages();
            }
        }

        private void RecomputeAverages()
        {
            if (_frameTimeHistory.Count == 0) return;

            float sumFrame = 0, sumCascade = 0, sumNeural = 0;
            int sumRays = 0;
            foreach (float f in _frameTimeHistory) sumFrame += f;
            foreach (float c in _cascadeTimeHistory) sumCascade += c;
            foreach (float n in _neuralTimeHistory) sumNeural += n;
            foreach (int r in _raysTracedHistory) sumRays += r;

            int count = _frameTimeHistory.Count;
            AverageFrameTimeMs = sumFrame / count;
            AverageCascadeTimeMs = sumCascade / count;
            AverageNeuralTimeMs = sumNeural / count;
            AverageRaysPerFrame = (float)sumRays / count;
        }

        /// <summary>
        /// Records quality metrics for reference comparison.
        /// </summary>
        public void RecordQualityMetrics(float psnr, float ssim)
        {
            lock (_lock)
            {
                _averagePSNR = _averagePSNR * 0.95f + psnr * 0.05f;
                _averageSSIM = _averageSSIM * 0.95f + ssim * 0.05f;
                _lastQualityScore = CalculateQualityScore(psnr, ssim);
                _qualityMetricsHistory.Enqueue(_lastQualityScore);
                if (_qualityMetricsHistory.Count > 120) _qualityMetricsHistory.Dequeue();
            }
        }

        private float CalculateQualityScore(float psnr, float ssim)
        {
            float psnrNormalized = MathF.Min(1.0f, MathF.Max(0.0f, (psnr - 20.0f) / 30.0f));
            return psnrNormalized * 0.5f + ssim * 0.5f;
        }

        /// <summary>
        /// Computes an adaptive quality target based on performance history.
        /// </summary>
        public AdaptiveQualityTarget ComputeAdaptiveTarget(float targetFrameTimeMs)
        {
            lock (_lock)
            {
                float currentFrameTime = AverageFrameTimeMs;
                float headroom = (targetFrameTimeMs - currentFrameTime) / targetFrameTimeMs;
                float qualityScale = MathF.Max(0.25f, MathF.Min(1.5f, 1.0f + headroom * 0.5f));

                return new AdaptiveQualityTarget
                {
                    TargetFrameTimeMs = targetFrameTimeMs,
                    CurrentFrameTimeMs = currentFrameTime,
                    QualityScale = qualityScale,
                    ReduceCascadeCount = currentFrameTime > targetFrameTimeMs * 1.1f,
                    ReduceRayCount = currentFrameTime > targetFrameTimeMs * 1.05f,
                    UseLowerResolution = currentFrameTime > targetFrameTimeMs * 1.2f,
                    PerformanceHeadroom = headroom,
                    DenoiserStrength = MathF.Max(0.3f, MathF.Min(1.0f, 1.0f - headroom * 0.3f))
                };
            }
        }

        /// <summary>
        /// Gets performance statistics as a formatted string.
        /// </summary>
        public string GetPerformanceReport()
        {
            lock (_lock)
            {
                return $"Frame: {AverageFrameTimeMs:F2}ms | Cascade: {AverageCascadeTimeMs:F2}ms | " +
                       $"Neural: {AverageNeuralTimeMs:F2}ms | Rays: {AverageRaysPerFrame:F0} | " +
                       $"Quality: {CurrentQualityScore:F3} | Frames: {_totalFramesTraced}";
            }
        }

        /// <summary>
        /// Resets all analytics data.
        /// </summary>
        public void Reset()
        {
            lock (_lock)
            {
                _frameTimeHistory.Clear();
                _cascadeTimeHistory.Clear();
                _neuralTimeHistory.Clear();
                _raysTracedHistory.Clear();
                _qualityMetricsHistory.Clear();
                _totalFramesTraced = 0;
                _totalRaysTraced = 0;
                _averagePSNR = 0;
                _averageSSIM = 0;
                _convergenceFrameCount = 0;
                _lastQualityScore = 0;
                AverageFrameTimeMs = 0;
                AverageCascadeTimeMs = 0;
                AverageNeuralTimeMs = 0;
                AverageRaysPerFrame = 0;
                CurrentQualityScore = 0;
                BudgetUtilization = 0;
            }
        }
    }
}
