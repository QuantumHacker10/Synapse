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
    /// Temporal stabilization system for frame-to-frame coherence.
    /// </summary>
    public class TemporalStabilizer
    {
        private required TemporalConfig _config;
        private required Vector3[] _historyBuffer;
        private required float[] _historyWeight;
        private required float[] _varianceBuffer;
        private int _width;
        private int _height;
        private int _historyLength;
        private bool _isInitialized;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;

        /// <summary>Current history length.</summary>
        public int HistoryLength => _historyLength;

        /// <summary>
        /// Initializes the temporal stabilizer.
        /// </summary>
        public void Initialize(int width, int height, TemporalConfig config)
        {
            _width = width;
            _height = height;
            _config = config;
            int pixelCount = width * height;
            _historyBuffer = new Vector3[pixelCount];
            _historyWeight = new float[pixelCount];
            _varianceBuffer = new float[pixelCount];
            _historyLength = 0;
            _isInitialized = true;
        }

        /// <summary>
        /// Reprojects the previous frame using motion vectors.
        /// </summary>
        public Vector3 Reproject(int x, int y, Vector2 velocity, GBuffer gbuffer)
        {
            if (_historyBuffer == null)
                return Vector3.Zero;

            float prevX = x + velocity.X;
            float prevY = y + velocity.Y;

            if (prevX < 0 || prevX >= _width || prevY < 0 || prevY >= _height)
                return Vector3.Zero;

            int prevIdx = (int)prevY * _width + (int)prevX;
            if (prevIdx >= 0 && prevIdx < _historyBuffer.Length && _historyWeight[prevIdx] > 0)
                return _historyBuffer[prevIdx];

            return Vector3.Zero;
        }

        /// <summary>
        /// Applies exponential moving average with adaptive blend factor.
        /// </summary>
        public Vector3 ApplyEMA(Vector3 current, Vector3 history, float blendFactor)
        {
            return Vector3.Lerp(history, current, 1.0f - blendFactor);
        }

        /// <summary>
        /// Performs variance clipping to prevent ghosting.
        /// </summary>
        public Vector3 VarianceClip(Vector3 current, Vector3 history, Vector3[] neighborhood,
            float clippingStrength)
        {
            if (neighborhood == null || neighborhood.Length == 0)
                return history;

            Vector3 mean = Vector3.Zero;
            Vector3 variance = Vector3.Zero;

            foreach (var sample in neighborhood)
                mean += sample;
            mean /= neighborhood.Length;

            foreach (var sample in neighborhood)
            {
                Vector3 diff = sample - mean;
                variance += new Vector3(diff.X * diff.X, diff.Y * diff.Y, diff.Z * diff.Z);
            }
            variance /= neighborhood.Length;

            Vector3 minBound = mean - new Vector3(
                MathF.Sqrt(variance.X) * clippingStrength,
                MathF.Sqrt(variance.Y) * clippingStrength,
                MathF.Sqrt(variance.Z) * clippingStrength);
            Vector3 maxBound = mean + new Vector3(
                MathF.Sqrt(variance.X) * clippingStrength,
                MathF.Sqrt(variance.Y) * clippingStrength,
                MathF.Sqrt(variance.Z) * clippingStrength);

            return Vector3.Clamp(history, minBound, maxBound);
        }

        /// <summary>
        /// Prevents ghosting by checking history confidence.
        /// </summary>
        public float ComputeGhostingPrevention(Vector3 current, Vector3 history,
            Vector2 velocity, GBuffer gbuffer, int x, int y)
        {
            float velocityMagnitude = velocity.Length();
            float motionConfidence = MathF.Exp(-velocityMagnitude * 10.0f);

            int idx = gbuffer.GetIndex(x, y);
            Vector3 currentNormal = gbuffer.Normals[idx];
            float currentDepth = gbuffer.Depth[idx];

            float depthDiff = 0;
            float normalDiff = 0;

            if (velocityMagnitude > 0.01f)
            {
                float prevX = Math.Clamp(x + (int)velocity.X, 0, _width - 1);
                float prevY = Math.Clamp(y + (int)velocity.Y, 0, _height - 1);
                int prevIdx = (int)prevY * _width + (int)prevX;

                if (prevIdx >= 0 && prevIdx < gbuffer.Depth.Length)
                {
                    float prevDepth = gbuffer.Depth[prevIdx];
                    depthDiff = MathF.Abs(currentDepth - prevDepth) / MathF.Max(0.001f, currentDepth);
                }
            }

            float depthConfidence = MathF.Exp(-depthDiff * 10.0f);
            return motionConfidence * depthConfidence;
        }

        /// <summary>
        /// Detects disocclusion using bilateral depth/stencil comparison.
        /// </summary>
        public bool DetectDisocclusion(GBuffer currentGBuffer, GBuffer previousGBuffer,
            int x, int y, Vector2 velocity, float threshold)
        {
            if (previousGBuffer == null)
                return true;

            float prevX = Math.Clamp(x + velocity.X, 0, _width - 1);
            float prevY = Math.Clamp(y + velocity.Y, 0, _height - 1);

            int currentIdx = currentGBuffer.GetIndex(x, y);
            int prevIdx = previousGBuffer.GetIndex((int)prevX, (int)prevY);

            float currentDepth = currentGBuffer.Depth[currentIdx];
            float prevDepth = previousGBuffer.Depth[prevIdx];

            Vector3 currentNormal = currentGBuffer.Normals[currentIdx];
            Vector3 prevNormal = previousGBuffer.Normals[prevIdx];

            float depthError = MathF.Abs(currentDepth - prevDepth) / MathF.Max(0.001f, currentDepth);
            float normalError = 1.0f - Vector3.Dot(currentNormal, prevNormal);

            return depthError > threshold || normalError > 0.5f;
        }

        /// <summary>
        /// Applies color box filter to history.
        /// </summary>
        public Vector3 ColorBoxFilterHistory(int x, int y, int radius)
        {
            if (_historyBuffer == null)
                return Vector3.Zero;

            Vector3 sum = Vector3.Zero;
            int count = 0;

            for (int dx = -radius; dx <= radius; dx++)
            {
                for (int dy = -radius; dy <= radius; dy++)
                {
                    int nx = Math.Clamp(x + dx, 0, _width - 1);
                    int ny = Math.Clamp(y + dy, 0, _height - 1);
                    int idx = ny * _width + nx;

                    if (_historyWeight[idx] > 0)
                    {
                        sum += _historyBuffer[idx];
                        count++;
                    }
                }
            }

            return count > 0 ? sum / count : Vector3.Zero;
        }

        /// <summary>
        /// Computes adaptive history length based on motion and variance.
        /// </summary>
        public int ComputeAdaptiveHistoryLength(Vector2 velocity, float variance)
        {
            float motionFactor = MathF.Exp(-velocity.Length() * 5.0f);
            float varianceFactor = MathF.Exp(-variance * 2.0f);
            float combinedFactor = motionFactor * varianceFactor;
            return Math.Clamp((int)(combinedFactor * _config.MaxHistoryLength), 1, _config.MaxHistoryLength);
        }

        /// <summary>
        /// Fast response to scene changes.
        /// </summary>
        public Vector3 FastResponse(Vector3 current, Vector3 history, float responseSpeed)
        {
            return Vector3.Lerp(history, current, responseSpeed);
        }

        /// <summary>
        /// Slow accumulation for stable regions.
        /// </summary>
        public Vector3 SlowAccumulation(Vector3 current, Vector3 history, float accumulationSpeed)
        {
            return Vector3.Lerp(current, history, accumulationSpeed);
        }

        /// <summary>
        /// Converts RGB to YCoCg color space for better filtering.
        /// </summary>
        public Vector3 RGBToYCoCg(Vector3 rgb)
        {
            float y = 0.25f * rgb.X + 0.5f * rgb.Y + 0.25f * rgb.Z;
            float co = 0.5f * rgb.X - 0.5f * rgb.Z;
            float cg = -0.25f * rgb.X + 0.5f * rgb.Y - 0.25f * rgb.Z;
            return new Vector3(y, co, cg);
        }

        /// <summary>
        /// Converts YCoCg back to RGB color space.
        /// </summary>
        public Vector3 YCoCgToRGB(Vector3 ycocg)
        {
            float r = ycocg.X + ycocg.Y - ycocg.Z;
            float g = ycocg.X + ycocg.Z;
            float b = ycocg.X - ycocg.Y - ycocg.Z;
            return new Vector3(r, g, b);
        }

        /// <summary>
        /// Temporal filter with full pipeline.
        /// </summary>
        public Vector3 ApplyTemporalFilter(int x, int y, Vector3 current, Vector2 velocity,
            GBuffer gbuffer, GBuffer previousGBuffer)
        {
            if (!_isInitialized || _historyBuffer == null)
                return current;

            int idx = gbuffer.GetIndex(x, y);

            bool disoccluded = DetectDisocclusion(gbuffer, previousGBuffer, x, y, velocity,
                _config.DisocclusionThreshold);

            if (disoccluded)
            {
                _historyBuffer[idx] = current;
                _historyWeight[idx] = 0;
                return current;
            }

            Vector3 history = Reproject(x, y, velocity, gbuffer);
            if (history.LengthSquared() < 0.0001f)
            {
                _historyBuffer[idx] = current;
                _historyWeight[idx] = 0;
                return current;
            }

            float ghostingConfidence = ComputeGhostingPrevention(current, history, velocity, gbuffer, x, y);

            float blendFactor = _config.BaseBlendFactor * ghostingConfidence;

            Vector3 filtered;
            switch (_config.Mode)
            {
                case TemporalFilterMode.ExponentialMovingAverage:
                    filtered = ApplyEMA(current, history, blendFactor);
                    break;

                case TemporalFilterMode.VarianceClipping:
                    Vector3[] neighborhood = GetNeighborhoodHistory(x, y, 1);
                    Vector3 clippedHistory = VarianceClip(current, history, neighborhood,
                        _config.VarianceClippingStrength);
                    filtered = ApplyEMA(current, clippedHistory, blendFactor);
                    break;

                case TemporalFilterMode.DisocclusionAware:
                    float responseBlend = FastResponse(current, history, _config.ResponseSpeed).Length();
                    float accumBlend = SlowAccumulation(current, history, _config.AccumulationSpeed).Length();
                    filtered = Vector3.Lerp(
                        FastResponse(current, history, _config.ResponseSpeed),
                        SlowAccumulation(current, history, _config.AccumulationSpeed),
                        ghostingConfidence);
                    break;

                default:
                    filtered = current;
                    break;
            }

            _historyBuffer[idx] = filtered;
            _historyWeight[idx] = MathF.Min(_historyWeight[idx] + 1, _config.MaxHistoryLength);

            return filtered;
        }

        private Vector3[] GetNeighborhoodHistory(int x, int y, int radius)
        {
            var neighborhood = new List<Vector3>();
            for (int dx = -radius; dx <= radius; dx++)
            {
                for (int dy = -radius; dy <= radius; dy++)
                {
                    int nx = Math.Clamp(x + dx, 0, _width - 1);
                    int ny = Math.Clamp(y + dy, 0, _height - 1);
                    int idx = ny * _width + nx;
                    if (_historyWeight[idx] > 0)
                        neighborhood.Add(_historyBuffer[idx]);
                }
            }
            return neighborhood.ToArray();
        }
    }
}
