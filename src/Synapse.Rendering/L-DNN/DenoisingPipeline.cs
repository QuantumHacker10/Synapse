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
    /// Multi-stage denoising pipeline for noise reduction in GI.
    /// </summary>
    public class DenoisingPipeline
    {
        private DenoiseConfig _config;
        private Vector3[] _tempBufferA;
        private Vector3[] _tempBufferB;
        private float[] _varianceBuffer;
        private int _width;
        private int _height;
        private bool _isInitialized;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>
        /// Initializes the denoising pipeline.
        /// </summary>
        public void Initialize(int width, int height, DenoiseConfig config)
        {
            _width = width;
            _height = height;
            _config = config;
            int pixelCount = width * height;
            _tempBufferA = new Vector3[pixelCount];
            _tempBufferB = new Vector3[pixelCount];
            _varianceBuffer = new float[pixelCount];
            _isInitialized = true;
        }

        /// <summary>
        /// Applies bilateral spatial filter.
        /// </summary>
        public void SpatialBilateralFilter(Vector3[] input, Vector3[] output, GBuffer gbuffer,
            int radius, float normalThreshold, float depthThreshold)
        {
            Parallel.For(radius, _height - radius, y =>
            {
                for (int x = radius; x < _width - radius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    Vector3 centerNormal = gbuffer.Normals[idx];
                    float centerDepth = gbuffer.Depth[idx];
                    Vector3 centerColor = input[idx];

                    Vector3 sum = Vector3.Zero;
                    float weightSum = 0;

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);

                            float depthDiff = MathF.Abs(centerDepth - gbuffer.Depth[nIdx]) /
                                             MathF.Max(0.001f, centerDepth);
                            float normalDiff = 1.0f - Vector3.Dot(centerNormal, gbuffer.Normals[nIdx]);
                            Vector3 colorDiff = centerColor - input[nIdx];
                            float colorDist = colorDiff.Length();

                            if (depthDiff > depthThreshold || normalDiff > normalThreshold)
                                continue;

                            float spatialWeight = MathF.Exp(-(dx * dx + dy * dy) / (2.0f * radius * radius));
                            float depthWeight = MathF.Exp(-depthDiff * 10.0f);
                            float normalWeight = MathF.Exp(-normalDiff * 5.0f);
                            float colorWeight = MathF.Exp(-colorDist * 3.0f);

                            float weight = spatialWeight * depthWeight * normalWeight * colorWeight;
                            sum += input[nIdx] * weight;
                            weightSum += weight;
                        }
                    }

                    output[idx] = weightSum > 0 ? sum / weightSum : centerColor;
                }
            });
        }

        /// <summary>
        /// Applies non-local means denoiser.
        /// </summary>
        public void NonLocalMeansDenoiser(Vector3[] input, Vector3[] output, GBuffer gbuffer,
            int patchRadius, int searchRadius, float h)
        {
            Parallel.For(searchRadius, _height - searchRadius, y =>
            {
                for (int x = searchRadius; x < _width - searchRadius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    Vector3 centerColor = input[idx];

                    Vector3 sum = Vector3.Zero;
                    float weightSum = 0;

                    for (int dx = -searchRadius; dx <= searchRadius; dx++)
                    {
                        for (int dy = -searchRadius; dy <= searchRadius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;

                            float patchDistance = ComputePatchDistance(input, x, y, nx, ny,
                                patchRadius, gbuffer);

                            float weight = MathF.Exp(-patchDistance / (h * h));
                            int nIdx = gbuffer.GetIndex(nx, ny);
                            sum += input[nIdx] * weight;
                            weightSum += weight;
                        }
                    }

                    output[idx] = weightSum > 0 ? sum / weightSum : centerColor;
                }
            });
        }

        private float ComputePatchDistance(Vector3[] image, int x1, int y1, int x2, int y2,
            int patchRadius, GBuffer gbuffer)
        {
            float distance = 0;
            int count = 0;

            for (int dx = -patchRadius; dx <= patchRadius; dx++)
            {
                for (int dy = -patchRadius; dy <= patchRadius; dy++)
                {
                    int nx1 = Math.Clamp(x1 + dx, 0, _width - 1);
                    int ny1 = Math.Clamp(y1 + dy, 0, _height - 1);
                    int nx2 = Math.Clamp(x2 + dx, 0, _width - 1);
                    int ny2 = Math.Clamp(y2 + dy, 0, _height - 1);

                    int idx1 = gbuffer.GetIndex(nx1, ny1);
                    int idx2 = gbuffer.GetIndex(nx2, ny2);

                    Vector3 diff = image[idx1] - image[idx2];
                    distance += diff.LengthSquared();
                    count++;
                }
            }

            return count > 0 ? distance / count : 0;
        }

        /// <summary>
        /// Applies wavelet transform denoiser.
        /// </summary>
        public void WaveletDenoiser(Vector3[] input, Vector3[] output, int levels, float threshold)
        {
            Vector3[] tempA = new Vector3[input.Length];
            Vector3[] tempB = new Vector3[input.Length];
            Array.Copy(input, tempA, input.Length);

            for (int level = 0; level < levels; level++)
            {
                int stride = 1 << level;
                WaveletDecompose(tempA, tempB, stride);
                SoftThreshold(tempB, threshold / (level + 1));
                WaveletReconstruct(tempA, tempB, stride);
            }

            Array.Copy(tempA, output, tempA.Length);
        }

        private void WaveletDecompose(Vector3[] input, Vector3[] detail, int stride)
        {
            for (int i = 0; i < input.Length; i++)
            {
                int left = Math.Max(0, i - stride);
                int right = Math.Min(input.Length - 1, i + stride);
                Vector3 average = (input[left] + input[right]) * 0.5f;
                detail[i] = input[i] - average;
                input[i] = average;
            }
        }

        private void WaveletReconstruct(Vector3[] approximation, Vector3[] detail, int stride)
        {
            for (int i = 0; i < approximation.Length; i++)
            {
                approximation[i] += detail[i];
            }
        }

        private void SoftThreshold(Vector3[] data, float threshold)
        {
            for (int i = 0; i < data.Length; i++)
            {
                float lum = 0.2126f * data[i].X + 0.7152f * data[i].Y + 0.0722f * data[i].Z;
                float sign = lum >= 0 ? 1.0f : -1.0f;
                float newLum = MathF.Max(0, MathF.Abs(lum) - threshold) * sign;
                float scale = MathF.Abs(lum) > 0.0001f ? newLum / lum : 0;
                data[i] *= scale;
            }
        }

        /// <summary>
        /// Applies normal-guided filter.
        /// </summary>
        public void NormalGuidedFilter(Vector3[] input, Vector3[] output, GBuffer gbuffer,
            int radius, float epsilon)
        {
            Parallel.For(radius, _height - radius, y =>
            {
                for (int x = radius; x < _width - radius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    Vector3 centerNormal = gbuffer.Normals[idx];

                    Vector3 meanP = Vector3.Zero;
                    Vector3 meanI = Vector3.Zero;
                    int count = 0;

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);

                            float normalSim = MathF.Max(0, Vector3.Dot(centerNormal, gbuffer.Normals[nIdx]));
                            if (normalSim < 0.7f)
                                continue;

                            meanP += gbuffer.Normals[nIdx];
                            meanI += input[nIdx];
                            count++;
                        }
                    }

                    if (count > 0)
                    {
                        meanP /= count;
                        meanI /= count;

                        Vector3 varP = Vector3.Zero;
                        Vector3 covPI = Vector3.Zero;

                        for (int dx = -radius; dx <= radius; dx++)
                        {
                            for (int dy = -radius; dy <= radius; dy++)
                            {
                                int nx = x + dx;
                                int ny = y + dy;
                                int nIdx = gbuffer.GetIndex(nx, ny);

                                float normalSim = MathF.Max(0, Vector3.Dot(centerNormal, gbuffer.Normals[nIdx]));
                                if (normalSim < 0.7f)
                                    continue;

                                Vector3 diffP = gbuffer.Normals[nIdx] - meanP;
                                Vector3 diffI = input[nIdx] - meanI;
                                varP += new Vector3(diffP.X * diffP.X, diffP.Y * diffP.Y, diffP.Z * diffP.Z);
                                covPI += new Vector3(diffP.X * diffI.X, diffP.Y * diffI.Y, diffP.Z * diffI.Z);
                            }
                        }

                        varP /= count;
                        covPI /= count;

                        Vector3 a = new Vector3(
                            covPI.X / (varP.X + epsilon),
                            covPI.Y / (varP.Y + epsilon),
                            covPI.Z / (varP.Z + epsilon));
                        Vector3 b = meanI - a * meanP;

                        output[idx] = a * gbuffer.Normals[idx] + b;
                    }
                    else
                    {
                        output[idx] = input[idx];
                    }
                }
            });
        }

        /// <summary>
        /// Estimates local variance.
        /// </summary>
        public void EstimateVariance(Vector3[] input, float[] output, GBuffer gbuffer, int radius)
        {
            Parallel.For(radius, _height - radius, y =>
            {
                for (int x = radius; x < _width - radius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    Vector3 mean = Vector3.Zero;
                    int count = 0;

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);
                            mean += input[nIdx];
                            count++;
                        }
                    }

                    mean /= count;

                    float variance = 0;
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);
                            Vector3 diff = input[nIdx] - mean;
                            variance += diff.LengthSquared();
                        }
                    }

                    output[idx] = variance / count;
                }
            });
        }

        /// <summary>
        /// Computes edge-stopping function.
        /// </summary>
        public float EdgeStoppingFunction(GBuffer gbuffer, int x1, int y1, int x2, int y2,
            float normalThreshold, float depthThreshold, float luminanceThreshold,
            Vector3[] luminance)
        {
            int idx1 = gbuffer.GetIndex(x1, y1);
            int idx2 = gbuffer.GetIndex(x2, y2);

            float depthDiff = MathF.Abs(gbuffer.Depth[idx1] - gbuffer.Depth[idx2]) /
                             MathF.Max(0.001f, gbuffer.Depth[idx1]);
            float normalDiff = 1.0f - Vector3.Dot(gbuffer.Normals[idx1], gbuffer.Normals[idx2]);

            float lum1 = luminance[idx1].Length();
            float lum2 = luminance[idx2].Length();
            float luminanceDiff = MathF.Abs(lum1 - lum2) / MathF.Max(0.001f, MathF.Max(lum1, lum2));

            float depthWeight = MathF.Exp(-depthDiff / depthThreshold);
            float normalWeight = MathF.Exp(-normalDiff / normalThreshold);
            float luminanceWeight = MathF.Exp(-luminanceDiff / luminanceThreshold);

            return depthWeight * normalWeight * luminanceWeight;
        }

        /// <summary>
        /// Progressive denoising with iterative refinement.
        /// </summary>
        public void ProgressiveDenoise(Vector3[] input, Vector3[] output, GBuffer gbuffer,
            int iterations, int baseRadius)
        {
            Vector3[] current = new Vector3[input.Length];
            Vector3[] next = new Vector3[input.Length];
            Array.Copy(input, current, input.Length);

            for (int iter = 0; iter < iterations; iter++)
            {
                int radius = baseRadius * (iter + 1);
                float strength = _config.Strength * (1.0f - (float)iter / iterations);

                SpatialBilateralFilter(current, next, gbuffer, radius,
                    _config.NormalThreshold * strength, _config.DepthThreshold * strength);

                Vector3[] temp = current;
                current = next;
                next = temp;
            }

            Array.Copy(current, output, current.Length);
        }

        /// <summary>
        /// Applies the full mixed denoiser pipeline.
        /// </summary>
        public void ApplyMixedPipeline(Vector3[] input, Vector3[] output, GBuffer gbuffer)
        {
            int pixelCount = _width * _height;
            Vector3[] stageA = new Vector3[pixelCount];
            Vector3[] stageB = new Vector3[pixelCount];
            Array.Copy(input, stageA, pixelCount);

            switch (_config.PrimaryDenoiser)
            {
                case DenoiserType.SpatialBilateral:
                    SpatialBilateralFilter(stageA, stageB, gbuffer, _config.SpatialRadius,
                        _config.NormalThreshold, _config.DepthThreshold);
                    break;
                case DenoiserType.NonLocalMeans:
                    NonLocalMeansDenoiser(stageA, stageB, gbuffer, 2, _config.SpatialRadius,
                        0.1f);
                    break;
                case DenoiserType.WaveletFilter:
                    WaveletDenoiser(stageA, stageB, 4, 0.05f);
                    break;
                default:
                    Array.Copy(stageA, stageB, pixelCount);
                    break;
            }

            if (_config.SecondaryDenoiser != DenoiserType.None)
            {
                switch (_config.SecondaryDenoiser)
                {
                    case DenoiserType.SpatialBilateral:
                        SpatialBilateralFilter(stageB, stageA, gbuffer, _config.SpatialRadius / 2,
                            _config.NormalThreshold, _config.DepthThreshold);
                        break;
                    case DenoiserType.TemporalAccumulation:
                        Array.Copy(stageB, stageA, pixelCount);
                        break;
                    default:
                        Array.Copy(stageB, stageA, pixelCount);
                        break;
                }
                Array.Copy(stageA, output, pixelCount);
            }
            else
            {
                Array.Copy(stageB, output, pixelCount);
            }
        }
    }
}
