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
    /// Ambient occlusion system supporting SSAO, GTAO, and contact shadows.
    /// </summary>
    public class AmbientOcclusionSystem
    {
        private float[] _aoBuffer;
        private float[] _temporalHistory;
        private Vector3[] _hemisphereKernel;
        private Vector3[] _noiseTexture;
        private int _width;
        private int _height;
        private int _kernelSize;
        private bool _isInitialized;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>AO result buffer.</summary>
        public float[] AOBuffer => _aoBuffer;

        /// <summary>
        /// Initializes the ambient occlusion system.
        /// </summary>
        public void Initialize(int width, int height, int kernelSize = 64)
        {
            _width = width;
            _height = height;
            _kernelSize = kernelSize;
            int pixelCount = width * height;

            _aoBuffer = new float[pixelCount];
            _temporalHistory = new float[pixelCount];
            _hemisphereKernel = new Vector3[kernelSize];
            _noiseTexture = new Vector3[16 * 16];

            GenerateHemisphereKernel();
            GenerateNoiseTexture();
            _isInitialized = true;
        }

        private void GenerateHemisphereKernel()
        {
            var rng = new RandomNumberGenerator(42);
            for (int i = 0; i < _kernelSize; i++)
            {
                float x = rng.NextFloat(-1.0f, 1.0f);
                float y = rng.NextFloat(-1.0f, 1.0f);
                float z = rng.NextFloat(0.0f, 1.0f);
                float len = MathF.Sqrt(x * x + y * y + z * z);
                x /= len;
                y /= len;
                z /= len;

                float scale = (float)i / _kernelSize;
                scale = 0.1f + scale * scale * 0.9f;
                _hemisphereKernel[i] = new Vector3(x, y, z) * scale;
            }
        }

        private void GenerateNoiseTexture()
        {
            var rng = new RandomNumberGenerator(123);
            for (int i = 0; i < _noiseTexture.Length; i++)
            {
                _noiseTexture[i] = new Vector3(
                    rng.NextFloat(-1.0f, 1.0f),
                    rng.NextFloat(-1.0f, 1.0f),
                    0.0f);
            }
        }

        /// <summary>
        /// Computes SSAO using hemisphere sampling.
        /// </summary>
        public void ComputeSSAO(GBuffer gbuffer, CameraState camera, int kernelSize, float radius)
        {
            if (!_isInitialized)
                return;

            Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    float depth = gbuffer.Depth[idx];
                    if (depth <= 0)
                    {
                        _aoBuffer[idx] = 1.0f;
                        continue;
                    }

                    Vector3 normal = gbuffer.Normals[idx];
                    Vector3 worldPos = ReconstructWorldPosition(x, y, depth, gbuffer, camera);

                    Vector3 noise = _noiseTexture[(x % 16) * 16 + (y % 16)];

                    Vector3 tangent = Vector3.Normalize(noise - normal * Vector3.Dot(normal, noise));
                    Vector3 bitangent = Vector3.Cross(normal, tangent);
                    Matrix4x4 tbn = new Matrix4x4(
                        tangent.X, bitangent.X, normal.X, 0,
                        tangent.Y, bitangent.Y, normal.Y, 0,
                        tangent.Z, bitangent.Z, normal.Z, 0,
                        0, 0, 0, 1);

                    float occlusion = 0;
                    int validSamples = 0;

                    for (int i = 0; i < kernelSize && i < _hemisphereKernel.Length; i++)
                    {
                        Vector3 sampleDir = Vector3.Transform(_hemisphereKernel[i], tbn);
                        Vector3 samplePos = worldPos + sampleDir * radius;

                        Vector3 sampleScreen = camera.ProjectToScreen(samplePos);
                        if (sampleScreen.X < 0 || sampleScreen.X >= 1 ||
                            sampleScreen.Y < 0 || sampleScreen.Y >= 1)
                            continue;

                        int samplePixX = (int)(sampleScreen.X * _width);
                        int samplePixY = (int)(sampleScreen.Y * _height);
                        samplePixX = Math.Clamp(samplePixX, 0, _width - 1);
                        samplePixY = Math.Clamp(samplePixY, 0, _height - 1);

                        float sampleDepth = gbuffer.Depth[gbuffer.GetIndex(samplePixX, samplePixY)];
                        float rangeCheck = MathF.Max(0, 1.0f - MathF.Abs(depth - sampleDepth) / radius);

                        if (sampleDepth < sampleScreen.Z)
                            occlusion += rangeCheck;

                        validSamples++;
                    }

                    _aoBuffer[idx] = validSamples > 0 ? 1.0f - (occlusion / validSamples) : 1.0f;
                }
            });
        }

        /// <summary>
        /// Computes GTAO (Ground Truth Ambient Occlusion) approximation.
        /// </summary>
        public void ComputeGTAO(GBuffer gbuffer, CameraState camera, int numDirections, float radius)
        {
            if (!_isInitialized)
                return;

            Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    float depth = gbuffer.Depth[idx];
                    if (depth <= 0)
                    {
                        _aoBuffer[idx] = 1.0f;
                        continue;
                    }

                    Vector3 normal = gbuffer.Normals[idx];
                    Vector3 worldPos = ReconstructWorldPosition(x, y, depth, gbuffer, camera);

                    float occlusion = 0;

                    for (int i = 0; i < numDirections; i++)
                    {
                        float angle = TWO_PI * i / numDirections;
                        Vector2 dir = new Vector2(MathF.Cos(angle), MathF.Sin(angle));

                        float horizonLow = 0;
                        float horizonHigh = 0;

                        for (int j = 1; j <= 8; j++)
                        {
                            float stepSize = radius * j / 8.0f;
                            Vector3 sampleOffset = new Vector3(dir.X, dir.Y, 0) * stepSize;
                            Vector3 samplePos = worldPos + sampleOffset;

                            Vector3 sampleScreen = camera.ProjectToScreen(samplePos);
                            if (sampleScreen.X < 0 || sampleScreen.X >= 1 ||
                                sampleScreen.Y < 0 || sampleScreen.Y >= 1)
                                continue;

                            int samplePixX = (int)(sampleScreen.X * _width);
                            int samplePixY = (int)(sampleScreen.Y * _height);
                            samplePixX = Math.Clamp(samplePixX, 0, _width - 1);
                            samplePixY = Math.Clamp(samplePixY, 0, _height - 1);

                            float sampleDepth = gbuffer.Depth[gbuffer.GetIndex(samplePixX, samplePixY)];
                            Vector3 sampleWorldPos = ReconstructWorldPosition(samplePixX, samplePixY,
                                sampleDepth, gbuffer, camera);

                            Vector3 horizonVec = sampleWorldPos - worldPos;
                            float horizonAngle = MathF.Atan2(horizonVec.Z, new Vector2(horizonVec.X, horizonVec.Y).Length());

                            if (horizonAngle > horizonLow)
                                horizonLow = horizonAngle;
                            if (horizonAngle < horizonHigh)
                                horizonHigh = horizonAngle;
                        }

                        occlusion += MathF.Max(0, MathF.Cos(horizonLow)) + MathF.Max(0, MathF.Cos(horizonHigh));
                    }

                    _aoBuffer[idx] = 1.0f - occlusion / (numDirections * 2.0f);
                }
            });
        }

        /// <summary>
        /// Temporally accumulates AO.
        /// </summary>
        public void TemporalAccumulate(float blendFactor)
        {
            for (int i = 0; i < _aoBuffer.Length; i++)
            {
                _aoBuffer[i] = _aoBuffer[i] * (1.0f - blendFactor) + _temporalHistory[i] * blendFactor;
                _temporalHistory[i] = _aoBuffer[i];
            }
        }

        /// <summary>
        /// Applies edge-preserving blur to AO.
        /// </summary>
        public void BlurAO(GBuffer gbuffer, int radius, float edgeThreshold)
        {
            float[] blurred = new float[_aoBuffer.Length];
            Array.Copy(_aoBuffer, blurred, _aoBuffer.Length);

            Parallel.For(radius, _height - radius, y =>
            {
                for (int x = radius; x < _width - radius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    float centerDepth = gbuffer.Depth[idx];
                    Vector3 centerNormal = gbuffer.Normals[idx];

                    float sum = 0;
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

                            if (depthDiff > edgeThreshold || normalDiff > 0.5f)
                                continue;

                            float spatialWeight = MathF.Exp(-(dx * dx + dy * dy) / (2.0f * radius * radius));
                            float depthWeight = MathF.Exp(-depthDiff * 10.0f);

                            sum += _aoBuffer[nIdx] * spatialWeight * depthWeight;
                            weightSum += spatialWeight * depthWeight;
                        }
                    }

                    blurred[idx] = weightSum > 0 ? sum / weightSum : _aoBuffer[idx];
                }
            });

            Array.Copy(blurred, _aoBuffer, blurred.Length);
        }

        /// <summary>
        /// Upscales AO from half-resolution to full-resolution.
        /// </summary>
        public void UpscaleAO(float[] halfResAO, int halfWidth, int halfHeight)
        {
            for (int x = 0; x < _width; x++)
            {
                for (int y = 0; y < _height; y++)
                {
                    float u = (float)x / _width * halfWidth;
                    float v = (float)y / _height * halfHeight;

                    int x0 = Math.Clamp((int)u, 0, halfWidth - 1);
                    int y0 = Math.Clamp((int)v, 0, halfHeight - 1);
                    int x1 = Math.Min(x0 + 1, halfWidth - 1);
                    int y1 = Math.Min(y0 + 1, halfHeight - 1);

                    float fx = u - x0;
                    float fy = v - y0;

                    float a00 = halfResAO[y0 * halfWidth + x0];
                    float a10 = halfResAO[y0 * halfWidth + x1];
                    float a01 = halfResAO[y1 * halfWidth + x0];
                    float a11 = halfResAO[y1 * halfWidth + x1];

                    float a0 = a00 * (1 - fx) + a10 * fx;
                    float a1 = a01 * (1 - fx) + a11 * fx;
                    _aoBuffer[y * _width + x] = a0 * (1 - fy) + a1 * fy;
                }
            }
        }

        /// <summary>
        /// Computes contact shadows.
        /// </summary>
        public float ComputeContactShadow(GBuffer gbuffer, CameraState camera, Vector3 worldPos,
            Vector3 lightDir, int numSteps, float maxDistance)
        {
            Vector3 startScreen = camera.ProjectToScreen(worldPos);
            Vector3 endPos = worldPos + lightDir * maxDistance;
            Vector3 endScreen = camera.ProjectToScreen(endPos);

            float shadow = 1.0f;

            for (int i = 1; i <= numSteps; i++)
            {
                float t = (float)i / numSteps;
                Vector3 sampleScreen = Vector3.Lerp(startScreen, endScreen, t);

                if (sampleScreen.X < 0 || sampleScreen.X >= 1 ||
                    sampleScreen.Y < 0 || sampleScreen.Y >= 1)
                    break;

                int pixX = (int)(sampleScreen.X * _width);
                int pixY = (int)(sampleScreen.Y * _height);
                pixX = Math.Clamp(pixX, 0, _width - 1);
                pixY = Math.Clamp(pixY, 0, _height - 1);

                float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                float sampleDepth = sampleScreen.Z;

                if (sampleDepth > sceneDepth && sampleDepth < sceneDepth + 0.1f)
                {
                    shadow *= 0.5f;
                }
            }

            return shadow;
        }

        private Vector3 ReconstructWorldPosition(int x, int y, float depth, GBuffer gbuffer, CameraState camera)
        {
            float ndcX = (float)x / _width * 2.0f - 1.0f;
            float ndcY = 1.0f - (float)y / _height * 2.0f;
            Vector3 viewPos = new Vector3(ndcX, ndcY, depth);
            return camera.UnprojectFromScreen(viewPos);
        }
    }
}
