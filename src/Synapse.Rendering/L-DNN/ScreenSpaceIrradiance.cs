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
    /// Screen-space global illumination system using ray marching and Hi-Z traversal.
    /// </summary>
    public class ScreenSpaceIrradiance
    {
        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;
        private const int MAX_MARCH_STEPS = 64;
        private const float STEP_SIZE = 1.0f;
        private const float Thickness = 0.5f;

        private Vector3[] _ssgiResult;
        private float[] _ssgiConfidence;
        private int _width;
        private int _height;
        private bool _isInitialized;

        /// <summary>SSGI result buffer.</summary>
        public Vector3[] Result => _ssgiResult;
        /// <summary>Confidence buffer.</summary>
        public float[] Confidence => _ssgiConfidence;

        /// <summary>
        /// Initializes the screen-space irradiance system.
        /// </summary>
        public void Initialize(int width, int height)
        {
            _width = width;
            _height = height;
            int pixelCount = width * height;
            _ssgiResult = new Vector3[pixelCount];
            _ssgiConfidence = new float[pixelCount];
            _isInitialized = true;
        }

        /// <summary>
        /// Computes screen-space GI using ray marching.
        /// </summary>
        public void ComputeSSGI(GBuffer gbuffer, CameraState camera, List<LightConfig> lights,
            int numRays, RandomNumberGenerator rng)
        {
            if (!_isInitialized) return;

            Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    float depth = gbuffer.Depth[idx];
                    if (depth <= 0)
                    {
                        _ssgiResult[idx] = Vector3.Zero;
                        _ssgiConfidence[idx] = 0;
                        continue;
                    }

                    Vector3 normal = gbuffer.Normals[idx];
                    Vector3 albedo = gbuffer.Albedo[idx];

                    Vector3 worldPos = ReconstructWorldPosition(x, y, depth, gbuffer, camera);
                    Vector3 tangent = GetTangent(normal, ref rng);
                    Vector3 bitangent = Vector3.Cross(normal, tangent);

                    Vector3 totalIrradiance = Vector3.Zero;
                    int hitCount = 0;

                    for (int r = 0; r < numRays; r++)
                    {
                        float phi = rng.NextFloat() * TWO_PI;
                        float cosTheta = MathF.Sqrt(rng.NextFloat());
                        float sinTheta = MathF.Sqrt(1.0f - cosTheta * cosTheta);

                        Vector3 sampleDir = tangent * (MathF.Cos(phi) * sinTheta) +
                                            bitangent * (MathF.Sin(phi) * sinTheta) +
                                            normal * cosTheta;

                        Vector3 hitRadiance = MarchScreenSpaceRay(worldPos, sampleDir,
                            gbuffer, camera, lights, rng);

                        if (hitRadiance.LengthSquared() > 0.0001f)
                        {
                            totalIrradiance += hitRadiance * cosTheta;
                            hitCount++;
                        }
                    }

                    if (hitCount > 0)
                    {
                        totalIrradiance /= hitCount;
                        totalIrradiance *= INV_PI;
                    }

                    _ssgiResult[idx] = totalIrradiance;
                    _ssgiConfidence[idx] = (float)hitCount / numRays;
                }
            });
        }

        /// <summary>
        /// Performs Hi-Z ray marching for efficient screen-space traversal.
        /// </summary>
        public Vector3 HiZRayMarch(Vector3 worldPos, Vector3 rayDir, GBuffer gbuffer,
            CameraState camera, int maxSteps)
        {
            Vector3 screenPos = camera.ProjectToScreen(worldPos);
            Vector3 screenDir = Vector3.Normalize(camera.ProjectToScreen(worldPos + rayDir) - screenPos);

            float currentDepth = screenPos.Z;
            float stepSize = STEP_SIZE;

            for (int step = 0; step < maxSteps; step++)
            {
                Vector3 sampleScreen = screenPos + screenDir * stepSize * (step + 1);
                if (sampleScreen.X < 0 || sampleScreen.X >= 1 || sampleScreen.Y < 0 || sampleScreen.Y >= 1)
                    return Vector3.Zero;

                int pixX = (int)(sampleScreen.X * _width);
                int pixY = (int)(sampleScreen.Y * _height);
                pixX = Math.Clamp(pixX, 0, _width - 1);
                pixY = Math.Clamp(pixY, 0, _height - 1);

                float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                float sampleDepth = sampleScreen.Z;

                if (sampleDepth > sceneDepth && sampleDepth < sceneDepth + STEP_SIZE)
                {
                    return gbuffer.Albedo[gbuffer.GetIndex(pixX, pixY)];
                }

                if (sampleDepth > sceneDepth + STEP_SIZE)
                {
                    stepSize *= 0.5f;
                }
            }

            return Vector3.Zero;
        }

        /// <summary>
        /// Marches a ray in screen-space and returns radiance at the hit point.
        /// </summary>
        public Vector3 MarchScreenSpaceRay(Vector3 worldOrigin, Vector3 worldDir,
            GBuffer gbuffer, CameraState camera, List<LightConfig> lights, RandomNumberGenerator rng)
        {
            Vector3 origin = camera.ProjectToScreen(worldOrigin);
            Vector3 target = camera.ProjectToScreen(worldOrigin + worldDir * 10.0f);
            Vector3 rayDir = target - origin;

            float rayLength = rayDir.Length();
            if (rayLength < 0.001f) return Vector3.Zero;
            rayDir /= rayLength;

            int numSteps = Math.Min(MAX_MARCH_STEPS, (int)(rayLength * 10));
            float stepLen = rayLength / numSteps;

            for (int step = 1; step <= numSteps; step++)
            {
                Vector3 samplePos = origin + rayDir * stepLen * step;
                if (samplePos.X < 0 || samplePos.X >= 1 || samplePos.Y < 0 || samplePos.Y >= 1)
                    return Vector3.Zero;

                int pixX = (int)(samplePos.X * _width);
                int pixY = (int)(samplePos.Y * _height);
                pixX = Math.Clamp(pixX, 0, _width - 1);
                pixY = Math.Clamp(pixY, 0, _height - 1);

                float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                float sampleDepth = samplePos.Z;

                if (sampleDepth > sceneDepth && sampleDepth < sceneDepth + STEP_SIZE)
                {
                    Vector3 hitNormal = gbuffer.Normals[gbuffer.GetIndex(pixX, pixY)];
                    Vector3 hitAlbedo = gbuffer.Albedo[gbuffer.GetIndex(pixX, pixY)];

                    float hitWeight = MathF.Max(0, Vector3.Dot(-worldDir, hitNormal));

                    Vector3 hitRadiance = Vector3.Zero;
                    foreach (var light in lights)
                    {
                        Vector3 lightDir;
                        float lightDist;
                        Vector3 lightColor;

                        if (light.Type == LightType.Directional)
                        {
                            lightDir = -light.Direction;
                            lightDist = float.MaxValue;
                            lightColor = light.Color * light.Intensity;
                        }
                        else
                        {
                            Vector3 toLight = light.Position - (worldOrigin + worldDir * stepLen * step);
                            lightDist = toLight.Length();
                            lightDir = toLight / lightDist;
                            float atten = 1.0f / (lightDist * lightDist);
                            lightColor = light.Color * light.Intensity * atten;
                        }

                        float NdotL = MathF.Max(0, Vector3.Dot(hitNormal, lightDir));
                        hitRadiance += hitAlbedo * lightColor * NdotL;
                    }

                    return hitRadiance * hitWeight;
                }
            }

            return Vector3.Zero;
        }

        /// <summary>
        /// Validates a screen-space hit using depth and normal comparison.
        /// </summary>
        public bool ValidateHit(Vector3 hitScreenPos, Vector3 hitNormal, float hitDepth,
            GBuffer gbuffer, CameraState camera)
        {
            if (hitScreenPos.X < 0 || hitScreenPos.X >= 1 || hitScreenPos.Y < 0 || hitScreenPos.Y >= 1)
                return false;

            int pixX = (int)(hitScreenPos.X * _width);
            int pixY = (int)(hitScreenPos.Y * _height);
            pixX = Math.Clamp(pixX, 0, _width - 1);
            pixY = Math.Clamp(pixY, 0, _height - 1);

            float sceneDepth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
            Vector3 sceneNormal = gbuffer.Normals[gbuffer.GetIndex(pixX, pixY)];

            float depthError = MathF.Abs(hitDepth - sceneDepth) / MathF.Max(0.001f, sceneDepth);
            float normalError = 1.0f - Vector3.Dot(hitNormal, sceneNormal);

            return depthError < 0.1f && normalError < 0.3f;
        }

        /// <summary>
        /// Performs bi-directional ray marching for better coverage.
        /// </summary>
        public Vector3 BidirectionalRayMarch(Vector3 worldPos, Vector3 rayDir, GBuffer gbuffer,
            CameraState camera, List<LightConfig> lights, RandomNumberGenerator rng)
        {
            Vector3 forwardResult = MarchScreenSpaceRay(worldPos, rayDir, gbuffer, camera, lights, rng);
            Vector3 backwardResult = MarchScreenSpaceRay(worldPos, -rayDir, gbuffer, camera, lights, rng);
            return (forwardResult + backwardResult) * 0.5f;
        }

        /// <summary>
        /// Detects back-face hits for two-sided surfaces.
        /// </summary>
        public bool IsBackFaceHit(Vector3 rayDir, Vector3 hitNormal)
        {
            return Vector3.Dot(rayDir, hitNormal) > 0;
        }

        /// <summary>
        /// Temporally reprojects screen-space hits from previous frame.
        /// </summary>
        public Vector3 TemporalReproject(int x, int y, Vector2 velocity,
            Vector3[] previousFrame, int prevWidth, int prevHeight)
        {
            float prevX = x + velocity.X;
            float prevY = y + velocity.Y;

            if (prevX < 0 || prevX >= prevWidth || prevY < 0 || prevY >= prevHeight)
                return Vector3.Zero;

            int prevIdx = (int)prevY * prevWidth + (int)prevX;
            if (prevIdx >= 0 && prevIdx < previousFrame.Length)
                return previousFrame[prevIdx];

            return Vector3.Zero;
        }

        /// <summary>
        /// Applies edge-stopping spatial filter to SSGI result.
        /// </summary>
        public void EdgeStoppingFilter(GBuffer gbuffer, int radius, float normalThreshold,
            float depthThreshold)
        {
            Vector3[] filtered = new Vector3[_ssgiResult.Length];
            Array.Copy(_ssgiResult, filtered, _ssgiResult.Length);

            Parallel.For(radius, _height - radius, y =>
            {
                for (int x = radius; x < _width - radius; x++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    Vector3 centerNormal = gbuffer.Normals[idx];
                    float centerDepth = gbuffer.Depth[idx];
                    Vector3 sum = Vector3.Zero;
                    float weightSum = 0;

                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        for (int dy = -radius; dy <= radius; dy++)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nIdx = gbuffer.GetIndex(nx, ny);

                            float depthDiff = MathF.Abs(centerDepth - gbuffer.Depth[nIdx]) / MathF.Max(0.001f, centerDepth);
                            float normalDiff = 1.0f - Vector3.Dot(centerNormal, gbuffer.Normals[nIdx]);

                            if (depthDiff > depthThreshold || normalDiff > normalThreshold)
                                continue;

                            float spatialWeight = MathF.Exp(-(dx * dx + dy * dy) / (2.0f * radius * radius));
                            float edgeWeight = MathF.Exp(-depthDiff * 10.0f) * MathF.Exp(-normalDiff * 5.0f);
                            float weight = spatialWeight * edgeWeight;

                            sum += _ssgiResult[nIdx] * weight;
                            weightSum += weight;
                        }
                    }

                    filtered[idx] = weightSum > 0 ? sum / weightSum : _ssgiResult[idx];
                }
            });

            Array.Copy(filtered, _ssgiResult, filtered.Length);
        }

        /// <summary>
        /// Falls back to world-space probes on screen-space miss.
        /// </summary>
        public Vector3 FallbackToWorldSpace(int x, int y, GBuffer gbuffer,
            IrradianceCacheManager probeCache, CameraState camera)
        {
            int idx = gbuffer.GetIndex(x, y);
            Vector3 worldPos = ReconstructWorldPosition(x, y, gbuffer.Depth[idx], gbuffer, camera);
            Vector3 normal = gbuffer.Normals[idx];

            float bestWeight = 0;
            Vector3 bestIrradiance = Vector3.Zero;

            var probes = probeCache.GetNearbyProbes(worldPos, normal, 5);
            foreach (var probe in probes)
            {
                if (!probe.IsValid) continue;
                float dist = (probe.Position - worldPos).Length();
                float weight = MathF.Exp(-dist * 0.1f) * probe.Importance;
                if (weight > bestWeight)
                {
                    bestWeight = weight;
                    bestIrradiance = ComputeProbeIrradiance(probe, normal);
                }
            }

            return bestIrradiance;
        }

        /// <summary>
        /// Computes irradiance from a probe given a direction.
        /// </summary>
        public Vector3 ComputeProbeIrradiance(IrradianceProbe probe, Vector3 direction)
        {
            if (probe.SHCoefficients == null || probe.SHCoefficients.Length < 9)
                return Vector3.Zero;

            float x = direction.X;
            float y = direction.Y;
            float z = direction.Z;

            Vector3 result = Vector3.Zero;
            result += new Vector3(probe.SHCoefficients[0].X, probe.SHCoefficients[0].Y, probe.SHCoefficients[0].Z) * 0.282095f;
            result += new Vector3(probe.SHCoefficients[1].X, probe.SHCoefficients[1].Y, probe.SHCoefficients[1].Z) * 0.488603f * y;
            result += new Vector3(probe.SHCoefficients[2].X, probe.SHCoefficients[2].Y, probe.SHCoefficients[2].Z) * 0.488603f * z;
            result += new Vector3(probe.SHCoefficients[3].X, probe.SHCoefficients[3].Y, probe.SHCoefficients[3].Z) * 0.488603f * x;
            result += new Vector3(probe.SHCoefficients[4].X, probe.SHCoefficients[4].Y, probe.SHCoefficients[4].Z) * 1.092548f * x * y;
            result += new Vector3(probe.SHCoefficients[5].X, probe.SHCoefficients[5].Y, probe.SHCoefficients[5].Z) * 1.092548f * y * z;
            result += new Vector3(probe.SHCoefficients[6].X, probe.SHCoefficients[6].Y, probe.SHCoefficients[6].Z) * 0.315392f * (3.0f * z * z - 1.0f);
            result += new Vector3(probe.SHCoefficients[7].X, probe.SHCoefficients[7].Y, probe.SHCoefficients[7].Z) * 1.092548f * x * z;
            result += new Vector3(probe.SHCoefficients[8].X, probe.SHCoefficients[8].Y, probe.SHCoefficients[8].Z) * 0.546274f * (x * x - y * y);

            return Vector3.Max(result, Vector3.Zero);
        }

        private Vector3 ReconstructWorldPosition(int x, int y, float depth, GBuffer gbuffer, CameraState camera)
        {
            float ndcX = (float)x / _width * 2.0f - 1.0f;
            float ndcY = 1.0f - (float)y / _height * 2.0f;
            Vector3 viewPos = new Vector3(ndcX, ndcY, depth);
            return camera.UnprojectFromScreen(viewPos);
        }

        private Vector3 GetTangent(Vector3 normal, ref RandomNumberGenerator rng)
        {
            Vector3 t;
            if (MathF.Abs(normal.X) > MathF.Abs(normal.Y))
                t = new Vector3(normal.Z, 0, -normal.X) / MathF.Sqrt(normal.X * normal.X + normal.Z * normal.Z);
            else
                t = new Vector3(0, -normal.Z, normal.Y) / MathF.Sqrt(normal.Y * normal.Y + normal.Z * normal.Z);
            return t;
        }
    }
}
