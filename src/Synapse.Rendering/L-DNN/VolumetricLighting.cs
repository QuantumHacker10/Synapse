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
    /// Volumetric lighting system using froxel-based volume rendering.
    /// </summary>
    public class VolumetricLighting
    {
        private required VolumeFogConfig _config;
        private required Vector3[,,] _froxelRadiance;
        private required float[,,] _froxelTransmittance;
        private required Vector3[,,] _temporalHistory;
        private int _gridX;
        private int _gridY;
        private int _gridZ;
        private required float[] _depthSlices;
        private bool _isInitialized;

        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;

        /// <summary>Froxel grid X resolution.</summary>
        public int GridX => _gridX;
        /// <summary>Froxel grid Y resolution.</summary>
        public int GridY => _gridY;
        /// <summary>Froxel grid Z resolution (depth slices).</summary>
        public int GridZ => _gridZ;

        /// <summary>
        /// Initializes the volumetric lighting system.
        /// </summary>
        public void Initialize(VolumeFogConfig config, int screenWitdth, int screenHeight)
        {
            _config = config;
            _gridX = config.GridResolutionXY;
            _gridY = config.GridResolutionXY;
            _gridZ = config.DepthSlices;

            _froxelRadiance = new Vector3[_gridX, _gridY, _gridZ];
            _froxelTransmittance = new float[_gridX, _gridY, _gridZ];
            _temporalHistory = new Vector3[_gridX, _gridY, _gridZ];
            _depthSlices = new float[_gridZ + 1];

            ComputeExponentialDepthSlices(config.StartDistance, config.MaxDistance);
            _isInitialized = true;
        }

        private void ComputeExponentialDepthSlices(float near, float far)
        {
            for (int i = 0; i <= _gridZ; i++)
            {
                float t = (float)i / _gridZ;
                _depthSlices[i] = near * MathF.Pow(far / near, t);
            }
        }

        /// <summary>
        /// Constructs the froxel grid from camera parameters.
        /// </summary>
        public void ConstructFroxelGrid(CameraState camera)
        {
            for (int z = 0; z < _gridZ; z++)
            {
                float depthNear = _depthSlices[z];
                float depthFar = _depthSlices[z + 1];

                for (int x = 0; x < _gridX; x++)
                {
                    for (int y = 0; y < _gridY; y++)
                    {
                        float u = (x + 0.5f) / _gridX;
                        float v = (y + 0.5f) / _gridY;

                        Vector3 viewPos = new Vector3(
                            (u * 2.0f - 1.0f) * camera.FieldOfView * camera.AspectRatio,
                            (1.0f - v * 2.0f) * camera.FieldOfView,
                            -1.0f);
                        viewPos = Vector3.Normalize(viewPos);

                        float volumeDepth = (depthNear + depthFar) * 0.5f;
                        float volumeSize = depthFar - depthNear;
                        float cellVolume = volumeSize * volumeDepth * volumeDepth / (_gridX * _gridY);

                        _froxelRadiance[x, y, z] = Vector3.Zero;
                        _froxelTransmittance[x, y, z] = 1.0f;
                    }
                }
            }
        }

        /// <summary>
        /// Injects lighting into froxels from light sources.
        /// </summary>
        public void InjectLighting(List<LightConfig> lights, GBuffer gbuffer, CameraState camera)
        {
            for (int z = 0; z < _gridZ; z++)
            {
                float depthNear = _depthSlices[z];
                float depthFar = _depthSlices[z + 1];
                float sliceDepth = (depthNear + depthFar) * 0.5f;

                for (int x = 0; x < _gridX; x++)
                {
                    for (int y = 0; y < _gridY; y++)
                    {
                        Vector3 worldPos = ComputeFroxelWorldPos(x, y, z, camera);
                        Vector3 totalLighting = Vector3.Zero;

                        foreach (var light in lights)
                        {
                            Vector3 lightContribution = ComputeLightContribution(light, worldPos, gbuffer, camera);
                            totalLighting += lightContribution;
                        }

                        float density = ComputeFogDensity(worldPos);
                        _froxelRadiance[x, y, z] = totalLighting * density * _config.MaxDensity;
                    }
                }
            }
        }

        private Vector3 ComputeLightContribution(LightConfig light, Vector3 worldPos,
            GBuffer gbuffer, CameraState camera)
        {
            switch (light.Type)
            {
                case LightType.Directional:
                    return light.Color * light.Intensity;

                case LightType.Point:
                    Vector3 toLight = light.Position - worldPos;
                    float dist = toLight.Length();
                    if (dist > light.Range)
                        return Vector3.Zero;
                    float atten = 1.0f / (dist * dist);
                    float rangeAtten = MathF.Max(0, 1.0f - MathF.Pow(dist / light.Range, 4));
                    return light.Color * light.Intensity * atten * rangeAtten;

                case LightType.Spot:
                    Vector3 spotToLight = light.Position - worldPos;
                    float spotDist = spotToLight.Length();
                    if (spotDist > light.Range)
                        return Vector3.Zero;
                    Vector3 spotDir = spotToLight / spotDist;
                    float spotAtten = 1.0f / (spotDist * spotDist);
                    float spotRangeAtten = MathF.Max(0, 1.0f - MathF.Pow(spotDist / light.Range, 4));
                    float cosAngle = Vector3.Dot(-spotDir, light.Direction);
                    float spotCos = MathF.Cos(light.OuterConeAngle);
                    float spotInnerCos = MathF.Cos(light.InnerConeAngle);
                    float spotFalloff = Math.Clamp((cosAngle - spotCos) / MathF.Max(0.001f, spotInnerCos - spotCos), 0, 1);
                    return light.Color * light.Intensity * spotAtten * spotRangeAtten * spotFalloff;

                default:
                    return Vector3.Zero;
            }
        }

        /// <summary>
        /// Evaluates the Henyey-Greenstein phase function.
        /// </summary>
        public float HenyeyGreenstein(float cosTheta, float g)
        {
            float g2 = g * g;
            float denom = 1.0f + g2 - 2.0f * g * cosTheta;
            return (1.0f - g2) / (4.0f * PI * denom * MathF.Sqrt(denom));
        }

        /// <summary>
        /// Evaluates a dual-lobe phase function.
        /// </summary>
        public float DualLobePhaseFunction(float cosTheta, float g1, float g2, float blend)
        {
            float phase1 = HenyeyGreenstein(cosTheta, g1);
            float phase2 = HenyeyGreenstein(cosTheta, g2);
            return blend * phase1 + (1.0f - blend) * phase2;
        }

        /// <summary>
        /// Integrates anisotropic scattering along a view ray.
        /// </summary>
        public Vector3 IntegrateAnisotropicScattering(Vector3 viewDir, Vector3 lightDir,
            float density, Vector3 scatteringCoeff)
        {
            float cosTheta = Vector3.Dot(viewDir, lightDir);
            float phase = HenyeyGreenstein(cosTheta, _config.Anisotropy);
            return scatteringCoeff * phase * density;
        }

        /// <summary>
        /// Performs temporal reprojection for froxels.
        /// </summary>
        public void TemporalReprojection(CameraState camera)
        {
            for (int x = 0; x < _gridX; x++)
            {
                for (int y = 0; y < _gridY; y++)
                {
                    for (int z = 0; z < _gridZ; z++)
                    {
                        Vector3 current = _froxelRadiance[x, y, z];
                        Vector3 history = _temporalHistory[x, y, z];

                        float blendFactor = _config.TemporalReprojectionStrength;
                        _froxelRadiance[x, y, z] = Vector3.Lerp(current, history, blendFactor);
                        _temporalHistory[x, y, z] = _froxelRadiance[x, y, z];
                    }
                }
            }
        }

        /// <summary>
        /// Computes fog density at a world position (height fog + optional cloud slab).
        /// </summary>
        public float ComputeFogDensity(Vector3 worldPos)
        {
            float heightDensity = MathF.Exp(-(worldPos.Y - _config.ReferenceHeight) * _config.HeightFalloff);
            float baseDensity = _config.MaxDensity * MathF.Max(0f, heightDensity);

            if (_config.NoiseScale > 0)
            {
                float noise = PerlinNoise3D(worldPos * _config.NoiseScale);
                baseDensity *= 0.5f + noise * 0.5f;
            }

            if (_config.EnableClouds)
                baseDensity += ComputeCloudDensity(worldPos);

            return baseDensity;
        }

        /// <summary>
        /// Procedural cloud density in a horizontal slab around CloudAltitude.
        /// </summary>
        public float ComputeCloudDensity(Vector3 worldPos)
        {
            float halfThickness = MathF.Max(1e-3f, _config.CloudThickness);
            float vertical = 1.0f - MathF.Abs(worldPos.Y - _config.CloudAltitude) / halfThickness;
            if (vertical <= 0f)
                return 0f;

            // Soft vertical falloff inside the slab.
            vertical = vertical * vertical * (3f - 2f * vertical);

            float scale = _config.CloudNoiseScale > 0 ? _config.CloudNoiseScale : 0.02f;
            float n1 = PerlinNoise3D(worldPos * scale);
            float n2 = PerlinNoise3D(worldPos * (scale * 2.3f) + new Vector3(17.1f, 0, 9.3f));
            float shape = n1 * 0.65f + n2 * 0.35f;

            // Remap coverage: higher coverage lowers the threshold for "cloud".
            float threshold = 1.0f - Math.Clamp(_config.CloudCoverage, 0f, 1f);
            float cloud = MathF.Max(0f, shape - threshold) / MathF.Max(1e-3f, 1f - threshold);
            cloud = cloud * cloud;

            return _config.MaxDensity * _config.CloudDensityScale * vertical * cloud;
        }

        /// <summary>
        /// Computes fog color at a world position.
        /// </summary>
        public Vector3 ComputeFogColor(Vector3 worldPos, Vector3 lightColor, float lightIntensity)
        {
            float density = ComputeFogDensity(worldPos);
            return _config.FogColor * density * lightColor * lightIntensity;
        }

        /// <summary>
        /// Computes light shaft / god ray contribution.
        /// </summary>
        public float ComputeLightShafts(CameraState camera, Vector3 lightDirection,
            GBuffer gbuffer, int numSamples)
        {
            float occlusion = 0;
            Vector3 sunPos = camera.Position - lightDirection * 100.0f;

            for (int i = 0; i < numSamples; i++)
            {
                float t = (float)i / numSamples;
                Vector3 samplePos = Vector3.Lerp(camera.Position, sunPos, t);
                Vector3 screenPos = camera.ProjectToScreen(samplePos);

                if (screenPos.X >= 0 && screenPos.X < 1 && screenPos.Y >= 0 && screenPos.Y < 1)
                {
                    int pixX = (int)(screenPos.X * gbuffer.Width);
                    int pixY = (int)(screenPos.Y * gbuffer.Height);
                    pixX = Math.Clamp(pixX, 0, gbuffer.Width - 1);
                    pixY = Math.Clamp(pixY, 0, gbuffer.Height - 1);

                    float depth = gbuffer.Depth[gbuffer.GetIndex(pixX, pixY)];
                    float sampleDepth = screenPos.Z;

                    if (sampleDepth > depth)
                        occlusion += 1.0f;
                }
            }

            return 1.0f - occlusion / numSamples;
        }

        /// <summary>
        /// Computes volumetric shadow by ray marching through froxels.
        /// </summary>
        public Vector3 ComputeVolumetricShadow(Vector3 worldPos, Vector3 lightDirection,
            CameraState camera)
        {
            Vector3 shadow = Vector3.One;
            Vector3 startScreen = camera.ProjectToScreen(worldPos);
            Vector3 endScreen = camera.ProjectToScreen(worldPos - lightDirection * _config.MaxDistance);

            int numSteps = 32;
            for (int i = 0; i < numSteps; i++)
            {
                float t = (float)i / numSteps;
                Vector3 sampleScreen = Vector3.Lerp(startScreen, endScreen, t);

                if (sampleScreen.X >= 0 && sampleScreen.X < 1 &&
                    sampleScreen.Y >= 0 && sampleScreen.Y < 1)
                {
                    int fx = (int)(sampleScreen.X * _gridX);
                    int fy = (int)(sampleScreen.Y * _gridY);
                    int fz = (int)(sampleScreen.Z * _gridZ);

                    fx = Math.Clamp(fx, 0, _gridX - 1);
                    fy = Math.Clamp(fy, 0, _gridY - 1);
                    fz = Math.Clamp(fz, 0, _gridZ - 1);

                    float transmittance = _froxelTransmittance[fx, fy, fz];
                    shadow *= transmittance;
                }
            }

            return shadow;
        }

        /// <summary>
        /// Integrates volume scattering along a view ray.
        /// </summary>
        public Vector3 IntegrateVolumeScattering(CameraState camera, Vector2 screenPos,
            Vector3 viewDirection, List<LightConfig> lights)
        {
            Vector3 accumulatedScattering = Vector3.Zero;
            Vector3 accumulatedTransmittance = Vector3.One;

            for (int z = 0; z < _gridZ; z++)
            {
                float depthNear = _depthSlices[z];
                float depthFar = _depthSlices[z + 1];
                float sliceThickness = depthFar - depthNear;

                int fx = (int)(screenPos.X * _gridX);
                int fy = (int)(screenPos.Y * _gridY);
                fx = Math.Clamp(fx, 0, _gridX - 1);
                fy = Math.Clamp(fy, 0, _gridY - 1);

                Vector3 sliceRadiance = _froxelRadiance[fx, fy, z];
                float sliceDensity = ComputeFogDensity(
                    camera.Position + viewDirection * (depthNear + depthFar) * 0.5f);

                Vector3 sliceAlbedo = _config.FogColor;
                float sliceTransmittance = MathF.Exp(-sliceDensity * sliceThickness);

                Vector3 sliceInScattering = sliceRadiance * (1.0f - sliceTransmittance) /
                    MathF.Max(0.001f, sliceDensity);
                accumulatedScattering += accumulatedTransmittance * sliceInScattering;
                accumulatedTransmittance *= sliceTransmittance;
            }

            return accumulatedScattering;
        }

        private Vector3 ComputeFroxelWorldPos(int x, int y, int z, CameraState camera)
        {
            float u = (x + 0.5f) / _gridX;
            float v = (y + 0.5f) / _gridY;
            float depth = (_depthSlices[z] + _depthSlices[z + 1]) * 0.5f;

            Vector3 screenPos = new Vector3(u, v, depth);
            return camera.UnprojectFromScreen(screenPos);
        }

        private float PerlinNoise3D(Vector3 pos)
        {
            int xi = (int)MathF.Floor(pos.X) & 255;
            int yi = (int)MathF.Floor(pos.Y) & 255;
            int zi = (int)MathF.Floor(pos.Z) & 255;

            float xf = pos.X - MathF.Floor(pos.X);
            float yf = pos.Y - MathF.Floor(pos.Y);
            float zf = pos.Z - MathF.Floor(pos.Z);

            float u = Fade(xf);
            float v = Fade(yf);
            float w = Fade(zf);

            float n000 = Grad(xi, yi, zi, xf, yf, zf);
            float n001 = Grad(xi, yi, zi + 1, xf, yf, zf - 1);
            float n010 = Grad(xi, yi + 1, zi, xf, yf - 1, zf);
            float n011 = Grad(xi, yi + 1, zi + 1, xf, yf - 1, zf - 1);
            float n100 = Grad(xi + 1, yi, zi, xf - 1, yf, zf);
            float n101 = Grad(xi + 1, yi, zi + 1, xf - 1, yf, zf - 1);
            float n110 = Grad(xi + 1, yi + 1, zi, xf - 1, yf - 1, zf);
            float n111 = Grad(xi + 1, yi + 1, zi + 1, xf - 1, yf - 1, zf - 1);

            float nx00 = Lerp(n000, n100, u);
            float nx01 = Lerp(n001, n101, u);
            float nx10 = Lerp(n010, n110, u);
            float nx11 = Lerp(n011, n111, u);

            float nxy0 = Lerp(nx00, nx10, v);
            float nxy1 = Lerp(nx01, nx11, v);

            return Lerp(nxy0, nxy1, w);
        }

        private float Fade(float t) => t * t * t * (t * (t * 6 - 15) + 10);
        private float Lerp(float a, float b, float t) => a + t * (b - a);

        private float Grad(int x, int y, int z, float dx, float dy, float dz)
        {
            int h = (x * 374761393 + y * 668265263 + z * 1274126177) & 15;
            float u = h < 8 ? dx : dy;
            float v = h < 4 ? dy : (h == 12 || h == 14 ? dx : dz);
            return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
        }
    }
}
