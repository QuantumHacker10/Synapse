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
    /// Reference path tracer for generating ground truth images and training data.
    /// Implements unidirectional path tracing with MIS, Russian roulette, and
    /// next event estimation.
    /// </summary>
    public class ReferencePathTracer
    {
        private const float EPSILON = 1e-6f;
        private const float RussianRouletteThreshold = 0.05f;
        private const int MAX_BOUNCES = 32;
        private const float PI = MathF.PI;
        private const float TWO_PI = 2.0f * MathF.PI;
        private const float INV_PI = 1.0f / MathF.PI;
        private const float INV_TWO_PI = 1.0f / (2.0f * MathF.PI);

        private FilmPixel[,] _film;
        private int _width;
        private int _height;
        private int _totalSamples;

        /// <summary>Reference image buffer.</summary>
        public FilmPixel[,] Film => _film;

        /// <summary>
        /// Initializes the path tracer for a given resolution.
        /// </summary>
        public void Initialize(int width, int height)
        {
            _width = width;
            _height = height;
            _film = new FilmPixel[width, height];
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                    _film[x, y] = new FilmPixel
                    {
                        Position = new Vector2Int(x, y),
                        Color = Vector3.Zero,
                        Radiance = Vector3.Zero,
                        SampleCount = 0,
                        Variance = 0,
                        IsConverged = false,
                        TileIndex = 0
                    };
        }

        /// <summary>
        /// Generates a reference image using path tracing.
        /// </summary>
        public void GenerateReferenceImage(GBuffer gbuffer, CameraState camera,
            List<LightConfig> lights, int samplesPerPixel, int maxDepth)
        {
            _totalSamples = samplesPerPixel;

            Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    var rng = new RandomNumberGenerator((uint)(x * _height + y + _totalSamples * 1000));
                    Vector3 pixelColor = Vector3.Zero;

                    for (int s = 0; s < samplesPerPixel; s++)
                    {
                        float jitterX = rng.NextFloat() - 0.5f;
                        float jitterY = rng.NextFloat() - 0.5f;
                        float u = (x + 0.5f + jitterX) / _width;
                        float v = (y + 0.5f + jitterY) / _height;

                        RayPayload ray = GenerateCameraRay(u, v, camera, ref rng);
                        Vector3 radiance = EstimateRadiance(ray, gbuffer, lights, maxDepth, ref rng);
                        pixelColor += radiance;
                    }

                    pixelColor /= samplesPerPixel;

                    _film[x, y] = _film[x, y] with
                    {
                        Color = pixelColor,
                        Radiance = pixelColor,
                        SampleCount = samplesPerPixel,
                        IsConverged = true
                    };
                }
            });
        }

        /// <summary>
        /// Generates a camera ray from screen-space coordinates.
        /// </summary>
        public RayPayload GenerateCameraRay(float u, float v, CameraState camera, ref RandomNumberGenerator rng)
        {
            float tanHalfFov = MathF.Tan(camera.FieldOfView * 0.5f);
            float aspect = camera.AspectRatio;
            float ndcX = (2.0f * u - 1.0f) * aspect * tanHalfFov;
            float ndcY = (1.0f - 2.0f * v) * tanHalfFov;
            Vector3 rayDir = Vector3.Normalize(camera.Right * ndcX + camera.Up * ndcY + camera.Forward);

            return new RayPayload
            {
                Origin = camera.Position,
                Direction = rayDir,
                MaxDistance = camera.FarPlane,
                Throughput = Vector3.One,
                Radiance = Vector3.Zero,
                BounceDepth = 0,
                RandomSeed = rng.NextUint(),
                Flags = RayFlags.None
            };
        }

        /// <summary>
        /// Estimates the radiance along a ray using path tracing.
        /// </summary>
        public Vector3 EstimateRadiance(RayPayload initialRay, GBuffer gbuffer,
            List<LightConfig> lights, int maxDepth, ref RandomNumberGenerator rng)
        {
            Vector3 accumulatedRadiance = Vector3.Zero;
            Vector3 throughput = initialRay.Throughput;
            Vector3 origin = initialRay.Origin;
            Vector3 direction = initialRay.Direction;
            int depth = 0;

            while (depth < maxDepth)
            {
                HitResult hit = TraceRay(origin, direction, gbuffer);
                if (!hit.DidHit)
                {
                    accumulatedRadiance += throughput * SampleEnvironmentMap(direction);
                    break;
                }

                MaterialProperties mat = hit.Material;
                accumulatedRadiance += throughput * mat.Emissive * mat.EmissiveIntensity;

                if (depth > 2)
                {
                    float continueProbability = MathF.Max(throughput.X,
                        MathF.Max(throughput.Y, throughput.Z));
                    if (rng.NextFloat() > continueProbability)
                        break;
                    throughput /= continueProbability;
                }

                Vector3 wo = -direction;
                Vector3 lightContribution = EstimateDirectLighting(hit, wo, lights, gbuffer, ref rng);
                accumulatedRadiance += throughput * lightContribution;

                BxDFResult bxdf = SampleBSDF(hit, wo, ref rng);
                if (!bxdf.IsValid || bxdf.PDF < EPSILON)
                    break;

                throughput *= bxdf.Value * MathF.Abs(Vector3.Dot(bxdf.SampledDirection, hit.Normal)) / bxdf.PDF;

                origin = hit.HitPosition + bxdf.SampledDirection * EPSILON;
                direction = bxdf.SampledDirection;
                depth++;
            }

            return accumulatedRadiance;
        }

        /// <summary>
        /// Performs ray tracing against the scene geometry.
        /// </summary>
        public HitResult TraceRay(Vector3 origin, Vector3 direction, GBuffer gbuffer)
        {
            float minDist = float.MaxValue;
            HitResult closest = new HitResult { DidHit = false };

            for (int x = 0; x < gbuffer.Width; x++)
            {
                for (int y = 0; y < gbuffer.Height; y++)
                {
                    int idx = gbuffer.GetIndex(x, y);
                    float depth = gbuffer.Depth[idx];
                    if (depth <= 0 || depth >= minDist) continue;

                    Vector3 hitPos = origin + direction * depth;
                    float cosAngle = Vector3.Dot(direction, gbuffer.Normals[idx]);
                    if (cosAngle >= 0) continue;

                    minDist = depth;
                    Vector3 normal = gbuffer.Normals[idx];
                    Vector3 albedo = gbuffer.Albedo[idx];
                    float roughness = gbuffer.MaterialProps[idx].X;
                    float metallic = gbuffer.MaterialProps[idx].Y;
                    Vector3 specular = gbuffer.Specular[idx];

                    closest = new HitResult
                    {
                        DidHit = true,
                        HitDistance = depth,
                        HitPosition = hitPos,
                        Normal = normal,
                        GeometricNormal = normal,
                        Albedo = albedo,
                        Material = new MaterialProperties
                        {
                            BaseColor = albedo,
                            SpecularColor = specular,
                            Roughness = roughness,
                            Metallic = metallic,
                            IOR = 1.5f,
                            Subsurface = 0,
                            SpecularTransmission = 0,
                            IsThinSurface = false,
                            Emissive = gbuffer.Emissive[idx],
                            EmissiveIntensity = 1.0f
                        },
                        UV = new Vector2((float)x / gbuffer.Width, (float)y / gbuffer.Height),
                        Barycentrics = new Vector3(1, 0, 0),
                        TriangleIndex = idx,
                        PrimitiveIndex = 0,
                        InstanceID = 0
                    };
                }
            }

            return closest;
        }

        /// <summary>
        /// Evaluates the BxDF for a given surface and direction pair.
        /// </summary>
        public Vector3 EvaluateBSDF(HitResult hit, Vector3 wi, Vector3 wo)
        {
            MaterialProperties mat = hit.Material;
            Vector3 n = hit.Normal;
            float NdotL = MathF.Max(0, Vector3.Dot(n, wi));
            float NdotV = MathF.Max(0, Vector3.Dot(n, wo));
            if (NdotL < EPSILON || NdotV < EPSILON) return Vector3.Zero;

            Vector3 diffuse = mat.BaseColor * INV_PI;
            Vector3 halfVec = Vector3.Normalize(wi + wo);
            float HdotV = MathF.Max(0, Vector3.Dot(halfVec, wo));
            float D = DistributionGGX(n, halfVec, mat.Roughness);
            float G = GeometrySmith(n, wi, wo, mat.Roughness);
            Vector3 F = FresnelSchlick(HdotV, Vector3.Lerp(new Vector3(0.04f), mat.BaseColor, mat.Metallic));
            Vector3 kD = (Vector3.One - F) * (1.0f - mat.Metallic);
            Vector3 specular = (D * G * F) / MathF.Max(EPSILON, 4.0f * NdotL * NdotV);
            return kD * diffuse + specular;
        }

        /// <summary>
        /// Samples the BSDF to generate a new direction.
        /// </summary>
        public BxDFResult SampleBSDF(HitResult hit, Vector3 wo, ref RandomNumberGenerator rng)
        {
            MaterialProperties mat = hit.Material;
            Vector3 n = hit.Normal;
            float r1 = rng.NextFloat();
            float r2 = rng.NextFloat();

            Vector3 tangent = GetOrthogonal(n, ref rng);
            Vector3 bitangent = Vector3.Cross(n, tangent);

            float roughness = MathF.Max(0.04f, mat.Roughness);
            float alpha = roughness * roughness;

            if (rng.NextFloat() < mat.Metallic * 0.5f)
            {
                float phi = TWO_PI * r1;
                float cosTheta = MathF.Sqrt((1.0f - r2) / (1.0f + (alpha * alpha - 1.0f) * r2));
                float sinTheta = MathF.Sqrt(1.0f - cosTheta * cosTheta);
                Vector3 halfVec = tangent * (MathF.Cos(phi) * sinTheta) +
                                  bitangent * (MathF.Sin(phi) * sinTheta) +
                                  n * cosTheta;
                Vector3 wi = Vector3.Reflect(-wo, halfVec);
                if (Vector3.Dot(wi, n) < 0)
                    return new BxDFResult { IsValid = false };

                float D = DistributionGGX(n, halfVec, roughness);
                float G = GeometrySmith(n, wi, wo, roughness);
                float HdotV = MathF.Max(0, Vector3.Dot(halfVec, wo));
                Vector3 F = FresnelSchlick(HdotV, Vector3.Lerp(new Vector3(0.04f), mat.BaseColor, mat.Metallic));
                float pdf = D * MathF.Max(0, Vector3.Dot(halfVec, n)) / MathF.Max(EPSILON, 4.0f * HdotV);
                Vector3 value = F * G * MathF.Max(0, Vector3.Dot(wi, n)) / MathF.Max(EPSILON, HdotV);

                return new BxDFResult
                {
                    Value = value,
                    SampledDirection = wi,
                    PDF = pdf,
                    IsValid = true,
                    IsDelta = false,
                    Component = BxDFComponent.SpecularReflection
                };
            }
            else
            {
                float phi = TWO_PI * r1;
                float cosTheta = MathF.Sqrt(r2);
                float sinTheta = MathF.Sqrt(1.0f - r2);
                Vector3 wi = tangent * (MathF.Cos(phi) * sinTheta) +
                              bitangent * (MathF.Sin(phi) * sinTheta) +
                              n * cosTheta;
                float pdf = cosTheta * INV_PI;
                Vector3 value = mat.BaseColor * INV_PI;
                return new BxDFResult
                {
                    Value = value,
                    SampledDirection = wi,
                    PDF = pdf,
                    IsValid = true,
                    IsDelta = false,
                    Component = BxDFComponent.Diffuse
                };
            }
        }

        /// <summary>
        /// Estimates direct lighting at a surface point using next event estimation.
        /// </summary>
        public Vector3 EstimateDirectLighting(HitResult hit, Vector3 wo,
            List<LightConfig> lights, GBuffer gbuffer, ref RandomNumberGenerator rng)
        {
            Vector3 directLight = Vector3.Zero;

            foreach (var light in lights)
            {
                if (light.Intensity < EPSILON) continue;

                Vector3 lightDir;
                float lightDistance;
                Vector3 lightRadiance;

                switch (light.Type)
                {
                    case LightType.Directional:
                        lightDir = -light.Direction;
                        lightDistance = float.MaxValue;
                        lightRadiance = light.Color * light.Intensity;
                        break;

                    case LightType.Point:
                        Vector3 toLight = light.Position - hit.HitPosition;
                        lightDistance = toLight.Length();
                        lightDir = toLight / lightDistance;
                        float attenuation = 1.0f / (lightDistance * lightDistance);
                        float rangeAtten = MathF.Max(0, 1.0f - MathF.Pow(lightDistance / MathF.Max(0.001f, light.Range), 4));
                        lightRadiance = light.Color * light.Intensity * attenuation * rangeAtten;
                        break;

                    case LightType.Spot:
                        Vector3 spotToLight = light.Position - hit.HitPosition;
                        lightDistance = spotToLight.Length();
                        lightDir = spotToLight / lightDistance;
                        float spotAtten = 1.0f / (lightDistance * lightDistance);
                        float spotRangeAtten = MathF.Max(0, 1.0f - MathF.Pow(lightDistance / MathF.Max(0.001f, light.Range), 4));
                        float cosAngle = Vector3.Dot(-lightDir, light.Direction);
                        float spotCos = MathF.Cos(light.OuterConeAngle);
                        float spotInnerCos = MathF.Cos(light.InnerConeAngle);
                        float spotFalloff = Math.Clamp((cosAngle - spotCos) / MathF.Max(EPSILON, spotInnerCos - spotCos), 0, 1);
                        lightRadiance = light.Color * light.Intensity * spotAtten * spotRangeAtten * spotFalloff;
                        break;

                    case LightType.AreaRect:
                        Vector2 areaSample = SampleRectangle(ref rng);
                        Vector3 lightPoint = light.Position +
                            light.Right * areaSample.X * light.AreaWidth * 0.5f +
                            light.Up * areaSample.Y * light.AreaHeight * 0.5f;
                        Vector3 areaToLight = lightPoint - hit.HitPosition;
                        lightDistance = areaToLight.Length();
                        lightDir = areaToLight / lightDistance;
                        float cosAtLight = MathF.Max(0, -Vector3.Dot(lightDir, light.Direction));
                        float areaPdf = 1.0f / (light.AreaWidth * light.AreaHeight);
                        lightRadiance = light.Color * light.Intensity * cosAtLight /
                            MathF.Max(EPSILON, lightDistance * lightDistance * areaPdf);
                        break;

                    case LightType.AreaDisc:
                        Vector2 discSample = SampleDisc(ref rng);
                        Vector3 discPoint = light.Position +
                            light.Right * discSample.X * light.AreaWidth * 0.5f +
                            light.Up * discSample.Y * light.AreaHeight * 0.5f;
                        Vector3 discToLight = discPoint - hit.HitPosition;
                        lightDistance = discToLight.Length();
                        lightDir = discToLight / lightDistance;
                        float discCosAtLight = MathF.Max(0, -Vector3.Dot(lightDir, light.Direction));
                        float discPdf = 1.0f / (PI * light.AreaWidth * light.AreaHeight * 0.25f);
                        lightRadiance = light.Color * light.Intensity * discCosAtLight /
                            MathF.Max(EPSILON, lightDistance * lightDistance * discPdf);
                        break;

                    default:
                        continue;
                }

                float NdotL = MathF.Max(0, Vector3.Dot(hit.Normal, lightDir));
                if (NdotL < EPSILON) continue;

                if (light.ShadowMethod == ShadowMethod.RayTraced)
                {
                    HitResult shadowHit = TraceRay(
                        hit.HitPosition + hit.Normal * EPSILON,
                        lightDir, gbuffer);
                    if (shadowHit.DidHit && shadowHit.HitDistance < lightDistance)
                        continue;
                }

                Vector3 bxdf = EvaluateBSDF(hit, lightDir, wo);
                directLight += bxdf * lightRadiance * NdotL;
            }

            return directLight;
        }

        /// <summary>
        /// Computes the GGX/Trowbridge-Reitz normal distribution function.
        /// </summary>
        public float DistributionGGX(Vector3 N, Vector3 H, float roughness)
        {
            float a = roughness * roughness;
            float a2 = a * a;
            float NdotH = MathF.Max(0, Vector3.Dot(N, H));
            float NdotH2 = NdotH * NdotH;
            float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
            return a2 / (PI * denom * denom);
        }

        /// <summary>
        /// Computes the Smith geometry shadowing/masking function.
        /// </summary>
        public float GeometrySmith(Vector3 N, Vector3 V, Vector3 L, float roughness)
        {
            float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
            float NdotV = MathF.Max(0, Vector3.Dot(N, V));
            float NdotL = MathF.Max(0, Vector3.Dot(N, L));
            float ggx1 = NdotV / (NdotV * (1.0f - k) + k);
            float ggx2 = NdotL / (NdotL * (1.0f - k) + k);
            return ggx1 * ggx2;
        }

        /// <summary>
        /// Evaluates the Fresnel-Schlick approximation.
        /// </summary>
        public Vector3 FresnelSchlick(float cosTheta, Vector3 F0)
        {
            float oneMinusCos = 1.0f - cosTheta;
            float oneMinusCos5 = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
            return F0 + (Vector3.One - F0) * oneMinusCos5;
        }

        /// <summary>
        /// Computes Fresnel for dielectric surfaces.
        /// </summary>
        public float FresnelDielectric(float cosTheta, float eta)
        {
            float sin2Theta = 1.0f - cosTheta * cosTheta;
            float eta2 = eta * eta;
            float discriminant = 1.0f - sin2Theta / eta2;
            if (discriminant < 0) return 1.0f;
            float cosT = MathF.Sqrt(discriminant);
            float rs = (eta * cosTheta - cosT) / (eta * cosTheta + cosT);
            float rp = (cosTheta - eta * cosT) / (cosTheta + eta * cosT);
            return (rs * rs + rp * rp) * 0.5f;
        }

        /// <summary>
        /// Generates an orthonormal basis from a normal vector.
        /// </summary>
        public Vector3 GetOrthogonal(Vector3 n, ref RandomNumberGenerator rng)
        {
            Vector3 t;
            if (MathF.Abs(n.X) > MathF.Abs(n.Y))
                t = new Vector3(n.Z, 0, -n.X) / MathF.Sqrt(n.X * n.X + n.Z * n.Z);
            else
                t = new Vector3(0, -n.Z, n.Y) / MathF.Sqrt(n.Y * n.Y + n.Z * n.Z);
            return t;
        }

        private Vector2 SampleRectangle(ref RandomNumberGenerator rng)
        {
            return new Vector2(rng.NextFloat() * 2.0f - 1.0f, rng.NextFloat() * 2.0f - 1.0f);
        }

        private Vector2 SampleDisc(ref RandomNumberGenerator rng)
        {
            float r = MathF.Sqrt(rng.NextFloat());
            float theta = TWO_PI * rng.NextFloat();
            return new Vector2(r * MathF.Cos(theta), r * MathF.Sin(theta));
        }

        private Vector3 SampleEnvironmentMap(Vector3 direction)
        {
            float sky = MathF.Max(0, direction.Y);
            return Vector3.Lerp(new Vector3(0.1f, 0.1f, 0.15f), new Vector3(0.4f, 0.6f, 0.9f), sky) * 0.5f;
        }

        /// <summary>
        /// Computes PSNR between the reference and denoised images.
        /// </summary>
        public float ComputePSNR(Vector3[,] reference, Vector3[,] denoised, int width, int height)
        {
            double mse = 0;
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    Vector3 diff = reference[x, y] - denoised[x, y];
                    mse += diff.X * diff.X + diff.Y * diff.Y + diff.Z * diff.Z;
                }
            }
            mse /= (width * height * 3);
            if (mse < 1e-10) return 100.0f;
            return (float)(10.0 * Math.Log10(1.0 / mse));
        }

        /// <summary>
        /// Computes SSIM between reference and denoised images.
        /// </summary>
        public float ComputeSSIM(Vector3[,] reference, Vector3[,] denoised, int width, int height)
        {
            double sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;
            int n = width * height * 3;
            const float C1 = 0.0001f;
            const float C2 = 0.0009f;

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        float rx = reference[x, y][c];
                        float ry = denoised[x, y][c];
                        sumX += rx; sumY += ry;
                        sumXX += rx * rx; sumYY += ry * ry;
                        sumXY += rx * ry;
                    }
                }
            }

            double meanX = sumX / n;
            double meanY = sumY / n;
            double varX = sumXX / n - meanX * meanX;
            double varY = sumYY / n - meanY * meanY;
            double cov = sumXY / n - meanX * meanY;

            double numerator = (2.0 * meanX * meanY + C1) * (2.0 * cov + C2);
            double denominator = (meanX * meanX + meanY * meanY + C1) * (varX + varY + C2);
            return (float)(numerator / denominator);
        }

        /// <summary>
        /// Approximates LPIPS perceptual distance.
        /// </summary>
        public float ComputeLPIPSApproximation(Vector3[,] reference, Vector3[,] denoised, int width, int height)
        {
            double totalDist = 0;
            int sampleCount = 0;
            int step = Math.Max(1, width / 64);

            for (int x = 0; x < width; x += step)
            {
                for (int y = 0; y < height; y += step)
                {
                    Vector3 diff = reference[x, y] - denoised[x, y];
                    float perceptualWeight = 0.5f + 0.5f * MathF.Abs(reference[x, y].Y);
                    totalDist += diff.Length() * perceptualWeight;
                    sampleCount++;
                }
            }

            return sampleCount > 0 ? (float)(totalDist / sampleCount) : 0;
        }

        /// <summary>
        /// Generates training data pairs from path tracing.
        /// </summary>
        public (Vector3[,] NoisyImage, Vector3[,] GroundTruth) GenerateTrainingPair(
            GBuffer gbuffer, CameraState camera, List<LightConfig> lights,
            int samplesPerPixel, int noisySamples, int maxDepth)
        {
            var groundTruth = new Vector3[_width, _height];
            var noisy = new Vector3[_width, _height];

            Parallel.For(0, _height, y =>
            {
                var rngGT = new RandomNumberGenerator((uint)(y * 1337 + 42));
                var rngNoisy = new RandomNumberGenerator((uint)(y * 2847 + 13));

                for (int x = 0; x < _width; x++)
                {
                    Vector3 gtColor = Vector3.Zero;
                    Vector3 noisyColor = Vector3.Zero;

                    for (int s = 0; s < samplesPerPixel; s++)
                    {
                        float jitterX = rngGT.NextFloat() - 0.5f;
                        float jitterY = rngGT.NextFloat() - 0.5f;
                        float u = (x + 0.5f + jitterX) / _width;
                        float v = (y + 0.5f + jitterY) / _height;

                        var ray = GenerateCameraRay(u, v, camera, ref rngGT);
                        gtColor += EstimateRadiance(ray, gbuffer, lights, maxDepth, ref rngGT);
                    }
                    gtColor /= samplesPerPixel;

                    for (int s = 0; s < noisySamples; s++)
                    {
                        float jitterX = rngNoisy.NextFloat() - 0.5f;
                        float jitterY = rngNoisy.NextFloat() - 0.5f;
                        float u = (x + 0.5f + jitterX) / _width;
                        float v = (y + 0.5f + jitterY) / _height;

                        var ray = GenerateCameraRay(u, v, camera, ref rngNoisy);
                        noisyColor += EstimateRadiance(ray, gbuffer, lights, maxDepth, ref rngNoisy);
                    }
                    noisyColor /= noisySamples;

                    groundTruth[x, y] = gtColor;
                    noisy[x, y] = noisyColor;
                }
            });

            return (noisy, groundTruth);
        }
    }

    /// <summary>
    /// Simple pseudo-random number generator for path tracing.
    /// </summary>
    public class RandomNumberGenerator
    {
        private uint _state;

        /// <summary>Creates a new RNG with the given seed.</summary>
        public RandomNumberGenerator(uint seed = 0)
        {
            _state = seed == 0 ? 1u : seed;
        }

        /// <summary>Generates a random unsigned integer.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public uint NextUint()
        {
            _state ^= _state << 13;
            _state ^= _state >> 17;
            _state ^= _state << 5;
            return _state;
        }

        /// <summary>Generates a random float in [0, 1).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextFloat() => (NextUint() & 0x00FFFFFF) / (float)0x01000000;

        /// <summary>Generates a random float in [min, max).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextFloat(float min, float max) => min + NextFloat() * (max - min);

        /// <summary>Generates a random integer in [min, max).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int NextInt(int min, int max) => min + (int)(NextUint() % (uint)(max - min));
    }
}
