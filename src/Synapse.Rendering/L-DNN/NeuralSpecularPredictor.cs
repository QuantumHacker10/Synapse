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
    /// Lightweight MLP predicting specular reflection / refraction radiance from
    /// G-Buffer features (view, normal, roughness, metallic, F0).
    /// </summary>
    public sealed class NeuralSpecularPredictor
    {
        private const int InputSize = 16;
        private const int HiddenSize = 32;
        private const int OutputSize = 3;

        private float[,] _w1 = null!;
        private float[,] _w2 = null!;
        private float[] _b1 = null!;
        private float[] _b2 = null!;
        private bool _initialized;

        public bool IsInitialized => _initialized;

        public void Initialize()
        {
            var rng = new RandomNumberGenerator(1337);
            _w1 = Xavier(InputSize, HiddenSize, ref rng);
            _w2 = Xavier(HiddenSize, OutputSize, ref rng);
            _b1 = new float[HiddenSize];
            _b2 = new float[OutputSize];
            _initialized = true;
        }

        public Vector3 Predict(GBufferSample sample, CameraState camera)
        {
            if (!_initialized) return Vector3.Zero;

            Vector3 viewDir = Vector3.Normalize(camera.Position - sample.WorldPosition);
            float NdotV = MathF.Max(0, Vector3.Dot(sample.Normal, viewDir));
            // Schlick F0 approximation from metallic/albedo.
            Vector3 f0 = Vector3.Lerp(new Vector3(0.04f), sample.Albedo, sample.Metallic);

            float[] input =
            [
                sample.Normal.X, sample.Normal.Y, sample.Normal.Z,
                viewDir.X, viewDir.Y, viewDir.Z,
                NdotV,
                sample.Roughness,
                sample.Metallic,
                f0.X, f0.Y, f0.Z,
                sample.Specular.X, sample.Specular.Y, sample.Specular.Z,
                sample.IsTranslucent ? 1f : 0f
            ];

            float[] h = Linear(input, _w1, _b1);
            for (int i = 0; i < h.Length; i++)
                h[i] = h[i] > 0 ? h[i] : h[i] * 0.01f;
            float[] o = Linear(h, _w2, _b2);
            for (int i = 0; i < o.Length; i++)
                o[i] = 1f / (1f + MathF.Exp(-o[i]));

            // Scale by Fresnel × (1-roughness) so glossy metals get stronger reflections.
            float gloss = (1f - sample.Roughness) * (0.2f + 0.8f * (1f - NdotV));
            return new Vector3(o[0], o[1], o[2]) * gloss * f0;
        }

        /// <summary>
        /// Approximates refraction tint for translucent pixels (IOR-agnostic).
        /// </summary>
        public Vector3 PredictRefraction(GBufferSample sample, CameraState camera)
        {
            if (!sample.IsTranslucent) return Vector3.Zero;
            Vector3 reflection = Predict(sample, camera);
            // Simple transmission remainder: albedo-tinted, inverse of reflection strength.
            float reflectStrength = Math.Clamp(reflection.Length(), 0, 1);
            return sample.Albedo * (1f - reflectStrength) * (1f - sample.Roughness * 0.5f);
        }

        private static float[,] Xavier(int fanIn, int fanOut, ref RandomNumberGenerator rng)
        {
            float limit = MathF.Sqrt(6f / (fanIn + fanOut));
            var w = new float[fanIn, fanOut];
            for (int i = 0; i < fanIn; i++)
                for (int j = 0; j < fanOut; j++)
                    w[i, j] = rng.NextFloat(-limit, limit);
            return w;
        }

        private static float[] Linear(float[] input, float[,] weights, float[] bias)
        {
            int outs = weights.GetLength(1);
            var output = new float[outs];
            for (int j = 0; j < outs; j++)
            {
                float sum = bias[j];
                for (int i = 0; i < input.Length; i++)
                    sum += input[i] * weights[i, j];
                output[j] = sum;
            }
            return output;
        }
    }
}
