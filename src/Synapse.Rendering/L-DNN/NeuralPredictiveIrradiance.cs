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
    /// Neural network system for predicting irradiance from G-Buffer features.
    /// Implements forward pass, feature extraction, online training, and
    /// inference optimization.
    /// </summary>
    public class NeuralPredictiveIrradiance
    {
        private const int INPUT_FEATURES = 32;
        private const int OUTPUT_FEATURES = 3;
        private const float LEARNING_RATE = 0.001f;
        private const float BETA1 = 0.9f;
        private const float BETA2 = 0.999f;
        private const float EPSILON_ADAM = 1e-8f;

        private int _hidden1Size = 128;
        private int _hidden2Size = 128;

        private required float[,] _weights1;
        private required float[,] _weights2;
        private required float[,] _weights3;
        private required float[] _bias1;
        private required float[] _bias2;
        private required float[] _bias3;
        private required float[,] _m1, _v1, _m2, _v2, _m3, _v3;
        private required float[] _mb1, _mb2, _mb3;
        private required float[] _vb1, _vb2, _vb3;
        private int _t;
        private bool _isInitialized;

        private required Queue<(Vector3[] Features, Vector3 Target)> _trainingBuffer;
        private int _maxTrainingBufferSize;
        private float _trainingLoss;
        private float _inferenceTime;
        private int _trainingStep;

        /// <summary>Whether the network is initialized.</summary>
        public bool IsInitialized => _isInitialized;
        /// <summary>Current training loss.</summary>
        public float TrainingLoss => _trainingLoss;
        /// <summary>Last inference time in milliseconds.</summary>
        public float InferenceTime => _inferenceTime;
        /// <summary>Total training steps performed.</summary>
        public int TrainingStep => _trainingStep;
        /// <summary>Active network size profile.</summary>
        public NeuralNetworkProfile Profile { get; private set; } = NeuralNetworkProfile.Full;
        /// <summary>Hidden layer 1 size.</summary>
        public int HiddenLayer1Size => _hidden1Size;
        /// <summary>Hidden layer 2 size.</summary>
        public int HiddenLayer2Size => _hidden2Size;

        /// <summary>
        /// Initializes the neural network architecture for a given size profile.
        /// </summary>
        public void Initialize(NeuralNetworkProfile profile = NeuralNetworkProfile.Full)
        {
            Profile = profile;
            (_hidden1Size, _hidden2Size) = profile switch
            {
                NeuralNetworkProfile.Tiny => (32, 32),
                NeuralNetworkProfile.Small => (64, 64),
                _ => (128, 128)
            };

            var rng = new RandomNumberGenerator(42);

            _weights1 = XavierInitialize(INPUT_FEATURES, _hidden1Size, ref rng);
            _weights2 = XavierInitialize(_hidden1Size, _hidden2Size, ref rng);
            _weights3 = XavierInitialize(_hidden2Size, OUTPUT_FEATURES, ref rng);
            _bias1 = new float[_hidden1Size];
            _bias2 = new float[_hidden2Size];
            _bias3 = new float[OUTPUT_FEATURES];

            _m1 = new float[INPUT_FEATURES, _hidden1Size];
            _v1 = new float[INPUT_FEATURES, _hidden1Size];
            _m2 = new float[_hidden1Size, _hidden2Size];
            _v2 = new float[_hidden1Size, _hidden2Size];
            _m3 = new float[_hidden2Size, OUTPUT_FEATURES];
            _v3 = new float[_hidden2Size, OUTPUT_FEATURES];
            _mb1 = new float[_hidden1Size];
            _mb2 = new float[_hidden2Size];
            _mb3 = new float[OUTPUT_FEATURES];
            _vb1 = new float[_hidden1Size];
            _vb2 = new float[_hidden2Size];
            _vb3 = new float[OUTPUT_FEATURES];
            _t = 0;

            _trainingBuffer = new Queue<(Vector3[], Vector3)>();
            _maxTrainingBufferSize = 10000;
            _trainingLoss = 0;
            _trainingStep = 0;
            _isInitialized = true;
        }

        private float[,] XavierInitialize(int fanIn, int fanOut, ref RandomNumberGenerator rng)
        {
            float limit = MathF.Sqrt(6.0f / (fanIn + fanOut));
            var weights = new float[fanIn, fanOut];
            for (int i = 0; i < fanIn; i++)
                for (int j = 0; j < fanOut; j++)
                    weights[i, j] = rng.NextFloat(-limit, limit);
            return weights;
        }

        /// <summary>
        /// Extracts features from the G-Buffer for neural prediction.
        /// </summary>
        public Vector3[] ExtractFeatures(GBufferSample sample, GBuffer gbuffer, int px, int py,
            CameraState camera)
        {
            var features = new float[INPUT_FEATURES];
            int idx = 0;

            features[idx++] = sample.Depth / 100.0f;
            features[idx++] = sample.Normal.X;
            features[idx++] = sample.Normal.Y;
            features[idx++] = sample.Normal.Z;
            features[idx++] = sample.Albedo.X;
            features[idx++] = sample.Albedo.Y;
            features[idx++] = sample.Albedo.Z;
            features[idx++] = sample.Specular.X;
            features[idx++] = sample.Roughness;
            features[idx++] = sample.Metallic;

            Vector3 viewDir = Vector3.Normalize(camera.Position - sample.WorldPosition);
            float NdotV = MathF.Max(0, Vector3.Dot(sample.Normal, viewDir));
            features[idx++] = NdotV;

            Vector3 screenPos3 = new Vector3(
                (float)px / gbuffer.Width * 2.0f - 1.0f,
                1.0f - (float)py / gbuffer.Height * 2.0f,
                sample.Depth);
            features[idx++] = screenPos3.X;
            features[idx++] = screenPos3.Y;
            features[idx++] = sample.Velocity.X;
            features[idx++] = sample.Velocity.Y;

            float edgeDepth = ComputeEdgeDepth(gbuffer, px, py);
            float edgeNormal = ComputeEdgeNormal(gbuffer, px, py);
            features[idx++] = edgeDepth;
            features[idx++] = edgeNormal;

            Vector3[] neighborFeatures = ExtractNeighborhoodFeatures(gbuffer, px, py);
            for (int i = 0; i < Math.Min(8, neighborFeatures.Length) && idx < INPUT_FEATURES; i++)
            {
                features[idx++] = neighborFeatures[i].X;
                if (idx < INPUT_FEATURES)
                    features[idx++] = neighborFeatures[i].Y;
                if (idx < INPUT_FEATURES)
                    features[idx++] = neighborFeatures[i].Z;
            }

            while (idx < INPUT_FEATURES)
                features[idx++] = 0;

            var result = new Vector3[INPUT_FEATURES];
            for (int i = 0; i < INPUT_FEATURES; i++)
                result[i] = new Vector3(features[i], 0, 0);
            return result;
        }

        private float ComputeEdgeDepth(GBuffer gbuffer, int x, int y)
        {
            float depth = gbuffer.Depth[gbuffer.GetIndex(x, y)];
            float maxDiff = 0;
            int[] offsets = { -1, 0, 1 };
            foreach (int dx in offsets)
            {
                foreach (int dy in offsets)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    int nx = Math.Clamp(x + dx, 0, gbuffer.Width - 1);
                    int ny = Math.Clamp(y + dy, 0, gbuffer.Height - 1);
                    float neighborDepth = gbuffer.Depth[gbuffer.GetIndex(nx, ny)];
                    maxDiff = MathF.Max(maxDiff, MathF.Abs(depth - neighborDepth));
                }
            }
            return maxDiff;
        }

        private float ComputeEdgeNormal(GBuffer gbuffer, int x, int y)
        {
            Vector3 normal = gbuffer.Normals[gbuffer.GetIndex(x, y)];
            float maxDiff = 0;
            int[] offsets = { -1, 0, 1 };
            foreach (int dx in offsets)
            {
                foreach (int dy in offsets)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    int nx = Math.Clamp(x + dx, 0, gbuffer.Width - 1);
                    int ny = Math.Clamp(y + dy, 0, gbuffer.Height - 1);
                    Vector3 neighborNormal = gbuffer.Normals[gbuffer.GetIndex(nx, ny)];
                    maxDiff = MathF.Max(maxDiff, 1.0f - Vector3.Dot(normal, neighborNormal));
                }
            }
            return maxDiff;
        }

        private Vector3[] ExtractNeighborhoodFeatures(GBuffer gbuffer, int x, int y)
        {
            var features = new List<Vector3>();
            int radius = 2;
            for (int dx = -radius; dx <= radius; dx++)
            {
                for (int dy = -radius; dy <= radius; dy++)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    int nx = Math.Clamp(x + dx, 0, gbuffer.Width - 1);
                    int ny = Math.Clamp(y + dy, 0, gbuffer.Height - 1);
                    int idx = gbuffer.GetIndex(nx, ny);
                    features.Add(new Vector3(
                        gbuffer.Depth[idx],
                        gbuffer.Normals[idx].Length(),
                        gbuffer.Albedo[idx].Length()));
                }
            }
            return features.ToArray();
        }

        /// <summary>
        /// Aggregates spatial features from the neighborhood.
        /// </summary>
        public Vector3[] SpatialFeatureAggregation(Vector3[] pixelFeatures, int width, int height, int x, int y)
        {
            var aggregated = new Vector3[INPUT_FEATURES];
            float totalWeight = 0;

            for (int dx = -2; dx <= 2; dx++)
            {
                for (int dy = -2; dy <= 2; dy++)
                {
                    int nx = Math.Clamp(x + dx, 0, width - 1);
                    int ny = Math.Clamp(y + dy, 0, height - 1);
                    int idx = ny * width + nx;
                    float spatialWeight = MathF.Exp(-(dx * dx + dy * dy) * 0.5f);

                    for (int f = 0; f < INPUT_FEATURES && f < pixelFeatures.Length; f++)
                        aggregated[f] += pixelFeatures[idx] * spatialWeight;
                    totalWeight += spatialWeight;
                }
            }

            if (totalWeight > 0)
                for (int f = 0; f < INPUT_FEATURES; f++)
                    aggregated[f] /= totalWeight;

            return aggregated;
        }

        /// <summary>
        /// Aggregates temporal features from reprojection.
        /// </summary>
        public Vector3[] TemporalFeatureAggregation(Vector3[] currentFeatures,
            Vector3[] previousFeatures, Vector2 velocity, int width, int height, int x, int y)
        {
            var temporalFeatures = new Vector3[INPUT_FEATURES];
            float reprojectionConfidence = MathF.Exp(-velocity.Length() * 5.0f);

            for (int f = 0; f < INPUT_FEATURES; f++)
            {
                if (previousFeatures != null && f < previousFeatures.Length)
                    temporalFeatures[f] = Vector3.Lerp(currentFeatures[f], previousFeatures[f], reprojectionConfidence * 0.8f);
                else
                    temporalFeatures[f] = currentFeatures[f];
            }

            return temporalFeatures;
        }

        /// <summary>
        /// Estimates confidence (variance prediction) for the irradiance estimate.
        /// </summary>
        public float EstimateConfidence(Vector3[] features, Vector3 predictedIrradiance)
        {
            float featureMagnitude = 0;
            for (int i = 0; i < features.Length; i++)
                featureMagnitude += features[i].X * features[i].X;
            featureMagnitude = MathF.Sqrt(featureMagnitude);

            float irradianceMagnitude = predictedIrradiance.Length();
            float normalizedIrradiance = MathF.Min(1.0f, irradianceMagnitude / 5.0f);
            float confidence = normalizedIrradiance * MathF.Exp(-featureMagnitude * 0.1f);
            return Math.Clamp(confidence, 0, 1);
        }

        /// <summary>
        /// Performs a forward pass through the neural network.
        /// </summary>
        public Vector3 ForwardPass(Vector3[] inputFeatures)
        {
            if (!_isInitialized)
                return Vector3.Zero;

            float[] input = new float[INPUT_FEATURES];
            for (int i = 0; i < INPUT_FEATURES && i < inputFeatures.Length; i++)
                input[i] = inputFeatures[i].X;

            float[] hidden1 = LinearForward(input, _weights1, _bias1);
            LeakyReLU(hidden1);
            float[] hidden2 = LinearForward(hidden1, _weights2, _bias2);
            LeakyReLU(hidden2);
            float[] output = LinearForward(hidden2, _weights3, _bias3);
            Sigmoid(output);

            return new Vector3(output[0], output[1], output[2]);
        }

        private float[] LinearForward(float[] input, float[,] weights, float[] bias)
        {
            int inputSize = weights.GetLength(0);
            int outputSize = weights.GetLength(1);
            float[] output = new float[outputSize];

            for (int j = 0; j < outputSize; j++)
            {
                float sum = bias[j];
                for (int i = 0; i < inputSize; i++)
                    sum += input[i] * weights[i, j];
                output[j] = sum;
            }

            return output;
        }

        private void LeakyReLU(float[] data, float slope = 0.01f)
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = data[i] > 0 ? data[i] : data[i] * slope;
        }

        private void ReLU(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = MathF.Max(0, data[i]);
        }

        private void Sigmoid(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = 1.0f / (1.0f + MathF.Exp(-data[i]));
        }

        private void TanhActivate(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = MathF.Tanh(data[i]);
        }

        /// <summary>
        /// Performs backpropagation and updates network weights using Adam optimizer.
        /// </summary>
        public void BackwardPass(Vector3[] input, Vector3 target, float learningRate)
        {
            if (!_isInitialized)
                return;

            float[] inputArr = new float[INPUT_FEATURES];
            for (int i = 0; i < INPUT_FEATURES; i++)
                inputArr[i] = input[i].X;

            float[] hidden1 = LinearForward(inputArr, _weights1, _bias1);
            float[] hidden1Act = (float[])hidden1.Clone();
            LeakyReLU(hidden1Act);

            float[] hidden2 = LinearForward(hidden1Act, _weights2, _bias2);
            float[] hidden2Act = (float[])hidden2.Clone();
            LeakyReLU(hidden2Act);

            float[] output = LinearForward(hidden2Act, _weights3, _bias3);
            float[] outputAct = (float[])output.Clone();
            Sigmoid(outputAct);

            float[] targetArr = { target.X, target.Y, target.Z };

            float[] outputGrad = new float[OUTPUT_FEATURES];
            float loss = 0;
            for (int i = 0; i < OUTPUT_FEATURES; i++)
            {
                float diff = outputAct[i] - targetArr[i];
                outputGrad[i] = 2.0f * diff / OUTPUT_FEATURES;
                loss += diff * diff;
            }
            _trainingLoss = _trainingLoss * 0.99f + loss * 0.01f;

            float[,] dW3 = new float[_hidden2Size, OUTPUT_FEATURES];
            float[] dB3 = new float[OUTPUT_FEATURES];
            float[] hidden2Grad = new float[_hidden2Size];

            for (int j = 0; j < OUTPUT_FEATURES; j++)
            {
                float outputDeriv = outputAct[j] * (1.0f - outputAct[j]);
                float delta = outputGrad[j] * outputDeriv;
                dB3[j] = delta;
                for (int i = 0; i < _hidden2Size; i++)
                {
                    dW3[i, j] = hidden2Act[i] * delta;
                    hidden2Grad[i] += _weights3[i, j] * delta;
                }
            }

            for (int i = 0; i < _hidden2Size; i++)
                hidden2Grad[i] *= hidden2Act[i] > 0 ? 1.0f : 0.01f;

            float[,] dW2 = new float[_hidden1Size, _hidden2Size];
            float[] dB2 = new float[_hidden2Size];
            float[] hidden1Grad = new float[_hidden1Size];

            for (int j = 0; j < _hidden2Size; j++)
            {
                float delta = hidden2Grad[j];
                dB2[j] = delta;
                for (int i = 0; i < _hidden1Size; i++)
                {
                    dW2[i, j] = hidden1Act[i] * delta;
                    hidden1Grad[i] += _weights2[i, j] * delta;
                }
            }

            for (int i = 0; i < _hidden1Size; i++)
                hidden1Grad[i] *= hidden1Act[i] > 0 ? 1.0f : 0.01f;

            float[,] dW1 = new float[INPUT_FEATURES, _hidden1Size];
            float[] dB1 = new float[_hidden1Size];

            for (int j = 0; j < _hidden1Size; j++)
            {
                float delta = hidden1Grad[j];
                dB1[j] = delta;
                for (int i = 0; i < INPUT_FEATURES; i++)
                    dW1[i, j] = inputArr[i] * delta;
            }

            _t++;
            float beta1PowT = MathF.Pow(BETA1, _t);
            float beta2PowT = MathF.Pow(BETA2, _t);

            UpdateAdamWeights(_weights1, _m1, _v1, dW1, learningRate, beta1PowT, beta2PowT, INPUT_FEATURES, _hidden1Size);
            UpdateAdamWeights(_weights2, _m2, _v2, dW2, learningRate, beta1PowT, beta2PowT, _hidden1Size, _hidden2Size);
            UpdateAdamWeights(_weights3, _m3, _v3, dW3, learningRate, beta1PowT, beta2PowT, _hidden2Size, OUTPUT_FEATURES);

            UpdateAdamBias(_bias1, _mb1, _vb1, dB1, learningRate, beta1PowT, beta2PowT);
            UpdateAdamBias(_bias2, _mb2, _vb2, dB2, learningRate, beta1PowT, beta2PowT);
            UpdateAdamBias(_bias3, _mb3, _vb3, dB3, learningRate, beta1PowT, beta2PowT);

            _trainingStep++;
        }

        private void UpdateAdamWeights(float[,] weights, float[,] m, float[,] v,
            float[,] grads, float lr, float beta1PowT, float beta2PowT, int rows, int cols)
        {
            float lrCorrected = lr * MathF.Sqrt(1.0f - beta2PowT) / (1.0f - beta1PowT);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m[i, j] = BETA1 * m[i, j] + (1.0f - BETA1) * grads[i, j];
                    v[i, j] = BETA2 * v[i, j] + (1.0f - BETA2) * grads[i, j] * grads[i, j];
                    float mHat = m[i, j] / (1.0f - beta1PowT);
                    float vHat = v[i, j] / (1.0f - beta2PowT);
                    weights[i, j] -= lrCorrected * mHat / (MathF.Sqrt(vHat) + EPSILON_ADAM);
                }
            }
        }

        private void UpdateAdamBias(float[] bias, float[] m, float[] v, float[] grads,
            float lr, float beta1PowT, float beta2PowT)
        {
            float lrCorrected = lr * MathF.Sqrt(1.0f - beta2PowT) / (1.0f - beta1PowT);
            for (int i = 0; i < bias.Length; i++)
            {
                m[i] = BETA1 * m[i] + (1.0f - BETA1) * grads[i];
                v[i] = BETA2 * v[i] + (1.0f - BETA2) * grads[i] * grads[i];
                float mHat = m[i] / (1.0f - beta1PowT);
                float vHat = v[i] / (1.0f - beta2PowT);
                bias[i] -= lrCorrected * mHat / (MathF.Sqrt(vHat) + EPSILON_ADAM);
            }
        }

        /// <summary>
        /// Collects training data from reference path tracer output.
        /// </summary>
        public void CollectTrainingData(Vector3[] features, Vector3 groundTruthIrradiance)
        {
            _trainingBuffer.Enqueue((features, groundTruthIrradiance));
            while (_trainingBuffer.Count > _maxTrainingBufferSize)
                _trainingBuffer.Dequeue();
        }

        /// <summary>
        /// Runs the online training loop on collected data.
        /// </summary>
        public void TrainOnCollectedData(int batchSize, float learningRate)
        {
            if (_trainingBuffer.Count < batchSize)
                return;

            var batch = _trainingBuffer.ToArray();
            var rng = new RandomNumberGenerator((uint)_trainingStep);

            float totalLoss = 0;
            for (int b = 0; b < batchSize; b++)
            {
                int idx = rng.NextInt(0, batch.Length);
                var (features, target) = batch[idx];
                ForwardPass(features);
                BackwardPass(features, target, learningRate);
                totalLoss += _trainingLoss;
            }
        }

        /// <summary>
        /// Computes L1 loss between prediction and target.
        /// </summary>
        public float ComputeL1Loss(Vector3 prediction, Vector3 target)
        {
            Vector3 diff = prediction - target;
            return (MathF.Abs(diff.X) + MathF.Abs(diff.Y) + MathF.Abs(diff.Z)) / 3.0f;
        }

        /// <summary>
        /// Computes L2 loss between prediction and target.
        /// </summary>
        public float ComputeL2Loss(Vector3 prediction, Vector3 target)
        {
            Vector3 diff = prediction - target;
            return (diff.X * diff.X + diff.Y * diff.Y + diff.Z * diff.Z) / 3.0f;
        }

        /// <summary>
        /// Computes perceptual loss (simplified).
        /// </summary>
        public float ComputePerceptualLoss(Vector3 prediction, Vector3 target)
        {
            float l2 = ComputeL2Loss(prediction, target);
            float predLum = 0.2126f * prediction.X + 0.7152f * prediction.Y + 0.0722f * prediction.Z;
            float tgtLum = 0.2126f * target.X + 0.7152f * target.Y + 0.0722f * target.Z;
            float lumLoss = (predLum - tgtLum) * (predLum - tgtLum);
            return l2 * 0.7f + lumLoss * 0.3f;
        }

        /// <summary>
        /// Serializes the network weights to a byte array.
        /// </summary>
        public byte[] Serialize()
        {
            using var ms = new System.IO.MemoryStream();
            using var bw = new System.IO.BinaryWriter(ms);
            bw.Write(INPUT_FEATURES);
            bw.Write(_hidden1Size);
            bw.Write(_hidden2Size);
            bw.Write(OUTPUT_FEATURES);
            bw.Write(_t);

            for (int i = 0; i < INPUT_FEATURES; i++)
                for (int j = 0; j < _hidden1Size; j++)
                    bw.Write(_weights1[i, j]);

            for (int i = 0; i < _hidden1Size; i++)
                for (int j = 0; j < _hidden2Size; j++)
                    bw.Write(_weights2[i, j]);

            for (int i = 0; i < _hidden2Size; i++)
                for (int j = 0; j < OUTPUT_FEATURES; j++)
                    bw.Write(_weights3[i, j]);

            foreach (float bias in _bias1)
                bw.Write(bias);
            foreach (float bias in _bias2)
                bw.Write(bias);
            foreach (float bias in _bias3)
                bw.Write(bias);

            return ms.ToArray();
        }

        /// <summary>
        /// Deserializes network weights from a byte array.
        /// </summary>
        public void Deserialize(byte[] data)
        {
            using var ms = new System.IO.MemoryStream(data);
            using var br = new System.IO.BinaryReader(ms);

            int inputFeat = br.ReadInt32();
            int hidden1 = br.ReadInt32();
            int hidden2 = br.ReadInt32();
            int outputFeat = br.ReadInt32();
            _t = br.ReadInt32();

            _hidden1Size = hidden1;
            _hidden2Size = hidden2;
            Profile = (hidden1, hidden2) switch
            {
                (32, 32) => NeuralNetworkProfile.Tiny,
                (64, 64) => NeuralNetworkProfile.Small,
                _ => NeuralNetworkProfile.Full
            };

            _weights1 = new float[inputFeat, hidden1];
            for (int i = 0; i < inputFeat; i++)
                for (int j = 0; j < hidden1; j++)
                    _weights1[i, j] = br.ReadSingle();

            _weights2 = new float[hidden1, hidden2];
            for (int i = 0; i < hidden1; i++)
                for (int j = 0; j < hidden2; j++)
                    _weights2[i, j] = br.ReadSingle();

            _weights3 = new float[hidden2, outputFeat];
            for (int i = 0; i < hidden2; i++)
                for (int j = 0; j < outputFeat; j++)
                    _weights3[i, j] = br.ReadSingle();

            _bias1 = new float[hidden1];
            _bias2 = new float[hidden2];
            _bias3 = new float[outputFeat];
            for (int i = 0; i < hidden1; i++)
                _bias1[i] = br.ReadSingle();
            for (int i = 0; i < hidden2; i++)
                _bias2[i] = br.ReadSingle();
            for (int i = 0; i < outputFeat; i++)
                _bias3[i] = br.ReadSingle();

            _m1 = new float[inputFeat, hidden1];
            _v1 = new float[inputFeat, hidden1];
            _m2 = new float[hidden1, hidden2];
            _v2 = new float[hidden1, hidden2];
            _m3 = new float[hidden2, outputFeat];
            _v3 = new float[hidden2, outputFeat];
            _mb1 = new float[hidden1];
            _mb2 = new float[hidden2];
            _mb3 = new float[outputFeat];

            _isInitialized = true;
        }

        /// <summary>
        /// Optimizes inference by fusing operations (simulated).
        /// </summary>
        public Vector3 OptimizedInference(Vector3[] features)
        {
            return ForwardPass(features);
        }

        /// <summary>
        /// Quantizes weights to reduced precision (simulated).
        /// </summary>
        public void QuantizeWeights(int bits)
        {
            float scale = MathF.Pow(2, bits) - 1;
            float invScale = 1.0f / scale;

            for (int i = 0; i < _weights1.GetLength(0); i++)
                for (int j = 0; j < _weights1.GetLength(1); j++)
                    _weights1[i, j] = MathF.Round(_weights1[i, j] * scale) * invScale;

            for (int i = 0; i < _weights2.GetLength(0); i++)
                for (int j = 0; j < _weights2.GetLength(1); j++)
                    _weights2[i, j] = MathF.Round(_weights2[i, j] * scale) * invScale;

            for (int i = 0; i < _weights3.GetLength(0); i++)
                for (int j = 0; j < _weights3.GetLength(1); j++)
                    _weights3[i, j] = MathF.Round(_weights3[i, j] * scale) * invScale;
        }
    }
}
