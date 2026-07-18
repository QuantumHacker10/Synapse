using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;


// ============================================================
// FILE: GradientCalculator.cs
// PATH: Evaluation/GradientCalculator.cs
// ============================================================


using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using GDNN.Core.NeuralNetwork;

namespace GDNN.Evaluation
{
    /// <summary>
    /// Configuration for gradient computation.
    /// </summary>
    public sealed class GradientConfig
    {
        /// <summary>Finite difference step size.</summary>
        public float Epsilon { get; set; } = 0.001f;

        /// <summary>Number of Hessian-vector product iterations.</summary>
        public int HessianIterations { get; set; } = 10;

        /// <summary>Normal smoothing kernel radius.</summary>
        public float NormalSmoothRadius { get; set; } = 0.01f;

        /// <summary>Number of Laplacian smoothing iterations.</summary>
        public int LaplacianIterations { get; set; } = 3;

        /// <summary>Laplacian smoothing weight.</summary>
        public float LaplacianWeight { get; set; } = 0.5f;

        /// <summary>Curvature computation sample distance.</summary>
        public float CurvatureSampleDistance { get; set; } = 0.01f;

        /// <summary>Maximum gradient magnitude (for clamping).</summary>
        public float MaxGradientMagnitude { get; set; } = 10.0f;

        /// <summary>Enable gradient normalization.</summary>
        public bool NormalizeGradients { get; set; } = true;

        /// <summary>Backpropagation learning rate for analytical gradients.</summary>
        public float BackpropLearningRate { get; set; } = 0.001f;
    }

    /// <summary>
    /// Represents the gradient field evaluation result for a region.
    /// </summary>
    public sealed class GradientFieldResult
    {
        /// <summary>Gradient vectors at each sample point.</summary>
        public Vector3[] Gradients { get; set; } = Array.Empty<Vector3>();

        /// <summary>SDF values at each sample point.</summary>
        public float[] SdfValues { get; set; } = Array.Empty<float>();

        /// <summary>Sample points.</summary>
        public Vector3[] Points { get; set; } = Array.Empty<Vector3>();

        /// <summary>Magnitudes of the gradients.</summary>
        public float[] Magnitudes { get; set; } = Array.Empty<float>();

        /// <summary>Number of valid gradient samples.</summary>
        public int ValidSamples { get; set; }
    }

    /// <summary>
    /// Visualization data for gradient fields.
    /// </summary>
    public sealed class GradientVisualizationData
    {
        /// <summary>Line segments (start, end) for arrow visualization.</summary>
        public List<(Vector3 Start, Vector3 End)> ArrowLines { get; set; } = new();

        /// <summary>Colors for each arrow.</summary>
        public List<Vector3> Colors { get; set; } = new();

        /// <summary>Points representing gradient origins.</summary>
        public Vector3[] Origins { get; set; } = Array.Empty<Vector3>();

        /// <summary>Gradient directions (normalized).</summary>
        public Vector3[] Directions { get; set; } = Array.Empty<Vector3>();

        /// <summary>Lengths of each arrow.</summary>
        public float[] Lengths { get; set; } = Array.Empty<float>();
    }

    /// <summary>
    /// Provides numerical and analytical gradient computation, Hessian-vector products,
    /// normal smoothing, curvature estimation, Laplacian smoothing, and gradient field
    /// visualization data generation for neural SDFs.
    /// </summary>
    public sealed class GradientCalculator : IDisposable
    {
        private readonly GradientConfig _config;
        private bool _disposed;

        /// <summary>Gets the configuration.</summary>
        public GradientConfig Config => _config;

        /// <summary>
        /// Initializes a new gradient calculator.
        /// </summary>
        /// <param name="config">Configuration. Uses defaults if null.</param>
        public GradientCalculator(GradientConfig? config = null)
        {
            _config = config ?? new GradientConfig();
        }

        /// <summary>
        /// Computes the gradient using forward differences (3 evaluations).
        /// Fastest but least accurate.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <returns>Gradient vector.</returns>
        public Vector3 ComputeForwardDifference(MicroMLP network, Vector3 point)
        {
            float f0 = network.Evaluate(point);
            float fx = network.Evaluate(point + new Vector3(_config.Epsilon, 0, 0));
            float fy = network.Evaluate(point + new Vector3(0, _config.Epsilon, 0));
            float fz = network.Evaluate(point + new Vector3(0, 0, _config.Epsilon));

            return ClampGradient(new Vector3(
                (fx - f0) / _config.Epsilon,
                (fy - f0) / _config.Epsilon,
                (fz - f0) / _config.Epsilon));
        }

        /// <summary>
        /// Computes the gradient using backward differences (3 evaluations).
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <returns>Gradient vector.</returns>
        public Vector3 ComputeBackwardDifference(MicroMLP network, Vector3 point)
        {
            float f0 = network.Evaluate(point);
            float fx = network.Evaluate(point - new Vector3(_config.Epsilon, 0, 0));
            float fy = network.Evaluate(point - new Vector3(0, _config.Epsilon, 0));
            float fz = network.Evaluate(point - new Vector3(0, 0, _config.Epsilon));

            return ClampGradient(new Vector3(
                (f0 - fx) / _config.Epsilon,
                (f0 - fy) / _config.Epsilon,
                (f0 - fz) / _config.Epsilon));
        }

        /// <summary>
        /// Computes the gradient using central differences (6 evaluations).
        /// More accurate than forward/backward.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <returns>Gradient vector.</returns>
        public Vector3 ComputeCentralDifference(MicroMLP network, Vector3 point)
        {
            float fxp = network.Evaluate(point + new Vector3(_config.Epsilon, 0, 0));
            float fxm = network.Evaluate(point - new Vector3(_config.Epsilon, 0, 0));
            float fyp = network.Evaluate(point + new Vector3(0, _config.Epsilon, 0));
            float fym = network.Evaluate(point - new Vector3(0, _config.Epsilon, 0));
            float fzp = network.Evaluate(point + new Vector3(0, 0, _config.Epsilon));
            float fzm = network.Evaluate(point - new Vector3(0, 0, _config.Epsilon));

            float inv2e = 1.0f / (2.0f * _config.Epsilon);

            return ClampGradient(new Vector3(
                (fxp - fxm) * inv2e,
                (fyp - fym) * inv2e,
                (fzp - fzm) * inv2e));
        }

        /// <summary>
        /// Computes the gradient using the network's built-in backpropagation.
        /// This is the most accurate method if supported.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <returns>Gradient vector.</returns>
        public Vector3 ComputeAnalyticalGradient(MicroMLP network, Vector3 point)
        {
            Vector3 grad = network.ComputeGradient(point);
            return _config.NormalizeGradients ? Vector3.Normalize(grad) : ClampGradient(grad);
        }

        /// <summary>
        /// Computes the gradient using the network's EvaluateWithGradient method.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <param name="sdfValue">Output SDF value.</param>
        /// <returns>Gradient vector.</returns>
        public Vector3 ComputeGradientWithValue(MicroMLP network, Vector3 point, out float sdfValue)
        {
            float sdf = network.EvaluateWithGradient(point, out Vector3 grad);
            sdfValue = sdf;
            return _config.NormalizeGradients ? Vector3.Normalize(grad) : ClampGradient(grad);
        }

        /// <summary>
        /// Computes Hessian-vector product using forward differences.
        /// This gives the directional derivative of the gradient in direction v.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <param name="v">Direction vector.</param>
        /// <returns>Hessian-vector product Hv.</returns>
        public Vector3 ComputeHessianVectorProduct(MicroMLP network, Vector3 point, Vector3 v)
        {
            Vector3 grad0 = network.ComputeGradient(point);
            Vector3 grad1 = network.ComputeGradient(point + v * _config.Epsilon);

            return (grad1 - grad0) / _config.Epsilon;
        }

        /// <summary>
        /// Computes the full Hessian matrix via finite differences of gradients.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <returns>3x3 Hessian as a tuple of row vectors.</returns>
        public (Vector3 Row0, Vector3 Row1, Vector3 Row2) ComputeFullHessian(MicroMLP network, Vector3 point)
        {
            Vector3 eX = new Vector3(_config.Epsilon, 0, 0);
            Vector3 eY = new Vector3(0, _config.Epsilon, 0);
            Vector3 eZ = new Vector3(0, 0, _config.Epsilon);

            Vector3 gradXP = network.ComputeGradient(point + eX);
            Vector3 gradXM = network.ComputeGradient(point - eX);
            Vector3 gradYP = network.ComputeGradient(point + eY);
            Vector3 gradYM = network.ComputeGradient(point - eY);
            Vector3 gradZP = network.ComputeGradient(point + eZ);
            Vector3 gradZM = network.ComputeGradient(point - eZ);

            float inv2e = 1.0f / (2.0f * _config.Epsilon);

            // Hessian rows are derivatives of gradient components
            Vector3 h0 = (gradXP - gradXM) * inv2e; // d/dx of gradient
            Vector3 h1 = (gradYP - gradYM) * inv2e; // d/dy of gradient
            Vector3 h2 = (gradZP - gradZM) * inv2e; // d/dz of gradient

            return (h0, h1, h2);
        }

        /// <summary>
        /// Estimates curvature using the Hessian eigenvalues.
        /// Returns mean curvature and Gaussian curvature.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <param name="meanCurvature">Output mean curvature.</param>
        /// <param name="gaussianCurvature">Output Gaussian curvature.</param>
        public void EstimateCurvature(MicroMLP network, Vector3 point,
            out float meanCurvature, out float gaussianCurvature)
        {
            var (h0, h1, h2) = ComputeFullHessian(network, point);
            Vector3 grad = network.ComputeGradient(point);
            float gradLen = grad.Length();

            if (gradLen < 1e-8f)
            {
                meanCurvature = 0;
                gaussianCurvature = 0;
                return;
            }

            Vector3 normal = grad / gradLen;

            // Project Hessian onto the tangent plane
            // H_tangent = (I - n*n^T) * H * (I - n*n^T)
            Matrix4x4 identity = Matrix4x4.Identity;
            Matrix4x4 nnT = new Matrix4x4(
                normal.X * normal.X, normal.X * normal.Y, normal.X * normal.Z, 0,
                normal.Y * normal.X, normal.Y * normal.Y, normal.Y * normal.Z, 0,
                normal.Z * normal.X, normal.Z * normal.Y, normal.Z * normal.Z, 0,
                0, 0, 0, 1
            );

            Matrix4x4 tangentProject = identity - nnT;
            Matrix4x4 hessian = new Matrix4x4(
                h0.X, h0.Y, h0.Z, 0,
                h1.X, h1.Y, h1.Z, 0,
                h2.X, h2.Y, h2.Z, 0,
                0, 0, 0, 1
            );

            // Compute mean curvature: trace of projected Hessian / (2 * |grad|)
            float trace = h0.X + h1.Y + h2.Z;
            meanCurvature = trace / (2.0f * gradLen);

            // Gaussian curvature approximation using principal curvatures
            // k1 * k2 ≈ det(H_tangent) / |grad|^4
            float k1k2 = (h0.X * (h1.Y * h2.Z - h1.Z * h2.Y)
                        - h0.Y * (h1.X * h2.Z - h1.Z * h2.X)
                        + h0.Z * (h1.X * h2.Y - h1.Y * h2.X))
                        / (gradLen * gradLen * gradLen * gradLen);

            gaussianCurvature = k1k2;
        }

        /// <summary>
        /// Computes principal curvatures at a surface point.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <param name="k1">Maximum principal curvature.</param>
        /// <param name="k2">Minimum principal curvature.</param>
        /// <param name="principalDir1">Principal direction for k1.</param>
        /// <param name="principalDir2">Principal direction for k2.</param>
        public void ComputePrincipalCurvatures(MicroMLP network, Vector3 point,
            out float k1, out float k2, out Vector3 principalDir1, out Vector3 principalDir2)
        {
            var (h0, h1, h2) = ComputeFullHessian(network, point);
            Vector3 grad = network.ComputeGradient(point);
            float gradLen = grad.Length();

            if (gradLen < 1e-8f)
            {
                k1 = k2 = 0;
                principalDir1 = principalDir2 = Vector3.UnitX;
                return;
            }

            Vector3 normal = grad / gradLen;

            // Build tangent-frame Hessian (2x2)
            Vector3 t1 = MathF.Abs(normal.X) < 0.9f
                ? Vector3.Cross(normal, Vector3.UnitX)
                : Vector3.Cross(normal, Vector3.UnitY);
            t1 = Vector3.Normalize(t1);
            Vector3 t2 = Vector3.Normalize(Vector3.Cross(normal, t1));

            // Project gradient onto tangent frame
            float h11 = Vector3.Dot(h0, t1);
            float h12 = Vector3.Dot(h0, t2);
            float h21 = Vector3.Dot(h1, t1);
            float h22 = Vector3.Dot(h1, t2);

            // Eigenvalue decomposition of 2x2 matrix
            float trace = h11 + h22;
            float det = h11 * h22 - h12 * h21;
            float discriminant = trace * trace - 4.0f * det;

            if (discriminant < 0)
            {
                k1 = k2 = trace * 0.5f;
                principalDir1 = t1;
                principalDir2 = t2;
                return;
            }

            float sqrtDisc = MathF.Sqrt(discriminant);
            k1 = (trace + sqrtDisc) * 0.5f;
            k2 = (trace - sqrtDisc) * 0.5f;

            // Compute principal directions
            if (MathF.Abs(h12) > 1e-8f)
            {
                principalDir1 = Vector3.Normalize(t1 * (k1 - h22) + t2 * h12);
                principalDir2 = Vector3.Normalize(t1 * (k2 - h22) + t2 * h12);
            }
            else
            {
                principalDir1 = h11 > h22 ? t1 : t2;
                principalDir2 = h11 > h22 ? t2 : t1;
            }
        }

        /// <summary>
        /// Smooths normals in a neighborhood by averaging.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Center point.</param>
        /// <param name="sampleCount">Number of samples in the neighborhood.</param>
        /// <returns>Smoothed normal.</returns>
        public Vector3 SmoothNormal(MicroMLP network, Vector3 point, int sampleCount = 8)
        {
            Vector3 smoothNormal = network.ComputeGradient(point);
            float totalWeight = 1.0f;

            // Random-ish sampling in a hemisphere
            for (int i = 0; i < sampleCount; i++)
            {
                float angle = (float)i / sampleCount * MathF.PI;
                float phi = (float)((i * 137.508) % 360) / 360.0f * MathF.PI * 2.0f;

                Vector3 offset = new Vector3(
                    MathF.Sin(angle) * MathF.Cos(phi) * _config.NormalSmoothRadius,
                    MathF.Cos(angle) * _config.NormalSmoothRadius,
                    MathF.Sin(angle) * MathF.Sin(phi) * _config.NormalSmoothRadius);

                Vector3 samplePoint = point + offset;
                Vector3 sampleNormal = network.ComputeGradient(samplePoint);

                float dist = offset.Length();
                float weight = MathF.Exp(-dist * dist / (_config.NormalSmoothRadius * _config.NormalSmoothRadius));

                // Ensure consistent orientation
                if (Vector3.Dot(sampleNormal, smoothNormal) < 0)
                    sampleNormal = -sampleNormal;

                smoothNormal += sampleNormal * weight;
                totalWeight += weight;
            }

            smoothNormal /= totalWeight;

            if (smoothNormal.LengthSquared() > 1e-10f)
                return Vector3.Normalize(smoothNormal);
            return network.ComputeGradient(point);
        }

        /// <summary>
        /// Applies Laplacian smoothing to a gradient field.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="points">Sample points.</param>
        /// <param name="gradients">Input/output gradient array.</param>
        public void LaplacianSmoothGradients(MicroMLP network, Vector3[] points, Vector3[] gradients)
        {
            if (points.Length != gradients.Length)
                throw new ArgumentException("Points and gradients must have the same length.");

            int count = points.Length;
            var smoothed = new Vector3[count];

            for (int iter = 0; iter < _config.LaplacianIterations; iter++)
            {
                for (int i = 0; i < count; i++)
                {
                    Vector3 laplacian = Vector3.Zero;
                    int neighbors = 0;

                    // Find neighbors within radius
                    for (int j = 0; j < count; j++)
                    {
                        if (i == j) continue;

                        float dist = Vector3.Distance(points[i], points[j]);
                        if (dist < _config.NormalSmoothRadius)
                        {
                            laplacian += gradients[j];
                            neighbors++;
                        }
                    }

                    if (neighbors > 0)
                    {
                        laplacian /= neighbors;
                        smoothed[i] = gradients[i] + _config.LaplacianWeight *
                            (laplacian - gradients[i]);
                    }
                    else
                    {
                        smoothed[i] = gradients[i];
                    }
                }

                // Copy smoothed back
                for (int i = 0; i < count; i++)
                {
                    gradients[i] = smoothed[i];
                }
            }

            // Re-normalize if configured
            if (_config.NormalizeGradients)
            {
                for (int i = 0; i < count; i++)
                {
                    if (gradients[i].LengthSquared() > 1e-10f)
                        gradients[i] = Vector3.Normalize(gradients[i]);
                }
            }
        }

        /// <summary>
        /// Computes the Laplacian of the SDF (divergence of gradient).
        /// This measures the sum of principal curvatures.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <returns>Laplacian value.</returns>
        public float ComputeLaplacian(MicroMLP network, Vector3 point)
        {
            float f0 = network.Evaluate(point);
            float fxp = network.Evaluate(point + new Vector3(_config.Epsilon, 0, 0));
            float fxm = network.Evaluate(point - new Vector3(_config.Epsilon, 0, 0));
            float fyp = network.Evaluate(point + new Vector3(0, _config.Epsilon, 0));
            float fym = network.Evaluate(point - new Vector3(0, _config.Epsilon, 0));
            float fzp = network.Evaluate(point + new Vector3(0, 0, _config.Epsilon));
            float fzm = network.Evaluate(point - new Vector3(0, 0, _config.Epsilon));

            float invE2 = 1.0f / (_config.Epsilon * _config.Epsilon);

            return (fxp + fxm + fyp + fym + fzp + fzm - 6.0f * f0) * invE2;
        }

        /// <summary>
        /// Computes gradient field for a grid of points.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="min">Minimum corner of the grid.</param>
        /// <param name="max">Maximum corner of the grid.</param>
        /// <param name="resolution">Number of samples along each axis.</param>
        /// <returns>Gradient field result.</returns>
        public GradientFieldResult ComputeGradientField(MicroMLP network, Vector3 min, Vector3 max, int resolution)
        {
            int totalPoints = resolution * resolution * resolution;
            var result = new GradientFieldResult
            {
                Points = new Vector3[totalPoints],
                Gradients = new Vector3[totalPoints],
                SdfValues = new float[totalPoints],
                Magnitudes = new float[totalPoints],
                ValidSamples = 0
            };

            Vector3 step = (max - min) / MathF.Max(1, resolution - 1);

            for (int z = 0; z < resolution; z++)
            for (int y = 0; y < resolution; y++)
            for (int x = 0; x < resolution; x++)
            {
                int idx = x + y * resolution + z * resolution * resolution;
                Vector3 point = min + new Vector3(x * step.X, y * step.Y, z * step.Z);

                result.Points[idx] = point;
                result.SdfValues[idx] = network.Evaluate(point);
                result.Gradients[idx] = network.ComputeGradient(point);
                result.Magnitudes[idx] = result.Gradients[idx].Length();

                if (result.Magnitudes[idx] > 1e-8f)
                    result.ValidSamples++;
            }

            return result;
        }

        /// <summary>
        /// Generates visualization data for a gradient field.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="points">Points to visualize.</param>
        /// <param name="arrowScale">Scale factor for arrow length.</param>
        /// <returns>Visualization data.</returns>
        public GradientVisualizationData GenerateVisualizationData(MicroMLP network,
            ReadOnlySpan<Vector3> points, float arrowScale = 0.1f)
        {
            var data = new GradientVisualizationData
            {
                Origins = new Vector3[points.Length],
                Directions = new Vector3[points.Length],
                Lengths = new float[points.Length]
            };

            for (int i = 0; i < points.Length; i++)
            {
                Vector3 point = points[i];
                Vector3 grad = network.ComputeGradient(point);
                float mag = grad.Length();

                data.Origins[i] = point;
                data.Lengths[i] = mag * arrowScale;

                if (mag > 1e-8f)
                {
                    data.Directions[i] = grad / mag;

                    // Color based on magnitude (blue=low, red=high)
                    float t = Math.Clamp(mag * 0.5f, 0, 1);
                    data.Colors.Add(new Vector3(t, 0.2f, 1.0f - t));
                }
                else
                {
                    data.Directions[i] = Vector3.UnitY;
                    data.Colors.Add(new Vector3(0.5f, 0.5f, 0.5f));
                }

                Vector3 end = point + data.Directions[i] * data.Lengths[i];
                data.ArrowLines.Add((point, end));
            }

            return data;
        }

        /// <summary>
        /// Computes the directional derivative of the SDF in a given direction.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <param name="direction">Direction.</param>
        /// <returns>Directional derivative.</returns>
        public float ComputeDirectionalDerivative(MicroMLP network, Vector3 point, Vector3 direction)
        {
            Vector3 grad = network.ComputeGradient(point);
            return Vector3.Dot(grad, Vector3.Normalize(direction));
        }

        /// <summary>
        /// Estimates the Hessian-vector product using the Pearlmutter method.
        /// More efficient than computing the full Hessian.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <param name="v">Vector for the product.</param>
        /// <returns>Hessian-vector product.</returns>
        public Vector3 PearlmutterHessianVector(MicroMLP network, Vector3 point, Vector3 v)
        {
            float eps = _config.Epsilon;

            // grad(f(x + eps*v)) - grad(f(x)) / eps
            Vector3 gradAtPoint = network.ComputeGradient(point);
            Vector3 gradAtOffset = network.ComputeGradient(point + v * eps);

            return (gradAtOffset - gradAtPoint) / eps;
        }

        /// <summary>
        /// Computes second-order directional derivative of the SDF.
        /// This is v^T * H * v where H is the Hessian.
        /// </summary>
        /// <param name="network">Neural network.</param>
        /// <param name="point">Evaluation point.</param>
        /// <param name="v">Direction vector.</param>
        /// <returns>Second-order directional derivative.</returns>
        public float ComputeSecondDirectionalDerivative(MicroMLP network, Vector3 point, Vector3 v)
        {
            Vector3 hv = PearlmutterHessianVector(network, point, v);
            return Vector3.Dot(v, hv);
        }

        /// <summary>
        /// Clamps gradient magnitude to configured maximum.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector3 ClampGradient(Vector3 gradient)
        {
            if (_config.NormalizeGradients)
            {
                float mag = gradient.Length();
                if (mag > 1e-8f)
                    return gradient / MathF.Min(mag, _config.MaxGradientMagnitude);
                return Vector3.Zero;
            }

            if (gradient.Length() > _config.MaxGradientMagnitude)
                return Vector3.Normalize(gradient) * _config.MaxGradientMagnitude;

            return gradient;
        }

        /// <summary>
        /// Disposes the gradient calculator.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// Provides specialized gradient operations for neural SDF evaluation.
    /// </summary>
    public static class GradientOps
    {
        /// <summary>
        /// Computes the angle between two gradient vectors.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float AngleBetween(Vector3 a, Vector3 b)
        {
            float dot = Vector3.Dot(
                a.LengthSquared() > 1e-10f ? Vector3.Normalize(a) : Vector3.Zero,
                b.LengthSquared() > 1e-10f ? Vector3.Normalize(b) : Vector3.Zero);
            return MathF.Acos(Math.Clamp(dot, -1.0f, 1.0f));
        }

        /// <summary>
        /// Computes the gradient magnitude at a point (convenience method).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float GradientMagnitude(MicroMLP network, Vector3 point)
        {
            return network.ComputeGradient(point).Length();
        }

        /// <summary>
        /// Returns true if the gradient is approximately zero (degenerate point).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsDegenerate(Vector3 gradient, float threshold = 1e-8f)
        {
            return gradient.LengthSquared() < threshold * threshold;
        }

        /// <summary>
        /// Computes the signed distance gradient field divergence at a point.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Divergence(MicroMLP network, Vector3 point, float epsilon = 0.001f)
        {
            Vector3 gxp = network.ComputeGradient(point + new Vector3(epsilon, 0, 0));
            Vector3 gxm = network.ComputeGradient(point - new Vector3(epsilon, 0, 0));
            Vector3 gyp = network.ComputeGradient(point + new Vector3(0, epsilon, 0));
            Vector3 gym = network.ComputeGradient(point - new Vector3(0, epsilon, 0));
            Vector3 gzp = network.ComputeGradient(point + new Vector3(0, 0, epsilon));
            Vector3 gzm = network.ComputeGradient(point - new Vector3(0, 0, epsilon));

            float inv2e = 1.0f / (2.0f * epsilon);

            float dgDx = (gxp.X - gxm.X) * inv2e;
            float dgDy = (gyp.Y - gym.Y) * inv2e;
            float dgDz = (gzp.Z - gzm.Z) * inv2e;

            return dgDx + dgDy + dgDz;
        }

        /// <summary>
        /// Computes the curl of the gradient field (should be ~0 for a gradient).
        /// Useful for detecting non-conservative fields.
        /// </summary>
        public static Vector3 Curl(MicroMLP network, Vector3 point, float epsilon = 0.001f)
        {
            Vector3 gxp = network.ComputeGradient(point + new Vector3(epsilon, 0, 0));
            Vector3 gxm = network.ComputeGradient(point - new Vector3(epsilon, 0, 0));
            Vector3 gyp = network.ComputeGradient(point + new Vector3(0, epsilon, 0));
            Vector3 gym = network.ComputeGradient(point - new Vector3(0, epsilon, 0));
            Vector3 gzp = network.ComputeGradient(point + new Vector3(0, 0, epsilon));
            Vector3 gzm = network.ComputeGradient(point - new Vector3(0, 0, epsilon));

            float inv2e = 1.0f / (2.0f * epsilon);

            float curlX = (gzp.Y - gzm.Y) * inv2e - (gyp.Z - gym.Z) * inv2e;
            float curlY = (gxp.Z - gxm.Z) * inv2e - (gzp.X - gzm.X) * inv2e;
            float curlZ = (gyp.X - gym.X) * inv2e - (gxp.Y - gxm.Y) * inv2e;

            return new Vector3(curlX, curlY, curlZ);
        }
    }

    /// <summary>
    /// Provides helper list type for gradient visualization.
    /// </summary>
    public sealed class List<T>
    {
        private T[] _items;
        private int _count;

        /// <summary>Gets the number of items.</summary>
        public int Count => _count;

        /// <summary>Gets or sets capacity.</summary>
        public int Capacity
        {
            get => _items.Length;
            set
            {
                if (value < _count) throw new ArgumentOutOfRangeException(nameof(value));
                if (value != _items.Length)
                {
                    var newItems = new T[value];
                    if (_count > 0) Array.Copy(_items, newItems, _count);
                    _items = newItems;
                }
            }
        }

        /// <summary>
        /// Initializes a new list with default capacity.
        /// </summary>
        public List() => _items = Array.Empty<T>();

        /// <summary>
        /// Initializes with specified capacity.
        /// </summary>
        public List(int capacity) => _items = capacity > 0 ? new T[capacity] : Array.Empty<T>();

        /// <summary>Adds an item.</summary>
        public void Add(T item)
        {
            if (_count == _items.Length)
                Capacity = _items.Length == 0 ? 4 : _items.Length * 2;
            _items[_count++] = item;
        }

        /// <summary>Clears all items.</summary>
        public void Clear()
        {
            Array.Clear(_items, 0, _count);
            _count = 0;
        }

        /// <summary>Gets item by index.</summary>
        public T this[int index]
        {
            get => index >= 0 && index < _count ? _items[index] : throw new IndexOutOfRangeException();
            set
            {
                if (index >= 0 && index < _count) _items[index] = value;
                else throw new IndexOutOfRangeException();
            }
        }

        /// <summary>Returns items as array.</summary>
        public T[] ToArray()
        {
            var result = new T[_count];
            if (_count > 0) Array.Copy(_items, result, _count);
            return result;
        }

        /// <summary>Removes item at specified index.</summary>
        public void RemoveAt(int index)
        {
            if (index < 0 || index >= _count) throw new ArgumentOutOfRangeException(nameof(index));
            _count--;
            if (index < _count)
                Array.Copy(_items, index + 1, _items, index, _count - index);
            _items[_count] = default!;
        }

        /// <summary>Sorts items using a comparison delegate.</summary>
        public void Sort(Comparison<T> comparison)
        {
            if (_count < 2) return;
            var span = _items.AsSpan(0, _count);
            span.Sort(comparison);
        }
    }
}
