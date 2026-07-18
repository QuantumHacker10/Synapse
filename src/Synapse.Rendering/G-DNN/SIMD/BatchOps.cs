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
// FILE: BatchOps.cs
// PATH: SIMD/BatchOps.cs
// ============================================================


// SPDX-License-Identifier: MIT
// GDNN Neural Geometry Engine - SIMD Batch Operations
// High-performance SIMD batch processing for neural network evaluation.

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace GDNN.SIMD;

/// <summary>
/// Provides SIMD-optimized batch processing operations for neural network inference,
/// including matrix-vector multiplication for entire layers, activation functions,
/// normalization, weight compression, and aligned memory operations.
/// </summary>
public static unsafe class BatchOps
{
    #region Matrix-Vector Multiply for Layers

    /// <summary>
    /// Multiplies a weight matrix by an input vector for a fully-connected layer.
    /// Weights are stored in row-major layout: weight[outNeuron * inNeuron + inIndex].
    /// </summary>
    /// <param name="weights">Weight matrix (outNeuron * inNeuron floats, row-major).</param>
    /// <param name="input">Input vector (inNeuron floats).</param>
    /// <param name="output">Output vector (outNeuron floats).</param>
    /// <param name="outNeuron">Number of output neurons.</param>
    /// <param name="inNeuron">Number of input neurons.</param>
    public static void MatVecMultiply(
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> input,
        Span<float> output,
        int outNeuron,
        int inNeuron)
    {
        if (weights.Length < outNeuron * inNeuron || input.Length < inNeuron || output.Length < outNeuron)
            throw new ArgumentException("Input spans are too small for the specified dimensions.");

        for (int outIdx = 0; outIdx < outNeuron; outIdx++)
        {
            int rowStart = outIdx * inNeuron;
            float sum = 0;

            int inIdx = 0;

            if (Avx2.IsSupported)
            {
                var sumVec = Vector256<float>.Zero;
                int simdEnd = inNeuron - (inNeuron % 8);

                for (; inIdx < simdEnd; inIdx += 8)
                {
                    var w = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(weights.Slice(rowStart + inIdx)));
                    var inp = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(input.Slice(inIdx)));
                    if (Fma.IsSupported)
                        sumVec = Fma.MultiplyAdd(w, inp, sumVec);
                    else
                        sumVec = Avx.Add(sumVec, Avx.Multiply(w, inp));
                }

                var upper = Avx.ExtractVector128(sumVec, 1);
                var lower = sumVec.GetLower();
                var summed128 = Sse.Add(upper, lower);
                summed128 = Sse.Add(summed128, Sse.MoveHighToLow(summed128, summed128));
                summed128 = Sse.AddScalar(summed128, Sse.Shuffle(summed128, summed128, 0x55));
                sum = summed128.ToScalar();
            }
            else if (Sse.IsSupported)
            {
                var sumVec = Vector128<float>.Zero;
                int simdEnd = inNeuron - (inNeuron % 4);

                for (; inIdx < simdEnd; inIdx += 4)
                {
                    var w = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(weights.Slice(rowStart + inIdx)));
                    var inp = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(input.Slice(inIdx)));
                    if (Fma.IsSupported)
                        sumVec = Fma.MultiplyAdd(w, inp, sumVec);
                    else
                        sumVec = Sse.Add(sumVec, Sse.Multiply(w, inp));
                }

                sumVec = Sse.Add(sumVec, Sse.MoveHighToLow(sumVec, sumVec));
                sumVec = Sse.AddScalar(sumVec, Sse.Shuffle(sumVec, sumVec, 0x55));
                sum = sumVec.ToScalar();
            }

            for (; inIdx < inNeuron; inIdx++)
                sum += weights[rowStart + inIdx] * input[inIdx];

            output[outIdx] = sum;
        }
    }

    /// <summary>
    /// Matrix-vector multiply with bias addition for a fully-connected layer.
    /// </summary>
    public static void MatVecMultiplyBias(
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int outNeuron,
        int inNeuron)
    {
        if (bias.Length < outNeuron)
            throw new ArgumentException("Bias span too small.");

        MatVecMultiply(weights, input, output, outNeuron, inNeuron);

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = outNeuron - (outNeuron % 8);
            for (; i < simdEnd; i += 8)
            {
                var o = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(output.Slice(i)));
                var b = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(bias.Slice(i)));
                Vector256.StoreUnsafe(Avx.Add(o, b), ref MemoryMarshal.GetReference(output.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = outNeuron - (outNeuron % 4);
            for (; i < simdEnd; i += 4)
            {
                var o = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(output.Slice(i)));
                var b = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(bias.Slice(i)));
                Vector128.StoreUnsafe(Sse.Add(o, b), ref MemoryMarshal.GetReference(output.Slice(i)));
            }
        }

        for (; i < outNeuron; i++)
            output[i] += bias[i];
    }

    /// <summary>
    /// Transposed matrix-vector multiply (input * W^T) for a fully-connected layer.
    /// Weights stored column-major: weight[inNeuron * outNeuron + outIndex].
    /// </summary>
    public static void MatVecMultiplyTransposed(
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> input,
        Span<float> output,
        int outNeuron,
        int inNeuron)
    {
        if (weights.Length < outNeuron * inNeuron || input.Length < inNeuron || output.Length < outNeuron)
            throw new ArgumentException("Input spans are too small.");

        for (int outIdx = 0; outIdx < outNeuron; outIdx++)
        {
            float sum = 0;
            int inIdx = 0;

            if (Avx2.IsSupported)
            {
                var sumVec = Vector256<float>.Zero;
                int simdEnd = inNeuron - (inNeuron % 8);

                for (; inIdx < simdEnd; inIdx += 8)
                {
                    var w = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(weights.Slice(inIdx * outNeuron + outIdx)));
                    var inp = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(input.Slice(inIdx)));
                    if (Fma.IsSupported)
                        sumVec = Fma.MultiplyAdd(w, inp, sumVec);
                    else
                        sumVec = Avx.Add(sumVec, Avx.Multiply(w, inp));
                }

                var upper = Avx.ExtractVector128(sumVec, 1);
                var lower = sumVec.GetLower();
                var summed128 = Sse.Add(upper, lower);
                summed128 = Sse.Add(summed128, Sse.MoveHighToLow(summed128, summed128));
                summed128 = Sse.AddScalar(summed128, Sse.Shuffle(summed128, summed128, 0x55));
                sum = summed128.ToScalar();
            }
            else if (Sse.IsSupported)
            {
                var sumVec = Vector128<float>.Zero;
                int simdEnd = inNeuron - (inNeuron % 4);

                for (; inIdx < simdEnd; inIdx += 4)
                {
                    var w = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(weights.Slice(inIdx * outNeuron + outIdx)));
                    var inp = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(input.Slice(inIdx)));
                    sumVec = Sse.Add(sumVec, Sse.Multiply(w, inp));
                }

                sumVec = Sse.Add(sumVec, Sse.MoveHighToLow(sumVec, sumVec));
                sumVec = Sse.AddScalar(sumVec, Sse.Shuffle(sumVec, sumVec, 0x55));
                sum = sumVec.ToScalar();
            }

            for (; inIdx < inNeuron; inIdx++)
                sum += weights[inIdx * outNeuron + outIdx] * input[inIdx];

            output[outIdx] = sum;
        }
    }

    /// <summary>
    /// Batch matrix-vector multiply for multiple inputs through the same layer.
    /// Processes batchSize independent inputs through the same weight matrix.
    /// </summary>
    /// <param name="weights">Weight matrix (outNeuron * inNeuron).</param>
    /// <param name="inputs">Batch of inputs (batchSize * inNeuron, row-major).</param>
    /// <param name="outputs">Batch of outputs (batchSize * outNeuron, row-major).</param>
    /// <param name="batchSize">Number of inputs in the batch.</param>
    /// <param name="outNeuron">Number of output neurons.</param>
    /// <param name="inNeuron">Number of input neurons.</param>
    public static void BatchMatVecMultiply(
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> inputs,
        Span<float> outputs,
        int batchSize,
        int outNeuron,
        int inNeuron)
    {
        if (weights.Length < outNeuron * inNeuron ||
            inputs.Length < batchSize * inNeuron ||
            outputs.Length < batchSize * outNeuron)
            throw new ArgumentException("Spans too small.");

        for (int batch = 0; batch < batchSize; batch++)
        {
            var inputSlice = inputs.Slice(batch * inNeuron, inNeuron);
            var outputSlice = outputs.Slice(batch * outNeuron, outNeuron);
            MatVecMultiply(weights, inputSlice, outputSlice, outNeuron, inNeuron);
        }
    }

    #endregion

    #region Activation Functions

    /// <summary>
    /// Applies ReLU activation in-place to a span of values using SIMD.
    /// ReLU(x) = max(0, x).
    /// </summary>
    /// <param name="values">The values to apply ReLU to (modified in-place).</param>
    public static void ApplyReluInPlace(Span<float> values)
    {
        int i = 0;

        if (Avx2.IsSupported)
        {
            var zero = Vector256<float>.Zero;
            int simdEnd = values.Length - (values.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                Vector256.StoreUnsafe(Avx.Max(v, zero), ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            var zero = Vector128<float>.Zero;
            int simdEnd = values.Length - (values.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                Vector128.StoreUnsafe(Sse.Max(v, zero), ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }

        for (; i < values.Length; i++)
            values[i] = MathF.Max(0, values[i]);
    }

    /// <summary>
    /// Applies LeakyReLU activation in-place using SIMD.
    /// LeakyReLU(x) = x >= 0 ? x : alpha * x.
    /// </summary>
    /// <param name="values">The values to modify.</param>
    /// <param name="alpha">Slope for negative values (default 0.01).</param>
    public static void ApplyLeakyReluInPlace(Span<float> values, float alpha = 0.01f)
    {
        int i = 0;

        if (Avx2.IsSupported)
        {
            var zero = Vector256<float>.Zero;
            var aVec = Vector256.Create(alpha);
            int simdEnd = values.Length - (values.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                var mask = Avx.Compare(v, zero, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                var neg = Avx.And(Avx.Multiply(v, aVec), Avx.Xor(mask, Vector256.Create(-1f)));
                var pos = Avx.And(v, mask);
                Vector256.StoreUnsafe(Avx.Or(pos, neg), ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            var zero = Vector128<float>.Zero;
            var aVec = Vector128.Create(alpha);
            int simdEnd = values.Length - (values.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                var mask = Sse.CompareGreaterThan(v, zero);
                var neg = Sse.And(Sse.Multiply(v, aVec), Sse.Xor(mask, Vector128.Create(-1f)));
                var pos = Sse.And(v, mask);
                Vector128.StoreUnsafe(Sse.Or(pos, neg), ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }

        for (; i < values.Length; i++)
            values[i] = values[i] > 0 ? values[i] : alpha * values[i];
    }

    /// <summary>
    /// Applies sigmoid activation in-place using fast SIMD approximation.
    /// Sigmoid(x) = 1 / (1 + exp(-x)).
    /// </summary>
    public static void ApplySigmoidInPlace(Span<float> values)
    {
        int i = 0;

        if (Avx2.IsSupported)
        {
            var one = Vector256.Create(1.0f);
            var negOne = Vector256.Create(-1.0f);
            var four = Vector256.Create(4.0f);
            var negFour = Vector256.Create(-4.0f);
            var half = Vector256.Create(0.5f);
            var q1 = Vector256.Create(0.25f);
            var q2 = Vector256.Create(0.020833333f);
            var zero256 = Vector256<float>.Zero;

            int simdEnd = values.Length - (values.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                var x2 = Avx.Multiply(v, v);
                var poly = Avx.Add(half, Avx.Multiply(v, Avx.Subtract(q1, Avx.Multiply(x2, q2))));

                var lowMask = Avx.Compare(v, negFour, FloatComparisonMode.OrderedLessThanNonSignaling);
                var highMask = Avx.Compare(v, four, FloatComparisonMode.OrderedGreaterThanNonSignaling);

                var result = Avx.BlendVariable(poly, zero256, lowMask);
                result = Avx.BlendVariable(result, one, highMask);
                Vector256.StoreUnsafe(result, ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }

        for (; i < values.Length; i++)
            values[i] = MathFunctions.ScalarFastSigmoid(values[i]);
    }

    /// <summary>
    /// Applies tanh activation in-place using fast SIMD approximation.
    /// </summary>
    public static void ApplyTanhInPlace(Span<float> values)
    {
        int i = 0;

        if (Avx2.IsSupported)
        {
            var one = Vector256.Create(1.0f);
            var negOne = Vector256.Create(-1.0f);
            var four = Vector256.Create(4.0f);
            var negFour = Vector256.Create(-4.0f);
            var numC0 = Vector256.Create(135135.0f);
            var numC1 = Vector256.Create(17325.0f);
            var numC2 = Vector256.Create(378.0f);
            var numC3 = Vector256.Create(1.0f);
            var denC0 = Vector256.Create(135135.0f);
            var denC1 = Vector256.Create(62370.0f);
            var denC2 = Vector256.Create(3150.0f);
            var denC3 = Vector256.Create(28.0f);

            int simdEnd = values.Length - (values.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                var absV = Avx.And(v, Vector256.Create(int.MaxValue).AsSingle());
                var sign = Avx.Compare(v, Vector256<float>.Zero, FloatComparisonMode.OrderedLessThanNonSignaling);
                var signMask = Avx.BlendVariable(one, negOne, sign);

                var x2 = Avx.Multiply(absV, absV);
                var num = Avx.Multiply(absV, Avx.Add(numC0, Avx.Multiply(x2, Avx.Add(numC1, Avx.Multiply(x2, Avx.Add(numC2, Avx.Multiply(x2, numC3)))))));
                var den = Avx.Add(denC0, Avx.Multiply(x2, Avx.Add(denC1, Avx.Multiply(x2, Avx.Add(denC2, Avx.Multiply(x2, denC3))))));

                var result = Avx.Divide(num, den);
                result = Avx.Min(result, one);
                result = Avx.Max(result, negOne);
                result = Avx.Multiply(result, signMask);

                Vector256.StoreUnsafe(result, ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }

        for (; i < values.Length; i++)
            values[i] = MathFunctions.ScalarFastTanh(values[i]);
    }

    /// <summary>
    /// Applies SiLU (Swish) activation in-place: x * sigmoid(x).
    /// </summary>
    public static void ApplySiluInPlace(Span<float> values)
    {
        int i = 0;
        if (Avx2.IsSupported)
        {
            var one = Vector256.Create(1.0f);
            var half = Vector256.Create(0.5f);
            var q1 = Vector256.Create(0.25f);
            var q2 = Vector256.Create(0.020833333f);
            var four = Vector256.Create(4.0f);
            var negFour = Vector256.Create(-4.0f);
            var zero256 = Vector256<float>.Zero;

            int simdEnd = values.Length - (values.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                var x2 = Avx.Multiply(v, v);
                var poly = Avx.Add(half, Avx.Multiply(v, Avx.Subtract(q1, Avx.Multiply(x2, q2))));
                var lowMask = Avx.Compare(v, negFour, FloatComparisonMode.OrderedLessThanNonSignaling);
                var highMask = Avx.Compare(v, four, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                var sig = Avx.BlendVariable(poly, zero256, lowMask);
                sig = Avx.BlendVariable(sig, one, highMask);
                Vector256.StoreUnsafe(Avx.Multiply(v, sig), ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }

        for (; i < values.Length; i++)
            values[i] = values[i] * MathFunctions.ScalarFastSigmoid(values[i]);
    }

    /// <summary>
    /// Applies GELU activation in-place using the tanh approximation.
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    /// </summary>
    public static void ApplyGeluInPlace(Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = MathFunctions.ScalarFastGelu(values[i]);
    }

    /// <summary>
    /// Applies ELU activation in-place: x > 0 ? x : alpha * (exp(x) - 1).
    /// </summary>
    /// <param name="values">The values to modify.</param>
    /// <param name="alpha">Scale for negative values (default 1.0).</param>
    public static void ApplyEluInPlace(Span<float> values, float alpha = 1.0f)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = MathFunctions.FastElu(values[i], alpha);
    }

    /// <summary>
    /// Applies softplus activation in-place: ln(1 + exp(x)).
    /// </summary>
    public static void ApplySoftplusInPlace(Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = MathFunctions.FastSoftplus(values[i]);
    }

    /// <summary>
    /// Applies Mish activation in-place: x * tanh(softplus(x)).
    /// </summary>
    public static void ApplyMishInPlace(Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = MathFunctions.FastMish(values[i]);
    }

    /// <summary>
    /// Applies softsign activation in-place: x / (1 + |x|).
    /// </summary>
    public static void ApplySoftsignInPlace(Span<float> values)
    {
        int i = 0;
        if (Avx2.IsSupported)
        {
            var one = Vector256.Create(1.0f);
            int simdEnd = values.Length - (values.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                var absV = Avx.And(v, Vector256.Create(int.MaxValue).AsSingle());
                var result = Avx.Divide(v, Avx.Add(one, absV));
                Vector256.StoreUnsafe(result, ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }

        for (; i < values.Length; i++)
            values[i] = values[i] / (1.0f + MathF.Abs(values[i]));
    }

    /// <summary>
    /// Applies SELU activation in-place.
    /// SELU(x) = scale * ELU(x, alpha) where scale=1.0507, alpha=1.6733.
    /// </summary>
    public static void ApplySeluInPlace(Span<float> values)
    {
        const float SeluAlpha = 1.6732632423543772f;
        const float SeluScale = 1.0507009873554805f;

        for (int i = 0; i < values.Length; i++)
            values[i] = SeluScale * MathFunctions.FastElu(values[i], SeluAlpha);
    }

    /// <summary>
    /// Applies CELU activation in-place: x > 0 ? x : alpha * (exp(x/alpha) - 1).
    /// </summary>
    /// <param name="values">The values to modify.</param>
    /// <param name="alpha">Scale parameter (default 1.0).</param>
    public static void ApplyCeluInPlace(Span<float> values, float alpha = 1.0f)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = MathFunctions.FastCelu(values[i], alpha);
    }

    /// <summary>
    /// Applies an arbitrary element-wise activation function to a span of values.
    /// </summary>
    /// <param name="values">The values to modify.</param>
    /// <param name="activation">The activation function to apply.</param>
    public static void ApplyActivation(Span<float> values, Func<float, float> activation)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = activation(values[i]);
    }

    #endregion

    #region Softmax

    /// <summary>
    /// Computes the softmax function over a span of values with numerical stability.
    /// Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x))).
    /// </summary>
    /// <param name="values">Input values. Overwritten with softmax output.</param>
    public static void SoftmaxInPlace(Span<float> values)
    {
        if (values.Length == 0) return;
        if (values.Length == 1) { values[0] = 1.0f; return; }

        float maxVal = values[0];
        for (int idx = 1; idx < values.Length; idx++)
            if (values[idx] > maxVal) maxVal = values[idx];

        float sum = 0;
        int i = 0;

        if (Avx2.IsSupported)
        {
            var maxVec = Vector256.Create(maxVal);
            int simdEnd = values.Length - (values.Length % 8);

            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                var shifted = Avx.Subtract(v, maxVec);
                Span<float> expBuffer = stackalloc float[8];
                for (int lane = 0; lane < 8; lane++)
                    expBuffer[lane] = MathF.Exp(shifted.GetElement(lane));
                var exp = Vector256.Create(expBuffer[0], expBuffer[1], expBuffer[2], expBuffer[3],
                    expBuffer[4], expBuffer[5], expBuffer[6], expBuffer[7]);
                Vector256.StoreUnsafe(exp, ref MemoryMarshal.GetReference(values.Slice(i)));
                var perm = Avx.Permute2x128(exp, exp, 0x21);
                var sumVec = Avx.Add(exp, perm);
                var lower = sumVec.GetLower();
                var hadd = Sse3.HorizontalAdd(lower, lower);
                sum += Sse.AddScalar(hadd, Sse.Shuffle(hadd, hadd, 0x01)).ToScalar();
            }
        }

        for (; i < values.Length; i++)
        {
            float e = MathFunctions.ScalarFastExp(values[i] - maxVal);
            values[i] = e;
            sum += e;
        }

        float invSum = MathFunctions.ScalarFastReciprocal(sum);
        i = 0;
        if (Avx2.IsSupported)
        {
            var invSumVec = Vector256.Create(invSum);
            int simdEnd = values.Length - (values.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                Vector256.StoreUnsafe(Avx.Multiply(v, invSumVec), ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }

        for (; i < values.Length; i++)
            values[i] *= invSum;
    }

    /// <summary>
    /// Computes the log-softmax function for numerical stability in loss computation.
    /// LogSoftmax(x_i) = x_i - max(x) - log(sum(exp(x_j - max(x)))).
    /// </summary>
    public static void LogSoftmaxInPlace(Span<float> values)
    {
        if (values.Length == 0) return;

        float maxVal = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > maxVal) maxVal = values[i];

        float sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            float e = MathFunctions.ScalarFastExp(values[i] - maxVal);
            sum += e;
        }

        float logSum = MathFunctions.FastLn(sum);
        for (int i = 0; i < values.Length; i++)
            values[i] = values[i] - maxVal - logSum;
    }

    /// <summary>
    /// Batch softmax computation for multiple independent vectors.
    /// </summary>
    /// <param name="data">Flattened data (rowCount * length, row-major).</param>
    /// <param name="length">Number of elements per row (vocabulary size, etc.).</param>
    /// <param name="rowCount">Number of independent softmax computations.</param>
    public static void BatchSoftmax(Span<float> data, int length, int rowCount)
    {
        for (int r = 0; r < rowCount; r++)
            SoftmaxInPlace(data.Slice(r * length, length));
    }

    #endregion

    #region Layer Normalization

    /// <summary>
    /// Applies layer normalization in-place.
    /// LN(x) = gamma * (x - mean) / sqrt(variance + eps) + beta.
    /// </summary>
    /// <param name="values">The values to normalize (modified in-place).</param>
    /// <param name="gamma">Scale parameters (per-element).</param>
    /// <param name="beta">Bias parameters (per-element).</param>
    /// <param name="epsilon">Numerical stability constant.</param>
    public static void LayerNormInPlace(
        Span<float> values,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        float epsilon = 1e-5f)
    {
        int length = values.Length;
        if (gamma.Length != length || beta.Length != length)
            throw new ArgumentException("gamma and beta must match the length of values.");

        float mean = 0;
        for (int idx = 0; idx < length; idx++)
            mean += values[idx];
        mean /= length;

        float variance = 0;
        for (int idx = 0; idx < length; idx++)
        {
            float diff = values[idx] - mean;
            variance += diff * diff;
        }
        variance /= length;

        float invStd = MathFunctions.ScalarFastRsqrt(variance + epsilon);

        int i = 0;
        if (Avx2.IsSupported)
        {
            var meanVec = Vector256.Create(mean);
            var invStdVec = Vector256.Create(invStd);
            int simdEnd = length - (length % 8);

            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                var g = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(gamma.Slice(i)));
                var b = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(beta.Slice(i)));
                var normalized = Avx.Multiply(Avx.Subtract(v, meanVec), invStdVec);
                var result = Avx.Add(Avx.Multiply(normalized, g), b);
                Vector256.StoreUnsafe(result, ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            var meanVec = Vector128.Create(mean);
            var invStdVec = Vector128.Create(invStd);
            int simdEnd = length - (length % 4);

            for (; i < simdEnd; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                var g = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(gamma.Slice(i)));
                var b = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(beta.Slice(i)));
                var normalized = Sse.Multiply(Sse.Subtract(v, meanVec), invStdVec);
                var result = Sse.Add(Sse.Multiply(normalized, g), b);
                Vector128.StoreUnsafe(result, ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }

        for (; i < length; i++)
        {
            float normalized = (values[i] - mean) * invStd;
            values[i] = normalized * gamma[i] + beta[i];
        }
    }

    /// <summary>
    /// Applies RMS normalization (root mean square normalization).
    /// RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps).
    /// </summary>
    public static void RMSNormInPlace(
        Span<float> values,
        ReadOnlySpan<float> gamma,
        float epsilon = 1e-6f)
    {
        int length = values.Length;
        if (gamma.Length != length)
            throw new ArgumentException("gamma must match the length of values.");

        float sumSq = 0;
        for (int idx = 0; idx < length; idx++)
            sumSq += values[idx] * values[idx];

        float rms = MathFunctions.ScalarFastRsqrt(sumSq / length + epsilon);

        int i = 0;
        if (Avx2.IsSupported)
        {
            var rmsVec = Vector256.Create(rms);
            int simdEnd = length - (length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                var g = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(gamma.Slice(i)));
                var result = Avx.Multiply(Avx.Multiply(v, rmsVec), g);
                Vector256.StoreUnsafe(result, ref MemoryMarshal.GetReference(values.Slice(i)));
            }
        }

        for (; i < length; i++)
            values[i] = values[i] * rms * gamma[i];
    }

    #endregion

    #region Batch Weight Compression/Decompression

    /// <summary>
    /// Quantizes float32 weights to int8 with scale/zero-point for memory-efficient storage.
    /// </summary>
    /// <param name="weights">Input float32 weights.</param>
    /// <param name="quantized">Output int8 quantized weights.</param>
    /// <param name="scale">Output quantization scale factor.</param>
    /// <param name="zeroPoint">Output quantization zero point.</param>
    public static void QuantizeInt8(
        ReadOnlySpan<float> weights,
        Span<sbyte> quantized,
        out float scale,
        out sbyte zeroPoint)
    {
        if (weights.Length != quantized.Length)
            throw new ArgumentException("Weights and quantized spans must have the same length.");

        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < weights.Length; i++)
        {
            if (weights[i] < min) min = weights[i];
            if (weights[i] > max) max = weights[i];
        }

        float range = max - min;
        if (range < 1e-10f) range = 1.0f;
        scale = range / 255.0f;
        zeroPoint = (sbyte)Math.Clamp((int)MathF.Round(-min / scale), -128, 127);

        for (int i = 0; i < weights.Length; i++)
        {
            quantized[i] = (sbyte)Math.Clamp((int)MathF.Round(weights[i] / scale + zeroPoint), -128, 127);
        }
    }

    /// <summary>
    /// Dequantizes int8 weights back to float32.
    /// </summary>
    public static void DequantizeInt8(
        ReadOnlySpan<sbyte> quantized,
        Span<float> weights,
        float scale,
        sbyte zeroPoint)
    {
        if (quantized.Length != weights.Length)
            throw new ArgumentException("Quantized and weights spans must have the same length.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            var scaleVec = Vector256.Create(scale);
            var zpVec = Vector256.Create((float)zeroPoint);
            int simdEnd = quantized.Length - (quantized.Length % 32);

            for (; i < simdEnd; i += 32)
            {
                ReadOnlySpan<byte> qBytesSpan = MemoryMarshal.Cast<sbyte, byte>(quantized.Slice(i, 32));
                var qBytes = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(qBytesSpan));

                var q16_lo = Avx2.ConvertToVector256Int16(qBytes.GetLower().AsSByte());
                var q16_hi = Avx2.ConvertToVector256Int16(qBytes.GetUpper().AsSByte());

                var q32_0 = Avx2.ConvertToVector256Int32(q16_lo.GetLower().AsInt16());
                var q32_1 = Avx2.ConvertToVector256Int32(q16_lo.GetUpper().AsInt16());
                var q32_2 = Avx2.ConvertToVector256Int32(q16_hi.GetLower().AsInt16());
                var q32_3 = Avx2.ConvertToVector256Int32(q16_hi.GetUpper().AsInt16());

                var f0 = Avx.Multiply(Avx.Subtract(Avx.ConvertToVector256Single(q32_0), zpVec), scaleVec);
                var f1 = Avx.Multiply(Avx.Subtract(Avx.ConvertToVector256Single(q32_1), zpVec), scaleVec);
                var f2 = Avx.Multiply(Avx.Subtract(Avx.ConvertToVector256Single(q32_2), zpVec), scaleVec);
                var f3 = Avx.Multiply(Avx.Subtract(Avx.ConvertToVector256Single(q32_3), zpVec), scaleVec);

                Vector256.StoreUnsafe(f0, ref MemoryMarshal.GetReference(weights.Slice(i)));
                Vector256.StoreUnsafe(f1, ref MemoryMarshal.GetReference(weights.Slice(i + 8)));
                Vector256.StoreUnsafe(f2, ref MemoryMarshal.GetReference(weights.Slice(i + 16)));
                Vector256.StoreUnsafe(f3, ref MemoryMarshal.GetReference(weights.Slice(i + 24)));
            }
        }

        for (; i < quantized.Length; i++)
            weights[i] = (quantized[i] - zeroPoint) * scale;
    }

    /// <summary>
    /// Compresses float32 weights using block floating-point quantization.
    /// Groups of 8 floats share a common 8-bit exponent, storing only mantissas.
    /// </summary>
    /// <param name="weights">Input float32 weights.</param>
    /// <param name="compressed">Output compressed data (shared exponents + mantissas).</param>
    /// <param name="blockSize">Number of floats per block (default 8).</param>
    public static void BlockCompress(
        ReadOnlySpan<float> weights,
        Span<byte> compressed,
        int blockSize = 8)
    {
        int blockCount = (weights.Length + blockSize - 1) / blockSize;
        int requiredBytes = blockCount * (1 + blockSize);
        if (compressed.Length < requiredBytes)
            throw new ArgumentException($"Output span too small. Need {requiredBytes} bytes.");

        int cIdx = 0;
        for (int block = 0; block < blockCount; block++)
        {
            int wStart = block * blockSize;
            int wEnd = Math.Min(wStart + blockSize, weights.Length);
            int count = wEnd - wStart;

            float maxAbs = 0;
            for (int i = wStart; i < wEnd; i++)
            {
                float abs = MathF.Abs(weights[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            int sharedExp = maxAbs > 0 ? (int)MathF.Floor(MathF.Log2(maxAbs)) : 0;
            sharedExp = Math.Clamp(sharedExp, -128, 127);
            compressed[cIdx++] = (byte)(sharedExp + 128);

            float scale = maxAbs > 0 ? MathF.Pow(2, sharedExp) / 127.0f : 0;

            for (int i = 0; i < blockSize; i++)
            {
                if (i < count)
                {
                    float val = weights[wStart + i];
                    float normalized = scale > 0 ? val / scale : 0;
                    compressed[cIdx++] = (byte)Math.Clamp((int)MathF.Round(normalized * 127 + 128), 0, 255);
                }
                else
                {
                    compressed[cIdx++] = 128;
                }
            }
        }
    }

    /// <summary>
    /// Decompresses block-compressed float32 weights.
    /// </summary>
    public static void BlockDecompress(
        ReadOnlySpan<byte> compressed,
        Span<float> weights,
        int blockSize = 8)
    {
        int blockCount = weights.Length / blockSize;
        if (weights.Length % blockSize != 0) blockCount++;

        int cIdx = 0;
        for (int block = 0; block < blockCount; block++)
        {
            int wStart = block * blockSize;
            int wEnd = Math.Min(wStart + blockSize, weights.Length);
            int count = wEnd - wStart;

            int sharedExp = compressed[cIdx++] - 128;
            float scale = MathF.Pow(2, sharedExp) / 127.0f;

            for (int i = 0; i < count; i++)
            {
                float normalized = (compressed[cIdx++] - 128) / 127.0f;
                weights[wStart + i] = normalized * scale;
            }

            cIdx += blockSize - count;
        }
    }

    #endregion

    #region Memory Copy with SIMD Alignment

    /// <summary>
    /// Copies a span of floats using SIMD-optimized block copy.
    /// Handles alignment and uses the widest available SIMD width.
    /// </summary>
    /// <param name="source">Source span.</param>
    /// <param name="destination">Destination span.</param>
    public static void SimdCopy(ReadOnlySpan<float> source, Span<float> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Source and destination must have the same length.");

        int i = 0;

        if (Avx2.IsSupported)
        {
            int simdEnd = source.Length - (source.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(source.Slice(i)));
                Vector256.StoreUnsafe(v, ref MemoryMarshal.GetReference(destination.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = source.Length - (source.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(source.Slice(i)));
                Vector128.StoreUnsafe(v, ref MemoryMarshal.GetReference(destination.Slice(i)));
            }
        }

        for (; i < source.Length; i++)
            destination[i] = source[i];
    }

    /// <summary>
    /// Copies a span of bytes using SIMD-optimized block copy.
    /// </summary>
    public static void SimdCopyBytes(ReadOnlySpan<byte> source, Span<byte> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Source and destination must have the same length.");

        source.CopyTo(destination);
    }

    /// <summary>
    /// Fills a span with a specified value using SIMD.
    /// </summary>
    /// <param name="destination">The span to fill.</param>
    /// <param name="value">The value to fill with.</param>
    public static void SimdFill(Span<float> destination, float value)
    {
        int i = 0;

        if (Avx2.IsSupported)
        {
            var v = Vector256.Create(value);
            int simdEnd = destination.Length - (destination.Length % 8);
            for (; i < simdEnd; i += 8)
                Vector256.StoreUnsafe(v, ref MemoryMarshal.GetReference(destination.Slice(i)));
        }
        else if (Sse.IsSupported)
        {
            var v = Vector128.Create(value);
            int simdEnd = destination.Length - (destination.Length % 4);
            for (; i < simdEnd; i += 4)
                Vector128.StoreUnsafe(v, ref MemoryMarshal.GetReference(destination.Slice(i)));
        }

        for (; i < destination.Length; i++)
            destination[i] = value;
    }

    /// <summary>
    /// Adds two float arrays element-wise using SIMD and stores the result.
    /// </summary>
    public static void SimdAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("All spans must have the same length.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector256.StoreUnsafe(Avx.Add(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector128.StoreUnsafe(Sse.Add(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < a.Length; i++)
            result[i] = a[i] + b[i];
    }

    /// <summary>
    /// Multiplies two float arrays element-wise using SIMD and stores the result.
    /// </summary>
    public static void SimdMultiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("All spans must have the same length.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector256.StoreUnsafe(Avx.Multiply(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector128.StoreUnsafe(Sse.Multiply(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < a.Length; i++)
            result[i] = a[i] * b[i];
    }

    /// <summary>
    /// Computes element-wise fused multiply-add: result = a * b + c using SIMD.
    /// Uses hardware FMA where available for better precision and performance.
    /// </summary>
    public static void SimdFMA(ReadOnlySpan<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != c.Length || a.Length != result.Length)
            throw new ArgumentException("All spans must have the same length.");

        int i = 0;
        if (Fma.IsSupported && Avx2.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                var vc = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(c.Slice(i)));
                Vector256.StoreUnsafe(Fma.MultiplyAdd(va, vb, vc), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Avx2.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                var vc = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(c.Slice(i)));
                Vector256.StoreUnsafe(Avx.Add(Avx.Multiply(va, vb), vc), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                var vc = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(c.Slice(i)));
                Vector128.StoreUnsafe(Sse.Add(Sse.Multiply(va, vb), vc), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < a.Length; i++)
            result[i] = a[i] * b[i] + c[i];
    }

    #endregion

    #region ArgMax and Top-K

    /// <summary>
    /// Finds the index of the maximum value in a span (argmax).
    /// </summary>
    /// <param name="values">The input values.</param>
    /// <returns>The index of the maximum value.</returns>
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        if (values.Length == 0) return -1;

        int maxIdx = 0;
        float maxVal = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > maxVal)
            {
                maxVal = values[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /// <summary>
    /// Finds the index of the minimum value in a span (argmin).
    /// </summary>
    public static int ArgMin(ReadOnlySpan<float> values)
    {
        if (values.Length == 0) return -1;

        int minIdx = 0;
        float minVal = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] < minVal)
            {
                minVal = values[i];
                minIdx = i;
            }
        }

        return minIdx;
    }

    /// <summary>
    /// Finds the top-K indices and values (descending order) in a span.
    /// </summary>
    /// <param name="values">The input values.</param>
    /// <param name="k">Number of top elements to find.</param>
    /// <param name="indices">Output indices (must be at least k).</param>
    /// <param name="topValues">Output values (must be at least k).</param>
    public static void TopK(ReadOnlySpan<float> values, int k, Span<int> indices, Span<float> topValues)
    {
        if (k > values.Length) k = values.Length;
        if (indices.Length < k || topValues.Length < k)
            throw new ArgumentException("Output spans too small for specified k.");

        var tempIndices = new int[values.Length];
        var tempValues = new float[values.Length];
        values.CopyTo(tempValues);
        for (int i = 0; i < tempValues.Length; i++) tempIndices[i] = i;

        for (int i = 0; i < k; i++)
        {
            int maxIdx = i;
            for (int j = i + 1; j < tempValues.Length; j++)
            {
                if (tempValues[j] > tempValues[maxIdx])
                    maxIdx = j;
            }

            (tempValues[i], tempValues[maxIdx]) = (tempValues[maxIdx], tempValues[i]);
            (tempIndices[i], tempIndices[maxIdx]) = (tempIndices[maxIdx], tempIndices[i]);
        }

        for (int i = 0; i < k; i++)
        {
            indices[i] = tempIndices[i];
            topValues[i] = tempValues[i];
        }
    }

    #endregion

    #region Dropout

    /// <summary>
    /// Applies dropout to a span of values in-place during training.
    /// Zeros out elements where the corresponding mask bit is set.
    /// </summary>
    /// <param name="values">Values to apply dropout to.</param>
    /// <param name="mask">Boolean mask (true = keep, false = drop).</param>
    /// <param name="scale">Scale factor (1 / (1 - dropout_rate)).</param>
    public static void DropoutInPlace(Span<float> values, ReadOnlySpan<bool> mask, float scale)
    {
        if (values.Length != mask.Length)
            throw new ArgumentException("Values and mask must have the same length.");

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = mask[i] ? values[i] * scale : 0;
        }
    }

    /// <summary>
    /// Applies inverted dropout to a span of values in-place.
    /// </summary>
    /// <param name="values">Values to modify.</param>
    /// <param name="dropoutRate">Dropout probability (0.0 to 1.0).</param>
    /// <param name="rng">Random number generator.</param>
    public static void DropoutInPlace(Span<float> values, float dropoutRate, Random rng)
    {
        if (dropoutRate <= 0) return;
        float scale = 1.0f / (1.0f - dropoutRate);

        for (int i = 0; i < values.Length; i++)
        {
            if (rng.NextDouble() < dropoutRate)
                values[i] = 0;
            else
                values[i] *= scale;
        }
    }

    #endregion

    #region Gradient Clipping

    /// <summary>
    /// Clips gradients by global norm to prevent exploding gradients.
    /// </summary>
    /// <param name="gradients">The gradients to clip (modified in-place).</param>
    /// <param name="maxNorm">The maximum allowed global norm.</param>
    /// <returns>The original global norm before clipping.</returns>
    public static float ClipGradientsByNorm(Span<float> gradients, float maxNorm)
    {
        float sumSq = 0;
        for (int idx = 0; idx < gradients.Length; idx++)
            sumSq += gradients[idx] * gradients[idx];

        float globalNorm = MathF.Sqrt(sumSq);
        if (globalNorm <= maxNorm) return globalNorm;

        float scale = maxNorm / globalNorm;
        int i = 0;
        if (Avx2.IsSupported)
        {
            var scaleVec = Vector256.Create(scale);
            int simdEnd = gradients.Length - (gradients.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var g = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(gradients.Slice(i)));
                Vector256.StoreUnsafe(Avx.Multiply(g, scaleVec), ref MemoryMarshal.GetReference(gradients.Slice(i)));
            }
        }

        for (; i < gradients.Length; i++)
            gradients[i] *= scale;

        return globalNorm;
    }

    /// <summary>
    /// Clips gradients element-wise to a maximum absolute value.
    /// </summary>
    public static void ClipGradientsByValue(Span<float> gradients, float maxValue)
    {
        int i = 0;
        if (Avx2.IsSupported)
        {
            var maxVec = Vector256.Create(maxValue);
            var negMaxVec = Vector256.Create(-maxValue);
            int simdEnd = gradients.Length - (gradients.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var g = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(gradients.Slice(i)));
                g = Avx.Min(g, maxVec);
                g = Avx.Max(g, negMaxVec);
                Vector256.StoreUnsafe(g, ref MemoryMarshal.GetReference(gradients.Slice(i)));
            }
        }

        for (; i < gradients.Length; i++)
            gradients[i] = MathF.Max(-maxValue, MathF.Min(maxValue, gradients[i]));
    }

    #endregion

    #region Weight Decay / L2 Regularization

    /// <summary>
    /// Applies L2 weight decay: weight -= lr * lambda * weight.
    /// </summary>
    /// <param name="weights">Weights to update (modified in-place).</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <param name="lambda">L2 regularization coefficient.</param>
    public static void ApplyWeightDecay(Span<float> weights, float learningRate, float lambda)
    {
        float factor = 1.0f - learningRate * lambda;

        int i = 0;
        if (Avx2.IsSupported)
        {
            var fVec = Vector256.Create(factor);
            int simdEnd = weights.Length - (weights.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var w = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(weights.Slice(i)));
                Vector256.StoreUnsafe(Avx.Multiply(w, fVec), ref MemoryMarshal.GetReference(weights.Slice(i)));
            }
        }

        for (; i < weights.Length; i++)
            weights[i] *= factor;
    }

    #endregion

    #region Layer Forward Pass (Combined)

    /// <summary>
    /// Executes a complete dense layer forward pass: output = activation(W * input + bias).
    /// </summary>
    /// <param name="weights">Layer weights (outNeuron * inNeuron).</param>
    /// <param name="bias">Layer bias (outNeuron).</param>
    /// <param name="input">Layer input (inNeuron).</param>
    /// <param name="output">Layer output (outNeuron).</param>
    /// <param name="activationType">The activation function to apply.</param>
    public static void DenseLayerForward(
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias,
        ReadOnlySpan<float> input,
        Span<float> output,
        ActivationType activationType)
    {
        MatVecMultiplyBias(weights, input, bias, output, bias.Length, input.Length);
        ApplyActivationByType(output, activationType);
    }

    /// <summary>
    /// Applies an activation function to a span based on an enum type.
    /// </summary>
    /// <param name="values">Values to activate.</param>
    /// <param name="type">The activation function type.</param>
    public static void ApplyActivationByType(Span<float> values, ActivationType type)
    {
        switch (type)
        {
            case ActivationType.None:
                break;
            case ActivationType.Relu:
                ApplyReluInPlace(values);
                break;
            case ActivationType.LeakyRelu:
                ApplyLeakyReluInPlace(values);
                break;
            case ActivationType.Sigmoid:
                ApplySigmoidInPlace(values);
                break;
            case ActivationType.Tanh:
                ApplyTanhInPlace(values);
                break;
            case ActivationType.Silu:
                ApplySiluInPlace(values);
                break;
            case ActivationType.Gelu:
                ApplyGeluInPlace(values);
                break;
            case ActivationType.Elu:
                ApplyEluInPlace(values);
                break;
            case ActivationType.Selu:
                ApplySeluInPlace(values);
                break;
            case ActivationType.Softplus:
                ApplySoftplusInPlace(values);
                break;
            case ActivationType.Mish:
                ApplyMishInPlace(values);
                break;
            case ActivationType.Softsign:
                ApplySoftsignInPlace(values);
                break;
            case ActivationType.Celu:
                ApplyCeluInPlace(values);
                break;
            default:
                break;
        }
    }

    /// <summary>
    /// Enumerates supported activation function types.
    /// </summary>
    public enum ActivationType
    {
        None = 0,
        Relu = 1,
        LeakyRelu = 2,
        Sigmoid = 3,
        Tanh = 4,
        Silu = 5,
        Gelu = 6,
        Elu = 7,
        Selu = 8,
        Softplus = 9,
        Mish = 10,
        Softsign = 11,
        Celu = 12
    }

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Computes the sum of all elements in a span using SIMD reduction.
    /// </summary>
    public static float ReduceSum(ReadOnlySpan<float> values)
    {
        float sum = 0;
        int i = 0;

        if (Avx2.IsSupported)
        {
            var sumVec = Vector256<float>.Zero;
            int simdEnd = values.Length - (values.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                sumVec = Avx.Add(sumVec, v);
            }

            var upper = Avx.ExtractVector128(sumVec, 1);
            var lower = sumVec.GetLower();
            var sum128 = Sse.Add(upper, lower);
            sum128 = Sse.Add(sum128, Sse.MoveHighToLow(sum128, sum128));
            sum128 = Sse.AddScalar(sum128, Sse.Shuffle(sum128, sum128, 0x55));
            sum = sum128.ToScalar();
        }
        else if (Sse.IsSupported)
        {
            var sumVec = Vector128<float>.Zero;
            int simdEnd = values.Length - (values.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(values.Slice(i)));
                sumVec = Sse.Add(sumVec, v);
            }
            sumVec = Sse.Add(sumVec, Sse.MoveHighToLow(sumVec, sumVec));
            sumVec = Sse.AddScalar(sumVec, Sse.Shuffle(sumVec, sumVec, 0x55));
            sum = sumVec.ToScalar();
        }

        for (; i < values.Length; i++)
            sum += values[i];

        return sum;
    }

    /// <summary>
    /// Computes the sum of squared elements in a span.
    /// </summary>
    public static float ReduceSumOfSquares(ReadOnlySpan<float> values)
    {
        float sum = 0;
        for (int i = 0; i < values.Length; i++)
            sum += values[i] * values[i];
        return sum;
    }

    /// <summary>
    /// Computes the mean of all elements in a span.
    /// </summary>
    public static float ReduceMean(ReadOnlySpan<float> values)
    {
        if (values.Length == 0) return 0;
        return ReduceSum(values) / values.Length;
    }

    /// <summary>
    /// Computes the variance of all elements in a span.
    /// </summary>
    public static float ReduceVariance(ReadOnlySpan<float> values)
    {
        if (values.Length == 0) return 0;
        float mean = ReduceMean(values);
        float sumSq = 0;
        for (int i = 0; i < values.Length; i++)
        {
            float diff = values[i] - mean;
            sumSq += diff * diff;
        }
        return sumSq / values.Length;
    }

    /// <summary>
    /// Computes the index of the maximum element (argmax) using SIMD.
    /// </summary>
    public static int ReduceArgMax(ReadOnlySpan<float> values)
    {
        if (values.Length == 0) return -1;

        int maxIdx = 0;
        float maxVal = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > maxVal)
            {
                maxVal = values[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    #endregion

    #region Element-wise Arithmetic

    /// <summary>
    /// Subtracts two float arrays element-wise: result = a - b.
    /// </summary>
    public static void SimdSubtract(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("All spans must have the same length.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector256.StoreUnsafe(Avx.Subtract(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector128.StoreUnsafe(Sse.Subtract(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < a.Length; i++)
            result[i] = a[i] - b[i];
    }

    /// <summary>
    /// Scales a float array and adds a bias element-wise: result = value * scale + bias.
    /// </summary>
    public static void SimdScaleAndBias(
        ReadOnlySpan<float> value, float scale, float bias, Span<float> result)
    {
        if (value.Length != result.Length)
            throw new ArgumentException("Value and result spans must have the same length.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            var sVec = Vector256.Create(scale);
            var bVec = Vector256.Create(bias);
            int simdEnd = value.Length - (value.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(value.Slice(i)));
                Vector256.StoreUnsafe(Avx.Add(Avx.Multiply(v, sVec), bVec), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            var sVec = Vector128.Create(scale);
            var bVec = Vector128.Create(bias);
            int simdEnd = value.Length - (value.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(value.Slice(i)));
                Vector128.StoreUnsafe(Sse.Add(Sse.Multiply(v, sVec), bVec), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < value.Length; i++)
            result[i] = value[i] * scale + bias;
    }

    /// <summary>
    /// Computes element-wise maximum of two arrays: result = max(a, b).
    /// </summary>
    public static void SimdMax(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("All spans must have the same length.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector256.StoreUnsafe(Avx.Max(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector128.StoreUnsafe(Sse.Max(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < a.Length; i++)
            result[i] = MathF.Max(a[i], b[i]);
    }

    /// <summary>
    /// Computes element-wise minimum of two arrays: result = min(a, b).
    /// </summary>
    public static void SimdMin(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("All spans must have the same length.");

        int i = 0;
        if (Avx2.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 8);
            for (; i < simdEnd; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector256.StoreUnsafe(Avx.Min(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }
        else if (Sse.IsSupported)
        {
            int simdEnd = a.Length - (a.Length % 4);
            for (; i < simdEnd; i += 4)
            {
                var va = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a.Slice(i)));
                var vb = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b.Slice(i)));
                Vector128.StoreUnsafe(Sse.Min(va, vb), ref MemoryMarshal.GetReference(result.Slice(i)));
            }
        }

        for (; i < a.Length; i++)
            result[i] = MathF.Min(a[i], b[i]);
    }

    #endregion

    #region Private Helpers

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalAdd256(Vector256<float> v)
    {
        var upper = Avx.ExtractVector128(v, 1);
        var lower = v.GetLower();
        var sum128 = Sse.Add(upper, lower);
        sum128 = Sse.Add(sum128, Sse.MoveHighToLow(sum128, sum128));
        sum128 = Sse.AddScalar(sum128, Sse.Shuffle(sum128, sum128, 0x55));
        return sum128.ToScalar();
    }

    #endregion
}
