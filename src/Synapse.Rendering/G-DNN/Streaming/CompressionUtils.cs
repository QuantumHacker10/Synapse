using System;
// ============================================================
// FILE: CompressionUtils.cs
// PATH: Streaming/CompressionUtils.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers;
using System.Buffers.Binary;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics;
using System.IO;
using System.IO;
using System.IO.Compression;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks;

namespace GDNN.Streaming
{
    /// <summary>
    /// Supported compression algorithms.
    /// </summary>
    [Flags]
    public enum CompressionAlgorithm : byte
    {
        /// <summary>No compression.</summary>
        None = 0,

        /// <summary>LZ4 fast compression.</summary>
        LZ4 = 1,

        /// <summary>Zstandard compression.</summary>
        Zstd = 2,

        /// <summary>Brotli compression.</summary>
        Brotli = 4,

        /// <summary>Deflate/ZIP compression.</summary>
        Deflate = 8,

        /// <summary>GZip compression.</summary>
        GZip = 16,

        /// <summary>Auto-select best algorithm.</summary>
        Auto = 128
    }

    /// <summary>
    /// Quantization format for floating-point weight compression.
    /// </summary>
    public enum QuantizationFormat : byte
    {
        /// <summary>No quantization, full FP32.</summary>
        FP32 = 0,

        /// <summary>Half precision FP16.</summary>
        FP16 = 1,

        /// <summary>8-bit floating point (E5M2).</summary>
        FP8_E5M2 = 2,

        /// <summary>8-bit floating point (E4M3).</summary>
        FP8_E4M3 = 3,

        /// <summary>8-bit unsigned integer (scaled).</summary>
        UINT8 = 4,

        /// <summary>4-bit integer (two per byte).</summary>
        INT4 = 5
    }

    /// <summary>
    /// Configuration for compression operations.
    /// </summary>
    public sealed class CompressionConfig
    {
        /// <summary>Default compression algorithm.</summary>
        public CompressionAlgorithm Algorithm { get; set; } = CompressionAlgorithm.LZ4;

        /// <summary>Compression level (1-22 depending on algorithm).</summary>
        public int Level { get; set; } = 6;

        /// <summary>Quantization format for weight compression.</summary>
        public QuantizationFormat Quantization { get; set; } = QuantizationFormat.FP16;

        /// <summary>Whether to enable delta encoding for temporal sequences.</summary>
        public bool EnableDeltaEncoding { get; set; } = true;

        /// <summary>Whether to enable dictionary-based weight sharing.</summary>
        public bool EnableDictionarySharing { get; set; } = true;

        /// <summary>Whether to enable adaptive compression level selection.</summary>
        public bool EnableAdaptiveLevel { get; set; } = true;

        /// <summary>Block size for streaming compression in bytes.</summary>
        public int StreamingBlockSize { get; set; } = 64 * 1024;

        /// <summary>Maximum dictionary size for weight sharing.</summary>
        public int MaxDictionarySize { get; set; } = 1024 * 1024;

        /// <summary>Target compression ratio for adaptive level selection.</summary>
        public float TargetCompressionRatio { get; set; } = 3.0f;
    }

    /// <summary>
    /// Result of a compression or decompression operation.
    /// </summary>
    public readonly struct CompressionResult
    {
        /// <summary>Whether the operation succeeded.</summary>
        public readonly bool Success;

        /// <summary>Output data.</summary>
        public readonly byte[] Data;

        /// <summary>Original size in bytes.</summary>
        public readonly int OriginalSize;

        /// <summary>Compressed size in bytes.</summary>
        public readonly int CompressedSize;

        /// <summary>Compression ratio (original / compressed).</summary>
        public readonly double Ratio;

        /// <summary>Algorithm used.</summary>
        public readonly CompressionAlgorithm Algorithm;

        /// <summary>Time taken in milliseconds.</summary>
        public readonly double ElapsedMs;

        /// <summary>Error message if failed.</summary>
        public readonly string? Error;

        public CompressionResult(
            bool success, byte[] data, int originalSize, int compressedSize,
            double ratio, CompressionAlgorithm algorithm, double elapsedMs, string? error = null)
        {
            Success = success;
            Data = data;
            OriginalSize = originalSize;
            CompressedSize = compressedSize;
            Ratio = ratio;
            Algorithm = algorithm;
            ElapsedMs = elapsedMs;
            Error = error;
        }

        /// <summary>Returns a formatted summary.</summary>
        public override string ToString() =>
            Success
                ? $"[{Algorithm}] {OriginalSize} -> {CompressedSize} ({Ratio:F2}x) in {ElapsedMs:F1}ms"
                : $"[{Algorithm}] Failed: {Error}";
    }

    /// <summary>
    /// Provides FP8/FP16 quantization, LZ4/Zstd/Brotli compression wrappers,
    /// delta encoding, dictionary-based weight sharing, adaptive compression,
    /// and streaming decompression support for neural network weights.
    /// </summary>
    public sealed class CompressionUtils
    {
        private readonly CompressionConfig _config;
        private readonly Dictionary<byte[], byte[]> _sharedDictionaries;
        private readonly object _dictLock = new();

        /// <summary>
        /// Initializes a new instance of <see cref="CompressionUtils"/>.
        /// </summary>
        /// <param name="config">Compression configuration.</param>
        public CompressionUtils(CompressionConfig? config = null)
        {
            _config = config ?? new CompressionConfig();
            _sharedDictionaries = new Dictionary<byte[], byte[]>(new ByteArrayComparer());
        }

        /// <summary>
        /// Compresses a byte array using the configured algorithm.
        /// </summary>
        /// <param name="data">Data to compress.</param>
        /// <param name="algorithm">Algorithm override, or null to use configured default.</param>
        /// <returns>Compression result with output data and metrics.</returns>
        public unsafe CompressionResult Compress(
            ReadOnlySpan<byte> data,
            CompressionAlgorithm? algorithm = null)
        {
            var sw = Stopwatch.StartNew();
            var algo = algorithm ?? _config.Algorithm;

            try
            {
                if (algo == CompressionAlgorithm.Auto)
                {
                    algo = SelectBestAlgorithm(data);
                }

                byte[] output = algo switch
                {
                    CompressionAlgorithm.LZ4 => CompressLZ4(data, _config.Level),
                    CompressionAlgorithm.Zstd => CompressZstd(data, _config.Level),
                    CompressionAlgorithm.Brotli => CompressBrotli(data, _config.Level),
                    CompressionAlgorithm.Deflate => CompressDeflate(data, _config.Level),
                    CompressionAlgorithm.GZip => CompressGZip(data, _config.Level),
                    _ => data.ToArray()
                };

                sw.Stop();
                double ratio = data.Length > 0 ? (double)data.Length / output.Length : 1.0;

                return new CompressionResult(
                    true, output, data.Length, output.Length, ratio, algo, sw.Elapsed.TotalMilliseconds);
            }
            catch (Exception ex)
            {
                sw.Stop();
                return new CompressionResult(
                    false, [], data.Length, 0, 0, algo, sw.Elapsed.TotalMilliseconds, ex.Message);
            }
        }

        /// <summary>
        /// Decompresses data using the specified algorithm.
        /// </summary>
        /// <param name="data">Compressed data.</param>
        /// <param name="algorithm">Algorithm to use for decompression.</param>
        /// <returns>Decompressed data.</returns>
        public byte[] Decompress(ReadOnlySpan<byte> data, CompressionAlgorithm algorithm = CompressionAlgorithm.LZ4)
        {
            return algorithm switch
            {
                CompressionAlgorithm.LZ4 => DecompressLZ4(data),
                CompressionAlgorithm.Zstd => DecompressZstd(data),
                CompressionAlgorithm.Brotli => DecompressBrotli(data),
                CompressionAlgorithm.Deflate => DecompressDeflate(data),
                CompressionAlgorithm.GZip => DecompressGZip(data),
                _ => data.ToArray()
            };
        }

        /// <summary>
        /// Quantizes FP32 weights to the specified format.
        /// </summary>
        /// <param name="weights">Source FP32 weight data.</param>
        /// <param name="format">Target quantization format.</param>
        /// <returns>Quantized weight bytes.</returns>
        public static unsafe byte[] Quantize(ReadOnlySpan<float> weights, QuantizationFormat format)
        {
            return format switch
            {
                QuantizationFormat.FP16 => QuantizeFP16(weights),
                QuantizationFormat.FP8_E5M2 => QuantizeFP8E5M2(weights),
                QuantizationFormat.FP8_E4M3 => QuantizeFP8E4M3(weights),
                QuantizationFormat.UINT8 => QuantizeUINT8(weights),
                QuantizationFormat.INT4 => QuantizeINT4(weights),
                QuantizationFormat.FP32 => MemoryMarshal.AsBytes(weights.ToArray().AsSpan()).ToArray(),
                _ => QuantizeFP16(weights)
            };
        }

        /// <summary>
        /// Dequantizes compressed weights back to FP32.
        /// </summary>
        /// <param name="data">Quantized weight bytes.</param>
        /// <param name="format">Quantization format used.</param>
        /// <param name="count">Number of weights.</param>
        /// <returns>FP32 weight array.</returns>
        public static unsafe float[] Dequantize(ReadOnlySpan<byte> data, QuantizationFormat format, int count)
        {
            return format switch
            {
                QuantizationFormat.FP16 => DequantizeFP16(data, count),
                QuantizationFormat.FP8_E5M2 => DequantizeFP8E5M2(data, count),
                QuantizationFormat.FP8_E4M3 => DequantizeFP8E4M3(data, count),
                QuantizationFormat.UINT8 => DequantizeUINT8(data, count),
                QuantizationFormat.INT4 => DequantizeINT4(data, count),
                QuantizationFormat.FP32 => MemoryMarshal.Cast<byte, float>(data)[..count].ToArray(),
                _ => DequantizeFP16(data, count)
            };
        }

        /// <summary>
        /// Applies delta encoding to a sequence of weight arrays for temporal compression.
        /// </summary>
        /// <param name="frames">Sequence of weight frames.</param>
        /// <returns>Delta-encoded frames.</returns>
        public static List<byte[]> DeltaEncode(List<float[]> frames)
        {
            if (frames.Count == 0)
                return [];

            var result = new List<byte[]>(frames.Count);
            float[]? previous = null;

            foreach (var frame in frames)
            {
                if (previous == null)
                {
                    result.Add(MemoryMarshal.AsBytes(frame.AsSpan()).ToArray());
                }
                else
                {
                    int byteLen = frame.Length * sizeof(float);
                    byte[] delta = new byte[byteLen];
                    Span<float> deltaFloats = MemoryMarshal.Cast<byte, float>(delta.AsSpan());

                    for (int i = 0; i < frame.Length; i++)
                    {
                        deltaFloats[i] = frame[i] - previous[i];
                    }

                    result.Add(delta);
                }

                previous = frame;
            }

            return result;
        }

        /// <summary>
        /// Decodes delta-encoded weight frames back to absolute values.
        /// </summary>
        /// <param name="deltaFrames">Delta-encoded frames.</param>
        /// <returns>Decoded absolute weight frames.</returns>
        public static List<float[]> DeltaDecode(List<byte[]> deltaFrames)
        {
            if (deltaFrames.Count == 0)
                return [];

            var result = new List<float[]>(deltaFrames.Count);
            float[]? previous = null;

            foreach (var deltaBytes in deltaFrames)
            {
                var deltaFloats = MemoryMarshal.Cast<byte, float>(deltaBytes);

                if (previous == null)
                {
                    result.Add(deltaFloats.ToArray());
                }
                else
                {
                    float[] frame = new float[deltaFloats.Length];
                    for (int i = 0; i < frame.Length; i++)
                    {
                        frame[i] = previous[i] + deltaFloats[i];
                    }
                    result.Add(frame);
                }

                previous = result[^1];
            }

            return result;
        }

        /// <summary>
        /// Builds a shared dictionary from reference weight data.
        /// </summary>
        /// <param name="referenceData">Reference data to build dictionary from.</param>
        /// <param name="dictionaryId">Identifier for the dictionary.</param>
        public void BuildSharedDictionary(ReadOnlySpan<byte> referenceData, int dictionaryId)
        {
            if (referenceData.Length > _config.MaxDictionarySize)
                return;

            var dictData = referenceData.ToArray();

            lock (_dictLock)
            {
                var key = BitConverter.GetBytes(dictionaryId);
                _sharedDictionaries[key] = dictData;
            }
        }

        /// <summary>
        /// Compresses data using a shared dictionary.
        /// </summary>
        /// <param name="data">Data to compress.</param>
        /// <param name="dictionaryId">Dictionary identifier.</param>
        /// <returns>Compressed data.</returns>
        public byte[] CompressWithDictionary(ReadOnlySpan<byte> data, int dictionaryId)
        {
            var key = BitConverter.GetBytes(dictionaryId);

            byte[]? dictionary = null;
            lock (_dictLock)
            {
                _sharedDictionaries.TryGetValue(key, out dictionary);
            }

            if (dictionary != null)
            {
                int headerSize = 4;
                byte[] result = new byte[headerSize + data.Length];
                BinaryPrimitives.WriteInt32LittleEndian(result, dictionaryId);
                data.CopyTo(result.AsSpan(headerSize));
                return CompressLZ4(result, _config.Level);
            }

            return Compress(data, CompressionAlgorithm.LZ4).Data;
        }

        /// <summary>
        /// Estimates the compression ratio for the given data without actually compressing.
        /// </summary>
        /// <param name="data">Data to estimate for.</param>
        /// <param name="algorithm">Algorithm to estimate.</param>
        /// <returns>Estimated compression ratio.</returns>
        public static unsafe double EstimateCompressionRatio(
            ReadOnlySpan<byte> data,
            CompressionAlgorithm algorithm = CompressionAlgorithm.LZ4)
        {
            if (data.Length == 0)
                return 1.0;

            int sampleSize = Math.Min(data.Length, 8192);
            var sample = data[..sampleSize];

            int uniqueBytes = 0;
            var seen = stackalloc bool[256];
            for (int i = 0; i < sample.Length; i++)
            {
                byte b = sample[i];
                if (!seen[b])
                {
                    seen[b] = true;
                    uniqueBytes++;
                }
            }

            double entropy = 0;
            for (int i = 0; i < 256; i++)
            {
                if (seen[i])
                {
                    double freq = 1.0 / uniqueBytes;
                    entropy -= freq * Math.Log2(freq);
                }
            }

            double estimatedRatio = algorithm switch
            {
                CompressionAlgorithm.LZ4 => Math.Max(1.0, 8.0 / entropy * 0.8),
                CompressionAlgorithm.Zstd => Math.Max(1.0, 8.0 / entropy * 0.9),
                CompressionAlgorithm.Brotli => Math.Max(1.0, 8.0 / entropy * 0.95),
                CompressionAlgorithm.Deflate => Math.Max(1.0, 8.0 / entropy * 0.75),
                CompressionAlgorithm.GZip => Math.Max(1.0, 8.0 / entropy * 0.75),
                _ => 1.0
            };

            return Math.Min(estimatedRatio, 32.0);
        }

        /// <summary>
        /// Selects the best compression algorithm based on data characteristics.
        /// </summary>
        private static CompressionAlgorithm SelectBestAlgorithm(ReadOnlySpan<byte> data)
        {
            if (data.Length < 128)
                return CompressionAlgorithm.None;

            double ratio = EstimateCompressionRatio(data, CompressionAlgorithm.LZ4);

            if (data.Length > 1024 * 1024)
                return CompressionAlgorithm.Zstd;

            if (ratio > 4.0)
                return CompressionAlgorithm.Brotli;

            return CompressionAlgorithm.LZ4;
        }

        /// <summary>
        /// Adapts compression level based on data and target ratio.
        /// </summary>
        public int AdaptCompressionLevel(ReadOnlySpan<byte> data)
        {
            if (!_config.EnableAdaptiveLevel)
                return _config.Level;

            double estimatedRatio = EstimateCompressionRatio(data);

            if (estimatedRatio >= _config.TargetCompressionRatio)
                return Math.Max(1, _config.Level - 3);

            if (estimatedRatio < 2.0)
                return Math.Min(22, _config.Level + 3);

            return _config.Level;
        }

        #region FP16 Quantization

        /// <summary>
        /// Quantizes FP32 weights to FP16 (IEEE 754 half-precision).
        /// </summary>
        public static unsafe byte[] QuantizeFP16(ReadOnlySpan<float> weights)
        {
            byte[] result = new byte[weights.Length * 2];
            Span<Half> dest = MemoryMarshal.Cast<byte, Half>(result.AsSpan());

            for (int i = 0; i < weights.Length; i++)
            {
                dest[i] = (Half)weights[i];
            }

            return result;
        }

        /// <summary>
        /// Dequantizes FP16 weights back to FP32.
        /// </summary>
        public static float[] DequantizeFP16(ReadOnlySpan<byte> data, int count)
        {
            var halves = MemoryMarshal.Cast<byte, Half>(data)[..count];
            float[] result = new float[count];

            for (int i = 0; i < count; i++)
            {
                result[i] = (float)halves[i];
            }

            return result;
        }

        #endregion

        #region FP8 Quantization

        /// <summary>
        /// Quantizes FP32 weights to FP8 E5M2 format.
        /// Format: 1 sign + 5 exponent + 2 mantissa bits.
        /// </summary>
        public static unsafe byte[] QuantizeFP8E5M2(ReadOnlySpan<float> weights)
        {
            byte[] result = new byte[weights.Length];

            for (int i = 0; i < weights.Length; i++)
            {
                result[i] = FloatToFP8E5M2(weights[i]);
            }

            return result;
        }

        /// <summary>
        /// Dequantizes FP8 E5M2 weights back to FP32.
        /// </summary>
        public static float[] DequantizeFP8E5M2(ReadOnlySpan<byte> data, int count)
        {
            float[] result = new float[count];

            for (int i = 0; i < count; i++)
            {
                result[i] = FP8E5M2ToFloat(data[i]);
            }

            return result;
        }

        /// <summary>
        /// Converts a float to FP8 E5M2 representation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte FloatToFP8E5M2(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            uint sign = (bits >> 24) & 0x80;
            int exponent = (int)((bits >> 23) & 0xFF) - 127 + 15;
            uint mantissa = (bits >> 21) & 0x03;

            if (exponent <= 0)
                return (byte)(sign | 0x01);
            if (exponent >= 31)
                return (byte)(sign | 0x7F);

            return (byte)(sign | ((uint)exponent << 2) | mantissa);
        }

        /// <summary>
        /// Converts FP8 E5M2 representation to float.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float FP8E5M2ToFloat(byte value)
        {
            uint sign = (uint)(value & 0x80) << 24;
            int exponent = ((value >> 2) & 0x1F) - 15 + 127;
            uint mantissa = (uint)(value & 0x03) << 21;

            if ((value & 0x7F) == 0)
                return BitConverter.UInt32BitsToSingle(sign);
            if ((value & 0x7F) == 0x7F)
                return BitConverter.UInt32BitsToSingle(sign | 0x7F800000);

            uint bits = sign | ((uint)exponent << 23) | mantissa;
            return BitConverter.UInt32BitsToSingle(bits);
        }

        /// <summary>
        /// Quantizes FP32 weights to FP8 E4M3 format.
        /// Format: 1 sign + 4 exponent + 3 mantissa bits.
        /// </summary>
        public static unsafe byte[] QuantizeFP8E4M3(ReadOnlySpan<float> weights)
        {
            byte[] result = new byte[weights.Length];

            for (int i = 0; i < weights.Length; i++)
            {
                result[i] = FloatToFP8E4M3(weights[i]);
            }

            return result;
        }

        /// <summary>
        /// Dequantizes FP8 E4M3 weights back to FP32.
        /// </summary>
        public static float[] DequantizeFP8E4M3(ReadOnlySpan<byte> data, int count)
        {
            float[] result = new float[count];

            for (int i = 0; i < count; i++)
            {
                result[i] = FP8E4M3ToFloat(data[i]);
            }

            return result;
        }

        /// <summary>
        /// Converts a float to FP8 E4M3 representation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte FloatToFP8E4M3(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            uint sign = (bits >> 24) & 0x80;
            int exponent = (int)((bits >> 23) & 0xFF) - 127 + 7;
            uint mantissa = (bits >> 20) & 0x07;

            if (exponent <= 0)
                return (byte)(sign | 0x01);
            if (exponent >= 15)
                return (byte)(sign | 0x7F);

            return (byte)(sign | ((uint)exponent << 3) | mantissa);
        }

        /// <summary>
        /// Converts FP8 E4M3 representation to float.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float FP8E4M3ToFloat(byte value)
        {
            uint sign = (uint)(value & 0x80) << 24;
            int exponent = ((value >> 3) & 0x0F) - 7 + 127;
            uint mantissa = (uint)(value & 0x07) << 20;

            if ((value & 0x7F) == 0)
                return BitConverter.UInt32BitsToSingle(sign);
            if ((value & 0x7F) == 0x7F)
                return BitConverter.UInt32BitsToSingle(sign | 0x7F800000);

            uint bits = sign | ((uint)exponent << 23) | mantissa;
            return BitConverter.UInt32BitsToSingle(bits);
        }

        #endregion

        #region UINT8 Quantization

        /// <summary>
        /// Quantizes FP32 weights to UINT8 with scale and zero-point.
        /// </summary>
        public static byte[] QuantizeUINT8(ReadOnlySpan<float> weights)
        {
            if (weights.Length == 0)
                return [];

            float min = float.MaxValue;
            float max = float.MinValue;

            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] < min)
                    min = weights[i];
                if (weights[i] > max)
                    max = weights[i];
            }

            float range = max - min;
            if (range < 1e-10f)
                range = 1.0f;

            float scale = range / 255.0f;
            byte zeroPoint = (byte)(-min / scale);

            byte[] result = new byte[weights.Length + 5];
            BitConverter.TryWriteBytes(result.AsSpan(0), scale);
            result[4] = zeroPoint;

            for (int i = 0; i < weights.Length; i++)
            {
                int quantized = (int)((weights[i] - min) / range * 255.0f + 0.5f);
                result[5 + i] = (byte)Math.Clamp(quantized, 0, 255);
            }

            return result;
        }

        /// <summary>
        /// Dequantizes UINT8 weights back to FP32.
        /// </summary>
        public static float[] DequantizeUINT8(ReadOnlySpan<byte> data, int count)
        {
            if (data.Length < 5)
                return [];

            float scale = BitConverter.ToSingle(data[..4]);
            byte zeroPoint = data[4];
            float[] result = new float[count];

            for (int i = 0; i < count && i + 5 < data.Length; i++)
            {
                result[i] = (data[5 + i] - zeroPoint) * scale;
            }

            return result;
        }

        #endregion

        #region INT4 Quantization

        /// <summary>
        /// Quantizes FP32 weights to INT4 (two values per byte).
        /// </summary>
        public static byte[] QuantizeINT4(ReadOnlySpan<float> weights)
        {
            if (weights.Length == 0)
                return [];

            float min = float.MaxValue;
            float max = float.MinValue;

            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] < min)
                    min = weights[i];
                if (weights[i] > max)
                    max = weights[i];
            }

            float range = max - min;
            if (range < 1e-10f)
                range = 1.0f;

            int byteCount = (weights.Length + 1) / 2;
            byte[] result = new byte[byteCount + 4];
            BitConverter.TryWriteBytes(result.AsSpan(0), range);
            BitConverter.TryWriteBytes(result.AsSpan(4), min);

            for (int i = 0; i < weights.Length; i++)
            {
                int quantized = (int)((weights[i] - min) / range * 15.0f + 0.5f);
                quantized = Math.Clamp(quantized, 0, 15);

                int byteIndex = 5 + i / 2;
                if (i % 2 == 0)
                    result[byteIndex] = (byte)(quantized & 0x0F);
                else
                    result[byteIndex] |= (byte)((quantized & 0x0F) << 4);
            }

            return result;
        }

        /// <summary>
        /// Dequantizes INT4 weights back to FP32.
        /// </summary>
        public static float[] DequantizeINT4(ReadOnlySpan<byte> data, int count)
        {
            if (data.Length < 4)
                return [];

            float range = BitConverter.ToSingle(data[..4]);
            float min = BitConverter.ToSingle(data[4..8]);
            float[] result = new float[count];

            for (int i = 0; i < count; i++)
            {
                int byteIndex = 8 + i / 2;
                if (byteIndex >= data.Length)
                    break;

                int nibble;
                if (i % 2 == 0)
                    nibble = data[byteIndex] & 0x0F;
                else
                    nibble = (data[byteIndex] >> 4) & 0x0F;

                result[i] = min + (nibble / 15.0f) * range;
            }

            return result;
        }

        #endregion

        #region LZ4 Compression

        /// <summary>
        /// Compresses data using LZ4 algorithm.
        /// Uses .NET's built-in Brotli as a stand-in; production would use native LZ4.
        /// </summary>
        private static byte[] CompressLZ4(ReadOnlySpan<byte> data, int level)
        {
            using var output = new MemoryStream();
            output.Write(BitConverter.GetBytes(data.Length), 0, 4);

            using (var compressor = new BrotliStream(output, CompressionLevel.Fastest, leaveOpen: true))
            {
                compressor.Write(data);
            }

            return output.ToArray();
        }

        /// <summary>
        /// Decompresses LZ4 compressed data.
        /// </summary>
        private static byte[] DecompressLZ4(ReadOnlySpan<byte> data)
        {
            if (data.Length < 4)
                return data.ToArray();

            int originalSize = BitConverter.ToInt32(data[..4]);
            using var input = new MemoryStream(data[4..].ToArray());
            using var decompressor = new BrotliStream(input, CompressionMode.Decompress);

            byte[] result = new byte[originalSize];
            int totalRead = 0;
            while (totalRead < originalSize)
            {
                int read = decompressor.Read(result, totalRead, originalSize - totalRead);
                if (read == 0)
                    break;
                totalRead += read;
            }

            return result;
        }

        #endregion

        #region Zstd Compression

        /// <summary>
        /// Compresses data using Zstandard algorithm.
        /// Uses Brotli as managed alternative.
        /// </summary>
        private static byte[] CompressZstd(ReadOnlySpan<byte> data, int level)
        {
            using var output = new MemoryStream();
            output.Write(BitConverter.GetBytes(data.Length), 0, 4);

            var brotliLevel = level <= 4
                ? CompressionLevel.Fastest
                : level <= 12
                    ? CompressionLevel.Optimal
                    : CompressionLevel.SmallestSize;

            using (var compressor = new BrotliStream(output, brotliLevel, leaveOpen: true))
            {
                compressor.Write(data);
            }

            return output.ToArray();
        }

        /// <summary>
        /// Decompresses Zstd compressed data.
        /// </summary>
        private static byte[] DecompressZstd(ReadOnlySpan<byte> data)
        {
            return DecompressLZ4(data);
        }

        #endregion

        #region Brotli Compression

        /// <summary>
        /// Compresses data using Brotli algorithm.
        /// </summary>
        private static byte[] CompressBrotli(ReadOnlySpan<byte> data, int level)
        {
            using var output = new MemoryStream();
            output.Write(BitConverter.GetBytes(data.Length), 0, 4);

            var brotliLevel = level switch
            {
                <= 4 => CompressionLevel.Fastest,
                <= 12 => CompressionLevel.Optimal,
                _ => CompressionLevel.SmallestSize
            };

            using (var compressor = new BrotliStream(output, brotliLevel, leaveOpen: true))
            {
                compressor.Write(data);
            }

            return output.ToArray();
        }

        /// <summary>
        /// Decompresses Brotli compressed data.
        /// </summary>
        private static byte[] DecompressBrotli(ReadOnlySpan<byte> data)
        {
            return DecompressLZ4(data);
        }

        #endregion

        #region Deflate Compression

        /// <summary>
        /// Compresses data using Deflate algorithm.
        /// </summary>
        private static byte[] CompressDeflate(ReadOnlySpan<byte> data, int level)
        {
            using var output = new MemoryStream();
            output.Write(BitConverter.GetBytes(data.Length), 0, 4);

            var compressLevel = level switch
            {
                <= 3 => System.IO.Compression.CompressionLevel.Fastest,
                <= 10 => System.IO.Compression.CompressionLevel.Optimal,
                _ => System.IO.Compression.CompressionLevel.SmallestSize
            };

            using (var compressor = new DeflateStream(output, compressLevel, leaveOpen: true))
            {
                compressor.Write(data);
            }

            return output.ToArray();
        }

        /// <summary>
        /// Decompresses Deflate compressed data.
        /// </summary>
        private static byte[] DecompressDeflate(ReadOnlySpan<byte> data)
        {
            if (data.Length < 4)
                return data.ToArray();

            int originalSize = BitConverter.ToInt32(data[..4]);
            using var input = new MemoryStream(data[4..].ToArray());
            using var decompressor = new DeflateStream(input, CompressionMode.Decompress);

            byte[] result = new byte[originalSize];
            int totalRead = 0;
            while (totalRead < originalSize)
            {
                int read = decompressor.Read(result, totalRead, originalSize - totalRead);
                if (read == 0)
                    break;
                totalRead += read;
            }

            return result;
        }

        #endregion

        #region GZip Compression

        /// <summary>
        /// Compresses data using GZip algorithm.
        /// </summary>
        private static byte[] CompressGZip(ReadOnlySpan<byte> data, int level)
        {
            using var output = new MemoryStream();
            output.Write(BitConverter.GetBytes(data.Length), 0, 4);

            var compressLevel = level switch
            {
                <= 3 => System.IO.Compression.CompressionLevel.Fastest,
                <= 10 => System.IO.Compression.CompressionLevel.Optimal,
                _ => System.IO.Compression.CompressionLevel.SmallestSize
            };

            using (var compressor = new GZipStream(output, compressLevel, leaveOpen: true))
            {
                compressor.Write(data);
            }

            return output.ToArray();
        }

        /// <summary>
        /// Decompresses GZip compressed data.
        /// </summary>
        private static byte[] DecompressGZip(ReadOnlySpan<byte> data)
        {
            if (data.Length < 4)
                return data.ToArray();

            int originalSize = BitConverter.ToInt32(data[..4]);
            using var input = new MemoryStream(data[4..].ToArray());
            using var decompressor = new GZipStream(input, CompressionMode.Decompress);

            byte[] result = new byte[originalSize];
            int totalRead = 0;
            while (totalRead < originalSize)
            {
                int read = decompressor.Read(result, totalRead, originalSize - totalRead);
                if (read == 0)
                    break;
                totalRead += read;
            }

            return result;
        }

        #endregion

        #region Streaming Compression

        /// <summary>
        /// Creates a streaming compressor that processes data in blocks.
        /// </summary>
        /// <param name="outputStream">Output stream to write compressed data to.</param>
        /// <param name="algorithm">Compression algorithm to use.</param>
        /// <returns>A streaming compressor.</returns>
        public StreamingCompressor CreateStreamingCompressor(
            Stream outputStream,
            CompressionAlgorithm algorithm = CompressionAlgorithm.LZ4)
        {
            return new StreamingCompressor(outputStream, algorithm, _config.StreamingBlockSize, _config.Level);
        }

        /// <summary>
        /// Creates a streaming decompressor that processes data in blocks.
        /// </summary>
        /// <param name="inputStream">Input stream to read compressed data from.</param>
        /// <param name="algorithm">Decompression algorithm to use.</param>
        /// <returns>A streaming decompressor.</returns>
        public StreamingDecompressor CreateStreamingDecompressor(
            Stream inputStream,
            CompressionAlgorithm algorithm = CompressionAlgorithm.LZ4)
        {
            return new StreamingDecompressor(inputStream, algorithm);
        }

        #endregion

        #region Dictionary-based Weight Sharing

        /// <summary>
        /// Extracts a compact dictionary from a set of weight arrays for sharing.
        /// </summary>
        /// <param name="weightSets">Collection of weight arrays to analyze.</param>
        /// <param name="maxPatterns">Maximum number of patterns to include.</param>
        /// <returns>Dictionary bytes.</returns>
        public static byte[] ExtractSharedDictionary(IEnumerable<float[]> weightSets, int maxPatterns = 256)
        {
            var histogram = new Dictionary<int, int>();
            int totalElements = 0;

            foreach (var weights in weightSets)
            {
                foreach (var w in weights)
                {
                    int quantized = (int)(w * 1000.0f);
                    if (histogram.ContainsKey(quantized))
                        histogram[quantized]++;
                    else
                        histogram[quantized] = 1;
                    totalElements++;
                }
            }

            var topPatterns = histogram
                .OrderByDescending(kvp => kvp.Value)
                .Take(maxPatterns)
                .ToList();

            byte[] result = new byte[topPatterns.Count * 8 + 4];
            BitConverter.TryWriteBytes(result.AsSpan(0), topPatterns.Count);

            int offset = 4;
            foreach (var pattern in topPatterns)
            {
                BitConverter.TryWriteBytes(result.AsSpan(offset), pattern.Key);
                BitConverter.TryWriteBytes(result.AsSpan(offset + 4), pattern.Value);
                offset += 8;
            }

            return result;
        }

        #endregion

        #region Compression Benchmarking

        /// <summary>
        /// Benchmarks all supported compression algorithms on the given data.
        /// </summary>
        /// <param name="data">Data to benchmark.</param>
        /// <returns>Benchmark results per algorithm.</returns>
        public List<CompressionResult> Benchmark(ReadOnlySpan<byte> data)
        {
            var results = new List<CompressionResult>();

            var algorithms = new[]
            {
                CompressionAlgorithm.LZ4,
                CompressionAlgorithm.Zstd,
                CompressionAlgorithm.Brotli,
                CompressionAlgorithm.Deflate,
                CompressionAlgorithm.GZip
            };

            foreach (var algo in algorithms)
            {
                if (data.Length > 0)
                {
                    var result = Compress(data, algo);
                    results.Add(result);
                }
            }

            return results;
        }

        #endregion

        /// <summary>
        /// Byte array comparer for dictionary keys.
        /// </summary>
        private sealed class ByteArrayComparer : IEqualityComparer<byte[]>
        {
            public bool Equals(byte[]? x, byte[]? y)
            {
                if (x == null && y == null)
                    return true;
                if (x == null || y == null)
                    return false;
                return x.AsSpan().SequenceEqual(y);
            }

            public int GetHashCode(byte[] obj)
            {
                return obj.Length > 0 ? obj[0] ^ obj[^1] ^ obj.Length : 0;
            }
        }
    }

    /// <summary>
    /// Streaming compressor that processes data in blocks for memory-efficient compression.
    /// </summary>
    public sealed class StreamingCompressor : IDisposable
    {
        private readonly Stream _outputStream;
        private readonly CompressionAlgorithm _algorithm;
        private readonly int _blockSize;
        private readonly int _level;
        private readonly byte[] _buffer;
        private int _bufferOffset;
        private long _totalBytesWritten;
        private long _totalBlocks;
        private bool _disposed;

        /// <summary>Total bytes written to the output stream.</summary>
        public long TotalBytesWritten => _totalBytesWritten;

        /// <summary>Number of blocks processed.</summary>
        public long TotalBlocks => _totalBlocks;

        internal StreamingCompressor(Stream outputStream, CompressionAlgorithm algorithm, int blockSize, int level)
        {
            _outputStream = outputStream ?? throw new ArgumentNullException(nameof(outputStream));
            _algorithm = algorithm;
            _blockSize = blockSize;
            _level = level;
            _buffer = new byte[blockSize];
            _bufferOffset = 0;

            byte[] header = BitConverter.GetBytes((int)algorithm);
            _outputStream.Write(header, 0, 4);
        }

        /// <summary>
        /// Writes data to the compressor.
        /// </summary>
        public void Write(ReadOnlySpan<byte> data)
        {
            int remaining = data.Length;
            int offset = 0;

            while (remaining > 0)
            {
                int space = _blockSize - _bufferOffset;
                int toCopy = Math.Min(remaining, space);
                data.Slice(offset, toCopy).CopyTo(_buffer.AsSpan(_bufferOffset));
                _bufferOffset += toCopy;
                offset += toCopy;
                remaining -= toCopy;

                if (_bufferOffset >= _blockSize)
                {
                    FlushBlock();
                }
            }
        }

        /// <summary>
        /// Flushes any remaining buffered data.
        /// </summary>
        public void Flush()
        {
            if (_bufferOffset > 0)
            {
                FlushBlock();
            }
        }

        /// <summary>
        /// Flushes a full block to the output stream.
        /// </summary>
        private void FlushBlock()
        {
            var compressed = CompressBlock(_buffer.AsSpan(0, _bufferOffset));

            byte[] lengthBytes = BitConverter.GetBytes(compressed.Length);
            _outputStream.Write(lengthBytes, 0, 4);
            _outputStream.Write(compressed, 0, compressed.Length);

            _totalBytesWritten += 4 + compressed.Length;
            _totalBlocks++;
            _bufferOffset = 0;
        }

        /// <summary>
        /// Compresses a block using the configured algorithm.
        /// </summary>
        private byte[] CompressBlock(ReadOnlySpan<byte> data)
        {
            using var output = new MemoryStream();
            using (var compressor = new BrotliStream(output, CompressionLevel.Fastest, leaveOpen: true))
            {
                compressor.Write(data);
            }
            return output.ToArray();
        }

        /// <summary>
        /// Disposes the streaming compressor and flushes remaining data.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            Flush();
        }
    }

    /// <summary>
    /// Streaming decompressor that processes block-compressed data.
    /// </summary>
    public sealed class StreamingDecompressor : IDisposable
    {
        private readonly Stream _inputStream;
        private readonly CompressionAlgorithm _algorithm;
        private readonly byte[] _lengthBuffer = new byte[4];
        private long _totalBytesRead;
        private long _totalBlocks;
        private bool _disposed;
        private bool _finished;

        /// <summary>Total bytes read from the input stream.</summary>
        public long TotalBytesRead => _totalBytesRead;

        /// <summary>Number of blocks decompressed.</summary>
        public long TotalBlocks => _totalBlocks;

        /// <summary>Whether all blocks have been read.</summary>
        public bool IsFinished => _finished;

        internal StreamingDecompressor(Stream inputStream, CompressionAlgorithm algorithm)
        {
            _inputStream = inputStream ?? throw new ArgumentNullException(nameof(inputStream));
            _algorithm = algorithm;

            if (_inputStream.Read(_lengthBuffer, 0, 4) < 4)
                _finished = true;
        }

        /// <summary>
        /// Reads and decompresses the next block.
        /// </summary>
        /// <param name="buffer">Buffer to write decompressed data into.</param>
        /// <returns>Number of bytes written to the buffer.</returns>
        public int ReadBlock(Span<byte> buffer)
        {
            if (_finished)
                return 0;

            if (_inputStream.Read(_lengthBuffer, 0, 4) < 4)
            {
                _finished = true;
                return 0;
            }

            int compressedSize = BitConverter.ToInt32(_lengthBuffer);
            if (compressedSize <= 0)
            {
                _finished = true;
                return 0;
            }

            byte[] compressedData = new byte[compressedSize];
            int totalRead = 0;
            while (totalRead < compressedSize)
            {
                int read = _inputStream.Read(compressedData, totalRead, compressedSize - totalRead);
                if (read == 0)
                { _finished = true; break; }
                totalRead += read;
            }

            byte[] decompressed = DecompressBlock(compressedData);
            int toCopy = Math.Min(decompressed.Length, buffer.Length);
            decompressed.AsSpan(0, toCopy).CopyTo(buffer);

            _totalBytesRead += 4 + compressedSize;
            _totalBlocks++;

            if (toCopy < decompressed.Length)
                throw new ArgumentException("Buffer too small for decompressed block.");

            return toCopy;
        }

        /// <summary>
        /// Decompresses a block using the configured algorithm.
        /// </summary>
        private byte[] DecompressBlock(ReadOnlySpan<byte> data)
        {
            using var input = new MemoryStream(data.ToArray());
            using var decompressor = new BrotliStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream();
            decompressor.CopyTo(output);
            return output.ToArray();
        }

        /// <summary>
        /// Disposes the streaming decompressor.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
        }
    }
}
