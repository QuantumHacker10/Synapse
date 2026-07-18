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
// FILE: SkinningWeights.cs
// PATH: Animation/SkinningWeights.cs
// ============================================================


using System;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace GDNN.Animation
{
    /// <summary>
    /// Manages per-vertex skinning weight data for skeletal mesh deformation.
    /// Supports configurable maximum bone influences, weight normalization,
    /// quantized storage (4-bit and 8-bit), compression, and LOD bone index remapping.
    /// </summary>
    public sealed class SkinningWeights : IDisposable
    {
        /// <summary>Default maximum number of bone influences per vertex.</summary>
        public const int DefaultMaxInfluences = 4;

        /// <summary>Maximum supported bone influences per vertex.</summary>
        public const int MaxSupportedInfluences = 8;

        /// <summary>Number of vertices managed by this instance.</summary>
        private int _vertexCount;

        /// <summary>Maximum bone influences per vertex.</summary>
        private int _maxInfluences;

        /// <summary>Raw weight data stored as float per (vertex, influence) pair.
        /// Layout: weights[vertex * _maxInfluences + influence].</summary>
        private float[] _weights;

        /// <summary>Bone index data stored as int per (vertex, influence) pair.
        /// Layout: boneIndices[vertex * _maxInfluences + influence].</summary>
        private int[] _boneIndices;

        /// <summary>Number of active influences per vertex (can be less than _maxInfluences).</summary>
        private int[] _influenceCounts;

        /// <summary>Quantized 8-bit weight storage (when using 8-bit quantization).</summary>
        private byte[] _quantizedWeights8;

        /// <summary>Quantized 4-bit weight storage (packed two per byte).</summary>
        private byte[] _quantizedWeights4;

        /// <summary>Whether the weights are currently in quantized form.</summary>
        private bool _isQuantized;

        /// <summary>Quantization mode currently in use.</summary>
        private QuantizationMode _quantizationMode;

        /// <summary>Whether this instance has been disposed.</summary>
        private bool _disposed;

        /// <summary>
        /// Represents the quantization mode for weight storage.
        /// </summary>
        public enum QuantizationMode : byte
        {
            /// <summary>Full 32-bit floating point weights.</summary>
            FullPrecision = 0,

            /// <summary>8-bit quantized weights (256 levels).</summary>
            EightBit = 1,

            /// <summary>4-bit quantized weights (16 levels, packed two per byte).</summary>
            FourBit = 2
        }

        /// <summary>
        /// Represents a single skinning influence (bone index + weight).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct Influence
        {
            /// <summary>Index of the bone affecting this vertex.</summary>
            public int BoneIndex;

            /// <summary>Weight of the influence (0..1).</summary>
            public float Weight;

            /// <summary>
            /// Initializes a new influence.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public Influence(int boneIndex, float weight)
            {
                BoneIndex = boneIndex;
                Weight = weight;
            }

            /// <summary>Default (identity) influence.</summary>
            public static readonly Influence Identity = new Influence(0, 1.0f);
        }

        /// <summary>
        /// Represents a quantized 8-bit influence (bone index + 8-bit weight).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct QuantizedInfluence8
        {
            /// <summary>Index of the bone.</summary>
            public ushort BoneIndex;

            /// <summary>Quantized weight (0-255).</summary>
            public byte Weight;

            /// <summary>Padding for alignment.</summary>
            public byte Padding;

            /// <summary>
            /// Decompresses to a full-precision influence.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public Influence Decompress()
            {
                return new Influence(BoneIndex, Weight / 255.0f);
            }

            /// <summary>
            /// Compresses a full-precision influence to 8-bit.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static QuantizedInfluence8 Compress(Influence influence)
            {
                return new QuantizedInfluence8
                {
                    BoneIndex = (ushort)influence.BoneIndex,
                    Weight = (byte)Math.Clamp((int)(influence.Weight * 255.0f + 0.5f), 0, 255),
                    Padding = 0
                };
            }
        }

        /// <summary>
        /// Represents quantized 4-bit influences packed into a single byte.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct PackedInfluences4
        {
            /// <summary>High nibble bone index (0-15).</summary>
            public byte HighBoneIndex;

            /// <summary>Low nibble bone index (0-15).</summary>
            public byte LowBoneIndex;

            /// <summary>High nibble weight (0-15).</summary>
            public byte HighWeight;

            /// <summary>Low nibble weight (0-15).</summary>
            public byte LowWeight;

            /// <summary>
            /// Unpacks two 4-bit influences from this packed byte pair.
            /// </summary>
            public (Influence high, Influence low) Unpack()
            {
                var high = new Influence(HighBoneIndex, HighWeight / 15.0f);
                var low = new Influence(LowBoneIndex, LowWeight / 15.0f);
                return (high, low);
            }

            /// <summary>
            /// Packs two influences into 4-bit representation.
            /// </summary>
            public static PackedInfluences4 Pack(Influence a, Influence b)
            {
                return new PackedInfluences4
                {
                    HighBoneIndex = (byte)(a.BoneIndex & 0x0F),
                    LowBoneIndex = (byte)(b.BoneIndex & 0x0F),
                    HighWeight = (byte)Math.Clamp((int)(a.Weight * 15.0f + 0.5f), 0, 15),
                    LowWeight = (byte)Math.Clamp((int)(b.Weight * 15.0f + 0.5f), 0, 15)
                };
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SkinningWeights"/> class.
        /// </summary>
        /// <param name="vertexCount">Number of vertices.</param>
        /// <param name="maxInfluences">Maximum bone influences per vertex (1-8).</param>
        public SkinningWeights(int vertexCount, int maxInfluences = DefaultMaxInfluences)
        {
            if (vertexCount < 0)
                throw new ArgumentOutOfRangeException(nameof(vertexCount));
            if (maxInfluences < 1 || maxInfluences > MaxSupportedInfluences)
                throw new ArgumentOutOfRangeException(nameof(maxInfluences), $"Must be between 1 and {MaxSupportedInfluences}.");

            _vertexCount = vertexCount;
            _maxInfluences = maxInfluences;
            _weights = new float[vertexCount * maxInfluences];
            _boneIndices = new int[vertexCount * maxInfluences];
            _influenceCounts = new int[vertexCount];
            _quantizedWeights8 = Array.Empty<byte>();
            _quantizedWeights4 = Array.Empty<byte>();
            _isQuantized = false;
            _quantizationMode = QuantizationMode.FullPrecision;
            _disposed = false;
        }

        /// <summary>Gets the number of vertices.</summary>
        public int VertexCount => _vertexCount;

        /// <summary>Gets the maximum bone influences per vertex.</summary>
        public int MaxInfluences => _maxInfluences;

        /// <summary>Gets whether the weights are currently quantized.</summary>
        public bool IsQuantized => _isQuantized;

        /// <summary>Gets the current quantization mode.</summary>
        public QuantizationMode CurrentQuantization => _quantizationMode;

        /// <summary>
        /// Gets or sets the weight for a specific vertex and influence slot.
        /// </summary>
        /// <param name="vertexIndex">Vertex index.</param>
        /// <param name="influenceIndex">Influence slot (0 to MaxInfluences-1).</param>
        public float this[int vertexIndex, int influenceIndex]
        {
            get
            {
                ValidateIndices(vertexIndex, influenceIndex);
                return _weights[vertexIndex * _maxInfluences + influenceIndex];
            }
            set
            {
                ValidateIndices(vertexIndex, influenceIndex);
                _weights[vertexIndex * _maxInfluences + influenceIndex] = value;
            }
        }

        /// <summary>
        /// Gets or sets the bone index for a specific vertex and influence slot.
        /// </summary>
        public int GetBoneIndex(int vertexIndex, int influenceIndex)
        {
            ValidateIndices(vertexIndex, influenceIndex);
            return _boneIndices[vertexIndex * _maxInfluences + influenceIndex];
        }

        /// <summary>
        /// Sets the bone index for a specific vertex and influence slot.
        /// </summary>
        public void SetBoneIndex(int vertexIndex, int influenceIndex, int boneIndex)
        {
            ValidateIndices(vertexIndex, influenceIndex);
            _boneIndices[vertexIndex * _maxInfluences + influenceIndex] = boneIndex;
        }

        /// <summary>
        /// Gets the number of active influences for a vertex.
        /// </summary>
        public int GetInfluenceCount(int vertexIndex)
        {
            if ((uint)vertexIndex >= (uint)_vertexCount)
                throw new ArgumentOutOfRangeException(nameof(vertexIndex));
            return _influenceCounts[vertexIndex];
        }

        /// <summary>
        /// Sets the number of active influences for a vertex.
        /// </summary>
        public void SetInfluenceCount(int vertexIndex, int count)
        {
            if ((uint)vertexIndex >= (uint)_vertexCount)
                throw new ArgumentOutOfRangeException(nameof(vertexIndex));
            if (count < 0 || count > _maxInfluences)
                throw new ArgumentOutOfRangeException(nameof(count));
            _influenceCounts[vertexIndex] = count;
        }

        /// <summary>
        /// Sets all influences for a vertex at once.
        /// </summary>
        /// <param name="vertexIndex">Vertex index.</param>
        /// <param name="influences">Span of influences to set (up to MaxInfluences).</param>
        public void SetInfluences(int vertexIndex, ReadOnlySpan<Influence> influences)
        {
            if ((uint)vertexIndex >= (uint)_vertexCount)
                throw new ArgumentOutOfRangeException(nameof(vertexIndex));

            int count = Math.Min(influences.Length, _maxInfluences);
            int baseIdx = vertexIndex * _maxInfluences;

            for (int i = 0; i < count; i++)
            {
                _weights[baseIdx + i] = influences[i].Weight;
                _boneIndices[baseIdx + i] = influences[i].BoneIndex;
            }

            for (int i = count; i < _maxInfluences; i++)
            {
                _weights[baseIdx + i] = 0f;
                _boneIndices[baseIdx + i] = 0;
            }

            _influenceCounts[vertexIndex] = count;
        }

        /// <summary>
        /// Gets all influences for a vertex.
        /// </summary>
        /// <param name="vertexIndex">Vertex index.</param>
        /// <param name="output">Span to write influences into.</param>
        /// <returns>Number of active influences written.</returns>
        public int GetInfluences(int vertexIndex, Span<Influence> output)
        {
            if ((uint)vertexIndex >= (uint)_vertexCount)
                throw new ArgumentOutOfRangeException(nameof(vertexIndex));

            int count = _influenceCounts[vertexIndex];
            int baseIdx = vertexIndex * _maxInfluences;
            int resultCount = Math.Min(count, output.Length);

            for (int i = 0; i < resultCount; i++)
            {
                output[i] = new Influence(_boneIndices[baseIdx + i], _weights[baseIdx + i]);
            }

            return resultCount;
        }

        /// <summary>
        /// Returns a span over the raw weight data for a vertex.
        /// </summary>
        public Span<float> GetWeightSpan(int vertexIndex)
        {
            if ((uint)vertexIndex >= (uint)_vertexCount)
                throw new ArgumentOutOfRangeException(nameof(vertexIndex));
            int baseIdx = vertexIndex * _maxInfluences;
            return new Span<float>(_weights, baseIdx, _maxInfluences);
        }

        /// <summary>
        /// Returns a span over the raw bone index data for a vertex.
        /// </summary>
        public Span<int> GetBoneIndexSpan(int vertexIndex)
        {
            if ((uint)vertexIndex >= (uint)_vertexCount)
                throw new ArgumentOutOfRangeException(nameof(vertexIndex));
            int baseIdx = vertexIndex * _maxInfluences;
            return new Span<int>(_boneIndices, baseIdx, _maxInfluences);
        }

        // ── Weight Normalization ─────────────────────────────────────────

        /// <summary>
        /// Normalizes the weights for a single vertex so they sum to 1.0.
        /// Only normalizes the active influences (up to the influence count).
        /// </summary>
        /// <param name="vertexIndex">Vertex index.</param>
        public void NormalizeWeights(int vertexIndex)
        {
            if ((uint)vertexIndex >= (uint)_vertexCount)
                throw new ArgumentOutOfRangeException(nameof(vertexIndex));

            int count = _influenceCounts[vertexIndex];
            if (count <= 0) return;

            int baseIdx = vertexIndex * _maxInfluences;
            float sum = 0f;

            for (int i = 0; i < count; i++)
            {
                sum += _weights[baseIdx + i];
            }

            if (sum > 1e-6f)
            {
                float invSum = 1.0f / sum;
                for (int i = 0; i < count; i++)
                {
                    _weights[baseIdx + i] *= invSum;
                }
            }
            else
            {
                if (count > 0)
                {
                    _weights[baseIdx] = 1.0f;
                    for (int i = 1; i < count; i++)
                    {
                        _weights[baseIdx + i] = 0f;
                    }
                }
            }
        }

        /// <summary>
        /// Normalizes weights for all vertices.
        /// </summary>
        public void NormalizeAllWeights()
        {
            for (int v = 0; v < _vertexCount; v++)
            {
                NormalizeWeights(v);
            }
        }

        /// <summary>
        /// Renormalizes weights after modification, clamping negative values to zero first.
        /// </summary>
        /// <param name="vertexIndex">Vertex index.</param>
        public void RenormalizeWeights(int vertexIndex)
        {
            if ((uint)vertexIndex >= (uint)_vertexCount)
                throw new ArgumentOutOfRangeException(nameof(vertexIndex));

            int count = _influenceCounts[vertexIndex];
            if (count <= 0) return;

            int baseIdx = vertexIndex * _maxInfluences;

            for (int i = 0; i < count; i++)
            {
                if (_weights[baseIdx + i] < 0f)
                    _weights[baseIdx + i] = 0f;
            }

            NormalizeWeights(vertexIndex);
        }

        /// <summary>
        /// Renormalizes weights for all vertices.
        /// </summary>
        public void RenormalizeAllWeights()
        {
            for (int v = 0; v < _vertexCount; v++)
            {
                RenormalizeWeights(v);
            }
        }

        /// <summary>
        /// Sorts influences by descending weight for a vertex, ensuring the most
        /// influential bones come first.
        /// </summary>
        /// <param name="vertexIndex">Vertex index.</param>
        public void SortByWeight(int vertexIndex)
        {
            if ((uint)vertexIndex >= (uint)_vertexCount)
                throw new ArgumentOutOfRangeException(nameof(vertexIndex));

            int count = _influenceCounts[vertexIndex];
            if (count <= 1) return;

            int baseIdx = vertexIndex * _maxInfluences;

            for (int i = 1; i < count; i++)
            {
                float keyWeight = _weights[baseIdx + i];
                int keyBone = _boneIndices[baseIdx + i];
                int j = i - 1;

                while (j >= 0 && _weights[baseIdx + j] < keyWeight)
                {
                    _weights[baseIdx + j + 1] = _weights[baseIdx + j];
                    _boneIndices[baseIdx + j + 1] = _boneIndices[baseIdx + j];
                    j--;
                }

                _weights[baseIdx + j + 1] = keyWeight;
                _boneIndices[baseIdx + j + 1] = keyBone;
            }
        }

        /// <summary>
        /// Sorts influences by descending weight for all vertices.
        /// </summary>
        public void SortAllByWeight()
        {
            for (int v = 0; v < _vertexCount; v++)
            {
                SortByWeight(v);
            }
        }

        /// <summary>
        /// Clamps the influence count per vertex, discarding the least significant influences
        /// and renormalizing.
        /// </summary>
        /// <param name="maxInfluencesPerVertex">Maximum influences to keep per vertex.</param>
        public void ClampInfluences(int maxInfluencesPerVertex)
        {
            if (maxInfluencesPerVertex < 1 || maxInfluencesPerVertex > _maxInfluences)
                throw new ArgumentOutOfRangeException(nameof(maxInfluencesPerVertex));

            for (int v = 0; v < _vertexCount; v++)
            {
                if (_influenceCounts[v] > maxInfluencesPerVertex)
                {
                    SortByWeight(v);
                    _influenceCounts[v] = maxInfluencesPerVertex;
                    NormalizeWeights(v);
                }
            }
        }

        // ── Quantization ─────────────────────────────────────────────────

        /// <summary>
        /// Quantizes weights to 8-bit precision. Converts weight values from float
        /// to byte (0-255) for compact storage.
        /// </summary>
        public unsafe void QuantizeTo8Bit()
        {
            if (_isQuantized) return;

            _quantizedWeights8 = new byte[_vertexCount * _maxInfluences];

            fixed (float* weightsPtr = _weights)
            fixed (byte* quantizedPtr = _quantizedWeights8)
            {
                int total = _vertexCount * _maxInfluences;
                for (int i = 0; i < total; i++)
                {
                    float w = weightsPtr[i];
                    quantizedPtr[i] = (byte)Math.Clamp((int)(w * 255.0f + 0.5f), 0, 255);
                }
            }

            _isQuantized = true;
            _quantizationMode = QuantizationMode.EightBit;
        }

        /// <summary>
        /// Quantizes weights to 4-bit precision. Two weights are packed per byte.
        /// </summary>
        public unsafe void QuantizeTo4Bit()
        {
            if (_isQuantized) return;

            int packedCount = (_vertexCount * _maxInfluences + 1) / 2;
            _quantizedWeights4 = new byte[packedCount];

            fixed (float* weightsPtr = _weights)
            fixed (byte* packedPtr = _quantizedWeights4)
            {
                int total = _vertexCount * _maxInfluences;
                for (int i = 0; i < total; i += 2)
                {
                    float w0 = weightsPtr[i];
                    byte hi = (byte)Math.Clamp((int)(w0 * 15.0f + 0.5f), 0, 15);

                    byte lo = 0;
                    if (i + 1 < total)
                    {
                        float w1 = weightsPtr[i + 1];
                        lo = (byte)Math.Clamp((int)(w1 * 15.0f + 0.5f), 0, 15);
                    }

                    packedPtr[i / 2] = (byte)((hi << 4) | lo);
                }
            }

            _isQuantized = true;
            _quantizationMode = QuantizationMode.FourBit;
        }

        /// <summary>
        /// Decompresses quantized 8-bit weights back to full-precision floats.
        /// </summary>
        public unsafe void Decompress8Bit()
        {
            if (!_isQuantized || _quantizationMode != QuantizationMode.EightBit)
                return;

            fixed (float* weightsPtr = _weights)
            fixed (byte* quantizedPtr = _quantizedWeights8)
            {
                int total = _vertexCount * _maxInfluences;
                for (int i = 0; i < total; i++)
                {
                    weightsPtr[i] = quantizedPtr[i] / 255.0f;
                }
            }

            _isQuantized = false;
            _quantizationMode = QuantizationMode.FullPrecision;
        }

        /// <summary>
        /// Decompresses quantized 4-bit weights back to full-precision floats.
        /// </summary>
        public unsafe void Decompress4Bit()
        {
            if (!_isQuantized || _quantizationMode != QuantizationMode.FourBit)
                return;

            fixed (float* weightsPtr = _weights)
            fixed (byte* packedPtr = _quantizedWeights4)
            {
                int total = _vertexCount * _maxInfluences;
                for (int i = 0; i < total; i++)
                {
                    int byteIndex = i / 2;
                    bool isHigh = (i & 1) == 0;
                    byte packed = packedPtr[byteIndex];
                    byte nibble = isHigh ? (byte)((packed >> 4) & 0x0F) : (byte)(packed & 0x0F);
                    weightsPtr[i] = nibble / 15.0f;
                }
            }

            _isQuantized = false;
            _quantizationMode = QuantizationMode.FullPrecision;
        }

        /// <summary>
        /// Decompresses any quantized weights back to full precision.
        /// </summary>
        public void Decompress()
        {
            if (!_isQuantized) return;

            switch (_quantizationMode)
            {
                case QuantizationMode.EightBit:
                    Decompress8Bit();
                    break;
                case QuantizationMode.FourBit:
                    Decompress4Bit();
                    break;
            }
        }

        // ── Compression ──────────────────────────────────────────────────

        /// <summary>
        /// Computes the memory footprint in bytes for the current weight storage.
        /// </summary>
        public int GetMemoryFootprint()
        {
            int baseSize = _vertexCount * _maxInfluences * (sizeof(float) + sizeof(int))
                         + _vertexCount * sizeof(int);

            return _quantizationMode switch
            {
                QuantizationMode.EightBit => baseSize + _quantizedWeights8.Length,
                QuantizationMode.FourBit => baseSize + _quantizedWeights4.Length,
                _ => baseSize
            };
        }

        /// <summary>
        /// Computes the compressed byte size for 8-bit quantized storage.
        /// </summary>
        public int GetCompressed8BitSize()
        {
            return _vertexCount * _maxInfluences; // 1 byte per weight
        }

        /// <summary>
        /// Computes the compressed byte size for 4-bit quantized storage.
        /// </summary>
        public int GetCompressed4BitSize()
        {
            return (_vertexCount * _maxInfluences + 1) / 2; // 2 weights per byte
        }

        /// <summary>
        /// Serializes the skinning weights to a compressed byte buffer.
        /// </summary>
        /// <param name="useQuantization">Whether to quantize before serializing.</param>
        /// <returns>Compressed byte array.</returns>
        public byte[] Serialize(bool useQuantization = true)
        {
            using var ms = new System.IO.MemoryStream();
            using var writer = new System.IO.BinaryWriter(ms);

            writer.Write((uint)0x534B494E); // Magic: "SKIN"
            writer.Write((uint)1); // Version
            writer.Write((uint)_vertexCount);
            writer.Write((uint)_maxInfluences);
            writer.Write((byte)_quantizationMode);

            if (useQuantization && !_isQuantized)
            {
                if (_maxInfluences <= 4)
                    QuantizeTo4Bit();
                else
                    QuantizeTo8Bit();
            }

            if (_isQuantized)
            {
                switch (_quantizationMode)
                {
                    case QuantizationMode.EightBit:
                        writer.Write(_quantizedWeights8);
                        break;
                    case QuantizationMode.FourBit:
                        writer.Write(_quantizedWeights4);
                        break;
                }
            }
            else
            {
                byte[] weightBytes = MemoryMarshal.AsBytes(
                    new Span<float>(_weights, 0, _vertexCount * _maxInfluences)).ToArray();
                writer.Write(weightBytes);
            }

            for (int v = 0; v < _vertexCount; v++)
            {
                int count = _influenceCounts[v];
                writer.Write((byte)count);

                int baseIdx = v * _maxInfluences;
                for (int i = 0; i < count; i++)
                {
                    writer.Write((ushort)_boneIndices[baseIdx + i]);
                }
            }

            writer.Flush();
            return ms.ToArray();
        }

        /// <summary>
        /// Deserializes skinning weights from a compressed byte buffer.
        /// </summary>
        /// <param name="data">Binary data.</param>
        /// <returns>A new <see cref="SkinningWeights"/> instance.</returns>
        public static SkinningWeights Deserialize(ReadOnlySpan<byte> data)
        {
            if (data.Length < 16)
                throw new ArgumentException("Data too short.");

            int offset = 0;
            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset)); offset += 4;
            if (magic != 0x534B494E)
                throw new InvalidDataException("Invalid skinning weights magic.");

            uint version = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset)); offset += 4;
            uint vertexCount = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset)); offset += 4;
            uint maxInfluences = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offset)); offset += 4;
            QuantizationMode quantMode = (QuantizationMode)data[offset]; offset += 1;

            var skinning = new SkinningWeights((int)vertexCount, (int)maxInfluences);

            switch (quantMode)
            {
                case QuantizationMode.EightBit:
                    int eightSize = (int)vertexCount * (int)maxInfluences;
                    skinning._quantizedWeights8 = data.Slice(offset, eightSize).ToArray();
                    offset += eightSize;
                    skinning._isQuantized = true;
                    skinning._quantizationMode = QuantizationMode.EightBit;
                    skinning.Decompress8Bit();
                    break;

                case QuantizationMode.FourBit:
                    int fourSize = ((int)vertexCount * (int)maxInfluences + 1) / 2;
                    skinning._quantizedWeights4 = data.Slice(offset, fourSize).ToArray();
                    offset += fourSize;
                    skinning._isQuantized = true;
                    skinning._quantizationMode = QuantizationMode.FourBit;
                    skinning.Decompress4Bit();
                    break;

                default:
                    int fullSize = (int)vertexCount * (int)maxInfluences * sizeof(float);
                    for (int i = 0; i < (int)vertexCount * (int)maxInfluences; i++)
                    {
                        skinning._weights[i] = BitConverter.Int32BitsToSingle(
                            BinaryPrimitives.ReadInt32LittleEndian(data.Slice(offset)));
                        offset += 4;
                    }
                    break;
            }

            for (int v = 0; v < (int)vertexCount; v++)
            {
                int count = data[offset]; offset += 1;
                skinning._influenceCounts[v] = count;

                int baseIdx = v * (int)maxInfluences;
                for (int i = 0; i < count; i++)
                {
                    skinning._boneIndices[baseIdx + i] = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(offset));
                    offset += 2;
                }
            }

            return skinning;
        }

        // ── LOD Bone Index Remapping ─────────────────────────────────────

        /// <summary>
        /// Remaps bone indices for LOD reduction. Vertices referencing bones
        /// that are mapped to -1 will have their influence redistributed.
        /// </summary>
        /// <param name="remapTable">Remap table where index = old bone index, value = new bone index (-1 to discard).</param>
        public void RemapBoneIndices(ReadOnlySpan<int> remapTable)
        {
            for (int v = 0; v < _vertexCount; v++)
            {
                int baseIdx = v * _maxInfluences;
                int activeCount = _influenceCounts[v];
                int writeIdx = 0;

                for (int i = 0; i < activeCount; i++)
                {
                    int oldBone = _boneIndices[baseIdx + i];
                    float w = _weights[baseIdx + i];

                    if ((uint)oldBone < (uint)remapTable.Length)
                    {
                        int newBone = remapTable[oldBone];
                        if (newBone >= 0)
                        {
                            _boneIndices[baseIdx + writeIdx] = newBone;
                            _weights[baseIdx + writeIdx] = w;
                            writeIdx++;
                        }
                    }
                }

                for (int i = writeIdx; i < activeCount; i++)
                {
                    _weights[baseIdx + i] = 0f;
                    _boneIndices[baseIdx + i] = 0;
                }

                _influenceCounts[v] = writeIdx;
            }

            RenormalizeAllWeights();
        }

        /// <summary>
        /// Creates a remap table that merges bones above a threshold index.
        /// Bones beyond <paramref name="maxBoneIndex"/> are mapped to the nearest valid bone.
        /// </summary>
        /// <param name="maxBoneIndex">Maximum valid bone index.</param>
        /// <returns>Remap table array.</returns>
        public static int[] CreateLodRemapTable(int maxBoneIndex)
        {
            int[] table = new int[maxBoneIndex + 256];
            for (int i = 0; i < table.Length; i++)
            {
                table[i] = Math.Min(i, maxBoneIndex);
            }
            return table;
        }

        /// <summary>
        /// Gets the maximum bone index referenced by any vertex.
        /// </summary>
        public int GetMaxBoneIndex()
        {
            int max = 0;
            for (int v = 0; v < _vertexCount; v++)
            {
                int baseIdx = v * _maxInfluences;
                int count = _influenceCounts[v];
                for (int i = 0; i < count; i++)
                {
                    if (_boneIndices[baseIdx + i] > max)
                        max = _boneIndices[baseIdx + i];
                }
            }
            return max;
        }

        /// <summary>
        /// Gets the set of all unique bone indices referenced by the mesh.
        /// </summary>
        public HashSet<int> GetReferencedBoneIndices()
        {
            HashSet<int> bones = new HashSet<int>();
            for (int v = 0; v < _vertexCount; v++)
            {
                int baseIdx = v * _maxInfluences;
                int count = _influenceCounts[v];
                for (int i = 0; i < count; i++)
                {
                    bones.Add(_boneIndices[baseIdx + i]);
                }
            }
            return bones;
        }

        // ── Batch Skinning Computation ───────────────────────────────────

        /// <summary>
        /// Computes skinned vertex positions by multiplying each vertex's bind-pose position
        /// by the skinning matrix contributions from all influencing bones.
        /// </summary>
        /// <param name="bindPositions">Bind-pose vertex positions (length >= VertexCount).</param>
        /// <param name="skinningMatrices">Skinning matrices per bone (length >= max bone index + 1).</param>
        /// <param name="outputPositions">Output skinned positions (length >= VertexCount).</param>
        public unsafe void ComputeSkinnedPositions(
            ReadOnlySpan<Vector3> bindPositions,
            ReadOnlySpan<Matrix4x4> skinningMatrices,
            Span<Vector3> outputPositions)
        {
            if (bindPositions.Length < _vertexCount)
                throw new ArgumentException("Bind positions too short.");
            if (outputPositions.Length < _vertexCount)
                throw new ArgumentException("Output positions too short.");

            fixed (float* weightsPtr = _weights)
            fixed (int* bonesPtr = _boneIndices)
            fixed (int* countsPtr = _influenceCounts)
            {
                for (int v = 0; v < _vertexCount; v++)
                {
                    Vector3 pos = bindPositions[v];
                    Vector3 skinned = Vector3.Zero;
                    int baseIdx = v * _maxInfluences;
                    int count = countsPtr[v];

                    for (int i = 0; i < count; i++)
                    {
                        float w = weightsPtr[baseIdx + i];
                        int bone = bonesPtr[baseIdx + i];

                        if (w > 1e-6f && (uint)bone < (uint)skinningMatrices.Length)
                        {
                            Vector4 transformed = Vector4.Transform(
                                new Vector4(pos.X, pos.Y, pos.Z, 1.0f),
                                skinningMatrices[bone]);

                            skinned += new Vector3(transformed.X, transformed.Y, transformed.Z) * w;
                        }
                    }

                    outputPositions[v] = skinned;
                }
            }
        }

        /// <summary>
        /// Computes skinned vertex normals by multiplying each vertex's bind-pose normal
        /// by the skinning matrix contributions (ignoring translation).
        /// </summary>
        /// <param name="bindNormals">Bind-pose vertex normals.</param>
        /// <param name="skinningMatrices">Skinning matrices per bone.</param>
        /// <param name="outputNormals">Output skinned normals.</param>
        public unsafe void ComputeSkinnedNormals(
            ReadOnlySpan<Vector3> bindNormals,
            ReadOnlySpan<Matrix4x4> skinningMatrices,
            Span<Vector3> outputNormals)
        {
            if (bindNormals.Length < _vertexCount)
                throw new ArgumentException("Bind normals too short.");
            if (outputNormals.Length < _vertexCount)
                throw new ArgumentException("Output normals too short.");

            fixed (float* weightsPtr = _weights)
            fixed (int* bonesPtr = _boneIndices)
            fixed (int* countsPtr = _influenceCounts)
            {
                for (int v = 0; v < _vertexCount; v++)
                {
                    Vector3 normal = bindNormals[v];
                    Vector3 skinned = Vector3.Zero;
                    int baseIdx = v * _maxInfluences;
                    int count = countsPtr[v];

                    for (int i = 0; i < count; i++)
                    {
                        float w = weightsPtr[baseIdx + i];
                        int bone = bonesPtr[baseIdx + i];

                        if (w > 1e-6f && (uint)bone < (uint)skinningMatrices.Length)
                        {
                            Matrix4x4 m = skinningMatrices[bone];
                            Vector3 transformed = Vector3.TransformNormal(normal, m);
                            skinned += transformed * w;
                        }
                    }

                    float len = skinned.Length();
                    outputNormals[v] = len > 1e-6f ? skinned / len : Vector3.UnitY;
                }
            }
        }

        /// <summary>
        /// Computes skinned positions and normals in a single batch pass.
        /// </summary>
        /// <param name="bindPositions">Bind-pose vertex positions.</param>
        /// <param name="bindNormals">Bind-pose vertex normals.</param>
        /// <param name="skinningMatrices">Skinning matrices per bone.</param>
        /// <param name="outputPositions">Output skinned positions.</param>
        /// <param name="outputNormals">Output skinned normals.</param>
        public unsafe void ComputeSkinnedPositionsAndNormals(
            ReadOnlySpan<Vector3> bindPositions,
            ReadOnlySpan<Vector3> bindNormals,
            ReadOnlySpan<Matrix4x4> skinningMatrices,
            Span<Vector3> outputPositions,
            Span<Vector3> outputNormals)
        {
            if (bindPositions.Length < _vertexCount)
                throw new ArgumentException("Bind positions too short.");
            if (bindNormals.Length < _vertexCount)
                throw new ArgumentException("Bind normals too short.");
            if (outputPositions.Length < _vertexCount)
                throw new ArgumentException("Output positions too short.");
            if (outputNormals.Length < _vertexCount)
                throw new ArgumentException("Output normals too short.");

            fixed (float* weightsPtr = _weights)
            fixed (int* bonesPtr = _boneIndices)
            fixed (int* countsPtr = _influenceCounts)
            {
                for (int v = 0; v < _vertexCount; v++)
                {
                    Vector3 pos = bindPositions[v];
                    Vector3 normal = bindNormals[v];
                    Vector3 skinnedPos = Vector3.Zero;
                    Vector3 skinnedNorm = Vector3.Zero;
                    int baseIdx = v * _maxInfluences;
                    int count = countsPtr[v];

                    for (int i = 0; i < count; i++)
                    {
                        float w = weightsPtr[baseIdx + i];
                        int bone = bonesPtr[baseIdx + i];

                        if (w > 1e-6f && (uint)bone < (uint)skinningMatrices.Length)
                        {
                            Matrix4x4 m = skinningMatrices[bone];

                            Vector4 transformedPos = Vector4.Transform(
                                new Vector4(pos.X, pos.Y, pos.Z, 1.0f), m);
                            skinnedPos += new Vector3(transformedPos.X, transformedPos.Y, transformedPos.Z) * w;

                            Vector3 transformedNorm = Vector3.TransformNormal(normal, m);
                            skinnedNorm += transformedNorm * w;
                        }
                    }

                    outputPositions[v] = skinnedPos;

                    float normLen = skinnedNorm.Length();
                    outputNormals[v] = normLen > 1e-6f ? skinnedNorm / normLen : Vector3.UnitY;
                }
            }
        }

        /// <summary>
        /// Computes skinned positions using SIMD-friendly batched processing.
        /// Processes 4 vertices at a time using Vector operations where possible.
        /// </summary>
        public unsafe void ComputeSkinnedPositionsBatched(
            ReadOnlySpan<Vector3> bindPositions,
            ReadOnlySpan<Matrix4x4> skinningMatrices,
            Span<Vector3> outputPositions)
        {
            if (bindPositions.Length < _vertexCount)
                throw new ArgumentException("Bind positions too short.");
            if (outputPositions.Length < _vertexCount)
                throw new ArgumentException("Output positions too short.");

            fixed (float* weightsPtr = _weights)
            fixed (int* bonesPtr = _boneIndices)
            fixed (int* countsPtr = _influenceCounts)
            {
                int v = 0;
                int batchEnd = _vertexCount - 3;

                for (; v < batchEnd; v += 4)
                {
                    Vector3 s0 = Vector3.Zero, s1 = Vector3.Zero, s2 = Vector3.Zero, s3 = Vector3.Zero;

                    for (int i = 0; i < _maxInfluences; i++)
                    {
                        float w0 = weightsPtr[v * _maxInfluences + i];
                        float w1 = weightsPtr[(v + 1) * _maxInfluences + i];
                        float w2 = weightsPtr[(v + 2) * _maxInfluences + i];
                        float w3 = weightsPtr[(v + 3) * _maxInfluences + i];

                        int b0 = bonesPtr[v * _maxInfluences + i];
                        int b1 = bonesPtr[(v + 1) * _maxInfluences + i];
                        int b2 = bonesPtr[(v + 2) * _maxInfluences + i];
                        int b3 = bonesPtr[(v + 3) * _maxInfluences + i];

                        if (w0 > 1e-6f && (uint)b0 < (uint)skinningMatrices.Length)
                        {
                            var t = Vector4.Transform(new Vector4(bindPositions[v], 1f), skinningMatrices[b0]);
                            s0 += new Vector3(t.X, t.Y, t.Z) * w0;
                        }
                        if (w1 > 1e-6f && (uint)b1 < (uint)skinningMatrices.Length)
                        {
                            var t = Vector4.Transform(new Vector4(bindPositions[v + 1], 1f), skinningMatrices[b1]);
                            s1 += new Vector3(t.X, t.Y, t.Z) * w1;
                        }
                        if (w2 > 1e-6f && (uint)b2 < (uint)skinningMatrices.Length)
                        {
                            var t = Vector4.Transform(new Vector4(bindPositions[v + 2], 1f), skinningMatrices[b2]);
                            s2 += new Vector3(t.X, t.Y, t.Z) * w2;
                        }
                        if (w3 > 1e-6f && (uint)b3 < (uint)skinningMatrices.Length)
                        {
                            var t = Vector4.Transform(new Vector4(bindPositions[v + 3], 1f), skinningMatrices[b3]);
                            s3 += new Vector3(t.X, t.Y, t.Z) * w3;
                        }
                    }

                    outputPositions[v] = s0;
                    outputPositions[v + 1] = s1;
                    outputPositions[v + 2] = s2;
                    outputPositions[v + 3] = s3;
                }

                for (; v < _vertexCount; v++)
                {
                    Vector3 pos = bindPositions[v];
                    Vector3 skinned = Vector3.Zero;
                    int baseIdx = v * _maxInfluences;
                    int count = countsPtr[v];

                    for (int i = 0; i < count; i++)
                    {
                        float w = weightsPtr[baseIdx + i];
                        int bone = bonesPtr[baseIdx + i];

                        if (w > 1e-6f && (uint)bone < (uint)skinningMatrices.Length)
                        {
                            var t = Vector4.Transform(new Vector4(pos, 1f), skinningMatrices[bone]);
                            skinned += new Vector3(t.X, t.Y, t.Z) * w;
                        }
                    }

                    outputPositions[v] = skinned;
                }
            }
        }

        // ── Validation ───────────────────────────────────────────────────

        /// <summary>
        /// Validates the skinning weight data for consistency.
        /// </summary>
        /// <param name="errorVertex">If validation fails, contains the offending vertex index.</param>
        /// <returns>True if valid; false otherwise.</returns>
        public bool Validate(out int errorVertex)
        {
            errorVertex = -1;

            for (int v = 0; v < _vertexCount; v++)
            {
                int count = _influenceCounts[v];
                if (count < 0 || count > _maxInfluences)
                {
                    errorVertex = v;
                    return false;
                }

                float sum = 0f;
                int baseIdx = v * _maxInfluences;

                for (int i = 0; i < count; i++)
                {
                    if (_weights[baseIdx + i] < 0f)
                    {
                        errorVertex = v;
                        return false;
                    }
                    sum += _weights[baseIdx + i];
                }

                if (count > 0 && Math.Abs(sum - 1.0f) > 0.01f)
                {
                    errorVertex = v;
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Validates and returns a human-readable error description.
        /// </summary>
        public string ValidateDetailed()
        {
            for (int v = 0; v < _vertexCount; v++)
            {
                int count = _influenceCounts[v];
                if (count < 0 || count > _maxInfluences)
                    return $"Vertex {v}: influence count {count} out of range [0, {_maxInfluences}].";

                float sum = 0f;
                int baseIdx = v * _maxInfluences;

                for (int i = 0; i < count; i++)
                {
                    if (_weights[baseIdx + i] < 0f)
                        return $"Vertex {v}, influence {i}: negative weight {_weights[baseIdx + i]}.";
                    sum += _weights[baseIdx + i];
                }

                if (count > 0 && Math.Abs(sum - 1.0f) > 0.01f)
                    return $"Vertex {v}: weights sum to {sum}, expected ~1.0.";
            }

            return "Valid";
        }

        /// <summary>
        /// Disposes this instance and releases all internal arrays.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _weights = Array.Empty<float>();
                _boneIndices = Array.Empty<int>();
                _influenceCounts = Array.Empty<int>();
                _quantizedWeights8 = Array.Empty<byte>();
                _quantizedWeights4 = Array.Empty<byte>();
                _vertexCount = 0;
                _disposed = true;
            }
        }

        // ── Private Helpers ──────────────────────────────────────────────

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ValidateIndices(int vertexIndex, int influenceIndex)
        {
            if ((uint)vertexIndex >= (uint)_vertexCount)
                throw new ArgumentOutOfRangeException(nameof(vertexIndex),
                    $"Vertex index {vertexIndex} out of range [0, {_vertexCount}).");
            if ((uint)influenceIndex >= (uint)_maxInfluences)
                throw new ArgumentOutOfRangeException(nameof(influenceIndex),
                    $"Influence index {influenceIndex} out of range [0, {_maxInfluences}).");
        }
    }
}
