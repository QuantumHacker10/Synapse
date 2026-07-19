using System;
// ============================================================
// FILE: JointTransform.cs
// PATH: Animation/JointTransform.cs
// ============================================================


using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Rendering.Compat;

namespace GDNN.Animation
{
    /// <summary>
    /// Represents the transform of a single joint in TRS (Translation, Rotation, Scale) form.
    /// Provides local/world space conversion, transform stream compression with delta encoding
    /// and quantization, quaternion quantization for network transmission, and interpolation.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct JointTransform : IEquatable<JointTransform>
    {
        /// <summary>Local-space translation.</summary>
        public Vector3 Translation;

        /// <summary>Local-space rotation quaternion.</summary>
        public Quaternion Rotation;

        /// <summary>Local-space scale.</summary>
        public Vector3 Scale;

        /// <summary>Cached world-space translation (computed on demand).</summary>
        internal Vector3 WorldTranslation;

        /// <summary>Cached world-space rotation (computed on demand).</summary>
        internal Quaternion WorldRotation;

        /// <summary>Cached world-space scale (computed on demand).</summary>
        internal Vector3 WorldScale;

        /// <summary>Whether the world-space values are up to date.</summary>
        internal byte _isWorldCachedFlags;

        /// <summary>Size of the compact representation in bytes.</summary>
        public const int CompactSize = 40;

        /// <summary>
        /// Initializes a joint transform with specified local-space TRS values.
        /// </summary>
        /// <param name="translation">Local translation.</param>
        /// <param name="rotation">Local rotation quaternion.</param>
        /// <param name="scale">Local scale.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public JointTransform(Vector3 translation, Quaternion rotation, Vector3 scale)
        {
            Translation = translation;
            Rotation = rotation;
            Scale = scale;
            WorldTranslation = Vector3.Zero;
            WorldRotation = Quaternion.Identity;
            WorldScale = Vector3.One;
            _isWorldCachedFlags = 0;
        }

        /// <summary>
        /// Gets the identity joint transform (zero translation, identity rotation, unit scale).
        /// </summary>
        public static JointTransform Identity => new JointTransform(
            Vector3.Zero, Quaternion.Identity, Vector3.One);

        /// <summary>
        /// Computes the local-space transform matrix from TRS.
        /// </summary>
        public readonly Matrix4x4 LocalMatrix
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                Matrix4x4 m = Matrix4x4.CreateScale(Scale);
                m *= Matrix4x4.CreateFromQuaternion(Rotation);
                m *= Matrix4x4.CreateTranslation(Translation);
                return m;
            }
        }

        /// <summary>
        /// Computes the world-space transform matrix from cached world TRS.
        /// </summary>
        public readonly Matrix4x4 WorldMatrix
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                Matrix4x4 m = Matrix4x4.CreateScale(WorldScale);
                m *= Matrix4x4.CreateFromQuaternion(WorldRotation);
                m *= Matrix4x4.CreateTranslation(WorldTranslation);
                return m;
            }
        }

        /// <summary>
        /// Gets whether the world-space transform is cached and valid.
        /// </summary>
        public bool IsWorldCached => _isWorldCachedFlags != 0;

        /// <summary>
        /// Composes this local transform with a parent's world transform to produce
        /// the world-space transform for this joint.
        /// </summary>
        /// <param name="parentWorld">Parent's world-space transform.</param>
        /// <returns>The composed world-space transform.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public JointTransform ComposeWithParent(JointTransform parentWorld)
        {
            JointTransform result;
            result.Translation = Translation;
            result.Rotation = Rotation;
            result.Scale = Scale;

            result.WorldRotation = Quaternion.Normalize(parentWorld.WorldRotation * Rotation);
            result.WorldScale = parentWorld.WorldScale * Scale;
            result.WorldTranslation = Vector3.Transform(Translation, parentWorld.WorldRotation)
                                    * parentWorld.WorldScale
                                    + parentWorld.WorldTranslation;
            result._isWorldCachedFlags = 1;

            return result;
        }

        /// <summary>
        /// Composes this transform with a parent's world transform, storing the result in-place.
        /// </summary>
        /// <param name="parentWorld">Parent's world-space transform.</param>
        public void ComposeWithParentInPlace(JointTransform parentWorld)
        {
            WorldRotation = Quaternion.Normalize(parentWorld.WorldRotation * Rotation);
            WorldScale = parentWorld.WorldScale * Scale;
            WorldTranslation = Vector3.Transform(Translation, parentWorld.WorldRotation)
                             * parentWorld.WorldScale
                             + parentWorld.WorldTranslation;
            _isWorldCachedFlags = 1;
        }

        /// <summary>
        /// Extracts the local transform from a world transform given the parent's world transform.
        /// </summary>
        /// <param name="world">World-space transform of this joint.</param>
        /// <param name="parentWorld">World-space transform of the parent.</param>
        /// <returns>Local-space transform.</returns>
        public static JointTransform ExtractLocal(JointTransform world, JointTransform parentWorld)
        {
            Quaternion invParentRot = Quaternion.Conjugate(parentWorld.WorldRotation);
            Vector3 invParentScale = new Vector3(
                parentWorld.WorldScale.X > 1e-6f ? 1.0f / parentWorld.WorldScale.X : 0f,
                parentWorld.WorldScale.Y > 1e-6f ? 1.0f / parentWorld.WorldScale.Y : 0f,
                parentWorld.WorldScale.Z > 1e-6f ? 1.0f / parentWorld.WorldScale.Z : 0f);

            Vector3 localTranslation = Vector3.Transform(
                world.WorldTranslation - parentWorld.WorldTranslation, invParentRot) * invParentScale;

            Quaternion localRotation = Quaternion.Normalize(invParentRot * world.WorldRotation);

            Vector3 localScale = new Vector3(
                parentWorld.WorldScale.X > 1e-6f ? world.WorldScale.X / parentWorld.WorldScale.X : 1f,
                parentWorld.WorldScale.Y > 1e-6f ? world.WorldScale.Y / parentWorld.WorldScale.Y : 1f,
                parentWorld.WorldScale.Z > 1e-6f ? world.WorldScale.Z / parentWorld.WorldScale.Z : 1f);

            return new JointTransform(localTranslation, localRotation, localScale);
        }

        // ── Interpolation and Extrapolation ──────────────────────────────

        /// <summary>
        /// Linearly interpolates between two joint transforms.
        /// Translation and scale are lerped; rotation is slerped.
        /// </summary>
        /// <param name="a">Start transform.</param>
        /// <param name="b">End transform.</param>
        /// <param name="t">Interpolation factor [0,1].</param>
        /// <returns>Interpolated transform.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static JointTransform Lerp(JointTransform a, JointTransform b, float t)
        {
            t = Math.Clamp(t, 0f, 1f);
            return new JointTransform(
                Vector3.Lerp(a.Translation, b.Translation, t),
                Quaternion.Slerp(a.Rotation, b.Rotation, t),
                Vector3.Lerp(a.Scale, b.Scale, t));
        }

        /// <summary>
        /// Spherically interpolates between two joint transforms using shortest-path rotation.
        /// </summary>
        public static JointTransform Slerp(JointTransform a, JointTransform b, float t)
        {
            t = Math.Clamp(t, 0f, 1f);

            Quaternion rotA = a.Rotation;
            Quaternion rotB = b.Rotation;

            if (Quaternion.Dot(rotA, rotB) < 0f)
            {
                rotB = -rotB;
            }

            return new JointTransform(
                Vector3.Lerp(a.Translation, b.Translation, t),
                Quaternion.Slerp(rotA, rotB, t),
                Vector3.Lerp(a.Scale, b.Scale, t));
        }

        /// <summary>
        /// Extrapolates a transform beyond the end of two known transforms.
        /// Useful for prediction in networked animation.
        /// </summary>
        /// <param name="previous">Previous frame transform.</param>
        /// <param name="current">Current frame transform.</param>
        /// <param name="t">Extrapolation factor (1.0 = current, >1.0 = beyond).</param>
        /// <returns>Extrapolated transform.</returns>
        public static JointTransform Extrapolate(JointTransform previous, JointTransform current, float t)
        {
            Vector3 deltaTranslation = current.Translation - previous.Translation;
            Vector3 deltaScale = current.Scale - previous.Scale;

            return new JointTransform(
                current.Translation + deltaTranslation * (t - 1.0f),
                current.Rotation,
                current.Scale + deltaScale * (t - 1.0f));
        }

        /// <summary>
        /// Catmull-Rom interpolation between four joint transforms.
        /// </summary>
        /// <param name="t0">Transform before start.</param>
        /// <param name="t1">Start transform.</param>
        /// <param name="t2">End transform.</param>
        /// <param name="t3">Transform after end.</param>
        /// <param name="t">Interpolation factor [0,1].</param>
        /// <returns>Interpolated transform.</returns>
        public static JointTransform CatmullRom(
            JointTransform t0, JointTransform t1,
            JointTransform t2, JointTransform t3, float t)
        {
            float t2_val = t * t;
            float t3_val = t2_val * t;

            Vector3 translation = 0.5f * (
                (2f * t1.Translation) +
                (-t0.Translation + t2.Translation) * t +
                (2f * t0.Translation - 5f * t1.Translation + 4f * t2.Translation - t3.Translation) * t2_val +
                (-t0.Translation + 3f * t1.Translation - 3f * t2.Translation + t3.Translation) * t3_val);

            Vector3 scale = 0.5f * (
                (2f * t1.Scale) +
                (-t0.Scale + t2.Scale) * t +
                (2f * t0.Scale - 5f * t1.Scale + 4f * t2.Scale - t3.Scale) * t2_val +
                (-t0.Scale + 3f * t1.Scale - 3f * t2.Scale + t3.Scale) * t3_val);

            Quaternion rotation = Quaternion.Slerp(
                Quaternion.Slerp(t1.Rotation, t2.Rotation, t),
                Quaternion.Slerp(t0.Rotation, t3.Rotation, t),
                t * (1f - t));

            return new JointTransform(translation, Quaternion.Normalize(rotation), scale);
        }

        // ── Transform Stream Compression ─────────────────────────────────

        /// <summary>
        /// Computes the delta (difference) between this transform and a reference transform.
        /// Used for delta encoding in transform streams.
        /// </summary>
        /// <param name="reference">Reference transform to diff against.</param>
        /// <returns>Delta transform.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public JointTransform Delta(JointTransform reference)
        {
            Quaternion invRefRot = Quaternion.Conjugate(reference.Rotation);
            Quaternion deltaRot = Quaternion.Normalize(Rotation * invRefRot);

            return new JointTransform(
                Translation - reference.Translation,
                deltaRot,
                Scale - reference.Scale);
        }

        /// <summary>
        /// Applies a delta transform to a reference transform to reconstruct the original.
        /// </summary>
        /// <param name="reference">Reference transform.</param>
        /// <param name="delta">Delta transform.</param>
        /// <returns>Reconstructed transform.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static JointTransform ApplyDelta(JointTransform reference, JointTransform delta)
        {
            return new JointTransform(
                reference.Translation + delta.Translation,
                Quaternion.Normalize(delta.Rotation * reference.Rotation),
                reference.Scale + delta.Scale);
        }

        /// <summary>
        /// Quantizes the translation to 16-bit fixed-point representation.
        /// Useful for network transmission with bounded precision.
        /// </summary>
        /// <param name="minBound">Minimum bound of the translation range.</param>
        /// <param name="maxBound">Maximum bound of the translation range.</param>
        /// <returns>Packed 64-bit value (3 x 16-bit channels).</returns>
        public readonly ulong QuantizeTranslation16(Vector3 minBound, Vector3 maxBound)
        {
            Vector3 range = maxBound - minBound;
            Vector3 invRange = new Vector3(
                range.X > 1e-6f ? 1.0f / range.X : 0f,
                range.Y > 1e-6f ? 1.0f / range.Y : 0f,
                range.Z > 1e-6f ? 1.0f / range.Z : 0f);

            Vector3 normalized = (Translation - minBound) * invRange;
            normalized = Vector3.Clamp(normalized, Vector3.Zero, Vector3.One);

            ushort x = (ushort)(normalized.X * 65535.0f + 0.5f);
            ushort y = (ushort)(normalized.Y * 65535.0f + 0.5f);
            ushort z = (ushort)(normalized.Z * 65535.0f + 0.5f);

            return (ulong)x | ((ulong)y << 16) | ((ulong)z << 32);
        }

        /// <summary>
        /// Dequantizes a 16-bit packed translation value.
        /// </summary>
        /// <param name="packed">Packed 64-bit value.</param>
        /// <param name="minBound">Minimum bound.</param>
        /// <param name="maxBound">Maximum bound.</param>
        /// <returns>Dequantized translation.</returns>
        public static Vector3 DequantizeTranslation16(ulong packed, Vector3 minBound, Vector3 maxBound)
        {
            ushort x = (ushort)(packed & 0xFFFF);
            ushort y = (ushort)((packed >> 16) & 0xFFFF);
            ushort z = (ushort)((packed >> 32) & 0xFFFF);

            Vector3 normalized = new Vector3(x / 65535.0f, y / 65535.0f, z / 65535.0f);
            return minBound + (maxBound - minBound) * normalized;
        }

        /// <summary>
        /// Quantizes the scale to 16-bit fixed-point representation.
        /// </summary>
        /// <param name="maxScale">Maximum scale value (symmetric range [-max, max]).</param>
        /// <returns>Packed 48-bit value (3 x 16-bit channels).</returns>
        public readonly ulong QuantizeScale16(float maxScale)
        {
            float invMax = maxScale > 1e-6f ? 1.0f / maxScale : 0f;

            float nx = (Scale.X * invMax + 1.0f) * 0.5f;
            float ny = (Scale.Y * invMax + 1.0f) * 0.5f;
            float nz = (Scale.Z * invMax + 1.0f) * 0.5f;

            nx = Math.Clamp(nx, 0f, 1f);
            ny = Math.Clamp(ny, 0f, 1f);
            nz = Math.Clamp(nz, 0f, 1f);

            ushort x = (ushort)(nx * 65535.0f + 0.5f);
            ushort y = (ushort)(ny * 65535.0f + 0.5f);
            ushort z = (ushort)(nz * 65535.0f + 0.5f);

            return (ulong)x | ((ulong)y << 16) | ((ulong)z << 32);
        }

        /// <summary>
        /// Dequantizes a 16-bit packed scale value.
        /// </summary>
        public static Vector3 DequantizeScale16(ulong packed, float maxScale)
        {
            ushort x = (ushort)(packed & 0xFFFF);
            ushort y = (ushort)((packed >> 16) & 0xFFFF);
            ushort z = (ushort)((packed >> 32) & 0xFFFF);

            Vector3 normalized = new Vector3(x / 65535.0f, y / 65535.0f, z / 65535.0f);
            return (normalized * 2.0f - Vector3.One) * maxScale;
        }

        // ── Quaternion Quantization ──────────────────────────────────────

        /// <summary>
        /// Quantizes a quaternion to the smallest-three representation.
        /// Stores the index of the largest component (2 bits) and the other three components
        /// as 10-bit signed values. Total: 32 bits.
        /// </summary>
        /// <returns>Packed 32-bit quantized quaternion.</returns>
        public readonly uint QuantizeRotation32()
        {
            Quaternion q = Quaternion.Normalize(Rotation);

            float absX = Math.Abs(q.X);
            float absY = Math.Abs(q.Y);
            float absZ = Math.Abs(q.Z);
            float absW = Math.Abs(q.W);

            int largestIndex = 0;
            float largestValue = absX;
            if (absY > largestValue)
            { largestIndex = 1; largestValue = absY; }
            if (absZ > largestValue)
            { largestIndex = 2; largestValue = absZ; }
            if (absW > largestValue)
            { largestIndex = 3; largestValue = absW; }

            float a, b, c;
            switch (largestIndex)
            {
                case 0:
                    a = q.Y;
                    b = q.Z;
                    c = q.W;
                    break;
                case 1:
                    a = q.X;
                    b = q.Z;
                    c = q.W;
                    break;
                case 2:
                    a = q.X;
                    b = q.Y;
                    c = q.W;
                    break;
                default:
                    a = q.X;
                    b = q.Y;
                    c = q.Z;
                    break;
            }

            float sqrt2Over2 = 0.70710678f;
            int quantA = (int)Math.Clamp((a / sqrt2Over2 + 1.0f) * 511.5f, 0, 1023);
            int quantB = (int)Math.Clamp((b / sqrt2Over2 + 1.0f) * 511.5f, 0, 1023);
            int quantC = (int)Math.Clamp((c / sqrt2Over2 + 1.0f) * 511.5f, 0, 1023);

            return (uint)((largestIndex << 30) | (quantA << 20) | (quantB << 10) | quantC);
        }

        /// <summary>
        /// Dequantizes a 32-bit smallest-three quaternion.
        /// </summary>
        /// <param name="packed">Packed 32-bit quaternion.</param>
        /// <returns>Dequantized quaternion.</returns>
        public static Quaternion DequantizeRotation32(uint packed)
        {
            int largestIndex = (int)(packed >> 30);
            int quantA = (int)((packed >> 20) & 0x3FF);
            int quantB = (int)((packed >> 10) & 0x3FF);
            int quantC = (int)(packed & 0x3FF);

            float sqrt2Over2 = 0.70710678f;
            float a = (quantA / 1023.0f - 0.5f) * 2.0f * sqrt2Over2;
            float b = (quantB / 1023.0f - 0.5f) * 2.0f * sqrt2Over2;
            float c = (quantC / 1023.0f - 0.5f) * 2.0f * sqrt2Over2;

            float largestComponent = (float)Math.Sqrt(
                Math.Max(0.0, 1.0 - a * a - b * b - c * c));

            Quaternion result;
            switch (largestIndex)
            {
                case 0:
                    result = new Quaternion(largestComponent, a, b, c);
                    break;
                case 1:
                    result = new Quaternion(a, largestComponent, b, c);
                    break;
                case 2:
                    result = new Quaternion(a, b, largestComponent, c);
                    break;
                default:
                    result = new Quaternion(a, b, c, largestComponent);
                    break;
            }

            return Quaternion.Normalize(result);
        }

        /// <summary>
        /// Quantizes rotation to 64-bit for higher precision.
        /// Uses 16-bit per component for 4 quaternion components.
        /// </summary>
        /// <returns>Packed 64-bit quaternion.</returns>
        public readonly ulong QuantizeRotation64()
        {
            Quaternion q = Quaternion.Normalize(Rotation);

            short PackComponent(float v)
            {
                return (short)Math.Clamp((int)(v * 32767.0f + (v >= 0 ? 0.5f : -0.5f)), -32768, 32767);
            }

            ushort x = (ushort)PackComponent(q.X);
            ushort y = (ushort)PackComponent(q.Y);
            ushort z = (ushort)PackComponent(q.Z);
            ushort w = (ushort)PackComponent(q.W);

            return (ulong)x | ((ulong)y << 16) | ((ulong)z << 32) | ((ulong)w << 48);
        }

        /// <summary>
        /// Dequantizes a 64-bit quaternion.
        /// </summary>
        public static Quaternion DequantizeRotation64(ulong packed)
        {
            short UnpackComponent(ushort v)
            {
                return (short)v;
            }

            float x = UnpackComponent((ushort)(packed & 0xFFFF)) / 32767.0f;
            float y = UnpackComponent((ushort)((packed >> 16) & 0xFFFF)) / 32767.0f;
            float z = UnpackComponent((ushort)((packed >> 32) & 0xFFFF)) / 32767.0f;
            float w = UnpackComponent((ushort)((packed >> 48) & 0xFFFF)) / 32767.0f;

            return Quaternion.Normalize(new Quaternion(x, y, z, w));
        }

        // ── Serialization ────────────────────────────────────────────────

        /// <summary>
        /// Serializes the joint transform to a compact byte buffer (40 bytes).
        /// </summary>
        public readonly byte[] Serialize()
        {
            byte[] buffer = new byte[CompactSize];
            SerializeTo(buffer);
            return buffer;
        }

        /// <summary>
        /// Serializes the joint transform into the provided buffer.
        /// </summary>
        /// <param name="buffer">Target buffer (must be >= CompactSize bytes).</param>
        public readonly void SerializeTo(Span<byte> buffer)
        {
            if (buffer.Length < CompactSize)
                throw new ArgumentException($"Buffer must be at least {CompactSize} bytes.");

            MemoryMarshal.Write(buffer, ref Unsafe.AsRef(in this));
        }

        /// <summary>
        /// Deserializes a joint transform from a byte buffer.
        /// </summary>
        /// <param name="data">Source data (must be >= CompactSize bytes).</param>
        /// <returns>Deserialized joint transform.</returns>
        public static JointTransform Deserialize(ReadOnlySpan<byte> data)
        {
            if (data.Length < CompactSize)
                throw new ArgumentException($"Data must be at least {CompactSize} bytes.");

            return MemoryMarshal.Read<JointTransform>(data);
        }

        // ── Equality and Comparison ──────────────────────────────────────

        /// <summary>
        /// Determines whether two transforms are approximately equal.
        /// </summary>
        /// <param name="other">Other transform.</param>
        /// <param name="tolerance">Tolerance for comparison.</param>
        /// <returns>True if approximately equal.</returns>
        public readonly bool ApproxEquals(JointTransform other, float tolerance = 1e-5f)
        {
            return Vector3.Distance(Translation, other.Translation) < tolerance
                && Vector3.Distance(Scale, other.Scale) < tolerance
                && Math.Abs(Math.Abs(Quaternion.Dot(Rotation, other.Rotation)) - 1.0f) < tolerance;
        }

        /// <inheritdoc/>
        public readonly bool Equals(JointTransform other)
        {
            return Translation.Equals(other.Translation)
                && Rotation.Equals(other.Rotation)
                && Scale.Equals(other.Scale);
        }

        /// <inheritdoc/>
        public override readonly bool Equals(object? obj) => obj is JointTransform other && Equals(other);

        /// <inheritdoc/>
        public override readonly int GetHashCode() => HashCode.Combine(Translation, Rotation, Scale);

        /// <summary>Equality operator.</summary>
        public static bool operator ==(JointTransform left, JointTransform right) => left.Equals(right);

        /// <summary>Inequality operator.</summary>
        public static bool operator !=(JointTransform left, JointTransform right) => !left.Equals(right);

        /// <summary>
        /// Linear interpolation operator.
        /// </summary>
        public static JointTransform operator +(JointTransform a, JointTransform b)
        {
            return new JointTransform(
                a.Translation + b.Translation,
                Quaternion.Normalize(a.Rotation * b.Rotation),
                a.Scale + b.Scale);
        }

        /// <summary>
        /// Scalar multiplication operator.
        /// </summary>
        public static JointTransform operator *(JointTransform t, float s)
        {
            return new JointTransform(
                t.Translation * s,
                t.Rotation,
                Vector3.Lerp(Vector3.One, t.Scale, s));
        }

        /// <summary>
        /// Returns a string representation of the transform.
        /// </summary>
        public override readonly string ToString()
        {
            return $"T({Translation.X:F3},{Translation.Y:F3},{Translation.Z:F3}) " +
                   $"R({Rotation.X:F3},{Rotation.Y:F3},{Rotation.Z:F3},{Rotation.W:F3}) " +
                   $"S({Scale.X:F3},{Scale.Y:F3},{Scale.Z:F3})";
        }

        // ── Static Factory Methods ───────────────────────────────────────

        /// <summary>
        /// Creates a JointTransform from a 4x4 matrix by decomposing into TRS.
        /// </summary>
        /// <param name="matrix">Matrix to decompose.</param>
        /// <returns>Decomposed JointTransform.</returns>
        public static JointTransform FromMatrix(Matrix4x4 matrix)
        {
            Vector3 translation = matrix.Translation;
            Vector3 scale = Matrix4x4Compat.GetScale(matrix);
            Matrix4x4 rotationMatrix = Matrix4x4.CreateScale(
                new Vector3(
                    scale.X > 1e-6f ? 1.0f / scale.X : 0f,
                    scale.Y > 1e-6f ? 1.0f / scale.Y : 0f,
                    scale.Z > 1e-6f ? 1.0f / scale.Z : 0f)) * matrix;

            Quaternion rotation = Quaternion.CreateFromRotationMatrix(rotationMatrix);
            return new JointTransform(translation, Quaternion.Normalize(rotation), scale);
        }

        /// <summary>
        /// Creates a JointTransform from Euler angles (in degrees).
        /// </summary>
        /// <param name="translation">Translation.</param>
        /// <param name="eulerDegrees">Euler angles in degrees (X, Y, Z).</param>
        /// <param name="scale">Scale.</param>
        public static JointTransform FromEulerAngles(Vector3 translation, Vector3 eulerDegrees, Vector3 scale)
        {
            float deg2Rad = MathF.PI / 180f;
            Quaternion rotation = Quaternion.CreateFromYawPitchRoll(
                eulerDegrees.Y * deg2Rad,
                eulerDegrees.X * deg2Rad,
                eulerDegrees.Z * deg2Rad);

            return new JointTransform(translation, rotation, scale);
        }

        /// <summary>
        /// Gets the Euler angles (in degrees) of this transform's rotation.
        /// </summary>
        public readonly Vector3 ToEulerAngles()
        {
            float rad2Deg = 180f / MathF.PI;

            double sinX = 2.0 * (Rotation.W * Rotation.X + Rotation.Y * Rotation.Z);
            double cosX = 1.0 - 2.0 * (Rotation.X * Rotation.X + Rotation.Y * Rotation.Y);
            float pitch = (float)Math.Atan2(sinX, cosX) * rad2Deg;

            double sinY = 2.0 * (Rotation.W * Rotation.Y - Rotation.Z * Rotation.X);
            float yaw = Math.Abs(sinY) >= 1.0
                ? (float)Math.CopySign(90.0, sinY) * rad2Deg
                : (float)Math.Asin(sinY) * rad2Deg;

            double sinZ = 2.0 * (Rotation.W * Rotation.Z + Rotation.X * Rotation.Y);
            double cosZ = 1.0 - 2.0 * (Rotation.Y * Rotation.Y + Rotation.Z * Rotation.Z);
            float roll = (float)Math.Atan2(sinZ, cosZ) * rad2Deg;

            return new Vector3(pitch, yaw, roll);
        }

        /// <summary>
        /// Blends two transforms additively.
        /// </summary>
        /// <param name="base">Base transform.</param>
        /// <param name="additive">Additive transform.</param>
        /// <param name="weight">Blend weight.</param>
        /// <returns>Result transform.</returns>
        public static JointTransform BlendAdditive(JointTransform @base, JointTransform additive, float weight)
        {
            Quaternion additiveRotation = Quaternion.Slerp(Quaternion.Identity, additive.Rotation, weight);

            return new JointTransform(
                @base.Translation + additive.Translation * weight,
                Quaternion.Normalize(additiveRotation * @base.Rotation),
                @base.Scale + (additive.Scale - Vector3.One) * weight);
        }
    }
}
