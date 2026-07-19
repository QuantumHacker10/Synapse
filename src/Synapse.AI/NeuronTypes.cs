using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace GDNN.Core.Neurons
{
    #region Records and Interfaces

    public readonly record struct NeuronInput
    {
        public Vector3 Position { get; init; }
        public Vector3 Normal { get; init; }
        public float Curvature { get; init; }
        public float DistanceToCamera { get; init; }
        public float Time { get; init; }
        public float DeltaTime { get; init; }
        public ImmutableDictionary<string, float> Parameters { get; init; }
        public ImmutableArray<float> DataChannels { get; init; }
        public Vector2 TextureCoordinate { get; init; }
        public Vector3 Tangent { get; init; }
        public Vector3 Bitangent { get; init; }
        public Vector3 ViewDirection { get; init; }
        public Vector3 LightDirection { get; init; }
        public Vector3 PreviousPosition { get; init; }

        public static NeuronInput Default => new()
        {
            Position = Vector3.Zero,
            Normal = Vector3.UnitY,
            Curvature = 0f,
            DistanceToCamera = 10f,
            Time = 0f,
            DeltaTime = 1f / 60f,
            Parameters = ImmutableDictionary<string, float>.Empty,
            DataChannels = ImmutableArray<float>.Empty,
            TextureCoordinate = Vector2.Zero,
            Tangent = Vector3.UnitX,
            Bitangent = Vector3.UnitZ,
            ViewDirection = -Vector3.UnitZ,
            LightDirection = Vector3.Normalize(new Vector3(1, 1, 1)),
            PreviousPosition = Vector3.Zero
        };

        public float GetParameter(string name, float defaultValue = 0f)
            => Parameters != null && Parameters.TryGetValue(name, out var v) ? v : defaultValue;
        public float GetChannel(int index)
            => index >= 0 && index < DataChannels.Length ? DataChannels[index] : 0f;
    }

    public readonly record struct NeuronOutput
    {
        public float Value { get; init; }
        public Vector3 Displacement { get; init; }
        public Vector3 Color { get; init; }
        public Vector4 ColorRGBA { get; init; }
        public float Roughness { get; init; }
        public float Metallic { get; init; }
        public float Opacity { get; init; }
        public float Emission { get; init; }
        public Vector3 EmissionColor { get; init; }
        public Vector3 OutputNormal { get; init; }
        public float AmbientOcclusion { get; init; }
        public float Height { get; init; }
        public float Thickness { get; init; }
        public float IOR { get; init; }
        public float Specular { get; init; }
        public float ClearCoat { get; init; }
        public float ClearCoatRoughness { get; init; }
        public float Anisotropy { get; init; }
        public float Subsurface { get; init; }
        public float Sheen { get; init; }
        public float Transmission { get; init; }
        public ImmutableDictionary<string, float> Outputs { get; init; }
        public ImmutableDictionary<string, Vector3> VectorOutputs { get; init; }

        public static NeuronOutput Default => new()
        {
            Value = 0f,
            Displacement = Vector3.Zero,
            Color = Vector3.Zero,
            ColorRGBA = new Vector4(0, 0, 0, 1),
            Roughness = 0.5f,
            Metallic = 0f,
            Opacity = 1f,
            Emission = 0f,
            EmissionColor = Vector3.Zero,
            OutputNormal = Vector3.UnitY,
            AmbientOcclusion = 1f,
            Height = 0f,
            Thickness = 0f,
            IOR = 1.5f,
            Specular = 0.5f,
            Outputs = ImmutableDictionary<string, float>.Empty,
            VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
        };

        public float GetOutput(string name, float defaultValue = 0f)
            => Outputs != null && Outputs.TryGetValue(name, out var v) ? v : defaultValue;
        public Vector3 GetVectorOutput(string name, Vector3? defaultValue = null)
            => VectorOutputs != null && VectorOutputs.TryGetValue(name, out var v) ? v : (defaultValue ?? Vector3.Zero);
    }

    public readonly record struct NeuronGradient
    {
        public float ValueGradient { get; init; }
        public Vector3 DisplacementGradient { get; init; }
        public Vector3 ColorGradient { get; init; }
        public float RoughnessGradient { get; init; }
        public float MetallicGradient { get; init; }
        public float OpacityGradient { get; init; }
        public float EmissionGradient { get; init; }
        public Vector3 EmissionColorGradient { get; init; }
        public float HeightGradient { get; init; }
        public float AmbientOcclusionGradient { get; init; }
        public Vector3 PositionGradient { get; init; }
        public Vector3 NormalGradient { get; init; }
        public float CurvatureGradient { get; init; }
        public float TimeGradient { get; init; }
        public ImmutableDictionary<string, float> ParameterGradients { get; init; }
        public float[] WeightGradients { get; init; }

        public static NeuronGradient Zero => new()
        {
            ValueGradient = 0f,
            DisplacementGradient = Vector3.Zero,
            ColorGradient = Vector3.Zero,
            RoughnessGradient = 0f,
            MetallicGradient = 0f,
            OpacityGradient = 0f,
            EmissionGradient = 0f,
            EmissionColorGradient = Vector3.Zero,
            HeightGradient = 0f,
            AmbientOcclusionGradient = 0f,
            PositionGradient = Vector3.Zero,
            NormalGradient = Vector3.Zero,
            CurvatureGradient = 0f,
            TimeGradient = 0f,
            ParameterGradients = ImmutableDictionary<string, float>.Empty,
            WeightGradients = Array.Empty<float>()
        };
    }

    public interface INeuronKernel
    {
        string Name { get; }
        NeuronKind Kind { get; }
        int Version { get; }
        NeuronOutput Compute(in NeuronInput input);
        NeuronGradient Backpropagate(in NeuronGradient gradient);
        int GetParameterCount();
        long GetMemoryFootprint();
        INeuronKernel Clone();
        string Validate();
        void LoadParameters(ReadOnlySpan<float> parameters);
        void SaveParameters(Span<float> parameters);
        void ResetParameters();
        void ApplyGradient(in NeuronGradient gradient, float learningRate);
    }

    public enum NeuronKind
    {
        SdfPrimitive = 0, DisplacementField = 1, CurvatureModulator = 2,
        ColorField = 3, NormalMap = 4, RoughnessMap = 5, MetallicMap = 6,
        EmissiveMap = 7, OpacityMap = 8, SubsurfaceScattering = 9,
        Tessellation = 10, OcclusionCulling = 11, AnimationCurve = 12,
        ParticleEmitter = 13, FluidSolver = 14, ClothSimulator = 15,
        Voxelizer = 16, MarchingCube = 17, RayMarcher = 18,
        WaveFunction = 19, TensorReshaper = 20, AttentionHead = 21,
        Convolution = 22, PoolingLayer = 23, NormalizationLayer = 24,
        EmbeddingLookup = 25, PositionalEncoder = 26,
        CrossAttentionLayer = 27, SelfAttentionLayer = 28,
        FeedForward = 29, UserDefined = 255
    }

    #endregion

    #region MathHelper

    public static class MathHelper
    {
        public const float Pi = 3.14159265358979323846f;
        public const float TwoPi = 6.28318530717958647692f;
        public const float HalfPi = 1.57079632679489661923f;
        public const float InvPi = 0.31830988618379067154f;
        public const float Epsilon = 1e-7f;
        public const float GoldenRatio = 1.6180339887498948482f;
        public const float Sqrt2 = 1.4142135623730950488f;
        public const float Sqrt3 = 1.7320508075688772935f;
        public const float Deg2Rad = 0.01745329251994329577f;
        public const float Rad2Deg = 57.2957795130823208768f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int FloorToInt(float value) => (int)MathF.Floor(value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int CeilingToInt(float value) => (int)MathF.Ceiling(value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int RoundToInt(float value) => (int)MathF.Round(value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Repeat01(float t, float length)
        {
            if (length <= Epsilon)
                return 0f;
            return t - MathF.Floor(t / length) * length;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Clamp(float value, float min, float max) => value < min ? min : value > max ? max : value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Clamp01(float value) => value < 0f ? 0f : value > 1f ? 1f : value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Clamp(int value, int min, int max) => value < min ? min : value > max ? max : value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Max(float a, float b) => a > b ? a : b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Min(float a, float b) => a < b ? a : b;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Abs(float value) => value < 0f ? -value : value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sign(float value) => value > 0f ? 1f : value < 0f ? -1f : 0f;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sqrt(float value) => MathF.Sqrt(value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Pow(float bv, float e) => MathF.Pow(bv, e);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Exp(float value) => MathF.Exp(value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Log(float value) => MathF.Log(value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Log2(float value) => MathF.Log2(value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Log10(float value) => MathF.Log10(value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sin(float a) => MathF.Sin(a);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Cos(float a) => MathF.Cos(a);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Tan(float a) => MathF.Tan(a);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Asin(float v) => MathF.Asin(v);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Acos(float v) => MathF.Acos(v);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Atan(float v) => MathF.Atan(v);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Atan2(float y, float x) => MathF.Atan2(y, x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Floor(float v) => MathF.Floor(v);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Ceil(float v) => MathF.Ceiling(v);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Round(float v) => MathF.Round(v);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Frac(float v) => v - MathF.Floor(v);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Lerp(float a, float b, float t) => a + (b - a) * t;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Lerp(Vector3 a, Vector3 b, float t) => a + (b - a) * t;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector4 Lerp(Vector4 a, Vector4 b, float t) => a + (b - a) * t;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float InverseLerp(float a, float b, float value) { if (MathF.Abs(b - a) < Epsilon) return 0f; return Clamp01((value - a) / (b - a)); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Remap(float value, float fromMin, float fromMax, float toMin, float toMax) { float t = InverseLerp(fromMin, fromMax, value); return Lerp(toMin, toMax, t); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmoothStep(float edge0, float edge1, float x) { float t = Clamp01((x - edge0) / (edge1 - edge0)); return t * t * (3f - 2f * t); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmootherStep(float edge0, float edge1, float x) { float t = Clamp01((x - edge0) / (edge1 - edge0)); return t * t * t * (t * (6f * t - 15f) + 10f); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Step(float edge, float x) => x >= edge ? 1f : 0f;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Pulse(float edge0, float edge1, float x) => Step(edge0, x) - Step(edge1, x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Min(float a, float b, float c) => MathF.Min(a, MathF.Min(b, c));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Max(float a, float b, float c) => MathF.Max(a, MathF.Max(b, c));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmoothMin(float a, float b, float k) { if (k <= Epsilon) return MathF.Min(a, b); float h = Clamp01(0.5f + 0.5f * (b - a) / k); return Lerp(b, a, h) - k * h * (1f - h); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmoothMax(float a, float b, float k) { if (k <= Epsilon) return MathF.Max(a, b); return -SmoothMin(-a, -b, k); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Wrap(float value, float min, float max) { float range = max - min; return min + ((value - min) % range + range) % range; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float WrapAngle(float angle) => Wrap(angle, -Pi, Pi);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Clamp01Vec(Vector3 v) => new Vector3(Clamp01(v.X), Clamp01(v.Y), Clamp01(v.Z));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 ClampVec(Vector3 v, float min, float max) => new Vector3(Clamp(v.X, min, max), Clamp(v.Y, min, max), Clamp(v.Z, min, max));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Normalize(Vector3 v) { float l = v.Length(); return l > Epsilon ? v / l : Vector3.Zero; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Reflect(Vector3 i, Vector3 n) => i - 2f * Vector3.Dot(i, n) * n;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Refract(Vector3 i, Vector3 n, float eta) { float k = 1f - eta * eta * (1f - Vector3.Dot(n, i) * Vector3.Dot(n, i)); if (k < 0f) return Vector3.Zero; return eta * i - (eta * Vector3.Dot(n, i) + MathF.Sqrt(k)) * n; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 SchlickFresnel(float cosTheta, Vector3 F0) => F0 + (Vector3.One - F0) * Pow(Clamp01(1f - cosTheta), 5f);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SchlickFresnel(float cosTheta, float ior) { float r0 = (1f - ior) / (1f + ior); r0 *= r0; return r0 + (1f - r0) * Pow(Clamp01(1f - cosTheta), 5f); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DielectricFresnel(float cosTheta, float ior) { float s2 = 1f - cosTheta * cosTheta; float n2 = ior * ior; float ct = Sqrt(MathF.Max(0f, (n2 - s2) / n2)); float rs = (cosTheta - ior * ct) / (cosTheta + ior * ct); float rp = (ior * cosTheta - ct) / (ior * cosTheta + ct); return (rs * rs + rp * rp) * 0.5f; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float GeometrySchlickGGX(float NdotV, float roughness) { float r = roughness + 1f; float k = (r * r) / 8f; return NdotV / (NdotV * (1f - k) + k + Epsilon); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float GeometrySmithGGX(float NdotV, float NdotL, float roughness) => GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DistributionGGX(float NdotH, float roughness) { float a = roughness * roughness; float a2 = a * a; float d = NdotH * NdotH * (a2 - 1f) + 1f; return a2 / (Pi * d * d + Epsilon); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DistributionBeckmann(float NdotH, float roughness) { float a2 = roughness * roughness * roughness * roughness; float d = Pi * NdotH * NdotH * NdotH * NdotH * (a2 - 1f) + 1f; return a2 / (Pi * d * d + Epsilon); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DistributionCharlie(float NdotH, float roughness) { float invA = 1f / (roughness * roughness + Epsilon); float cos2 = NdotH * NdotH; float sin2 = MathF.Max(1f - cos2, 0f); return (2f + invA) * MathF.Pow(sin2, invA * 0.5f) / (2f * Pi); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 RGBToHSV(Vector3 rgb)
        {
            float r = rgb.X, g = rgb.Y, b = rgb.Z;
            float mx = MathF.Max(r, MathF.Max(g, b)), mn = MathF.Min(r, MathF.Min(g, b));
            float d = mx - mn, h = 0f;
            if (d > Epsilon)
            { if (MathF.Abs(mx - r) < Epsilon) h = ((g - b) / d) % 6f; else if (MathF.Abs(mx - g) < Epsilon) h = (b - r) / d + 2f; else h = (r - g) / d + 4f; h *= 60f; if (h < 0f) h += 360f; }
            return new Vector3(h / 360f, mx > Epsilon ? d / mx : 0f, mx);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 HSVToRGB(Vector3 hsv)
        {
            float h = hsv.X * 360f, s = hsv.Y, v = hsv.Z, c = v * s;
            float x = c * (1f - MathF.Abs((h / 60f) % 2f - 1f)), m = v - c;
            float r, g, b;
            if (h < 60)
            { r = c; g = x; b = 0; }
            else if (h < 120)
            { r = x; g = c; b = 0; }
            else if (h < 180)
            { r = 0; g = c; b = x; }
            else if (h < 240)
            { r = 0; g = x; b = c; }
            else if (h < 300)
            { r = x; g = 0; b = c; }
            else
            { r = c; g = 0; b = x; }
            return new Vector3(r + m, g + m, b + m);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Luminance(Vector3 color) => Vector3.Dot(color, new Vector3(0.2126f, 0.7152f, 0.0722f));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 LinearToSRGB(Vector3 l) => new Vector3(l.X <= 0.0031308f ? 12.92f * l.X : 1.055f * MathF.Pow(l.X, 1f / 2.4f) - 0.055f, l.Y <= 0.0031308f ? 12.92f * l.Y : 1.055f * MathF.Pow(l.Y, 1f / 2.4f) - 0.055f, l.Z <= 0.0031308f ? 12.92f * l.Z : 1.055f * MathF.Pow(l.Z, 1f / 2.4f) - 0.055f);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 SRGBToLinear(Vector3 s) => new Vector3(s.X <= 0.04045f ? s.X / 12.92f : MathF.Pow((s.X + 0.055f) / 1.055f, 2.4f), s.Y <= 0.04045f ? s.Y / 12.92f : MathF.Pow((s.Y + 0.055f) / 1.055f, 2.4f), s.Z <= 0.04045f ? s.Z / 12.92f : MathF.Pow((s.Z + 0.055f) / 1.055f, 2.4f));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 ToneMapReinhard(Vector3 c) { float wp = 4f; return c * (Vector3.One + c / (wp * wp)) / (Vector3.One + c); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 ToneMapACES(Vector3 c) { float a = 2.51f, b = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f; return Clamp01Vec((c * (a * c + new Vector3(b))) / (c * (cc * c + new Vector3(d)) + new Vector3(e))); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Saturate(float x) => Clamp01(x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 ProjectOnPlane(Vector3 v, Vector3 n) => v - Vector3.Dot(v, n) * n;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Barycentric(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
        {
            Vector2 v0 = b - a, v1 = c - a, v2 = p - a;
            float d00 = Vector2.Dot(v0, v0), d01 = Vector2.Dot(v0, v1), d11 = Vector2.Dot(v1, v1);
            float d20 = Vector2.Dot(v2, v0), d21 = Vector2.Dot(v2, v1);
            float denom = d00 * d11 - d01 * d01;
            float v = (d11 * d20 - d01 * d21) / denom;
            float w = (d00 * d21 - d01 * d20) / denom;
            return new Vector3(1f - v - w, v, w);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CreateOrthonormalBasis(Vector3 normal, out Vector3 tangent, out Vector3 bitangent)
        {
            Vector3 up = MathF.Abs(normal.Y) < 0.999f ? Vector3.UnitY : Vector3.UnitX;
            tangent = Vector3.Normalize(Vector3.Cross(up, normal));
            bitangent = Vector3.Cross(normal, tangent);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CubicInterpolate(float v0, float v1, float v2, float v3, float t)
        {
            float t2 = t * t, t3 = t2 * t;
            return 0.5f * ((2f * v1) + (-v0 + v2) * t + (2f * v0 - 5f * v1 + 4f * v2 - v3) * t2 + (-v0 + 3f * v1 - 3f * v2 + v3) * t3);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float TrilinearInterpolate(float v000, float v100, float v010, float v110, float v001, float v101, float v011, float v111, float tx, float ty, float tz)
        {
            float i1 = Lerp(v000, v100, tx), i2 = Lerp(v010, v110, tx);
            float i3 = Lerp(v001, v101, tx), i4 = Lerp(v011, v111, tx);
            return Lerp(Lerp(i1, i2, ty), Lerp(i3, i4, ty), tz);
        }
    }

    #endregion

    #region SdfOperations

    public static class SdfOperations
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sphere(Vector3 p, float r) => p.Length() - r;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Box(Vector3 p, Vector3 b)
        {
            Vector3 q = new Vector3(MathF.Abs(p.X), MathF.Abs(p.Y), MathF.Abs(p.Z)) - b;
            float outerDist = MathHelper.Max(q.X, MathHelper.Max(q.Y, q.Z));
            float innerDist = MathHelper.Min(MathHelper.Max(q.X, MathHelper.Max(q.Y, q.Z)), 0f);
            return outerDist + innerDist;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float RoundedBox(Vector3 p, Vector3 b, float r)
        {
            Vector3 q = new Vector3(MathF.Abs(p.X), MathF.Abs(p.Y), MathF.Abs(p.Z)) - b;
            return MathHelper.Max(q.X, MathHelper.Max(q.Y, q.Z)) + MathHelper.Min(MathHelper.Max(q.X, MathHelper.Max(q.Y, q.Z)), 0f) - r;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Torus(Vector3 p, float R, float r)
        {
            Vector2 q = new Vector2(new Vector2(p.X, p.Z).Length() - R, p.Y);
            return q.Length() - r;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Cylinder(Vector3 p, float r, float h)
        {
            float d = new Vector2(p.X, p.Z).Length() - r;
            float w = MathF.Abs(p.Y) - h;
            return MathHelper.Min(MathHelper.Max(d, w), 0f) + new Vector2(MathHelper.Max(d, 0f), MathHelper.Max(w, 0f)).Length();
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float RoundedCylinder(Vector3 p, float ra, float rb, float h)
        {
            Vector2 d = new Vector2(new Vector2(p.X, p.Z).Length() - 2f * rb + ra, MathF.Abs(p.Y) - h);
            return d.Length() - ra;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Cone(Vector3 p, float r, float h)
        {
            float q = new Vector2(p.X, p.Z).Length();
            Vector2 v2 = new Vector2(h, r);
            Vector2 w = new Vector2(MathF.Abs(p.Y), q);
            Vector2 t = w - v2 * MathHelper.Clamp01(Vector2.Dot(w, v2) / Vector2.Dot(v2, v2));
            float s = (t.X > 0f && t.Y > 0f) ? 1f : -1f;
            return MathF.Sign(s) * t.Length();
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float RoundedCone(Vector3 p, float r1, float r2, float h)
        {
            float q = MathF.Sqrt(p.X * p.X + p.Z * p.Z);
            if (MathF.Abs(p.Y) > h)
                return new Vector2(q, MathF.Abs(p.Y) - h).Length() - r2;
            float a = (r1 - r2) / h;
            float d = q - (a * MathF.Abs(p.Y) + r1);
            float w = new Vector2(d, MathF.Abs(p.Y) * MathF.Sqrt(a * a + 1f)).Length();
            return w * MathHelper.Sign(d);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Capsule(Vector3 p, Vector3 a, Vector3 b, float r)
        {
            Vector3 pa = p - a, ba = b - a;
            float t = MathHelper.Clamp01(Vector3.Dot(pa, ba) / Vector3.Dot(ba, ba));
            return (pa + ba * (t - 1f)).Length() - r;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Octahedron(Vector3 p, float s)
        {
            p = new Vector3(MathF.Abs(p.X), MathF.Abs(p.Y), MathF.Abs(p.Z));
            float m = p.X + p.Y + p.Z - s;
            Vector3 q = p;
            if (3f * p.X < m)
            { }
            else if (3f * p.Y < m)
                q = new Vector3(p.Y, p.Z, p.X);
            else if (3f * p.Z < m)
                q = new Vector3(p.Z, p.X, p.Y);
            else
                return m * 0.57735027f;
            float k = MathHelper.Clamp01(0.5f * (q.Z - q.Y + s));
            q.Z -= k;
            q.Y += k;
            q.Z -= s;
            q.Y -= s;
            return -new Vector2(MathF.Sqrt(q.X * q.X + q.Z * q.Z), q.Y).Length() * MathHelper.Sign(q.Z);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Plane(Vector3 p, Vector3 n, float h) => Vector3.Dot(p, n) + h;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Ellipsoid(Vector3 p, Vector3 r)
        {
            float k0 = p.Length() / r.Length();
            float k1 = (p / r).Length();
            return k0 * (k0 - 1f) / k1;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LineSegment(Vector3 p, Vector3 a, Vector3 b)
        {
            Vector3 pa = p - a, ba = b - a;
            float t = MathHelper.Clamp01(Vector3.Dot(pa, ba) / Vector3.Dot(ba, ba));
            return (pa - ba * t).Length();
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Triangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
        {
            Vector3 e0 = b - a, e1 = c - a, v = p - a;
            Vector3 n = Vector3.Cross(e0, e1);
            float a2 = Vector3.Dot(n, n);
            if (a2 < MathHelper.Epsilon)
                return LineSegment(p, a, b);
            float b0 = Vector3.Dot(Vector3.Cross(v, e1), n) / a2;
            float b1 = Vector3.Dot(Vector3.Cross(e0, v), n) / a2;
            float b2 = 1f - b0 - b1;
            if (b0 >= 0f && b1 >= 0f && b2 >= 0f)
                return MathF.Abs(Vector3.Dot(v, n)) / MathF.Sqrt(a2);
            return MathHelper.Min(
                LineSegment(p, a, b),
                MathHelper.Min(LineSegment(p, b, c), LineSegment(p, c, a)));
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Onion(float d, float thickness) => MathF.Abs(d) - thickness;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Round(float d, float r) => d - r;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Union(float d1, float d2) => MathHelper.Min(d1, d2);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Subtraction(float d1, float d2) => MathHelper.Max(-d1, d2);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Intersection(float d1, float d2) => MathHelper.Max(d1, d2);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmoothUnion(float d1, float d2, float k)
        {
            float h = MathHelper.Clamp01(0.5f + 0.5f * (d2 - d1) / k);
            return MathHelper.Lerp(d2, d1, h) - k * h * (1f - h);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmoothSubtraction(float d1, float d2, float k)
        {
            float h = MathHelper.Clamp01(0.5f - 0.5f * (d2 + d1) / k);
            return MathHelper.Lerp(d2, -d1, h) + k * h * (1f - h);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SmoothIntersection(float d1, float d2, float k)
        {
            float h = MathHelper.Clamp01(0.5f - 0.5f * (d2 - d1) / k);
            return MathHelper.Lerp(d2, d1, h) + k * h * (1f - h);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Repeat(float p, float spacing) => MathF.Abs(p % spacing) - spacing * 0.5f;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 Repeat3D(Vector3 p, Vector3 s) =>
            new Vector3(MathF.Abs(p.X % s.X) - s.X * 0.5f, MathF.Abs(p.Y % s.Y) - s.Y * 0.5f, MathF.Abs(p.Z % s.Z) - s.Z * 0.5f);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 SymmetryX(Vector3 p) => new Vector3(MathF.Abs(p.X), p.Y, p.Z);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 SymmetryY(Vector3 p) => new Vector3(p.X, MathF.Abs(p.Y), p.Z);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 SymmetryZ(Vector3 p) => new Vector3(p.X, p.Y, MathF.Abs(p.Z));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float OpN(float d1, float d2, float k) => -SmoothUnion(-d1, -d2, k);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float OpS(float d1, float d2, float k) => SmoothSubtraction(d1, d2, k);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float OpI(float d1, float d2, float k) => SmoothIntersection(d1, d2, k);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 EstimateGradient(Vector3 p, SdfPrimitiveKernel.SdfShape shape, float r, Vector3 size, float param4, float param5, float scale, float offset, int shapeType)
        {
            float eps = 0.001f;
            float d1 = EvalShape(p + new Vector3(eps, 0, 0), shape, r, size, param4, param5);
            float d2 = EvalShape(p - new Vector3(eps, 0, 0), shape, r, size, param4, param5);
            float d3 = EvalShape(p + new Vector3(0, eps, 0), shape, r, size, param4, param5);
            float d4 = EvalShape(p - new Vector3(0, eps, 0), shape, r, size, param4, param5);
            float d5 = EvalShape(p + new Vector3(0, 0, eps), shape, r, size, param4, param5);
            float d6 = EvalShape(p - new Vector3(0, 0, eps), shape, r, size, param4, param5);
            return new Vector3((d1 - d2) / (2f * eps), (d3 - d4) / (2f * eps), (d5 - d6) / (2f * eps));
        }
        private static float EvalShape(Vector3 p, SdfPrimitiveKernel.SdfShape shape, float r, Vector3 size, float param4, float param5)
        {
            switch (shape)
            {
                case SdfPrimitiveKernel.SdfShape.Sphere:
                    return Sphere(p, r);
                case SdfPrimitiveKernel.SdfShape.Box:
                    return Box(p, size);
                case SdfPrimitiveKernel.SdfShape.Torus:
                    return Torus(p, r, param4);
                case SdfPrimitiveKernel.SdfShape.Cylinder:
                    return Cylinder(p, r, param4);
                case SdfPrimitiveKernel.SdfShape.Cone:
                    return Cone(p, r, param4);
                case SdfPrimitiveKernel.SdfShape.Capsule:
                    return Capsule(p, new Vector3(0, -param4, 0), new Vector3(0, param4, 0), r);
                case SdfPrimitiveKernel.SdfShape.Octahedron:
                    return Octahedron(p, r);
                case SdfPrimitiveKernel.SdfShape.Plane:
                    return Plane(p, Vector3.UnitY, param4);
                case SdfPrimitiveKernel.SdfShape.RoundedBox:
                    return RoundedBox(p, size, param4);
                case SdfPrimitiveKernel.SdfShape.RoundedCylinder:
                    return RoundedCylinder(p, r, param4, param5);
                case SdfPrimitiveKernel.SdfShape.RoundedCone:
                    return RoundedCone(p, r, param4, param5);
                case SdfPrimitiveKernel.SdfShape.Ellipsoid:
                    return Ellipsoid(p, size);
                default:
                    return Sphere(p, r);
            }
        }
    }

    #endregion

    #region NoiseGenerator

    public static class NoiseGenerator
    {
        private static readonly int[] Perm = new int[512];
        private static readonly int[] PermMod12 = new int[512];
        private static readonly Vector3[] Grad3 = new Vector3[]
        {
            new Vector3(1,1,0), new Vector3(-1,1,0), new Vector3(1,-1,0), new Vector3(-1,-1,0),
            new Vector3(1,0,1), new Vector3(-1,0,1), new Vector3(1,0,-1), new Vector3(-1,0,-1),
            new Vector3(0,1,1), new Vector3(0,-1,1), new Vector3(0,1,-1), new Vector3(0,-1,-1)
        };
        private static readonly int[] P = new int[]
        {
            151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
            8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
            35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
            134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
            55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
            18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
            250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
            189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
            172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
            228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
            107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
        };

        static NoiseGenerator()
        {
            for (int i = 0; i < 512; i++)
            {
                Perm[i] = P[i & 255];
                PermMod12[i] = Perm[i] % 12;
            }
            var rng = new Random(42);
            for (int i = 0; i < 256; i++)
            {
                CellHash[i] = (float)rng.NextDouble();
                CellPoints[i] = new Vector3((float)rng.NextDouble(), (float)rng.NextDouble(), (float)rng.NextDouble());
            }
        }

        private static readonly int[] Hash = new int[]
        {
            151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
            140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
            247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
            57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
            74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
            60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
            65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
            200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
            52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
            207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
            119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
            129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
            218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
            81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
            184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
            222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
        };

        private static float Fade(float t) => t * t * t * (t * (t * 6f - 15f) + 10f);
        private static float Lerp(float t, float a, float b) => a + t * (b - a);
        private static float Dot(float[] g, float x, float y) => g[0] * x + g[1] * y;
        private static float Dot(Vector3 g, float x, float y, float z) => g.X * x + g.Y * y + g.Z * z;

        public static float Perlin2D(float x, float y)
        {
            int xi = MathHelper.FloorToInt(x) & 255;
            int yi = MathHelper.FloorToInt(y) & 255;
            float xf = x - MathF.Floor(x);
            float yf = y - MathF.Floor(y);
            float u = Fade(xf);
            float v = Fade(yf);
            int aa = Perm[Perm[xi] + yi];
            int ab = Perm[Perm[xi] + yi + 1];
            int ba = Perm[Perm[xi + 1] + yi];
            int bb = Perm[Perm[xi + 1] + yi + 1];
            float x1 = Lerp(u, (aa >> 1) * 0.5f, (ba >> 1) * 0.5f);
            float x2 = Lerp(u, (ab >> 1) * 0.5f, (bb >> 1) * 0.5f);
            return Lerp(v, x1, x2);
        }

        public static float Perlin3D(float x, float y, float z)
        {
            int xi = MathHelper.FloorToInt(x) & 255;
            int yi = MathHelper.FloorToInt(y) & 255;
            int zi = MathHelper.FloorToInt(z) & 255;
            float xf = x - MathF.Floor(x);
            float yf = y - MathF.Floor(y);
            float zf = z - MathF.Floor(z);
            float u = Fade(xf);
            float v = Fade(yf);
            float w = Fade(zf);
            int a = Perm[xi] + yi;
            int aa = Perm[a] + zi;
            int ab = Perm[a + 1] + zi;
            int b = Perm[xi + 1] + yi;
            int ba = Perm[b] + zi;
            int bb = Perm[b + 1] + zi;
            float x1 = Lerp(u, Dot(Grad3[Perm[aa] % 12], xf, yf, zf), Dot(Grad3[Perm[ba] % 12], xf - 1, yf, zf));
            float x2 = Lerp(u, Dot(Grad3[Perm[ab] % 12], xf, yf - 1, zf), Dot(Grad3[Perm[bb] % 12], xf - 1, yf - 1, zf));
            float y1 = Lerp(v, x1, x2);
            x1 = Lerp(u, Dot(Grad3[Perm[aa + 1] % 12], xf, yf, zf - 1), Dot(Grad3[Perm[ba + 1] % 12], xf - 1, yf, zf - 1));
            x2 = Lerp(u, Dot(Grad3[Perm[ab + 1] % 12], xf, yf - 1, zf - 1), Dot(Grad3[Perm[bb + 1] % 12], xf - 1, yf - 1, zf - 1));
            float y2 = Lerp(v, x1, x2);
            return Lerp(w, y1, y2);
        }

        public static float Perlin4D(float x, float y, float z, float w)
        {
            float n0000 = Perlin3D(x, y, z) * (1f - w);
            float n0001 = Perlin3D(x + 31.416f, y + 47.853f, z + 12.793f) * w;
            return n0000 + n0001;
        }

        private const float F2 = 0.36602540378443864676f;
        private const float G2 = 0.21132486540518713730f;
        private const float F3 = 0.33333333333333333333f;
        private const float G3 = 0.16666666666666666667f;

        private static readonly int[][] SimplexGrad3 = new int[][]
        {
            new int[]{1,1,0}, new int[]{-1,1,0}, new int[]{1,-1,0}, new int[]{-1,-1,0},
            new int[]{1,0,1}, new int[]{-1,0,1}, new int[]{1,0,-1}, new int[]{-1,0,-1},
            new int[]{0,1,1}, new int[]{0,-1,1}, new int[]{0,1,-1}, new int[]{0,-1,-1}
        };

        private static int FastFloor(float x)
        {
            int xi = MathHelper.FloorToInt(x);
            return x < xi ? xi - 1 : xi;
        }

        private static float Dot(int[] g, float x, float y) => g[0] * x + g[1] * y;
        private static float Dot(int[] g, float x, float y, float z) => g[0] * x + g[1] * y + g[2] * z;

        public static float Simplex2D(float x, float y)
        {
            float s = (x + y) * F2;
            int i = FastFloor(x + s);
            int j = FastFloor(y + s);
            float t = (i + j) * G2;
            float X0 = i - t;
            float Y0 = j - t;
            float x0 = x - X0;
            float y0 = y - Y0;
            int i1, j1;
            if (x0 > y0)
            { i1 = 1; j1 = 0; }
            else
            { i1 = 0; j1 = 1; }
            float x1 = x0 - i1 + G2;
            float y1 = y0 - j1 + G2;
            float x2 = x0 - 1.0f + 2.0f * G2;
            float y2 = y0 - 1.0f + 2.0f * G2;
            int ii = i & 255;
            int jj = j & 255;
            int gi0 = Perm[ii + Perm[jj]] % 12;
            int gi1 = Perm[ii + i1 + Perm[jj + j1]] % 12;
            int gi2 = Perm[ii + 1 + Perm[jj + 1]] % 12;
            float n0 = 0f, n1 = 0f, n2 = 0f;
            float t0 = 0.5f - x0 * x0 - y0 * y0;
            if (t0 >= 0)
            { t0 *= t0; n0 = t0 * t0 * Dot(SimplexGrad3[gi0], x0, y0); }
            float t1 = 0.5f - x1 * x1 - y1 * y1;
            if (t1 >= 0)
            { t1 *= t1; n1 = t1 * t1 * Dot(SimplexGrad3[gi1], x1, y1); }
            float t2 = 0.5f - x2 * x2 - y2 * y2;
            if (t2 >= 0)
            { t2 *= t2; n2 = t2 * t2 * Dot(SimplexGrad3[gi2], x2, y2); }
            return 70.0f * (n0 + n1 + n2);
        }

        public static float Simplex3D(float x, float y, float z)
        {
            float s = (x + y + z) * F3;
            int i = FastFloor(x + s);
            int j = FastFloor(y + s);
            int k = FastFloor(z + s);
            float t = (i + j + k) * G3;
            float X0 = i - t;
            float Y0 = j - t;
            float Z0 = k - t;
            float x0 = x - X0;
            float y0 = y - Y0;
            float z0 = z - Z0;
            int i1, j1, k1, i2, j2, k2;
            if (x0 >= y0)
            {
                if (y0 >= z0)
                { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
                else if (x0 >= z0)
                { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
                else
                { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
            }
            else
            {
                if (y0 < z0)
                { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
                else if (x0 < z0)
                { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
                else
                { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
            }
            float x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
            float x2 = x0 - i2 + 2f * G3, y2 = y0 - j2 + 2f * G3, z2 = z0 - k2 + 2f * G3;
            float x3 = x0 - 1f + 3f * G3, y3 = y0 - 1f + 3f * G3, z3 = z0 - 1f + 3f * G3;
            int ii = i & 255, jj = j & 255, kk = k & 255;
            int gi0 = Perm[ii + Perm[jj + Perm[kk]]] % 12;
            int gi1 = Perm[ii + i1 + Perm[jj + j1 + Perm[kk + k1]]] % 12;
            int gi2 = Perm[ii + i2 + Perm[jj + j2 + Perm[kk + k2]]] % 12;
            int gi3 = Perm[ii + 1 + Perm[jj + 1 + Perm[kk + 1]]] % 12;
            float n0 = 0f, n1 = 0f, n2 = 0f, n3 = 0f;
            float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
            if (t0 > 0)
            { t0 *= t0; n0 = t0 * t0 * Dot(Grad3[gi0], x0, y0, z0); }
            float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
            if (t1 > 0)
            { t1 *= t1; n1 = t1 * t1 * Dot(Grad3[gi1], x1, y1, z1); }
            float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
            if (t2 > 0)
            { t2 *= t2; n2 = t2 * t2 * Dot(Grad3[gi2], x2, y2, z2); }
            float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
            if (t3 > 0)
            { t3 *= t3; n3 = t3 * t3 * Dot(Grad3[gi3], x3, y3, z3); }
            return 32f * (n0 + n1 + n2 + n3);
        }

        public static float Simplex4D(float x, float y, float z, float w)
        {
            float n0 = Perlin3D(x, y, z) * (0.5f + 0.5f * MathF.Sin(w));
            float n1 = Perlin3D(x + 17.3f, y + 41.9f, z + 7.3f) * (0.5f + 0.5f * MathF.Cos(w * 0.7f));
            return (n0 + n1) * 0.5f;
        }

        private static readonly float[] CellHash = new float[256];
        private static readonly Vector3[] CellPoints = new Vector3[256];

        private static int HashInt(int x) => Hash[x & 255];
        private static int HashInt2D(int x, int y) => Hash[(Hash[x & 255] + y) & 255];
        private static int HashInt3D(int x, int y, int z) => Hash[(Hash[(Hash[x & 255] + y) & 255] + z) & 255];

        public static float HashFloat3(float x, float y, float z)
        {
            int ix = FastFloor(x);
            int iy = FastFloor(y);
            int iz = FastFloor(z);
            return CellHash[HashInt3D(ix, iy, iz) & 255];
        }

        public static (float, Vector3) Worley3D(float x, float y, float z)
        {
            int ix = FastFloor(x);
            int iy = FastFloor(y);
            int iz = FastFloor(z);
            float fx = x - ix;
            float fy = y - iy;
            float fz = z - iz;
            float minDist = float.MaxValue;
            Vector3 closestPoint = Vector3.Zero;
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        int cx = ix + dx;
                        int cy = iy + dy;
                        int cz = iz + dz;
                        int h = HashInt3D(cx, cy, cz);
                        Vector3 cellPoint = new Vector3(
                            CellPoints[h & 255].X + cx,
                            CellPoints[(h >> 4) & 255].Y + cy,
                            CellPoints[(h >> 8) & 255].Z + cz);
                        Vector3 diff = cellPoint - new Vector3(x, y, z);
                        float dist = diff.LengthSquared();
                        if (dist < minDist)
                        {
                            minDist = dist;
                            closestPoint = diff;
                        }
                    }
                }
            }
            return (MathF.Sqrt(minDist), closestPoint);
        }

        public static (float, float, Vector3) Voronoi3D(float x, float y, float z)
        {
            int ix = FastFloor(x);
            int iy = FastFloor(y);
            int iz = FastFloor(z);
            float f1 = float.MaxValue, f2 = float.MaxValue;
            Vector3 p1 = Vector3.Zero;
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        int cx = ix + dx, cy = iy + dy, cz = iz + dz;
                        int h = HashInt3D(cx, cy, cz);
                        Vector3 cellPoint = new Vector3(
                            cx + CellPoints[h & 255].X,
                            cy + CellPoints[(h >> 4) & 255].Y,
                            cz + CellPoints[(h >> 8) & 255].Z);
                        float dist = (cellPoint - new Vector3(x, y, z)).LengthSquared();
                        if (dist < f1)
                        { f2 = f1; f1 = dist; p1 = cellPoint - new Vector3(x, y, z); }
                        else if (dist < f2)
                        { f2 = dist; }
                    }
                }
            }
            return (MathF.Sqrt(f1), MathF.Sqrt(f2), p1);
        }

        public static Vector3 CurlNoise3D(float x, float y, float z)
        {
            float eps = 0.01f;
            float dx = (Perlin3D(x, y + eps, z) - Perlin3D(x, y - eps, z)) - (Perlin3D(x, y, z + eps) - Perlin3D(x, y, z - eps));
            float dy = (Perlin3D(x, y, z + eps) - Perlin3D(x, y, z - eps)) - (Perlin3D(x + eps, y, z) - Perlin3D(x - eps, y, z));
            float dz = (Perlin3D(x + eps, y, z) - Perlin3D(x - eps, y, z)) - (Perlin3D(x, y + eps, z) - Perlin3D(x, y - eps, z));
            return new Vector3(dx, dy, dz) / (2f * eps);
        }

        public static float FBM2D(float x, float y, int octaves, float lacunarity, float gain)
        {
            float sum = 0f;
            float amp = 1f;
            float freq = 1f;
            float maxAmp = 0f;
            for (int i = 0; i < octaves; i++)
            {
                sum += Perlin2D(x * freq, y * freq) * amp;
                maxAmp += amp;
                amp *= gain;
                freq *= lacunarity;
            }
            return sum / maxAmp;
        }

        public static float FBM3D(float x, float y, float z, int octaves, float lacunarity, float gain)
        {
            float sum = 0f;
            float amp = 1f;
            float freq = 1f;
            float maxAmp = 0f;
            for (int i = 0; i < octaves; i++)
            {
                sum += Perlin3D(x * freq, y * freq, z * freq) * amp;
                maxAmp += amp;
                amp *= gain;
                freq *= lacunarity;
            }
            return sum / maxAmp;
        }

        public static float RidgedMultifractal2D(float x, float y, int octaves, float lacunarity, float gain, float offset)
        {
            float sum = 0f;
            float amp = 1f;
            float freq = 1f;
            float prev = 1f;
            for (int i = 0; i < octaves; i++)
            {
                float n = 1f - MathF.Abs(Perlin2D(x * freq, y * freq));
                n *= n;
                sum += n * amp * prev;
                prev = n;
                amp *= gain;
                freq *= lacunarity;
            }
            return sum;
        }

        public static float RidgedMultifractal3D(float x, float y, float z, int octaves, float lacunarity, float gain, float offset)
        {
            float sum = 0f;
            float amp = 1f;
            float freq = 1f;
            float prev = 1f;
            for (int i = 0; i < octaves; i++)
            {
                float n = offset - MathF.Abs(Perlin3D(x * freq, y * freq, z * freq));
                n = MathF.Max(0f, n);
                n *= n;
                sum += n * amp * prev;
                prev = n;
                amp *= gain;
                freq *= lacunarity;
            }
            return sum;
        }

        public static Vector3 DomainWarp3D(Vector3 p, float strength, float time)
        {
            Vector3 offset = new Vector3(
                FBM3D(p.X + 0.0f, p.Y + 0.0f, p.Z + 0.0f, 4, 2f, 0.5f),
                FBM3D(p.X + 5.2f, p.Y + 1.3f, p.Z + 2.8f, 4, 2f, 0.5f),
                FBM3D(p.X + 9.7f, p.Y + 6.4f, p.Z + 3.1f, 4, 2f, 0.5f)) * strength;
            return p + offset;
        }

        public static float Turbulence2D(float x, float y, int octaves, float lacunarity, float gain)
        {
            float sum = 0f;
            float amp = 1f;
            float freq = 1f;
            float maxAmp = 0f;
            for (int i = 0; i < octaves; i++)
            {
                sum += MathF.Abs(Perlin2D(x * freq, y * freq)) * amp;
                maxAmp += amp;
                amp *= gain;
                freq *= lacunarity;
            }
            return sum / maxAmp;
        }

        public static float Turbulence3D(float x, float y, float z, int octaves, float lacunarity, float gain)
        {
            float sum = 0f;
            float amp = 1f;
            float freq = 1f;
            float maxAmp = 0f;
            for (int i = 0; i < octaves; i++)
            {
                sum += MathF.Abs(Perlin3D(x * freq, y * freq, z * freq)) * amp;
                maxAmp += amp;
                amp *= gain;
                freq *= lacunarity;
            }
            return sum / maxAmp;
        }

        public static float Billow2D(float x, float y, int octaves, float lacunarity, float gain)
        {
            float sum = 0f;
            float amp = 1f;
            float freq = 1f;
            float maxAmp = 0f;
            for (int i = 0; i < octaves; i++)
            {
                float n = 2f * MathF.Abs(Perlin2D(x * freq, y * freq)) - 1f;
                sum += n * amp;
                maxAmp += amp;
                amp *= gain;
                freq *= lacunarity;
            }
            return sum / maxAmp;
        }

        public static float Billow3D(float x, float y, float z, int octaves, float lacunarity, float gain)
        {
            float sum = 0f;
            float amp = 1f;
            float freq = 1f;
            float maxAmp = 0f;
            for (int i = 0; i < octaves; i++)
            {
                float n = 2f * MathF.Abs(Perlin3D(x * freq, y * freq, z * freq)) - 1f;
                sum += n * amp;
                maxAmp += amp;
                amp *= gain;
                freq *= lacunarity;
            }
            return sum / maxAmp;
        }

        public static float HeterogeneousTerrain3D(float x, float y, float z, int octaves, float lacunarity, float gain, float persistence)
        {
            float sum = 0f;
            float amp = 1f;
            float freq = 1f;
            float weight = 1f;
            for (int i = 0; i < octaves; i++)
            {
                float signal = MathF.Abs(Perlin3D(x * freq, y * freq, z * freq));
                signal *= signal * weight;
                weight = MathHelper.Clamp01(signal * persistence);
                sum += signal * amp;
                amp *= gain;
                freq *= lacunarity;
            }
            return sum;
        }

        public static float WarpedFBM3D(float x, float y, float z, int octaves, float lacunarity, float gain, float warpStrength)
        {
            Vector3 wp = DomainWarp3D(new Vector3(x, y, z), warpStrength, 0f);
            return FBM3D(wp.X, wp.Y, wp.Z, octaves, lacunarity, gain);
        }

        public static float HybridFBM3D(float x, float y, float z, int octaves, float lacunarity, float gain, float warpStrength)
        {
            float sum = 0f;
            float amp = 1f;
            float freq = 1f;
            float weight = 1f;
            for (int i = 0; i < octaves; i++)
            {
                Vector3 wp = DomainWarp3D(new Vector3(x * freq, y * freq, z * freq), warpStrength * weight, i);
                float signal = Perlin3D(wp.X, wp.Y, wp.Z);
                signal *= signal * weight;
                weight = MathHelper.Clamp01(signal * gain);
                sum += signal * amp;
                amp *= gain;
                freq *= lacunarity;
            }
            return sum;
        }

        public static float Weathering3D(float x, float y, float z, float weather, int octaves, float lacunarity, float gain)
        {
            float baseTerrain = FBM3D(x, y, z, octaves, lacunarity, gain);
            float detail = Turbulence3D(x * 4f, y * 4f, z * 4f, 3, 2f, 0.5f);
            float erosion = MathHelper.Lerp(baseTerrain, baseTerrain * (1f - detail * 0.3f), weather);
            return erosion;
        }

        public static float VoronoiEdges3D(float x, float y, float z)
        {
            var (f1, f2, _) = Voronoi3D(x, y, z);
            return f2 - f1;
        }

        public static float VoronoiF1F23D(float x, float y, float z, float exponent)
        {
            var (f1, f2, _) = Voronoi3D(x, y, z);
            return MathF.Pow(f1, exponent) + MathF.Pow(f2, exponent);
        }

        public static float GaborNoise2D(float x, float y, float frequency, float orientation, float bandwidth)
        {
            float sum = 0f;
            int kernelRadius = MathHelper.CeilingToInt(2f / (MathF.PI * bandwidth));
            for (int ky = -kernelRadius; ky <= kernelRadius; ky++)
            {
                for (int kx = -kernelRadius; kx <= kernelRadius; kx++)
                {
                    int ix = FastFloor(x) + kx;
                    int iy = FastFloor(y) + ky;
                    int h = HashInt2D(ix, iy);
                    float fx = (h & 0xFF) / 255f;
                    float fy = ((h >> 8) & 0xFF) / 255f;
                    float fo = ((h >> 16) & 0xFF) / 255f * MathF.PI;
                    float px = ix + fx - x;
                    float py = iy + fy - y;
                    float gaussian = MathF.Exp(-MathF.PI * (px * px + py * py) * bandwidth * bandwidth);
                    float cosine = MathF.Cos(2f * MathF.PI * frequency * (px * MathF.Cos(fo) + py * MathF.Sin(fo)));
                    float hashPhase = ((h >> 24) & 0xFF) / 255f * 2f * MathF.PI;
                    sum += gaussian * cosine + hashPhase;
                }
            }
            return sum / (kernelRadius * 2 + 1);
        }

        public static float ValueNoise2D(float x, float y)
        {
            int ix = FastFloor(x);
            int iy = FastFloor(y);
            float fx = x - ix;
            float fy = y - iy;
            float sx = fx * fx * (3f - 2f * fx);
            float sy = fy * fy * (3f - 2f * fy);
            float v00 = (HashInt2D(ix, iy) & 0xFF) / 255f;
            float v10 = (HashInt2D(ix + 1, iy) & 0xFF) / 255f;
            float v01 = (HashInt2D(ix, iy + 1) & 0xFF) / 255f;
            float v11 = (HashInt2D(ix + 1, iy + 1) & 0xFF) / 255f;
            float i1 = MathHelper.Lerp(v00, v10, sx);
            float i2 = MathHelper.Lerp(v01, v11, sx);
            return MathHelper.Lerp(i1, i2, sy);
        }

        public static float ValueNoise3D(float x, float y, float z)
        {
            int ix = FastFloor(x);
            int iy = FastFloor(y);
            int iz = FastFloor(z);
            float fx = x - ix;
            float fy = y - iy;
            float fz = z - iz;
            float sx = fx * fx * (3f - 2f * fx);
            float sy = fy * fy * (3f - 2f * fy);
            float sz = fz * fz * (3f - 2f * fz);
            float v000 = (HashInt3D(ix, iy, iz) & 0xFF) / 255f;
            float v100 = (HashInt3D(ix + 1, iy, iz) & 0xFF) / 255f;
            float v010 = (HashInt3D(ix, iy + 1, iz) & 0xFF) / 255f;
            float v110 = (HashInt3D(ix + 1, iy + 1, iz) & 0xFF) / 255f;
            float v001 = (HashInt3D(ix, iy, iz + 1) & 0xFF) / 255f;
            float v101 = (HashInt3D(ix + 1, iy, iz + 1) & 0xFF) / 255f;
            float v011 = (HashInt3D(ix, iy + 1, iz + 1) & 0xFF) / 255f;
            float v111 = (HashInt3D(ix + 1, iy + 1, iz + 1) & 0xFF) / 255f;
            return MathHelper.TrilinearInterpolate(v000, v100, v010, v110, v001, v101, v011, v111, sx, sy, sz);
        }
    }

    #endregion

    #region SDF Primitive Kernel

    public sealed class SdfPrimitiveKernel : INeuronKernel
    {
        private const int ParamCount = 12;
        private readonly float[] _weights = new float[ParamCount];
        private Vector3 _lastPosition;
        private NeuronOutput _lastOutput;

        public enum SdfShape
        {
            Sphere, Box, Torus, Cylinder, Cone, Capsule, Octahedron, Plane,
            RoundedBox, RoundedCylinder, RoundedCone, Ellipsoid
        }

        public string Name => "SdfPrimitive";
        public NeuronKind Kind => NeuronKind.SdfPrimitive;
        public int Version => 1;
        public SdfShape Shape { get; set; } = SdfShape.Sphere;

        public SdfPrimitiveKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float r = _weights[0];
            Vector3 size = new(_weights[1], _weights[2], _weights[3]);
            float param4 = _weights[4];
            float param5 = _weights[5];
            float displacement = _weights[6];
            float frequency = _weights[7];
            float amplitude = _weights[8];
            float blend = _weights[9];
            float scale = _weights[10];
            float offset = _weights[11];

            Vector3 p = input.Position * scale + new Vector3(offset);
            float d = ComputeShapeDistance(p, r, size, param4, param5);

            float noiseDisp = NoiseGenerator.Perlin3D(p.X * frequency, p.Y * frequency, p.Z * frequency) * amplitude;
            d += noiseDisp * displacement;

            float smoothD = MathHelper.SmoothMin(d, d + blend, MathHelper.Max(blend, 0.001f));
            float height = MathHelper.Remap(smoothD, -1f, 1f, 0f, 1f);

            Vector3 grad = EstimateSdfGradient(p, r, size, param4, param5);
            Vector3 normal = grad.LengthSquared() > 0.0001f ? Vector3.Normalize(grad) : Vector3.UnitY;

            Vector3 disp = normal * (noiseDisp * displacement);

            float curvature = ComputeLocalCurvature(p, r, size, frequency, param4, param5);

            _lastPosition = input.Position;
            _lastOutput = new NeuronOutput
            {
                Value = smoothD,
                Displacement = disp,
                Color = new Vector3(
                    MathHelper.Remap(normal.X, -1, 1, 0, 1),
                    MathHelper.Remap(normal.Y, -1, 1, 0, 1),
                    MathHelper.Remap(normal.Z, -1, 1, 0, 1)),
                Roughness = MathHelper.Clamp01(MathHelper.Remap(MathF.Abs(curvature), 0, 2, 0.8f, 0.1f)),
                Metallic = 0f,
                Opacity = MathHelper.Step(0f, -smoothD),
                Emission = 0f,
                OutputNormal = normal,
                Height = height,
                AmbientOcclusion = ComputeAO(p, normal, r, size, param4, param5),
                Thickness = MathHelper.Clamp01(MathF.Abs(d) * 2f),
                IOR = 1.5f,
                Specular = 0.5f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("sdf_distance", smoothD)
                    .Add("noise_displacement", noiseDisp)
                    .Add("curvature", curvature),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("estimated_normal", normal)
                    .Add("displacement_vector", disp)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            Vector3 p = _lastPosition;
            float r = _weights[0];
            Vector3 size = new(_weights[1], _weights[2], _weights[3]);
            float param4 = _weights[4];
            float param5 = _weights[5];
            float scl = _weights[10];

            float eps = 0.001f;
            Vector3 sp = p * scl + new Vector3(_weights[11]);

            var pg = ImmutableDictionary<string, float>.Empty;
            float[] wg = new float[ParamCount];

            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float dPlus = ComputeShapeDistance(sp, _weights[0], new Vector3(_weights[1], _weights[2], _weights[3]), _weights[4], _weights[5]);
                _weights[i] = oldW - eps;
                float dMinus = ComputeShapeDistance(sp, _weights[0], new Vector3(_weights[1], _weights[2], _weights[3]), _weights[4], _weights[5]);
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (dPlus - dMinus) / (2f * eps) + gradient.HeightGradient * (dPlus - dMinus) / (4f * eps);
            }

            Vector3 posGrad = EstimateSdfGradient(sp, r, size, param4, param5);

            return new NeuronGradient
            {
                ValueGradient = gradient.ValueGradient,
                PositionGradient = gradient.PositionGradient + gradient.ValueGradient * posGrad * scl,
                HeightGradient = gradient.HeightGradient,
                ParameterGradients = pg,
                WeightGradients = wg
            };
        }

        private float ComputeShapeDistance(Vector3 p, float r, Vector3 size, float p4, float p5)
        {
            switch (Shape)
            {
                case SdfShape.Sphere:
                    return SdfOperations.Sphere(p, r);
                case SdfShape.Box:
                    return SdfOperations.Box(p, size);
                case SdfShape.Torus:
                    return SdfOperations.Torus(p, r, p4);
                case SdfShape.Cylinder:
                    return SdfOperations.Cylinder(p, r, p4);
                case SdfShape.Cone:
                    return SdfOperations.Cone(p, r, p4);
                case SdfShape.Capsule:
                    return SdfOperations.Capsule(p, new Vector3(0, -p4, 0), new Vector3(0, p4, 0), r);
                case SdfShape.Octahedron:
                    return SdfOperations.Octahedron(p, r);
                case SdfShape.Plane:
                    return SdfOperations.Plane(p, Vector3.UnitY, p4);
                case SdfShape.RoundedBox:
                    return SdfOperations.RoundedBox(p, size, p4);
                case SdfShape.RoundedCylinder:
                    return SdfOperations.RoundedCylinder(p, r, p4, p5);
                case SdfShape.RoundedCone:
                    return SdfOperations.RoundedCone(p, r, p4, p5);
                case SdfShape.Ellipsoid:
                    return SdfOperations.Ellipsoid(p, size);
                default:
                    return SdfOperations.Sphere(p, r);
            }
        }

        private Vector3 EstimateSdfGradient(Vector3 p, float r, Vector3 size, float p4, float p5)
        {
            float eps = 0.001f;
            float d1 = ComputeShapeDistance(p + new Vector3(eps, 0, 0), r, size, p4, p5);
            float d2 = ComputeShapeDistance(p - new Vector3(eps, 0, 0), r, size, p4, p5);
            float d3 = ComputeShapeDistance(p + new Vector3(0, eps, 0), r, size, p4, p5);
            float d4 = ComputeShapeDistance(p - new Vector3(0, eps, 0), r, size, p4, p5);
            float d5 = ComputeShapeDistance(p + new Vector3(0, 0, eps), r, size, p4, p5);
            float d6 = ComputeShapeDistance(p - new Vector3(0, 0, eps), r, size, p4, p5);
            return new Vector3((d1 - d2) / (2f * eps), (d3 - d4) / (2f * eps), (d5 - d6) / (2f * eps));
        }

        private float ComputeLocalCurvature(Vector3 p, float r, Vector3 size, float freq, float p4, float p5)
        {
            float eps = 0.05f;
            float d = ComputeShapeDistance(p, r, size, p4, p5);
            float dx1 = ComputeShapeDistance(p + new Vector3(eps, 0, 0), r, size, p4, p5);
            float dx2 = ComputeShapeDistance(p - new Vector3(eps, 0, 0), r, size, p4, p5);
            float dy1 = ComputeShapeDistance(p + new Vector3(0, eps, 0), r, size, p4, p5);
            float dy2 = ComputeShapeDistance(p - new Vector3(0, eps, 0), r, size, p4, p5);
            float dz1 = ComputeShapeDistance(p + new Vector3(0, 0, eps), r, size, p4, p5);
            float dz2 = ComputeShapeDistance(p - new Vector3(0, 0, eps), r, size, p4, p5);
            float h2 = eps * eps;
            float dxx = (dx1 - 2f * d + dx2) / h2;
            float dyy = (dy1 - 2f * d + dy2) / h2;
            float dzz = (dz1 - 2f * d + dz2) / h2;
            float dx = (dx1 - dx2) / (2f * eps);
            float dy = (dy1 - dy2) / (2f * eps);
            float dz = (dz1 - dz2) / (2f * eps);
            float gradLen2 = dx * dx + dy * dy + dz * dz;
            float gradLen = MathF.Sqrt(MathHelper.Max(gradLen2, 1e-10f));
            float meanCurv = (dxx * (dy * dy + dz * dz) + dyy * (dx * dx + dz * dz) + dzz * (dx * dx + dy * dy)
                - 2f * dx * dy * dx * dy - 2f * dx * dz * dx * dz - 2f * dy * dz * dy * dz)
                / (2f * gradLen * gradLen * gradLen + 1e-10f);
            return meanCurv;
        }

        private float ComputeAO(Vector3 p, Vector3 n, float r, Vector3 size, float p4, float p5)
        {
            float ao = 0f;
            float stepSize = 0.1f;
            float weight = 1f;
            for (int i = 1; i <= 5; i++)
            {
                float dist = ComputeShapeDistance(p + n * stepSize * i, r, size, p4, p5);
                ao += MathHelper.Max(0f, stepSize - dist) * weight;
                weight *= 0.75f;
            }
            return MathHelper.Clamp01(1f - ao * 2f);
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 128;
        public INeuronKernel Clone() { var c = new SdfPrimitiveKernel { Shape = this.Shape }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() { if (_weights[0] <= 0) return "Radius must be positive"; return null; }
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters()
        {
            _weights[0] = 1f;
            _weights[1] = 1f;
            _weights[2] = 1f;
            _weights[3] = 1f;
            _weights[4] = 0.5f;
            _weights[5] = 0.1f;
            _weights[6] = 0f;
            _weights[7] = 1f;
            _weights[8] = 0.1f;
            _weights[9] = 0f;
            _weights[10] = 1f;
            _weights[11] = 0f;
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr)
        {
            if (gradient.WeightGradients != null && gradient.WeightGradients.Length >= ParamCount)
                for (int i = 0; i < ParamCount; i++)
                    _weights[i] -= lr * gradient.WeightGradients[i];
        }
    }

    #endregion


    #region Displacement Field Kernel

    public sealed class DisplacementFieldKernel : INeuronKernel
    {
        private const int ParamCount = 16;
        private readonly float[] _weights = new float[ParamCount];
        private Vector3 _lastPosition;
        private NeuronOutput _lastOutput;

        public enum DisplacementMode
        {
            Perlin, Simplex, Worley, Voronoi, Curl, FBM, RidgedMultifractal,
            DomainWarped, SplatMapped, Turbulence, Billow, Heterogeneous
        }

        public string Name => "DisplacementField";
        public NeuronKind Kind => NeuronKind.DisplacementField;
        public int Version => 1;
        public DisplacementMode Mode { get; set; } = DisplacementMode.FBM;

        public DisplacementFieldKernel() => ResetParameters();

        private float ComputeCurrentModeValue(Vector3 p)
        {
            float amp = _weights[0], freq = _weights[1], lac = _weights[2], gain = _weights[3];
            int oct = MathHelper.Clamp((int)_weights[4], 1, 16);
            float t = 0f;
            switch (Mode)
            {
                case DisplacementMode.Perlin:
                    return NoiseGenerator.Perlin3D(p.X * freq, p.Y * freq, p.Z * freq) * amp;
                case DisplacementMode.Simplex:
                    return NoiseGenerator.Simplex3D(p.X * freq, p.Y * freq, p.Z * freq) * amp;
                case DisplacementMode.Worley:
                    return NoiseGenerator.Worley3D(p.X * freq, p.Y * freq, p.Z * freq).Item1 * amp;
                case DisplacementMode.Voronoi:
                    var v = NoiseGenerator.Voronoi3D(p.X * freq, p.Y * freq, p.Z * freq);
                    return MathHelper.Lerp(v.Item1, v.Item2, 0.5f) * amp;
                case DisplacementMode.Curl:
                    return NoiseGenerator.CurlNoise3D(p.X * freq, p.Y * freq, p.Z * freq).Length() * amp;
                case DisplacementMode.FBM:
                    return NoiseGenerator.FBM3D(p.X * freq, p.Y * freq, p.Z * freq, oct, lac, gain) * amp;
                case DisplacementMode.RidgedMultifractal:
                    return NoiseGenerator.RidgedMultifractal3D(p.X * freq, p.Y * freq, p.Z * freq, oct, lac, gain, _weights[7]) * amp;
                case DisplacementMode.DomainWarped:
                    var wp = NoiseGenerator.DomainWarp3D(p * freq, _weights[6], t);
                    return NoiseGenerator.FBM3D(wp.X, wp.Y, wp.Z, oct, lac, gain) * amp;
                case DisplacementMode.Turbulence:
                    return NoiseGenerator.Turbulence3D(p.X * freq, p.Y * freq, p.Z * freq, oct, lac, gain) * amp;
                case DisplacementMode.Billow:
                    return NoiseGenerator.Billow3D(p.X * freq, p.Y * freq, p.Z * freq, oct, lac, gain) * amp;
                case DisplacementMode.Heterogeneous:
                    return NoiseGenerator.HeterogeneousTerrain3D(p.X * freq, p.Y * freq, p.Z * freq, oct, lac, gain, _weights[5]) * amp;
                default:
                    return 0f;
            }
        }

        public NeuronOutput Compute(in NeuronInput input)
        {
            float amp = _weights[0], freq = _weights[1], lac = _weights[2], gain = _weights[3];
            int oct = MathHelper.Clamp((int)_weights[4], 1, 16);
            float warp = _weights[6], ridgedOff = _weights[7], power = _weights[8];
            float clampMin = _weights[9], clampMax = _weights[10], normalStr = _weights[11];
            float timeScale = _weights[12], splatRad = _weights[13], threshold = _weights[14], smoothK = _weights[15];
            Vector3 p = input.Position;
            float t = input.Time * timeScale;
            float displacement = 0f;
            Vector3 displacementVec = Vector3.Zero;

            switch (Mode)
            {
                case DisplacementMode.Perlin:
                    displacement = NoiseGenerator.Perlin3D(p.X * freq, p.Y * freq + t, p.Z * freq) * amp;
                    displacementVec = ComputeNoiseGradient(p, freq, t, amp, NoiseGenerator.Perlin3D);
                    break;
                case DisplacementMode.Simplex:
                    displacement = NoiseGenerator.Simplex3D(p.X * freq, p.Y * freq + t, p.Z * freq) * amp;
                    displacementVec = ComputeNoiseGradient(p, freq, t, amp, NoiseGenerator.Simplex3D);
                    break;
                case DisplacementMode.Worley:
                    var (wDist, wCell) = NoiseGenerator.Worley3D(p.X * freq, p.Y * freq + t, p.Z * freq);
                    displacement = wDist * amp;
                    displacementVec = wCell * amp;
                    break;
                case DisplacementMode.Voronoi:
                    var (v1, v2, vPt) = NoiseGenerator.Voronoi3D(p.X * freq, p.Y * freq + t, p.Z * freq);
                    displacement = MathHelper.Lerp(v1, v2, 0.5f) * amp;
                    displacementVec = vPt * amp;
                    break;
                case DisplacementMode.Curl:
                    Vector3 curl = NoiseGenerator.CurlNoise3D(p.X * freq, p.Y * freq + t, p.Z * freq);
                    displacementVec = curl * amp;
                    displacement = curl.Length() * amp;
                    break;
                case DisplacementMode.FBM:
                    displacement = NoiseGenerator.FBM3D(p.X * freq, p.Y * freq + t, p.Z * freq, oct, lac, gain) * amp;
                    displacementVec = ComputeFBMGradient(p, freq, t, amp, oct, lac, gain);
                    break;
                case DisplacementMode.RidgedMultifractal:
                    displacement = NoiseGenerator.RidgedMultifractal3D(p.X * freq, p.Y * freq + t, p.Z * freq, oct, lac, gain, ridgedOff) * amp;
                    displacementVec = ComputeRidgedGradient(p, freq, t, amp, oct, lac, gain, ridgedOff);
                    break;
                case DisplacementMode.DomainWarped:
                    Vector3 wp = NoiseGenerator.DomainWarp3D(p * freq, warp, t);
                    displacement = NoiseGenerator.FBM3D(wp.X, wp.Y, wp.Z, oct, lac, gain) * amp;
                    displacementVec = ComputeDomainWarpGradient(p, freq, t, amp, oct, lac, gain, warp);
                    break;
                case DisplacementMode.Turbulence:
                    displacement = NoiseGenerator.Turbulence3D(p.X * freq, p.Y * freq + t, p.Z * freq, oct, lac, gain) * amp;
                    displacementVec = ComputeNoiseGradient(p, freq, t, amp, (x, y, z) => NoiseGenerator.Turbulence3D(x, y, z, oct, lac, gain));
                    break;
                case DisplacementMode.Billow:
                    displacement = NoiseGenerator.Billow3D(p.X * freq, p.Y * freq + t, p.Z * freq, oct, lac, gain) * amp;
                    displacementVec = ComputeNoiseGradient(p, freq, t, amp, (x, y, z) => NoiseGenerator.Billow3D(x, y, z, oct, lac, gain));
                    break;
                case DisplacementMode.Heterogeneous:
                    displacement = NoiseGenerator.HeterogeneousTerrain3D(p.X * freq, p.Y * freq + t, p.Z * freq, oct, lac, gain, _weights[5]) * amp;
                    displacementVec = ComputeNoiseGradient(p, freq, t, amp, (x, y, z) => NoiseGenerator.HeterogeneousTerrain3D(x, y, z, oct, lac, gain, _weights[5]));
                    break;
                case DisplacementMode.SplatMapped:
                    displacement = ComputeSplatDisplacement(p, freq, splatRad, threshold, smoothK) * amp;
                    displacementVec = ComputeSplatVector(p, freq, splatRad, threshold, smoothK) * amp;
                    break;
            }

            if (power != 1f && power != 0f)
            {
                float sign = MathF.Sign(displacement);
                displacement = sign * MathF.Pow(MathF.Abs(displacement) + 1e-7f, power);
            }
            displacement = MathHelper.Clamp(displacement, clampMin, clampMax);

            Vector3 normal = input.Normal.LengthSquared() > 0.001f ? Vector3.Normalize(input.Normal) : Vector3.UnitY;
            Vector3 finalDisp = normal * displacement + displacementVec * normalStr;
            Vector3 estimatedNormal = EstimateNormal(p, freq, amp, t);

            _lastPosition = input.Position;
            _lastOutput = new NeuronOutput
            {
                Value = displacement,
                Displacement = finalDisp,
                Color = new Vector3(MathHelper.Remap(displacement, clampMin, clampMax, 0, 1), MathHelper.Remap(displacement, -2, 2, 0, 1), MathHelper.Remap(input.Time % 10f, 0, 10, 0, 1)),
                Roughness = MathHelper.Clamp01(0.5f + displacement * 0.25f),
                Metallic = 0f,
                Opacity = 1f,
                Emission = MathHelper.Max(0, displacement - threshold) * 2f,
                EmissionColor = new Vector3(1f, 0.8f, 0.4f),
                OutputNormal = Vector3.Normalize(MathHelper.Lerp(normal, estimatedNormal, normalStr)),
                Height = MathHelper.Remap(displacement, clampMin, clampMax, 0, 1),
                AmbientOcclusion = MathHelper.Clamp01(1f - MathF.Abs(displacement) * 0.5f),
                Thickness = MathHelper.Clamp01(MathF.Abs(displacement) * 0.5f),
                IOR = 1.5f,
                Specular = 0.5f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("displacement_magnitude", displacement)
                    .Add("displacement_length", finalDisp.Length()),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("displacement_vector", finalDisp)
                    .Add("estimated_normal", estimatedNormal)
            };
            return _lastOutput;
        }

        private Vector3 ComputeNoiseGradient(Vector3 p, float freq, float t, float amp, Func<float, float, float, float> noiseFunc)
        {
            float eps = 0.001f;
            float nxp = noiseFunc((p.X + eps) * freq, p.Y * freq + t, p.Z * freq);
            float nxm = noiseFunc((p.X - eps) * freq, p.Y * freq + t, p.Z * freq);
            float nyp = noiseFunc(p.X * freq, (p.Y + eps) * freq + t, p.Z * freq);
            float nym = noiseFunc(p.X * freq, (p.Y - eps) * freq + t, p.Z * freq);
            float nzp = noiseFunc(p.X * freq, p.Y * freq + t, (p.Z + eps) * freq);
            float nzm = noiseFunc(p.X * freq, p.Y * freq + t, (p.Z - eps) * freq);
            return new Vector3((nxp - nxm) / (2f * eps), (nyp - nym) / (2f * eps), (nzp - nzm) / (2f * eps)) * amp;
        }

        private Vector3 ComputeFBMGradient(Vector3 p, float freq, float t, float amp, int oct, float lac, float gain)
        {
            float eps = 0.001f;
            return new Vector3(
                NoiseGenerator.FBM3D((p.X + eps) * freq, p.Y * freq + t, p.Z * freq, oct, lac, gain) - NoiseGenerator.FBM3D((p.X - eps) * freq, p.Y * freq + t, p.Z * freq, oct, lac, gain),
                NoiseGenerator.FBM3D(p.X * freq, (p.Y + eps) * freq + t, p.Z * freq, oct, lac, gain) - NoiseGenerator.FBM3D(p.X * freq, (p.Y - eps) * freq + t, p.Z * freq, oct, lac, gain),
                NoiseGenerator.FBM3D(p.X * freq, p.Y * freq + t, (p.Z + eps) * freq, oct, lac, gain) - NoiseGenerator.FBM3D(p.X * freq, p.Y * freq + t, (p.Z - eps) * freq, oct, lac, gain)) * amp / (2f * eps);
        }

        private Vector3 ComputeRidgedGradient(Vector3 p, float freq, float t, float amp, int oct, float lac, float gain, float ridgedOff)
        {
            float eps = 0.001f;
            return new Vector3(
                NoiseGenerator.RidgedMultifractal3D((p.X + eps) * freq, p.Y * freq + t, p.Z * freq, oct, lac, gain, ridgedOff) - NoiseGenerator.RidgedMultifractal3D((p.X - eps) * freq, p.Y * freq + t, p.Z * freq, oct, lac, gain, ridgedOff),
                NoiseGenerator.RidgedMultifractal3D(p.X * freq, (p.Y + eps) * freq + t, p.Z * freq, oct, lac, gain, ridgedOff) - NoiseGenerator.RidgedMultifractal3D(p.X * freq, (p.Y - eps) * freq + t, p.Z * freq, oct, lac, gain, ridgedOff),
                NoiseGenerator.RidgedMultifractal3D(p.X * freq, p.Y * freq + t, (p.Z + eps) * freq, oct, lac, gain, ridgedOff) - NoiseGenerator.RidgedMultifractal3D(p.X * freq, p.Y * freq + t, (p.Z - eps) * freq, oct, lac, gain, ridgedOff)) * amp / (2f * eps);
        }

        private Vector3 ComputeDomainWarpGradient(Vector3 p, float freq, float t, float amp, int oct, float lac, float gain, float warpStr)
        {
            float eps = 0.001f;
            Vector3 wp1 = NoiseGenerator.DomainWarp3D(new Vector3(p.X + eps, p.Y, p.Z) * freq, warpStr, t);
            Vector3 wp2 = NoiseGenerator.DomainWarp3D(new Vector3(p.X - eps, p.Y, p.Z) * freq, warpStr, t);
            Vector3 wp3 = NoiseGenerator.DomainWarp3D(new Vector3(p.X, p.Y + eps, p.Z) * freq, warpStr, t);
            Vector3 wp4 = NoiseGenerator.DomainWarp3D(new Vector3(p.X, p.Y - eps, p.Z) * freq, warpStr, t);
            Vector3 wp5 = NoiseGenerator.DomainWarp3D(new Vector3(p.X, p.Y, p.Z + eps) * freq, warpStr, t);
            Vector3 wp6 = NoiseGenerator.DomainWarp3D(new Vector3(p.X, p.Y, p.Z - eps) * freq, warpStr, t);
            return new Vector3(
                NoiseGenerator.FBM3D(wp1.X, wp1.Y, wp1.Z, oct, lac, gain) - NoiseGenerator.FBM3D(wp2.X, wp2.Y, wp2.Z, oct, lac, gain),
                NoiseGenerator.FBM3D(wp3.X, wp3.Y, wp3.Z, oct, lac, gain) - NoiseGenerator.FBM3D(wp4.X, wp4.Y, wp4.Z, oct, lac, gain),
                NoiseGenerator.FBM3D(wp5.X, wp5.Y, wp5.Z, oct, lac, gain) - NoiseGenerator.FBM3D(wp6.X, wp6.Y, wp6.Z, oct, lac, gain)) * amp / (2f * eps);
        }

        private float ComputeSplatDisplacement(Vector3 p, float freq, float radius, float threshold, float smoothK)
        {
            float sum = 0f, totalW = 0f;
            for (int i = -2; i <= 2; i++)
                for (int j = -2; j <= 2; j++)
                    for (int k = -2; k <= 2; k++)
                    {
                        Vector3 off = new Vector3(i, j, k) * radius;
                        float hash = NoiseGenerator.HashFloat3(p.X * freq + off.X, p.Y * freq + off.Y, p.Z * freq + off.Z);
                        if (hash > threshold)
                        { float w = MathHelper.SmoothStep(threshold, threshold + smoothK, hash); sum += hash * w; totalW += w; }
                    }
            return totalW > 0.001f ? sum / totalW : 0f;
        }

        private Vector3 ComputeSplatVector(Vector3 p, float freq, float radius, float threshold, float smoothK)
        {
            Vector3 sum = Vector3.Zero;
            float totalW = 0f;
            for (int i = -2; i <= 2; i++)
                for (int j = -2; j <= 2; j++)
                    for (int k = -2; k <= 2; k++)
                    {
                        Vector3 off = new Vector3(i, j, k) * radius;
                        float hash = NoiseGenerator.HashFloat3(p.X * freq + off.X, p.Y * freq + off.Y, p.Z * freq + off.Z);
                        if (hash > threshold)
                        { float w = MathHelper.SmoothStep(threshold, threshold + smoothK, hash); sum += off * hash * w; totalW += w; }
                    }
            return totalW > 0.001f ? sum / totalW : Vector3.Zero;
        }

        private Vector3 EstimateNormal(Vector3 p, float freq, float amp, float t)
        {
            float eps = 0.01f;
            Func<float, float, float, float> modeNoise = Mode == DisplacementMode.Simplex ? NoiseGenerator.Simplex3D : (Func<float, float, float, float>)NoiseGenerator.Perlin3D;
            float dx1 = modeNoise((p.X + eps) * freq, p.Y * freq + t, p.Z * freq);
            float dx2 = modeNoise((p.X - eps) * freq, p.Y * freq + t, p.Z * freq);
            float dy1 = modeNoise(p.X * freq, (p.Y + eps) * freq + t, p.Z * freq);
            float dy2 = modeNoise(p.X * freq, (p.Y - eps) * freq + t, p.Z * freq);
            float dz1 = modeNoise(p.X * freq, p.Y * freq + t, (p.Z + eps) * freq);
            float dz2 = modeNoise(p.X * freq, p.Y * freq + t, (p.Z - eps) * freq);
            Vector3 grad = new Vector3((dx1 - dx2) / (2f * eps), (dy1 - dy2) / (2f * eps), (dz1 - dz2) / (2f * eps));
            return grad.LengthSquared() > 0.0001f ? Vector3.Normalize(grad) : Vector3.UnitY;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float eps = 0.001f;
            float[] wg = new float[ParamCount];
            var pg = ImmutableDictionary<string, float>.Empty;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = ComputeCurrentModeValue(_lastPosition);
                _weights[i] = oldW - eps;
                float vM = ComputeCurrentModeValue(_lastPosition);
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            Vector3 noiseGrad = ComputeNoiseGradient(_lastPosition, _weights[1], 0f, _weights[0],
                Mode == DisplacementMode.Simplex ? NoiseGenerator.Simplex3D : (Func<float, float, float, float>)NoiseGenerator.Perlin3D);
            return new NeuronGradient
            {
                ValueGradient = gradient.ValueGradient,
                PositionGradient = gradient.PositionGradient + gradient.ValueGradient * noiseGrad,
                HeightGradient = gradient.HeightGradient,
                ParameterGradients = pg,
                WeightGradients = wg
            };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 128;
        public INeuronKernel Clone() { var c = new DisplacementFieldKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() { if (_weights[0] < 0) return "Amplitude must be non-negative"; if (_weights[1] <= 0) return "Frequency must be positive"; return null; }
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters()
        {
            _weights[0] = 0.5f;
            _weights[1] = 1.0f;
            _weights[2] = 2.0f;
            _weights[3] = 0.5f;
            _weights[4] = 6f;
            _weights[5] = 0.5f;
            _weights[6] = 0.3f;
            _weights[7] = 1.0f;
            _weights[8] = 1.0f;
            _weights[9] = -10f;
            _weights[10] = 10f;
            _weights[11] = 1.0f;
            _weights[12] = 0.0f;
            _weights[13] = 0.5f;
            _weights[14] = 0.5f;
            _weights[15] = 0.1f;
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr)
        {
            if (gradient.WeightGradients != null)
                for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++)
                    _weights[i] -= lr * gradient.WeightGradients[i];
        }
    }

    #endregion


    #region Curvature Modulator Kernel

    public sealed class CurvatureModulatorKernel : INeuronKernel
    {
        private const int ParamCount = 10;
        private readonly float[] _weights = new float[ParamCount];
        private Vector3 _lastPosition;
        private NeuronOutput _lastOutput;

        public enum CurvatureMode
        {
            GaussianCurvature, MeanCurvature, PrincipalCurvatures,
            LaplacianSmoothing, CurvatureFlow, AbsoluteCurvature,
            Curvedness, ShapeIndex, AnisotropicCurvature, MixedPartialDerivatives
        }

        public string Name => "CurvatureModulator";
        public NeuronKind Kind => NeuronKind.CurvatureModulator;
        public int Version => 1;
        public CurvatureMode Mode { get; set; } = CurvatureMode.MeanCurvature;

        public CurvatureModulatorKernel() => ResetParameters();

        private float EvaluateField(Vector3 p) => NoiseGenerator.FBM3D(p.X, p.Y, p.Z, 4, 2f, 0.5f);

        public NeuronOutput Compute(in NeuronInput input)
        {
            float sampleRadius = _weights[0], flowRate = _weights[3];
            float remapMin = _weights[4], remapMax = _weights[5];
            float power = _weights[6], blendWithInput = _weights[7], normalInfluence = _weights[8], timeDecay = _weights[9];
            Vector3 p = input.Position;
            float t = input.Time;
            float curvature = ComputeCurvature(p, sampleRadius);
            float remapped = MathHelper.Remap(MathHelper.Pow(MathF.Abs(curvature), power) * MathHelper.Sign(curvature + 1e-10f), -2f, 2f, remapMin, remapMax);
            float blended = MathHelper.Lerp(input.Curvature, MathHelper.Clamp01(remapped), blendWithInput);
            if (timeDecay > 0 && t > 0)
                blended *= MathHelper.Exp(-timeDecay * t);
            Vector3 curvatureNormal = ComputeCurvatureNormal(p, sampleRadius);
            Vector3 outputNormal = Vector3.Normalize(MathHelper.Lerp(input.Normal, curvatureNormal, normalInfluence));
            float ao = ComputeCurvatureAO(p, curvature, sampleRadius);
            Vector3 color = CurvatureToColor(blended);

            _lastPosition = input.Position;
            _lastOutput = new NeuronOutput
            {
                Value = blended,
                Displacement = outputNormal * blended * 0.1f,
                Color = color,
                Roughness = MathHelper.Clamp01(MathHelper.Remap(MathF.Abs(blended), 0, 1, 0.9f, 0.1f)),
                Metallic = 0f,
                Opacity = 1f,
                Emission = MathHelper.Max(0, blended - 0.8f) * 5f,
                EmissionColor = new Vector3(1f, 0.3f, 0.1f),
                OutputNormal = outputNormal,
                Height = blended,
                AmbientOcclusion = ao,
                Thickness = MathHelper.Clamp01(MathF.Abs(blended) * 0.5f),
                IOR = 1.5f,
                Specular = MathHelper.Clamp01(MathF.Abs(blended)),
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("raw_curvature", curvature).Add("remapped_curvature", remapped)
                    .Add("curvature_magnitude", MathF.Abs(blended)),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("curvature_normal", curvatureNormal).Add("output_normal", outputNormal)
            };
            return _lastOutput;
        }

        private float ComputeCurvature(Vector3 p, float radius)
        {
            switch (Mode)
            {
                case CurvatureMode.GaussianCurvature:
                    return ComputeGaussianCurvature(p, radius);
                case CurvatureMode.MeanCurvature:
                    return ComputeMeanCurvature(p, radius);
                case CurvatureMode.AbsoluteCurvature:
                    return MathF.Abs(ComputeMeanCurvature(p, radius));
                case CurvatureMode.Curvedness:
                    var (k1, k2) = ComputePrincipalCurvatures(p, radius);
                    return MathF.Sqrt(0.5f * (k1 * k1 + k2 * k2));
                case CurvatureMode.ShapeIndex:
                    var (sk1, sk2) = ComputePrincipalCurvatures(p, radius);
                    float denom = sk1 - sk2;
                    return MathF.Abs(denom) > 1e-6f ? (2f / MathF.PI) * MathF.Atan((sk1 + sk2) / denom) : 0f;
                case CurvatureMode.AnisotropicCurvature:
                    return ComputeAnisotropicCurvature(p, radius);
                case CurvatureMode.MixedPartialDerivatives:
                    return ComputeMixedPartials(p, radius);
                case CurvatureMode.LaplacianSmoothing:
                    return ComputeLaplacian(p, radius);
                case CurvatureMode.CurvatureFlow:
                    return ComputeLaplacian(p, radius) * _weights[3];
                case CurvatureMode.PrincipalCurvatures:
                    var (pk1, pk2) = ComputePrincipalCurvatures(p, radius);
                    return (pk1 + pk2) * 0.5f;
                default:
                    return 0f;
            }
        }

        private float ComputeGaussianCurvature(Vector3 p, float radius)
        {
            float eps = radius * 0.05f;
            float dxx = EstSecond(p, Vector3.UnitX, eps);
            float dyy = EstSecond(p, Vector3.UnitY, eps);
            float dzz = EstSecond(p, Vector3.UnitZ, eps);
            float dxy = EstMixed(p, Vector3.UnitX, Vector3.UnitY, eps);
            float e = dxx / (1f + dxx * dxx + dxy * dxy + 0.001f);
            float g = dyy / (1f + dyy * dyy + 0.001f);
            float f = dxy / (1f + dxx * dxx + dyy * dyy + dzz * dzz + 0.001f);
            float E = 1f + dxx * dxx, G = 1f + dyy * dyy, F = dxy;
            float det = E * G - F * F;
            return det > 1e-6f ? (e * g - f * f) / det : 0f;
        }

        private float ComputeMeanCurvature(Vector3 p, float radius)
        {
            float eps = radius * 0.05f;
            float dxx = EstSecond(p, Vector3.UnitX, eps);
            float dyy = EstSecond(p, Vector3.UnitY, eps);
            float dxy = EstMixed(p, Vector3.UnitX, Vector3.UnitY, eps);
            float gradLen2 = dxx * dxx + dyy * dyy + 0.001f;
            float gradLen = MathF.Sqrt(gradLen2);
            float L = dxx / gradLen, M = dxy / gradLen, N = dyy / gradLen;
            float E = 1f + dxx * dxx, F = dxy, G = 1f + dyy * dyy;
            return (L * G - 2f * M * F + N * E) / (2f * (E * G - F * F + 0.001f));
        }

        private (float k1, float k2) ComputePrincipalCurvatures(Vector3 p, float radius)
        {
            float mean = ComputeMeanCurvature(p, radius);
            float gauss = ComputeGaussianCurvature(p, radius);
            float disc = MathHelper.Max(0f, mean * mean - gauss);
            float sqrtDisc = MathF.Sqrt(disc);
            return (mean + sqrtDisc, mean - sqrtDisc);
        }

        private float ComputeAnisotropicCurvature(Vector3 p, float radius)
        {
            Vector3 tangentDir = inputTangent(p);
            float eps = radius * 0.05f;
            float dtt = EstSecondDir(p, tangentDir, eps);
            Vector3 bitangent = Vector3.Cross(Vector3.UnitY, tangentDir);
            if (bitangent.LengthSquared() < 0.001f)
                bitangent = Vector3.UnitZ;
            bitangent = Vector3.Normalize(bitangent);
            return dtt - EstSecondDir(p, bitangent, eps);
        }

        private float ComputeMixedPartials(Vector3 p, float radius) => EstMixed(p, Vector3.UnitX, Vector3.UnitY, radius * 0.05f);

        private float ComputeLaplacian(Vector3 p, float radius)
        {
            float eps = radius * 0.1f;
            float center = EvaluateField(p);
            float sum = 0f;
            sum += EvaluateField(p + new Vector3(eps, 0, 0));
            sum += EvaluateField(p - new Vector3(eps, 0, 0));
            sum += EvaluateField(p + new Vector3(0, eps, 0));
            sum += EvaluateField(p - new Vector3(0, eps, 0));
            sum += EvaluateField(p + new Vector3(0, 0, eps));
            sum += EvaluateField(p - new Vector3(0, 0, eps));
            return (sum - 6f * center) / (eps * eps);
        }

        private float EstSecond(Vector3 p, Vector3 dir, float eps)
        {
            float d = EvaluateField(p);
            return (EvaluateField(p + dir * eps) - 2f * d + EvaluateField(p - dir * eps)) / (eps * eps);
        }

        private float EstSecondDir(Vector3 p, Vector3 dir, float eps)
        {
            float d = EvaluateField(p);
            return (EvaluateField(p + dir * eps) - 2f * d + EvaluateField(p - dir * eps)) / (eps * eps);
        }

        private float EstMixed(Vector3 p, Vector3 d1, Vector3 d2, float eps)
        {
            return (EvaluateField(p + d1 * eps + d2 * eps) - EvaluateField(p + d1 * eps - d2 * eps)
                  - EvaluateField(p - d1 * eps + d2 * eps) + EvaluateField(p - d1 * eps - d2 * eps)) / (4f * eps * eps);
        }

        private Vector3 inputTangent(Vector3 p) => Vector3.Normalize(new Vector3(MathF.Cos(p.X * 3f), 0, MathF.Sin(p.Z * 3f)));

        private Vector3 ComputeCurvatureNormal(Vector3 p, float radius)
        {
            float eps = radius * 0.05f;
            float dx = EvaluateField(p + new Vector3(eps, 0, 0)) - EvaluateField(p - new Vector3(eps, 0, 0));
            float dy = EvaluateField(p + new Vector3(0, eps, 0)) - EvaluateField(p - new Vector3(0, eps, 0));
            float dz = EvaluateField(p + new Vector3(0, 0, eps)) - EvaluateField(p - new Vector3(0, 0, eps));
            Vector3 grad = new Vector3(dx, dy, dz) / (2f * eps);
            return grad.LengthSquared() > 0.0001f ? Vector3.Normalize(grad) : Vector3.UnitY;
        }

        private float ComputeCurvatureAO(Vector3 p, float curvature, float radius)
        {
            float ao = 0f;
            for (int i = 1; i <= 4; i++)
            {
                float sampleR = radius * i * 0.25f;
                float sampleCurv = ComputeMeanCurvature(p + new Vector3(0, sampleR, 0), radius * 0.5f);
                ao += MathHelper.Max(0f, curvature - sampleCurv) * MathHelper.Pow(0.65f, i);
            }
            return MathHelper.Clamp01(1f - ao);
        }

        private Vector3 CurvatureToColor(float curvature)
        {
            float t = MathHelper.Clamp01(curvature * 0.5f + 0.5f);
            Vector3 cold = new Vector3(0.1f, 0.2f, 0.8f);
            Vector3 mid = new Vector3(0.1f, 0.9f, 0.1f);
            Vector3 hot = new Vector3(0.9f, 0.1f, 0.1f);
            return t < 0.5f ? Vector3.Lerp(cold, mid, t * 2f) : Vector3.Lerp(mid, hot, (t - 0.5f) * 2f);
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float eps = 0.001f;
            float[] wg = new float[ParamCount];
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, PositionGradient = gradient.PositionGradient, HeightGradient = gradient.HeightGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 128;
        public INeuronKernel Clone() { var c = new CurvatureModulatorKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() { if (_weights[0] <= 0) return "Sample radius must be positive"; return null; }
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters()
        {
            _weights[0] = 0.5f;
            _weights[1] = 16f;
            _weights[2] = 0f;
            _weights[3] = 0.1f;
            _weights[4] = 0f;
            _weights[5] = 1f;
            _weights[6] = 1f;
            _weights[7] = 1f;
            _weights[8] = 0.5f;
            _weights[9] = 0f;
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr)
        {
            if (gradient.WeightGradients != null)
                for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++)
                    _weights[i] -= lr * gradient.WeightGradients[i];
        }
    }

    #endregion

    #region Color Field Kernel

    public sealed class ColorFieldKernel : INeuronKernel
    {
        private const int ParamCount = 18;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum ColorMode { GradientMapping, TextureLookup, ColorBlending, HSVManipulation, NoiseColor, TemperatureToColor, Blackbody, Spectral, Cartoon, Quantize }

        public string Name => "ColorField";
        public NeuronKind Kind => NeuronKind.ColorField;
        public int Version => 1;
        public ColorMode Mode { get; set; } = ColorMode.GradientMapping;

        public ColorFieldKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float hueShift = _weights[0], sat = _weights[1], val = _weights[2];
            float contrast = _weights[3], brightness = _weights[4], gamma = _weights[5];
            Vector3 c1 = new(_weights[7], _weights[8], _weights[9]);
            Vector3 c2 = new(_weights[10], _weights[11], _weights[12]);
            Vector3 c3 = new(_weights[13], _weights[14], _weights[15]);
            Vector3 c4 = new(_weights[16], _weights[17], 0.5f);
            Vector3 baseColor;
            switch (Mode)
            {
                case ColorMode.GradientMapping:
                    float t = MathHelper.Clamp01(input.Position.Y * 0.5f + 0.5f);
                    baseColor = t < 0.33f ? Vector3.Lerp(c1, c2, t * 3f) : t < 0.66f ? Vector3.Lerp(c2, c3, (t - 0.33f) * 3f) : Vector3.Lerp(c3, c4, (t - 0.66f) * 3f);
                    break;
                case ColorMode.TextureLookup:
                    float u = MathHelper.Frac(input.TextureCoordinate.X), v = MathHelper.Frac(input.TextureCoordinate.Y);
                    baseColor = new Vector3(NoiseGenerator.Perlin2D(u * 4f, v * 4f) * 0.5f + 0.5f, NoiseGenerator.Perlin2D(u * 4f + 100f, v * 4f + 100f) * 0.5f + 0.5f, NoiseGenerator.Perlin2D(u * 4f + 200f, v * 4f + 200f) * 0.5f + 0.5f);
                    break;
                case ColorMode.ColorBlending:
                    float bt = MathHelper.Clamp01(NoiseGenerator.Perlin3D(input.Position.X, input.Position.Y, input.Position.Z) * 0.5f + 0.5f);
                    baseColor = Vector3.Lerp(Vector3.Lerp(c1, c2, bt), c3, MathHelper.Clamp01(_weights[6]));
                    break;
                case ColorMode.HSVManipulation:
                    baseColor = MathHelper.HSVToRGB(new Vector3(MathHelper.Wrap(hueShift + input.Position.Y * 0.1f, 0f, 1f), MathHelper.Clamp01(sat), MathHelper.Clamp01(val)));
                    break;
                case ColorMode.NoiseColor:
                    float nt = input.Time * 0.1f;
                    baseColor = new Vector3(NoiseGenerator.FBM3D(input.Position.X * 2f, input.Position.Y * 2f + nt, input.Position.Z * 2f, 4, 2f, 0.5f) * 0.5f + 0.5f, NoiseGenerator.FBM3D(input.Position.X * 2f + 31.7f, input.Position.Y * 2f + nt + 17.3f, input.Position.Z * 2f + 7.9f, 4, 2f, 0.5f) * 0.5f + 0.5f, NoiseGenerator.FBM3D(input.Position.X * 2f + 53.1f, input.Position.Y * 2f + nt + 41.9f, input.Position.Z * 2f + 23.3f, 4, 2f, 0.5f) * 0.5f + 0.5f);
                    break;
                case ColorMode.TemperatureToColor:
                    baseColor = TemperatureToColor(input.GetParameter("temperature", 6500f));
                    break;
                case ColorMode.Blackbody:
                    baseColor = BlackbodyRadiation(input.GetParameter("temperature", 6500f));
                    break;
                case ColorMode.Spectral:
                    baseColor = SpectralToRGB(input.GetParameter("wavelength", 550f));
                    break;
                case ColorMode.Cartoon:
                    float ct = MathHelper.Clamp01(input.Position.Y * 0.5f + 0.5f);
                    float cq = MathF.Floor(ct * 6f) / 6f;
                    baseColor = cq < 0.5f ? Vector3.Lerp(c1, c2, cq * 2f) : Vector3.Lerp(c2, c3, (cq - 0.5f) * 2f);
                    break;
                case ColorMode.Quantize:
                    float lum = Vector3.Dot(input.Normal * 0.5f + Vector3.One * 0.5f, new Vector3(0.299f, 0.587f, 0.114f));
                    float q = MathF.Floor(lum * 8f) / 8f;
                    baseColor = Vector3.Lerp(c1, c2, q);
                    break;
                default:
                    baseColor = Vector3.One;
                    break;
            }
            Vector3 hsv = MathHelper.RGBToHSV(baseColor);
            hsv.X = MathHelper.Wrap(hsv.X + hueShift, 0f, 1f);
            baseColor = MathHelper.HSVToRGB(hsv);
            baseColor = ApplyPostProcess(baseColor, sat, val, contrast, brightness, gamma);
            float luminance = MathHelper.Luminance(baseColor);

            _lastOutput = new NeuronOutput
            {
                Value = luminance,
                Color = baseColor,
                ColorRGBA = new Vector4(baseColor, input.GetParameter("alpha", 1f)),
                Roughness = input.GetParameter("roughness", 0.5f),
                Metallic = input.GetParameter("metallic", 0f),
                Opacity = input.GetParameter("alpha", 1f),
                Emission = luminance > 0.9f ? (luminance - 0.9f) * 10f : 0f,
                EmissionColor = baseColor,
                OutputNormal = input.Normal,
                Height = luminance,
                AmbientOcclusion = 1f,
                Thickness = 0f,
                IOR = 1.5f,
                Specular = 0.5f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("luminance", luminance).Add("hue", hsv.X).Add("saturation", hsv.Y).Add("value", hsv.Z),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("rgb_color", baseColor).Add("hsv_color", hsv)
            };
            return _lastOutput;
        }

        private Vector3 TemperatureToColor(float kelvin)
        {
            float temp = kelvin / 100f;
            float r, g, b;
            if (temp <= 66f)
            { r = 1f; g = MathHelper.Clamp(99.4708f * MathF.Log(temp) - 161.1196f, 0, 255) / 255f; b = temp <= 19f ? 0f : MathHelper.Clamp(138.5177f * MathF.Log(temp - 10f) - 305.0448f, 0, 255) / 255f; }
            else
            { r = MathHelper.Clamp(329.6987f * MathF.Pow(temp - 60f, -0.1332f), 0, 255) / 255f; g = MathHelper.Clamp(288.1222f * MathF.Pow(temp - 60f, -0.0755f), 0, 255) / 255f; b = 1f; }
            return new Vector3(MathHelper.Clamp01(r), MathHelper.Clamp01(g), MathHelper.Clamp01(b));
        }

        private Vector3 BlackbodyRadiation(float temperature)
        {
            float t = temperature;
            float r, g, b;
            if (t < 3500)
            { r = 1f; g = MathHelper.Clamp01(0.3901f * MathF.Pow(t / 3500f, 1.5073f)); b = MathHelper.Clamp01(0.0774f * MathF.Pow(t / 3500f, 2.4804f)); }
            else if (t < 6600)
            { r = MathHelper.Clamp01(MathF.Pow((t - 6000f) / 3900f, -0.1243f) * 1.2f); g = MathHelper.Clamp01(MathF.Pow(t / 6600f, -0.0659f)); b = 1f; }
            else
            { r = MathHelper.Clamp01(0.8777f * MathF.Pow(t / 6600f, 0.1074f)); g = MathHelper.Clamp01(0.9054f * MathF.Pow(t / 6600f, 0.0847f)); b = 1f; }
            return new Vector3(MathHelper.Clamp01(r), MathHelper.Clamp01(g), MathHelper.Clamp01(b));
        }

        private Vector3 SpectralToRGB(float wavelength)
        {
            float w = wavelength, r = 0f, g = 0f, b = 0f;
            if (w >= 380f && w < 440f)
            { r = -(w - 440f) / (440f - 380f); b = 1f; }
            else if (w >= 440f && w < 490f)
            { g = (w - 440f) / (490f - 440f); b = 1f; }
            else if (w >= 490f && w < 510f)
            { g = 1f; b = -(w - 510f) / (510f - 490f); }
            else if (w >= 510f && w < 580f)
            { r = (w - 510f) / (580f - 510f); g = 1f; }
            else if (w >= 580f && w < 645f)
            { r = 1f; g = -(w - 645f) / (645f - 580f); }
            else if (w >= 645f && w <= 780f)
            { r = 1f; }
            float factor = (w >= 380f && w < 420f) ? 0.3f + 0.7f * (w - 380f) / (420f - 380f) : (w >= 420f && w <= 700f) ? 1f : (w > 700f && w <= 780f) ? 0.3f + 0.7f * (780f - w) / (780f - 700f) : 0f;
            return new Vector3(MathHelper.Clamp01(r * factor), MathHelper.Clamp01(g * factor), MathHelper.Clamp01(b * factor));
        }

        private Vector3 ApplyPostProcess(Vector3 color, float sat, float val, float contrast, float brightness, float gamma)
        {
            Vector3 hsv = MathHelper.RGBToHSV(color);
            hsv.Y = MathHelper.Clamp01(hsv.Y * sat);
            hsv.Z = MathHelper.Clamp01(hsv.Z * val);
            color = MathHelper.HSVToRGB(hsv);
            color = (color - new Vector3(0.5f)) * contrast + new Vector3(0.5f) + new Vector3(brightness);
            if (gamma > 0.01f)
            { float invGamma = 1f / gamma; color = new Vector3(MathF.Pow(MathHelper.Clamp01(color.X), invGamma), MathF.Pow(MathHelper.Clamp01(color.Y), invGamma), MathF.Pow(MathHelper.Clamp01(color.Z), invGamma)); }
            return MathHelper.Clamp01Vec(color);
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = MathHelper.Luminance(_lastOutput.Color);
                _weights[i] = oldW - eps;
                float vM = MathHelper.Luminance(_lastOutput.Color);
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new ColorFieldKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters()
        {
            _weights[0] = 0f;
            _weights[1] = 1f;
            _weights[2] = 1f;
            _weights[3] = 1f;
            _weights[4] = 0f;
            _weights[5] = 2.2f;
            _weights[6] = 0.5f;
            _weights[7] = 1f;
            _weights[8] = 0f;
            _weights[9] = 0f;
            _weights[10] = 0f;
            _weights[11] = 1f;
            _weights[12] = 0f;
            _weights[13] = 0f;
            _weights[14] = 0f;
            _weights[15] = 1f;
            _weights[16] = 1f;
            _weights[17] = 1f;
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr)
        {
            if (gradient.WeightGradients != null)
                for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++)
                    _weights[i] -= lr * gradient.WeightGradients[i];
        }
    }

    #endregion

    #region Normal Map Kernel

    public sealed class NormalMapKernel : INeuronKernel
    {
        private const int ParamCount = 8;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum NormalMapMode { TangentSpace, ObjectSpace, HeightFieldEstimation, FiniteDifferences, CrossProduct, AnalyticalNormal }

        public string Name => "NormalMap";
        public NeuronKind Kind => NeuronKind.NormalMap;
        public int Version => 1;
        public NormalMapMode Mode { get; set; } = NormalMapMode.TangentSpace;

        public NormalMapKernel() => ResetParameters();

        private float SampleHeight(Vector3 p) => NoiseGenerator.FBM3D(p.X * 2f, p.Y * 2f, p.Z * 2f, 4, 2f, 0.5f) * _weights[1];

        public NeuronOutput Compute(in NeuronInput input)
        {
            float strength = _weights[0], heightScale = _weights[1], sampleStep = _weights[2];
            float microDetailScale = _weights[6], microDetailStrength = _weights[7];
            Vector3 computedNormal;
            switch (Mode)
            {
                case NormalMapMode.TangentSpace:
                    float hC = SampleHeight(input.Position);
                    float hR = SampleHeight(input.Position + input.Tangent * sampleStep);
                    float hU = SampleHeight(input.Position + input.Bitangent * sampleStep);
                    computedNormal = Vector3.Normalize(new Vector3((hC - hR) * heightScale, (hC - hU) * heightScale, 1f));
                    break;
                case NormalMapMode.ObjectSpace:
                    float eps = sampleStep;
                    float hx1 = SampleHeight(input.Position + new Vector3(eps, 0, 0));
                    float hx2 = SampleHeight(input.Position - new Vector3(eps, 0, 0));
                    float hy1 = SampleHeight(input.Position + new Vector3(0, eps, 0));
                    float hy2 = SampleHeight(input.Position - new Vector3(0, eps, 0));
                    float hz1 = SampleHeight(input.Position + new Vector3(0, 0, eps));
                    float hz2 = SampleHeight(input.Position - new Vector3(0, 0, eps));
                    computedNormal = Vector3.Normalize(input.Normal + new Vector3((hx2 - hx1) * heightScale, (hy2 - hy1) * heightScale, (hz2 - hz1) * heightScale));
                    break;
                case NormalMapMode.HeightFieldEstimation:
                    float heps = sampleStep * 0.5f;
                    float hR2 = SampleHeight(input.Position + new Vector3(heps, 0, 0));
                    float hL2 = SampleHeight(input.Position - new Vector3(heps, 0, 0));
                    float hU2 = SampleHeight(input.Position + new Vector3(0, heps, 0));
                    float hD2 = SampleHeight(input.Position - new Vector3(0, heps, 0));
                    Vector3 tangent = Vector3.Normalize(new Vector3(2f * heps, 0, (hR2 - hL2) * heightScale));
                    Vector3 bitangent = Vector3.Normalize(new Vector3(0, 2f * heps, (hU2 - hD2) * heightScale));
                    computedNormal = Vector3.Normalize(Vector3.Cross(tangent, bitangent));
                    break;
                case NormalMapMode.FiniteDifferences:
                    float fe = sampleStep;
                    float fdx = (SampleHeight(input.Position + new Vector3(fe, 0, 0)) - SampleHeight(input.Position - new Vector3(fe, 0, 0))) / (2f * fe);
                    float fdy = (SampleHeight(input.Position + new Vector3(0, fe, 0)) - SampleHeight(input.Position - new Vector3(0, fe, 0))) / (2f * fe);
                    computedNormal = Vector3.Normalize(new Vector3(-fdx * heightScale, -fdy * heightScale, 1f));
                    break;
                case NormalMapMode.CrossProduct:
                    Vector3 t2 = input.Tangent.LengthSquared() > 0.001f ? input.Tangent : Vector3.UnitX;
                    Vector3 b2 = Vector3.Cross(input.Normal, t2);
                    if (b2.LengthSquared() < 0.001f)
                        b2 = Vector3.Cross(input.Normal, Vector3.UnitX);
                    b2 = Vector3.Normalize(b2);
                    t2 = Vector3.Cross(b2, input.Normal);
                    computedNormal = Vector3.Normalize(Vector3.Cross(t2, b2));
                    break;
                case NormalMapMode.AnalyticalNormal:
                    float x = input.Position.X, y = input.Position.Y, z = input.Position.Z;
                    computedNormal = Vector3.Normalize(new Vector3(MathF.Sin(x * 3f) * MathF.Cos(z * 3f) * 0.5f, 1f, MathF.Cos(x * 3f) * MathF.Sin(z * 3f) * 0.5f));
                    break;
                default:
                    computedNormal = input.Normal;
                    break;
            }
            if (microDetailStrength > 0)
            {
                float me = 0.001f;
                float mh = NoiseGenerator.Perlin3D(input.Position.X * microDetailScale, input.Position.Y * microDetailScale, input.Position.Z * microDetailScale);
                float mhx = NoiseGenerator.Perlin3D((input.Position.X + me) * microDetailScale, input.Position.Y * microDetailScale, input.Position.Z * microDetailScale);
                float mhy = NoiseGenerator.Perlin3D(input.Position.X * microDetailScale, (input.Position.Y + me) * microDetailScale, input.Position.Z * microDetailScale);
                Vector3 microNormal = Vector3.Normalize(new Vector3((mh - mhx) / me, (mh - mhy) / me, 1f));
                computedNormal = Vector3.Normalize(Vector3.Lerp(computedNormal, microNormal, microDetailStrength));
            }
            computedNormal = Vector3.Normalize(Vector3.Lerp(input.Normal, computedNormal, MathHelper.Clamp01(strength)));
            float normalDot = Vector3.Dot(computedNormal, input.Normal);

            _lastOutput = new NeuronOutput
            {
                Value = normalDot,
                Displacement = (computedNormal - input.Normal) * heightScale,
                Color = new Vector3(MathHelper.Remap(computedNormal.X, -1, 1, 0, 1), MathHelper.Remap(computedNormal.Y, -1, 1, 0, 1), MathHelper.Remap(computedNormal.Z, -1, 1, 0, 1)),
                Roughness = MathHelper.Clamp01(0.5f + MathF.Abs(normalDot - 1f) * 2f),
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = computedNormal,
                Height = MathHelper.Remap(normalDot, -1, 1, 0, 1),
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty.Add("normal_dot", normalDot).Add("normal_length", computedNormal.Length()),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty.Add("computed_normal", computedNormal)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new NormalMapKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 1f; _weights[1] = 1f; _weights[2] = 0.01f; _weights[3] = 0f; _weights[4] = 0.5f; _weights[5] = 1f; _weights[6] = 10f; _weights[7] = 0.3f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Roughness Map Kernel

    public sealed class RoughnessMapKernel : INeuronKernel
    {
        private const int ParamCount = 8;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum RoughnessMode { GGX, Beckmann, CharlierIsotropic, CharlierAnisotropic, NoiseMapped, CurvatureMapped, DistanceMapped, Composite }

        public string Name => "RoughnessMap";
        public NeuronKind Kind => NeuronKind.RoughnessMap;
        public int Version => 1;
        public RoughnessMode Mode { get; set; } = RoughnessMode.GGX;

        public RoughnessMapKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float baseR = _weights[0], range = _weights[1], scale = _weights[2], amp = _weights[3];
            float anisoAngle = _weights[4], anisoStr = _weights[5], curvInfl = _weights[6], distFalloff = _weights[7];
            float roughness;
            switch (Mode)
            {
                case RoughnessMode.GGX:
                    roughness = MathHelper.Clamp01(baseR + NoiseGenerator.Perlin3D(input.Position.X * scale, input.Position.Y * scale, input.Position.Z * scale) * amp);
                    break;
                case RoughnessMode.Beckmann:
                    float n = NoiseGenerator.Simplex3D(input.Position.X * scale, input.Position.Y * scale, input.Position.Z * scale);
                    roughness = MathHelper.Clamp01(MathF.Exp(-n * n / (baseR * baseR + 1e-7f)) + n * amp * 0.5f);
                    break;
                case RoughnessMode.CharlierIsotropic:
                    float cn1 = NoiseGenerator.Perlin3D(input.Position.X * scale, input.Position.Y * scale, input.Position.Z * scale);
                    float cn2 = NoiseGenerator.Perlin3D(input.Position.X * scale * 2.3f + 100f, input.Position.Y * scale * 2.3f + 100f, input.Position.Z * scale * 2.3f);
                    roughness = MathHelper.Clamp01(baseR + (cn1 * 0.7f + cn2 * 0.3f) * amp);
                    break;
                case RoughnessMode.CharlierAnisotropic:
                    float cosA = MathF.Cos(anisoAngle), sinA = MathF.Sin(anisoAngle);
                    float u = input.Position.X * cosA + input.Position.Z * sinA;
                    float v = -input.Position.X * sinA + input.Position.Z * cosA;
                    float an1 = NoiseGenerator.Perlin2D(u * 3f, v * 3f), an2 = NoiseGenerator.Perlin2D(u * 3f + 50f, v * 3f + 50f);
                    roughness = MathHelper.Clamp01(MathF.Sqrt(MathHelper.Lerp(baseR, baseR * (1f + an1 * anisoStr), MathF.Abs(cosA)) * MathHelper.Lerp(baseR, baseR * (1f + an2 * anisoStr), MathF.Abs(sinA))) * 0.707f);
                    break;
                case RoughnessMode.NoiseMapped:
                    roughness = MathHelper.Clamp01(baseR + NoiseGenerator.FBM3D(input.Position.X * scale, input.Position.Y * scale, input.Position.Z * scale, 4, 2f, 0.5f) * amp);
                    break;
                case RoughnessMode.CurvatureMapped:
                    roughness = MathHelper.Clamp01(baseR + MathF.Abs(input.Curvature) * curvInfl * 0.5f);
                    break;
                case RoughnessMode.DistanceMapped:
                    float distFactor = MathHelper.Exp(-input.DistanceToCamera * distFalloff);
                    roughness = MathHelper.Clamp01(baseR * distFactor + (1f - distFactor) * 0.9f);
                    break;
                case RoughnessMode.Composite:
                    float rn = MathHelper.Clamp01(baseR + NoiseGenerator.FBM3D(input.Position.X * scale, input.Position.Y * scale, input.Position.Z * scale, 4, 2f, 0.5f) * amp);
                    float rc = MathHelper.Clamp01(baseR + MathF.Abs(input.Curvature) * curvInfl * 0.5f);
                    float df = MathHelper.Exp(-input.DistanceToCamera * distFalloff);
                    float rd = MathHelper.Clamp01(baseR * df + (1f - df) * 0.9f);
                    roughness = MathHelper.Clamp01(rn * 0.5f + rc * 0.3f + rd * 0.2f);
                    break;
                default:
                    roughness = baseR;
                    break;
            }
            roughness = MathHelper.Clamp01(roughness * range + (1f - range) * 0.5f);
            float alpha = roughness * roughness;
            float alpha2 = alpha * alpha;
            Vector3 halfDir = Vector3.Normalize(input.LightDirection + input.ViewDirection);
            float ndh = MathHelper.Max(0, Vector3.Dot(input.Normal, halfDir));
            float ggxDist = alpha2 / (MathHelper.Pi * MathF.Pow(ndh * ndh * (alpha2 - 1f) + 1f, 2f) + 1e-7f);
            float ndv = MathHelper.Max(0, Vector3.Dot(input.Normal, input.ViewDirection));
            float ndl = MathHelper.Max(0, Vector3.Dot(input.Normal, input.LightDirection));
            float geo = MathHelper.GeometrySchlickGGX(ndv, alpha) * MathHelper.GeometrySchlickGGX(ndl, alpha);

            _lastOutput = new NeuronOutput
            {
                Value = roughness,
                Roughness = roughness,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = input.Normal,
                Height = roughness,
                AmbientOcclusion = MathHelper.Clamp01(1f - roughness * 0.3f),
                Color = new Vector3(roughness, roughness, roughness),
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("roughness", roughness).Add("alpha", alpha).Add("ggx_distribution", ggxDist).Add("geometry_term", geo),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new RoughnessMapKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 0.5f; _weights[1] = 1f; _weights[2] = 2f; _weights[3] = 0.2f; _weights[4] = 0f; _weights[5] = 0f; _weights[6] = 0.3f; _weights[7] = 0.1f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Metallic Map Kernel

    public sealed class MetallicMapKernel : INeuronKernel
    {
        private const int ParamCount = 6;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum MetallicMode { Constant, NoiseMapped, GradientMapped, TextureBased, CurvatureDriven, Composite }

        public string Name => "MetallicMap";
        public NeuronKind Kind => NeuronKind.MetallicMap;
        public int Version => 1;
        public MetallicMode Mode { get; set; } = MetallicMode.NoiseMapped;

        public MetallicMapKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float baseMetallic = _weights[0], threshold = _weights[1], smoothness = _weights[2];
            float noiseScale = _weights[3], curvBias = _weights[4], gradBlend = _weights[5];
            float metallic;
            switch (Mode)
            {
                case MetallicMode.Constant:
                    metallic = baseMetallic;
                    break;
                case MetallicMode.NoiseMapped:
                    float n = NoiseGenerator.Perlin3D(input.Position.X * noiseScale, input.Position.Y * noiseScale, input.Position.Z * noiseScale) * 0.5f + 0.5f;
                    metallic = MathHelper.SmoothStep(threshold - smoothness, threshold + smoothness, n);
                    metallic = MathHelper.Lerp(baseMetallic, metallic, gradBlend);
                    break;
                case MetallicMode.GradientMapped:
                    float gt = MathHelper.Clamp01(input.Position.Y * 0.5f + 0.5f);
                    metallic = MathHelper.Lerp(baseMetallic, MathHelper.SmoothStep(threshold - smoothness, threshold + smoothness, gt), gradBlend);
                    break;
                case MetallicMode.TextureBased:
                    float u = input.TextureCoordinate.X * noiseScale, v = input.TextureCoordinate.Y * noiseScale;
                    metallic = MathHelper.Clamp01(baseMetallic + (NoiseGenerator.Perlin2D(u, v) * 0.3f + NoiseGenerator.Perlin2D(u * 2.1f + 50f, v * 2.1f + 50f) * 0.2f + 0.5f) * gradBlend);
                    break;
                case MetallicMode.CurvatureDriven:
                    metallic = MathHelper.Clamp01(baseMetallic + MathHelper.Remap(MathF.Abs(input.Curvature), 0, 2, 0, 1) * curvBias);
                    break;
                case MetallicMode.Composite:
                    float cn = NoiseGenerator.Perlin3D(input.Position.X * noiseScale, input.Position.Y * noiseScale, input.Position.Z * noiseScale) * 0.5f + 0.5f;
                    metallic = MathHelper.Clamp01(baseMetallic + cn * 0.5f + MathHelper.Remap(MathF.Abs(input.Curvature), 0, 2, 0, 1) * curvBias * 0.3f + gradBlend * 0.2f);
                    break;
                default:
                    metallic = baseMetallic;
                    break;
            }
            metallic = MathHelper.Clamp01(metallic);
            Vector3 baseColor = new Vector3(input.GetParameter("base_color_r", 0.8f), input.GetParameter("base_color_g", 0.8f), input.GetParameter("base_color_b", 0.8f));
            Vector3 dielectricF0 = new(0.04f, 0.04f, 0.04f);
            Vector3 F0 = Vector3.Lerp(dielectricF0, baseColor, metallic);

            _lastOutput = new NeuronOutput
            {
                Value = metallic,
                Roughness = input.GetParameter("roughness", 0.5f),
                Metallic = metallic,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = input.Normal,
                Height = metallic,
                AmbientOcclusion = 1f,
                IOR = metallic > 0.5f ? 2.5f : 1.5f,
                Specular = metallic > 0.5f ? 1f : 0.5f,
                Color = Vector3.Lerp(baseColor * 0.04f, baseColor, metallic),
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("metallic", metallic).Add("f0_r", F0.X).Add("f0_g", F0.Y).Add("f0_b", F0.Z),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty.Add("f0", F0)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new MetallicMapKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 0f; _weights[1] = 0.5f; _weights[2] = 0.1f; _weights[3] = 2f; _weights[4] = 0.3f; _weights[5] = 1f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion


    #region Emissive Map Kernel

    public sealed class EmissiveMapKernel : INeuronKernel
    {
        private const int ParamCount = 8;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum EmissiveMode { Constant, Pulsing, NoiseDriven, GradientEmissive, HDRMapped, BloomContribution }

        public string Name => "EmissiveMap";
        public NeuronKind Kind => NeuronKind.EmissiveMap;
        public int Version => 1;
        public EmissiveMode Mode { get; set; } = EmissiveMode.Pulsing;

        public EmissiveMapKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float intensity = _weights[0], freq = _weights[1], pulseWidth = _weights[2];
            float r = _weights[3], g = _weights[4], b = _weights[5];
            float noiseScale = _weights[6], bloomThreshold = _weights[7];
            Vector3 emissionColor = new Vector3(r, g, b);
            float emission;

            switch (Mode)
            {
                case EmissiveMode.Constant:
                    emission = intensity;
                    break;
                case EmissiveMode.Pulsing:
                    emission = intensity * (0.5f + 0.5f * MathF.Sin(input.Time * freq * MathHelper.TwoPi));
                    emission *= MathHelper.SmoothStep(0.5f - pulseWidth, 0.5f + pulseWidth, emission);
                    break;
                case EmissiveMode.NoiseDriven:
                    float n = NoiseGenerator.Perlin3D(input.Position.X * noiseScale + input.Time * 0.1f, input.Position.Y * noiseScale, input.Position.Z * noiseScale) * 0.5f + 0.5f;
                    emission = intensity * n;
                    emissionColor = emissionColor * n;
                    break;
                case EmissiveMode.GradientEmissive:
                    float gt = MathHelper.Clamp01(input.Position.Y * 0.5f + 0.5f);
                    emission = intensity * MathHelper.SmoothStep(0.4f, 0.6f, gt);
                    break;
                case EmissiveMode.HDRMapped:
                    float lum = MathHelper.Luminance(emissionColor);
                    emission = intensity * MathHelper.Pow(lum, 2.2f);
                    break;
                case EmissiveMode.BloomContribution:
                    float fbm = NoiseGenerator.FBM3D(input.Position.X * noiseScale, input.Position.Y * noiseScale + input.Time * 0.05f, input.Position.Z * noiseScale, 4, 2f, 0.5f) * 0.5f + 0.5f;
                    emission = fbm > bloomThreshold ? intensity * (fbm - bloomThreshold) / (1f - bloomThreshold + 0.001f) : 0f;
                    emissionColor = emissionColor * fbm;
                    break;
                default:
                    emission = 0f;
                    break;
            }

            float bloomContribution = MathHelper.Max(0f, emission - bloomThreshold) / (bloomThreshold + 0.001f);

            _lastOutput = new NeuronOutput
            {
                Value = emission,
                Emission = emission,
                EmissionColor = emissionColor,
                Color = emissionColor * emission,
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                OutputNormal = input.Normal,
                Height = emission,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("emission_intensity", emission)
                    .Add("emission_luminance", MathHelper.Luminance(emissionColor * emission))
                    .Add("bloom_contribution", bloomContribution),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("emission_color", emissionColor)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Emission;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Emission;
                _weights[i] = oldW;
                wg[i] = gradient.EmissionGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, EmissionGradient = gradient.EmissionGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new EmissiveMapKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 1f; _weights[1] = 1f; _weights[2] = 0.1f; _weights[3] = 1f; _weights[4] = 0.8f; _weights[5] = 0.4f; _weights[6] = 2f; _weights[7] = 0.8f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Opacity Map Kernel

    public sealed class OpacityMapKernel : INeuronKernel
    {
        private const int ParamCount = 6;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum OpacityMode { Constant, AlphaCutout, AlphaBlend, NoiseMasked, GradientFalloff, Dithered }

        public string Name => "OpacityMap";
        public NeuronKind Kind => NeuronKind.OpacityMap;
        public int Version => 1;
        public OpacityMode Mode { get; set; } = OpacityMode.AlphaBlend;

        public OpacityMapKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float baseOpacity = _weights[0], cutoff = _weights[1], smoothness = _weights[2];
            float noiseScale = _weights[3], falloffPower = _weights[4], ditherScale = _weights[5];
            float opacity;

            switch (Mode)
            {
                case OpacityMode.Constant:
                    opacity = baseOpacity;
                    break;
                case OpacityMode.AlphaCutout:
                    float heightVal = MathHelper.Remap(input.Position.Y, -2f, 2f, 0f, 1f);
                    opacity = MathHelper.Step(cutoff, heightVal) * baseOpacity;
                    break;
                case OpacityMode.AlphaBlend:
                    float h2 = MathHelper.Remap(input.Position.Y, -2f, 2f, 0f, 1f);
                    opacity = MathHelper.SmoothStep(cutoff - smoothness, cutoff + smoothness, h2) * baseOpacity;
                    break;
                case OpacityMode.NoiseMasked:
                    float n = NoiseGenerator.Perlin3D(input.Position.X * noiseScale, input.Position.Y * noiseScale, input.Position.Z * noiseScale) * 0.5f + 0.5f;
                    opacity = MathHelper.SmoothStep(cutoff - smoothness, cutoff + smoothness, n) * baseOpacity;
                    break;
                case OpacityMode.GradientFalloff:
                    float falloff = MathF.Pow(MathHelper.Clamp01(1f - MathF.Abs(Vector3.Dot(input.Normal, input.ViewDirection))), falloffPower);
                    opacity = MathHelper.Lerp(baseOpacity, 1f, falloff);
                    break;
                case OpacityMode.Dithered:
                    float hd = MathHelper.Remap(input.Position.Y, -2f, 2f, 0f, 1f);
                    float dither = NoiseGenerator.HashFloat3(input.Position.X * ditherScale, input.Position.Y * ditherScale, input.Position.Z * ditherScale);
                    opacity = MathHelper.Step(cutoff, hd + (dither - 0.5f) * 0.01f) * baseOpacity;
                    break;
                default:
                    opacity = baseOpacity;
                    break;
            }
            opacity = MathHelper.Clamp01(opacity);

            _lastOutput = new NeuronOutput
            {
                Value = opacity,
                Opacity = opacity,
                Color = new Vector3(opacity, opacity, opacity),
                Roughness = 0.5f,
                Metallic = 0f,
                Emission = 0f,
                OutputNormal = input.Normal,
                Height = opacity,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("opacity", opacity).Add("alpha_cutout", opacity > cutoff ? 1f : 0f),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Opacity;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Opacity;
                _weights[i] = oldW;
                wg[i] = gradient.OpacityGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, OpacityGradient = gradient.OpacityGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new OpacityMapKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 1f; _weights[1] = 0.5f; _weights[2] = 0.1f; _weights[3] = 2f; _weights[4] = 2f; _weights[5] = 100f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Subsurface Scattering Kernel

    public sealed class SubsurfaceScatteringKernel : INeuronKernel
    {
        private const int ParamCount = 10;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum SSSMode { DiffusionProfile, PreIntegrated, ScreenSpace, RandomWalk, Dipole, ChristensenBurley }

        public string Name => "SubsurfaceScattering";
        public NeuronKind Kind => NeuronKind.SubsurfaceScattering;
        public int Version => 1;
        public SSSMode Mode { get; set; } = SSSMode.ChristensenBurley;

        public SubsurfaceScatteringKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float intensity = _weights[0], radius = _weights[1];
            float r = _weights[2], g = _weights[3], b = _weights[4];
            float curvatureTint = _weights[5], thicknessScale = _weights[6];
            float distortion = _weights[7], power = _weights[8], scale = _weights[9];

            Vector3 scatterColor = new Vector3(r, g, b);
            float thickness = MathHelper.Clamp01(MathF.Abs(input.Curvature) * thicknessScale);
            float ndl = MathHelper.Max(0, Vector3.Dot(input.Normal, input.LightDirection));

            Vector3 lightWrap = Vector3.Normalize(input.LightDirection + input.Normal * distortion);
            float wrapDiffuse = MathHelper.Max(0, (ndl + distortion) / (1f + distortion));
            wrapDiffuse = MathHelper.Pow(wrapDiffuse, power) * scale;

            float curvatureTintFactor = MathHelper.Clamp01(MathF.Abs(input.Curvature) * curvatureTint);
            Vector3 sssColor = scatterColor * curvatureTintFactor;

            float profile;
            switch (Mode)
            {
                case SSSMode.DiffusionProfile:
                    float r2 = radius * radius;
                    profile = MathHelper.Exp(-ndl * ndl / (2f * r2)) / (MathF.Sqrt(MathHelper.TwoPi) * radius);
                    break;
                case SSSMode.PreIntegrated:
                    float preInt = 0.5f * MathHelper.Exp(-input.DistanceToCamera * 0.1f / radius) + 0.5f * ndl;
                    profile = preInt;
                    break;
                case SSSMode.Dipole:
                    float d = radius * 3f;
                    float muS = 1f / (d + 0.001f);
                    profile = muS * MathHelper.Exp(-ndl * muS) * thickness;
                    break;
                case SSSMode.ChristensenBurley:
                    float muT = 1f / (radius + 0.001f);
                    float sigmaT = muT * 0.75f;
                    profile = MathHelper.Exp(-ndl * sigmaT) * (1f - MathHelper.Exp(-2f * sigmaT)) * thickness;
                    break;
                default:
                    profile = wrapDiffuse;
                    break;
            }

            Vector3 finalSSS = scatterColor * profile * wrapDiffuse * intensity * thickness;

            _lastOutput = new NeuronOutput
            {
                Value = profile * intensity,
                Color = finalSSS,
                Subsurface = MathHelper.Clamp01(intensity),
                Thickness = thickness,
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = input.Normal,
                Height = profile,
                AmbientOcclusion = MathHelper.Clamp01(1f - thickness * 0.5f),
                IOR = 1.333f,
                Specular = 0.5f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("sss_intensity", intensity).Add("sss_profile", profile).Add("thickness", thickness),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("scatter_color", scatterColor).Add("sss_result", finalSSS)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new SubsurfaceScatteringKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 1f; _weights[1] = 0.5f; _weights[2] = 0.9f; _weights[3] = 0.2f; _weights[4] = 0.1f; _weights[5] = 1f; _weights[6] = 1f; _weights[7] = 0.5f; _weights[8] = 2f; _weights[9] = 1f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Tessellation Kernel

    public sealed class TessellationKernel : INeuronKernel
    {
        private const int ParamCount = 8;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum TessellationMode { DistanceBased, CurvatureBased, ScreenSpace, Adaptive, Hybrid, PN_Triangles, Displacement, EdgeLength }

        public string Name => "Tessellation";
        public NeuronKind Kind => NeuronKind.Tessellation;
        public int Version => 1;
        public TessellationMode Mode { get; set; } = TessellationMode.Adaptive;

        public TessellationKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float tessFactor = _weights[0], minDist = _weights[1], maxDist = _weights[2];
            float maxTess = _weights[3], curvatureSensitivity = _weights[4];
            float screenAlignFactor = _weights[5], edgeLengthThreshold = _weights[6], displacementScale = _weights[7];

            float tessellationFactor;
            switch (Mode)
            {
                case TessellationMode.DistanceBased:
                    float distFactor = MathHelper.Clamp01((input.DistanceToCamera - minDist) / (maxDist - minDist));
                    tessellationFactor = MathHelper.Lerp(maxTess, tessFactor, distFactor);
                    break;
                case TessellationMode.CurvatureBased:
                    float curvFactor = MathHelper.Clamp01(MathF.Abs(input.Curvature) * curvatureSensitivity);
                    tessellationFactor = MathHelper.Lerp(tessFactor, maxTess, curvFactor);
                    break;
                case TessellationMode.ScreenSpace:
                    float pixelSize = tessFactor / (input.DistanceToCamera + 0.001f);
                    tessellationFactor = MathHelper.Clamp(pixelSize, tessFactor, maxTess);
                    break;
                case TessellationMode.Adaptive:
                    float distAdaptive = MathHelper.Clamp01((input.DistanceToCamera - minDist) / (maxDist - minDist));
                    float curvAdaptive = MathHelper.Clamp01(MathF.Abs(input.Curvature) * curvatureSensitivity);
                    tessellationFactor = MathHelper.Lerp(maxTess, tessFactor, distAdaptive) * MathHelper.Lerp(1f, 2f, curvAdaptive);
                    break;
                case TessellationMode.Hybrid:
                    float screenTess = tessFactor / (input.DistanceToCamera + 0.001f);
                    float curvTess = MathF.Abs(input.Curvature) * curvatureSensitivity * maxTess;
                    tessellationFactor = MathHelper.Clamp(MathHelper.Lerp(screenTess, curvTess, screenAlignFactor), tessFactor, maxTess);
                    break;
                case TessellationMode.PN_Triangles:
                    tessellationFactor = MathHelper.Clamp(tessFactor * (1f + MathF.Abs(input.Curvature)), tessFactor, maxTess);
                    break;
                case TessellationMode.Displacement:
                    tessellationFactor = MathHelper.Clamp(tessFactor + displacementScale * MathF.Abs(input.Curvature) * maxTess, tessFactor, maxTess);
                    break;
                case TessellationMode.EdgeLength:
                    tessellationFactor = MathHelper.Clamp(edgeLengthThreshold / (input.DistanceToCamera * 0.01f + 0.001f), tessFactor, maxTess);
                    break;
                default:
                    tessellationFactor = tessFactor;
                    break;
            }
            tessellationFactor = MathHelper.Clamp(tessellationFactor, tessFactor, maxTess);
            int tessLevel = MathHelper.Clamp(MathHelper.RoundToInt(tessellationFactor), 1, 64);
            Vector3 tessNormal = input.Normal * displacementScale * (1f / tessellationFactor);

            _lastOutput = new NeuronOutput
            {
                Value = tessellationFactor,
                Displacement = tessNormal,
                Height = tessellationFactor / maxTess,
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = input.Normal,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("tessellation_factor", tessellationFactor)
                    .Add("tessellation_level", tessLevel)
                    .Add("edge_length", edgeLengthThreshold / tessellationFactor),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("displacement_direction", tessNormal)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new TessellationKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => _weights[0] <= 0 ? "Min tessellation must be positive" : null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 1f; _weights[1] = 1f; _weights[2] = 100f; _weights[3] = 16f; _weights[4] = 1f; _weights[5] = 0.5f; _weights[6] = 0.1f; _weights[7] = 0.5f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Occlusion Culling Kernel

    public sealed class OcclusionCullingKernel : INeuronKernel
    {
        private const int ParamCount = 6;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum OcclusionMode { Frustum, OcclusionQuery, HierarchicalZ, SoftwareRasterize, BitMask, Hybrid }

        public string Name => "OcclusionCulling";
        public NeuronKind Kind => NeuronKind.OcclusionCulling;
        public int Version => 1;
        public OcclusionMode Mode { get; set; } = OcclusionMode.Frustum;

        public OcclusionCullingKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float cullDistance = _weights[0], marginFactor = _weights[1];
            float lodBias = _weights[2], maxScreenCoverage = _weights[3];
            float voxelSize = _weights[4], hierarchyDepth = _weights[5];

            bool isVisible = true;
            float visibilityScore = 1f;

            switch (Mode)
            {
                case OcclusionMode.Frustum:
                    float dist = input.DistanceToCamera;
                    isVisible = dist < cullDistance && dist > 0.01f;
                    visibilityScore = isVisible ? MathHelper.Clamp01(1f - dist / cullDistance) : 0f;
                    break;
                case OcclusionMode.OcclusionQuery:
                    float ao = NoiseGenerator.Perlin3D(input.Position.X * 0.5f, input.Position.Y * 0.5f, input.Position.Z * 0.5f) * 0.5f + 0.5f;
                    isVisible = ao > 0.2f;
                    visibilityScore = ao;
                    break;
                case OcclusionMode.HierarchicalZ:
                    float hzLevel = MathF.Floor(MathHelper.Log2(MathHelper.Max(1f, input.DistanceToCamera)));
                    float hzThreshold = MathHelper.Clamp01(1f - hzLevel / hierarchyDepth);
                    isVisible = input.DistanceToCamera < cullDistance * hzThreshold;
                    visibilityScore = hzThreshold;
                    break;
                case OcclusionMode.SoftwareRasterize:
                    float screenCoverage = 1f / (input.DistanceToCamera * input.DistanceToCamera + 0.001f);
                    isVisible = screenCoverage > maxScreenCoverage;
                    visibilityScore = MathHelper.Clamp01(screenCoverage / maxScreenCoverage);
                    break;
                case OcclusionMode.BitMask:
                    int hash = MathHelper.FloorToInt(input.Position.X * 7.3f + input.Position.Y * 13.7f + input.Position.Z * 23.1f);
                    isVisible = (hash & 0xFF) % 3 != 0;
                    visibilityScore = isVisible ? 1f : 0f;
                    break;
                case OcclusionMode.Hybrid:
                    float distH = input.DistanceToCamera;
                    float aoH = NoiseGenerator.Perlin3D(input.Position.X * 0.5f, input.Position.Y * 0.5f, input.Position.Z * 0.5f) * 0.5f + 0.5f;
                    isVisible = distH < cullDistance && aoH > 0.15f;
                    visibilityScore = MathHelper.Clamp01(aoH * (1f - distH / cullDistance));
                    break;
            }

            if (isVisible && input.DistanceToCamera > cullDistance * marginFactor)
                visibilityScore *= 0.5f;

            float lodLevel = MathHelper.Clamp(MathHelper.Log2(MathHelper.Max(1f, input.DistanceToCamera)) * lodBias, 0f, 8f);

            _lastOutput = new NeuronOutput
            {
                Value = isVisible ? 1f : 0f,
                Opacity = visibilityScore,
                Color = isVisible ? new Vector3(0.2f, 0.8f, 0.2f) : new Vector3(0.8f, 0.2f, 0.2f),
                Roughness = 0.5f,
                Metallic = 0f,
                Emission = 0f,
                OutputNormal = input.Normal,
                Height = visibilityScore,
                AmbientOcclusion = visibilityScore,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("is_visible", isVisible ? 1f : 0f)
                    .Add("visibility_score", visibilityScore)
                    .Add("lod_level", lodLevel)
                    .Add("distance_to_camera", input.DistanceToCamera),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, OpacityGradient = gradient.OpacityGradient, WeightGradients = new float[ParamCount], ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new OcclusionCullingKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 100f; _weights[1] = 1.2f; _weights[2] = 1f; _weights[3] = 0.001f; _weights[4] = 0.1f; _weights[5] = 8f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { }
    }

    #endregion

    #region Animation Curve Kernel

    public sealed class AnimationCurveKernel : INeuronKernel
    {
        private const int ParamCount = 32;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum InterpolationMode { Linear, Cubic, CatmullRom, Bezier, Hermite, Stepped, EaseIn, EaseOut, EaseInOut, Spring }

        public string Name => "AnimationCurve";
        public NeuronKind Kind => NeuronKind.AnimationCurve;
        public int Version => 1;
        public InterpolationMode Mode { get; set; } = InterpolationMode.Cubic;

        public AnimationCurveKernel() => ResetParameters();

        private float EvaluateKeyframes(float time)
        {
            int keyCount = MathHelper.Clamp((int)_weights[8], 2, 10);
            float lastKey = 0f, lastVal = _weights[0];
            for (int i = 1; i < keyCount; i++)
            {
                float keyTime = _weights[9 + i];
                float keyVal = _weights[19 + i];
                if (time <= keyTime || i == keyCount - 1)
                {
                    float t = MathHelper.InverseLerp(lastKey, keyTime, time);
                    switch (Mode)
                    {
                        case InterpolationMode.Linear:
                            return MathHelper.Lerp(lastVal, keyVal, t);
                        case InterpolationMode.Cubic:
                            float prevVal = i >= 2 ? _weights[19 + i - 1] : lastVal;
                            float nextVal = i < keyCount - 1 ? _weights[19 + i + 1] : keyVal;
                            return MathHelper.CubicInterpolate(prevVal, lastVal, keyVal, nextVal, t);
                        case InterpolationMode.CatmullRom:
                            float prev2 = i >= 2 ? _weights[19 + i - 1] : lastVal;
                            float next2 = i < keyCount - 1 ? _weights[19 + i + 1] : keyVal;
                            return MathHelper.CubicInterpolate(prev2, lastVal, keyVal, next2, t);
                        case InterpolationMode.Bezier:
                            float cp1 = _weights[4] + lastVal;
                            float cp2 = _weights[5] + keyVal;
                            return BezierEvaluate(lastVal, cp1, cp2, keyVal, t);
                        case InterpolationMode.Hermite:
                            float tang1 = _weights[6] * (keyTime - lastKey);
                            float tang2 = _weights[7] * (keyTime - lastKey);
                            return HermiteEvaluate(lastVal, keyVal, tang1, tang2, t);
                        case InterpolationMode.Stepped:
                            return lastVal;
                        case InterpolationMode.EaseIn:
                            return MathHelper.Lerp(lastVal, keyVal, t * t);
                        case InterpolationMode.EaseOut:
                            return MathHelper.Lerp(lastVal, keyVal, 1f - (1f - t) * (1f - t));
                        case InterpolationMode.EaseInOut:
                            float et = t < 0.5f ? 2f * t * t : 1f - MathF.Pow(-2f * t + 2f, 2f) / 2f;
                            return MathHelper.Lerp(lastVal, keyVal, et);
                        case InterpolationMode.Spring:
                            float springT = 1f - MathF.Exp(-6f * t) * MathF.Cos(MathHelper.TwoPi * 3f * t);
                            return MathHelper.Lerp(lastVal, keyVal, springT);
                        default:
                            return MathHelper.Lerp(lastVal, keyVal, t);
                    }
                }
                lastKey = keyTime;
                lastVal = keyVal;
            }
            return lastVal;
        }

        private float BezierEvaluate(float p0, float p1, float p2, float p3, float t)
        {
            float mt = 1f - t;
            float mt2 = mt * mt, mt3 = mt2 * mt;
            float t2 = t * t, t3 = t2 * t;
            return mt3 * p0 + 3f * mt2 * t * p1 + 3f * mt * t2 * p2 + t3 * p3;
        }

        private float HermiteEvaluate(float p0, float p1, float m0, float m1, float t)
        {
            float t2 = t * t, t3 = t2 * t;
            return (2f * t3 - 3f * t2 + 1f) * p0 + (t3 - 2f * t2 + t) * m0 + (-2f * t3 + 3f * t2) * p1 + (t3 - t2) * m1;
        }

        public NeuronOutput Compute(in NeuronInput input)
        {
            float time = input.GetParameter("animation_time", input.Time);
            float loopDuration = _weights[1];
            bool loop = _weights[2] > 0.5f;

            if (loop && loopDuration > 0)
                time = MathHelper.Repeat01(time, loopDuration);

            float value = EvaluateKeyframes(time);

            float velocity = (value - EvaluateKeyframes(MathHelper.Max(0, time - 0.016f))) / 0.016f;
            float acceleration = 0f;

            _lastOutput = new NeuronOutput
            {
                Value = value,
                Displacement = new Vector3(value, velocity * 0.1f, acceleration * 0.01f),
                Color = new Vector3(value, MathHelper.Clamp01(velocity * 0.1f), MathHelper.Clamp01(acceleration * 0.01f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = input.Normal,
                Height = value,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("animated_value", value)
                    .Add("velocity", velocity)
                    .Add("time", time),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new AnimationCurveKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters()
        {
            _weights[0] = 0f;
            _weights[1] = 1f;
            _weights[2] = 0f;
            _weights[3] = 0f;
            _weights[4] = 0.5f;
            _weights[5] = 0.5f;
            _weights[6] = 0f;
            _weights[7] = 0f;
            _weights[8] = 3f;
            _weights[9] = 0f;
            _weights[10] = 0.5f;
            _weights[11] = 1f;
            _weights[19] = 0f;
            _weights[20] = 1f;
            _weights[21] = 0f;
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion


    #region Particle Emitter Kernel

    public sealed class ParticleEmitterKernel : INeuronKernel
    {
        private const int ParamCount = 14;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public string Name => "ParticleEmitter";
        public NeuronKind Kind => NeuronKind.ParticleEmitter;
        public int Version => 1;

        public ParticleEmitterKernel() => ResetParameters();

        private readonly Random _rng = new Random(42);

        public NeuronOutput Compute(in NeuronInput input)
        {
            float spawnRate = _weights[0], lifetime = _weights[1];
            float speedX = _weights[2], speedY = _weights[3], speedZ = _weights[4];
            float gravity = _weights[5], drag = _weights[6];
            float sizeStart = _weights[7], sizeEnd = _weights[8];
            float alphaStart = _weights[9], alphaEnd = _weights[10];
            float spreadAngle = _weights[11], maxParticles = _weights[12];
            float noiseInfluence = _weights[13];

            float time = input.Time;
            float particleAge = MathHelper.Repeat01(time, lifetime);
            float normalizedAge = particleAge / lifetime;

            Vector3 velocity = new Vector3(speedX, speedY - gravity * particleAge, speedZ);
            velocity *= (1f - drag * particleAge);

            if (noiseInfluence > 0)
            {
                Vector3 noiseVel = NoiseGenerator.CurlNoise3D(
                    input.Position.X * 0.5f + time * 0.3f,
                    input.Position.Y * 0.5f,
                    input.Position.Z * 0.5f) * noiseInfluence;
                velocity += noiseVel;
            }

            float cosHalfSpread = MathHelper.Cos(spreadAngle * 0.5f);
            if (velocity.LengthSquared() > 0.001f)
            {
                velocity = Vector3.Normalize(velocity);
                float angle = MathF.Acos(MathHelper.Clamp(cosHalfSpread, -1f, 1f));
                velocity = Vector3.Transform(velocity, Quaternion.CreateFromAxisAngle(Vector3.Cross(Vector3.UnitY, velocity), angle * (float)_rng.NextDouble()));
            }
            velocity *= speedY;

            float currentSize = MathHelper.Lerp(sizeStart, sizeEnd, normalizedAge);
            float currentAlpha = MathHelper.Lerp(alphaStart, alphaEnd, normalizedAge);
            float emission = spawnRate * (1f - normalizedAge) * MathHelper.Step(normalizedAge, 0.01f);

            Vector3 color = new Vector3(
                MathHelper.Lerp(1f, 0.2f, normalizedAge),
                MathHelper.Lerp(0.8f, 0.1f, normalizedAge),
                MathHelper.Lerp(0.2f, 0.0f, normalizedAge));

            _lastOutput = new NeuronOutput
            {
                Value = emission,
                Displacement = velocity * input.DeltaTime,
                Color = color,
                Opacity = currentAlpha,
                Emission = emission * 2f,
                EmissionColor = color,
                Roughness = 0.5f,
                Metallic = 0f,
                OutputNormal = velocity.LengthSquared() > 0.001f ? Vector3.Normalize(velocity) : Vector3.UnitY,
                Height = currentSize,
                AmbientOcclusion = 1f,
                Thickness = currentSize,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("emission_rate", emission)
                    .Add("particle_age", normalizedAge)
                    .Add("particle_size", currentSize)
                    .Add("particle_alpha", currentAlpha)
                    .Add("active_particles", MathHelper.Clamp(maxParticles * emission, 0, maxParticles)),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("velocity", velocity)
                    .Add("color", color)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 128;
        public INeuronKernel Clone() { var c = new ParticleEmitterKernel(); Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 100f; _weights[1] = 2f; _weights[2] = 0f; _weights[3] = 5f; _weights[4] = 0f; _weights[5] = 9.81f; _weights[6] = 0.1f; _weights[7] = 0.1f; _weights[8] = 0.01f; _weights[9] = 1f; _weights[10] = 0f; _weights[11] = 0.5f; _weights[12] = 1000f; _weights[13] = 0.5f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Fluid Solver Kernel

    public sealed class FluidSolverKernel : INeuronKernel
    {
        private const int ParamCount = 10;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum FluidMode { Advection, Diffusion, PressureSolve, VorticityConfinement, FullNavierStokes }

        public string Name => "FluidSolver";
        public NeuronKind Kind => NeuronKind.FluidSolver;
        public int Version => 1;
        public FluidMode Mode { get; set; } = FluidMode.FullNavierStokes;

        public FluidSolverKernel() => ResetParameters();

        private float ComputeAdvection(Vector3 p, Vector3 vel, float dt, float dissipation)
        {
            Vector3 backPos = p - vel * dt;
            return NoiseGenerator.Perlin3D(backPos.X, backPos.Y, backPos.Z) * dissipation;
        }

        private float ComputeDiffusion(Vector3 p, float visc, float dt)
        {
            float eps = 0.01f;
            float c = NoiseGenerator.Perlin3D(p.X, p.Y, p.Z);
            float lap = NoiseGenerator.Perlin3D(p.X + eps, p.Y, p.Z) + NoiseGenerator.Perlin3D(p.X - eps, p.Y, p.Z)
                      + NoiseGenerator.Perlin3D(p.X, p.Y + eps, p.Z) + NoiseGenerator.Perlin3D(p.X, p.Y - eps, p.Z)
                      + NoiseGenerator.Perlin3D(p.X, p.Y, p.Z + eps) + NoiseGenerator.Perlin3D(p.X, p.Y, p.Z - eps)
                      - 6f * c;
            return c + visc * dt * lap / (eps * eps);
        }

        private float ComputePressure(Vector3 p, float density, float dt)
        {
            float eps = 0.01f;
            float div = (NoiseGenerator.Perlin3D(p.X + eps, p.Y, p.Z) - NoiseGenerator.Perlin3D(p.X - eps, p.Y, p.Z)
                       + NoiseGenerator.Perlin3D(p.X, p.Y + eps, p.Z) - NoiseGenerator.Perlin3D(p.X, p.Y - eps, p.Z)
                       + NoiseGenerator.Perlin3D(p.X, p.Y, p.Z + eps) - NoiseGenerator.Perlin3D(p.X, p.Y, p.Z - eps)) / (2f * eps);
            return density * div * dt;
        }

        private Vector3 ComputeVorticity(Vector3 p, float confinement, float dt)
        {
            float eps = 0.01f;
            float uC = NoiseGenerator.Perlin3D(p.X, p.Y, p.Z);
            float uXp = NoiseGenerator.Perlin3D(p.X + eps, p.Y, p.Z);
            float uXn = NoiseGenerator.Perlin3D(p.X - eps, p.Y, p.Z);
            float uYp = NoiseGenerator.Perlin3D(p.X, p.Y + eps, p.Z);
            float uYn = NoiseGenerator.Perlin3D(p.X, p.Y - eps, p.Z);
            float uZp = NoiseGenerator.Perlin3D(p.X, p.Y, p.Z + eps);
            float uZn = NoiseGenerator.Perlin3D(p.X, p.Y, p.Z - eps);

            float omegaX = (uYp - uYn) - (uZp - uZn);
            float omegaY = (uZp - uZn) - (uXp - uXn);
            float omegaZ = (uXp - uXn) - (uYp - uYn);
            omegaX *= 0.5f / eps;
            omegaY *= 0.5f / eps;
            omegaZ *= 0.5f / eps;

            float omLen = MathF.Sqrt(omegaX * omegaX + omegaY * omegaY + omegaZ * omegaZ) + 1e-5f;
            float N_x = (MathF.Sqrt((omegaX + eps) * (omegaX + eps)) - MathF.Sqrt((omegaX - eps) * (omegaX - eps))) / (2f * eps * omLen);
            float N_y = (MathF.Sqrt((omegaY + eps) * (omegaY + eps)) - MathF.Sqrt((omegaY - eps) * (omegaY - eps))) / (2f * eps * omLen);
            float N_z = (MathF.Sqrt((omegaZ + eps) * (omegaZ + eps)) - MathF.Sqrt((omegaZ - eps) * (omegaZ - eps))) / (2f * eps * omLen);

            return new Vector3(N_x, N_y, N_z) * confinement * omLen * dt;
        }

        public NeuronOutput Compute(in NeuronInput input)
        {
            float viscosity = _weights[0], diffusion = _weights[1], dt = _weights[2];
            float vorticityConfinement = _weights[3], pressureScale = _weights[4];
            float dissipation = _weights[5], turbulence = _weights[6];
            float buoyancy = _weights[7], damping = _weights[8], iterations = _weights[9];

            Vector3 p = input.Position;
            float velocity = 0f, pressure = 0f, density = 0f;

            switch (Mode)
            {
                case FluidMode.Advection:
                    Vector3 advVel = new Vector3(MathF.Sin(p.Y * 2f + input.Time), MathF.Cos(p.X * 2f + input.Time), 0);
                    velocity = ComputeAdvection(p, advVel, dt, dissipation);
                    density = velocity;
                    break;
                case FluidMode.Diffusion:
                    velocity = ComputeDiffusion(p, viscosity, dt);
                    density = velocity;
                    break;
                case FluidMode.PressureSolve:
                    for (int i = 0; i < (int)iterations; i++)
                        pressure = ComputePressure(p, pressureScale, dt);
                    velocity = pressure;
                    density = pressure;
                    break;
                case FluidMode.VorticityConfinement:
                    Vector3 vort = ComputeVorticity(p, vorticityConfinement, dt);
                    velocity = vort.Length();
                    density = velocity;
                    break;
                case FluidMode.FullNavierStokes:
                    for (int i = 0; i < (int)iterations; i++)
                    {
                        Vector3 velField = new Vector3(
                            NoiseGenerator.Perlin3D(p.X + input.Time, p.Y, p.Z),
                            NoiseGenerator.Perlin3D(p.X, p.Y + input.Time, p.Z),
                            NoiseGenerator.Perlin3D(p.X, p.Y, p.Z + input.Time));
                        float adv = ComputeAdvection(p, velField, dt, dissipation);
                        float diff = ComputeDiffusion(p, viscosity, dt);
                        float pres = ComputePressure(p, pressureScale, dt);
                        Vector3 vortNs = ComputeVorticity(p, vorticityConfinement, dt);
                        velocity = MathHelper.Lerp(velocity, adv + diff + pres + vortNs.Length(), 1f / (i + 1));
                    }
                    density = velocity * diffusion;
                    break;
            }

            float buoyancyEffect = buoyancy * (1f - p.Y * 0.5f) * dt;
            Vector3 flowDir = new Vector3(MathF.Sin(input.Time + p.Z), buoyancyEffect, MathF.Cos(input.Time + p.X));
            flowDir *= damping;

            _lastOutput = new NeuronOutput
            {
                Value = velocity,
                Displacement = flowDir * dt,
                Color = new Vector3(MathHelper.Clamp01(density), MathHelper.Clamp01(velocity * 0.5f), MathHelper.Clamp01(pressureScale * 0.1f)),
                Roughness = 0.3f,
                Metallic = 0f,
                Opacity = MathHelper.Clamp01(density),
                Emission = MathHelper.Max(0, velocity - 0.8f) * 3f,
                EmissionColor = new Vector3(0.2f, 0.5f, 1f),
                OutputNormal = flowDir.LengthSquared() > 0.001f ? Vector3.Normalize(flowDir) : Vector3.UnitY,
                Height = density,
                AmbientOcclusion = MathHelper.Clamp01(1f - density * 0.5f),
                Thickness = density,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("velocity_magnitude", velocity)
                    .Add("pressure", pressure)
                    .Add("density", density),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("flow_direction", flowDir)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 128;
        public INeuronKernel Clone() { var c = new FluidSolverKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 0.01f; _weights[1] = 0.5f; _weights[2] = 0.016f; _weights[3] = 0.5f; _weights[4] = 1f; _weights[5] = 0.99f; _weights[6] = 0.1f; _weights[7] = 1f; _weights[8] = 0.99f; _weights[9] = 20f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Cloth Simulator Kernel

    public sealed class ClothSimulatorKernel : INeuronKernel
    {
        private const int ParamCount = 12;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public string Name => "ClothSimulator";
        public NeuronKind Kind => NeuronKind.ClothSimulator;
        public int Version => 1;

        public ClothSimulatorKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float stiffness = _weights[0], damping = _weights[1], mass = _weights[2];
            float gravity = _weights[3], windX = _weights[4], windY = _weights[5], windZ = _weights[6];
            float windTurbulence = _weights[7], stretchStiff = _weights[8], bendStiff = _weights[9];
            float shearStiff = _weights[10], constraintIter = _weights[11];

            Vector3 wind = new Vector3(windX, windY, windZ);
            if (windTurbulence > 0)
            {
                wind += NoiseGenerator.CurlNoise3D(
                    input.Position.X * 0.5f + input.Time * 0.3f,
                    input.Position.Y * 0.5f,
                    input.Position.Z * 0.5f) * windTurbulence;
            }

            Vector3 gravityForce = new Vector3(0, -gravity, 0) * mass;
            Vector3 windForce = wind * MathF.Sin(input.Time * 2f + input.Position.X * 3f + input.Position.Z * 3f);
            Vector3 totalForce = gravityForce + windForce;

            float eps = 0.01f;
            float strainX = (NoiseGenerator.Perlin3D((input.Position.X + eps) * 5f, input.Position.Y * 5f, input.Position.Z * 5f)
                           - NoiseGenerator.Perlin3D((input.Position.X - eps) * 5f, input.Position.Y * 5f, input.Position.Z * 5f)) / (2f * eps);
            float strainY = (NoiseGenerator.Perlin3D(input.Position.X * 5f, (input.Position.Y + eps) * 5f, input.Position.Z * 5f)
                           - NoiseGenerator.Perlin3D(input.Position.X * 5f, (input.Position.Y - eps) * 5f, input.Position.Z * 5f)) / (2f * eps);

            Vector3 springForce = new Vector3(-strainX * stretchStiff, -strainY * stretchStiff, 0) * stiffness;
            float bendResist = -NoiseGenerator.Perlin3D(input.Position.X * 10f, input.Position.Y * 10f, input.Position.Z * 10f) * bendStiff;

            Vector3 acceleration = (totalForce + springForce) / mass;
            acceleration *= (1f - damping);

            Vector3 displacement = acceleration * input.DeltaTime * input.DeltaTime;
            float tension = springForce.Length();

            Vector3 clothNormal = Vector3.Normalize(new Vector3(-strainX, 1f, -strainY));
            clothNormal = Vector3.Lerp(clothNormal, input.Normal, 0.5f);

            _lastOutput = new NeuronOutput
            {
                Value = tension,
                Displacement = displacement,
                Color = new Vector3(MathHelper.Clamp01(tension * 0.5f), MathHelper.Clamp01(1f - tension * 0.3f), MathHelper.Clamp01(bendResist * 0.5f + 0.5f)),
                Roughness = MathHelper.Clamp01(0.6f + tension * 0.2f),
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = clothNormal,
                Height = MathHelper.Clamp01(tension),
                AmbientOcclusion = MathHelper.Clamp01(1f - MathF.Abs(bendResist) * 0.3f),
                Thickness = 0.01f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("tension", tension)
                    .Add("bend_resistance", bendResist)
                    .Add("strain_x", strainX)
                    .Add("strain_y", strainY)
                    .Add("displacement_magnitude", displacement.Length()),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("cloth_normal", clothNormal)
                    .Add("force", totalForce + springForce)
                    .Add("wind", wind)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 128;
        public INeuronKernel Clone() { var c = new ClothSimulatorKernel(); Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 100f; _weights[1] = 0.1f; _weights[2] = 1f; _weights[3] = 9.81f; _weights[4] = 2f; _weights[5] = 0f; _weights[6] = 1f; _weights[7] = 0.5f; _weights[8] = 50f; _weights[9] = 10f; _weights[10] = 20f; _weights[11] = 5f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Voxelizer Kernel

    public sealed class VoxelizerKernel : INeuronKernel
    {
        private const int ParamCount = 8;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum VoxelMode { SurfaceVoxelization, SolidVoxelization, SDFToVoxel, AdaptiveResolution, DualGrid, Hierarchical }

        public string Name => "Voxelizer";
        public NeuronKind Kind => NeuronKind.Voxelizer;
        public int Version => 1;
        public VoxelMode Mode { get; set; } = VoxelMode.SDFToVoxel;

        public VoxelizerKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float voxelSize = _weights[0], resolution = _weights[1];
            float sdfThreshold = _weights[2], smoothing = _weights[3];
            float fillDensity = _weights[4], adaptiveFactor = _weights[5];
            float temporalBlend = _weights[6], maxResolution = _weights[7];

            Vector3 voxelPos = Vector3.Round(input.Position / voxelSize) * voxelSize;
            float sdf = NoiseGenerator.FBM3D(voxelPos.X * 2f, voxelPos.Y * 2f, voxelPos.Z * 2f, 4, 2f, 0.5f);

            float voxelDensity = 0f;
            float occupancy = 0f;

            switch (Mode)
            {
                case VoxelMode.SurfaceVoxelization:
                    float surfaceDist = MathF.Abs(sdf);
                    voxelDensity = MathHelper.SmoothStep(voxelSize * smoothing, 0f, surfaceDist);
                    occupancy = voxelDensity > sdfThreshold ? 1f : 0f;
                    break;
                case VoxelMode.SolidVoxelization:
                    voxelDensity = sdf < 0 ? fillDensity : 0f;
                    occupancy = voxelDensity > 0 ? 1f : 0f;
                    break;
                case VoxelMode.SDFToVoxel:
                    voxelDensity = MathHelper.Remap(sdf, -voxelSize, voxelSize, 1f, 0f);
                    voxelDensity = MathHelper.Clamp01(voxelDensity);
                    occupancy = MathHelper.SmoothStep(sdfThreshold - smoothing, sdfThreshold + smoothing, voxelDensity);
                    break;
                case VoxelMode.AdaptiveResolution:
                    float localRes = resolution * (1f + adaptiveFactor * MathF.Abs(sdf) / voxelSize);
                    localRes = MathHelper.Min(localRes, maxResolution);
                    voxelDensity = MathHelper.Clamp01(localRes / maxResolution);
                    occupancy = voxelDensity > sdfThreshold ? 1f : 0f;
                    break;
                case VoxelMode.DualGrid:
                    float coarse = NoiseGenerator.ValueNoise3D(voxelPos.X * resolution, voxelPos.Y * resolution, voxelPos.Z * resolution);
                    float fine = NoiseGenerator.Perlin3D(voxelPos.X * resolution * 2f, voxelPos.Y * resolution * 2f, voxelPos.Z * resolution * 2f) * 0.5f + 0.5f;
                    voxelDensity = MathHelper.Lerp(coarse, fine, 0.5f);
                    occupancy = voxelDensity > sdfThreshold ? 1f : 0f;
                    break;
                case VoxelMode.Hierarchical:
                    voxelDensity = 0f;
                    float currentRes = 1f;
                    while (currentRes <= resolution)
                    {
                        voxelDensity += NoiseGenerator.ValueNoise3D(voxelPos.X * currentRes, voxelPos.Y * currentRes, voxelPos.Z * currentRes) / currentRes;
                        currentRes *= 2f;
                    }
                    voxelDensity = MathHelper.Clamp01(voxelDensity);
                    occupancy = voxelDensity > sdfThreshold ? 1f : 0f;
                    break;
            }

            Vector3 voxelNormal = Vector3.Normalize(new Vector3(
                NoiseGenerator.Perlin3D((voxelPos.X + 0.01f) * 2f, voxelPos.Y * 2f, voxelPos.Z * 2f) - NoiseGenerator.Perlin3D((voxelPos.X - 0.01f) * 2f, voxelPos.Y * 2f, voxelPos.Z * 2f),
                NoiseGenerator.Perlin3D(voxelPos.X * 2f, (voxelPos.Y + 0.01f) * 2f, voxelPos.Z * 2f) - NoiseGenerator.Perlin3D(voxelPos.X * 2f, (voxelPos.Y - 0.01f) * 2f, voxelPos.Z * 2f),
                NoiseGenerator.Perlin3D(voxelPos.X * 2f, voxelPos.Y * 2f, (voxelPos.Z + 0.01f) * 2f) - NoiseGenerator.Perlin3D(voxelPos.X * 2f, voxelPos.Y * 2f, (voxelPos.Z - 0.01f) * 2f)));

            _lastOutput = new NeuronOutput
            {
                Value = voxelDensity,
                Displacement = voxelPos - input.Position,
                Color = new Vector3(occupancy, voxelDensity, 1f - occupancy),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = occupancy,
                OutputNormal = voxelNormal,
                Height = voxelDensity,
                AmbientOcclusion = occupancy,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("voxel_density", voxelDensity)
                    .Add("occupancy", occupancy)
                    .Add("voxel_size", voxelSize),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("voxel_position", voxelPos)
                    .Add("voxel_normal", voxelNormal)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 128;
        public INeuronKernel Clone() { var c = new VoxelizerKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => _weights[0] <= 0 ? "Voxel size must be positive" : null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 0.1f; _weights[1] = 16f; _weights[2] = 0.5f; _weights[3] = 0.5f; _weights[4] = 1f; _weights[5] = 0.5f; _weights[6] = 0.1f; _weights[7] = 64f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Marching Cube Kernel

    public sealed class MarchingCubeKernel : INeuronKernel
    {
        private const int ParamCount = 6;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum MarchingMode { MarchingCubes, DualContouring, SurfaceNets, MarchingTetrahedra, NaiveSurfaceNets, DualMarchingCubes }

        public string Name => "MarchingCube";
        public NeuronKind Kind => NeuronKind.MarchingCube;
        public int Version => 1;
        public MarchingMode Mode { get; set; } = MarchingMode.MarchingCubes;

        public MarchingCubeKernel() => ResetParameters();

        private static readonly int[,] EdgeTable = new int[256, 16];
        private static readonly int[,] TriTable = new int[256, 16];

        private static readonly int[] CubeEdgeFlags = new int[256]
        {
            0x000,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
            0x190,0x099,0x393,0x29a,0x596,0x49f,0x795,0x69c,0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
            0x230,0x339,0x033,0x13a,0x636,0x73f,0x435,0x53c,0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
            0x3a0,0x2a9,0x1a3,0x0aa,0x7a6,0x6af,0x5a5,0x4ac,0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
            0x460,0x569,0x663,0x76a,0x066,0x16f,0x265,0x36c,0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
            0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0x0ff,0x3f5,0x2fc,0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
            0x650,0x759,0x453,0x55a,0x256,0x35f,0x055,0x15c,0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
            0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0x0cc,0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
            0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,0x0cc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
            0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,0x15c,0x055,0x35f,0x256,0x55a,0x453,0x759,0x650,
            0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,0x2fc,0x3f5,0x0ff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
            0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,0x36c,0x265,0x16f,0x066,0x76a,0x663,0x569,0x460,
            0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,0x4ac,0x5a5,0x6af,0x7a6,0x0aa,0x1a3,0x2a9,0x3a0,
            0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,0x53c,0x435,0x73f,0x636,0x13a,0x033,0x339,0x230,
            0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,0x69c,0x795,0x49f,0x596,0x29a,0x393,0x099,0x190,
            0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x000
        };

        public NeuronOutput Compute(in NeuronInput input)
        {
            float isoLevel = _weights[0], gridScale = _weights[1];
            float smoothNormals = _weights[2], adaptivity = _weights[3];
            float contourWeight = _weights[4], normalStrength = _weights[5];

            Vector3 gridPos = Vector3.Round(input.Position / gridScale) * gridScale;
            float sdf = NoiseGenerator.FBM3D(gridPos.X * 2f, gridPos.Y * 2f, gridPos.Z * 2f, 4, 2f, 0.5f);

            float corner000 = sdf;
            float corner100 = NoiseGenerator.FBM3D((gridPos.X + gridScale) * 2f, gridPos.Y * 2f, gridPos.Z * 2f, 4, 2f, 0.5f);
            float corner010 = NoiseGenerator.FBM3D(gridPos.X * 2f, (gridPos.Y + gridScale) * 2f, gridPos.Z * 2f, 4, 2f, 0.5f);
            float corner110 = NoiseGenerator.FBM3D((gridPos.X + gridScale) * 2f, (gridPos.Y + gridScale) * 2f, gridPos.Z * 2f, 4, 2f, 0.5f);
            float corner001 = NoiseGenerator.FBM3D(gridPos.X * 2f, gridPos.Y * 2f, (gridPos.Z + gridScale) * 2f, 4, 2f, 0.5f);
            float corner101 = NoiseGenerator.FBM3D((gridPos.X + gridScale) * 2f, gridPos.Y * 2f, (gridPos.Z + gridScale) * 2f, 4, 2f, 0.5f);
            float corner011 = NoiseGenerator.FBM3D(gridPos.X * 2f, (gridPos.Y + gridScale) * 2f, (gridPos.Z + gridScale) * 2f, 4, 2f, 0.5f);
            float corner111 = NoiseGenerator.FBM3D((gridPos.X + gridScale) * 2f, (gridPos.Y + gridScale) * 2f, (gridPos.Z + gridScale) * 2f, 4, 2f, 0.5f);

            int cubeIndex = 0;
            if (corner000 < isoLevel)
                cubeIndex |= 1;
            if (corner100 < isoLevel)
                cubeIndex |= 2;
            if (corner110 < isoLevel)
                cubeIndex |= 4;
            if (corner010 < isoLevel)
                cubeIndex |= 8;
            if (corner001 < isoLevel)
                cubeIndex |= 16;
            if (corner101 < isoLevel)
                cubeIndex |= 32;
            if (corner111 < isoLevel)
                cubeIndex |= 64;
            if (corner011 < isoLevel)
                cubeIndex |= 128;

            int edgeFlags = CubeEdgeFlags[cubeIndex];
            int triCount = 0;
            for (int i = 0; i < 16; i++)
                if (TriTable[cubeIndex, i] >= 0)
                    triCount++;
            triCount /= 3;

            float surfaceTension = MathF.Abs(sdf) / gridScale;
            float edgeLength = gridScale * (1f + adaptivity * surfaceTension);

            Vector3 estimatedNormal = Vector3.Normalize(new Vector3(
                corner100 + corner110 + corner101 + corner111 - corner000 - corner010 - corner001 - corner011,
                corner010 + corner110 + corner011 + corner111 - corner000 - corner100 - corner001 - corner101,
                corner001 + corner101 + corner011 + corner111 - corner000 - corner100 - corner010 - corner110));
            estimatedNormal = Vector3.Lerp(estimatedNormal, input.Normal, 1f - normalStrength);

            float contour = MathHelper.SmoothStep(0f, contourWeight * gridScale, MathF.Abs(sdf));
            float density = 1f - contour;

            _lastOutput = new NeuronOutput
            {
                Value = sdf,
                Displacement = estimatedNormal * sdf * 0.1f,
                Color = new Vector3(MathHelper.Clamp01(surfaceTension), MathHelper.Clamp01(density), MathHelper.Clamp01(triCount / 12f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = density,
                OutputNormal = estimatedNormal,
                Height = density,
                AmbientOcclusion = density,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("sdf_value", sdf)
                    .Add("triangle_count", triCount)
                    .Add("edge_length", edgeLength)
                    .Add("surface_tension", surfaceTension)
                    .Add("density", density),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("estimated_normal", estimatedNormal)
                    .Add("grid_position", gridPos)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + sizeof(int) * 256 * 16 + 128;
        public INeuronKernel Clone() { var c = new MarchingCubeKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => _weights[1] <= 0 ? "Grid scale must be positive" : null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 0f; _weights[1] = 0.1f; _weights[2] = 1f; _weights[3] = 0.5f; _weights[4] = 1f; _weights[5] = 1f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion


    #region Ray Marcher Kernel

    public sealed class RayMarcherKernel : INeuronKernel
    {
        private const int ParamCount = 12;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum RayMarchMode { SphereTracing, AmbientOcclusion, SoftShadows, Reflections, Refractions, FullPathTracing }

        public string Name => "RayMarcher";
        public NeuronKind Kind => NeuronKind.RayMarcher;
        public int Version => 1;
        public RayMarchMode Mode { get; set; } = RayMarchMode.SphereTracing;

        public RayMarcherKernel() => ResetParameters();

        private float SDFScene(Vector3 p)
        {
            float sphere = SdfOperations.Sphere(p, 1f);
            float box = SdfOperations.Box(p - new Vector3(1.5f, 0, 0), new Vector3(0.5f, 0.5f, 0.5f));
            float plane = SdfOperations.Plane(p, Vector3.UnitY, 0f);
            float scene = SdfOperations.SmoothUnion(sphere, box, 0.3f);
            scene = SdfOperations.Union(scene, plane);
            float noiseDisp = NoiseGenerator.Perlin3D(p.X * 3f, p.Y * 3f + _weights[9], p.Z * 3f) * _weights[10];
            scene += noiseDisp;
            return scene;
        }

        private Vector3 EstimateNormal(Vector3 p)
        {
            float eps = 0.001f;
            return Vector3.Normalize(new Vector3(
                SDFScene(p + new Vector3(eps, 0, 0)) - SDFScene(p - new Vector3(eps, 0, 0)),
                SDFScene(p + new Vector3(0, eps, 0)) - SDFScene(p - new Vector3(0, eps, 0)),
                SDFScene(p + new Vector3(0, 0, eps)) - SDFScene(p - new Vector3(0, 0, eps))));
        }

        private float ComputeAO(Vector3 p, Vector3 n)
        {
            float ao = 0f;
            float weight = 1f;
            for (int i = 1; i <= 5; i++)
            {
                float dist = SDFScene(p + n * 0.1f * i);
                ao += MathHelper.Max(0f, 0.1f - dist) * weight;
                weight *= 0.5f;
            }
            return MathHelper.Clamp01(1f - ao * 4f);
        }

        private float ComputeSoftShadow(Vector3 ro, Vector3 rd, float tMin, float tMax, float k)
        {
            float res = 1f;
            float t = tMin;
            for (int i = 0; i < 32; i++)
            {
                float h = SDFScene(ro + rd * t);
                if (h < 0.001f)
                    return 0f;
                res = MathHelper.Min(res, k * h / t);
                t += MathHelper.Clamp(h, 0.01f, 0.5f);
                if (t > tMax)
                    break;
            }
            return MathHelper.Clamp01(res);
        }

        public NeuronOutput Compute(in NeuronInput input)
        {
            float maxDist = _weights[0], maxSteps = _weights[1];
            float aoIntensity = _weights[2], shadowSoftness = _weights[3];
            float reflectivity = _weights[4], refractivity = _weights[5];
            float ior = _weights[6], emissionStrength = _weights[7];
            float fogDensity = _weights[8], animSpeed = _weights[9];
            float displacementAmount = _weights[10], normalEps = _weights[11];

            Vector3 ro = input.Position - input.ViewDirection * 2f;
            Vector3 rd = Vector3.Normalize(input.ViewDirection);

            float totalDist = 0f;
            float lastDist = float.MaxValue;
            Vector3 hitPoint = Vector3.Zero;
            bool hit = false;
            int steps = 0;

            for (int i = 0; i < (int)maxSteps; i++)
            {
                Vector3 p = ro + rd * totalDist;
                float d = SDFScene(p);
                if (d < 0.001f * totalDist)
                { hit = true; hitPoint = p; steps = i; break; }
                totalDist += d;
                if (totalDist > maxDist)
                    break;
            }

            float colorValue = 0f;
            Vector3 hitColor = Vector3.Zero;
            float ao = 0f;
            float shadow = 1f;

            if (hit)
            {
                Vector3 n = EstimateNormal(hitPoint);
                float ndl = MathHelper.Max(0, Vector3.Dot(n, input.LightDirection));
                ao = ComputeAO(hitPoint, n) * aoIntensity;
                shadow = ComputeSoftShadow(hitPoint + n * 0.01f, input.LightDirection, 0.02f, 10f, shadowSoftness);

                Vector3 baseColor = new Vector3(0.8f, 0.2f, 0.1f);
                if (hitPoint.Y < 0.05f)
                    baseColor = new Vector3(0.3f, 0.5f, 0.2f);

                Vector3 diffuse = baseColor * ndl * shadow;
                Vector3 ambient = baseColor * 0.1f * ao;

                Vector3 viewDir = Vector3.Normalize(ro - hitPoint);
                Vector3 halfDir = Vector3.Normalize(input.LightDirection + viewDir);
                float ndh = MathHelper.Max(0, Vector3.Dot(n, halfDir));
                float spec = MathHelper.Pow(ndh, 32f) * (1f - ao) * shadow;

                hitColor = diffuse + ambient + new Vector3(spec);
                colorValue = MathHelper.Luminance(hitColor);

                if (reflectivity > 0)
                {
                    Vector3 reflDir = Vector3.Reflect(-viewDir, n);
                    Vector3 reflColor = new Vector3(0.1f, 0.3f, 0.8f) * reflectivity;
                    hitColor += reflColor * (1f - ao);
                }

                if (refractivity > 0)
                {
                    Vector3 refrDir = MathHelper.Refract(-viewDir, n, 1f / ior);
                    Vector3 refrColor = new Vector3(0.8f, 0.9f, 1f) * refractivity;
                    hitColor += refrColor * (1f - ao);
                }

                float fog = MathHelper.Exp(-totalDist * fogDensity);
                hitColor = Vector3.Lerp(new Vector3(0.5f, 0.6f, 0.8f), hitColor, fog);
            }

            _lastOutput = new NeuronOutput
            {
                Value = colorValue,
                Displacement = hit ? rd * totalDist : Vector3.Zero,
                Color = hitColor,
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = hit ? 1f : 0f,
                Emission = emissionStrength * (hit ? colorValue : 0f),
                EmissionColor = hitColor,
                OutputNormal = hit ? EstimateNormal(hitPoint) : input.Normal,
                Height = hit ? totalDist / maxDist : 0f,
                AmbientOcclusion = hit ? ao : 1f,
                Thickness = hit ? totalDist : 0f,
                IOR = ior,
                Specular = 0.5f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("ray_distance", totalDist)
                    .Add("hit", hit ? 1f : 0f)
                    .Add("ao", ao)
                    .Add("shadow", shadow)
                    .Add("steps", steps),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("hit_point", hitPoint)
                    .Add("hit_normal", hit ? EstimateNormal(hitPoint) : Vector3.Zero)
                    .Add("ray_direction", rd)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 1024;
        public INeuronKernel Clone() { var c = new RayMarcherKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 20f; _weights[1] = 64f; _weights[2] = 1f; _weights[3] = 16f; _weights[4] = 0.3f; _weights[5] = 0f; _weights[6] = 1.333f; _weights[7] = 0f; _weights[8] = 0.05f; _weights[9] = 0f; _weights[10] = 0.1f; _weights[11] = 0.001f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Wave Function Kernel

    public sealed class WaveFunctionKernel : INeuronKernel
    {
        private const int ParamCount = 8;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum WaveMode { SchrodingerPropagation, PlaneWave, GaussianPacket, InterferencePattern, QuantumHarmonicOscillator, ProbabilityDensity }

        public string Name => "WaveFunction";
        public NeuronKind Kind => NeuronKind.WaveFunction;
        public int Version => 1;
        public WaveMode Mode { get; set; } = WaveMode.GaussianPacket;

        public WaveFunctionKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float frequency = _weights[0], amplitude = _weights[1];
            float decay = _weights[2], phase = _weights[3];
            float spread = _weights[4], interferenceScale = _weights[5];
            float timeScale = _weights[6], normalize = _weights[7];

            Vector3 p = input.Position;
            float t = input.Time * timeScale;
            float psi = 0f;
            Vector3 psiGrad = Vector3.Zero;
            float probability = 0f;

            switch (Mode)
            {
                case WaveMode.SchrodingerPropagation:
                    float envelope = MathHelper.Exp(-p.LengthSquared() / (2f * spread * spread));
                    float oscillation = MathF.Cos(frequency * p.Z - phase * t);
                    psi = amplitude * envelope * oscillation * MathHelper.Exp(-decay * t * t);
                    psiGrad = new Vector3(
                        -p.X / (spread * spread) * psi,
                        -p.Y / (spread * spread) * psi,
                        -frequency * MathF.Sin(frequency * p.Z - phase * t) * envelope * amplitude * MathHelper.Exp(-decay * t * t));
                    break;
                case WaveMode.PlaneWave:
                    psi = amplitude * MathF.Cos(Vector3.Dot(p, new Vector3(frequency, frequency * 0.7f, frequency * 0.5f)) - phase * t);
                    psiGrad = -amplitude * new Vector3(frequency, frequency * 0.7f, frequency * 0.5f) * MathF.Sin(Vector3.Dot(p, new Vector3(frequency, frequency * 0.7f, frequency * 0.5f)) - phase * t);
                    break;
                case WaveMode.GaussianPacket:
                    float sigma2 = spread * spread;
                    psi = amplitude * MathF.Exp(-p.LengthSquared() / (2f * sigma2)) * MathF.Cos(frequency * p.X - phase * t);
                    psiGrad = new Vector3(
                        -p.X / sigma2 * psi - amplitude * frequency * MathF.Exp(-p.LengthSquared() / (2f * sigma2)) * MathF.Sin(frequency * p.X - phase * t),
                        -p.Y / sigma2 * psi,
                        -p.Z / sigma2 * psi);
                    break;
                case WaveMode.InterferencePattern:
                    float w1 = MathF.Cos(frequency * p.X + phase * t);
                    float w2 = MathF.Cos(frequency * p.Y + phase * t * 0.7f);
                    float w3 = MathF.Cos(frequency * p.Z + phase * t * 1.3f);
                    psi = amplitude * (w1 * w2 + w2 * w3 + w1 * w3) / 3f;
                    psiGrad = amplitude * new Vector3(
                        -frequency * MathF.Sin(frequency * p.X + phase * t) * (w2 + w3) / 3f,
                        -frequency * MathF.Sin(frequency * p.Y + phase * t * 0.7f) * (w1 + w3) / 3f,
                        -frequency * MathF.Sin(frequency * p.Z + phase * t * 1.3f) * (w1 + w2) / 3f);
                    break;
                case WaveMode.QuantumHarmonicOscillator:
                    float omega = frequency;
                    float alpha = MathF.Sqrt(omega);
                    float xi = alpha * p.X;
                    float hermite = 1f;
                    if (normalize > 1.5f)
                        hermite = 2f * xi;
                    else if (normalize > 0.5f)
                        hermite = 4f * xi * xi - 2f;
                    psi = amplitude * hermite * MathF.Exp(-xi * xi / 2f) * MathF.Cos(omega * (normalize + 0.5f) * t - phase);
                    psiGrad = new Vector3(
                        amplitude * (alpha * hermite * (-xi) + alpha * (normalize > 1.5f ? 2f * alpha : 8f * alpha * xi)) * MathF.Exp(-xi * xi / 2f) * MathF.Cos(omega * (normalize + 0.5f) * t - phase),
                        0, 0);
                    break;
                case WaveMode.ProbabilityDensity:
                    float envelope2 = MathHelper.Exp(-p.LengthSquared() / (spread * spread));
                    psi = amplitude * envelope2 * (MathF.Cos(frequency * p.X - phase * t) + MathF.Cos(frequency * p.Y - phase * t * 1.1f));
                    psiGrad = new Vector3(
                        -2f * p.X / (spread * spread) * psi - amplitude * envelope2 * frequency * MathF.Sin(frequency * p.X - phase * t),
                        -2f * p.Y / (spread * spread) * psi - amplitude * envelope2 * frequency * MathF.Sin(frequency * p.Y - phase * t * 1.1f),
                        -2f * p.Z / (spread * spread) * psi);
                    break;
            }

            probability = psi * psi;
            float phaseAngle = MathF.Atan2(psiGrad.Y, psiGrad.X);

            _lastOutput = new NeuronOutput
            {
                Value = psi,
                Displacement = psiGrad * 0.1f,
                Color = new Vector3(MathHelper.Clamp01(probability), MathHelper.Clamp01(MathF.Abs(psi)), MathHelper.Clamp01(phaseAngle / MathHelper.TwoPi + 0.5f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = MathHelper.Clamp01(probability),
                Emission = probability * amplitude,
                EmissionColor = new Vector3(0.3f, 0.5f, 1f),
                OutputNormal = psiGrad.LengthSquared() > 0.0001f ? Vector3.Normalize(psiGrad) : Vector3.UnitY,
                Height = probability,
                AmbientOcclusion = MathHelper.Clamp01(1f - probability),
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("wave_function", psi)
                    .Add("probability_density", probability)
                    .Add("phase", phaseAngle),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("gradient", psiGrad)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new WaveFunctionKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 5f; _weights[1] = 1f; _weights[2] = 0.1f; _weights[3] = 2f; _weights[4] = 1f; _weights[5] = 1f; _weights[6] = 1f; _weights[7] = 0f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Tensor Reshaper Kernel

    public sealed class TensorReshaperKernel : INeuronKernel
    {
        private const int ParamCount = 8;
        private readonly float[] _weights = new float[ParamCount];
        private float[] _matrixData;
        private NeuronOutput _lastOutput;

        public enum TensorOp { MatrixMultiply, Transpose, Reshape, Broadcast, Reduce, Concat, Split, Elementwise }

        public string Name => "TensorReshaper";
        public NeuronKind Kind => NeuronKind.TensorReshaper;
        public int Version => 1;
        public TensorOp Operation { get; set; } = TensorOp.MatrixMultiply;

        public TensorReshaperKernel() { _matrixData = new float[16]; ResetParameters(); }

        private float MatMul(int row, int col, float[] a, float[] b, int m, int n, int k)
        {
            float sum = 0f;
            for (int l = 0; l < k; l++)
                sum += a[row * k + l] * b[l * n + col];
            return sum;
        }

        public NeuronOutput Compute(in NeuronInput input)
        {
            float srcRows = _weights[0], srcCols = _weights[1];
            float dstRows = _weights[2], dstCols = _weights[3];
            float scale = _weights[4], bias = _weights[5];
            float axis = _weights[6], keepDims = _weights[7];

            Vector3 result = Vector3.Zero;

            switch (Operation)
            {
                case TensorOp.MatrixMultiply:
                    float[,,] A = new float[2, 2, 1];
                    A[0, 0, 0] = input.Position.X;
                    A[0, 1, 0] = input.Position.Y;
                    A[1, 0, 0] = input.Position.Z;
                    A[1, 1, 0] = 1f;
                    float r00 = input.Position.X * _matrixData[0] + input.Position.Y * _matrixData[4] + input.Position.Z * _matrixData[8] + _matrixData[12];
                    float r01 = input.Position.X * _matrixData[1] + input.Position.Y * _matrixData[5] + input.Position.Z * _matrixData[9] + _matrixData[13];
                    float r10 = input.Position.X * _matrixData[2] + input.Position.Y * _matrixData[6] + input.Position.Z * _matrixData[10] + _matrixData[14];
                    result = new Vector3(r00 * scale + bias, r01 * scale + bias, r10 * scale + bias);
                    break;
                case TensorOp.Transpose:
                    result = new Vector3(input.Position.X, input.Position.Z, input.Position.Y);
                    break;
                case TensorOp.Reshape:
                    float total = input.Position.X + input.Position.Y + input.Position.Z;
                    result = new Vector3(total / 3f, total / 3f, total / 3f);
                    break;
                case TensorOp.Broadcast:
                    float mean = (input.Position.X + input.Position.Y + input.Position.Z) / 3f;
                    result = new Vector3(mean, mean, mean);
                    break;
                case TensorOp.Reduce:
                    float sum = 0f;
                    if (axis < 0.5f)
                        sum = input.Position.X + input.Position.Y + input.Position.Z;
                    else if (axis < 1.5f)
                        sum = input.Position.X * input.Position.Y * input.Position.Z;
                    else
                        sum = MathF.Max(MathF.Max(input.Position.X, input.Position.Y), input.Position.Z);
                    result = new Vector3(sum, sum, sum);
                    break;
                case TensorOp.Concat:
                    result = new Vector3(input.Position.X * scale, input.Position.Y * scale, input.Position.Z * scale + bias);
                    break;
                case TensorOp.Split:
                    result = new Vector3(input.Position.X / (scale + 0.001f), input.Position.Y / (scale + 0.001f), input.Position.Z / (scale + 0.001f));
                    break;
                case TensorOp.Elementwise:
                    result = new Vector3(
                        MathHelper.Pow(MathF.Abs(input.Position.X) + 0.001f, scale) + bias,
                        MathHelper.Pow(MathF.Abs(input.Position.Y) + 0.001f, scale) + bias,
                        MathHelper.Pow(MathF.Abs(input.Position.Z) + 0.001f, scale) + bias);
                    break;
            }

            _lastOutput = new NeuronOutput
            {
                Value = MathHelper.Luminance(result),
                Displacement = result - input.Position,
                Color = result,
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = result.LengthSquared() > 0.0001f ? Vector3.Normalize(result) : Vector3.UnitY,
                Height = result.Length(),
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("result_x", result.X).Add("result_y", result.Y).Add("result_z", result.Z),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("result", result)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount + 16;
        public long GetMemoryFootprint() => sizeof(float) * (ParamCount + 16) + 64;
        public INeuronKernel Clone() { var c = new TensorReshaperKernel { Operation = this.Operation }; Array.Copy(_weights, c._weights, ParamCount); Array.Copy(_matrixData, c._matrixData, 16); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount + 16) { p.Slice(0, ParamCount).CopyTo(_weights); p.Slice(ParamCount, 16).CopyTo(_matrixData); } }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount + 16) { _weights.AsSpan(0, ParamCount).CopyTo(p); _matrixData.AsSpan(0, 16).CopyTo(p.Slice(ParamCount)); } }
        public void ResetParameters()
        {
            _weights[0] = 4f;
            _weights[1] = 4f;
            _weights[2] = 4f;
            _weights[3] = 4f;
            _weights[4] = 1f;
            _weights[5] = 0f;
            _weights[6] = 0f;
            _weights[7] = 1f;
            _matrixData[0] = 1f;
            _matrixData[5] = 1f;
            _matrixData[10] = 1f;
            _matrixData[15] = 1f;
            for (int i = 1; i < 16; i++)
                if (i != 0 && i != 5 && i != 10 && i != 15)
                    _matrixData[i] = 0f;
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Attention Head Kernel

    public sealed class AttentionHeadKernel : INeuronKernel
    {
        private const int ParamCount = 10;
        private readonly float[] _weights = new float[ParamCount];
        private readonly float[] _queryWeights = new float[16];
        private readonly float[] _keyWeights = new float[16];
        private readonly float[] _valueWeights = new float[16];
        private NeuronOutput _lastOutput;

        public string Name => "AttentionHead";
        public NeuronKind Kind => NeuronKind.AttentionHead;
        public int Version => 1;

        public AttentionHeadKernel() => ResetParameters();

        private float ApplyLinear(float[] weights, float x, float y, float z, int row)
        {
            return weights[row * 3] * x + weights[row * 3 + 1] * y + weights[row * 3 + 2] * z;
        }

        public NeuronOutput Compute(in NeuronInput input)
        {
            float scale = _weights[0], dropout = _weights[1];
            int numHeads = MathHelper.Clamp((int)_weights[2], 1, 8);
            int headDim = MathHelper.Clamp((int)_weights[3], 4, 64);
            float temperature = _weights[4];
            bool causal = _weights[5] > 0.5f;
            float positionalEncoding = _weights[6];
            float maxSeqLen = _weights[7];
            float keyNormalization = _weights[8];
            float outputScale = _weights[9];

            Vector3 queryVec = new Vector3(
                ApplyLinear(_queryWeights, input.Position.X, input.Position.Y, input.Position.Z, 0),
                ApplyLinear(_queryWeights, input.Position.X, input.Position.Y, input.Position.Z, 1),
                ApplyLinear(_queryWeights, input.Position.X, input.Position.Y, input.Position.Z, 2));

            Vector3 keyVec = new Vector3(
                ApplyLinear(_keyWeights, input.Position.X, input.Position.Y, input.Position.Z, 0),
                ApplyLinear(_keyWeights, input.Position.X, input.Position.Y, input.Position.Z, 1),
                ApplyLinear(_keyWeights, input.Position.X, input.Position.Y, input.Position.Z, 2));

            Vector3 valueVec = new Vector3(
                ApplyLinear(_valueWeights, input.Position.X, input.Position.Y, input.Position.Z, 0),
                ApplyLinear(_valueWeights, input.Position.X, input.Position.Y, input.Position.Z, 1),
                ApplyLinear(_valueWeights, input.Position.X, input.Position.Y, input.Position.Z, 2));

            float attnScore = Vector3.Dot(queryVec, keyVec);
            if (keyNormalization > 0.5f)
                attnScore /= MathF.Sqrt(headDim + 0.001f);
            else
                attnScore *= scale;

            attnScore /= (temperature + 0.001f);
            float attnWeight = MathHelper.Exp(MathHelper.Min(attnScore, 20f));
            attnWeight = MathHelper.Clamp01(attnWeight);

            if (dropout > 0)
                attnWeight *= (1f - dropout);

            Vector3 attended = valueVec * attnWeight * outputScale;
            float entropy = -attnWeight * MathF.Log(attnWeight + 1e-7f) - (1f - attnWeight) * MathF.Log(1f - attnWeight + 1e-7f);

            _lastOutput = new NeuronOutput
            {
                Value = attnWeight,
                Displacement = attended - input.Position,
                Color = new Vector3(MathHelper.Clamp01(attnWeight), MathHelper.Clamp01(entropy), MathHelper.Clamp01(attnScore * 0.1f + 0.5f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = attended.LengthSquared() > 0.0001f ? Vector3.Normalize(attended) : input.Normal,
                Height = attnWeight,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("attention_weight", attnWeight)
                    .Add("attention_score", attnScore)
                    .Add("entropy", entropy)
                    .Add("num_heads", numHeads),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("query", queryVec)
                    .Add("key", keyVec)
                    .Add("value", valueVec)
                    .Add("attended_output", attended)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount + 48];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount + 48;
        public long GetMemoryFootprint() => sizeof(float) * (ParamCount + 48) + 64;
        public INeuronKernel Clone() { var c = new AttentionHeadKernel(); Array.Copy(_weights, c._weights, ParamCount); Array.Copy(_queryWeights, c._queryWeights, 16); Array.Copy(_keyWeights, c._keyWeights, 16); Array.Copy(_valueWeights, c._valueWeights, 16); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount + 48) { p.Slice(0, ParamCount).CopyTo(_weights); p.Slice(ParamCount, 16).CopyTo(_queryWeights); p.Slice(ParamCount + 16, 16).CopyTo(_keyWeights); p.Slice(ParamCount + 32, 16).CopyTo(_valueWeights); } }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount + 48) { _weights.AsSpan(0, ParamCount).CopyTo(p); _queryWeights.AsSpan().CopyTo(p.Slice(ParamCount)); _keyWeights.AsSpan().CopyTo(p.Slice(ParamCount + 16)); _valueWeights.AsSpan().CopyTo(p.Slice(ParamCount + 32)); } }
        public void ResetParameters()
        {
            _weights[0] = 1f;
            _weights[1] = 0f;
            _weights[2] = 1f;
            _weights[3] = 16f;
            _weights[4] = 1f;
            _weights[5] = 0f;
            _weights[6] = 1f;
            _weights[7] = 512f;
            _weights[8] = 1f;
            _weights[9] = 1f;
            var rng = new Random(42);
            for (int i = 0; i < 16; i++)
            { _queryWeights[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f; _keyWeights[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f; _valueWeights[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f; }
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion


    #region Convolution Kernel

    public sealed class ConvolutionKernel : INeuronKernel
    {
        private const int ParamCount = 8;
        private readonly float[] _weights = new float[ParamCount];
        private readonly float[] _kernelWeights = new float[25];
        private NeuronOutput _lastOutput;

        public enum ConvMode { Conv2D, Conv3D, Depthwise, Dilated, Transposed, Grouped }

        public string Name => "Convolution";
        public NeuronKind Kind => NeuronKind.Convolution;
        public int Version => 1;
        public ConvMode Mode { get; set; } = ConvMode.Conv2D;

        public ConvolutionKernel() { ResetParameters(); }

        private float SampleInput(Vector3 p, int dx, int dy)
        {
            return NoiseGenerator.Perlin3D(p.X + dx * 0.1f, p.Y + dy * 0.1f, p.Z);
        }

        public NeuronOutput Compute(in NeuronInput input)
        {
            float stride = _weights[0], padding = _weights[1];
            float dilation = _weights[2], groups = _weights[3];
            float bias = _weights[4], activation = _weights[5];
            float inputScale = _weights[6], outputScale = _weights[7];

            float result = 0f;
            int kernelSize = 5;

            switch (Mode)
            {
                case ConvMode.Conv2D:
                    for (int ky = 0; ky < kernelSize; ky++)
                        for (int kx = 0; kx < kernelSize; kx++)
                        {
                            float inputVal = SampleInput(input.Position, kx - 2, ky - 2) * inputScale;
                            result += inputVal * _kernelWeights[ky * kernelSize + kx];
                        }
                    break;
                case ConvMode.Conv3D:
                    for (int kz = 0; kz < 3; kz++)
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++)
                            {
                                float inputVal = NoiseGenerator.Perlin3D(
                                    input.Position.X + (kx - 1) * 0.1f,
                                    input.Position.Y + (ky - 1) * 0.1f,
                                    input.Position.Z + (kz - 1) * 0.1f) * inputScale;
                                result += inputVal * _kernelWeights[kz * 9 + ky * 3 + kx];
                            }
                    break;
                case ConvMode.Depthwise:
                    for (int ky = 0; ky < kernelSize; ky++)
                        for (int kx = 0; kx < kernelSize; kx++)
                        {
                            float inputVal = SampleInput(input.Position, kx - 2, ky - 2) * inputScale;
                            result += inputVal * _kernelWeights[(ky * kernelSize + kx) % 25] * (1f + MathF.Abs(input.Position.Z) * 0.1f);
                        }
                    break;
                case ConvMode.Dilated:
                    for (int ky = 0; ky < kernelSize; ky++)
                        for (int kx = 0; kx < kernelSize; kx++)
                        {
                            int dx = (kx - 2) * (int)dilation;
                            int dy = (ky - 2) * (int)dilation;
                            float inputVal = SampleInput(input.Position, dx, dy) * inputScale;
                            result += inputVal * _kernelWeights[ky * kernelSize + kx];
                        }
                    break;
                case ConvMode.Transposed:
                    for (int ky = 0; ky < kernelSize; ky++)
                        for (int kx = 0; kx < kernelSize; kx++)
                        {
                            float inputVal = SampleInput(input.Position, kx - 2, ky - 2) * inputScale;
                            result += inputVal * _kernelWeights[(kernelSize - 1 - ky) * kernelSize + (kernelSize - 1 - kx)];
                        }
                    break;
                case ConvMode.Grouped:
                    int groupSize = (int)groups;
                    for (int ky = 0; ky < kernelSize; ky++)
                        for (int kx = 0; kx < kernelSize; kx++)
                        {
                            int group = (ky * kernelSize + kx) % groupSize;
                            float inputVal = SampleInput(input.Position, kx - 2, ky - 2) * inputScale;
                            result += inputVal * _kernelWeights[ky * kernelSize + kx] * (group == 0 ? 1f : 0.5f);
                        }
                    break;
            }

            result = result * outputScale + bias;
            if (activation > 0.5f)
                result = MathHelper.Max(0, result);
            else if (activation > 1.5f)
                result = 1f / (1f + MathF.Exp(-result));

            _lastOutput = new NeuronOutput
            {
                Value = result,
                Displacement = new Vector3(result, result * 0.5f, result * 0.25f),
                Color = new Vector3(MathHelper.Clamp01(result * 0.5f + 0.5f), MathHelper.Clamp01(result * 0.3f), MathHelper.Clamp01(-result * 0.2f + 0.5f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = input.Normal,
                Height = result,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("conv_output", result).Add("activation_type", activation),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount + 25];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            for (int i = 0; i < 25; i++)
            {
                float oldW = _kernelWeights[i];
                _kernelWeights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _kernelWeights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _kernelWeights[i] = oldW;
                wg[ParamCount + i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount + 25;
        public long GetMemoryFootprint() => sizeof(float) * (ParamCount + 25) + 64;
        public INeuronKernel Clone() { var c = new ConvolutionKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); Array.Copy(_kernelWeights, c._kernelWeights, 25); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount + 25) { p.Slice(0, ParamCount).CopyTo(_weights); p.Slice(ParamCount, 25).CopyTo(_kernelWeights); } }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount + 25) { _weights.AsSpan(0, ParamCount).CopyTo(p); _kernelWeights.AsSpan().CopyTo(p.Slice(ParamCount)); } }
        public void ResetParameters()
        {
            _weights[0] = 1f;
            _weights[1] = 0f;
            _weights[2] = 1f;
            _weights[3] = 1f;
            _weights[4] = 0f;
            _weights[5] = 1f;
            _weights[6] = 1f;
            _weights[7] = 1f;
            var rng = new Random(42);
            float std = MathF.Sqrt(2f / 25f);
            for (int i = 0; i < 25; i++)
                _kernelWeights[i] = (float)(rng.NextDouble() * 2 - 1) * std;
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) { for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length - ParamCount, 25); i++) _kernelWeights[i] -= lr * gradient.WeightGradients[ParamCount + i]; } }
    }

    #endregion

    #region Pooling Layer Kernel

    public sealed class PoolingLayerKernel : INeuronKernel
    {
        private const int ParamCount = 6;
        private readonly float[] _weights = new float[ParamCount];
        private NeuronOutput _lastOutput;

        public enum PoolMode { Max, Average, GlobalMax, GlobalAverage, Adaptive, Stochastic }

        public string Name => "PoolingLayer";
        public NeuronKind Kind => NeuronKind.PoolingLayer;
        public int Version => 1;
        public PoolMode Mode { get; set; } = PoolMode.Max;

        public PoolingLayerKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float poolSize = _weights[0], stride = _weights[1];
            float padding = _weights[2], dilation = _weights[3];
            float ceilMode = _weights[4], keepDim = _weights[5];

            float result = 0f;
            int ps = MathHelper.Clamp((int)poolSize, 1, 8);

            float[] localValues = new float[ps * ps];
            int idx = 0;
            for (int ky = 0; ky < ps; ky++)
                for (int kx = 0; kx < ps; kx++)
                    localValues[idx++] = NoiseGenerator.Perlin2D(
                        input.Position.X * 10f + kx * (float)dilation,
                        input.Position.Y * 10f + ky * (float)dilation) * 2f - 1f;

            switch (Mode)
            {
                case PoolMode.Max:
                    result = localValues[0];
                    for (int i = 1; i < localValues.Length; i++)
                        result = MathHelper.Max(result, localValues[i]);
                    break;
                case PoolMode.Average:
                    float sum = 0f;
                    for (int i = 0; i < localValues.Length; i++)
                        sum += localValues[i];
                    result = sum / localValues.Length;
                    break;
                case PoolMode.GlobalMax:
                    result = MathF.Abs(input.Position.X) > MathF.Abs(input.Position.Y) ? MathF.Abs(input.Position.X) : MathF.Abs(input.Position.Y);
                    result = MathHelper.Max(result, MathF.Abs(input.Position.Z));
                    break;
                case PoolMode.GlobalAverage:
                    result = (MathF.Abs(input.Position.X) + MathF.Abs(input.Position.Y) + MathF.Abs(input.Position.Z)) / 3f;
                    break;
                case PoolMode.Adaptive:
                    float inputSize = MathHelper.Max(MathHelper.Max(MathF.Abs(input.Position.X), MathF.Abs(input.Position.Y)), MathF.Abs(input.Position.Z));
                    result = inputSize * poolSize;
                    break;
                case PoolMode.Stochastic:
                    float maxVal = localValues[0];
                    for (int i = 1; i < localValues.Length; i++)
                        maxVal = MathHelper.Max(maxVal, localValues[i]);
                    float sumExp = 0f;
                    for (int i = 0; i < localValues.Length; i++)
                        sumExp += MathF.Exp(localValues[i] - maxVal);
                    result = maxVal + MathF.Log(sumExp) - MathF.Log(localValues.Length);
                    break;
            }

            float outputH = MathHelper.Ceil((input.Position.Y * 10f - poolSize + 2f * padding) / stride + 1f);
            float outputW = MathHelper.Ceil((input.Position.X * 10f - poolSize + 2f * padding) / stride + 1f);

            _lastOutput = new NeuronOutput
            {
                Value = result,
                Displacement = new Vector3(result, outputW * 0.01f, outputH * 0.01f),
                Color = new Vector3(MathHelper.Clamp01(result * 0.5f + 0.5f), MathHelper.Clamp01(result * 0.3f), MathHelper.Clamp01(0.5f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = input.Normal,
                Height = result,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("pool_output", result)
                    .Add("output_width", outputW)
                    .Add("output_height", outputH),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount;
        public long GetMemoryFootprint() => sizeof(float) * ParamCount + 64;
        public INeuronKernel Clone() { var c = new PoolingLayerKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount) p.Slice(0, ParamCount).CopyTo(_weights); }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount) _weights.AsSpan(0, ParamCount).CopyTo(p); }
        public void ResetParameters() { _weights[0] = 2f; _weights[1] = 2f; _weights[2] = 0f; _weights[3] = 1f; _weights[4] = 0f; _weights[5] = 1f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Normalization Layer Kernel

    public sealed class NormalizationLayerKernel : INeuronKernel
    {
        private const int ParamCount = 8;
        private readonly float[] _weights = new float[ParamCount];
        private readonly float[] _gamma = new float[8];
        private readonly float[] _beta = new float[8];
        private readonly float[] _runningMean = new float[8];
        private readonly float[] _runningVar = new float[8];
        private NeuronOutput _lastOutput;

        public enum NormMode { BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm, WeightNorm }

        public string Name => "NormalizationLayer";
        public NeuronKind Kind => NeuronKind.NormalizationLayer;
        public int Version => 1;
        public NormMode Mode { get; set; } = NormMode.LayerNorm;

        public NormalizationLayerKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float epsilon = _weights[0], momentum = _weights[1];
            float numGroups = _weights[2], elementCount = _weights[3];
            float affine = _weights[4], trackRunningStats = _weights[5];
            float channelDim = _weights[6], normalizedShape = _weights[7];

            Vector3 values = new Vector3(input.Position.X, input.Position.Y, input.Position.Z);
            Vector3 normalized = Vector3.Zero;

            switch (Mode)
            {
                case NormMode.BatchNorm:
                    float mean = (values.X + values.Y + values.Z) / 3f;
                    float variance = ((values.X - mean) * (values.X - mean) + (values.Y - mean) * (values.Y - mean) + (values.Z - mean) * (values.Z - mean)) / 3f;
                    float std = MathF.Sqrt(variance + epsilon);
                    normalized = new Vector3((values.X - mean) / std, (values.Y - mean) / std, (values.Z - mean) / std);
                    if (trackRunningStats > 0.5f)
                    {
                        _runningMean[0] = MathHelper.Lerp(_runningMean[0], mean, momentum);
                        _runningVar[0] = MathHelper.Lerp(_runningVar[0], variance, momentum);
                    }
                    break;
                case NormMode.LayerNorm:
                    float lmean = (values.X + values.Y + values.Z) / 3f;
                    float lvar = ((values.X - lmean) * (values.X - lmean) + (values.Y - lmean) * (values.Y - lmean) + (values.Z - lmean) * (values.Z - lmean)) / 3f;
                    float lstd = MathF.Sqrt(lvar + epsilon);
                    normalized = new Vector3((values.X - lmean) / lstd, (values.Y - lmean) / lstd, (values.Z - lmean) / lstd);
                    break;
                case NormMode.GroupNorm:
                    float gmean = (values.X + values.Y + values.Z) / 3f;
                    float gvar = ((values.X - gmean) * (values.X - gmean) + (values.Y - gmean) * (values.Y - gmean) + (values.Z - gmean) * (values.Z - gmean)) / 3f;
                    float gstd = MathF.Sqrt(gvar + epsilon);
                    int gCount = Math.Max(1, (int)numGroups);
                    normalized = new Vector3((values.X - gmean) / gstd, (values.Y - gmean) / gstd, (values.Z - gmean) / gstd);
                    break;
                case NormMode.InstanceNorm:
                    float imean = (values.X + values.Y + values.Z) / 3f;
                    float ivar = ((values.X - imean) * (values.X - imean) + (values.Y - imean) * (values.Y - imean) + (values.Z - imean) * (values.Z - imean)) / 3f;
                    float istd = MathF.Sqrt(ivar + epsilon);
                    normalized = new Vector3((values.X - imean) / istd, (values.Y - imean) / istd, (values.Z - imean) / istd);
                    break;
                case NormMode.RMSNorm:
                    float rms = MathF.Sqrt((values.X * values.X + values.Y * values.Y + values.Z * values.Z) / 3f + epsilon);
                    normalized = values / rms;
                    break;
                case NormMode.WeightNorm:
                    float wLen = MathF.Sqrt(_gamma[0] * _gamma[0] + _gamma[1] * _gamma[1] + _gamma[2] * _gamma[2] + epsilon);
                    Vector3 wNorm = new Vector3(_gamma[0] / wLen, _gamma[1] / wLen, _gamma[2] / wLen);
                    float dotProd = Vector3.Dot(values, wNorm);
                    normalized = values * wLen;
                    break;
            }

            if (affine > 0.5f)
            {
                normalized = new Vector3(
                    normalized.X * _gamma[0] + _beta[0],
                    normalized.Y * _gamma[1] + _beta[1],
                    normalized.Z * _gamma[2] + _beta[2]);
            }

            _lastOutput = new NeuronOutput
            {
                Value = MathHelper.Luminance(normalized),
                Displacement = normalized - input.Position,
                Color = normalized,
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = normalized.LengthSquared() > 0.0001f ? Vector3.Normalize(normalized) : input.Normal,
                Height = normalized.Length(),
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("norm_x", normalized.X).Add("norm_y", normalized.Y).Add("norm_z", normalized.Z),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("normalized_output", normalized)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount + 24];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            for (int i = 0; i < 8; i++)
            {
                float oldG = _gamma[i];
                _gamma[i] = oldG + eps;
                float vP = _lastOutput.Value;
                _gamma[i] = oldG - eps;
                float vM = _lastOutput.Value;
                _gamma[i] = oldG;
                wg[ParamCount + i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            for (int i = 0; i < 8; i++)
            {
                float oldB = _beta[i];
                _beta[i] = oldB + eps;
                float vP = _lastOutput.Value;
                _beta[i] = oldB - eps;
                float vM = _lastOutput.Value;
                _beta[i] = oldB;
                wg[ParamCount + 8 + i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount + 16;
        public long GetMemoryFootprint() => sizeof(float) * (ParamCount + 32) + 64;
        public INeuronKernel Clone() { var c = new NormalizationLayerKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); Array.Copy(_gamma, c._gamma, 8); Array.Copy(_beta, c._beta, 8); return c; }
        public string? Validate() => _weights[0] <= 0 ? "Epsilon must be positive" : null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount + 16) { p.Slice(0, ParamCount).CopyTo(_weights); p.Slice(ParamCount, 8).CopyTo(_gamma); p.Slice(ParamCount + 8, 8).CopyTo(_beta); } }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount + 16) { _weights.AsSpan(0, ParamCount).CopyTo(p); _gamma.AsSpan().CopyTo(p.Slice(ParamCount)); _beta.AsSpan().CopyTo(p.Slice(ParamCount + 8)); } }
        public void ResetParameters() { _weights[0] = 1e-5f; _weights[1] = 0.1f; _weights[2] = 1f; _weights[3] = 3f; _weights[4] = 1f; _weights[5] = 1f; _weights[6] = 0f; _weights[7] = 3f; for (int i = 0; i < 8; i++) { _gamma[i] = 1f; _beta[i] = 0f; _runningMean[i] = 0f; _runningVar[i] = 1f; } }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) { for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length - ParamCount, 8); i++) _gamma[i] -= lr * gradient.WeightGradients[ParamCount + i]; } }
    }

    #endregion

    #region Embedding Lookup Kernel

    public sealed class EmbeddingLookupKernel : INeuronKernel
    {
        private const int ParamCount = 4;
        private readonly float[] _weights = new float[ParamCount];
        private readonly float[] _embeddings = new float[128];
        private NeuronOutput _lastOutput;

        public string Name => "EmbeddingLookup";
        public NeuronKind Kind => NeuronKind.EmbeddingLookup;
        public int Version => 1;

        public EmbeddingLookupKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float vocabSize = _weights[0], embedDim = _weights[1];
            float scale = _weights[2], paddingIdx = _weights[3];

            int idx = MathHelper.Clamp(MathHelper.FloorToInt(input.Position.X * 10f + 64f), 0, 127);
            float e0 = _embeddings[idx];
            float e1 = _embeddings[(idx + 1) % 128];
            float e2 = _embeddings[(idx + 2) % 128];

            Vector3 embedding = new Vector3(e0, e1, e2) * scale;

            float norm = embedding.Length();
            if (norm > 0.001f)
                embedding = embedding / MathF.Sqrt(embedDim + 0.001f);

            _lastOutput = new NeuronOutput
            {
                Value = norm,
                Displacement = embedding,
                Color = MathHelper.Clamp01Vec(embedding * 0.5f + new Vector3(0.5f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = embedding.LengthSquared() > 0.0001f ? Vector3.Normalize(embedding) : input.Normal,
                Height = norm,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("embedding_norm", norm).Add("index", idx),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("embedding", embedding)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount + 128];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            int idx = MathHelper.Clamp(MathHelper.FloorToInt(_lastOutput.Value * 10f + 64f), 0, 127);
            for (int i = 0; i < 3; i++)
            {
                int ei = (idx + i) % 128;
                float oldE = _embeddings[ei];
                _embeddings[ei] = oldE + eps;
                float vP = _lastOutput.Value;
                _embeddings[ei] = oldE - eps;
                float vM = _lastOutput.Value;
                _embeddings[ei] = oldE;
                wg[ParamCount + ei] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount + 128;
        public long GetMemoryFootprint() => sizeof(float) * (ParamCount + 128) + 64;
        public INeuronKernel Clone() { var c = new EmbeddingLookupKernel(); Array.Copy(_weights, c._weights, ParamCount); Array.Copy(_embeddings, c._embeddings, 128); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount + 128) { p.Slice(0, ParamCount).CopyTo(_weights); p.Slice(ParamCount, 128).CopyTo(_embeddings); } }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount + 128) { _weights.AsSpan(0, ParamCount).CopyTo(p); _embeddings.AsSpan().CopyTo(p.Slice(ParamCount)); } }
        public void ResetParameters()
        {
            _weights[0] = 1000f;
            _weights[1] = 64f;
            _weights[2] = 1f;
            _weights[3] = -1f;
            var rng = new Random(42);
            for (int i = 0; i < 128; i++)
                _embeddings[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) { for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; int idx = MathHelper.Clamp(MathHelper.FloorToInt(_lastOutput.Value * 10f + 64f), 0, 127); for (int i = 0; i < 3; i++) { int ei = (idx + i) % 128; if (ParamCount + ei < gradient.WeightGradients.Length) _embeddings[ei] -= lr * gradient.WeightGradients[ParamCount + ei]; } } }
    }

    #endregion

    #region Positional Encoder Kernel

    public sealed class PositionalEncoderKernel : INeuronKernel
    {
        private const int ParamCount = 6;
        private readonly float[] _weights = new float[ParamCount];
        private readonly float[] _learnedEmbeddings = new float[64];
        private NeuronOutput _lastOutput;

        public enum PositionalMode { Sinusoidal, Learned, Rotary, Relative, ALiBi, Factorized }

        public string Name => "PositionalEncoder";
        public NeuronKind Kind => NeuronKind.PositionalEncoder;
        public int Version => 1;
        public PositionalMode Mode { get; set; } = PositionalMode.Sinusoidal;

        public PositionalEncoderKernel() => ResetParameters();

        private float SinusoidalEncoding(int idx, float position, int dim)
        {
            float freq = 1f / MathF.Pow(10000f, (float)(2 * (idx / 2)) / dim);
            return (idx % 2 == 0) ? MathF.Sin(position * freq) : MathF.Cos(position * freq);
        }

        private float RotaryEncoding(int idx, float position, int dim)
        {
            float freq = 1f / MathF.Pow(10000f, (float)idx / dim);
            float angle = position * freq;
            return (idx % 2 == 0) ? MathF.Cos(angle) : MathF.Sin(angle);
        }

        public NeuronOutput Compute(in NeuronInput input)
        {
            float maxPosition = _weights[0], numFrequencies = _weights[1];
            float scaleFactor = _weights[2], temperature = _weights[3];
            float learnedWeight = _weights[4], dropout = _weights[5];

            int pos = MathHelper.Clamp(MathHelper.RoundToInt(input.Position.X * 10f), 0, 63);
            Vector3 encoded = Vector3.Zero;

            switch (Mode)
            {
                case PositionalMode.Sinusoidal:
                    encoded = new Vector3(
                        SinusoidalEncoding(0, pos, (int)numFrequencies),
                        SinusoidalEncoding(1, pos, (int)numFrequencies),
                        SinusoidalEncoding(2, pos, (int)numFrequencies));
                    encoded *= scaleFactor;
                    break;
                case PositionalMode.Learned:
                    encoded = new Vector3(
                        _learnedEmbeddings[pos % 64],
                        _learnedEmbeddings[(pos + 1) % 64],
                        _learnedEmbeddings[(pos + 2) % 64]);
                    encoded = Vector3.Lerp(encoded, encoded * learnedWeight, learnedWeight);
                    break;
                case PositionalMode.Rotary:
                    encoded = new Vector3(
                        RotaryEncoding(0, pos, (int)numFrequencies),
                        RotaryEncoding(1, pos, (int)numFrequencies),
                        RotaryEncoding(2, pos, (int)numFrequencies));
                    float cosRot = MathF.Cos(pos * 0.1f), sinRot = MathF.Sin(pos * 0.1f);
                    encoded = new Vector3(encoded.X * cosRot - encoded.Y * sinRot, encoded.X * sinRot + encoded.Y * cosRot, encoded.Z);
                    break;
                case PositionalMode.Relative:
                    float relPos = MathHelper.Clamp01(pos / maxPosition);
                    encoded = new Vector3(relPos, relPos * relPos, MathF.Sqrt(relPos + 0.001f)) * scaleFactor;
                    break;
                case PositionalMode.ALiBi:
                    float alibiScale = MathF.Pow(2f, -8f * pos / maxPosition);
                    encoded = new Vector3(pos * alibiScale, (pos + 1) * alibiScale, (pos + 2) * alibiScale);
                    break;
                case PositionalMode.Factorized:
                    float p1 = pos / 8f, p2 = pos % 8;
                    encoded = new Vector3(SinusoidalEncoding(0, p1, 8), SinusoidalEncoding(1, p2, 8), SinusoidalEncoding(2, p1 + p2, 8));
                    encoded *= scaleFactor;
                    break;
            }

            if (dropout > 0)
                encoded *= (1f - dropout);

            float positionNorm = encoded.Length();

            _lastOutput = new NeuronOutput
            {
                Value = positionNorm,
                Displacement = encoded,
                Color = MathHelper.Clamp01Vec(encoded * 0.5f + new Vector3(0.5f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = encoded.LengthSquared() > 0.0001f ? Vector3.Normalize(encoded) : input.Normal,
                Height = positionNorm,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("position_norm", positionNorm).Add("position_index", pos),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("encoded_position", encoded)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount + 64];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            if (Mode == PositionalMode.Learned)
            {
                int pos = MathHelper.Clamp(MathHelper.RoundToInt(_lastOutput.Value * 10f), 0, 63);
                for (int i = 0; i < 3; i++)
                {
                    int ei = (pos + i) % 64;
                    float oldE = _learnedEmbeddings[ei];
                    _learnedEmbeddings[ei] = oldE + eps;
                    float vP = _lastOutput.Value;
                    _learnedEmbeddings[ei] = oldE - eps;
                    float vM = _lastOutput.Value;
                    _learnedEmbeddings[ei] = oldE;
                    wg[ParamCount + ei] = gradient.ValueGradient * (vP - vM) / (2f * eps);
                }
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount + 64;
        public long GetMemoryFootprint() => sizeof(float) * (ParamCount + 64) + 64;
        public INeuronKernel Clone() { var c = new PositionalEncoderKernel { Mode = this.Mode }; Array.Copy(_weights, c._weights, ParamCount); Array.Copy(_learnedEmbeddings, c._learnedEmbeddings, 64); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount + 64) { p.Slice(0, ParamCount).CopyTo(_weights); p.Slice(ParamCount, 64).CopyTo(_learnedEmbeddings); } }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount + 64) { _weights.AsSpan(0, ParamCount).CopyTo(p); _learnedEmbeddings.AsSpan().CopyTo(p.Slice(ParamCount)); } }
        public void ResetParameters() { _weights[0] = 64f; _weights[1] = 16f; _weights[2] = 1f; _weights[3] = 1f; _weights[4] = 0.5f; _weights[5] = 0f; var rng = new Random(42); for (int i = 0; i < 64; i++) _learnedEmbeddings[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f; }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) { for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; } }
    }

    #endregion

    #region Cross Attention Layer Kernel

    public sealed class CrossAttentionLayerKernel : INeuronKernel
    {
        private const int ParamCount = 8;
        private readonly float[] _weights = new float[ParamCount];
        private readonly float[] _qWeights = new float[9];
        private readonly float[] _kWeights = new float[9];
        private readonly float[] _vWeights = new float[9];
        private NeuronOutput _lastOutput;

        public string Name => "CrossAttentionLayer";
        public NeuronKind Kind => NeuronKind.CrossAttentionLayer;
        public int Version => 1;

        public CrossAttentionLayerKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float embedDim = _weights[0], numHeads = _weights[1];
            float dropout = _weights[2], temperature = _weights[3];
            float scale = MathHelper.Sqrt(embedDim + 0.001f);

            Vector3 q = new Vector3(
                Vector3.Dot(input.Position, new Vector3(_qWeights[0], _qWeights[1], _qWeights[2])),
                Vector3.Dot(input.Position, new Vector3(_qWeights[3], _qWeights[4], _qWeights[5])),
                Vector3.Dot(input.Position, new Vector3(_qWeights[6], _qWeights[7], _qWeights[8])));

            Vector3 k = new Vector3(
                Vector3.Dot(input.LightDirection, new Vector3(_kWeights[0], _kWeights[1], _kWeights[2])),
                Vector3.Dot(input.LightDirection, new Vector3(_kWeights[3], _kWeights[4], _kWeights[5])),
                Vector3.Dot(input.LightDirection, new Vector3(_kWeights[6], _kWeights[7], _kWeights[8])));

            Vector3 v = new Vector3(
                Vector3.Dot(input.LightDirection, new Vector3(_vWeights[0], _vWeights[1], _vWeights[2])),
                Vector3.Dot(input.LightDirection, new Vector3(_vWeights[3], _vWeights[4], _vWeights[5])),
                Vector3.Dot(input.LightDirection, new Vector3(_vWeights[6], _vWeights[7], _vWeights[8])));

            float attnScore = Vector3.Dot(q, k) / (temperature * scale);
            float attnWeight = MathHelper.Exp(MathHelper.Min(attnScore, 20f));
            float sumExp = attnWeight + MathHelper.Exp(0f);
            attnWeight /= sumExp;
            if (dropout > 0)
                attnWeight *= (1f - dropout);

            Vector3 attended = v * attnWeight;
            float entropy = -attnWeight * MathF.Log(attnWeight + 1e-7f) - (1f - attnWeight) * MathF.Log(1f - attnWeight + 1e-7f);

            _lastOutput = new NeuronOutput
            {
                Value = attnWeight,
                Displacement = attended - input.Position,
                Color = new Vector3(MathHelper.Clamp01(attnWeight), MathHelper.Clamp01(entropy), MathHelper.Clamp01(attnScore * 0.1f + 0.5f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = attended.LengthSquared() > 0.0001f ? Vector3.Normalize(attended) : input.Normal,
                Height = attnWeight,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("attention_weight", attnWeight).Add("entropy", entropy),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("query", q).Add("key", k).Add("value", v).Add("attended", attended)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount + 27];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            for (int i = 0; i < 27; i++)
            {
                float[] allW = i < 9 ? _qWeights : i < 18 ? _kWeights : _vWeights;
                int wi = i % 9;
                float oldW = allW[wi];
                allW[wi] = oldW + eps;
                float vP = _lastOutput.Value;
                allW[wi] = oldW - eps;
                float vM = _lastOutput.Value;
                allW[wi] = oldW;
                wg[ParamCount + i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount + 27;
        public long GetMemoryFootprint() => sizeof(float) * (ParamCount + 27) + 64;
        public INeuronKernel Clone() { var c = new CrossAttentionLayerKernel(); Array.Copy(_weights, c._weights, ParamCount); Array.Copy(_qWeights, c._qWeights, 9); Array.Copy(_kWeights, c._kWeights, 9); Array.Copy(_vWeights, c._vWeights, 9); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount + 27) { p.Slice(0, ParamCount).CopyTo(_weights); p.Slice(ParamCount, 9).CopyTo(_qWeights); p.Slice(ParamCount + 9, 9).CopyTo(_kWeights); p.Slice(ParamCount + 18, 9).CopyTo(_vWeights); } }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount + 27) { _weights.AsSpan(0, ParamCount).CopyTo(p); _qWeights.AsSpan().CopyTo(p.Slice(ParamCount)); _kWeights.AsSpan().CopyTo(p.Slice(ParamCount + 9)); _vWeights.AsSpan().CopyTo(p.Slice(ParamCount + 18)); } }
        public void ResetParameters()
        {
            _weights[0] = 64f;
            _weights[1] = 8f;
            _weights[2] = 0f;
            _weights[3] = 1f;
            _weights[4] = 0f;
            _weights[5] = 0f;
            _weights[6] = 0f;
            _weights[7] = 0f;
            var rng = new Random(42);
            float std = MathF.Sqrt(2f / 9f);
            for (int i = 0; i < 9; i++)
            { _qWeights[i] = (float)(rng.NextDouble() * 2 - 1) * std; _kWeights[i] = (float)(rng.NextDouble() * 2 - 1) * std; _vWeights[i] = (float)(rng.NextDouble() * 2 - 1) * std; }
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) { for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; } }
    }

    #endregion

    #region Self Attention Layer Kernel

    public sealed class SelfAttentionLayerKernel : INeuronKernel
    {
        private const int ParamCount = 6;
        private readonly float[] _weights = new float[ParamCount];
        private readonly float[] _projWeights = new float[9];
        private NeuronOutput _lastOutput;

        public string Name => "SelfAttentionLayer";
        public NeuronKind Kind => NeuronKind.SelfAttentionLayer;
        public int Version => 1;

        public SelfAttentionLayerKernel() => ResetParameters();

        public NeuronOutput Compute(in NeuronInput input)
        {
            float embedDim = _weights[0], numHeads = _weights[1];
            float dropout = _weights[2], temperature = _weights[3];
            float maxSeqLen = _weights[4], causalMask = _weights[5];

            Vector3 x = input.Position;
            float scale = MathHelper.Sqrt(embedDim + 0.001f);
            Vector3 q = x, k = x, v = x;

            float attnScore = Vector3.Dot(q, k) / (temperature * scale);
            if (causalMask > 0.5f)
            {
                float pos = input.Position.X * 10f;
                if (pos > input.Position.Y * 10f)
                    attnScore = float.MinValue;
            }
            float attnWeight = MathHelper.Exp(MathHelper.Min(attnScore, 20f));
            attnWeight /= (attnWeight + 1f);
            if (dropout > 0)
                attnWeight *= (1f - dropout);

            Vector3 attended = v * attnWeight;
            Vector3 projected = new Vector3(
                Vector3.Dot(attended, new Vector3(_projWeights[0], _projWeights[1], _projWeights[2])),
                Vector3.Dot(attended, new Vector3(_projWeights[3], _projWeights[4], _projWeights[5])),
                Vector3.Dot(attended, new Vector3(_projWeights[6], _projWeights[7], _projWeights[8])));
            Vector3 residual = x + projected;
            float entropy = -attnWeight * MathF.Log(attnWeight + 1e-7f) - (1f - attnWeight) * MathF.Log(1f - attnWeight + 1e-7f);

            _lastOutput = new NeuronOutput
            {
                Value = attnWeight,
                Displacement = residual - input.Position,
                Color = new Vector3(MathHelper.Clamp01(attnWeight), MathHelper.Clamp01(entropy), MathHelper.Clamp01(residual.Length() * 0.1f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = residual.LengthSquared() > 0.0001f ? Vector3.Normalize(residual) : input.Normal,
                Height = attnWeight,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("attention_weight", attnWeight).Add("entropy", entropy).Add("residual_norm", residual.Length()),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("attended", attended).Add("projected", projected).Add("residual", residual)
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount + 9];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            for (int i = 0; i < 9; i++)
            {
                float oldW = _projWeights[i];
                _projWeights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _projWeights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _projWeights[i] = oldW;
                wg[ParamCount + i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount + 9;
        public long GetMemoryFootprint() => sizeof(float) * (ParamCount + 9) + 64;
        public INeuronKernel Clone() { var c = new SelfAttentionLayerKernel(); Array.Copy(_weights, c._weights, ParamCount); Array.Copy(_projWeights, c._projWeights, 9); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount + 9) { p.Slice(0, ParamCount).CopyTo(_weights); p.Slice(ParamCount, 9).CopyTo(_projWeights); } }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount + 9) { _weights.AsSpan(0, ParamCount).CopyTo(p); _projWeights.AsSpan().CopyTo(p.Slice(ParamCount)); } }
        public void ResetParameters()
        {
            _weights[0] = 64f;
            _weights[1] = 8f;
            _weights[2] = 0f;
            _weights[3] = 1f;
            _weights[4] = 512f;
            _weights[5] = 0f;
            var rng = new Random(42);
            float std = MathF.Sqrt(2f / 9f);
            for (int i = 0; i < 9; i++)
                _projWeights[i] = (float)(rng.NextDouble() * 2 - 1) * std;
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; }
    }

    #endregion

    #region Feed Forward Kernel

    public sealed class FeedForwardKernel : INeuronKernel
    {
        private const int ParamCount = 6;
        private readonly float[] _weights = new float[ParamCount];
        private readonly float[] _fc1Weights = new float[16];
        private readonly float[] _fc1Bias = new float[4];
        private readonly float[] _fc2Weights = new float[16];
        private readonly float[] _fc2Bias = new float[3];
        private NeuronOutput _lastOutput;

        public enum ActivationFunction { ReLU, GELU, SiLU, Tanh, Sigmoid, Mish, Swish, LeakyReLU }

        public string Name => "FeedForward";
        public NeuronKind Kind => NeuronKind.FeedForward;
        public int Version => 1;
        public ActivationFunction Activation { get; set; } = ActivationFunction.GELU;

        public FeedForwardKernel() => ResetParameters();

        private float ApplyActivation(float x)
        {
            switch (Activation)
            {
                case ActivationFunction.ReLU:
                    return MathHelper.Max(0, x);
                case ActivationFunction.GELU:
                    return 0.5f * x * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (x + 0.044715f * x * x * x)));
                case ActivationFunction.SiLU:
                    return x / (1f + MathF.Exp(-x));
                case ActivationFunction.Tanh:
                    return MathF.Tanh(x);
                case ActivationFunction.Sigmoid:
                    return 1f / (1f + MathF.Exp(-x));
                case ActivationFunction.Mish:
                    float sp = MathF.Log(1f + MathF.Exp(x));
                    return x * MathF.Tanh(sp);
                case ActivationFunction.Swish:
                    return x / (1f + MathF.Exp(-x));
                case ActivationFunction.LeakyReLU:
                    return x > 0 ? x : 0.01f * x;
                default:
                    return MathHelper.Max(0, x);
            }
        }

        public NeuronOutput Compute(in NeuronInput input)
        {
            float hiddenDim = _weights[0], outputDim = _weights[1];
            float dropout = _weights[2], residualScale = _weights[3];
            float preNorm = _weights[4], activationType = _weights[5];

            Vector3 x = input.Position;
            float[] hidden = new float[4];
            for (int i = 0; i < 4; i++)
            {
                float sum = _fc1Bias[i];
                for (int j = 0; j < 3; j++)
                    sum += x[j] * _fc1Weights[i * 3 + j];
                hidden[i] = ApplyActivation(sum);
                if (dropout > 0 && dropout < 1f)
                    hidden[i] *= (1f - dropout);
            }

            Vector3 output = Vector3.Zero;
            for (int i = 0; i < 3; i++)
            {
                float sum = _fc2Bias[i];
                for (int j = 0; j < 4; j++)
                    sum += hidden[j] * _fc2Weights[i * 4 + j];
                output[i] = sum;
            }

            Vector3 residual = x * residualScale + output;
            float outputNorm = residual.Length();

            _lastOutput = new NeuronOutput
            {
                Value = outputNorm,
                Displacement = residual - input.Position,
                Color = new Vector3(MathHelper.Clamp01(residual.X * 0.5f + 0.5f), MathHelper.Clamp01(residual.Y * 0.5f + 0.5f), MathHelper.Clamp01(residual.Z * 0.5f + 0.5f)),
                Roughness = 0.5f,
                Metallic = 0f,
                Opacity = 1f,
                Emission = 0f,
                OutputNormal = residual.LengthSquared() > 0.0001f ? Vector3.Normalize(residual) : input.Normal,
                Height = outputNorm,
                AmbientOcclusion = 1f,
                Outputs = ImmutableDictionary<string, float>.Empty
                    .Add("output_norm", outputNorm).Add("hidden_0", hidden[0]).Add("hidden_1", hidden[1]),
                VectorOutputs = ImmutableDictionary<string, Vector3>.Empty
                    .Add("output", residual).Add("hidden", new Vector3(hidden[0], hidden[1], hidden[2]))
            };
            return _lastOutput;
        }

        public NeuronGradient Backpropagate(in NeuronGradient gradient)
        {
            float[] wg = new float[ParamCount + 38];
            float eps = 0.001f;
            for (int i = 0; i < ParamCount; i++)
            {
                float oldW = _weights[i];
                _weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _weights[i] = oldW;
                wg[i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            for (int i = 0; i < 16; i++)
            {
                float oldW = _fc1Weights[i];
                _fc1Weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _fc1Weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _fc1Weights[i] = oldW;
                wg[ParamCount + i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            for (int i = 0; i < 16; i++)
            {
                float oldW = _fc2Weights[i];
                _fc2Weights[i] = oldW + eps;
                float vP = _lastOutput.Value;
                _fc2Weights[i] = oldW - eps;
                float vM = _lastOutput.Value;
                _fc2Weights[i] = oldW;
                wg[ParamCount + 16 + i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            for (int i = 0; i < 4; i++)
            {
                float oldB = _fc1Bias[i];
                _fc1Bias[i] = oldB + eps;
                float vP = _lastOutput.Value;
                _fc1Bias[i] = oldB - eps;
                float vM = _lastOutput.Value;
                _fc1Bias[i] = oldB;
                wg[ParamCount + 32 + i] = gradient.ValueGradient * (vP - vM) / (2f * eps);
            }
            return new NeuronGradient { ValueGradient = gradient.ValueGradient, WeightGradients = wg, ParameterGradients = ImmutableDictionary<string, float>.Empty };
        }

        public int GetParameterCount() => ParamCount + 36;
        public long GetMemoryFootprint() => sizeof(float) * (ParamCount + 36) + 64;
        public INeuronKernel Clone() { var c = new FeedForwardKernel { Activation = this.Activation }; Array.Copy(_weights, c._weights, ParamCount); Array.Copy(_fc1Weights, c._fc1Weights, 16); Array.Copy(_fc1Bias, c._fc1Bias, 4); Array.Copy(_fc2Weights, c._fc2Weights, 16); Array.Copy(_fc2Bias, c._fc2Bias, 3); return c; }
        public string? Validate() => null;
        public void LoadParameters(ReadOnlySpan<float> p) { if (p.Length >= ParamCount + 36) { p.Slice(0, ParamCount).CopyTo(_weights); p.Slice(ParamCount, 16).CopyTo(_fc1Weights); p.Slice(ParamCount + 16, 4).CopyTo(_fc1Bias); p.Slice(ParamCount + 20, 16).CopyTo(_fc2Weights); p.Slice(ParamCount + 36 - 3, 3).CopyTo(_fc2Bias); } }
        public void SaveParameters(Span<float> p) { if (p.Length >= ParamCount + 36) { _weights.AsSpan(0, ParamCount).CopyTo(p); _fc1Weights.AsSpan().CopyTo(p.Slice(ParamCount)); _fc1Bias.AsSpan().CopyTo(p.Slice(ParamCount + 16)); _fc2Weights.AsSpan().CopyTo(p.Slice(ParamCount + 20)); _fc2Bias.AsSpan().CopyTo(p.Slice(ParamCount + 36 - 3)); } }
        public void ResetParameters()
        {
            _weights[0] = 64f;
            _weights[1] = 3f;
            _weights[2] = 0f;
            _weights[3] = 1f;
            _weights[4] = 0f;
            _weights[5] = (float)ActivationFunction.GELU;
            var rng = new Random(42);
            float std1 = MathF.Sqrt(2f / 3f), std2 = MathF.Sqrt(2f / 4f);
            for (int i = 0; i < 16; i++)
                _fc1Weights[i] = (float)(rng.NextDouble() * 2 - 1) * std1;
            for (int i = 0; i < 4; i++)
                _fc1Bias[i] = 0f;
            for (int i = 0; i < 16; i++)
                _fc2Weights[i] = (float)(rng.NextDouble() * 2 - 1) * std2;
            for (int i = 0; i < 3; i++)
                _fc2Bias[i] = 0f;
        }
        public void ApplyGradient(in NeuronGradient gradient, float lr) { if (gradient.WeightGradients != null) { for (int i = 0; i < MathHelper.Min(gradient.WeightGradients.Length, ParamCount); i++) _weights[i] -= lr * gradient.WeightGradients[i]; } }
    }

    #endregion


    #region Neuron Kernel Registry

    public sealed class NeuronKernelRegistry
    {
        private readonly Dictionary<NeuronKind, Func<INeuronKernel>> _factories = new Dictionary<NeuronKind, Func<INeuronKernel>>();
        private readonly Dictionary<NeuronKind, INeuronKernel> _instances = new Dictionary<NeuronKind, INeuronKernel>();
        private readonly Dictionary<string, NeuronKind> _nameToKind = new Dictionary<string, NeuronKind>();

        private static readonly Lazy<NeuronKernelRegistry> _instance = new Lazy<NeuronKernelRegistry>(() => new NeuronKernelRegistry());

        public static NeuronKernelRegistry Instance => _instance.Value;

        public int RegisteredCount => _factories.Count;

        private NeuronKernelRegistry()
        {
            RegisterDefaults();
        }

        private void RegisterDefaults()
        {
            Register(NeuronKind.SdfPrimitive, () => new SdfPrimitiveKernel());
            Register(NeuronKind.DisplacementField, () => new DisplacementFieldKernel());
            Register(NeuronKind.CurvatureModulator, () => new CurvatureModulatorKernel());
            Register(NeuronKind.ColorField, () => new ColorFieldKernel());
            Register(NeuronKind.NormalMap, () => new NormalMapKernel());
            Register(NeuronKind.RoughnessMap, () => new RoughnessMapKernel());
            Register(NeuronKind.MetallicMap, () => new MetallicMapKernel());
            Register(NeuronKind.EmissiveMap, () => new EmissiveMapKernel());
            Register(NeuronKind.OpacityMap, () => new OpacityMapKernel());
            Register(NeuronKind.SubsurfaceScattering, () => new SubsurfaceScatteringKernel());
            Register(NeuronKind.Tessellation, () => new TessellationKernel());
            Register(NeuronKind.OcclusionCulling, () => new OcclusionCullingKernel());
            Register(NeuronKind.AnimationCurve, () => new AnimationCurveKernel());
            Register(NeuronKind.ParticleEmitter, () => new ParticleEmitterKernel());
            Register(NeuronKind.FluidSolver, () => new FluidSolverKernel());
            Register(NeuronKind.ClothSimulator, () => new ClothSimulatorKernel());
            Register(NeuronKind.Voxelizer, () => new VoxelizerKernel());
            Register(NeuronKind.MarchingCube, () => new MarchingCubeKernel());
            Register(NeuronKind.RayMarcher, () => new RayMarcherKernel());
            Register(NeuronKind.WaveFunction, () => new WaveFunctionKernel());
            Register(NeuronKind.TensorReshaper, () => new TensorReshaperKernel());
            Register(NeuronKind.AttentionHead, () => new AttentionHeadKernel());
            Register(NeuronKind.Convolution, () => new ConvolutionKernel());
            Register(NeuronKind.PoolingLayer, () => new PoolingLayerKernel());
            Register(NeuronKind.NormalizationLayer, () => new NormalizationLayerKernel());
            Register(NeuronKind.EmbeddingLookup, () => new EmbeddingLookupKernel());
            Register(NeuronKind.PositionalEncoder, () => new PositionalEncoderKernel());
            Register(NeuronKind.CrossAttentionLayer, () => new CrossAttentionLayerKernel());
            Register(NeuronKind.SelfAttentionLayer, () => new SelfAttentionLayerKernel());
            Register(NeuronKind.FeedForward, () => new FeedForwardKernel());
        }

        public void Register(NeuronKind kind, Func<INeuronKernel> factory)
        {
            _factories[kind] = factory;
            var kernel = factory();
            _nameToKind[kernel.Name] = kind;
        }

        public INeuronKernel Create(NeuronKind kind)
        {
            if (_factories.TryGetValue(kind, out var factory))
                return factory();
            throw new ArgumentException($"No kernel registered for kind: {kind}");
        }

        public INeuronKernel Create(string name)
        {
            if (_nameToKind.TryGetValue(name, out var kind))
                return Create(kind);
            throw new ArgumentException($"No kernel registered with name: {name}");
        }

        public INeuronKernel GetOrCreate(NeuronKind kind)
        {
            if (!_instances.TryGetValue(kind, out var instance))
            {
                instance = Create(kind);
                _instances[kind] = instance;
            }
            return instance;
        }

        public bool IsRegistered(NeuronKind kind) => _factories.ContainsKey(kind);

        public bool IsRegistered(string name) => _nameToKind.ContainsKey(name);

        public NeuronKind GetKindByName(string name)
        {
            if (_nameToKind.TryGetValue(name, out var kind))
                return kind;
            return NeuronKind.UserDefined;
        }

        public string GetNameByKind(NeuronKind kind)
        {
            if (_factories.TryGetValue(kind, out var factory))
            {
                var kernel = factory();
                return kernel.Name;
            }
            return "Unknown";
        }

        public IEnumerable<NeuronKind> GetAllRegisteredKinds() => _factories.Keys;

        public IEnumerable<string> GetAllRegisteredNames() => _nameToKind.Keys;

        public IReadOnlyDictionary<NeuronKind, Func<INeuronKernel>> GetAllFactories() => _factories;

        public void DiscoverAndRegister()
        {
            var assembly = Assembly.GetExecutingAssembly();
            var kernelTypes = assembly.GetTypes()
                .Where(t => typeof(INeuronKernel).IsAssignableFrom(t) && !t.IsInterface && !t.IsAbstract);

            foreach (var type in kernelTypes)
            {
                try
                {
                    var kernel = (INeuronKernel)Activator.CreateInstance(type);
                    if (kernel != null && !_factories.ContainsKey(kernel.Kind))
                    {
                        Register(kernel.Kind, () => (INeuronKernel)Activator.CreateInstance(type));
                    }
                }
                catch { }
            }
        }

        public void DiscoverAndRegister(Assembly assembly)
        {
            var kernelTypes = assembly.GetTypes()
                .Where(t => typeof(INeuronKernel).IsAssignableFrom(t) && !t.IsInterface && !t.IsAbstract);

            foreach (var type in kernelTypes)
            {
                try
                {
                    var kernel = (INeuronKernel)Activator.CreateInstance(type);
                    if (kernel != null && !_factories.ContainsKey(kernel.Kind))
                    {
                        Register(kernel.Kind, () => (INeuronKernel)Activator.CreateInstance(type));
                    }
                }
                catch { }
            }
        }

        public NeuronOutput ComputeWithKernel(NeuronKind kind, in NeuronInput input)
        {
            var kernel = Create(kind);
            return kernel.Compute(input);
        }

        public float[] GetParameters(NeuronKind kind)
        {
            var kernel = Create(kind);
            int paramCount = kernel.GetParameterCount();
            var parameters = new float[paramCount];
            kernel.SaveParameters(parameters);
            return parameters;
        }

        public void SetParameters(NeuronKind kind, float[] parameters)
        {
            var kernel = Create(kind);
            kernel.LoadParameters(parameters);
        }

        public long GetMemoryFootprint(NeuronKind kind)
        {
            var kernel = Create(kind);
            return kernel.GetMemoryFootprint();
        }

        public string? ValidateAll()
        {
            var errors = new List<string>();
            foreach (var kvp in _factories)
            {
                try
                {
                    var kernel = kvp.Value();
                    var error = kernel.Validate();
                    if (error != null)
                        errors.Add($"{kvp.Key}: {error}");
                }
                catch (Exception ex)
                {
                    errors.Add($"{kvp.Key}: Exception during validation - {ex.Message}");
                }
            }
            return errors.Count > 0 ? string.Join("; ", errors) : null;
        }
    }

    #endregion
}
