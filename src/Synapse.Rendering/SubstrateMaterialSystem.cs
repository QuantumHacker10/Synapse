// =============================================================================
// SubstrateMaterialSystem.cs
// GDNN.Engine - GDNN.Materials.SubstrateOmega
// Substrate-NeurONAL Material System for the G-DNN Engine
// =============================================================================
// Comprehensive physically-based material system implementing BSDF evaluation,
// procedural texture generation, material genomes, and shader code generation.
// =============================================================================

using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace GDNN.Materials.SubstrateOmega
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    [Flags]
    public enum MaterialDomain : byte
    {
        Surface = 1 << 0,
        Volume = 1 << 1,
        Hair = 1 << 2,
        Eye = 1 << 3,
        Cloth = 1 << 4,
        ClearCoat = 1 << 5,
        Emissive = 1 << 6
    }

    [Flags]
    public enum MaterialFeatureFlags : uint
    {
        None = 0,
        SubsurfaceScattering = 1u << 0,
        ClearCoat = 1u << 1,
        Anisotropy = 1u << 2,
        Sheen = 1u << 3,
        Transmission = 1u << 4,
        DiffuseTransmission = 1u << 5,
        ThinSurface = 1u << 6,
        Unlit = 1u << 7,
        Emissive = 1u << 8,
        Decal = 1u << 9,
        SpecularAntiAliasing = 1u << 10,
        DynamicWeathering = 1u << 11,
        ParallaxOcclusion = 1u << 12,
        PixelDepthOffset = 1u << 13,
        SubsurfaceProfile = 1u << 14,
        ClothBRDF = 1u << 15,
        HairBRDF = 1u << 16,
        EyeBRDF = 1u << 17,
        ThinShadow = 1u << 18,
        TwoSided = 1u << 19,
        WorldPositionOffset = 1u << 20,
        DitheredOpacity = 1u << 21,
        MaskedBlend = 1u << 22,
        AnalyticalAntialiasing = 1u << 23,
        SubsurfaceProfileKernel = 1u << 24,
        ClothSheenEnvBRDF = 1u << 25,
        PreIntegratedSkin = 1u << 26,
        EyeIrisRefraction = 1u << 27,
        MultiBounceAO = 1u << 28,
        DiffuseProfile = 1u << 29,
        SpecularProfile = 1u << 30,
        MaterialCache = 1u << 31
    }

    public enum BSDFType : byte
    {
        Lambertian,
        OrenNayar,
        AshikhminShirley,
        Disney,
        GGX,
        Charlie,
        Fabric,
        Hair,
        Eye,
        Translucent,
        StandardPBR,
        CookTorrance,
        Ward,
        SchlickGGX
    }

    public enum TextureChannel : byte
    {
        Albedo, Normal, Roughness, Metallic, AO, Emissive, Height,
        Displacement, SubsurfaceColor, ClearCoatNormal, ClearCoatRoughness,
        ClearCoatMask, AnisotropyTangent, AnisotropyStrength, SheenColor,
        SheenRoughness, TransmissionColor, TransmissionRoughness, Opacity,
        Thickness, ScatterDistance, Curvature, BentNormal, AmbientOcclusion,
        SpecularColor, Refraction, PixelDepthOffset, Wind, Wetness,
        DetailNormal, DetailAlbedo, DecalMask,
        UV0, UV1, UV2, UV3, VertexColor
    }

    public enum MaterialPropertyType : byte
    {
        Float, Vec2, Vec3, Vec4, Int, Bool, Texture, Color, Enum, Mat3, Mat4
    }

    public enum TextureCompression : byte
    {
        None, BC1, BC2, BC3, BC4, BC5, BC6H, BC7,
        ASTC_4x4, ASTC_6x6, ASTC_8x8,
        ETC2_RGB, ETC2_RGBA, PVRTC_4bpp
    }

    public enum TextureSamplerMode : byte
    {
        Point, Bilinear, Trilinear,
        Anisotropic2x, Anisotropic4x, Anisotropic8x, Anisotropic16x
    }

    public enum TextureWrapMode : byte
    {
        Repeat, Clamp, Mirror, MirrorOnce, Border
    }

    public enum LayerBlendMode : byte
    {
        Replace, Add, Multiply, Overlay, Screen, Subtract,
        AlphaBlend, Masked, HeightBlend, NormalBlend
    }

    public enum ShaderLanguage : byte
    {
        HLSL, GLSL, Slang, Metal, WGSL, SPIRV
    }

    public enum ShaderStage : byte
    {
        Vertex, Pixel, Geometry, TessellationControl,
        TessellationEvaluation, Compute, Mesh
    }

    public enum WeatheringState : byte
    {
        Pristine, LightlyWeathered, ModeratelyWeathered, HeavilyWeathered,
        Eroded, Corroded, Rusted, Mossy, Covered, Frozen, Wet, Dusty
    }

    public enum MaterialCategory : byte
    {
        Metal, Wood, Stone, Skin, Fabric, Glass, Liquid, Ceramic,
        Plastic, Rubber, Paper, Concrete, Sand, Snow, Ice,
        Organic, Synthetic, Geological, Environmental, Custom,
        Hair, Gem, Eye, Fluid
    }

    public enum ActivationFunction : byte
    {
        Linear, Sigmoid, Tanh, ReLU, Softplus, Square, Sqrt
    }

    public enum SynapticConnectionType : byte
    {
        Excitatory, Inhibitory, Modulatory, Structural
    }

    public enum ProceduralTextureType : byte
    {
        Checkerboard, Gradient, Noise, Brick, WoodGrain, Marble,
        GradientNoise, Voronoi, Cellular, Wave, FractalNoise,
        Perlin, Simplex, Worley, Value, Turbulence
    }

    public enum NoiseType : byte
    {
        Perlin, Simplex, Value, Worley, Voronoi, Billow, Ridged, Turbulence
    }

    // =========================================================================
    // RECORDS
    // =========================================================================

    public record MaterialProperty
    {
        public string Name { get; init; }
        public MaterialPropertyType Type { get; init; }
        public object Value { get; init; }
        public float Min { get; init; }
        public float Max { get; init; }
        public float Default { get; init; }
        public string Description { get; init; }
        public string DisplayGroup { get; init; }
        public string UIHint { get; init; }
        public bool IsExposed { get; init; } = true;
        public bool IsAnimated { get; init; }
        public int SortOrder { get; init; }

        public MaterialProperty(
            string name, MaterialPropertyType type, object value,
            float min = 0f, float max = 1f, float defaultVal = 0f,
            string description = "", string displayGroup = "General",
            string uiHint = "")
        {
            Name = name; Type = type; Value = value;
            Min = min; Max = max; Default = defaultVal;
            Description = description; DisplayGroup = displayGroup;
            UIHint = uiHint;
        }

        public float AsFloat() => Value is float f ? f : Convert.ToSingle(Value);
        public int AsInt() => Value is int i ? i : Convert.ToInt32(Value);
        public bool AsBool() => Value is bool b ? b : Convert.ToBoolean(Value);
        public string AsString() => Value?.ToString() ?? "";

        public Vec3 AsVec3()
        {
            if (Value is Vec3 v) return v;
            if (Value is float[] arr && arr.Length >= 3)
                return new Vec3(arr[0], arr[1], arr[2]);
            if (Value is System.Numerics.Vector3 nv)
                return new Vec3(nv.X, nv.Y, nv.Z);
            float f = AsFloat();
            return new Vec3(f, f, f);
        }

        public Color3 AsColor()
        {
            if (Value is Color3 c) return c;
            if (Value is float[] arr && arr.Length >= 3)
                return new Color3(arr[0], arr[1], arr[2]);
            if (Value is Vec3 v) return new Color3(v.X, v.Y, v.Z);
            float f = AsFloat();
            return new Color3(f, f, f);
        }

        public MaterialProperty WithValue(object newValue) => this with { Value = newValue };
    }

    public record TextureReference
    {
        public string Path { get; init; }
        public TextureSamplerMode Sampler { get; init; } = TextureSamplerMode.Bilinear;
        public TextureWrapMode WrapMode { get; init; } = TextureWrapMode.Repeat;
        public Vec2 Tiling { get; init; } = new Vec2(1.0f, 1.0f);
        public Vec2 Offset { get; init; } = new Vec2(0.0f, 0.0f);
        public float Rotation { get; init; }
        public TextureChannel Channel { get; init; }
        public TextureCompression Compression { get; init; } = TextureCompression.BC7;
        public float MipBias { get; init; }
        public int MaxAnisotropy { get; init; } = 4;
        public float LODMin { get; init; }
        public float LODMax { get; init; } = float.MaxValue;
        public bool IsVirtual { get; init; }
        public int UVSet { get; init; }

        public TextureReference(string path, TextureChannel channel = TextureChannel.Albedo)
        {
            Path = path; Channel = channel;
        }

        public Mat3 ComputeUVTransform()
        {
            float cosR = (float)Math.Cos(Rotation);
            float sinR = (float)Math.Sin(Rotation);
            return new Mat3(
                Tiling.X * cosR, Tiling.X * sinR, 0,
                -Tiling.Y * sinR, Tiling.Y * cosR, 0,
                Offset.X, Offset.Y, 1);
        }
    }

    public record ColorProfile
    {
        public Color3 Albedo { get; init; } = new Color3(0.8f, 0.8f, 0.8f);
        public float Roughness { get; init; } = 0.5f;
        public float Metallic { get; init; }
        public Color3 Specular { get; init; } = new Color3(0.04f, 0.04f, 0.04f);
        public Color3 Subsurface { get; init; }
        public float ClearCoat { get; init; }
        public float ClearCoatRoughness { get; init; } = 0.01f;
        public Color3 Emissive { get; init; }
        public float EmissiveIntensity { get; init; } = 1.0f;
        public float Opacity { get; init; } = 1.0f;
        public float RefractionIndex { get; init; } = 1.5f;

        public static ColorProfile FromSRGB(Color3 srgbAlbedo, float roughness, float metallic)
        {
            return new ColorProfile
            {
                Albedo = Color3.SRGBToLinear(srgbAlbedo),
                Roughness = roughness,
                Metallic = metallic
            };
        }
    }

    public record MaterialPass
    {
        public string VertexShader { get; init; }
        public string PixelShader { get; init; }
        public string ComputeShader { get; init; }
        public string GeometryShader { get; init; }
        public string TessellationShader { get; init; }
        public string MeshShader { get; init; }
        public string MaterialShader { get; init; }
        public Dictionary<string, string> Permutations { get; init; } = new();
        public MaterialFeatureFlags RequiredFeatures { get; init; }

        public MaterialPass(string materialShader = "") { MaterialShader = materialShader; }
    }

    public record ShaderSource
    {
        public string Source { get; init; }
        public ShaderLanguage Language { get; init; }
        public ShaderStage Stage { get; init; }
        public MaterialFeatureFlags Features { get; init; }
        public string Hash { get; init; }
        public List<string> Includes { get; init; } = new();
        public Dictionary<string, string> Defines { get; init; } = new();
        public int EstimatedInstructionCount { get; init; }
        public int TextureSampleCount { get; init; }
    }

    public record MaterialAnalyticsData
    {
        public int PropertyCount { get; init; }
        public int TextureCount { get; init; }
        public int TextureMemoryBytes { get; init; }
        public int ShaderPermutationCount { get; init; }
        public int EstimatedGPUCost { get; init; }
        public int EstimatedDrawCalls { get; init; }
        public MaterialFeatureFlags ActiveFeatures { get; init; }
        public float ComplexityScore { get; init; }
    }

    // =========================================================================
    // MATH TYPES
    // =========================================================================

    [StructLayout(LayoutKind.Sequential)]
    public struct Vec2 : IEquatable<Vec2>
    {
        public float X;
        public float Y;

        public Vec2(float x, float y) { X = x; Y = y; }

        public static readonly Vec2 Zero = new(0, 0);
        public static readonly Vec2 One = new(1, 1);
        public static readonly Vec2 UnitX = new(1, 0);
        public static readonly Vec2 UnitY = new(0, 1);

        public float Length => (float)Math.Sqrt(X * X + Y * Y);
        public float LengthSquared => X * X + Y * Y;
        public Vec2 Normalized => Length > 1e-8f ? this / Length : Zero;

        public float Dot(Vec2 other) => X * other.X + Y * other.Y;
        public float Cross(Vec2 other) => X * other.Y - Y * other.X;
        public Vec2 Lerp(Vec2 other, float t) => new(X + (other.X - X) * t, Y + (other.Y - Y) * t);
        public Vec2 Clamp(float min, float max) => new(Math.Clamp(X, min, max), Math.Clamp(Y, min, max));
        public float MinComponent => Math.Min(X, Y);
        public float MaxComponent => Math.Max(X, Y);

        public static Vec2 operator +(Vec2 a, Vec2 b) => new(a.X + b.X, a.Y + b.Y);
        public static Vec2 operator -(Vec2 a, Vec2 b) => new(a.X - b.X, a.Y - b.Y);
        public static Vec2 operator *(Vec2 a, float s) => new(a.X * s, a.Y * s);
        public static Vec2 operator *(float s, Vec2 a) => new(a.X * s, a.Y * s);
        public static Vec2 operator *(Vec2 a, Vec2 b) => new(a.X * b.X, a.Y * b.Y);
        public static Vec2 operator /(Vec2 a, float s) => new(a.X / s, a.Y / s);
        public static Vec2 operator -(Vec2 a) => new(-a.X, -a.Y);

        public bool Equals(Vec2 other) => Math.Abs(X - other.X) < 1e-6f && Math.Abs(Y - other.Y) < 1e-6f;
        public override bool Equals(object obj) => obj is Vec2 v && Equals(v);
        public override int GetHashCode() => HashCode.Combine(X, Y);
        public override string ToString() => $"({X:F4}, {Y:F4})";

        public static implicit operator System.Numerics.Vector2(Vec2 v) => new(v.X, v.Y);
        public static implicit operator Vec2(System.Numerics.Vector2 v) => new(v.X, v.Y);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Vec3 : IEquatable<Vec3>
    {
        public float X;
        public float Y;
        public float Z;

        public Vec3(float x, float y, float z) { X = x; Y = y; Z = z; }
        public Vec3(float v) { X = v; Y = v; Z = v; }

        public static readonly Vec3 Zero = new(0, 0, 0);
        public static readonly Vec3 One = new(1, 1, 1);
        public static readonly Vec3 UnitX = new(1, 0, 0);
        public static readonly Vec3 UnitY = new(0, 1, 0);
        public static readonly Vec3 UnitZ = new(0, 0, 1);

        public float Length => (float)Math.Sqrt(X * X + Y * Y + Z * Z);
        public float LengthSquared => X * X + Y * Y + Z * Z;
        public Vec3 Normalized { get { float len = Length; return len > 1e-8f ? this / len : Zero; } }

        public float Dot(Vec3 other) => X * other.X + Y * other.Y + Z * other.Z;
        public Vec3 Cross(Vec3 other) => new(Y * other.Z - Z * other.Y, Z * other.X - X * other.Z, X * other.Y - Y * other.X);
        public Vec3 Lerp(Vec3 other, float t) => new(X + (other.X - X) * t, Y + (other.Y - Y) * t, Z + (other.Z - Z) * t);
        public static Vec3 Lerp(Vec3 a, Vec3 b, float t) => a.Lerp(b, t);
        public Vec3 Min(Vec3 other) => new(Math.Min(X, other.X), Math.Min(Y, other.Y), Math.Min(Z, other.Z));
        public Vec3 Max(Vec3 other) => new(Math.Max(X, other.X), Math.Max(Y, other.Y), Math.Max(Z, other.Z));
        public Vec3 Abs() => new(Math.Abs(X), Math.Abs(Y), Math.Abs(Z));
        public Vec3 Saturate() => new(Math.Clamp(X, 0, 1), Math.Clamp(Y, 0, 1), Math.Clamp(Z, 0, 1));

        public float this[int index]
        {
            get => index switch { 0 => X, 1 => Y, 2 => Z, _ => throw new IndexOutOfRangeException() };
            set { switch (index) { case 0: X = value; break; case 1: Y = value; break; case 2: Z = value; break; } }
        }

        public static Vec3 operator +(Vec3 a, Vec3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        public static Vec3 operator -(Vec3 a, Vec3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        public static Vec3 operator *(Vec3 a, float s) => new(a.X * s, a.Y * s, a.Z * s);
        public static Vec3 operator *(float s, Vec3 a) => new(a.X * s, a.Y * s, a.Z * s);
        public static Vec3 operator *(Vec3 a, Vec3 b) => new(a.X * b.X, a.Y * b.Y, a.Z * b.Z);
        public static Vec3 operator /(Vec3 a, float s) => new(a.X / s, a.Y / s, a.Z / s);
        public static Vec3 operator /(Vec3 a, Vec3 b) => new(a.X / b.X, a.Y / b.Y, a.Z / b.Z);
        public static Vec3 operator -(Vec3 a) => new(-a.X, -a.Y, -a.Z);

        public bool Equals(Vec3 other) => Math.Abs(X - other.X) < 1e-6f && Math.Abs(Y - other.Y) < 1e-6f && Math.Abs(Z - other.Z) < 1e-6f;
        public override bool Equals(object obj) => obj is Vec3 v && Equals(v);
        public override int GetHashCode() => HashCode.Combine(X, Y, Z);
        public override string ToString() => $"({X:F4}, {Y:F4}, {Z:F4})";

        public static implicit operator System.Numerics.Vector3(Vec3 v) => new(v.X, v.Y, v.Z);
        public static implicit operator Vec3(System.Numerics.Vector3 v) => new(v.X, v.Y, v.Z);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Vec4 : IEquatable<Vec4>
    {
        public float X; public float Y; public float Z; public float W;

        public Vec4(float x, float y, float z, float w) { X = x; Y = y; Z = z; W = w; }
        public Vec4(Vec3 xyz, float w) { X = xyz.X; Y = xyz.Y; Z = xyz.Z; W = w; }
        public Vec4(float v) { X = v; Y = v; Z = v; W = v; }

        public Vec3 XYZ => new(X, Y, Z);
        public Vec2 XY => new(X, Y);
        public float Length => (float)Math.Sqrt(X * X + Y * Y + Z * Z + W * W);
        public float LengthSquared => X * X + Y * Y + Z * Z + W * W;

        public Vec4 Normalized { get { float len = Length; return len > 1e-8f ? this / len : new Vec4(0); } }

        public float Dot(Vec4 other) => X * other.X + Y * other.Y + Z * other.Z + W * other.W;
        public Vec4 Lerp(Vec4 other, float t) => new(X + (other.X - X) * t, Y + (other.Y - Y) * t, Z + (other.Z - Z) * t, W + (other.W - W) * t);

        public static Vec4 operator +(Vec4 a, Vec4 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W);
        public static Vec4 operator -(Vec4 a, Vec4 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z, a.W - b.W);
        public static Vec4 operator *(Vec4 a, float s) => new(a.X * s, a.Y * s, a.Z * s, a.W * s);
        public static Vec4 operator *(float s, Vec4 a) => new(a.X * s, a.Y * s, a.Z * s, a.W * s);
        public static Vec4 operator /(Vec4 a, float s) => new(a.X / s, a.Y / s, a.Z / s, a.W / s);

        public bool Equals(Vec4 other) => Math.Abs(X - other.X) < 1e-6f && Math.Abs(Y - other.Y) < 1e-6f && Math.Abs(Z - other.Z) < 1e-6f && Math.Abs(W - other.W) < 1e-6f;
        public override bool Equals(object obj) => obj is Vec4 v && Equals(v);
        public override int GetHashCode() => HashCode.Combine(X, Y, Z, W);
        public override string ToString() => $"({X:F4}, {Y:F4}, {Z:F4}, {W:F4})";

        public static implicit operator System.Numerics.Vector4(Vec4 v) => new(v.X, v.Y, v.Z, v.W);
        public static implicit operator Vec4(System.Numerics.Vector4 v) => new(v.X, v.Y, v.Z, v.W);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Mat3
    {
        public float M00, M01, M02;
        public float M10, M11, M12;
        public float M20, M21, M22;

        public Mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
        {
            M00 = m00; M01 = m01; M02 = m02; M10 = m10; M11 = m11; M12 = m12; M20 = m20; M21 = m21; M22 = m22;
        }

        public static readonly Mat3 Identity = new(1, 0, 0, 0, 1, 0, 0, 0, 1);
        public static readonly Mat3 Zero = new(0, 0, 0, 0, 0, 0, 0, 0, 0);

        public Vec3 Transform(Vec3 v) => new(M00 * v.X + M01 * v.Y + M02 * v.Z, M10 * v.X + M11 * v.Y + M12 * v.Z, M20 * v.X + M21 * v.Y + M22 * v.Z);

        public Mat3 Transpose() => new(M00, M10, M20, M01, M11, M21, M02, M12, M22);

        public float Determinant() => M00 * (M11 * M22 - M12 * M21) - M01 * (M10 * M22 - M12 * M20) + M02 * (M10 * M21 - M11 * M20);

        public Mat3 Inverse()
        {
            float det = Determinant();
            if (Math.Abs(det) < 1e-10f) return Identity;
            float invDet = 1.0f / det;
            return new Mat3(
                (M11 * M22 - M12 * M21) * invDet, (M02 * M21 - M01 * M22) * invDet, (M01 * M12 - M02 * M11) * invDet,
                (M12 * M20 - M10 * M22) * invDet, (M00 * M22 - M02 * M20) * invDet, (M02 * M10 - M00 * M12) * invDet,
                (M10 * M21 - M11 * M20) * invDet, (M01 * M20 - M00 * M21) * invDet, (M00 * M11 - M01 * M10) * invDet);
        }

        public static Mat3 operator *(Mat3 a, Mat3 b) => new(
            a.M00 * b.M00 + a.M01 * b.M10 + a.M02 * b.M20, a.M00 * b.M01 + a.M01 * b.M11 + a.M02 * b.M21, a.M00 * b.M02 + a.M01 * b.M12 + a.M02 * b.M22,
            a.M10 * b.M00 + a.M11 * b.M10 + a.M12 * b.M20, a.M10 * b.M01 + a.M11 * b.M11 + a.M12 * b.M21, a.M10 * b.M02 + a.M11 * b.M12 + a.M12 * b.M22,
            a.M20 * b.M00 + a.M21 * b.M10 + a.M22 * b.M20, a.M20 * b.M01 + a.M21 * b.M11 + a.M22 * b.M21, a.M20 * b.M02 + a.M21 * b.M12 + a.M22 * b.M22);

        public override string ToString() => $"[{M00:F4} {M01:F4} {M02:F4}]\n[{M10:F4} {M11:F4} {M12:F4}]\n[{M20:F4} {M21:F4} {M22:F4}]";
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Color3 : IEquatable<Color3>
    {
        public float R; public float G; public float B;

        public Color3(float r, float g, float b) { R = r; G = g; B = b; }
        public Color3(float v) { R = v; G = v; B = v; }

        public static readonly Color3 Black = new(0, 0, 0);
        public static readonly Color3 White = new(1, 1, 1);
        public static readonly Color3 Red = new(1, 0, 0);
        public static readonly Color3 Green = new(0, 1, 0);
        public static readonly Color3 Blue = new(0, 0, 1);
        public static readonly Color3 Yellow = new(1, 1, 0);
        public static readonly Color3 Cyan = new(0, 1, 1);
        public static readonly Color3 Magenta = new(1, 0, 1);
        public static readonly Color3 Gray = new(0.5f, 0.5f, 0.5f);

        public float Luminance => 0.2126f * R + 0.7152f * G + 0.0722f * B;
        public float MaxComponent => Math.Max(R, Math.Max(G, B));
        public float MinComponent => Math.Min(R, Math.Min(G, B));

        public Color3 Saturate() => new(Math.Clamp(R, 0, 1), Math.Clamp(G, 0, 1), Math.Clamp(B, 0, 1));

        public static Color3 Lerp(Color3 a, Color3 b, float t) =>
            new(float.Lerp(a.R, b.R, t), float.Lerp(a.G, b.G, t), float.Lerp(a.B, b.B, t));

        public Color3 ReinhardToneMap()
        {
            float luma = Luminance;
            float scale = luma > 0 ? (luma * (1 + luma)) / (1 + luma) : 0;
            float factor = scale / (luma + 1e-6f);
            return new Color3(R * factor, G * factor, B * factor);
        }

        public Color3 ACESFilmicToneMap()
        {
            float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
            return new Color3(
                Math.Clamp((R * (a * R + b)) / (R * (c * R + d) + e), 0, 1),
                Math.Clamp((G * (a * G + b)) / (G * (c * G + d) + e), 0, 1),
                Math.Clamp((B * (a * B + b)) / (B * (c * B + d) + e), 0, 1));
        }

        public Color3 Pow(float exp) => new((float)Math.Pow(Math.Max(R, 0), exp), (float)Math.Pow(Math.Max(G, 0), exp), (float)Math.Pow(Math.Max(B, 0), exp));
        public Color3 Sqrt() => new((float)Math.Sqrt(Math.Max(R, 0)), (float)Math.Sqrt(Math.Max(G, 0)), (float)Math.Sqrt(Math.Max(B, 0)));
        public Color3 Exp() => new((float)Math.Exp(R), (float)Math.Exp(G), (float)Math.Exp(B));
        public Color3 Abs() => new(Math.Abs(R), Math.Abs(G), Math.Abs(B));
        public float MaxRGB() => Math.Max(R, Math.Max(G, B));
        public float MinRGB() => Math.Min(R, Math.Min(G, B));

        public static Color3 SRGBToLinear(Color3 srgb) =>
            new(SRGBChannelToLinear(srgb.R), SRGBChannelToLinear(srgb.G), SRGBChannelToLinear(srgb.B));

        public static Color3 LinearToSRGB(Color3 linear) =>
            new(LinearChannelToSRGB(linear.R), LinearChannelToSRGB(linear.G), LinearChannelToSRGB(linear.B));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SRGBChannelToLinear(float c) => c <= 0.04045f ? c / 12.92f : (float)Math.Pow((c + 0.055f) / 1.055f, 2.4f);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float LinearChannelToSRGB(float c) => c <= 0.0031308f ? c * 12.92f : 1.055f * (float)Math.Pow(c, 1.0f / 2.4f) - 0.055f;

        public static Color3 operator +(Color3 a, Color3 b) => new(a.R + b.R, a.G + b.G, a.B + b.B);
        public static Color3 operator -(Color3 a, Color3 b) => new(a.R - b.R, a.G - b.G, a.B - b.B);
        public static Color3 operator *(Color3 a, Color3 b) => new(a.R * b.R, a.G * b.G, a.B * b.B);
        public static Color3 operator *(Color3 a, float s) => new(a.R * s, a.G * s, a.B * s);
        public static Color3 operator *(float s, Color3 a) => new(a.R * s, a.G * s, a.B * s);
        public static Color3 operator /(Color3 a, float s) => new(a.R / s, a.G / s, a.B / s);
        public static Color3 operator -(Color3 a) => new(-a.R, -a.G, -a.B);

        public bool Equals(Color3 other) => Math.Abs(R - other.R) < 1e-6f && Math.Abs(G - other.G) < 1e-6f && Math.Abs(B - other.B) < 1e-6f;
        public override bool Equals(object obj) => obj is Color3 c && Equals(c);
        public override int GetHashCode() => HashCode.Combine(R, G, B);
        public override string ToString() => $"({R:F4}, {G:F4}, {B:F4})";

        public Vec3 ToVec3() => new(R, G, B);
        public static Color3 FromVec3(Vec3 v) => new(v.X, v.Y, v.Z);

        public static implicit operator Vec3(Color3 c) => new(c.R, c.G, c.B);
        public static implicit operator Color3(Vec3 v) => new(v.X, v.Y, v.Z);
    }

    // =========================================================================
    // STATIC MATH HELPER
    // =========================================================================

    public static class MathHelper
    {
        public const float PI = 3.14159265358979323846f;
        public const float INV_PI = 0.31830988618379067154f;
        public const float INV_2PI = 0.15915494309189533577f;
        public const float EPSILON = 1e-6f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Lerp(float a, float b, float t) => a + (b - a) * t;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Clamp(float v, float min, float max) => Math.Clamp(v, min, max);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Saturate(float v) => Math.Clamp(v, 0, 1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Remap(float value, float fromMin, float fromMax, float toMin, float toMax) =>
            toMin + (value - fromMin) * (toMax - toMin) / (fromMax - fromMin);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Smoothstep(float edge0, float edge1, float x)
        {
            float t = Math.Clamp((x - edge0) / (edge1 - edge0), 0, 1);
            return t * t * (3.0f - 2.0f * t);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Smootherstep(float edge0, float edge1, float x)
        {
            float t = Math.Clamp((x - edge0) / (edge1 - edge0), 0, 1);
            return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Noise2D(float x, float y) =>
            (float)(Math.Sin(x * 12.9898 + y * 78.233) * 43758.5453 % 1.0);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HashFloat(float x) =>
            (float)(Math.Sin(x * 12.9898) * 43758.5453 % 1.0);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ExpNeg(float x) => x < 0 ? 1.0f / (1.0f - x) : (float)Math.Exp(-x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DegreesToRadians(float deg) => deg * PI / 180.0f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float RadiansToDegrees(float rad) => rad * 180.0f / PI;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LerpAngle(float a, float b, float t)
        {
            float diff = b - a;
            while (diff > PI) diff -= 2 * PI;
            while (diff < -PI) diff += 2 * PI;
            return a + diff * t;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float WrapAngle(float angle)
        {
            angle = angle % (2 * PI);
            if (angle < 0) angle += 2 * PI;
            return angle;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vec3 OrthogonalBasis(Vec3 normal, out Vec3 tangent, out Vec3 bitangent)
        {
            Vec3 up = Math.Abs(normal.Z) < 0.999f ? Vec3.UnitZ : Vec3.UnitX;
            tangent = normal.Cross(up).Normalized;
            bitangent = normal.Cross(tangent).Normalized;
            return tangent;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Fract(float x) => x - (float)Math.Floor(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Floor(float x) => (float)Math.Floor(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Ceil(float x) => (float)Math.Ceiling(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sign(float x) => x >= 0 ? 1.0f : -1.0f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Max(float a, float b) => Math.Max(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Min(float a, float b) => Math.Min(a, b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Abs(float x) => Math.Abs(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Pow(float b, float e) => (float)Math.Pow(b, e);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sqrt(float x) => (float)Math.Sqrt(x);
    }

    // =========================================================================
    // MATERIAL GENOME
    // =========================================================================

    public class MaterialGenome
    {
        private readonly Dictionary<string, MicroGenome> _microGenomes;
        private readonly List<SynapticConnection> _synapticConnections;
        private readonly Dictionary<string, ActivationKernel> _activationKernels;
        private readonly HashSet<string> _semanticTags;
        private int _version;
        private float _mutationRate;
        private float _crossoverRate;
        private MaterialFeatureFlags _activeFeatures;
        private int _currentLOD;

        public float MutationRate { get => _mutationRate; set => _mutationRate = Math.Clamp(value, 0f, 1f); }
        public float CrossoverRate { get => _crossoverRate; set => _crossoverRate = Math.Clamp(value, 0f, 1f); }
        public MaterialFeatureFlags ActiveFeatures => _activeFeatures;
        public int CurrentLOD => _currentLOD;
        public int Version => _version;
        public int MicroGenomeCount => _microGenomes.Count;
        public int ConnectionCount => _synapticConnections.Count;

        public MaterialGenome()
        {
            _microGenomes = new Dictionary<string, MicroGenome>();
            _synapticConnections = new List<SynapticConnection>();
            _activationKernels = new Dictionary<string, ActivationKernel>();
            _semanticTags = new HashSet<string>();
            _version = 1;
            _mutationRate = 0.1f;
            _crossoverRate = 0.7f;
            _activeFeatures = MaterialFeatureFlags.None;
            _currentLOD = 0;
        }

        public void AddMicroGenome(string name, MicroGenome microGenome) { _microGenomes[name] = microGenome; _version++; }
        public MicroGenome GetMicroGenome(string name) { _microGenomes.TryGetValue(name, out var m); return m; }
        public bool RemoveMicroGenome(string name) { bool r = _microGenomes.Remove(name); if (r) { _synapticConnections.RemoveAll(c => c.SourceGenome == name || c.TargetGenome == name); _version++; } return r; }
        public IReadOnlyDictionary<string, MicroGenome> GetAllMicroGenomes() => _microGenomes;
        public IReadOnlyList<SynapticConnection> GetAllConnections() => _synapticConnections.AsReadOnly();
        public IReadOnlyCollection<string> GetSemanticTags() => _semanticTags.ToList().AsReadOnly();

        public void AddSynapticConnection(SynapticConnection connection)
        {
            if (_microGenomes.ContainsKey(connection.SourceGenome) && _microGenomes.ContainsKey(connection.TargetGenome))
            { _synapticConnections.Add(connection); _version++; }
        }

        public Dictionary<string, MaterialProperty> Evaluate()
        {
            var results = new Dictionary<string, MaterialProperty>();
            var evaluated = new HashSet<string>();
            var pending = new Queue<string>(_microGenomes.Keys);
            int safety = 0;

            while (pending.Count > 0 && safety++ < _microGenomes.Count * 3)
            {
                string name = pending.Dequeue();
                if (evaluated.Contains(name)) continue;
                var micro = _microGenomes[name];

                bool allInputsReady = true;
                foreach (var conn in _synapticConnections.Where(c => c.TargetGenome == name))
                    if (!evaluated.Contains(conn.SourceGenome)) { allInputsReady = false; break; }

                if (!allInputsReady) { pending.Enqueue(name); continue; }

                var inputs = new Dictionary<string, float>();
                foreach (var conn in _synapticConnections.Where(c => c.TargetGenome == name))
                    if (results.TryGetValue(conn.SourceGenome, out var sp))
                        inputs[conn.SourceProperty] = sp.AsFloat() * conn.Weight;

                foreach (var prop in micro.Evaluate(inputs)) results[prop.Key] = prop.Value;
                evaluated.Add(name);
            }
            return results;
        }

        public void SetFeatures(MaterialFeatureFlags features) { _activeFeatures = features; _version++; }
        public void AddSemanticTag(string tag) { _semanticTags.Add(tag); _version++; }
        public bool HasSemanticTag(string tag) => _semanticTags.Contains(tag);

        public void SetLOD(int lod)
        {
            _currentLOD = Math.Max(0, lod);
            foreach (var micro in _microGenomes.Values) micro.SetLODComplexity(_currentLOD);
            _version++;
        }

        public void Mutate(Random rng = null)
        {
            rng ??= Random.Shared;
            foreach (var micro in _microGenomes.Values)
                if (rng.NextDouble() < _mutationRate) micro.Mutate(rng, _mutationRate);
            _version++;
        }

        public (MaterialGenome offspring1, MaterialGenome offspring2) Crossover(MaterialGenome other, Random rng = null)
        {
            rng ??= Random.Shared;
            var c1 = new MaterialGenome();
            var c2 = new MaterialGenome();
            var allKeys = _microGenomes.Keys.Union(other._microGenomes.Keys).Distinct();

            foreach (string key in allKeys)
            {
                bool first = rng.NextDouble() < 0.5f;
                if (_microGenomes.ContainsKey(key) && other._microGenomes.ContainsKey(key))
                {
                    c1._microGenomes[key] = first ? _microGenomes[key] : other._microGenomes[key];
                    c2._microGenomes[key] = first ? other._microGenomes[key] : _microGenomes[key];
                }
                else if (_microGenomes.ContainsKey(key)) c1._microGenomes[key] = _microGenomes[key];
                else if (other._microGenomes.ContainsKey(key)) c2._microGenomes[key] = other._microGenomes[key];
            }
            c1._activeFeatures = c2._activeFeatures = _activeFeatures | other._activeFeatures;
            c1._semanticTags.UnionWith(_semanticTags);
            c2._semanticTags.UnionWith(other._semanticTags);
            return (c1, c2);
        }

        public byte[] Serialize()
        {
            using var ms = new System.IO.MemoryStream();
            using var w = new System.IO.BinaryWriter(ms);
            w.Write(_version); w.Write(_mutationRate); w.Write(_crossoverRate);
            w.Write((uint)_activeFeatures); w.Write(_currentLOD);
            w.Write(_semanticTags.Count);
            foreach (string tag in _semanticTags) w.Write(tag);
            w.Write(_microGenomes.Count);
            foreach (var kvp in _microGenomes) { w.Write(kvp.Key); kvp.Value.Serialize(w); }
            w.Write(_synapticConnections.Count);
            foreach (var c in _synapticConnections)
            { w.Write(c.SourceGenome); w.Write(c.SourceProperty); w.Write(c.TargetGenome); w.Write(c.TargetProperty); w.Write(c.Weight); w.Write((byte)c.Type); }
            return ms.ToArray();
        }

        public static MaterialGenome Deserialize(byte[] data)
        {
            using var ms = new System.IO.MemoryStream(data);
            using var r = new System.IO.BinaryReader(ms);
            var g = new MaterialGenome { _version = r.ReadInt32(), _mutationRate = r.ReadSingle(), _crossoverRate = r.ReadSingle(), _activeFeatures = (MaterialFeatureFlags)r.ReadUInt32(), _currentLOD = r.ReadInt32() };
            int tc = r.ReadInt32(); for (int i = 0; i < tc; i++) g._semanticTags.Add(r.ReadString());
            int mc = r.ReadInt32(); for (int i = 0; i < mc; i++) { string n = r.ReadString(); g._microGenomes[n] = MicroGenome.Deserialize(r); }
            int cc = r.ReadInt32(); for (int i = 0; i < cc; i++) g._synapticConnections.Add(new SynapticConnection { SourceGenome = r.ReadString(), SourceProperty = r.ReadString(), TargetGenome = r.ReadString(), TargetProperty = r.ReadString(), Weight = r.ReadSingle(), Type = (SynapticConnectionType)r.ReadByte() });
            return g;
        }

        public string ComputeHash()
        {
            using var sha = SHA256.Create();
            return Convert.ToBase64String(sha.ComputeHash(Serialize()));
        }

        public MaterialGenome Clone() => Deserialize(Serialize());

        public void MergeWith(MaterialGenome other)
        {
            foreach (var kvp in other._microGenomes) _microGenomes[kvp.Key] = kvp.Value;
            _synapticConnections.AddRange(other._synapticConnections);
            _semanticTags.UnionWith(other._semanticTags);
            _activeFeatures |= other._activeFeatures;
            _version++;
        }

        public float ComputeSimilarity(MaterialGenome other)
        {
            if (other == null) return 0;
            var tagsA = _semanticTags;
            var tagsB = other._semanticTags;
            if (tagsA.Count == 0 && tagsB.Count == 0) return 1.0f;
            int intersection = tagsA.Intersect(tagsB).Count();
            int union = tagsA.Union(tagsB).Count();
            return union > 0 ? (float)intersection / union : 0;
        }
    }

    public class MicroGenome
    {
        private readonly Dictionary<string, PropertyGene> _genes;
        private readonly string _name;
        private int _lodComplexity;
        private int _version;

        public string Name => _name;
        public int Version => _version;
        public int GeneCount => _genes.Count;

        public MicroGenome(string name) { _name = name; _genes = new Dictionary<string, PropertyGene>(); _version = 1; }

        public void AddGene(string propertyName, PropertyGene gene) { _genes[propertyName] = gene; _version++; }
        public PropertyGene GetGene(string propertyName) { _genes.TryGetValue(propertyName, out var g); return g; }
        public IReadOnlyDictionary<string, PropertyGene> GetAllGenes() => _genes;
        public void SetLODComplexity(int lod) { _lodComplexity = lod; }

        public Dictionary<string, MaterialProperty> Evaluate(Dictionary<string, float> inputs)
        {
            var results = new Dictionary<string, MaterialProperty>();
            foreach (var kvp in _genes)
            {
                float inputVal = inputs.TryGetValue(kvp.Key, out var val) ? val : 0f;
                float geneValue = kvp.Value.Evaluate(inputVal, _lodComplexity);
                results[kvp.Key] = new MaterialProperty(kvp.Key, kvp.Value.PropertyType, geneValue, kvp.Value.MinValue, kvp.Value.MaxValue, kvp.Value.DefaultValue);
            }
            return results;
        }

        public void Mutate(Random rng, float rate)
        {
            foreach (var gene in _genes.Values) if (rng.NextDouble() < rate) gene.Mutate(rng);
            _version++;
        }

        public void Serialize(System.IO.BinaryWriter writer)
        {
            writer.Write(_name); writer.Write(_version); writer.Write(_genes.Count);
            foreach (var kvp in _genes) { writer.Write(kvp.Key); kvp.Value.Serialize(writer); }
        }

        public static MicroGenome Deserialize(System.IO.BinaryReader reader)
        {
            string name = reader.ReadString();
            int version = reader.ReadInt32();
            var micro = new MicroGenome(name) { _version = version };
            int gc = reader.ReadInt32();
            for (int i = 0; i < gc; i++) { string pn = reader.ReadString(); micro._genes[pn] = PropertyGene.Deserialize(reader); }
            return micro;
        }
    }

    public class PropertyGene
    {
        public string PropertyName { get; set; }
        public MaterialPropertyType PropertyType { get; set; }
        public float Value { get; set; }
        public float MinValue { get; set; }
        public float MaxValue { get; set; } = 1.0f;
        public float DefaultValue { get; set; }
        public float MutationRange { get; set; } = 0.1f;
        public int ExpressionPriority { get; set; }
        public int LODMin { get; set; }
        public int LODMax { get; set; } = 10;
        public ActivationFunction Activation { get; set; } = ActivationFunction.Linear;

        public float Evaluate(float input, int currentLOD)
        {
            if (currentLOD < LODMin || currentLOD > LODMax) return DefaultValue;
            float raw = Value + input;
            return Activation switch
            {
                ActivationFunction.Linear => Math.Clamp(raw, MinValue, MaxValue),
                ActivationFunction.Sigmoid => Math.Clamp(1.0f / (1.0f + (float)Math.Exp(-raw)), MinValue, MaxValue),
                ActivationFunction.Tanh => Math.Clamp((float)Math.Tanh(raw), MinValue, MaxValue),
                ActivationFunction.ReLU => Math.Clamp(Math.Max(0, raw), MinValue, MaxValue),
                ActivationFunction.Softplus => Math.Clamp((float)Math.Log(1 + Math.Exp(raw)), MinValue, MaxValue),
                ActivationFunction.Square => Math.Clamp(raw * raw, MinValue, MaxValue),
                ActivationFunction.Sqrt => Math.Clamp((float)Math.Sqrt(Math.Abs(raw)), MinValue, MaxValue),
                _ => Math.Clamp(raw, MinValue, MaxValue)
            };
        }

        public void Mutate(Random rng)
        {
            float range = (MaxValue - MinValue) * MutationRange;
            Value += (float)(rng.NextDouble() * 2.0 - 1.0) * range;
            Value = Math.Clamp(Value, MinValue, MaxValue);
        }

        public void Serialize(System.IO.BinaryWriter writer)
        {
            writer.Write(PropertyName); writer.Write((byte)PropertyType); writer.Write(Value); writer.Write(MinValue);
            writer.Write(MaxValue); writer.Write(DefaultValue); writer.Write(MutationRange);
            writer.Write(ExpressionPriority); writer.Write(LODMin); writer.Write(LODMax); writer.Write((byte)Activation);
        }

        public static PropertyGene Deserialize(System.IO.BinaryReader reader) =>
            new PropertyGene { PropertyName = reader.ReadString(), PropertyType = (MaterialPropertyType)reader.ReadByte(), Value = reader.ReadSingle(), MinValue = reader.ReadSingle(), MaxValue = reader.ReadSingle(), DefaultValue = reader.ReadSingle(), MutationRange = reader.ReadSingle(), ExpressionPriority = reader.ReadInt32(), LODMin = reader.ReadInt32(), LODMax = reader.ReadInt32(), Activation = (ActivationFunction)reader.ReadByte() };
    }

    public class SynapticConnection
    {
        public string SourceGenome { get; set; }
        public string SourceProperty { get; set; }
        public string TargetGenome { get; set; }
        public string TargetProperty { get; set; }
        public float Weight { get; set; } = 1.0f;
        public SynapticConnectionType Type { get; set; } = SynapticConnectionType.Excitatory;
    }

    public class ActivationKernel
    {
        public string Name { get; set; }
        public float Strength { get; set; } = 1.0f;
        public float Bias { get; set; }
        public ActivationFunction Function { get; set; } = ActivationFunction.Linear;
        public List<float> Parameters { get; set; } = new();

        public float Apply(float input)
        {
            float x = input * Strength + Bias;
            return Function switch
            {
                ActivationFunction.Linear => x,
                ActivationFunction.Sigmoid => 1.0f / (1.0f + (float)Math.Exp(-x)),
                ActivationFunction.Tanh => (float)Math.Tanh(x),
                ActivationFunction.ReLU => Math.Max(0, x),
                ActivationFunction.Softplus => (float)Math.Log(1 + Math.Exp(x)),
                ActivationFunction.Square => x * x,
                ActivationFunction.Sqrt => (float)Math.Sqrt(Math.Abs(x)),
                _ => x
            };
        }
    }

    // =========================================================================
    // SUBSTRATE MATERIAL
    // =========================================================================

    public class SubstrateMaterial : IDisposable
    {
        private readonly Dictionary<string, MaterialProperty> _properties;
        private readonly Dictionary<TextureChannel, TextureReference> _textureSlots;
        private readonly Dictionary<string, object> _metadata;
        private readonly List<MaterialLayer> _layers;
        private MaterialGenome _genome;
        private MaterialFeatureFlags _featureFlags;
        private MaterialDomain _domain;
        private string _name;
        private string _id;
        private int _version;
        private bool _isDirty;
        private ulong _hash;
        private SubstrateMaterial _parent;
        private readonly Dictionary<string, MaterialProperty> _overrides;
        private readonly List<string> _changeLog;
        private readonly object _lock = new();
        private bool _disposed;

        public string Name { get => _name; set { _name = value; MarkDirty(); } }
        public string Id => _id;
        public MaterialDomain Domain { get => _domain; set { _domain = value; MarkDirty(); } }
        public MaterialFeatureFlags FeatureFlags { get => _featureFlags; set { _featureFlags = value; MarkDirty(); } }
        public int Version => _version;
        public bool IsDirty => _isDirty;
        public MaterialGenome Genome { get => _genome; set { _genome = value; MarkDirty(); } }
        public SubstrateMaterial Parent => _parent;
        public IReadOnlyDictionary<string, MaterialProperty> Properties => _properties;
        public IReadOnlyDictionary<TextureChannel, TextureReference> TextureSlots => _textureSlots;
        public IReadOnlyList<MaterialLayer> Layers => _layers;

        public SubstrateMaterial(string name = "UntitledMaterial")
        {
            _id = Guid.NewGuid().ToString("N");
            _name = name;
            _properties = new Dictionary<string, MaterialProperty>();
            _textureSlots = new Dictionary<TextureChannel, TextureReference>();
            _metadata = new Dictionary<string, object>();
            _layers = new List<MaterialLayer>();
            _overrides = new Dictionary<string, MaterialProperty>();
            _changeLog = new List<string>();
            _featureFlags = MaterialFeatureFlags.None;
            _domain = MaterialDomain.Surface;
            _version = 1;
            _isDirty = true;
        }

        public void SetProperty(string name, MaterialProperty property) { lock (_lock) { _properties[name] = property; _changeLog.Add($"Set:{name}"); MarkDirty(); } }
        public void SetProperty(string name, float value) => SetProperty(name, new MaterialProperty(name, MaterialPropertyType.Float, value));
        public void SetProperty(string name, Color3 value) => SetProperty(name, new MaterialProperty(name, MaterialPropertyType.Color, value));
        public void SetProperty(string name, Vec3 value) => SetProperty(name, new MaterialProperty(name, MaterialPropertyType.Vec3, value));

        public MaterialProperty GetProperty(string name)
        {
            lock (_lock)
            {
                if (_properties.TryGetValue(name, out var p)) return p;
                if (_overrides.TryGetValue(name, out var o)) return o;
                if (_parent != null) return _parent.GetProperty(name);
                return null;
            }
        }

        public float GetFloat(string name, float fallback = 0f) { var p = GetProperty(name); return p != null ? p.AsFloat() : fallback; }
        public Color3 GetColor(string name, Color3 fallback = default) { var p = GetProperty(name); return p != null ? p.AsColor() : fallback; }
        public Vec3 GetVec3(string name, Vec3 fallback = default) { var p = GetProperty(name); return p != null ? p.AsVec3() : fallback; }
        public int GetInt(string name, int fallback = 0) { var p = GetProperty(name); return p != null ? p.AsInt() : fallback; }
        public bool GetBool(string name, bool fallback = false) { var p = GetProperty(name); return p != null ? p.AsBool() : fallback; }

        public void SetTexture(TextureChannel channel, TextureReference texture) { lock (_lock) { _textureSlots[channel] = texture; _changeLog.Add($"Tex:{channel}"); MarkDirty(); } }
        public TextureReference GetTexture(TextureChannel channel) { _textureSlots.TryGetValue(channel, out var t); return t; }
        public bool RemoveTexture(TextureChannel channel) { lock (_lock) { bool r = _textureSlots.Remove(channel); if (r) MarkDirty(); return r; } }

        public void SetMetadata(string key, object value) => _metadata[key] = value;
        public T GetMetadata<T>(string key, T fallback = default) => _metadata.TryGetValue(key, out var v) && v is T t ? t : fallback;

        public void SetOverride(string name, MaterialProperty property) { lock (_lock) { _overrides[name] = property; _changeLog.Add($"Override:{name}"); MarkDirty(); } }
        public void ClearOverrides() { lock (_lock) { _overrides.Clear(); MarkDirty(); } }

        public void SetParent(SubstrateMaterial parent) { _parent = parent; MarkDirty(); }

        public void AddLayer(MaterialLayer layer) { lock (_lock) { _layers.Add(layer); MarkDirty(); } }
        public bool RemoveLayer(string name) { lock (_lock) { int i = _layers.FindIndex(l => l.Name == name); if (i >= 0) { _layers.RemoveAt(i); MarkDirty(); return true; } return false; } }
        public MaterialLayer GetLayer(string name) => _layers.Find(l => l.Name == name);

        public SubstrateMaterial CreateVariant(string variantName, Dictionary<string, object> overrides)
        {
            var v = new SubstrateMaterial(variantName) { _parent = this, _domain = _domain, _featureFlags = _featureFlags };
            foreach (var kvp in _textureSlots) v._textureSlots[kvp.Key] = kvp.Value;
            foreach (var kvp in _properties) v._properties[kvp.Key] = kvp.Value;
            if (_genome != null) v._genome = _genome.Clone();
            foreach (var kvp in overrides)
                if (_properties.TryGetValue(kvp.Key, out var bp)) v._overrides[kvp.Key] = bp.WithValue(kvp.Value);
            v.MarkDirty();
            return v;
        }

        public SubstrateMaterial Clone(string newName = null)
        {
            var c = new SubstrateMaterial(newName ?? $"{_name}_Clone") { _domain = _domain, _featureFlags = _featureFlags, _parent = _parent, _version = _version };
            foreach (var kvp in _properties) c._properties[kvp.Key] = kvp.Value;
            foreach (var kvp in _textureSlots) c._textureSlots[kvp.Key] = kvp.Value;
            foreach (var kvp in _metadata) c._metadata[kvp.Key] = kvp.Value;
            foreach (var kvp in _overrides) c._overrides[kvp.Key] = kvp.Value;
            foreach (var l in _layers) c._layers.Add(l);
            if (_genome != null) c._genome = _genome.Clone();
            return c;
        }

        public void MergeWith(SubstrateMaterial other, bool overrideExisting = false)
        {
            lock (_lock)
            {
                foreach (var kvp in other._properties) if (overrideExisting || !_properties.ContainsKey(kvp.Key)) _properties[kvp.Key] = kvp.Value;
                foreach (var kvp in other._textureSlots) if (overrideExisting || !_textureSlots.ContainsKey(kvp.Key)) _textureSlots[kvp.Key] = kvp.Value;
                _featureFlags |= other._featureFlags;
                MarkDirty();
            }
        }

        public ulong ComputeHash()
        {
            if (!_isDirty) return _hash;
            using var ms = new System.IO.MemoryStream();
            using var w = new System.IO.BinaryWriter(ms);
            w.Write(_name ?? ""); w.Write((uint)_featureFlags); w.Write((byte)_domain);
            var sp = _properties.OrderBy(k => k.Key).ToList();
            w.Write(sp.Count);
            foreach (var kvp in sp) { w.Write(kvp.Key); w.Write((byte)kvp.Value.Type); w.Write(kvp.Value.AsFloat()); }
            var st = _textureSlots.OrderBy(k => k.Key).ToList();
            w.Write(st.Count);
            foreach (var kvp in st) { w.Write((byte)kvp.Key); w.Write(kvp.Value?.Path ?? ""); }
            byte[] data = ms.ToArray();
            _hash = ComputeXXHash64(data);
            _isDirty = false;
            return _hash;
        }

        public bool CompareTo(SubstrateMaterial other) => other != null && ComputeHash() == other.ComputeHash();

        public IReadOnlyList<string> GetChangeLog() => _changeLog.AsReadOnly();
        public void ClearChangeLog() { lock (_lock) { _changeLog.Clear(); } }

        public void InitializeDefaults()
        {
            SetProperty("BaseColor", new MaterialProperty("BaseColor", MaterialPropertyType.Color, new Color3(0.8f, 0.8f, 0.8f), 0, 1, 0.8f, "Base albedo color", "Surface"));
            SetProperty("Roughness", new MaterialProperty("Roughness", MaterialPropertyType.Float, 0.5f, 0, 1, 0.5f, "Surface roughness", "Surface", "Slider"));
            SetProperty("Metallic", new MaterialProperty("Metallic", MaterialPropertyType.Float, 0.0f, 0, 1, 0f, "Metallic reflectance", "Surface", "Slider"));
            SetProperty("Specular", new MaterialProperty("Specular", MaterialPropertyType.Float, 0.5f, 0, 1, 0.5f, "Specular intensity", "Surface", "Slider"));
            SetProperty("NormalStrength", new MaterialProperty("NormalStrength", MaterialPropertyType.Float, 1.0f, 0, 2, 1f, "Normal map strength", "Surface", "Slider"));
            SetProperty("AOStrength", new MaterialProperty("AOStrength", MaterialPropertyType.Float, 1.0f, 0, 1, 1f, "AO strength", "Surface", "Slider"));
            SetProperty("EmissiveIntensity", new MaterialProperty("EmissiveIntensity", MaterialPropertyType.Float, 0.0f, 0, 100, 0f, "Emissive intensity", "Emissive", "Slider"));
            SetProperty("Opacity", new MaterialProperty("Opacity", MaterialPropertyType.Float, 1.0f, 0, 1, 1f, "Opacity", "Transparency", "Slider"));
            SetProperty("SubsurfaceRadius", new MaterialProperty("SubsurfaceRadius", MaterialPropertyType.Vec3, new Vec3(1.0f, 0.2f, 0.1f), 0, 5, 1f, "SSS radius", "Subsurface"));
            SetProperty("SubsurfaceColor", new MaterialProperty("SubsurfaceColor", MaterialPropertyType.Color, new Color3(0.8f, 0.2f, 0.1f), 0, 1, 0.8f, "SSS color", "Subsurface"));
            SetProperty("ClearCoat", new MaterialProperty("ClearCoat", MaterialPropertyType.Float, 0.0f, 0, 1, 0f, "Clear coat intensity", "ClearCoat", "Slider"));
            SetProperty("ClearCoatRoughness", new MaterialProperty("ClearCoatRoughness", MaterialPropertyType.Float, 0.01f, 0, 1, 0.01f, "Clear coat roughness", "ClearCoat", "Slider"));
            SetProperty("Anisotropy", new MaterialProperty("Anisotropy", MaterialPropertyType.Float, 0.0f, -1, 1, 0f, "Anisotropy", "Anisotropy", "Slider"));
            SetProperty("AnisotropyRotation", new MaterialProperty("AnisotropyRotation", MaterialPropertyType.Float, 0.0f, 0, 6.2832f, 0f, "Anisotropy rotation", "Anisotropy", "Slider"));
            SetProperty("Sheen", new MaterialProperty("Sheen", MaterialPropertyType.Float, 0.0f, 0, 1, 0f, "Sheen intensity", "Sheen", "Slider"));
            SetProperty("SheenRoughness", new MaterialProperty("SheenRoughness", MaterialPropertyType.Float, 0.5f, 0, 1, 0.5f, "Sheen roughness", "Sheen", "Slider"));
            SetProperty("Transmission", new MaterialProperty("Transmission", MaterialPropertyType.Float, 0.0f, 0, 1, 0f, "Transmission", "Transparency", "Slider"));
            SetProperty("IOR", new MaterialProperty("IOR", MaterialPropertyType.Float, 1.5f, 1, 2.5f, 1.5f, "IOR", "Transparency", "Slider"));
            SetProperty("Thickness", new MaterialProperty("Thickness", MaterialPropertyType.Float, 0.5f, 0, 10, 0.5f, "Thickness", "Transparency", "Slider"));
            SetProperty("DisplacementScale", new MaterialProperty("DisplacementScale", MaterialPropertyType.Float, 0.1f, 0, 5, 0.1f, "Displacement scale", "Surface", "Slider"));
            SetProperty("TessellationFactor", new MaterialProperty("TessellationFactor", MaterialPropertyType.Float, 0.0f, 0, 64, 0f, "Tessellation factor", "Geometry", "Slider"));
            MarkDirty();
        }

        public Dictionary<TextureChannel, string> GetActiveTexturePaths()
        {
            var result = new Dictionary<TextureChannel, string>();
            foreach (var kvp in _textureSlots)
                if (!string.IsNullOrEmpty(kvp.Value?.Path))
                    result[kvp.Key] = kvp.Value.Path;
            return result;
        }

        public MaterialFeatureFlags ComputeActiveFeatures()
        {
            var flags = MaterialFeatureFlags.None;
            if (GetFloat("SubsurfaceRadius") > 0 || GetFloat("SubsurfaceColor") != default) flags |= MaterialFeatureFlags.SubsurfaceScattering;
            if (GetFloat("ClearCoat") > 0) flags |= MaterialFeatureFlags.ClearCoat;
            if (Math.Abs(GetFloat("Anisotropy")) > 0) flags |= MaterialFeatureFlags.Anisotropy;
            if (GetFloat("Sheen") > 0) flags |= MaterialFeatureFlags.Sheen;
            if (GetFloat("Transmission") > 0) flags |= MaterialFeatureFlags.Transmission;
            if (GetFloat("EmissiveIntensity") > 0) flags |= MaterialFeatureFlags.Emissive;
            if (GetFloat("Opacity") < 1) flags |= MaterialFeatureFlags.DitheredOpacity;
            if (_layers.Count > 1) flags |= MaterialFeatureFlags.MaskedBlend;
            if (_textureSlots.ContainsKey(TextureChannel.Height)) flags |= MaterialFeatureFlags.ParallaxOcclusion;
            if (_textureSlots.ContainsKey(TextureChannel.PixelDepthOffset)) flags |= MaterialFeatureFlags.PixelDepthOffset;
            if (_textureSlots.ContainsKey(TextureChannel.SubsurfaceColor)) flags |= MaterialFeatureFlags.DiffuseProfile;
            if (_domain == MaterialDomain.Cloth) flags |= MaterialFeatureFlags.ClothBRDF;
            if (_domain == MaterialDomain.Hair) flags |= MaterialFeatureFlags.HairBRDF;
            if (_domain == MaterialDomain.Eye) flags |= MaterialFeatureFlags.EyeBRDF;
            _featureFlags = flags;
            return flags;
        }

        public List<string> GetMissingTextures()
        {
            var missing = new List<string>();
            var required = new[] { TextureChannel.Albedo, TextureChannel.Normal, TextureChannel.Roughness };
            foreach (var ch in required)
                if (!_textureSlots.ContainsKey(ch) || string.IsNullOrEmpty(_textureSlots[ch]?.Path))
                    missing.Add(ch.ToString());
            return missing;
        }

        public float ComputeComplexityScore()
        {
            float score = _properties.Count * 0.1f;
            score += _textureSlots.Count * 0.5f;
            score += _layers.Count * 0.3f;
            if (_featureFlags.HasFlag(MaterialFeatureFlags.SubsurfaceScattering)) score += 2.0f;
            if (_featureFlags.HasFlag(MaterialFeatureFlags.ClearCoat)) score += 1.5f;
            if (_featureFlags.HasFlag(MaterialFeatureFlags.Anisotropy)) score += 1.0f;
            if (_featureFlags.HasFlag(MaterialFeatureFlags.Sheen)) score += 1.0f;
            if (_featureFlags.HasFlag(MaterialFeatureFlags.Transmission)) score += 2.0f;
            if (_featureFlags.HasFlag(MaterialFeatureFlags.HairBRDF)) score += 3.0f;
            if (_featureFlags.HasFlag(MaterialFeatureFlags.EyeBRDF)) score += 3.0f;
            if (_genome != null) score += _genome.MicroGenomeCount * 0.5f;
            return score;
        }

        private void MarkDirty() { _isDirty = true; _version++; }

        private static ulong ComputeXXHash64(byte[] data)
        {
            const ulong P1 = 11400714785074694791UL, P2 = 14029467366897019727UL, P3 = 1609587929392869UL, P4 = 9650029242287828579UL, P5 = 2862933555777941757UL;
            ulong v1 = P5 + (uint)data.Length, v2 = P4, v3 = P3, v4 = P2;
            int offset = 0, rem = data.Length;
            while (rem >= 32)
            {
                v1 = Rnd(v1, BitConverter.ToUInt64(data, offset)); offset += 8;
                v2 = Rnd(v2, BitConverter.ToUInt64(data, offset)); offset += 8;
                v3 = Rnd(v3, BitConverter.ToUInt64(data, offset)); offset += 8;
                v4 = Rnd(v4, BitConverter.ToUInt64(data, offset)); offset += 8;
                rem -= 32;
            }
            ulong h = RotL(v1, 1) + RotL(v2, 7) + RotL(v3, 12) + RotL(v4, 18);
            while (rem >= 8) { h ^= Rnd(0, BitConverter.ToUInt64(data, offset)); h = RotL(h, 27) * P1 + P4; offset += 8; rem -= 8; }
            while (rem > 0) { h ^= (ulong)data[offset] * P5; h = RotL(h, 11) * P1; offset++; rem--; }
            h ^= (uint)data.Length; h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
            return h;
            static ulong Rnd(ulong acc, ulong input) { acc += input * P2; acc = RotL(acc, 31); acc *= P1; return acc; }
            static ulong RotL(ulong v, int c) => (v << c) | (v >> (64 - c));
        }

        public void Dispose()
        {
            if (!_disposed) { _properties.Clear(); _textureSlots.Clear(); _metadata.Clear(); _layers.Clear(); _overrides.Clear(); _changeLog.Clear(); _disposed = true; }
            GC.SuppressFinalize(this);
        }
    }

    public class MaterialLayer
    {
        public string Name { get; set; }
        public float Opacity { get; set; } = 1.0f;
        public LayerBlendMode BlendMode { get; set; } = LayerBlendMode.AlphaBlend;
        public int UVSet { get; set; }
        public SubstrateMaterial Material { get; set; }
        public TextureReference Mask { get; set; }
        public float HeightBlendSharpness { get; set; } = 10.0f;
        public bool IsEnabled { get; set; } = true;
        public float SortOrder { get; set; }
    }

    // =========================================================================
    // BSDF EVALUATOR
    // =========================================================================

    public class BsdfEvaluator
    {
        private const float PI = 3.14159265358979323846f;
        private const float INV_PI = 0.31830988618379067154f;
        private const float INV_2PI = 0.15915494309189533577f;
        private const float INV_4PI = 0.07957747154594766788f;
        private const float EPSILON = 1e-6f;
        private const float MIN_ROUGHNESS = 0.04f;
        private const float F0_DIELECTRIC = 0.04f;

        public Color3 EvaluateBxDF(Vec3 incoming, Vec3 outgoing, SubstrateMaterial material, Vec3 normal)
        {
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            if (NdotV < EPSILON || NdotL < EPSILON) return Color3.Black;

            Color3 albedo = material.GetColor("BaseColor", new Color3(0.8f));
            float roughness = Math.Max(material.GetFloat("Roughness", 0.5f), MIN_ROUGHNESS);
            float metallic = material.GetFloat("Metallic", 0f);
            float specular = material.GetFloat("Specular", 0.5f);
            float clearCoat = material.GetFloat("ClearCoat", 0f);
            float clearCoatRough = Math.Max(material.GetFloat("ClearCoatRoughness", 0.01f), MIN_ROUGHNESS);
            float sheen = material.GetFloat("Sheen", 0f);
            float sheenRoughness = material.GetFloat("SheenRoughness", 0.5f);
            float transmission = material.GetFloat("Transmission", 0f);

            Vec3 halfVec = (incoming + outgoing).Normalized;
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float VdotH = Math.Max(outgoing.Dot(halfVec), 0.0f);
            float LdotH = Math.Max(incoming.Dot(halfVec), 0.0f);

            float fresnelBase = 0.5f + 2.0f * specular * F0_DIELECTRIC;
            Color3 diffuseColor = albedo * (1.0f - metallic);
            Color3 specularColor = Color3.Lerp(new Color3(fresnelBase), albedo, metallic);

            Color3 diffuse = EvaluateDiffuse(incoming, outgoing, normal, diffuseColor, roughness, material);
            Color3 specularResult = EvaluateSpecular(incoming, outgoing, normal, halfVec, specularColor, roughness, metallic, material);
            Color3 clearCoatResult = clearCoat > EPSILON ? EvaluateClearCoat(incoming, outgoing, normal, halfVec, clearCoat, clearCoatRough) : Color3.Black;
            Color3 sheenResult = sheen > EPSILON ? EvaluateSheen(incoming, outgoing, normal, halfVec, albedo, sheen, sheenRoughness) : Color3.Black;
            Color3 transmissionResult = transmission > EPSILON ? EvaluateTransmission(incoming, outgoing, normal, material, transmission) : Color3.Black;

            Color3 result = diffuse + specularResult;
            float ccFresnel = SchlickFresnel(VdotH, 0.04f);
            float ccWeight = clearCoat * ccFresnel;
            result = result * (1.0f - ccWeight) + clearCoatResult * ccWeight;
            result = result + sheenResult * (1.0f - metallic);
            result = result + transmissionResult * transmission;

            return result * NdotL;
        }

        public Color3 EvaluateBxDFWithAO(Vec3 incoming, Vec3 outgoing, SubstrateMaterial material, Vec3 normal, float ao)
        {
            return EvaluateBxDF(incoming, outgoing, material, normal) * ao;
        }

        public Color3 EvaluateDiffuse(Vec3 incoming, Vec3 outgoing, Vec3 normal, Color3 diffuseColor, float roughness, SubstrateMaterial material)
        {
            float diffuseFactor = EvaluateDisneyDiffuse(incoming, outgoing, normal, roughness);
            return diffuseColor * diffuseFactor * INV_PI;
        }

        private float EvaluateDisneyDiffuse(Vec3 incoming, Vec3 outgoing, Vec3 normal, float roughness)
        {
            float NdotV = Math.Max(normal.Dot(outgoing), 0.0f);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float FD90 = 0.5f + 2.0f * roughness * (incoming + outgoing).LengthSquared * 0.25f;
            float lightScatter = 1.0f + (FD90 - 1.0f) * (float)Math.Pow(1.0f - NdotL, 5.0f);
            float viewScatter = 1.0f + (FD90 - 1.0f) * (float)Math.Pow(1.0f - NdotV, 5.0f);
            return lightScatter * viewScatter;
        }

        public float LambertianDiffuse() => INV_PI;

        public float EvaluateOrenNayar(Vec3 incoming, Vec3 outgoing, Vec3 normal, float roughness)
        {
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), 0.0f);
            float angleL = (float)Math.Acos(Math.Min(NdotL, 1.0f));
            float angleV = (float)Math.Acos(Math.Min(NdotV, 1.0f));
            float alpha = Math.Max(angleL, angleV);
            float beta = Math.Min(angleL, angleV);
            float sigma2 = roughness * roughness;
            float A = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f);
            float B = 0.45f * sigma2 / (sigma2 + 0.09f);
            Vec3 lPerp = incoming - normal * NdotL;
            Vec3 vPerp = outgoing - normal * NdotV;
            float cosPhiDiff = (lPerp.LengthSquared > EPSILON && vPerp.LengthSquared > EPSILON)
                ? Math.Max(lPerp.Normalized.Dot(vPerp.Normalized), 0.0f) : 0.0f;
            return INV_PI * (A + B * cosPhiDiff * (float)Math.Sin(alpha) * (float)Math.Tan(beta));
        }

        public Color3 EvaluateSpecular(Vec3 incoming, Vec3 outgoing, Vec3 normal, Vec3 halfVec,
            Color3 specularColor, float roughness, float metallic, SubstrateMaterial material)
        {
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float VdotH = Math.Max(outgoing.Dot(halfVec), 0.0f);
            float D = EvaluateGGXDistribution(NdotH, roughness);
            float G = EvaluateSmithGGXGeometry(NdotV, NdotL, roughness);
            Color3 F = EvaluateSchlickFresnel(VdotH, specularColor);
            float specBRDF = (D * G) / Math.Max(4.0f * NdotV * NdotL, EPSILON);
            return F * specBRDF;
        }

        public Color3 EvaluateSpecularDirect(Vec3 incoming, Vec3 outgoing, Vec3 normal, Vec3 halfVec,
            Color3 specularColor, float roughness, float metallic)
        {
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float VdotH = Math.Max(outgoing.Dot(halfVec), 0.0f);
            float D = EvaluateGGXDistribution(NdotH, roughness);
            float G = EvaluateSmithGGXGeometry(NdotV, NdotL, roughness);
            Color3 F = EvaluateSchlickFresnel(VdotH, specularColor);
            return F * D * G / Math.Max(4.0f * NdotV * NdotL, EPSILON);
        }

        public float EvaluateGGXDistribution(float NdotH, float roughness)
        {
            float a = roughness * roughness;
            float a2 = a * a;
            float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
            return a2 / (PI * denom * denom + EPSILON);
        }

        public float EvaluateBeckmannDistribution(float NdotH, float roughness)
        {
            float a = roughness * roughness;
            float a2 = a * a;
            float cos2h = NdotH * NdotH;
            float denom = cos2h * cos2h;
            return (float)Math.Exp((cos2h - 1.0f) / (a2 * cos2h + EPSILON)) / (PI * a2 * denom + EPSILON);
        }

        public float EvaluateGGXAnisotropicDistribution(float TdotH, float BdotH, float NdotH,
            float roughnessT, float roughnessB)
        {
            float at = roughnessT * roughnessT;
            float ab = roughnessB * roughnessB;
            float a2 = at * ab;
            float d = TdotH * TdotH / (at * at) + BdotH * BdotH / (ab * ab) + NdotH * NdotH;
            return 1.0f / (PI * a2 * d * d + EPSILON);
        }

        public float EvaluateSmithGGXGeometry(float NdotV, float NdotL, float roughness)
        {
            float a = roughness * roughness;
            float a2 = a * a;
            float GGXV = NdotV * (float)Math.Sqrt(NdotV * NdotV * (1.0f - a2) + a2);
            float GGXL = NdotL * (float)Math.Sqrt(NdotL * NdotL * (1.0f - a2) + a2);
            return (2.0f * NdotV * NdotL) / (GGXV + GGXL + EPSILON);
        }

        public float EvaluateSmithGGXGeometrySeparable(float NdotV, float NdotL, float roughness)
        {
            return EvaluateGGXGeometrySingle(NdotV, roughness) * EvaluateGGXGeometrySingle(NdotL, roughness);
        }

        public float EvaluateGGXGeometrySingle(float NdotDir, float roughness)
        {
            float a = roughness * roughness;
            float a2 = a * a;
            float denom = NdotDir * (float)Math.Sqrt(1.0f - a2) + a2;
            return (2.0f * NdotDir) / (NdotDir + denom + EPSILON);
        }

        public float EvaluateSmithGGXGeometryAnisotropic(float NdotV, float NdotL,
            float TdotV, float BdotV, float TdotL, float BdotL, float roughnessT, float roughnessB)
        {
            float at = roughnessT * roughnessT;
            float ab = roughnessB * roughnessB;
            float GV = 1.0f / ((float)Math.Sqrt(TdotV * TdotV / (at * at) + BdotV * BdotV / (ab * ab) + NdotV * NdotV));
            float GL = 1.0f / ((float)Math.Sqrt(TdotL * TdotL / (at * at) + BdotL * BdotL / (ab * ab) + NdotL * NdotL));
            return GV * GL;
        }

        public float EvaluateKelemenGeometry(float VdotH, float LdotH)
        {
            return 0.25f / (VdotH * VdotH + EPSILON);
        }

        public float EvaluateImplicitGeometry()
        {
            return 0.5f;
        }

        public Color3 EvaluateAshikhminShirley(Vec3 incoming, Vec3 outgoing, Vec3 normal, Vec3 halfVec,
            Color3 specularColor, float roughnessX, float roughnessY)
        {
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float VdotH = Math.Max(outgoing.Dot(halfVec), 0.0f);

            Vec3 tangent = GetTangent(normal);
            Vec3 bitangent = normal.Cross(tangent).Normalized;
            float HdotT = halfVec.Dot(tangent);
            float HdotB = halfVec.Dot(bitangent);

            float hx = roughnessX * roughnessX;
            float hy = roughnessY * roughnessY;
            float D = 1.0f / (PI * hx * hy *
                (HdotT * HdotT / (hx * hx) + HdotB * HdotB / (hy * hy) + NdotH * NdotH) *
                (HdotT * HdotT / (hx * hx) + HdotB * HdotB / (hy * hy) + NdotH * NdotH) + EPSILON);

            Color3 F = EvaluateSchlickFresnel(VdotH, specularColor);
            float denom = 4.0f * NdotL * NdotV * (NdotL + NdotV - NdotL * NdotV);
            return F * D * Math.Max(denom, EPSILON);
        }

        public Color3 EvaluateWard(Vec3 incoming, Vec3 outgoing, Vec3 normal, Vec3 tangent,
            Color3 specularColor, float alphaX, float alphaY)
        {
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            Vec3 halfVec = (incoming + outgoing).Normalized;
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);

            Vec3 bitangent = normal.Cross(tangent).Normalized;
            float HdotT = halfVec.Dot(tangent);
            float HdotB = halfVec.Dot(bitangent);

            float exp = -2.0f * ((HdotT * HdotT) / (alphaX * alphaX) + (HdotB * HdotB) / (alphaY * alphaY)) / (NdotH * NdotH + EPSILON);
            float D = (float)Math.Exp(exp) / (4.0f * PI * alphaX * alphaY * (float)Math.Sqrt(NdotL * NdotV) + EPSILON);

            Color3 F = EvaluateSchlickFresnel(NdotH, specularColor);
            return F * D;
        }

        public Color3 EvaluateCookTorrance(Vec3 incoming, Vec3 outgoing, Vec3 normal, Vec3 halfVec,
            Color3 specularColor, float roughness, float metallic)
        {
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float VdotH = Math.Max(outgoing.Dot(halfVec), 0.0f);

            float D = EvaluateBeckmannDistribution(NdotH, roughness);
            float G = EvaluateSmithGGXGeometry(NdotV, NdotL, roughness);
            Color3 F = EvaluateSchlickFresnel(VdotH, specularColor);
            return F * D * G / Math.Max(4.0f * NdotV * NdotL, EPSILON);
        }

        public Color3 EvaluateClearCoat(Vec3 incoming, Vec3 outgoing, Vec3 normal, Vec3 halfVec,
            float clearCoat, float roughness)
        {
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float VdotH = Math.Max(outgoing.Dot(halfVec), 0.0f);
            float D = EvaluateGGXDistribution(NdotH, roughness);
            float G = EvaluateSmithGGXGeometry(NdotV, NdotL, roughness);
            float F = SchlickFresnel(VdotH, F0_DIELECTRIC);
            return new Color3((D * G * F) / Math.Max(4.0f * NdotV * NdotL, EPSILON) * clearCoat);
        }

        public Color3 EvaluateSheen(Vec3 incoming, Vec3 outgoing, Vec3 normal, Vec3 halfVec,
            Color3 baseColor, float sheenIntensity, float roughness)
        {
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float D = EvaluateCharlieDistribution(NdotH, roughness);
            float G = EvaluateCharlieGeometry(NdotV, NdotL, roughness);
            return baseColor * D * G * sheenIntensity;
        }

        public float EvaluateCharlieDistribution(float NdotH, float roughness)
        {
            float invR = 1.0f / Math.Max(roughness, 0.001f);
            float cos2h = NdotH * NdotH;
            float sin2h = Math.Max(1.0f - cos2h, 0.0f);
            return (2.0f + invR) * (float)Math.Pow(sin2h, invR * 0.5f) / (2.0f * PI);
        }

        public float EvaluateCharlieGeometry(float NdotV, float NdotL, float roughness)
        {
            float a = roughness * roughness;
            float a2 = a * a;
            float GGXV = NdotV * (NdotV * (1.0f - a2) + a2);
            float GGXL = NdotL * (NdotL * (1.0f - a2) + a2);
            return 1.0f / (GGXV + GGXL + EPSILON);
        }

        public Color3 EvaluateFabricBRDF(Vec3 incoming, Vec3 outgoing, Vec3 normal, Vec3 halfVec,
            Color3 baseColor, float roughness, float sheen)
        {
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float VdotH = Math.Max(outgoing.Dot(halfVec), 0.0f);

            // Diffuse component
            float diffuse = EvaluateDisneyDiffuse(incoming, outgoing, normal, roughness) * INV_PI;
            Color3 diffResult = baseColor * diffuse;

            // Sheen component
            float D = EvaluateCharlieDistribution(NdotH, roughness);
            float G = EvaluateCharlieGeometry(NdotV, NdotL, roughness);
            Color3 sheenColor = baseColor * sheen * D * G;

            return diffResult + sheenColor;
        }

        public Color3 EvaluateTransmission(Vec3 incoming, Vec3 outgoing, Vec3 normal,
            SubstrateMaterial material, float transmission)
        {
            float NdotV = Math.Max(normal.Dot(outgoing), 0.0f);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float ior = material.GetFloat("IOR", 1.5f);
            float thickness = material.GetFloat("Thickness", 0.5f);
            float roughness = Math.Max(material.GetFloat("Roughness", 0.5f), MIN_ROUGHNESS);
            Color3 transColor = material.GetColor("BaseColor", Color3.White);

            float F = SchlickFresnel(NdotV, F0FromIOR(ior));
            float T = 1.0f - F;
            float absorption = (float)Math.Exp(-thickness * 2.0f);
            Color3 absorbedColor = transColor * absorption;
            float diffuseTransmission = INV_PI * T * NdotL;
            return absorbedColor * diffuseTransmission;
        }

        public Color3 EvaluateSubsurface(Vec3 incoming, Vec3 outgoing, Vec3 position,
            Vec3 normal, SubstrateMaterial material)
        {
            Vec3 scatterDistance = material.GetVec3("SubsurfaceRadius", new Vec3(1.0f, 0.2f, 0.1f));
            Color3 subsurfaceColor = material.GetColor("SubsurfaceColor", new Color3(0.8f, 0.2f, 0.1f));
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);

            float avgScatter = (scatterDistance.X + scatterDistance.Y + scatterDistance.Z) / 3.0f;
            float albedo = subsurfaceColor.Luminance;
            float sigmaTr = (float)Math.Sqrt(3.0f * (1.0f - albedo));
            float d = Math.Max(avgScatter, 0.001f);
            float r = (incoming - outgoing).Length;
            float profileR = (float)Math.Exp(-sigmaTr * r / d) / (4.0f * PI * d);
            float diffuse = (1.0f - F0_DIELECTRIC) * albedo * INV_PI;
            float sss = diffuse * NdotL + profileR * albedo * 0.5f;
            return subsurfaceColor * sss;
        }

        public Color3 EvaluateSubsurfaceProfile(Vec3 incoming, Vec3 outgoing, Vec3 normal,
            SubsurfaceProfile profile, Color3 subsurfaceColor)
        {
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float r = (incoming - outgoing).Length;

            // Multi-lobe diffusion profile
            Color3 result = Color3.Black;
            for (int i = 0; i < profile.LobeCount; i++)
            {
                float weight = profile.Weights[i];
                float spread = profile.Spreads[i];
                float falloff = profile.Falloffs[i];
                float profileValue = weight * (float)Math.Exp(-r / spread) / (4.0f * PI * spread * spread + EPSILON);
                result = result + new Color3(profileValue * falloff);
            }

            float diffuse = (1.0f - F0_DIELECTRIC) * INV_PI * NdotL;
            return subsurfaceColor * result * diffuse;
        }

        public Color3 EvaluateAnisotropic(Vec3 incoming, Vec3 outgoing, Vec3 normal,
            Vec3 tangent, Vec3 bitangent, Color3 specularColor,
            float roughnessT, float roughnessB, float anisotropy)
        {
            Vec3 halfVec = (incoming + outgoing).Normalized;
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float VdotH = Math.Max(outgoing.Dot(halfVec), 0.0f);

            float TdotH = tangent.Dot(halfVec);
            float BdotH = bitangent.Dot(halfVec);
            float at = roughnessT * roughnessT;
            float ab = roughnessB * roughnessB;
            float a2 = at * ab;
            float d = TdotH * TdotH / (at * at) + BdotH * BdotH / (ab * ab) + NdotH * NdotH;
            float D = 1.0f / (PI * a2 * d * d + EPSILON);

            float TdotV = tangent.Dot(outgoing);
            float BdotV = bitangent.Dot(outgoing);
            float TdotL = tangent.Dot(incoming);
            float BdotL = bitangent.Dot(incoming);
            float GV = 1.0f / ((float)Math.Sqrt(TdotV * TdotV / (at * at) + BdotV * BdotV / (ab * ab) + NdotV * NdotV));
            float GL = 1.0f / ((float)Math.Sqrt(TdotL * TdotL / (at * at) + BdotL * BdotL / (ab * ab) + NdotL * NdotL));
            float G = GV * GL;

            Color3 F = EvaluateSchlickFresnel(VdotH, specularColor);
            return F * D * G / Math.Max(4.0f * NdotV * NdotL, EPSILON);
        }

        public Color3 EvaluateHairMarschner(Vec3 incoming, Vec3 outgoing, Vec3 tangent,
            SubstrateMaterial material, float hairAlpha)
        {
            Color3 baseColor = material.GetColor("BaseColor", new Color3(0.3f, 0.15f, 0.05f));
            float roughness = material.GetFloat("Roughness", 0.3f);
            Vec3 normal = GetHairNormal(incoming, tangent);
            float thetaI = (float)Math.Asin(Math.Clamp(incoming.Dot(normal), -1, 1));
            float thetaR = (float)Math.Asin(Math.Clamp(outgoing.Dot(normal), -1, 1));

            float beta = roughness * roughness;
            float thetaD = thetaI - thetaR;
            float M_h = (float)Math.Exp(-thetaD * thetaD / (2.0f * beta));
            float phi = ComputeAzimuthalAngle(incoming, outgoing, tangent, normal);

            float fr = SchlickFresnel(Math.Abs((float)Math.Sin(thetaI)), F0_DIELECTRIC);
            Color3 R = new Color3(fr) * M_h * HairN_R(phi);
            float ft = 1.0f - fr;
            Color3 TT = baseColor * ft * ft * M_h * HairN_TT(phi);
            Color3 TRT = baseColor * baseColor * fr * M_h * HairN_TRT(phi);
            float ms = ft * ft * fr / (1.0f - fr * 0.5f + EPSILON);
            Color3 MS = baseColor * ms * (1.0f - M_h) * INV_PI;
            return (R + TT + TRT + MS) * Math.Max(normal.Dot(incoming), 0.0f);
        }

        public Color3 EvaluateHairKajiyaKay(Vec3 incoming, Vec3 outgoing, Vec3 tangent,
            SubstrateMaterial material)
        {
            Color3 baseColor = material.GetColor("BaseColor", new Color3(0.3f, 0.15f, 0.05f));
            float roughness = material.GetFloat("Roughness", 0.3f);
            float specularExponent = 1.0f / (roughness * roughness + 0.001f);

            float sinL = (float)Math.Sqrt(Math.Max(1.0f - incoming.Dot(tangent) * incoming.Dot(tangent), 0));
            float sinV = (float)Math.Sqrt(Math.Max(1.0f - outgoing.Dot(tangent) * outgoing.Dot(tangent), 0));
            float diffuse = sinL * sinV + incoming.Dot(tangent) * outgoing.Dot(tangent);

            Vec3 halfVec = (incoming + outgoing).Normalized;
            float sinH = (float)Math.Sqrt(Math.Max(1.0f - halfVec.Dot(tangent) * halfVec.Dot(tangent), 0));
            float specular = (float)Math.Pow(Math.Max(sinH, 0), specularExponent);

            return baseColor * diffuse * INV_PI + new Color3(specular * 0.5f);
        }

        public Color3 EvaluateEye(Vec3 incoming, Vec3 outgoing, Vec3 position, Vec3 normal,
            SubstrateMaterial material)
        {
            Color3 baseColor = material.GetColor("BaseColor", new Color3(0.4f, 0.25f, 0.15f));
            float roughness = material.GetFloat("Roughness", 0.1f);
            float ior = material.GetFloat("IOR", 1.376f);
            float NdotV = Math.Max(normal.Dot(outgoing), EPSILON);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);

            float irisPattern = GenerateIrisPattern(position);
            Color3 irisColor = baseColor * (0.5f + 0.5f * irisPattern);
            float irisAbsorption = (float)Math.Exp(-irisPattern * 0.5f);

            float angle = (float)Math.Acos(Math.Clamp(Math.Abs(outgoing.Dot(normal)), 0, 1));
            float scleraBlend = Math.Clamp(angle / (PI * 0.5f), 0, 1);
            Color3 scleraColor = new Color3(0.95f, 0.93f, 0.88f);
            Color3 eyeColor = Color3.Lerp(irisColor, scleraColor, scleraBlend * scleraBlend);

            Vec3 halfVec = (incoming + outgoing).Normalized;
            float D = EvaluateGGXDistribution(Math.Max(normal.Dot(halfVec), 0), roughness * 0.1f);
            float G = EvaluateSmithGGXGeometry(NdotV, NdotL, roughness * 0.1f);
            float F = SchlickFresnel(Math.Max(halfVec.Dot(outgoing), 0), F0FromIOR(ior));
            Color3 corneaSpec = new Color3(D * G * F / Math.Max(4.0f * NdotV * NdotL, EPSILON));

            Color3 diffuse = eyeColor * irisAbsorption * INV_PI;
            return diffuse + corneaSpec;
        }

        public float EvaluateMISWeight(float pdf1, float pdf2)
        {
            float w1 = pdf1 * pdf1;
            float w2 = pdf2 * pdf2;
            return w1 / (w1 + w2 + EPSILON);
        }

        public float EvaluateMISWeightPower(float pdf1, float pdf2, float power = 2)
        {
            float w1 = (float)Math.Pow(pdf1, power);
            float w2 = (float)Math.Pow(pdf2, power);
            return w1 / (w1 + w2 + EPSILON);
        }

        public float PdfGGXReflection(Vec3 incoming, Vec3 outgoing, Vec3 normal, float roughness)
        {
            Vec3 halfVec = (incoming + outgoing).Normalized;
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float NdotL = Math.Max(normal.Dot(incoming), 0.0f);
            float VdotH = Math.Max(outgoing.Dot(halfVec), EPSILON);
            float D = EvaluateGGXDistribution(NdotH, roughness);
            return D * NdotH / (4.0f * VdotH + EPSILON);
        }

        public float PdfCosineWeighted(Vec3 incoming, Vec3 normal)
        {
            return Math.Max(normal.Dot(incoming), 0.0f) * INV_PI;
        }

        public float PdfCharlie(Vec3 incoming, Vec3 outgoing, Vec3 normal, float roughness)
        {
            Vec3 halfVec = (incoming + outgoing).Normalized;
            float NdotH = Math.Max(normal.Dot(halfVec), 0.0f);
            float VdotH = Math.Max(outgoing.Dot(halfVec), EPSILON);
            float D = EvaluateCharlieDistribution(NdotH, roughness);
            return D * NdotH / (4.0f * VdotH + EPSILON);
        }

        public float PdfOrenNayar(Vec3 incoming, Vec3 outgoing, Vec3 normal, float roughness)
        {
            return Math.Max(normal.Dot(incoming), 0.0f) * INV_PI;
        }

        public Color3 IntegrateBRDF(Vec3 outgoing, SubstrateMaterial material, Vec3 normal, int sampleCount = 128)
        {
            Color3 irradiance = Color3.Black;
            var rng = Random.Shared;
            for (int i = 0; i < sampleCount; i++)
            {
                float u1 = (float)rng.NextDouble();
                float u2 = (float)rng.NextDouble();
                Vec3 tangent = GetTangent(normal);
                Vec3 bitangent = normal.Cross(tangent).Normalized;
                float theta = (float)Math.Acos(Math.Sqrt(1.0f - u1));
                float phi = 2.0f * PI * u2;
                Vec3 incoming = (normal * (float)Math.Cos(theta) +
                    tangent * ((float)Math.Sin(theta) * (float)Math.Cos(phi)) +
                    bitangent * ((float)Math.Sin(theta) * (float)Math.Sin(phi))).Normalized;
                irradiance = irradiance + EvaluateBxDF(incoming, outgoing, material, normal) * Math.Max(normal.Dot(incoming), 0.0f);
            }
            return irradiance / sampleCount;
        }

        // Fresnel
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SchlickFresnel(float cosTheta, float f0)
        {
            float f = 1.0f - Math.Clamp(cosTheta, 0, 1);
            float f3 = f * f * f;
            return f0 + (1.0f - f0) * f3 * f * f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Color3 EvaluateSchlickFresnel(float cosTheta, Color3 f0)
        {
            float f = 1.0f - Math.Clamp(cosTheta, 0, 1);
            float f3 = f * f * f;
            return f0 + (new Color3(1.0f) - f0) * f3 * f * f;
        }

        public static float ExactFresnelDielectric(float cosThetaI, float eta)
        {
            float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);
            if (sinThetaTSq >= 1.0f) return 1.0f;
            float cosThetaT = (float)Math.Sqrt(1.0f - sinThetaTSq);
            float rP = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
            float rS = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
            return (rP * rP + rS * rS) * 0.5f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float F0FromIOR(float ior) { float r = (ior - 1.0f) / (ior + 1.0f); return r * r; }

        // Utilities
        private static Vec3 GetTangent(Vec3 normal)
        {
            Vec3 up = Math.Abs(normal.Z) < 0.999f ? Vec3.UnitZ : Vec3.UnitX;
            return normal.Cross(up).Normalized;
        }

        private static Vec3 GetHairNormal(Vec3 incoming, Vec3 tangent)
        {
            Vec3 perp = incoming - tangent * incoming.Dot(tangent);
            return perp.LengthSquared > EPSILON ? perp.Normalized : GetTangent(tangent);
        }

        private float ComputeAzimuthalAngle(Vec3 incoming, Vec3 outgoing, Vec3 tangent, Vec3 normal)
        {
            Vec3 bitangent = tangent.Cross(normal).Normalized;
            float phiI = (float)Math.Atan2(incoming.Dot(bitangent), incoming.Dot(tangent));
            float phiR = (float)Math.Atan2(outgoing.Dot(bitangent), outgoing.Dot(tangent));
            return Math.Abs(phiI - phiR);
        }

        private float HairN_R(float phi) => (1.0f / (2.0f * PI)) * (1.0f + (float)Math.Cos(phi));
        private float HairN_TT(float phi) { float s = (float)Math.Sin(0.5f * (phi - PI)); return (1.0f / (4.0f * PI)) * s * s; }
        private float HairN_TRT(float phi) => (1.0f / (8.0f * PI)) * (1.0f + (float)Math.Cos(phi));

        private float GenerateIrisPattern(Vec3 position)
        {
            float r = (float)Math.Sqrt(position.X * position.X + position.Y * position.Y);
            float theta = (float)Math.Atan2(position.Y, position.X);
            float p = 0;
            p += 0.5f * (float)Math.Sin(theta * 12.0f + r * 20.0f);
            p += 0.25f * (float)Math.Sin(theta * 24.0f - r * 15.0f);
            p += 0.125f * (float)Math.Cos(theta * 36.0f + r * 30.0f);
            return Math.Clamp(p * 0.5f + 0.5f, 0, 1);
        }
    }

    // =========================================================================
    // SUBSURFACE PROFILE
    // =========================================================================

    public class SubsurfaceProfile
    {
        public int LobeCount { get; set; } = 3;
        public float[] Weights { get; set; } = new float[] { 0.5f, 0.35f, 0.15f };
        public float[] Spreads { get; set; } = new float[] { 0.03f, 0.15f, 1.0f };
        public float[] Falloffs { get; set; } = new float[] { 0.3f, 0.6f, 1.0f };
        public Color3 ScatterColor { get; set; } = new Color3(0.8f, 0.2f, 0.1f);

        public SubsurfaceProfile() { }

        public SubsurfaceProfile(float[] weights, float[] spreads, float[] falloffs, Color3 scatter)
        {
            LobeCount = weights.Length;
            Weights = weights;
            Spreads = spreads;
            Falloffs = falloffs;
            ScatterColor = scatter;
        }

        public float EvaluateProfile(float distance)
        {
            float result = 0;
            for (int i = 0; i < LobeCount; i++)
                result += Weights[i] * (float)Math.Exp(-distance / (Spreads[i] + 1e-6f));
            return result;
        }

        public Color3 EvaluateProfileColor(float distance)
        {
            return ScatterColor * EvaluateProfile(distance);
        }

        public static SubsurfaceProfile Lerp(SubsurfaceProfile a, SubsurfaceProfile b, float t)
        {
            int count = Math.Max(a.LobeCount, b.LobeCount);
            var weights = new float[count];
            var spreads = new float[count];
            var falloffs = new float[count];
            for (int i = 0; i < count; i++)
            {
                float wa = i < a.LobeCount ? a.Weights[i] : 0;
                float wb = i < b.LobeCount ? b.Weights[i] : 0;
                weights[i] = MathHelper.Lerp(wa, wb, t);
                spreads[i] = MathHelper.Lerp(
                    i < a.LobeCount ? a.Spreads[i] : 0,
                    i < b.LobeCount ? b.Spreads[i] : 0, t);
                falloffs[i] = MathHelper.Lerp(
                    i < a.LobeCount ? a.Falloffs[i] : 0,
                    i < b.LobeCount ? b.Falloffs[i] : 0, t);
            }
            var color = Color3.Lerp(a.ScatterColor, b.ScatterColor, t);
            return new SubsurfaceProfile(weights, spreads, falloffs, color);
        }

        public static SubsurfaceProfile FromSpectralData(float[] wavelengths, float[] scatteringCoefficients, int lobeCount = 3)
        {
            var weights = new float[lobeCount];
            var spreads = new float[lobeCount];
            var falloffs = new float[lobeCount];
            float totalCoeff = scatteringCoefficients.Sum();

            for (int i = 0; i < lobeCount; i++)
            {
                float t = (float)(i + 0.5) / lobeCount;
                int idx = Math.Min((int)(t * wavelengths.Length), wavelengths.Length - 1);
                weights[i] = scatteringCoefficients[idx] / (totalCoeff + 1e-6f);
                spreads[i] = wavelengths[idx] * 0.001f;
                falloffs[i] = 1.0f - weights[i];
            }

            float r = scatteringCoefficients.Length > 0 ? scatteringCoefficients[0] / (totalCoeff + 1e-6f) : 0.8f;
            float g = scatteringCoefficients.Length > 1 ? scatteringCoefficients[1] / (totalCoeff + 1e-6f) : 0.2f;
            float b = scatteringCoefficients.Length > 2 ? scatteringCoefficients[2] / (totalCoeff + 1e-6f) : 0.1f;

            return new SubsurfaceProfile(weights, spreads, falloffs, new Color3(r, g, b));
        }
    }

    // =========================================================================
    // MATERIAL LAYER STACK
    // =========================================================================

    public class MaterialLayerStack
    {
        private readonly List<MaterialLayerEntry> _layers;
        private bool _isDirty;

        public int LayerCount => _layers.Count;
        public bool IsDirty => _isDirty;

        public MaterialLayerStack() { _layers = new List<MaterialLayerEntry>(); _isDirty = true; }

        public void AddLayer(MaterialLayerEntry layer) { _layers.Add(layer); _layers.Sort((a, b) => a.SortOrder.CompareTo(b.SortOrder)); _isDirty = true; }
        public bool RemoveLayer(string name) { int i = _layers.FindIndex(l => l.Name == name); if (i >= 0) { _layers.RemoveAt(i); _isDirty = true; return true; } return false; }
        public void MoveLayer(string name, int newIndex) { int i = _layers.FindIndex(l => l.Name == name); if (i >= 0 && newIndex >= 0 && newIndex < _layers.Count) { var l = _layers[i]; _layers.RemoveAt(i); _layers.Insert(newIndex, l); _isDirty = true; } }
        public MaterialLayerEntry GetLayer(string name) => _layers.Find(l => l.Name == name);
        public void SetLayerEnabled(string name, bool enabled) { var l = GetLayer(name); if (l != null) { l.IsEnabled = enabled; _isDirty = true; } }
        public void SetLayerOpacity(string name, float opacity) { var l = GetLayer(name); if (l != null) { l.Opacity = MathHelper.Clamp(opacity, 0, 1); _isDirty = true; } }
        public IReadOnlyList<MaterialLayerEntry> GetAllLayers() => _layers.AsReadOnly();

        public BlendedMaterialResult Evaluate(Vec2 uv, float maskValue = 1.0f)
        {
            var result = new BlendedMaterialResult();
            var active = _layers.Where(l => l.IsEnabled).ToList();
            if (active.Count == 0) return result;

            var baseLayer = active.FirstOrDefault(l => l.IsBase);
            if (baseLayer != null)
            {
                result.BaseColor = baseLayer.GetBaseColor(uv);
                result.Roughness = baseLayer.GetRoughness(uv);
                result.Metallic = baseLayer.GetMetallic(uv);
                result.Normal = baseLayer.GetNormal(uv);
                result.AO = baseLayer.GetAO(uv);
                result.Emissive = baseLayer.GetEmissive(uv);
            }

            foreach (var layer in active.Where(l => !l.IsBase))
            {
                float layerMask = layer.GetMask(uv) * maskValue;
                float layerOpacity = layer.Opacity * layerMask;
                if (layerOpacity < MathHelper.EPSILON) continue;

                Color3 lbc = layer.GetBaseColor(uv);
                float lr = layer.GetRoughness(uv);
                float lm = layer.GetMetallic(uv);
                Vec3 ln = layer.GetNormal(uv);
                float la = layer.GetAO(uv);
                Color3 le = layer.GetEmissive(uv);

                switch (layer.BlendMode)
                {
                    case LayerBlendMode.Replace:
                        result.BaseColor = Color3.Lerp(result.BaseColor, lbc, layerOpacity);
                        result.Roughness = MathHelper.Lerp(result.Roughness, lr, layerOpacity);
                        result.Metallic = MathHelper.Lerp(result.Metallic, lm, layerOpacity);
                        result.Normal = Vec3.Lerp(result.Normal, ln, layerOpacity).Normalized;
                        result.AO = MathHelper.Lerp(result.AO, la, layerOpacity);
                        result.Emissive = Color3.Lerp(result.Emissive, le, layerOpacity);
                        break;
                    case LayerBlendMode.Add:
                        result.BaseColor = result.BaseColor + lbc * layerOpacity;
                        result.Roughness = MathHelper.Clamp(result.Roughness + lr * layerOpacity, 0, 1);
                        result.Metallic = MathHelper.Clamp(result.Metallic + lm * layerOpacity, 0, 1);
                        result.Emissive = result.Emissive + le * layerOpacity;
                        break;
                    case LayerBlendMode.Multiply:
                        result.BaseColor = Color3.Lerp(result.BaseColor, result.BaseColor * lbc, layerOpacity);
                        result.AO = MathHelper.Lerp(result.AO, result.AO * la, layerOpacity);
                        break;
                    case LayerBlendMode.Overlay:
                        Color3 ov = OverlayBlend(result.BaseColor, lbc);
                        result.BaseColor = Color3.Lerp(result.BaseColor, ov, layerOpacity);
                        break;
                    case LayerBlendMode.Screen:
                        result.BaseColor = Color3.Lerp(result.BaseColor, ScreenBlend(result.BaseColor, lbc), layerOpacity);
                        break;
                    case LayerBlendMode.Subtract:
                        result.BaseColor = result.BaseColor - lbc * layerOpacity;
                        break;
                    case LayerBlendMode.AlphaBlend:
                        result.BaseColor = Color3.Lerp(result.BaseColor, lbc, layerOpacity);
                        result.Roughness = MathHelper.Lerp(result.Roughness, lr, layerOpacity);
                        result.Metallic = MathHelper.Lerp(result.Metallic, lm, layerOpacity);
                        break;
                    case LayerBlendMode.HeightBlend:
                        float blend = MathHelper.Saturate((layerOpacity) * layer.HeightBlendSharpness);
                        result.BaseColor = Color3.Lerp(result.BaseColor, lbc, blend);
                        result.Roughness = MathHelper.Lerp(result.Roughness, lr, blend);
                        break;
                    case LayerBlendMode.NormalBlend:
                        result.Normal = BlendNormals(result.Normal, ln, layerOpacity);
                        break;
                }
            }

            result.Normal = result.Normal.Normalized;
            result.AO = MathHelper.Clamp(result.AO, 0, 1);
            result.Roughness = MathHelper.Clamp(result.Roughness, 0.04f, 1);
            result.Metallic = MathHelper.Clamp(result.Metallic, 0, 1);
            _isDirty = false;
            return result;
        }

        public BlendedMaterialResult[] EvaluateBatch(Vec2[] uvs, float maskValue = 1.0f)
        {
            var results = new BlendedMaterialResult[uvs.Length];
            for (int i = 0; i < uvs.Length; i++) results[i] = Evaluate(uvs[i], maskValue);
            return results;
        }

        private static Color3 OverlayBlend(Color3 b, Color3 o) =>
            new Color3(
                b.R < 0.5f ? 2 * b.R * o.R : 1 - 2 * (1 - b.R) * (1 - o.R),
                b.G < 0.5f ? 2 * b.G * o.G : 1 - 2 * (1 - b.G) * (1 - o.G),
                b.B < 0.5f ? 2 * b.B * o.B : 1 - 2 * (1 - b.B) * (1 - o.B));

        private static Color3 ScreenBlend(Color3 a, Color3 b) =>
            new Color3(1 - (1 - a.R) * (1 - b.R), 1 - (1 - a.G) * (1 - b.G), 1 - (1 - a.B) * (1 - b.B));

        private static Vec3 BlendNormals(Vec3 n1, Vec3 n2, float t)
        {
            Vec3 r = n1 * (2.0f * n2.Z) + n2 * (2.0f * n1.Z - 1.0f);
            return Vec3.Lerp(n1, r, t).Normalized;
        }
    }

    public class MaterialLayerEntry
    {
        public string Name { get; set; }
        public float Opacity { get; set; } = 1.0f;
        public LayerBlendMode BlendMode { get; set; } = LayerBlendMode.AlphaBlend;
        public float SortOrder { get; set; }
        public bool IsBase { get; set; }
        public bool IsEnabled { get; set; } = true;
        public float HeightBlendSharpness { get; set; } = 10.0f;

        private SubstrateMaterial _material;
        private TextureReference _maskTexture;

        public void SetMaterial(SubstrateMaterial material) => _material = material;
        public void SetMask(TextureReference mask) => _maskTexture = mask;

        public Color3 GetBaseColor(Vec2 uv) => _material?.GetColor("BaseColor", new Color3(0.8f)) ?? new Color3(0.8f);
        public float GetRoughness(Vec2 uv) => _material?.GetFloat("Roughness", 0.5f) ?? 0.5f;
        public float GetMetallic(Vec2 uv) => _material?.GetFloat("Metallic", 0f) ?? 0f;
        public Vec3 GetNormal(Vec2 uv) => _material?.GetVec3("Normal", Vec3.UnitZ) ?? Vec3.UnitZ;
        public float GetAO(Vec2 uv) => _material?.GetFloat("AO", 1f) ?? 1f;
        public Color3 GetEmissive(Vec2 uv) => _material?.GetColor("Emissive", Color3.Black) ?? Color3.Black;
        public float GetMask(Vec2 uv) => _maskTexture != null ? 1.0f : 1.0f;
    }

    public class BlendedMaterialResult
    {
        public Color3 BaseColor { get; set; } = new Color3(0.8f);
        public float Roughness { get; set; } = 0.5f;
        public float Metallic { get; set; }
        public Vec3 Normal { get; set; } = Vec3.UnitZ;
        public float AO { get; set; } = 1.0f;
        public Color3 Emissive { get; set; }
        public float Opacity { get; set; } = 1.0f;
        public float Specular { get; set; } = 0.5f;
        public float Subsurface { get; set; }
        public float ClearCoat { get; set; }
        public float Sheen { get; set; }
    }

    // =========================================================================
    // TEXTURE MANAGER
    // =========================================================================

    public class TextureManager : IDisposable
    {
        private readonly ConcurrentDictionary<string, ManagedTexture> _textureCache;
        private readonly ConcurrentDictionary<string, TextureAtlas> _atlases;
        private long _totalMemoryBytes;
        private int _maxMemoryBytes;
        private bool _disposed;

        public int TextureCount => _textureCache.Count;
        public long TotalMemoryBytes => Interlocked.Read(ref _totalMemoryBytes);

        public TextureManager(int maxMemoryBytes = 512 * 1024 * 1024)
        {
            _textureCache = new ConcurrentDictionary<string, ManagedTexture>();
            _atlases = new ConcurrentDictionary<string, TextureAtlas>();
            _maxMemoryBytes = maxMemoryBytes;
        }

        public ManagedTexture LoadTexture(string path, int width = 0, int height = 0, TextureCompression format = TextureCompression.None, bool generateMipmaps = true)
        {
            if (_textureCache.TryGetValue(path, out var cached)) { cached.LastAccessTime = DateTime.UtcNow; cached.AccessCount++; return cached; }

            byte[] data = System.IO.File.Exists(path) ? System.IO.File.ReadAllBytes(path) : new byte[Math.Max(width, 1) * Math.Max(height, 1) * 4];
            int w = width > 0 ? width : 256;
            int h = height > 0 ? height : 256;

            var tex = new ManagedTexture
            {
                Path = path, Width = w, Height = h, Format = format, Data = data,
                MipLevels = generateMipmaps ? ComputeMipLevels(w, h) : 1,
                SizeBytes = data.Length, LastAccessTime = DateTime.UtcNow, AccessCount = 1
            };

            if (generateMipmaps && data.Length > 0) tex.MipData = GenerateMipmaps(data, w, h, format);

            _textureCache[path] = tex;
            Interlocked.Add(ref _totalMemoryBytes, tex.SizeBytes);
            return tex;
        }

        public ManagedTexture CreateProceduralTexture(string name, int width, int height, ProceduralTextureType type, int seed = 0)
        {
            var gen = new ProceduralTextureGenerator(seed);
            byte[] data = gen.Generate(type, width, height);
            var tex = new ManagedTexture
            {
                Path = $"procedural://{name}", Width = width, Height = height,
                Format = TextureCompression.None, Data = data,
                MipLevels = ComputeMipLevels(width, height), SizeBytes = data.Length,
                LastAccessTime = DateTime.UtcNow, AccessCount = 1
            };
            tex.MipData = GenerateMipmaps(data, width, height, TextureCompression.None);
            _textureCache[tex.Path] = tex;
            Interlocked.Add(ref _totalMemoryBytes, tex.SizeBytes);
            return tex;
        }

        public bool UnloadTexture(string path)
        {
            if (_textureCache.TryRemove(path, out var tex))
            {
                Interlocked.Add(ref _totalMemoryBytes, -tex.SizeBytes);
                tex.Data = null; tex.MipData = null;
                return true;
            }
            return false;
        }

        public int EvictLRU(int count = 0)
        {
            long targetBytes = count > 0 ? count * 1024L * 1024L : _maxMemoryBytes / 2;
            int evicted = 0;
            foreach (var kvp in _textureCache.OrderBy(k => k.Value.LastAccessTime).ToList())
            {
                if (Interlocked.Read(ref _totalMemoryBytes) <= targetBytes) break;
                if (UnloadTexture(kvp.Key)) evicted++;
            }
            return evicted;
        }

        public byte[][] GenerateMipmaps(byte[] baseData, int width, int height, TextureCompression format)
        {
            int levels = ComputeMipLevels(width, height);
            var mipData = new byte[levels - 1][];
            int cw = width, ch = height;
            byte[] cd = baseData;
            for (int i = 1; i < levels; i++)
            {
                int nw = Math.Max(1, cw / 2), nh = Math.Max(1, ch / 2);
                var ds = Downsample(cd, cw, ch, nw, nh);
                mipData[i - 1] = ds;
                cd = ds; cw = nw; ch = nh;
            }
            return mipData;
        }

        public TextureAtlas CreateAtlas(string name, List<string> texturePaths, int atlasSize = 4096)
        {
            foreach (var p in texturePaths)
                if (!_textureCache.ContainsKey(p)) LoadTexture(p);

            var atlas = new TextureAtlas { Name = name, Width = atlasSize, Height = atlasSize, Entries = new List<TextureAtlasEntry>() };
            int cols = (int)Math.Ceiling(Math.Sqrt(texturePaths.Count));
            int rows = (int)Math.Ceiling((float)texturePaths.Count / cols);

            for (int i = 0; i < texturePaths.Count; i++)
            {
                int col = i % cols, row = i / cols;
                atlas.Entries.Add(new TextureAtlasEntry
                {
                    TexturePath = texturePaths[i],
                    U0 = (float)col / cols, V0 = (float)row / rows,
                    U1 = (float)(col + 1) / cols, V1 = (float)(row + 1) / rows,
                    Width = atlasSize / cols, Height = atlasSize / rows
                });
            }
            _atlases[name] = atlas;
            return atlas;
        }

        public byte[] CompressTexture(byte[] rgbaData, int width, int height, TextureCompression format) =>
            format switch
            {
                TextureCompression.BC1 => CompressBC1(rgbaData, width, height),
                TextureCompression.BC3 => CompressBC3(rgbaData, width, height),
                TextureCompression.BC4 => CompressBC4(rgbaData, width, height),
                TextureCompression.BC5 => CompressBC5(rgbaData, width, height),
                TextureCompression.BC7 => CompressBC7(rgbaData, width, height),
                _ => rgbaData
            };

        public TextureMemoryStats GetMemoryStats()
        {
            var s = new TextureMemoryStats();
            foreach (var tex in _textureCache.Values)
            {
                s.TotalTextures++; s.TotalBytes += tex.SizeBytes;
                if (tex.MipData != null) s.MipmappedTextures++;
                if (tex.IsStreaming) s.StreamingTextures++;
                if (tex.Format != TextureCompression.None) s.CompressedTextures++;
            }
            s.AtlasCount = _atlases.Count;
            return s;
        }

        private static int ComputeMipLevels(int w, int h) => (int)Math.Floor(Math.Log2(Math.Max(w, h))) + 1;

        private static byte[] Downsample(byte[] src, int srcW, int srcH, int dstW, int dstH)
        {
            int bpp = 4;
            byte[] dst = new byte[dstW * dstH * bpp];
            for (int y = 0; y < dstH; y++)
                for (int x = 0; x < dstW; x++)
                {
                    int sx = Math.Min(x * srcW / dstW, srcW - 1);
                    int sy = Math.Min(y * srcH / dstH, srcH - 1);
                    int si = (sy * srcW + sx) * bpp;
                    int di = (y * dstW + x) * bpp;
                    if (si + 3 < src.Length && di + 3 < dst.Length)
                    {
                        dst[di] = src[si]; dst[di + 1] = src[si + 1];
                        dst[di + 2] = src[si + 2]; dst[di + 3] = src[si + 3];
                    }
                }
            return dst;
        }

        private static byte[] CompressBC1(byte[] data, int w, int h) => data;
        private static byte[] CompressBC3(byte[] data, int w, int h) => data;
        private static byte[] CompressBC4(byte[] data, int w, int h) => data;
        private static byte[] CompressBC5(byte[] data, int w, int h) => data;
        private static byte[] CompressBC7(byte[] data, int w, int h) => data;

        public void Dispose()
        {
            if (!_disposed)
            {
                foreach (var tex in _textureCache.Values) { tex.Data = null; tex.MipData = null; }
                _textureCache.Clear(); _atlases.Clear();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }

    public class ManagedTexture
    {
        public string Path { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public TextureCompression Format { get; set; }
        public byte[] Data { get; set; }
        public byte[][] MipData { get; set; }
        public int MipLevels { get; set; }
        public long SizeBytes { get; set; }
        public DateTime LastAccessTime { get; set; }
        public int AccessCount { get; set; }
        public bool IsStreaming { get; set; }
    }

    public class TextureAtlas
    {
        public string Name { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public List<TextureAtlasEntry> Entries { get; set; } = new();
    }

    public class TextureAtlasEntry
    {
        public string TexturePath { get; set; }
        public float U0 { get; set; }
        public float V0 { get; set; }
        public float U1 { get; set; }
        public float V1 { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
    }

    public class TextureMemoryStats
    {
        public int TotalTextures { get; set; }
        public long TotalBytes { get; set; }
        public int MipmappedTextures { get; set; }
        public int StreamingTextures { get; set; }
        public int CompressedTextures { get; set; }
        public int AtlasCount { get; set; }
    }

    // =========================================================================
    // PROCEDURAL TEXTURE GENERATOR
    // =========================================================================

    public class ProceduralTextureGenerator
    {
        private readonly Random _rng;

        public ProceduralTextureGenerator(int seed = 0) { _rng = new Random(seed); }

        public byte[] Generate(ProceduralTextureType type, int width, int height) =>
            type switch
            {
                ProceduralTextureType.Checkerboard => GenerateCheckerboard(width, height),
                ProceduralTextureType.Gradient => GenerateGradient(width, height),
                ProceduralTextureType.Noise => GenerateNoise(width, height),
                ProceduralTextureType.Brick => GenerateBrick(width, height),
                ProceduralTextureType.WoodGrain => GenerateWoodGrain(width, height),
                ProceduralTextureType.Marble => GenerateMarble(width, height),
                ProceduralTextureType.GradientNoise => GenerateGradientNoise(width, height),
                ProceduralTextureType.Voronoi => GenerateVoronoi(width, height),
                ProceduralTextureType.Cellular => GenerateCellular(width, height),
                ProceduralTextureType.Wave => GenerateWave(width, height),
                ProceduralTextureType.FractalNoise => GenerateFractalNoise(width, height),
                ProceduralTextureType.Turbulence => GenerateTurbulence(width, height),
                _ => GenerateNoise(width, height)
            };

        private byte[] GenerateCheckerboard(int w, int h)
        {
            var data = new byte[w * h * 4];
            int tileSize = Math.Max(4, Math.Min(w, h) / 8);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    bool white = ((x / tileSize) + (y / tileSize)) % 2 == 0;
                    byte c = (byte)(white ? 240 : 32);
                    int i = (y * w + x) * 4;
                    data[i] = c; data[i + 1] = c; data[i + 2] = c; data[i + 3] = 255;
                }
            return data;
        }

        private byte[] GenerateGradient(int w, int h)
        {
            var data = new byte[w * h * 4];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float t = (float)x / w;
                    byte c = (byte)(t * 255);
                    int i = (y * w + x) * 4;
                    data[i] = c; data[i + 1] = c; data[i + 2] = c; data[i + 3] = 255;
                }
            return data;
        }

        private byte[] GenerateNoise(int w, int h)
        {
            var data = new byte[w * h * 4];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    byte c = (byte)(_rng.Next(256));
                    int i = (y * w + x) * 4;
                    data[i] = c; data[i + 1] = c; data[i + 2] = c; data[i + 3] = 255;
                }
            return data;
        }

        private byte[] GenerateBrick(int w, int h)
        {
            var data = new byte[w * h * 4];
            int brickW = 64, brickH = 32, mortarW = 2;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int row = y / (brickH + mortarW);
                    int offsetX = (row % 2 == 0) ? 0 : brickW / 2;
                    int localX = (x + offsetX) % (brickW + mortarW);
                    int localY = y % (brickH + mortarW);
                    bool isMortar = localX < mortarW || localY < mortarW;

                    byte c;
                    if (isMortar)
                        c = 128;
                    else
                    {
                        float noise = (float)_rng.NextDouble() * 30 - 15;
                        c = (byte)Math.Clamp(180 + noise, 0, 255);
                    }
                    int i = (y * w + x) * 4;
                    data[i] = c; data[i + 1] = (byte)(c * 0.85f); data[i + 2] = (byte)(c * 0.7f); data[i + 3] = 255;
                }
            return data;
        }

        private byte[] GenerateWoodGrain(int w, int h)
        {
            var data = new byte[w * h * 4];
            float freq = 0.02f;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float fx = x * freq;
                    float fy = y * freq;
                    float n = PerlinNoise2D(fx, fy);
                    float ring = (float)Math.Sin((fx * 3 + fy * 2 + n * 5) * 4.0f) * 0.5f + 0.5f;
                    ring = ring * ring;
                    float grain = PerlinNoise2D(fx * 10, fy * 10) * 0.1f;
                    float val = MathHelper.Clamp(ring + grain, 0, 1);
                    byte r = (byte)(Math.Clamp(val * 180 + 40, 0, 255));
                    byte g = (byte)(Math.Clamp(val * 120 + 20, 0, 255));
                    byte b = (byte)(Math.Clamp(val * 60 + 10, 0, 255));
                    int i = (y * w + x) * 4;
                    data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
                }
            return data;
        }

        private byte[] GenerateMarble(int w, int h)
        {
            var data = new byte[w * h * 4];
            float freq = 0.015f;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float fx = x * freq;
                    float fy = y * freq;
                    float turb = Turbulence2D(fx, fy, 6) * 4.0f;
                    float val = (float)Math.Sin(fx * 2 + turb) * 0.5f + 0.5f;
                    val = val * 0.8f + 0.1f;
                    byte c = (byte)(Math.Clamp(val * 255, 0, 255));
                    int i = (y * w + x) * 4;
                    data[i] = c; data[i + 1] = (byte)(c * 0.98f); data[i + 2] = (byte)(c * 0.96f); data[i + 3] = 255;
                }
            return data;
        }

        private byte[] GenerateGradientNoise(int w, int h)
        {
            var data = new byte[w * h * 4];
            float freq = 0.01f;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float val = PerlinNoise2D(x * freq, y * freq) * 0.5f + 0.5f;
                    byte c = (byte)(Math.Clamp(val * 255, 0, 255));
                    int i = (y * w + x) * 4;
                    data[i] = c; data[i + 1] = c; data[i + 2] = c; data[i + 3] = 255;
                }
            return data;
        }

        private byte[] GenerateVoronoi(int w, int h)
        {
            var data = new byte[w * h * 4];
            int cellSize = 32;
            int pointsX = w / cellSize + 2;
            int pointsY = h / cellSize + 2;
            var points = new Vec2[pointsX * pointsY];
            for (int i = 0; i < points.Length; i++)
                points[i] = new Vec2((float)_rng.NextDouble() * cellSize, (float)_rng.NextDouble() * cellSize);

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int cx = x / cellSize, cy = y / cellSize;
                    float minDist = float.MaxValue;
                    float secondDist = float.MaxValue;
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int px = cx + dx, py = cy + dy;
                            if (px < 0 || px >= pointsX || py < 0 || py >= pointsY) continue;
                            int pi = py * pointsX + px;
                            Vec2 pt = new Vec2(px * cellSize + points[pi].X, py * cellSize + points[pi].Y);
                            float dist = (new Vec2(x, y) - pt).Length;
                            if (dist < minDist) { secondDist = minDist; minDist = dist; }
                            else if (dist < secondDist) secondDist = dist;
                        }
                    float edge = secondDist - minDist;
                    float val = MathHelper.Clamp(edge / 8.0f, 0, 1);
                    byte c = (byte)(val * 255);
                    int i = (y * w + x) * 4;
                    data[i] = c; data[i + 1] = c; data[i + 2] = c; data[i + 3] = 255;
                }
            return data;
        }

        private byte[] GenerateCellular(int w, int h) => GenerateVoronoi(w, h);

        private byte[] GenerateWave(int w, int h)
        {
            var data = new byte[w * h * 4];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float val = (float)Math.Sin(x * 0.1f + y * 0.05f) * 0.5f + 0.5f;
                    val += (float)Math.Sin(x * 0.05f - y * 0.08f) * 0.3f;
                    val = MathHelper.Clamp(val, 0, 1);
                    byte c = (byte)(val * 255);
                    int i = (y * w + x) * 4;
                    data[i] = c; data[i + 1] = c; data[i + 2] = c; data[i + 3] = 255;
                }
            return data;
        }

        private byte[] GenerateFractalNoise(int w, int h)
        {
            var data = new byte[w * h * 4];
            float freq = 0.01f;
            int octaves = 6;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float val = 0, amp = 1, f = freq, totalAmp = 0;
                    for (int o = 0; o < octaves; o++)
                    {
                        val += PerlinNoise2D(x * f, y * f) * amp;
                        totalAmp += amp;
                        amp *= 0.5f;
                        f *= 2;
                    }
                    val = val / totalAmp * 0.5f + 0.5f;
                    byte c = (byte)(Math.Clamp(val * 255, 0, 255));
                    int i = (y * w + x) * 4;
                    data[i] = c; data[i + 1] = c; data[i + 2] = c; data[i + 3] = 255;
                }
            return data;
        }

        private byte[] GenerateTurbulence(int w, int h)
        {
            var data = new byte[w * h * 4];
            float freq = 0.01f;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float val = Turbulence2D(x * freq, y * freq, 6);
                    val = MathHelper.Clamp(val, 0, 1);
                    byte c = (byte)(val * 255);
                    int i = (y * w + x) * 4;
                    data[i] = c; data[i + 1] = c; data[i + 2] = c; data[i + 3] = 255;
                }
            return data;
        }

        // Perlin noise helpers
        private static readonly int[] Permutation = { 151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };

        private float PerlinNoise2D(float x, float y)
        {
            int xi = ((int)Math.Floor(x) & 255);
            int yi = ((int)Math.Floor(y) & 255);
            float xf = x - (float)Math.Floor(x);
            float yf = y - (float)Math.Floor(y);
            float u = Fade(xf);
            float v = Fade(yf);

            int aa = Permutation[(Permutation[xi] + yi) & 255];
            int ab = Permutation[(Permutation[xi] + yi + 1) & 255];
            int ba = Permutation[(Permutation[(xi + 1) & 255] + yi) & 255];
            int bb = Permutation[(Permutation[(xi + 1) & 255] + yi + 1) & 255];

            float x1 = MathHelper.Lerp(Grad(aa, xf, yf), Grad(ba, xf - 1, yf), u);
            float x2 = MathHelper.Lerp(Grad(ab, xf, yf - 1), Grad(bb, xf - 1, yf - 1), u);
            return MathHelper.Lerp(x1, x2, v);
        }

        private float Turbulence2D(float x, float y, int octaves)
        {
            float val = 0, amp = 1, freq = 1, totalAmp = 0;
            for (int i = 0; i < octaves; i++)
            {
                val += Math.Abs(PerlinNoise2D(x * freq, y * freq)) * amp;
                totalAmp += amp;
                amp *= 0.5f;
                freq *= 2;
            }
            return val / totalAmp;
        }

        private static float Fade(float t) => t * t * t * (t * (t * 6 - 15) + 10);
        private static float Grad(int hash, float x, float y)
        {
            int h = hash & 3;
            float u = h < 2 ? x : y;
            float v = h < 2 ? y : x;
            return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
        }
    }

    // =========================================================================
    // MATERIAL EVALUATOR
    // =========================================================================

    public class MaterialEvaluator
    {
        private readonly BsdfEvaluator _bsdf;

        public MaterialEvaluator() { _bsdf = new BsdfEvaluator(); }

        public BlendedMaterialResult EvaluatePerPixel(SubstrateMaterial material, Vec2 uv, Vec3 position, Vec3 normal, Vec3 tangent)
        {
            var result = new BlendedMaterialResult();
            result.BaseColor = material.GetColor("BaseColor", new Color3(0.8f));
            result.Roughness = material.GetFloat("Roughness", 0.5f);
            result.Metallic = material.GetFloat("Metallic", 0f);
            result.Normal = ComputeTangentSpaceNormal(material, uv, normal, tangent);
            result.AO = material.GetFloat("AO", 1f) * material.GetFloat("AOStrength", 1f);
            result.Emissive = material.GetColor("Emissive", Color3.Black) * material.GetFloat("EmissiveIntensity", 0f);
            result.Opacity = material.GetFloat("Opacity", 1f);
            result.Specular = material.GetFloat("Specular", 0.5f);
            result.Subsurface = material.GetVec3("SubsurfaceRadius", Vec3.Zero).Length;
            result.ClearCoat = material.GetFloat("ClearCoat", 0f);
            result.Sheen = material.GetFloat("Sheen", 0f);

            // Interpolate across UV seams
            if (material.TextureSlots.ContainsKey(TextureChannel.Albedo))
            {
                var tex = material.TextureSlots[TextureChannel.Albedo];
                if (tex != null && !tex.Tiling.Equals(Vec2.One))
                {
                    result.BaseColor = SampleTiledAlbedo(tex, uv);
                }
            }

            // Apply material layers
            if (material.Layers.Count > 0)
            {
                var stack = new MaterialLayerStack();
                foreach (var layer in material.Layers)
                {
                    var entry = new MaterialLayerEntry
                    {
                        Name = layer.Name,
                        Opacity = layer.Opacity,
                        BlendMode = layer.BlendMode,
                        SortOrder = layer.SortOrder,
                        IsBase = stack.LayerCount == 0
                    };
                    if (layer.Material != null) entry.SetMaterial(layer.Material);
                    stack.AddLayer(entry);
                }
                var layered = stack.Evaluate(uv);
                result.BaseColor = layered.BaseColor;
                result.Roughness = layered.Roughness;
                result.Metallic = layered.Metallic;
                result.Normal = layered.Normal;
            }

            return result;
        }

        public Color3 EvaluateBSDF(SubstrateMaterial material, Vec3 incoming, Vec3 outgoing, Vec3 position, Vec3 normal)
        {
            return _bsdf.EvaluateBxDF(incoming, outgoing, material, normal);
        }

        public Color3 EvaluateFullShading(SubstrateMaterial material, Vec3 incoming, Vec3 outgoing, Vec3 position, Vec3 normal, Vec3 tangent, float ao)
        {
            var matResult = EvaluatePerPixel(material, new Vec2(0, 0), position, normal, tangent);
            Vec3 shadingNormal = matResult.Normal.Normalized;

            Color3 diffuse = matResult.BaseColor * (1 - matResult.Metallic) * ao;
            Color3 specular = Color3.Lerp(new Color3(0.04f), matResult.BaseColor, matResult.Metallic);

            float NdotL = Math.Max(shadingNormal.Dot(incoming), 0);
            float NdotV = Math.Max(shadingNormal.Dot(outgoing), MathHelper.EPSILON);
            Vec3 halfVec = (incoming + outgoing).Normalized;
            float NdotH = Math.Max(shadingNormal.Dot(halfVec), 0);
            float VdotH = Math.Max(outgoing.Dot(halfVec), 0);

            float D = _bsdf.EvaluateGGXDistribution(NdotH, matResult.Roughness);
            float G = _bsdf.EvaluateSmithGGXGeometry(NdotV, NdotL, matResult.Roughness);
            Color3 F = BsdfEvaluator.EvaluateSchlickFresnel(VdotH, specular);

            Color3 diffuseBRDF = diffuse * MathHelper.INV_PI;
            Color3 specularBRDF = F * D * G / Math.Max(4 * NdotV * NdotL, MathHelper.EPSILON);
            Color3 result = (diffuseBRDF + specularBRDF) * NdotL;

            // Clear coat
            if (matResult.ClearCoat > MathHelper.EPSILON)
            {
                float ccD = _bsdf.EvaluateGGXDistribution(NdotH, 0.01f);
                float ccG = _bsdf.EvaluateSmithGGXGeometry(NdotV, NdotL, 0.01f);
                float ccF = BsdfEvaluator.SchlickFresnel(VdotH, 0.04f);
                float ccWeight = matResult.ClearCoat * ccF;
                Color3 ccSpec = new Color3(ccD * ccG * ccF / Math.Max(4 * NdotV * NdotL, MathHelper.EPSILON));
                result = result * (1 - ccWeight) + ccSpec * ccWeight;
            }

            // Sheen
            if (matResult.Sheen > MathHelper.EPSILON)
            {
                float sheenD = _bsdf.EvaluateCharlieDistribution(NdotH, matResult.Roughness);
                float sheenG = _bsdf.EvaluateCharlieGeometry(NdotV, NdotL, matResult.Roughness);
                result = result + matResult.BaseColor * sheenD * sheenG * matResult.Sheen * (1 - matResult.Metallic);
            }

            return result + matResult.Emissive;
        }

        public Vec3 ComputeTangentSpaceNormal(SubstrateMaterial material, Vec2 uv, Vec3 geometricNormal, Vec3 geometricTangent)
        {
            var normalTex = material.GetTexture(TextureChannel.Normal);
            if (normalTex == null) return geometricNormal;

            float strength = material.GetFloat("NormalStrength", 1.0f);
            Vec3 tangentSpaceNormal = new Vec3(0, 0, 1);

            Vec3 T = geometricTangent.Normalized;
            Vec3 N = geometricNormal.Normalized;
            Vec3 B = N.Cross(T).Normalized;
            Mat3 TBN = new Mat3(T.X, B.X, N.X, T.Y, B.Y, N.Y, T.Z, B.Z, N.Z);

            Vec3 normal = TBN.Transform(tangentSpaceNormal);
            return Vec3.Lerp(geometricNormal, normal.Normalized, strength).Normalized;
        }

        public float ComputeVisibility(Vec3 position, Vec3 direction, float maxDistance, Func<Vec3, float> sampleHeight)
        {
            float t = 0;
            float stepSize = maxDistance / 32;
            float prevHeight = sampleHeight(position);
            for (int i = 0; i < 32; i++)
            {
                Vec3 samplePos = position + direction * t;
                float height = sampleHeight(samplePos);
                if (height > prevHeight + 0.001f) return 0;
                t += stepSize;
                prevHeight = height;
            }
            return 1;
        }

        public float ComputeParallaxOcclusionMapping(Vec2 uv, Vec3 viewDirTS, SubstrateMaterial material, float heightScale, int minSamples = 8, int maxSamples = 32)
        {
            float stepSize = 1.0f / MathHelper.Lerp(maxSamples, minSamples, Math.Abs(viewDirTS.Z));
            float currentHeight = 1.0f;
            Vec2 dt = new Vec2(viewDirTS.X, viewDirTS.Y) / (viewDirTS.Z + MathHelper.EPSILON) * stepSize * heightScale;
            Vec2 currentUV = uv;
            float prevHeight = 1.0f;

            for (int i = 0; i < maxSamples; i++)
            {
                currentUV -= dt;
                float sampledHeight = SampleHeightMap(material, currentUV);
                if (sampledHeight >= currentHeight) break;
                prevHeight = currentHeight;
                currentHeight -= stepSize;
            }

            float t = (currentHeight - 1.0f) / ((currentHeight - 1.0f) - (prevHeight - 1.0f) + MathHelper.EPSILON);
            return t;
        }

        public float ComputeCurvature(Vec3 position, Vec3 normal, Func<Vec3, Vec3> sampleNormal, float epsilon = 0.01f)
        {
            Vec3 dPdx = sampleNormal(position + new Vec3(epsilon, 0, 0)) - normal;
            Vec3 dPdy = sampleNormal(position + new Vec3(0, epsilon, 0)) - normal;
            float dNdx = dPdx.Length;
            float dNdy = dPdy.Length;
            return (dNdx + dNdy) * 0.5f / epsilon;
        }

        public float ComputeAmbientOcclusion(Vec3 position, Vec3 normal, Func<Vec3, float> sceneSDF, int sampleCount = 16, float radius = 1.0f)
        {
            float ao = 0;
            var rng = new Random();
            for (int i = 0; i < sampleCount; i++)
            {
                float u1 = (float)rng.NextDouble();
                float u2 = (float)rng.NextDouble();
                float theta = 2 * MathHelper.PI * u1;
                float phi = (float)Math.Acos(2 * u2 - 1);
                Vec3 sampleDir = new Vec3(
                    (float)(Math.Sin(phi) * Math.Cos(theta)),
                    (float)(Math.Sin(phi) * Math.Sin(theta)),
                    (float)Math.Cos(phi));
                if (sampleDir.Dot(normal) < 0) sampleDir = -sampleDir;
                float dist = sceneSDF(position + sampleDir * radius);
                ao += MathHelper.Clamp(dist / radius, 0, 1);
            }
            return ao / sampleCount;
        }

        public Color3 ComputeSubsurfaceScattering(Vec3 position, Vec3 normal, Vec3 lightDir, SubsurfaceProfile profile, Color3 lightColor)
        {
            float NdotL = Math.Max(normal.Dot(lightDir), 0);
            float distort = 0.2f;
            Vec3 distortedNormal = (normal + lightDir * distort).Normalized;
            float wrap = MathHelper.Saturate((normal.Dot(lightDir) + distort) / (1 + distort));
            Color3 sss = profile.EvaluateProfileColor(1.0f - wrap);
            return sss * lightColor * wrap;
        }

        private float SampleHeightMap(SubstrateMaterial material, Vec2 uv)
        {
            var heightTex = material.GetTexture(TextureChannel.Height);
            if (heightTex == null) return 0;
            return material.GetFloat("DisplacementScale", 0.1f) * 0.5f;
        }

        private Color3 SampleTiledAlbedo(TextureReference tex, Vec2 uv)
        {
            Vec2 tiledUV = new Vec2(
                (uv.X * tex.Tiling.X + tex.Offset.X) % 1.0f,
                (uv.Y * tex.Tiling.Y + tex.Offset.Y) % 1.0f);
            return new Color3(0.8f);
        }
    }

    // =========================================================================
    // MATERIAL SHADER GENERATOR
    // =========================================================================

    public class MaterialShaderGenerator
    {
        public ShaderSource GenerateShaderSource(SubstrateMaterial material, ShaderLanguage language = ShaderLanguage.HLSL, ShaderStage stage = ShaderStage.Pixel)
        {
            var features = material.ComputeActiveFeatures();
            string source = language switch
            {
                ShaderLanguage.HLSL => GenerateHLSL(material, features, stage),
                ShaderLanguage.GLSL => GenerateGLSL(material, features, stage),
                ShaderLanguage.Slang => GenerateSlang(material, features, stage),
                _ => GenerateHLSL(material, features, stage)
            };

            var defines = GenerateDefines(features);
            string hash = ComputeShaderHash(source);

            return new ShaderSource
            {
                Source = source,
                Language = language,
                Stage = stage,
                Features = features,
                Hash = hash,
                Defines = defines,
                EstimatedInstructionCount = EstimateInstructions(source),
                TextureSampleCount = material.TextureSlots.Count
            };
        }

        public List<ShaderSource> GenerateAllPermutations(SubstrateMaterial material, ShaderLanguage language = ShaderLanguage.HLSL)
        {
            var results = new List<ShaderSource>();
            var features = material.ComputeActiveFeatures();

            // Generate for each feature combination (limited to avoid explosion)
            var featureList = new[]
            {
                MaterialFeatureFlags.SubsurfaceScattering,
                MaterialFeatureFlags.ClearCoat,
                MaterialFeatureFlags.Anisotropy,
                MaterialFeatureFlags.Sheen,
                MaterialFeatureFlags.Transmission,
                MaterialFeatureFlags.Emissive
            };

            // Base shader
            results.Add(GenerateShaderSource(material, language, ShaderStage.Vertex));
            results.Add(GenerateShaderSource(material, language, ShaderStage.Pixel));

            // Feature-specific permutations
            foreach (var flag in featureList)
            {
                if (features.HasFlag(flag))
                {
                    var variant = material.Clone();
                    variant.FeatureFlags = flag;
                    results.Add(GenerateShaderSource(variant, language, ShaderStage.Pixel));
                }
            }

            return results;
        }

        public Dictionary<ShaderLanguage, ShaderSource> GenerateAllLanguages(SubstrateMaterial material, ShaderStage stage = ShaderStage.Pixel)
        {
            var results = new Dictionary<ShaderLanguage, ShaderSource>();
            foreach (ShaderLanguage lang in Enum.GetValues(typeof(ShaderLanguage)))
            {
                if (lang != ShaderLanguage.SPIRV) // SPIRV typically compiled from HLSL/GLSL
                    results[lang] = GenerateShaderSource(material, lang, stage);
            }
            return results;
        }

        private string GenerateHLSL(SubstrateMaterial material, MaterialFeatureFlags features, ShaderStage stage)
        {
            if (stage == ShaderStage.Vertex) return GenerateHLSLVertex(material, features);

            var sb = new StringBuilder();
            sb.AppendLine("// Auto-generated Substrate Material HLSL Shader");
            sb.AppendLine($"// Features: {features}");
            sb.AppendLine("#pragma target 6.0");
            sb.AppendLine("#include \"Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl\"");
            sb.AppendLine("#include \"Packages/com.unity.render-pipelines.core/ShaderLibrary/BSDF.hlsl\"");
            sb.AppendLine("#include \"Packages/com.unity.render-pipelines.core/ShaderLibrary/NormalSurfaceGradient.hlsl\"");
            sb.AppendLine();

            // Uniforms
            sb.AppendLine("CBUFFER_START(SubstrateMaterial)");
            sb.AppendLine("    float4 _BaseColor;");
            sb.AppendLine("    float _Roughness;");
            sb.AppendLine("    float _Metallic;");
            sb.AppendLine("    float _Specular;");
            sb.AppendLine("    float _NormalStrength;");
            sb.AppendLine("    float _AOStrength;");
            sb.AppendLine("    float _EmissiveIntensity;");
            sb.AppendLine("    float _Opacity;");

            if (features.HasFlag(MaterialFeatureFlags.SubsurfaceScattering))
            {
                sb.AppendLine("    float3 _SubsurfaceRadius;");
                sb.AppendLine("    float4 _SubsurfaceColor;");
            }
            if (features.HasFlag(MaterialFeatureFlags.ClearCoat))
            {
                sb.AppendLine("    float _ClearCoat;");
                sb.AppendLine("    float _ClearCoatRoughness;");
            }
            if (features.HasFlag(MaterialFeatureFlags.Anisotropy))
            {
                sb.AppendLine("    float _Anisotropy;");
                sb.AppendLine("    float _AnisotropyRotation;");
            }
            if (features.HasFlag(MaterialFeatureFlags.Sheen))
            {
                sb.AppendLine("    float _Sheen;");
                sb.AppendLine("    float _SheenRoughness;");
            }
            if (features.HasFlag(MaterialFeatureFlags.Transmission))
            {
                sb.AppendLine("    float _Transmission;");
                sb.AppendLine("    float _IOR;");
                sb.AppendLine("    float _Thickness;");
            }
            sb.AppendLine("CBUFFER_END");
            sb.AppendLine();

            // Texture declarations
            foreach (var kvp in material.TextureSlots)
            {
                string texName = $"_{kvp.Key}";
                sb.AppendLine($"TEXTURE2D({texName});");
                sb.AppendLine($"SAMPLER(sampler{texName});");
            }
            sb.AppendLine();

            // Structs
            sb.AppendLine("struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; float4 tangentOS : TANGENT; float2 uv : TEXCOORD0; float2 uv1 : TEXCOORD1; };");
            sb.AppendLine("struct Varyings { float4 positionCS : SV_POSITION; float3 positionWS : TEXCOORD0; float3 normalWS : TEXCOORD1; float4 tangentWS : TEXCOORD2; float2 uv : TEXCOORD3; float3 viewDirWS : TEXCOORD4; };");
            sb.AppendLine();

            // Main functions
            sb.AppendLine("float3 GetBaseColor(Varyings input) {");
            sb.AppendLine($"    float4 baseColor = SAMPLE_TEXTURE2D(_Albedo, sampler_Albedo, input.uv);");
            sb.AppendLine("    return baseColor.rgb * _BaseColor.rgb;");
            sb.AppendLine("}");
            sb.AppendLine();

            sb.AppendLine("float GetRoughness(Varyings input) {");
            sb.AppendLine($"    float roughness = SAMPLE_TEXTURE2D(_Roughness, sampler_Roughness, input.uv).r;");
            sb.AppendLine("    return max(roughness * _Roughness, 0.04);");
            sb.AppendLine("}");
            sb.AppendLine();

            sb.AppendLine("float GetMetallic(Varyings input) {");
            sb.AppendLine($"    return SAMPLE_TEXTURE2D(_Metallic, sampler_Metallic, input.uv).r * _Metallic;");
            sb.AppendLine("}");
            sb.AppendLine();

            if (features.HasFlag(MaterialFeatureFlags.ClearCoat))
            {
                sb.AppendLine("float EvaluateClearCoat(float3 V, float3 L, float3 N, float3 H) {");
                sb.AppendLine("    float NdotH = saturate(dot(N, H));");
                sb.AppendLine("    float NdotV = saturate(dot(N, V));");
                sb.AppendLine("    float NdotL = saturate(dot(N, L));");
                sb.AppendLine("    float D = D_GGX(NdotH, _ClearCoatRoughness);");
                sb.AppendLine("    float G = V_SmithJointGGX(NdotV, NdotL, _ClearCoatRoughness);");
                sb.AppendLine("    float F = F_Schlick(dot(H, V), 0.04);");
                sb.AppendLine("    return _ClearCoat * D * G * F / max(4 * NdotV * NdotL, 0.001);");
                sb.AppendLine("}");
                sb.AppendLine();
            }

            if (features.HasFlag(MaterialFeatureFlags.Sheen))
            {
                sb.AppendLine("float3 EvaluateSheen(float3 V, float3 L, float3 N, float3 H, float3 baseColor) {");
                sb.AppendLine("    float NdotH = saturate(dot(N, H));");
                sb.AppendLine("    float NdotV = saturate(dot(N, V));");
                sb.AppendLine("    float NdotL = saturate(dot(N, L));");
                sb.AppendLine("    float D = CharlieD(NdotH, _SheenRoughness);");
                sb.AppendLine("    float G = CharlieV(NdotV, NdotL, _SheenRoughness);");
                sb.AppendLine("    return baseColor * _Sheen * D * G;");
                sb.AppendLine("}");
                sb.AppendLine();
            }

            // Fragment entry
            sb.AppendLine("float4 frag(Varyings input) : SV_Target {");
            sb.AppendLine("    float3 N = normalize(input.normalWS);");
            sb.AppendLine("    float3 V = normalize(input.viewDirWS);");
            sb.AppendLine("    float3 L = normalize(_MainLightPosition.xyz);");
            sb.AppendLine("    float3 H = normalize(V + L);");
            sb.AppendLine();
            sb.AppendLine("    float3 baseColor = GetBaseColor(input);");
            sb.AppendLine("    float roughness = GetRoughness(input);");
            sb.AppendLine("    float metallic = GetMetallic(input);");
            sb.AppendLine("    float ao = SAMPLE_TEXTURE2D(_AO, sampler_AO, input.uv).r * _AOStrength;");
            sb.AppendLine();
            sb.AppendLine("    float NdotV = max(dot(N, V), 0.001);");
            sb.AppendLine("    float NdotL = max(dot(N, L), 0);");
            sb.AppendLine("    float NdotH = max(dot(N, H), 0);");
            sb.AppendLine("    float VdotH = max(dot(V, H), 0);");
            sb.AppendLine();
            sb.AppendLine("    float3 diffuseColor = baseColor * (1 - metallic);");
            sb.AppendLine("    float3 specularColor = lerp(0.04, baseColor, metallic);");
            sb.AppendLine();
            sb.AppendLine("    float D = D_GGX(NdotH, roughness);");
            sb.AppendLine("    float G = V_SmithJointGGX(NdotV, NdotL, roughness);");
            sb.AppendLine("    float3 F = F_Schlick(VdotH, specularColor);");
            sb.AppendLine();
            sb.AppendLine("    float3 diffuse = diffuseColor * Lambert();");
            sb.AppendLine("    float3 specular = F * D * G;");
            sb.AppendLine();
            sb.AppendLine("    float3 color = (diffuse + specular) * NdotL * ao;");

            if (features.HasFlag(MaterialFeatureFlags.ClearCoat))
                sb.AppendLine("    color += EvaluateClearCoat(V, L, N, H);");

            if (features.HasFlag(MaterialFeatureFlags.Sheen))
                sb.AppendLine("    color += EvaluateSheen(V, L, N, H, baseColor);");

            sb.AppendLine("    color += baseColor * SAMPLE_TEXTURE2D(_Emissive, sampler_Emissive, input.uv).rgb * _EmissiveIntensity;");
            sb.AppendLine("    return float4(color, _Opacity);");
            sb.AppendLine("}");

            return sb.ToString();
        }

        private string GenerateHLSLVertex(SubstrateMaterial material, MaterialFeatureFlags features)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// Auto-generated Substrate Material Vertex Shader");
            sb.AppendLine("#pragma target 6.0");
            sb.AppendLine("#include \"Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl\"");
            sb.AppendLine();
            sb.AppendLine("struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; float4 tangentOS : TANGENT; float2 uv : TEXCOORD0; };");
            sb.AppendLine("struct Varyings { float4 positionCS : SV_POSITION; float3 positionWS : TEXCOORD0; float3 normalWS : TEXCOORD1; float4 tangentWS : TEXCOORD2; float2 uv : TEXCOORD3; float3 viewDirWS : TEXCOORD4; };");
            sb.AppendLine();

            if (features.HasFlag(MaterialFeatureFlags.WorldPositionOffset))
            {
                sb.AppendLine("float3 _WPOAmplitude;");
                sb.AppendLine("float _WPOFrequency;");
                sb.AppendLine("float _WPOSpeed;");
                sb.AppendLine("float3 ApplyWorldPositionOffset(float3 posOS, float2 uv) {");
                sb.AppendLine("    float wave = sin(uv.x * _WPOFrequency + _Time.y * _WPOSpeed);");
                sb.AppendLine("    return posOS + _WPOAmplitude * wave;");
                sb.AppendLine("}");
                sb.AppendLine();
            }

            sb.AppendLine("Varyings vert(Attributes input) {");
            sb.AppendLine("    Varyings output;");
            sb.AppendLine("    float3 posOS = input.positionOS.xyz;");

            if (features.HasFlag(MaterialFeatureFlags.WorldPositionOffset))
                sb.AppendLine("    posOS = ApplyWorldPositionOffset(posOS, input.uv);");

            sb.AppendLine("    output.positionCS = TransformObjectToHClip(posOS);");
            sb.AppendLine("    output.positionWS = TransformObjectToWorld(posOS);");
            sb.AppendLine("    output.normalWS = TransformObjectToWorldNormal(input.normalOS);");
            sb.AppendLine("    output.tangentWS = float4(TransformObjectToWorldDir(input.tangentOS.xyz), input.tangentOS.w);");
            sb.AppendLine("    output.uv = input.uv;");
            sb.AppendLine("    output.viewDirWS = GetWorldSpaceViewDir(output.positionWS);");
            sb.AppendLine("    return output;");
            sb.AppendLine("}");

            return sb.ToString();
        }

        private string GenerateGLSL(SubstrateMaterial material, MaterialFeatureFlags features, ShaderStage stage)
        {
            if (stage == ShaderStage.Vertex) return GenerateGLSLVertex(material, features);

            var sb = new StringBuilder();
            sb.AppendLine("// Auto-generated Substrate Material GLSL Shader");
            sb.AppendLine("#version 450");
            sb.AppendLine("#extension GL_EXT_nonuniform_qualifier : enable");
            sb.AppendLine();
            sb.AppendLine("layout(std140, binding = 0) uniform SubstrateMaterial {");
            sb.AppendLine("    vec4 BaseColor;");
            sb.AppendLine("    float Roughness;");
            sb.AppendLine("    float Metallic;");
            sb.AppendLine("    float Specular;");
            sb.AppendLine("    float NormalStrength;");
            sb.AppendLine("    float AOStrength;");
            sb.AppendLine("    float EmissiveIntensity;");
            sb.AppendLine("    float Opacity;");

            if (features.HasFlag(MaterialFeatureFlags.SubsurfaceScattering))
            {
                sb.AppendLine("    vec3 SubsurfaceRadius;");
                sb.AppendLine("    vec4 SubsurfaceColor;");
            }
            if (features.HasFlag(MaterialFeatureFlags.ClearCoat))
            {
                sb.AppendLine("    float ClearCoat;");
                sb.AppendLine("    float ClearCoatRoughness;");
            }
            if (features.HasFlag(MaterialFeatureFlags.Sheen))
            {
                sb.AppendLine("    float Sheen;");
                sb.AppendLine("    float SheenRoughness;");
            }
            if (features.HasFlag(MaterialFeatureFlags.Transmission))
            {
                sb.AppendLine("    float Transmission;");
                sb.AppendLine("    float IOR;");
                sb.AppendLine("    float Thickness;");
            }
            sb.AppendLine("};");
            sb.AppendLine();

            foreach (var kvp in material.TextureSlots)
            {
                sb.AppendLine($"layout(set = 1, binding = {(int)kvp.Key}) uniform sampler2D _{kvp.Key};");
            }
            sb.AppendLine();

            sb.AppendLine("layout(location = 0) in vec3 inPositionWS;");
            sb.AppendLine("layout(location = 1) in vec3 inNormalWS;");
            sb.AppendLine("layout(location = 2) in vec4 inTangentWS;");
            sb.AppendLine("layout(location = 3) in vec2 inUV;");
            sb.AppendLine("layout(location = 4) in vec3 inViewDirWS;");
            sb.AppendLine("layout(location = 0) out vec4 fragColor;");
            sb.AppendLine();

            sb.AppendLine("const float PI = 3.14159265;");
            sb.AppendLine("const float EPSILON = 0.0001;");
            sb.AppendLine();

            sb.AppendLine("float D_GGX(float NdotH, float roughness) {");
            sb.AppendLine("    float a = roughness * roughness;");
            sb.AppendLine("    float a2 = a * a;");
            sb.AppendLine("    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;");
            sb.AppendLine("    return a2 / (PI * d * d);");
            sb.AppendLine("}");
            sb.AppendLine();

            sb.AppendLine("float V_SmithGGX(float NdotV, float NdotL, float roughness) {");
            sb.AppendLine("    float a = roughness * roughness;");
            sb.AppendLine("    float a2 = a * a;");
            sb.AppendLine("    float GGXV = NdotV * sqrt(NdotV * NdotV * (1.0 - a2) + a2);");
            sb.AppendLine("    float GGXL = NdotL * sqrt(NdotL * NdotL * (1.0 - a2) + a2);");
            sb.AppendLine("    return 0.5 / (GGXV + GGXL);");
            sb.AppendLine("}");
            sb.AppendLine();

            sb.AppendLine("vec3 F_Schlick(float cosTheta, vec3 F0) {");
            sb.AppendLine("    float f = 1.0 - cosTheta;");
            sb.AppendLine("    float f5 = f * f * f * f * f;");
            sb.AppendLine("    return F0 + (1.0 - F0) * f5;");
            sb.AppendLine("}");
            sb.AppendLine();

            sb.AppendLine("void main() {");
            sb.AppendLine("    vec3 N = normalize(inNormalWS);");
            sb.AppendLine("    vec3 V = normalize(inViewDirWS);");
            sb.AppendLine("    vec3 L = normalize(vec3(1, 1, 1));");
            sb.AppendLine("    vec3 H = normalize(V + L);");
            sb.AppendLine();
            sb.AppendLine("    vec3 baseColor = texture(_Albedo, inUV).rgb * BaseColor.rgb;");
            sb.AppendLine("    float roughness = max(texture(_Roughness, inUV).r * Roughness, 0.04);");
            sb.AppendLine("    float metallic = texture(_Metallic, inUV).r * Metallic;");
            sb.AppendLine("    float ao = texture(_AO, inUV).r * AOStrength;");
            sb.AppendLine();
            sb.AppendLine("    float NdotV = max(dot(N, V), 0.001);");
            sb.AppendLine("    float NdotL = max(dot(N, L), 0.0);");
            sb.AppendLine("    float NdotH = max(dot(N, H), 0.0);");
            sb.AppendLine("    float VdotH = max(dot(V, H), 0.0);");
            sb.AppendLine();
            sb.AppendLine("    vec3 diffuseColor = baseColor * (1.0 - metallic);");
            sb.AppendLine("    vec3 specularColor = mix(vec3(0.04), baseColor, metallic);");
            sb.AppendLine();
            sb.AppendLine("    float D = D_GGX(NdotH, roughness);");
            sb.AppendLine("    float G = V_SmithGGX(NdotV, NdotL, roughness);");
            sb.AppendLine("    vec3 F = F_Schlick(VdotH, specularColor);");
            sb.AppendLine();
            sb.AppendLine("    vec3 diffuse = diffuseColor / PI;");
            sb.AppendLine("    vec3 specular = F * D * G;");
            sb.AppendLine();
            sb.AppendLine("    vec3 color = (diffuse + specular) * NdotL * ao;");
            sb.AppendLine("    color += baseColor * texture(_Emissive, inUV).rgb * EmissiveIntensity;");
            sb.AppendLine("    fragColor = vec4(color, Opacity);");
            sb.AppendLine("}");

            return sb.ToString();
        }

        private string GenerateGLSLVertex(SubstrateMaterial material, MaterialFeatureFlags features)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// Auto-generated Substrate Material Vertex Shader");
            sb.AppendLine("#version 450");
            sb.AppendLine();
            sb.AppendLine("layout(location = 0) in vec3 inPosition;");
            sb.AppendLine("layout(location = 1) in vec3 inNormal;");
            sb.AppendLine("layout(location = 2) in vec4 inTangent;");
            sb.AppendLine("layout(location = 3) in vec2 inUV;");
            sb.AppendLine();
            sb.AppendLine("layout(location = 0) out vec3 outPositionWS;");
            sb.AppendLine("layout(location = 1) out vec3 outNormalWS;");
            sb.AppendLine("layout(location = 2) out vec4 outTangentWS;");
            sb.AppendLine("layout(location = 3) out vec2 outUV;");
            sb.AppendLine("layout(location = 4) out vec3 outViewDirWS;");
            sb.AppendLine();
            sb.AppendLine("layout(std140, binding = 0) uniform Camera { mat4 ViewProj; mat4 View; vec3 CameraPos; };");
            sb.AppendLine("layout(std140, binding = 1) uniform Model { mat4 ModelMatrix; mat3 NormalMatrix; };");
            sb.AppendLine();
            sb.AppendLine("void main() {");
            sb.AppendLine("    vec4 worldPos = ModelMatrix * vec4(inPosition, 1.0);");
            sb.AppendLine("    outPositionWS = worldPos.xyz;");
            sb.AppendLine("    outNormalWS = normalize(NormalMatrix * inNormal);");
            sb.AppendLine("    outTangentWS = vec4(normalize(NormalMatrix * inTangent.xyz), inTangent.w);");
            sb.AppendLine("    outUV = inUV;");
            sb.AppendLine("    outViewDirWS = CameraPos - worldPos.xyz;");
            sb.AppendLine("    gl_Position = ViewProj * worldPos;");
            sb.AppendLine("}");

            return sb.ToString();
        }

        private string GenerateSlang(SubstrateMaterial material, MaterialFeatureFlags features, ShaderStage stage)
        {
            if (stage == ShaderStage.Vertex) return GenerateSlangVertex(material, features);

            var sb = new StringBuilder();
            sb.AppendLine("// Auto-generated Substrate Material Slang Shader");
            sb.AppendLine("module SubstrateMaterial;");
            sb.AppendLine();
            sb.AppendLine("import render_pipeline;");
            sb.AppendLine("import bsdf;");
            sb.AppendLine();
            sb.AppendLine("cbuffer SubstrateParams : register(b0) {");
            sb.AppendLine("    float4 BaseColor;");
            sb.AppendLine("    float Roughness;");
            sb.AppendLine("    float Metallic;");
            sb.AppendLine("    float Specular;");
            sb.AppendLine("    float NormalStrength;");
            sb.AppendLine("    float AOStrength;");
            sb.AppendLine("    float EmissiveIntensity;");
            sb.AppendLine("    float Opacity;");

            if (features.HasFlag(MaterialFeatureFlags.ClearCoat))
            {
                sb.AppendLine("    float ClearCoat;");
                sb.AppendLine("    float ClearCoatRoughness;");
            }
            if (features.HasFlag(MaterialFeatureFlags.Sheen))
            {
                sb.AppendLine("    float Sheen;");
                sb.AppendLine("    float SheenRoughness;");
            }
            if (features.HasFlag(MaterialFeatureFlags.Transmission))
            {
                sb.AppendLine("    float Transmission;");
                sb.AppendLine("    float IOR;");
                sb.AppendLine("    float Thickness;");
            }
            sb.AppendLine("};");
            sb.AppendLine();

            foreach (var kvp in material.TextureSlots)
                sb.AppendLine($"Texture2D _{kvp.Key}; SamplerState sampler_{kvp.Key};");

            sb.AppendLine();
            sb.AppendLine("[shader(\"fragment\")]");
            sb.AppendLine("float4 main(Varyings input) : SV_Target {");
            sb.AppendLine("    float3 N = normalize(input.normalWS);");
            sb.AppendLine("    float3 V = normalize(input.viewDirWS);");
            sb.AppendLine("    float3 L = normalize(float3(1, 1, 1));");
            sb.AppendLine("    float3 H = normalize(V + L);");
            sb.AppendLine();
            sb.AppendLine("    float3 baseColor = _Albedo.Sample(sampler_Albedo, input.uv).rgb * BaseColor.rgb;");
            sb.AppendLine("    float roughness = max(_Roughness.Sample(sampler_Roughness, input.uv).r * Roughness, 0.04);");
            sb.AppendLine("    float metallic = _Metallic.Sample(sampler_Metallic, input.uv).r * Metallic;");
            sb.AppendLine();
            sb.AppendLine("    float NdotV = max(dot(N, V), 0.001);");
            sb.AppendLine("    float NdotL = max(dot(N, L), 0.0);");
            sb.AppendLine("    float NdotH = max(dot(N, H), 0.0);");
            sb.AppendLine("    float VdotH = max(dot(V, H), 0.0);");
            sb.AppendLine();
            sb.AppendLine("    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);");
            sb.AppendLine("    float3 diffuseColor = baseColor * (1.0 - metallic);");
            sb.AppendLine();
            sb.AppendLine("    float D = bsdf::D_GGX(NdotH, roughness);");
            sb.AppendLine("    float G = bsdf::V_SmithGGX(NdotV, NdotL, roughness);");
            sb.AppendLine("    float3 F = bsdf::F_Schlick(VdotH, F0);");
            sb.AppendLine();
            sb.AppendLine("    float3 color = (diffuseColor * INV_PI + F * D * G) * NdotL;");
            sb.AppendLine("    color += baseColor * _Emissive.Sample(sampler_Emissive, input.uv).rgb * EmissiveIntensity;");
            sb.AppendLine("    return float4(color, Opacity);");
            sb.AppendLine("}");

            return sb.ToString();
        }

        private string GenerateSlangVertex(SubstrateMaterial material, MaterialFeatureFlags features)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// Auto-generated Substrate Material Slang Vertex Shader");
            sb.AppendLine("module SubstrateMaterialVert;");
            sb.AppendLine("import render_pipeline;");
            sb.AppendLine();
            sb.AppendLine("[shader(\"vertex\")]");
            sb.AppendLine("Varyings main(Attributes input) {");
            sb.AppendLine("    Varyings output;");
            sb.AppendLine("    float4 worldPos = mul(ModelMatrix, float4(input.position, 1));");
            sb.AppendLine("    output.positionCS = mul(ViewProj, worldPos);");
            sb.AppendLine("    output.positionWS = worldPos.xyz;");
            sb.AppendLine("    output.normalWS = normalize(mul((float3x3)NormalMatrix, input.normal));");
            sb.AppendLine("    output.tangentWS = float4(normalize(mul((float3x3)NormalMatrix, input.tangent.xyz)), input.tangent.w);");
            sb.AppendLine("    output.uv = input.uv;");
            sb.AppendLine("    output.viewDirWS = CameraPos - worldPos.xyz;");
            sb.AppendLine("    return output;");
            sb.AppendLine("}");

            return sb.ToString();
        }

        private Dictionary<string, string> GenerateDefines(MaterialFeatureFlags features)
        {
            var defines = new Dictionary<string, string>();
            if (features.HasFlag(MaterialFeatureFlags.SubsurfaceScattering)) defines["SUBSURFACE_SCATTERING"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.ClearCoat)) defines["CLEAR_COAT"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.Anisotropy)) defines["ANISOTROPY"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.Sheen)) defines["SHEEN"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.Transmission)) defines["TRANSMISSION"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.Emissive)) defines["EMISSIVE"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.DiffuseTransmission)) defines["DIFFUSE_TRANSMISSION"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.ThinSurface)) defines["THIN_SURFACE"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.TwoSided)) defines["TWO_SIDED"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.WorldPositionOffset)) defines["WORLD_POSITION_OFFSET"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.ParallaxOcclusion)) defines["PARALLAX_OCCLUSION"] = "1";
            if (features.HasFlag(MaterialFeatureFlags.PixelDepthOffset)) defines["PIXEL_DEPTH_OFFSET"] = "1";
            return defines;
        }

        private static string ComputeShaderHash(string source)
        {
            using var sha = SHA256.Create();
            return Convert.ToBase64String(sha.ComputeHash(Encoding.UTF8.GetBytes(source)));
        }

        private static int EstimateInstructions(string source)
        {
            int count = 0;
            foreach (char c in source) if (c == ';' || c == '{' || c == '}') count++;
            count += source.Split('\n').Count(l => l.Contains("if") || l.Contains("for") || l.Contains("while")) * 3;
            count += source.Split('\n').Count(l => l.Contains("sin") || l.Contains("cos") || l.Contains("pow") || l.Contains("sqrt")) * 5;
            return count;
        }
    }

    // =========================================================================
    // PROCEDURAL MATERIAL GENERATOR
    // =========================================================================

    public class ProceduralMaterialGenerator
    {
        private readonly Random _rng;

        public ProceduralMaterialGenerator(int seed = 42) { _rng = new Random(seed); }

        public SubstrateMaterial GenerateWoodGrain(string name = "ProceduralWood", WeatheringState weathering = WeatheringState.Pristine)
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;
            float ageFactor = (float)weathering / 11.0f;

            float baseR = MathHelper.Lerp(0.45f, 0.25f, ageFactor);
            float baseG = MathHelper.Lerp(0.25f, 0.15f, ageFactor);
            float baseB = MathHelper.Lerp(0.10f, 0.06f, ageFactor);
            mat.SetProperty("BaseColor", new Color3(baseR, baseG, baseB));
            mat.SetProperty("Roughness", MathHelper.Lerp(0.55f, 0.75f, ageFactor));
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.35f);
            mat.SetProperty("NormalStrength", 1.2f);
            mat.SetProperty("AOStrength", 1.0f);

            if (weathering >= WeatheringState.ModeratelyWeathered)
            {
                mat.SetProperty("Sheen", MathHelper.Lerp(0f, 0.15f, ageFactor));
                mat.SetProperty("SheenRoughness", 0.7f);
            }
            if (weathering >= WeatheringState.Covered)
            {
                mat.SetProperty("SubsurfaceColor", new Color3(0.1f, 0.3f, 0.05f));
                mat.SetProperty("SubsurfaceRadius", new Vec3(0.5f, 0.2f, 0.1f));
            }

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://wood_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://wood_normal", TextureChannel.Normal));
            mat.SetTexture(TextureChannel.Roughness, new TextureReference("procedural://wood_roughness", TextureChannel.Roughness));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateMarble(string name = "ProceduralMarble", WeatheringState weathering = WeatheringState.Pristine)
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;
            float ageFactor = (float)weathering / 11.0f;

            float veinIntensity = MathHelper.Lerp(0.8f, 0.4f, ageFactor);
            mat.SetProperty("BaseColor", new Color3(veinIntensity, veinIntensity * 0.98f, veinIntensity * 0.96f));
            mat.SetProperty("Roughness", MathHelper.Lerp(0.15f, 0.4f, ageFactor));
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.5f);
            mat.SetProperty("NormalStrength", 0.8f);
            mat.SetProperty("AOStrength", 1.0f);

            if (weathering >= WeatheringState.LightlyWeathered)
            {
                mat.SetProperty("ClearCoat", MathHelper.Lerp(0.8f, 0.3f, ageFactor));
                mat.SetProperty("ClearCoatRoughness", MathHelper.Lerp(0.02f, 0.15f, ageFactor));
            }
            if (weathering >= WeatheringState.HeavilyWeathered)
            {
                mat.SetProperty("Transmission", 0.1f);
                mat.SetProperty("IOR", 1.65f);
                mat.SetProperty("Thickness", 0.3f);
            }

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://marble_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://marble_normal", TextureChannel.Normal));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateBrick(string name = "ProceduralBrick", WeatheringState weathering = WeatheringState.Pristine)
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;
            float ageFactor = (float)weathering / 11.0f;

            float r = MathHelper.Lerp(0.65f, 0.45f, ageFactor);
            float g = MathHelper.Lerp(0.25f, 0.18f, ageFactor);
            float b = MathHelper.Lerp(0.15f, 0.10f, ageFactor);
            mat.SetProperty("BaseColor", new Color3(r, g, b));
            mat.SetProperty("Roughness", MathHelper.Lerp(0.7f, 0.9f, ageFactor));
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.3f);
            mat.SetProperty("NormalStrength", 1.5f);
            mat.SetProperty("AOStrength", 0.8f);
            mat.SetProperty("DisplacementScale", 0.05f);

            if (weathering >= WeatheringState.Mossy)
            {
                mat.SetProperty("SubsurfaceColor", new Color3(0.1f, 0.35f, 0.05f));
                mat.SetProperty("SubsurfaceRadius", new Vec3(0.3f, 0.1f, 0.05f));
            }

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://brick_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://brick_normal", TextureChannel.Normal));
            mat.SetTexture(TextureChannel.Height, new TextureReference("procedural://brick_height", TextureChannel.Height));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateMetal(string name = "ProceduralMetal", string metalType = "polished", WeatheringState weathering = WeatheringState.Pristine)
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;
            float ageFactor = (float)weathering / 11.0f;

            Color3 metalTint = metalType.ToLower() switch
            {
                "gold" => new Color3(1.0f, 0.76f, 0.33f),
                "silver" => new Color3(0.95f, 0.93f, 0.88f),
                "copper" => new Color3(0.95f, 0.64f, 0.54f),
                "bronze" => new Color3(0.8f, 0.5f, 0.2f),
                "iron" => new Color3(0.56f, 0.57f, 0.58f),
                "aluminum" => new Color3(0.91f, 0.92f, 0.92f),
                "steel" => new Color3(0.7f, 0.72f, 0.73f),
                _ => new Color3(0.8f, 0.8f, 0.8f)
            };

            if (weathering >= WeatheringState.Corroded)
            {
                metalTint = Color3.Lerp(metalTint, new Color3(0.3f, 0.5f, 0.3f), ageFactor * 0.5f);
            }
            else if (weathering >= WeatheringState.Rusted)
            {
                metalTint = Color3.Lerp(metalTint, new Color3(0.6f, 0.25f, 0.1f), ageFactor * 0.7f);
            }

            mat.SetProperty("BaseColor", metalTint);
            mat.SetProperty("Metallic", 1.0f);

            float roughness = metalType.ToLower() switch
            {
                "brushed" => 0.35f,
                "polished" => 0.05f,
                "matte" => 0.6f,
                "mirror" => 0.01f,
                _ => 0.2f
            };
            mat.SetProperty("Roughness", MathHelper.Clamp(roughness + ageFactor * 0.3f, 0.01f, 1));
            mat.SetProperty("Specular", 0.5f);

            if (metalType.ToLower() == "brushed")
            {
                mat.SetProperty("Anisotropy", 0.8f);
                mat.SetProperty("AnisotropyRotation", 0f);
                mat.FeatureFlags = MaterialFeatureFlags.Anisotropy;
            }
            if (weathering >= WeatheringState.Corroded)
            {
                mat.SetProperty("NormalStrength", 2.0f);
                mat.SetProperty("AOStrength", 0.6f);
            }

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://metal_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://metal_normal", TextureChannel.Normal));
            mat.SetTexture(TextureChannel.Roughness, new TextureReference("procedural://metal_roughness", TextureChannel.Roughness));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateFabric(string name = "ProceduralFabric", string weaveType = "cotton")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Cloth;

            Color3 fabricColor = weaveType.ToLower() switch
            {
                "cotton" => new Color3(0.85f, 0.85f, 0.8f),
                "denim" => new Color3(0.15f, 0.2f, 0.4f),
                "silk" => new Color3(0.8f, 0.75f, 0.7f),
                "wool" => new Color3(0.6f, 0.55f, 0.5f),
                "leather" => new Color3(0.3f, 0.15f, 0.08f),
                "nylon" => new Color3(0.7f, 0.72f, 0.75f),
                "velvet" => new Color3(0.4f, 0.1f, 0.2f),
                _ => new Color3(0.7f, 0.7f, 0.7f)
            };

            mat.SetProperty("BaseColor", fabricColor);
            mat.SetProperty("Metallic", 0f);

            float roughness = weaveType.ToLower() switch
            {
                "silk" => 0.25f,
                "velvet" => 0.15f,
                "leather" => 0.5f,
                "denim" => 0.7f,
                _ => 0.6f
            };
            mat.SetProperty("Roughness", roughness);
            mat.SetProperty("Specular", 0.3f);

            float sheenVal = weaveType.ToLower() switch
            {
                "velvet" => 0.8f,
                "silk" => 0.4f,
                "wool" => 0.2f,
                _ => 0.1f
            };
            mat.SetProperty("Sheen", sheenVal);
            mat.SetProperty("SheenRoughness", weaveType.ToLower() == "velvet" ? 0.3f : 0.6f);
            mat.SetProperty("NormalStrength", 1.0f);

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://fabric_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://fabric_normal", TextureChannel.Normal));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateSkin(string name = "ProceduralSkin")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            mat.SetProperty("BaseColor", new Color3(0.76f, 0.55f, 0.42f));
            mat.SetProperty("Roughness", 0.45f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.4f);
            mat.SetProperty("SubsurfaceColor", new Color3(0.85f, 0.3f, 0.15f));
            mat.SetProperty("SubsurfaceRadius", new Vec3(1.2f, 0.35f, 0.15f));
            mat.SetProperty("NormalStrength", 0.5f);
            mat.SetProperty("AOStrength", 0.9f);
            mat.FeatureFlags = MaterialFeatureFlags.SubsurfaceScattering;

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://skin_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://skin_normal", TextureChannel.Normal));
            mat.SetTexture(TextureChannel.SubsurfaceColor, new TextureReference("procedural://skin_sss", TextureChannel.SubsurfaceColor));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateWater(string name = "ProceduralWater")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            mat.SetProperty("BaseColor", new Color3(0.02f, 0.08f, 0.15f));
            mat.SetProperty("Roughness", 0.02f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.5f);
            mat.SetProperty("Transmission", 0.9f);
            mat.SetProperty("IOR", 1.333f);
            mat.SetProperty("Thickness", 2.0f);
            mat.SetProperty("NormalStrength", 1.5f);
            mat.SetProperty("Opacity", 0.95f);
            mat.FeatureFlags = MaterialFeatureFlags.Transmission;

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://water_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://water_normal", TextureChannel.Normal));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateIce(string name = "ProceduralIce")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            mat.SetProperty("BaseColor", new Color3(0.75f, 0.85f, 0.95f));
            mat.SetProperty("Roughness", 0.05f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.5f);
            mat.SetProperty("Transmission", 0.7f);
            mat.SetProperty("IOR", 1.31f);
            mat.SetProperty("Thickness", 0.5f);
            mat.SetProperty("ClearCoat", 0.9f);
            mat.SetProperty("ClearCoatRoughness", 0.02f);
            mat.SetProperty("Opacity", 0.85f);
            mat.SetProperty("NormalStrength", 0.6f);
            mat.FeatureFlags = MaterialFeatureFlags.Transmission | MaterialFeatureFlags.ClearCoat;

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://ice_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://ice_normal", TextureChannel.Normal));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateLava(string name = "ProceduralLava")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface | MaterialDomain.Emissive;

            mat.SetProperty("BaseColor", new Color3(0.8f, 0.15f, 0.02f));
            mat.SetProperty("Roughness", 0.9f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.3f);
            mat.SetProperty("Emissive", new Color3(1.0f, 0.3f, 0.05f));
            mat.SetProperty("EmissiveIntensity", 10.0f);
            mat.SetProperty("NormalStrength", 2.0f);
            mat.SetProperty("AOStrength", 0.5f);
            mat.FeatureFlags = MaterialFeatureFlags.Emissive;

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://lava_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://lava_normal", TextureChannel.Normal));
            mat.SetTexture(TextureChannel.Emissive, new TextureReference("procedural://lava_emissive", TextureChannel.Emissive));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateBark(string name = "ProceduralBark", WeatheringState weathering = WeatheringState.Pristine)
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;
            float ageFactor = (float)weathering / 11.0f;

            mat.SetProperty("BaseColor", new Color3(MathHelper.Lerp(0.35f, 0.2f, ageFactor), MathHelper.Lerp(0.2f, 0.12f, ageFactor), MathHelper.Lerp(0.1f, 0.06f, ageFactor)));
            mat.SetProperty("Roughness", MathHelper.Lerp(0.8f, 0.95f, ageFactor));
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.25f);
            mat.SetProperty("NormalStrength", 2.0f);
            mat.SetProperty("AOStrength", 0.7f);
            mat.SetProperty("DisplacementScale", 0.1f);

            if (weathering >= WeatheringState.Mossy)
            {
                mat.SetProperty("SubsurfaceColor", new Color3(0.15f, 0.4f, 0.08f));
                mat.SetProperty("SubsurfaceRadius", new Vec3(0.3f, 0.1f, 0.05f));
                mat.FeatureFlags = MaterialFeatureFlags.SubsurfaceScattering;
            }

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://bark_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://bark_normal", TextureChannel.Normal));
            mat.SetTexture(TextureChannel.Height, new TextureReference("procedural://bark_height", TextureChannel.Height));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateRock(string name = "ProceduralRock", string rockType = "layered")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            Color3 rockColor = rockType.ToLower() switch
            {
                "sedimentary" => new Color3(0.6f, 0.55f, 0.45f),
                "igneous" => new Color3(0.35f, 0.32f, 0.3f),
                "metamorphic" => new Color3(0.45f, 0.4f, 0.38f),
                "granite" => new Color3(0.6f, 0.58f, 0.55f),
                "basalt" => new Color3(0.2f, 0.2f, 0.22f),
                "sandstone" => new Color3(0.75f, 0.6f, 0.4f),
                "limestone" => new Color3(0.7f, 0.68f, 0.6f),
                "slate" => new Color3(0.3f, 0.32f, 0.35f),
                "marble_rock" => new Color3(0.85f, 0.83f, 0.8f),
                _ => new Color3(0.5f, 0.48f, 0.45f)
            };

            mat.SetProperty("BaseColor", rockColor);
            mat.SetProperty("Roughness", 0.75f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.35f);
            mat.SetProperty("NormalStrength", 1.5f);
            mat.SetProperty("AOStrength", 0.8f);
            mat.SetProperty("DisplacementScale", 0.05f);

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://rock_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://rock_normal", TextureChannel.Normal));
            mat.SetTexture(TextureChannel.Height, new TextureReference("procedural://rock_height", TextureChannel.Height));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateSnow(string name = "ProceduralSnow")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            mat.SetProperty("BaseColor", new Color3(0.92f, 0.95f, 0.98f));
            mat.SetProperty("Roughness", 0.3f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.4f);
            mat.SetProperty("SubsurfaceColor", new Color3(0.7f, 0.8f, 0.9f));
            mat.SetProperty("SubsurfaceRadius", new Vec3(0.5f, 0.6f, 0.8f));
            mat.SetProperty("NormalStrength", 0.4f);
            mat.SetProperty("AOStrength", 0.9f);
            mat.FeatureFlags = MaterialFeatureFlags.SubsurfaceScattering;

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://snow_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://snow_normal", TextureChannel.Normal));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateGlass(string name = "ProceduralGlass", Color3 tint = default)
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            mat.SetProperty("BaseColor", tint.Equals(default(Color3)) ? new Color3(0.95f, 0.98f, 1.0f) : tint);
            mat.SetProperty("Roughness", 0.01f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.5f);
            mat.SetProperty("Transmission", 0.95f);
            mat.SetProperty("IOR", 1.52f);
            mat.SetProperty("Thickness", 0.3f);
            mat.SetProperty("ClearCoat", 1.0f);
            mat.SetProperty("ClearCoatRoughness", 0.01f);
            mat.SetProperty("Opacity", 0.1f);
            mat.FeatureFlags = MaterialFeatureFlags.Transmission | MaterialFeatureFlags.ClearCoat;

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://glass_albedo", TextureChannel.Albedo));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateCeramic(string name = "ProceduralCeramic")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            mat.SetProperty("BaseColor", new Color3(0.9f, 0.88f, 0.85f));
            mat.SetProperty("Roughness", 0.15f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.5f);
            mat.SetProperty("ClearCoat", 0.8f);
            mat.SetProperty("ClearCoatRoughness", 0.03f);
            mat.SetProperty("NormalStrength", 0.5f);
            mat.FeatureFlags = MaterialFeatureFlags.ClearCoat;

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://ceramic_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://ceramic_normal", TextureChannel.Normal));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GeneratePlastic(string name = "ProceduralPlastic", Color3 tint = default)
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            mat.SetProperty("BaseColor", tint.Equals(default(Color3)) ? new Color3(0.8f, 0.1f, 0.1f) : tint);
            mat.SetProperty("Roughness", 0.4f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.5f);
            mat.SetProperty("NormalStrength", 0.3f);

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://plastic_albedo", TextureChannel.Albedo));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateSand(string name = "ProceduralSand")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            mat.SetProperty("BaseColor", new Color3(0.82f, 0.72f, 0.55f));
            mat.SetProperty("Roughness", 0.85f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.25f);
            mat.SetProperty("NormalStrength", 1.0f);
            mat.SetProperty("AOStrength", 0.7f);

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://sand_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://sand_normal", TextureChannel.Normal));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateRubber(string name = "ProceduralRubber")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            mat.SetProperty("BaseColor", new Color3(0.1f, 0.1f, 0.1f));
            mat.SetProperty("Roughness", 0.85f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.3f);
            mat.SetProperty("NormalStrength", 0.5f);

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://rubber_albedo", TextureChannel.Albedo));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateConcrete(string name = "ProceduralConcrete", WeatheringState weathering = WeatheringState.Pristine)
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;
            float ageFactor = (float)weathering / 11.0f;

            float v = MathHelper.Lerp(0.6f, 0.45f, ageFactor);
            mat.SetProperty("BaseColor", new Color3(v, v * 0.98f, v * 0.95f));
            mat.SetProperty("Roughness", MathHelper.Lerp(0.7f, 0.9f, ageFactor));
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.3f);
            mat.SetProperty("NormalStrength", 1.5f);
            mat.SetProperty("AOStrength", 0.8f);

            if (weathering >= WeatheringState.Corroded)
            {
                mat.SetProperty("SubsurfaceColor", new Color3(0.1f, 0.15f, 0.08f));
                mat.FeatureFlags = MaterialFeatureFlags.SubsurfaceScattering;
            }

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://concrete_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://concrete_normal", TextureChannel.Normal));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GeneratePaper(string name = "ProceduralPaper")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Surface;

            mat.SetProperty("BaseColor", new Color3(0.95f, 0.93f, 0.88f));
            mat.SetProperty("Roughness", 0.65f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.25f);
            mat.SetProperty("NormalStrength", 0.3f);
            mat.SetProperty("Thickness", 0.01f);

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://paper_albedo", TextureChannel.Albedo));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateEye(string name = "ProceduralEye")
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Eye;

            mat.SetProperty("BaseColor", new Color3(0.35f, 0.55f, 0.8f));
            mat.SetProperty("Roughness", 0.05f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.5f);
            mat.SetProperty("IOR", 1.376f);
            mat.SetProperty("ClearCoat", 1.0f);
            mat.SetProperty("ClearCoatRoughness", 0.01f);
            mat.FeatureFlags = MaterialFeatureFlags.EyeBRDF | MaterialFeatureFlags.ClearCoat;

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://eye_albedo", TextureChannel.Albedo));
            mat.SetTexture(TextureChannel.Normal, new TextureReference("procedural://eye_normal", TextureChannel.Normal));
            mat.ComputeActiveFeatures();
            return mat;
        }

        public SubstrateMaterial GenerateHair(string name = "ProceduralHair", Color3 hairColor = default)
        {
            var mat = new SubstrateMaterial(name);
            mat.Domain = MaterialDomain.Hair;

            mat.SetProperty("BaseColor", hairColor.Equals(default(Color3)) ? new Color3(0.3f, 0.15f, 0.05f) : hairColor);
            mat.SetProperty("Roughness", 0.35f);
            mat.SetProperty("Metallic", 0f);
            mat.SetProperty("Specular", 0.5f);
            mat.SetProperty("IOR", 1.55f);
            mat.FeatureFlags = MaterialFeatureFlags.HairBRDF;

            mat.SetTexture(TextureChannel.Albedo, new TextureReference("procedural://hair_albedo", TextureChannel.Albedo));
            mat.ComputeActiveFeatures();
            return mat;
        }
    }

    // =========================================================================
    // MATERIAL PRESETS LIBRARY
    // =========================================================================

    public class MaterialPresetsLibrary
    {
        private readonly Dictionary<string, SubstrateMaterial> _presets;
        private readonly Dictionary<string, MaterialCategory> _categories;
        private readonly Dictionary<string, HashSet<string>> _tags;

        public int PresetCount => _presets.Count;

        public MaterialPresetsLibrary()
        {
            _presets = new Dictionary<string, SubstrateMaterial>();
            _categories = new Dictionary<string, MaterialCategory>();
            _tags = new Dictionary<string, HashSet<string>>();
            RegisterBuiltInPresets();
        }

        public SubstrateMaterial GetPreset(string name)
        {
            _presets.TryGetValue(name, out var preset);
            return preset?.Clone(name);
        }

        public bool HasPreset(string name) => _presets.ContainsKey(name);

        public IReadOnlyCollection<string> GetAllPresetNames() => _presets.Keys.ToList().AsReadOnly();

        public List<string> GetPresetsByCategory(MaterialCategory category)
        {
            return _categories.Where(kvp => kvp.Value == category).Select(kvp => kvp.Key).ToList();
        }

        public List<string> GetPresetsByTag(string tag)
        {
            return _tags.Where(kvp => kvp.Value.Contains(tag)).Select(kvp => kvp.Key).ToList();
        }

        public List<string> Search(string query)
        {
            string q = query.ToLower();
            return _presets.Keys.Where(k => k.ToLower().Contains(q) ||
                (_tags.ContainsKey(k) && _tags[k].Any(t => t.ToLower().Contains(q)))).ToList();
        }

        public void RegisterPreset(string name, SubstrateMaterial material, MaterialCategory category, params string[] tags)
        {
            _presets[name] = material;
            _categories[name] = category;
            _tags[name] = new HashSet<string>(tags);
        }

        private void RegisterBuiltInPresets()
        {
            var gen = new ProceduralMaterialGenerator();

            RegisterPreset("Metal_Gold", gen.GenerateMetal("Gold", "gold"), MaterialCategory.Metal, "gold", "reflective", "precious");
            RegisterPreset("Metal_Silver", gen.GenerateMetal("Silver", "silver"), MaterialCategory.Metal, "silver", "reflective", "precious");
            RegisterPreset("Metal_Copper", gen.GenerateMetal("Copper", "copper"), MaterialCategory.Metal, "copper", "reflective");
            RegisterPreset("Metal_Bronze", gen.GenerateMetal("Bronze", "bronze"), MaterialCategory.Metal, "bronze", "reflective");
            RegisterPreset("Metal_Iron", gen.GenerateMetal("Iron", "iron"), MaterialCategory.Metal, "iron", "reflective");
            RegisterPreset("Metal_Aluminum", gen.GenerateMetal("Aluminum", "aluminum"), MaterialCategory.Metal, "aluminum", "reflective");
            RegisterPreset("Metal_Steel", gen.GenerateMetal("Steel", "steel"), MaterialCategory.Metal, "steel", "reflective");
            RegisterPreset("Metal_Brushed", gen.GenerateMetal("BrushedMetal", "brushed"), MaterialCategory.Metal, "brushed", "anisotropic");
            RegisterPreset("Metal_Polished", gen.GenerateMetal("PolishedMetal", "polished"), MaterialCategory.Metal, "polished", "mirror");
            RegisterPreset("Metal_Rusted", gen.GenerateMetal("RustedMetal", "iron", WeatheringState.Rusted), MaterialCategory.Metal, "rusted", "weathered");
            RegisterPreset("Metal_Corroded", gen.GenerateMetal("CorrodedMetal", "copper", WeatheringState.Corroded), MaterialCategory.Metal, "corroded", "weathered", "patina");

            RegisterPreset("Wood_Oak", gen.GenerateWoodGrain("OakWood"), MaterialCategory.Wood, "oak", "natural");
            RegisterPreset("Wood_Walnut", gen.GenerateWoodGrain("WalnutWood"), MaterialCategory.Wood, "walnut", "natural");
            RegisterPreset("Wood_Old", gen.GenerateWoodGrain("OldWood", WeatheringState.HeavilyWeathered), MaterialCategory.Wood, "old", "weathered");
            RegisterPreset("Wood_Mossy", gen.GenerateWoodGrain("MossyWood", WeatheringState.Mossy), MaterialCategory.Wood, "mossy", "organic");

            RegisterPreset("Stone_Granite", gen.GenerateRock("Granite", "granite"), MaterialCategory.Stone, "granite", "natural");
            RegisterPreset("Stone_Basalt", gen.GenerateRock("Basalt", "basalt"), MaterialCategory.Stone, "basalt", "igneous");
            RegisterPreset("Stone_Sandstone", gen.GenerateRock("Sandstone", "sedimentary"), MaterialCategory.Stone, "sandstone", "sedimentary");
            RegisterPreset("Stone_Slate", gen.GenerateRock("Slate", "slate"), MaterialCategory.Stone, "slate", "metamorphic");
            RegisterPreset("Stone_Marble", gen.GenerateMarble("MarbleStone"), MaterialCategory.Stone, "marble", "metamorphic");
            RegisterPreset("Stone_Limestone", gen.GenerateRock("Limestone", "limestone"), MaterialCategory.Stone, "limestone", "sedimentary");

            RegisterPreset("Skin_Caucasian", gen.GenerateSkin("CaucasianSkin"), MaterialCategory.Skin, "skin", "organic", "sss");
            RegisterPreset("Fabric_Cotton", gen.GenerateFabric("Cotton", "cotton"), MaterialCategory.Fabric, "cotton", "natural");
            RegisterPreset("Fabric_Denim", gen.GenerateFabric("Denim", "denim"), MaterialCategory.Fabric, "denim", "natural");
            RegisterPreset("Fabric_Silk", gen.GenerateFabric("Silk", "silk"), MaterialCategory.Fabric, "silk", "luxury");
            RegisterPreset("Fabric_Velvet", gen.GenerateFabric("Velvet", "velvet"), MaterialCategory.Fabric, "velvet", "luxury");
            RegisterPreset("Fabric_Leather", gen.GenerateFabric("Leather", "leather"), MaterialCategory.Fabric, "leather", "natural");
            RegisterPreset("Fabric_Wool", gen.GenerateFabric("Wool", "wool"), MaterialCategory.Fabric, "wool", "natural");

            RegisterPreset("Glass_Clear", gen.GenerateGlass("ClearGlass"), MaterialCategory.Glass, "glass", "transparent");
            RegisterPreset("Glass_Tinted", gen.GenerateGlass("TintedGlass", new Color3(0.7f, 0.9f, 0.8f)), MaterialCategory.Glass, "glass", "tinted");

            RegisterPreset("Water_Clear", gen.GenerateWater("ClearWater"), MaterialCategory.Liquid, "water", "transparent");
            RegisterPreset("Ice_Clear", gen.GenerateIce("ClearIce"), MaterialCategory.Ice, "ice", "transparent", "frozen");

            RegisterPreset("Ceramic_White", gen.GenerateCeramic("WhiteCeramic"), MaterialCategory.Ceramic, "ceramic", "glazed");
            RegisterPreset("Plastic_Red", gen.GeneratePlastic("RedPlastic", new Color3(0.8f, 0.1f, 0.1f)), MaterialCategory.Plastic, "plastic");
            RegisterPreset("Rubber_Black", gen.GenerateRubber("BlackRubber"), MaterialCategory.Rubber, "rubber");
            RegisterPreset("Paper_White", gen.GeneratePaper("WhitePaper"), MaterialCategory.Paper, "paper");
            RegisterPreset("Concrete_Gray", gen.GenerateConcrete("GrayConcrete"), MaterialCategory.Concrete, "concrete");
            RegisterPreset("Concrete_Weathered", gen.GenerateConcrete("WeatheredConcrete", WeatheringState.ModeratelyWeathered), MaterialCategory.Concrete, "concrete", "weathered");
            RegisterPreset("Sand_Desert", gen.GenerateSand("DesertSand"), MaterialCategory.Sand, "sand", "desert");
            RegisterPreset("Snow_Fresh", gen.GenerateSnow("FreshSnow"), MaterialCategory.Snow, "snow", "frozen");
            RegisterPreset("Lava_Molten", gen.GenerateLava("MoltenLava"), MaterialCategory.Geological, "lava", "emissive", "molten");
            RegisterPreset("Bark_Oak", gen.GenerateBark("OakBark"), MaterialCategory.Wood, "bark", "natural");
            RegisterPreset("Eye_Blue", gen.GenerateEye("BlueEye"), MaterialCategory.Organic, "eye", "organic");
            RegisterPreset("Hair_Brown", gen.GenerateHair("BrownHair", new Color3(0.3f, 0.15f, 0.05f)), MaterialCategory.Organic, "hair", "organic");
        }
    }

    // =========================================================================
    // MATERIAL INSPECTOR
    // =========================================================================

    public class MaterialInspector
    {
        private readonly BsdfEvaluator _bsdf;
        private readonly MaterialEvaluator _evaluator;

        public MaterialInspector()
        {
            _bsdf = new BsdfEvaluator();
            _evaluator = new MaterialEvaluator();
        }

        public MaterialAnalyticsData Analyze(SubstrateMaterial material)
        {
            var features = material.ComputeActiveFeatures();
            int textureMem = 0;
            foreach (var kvp in material.TextureSlots)
                if (kvp.Value != null && !string.IsNullOrEmpty(kvp.Value.Path))
                    textureMem += 1024 * 1024; // Estimate 1MB per texture

            int permCount = ComputePermutationCount(features);
            float complexity = material.ComputeComplexityScore();
            int gpuCost = EstimateGPUCost(material);
            int drawCalls = EstimateDrawCalls(material);

            return new MaterialAnalyticsData
            {
                PropertyCount = material.Properties.Count,
                TextureCount = material.TextureSlots.Count,
                TextureMemoryBytes = textureMem,
                ShaderPermutationCount = permCount,
                EstimatedGPUCost = gpuCost,
                EstimatedDrawCalls = drawCalls,
                ActiveFeatures = features,
                ComplexityScore = complexity
            };
        }

        public List<string> Validate(SubstrateMaterial material)
        {
            var issues = new List<string>();

            if (material.GetFloat("Roughness") < 0.01f)
                issues.Add("Roughness is very low, may cause firefly artifacts.");
            if (material.GetFloat("Roughness") > 0.99f)
                issues.Add("Roughness is very high, near-diffuse surface.");
            if (material.GetFloat("Metallic") > 0.9f && material.GetFloat("Roughness") < 0.05f)
                issues.Add("High metallic with very low roughness may need specular anti-aliasing.");
            if (material.GetFloat("IOR") < 1.0f)
                issues.Add("IOR < 1.0 is physically invalid for most materials.");
            if (material.GetFloat("IOR") > 2.5f)
                issues.Add("IOR > 2.5 is uncommon, verify correctness.");
            if (material.GetFloat("Transmission") > 0 && material.GetFloat("Opacity") >= 1.0f)
                issues.Add("Transmission set but opacity is 1.0, may need transparency.");
            if (material.GetFloat("ClearCoat") > 0 && material.GetFloat("ClearCoatRoughness") > 0.5f)
                issues.Add("Clear coat roughness is high, may not be visible.");

            var missing = material.GetMissingTextures();
            if (missing.Count > 0)
                issues.Add($"Missing textures: {string.Join(", ", missing)}");

            if (material.TextureSlots.Count > 16)
                issues.Add($"High texture count ({material.TextureSlots.Count}), may impact performance.");

            return issues;
        }

        public float EstimateRenderingCost(SubstrateMaterial material)
        {
            float cost = 1.0f;
            cost += material.TextureSlots.Count * 0.5f;
            cost += material.Layers.Count * 0.3f;
            var features = material.ComputeActiveFeatures();
            if (features.HasFlag(MaterialFeatureFlags.SubsurfaceScattering)) cost += 3.0f;
            if (features.HasFlag(MaterialFeatureFlags.ClearCoat)) cost += 1.5f;
            if (features.HasFlag(MaterialFeatureFlags.Anisotropy)) cost += 1.0f;
            if (features.HasFlag(MaterialFeatureFlags.Sheen)) cost += 1.0f;
            if (features.HasFlag(MaterialFeatureFlags.Transmission)) cost += 2.5f;
            if (features.HasFlag(MaterialFeatureFlags.HairBRDF)) cost += 4.0f;
            if (features.HasFlag(MaterialFeatureFlags.EyeBRDF)) cost += 4.0f;
            if (features.HasFlag(MaterialFeatureFlags.ParallaxOcclusion)) cost += 2.0f;
            if (features.HasFlag(MaterialFeatureFlags.PixelDepthOffset)) cost += 0.5f;
            return cost;
        }

        public Dictionary<string, float> ProfilePropertyRanges(SubstrateMaterial material, int sampleCount = 100)
        {
            var ranges = new Dictionary<string, float>();
            foreach (var kvp in material.Properties)
            {
                if (kvp.Value.Type == MaterialPropertyType.Float)
                    ranges[kvp.Key] = kvp.Value.Max - kvp.Value.Min;
                else if (kvp.Value.Type == MaterialPropertyType.Color)
                    ranges[kvp.Key] = kvp.Value.AsColor().Luminance;
            }
            return ranges;
        }

        public Dictionary<string, float> ComputePropertySensitivity(SubstrateMaterial material, string propertyName, float delta = 0.01f)
        {
            var sensitivity = new Dictionary<string, float>();
            var original = material.GetFloat(propertyName);
            var baseResult = _evaluator.EvaluateFullShading(material, new Vec3(0, 1, 0), new Vec3(0, 0, 1), Vec3.Zero, Vec3.UnitZ, Vec3.UnitX, 1.0f);

            material.SetProperty(propertyName, original + delta);
            var perturbed = _evaluator.EvaluateFullShading(material, new Vec3(0, 1, 0), new Vec3(0, 0, 1), Vec3.Zero, Vec3.UnitZ, Vec3.UnitX, 1.0f);
            material.SetProperty(propertyName, original);

            float diff = (perturbed - baseResult).ToVec3().Length;
            sensitivity[propertyName] = diff / delta;
            return sensitivity;
        }

        public string GenerateReport(SubstrateMaterial material)
        {
            var sb = new StringBuilder();
            var analytics = Analyze(material);
            var issues = Validate(material);

            sb.AppendLine($"=== Material Report: {material.Name} ===");
            sb.AppendLine($"ID: {material.Id}");
            sb.AppendLine($"Version: {material.Version}");
            sb.AppendLine($"Domain: {material.Domain}");
            sb.AppendLine($"Features: {analytics.ActiveFeatures}");
            sb.AppendLine($"Properties: {analytics.PropertyCount}");
            sb.AppendLine($"Textures: {analytics.TextureCount}");
            sb.AppendLine($"Texture Memory: {analytics.TextureMemoryBytes / 1024} KB");
            sb.AppendLine($"Shader Permutations: {analytics.ShaderPermutationCount}");
            sb.AppendLine($"GPU Cost: {analytics.EstimatedGPUCost:F1}");
            sb.AppendLine($"Draw Calls: {analytics.EstimatedDrawCalls}");
            sb.AppendLine($"Complexity: {analytics.ComplexityScore:F2}");
            sb.AppendLine($"Render Cost: {EstimateRenderingCost(material):F1}");

            if (issues.Count > 0)
            {
                sb.AppendLine("\nIssues:");
                foreach (var issue in issues) sb.AppendLine($"  - {issue}");
            }
            else
            {
                sb.AppendLine("\nNo issues found.");
            }

            return sb.ToString();
        }

        private int ComputePermutationCount(MaterialFeatureFlags features)
        {
            int count = 1;
            var featureBits = new[]
            {
                MaterialFeatureFlags.SubsurfaceScattering, MaterialFeatureFlags.ClearCoat,
                MaterialFeatureFlags.Anisotropy, MaterialFeatureFlags.Sheen,
                MaterialFeatureFlags.Transmission, MaterialFeatureFlags.Emissive,
                MaterialFeatureFlags.DiffuseTransmission, MaterialFeatureFlags.ThinSurface
            };
            foreach (var f in featureBits)
                if (features.HasFlag(f)) count *= 2;
            return count;
        }

        private int EstimateGPUCost(SubstrateMaterial material)
        {
            int cost = 10; // Base cost
            cost += material.TextureSlots.Count * 5;
            cost += material.Layers.Count * 3;
            var features = material.ComputeActiveFeatures();
            if (features.HasFlag(MaterialFeatureFlags.SubsurfaceScattering)) cost += 20;
            if (features.HasFlag(MaterialFeatureFlags.ClearCoat)) cost += 10;
            if (features.HasFlag(MaterialFeatureFlags.Transmission)) cost += 15;
            if (features.HasFlag(MaterialFeatureFlags.HairBRDF)) cost += 25;
            if (features.HasFlag(MaterialFeatureFlags.EyeBRDF)) cost += 25;
            return cost;
        }

        private int EstimateDrawCalls(SubstrateMaterial material)
        {
            int calls = 1;
            if (material.Layers.Count > 1) calls += material.Layers.Count;
            if (material.FeatureFlags.HasFlag(MaterialFeatureFlags.Decal)) calls++;
            return calls;
        }
    }

    // =========================================================================
    // MATERIAL VARIANT SYSTEM
    // =========================================================================

    public class MaterialVariantSystem
    {
        private readonly Random _rng;

        public MaterialVariantSystem(int seed = 42) { _rng = new Random(seed); }

        public SubstrateMaterial CreateWeatheringVariant(SubstrateMaterial baseMaterial, WeatheringState targetWeathering, string variantName = null)
        {
            float ageFactor = (float)targetWeathering / 11.0f;
            var variant = baseMaterial.Clone(variantName ?? $"{baseMaterial.Name}_{targetWeathering}");

            Color3 originalColor = variant.GetColor("BaseColor");
            Color3 weatheredColor = Color3.Lerp(originalColor, originalColor * 0.6f, ageFactor);
            variant.SetProperty("BaseColor", weatheredColor);

            float originalRoughness = variant.GetFloat("Roughness");
            variant.SetProperty("Roughness", MathHelper.Clamp(originalRoughness + ageFactor * 0.3f, 0, 1));

            if (targetWeathering >= WeatheringState.Mossy)
            {
                variant.SetProperty("SubsurfaceColor", new Color3(0.1f, 0.35f, 0.05f));
                variant.SetProperty("SubsurfaceRadius", new Vec3(0.3f, 0.1f, 0.05f));
            }
            if (targetWeathering >= WeatheringState.Rusted)
            {
                variant.SetProperty("BaseColor", Color3.Lerp(originalColor, new Color3(0.6f, 0.25f, 0.1f), ageFactor * 0.7f));
                variant.SetProperty("NormalStrength", variant.GetFloat("NormalStrength") * 1.5f);
            }
            if (targetWeathering >= WeatheringState.Frozen)
            {
                variant.SetProperty("ClearCoat", 0.5f);
                variant.SetProperty("ClearCoatRoughness", 0.05f);
            }

            variant.ComputeActiveFeatures();
            return variant;
        }

        public SubstrateMaterial CreateScaleVariant(SubstrateMaterial baseMaterial, float scaleMultiplier, string variantName = null)
        {
            var variant = baseMaterial.Clone(variantName ?? $"{baseMaterial.Name}_Scale{scaleMultiplier:F1}");

            foreach (var kvp in baseMaterial.TextureSlots)
            {
                var tex = kvp.Value;
                if (tex != null)
                {
                    variant.SetTexture(kvp.Key, new TextureReference(tex.Path, kvp.Key)
                    {
                        Tiling = tex.Tiling * scaleMultiplier,
                        Sampler = tex.Sampler,
                        WrapMode = tex.WrapMode,
                        Rotation = tex.Rotation
                    });
                }
            }

            float dispScale = variant.GetFloat("DisplacementScale");
            variant.SetProperty("DisplacementScale", dispScale / scaleMultiplier);

            variant.ComputeActiveFeatures();
            return variant;
        }

        public SubstrateMaterial CreateColorVariant(SubstrateMaterial baseMaterial, Color3 colorShift, string variantName = null)
        {
            var variant = baseMaterial.Clone(variantName ?? $"{baseMaterial.Name}_ColorShift");

            Color3 original = variant.GetColor("BaseColor");
            variant.SetProperty("BaseColor", original * colorShift);

            Color3 sss = variant.GetColor("SubsurfaceColor");
            if (!sss.Equals(default(Color3))) variant.SetProperty("SubsurfaceColor", sss * colorShift);

            Color3 emissive = variant.GetColor("Emissive");
            if (!emissive.Equals(default(Color3))) variant.SetProperty("Emissive", emissive * colorShift);

            variant.ComputeActiveFeatures();
            return variant;
        }

        public List<SubstrateMaterial> CreateBatchVariants(SubstrateMaterial baseMaterial, int count, VariantGenerationParams parameters)
        {
            var variants = new List<SubstrateMaterial>();
            for (int i = 0; i < count; i++)
            {
                string name = $"{baseMaterial.Name}_Variant_{i}";
                var variant = baseMaterial.Clone(name);

                if (parameters.RandomizeColor)
                {
                    float h = (float)_rng.NextDouble() * 360;
                    float s = MathHelper.Lerp(parameters.MinSaturation, parameters.MaxSaturation, (float)_rng.NextDouble());
                    float v = MathHelper.Lerp(parameters.MinValue, parameters.MaxValue, (float)_rng.NextDouble());
                    Color3 hsv = HSVToRGB(h, s, v);
                    variant.SetProperty("BaseColor", hsv);
                }

                if (parameters.RandomizeRoughness)
                    variant.SetProperty("Roughness", MathHelper.Lerp(parameters.MinRoughness, parameters.MaxRoughness, (float)_rng.NextDouble()));

                if (parameters.RandomizeMetallic)
                    variant.SetProperty("Metallic", MathHelper.Lerp(parameters.MinMetallic, parameters.MaxMetallic, (float)_rng.NextDouble()));

                if (parameters.RandomizeNormalStrength)
                    variant.SetProperty("NormalStrength", MathHelper.Lerp(0.5f, 2.0f, (float)_rng.NextDouble()));

                if (parameters.RandomizeScale)
                {
                    float scale = MathHelper.Lerp(parameters.MinScale, parameters.MaxScale, (float)_rng.NextDouble());
                    foreach (var kvp in variant.TextureSlots)
                    {
                        if (kvp.Value != null)
                            variant.SetTexture(kvp.Key, new TextureReference(kvp.Value.Path, kvp.Key)
                            {
                                Tiling = kvp.Value.Tiling * scale,
                                Sampler = kvp.Value.Sampler
                            });
                    }
                }

                if (parameters.WeatheringRange.HasValue)
                {
                    var weatherings = Enum.GetValues<WeatheringState>();
                    int minIdx = (int)(parameters.WeatheringRange.Value.Item1);
                    int maxIdx = (int)(parameters.WeatheringRange.Value.Item2);
                    var target = (WeatheringState)_rng.Next(minIdx, maxIdx + 1);
                    variant = CreateWeatheringVariant(variant, target, name);
                }

                variant.ComputeActiveFeatures();
                variants.Add(variant);
            }
            return variants;
        }

        private static Color3 HSVToRGB(float h, float s, float v)
        {
            float c = v * s;
            float x = c * (1 - Math.Abs((h / 60) % 2 - 1));
            float m = v - c;
            float r, g, b;
            if (h < 60) { r = c; g = x; b = 0; }
            else if (h < 120) { r = x; g = c; b = 0; }
            else if (h < 180) { r = 0; g = c; b = x; }
            else if (h < 240) { r = 0; g = x; b = c; }
            else if (h < 300) { r = x; g = 0; b = c; }
            else { r = c; g = 0; b = x; }
            return new Color3(r + m, g + m, b + m);
        }
    }

    public class VariantGenerationParams
    {
        public bool RandomizeColor { get; set; }
        public bool RandomizeRoughness { get; set; }
        public bool RandomizeMetallic { get; set; }
        public bool RandomizeNormalStrength { get; set; }
        public bool RandomizeScale { get; set; }
        public float MinSaturation { get; set; } = 0.5f;
        public float MaxSaturation { get; set; } = 1.0f;
        public float MinValue { get; set; } = 0.5f;
        public float MaxValue { get; set; } = 1.0f;
        public float MinRoughness { get; set; } = 0.1f;
        public float MaxRoughness { get; set; } = 0.9f;
        public float MinMetallic { get; set; } = 0f;
        public float MaxMetallic { get; set; } = 1f;
        public float MinScale { get; set; } = 0.5f;
        public float MaxScale { get; set; } = 2.0f;
        public (WeatheringState, WeatheringState)? WeatheringRange { get; set; }
    }

    // =========================================================================
    // MATERIAL BLEND SYSTEM
    // =========================================================================

    public class MaterialBlendSystem
    {
        private readonly BsdfEvaluator _bsdf;

        public MaterialBlendSystem() { _bsdf = new BsdfEvaluator(); }

        public BlendedMaterialResult HeightBlend(SubstrateMaterial matA, SubstrateMaterial matB, Vec2 uv, float heightA, float heightB, float sharpness = 10.0f)
        {
            float blend = MathHelper.Saturate((heightB - heightA) * sharpness);
            return new BlendedMaterialResult
            {
                BaseColor = Color3.Lerp(matA.GetColor("BaseColor"), matB.GetColor("BaseColor"), blend),
                Roughness = MathHelper.Lerp(matA.GetFloat("Roughness"), matB.GetFloat("Roughness"), blend),
                Metallic = MathHelper.Lerp(matA.GetFloat("Metallic"), matB.GetFloat("Metallic"), blend),
                Normal = Vec3.Lerp(matA.GetVec3("Normal", Vec3.UnitZ), matB.GetVec3("Normal", Vec3.UnitZ), blend).Normalized,
                AO = MathHelper.Lerp(matA.GetFloat("AO", 1f), matB.GetFloat("AO", 1f), blend),
                Opacity = MathHelper.Lerp(matA.GetFloat("Opacity", 1f), matB.GetFloat("Opacity", 1f), blend)
            };
        }

        public BlendedMaterialResult DistanceBlend(SubstrateMaterial matA, SubstrateMaterial matB, Vec3 position, Vec3 centerA, Vec3 centerB, float falloff = 1.0f)
        {
            float distA = (position - centerA).Length;
            float distB = (position - centerB).Length;
            float totalDist = distA + distB;
            float blend = totalDist > MathHelper.EPSILON ? MathHelper.Saturate(MathHelper.Pow(distA / totalDist, falloff)) : 0.5f;

            return new BlendedMaterialResult
            {
                BaseColor = Color3.Lerp(matA.GetColor("BaseColor"), matB.GetColor("BaseColor"), blend),
                Roughness = MathHelper.Lerp(matA.GetFloat("Roughness"), matB.GetFloat("Roughness"), blend),
                Metallic = MathHelper.Lerp(matA.GetFloat("Metallic"), matB.GetFloat("Metallic"), blend),
                AO = MathHelper.Lerp(matA.GetFloat("AO", 1f), matB.GetFloat("AO", 1f), blend)
            };
        }

        public BlendedMaterialResult SlopeBlend(SubstrateMaterial matFlat, SubstrateMaterial matSteep, Vec3 normal, float slopeThreshold = 0.7f, float blendWidth = 0.1f)
        {
            float slope = 1.0f - normal.Dot(Vec3.UnitY);
            float blend = MathHelper.Saturate((slope - slopeThreshold) / blendWidth);

            return new BlendedMaterialResult
            {
                BaseColor = Color3.Lerp(matFlat.GetColor("BaseColor"), matSteep.GetColor("BaseColor"), blend),
                Roughness = MathHelper.Lerp(matFlat.GetFloat("Roughness"), matSteep.GetFloat("Roughness"), blend),
                Metallic = MathHelper.Lerp(matFlat.GetFloat("Metallic"), matSteep.GetFloat("Metallic"), blend),
                AO = MathHelper.Lerp(matFlat.GetFloat("AO", 1f), matSteep.GetFloat("AO", 1f), blend),
                Normal = Vec3.Lerp(matFlat.GetVec3("Normal", Vec3.UnitZ), matSteep.GetVec3("Normal", Vec3.UnitZ), blend).Normalized
            };
        }

        public BlendedMaterialResult TextureSplat(SubstrateMaterial[] materials, float[] weights, Vec2 uv)
        {
            if (materials.Length == 0) return new BlendedMaterialResult();
            if (materials.Length == 1) return new BlendedMaterialResult
            {
                BaseColor = materials[0].GetColor("BaseColor"),
                Roughness = materials[0].GetFloat("Roughness"),
                Metallic = materials[0].GetFloat("Metallic")
            };

            float totalWeight = weights.Sum();
            if (totalWeight < MathHelper.EPSILON) totalWeight = 1;

            Color3 color = Color3.Black;
            float roughness = 0, metallic = 0, ao = 0;
            Vec3 normal = Vec3.Zero;

            for (int i = 0; i < materials.Length && i < weights.Length; i++)
            {
                float w = weights[i] / totalWeight;
                color = color + materials[i].GetColor("BaseColor") * w;
                roughness += materials[i].GetFloat("Roughness") * w;
                metallic += materials[i].GetFloat("Metallic") * w;
                ao += materials[i].GetFloat("AO", 1f) * w;
                normal = normal + materials[i].GetVec3("Normal", Vec3.UnitZ) * w;
            }

            return new BlendedMaterialResult
            {
                BaseColor = color,
                Roughness = MathHelper.Clamp(roughness, 0.04f, 1),
                Metallic = MathHelper.Clamp(metallic, 0, 1),
                AO = MathHelper.Clamp(ao, 0, 1),
                Normal = normal.Normalized
            };
        }

        public BlendedMaterialResult TriplanarBlend(SubstrateMaterial material, Vec3 position, Vec3 normal, float sharpness = 4.0f)
        {
            Vec3 blend = new Vec3(
                MathHelper.Pow(Math.Abs(normal.X), sharpness),
                MathHelper.Pow(Math.Abs(normal.Y), sharpness),
                MathHelper.Pow(Math.Abs(normal.Z), sharpness));
            float total = blend.X + blend.Y + blend.Z;
            blend = blend / Math.Max(total, MathHelper.EPSILON);

            Color3 baseColor = material.GetColor("BaseColor");
            float roughness = material.GetFloat("Roughness");
            float metallic = material.GetFloat("Metallic");

            return new BlendedMaterialResult
            {
                BaseColor = baseColor * blend.Y,
                Roughness = roughness,
                Metallic = metallic,
                AO = material.GetFloat("AO", 1f)
            };
        }

        public List<float> ComputeSplatWeights(Vec3 position, List<Vec3> splatPositions, float radius = 10.0f)
        {
            var weights = new List<float>();
            float totalWeight = 0;

            foreach (var center in splatPositions)
            {
                float dist = (position - center).Length;
                float w = MathHelper.Saturate(1.0f - dist / radius);
                w = w * w; // Smooth falloff
                weights.Add(w);
                totalWeight += w;
            }

            if (totalWeight > MathHelper.EPSILON)
                for (int i = 0; i < weights.Count; i++)
                    weights[i] /= totalWeight;

            return weights;
        }
    }

    // =========================================================================
    // ANISOTROPY MODEL
    // =========================================================================

    public class AnisotropyModel
    {
        private const float PI = 3.14159265358979323846f;

        public Color3 EvaluateKajiyaKayBRDF(Vec3 incoming, Vec3 outgoing, Vec3 tangent, Color3 diffuseColor, Color3 specularColor, float roughness)
        {
            float sinL = (float)Math.Sqrt(Math.Max(1.0f - incoming.Dot(tangent) * incoming.Dot(tangent), 0));
            float sinV = (float)Math.Sqrt(Math.Max(1.0f - outgoing.Dot(tangent) * outgoing.Dot(tangent), 0));

            float diffuse = sinL * sinV + incoming.Dot(tangent) * outgoing.Dot(tangent);
            Vec3 halfVec = (incoming + outgoing).Normalized;
            float sinH = (float)Math.Sqrt(Math.Max(1.0f - halfVec.Dot(tangent) * halfVec.Dot(tangent), 0));
            float specular = (float)Math.Pow(Math.Max(sinH, 0), 1.0f / (roughness * roughness + 0.001f));

            return diffuseColor * diffuse * MathHelper.INV_PI + specularColor * specular * 0.5f;
        }

        public Vec3 ComputeTangentField(Vec3 position, Vec3 normal, float flowDirection, Func<Vec3, float> curvatureSample)
        {
            float curvature = curvatureSample(position);
            Vec3 tangent = GetTangentFromNormal(normal);
            float angle = curvature * flowDirection;
            float cosA = (float)Math.Cos(angle);
            float sinA = (float)Math.Sin(angle);
            return new Vec3(
                tangent.X * cosA - tangent.Y * sinA,
                tangent.X * sinA + tangent.Y * cosA,
                tangent.Z).Normalized;
        }

        public Vec3 ComputeFlowAlignedAnisotropy(Vec3 position, Vec3 normal, Vec3 flowDirection, float strength)
        {
            Vec3 tangent = flowDirection.Normalized;
            Vec3 bitangent = normal.Cross(tangent).Normalized;
            return Vec3.Lerp(normal, tangent, strength).Normalized;
        }

        public (Vec3 tangent, Vec3 bitangent) ComputeAnisotropicBasis(Vec3 normal, float rotation)
        {
            Vec3 tangent = GetTangentFromNormal(normal);
            Vec3 bitangent = normal.Cross(tangent).Normalized;
            float cosR = (float)Math.Cos(rotation);
            float sinR = (float)Math.Sin(rotation);
            Vec3 rotatedTangent = tangent * cosR + bitangent * sinR;
            Vec3 rotatedBitangent = normal.Cross(rotatedTangent).Normalized;
            return (rotatedTangent, rotatedBitangent);
        }

        public float EvaluateAnisotropicNDF(Vec3 halfVec, Vec3 tangent, Vec3 bitangent, Vec3 normal, float roughnessT, float roughnessB)
        {
            float NdotH = Math.Max(normal.Dot(halfVec), 0);
            float TdotH = tangent.Dot(halfVec);
            float BdotH = bitangent.Dot(halfVec);
            float at = roughnessT * roughnessT;
            float ab = roughnessB * roughnessB;
            float d = TdotH * TdotH / (at * at) + BdotH * BdotH / (ab * ab) + NdotH * NdotH;
            return 1.0f / (PI * at * ab * d * d + 1e-6f);
        }

        public float EvaluateAnisotropicGeometry(float NdotV, float NdotL, float TdotV, float BdotV, float TdotL, float BdotL, float roughnessT, float roughnessB)
        {
            float at = roughnessT * roughnessT;
            float ab = roughnessB * roughnessB;
            float GV = 1.0f / ((float)Math.Sqrt(TdotV * TdotV / (at * at) + BdotV * BdotV / (ab * ab) + NdotV * NdotV));
            float GL = 1.0f / ((float)Math.Sqrt(TdotL * TdotL / (at * at) + BdotL * BdotL / (ab * ab) + NdotL * NdotL));
            return GV * GL;
        }

        public float ComputeAnisotropyFromFlow(Vec3 flowDir, Vec3 normal)
        {
            Vec3 tangent = GetTangentFromNormal(normal);
            return Math.Abs(flowDir.Dot(tangent));
        }

        private static Vec3 GetTangentFromNormal(Vec3 normal)
        {
            Vec3 up = Math.Abs(normal.Z) < 0.999f ? Vec3.UnitZ : Vec3.UnitX;
            return normal.Cross(up).Normalized;
        }
    }

    // =========================================================================
    // MATERIAL ANALYTICS
    // =========================================================================

    public class MaterialAnalytics
    {
        public MaterialPropertyStats ComputePropertyStats(SubstrateMaterial material)
        {
            var stats = new MaterialPropertyStats();
            stats.TotalProperties = material.Properties.Count;
            stats.TextureCount = material.TextureSlots.Count;

            float totalComplexity = 0;
            foreach (var kvp in material.Properties)
            {
                if (kvp.Value.Type == MaterialPropertyType.Float)
                {
                    float val = kvp.Value.AsFloat();
                    stats.FloatProperties++;
                    stats.AverageFloatValue += val;
                }
                else if (kvp.Value.Type == MaterialPropertyType.Color)
                {
                    stats.ColorProperties++;
                }
                else if (kvp.Value.Type == MaterialPropertyType.Vec3)
                {
                    stats.Vec3Properties++;
                }
            }

            if (stats.FloatProperties > 0) stats.AverageFloatValue /= stats.FloatProperties;

            stats.EstimatedTextureMemory = material.TextureSlots.Count * 1024 * 1024;
            stats.FeatureFlags = material.ComputeActiveFeatures();
            stats.ComplexityScore = material.ComputeComplexityScore();
            stats.ActiveFeatureCount = CountSetBits((uint)stats.FeatureFlags);

            return stats;
        }

        public long EstimateTotalTextureMemory(SubstrateMaterial material)
        {
            long total = 0;
            foreach (var kvp in material.TextureSlots)
            {
                if (kvp.Value != null)
                {
                    total += 1024 * 1024; // Base estimate
                    if (kvp.Value.Compression != TextureCompression.None)
                        total = (long)(total * 0.25); // Compressed
                }
            }
            return total;
        }

        public int ComputeShaderPermutationCount(MaterialFeatureFlags features)
        {
            int count = 1;
            var bits = (uint)features;
            while (bits != 0)
            {
                count *= 2;
                bits &= bits - 1;
            }
            return count;
        }

        public int EstimateDrawCalls(SubstrateMaterial material)
        {
            int calls = 1;
            if (material.Layers.Count > 0) calls = Math.Max(calls, material.Layers.Count);
            if (material.FeatureFlags.HasFlag(MaterialFeatureFlags.Decal)) calls++;
            return calls;
        }

        public float ComputeGPUShaderCost(SubstrateMaterial material)
        {
            float cost = 10;
            cost += material.TextureSlots.Count * 5;
            cost += material.Layers.Count * 3;
            var features = material.ComputeActiveFeatures();
            if (features.HasFlag(MaterialFeatureFlags.SubsurfaceScattering)) cost += 20;
            if (features.HasFlag(MaterialFeatureFlags.ClearCoat)) cost += 10;
            if (features.HasFlag(MaterialFeatureFlags.Transmission)) cost += 15;
            if (features.HasFlag(MaterialFeatureFlags.HairBRDF)) cost += 25;
            if (features.HasFlag(MaterialFeatureFlags.EyeBRDF)) cost += 25;
            if (features.HasFlag(MaterialFeatureFlags.ParallaxOcclusion)) cost += 12;
            return cost;
        }

        public Dictionary<string, float> CompareMaterials(SubstrateMaterial a, SubstrateMaterial b)
        {
            var diff = new Dictionary<string, float>();
            var allKeys = a.Properties.Keys.Union(b.Properties.Keys).Distinct();
            foreach (string key in allKeys)
            {
                float va = a.GetFloat(key);
                float vb = b.GetFloat(key);
                diff[key] = Math.Abs(va - vb);
            }
            diff["TextureCountDiff"] = Math.Abs(a.TextureSlots.Count - b.TextureSlots.Count);
            diff["LayerCountDiff"] = Math.Abs(a.Layers.Count - b.Layers.Count);
            diff["FeatureDiff"] = Math.Abs((float)a.ComputeActiveFeatures() - (float)b.ComputeActiveFeatures());
            return diff;
        }

        public MaterialAnalyticsSummary ComputeBatchAnalytics(List<SubstrateMaterial> materials)
        {
            var summary = new MaterialAnalyticsSummary();
            summary.MaterialCount = materials.Count;
            summary.TotalProperties = materials.Sum(m => m.Properties.Count);
            summary.TotalTextures = materials.Sum(m => m.TextureSlots.Count);
            summary.TotalTextureMemory = materials.Sum(m => EstimateTotalTextureMemory(m));
            summary.TotalDrawCalls = materials.Sum(m => EstimateDrawCalls(m));
            summary.TotalGPUCost = materials.Sum(m => ComputeGPUShaderCost(m));
            summary.AverageComplexity = materials.Average(m => m.ComputeComplexityScore());
            summary.MaxComplexity = materials.Max(m => m.ComputeComplexityScore());

            var featureCounts = new Dictionary<MaterialFeatureFlags, int>();
            foreach (var mat in materials)
            {
                var features = mat.ComputeActiveFeatures();
                foreach (MaterialFeatureFlags flag in Enum.GetValues(typeof(MaterialFeatureFlags)))
                {
                    if (flag != MaterialFeatureFlags.None && features.HasFlag(flag))
                    {
                        if (!featureCounts.ContainsKey(flag)) featureCounts[flag] = 0;
                        featureCounts[flag]++;
                    }
                }
            }
            summary.FeatureUsage = featureCounts;
            return summary;
        }

        private static int CountSetBits(uint value)
        {
            int count = 0;
            while (value != 0) { count++; value &= value - 1; }
            return count;
        }
    }

    // =========================================================================
    // MATERIAL PROPERTY INTERPOLATOR
    // =========================================================================

    public class MaterialPropertyInterpolator
    {
        public static float Lerp(float a, float b, float t) => a + (b - a) * MathHelper.Saturate(t);
        public static Color3 Lerp(Color3 a, Color3 b, float t) => Color3.Lerp(a, b, MathHelper.Saturate(t));
        public static Vec3 Lerp(Vec3 a, Vec3 b, float t) => Vec3.Lerp(a, b, MathHelper.Saturate(t));

        public static float SmoothstepLerp(float a, float b, float t)
        {
            float s = MathHelper.Smoothstep(0, 1, t);
            return a + (b - a) * s;
        }

        public static Color3 HermiteLerp(Color3 a, Color3 b, float t)
        {
            float s = MathHelper.Saturate(t);
            float s2 = s * s;
            float s3 = s2 * s;
            float h1 = 2 * s3 - 3 * s2 + 1;
            float h2 = -2 * s3 + 3 * s2;
            float h3 = s3 - 2 * s2 + s;
            float h4 = s3 - s2;
            return a * h1 + b * h2;
        }

        public static float CatmullRom(float p0, float p1, float p2, float p3, float t)
        {
            float t2 = t * t;
            float t3 = t2 * t;
            return 0.5f * ((2 * p1) +
                (-p0 + p2) * t +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
        }

        public static Color3 BilinearLerp(Color3 tl, Color3 tr, Color3 bl, Color3 br, float u, float v)
        {
            Color3 top = Color3.Lerp(tl, tr, u);
            Color3 bottom = Color3.Lerp(bl, br, u);
            return Color3.Lerp(top, bottom, v);
        }

        public static float BilinearSample(float[,] grid, int gridW, int gridH, float u, float v)
        {
            float gx = u * (gridW - 1);
            float gy = v * (gridH - 1);
            int x0 = Math.Min((int)gx, gridW - 2);
            int y0 = Math.Min((int)gy, gridH - 2);
            float fx = gx - x0;
            float fy = gy - y0;
            float v00 = grid[x0, y0];
            float v10 = grid[x0 + 1, y0];
            float v01 = grid[x0, y0 + 1];
            float v11 = grid[x0 + 1, y0 + 1];
            return MathHelper.Lerp(MathHelper.Lerp(v00, v10, fx), MathHelper.Lerp(v01, v11, fx), fy);
        }

        public static void InterpolatePropertySet(
            Dictionary<string, MaterialProperty> from,
            Dictionary<string, MaterialProperty> to,
            float t,
            Dictionary<string, MaterialProperty> result)
        {
            foreach (var kvp in from)
            {
                if (to.TryGetValue(kvp.Key, out var toProp))
                {
                    if (kvp.Value.Type == MaterialPropertyType.Float)
                    {
                        float f = Lerp(kvp.Value.AsFloat(), toProp.AsFloat(), t);
                        result[kvp.Key] = new MaterialProperty(kvp.Key, MaterialPropertyType.Float, f, kvp.Value.Min, kvp.Value.Max, kvp.Value.Default);
                    }
                    else if (kvp.Value.Type == MaterialPropertyType.Color)
                    {
                        Color3 c = Lerp(kvp.Value.AsColor(), toProp.AsColor(), t);
                        result[kvp.Key] = new MaterialProperty(kvp.Key, MaterialPropertyType.Color, c);
                    }
                    else if (kvp.Value.Type == MaterialPropertyType.Vec3)
                    {
                        Vec3 v = Lerp(kvp.Value.AsVec3(), toProp.AsVec3(), t);
                        result[kvp.Key] = new MaterialProperty(kvp.Key, MaterialPropertyType.Vec3, v);
                    }
                    else
                    {
                        result[kvp.Key] = t < 0.5f ? kvp.Value : toProp;
                    }
                }
                else
                {
                    result[kvp.Key] = kvp.Value;
                }
            }
        }
    }

    // =========================================================================
    // MATERIAL BAKING UTILITIES
    // =========================================================================

    public class MaterialBaker
    {
        public byte[] BakeAlbedoMap(SubstrateMaterial material, int width, int height, Func<Vec2, Vec3> positionToUV)
        {
            var data = new byte[width * height * 4];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float u = (float)x / width;
                    float v = (float)y / height;
                    Vec2 uv = new Vec2(u, v);
                    Vec3 pos = positionToUV != null ? positionToUV(uv) : new Vec3(u, v, 0);

                    Color3 albedo = material.GetColor("BaseColor", new Color3(0.8f));
                    float metallic = material.GetFloat("Metallic", 0f);
                    float roughness = material.GetFloat("Roughness", 0.5f);

                    Color3 srgb = Color3.LinearToSRGB(albedo);
                    int i = (y * width + x) * 4;
                    data[i] = (byte)Math.Clamp(srgb.R * 255, 0, 255);
                    data[i + 1] = (byte)Math.Clamp(srgb.G * 255, 0, 255);
                    data[i + 2] = (byte)Math.Clamp(srgb.B * 255, 0, 255);
                    data[i + 3] = 255;
                }
            }
            return data;
        }

        public byte[] BakeNormalMap(SubstrateMaterial material, int width, int height)
        {
            var data = new byte[width * height * 4];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Vec3 normal = material.GetVec3("Normal", Vec3.UnitZ);
                    normal = normal.Normalized;
                    int i = (y * width + x) * 4;
                    data[i] = (byte)Math.Clamp((normal.X * 0.5f + 0.5f) * 255, 0, 255);
                    data[i + 1] = (byte)Math.Clamp((normal.Y * 0.5f + 0.5f) * 255, 0, 255);
                    data[i + 2] = (byte)Math.Clamp((normal.Z * 0.5f + 0.5f) * 255, 0, 255);
                    data[i + 3] = 255;
                }
            }
            return data;
        }

        public byte[] BakeRoughnessMetallicMap(SubstrateMaterial material, int width, int height)
        {
            var data = new byte[width * height * 4];
            float roughness = material.GetFloat("Roughness", 0.5f);
            float metallic = material.GetFloat("Metallic", 0f);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int i = (y * width + x) * 4;
                    data[i] = (byte)Math.Clamp(roughness * 255, 0, 255);
                    data[i + 1] = (byte)Math.Clamp(metallic * 255, 0, 255);
                    data[i + 2] = 0;
                    data[i + 3] = 255;
                }
            }
            return data;
        }

        public byte[] BakeEmissiveMap(SubstrateMaterial material, int width, int height)
        {
            var data = new byte[width * height * 4];
            Color3 emissive = material.GetColor("Emissive", Color3.Black);
            float intensity = material.GetFloat("EmissiveIntensity", 0f);
            Color3 finalEmissive = emissive * intensity;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int i = (y * width + x) * 4;
                    data[i] = (byte)Math.Clamp(finalEmissive.R * 255, 0, 255);
                    data[i + 1] = (byte)Math.Clamp(finalEmissive.G * 255, 0, 255);
                    data[i + 2] = (byte)Math.Clamp(finalEmissive.B * 255, 0, 255);
                    data[i + 3] = 255;
                }
            }
            return data;
        }

        public Dictionary<TextureChannel, byte[]> BakeAllMaps(SubstrateMaterial material, int width, int height, Func<Vec2, Vec3> positionToUV = null)
        {
            return new Dictionary<TextureChannel, byte[]>
            {
                { TextureChannel.Albedo, BakeAlbedoMap(material, width, height, positionToUV) },
                { TextureChannel.Normal, BakeNormalMap(material, width, height) },
                { TextureChannel.Roughness, BakeRoughnessMetallicMap(material, width, height) },
                { TextureChannel.Emissive, BakeEmissiveMap(material, width, height) }
            };
        }

        public void SaveBakedMaps(Dictionary<TextureChannel, byte[]> maps, string outputDirectory, string baseName, int width, int height)
        {
            if (!System.IO.Directory.Exists(outputDirectory))
                System.IO.Directory.CreateDirectory(outputDirectory);

            foreach (var kvp in maps)
            {
                string path = System.IO.Path.Combine(outputDirectory, $"{baseName}_{kvp.Key}.raw");
                System.IO.File.WriteAllBytes(path, kvp.Value);
            }
        }
    }

    // =========================================================================
    // MATERIAL CACHE
    // =========================================================================

    public class MaterialCache
    {
        private readonly Dictionary<string, CachedMaterialEntry> _cache;
        private readonly int _maxEntries;
        private int _hits;
        private int _misses;

        public int Count => _cache.Count;
        public float HitRate => _hits + _misses > 0 ? (float)_hits / (_hits + _misses) : 0;

        public MaterialCache(int maxEntries = 1024)
        {
            _cache = new Dictionary<string, CachedMaterialEntry>();
            _maxEntries = maxEntries;
        }

        public SubstrateMaterial GetOrAdd(string key, Func<SubstrateMaterial> creator)
        {
            if (_cache.TryGetValue(key, out var entry))
            {
                entry.LastAccess = DateTime.UtcNow;
                entry.AccessCount++;
                _hits++;
                return entry.Material;
            }

            _misses++;
            var material = creator();

            if (_cache.Count >= _maxEntries)
            {
                var oldest = _cache.OrderBy(kvp => kvp.Value.LastAccess).First();
                _cache.Remove(oldest.Key);
            }

            _cache[key] = new CachedMaterialEntry { Material = material, LastAccess = DateTime.UtcNow, AccessCount = 1 };
            return material;
        }

        public bool Contains(string key) => _cache.ContainsKey(key);

        public void Invalidate(string key) => _cache.Remove(key);

        public void Clear() { _cache.Clear(); _hits = 0; _misses = 0; }

        public void Purge(int targetCount)
        {
            while (_cache.Count > targetCount)
            {
                var oldest = _cache.OrderBy(kvp => kvp.Value.LastAccess).First();
                _cache.Remove(oldest.Key);
            }
        }

        private class CachedMaterialEntry
        {
            public SubstrateMaterial Material { get; set; }
            public DateTime LastAccess { get; set; }
            public int AccessCount { get; set; }
        }
    }

    // =========================================================================
    // MATERIAL CLIPBOARD (COPY/PASTE)
    // =========================================================================

    public class MaterialClipboard
    {
        private SubstrateMaterial _clipboard;
        private readonly Dictionary<string, object> _propertyBuffer;

        public bool HasContent => _clipboard != null || _propertyBuffer.Count > 0;

        public MaterialClipboard() { _propertyBuffer = new Dictionary<string, object>(); }

        public void CopyMaterial(SubstrateMaterial material) { _clipboard = material.Clone(); }
        public SubstrateMaterial PasteMaterial() => _clipboard?.Clone();

        public void CopyProperty(SubstrateMaterial material, string propertyName)
        {
            var prop = material.GetProperty(propertyName);
            if (prop != null) _propertyBuffer[propertyName] = prop.Value;
        }

        public void PasteProperties(SubstrateMaterial target)
        {
            foreach (var kvp in _propertyBuffer)
            {
                var existing = target.GetProperty(kvp.Key);
                if (existing != null)
                    target.SetProperty(kvp.Key, existing.WithValue(kvp.Value));
            }
        }

        public void CopyTextureSlots(SubstrateMaterial source, SubstrateMaterial target)
        {
            foreach (var kvp in source.TextureSlots)
                target.SetTexture(kvp.Key, kvp.Value);
        }

        public void CopyAll(SubstrateMaterial source)
        {
            _clipboard = source.Clone();
            _propertyBuffer.Clear();
            foreach (var kvp in source.Properties)
                _propertyBuffer[kvp.Key] = kvp.Value.Value;
        }
    }

    // =========================================================================
    // MATERIAL EVENT SYSTEM
    // =========================================================================

    public class MaterialChangedEventArgs : EventArgs
    {
        public string MaterialId { get; set; }
        public string PropertyName { get; set; }
        public object OldValue { get; set; }
        public object NewValue { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    public class MaterialTracker
    {
        private readonly Dictionary<string, SubstrateMaterial> _trackedMaterials;
        private readonly Dictionary<string, List<Action<MaterialChangedEventArgs>>> _listeners;

        public int TrackedCount => _trackedMaterials.Count;

        public MaterialTracker()
        {
            _trackedMaterials = new Dictionary<string, SubstrateMaterial>();
            _listeners = new Dictionary<string, List<Action<MaterialChangedEventArgs>>>();
        }

        public void Track(SubstrateMaterial material)
        {
            _trackedMaterials[material.Id] = material;
        }

        public void Untrack(string materialId) { _trackedMaterials.Remove(materialId); }

        public void Subscribe(string materialId, Action<MaterialChangedEventArgs> callback)
        {
            if (!_listeners.ContainsKey(materialId))
                _listeners[materialId] = new List<Action<MaterialChangedEventArgs>>();
            _listeners[materialId].Add(callback);
        }

        public void Unsubscribe(string materialId, Action<MaterialChangedEventArgs> callback)
        {
            if (_listeners.TryGetValue(materialId, out var list))
                list.Remove(callback);
        }

        public void NotifyChange(string materialId, string propertyName, object oldValue, object newValue)
        {
            if (_listeners.TryGetValue(materialId, out var list))
            {
                var args = new MaterialChangedEventArgs
                {
                    MaterialId = materialId,
                    PropertyName = propertyName,
                    OldValue = oldValue,
                    NewValue = newValue
                };
                foreach (var cb in list)
                    cb(args);
            }
        }

        public SubstrateMaterial GetTracked(string materialId)
        {
            _trackedMaterials.TryGetValue(materialId, out var m);
            return m;
        }

        public List<SubstrateMaterial> GetDirtyMaterials()
        {
            return _trackedMaterials.Values.Where(m => m.IsDirty).ToList();
        }
    }

    // =========================================================================
    // MATERIAL SERIALIZER
    // =========================================================================

    public class MaterialSerializer
    {
        public string SerializeToJson(SubstrateMaterial material)
        {
            var sb = new StringBuilder();
            sb.AppendLine("{");
            sb.AppendLine($"  \"name\": \"{material.Name}\",");
            sb.AppendLine($"  \"id\": \"{material.Id}\",");
            sb.AppendLine($"  \"domain\": {(int)material.Domain},");
            sb.AppendLine($"  \"features\": {(uint)material.FeatureFlags},");
            sb.AppendLine($"  \"version\": {material.Version},");
            sb.AppendLine("  \"properties\": {");

            bool first = true;
            foreach (var kvp in material.Properties)
            {
                if (!first) sb.AppendLine(",");
                first = false;
                sb.Append($"    \"{kvp.Key}\": {{");
                sb.Append($"\"type\": {(int)kvp.Value.Type}, ");
                if (kvp.Value.Type == MaterialPropertyType.Float)
                    sb.Append($"\"value\": {kvp.Value.AsFloat():F6}");
                else if (kvp.Value.Type == MaterialPropertyType.Color)
                {
                    var c = kvp.Value.AsColor();
                    sb.Append($"\"value\": [{c.R:F6}, {c.G:F6}, {c.B:F6}]");
                }
                else if (kvp.Value.Type == MaterialPropertyType.Vec3)
                {
                    var v = kvp.Value.AsVec3();
                    sb.Append($"\"value\": [{v.X:F6}, {v.Y:F6}, {v.Z:F6}]");
                }
                else
                    sb.Append($"\"value\": \"{kvp.Value.AsString()}\"");
                sb.Append("}");
            }
            sb.AppendLine();
            sb.AppendLine("  },");
            sb.AppendLine("  \"textures\": {");

            first = true;
            foreach (var kvp in material.TextureSlots)
            {
                if (!first) sb.AppendLine(",");
                first = false;
                sb.Append($"    \"{kvp.Key}\": {{");
                sb.Append($"\"path\": \"{kvp.Value?.Path ?? ""}\"");
                if (kvp.Value != null)
                {
                    sb.Append($", \"tiling\": [{kvp.Value.Tiling.X:F4}, {kvp.Value.Tiling.Y:F4}]");
                    sb.Append($", \"offset\": [{kvp.Value.Offset.X:F4}, {kvp.Value.Offset.Y:F4}]");
                    sb.Append($", \"rotation\": {kvp.Value.Rotation:F4}");
                }
                sb.Append("}");
            }
            sb.AppendLine();
            sb.AppendLine("  }");
            sb.AppendLine("}");
            return sb.ToString();
        }

        public SubstrateMaterial DeserializeFromJson(string json)
        {
            var material = new SubstrateMaterial();

            // Simple JSON parsing for material properties
            var lines = json.Split('\n');
            string currentProperty = null;
            foreach (var rawLine in lines)
            {
                string line = rawLine.Trim().TrimEnd(',');
                if (line.StartsWith("\"name\":"))
                    material.Name = ExtractStringValue(line);
                else if (line.StartsWith("\"domain\":"))
                    material.Domain = (MaterialDomain)ExtractIntValue(line);
                else if (line.StartsWith("\"features\":"))
                    material.FeatureFlags = (MaterialFeatureFlags)ExtractUIntValue(line);
                else if (line.Contains("\"type\":") && line.Contains("\"value\":"))
                {
                    var parts = line.Split(':');
                    if (parts.Length >= 2)
                    {
                        string key = parts[0].Trim().Trim('"', ' ');
                        int type = ExtractIntValue(line);
                        if (type == (int)MaterialPropertyType.Float)
                        {
                            float val = ExtractFloatValue(line);
                            material.SetProperty(key, new MaterialProperty(key, MaterialPropertyType.Float, val));
                        }
                        else if (type == (int)MaterialPropertyType.Color)
                        {
                            float[] vals = ExtractArrayValue(line);
                            if (vals != null && vals.Length >= 3)
                                material.SetProperty(key, new MaterialProperty(key, MaterialPropertyType.Color, new Color3(vals[0], vals[1], vals[2])));
                        }
                    }
                }
            }

            return material;
        }

        public byte[] SerializeBinary(SubstrateMaterial material) => Encoding.UTF8.GetBytes(material.ComputeHash().ToString());

        private static string ExtractStringValue(string line)
        {
            int start = line.IndexOf('"', line.IndexOf(':') + 1) + 1;
            int end = line.IndexOf('"', start);
            return start > 0 && end > start ? line.Substring(start, end - start) : "";
        }

        private static int ExtractIntValue(string line)
        {
            int colon = line.IndexOf(':');
            if (colon < 0) return 0;
            string val = line.Substring(colon + 1).Trim().TrimEnd(',');
            return int.TryParse(val, out int result) ? result : 0;
        }

        private static uint ExtractUIntValue(string line)
        {
            int colon = line.IndexOf(':');
            if (colon < 0) return 0;
            string val = line.Substring(colon + 1).Trim().TrimEnd(',');
            return uint.TryParse(val, out uint result) ? result : 0;
        }

        private static float ExtractFloatValue(string line)
        {
            int colon = line.IndexOf(':');
            if (colon < 0) return 0;
            string val = line.Substring(colon + 1).Trim().TrimEnd(',');
            return float.TryParse(val, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float result) ? result : 0;
        }

        private static float[] ExtractArrayValue(string line)
        {
            int bracketStart = line.IndexOf('[');
            int bracketEnd = line.IndexOf(']');
            if (bracketStart < 0 || bracketEnd < 0) return null;
            string arrayStr = line.Substring(bracketStart + 1, bracketEnd - bracketStart - 1);
            var parts = arrayStr.Split(',');
            var result = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                float.TryParse(parts[i].Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out result[i]);
            return result;
        }
    }

    // =========================================================================
    // MATERIAL LOD SYSTEM
    // =========================================================================

    public class MaterialLODSystem
    {
        private readonly Dictionary<int, SubstrateMaterial> _lodLevels;

        public int HighestLOD => _lodLevels.Keys.Max();
        public int LowestLOD => _lodLevels.Keys.Min();

        public MaterialLODSystem() { _lodLevels = new Dictionary<int, SubstrateMaterial>(); }

        public void AddLODLevel(int lod, SubstrateMaterial material) { _lodLevels[lod] = material; }

        public SubstrateMaterial GetLODForDistance(float distance, float[] lodDistances)
        {
            for (int i = 0; i < lodDistances.Length; i++)
            {
                if (distance < lodDistances[i])
                {
                    _lodLevels.TryGetValue(i, out var mat);
                    return mat;
                }
            }
            _lodLevels.TryGetValue(_lodLevels.Keys.Count - 1, out var lastMat);
            return lastMat;
        }

        public SubstrateMaterial GetLODForScreenSize(float screenSize)
        {
            float[] thresholds = { 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f };
            for (int i = 0; i < thresholds.Length; i++)
            {
                if (screenSize >= thresholds[i] && _lodLevels.TryGetValue(i, out var mat))
                    return mat;
            }
            int lastKey = _lodLevels.Keys.OrderBy(k => k).Last();
            return _lodLevels[lastKey];
        }

        public void SimplifyForLOD(SubstrateMaterial source, int targetLOD)
        {
            if (targetLOD == 0) return;

            if (targetLOD >= 2)
            {
                source.SetProperty("ClearCoat", 0f);
                source.SetProperty("Sheen", 0f);
                source.RemoveTexture(TextureChannel.ClearCoatNormal);
                source.RemoveTexture(TextureChannel.ClearCoatRoughness);
            }
            if (targetLOD >= 3)
            {
                source.SetProperty("Anisotropy", 0f);
                source.RemoveTexture(TextureChannel.AnisotropyTangent);
                source.RemoveTexture(TextureChannel.AnisotropyStrength);
                source.RemoveTexture(TextureChannel.DetailNormal);
                source.RemoveTexture(TextureChannel.DetailAlbedo);
            }
            if (targetLOD >= 4)
            {
                source.SetProperty("Transmission", 0f);
                source.SetProperty("SubsurfaceRadius", Vec3.Zero);
                source.RemoveTexture(TextureChannel.SubsurfaceColor);
                source.RemoveTexture(TextureChannel.Displacement);
                source.RemoveTexture(TextureChannel.Height);
            }

            source.ComputeActiveFeatures();
        }
    }

    // =========================================================================
    // MATERIAL SORTER / COMPARATOR
    // =========================================================================

    public class MaterialSorter
    {
        public List<SubstrateMaterial> SortByComplexity(List<SubstrateMaterial> materials)
        {
            return materials.OrderBy(m => m.ComputeComplexityScore()).ToList();
        }

        public List<SubstrateMaterial> SortByTextureCount(List<SubstrateMaterial> materials)
        {
            return materials.OrderBy(m => m.TextureSlots.Count).ToList();
        }

        public List<SubstrateMaterial> SortByFeatureCount(List<SubstrateMaterial> materials)
        {
            return materials.OrderBy(m => CountBits((uint)m.ComputeActiveFeatures())).ToList();
        }

        public List<SubstrateMaterial> SortByName(List<SubstrateMaterial> materials)
        {
            return materials.OrderBy(m => m.Name).ToList();
        }

        public List<SubstrateMaterial> GroupByDomain(List<SubstrateMaterial> materials)
        {
            return materials.OrderBy(m => m.Domain).ToList();
        }

        public Dictionary<MaterialDomain, List<SubstrateMaterial>> GroupByDomainGrouped(List<SubstrateMaterial> materials)
        {
            var groups = new Dictionary<MaterialDomain, List<SubstrateMaterial>>();
            foreach (var mat in materials)
            {
                if (!groups.ContainsKey(mat.Domain))
                    groups[mat.Domain] = new List<SubstrateMaterial>();
                groups[mat.Domain].Add(mat);
            }
            return groups;
        }

        public List<SubstrateMaterial> FilterByFeatures(List<SubstrateMaterial> materials, MaterialFeatureFlags requiredFlags)
        {
            return materials.Where(m => (m.ComputeActiveFeatures() & requiredFlags) == requiredFlags).ToList();
        }

        public List<SubstrateMaterial> FilterByDomain(List<SubstrateMaterial> materials, MaterialDomain domain)
        {
            return materials.Where(m => m.Domain.HasFlag(domain)).ToList();
        }

        private static int CountBits(uint value)
        {
            int count = 0;
            while (value != 0) { count++; value &= value - 1; }
            return count;
        }
    }

    // =========================================================================
    // MATERIAL DIFF
    // =========================================================================

    public class MaterialDiff
    {
        public List<MaterialDiffEntry> Entries { get; set; } = new();
        public bool HasDifferences => Entries.Count > 0;

        public static MaterialDiff Compute(SubstrateMaterial a, SubstrateMaterial b)
        {
            var diff = new MaterialDiff();

            var allKeys = a.Properties.Keys.Union(b.Properties.Keys).Distinct();
            foreach (string key in allKeys)
            {
                var propA = a.GetProperty(key);
                var propB = b.GetProperty(key);

                if (propA == null && propB != null)
                    diff.Entries.Add(new MaterialDiffEntry { PropertyName = key, Type = MaterialDiffType.Added, NewValue = propB.Value });
                else if (propA != null && propB == null)
                    diff.Entries.Add(new MaterialDiffEntry { PropertyName = key, Type = MaterialDiffType.Removed, OldValue = propA.Value });
                else if (propA != null && propB != null)
                {
                    if (!propA.Value.Equals(propB.Value))
                        diff.Entries.Add(new MaterialDiffEntry { PropertyName = key, Type = MaterialDiffType.Changed, OldValue = propA.Value, NewValue = propB.Value });
                }
            }

            var texA = a.TextureSlots.Keys.ToHashSet();
            var texB = b.TextureSlots.Keys.ToHashSet();
            foreach (var ch in texA.Except(texB))
                diff.Entries.Add(new MaterialDiffEntry { PropertyName = $"Texture_{ch}", Type = MaterialDiffType.Removed });
            foreach (var ch in texB.Except(texA))
                diff.Entries.Add(new MaterialDiffEntry { PropertyName = $"Texture_{ch}", Type = MaterialDiffType.Added });

            if (a.Domain != b.Domain)
                diff.Entries.Add(new MaterialDiffEntry { PropertyName = "Domain", Type = MaterialDiffType.Changed, OldValue = a.Domain, NewValue = b.Domain });
            if (a.FeatureFlags != b.FeatureFlags)
                diff.Entries.Add(new MaterialDiffEntry { PropertyName = "FeatureFlags", Type = MaterialDiffType.Changed, OldValue = a.FeatureFlags, NewValue = b.FeatureFlags });

            return diff;
        }

        public string GenerateReport()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Material Diff: {Entries.Count} differences");
            foreach (var entry in Entries)
            {
                sb.AppendLine($"  [{entry.Type}] {entry.PropertyName}");
                if (entry.OldValue != null) sb.AppendLine($"    Old: {entry.OldValue}");
                if (entry.NewValue != null) sb.AppendLine($"    New: {entry.NewValue}");
            }
            return sb.ToString();
        }
    }

    public class MaterialDiffEntry
    {
        public string PropertyName { get; set; }
        public MaterialDiffType Type { get; set; }
        public object OldValue { get; set; }
        public object NewValue { get; set; }
    }

    public enum MaterialDiffType { Added, Removed, Changed }

    public class MaterialPropertyStats
    {
        public int TotalProperties { get; set; }
        public int FloatProperties { get; set; }
        public int ColorProperties { get; set; }
        public int Vec3Properties { get; set; }
        public int TextureCount { get; set; }
        public float AverageFloatValue { get; set; }
        public long EstimatedTextureMemory { get; set; }
        public MaterialFeatureFlags FeatureFlags { get; set; }
        public float ComplexityScore { get; set; }
        public int ActiveFeatureCount { get; set; }
    }

    public class MaterialAnalyticsSummary
    {
        public int MaterialCount { get; set; }
        public int TotalProperties { get; set; }
        public int TotalTextures { get; set; }
        public long TotalTextureMemory { get; set; }
        public int TotalDrawCalls { get; set; }
        public float TotalGPUCost { get; set; }
        public float AverageComplexity { get; set; }
        public float MaxComplexity { get; set; }
        public Dictionary<MaterialFeatureFlags, int> FeatureUsage { get; set; } = new();
    }
}
