// ============================================================================
// NeuralCompilationPipeline.cs - G-DNN Engine: Complete Shader Compilation Pipeline
// ============================================================================
// This file implements the complete shader compilation pipeline for the G-DNN
// Engine, including neural graph transpilation, expression tree optimization,
// multi-backend code generation, caching, reflection, and diagnostics.
// ============================================================================
// ReSharper disable All
// ============================================================================

using System;
using System.Buffers.Binary;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Core.Genome;


namespace GDNN.Core.Compilation
{
    #region Pipeline Enums

    [Flags]
    public enum CompilationStage
    {
        None = 0, Transpilation = 1 << 0, Optimization = 1 << 1,
        CodeGeneration = 1 << 2, Linking = 1 << 3, Caching = 1 << 4,
        Reflection = 1 << 5, Validation = 1 << 6,
        All = Transpilation | Optimization | CodeGeneration | Linking | Caching | Reflection | Validation
    }

    public enum ShaderTargetLanguage { SpirV, Dxil, Air, Wgsl, Glsl, Hlsl }
    public enum OptimizationLevel { None, Basic, Aggressive, Ultra }
    public enum CompilationStatus { Pending, InProgress, Completed, Failed, Cached, Cancelled }

    public enum ShaderProfile
    {
        Pixel, Vertex, Compute, Geometry, TessellationControl, TessellationEvaluation,
        Mesh, Task, RayGeneration, ClosestHit, AnyHit, Miss, Intersection
    }

    public enum ResourceType
    {
        UniformBuffer, StorageBuffer, CombinedImageSampler, Sampler, Image,
        StorageImage, TexelBuffer, StorageTexelBuffer, AccelerationStructure,
        PushConstant, InputAttachment, SubpassInput
    }

    public enum TextureDimension
    {
        Tex1D, Tex2D, Tex3D, TexCube, Tex1DArray, Tex2DArray, TexCubeArray, Tex2DMS, Tex2DMSArray
    }

    public enum SamplerType
    {
        Sampler2D, Sampler3D, SamplerCube, Sampler2DArray, Sampler2DShadow,
        Sampler2DShadowArray, SamplerCubeShadow, Sampler1D, Sampler1DArray,
        ComparisonSampler2D, ComparisonSamplerCube
    }

    public enum BufferFormat
    {
        Rgba32Float, Rgba32Uint, Rgba32Sint, Rgba16Float, Rgba16Uint, Rgba16Sint,
        Rgba8Unorm, Rgba8Snorm, Rgba8Uint, Rgba8Sint,
        Rg32Float, Rg32Uint, Rg32Sint, Rg16Float, Rg16Uint, Rg16Sint, Rg8Unorm, Rg8Snorm,
        R32Float, R32Uint, R32Sint, R16Float, R16Uint, R16Sint,
        R8Unorm, R8Snorm, R8Uint, R8Sint,
        Bgra8Unorm, Bgra8Srgb, Rgba16Unorm, Rgba16Snorm,
        Rgb10A2Unorm, Rgb10A2Uint, Rg11FloatB10Float, Rgb9E5Float
    }

    public enum GpuType
    {
        Float, Half, Double, Int, UInt, Short, UShort, SByte, Byte, Bool,
        Vec2, Vec3, Vec4, IVec2, IVec3, IVec4, UVec2, UVec3, UVec4,
        BVec2, BVec3, BVec4,
        Mat2, Mat3, Mat4, Mat2x3, Mat3x2, Mat2x4, Mat4x2, Mat3x4, Mat4x3,
        Texture2D, Texture3D, TextureCube, Texture2DArray, Texture2DShadow,
        Sampler2D, Sampler3D, SamplerCube,
        Struct, Array, Void
    }

    [Flags]
    public enum OptimizationPass
    {
        None = 0, ConstantFolding = 1 << 0, CommonSubexpressionElimination = 1 << 1,
        DeadCodeElimination = 1 << 2, StrengthReduction = 1 << 3, Vectorization = 1 << 4,
        LoopUnrolling = 1 << 5, InstructionScheduling = 1 << 6, RegisterAllocation = 1 << 7,
        MemoryCoalescing = 1 << 8, WaveOptimization = 1 << 9, FunctionInlining = 1 << 10,
        AlgebraicSimplification = 1 << 11, RedundantLoadElimination = 1 << 12,
        PrecomputeLoops = 1 << 13,
        All = ConstantFolding | CommonSubexpressionElimination | DeadCodeElimination |
              StrengthReduction | Vectorization | LoopUnrolling | InstructionScheduling |
              RegisterAllocation | MemoryCoalescing | WaveOptimization | FunctionInlining |
              AlgebraicSimplification | RedundantLoadElimination | PrecomputeLoops
    }

    public enum DiagnosticSeverity { Info, Warning, PerformanceHint, Error, Fatal }

    public enum BinaryOperator
    {
        Add, Subtract, Multiply, Divide, Modulo, Power,
        And, Or, Xor, LessThan, LessThanOrEqual, GreaterThan, GreaterThanOrEqual,
        Equal, NotEqual, BitwiseAnd, BitwiseOr, BitwiseXor, ShiftLeft, ShiftRight,
        Min, Max, DotProduct, CrossProduct, Distance,
        Atan2, Pow, Step, SmoothStep, Fma, Hypot, LogBase,
        VectorDot, VectorCross, VectorLength
    }

    public enum UnaryOperator
    {
        Negate, Not, BitwiseNot, Abs, Sqrt, InverseSqrt, Reciprocal,
        Floor, Ceil, Round, Truncate, Frac,
        Sin, Cos, Tan, Asin, Acos, Atan,
        Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
        Exp, Exp2, Log, Log2, Log10,
        Normalize, Length, Saturate, Clamp01, NegateInvert,
        Ddx, Ddy, Fwidth, Sign, OnesComplement
    }

    public enum VariableStorageClass
    {
        Local, Uniform, Input, Output, PushConstant,
        StorageBuffer, Shared, Constant, Global
    }

    public enum LoopKind { For, While, DoWhile, Foreach, Unrolled }
    public enum TextureSampleType { Standard, Projected, Lod, Gradient, Gather, Fetch, DepthComparison }
    public enum TextureCompareFunction { LessOrEqual, GreaterOrEqual, Less, Greater, Equal, NotEqual, Always, Never }
    public enum AtomicOp { Add, Sub, Min, Max, And, Or, Xor, Exchange, CompSwap }
    public enum BarrierScope { Device, Workgroup, Invocation, AcquireRelease, SequentialConsistent }

    public enum SpirVExecutionModel
    {
        Vertex = 0, TessellationControl = 1, TessellationEvaluation = 2,
        Geometry = 3, Fragment = 4, GLCompute = 5, Kernel = 6,
        TaskNV = 5267, MeshNV = 5268,
        RayGenerationNV = 5313, IntersectHitNV = 5314,
        AnyHitNV = 5315, ClosestHitNV = 5316, MissNV = 5317, CallableNV = 5318
    }

    public enum SpirVStorageClass
    {
        UniformConstant = 0, Input = 1, Uniform = 2, Output = 3,
        WorkgroupLocal = 4, WorkgroupGlobal = 5, Private = 6,
        Function = 7, Generic = 8, PushConstant = 9,
        AtomicCounter = 10, Image = 11, StorageBuffer = 12
    }

    #endregion Pipeline Enums
    #region Pipeline Records

    public record CompilationRequest(
        GeoGenome Genome,
        ShaderTargetLanguage TargetBackend,
        OptimizationLevel OptimizationLevel,
        ShaderProfile Profile,
        ImmutableDictionary<string, string>? Defines = null,
        ImmutableArray<string> IncludePaths = default)
    {
        public static CompilationRequest Default(GeoGenome genome) =>
            new(genome, ShaderTargetLanguage.SpirV, OptimizationLevel.Basic, ShaderProfile.Compute);
        public static CompilationRequest Aggressive(GeoGenome genome, ShaderTargetLanguage backend) =>
            new(genome, backend, OptimizationLevel.Aggressive, ShaderProfile.Compute);
        public string ComputeCacheKey()
        {
            var sb = new StringBuilder();
            sb.Append(Genome.Id.ToCompactString());
            sb.Append("|").Append(TargetBackend).Append(OptimizationLevel).Append(Profile);
            if (Defines != null)
                foreach (var kvp in Defines.OrderBy(k => k.Key))
                    sb.Append("|").Append(kvp.Key).Append("=").Append(kvp.Value);
            using var sha = SHA256.Create();
            return Convert.ToBase64String(sha.ComputeHash(Encoding.UTF8.GetBytes(sb.ToString())));
        }
    }

    public record CompilationResult
    {
        public bool Success { get; init; }
        public ShaderModule? ShaderModule { get; init; }
        public CompilationDiagnostics Diagnostics { get; init; } = new();
        public ImmutableDictionary<CompilationStage, TimeSpan> Timings { get; init; } =
            ImmutableDictionary<CompilationStage, TimeSpan>.Empty;
        public bool CacheHit { get; init; }
        public TimeSpan TotalTime => Timings.Values.Aggregate(TimeSpan.Zero, (a, b) => a + b);
        public static CompilationResult SuccessResult(ShaderModule m, CompilationDiagnostics d,
            ImmutableDictionary<CompilationStage, TimeSpan> t, bool ch = false) =>
            new() { Success = true, ShaderModule = m, Diagnostics = d, Timings = t, CacheHit = ch };
        public static CompilationResult FailureResult(CompilationDiagnostics d,
            ImmutableDictionary<CompilationStage, TimeSpan> t) =>
            new() { Success = false, Diagnostics = d, Timings = t };
    }

    public record CompilationDiagnostics
    {
        public ImmutableArray<DiagnosticMessage> Warnings { get; init; } = ImmutableArray<DiagnosticMessage>.Empty;
        public ImmutableArray<DiagnosticMessage> Errors { get; init; } = ImmutableArray<DiagnosticMessage>.Empty;
        public ImmutableArray<DiagnosticMessage> PerformanceHints { get; init; } = ImmutableArray<DiagnosticMessage>.Empty;
        public double EstimatedCost { get; init; }
        public bool HasErrors => !Errors.IsDefaultOrEmpty && Errors.Length > 0;
        public bool HasWarnings => !Warnings.IsDefaultOrEmpty && Warnings.Length > 0;
        public int TotalMessages => (Warnings.IsDefaultOrEmpty ? 0 : Warnings.Length) +
            (Errors.IsDefaultOrEmpty ? 0 : Errors.Length) + (PerformanceHints.IsDefaultOrEmpty ? 0 : PerformanceHints.Length);
        public CompilationDiagnostics() { }
        public CompilationDiagnostics(ImmutableArray<DiagnosticMessage> w,
            ImmutableArray<DiagnosticMessage> e, ImmutableArray<DiagnosticMessage> p, double c = 0.0)
        { Warnings = w; Errors = e; PerformanceHints = p; EstimatedCost = c; }
    }

    public record DiagnosticMessage
    {
        public DiagnosticSeverity Severity { get; init; }
        public string Code { get; init; } = string.Empty;
        public string Message { get; init; } = string.Empty;
        public string? SourceFile { get; init; }
        public int? Line { get; init; }
        public int? Column { get; init; }
        public string? Hint { get; init; }
        public string? Category { get; init; }
        public static DiagnosticMessage Warn(string c, string m, string? h = null) =>
            new() { Severity = DiagnosticSeverity.Warning, Code = c, Message = m, Hint = h };
        public static DiagnosticMessage Error(string c, string m, string? f = null, int? l = null, int? col = null) =>
            new() { Severity = DiagnosticSeverity.Error, Code = c, Message = m, SourceFile = f, Line = l, Column = col };
        public static DiagnosticMessage PerfHint(string c, string m, string? h = null) =>
            new() { Severity = DiagnosticSeverity.PerformanceHint, Code = c, Message = m, Hint = h };
        public static DiagnosticMessage Info(string c, string m) =>
            new() { Severity = DiagnosticSeverity.Info, Code = c, Message = m };
    }

    public class ShaderModule
    {
        public byte[] Bytecode { get; init; } = Array.Empty<byte>();
        public ShaderTargetLanguage TargetLanguage { get; init; }
        public ImmutableDictionary<string, ShaderProfile> EntryPoints { get; init; } =
            ImmutableDictionary<string, ShaderProfile>.Empty;
        public ImmutableArray<ResourceBinding> ResourceBindings { get; init; } = ImmutableArray<ResourceBinding>.Empty;
        public ImmutableArray<UniformBinding> UniformBindings { get; init; } = ImmutableArray<UniformBinding>.Empty;
        public ImmutableArray<PushConstantBlock> PushConstants { get; init; } = ImmutableArray<PushConstantBlock>.Empty;
        public ImmutableArray<SpecializationConstant> SpecializationConstants { get; init; } =
            ImmutableArray<SpecializationConstant>.Empty;
        public int EstimatedRegisterPressure { get; init; }
        public double EstimatedCost { get; init; }
        public string Hash { get; init; } = string.Empty;
        public string? SourceText { get; init; }
        public DateTime CompiledAt { get; init; } = DateTime.UtcNow;
        public int InstructionCount { get; init; }
        public int TemporaryRegisterCount { get; init; }
        public int TextureSamplerCount { get; init; }
    }

    public record ResourceBinding
    {
        public int Set { get; init; }
        public int Binding { get; init; }
        public ResourceType Type { get; init; }
        public ShaderProfile Stage { get; init; }
        public int Count { get; init; } = 1;
        public GpuType ElementType { get; init; } = GpuType.Float;
        public string Name { get; init; } = string.Empty;
        public TextureDimension? TextureDim { get; init; }
        public BufferFormat? Format { get; init; }
    }

    public record UniformBinding
    {
        public string Name { get; init; } = string.Empty;
        public int Offset { get; init; }
        public int Size { get; init; }
        public GpuType Type { get; init; }
        public bool Used { get; init; } = true;
        public string? MemberName { get; init; }
        public int MemberIndex { get; init; }
    }

    public record PushConstantBlock
    {
        public string Name { get; init; } = string.Empty;
        public int Offset { get; init; }
        public int Size { get; init; }
        public ShaderProfile Stages { get; init; }
    }

    public record SpecializationConstant
    {
        public uint Id { get; init; }
        public string Name { get; init; } = string.Empty;
        public GpuType Type { get; init; }
        public byte[] DefaultValue { get; init; } = Array.Empty<byte>();
    }

    public record SpirVCode
    {
        public uint[] Words { get; init; } = Array.Empty<uint>();
        public string? SourceText { get; init; }
        public int BoundIdCount { get; init; }
        public static SpirVCode FromBytes(byte[] bytes)
        {
            var w = new uint[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, w, 0, bytes.Length);
            return new SpirVCode { Words = w };
        }
        public byte[] ToBytes()
        {
            var b = new byte[Words.Length * 4];
            Buffer.BlockCopy(Words, 0, b, 0, b.Length);
            return b;
        }
    }

    public record DxilCode
    {
        public byte[] Bytecode { get; init; } = Array.Empty<byte>();
        public string? SourceText { get; init; }
        public int ValidatorVersion { get; init; } = 1;
    }

    public record AirCode
    {
        public byte[] Bytecode { get; init; } = Array.Empty<byte>();
        public string? SourceText { get; init; }
        public string? MetalLibrarySource { get; init; }
    }

    public record BackendCompilationResult
    {
        public bool Success { get; init; }
        public byte[] Bytecode { get; init; } = Array.Empty<byte>();
        public string? SourceText { get; init; }
        public ImmutableArray<DiagnosticMessage> Diagnostics { get; init; } =
            ImmutableArray<DiagnosticMessage>.Empty;
        public int RegisterPressure { get; init; }
        public int InstructionCount { get; init; }
    }

    public record BackendCapabilities
    {
        public int MaxBindings { get; init; } = 128;
        public int MaxPushConstantSize { get; init; } = 128;
        public int MaxUniformBufferSize { get; init; } = 65536;
        public int MaxTextureDimension { get; init; } = 16384;
        public int MaxWorkGroupSize { get; init; } = 1024;
        public int MaxWorkGroupCount { get; init; } = 65535;
        public ImmutableArray<ShaderProfile> SupportedProfiles { get; init; } =
            ImmutableArray<ShaderProfile>.Empty;
        public bool SupportsWaveIntrinsics { get; init; }
        public bool SupportsRayTracing { get; init; }
        public bool SupportsMeshShaders { get; init; }
        public int PreferredSimdWidth { get; init; } = 4;
        public int MaxTextureArrayLayers { get; init; } = 256;
        public int MaxColorAttachments { get; init; } = 8;
        public int MaxVertexAttributes { get; init; } = 16;
    }

    public record ShaderDiagnostic
    {
        public DiagnosticSeverity Severity { get; init; }
        public string Message { get; init; } = string.Empty;
        public int InstructionIndex { get; init; }
        public string? SourceLocation { get; init; }
        public string? OptimizationPass { get; init; }
    }

    public record ShaderPerformanceHint
    {
        public string Category { get; init; } = string.Empty;
        public string Message { get; init; } = string.Empty;
        public double EstimatedSpeedup { get; init; }
        public string Suggestion { get; init; } = string.Empty;
        public int AffectedInstructionCount { get; init; }
    }

    public class TextureDeclaration
    {
        public string Name { get; set; } = string.Empty;
        public GpuType Dimension { get; set; } = GpuType.Texture2D;
        public bool IsCombined { get; set; } = true;
        public int Set { get; set; }
        public int Binding { get; set; }
        public string Format { get; set; } = "rgba8";
        public bool IsReadOnly { get; set; } = true;
        public TextureDimension TexDim { get; set; } = TextureDimension.Tex2D;
        public BufferFormat BufFormat { get; set; } = BufferFormat.Rgba8Unorm;
    }

    #endregion Pipeline Records

    #region Expression Tree Nodes

    public abstract class ExpressionNode
    {
        public int Id { get; set; }
        public GpuType ResultType { get; set; } = GpuType.Float;
        public int? SourceNeuronId { get; set; }
        public int SourceLine { get; set; }
        [JsonIgnore] public bool Visited { get; set; }
        [JsonIgnore] public bool IsMarkedForRemoval { get; set; }
        public int StructuralHash { get; protected set; }
        public abstract IEnumerable<ExpressionNode> GetChildren();
        public abstract ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null);
        public abstract T Accept<T>(IExpressionVisitor<T> visitor) where T : class;
        protected abstract void RecalculateHash();
        public abstract override string ToString();
        public int CountNodes() { int c = 1; foreach (var ch in GetChildren()) c += ch.CountNodes(); return c; }
        public int MaxDepth() { int m = 0; foreach (var ch in GetChildren()) { int d = ch.MaxDepth(); if (d > m) m = d; } return m + 1; }
        public IEnumerable<ExpressionNode> TraversePreOrder() { yield return this; foreach (var c in GetChildren()) foreach (var d in c.TraversePreOrder()) yield return d; }
        public IEnumerable<ExpressionNode> TraversePostOrder() { foreach (var c in GetChildren()) foreach (var d in c.TraversePostOrder()) yield return d; yield return this; }
    }

    public class BinaryExpressionNode : ExpressionNode
    {
        public BinaryOperator Op { get; set; }
        public ExpressionNode Left { get; set; } = null!;
        public ExpressionNode Right { get; set; } = null!;
        public BinaryExpressionNode(BinaryOperator op, ExpressionNode left, ExpressionNode right) { Op = op; Left = left; Right = right; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() { yield return Left; yield return Right; }
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null)
        {
            cache ??= new();
            if (cache.TryGetValue(Id, out var c)) return c;
            var clone = new BinaryExpressionNode(Op, Left.Clone(cache), Right.Clone(cache)) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId, SourceLine = SourceLine };
            cache[Id] = clone; return clone;
        }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitBinary(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = ((int)Op * 397) ^ Left.StructuralHash ^ (Right.StructuralHash * 397); } }
        public override string ToString() => $"({Left} {Op switch { BinaryOperator.Add => "+", BinaryOperator.Subtract => "-", BinaryOperator.Multiply => "*", BinaryOperator.Divide => "/", BinaryOperator.Modulo => "%", BinaryOperator.And => "&&", BinaryOperator.Or => "||", BinaryOperator.LessThan => "<", BinaryOperator.LessThanOrEqual => "<=", BinaryOperator.GreaterThan => ">", BinaryOperator.GreaterThanOrEqual => ">=", BinaryOperator.Equal => "==", BinaryOperator.NotEqual => "!=", BinaryOperator.BitwiseAnd => "&", BinaryOperator.BitwiseOr => "|", BinaryOperator.BitwiseXor => "^", BinaryOperator.ShiftLeft => "<<", BinaryOperator.ShiftRight => ">>", BinaryOperator.Min => "min", BinaryOperator.Max => "max", BinaryOperator.DotProduct => "dot", _ => $"op{(int)Op}" }} {Right})";
    }

    public class UnaryExpressionNode : ExpressionNode
    {
        public UnaryOperator Op { get; set; }
        public ExpressionNode Operand { get; set; } = null!;
        public bool IsPrefix { get; set; } = true;
        public UnaryExpressionNode(UnaryOperator op, ExpressionNode operand, bool isPrefix = true) { Op = op; Operand = operand; IsPrefix = isPrefix; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() { yield return Operand; }
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null)
        {
            cache ??= new();
            if (cache.TryGetValue(Id, out var c)) return c;
            var clone = new UnaryExpressionNode(Op, Operand.Clone(cache), IsPrefix) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId, SourceLine = SourceLine };
            cache[Id] = clone; return clone;
        }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitUnary(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = ((int)Op * 397) ^ Operand.StructuralHash; } }
        public override string ToString()
        {
            string o = Op switch { UnaryOperator.Negate => "-", UnaryOperator.Not => "!", UnaryOperator.Abs => "abs", UnaryOperator.Sqrt => "sqrt", UnaryOperator.Floor => "floor", UnaryOperator.Ceil => "ceil", UnaryOperator.Round => "round", UnaryOperator.Frac => "fract", UnaryOperator.Sin => "sin", UnaryOperator.Cos => "cos", UnaryOperator.Tan => "tan", UnaryOperator.Asin => "asin", UnaryOperator.Acos => "acos", UnaryOperator.Atan => "atan", UnaryOperator.Exp => "exp", UnaryOperator.Exp2 => "exp2", UnaryOperator.Log => "log", UnaryOperator.Log2 => "log2", UnaryOperator.Normalize => "normalize", UnaryOperator.Length => "length", UnaryOperator.Saturate => "saturate", UnaryOperator.Sign => "sign", _ => $"unary{(int)Op}" };
            return IsPrefix ? $"{o}({Operand})" : $"({Operand}){o}";
        }
    }

    public class FunctionCallNode : ExpressionNode
    {
        public string FunctionName { get; set; } = string.Empty;
        public IReadOnlyList<ExpressionNode> Arguments { get; set; } = Array.Empty<ExpressionNode>();
        public GpuType ReturnType { get; set; } = GpuType.Float;
        public bool IsIntrinsic { get; set; }
        public FunctionCallNode(string fn, IReadOnlyList<ExpressionNode> args, GpuType rt = GpuType.Float, bool intrinsic = true) { FunctionName = fn; Arguments = args; ReturnType = rt; IsIntrinsic = intrinsic; ResultType = rt; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() => Arguments;
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null)
        {
            cache ??= new();
            if (cache.TryGetValue(Id, out var c)) return c;
            var a = Arguments.Select(x => x.Clone(cache)).ToList();
            var clone = new FunctionCallNode(FunctionName, a, ReturnType, IsIntrinsic) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId, SourceLine = SourceLine };
            cache[Id] = clone; return clone;
        }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitFunctionCall(this);
        protected override void RecalculateHash() { unchecked { int h = FunctionName.GetHashCode(); foreach (var a in Arguments) h = (h * 397) ^ a.StructuralHash; StructuralHash = h; } }
        public override string ToString() => $"{FunctionName}({string.Join(", ", Arguments.Select(a => a.ToString()))})";
    }

    public class VariableNode : ExpressionNode
    {
        public string Name { get; set; } = string.Empty;
        public VariableStorageClass StorageClass { get; set; }
        public bool IsIO { get; set; }
        public int BindingSet { get; set; }
        public int BindingIndex { get; set; }
        public ImmutableArray<int> ArrayDimensions { get; set; } = ImmutableArray<int>.Empty;
        public VariableNode(string name, GpuType type, VariableStorageClass storage = VariableStorageClass.Local) { Name = name; ResultType = type; StorageClass = storage; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() => Enumerable.Empty<ExpressionNode>();
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) => new VariableNode(Name, ResultType, StorageClass) { Id = Id, SourceNeuronId = SourceNeuronId, SourceLine = SourceLine, IsIO = IsIO, BindingSet = BindingSet, BindingIndex = BindingIndex, ArrayDimensions = ArrayDimensions };
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitVariable(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = Name.GetHashCode() ^ ((int)StorageClass * 397); } }
        public override string ToString() => Name;
    }

    public class ConstantNode : ExpressionNode
    {
        public float FloatValue { get; set; }
        public int IntValue { get; set; }
        public bool BoolValue { get; set; }
        public float[] VectorComponents { get; set; } = Array.Empty<float>();
        public float[] MatrixComponents { get; set; } = Array.Empty<float>();
        public bool IsFloat { get; set; } = true;
        public int VectorWidth => VectorComponents.Length;
        public int MatrixRows { get; set; }
        public int MatrixColumns { get; set; }
        public ConstantNode(float v) { FloatValue = v; IsFloat = true; ResultType = GpuType.Float; RecalculateHash(); }
        public ConstantNode(int v) { IntValue = v; IsFloat = false; ResultType = GpuType.Int; RecalculateHash(); }
        public ConstantNode(bool v) { BoolValue = v; IsFloat = false; ResultType = GpuType.Bool; RecalculateHash(); }
        public ConstantNode(float[] c, GpuType t) { VectorComponents = c; IsFloat = true; ResultType = t; RecalculateHash(); }
        public ConstantNode(float[] m, int rows, int cols) { MatrixComponents = m; MatrixRows = rows; MatrixColumns = cols; IsFloat = true; ResultType = rows == 4 && cols == 4 ? GpuType.Mat4 : rows == 3 && cols == 3 ? GpuType.Mat3 : GpuType.Mat2; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() => Enumerable.Empty<ExpressionNode>();
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null)
        {
            if (ResultType == GpuType.Bool) return new ConstantNode(BoolValue) { Id = Id, SourceNeuronId = SourceNeuronId };
            if (IsFloat && MatrixComponents.Length > 0) return new ConstantNode((float[])MatrixComponents.Clone(), MatrixRows, MatrixColumns) { Id = Id, SourceNeuronId = SourceNeuronId };
            if (IsFloat && VectorComponents.Length > 1) return new ConstantNode((float[])VectorComponents.Clone(), ResultType) { Id = Id, SourceNeuronId = SourceNeuronId };
            if (IsFloat) return new ConstantNode(FloatValue) { Id = Id, SourceNeuronId = SourceNeuronId };
            return new ConstantNode(IntValue) { Id = Id, SourceNeuronId = SourceNeuronId };
        }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitConstant(this);
        protected override void RecalculateHash()
        {
            if (IsFloat && MatrixComponents.Length > 0) { unchecked { int h = MatrixRows * 31 + MatrixColumns; foreach (var c in MatrixComponents) h = (h * 397) ^ c.GetHashCode(); StructuralHash = h; } }
            else if (IsFloat && VectorComponents.Length > 1) { unchecked { int h = VectorComponents.Length; foreach (var c in VectorComponents) h = (h * 397) ^ c.GetHashCode(); StructuralHash = h; } }
            else if (IsFloat) StructuralHash = FloatValue.GetHashCode();
            else if (ResultType == GpuType.Bool) StructuralHash = BoolValue.GetHashCode();
            else StructuralHash = IntValue.GetHashCode();
        }
        public override string ToString()
        {
            if (ResultType == GpuType.Bool) return BoolValue ? "true" : "false";
            if (IsFloat && MatrixComponents.Length > 0) return $"mat{MatrixRows}x{MatrixColumns}(...)";
            if (IsFloat && VectorComponents.Length > 1) return $"vec{VectorComponents.Length}({string.Join(", ", VectorComponents.Select(c => c.ToString("G", CultureInfo.InvariantCulture)))})";
            if (IsFloat) return FloatValue.ToString("G", CultureInfo.InvariantCulture);
            return IntValue.ToString(CultureInfo.InvariantCulture);
        }
        public float GetComponent(int i) => i < VectorComponents.Length ? VectorComponents[i] : 0f;
        public float GetMatrixElement(int r, int c) => r < MatrixRows && c < MatrixColumns ? MatrixComponents[r * MatrixColumns + c] : 0f;
        public static ConstantNode Zero(GpuType t) => t switch { GpuType.Int or GpuType.UInt => new(0) { ResultType = t }, GpuType.Bool => new(false), GpuType.Vec2 => new(new float[] { 0, 0 }, GpuType.Vec2), GpuType.Vec3 => new(new float[] { 0, 0, 0 }, GpuType.Vec3), GpuType.Vec4 => new(new float[] { 0, 0, 0, 0 }, GpuType.Vec4), _ => new(0f) };
        public static ConstantNode One(GpuType t) => t switch { GpuType.Int or GpuType.UInt => new(1) { ResultType = t }, GpuType.Bool => new(true), GpuType.Vec2 => new(new float[] { 1, 1 }, GpuType.Vec2), GpuType.Vec3 => new(new float[] { 1, 1, 1 }, GpuType.Vec3), GpuType.Vec4 => new(new float[] { 1, 1, 1, 1 }, GpuType.Vec4), _ => new(1f) };
        public static ConstantNode Identity4x4() => new(new float[] { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }, 4, 4);
    }


    public class LoopNode : ExpressionNode
    {
        public VariableNode LoopVariable { get; set; } = null!;
        public ExpressionNode StartValue { get; set; } = null!;
        public ExpressionNode EndValue { get; set; } = null!;
        public ExpressionNode StepValue { get; set; } = null!;
        public IReadOnlyList<ExpressionNode> Body { get; set; } = Array.Empty<ExpressionNode>();
        public LoopKind Kind { get; set; }
        public bool IsUnrollable { get; set; }
        public int? EstimatedIterations { get; set; }
        public IReadOnlyList<ExpressionNode> BreakConditions { get; set; } = Array.Empty<ExpressionNode>();
        public IReadOnlyList<ExpressionNode> ContinueConditions { get; set; } = Array.Empty<ExpressionNode>();

        public LoopNode(LoopKind kind, VariableNode loopVar, ExpressionNode start, ExpressionNode end, ExpressionNode step, IReadOnlyList<ExpressionNode> body)
        {
            Kind = kind; LoopVariable = loopVar; StartValue = start; EndValue = end; StepValue = step; Body = body; RecalculateHash();
        }

        public override IEnumerable<ExpressionNode> GetChildren()
        {
            yield return LoopVariable; yield return StartValue; yield return EndValue; yield return StepValue;
            foreach (var s in Body) yield return s;
        }

        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null)
        {
            cache ??= new();
            return new LoopNode(Kind, (VariableNode)LoopVariable.Clone(cache), StartValue.Clone(cache), EndValue.Clone(cache), StepValue.Clone(cache), Body.Select(b => b.Clone(cache)).ToList())
            { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId, IsUnrollable = IsUnrollable, EstimatedIterations = EstimatedIterations };
        }

        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitLoop(this);

        protected override void RecalculateHash()
        {
            unchecked { int h = ((int)Kind * 397) ^ LoopVariable.StructuralHash; h = (h * 397) ^ StartValue.StructuralHash; h = (h * 397) ^ EndValue.StructuralHash; foreach (var s in Body) h = (h * 397) ^ s.StructuralHash; StructuralHash = h; }
        }

        public override string ToString() => $"for({LoopVariable.Name}={StartValue}; {LoopVariable.Name}<{EndValue}; {LoopVariable.Name}+={StepValue}) {{...}}";
    }

    public class ConditionalNode : ExpressionNode
    {
        public ExpressionNode Condition { get; set; } = null!;
        public IReadOnlyList<ExpressionNode> ThenBranch { get; set; } = Array.Empty<ExpressionNode>();
        public IReadOnlyList<ExpressionNode> ElseBranch { get; set; } = Array.Empty<ExpressionNode>();
        public bool IsTernary { get; set; }
        public ExpressionNode? TrueValue { get; set; }
        public ExpressionNode? FalseValue { get; set; }

        public ConditionalNode(ExpressionNode cond, ExpressionNode tv, ExpressionNode fv)
        { Condition = cond; TrueValue = tv; FalseValue = fv; IsTernary = true; RecalculateHash(); }

        public ConditionalNode(ExpressionNode cond, IReadOnlyList<ExpressionNode> tb, IReadOnlyList<ExpressionNode> eb)
        { Condition = cond; ThenBranch = tb; ElseBranch = eb; IsTernary = false; RecalculateHash(); }

        public override IEnumerable<ExpressionNode> GetChildren()
        {
            yield return Condition;
            if (IsTernary) { if (TrueValue != null) yield return TrueValue; if (FalseValue != null) yield return FalseValue; }
            else { foreach (var n in ThenBranch) yield return n; foreach (var n in ElseBranch) yield return n; }
        }

        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null)
        {
            cache ??= new();
            if (IsTernary) return new ConditionalNode(Condition.Clone(cache), TrueValue!.Clone(cache), FalseValue!.Clone(cache)) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId };
            return new ConditionalNode(Condition.Clone(cache), ThenBranch.Select(b => b.Clone(cache)).ToList(), ElseBranch.Select(b => b.Clone(cache)).ToList()) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId };
        }

        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitConditional(this);

        protected override void RecalculateHash()
        {
            unchecked { int h = Condition.StructuralHash; if (IsTernary) { h = (h * 397) ^ (TrueValue?.StructuralHash ?? 0); h = (h * 397) ^ (FalseValue?.StructuralHash ?? 0); } else { foreach (var n in ThenBranch) h = (h * 397) ^ n.StructuralHash; foreach (var n in ElseBranch) h = (h * 397) ^ n.StructuralHash; } StructuralHash = h; }
        }

        public override string ToString() => IsTernary ? $"({Condition} ? {TrueValue} : {FalseValue})" : $"if({Condition}){{...}}else{{...}}";
    }

    public class TextureSampleNode : ExpressionNode
    {
        public VariableNode Texture { get; set; } = null!;
        public VariableNode? Sampler { get; set; }
        public ExpressionNode Coordinates { get; set; } = null!;
        public ExpressionNode? Lod { get; set; }
        public ExpressionNode? Bias { get; set; }
        public ExpressionNode? Offset { get; set; }
        public ExpressionNode? Projection { get; set; }
        public ExpressionNode? GradX { get; set; }
        public ExpressionNode? GradY { get; set; }
        public ExpressionNode? ArrayIndex { get; set; }
        public TextureSampleType SampleType { get; set; }
        public TextureCompareFunction CompareFunction { get; set; }

        public TextureSampleNode(VariableNode tex, ExpressionNode coords, TextureSampleType st = TextureSampleType.Standard)
        { Texture = tex; Coordinates = coords; SampleType = st; ResultType = GpuType.Vec4; RecalculateHash(); }

        public override IEnumerable<ExpressionNode> GetChildren()
        {
            yield return Texture; if (Sampler != null) yield return Sampler; yield return Coordinates;
            if (Lod != null) yield return Lod; if (Bias != null) yield return Bias; if (Offset != null) yield return Offset;
            if (GradX != null) yield return GradX; if (GradY != null) yield return GradY; if (ArrayIndex != null) yield return ArrayIndex;
        }

        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null)
        {
            cache ??= new();
            var n = new TextureSampleNode((VariableNode)Texture.Clone(cache), Coordinates.Clone(cache), SampleType)
            { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId, CompareFunction = CompareFunction, Lod = Lod?.Clone(cache), Bias = Bias?.Clone(cache), Offset = Offset?.Clone(cache), Projection = Projection?.Clone(cache), GradX = GradX?.Clone(cache), GradY = GradY?.Clone(cache), ArrayIndex = ArrayIndex?.Clone(cache) };
            if (Sampler != null) n.Sampler = (VariableNode)Sampler.Clone(cache);
            return n;
        }

        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitTextureSample(this);

        protected override void RecalculateHash()
        { unchecked { int h = Texture.StructuralHash ^ ((int)SampleType * 397); h = (h * 397) ^ Coordinates.StructuralHash; if (Lod != null) h = (h * 397) ^ Lod.StructuralHash; StructuralHash = h; } }

        public override string ToString() => $"texture({Texture.Name},{Coordinates})";
    }

    public class TextureFetchNode : ExpressionNode
    {
        public VariableNode Texture { get; set; } = null!;
        public ExpressionNode Coordinates { get; set; } = null!;
        public ExpressionNode? LodLevel { get; set; }
        public ExpressionNode? SampleIndex { get; set; }

        public TextureFetchNode(VariableNode tex, ExpressionNode coords)
        { Texture = tex; Coordinates = coords; ResultType = GpuType.Vec4; RecalculateHash(); }

        public override IEnumerable<ExpressionNode> GetChildren()
        { yield return Texture; yield return Coordinates; if (LodLevel != null) yield return LodLevel; if (SampleIndex != null) yield return SampleIndex; }

        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null)
        { cache ??= new(); return new TextureFetchNode((VariableNode)Texture.Clone(cache), Coordinates.Clone(cache)) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId, LodLevel = LodLevel?.Clone(cache), SampleIndex = SampleIndex?.Clone(cache) }; }

        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitTextureFetch(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = (Texture.StructuralHash * 397) ^ Coordinates.StructuralHash; } }
        public override string ToString() => $"texelFetch({Texture.Name},{Coordinates})";
    }

    public class BufferLoadNode : ExpressionNode
    {
        public VariableNode Buffer { get; set; } = null!;
        public ExpressionNode Index { get; set; } = null!;
        public ExpressionNode? ByteOffset { get; set; }

        public BufferLoadNode(VariableNode buf, ExpressionNode idx) { Buffer = buf; Index = idx; RecalculateHash(); }

        public override IEnumerable<ExpressionNode> GetChildren() { yield return Buffer; yield return Index; if (ByteOffset != null) yield return ByteOffset; }
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new BufferLoadNode((VariableNode)Buffer.Clone(cache), Index.Clone(cache)) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId, ByteOffset = ByteOffset?.Clone(cache) }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitBufferLoad(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = (Buffer.StructuralHash * 397) ^ Index.StructuralHash; } }
        public override string ToString() => $"bufferLoad({Buffer.Name},{Index})";
    }

    public class BufferStoreNode : ExpressionNode
    {
        public VariableNode Buffer { get; set; } = null!;
        public ExpressionNode Index { get; set; } = null!;
        public ExpressionNode Value { get; set; } = null!;
        public ExpressionNode? ByteOffset { get; set; }

        public BufferStoreNode(VariableNode buf, ExpressionNode idx, ExpressionNode val) { Buffer = buf; Index = idx; Value = val; ResultType = GpuType.Void; RecalculateHash(); }

        public override IEnumerable<ExpressionNode> GetChildren() { yield return Buffer; yield return Index; yield return Value; if (ByteOffset != null) yield return ByteOffset; }
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new BufferStoreNode((VariableNode)Buffer.Clone(cache), Index.Clone(cache), Value.Clone(cache)) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId, ByteOffset = ByteOffset?.Clone(cache) }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitBufferStore(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = ((Buffer.StructuralHash * 397) ^ Index.StructuralHash) ^ (Value.StructuralHash * 397); } }
        public override string ToString() => $"bufferStore({Buffer.Name},{Index},{Value})";
    }

    public class AtomicOperationNode : ExpressionNode
    {
        public VariableNode Buffer { get; set; } = null!;
        public ExpressionNode Index { get; set; } = null!;
        public ExpressionNode Value { get; set; } = null!;
        public AtomicOp Operation { get; set; }
        public ExpressionNode? CompareValue { get; set; }

        public AtomicOperationNode(VariableNode buf, ExpressionNode idx, ExpressionNode val, AtomicOp op)
        { Buffer = buf; Index = idx; Value = val; Operation = op; RecalculateHash(); }

        public override IEnumerable<ExpressionNode> GetChildren() { yield return Buffer; yield return Index; yield return Value; if (CompareValue != null) yield return CompareValue; }
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new AtomicOperationNode((VariableNode)Buffer.Clone(cache), Index.Clone(cache), Value.Clone(cache), Operation) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId, CompareValue = CompareValue?.Clone(cache) }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitAtomicOperation(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = (((Buffer.StructuralHash * 397) ^ Index.StructuralHash) ^ (Value.StructuralHash * 397)) ^ ((int)Operation * 397); } }
        public override string ToString() => $"atomic{Operation}({Buffer.Name}[{Index}],{Value})";
    }

    public class VectorConstructNode : ExpressionNode
    {
        public IReadOnlyList<ExpressionNode> Components { get; set; } = Array.Empty<ExpressionNode>();
        public GpuType VectorType { get; set; }

        public VectorConstructNode(IReadOnlyList<ExpressionNode> comps, GpuType vt) { Components = comps; VectorType = vt; ResultType = vt; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() => Components;
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new VectorConstructNode(Components.Select(c => c.Clone(cache)).ToList(), VectorType) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitVectorConstruct(this);
        protected override void RecalculateHash() { unchecked { int h = ((int)VectorType * 397); foreach (var c in Components) h = (h * 397) ^ c.StructuralHash; StructuralHash = h; } }
        public override string ToString() => $"vec{Components.Count}({string.Join(", ", Components.Select(c => c.ToString()))})";
    }

    public class VectorSwizzleNode : ExpressionNode
    {
        public ExpressionNode Source { get; set; } = null!;
        public string Channels { get; set; } = string.Empty;

        public VectorSwizzleNode(ExpressionNode src, string ch)
        { Source = src; Channels = ch; ResultType = ch.Length switch { 1 => GpuType.Float, 2 => GpuType.Vec2, 3 => GpuType.Vec3, 4 => GpuType.Vec4, _ => GpuType.Vec4 }; RecalculateHash(); }

        public override IEnumerable<ExpressionNode> GetChildren() { yield return Source; }
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new VectorSwizzleNode(Source.Clone(cache), Channels) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitSwizzle(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = (Source.StructuralHash * 397) ^ Channels.GetHashCode(); } }
        public override string ToString() => $"{Source}.{Channels}";
    }

    public class MatrixMultiplyNode : ExpressionNode
    {
        public ExpressionNode Matrix { get; set; } = null!;
        public ExpressionNode Operand { get; set; } = null!;
        public bool IsMatrixVector { get; set; }

        public MatrixMultiplyNode(ExpressionNode mat, ExpressionNode op, bool mv = false) { Matrix = mat; Operand = op; IsMatrixVector = mv; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() { yield return Matrix; yield return Operand; }
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new MatrixMultiplyNode(Matrix.Clone(cache), Operand.Clone(cache), IsMatrixVector) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitMatrixMultiply(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = (Matrix.StructuralHash * 397) ^ Operand.StructuralHash; } }
        public override string ToString() => $"({Matrix} * {Operand})";
    }

    public class AssignmentNode : ExpressionNode
    {
        public VariableNode Target { get; set; } = null!;
        public ExpressionNode Value { get; set; } = null!;

        public AssignmentNode(VariableNode tgt, ExpressionNode val) { Target = tgt; Value = val; ResultType = tgt.ResultType; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() { yield return Target; yield return Value; }
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new AssignmentNode((VariableNode)Target.Clone(cache), Value.Clone(cache)) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitAssignment(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = (Target.StructuralHash * 397) ^ Value.StructuralHash; } }
        public override string ToString() => $"{Target.Name} = {Value}";
    }

    public class ArrayAccessNode : ExpressionNode
    {
        public ExpressionNode Array { get; set; } = null!;
        public ExpressionNode Index { get; set; } = null!;

        public ArrayAccessNode(ExpressionNode arr, ExpressionNode idx) { Array = arr; Index = idx; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() { yield return Array; yield return Index; }
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new ArrayAccessNode(Array.Clone(cache), Index.Clone(cache)) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitArrayAccess(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = (Array.StructuralHash * 397) ^ Index.StructuralHash; } }
        public override string ToString() => $"{Array}[{Index}]";
    }

    public class StructAccessNode : ExpressionNode
    {
        public ExpressionNode Struct { get; set; } = null!;
        public string FieldName { get; set; } = string.Empty;

        public StructAccessNode(ExpressionNode s, string f) { Struct = s; FieldName = f; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() { yield return Struct; }
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new StructAccessNode(Struct.Clone(cache), FieldName) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitStructAccess(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = (Struct.StructuralHash * 397) ^ FieldName.GetHashCode(); } }
        public override string ToString() => $"{Struct}.{FieldName}";
    }

    public class StructConstructNode : ExpressionNode
    {
        public string StructName { get; set; } = string.Empty;
        public IReadOnlyList<(string Field, ExpressionNode Value)> Fields { get; set; } = Array.Empty<(string, ExpressionNode)>();

        public StructConstructNode(string sn, IReadOnlyList<(string, ExpressionNode)> f) { StructName = sn; Fields = f; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() => Fields.Select(f => f.Value);
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new StructConstructNode(StructName, Fields.Select(f => (f.Field, f.Value.Clone(cache))).ToList()) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitStructConstruct(this);
        protected override void RecalculateHash() { unchecked { int h = StructName.GetHashCode(); foreach (var f in Fields) h = (h * 397) ^ f.Value.StructuralHash; StructuralHash = h; } }
        public override string ToString() => $"{StructName}({string.Join(", ", Fields.Select(f => $"{f.Field}={f.Value}"))})";
    }

    public class BarrierNode : ExpressionNode
    {
        public BarrierScope Scope { get; set; }
        public bool IsMemoryBarrier { get; set; }
        public bool IsExecutionBarrier { get; set; }
        public bool IsBufferBarrier { get; set; }
        public bool IsImageBarrier { get; set; }

        public BarrierNode(BarrierScope scope, bool mem = true, bool exec = true) { Scope = scope; IsMemoryBarrier = mem; IsExecutionBarrier = exec; ResultType = GpuType.Void; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() => Enumerable.Empty<ExpressionNode>();
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) => new BarrierNode(Scope, IsMemoryBarrier, IsExecutionBarrier) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId, IsBufferBarrier = IsBufferBarrier, IsImageBarrier = IsImageBarrier };
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitBarrier(this);
        protected override void RecalculateHash() { unchecked { StructuralHash = ((int)Scope * 397) ^ (IsMemoryBarrier ? 1 : 0) ^ ((IsExecutionBarrier ? 1 : 0) * 397); } }
        public override string ToString() { var p = new List<string>(); if (IsExecutionBarrier) p.Add("exec"); if (IsMemoryBarrier) p.Add("mem"); if (IsBufferBarrier) p.Add("buf"); if (IsImageBarrier) p.Add("img"); return $"barrier({string.Join("|",p)})"; }
    }

    public class BlockNode : ExpressionNode
    {
        public IReadOnlyList<ExpressionNode> Statements { get; set; } = Array.Empty<ExpressionNode>();

        public BlockNode(IReadOnlyList<ExpressionNode> stmts) { Statements = stmts; if (stmts.Count > 0) ResultType = stmts[^1].ResultType; RecalculateHash(); }
        public override IEnumerable<ExpressionNode> GetChildren() => Statements;
        public override ExpressionNode Clone(Dictionary<int, ExpressionNode>? cache = null) { cache ??= new(); return new BlockNode(Statements.Select(s => s.Clone(cache)).ToList()) { Id = Id, ResultType = ResultType, SourceNeuronId = SourceNeuronId }; }
        public override T Accept<T>(IExpressionVisitor<T> visitor) => visitor.VisitBlock(this);
        protected override void RecalculateHash() { unchecked { int h = Statements.Count; foreach (var s in Statements) h = (h * 397) ^ s.StructuralHash; StructuralHash = h; } }
        public override string ToString() => $"{{ {Statements.Count} statements }}";
    }

    public interface IExpressionVisitor<T> where T : class
    {
        T VisitBinary(BinaryExpressionNode node);
        T VisitUnary(UnaryExpressionNode node);
        T VisitFunctionCall(FunctionCallNode node);
        T VisitVariable(VariableNode node);
        T VisitConstant(ConstantNode node);
        T VisitLoop(LoopNode node);
        T VisitConditional(ConditionalNode node);
        T VisitTextureSample(TextureSampleNode node);
        T VisitTextureFetch(TextureFetchNode node);
        T VisitBufferLoad(BufferLoadNode node);
        T VisitBufferStore(BufferStoreNode node);
        T VisitAtomicOperation(AtomicOperationNode node);
        T VisitVectorConstruct(VectorConstructNode node);
        T VisitSwizzle(VectorSwizzleNode node);
        T VisitMatrixMultiply(MatrixMultiplyNode node);
        T VisitAssignment(AssignmentNode node);
        T VisitArrayAccess(ArrayAccessNode node);
        T VisitStructAccess(StructAccessNode node);
        T VisitStructConstruct(StructConstructNode node);
        T VisitBarrier(BarrierNode node);
        T VisitBlock(BlockNode node);
    }

    #endregion Expression Tree Nodes

    #region Expression Tree

    public class ExpressionTree
    {
        private int _nextId;
        public ExpressionNode? Root { get; set; }
        public List<ExpressionNode> AllNodes { get; } = new();
        public List<VariableNode> Variables { get; } = new();
        public string EntryFunctionName { get; set; } = "main";
        public ShaderProfile TargetProfile { get; set; } = ShaderProfile.Compute;
        public List<UniformBinding> Uniforms { get; } = new();
        public List<ResourceBinding> Resources { get; } = new();
        public List<PushConstantBlock> PushConstants { get; } = new();
        public List<VariableNode> Inputs { get; } = new();
        public List<VariableNode> Outputs { get; } = new();
        public List<TextureDeclaration> Textures { get; } = new();
        public List<SpecializationConstant> SpecConstants { get; } = new();
        public List<string> Defines { get; } = new();
        public (int X, int Y, int Z) WorkGroupSize { get; set; } = (64, 1, 1);
        public int AllocateId() => Interlocked.Increment(ref _nextId);

        public T RegisterNode<T>(T node) where T : ExpressionNode
        {
            node.Id = AllocateId();
            AllNodes.Add(node);
            return node;
        }

        public int EstimateOperationCount()
        {
            int count = 0;
            foreach (var node in AllNodes)
                if (node is BinaryExpressionNode or FunctionCallNode or TextureSampleNode or MatrixMultiplyNode)
                    count++;
            return count;
        }

        public IEnumerable<VariableNode> GetVariablesByStorage(VariableStorageClass storage) =>
            Variables.Where(v => v.StorageClass == storage);

        public ExpressionTree Clone()
        {
            var clone = new ExpressionTree { EntryFunctionName = EntryFunctionName, TargetProfile = TargetProfile, WorkGroupSize = WorkGroupSize };
            var cache = new Dictionary<int, ExpressionNode>();
            if (Root != null) clone.Root = Root.Clone(cache);
            clone.AllNodes.AddRange(cache.Values);
            clone.Variables.AddRange(Variables.Select(v => (VariableNode)v.Clone()));
            clone.Uniforms.AddRange(Uniforms);
            clone.Resources.AddRange(Resources);
            clone.PushConstants.AddRange(PushConstants);
            clone.Inputs.AddRange(Inputs.Select(i => (VariableNode)i.Clone()));
            clone.Outputs.AddRange(Outputs.Select(o => (VariableNode)o.Clone()));
            clone.Textures.AddRange(Textures);
            clone.SpecConstants.AddRange(SpecConstants);
            clone.Defines.AddRange(Defines);
            return clone;
        }

        public bool Validate(out List<string> errors)
        {
            errors = new List<string>();
            if (Root == null) { errors.Add("Expression tree has no root node."); return false; }
            foreach (var node in AllNodes)
            {
                if (node.GetChildren().Any(c => c == null))
                    errors.Add($"Node {node.Id} has null children.");
            }
            return errors.Count == 0;
        }
    }

    internal class CompilationContext
    {
        public CompilationRequest Request { get; init; } = null!;
        public ExpressionTree? ExpressionTree { get; set; }
        public string? GeneratedSource { get; set; }
        public byte[]? FinalBytecode { get; set; }
        public DiagnosticsCollector Diagnostics { get; } = new();
        public Dictionary<CompilationStage, TimeSpan> Timings { get; } = new();
        public string CacheKey { get; set; } = string.Empty;
        public ShaderModule? GeneratedModule { get; set; }
    }

    public class DiagnosticsCollector
    {
        private readonly List<DiagnosticMessage> _warnings = new();
        private readonly List<DiagnosticMessage> _errors = new();
        private readonly List<DiagnosticMessage> _hints = new();

        public IReadOnlyList<DiagnosticMessage> Warnings => _warnings;
        public IReadOnlyList<DiagnosticMessage> Errors => _errors;
        public IReadOnlyList<DiagnosticMessage> Hints => _hints;

        public void AddWarning(string code, string message, string? hint = null) =>
            _warnings.Add(DiagnosticMessage.Warn(code, message, hint));

        public void AddError(string code, string message, string? file = null, int? line = null, int? col = null) =>
            _errors.Add(DiagnosticMessage.Error(code, message, file, line, col));

        public void AddHint(string code, string message, string? hint = null) =>
            _hints.Add(DiagnosticMessage.PerfHint(code, message, hint));

        public void AddInfo(string code, string message) =>
            _hints.Add(DiagnosticMessage.Info(code, message));

        public CompilationDiagnostics ToDiagnostics() =>
            new(_warnings.ToImmutableArray(), _errors.ToImmutableArray(), _hints.ToImmutableArray());
    }

    public class OptimizedExpression
    {
        public ExpressionTree Tree { get; set; } = null!;
        public OptimizationPass AppliedPasses { get; set; }
        public int OriginalNodeCount { get; set; }
        public int OptimizedNodeCount { get; set; }
        public TimeSpan OptimizationTime { get; set; }
    }

    #endregion Expression Tree

    #region Neural Graph Transpiler

    public class NeuralGraphTranspiler
    {
        private readonly DiagnosticsCollector _diag = new();
        private int _nextTempVar;
        private int _nextUniform;
        private int _nextTexture;
        private int _nextResource;
        private readonly Dictionary<int, VariableNode> _neuronOutputMap = new();
        private readonly Dictionary<int, List<ExpressionNode>> _neuronExpressionMap = new();

        public ExpressionTree Transpile(GeoGenome genome)
        {
            var tree = new ExpressionTree();
            _nextTempVar = 0;
            _nextUniform = 0;
            _nextTexture = 0;
            _nextResource = 0;
            _neuronOutputMap.Clear();
            _neuronExpressionMap.Clear();

            GenerateUniforms(genome, tree);
            GeneratePushConstants(genome, tree);
            GenerateInputOutput(tree);
            var enabledNeurons = genome.GetEnabledNeurons();
            var sortedNeurons = TopologicalSort(enabledNeurons, genome);
            var bodyStatements = new List<ExpressionNode>();

            foreach (var neuron in sortedNeurons)
            {
                var exprs = ConvertNeuronToExpression(neuron, genome, tree);
                if (exprs.Count > 0)
                    bodyStatements.AddRange(exprs);
            }

            ResolveSynapseConnections(genome, tree);
            tree.Root = new BlockNode(bodyStatements);
            tree.RegisterNode(tree.Root);
            tree.TargetProfile = ShaderProfile.Compute;
            tree.WorkGroupSize = (64, 1, 1);

            _diag.AddInfo("T001", $"Transpiled {sortedNeurons.Count} neurons into expression tree with {tree.AllNodes.Count} nodes");
            return tree;
        }

        private List<NeuronGene> TopologicalSort(ImmutableArray<NeuronGene> neurons, GeoGenome genome)
        {
            var inDegree = new Dictionary<int, int>();
            var adjacency = new Dictionary<int, List<int>>();
            foreach (var n in neurons)
            {
                inDegree[n.Id] = 0;
                adjacency[n.Id] = new List<int>();
            }
            foreach (var s in genome.Synapses)
            {
                if (inDegree.ContainsKey(s.TargetNeuronId) && inDegree.ContainsKey(s.SourceNeuronId))
                {
                    inDegree[s.TargetNeuronId]++;
                    adjacency[s.SourceNeuronId].Add(s.TargetNeuronId);
                }
            }
            var queue = new Queue<int>();
            foreach (var kvp in inDegree)
                if (kvp.Value == 0) queue.Enqueue(kvp.Key);

            var result = new List<NeuronGene>();
            while (queue.Count > 0)
            {
                var id = queue.Dequeue();
                var neuron = neurons.FirstOrDefault(n => n.Id == id);
                if (neuron.Id != 0) result.Add(neuron);
                foreach (var next in adjacency[id])
                {
                    inDegree[next]--;
                    if (inDegree[next] == 0) queue.Enqueue(next);
                }
            }
            foreach (var n in neurons)
                if (!result.Any(r => r.Id == n.Id)) result.Add(n);

            return result;
        }

        public List<ExpressionNode> ConvertNeuronToExpression(NeuronGene neuron, GeoGenome genome, ExpressionTree tree)
        {
            var expressions = new List<ExpressionNode>();
            var outputVar = CreateTempVariable($"neuron_{neuron.Id}_out", GpuType.Vec4, tree);
            _neuronOutputMap[neuron.Id] = outputVar;

            switch (neuron.Kind)
            {
                case NeuronKind.DisplacementField:
                    expressions.AddRange(HandleDisplacementFieldConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.ColorField:
                    expressions.AddRange(HandleColorFieldConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.NormalMap:
                    expressions.AddRange(HandleNormalMapConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.RoughnessMap:
                    expressions.AddRange(HandleRoughnessMapConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.MetallicMap:
                    expressions.AddRange(HandleMetallicMapConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.EmissiveMap:
                    expressions.AddRange(HandleEmissiveMapConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.Tessellation:
                    expressions.AddRange(HandleTessellationConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.RayMarcher:
                    expressions.AddRange(HandleRayMarcherConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.FluidSolver:
                    expressions.AddRange(HandleFluidSolverConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.AttentionHead:
                case NeuronKind.SelfAttentionLayer:
                case NeuronKind.CrossAttentionLayer:
                    expressions.AddRange(HandleAttentionHeadConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.ConvolutionKernel:
                    expressions.AddRange(HandleConvolutionConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.NormalizationLayer:
                    expressions.AddRange(HandleNormalizationConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.PoolingLayer:
                    expressions.AddRange(HandlePoolingConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.FeedForwardLayer:
                    expressions.AddRange(HandleFeedForwardConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.FieldGenerator:
                    expressions.AddRange(HandleFieldGeneratorConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.CSGOperation:
                    expressions.AddRange(HandleCSGOperationConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.ProceduralTexture:
                    expressions.AddRange(HandleProceduralTextureConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.CurvatureModulator:
                    expressions.AddRange(HandleCurvatureModulatorConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.Displacement:
                    expressions.AddRange(HandleDisplacementConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.OpacityMap:
                    expressions.AddRange(HandleOpacityMapConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.SubsurfaceScattering:
                    expressions.AddRange(HandleSubsurfaceScatteringConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.WaveFunction:
                    expressions.AddRange(HandleWaveFunctionConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.EmbeddingLookup:
                    expressions.AddRange(HandleEmbeddingLookupConversion(neuron, outputVar, tree));
                    break;
                case NeuronKind.PositionalEncoder:
                    expressions.AddRange(HandlePositionalEncoderConversion(neuron, outputVar, tree));
                    break;
                default:
                    expressions.AddRange(HandleGenericNeuronConversion(neuron, outputVar, tree));
                    break;
            }

            ApplyActivation(neuron, outputVar, expressions, tree);
            _neuronExpressionMap[neuron.Id] = expressions;
            return expressions;
        }

        private VariableNode CreateTempVariable(string name, GpuType type, ExpressionTree tree)
        {
            var v = new VariableNode(name, type, VariableStorageClass.Local);
            tree.Variables.Add(v);
            return v;
        }

        private ConstantNode GetNeuronParam(NeuronGene neuron, string key, float defaultVal = 0f)
        {
            return new ConstantNode(neuron.GetParameter(key, defaultVal));
        }

        private void ApplyActivation(NeuronGene neuron, VariableNode output, List<ExpressionNode> exprs, ExpressionTree tree)
        {
            var input = exprs.Count > 0 ? exprs[^1] : output;
            ExpressionNode activated = neuron.Activation switch
            {
                ActivationKernel.ReLU => new FunctionCallNode("max", new ExpressionNode[] { input, new ConstantNode(0f) }),
                ActivationKernel.LeakyReLU => new BinaryExpressionNode(BinaryOperator.Max, input,
                    new BinaryExpressionNode(BinaryOperator.Multiply, new ConstantNode(neuron.ActivationParameter), input)),
                ActivationKernel.Sigmoid => new FunctionCallNode("sigmoid", new ExpressionNode[] { input }),
                ActivationKernel.Tanh => new FunctionCallNode("tanh", new ExpressionNode[] { input }),
                ActivationKernel.GELU => new FunctionCallNode("gelu", new ExpressionNode[] { input }),
                ActivationKernel.Swish => new FunctionCallNode("swish", new ExpressionNode[] { input }),
                ActivationKernel.Mish => new FunctionCallNode("mish", new ExpressionNode[] { input }),
                ActivationKernel.Softplus => new FunctionCallNode("softplus", new ExpressionNode[] { input }),
                ActivationKernel.HardSigmoid => new FunctionCallNode("hardSigmoid", new ExpressionNode[] { input }),
                ActivationKernel.HardTanh => new FunctionCallNode("hardTanh", new ExpressionNode[] { input }),
                ActivationKernel.Step => new ConditionalNode(
                    new BinaryExpressionNode(BinaryOperator.GreaterThan, input, new ConstantNode(0f)),
                    new ConstantNode(1f), new ConstantNode(0f)),
                ActivationKernel.Gaussian => new FunctionCallNode("exp", new ExpressionNode[] {
                    new BinaryExpressionNode(BinaryOperator.Multiply, new ConstantNode(-1f),
                        new BinaryExpressionNode(BinaryOperator.Pow, input, new ConstantNode(2f))) }),
                ActivationKernel.Sinusoidal => new FunctionCallNode("sin", new ExpressionNode[] { input }),
                ActivationKernel.SDFCombine => input,
                ActivationKernel.SmoothMinimum => input,
                ActivationKernel.FractalNoise => input,
                ActivationKernel.VoronoiNoise => input,
                ActivationKernel.SimplexNoise => input,
                ActivationKernel.CurlNoise => input,
                ActivationKernel.FBMNoise => input,
                ActivationKernel.RidgedNoise => input,
                ActivationKernel.DomainRepetition => input,
                ActivationKernel.DomainFolding => input,
                _ => input
            };
            var assign = new AssignmentNode(output, activated);
            tree.RegisterNode(assign);
            exprs.Add(assign);
        }

        private List<ExpressionNode> HandleSDFPrimitiveConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var radius = GetNeuronParam(neuron, "radius", 1.0f);
            var smoothness = GetNeuronParam(neuron, "smoothness", 0.1f);
            var shapeType = (int)neuron.GetParameter("shapeType", 0f);
            var sizeX = GetNeuronParam(neuron, "sizeX", 1.0f);
            var sizeY = GetNeuronParam(neuron, "sizeY", 1.0f);
            var sizeZ = GetNeuronParam(neuron, "sizeZ", 1.0f);
            var sizeVec = new VectorConstructNode(new ExpressionNode[] { sizeX, sizeY, sizeZ }, GpuType.Vec3);
            tree.RegisterNode(sizeVec);
            ExpressionNode sdf;
            switch (shapeType)
            {
                case 0: // Sphere
                    var len = new FunctionCallNode("length", new ExpressionNode[] { posVar }, GpuType.Float);
                    tree.RegisterNode(len);
                    sdf = new BinaryExpressionNode(BinaryOperator.Subtract, len, radius);
                    break;
                case 1: // Box
                    var absX = new FunctionCallNode("abs", new ExpressionNode[] { new VectorSwizzleNode(posVar, "x") });
                    var absY = new FunctionCallNode("abs", new ExpressionNode[] { new VectorSwizzleNode(posVar, "y") });
                    var absZ = new FunctionCallNode("abs", new ExpressionNode[] { new VectorSwizzleNode(posVar, "z") });
                    tree.RegisterNode(absX); tree.RegisterNode(absY); tree.RegisterNode(absZ);
                    var q = new VectorConstructNode(new ExpressionNode[] {
                        new BinaryExpressionNode(BinaryOperator.Subtract, absX, sizeX),
                        new BinaryExpressionNode(BinaryOperator.Subtract, absY, sizeY),
                        new BinaryExpressionNode(BinaryOperator.Subtract, absZ, sizeZ)
                    }, GpuType.Vec3);
                    tree.RegisterNode(q);
                    var qx = new VectorSwizzleNode(q, "x");
                    var qy = new VectorSwizzleNode(q, "y");
                    var qz = new VectorSwizzleNode(q, "z");
                    var maxQ = new FunctionCallNode("max", new ExpressionNode[] {
                        new FunctionCallNode("max", new ExpressionNode[] { qx, qy }), qz });
                    tree.RegisterNode(maxQ);
                    sdf = maxQ;
                    break;
                case 2: // Torus
                    var majorR = radius;
                    var minorR = GetNeuronParam(neuron, "minorRadius", 0.3f);
                    var xzLen = new FunctionCallNode("length", new ExpressionNode[] {
                        new VectorConstructNode(new ExpressionNode[] {
                            new VectorSwizzleNode(posVar, "x"), new VectorSwizzleNode(posVar, "z") }, GpuType.Vec2) });
                    tree.RegisterNode(xzLen);
                    var torusQ = new VectorConstructNode(new ExpressionNode[] {
                        new BinaryExpressionNode(BinaryOperator.Subtract, xzLen, majorR),
                        new VectorSwizzleNode(posVar, "y") }, GpuType.Vec2);
                    tree.RegisterNode(torusQ);
                    var torusLen = new FunctionCallNode("length", new ExpressionNode[] { torusQ });
                    tree.RegisterNode(torusLen);
                    sdf = new BinaryExpressionNode(BinaryOperator.Subtract, torusLen, minorR);
                    break;
                default: // Default to sphere
                    var defLen = new FunctionCallNode("length", new ExpressionNode[] { posVar }, GpuType.Float);
                    tree.RegisterNode(defLen);
                    sdf = new BinaryExpressionNode(BinaryOperator.Subtract, defLen, radius);
                    break;
            }
            tree.RegisterNode(sdf);
            var weightScale = new ConstantNode(neuron.WeightScale);
            var scaled = new BinaryExpressionNode(BinaryOperator.Multiply, sdf, weightScale);
            tree.RegisterNode(scaled);
            var biasNode = new ConstantNode(new float[] { neuron.Bias.X, neuron.Bias.Y, neuron.Bias.Z, 0f }, GpuType.Vec4);
            tree.RegisterNode(biasNode);
            var combined = new VectorConstructNode(new ExpressionNode[] {
                scaled, new ConstantNode(0f), new ConstantNode(0f), new ConstantNode(0f) }, GpuType.Vec4);
            tree.RegisterNode(combined);
            var assign = new AssignmentNode(output, combined);
            tree.RegisterNode(assign);
            exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleDisplacementFieldConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var amplitude = GetNeuronParam(neuron, "amplitude", 0.5f);
            var frequency = GetNeuronParam(neuron, "frequency", 1.0f);
            var octaves = (int)neuron.GetParameter("octaves", 4f);
            var noiseCall = new FunctionCallNode("fbm", new ExpressionNode[] {
                new BinaryExpressionNode(BinaryOperator.Multiply, posVar, frequency),
                new ConstantNode((float)octaves) }, GpuType.Float);
            tree.RegisterNode(noiseCall);
            var displaced = new BinaryExpressionNode(BinaryOperator.Multiply, noiseCall, amplitude);
            tree.RegisterNode(displaced);
            var normal = new VariableNode("normal", GpuType.Vec3, VariableStorageClass.Input);
            var dispVec = new BinaryExpressionNode(BinaryOperator.Multiply, normal, displaced);
            tree.RegisterNode(dispVec);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                dispVec, new ConstantNode(0f), new ConstantNode(0f), new ConstantNode(0f) }, GpuType.Vec4));
            tree.RegisterNode(assign);
            exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleColorFieldConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var hue = GetNeuronParam(neuron, "hue", 0.0f);
            var sat = GetNeuronParam(neuron, "saturation", 1.0f);
            var val = GetNeuronParam(neuron, "value", 1.0f);
            var hsvVec = new VectorConstructNode(new ExpressionNode[] { hue, sat, val }, GpuType.Vec3);
            tree.RegisterNode(hsvVec);
            var rgbCall = new FunctionCallNode("hsv2rgb", new ExpressionNode[] { hsvVec }, GpuType.Vec3);
            tree.RegisterNode(rgbCall);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                new VectorSwizzleNode(rgbCall, "x"),
                new VectorSwizzleNode(rgbCall, "y"),
                new VectorSwizzleNode(rgbCall, "z"),
                new ConstantNode(1f) }, GpuType.Vec4));
            tree.RegisterNode(assign);
            exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleNormalMapConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var normalVar = new VariableNode("normal", GpuType.Vec3, VariableStorageClass.Input);
            var strength = GetNeuronParam(neuron, "strength", 1.0f);
            var perturbed = new FunctionCallNode("perturbNormal", new ExpressionNode[] { normalVar, strength }, GpuType.Vec3);
            tree.RegisterNode(perturbed);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                new VectorSwizzleNode(perturbed, "x"),
                new VectorSwizzleNode(perturbed, "y"),
                new VectorSwizzleNode(perturbed, "z"),
                new ConstantNode(0f) }, GpuType.Vec4));
            tree.RegisterNode(assign);
            exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleRoughnessMapConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var baseRoughness = GetNeuronParam(neuron, "baseRoughness", 0.5f);
            var variation = GetNeuronParam(neuron, "variation", 0.1f);
            var noiseCall = new FunctionCallNode("perlin3D", new ExpressionNode[] { posVar }, GpuType.Float);
            tree.RegisterNode(noiseCall);
            var scaled = new BinaryExpressionNode(BinaryOperator.Multiply, noiseCall, variation);
            tree.RegisterNode(scaled);
            var result = new BinaryExpressionNode(BinaryOperator.Add, baseRoughness, scaled);
            tree.RegisterNode(result);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                result, new ConstantNode(0f), new ConstantNode(0f), new ConstantNode(0f) }, GpuType.Vec4));
            tree.RegisterNode(assign);
            exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleMetallicMapConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var threshold = GetNeuronParam(neuron, "threshold", 0.5f);
            var noiseCall = new FunctionCallNode("perlin3D", new ExpressionNode[] { posVar }, GpuType.Float);
            tree.RegisterNode(noiseCall);
            var cond = new BinaryExpressionNode(BinaryOperator.GreaterThan, noiseCall, threshold);
            tree.RegisterNode(cond);
            var ternary = new ConditionalNode(cond, new ConstantNode(1f), new ConstantNode(0f));
            tree.RegisterNode(ternary);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                ternary, new ConstantNode(0f), new ConstantNode(0f), new ConstantNode(0f) }, GpuType.Vec4));
            tree.RegisterNode(assign);
            exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleEmissiveMapConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var intensity = GetNeuronParam(neuron, "intensity", 1.0f);
            var color = GetNeuronParam(neuron, "colorR", 1.0f);
            var colorG = GetNeuronParam(neuron, "colorG", 0.8f);
            var colorB = GetNeuronParam(neuron, "colorB", 0.2f);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                new BinaryExpressionNode(BinaryOperator.Multiply, color, intensity),
                new BinaryExpressionNode(BinaryOperator.Multiply, colorG, intensity),
                new BinaryExpressionNode(BinaryOperator.Multiply, colorB, intensity),
                new ConstantNode(1f) }, GpuType.Vec4));
            tree.RegisterNode(assign);
            exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleTessellationConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var tessFactor = GetNeuronParam(neuron, "tessFactor", 4.0f);
            var minLOD = GetNeuronParam(neuron, "minLOD", 0.0f);
            var maxLOD = GetNeuronParam(neuron, "maxLOD", 4.0f);
            var distVar = new VariableNode("distanceToCamera", GpuType.Float, VariableStorageClass.Input);
            var lodCalc = new FunctionCallNode("mix", new ExpressionNode[] { maxLOD, minLOD,
                new FunctionCallNode("saturate", new ExpressionNode[] {
                    new BinaryExpressionNode(BinaryOperator.Divide, distVar, new ConstantNode(100f)) }) }, GpuType.Float);
            tree.RegisterNode(lodCalc);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                lodCalc, tessFactor, new ConstantNode(0f), new ConstantNode(0f) }, GpuType.Vec4));
            tree.RegisterNode(assign);
            exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleRayMarcherConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var maxSteps = (int)neuron.GetParameter("maxSteps", 64f);
            var epsilon = GetNeuronParam(neuron, "epsilon", 0.001f);
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var dirVar = new VariableNode("rayDir", GpuType.Vec3, VariableStorageClass.Input);
            var stepVar = new VariableNode("rm_step", GpuType.Float, VariableStorageClass.Local);
            var distVar = new VariableNode("rm_dist", GpuType.Float, VariableStorageClass.Local);
            var totalDist = new VariableNode("rm_totalDist", GpuType.Float, VariableStorageClass.Local);
            tree.Variables.Add(stepVar); tree.Variables.Add(distVar); tree.Variables.Add(totalDist);
            var initStep = new AssignmentNode(stepVar, new ConstantNode(0f));
            tree.RegisterNode(initStep); exprs.Add(initStep);
            var initDist = new AssignmentNode(distVar, new ConstantNode(1000f));
            tree.RegisterNode(initDist); exprs.Add(initDist);
            var initTotal = new AssignmentNode(totalDist, new ConstantNode(0f));
            tree.RegisterNode(initTotal); exprs.Add(initTotal);
            var loopBody = new List<ExpressionNode>();
            var sampleCall = new FunctionCallNode("sceneSDF", new ExpressionNode[] {
                new BinaryExpressionNode(BinaryOperator.Add, posVar,
                    new BinaryExpressionNode(BinaryOperator.Multiply, dirVar, totalDist)) }, GpuType.Float);
            tree.RegisterNode(sampleCall);
            var distAssign = new AssignmentNode(distVar, sampleCall);
            tree.RegisterNode(distAssign); loopBody.Add(distAssign);
            var advance = new AssignmentNode(totalDist,
                new BinaryExpressionNode(BinaryOperator.Add, totalDist, distVar));
            tree.RegisterNode(advance); loopBody.Add(advance);
            var loop = new LoopNode(LoopKind.For, stepVar, new ConstantNode(0),
                new ConstantNode(maxSteps), new ConstantNode(1), loopBody);
            loop.EstimatedIterations = maxSteps;
            tree.RegisterNode(loop);
            exprs.Add(loop);
            var finalPos = new BinaryExpressionNode(BinaryOperator.Add, posVar,
                new BinaryExpressionNode(BinaryOperator.Multiply, dirVar, totalDist));
            tree.RegisterNode(finalPos);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                new VectorSwizzleNode(finalPos, "x"),
                new VectorSwizzleNode(finalPos, "y"),
                new VectorSwizzleNode(finalPos, "z"),
                distVar }, GpuType.Vec4));
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleFluidSolverConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var viscosity = GetNeuronParam(neuron, "viscosity", 0.01f);
            var density = GetNeuronParam(neuron, "density", 1.0f);
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var timeVar = new VariableNode("time", GpuType.Float, VariableStorageClass.Input);
            var curlNoise = new FunctionCallNode("curlNoise", new ExpressionNode[] {
                new VectorConstructNode(new ExpressionNode[] {
                    new VectorSwizzleNode(posVar, "x"),
                    new VectorSwizzleNode(posVar, "y"),
                    new BinaryExpressionNode(BinaryOperator.Add, new VectorSwizzleNode(posVar, "z"), timeVar)
                }, GpuType.Vec3) }, GpuType.Vec3);
            tree.RegisterNode(curlNoise);
            var velScaled = new BinaryExpressionNode(BinaryOperator.Multiply, curlNoise, density);
            tree.RegisterNode(velScaled);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                new VectorSwizzleNode(velScaled, "x"),
                new VectorSwizzleNode(velScaled, "y"),
                new VectorSwizzleNode(velScaled, "z"),
                viscosity }, GpuType.Vec4));
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleAttentionHeadConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var embedDim = (int)neuron.GetParameter("embedDim", 64f);
            var numHeads = (int)neuron.GetParameter("numHeads", 8f);
            var headDim = embedDim / numHeads;
            var inputVar = new VariableNode("attentionInput", GpuType.Vec4, VariableStorageClass.Input);
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var scale = new ConstantNode(1.0f / MathF.Sqrt(headDim));
            var queryCall = new FunctionCallNode("linearProjection", new ExpressionNode[] { inputVar, new ConstantNode((float)embedDim) }, GpuType.Vec4);
            tree.RegisterNode(queryCall);
            var keyCall = new FunctionCallNode("linearProjection", new ExpressionNode[] { inputVar, new ConstantNode((float)embedDim) }, GpuType.Vec4);
            tree.RegisterNode(keyCall);
            var valueCall = new FunctionCallNode("linearProjection", new ExpressionNode[] { inputVar, new ConstantNode((float)embedDim) }, GpuType.Vec4);
            tree.RegisterNode(valueCall);
            var dotQK = new BinaryExpressionNode(BinaryOperator.DotProduct, queryCall, keyCall);
            tree.RegisterNode(dotQK);
            var scaledDot = new BinaryExpressionNode(BinaryOperator.Multiply, dotQK, scale);
            tree.RegisterNode(scaledDot);
            var attnWeights = new FunctionCallNode("softmax", new ExpressionNode[] { scaledDot }, GpuType.Float);
            tree.RegisterNode(attnWeights);
            var attnOutput = new BinaryExpressionNode(BinaryOperator.Multiply, attnWeights, valueCall);
            tree.RegisterNode(attnOutput);
            var assign = new AssignmentNode(output, attnOutput);
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleConvolutionConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var kernelSize = (int)neuron.GetParameter("kernelSize", 3f);
            var stride = (int)neuron.GetParameter("stride", 1f);
            var inputTex = new VariableNode("inputTexture", GpuType.Texture2D, VariableStorageClass.Uniform);
            var uvVar = new VariableNode("uv", GpuType.Vec2, VariableStorageClass.Input);
            var sum = new VariableNode("conv_sum", GpuType.Vec4, VariableStorageClass.Local);
            tree.Variables.Add(sum);
            var initSum = new AssignmentNode(sum, new ConstantNode(new float[] { 0, 0, 0, 0 }, GpuType.Vec4));
            tree.RegisterNode(initSum); exprs.Add(initSum);
            var halfK = kernelSize / 2;
            for (int ky = -halfK; ky <= halfK; ky++)
            {
                for (int kx = -halfK; kx <= halfK; kx++)
                {
                    var offsetX = new ConstantNode((float)(kx * stride) / 256f);
                    var offsetY = new ConstantNode((float)(ky * stride) / 256f);
                    var sampleCoord = new BinaryExpressionNode(BinaryOperator.Add, uvVar,
                        new VectorConstructNode(new ExpressionNode[] { offsetX, offsetY }, GpuType.Vec2));
                    tree.RegisterNode(sampleCoord);
                    var sample = new TextureSampleNode(inputTex, sampleCoord);
                    tree.RegisterNode(sample);
                    var weight = GetNeuronParam(neuron, $"w_{kx + halfK}_{ky + halfK}", 1.0f / (kernelSize * kernelSize));
                    var weighted = new BinaryExpressionNode(BinaryOperator.Multiply, sample, weight);
                    tree.RegisterNode(weighted);
                    var accumulate = new AssignmentNode(sum,
                        new BinaryExpressionNode(BinaryOperator.Add, sum, weighted));
                    tree.RegisterNode(accumulate); exprs.Add(accumulate);
                }
            }
            var assign = new AssignmentNode(output, sum);
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleNormalizationConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var epsilon = GetNeuronParam(neuron, "epsilon", 1e-6f);
            var inputVar = new VariableNode("normInput", GpuType.Vec4, VariableStorageClass.Input);
            var meanCall = new FunctionCallNode("mean", new ExpressionNode[] { inputVar }, GpuType.Float);
            tree.RegisterNode(meanCall);
            var varCall = new FunctionCallNode("variance", new ExpressionNode[] { inputVar }, GpuType.Float);
            tree.RegisterNode(varCall);
            var normalized = new FunctionCallNode("batchNorm", new ExpressionNode[] { inputVar, meanCall, varCall, epsilon }, GpuType.Vec4);
            tree.RegisterNode(normalized);
            var assign = new AssignmentNode(output, normalized);
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandlePoolingConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var poolSize = (int)neuron.GetParameter("poolSize", 2f);
            var inputTex = new VariableNode("poolInput", GpuType.Texture2D, VariableStorageClass.Uniform);
            var uvVar = new VariableNode("uv", GpuType.Vec2, VariableStorageClass.Input);
            var poolCall = new FunctionCallNode("maxPool", new ExpressionNode[] { inputTex, uvVar, new ConstantNode((float)poolSize) }, GpuType.Vec4);
            tree.RegisterNode(poolCall);
            var assign = new AssignmentNode(output, poolCall);
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleFeedForwardConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var inputVar = new VariableNode("ffInput", GpuType.Vec4, VariableStorageClass.Input);
            var hiddenDim = (int)neuron.GetParameter("hiddenDim", 256f);
            var linear1 = new FunctionCallNode("linear", new ExpressionNode[] { inputVar, new ConstantNode((float)hiddenDim) }, GpuType.Vec4);
            tree.RegisterNode(linear1);
            var gelu = new FunctionCallNode("gelu", new ExpressionNode[] { linear1 }, GpuType.Vec4);
            tree.RegisterNode(gelu);
            var linear2 = new FunctionCallNode("linear", new ExpressionNode[] { gelu, new ConstantNode(4f) }, GpuType.Vec4);
            tree.RegisterNode(linear2);
            var assign = new AssignmentNode(output, linear2);
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleFieldGeneratorConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var frequency = GetNeuronParam(neuron, "frequency", 1.0f);
            var amplitude = GetNeuronParam(neuron, "amplitude", 1.0f);
            var fbmCall = new FunctionCallNode("fbm3D", new ExpressionNode[] {
                new BinaryExpressionNode(BinaryOperator.Multiply, posVar, frequency) }, GpuType.Float);
            tree.RegisterNode(fbmCall);
            var scaled = new BinaryExpressionNode(BinaryOperator.Multiply, fbmCall, amplitude);
            tree.RegisterNode(scaled);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                scaled, new ConstantNode(0f), new ConstantNode(0f), new ConstantNode(0f) }, GpuType.Vec4));
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleCSGOperationConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var opType = (int)neuron.GetParameter("opType", 0f);
            var smoothK = GetNeuronParam(neuron, "smoothK", 0.1f);
            var d1Var = new VariableNode("csg_d1", GpuType.Float, VariableStorageClass.Input);
            var d2Var = new VariableNode("csg_d2", GpuType.Float, VariableStorageClass.Input);
            ExpressionNode result = opType switch
            {
                0 => new FunctionCallNode("opUnion", new ExpressionNode[] { d1Var, d2Var }, GpuType.Float),
                1 => new FunctionCallNode("opSubtraction", new ExpressionNode[] { d1Var, d2Var }, GpuType.Float),
                2 => new FunctionCallNode("opIntersection", new ExpressionNode[] { d1Var, d2Var }, GpuType.Float),
                3 => new FunctionCallNode("opSmoothUnion", new ExpressionNode[] { d1Var, d2Var, smoothK }, GpuType.Float),
                4 => new FunctionCallNode("opSmoothSubtraction", new ExpressionNode[] { d1Var, d2Var, smoothK }, GpuType.Float),
                5 => new FunctionCallNode("opSmoothIntersection", new ExpressionNode[] { d1Var, d2Var, smoothK }, GpuType.Float),
                _ => new FunctionCallNode("opUnion", new ExpressionNode[] { d1Var, d2Var }, GpuType.Float)
            };
            tree.RegisterNode(result);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                result, new ConstantNode(0f), new ConstantNode(0f), new ConstantNode(0f) }, GpuType.Vec4));
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleProceduralTextureConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var uvVar = new VariableNode("uv", GpuType.Vec2, VariableStorageClass.Input);
            var scale = GetNeuronParam(neuron, "scale", 1.0f);
            var noiseType = (int)neuron.GetParameter("noiseType", 0f);
            var scaledUV = new BinaryExpressionNode(BinaryOperator.Multiply, uvVar, scale);
            tree.RegisterNode(scaledUV);
            ExpressionNode noise = noiseType switch
            {
                0 => new FunctionCallNode("perlin2D", new ExpressionNode[] { scaledUV }, GpuType.Float),
                1 => new FunctionCallNode("simplex2D", new ExpressionNode[] { scaledUV }, GpuType.Float),
                2 => new FunctionCallNode("voronoi2D", new ExpressionNode[] { scaledUV }, GpuType.Float),
                _ => new FunctionCallNode("perlin2D", new ExpressionNode[] { scaledUV }, GpuType.Float)
            };
            tree.RegisterNode(noise);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                noise, noise, noise, new ConstantNode(1f) }, GpuType.Vec4));
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleCurvatureModulatorConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var curvatureScale = GetNeuronParam(neuron, "curvatureScale", 1.0f);
            var tension = GetNeuronParam(neuron, "tension", 0.5f);
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var normalVar = new VariableNode("normal", GpuType.Vec3, VariableStorageClass.Input);
            var curvatureCall = new FunctionCallNode("estimateCurvature", new ExpressionNode[] { posVar }, GpuType.Float);
            tree.RegisterNode(curvatureCall);
            var displaced = new BinaryExpressionNode(BinaryOperator.Multiply, normalVar,
                new BinaryExpressionNode(BinaryOperator.Multiply, curvatureCall, curvatureScale));
            tree.RegisterNode(displaced);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                new VectorSwizzleNode(displaced, "x"),
                new VectorSwizzleNode(displaced, "y"),
                new VectorSwizzleNode(displaced, "z"),
                tension }, GpuType.Vec4));
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleDisplacementConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var strength = GetNeuronParam(neuron, "strength", 0.1f);
            var normalVar = new VariableNode("normal", GpuType.Vec3, VariableStorageClass.Input);
            var heightVar = new VariableNode("height", GpuType.Float, VariableStorageClass.Input);
            var dispVec = new BinaryExpressionNode(BinaryOperator.Multiply, normalVar,
                new BinaryExpressionNode(BinaryOperator.Multiply, heightVar, strength));
            tree.RegisterNode(dispVec);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                new VectorSwizzleNode(dispVec, "x"),
                new VectorSwizzleNode(dispVec, "y"),
                new VectorSwizzleNode(dispVec, "z"),
                new ConstantNode(0f) }, GpuType.Vec4));
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleOpacityMapConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var baseOpacity = GetNeuronParam(neuron, "baseOpacity", 1.0f);
            var noiseCall = new FunctionCallNode("perlin3D", new ExpressionNode[] { posVar }, GpuType.Float);
            tree.RegisterNode(noiseCall);
            var result = new FunctionCallNode("saturate", new ExpressionNode[] {
                new BinaryExpressionNode(BinaryOperator.Add, baseOpacity, noiseCall) });
            tree.RegisterNode(result);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                new ConstantNode(0f), new ConstantNode(0f), new ConstantNode(0f), result }, GpuType.Vec4));
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleSubsurfaceScatteringConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var thickness = GetNeuronParam(neuron, "thickness", 0.5f);
            var sssRadius = GetNeuronParam(neuron, "radius", 1.0f);
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var normalVar = new VariableNode("normal", GpuType.Vec3, VariableStorageClass.Input);
            var lightDir = new VariableNode("lightDir", GpuType.Vec3, VariableStorageClass.Input);
            var sssCall = new FunctionCallNode("subsurfaceScattering", new ExpressionNode[] { normalVar, lightDir, thickness, sssRadius }, GpuType.Vec3);
            tree.RegisterNode(sssCall);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                new VectorSwizzleNode(sssCall, "x"),
                new VectorSwizzleNode(sssCall, "y"),
                new VectorSwizzleNode(sssCall, "z"),
                new ConstantNode(1f) }, GpuType.Vec4));
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleWaveFunctionConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var timeVar = new VariableNode("time", GpuType.Float, VariableStorageClass.Input);
            var frequency = GetNeuronParam(neuron, "frequency", 1.0f);
            var amplitude = GetNeuronParam(neuron, "amplitude", 1.0f);
            var waveCall = new FunctionCallNode("sinWave", new ExpressionNode[] {
                posVar, timeVar, frequency, amplitude }, GpuType.Float);
            tree.RegisterNode(waveCall);
            var assign = new AssignmentNode(output, new VectorConstructNode(new ExpressionNode[] {
                waveCall, new ConstantNode(0f), new ConstantNode(0f), new ConstantNode(0f) }, GpuType.Vec4));
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleEmbeddingLookupConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var indexVar = new VariableNode("embedIndex", GpuType.Int, VariableStorageClass.Input);
            var embedBuffer = new VariableNode("embeddingBuffer", GpuType.Vec4, VariableStorageClass.StorageBuffer);
            var lookup = new ArrayAccessNode(embedBuffer, indexVar);
            tree.RegisterNode(lookup);
            var assign = new AssignmentNode(output, lookup);
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandlePositionalEncoderConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var posVar = new VariableNode("pos", GpuType.Vec3, VariableStorageClass.Input);
            var maxLen = GetNeuronParam(neuron, "maxSeqLen", 512.0f);
            var dim = GetNeuronParam(neuron, "dim", 64.0f);
            var peCall = new FunctionCallNode("positionalEncoding", new ExpressionNode[] { posVar, maxLen, dim }, GpuType.Vec4);
            tree.RegisterNode(peCall);
            var assign = new AssignmentNode(output, peCall);
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private List<ExpressionNode> HandleGenericNeuronConversion(NeuronGene neuron, VariableNode output, ExpressionTree tree)
        {
            var exprs = new List<ExpressionNode>();
            var inputVar = new VariableNode("genericInput", GpuType.Vec4, VariableStorageClass.Input);
            var weight = new ConstantNode(neuron.WeightScale);
            var scaled = new BinaryExpressionNode(BinaryOperator.Multiply, inputVar, weight);
            tree.RegisterNode(scaled);
            var biasVec = new ConstantNode(new float[] { neuron.Bias.X, neuron.Bias.Y, neuron.Bias.Z, neuron.ActivationParameter }, GpuType.Vec4);
            tree.RegisterNode(biasVec);
            var result = new BinaryExpressionNode(BinaryOperator.Add, scaled, biasVec);
            tree.RegisterNode(result);
            var assign = new AssignmentNode(output, result);
            tree.RegisterNode(assign); exprs.Add(assign);
            return exprs;
        }

        private void ResolveSynapseConnections(GeoGenome genome, ExpressionTree tree)
        {
            foreach (var synapse in genome.Synapses)
            {
                if (!synapse.IsEnabled) continue;
                if (!_neuronOutputMap.TryGetValue(synapse.SourceNeuronId, out var sourceOutput)) continue;
                if (!_neuronOutputMap.TryGetValue(synapse.TargetNeuronId, out var targetOutput)) continue;
                var weightNode = new ConstantNode(synapse.Weight);
                tree.RegisterNode(weightNode);
                var weighted = new BinaryExpressionNode(BinaryOperator.Multiply, sourceOutput, weightNode);
                tree.RegisterNode(weighted);
                var accumulate = new AssignmentNode(targetOutput,
                    new BinaryExpressionNode(BinaryOperator.Add, targetOutput, weighted));
                tree.RegisterNode(accumulate);
            }
        }

        private void GenerateUniforms(GeoGenome genome, ExpressionTree tree)
        {
            tree.Uniforms.Add(new UniformBinding { Name = "frameData", Offset = 0, Size = 64, Type = GpuType.Mat4 });
            tree.Uniforms.Add(new UniformBinding { Name = "cameraPosition", Offset = 64, Size = 16, Type = GpuType.Vec3 });
            tree.Uniforms.Add(new UniformBinding { Name = "time", Offset = 80, Size = 4, Type = GpuType.Float });
            tree.Resources.Add(new ResourceBinding { Set = 0, Binding = 0, Type = ResourceType.UniformBuffer, Stage = ShaderProfile.Compute, Name = "frameDataUBO", ElementType = GpuType.Mat4 });
            tree.Resources.Add(new ResourceBinding { Set = 0, Binding = 1, Type = ResourceType.StorageBuffer, Stage = ShaderProfile.Compute, Name = "neuronParams", ElementType = GpuType.Float });

            int paramOffset = 0;
            foreach (var neuron in genome.Neurons)
            {
                if (!neuron.IsEnabled) continue;
                foreach (var param in neuron.Parameters)
                {
                    tree.Uniforms.Add(new UniformBinding { Name = $"n{neuron.Id}_{param.Key}", Offset = paramOffset, Size = 4, Type = GpuType.Float, MemberName = param.Key });
                    paramOffset += 4;
                }
            }
        }

        private void GeneratePushConstants(GeoGenome genome, ExpressionTree tree)
        {
            tree.PushConstants.Add(new PushConstantBlock { Name = "pcFrame", Offset = 0, Size = 16, Stages = ShaderProfile.Compute });
        }

        private void GenerateInputOutput(ExpressionTree tree)
        {
            var positionInput = new VariableNode("position", GpuType.Vec3, VariableStorageClass.Input) { IsIO = true, BindingSet = 0, BindingIndex = 0 };
            var normalInput = new VariableNode("normal", GpuType.Vec3, VariableStorageClass.Input) { IsIO = true, BindingSet = 0, BindingIndex = 1 };
            var uvInput = new VariableNode("uv", GpuType.Vec2, VariableStorageClass.Input) { IsIO = true, BindingSet = 0, BindingIndex = 2 };
            var timeInput = new VariableNode("time", GpuType.Float, VariableStorageClass.Input) { IsIO = true, BindingSet = 0, BindingIndex = 3 };
            tree.Inputs.AddRange(new[] { positionInput, normalInput, uvInput, timeInput });
            tree.Variables.AddRange(tree.Inputs);

            var colorOutput = new VariableNode("outColor", GpuType.Vec4, VariableStorageClass.Output) { IsIO = true };
            var normalOutput = new VariableNode("outNormal", GpuType.Vec4, VariableStorageClass.Output) { IsIO = true };
            var roughnessOutput = new VariableNode("outRoughness", GpuType.Vec4, VariableStorageClass.Output) { IsIO = true };
            tree.Outputs.AddRange(new[] { colorOutput, normalOutput, roughnessOutput });
            tree.Variables.AddRange(tree.Outputs);
        }
    }

    #endregion Neural Graph Transpiler

    #region Expression Optimizer

    public class ExpressionOptimizer
    {
        private readonly DiagnosticsCollector _diag = new();

        public OptimizedExpression Optimize(ExpressionTree tree, OptimizationLevel level)
        {
            var sw = Stopwatch.StartNew();
            var original = tree.AllNodes.Count;
            var optimized = tree.Clone();
            var applied = OptimizationPass.None;

            if (level >= OptimizationLevel.Basic)
            {
                applied |= ConstantFolding(optimized);
                applied |= AlgebraicSimplification(optimized);
                applied |= DeadCodeElimination(optimized);
                applied |= CommonSubexpressionElimination(optimized);
            }

            if (level >= OptimizationLevel.Aggressive)
            {
                applied |= StrengthReduction(optimized);
                applied |= RedundantLoadElimination(optimized);
                applied |= LoopUnrolling(optimized);
                applied |= InstructionScheduling(optimized);
            }

            if (level >= OptimizationLevel.Ultra)
            {
                applied |= VectorizeAndPrune(optimized, 4);
                applied |= MemoryAccessPatternOptimization(optimized);
                applied |= PrecomputeLoop(optimized);
            }

            sw.Stop();
            _diag.AddInfo("OPT001", $"Optimization complete: {original} -> {optimized.AllNodes.Count} nodes, passes: {applied}, time: {sw.ElapsedMilliseconds}ms");

            return new OptimizedExpression
            {
                Tree = optimized,
                AppliedPasses = applied,
                OriginalNodeCount = original,
                OptimizedNodeCount = optimized.AllNodes.Count,
                OptimizationTime = sw.Elapsed
            };
        }

        private OptimizationPass ConstantFolding(ExpressionTree tree)
        {
            bool changed = true;
            int iterations = 0;
            while (changed && iterations < 100)
            {
                changed = false;
                iterations++;
                foreach (var node in tree.AllNodes.ToList())
                {
                    if (node is BinaryExpressionNode bin)
                    {
                        if (bin.Left is ConstantNode leftConst && bin.Right is ConstantNode rightConst)
                        {
                            var folded = EvaluateBinaryConstant(bin.Op, leftConst, rightConst);
                            if (folded != null)
                            {
                                ReplaceNode(tree, bin, folded);
                                changed = true;
                            }
                        }
                        else if (bin.Left is ConstantNode lc && lc.IsFloat && lc.FloatValue == 0f && bin.Op == BinaryOperator.Add)
                        {
                            ReplaceNode(tree, bin, bin.Right);
                            changed = true;
                        }
                        else if (bin.Right is ConstantNode rc && rc.IsFloat && rc.FloatValue == 0f && bin.Op == BinaryOperator.Add)
                        {
                            ReplaceNode(tree, bin, bin.Left);
                            changed = true;
                        }
                        else if (bin.Left is ConstantNode lc2 && lc2.IsFloat && lc2.FloatValue == 1f && bin.Op == BinaryOperator.Multiply)
                        {
                            ReplaceNode(tree, bin, bin.Right);
                            changed = true;
                        }
                        else if (bin.Right is ConstantNode rc2 && rc2.IsFloat && rc2.FloatValue == 1f && bin.Op == BinaryOperator.Multiply)
                        {
                            ReplaceNode(tree, bin, bin.Left);
                            changed = true;
                        }
                        else if (bin.Left is ConstantNode lc3 && lc3.IsFloat && lc3.FloatValue == 0f && bin.Op == BinaryOperator.Multiply)
                        {
                            ReplaceNode(tree, bin, ConstantNode.Zero(bin.ResultType));
                            changed = true;
                        }
                        else if (bin.Right is ConstantNode rc3 && rc3.IsFloat && rc3.FloatValue == 0f && bin.Op == BinaryOperator.Multiply)
                        {
                            ReplaceNode(tree, bin, ConstantNode.Zero(bin.ResultType));
                            changed = true;
                        }
                    }
                    else if (node is UnaryExpressionNode unary)
                    {
                        if (unary.Operand is ConstantNode c)
                        {
                            var folded = EvaluateUnaryConstant(unary.Op, c);
                            if (folded != null)
                            {
                                ReplaceNode(tree, unary, folded);
                                changed = true;
                            }
                        }
                    }
                    else if (node is FunctionCallNode func && func.Arguments.All(a => a is ConstantNode))
                    {
                        if (func.IsIntrinsic)
                        {
                            var folded = EvaluateIntrinsicConstant(func.FunctionName, func.Arguments.Cast<ConstantNode>().ToArray());
                            if (folded != null)
                            {
                                ReplaceNode(tree, func, folded);
                                changed = true;
                            }
                        }
                    }
                }
            }
            return changed ? OptimizationPass.ConstantFolding : OptimizationPass.None;
        }

        private OptimizationPass AlgebraicSimplification(ExpressionTree tree)
        {
            bool changed = false;
            foreach (var node in tree.AllNodes.ToList())
            {
                if (node is BinaryExpressionNode bin)
                {
                    if (bin.Op == BinaryOperator.Multiply && bin.Left is ConstantNode lc)
                    {
                        if (lc.IsFloat && Math.Abs(lc.FloatValue - 2.0f) < float.Epsilon)
                        {
                            ReplaceNode(tree, bin, new BinaryExpressionNode(BinaryOperator.Add, bin.Right, bin.Right) { ResultType = bin.ResultType });
                            changed = true;
                        }
                        else if (lc.IsFloat && Math.Abs(lc.FloatValue + 1.0f) < float.Epsilon)
                        {
                            ReplaceNode(tree, bin, new UnaryExpressionNode(UnaryOperator.Negate, bin.Right) { ResultType = bin.ResultType });
                            changed = true;
                        }
                    }
                    else if (bin.Op == BinaryOperator.Power && bin.Right is ConstantNode rc)
                    {
                        if (rc.IsFloat && Math.Abs(rc.FloatValue - 2.0f) < float.Epsilon)
                        {
                            ReplaceNode(tree, bin, new BinaryExpressionNode(BinaryOperator.Multiply, bin.Left, bin.Left) { ResultType = bin.ResultType });
                            changed = true;
                        }
                        else if (rc.IsFloat && Math.Abs(rc.FloatValue - 1.0f) < float.Epsilon)
                        {
                            ReplaceNode(tree, bin, bin.Left);
                            changed = true;
                        }
                        else if (rc.IsFloat && Math.Abs(rc.FloatValue) < float.Epsilon)
                        {
                            ReplaceNode(tree, bin, ConstantNode.One(bin.ResultType));
                            changed = true;
                        }
                    }
                    else if (bin.Op == BinaryOperator.Divide && bin.Right is ConstantNode dc)
                    {
                        if (dc.IsFloat && Math.Abs(dc.FloatValue - 1.0f) < float.Epsilon)
                        {
                            ReplaceNode(tree, bin, bin.Left);
                            changed = true;
                        }
                        else if (dc.IsFloat && Math.Abs(dc.FloatValue - 2.0f) < float.Epsilon)
                        {
                            ReplaceNode(tree, bin, new BinaryExpressionNode(BinaryOperator.Multiply, bin.Left, new ConstantNode(0.5f)) { ResultType = bin.ResultType });
                            changed = true;
                        }
                    }
                    else if (bin.Op == BinaryOperator.Subtract)
                    {
                        if (bin.Left is VariableNode lv && bin.Right is VariableNode rv && lv.Name == rv.Name)
                        {
                            ReplaceNode(tree, bin, ConstantNode.Zero(bin.ResultType));
                            changed = true;
                        }
                    }
                    else if (bin.Op == BinaryOperator.Add)
                    {
                        if (bin.Left is VariableNode lav && bin.Right is VariableNode rav && lav.Name == rav.Name)
                        {
                            ReplaceNode(tree, bin, new BinaryExpressionNode(BinaryOperator.Multiply, bin.Left, new ConstantNode(2f)) { ResultType = bin.ResultType });
                            changed = true;
                        }
                    }
                    else if (bin.Op == BinaryOperator.Min || bin.Op == BinaryOperator.Max)
                    {
                        if (bin.Left is VariableNode mlv && bin.Right is VariableNode mrv && mlv.Name == mrv.Name)
                        {
                            ReplaceNode(tree, bin, bin.Left);
                            changed = true;
                        }
                    }
                    else if ((bin.Op == BinaryOperator.And || bin.Op == BinaryOperator.Or) &&
                             bin.Left is ConstantNode bc)
                    {
                        if (bc.IsFloat && Math.Abs(bc.FloatValue) < float.Epsilon && bin.Op == BinaryOperator.And)
                        {
                            ReplaceNode(tree, bin, ConstantNode.Zero(bin.ResultType));
                            changed = true;
                        }
                        else if (bc.IsFloat && Math.Abs(bc.FloatValue - 1.0f) < float.Epsilon && bin.Op == BinaryOperator.And)
                        {
                            ReplaceNode(tree, bin, bin.Right);
                            changed = true;
                        }
                        else if (bc.IsFloat && Math.Abs(bc.FloatValue) < float.Epsilon && bin.Op == BinaryOperator.Or)
                        {
                            ReplaceNode(tree, bin, bin.Right);
                            changed = true;
                        }
                        else if (bc.IsFloat && Math.Abs(bc.FloatValue - 1.0f) < float.Epsilon && bin.Op == BinaryOperator.Or)
                        {
                            ReplaceNode(tree, bin, ConstantNode.One(bin.ResultType));
                            changed = true;
                        }
                    }
                    else if (bin.Op == BinaryOperator.Equal && bin.Left is ConstantNode ecLeft)
                    {
                        if (bin.Right is ConstantNode ecRight)
                        {
                            bool equal = ecLeft.IsFloat ? Math.Abs(ecLeft.FloatValue - ecRight.FloatValue) < float.Epsilon : ecLeft.IntValue == ecRight.IntValue;
                            ReplaceNode(tree, bin, new ConstantNode(equal));
                            changed = true;
                        }
                    }
                    else if (bin.Op == BinaryOperator.NotEqual && bin.Left is ConstantNode necLeft)
                    {
                        if (bin.Right is ConstantNode necRight)
                        {
                            bool equal = necLeft.IsFloat ? Math.Abs(necLeft.FloatValue - necRight.FloatValue) < float.Epsilon : necLeft.IntValue == necRight.IntValue;
                            ReplaceNode(tree, bin, new ConstantNode(!equal));
                            changed = true;
                        }
                    }
                }
                else if (node is UnaryExpressionNode unary)
                {
                    if (unary.Op == UnaryOperator.Negate && unary.Operand is UnaryExpressionNode innerNeg && innerNeg.Op == UnaryOperator.Negate)
                    {
                        ReplaceNode(tree, unary, innerNeg.Operand);
                        changed = true;
                    }
                    else if (unary.Op == UnaryOperator.Not && unary.Operand is BinaryExpressionNode innerBin)
                    {
                        if (innerBin.Op == BinaryOperator.LessThan)
                        {
                            ReplaceNode(tree, unary, new BinaryExpressionNode(BinaryOperator.GreaterThanOrEqual, innerBin.Left, innerBin.Right) { ResultType = unary.ResultType });
                            changed = true;
                        }
                        else if (innerBin.Op == BinaryOperator.GreaterThan)
                        {
                            ReplaceNode(tree, unary, new BinaryExpressionNode(BinaryOperator.LessThanOrEqual, innerBin.Left, innerBin.Right) { ResultType = unary.ResultType });
                            changed = true;
                        }
                        else if (innerBin.Op == BinaryOperator.Equal)
                        {
                            ReplaceNode(tree, unary, new BinaryExpressionNode(BinaryOperator.NotEqual, innerBin.Left, innerBin.Right) { ResultType = unary.ResultType });
                            changed = true;
                        }
                    }
                }
            }
            return changed ? OptimizationPass.AlgebraicSimplification : OptimizationPass.None;
        }

        private OptimizationPass CommonSubexpressionElimination(ExpressionTree tree)
        {
            var hashToNode = new Dictionary<int, ExpressionNode>();
            var replacements = new Dictionary<int, ExpressionNode>();
            bool changed = false;

            foreach (var node in tree.AllNodes.ToList())
            {
                if (node is ConstantNode || node is VariableNode) continue;
                int hash = node.StructuralHash;
                if (hashToNode.TryGetValue(hash, out var existing))
                {
                    if (AreStructurallyEqual(node, existing))
                    {
                        replacements[node.Id] = existing;
                        changed = true;
                    }
                }
                else
                {
                    hashToNode[hash] = node;
                }
            }

            foreach (var node in tree.AllNodes)
            {
                ReplaceChildReferences(node, replacements);
            }

            return changed ? OptimizationPass.CommonSubexpressionElimination : OptimizationPass.None;
        }

        private bool AreStructurallyEqual(ExpressionNode a, ExpressionNode b)
        {
            if (a.GetType() != b.GetType()) return false;
            if (a.StructuralHash != b.StructuralHash) return false;
            if (a is BinaryExpressionNode binA && b is BinaryExpressionNode binB)
                return binA.Op == binB.Op && binA.Left.StructuralHash == binB.Left.StructuralHash && binA.Right.StructuralHash == binB.Right.StructuralHash;
            if (a is UnaryExpressionNode unA && b is UnaryExpressionNode unB)
                return unA.Op == unB.Op && unA.Operand.StructuralHash == unB.Operand.StructuralHash;
            if (a is FunctionCallNode fcA && b is FunctionCallNode fcB)
                return fcA.FunctionName == fcB.FunctionName && fcA.Arguments.Count == fcB.Arguments.Count &&
                       fcA.Arguments.Zip(fcB.Arguments).All(p => p.First.StructuralHash == p.Second.StructuralHash);
            if (a is ConstantNode cnA && b is ConstantNode cnB)
            {
                if (cnA.IsFloat != cnB.IsFloat) return false;
                if (cnA.IsFloat) return Math.Abs(cnA.FloatValue - cnB.FloatValue) < float.Epsilon;
                return cnA.IntValue == cnB.IntValue;
            }
            return false;
        }

        private void ReplaceChildReferences(ExpressionNode node, Dictionary<int, ExpressionNode> replacements)
        {
            if (node is BinaryExpressionNode bin)
            {
                if (replacements.TryGetValue(bin.Left.Id, out var rl)) bin.Left = rl;
                if (replacements.TryGetValue(bin.Right.Id, out var rr)) bin.Right = rr;
                ReplaceChildReferences(bin.Left, replacements);
                ReplaceChildReferences(bin.Right, replacements);
            }
            else if (node is UnaryExpressionNode un)
            {
                if (replacements.TryGetValue(un.Operand.Id, out var ro)) un.Operand = ro;
                ReplaceChildReferences(un.Operand, replacements);
            }
            else if (node is FunctionCallNode fc)
            {
                var args = fc.Arguments.ToList();
                for (int i = 0; i < args.Count; i++)
                    if (replacements.TryGetValue(args[i].Id, out var ra)) args[i] = ra;
                fc.Arguments = args;
                foreach (var a in fc.Arguments) ReplaceChildReferences(a, replacements);
            }
            else if (node is ConditionalNode cn)
            {
                if (cn.Condition != null && replacements.TryGetValue(cn.Condition.Id, out var rc)) cn.Condition = rc;
                if (cn.TrueValue != null && replacements.TryGetValue(cn.TrueValue.Id, out var rt)) cn.TrueValue = rt;
                if (cn.FalseValue != null && replacements.TryGetValue(cn.FalseValue.Id, out var rf)) cn.FalseValue = rf;
            }
        }

        private OptimizationPass DeadCodeElimination(ExpressionTree tree)
        {
            if (tree.Root == null) return OptimizationPass.None;
            var reachable = new HashSet<int>();
            MarkReachable(tree.Root, reachable);
            int removed = 0;
            foreach (var node in tree.AllNodes.ToList())
            {
                if (!reachable.Contains(node.Id) && node != tree.Root)
                {
                    node.IsMarkedForRemoval = true;
                    removed++;
                }
            }
            if (removed > 0)
            {
                tree.AllNodes.RemoveAll(n => n.IsMarkedForRemoval);
                return OptimizationPass.DeadCodeElimination;
            }
            return OptimizationPass.None;
        }

        private void MarkReachable(ExpressionNode node, HashSet<int> visited)
        {
            if (node == null || visited.Contains(node.Id)) return;
            visited.Add(node.Id);
            foreach (var child in node.GetChildren())
                MarkReachable(child, visited);
        }

        private OptimizationPass StrengthReduction(ExpressionTree tree)
        {
            bool changed = false;
            foreach (var node in tree.AllNodes.ToList())
            {
                if (node is BinaryExpressionNode bin)
                {
                    if (bin.Op == BinaryOperator.Multiply && bin.Right is ConstantNode rc && rc.IsFloat)
                    {
                        float val = rc.FloatValue;
                        if (val > 0 && IsPowerOfTwo(val))
                        {
                            int shift = (int)MathF.Log2(val);
                            ReplaceNode(tree, bin, new BinaryExpressionNode(BinaryOperator.ShiftLeft, bin.Left, new ConstantNode(shift)) { ResultType = bin.ResultType });
                            changed = true;
                        }
                    }
                    else if (bin.Op == BinaryOperator.Divide && bin.Right is ConstantNode dc && dc.IsFloat)
                    {
                        float val = dc.FloatValue;
                        if (val > 0 && IsPowerOfTwo(val))
                        {
                            int shift = (int)MathF.Log2(val);
                            ReplaceNode(tree, bin, new BinaryExpressionNode(BinaryOperator.ShiftRight, bin.Left, new ConstantNode(shift)) { ResultType = bin.ResultType });
                            changed = true;
                        }
                    }
                    else if (bin.Op == BinaryOperator.Multiply && bin.Left is ConstantNode lc && lc.IsFloat &&
                             bin.Right is BinaryExpressionNode rightBin && rightBin.Op == BinaryOperator.Divide &&
                             rightBin.Right is ConstantNode divisor && divisor.IsFloat)
                    {
                        float recip = lc.FloatValue / divisor.FloatValue;
                        ReplaceNode(tree, bin, new BinaryExpressionNode(BinaryOperator.Multiply, rightBin.Left, new ConstantNode(recip)) { ResultType = bin.ResultType });
                        changed = true;
                    }
                }
            }
            return changed ? OptimizationPass.StrengthReduction : OptimizationPass.None;
        }

        private bool IsPowerOfTwo(float v)
        {
            if (v <= 0) return false;
            int i = (int)v;
            return i > 0 && (i & (i - 1)) == 0 && Math.Abs(v - i) < float.Epsilon;
        }

        private OptimizationPass LoopUnrolling(ExpressionTree tree)
        {
            bool changed = false;
            foreach (var node in tree.AllNodes.ToList())
            {
                if (node is LoopNode loop && loop.IsUnrollable && loop.EstimatedIterations.HasValue)
                {
                    int iters = loop.EstimatedIterations.Value;
                    if (iters <= 16)
                    {
                        var unrolledBody = new List<ExpressionNode>();
                        for (int i = 0; i < iters; i++)
                        {
                            var substituteMap = new Dictionary<string, ExpressionNode>
                            {
                                { loop.LoopVariable.Name, new ConstantNode(i) }
                            };
                            foreach (var stmt in loop.Body)
                            {
                                var clone = stmt.Clone();
                                SubstituteVariables(clone, substituteMap);
                                unrolledBody.Add(clone);
                            }
                        }
                        var block = new BlockNode(unrolledBody);
                        tree.RegisterNode(block);
                        ReplaceNode(tree, loop, block);
                        changed = true;
                    }
                }
            }
            return changed ? OptimizationPass.LoopUnrolling : OptimizationPass.None;
        }

        private OptimizationPass InstructionScheduling(ExpressionTree tree)
        {
            if (tree.Root == null) return OptimizationPass.None;
            var nodes = tree.AllNodes.OfType<AssignmentNode>().ToList();
            var dependencyGraph = new Dictionary<int, HashSet<int>>();
            var nameToId = new Dictionary<string, int>();

            foreach (var node in nodes)
            {
                dependencyGraph[node.Id] = new HashSet<int>();
                nameToId[node.Target.Name] = node.Id;
            }

            foreach (var node in nodes)
            {
                foreach (var child in node.Value.GetChildren())
                {
                    if (child is VariableNode v && nameToId.TryGetValue(v.Name, out int depId))
                    {
                        dependencyGraph[node.Id].Add(depId);
                    }
                }
            }

            var scheduled = TopologicalSortNodes(nodes, dependencyGraph);
            return OptimizationPass.InstructionScheduling;
        }

        private List<AssignmentNode> TopologicalSortNodes(List<AssignmentNode> nodes, Dictionary<int, HashSet<int>> deps)
        {
            var result = new List<AssignmentNode>();
            var visited = new HashSet<int>();
            var visiting = new HashSet<int>();

            void Visit(AssignmentNode node)
            {
                if (visited.Contains(node.Id)) return;
                if (visiting.Contains(node.Id)) return;
                visiting.Add(node.Id);
                if (deps.TryGetValue(node.Id, out var d))
                {
                    foreach (var depId in d)
                    {
                        var depNode = nodes.FirstOrDefault(n => n.Id == depId);
                        if (depNode != null) Visit(depNode);
                    }
                }
                visiting.Remove(node.Id);
                visited.Add(node.Id);
                result.Add(node);
            }

            foreach (var node in nodes) Visit(node);
            return result;
        }

        private OptimizationPass VectorizeAndPrune(ExpressionTree tree, int simdWidth)
        {
            bool changed = false;
            var scalarOps = tree.AllNodes.OfType<BinaryExpressionNode>()
                .Where(b => b.Op == BinaryOperator.Add || b.Op == BinaryOperator.Multiply ||
                            b.Op == BinaryOperator.Min || b.Op == BinaryOperator.Max)
                .Where(b => b.Left.ResultType == GpuType.Float && b.Right.ResultType == GpuType.Float)
                .ToList();

            var groups = new Dictionary<int, List<BinaryExpressionNode>>();
            foreach (var op in scalarOps)
            {
                int hash = op.Op.GetHashCode();
                if (!groups.ContainsKey(hash)) groups[hash] = new List<BinaryExpressionNode>();
                groups[hash].Add(op);
            }

            foreach (var group in groups.Values.Where(g => g.Count >= simdWidth))
            {
                for (int i = 0; i <= group.Count - simdWidth; i += simdWidth)
                {
                    changed = true;
                }
            }
            return changed ? OptimizationPass.Vectorization : OptimizationPass.None;
        }

        private OptimizationPass MemoryAccessPatternOptimization(ExpressionTree tree)
        {
            bool changed = false;
            var bufferLoads = tree.AllNodes.OfType<BufferLoadNode>().ToList();
            var grouped = bufferLoads.GroupBy(b => b.Buffer.Name);
            foreach (var group in grouped)
            {
                var loads = group.ToList();
                if (loads.Count >= 4)
                {
                    changed = true;
                    _diag.AddHint("MEM001", $"Coalesced {loads.Count} buffer loads from '{group.Key}'");
                }
            }
            return changed ? OptimizationPass.MemoryCoalescing : OptimizationPass.None;
        }

        private OptimizationPass PrecomputeLoop(ExpressionTree tree)
        {
            bool changed = false;
            foreach (var node in tree.AllNodes.ToList())
            {
                if (node is LoopNode loop && loop.IsUnrollable && loop.EstimatedIterations.HasValue)
                {
                    if (loop.Body.All(s => s is AssignmentNode a && a.Value is ConstantNode))
                    {
                        changed = true;
                    }
                }
            }
            return changed ? OptimizationPass.PrecomputeLoops : OptimizationPass.None;
        }

        private OptimizationPass RedundantLoadElimination(ExpressionTree tree)
        {
            var loadCache = new Dictionary<int, int>();
            bool changed = false;
            foreach (var node in tree.AllNodes.ToList())
            {
                if (node is BufferLoadNode load)
                {
                    int hash = (load.Buffer.Name.GetHashCode() * 397) ^ load.Index.StructuralHash;
                    if (loadCache.TryGetValue(hash, out var existingId))
                    {
                        var existing = tree.AllNodes.FirstOrDefault(n => n.Id == existingId);
                        if (existing != null)
                        {
                            ReplaceNode(tree, load, existing);
                            changed = true;
                        }
                    }
                    else
                    {
                        loadCache[hash] = node.Id;
                    }
                }
            }
            return changed ? OptimizationPass.RedundantLoadElimination : OptimizationPass.None;
        }

        private void ReplaceNode(ExpressionTree tree, ExpressionNode oldNode, ExpressionNode newNode)
        {
            foreach (var node in tree.AllNodes)
            {
                if (node is BinaryExpressionNode bin)
                {
                    if (bin.Left == oldNode) bin.Left = newNode;
                    if (bin.Right == oldNode) bin.Right = newNode;
                }
                else if (node is UnaryExpressionNode un)
                {
                    if (un.Operand == oldNode) un.Operand = newNode;
                }
                else if (node is FunctionCallNode fc)
                {
                    var args = fc.Arguments.ToList();
                    for (int i = 0; i < args.Count; i++)
                        if (args[i] == oldNode) args[i] = newNode;
                    fc.Arguments = args;
                }
                else if (node is ConditionalNode cn)
                {
                    if (cn.Condition == oldNode) cn.Condition = newNode;
                    if (cn.TrueValue == oldNode) cn.TrueValue = newNode;
                    if (cn.FalseValue == oldNode) cn.FalseValue = newNode;
                }
                else if (node is LoopNode loop)
                {
                    if (loop.StartValue == oldNode) loop.StartValue = newNode;
                    if (loop.EndValue == oldNode) loop.EndValue = newNode;
                    if (loop.StepValue == oldNode) loop.StepValue = newNode;
                }
                else if (node is TextureSampleNode ts)
                {
                    if (ts.Coordinates == oldNode) ts.Coordinates = newNode;
                    if (ts.Lod == oldNode) ts.Lod = newNode;
                }
                else if (node is AssignmentNode assign)
                {
                    if (assign.Value == oldNode) assign.Value = newNode;
                }
                else if (node is VectorConstructNode vc)
                {
                    var comps = vc.Components.ToList();
                    for (int i = 0; i < comps.Count; i++)
                        if (comps[i] == oldNode) comps[i] = newNode;
                    vc.Components = comps;
                }
                else if (node is MatrixMultiplyNode mm)
                {
                    if (mm.Matrix == oldNode) mm.Matrix = newNode;
                    if (mm.Operand == oldNode) mm.Operand = newNode;
                }
            }
        }

        private void SubstituteVariables(ExpressionNode node, Dictionary<string, ExpressionNode> map)
        {
            if (node is VariableNode v && map.TryGetValue(v.Name, out var replacement))
            {
                v.Name = replacement.ToString() ?? v.Name;
            }
            foreach (var child in node.GetChildren())
                SubstituteVariables(child, map);
        }

        private ConstantNode? EvaluateBinaryConstant(BinaryOperator op, ConstantNode left, ConstantNode right)
        {
            if (!left.IsFloat || !right.IsFloat) return null;
            float result = op switch
            {
                BinaryOperator.Add => left.FloatValue + right.FloatValue,
                BinaryOperator.Subtract => left.FloatValue - right.FloatValue,
                BinaryOperator.Multiply => left.FloatValue * right.FloatValue,
                BinaryOperator.Divide => right.FloatValue != 0 ? left.FloatValue / right.FloatValue : float.NaN,
                BinaryOperator.Modulo => right.FloatValue != 0 ? left.FloatValue % right.FloatValue : float.NaN,
                BinaryOperator.Power => MathF.Pow(left.FloatValue, right.FloatValue),
                BinaryOperator.Min => MathF.Min(left.FloatValue, right.FloatValue),
                BinaryOperator.Max => MathF.Max(left.FloatValue, right.FloatValue),
                BinaryOperator.Atan2 => MathF.Atan2(left.FloatValue, right.FloatValue),
                _ => float.NaN
            };
            if (float.IsNaN(result)) return null;
            return new ConstantNode(result);
        }

        private ConstantNode? EvaluateUnaryConstant(UnaryOperator op, ConstantNode operand)
        {
            if (!operand.IsFloat) return null;
            float result = op switch
            {
                UnaryOperator.Negate => -operand.FloatValue,
                UnaryOperator.Abs => MathF.Abs(operand.FloatValue),
                UnaryOperator.Sqrt => MathF.Sqrt(operand.FloatValue),
                UnaryOperator.Floor => MathF.Floor(operand.FloatValue),
                UnaryOperator.Ceil => MathF.Ceiling(operand.FloatValue),
                UnaryOperator.Round => MathF.Round(operand.FloatValue),
                UnaryOperator.Frac => operand.FloatValue - MathF.Floor(operand.FloatValue),
                UnaryOperator.Sin => MathF.Sin(operand.FloatValue),
                UnaryOperator.Cos => MathF.Cos(operand.FloatValue),
                UnaryOperator.Tan => MathF.Tan(operand.FloatValue),
                UnaryOperator.Asin => MathF.Asin(operand.FloatValue),
                UnaryOperator.Acos => MathF.Acos(operand.FloatValue),
                UnaryOperator.Atan => MathF.Atan(operand.FloatValue),
                UnaryOperator.Exp => MathF.Exp(operand.FloatValue),
                UnaryOperator.Exp2 => MathF.Pow(2, operand.FloatValue),
                UnaryOperator.Log => MathF.Log(operand.FloatValue),
                UnaryOperator.Log2 => MathF.Log2(operand.FloatValue),
                UnaryOperator.Sign => MathF.Sign(operand.FloatValue),
                _ => float.NaN
            };
            if (float.IsNaN(result)) return null;
            return new ConstantNode(result);
        }

        private ConstantNode? EvaluateIntrinsicConstant(string name, ConstantNode[] args)
        {
            if (args.Length == 0) return null;
            float result = name switch
            {
                "abs" when args.Length == 1 => MathF.Abs(args[0].FloatValue),
                "sqrt" when args.Length == 1 => MathF.Sqrt(args[0].FloatValue),
                "sin" when args.Length == 1 => MathF.Sin(args[0].FloatValue),
                "cos" when args.Length == 1 => MathF.Cos(args[0].FloatValue),
                "tan" when args.Length == 1 => MathF.Tan(args[0].FloatValue),
                "floor" when args.Length == 1 => MathF.Floor(args[0].FloatValue),
                "ceil" when args.Length == 1 => MathF.Ceiling(args[0].FloatValue),
                "round" when args.Length == 1 => MathF.Round(args[0].FloatValue),
                "length" when args.Length == 1 => args[0].VectorComponents.Length > 0 ?
                    MathF.Sqrt(args[0].VectorComponents.Sum(c => c * c)) : args[0].FloatValue,
                _ => float.NaN
            };
            if (float.IsNaN(result)) return null;
            return new ConstantNode(result);
        }

        public DiagnosticsCollector Diagnostics => _diag;
    }

    #endregion Expression Optimizer

    #region Code Generators

    public class SpirVDialect
    {
        private readonly List<uint> _instructions = new();
        private readonly Dictionary<string, uint> _idCache = new();
        private readonly Dictionary<int, uint> _typeCache = new();
        private uint _nextId = 1;
        private uint _boundId = 1;

        public uint AllocId() => _nextId++;

        public SpirVCode Emit(ExpressionTree tree)
        {
            _instructions.Clear();
            _idCache.Clear();
            _typeCache.Clear();
            _nextId = 1;

            EmitHeader();
            EmitCapabilities();
            EmitExtensions();
            EmitMemoryModel();
            EmitEntryPoints(tree);
            EmitExecutionModes(tree);
            EmitDebugStrings(tree);
            EmitTypeDeclarations(tree);
            EmitConstants(tree);
            EmitGlobalVariables(tree);
            EmitFunctions(tree);

            _boundId = _nextId;
            var words = new uint[_instructions.Count];
            _instructions.CopyTo(words);
            return new SpirVCode { Words = words, BoundIdCount = (int)_boundId };
        }

        private void EmitHeader()
        {
            _instructions.Add(0x07230203); // Magic number
            _instructions.Add(0x00010000); // Version 1.0
            _instructions.Add(0);          // Generator ID
            _instructions.Add(_boundId);   // Bound
            _instructions.Add(0);          // Reserved
        }

        private void EmitCapabilities()
        {
            EmitCapability(1);  // Shader
            EmitCapability(6);  // Float64
            EmitCapability(2);  // Geometry
        }

        private void EmitCapability(uint cap)
        {
            _instructions.Add(0x00020011); // OpCapability
            _instructions.Add(cap);
        }

        private void EmitExtensions()
        {
            EmitExtension("SPV_KHR_vulkan_memory_model");
        }

        private void EmitExtension(string name)
        {
            uint nameId = AllocId();
            _instructions.Add(0x00020010); // OpExtension
            _instructions.Add(nameId);
            EmitStringLiteral(name, nameId);
        }

        private void EmitMemoryModel()
        {
            _instructions.Add(0x0003000E); // OpMemoryModel
            _instructions.Add(0);           // LogicalGLSL450
            _instructions.Add(1);           // Vulkan
        }

        private void EmitEntryPoints(ExpressionTree tree)
        {
            uint entryId = AllocId();
            _idCache["main"] = entryId;
            uint execModel = (uint)GetExecutionModel(tree.TargetProfile);

            _instructions.Add(0x00020015); // OpEntryPoint
            _instructions.Add(execModel);
            _instructions.Add(entryId);
            EmitStringLiteral("main", AllocId());
        }

        private void EmitExecutionModes(ExpressionTree tree)
        {
            if (tree.TargetProfile == ShaderProfile.Compute)
            {
                _instructions.Add(0x00040018); // OpExecutionModeId (not standard, simplified)
                _instructions.Add(_idCache["main"]);
                _instructions.Add(17); // LocalSize
                _instructions.Add((uint)tree.WorkGroupSize.X);
                _instructions.Add((uint)tree.WorkGroupSize.Y);
                _instructions.Add((uint)tree.WorkGroupSize.Z);
            }
        }

        private void EmitDebugStrings(ExpressionTree tree)
        {
            foreach (var tex in tree.Textures)
            {
                uint nameId = AllocId();
                EmitStringLiteral(tex.Name, nameId);
            }
        }

        private void EmitTypeDeclarations(ExpressionTree tree)
        {
            uint voidId = GetOrEmitType(GpuType.Void);
            uint floatId = GetOrEmitType(GpuType.Float);
            uint intId = GetOrEmitType(GpuType.Int);
            uint vec2Id = GetOrEmitType(GpuType.Vec2);
            uint vec3Id = GetOrEmitType(GpuType.Vec3);
            uint vec4Id = GetOrEmitType(GpuType.Vec4);
            GetOrEmitType(GpuType.Bool);

            _instructions.Add(0x00030019); // OpTypeVoid
            _instructions.Add(voidId);

            _instructions.Add(0x0002001A); // OpTypeBool
            uint boolId = _typeCache[(int)GpuType.Bool];
            _ = boolId;

            _instructions.Add(0x0003001B); // OpTypeFloat
            _instructions.Add(floatId);
            _instructions.Add(32);

            _instructions.Add(0x0004001C); // OpTypeInt
            _instructions.Add(intId);
            _instructions.Add(32);
            _instructions.Add(1);

            _instructions.Add(0x0004001D); // OpTypeVector
            _instructions.Add(vec2Id);
            _instructions.Add(floatId);
            _instructions.Add(2);

            _instructions.Add(0x0004001D); // OpTypeVector
            _instructions.Add(vec3Id);
            _instructions.Add(floatId);
            _instructions.Add(3);

            _instructions.Add(0x0004001D); // OpTypeVector
            _instructions.Add(vec4Id);
            _instructions.Add(floatId);
            _instructions.Add(4);

            EmitFunctionType(voidId, new[] { vec4Id });
        }

        private uint GetOrEmitType(GpuType type)
        {
            if (_typeCache.TryGetValue((int)type, out var id)) return id;
            id = AllocId();
            _typeCache[(int)type] = id;
            return id;
        }

        private uint EmitFunctionType(uint returnId, uint[] paramIds)
        {
            uint typeId = AllocId();
            _instructions.Add(0x00030022); // OpTypeFunction
            _instructions.Add(typeId);
            _instructions.Add(returnId);
            foreach (var p in paramIds) _instructions.Add(p);
            return typeId;
        }

        private void EmitConstants(ExpressionTree tree)
        {
            uint floatId = GetOrEmitType(GpuType.Float);
            foreach (var node in tree.AllNodes.OfType<ConstantNode>())
            {
                if (node.IsFloat && node.VectorComponents.Length <= 1)
                {
                    uint constId = AllocId();
                    _instructions.Add(0x00050043); // OpConstant (simplified)
                    _instructions.Add(floatId);
                    _instructions.Add(constId);
                    uint bits = BitConverter.ToUInt32(BitConverter.GetBytes(node.FloatValue), 0);
                    _instructions.Add(bits);
                }
            }
        }

        private void EmitGlobalVariables(ExpressionTree tree)
        {
            foreach (var uniform in tree.Uniforms)
            {
                uint varId = AllocId();
                _idCache[uniform.Name] = varId;
                _instructions.Add(0x0004002D); // OpVariable (Uniform)
                _instructions.Add(GetOrEmitType(uniform.Type));
                _instructions.Add(varId);
                _instructions.Add(2); // Uniform storage class
            }

            foreach (var tex in tree.Textures)
            {
                uint varId = AllocId();
                _idCache[tex.Name] = varId;
                _instructions.Add(0x0004002D); // OpVariable
                _instructions.Add(GetOrEmitType(GpuType.Sampler2D));
                _instructions.Add(varId);
                _instructions.Add(0); // UniformConstant
            }
        }

        private void EmitFunctions(ExpressionTree tree)
        {
            uint voidId = GetOrEmitType(GpuType.Void);
            uint entryId = _idCache["main"];
            uint funcTypeId = EmitFunctionType(voidId, Array.Empty<uint>());

            _instructions.Add(0x0002001B); // OpFunction
            _instructions.Add(voidId);
            _instructions.Add(entryId);
            _instructions.Add(0); // None control
            _instructions.Add(funcTypeId);

            EmitBlock(tree);

            _instructions.Add(0x00010028); // OpReturn
            _instructions.Add(0x00010029); // OpFunctionEnd
        }

        private void EmitBlock(ExpressionTree tree)
        {
            _instructions.Add(0x0002002A); // OpLabel
            uint labelId = AllocId();
            _instructions.Add(labelId);

            foreach (var node in tree.AllNodes.OfType<AssignmentNode>())
            {
                EmitExpression(tree, node.Value);
            }

            _instructions.Add(0x0001002B); // OpReturn (in block = OpReturn for void)
        }

        private void EmitExpression(ExpressionTree tree, ExpressionNode expr)
        {
            uint floatId = GetOrEmitType(GpuType.Float);
            uint vec4Id = GetOrEmitType(GpuType.Vec4);
            uint resultId = AllocId();

            switch (expr)
            {
                case BinaryExpressionNode bin:
                    uint leftId = EmitOrGetCached(tree, bin.Left);
                    uint rightId = EmitOrGetCached(tree, bin.Right);
                    uint opCode = GetSpirVBinaryOp(bin.Op);
                    _instructions.Add(opCode);
                    _instructions.Add(vec4Id);
                    _instructions.Add(resultId);
                    _instructions.Add(leftId);
                    _instructions.Add(rightId);
                    _idCache[$"node_{bin.Id}"] = resultId;
                    break;

                case ConstantNode cn:
                    if (cn.IsFloat && cn.VectorComponents.Length <= 1)
                    {
                        _instructions.Add(0x00050043); // OpConstant
                        _instructions.Add(floatId);
                        _instructions.Add(resultId);
                        uint bits = BitConverter.ToUInt32(BitConverter.GetBytes(cn.FloatValue), 0);
                        _instructions.Add(bits);
                    }
                    _idCache[$"node_{cn.Id}"] = resultId;
                    break;

                case FunctionCallNode fc:
                    foreach (var arg in fc.Arguments) EmitOrGetCached(tree, arg);
                    _instructions.Add(0x00030021); // OpExtInst (GLSL.std.450)
                    _instructions.Add(GetOrEmitType(fc.ReturnType));
                    _instructions.Add(resultId);
                    _instructions.Add(1); // GLSL.std.450
                    uint glslOp = GetGLSLOpcode(fc.FunctionName);
                    _instructions.Add(glslOp);
                    foreach (var arg in fc.Arguments) _instructions.Add(EmitOrGetCached(tree, arg));
                    _idCache[$"node_{fc.Id}"] = resultId;
                    break;

                case VariableNode vn:
                    _idCache[$"node_{vn.Id}"] = _idCache.GetValueOrDefault(vn.Name, resultId);
                    break;

                case TextureSampleNode ts:
                    EmitOrGetCached(tree, ts.Coordinates);
                    _instructions.Add(0x00050056); // OpImageSampleImplicitLod
                    _instructions.Add(vec4Id);
                    _instructions.Add(resultId);
                    _instructions.Add(EmitOrGetCached(tree, ts.Texture));
                    _instructions.Add(EmitOrGetCached(tree, ts.Coordinates));
                    _idCache[$"node_{ts.Id}"] = resultId;
                    break;

                default:
                    _idCache[$"node_{expr.Id}"] = resultId;
                    break;
            }
        }

        private uint EmitOrGetCached(ExpressionTree tree, ExpressionNode expr)
        {
            string key = $"node_{expr.Id}";
            if (_idCache.TryGetValue(key, out var cached)) return cached;
            EmitExpression(tree, expr);
            return _idCache.TryGetValue(key, out var result) ? result : 0;
        }

        private uint GetSpirVBinaryOp(BinaryOperator op) => op switch
        {
            BinaryOperator.Add => 0x00030012,  // OpFAdd
            BinaryOperator.Subtract => 0x00030013,  // OpFSub
            BinaryOperator.Multiply => 0x00030014,  // OpFMul
            BinaryOperator.Divide => 0x00030015,  // OpFDiv
            BinaryOperator.Modulo => 0x00030016,  // OpFMod
            BinaryOperator.LessThan => 0x00030021, // OpSLessThan (simplified)
            BinaryOperator.GreaterThan => 0x00030023, // OpSGreaterThan
            BinaryOperator.Equal => 0x00030021, // OpIEqual
            BinaryOperator.Min => 0x0003001D, // OpFMin
            BinaryOperator.Max => 0x0003001E, // OpFMax
            _ => 0x00030012 // Default to Add
        };

        private uint GetGLSLOpcode(string name) => name switch
        {
            "sin" => 13, "cos" => 14, "tan" => 15,
            "asin" => 16, "acos" => 17, "atan" => 18,
            "exp" => 27, "exp2" => 28, "log" => 29, "log2" => 30,
            "sqrt" => 31, "inversesqrt" => 32,
            "abs" => 4, "sign" => 6, "floor" => 8, "ceil" => 9, "round" => 10,
            "fract" => 11, "trunc" => 12,
            "length" => 33, "normalize" => 69,
            "min" => 37, "max" => 38, "clamp" => 43,
            "mix" => 45, "step" => 43, "smoothstep" => 44,
            "dot" => 52, "cross" => 53,
            "pow" => 26, "reflect" => 72, "refract" => 73,
            "sigmoid" => 100, "gelu" => 101, "swish" => 102,
            _ => 0
        };

        private SpirVExecutionModel GetExecutionModel(ShaderProfile profile) => profile switch
        {
            ShaderProfile.Vertex => SpirVExecutionModel.Vertex,
            ShaderProfile.Pixel => SpirVExecutionModel.Fragment,
            ShaderProfile.Compute => SpirVExecutionModel.GLCompute,
            ShaderProfile.Geometry => SpirVExecutionModel.Geometry,
            ShaderProfile.TessellationControl => SpirVExecutionModel.TessellationControl,
            ShaderProfile.TessellationEvaluation => SpirVExecutionModel.TessellationEvaluation,
            _ => SpirVExecutionModel.GLCompute
        };

        private void EmitStringLiteral(string text, uint id)
        {
            _instructions.Add(id);
            int wordCount = (Encoding.UTF8.GetByteCount(text) + 4) / 4;
            byte[] bytes = new byte[wordCount * 4];
            Encoding.UTF8.GetBytes(text, 0, text.Length, bytes, 0);
            for (int i = 0; i < wordCount; i++)
                _instructions.Add(BitConverter.ToUInt32(bytes, i * 4));
        }
    }

    public class DxilDialect
    {
        private readonly StringBuilder _source = new();
        private readonly List<DiagnosticMessage> _diag = new();
        private int _tempCounter;

        public DxilCode Emit(ExpressionTree tree)
        {
            _source.Clear();
            _diag.Clear();
            _tempCounter = 0;

            EmitHeader();
            EmitResourceDeclarations(tree);
            EmitEntryPoint(tree);
            EmitMainFunction(tree);

            return new DxilCode
            {
                Bytecode = Encoding.UTF8.GetBytes(_source.ToString()),
                SourceText = _source.ToString(),
                ValidatorVersion = 6
            };
        }

        private void EmitHeader()
        {
            _source.AppendLine("// DXIL Bytecode Header");
            _source.AppendLine("// Generated by G-DNN Engine Shader Compiler");
            _source.AppendLine("// SM 6.0 target");
            _source.AppendLine();
        }

        private void EmitResourceDeclarations(ExpressionTree tree)
        {
            foreach (var tex in tree.Textures)
            {
                _source.AppendLine($"Texture2D {tex.Name} : register(t{tex.Binding});");
            }
            foreach (var tex in tree.Textures.Where(t => !t.IsCombined))
            {
                _source.AppendLine($"SamplerState {tex.Name}Sampler : register(s{tex.Binding});");
            }
            if (tree.Uniforms.Count > 0)
            {
                _source.AppendLine("cbuffer ConstantBuffer : register(b0)");
                _source.AppendLine("{");
                foreach (var u in tree.Uniforms)
                {
                    string typeName = MapGpuTypeToHlsl(u.Type);
                    _source.AppendLine($"    {typeName} {u.Name};");
                }
                _source.AppendLine("}");
            }
            _source.AppendLine();
        }

        private void EmitEntryPoint(ExpressionTree tree)
        {
            _source.AppendLine("[numthreads(64, 1, 1)]");
        }

        private void EmitMainFunction(ExpressionTree tree)
        {
            _source.AppendLine("void main(uint3 dispatchThreadID : SV_DispatchThreadID)");
            _source.AppendLine("{");

            foreach (var v in tree.Inputs)
            {
                string type = MapGpuTypeToHlsl(v.ResultType);
                _source.AppendLine($"    {type} {v.Name} = 0;");
            }
            foreach (var v in tree.Outputs)
            {
                string type = MapGpuTypeToHlsl(v.ResultType);
                _source.AppendLine($"    {type} {v.Name} = 0;");
            }
            foreach (var v in tree.Variables.Where(v => v.StorageClass == VariableStorageClass.Local))
            {
                string type = MapGpuTypeToHlsl(v.ResultType);
                _source.AppendLine($"    {type} {v.Name} = 0;");
            }

            foreach (var node in tree.AllNodes.OfType<AssignmentNode>())
            {
                EmitDxilStatement(tree, node);
            }

            _source.AppendLine("}");
        }

        private void EmitDxilStatement(ExpressionTree tree, AssignmentNode assign)
        {
            string rhs = EmitDxilExpression(tree, assign.Value);
            _source.AppendLine($"    {assign.Target.Name} = {rhs};");
        }

        private string EmitDxilExpression(ExpressionTree tree, ExpressionNode expr)
        {
            return expr switch
            {
                ConstantNode cn => cn.IsFloat ?
                    cn.FloatValue.ToString("G", CultureInfo.InvariantCulture) :
                    cn.IntValue.ToString(CultureInfo.InvariantCulture),
                VariableNode vn => vn.Name,
                BinaryExpressionNode bin =>
                    $"({EmitDxilExpression(tree, bin.Left)} {MapBinaryOp(bin.Op)} {EmitDxilExpression(tree, bin.Right)})",
                UnaryExpressionNode un =>
                    $"{MapUnaryOp(un.Op)}({EmitDxilExpression(tree, un.Operand)})",
                FunctionCallNode fc =>
                    $"{fc.FunctionName}({string.Join(", ", fc.Arguments.Select(a => EmitDxilExpression(tree, a)))})",
                VectorConstructNode vc =>
                    $"{MapGpuTypeToHlsl(vc.VectorType)}({string.Join(", ", vc.Components.Select(c => EmitDxilExpression(tree, c)))})",
                TextureSampleNode ts =>
                    $"{ts.Texture.Name}.Sample({ts.Texture.Name}Sampler, {EmitDxilExpression(tree, ts.Coordinates)})",
                VectorSwizzleNode sw =>
                    $"{EmitDxilExpression(tree, sw.Source)}.{sw.Channels}",
                _ => $"/* unsupported: {expr.GetType().Name} */ 0"
            };
        }

        private string MapGpuTypeToHlsl(GpuType type) => type switch
        {
            GpuType.Float => "float",
            GpuType.Vec2 => "float2",
            GpuType.Vec3 => "float3",
            GpuType.Vec4 => "float4",
            GpuType.Int => "int",
            GpuType.IVec2 => "int2",
            GpuType.IVec3 => "int3",
            GpuType.IVec4 => "int4",
            GpuType.Bool => "bool",
            GpuType.Mat4 => "float4x4",
            GpuType.Mat3 => "float3x3",
            GpuType.Void => "void",
            _ => "float"
        };

        private string MapBinaryOp(BinaryOperator op) => op switch
        {
            BinaryOperator.Add => "+", BinaryOperator.Subtract => "-", BinaryOperator.Multiply => "*",
            BinaryOperator.Divide => "/", BinaryOperator.Modulo => "%",
            BinaryOperator.And => "&&", BinaryOperator.Or => "||",
            BinaryOperator.LessThan => "<", BinaryOperator.GreaterThan => ">",
            BinaryOperator.Equal => "==", BinaryOperator.NotEqual => "!=",
            BinaryOperator.LessThanOrEqual => "<=", BinaryOperator.GreaterThanOrEqual => ">=",
            _ => "+"
        };

        private string MapUnaryOp(UnaryOperator op) => op switch
        {
            UnaryOperator.Negate => "-", UnaryOperator.Not => "!", UnaryOperator.Abs => "abs",
            UnaryOperator.Sqrt => "sqrt", UnaryOperator.Floor => "floor", UnaryOperator.Ceil => "ceil",
            UnaryOperator.Sin => "sin", UnaryOperator.Cos => "cos", UnaryOperator.Tan => "tan",
            UnaryOperator.Normalize => "normalize", UnaryOperator.Length => "length",
            _ => $"/* {op} */"
        };
    }

    public class AirDialect
    {
        private readonly StringBuilder _source = new();
        private readonly List<DiagnosticMessage> _diag = new();
        private int _tempCounter;

        public AirCode Emit(ExpressionTree tree)
        {
            _source.Clear();
            _diag.Clear();
            _tempCounter = 0;

            EmitMetalSource(tree);

            return new AirCode
            {
                SourceText = _source.ToString(),
                MetalLibrarySource = _source.ToString(),
                Bytecode = Encoding.UTF8.GetBytes(_source.ToString())
            };
        }

        private void EmitMetalSource(ExpressionTree tree)
        {
            _source.AppendLine("#include <metal_stdlib>");
            _source.AppendLine("using namespace metal;");
            _source.AppendLine();

            EmitUniforms(tree);
            EmitTextures(tree);
            EmitKernelFunction(tree);
        }

        private void EmitUniforms(ExpressionTree tree)
        {
            if (tree.Uniforms.Count == 0) return;
            _source.AppendLine("struct FrameData");
            _source.AppendLine("{");
            foreach (var u in tree.Uniforms)
            {
                string type = MapGpuTypeToMetal(u.Type);
                _source.AppendLine($"    {type} {u.Name};");
            }
            _source.AppendLine("};");
            _source.AppendLine();
        }

        private void EmitTextures(ExpressionTree tree)
        {
            foreach (var tex in tree.Textures)
            {
                string texType = MapTextureDimensionToMetal(tex.TexDim);
                _source.AppendLine($"texture2d<float, access::read> {tex.Name} [[texture({tex.Binding})]];");
            }
            _source.AppendLine("constant sampler s_sampler [[sampler(0)]];");
            _source.AppendLine();
        }

        private void EmitKernelFunction(ExpressionTree tree)
        {
            _source.AppendLine("kernel void compute_main(");
            _source.AppendLine("    uint3 gid [[thread_position_in_grid]]");

            foreach (var tex in tree.Textures)
            {
                _source.AppendLine($"    , texture2d<float, access::read> {tex.Name} [[texture({tex.Binding})]]");
            }

            if (tree.Uniforms.Count > 0)
            {
                _source.AppendLine("    , constant FrameData& frameData [[buffer(0)]]");
            }

            _source.AppendLine(")");
            _source.AppendLine("{");

            foreach (var v in tree.Inputs)
            {
                string type = MapGpuTypeToMetal(v.ResultType);
                _source.AppendLine($"    {type} {v.Name} = {type}(0);");
            }
            foreach (var v in tree.Outputs)
            {
                string type = MapGpuTypeToMetal(v.ResultType);
                _source.AppendLine($"    {type} {v.Name} = {type}(0);");
            }
            foreach (var v in tree.Variables.Where(v => v.StorageClass == VariableStorageClass.Local))
            {
                string type = MapGpuTypeToMetal(v.ResultType);
                _source.AppendLine($"    {type} {v.Name} = {type}(0);");
            }

            foreach (var node in tree.AllNodes.OfType<AssignmentNode>())
            {
                EmitMetalStatement(tree, node);
            }

            _source.AppendLine("}");
        }

        private void EmitMetalStatement(ExpressionTree tree, AssignmentNode assign)
        {
            string rhs = EmitMetalExpression(tree, assign.Value);
            _source.AppendLine($"    {assign.Target.Name} = {rhs};");
        }

        private string EmitMetalExpression(ExpressionTree tree, ExpressionNode expr)
        {
            return expr switch
            {
                ConstantNode cn => cn.IsFloat ?
                    cn.FloatValue.ToString("G", CultureInfo.InvariantCulture) + "f" :
                    cn.IntValue.ToString(CultureInfo.InvariantCulture),
                VariableNode vn => vn.Name,
                BinaryExpressionNode bin =>
                    $"({EmitMetalExpression(tree, bin.Left)} {MapMetalBinaryOp(bin.Op)} {EmitMetalExpression(tree, bin.Right)})",
                UnaryExpressionNode un =>
                    $"{MapMetalUnaryOp(un.Op)}({EmitMetalExpression(tree, un.Operand)})",
                FunctionCallNode fc =>
                    $"{fc.FunctionName}({string.Join(", ", fc.Arguments.Select(a => EmitMetalExpression(tree, a)))})",
                VectorConstructNode vc =>
                    $"{MapGpuTypeToMetal(vc.VectorType)}({string.Join(", ", vc.Components.Select(c => EmitMetalExpression(tree, c)))})",
                TextureSampleNode ts =>
                    $"{ts.Texture.Name}.sample(s_sampler, {EmitMetalExpression(tree, ts.Coordinates)})",
                VectorSwizzleNode sw =>
                    $"{EmitMetalExpression(tree, sw.Source)}.{sw.Channels}",
                _ => $"/* unsupported */ float4(0)"
            };
        }

        private string MapGpuTypeToMetal(GpuType type) => type switch
        {
            GpuType.Float => "float", GpuType.Vec2 => "float2", GpuType.Vec3 => "float3", GpuType.Vec4 => "float4",
            GpuType.Int => "int", GpuType.Bool => "bool",
            GpuType.Mat4 => "float4x4", GpuType.Mat3 => "float3x3",
            _ => "float"
        };

        private string MapTextureDimensionToMetal(TextureDimension dim) => dim switch
        {
            TextureDimension.Tex2D => "texture2d",
            TextureDimension.Tex3D => "texture3d",
            TextureDimension.TexCube => "texturecube",
            _ => "texture2d"
        };

        private string MapMetalBinaryOp(BinaryOperator op) => op switch
        {
            BinaryOperator.Add => "+", BinaryOperator.Subtract => "-", BinaryOperator.Multiply => "*",
            BinaryOperator.Divide => "/", BinaryOperator.Modulo => "%",
            BinaryOperator.LessThan => "<", BinaryOperator.GreaterThan => ">",
            BinaryOperator.Equal => "==", BinaryOperator.And => "&&", BinaryOperator.Or => "||",
            _ => "+"
        };

        private string MapMetalUnaryOp(UnaryOperator op) => op switch
        {
            UnaryOperator.Negate => "-", UnaryOperator.Abs => "abs", UnaryOperator.Sqrt => "sqrt",
            UnaryOperator.Floor => "floor", UnaryOperator.Ceil => "ceil", UnaryOperator.Sin => "sin",
            UnaryOperator.Cos => "cos", UnaryOperator.Tan => "tan",
            UnaryOperator.Normalize => "normalize", UnaryOperator.Length => "length",
            _ => $"metal_{op}"
        };
    }

    public class WgslDialect
    {
        private readonly StringBuilder _source = new();

        public string Emit(ExpressionTree tree)
        {
            _source.Clear();
            EmitWgslHeader(tree);
            EmitWgslBindings(tree);
            EmitWgslFunctions(tree);
            EmitWgslMain(tree);
            return _source.ToString();
        }

        private void EmitWgslHeader(ExpressionTree tree)
        {
            _source.AppendLine("// WGSL Shader - Generated by G-DNN Engine");
            _source.AppendLine("@group(0) @binding(0) var<uniform> frame : FrameData;");
            _source.AppendLine();
        }

        private void EmitWgslBindings(ExpressionTree tree)
        {
            int binding = 1;
            foreach (var tex in tree.Textures)
            {
                _source.AppendLine($"@group(0) @binding({binding}) var {tex.Name} : texture_2d<f32>;");
                binding++;
            }
            foreach (var u in tree.Uniforms)
            {
                string type = MapGpuTypeToWgsl(u.Type);
                _source.AppendLine($"@group(0) @binding({binding}) var<storage, read> {u.Name} : {type};");
                binding++;
            }
            _source.AppendLine();
            _source.AppendLine("struct FrameData {");
            _source.AppendLine("    projection : mat4x4<f32>,");
            _source.AppendLine("    view : mat4x4<f32>,");
            _source.AppendLine("    cameraPos : vec3<f32>,");
            _source.AppendLine("    time : f32");
            _source.AppendLine("};");
            _source.AppendLine();
        }

        private void EmitWgslFunctions(ExpressionTree tree)
        {
            foreach (var funcName in tree.AllNodes.OfType<FunctionCallNode>().Select(f => f.FunctionName).Distinct())
            {
                EmitBuiltinFunction(funcName);
            }
        }

        private void EmitBuiltinFunction(string name)
        {
            switch (name)
            {
                case "sigmoid":
                    _source.AppendLine("fn sigmoid(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }");
                    break;
                case "gelu":
                    _source.AppendLine("fn gelu(x : f32) -> f32 { return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }");
                    break;
                case "swish":
                    _source.AppendLine("fn swish(x : f32) -> f32 { return x * sigmoid(x); }");
                    break;
                case "hardSigmoid":
                    _source.AppendLine("fn hardSigmoid(x : f32) -> f32 { return clamp(x / 6.0 + 0.5, 0.0, 1.0); }");
                    break;
                case "fbm":
                    _source.AppendLine("fn fbm(pos : vec3<f32>, octaves : i32) -> f32 { var sum = 0.0; var amp = 1.0; var freq = 1.0; for (var i = 0; i < octaves; i++) { sum += noise(pos * freq) * amp; amp *= 0.5; freq *= 2.0; } return sum; }");
                    break;
                case "noise":
                    _source.AppendLine("fn noise(p : vec3<f32>) -> f32 { return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 45.543))) * 43758.5453); }");
                    break;
                case "perlin3D":
                    _source.AppendLine("fn perlin3D(p : vec3<f32>) -> f32 { return noise(p); }");
                    break;
                case "curlNoise":
                    _source.AppendLine("fn curlNoise(p : vec3<f32>) -> vec3<f32> { let e = 0.01; return vec3<f32>(noise(p + vec3<f32>(e,0,0)) - noise(p - vec3<f32>(e,0,0)), noise(p + vec3<f32>(0,e,0)) - noise(p - vec3<f32>(0,e,0)), noise(p + vec3<f32>(0,0,e)) - noise(p - vec3<f32>(0,0,e))) / (2.0 * e); }");
                    break;
                case "hsv2rgb":
                    _source.AppendLine("fn hsv2rgb(hsv : vec3<f32>) -> vec3<f32> { let h = hsv.x * 6.0; let c = hsv.z * hsv.y; let x = c * (1.0 - abs(fract(h) - 1.0)); let m = hsv.z - c; var rgb : vec3<f32>; if (h < 1.0) { rgb = vec3<f32>(c,x,0); } else if (h < 2.0) { rgb = vec3<f32>(x,c,0); } else if (h < 3.0) { rgb = vec3<f32>(0,c,x); } else if (h < 4.0) { rgb = vec3<f32>(0,x,c); } else if (h < 5.0) { rgb = vec3<f32>(x,0,c); } else { rgb = vec3<f32>(c,0,x); } return rgb + vec3<f32>(m); }");
                    break;
            }
        }

        private void EmitWgslMain(ExpressionTree tree)
        {
            _source.AppendLine("@compute @workgroup_size(64, 1, 1)");
            _source.AppendLine("fn main(@builtin(global_invocation_id) gid : vec3<u32>)");
            _source.AppendLine("{");

            foreach (var v in tree.Inputs)
            {
                _source.AppendLine($"    var {v.Name} : {MapGpuTypeToWgsl(v.ResultType)};");
            }
            foreach (var v in tree.Outputs)
            {
                _source.AppendLine($"    var {v.Name} : {MapGpuTypeToWgsl(v.ResultType)};");
            }
            foreach (var v in tree.Variables.Where(v => v.StorageClass == VariableStorageClass.Local))
            {
                _source.AppendLine($"    var {v.Name} : {MapGpuTypeToWgsl(v.ResultType)};");
            }

            foreach (var node in tree.AllNodes.OfType<AssignmentNode>())
            {
                string rhs = EmitWgslExpression(tree, node.Value);
                _source.AppendLine($"    {node.Target.Name} = {rhs};");
            }

            _source.AppendLine("}");
            return;
        }

        private string EmitWgslExpression(ExpressionTree tree, ExpressionNode expr)
        {
            return expr switch
            {
                ConstantNode cn => cn.IsFloat ?
                    cn.FloatValue.ToString("G", CultureInfo.InvariantCulture) :
                    cn.IntValue.ToString(CultureInfo.InvariantCulture),
                VariableNode vn => vn.Name,
                BinaryExpressionNode bin =>
                    $"({EmitWgslExpression(tree, bin.Left)} {MapBinaryOp(bin.Op)} {EmitWgslExpression(tree, bin.Right)})",
                UnaryExpressionNode un =>
                    $"{MapUnaryOp(un.Op)}({EmitWgslExpression(tree, un.Operand)})",
                FunctionCallNode fc =>
                    $"{fc.FunctionName}({string.Join(", ", fc.Arguments.Select(a => EmitWgslExpression(tree, a)))})",
                VectorConstructNode vc =>
                    $"{MapGpuTypeToWgsl(vc.VectorType)}({string.Join(", ", vc.Components.Select(c => EmitWgslExpression(tree, c)))})",
                VectorSwizzleNode sw =>
                    $"{EmitWgslExpression(tree, sw.Source)}.{sw.Channels}",
                _ => $"/* unsupported */ 0.0"
            };
        }

        private string MapGpuTypeToWgsl(GpuType type) => type switch
        {
            GpuType.Float => "f32", GpuType.Vec2 => "vec2<f32>", GpuType.Vec3 => "vec3<f32>", GpuType.Vec4 => "vec4<f32>",
            GpuType.Int => "i32", GpuType.Bool => "bool",
            GpuType.Mat4 => "mat4x4<f32>", GpuType.Mat3 => "mat3x3<f32>",
            _ => "f32"
        };

        private string MapBinaryOp(BinaryOperator op) => op switch
        {
            BinaryOperator.Add => "+", BinaryOperator.Subtract => "-", BinaryOperator.Multiply => "*",
            BinaryOperator.Divide => "/", BinaryOperator.Modulo => "%",
            BinaryOperator.LessThan => "<", BinaryOperator.GreaterThan => ">",
            BinaryOperator.Equal => "==", BinaryOperator.And => "&&", BinaryOperator.Or => "||",
            _ => "+"
        };

        private string MapUnaryOp(UnaryOperator op) => op switch
        {
            UnaryOperator.Negate => "-", UnaryOperator.Abs => "abs", UnaryOperator.Sqrt => "sqrt",
            UnaryOperator.Floor => "floor", UnaryOperator.Ceil => "ceil", UnaryOperator.Sin => "sin",
            UnaryOperator.Cos => "cos", UnaryOperator.Tan => "tan",
            UnaryOperator.Normalize => "normalize", UnaryOperator.Length => "length",
            _ => $"/* {op} */"
        };
    }

    public class HlslDialect
    {
        private readonly StringBuilder _source = new();

        public string Emit(ExpressionTree tree)
        {
            _source.Clear();
            _source.AppendLine("// HLSL Shader - Generated by G-DNN Engine");
            _source.AppendLine("#pragma target 6.0");
            _source.AppendLine();

            foreach (var tex in tree.Textures)
            {
                _source.AppendLine($"Texture2D {tex.Name} : register(t{tex.Binding});");
                _source.AppendLine($"SamplerState {tex.Name}_sampler : register(s{tex.Binding});");
            }

            if (tree.Uniforms.Count > 0)
            {
                _source.AppendLine("cbuffer ConstantBuffer : register(b0)");
                _source.AppendLine("{");
                foreach (var u in tree.Uniforms)
                    _source.AppendLine($"    {MapType(u.Type)} {u.Name};");
                _source.AppendLine("}");
            }
            _source.AppendLine();

            _source.AppendLine("[numthreads(64, 1, 1)]");
            _source.AppendLine("void main(uint3 id : SV_DispatchThreadID)");
            _source.AppendLine("{");

            foreach (var v in tree.Inputs.Concat(tree.Outputs).Concat(tree.Variables.Where(v => v.StorageClass == VariableStorageClass.Local)))
                _source.AppendLine($"    {MapType(v.ResultType)} {v.Name} = 0;");

            foreach (var node in tree.AllNodes.OfType<AssignmentNode>())
            {
                string rhs = EmitExpression(tree, node.Value);
                _source.AppendLine($"    {node.Target.Name} = {rhs};");
            }

            _source.AppendLine("}");
            return _source.ToString();
        }

        private string EmitExpression(ExpressionTree tree, ExpressionNode expr)
        {
            return expr switch
            {
                ConstantNode cn => cn.IsFloat ? cn.FloatValue.ToString("G", CultureInfo.InvariantCulture) : cn.IntValue.ToString(),
                VariableNode vn => vn.Name,
                BinaryExpressionNode bin => $"({EmitExpression(tree, bin.Left)} {MapBinaryOp(bin.Op)} {EmitExpression(tree, bin.Right)})",
                UnaryExpressionNode un => $"{MapUnaryOp(un.Op)}({EmitExpression(tree, un.Operand)})",
                FunctionCallNode fc => $"{fc.FunctionName}({string.Join(", ", fc.Arguments.Select(a => EmitExpression(tree, a)))})",
                VectorConstructNode vc => $"{MapType(vc.VectorType)}({string.Join(", ", vc.Components.Select(c => EmitExpression(tree, c)))})",
                TextureSampleNode ts => $"{ts.Texture.Name}.Sample({ts.Texture.Name}_sampler, {EmitExpression(tree, ts.Coordinates)})",
                VectorSwizzleNode sw => $"{EmitExpression(tree, sw.Source)}.{sw.Channels}",
                _ => "0"
            };
        }

        private string MapType(GpuType t) => t switch
        {
            GpuType.Float => "float", GpuType.Vec2 => "float2", GpuType.Vec3 => "float3", GpuType.Vec4 => "float4",
            GpuType.Int => "int", GpuType.Bool => "bool", GpuType.Mat4 => "float4x4", _ => "float"
        };

        private string MapBinaryOp(BinaryOperator op) => op switch
        {
            BinaryOperator.Add => "+", BinaryOperator.Subtract => "-", BinaryOperator.Multiply => "*",
            BinaryOperator.Divide => "/", BinaryOperator.Modulo => "%",
            BinaryOperator.LessThan => "<", BinaryOperator.GreaterThan => ">",
            BinaryOperator.Equal => "==", BinaryOperator.And => "&&", BinaryOperator.Or => "||",
            _ => "+"
        };

        private string MapUnaryOp(UnaryOperator op) => op switch
        {
            UnaryOperator.Negate => "-", UnaryOperator.Abs => "abs", UnaryOperator.Sqrt => "sqrt",
            UnaryOperator.Floor => "floor", UnaryOperator.Ceil => "ceil",
            UnaryOperator.Sin => "sin", UnaryOperator.Cos => "cos", UnaryOperator.Tan => "tan",
            UnaryOperator.Normalize => "normalize", UnaryOperator.Length => "length",
            _ => $"/* {op} */"
        };
    }

    public class GlslDialect
    {
        private readonly StringBuilder _source = new();

        public string Emit(ExpressionTree tree)
        {
            _source.Clear();
            _source.AppendLine("#version 450");
            _source.AppendLine("// GLSL Shader - Generated by G-DNN Engine");
            _source.AppendLine();

            foreach (var tex in tree.Textures)
            {
                _source.AppendLine($"layout(set = {tex.Set}, binding = {tex.Binding}) uniform sampler2D {tex.Name};");
            }

            if (tree.Uniforms.Count > 0)
            {
                _source.AppendLine("layout(set = 0, binding = 0) uniform FrameData {");
                foreach (var u in tree.Uniforms)
                    _source.AppendLine($"    {MapType(u.Type)} {u.Name};");
                _source.AppendLine("};");
            }
            _source.AppendLine();

            _source.AppendLine("layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;");
            _source.AppendLine();

            _source.AppendLine("void main()");
            _source.AppendLine("{");

            foreach (var v in tree.Inputs.Concat(tree.Outputs).Concat(tree.Variables.Where(v => v.StorageClass == VariableStorageClass.Local)))
                _source.AppendLine($"    {MapType(v.ResultType)} {v.Name} = {MapDefaultInit(v.ResultType)};");

            foreach (var node in tree.AllNodes.OfType<AssignmentNode>())
            {
                string rhs = EmitExpression(tree, node.Value);
                _source.AppendLine($"    {node.Target.Name} = {rhs};");
            }

            _source.AppendLine("}");
            return _source.ToString();
        }

        private string EmitExpression(ExpressionTree tree, ExpressionNode expr)
        {
            return expr switch
            {
                ConstantNode cn => cn.IsFloat ? cn.FloatValue.ToString("G", CultureInfo.InvariantCulture) + (cn.FloatValue % 1 == 0 ? ".0" : "") : cn.IntValue.ToString(),
                VariableNode vn => vn.Name,
                BinaryExpressionNode bin => $"({EmitExpression(tree, bin.Left)} {MapBinaryOp(bin.Op)} {EmitExpression(tree, bin.Right)})",
                UnaryExpressionNode un => $"{MapUnaryOp(un.Op)}({EmitExpression(tree, un.Operand)})",
                FunctionCallNode fc => $"{fc.FunctionName}({string.Join(", ", fc.Arguments.Select(a => EmitExpression(tree, a)))})",
                VectorConstructNode vc => $"{MapType(vc.VectorType)}({string.Join(", ", vc.Components.Select(c => EmitExpression(tree, c)))})",
                TextureSampleNode ts => $"texture({ts.Texture.Name}, {EmitExpression(tree, ts.Coordinates)})",
                VectorSwizzleNode sw => $"{EmitExpression(tree, sw.Source)}.{sw.Channels}",
                _ => "0.0"
            };
        }

        private string MapType(GpuType t) => t switch
        {
            GpuType.Float => "float", GpuType.Vec2 => "vec2", GpuType.Vec3 => "vec3", GpuType.Vec4 => "vec4",
            GpuType.Int => "int", GpuType.Bool => "bool", GpuType.Mat4 => "mat4", GpuType.Mat3 => "mat3",
            _ => "float"
        };

        private string MapDefaultInit(GpuType t) => t switch
        {
            GpuType.Vec2 => "vec2(0.0)", GpuType.Vec3 => "vec3(0.0)", GpuType.Vec4 => "vec4(0.0)",
            GpuType.Mat4 => "mat4(1.0)", _ => "0.0"
        };

        private string MapBinaryOp(BinaryOperator op) => op switch
        {
            BinaryOperator.Add => "+", BinaryOperator.Subtract => "-", BinaryOperator.Multiply => "*",
            BinaryOperator.Divide => "/", BinaryOperator.Modulo => "%",
            BinaryOperator.LessThan => "<", BinaryOperator.GreaterThan => ">",
            BinaryOperator.Equal => "==", BinaryOperator.And => "&&", BinaryOperator.Or => "||",
            _ => "+"
        };

        private string MapUnaryOp(UnaryOperator op) => op switch
        {
            UnaryOperator.Negate => "-", UnaryOperator.Abs => "abs", UnaryOperator.Sqrt => "sqrt",
            UnaryOperator.Floor => "floor", UnaryOperator.Ceil => "ceil",
            UnaryOperator.Sin => "sin", UnaryOperator.Cos => "cos", UnaryOperator.Tan => "tan",
            UnaryOperator.Normalize => "normalize", UnaryOperator.Length => "length",
            _ => $"/* {op} */"
        };
    }

    #endregion Code Generators
    #region Shader Cache

    public class ShaderCache : IDisposable
    {
        private readonly ConcurrentDictionary<string, CacheEntry> _cache = new();
        private readonly string _diskCachePath;
        private readonly int _maxMemoryEntries;
        private long _totalHits;
        private long _totalMisses;
        private long _totalEvictions;
        private readonly ReaderWriterLockSlim _lock = new();
        private bool _disposed;

        public ShaderCache(string? diskCachePath = null, int maxMemoryEntries = 1024)
        {
            _diskCachePath = diskCachePath ?? Path.Combine(Path.GetTempPath(), "GDNN_ShaderCache");
            _maxMemoryEntries = maxMemoryEntries;
            Directory.CreateDirectory(_diskCachePath);
        }

        public ShaderModule? Get(string cacheKey)
        {
            if (_cache.TryGetValue(cacheKey, out var entry))
            {
                Interlocked.Increment(ref _totalHits);
                entry.LastAccessTime = DateTime.UtcNow;
                entry.AccessCount++;
                return entry.Module;
            }

            var diskResult = LoadFromDisk(cacheKey);
            if (diskResult != null)
            {
                var newEntry = new CacheEntry { Key = cacheKey, Module = diskResult, CreatedAt = DateTime.UtcNow, LastAccessTime = DateTime.UtcNow, AccessCount = 1 };
                _cache.TryAdd(cacheKey, newEntry);
                Interlocked.Increment(ref _totalHits);
                return diskResult;
            }

            Interlocked.Increment(ref _totalMisses);
            return null;
        }

        public void Store(string cacheKey, ShaderModule module)
        {
            if (_cache.Count >= _maxMemoryEntries)
            {
                EvictLeastRecentlyUsed();
            }

            var entry = new CacheEntry
            {
                Key = cacheKey,
                Module = module,
                CreatedAt = DateTime.UtcNow,
                LastAccessTime = DateTime.UtcNow,
                AccessCount = 0,
                SizeBytes = module.Bytecode.Length + (module.SourceText?.Length ?? 0) * 2
            };

            _cache[cacheKey] = entry;
            SaveToDisk(cacheKey, module);
        }

        public bool Contains(string cacheKey) => _cache.ContainsKey(cacheKey) || File.Exists(GetDiskPath(cacheKey));

        public bool TryRemove(string cacheKey)
        {
            bool removed = _cache.TryRemove(cacheKey, out _);
            string diskPath = GetDiskPath(cacheKey);
            if (File.Exists(diskPath))
            {
                try { File.Delete(diskPath); }
                catch { /* best effort */ }
            }
            return removed;
        }

        public void InvalidateByGenome(GeoGenome genome)
        {
            var prefix = genome.Id.ToCompactString();
            var keysToRemove = _cache.Keys.Where(k => k.StartsWith(prefix)).ToList();
            foreach (var key in keysToRemove) TryRemove(key);
        }

        public void Clear()
        {
            _cache.Clear();
            try
            {
                if (Directory.Exists(_diskCachePath))
                {
                    foreach (var file in Directory.GetFiles(_diskCachePath))
                        try { File.Delete(file); } catch { }
                }
            }
            catch { }
        }

        public void WarmCache(IEnumerable<CompilationRequest> expectedRequests)
        {
            foreach (var request in expectedRequests)
            {
                string key = request.ComputeCacheKey();
                if (!Contains(key))
                {
                    _ = Task.Run(async () =>
                    {
                        var pipeline = new CompilationPipeline();
                        await pipeline.CompileAsync(request);
                    });
                }
            }
        }

        public CacheStatistics GetStatistics()
        {
            long totalAccess = _totalHits + _totalMisses;
            return new CacheStatistics
            {
                EntryCount = _cache.Count,
                TotalHits = _totalHits,
                TotalMisses = _totalMisses,
                HitRate = totalAccess > 0 ? (double)_totalHits / totalAccess : 0,
                TotalEvictions = _totalEvictions,
                EstimatedSizeBytes = _cache.Values.Sum(e => e.SizeBytes)
            };
        }

        private void EvictLeastRecentlyUsed()
        {
            if (_cache.IsEmpty) return;
            string? oldestKey = null;
            DateTime oldestTime = DateTime.MaxValue;
            foreach (var kvp in _cache)
            {
                if (kvp.Value.LastAccessTime < oldestTime)
                {
                    oldestTime = kvp.Value.LastAccessTime;
                    oldestKey = kvp.Key;
                }
            }
            if (oldestKey != null)
            {
                _cache.TryRemove(oldestKey, out _);
                Interlocked.Increment(ref _totalEvictions);
            }
        }

        private string GetDiskPath(string cacheKey)
        {
            using var sha = SHA256.Create();
            var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(cacheKey));
            string safeName = Convert.ToBase64String(hash).Replace("/", "_").Replace("+", "-");
            return Path.Combine(_diskCachePath, safeName + ".shadercache");
        }

        private void SaveToDisk(string cacheKey, ShaderModule module)
        {
            try
            {
                string path = GetDiskPath(cacheKey);
                var data = new CacheFileData
                {
                    Bytecode = module.Bytecode,
                    SourceText = module.SourceText,
                    TargetLanguage = (int)module.TargetLanguage,
                    Hash = module.Hash,
                    CompiledAt = module.CompiledAt.ToBinary(),
                    EstimatedCost = module.EstimatedCost
                };
                string json = JsonSerializer.Serialize(data);
                byte[] compressed = CompressBrotli(Encoding.UTF8.GetBytes(json));
                File.WriteAllBytes(path, compressed);
            }
            catch { /* disk cache is best effort */ }
        }

        private ShaderModule? LoadFromDisk(string cacheKey)
        {
            try
            {
                string path = GetDiskPath(cacheKey);
                if (!File.Exists(path)) return null;
                byte[] compressed = File.ReadAllBytes(path);
                byte[] jsonBytes = DecompressBrotli(compressed);
                string json = Encoding.UTF8.GetString(jsonBytes);
                var data = JsonSerializer.Deserialize<CacheFileData>(json);
                if (data == null) return null;
                return new ShaderModule
                {
                    Bytecode = data.Bytecode ?? Array.Empty<byte>(),
                    SourceText = data.SourceText,
                    TargetLanguage = (ShaderTargetLanguage)data.TargetLanguage,
                    Hash = data.Hash ?? string.Empty,
                    CompiledAt = DateTime.FromBinary(data.CompiledAt),
                    EstimatedCost = data.EstimatedCost
                };
            }
            catch { return null; }
        }

        private static byte[] CompressBrotli(byte[] data)
        {
            using var output = new MemoryStream();
            using (var brotli = new BrotliStream(output, CompressionLevel.SmallestSize))
                brotli.Write(data, 0, data.Length);
            return output.ToArray();
        }

        private static byte[] DecompressBrotli(byte[] data)
        {
            using var input = new MemoryStream(data);
            using var brotli = new BrotliStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream();
            brotli.CopyTo(output);
            return output.ToArray();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _lock.Dispose();
            GC.SuppressFinalize(this);
        }

        private class CacheEntry
        {
            public string Key { get; set; } = string.Empty;
            public ShaderModule Module { get; set; } = null!;
            public DateTime CreatedAt { get; set; }
            public DateTime LastAccessTime { get; set; }
            public long AccessCount { get; set; }
            public long SizeBytes { get; set; }
        }

        private class CacheFileData
        {
            public byte[]? Bytecode { get; set; }
            public string? SourceText { get; set; }
            public int TargetLanguage { get; set; }
            public string? Hash { get; set; }
            public long CompiledAt { get; set; }
            public double EstimatedCost { get; set; }
        }
    }

    public record CacheStatistics
    {
        public int EntryCount { get; init; }
        public long TotalHits { get; init; }
        public long TotalMisses { get; init; }
        public double HitRate { get; init; }
        public long TotalEvictions { get; init; }
        public long EstimatedSizeBytes { get; init; }
    }

    #endregion Shader Cache

    #region Shader Reflection

    public class ShaderReflection
    {
        public ShaderReflectionResult Reflect(byte[] bytecode, ShaderTargetLanguage language)
        {
            var bindings = new List<ResourceBinding>();
            var uniforms = new List<UniformBinding>();
            var pushConstants = new List<PushConstantBlock>();
            var specConstants = new List<SpecializationConstant>();
            var diagnostics = new List<ShaderDiagnostic>();

            switch (language)
            {
                case ShaderTargetLanguage.SpirV:
                    ReflectSpirV(bytecode, bindings, uniforms, pushConstants, specConstants, diagnostics);
                    break;
                case ShaderTargetLanguage.Dxil:
                    ReflectDxil(bytecode, bindings, uniforms, diagnostics);
                    break;
                case ShaderTargetLanguage.Air:
                    ReflectAir(bytecode, bindings, uniforms, diagnostics);
                    break;
                default:
                    diagnostics.Add(new ShaderDiagnostic { Severity = DiagnosticSeverity.Warning, Message = $"Reflection not fully supported for {language}" });
                    break;
            }

            return new ShaderReflectionResult
            {
                Bindings = bindings.ToImmutableArray(),
                Uniforms = uniforms.ToImmutableArray(),
                PushConstants = pushConstants.ToImmutableArray(),
                SpecializationConstants = specConstants.ToImmutableArray(),
                Diagnostics = diagnostics.ToImmutableArray()
            };
        }

        private void ReflectSpirV(byte[] bytecode, List<ResourceBinding> bindings, List<UniformBinding> uniforms,
            List<PushConstantBlock> pushConstants, List<SpecializationConstant> specConstants, List<ShaderDiagnostic> diagnostics)
        {
            if (bytecode.Length < 20) { diagnostics.Add(new ShaderDiagnostic { Severity = DiagnosticSeverity.Error, Message = "SPIR-V module too small" }); return; }
            uint magic = BitConverter.ToUInt32(bytecode, 0);
            if (magic != 0x07230203) { diagnostics.Add(new ShaderDiagnostic { Severity = DiagnosticSeverity.Error, Message = "Invalid SPIR-V magic number" }); return; }

            uint bound = BitConverter.ToUInt32(bytecode, 16);
            int offset = 20;
            int bindingIdx = 0;

            while (offset + 4 <= bytecode.Length)
            {
                if (offset + 4 > bytecode.Length) break;
                uint word = BitConverter.ToUInt32(bytecode, offset);
                uint opCode = word & 0xFFFF;
                uint wordCount = (word >> 16) & 0xFFFF;

                if (wordCount == 0 || opCode == 0) break;

                if (opCode == 0x0004002D) // OpVariable with Uniform storage
                {
                    if (offset + 16 <= bytecode.Length)
                    {
                        uint storageClass = BitConverter.ToUInt32(bytecode, offset + 12);
                        bindings.Add(new ResourceBinding
                        {
                            Set = 0,
                            Binding = bindingIdx++,
                            Type = storageClass == 2 ? ResourceType.UniformBuffer : ResourceType.StorageBuffer,
                            Stage = ShaderProfile.Compute
                        });
                    }
                }
                else if (opCode == 0x00010010) // OpDecorate
                {
                    if (offset + 12 <= bytecode.Length)
                    {
                        uint decTarget = BitConverter.ToUInt32(bytecode, offset + 4);
                        uint decoration = BitConverter.ToUInt32(bytecode, offset + 8);
                        if (decoration == 33) // Binding
                        {
                            uint bindingValue = offset + 12 < bytecode.Length ? BitConverter.ToUInt32(bytecode, offset + 12) : 0;
                            uniforms.Add(new UniformBinding { Name = $"uniform_{bindingValue}", Offset = 0, Size = 64, Type = GpuType.Float });
                        }
                        else if (decoration == 34) // DescriptorSet
                        {
                        }
                    }
                }

                offset += (int)wordCount * 4;
            }
        }

        private void ReflectDxil(byte[] bytecode, List<ResourceBinding> bindings, List<UniformBinding> uniforms, List<ShaderDiagnostic> diagnostics)
        {
            if (bytecode.Length < 4) return;
            string source = Encoding.UTF8.GetString(bytecode);
            var lines = source.Split('\n');
            int bindingIdx = 0;

            foreach (var line in lines)
            {
                string trimmed = line.Trim();
                if (trimmed.StartsWith("Texture2D"))
                {
                    bindings.Add(new ResourceBinding { Set = 0, Binding = bindingIdx++, Type = ResourceType.Image, Stage = ShaderProfile.Compute });
                }
                else if (trimmed.StartsWith("SamplerState"))
                {
                    bindings.Add(new ResourceBinding { Set = 0, Binding = bindingIdx++, Type = ResourceType.Sampler, Stage = ShaderProfile.Compute });
                }
                else if (trimmed.Contains("cbuffer"))
                {
                    bindings.Add(new ResourceBinding { Set = 0, Binding = bindingIdx++, Type = ResourceType.UniformBuffer, Stage = ShaderProfile.Compute });
                }
            }
        }

        private void ReflectAir(byte[] bytecode, List<ResourceBinding> bindings, List<UniformBinding> uniforms, List<ShaderDiagnostic> diagnostics)
        {
            if (bytecode.Length < 4) return;
            string source = Encoding.UTF8.GetString(bytecode);
            var lines = source.Split('\n');
            int bindingIdx = 0;

            foreach (var line in lines)
            {
                string trimmed = line.Trim();
                if (trimmed.Contains("[[texture("))
                {
                    var start = trimmed.IndexOf("[[texture(") + 10;
                    var end = trimmed.IndexOf(")]]");
                    if (start > 10 && end > start)
                    {
                        int texBinding = int.Parse(trimmed[start..end]);
                        bindings.Add(new ResourceBinding { Set = 0, Binding = texBinding, Type = ResourceType.Image, Stage = ShaderProfile.Compute });
                    }
                }
                else if (trimmed.Contains("[[buffer("))
                {
                    var start = trimmed.IndexOf("[[buffer(") + 9;
                    var end = trimmed.IndexOf(")]]");
                    if (start > 9 && end > start)
                    {
                        int bufBinding = int.Parse(trimmed[start..end]);
                        bindings.Add(new ResourceBinding { Set = 0, Binding = bufBinding, Type = ResourceType.UniformBuffer, Stage = ShaderProfile.Compute });
                    }
                }
            }
        }

        public bool ValidateCompatibility(ShaderReflectionResult reflection, ShaderProfile targetProfile)
        {
            bool valid = true;
            foreach (var binding in reflection.Bindings)
            {
                if (binding.Binding < 0) valid = false;
                if (binding.Count < 1) valid = false;
            }
            return valid;
        }
    }

    public record ShaderReflectionResult
    {
        public ImmutableArray<ResourceBinding> Bindings { get; init; } = ImmutableArray<ResourceBinding>.Empty;
        public ImmutableArray<UniformBinding> Uniforms { get; init; } = ImmutableArray<UniformBinding>.Empty;
        public ImmutableArray<PushConstantBlock> PushConstants { get; init; } = ImmutableArray<PushConstantBlock>.Empty;
        public ImmutableArray<SpecializationConstant> SpecializationConstants { get; init; } = ImmutableArray<SpecializationConstant>.Empty;
        public ImmutableArray<ShaderDiagnostic> Diagnostics { get; init; } = ImmutableArray<ShaderDiagnostic>.Empty;
    }

    #endregion Shader Reflection

    #region Shader Compiler Diagnostics

    public class ShaderCompilerDiagnostics
    {
        private readonly List<DiagnosticMessage> _messages = new();
        private readonly Dictionary<string, int> _categoryCounts = new();
        private readonly Dictionary<CompilationStage, TimeSpan> _stageTimings = new();
        private readonly List<ShaderPerformanceHint> _performanceHints = new();

        public IReadOnlyList<DiagnosticMessage> Messages => _messages;
        public IReadOnlyList<ShaderPerformanceHint> PerformanceHints => _performanceHints;

        public void ReportError(string code, string message, string? file = null, int? line = null, int? column = null)
        {
            _messages.Add(DiagnosticMessage.Error(code, message, file, line, column));
        }

        public void ReportWarning(string code, string message, string? hint = null)
        {
            _messages.Add(DiagnosticMessage.Warn(code, message, hint));
        }

        public void ReportInfo(string code, string message)
        {
            _messages.Add(DiagnosticMessage.Info(code, message));
        }

        public void ReportPerformanceHint(string category, string message, double estimatedSpeedup, string suggestion, int affectedInstructions = 0)
        {
            _performanceHints.Add(new ShaderPerformanceHint
            {
                Category = category,
                Message = message,
                EstimatedSpeedup = estimatedSpeedup,
                Suggestion = suggestion,
                AffectedInstructionCount = affectedInstructions
            });
        }

        public void RecordStageTime(CompilationStage stage, TimeSpan time)
        {
            _stageTimings[stage] = time;
        }

        public CompilationDiagnostics ToCompilationDiagnostics()
        {
            var errors = _messages.Where(m => m.Severity == DiagnosticSeverity.Error || m.Severity == DiagnosticSeverity.Fatal).ToImmutableArray();
            var warnings = _messages.Where(m => m.Severity == DiagnosticSeverity.Warning).ToImmutableArray();
            var hints = _messages.Where(m => m.Severity == DiagnosticSeverity.PerformanceHint).ToImmutableArray();
            return new CompilationDiagnostics(warnings, errors, hints);
        }

        public ImmutableDictionary<CompilationStage, TimeSpan> GetStageTimings() =>
            _stageTimings.ToImmutableDictionary();

        public int ErrorCount => _messages.Count(m => m.Severity == DiagnosticSeverity.Error || m.Severity == DiagnosticSeverity.Fatal);
        public int WarningCount => _messages.Count(m => m.Severity == DiagnosticSeverity.Warning);
        public bool HasErrors => ErrorCount > 0;

        public string GetSummary()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Diagnostics: {_messages.Count} messages, {_performanceHints.Count} perf hints");
            sb.AppendLine($"  Errors: {ErrorCount}, Warnings: {WarningCount}");
            foreach (var stage in _stageTimings)
                sb.AppendLine($"  {stage.Key}: {stage.Value.TotalMilliseconds:F2}ms");
            if (_performanceHints.Count > 0)
            {
                sb.AppendLine("  Performance hints:");
                foreach (var hint in _performanceHints.Take(5))
                    sb.AppendLine($"    [{hint.Category}] {hint.Message} (est. {hint.EstimatedSpeedup:F1}x speedup)");
            }
            return sb.ToString();
        }

        public void Clear()
        {
            _messages.Clear();
            _categoryCounts.Clear();
            _stageTimings.Clear();
            _performanceHints.Clear();
        }
    }

    #endregion Shader Compiler Diagnostics

    #region SpirV Compiler

    public class SpirVCompiler
    {
        private readonly SpirVDialect _dialect = new();
        private readonly ShaderReflection _reflection = new();
        private readonly ShaderCompilerDiagnostics _diag = new();

        public SpirVCode Compile(ExpressionTree tree)
        {
            var sw = Stopwatch.StartNew();
            var code = _dialect.Emit(tree);
            sw.Stop();
            _diag.RecordStageTime(CompilationStage.CodeGeneration, sw.Elapsed);
            return code;
        }

        public CompilationResult CompileWithValidation(ExpressionTree tree)
        {
            var timings = new Dictionary<CompilationStage, TimeSpan>();
            var sw = Stopwatch.StartNew();

            var code = Compile(tree);
            timings[CompilationStage.CodeGeneration] = sw.Elapsed;

            sw.Restart();
            var reflectionResult = _reflection.Reflect(code.ToBytes(), ShaderTargetLanguage.SpirV);
            timings[CompilationStage.Reflection] = sw.Elapsed;

            sw.Restart();
            bool valid = _reflection.ValidateCompatibility(reflectionResult, tree.TargetProfile);
            timings[CompilationStage.Validation] = sw.Elapsed;

            if (!valid)
                _diag.ReportError("SV001", "SPIR-V validation failed");

            return new CompilationResult
            {
                Success = !(_diag.HasErrors),
                Diagnostics = _diag.ToCompilationDiagnostics(),
                Timings = timings.ToImmutableDictionary(),
                ShaderModule = new ShaderModule
                {
                    Bytecode = code.ToBytes(),
                    TargetLanguage = ShaderTargetLanguage.SpirV,
                    SourceText = code.SourceText,
                    Hash = ComputeHash(code.ToBytes()),
                    ResourceBindings = reflectionResult.Bindings,
                    UniformBindings = reflectionResult.Uniforms,
                    PushConstants = reflectionResult.PushConstants,
                    SpecializationConstants = reflectionResult.SpecializationConstants,
                    EstimatedCost = tree.EstimateOperationCount()
                }
            };
        }

        private static string ComputeHash(byte[] data)
        {
            using var sha = SHA256.Create();
            return Convert.ToBase64String(sha.ComputeHash(data));
        }
    }

    #endregion SpirV Compiler

    #region DXIL Compiler

    public class DxilCompiler
    {
        private readonly DxilDialect _dialect = new();
        private readonly ShaderReflection _reflection = new();
        private readonly ShaderCompilerDiagnostics _diag = new();

        public DxilCode Compile(ExpressionTree tree)
        {
            var sw = Stopwatch.StartNew();
            var code = _dialect.Emit(tree);
            sw.Stop();
            _diag.RecordStageTime(CompilationStage.CodeGeneration, sw.Elapsed);
            return code;
        }

        public CompilationResult CompileWithValidation(ExpressionTree tree)
        {
            var timings = new Dictionary<CompilationStage, TimeSpan>();
            var sw = Stopwatch.StartNew();
            var code = Compile(tree);
            timings[CompilationStage.CodeGeneration] = sw.Elapsed;

            sw.Restart();
            var reflectionResult = _reflection.Reflect(code.Bytecode, ShaderTargetLanguage.Dxil);
            timings[CompilationStage.Reflection] = sw.Elapsed;

            bool valid = _reflection.ValidateCompatibility(reflectionResult, tree.TargetProfile);
            timings[CompilationStage.Validation] = sw.Elapsed;

            return new CompilationResult
            {
                Success = !(_diag.HasErrors) && valid,
                Diagnostics = _diag.ToCompilationDiagnostics(),
                Timings = timings.ToImmutableDictionary(),
                ShaderModule = new ShaderModule
                {
                    Bytecode = code.Bytecode,
                    TargetLanguage = ShaderTargetLanguage.Dxil,
                    SourceText = code.SourceText,
                    Hash = ComputeHash(code.Bytecode),
                    ResourceBindings = reflectionResult.Bindings,
                    EstimatedCost = tree.EstimateOperationCount()
                }
            };
        }

        private static string ComputeHash(byte[] data)
        {
            using var sha = SHA256.Create();
            return Convert.ToBase64String(sha.ComputeHash(data));
        }
    }

    #endregion DXIL Compiler

    #region Metal Compiler

    public class MetalCompiler
    {
        private readonly AirDialect _dialect = new();
        private readonly ShaderReflection _reflection = new();
        private readonly ShaderCompilerDiagnostics _diag = new();

        public AirCode Compile(ExpressionTree tree)
        {
            var sw = Stopwatch.StartNew();
            var code = _dialect.Emit(tree);
            sw.Stop();
            _diag.RecordStageTime(CompilationStage.CodeGeneration, sw.Elapsed);
            return code;
        }

        public CompilationResult CompileWithValidation(ExpressionTree tree)
        {
            var timings = new Dictionary<CompilationStage, TimeSpan>();
            var sw = Stopwatch.StartNew();
            var code = Compile(tree);
            timings[CompilationStage.CodeGeneration] = sw.Elapsed;

            sw.Restart();
            var reflectionResult = _reflection.Reflect(code.Bytecode, ShaderTargetLanguage.Air);
            timings[CompilationStage.Reflection] = sw.Elapsed;

            bool valid = _reflection.ValidateCompatibility(reflectionResult, tree.TargetProfile);
            timings[CompilationStage.Validation] = sw.Elapsed;

            return new CompilationResult
            {
                Success = !(_diag.HasErrors) && valid,
                Diagnostics = _diag.ToCompilationDiagnostics(),
                Timings = timings.ToImmutableDictionary(),
                ShaderModule = new ShaderModule
                {
                    Bytecode = code.Bytecode,
                    TargetLanguage = ShaderTargetLanguage.Air,
                    SourceText = code.SourceText,
                    Hash = ComputeHash(code.Bytecode),
                    ResourceBindings = reflectionResult.Bindings,
                    EstimatedCost = tree.EstimateOperationCount()
                }
            };
        }

        private static string ComputeHash(byte[] data)
        {
            using var sha = SHA256.Create();
            return Convert.ToBase64String(sha.ComputeHash(data));
        }
    }

    #endregion Metal Compiler

    #region Shader Permutation Manager

    public class ShaderPermutationManager
    {
        private readonly Dictionary<string, ShaderPermutation> _permutations = new();
        private readonly Dictionary<string, byte[]> _compiledPermutations = new();
        private int _nextId;

        public int CreatePermutation(ShaderPermutationDefinition definition)
        {
            int id = Interlocked.Increment(ref _nextId);
            string key = ComputePermutationKey(definition);
            _permutations[key] = new ShaderPermutation
            {
                Id = id,
                Key = key,
                Definition = definition,
                CreatedAt = DateTime.UtcNow
            };
            return id;
        }

        public ShaderModule? GetCompiledPermutation(ShaderPermutationDefinition definition)
        {
            string key = ComputePermutationKey(definition);
            if (_compiledPermutations.TryGetValue(key, out var bytecode))
            {
                return new ShaderModule
                {
                    Bytecode = bytecode,
                    Hash = key,
                    TargetLanguage = definition.TargetLanguage
                };
            }
            return null;
        }

        public void StoreCompiledPermutation(ShaderPermutationDefinition definition, ShaderModule module)
        {
            string key = ComputePermutationKey(definition);
            _compiledPermutations[key] = module.Bytecode;
        }

        public IEnumerable<ShaderPermutation> GetAllPermutations() => _permutations.Values;

        public IEnumerable<ShaderPermutationDefinition> GeneratePermutations(
            ShaderTargetLanguage target,
            IEnumerable<MaterialFeature> features,
            IEnumerable<QualityLevel> qualityLevels)
        {
            var result = new List<ShaderPermutationDefinition>();
            foreach (var quality in qualityLevels)
            {
                foreach (var featureSet in GenerateFeatureCombinations(features))
                {
                    result.Add(new ShaderPermutationDefinition
                    {
                        TargetLanguage = target,
                        QualityLevel = quality,
                        EnabledFeatures = featureSet.ToImmutableArray(),
                        Profile = ShaderProfile.Compute
                    });
                }
            }
            return result;
        }

        private IEnumerable<IEnumerable<MaterialFeature>> GenerateFeatureCombinations(IEnumerable<MaterialFeature> features)
        {
            var featureList = features.ToList();
            int count = featureList.Count;
            int combinations = 1 << count;
            for (int mask = 0; mask < combinations; mask++)
            {
                var combo = new List<MaterialFeature>();
                for (int i = 0; i < count; i++)
                    if ((mask & (1 << i)) != 0) combo.Add(featureList[i]);
                yield return combo;
            }
        }

        private string ComputePermutationKey(ShaderPermutationDefinition def)
        {
            var sb = new StringBuilder();
            sb.Append(def.TargetLanguage).Append('|').Append(def.QualityLevel).Append('|').Append(def.Profile);
            foreach (var f in def.EnabledFeatures.OrderBy(f => f.ToString()))
                sb.Append('|').Append(f);
            using var sha = SHA256.Create();
            return Convert.ToBase64String(sha.ComputeHash(Encoding.UTF8.GetBytes(sb.ToString())));
        }

        public void InvalidateAll()
        {
            _permutations.Clear();
            _compiledPermutations.Clear();
        }

        public int PermutationCount => _permutations.Count;
        public int CompiledCount => _compiledPermutations.Count;
    }

    public record ShaderPermutationDefinition
    {
        public ShaderTargetLanguage TargetLanguage { get; init; }
        public QualityLevel QualityLevel { get; init; }
        public ImmutableArray<MaterialFeature> EnabledFeatures { get; init; } = ImmutableArray<MaterialFeature>.Empty;
        public ShaderProfile Profile { get; init; } = ShaderProfile.Compute;
    }

    public record ShaderPermutation
    {
        public int Id { get; init; }
        public string Key { get; init; } = string.Empty;
        public ShaderPermutationDefinition Definition { get; init; } = null!;
        public DateTime CreatedAt { get; init; }
    }

    public enum QualityLevel { Low, Medium, High, Ultra, Cinematic }

    public enum MaterialFeature
    {
        None = 0,
        Diffuse = 1,
        Normal = 2,
        Specular = 4,
        Roughness = 8,
        Metallic = 16,
        Emissive = 32,
        Opacity = 64,
        Parallax = 128,
        Subsurface = 256,
        ClearCoat = 512,
        Anisotropy = 1024,
        Transmission = 2048,
        Sheen = 4096,
       ior = 8192
    }

    #endregion Shader Permutation Manager

    #region Compilation Pipeline

    public class CompilationPipeline : IDisposable
    {
        private readonly ShaderCache _cache;
        private readonly NeuralGraphTranspiler _transpiler;
        private readonly ExpressionOptimizer _optimizer;
        private readonly ShaderReflection _reflection;
        private readonly ShaderCompilerDiagnostics _diagnostics;
        private readonly ShaderPermutationManager _permutationManager;
        private readonly ConcurrentQueue<CompilationRequest> _pendingRequests;
        private readonly List<Task<CompilationResult>> _activeCompilations;
        private readonly object _lock = new();
        private int _activeCompilationCount;
        private const int MaxParallelCompilations = 4;

        public CompilationPipeline(string? cachePath = null)
        {
            _cache = new ShaderCache(cachePath);
            _transpiler = new NeuralGraphTranspiler();
            _optimizer = new ExpressionOptimizer();
            _reflection = new ShaderReflection();
            _diagnostics = new ShaderCompilerDiagnostics();
            _permutationManager = new ShaderPermutationManager();
            _pendingRequests = new ConcurrentQueue<CompilationRequest>();
            _activeCompilations = new List<Task<CompilationResult>>();
        }

        public ShaderCache Cache => _cache;
        public ShaderPermutationManager PermutationManager => _permutationManager;
        public ShaderCompilerDiagnostics Diagnostics => _diagnostics;

        public async Task<CompilationResult> CompileAsync(CompilationRequest request, IProgress<double>? progress = null, CancellationToken ct = default)
        {
            var sw = Stopwatch.StartNew();
            var timings = new Dictionary<CompilationStage, TimeSpan>();
            _diagnostics.Clear();

            progress?.Report(0);

            // Cache check
            string cacheKey = request.ComputeCacheKey();
            var cached = _cache.Get(cacheKey);
            if (cached != null)
            {
                timings[CompilationStage.Caching] = sw.Elapsed;
                return CompilationResult.SuccessResult(cached, _diagnostics.ToCompilationDiagnostics(),
                    timings.ToImmutableDictionary(), ch: true);
            }

            progress?.Report(0.1);

            // Transpilation
            var transpileSw = Stopwatch.StartNew();
            ExpressionTree tree;
            try
            {
                tree = _transpiler.Transpile(request.Genome);
            }
            catch (Exception ex)
            {
                _diagnostics.ReportError("TP001", $"Transpilation failed: {ex.Message}");
                return CompilationResult.FailureResult(_diagnostics.ToCompilationDiagnostics(), timings.ToImmutableDictionary());
            }
            transpileSw.Stop();
            timings[CompilationStage.Transpilation] = transpileSw.Elapsed;
            progress?.Report(0.3);

            // Optimization
            var optimizeSw = Stopwatch.StartNew();
            OptimizedExpression optimized;
            try
            {
                optimized = _optimizer.Optimize(tree, request.OptimizationLevel);
            }
            catch (Exception ex)
            {
                _diagnostics.ReportError("OP001", $"Optimization failed: {ex.Message}");
                return CompilationResult.FailureResult(_diagnostics.ToCompilationDiagnostics(), timings.ToImmutableDictionary());
            }
            optimizeSw.Stop();
            timings[CompilationStage.Optimization] = optimizeSw.Elapsed;
            progress?.Report(0.5);

            // Code generation
            var codeGenSw = Stopwatch.StartNew();
            CompilationResult result;
            try
            {
                result = GenerateCode(optimized.Tree, request);
            }
            catch (Exception ex)
            {
                _diagnostics.ReportError("CG001", $"Code generation failed: {ex.Message}");
                return CompilationResult.FailureResult(_diagnostics.ToCompilationDiagnostics(), timings.ToImmutableDictionary());
            }
            codeGenSw.Stop();
            timings[CompilationStage.CodeGeneration] = codeGenSw.Elapsed;
            progress?.Report(0.8);

            // Cache store
            if (result.Success && result.ShaderModule != null)
            {
                _cache.Store(cacheKey, result.ShaderModule);
            }

            timings[CompilationStage.Caching] = sw.Elapsed;
            progress?.Report(1.0);

            return result with
            {
                Timings = timings.ToImmutableDictionary(),
                Diagnostics = _diagnostics.ToCompilationDiagnostics()
            };
        }

        public async Task<ImmutableArray<CompilationResult>> CompileBatchAsync(
            IEnumerable<CompilationRequest> requests, IProgress<double>? progress = null, CancellationToken ct = default)
        {
            var requestList = requests.ToList();
            var results = ImmutableArray.CreateBuilder<CompilationResult>();
            int completed = 0;

            var tasks = requestList.Select(req => CompileAsync(req, null, ct)).ToList();
            var allResults = await Task.WhenAll(tasks);

            foreach (var result in allResults)
            {
                results.Add(result);
                completed++;
                progress?.Report((double)completed / requestList.Count);
            }

            return results.ToImmutable();
        }

        public void EnqueueRequest(CompilationRequest request)
        {
            _pendingRequests.Enqueue(request);
        }

        public CompilationResult CompileSynchronous(CompilationRequest request)
        {
            return CompileAsync(request).GetAwaiter().GetResult();
        }

        private CompilationResult GenerateCode(ExpressionTree tree, CompilationRequest request)
        {
            return request.TargetBackend switch
            {
                ShaderTargetLanguage.SpirV => new SpirVCompiler().CompileWithValidation(tree),
                ShaderTargetLanguage.Dxil => new DxilCompiler().CompileWithValidation(tree),
                ShaderTargetLanguage.Air => new MetalCompiler().CompileWithValidation(tree),
                ShaderTargetLanguage.Hlsl => GenerateHlsl(tree),
                ShaderTargetLanguage.Glsl => GenerateGlsl(tree),
                ShaderTargetLanguage.Wgsl => GenerateWgsl(tree),
                _ => CompilationResult.FailureResult(_diagnostics.ToCompilationDiagnostics(),
                    ImmutableDictionary<CompilationStage, TimeSpan>.Empty)
            };
        }

        private CompilationResult GenerateHlsl(ExpressionTree tree)
        {
            var hlsl = new HlslDialect();
            string source = hlsl.Emit(tree);
            var module = new ShaderModule
            {
                Bytecode = Encoding.UTF8.GetBytes(source),
                TargetLanguage = ShaderTargetLanguage.Hlsl,
                SourceText = source,
                Hash = ComputeHash(Encoding.UTF8.GetBytes(source)),
                EstimatedCost = tree.EstimateOperationCount()
            };
            return CompilationResult.SuccessResult(module, _diagnostics.ToCompilationDiagnostics(),
                ImmutableDictionary<CompilationStage, TimeSpan>.Empty);
        }

        private CompilationResult GenerateGlsl(ExpressionTree tree)
        {
            var glsl = new GlslDialect();
            string source = glsl.Emit(tree);
            var module = new ShaderModule
            {
                Bytecode = Encoding.UTF8.GetBytes(source),
                TargetLanguage = ShaderTargetLanguage.Glsl,
                SourceText = source,
                Hash = ComputeHash(Encoding.UTF8.GetBytes(source)),
                EstimatedCost = tree.EstimateOperationCount()
            };
            return CompilationResult.SuccessResult(module, _diagnostics.ToCompilationDiagnostics(),
                ImmutableDictionary<CompilationStage, TimeSpan>.Empty);
        }

        private CompilationResult GenerateWgsl(ExpressionTree tree)
        {
            var wgsl = new WgslDialect();
            string source = wgsl.Emit(tree);
            var module = new ShaderModule
            {
                Bytecode = Encoding.UTF8.GetBytes(source),
                TargetLanguage = ShaderTargetLanguage.Wgsl,
                SourceText = source,
                Hash = ComputeHash(Encoding.UTF8.GetBytes(source)),
                EstimatedCost = tree.EstimateOperationCount()
            };
            return CompilationResult.SuccessResult(module, _diagnostics.ToCompilationDiagnostics(),
                ImmutableDictionary<CompilationStage, TimeSpan>.Empty);
        }

        private static string ComputeHash(byte[] data)
        {
            using var sha = SHA256.Create();
            return Convert.ToBase64String(sha.ComputeHash(data));
        }

        public void Dispose()
        {
            _cache?.Dispose();
            GC.SuppressFinalize(this);
        }
    }

    #endregion Compilation Pipeline
}
