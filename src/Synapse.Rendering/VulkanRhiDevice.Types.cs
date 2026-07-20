// =============================================================================
// GDNN Engine - Vulkan 1.4 Render Hardware Interface Backend
// File: VulkanRhiDevice.cs
// Description: Complete Vulkan RHI implementation for the G-DNN Engine
// Author: GDNN Engine Team
// Version: 1.0.0
// =============================================================================

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GDNN.Rendering.Compat;

namespace GDNN.RHI.Vulkan
{
    // =========================================================================
    // ENUMS
    // =========================================================================

    /// <summary>Vulkan result codes returned from API calls</summary>
    public enum VulkanResult
    {
        Success = 0,
        NotReady = 1,
        Timeout = 2,
        EventSet = 3,
        EventReset = 4,
        Incomplete = 5,
        ErrorOutOfHostMemory = -1,
        ErrorOutOfDeviceMemory = -2,
        ErrorInitializationFailed = -3,
        ErrorDeviceLost = -4,
        ErrorMemoryMapFailed = -5,
        ErrorLayerNotPresent = -6,
        ErrorExtensionNotPresent = -7,
        ErrorFeatureNotPresent = -8,
        ErrorIncompatibleDriver = -9,
        ErrorTooManyObjects = -10,
        ErrorFormatNotSupported = -11,
        ErrorFragmentedPool = -12,
        ErrorSurfaceLost = -13,
        ErrorNativeWindowInUse = -14,
        ErrorOutOfDate = -1000001000,
        SuboptimalKHR = 1000001003,
        ErrorOutOfDateKHR = -1000001000,
        ErrorSurfaceLost2 = -1000001001,
        ErrorIncompatibleDisplay = -1000003001,
        ErrorInvalidCommandBuffer = -1000012000,
        ErrorInvalidDescriptorSet = -1000012001,
        ErrorInvalidDescriptorPool = -1000012002,
        ErrorInvalidDescriptorSetLayout = -1000012003,
        ErrorInvalidPipeline = -1000012004,
        ErrorInvalidRenderPass = -1000012005,
        ErrorInvalidPipelineCache = -1000012006,
        ErrorInvalidSampler = -1000012007,
        ErrorInvalidBuffer = -1000012008,
        ErrorInvalidImage = -1000012009,
        ErrorInvalidShaderModule = -1000012010,
        ErrorInvalidPipelineLayout = -1000012011,
        ErrorOutOfPoolMemory = -1000012012,
        ErrorInvalidExternalHandle = -1000012013,
        ErrorUnknown = -1000013000
    }

    /// <summary>Pipeline bind points</summary>
    public enum PipelineBindPoint
    {
        Graphics = 0,
        Compute = 1,
        RayTracingKHR = 1000165000
    }

    /// <summary>Descriptor types</summary>
    public enum DescriptorType
    {
        Sampler = 0,
        CombinedImageSampler = 1,
        SampledImage = 2,
        StorageImage = 3,
        UniformTexelBuffer = 4,
        StorageTexelBuffer = 5,
        UniformBuffer = 6,
        StorageBuffer = 7,
        UniformBufferDynamic = 8,
        StorageBufferDynamic = 9,
        InputAttachment = 10,
        InlineUniformBlock = 1000138000,
        AccelerationStructureKHR = 1000150000
    }

    /// <summary>Shader stage flags</summary>
    [Flags]
    public enum ShaderStageFlag
    {
        Vertex = 0x00000001,
        TessellationControl = 0x00000002,
        TessellationEvaluation = 0x00000004,
        Geometry = 0x00000008,
        Fragment = 0x00000010,
        Compute = 0x00000020,
        AllGraphics = 0x0000001F,
        All = 0x7FFFFFFF,
        RayGenerationKHR = 0x00000100,
        AnyHitKHR = 0x00000200,
        ClosestHitKHR = 0x00000400,
        MissKHR = 0x00000800,
        IntersectionKHR = 0x00001000,
        CallableKHR = 0x00002000,
        TaskNV = 0x00000040,
        MeshNV = 0x00000080
    }

    /// <summary>Image aspect flags</summary>
    [Flags]
    public enum ImageAspectFlag
    {
        Color = 0x00000001,
        Depth = 0x00000002,
        Stencil = 0x00000004,
        Metadata = 0x00000008,
        Plane0 = 0x00000010,
        Plane1 = 0x00000020,
        Plane2 = 0x00000040
    }

    /// <summary>Image usage flags</summary>
    [Flags]
    public enum ImageUsageFlag
    {
        TransferSrc = 0x00000001,
        TransferDst = 0x00000002,
        Sampled = 0x00000004,
        Storage = 0x00000008,
        ColorAttachment = 0x00000010,
        DepthStencilAttachment = 0x00000020,
        TransientAttachment = 0x00000040,
        InputAttachment = 0x00000080
    }

    /// <summary>Format feature flags</summary>
    [Flags]
    public enum FormatFeatureFlag
    {
        SampledImage = 0x00000001,
        StorageImage = 0x00000002,
        StorageImageAtomic = 0x00000004,
        VertexBuffer = 0x00000008,
        ColorAttachment = 0x00000010,
        ColorAttachmentBlend = 0x00000020,
        DepthStencilAttachment = 0x00000040,
        BlitSrc = 0x00000080,
        BlitDst = 0x00000100,
        SampledImageFilterLinear = 0x00000200,
        TransferSrc = 0x00000400,
        TransferDst = 0x00000800,
        MidpointChromaSamples = 0x00002000,
        CositedChromaSamples = 0x00004000,
        SampledImageFilterMinmax = 0x00008000,
        SampledImageFilterConversionChromaRate = 0x00080000
    }

    /// <summary>Sample count flags</summary>
    [Flags]
    public enum SampleCountFlag
    {
        Count1 = 0x00000001,
        Count2 = 0x00000002,
        Count4 = 0x00000004,
        Count8 = 0x00000008,
        Count16 = 0x00000010,
        Count32 = 0x00000020,
        Count64 = 0x00000040
    }

    /// <summary>Buffer usage flags</summary>
    [Flags]
    public enum BufferUsageFlag
    {
        TransferSrc = 0x00000001,
        TransferDst = 0x00000002,
        UniformTexelBuffer = 0x00000004,
        StorageTexelBuffer = 0x00000008,
        UniformBuffer = 0x00000010,
        StorageBuffer = 0x00000020,
        IndexBuffer = 0x00000040,
        VertexBuffer = 0x00000080,
        IndirectBuffer = 0x00000100,
        ShaderDeviceAddress = 0x00020000,
        AccelerationStructureBuildInputReadOnly = 0x00080000,
        ShaderBindingTableKHR = 0x00000400,
        AccelerationStructureStorageKHR = 0x00100000
    }

    /// <summary>Memory property flags</summary>
    [Flags]
    public enum MemoryPropertyFlag
    {
        DeviceLocal = 0x00000001,
        HostVisible = 0x00000002,
        HostCoherent = 0x00000004,
        HostCached = 0x00000008,
        LazilyAllocated = 0x00000010,
        Protected = 0x00000020,
        DeviceCoherentAMD = 0x00000040,
        DeviceUncachedAMD = 0x00000080
    }

    /// <summary>Command buffer usage flags</summary>
    [Flags]
    public enum CommandBufferUsageFlag
    {
        OneTimeSubmit = 0x00000001,
        RenderPassContinue = 0x00000002,
        SimultaneousUse = 0x00000004
    }

    /// <summary>Pipeline creation flags</summary>
    [Flags]
    public enum PipelineCreateFlag
    {
        None = 0,
        DisableOptimization = 0x00000001,
        AllowDerivatives = 0x00000002,
        Derivative = 0x00000004,
        FragmentShadingRateAttachment = 0x00200000,
        FragmentDensityMapAttachment = 0x00400000,
        LibraryBit = 0x00001000,
        RayTracingSkipTriangles = 0x00001000,
        RayTracingSkipAABBs = 0x00002000,
        RayTracingNoNullMissShaders = 0x00004000,
        RayTracingNoNullIntersectionShaders = 0x00008000,
        RayTracingShaderGroupHandleCaptureReplay = 0x00010000,
        RayTracingShaderGroupHandleCaptureReplayBit = 0x00020000,
        RayTracingShaderGroupHandleStrict = 0x00040000,
        RayTracingPipelineSkipInstances = 0x00080000,
        RayTracingPipelineSkipTriangles = 0x00100000,
        RayTracingPipelineSkipAABBs = 0x00200000,
        RayTracingPipelineDeferredHostOperations = 0x00400000,
        Library = 0x08000000,
        DescriptorBuffer = 0x20000000
    }

    /// <summary>Queue family flags</summary>
    [Flags]
    public enum QueueFlag
    {
        Graphics = 0x00000001,
        Compute = 0x00000002,
        Transfer = 0x00000004,
        SparseBinding = 0x00000008,
        Protected = 0x00000010,
        VideoDecode = 0x00000020,
        VideoEncode = 0x00000040,
        OpticalFlow = 0x00000100
    }

    /// <summary>Vulkan format enumeration</summary>
    public enum VulkanFormat
    {
        Undefined = 0,
        R4G4UnormPack8 = 1,
        R4G4B4A4UnormPack16 = 2,
        B4G4R4A4UnormPack16 = 3,
        R5G6B5UnormPack16 = 4,
        B5G6R5UnormPack16 = 5,
        R5G5B5A1UnormPack16 = 6,
        B5G5R5A1UnormPack16 = 7,
        A1R5G5B5UnormPack16 = 8,
        R8Unorm = 9,
        R8Snorm = 10,
        R8Uscaled = 11,
        R8Sscaled = 12,
        R8Uint = 13,
        R8Sint = 14,
        R8Srgb = 15,
        R8G8Unorm = 16,
        R8G8Snorm = 17,
        R8G8Uscaled = 18,
        R8G8Sscaled = 19,
        R8G8Uint = 20,
        R8G8Sint = 21,
        R8G8Srgb = 22,
        R8G8B8Unorm = 23,
        R8G8B8Snorm = 24,
        R8G8B8Uscaled = 25,
        R8G8B8Sscaled = 26,
        R8G8B8Uint = 27,
        R8G8B8Sint = 28,
        R8G8B8Srgb = 29,
        B8G8R8Unorm = 30,
        B8G8R8Snorm = 31,
        B8G8R8Uscaled = 32,
        B8G8R8Sscaled = 33,
        B8G8R8Uint = 34,
        B8G8R8Sint = 35,
        B8G8R8Srgb = 36,
        R8G8B8A8Unorm = 37,
        R8G8B8A8Snorm = 38,
        R8G8B8A8Uscaled = 39,
        R8G8B8A8Sscaled = 40,
        R8G8B8A8Uint = 41,
        R8G8B8A8Sint = 42,
        R8G8B8A8Srgb = 43,
        B8G8R8A8Unorm = 44,
        B8G8R8A8Snorm = 45,
        B8G8R8A8Uscaled = 46,
        B8G8R8A8Sscaled = 47,
        B8G8R8A8Uint = 48,
        B8G8R8A8Sint = 49,
        B8G8R8A8Srgb = 50,
        A8B8G8R8UnormPack32 = 51,
        A8B8G8R8SnormPack32 = 52,
        A8B8G8R8UscaledPack32 = 53,
        A8B8G8R8SscaledPack32 = 54,
        A8B8G8R8UintPack32 = 55,
        A8B8G8R8SintPack32 = 56,
        A8B8G8R8SrgbPack32 = 57,
        A2R10G10B10UnormPack32 = 58,
        A2R10G10B10SnormPack32 = 59,
        A2R10G10B10UscaledPack32 = 60,
        A2R10G10B10SscaledPack32 = 61,
        A2R10G10B10UintPack32 = 62,
        A2R10G10B10SintPack32 = 63,
        A2B10G10R10UnormPack32 = 64,
        A2B10G10R10SnormPack32 = 65,
        A2B10G10R10UscaledPack32 = 66,
        A2B10G10R10SscaledPack32 = 67,
        A2B10G10R10UintPack32 = 68,
        A2B10G10R10SintPack32 = 69,
        R16Unorm = 70,
        R16Snorm = 71,
        R16Uscaled = 72,
        R16Sscaled = 73,
        R16Uint = 74,
        R16Sint = 75,
        R16Sfloat = 76,
        R16G16Unorm = 77,
        R16G16Snorm = 78,
        R16G16Uscaled = 79,
        R16G16Sscaled = 80,
        R16G16Uint = 81,
        R16G16Sint = 82,
        R16G16Sfloat = 83,
        R16G16B16Unorm = 84,
        R16G16B16Snorm = 85,
        R16G16B16Uscaled = 86,
        R16G16B16Sscaled = 87,
        R16G16B16Uint = 88,
        R16G16B16Sint = 89,
        R16G16B16Sfloat = 90,
        R16G16B16A16Unorm = 91,
        R16G16B16A16Snorm = 92,
        R16G16B16A16Uscaled = 93,
        R16G16B16A16Sscaled = 94,
        R16G16B16A16Uint = 95,
        R16G16B16A16Sint = 96,
        R16G16B16A16Sfloat = 97,
        R32Uint = 98,
        R32Sint = 99,
        R32Sfloat = 100,
        R32G32Uint = 101,
        R32G32Sint = 102,
        R32G32Sfloat = 103,
        R32G32B32Uint = 104,
        R32G32B32Sint = 105,
        R32G32B32Sfloat = 106,
        R32G32B32A32Uint = 107,
        R32G32B32A32Sint = 108,
        R32G32B32A32Sfloat = 109,
        R64Uint = 110,
        R64Sint = 111,
        R64Sfloat = 112,
        R64G64Uint = 113,
        R64G64Sint = 114,
        R64G64Sfloat = 115,
        R64G64B64Uint = 116,
        R64G64B64Sint = 117,
        R64G64B64Sfloat = 118,
        R64G64B64A64Uint = 119,
        R64G64B64A64Sint = 120,
        R64G64B64A64Sfloat = 121,
        B10G11R11UfloatPack32 = 122,
        E5B9G9R9UfloatPack32 = 123,
        D16Unorm = 124,
        X8D24UnormPack32 = 125,
        D32Sfloat = 126,
        S8Uint = 127,
        D16UnormS8Uint = 128,
        D24UnormS8Uint = 129,
        D32SfloatS8Uint = 130,
        BC1RgbUnormBlock = 131,
        BC1RgbSrgbBlock = 132,
        BC1RgbaUnormBlock = 133,
        BC1RgbaSrgbBlock = 134,
        BC2UnormBlock = 135,
        BC2SrgbBlock = 136,
        BC3UnormBlock = 137,
        BC3SrgbBlock = 138,
        BC4UnormBlock = 139,
        BC4SnormBlock = 140,
        BC5UnormBlock = 141,
        BC5SnormBlock = 142,
        BC6HUfloatBlock = 143,
        BC6HSfloatBlock = 144,
        BC7UnormBlock = 145,
        BC7SrgbBlock = 146,
        Etc2R8G8B8UnormBlock = 147,
        Etc2R8G8B8SrgbBlock = 148,
        Etc2R8G8B8A1UnormBlock = 149,
        Etc2R8G8B8A1SrgbBlock = 150,
        EacR11UnormBlock = 151,
        EacR11SnormBlock = 152,
        EacR11G11UnormBlock = 153,
        EacR11G11SnormBlock = 154,
        Astc4x4UnormBlock = 155,
        Astc4x4SrgbBlock = 156,
        Astc5x4UnormBlock = 157,
        Astc5x4SrgbBlock = 158,
        Astc5x5UnormBlock = 159,
        Astc5x5SrgbBlock = 160,
        Astc6x5UnormBlock = 161,
        Astc6x5SrgbBlock = 162,
        Astc6x6UnormBlock = 163,
        Astc6x6SrgbBlock = 164,
        Astc8x5UnormBlock = 165,
        Astc8x5SrgbBlock = 166,
        Astc8x6UnormBlock = 167,
        Astc8x6SrgbBlock = 168,
        Astc8x8UnormBlock = 169,
        Astc8x8SrgbBlock = 170,
        Astc10x5UnormBlock = 171,
        Astc10x5SrgbBlock = 172,
        Astc10x6UnormBlock = 173,
        Astc10x6SrgbBlock = 174,
        Astc10x8UnormBlock = 175,
        Astc10x8SrgbBlock = 176,
        Astc10x10UnormBlock = 177,
        Astc10x10SrgbBlock = 178,
        Astc12x10UnormBlock = 179,
        Astc12x10SrgbBlock = 180,
        Astc12x12UnormBlock = 181,
        Astc12x12SrgbBlock = 182
    }

    /// <summary>Vulkan present mode</summary>
    public enum PresentMode
    {
        Immediate = 0,
        Fifo = 1,
        FifoRelaxed = 2,
        Mailbox = 3
    }

    /// <summary>Vulkan sharing mode</summary>
    public enum SharingMode
    {
        Exclusive = 0,
        Concurrent = 1
    }

    /// <summary>Vulkan image type</summary>
    public enum ImageType
    {
        Type1D = 0,
        Type2D = 1,
        Type3D = 2
    }

    /// <summary>Vulkan image tiling</summary>
    public enum ImageTiling
    {
        Linear = 0,
        Optimal = 1
    }

    /// <summary>Vulkan image view type</summary>
    public enum ImageViewType
    {
        Type1D = 0,
        Type2D = 1,
        Type2DArray = 2,
        Cube = 3,
        CubeArray = 4,
        Type3D = 5
    }

    /// <summary>Vulkan filter</summary>
    public enum Filter
    {
        Nearest = 0,
        Linear = 1,
        Cubic = 1000015000
    }

    /// <summary>Vulkan sampler address mode</summary>
    public enum SamplerAddressMode
    {
        Repeat = 0,
        MirroredRepeat = 1,
        ClampToEdge = 2,
        ClampToBorder = 3,
        MirrorClampToEdge = 4
    }

    /// <summary>Vulkan comparison operation</summary>
    public enum CompareOp
    {
        Never = 0,
        Less = 1,
        Equal = 2,
        LessOrEqual = 3,
        Greater = 4,
        NotEqual = 5,
        GreaterOrEqual = 6,
        Always = 7
    }

    /// <summary>Vulkan stencil operation</summary>
    public enum StencilOp
    {
        Keep = 0,
        Zero = 1,
        Replace = 2,
        IncrementAndClamp = 3,
        DecrementAndClamp = 4,
        Invert = 5,
        IncrementAndWrap = 6,
        DecrementAndWrap = 7
    }

    /// <summary>Vulkan logic operation</summary>
    public enum LogicOp
    {
        Clear = 0, Set = 1, Copy = 2, CopyInverted = 3,
        NoOp = 4, Invert = 5, And = 6, Nand = 7,
        Or = 8, Nor = 9, Xor = 10, Equiv = 11,
        AndReverse = 12, AndInverted = 13, OrReverse = 14, OrInverted = 15
    }

    /// <summary>Vulkan blend factor</summary>
    public enum BlendFactor
    {
        Zero = 0, One = 1, SrcColor = 2, OneMinusSrcColor = 3,
        SrcAlpha = 4, OneMinusSrcAlpha = 5, DstAlpha = 6, OneMinusDstAlpha = 7,
        DstColor = 8, OneMinusDstColor = 9, SrcAlphaSaturate = 10,
        ConstantColor = 11, OneMinusConstantColor = 12, ConstantAlpha = 13,
        OneMinusConstantAlpha = 14, Src1Color = 15, OneMinusSrc1Color = 16,
        Src1Alpha = 17, OneMinusSrc1Alpha = 18
    }

    /// <summary>Vulkan blend operation</summary>
    public enum BlendOp
    {
        Add = 0, Subtract = 1, ReverseSubtract = 2, Min = 3, Max = 4
    }

    /// <summary>Vulkan polygon mode</summary>
    public enum PolygonMode
    {
        Fill = 0, Line = 1, Point = 2
    }

    /// <summary>Vulkan front face</summary>
    public enum FrontFace
    {
        CounterClockwise = 0,
        Clockwise = 1
    }

    /// <summary>Vulkan cull mode flags</summary>
    [Flags]
    public enum CullModeFlag
    {
        None = 0, Front = 0x00000001, Back = 0x00000002, FrontAndBack = 0x00000003
    }

    /// <summary>Vulkan primitive topology</summary>
    public enum PrimitiveTopology
    {
        PointList = 0, LineList = 1, LineStrip = 2, TriangleList = 3,
        TriangleStrip = 4, TriangleFan = 5, LineListWithAdjacency = 6,
        LineStripWithAdjacency = 7, TriangleListWithAdjacency = 8,
        TriangleStripWithAdjacency = 9, PatchList = 10
    }

    /// <summary>Vulkan index type</summary>
    public enum IndexType
    {
        Uint16 = 0, Uint32 = 1, None = 1000165000
    }

    /// <summary>Vulkan pipeline stage flags</summary>
    [Flags]
    public enum PipelineStageFlag
    {
        TopOfPipe = 0x00000001, DrawIndirect = 0x00000002,
        VertexInput = 0x00000004, VertexShader = 0x00000008,
        TessellationControlShader = 0x00000010, TessellationEvaluationShader = 0x00000020,
        GeometryShader = 0x00000040, FragmentShader = 0x00000080,
        EarlyFragmentTests = 0x00000100, LateFragmentTests = 0x00000200,
        ColorAttachmentOutput = 0x00000400, ComputeShader = 0x00000800,
        Transfer = 0x00001000, BottomOfPipe = 0x00002000,
        Host = 0x00004000, AllGraphics = 0x00008000,
        AllCommands = 0x00010000, RayTracingShader = 0x00200000,
        AccelerationStructureCopy = 0x01000000,
        TaskShaderNV = 0x00080000, MeshShaderNV = 0x00100000
    }

    /// <summary>Vulkan access flags</summary>
    [Flags]
    public enum AccessFlag
    {
        None = 0, IndirectCommandRead = 0x00000001, IndexRead = 0x00000002,
        VertexAttributeRead = 0x00000004, UniformRead = 0x00000008,
        InputAttachmentRead = 0x00000010, ShaderRead = 0x00000020,
        ShaderWrite = 0x00000040, ColorAttachmentRead = 0x00000080,
        ColorAttachmentWrite = 0x00000100, DepthStencilAttachmentRead = 0x00000200,
        DepthStencilAttachmentWrite = 0x00000400, TransferRead = 0x00000800,
        TransferWrite = 0x00001000, HostRead = 0x00002000,
        HostWrite = 0x00004000, MemoryRead = 0x00008000,
        MemoryWrite = 0x00010000, AccelerationStructureRead = 0x00200000,
        AccelerationStructureWrite = 0x00400000, ShaderStorageRead = 0x00800000,
        ShaderStorageWrite = 0x01000000
    }

    /// <summary>Vulkan dependency flags</summary>
    [Flags]
    public enum DependencyFlag
    {
        ByRegion = 0x00000001, ViewLocal = 0x00000002, DeviceGroup = 0x00000004
    }

    /// <summary>Vulkan dynamic state</summary>
    public enum DynamicState
    {
        Viewport = 0, Scissor = 1, LineWidth = 2, DepthBias = 3,
        BlendConstants = 4, DepthBounds = 5, StencilCompareMask = 6,
        StencilWriteMask = 7, StencilReference = 8,
        ViewportCount = 1000014000, DiscardRectangle = 1000005000,
        SampleLocations = 1000014001
    }

    /// <summary>Vulkan subpass contents</summary>
    public enum SubpassContents
    {
        Inline = 0, SecondaryCommandBuffers = 1
    }

    /// <summary>Vulkan attachment load operation</summary>
    public enum AttachmentLoadOp
    {
        Load = 0, Clear = 1, DontCare = 2
    }

    /// <summary>Vulkan attachment store operation</summary>
    public enum AttachmentStoreOp
    {
        Store = 0, DontCare = 1
    }

    /// <summary>Vulkan image layout</summary>
    public enum ImageLayout
    {
        Undefined = 0, General = 1, ColorAttachmentOptimal = 2,
        DepthStencilAttachmentOptimal = 3, DepthStencilReadOnlyOptimal = 4,
        ShaderReadOnlyOptimal = 5, TransferSrcOptimal = 6,
        TransferDstOptimal = 7, Preinitialized = 8,
        PresentSrcKHR = 1000001002, SharedPresentKHR = 1000111000,
        FragmentDensityMapOptimalEXT = 1000218000,
        FragmentShadingRateAttachmentOptimalKHR = 1000164003
    }

    /// <summary>Vulkan color component flags</summary>
    [Flags]
    public enum ColorComponentFlag
    {
        R = 0x00000001, G = 0x00000002, B = 0x00000004, A = 0x00000008,
        RGBA = R | G | B | A
    }

    /// <summary>Vulkan vertex input rate</summary>
    public enum VertexInputRate
    {
        Vertex = 0, Instance = 1
    }

    /// <summary>Vulkan surface transform</summary>
    [Flags]
    public enum SurfaceTransformFlag
    {
        Identity = 0x00000001, Rotate90 = 0x00000002,
        Rotate180 = 0x00000004, Rotate270 = 0x00000008,
        HorizontalMirror = 0x00000010, HorizontalMirrorRotate90 = 0x00000020,
        HorizontalMirrorRotate180 = 0x00000040, HorizontalMirrorRotate270 = 0x00000080,
        Inherit = 0x00000100
    }

    /// <summary>Vulkan composite alpha</summary>
    [Flags]
    public enum CompositeAlphaFlag
    {
        Opaque = 0x00000001, PreMultiplied = 0x00000002,
        PostMultiplied = 0x00000004, Inherit = 0x00000008
    }

    /// <summary>Vulkan descriptor pool create flags</summary>
    [Flags]
    public enum DescriptorPoolCreateFlag
    {
        None = 0,
        FreeDescriptorSet = 0x00000001,
        UpdateAfterBind = 0x00000002,
        HostOnly = 0x00000004
    }

    /// <summary>Vulkan descriptor set layout create flags</summary>
    [Flags]
    public enum DescriptorSetLayoutCreateFlag
    {
        None = 0,
        PushDescriptor = 0x00000001,
        PushDescriptorKhr = 0x00000001,
        UpdateAfterBindPool = 0x00000002,
        HostOnlyPool = 0x00000004
    }

    /// <summary>Vulkan command pool create flags</summary>
    [Flags]
    public enum CommandPoolCreateFlag
    {
        None = 0,
        ResetCommandBuffer = 0x00000001,
        ResetPoolTransient = 0x00000002
    }

    /// <summary>Vulkan fence create flags</summary>
    [Flags]
    public enum FenceCreateFlag
    {
        None = 0,
        Signaled = 0x00000001
    }

    /// <summary>Vulkan semaphore type</summary>
    public enum SemaphoreType
    {
        Binary = 0,
        Timeline = 1
    }

    /// <summary>Vulkan query type</summary>
    public enum QueryType
    {
        Occlusion = 0,
        PipelineStatistics = 1,
        Timestamp = 2,
        AccelerationStructureCompactedSizeKHR = 1000150000,
        AccelerationStructureSerializationSizeKHR = 1000150001,
        AccelerationStructureInstanceBuffer = 1000156000
    }

    // =========================================================================
    // STRUCTS
    // =========================================================================

    /// <summary>Represents a viewport for rendering</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Viewport
    {
        public float X;
        public float Y;
        public float Width;
        public float Height;
        public float MinDepth;
        public float MaxDepth;

        public Viewport(float x, float y, float width, float height, float minDepth = 0f, float maxDepth = 1f)
        {
            X = x;
            Y = y;
            Width = width;
            Height = height;
            MinDepth = minDepth;
            MaxDepth = maxDepth;
        }
    }

    /// <summary>2D rectangle for scissor/viewport</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect2D
    {
        public Offset2D Offset;
        public Extent2D Extent;
        public Rect2D(int x, int y, uint width, uint height) { Offset = new Offset2D(x, y); Extent = new Extent2D(width, height); }
        public Rect2D(Offset2D offset, Extent2D extent) { Offset = offset; Extent = extent; }
    }

    /// <summary>2D integer offset</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Offset2D
    {
        public int X;
        public int Y;
        public Offset2D(int x, int y) { X = x; Y = y; }
    }

    /// <summary>2D unsigned extent</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Extent2D
    {
        public uint Width;
        public uint Height;
        public Extent2D(uint width, uint height) { Width = width; Height = height; }
    }

    /// <summary>3D unsigned extent</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Extent3D
    {
        public uint Width;
        public uint Height;
        public uint Depth;
        public Extent3D(uint width, uint height, uint depth) { Width = width; Height = height; Depth = depth; }
    }

    /// <summary>3D integer offset</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Offset3D
    {
        public int X;
        public int Y;
        public int Z;
        public Offset3D(int x, int y, int z) { X = x; Y = y; Z = z; }
    }

    /// <summary>Clear color value union</summary>
    [StructLayout(LayoutKind.Explicit, Size = 16)]
    public struct ClearColorValue
    {
        [FieldOffset(0)] public float Float0;
        [FieldOffset(4)] public float Float1;
        [FieldOffset(8)] public float Float2;
        [FieldOffset(12)] public float Float3;
        [FieldOffset(0)] public int Int0;
        [FieldOffset(4)] public int Int1;
        [FieldOffset(8)] public int Int2;
        [FieldOffset(12)] public int Int3;
        [FieldOffset(0)] public uint Uint0;
        [FieldOffset(4)] public uint Uint1;
        [FieldOffset(8)] public uint Uint2;
        [FieldOffset(12)] public uint Uint3;

        public static ClearColorValue Float(float r, float g, float b, float a) =>
            new ClearColorValue { Float0 = r, Float1 = g, Float2 = b, Float3 = a };
        public static ClearColorValue Int(int r, int g, int b, int a) =>
            new ClearColorValue { Int0 = r, Int1 = g, Int2 = b, Int3 = a };
        public static ClearColorValue Uint(uint r, uint g, uint b, uint a) =>
            new ClearColorValue { Uint0 = r, Uint1 = g, Uint2 = b, Uint3 = a };
    }

    /// <summary>Clear depth/stencil value</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ClearDepthStencilValue
    {
        public float Depth;
        public uint Stencil;
        public ClearDepthStencilValue(float depth, uint stencil) { Depth = depth; Stencil = stencil; }
    }

    /// <summary>Clear value union for render pass clears</summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct ClearValue
    {
        [FieldOffset(0)] public ClearColorValue Color;
        [FieldOffset(0)] public ClearDepthStencilValue DepthStencil;

        public static ClearValue ColorClear(float r, float g, float b, float a) =>
            new ClearValue { Color = ClearColorValue.Float(r, g, b, a) };
        public static ClearValue DepthStencilClear(float depth = 1.0f, uint stencil = 0) =>
            new ClearValue { DepthStencil = new ClearDepthStencilValue(depth, stencil) };
    }

    /// <summary>Global memory barrier</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MemoryBarrier
    {
        public AccessFlag SrcAccessMask;
        public AccessFlag DstAccessMask;
    }

    /// <summary>Buffer memory barrier</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BufferMemoryBarrier
    {
        public uint SrcQueueFamilyIndex;
        public uint DstQueueFamilyIndex;
        public IntPtr Buffer;
        public ulong Offset;
        public ulong Size;
        public AccessFlag SrcAccessMask;
        public AccessFlag DstAccessMask;
    }

    /// <summary>Image memory barrier</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ImageMemoryBarrier
    {
        public AccessFlag SrcAccessMask;
        public AccessFlag DstAccessMask;
        public ImageLayout OldLayout;
        public ImageLayout NewLayout;
        public uint SrcQueueFamilyIndex;
        public uint DstQueueFamilyIndex;
        public IntPtr Image;
        public ImageSubresourceRange SubresourceRange;
    }

    /// <summary>Image subresource range</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ImageSubresourceRange
    {
        public ImageAspectFlag AspectMask;
        public uint BaseMipLevel;
        public uint LevelCount;
        public uint BaseArrayLayer;
        public uint LayerCount;

        public ImageSubresourceRange(ImageAspectFlag aspectMask, uint baseMipLevel, uint levelCount, uint baseArrayLayer, uint layerCount)
        {
            AspectMask = aspectMask;
            BaseMipLevel = baseMipLevel;
            LevelCount = levelCount;
            BaseArrayLayer = baseArrayLayer;
            LayerCount = layerCount;
        }
    }

    /// <summary>Image subresource layers for copy operations</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ImageSubresourceLayers
    {
        public ImageAspectFlag AspectMask;
        public uint MipLevel;
        public uint BaseArrayLayer;
        public uint LayerCount;

        public ImageSubresourceLayers(ImageAspectFlag aspectMask, uint mipLevel, uint baseArrayLayer, uint layerCount)
        {
            AspectMask = aspectMask;
            MipLevel = mipLevel;
            BaseArrayLayer = baseArrayLayer;
            LayerCount = layerCount;
        }
    }

    /// <summary>Buffer copy region</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BufferCopy
    {
        public ulong SrcOffset;
        public ulong DstOffset;
        public ulong Size;
        public BufferCopy(ulong srcOffset, ulong dstOffset, ulong size) { SrcOffset = srcOffset; DstOffset = dstOffset; Size = size; }
    }

    /// <summary>Buffer to image copy region</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BufferImageCopy
    {
        public ulong BufferOffset;
        public uint BufferRowLength;
        public uint BufferImageHeight;
        public ImageSubresourceLayers ImageSubresource;
        public Offset3D ImageOffset;
        public Extent3D ImageExtent;
    }

    /// <summary>Push constant range</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PushConstantRange
    {
        public ShaderStageFlag StageFlags;
        public uint Offset;
        public uint Size;
        public PushConstantRange(ShaderStageFlag stageFlags, uint offset, uint size) { StageFlags = stageFlags; Offset = offset; Size = size; }
    }

    /// <summary>Vertex input attribute description</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct VertexInputAttributeDescription
    {
        public uint Location;
        public uint Binding;
        public VulkanFormat Format;
        public uint Offset;
        public VertexInputAttributeDescription(uint location, uint binding, VulkanFormat format, uint offset)
        { Location = location; Binding = binding; Format = format; Offset = offset; }
    }

    /// <summary>Vertex input binding description</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct VertexInputBindingDescription
    {
        public uint Binding;
        public uint Stride;
        public VertexInputRate InputRate;
        public VertexInputBindingDescription(uint binding, uint stride, VertexInputRate inputRate)
        { Binding = binding; Stride = stride; InputRate = inputRate; }
    }

    /// <summary>Stencil operation state</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct StencilOpState
    {
        public StencilOp FailOp;
        public StencilOp PassOp;
        public StencilOp DepthFailOp;
        public CompareOp CompareOp;
        public uint CompareMask;
        public uint WriteMask;
        public uint Reference;

        public StencilOpState(StencilOp failOp, StencilOp passOp, StencilOp depthFailOp, CompareOp compareOp)
        {
            FailOp = failOp;
            PassOp = passOp;
            DepthFailOp = depthFailOp;
            CompareOp = compareOp;
            CompareMask = 0xFFFFFFFF;
            WriteMask = 0xFFFFFFFF;
            Reference = 0;
        }
    }

    /// <summary>Pipeline color blend attachment state</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineColorBlendAttachmentState
    {
        public bool BlendEnable;
        public BlendFactor SrcColorBlendFactor;
        public BlendFactor DstColorBlendFactor;
        public BlendOp ColorBlendOp;
        public BlendFactor SrcAlphaBlendFactor;
        public BlendFactor DstAlphaBlendFactor;
        public BlendOp AlphaBlendOp;
        public ColorComponentFlag ColorWriteMask;

        public static PipelineColorBlendAttachmentState Disabled() => new PipelineColorBlendAttachmentState
        {
            BlendEnable = false,
            SrcColorBlendFactor = BlendFactor.One,
            DstColorBlendFactor = BlendFactor.Zero,
            ColorBlendOp = BlendOp.Add,
            SrcAlphaBlendFactor = BlendFactor.One,
            DstAlphaBlendFactor = BlendFactor.Zero,
            AlphaBlendOp = BlendOp.Add,
            ColorWriteMask = ColorComponentFlag.RGBA
        };

        public static PipelineColorBlendAttachmentState AlphaBlending() => new PipelineColorBlendAttachmentState
        {
            BlendEnable = true,
            SrcColorBlendFactor = BlendFactor.SrcAlpha,
            DstColorBlendFactor = BlendFactor.OneMinusSrcAlpha,
            ColorBlendOp = BlendOp.Add,
            SrcAlphaBlendFactor = BlendFactor.One,
            DstAlphaBlendFactor = BlendFactor.OneMinusSrcAlpha,
            AlphaBlendOp = BlendOp.Add,
            ColorWriteMask = ColorComponentFlag.RGBA
        };
    }

    /// <summary>Attachment description for render passes</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct AttachmentDescription
    {
        public AttachmentLoadOp LoadOp;
        public AttachmentStoreOp StoreOp;
        public AttachmentLoadOp StencilLoadOp;
        public AttachmentStoreOp StencilStoreOp;
        public ImageLayout InitialLayout;
        public ImageLayout FinalLayout;
        public VulkanFormat Format;
        public SampleCountFlag Samples;
    }

    /// <summary>Attachment reference</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct AttachmentReference
    {
        public uint Attachment;
        public ImageLayout Layout;
        public AttachmentReference(uint attachment, ImageLayout layout) { Attachment = attachment; Layout = layout; }
    }

    /// <summary>Subpass description</summary>
    public struct SubpassDescription
    {
        public PipelineBindPoint PipelineBindPoint;
        public AttachmentReference[] ColorAttachments;
        public AttachmentReference[] InputAttachments;
        public AttachmentReference[] ResolveAttachments;
        public AttachmentReference DepthStencilAttachment;
        public uint[] PreserveAttachments;
    }

    /// <summary>Subpass dependency</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SubpassDependency
    {
        public uint SrcSubpass;
        public uint DstSubpass;
        public PipelineStageFlag SrcStageMask;
        public PipelineStageFlag DstStageMask;
        public AccessFlag SrcAccessMask;
        public AccessFlag DstAccessMask;
        public DependencyFlag DependencyFlags;
    }

    /// <summary>Specialization map entry</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SpecializationMapEntry
    {
        public uint ConstantID;
        public uint Offset;
        public uint Size;
    }

    /// <summary>Specialization info</summary>
    public struct SpecializationInfo
    {
        public SpecializationMapEntry[] MapEntries;
        public byte[] Data;
    }

    /// <summary>Stencil op state for dynamic rendering</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineDepthStencilStateCreateInfo
    {
        public bool DepthTestEnable;
        public bool DepthWriteEnable;
        public CompareOp DepthCompareOp;
        public bool DepthBoundsTestEnable;
        public bool StencilTestEnable;
        public StencilOpState Front;
        public StencilOpState Back;
        public float MinDepthBounds;
        public float MaxDepthBounds;
    }

    /// <summary>Surface capabilities</summary>
    public struct SurfaceCapabilities
    {
        public uint MinImageCount;
        public uint MaxImageCount;
        public Extent2D CurrentExtent;
        public Extent2D MinImageExtent;
        public Extent2D MaxImageExtent;
        public uint MaxImageArrayLayers;
        public SurfaceTransformFlag SupportedTransforms;
        public SurfaceTransformFlag CurrentTransform;
        public CompositeAlphaFlag SupportedCompositeAlpha;
        public ImageUsageFlag SupportedUsageFlags;
    }

    /// <summary>Surface format</summary>
    public struct SurfaceFormatKHR
    {
        public VulkanFormat Format;
        public PresentMode ColorSpace;
    }

    /// <summary>Queue family properties</summary>
    public struct QueueFamilyProperties
    {
        public QueueFlag QueueFlags;
        public uint QueueCount;
        public uint TimestampValidBits;
        public Extent3D MinImageTransferGranularity;
    }

    /// <summary>Physical device memory properties</summary>
    public struct PhysicalDeviceMemoryProperties
    {
        public uint MemoryTypeCount;
        public MemoryType[] MemoryTypes;
        public uint MemoryHeapCount;
        public MemoryHeap[] MemoryHeaps;
    }

    /// <summary>Memory type</summary>
    public struct MemoryType
    {
        public MemoryPropertyFlag PropertyFlags;
        public uint HeapIndex;
    }

    /// <summary>Memory heap</summary>
    public struct MemoryHeap
    {
        public ulong Size;
        public MemoryHeapFlag Flags;
    }

    /// <summary>Memory heap flags</summary>
    [Flags]
    public enum MemoryHeapFlag
    {
        DeviceLocal = 0x00000001,
        MultiInstance = 0x00000002,
        MultiInstanceBit = 0x00000002
    }

    /// <summary>Buffer copy region with full parameters</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BufferCopy2
    {
        public ulong SrcOffset;
        public ulong DstOffset;
        public ulong Size;
    }

    /// <summary>Image blit region</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ImageBlit
    {
        public ImageSubresourceLayers SrcSubresource;
        public Offset3D[] SrcOffsets;
        public ImageSubresourceLayers DstSubresource;
        public Offset3D[] DstOffsets;
    }

    /// <summary>Image copy region</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ImageCopy
    {
        public ImageSubresourceLayers SrcSubresource;
        public Offset3D SrcOffset;
        public ImageSubresourceLayers DstSubresource;
        public Offset3D DstOffset;
        public Extent3D Extent;
    }

    /// <summary>Image resolve region</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ImageResolve
    {
        public ImageSubresourceLayers SrcSubresource;
        public Offset3D SrcOffset;
        public ImageSubresourceLayers DstSubresource;
        public Offset3D DstOffset;
        public Extent3D Extent;
    }

    /// <summary>Descriptor buffer binding info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorAddressInfoEXT
    {
        public ulong Address;
        public ulong Range;
        public VulkanFormat Format;
    }

    /// <summary>Write descriptor set acceleration structure</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct WriteDescriptorSetAccelerationStructureKHR
    {
        public uint AccelerationStructureCount;
        public IntPtr[] AccelerationStructures;
    }

    /// <summary>Timeline semaphore submit info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct TimelineSemaphoreSubmitInfo
    {
        public uint WaitSemaphoreValueCount;
        public ulong[] WaitSemaphoreValues;
        public uint SignalSemaphoreValueCount;
        public ulong[] SignalSemaphoreValues;
    }

    /// <summary>Pipeline shader stage create info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineShaderStageCreateInfo
    {
        public PipelineShaderStageCreateFlag Flags;
        public ShaderStageFlag Stage;
        public IntPtr Module;
        public string Name;
        public SpecializationInfo SpecializationInfo;
    }

    /// <summary>Pipeline shader stage create flags</summary>
    [Flags]
    public enum PipelineShaderStageCreateFlag
    {
        None = 0,
        RequireFullSubgroups = 0x00000001
    }

    /// <summary>Rasterization state create info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineRasterizationStateCreateInfo
    {
        public bool DepthClampEnable;
        public bool RasterizerDiscardEnable;
        public PolygonMode PolygonMode;
        public CullModeFlag CullMode;
        public FrontFace FrontFace;
        public bool DepthBiasEnable;
        public float DepthBiasConstantFactor;
        public float DepthBiasClamp;
        public float DepthBiasSlopeFactor;
        public float LineWidth;
    }

    /// <summary>Multisample state create info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineMultisampleStateCreateInfo
    {
        public SampleCountFlag RasterizationSamples;
        public bool SampleShadingEnable;
        public float MinSampleShading;
        public uint[] SampleMask;
        public bool AlphaToCoverageEnable;
        public bool AlphaToOneEnable;
    }

    /// <summary>Color blend state create info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineColorBlendStateCreateInfo
    {
        public bool LogicOpEnable;
        public LogicOp LogicOp;
        public PipelineColorBlendAttachmentState[] Attachments;
        public float[] BlendConstants;
    }

    /// <summary>Viewport state create info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineViewportStateCreateInfo
    {
        public uint ViewportCount;
        public uint ScissorCount;
        public bool IgnoreViewports;
        public bool IgnoreScissors;
    }

    /// <summary>Input assembly state create info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineInputAssemblyStateCreateInfo
    {
        public PrimitiveTopology Topology;
        public bool PrimitiveRestartEnable;
    }

    /// <summary>Tessellation state create info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineTessellationStateCreateInfo
    {
        public uint PatchControlPoints;
    }

    /// <summary>Dynamic state create info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineDynamicStateCreateInfo
    {
        public DynamicState[] DynamicStates;
    }

    /// <summary>Vertex input state create info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PipelineVertexInputStateCreateInfo
    {
        public VertexInputBindingDescription[] VertexBindingDescriptions;
        public VertexInputAttributeDescription[] VertexAttributeDescriptions;
    }

    // =========================================================================
    // DESCRIPTION STRUCTS
    // =========================================================================

    /// <summary>Description for creating an RHI device</summary>
}
