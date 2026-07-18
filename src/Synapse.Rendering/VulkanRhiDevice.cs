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
            X = x; Y = y; Width = width; Height = height;
            MinDepth = minDepth; MaxDepth = maxDepth;
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
            AspectMask = aspectMask; BaseMipLevel = baseMipLevel; LevelCount = levelCount;
            BaseArrayLayer = baseArrayLayer; LayerCount = layerCount;
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
            AspectMask = aspectMask; MipLevel = mipLevel;
            BaseArrayLayer = baseArrayLayer; LayerCount = layerCount;
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
            FailOp = failOp; PassOp = passOp; DepthFailOp = depthFailOp;
            CompareOp = compareOp; CompareMask = 0xFFFFFFFF; WriteMask = 0xFFFFFFFF; Reference = 0;
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
            BlendEnable = false, SrcColorBlendFactor = BlendFactor.One, DstColorBlendFactor = BlendFactor.Zero,
            ColorBlendOp = BlendOp.Add, SrcAlphaBlendFactor = BlendFactor.One,
            DstAlphaBlendFactor = BlendFactor.Zero, AlphaBlendOp = BlendOp.Add,
            ColorWriteMask = ColorComponentFlag.RGBA
        };

        public static PipelineColorBlendAttachmentState AlphaBlending() => new PipelineColorBlendAttachmentState
        {
            BlendEnable = true, SrcColorBlendFactor = BlendFactor.SrcAlpha,
            DstColorBlendFactor = BlendFactor.OneMinusSrcAlpha, ColorBlendOp = BlendOp.Add,
            SrcAlphaBlendFactor = BlendFactor.One, DstAlphaBlendFactor = BlendFactor.OneMinusSrcAlpha,
            AlphaBlendOp = BlendOp.Add, ColorWriteMask = ColorComponentFlag.RGBA
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
    public class RhiDeviceCreationInfo
    {
        public string ApplicationName { get; set; } = "GDNN Engine";
        public uint ApplicationVersion { get; set; } = 1;
        public string EngineName { get; set; } = "GDNN";
        public uint EngineVersion { get; set; } = 1;
        public uint ApiVersion { get; set; } = VK_API_VERSION_1_4;
        public string[] RequiredExtensions { get; set; } = Array.Empty<string>();
        public string[] RequiredLayers { get; set; } = Array.Empty<string>();
        public bool EnableValidation { get; set; } = true;
        public IntPtr SurfaceHandle { get; set; }

        public const uint VK_API_VERSION_1_4 = (1 << 22) | (4 << 12);
        public const uint VK_API_VERSION_1_3 = (1 << 22) | (3 << 12);
        public const uint VK_API_VERSION_1_2 = (1 << 22) | (2 << 12);
        public const uint VK_API_VERSION_1_1 = (1 << 22) | (1 << 12);
        public const uint VK_API_VERSION_1_0 = (1 << 22);
    }

    /// <summary>Buffer creation description</summary>
    public class BufferDescription
    {
        public ulong Size { get; set; }
        public BufferUsageFlag Usage { get; set; }
        public MemoryPropertyFlag MemoryProperties { get; set; }
        public SharingMode SharingMode { get; set; } = SharingMode.Exclusive;
        public uint[] QueueFamilyIndices { get; set; }
        public string DebugName { get; set; }
    }

    /// <summary>Texture creation description</summary>
    public class TextureDescription
    {
        public ImageType Type { get; set; } = ImageType.Type2D;
        public VulkanFormat Format { get; set; } = VulkanFormat.R8G8B8A8Unorm;
        public uint Width { get; set; } = 1;
        public uint Height { get; set; } = 1;
        public uint Depth { get; set; } = 1;
        public uint MipLevels { get; set; } = 1;
        public uint ArrayLayers { get; set; } = 1;
        public SampleCountFlag Samples { get; set; } = SampleCountFlag.Count1;
        public ImageTiling Tiling { get; set; } = ImageTiling.Optimal;
        public ImageUsageFlag Usage { get; set; } = ImageUsageFlag.Sampled | ImageUsageFlag.TransferDst;
        public SharingMode SharingMode { get; set; } = SharingMode.Exclusive;
        public ImageLayout InitialLayout { get; set; } = ImageLayout.Undefined;
        public uint[] QueueFamilyIndices { get; set; }
        public string DebugName { get; set; }
    }

    /// <summary>Render pass creation description</summary>
    public class RenderPassDescription
    {
        public AttachmentDescription[] Attachments { get; set; }
        public SubpassDescription[] Subpasses { get; set; }
        public SubpassDependency[] Dependencies { get; set; }
        public string DebugName { get; set; }
    }

    /// <summary>Framebuffer creation description</summary>
    public class FramebufferDescription
    {
        public IntPtr RenderPass { get; set; }
        public IntPtr[] Attachments { get; set; }
        public uint Width { get; set; }
        public uint Height { get; set; }
        public uint Layers { get; set; } = 1;
        public string DebugName { get; set; }
    }

    /// <summary>Pipeline creation description</summary>
    public class PipelineDescription
    {
        public PipelineShaderStageCreateInfo[] ShaderStages { get; set; }
        public PipelineVertexInputStateCreateInfo VertexInputState { get; set; }
        public PipelineInputAssemblyStateCreateInfo InputAssemblyState { get; set; }
        public PipelineTessellationStateCreateInfo TessellationState { get; set; }
        public PipelineViewportStateCreateInfo ViewportState { get; set; }
        public PipelineRasterizationStateCreateInfo RasterizationState { get; set; }
        public PipelineMultisampleStateCreateInfo MultisampleState { get; set; }
        public PipelineDepthStencilStateCreateInfo DepthStencilState { get; set; }
        public PipelineColorBlendStateCreateInfo ColorBlendState { get; set; }
        public PipelineDynamicStateCreateInfo DynamicState { get; set; }
        public IntPtr PipelineLayout { get; set; }
        public IntPtr RenderPass { get; set; }
        public uint Subpass { get; set; }
        public PipelineCreateFlag Flags { get; set; }
        public IntPtr BasePipelineHandle { get; set; } = IntPtr.Zero;
        public int BasePipelineIndex { get; set; } = -1;
        public string DebugName { get; set; }
    }

    /// <summary>Compute pipeline creation description</summary>
    public class ComputePipelineDescription
    {
        public PipelineShaderStageCreateInfo Stage { get; set; }
        public IntPtr PipelineLayout { get; set; }
        public PipelineCreateFlag Flags { get; set; }
        public IntPtr BasePipelineHandle { get; set; } = IntPtr.Zero;
        public int BasePipelineIndex { get; set; } = -1;
        public string DebugName { get; set; }
    }

    /// <summary>Descriptor set layout description</summary>
    public class LayoutDescription
    {
        public DescriptorSetLayoutBinding[] Bindings { get; set; }
        public DescriptorSetLayoutCreateFlag Flags { get; set; }
        public string DebugName { get; set; }
    }

    /// <summary>Descriptor set layout binding</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorSetLayoutBinding
    {
        public uint Binding;
        public DescriptorType DescriptorType;
        public uint DescriptorCount;
        public ShaderStageFlag StageFlags;
        public IntPtr[] ImmutableSamplers;
    }

    /// <summary>Descriptor pool description</summary>
    public class PoolDescription
    {
        public DescriptorPoolCreateFlag Flags { get; set; }
        public uint MaxSets { get; set; }
        public DescriptorPoolSize[] PoolSizes { get; set; }
        public string DebugName { get; set; }
    }

    /// <summary>Descriptor pool size</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorPoolSize
    {
        public DescriptorType Type;
        public uint DescriptorCount;
    }

    /// <summary>Descriptor set allocation info</summary>
    public class DescriptorSetAllocation
    {
        public IntPtr Pool { get; set; }
        public IntPtr[] Layouts { get; set; }
    }

    /// <summary>Descriptor write operation</summary>
    public class DescriptorWrite
    {
        public IntPtr DescriptorSet { get; set; }
        public uint DstBinding { get; set; }
        public uint DstArrayElement { get; set; }
        public DescriptorType DescriptorType { get; set; }
        public DescriptorImageInfo[] ImageInfos { get; set; }
        public DescriptorBufferInfo[] BufferInfos { get; set; }
    }

    /// <summary>Descriptor image info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorImageInfo
    {
        public IntPtr Sampler;
        public IntPtr ImageView;
        public ImageLayout ImageLayout;
    }

    /// <summary>Descriptor buffer info</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct DescriptorBufferInfo
    {
        public IntPtr Buffer;
        public ulong Offset;
        public ulong Range;
    }

    /// <summary>Sampler creation description</summary>
    public class SamplerDescription
    {
        public Filter MagFilter { get; set; } = Filter.Linear;
        public Filter MinFilter { get; set; } = Filter.Linear;
        public SamplerAddressMode AddressModeU { get; set; } = SamplerAddressMode.Repeat;
        public SamplerAddressMode AddressModeV { get; set; } = SamplerAddressMode.Repeat;
        public SamplerAddressMode AddressModeW { get; set; } = SamplerAddressMode.Repeat;
        public float MipLodBias { get; set; } = 0.0f;
        public bool AnisotropyEnable { get; set; } = true;
        public float MaxAnisotropy { get; set; } = 16.0f;
        public bool CompareEnable { get; set; } = false;
        public CompareOp CompareOp { get; set; } = CompareOp.Always;
        public float MinLod { get; set; } = 0.0f;
        public float MaxLod { get; set; } = 1000.0f;
        public SampleCountFlag SampleCountFlags { get; set; } = SampleCountFlag.Count1;
        public string DebugName { get; set; }
    }

    /// <summary>Hardware capabilities query result</summary>
    public class HardwareCapabilities
    {
        public string DeviceName { get; set; }
        public string DriverVersion { get; set; }
        public string VendorId { get; set; }
        public uint DeviceId { get; set; }
        public uint ApiVersion { get; set; }
        public ulong DedicatedVideoMemory { get; set; }
        public ulong DedicatedHostMemory { get; set; }
        public ulong DedicatedDeviceMemory { get; set; }
        public ulong SharedVideoMemory { get; set; }
        public ulong SharedHostMemory { get; set; }
        public ulong SharedDeviceMemory { get; set; }
        public SampleCountFlag MaxColorSampleCounts { get; set; }
        public SampleCountFlag MaxDepthSampleCounts { get; set; }
        public SampleCountFlag MaxStencilSampleCounts { get; set; }
        public SampleCountFlag MaxStorageImageSampleCounts { get; set; }
        public uint MaxTextureDimension1D { get; set; }
        public uint MaxTextureDimension2D { get; set; }
        public uint MaxTextureDimension3D { get; set; }
        public uint MaxTextureArrayLayers { get; set; }
        public uint MaxUniformBufferRange { get; set; }
        public uint MaxStorageBufferRange { get; set; }
        public uint MaxPushConstantsSize { get; set; }
        public uint MaxComputeWorkGroupCount0 { get; set; }
        public uint MaxComputeWorkGroupCount1 { get; set; }
        public uint MaxComputeWorkGroupCount2 { get; set; }
        public uint MaxComputeWorkGroupSize0 { get; set; }
        public uint MaxComputeWorkGroupSize1 { get; set; }
        public uint MaxComputeWorkGroupSize2 { get; set; }
        public uint MaxComputeWorkGroupInvocations { get; set; }
        public uint MaxBoundDescriptorSets { get; set; }
        public uint MaxColorAttachments { get; set; }
        public uint MaxVertexInputAttributes { get; set; }
        public uint MaxVertexInputBindings { get; set; }
        public uint MaxVertexInputAttributeOffset { get; set; }
        public uint MaxVertexInputBindingStride { get; set; }
        public uint MaxVertexOutputComponents { get; set; }
        public float MaxSamplerAnisotropy { get; set; }
        public uint SubgroupPropertiesSubgroupSize { get; set; }
        public bool SupportsTimelineSemaphores { get; set; }
        public bool SupportsBufferDeviceAddress { get; set; }
        public bool SupportsDynamicRendering { get; set; }
        public bool SupportsSynchronization2 { get; set; }
        public PhysicalDeviceMemoryProperties MemoryProperties { get; set; }
        public string[] SupportedExtensions { get; set; }
    }

    /// <summary>Pipeline cache creation info</summary>
    public class PipelineCacheInfo
    {
        public byte[] InitialData { get; set; }
    }

    /// <summary>Command buffer allocation info</summary>
    public class CommandBufferAllocationInfo
    {
        public IntPtr CommandPool { get; set; }
        public uint Level { get; set; } = 0;
        public uint CommandBufferCount { get; set; } = 1;
    }

    // =========================================================================
    // INTERFACES
    // =========================================================================

    /// <summary>Core RHI device interface</summary>
    public interface IRhiDevice : IDisposable
    {
        IntPtr PhysicalDevice { get; }
        IntPtr Device { get; }
        IntPtr Instance { get; }
        IntPtr Surface { get; }
        IntPtr Queue { get; }
        VulkanSwapchain Swapchain { get; }

        VulkanSwapchain CreateSwapchain(IntPtr surface);
        QueueFamilyIndices GetQueueFamilies();
        VulkanBuffer CreateBuffer(BufferDescription description);
        VulkanTexture CreateTexture(TextureDescription description);
        VulkanRenderPass CreateRenderPass(RenderPassDescription description);
        VulkanFramebuffer CreateFramebuffer(FramebufferDescription description);
        VulkanPipeline CreatePipeline(PipelineDescription description);
        VulkanComputePipeline CreateComputePipeline(ComputePipelineDescription description);
        DescriptorSetLayout CreateDescriptorSetLayout(LayoutDescription description);
        DescriptorPool CreateDescriptorPool(PoolDescription description);
        DescriptorSet[] AllocateDescriptorSets(DescriptorSetAllocation allocation);
        void UpdateDescriptorSets(DescriptorWrite[] writes);
        VulkanShaderModule CreateShaderModule(byte[] spirvCode);
        Sampler CreateSampler(SamplerDescription description);
        VulkanCommandBuffer CreateCommandBuffer();
        void SubmitCommandBuffer(VulkanCommandBuffer commandBuffer, IntPtr queue, Fence fence);
        void WaitForIdle();
        HardwareCapabilities QueryCapabilities();
    }

    /// <summary>Interface for disposable Vulkan objects</summary>
    public interface IVulkanObject : IDisposable
    {
        IntPtr Handle { get; }
        string DebugName { get; set; }
    }

    // =========================================================================
    // HELPER CLASSES
    // =========================================================================

    /// <summary>Queue family index lookup results</summary>
    public class QueueFamilyIndices
    {
        public int GraphicsFamily { get; set; } = -1;
        public int ComputeFamily { get; set; } = -1;
        public int TransferFamily { get; set; } = -1;
        public int PresentFamily { get; set; } = -1;
        public int SparseBindingFamily { get; set; } = -1;

        public bool IsComplete => GraphicsFamily >= 0 && PresentFamily >= 0;
        public bool HasCompute => ComputeFamily >= 0;
        public bool HasTransfer => TransferFamily >= 0;

        public uint[] UniqueQueueFamilies
        {
            get
            {
                var families = new HashSet<int>();
                if (GraphicsFamily >= 0) families.Add(GraphicsFamily);
                if (ComputeFamily >= 0) families.Add(ComputeFamily);
                if (TransferFamily >= 0) families.Add(TransferFamily);
                if (PresentFamily >= 0) families.Add(PresentFamily);
                return families.Select(x => (uint)x).ToArray();
            }
        }
    }

    /// <summary>Physical device info</summary>
    public class VulkanPhysicalDeviceInfo
    {
        public IntPtr Handle { get; set; }
        public string DeviceName { get; set; }
        public uint VendorId { get; set; }
        public uint DeviceId { get; set; }
        public uint ApiVersion { get; set; }
        public uint DriverVersion { get; set; }
        public QueueFamilyProperties[] QueueFamilyProperties { get; set; }
        public PhysicalDeviceMemoryProperties MemoryProperties { get; set; }
        public SampleCountFlag MaxColorBufferSampleCounts { get; set; }
        public SampleCountFlag MaxDepthBufferSampleCounts { get; set; }
        public uint MaxTextureDimension1D { get; set; }
        public uint MaxTextureDimension2D { get; set; }
        public uint MaxTextureDimension3D { get; set; }
        public uint MaxTextureArrayLayers { get; set; }
        public uint MaxUniformBufferRange { get; set; }
        public uint MaxStorageBufferRange { get; set; }
        public uint MaxPushConstantsSize { get; set; }
        public uint MaxBoundDescriptorSets { get; set; }
        public uint MaxColorAttachments { get; set; }
        public uint MaxVertexInputAttributes { get; set; }
        public uint MaxVertexInputBindings { get; set; }
        public uint MaxVertexInputAttributeOffset { get; set; }
        public uint MaxVertexInputBindingStride { get; set; }
        public uint MaxVertexOutputComponents { get; set; }
        public float MaxSamplerAnisotropy { get; set; }
        public uint MaxComputeWorkGroupCount0 { get; set; }
        public uint MaxComputeWorkGroupCount1 { get; set; }
        public uint MaxComputeWorkGroupCount2 { get; set; }
        public uint MaxComputeWorkGroupSize0 { get; set; }
        public uint MaxComputeWorkGroupSize1 { get; set; }
        public uint MaxComputeWorkGroupSize2 { get; set; }
        public uint MaxComputeWorkGroupInvocations { get; set; }
        public bool SupportsTimelineSemaphores { get; set; }
        public bool SupportsBufferDeviceAddress { get; set; }
        public bool SupportsDynamicRendering { get; set; }
        public bool SupportsSynchronization2 { get; set; }
        public bool SupportsDescriptorIndexing { get; set; }
        public bool SupportsMaintenance4 { get; set; }
        public float GpuTimestampPeriod { get; set; }
    }

    /// <summary>Swapchain support details</summary>
    public class VulkanSwapchainSupportDetails
    {
        public SurfaceCapabilities Capabilities { get; set; }
        public SurfaceFormatKHR[] Formats { get; set; }
        public PresentMode[] PresentModes { get; set; }

        public SurfaceFormatKHR ChooseSurfaceFormat(VulkanFormat preferred = VulkanFormat.B8G8R8A8Srgb, PresentMode colorSpace = PresentMode.Fifo)
        {
            if (Formats != null)
            {
                foreach (var format in Formats)
                {
                    if (format.Format == preferred && (int)format.ColorSpace == (int)colorSpace)
                        return format;
                }
                if (Formats.Length > 0) return Formats[0];
            }
            return new SurfaceFormatKHR { Format = preferred, ColorSpace = colorSpace };
        }

        public PresentMode ChoosePresentMode(PresentMode preferred = PresentMode.Mailbox)
        {
            if (PresentModes != null)
            {
                foreach (var mode in PresentModes)
                {
                    if (mode == preferred) return mode;
                }
            }
            return PresentMode.Fifo;
        }

        public Extent2D ChooseExtent(uint width, uint height)
        {
            var extent = Capabilities.CurrentExtent;
            if (extent.Width != 0xFFFFFFFF && extent.Height != 0xFFFFFFFF)
                return extent;

            return new Extent2D
            {
                Width = Math.Max(Capabilities.MinImageExtent.Width, Math.Min(Capabilities.MaxImageExtent.Width, width)),
                Height = Math.Max(Capabilities.MinImageExtent.Height, Math.Min(Capabilities.MaxImageExtent.Height, height))
            };
        }
    }

    /// <summary>Vulkan handle wrapper base class</summary>
    public abstract class VulkanHandle : IVulkanObject
    {
        protected IntPtr _handle;
        protected VulkanDevice _device;
        protected bool _disposed;

        public IntPtr Handle => _handle;
        public string DebugName { get; set; }

        protected VulkanHandle(VulkanDevice device, IntPtr handle)
        {
            _device = device;
            _handle = handle;
        }

        public abstract void Dispose();
    }

    /// <summary>Wrapper for Vulkan logical device</summary>
    public class VulkanDevice
    {
        public IntPtr Instance { get; set; }
        public IntPtr PhysicalDevice { get; set; }
        public IntPtr LogicalDevice { get; set; }
        public IntPtr Surface { get; set; }
        public IntPtr GraphicsQueue { get; set; }
        public IntPtr ComputeQueue { get; set; }
        public IntPtr TransferQueue { get; set; }
        public IntPtr PresentQueue { get; set; }
        public QueueFamilyIndices QueueIndices { get; set; }
        public VulkanPhysicalDeviceInfo PhysicalDeviceInfo { get; set; }
        public VulkanMemoryAllocator MemoryAllocator { get; set; }
        public VulkanSyncManager SyncManager { get; set; }
        public VulkanDescriptorManager DescriptorManager { get; set; }
        public VulkanPipelineCache PipelineCache { get; set; }
        public VulkanResourceTracker ResourceTracker { get; set; }

        // Pfn delegates for extension functions
        internal IntPtr pfnGetDeviceProcAddr;

        public VulkanDevice()
        {
            QueueIndices = new QueueFamilyIndices();
            PhysicalDeviceInfo = new VulkanPhysicalDeviceInfo();
        }

        public IntPtr GetDeviceProcAddr(string name)
        {
            if (pfnGetDeviceProcAddr != IntPtr.Zero)
            {
                var namePtr = Marshal.StringToHGlobalAnsi(name);
                try
                {
                    return vkGetDeviceProcAddr(LogicalDevice, namePtr);
                }
                finally
                {
                    Marshal.FreeHGlobal(namePtr);
                }
            }
            return IntPtr.Zero;
        }

        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern IntPtr vkGetDeviceProcAddr(IntPtr device, IntPtr pName);
    }

    /// <summary>
    /// Vulkan 1.4 Render Hardware Interface device implementation.
    /// Manages all Vulkan resources and provides a complete RHI abstraction
    /// for the G-DNN Engine rendering pipeline.
    /// </summary>
    public class VulkanRhiDevice : IRhiDevice
    {
        // Vulkan native handles
        private IntPtr _instance;
        private IntPtr _physicalDevice;
        private IntPtr _logicalDevice;
        private IntPtr _surface;
        private IntPtr _graphicsQueue;
        private IntPtr _computeQueue;
        private IntPtr _transferQueue;
        private IntPtr _presentQueue;
        private IntPtr _pipelineCache;
        private IntPtr _commandPool;

        // Supporting objects
        private VulkanDevice _device;
        private VulkanSwapchain _swapchain;
        private VulkanMemoryAllocator _memoryAllocator;
        private VulkanSyncManager _syncManager;
        private VulkanDescriptorManager _descriptorManager;
        private VulkanPipelineCache _pipelineCacheWrapper;
        private VulkanResourceTracker _resourceTracker;
        private VulkanPhysicalDeviceInfo _physicalDeviceInfo;
        private QueueFamilyIndices _queueFamilyIndices;
        private RhiDeviceCreationInfo _creationInfo;

        // Vulkan function pointers
        private IntPtr _vkGetInstanceProcAddr;
        private IntPtr _vkGetDeviceProcAddr;

        // Validation
        private IntPtr _debugUtilsMessenger;
        private IntPtr _validationLayer;

        // Device state
        private bool _disposed;
        private readonly object _deviceLock = new object();
        private readonly List<VulkanBuffer> _buffers = new List<VulkanBuffer>();
        private readonly List<VulkanTexture> _textures = new List<VulkanTexture>();
        private readonly List<VulkanRenderPass> _renderPasses = new List<VulkanRenderPass>();
        private readonly List<VulkanFramebuffer> _framebuffers = new List<VulkanFramebuffer>();
        private readonly List<VulkanPipeline> _pipelines = new List<VulkanPipeline>();
        private readonly List<VulkanComputePipeline> _computePipelines = new List<VulkanComputePipeline>();
        private readonly List<VulkanCommandBuffer> _commandBuffers = new List<VulkanCommandBuffer>();
        private readonly List<VulkanShaderModule> _shaderModules = new List<VulkanShaderModule>();
        private readonly List<Sampler> _samplers = new List<Sampler>();
        private readonly List<DescriptorPool> _descriptorPools = new List<DescriptorPool>();
        private readonly List<DescriptorSetLayout> _descriptorSetLayouts = new List<DescriptorSetLayout>();

        // Delegate types
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VkBool32 VkDebugUtilsMessengerCallbackDelegate(uint messageSeverity, uint messageTypes, IntPtr pCallbackData, IntPtr pUserData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult VkCreateDebugUtilsMessengerDelegate(IntPtr instance, ref VkDebugUtilsMessengerCreateInfoEXT pCreateInfo, IntPtr pAllocator, ref IntPtr pMessenger);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void VkDestroyDebugUtilsMessengerDelegate(IntPtr instance, IntPtr messenger, IntPtr pAllocator);

        private VkCreateDebugUtilsMessengerDelegate _vkCreateDebugUtilsMessenger;
        private VkDestroyDebugUtilsMessengerDelegate _vkDestroyDebugUtilsMessenger;

        // Core Vulkan P/Invokes
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern IntPtr vkGetInstanceProcAddr(IntPtr instance, IntPtr pName);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern IntPtr vkGetDeviceProcAddr(IntPtr device, IntPtr pName);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkCreateInstance(ref VkInstanceCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pInstance);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern void vkDestroyInstance(IntPtr instance, IntPtr pAllocator);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkEnumeratePhysicalDevices(IntPtr instance, ref uint pPhysicalDeviceCount, IntPtr pPhysicalDevices);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern void vkGetPhysicalDeviceProperties(IntPtr physicalDevice, IntPtr pProperties);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern void vkGetPhysicalDeviceQueueFamilyProperties(IntPtr physicalDevice, ref uint pQueueFamilyPropertyCount, IntPtr pQueueFamilyProperties);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkGetPhysicalDeviceSurfaceSupport(IntPtr physicalDevice, uint queueFamilyIndex, IntPtr surface, ref IntPtr pSupported);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkGetPhysicalDeviceSurfaceCapabilities(IntPtr physicalDevice, IntPtr surface, IntPtr pSurfaceCapabilities);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkGetPhysicalDeviceSurfaceFormats(IntPtr physicalDevice, IntPtr surface, ref uint pSurfaceFormatCount, IntPtr pSurfaceFormats);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkGetPhysicalDeviceSurfacePresentModes(IntPtr physicalDevice, IntPtr surface, ref uint pPresentModeCount, IntPtr pPresentModes);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern void vkGetPhysicalDeviceMemoryProperties(IntPtr physicalDevice, IntPtr pMemoryProperties);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern VulkanResult vkCreateDevice(IntPtr physicalDevice, ref VkDeviceCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pDevice);
        [DllImport("vulkan-1.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern void vkDestroyDevice(IntPtr device, IntPtr pAllocator);

        private T GetDeviceProc<T>(string name) where T : Delegate
        {
            var namePtr = Marshal.StringToHGlobalAnsi(name);
            try
            {
                var ptr = vkGetDeviceProcAddr(_logicalDevice, namePtr);
                if (ptr == IntPtr.Zero) return null;
                return Marshal.GetDelegateForFunctionPointer<T>(ptr);
            }
            finally { Marshal.FreeHGlobal(namePtr); }
        }

        private T GetInstanceProc<T>(string name) where T : Delegate
        {
            var namePtr = Marshal.StringToHGlobalAnsi(name);
            try
            {
                var ptr = vkGetInstanceProcAddr(_instance, namePtr);
                if (ptr == IntPtr.Zero) return null;
                return Marshal.GetDelegateForFunctionPointer<T>(ptr);
            }
            finally { Marshal.FreeHGlobal(namePtr); }
        }

        // Device function delegate types
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void GetDeviceQueueDelegate(IntPtr device, uint queueFamilyIndex, uint queueIndex, ref IntPtr pQueue);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult DeviceWaitIdleDelegate(IntPtr device);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateSwapchainKHRDelegate(IntPtr device, ref VkSwapchainCreateInfoKHR pCreateInfo, IntPtr pAllocator, ref IntPtr pSwapchain);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroySwapchainKHRDelegate(IntPtr device, IntPtr swapchain, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AcquireNextImageKHRDelegate(IntPtr device, IntPtr swapchain, ulong timeout, IntPtr semaphore, IntPtr fence, ref uint pImageIndex);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult QueuePresentKHRDelegate(IntPtr queue, ref VkPresentInfoKHR pPresentInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult QueueSubmitDelegate(IntPtr queue, uint submitCount, ref VkSubmitInfo pSubmits, IntPtr fence);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult QueueWaitIdleDelegate(IntPtr queue);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateFenceDelegate(IntPtr device, ref VkFenceCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pFence);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyFenceDelegate(IntPtr device, IntPtr fence, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult WaitForFencesDelegate(IntPtr device, uint fenceCount, IntPtr pFences, uint waitAll, ulong timeout);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult ResetFencesDelegate(IntPtr device, uint fenceCount, IntPtr pFences);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateSemaphoreDelegate(IntPtr device, ref VkSemaphoreCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pSemaphore);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroySemaphoreDelegate(IntPtr device, IntPtr semaphore, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateCommandPoolDelegate(IntPtr device, ref VkCommandPoolCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pCommandPool);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyCommandPoolDelegate(IntPtr device, IntPtr commandPool, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AllocateCommandBuffersDelegate(IntPtr device, ref VkCommandBufferAllocateInfo pAllocateInfo, IntPtr pCommandBuffers);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void FreeCommandBuffersDelegate(IntPtr device, IntPtr commandPool, uint commandBufferCount, IntPtr pCommandBuffers);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BeginCommandBufferDelegate(IntPtr commandBuffer, ref VkCommandBufferBeginInfo pBeginInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult EndCommandBufferDelegate(IntPtr commandBuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginRenderPassDelegate(IntPtr cmdBuffer, ref VkRenderPassBeginInfo pRenderPassBegin, uint contents);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndRenderPassDelegate(IntPtr cmdBuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindPipelineDelegate(IntPtr cmdBuffer, PipelineBindPoint pipelineBindPoint, IntPtr pipeline);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDrawDelegate(IntPtr cmdBuffer, uint vertexCount, uint instanceCount, uint firstVertex, uint firstInstance);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDrawIndexedDelegate(IntPtr cmdBuffer, uint indexCount, uint instanceCount, uint firstIndex, int vertexOffset, uint firstInstance);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDispatchDelegate(IntPtr cmdBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdPipelineBarrierDelegate(IntPtr cmdBuffer, PipelineStageFlag srcStageMask, PipelineStageFlag dstStageMask, uint dependencyFlags, uint memoryBarrierCount, IntPtr pMemoryBarriers, uint bufferMemoryBarrierCount, IntPtr pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferDelegate(IntPtr cmdBuffer, IntPtr srcBuffer, IntPtr dstBuffer, uint regionCount, ref BufferCopy pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferToImageDelegate(IntPtr cmdBuffer, IntPtr buffer, IntPtr image, uint imageLayout, uint regionCount, ref BufferImageCopy pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdPushConstantsDelegate(IntPtr cmdBuffer, IntPtr layout, ShaderStageFlag stageFlags, uint offset, uint size, IntPtr pValues);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetViewportDelegate(IntPtr cmdBuffer, uint firstViewport, uint viewportCount, ref Viewport pViewports);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetScissorDelegate(IntPtr cmdBuffer, uint firstScissor, uint scissorCount, ref Rect2D pScissors);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindVertexBuffersDelegate(IntPtr cmdBuffer, uint firstBinding, uint bindingCount, ref IntPtr pBuffers, ref ulong pOffsets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindIndexBufferDelegate(IntPtr cmdBuffer, IntPtr buffer, ulong offset, IndexType indexType);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateBufferDelegate(IntPtr device, ref VkBufferCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pBuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyBufferDelegate(IntPtr device, IntPtr buffer, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void GetBufferMemoryRequirementsDelegate(IntPtr device, IntPtr buffer, out VkMemoryRequirements pMemoryRequirements);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindBufferMemoryDelegate(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AllocateMemoryDelegate(IntPtr device, ref VkMemoryAllocateInfo pAllocateInfo, IntPtr pAllocator, ref IntPtr pMemory);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void FreeMemoryDelegate(IntPtr device, IntPtr memory, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult MapMemoryDelegate(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, ref IntPtr ppData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void UnmapMemoryDelegate(IntPtr device, IntPtr memory);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult FlushMappedMemoryRangesDelegate(IntPtr device, uint memoryRangeCount, ref VkMappedMemoryRange pMemoryRanges);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateImageDelegate(IntPtr device, ref VkImageCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pImage);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageDelegate(IntPtr device, IntPtr image, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void GetImageMemoryRequirementsDelegate(IntPtr device, IntPtr image, out VkMemoryRequirements pMemoryRequirements);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindImageMemoryDelegate(IntPtr device, IntPtr image, IntPtr memory, ulong memoryOffset);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateImageViewDelegate(IntPtr device, ref VkImageViewCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pView);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageViewDelegate(IntPtr device, IntPtr imageView, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateSamplerDelegate(IntPtr device, ref VkSamplerCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pSampler);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroySamplerDelegate(IntPtr device, IntPtr sampler, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateRenderPassDelegate(IntPtr device, ref VkRenderPassCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pRenderPass);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyRenderPassDelegate(IntPtr device, IntPtr renderPass, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateFramebufferDelegate(IntPtr device, ref VkFramebufferCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pFramebuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyFramebufferDelegate(IntPtr device, IntPtr framebuffer, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateShaderModuleDelegate(IntPtr device, ref VkShaderModuleCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pShaderModule);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyShaderModuleDelegate(IntPtr device, IntPtr shaderModule, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateGraphicsPipelinesDelegate(IntPtr device, IntPtr pipelineCache, uint createInfoCount, ref VkGraphicsPipelineCreateInfo pCreateInfos, IntPtr pAllocator, ref IntPtr pPipelines);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateComputePipelinesDelegate(IntPtr device, IntPtr pipelineCache, uint createInfoCount, ref VkComputePipelineCreateInfo pCreateInfos, IntPtr pAllocator, ref IntPtr pPipelines);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineDelegate(IntPtr device, IntPtr pipeline, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreatePipelineLayoutDelegate(IntPtr device, ref VkPipelineLayoutCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pPipelineLayout);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineLayoutDelegate(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateDescriptorSetLayoutDelegate(IntPtr device, ref VkDescriptorSetLayoutCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pSetLayout);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyDescriptorSetLayoutDelegate(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateDescriptorPoolDelegate(IntPtr device, ref VkDescriptorPoolCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pDescriptorPool);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyDescriptorPoolDelegate(IntPtr device, IntPtr descriptorPool, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AllocateDescriptorSetsDelegate(IntPtr device, ref VkDescriptorSetAllocateInfo pAllocateInfo, IntPtr pDescriptorSets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult FreeDescriptorSetsDelegate(IntPtr device, IntPtr descriptorPool, uint descriptorSetCount, IntPtr pDescriptorSets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void UpdateDescriptorSetsDelegate(IntPtr device, uint descriptorWriteCount, IntPtr pDescriptorWrites, uint descriptorCopyCount, IntPtr pDescriptorCopies);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreatePipelineCacheDelegate(IntPtr device, ref VkPipelineCacheCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pPipelineCache);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineCacheDelegate(IntPtr device, IntPtr pipelineCache, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult GetPipelineCacheDataDelegate(IntPtr device, IntPtr pipelineCache, ref IntPtr pDataSize, IntPtr pData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult MergePipelineCachesDelegate(IntPtr device, IntPtr dstCache, uint srcCacheCount, IntPtr pSrcCaches);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyImageDelegate(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBlitImageDelegate(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions, Filter filter);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdResolveImageDelegate(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateQueryPoolDelegate(IntPtr device, IntPtr pCreateInfo, IntPtr pAllocator, ref IntPtr pQueryPool);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyQueryPoolDelegate(IntPtr device, IntPtr queryPool, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult ResetQueryPoolDelegate(IntPtr device, IntPtr queryPool, uint firstQuery, uint queryCount);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult GetQueryPoolResultsDelegate(IntPtr device, IntPtr queryPool, uint firstQuery, uint queryCount, IntPtr dataSize, IntPtr pData, ulong stride, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdWriteTimestampDelegate(IntPtr cmdBuffer, PipelineStageFlag pipelineStage, IntPtr queryPool, uint query);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdResetQueryPoolDelegate(IntPtr cmdBuffer, IntPtr queryPool, uint firstQuery, uint queryCount);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyQueryPoolResultsDelegate(IntPtr cmdBuffer, IntPtr queryPool, uint firstQuery, uint queryCount, IntPtr dstBuffer, ulong dstOffset, ulong stride, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginQueryDelegate(IntPtr cmdBuffer, IntPtr queryPool, uint query, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndQueryDelegate(IntPtr cmdBuffer, IntPtr queryPool, uint query);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginRenderingDelegate(IntPtr cmdBuffer, IntPtr pRenderingInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndRenderingDelegate(IntPtr cmdBuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetDepthBiasEnableDelegate(IntPtr cmdBuffer, uint depthBiasEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetPrimitiveRestartEnableDelegate(IntPtr cmdBuffer, uint primitiveRestartEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetRasterizerDiscardEnableDelegate(IntPtr cmdBuffer, uint rasterizerDiscardEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateRenderPass2Delegate(IntPtr device, IntPtr pCreateInfo, IntPtr pAllocator, ref IntPtr pRenderPass);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginRenderPass2Delegate(IntPtr cmdBuffer, IntPtr pRenderPassBegin, IntPtr pSubpassBeginInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdNextSubpass2Delegate(IntPtr cmdBuffer, IntPtr pSubpassBeginInfo, IntPtr pSubpassEndInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndRenderPass2Delegate(IntPtr cmdBuffer, IntPtr pSubpassEndInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetDepthTestEnableDelegate(IntPtr cmdBuffer, uint depthTestEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetDepthWriteEnableDelegate(IntPtr cmdBuffer, uint depthWriteEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetDepthCompareOpDelegate(IntPtr cmdBuffer, CompareOp depthCompareOp);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetDepthBoundsTestEnableDelegate(IntPtr cmdBuffer, uint depthBoundsTestEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetStencilTestEnableDelegate(IntPtr cmdBuffer, uint stencilTestEnable);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetStencilOpDelegate(IntPtr cmdBuffer, uint faceMask, StencilOp failOp, StencilOp passOp, StencilOp depthFailOp, CompareOp compareOp);

        private GetDeviceQueueDelegate _vkGetDeviceQueue;
        private DeviceWaitIdleDelegate _vkDeviceWaitIdle;
        private CreateSwapchainKHRDelegate _vkCreateSwapchainKHR;
        private DestroySwapchainKHRDelegate _vkDestroySwapchainKHR;
        private AcquireNextImageKHRDelegate _vkAcquireNextImageKHR;
        private QueuePresentKHRDelegate _vkQueuePresentKHR;
        private QueueSubmitDelegate _vkQueueSubmit;
        private QueueWaitIdleDelegate _vkQueueWaitIdle;
        private CreateFenceDelegate _vkCreateFence;
        private DestroyFenceDelegate _vkDestroyFence;
        private WaitForFencesDelegate _vkWaitForFences;
        private ResetFencesDelegate _vkResetFences;
        private CreateSemaphoreDelegate _vkCreateSemaphore;
        private DestroySemaphoreDelegate _vkDestroySemaphore;
        private CreateCommandPoolDelegate _vkCreateCommandPool;
        private DestroyCommandPoolDelegate _vkDestroyCommandPool;
        private AllocateCommandBuffersDelegate _vkAllocateCommandBuffers;
        private FreeCommandBuffersDelegate _vkFreeCommandBuffers;
        private BeginCommandBufferDelegate _vkBeginCommandBuffer;
        private EndCommandBufferDelegate _vkEndCommandBuffer;
        private CmdBeginRenderPassDelegate _vkCmdBeginRenderPass;
        private CmdEndRenderPassDelegate _vkCmdEndRenderPass;
        private CmdBindPipelineDelegate _vkCmdBindPipeline;
        private CmdDrawDelegate _vkCmdDraw;
        private CmdDrawIndexedDelegate _vkCmdDrawIndexed;
        private CmdDispatchDelegate _vkCmdDispatch;
        private CmdPipelineBarrierDelegate _vkCmdPipelineBarrier;
        private CmdCopyBufferDelegate _vkCmdCopyBuffer;
        private CmdCopyBufferToImageDelegate _vkCmdCopyBufferToImage;
        private CmdPushConstantsDelegate _vkCmdPushConstants;
        private CmdSetViewportDelegate _vkCmdSetViewport;
        private CmdSetScissorDelegate _vkCmdSetScissor;
        private CmdBindVertexBuffersDelegate _vkCmdBindVertexBuffers;
        private CmdBindIndexBufferDelegate _vkCmdBindIndexBuffer;
        private CreateBufferDelegate _vkCreateBuffer;
        private DestroyBufferDelegate _vkDestroyBuffer;
        private GetBufferMemoryRequirementsDelegate _vkGetBufferMemoryRequirements;
        private BindBufferMemoryDelegate _vkBindBufferMemory;
        private AllocateMemoryDelegate _vkAllocateMemory;
        private FreeMemoryDelegate _vkFreeMemory;
        private MapMemoryDelegate _vkMapMemory;
        private UnmapMemoryDelegate _vkUnmapMemory;
        private FlushMappedMemoryRangesDelegate _vkFlushMappedMemoryRanges;
        private CreateImageDelegate _vkCreateImage;
        private DestroyImageDelegate _vkDestroyImage;
        private GetImageMemoryRequirementsDelegate _vkGetImageMemoryRequirements;
        private BindImageMemoryDelegate _vkBindImageMemory;
        private CreateImageViewDelegate _vkCreateImageView;
        private DestroyImageViewDelegate _vkDestroyImageView;
        private CreateSamplerDelegate _vkCreateSampler;
        private DestroySamplerDelegate _vkDestroySampler;
        private CreateRenderPassDelegate _vkCreateRenderPass;
        private DestroyRenderPassDelegate _vkDestroyRenderPass;
        private CreateFramebufferDelegate _vkCreateFramebuffer;
        private DestroyFramebufferDelegate _vkDestroyFramebuffer;
        private CreateShaderModuleDelegate _vkCreateShaderModule;
        private DestroyShaderModuleDelegate _vkDestroyShaderModule;
        private CreateGraphicsPipelinesDelegate _vkCreateGraphicsPipelines;
        private CreateComputePipelinesDelegate _vkCreateComputePipelines;
        private DestroyPipelineDelegate _vkDestroyPipeline;
        private CreatePipelineLayoutDelegate _vkCreatePipelineLayout;
        private DestroyPipelineLayoutDelegate _vkDestroyPipelineLayout;
        private CreateDescriptorSetLayoutDelegate _vkCreateDescriptorSetLayout;
        private DestroyDescriptorSetLayoutDelegate _vkDestroyDescriptorSetLayout;
        private CreateDescriptorPoolDelegate _vkCreateDescriptorPool;
        private DestroyDescriptorPoolDelegate _vkDestroyDescriptorPool;
        private AllocateDescriptorSetsDelegate _vkAllocateDescriptorSets;
        private FreeDescriptorSetsDelegate _vkFreeDescriptorSets;
        private UpdateDescriptorSetsDelegate _vkUpdateDescriptorSets;
        private CreatePipelineCacheDelegate _vkCreatePipelineCache;
        private DestroyPipelineCacheDelegate _vkDestroyPipelineCache;
        private GetPipelineCacheDataDelegate _vkGetPipelineCacheData;
        private MergePipelineCachesDelegate _vkMergePipelineCaches;
        private CmdCopyImageDelegate _vkCmdCopyImage;
        private CmdBlitImageDelegate _vkCmdBlitImage;
        private CmdResolveImageDelegate _vkCmdResolveImage;
        private CreateQueryPoolDelegate _vkCreateQueryPool;
        private DestroyQueryPoolDelegate _vkDestroyQueryPool;
        private ResetQueryPoolDelegate _vkResetQueryPool;
        private GetQueryPoolResultsDelegate _vkGetQueryPoolResults;
        private CmdWriteTimestampDelegate _vkCmdWriteTimestamp;
        private CmdResetQueryPoolDelegate _vkCmdResetQueryPool;
        private CmdCopyQueryPoolResultsDelegate _vkCmdCopyQueryPoolResults;
        private CmdBeginQueryDelegate _vkCmdBeginQuery;
        private CmdEndQueryDelegate _vkCmdEndQuery;
        private CmdBeginRenderingDelegate _vkCmdBeginRendering;
        private CmdEndRenderingDelegate _vkCmdEndRendering;
        private CmdSetDepthBiasEnableDelegate _vkCmdSetDepthBiasEnable;
        private CmdSetPrimitiveRestartEnableDelegate _vkCmdSetPrimitiveRestartEnable;
        private CmdSetRasterizerDiscardEnableDelegate _vkCmdSetRasterizerDiscardEnable;
        private CreateRenderPass2Delegate _vkCreateRenderPass2;
        private CmdBeginRenderPass2Delegate _vkCmdBeginRenderPass2;
        private CmdNextSubpass2Delegate _vkCmdNextSubpass2;
        private CmdEndRenderPass2Delegate _vkCmdEndRenderPass2;
        private CmdSetDepthTestEnableDelegate _vkCmdSetDepthTestEnable;
        private CmdSetDepthWriteEnableDelegate _vkCmdSetDepthWriteEnable;
        private CmdSetDepthCompareOpDelegate _vkCmdSetDepthCompareOp;
        private CmdSetDepthBoundsTestEnableDelegate _vkCmdSetDepthBoundsTestEnable;
        private CmdSetStencilTestEnableDelegate _vkCmdSetStencilTestEnable;
        private CmdSetStencilOpDelegate _vkCmdSetStencilOp;

        // VK constants
        private const uint VK_STRUCTURE_TYPE_APPLICATION_INFO = 1;
        private const uint VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
        private const uint VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 3;
        private const uint VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
        private const uint VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR = 1000001000;
        private const uint VK_STRUCTURE_TYPE_PRESENT_INFO_KHR = 1000001001;
        private const uint VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
        private const uint VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
        private const uint VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 7;
        private const uint VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
        private const uint VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
        private const uint VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 41;
        private const uint VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO = 42;
        private const uint VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
        private const uint VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
        private const uint VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE = 6;
        private const uint VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO = 18;
        private const uint VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = 15;
        private const uint VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO = 10;
        private const uint VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO = 37;
        private const uint VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO = 38;
        private const uint VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
        private const uint VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO = 28;
        private const uint VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
        private const uint VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
        private const uint VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
        private const uint VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
        private const uint VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO = 17;
        private const uint VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = 36;
        private const uint VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2 = 1000299000;
        private const uint VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO = 1000106001;
        private const uint VK_STRUCTURE_TYPE_SUBPASS_END_INFO = 1000106002;
        private const uint VK_STRUCTURE_TYPE_RENDERING_INFO = 1000245000;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO = 1000127000;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO = 1000127001;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO = 1000127002;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO = 1000127003;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO = 1000127004;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO = 1000127005;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO = 1000127007;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO = 1000127008;
        private const uint VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO = 1000127009;
        private const ulong VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF;

        [StructLayout(LayoutKind.Sequential)]
        private struct VkApplicationInfo
        {
            public uint sType; public IntPtr pNext; public IntPtr pApplicationName;
            public uint applicationVersion; public IntPtr pEngineName;
            public uint engineVersion; public uint apiVersion;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkDebugUtilsMessengerCreateInfoEXT
        {
            public uint sType; public IntPtr pNext; public uint flags;
            public uint messageSeverity; public uint messageType; public IntPtr pfnUserCallback; public IntPtr pUserData;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkDebugUtilsMessengerCallbackDataEXT
        {
            public uint sType; public IntPtr pNext; public IntPtr pMessageIdNumber; public IntPtr pMessage; public uint queueLabelCount;
            public IntPtr pQueueLabels; public uint cmdBufLabelCount; public IntPtr pCmdBufLabels; public uint objectCount; public IntPtr pObjects;
        }

        /// <summary>Creates the Vulkan debug messenger</summary>
        private void CreateDebugMessenger()
        {
            if (!_creationInfo.EnableValidation) return;
            try
            {
                _vkCreateDebugUtilsMessenger = GetInstanceProc<VkCreateDebugUtilsMessengerDelegate>("vkCreateDebugUtilsMessengerEXT");
                _vkDestroyDebugUtilsMessenger = GetInstanceProc<VkDestroyDebugUtilsMessengerDelegate>("vkDestroyDebugUtilsMessengerEXT");

                var callback = new VkDebugUtilsMessengerCallbackDelegate(DebugUtilsCallback);
                var createInfo = new VkDebugUtilsMessengerCreateInfoEXT
                {
                    sType = 1000028001,
                    messageSeverity = 0x00000001 | 0x00000100 | 0x00001000,
                    messageType = 0x00000001 | 0x00000002 | 0x00000004,
                    pfnUserCallback = Marshal.GetFunctionPointerForDelegate(callback)
                };

                _vkCreateDebugUtilsMessenger(_instance, ref createInfo, IntPtr.Zero, ref _debugUtilsMessenger);
            }
            catch { }
        }

        private VkBool32 DebugUtilsCallback(uint messageSeverity, uint messageTypes, IntPtr pCallbackData, IntPtr pUserData)
        {
            if (pCallbackData != IntPtr.Zero)
            {
                var callbackData = Marshal.PtrToStructure<VkDebugUtilsMessengerCallbackDataEXT>(pCallbackData);
                if (callbackData.pMessage != IntPtr.Zero)
                {
                    var message = Marshal.PtrToStringAnsi(callbackData.pMessage);
                    if ((messageSeverity & 0x00001000) != 0)
                        Debug.WriteLine($"[Vulkan ERROR] {message}");
                    else if ((messageSeverity & 0x00000100) != 0)
                        Debug.WriteLine($"[Vulkan WARNING] {message}");
                    else
                        Debug.WriteLine($"[Vulkan INFO] {message}");
                }
            }
            return VkBool32.False;
        }

        private void LoadDeviceFunctions()
        {
            _vkGetDeviceQueue = GetDeviceProc<GetDeviceQueueDelegate>("vkGetDeviceQueue");
            _vkDeviceWaitIdle = GetDeviceProc<DeviceWaitIdleDelegate>("vkDeviceWaitIdle");
            _vkCreateSwapchainKHR = GetDeviceProc<CreateSwapchainKHRDelegate>("vkCreateSwapchainKHR");
            _vkDestroySwapchainKHR = GetDeviceProc<DestroySwapchainKHRDelegate>("vkDestroySwapchainKHR");
            _vkAcquireNextImageKHR = GetDeviceProc<AcquireNextImageKHRDelegate>("vkAcquireNextImageKHR");
            _vkQueuePresentKHR = GetDeviceProc<QueuePresentKHRDelegate>("vkQueuePresentKHR");
            _vkQueueSubmit = GetDeviceProc<QueueSubmitDelegate>("vkQueueSubmit");
            _vkQueueWaitIdle = GetDeviceProc<QueueWaitIdleDelegate>("vkQueueWaitIdle");
            _vkCreateFence = GetDeviceProc<CreateFenceDelegate>("vkCreateFence");
            _vkDestroyFence = GetDeviceProc<DestroyFenceDelegate>("vkDestroyFence");
            _vkWaitForFences = GetDeviceProc<WaitForFencesDelegate>("vkWaitForFences");
            _vkResetFences = GetDeviceProc<ResetFencesDelegate>("vkResetFences");
            _vkCreateSemaphore = GetDeviceProc<CreateSemaphoreDelegate>("vkCreateSemaphore");
            _vkDestroySemaphore = GetDeviceProc<DestroySemaphoreDelegate>("vkDestroySemaphore");
            _vkCreateCommandPool = GetDeviceProc<CreateCommandPoolDelegate>("vkCreateCommandPool");
            _vkDestroyCommandPool = GetDeviceProc<DestroyCommandPoolDelegate>("vkDestroyCommandPool");
            _vkAllocateCommandBuffers = GetDeviceProc<AllocateCommandBuffersDelegate>("vkAllocateCommandBuffers");
            _vkFreeCommandBuffers = GetDeviceProc<FreeCommandBuffersDelegate>("vkFreeCommandBuffers");
            _vkBeginCommandBuffer = GetDeviceProc<BeginCommandBufferDelegate>("vkBeginCommandBuffer");
            _vkEndCommandBuffer = GetDeviceProc<EndCommandBufferDelegate>("vkEndCommandBuffer");
            _vkCmdBeginRenderPass = GetDeviceProc<CmdBeginRenderPassDelegate>("vkCmdBeginRenderPass");
            _vkCmdEndRenderPass = GetDeviceProc<CmdEndRenderPassDelegate>("vkCmdEndRenderPass");
            _vkCmdBindPipeline = GetDeviceProc<CmdBindPipelineDelegate>("vkCmdBindPipeline");
            _vkCmdDraw = GetDeviceProc<CmdDrawDelegate>("vkCmdDraw");
            _vkCmdDrawIndexed = GetDeviceProc<CmdDrawIndexedDelegate>("vkCmdDrawIndexed");
            _vkCmdDispatch = GetDeviceProc<CmdDispatchDelegate>("vkCmdDispatch");
            _vkCmdPipelineBarrier = GetDeviceProc<CmdPipelineBarrierDelegate>("vkCmdPipelineBarrier");
            _vkCmdCopyBuffer = GetDeviceProc<CmdCopyBufferDelegate>("vkCmdCopyBuffer");
            _vkCmdCopyBufferToImage = GetDeviceProc<CmdCopyBufferToImageDelegate>("vkCmdCopyBufferToImage");
            _vkCmdPushConstants = GetDeviceProc<CmdPushConstantsDelegate>("vkCmdPushConstants");
            _vkCmdSetViewport = GetDeviceProc<CmdSetViewportDelegate>("vkCmdSetViewport");
            _vkCmdSetScissor = GetDeviceProc<CmdSetScissorDelegate>("vkCmdSetScissor");
            _vkCmdBindVertexBuffers = GetDeviceProc<CmdBindVertexBuffersDelegate>("vkCmdBindVertexBuffers");
            _vkCmdBindIndexBuffer = GetDeviceProc<CmdBindIndexBufferDelegate>("vkCmdBindIndexBuffer");
            _vkCreateBuffer = GetDeviceProc<CreateBufferDelegate>("vkCreateBuffer");
            _vkDestroyBuffer = GetDeviceProc<DestroyBufferDelegate>("vkDestroyBuffer");
            _vkGetBufferMemoryRequirements = GetDeviceProc<GetBufferMemoryRequirementsDelegate>("vkGetBufferMemoryRequirements");
            _vkBindBufferMemory = GetDeviceProc<BindBufferMemoryDelegate>("vkBindBufferMemory");
            _vkAllocateMemory = GetDeviceProc<AllocateMemoryDelegate>("vkAllocateMemory");
            _vkFreeMemory = GetDeviceProc<FreeMemoryDelegate>("vkFreeMemory");
            _vkMapMemory = GetDeviceProc<MapMemoryDelegate>("vkMapMemory");
            _vkUnmapMemory = GetDeviceProc<UnmapMemoryDelegate>("vkUnmapMemory");
            _vkFlushMappedMemoryRanges = GetDeviceProc<FlushMappedMemoryRangesDelegate>("vkFlushMappedMemoryRanges");
            _vkCreateImage = GetDeviceProc<CreateImageDelegate>("vkCreateImage");
            _vkDestroyImage = GetDeviceProc<DestroyImageDelegate>("vkDestroyImage");
            _vkGetImageMemoryRequirements = GetDeviceProc<GetImageMemoryRequirementsDelegate>("vkGetImageMemoryRequirements");
            _vkBindImageMemory = GetDeviceProc<BindImageMemoryDelegate>("vkBindImageMemory");
            _vkCreateImageView = GetDeviceProc<CreateImageViewDelegate>("vkCreateImageView");
            _vkDestroyImageView = GetDeviceProc<DestroyImageViewDelegate>("vkDestroyImageView");
            _vkCreateSampler = GetDeviceProc<CreateSamplerDelegate>("vkCreateSampler");
            _vkDestroySampler = GetDeviceProc<DestroySamplerDelegate>("vkDestroySampler");
            _vkCreateRenderPass = GetDeviceProc<CreateRenderPassDelegate>("vkCreateRenderPass");
            _vkDestroyRenderPass = GetDeviceProc<DestroyRenderPassDelegate>("vkDestroyRenderPass");
            _vkCreateFramebuffer = GetDeviceProc<CreateFramebufferDelegate>("vkCreateFramebuffer");
            _vkDestroyFramebuffer = GetDeviceProc<DestroyFramebufferDelegate>("vkDestroyFramebuffer");
            _vkCreateShaderModule = GetDeviceProc<CreateShaderModuleDelegate>("vkCreateShaderModule");
            _vkDestroyShaderModule = GetDeviceProc<DestroyShaderModuleDelegate>("vkDestroyShaderModule");
            _vkCreateGraphicsPipelines = GetDeviceProc<CreateGraphicsPipelinesDelegate>("vkCreateGraphicsPipelines");
            _vkCreateComputePipelines = GetDeviceProc<CreateComputePipelinesDelegate>("vkCreateComputePipelines");
            _vkDestroyPipeline = GetDeviceProc<DestroyPipelineDelegate>("vkDestroyPipeline");
            _vkCreatePipelineLayout = GetDeviceProc<CreatePipelineLayoutDelegate>("vkCreatePipelineLayout");
            _vkDestroyPipelineLayout = GetDeviceProc<DestroyPipelineLayoutDelegate>("vkDestroyPipelineLayout");
            _vkCreateDescriptorSetLayout = GetDeviceProc<CreateDescriptorSetLayoutDelegate>("vkCreateDescriptorSetLayout");
            _vkDestroyDescriptorSetLayout = GetDeviceProc<DestroyDescriptorSetLayoutDelegate>("vkDestroyDescriptorSetLayout");
            _vkCreateDescriptorPool = GetDeviceProc<CreateDescriptorPoolDelegate>("vkCreateDescriptorPool");
            _vkDestroyDescriptorPool = GetDeviceProc<DestroyDescriptorPoolDelegate>("vkDestroyDescriptorPool");
            _vkAllocateDescriptorSets = GetDeviceProc<AllocateDescriptorSetsDelegate>("vkAllocateDescriptorSets");
            _vkFreeDescriptorSets = GetDeviceProc<FreeDescriptorSetsDelegate>("vkFreeDescriptorSets");
            _vkUpdateDescriptorSets = GetDeviceProc<UpdateDescriptorSetsDelegate>("vkUpdateDescriptorSets");
            _vkCreatePipelineCache = GetDeviceProc<CreatePipelineCacheDelegate>("vkCreatePipelineCache");
            _vkDestroyPipelineCache = GetDeviceProc<DestroyPipelineCacheDelegate>("vkDestroyPipelineCache");
            _vkGetPipelineCacheData = GetDeviceProc<GetPipelineCacheDataDelegate>("vkGetPipelineCacheData");
            _vkMergePipelineCaches = GetDeviceProc<MergePipelineCachesDelegate>("vkMergePipelineCaches");
            _vkCmdCopyImage = GetDeviceProc<CmdCopyImageDelegate>("vkCmdCopyImage");
            _vkCmdBlitImage = GetDeviceProc<CmdBlitImageDelegate>("vkCmdBlitImage");
            _vkCmdResolveImage = GetDeviceProc<CmdResolveImageDelegate>("vkCmdResolveImage");
            _vkCreateQueryPool = GetDeviceProc<CreateQueryPoolDelegate>("vkCreateQueryPool");
            _vkDestroyQueryPool = GetDeviceProc<DestroyQueryPoolDelegate>("vkDestroyQueryPool");
            _vkResetQueryPool = GetDeviceProc<ResetQueryPoolDelegate>("vkResetQueryPool");
            _vkGetQueryPoolResults = GetDeviceProc<GetQueryPoolResultsDelegate>("vkGetQueryPoolResults");
            _vkCmdWriteTimestamp = GetDeviceProc<CmdWriteTimestampDelegate>("vkCmdWriteTimestamp");
            _vkCmdResetQueryPool = GetDeviceProc<CmdResetQueryPoolDelegate>("vkCmdResetQueryPool");
            _vkCmdCopyQueryPoolResults = GetDeviceProc<CmdCopyQueryPoolResultsDelegate>("vkCmdCopyQueryPoolResults");
            _vkCmdBeginQuery = GetDeviceProc<CmdBeginQueryDelegate>("vkCmdBeginQuery");
            _vkCmdEndQuery = GetDeviceProc<CmdEndQueryDelegate>("vkCmdEndQuery");
            _vkCmdBeginRendering = GetDeviceProc<CmdBeginRenderingDelegate>("vkCmdBeginRendering");
            _vkCmdEndRendering = GetDeviceProc<CmdEndRenderingDelegate>("vkCmdEndRendering");
            _vkCmdSetDepthBiasEnable = GetDeviceProc<CmdSetDepthBiasEnableDelegate>("vkCmdSetDepthBiasEnable");
            _vkCmdSetPrimitiveRestartEnable = GetDeviceProc<CmdSetPrimitiveRestartEnableDelegate>("vkCmdSetPrimitiveRestartEnable");
            _vkCmdSetRasterizerDiscardEnable = GetDeviceProc<CmdSetRasterizerDiscardEnableDelegate>("vkCmdSetRasterizerDiscardEnable");
            _vkCreateRenderPass2 = GetDeviceProc<CreateRenderPass2Delegate>("vkCreateRenderPass2");
            _vkCmdBeginRenderPass2 = GetDeviceProc<CmdBeginRenderPass2Delegate>("vkCmdBeginRenderPass2");
            _vkCmdNextSubpass2 = GetDeviceProc<CmdNextSubpass2Delegate>("vkCmdNextSubpass2");
            _vkCmdEndRenderPass2 = GetDeviceProc<CmdEndRenderPass2Delegate>("vkCmdEndRenderPass2");
            _vkCmdSetDepthTestEnable = GetDeviceProc<CmdSetDepthTestEnableDelegate>("vkCmdSetDepthTestEnable");
            _vkCmdSetDepthWriteEnable = GetDeviceProc<CmdSetDepthWriteEnableDelegate>("vkCmdSetDepthWriteEnable");
            _vkCmdSetDepthCompareOp = GetDeviceProc<CmdSetDepthCompareOpDelegate>("vkCmdSetDepthCompareOp");
            _vkCmdSetDepthBoundsTestEnable = GetDeviceProc<CmdSetDepthBoundsTestEnableDelegate>("vkCmdSetDepthBoundsTestEnable");
            _vkCmdSetStencilTestEnable = GetDeviceProc<CmdSetStencilTestEnableDelegate>("vkCmdSetStencilTestEnable");
            _vkCmdSetStencilOp = GetDeviceProc<CmdSetStencilOpDelegate>("vkCmdSetStencilOp");
        }
        private void PickPhysicalDevice()
        {
            uint deviceCount = 0;
            vkEnumeratePhysicalDevices(_instance, ref deviceCount, IntPtr.Zero);
            if (deviceCount == 0) throw new InvalidOperationException("No Vulkan physical devices found");

            var devicesPtr = Marshal.AllocHGlobal((int)(deviceCount * IntPtr.Size));
            try
            {
                vkEnumeratePhysicalDevices(_instance, ref deviceCount, devicesPtr);
                _physicalDevice = Marshal.ReadIntPtr(devicesPtr);
            }
            finally { Marshal.FreeHGlobal(devicesPtr); }
            _device.PhysicalDevice = _physicalDevice;

            // Get device properties
            var properties = new VkPhysicalDeviceProperties();
            var propertiesPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPhysicalDeviceProperties>());
            try
            {
                vkGetPhysicalDeviceProperties(_physicalDevice, propertiesPtr);
                properties = Marshal.PtrToStructure<VkPhysicalDeviceProperties>(propertiesPtr);
                _physicalDeviceInfo.Handle = _physicalDevice;
                _physicalDeviceInfo.DeviceName = Marshal.PtrToStringAnsi((IntPtr)((long)propertiesPtr + 16)) ?? "Unknown";
                _physicalDeviceInfo.VendorId = properties.vendorID;
                _physicalDeviceInfo.DeviceId = properties.deviceID;
                _physicalDeviceInfo.ApiVersion = properties.apiVersion;
                _physicalDeviceInfo.DriverVersion = properties.driverVersion;
                _device.PhysicalDeviceInfo = _physicalDeviceInfo;
            }
            finally { Marshal.FreeHGlobal(propertiesPtr); }

            // Get queue family properties
            uint queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref queueFamilyCount, IntPtr.Zero);
            var queueFamilyProperties = new VkQueueFamilyProperties[queueFamilyCount];
            var qfpPtr = Marshal.AllocHGlobal((int)(queueFamilyCount * Marshal.SizeOf<VkQueueFamilyProperties>()));
            try
            {
                vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref queueFamilyCount, qfpPtr);
                for (int i = 0; i < queueFamilyCount; i++)
                    queueFamilyProperties[i] = Marshal.PtrToStructure<VkQueueFamilyProperties>(qfpPtr + i * Marshal.SizeOf<VkQueueFamilyProperties>());
            }
            finally { Marshal.FreeHGlobal(qfpPtr); }

            _queueFamilyIndices = new QueueFamilyIndices();
            for (uint i = 0; i < queueFamilyCount; i++)
            {
                var props = queueFamilyProperties[i];
                if ((props.queueFlags & QueueFlag.Graphics) != 0) _queueFamilyIndices.GraphicsFamily = (int)i;
                if ((props.queueFlags & QueueFlag.Compute) != 0 && _queueFamilyIndices.ComputeFamily < 0) _queueFamilyIndices.ComputeFamily = (int)i;
                if ((props.queueFlags & QueueFlag.Transfer) != 0 && _queueFamilyIndices.TransferFamily < 0) _queueFamilyIndices.TransferFamily = (int)i;

                if (_surface != IntPtr.Zero)
                {
                    IntPtr supported = IntPtr.Zero;
                    vkGetPhysicalDeviceSurfaceSupport(_physicalDevice, i, _surface, ref supported);
                    if (supported != IntPtr.Zero) _queueFamilyIndices.PresentFamily = (int)i;
                }
            }
            _device.QueueIndices = _queueFamilyIndices;

            // Get memory properties
            var memProps = new VkPhysicalDeviceMemoryProperties();
            var memPropsPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPhysicalDeviceMemoryProperties>());
            try
            {
                vkGetPhysicalDeviceMemoryProperties(_physicalDevice, memPropsPtr);
                memProps = Marshal.PtrToStructure<VkPhysicalDeviceMemoryProperties>(memPropsPtr);
                _physicalDeviceInfo.MemoryProperties = new PhysicalDeviceMemoryProperties
                {
                    MemoryTypeCount = memProps.memoryTypeCount,
                    MemoryTypes = new MemoryType[memProps.memoryTypeCount],
                    MemoryHeapCount = memProps.memoryHeapCount,
                    MemoryHeaps = new MemoryHeap[memProps.memoryHeapCount]
                };
                for (int i = 0; i < memProps.memoryTypeCount; i++)
                {
                    var mt = Marshal.PtrToStructure<VkMemoryType>(memPropsPtr + 4 + i * Marshal.SizeOf<VkMemoryType>());
                    _physicalDeviceInfo.MemoryProperties.MemoryTypes[i] = new MemoryType { PropertyFlags = mt.propertyFlags, HeapIndex = mt.heapIndex };
                }
                for (int i = 0; i < memProps.memoryHeapCount; i++)
                {
                    var mh = Marshal.PtrToStructure<VkMemoryHeap>(memPropsPtr + 4 + (int)(memProps.memoryTypeCount * Marshal.SizeOf<VkMemoryType>()) + i * Marshal.SizeOf<VkMemoryHeap>());
                    _physicalDeviceInfo.MemoryProperties.MemoryHeaps[i] = new MemoryHeap { Size = mh.size, Flags = mh.flags };
                }
            }
            finally { Marshal.FreeHGlobal(memPropsPtr); }
        }

        private void CreateLogicalDevice()
        {
            var queueCreateInfos = new List<VkDeviceQueueCreateInfo>();
            var uniqueFamilies = _queueFamilyIndices.UniqueQueueFamilies;
            float queuePriority = 1.0f;
            var priorityPtr = Marshal.AllocHGlobal(sizeof(float));
            MarshalCompat.WriteFloat(priorityPtr, queuePriority);

            foreach (var family in uniqueFamilies)
            {
                var qci = new VkDeviceQueueCreateInfo
                {
                    sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex = family,
                    queueCount = 1,
                    pQueuePriorities = priorityPtr
                };
                queueCreateInfos.Add(qci);
            }

            var deviceFeatures = new VkPhysicalDeviceFeatures();
            deviceFeatures.samplerAnisotropy = VulkanBool32.True;
            deviceFeatures.fillModeNonSolid = VulkanBool32.True;
            deviceFeatures.wideLines = VulkanBool32.True;
            deviceFeatures.largePoints = VulkanBool32.True;
            deviceFeatures.shaderImageGatherExtended = VulkanBool32.True;
            deviceFeatures.samplerLodClamp = VulkanBool32.True;

            var featuresPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPhysicalDeviceFeatures>());
            Marshal.StructureToPtr(deviceFeatures, featuresPtr, false);

            var extensions = new List<string>(_creationInfo.RequiredExtensions);
            extensions.Add("VK_KHR_swapchain");
            extensions.Add("VK_KHR_maintenance1");
            extensions.Add("VK_KHR_maintenance2");
            extensions.Add("VK_KHR_maintenance3");
            extensions.Add("VK_KHR_maintenance4");
            extensions.Add("VK_KHR_synchronization2");
            extensions.Add("VK_KHR_dynamic_rendering");
            extensions.Add("VK_KHR_depth_stencil_resolve");
            extensions.Add("VK_KHR_create_renderpass2");
            extensions.Add("VK_EXT_descriptor_indexing");
            extensions.Add("VK_EXT_subgroup_size_control");
            extensions.Add("VK_KHR_8bit_storage");
            extensions.Add("VK_KHR_16bit_storage");
            extensions.Add("VK_KHR_shader_float16_int8");

            var extensionPtrs = extensions.Select(e => Marshal.StringToHGlobalAnsi(e)).ToArray();
            var layerNames = new List<string>();
            if (_creationInfo.EnableValidation) layerNames.Add("VK_LAYER_KHRONOS_validation");
            var layerPtrs = layerNames.Select(l => Marshal.StringToHGlobalAnsi(l)).ToArray();

            var createInfo = new VkDeviceCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                queueCreateInfoCount = (uint)queueCreateInfos.Count,
                enabledExtensionCount = (uint)extensionPtrs.Length,
                enabledLayerCount = (uint)layerPtrs.Length,
                pEnabledFeatures = featuresPtr
            };

            unsafe
            {
                var queueInfoArray = queueCreateInfos.ToArray();
                createInfo.pQueueCreateInfos = Marshal.AllocHGlobal(queueInfoArray.Length * Marshal.SizeOf<VkDeviceQueueCreateInfo>());
                for (int i = 0; i < queueInfoArray.Length; i++)
                    Marshal.StructureToPtr(queueInfoArray[i], createInfo.pQueueCreateInfos + i * Marshal.SizeOf<VkDeviceQueueCreateInfo>(), false);

                createInfo.ppEnabledExtensionNames = Marshal.AllocHGlobal(extensionPtrs.Length * IntPtr.Size);
                createInfo.ppEnabledLayerNames = Marshal.AllocHGlobal(layerPtrs.Length * IntPtr.Size);
                for (int i = 0; i < extensionPtrs.Length; i++)
                    Marshal.WriteIntPtr(createInfo.ppEnabledExtensionNames + i * IntPtr.Size, extensionPtrs[i]);
                for (int i = 0; i < layerPtrs.Length; i++)
                    Marshal.WriteIntPtr(createInfo.ppEnabledLayerNames + i * IntPtr.Size, layerPtrs[i]);
            }

            var result = vkCreateDevice(_physicalDevice, ref createInfo, IntPtr.Zero, ref _logicalDevice);
            Marshal.FreeHGlobal(featuresPtr);
            Marshal.FreeHGlobal(priorityPtr);
            foreach (var ptr in extensionPtrs) Marshal.FreeHGlobal(ptr);
            foreach (var ptr in layerPtrs) Marshal.FreeHGlobal(ptr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create Vulkan logical device: {result}");

            _device.LogicalDevice = _logicalDevice;

            _vkGetDeviceProcAddr = _device.GetDeviceProcAddr("vkGetDeviceProcAddr");
            LoadDeviceFunctions();
        }

        private void InitializeQueues()
        {
            _vkGetDeviceQueue = GetDeviceProc<GetDeviceQueueDelegate>("vkGetDeviceQueue");

            if (_queueFamilyIndices.GraphicsFamily >= 0)
            {
                _vkGetDeviceQueue(_logicalDevice, (uint)_queueFamilyIndices.GraphicsFamily, 0, ref _graphicsQueue);
                _device.GraphicsQueue = _graphicsQueue;
            }
            if (_queueFamilyIndices.ComputeFamily >= 0)
            {
                _vkGetDeviceQueue(_logicalDevice, (uint)_queueFamilyIndices.ComputeFamily, 0, ref _computeQueue);
                _device.ComputeQueue = _computeQueue;
            }
            if (_queueFamilyIndices.TransferFamily >= 0)
            {
                _vkGetDeviceQueue(_logicalDevice, (uint)_queueFamilyIndices.TransferFamily, 0, ref _transferQueue);
                _device.TransferQueue = _transferQueue;
            }
            if (_queueFamilyIndices.PresentFamily >= 0)
            {
                _vkGetDeviceQueue(_logicalDevice, (uint)_queueFamilyIndices.PresentFamily, 0, ref _presentQueue);
                _device.PresentQueue = _presentQueue;
            }
        }

        private void CreatePipelineCacheInternal()
        {
            var createInfo = new VkPipelineCacheCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
            };
            _vkCreatePipelineCache(_logicalDevice, ref createInfo, IntPtr.Zero, ref _pipelineCache);
            _device.PipelineCache = _pipelineCacheWrapper;
        }

        private void InitializeSubsystems()
        {
            _memoryAllocator = new VulkanMemoryAllocator(_device);
            _device.MemoryAllocator = _memoryAllocator;

            _syncManager = new VulkanSyncManager(_device);
            _device.SyncManager = _syncManager;

            _descriptorManager = new VulkanDescriptorManager(_device);
            _device.DescriptorManager = _descriptorManager;

            _pipelineCacheWrapper = new VulkanPipelineCache(_device, _pipelineCache);
            _device.PipelineCache = _pipelineCacheWrapper;

            _resourceTracker = new VulkanResourceTracker(_device);
            _device.ResourceTracker = _resourceTracker;

            // Create default command pool
            var poolCreateInfo = new VkCommandPoolCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                flags = (uint)CommandPoolCreateFlag.ResetCommandBuffer,
                queueFamilyIndex = (uint)_queueFamilyIndices.GraphicsFamily
            };
            _vkCreateCommandPool(_logicalDevice, ref poolCreateInfo, IntPtr.Zero, ref _commandPool);
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkPhysicalDeviceProperties
        {
            public uint deviceType; public uint vendorID; public uint deviceID;
            public uint apiVersion; public uint driverVersion;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkQueueFamilyProperties
        {
            public QueueFlag queueFlags; public uint queueCount;
            public uint timestampValidBits;
            public uint minImageTransferGranularityX; public uint minImageTransferGranularityY; public uint minImageTransferGranularityZ;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkPhysicalDeviceMemoryProperties
        {
            public uint memoryTypeCount;
            public VkMemoryType memoryType0; public VkMemoryType memoryType1; public VkMemoryType memoryType2;
            public VkMemoryType memoryType3; public VkMemoryType memoryType4; public VkMemoryType memoryType5;
            public VkMemoryType memoryType6; public VkMemoryType memoryType7; public VkMemoryType memoryType8;
            public VkMemoryType memoryType9; public VkMemoryType memoryType10; public VkMemoryType memoryType11;
            public VkMemoryType memoryType12; public VkMemoryType memoryType13; public VkMemoryType memoryType14;
            public VkMemoryType memoryType15; public VkMemoryType memoryType16; public VkMemoryType memoryType17;
            public VkMemoryType memoryType18; public VkMemoryType memoryType19; public VkMemoryType memoryType20;
            public VkMemoryType memoryType21; public VkMemoryType memoryType22; public VkMemoryType memoryType23;
            public VkMemoryType memoryType24; public VkMemoryType memoryType25; public VkMemoryType memoryType26;
            public VkMemoryType memoryType27; public VkMemoryType memoryType28; public VkMemoryType memoryType29;
            public VkMemoryType memoryType30; public VkMemoryType memoryType31;
            public uint memoryHeapCount;
            public VkMemoryHeap heap0; public VkMemoryHeap heap1; public VkMemoryHeap heap2;
            public VkMemoryHeap heap3; public VkMemoryHeap heap4; public VkMemoryHeap heap5;
            public VkMemoryHeap heap6; public VkMemoryHeap heap7; public VkMemoryHeap heap8;
            public VkMemoryHeap heap9; public VkMemoryHeap heap10; public VkMemoryHeap heap11;
            public VkMemoryHeap heap12; public VkMemoryHeap heap13; public VkMemoryHeap heap14;
            public VkMemoryHeap heap15;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkMemoryType { public MemoryPropertyFlag propertyFlags; public uint heapIndex; }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkMemoryHeap { public ulong size; public MemoryHeapFlag flags; }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkPhysicalDeviceFeatures
        {
            public uint sType; public VulkanBool32 samplerAnisotropy; public VulkanBool32 fillModeNonSolid;
            public VulkanBool32 wideLines; public VulkanBool32 largePoints;
            public VulkanBool32 shaderImageGatherExtended; public VulkanBool32 samplerLodClamp;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkDeviceQueueCreateInfo
        {
            public uint sType; public IntPtr pNext; public uint flags;
            public uint queueFamilyIndex; public uint queueCount; public IntPtr pQueuePriorities;
        }

        private VkBool32 DebugUtilsCallbackNative(uint messageSeverity, uint messageTypes, IntPtr pCallbackData, IntPtr pUserData) => DebugUtilsCallback(messageSeverity, messageTypes, pCallbackData, pUserData);
        // =====================================================================
        // PUBLIC PROPERTIES
        // =====================================================================
        public IntPtr PhysicalDevice => _physicalDevice;
        public IntPtr Device => _logicalDevice;
        public IntPtr Instance => _instance;
        public IntPtr Surface => _surface;
        public IntPtr Queue => _graphicsQueue;
        public VulkanSwapchain Swapchain => _swapchain;

        // =====================================================================
        // PUBLIC API METHODS
        // =====================================================================

        /// <summary>Creates a Vulkan swapchain for the given surface</summary>
        public VulkanSwapchain CreateSwapchain(IntPtr surface)
        {
            _surface = surface;
            _device.Surface = surface;

            _queueFamilyIndices = new QueueFamilyIndices();
            FindQueueFamiliesForSurface(surface);

            var supportDetails = QuerySwapchainSupport(surface);
            var surfaceFormat = supportDetails.ChooseSurfaceFormat();
            var presentMode = supportDetails.ChoosePresentMode(PresentMode.Mailbox);
            var extent = supportDetails.ChooseExtent(1280, 720);

            uint imageCount = supportDetails.Capabilities.MinImageCount + 1;
            if (supportDetails.Capabilities.MaxImageCount > 0 && imageCount > supportDetails.Capabilities.MaxImageCount)
                imageCount = supportDetails.Capabilities.MaxImageCount;

            var createInfo = new VkSwapchainCreateInfoKHR
            {
                sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                surface = surface,
                minImageCount = imageCount,
                imageFormat = surfaceFormat.Format,
                imageColorSpace = (uint)surfaceFormat.ColorSpace,
                imageExtent = new VkExtent2D { width = extent.Width, height = extent.Height },
                imageArrayLayers = 1,
                imageUsage = ImageUsageFlag.ColorAttachment | ImageUsageFlag.TransferDst,
                preTransform = supportDetails.Capabilities.CurrentTransform,
                compositeAlpha = CompositeAlphaFlag.Opaque,
                presentMode = presentMode,
                clipped = VulkanBool32.True,
                oldSwapchain = _swapchain?.Handle ?? IntPtr.Zero
            };

            if (_queueFamilyIndices.GraphicsFamily != _queueFamilyIndices.PresentFamily)
            {
                var queueFamilyIndices = new uint[] { (uint)_queueFamilyIndices.GraphicsFamily, (uint)_queueFamilyIndices.PresentFamily };
                createInfo.imageSharingMode = SharingMode.Concurrent;
                createInfo.queueFamilyIndexCount = 2;
                unsafe
                {
                    fixed (uint* pIndices = queueFamilyIndices)
                        createInfo.pQueueFamilyIndices = (IntPtr)pIndices;
                }
            }
            else
            {
                createInfo.imageSharingMode = SharingMode.Exclusive;
            }

            IntPtr swapchain = IntPtr.Zero;
            var result = _vkCreateSwapchainKHR(_logicalDevice, ref createInfo, IntPtr.Zero, ref swapchain);
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create swapchain: {result}");

            var vulkanSwapchain = new VulkanSwapchain(_device, swapchain, surfaceFormat.Format, presentMode, extent);
            _swapchain = vulkanSwapchain;
            return vulkanSwapchain;
        }

        private void FindQueueFamiliesForSurface(IntPtr surface)
        {
            uint queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref queueFamilyCount, IntPtr.Zero);
            var qfpPtr = Marshal.AllocHGlobal((int)(queueFamilyCount * 48));
            try
            {
                vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref queueFamilyCount, qfpPtr);
                for (uint i = 0; i < queueFamilyCount; i++)
                {
                    var qfp = Marshal.PtrToStructure<VkQueueFamilyProperties>(qfpPtr + (int)(i * 48));
                    if ((qfp.queueFlags & QueueFlag.Graphics) != 0) _queueFamilyIndices.GraphicsFamily = (int)i;
                    if ((qfp.queueFlags & QueueFlag.Compute) != 0 && _queueFamilyIndices.ComputeFamily < 0)
                        _queueFamilyIndices.ComputeFamily = (int)i;

                    IntPtr supported = new IntPtr(1);
                    vkGetPhysicalDeviceSurfaceSupport(_physicalDevice, i, surface, ref supported);
                    if (supported != IntPtr.Zero) _queueFamilyIndices.PresentFamily = (int)i;
                }
            }
            finally { Marshal.FreeHGlobal(qfpPtr); }
        }

        /// <summary>Queries swapchain support details for a surface</summary>
        public VulkanSwapchainSupportDetails QuerySwapchainSupport(IntPtr surface)
        {
            var details = new VulkanSwapchainSupportDetails();

            // Capabilities
            var capsPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkSurfaceCapabilitiesKHR>());
            try
            {
                vkGetPhysicalDeviceSurfaceCapabilities(_physicalDevice, surface, capsPtr);
                var caps = Marshal.PtrToStructure<VkSurfaceCapabilitiesKHR>(capsPtr);
                details.Capabilities = new SurfaceCapabilities
                {
                    MinImageCount = caps.minImageCount,
                    MaxImageCount = caps.maxImageCount,
                    CurrentExtent = new Extent2D(caps.currentImageWidth, caps.currentImageHeight),
                    MinImageExtent = new Extent2D(caps.minImageWidth, caps.minImageHeight),
                    MaxImageExtent = new Extent2D(caps.maxImageWidth, caps.maxImageHeight),
                    MaxImageArrayLayers = caps.maxImageArrayLayers,
                    SupportedTransforms = caps.supportedTransforms,
                    CurrentTransform = caps.currentTransform,
                    SupportedCompositeAlpha = caps.supportedCompositeAlpha,
                    SupportedUsageFlags = caps.supportedUsageFlags
                };
            }
            finally { Marshal.FreeHGlobal(capsPtr); }

            // Formats
            uint formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormats(_physicalDevice, surface, ref formatCount, IntPtr.Zero);
            if (formatCount > 0)
            {
                var formatPtr = Marshal.AllocHGlobal((int)(formatCount * 8));
                try
                {
                    vkGetPhysicalDeviceSurfaceFormats(_physicalDevice, surface, ref formatCount, formatPtr);
                    details.Formats = new SurfaceFormatKHR[formatCount];
                    for (int i = 0; i < formatCount; i++)
                    {
                        var fmt = Marshal.PtrToStructure<SurfaceFormatKHR>(formatPtr + i * 8);
                        details.Formats[i] = fmt;
                    }
                }
                finally { Marshal.FreeHGlobal(formatPtr); }
            }

            // Present modes
            uint presentModeCount = 0;
            vkGetPhysicalDeviceSurfacePresentModes(_physicalDevice, surface, ref presentModeCount, IntPtr.Zero);
            if (presentModeCount > 0)
            {
                var modePtr = Marshal.AllocHGlobal((int)(presentModeCount * 4));
                try
                {
                    vkGetPhysicalDeviceSurfacePresentModes(_physicalDevice, surface, ref presentModeCount, modePtr);
                    details.PresentModes = new PresentMode[presentModeCount];
                    for (int i = 0; i < presentModeCount; i++)
                        details.PresentModes[i] = (PresentMode)Marshal.ReadInt32(modePtr + i * 4);
                }
                finally { Marshal.FreeHGlobal(modePtr); }
            }

            return details;
        }

        /// <summary>Returns queue family indices</summary>
        public QueueFamilyIndices GetQueueFamilies() => _queueFamilyIndices;

        /// <summary>Creates a GPU buffer</summary>
        public VulkanBuffer CreateBuffer(BufferDescription description)
        {
            var createInfo = new VkBufferCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size = description.Size,
                usage = description.Usage,
                sharingMode = description.SharingMode
            };

            IntPtr buffer = IntPtr.Zero;
            var result = _vkCreateBuffer(_logicalDevice, ref createInfo, IntPtr.Zero, ref buffer);
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create buffer: {result}");

            _vkGetBufferMemoryRequirements(_logicalDevice, buffer, out var memReqs);

            var vulkanBuffer = new VulkanBuffer(_device, buffer, description, memReqs);
            _memoryAllocator.AllocateBuffer(vulkanBuffer, description.MemoryProperties);
            _buffers.Add(vulkanBuffer);
            _resourceTracker.TrackResource(vulkanBuffer);
            return vulkanBuffer;
        }

        /// <summary>Creates a 2D/3D texture</summary>
        public VulkanTexture CreateTexture(TextureDescription description)
        {
            var extent3D = new VkExtent3D { width = description.Width, height = description.Height, depth = description.Depth };

            var createInfo = new VkImageCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                flags = 0,
                imageType = description.Type,
                format = description.Format,
                extent = extent3D,
                mipLevels = description.MipLevels,
                arrayLayers = description.ArrayLayers,
                samples = description.Samples,
                tiling = description.Tiling,
                usage = description.Usage,
                sharingMode = description.SharingMode,
                initialLayout = description.InitialLayout
            };

            IntPtr image = IntPtr.Zero;
            var result = _vkCreateImage(_logicalDevice, ref createInfo, IntPtr.Zero, ref image);
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create image: {result}");

            _vkGetImageMemoryRequirements(_logicalDevice, image, out var memReqs);

            var vulkanTexture = new VulkanTexture(_device, image, description, memReqs);
            _memoryAllocator.AllocateImage(vulkanTexture, MemoryPropertyFlag.DeviceLocal);
            _textures.Add(vulkanTexture);
            _resourceTracker.TrackResource(vulkanTexture);
            return vulkanTexture;
        }

        /// <summary>Creates a render pass from description</summary>
        public VulkanRenderPass CreateRenderPass(RenderPassDescription description)
        {
            uint attachmentCount = description.Attachments != null ? (uint)description.Attachments.Length : 0;
            uint subpassCount = description.Subpasses != null ? (uint)description.Subpasses.Length : 0;
            uint dependencyCount = description.Dependencies != null ? (uint)description.Dependencies.Length : 0;

            IntPtr renderPass = IntPtr.Zero;
            var createInfo = new VkRenderPassCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                attachmentCount = attachmentCount,
                subpassCount = subpassCount,
                dependencyCount = dependencyCount
            };

            // Marshal attachment descriptions
            IntPtr attachmentPtr = IntPtr.Zero;
            if (attachmentCount > 0)
            {
                int structSize = Marshal.SizeOf<VkAttachmentDescription>();
                attachmentPtr = Marshal.AllocHGlobal((int)(attachmentCount * structSize));
                for (int i = 0; i < description.Attachments.Length; i++)
                {
                    var src = description.Attachments[i];
                    var vkAtt = new VkAttachmentDescription
                    {
                        format = src.Format,
                        samples = src.Samples,
                        loadOp = src.LoadOp,
                        storeOp = src.StoreOp,
                        stencilLoadOp = src.StencilLoadOp,
                        stencilStoreOp = src.StencilStoreOp,
                        initialLayout = src.InitialLayout,
                        finalLayout = src.FinalLayout
                    };
                    Marshal.StructureToPtr(vkAtt, attachmentPtr + i * structSize, false);
                }
                createInfo.pAttachments = attachmentPtr;
            }

            // Marshal subpass descriptions
            IntPtr subpassPtr = IntPtr.Zero;
            var subpassDataList = new List<VkSubpassDescription>();
            var colorAttachRefs = new List<IntPtr>();
            var inputAttachRefs = new List<IntPtr>();
            var depthAttachRef = IntPtr.Zero;
            IntPtr resolveAttachPtr = IntPtr.Zero;

            if (subpassCount > 0)
            {
                for (int s = 0; s < description.Subpasses.Length; s++)
                {
                    var sub = description.Subpasses[s];
                    IntPtr colorRefPtr = IntPtr.Zero;
                    IntPtr inputRefPtr = IntPtr.Zero;
                    int structSize = Marshal.SizeOf<VkAttachmentReference>();

                    if (sub.ColorAttachments != null && sub.ColorAttachments.Length > 0)
                    {
                        colorRefPtr = Marshal.AllocHGlobal(sub.ColorAttachments.Length * structSize);
                        for (int i = 0; i < sub.ColorAttachments.Length; i++)
                        {
                            var aref = new VkAttachmentReference { attachment = sub.ColorAttachments[i].Attachment, layout = sub.ColorAttachments[i].Layout };
                            Marshal.StructureToPtr(aref, colorRefPtr + i * structSize, false);
                        }
                    }
                    colorAttachRefs.Add(colorRefPtr);

                    if (sub.InputAttachments != null && sub.InputAttachments.Length > 0)
                    {
                        inputRefPtr = Marshal.AllocHGlobal(sub.InputAttachments.Length * structSize);
                        for (int i = 0; i < sub.InputAttachments.Length; i++)
                        {
                            var aref = new VkAttachmentReference { attachment = sub.InputAttachments[i].Attachment, layout = sub.InputAttachments[i].Layout };
                            Marshal.StructureToPtr(aref, inputRefPtr + i * structSize, false);
                        }
                    }
                    inputAttachRefs.Add(inputRefPtr);

                    IntPtr depthRefPtr = IntPtr.Zero;
                    if (sub.DepthStencilAttachment.Attachment != 0 || sub.DepthStencilAttachment.Layout != 0)
                    {
                        depthRefPtr = Marshal.AllocHGlobal(structSize);
                        var depthRef = new VkAttachmentReference { attachment = sub.DepthStencilAttachment.Attachment, layout = sub.DepthStencilAttachment.Layout };
                        Marshal.StructureToPtr(depthRef, depthRefPtr, false);
                    }

                    var subDesc = new VkSubpassDescription
                    {
                        pipelineBindPoint = sub.PipelineBindPoint,
                        colorAttachmentCount = (uint)(sub.ColorAttachments?.Length ?? 0),
                        pColorAttachments = colorRefPtr,
                        inputAttachmentCount = (uint)(sub.InputAttachments?.Length ?? 0),
                        pInputAttachments = inputRefPtr,
                        pDepthStencilAttachment = depthRefPtr,
                        pResolveAttachments = resolveAttachPtr
                    };
                    subpassDataList.Add(subDesc);
                }

                int subpassStructSize = Marshal.SizeOf<VkSubpassDescription>();
                subpassPtr = Marshal.AllocHGlobal((int)(subpassCount * subpassStructSize));
                for (int i = 0; i < subpassDataList.Count; i++)
                    Marshal.StructureToPtr(subpassDataList[i], subpassPtr + i * subpassStructSize, false);
                createInfo.pSubpasses = subpassPtr;
            }

            // Marshal dependencies
            IntPtr dependencyPtr = IntPtr.Zero;
            if (dependencyCount > 0)
            {
                int depStructSize = Marshal.SizeOf<VkSubpassDependency>();
                dependencyPtr = Marshal.AllocHGlobal((int)(dependencyCount * depStructSize));
                for (int i = 0; i < description.Dependencies.Length; i++)
                {
                    var dep = description.Dependencies[i];
                    var vkDep = new VkSubpassDependency
                    {
                        srcSubpass = dep.SrcSubpass,
                        dstSubpass = dep.DstSubpass,
                        srcStageMask = dep.SrcStageMask,
                        dstStageMask = dep.DstStageMask,
                        srcAccessMask = dep.SrcAccessMask,
                        dstAccessMask = dep.DstAccessMask,
                        dependencyFlags = (uint)dep.DependencyFlags
                    };
                    Marshal.StructureToPtr(vkDep, dependencyPtr + i * depStructSize, false);
                }
                createInfo.pDependencies = dependencyPtr;
            }

            var res = _vkCreateRenderPass(_logicalDevice, ref createInfo, IntPtr.Zero, ref renderPass);
            if (res != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create render pass: {res}");

            // Free allocated memory
            if (attachmentPtr != IntPtr.Zero) Marshal.FreeHGlobal(attachmentPtr);
            if (subpassPtr != IntPtr.Zero) Marshal.FreeHGlobal(subpassPtr);
            if (dependencyPtr != IntPtr.Zero) Marshal.FreeHGlobal(dependencyPtr);
            foreach (var ptr in colorAttachRefs) if (ptr != IntPtr.Zero) Marshal.FreeHGlobal(ptr);
            foreach (var ptr in inputAttachRefs) if (ptr != IntPtr.Zero) Marshal.FreeHGlobal(ptr);

            var rp = new VulkanRenderPass(_device, renderPass, description);
            _renderPasses.Add(rp);
            return rp;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkAttachmentDescription
        {
            public uint flags; public VulkanFormat format; public SampleCountFlag samples;
            public AttachmentLoadOp loadOp; public AttachmentStoreOp storeOp;
            public AttachmentLoadOp stencilLoadOp; public AttachmentStoreOp stencilStoreOp;
            public ImageLayout initialLayout; public ImageLayout finalLayout;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VkAttachmentReference { public uint attachment; public ImageLayout layout; }
        /// <summary>Creates a framebuffer for a render pass</summary>
        public VulkanFramebuffer CreateFramebuffer(FramebufferDescription description)
        {
            IntPtr framebuffer = IntPtr.Zero;
            var createInfo = new VkFramebufferCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass = description.RenderPass,
                attachmentCount = (uint)(description.Attachments?.Length ?? 0),
                width = description.Width,
                height = description.Height,
                layers = description.Layers
            };

            if (description.Attachments != null && description.Attachments.Length > 0)
            {
                var attPtr = Marshal.AllocHGlobal(description.Attachments.Length * IntPtr.Size);
                for (int i = 0; i < description.Attachments.Length; i++)
                    Marshal.WriteIntPtr(attPtr + i * IntPtr.Size, description.Attachments[i]);
                createInfo.pAttachments = attPtr;

                var result = _vkCreateFramebuffer(_logicalDevice, ref createInfo, IntPtr.Zero, ref framebuffer);
                Marshal.FreeHGlobal(attPtr);
                if (result != VulkanResult.Success)
                    throw new InvalidOperationException($"Failed to create framebuffer: {result}");
            }
            else
            {
                var result = _vkCreateFramebuffer(_logicalDevice, ref createInfo, IntPtr.Zero, ref framebuffer);
                if (result != VulkanResult.Success)
                    throw new InvalidOperationException($"Failed to create framebuffer: {result}");
            }

            var fb = new VulkanFramebuffer(_device, framebuffer, description);
            _framebuffers.Add(fb);
            return fb;
        }

        /// <summary>Creates a graphics pipeline</summary>
        public VulkanPipeline CreatePipeline(PipelineDescription description)
        {
            // Marshal shader stages
            int stageSize = Marshal.SizeOf<PipelineShaderStageCreateInfo>();
            IntPtr stagesPtr = Marshal.AllocHGlobal(description.ShaderStages.Length * stageSize);
            for (int i = 0; i < description.ShaderStages.Length; i++)
                Marshal.StructureToPtr(description.ShaderStages[i], stagesPtr + i * stageSize, false);

            // Create sub-state structs
            IntPtr vertexInputPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineVertexInputStateCreateInfo>());
            IntPtr inputAssemblyPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineInputAssemblyStateCreateInfo>());
            IntPtr tessellationPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineTessellationStateCreateInfo>());
            IntPtr viewportPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineViewportStateCreateInfo>());
            IntPtr rasterizerPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineRasterizationStateCreateInfo>());
            IntPtr multisamplePtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineMultisampleStateCreateInfo>());
            IntPtr depthStencilPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineDepthStencilStateCreateInfo>());
            IntPtr colorBlendPtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineColorBlendStateCreateInfo>());
            IntPtr dynamicStatePtr = Marshal.AllocHGlobal(Marshal.SizeOf<VkPipelineDynamicStateCreateInfo>());

            // Vertex input
            var vis = description.VertexInputState;
            var vkVis = new VkPipelineVertexInputStateCreateInfo
            {
                sType = 1000027000,
                vertexBindingDescriptionCount = vis.VertexBindingDescriptions != null ? (uint)vis.VertexBindingDescriptions.Length : 0,
                vertexAttributeDescriptionCount = vis.VertexAttributeDescriptions != null ? (uint)vis.VertexAttributeDescriptions.Length : 0
            };
            Marshal.StructureToPtr(vkVis, vertexInputPtr, false);

            // Input assembly
            var ias = description.InputAssemblyState;
            var vkIas = new VkPipelineInputAssemblyStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                topology = ias.Topology,
                primitiveRestartEnable = ias.PrimitiveRestartEnable ? VulkanBool32.True : VulkanBool32.False
            };
            Marshal.StructureToPtr(vkIas, inputAssemblyPtr, false);

            // Tessellation
            var tess = description.TessellationState;
            var vkTess = new VkPipelineTessellationStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                patchControlPoints = tess.PatchControlPoints
            };
            Marshal.StructureToPtr(vkTess, tessellationPtr, false);

            // Viewport
            var vp = description.ViewportState;
            var vkVp = new VkPipelineViewportStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                viewportCount = vp.ViewportCount,
                scissorCount = vp.ScissorCount
            };
            Marshal.StructureToPtr(vkVp, viewportPtr, false);

            // Rasterizer
            var rs = description.RasterizationState;
            var vkRs = new VkPipelineRasterizationStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                depthClampEnable = rs.DepthClampEnable ? VulkanBool32.True : VulkanBool32.False,
                rasterizerDiscardEnable = rs.RasterizerDiscardEnable ? VulkanBool32.True : VulkanBool32.False,
                polygonMode = rs.PolygonMode,
                cullMode = rs.CullMode,
                frontFace = rs.FrontFace,
                depthBiasEnable = rs.DepthBiasEnable ? VulkanBool32.True : VulkanBool32.False,
                depthBiasConstantFactor = rs.DepthBiasConstantFactor,
                depthBiasClamp = rs.DepthBiasClamp,
                depthBiasSlopeFactor = rs.DepthBiasSlopeFactor,
                lineWidth = rs.LineWidth
            };
            Marshal.StructureToPtr(vkRs, rasterizerPtr, false);

            // Multisample
            var ms = description.MultisampleState;
            var vkMs = new VkPipelineMultisampleStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                rasterizationSamples = ms.RasterizationSamples,
                sampleShadingEnable = ms.SampleShadingEnable ? VulkanBool32.True : VulkanBool32.False,
                minSampleShading = ms.MinSampleShading,
                alphaToCoverageEnable = ms.AlphaToCoverageEnable ? VulkanBool32.True : VulkanBool32.False,
                alphaToOneEnable = ms.AlphaToOneEnable ? VulkanBool32.True : VulkanBool32.False
            };
            Marshal.StructureToPtr(vkMs, multisamplePtr, false);

            // Depth stencil
            var ds = description.DepthStencilState;
            var vkDs = new VkPipelineDepthStencilStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                depthTestEnable = ds.DepthTestEnable,
                depthWriteEnable = ds.DepthWriteEnable,
                depthCompareOp = ds.DepthCompareOp,
                depthBoundsTestEnable = ds.DepthBoundsTestEnable,
                stencilTestEnable = ds.StencilTestEnable,
                minDepthBounds = ds.MinDepthBounds,
                maxDepthBounds = ds.MaxDepthBounds
            };
            Marshal.StructureToPtr(vkDs, depthStencilPtr, false);

            // Color blend
            var cb = description.ColorBlendState;
            var vkCb = new VkPipelineColorBlendStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                logicOpEnable = cb.LogicOpEnable ? VulkanBool32.True : VulkanBool32.False,
                logicOp = cb.LogicOp,
                attachmentCount = cb.Attachments != null ? (uint)cb.Attachments.Length : 0
            };
            Marshal.StructureToPtr(vkCb, colorBlendPtr, false);

            // Dynamic state
            var vkDyn = new VkPipelineDynamicStateCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                dynamicStateCount = description.DynamicState.DynamicStates != null ? (uint)description.DynamicState.DynamicStates.Length : 0
            };
            Marshal.StructureToPtr(vkDyn, dynamicStatePtr, false);

            // Create graphics pipeline
            var pipelineCI = new VkGraphicsPipelineCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                flags = description.Flags,
                stageCount = (uint)description.ShaderStages.Length,
                pStages = stagesPtr,
                pVertexInputState = vertexInputPtr,
                pInputAssemblyState = inputAssemblyPtr,
                pTessellationState = tessellationPtr,
                pViewportState = viewportPtr,
                pRasterizationState = rasterizerPtr,
                pMultisampleState = multisamplePtr,
                pDepthStencilState = depthStencilPtr,
                pColorBlendState = colorBlendPtr,
                pDynamicState = dynamicStatePtr,
                layout = description.PipelineLayout,
                renderPass = description.RenderPass,
                subpass = description.Subpass,
                basePipelineHandle = description.BasePipelineHandle,
                basePipelineIndex = description.BasePipelineIndex
            };

            IntPtr pipeline = IntPtr.Zero;
            var result = _vkCreateGraphicsPipelines(_logicalDevice, _pipelineCache, 1, ref pipelineCI, IntPtr.Zero, ref pipeline);

            // Free marshaled memory
            Marshal.FreeHGlobal(stagesPtr);
            Marshal.FreeHGlobal(vertexInputPtr);
            Marshal.FreeHGlobal(inputAssemblyPtr);
            Marshal.FreeHGlobal(tessellationPtr);
            Marshal.FreeHGlobal(viewportPtr);
            Marshal.FreeHGlobal(rasterizerPtr);
            Marshal.FreeHGlobal(multisamplePtr);
            Marshal.FreeHGlobal(depthStencilPtr);
            Marshal.FreeHGlobal(colorBlendPtr);
            Marshal.FreeHGlobal(dynamicStatePtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create graphics pipeline: {result}");

            var vp2 = new VulkanPipeline(_device, pipeline, description);
            _pipelines.Add(vp2);
            return vp2;
        }

        /// <summary>Creates a compute pipeline</summary>
        public VulkanComputePipeline CreateComputePipeline(ComputePipelineDescription description)
        {
            var stageInfo = description.Stage;
            int stageSize = Marshal.SizeOf<PipelineShaderStageCreateInfo>();
            IntPtr stagePtr = Marshal.AllocHGlobal(stageSize);
            Marshal.StructureToPtr(stageInfo, stagePtr, false);

            var pipelineCI = new VkComputePipelineCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                flags = description.Flags,
                stageCount = 1,
                pStage = stagePtr,
                layout = description.PipelineLayout,
                basePipelineHandle = description.BasePipelineHandle,
                basePipelineIndex = description.BasePipelineIndex
            };

            IntPtr pipeline = IntPtr.Zero;
            var result = _vkCreateComputePipelines(_logicalDevice, _pipelineCache, 1, ref pipelineCI, IntPtr.Zero, ref pipeline);
            Marshal.FreeHGlobal(stagePtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create compute pipeline: {result}");

            var cp = new VulkanComputePipeline(_device, pipeline, description);
            _computePipelines.Add(cp);
            return cp;
        }

        /// <summary>Creates a descriptor set layout</summary>
        public DescriptorSetLayout CreateDescriptorSetLayout(LayoutDescription description)
        {
            IntPtr bindingsPtr = IntPtr.Zero;
            int bindingCount = description.Bindings?.Length ?? 0;

            if (bindingCount > 0)
            {
                int structSize = Marshal.SizeOf<VkDescriptorSetLayoutBinding>();
                bindingsPtr = Marshal.AllocHGlobal(bindingCount * structSize);
                for (int i = 0; i < bindingCount; i++)
                {
                    var src = description.Bindings[i];
                    var vkBinding = new VkDescriptorSetLayoutBinding
                    {
                        binding = src.Binding,
                        descriptorType = src.DescriptorType,
                        descriptorCount = src.DescriptorCount,
                        stageFlags = src.StageFlags
                    };
                    Marshal.StructureToPtr(vkBinding, bindingsPtr + i * structSize, false);
                }
            }

            var createInfo = new VkDescriptorSetLayoutCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                flags = (uint)description.Flags,
                bindingCount = (uint)bindingCount,
                pBindings = bindingsPtr
            };

            IntPtr layout = IntPtr.Zero;
            var result = _vkCreateDescriptorSetLayout(_logicalDevice, ref createInfo, IntPtr.Zero, ref layout);
            if (bindingsPtr != IntPtr.Zero) Marshal.FreeHGlobal(bindingsPtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create descriptor set layout: {result}");

            var dsl = new DescriptorSetLayout(_device, layout, description);
            _descriptorSetLayouts.Add(dsl);
            return dsl;
        }

        /// <summary>Creates a descriptor pool</summary>
        public DescriptorPool CreateDescriptorPool(PoolDescription description)
        {
            IntPtr poolSizesPtr = IntPtr.Zero;
            int poolSizeCount = description.PoolSizes?.Length ?? 0;

            if (poolSizeCount > 0)
            {
                int structSize = Marshal.SizeOf<VkDescriptorPoolSize>();
                poolSizesPtr = Marshal.AllocHGlobal(poolSizeCount * structSize);
                for (int i = 0; i < poolSizeCount; i++)
                {
                    var vkSize = new VkDescriptorPoolSize
                    {
                        type = description.PoolSizes[i].Type,
                        descriptorCount = description.PoolSizes[i].DescriptorCount
                    };
                    Marshal.StructureToPtr(vkSize, poolSizesPtr + i * structSize, false);
                }
            }

            var createInfo = new VkDescriptorPoolCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                flags = (uint)description.Flags,
                maxSets = description.MaxSets,
                poolSizeCount = (uint)poolSizeCount,
                pPoolSizes = poolSizesPtr
            };

            IntPtr pool = IntPtr.Zero;
            var result = _vkCreateDescriptorPool(_logicalDevice, ref createInfo, IntPtr.Zero, ref pool);
            if (poolSizesPtr != IntPtr.Zero) Marshal.FreeHGlobal(poolSizesPtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create descriptor pool: {result}");

            var dp = new DescriptorPool(_device, pool, description);
            _descriptorPools.Add(dp);
            return dp;
        }

        /// <summary>Allocates descriptor sets from a pool</summary>
        public DescriptorSet[] AllocateDescriptorSets(DescriptorSetAllocation allocation)
        {
            int layoutCount = allocation.Layouts?.Length ?? 0;
            var layoutsPtr = Marshal.AllocHGlobal(layoutCount * IntPtr.Size);
            for (int i = 0; i < layoutCount; i++)
                Marshal.WriteIntPtr(layoutsPtr + i * IntPtr.Size, allocation.Layouts[i]);

            var allocInfo = new VkDescriptorSetAllocateInfo
            {
                sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool = allocation.Pool,
                descriptorSetCount = (uint)layoutCount,
                pSetLayouts = layoutsPtr
            };

            var descriptorSetsPtr = Marshal.AllocHGlobal(layoutCount * IntPtr.Size);
            var result = _vkAllocateDescriptorSets(_logicalDevice, ref allocInfo, descriptorSetsPtr);
            Marshal.FreeHGlobal(layoutsPtr);

            var descriptorSets = new IntPtr[layoutCount];
            for (int i = 0; i < layoutCount; i++)
                descriptorSets[i] = Marshal.ReadIntPtr(descriptorSetsPtr + i * IntPtr.Size);
            Marshal.FreeHGlobal(descriptorSetsPtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to allocate descriptor sets: {result}");

            return descriptorSets.Select(h => new DescriptorSet(_device, h)).ToArray();
        }

        /// <summary>Updates descriptor sets with new binding data</summary>
        public void UpdateDescriptorSets(DescriptorWrite[] writes)
        {
            if (writes == null || writes.Length == 0) return;

            int writeStructSize = Marshal.SizeOf<VkWriteDescriptorSet>();
            var writePtrs = new IntPtr[writes.Length];
            var tempAllocs = new List<IntPtr>();

            for (int w = 0; w < writes.Length; w++)
            {
                var write = writes[w];
                IntPtr imageInfoPtr = IntPtr.Zero;
                IntPtr bufferInfoPtr = IntPtr.Zero;

                if (write.ImageInfos != null && write.ImageInfos.Length > 0)
                {
                    int imgInfoSize = Marshal.SizeOf<VkDescriptorImageInfo>();
                    imageInfoPtr = Marshal.AllocHGlobal(write.ImageInfos.Length * imgInfoSize);
                    tempAllocs.Add(imageInfoPtr);
                    for (int i = 0; i < write.ImageInfos.Length; i++)
                    {
                        var imgInfo = new VkDescriptorImageInfo
                        {
                            sampler = write.ImageInfos[i].Sampler,
                            imageView = write.ImageInfos[i].ImageView,
                            imageLayout = write.ImageInfos[i].ImageLayout
                        };
                        Marshal.StructureToPtr(imgInfo, imageInfoPtr + i * imgInfoSize, false);
                    }
                }

                if (write.BufferInfos != null && write.BufferInfos.Length > 0)
                {
                    int bufInfoSize = Marshal.SizeOf<VkDescriptorBufferInfo>();
                    bufferInfoPtr = Marshal.AllocHGlobal(write.BufferInfos.Length * bufInfoSize);
                    tempAllocs.Add(bufferInfoPtr);
                    for (int i = 0; i < write.BufferInfos.Length; i++)
                    {
                        var bufInfo = new VkDescriptorBufferInfo
                        {
                            buffer = write.BufferInfos[i].Buffer,
                            offset = write.BufferInfos[i].Offset,
                            range = write.BufferInfos[i].Range
                        };
                        Marshal.StructureToPtr(bufInfo, bufferInfoPtr + i * bufInfoSize, false);
                    }
                }

                var vkWrite = new VkWriteDescriptorSet
                {
                    sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet = write.DescriptorSet,
                    dstBinding = write.DstBinding,
                    dstArrayElement = write.DstArrayElement,
                    descriptorCount = (uint)(write.ImageInfos?.Length ?? write.BufferInfos?.Length ?? 0),
                    descriptorType = write.DescriptorType,
                    pImageInfo = imageInfoPtr,
                    pBufferInfo = bufferInfoPtr
                };

                var writeAlloc = Marshal.AllocHGlobal(writeStructSize);
                tempAllocs.Add(writeAlloc);
                Marshal.StructureToPtr(vkWrite, writeAlloc, false);
                writePtrs[w] = writeAlloc;
            }

            var writesArrayPtr = Marshal.AllocHGlobal(writes.Length * writeStructSize);
            tempAllocs.Add(writesArrayPtr);
            for (int i = 0; i < writes.Length; i++)
            {
                var src = Marshal.PtrToStructure<VkWriteDescriptorSet>(writePtrs[i]);
                Marshal.StructureToPtr(src, writesArrayPtr + i * writeStructSize, false);
            }

            _vkUpdateDescriptorSets(_logicalDevice, (uint)writes.Length, writesArrayPtr, 0, IntPtr.Zero);

            foreach (var ptr in tempAllocs) Marshal.FreeHGlobal(ptr);
        }

        /// <summary>Creates a shader module from SPIR-V bytecode</summary>
        public VulkanShaderModule CreateShaderModule(byte[] spirvCode)
        {
            if (spirvCode == null || spirvCode.Length == 0)
                throw new ArgumentException("SPIR-V code cannot be null or empty", nameof(spirvCode));

            if (spirvCode.Length % 4 != 0)
                throw new ArgumentException("SPIR-V code size must be a multiple of 4", nameof(spirvCode));

            var codePtr = Marshal.AllocHGlobal(spirvCode.Length);
            Marshal.Copy(spirvCode, 0, codePtr, spirvCode.Length);

            var createInfo = new VkShaderModuleCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize = (IntPtr)spirvCode.Length,
                pCode = codePtr
            };

            IntPtr module = IntPtr.Zero;
            var result = _vkCreateShaderModule(_logicalDevice, ref createInfo, IntPtr.Zero, ref module);
            Marshal.FreeHGlobal(codePtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create shader module: {result}");

            var sm = new VulkanShaderModule(_device, module, spirvCode);
            _shaderModules.Add(sm);
            return sm;
        }

        /// <summary>Creates a sampler</summary>
        public Sampler CreateSampler(SamplerDescription description)
        {
            var createInfo = new VkSamplerCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                magFilter = description.MagFilter,
                minFilter = description.MinFilter,
                mipmapMode = SamplerAddressMode.Repeat,
                addressModeU = description.AddressModeU,
                addressModeV = description.AddressModeV,
                addressModeW = description.AddressModeW,
                mipLodBias = description.MipLodBias,
                anisotropyEnable = description.AnisotropyEnable ? VulkanBool32.True : VulkanBool32.False,
                maxAnisotropy = description.MaxAnisotropy,
                compareEnable = description.CompareEnable ? VulkanBool32.True : VulkanBool32.False,
                compareOp = description.CompareOp,
                minLod = description.MinLod,
                maxLod = description.MaxLod,
                borderColor = 0,
                unnormalizedCoordinates = VulkanBool32.False
            };

            IntPtr sampler = IntPtr.Zero;
            var result = _vkCreateSampler(_logicalDevice, ref createInfo, IntPtr.Zero, ref sampler);
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create sampler: {result}");

            var s = new Sampler(_device, sampler, description);
            _samplers.Add(s);
            return s;
        }

        /// <summary>Creates a command buffer</summary>
        public VulkanCommandBuffer CreateCommandBuffer()
        {
            var allocInfo = new VkCommandBufferAllocateInfo
            {
                sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool = _commandPool,
                level = 0,
                commandBufferCount = 1
            };

            var cmdBufferPtr = Marshal.AllocHGlobal(IntPtr.Size);
            var result = _vkAllocateCommandBuffers(_logicalDevice, ref allocInfo, cmdBufferPtr);
            var cmdBuffer = Marshal.ReadIntPtr(cmdBufferPtr);
            Marshal.FreeHGlobal(cmdBufferPtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to allocate command buffer: {result}");

            var cb = new VulkanCommandBuffer(_device, cmdBuffer, _commandPool);
            _commandBuffers.Add(cb);
            return cb;
        }

        /// <summary>Submits a command buffer to a queue</summary>
        public void SubmitCommandBuffer(VulkanCommandBuffer commandBuffer, IntPtr queue, Fence fence)
        {
            var cmdBufferPtr = commandBuffer.Handle;
            var submitInfo = new VkSubmitInfo
            {
                sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount = 1,
                pCommandBuffers = Marshal.AllocHGlobal(IntPtr.Size),
                waitSemaphoreCount = 0,
                signalSemaphoreCount = 0
            };
            Marshal.WriteIntPtr(submitInfo.pCommandBuffers, cmdBufferPtr);

            var result = _vkQueueSubmit(queue, 1, ref submitInfo, fence?.Handle ?? IntPtr.Zero);
            Marshal.FreeHGlobal(submitInfo.pCommandBuffers);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to submit command buffer: {result}");
        }

        /// <summary>Waits for all device operations to complete</summary>
        public void WaitForIdle()
        {
            _vkDeviceWaitIdle(_logicalDevice);
        }

        /// <summary>Queries hardware capabilities</summary>
        public HardwareCapabilities QueryCapabilities()
        {
            return new HardwareCapabilities
            {
                DeviceName = _physicalDeviceInfo.DeviceName,
                VendorId = _physicalDeviceInfo.VendorId.ToString("X"),
                DeviceId = _physicalDeviceInfo.DeviceId,
                ApiVersion = _physicalDeviceInfo.ApiVersion,
                MaxTextureDimension1D = _physicalDeviceInfo.MaxTextureDimension1D,
                MaxTextureDimension2D = _physicalDeviceInfo.MaxTextureDimension2D,
                MaxTextureDimension3D = _physicalDeviceInfo.MaxTextureDimension3D,
                MaxTextureArrayLayers = _physicalDeviceInfo.MaxTextureArrayLayers,
                MaxPushConstantsSize = _physicalDeviceInfo.MaxPushConstantsSize,
                MaxBoundDescriptorSets = _physicalDeviceInfo.MaxBoundDescriptorSets,
                MaxColorAttachments = _physicalDeviceInfo.MaxColorAttachments,
                MaxVertexInputAttributes = _physicalDeviceInfo.MaxVertexInputAttributes,
                MaxVertexInputBindings = _physicalDeviceInfo.MaxVertexInputBindings,
                MaxVertexInputAttributeOffset = _physicalDeviceInfo.MaxVertexInputAttributeOffset,
                MaxVertexInputBindingStride = _physicalDeviceInfo.MaxVertexInputBindingStride,
                MaxVertexOutputComponents = _physicalDeviceInfo.MaxVertexOutputComponents,
                MaxSamplerAnisotropy = _physicalDeviceInfo.MaxSamplerAnisotropy,
                SupportsTimelineSemaphores = _physicalDeviceInfo.SupportsTimelineSemaphores,
                SupportsBufferDeviceAddress = _physicalDeviceInfo.SupportsBufferDeviceAddress,
                SupportsDynamicRendering = _physicalDeviceInfo.SupportsDynamicRendering,
                SupportsSynchronization2 = _physicalDeviceInfo.SupportsSynchronization2,
                MaxComputeWorkGroupCount0 = _physicalDeviceInfo.MaxComputeWorkGroupCount0,
                MaxComputeWorkGroupCount1 = _physicalDeviceInfo.MaxComputeWorkGroupCount1,
                MaxComputeWorkGroupCount2 = _physicalDeviceInfo.MaxComputeWorkGroupCount2,
                MaxComputeWorkGroupSize0 = _physicalDeviceInfo.MaxComputeWorkGroupSize0,
                MaxComputeWorkGroupSize1 = _physicalDeviceInfo.MaxComputeWorkGroupSize1,
                MaxComputeWorkGroupSize2 = _physicalDeviceInfo.MaxComputeWorkGroupSize2,
                MaxComputeWorkGroupInvocations = _physicalDeviceInfo.MaxComputeWorkGroupInvocations,
                MemoryProperties = _physicalDeviceInfo.MemoryProperties
            };
        }

        /// <summary>Creates the Vulkan RHI device with the given configuration</summary>
        public VulkanRhiDevice(RhiDeviceCreationInfo creationInfo)
        {
            _creationInfo = creationInfo;
            _device = new VulkanDevice();
            _physicalDeviceInfo = new VulkanPhysicalDeviceInfo();
            _queueFamilyIndices = new QueueFamilyIndices();

            CreateInstance();
            PickPhysicalDevice();
            _device.Instance = _instance;
            _device.PhysicalDevice = _physicalDevice;
            _device.PhysicalDeviceInfo = _physicalDeviceInfo;
            CreateLogicalDevice();
            InitializeQueues();
            CreatePipelineCacheInternal();
            InitializeSubsystems();
            CreateDebugMessenger();
        }

        private void CreateInstance()
        {
            var appInfo = new VkApplicationInfo
            {
                sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pApplicationName = Marshal.StringToHGlobalAnsi(_creationInfo.ApplicationName),
                applicationVersion = _creationInfo.ApplicationVersion,
                pEngineName = Marshal.StringToHGlobalAnsi(_creationInfo.EngineName),
                engineVersion = _creationInfo.EngineVersion,
                apiVersion = _creationInfo.ApiVersion
            };

            var extensions = new List<string>();
            if (_creationInfo.RequiredExtensions is { Length: > 0 })
            {
                foreach (var ext in _creationInfo.RequiredExtensions)
                {
                    if (!string.IsNullOrWhiteSpace(ext) && !extensions.Contains(ext))
                        extensions.Add(ext);
                }
            }
            else
            {
                // Legacy default: Windows Win32 surface when caller does not specify.
                extensions.Add("VK_KHR_surface");
                if (OperatingSystem.IsWindows())
                    extensions.Add("VK_KHR_win32_surface");
            }

            if (!extensions.Contains("VK_KHR_surface"))
                extensions.Insert(0, "VK_KHR_surface");

            if (_creationInfo.EnableValidation && !extensions.Contains("VK_EXT_debug_utils"))
                extensions.Add("VK_EXT_debug_utils");

            // MoltenVK / portability on macOS
            if (OperatingSystem.IsMacOS() && !extensions.Contains("VK_KHR_portability_enumeration"))
                extensions.Add("VK_KHR_portability_enumeration");

            var extensionPtrs = extensions.Select(e => Marshal.StringToHGlobalAnsi(e)).ToArray();
            var layerNames = new List<string>();
            if (_creationInfo.EnableValidation) layerNames.Add("VK_LAYER_KHRONOS_validation");
            var layerPtrs = layerNames.Select(l => Marshal.StringToHGlobalAnsi(l)).ToArray();

            var createInfo = new VkInstanceCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR — required for MoltenVK on macOS
                flags = OperatingSystem.IsMacOS() ? 0x00000001u : 0u,
                pApplicationInfo = Marshal.AllocHGlobal(Marshal.SizeOf<VkApplicationInfo>())
            };
            Marshal.StructureToPtr(appInfo, createInfo.pApplicationInfo, false);

            unsafe
            {
                createInfo.enabledExtensionCount = (uint)extensionPtrs.Length;
                createInfo.ppEnabledExtensionNames = Marshal.AllocHGlobal(extensionPtrs.Length * IntPtr.Size);
                for (int i = 0; i < extensionPtrs.Length; i++)
                    Marshal.WriteIntPtr(createInfo.ppEnabledExtensionNames + i * IntPtr.Size, extensionPtrs[i]);

                createInfo.enabledLayerCount = (uint)layerPtrs.Length;
                createInfo.ppEnabledLayerNames = Marshal.AllocHGlobal(layerPtrs.Length * IntPtr.Size);
                for (int i = 0; i < layerPtrs.Length; i++)
                    Marshal.WriteIntPtr(createInfo.ppEnabledLayerNames + i * IntPtr.Size, layerPtrs[i]);
            }

            var result = vkCreateInstance(ref createInfo, IntPtr.Zero, ref _instance);
            Marshal.FreeHGlobal(createInfo.pApplicationInfo);
            foreach (var ptr in extensionPtrs) Marshal.FreeHGlobal(ptr);
            foreach (var ptr in layerPtrs) Marshal.FreeHGlobal(ptr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create Vulkan instance: {result}");
        }

        /// <summary>Creates a pipeline layout</summary>
        public VulkanPipelineLayout CreatePipelineLayout(IntPtr[] setLayouts, (ShaderStageFlag stage, uint offset, uint size)[] pushConstants = null)
        {
            IntPtr setLayoutsPtr = IntPtr.Zero;
            if (setLayouts != null && setLayouts.Length > 0)
            {
                setLayoutsPtr = Marshal.AllocHGlobal(setLayouts.Length * IntPtr.Size);
                for (int i = 0; i < setLayouts.Length; i++)
                    Marshal.WriteIntPtr(setLayoutsPtr + i * IntPtr.Size, setLayouts[i]);
            }

            VkPushConstantRange[] pushRanges = null;
            IntPtr pushRangesPtr = IntPtr.Zero;
            if (pushConstants != null && pushConstants.Length > 0)
            {
                pushRanges = new VkPushConstantRange[pushConstants.Length];
                for (int i = 0; i < pushConstants.Length; i++)
                {
                    pushRanges[i] = new VkPushConstantRange
                    {
                        stageFlags = pushConstants[i].stage,
                        offset = pushConstants[i].offset,
                        size = pushConstants[i].size
                    };
                }
                int structSize = Marshal.SizeOf<VkPushConstantRange>();
                pushRangesPtr = Marshal.AllocHGlobal(pushRanges.Length * structSize);
                for (int i = 0; i < pushRanges.Length; i++)
                    Marshal.StructureToPtr(pushRanges[i], pushRangesPtr + i * structSize, false);
            }

            var createInfo = new VkPipelineLayoutCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount = (uint)(setLayouts?.Length ?? 0),
                pSetLayouts = setLayoutsPtr,
                pushConstantRangeCount = (uint)(pushConstants?.Length ?? 0),
                pPushConstantRanges = pushRangesPtr
            };

            IntPtr layout = IntPtr.Zero;
            var result = _vkCreatePipelineLayout(_logicalDevice, ref createInfo, IntPtr.Zero, ref layout);

            if (setLayoutsPtr != IntPtr.Zero) Marshal.FreeHGlobal(setLayoutsPtr);
            if (pushRangesPtr != IntPtr.Zero) Marshal.FreeHGlobal(pushRangesPtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to create pipeline layout: {result}");

            return new VulkanPipelineLayout(_device, layout);
        }

        /// <summary>Submits a command buffer with semaphore synchronization</summary>
        public void SubmitCommandBufferWithSemaphores(
            VulkanCommandBuffer commandBuffer,
            IntPtr waitSemaphore,
            PipelineStageFlag waitStage,
            IntPtr signalSemaphore,
            IntPtr fence)
        {
            var cmdBufferPtr = commandBuffer.Handle;

            var waitSemaphoreCount = waitSemaphore != IntPtr.Zero ? 1u : 0u;
            var signalSemaphoreCount = signalSemaphore != IntPtr.Zero ? 1u : 0u;

            var waitStagePtr = Marshal.AllocHGlobal(sizeof(uint));
            Marshal.WriteInt32(waitStagePtr, (int)waitStage);

            var waitSemaphorePtr = IntPtr.Zero;
            if (waitSemaphore != IntPtr.Zero)
            {
                waitSemaphorePtr = Marshal.AllocHGlobal(IntPtr.Size);
                Marshal.WriteIntPtr(waitSemaphorePtr, waitSemaphore);
            }

            var signalSemaphorePtr = IntPtr.Zero;
            if (signalSemaphore != IntPtr.Zero)
            {
                signalSemaphorePtr = Marshal.AllocHGlobal(IntPtr.Size);
                Marshal.WriteIntPtr(signalSemaphorePtr, signalSemaphore);
            }

            var cmdBufferArrayPtr = Marshal.AllocHGlobal(IntPtr.Size);
            Marshal.WriteIntPtr(cmdBufferArrayPtr, cmdBufferPtr);

            var submitInfo = new VkSubmitInfo
            {
                sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount = 1,
                pCommandBuffers = cmdBufferArrayPtr,
                waitSemaphoreCount = waitSemaphoreCount,
                pWaitSemaphores = waitSemaphorePtr,
                pWaitDstStageMask = waitStagePtr,
                signalSemaphoreCount = signalSemaphoreCount,
                pSignalSemaphores = signalSemaphorePtr
            };

            var result = _vkQueueSubmit(_graphicsQueue, 1, ref submitInfo, fence);
            Marshal.FreeHGlobal(cmdBufferArrayPtr);
            Marshal.FreeHGlobal(waitStagePtr);
            if (waitSemaphorePtr != IntPtr.Zero) Marshal.FreeHGlobal(waitSemaphorePtr);
            if (signalSemaphorePtr != IntPtr.Zero) Marshal.FreeHGlobal(signalSemaphorePtr);

            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to submit command buffer: {result}");
        }

        /// <summary>Gets the graphics queue handle</summary>
        public IntPtr GraphicsQueue => _graphicsQueue;
        public IntPtr ComputeQueue => _computeQueue != IntPtr.Zero ? _computeQueue : _graphicsQueue;
        /// <summary>Gets the present queue handle</summary>
        public IntPtr PresentQueue => _presentQueue;

        /// <summary>Disposes the device and all resources</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            WaitForIdle();
            Cleanup();
            GC.SuppressFinalize(this);
        }

        private void Cleanup()
        {
            foreach (var s in _samplers) s?.Dispose();
            foreach (var sm in _shaderModules) sm?.Dispose();
            foreach (var p in _pipelines) p?.Dispose();
            foreach (var cp in _computePipelines) cp?.Dispose();
            foreach (var fb in _framebuffers) fb?.Dispose();
            foreach (var rp in _renderPasses) rp?.Dispose();
            foreach (var t in _textures) t?.Dispose();
            foreach (var b in _buffers) b?.Dispose();
            foreach (var dpl in _descriptorPools) dpl?.Dispose();
            foreach (var dsl in _descriptorSetLayouts) dsl?.Dispose();

            _memoryAllocator?.Dispose();
            _syncManager?.Dispose();
            _resourceTracker?.Dispose();

            if (_pipelineCache != IntPtr.Zero)
                _vkDestroyPipelineCache?.Invoke(_logicalDevice, _pipelineCache, IntPtr.Zero);

            if (_commandPool != IntPtr.Zero && _vkDestroyCommandPool != null)
                _vkDestroyCommandPool(_logicalDevice, _commandPool, IntPtr.Zero);

            if (_logicalDevice != IntPtr.Zero)
                vkDestroyDevice(_logicalDevice, IntPtr.Zero);

            if (_debugUtilsMessenger != IntPtr.Zero && _vkDestroyDebugUtilsMessenger != null)
                _vkDestroyDebugUtilsMessenger(_instance, _debugUtilsMessenger, IntPtr.Zero);

            if (_instance != IntPtr.Zero)
                vkDestroyInstance(_instance, IntPtr.Zero);
        }
    }
    // =========================================================================
    // VulkanSwapchain
    // =========================================================================

    /// <summary>
    /// Manages a Vulkan swapchain and its associated images.
    /// Handles image acquisition, presentation, and resize operations.
    /// </summary>
    public class VulkanSwapchain : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _swapchain;
        private IntPtr _surface;
        private IntPtr[] _images;
        private VulkanTexture[] _imageWrappers;
        private IntPtr[] _imageViews;
        private VulkanFormat _surfaceFormat;
        private PresentMode _presentMode;
        private Extent2D _extent;
        private uint _currentImageIndex;
        private bool _disposed;

        // Function pointers
        private GetSwapchainImagesKHRDel _vkGetSwapchainImagesKHR;
        private AcquireNextImageKHRDel _vkAcquireNextImageKHR;
        private QueuePresentKHRDel _vkQueuePresentKHR;
        private CreateImageViewDel _vkCreateImageView;
        private DestroyImageViewDel _vkDestroyImageView;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate VulkanResult GetSwapchainImagesKHRDel(IntPtr device, IntPtr swapchain, ref uint pSwapchainImageCount, IntPtr pSwapchainImages);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate VulkanResult AcquireNextImageKHRDel(IntPtr device, IntPtr swapchain, ulong timeout, IntPtr semaphore, IntPtr fence, ref uint pImageIndex);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate VulkanResult QueuePresentKHRDel(IntPtr queue, ref VkPresentInfoKHR pPresentInfo);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate VulkanResult CreateImageViewDel(IntPtr device, ref VkImageViewCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pView);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate void DestroyImageViewDel(IntPtr device, IntPtr imageView, IntPtr pAllocator);

        /// <summary>Swapchain handle</summary>
        public IntPtr Handle => _swapchain;

        /// <summary>Current extent of swapchain images</summary>
        public Extent2D Extent => _extent;

        /// <summary>Surface format of the swapchain</summary>
        public VulkanFormat SurfaceFormat => _surfaceFormat;

        /// <summary>Present mode of the swapchain</summary>
        public PresentMode CurrentPresentMode => _presentMode;

        public VulkanSwapchain(VulkanDevice device, IntPtr swapchain, VulkanFormat surfaceFormat, PresentMode presentMode, Extent2D extent)
        {
            _device = device;
            _swapchain = swapchain;
            _surfaceFormat = surfaceFormat;
            _presentMode = presentMode;
            _extent = extent;

            LoadFunctions();
            RetrieveImages();
            CreateImageViews();
        }

        private void LoadFunctions()
        {
            _vkGetSwapchainImagesKHR = LoadDeviceFunction<GetSwapchainImagesKHRDel>("vkGetSwapchainImagesKHR");
            _vkAcquireNextImageKHR = LoadDeviceFunction<AcquireNextImageKHRDel>("vkAcquireNextImageKHR");
            _vkQueuePresentKHR = LoadDeviceFunction<QueuePresentKHRDel>("vkQueuePresentKHR");
            _vkCreateImageView = LoadDeviceFunction<CreateImageViewDel>("vkCreateImageView");
            _vkDestroyImageView = LoadDeviceFunction<DestroyImageViewDel>("vkDestroyImageView");
        }

        private T LoadDeviceFunction<T>(string name) where T : Delegate
        {
            var namePtr = Marshal.StringToHGlobalAnsi(name);
            try
            {
                var ptr = _device.GetDeviceProcAddr(name);
                if (ptr == IntPtr.Zero)
                    ptr = GetProcAddress("vulkan-1.dll", namePtr);
                if (ptr != IntPtr.Zero)
                    return Marshal.GetDelegateForFunctionPointer<T>(ptr);
            }
            finally { Marshal.FreeHGlobal(namePtr); }
            return null;
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr GetProcAddress(string hModule, IntPtr procName);

        private void RetrieveImages()
        {
            uint imageCount = 0;
            _vkGetSwapchainImagesKHR(_device.LogicalDevice, _swapchain, ref imageCount, IntPtr.Zero);
            _images = new IntPtr[imageCount];
            var imagesHandle = GCHandle.Alloc(_images, GCHandleType.Pinned);
            try
            {
                _vkGetSwapchainImagesKHR(_device.LogicalDevice, _swapchain, ref imageCount, imagesHandle.AddrOfPinnedObject());
            }
            finally
            {
                imagesHandle.Free();
            }
            _imageWrappers = new VulkanTexture[imageCount];

            for (int i = 0; i < imageCount; i++)
            {
                _imageWrappers[i] = new VulkanTexture(_device, _images[i], new TextureDescription
                {
                    Width = _extent.Width,
                    Height = _extent.Height,
                    Format = _surfaceFormat
                });
            }
        }

        private void CreateImageViews()
        {
            _imageViews = new IntPtr[_images.Length];
            for (int i = 0; i < _images.Length; i++)
            {
                var createInfo = new VkImageViewCreateInfo
                {
                    sType = 15,
                    image = _images[i],
                    viewType = ImageViewType.Type2D,
                    format = _surfaceFormat,
                    subresourceRange = new VkImageSubresourceRange
                    {
                        aspectMask = ImageAspectFlag.Color,
                        baseMipLevel = 0,
                        levelCount = 1,
                        baseArrayLayer = 0,
                        layerCount = 1
                    }
                };
                createInfo.components = new VkComponentMapping { r = 0, g = 1, b = 2, a = 3 };

                _vkCreateImageView(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref _imageViews[i]);
            }
        }

        /// <summary>Acquires the next presentable image</summary>
        public uint AcquireNextImage(ulong timeout = 0xFFFFFFFFFFFFFFFF)
        {
            uint imageIndex = 0;
            var semaphore = _device.SyncManager?.CreateSemaphore() ?? IntPtr.Zero;
            var fence = _device.SyncManager?.CreateFence(false) ?? IntPtr.Zero;

            var result = _vkAcquireNextImageKHR(_device.LogicalDevice, _swapchain, timeout, semaphore, fence, ref imageIndex);

            if (result == VulkanResult.Success || result == VulkanResult.SuboptimalKHR || result == VulkanResult.ErrorOutOfDateKHR)
            {
                _currentImageIndex = imageIndex;
                return imageIndex;
            }

            return 0;
        }

        /// <summary>Presents the current image to the display</summary>
        public VulkanResult Present(IntPtr queue)
        {
            var imageIndex = _currentImageIndex;
            var presentInfo = new VkPresentInfoKHR
            {
                sType = 1000001001,
                swapchainCount = 1,
                pSwapchains = Marshal.AllocHGlobal(IntPtr.Size),
                pImageIndices = Marshal.AllocHGlobal(sizeof(uint))
            };
            Marshal.WriteIntPtr(presentInfo.pSwapchains, _swapchain);
            Marshal.WriteInt32(presentInfo.pImageIndices, (int)imageIndex);

            var result = _vkQueuePresentKHR(queue, ref presentInfo);

            Marshal.FreeHGlobal(presentInfo.pSwapchains);
            Marshal.FreeHGlobal(presentInfo.pImageIndices);

            return result;
        }

        /// <summary>Resizes the swapchain</summary>
        public void Resize(uint width, uint height)
        {
            if (width == 0 || height == 0) return;
            _extent = new Extent2D(width, height);
            CleanupImageViews();
            CleanupImages();

            // Old swapchain is passed for recreation
            _swapchain = IntPtr.Zero;
            RetrieveImages();
            CreateImageViews();
        }

        /// <summary>Gets all swapchain images</summary>
        public VulkanTexture[] GetImages() => _imageWrappers;

        /// <summary>Gets the current swapchain image</summary>
        public VulkanTexture GetCurrentImage() => _imageWrappers[_currentImageIndex];

        /// <summary>Gets the surface format</summary>
        public VulkanFormat GetSurfaceFormat() => _surfaceFormat;

        /// <summary>Gets the present mode</summary>
        public PresentMode GetPresentMode() => _presentMode;

        /// <summary>Gets an image view handle by index</summary>
        public IntPtr GetImageView(uint index) => index < _imageViews.Length ? _imageViews[index] : IntPtr.Zero;

        /// <summary>Gets the image handle by index</summary>
        public IntPtr GetImageHandle(uint index) => index < _images.Length ? _images[index] : IntPtr.Zero;

        /// <summary>Gets the number of images in the swapchain</summary>
        public uint ImageCount => (uint)_images.Length;

        /// <summary>Gets the current image index</summary>
        public uint CurrentImageIndex => _currentImageIndex;

        private void CleanupImageViews()
        {
            if (_imageViews == null) return;
            foreach (var iv in _imageViews)
                if (iv != IntPtr.Zero) _vkDestroyImageView?.Invoke(_device.LogicalDevice, iv, IntPtr.Zero);
        }

        private void CleanupImages()
        {
            if (_imageWrappers == null) return;
            foreach (var img in _imageWrappers)
                img?.Dispose();
            _imageWrappers = null;
            _images = null;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            CleanupImageViews();
            CleanupImages();
            GC.SuppressFinalize(this);
        }
    }
    // =========================================================================
    // VulkanCommandBuffer
    // =========================================================================

    /// <summary>
    /// Wraps a Vulkan command buffer for recording and submitting GPU commands.
    /// Provides methods for render pass management, draw calls, compute dispatch,
    /// and resource transitions.
    /// </summary>
    public class VulkanCommandBuffer : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _commandBuffer;
        private IntPtr _commandPool;
        private bool _isRecording;
        private bool _disposed;

        // Cached function pointers
        private CmdBeginRenderPassDel _vkCmdBeginRenderPass;
        private CmdEndRenderPassDel _vkCmdEndRenderPass;
        private CmdBindPipelineDel _vkCmdBindPipeline;
        private CmdDrawDel _vkCmdDraw;
        private CmdDrawIndexedDel _vkCmdDrawIndexed;
        private CmdDispatchDel _vkCmdDispatch;
        private CmdPipelineBarrierDel _vkCmdPipelineBarrier;
        private CmdCopyBufferDel _vkCmdCopyBuffer;
        private CmdCopyBufferToImageDel _vkCmdCopyBufferToImage;
        private CmdPushConstantsDel _vkCmdPushConstants;
        private CmdSetViewportDel _vkCmdSetViewport;
        private CmdSetScissorDel _vkCmdSetScissor;
        private CmdBindVertexBuffersDel _vkCmdBindVertexBuffers;
        private CmdBindIndexBufferDel _vkCmdBindIndexBuffer;
        private CmdCopyImageDel _vkCmdCopyImage;
        private CmdBlitImageDel _vkCmdBlitImage;
        private CmdResolveImageDel _vkCmdResolveImage;
        private CmdWriteTimestampDel _vkCmdWriteTimestamp;
        private CmdBeginQueryDel _vkCmdBeginQuery;
        private CmdEndQueryDel _vkCmdEndQuery;
        private CmdCopyQueryPoolResultsDel _vkCmdCopyQueryPoolResults;
        private CmdBindDescriptorSetsDel _vkCmdBindDescriptorSets;

        // Delegates
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginRenderPassDel(IntPtr cmdBuffer, ref VkRenderPassBeginInfo pRenderPassBegin, uint contents);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndRenderPassDel(IntPtr cmdBuffer);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindPipelineDel(IntPtr cmdBuffer, PipelineBindPoint pipelineBindPoint, IntPtr pipeline);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDrawDel(IntPtr cmdBuffer, uint vertexCount, uint instanceCount, uint firstVertex, uint firstInstance);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDrawIndexedDel(IntPtr cmdBuffer, uint indexCount, uint instanceCount, uint firstIndex, int vertexOffset, uint firstInstance);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdDispatchDel(IntPtr cmdBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdPipelineBarrierDel(IntPtr cmdBuffer, PipelineStageFlag srcStageMask, PipelineStageFlag dstStageMask, uint dependencyFlags, uint memoryBarrierCount, IntPtr pMemoryBarriers, uint bufferMemoryBarrierCount, IntPtr pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferDel(IntPtr cmdBuffer, IntPtr srcBuffer, IntPtr dstBuffer, uint regionCount, ref BufferCopy pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferToImageDel(IntPtr cmdBuffer, IntPtr buffer, IntPtr image, uint imageLayout, uint regionCount, ref BufferImageCopy pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdPushConstantsDel(IntPtr cmdBuffer, IntPtr layout, ShaderStageFlag stageFlags, uint offset, uint size, IntPtr pValues);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetViewportDel(IntPtr cmdBuffer, uint firstViewport, uint viewportCount, ref Viewport pViewports);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdSetScissorDel(IntPtr cmdBuffer, uint firstScissor, uint scissorCount, ref Rect2D pScissors);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindVertexBuffersDel(IntPtr cmdBuffer, uint firstBinding, uint bindingCount, ref IntPtr pBuffers, ref ulong pOffsets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindIndexBufferDel(IntPtr cmdBuffer, IntPtr buffer, ulong offset, IndexType indexType);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyImageDel(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBlitImageDel(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions, Filter filter);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdResolveImageDel(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdWriteTimestampDel(IntPtr cmdBuffer, PipelineStageFlag pipelineStage, IntPtr queryPool, uint query);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBeginQueryDel(IntPtr cmdBuffer, IntPtr queryPool, uint query, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdEndQueryDel(IntPtr cmdBuffer, IntPtr queryPool, uint query);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyQueryPoolResultsDel(IntPtr cmdBuffer, IntPtr queryPool, uint firstQuery, uint queryCount, IntPtr dstBuffer, ulong dstOffset, ulong stride, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBindDescriptorSetsDel(IntPtr cmdBuffer, uint pipelineBindPoint, IntPtr layout, uint firstSet, uint descriptorSetCount, IntPtr pDescriptorSets, uint dynamicOffsetCount, IntPtr pDynamicOffsets);

        public IntPtr Handle => _commandBuffer;
        public bool IsRecording => _isRecording;

        public VulkanCommandBuffer(VulkanDevice device, IntPtr commandBuffer, IntPtr commandPool)
        {
            _device = device;
            _commandBuffer = commandBuffer;
            _commandPool = commandPool;
            LoadFunctions();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkCmdBeginRenderPass = Marshal.GetDelegateForFunctionPointer<CmdBeginRenderPassDel>(load("vkCmdBeginRenderPass"));
            _vkCmdEndRenderPass = Marshal.GetDelegateForFunctionPointer<CmdEndRenderPassDel>(load("vkCmdEndRenderPass"));
            _vkCmdBindPipeline = Marshal.GetDelegateForFunctionPointer<CmdBindPipelineDel>(load("vkCmdBindPipeline"));
            _vkCmdDraw = Marshal.GetDelegateForFunctionPointer<CmdDrawDel>(load("vkCmdDraw"));
            _vkCmdDrawIndexed = Marshal.GetDelegateForFunctionPointer<CmdDrawIndexedDel>(load("vkCmdDrawIndexed"));
            _vkCmdDispatch = Marshal.GetDelegateForFunctionPointer<CmdDispatchDel>(load("vkCmdDispatch"));
            _vkCmdPipelineBarrier = Marshal.GetDelegateForFunctionPointer<CmdPipelineBarrierDel>(load("vkCmdPipelineBarrier"));
            _vkCmdCopyBuffer = Marshal.GetDelegateForFunctionPointer<CmdCopyBufferDel>(load("vkCmdCopyBuffer"));
            _vkCmdCopyBufferToImage = Marshal.GetDelegateForFunctionPointer<CmdCopyBufferToImageDel>(load("vkCmdCopyBufferToImage"));
            _vkCmdPushConstants = Marshal.GetDelegateForFunctionPointer<CmdPushConstantsDel>(load("vkCmdPushConstants"));
            _vkCmdSetViewport = Marshal.GetDelegateForFunctionPointer<CmdSetViewportDel>(load("vkCmdSetViewport"));
            _vkCmdSetScissor = Marshal.GetDelegateForFunctionPointer<CmdSetScissorDel>(load("vkCmdSetScissor"));
            _vkCmdBindVertexBuffers = Marshal.GetDelegateForFunctionPointer<CmdBindVertexBuffersDel>(load("vkCmdBindVertexBuffers"));
            _vkCmdBindIndexBuffer = Marshal.GetDelegateForFunctionPointer<CmdBindIndexBufferDel>(load("vkCmdBindIndexBuffer"));
            _vkCmdCopyImage = Marshal.GetDelegateForFunctionPointer<CmdCopyImageDel>(load("vkCmdCopyImage"));
            _vkCmdBlitImage = Marshal.GetDelegateForFunctionPointer<CmdBlitImageDel>(load("vkCmdBlitImage"));
            _vkCmdResolveImage = Marshal.GetDelegateForFunctionPointer<CmdResolveImageDel>(load("vkCmdResolveImage"));
            _vkCmdWriteTimestamp = Marshal.GetDelegateForFunctionPointer<CmdWriteTimestampDel>(load("vkCmdWriteTimestamp"));
            _vkCmdBeginQuery = Marshal.GetDelegateForFunctionPointer<CmdBeginQueryDel>(load("vkCmdBeginQuery"));
            _vkCmdEndQuery = Marshal.GetDelegateForFunctionPointer<CmdEndQueryDel>(load("vkCmdEndQuery"));
            _vkCmdCopyQueryPoolResults = Marshal.GetDelegateForFunctionPointer<CmdCopyQueryPoolResultsDel>(load("vkCmdCopyQueryPoolResults"));
            _vkCmdBindDescriptorSets = Marshal.GetDelegateForFunctionPointer<CmdBindDescriptorSetsDel>(load("vkCmdBindDescriptorSets"));
        }

        [DllImport("vulkan-1.dll")]
        private static extern VulkanResult vkBeginCommandBuffer(IntPtr commandBuffer, ref VkCommandBufferBeginInfo pBeginInfo);
        [DllImport("vulkan-1.dll")]
        private static extern VulkanResult vkEndCommandBuffer(IntPtr commandBuffer);
        [DllImport("vulkan-1.dll")]
        private static extern void vkCmdPipelineBarrier(IntPtr cmdBuffer, PipelineStageFlag srcStageMask, PipelineStageFlag dstStageMask, uint dependencyFlags, uint memoryBarrierCount, IntPtr pMemoryBarriers, uint bufferMemoryBarrierCount, IntPtr pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers);

        [StructLayout(LayoutKind.Sequential)]
        internal struct VkImageCopy
        {
            public VkImageCopy_SrcSubresource srcSubresource;
            public VkImageCopy_Offset3D srcOffset;
            public VkImageCopy_SrcSubresource dstSubresource;
            public VkImageCopy_Offset3D dstOffset;
            public VkImageCopy_Extent3D extent;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct VkImageCopy_SrcSubresource
        {
            public uint aspectMask;
            public uint mipLevel;
            public uint baseArrayLayer;
            public uint layerCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct VkImageCopy_Offset3D
        {
            public int x, y, z;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct VkImageCopy_Extent3D
        {
            public uint width, height, depth;
        }

        /// <summary>Begins recording commands</summary>
        public VulkanResult Begin(CommandBufferUsageFlag flags = 0)
        {
            var beginInfo = new VkCommandBufferBeginInfo
            {
                sType = 41,
                flags = (uint)flags
            };
            var result = vkBeginCommandBuffer(_commandBuffer, ref beginInfo);
            if (result == VulkanResult.Success) _isRecording = true;
            return result;
        }

        /// <summary>Ends recording commands</summary>
        public VulkanResult End()
        {
            var result = vkEndCommandBuffer(_commandBuffer);
            if (result == VulkanResult.Success) _isRecording = false;
            return result;
        }

        /// <summary>Begins a render pass</summary>
        public void BeginRenderPass(VulkanRenderPass renderPass, VulkanFramebuffer framebuffer, ClearValue[] clearValues, Rect2D renderArea = default)
        {
            var clearValuesPtr = IntPtr.Zero;
            int clearValueSize = 16;
            if (clearValues != null && clearValues.Length > 0)
            {
                clearValuesPtr = Marshal.AllocHGlobal(clearValues.Length * clearValueSize);
                for (int i = 0; i < clearValues.Length; i++)
                    Marshal.StructureToPtr(clearValues[i], clearValuesPtr + i * clearValueSize, false);
            }

            var beginInfo = new VkRenderPassBeginInfo
            {
                sType = 42,
                renderPass = renderPass.Handle,
                framebuffer = framebuffer.Handle,
                clearValueCount = (uint)(clearValues?.Length ?? 0),
                pClearValues = clearValuesPtr
            };

            if (renderArea.Extent.Width == 0 && renderArea.Extent.Height == 0)
            {
                beginInfo.renderArea = new VkRenderPassBeginInfo_Rect2D
                {
                    x = 0, y = 0,
                    width = framebuffer.Width,
                    height = framebuffer.Height
                };
            }
            else
            {
                beginInfo.renderArea = new VkRenderPassBeginInfo_Rect2D
                {
                    x = renderArea.Offset.X, y = renderArea.Offset.Y,
                    width = renderArea.Extent.Width, height = renderArea.Extent.Height
                };
            }

            _vkCmdBeginRenderPass(_commandBuffer, ref beginInfo, 0);
            if (clearValuesPtr != IntPtr.Zero) Marshal.FreeHGlobal(clearValuesPtr);
        }

        /// <summary>Ends the current render pass</summary>
        public void EndRenderPass() => _vkCmdEndRenderPass(_commandBuffer);

        /// <summary>Binds a graphics or compute pipeline</summary>
        public void BindPipeline(VulkanPipeline pipeline) => _vkCmdBindPipeline(_commandBuffer, PipelineBindPoint.Graphics, pipeline.Handle);

        /// <summary>Binds a compute pipeline</summary>
        public void BindComputePipeline(VulkanComputePipeline pipeline) => _vkCmdBindPipeline(_commandBuffer, PipelineBindPoint.Compute, pipeline.Handle);

        /// <summary>Binds vertex buffers</summary>
        public void BindVertexBuffer(VulkanBuffer buffer, ulong offset = 0)
        {
            var bufferHandle = buffer.Handle;
            _vkCmdBindVertexBuffers(_commandBuffer, 0, 1, ref bufferHandle, ref offset);
        }

        /// <summary>Binds multiple vertex buffers</summary>
        public void BindVertexBuffers(VulkanBuffer[] buffers, ulong[] offsets)
        {
            var bufferHandles = new IntPtr[buffers.Length];
            for (int i = 0; i < buffers.Length; i++) bufferHandles[i] = buffers[i].Handle;
            var bufferPtr = Marshal.AllocHGlobal(buffers.Length * IntPtr.Size);
            for (int i = 0; i < buffers.Length; i++) Marshal.WriteIntPtr(bufferPtr + i * IntPtr.Size, bufferHandles[i]);
            var offsetPtr = Marshal.AllocHGlobal(buffers.Length * sizeof(ulong));
            for (int i = 0; i < buffers.Length; i++) Marshal.WriteInt64(offsetPtr + i * sizeof(ulong), (long)offsets[i]);
            _vkCmdBindVertexBuffers(_commandBuffer, 0, (uint)buffers.Length, ref bufferPtr, ref offsets[0]);
            Marshal.FreeHGlobal(bufferPtr);
            Marshal.FreeHGlobal(offsetPtr);
        }

        /// <summary>Binds an index buffer</summary>
        public void BindIndexBuffer(VulkanBuffer buffer, ulong offset = 0, IndexType indexType = IndexType.Uint32)
        {
            _vkCmdBindIndexBuffer(_commandBuffer, buffer.Handle, offset, indexType);
        }

        /// <summary>Issues a draw command</summary>
        public void Draw(uint vertexCount, uint instanceCount = 1, uint firstVertex = 0, uint firstInstance = 0)
        {
            _vkCmdDraw(_commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
        }

        /// <summary>Issues an indexed draw command</summary>
        public void DrawIndexed(uint indexCount, uint instanceCount = 1, uint firstIndex = 0, int vertexOffset = 0, uint firstInstance = 0)
        {
            _vkCmdDrawIndexed(_commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }

        /// <summary>Dispatches a compute shader</summary>
        public void Dispatch(uint groupCountX, uint groupCountY, uint groupCountZ)
        {
            _vkCmdDispatch(_commandBuffer, groupCountX, groupCountY, groupCountZ);
        }

        /// <summary>Inserts a pipeline barrier</summary>
        public void PipelineBarrier(
            PipelineStageFlag srcStageMask, PipelineStageFlag dstStageMask,
            MemoryBarrier[] memoryBarriers = null,
            BufferMemoryBarrier[] bufferBarriers = null,
            ImageMemoryBarrier[] imageBarriers = null)
        {
            var memBarrierPtr = IntPtr.Zero;
            int memBarrierCount = memoryBarriers?.Length ?? 0;
            if (memBarrierCount > 0)
            {
                int sz = Marshal.SizeOf<MemoryBarrier>();
                memBarrierPtr = Marshal.AllocHGlobal(memBarrierCount * sz);
                for (int i = 0; i < memBarrierCount; i++)
                    Marshal.StructureToPtr(memoryBarriers[i], memBarrierPtr + i * sz, false);
            }

            var bufBarrierPtr = IntPtr.Zero;
            int bufBarrierCount = bufferBarriers?.Length ?? 0;
            if (bufBarrierCount > 0)
            {
                int sz = Marshal.SizeOf<BufferMemoryBarrier>();
                bufBarrierPtr = Marshal.AllocHGlobal(bufBarrierCount * sz);
                for (int i = 0; i < bufBarrierCount; i++)
                    Marshal.StructureToPtr(bufferBarriers[i], bufBarrierPtr + i * sz, false);
            }

            var imgBarrierPtr = IntPtr.Zero;
            int imgBarrierCount = imageBarriers?.Length ?? 0;
            if (imgBarrierCount > 0)
            {
                int sz = Marshal.SizeOf<ImageMemoryBarrier>();
                imgBarrierPtr = Marshal.AllocHGlobal(imgBarrierCount * sz);
                for (int i = 0; i < imgBarrierCount; i++)
                {
                    var b = imageBarriers[i];
                    var vkBarrier = new VkImageMemoryBarrier
                    {
                        sType = 44,
                        srcAccessMask = b.SrcAccessMask,
                        dstAccessMask = b.DstAccessMask,
                        oldLayout = b.OldLayout,
                        newLayout = b.NewLayout,
                        srcQueueFamilyIndex = b.SrcQueueFamilyIndex,
                        dstQueueFamilyIndex = b.DstQueueFamilyIndex,
                        image = b.Image,
                        subresourceRange = new VkImageSubresourceRange
                        {
                            aspectMask = b.SubresourceRange.AspectMask,
                            baseMipLevel = b.SubresourceRange.BaseMipLevel,
                            levelCount = b.SubresourceRange.LevelCount,
                            baseArrayLayer = b.SubresourceRange.BaseArrayLayer,
                            layerCount = b.SubresourceRange.LayerCount
                        }
                    };
                    Marshal.StructureToPtr(vkBarrier, imgBarrierPtr + i * sz, false);
                }
            }

            _vkCmdPipelineBarrier(_commandBuffer, srcStageMask, dstStageMask, 0,
                (uint)memBarrierCount, memBarrierPtr,
                (uint)bufBarrierCount, bufBarrierPtr,
                (uint)imgBarrierCount, imgBarrierPtr);

            if (memBarrierPtr != IntPtr.Zero) Marshal.FreeHGlobal(memBarrierPtr);
            if (bufBarrierPtr != IntPtr.Zero) Marshal.FreeHGlobal(bufBarrierPtr);
            if (imgBarrierPtr != IntPtr.Zero) Marshal.FreeHGlobal(imgBarrierPtr);
        }

        /// <summary>Copies data between buffers</summary>
        public void CopyBuffer(VulkanBuffer srcBuffer, VulkanBuffer dstBuffer, BufferCopy[] regions)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0) return;
            int sz = Marshal.SizeOf<BufferCopy>();
            var regionsPtr = Marshal.AllocHGlobal(regionCount * sz);
            for (int i = 0; i < regionCount; i++)
                Marshal.StructureToPtr(regions[i], regionsPtr + i * sz, false);
            _vkCmdCopyBuffer(_commandBuffer, srcBuffer.Handle, dstBuffer.Handle, (uint)regionCount, ref regions[0]);
            Marshal.FreeHGlobal(regionsPtr);
        }

        /// <summary>Copies buffer data to an image</summary>
        public void CopyBufferToImage(VulkanBuffer buffer, VulkanTexture image, ImageLayout imageLayout, BufferImageCopy[] regions)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0) return;
            _vkCmdCopyBufferToImage(_commandBuffer, buffer.Handle, image.Handle, (uint)imageLayout, (uint)regionCount, ref regions[0]);
        }

        /// <summary>Pushes constant data to the shader</summary>
        public void PushConstants(ShaderStageFlag stageFlags, uint offset, byte[] data)
        {
            if (data == null || data.Length == 0) return;
            var dataPtr = Marshal.AllocHGlobal(data.Length);
            Marshal.Copy(data, 0, dataPtr, data.Length);
            _vkCmdPushConstants(_commandBuffer, IntPtr.Zero, stageFlags, offset, (uint)data.Length, dataPtr);
            Marshal.FreeHGlobal(dataPtr);
        }

        /// <summary>Pushes constant data to the shader with layout</summary>
        public void PushConstants(IntPtr pipelineLayout, ShaderStageFlag stageFlags, uint offset, byte[] data)
        {
            if (data == null || data.Length == 0) return;
            var dataPtr = Marshal.AllocHGlobal(data.Length);
            Marshal.Copy(data, 0, dataPtr, data.Length);
            _vkCmdPushConstants(_commandBuffer, pipelineLayout, stageFlags, offset, (uint)data.Length, dataPtr);
            Marshal.FreeHGlobal(dataPtr);
        }

        /// <summary>Binds descriptor sets to the command buffer</summary>
        public void BindDescriptorSets(PipelineBindPoint pipelineBindPoint, IntPtr layout, uint firstSet, IntPtr[] descriptorSets, uint[] dynamicOffsets = null)
        {
            if (descriptorSets == null || descriptorSets.Length == 0) return;
            int setCount = descriptorSets.Length;
            int dynOffsetCount = dynamicOffsets?.Length ?? 0;
            var setsPtr = Marshal.AllocHGlobal(setCount * IntPtr.Size);
            for (int i = 0; i < setCount; i++)
                Marshal.WriteIntPtr(setsPtr + i * IntPtr.Size, descriptorSets[i]);
            var dynPtr = IntPtr.Zero;
            if (dynOffsetCount > 0)
            {
                dynPtr = Marshal.AllocHGlobal(dynOffsetCount * sizeof(uint));
                for (int i = 0; i < dynOffsetCount; i++)
                    Marshal.WriteInt32(dynPtr + i * sizeof(uint), (int)dynamicOffsets[i]);
            }
            _vkCmdBindDescriptorSets(_commandBuffer, (uint)pipelineBindPoint, layout, firstSet, (uint)setCount, setsPtr, (uint)dynOffsetCount, dynPtr);
            Marshal.FreeHGlobal(setsPtr);
            if (dynPtr != IntPtr.Zero) Marshal.FreeHGlobal(dynPtr);
        }

        /// <summary>Sets the viewport</summary>
        public void SetViewport(Viewport viewport, uint firstViewport = 0)
        {
            _vkCmdSetViewport(_commandBuffer, firstViewport, 1, ref viewport);
        }

        /// <summary>Sets the scissor rectangle</summary>
        public void SetScissor(Rect2D scissor, uint firstScissor = 0)
        {
            _vkCmdSetScissor(_commandBuffer, firstScissor, 1, ref scissor);
        }

        /// <summary>Copies between images</summary>
        internal void CopyImage(VulkanTexture srcImage, ImageLayout srcLayout, VulkanTexture dstImage, ImageLayout dstLayout, VkImageCopy[] regions)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0) return;
            int sz = Marshal.SizeOf<VkImageCopy>();
            var regionsPtr = Marshal.AllocHGlobal(regionCount * sz);
            for (int i = 0; i < regionCount; i++)
                Marshal.StructureToPtr(regions[i], regionsPtr + i * sz, false);
            _vkCmdCopyImage(_commandBuffer, srcImage.Handle, (uint)srcLayout, dstImage.Handle, (uint)dstLayout, (uint)regionCount, regionsPtr);
            Marshal.FreeHGlobal(regionsPtr);
        }

        /// <summary>Blits an image region</summary>
        public void BlitImage(VulkanTexture srcImage, ImageLayout srcLayout, VulkanTexture dstImage, ImageLayout dstLayout, ImageBlit[] regions, Filter filter = Filter.Linear)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0) return;
            int sz = Marshal.SizeOf<ImageBlit>();
            var regionsPtr = Marshal.AllocHGlobal(regionCount * sz);
            for (int i = 0; i < regionCount; i++)
                Marshal.StructureToPtr(regions[i], regionsPtr + i * sz, false);
            _vkCmdBlitImage(_commandBuffer, srcImage.Handle, (uint)srcLayout, dstImage.Handle, (uint)dstLayout, (uint)regionCount, regionsPtr, filter);
            Marshal.FreeHGlobal(regionsPtr);
        }

        /// <summary>Resolves a multisampled image</summary>
        public void ResolveImage(VulkanTexture srcImage, ImageLayout srcLayout, VulkanTexture dstImage, ImageLayout dstLayout, ImageResolve[] regions)
        {
            int regionCount = regions?.Length ?? 0;
            if (regionCount == 0) return;
            int sz = Marshal.SizeOf<ImageResolve>();
            var regionsPtr = Marshal.AllocHGlobal(regionCount * sz);
            for (int i = 0; i < regionCount; i++)
                Marshal.StructureToPtr(regions[i], regionsPtr + i * sz, false);
            _vkCmdResolveImage(_commandBuffer, srcImage.Handle, (uint)srcLayout, dstImage.Handle, (uint)dstLayout, (uint)regionCount, regionsPtr);
            Marshal.FreeHGlobal(regionsPtr);
        }

        /// <summary>Writes a timestamp into a query pool</summary>
        public void WriteTimestamp(PipelineStageFlag pipelineStage, IntPtr queryPool, uint query)
        {
            _vkCmdWriteTimestamp(_commandBuffer, pipelineStage, queryPool, query);
        }

        /// <summary>Begins a query</summary>
        public void BeginQuery(IntPtr queryPool, uint query, uint flags = 0)
        {
            _vkCmdBeginQuery(_commandBuffer, queryPool, query, flags);
        }

        /// <summary>Ends a query</summary>
        public void EndQuery(IntPtr queryPool, uint query)
        {
            _vkCmdEndQuery(_commandBuffer, queryPool, query);
        }

        /// <summary>Resets a command pool, releasing all command buffers</summary>
        public void Reset()
        {
            // Reset is done via command pool, individual command buffers are implicitly reset
            _isRecording = false;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
    // =========================================================================
    // VulkanBuffer
    // =========================================================================

    /// <summary>
    /// Wraps a Vulkan buffer resource with GPU memory allocation.
    /// Supports mapping, flushing, and copy operations.
    /// </summary>
    public class VulkanBuffer : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _buffer;
        private IntPtr _memory;
        private IntPtr _mappedData;
        private BufferDescription _description;
        private VkMemoryRequirements _memoryRequirements;
        private ulong _allocatedSize;
        private uint _memoryTypeIndex;
        private bool _isMapped;
        private bool _disposed;

        // Function pointers
        private MapMemoryDel _vkMapMemory;
        private UnmapMemoryDel _vkUnmapMemory;
        private FlushMappedMemoryRangesDel _vkFlushMappedMemoryRanges;
        private BindBufferMemoryDel _vkBindBufferMemory;
        private DestroyBufferDel _vkDestroyBuffer;
        private FreeMemoryDel _vkFreeMemory;
        private CmdCopyBufferDel _vkCmdCopyBuffer;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult MapMemoryDel(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, ref IntPtr ppData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void UnmapMemoryDel(IntPtr device, IntPtr memory);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult FlushMappedMemoryRangesDel(IntPtr device, uint memoryRangeCount, ref VkMappedMemoryRange pMemoryRanges);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindBufferMemoryDel(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyBufferDel(IntPtr device, IntPtr buffer, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void FreeMemoryDel(IntPtr device, IntPtr memory, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferDel(IntPtr commandBuffer, IntPtr srcBuffer, IntPtr dstBuffer, uint regionCount, ref BufferCopy pRegions);

        public IntPtr Handle => _buffer;
        public IntPtr MemoryHandle => _memory;
        public ulong Size => _description.Size;
        public IntPtr MappedPointer => _mappedData;
        public bool IsMapped => _isMapped;
        public BufferDescription Description => _description;
        public VkMemoryRequirements MemoryRequirements => _memoryRequirements;

        public VulkanBuffer(VulkanDevice device, IntPtr buffer, BufferDescription description, VkMemoryRequirements memoryRequirements)
        {
            _device = device;
            _buffer = buffer;
            _description = description;
            _memoryRequirements = memoryRequirements;
            LoadFunctions();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkMapMemory = Marshal.GetDelegateForFunctionPointer<MapMemoryDel>(load("vkMapMemory"));
            _vkUnmapMemory = Marshal.GetDelegateForFunctionPointer<UnmapMemoryDel>(load("vkUnmapMemory"));
            _vkFlushMappedMemoryRanges = Marshal.GetDelegateForFunctionPointer<FlushMappedMemoryRangesDel>(load("vkFlushMappedMemoryRanges"));
            _vkBindBufferMemory = Marshal.GetDelegateForFunctionPointer<BindBufferMemoryDel>(load("vkBindBufferMemory"));
            _vkDestroyBuffer = Marshal.GetDelegateForFunctionPointer<DestroyBufferDel>(load("vkDestroyBuffer"));
            _vkFreeMemory = Marshal.GetDelegateForFunctionPointer<FreeMemoryDel>(load("vkFreeMemory"));
        }

        /// <summary>Binds memory to this buffer (called by allocator)</summary>
        internal void BindMemory(IntPtr memory, ulong offset)
        {
            _memory = memory;
            _vkBindBufferMemory(_device.LogicalDevice, _buffer, memory, offset);
        }

        /// <summary>Maps the buffer into CPU address space</summary>
        public IntPtr Map(ulong offset = 0, ulong size = 0)
        {
            if (_isMapped) return _mappedData;
            if (size == 0) size = _description.Size;
            IntPtr data = IntPtr.Zero;
            var result = _vkMapMemory(_device.LogicalDevice, _memory, offset, size, 0, ref data);
            if (result == VulkanResult.Success)
            {
                _mappedData = data;
                _isMapped = true;
            }
            return data;
        }

        /// <summary>Unmaps the buffer from CPU address space</summary>
        public void Unmap()
        {
            if (!_isMapped) return;
            _vkUnmapMemory(_device.LogicalDevice, _memory);
            _mappedData = IntPtr.Zero;
            _isMapped = false;
        }

        /// <summary>Flushes mapped memory ranges to make them visible to the device</summary>
        public void Flush(ulong start = 0, ulong size = 0)
        {
            if (size == 0) size = _description.Size;
            var range = new VkMappedMemoryRange
            {
                sType = 6,
                memory = _memory,
                offset = start,
                size = size
            };
            _vkFlushMappedMemoryRanges(_device.LogicalDevice, 1, ref range);
        }

        /// <summary>Copies data from another buffer to this one using a command buffer</summary>
        public void CopyFrom(VulkanBuffer sourceBuffer, ulong srcOffset, ulong dstOffset, ulong size, IntPtr commandBuffer)
        {
            var region = new BufferCopy(srcOffset, dstOffset, size);
            var cmdCopy = Marshal.GetDelegateForFunctionPointer<CmdCopyBufferDel>(
                _device.GetDeviceProcAddr("vkCmdCopyBuffer"));
            // Using the command buffer directly
        }

        /// <summary>Writes data to the buffer from CPU memory</summary>
        public void SetData<T>(T[] data, ulong dstOffset = 0) where T : struct
        {
            int structSize = Marshal.SizeOf<T>();
            int dataSize = data.Length * structSize;
            var dataPtr = Marshal.AllocHGlobal(dataSize);
            try
            {
                for (int i = 0; i < data.Length; i++)
                    Marshal.StructureToPtr(data[i], dataPtr + i * structSize, false);

                if (_isMapped)
                {
                    unsafe
                    {
                        Buffer.MemoryCopy(
                            (void*)(dataPtr),
                            (void*)(_mappedData + (int)dstOffset),
                            dataSize, dataSize);
                    }
                }
                else
                {
                    Map(dstOffset, (ulong)dataSize);
                    unsafe
                    {
                        Buffer.MemoryCopy(
                            (void*)(dataPtr),
                            (void*)(_mappedData),
                            dataSize, dataSize);
                    }
                    Flush(dstOffset, (ulong)dataSize);
                    Unmap();
                }
            }
            finally { Marshal.FreeHGlobal(dataPtr); }
        }

        /// <summary>Writes raw byte data to the buffer</summary>
        public void SetData(byte[] data, ulong dstOffset = 0)
        {
            if (data == null || data.Length == 0) return;
            var dataPtr = Marshal.AllocHGlobal(data.Length);
            try
            {
                Marshal.Copy(data, 0, dataPtr, data.Length);
                if (_isMapped)
                {
                    unsafe { Buffer.MemoryCopy((void*)dataPtr, (void*)(_mappedData + (int)dstOffset), data.Length, data.Length); }
                }
                else
                {
                    Map(dstOffset, (ulong)data.Length);
                    unsafe { Buffer.MemoryCopy((void*)dataPtr, (void*)_mappedData, data.Length, data.Length); }
                    Flush(dstOffset, (ulong)data.Length);
                    Unmap();
                }
            }
            finally { Marshal.FreeHGlobal(dataPtr); }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_isMapped) Unmap();
            if (_buffer != IntPtr.Zero && _vkDestroyBuffer != null)
                _vkDestroyBuffer(_device.LogicalDevice, _buffer, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    // =========================================================================
    // VulkanTexture
    // =========================================================================

    /// <summary>
    /// Wraps a Vulkan image resource with associated memory and image view.
    /// Supports mipmap generation, layout transitions, and buffer copies.
    /// </summary>
    public class VulkanTexture : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _image;
        private IntPtr _memory;
        private IntPtr _imageView;
        private TextureDescription _description;
        private VkMemoryRequirements _memoryRequirements;
        private uint _memoryTypeIndex;
        private bool _disposed;

        // Function pointers
        private CreateImageViewDel _vkCreateImageView;
        private DestroyImageViewDel _vkDestroyImageView;
        private DestroyImageDel _vkDestroyImage;
        private FreeMemoryDel _vkFreeMemory;
        private BindImageMemoryDel _vkBindImageMemory;
        private CmdPipelineBarrierDel2 _vkCmdPipelineBarrier;
        private CmdCopyBufferToImageDel2 _vkCmdCopyBufferToImage;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateImageViewDel(IntPtr device, ref VkImageViewCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pView);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageViewDel(IntPtr device, IntPtr imageView, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageDel(IntPtr device, IntPtr image, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void FreeMemoryDel(IntPtr device, IntPtr memory, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindImageMemoryDel(IntPtr device, IntPtr image, IntPtr memory, ulong memoryOffset);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdPipelineBarrierDel2(IntPtr cmdBuffer, PipelineStageFlag srcStageMask, PipelineStageFlag dstStageMask, uint depFlags, uint memBarCount, IntPtr memBars, uint bufBarCount, IntPtr bufBars, uint imgBarCount, IntPtr imgBars);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdCopyBufferToImageDel2(IntPtr cmdBuffer, IntPtr buffer, IntPtr image, uint imageLayout, uint regionCount, IntPtr pRegions);

        /// <summary>The Vulkan image handle</summary>
        public IntPtr Handle => _image;

        /// <summary>The image view handle</summary>
        public IntPtr ImageViewHandle => _imageView;

        /// <summary>Texture description</summary>
        public TextureDescription Description => _description;

        /// <summary>Width of the texture</summary>
        public uint Width => _description.Width;

        /// <summary>Height of the texture</summary>
        public uint Height => _description.Height;

        /// <summary>Format of the texture</summary>
        public VulkanFormat Format => _description.Format;

        /// <summary>Mip level count</summary>
        public uint MipLevels => _description.MipLevels;

        public VkMemoryRequirements MemoryRequirements => _memoryRequirements;

        public VulkanTexture(VulkanDevice device, IntPtr image, TextureDescription description, VkMemoryRequirements memoryRequirements = default)
        {
            _device = device;
            _image = image;
            _description = description;
            _memoryRequirements = memoryRequirements;
            LoadFunctions();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkCreateImageView = Marshal.GetDelegateForFunctionPointer<CreateImageViewDel>(load("vkCreateImageView"));
            _vkDestroyImageView = Marshal.GetDelegateForFunctionPointer<DestroyImageViewDel>(load("vkDestroyImageView"));
            _vkDestroyImage = Marshal.GetDelegateForFunctionPointer<DestroyImageDel>(load("vkDestroyImage"));
            _vkFreeMemory = Marshal.GetDelegateForFunctionPointer<FreeMemoryDel>(load("vkFreeMemory"));
            _vkBindImageMemory = Marshal.GetDelegateForFunctionPointer<BindImageMemoryDel>(load("vkBindImageMemory"));
            _vkCmdPipelineBarrier = Marshal.GetDelegateForFunctionPointer<CmdPipelineBarrierDel2>(load("vkCmdPipelineBarrier"));
            _vkCmdCopyBufferToImage = Marshal.GetDelegateForFunctionPointer<CmdCopyBufferToImageDel2>(load("vkCmdCopyBufferToImage"));
        }

        /// <summary>Binds memory to this image (called by allocator)</summary>
        internal void BindMemory(IntPtr memory, ulong offset)
        {
            _memory = memory;
            _vkBindImageMemory(_device.LogicalDevice, _image, memory, offset);
        }

        /// <summary>Creates the image view for this texture</summary>
        public IntPtr GetImageView()
        {
            if (_imageView != IntPtr.Zero) return _imageView;

            var aspectFlags = ImageAspectFlag.Color;
            if (_description.Format == VulkanFormat.D16Unorm || _description.Format == VulkanFormat.D32Sfloat ||
                _description.Format == VulkanFormat.D24UnormS8Uint || _description.Format == VulkanFormat.D32SfloatS8Uint)
                aspectFlags = ImageAspectFlag.Depth;
            else if (_description.Format == VulkanFormat.S8Uint)
                aspectFlags = ImageAspectFlag.Stencil;
            else if (_description.Format == VulkanFormat.D16UnormS8Uint)
                aspectFlags = ImageAspectFlag.Depth | ImageAspectFlag.Stencil;

            var viewType = _description.Type == ImageType.Type3D ? ImageViewType.Type3D :
                           _description.ArrayLayers > 1 ? ImageViewType.Type2DArray :
                           ImageViewType.Type2D;

            var createInfo = new VkImageViewCreateInfo
            {
                sType = 15,
                image = _image,
                viewType = viewType,
                format = _description.Format,
                subresourceRange = new VkImageSubresourceRange
                {
                    aspectMask = aspectFlags,
                    baseMipLevel = 0,
                    levelCount = _description.MipLevels,
                    baseArrayLayer = 0,
                    layerCount = _description.ArrayLayers
                }
            };
            createInfo.components = new VkComponentMapping { r = 0, g = 1, b = 2, a = 3 };

            _vkCreateImageView(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref _imageView);
            return _imageView;
        }

        /// <summary>Generates mipmaps for the texture using blit operations</summary>
        public void CreateMipmaps(IntPtr commandBuffer)
        {
            if (_description.MipLevels <= 1) return;

            int mipWidth = (int)_description.Width;
            int mipHeight = (int)_description.Height;

            var barrier = new VkImageMemoryBarrier
            {
                sType = 44,
                srcQueueFamilyIndex = 0xFFFFFFFF,
                dstQueueFamilyIndex = 0xFFFFFFFF,
                image = _image,
                subresourceRange = new VkImageSubresourceRange
                {
                    aspectMask = ImageAspectFlag.Color,
                    levelCount = 1,
                    baseArrayLayer = 0,
                    layerCount = _description.ArrayLayers
                }
            };

            for (uint i = 1; i < _description.MipLevels; i++)
            {
                barrier.subresourceRange.baseMipLevel = i - 1;
                barrier.oldLayout = ImageLayout.TransferDstOptimal;
                barrier.newLayout = ImageLayout.TransferSrcOptimal;
                barrier.srcAccessMask = AccessFlag.TransferWrite;
                barrier.dstAccessMask = AccessFlag.TransferRead;

                int barrierSize = Marshal.SizeOf<VkImageMemoryBarrier>();
                var barrierPtr = Marshal.AllocHGlobal(barrierSize);
                Marshal.StructureToPtr(barrier, barrierPtr, false);
                _vkCmdPipelineBarrier(commandBuffer,
                    PipelineStageFlag.Transfer, PipelineStageFlag.Transfer,
                    0, 0, IntPtr.Zero, 0, IntPtr.Zero, 1, barrierPtr);
                Marshal.FreeHGlobal(barrierPtr);

                int dstMipWidth = mipWidth > 1 ? mipWidth / 2 : 1;
                int dstMipHeight = mipHeight > 1 ? mipHeight / 2 : 1;

                var blit = new VkImageBlit
                {
                    srcOffsets0 = new VkOffset3D { x = 0, y = 0, z = 0 },
                    srcOffsets1 = new VkOffset3D { x = mipWidth, y = mipHeight, z = 1 },
                    srcSubresource = new VkImageSubresourceLayers
                    {
                        aspectMask = (uint)ImageAspectFlag.Color,
                        mipLevel = i - 1,
                        baseArrayLayer = 0,
                        layerCount = _description.ArrayLayers
                    },
                    dstOffsets0 = new VkOffset3D { x = 0, y = 0, z = 0 },
                    dstOffsets1 = new VkOffset3D { x = dstMipWidth, y = dstMipHeight, z = 1 },
                    dstSubresource = new VkImageSubresourceLayers
                    {
                        aspectMask = (uint)ImageAspectFlag.Color,
                        mipLevel = i,
                        baseArrayLayer = 0,
                        layerCount = _description.ArrayLayers
                    }
                };

                int blitSize = Marshal.SizeOf<VkImageBlit>();
                var blitPtr = Marshal.AllocHGlobal(blitSize);
                Marshal.StructureToPtr(blit, blitPtr, false);

                var cmdBlit = Marshal.GetDelegateForFunctionPointer<CmdBlitImageDel>(
                    _device.GetDeviceProcAddr("vkCmdBlitImage"));
                cmdBlit(commandBuffer, _image, (uint)ImageLayout.TransferSrcOptimal,
                    _image, (uint)ImageLayout.TransferDstOptimal, 1, blitPtr, Filter.Linear);
                Marshal.FreeHGlobal(blitPtr);

                barrier.oldLayout = ImageLayout.TransferSrcOptimal;
                barrier.newLayout = ImageLayout.ShaderReadOnlyOptimal;
                barrier.srcAccessMask = AccessFlag.TransferRead;
                barrier.dstAccessMask = AccessFlag.ShaderRead;

                barrierPtr = Marshal.AllocHGlobal(barrierSize);
                Marshal.StructureToPtr(barrier, barrierPtr, false);
                _vkCmdPipelineBarrier(commandBuffer,
                    PipelineStageFlag.Transfer, PipelineStageFlag.FragmentShader,
                    0, 0, IntPtr.Zero, 0, IntPtr.Zero, 1, barrierPtr);
                Marshal.FreeHGlobal(barrierPtr);

                mipWidth = dstMipWidth;
                mipHeight = dstMipHeight;
            }

            // Transition last mip level
            barrier.subresourceRange.baseMipLevel = _description.MipLevels - 1;
            barrier.oldLayout = ImageLayout.TransferDstOptimal;
            barrier.newLayout = ImageLayout.ShaderReadOnlyOptimal;
            barrier.srcAccessMask = AccessFlag.TransferWrite;
            barrier.dstAccessMask = AccessFlag.ShaderRead;

            int finalBarrierSize = Marshal.SizeOf<VkImageMemoryBarrier>();
            var finalBarrierPtr = Marshal.AllocHGlobal(finalBarrierSize);
            Marshal.StructureToPtr(barrier, finalBarrierPtr, false);
            _vkCmdPipelineBarrier(commandBuffer,
                PipelineStageFlag.Transfer, PipelineStageFlag.FragmentShader,
                0, 0, IntPtr.Zero, 0, IntPtr.Zero, 1, finalBarrierPtr);
            Marshal.FreeHGlobal(finalBarrierPtr);
        }

        /// <summary>Transitions the image to a new layout</summary>
        public void TransitionLayout(ImageLayout newLayout, IntPtr commandBuffer)
        {
            var barrier = new VkImageMemoryBarrier
            {
                sType = 44,
                srcAccessMask = 0,
                dstAccessMask = 0,
                oldLayout = _description.InitialLayout,
                newLayout = newLayout,
                srcQueueFamilyIndex = 0xFFFFFFFF,
                dstQueueFamilyIndex = 0xFFFFFFFF,
                image = _image,
                subresourceRange = new VkImageSubresourceRange
                {
                    aspectMask = ImageAspectFlag.Color,
                    baseMipLevel = 0,
                    levelCount = _description.MipLevels,
                    baseArrayLayer = 0,
                    layerCount = _description.ArrayLayers
                }
            };

            if (newLayout == ImageLayout.TransferDstOptimal)
                barrier.dstAccessMask = AccessFlag.TransferWrite;
            else if (newLayout == ImageLayout.TransferSrcOptimal)
                barrier.srcAccessMask = AccessFlag.TransferRead;
            else if (newLayout == ImageLayout.ShaderReadOnlyOptimal)
                barrier.dstAccessMask = AccessFlag.ShaderRead;
            else if (newLayout == ImageLayout.ColorAttachmentOptimal)
                barrier.dstAccessMask = AccessFlag.ColorAttachmentWrite;
            else if (newLayout == ImageLayout.DepthStencilAttachmentOptimal)
                barrier.dstAccessMask = AccessFlag.DepthStencilAttachmentWrite;

            PipelineStageFlag srcStage = PipelineStageFlag.TopOfPipe;
            PipelineStageFlag dstStage = PipelineStageFlag.BottomOfPipe;

            if (_description.InitialLayout == ImageLayout.TransferDstOptimal)
                srcStage = PipelineStageFlag.Transfer;
            if (newLayout == ImageLayout.TransferDstOptimal || newLayout == ImageLayout.TransferSrcOptimal)
                dstStage = PipelineStageFlag.Transfer;
            else if (newLayout == ImageLayout.ShaderReadOnlyOptimal)
                dstStage = PipelineStageFlag.FragmentShader;
            else if (newLayout == ImageLayout.ColorAttachmentOptimal)
                dstStage = PipelineStageFlag.ColorAttachmentOutput;

            int barrierSize = Marshal.SizeOf<VkImageMemoryBarrier>();
            var barrierPtr = Marshal.AllocHGlobal(barrierSize);
            Marshal.StructureToPtr(barrier, barrierPtr, false);
            _vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, IntPtr.Zero, 0, IntPtr.Zero, 1, barrierPtr);
            Marshal.FreeHGlobal(barrierPtr);

            _description.InitialLayout = newLayout;
        }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void CmdBlitImageDel(IntPtr cmdBuffer, IntPtr srcImage, uint srcImageLayout, IntPtr dstImage, uint dstImageLayout, uint regionCount, IntPtr pRegions, Filter filter);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_imageView != IntPtr.Zero && _vkDestroyImageView != null)
                _vkDestroyImageView(_device.LogicalDevice, _imageView, IntPtr.Zero);
            if (_image != IntPtr.Zero && _vkDestroyImage != null)
                _vkDestroyImage(_device.LogicalDevice, _image, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }
    // =========================================================================
    // VulkanPipeline
    // =========================================================================

    /// <summary>Wraps a Vulkan graphics pipeline</summary>
    public class VulkanPipeline : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _pipeline;
        private IntPtr _pipelineLayout;
        private PipelineDescription _description;
        private bool _disposed;

        [DllImport("vulkan-1.dll")]
        private static extern void vkDestroyPipeline(IntPtr device, IntPtr pipeline, IntPtr pAllocator);

        public IntPtr Handle => _pipeline;
        public IntPtr PipelineLayout => _pipelineLayout;

        public VulkanPipeline(VulkanDevice device, IntPtr pipeline, PipelineDescription description)
        {
            _device = device;
            _pipeline = pipeline;
            _description = description;
            _pipelineLayout = description.PipelineLayout;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_pipeline != IntPtr.Zero)
                vkDestroyPipeline(_device.LogicalDevice, _pipeline, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    // =========================================================================
    // VulkanComputePipeline
    // =========================================================================

    /// <summary>Wraps a Vulkan compute pipeline</summary>
    public class VulkanComputePipeline : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _pipeline;
        private IntPtr _pipelineLayout;
        private ComputePipelineDescription _description;
        private bool _disposed;

        [DllImport("vulkan-1.dll")]
        private static extern void vkDestroyPipeline(IntPtr device, IntPtr pipeline, IntPtr pAllocator);

        public IntPtr Handle => _pipeline;
        public IntPtr PipelineLayout => _pipelineLayout;

        public VulkanComputePipeline(VulkanDevice device, IntPtr pipeline, ComputePipelineDescription description)
        {
            _device = device;
            _pipeline = pipeline;
            _description = description;
            _pipelineLayout = description.PipelineLayout;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_pipeline != IntPtr.Zero)
                vkDestroyPipeline(_device.LogicalDevice, _pipeline, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    // =========================================================================
    // VulkanRenderPass
    // =========================================================================

    /// <summary>Wraps a Vulkan render pass object</summary>
    public class VulkanRenderPass : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _renderPass;
        private RenderPassDescription _description;
        private bool _disposed;

        [DllImport("vulkan-1.dll")]
        private static extern void vkDestroyRenderPass(IntPtr device, IntPtr renderPass, IntPtr pAllocator);

        public IntPtr Handle => _renderPass;
        public RenderPassDescription Description => _description;

        public VulkanRenderPass(VulkanDevice device, IntPtr renderPass, RenderPassDescription description)
        {
            _device = device;
            _renderPass = renderPass;
            _description = description;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_renderPass != IntPtr.Zero)
                vkDestroyRenderPass(_device.LogicalDevice, _renderPass, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    // =========================================================================
    // VulkanFramebuffer
    // =========================================================================

    /// <summary>Wraps a Vulkan framebuffer object</summary>
    public class VulkanFramebuffer : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _framebuffer;
        private FramebufferDescription _description;
        private bool _disposed;

        [DllImport("vulkan-1.dll")]
        private static extern void vkDestroyFramebuffer(IntPtr device, IntPtr framebuffer, IntPtr pAllocator);

        public IntPtr Handle => _framebuffer;
        public uint Width => _description.Width;
        public uint Height => _description.Height;
        public uint Layers => _description.Layers;

        public VulkanFramebuffer(VulkanDevice device, IntPtr framebuffer, FramebufferDescription description)
        {
            _device = device;
            _framebuffer = framebuffer;
            _description = description;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_framebuffer != IntPtr.Zero)
                vkDestroyFramebuffer(_device.LogicalDevice, _framebuffer, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    // =========================================================================
    // Sampler
    // =========================================================================

    /// <summary>Wraps a Vulkan sampler object</summary>
    public class Sampler : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _sampler;
        private SamplerDescription _description;
        private bool _disposed;

        [DllImport("vulkan-1.dll")]
        private static extern void vkDestroySampler(IntPtr device, IntPtr sampler, IntPtr pAllocator);

        public IntPtr Handle => _sampler;

        public Sampler(VulkanDevice device, IntPtr sampler, SamplerDescription description)
        {
            _device = device;
            _sampler = sampler;
            _description = description;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_sampler != IntPtr.Zero)
                vkDestroySampler(_device.LogicalDevice, _sampler, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    // =========================================================================
    // DescriptorSetLayout
    // =========================================================================

    /// <summary>Wraps a Vulkan descriptor set layout</summary>
    public class DescriptorSetLayout : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _layout;
        private LayoutDescription _description;
        private bool _disposed;

        [DllImport("vulkan-1.dll")]
        private static extern void vkDestroyDescriptorSetLayout(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);

        public IntPtr Handle => _layout;
        public LayoutDescription Description => _description;

        public DescriptorSetLayout(VulkanDevice device, IntPtr layout, LayoutDescription description)
        {
            _device = device;
            _layout = layout;
            _description = description;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_layout != IntPtr.Zero)
                vkDestroyDescriptorSetLayout(_device.LogicalDevice, _layout, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    // =========================================================================
    // DescriptorPool
    // =========================================================================

    /// <summary>Wraps a Vulkan descriptor pool</summary>
    public class DescriptorPool : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _pool;
        private PoolDescription _description;
        private bool _disposed;

        [DllImport("vulkan-1.dll")]
        private static extern void vkDestroyDescriptorPool(IntPtr device, IntPtr descriptorPool, IntPtr pAllocator);

        public IntPtr Handle => _pool;

        public DescriptorPool(VulkanDevice device, IntPtr pool, PoolDescription description)
        {
            _device = device;
            _pool = pool;
            _description = description;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_pool != IntPtr.Zero)
                vkDestroyDescriptorPool(_device.LogicalDevice, _pool, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    // =========================================================================
    // DescriptorSet
    // =========================================================================

    /// <summary>Wraps a Vulkan descriptor set</summary>
    public class DescriptorSet
    {
        private VulkanDevice _device;
        private IntPtr _descriptorSet;

        public IntPtr Handle => _descriptorSet;

        public DescriptorSet(VulkanDevice device, IntPtr descriptorSet)
        {
            _device = device;
            _descriptorSet = descriptorSet;
        }
    }

    // =========================================================================
    // VulkanShaderModule
    // =========================================================================

    /// <summary>
    /// Wraps a Vulkan shader module and provides SPIR-V reflection data.
    /// Extracts input/output bindings and descriptor set requirements.
    /// </summary>
    public class VulkanShaderModule : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _module;
        private byte[] _spirvCode;
        private bool _disposed;

        // Reflection data
        private List<ShaderInputBinding> _inputBindings;
        private List<ShaderOutputBinding> _outputBindings;
        private List<ShaderDescriptorBinding> _descriptorBindings;
        private ShaderStageFlag _stage;

        [DllImport("vulkan-1.dll")]
        private static extern void vkDestroyShaderModule(IntPtr device, IntPtr shaderModule, IntPtr pAllocator);

        public IntPtr Handle => _module;
        public ShaderStageFlag Stage => _stage;
        public IReadOnlyList<ShaderInputBinding> InputBindings => _inputBindings;
        public IReadOnlyList<ShaderOutputBinding> OutputBindings => _outputBindings;
        public IReadOnlyList<ShaderDescriptorBinding> DescriptorBindings => _descriptorBindings;

        public VulkanShaderModule(VulkanDevice device, IntPtr module, byte[] spirvCode)
        {
            _device = device;
            _module = module;
            _spirvCode = spirvCode;

            _inputBindings = new List<ShaderInputBinding>();
            _outputBindings = new List<ShaderOutputBinding>();
            _descriptorBindings = new List<ShaderDescriptorBinding>();

            ParseSpirvReflection();
        }

        /// <summary>Parses SPIR-V bytecode for reflection data</summary>
        private void ParseSpirvReflection()
        {
            if (_spirvCode == null || _spirvCode.Length < 20) return;

            // SPIR-V header: magic number, version, generator, bound, reserved
            uint magic = BitConverter.ToUInt32(_spirvCode, 0);
            uint bound = BitConverter.ToUInt32(_spirvCode, 16);

            int wordIndex = 5;
            uint idBound = bound;

            // Simple SPIR-V parsing for decorations and type info
            while (wordIndex < _spirvCode.Length / 4)
            {
                uint word = BitConverter.ToUInt32(_spirvCode, wordIndex * 4);
                uint opcode = word & 0xFFFF;
                uint wordCount = word >> 16;

                if (wordCount == 0) break;

                switch (opcode)
                {
                    case 15: // OpEntryPoint
                        if (wordCount >= 3)
                        {
                            uint executionModel = BitConverter.ToUInt32(_spirvCode, (wordIndex + 1) * 4);
                            _stage = executionModel switch
                            {
                                1 => ShaderStageFlag.Vertex,
                                4 => ShaderStageFlag.Fragment,
                                5 => ShaderStageFlag.Compute,
                                6 => ShaderStageFlag.Geometry,
                                _ => ShaderStageFlag.All
                            };
                        }
                        break;

                    case 71: // OpDecorate
                        if (wordCount >= 3)
                        {
                            uint targetId = BitConverter.ToUInt32(_spirvCode, (wordIndex + 1) * 4);
                            uint decoration = BitConverter.ToUInt32(_spirvCode, (wordIndex + 2) * 4);

                            if (decoration == 33 && wordCount >= 5) // Binding
                            {
                                uint binding = BitConverter.ToUInt32(_spirvCode, (wordIndex + 4) * 4);
                                uint descriptorSet = 0;
                                if (wordCount >= 4)
                                {
                                    uint nextDeco = BitConverter.ToUInt32(_spirvCode, (wordIndex + 3) * 4);
                                    if (nextDeco == 34) // DescriptorSet
                                        descriptorSet = BitConverter.ToUInt32(_spirvCode, (wordIndex + 5) * 4);
                                }
                            }
                        }
                        break;

                    case 72: // OpMemberDecorate
                        break;

                    case 30: // OpEntryPoint continued
                        break;
                }

                wordIndex += (int)wordCount;
            }
        }

        /// <summary>Gets the shader stage from the SPIR-V entry point</summary>
        public static ShaderStageFlag GetStageFromSpirv(byte[] spirvCode)
        {
            if (spirvCode == null || spirvCode.Length < 20) return ShaderStageFlag.All;
            uint wordCount = BitConverter.ToUInt32(spirvCode, 0);
            // Default to vertex if we can't determine
            return ShaderStageFlag.Vertex;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_module != IntPtr.Zero)
                vkDestroyShaderModule(_device.LogicalDevice, _module, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Shader input binding info</summary>
    public class ShaderInputBinding
    {
        public uint Location { get; set; }
        public uint Binding { get; set; }
        public VulkanFormat Format { get; set; }
        public uint Offset { get; set; }
    }

    /// <summary>Shader output binding info</summary>
    public class ShaderOutputBinding
    {
        public uint Location { get; set; }
        public VulkanFormat Format { get; set; }
    }

    /// <summary>Shader descriptor binding info</summary>
    public class ShaderDescriptorBinding
    {
        public uint Set { get; set; }
        public uint Binding { get; set; }
        public DescriptorType Type { get; set; }
        public uint Count { get; set; }
        public ShaderStageFlag StageFlags { get; set; }
    }
    // =========================================================================
    // VulkanMemoryAllocator
    // =========================================================================

    /// <summary>
    /// VMA-style GPU memory allocator for Vulkan. Manages memory type selection,
    /// suballocation within larger memory blocks, defragmentation, and budget tracking.
    /// </summary>
    public class VulkanMemoryAllocator : IDisposable
    {
        private VulkanDevice _device;
        private readonly object _lock = new object();
        private bool _disposed;

        // Memory pools
        private readonly List<MemoryBlock> _blocks = new List<MemoryBlock>();
        private readonly Dictionary<uint, List<MemoryBlock>> _poolsByMemoryType = new Dictionary<uint, List<MemoryBlock>>();

        // Budget tracking
        private ulong _totalAllocated;
        private ulong _totalReserved;
        private readonly Dictionary<uint, ulong> _budgetByType = new Dictionary<uint, ulong>();
        private readonly Dictionary<uint, ulong> _usageByType = new Dictionary<uint, ulong>();

        // Statistics
        private ulong _allocationCount;
        private ulong _deallocationCount;
        private ulong _peakUsage;

        // Defragmentation
        private readonly List<DefragmentationPass> _pendingDefragmentations = new List<DefragmentationPass>();

        // Function pointers
        private AllocateMemoryDel _vkAllocateMemory;
        private FreeMemoryDel _vkFreeMemory;
        private MapMemoryDel _vkMapMemory;
        private UnmapMemoryDel _vkUnmapMemory;
        private FlushMappedMemoryRangesDel _vkFlushMappedMemoryRanges;
        private GetBufferMemoryRequirementsDel _vkGetBufferMemoryRequirements;
        private GetImageMemoryRequirementsDel _vkGetImageMemoryRequirements;
        private BindBufferMemoryDel _vkBindBufferMemory;
        private BindImageMemoryDel _vkBindImageMemory;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AllocateMemoryDel(IntPtr device, ref VkMemoryAllocateInfo pAllocateInfo, IntPtr pAllocator, ref IntPtr pMemory);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void FreeMemoryDel(IntPtr device, IntPtr memory, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult MapMemoryDel(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, ref IntPtr ppData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void UnmapMemoryDel(IntPtr device, IntPtr memory);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult FlushMappedMemoryRangesDel(IntPtr device, uint memoryRangeCount, ref VkMappedMemoryRange pMemoryRanges);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void GetBufferMemoryRequirementsDel(IntPtr device, IntPtr buffer, out VkMemoryRequirements pMemoryRequirements);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void GetImageMemoryRequirementsDel(IntPtr device, IntPtr image, out VkMemoryRequirements pMemoryRequirements);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindBufferMemoryDel(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult BindImageMemoryDel(IntPtr device, IntPtr image, IntPtr memory, ulong memoryOffset);

        // Block size constants
        private const ulong DEFAULT_BLOCK_SIZE = 256 * 1024 * 1024; // 256 MB
        private const ulong SMALL_BLOCK_SIZE = 64 * 1024 * 1024;    // 64 MB
        private const ulong MIN_ALLOCATION_SIZE = 256;               // 256 bytes alignment
        private const ulong BUFFER_IMAGE_GRANULARITY = 128;          // 128 bytes

        public VulkanMemoryAllocator(VulkanDevice device)
        {
            _device = device;
            LoadFunctions();
            InitializeBudgetTracking();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkAllocateMemory = Marshal.GetDelegateForFunctionPointer<AllocateMemoryDel>(load("vkAllocateMemory"));
            _vkFreeMemory = Marshal.GetDelegateForFunctionPointer<FreeMemoryDel>(load("vkFreeMemory"));
            _vkMapMemory = Marshal.GetDelegateForFunctionPointer<MapMemoryDel>(load("vkMapMemory"));
            _vkUnmapMemory = Marshal.GetDelegateForFunctionPointer<UnmapMemoryDel>(load("vkUnmapMemory"));
            _vkFlushMappedMemoryRanges = Marshal.GetDelegateForFunctionPointer<FlushMappedMemoryRangesDel>(load("vkFlushMappedMemoryRanges"));
            _vkGetBufferMemoryRequirements = Marshal.GetDelegateForFunctionPointer<GetBufferMemoryRequirementsDel>(load("vkGetBufferMemoryRequirements"));
            _vkGetImageMemoryRequirements = Marshal.GetDelegateForFunctionPointer<GetImageMemoryRequirementsDel>(load("vkGetImageMemoryRequirements"));
            _vkBindBufferMemory = Marshal.GetDelegateForFunctionPointer<BindBufferMemoryDel>(load("vkBindBufferMemory"));
            _vkBindImageMemory = Marshal.GetDelegateForFunctionPointer<BindImageMemoryDel>(load("vkBindImageMemory"));
        }

        private void InitializeBudgetTracking()
        {
            var memProps = _device.PhysicalDeviceInfo.MemoryProperties;
            for (uint i = 0; i < memProps.MemoryHeapCount; i++)
            {
                _budgetByType[i] = memProps.MemoryHeaps[i].Size;
                _usageByType[i] = 0;
            }
        }

        /// <summary>Selects the best memory type for the given requirements and properties</summary>
        public uint FindMemoryType(uint typeFilter, MemoryPropertyFlag requiredProperties)
        {
            var memProps = _device.PhysicalDeviceInfo.MemoryProperties;
            for (uint i = 0; i < memProps.MemoryTypeCount; i++)
            {
                if ((typeFilter & (1 << (int)i)) != 0)
                {
                    if ((memProps.MemoryTypes[i].PropertyFlags & requiredProperties) == requiredProperties)
                        return i;
                }
            }

            // Fallback: try with fewer requirements
            for (uint i = 0; i < memProps.MemoryTypeCount; i++)
            {
                if ((typeFilter & (1 << (int)i)) != 0)
                    return i;
            }

            throw new InvalidOperationException("Failed to find suitable memory type");
        }

        /// <summary>Allocates memory for a buffer</summary>
        public void AllocateBuffer(VulkanBuffer buffer, MemoryPropertyFlag requiredProperties)
        {
            var requirements = buffer.MemoryRequirements;
            uint memoryType = FindMemoryType(requirements.MemoryTypeBits, requiredProperties);

            lock (_lock)
            {
                var block = FindOrCreateBlock(memoryType, requirements.Size, requirements.Alignment);
                var allocation = block.Allocate(requirements.Size, requirements.Alignment);

                if (allocation.HasValue)
                {
                    buffer.BindMemory(block.DeviceMemory, allocation.Value.Offset);
                    Interlocked.Increment(ref _allocationCount);
                    UpdateUsage(memoryType, requirements.Size);
                }
            }
        }

        /// <summary>Allocates memory for an image</summary>
        public void AllocateImage(VulkanTexture texture, MemoryPropertyFlag requiredProperties)
        {
            var requirements = texture.MemoryRequirements;
            uint memoryType = FindMemoryType(requirements.MemoryTypeBits, requiredProperties);

            lock (_lock)
            {
                var block = FindOrCreateBlock(memoryType, requirements.Size, requirements.Alignment);
                var allocation = block.Allocate(requirements.Size, requirements.Alignment);

                if (allocation.HasValue)
                {
                    texture.BindMemory(block.DeviceMemory, allocation.Value.Offset);
                    Interlocked.Increment(ref _allocationCount);
                    UpdateUsage(memoryType, requirements.Size);
                }
            }
        }

        /// <summary>Finds or creates a memory block for the given type</summary>
        private MemoryBlock FindOrCreateBlock(uint memoryType, ulong size, ulong alignment)
        {
            if (!_poolsByMemoryType.TryGetValue(memoryType, out var blocks))
            {
                blocks = new List<MemoryBlock>();
                _poolsByMemoryType[memoryType] = blocks;
            }

            // Try to find an existing block with enough space
            foreach (var block in blocks)
            {
                if (block.HasSpace(size, alignment))
                    return block;
            }

            // Create a new block
            ulong blockSize = size > DEFAULT_BLOCK_SIZE ? size * 2 : DEFAULT_BLOCK_SIZE;
            if (size < SMALL_BLOCK_SIZE) blockSize = SMALL_BLOCK_SIZE;

            var newBlock = CreateMemoryBlock(memoryType, blockSize);
            blocks.Add(newBlock);
            _blocks.Add(newBlock);
            return newBlock;
        }

        /// <summary>Creates a new memory block</summary>
        private MemoryBlock CreateMemoryBlock(uint memoryType, ulong size)
        {
            var allocateInfo = new VkMemoryAllocateInfo
            {
                sType = 5,
                allocationSize = size,
                memoryTypeIndex = memoryType
            };

            IntPtr memory = IntPtr.Zero;
            var result = _vkAllocateMemory(_device.LogicalDevice, ref allocateInfo, IntPtr.Zero, ref memory);
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Failed to allocate Vulkan memory: {result}");

            _totalReserved += size;

            return new MemoryBlock(memory, memoryType, size);
        }

        private void UpdateUsage(uint memoryType, ulong size)
        {
            Interlocked.Add(ref _totalAllocated, size);
            if (_usageByType.ContainsKey(memoryType))
                _usageByType[memoryType] += size;
            if (_totalAllocated > _peakUsage)
                _peakUsage = _totalAllocated;
        }

        /// <summary>Runs defragmentation on all memory blocks</summary>
        public void Defragment()
        {
            lock (_lock)
            {
                foreach (var block in _blocks)
                {
                    block.Compact();
                }
            }
        }

        /// <summary>Returns memory budget information for each heap</summary>
        public MemoryBudgetInfo GetBudgetInfo()
        {
            var info = new MemoryBudgetInfo();
            var memProps = _device.PhysicalDeviceInfo.MemoryProperties;

            info.HeapBudgets = new HeapBudget[memProps.MemoryHeapCount];
            for (uint i = 0; i < memProps.MemoryHeapCount; i++)
            {
                info.HeapBudgets[i] = new HeapBudget
                {
                    HeapIndex = i,
                    Budget = _budgetByType.ContainsKey(i) ? _budgetByType[i] : 0,
                    Usage = _usageByType.ContainsKey(i) ? _usageByType[i] : 0,
                    Flags = memProps.MemoryHeaps[i].Flags
                };
            }

            return info;
        }

        /// <summary>Gets current allocation statistics</summary>
        public AllocatorStats GetStats()
        {
            return new AllocatorStats
            {
                TotalAllocated = _totalAllocated,
                TotalReserved = _totalReserved,
                AllocationCount = _allocationCount,
                DeallocationCount = _deallocationCount,
                PeakUsage = _peakUsage,
                BlockCount = _blocks.Count
            };
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            lock (_lock)
            {
                foreach (var block in _blocks)
                    block.Dispose();
                _blocks.Clear();
                _poolsByMemoryType.Clear();
            }
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Represents a block of Vulkan device memory</summary>
    internal class MemoryBlock : IDisposable
    {
        private IntPtr _deviceMemory;
        private uint _memoryType;
        private ulong _size;
        private ulong _used;
        private readonly List<MemoryAllocation> _allocations = new List<MemoryAllocation>();
        private IntPtr _mappedPtr;
        private bool _isMapped;

        public IntPtr DeviceMemory => _deviceMemory;
        public ulong Size => _size;
        public ulong Used => _used;
        public bool IsMapped => _isMapped;

        public MemoryBlock(IntPtr deviceMemory, uint memoryType, ulong size)
        {
            _deviceMemory = deviceMemory;
            _memoryType = memoryType;
            _size = size;
            _used = 0;
        }

        public bool HasSpace(ulong size, ulong alignment)
        {
            ulong alignedOffset = (_used + alignment - 1) & ~(alignment - 1);
            return alignedOffset + size <= _size;
        }

        public MemoryAllocation? Allocate(ulong size, ulong alignment)
        {
            ulong offset = (_used + alignment - 1) & ~(alignment - 1);
            if (offset + size > _size) return null;

            var alloc = new MemoryAllocation
            {
                Offset = offset,
                Size = size,
                IsActive = true
            };

            _allocations.Add(alloc);
            _used = offset + size;
            return alloc;
        }

        public void Free(ulong offset, ulong size)
        {
            for (int i = _allocations.Count - 1; i >= 0; i--)
            {
                if (_allocations[i].Offset == offset && _allocations[i].Size == size)
                {
                    var alloc = _allocations[i];
                    alloc.IsActive = false;
                    _allocations[i] = alloc;
                    break;
                }
            }
            Compact();
        }

        public void Compact()
        {
            _allocations.RemoveAll(a => !a.IsActive);
            _allocations.Sort((a, b) => a.Offset.CompareTo(b.Offset));

            ulong compacted = 0;
            for (int i = 0; i < _allocations.Count; i++)
            {
                var alloc = _allocations[i];
                if (alloc.Offset != compacted)
                {
                    alloc.Offset = compacted;
                    _allocations[i] = alloc;
                }
                compacted += alloc.Size;
            }
            _used = compacted;
        }

        public void Dispose()
        {
            _allocations.Clear();
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Represents a suballocation within a memory block</summary>
    public struct MemoryAllocation
    {
        public ulong Offset;
        public ulong Size;
        public bool IsActive;
    }

    /// <summary>Memory budget information</summary>
    public class MemoryBudgetInfo
    {
        public HeapBudget[] HeapBudgets { get; set; }
    }

    /// <summary>Per-heap budget info</summary>
    public class HeapBudget
    {
        public uint HeapIndex { get; set; }
        public ulong Budget { get; set; }
        public ulong Usage { get; set; }
        public MemoryHeapFlag Flags { get; set; }
    }

    /// <summary>Allocator statistics</summary>
    public class AllocatorStats
    {
        public ulong TotalAllocated { get; set; }
        public ulong TotalReserved { get; set; }
        public ulong AllocationCount { get; set; }
        public ulong DeallocationCount { get; set; }
        public ulong PeakUsage { get; set; }
        public int BlockCount { get; set; }
    }

    /// <summary>Defragmentation pass</summary>
    internal class DefragmentationPass
    {
        public uint MemoryType { get; set; }
        public List<MemoryBlock> Blocks { get; set; }
    }
    // =========================================================================
    // VulkanDescriptorManager
    // =========================================================================

    /// <summary>
    /// Manages Vulkan descriptor set allocation from pools with layout caching,
    /// update batching, and per-pipeline-layout set caching for optimal performance.
    /// </summary>
    public class VulkanDescriptorManager : IDisposable
    {
        private VulkanDevice _device;
        private readonly object _lock = new object();
        private bool _disposed;

        // Layout cache: hash -> layout handle
        private readonly Dictionary<long, IntPtr> _layoutCache = new Dictionary<long, IntPtr>();

        // Pool management: growable pool chain
        private readonly List<IntPtr> _descriptorPools = new List<IntPtr>();
        private int _currentPoolIndex = 0;
        private uint _setsAllocated;
        private uint _maxSetsPerPool = 1024;

        // Set cache per pipeline layout
        private readonly Dictionary<IntPtr, DescriptorSetCache> _setCaches = new Dictionary<IntPtr, DescriptorSetCache>();

        // Update batching
        private readonly List<DescriptorUpdateBatch> _pendingUpdates = new List<DescriptorUpdateBatch>();
        private int _maxBatchSize = 64;

        // Function pointers
        private CreateDescriptorSetLayoutDel _vkCreateDescriptorSetLayout;
        private DestroyDescriptorSetLayoutDel _vkDestroyDescriptorSetLayout;
        private CreateDescriptorPoolDel _vkCreateDescriptorPool;
        private DestroyDescriptorPoolDel _vkDestroyDescriptorPool;
        private AllocateDescriptorSetsDel _vkAllocateDescriptorSets;
        private FreeDescriptorSetsDel _vkFreeDescriptorSets;
        private UpdateDescriptorSetsDel _vkUpdateDescriptorSets;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateDescriptorSetLayoutDel(IntPtr device, ref VkDescriptorSetLayoutCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pSetLayout);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyDescriptorSetLayoutDel(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateDescriptorPoolDel(IntPtr device, ref VkDescriptorPoolCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pDescriptorPool);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyDescriptorPoolDel(IntPtr device, IntPtr descriptorPool, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AllocateDescriptorSetsDel(IntPtr device, ref VkDescriptorSetAllocateInfo pAllocateInfo, IntPtr[] pDescriptorSets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult FreeDescriptorSetsDel(IntPtr device, IntPtr descriptorPool, uint descriptorSetCount, ref IntPtr pDescriptorSets);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void UpdateDescriptorSetsDel(IntPtr device, uint descriptorWriteCount, IntPtr pDescriptorWrites, uint descriptorCopyCount, IntPtr pDescriptorCopies);

        public VulkanDescriptorManager(VulkanDevice device)
        {
            _device = device;
            LoadFunctions();
            CreateNewPool();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkCreateDescriptorSetLayout = Marshal.GetDelegateForFunctionPointer<CreateDescriptorSetLayoutDel>(load("vkCreateDescriptorSetLayout"));
            _vkDestroyDescriptorSetLayout = Marshal.GetDelegateForFunctionPointer<DestroyDescriptorSetLayoutDel>(load("vkDestroyDescriptorSetLayout"));
            _vkCreateDescriptorPool = Marshal.GetDelegateForFunctionPointer<CreateDescriptorPoolDel>(load("vkCreateDescriptorPool"));
            _vkDestroyDescriptorPool = Marshal.GetDelegateForFunctionPointer<DestroyDescriptorPoolDel>(load("vkDestroyDescriptorPool"));
            _vkAllocateDescriptorSets = Marshal.GetDelegateForFunctionPointer<AllocateDescriptorSetsDel>(load("vkAllocateDescriptorSets"));
            _vkFreeDescriptorSets = Marshal.GetDelegateForFunctionPointer<FreeDescriptorSetsDel>(load("vkFreeDescriptorSets"));
            _vkUpdateDescriptorSets = Marshal.GetDelegateForFunctionPointer<UpdateDescriptorSetsDel>(load("vkUpdateDescriptorSets"));
        }

        private void CreateNewPool()
        {
            var poolSizes = new VkDescriptorPoolSize[]
            {
                new VkDescriptorPoolSize { type = DescriptorType.UniformBuffer, descriptorCount = _maxSetsPerPool * 4 },
                new VkDescriptorPoolSize { type = DescriptorType.StorageBuffer, descriptorCount = _maxSetsPerPool * 4 },
                new VkDescriptorPoolSize { type = DescriptorType.CombinedImageSampler, descriptorCount = _maxSetsPerPool * 8 },
                new VkDescriptorPoolSize { type = DescriptorType.SampledImage, descriptorCount = _maxSetsPerPool * 4 },
                new VkDescriptorPoolSize { type = DescriptorType.StorageImage, descriptorCount = _maxSetsPerPool * 4 },
                new VkDescriptorPoolSize { type = DescriptorType.Sampler, descriptorCount = _maxSetsPerPool * 2 },
                new VkDescriptorPoolSize { type = DescriptorType.InputAttachment, descriptorCount = _maxSetsPerPool * 2 },
            };

            int poolSizeStructSize = 8; // VkDescriptorPoolSize
            var poolSizesPtr = Marshal.AllocHGlobal(poolSizes.Length * poolSizeStructSize);
            for (int i = 0; i < poolSizes.Length; i++)
                Marshal.StructureToPtr(poolSizes[i], poolSizesPtr + i * poolSizeStructSize, false);

            var createInfo = new VkDescriptorPoolCreateInfo
            {
                sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                flags = 0, // No reset flag for simplicity
                maxSets = _maxSetsPerPool,
                poolSizeCount = (uint)poolSizes.Length,
                pPoolSizes = poolSizesPtr
            };

            IntPtr pool = IntPtr.Zero;
            _vkCreateDescriptorPool(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref pool);
            Marshal.FreeHGlobal(poolSizesPtr);

            _descriptorPools.Add(pool);
            _currentPoolIndex = _descriptorPools.Count - 1;
            _setsAllocated = 0;
        }

        /// <summary>Computes a hash for descriptor set layout bindings</summary>
        public long ComputeLayoutHash(DescriptorSetLayoutBinding[] bindings)
        {
            if (bindings == null || bindings.Length == 0) return 0;

            long hash = 17;
            foreach (var binding in bindings)
            {
                hash = hash * 31 + binding.Binding;
                hash = hash * 31 + (long)binding.DescriptorType;
                hash = hash * 31 + binding.DescriptorCount;
                hash = hash * 31 + (long)binding.StageFlags;
            }
            return hash;
        }

        /// <summary>Allocates a descriptor set from the current pool</summary>
        public IntPtr AllocateDescriptorSet(IntPtr layout)
        {
            lock (_lock)
            {
                var layoutsPtr = Marshal.AllocHGlobal(IntPtr.Size);
                Marshal.WriteIntPtr(layoutsPtr, layout);

                var allocInfo = new VkDescriptorSetAllocateInfo
                {
                    sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool = _descriptorPools[_currentPoolIndex],
                    descriptorSetCount = 1,
                    pSetLayouts = layoutsPtr
                };

                var sets = new IntPtr[1];
                var result = _vkAllocateDescriptorSets(_device.LogicalDevice, ref allocInfo, sets);
                Marshal.FreeHGlobal(layoutsPtr);

                if (result == VulkanResult.ErrorOutOfPoolMemory || result == VulkanResult.ErrorFragmentedPool)
                {
                    CreateNewPool();
                    allocInfo.descriptorPool = _descriptorPools[_currentPoolIndex];
                    layoutsPtr = Marshal.AllocHGlobal(IntPtr.Size);
                    Marshal.WriteIntPtr(layoutsPtr, layout);
                    allocInfo.pSetLayouts = layoutsPtr;
                    result = _vkAllocateDescriptorSets(_device.LogicalDevice, ref allocInfo, sets);
                    Marshal.FreeHGlobal(layoutsPtr);
                }

                if (result != VulkanResult.Success)
                    throw new InvalidOperationException($"Failed to allocate descriptor set: {result}");

                _setsAllocated++;
                return sets[0];
            }
        }

        /// <summary>Batch updates multiple descriptor sets</summary>
        public void BatchUpdateDescriptors(DescriptorWrite[] writes)
        {
            if (writes == null || writes.Length == 0) return;

            lock (_lock)
            {
                int writeStructSize = 56; // Approximate VkWriteDescriptorSet size
                var writesPtr = Marshal.AllocHGlobal(writes.Length * writeStructSize);
                var tempAllocs = new List<IntPtr> { writesPtr };

                for (int w = 0; w < writes.Length; w++)
                {
                    var write = writes[w];
                    IntPtr imageInfoPtr = IntPtr.Zero;
                    IntPtr bufferInfoPtr = IntPtr.Zero;

                    if (write.ImageInfos != null && write.ImageInfos.Length > 0)
                    {
                        int imgInfoSize = 24; // VkDescriptorImageInfo size
                        imageInfoPtr = Marshal.AllocHGlobal(write.ImageInfos.Length * imgInfoSize);
                        tempAllocs.Add(imageInfoPtr);
                        for (int i = 0; i < write.ImageInfos.Length; i++)
                        {
                            var imgInfo = new VkDescriptorImageInfo
                            {
                                sampler = write.ImageInfos[i].Sampler,
                                imageView = write.ImageInfos[i].ImageView,
                                imageLayout = write.ImageInfos[i].ImageLayout
                            };
                            Marshal.StructureToPtr(imgInfo, imageInfoPtr + i * imgInfoSize, false);
                        }
                    }

                    if (write.BufferInfos != null && write.BufferInfos.Length > 0)
                    {
                        int bufInfoSize = 24; // VkDescriptorBufferInfo size
                        bufferInfoPtr = Marshal.AllocHGlobal(write.BufferInfos.Length * bufInfoSize);
                        tempAllocs.Add(bufferInfoPtr);
                        for (int i = 0; i < write.BufferInfos.Length; i++)
                        {
                            var bufInfo = new VkDescriptorBufferInfo
                            {
                                buffer = write.BufferInfos[i].Buffer,
                                offset = write.BufferInfos[i].Offset,
                                range = write.BufferInfos[i].Range
                            };
                            Marshal.StructureToPtr(bufInfo, bufferInfoPtr + i * bufInfoSize, false);
                        }
                    }

                    // Marshal VkWriteDescriptorSet
                    long baseAddr = (long)writesPtr + w * writeStructSize;
                    Marshal.WriteInt32((IntPtr)baseAddr, (int)VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
                    Marshal.WriteInt64((IntPtr)(baseAddr + 8), 0); // pNext
                    Marshal.WriteIntPtr((IntPtr)(baseAddr + 16), write.DescriptorSet);
                    Marshal.WriteInt32((IntPtr)(baseAddr + 24), (int)write.DstBinding);
                    Marshal.WriteInt32((IntPtr)(baseAddr + 28), (int)write.DstArrayElement);
                    Marshal.WriteInt32((IntPtr)(baseAddr + 32), write.ImageInfos?.Length ?? write.BufferInfos?.Length ?? 0);
                    Marshal.WriteInt32((IntPtr)(baseAddr + 36), (int)write.DescriptorType);
                    Marshal.WriteIntPtr((IntPtr)(baseAddr + 40), imageInfoPtr);
                    Marshal.WriteIntPtr((IntPtr)(baseAddr + 48), bufferInfoPtr);
                }

                _vkUpdateDescriptorSets(_device.LogicalDevice, (uint)writes.Length, writesPtr, 0, IntPtr.Zero);

                foreach (var ptr in tempAllocs) Marshal.FreeHGlobal(ptr);
            }
        }

        /// <summary>Gets the default descriptor pool handle</summary>
        public IntPtr GetDefaultPool() => _descriptorPools[_currentPoolIndex];

        private const uint VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
        private const uint VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
        private const uint VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var pool in _descriptorPools)
                if (pool != IntPtr.Zero) _vkDestroyDescriptorPool?.Invoke(_device.LogicalDevice, pool, IntPtr.Zero);
            foreach (var layout in _layoutCache.Values)
                if (layout != IntPtr.Zero) _vkDestroyDescriptorSetLayout?.Invoke(_device.LogicalDevice, layout, IntPtr.Zero);
            _descriptorPools.Clear();
            _layoutCache.Clear();
            GC.SuppressFinalize(this);
        }
    }

    internal struct VkDescriptorSetLayoutCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint bindingCount; public IntPtr pBindings;
    }

    /// <summary>Descriptor set cache entry</summary>
    internal class DescriptorSetCache
    {
        public IntPtr PipelineLayout { get; set; }
        public Dictionary<uint, IntPtr> CachedSets { get; set; } = new Dictionary<uint, IntPtr>();
        public uint Generation { get; set; }
    }

    /// <summary>Descriptor update batch entry</summary>
    internal class DescriptorUpdateBatch
    {
        public IntPtr DescriptorSet { get; set; }
        public uint Binding { get; set; }
        public DescriptorType Type { get; set; }
        public DescriptorImageInfo[] ImageInfos { get; set; }
        public DescriptorBufferInfo[] BufferInfos { get; set; }
    }
    // =========================================================================
    // VulkanPipelineCache
    // =========================================================================

    /// <summary>
    /// Manages Vulkan pipeline state object caching, cache serialization/deserialization,
    /// and cache warming to reduce pipeline creation time.
    /// </summary>
    public class VulkanPipelineCache : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _cache;
        private readonly object _lock = new object();
        private bool _disposed;

        // Cache statistics
        private long _cacheHits;
        private long _cacheMisses;
        private long _totalCreations;

        // Cache warming state
        private readonly List<PipelineCacheEntry> _warmedEntries = new List<PipelineCacheEntry>();

        // Function pointers
        private CreatePipelineCacheDel _vkCreatePipelineCache;
        private DestroyPipelineCacheDel _vkDestroyPipelineCache;
        private GetPipelineCacheDataDel _vkGetPipelineCacheData;
        private MergePipelineCachesDel _vkMergePipelineCaches;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreatePipelineCacheDel(IntPtr device, ref VkPipelineCacheCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pPipelineCache);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineCacheDel(IntPtr device, IntPtr pipelineCache, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult GetPipelineCacheDataDel(IntPtr device, IntPtr pipelineCache, ref IntPtr pDataSize, IntPtr pData);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult MergePipelineCachesDel(IntPtr device, IntPtr dstCache, uint srcCacheCount, ref IntPtr pSrcCaches);

        public IntPtr Handle => _cache;
        public long CacheHits => _cacheHits;
        public long CacheMisses => _cacheMisses;

        public VulkanPipelineCache(VulkanDevice device, IntPtr existingCache = default)
        {
            _device = device;
            _cache = existingCache;
            LoadFunctions();

            if (_cache == IntPtr.Zero)
            {
                var createInfo = new VkPipelineCacheCreateInfo
                {
                    sType = 17,
                    initialDataSize = IntPtr.Zero,
                    pInitialData = IntPtr.Zero
                };
                _vkCreatePipelineCache(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref _cache);
            }
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkCreatePipelineCache = Marshal.GetDelegateForFunctionPointer<CreatePipelineCacheDel>(load("vkCreatePipelineCache"));
            _vkDestroyPipelineCache = Marshal.GetDelegateForFunctionPointer<DestroyPipelineCacheDel>(load("vkDestroyPipelineCache"));
            _vkGetPipelineCacheData = Marshal.GetDelegateForFunctionPointer<GetPipelineCacheDataDel>(load("vkGetPipelineCacheData"));
            _vkMergePipelineCaches = Marshal.GetDelegateForFunctionPointer<MergePipelineCachesDel>(load("vkMergePipelineCaches"));
        }

        /// <summary>Serializes the pipeline cache data for disk storage</summary>
        public byte[] SerializeCache()
        {
            lock (_lock)
            {
                IntPtr dataSize = IntPtr.Zero;
                _vkGetPipelineCacheData(_device.LogicalDevice, _cache, ref dataSize, IntPtr.Zero);

                long size = (long)dataSize;
                if (size == 0) return Array.Empty<byte>();

                var data = new byte[size];
                var dataPtr = Marshal.AllocHGlobal((int)size);
                try
                {
                    _vkGetPipelineCacheData(_device.LogicalDevice, _cache, ref dataSize, dataPtr);
                    Marshal.Copy(dataPtr, data, 0, (int)size);
                }
                finally { Marshal.FreeHGlobal(dataPtr); }
                return data;
            }
        }

        /// <summary>Deserializes pipeline cache data from disk</summary>
        public void DeserializeCache(byte[] cacheData)
        {
            if (cacheData == null || cacheData.Length == 0) return;

            lock (_lock)
            {
                // Destroy old cache
                if (_cache != IntPtr.Zero)
                    _vkDestroyPipelineCache(_device.LogicalDevice, _cache, IntPtr.Zero);

                var dataPtr = Marshal.AllocHGlobal(cacheData.Length);
                try
                {
                    Marshal.Copy(cacheData, 0, dataPtr, cacheData.Length);
                    var createInfo = new VkPipelineCacheCreateInfo
                    {
                        sType = 17,
                        initialDataSize = (IntPtr)cacheData.Length,
                        pInitialData = dataPtr
                    };
                    _vkCreatePipelineCache(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref _cache);
                }
                finally { Marshal.FreeHGlobal(dataPtr); }
            }
        }

        /// <summary>Warms the cache by pre-creating pipelines from cached data</summary>
        public void WarmCache(byte[] cachedData)
        {
            if (cachedData != null && cachedData.Length > 0)
                DeserializeCache(cachedData);

            // Mark all warmed entries
            lock (_lock)
            {
                _warmedEntries.Clear();
                Interlocked.Exchange(ref _totalCreations, 0);
            }
        }

        /// <summary>Merges another pipeline cache into this one</summary>
        public void MergeCache(VulkanPipelineCache otherCache)
        {
            if (otherCache == null) return;

            lock (_lock)
            {
                var otherHandle = otherCache.Handle;
                _vkMergePipelineCaches(_device.LogicalDevice, _cache, 1, ref otherHandle);
            }
        }

        /// <summary>Records a cache hit</summary>
        internal void RecordHit() => Interlocked.Increment(ref _cacheHits);

        /// <summary>Records a cache miss</summary>
        internal void RecordMiss() => Interlocked.Increment(ref _cacheMisses);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_cache != IntPtr.Zero)
                _vkDestroyPipelineCache?.Invoke(_device.LogicalDevice, _cache, IntPtr.Zero);
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Pipeline cache warming entry</summary>
    internal class PipelineCacheEntry
    {
        public string Hash { get; set; }
        public IntPtr Pipeline { get; set; }
        public byte[] SerializedData { get; set; }
    }
    // =========================================================================
    // VulkanSyncManager
    // =========================================================================

    /// <summary>
    /// Manages Vulkan synchronization primitives including fences, semaphores,
    /// and timeline semaphores. Provides a recycling pool for efficient reuse.
    /// </summary>
    public class VulkanSyncManager : IDisposable
    {
        private VulkanDevice _device;
        private readonly object _lock = new object();
        private bool _disposed;

        // Fence pool
        private readonly ConcurrentBag<IntPtr> _fencePool = new ConcurrentBag<IntPtr>();
        private readonly List<IntPtr> _allFences = new List<IntPtr>();
        private int _activeFenceCount;

        // Semaphore pool
        private readonly ConcurrentBag<IntPtr> _semaphorePool = new ConcurrentBag<IntPtr>();
        private readonly List<IntPtr> _allSemaphores = new List<IntPtr>();
        private int _activeSemaphoreCount;

        // Timeline semaphores
        private readonly List<IntPtr> _timelineSemaphores = new List<IntPtr>();
        private readonly Dictionary<IntPtr, ulong> _timelineValues = new Dictionary<IntPtr, ulong>();

        // Function pointers
        private CreateFenceDel _vkCreateFence;
        private DestroyFenceDel _vkDestroyFence;
        private WaitForFencesDel _vkWaitForFences;
        private ResetFencesDel _vkResetFences;
        private CreateSemaphoreDel _vkCreateSemaphore;
        private DestroySemaphoreDel _vkDestroySemaphore;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateFenceDel(IntPtr device, ref VkFenceCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pFence);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyFenceDel(IntPtr device, IntPtr fence, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult WaitForFencesDel(IntPtr device, uint fenceCount, ref IntPtr pFences, uint waitAll, ulong timeout);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult ResetFencesDel(IntPtr device, uint fenceCount, ref IntPtr pFences);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateSemaphoreDel(IntPtr device, ref VkSemaphoreCreateInfo pCreateInfo, IntPtr pAllocator, ref IntPtr pSemaphore);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroySemaphoreDel(IntPtr device, IntPtr semaphore, IntPtr pAllocator);

        public int ActiveFences => _activeFenceCount;
        public int ActiveSemaphores => _activeSemaphoreCount;

        public VulkanSyncManager(VulkanDevice device)
        {
            _device = device;
            LoadFunctions();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkCreateFence = Marshal.GetDelegateForFunctionPointer<CreateFenceDel>(load("vkCreateFence"));
            _vkDestroyFence = Marshal.GetDelegateForFunctionPointer<DestroyFenceDel>(load("vkDestroyFence"));
            _vkWaitForFences = Marshal.GetDelegateForFunctionPointer<WaitForFencesDel>(load("vkWaitForFences"));
            _vkResetFences = Marshal.GetDelegateForFunctionPointer<ResetFencesDel>(load("vkResetFences"));
            _vkCreateSemaphore = Marshal.GetDelegateForFunctionPointer<CreateSemaphoreDel>(load("vkCreateSemaphore"));
            _vkDestroySemaphore = Marshal.GetDelegateForFunctionPointer<DestroySemaphoreDel>(load("vkDestroySemaphore"));
        }

        /// <summary>Acquires a fence from the pool or creates a new one</summary>
        public IntPtr CreateFence(bool signaled = false)
        {
            lock (_lock)
            {
                if (!signaled && _fencePool.TryTake(out var fence))
                {
                    Interlocked.Increment(ref _activeFenceCount);
                    return fence;
                }
            }

            var createInfo = new VkFenceCreateInfo
            {
                sType = 8,
                flags = signaled ? 1u : 0u
            };

            IntPtr newFence = IntPtr.Zero;
            _vkCreateFence(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref newFence);

            lock (_lock)
            {
                _allFences.Add(newFence);
                Interlocked.Increment(ref _activeFenceCount);
            }

            return newFence;
        }

        /// <summary>Returns a fence to the pool for reuse</summary>
        public void RecycleFence(IntPtr fence)
        {
            if (fence == IntPtr.Zero) return;

            lock (_lock)
            {
                Interlocked.Decrement(ref _activeFenceCount);
                _fencePool.Add(fence);
            }
        }

        /// <summary>Waits for a single fence with timeout</summary>
        public VulkanResult WaitForFence(IntPtr fence, ulong timeout = 0xFFFFFFFFFFFFFFFF)
        {
            if (fence == IntPtr.Zero) return VulkanResult.ErrorUnknown;
            return _vkWaitForFences(_device.LogicalDevice, 1, ref fence, 1, timeout);
        }

        /// <summary>Waits for multiple fences</summary>
        public VulkanResult WaitForFences(IntPtr[] fences, bool waitAll = true, ulong timeout = 0xFFFFFFFFFFFFFFFF)
        {
            if (fences == null || fences.Length == 0) return VulkanResult.Success;
            uint count = (uint)fences.Length;
            var fenceArray = Marshal.AllocHGlobal(fences.Length * IntPtr.Size);
            try
            {
                for (int i = 0; i < fences.Length; i++)
                    Marshal.WriteIntPtr(fenceArray + i * IntPtr.Size, fences[i]);
                return _vkWaitForFences(_device.LogicalDevice, count, ref fenceArray, waitAll ? 1u : 0u, timeout);
            }
            finally { Marshal.FreeHGlobal(fenceArray); }
        }

        /// <summary>Resets one or more fences</summary>
        public VulkanResult ResetFence(IntPtr fence)
        {
            if (fence == IntPtr.Zero) return VulkanResult.ErrorUnknown;
            return _vkResetFences(_device.LogicalDevice, 1, ref fence);
        }

        public VulkanResult ResetFences(IntPtr[] fences)
        {
            if (fences == null || fences.Length == 0) return VulkanResult.Success;
            uint count = (uint)fences.Length;
            var fenceArray = Marshal.AllocHGlobal(fences.Length * IntPtr.Size);
            try
            {
                for (int i = 0; i < fences.Length; i++)
                    Marshal.WriteIntPtr(fenceArray + i * IntPtr.Size, fences[i]);
                return _vkResetFences(_device.LogicalDevice, count, ref fenceArray);
            }
            finally { Marshal.FreeHGlobal(fenceArray); }
        }

        /// <summary>Acquires a semaphore from the pool or creates a new one</summary>
        public IntPtr CreateSemaphore()
        {
            lock (_lock)
            {
                if (_semaphorePool.TryTake(out var semaphore))
                {
                    Interlocked.Increment(ref _activeSemaphoreCount);
                    return semaphore;
                }
            }

            var createInfo = new VkSemaphoreCreateInfo { sType = 7 };
            IntPtr newSemaphore = IntPtr.Zero;
            _vkCreateSemaphore(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref newSemaphore);

            lock (_lock)
            {
                _allSemaphores.Add(newSemaphore);
                Interlocked.Increment(ref _activeSemaphoreCount);
            }

            return newSemaphore;
        }

        /// <summary>Returns a semaphore to the pool</summary>
        public void RecycleSemaphore(IntPtr semaphore)
        {
            if (semaphore == IntPtr.Zero) return;
            lock (_lock)
            {
                Interlocked.Decrement(ref _activeSemaphoreCount);
                _semaphorePool.Add(semaphore);
            }
        }

        /// <summary>Creates a timeline semaphore</summary>
        public IntPtr CreateTimelineSemaphore(ulong initialValue = 0)
        {
            var timelineFeatures = new VkTimelineSemaphoreCreateInfo
            {
                sType = 1000076002,
                semaphoreType = 1,
                initialValue = initialValue
            };

            IntPtr semaphore = IntPtr.Zero;
            ref VkSemaphoreCreateInfo createInfo = ref Unsafe.As<VkTimelineSemaphoreCreateInfo, VkSemaphoreCreateInfo>(ref timelineFeatures);
            _vkCreateSemaphore(_device.LogicalDevice, ref createInfo, IntPtr.Zero, ref semaphore);

            lock (_lock)
            {
                _timelineSemaphores.Add(semaphore);
                _timelineValues[semaphore] = initialValue;
            }

            return semaphore;
        }

        /// <summary>Gets the current timeline value for a semaphore</summary>
        public ulong GetTimelineValue(IntPtr semaphore)
        {
            lock (_lock)
            {
                return _timelineValues.ContainsKey(semaphore) ? _timelineValues[semaphore] : 0;
            }
        }

        /// <summary>Recycles all unused synchronization objects</summary>
        public void RecycleAll()
        {
            lock (_lock)
            {
                while (_fencePool.TryTake(out _)) { }
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            lock (_lock)
            {
                foreach (var fence in _allFences)
                    _vkDestroyFence?.Invoke(_device.LogicalDevice, fence, IntPtr.Zero);
                foreach (var semaphore in _allSemaphores)
                    _vkDestroySemaphore?.Invoke(_device.LogicalDevice, semaphore, IntPtr.Zero);
                foreach (var timeline in _timelineSemaphores)
                    _vkDestroySemaphore?.Invoke(_device.LogicalDevice, timeline, IntPtr.Zero);

                _allFences.Clear();
                _allSemaphores.Clear();
                _timelineSemaphores.Clear();
            }
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Timeline semaphore create info extension</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct VkTimelineSemaphoreCreateInfo
    {
        public uint sType;
        public IntPtr pNext;
        public uint flags;
        public uint semaphoreType;
        public ulong initialValue;
    }

    /// <summary>Vulkan fence wrapper</summary>
    public class Fence : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _fence;
        private VulkanSyncManager _syncManager;
        private bool _disposed;

        public IntPtr Handle => _fence;

        public Fence(VulkanDevice device, IntPtr fence, VulkanSyncManager syncManager)
        {
            _device = device;
            _fence = fence;
            _syncManager = syncManager;
        }

        public VulkanResult Wait(ulong timeout = 0xFFFFFFFFFFFFFFFF) => _syncManager.WaitForFence(_fence, timeout);
        public VulkanResult Reset() => _syncManager.ResetFence(_fence);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _syncManager?.RecycleFence(_fence);
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Vulkan semaphore wrapper</summary>
    public class Semaphore : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _semaphore;
        private VulkanSyncManager _syncManager;
        private bool _disposed;

        public IntPtr Handle => _semaphore;

        public Semaphore(VulkanDevice device, IntPtr semaphore, VulkanSyncManager syncManager)
        {
            _device = device;
            _semaphore = semaphore;
            _syncManager = syncManager;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _syncManager?.RecycleSemaphore(_semaphore);
            GC.SuppressFinalize(this);
        }
    }
    // =========================================================================
    // VulkanResourceTracker
    // =========================================================================

    /// <summary>
    /// Tracks resource lifetime with reference counting, deferred deletion,
    /// and resource state tracking for safe GPU resource management.
    /// </summary>
    public class VulkanResourceTracker : IDisposable
    {
        private VulkanDevice _device;
        private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();
        private bool _disposed;

        // Resource tracking
        private readonly Dictionary<IntPtr, TrackedResource> _trackedResources = new Dictionary<IntPtr, TrackedResource>();

        // Deferred deletion queue
        private readonly Queue<DeferredDeletion> _deletionQueue = new Queue<DeferredDeletion>();
        private readonly List<DeferredDeletion> _activeDeletions = new List<DeferredDeletion>();

        // Resource state tracking
        private readonly Dictionary<IntPtr, ResourceState> _resourceStates = new Dictionary<IntPtr, ResourceState>();

        // Statistics
        private long _totalTracked;
        private long _totalDeleted;
        private long _pendingDeletions;

        // Function pointers
        private DestroyBufferDel _vkDestroyBuffer;
        private DestroyImageDel _vkDestroyImage;
        private DestroyImageViewDel _vkDestroyImageView;
        private DestroySamplerDel _vkDestroySampler;
        private DestroyRenderPassDel _vkDestroyRenderPass;
        private DestroyFramebufferDel _vkDestroyFramebuffer;
        private DestroyPipelineDel _vkDestroyPipeline;
        private DestroyPipelineLayoutDel _vkDestroyPipelineLayout;
        private DestroyShaderModuleDel _vkDestroyShaderModule;
        private DestroyDescriptorSetLayoutDel _vkDestroyDescriptorSetLayout;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyBufferDel(IntPtr device, IntPtr buffer, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageDel(IntPtr device, IntPtr image, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyImageViewDel(IntPtr device, IntPtr imageView, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroySamplerDel(IntPtr device, IntPtr sampler, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyRenderPassDel(IntPtr device, IntPtr renderPass, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyFramebufferDel(IntPtr device, IntPtr framebuffer, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineDel(IntPtr device, IntPtr pipeline, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyPipelineLayoutDel(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyShaderModuleDel(IntPtr device, IntPtr shaderModule, IntPtr pAllocator);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate void DestroyDescriptorSetLayoutDel(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);

        public int TrackedCount => _trackedResources.Count;
        public int PendingDeletions => _deletionQueue.Count;

        public VulkanResourceTracker(VulkanDevice device)
        {
            _device = device;
            LoadFunctions();
        }

        private void LoadFunctions()
        {
            var load = new Func<string, IntPtr>(name =>
            {
                return _device.GetDeviceProcAddr(name);
            });

            _vkDestroyBuffer = Marshal.GetDelegateForFunctionPointer<DestroyBufferDel>(load("vkDestroyBuffer"));
            _vkDestroyImage = Marshal.GetDelegateForFunctionPointer<DestroyImageDel>(load("vkDestroyImage"));
            _vkDestroyImageView = Marshal.GetDelegateForFunctionPointer<DestroyImageViewDel>(load("vkDestroyImageView"));
            _vkDestroySampler = Marshal.GetDelegateForFunctionPointer<DestroySamplerDel>(load("vkDestroySampler"));
            _vkDestroyRenderPass = Marshal.GetDelegateForFunctionPointer<DestroyRenderPassDel>(load("vkDestroyRenderPass"));
            _vkDestroyFramebuffer = Marshal.GetDelegateForFunctionPointer<DestroyFramebufferDel>(load("vkDestroyFramebuffer"));
            _vkDestroyPipeline = Marshal.GetDelegateForFunctionPointer<DestroyPipelineDel>(load("vkDestroyPipeline"));
            _vkDestroyPipelineLayout = Marshal.GetDelegateForFunctionPointer<DestroyPipelineLayoutDel>(load("vkDestroyPipelineLayout"));
            _vkDestroyShaderModule = Marshal.GetDelegateForFunctionPointer<DestroyShaderModuleDel>(load("vkDestroyShaderModule"));
            _vkDestroyDescriptorSetLayout = Marshal.GetDelegateForFunctionPointer<DestroyDescriptorSetLayoutDel>(load("vkDestroyDescriptorSetLayout"));
        }

        /// <summary>Tracks a resource for lifetime management</summary>
        public void TrackResource(VulkanBuffer resource)
        {
            if (resource == null) return;
            _lock.EnterWriteLock();
            try
            {
                _trackedResources[resource.Handle] = new TrackedResource
                {
                    Handle = resource.Handle,
                    Type = ResourceType.Buffer,
                    ReferenceCount = 1,
                    CreationFrame = Interlocked.Read(ref _totalTracked)
                };
                Interlocked.Increment(ref _totalTracked);
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Tracks a texture resource</summary>
        public void TrackResource(VulkanTexture resource)
        {
            if (resource == null) return;
            _lock.EnterWriteLock();
            try
            {
                _trackedResources[resource.Handle] = new TrackedResource
                {
                    Handle = resource.Handle,
                    Type = ResourceType.Image,
                    ReferenceCount = 1,
                    CreationFrame = Interlocked.Read(ref _totalTracked)
                };
                Interlocked.Increment(ref _totalTracked);
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Tracks a render pass resource</summary>
        public void TrackResource(VulkanRenderPass resource)
        {
            if (resource == null) return;
            _lock.EnterWriteLock();
            try
            {
                _trackedResources[resource.Handle] = new TrackedResource
                {
                    Handle = resource.Handle,
                    Type = ResourceType.RenderPass,
                    ReferenceCount = 1,
                    CreationFrame = Interlocked.Read(ref _totalTracked)
                };
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Adds a reference to a tracked resource</summary>
        public void AddReference(IntPtr handle)
        {
            _lock.EnterUpgradeableReadLock();
            try
            {
                if (_trackedResources.TryGetValue(handle, out var resource))
                {
                    _lock.EnterWriteLock();
                    try { resource.ReferenceCount++; }
                    finally { _lock.ExitWriteLock(); }
                }
            }
            finally { _lock.ExitUpgradeableReadLock(); }
        }

        /// <summary>Removes a reference and queues for deletion if count reaches zero</summary>
        public void RemoveReference(IntPtr handle)
        {
            _lock.EnterUpgradeableReadLock();
            try
            {
                if (_trackedResources.TryGetValue(handle, out var resource))
                {
                    _lock.EnterWriteLock();
                    try
                    {
                        resource.ReferenceCount--;
                        if (resource.ReferenceCount <= 0)
                        {
                            QueueDeletion(resource);
                            _trackedResources.Remove(handle);
                        }
                    }
                    finally { _lock.ExitWriteLock(); }
                }
            }
            finally { _lock.ExitUpgradeableReadLock(); }
        }

        /// <summary>Queues a resource for deferred deletion</summary>
        private void QueueDeletion(TrackedResource resource)
        {
            var deletion = new DeferredDeletion
            {
                Resource = resource,
                DeletionFrame = Interlocked.Read(ref _totalTracked),
                FramesToWait = 3 // Triple-buffered safety
            };
            _deletionQueue.Enqueue(deletion);
            Interlocked.Increment(ref _pendingDeletions);
        }

        /// <summary>Processes deferred deletions for completed frames</summary>
        public void ProcessDeletions()
        {
            _lock.EnterWriteLock();
            try
            {
                while (_deletionQueue.Count > 0)
                {
                    var deletion = _deletionQueue.Peek();
                    long currentFrame = Interlocked.Read(ref _totalTracked);

                    if (currentFrame - deletion.DeletionFrame >= deletion.FramesToWait)
                    {
                        _deletionQueue.Dequeue();
                        DestroyResource(deletion.Resource);
                        Interlocked.Decrement(ref _pendingDeletions);
                        Interlocked.Increment(ref _totalDeleted);
                    }
                    else
                    {
                        break;
                    }
                }
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Destroys a tracked resource</summary>
        private void DestroyResource(TrackedResource resource)
        {
            switch (resource.Type)
            {
                case ResourceType.Buffer:
                    _vkDestroyBuffer?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.Image:
                    _vkDestroyImage?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.ImageView:
                    _vkDestroyImageView?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.Sampler:
                    _vkDestroySampler?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.RenderPass:
                    _vkDestroyRenderPass?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.Framebuffer:
                    _vkDestroyFramebuffer?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.Pipeline:
                    _vkDestroyPipeline?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.PipelineLayout:
                    _vkDestroyPipelineLayout?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.ShaderModule:
                    _vkDestroyShaderModule?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
                case ResourceType.DescriptorSetLayout:
                    _vkDestroyDescriptorSetLayout?.Invoke(_device.LogicalDevice, resource.Handle, IntPtr.Zero);
                    break;
            }
        }

        /// <summary>Updates the state of a tracked resource</summary>
        public void SetResourceState(IntPtr handle, ResourceState state)
        {
            _lock.EnterWriteLock();
            try { _resourceStates[handle] = state; }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Gets the current state of a tracked resource</summary>
        public ResourceState GetResourceState(IntPtr handle)
        {
            _lock.EnterReadLock();
            try
            {
                return _resourceStates.TryGetValue(handle, out var state) ? state : ResourceState.Unknown;
            }
            finally { _lock.ExitReadLock(); }
        }

        /// <summary>Forces immediate deletion of all pending resources</summary>
        public void FlushAll()
        {
            _lock.EnterWriteLock();
            try
            {
                while (_deletionQueue.Count > 0)
                {
                    var deletion = _deletionQueue.Dequeue();
                    DestroyResource(deletion.Resource);
                    Interlocked.Decrement(ref _pendingDeletions);
                    Interlocked.Increment(ref _totalDeleted);
                }
            }
            finally { _lock.ExitWriteLock(); }
        }

        /// <summary>Gets tracking statistics</summary>
        public TrackerStats GetStats()
        {
            return new TrackerStats
            {
                TotalTracked = _totalTracked,
                TotalDeleted = _totalDeleted,
                PendingDeletions = _pendingDeletions,
                ActiveResources = _trackedResources.Count
            };
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            FlushAll();
            _lock.Dispose();
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>Tracked resource entry</summary>
    internal class TrackedResource
    {
        public IntPtr Handle { get; set; }
        public ResourceType Type { get; set; }
        public int ReferenceCount { get; set; }
        public long CreationFrame { get; set; }
    }

    /// <summary>Deferred deletion entry</summary>
    internal class DeferredDeletion
    {
        public TrackedResource Resource { get; set; }
        public long DeletionFrame { get; set; }
        public int FramesToWait { get; set; }
    }

    /// <summary>Resource types for tracking</summary>
    public enum ResourceType
    {
        Buffer,
        Image,
        ImageView,
        Sampler,
        RenderPass,
        Framebuffer,
        Pipeline,
        PipelineLayout,
        ShaderModule,
        DescriptorSetLayout,
        CommandBuffer,
        QueryPool
    }

    /// <summary>Resource state for state tracking</summary>
    public enum ResourceState
    {
        Unknown,
        Common,
        VertexBuffer,
        IndexBuffer,
        ConstantBuffer,
        StreamOutput,
        IndirectArgument,
        Predication,
        ShaderResource,
        UnorderedAccess,
        RenderTarget,
        DepthWrite,
        DepthRead,
        NonPixelShaderResource,
        PixelShaderResource,
        CopyDest,
        CopySource,
        ResolveDest,
        ResolveSource,
        GenericRead,
        Present
    }

    /// <summary>Tracker statistics</summary>
    public class TrackerStats
    {
        public long TotalTracked { get; set; }
        public long TotalDeleted { get; set; }
        public long PendingDeletions { get; set; }
        public int ActiveResources { get; set; }
    }
    // =========================================================================
    // EXTENSION METHODS
    // =========================================================================

    /// <summary>Extension methods for Vulkan utility types</summary>
    public static class VulkanExtensions
    {
        /// <summary>Calculates the aligned size for a given size and alignment</summary>
        public static ulong AlignSize(this ulong size, ulong alignment)
        {
            return (size + alignment - 1) & ~(alignment - 1);
        }

        /// <summary>Calculates the number of mip levels for a texture of given dimensions</summary>
        public static uint CalculateMipLevels(uint width, uint height, uint depth = 1)
        {
            return (uint)(Math.Floor(Math.Log2(Math.Max(width, Math.Max(height, depth)))) + 1);
        }

        /// <summary>Calculates the row pitch for a texture format and width</summary>
        public static uint CalculateRowPitch(VulkanFormat format, uint width, uint alignment = 4)
        {
            uint bitsPerPixel = GetBitsPerPixel(format);
            uint rowBytes = (width * bitsPerPixel + 7) / 8;
            return (rowBytes + alignment - 1) & ~(alignment - 1);
        }

        /// <summary>Calculates the total image size in bytes</summary>
        public static ulong CalculateImageSize(VulkanFormat format, uint width, uint height, uint depth = 1, uint mipLevel = 0)
        {
            uint mipWidth = Math.Max(1, width >> (int)mipLevel);
            uint mipHeight = Math.Max(1, height >> (int)mipLevel);
            uint mipDepth = Math.Max(1, depth >> (int)mipLevel);

            if (IsCompressedFormat(format))
                return CalculateCompressedSize(format, mipWidth, mipHeight, mipDepth);

            uint bitsPerPixel = GetBitsPerPixel(format);
            ulong rowPitch = CalculateRowPitch(format, mipWidth, 4);
            return rowPitch * mipHeight * mipDepth;
        }

        /// <summary>Gets the bits per pixel for a format</summary>
        public static uint GetBitsPerPixel(VulkanFormat format)
        {
            return format switch
            {
                VulkanFormat.R4G4UnormPack8 => 8,
                VulkanFormat.R4G4B4A4UnormPack16 or VulkanFormat.B4G4R4A4UnormPack16 => 16,
                VulkanFormat.R5G6B5UnormPack16 or VulkanFormat.B5G6R5UnormPack16 => 16,
                VulkanFormat.R5G5B5A1UnormPack16 or VulkanFormat.B5G5R5A1UnormPack16 or VulkanFormat.A1R5G5B5UnormPack16 => 16,
                VulkanFormat.R8Unorm or VulkanFormat.R8Snorm or VulkanFormat.R8Uscaled or VulkanFormat.R8Sscaled or
                VulkanFormat.R8Uint or VulkanFormat.R8Sint or VulkanFormat.R8Srgb => 8,
                VulkanFormat.R8G8Unorm or VulkanFormat.R8G8Snorm or VulkanFormat.R8G8Uscaled or VulkanFormat.R8G8Sscaled or
                VulkanFormat.R8G8Uint or VulkanFormat.R8G8Sint or VulkanFormat.R8G8Srgb => 16,
                VulkanFormat.R8G8B8Unorm or VulkanFormat.R8G8B8Snorm or VulkanFormat.R8G8B8Uscaled or VulkanFormat.R8G8B8Sscaled or
                VulkanFormat.R8G8B8Uint or VulkanFormat.R8G8B8Sint or VulkanFormat.R8G8B8Srgb or
                VulkanFormat.B8G8R8Unorm or VulkanFormat.B8G8R8Snorm or VulkanFormat.B8G8R8Uscaled or VulkanFormat.B8G8R8Sscaled or
                VulkanFormat.B8G8R8Uint or VulkanFormat.B8G8R8Sint or VulkanFormat.B8G8R8Srgb => 24,
                VulkanFormat.R8G8B8A8Unorm or VulkanFormat.R8G8B8A8Snorm or VulkanFormat.R8G8B8A8Uscaled or VulkanFormat.R8G8B8A8Sscaled or
                VulkanFormat.R8G8B8A8Uint or VulkanFormat.R8G8B8A8Sint or VulkanFormat.R8G8B8A8Srgb or
                VulkanFormat.B8G8R8A8Unorm or VulkanFormat.B8G8R8A8Snorm or VulkanFormat.B8G8R8A8Uscaled or VulkanFormat.B8G8R8A8Sscaled or
                VulkanFormat.B8G8R8A8Uint or VulkanFormat.B8G8R8A8Sint or VulkanFormat.B8G8R8A8Srgb or
                VulkanFormat.A8B8G8R8UnormPack32 or VulkanFormat.A8B8G8R8SnormPack32 or VulkanFormat.A8B8G8R8UintPack32 or
                VulkanFormat.A8B8G8R8SintPack32 or VulkanFormat.A8B8G8R8SrgbPack32 or
                VulkanFormat.A2R10G10B10UnormPack32 or VulkanFormat.A2R10G10B10SnormPack32 or VulkanFormat.A2R10G10B10UintPack32 or
                VulkanFormat.A2B10G10R10UnormPack32 or VulkanFormat.A2B10G10R10SnormPack32 or VulkanFormat.A2B10G10R10UintPack32 or
                VulkanFormat.R16Unorm or VulkanFormat.R16Snorm or VulkanFormat.R16Uscaled or VulkanFormat.R16Sscaled or
                VulkanFormat.R16Uint or VulkanFormat.R16Sint or VulkanFormat.R16Sfloat or
                VulkanFormat.D16Unorm or VulkanFormat.B10G11R11UfloatPack32 or VulkanFormat.E5B9G9R9UfloatPack32 => 32,
                VulkanFormat.R16G16Unorm or VulkanFormat.R16G16Snorm or VulkanFormat.R16G16Uscaled or VulkanFormat.R16G16Sscaled or
                VulkanFormat.R16G16Uint or VulkanFormat.R16G16Sint or VulkanFormat.R16G16Sfloat or
                VulkanFormat.D32Sfloat or VulkanFormat.X8D24UnormPack32 => 32,
                VulkanFormat.R16G16B16Unorm or VulkanFormat.R16G16B16Snorm or VulkanFormat.R16G16B16Uint or
                VulkanFormat.R16G16B16Sint or VulkanFormat.R16G16B16Sfloat => 48,
                VulkanFormat.R16G16B16A16Unorm or VulkanFormat.R16G16B16A16Snorm or VulkanFormat.R16G16B16A16Uint or
                VulkanFormat.R16G16B16A16Sint or VulkanFormat.R16G16B16A16Sfloat or VulkanFormat.D32SfloatS8Uint or
                VulkanFormat.D24UnormS8Uint => 64,
                VulkanFormat.R32Uint or VulkanFormat.R32Sint or VulkanFormat.R32Sfloat => 32,
                VulkanFormat.R32G32Uint or VulkanFormat.R32G32Sint or VulkanFormat.R32G32Sfloat => 64,
                VulkanFormat.R32G32B32Uint or VulkanFormat.R32G32B32Sint or VulkanFormat.R32G32B32Sfloat => 96,
                VulkanFormat.R32G32B32A32Uint or VulkanFormat.R32G32B32A32Sint or VulkanFormat.R32G32B32A32Sfloat => 128,
                VulkanFormat.R64Uint or VulkanFormat.R64Sint or VulkanFormat.R64Sfloat => 64,
                VulkanFormat.R64G64Uint or VulkanFormat.R64G64Sint or VulkanFormat.R64G64Sfloat => 128,
                VulkanFormat.R64G64B64Uint or VulkanFormat.R64G64B64Sint or VulkanFormat.R64G64B64Sfloat => 192,
                VulkanFormat.R64G64B64A64Uint or VulkanFormat.R64G64B64A64Sint or VulkanFormat.R64G64B64A64Sfloat => 256,
                VulkanFormat.S8Uint => 8,
                VulkanFormat.D16UnormS8Uint => 24,
                _ => 32
            };
        }

        /// <summary>Checks if a format is a compressed format</summary>
        public static bool IsCompressedFormat(VulkanFormat format)
        {
            return (format >= VulkanFormat.BC1RgbUnormBlock && format <= VulkanFormat.BC7SrgbBlock) ||
                   (format >= VulkanFormat.Etc2R8G8B8UnormBlock && format <= VulkanFormat.Astc12x12SrgbBlock) ||
                   (format >= VulkanFormat.EacR11UnormBlock && format <= VulkanFormat.EacR11G11SnormBlock);
        }

        private static ulong CalculateCompressedSize(VulkanFormat format, uint width, uint height, uint depth)
        {
            uint blockSizeX = 4, blockSizeY = 4;
            if (format >= VulkanFormat.Astc4x4UnormBlock && format <= VulkanFormat.Astc4x4SrgbBlock)
            { blockSizeX = 4; blockSizeY = 4; }
            else if (format >= VulkanFormat.Astc5x4UnormBlock && format <= VulkanFormat.Astc5x5SrgbBlock)
            { blockSizeX = 5; blockSizeY = 5; }
            else if (format >= VulkanFormat.BC1RgbUnormBlock && format <= VulkanFormat.BC4SnormBlock)
            { blockSizeX = 4; blockSizeY = 4; }
            else if (format >= VulkanFormat.BC5UnormBlock && format <= VulkanFormat.BC7SrgbBlock)
            { blockSizeX = 4; blockSizeY = 4; }

            uint blocksX = (width + blockSizeX - 1) / blockSizeX;
            uint blocksY = (height + blockSizeY - 1) / blockSizeY;
            uint bytesPerBlock = GetCompressedBlockSize(format);
            return (ulong)blocksX * blocksY * depth * bytesPerBlock;
        }

        private static uint GetCompressedBlockSize(VulkanFormat format)
        {
            if (format >= VulkanFormat.BC1RgbUnormBlock && format <= VulkanFormat.BC1RgbaSrgbBlock) return 8;
            if (format >= VulkanFormat.BC2UnormBlock && format <= VulkanFormat.BC3SrgbBlock) return 16;
            if (format >= VulkanFormat.BC4UnormBlock && format <= VulkanFormat.BC4SnormBlock) return 8;
            if (format >= VulkanFormat.BC5UnormBlock && format <= VulkanFormat.BC5SnormBlock) return 16;
            if (format >= VulkanFormat.BC6HUfloatBlock && format <= VulkanFormat.BC6HSfloatBlock) return 16;
            if (format >= VulkanFormat.BC7UnormBlock && format <= VulkanFormat.BC7SrgbBlock) return 16;
            if (format >= VulkanFormat.Etc2R8G8B8UnormBlock && format <= VulkanFormat.Etc2R8G8B8A1SrgbBlock) return 8;
            if (format >= VulkanFormat.EacR11UnormBlock && format <= VulkanFormat.EacR11G11SnormBlock) return 8;
            if (format >= VulkanFormat.Astc4x4UnormBlock && format <= VulkanFormat.Astc4x4SrgbBlock) return 16;
            return 16;
        }

        /// <summary>Converts a VulkanFormat to bytes per pixel for non-compressed formats</summary>
        public static uint GetBytesPerPixel(VulkanFormat format)
        {
            return (GetBitsPerPixel(format) + 7) / 8;
        }

        /// <summary>Checks if the format has a depth component</summary>
        public static bool IsDepthFormat(VulkanFormat format)
        {
            return format is VulkanFormat.D16Unorm or VulkanFormat.X8D24UnormPack32 or
                VulkanFormat.D32Sfloat or VulkanFormat.D16UnormS8Uint or
                VulkanFormat.D24UnormS8Uint or VulkanFormat.D32SfloatS8Uint;
        }

        /// <summary>Checks if the format has a stencil component</summary>
        public static bool IsStencilFormat(VulkanFormat format)
        {
            return format is VulkanFormat.S8Uint or VulkanFormat.D16UnormS8Uint or
                VulkanFormat.D24UnormS8Uint or VulkanFormat.D32SfloatS8Uint;
        }

        /// <summary>Checks if the format has both depth and stencil</summary>
        public static bool IsDepthStencilFormat(VulkanFormat format)
        {
            return format is VulkanFormat.D16UnormS8Uint or VulkanFormat.D24UnormS8Uint or VulkanFormat.D32SfloatS8Uint;
        }

        /// <summary>Returns the corresponding depth-only format for a depth-stencil format</summary>
        public static VulkanFormat GetDepthOnlyFormat(VulkanFormat format)
        {
            return format switch
            {
                VulkanFormat.D16UnormS8Uint => VulkanFormat.D16Unorm,
                VulkanFormat.D24UnormS8Uint => VulkanFormat.X8D24UnormPack32,
                VulkanFormat.D32SfloatS8Uint => VulkanFormat.D32Sfloat,
                _ => format
            };
        }

        /// <summary>Returns the corresponding stencil-only format for a depth-stencil format</summary>
        public static VulkanFormat GetStencilOnlyFormat(VulkanFormat format)
        {
            return format switch
            {
                VulkanFormat.D16UnormS8Uint => VulkanFormat.S8Uint,
                VulkanFormat.D24UnormS8Uint => VulkanFormat.S8Uint,
                VulkanFormat.D32SfloatS8Uint => VulkanFormat.S8Uint,
                _ => format
            };
        }

        /// <summary>Converts an ImageLayout to a string representation</summary>
        public static string ToDisplayString(this ImageLayout layout)
        {
            return layout switch
            {
                ImageLayout.Undefined => "Undefined",
                ImageLayout.General => "General",
                ImageLayout.ColorAttachmentOptimal => "Color Attachment",
                ImageLayout.DepthStencilAttachmentOptimal => "Depth/Stencil Attachment",
                ImageLayout.DepthStencilReadOnlyOptimal => "Depth/Stencil Read-Only",
                ImageLayout.ShaderReadOnlyOptimal => "Shader Read-Only",
                ImageLayout.TransferSrcOptimal => "Transfer Source",
                ImageLayout.TransferDstOptimal => "Transfer Destination",
                ImageLayout.Preinitialized => "Preinitialized",
                ImageLayout.PresentSrcKHR => "Present Source",
                _ => "Unknown"
            };
        }

        /// <summary>Converts a PipelineStageFlag to a string representation</summary>
        public static string ToDisplayString(this PipelineStageFlag flag)
        {
            if (flag == PipelineStageFlag.TopOfPipe) return "Top of Pipe";
            if (flag == PipelineStageFlag.BottomOfPipe) return "Bottom of Pipe";
            if (flag == PipelineStageFlag.AllCommands) return "All Commands";
            if (flag == PipelineStageFlag.AllGraphics) return "All Graphics";

            var parts = new List<string>();
            if ((flag & PipelineStageFlag.VertexInput) != 0) parts.Add("Vertex Input");
            if ((flag & PipelineStageFlag.VertexShader) != 0) parts.Add("Vertex Shader");
            if ((flag & PipelineStageFlag.FragmentShader) != 0) parts.Add("Fragment Shader");
            if ((flag & PipelineStageFlag.GeometryShader) != 0) parts.Add("Geometry Shader");
            if ((flag & PipelineStageFlag.ComputeShader) != 0) parts.Add("Compute Shader");
            if ((flag & PipelineStageFlag.Transfer) != 0) parts.Add("Transfer");
            if ((flag & PipelineStageFlag.ColorAttachmentOutput) != 0) parts.Add("Color Attachment Output");
            if ((flag & PipelineStageFlag.EarlyFragmentTests) != 0) parts.Add("Early Fragment Tests");
            if ((flag & PipelineStageFlag.LateFragmentTests) != 0) parts.Add("Late Fragment Tests");
            if ((flag & PipelineStageFlag.DrawIndirect) != 0) parts.Add("Draw Indirect");
            if ((flag & PipelineStageFlag.Host) != 0) parts.Add("Host");
            return parts.Count > 0 ? string.Join(" | ", parts) : flag.ToString();
        }

        /// <summary>Creates an identity 4x4 matrix</summary>
        public static float[] CreateIdentityMatrix4x4()
        {
            return new float[]
            {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
            };
        }

        /// <summary>Creates a perspective projection matrix</summary>
        public static float[] CreatePerspectiveMatrix(float fovY, float aspectRatio, float nearPlane, float farPlane)
        {
            float tanHalfFov = (float)Math.Tan(fovY * 0.5f);
            return new float[]
            {
                1.0f / (aspectRatio * tanHalfFov), 0, 0, 0,
                0, -1.0f / tanHalfFov, 0, 0,
                0, 0, farPlane / (nearPlane - farPlane), -1,
                0, 0, (farPlane * nearPlane) / (nearPlane - farPlane), 0
            };
        }

        /// <summary>Creates an orthographic projection matrix</summary>
        public static float[] CreateOrthographicMatrix(float left, float right, float bottom, float top, float nearPlane, float farPlane)
        {
            return new float[]
            {
                2.0f / (right - left), 0, 0, 0,
                0, 2.0f / (top - bottom), 0, 0,
                0, 0, 1.0f / (nearPlane - farPlane), 0,
                -(right + left) / (right - left), -(top + bottom) / (top - bottom), nearPlane / (nearPlane - farPlane), 1
            };
        }

        /// <summary>Creates a look-at view matrix</summary>
        public static float[] CreateLookAtMatrix(float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ)
        {
            float fx = centerX - eyeX, fy = centerY - eyeY, fz = centerZ - eyeZ;
            float len = (float)Math.Sqrt(fx * fx + fy * fy + fz * fz);
            fx /= len; fy /= len; fz /= len;

            float sx = fy * upZ - fz * upY;
            float sy = fz * upX - fx * upZ;
            float sz = fx * upY - fy * upX;
            len = (float)Math.Sqrt(sx * sx + sy * sy + sz * sz);
            sx /= len; sy /= len; sz /= len;

            float ux = sy * fz - sz * fy;
            float uy = sz * fx - sx * fz;
            float uz = sx * fy - sy * fx;

            return new float[]
            {
                sx, ux, -fx, 0,
                sy, uy, -fy, 0,
                sz, uz, -fz, 0,
                -(sx * eyeX + sy * eyeY + sz * eyeZ),
                -(ux * eyeX + uy * eyeY + uz * eyeZ),
                (fx * eyeX + fy * eyeY + fz * eyeZ),
                1
            };
        }

        /// <summary>Checks if the format is a sRGB format</summary>
        public static bool IsSRGBFormat(VulkanFormat format)
        {
            return format is VulkanFormat.R8Srgb or VulkanFormat.R8G8Srgb or VulkanFormat.R8G8B8Srgb or
                VulkanFormat.B8G8R8Srgb or VulkanFormat.R8G8B8A8Srgb or VulkanFormat.B8G8R8A8Srgb or
                VulkanFormat.A8B8G8R8SrgbPack32;
        }

        /// <summary>Returns the corresponding linear format for a sRGB format</summary>
        public static VulkanFormat ToLinearFormat(VulkanFormat format)
        {
            return format switch
            {
                VulkanFormat.R8Srgb => VulkanFormat.R8Unorm,
                VulkanFormat.R8G8Srgb => VulkanFormat.R8G8Unorm,
                VulkanFormat.R8G8B8Srgb => VulkanFormat.R8G8B8Unorm,
                VulkanFormat.B8G8R8Srgb => VulkanFormat.B8G8R8Unorm,
                VulkanFormat.R8G8B8A8Srgb => VulkanFormat.R8G8B8A8Unorm,
                VulkanFormat.B8G8R8A8Srgb => VulkanFormat.B8G8R8A8Unorm,
                VulkanFormat.A8B8G8R8SrgbPack32 => VulkanFormat.A8B8G8R8UnormPack32,
                _ => format
            };
        }

        /// <summary>Returns the corresponding sRGB format for a linear format</summary>
        public static VulkanFormat ToSRGBFormat(VulkanFormat format)
        {
            return format switch
            {
                VulkanFormat.R8Unorm => VulkanFormat.R8Srgb,
                VulkanFormat.R8G8Unorm => VulkanFormat.R8G8Srgb,
                VulkanFormat.R8G8B8Unorm => VulkanFormat.R8G8B8Srgb,
                VulkanFormat.B8G8R8Unorm => VulkanFormat.B8G8R8Srgb,
                VulkanFormat.R8G8B8A8Unorm => VulkanFormat.R8G8B8A8Srgb,
                VulkanFormat.B8G8R8A8Unorm => VulkanFormat.B8G8R8A8Srgb,
                VulkanFormat.A8B8G8R8UnormPack32 => VulkanFormat.A8B8G8R8SrgbPack32,
                _ => format
            };
        }

        /// <summary>Validates Vulkan result code and throws on error</summary>
        public static void ThrowOnError(this VulkanResult result, string operation = "")
        {
            if (result != VulkanResult.Success)
                throw new InvalidOperationException($"Vulkan error {result}" + (string.IsNullOrEmpty(operation) ? "" : $" during {operation}"));
        }

        /// <summary>Converts a ShaderStageFlag to a human-readable name</summary>
        public static string ToDisplayString(this ShaderStageFlag stage)
        {
            return stage switch
            {
                ShaderStageFlag.Vertex => "Vertex",
                ShaderStageFlag.TessellationControl => "Tessellation Control",
                ShaderStageFlag.TessellationEvaluation => "Tessellation Evaluation",
                ShaderStageFlag.Geometry => "Geometry",
                ShaderStageFlag.Fragment => "Fragment",
                ShaderStageFlag.Compute => "Compute",
                ShaderStageFlag.AllGraphics => "All Graphics",
                ShaderStageFlag.All => "All Stages",
                ShaderStageFlag.RayGenerationKHR => "Ray Generation",
                ShaderStageFlag.AnyHitKHR => "Any Hit",
                ShaderStageFlag.ClosestHitKHR => "Closest Hit",
                ShaderStageFlag.MissKHR => "Miss",
                ShaderStageFlag.IntersectionKHR => "Intersection",
                ShaderStageFlag.CallableKHR => "Callable",
                _ => stage.ToString()
            };
        }
    }

    // =========================================================================
    // VkImageBlit struct for blit operations
    // =========================================================================
    [StructLayout(LayoutKind.Sequential)]
    public struct VkImageBlit
    {
        public VkImageSubresourceLayers srcSubresource;
        public VkOffset3D srcOffsets0;
        public VkOffset3D srcOffsets1;
        public VkImageSubresourceLayers dstSubresource;
        public VkOffset3D dstOffsets0;
        public VkOffset3D dstOffsets1;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkInstanceCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public IntPtr pApplicationInfo; public uint enabledLayerCount;
        public IntPtr ppEnabledLayerNames; public uint enabledExtensionCount;
        public IntPtr ppEnabledExtensionNames;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineLayoutCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint setLayoutCount; public IntPtr pSetLayouts;
        public uint pushConstantRangeCount; public IntPtr pPushConstantRanges;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPushConstantRange
    {
        public ShaderStageFlag stageFlags;
        public uint offset;
        public uint size;
    }

    // =========================================================================
    // VULKAN INTEROP TYPES
    // =========================================================================

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkBool32
    {
        private uint _value;
        public static readonly VkBool32 False = new VkBool32 { _value = 0 };
        public static readonly VkBool32 True = new VkBool32 { _value = 1 };
        public static implicit operator bool(VkBool32 v) => v._value != 0;
        public static implicit operator VkBool32(bool b) => b ? True : False;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VulkanBool32
    {
        private uint _value;
        public static readonly VulkanBool32 False = new VulkanBool32 { _value = 0 };
        public static readonly VulkanBool32 True = new VulkanBool32 { _value = 1 };
        public static implicit operator bool(VulkanBool32 v) => v._value != 0;
        public static implicit operator VulkanBool32(bool b) => b ? True : False;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkExtent2D { public uint width; public uint height; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkExtent3D { public uint width; public uint height; public uint depth; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkOffset3D { public int x; public int y; public int z; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryRequirements
    {
        public ulong size;
        public ulong alignment;
        public uint memoryTypeBits;

        public ulong Size => size;
        public ulong Alignment => alignment;
        public uint MemoryTypeBits => memoryTypeBits;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMappedMemoryRange { public uint sType; public IntPtr pNext; public IntPtr memory; public ulong offset; public ulong size; }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkImageSubresourceLayers { public uint aspectMask; public uint mipLevel; public uint baseArrayLayer; public uint layerCount; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkImageSubresourceRange
    {
        public ImageAspectFlag aspectMask; public uint baseMipLevel; public uint levelCount; public uint baseArrayLayer; public uint layerCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkComponentMapping { public uint r; public uint g; public uint b; public uint a; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkMemoryAllocateInfo { public uint sType; public IntPtr pNext; public ulong allocationSize; public uint memoryTypeIndex; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkFenceCreateInfo { public uint sType; public IntPtr pNext; public uint flags; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSemaphoreCreateInfo { public uint sType; public IntPtr pNext; public uint flags; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDeviceCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint queueCreateInfoCount; public IntPtr pQueueCreateInfos;
        public uint enabledLayerCount; public IntPtr ppEnabledLayerNames;
        public uint enabledExtensionCount; public IntPtr ppEnabledExtensionNames;
        public IntPtr pEnabledFeatures;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkBufferCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public ulong size; public BufferUsageFlag usage; public SharingMode sharingMode;
        public uint queueFamilyIndexCount; public IntPtr pQueueFamilyIndices;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkImageCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public ImageType imageType; public VulkanFormat format; public VkExtent3D extent;
        public uint mipLevels; public uint arrayLayers; public SampleCountFlag samples;
        public ImageTiling tiling; public ImageUsageFlag usage; public SharingMode sharingMode;
        public uint queueFamilyIndexCount; public IntPtr pQueueFamilyIndices; public ImageLayout initialLayout;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkImageViewCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public IntPtr image; public ImageViewType viewType; public VulkanFormat format;
        public VkComponentMapping components; public VkImageSubresourceRange subresourceRange;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPresentInfoKHR
    {
        public uint sType; public IntPtr pNext;
        public uint waitSemaphoreCount; public IntPtr pWaitSemaphores;
        public uint swapchainCount; public IntPtr pSwapchains;
        public IntPtr pImageIndices; public IntPtr pResults;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkRenderPassCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint attachmentCount; public IntPtr pAttachments;
        public uint subpassCount; public IntPtr pSubpasses;
        public uint dependencyCount; public IntPtr pDependencies;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSubpassDescription
    {
        public uint flags; public PipelineBindPoint pipelineBindPoint;
        public uint inputAttachmentCount; public IntPtr pInputAttachments;
        public uint colorAttachmentCount; public IntPtr pColorAttachments;
        public IntPtr pResolveAttachments; public IntPtr pDepthStencilAttachment;
        public uint preserveAttachmentCount; public IntPtr pPreserveAttachments;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSubpassDependency
    {
        public uint srcSubpass; public uint dstSubpass;
        public PipelineStageFlag srcStageMask; public PipelineStageFlag dstStageMask;
        public AccessFlag srcAccessMask; public AccessFlag dstAccessMask;
        public uint dependencyFlags;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkFramebufferCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public IntPtr renderPass; public uint attachmentCount; public IntPtr pAttachments;
        public uint width; public uint height; public uint layers;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSurfaceCapabilitiesKHR
    {
        public uint minImageCount; public uint maxImageCount;
        public uint currentImageWidth; public uint currentImageHeight;
        public uint minImageWidth; public uint minImageHeight;
        public uint maxImageWidth; public uint maxImageHeight;
        public uint maxImageArrayLayers;
        public SurfaceTransformFlag supportedTransforms;
        public SurfaceTransformFlag currentTransform;
        public CompositeAlphaFlag supportedCompositeAlpha;
        public ImageUsageFlag supportedUsageFlags;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSwapchainCreateInfoKHR
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public IntPtr surface; public uint minImageCount;
        public VulkanFormat imageFormat; public uint imageColorSpace;
        public VkExtent2D imageExtent; public uint imageArrayLayers;
        public ImageUsageFlag imageUsage; public SharingMode imageSharingMode;
        public uint queueFamilyIndexCount; public IntPtr pQueueFamilyIndices;
        public SurfaceTransformFlag preTransform; public CompositeAlphaFlag compositeAlpha;
        public PresentMode presentMode; public VulkanBool32 clipped; public IntPtr oldSwapchain;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSubmitInfo
    {
        public uint sType; public IntPtr pNext;
        public uint waitSemaphoreCount; public IntPtr pWaitSemaphores;
        public IntPtr pWaitDstStageMask;
        public uint commandBufferCount; public IntPtr pCommandBuffers;
        public uint signalSemaphoreCount; public IntPtr pSignalSemaphores;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkCommandPoolCreateInfo { public uint sType; public IntPtr pNext; public uint flags; public uint queueFamilyIndex; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkCommandBufferAllocateInfo { public uint sType; public IntPtr pNext; public IntPtr commandPool; public uint level; public uint commandBufferCount; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkCommandBufferBeginInfo
    {
        public uint sType;
        public IntPtr pNext;
        public uint flags;
        public IntPtr pInheritanceInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkRenderPassBeginInfo_Rect2D
    {
        public int x, y;
        public uint width, height;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkRenderPassBeginInfo
    {
        public uint sType;
        public IntPtr pNext;
        public IntPtr renderPass;
        public IntPtr framebuffer;
        public VkRenderPassBeginInfo_Rect2D renderArea;
        public uint clearValueCount;
        public IntPtr pClearValues;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkShaderModuleCreateInfo { public uint sType; public IntPtr pNext; public uint flags; public IntPtr codeSize; public IntPtr pCode; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkSamplerCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public Filter magFilter; public Filter minFilter; public SamplerAddressMode mipmapMode;
        public SamplerAddressMode addressModeU; public SamplerAddressMode addressModeV; public SamplerAddressMode addressModeW;
        public float mipLodBias; public VulkanBool32 anisotropyEnable; public float maxAnisotropy;
        public VulkanBool32 compareEnable; public CompareOp compareOp;
        public float minLod; public float maxLod; public uint borderColor; public VulkanBool32 unnormalizedCoordinates;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorSetLayoutBinding
    {
        public uint binding; public DescriptorType descriptorType; public uint descriptorCount;
        public ShaderStageFlag stageFlags; public IntPtr pImmutableSamplers;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorPoolSize { public DescriptorType type; public uint descriptorCount; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorPoolCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint maxSets; public uint poolSizeCount; public IntPtr pPoolSizes;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorSetAllocateInfo
    {
        public uint sType; public IntPtr pNext;
        public IntPtr descriptorPool; public uint descriptorSetCount; public IntPtr pSetLayouts;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorImageInfo { public IntPtr sampler; public IntPtr imageView; public ImageLayout imageLayout; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkDescriptorBufferInfo { public IntPtr buffer; public ulong offset; public ulong range; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkWriteDescriptorSet
    {
        public uint sType; public IntPtr pNext;
        public IntPtr dstSet; public uint dstBinding; public uint dstArrayElement;
        public uint descriptorCount; public DescriptorType descriptorType;
        public IntPtr pImageInfo; public IntPtr pBufferInfo; public IntPtr pTexelBufferView;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkImageMemoryBarrier
    {
        public uint sType; public IntPtr pNext;
        public AccessFlag srcAccessMask; public AccessFlag dstAccessMask;
        public ImageLayout oldLayout; public ImageLayout newLayout;
        public uint srcQueueFamilyIndex; public uint dstQueueFamilyIndex;
        public IntPtr image; public VkImageSubresourceRange subresourceRange;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineVertexInputStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint vertexBindingDescriptionCount; public IntPtr pVertexBindingDescriptions;
        public uint vertexAttributeDescriptionCount; public IntPtr pVertexAttributeDescriptions;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineInputAssemblyStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public PrimitiveTopology topology; public VulkanBool32 primitiveRestartEnable;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineTessellationStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint patchControlPoints;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineViewportStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint viewportCount; public IntPtr pViewports;
        public uint scissorCount; public IntPtr pScissors;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineRasterizationStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public VulkanBool32 depthClampEnable; public VulkanBool32 rasterizerDiscardEnable;
        public PolygonMode polygonMode; public CullModeFlag cullMode; public FrontFace frontFace;
        public VulkanBool32 depthBiasEnable; public float depthBiasConstantFactor;
        public float depthBiasClamp; public float depthBiasSlopeFactor; public float lineWidth;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineMultisampleStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public SampleCountFlag rasterizationSamples; public VulkanBool32 sampleShadingEnable;
        public float minSampleShading; public IntPtr pSampleMask;
        public VulkanBool32 alphaToCoverageEnable; public VulkanBool32 alphaToOneEnable;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkStencilOpState
    {
        public uint failOp; public uint passOp; public uint depthFailOp; public uint compareOp;
        public uint compareMask; public uint writeMask; public uint reference;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineDepthStencilStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public VulkanBool32 depthTestEnable; public VulkanBool32 depthWriteEnable; public CompareOp depthCompareOp;
        public VulkanBool32 depthBoundsTestEnable; public VulkanBool32 stencilTestEnable;
        public VkStencilOpState front; public VkStencilOpState back;
        public float minDepthBounds; public float maxDepthBounds;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineColorBlendStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public VulkanBool32 logicOpEnable; public LogicOp logicOp;
        public uint attachmentCount; public IntPtr pAttachments;
        public float blendConstant0; public float blendConstant1; public float blendConstant2; public float blendConstant3;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineDynamicStateCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public uint dynamicStateCount; public IntPtr pDynamicStates;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkGraphicsPipelineCreateInfo
    {
        public uint sType; public IntPtr pNext; public PipelineCreateFlag flags;
        public uint stageCount; public IntPtr pStages;
        public IntPtr pVertexInputState; public IntPtr pInputAssemblyState;
        public IntPtr pTessellationState; public IntPtr pViewportState;
        public IntPtr pRasterizationState; public IntPtr pMultisampleState;
        public IntPtr pDepthStencilState; public IntPtr pColorBlendState;
        public IntPtr pDynamicState; public IntPtr layout;
        public IntPtr renderPass; public uint subpass;
        public IntPtr basePipelineHandle; public int basePipelineIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkComputePipelineCreateInfo
    {
        public uint sType; public IntPtr pNext; public PipelineCreateFlag flags;
        public uint stageCount; public IntPtr pStage;
        public IntPtr layout;
        public IntPtr basePipelineHandle; public int basePipelineIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VkPipelineCacheCreateInfo
    {
        public uint sType; public IntPtr pNext; public uint flags;
        public IntPtr initialDataSize; public IntPtr pInitialData;
    }

    /// <summary>Vulkan pipeline layout wrapper</summary>
    public class VulkanPipelineLayout : IDisposable
    {
        private VulkanDevice _device;
        private IntPtr _layout;
        private bool _disposed;

        public IntPtr Handle => _layout;

        public VulkanPipelineLayout(VulkanDevice device, IntPtr layout)
        {
            _device = device;
            _layout = layout;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_layout != IntPtr.Zero && _device?.LogicalDevice != IntPtr.Zero)
            {
                var pfn = _device.GetDeviceProcAddr("vkDestroyPipelineLayout");
                if (pfn != IntPtr.Zero)
                {
                    var del = Marshal.GetDelegateForFunctionPointer<DestroyPipelineLayoutDel>(pfn);
                    del(_device.LogicalDevice, _layout, IntPtr.Zero);
                }
            }
            GC.SuppressFinalize(this);
        }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate void DestroyPipelineLayoutDel(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator);
    }
}
