using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using GDNN.Lighting.LDNN;
using GDNN.Materials.SubstrateOmega;
using GDNN.Polygonization;
using GDNN.Rendering;
using GDNN.Rendering.Bridge;
using GDNN.Rendering.Compat;
using GDNN.Rendering.FrameGraph;
using GDNN.Rendering.FrameGraph.Passes;
using GDNN.Rendering.LOD;
using GDNN.Rendering.Quality;
using GDNN.Rendering.MeshIO;
using GDNN.Rendering.RayTracing;
using GDNN.Rendering.Shaders;
using GDNN.RHI.Vulkan;
using GDNN.Scene;
using LightingParams = GDNN.Scene.LightingParams;

namespace GDNN.Rendering.Engine
{
    public class SceneRenderer : IDisposable
    {
        private const int MAX_FRAMES_IN_FLIGHT = 2;
        /// <summary>Albedo, normals, depth, material, velocity.</summary>
        private const int GBUFFER_ATTACHMENT_COUNT = 5;
        private const int SHADOW_MAP_SIZE = 2048;
        /// <summary>Vulkan minUniformBufferOffsetAlignment is typically ≤256.</summary>
        private const int UboAlign = 256;
        private const int MinDrawSlots = 32;

        private VulkanRhiDevice _rhi;
        private MaterialBridge _materialBridge;
        private LDNNBridge _ldnnBridge;
        private PostProcessBridge _postProcessBridge;
        private MeshLoader _meshLoader;
        private LodManager _lodManager;
        private bool _disposed;
        private SampleCountFlag _msaaSamples = SampleCountFlag.Count1;

        private VulkanRenderPass _gbufferRenderPass;
        private VulkanPipelineLayout _gbufferPipelineLayout;
        private VulkanPipeline _gbufferPipeline;
        private VulkanTexture[] _gbufferAlbedo;
        private VulkanTexture[] _gbufferNormals;
        private VulkanTexture[] _gbufferDepth;
        private VulkanTexture[] _gbufferMaterial;
        private VulkanTexture[] _gbufferVelocity;
        private VulkanTexture[] _depthImages;
        private RayTracingPipeline? _rtPipeline;
        private VulkanFramebuffer[] _gbufferFramebuffers;
        private DescriptorSetLayout _gbufferDescriptorSetLayout;
        private DescriptorPool _gbufferDescriptorPool;
        private DescriptorSet[] _gbufferDescriptorSets;
        private VulkanBuffer[] _cameraUBOs = new VulkanBuffer[MAX_FRAMES_IN_FLIGHT];
        private IntPtr[] _cameraMapped = new IntPtr[MAX_FRAMES_IN_FLIGHT];
        private VulkanBuffer[] _modelUBOs = new VulkanBuffer[MAX_FRAMES_IN_FLIGHT];
        private IntPtr[] _modelMapped = new IntPtr[MAX_FRAMES_IN_FLIGHT];
        private VulkanBuffer[] _materialUBOs = new VulkanBuffer[MAX_FRAMES_IN_FLIGHT];
        private IntPtr[] _materialMapped = new IntPtr[MAX_FRAMES_IN_FLIGHT];
        private VulkanBuffer[] _shadowDrawUBOs = new VulkanBuffer[MAX_FRAMES_IN_FLIGHT];
        private IntPtr[] _shadowDrawMapped = new IntPtr[MAX_FRAMES_IN_FLIGHT];
        private int _drawSlotCount;

        private VulkanRenderPass _lightingRenderPass;
        private VulkanPipelineLayout _lightingPipelineLayout;
        private VulkanPipeline _lightingPipeline;
        private VulkanFramebuffer[] _lightingFramebuffers;
        private VulkanTexture? _hdrColorTarget;
        private DescriptorSetLayout _lightingDescriptorSetLayout;
        private DescriptorPool _lightingDescriptorPool;
        private DescriptorSet[] _lightingDescriptorSets;
        private Sampler _gbufferSampler;
        private Sampler _shadowSampler;
        private Vector3 _cameraPos = new(0, 2, 8);
        private Vector3 _bakedCameraPos = new(0, 2, 8);
        private float _lightIntensity = 3.2f;
        private VulkanBuffer _fullscreenVertexBuffer;

        private VulkanRenderPass _shadowRenderPass;
        private VulkanPipelineLayout _shadowPipelineLayout;
        private VulkanPipeline _shadowPipeline;
        private VulkanTexture[] _shadowDepthImages;
        private VulkanFramebuffer[] _shadowFramebuffers;
        private DescriptorSetLayout _shadowDescriptorSetLayout;
        private DescriptorPool _shadowDescriptorPool;
        private DescriptorSet[] _shadowDescriptorSets;
        private VulkanBuffer[] _shadowUBOs = new VulkanBuffer[MAX_FRAMES_IN_FLIGHT];
        private IntPtr[] _shadowMapped = new IntPtr[MAX_FRAMES_IN_FLIGHT];

        private VulkanRenderPass _postProcessRenderPass;
        private VulkanPipelineLayout _postProcessPipelineLayout;
        private VulkanPipeline _postProcessPipeline;
        private VulkanFramebuffer[]? _postProcessFramebuffers;
        private DescriptorSetLayout _postProcessDescriptorSetLayout;
        private DescriptorPool _postProcessDescriptorPool;
        private DescriptorSet[] _postProcessDescriptorSets;

        private VulkanBuffer _vertexBuffer;
        private VulkanBuffer _indexBuffer;
        private uint _indexCount;

        private List<SceneMeshData> _sceneMeshes = new();
        private List<SceneLightData> _sceneLights = new();
        private List<SubstrateMaterial> _sceneMaterials = new();
        private readonly List<MeshDraw> _draws = new();
        private readonly Dictionary<Guid, int> _entityProxyMeshes = new();
        private readonly Dictionary<Guid, (Vector3 Center, Vector3 HalfExtents)> _entityBounds = new();
        private readonly Dictionary<Guid, int> _gizmoProxyMeshes = new();
        private GBufferReadback? _gBufferReadback;
        private GBufferSnapshot? _pendingGBufferSnapshot;
        private bool _giUsesGpuReadback;
        private static readonly Guid GizmoGridId = Guid.Parse("00000000-0000-0000-0000-000000000010");
        private static readonly Guid GizmoSelectionId = Guid.Parse("00000000-0000-0000-0000-000000000011");
        private static readonly Guid GizmoAxisXId = Guid.Parse("00000000-0000-0000-0000-000000000012");
        private static readonly Guid GizmoAxisYId = Guid.Parse("00000000-0000-0000-0000-000000000013");
        private static readonly Guid GizmoAxisZId = Guid.Parse("00000000-0000-0000-0000-000000000014");

        private Vector3 _dynamicLightDir = Vector3.Normalize(new Vector3(0.5f, 1f, 0.5f));
        private float _dynamicAmbient = 0.04f;
        private float _giBoost;
        private Vector3 _lastLightingHash;
        private VulkanTexture? _giIrradianceTexture;
        private VulkanTexture? _aoTexture;
        private VulkanTexture? _fogTexture;
        private Vector3[,]? _lastGiIrradiance;

        private int _auxUploadFrame;
        private int _lastFogUploadFrame = -100;
        private float _smoothedExposure = 1.15f;

        private RenderFrameGraph _frameGraph = null!;
        private SceneWorld _sceneWorld = null!;
        private BatchedTextureUploader? _batchedUploader;
        private MaterialTextureSystem? _materialTextures;
        private LdnnComputeSupport? _ldnnCompute;
        private RenderingAlgorithmHub? _algorithmHub;
        private FrameGraphContext? _activeFgContext;
        private Matrix4x4 _prevViewProjection = Matrix4x4.Identity;
        private Matrix4x4 _currentViewProjection = Matrix4x4.Identity;
        private bool _hasPrevViewProjection;
        private readonly Matrix4x4[] _cascadeLightVP = new Matrix4x4[3];
        private readonly Dictionary<int, int> _lodGroupToDraw = new();
        private int _gdnnPresentMeshIndex = -1;
        private VulkanTexture? _taaHistory;
        private VulkanTexture?[]? _drawAlbedoCache;
        private VulkanTexture?[]? _drawNormalCache;
        private VulkanTexture?[]? _drawOrmCache;
        private float[] _extraLightPack = Array.Empty<float>();
        private int _lastVtAtlasUploadFrame = -100;

        private int _width;
        private int _height;
        private bool _initialized;

        /// <summary>Extra PSSM cascades beyond the primary shadow map (0–2), driven by quality.</summary>
        private int _extraShadowCascades = 2;
        private bool _enableGi = true;
        private bool _enableSsao = true;
        private bool _enableBloom = true;
        private bool _enableTaa = true;
        private int _maxLodBias;
        private float _physicsFieldTemperature = 293f;
        private RuntimeRenderQuality? _activeQuality;

        public int Width => _width;
        public int Height => _height;
        public bool IsInitialized => _initialized;
        public SampleCountFlag MSAASamples => _msaaSamples;
        public MaterialBridge Materials => _materialBridge;
        public LDNNBridge GlobalIllumination => _ldnnBridge;
        public PostProcessBridge PostProcess => _postProcessBridge;
        public MeshLoader Meshes => _meshLoader;
        public LodManager LOD => _lodManager;
        /// <summary>GPU-first frame graph driving the present path.</summary>
        public RenderFrameGraph FrameGraph => _frameGraph;
        /// <summary>Scene snapshot shared with FrameGraph passes.</summary>
        public SceneWorld World => _sceneWorld;
        public MaterialTextureSystem? MaterialTextures => _materialTextures;
        public RenderingAlgorithmHub? Algorithms => _algorithmHub;
        public int MeshCount => _sceneMeshes.Count;
        public int TriangleCount => (int)(_indexCount / 3);
        /// <summary>True when the last GI pass consumed GPU readback G-buffer data.</summary>
        public bool GiUsesGpuReadback => _giUsesGpuReadback;
        /// <summary>Hardware / CPU hybrid ray tracing pipeline (null until Initialize).</summary>
        public RayTracingPipeline? RayTracing => _rtPipeline;
        /// <summary>Whether VK_KHR_ray_tracing_pipeline was detected.</summary>
        public bool IsHardwareRayTracingSupported => _rtPipeline?.IsSupported ?? false;

        public SceneRenderer(VulkanRhiDevice rhi)
        {
            _rhi = rhi;
        }

        public void Initialize(int width, int height)
        {
            _width = width;
            _height = height;

            var caps = _rhi.QueryCapabilities();
            _msaaSamples = SampleCountFlag.Count1;
            if (caps.MaxColorSampleCounts.HasFlag(SampleCountFlag.Count4))
                _msaaSamples = SampleCountFlag.Count4;
            else if (caps.MaxColorSampleCounts.HasFlag(SampleCountFlag.Count2))
                _msaaSamples = SampleCountFlag.Count2;

            _materialBridge = new MaterialBridge(_rhi);
            _ldnnBridge = new LDNNBridge(width, height);
            _ldnnBridge.Initialize();
            _ldnnBridge.Resize(width, height);
            _gBufferReadback = new GBufferReadback(_rhi);

            _postProcessBridge = new PostProcessBridge(width, height);
            _meshLoader = new MeshLoader();
            _lodManager = new LodManager();
            _sceneWorld = new SceneWorld { Lod = _lodManager };
            _frameGraph = FrameGraphFactory.CreateDefault();
            _batchedUploader = new BatchedTextureUploader(_rhi);
            _materialTextures = new MaterialTextureSystem(_rhi);
            _ldnnCompute = new LdnnComputeSupport(_rhi);
            _algorithmHub = new RenderingAlgorithmHub(_rhi, width, height);
            _ldnnCompute.AttachDispatcher(_algorithmHub.Compute);

            CreateGBufferRenderPass();
            CreateGBufferResources();
            CreateGBufferFramebuffers();
            CreateGBufferDescriptorResources();
            CreateGBufferPipeline();

            CreateLightingRenderPass();
            CreateHdrColorTargets();
            CreateLightingFramebuffers();
            CreateLightingDescriptorResources();
            CreateLightingPipeline();

            CreateUniformBuffers();

            CreateShadowRenderPass();
            CreateShadowResources();
            CreateShadowFramebuffers();
            CreateShadowDescriptorResources();
            CreateShadowPipeline();

            CreatePostProcessResources();

            _rtPipeline = new RayTracingPipeline(_rhi);
            if (_rtPipeline.IsSupported)
                _rtPipeline.CreateDenoiser(width, height);

            _initialized = true;
            SeedBlackGiTexture();
        }

        private void SeedBlackGiTexture()
        {
            var empty = new Vector3[_width, _height];
            UploadGiIrradianceTexture(empty);
            UploadFogTexture(empty);

            var aoOne = new float[_width, _height];
            for (int y = 0; y < _height; y++)
                for (int x = 0; x < _width; x++)
                    aoOne[x, y] = 1f;
            UploadAoTexture(aoOne);
        }

        private void CreateGBufferRenderPass()
        {
            var attachments = new AttachmentDescription[GBUFFER_ATTACHMENT_COUNT + 1];

            attachments[0] = new AttachmentDescription
            {
                Format = VulkanFormat.R16G16B16A16Sfloat,
                Samples = SampleCountFlag.Count1,
                LoadOp = AttachmentLoadOp.Clear,
                StoreOp = AttachmentStoreOp.Store,
                StencilLoadOp = AttachmentLoadOp.DontCare,
                StencilStoreOp = AttachmentStoreOp.DontCare,
                InitialLayout = ImageLayout.Undefined,
                FinalLayout = ImageLayout.ShaderReadOnlyOptimal
            };

            for (int i = 1; i < GBUFFER_ATTACHMENT_COUNT; i++)
            {
                attachments[i] = new AttachmentDescription
                {
                    Format = VulkanFormat.R16G16B16A16Sfloat,
                    Samples = SampleCountFlag.Count1,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store,
                    StencilLoadOp = AttachmentLoadOp.DontCare,
                    StencilStoreOp = AttachmentStoreOp.DontCare,
                    InitialLayout = ImageLayout.Undefined,
                    FinalLayout = ImageLayout.ShaderReadOnlyOptimal
                };
            }

            attachments[GBUFFER_ATTACHMENT_COUNT] = new AttachmentDescription
            {
                Format = VulkanFormat.D32Sfloat,
                Samples = SampleCountFlag.Count1,
                LoadOp = AttachmentLoadOp.Clear,
                StoreOp = AttachmentStoreOp.DontCare,
                StencilLoadOp = AttachmentLoadOp.DontCare,
                StencilStoreOp = AttachmentStoreOp.DontCare,
                InitialLayout = ImageLayout.Undefined,
                FinalLayout = ImageLayout.DepthStencilAttachmentOptimal
            };

            var colorRefs = new AttachmentReference[GBUFFER_ATTACHMENT_COUNT];
            for (int i = 0; i < GBUFFER_ATTACHMENT_COUNT; i++)
                colorRefs[i] = new AttachmentReference((uint)i, ImageLayout.ColorAttachmentOptimal);

            var subpass = new SubpassDescription
            {
                PipelineBindPoint = PipelineBindPoint.Graphics,
                ColorAttachments = colorRefs,
                DepthStencilAttachment = new AttachmentReference((uint)GBUFFER_ATTACHMENT_COUNT, ImageLayout.DepthStencilAttachmentOptimal)
            };

            var dependency = new SubpassDependency
            {
                SrcSubpass = 0xFFFFFFFF,
                DstSubpass = 0,
                SrcStageMask = PipelineStageFlag.ColorAttachmentOutput | PipelineStageFlag.EarlyFragmentTests,
                DstStageMask = PipelineStageFlag.ColorAttachmentOutput | PipelineStageFlag.EarlyFragmentTests,
                SrcAccessMask = 0,
                DstAccessMask = AccessFlag.ColorAttachmentWrite | AccessFlag.DepthStencilAttachmentWrite,
            };

            _gbufferRenderPass = _rhi.CreateRenderPass(new RenderPassDescription
            {
                Attachments = attachments,
                Subpasses = new[] { subpass },
                Dependencies = new[] { dependency }
            });
        }

        private void CreateGBufferResources()
        {
            var extent = _rhi.Swapchain.Extent;
            var images = _rhi.Swapchain.GetImages();
            int count = images.Length;

            _gbufferAlbedo = new VulkanTexture[count];
            _gbufferNormals = new VulkanTexture[count];
            _gbufferDepth = new VulkanTexture[count];
            _gbufferMaterial = new VulkanTexture[count];
            _gbufferVelocity = new VulkanTexture[count];
            _depthImages = new VulkanTexture[count];

            for (int i = 0; i < count; i++)
            {
                _gbufferAlbedo[i] = CreateGBufferColorTarget(extent);
                _gbufferNormals[i] = CreateGBufferColorTarget(extent);
                _gbufferDepth[i] = CreateGBufferColorTarget(extent);
                _gbufferMaterial[i] = CreateGBufferColorTarget(extent);
                _gbufferVelocity[i] = CreateGBufferColorTarget(extent);

                _depthImages[i] = _rhi.CreateTexture(new TextureDescription
                {
                    Width = extent.Width,
                    Height = extent.Height,
                    Format = VulkanFormat.D32Sfloat,
                    Usage = ImageUsageFlag.DepthStencilAttachment,
                    Tiling = ImageTiling.Optimal,
                    InitialLayout = ImageLayout.Undefined,
                    Samples = SampleCountFlag.Count1,
                });
            }
        }

        private VulkanTexture CreateGBufferColorTarget(Extent2D extent)
            => _rhi.CreateTexture(new TextureDescription
            {
                Width = extent.Width,
                Height = extent.Height,
                Format = VulkanFormat.R16G16B16A16Sfloat,
                Usage = ImageUsageFlag.ColorAttachment | ImageUsageFlag.Sampled | ImageUsageFlag.TransferSrc,
                Tiling = ImageTiling.Optimal,
                InitialLayout = ImageLayout.Undefined,
                Samples = SampleCountFlag.Count1,
            });

        private void CreateGBufferFramebuffers()
        {
            var extent = _rhi.Swapchain.Extent;
            var images = _rhi.Swapchain.GetImages();
            _gbufferFramebuffers = new VulkanFramebuffer[images.Length];

            for (int i = 0; i < images.Length; i++)
            {
                var attachments = new IntPtr[GBUFFER_ATTACHMENT_COUNT + 1];
                attachments[0] = _gbufferAlbedo[i].GetImageView();
                attachments[1] = _gbufferNormals[i].GetImageView();
                attachments[2] = _gbufferDepth[i].GetImageView();
                attachments[3] = _gbufferMaterial[i].GetImageView();
                attachments[4] = _gbufferVelocity[i].GetImageView();
                attachments[5] = _depthImages[i].GetImageView();

                _gbufferFramebuffers[i] = _rhi.CreateFramebuffer(new FramebufferDescription
                {
                    RenderPass = _gbufferRenderPass.Handle,
                    Attachments = attachments,
                    Width = extent.Width,
                    Height = extent.Height,
                    Layers = 1
                });
            }
        }

        private void CreateGBufferDescriptorResources()
        {
            _gbufferDescriptorSetLayout = _rhi.CreateDescriptorSetLayout(new LayoutDescription
            {
                Bindings = new[]
                {
                    new DescriptorSetLayoutBinding { Binding = 0, DescriptorType = DescriptorType.UniformBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlag.Vertex | ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 1, DescriptorType = DescriptorType.UniformBufferDynamic, DescriptorCount = 1, StageFlags = ShaderStageFlag.Vertex },
                    new DescriptorSetLayoutBinding { Binding = 2, DescriptorType = DescriptorType.UniformBufferDynamic, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 3, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 4, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 5, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                }
            });

            _gbufferDescriptorPool = _rhi.CreateDescriptorPool(new PoolDescription
            {
                PoolSizes = new[]
                {
                    new DescriptorPoolSize { Type = DescriptorType.UniformBuffer, DescriptorCount = MAX_FRAMES_IN_FLIGHT },
                    new DescriptorPoolSize { Type = DescriptorType.UniformBufferDynamic, DescriptorCount = MAX_FRAMES_IN_FLIGHT * 2 },
                    new DescriptorPoolSize { Type = DescriptorType.CombinedImageSampler, DescriptorCount = MAX_FRAMES_IN_FLIGHT * 3 },
                },
                MaxSets = MAX_FRAMES_IN_FLIGHT
            });

            var layouts = new IntPtr[MAX_FRAMES_IN_FLIGHT];
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
                layouts[i] = _gbufferDescriptorSetLayout.Handle;

            _gbufferDescriptorSets = _rhi.AllocateDescriptorSets(new DescriptorSetAllocation { Pool = _gbufferDescriptorPool.Handle, Layouts = layouts });
        }

        private void CreateGBufferPipeline()
        {
            var vertSpv = AaaDeferredShaders.CompileGBufferVertex();
            var fragSpv = AaaDeferredShaders.CompileGBufferFragment();

            var vertModule = _rhi.CreateShaderModule(vertSpv);
            var fragModule = _rhi.CreateShaderModule(fragSpv);

            _gbufferPipelineLayout = _rhi.CreatePipelineLayout(new[] { _gbufferDescriptorSetLayout.Handle });

            var shaderStages = new PipelineShaderStageCreateInfo[]
            {
                new PipelineShaderStageCreateInfo { Stage = ShaderStageFlag.Vertex, Module = vertModule.Handle, Name = "main" },
                new PipelineShaderStageCreateInfo { Stage = ShaderStageFlag.Fragment, Module = fragModule.Handle, Name = "main" }
            };

            var colorBlendAttachments = new PipelineColorBlendAttachmentState[GBUFFER_ATTACHMENT_COUNT];
            for (int i = 0; i < GBUFFER_ATTACHMENT_COUNT; i++)
                colorBlendAttachments[i] = PipelineColorBlendAttachmentState.Disabled();

            _gbufferPipeline = _rhi.CreatePipeline(new PipelineDescription
            {
                ShaderStages = shaderStages,
                VertexInputState = new PipelineVertexInputStateCreateInfo
                {
                    VertexBindingDescriptions = new[] { new VertexInputBindingDescription(0, 12 * sizeof(float), VertexInputRate.Vertex) },
                    VertexAttributeDescriptions = new[]
                    {
                        new VertexInputAttributeDescription(0, 0, VulkanFormat.R32G32B32Sfloat, 0),
                        new VertexInputAttributeDescription(1, 0, VulkanFormat.R32G32B32Sfloat, 3 * sizeof(float)),
                        new VertexInputAttributeDescription(2, 0, VulkanFormat.R32G32Sfloat, 6 * sizeof(float)),
                        new VertexInputAttributeDescription(3, 0, VulkanFormat.R32G32B32Sfloat, 8 * sizeof(float)),
                    }
                },
                InputAssemblyState = new PipelineInputAssemblyStateCreateInfo { Topology = GDNN.RHI.Vulkan.PrimitiveTopology.TriangleList },
                TessellationState = new PipelineTessellationStateCreateInfo { PatchControlPoints = 0 },
                ViewportState = new PipelineViewportStateCreateInfo { ViewportCount = 1, ScissorCount = 1 },
                RasterizationState = new PipelineRasterizationStateCreateInfo
                {
                    DepthClampEnable = false,
                    RasterizerDiscardEnable = false,
                    PolygonMode = PolygonMode.Fill,
                    CullMode = CullModeFlag.Back,
                    FrontFace = FrontFace.CounterClockwise,
                    DepthBiasEnable = false,
                    LineWidth = 1.0f
                },
                MultisampleState = new PipelineMultisampleStateCreateInfo { RasterizationSamples = SampleCountFlag.Count1 },
                DepthStencilState = new PipelineDepthStencilStateCreateInfo
                {
                    DepthTestEnable = true,
                    DepthWriteEnable = true,
                    DepthCompareOp = CompareOp.LessOrEqual
                },
                ColorBlendState = new PipelineColorBlendStateCreateInfo { Attachments = colorBlendAttachments },
                DynamicState = new PipelineDynamicStateCreateInfo
                {
                    DynamicStates = new[] { DynamicState.Viewport, DynamicState.Scissor }
                },
                PipelineLayout = _gbufferPipelineLayout.Handle,
                RenderPass = _gbufferRenderPass.Handle,
                Subpass = 0
            });
        }

        private void CreateLightingRenderPass()
        {
            var colorAttachment = new AttachmentDescription
            {
                Format = VulkanFormat.R16G16B16A16Sfloat,
                Samples = SampleCountFlag.Count1,
                LoadOp = AttachmentLoadOp.Clear,
                StoreOp = AttachmentStoreOp.Store,
                StencilLoadOp = AttachmentLoadOp.DontCare,
                StencilStoreOp = AttachmentStoreOp.DontCare,
                InitialLayout = ImageLayout.Undefined,
                FinalLayout = ImageLayout.ShaderReadOnlyOptimal
            };

            var subpass = new SubpassDescription
            {
                PipelineBindPoint = PipelineBindPoint.Graphics,
                ColorAttachments = new[] { new AttachmentReference(0, ImageLayout.ColorAttachmentOptimal) },
            };

            var dependency = new SubpassDependency
            {
                SrcSubpass = 0xFFFFFFFF,
                DstSubpass = 0,
                SrcStageMask = PipelineStageFlag.ColorAttachmentOutput,
                DstStageMask = PipelineStageFlag.ColorAttachmentOutput,
                SrcAccessMask = 0,
                DstAccessMask = AccessFlag.ColorAttachmentWrite,
            };

            _lightingRenderPass = _rhi.CreateRenderPass(new RenderPassDescription
            {
                Attachments = new[] { colorAttachment },
                Subpasses = new[] { subpass },
                Dependencies = new[] { dependency }
            });
        }

        private void CreateHdrColorTargets()
        {
            var extent = _rhi.Swapchain.Extent;
            _hdrColorTarget?.Dispose();
            _hdrColorTarget = _rhi.CreateTexture(new TextureDescription
            {
                Width = extent.Width,
                Height = extent.Height,
                Format = VulkanFormat.R16G16B16A16Sfloat,
                Usage = ImageUsageFlag.ColorAttachment | ImageUsageFlag.Sampled | ImageUsageFlag.TransferSrc | ImageUsageFlag.TransferDst,
                Tiling = ImageTiling.Optimal,
                InitialLayout = ImageLayout.Undefined,
                Samples = SampleCountFlag.Count1,
            });
        }

        private void CreateLightingFramebuffers()
        {
            var extent = _rhi.Swapchain.Extent;
            if (_hdrColorTarget == null)
                CreateHdrColorTargets();

            if (_lightingFramebuffers != null)
            {
                foreach (var fb in _lightingFramebuffers)
                    fb?.Dispose();
            }

            var images = _rhi.Swapchain.GetImages();
            _lightingFramebuffers = new VulkanFramebuffer[images.Length];
            for (int i = 0; i < images.Length; i++)
            {
                _lightingFramebuffers[i] = _rhi.CreateFramebuffer(new FramebufferDescription
                {
                    RenderPass = _lightingRenderPass.Handle,
                    Attachments = new[] { _hdrColorTarget!.GetImageView() },
                    Width = extent.Width,
                    Height = extent.Height,
                    Layers = 1
                });
            }
        }

        private void CreateLightingDescriptorResources()
        {
            CreateGiIrradianceTexture();
            CreateAoTexture();
            CreateFogTexture();

            _lightingDescriptorSetLayout = _rhi.CreateDescriptorSetLayout(new LayoutDescription
            {
                Bindings = new[]
                {
                    new DescriptorSetLayoutBinding { Binding = 0, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 1, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 2, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 3, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 4, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 5, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 6, DescriptorType = DescriptorType.UniformBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 7, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 8, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 9, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 10, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                }
            });

            _lightingDescriptorPool = _rhi.CreateDescriptorPool(new PoolDescription
            {
                PoolSizes = new[]
                {
                    new DescriptorPoolSize { Type = DescriptorType.CombinedImageSampler, DescriptorCount = MAX_FRAMES_IN_FLIGHT * 10 },
                    new DescriptorPoolSize { Type = DescriptorType.UniformBuffer, DescriptorCount = MAX_FRAMES_IN_FLIGHT },
                },
                MaxSets = MAX_FRAMES_IN_FLIGHT
            });

            var layouts = new IntPtr[MAX_FRAMES_IN_FLIGHT];
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
                layouts[i] = _lightingDescriptorSetLayout.Handle;

            _lightingDescriptorSets = _rhi.AllocateDescriptorSets(new DescriptorSetAllocation { Pool = _lightingDescriptorPool.Handle, Layouts = layouts });

            _gbufferSampler = _rhi.CreateSampler(new SamplerDescription
            {
                MagFilter = Filter.Linear,
                MinFilter = Filter.Linear,
                AddressModeU = SamplerAddressMode.ClampToEdge,
                AddressModeV = SamplerAddressMode.ClampToEdge,
                AddressModeW = SamplerAddressMode.ClampToEdge,
                AnisotropyEnable = false,
                MaxAnisotropy = 1.0f,
                CompareEnable = false,
                CompareOp = CompareOp.Always,
                MinLod = 0.0f,
                MaxLod = 1.0f,
            });

            _shadowSampler = _rhi.CreateSampler(new SamplerDescription
            {
                MagFilter = Filter.Nearest,
                MinFilter = Filter.Nearest,
                AddressModeU = SamplerAddressMode.ClampToBorder,
                AddressModeV = SamplerAddressMode.ClampToBorder,
                AddressModeW = SamplerAddressMode.ClampToBorder,
                AnisotropyEnable = false,
                MaxAnisotropy = 1.0f,
                CompareEnable = false,
                CompareOp = CompareOp.Always,
                MinLod = 0.0f,
                MaxLod = 1.0f,
            });
        }

        /// <summary>Binds G-buffer (by swapchain image), GI, shadow map, and light VP for the lighting pass.</summary>
        private void UpdateLightingDescriptors(uint imageIndex, int frameIndex)
        {
            int img = (int)imageIndex;
            if (_gbufferAlbedo == null || img < 0 || img >= _gbufferAlbedo.Length)
                return;
            if (frameIndex < 0 || frameIndex >= MAX_FRAMES_IN_FLIGHT)
                return;

            var shadowView = _shadowDepthImages != null && frameIndex < _shadowDepthImages.Length
                ? _shadowDepthImages[frameIndex].GetImageView()
                : _gbufferDepth[img].GetImageView();

            EnsureShadowCascades();
            WriteCascadeShadowUbo(frameIndex);

            var cascade1View = shadowView;
            var cascade2View = shadowView;
            if (_shadowCascadeDepth != null && _shadowCascadeDepth.Length >= frameIndex * 2 + 2)
            {
                cascade1View = _shadowCascadeDepth[frameIndex * 2].GetImageView();
                cascade2View = _shadowCascadeDepth[frameIndex * 2 + 1].GetImageView();
            }

            _rhi.UpdateDescriptorSets(new[]
            {
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 0, DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _gbufferAlbedo[img].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 1, DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _gbufferNormals[img].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 2, DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _gbufferDepth[img].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 3, DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _gbufferMaterial[img].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 4, DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _giIrradianceTexture?.GetImageView() ?? _gbufferAlbedo[img].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 5, DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[] { new DescriptorImageInfo { Sampler = _shadowSampler.Handle, ImageView = shadowView, ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 6, DstArrayElement = 0,
                    DescriptorType = DescriptorType.UniformBuffer,
                    BufferInfos = new[] { new DescriptorBufferInfo { Buffer = _shadowUBOs[frameIndex].Handle, Offset = 0, Range = (ulong)Marshal.SizeOf<CascadeShadowUBO>() } }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 7, DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _aoTexture?.GetImageView() ?? _gbufferAlbedo[img].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 8, DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _fogTexture?.GetImageView() ?? _gbufferAlbedo[img].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 9, DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[] { new DescriptorImageInfo { Sampler = _shadowSampler.Handle, ImageView = cascade1View, ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _lightingDescriptorSets[frameIndex].Handle,
                    DstBinding = 10, DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[] { new DescriptorImageInfo { Sampler = _shadowSampler.Handle, ImageView = cascade2View, ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                },
            });
        }

        private void CreateGiIrradianceTexture()
        {
            var extent = _rhi.Swapchain.Extent;
            _giIrradianceTexture = _rhi.CreateTexture(new TextureDescription
            {
                Width = extent.Width,
                Height = extent.Height,
                Format = VulkanFormat.R16G16B16A16Sfloat,
                Usage = ImageUsageFlag.Sampled | ImageUsageFlag.TransferDst,
                Tiling = ImageTiling.Optimal,
                InitialLayout = ImageLayout.Undefined,
                Samples = SampleCountFlag.Count1,
            });
        }

        private void CreateAoTexture()
        {
            var extent = _rhi.Swapchain.Extent;
            _aoTexture = _rhi.CreateTexture(new TextureDescription
            {
                Width = extent.Width,
                Height = extent.Height,
                Format = VulkanFormat.R16G16B16A16Sfloat,
                Usage = ImageUsageFlag.Sampled | ImageUsageFlag.TransferDst,
                Tiling = ImageTiling.Optimal,
                InitialLayout = ImageLayout.Undefined,
                Samples = SampleCountFlag.Count1,
            });
        }

        private void CreateFogTexture()
        {
            var extent = _rhi.Swapchain.Extent;
            _fogTexture?.Dispose();
            _fogTexture = _rhi.CreateTexture(new TextureDescription
            {
                Width = extent.Width,
                Height = extent.Height,
                Format = VulkanFormat.R16G16B16A16Sfloat,
                Usage = ImageUsageFlag.Sampled | ImageUsageFlag.TransferDst,
                Tiling = ImageTiling.Optimal,
                InitialLayout = ImageLayout.Undefined,
                Samples = SampleCountFlag.Count1,
            });
        }

        private unsafe void UploadHalfRgbaTexture(VulkanTexture? texture, Vector3[,] field)
        {
            if (!_initialized || texture == null || field == null)
                return;

            int w = field.GetLength(0);
            int h = field.GetLength(1);
            if (w <= 0 || h <= 0 || w != _width || h != _height)
                return;

            int pixelCount = w * h;
            var bytes = new byte[pixelCount * 8];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    var c = field[x, y];
                    int idx = (y * w + x) * 8;
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx, 2), (Half)c.X);
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx + 2, 2), (Half)c.Y);
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx + 4, 2), (Half)c.Z);
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx + 6, 2), (Half)1f);
                }
            }

            UploadHalfBytes(texture, bytes);
        }

        private unsafe void UploadHalfBytes(VulkanTexture texture, byte[] bytes)
        {
            var staging = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)bytes.Length,
                Usage = BufferUsageFlag.TransferSrc,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });
            var mapped = staging.Map();
            Marshal.Copy(bytes, 0, mapped, bytes.Length);
            staging.Unmap();

            var cmd = _rhi.CreateCommandBuffer();
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            cmd.PipelineBarrier(
                PipelineStageFlag.TopOfPipe,
                PipelineStageFlag.Transfer,
                imageBarriers: new[]
                {
                    new ImageMemoryBarrier
                    {
                        Image = texture.Handle,
                        OldLayout = ImageLayout.Undefined,
                        NewLayout = ImageLayout.TransferDstOptimal,
                        DstAccessMask = AccessFlag.TransferWrite,
                        SubresourceRange = new ImageSubresourceRange
                        {
                            AspectMask = ImageAspectFlag.Color,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    }
                });
            cmd.CopyBufferToImage(staging, texture, ImageLayout.TransferDstOptimal, new[]
            {
                new BufferImageCopy
                {
                    BufferOffset = 0,
                    BufferRowLength = 0,
                    BufferImageHeight = 0,
                    ImageSubresource = new ImageSubresourceLayers
                    {
                        AspectMask = ImageAspectFlag.Color,
                        MipLevel = 0,
                        BaseArrayLayer = 0,
                        LayerCount = 1
                    },
                    ImageOffset = new Offset3D(),
                    ImageExtent = new Extent3D((uint)_width, (uint)_height, 1)
                }
            });
            cmd.PipelineBarrier(
                PipelineStageFlag.Transfer,
                PipelineStageFlag.FragmentShader,
                imageBarriers: new[]
                {
                    new ImageMemoryBarrier
                    {
                        Image = texture.Handle,
                        OldLayout = ImageLayout.TransferDstOptimal,
                        NewLayout = ImageLayout.ShaderReadOnlyOptimal,
                        SrcAccessMask = AccessFlag.TransferWrite,
                        DstAccessMask = AccessFlag.ShaderRead,
                        SubresourceRange = new ImageSubresourceRange
                        {
                            AspectMask = ImageAspectFlag.Color,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    }
                });
            cmd.End();
            _rhi.SubmitCommandBuffer(cmd, _rhi.GraphicsQueue, null);
            _rhi.WaitForIdle();
            staging.Dispose();
        }

        private void UploadFogTexture(Vector3[,] fog)
            => UploadHalfRgbaTexture(_fogTexture, fog);

        private unsafe void UploadGiIrradianceTexture(Vector3[,] irradiance)
        {
            if (!_initialized || _giIrradianceTexture == null || irradiance == null)
                return;

            int w = irradiance.GetLength(0);
            int h = irradiance.GetLength(1);
            if (w <= 0 || h <= 0)
                return;

            if (w != _width || h != _height)
                return;

            int pixelCount = w * h;
            var bytes = new byte[pixelCount * 8];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    var c = irradiance[x, y];
                    int idx = (y * w + x) * 8;
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx, 2), (Half)c.X);
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx + 2, 2), (Half)c.Y);
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx + 4, 2), (Half)c.Z);
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx + 6, 2), (Half)1f);
                }
            }

            var staging = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)bytes.Length,
                Usage = BufferUsageFlag.TransferSrc,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });
            var mapped = staging.Map();
            Marshal.Copy(bytes, 0, mapped, bytes.Length);
            staging.Unmap();

            var cmd = _rhi.CreateCommandBuffer();
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            cmd.PipelineBarrier(
                PipelineStageFlag.TopOfPipe,
                PipelineStageFlag.Transfer,
                imageBarriers: new[]
                {
                    new ImageMemoryBarrier
                    {
                        Image = _giIrradianceTexture.Handle,
                        OldLayout = ImageLayout.Undefined,
                        NewLayout = ImageLayout.TransferDstOptimal,
                        DstAccessMask = AccessFlag.TransferWrite,
                        SubresourceRange = new ImageSubresourceRange
                        {
                            AspectMask = ImageAspectFlag.Color,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    }
                });

            cmd.CopyBufferToImage(staging, _giIrradianceTexture, ImageLayout.TransferDstOptimal, new[]
            {
                new BufferImageCopy
                {
                    BufferOffset = 0,
                    ImageSubresource = new ImageSubresourceLayers
                    {
                        AspectMask = ImageAspectFlag.Color,
                        MipLevel = 0,
                        BaseArrayLayer = 0,
                        LayerCount = 1
                    },
                    ImageExtent = new Extent3D((uint)w, (uint)h, 1)
                }
            });

            cmd.PipelineBarrier(
                PipelineStageFlag.Transfer,
                PipelineStageFlag.FragmentShader,
                imageBarriers: new[]
                {
                    new ImageMemoryBarrier
                    {
                        Image = _giIrradianceTexture.Handle,
                        OldLayout = ImageLayout.TransferDstOptimal,
                        NewLayout = ImageLayout.ShaderReadOnlyOptimal,
                        SrcAccessMask = AccessFlag.TransferWrite,
                        DstAccessMask = AccessFlag.ShaderRead,
                        SubresourceRange = new ImageSubresourceRange
                        {
                            AspectMask = ImageAspectFlag.Color,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    }
                });
            cmd.End();
            _rhi.SubmitCommandBuffer(cmd, _rhi.GraphicsQueue, null);
            _rhi.WaitForIdle();
            staging.Dispose();
            _lastGiIrradiance = irradiance;
        }

        private unsafe void UploadAoTexture(float[,] ao)
        {
            if (!_initialized || _aoTexture == null || ao == null)
                return;

            int w = ao.GetLength(0);
            int h = ao.GetLength(1);
            if (w != _width || h != _height || w <= 0 || h <= 0)
                return;

            int pixelCount = w * h;
            var bytes = new byte[pixelCount * 8];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float v = Math.Clamp(ao[x, y], 0f, 1f);
                    int idx = (y * w + x) * 8;
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx, 2), (Half)v);
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx + 2, 2), (Half)v);
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx + 4, 2), (Half)v);
                    BitConverter.TryWriteBytes(bytes.AsSpan(idx + 6, 2), (Half)1f);
                }
            }

            var staging = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)bytes.Length,
                Usage = BufferUsageFlag.TransferSrc,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });
            var mapped = staging.Map();
            Marshal.Copy(bytes, 0, mapped, bytes.Length);
            staging.Unmap();

            var cmd = _rhi.CreateCommandBuffer();
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            cmd.PipelineBarrier(
                PipelineStageFlag.TopOfPipe,
                PipelineStageFlag.Transfer,
                imageBarriers: new[]
                {
                    new ImageMemoryBarrier
                    {
                        Image = _aoTexture.Handle,
                        OldLayout = ImageLayout.Undefined,
                        NewLayout = ImageLayout.TransferDstOptimal,
                        DstAccessMask = AccessFlag.TransferWrite,
                        SubresourceRange = new ImageSubresourceRange
                        {
                            AspectMask = ImageAspectFlag.Color,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    }
                });

            cmd.CopyBufferToImage(staging, _aoTexture, ImageLayout.TransferDstOptimal, new[]
            {
                new BufferImageCopy
                {
                    BufferOffset = 0,
                    ImageSubresource = new ImageSubresourceLayers
                    {
                        AspectMask = ImageAspectFlag.Color,
                        MipLevel = 0,
                        BaseArrayLayer = 0,
                        LayerCount = 1
                    },
                    ImageExtent = new Extent3D((uint)w, (uint)h, 1)
                }
            });

            cmd.PipelineBarrier(
                PipelineStageFlag.Transfer,
                PipelineStageFlag.FragmentShader,
                imageBarriers: new[]
                {
                    new ImageMemoryBarrier
                    {
                        Image = _aoTexture.Handle,
                        OldLayout = ImageLayout.TransferDstOptimal,
                        NewLayout = ImageLayout.ShaderReadOnlyOptimal,
                        SrcAccessMask = AccessFlag.TransferWrite,
                        DstAccessMask = AccessFlag.ShaderRead,
                        SubresourceRange = new ImageSubresourceRange
                        {
                            AspectMask = ImageAspectFlag.Color,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    }
                });
            cmd.End();
            _rhi.SubmitCommandBuffer(cmd, _rhi.GraphicsQueue, null);
            _rhi.WaitForIdle();
            staging.Dispose();
        }

        private void CreateLightingPipeline()
        {
            var dir = _dynamicLightDir;
            PackExtraLights();
            var vertSpv = EmbeddedShaders.CompileLightingVertex();
            var fragSpv = AaaDeferredShaders.CompileLightingFragment(
                dir.X, dir.Y, dir.Z,
                _dynamicAmbient, _giBoost,
                _cameraPos.X, _cameraPos.Y, _cameraPos.Z,
                _lightIntensity,
                bloomStrength: 0.3f,
                extraLights: _extraLightPack);

            _lightingPipeline?.Dispose();

            var vertModule = _rhi.CreateShaderModule(vertSpv);
            var fragModule = _rhi.CreateShaderModule(fragSpv);

            _lightingPipelineLayout = _rhi.CreatePipelineLayout(new[] { _lightingDescriptorSetLayout.Handle });

            var shaderStages = new PipelineShaderStageCreateInfo[]
            {
                new PipelineShaderStageCreateInfo { Stage = ShaderStageFlag.Vertex, Module = vertModule.Handle, Name = "main" },
                new PipelineShaderStageCreateInfo { Stage = ShaderStageFlag.Fragment, Module = fragModule.Handle, Name = "main" }
            };

            _lightingPipeline = _rhi.CreatePipeline(new PipelineDescription
            {
                ShaderStages = shaderStages,
                VertexInputState = new PipelineVertexInputStateCreateInfo
                {
                    VertexBindingDescriptions = Array.Empty<VertexInputBindingDescription>(),
                    VertexAttributeDescriptions = Array.Empty<VertexInputAttributeDescription>()
                },
                InputAssemblyState = new PipelineInputAssemblyStateCreateInfo { Topology = GDNN.RHI.Vulkan.PrimitiveTopology.TriangleList },
                TessellationState = new PipelineTessellationStateCreateInfo { PatchControlPoints = 0 },
                ViewportState = new PipelineViewportStateCreateInfo { ViewportCount = 1, ScissorCount = 1 },
                RasterizationState = new PipelineRasterizationStateCreateInfo
                {
                    DepthClampEnable = false,
                    RasterizerDiscardEnable = false,
                    PolygonMode = PolygonMode.Fill,
                    CullMode = CullModeFlag.Back,
                    FrontFace = FrontFace.CounterClockwise,
                    DepthBiasEnable = false,
                    LineWidth = 1.0f
                },
                MultisampleState = new PipelineMultisampleStateCreateInfo { RasterizationSamples = SampleCountFlag.Count1 },
                DepthStencilState = new PipelineDepthStencilStateCreateInfo
                {
                    DepthTestEnable = false,
                    DepthWriteEnable = false
                },
                ColorBlendState = new PipelineColorBlendStateCreateInfo
                {
                    Attachments = new[] { PipelineColorBlendAttachmentState.Disabled() }
                },
                DynamicState = new PipelineDynamicStateCreateInfo
                {
                    DynamicStates = new[] { DynamicState.Viewport, DynamicState.Scissor }
                },
                PipelineLayout = _lightingPipelineLayout.Handle,
                RenderPass = _lightingRenderPass.Handle,
                Subpass = 0
            });
        }

        private void PackExtraLights()
        {
            // Up to 3 extra lights after the primary directional (8 floats each).
            int extra = Math.Min(3, Math.Max(0, _sceneLights.Count - 1));
            if (_extraLightPack.Length != extra * 8)
                _extraLightPack = new float[extra * 8];
            for (int i = 0; i < extra; i++)
            {
                var L = _sceneLights[i + 1];
                var dir = Vector3.Normalize(-L.Direction);
                // Point-ish: encode position relative to camera via dir*range when Range is finite.
                bool point = L.Range > 0.01f && L.Range < 500f;
                int o = i * 8;
                if (point)
                {
                    var rel = L.Position - _cameraPos;
                    _extraLightPack[o] = rel.X;
                    _extraLightPack[o + 1] = rel.Y;
                    _extraLightPack[o + 2] = rel.Z;
                    _extraLightPack[o + 7] = L.Range;
                }
                else
                {
                    _extraLightPack[o] = dir.X;
                    _extraLightPack[o + 1] = dir.Y;
                    _extraLightPack[o + 2] = dir.Z;
                    _extraLightPack[o + 7] = 0f;
                }
                _extraLightPack[o + 3] = L.Color.X;
                _extraLightPack[o + 4] = L.Color.Y;
                _extraLightPack[o + 5] = L.Color.Z;
                _extraLightPack[o + 6] = Math.Clamp(L.Intensity, 0.2f, 6f);
            }
        }

        private void CreateShadowRenderPass()
        {
            var depthAttachment = new AttachmentDescription
            {
                Format = VulkanFormat.D32Sfloat,
                Samples = SampleCountFlag.Count1,
                LoadOp = AttachmentLoadOp.Clear,
                StoreOp = AttachmentStoreOp.Store,
                StencilLoadOp = AttachmentLoadOp.DontCare,
                StencilStoreOp = AttachmentStoreOp.DontCare,
                InitialLayout = ImageLayout.Undefined,
                FinalLayout = ImageLayout.ShaderReadOnlyOptimal
            };

            var subpass = new SubpassDescription
            {
                PipelineBindPoint = PipelineBindPoint.Graphics,
                DepthStencilAttachment = new AttachmentReference(0, ImageLayout.DepthStencilAttachmentOptimal)
            };

            var dependency = new SubpassDependency
            {
                SrcSubpass = 0xFFFFFFFF,
                DstSubpass = 0,
                SrcStageMask = PipelineStageFlag.EarlyFragmentTests,
                DstStageMask = PipelineStageFlag.EarlyFragmentTests,
                SrcAccessMask = 0,
                DstAccessMask = AccessFlag.DepthStencilAttachmentWrite,
            };

            _shadowRenderPass = _rhi.CreateRenderPass(new RenderPassDescription
            {
                Attachments = new[] { depthAttachment },
                Subpasses = new[] { subpass },
                Dependencies = new[] { dependency }
            });
        }

        private void CreateShadowResources()
        {
            _shadowDepthImages = new VulkanTexture[MAX_FRAMES_IN_FLIGHT];

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _shadowDepthImages[i] = _rhi.CreateTexture(new TextureDescription
                {
                    Width = (uint)SHADOW_MAP_SIZE,
                    Height = (uint)SHADOW_MAP_SIZE,
                    Format = VulkanFormat.D32Sfloat,
                    Usage = ImageUsageFlag.DepthStencilAttachment | ImageUsageFlag.Sampled,
                    Tiling = ImageTiling.Optimal,
                    InitialLayout = ImageLayout.Undefined,
                    Samples = SampleCountFlag.Count1,
                });
            }
        }

        private void CreateShadowFramebuffers()
        {
            _shadowFramebuffers = new VulkanFramebuffer[MAX_FRAMES_IN_FLIGHT];

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _shadowFramebuffers[i] = _rhi.CreateFramebuffer(new FramebufferDescription
                {
                    RenderPass = _shadowRenderPass.Handle,
                    Attachments = new[] { _shadowDepthImages[i].GetImageView() },
                    Width = (uint)SHADOW_MAP_SIZE,
                    Height = (uint)SHADOW_MAP_SIZE,
                    Layers = 1
                });
            }
        }

        private void CreateShadowDescriptorResources()
        {
            _shadowDescriptorSetLayout = _rhi.CreateDescriptorSetLayout(new LayoutDescription
            {
                Bindings = new[]
                {
                    new DescriptorSetLayoutBinding { Binding = 0, DescriptorType = DescriptorType.UniformBufferDynamic, DescriptorCount = 1, StageFlags = ShaderStageFlag.Vertex },
                }
            });

            _shadowDescriptorPool = _rhi.CreateDescriptorPool(new PoolDescription
            {
                PoolSizes = new[]
                {
                    new DescriptorPoolSize { Type = DescriptorType.UniformBufferDynamic, DescriptorCount = MAX_FRAMES_IN_FLIGHT },
                },
                MaxSets = MAX_FRAMES_IN_FLIGHT
            });

            var layouts = new IntPtr[MAX_FRAMES_IN_FLIGHT];
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
                layouts[i] = _shadowDescriptorSetLayout.Handle;

            _shadowDescriptorSets = _rhi.AllocateDescriptorSets(new DescriptorSetAllocation { Pool = _shadowDescriptorPool.Handle, Layouts = layouts });
            BindDrawUniformDescriptors();
        }

        private void CreateUniformBuffers()
        {
            ulong cameraSize = (ulong)Marshal.SizeOf<CameraUBO>();
            ulong shadowSize = (ulong)Marshal.SizeOf<CascadeShadowUBO>();

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _cameraUBOs[i] = _rhi.CreateBuffer(new BufferDescription
                {
                    Size = cameraSize,
                    Usage = BufferUsageFlag.UniformBuffer,
                    MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
                });
                _cameraMapped[i] = _cameraUBOs[i].Map();

                // Light VP for lighting pass (world → shadow clip), not per-draw.
                _shadowUBOs[i] = _rhi.CreateBuffer(new BufferDescription
                {
                    Size = shadowSize,
                    Usage = BufferUsageFlag.UniformBuffer,
                    MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
                });
                _shadowMapped[i] = _shadowUBOs[i].Map();
            }

            EnsureDrawUniformRings(MinDrawSlots);
        }

        /// <summary>Allocates/resizes per-draw dynamic UBO rings (model, material, shadow MVP).</summary>
        private void EnsureDrawUniformRings(int drawCount)
        {
            int slots = Math.Max(MinDrawSlots, drawCount);
            if (slots <= _drawSlotCount && _modelUBOs[0] != null)
                return;

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _modelUBOs[i]?.Unmap();
                _modelUBOs[i]?.Dispose();
                _materialUBOs[i]?.Unmap();
                _materialUBOs[i]?.Dispose();
                _shadowDrawUBOs[i]?.Unmap();
                _shadowDrawUBOs[i]?.Dispose();

                ulong modelBytes = (ulong)(slots * UboAlign);
                _modelUBOs[i] = _rhi.CreateBuffer(new BufferDescription
                {
                    Size = modelBytes,
                    Usage = BufferUsageFlag.UniformBuffer,
                    MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
                });
                _modelMapped[i] = _modelUBOs[i].Map();

                _materialUBOs[i] = _rhi.CreateBuffer(new BufferDescription
                {
                    Size = modelBytes,
                    Usage = BufferUsageFlag.UniformBuffer,
                    MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
                });
                _materialMapped[i] = _materialUBOs[i].Map();

                _shadowDrawUBOs[i] = _rhi.CreateBuffer(new BufferDescription
                {
                    Size = modelBytes,
                    Usage = BufferUsageFlag.UniformBuffer,
                    MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
                });
                _shadowDrawMapped[i] = _shadowDrawUBOs[i].Map();
            }

            _drawSlotCount = slots;
            BindDrawUniformDescriptors();
        }

        private void BindDrawUniformDescriptors()
        {
            if (_gbufferDescriptorSets == null || _modelUBOs[0] == null)
                return;

            ulong modelRange = (ulong)Marshal.SizeOf<Matrix4x4>();
            ulong matRange = (ulong)Marshal.SizeOf<MaterialUBO>();

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _rhi.UpdateDescriptorSets(new[]
                {
                    new DescriptorWrite
                    {
                        DescriptorSet = _gbufferDescriptorSets[i].Handle, DstBinding = 0, DstArrayElement = 0,
                        DescriptorType = DescriptorType.UniformBuffer,
                        BufferInfos = new[] { new DescriptorBufferInfo { Buffer = _cameraUBOs[i].Handle, Offset = 0, Range = (ulong)Marshal.SizeOf<CameraUBO>() } }
                    },
                    new DescriptorWrite
                    {
                        DescriptorSet = _gbufferDescriptorSets[i].Handle, DstBinding = 1, DstArrayElement = 0,
                        DescriptorType = DescriptorType.UniformBufferDynamic,
                        BufferInfos = new[] { new DescriptorBufferInfo { Buffer = _modelUBOs[i].Handle, Offset = 0, Range = modelRange } }
                    },
                    new DescriptorWrite
                    {
                        DescriptorSet = _gbufferDescriptorSets[i].Handle, DstBinding = 2, DstArrayElement = 0,
                        DescriptorType = DescriptorType.UniformBufferDynamic,
                        BufferInfos = new[] { new DescriptorBufferInfo { Buffer = _materialUBOs[i].Handle, Offset = 0, Range = matRange } }
                    },
                });

                if (_shadowDescriptorSets != null && _shadowDrawUBOs[i] != null)
                {
                    _rhi.UpdateDescriptorSets(new[]
                    {
                        new DescriptorWrite
                        {
                            DescriptorSet = _shadowDescriptorSets[i].Handle,
                            DstBinding = 0, DstArrayElement = 0,
                            DescriptorType = DescriptorType.UniformBufferDynamic,
                            BufferInfos = new[] { new DescriptorBufferInfo { Buffer = _shadowDrawUBOs[i].Handle, Offset = 0, Range = modelRange } }
                        }
                    });
                }
            }
        }

        private void CreateShadowPipeline()
        {
            var vertSpv = EmbeddedShaders.CompileShadowVertex();
            var fragSpv = EmbeddedShaders.CompileShadowFragment();

            var vertModule = _rhi.CreateShaderModule(vertSpv);
            var fragModule = _rhi.CreateShaderModule(fragSpv);

            _shadowPipelineLayout = _rhi.CreatePipelineLayout(new[] { _shadowDescriptorSetLayout.Handle });

            var shaderStages = new PipelineShaderStageCreateInfo[]
            {
                new PipelineShaderStageCreateInfo { Stage = ShaderStageFlag.Vertex, Module = vertModule.Handle, Name = "main" },
                new PipelineShaderStageCreateInfo { Stage = ShaderStageFlag.Fragment, Module = fragModule.Handle, Name = "main" }
            };

            _shadowPipeline = _rhi.CreatePipeline(new PipelineDescription
            {
                ShaderStages = shaderStages,
                VertexInputState = new PipelineVertexInputStateCreateInfo
                {
                    VertexBindingDescriptions = new[] { new VertexInputBindingDescription(0, 12 * sizeof(float), VertexInputRate.Vertex) },
                    VertexAttributeDescriptions = new[]
                    {
                        new VertexInputAttributeDescription(0, 0, VulkanFormat.R32G32B32Sfloat, 0),
                    }
                },
                InputAssemblyState = new PipelineInputAssemblyStateCreateInfo { Topology = GDNN.RHI.Vulkan.PrimitiveTopology.TriangleList },
                TessellationState = new PipelineTessellationStateCreateInfo { PatchControlPoints = 0 },
                ViewportState = new PipelineViewportStateCreateInfo { ViewportCount = 1, ScissorCount = 1 },
                RasterizationState = new PipelineRasterizationStateCreateInfo
                {
                    DepthClampEnable = false,
                    RasterizerDiscardEnable = false,
                    PolygonMode = PolygonMode.Fill,
                    CullMode = CullModeFlag.Front,
                    FrontFace = FrontFace.CounterClockwise,
                    DepthBiasEnable = true,
                    DepthBiasConstantFactor = 1.25f,
                    DepthBiasSlopeFactor = 1.75f,
                    DepthBiasClamp = 0.0f,
                    LineWidth = 1.0f
                },
                MultisampleState = new PipelineMultisampleStateCreateInfo { RasterizationSamples = SampleCountFlag.Count1 },
                DepthStencilState = new PipelineDepthStencilStateCreateInfo
                {
                    DepthTestEnable = true,
                    DepthWriteEnable = true,
                    DepthCompareOp = CompareOp.LessOrEqual
                },
                ColorBlendState = new PipelineColorBlendStateCreateInfo { Attachments = Array.Empty<PipelineColorBlendAttachmentState>() },
                DynamicState = new PipelineDynamicStateCreateInfo
                {
                    DynamicStates = new[] { DynamicState.Viewport, DynamicState.Scissor, DynamicState.DepthBias }
                },
                PipelineLayout = _shadowPipelineLayout.Handle,
                RenderPass = _shadowRenderPass.Handle,
                Subpass = 0
            });
        }

        private void CreatePostProcessResources()
        {
            var colorAttachment = new AttachmentDescription
            {
                Format = _rhi.Swapchain.GetSurfaceFormat(),
                Samples = SampleCountFlag.Count1,
                LoadOp = AttachmentLoadOp.Clear,
                StoreOp = AttachmentStoreOp.Store,
                StencilLoadOp = AttachmentLoadOp.DontCare,
                StencilStoreOp = AttachmentStoreOp.DontCare,
                InitialLayout = ImageLayout.Undefined,
                FinalLayout = ImageLayout.PresentSrcKHR
            };

            var subpass = new SubpassDescription
            {
                PipelineBindPoint = PipelineBindPoint.Graphics,
                ColorAttachments = new[] { new AttachmentReference(0, ImageLayout.ColorAttachmentOptimal) },
            };

            var dependency = new SubpassDependency
            {
                SrcSubpass = 0xFFFFFFFF,
                DstSubpass = 0,
                SrcStageMask = PipelineStageFlag.ColorAttachmentOutput | PipelineStageFlag.FragmentShader,
                DstStageMask = PipelineStageFlag.ColorAttachmentOutput,
                SrcAccessMask = AccessFlag.ShaderRead | AccessFlag.ColorAttachmentWrite,
                DstAccessMask = AccessFlag.ColorAttachmentWrite,
            };

            _postProcessRenderPass = _rhi.CreateRenderPass(new RenderPassDescription
            {
                Attachments = new[] { colorAttachment },
                Subpasses = new[] { subpass },
                Dependencies = new[] { dependency }
            });

            var extent = _rhi.Swapchain.Extent;
            var images = _rhi.Swapchain.GetImages();
            _postProcessFramebuffers = new VulkanFramebuffer[images.Length];
            for (int i = 0; i < images.Length; i++)
            {
                _postProcessFramebuffers[i] = _rhi.CreateFramebuffer(new FramebufferDescription
                {
                    RenderPass = _postProcessRenderPass.Handle,
                    Attachments = new[] { _rhi.Swapchain.GetImageView((uint)i) },
                    Width = extent.Width,
                    Height = extent.Height,
                    Layers = 1
                });
            }

            _postProcessDescriptorSetLayout = _rhi.CreateDescriptorSetLayout(new LayoutDescription
            {
                Bindings = new[]
                {
                    new DescriptorSetLayoutBinding
                    {
                        Binding = 0,
                        DescriptorType = DescriptorType.CombinedImageSampler,
                        DescriptorCount = 1,
                        StageFlags = ShaderStageFlag.Fragment
                    },
                    new DescriptorSetLayoutBinding
                    {
                        Binding = 1,
                        DescriptorType = DescriptorType.CombinedImageSampler,
                        DescriptorCount = 1,
                        StageFlags = ShaderStageFlag.Fragment
                    },
                    new DescriptorSetLayoutBinding
                    {
                        Binding = 2,
                        DescriptorType = DescriptorType.CombinedImageSampler,
                        DescriptorCount = 1,
                        StageFlags = ShaderStageFlag.Fragment
                    },
                }
            });

            _postProcessDescriptorPool = _rhi.CreateDescriptorPool(new PoolDescription
            {
                PoolSizes = new[]
                {
                    new DescriptorPoolSize { Type = DescriptorType.CombinedImageSampler, DescriptorCount = MAX_FRAMES_IN_FLIGHT * 3 }
                },
                MaxSets = MAX_FRAMES_IN_FLIGHT
            });

            var layouts = new IntPtr[MAX_FRAMES_IN_FLIGHT];
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
                layouts[i] = _postProcessDescriptorSetLayout.Handle;
            _postProcessDescriptorSets = _rhi.AllocateDescriptorSets(new DescriptorSetAllocation
            {
                Pool = _postProcessDescriptorPool.Handle,
                Layouts = layouts
            });

            _postProcessPipelineLayout = _rhi.CreatePipelineLayout(new[] { _postProcessDescriptorSetLayout.Handle });

            EnsureTaaHistory();

            var vertSpv = EmbeddedShaders.CompilePostProcessVertex();
            float tw = _width > 0 ? 1f / _width : 1f / 1920f;
            float th = _height > 0 ? 1f / _height : 1f / 1080f;
            var tonemapSpv = AaaDeferredShaders.CompileHdrPostFragment(0.45f, _smoothedExposure, tw, th);
            var vertModule = _rhi.CreateShaderModule(vertSpv);
            var tonemapModule = _rhi.CreateShaderModule(tonemapSpv);

            _postProcessPipeline = CreateFullscreenPipeline(vertModule.Handle, tonemapModule.Handle, _postProcessPipelineLayout.Handle, _postProcessRenderPass.Handle);
            _lastPostExposure = _smoothedExposure;
        }

        private void EnsureTaaHistory()
        {
            var extent = _rhi.Swapchain.Extent;
            if (_taaHistory != null && _taaHistory.Width == extent.Width && _taaHistory.Height == extent.Height)
                return;
            _taaHistory?.Dispose();
            _taaHistory = _rhi.CreateTexture(new TextureDescription
            {
                Width = extent.Width,
                Height = extent.Height,
                Format = VulkanFormat.R16G16B16A16Sfloat,
                Usage = ImageUsageFlag.Sampled | ImageUsageFlag.TransferDst | ImageUsageFlag.TransferSrc | ImageUsageFlag.ColorAttachment,
                Tiling = ImageTiling.Optimal,
                InitialLayout = ImageLayout.Undefined,
                Samples = SampleCountFlag.Count1,
            });
        }

        private float _lastPostExposure = 1.15f;

        private void MaybeRebuildPostProcessPipeline()
        {
            if (_postProcessPipeline == null || _postProcessPipelineLayout == null || _postProcessRenderPass == null)
                return;
            if (MathF.Abs(_smoothedExposure - _lastPostExposure) < 0.04f)
                return;

            float tw = _width > 0 ? 1f / _width : 1f / 1920f;
            float th = _height > 0 ? 1f / _height : 1f / 1080f;
            var tonemapSpv = AaaDeferredShaders.CompileHdrPostFragment(0.45f, _smoothedExposure, tw, th);
            var vertSpv = EmbeddedShaders.CompilePostProcessVertex();
            var vertModule = _rhi.CreateShaderModule(vertSpv);
            var tonemapModule = _rhi.CreateShaderModule(tonemapSpv);
            _postProcessPipeline.Dispose();
            _postProcessPipeline = CreateFullscreenPipeline(vertModule.Handle, tonemapModule.Handle, _postProcessPipelineLayout.Handle, _postProcessRenderPass.Handle);
            _lastPostExposure = _smoothedExposure;
        }

        private VulkanPipeline CreateFullscreenPipeline(IntPtr vertModule, IntPtr fragModule, IntPtr layout, IntPtr renderPass)
        {
            return _rhi.CreatePipeline(new PipelineDescription
            {
                ShaderStages = new[]
                {
                    new PipelineShaderStageCreateInfo { Stage = ShaderStageFlag.Vertex, Module = vertModule, Name = "main" },
                    new PipelineShaderStageCreateInfo { Stage = ShaderStageFlag.Fragment, Module = fragModule, Name = "main" }
                },
                VertexInputState = new PipelineVertexInputStateCreateInfo
                {
                    VertexBindingDescriptions = Array.Empty<VertexInputBindingDescription>(),
                    VertexAttributeDescriptions = Array.Empty<VertexInputAttributeDescription>()
                },
                InputAssemblyState = new PipelineInputAssemblyStateCreateInfo { Topology = GDNN.RHI.Vulkan.PrimitiveTopology.TriangleList },
                TessellationState = new PipelineTessellationStateCreateInfo { PatchControlPoints = 0 },
                ViewportState = new PipelineViewportStateCreateInfo { ViewportCount = 1, ScissorCount = 1 },
                RasterizationState = new PipelineRasterizationStateCreateInfo
                {
                    DepthClampEnable = false,
                    RasterizerDiscardEnable = false,
                    PolygonMode = PolygonMode.Fill,
                    CullMode = CullModeFlag.None,
                    FrontFace = FrontFace.CounterClockwise,
                    DepthBiasEnable = false,
                    LineWidth = 1.0f
                },
                MultisampleState = new PipelineMultisampleStateCreateInfo { RasterizationSamples = SampleCountFlag.Count1 },
                DepthStencilState = new PipelineDepthStencilStateCreateInfo
                {
                    DepthTestEnable = false,
                    DepthWriteEnable = false
                },
                ColorBlendState = new PipelineColorBlendStateCreateInfo
                {
                    Attachments = new[] { PipelineColorBlendAttachmentState.Disabled() }
                },
                DynamicState = new PipelineDynamicStateCreateInfo
                {
                    DynamicStates = new[] { DynamicState.Viewport, DynamicState.Scissor }
                },
                PipelineLayout = layout,
                RenderPass = renderPass,
                Subpass = 0
            });
        }

        private void UpdatePostProcessDescriptors(int frameIndex)
        {
            if (_postProcessDescriptorSets == null || _hdrColorTarget == null)
                return;
            if (frameIndex < 0 || frameIndex >= MAX_FRAMES_IN_FLIGHT)
                return;

            EnsureTaaHistory();
            int img = Math.Clamp(frameIndex, 0, (_gbufferVelocity?.Length ?? 1) - 1);
            var velView = _gbufferVelocity != null && img < _gbufferVelocity.Length
                ? _gbufferVelocity[img].GetImageView()
                : _hdrColorTarget.GetImageView();
            var histView = _taaHistory?.GetImageView() ?? _hdrColorTarget.GetImageView();

            _rhi.UpdateDescriptorSets(new[]
            {
                new DescriptorWrite
                {
                    DescriptorSet = _postProcessDescriptorSets[frameIndex].Handle,
                    DstBinding = 0,
                    DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[]
                    {
                        new DescriptorImageInfo
                        {
                            Sampler = _gbufferSampler.Handle,
                            ImageView = _hdrColorTarget.GetImageView(),
                            ImageLayout = ImageLayout.ShaderReadOnlyOptimal
                        }
                    }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _postProcessDescriptorSets[frameIndex].Handle,
                    DstBinding = 1,
                    DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[]
                    {
                        new DescriptorImageInfo
                        {
                            Sampler = _gbufferSampler.Handle,
                            ImageView = velView,
                            ImageLayout = ImageLayout.ShaderReadOnlyOptimal
                        }
                    }
                },
                new DescriptorWrite
                {
                    DescriptorSet = _postProcessDescriptorSets[frameIndex].Handle,
                    DstBinding = 2,
                    DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    ImageInfos = new[]
                    {
                        new DescriptorImageInfo
                        {
                            Sampler = _gbufferSampler.Handle,
                            ImageView = histView,
                            ImageLayout = ImageLayout.ShaderReadOnlyOptimal
                        }
                    }
                }
            });
        }

        public int AddMesh(float[] vertices, int vertexStride, uint[] indices, int materialIndex = 0)
        {
            var mesh = new SceneMeshData
            {
                VertexData = vertices,
                VertexStride = vertexStride,
                IndexData = indices,
                MaterialIndex = materialIndex,
                WorldMatrix = Matrix4x4.Identity
            };
            _sceneMeshes.Add(mesh);
            return _sceneMeshes.Count - 1;
        }

        public int AddMaterial(SubstrateMaterial material)
        {
            _sceneMaterials.Add(material);
            return _sceneMaterials.Count - 1;
        }

        public void SetMeshWorldMatrix(int meshIndex, Matrix4x4 worldMatrix)
        {
            if (meshIndex >= 0 && meshIndex < _sceneMeshes.Count)
                _sceneMeshes[meshIndex].WorldMatrix = worldMatrix;
        }

        public int AddLight(Vector3 position, Vector3 direction, Vector3 color, float intensity, float range = 100f)
        {
            var light = new SceneLightData
            {
                Position = position,
                Direction = direction,
                Color = color,
                Intensity = intensity,
                Range = range
            };
            _sceneLights.Add(light);
            return _sceneLights.Count - 1;
        }

        /// <summary>
        /// Applies LLM lighting params to the deferred scene lights and L-DNN bridge.
        /// </summary>
        public void ApplyLlmLighting(LightingParams parameters)
        {
            ArgumentNullException.ThrowIfNull(parameters);
            if (!_initialized)
                return;

            var lights = LlmSceneApplicator.ApplyLighting(parameters);
            foreach (var light in lights)
            {
                if (light.Type == LightType.Directional)
                {
                    AddLight(
                        position: -light.Direction * 20f,
                        direction: light.Direction,
                        color: light.Color,
                        intensity: light.Intensity);
                }
            }

            _ldnnBridge.ApplyLlmLighting(parameters);
            RefreshDynamicLightingFromScene();
        }

        /// <summary>Creates or updates a visible proxy mesh for a scene entity (Genome, Volume, Character, Mesh).</summary>
        public void SyncEntityProxy(Guid id, string type, Vector3 position, Vector3 scale, Vector3 rotationEuler = default)
        {
            if (!_initialized)
                return;
            if (type.Equals("Light", StringComparison.OrdinalIgnoreCase) ||
                type.Equals("Camera", StringComparison.OrdinalIgnoreCase) ||
                type.Equals("Empty", StringComparison.OrdinalIgnoreCase))
                return;

            float sx = Math.Max(scale.X, 0.05f);
            float sy = Math.Max(scale.Y, 0.05f);
            float sz = Math.Max(scale.Z, 0.05f);
            var rot = Matrix4x4.CreateRotationX(MathHelper.Deg2Rad(rotationEuler.X))
                    * Matrix4x4.CreateRotationY(MathHelper.Deg2Rad(rotationEuler.Y))
                    * Matrix4x4.CreateRotationZ(MathHelper.Deg2Rad(rotationEuler.Z));
            var world = Matrix4x4.CreateScale(sx, sy, sz) * rot * Matrix4x4.CreateTranslation(position);

            _entityBounds[id] = (position, new Vector3(sx * 0.5f, sy * 0.5f, sz * 0.5f));

            int materialIndex = type.ToUpperInvariant() switch
            {
                "GENOME" => 3,
                "VOLUME" => 2,
                "CHARACTER" => 1,
                _ => 0
            };

            if (_entityProxyMeshes.TryGetValue(id, out int meshIdx))
            {
                SetMeshWorldMatrix(meshIdx, world);
                return;
            }

            meshIdx = CreateCube(materialIndex, world);
            _entityProxyMeshes[id] = meshIdx;
            UploadSceneGeometry();
        }

        public bool TryGetEntityBounds(Guid id, out Vector3 center, out Vector3 halfExtents)
        {
            if (_entityBounds.TryGetValue(id, out var b))
            {
                center = b.Center;
                halfExtents = b.HalfExtents;
                return true;
            }
            center = default;
            halfExtents = default;
            return false;
        }
        public void SyncEditorGizmos(bool showGrid, bool showGizmos, Guid selectedEntityId, Vector3 selectedPosition, Vector3 selectedScale)
        {
            if (!_initialized)
                return;

            ClearGizmoProxies();

            if (showGrid)
            {
                var gridWorld = Matrix4x4.CreateScale(20f, 1f, 20f) * Matrix4x4.CreateTranslation(0, 0, 0);
                AddGizmoProxy(GizmoGridId, CreatePlaneMesh(0, gridWorld));
            }

            if (showGizmos && selectedEntityId != Guid.Empty)
            {
                float axisLen = MathF.Max(selectedScale.X, MathF.Max(selectedScale.Y, selectedScale.Z)) * 0.75f + 0.5f;
                var selWorld = Matrix4x4.CreateScale(selectedScale.X * 1.05f, selectedScale.Y * 1.05f, selectedScale.Z * 1.05f)
                               * Matrix4x4.CreateTranslation(selectedPosition);
                AddGizmoProxy(GizmoSelectionId, CreateCube(2, selWorld));

                AddGizmoProxy(GizmoAxisXId, CreateCube(1,
                    Matrix4x4.CreateScale(axisLen, 0.06f, 0.06f) * Matrix4x4.CreateTranslation(selectedPosition + Vector3.UnitX * axisLen * 0.5f)));
                AddGizmoProxy(GizmoAxisYId, CreateCube(3,
                    Matrix4x4.CreateScale(0.06f, axisLen, 0.06f) * Matrix4x4.CreateTranslation(selectedPosition + Vector3.UnitY * axisLen * 0.5f)));
                AddGizmoProxy(GizmoAxisZId, CreateCube(0,
                    Matrix4x4.CreateScale(0.06f, 0.06f, axisLen) * Matrix4x4.CreateTranslation(selectedPosition + Vector3.UnitZ * axisLen * 0.5f)));
            }

            UploadSceneGeometry();
        }

        private void ClearGizmoProxies()
        {
            foreach (var kv in _gizmoProxyMeshes)
            {
                if (kv.Value >= 0 && kv.Value < _sceneMeshes.Count)
                    _sceneMeshes[kv.Value].WorldMatrix = Matrix4x4.CreateScale(0);
            }
            _gizmoProxyMeshes.Clear();
        }

        private void AddGizmoProxy(Guid id, int meshIdx)
        {
            _gizmoProxyMeshes[id] = meshIdx;
        }

        /// <summary>Reads GPU G-buffer attachments from the previous submitted frame.</summary>
        public void ConsumePendingGBufferReadback()
        {
            if (_pendingGBufferSnapshot == null || !_initialized)
                return;

            var snap = _pendingGBufferSnapshot;
            _pendingGBufferSnapshot = null;
            _ldnnBridge.IngestGpuSnapshot(snap);
            _giUsesGpuReadback = true;
        }

        /// <summary>Schedules a full G-buffer GPU readback after the frame fence signals.</summary>
        public void ScheduleGBufferReadback(uint imageIndex, int frameIndex)
        {
            if (!_initialized || _gBufferReadback == null)
                return;
            if (_gbufferAlbedo == null || imageIndex >= _gbufferAlbedo.Length)
                return;

            int w = (int)_rhi.Swapchain.Extent.Width;
            int h = (int)_rhi.Swapchain.Extent.Height;
            if (_gBufferReadback.TryRead(
                    _gbufferAlbedo[imageIndex],
                    _gbufferNormals[imageIndex],
                    _gbufferDepth[imageIndex],
                    _gbufferMaterial[imageIndex],
                    _gbufferVelocity[imageIndex],
                    w, h,
                    out var snapshot))
            {
                _pendingGBufferSnapshot = snapshot;
            }
        }

        /// <summary>Copies deferred scene lights into the L-DNN bridge and refreshes GPU lighting.</summary>
        public void PushLightsToGlobalIllumination()
        {
            if (!_initialized)
                return;

            var configs = new List<LightConfig>();
            foreach (var light in _sceneLights)
            {
                configs.Add(new LightConfig
                {
                    Type = LightType.Directional,
                    Direction = Vector3.Normalize(light.Direction),
                    Color = light.Color,
                    Intensity = light.Intensity,
                    Range = light.Range,
                    ShadowMethod = ShadowMethod.NeuralPredictive,
                    ShadowBias = 0.005f,
                    ShadowSamples = 16,
                    Importance = 1.0f
                });
            }

            _ldnnBridge.SetLights(configs);
            RefreshDynamicLightingFromScene();
        }

        private void RefreshDynamicLightingFromScene()
        {
            if (_sceneLights.Count == 0)
                return;

            var primary = _sceneLights[0];
            _dynamicLightDir = Vector3.Normalize(-primary.Direction);
            _lightIntensity = Math.Clamp(primary.Intensity, 0.5f, 8f);
            PackExtraLights();
            RebuildLightingPipelineIfNeeded();
        }

        private void RebuildLightingPipelineIfNeeded()
        {
            var hash = new Vector3(
                _dynamicLightDir.X + _cameraPos.X * 0.01f + _sceneLights.Count * 0.1f,
                _dynamicAmbient + _giBoost + _lightIntensity * 0.1f,
                _extraLightPack.Length * 0.01f + _dynamicLightDir.Z + _cameraPos.Z * 0.01f);
            if (Vector3.DistanceSquared(hash, _lastLightingHash) < 1e-8f)
                return;
            _lastLightingHash = hash;
            CreateLightingPipeline();
        }

        public bool LoadMeshFromFile(string filePath, int materialIndex = 0, Matrix4x4? worldMatrix = null)
        {
            if (!File.Exists(filePath))
                return false;

            var asset = _meshLoader.LoadSync(filePath);
            if (asset == null)
                return false;

            foreach (var prim in asset.Primitives)
            {
                var vertices = new float[prim.Vertices.Count * 12];
                for (int v = 0; v < prim.Vertices.Count; v++)
                {
                    var mv = prim.Vertices[v];
                    int offset = v * 12;
                    vertices[offset + 0] = mv.Position.X;
                    vertices[offset + 1] = mv.Position.Y;
                    vertices[offset + 2] = mv.Position.Z;
                    vertices[offset + 3] = mv.Normal.X;
                    vertices[offset + 4] = mv.Normal.Y;
                    vertices[offset + 5] = mv.Normal.Z;
                    vertices[offset + 6] = mv.TexCoord0.X;
                    vertices[offset + 7] = mv.TexCoord0.Y;
                    vertices[offset + 8] = mv.Color0.X;
                    vertices[offset + 9] = mv.Color0.Y;
                    vertices[offset + 10] = mv.Color0.Z;
                    vertices[offset + 11] = 1.0f;
                }

                var indices = prim.Indices.ToArray();
                int meshIdx = AddMesh(vertices, 12, indices, materialIndex);
                SetMeshWorldMatrix(meshIdx, worldMatrix ?? Matrix4x4.Identity);

                if (asset.Materials.Count > prim.MaterialIndex)
                {
                    var meshMat = asset.Materials[prim.MaterialIndex];
                    var subMat = new SubstrateMaterial(meshMat.Name);
                    subMat.InitializeDefaults();
                    subMat.SetProperty("BaseColor", new Color3(meshMat.BaseColor.X, meshMat.BaseColor.Y, meshMat.BaseColor.Z));
                    subMat.SetProperty("Roughness", meshMat.Roughness);
                    subMat.SetProperty("Metallic", meshMat.Metallic);
                    subMat.SetProperty("EmissiveIntensity", meshMat.EmissiveIntensity);
                    _sceneMeshes[^1].MaterialIndex = AddMaterial(subMat);
                }
            }

            return true;
        }

        public void UploadSceneGeometry()
        {
            if (_sceneMeshes.Count == 0)
                return;

            int totalVertices = 0;
            int totalIndices = 0;
            foreach (var mesh in _sceneMeshes)
            {
                totalVertices += mesh.VertexData.Length / mesh.VertexStride;
                totalIndices += mesh.IndexData?.Length ?? 0;
            }

            var allVertices = new float[totalVertices * 12];
            var allIndices = new uint[totalIndices];
            _draws.Clear();
            _lodGroupToDraw.Clear();
            _lodManager?.Clear();
            int vOffset = 0;
            int iOffset = 0;
            int vertexOffset = 0;
            int meshIndex = 0;

            foreach (var mesh in _sceneMeshes)
            {
                int meshVertexCount = mesh.VertexData.Length / mesh.VertexStride;
                Array.Copy(mesh.VertexData, 0, allVertices, vOffset, mesh.VertexData.Length);
                vOffset += mesh.VertexData.Length;

                int firstIndex = iOffset;
                int indexCount = 0;
                if (mesh.IndexData != null)
                {
                    indexCount = mesh.IndexData.Length;
                    for (int i = 0; i < mesh.IndexData.Length; i++)
                        allIndices[iOffset + i] = mesh.IndexData[i] + (uint)vertexOffset;
                    iOffset += mesh.IndexData.Length;
                }

                if (indexCount > 0)
                {
                    int drawIndex = _draws.Count;
                    _draws.Add(new MeshDraw
                    {
                        MeshIndex = meshIndex,
                        FirstIndex = (uint)firstIndex,
                        IndexCount = (uint)indexCount,
                        MaterialIndex = mesh.MaterialIndex,
                        WorldMatrix = mesh.WorldMatrix
                    });

                    // Wire LodManager → draw index ranges (LOD0 full, LOD1 half indices).
                    var lodGroup = _lodManager.RegisterGroup($"mesh_{meshIndex}", 2f);
                    lodGroup.LocalReferencePoint = mesh.WorldMatrix.Translation;
                    lodGroup.Levels.Add(new LodLevel
                    {
                        Level = 0,
                        ScreenCoverageThreshold = 0.15f,
                        MinDistance = 0f,
                        MaxDistance = 40f,
                        TriangleCount = indexCount / 3,
                        MeshData = new LodMeshRange((uint)firstIndex, (uint)indexCount)
                    });
                    lodGroup.Levels.Add(new LodLevel
                    {
                        Level = 1,
                        ScreenCoverageThreshold = 0.04f,
                        MinDistance = 40f,
                        MaxDistance = 500f,
                        TriangleCount = Math.Max(1, indexCount / 6),
                        MeshData = new LodMeshRange((uint)firstIndex, (uint)Math.Max(3, (indexCount / 6) * 3))
                    });
                    _lodGroupToDraw[lodGroup.Id] = drawIndex;
                }

                vertexOffset += meshVertexCount;
                meshIndex++;
            }

            _indexCount = (uint)totalIndices;
            EnsureDrawUniformRings(_draws.Count);

            var vertexBytes = MemoryMarshal.AsBytes(allVertices.AsSpan());
            _vertexBuffer = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)vertexBytes.Length,
                Usage = BufferUsageFlag.VertexBuffer | BufferUsageFlag.TransferDst,
                MemoryProperties = MemoryPropertyFlag.DeviceLocal
            });

            var indexBytes = MemoryMarshal.AsBytes(allIndices.AsSpan());
            _indexBuffer = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)indexBytes.Length,
                Usage = BufferUsageFlag.IndexBuffer | BufferUsageFlag.TransferDst,
                MemoryProperties = MemoryPropertyFlag.DeviceLocal
            });

            UploadToGpu(_vertexBuffer, vertexBytes);
            UploadToGpu(_indexBuffer, indexBytes);
        }

        private unsafe void UploadToGpu(VulkanBuffer dst, ReadOnlySpan<byte> data)
        {
            var staging = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)data.Length,
                Usage = BufferUsageFlag.TransferSrc,
                MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
            });

            var mapped = staging.Map();
            fixed (byte* src = data)
            { System.Buffer.MemoryCopy(src, (void*)mapped, data.Length, data.Length); }
            staging.Unmap();

            var cmd = _rhi.CreateCommandBuffer();
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            cmd.CopyBuffer(staging, dst, new[] { new BufferCopy(0, 0, (ulong)data.Length) });
            cmd.End();
            _rhi.SubmitCommandBuffer(cmd, _rhi.GraphicsQueue, null);
            _rhi.WaitForIdle();
            staging.Dispose();
        }

        public unsafe void UpdateUniforms(int frameIndex, Matrix4x4 view, Matrix4x4 projection, Vector3 cameraPos, float time)
        {
            _cameraPos = cameraPos;
            var extent = _rhi.Swapchain.Extent;
            var camUBO = _materialBridge.BuildCameraUBO(view, projection, cameraPos, time, extent.Width, extent.Height);
            // Camera binding 0: { mat4 ViewProjection; mat4 PrevViewProjection } for velocity.
            var vp = camUBO.ViewProjection;
            var prevVp = _hasPrevViewProjection ? _prevViewProjection : vp;
            Marshal.StructureToPtr(vp, _cameraMapped[frameIndex], false);
            Marshal.StructureToPtr(prevVp, IntPtr.Add(_cameraMapped[frameIndex], 64), false);

            var lightVP = Matrix4x4.Identity;
            Matrix4x4 lightView = Matrix4x4.Identity;
            Matrix4x4 lightProj = Matrix4x4.Identity;
            if (_sceneLights.Count > 0)
            {
                var light = _sceneLights[0];
                var lightDir = Vector3.Normalize(light.Direction);
                // Camera-centered ortho (PSSM-lite): near-field shadow resolution stays high.
                var focus = cameraPos;
                var lightPos = focus - lightDir * 35.0f;
                var up = MathF.Abs(Vector3.Dot(lightDir, Vector3.UnitY)) > 0.95f ? Vector3.UnitZ : Vector3.UnitY;
                lightView = Matrix4x4.CreateLookAt(lightPos, focus, up);
                lightProj = Matrix4x4.CreateOrthographic(28.0f, 28.0f, 0.5f, 90.0f);
                lightProj.M22 *= -1;
                lightVP = lightView * lightProj;
            }
            _cascadeLightVP[0] = lightVP;
            WriteCascadeShadowUbo(frameIndex);

            // G-DNN LOD selection for registered groups.
            _lodManager?.UpdateAll(cameraPos, MathF.PI / 3f, extent.Height);
            ApplySelectedLodDraws();

            EnsureDrawUniformRings(Math.Max(1, _draws.Count));
            if (_drawAlbedoCache == null || _drawAlbedoCache.Length < _draws.Count)
            {
                _drawAlbedoCache = new VulkanTexture[_draws.Count];
                _drawNormalCache = new VulkanTexture[_draws.Count];
                _drawOrmCache = new VulkanTexture[_draws.Count];
            }
            int drawCount = Math.Min(_draws.Count, _drawSlotCount);
            for (int d = 0; d < drawCount; d++)
            {
                var draw = _draws[d];
                if (draw.MeshIndex >= 0 && draw.MeshIndex < _sceneMeshes.Count)
                {
                    draw.WorldMatrix = _sceneMeshes[draw.MeshIndex].WorldMatrix;
                    draw.MaterialIndex = _sceneMeshes[draw.MeshIndex].MaterialIndex;
                    _draws[d] = draw;
                }

                var modelPtr = IntPtr.Add(_modelMapped[frameIndex], d * UboAlign);
                Marshal.StructureToPtr(draw.WorldMatrix, modelPtr, false);

                MaterialUBO matUBO;
                SubstrateMaterial? subMat = null;
                if (draw.MaterialIndex >= 0 && draw.MaterialIndex < _sceneMaterials.Count)
                {
                    subMat = _sceneMaterials[draw.MaterialIndex];
                    matUBO = _materialBridge.ExtractProperties(subMat);
                    _drawAlbedoCache![d] = _materialTextures?.ResolveAlbedo(subMat);
                    _drawNormalCache![d] = _materialTextures?.ResolveNormal(subMat);
                    // Floor / material 0: prefer VT atlas when resident tiles exist.
                    if (d == 0 && _materialTextures?.VirtualTextureAtlas != null &&
                        (_algorithmHub?.VirtualTextures.ResidentTiles ?? 0) > 0)
                        _drawAlbedoCache[d] = _materialTextures.VirtualTextureAtlas;
                    _drawOrmCache![d] = _materialTextures?.ResolveOrm(subMat);
                }
                else
                {
                    matUBO = DefaultMaterialUbo();
                    _drawAlbedoCache![d] = _materialTextures?.White;
                    _drawNormalCache![d] = _materialTextures?.FlatNormal;
                    _drawOrmCache![d] = _materialTextures?.OrmDefault;
                }

                var matPtr = IntPtr.Add(_materialMapped[frameIndex], d * UboAlign);
                Marshal.StructureToPtr(matUBO, matPtr, false);

                var shadowMvp = draw.WorldMatrix * lightVP;
                var shadowPtr = IntPtr.Add(_shadowDrawMapped[frameIndex], d * UboAlign);
                Marshal.StructureToPtr(shadowMvp, shadowPtr, false);
            }

            if (_algorithmHub != null && _sceneLights.Count > 0)
                _algorithmHub.TickShadows(cameraPos, lightView, lightProj);

            if (Vector3.DistanceSquared(_cameraPos, _bakedCameraPos) > 0.25f)
            {
                _bakedCameraPos = _cameraPos;
                _lastLightingHash = Vector3.Zero;
                RebuildLightingPipelineIfNeeded();
            }
        }

        private static MaterialUBO DefaultMaterialUbo() => new()
        {
            BaseColor = new Vector4(0.8f, 0.8f, 0.8f, 1f),
            Emissive = Vector4.Zero,
            Roughness = 0.5f,
            Metallic = 0f,
            AO = 1f,
            NormalScale = 1f,
            Opacity = 1f,
            Specular = 0.5f
        };

        /// <summary>
        /// Native FrameGraph present path: CPU producers (L-DNN) then GPU passes
        /// (G-DNN cull/algorithms → shadows → G-buffer → lighting → particles → post).
        /// Single executor — no parallel legacy pass loop.
        /// </summary>
        public void ExecuteFrame(
            VulkanCommandBuffer cmd,
            uint imageIndex,
            int frameIndex,
            Matrix4x4 view,
            Matrix4x4 projection,
            Vector3 cameraPos,
            Vector3 cameraForward,
            Vector3 cameraRight,
            float time)
        {
            if (!_initialized)
                return;

            UpdateUniforms(frameIndex, view, projection, cameraPos, time);
            _currentViewProjection = view * projection;

            var ctx = new FrameGraphContext
            {
                Rhi = _rhi,
                Cmd = cmd,
                ImageIndex = imageIndex,
                FrameIndex = frameIndex,
                Width = _width,
                Height = _height,
                World = _sceneWorld,
                Backend = this,
                View = view,
                Projection = projection,
                CameraPos = cameraPos,
                CameraForward = cameraForward,
                CameraRight = cameraRight,
                Time = time,
                RunLdnnCpuProducers = true,
                PhysicsFieldTemperature = _physicsFieldTemperature,
                Quality = _activeQuality
            };
            _activeFgContext = ctx;

            // Phase 1 — L-DNN Hybrid producers + texture upload (outside cmd buffer).
            _frameGraph.ExecuteCpuProducers(ctx);

            // Phase 2 — GPU recording via the same FrameGraph.
            ctx.RunLdnnCpuProducers = false;
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            _frameGraph.ExecuteGpuPasses(ctx);
            cmd.End();

            _prevViewProjection = _currentViewProjection;
            _hasPrevViewProjection = true;
            _activeFgContext = null;
        }

        /// <summary>
        /// Applies adaptive quality from Infrastructure into G-DNN LOD, shadows, L-DNN and post.
        /// </summary>
        public void ApplyRuntimeQuality(RuntimeRenderQuality quality)
        {
            ArgumentNullException.ThrowIfNull(quality);
            _activeQuality = quality;
            _enableGi = quality.EnableGlobalIllumination;
            _enableSsao = quality.EnableSsao;
            _enableBloom = quality.EnableBloom;
            _enableTaa = quality.EnableTaa;
            _extraShadowCascades = Math.Clamp(quality.ShadowCascades - 1, 0, 2);
            _maxLodBias = Math.Clamp(quality.MaxLodLevel, 0, 8);

            if (_ldnnBridge?.Renderer?.Config is { } cfg)
            {
                cfg.QualityMode = quality.ShadowQuality >= 2
                    ? LDNNQualityMode.HybridRT
                    : LDNNQualityMode.NeuralOnly;
                cfg.GIComputationMode = !quality.EnableGlobalIllumination
                    ? GIComputationMode.SSGI
                    : quality.EnableScreenSpaceGi
                        ? GIComputationMode.Hybrid
                        : GIComputationMode.RadianceCascades;
            }

            if (_algorithmHub?.NeuralLod?.Config is { } lodCfg)
                lodCfg.MaxLodLevel = Math.Max(1, 4 - _maxLodBias);
        }

        /// <summary>
        /// Feeds living-law / continuum field temperature into volumetric fog warmth.
        /// Native Physics → Rendering bridge on the present path.
        /// </summary>
        public void ApplyPhysicsFieldInfluence(float averageTemperatureKelvin)
        {
            if (!float.IsFinite(averageTemperatureKelvin))
                return;

            _physicsFieldTemperature = averageTemperatureKelvin;
            _ldnnBridge?.ApplyPhysicsFieldTemperature(averageTemperatureKelvin, _width, _height);
        }

        public void OnFrameGraphCull(FrameGraphContext context)
        {
            _lodManager.UpdateAll(context.CameraPos, MathF.PI / 3f, context.Height);
            ApplySelectedLodDraws();
        }

        public void OnFrameGraphAlgorithms(FrameGraphContext context)
        {
            _algorithmHub?.TickCull(
                context.CameraPos,
                context.CameraForward,
                context.View * context.Projection,
                context.Width,
                context.Height,
                context.Time);
            TryInjectGdnnPresentMesh();
        }

        /// <summary>
        /// Uploads a freshly polygonized G-DNN mesh into the Vulkan G-buffer draw list
        /// when <see cref="RenderingAlgorithmHub"/> signals a new present mesh.
        /// </summary>
        private void TryInjectGdnnPresentMesh()
        {
            var mesh = _algorithmHub?.PeekPendingPresentMesh();
            if (mesh == null || mesh.TriangleCount <= 0 || !_initialized)
                return;

            try
            {
                var (vertices, indices) = PackNeuralMeshForGBuffer(mesh);
                if (_gdnnPresentMeshIndex >= 0 && _gdnnPresentMeshIndex < _sceneMeshes.Count)
                {
                    var slot = _sceneMeshes[_gdnnPresentMeshIndex];
                    slot.VertexData = vertices;
                    slot.VertexStride = 12;
                    slot.IndexData = indices;
                    slot.WorldMatrix = Matrix4x4.Identity;
                }
                else
                {
                    _gdnnPresentMeshIndex = AddMesh(vertices, 12, indices, materialIndex: 3);
                    SetMeshWorldMatrix(_gdnnPresentMeshIndex, Matrix4x4.Identity);
                }

                UploadSceneGeometry();
                _algorithmHub?.AcknowledgePresentMesh();
            }
            catch (Exception ex)
            {
                Synapse.Infrastructure.Logging.SynapseLogger.Default.Warn(
                    "SceneRenderer", "G-DNN present mesh inject skipped.", ex);
            }
        }

        private static (float[] Vertices, uint[] Indices) PackNeuralMeshForGBuffer(NeuralPolygonMesh mesh)
        {
            var vertices = new float[mesh.VertexCount * 12];
            for (int i = 0; i < mesh.VertexCount; i++)
            {
                var p = mesh.Positions[i];
                var n = i < mesh.Normals.Length ? mesh.Normals[i] : Vector3.UnitY;
                if (n.LengthSquared() < 1e-8f)
                    n = Vector3.UnitY;
                else
                    n = Vector3.Normalize(n);

                int o = i * 12;
                vertices[o] = p.X;
                vertices[o + 1] = p.Y;
                vertices[o + 2] = p.Z;
                vertices[o + 3] = n.X;
                vertices[o + 4] = n.Y;
                vertices[o + 5] = n.Z;
                // Spherical UV so procedural / VT textures paint the neural mesh.
                float u = 0.5f + MathF.Atan2(p.Z, p.X) / (MathF.PI * 2f);
                float v = 0.5f - MathF.Asin(Math.Clamp(n.Y, -1f, 1f)) / MathF.PI;
                vertices[o + 6] = u;
                vertices[o + 7] = v;
                // Genome / cluster-hash albedo so dense meshlets read as Nanite tiles in G-buffer.
                uint h = unchecked((uint)(i * 2654435761u));
                vertices[o + 8] = 0.30f + ((h) & 255) / 255f * 0.50f;
                vertices[o + 9] = 0.32f + ((h >> 8) & 255) / 255f * 0.48f;
                vertices[o + 10] = 0.34f + ((h >> 16) & 255) / 255f * 0.46f;
                vertices[o + 11] = 1f;
            }

            var indices = new uint[mesh.Indices.Length];
            for (int i = 0; i < mesh.Indices.Length; i++)
                indices[i] = (uint)mesh.Indices[i];
            return (vertices, indices);
        }

        public void OnFrameGraphParticles(FrameGraphContext context)
        {
            _algorithmHub?.TickPost(context.CameraPos, context.Time);
        }

        public void OnFrameGraphLdnn(FrameGraphContext context)
        {
            if (context.RunLdnnCpuProducers)
            {
                if (!_enableGi)
                    return;
                RenderGI(context.View, context.Projection, context.CameraPos, context.CameraForward, context.CameraRight);
                return;
            }

            // GPU phase: optional resident compute SSAO when SPIR-V is wired.
            if (_enableSsao)
                _ = _ldnnCompute?.TryDispatchSsao(context.Cmd, (_width + 7) / 8, (_height + 7) / 8);
        }

        public void OnFrameGraphShadow(FrameGraphContext context)
        {
            RenderShadowPass(context.Cmd, context.FrameIndex);
            if (_extraShadowCascades > 0)
                RenderShadowCascadesExtra(context.Cmd, context.FrameIndex, _extraShadowCascades);
        }

        public void OnFrameGraphGBuffer(FrameGraphContext context)
        {
            RecordGBufferPass(context.Cmd, context.ImageIndex, context.FrameIndex);
        }

        public void OnFrameGraphLighting(FrameGraphContext context)
        {
            RecordLightingPass(context.Cmd, context.ImageIndex, context.FrameIndex);
        }

        public void OnFrameGraphPost(FrameGraphContext context)
        {
            RecordPostPass(context.Cmd, context.ImageIndex, context.FrameIndex);
        }

        public void RecordCommandBuffer(VulkanCommandBuffer cmd, uint imageIndex, int frameIndex)
        {
            // Legacy entry: GPU passes only (L-DNN must have been run via ExecuteFrame / RenderGI).
            var ctx = new FrameGraphContext
            {
                Rhi = _rhi,
                Cmd = cmd,
                ImageIndex = imageIndex,
                FrameIndex = frameIndex,
                Width = _width,
                Height = _height,
                World = _sceneWorld,
                Backend = this,
                View = Matrix4x4.Identity,
                Projection = Matrix4x4.Identity,
                CameraPos = _cameraPos,
                CameraForward = -Vector3.UnitZ,
                CameraRight = Vector3.UnitX,
                RunLdnnCpuProducers = false,
                PhysicsFieldTemperature = _physicsFieldTemperature,
                Quality = _activeQuality
            };
            _activeFgContext = ctx;
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            _frameGraph.ExecuteGpuPasses(ctx);
            cmd.End();
            _activeFgContext = null;
        }

        private void RecordGBufferPass(VulkanCommandBuffer cmd, uint imageIndex, int frameIndex)
        {
            var extent = _rhi.Swapchain.Extent;
            var gbufferClear = new ClearValue[]
            {
                ClearValue.ColorClear(0, 0, 0, 0),
                ClearValue.ColorClear(0, 0, 0, 0),
                ClearValue.ColorClear(0, 0, 0, 0),
                ClearValue.ColorClear(0, 0, 0, 0),
                ClearValue.ColorClear(0, 0, 0, 0),
                ClearValue.DepthStencilClear(1.0f, 0)
            };

            cmd.BeginRenderPass(_gbufferRenderPass, _gbufferFramebuffers[imageIndex], gbufferClear);
            cmd.BindPipeline(_gbufferPipeline);
            cmd.SetViewport(new Viewport { X = 0, Y = 0, Width = extent.Width, Height = extent.Height, MinDepth = 0, MaxDepth = 1 });
            cmd.SetScissor(new Rect2D { Extent = new Extent2D(extent.Width, extent.Height) });

            if (_vertexBuffer != null && _draws.Count > 0)
            {
                cmd.BindVertexBuffer(_vertexBuffer);
                cmd.BindIndexBuffer(_indexBuffer, 0, IndexType.Uint32);
                int drawCount = Math.Min(_draws.Count, _drawSlotCount);
                var samp = _materialTextures?.Sampler ?? _gbufferSampler;
                for (int d = 0; d < drawCount; d++)
                {
                    var alb = _drawAlbedoCache != null && d < _drawAlbedoCache.Length && _drawAlbedoCache[d] != null
                        ? _drawAlbedoCache[d]! : _materialTextures?.White ?? _gbufferAlbedo[0];
                    var nrm = _drawNormalCache != null && d < _drawNormalCache.Length && _drawNormalCache[d] != null
                        ? _drawNormalCache[d]! : _materialTextures?.FlatNormal ?? _gbufferNormals[0];
                    var orm = _drawOrmCache != null && d < _drawOrmCache.Length && _drawOrmCache[d] != null
                        ? _drawOrmCache[d]! : _materialTextures?.OrmDefault ?? _gbufferMaterial[0];

                    _rhi.UpdateDescriptorSets(new[]
                    {
                        new DescriptorWrite
                        {
                            DescriptorSet = _gbufferDescriptorSets[frameIndex].Handle,
                            DstBinding = 3, DstArrayElement = 0,
                            DescriptorType = DescriptorType.CombinedImageSampler,
                            ImageInfos = new[] { new DescriptorImageInfo { Sampler = samp.Handle, ImageView = alb.GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                        },
                        new DescriptorWrite
                        {
                            DescriptorSet = _gbufferDescriptorSets[frameIndex].Handle,
                            DstBinding = 4, DstArrayElement = 0,
                            DescriptorType = DescriptorType.CombinedImageSampler,
                            ImageInfos = new[] { new DescriptorImageInfo { Sampler = samp.Handle, ImageView = nrm.GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                        },
                        new DescriptorWrite
                        {
                            DescriptorSet = _gbufferDescriptorSets[frameIndex].Handle,
                            DstBinding = 5, DstArrayElement = 0,
                            DescriptorType = DescriptorType.CombinedImageSampler,
                            ImageInfos = new[] { new DescriptorImageInfo { Sampler = samp.Handle, ImageView = orm.GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                        },
                    });

                    uint dyn = (uint)(d * UboAlign);
                    cmd.BindDescriptorSets(
                        PipelineBindPoint.Graphics,
                        _gbufferPipelineLayout.Handle,
                        0,
                        new[] { _gbufferDescriptorSets[frameIndex].Handle },
                        new[] { dyn, dyn });
                    var draw = _draws[d];
                    cmd.DrawIndexed(draw.IndexCount, 1, draw.FirstIndex, 0, 0);
                }
            }

            cmd.EndRenderPass();
        }

        private void RecordLightingPass(VulkanCommandBuffer cmd, uint imageIndex, int frameIndex)
        {
            var extent = _rhi.Swapchain.Extent;
            UpdateLightingDescriptors(imageIndex, frameIndex);

            var lightingClear = new ClearValue[]
            {
                ClearValue.ColorClear(0.02f, 0.02f, 0.04f, 1.0f)
            };

            cmd.BeginRenderPass(_lightingRenderPass, _lightingFramebuffers[imageIndex], lightingClear);
            cmd.BindPipeline(_lightingPipeline);
            cmd.SetViewport(new Viewport { X = 0, Y = 0, Width = extent.Width, Height = extent.Height, MinDepth = 0, MaxDepth = 1 });
            cmd.SetScissor(new Rect2D { Extent = new Extent2D(extent.Width, extent.Height) });
            cmd.BindDescriptorSets(PipelineBindPoint.Graphics, _lightingPipelineLayout.Handle, 0, new[] { _lightingDescriptorSets[frameIndex].Handle });
            cmd.Draw(3, 1, 0, 0);
            cmd.EndRenderPass();
        }

        private void RecordPostPass(VulkanCommandBuffer cmd, uint imageIndex, int frameIndex)
        {
            var extent = _rhi.Swapchain.Extent;
            EnsureTaaHistory();
            UpdatePostProcessDescriptors(frameIndex);

            if (_gbufferVelocity != null && imageIndex < _gbufferVelocity.Length)
            {
                _rhi.UpdateDescriptorSets(new[]
                {
                    new DescriptorWrite
                    {
                        DescriptorSet = _postProcessDescriptorSets[frameIndex].Handle,
                        DstBinding = 1, DstArrayElement = 0,
                        DescriptorType = DescriptorType.CombinedImageSampler,
                        ImageInfos = new[]
                        {
                            new DescriptorImageInfo
                            {
                                Sampler = _gbufferSampler.Handle,
                                ImageView = _gbufferVelocity[imageIndex].GetImageView(),
                                ImageLayout = ImageLayout.ShaderReadOnlyOptimal
                            }
                        }
                    }
                });
            }

            var postClear = new ClearValue[]
            {
                ClearValue.ColorClear(0.0f, 0.0f, 0.0f, 1.0f)
            };
            cmd.BeginRenderPass(_postProcessRenderPass, _postProcessFramebuffers![imageIndex], postClear);
            cmd.BindPipeline(_postProcessPipeline);
            cmd.SetViewport(new Viewport { X = 0, Y = 0, Width = extent.Width, Height = extent.Height, MinDepth = 0, MaxDepth = 1 });
            cmd.SetScissor(new Rect2D { Extent = new Extent2D(extent.Width, extent.Height) });
            cmd.BindDescriptorSets(PipelineBindPoint.Graphics, _postProcessPipelineLayout.Handle, 0, new[] { _postProcessDescriptorSets[frameIndex].Handle });
            cmd.Draw(3, 1, 0, 0);
            cmd.EndRenderPass();

            // Copy HDR → TAA history for next-frame temporal blend.
            if (_taaHistory != null && _hdrColorTarget != null)
            {
                cmd.PipelineBarrier(
                    PipelineStageFlag.ColorAttachmentOutput,
                    PipelineStageFlag.Transfer,
                    imageBarriers: new[]
                    {
                        new ImageMemoryBarrier
                        {
                            Image = _hdrColorTarget.Handle,
                            OldLayout = ImageLayout.ShaderReadOnlyOptimal,
                            NewLayout = ImageLayout.TransferSrcOptimal,
                            SrcAccessMask = AccessFlag.ShaderRead,
                            DstAccessMask = AccessFlag.TransferRead,
                            SubresourceRange = new ImageSubresourceRange
                            {
                                AspectMask = ImageAspectFlag.Color,
                                BaseMipLevel = 0, LevelCount = 1,
                                BaseArrayLayer = 0, LayerCount = 1
                            }
                        },
                        new ImageMemoryBarrier
                        {
                            Image = _taaHistory.Handle,
                            OldLayout = ImageLayout.Undefined,
                            NewLayout = ImageLayout.TransferDstOptimal,
                            DstAccessMask = AccessFlag.TransferWrite,
                            SubresourceRange = new ImageSubresourceRange
                            {
                                AspectMask = ImageAspectFlag.Color,
                                BaseMipLevel = 0, LevelCount = 1,
                                BaseArrayLayer = 0, LayerCount = 1
                            }
                        }
                    });
                cmd.CopyImage(_hdrColorTarget, ImageLayout.TransferSrcOptimal, _taaHistory, ImageLayout.TransferDstOptimal, new[]
                {
                    new ImageCopy
                    {
                        SrcSubresource = new ImageSubresourceLayers { AspectMask = ImageAspectFlag.Color, MipLevel = 0, BaseArrayLayer = 0, LayerCount = 1 },
                        DstSubresource = new ImageSubresourceLayers { AspectMask = ImageAspectFlag.Color, MipLevel = 0, BaseArrayLayer = 0, LayerCount = 1 },
                        Extent = new Extent3D(extent.Width, extent.Height, 1)
                    }
                });
                cmd.PipelineBarrier(
                    PipelineStageFlag.Transfer,
                    PipelineStageFlag.FragmentShader,
                    imageBarriers: new[]
                    {
                        new ImageMemoryBarrier
                        {
                            Image = _taaHistory.Handle,
                            OldLayout = ImageLayout.TransferDstOptimal,
                            NewLayout = ImageLayout.ShaderReadOnlyOptimal,
                            SrcAccessMask = AccessFlag.TransferWrite,
                            DstAccessMask = AccessFlag.ShaderRead,
                            SubresourceRange = new ImageSubresourceRange
                            {
                                AspectMask = ImageAspectFlag.Color,
                                BaseMipLevel = 0, LevelCount = 1,
                                BaseArrayLayer = 0, LayerCount = 1
                            }
                        },
                        new ImageMemoryBarrier
                        {
                            Image = _hdrColorTarget.Handle,
                            OldLayout = ImageLayout.TransferSrcOptimal,
                            NewLayout = ImageLayout.ShaderReadOnlyOptimal,
                            SrcAccessMask = AccessFlag.TransferRead,
                            DstAccessMask = AccessFlag.ShaderRead,
                            SubresourceRange = new ImageSubresourceRange
                            {
                                AspectMask = ImageAspectFlag.Color,
                                BaseMipLevel = 0, LevelCount = 1,
                                BaseArrayLayer = 0, LayerCount = 1
                            }
                        }
                    });
            }
        }

        public void RenderShadowPass(VulkanCommandBuffer cmd, int frameIndex)
        {
            var shadowSize = new Extent2D((uint)SHADOW_MAP_SIZE, (uint)SHADOW_MAP_SIZE);

            var shadowClear = new ClearValue[]
            {
                ClearValue.DepthStencilClear(1.0f, 0)
            };

            cmd.BeginRenderPass(_shadowRenderPass, _shadowFramebuffers[frameIndex], shadowClear);
            cmd.BindPipeline(_shadowPipeline);
            cmd.SetViewport(new Viewport { X = 0, Y = 0, Width = shadowSize.Width, Height = shadowSize.Height, MinDepth = 0, MaxDepth = 1 });
            cmd.SetScissor(new Rect2D { Extent = shadowSize });

            if (_vertexBuffer != null && _draws.Count > 0)
            {
                cmd.BindVertexBuffer(_vertexBuffer);
                cmd.BindIndexBuffer(_indexBuffer, 0, IndexType.Uint32);
                int drawCount = Math.Min(_draws.Count, _drawSlotCount);
                for (int d = 0; d < drawCount; d++)
                {
                    uint dyn = (uint)(d * UboAlign);
                    cmd.BindDescriptorSets(
                        PipelineBindPoint.Graphics,
                        _shadowPipelineLayout.Handle,
                        0,
                        new[] { _shadowDescriptorSets[frameIndex].Handle },
                        new[] { dyn });
                    var draw = _draws[d];
                    cmd.DrawIndexed(draw.IndexCount, 1, draw.FirstIndex, 0, 0);
                }
            }

            cmd.EndRenderPass();
        }

        /// <summary>
        /// Extra cascade shadow maps (near/mid/far) for CSM. Cascade 0 is the primary
        /// <see cref="RenderShadowPass"/>; this records cascade 1–N into dedicated targets when available.
        /// </summary>
        private void RenderShadowCascadesExtra(VulkanCommandBuffer cmd, int frameIndex, int extraCount = 2)
        {
            EnsureShadowCascades();
            if (_shadowCascadeDepth == null || _shadowCascadeFramebuffers == null)
                return;

            extraCount = Math.Clamp(extraCount, 0, 2);
            // Cascades 1 and 2 use wider orthos centered on the camera (PSSM-style).
            float[] orthoSizes = { 56f, 120f };
            for (int c = 0; c < extraCount; c++)
            {
                int slot = frameIndex * 2 + c;
                if (slot >= _shadowCascadeFramebuffers.Length)
                    break;

                UpdateCascadeShadowDraws(frameIndex, orthoSizes[c]);

                var shadowSize = new Extent2D((uint)SHADOW_MAP_SIZE, (uint)SHADOW_MAP_SIZE);
                cmd.BeginRenderPass(_shadowRenderPass, _shadowCascadeFramebuffers[slot], new[]
                {
                    ClearValue.DepthStencilClear(1.0f, 0)
                });
                cmd.BindPipeline(_shadowPipeline);
                cmd.SetViewport(new Viewport { X = 0, Y = 0, Width = shadowSize.Width, Height = shadowSize.Height, MinDepth = 0, MaxDepth = 1 });
                cmd.SetScissor(new Rect2D { Extent = shadowSize });

                if (_vertexBuffer != null && _draws.Count > 0)
                {
                    cmd.BindVertexBuffer(_vertexBuffer);
                    cmd.BindIndexBuffer(_indexBuffer, 0, IndexType.Uint32);
                    int drawCount = Math.Min(_draws.Count, _drawSlotCount);
                    for (int d = 0; d < drawCount; d++)
                    {
                        uint dyn = (uint)(d * UboAlign);
                        cmd.BindDescriptorSets(
                            PipelineBindPoint.Graphics,
                            _shadowPipelineLayout.Handle,
                            0,
                            new[] { _shadowDescriptorSets[frameIndex].Handle },
                            new[] { dyn });
                        var draw = _draws[d];
                        cmd.DrawIndexed(draw.IndexCount, 1, draw.FirstIndex, 0, 0);
                    }
                }

                cmd.EndRenderPass();
            }
        }

        private VulkanTexture[]? _shadowCascadeDepth;
        private VulkanFramebuffer[]? _shadowCascadeFramebuffers;

        private void EnsureShadowCascades()
        {
            if (_shadowCascadeDepth != null)
                return;

            int count = MAX_FRAMES_IN_FLIGHT * 2;
            _shadowCascadeDepth = new VulkanTexture[count];
            _shadowCascadeFramebuffers = new VulkanFramebuffer[count];
            for (int i = 0; i < count; i++)
            {
                _shadowCascadeDepth[i] = _rhi.CreateTexture(new TextureDescription
                {
                    Width = (uint)SHADOW_MAP_SIZE,
                    Height = (uint)SHADOW_MAP_SIZE,
                    Format = VulkanFormat.D32Sfloat,
                    Usage = ImageUsageFlag.DepthStencilAttachment | ImageUsageFlag.Sampled,
                    Tiling = ImageTiling.Optimal,
                    InitialLayout = ImageLayout.Undefined,
                    Samples = SampleCountFlag.Count1,
                });
                _shadowCascadeFramebuffers[i] = _rhi.CreateFramebuffer(new FramebufferDescription
                {
                    RenderPass = _shadowRenderPass.Handle,
                    Attachments = new[] { _shadowCascadeDepth[i].GetImageView() },
                    Width = (uint)SHADOW_MAP_SIZE,
                    Height = (uint)SHADOW_MAP_SIZE,
                    Layers = 1
                });
            }
        }

        private void UpdateCascadeShadowDraws(int frameIndex, float orthoSize)
        {
            if (_sceneLights.Count == 0 || frameIndex < 0 || frameIndex >= MAX_FRAMES_IN_FLIGHT)
                return;

            var light = _sceneLights[0];
            var lightDir = Vector3.Normalize(light.Direction);
            var focus = _cameraPos;
            var lightPos = focus - lightDir * (orthoSize * 1.2f);
            var up = MathF.Abs(Vector3.Dot(lightDir, Vector3.UnitY)) > 0.95f ? Vector3.UnitZ : Vector3.UnitY;
            var lightView = Matrix4x4.CreateLookAt(lightPos, focus, up);
            var lightProj = Matrix4x4.CreateOrthographic(orthoSize, orthoSize, 0.5f, orthoSize * 3.5f);
            lightProj.M22 *= -1;
            var lightVP = lightView * lightProj;
            int cascadeIndex = orthoSize < 70f ? 1 : 2;
            _cascadeLightVP[cascadeIndex] = lightVP;

            int drawCount = Math.Min(_draws.Count, _drawSlotCount);
            for (int d = 0; d < drawCount; d++)
            {
                var shadowMvp = _draws[d].WorldMatrix * lightVP;
                var shadowPtr = IntPtr.Add(_shadowDrawMapped[frameIndex], d * UboAlign);
                Marshal.StructureToPtr(shadowMvp, shadowPtr, false);
            }
        }

        private void WriteCascadeShadowUbo(int frameIndex)
        {
            if (frameIndex < 0 || frameIndex >= MAX_FRAMES_IN_FLIGHT || _shadowMapped[frameIndex] == IntPtr.Zero)
                return;

            var ubo = new CascadeShadowUBO
            {
                Cascade0 = _cascadeLightVP[0],
                Cascade1 = _cascadeLightVP[1] == default ? _cascadeLightVP[0] : _cascadeLightVP[1],
                Cascade2 = _cascadeLightVP[2] == default ? _cascadeLightVP[0] : _cascadeLightVP[2],
                // Distance splits matching ortho cascade sizes (28 / 56 / 120).
                Splits = new Vector4(18f, 48f, 3f, 0f)
            };
            Marshal.StructureToPtr(ubo, _shadowMapped[frameIndex], false);
        }

        private void ApplySelectedLodDraws()
        {
            if (_lodManager == null || _draws.Count == 0)
                return;

            foreach (var (group, level) in _lodManager.GetAllSelectedLevels())
            {
                if (!_lodGroupToDraw.TryGetValue(group.Id, out int drawIndex))
                    continue;
                if ((uint)drawIndex >= (uint)_draws.Count)
                    continue;
                if (level.MeshData is not LodMeshRange range)
                    continue;

                var draw = _draws[drawIndex];
                draw.FirstIndex = range.FirstIndex;
                draw.IndexCount = range.IndexCount;
                _draws[drawIndex] = draw;
            }
        }

        /// <summary>
        /// Runs L-DNN global illumination using scene camera and lights, then feeds GI boost into lighting.
        /// </summary>
        public void RenderGI(
            Matrix4x4 view,
            Matrix4x4 projection,
            Vector3 cameraPos,
            Vector3 cameraForward,
            Vector3 cameraRight)
        {
            if (!_initialized)
                return;

            var up = Vector3.Normalize(Vector3.Cross(cameraRight, cameraForward));
            var aspect = _height > 0 ? (float)_width / _height : 16f / 9f;
            _ldnnBridge.UpdateCamera(
                view, projection, cameraPos, cameraForward, cameraRight, up,
                60f, aspect, 0.1f, 100f);

            if (_sceneLights.Count > 0)
                PushLightsToGlobalIllumination();

            // Prefer fresh GPU readback; otherwise reuse resident GPU G-buffer (no constant fill).
            if (!_giUsesGpuReadback)
            {
                if (!_ldnnBridge.TryRestoreResidentGBuffer())
                    _ldnnBridge.FillGBufferFromConstants(10.0f, Vector3.UnitY, new Vector3(0.5f, 0.5f, 0.5f));
            }
            _giUsesGpuReadback = false;

            // Nanite-like: paint meshlet visibility into L-DNN CPU G-buffer before Hybrid GI.
            if (_algorithmHub != null)
            {
                _ldnnBridge.OverlayMeshletGBuffer(
                    (depth, normals, albedo, w, h) =>
                        _algorithmHub.CompositeMeshletsIntoLdnnGBuffer(depth, normals, albedo, w, h));
            }

            // Hybrid RT teacher is synthetic + CPU-heavy — keep off the realtime present path.
            var irradiance = _ldnnBridge.RenderGI();
            ApplyGiBoostFromIrradiance(irradiance);
        }

        /// <summary>Last industrial GI path used by the L-DNN bridge.</summary>
        public GiComputePath LastGiPath => _ldnnBridge?.LastGiPath ?? GiComputePath.None;

        private void ApplyGiBoostFromIrradiance(Vector3[,] irradiance)
        {
            if (irradiance == null || irradiance.GetLength(0) == 0)
                return;

            double sum = 0;
            int w = irradiance.GetLength(0);
            int h = irradiance.GetLength(1);
            int step = Math.Max(1, Math.Max(w, h) / 24);
            int samples = 0;
            for (int y = 0; y < h; y += step)
            {
                for (int x = 0; x < w; x += step)
                {
                    sum += irradiance[x, y].Length();
                    samples++;
                }
            }

            if (samples == 0)
                return;

            float mean = (float)(sum / samples);
            // GI drives exposure; flat ambient stays low so L-DNN dominates shadowed areas.
            _giBoost = Math.Clamp(mean * 0.28f, 0.05f, 0.72f);
            _dynamicAmbient = Math.Clamp(0.035f - mean * 0.01f, 0.02f, 0.05f);
            float targetExposure = Math.Clamp(1.35f / MathF.Max(0.35f, mean * 2.5f + 0.4f), 0.7f, 1.6f);
            _smoothedExposure = _smoothedExposure * 0.92f + targetExposure * 0.08f;

            _auxUploadFrame++;
            // Every-frame full-res AO + fog + meshlet resolve (Lumen-like stability, no sparse fill).
            float[,]? aoField = _ldnnBridge.GetAoField();
            Vector3[,]? fogField = _ldnnBridge.GetFogInScatterField();
            if (fogField != null && _algorithmHub != null)
            {
                _algorithmHub.TickPost(_cameraPos, _auxUploadFrame / 60f);
                _algorithmHub.CompositeParticlesIntoFog(fogField, _currentViewProjection, _width, _height);
                _algorithmHub.CompositeMeshletsIntoFog(fogField, _width, _height);
                _algorithmHub.CompositeVirtualTexturesIntoFog(fogField, _width, _height);
            }
            _lastFogUploadFrame = _auxUploadFrame;

            if (aoField != null && _algorithmHub != null)
                _algorithmHub.CompositeSdfAo(aoField, _width, _height);

            // Meshlet cluster albedo → irradiance so deferred lighting sees Nanite-like tiles.
            if (_algorithmHub != null)
                _algorithmHub.CompositeMeshletsIntoIrradiance(irradiance, _width, _height);

            // Upload VT atlas for G-buffer sampling occasionally.
            if (_materialTextures != null && _algorithmHub != null &&
                (_auxUploadFrame - _lastVtAtlasUploadFrame >= 8))
            {
                var vt = _algorithmHub.VirtualTextures;
                if (vt.ResidentTiles > 0)
                {
                    vt.BlitResidentPagesToAtlas();
                    _materialTextures.UploadVirtualTextureAtlas(
                        vt.PhysicalColorData, vt.PhysicalTextureWidth, vt.PhysicalTextureHeight);
                    _lastVtAtlasUploadFrame = _auxUploadFrame;
                }
            }

            // Phase 2: single Submit/Wait for GI (+ optional AO/fog).
            if (_batchedUploader != null)
            {
                _batchedUploader.Upload(
                    _giIrradianceTexture, irradiance,
                    aoField != null ? _aoTexture : null, aoField,
                    fogField != null ? _fogTexture : null, fogField,
                    _width, _height);
            }
            else
            {
                UploadGiIrradianceTexture(irradiance);
                if (aoField != null)
                    UploadAoTexture(aoField);
                if (fogField != null)
                    UploadFogTexture(fogField);
            }

            RebuildLightingPipelineIfNeeded();
            MaybeRebuildPostProcessPipeline();
        }

        /// <summary>
        /// Runs <see cref="RayTracingPipeline.TraceRays"/> on synthetic G-Buffer
        /// proxies so HybridRT quality mode has a wired RT path even without
        /// full GPU readback of Vulkan attachments.
        /// </summary>
        public void RenderHybridRayTracingTeacher()
        {
            if (_rtPipeline == null || !_rtPipeline.IsSupported)
                return;

            int pixelCount = _width * _height;
            var color = new float[pixelCount * 4];
            var normals = new float[pixelCount * 4];
            var depth = new float[pixelCount];
            var velocity = new float[pixelCount * 2];

            for (int i = 0; i < pixelCount; i++)
            {
                color[i * 4] = 0.5f;
                color[i * 4 + 1] = 0.5f;
                color[i * 4 + 2] = 0.5f;
                color[i * 4 + 3] = 1f;
                normals[i * 4 + 1] = 1f;
                depth[i] = 10f;
            }

            int lightCount = Math.Max(1, _sceneLights.Count);
            var lightPos = new float[lightCount * 3];
            var lightCol = new float[lightCount * 3];
            var lightInt = new float[lightCount];
            for (int l = 0; l < lightCount; l++)
            {
                var light = l < _sceneLights.Count ? _sceneLights[l] : null;
                lightPos[l * 3 + 1] = light?.Position.Y ?? 5f;
                lightCol[l * 3] = lightCol[l * 3 + 1] = lightCol[l * 3 + 2] = 1f;
                lightInt[l] = light?.Intensity ?? 1f;
            }

            _rtPipeline.TraceRays(
                color, normals, depth, velocity,
                _width, _height,
                Matrix4x4.Identity, Matrix4x4.Identity, Vector3.Zero,
                lightPos, lightCol, lightInt, lightCount,
                maxBounces: 2, samplesPerPixel: 1);
        }

        public void LoadDemoScene()
        {
            var material = new SubstrateMaterial("DemoMaterial");
            material.InitializeDefaults();
            material.SetProperty("BaseColor", new Color3(0.6f, 0.2f, 0.2f));
            material.SetProperty("Roughness", 0.4f);
            material.SetProperty("Metallic", 0.0f);
            AddMaterial(material);

            var material2 = new SubstrateMaterial("MetalMaterial");
            material2.InitializeDefaults();
            material2.SetProperty("BaseColor", new Color3(0.9f, 0.9f, 1.0f));
            material2.SetProperty("Roughness", 0.1f);
            material2.SetProperty("Metallic", 1.0f);
            AddMaterial(material2);

            var material3 = new SubstrateMaterial("RoughMaterial");
            material3.InitializeDefaults();
            material3.SetProperty("BaseColor", new Color3(0.2f, 0.5f, 0.8f));
            material3.SetProperty("Roughness", 0.9f);
            material3.SetProperty("Metallic", 0.0f);
            AddMaterial(material3);

            var material4 = new SubstrateMaterial("EmissiveMaterial");
            material4.InitializeDefaults();
            material4.SetProperty("BaseColor", new Color3(1.0f, 0.8f, 0.2f));
            material4.SetProperty("Roughness", 0.3f);
            material4.SetProperty("Metallic", 0.0f);
            material4.SetProperty("EmissiveIntensity", 5.0f);
            AddMaterial(material4);

            CreateCube(0, Matrix4x4.CreateTranslation(0, 0, 0));
            CreateCube(1, Matrix4x4.CreateTranslation(3, 0, 0));
            CreateCube(2, Matrix4x4.CreateTranslation(-3, 0, 0));
            CreateCube(3, Matrix4x4.CreateTranslation(0, 3, 0) * Matrix4x4.CreateScale(0.5f));
            CreatePlane(Matrix4x4.CreateTranslation(0, -1.5f, 0) * Matrix4x4.CreateScale(10.0f));

            AddLight(
                new Vector3(5, 8, 5),
                new Vector3(-0.5f, -1.0f, -0.5f),
                new Vector3(1.0f, 0.95f, 0.9f),
                3.0f, 100.0f
            );

            AddLight(
                new Vector3(-5, 4, -3),
                new Vector3(0.3f, -0.8f, 0.5f),
                new Vector3(0.4f, 0.6f, 1.0f),
                1.5f, 50.0f
            );

            _ldnnBridge.AddDirectionalLight(
                new Vector3(-0.5f, -1.0f, -0.5f),
                new Vector3(1.0f, 0.95f, 0.9f), 3.0f
            );
            _ldnnBridge.AddPointLight(
                new Vector3(-5, 4, -3),
                new Vector3(0.4f, 0.6f, 1.0f), 1.5f, 50.0f
            );

            UploadSceneGeometry();
        }

        private int CreateCube(int materialIndex, Matrix4x4 world)
        {
            float s = 1.0f;
            var vertices = new float[]
            {
                -s,-s, s,  0, 0, 1,  0,0,  1,0,0,1,
                 s,-s, s,  0, 0, 1,  1,0,  0,1,0,1,
                 s, s, s,  0, 0, 1,  1,1,  0,0,1,1,
                -s, s, s,  0, 0, 1,  0,1,  1,1,0,1,

                 s,-s,-s,  0, 0,-1,  0,0,  1,0,0,1,
                -s,-s,-s,  0, 0,-1,  1,0,  0,1,0,1,
                -s, s,-s,  0, 0,-1,  1,1,  0,0,1,1,
                 s, s,-s,  0, 0,-1,  0,1,  1,1,0,1,

                 s,-s, s,  1, 0, 0,  0,0,  0,1,0,1,
                 s,-s,-s,  1, 0, 0,  1,0,  1,0,0,1,
                 s, s,-s,  1, 0, 0,  1,1,  0,0,1,1,
                 s, s, s,  1, 0, 0,  0,1,  1,1,0,1,

                -s,-s,-s, -1, 0, 0,  0,0,  0,1,0,1,
                -s,-s, s, -1, 0, 0,  1,0,  1,0,0,1,
                -s, s, s, -1, 0, 0,  1,1,  0,0,1,1,
                -s, s,-s, -1, 0, 0,  0,1,  1,1,0,1,

                -s, s, s,  0, 1, 0,  0,0,  0,0,1,1,
                 s, s, s,  0, 1, 0,  1,0,  1,1,0,1,
                 s, s,-s,  0, 1, 0,  1,1,  0,1,0,1,
                -s, s,-s,  0, 1, 0,  0,1,  1,0,0,1,

                -s,-s,-s,  0,-1, 0,  0,0,  0,0,1,1,
                 s,-s,-s,  0,-1, 0,  1,0,  1,1,0,1,
                 s,-s, s,  0,-1, 0,  1,1,  0,1,0,1,
                -s,-s, s,  0,-1, 0,  0,1,  1,0,0,1,
            };

            var indices = new uint[]
            {
                0,1,2, 2,3,0, 4,5,6, 6,7,4, 8,9,10, 10,11,8,
                12,13,14, 14,15,12, 16,17,18, 18,19,16, 20,21,22, 22,23,20,
            };

            int meshIdx = AddMesh(vertices, 12, indices, materialIndex);
            SetMeshWorldMatrix(meshIdx, world);
            return meshIdx;
        }

        private void CreatePlane(Matrix4x4 world)
        {
            float s = 1.0f;
            var vertices = new float[]
            {
                -s, 0,-s,  0, 1, 0,  0,0,  0.5f,0.5f,0.5f,1,
                 s, 0,-s,  0, 1, 0,  1,0,  0.5f,0.5f,0.5f,1,
                 s, 0, s,  0, 1, 0,  1,1,  0.5f,0.5f,0.5f,1,
                -s, 0, s,  0, 1, 0,  0,1,  0.5f,0.5f,0.5f,1,
            };

            var indices = new uint[] { 0, 1, 2, 2, 3, 0 };

            int meshIdx = AddMesh(vertices, 12, indices, 0);
            SetMeshWorldMatrix(meshIdx, world);
        }

        private int CreatePlaneMesh(int materialIndex, Matrix4x4 world)
        {
            float s = 1.0f;
            var vertices = new float[]
            {
                -s, 0,-s,  0, 1, 0,  0,0,  0.35f,0.38f,0.42f,0.35f,
                 s, 0,-s,  0, 1, 0,  1,0,  0.35f,0.38f,0.42f,0.35f,
                 s, 0, s,  0, 1, 0,  1,1,  0.35f,0.38f,0.42f,0.35f,
                -s, 0, s,  0, 1, 0,  0,1,  0.35f,0.38f,0.42f,0.35f,
            };
            var indices = new uint[] { 0, 1, 2, 2, 3, 0 };
            int meshIdx = AddMesh(vertices, 12, indices, materialIndex);
            SetMeshWorldMatrix(meshIdx, world);
            return meshIdx;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _cameraUBOs[i]?.Unmap();
                _cameraUBOs[i]?.Dispose();
                _modelUBOs[i]?.Unmap();
                _modelUBOs[i]?.Dispose();
                _materialUBOs[i]?.Unmap();
                _materialUBOs[i]?.Dispose();
                _shadowUBOs[i]?.Unmap();
                _shadowUBOs[i]?.Dispose();
                _shadowDrawUBOs[i]?.Unmap();
                _shadowDrawUBOs[i]?.Dispose();
            }

            _vertexBuffer?.Dispose();
            _indexBuffer?.Dispose();
            _fullscreenVertexBuffer?.Dispose();

            _gbufferPipeline?.Dispose();
            _gbufferPipelineLayout?.Dispose();
            _gbufferRenderPass?.Dispose();
            _gbufferDescriptorPool?.Dispose();
            _gbufferDescriptorSetLayout?.Dispose();

            _lightingPipeline?.Dispose();
            _lightingPipelineLayout?.Dispose();
            _lightingRenderPass?.Dispose();
            _lightingDescriptorPool?.Dispose();
            _lightingDescriptorSetLayout?.Dispose();

            _postProcessPipeline?.Dispose();
            _postProcessPipelineLayout?.Dispose();
            _postProcessRenderPass?.Dispose();
            _postProcessDescriptorPool?.Dispose();
            _postProcessDescriptorSetLayout?.Dispose();

            _shadowPipeline?.Dispose();
            _shadowPipelineLayout?.Dispose();
            _shadowRenderPass?.Dispose();
            _shadowDescriptorPool?.Dispose();
            _shadowDescriptorSetLayout?.Dispose();

            _gbufferSampler?.Dispose();
            _shadowSampler?.Dispose();

            foreach (var fb in _gbufferFramebuffers)
                fb?.Dispose();
            foreach (var fb in _lightingFramebuffers)
                fb?.Dispose();
            if (_postProcessFramebuffers != null)
                foreach (var fb in _postProcessFramebuffers)
                    fb?.Dispose();
            foreach (var fb in _shadowFramebuffers)
                fb?.Dispose();

            if (_hdrColorTarget != null)
                _hdrColorTarget.Dispose();
            _taaHistory?.Dispose();
            if (_gbufferAlbedo != null)
                foreach (var t in _gbufferAlbedo)
                    t?.Dispose();
            if (_gbufferNormals != null)
                foreach (var t in _gbufferNormals)
                    t?.Dispose();
            if (_gbufferDepth != null)
                foreach (var t in _gbufferDepth)
                    t?.Dispose();
            if (_gbufferMaterial != null)
                foreach (var t in _gbufferMaterial)
                    t?.Dispose();
            if (_gbufferVelocity != null)
                foreach (var t in _gbufferVelocity)
                    t?.Dispose();
            if (_depthImages != null)
                foreach (var t in _depthImages)
                    t?.Dispose();
            if (_shadowDepthImages != null)
                foreach (var t in _shadowDepthImages)
                    t?.Dispose();
            _rtPipeline?.Dispose();
            _gBufferReadback?.Dispose();

            _giIrradianceTexture?.Dispose();
            _aoTexture?.Dispose();
            _fogTexture?.Dispose();
            _batchedUploader?.Dispose();
            _materialTextures?.Dispose();
            _ldnnCompute?.Dispose();
            if (_shadowCascadeDepth != null)
                foreach (var t in _shadowCascadeDepth)
                    t?.Dispose();
            if (_shadowCascadeFramebuffers != null)
                foreach (var fb in _shadowCascadeFramebuffers)
                    fb?.Dispose();

            _materialBridge?.Dispose();
            _ldnnBridge?.Dispose();
            _postProcessBridge?.Dispose();
            _algorithmHub?.Dispose();
            _batchedUploader?.Dispose();
            _materialTextures?.Dispose();
            _ldnnCompute?.Dispose();
            _lodManager?.Clear();
            _lodGroupToDraw.Clear();
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct CascadeShadowUBO
        {
            public Matrix4x4 Cascade0;
            public Matrix4x4 Cascade1;
            public Matrix4x4 Cascade2;
            public Vector4 Splits;
        }

        private readonly struct LodMeshRange
        {
            public readonly uint FirstIndex;
            public readonly uint IndexCount;
            public LodMeshRange(uint firstIndex, uint indexCount)
            {
                FirstIndex = firstIndex;
                IndexCount = indexCount;
            }
        }

        private struct MeshDraw
        {
            public int MeshIndex;
            public uint FirstIndex;
            public uint IndexCount;
            public int MaterialIndex;
            public Matrix4x4 WorldMatrix;
        }

        private class SceneMeshData
        {
            public float[] VertexData;
            public int VertexStride;
            public uint[] IndexData;
            public int MaterialIndex;
            public Matrix4x4 WorldMatrix;
        }

        private class SceneLightData
        {
            public Vector3 Position;
            public Vector3 Direction;
            public Vector3 Color;
            public float Intensity;
            public float Range;
        }
    }
}
