using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using GDNN.Lighting.LDNN;
using GDNN.Materials.SubstrateOmega;
using GDNN.Rendering.Compat;
using GDNN.Rendering.Bridge;
using GDNN.Rendering.LOD;
using GDNN.Rendering.MeshIO;
using GDNN.Rendering.Shaders;
using GDNN.RHI.Vulkan;
using GDNN.Scene;

namespace GDNN.Rendering.Engine
{
    public class SceneRenderer : IDisposable
    {
        private const int MAX_FRAMES_IN_FLIGHT = 2;
        private const int GBUFFER_ATTACHMENT_COUNT = 4;
        private const int SHADOW_MAP_SIZE = 2048;

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
        private VulkanTexture[] _depthImages;
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

        private VulkanRenderPass _lightingRenderPass;
        private VulkanPipelineLayout _lightingPipelineLayout;
        private VulkanPipeline _lightingPipeline;
        private VulkanFramebuffer[] _lightingFramebuffers;
        private DescriptorSetLayout _lightingDescriptorSetLayout;
        private DescriptorPool _lightingDescriptorPool;
        private DescriptorSet[] _lightingDescriptorSets;
        private Sampler _gbufferSampler;
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
        private DescriptorSetLayout _postProcessDescriptorSetLayout;
        private DescriptorPool _postProcessDescriptorPool;
        private DescriptorSet[] _postProcessDescriptorSets;

        private VulkanBuffer _vertexBuffer;
        private VulkanBuffer _indexBuffer;
        private uint _indexCount;

        private List<SceneMeshData> _sceneMeshes = new();
        private List<SceneLightData> _sceneLights = new();
        private List<SubstrateMaterial> _sceneMaterials = new();

        private int _width;
        private int _height;
        private bool _initialized;

        public int Width => _width;
        public int Height => _height;
        public bool IsInitialized => _initialized;
        public SampleCountFlag MSAASamples => _msaaSamples;
        public MaterialBridge Materials => _materialBridge;
        public LDNNBridge GlobalIllumination => _ldnnBridge;
        public PostProcessBridge PostProcess => _postProcessBridge;
        public MeshLoader Meshes => _meshLoader;
        public LodManager LOD => _lodManager;
        public int MeshCount => _sceneMeshes.Count;
        public int TriangleCount => (int)(_indexCount / 3);

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

            _postProcessBridge = new PostProcessBridge(width, height);
            _meshLoader = new MeshLoader();
            _lodManager = new LodManager();

            CreateGBufferRenderPass();
            CreateGBufferResources();
            CreateGBufferFramebuffers();
            CreateGBufferDescriptorResources();
            CreateGBufferPipeline();

            CreateLightingRenderPass();
            CreateLightingFramebuffers();
            CreateLightingDescriptorResources();
            CreateLightingPipeline();

            CreateShadowRenderPass();
            CreateShadowResources();
            CreateShadowFramebuffers();
            CreateShadowDescriptorResources();
            CreateShadowPipeline();

            CreatePostProcessResources();
            CreateUniformBuffers();

            _initialized = true;
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
            _depthImages = new VulkanTexture[count];

            for (int i = 0; i < count; i++)
            {
                _gbufferAlbedo[i] = _rhi.CreateTexture(new TextureDescription
                {
                    Width = extent.Width, Height = extent.Height,
                    Format = VulkanFormat.R16G16B16A16Sfloat,
                    Usage = ImageUsageFlag.ColorAttachment | ImageUsageFlag.Sampled,
                    Tiling = ImageTiling.Optimal,
                    InitialLayout = ImageLayout.Undefined,
                    Samples = SampleCountFlag.Count1,
                });

                _gbufferNormals[i] = _rhi.CreateTexture(new TextureDescription
                {
                    Width = extent.Width, Height = extent.Height,
                    Format = VulkanFormat.R16G16B16A16Sfloat,
                    Usage = ImageUsageFlag.ColorAttachment | ImageUsageFlag.Sampled,
                    Tiling = ImageTiling.Optimal,
                    InitialLayout = ImageLayout.Undefined,
                    Samples = SampleCountFlag.Count1,
                });

                _gbufferDepth[i] = _rhi.CreateTexture(new TextureDescription
                {
                    Width = extent.Width, Height = extent.Height,
                    Format = VulkanFormat.R16G16B16A16Sfloat,
                    Usage = ImageUsageFlag.ColorAttachment | ImageUsageFlag.Sampled,
                    Tiling = ImageTiling.Optimal,
                    InitialLayout = ImageLayout.Undefined,
                    Samples = SampleCountFlag.Count1,
                });

                _gbufferMaterial[i] = _rhi.CreateTexture(new TextureDescription
                {
                    Width = extent.Width, Height = extent.Height,
                    Format = VulkanFormat.R16G16B16A16Sfloat,
                    Usage = ImageUsageFlag.ColorAttachment | ImageUsageFlag.Sampled,
                    Tiling = ImageTiling.Optimal,
                    InitialLayout = ImageLayout.Undefined,
                    Samples = SampleCountFlag.Count1,
                });

                _depthImages[i] = _rhi.CreateTexture(new TextureDescription
                {
                    Width = extent.Width, Height = extent.Height,
                    Format = VulkanFormat.D32Sfloat,
                    Usage = ImageUsageFlag.DepthStencilAttachment,
                    Tiling = ImageTiling.Optimal,
                    InitialLayout = ImageLayout.Undefined,
                    Samples = SampleCountFlag.Count1,
                });
            }
        }

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
                attachments[4] = _depthImages[i].GetImageView();

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
                    new DescriptorSetLayoutBinding { Binding = 1, DescriptorType = DescriptorType.UniformBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlag.Vertex },
                    new DescriptorSetLayoutBinding { Binding = 2, DescriptorType = DescriptorType.UniformBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                }
            });

            _gbufferDescriptorPool = _rhi.CreateDescriptorPool(new PoolDescription
            {
                PoolSizes = new[]
                {
                    new DescriptorPoolSize { Type = DescriptorType.UniformBuffer, DescriptorCount = MAX_FRAMES_IN_FLIGHT * 3 },
                },
                MaxSets = MAX_FRAMES_IN_FLIGHT
            });

            var layouts = new IntPtr[MAX_FRAMES_IN_FLIGHT];
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) layouts[i] = _gbufferDescriptorSetLayout.Handle;

            _gbufferDescriptorSets = _rhi.AllocateDescriptorSets(new DescriptorSetAllocation { Pool = _gbufferDescriptorPool.Handle, Layouts = layouts });
        }

        private void CreateGBufferPipeline()
        {
            var vertSpv = EmbeddedShaders.CompileGBufferVertex();
            var fragSpv = EmbeddedShaders.CompileGBufferFragment();

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
                    DepthClampEnable = false, RasterizerDiscardEnable = false,
                    PolygonMode = PolygonMode.Fill, CullMode = CullModeFlag.Back,
                    FrontFace = FrontFace.CounterClockwise, DepthBiasEnable = false, LineWidth = 1.0f
                },
                MultisampleState = new PipelineMultisampleStateCreateInfo { RasterizationSamples = SampleCountFlag.Count1 },
                DepthStencilState = new PipelineDepthStencilStateCreateInfo
                {
                    DepthTestEnable = true, DepthWriteEnable = true, DepthCompareOp = CompareOp.LessOrEqual
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

        private void CreateLightingFramebuffers()
        {
            var extent = _rhi.Swapchain.Extent;
            var images = _rhi.Swapchain.GetImages();
            _lightingFramebuffers = new VulkanFramebuffer[images.Length];

            for (int i = 0; i < images.Length; i++)
            {
                var attachments = new IntPtr[] { _rhi.Swapchain.GetImageView((uint)i) };

                _lightingFramebuffers[i] = _rhi.CreateFramebuffer(new FramebufferDescription
                {
                    RenderPass = _lightingRenderPass.Handle,
                    Attachments = attachments,
                    Width = extent.Width,
                    Height = extent.Height,
                    Layers = 1
                });
            }
        }

        private void CreateLightingDescriptorResources()
        {
            _lightingDescriptorSetLayout = _rhi.CreateDescriptorSetLayout(new LayoutDescription
            {
                Bindings = new[]
                {
                    new DescriptorSetLayoutBinding { Binding = 0, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 1, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 2, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                    new DescriptorSetLayoutBinding { Binding = 3, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlag.Fragment },
                }
            });

            _lightingDescriptorPool = _rhi.CreateDescriptorPool(new PoolDescription
            {
                PoolSizes = new[]
                {
                    new DescriptorPoolSize { Type = DescriptorType.CombinedImageSampler, DescriptorCount = MAX_FRAMES_IN_FLIGHT * 4 },
                },
                MaxSets = MAX_FRAMES_IN_FLIGHT
            });

            var layouts = new IntPtr[MAX_FRAMES_IN_FLIGHT];
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) layouts[i] = _lightingDescriptorSetLayout.Handle;

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

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _rhi.UpdateDescriptorSets(new[]
                {
                    new DescriptorWrite
                    {
                        DescriptorSet = _lightingDescriptorSets[i].Handle,
                        DstBinding = 0, DstArrayElement = 0,
                        DescriptorType = DescriptorType.CombinedImageSampler,
                        ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _gbufferAlbedo[i].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                    },
                    new DescriptorWrite
                    {
                        DescriptorSet = _lightingDescriptorSets[i].Handle,
                        DstBinding = 1, DstArrayElement = 0,
                        DescriptorType = DescriptorType.CombinedImageSampler,
                        ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _gbufferNormals[i].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                    },
                    new DescriptorWrite
                    {
                        DescriptorSet = _lightingDescriptorSets[i].Handle,
                        DstBinding = 2, DstArrayElement = 0,
                        DescriptorType = DescriptorType.CombinedImageSampler,
                        ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _gbufferDepth[i].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                    },
                    new DescriptorWrite
                    {
                        DescriptorSet = _lightingDescriptorSets[i].Handle,
                        DstBinding = 3, DstArrayElement = 0,
                        DescriptorType = DescriptorType.CombinedImageSampler,
                        ImageInfos = new[] { new DescriptorImageInfo { Sampler = _gbufferSampler.Handle, ImageView = _gbufferMaterial[i].GetImageView(), ImageLayout = ImageLayout.ShaderReadOnlyOptimal } }
                    },
                });
            }
        }

        private void CreateLightingPipeline()
        {
            var vertSpv = EmbeddedShaders.CompileLightingVertex();
            var fragSpv = EmbeddedShaders.CompileLightingFragment();

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
                    DepthClampEnable = false, RasterizerDiscardEnable = false,
                    PolygonMode = PolygonMode.Fill, CullMode = CullModeFlag.Back,
                    FrontFace = FrontFace.CounterClockwise, DepthBiasEnable = false, LineWidth = 1.0f
                },
                MultisampleState = new PipelineMultisampleStateCreateInfo { RasterizationSamples = SampleCountFlag.Count1 },
                DepthStencilState = new PipelineDepthStencilStateCreateInfo
                {
                    DepthTestEnable = false, DepthWriteEnable = false
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
                    new DescriptorSetLayoutBinding { Binding = 0, DescriptorType = DescriptorType.UniformBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlag.Vertex },
                }
            });

            _shadowDescriptorPool = _rhi.CreateDescriptorPool(new PoolDescription
            {
                PoolSizes = new[]
                {
                    new DescriptorPoolSize { Type = DescriptorType.UniformBuffer, DescriptorCount = MAX_FRAMES_IN_FLIGHT },
                },
                MaxSets = MAX_FRAMES_IN_FLIGHT
            });

            var layouts = new IntPtr[MAX_FRAMES_IN_FLIGHT];
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) layouts[i] = _shadowDescriptorSetLayout.Handle;

            _shadowDescriptorSets = _rhi.AllocateDescriptorSets(new DescriptorSetAllocation { Pool = _shadowDescriptorPool.Handle, Layouts = layouts });

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _rhi.UpdateDescriptorSets(new[]
                {
                    new DescriptorWrite
                    {
                        DescriptorSet = _shadowDescriptorSets[i].Handle,
                        DstBinding = 0, DstArrayElement = 0,
                        DescriptorType = DescriptorType.UniformBuffer,
                        BufferInfos = new[] { new DescriptorBufferInfo { Buffer = _shadowUBOs[i].Handle, Offset = 0, Range = (ulong)Marshal.SizeOf<Matrix4x4>() } }
                    }
                });
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
                    DepthClampEnable = false, RasterizerDiscardEnable = false,
                    PolygonMode = PolygonMode.Fill, CullMode = CullModeFlag.Front,
                    FrontFace = FrontFace.CounterClockwise, DepthBiasEnable = true,
                    DepthBiasConstantFactor = 1.25f, DepthBiasSlopeFactor = 1.75f,
                    DepthBiasClamp = 0.0f, LineWidth = 1.0f
                },
                MultisampleState = new PipelineMultisampleStateCreateInfo { RasterizationSamples = SampleCountFlag.Count1 },
                DepthStencilState = new PipelineDepthStencilStateCreateInfo
                {
                    DepthTestEnable = true, DepthWriteEnable = true, DepthCompareOp = CompareOp.LessOrEqual
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
            _postProcessRenderPass = _lightingRenderPass;
        }

        private void CreateUniformBuffers()
        {
            ulong cameraSize = (ulong)Marshal.SizeOf<CameraUBO>();
            ulong modelSize = (ulong)Marshal.SizeOf<Matrix4x4>();
            ulong materialSize = (ulong)Marshal.SizeOf<MaterialUBO>();
            ulong lightUboSize = (ulong)Marshal.SizeOf<LightUBO>();
            ulong shadowSize = (ulong)Marshal.SizeOf<Matrix4x4>();

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _cameraUBOs[i] = _rhi.CreateBuffer(new BufferDescription { Size = cameraSize, Usage = BufferUsageFlag.UniformBuffer, MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent });
                _cameraMapped[i] = _cameraUBOs[i].Map();

                _modelUBOs[i] = _rhi.CreateBuffer(new BufferDescription { Size = modelSize, Usage = BufferUsageFlag.UniformBuffer, MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent });
                _modelMapped[i] = _modelUBOs[i].Map();

                _materialUBOs[i] = _rhi.CreateBuffer(new BufferDescription { Size = materialSize, Usage = BufferUsageFlag.UniformBuffer, MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent });
                _materialMapped[i] = _materialUBOs[i].Map();

                _shadowUBOs[i] = _rhi.CreateBuffer(new BufferDescription { Size = shadowSize, Usage = BufferUsageFlag.UniformBuffer, MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent });
                _shadowMapped[i] = _shadowUBOs[i].Map();
            }
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

        public bool LoadMeshFromFile(string filePath, int materialIndex = 0, Matrix4x4? worldMatrix = null)
        {
            if (!File.Exists(filePath)) return false;

            var asset = _meshLoader.LoadSync(filePath);
            if (asset == null) return false;

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

                var lodGroup = _lodManager.RegisterGroup(
                    asset.Name + "_prim" + prim.Bounds.Center.ToString(),
                    prim.Bounds.Extents.Length());
                lodGroup.ReferenceRadius = prim.Bounds.Extents.Length();
            }

            return true;
        }

        public void UploadSceneGeometry()
        {
            if (_sceneMeshes.Count == 0) return;

            int totalVertices = 0;
            int totalIndices = 0;
            foreach (var mesh in _sceneMeshes)
            {
                totalVertices += mesh.VertexData.Length / mesh.VertexStride;
                totalIndices += mesh.IndexData?.Length ?? 0;
            }

            var allVertices = new float[totalVertices * 12];
            var allIndices = new uint[totalIndices];
            int vOffset = 0;
            int iOffset = 0;
            int vertexOffset = 0;

            foreach (var mesh in _sceneMeshes)
            {
                int meshVertexCount = mesh.VertexData.Length / mesh.VertexStride;
                Array.Copy(mesh.VertexData, 0, allVertices, vOffset, mesh.VertexData.Length);
                vOffset += mesh.VertexData.Length;

                if (mesh.IndexData != null)
                {
                    for (int i = 0; i < mesh.IndexData.Length; i++)
                        allIndices[iOffset + i] = mesh.IndexData[i] + (uint)vertexOffset;
                    iOffset += mesh.IndexData.Length;
                }

                vertexOffset += meshVertexCount;
            }

            _indexCount = (uint)totalIndices;

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
                        DescriptorType = DescriptorType.UniformBuffer,
                        BufferInfos = new[] { new DescriptorBufferInfo { Buffer = _modelUBOs[i].Handle, Offset = 0, Range = (ulong)Marshal.SizeOf<Matrix4x4>() } }
                    },
                    new DescriptorWrite
                    {
                        DescriptorSet = _gbufferDescriptorSets[i].Handle, DstBinding = 2, DstArrayElement = 0,
                        DescriptorType = DescriptorType.UniformBuffer,
                        BufferInfos = new[] { new DescriptorBufferInfo { Buffer = _materialUBOs[i].Handle, Offset = 0, Range = (ulong)Marshal.SizeOf<MaterialUBO>() } }
                    },
                });
            }
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
            fixed (byte* src = data) { System.Buffer.MemoryCopy(src, (void*)mapped, data.Length, data.Length); }
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
            var extent = _rhi.Swapchain.Extent;
            var camUBO = _materialBridge.BuildCameraUBO(view, projection, cameraPos, time, extent.Width, extent.Height);
            Marshal.StructureToPtr(camUBO, _cameraMapped[frameIndex], false);

            var model = Matrix4x4.Identity;
            Marshal.StructureToPtr(model, _modelMapped[frameIndex], false);

            if (_sceneMaterials.Count > 0)
            {
                var matUBO = _materialBridge.ExtractProperties(_sceneMaterials[0]);
                Marshal.StructureToPtr(matUBO, _materialMapped[frameIndex], false);
            }

            var lightVP = Matrix4x4.Identity;
            if (_sceneLights.Count > 0)
            {
                var light = _sceneLights[0];
                var lightDir = Vector3.Normalize(light.Direction);
                var lightPos = -lightDir * 20.0f;
                var lightView = Matrix4x4.CreateLookAt(lightPos, Vector3.Zero, Vector3.UnitY);
                var lightProj = Matrix4x4.CreateOrthographic(40.0f, 40.0f, 0.1f, 100.0f);
                lightProj.M11 *= -1;
                lightVP = lightView * lightProj;
            }
            Marshal.StructureToPtr(lightVP, _shadowMapped[frameIndex], false);
        }

        public void RecordCommandBuffer(VulkanCommandBuffer cmd, uint imageIndex, int frameIndex)
        {
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            var extent = _rhi.Swapchain.Extent;

            var gbufferClear = new ClearValue[]
            {
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

            if (_vertexBuffer != null && _indexCount > 0)
            {
                cmd.BindVertexBuffer(_vertexBuffer);
                cmd.BindIndexBuffer(_indexBuffer, 0, IndexType.Uint32);
                cmd.BindDescriptorSets(PipelineBindPoint.Graphics, _gbufferPipelineLayout.Handle, 0, new[] { _gbufferDescriptorSets[frameIndex].Handle });
                cmd.DrawIndexed(_indexCount, 1, 0, 0, 0);
            }

            cmd.EndRenderPass();

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

            cmd.End();
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

            if (_vertexBuffer != null && _indexCount > 0)
            {
                cmd.BindVertexBuffer(_vertexBuffer);
                cmd.BindIndexBuffer(_indexBuffer, 0, IndexType.Uint32);
                cmd.BindDescriptorSets(PipelineBindPoint.Graphics, _shadowPipelineLayout.Handle, 0, new[] { _shadowDescriptorSets[frameIndex].Handle });
                cmd.DrawIndexed(_indexCount, 1, 0, 0, 0);
            }

            cmd.EndRenderPass();
        }

        public void RenderGI()
        {
            if (!_initialized) return;

            if (_sceneLights.Count > 0 && _sceneMeshes.Count > 0)
            {
                var view = Matrix4x4.Identity;
                var proj = Matrix4x4.Identity;
                var pos = Vector3.Zero;

                _ldnnBridge.UpdateCamera(view, proj, pos, RenderingMath.Forward, RenderingMath.Right, Vector3.UnitY, 60.0f, (float)_width / _height, 0.1f, 100.0f);
            }

            _ldnnBridge.FillGBufferFromConstants(10.0f, Vector3.UnitY, new Vector3(0.5f, 0.5f, 0.5f));
            _ldnnBridge.RenderGI();
        }

        public void RenderPostProcess(float aspectRatio)
        {
            if (!_initialized) return;
            _postProcessBridge.Process(aspectRatio);
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

        private void CreateCube(int materialIndex, Matrix4x4 world)
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

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _cameraUBOs[i]?.Unmap(); _cameraUBOs[i]?.Dispose();
                _modelUBOs[i]?.Unmap(); _modelUBOs[i]?.Dispose();
                _materialUBOs[i]?.Unmap(); _materialUBOs[i]?.Dispose();
                _shadowUBOs[i]?.Unmap(); _shadowUBOs[i]?.Dispose();
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

            _shadowPipeline?.Dispose();
            _shadowPipelineLayout?.Dispose();
            _shadowRenderPass?.Dispose();
            _shadowDescriptorPool?.Dispose();
            _shadowDescriptorSetLayout?.Dispose();

            _gbufferSampler?.Dispose();

            foreach (var fb in _gbufferFramebuffers) fb?.Dispose();
            foreach (var fb in _lightingFramebuffers) fb?.Dispose();
            foreach (var fb in _shadowFramebuffers) fb?.Dispose();

            if (_gbufferAlbedo != null) foreach (var t in _gbufferAlbedo) t?.Dispose();
            if (_gbufferNormals != null) foreach (var t in _gbufferNormals) t?.Dispose();
            if (_gbufferDepth != null) foreach (var t in _gbufferDepth) t?.Dispose();
            if (_gbufferMaterial != null) foreach (var t in _gbufferMaterial) t?.Dispose();
            if (_depthImages != null) foreach (var t in _depthImages) t?.Dispose();
            if (_shadowDepthImages != null) foreach (var t in _shadowDepthImages) t?.Dispose();

            _materialBridge?.Dispose();
            _ldnnBridge?.Dispose();
            _postProcessBridge?.Dispose();
            _lodManager?.Clear();
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
