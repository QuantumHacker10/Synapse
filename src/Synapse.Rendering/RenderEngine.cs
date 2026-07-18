// =============================================================================
// RenderEngine.cs - GDNN Engine: Complete Vulkan Render Loop
// Manages frame synchronization, swapchain presentation, and render pipeline
// =============================================================================

using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using GDNN.RHI.Vulkan;
using GDNN.Platform;
using GDNN.Rendering.Shaders;
using GDNN.Rendering.Bridge;

namespace GDNN.Rendering.Engine
{
    public class RenderEngine : IDisposable
    {
        private const int MAX_FRAMES_IN_FLIGHT = 2;
        private const string APP_NAME = "GDNN Synapse Engine";

        private VulkanRhiDevice _rhi;
        private IntPtr _window;
        private IntPtr _surface;
        private bool _disposed;
        private SceneRenderer _sceneRenderer;

        private IntPtr[] _imageAvailableSemaphores = new IntPtr[MAX_FRAMES_IN_FLIGHT];
        private IntPtr[] _renderFinishedSemaphores = new IntPtr[MAX_FRAMES_IN_FLIGHT];
        private IntPtr[] _inFlightFences = new IntPtr[MAX_FRAMES_IN_FLIGHT];
        private VulkanCommandBuffer[] _commandBuffers = new VulkanCommandBuffer[MAX_FRAMES_IN_FLIGHT];
        private uint _currentFrame;

        private VulkanRenderPass _renderPass;
        private VulkanPipelineLayout _pipelineLayout;
        private VulkanPipeline _graphicsPipeline;
        private VulkanTexture[] _swapchainDepthImages;
        private VulkanFramebuffer[] _framebuffers;
        private DescriptorSetLayout _descriptorSetLayout;
        private DescriptorPool _descriptorPool;
        private DescriptorSet[] _descriptorSets;
        private VulkanBuffer[] _uniformBuffers = new VulkanBuffer[MAX_FRAMES_IN_FLIGHT];
        private IntPtr[] _uniformMapped = new IntPtr[MAX_FRAMES_IN_FLIGHT];
        private VulkanBuffer _vertexBuffer;
        private VulkanBuffer _indexBuffer;
        private uint _indexCount;

        private Stopwatch _frameTimer = new();
        private float _deltaTime;
        private float _totalTime;
        private float _fps;
        private Stopwatch _fpsTimer = new();
        private int _fpsFrameCount;

        private Vector3 _cameraPos = new(0, 2, 5);
        private Vector3 _cameraFront = new(0, 0, -1);
        private Vector3 _cameraUp = Vector3.UnitY;
        private float _yaw = -90.0f;
        private float _pitch = 0.0f;
        private double _lastMouseX, _lastMouseY;
        private bool _firstMouse = true;
        private float _fov = 60.0f;
        private bool _useGlfw = true;
        private bool _paused;
        private IntPtr _externalHwnd;
        private int _externalWidth = 1280;
        private int _externalHeight = 720;
        private string _sceneName = "Demo Scene";

        private IntPtr _vkCreateSemaphore;
        private IntPtr _vkCreateFence;
        private IntPtr _vkDestroyFence;
        private IntPtr _vkWaitForFences;
        private IntPtr _vkResetFences;
        private IntPtr _vkQueueSubmit;
        private IntPtr _vkQueuePresentKHR;
        private IntPtr _vkAcquireNextImageKHR;
        private IntPtr _vkDeviceWaitIdle;

        public IntPtr Window => _window;
        public bool IsRunning => !_disposed;
        public bool IsPaused { get => _paused; set => _paused = value; }
        public float DeltaTime => _deltaTime;
        public float TotalTime => _totalTime;
        public float FPS => _fps;
        public string SceneName => _sceneName;

        public unsafe void Initialize(int width = 1280, int height = 720, bool enableValidation = true)
        {
            NativeLibraryResolver.EnsureRegistered();
            _useGlfw = true;
            _window = GlfwWindow.CreateWindow(width, height, APP_NAME);
            var extensions = GlfwWindow.GetRequiredInstanceExtensions();
            InitializeVulkanCore(width, height, enableValidation, extensions,
                () => GlfwWindow.CreateVulkanSurface(_rhi.Instance, _window));
        }

        public unsafe void InitializeFromHwnd(IntPtr hwnd, int width = 1280, int height = 720, bool enableValidation = true)
        {
            if (!OperatingSystem.IsWindows())
                throw new PlatformNotSupportedException("InitializeFromHwnd is only supported on Windows. Use Initialize() (GLFW) on Linux/macOS.");
            if (hwnd == IntPtr.Zero) throw new ArgumentNullException(nameof(hwnd));
            NativeLibraryResolver.EnsureRegistered();
            _useGlfw = false;
            _externalHwnd = hwnd;
            _externalWidth = Math.Max(1, width);
            _externalHeight = Math.Max(1, height);
            _window = IntPtr.Zero;
            InitializeVulkanCore(_externalWidth, _externalHeight, enableValidation,
                new[] { "VK_KHR_surface", "VK_KHR_win32_surface" },
                () => Win32VulkanSurface.CreateSurface(_rhi.Instance, hwnd));
        }

        private unsafe void InitializeVulkanCore(int width, int height, bool enableValidation, string[] extensions, Func<IntPtr> createSurface)
        {
            _rhi = new VulkanRhiDevice(new RhiDeviceCreationInfo
            {
                ApplicationName = APP_NAME,
                EnableValidation = enableValidation,
                RequiredExtensions = extensions
            });

            _surface = createSurface();
            _rhi.CreateSwapchain(_surface);

            LoadInstanceFunctions();
            LoadDeviceFunctions();

            CreateRenderPass();
            CreateDepthResources();
            CreateFramebuffers();
            CreateDescriptorResources();
            CreateGraphicsPipeline();
            CreateSceneGeometry();
            CreateUniformBuffers();
            CreateSyncObjects();
            AllocateCommandBuffers();

            _sceneRenderer = new SceneRenderer(_rhi);
            _sceneRenderer.Initialize((int)_rhi.Swapchain.Extent.Width, (int)_rhi.Swapchain.Extent.Height);
            _sceneRenderer.LoadDemoScene();

            _frameTimer.Start();
            _fpsTimer.Start();
        }

        public (Vector3 Position, Vector3 Front, Vector3 Up, float Yaw, float Pitch, float Fov) GetCamera() =>
            (_cameraPos, _cameraFront, _cameraUp, _yaw, _pitch, _fov);

        public void SetCamera(Vector3 position, float yaw, float pitch, float fov = 60f)
        {
            _cameraPos = position;
            _yaw = yaw;
            _pitch = Math.Clamp(pitch, -89f, 89f);
            _fov = fov;
            _cameraFront = Vector3.Normalize(new Vector3(
                MathF.Cos(MathHelper.Deg2Rad(_yaw)) * MathF.Cos(MathHelper.Deg2Rad(_pitch)),
                MathF.Sin(MathHelper.Deg2Rad(_pitch)),
                MathF.Sin(MathHelper.Deg2Rad(_yaw)) * MathF.Cos(MathHelper.Deg2Rad(_pitch))));
        }

        public void ApplyCameraDelta(float yawDelta, float pitchDelta, Vector3 move)
        {
            _yaw += yawDelta;
            _pitch = Math.Clamp(_pitch + pitchDelta, -89f, 89f);
            _cameraFront = Vector3.Normalize(new Vector3(
                MathF.Cos(MathHelper.Deg2Rad(_yaw)) * MathF.Cos(MathHelper.Deg2Rad(_pitch)),
                MathF.Sin(MathHelper.Deg2Rad(_pitch)),
                MathF.Sin(MathHelper.Deg2Rad(_yaw)) * MathF.Cos(MathHelper.Deg2Rad(_pitch))));
            _cameraPos += move;
        }

        public void LoadSceneName(string name) => _sceneName = name ?? "Scene";

        public void RequestQuit()
        {
            if (_useGlfw && _window != IntPtr.Zero)
                GlfwWindow.SetWindowShouldClose(_window, 1);
            else
                _disposed = true;
        }

        public void NotifyExternalResize(int width, int height)
        {
            _externalWidth = Math.Max(1, width);
            _externalHeight = Math.Max(1, height);
        }

        private static IntPtr _vulkanLib;

        private unsafe void LoadInstanceFunctions()
        {
            _vulkanLib = NativeLibraryResolver.LoadVulkan();
            _vkQueuePresentKHR = NativeLibraryResolver.GetExport(_vulkanLib, "vkQueuePresentKHR");
            _vkAcquireNextImageKHR = NativeLibraryResolver.GetExport(_vulkanLib, "vkAcquireNextImageKHR");
        }

        private unsafe void LoadDeviceFunctions()
        {
            var device = _rhi.Device;
            _vkCreateSemaphore = LoadDevProc(device, "vkCreateSemaphore");
            _vkCreateFence = LoadDevProc(device, "vkCreateFence");
            _vkDestroyFence = LoadDevProc(device, "vkDestroyFence");
            _vkWaitForFences = LoadDevProc(device, "vkWaitForFences");
            _vkResetFences = LoadDevProc(device, "vkResetFences");
            _vkQueueSubmit = LoadDevProc(device, "vkQueueSubmit");
            _vkDeviceWaitIdle = LoadDevProc(device, "vkDeviceWaitIdle");
            _vkQueuePresentKHR = LoadDevProc(device, "vkQueuePresentKHR");
            _vkAcquireNextImageKHR = LoadDevProc(device, "vkAcquireNextImageKHR");
        }

        private static unsafe IntPtr LoadDevProc(IntPtr device, string name)
        {
            var namePtr = Marshal.StringToHGlobalAnsi(name);
            try
            {
                var pfn = vkGetDeviceProcAddr(device, namePtr);
                if (pfn == IntPtr.Zero && _vulkanLib != IntPtr.Zero)
                    pfn = NativeLibraryResolver.GetExport(_vulkanLib, name);
                return pfn;
            }
            finally { Marshal.FreeHGlobal(namePtr); }
        }

        [DllImport("vulkan-1", EntryPoint = "vkGetDeviceProcAddr")]
        private static extern IntPtr vkGetDeviceProcAddr(IntPtr device, IntPtr pName);

        private IntPtr CreateSemaphoreRaw()
        {
            var ci = new VkSemaphoreCreateInfoRaw { sType = 7 };
            IntPtr semaphore = IntPtr.Zero;
            if (_vkCreateSemaphore != IntPtr.Zero)
                Marshal.GetDelegateForFunctionPointer<CreateSemaphoreDel>(_vkCreateSemaphore)(_rhi.Device, ref ci, IntPtr.Zero, ref semaphore);
            return semaphore;
        }

        private IntPtr CreateFenceRaw(bool signaled)
        {
            var ci = new VkFenceCreateInfoRaw { sType = 8, flags = signaled ? 1u : 0u };
            IntPtr fence = IntPtr.Zero;
            if (_vkCreateFence != IntPtr.Zero)
                Marshal.GetDelegateForFunctionPointer<CreateFenceDel>(_vkCreateFence)(_rhi.Device, ref ci, IntPtr.Zero, ref fence);
            return fence;
        }

        private void WaitForFenceRaw(IntPtr fence)
        {
            var fencePtr = Marshal.AllocHGlobal(IntPtr.Size);
            Marshal.WriteIntPtr(fencePtr, fence);
            Marshal.GetDelegateForFunctionPointer<WaitForFencesDel>(_vkWaitForFences)(_rhi.Device, 1, ref fencePtr, 1, 0xFFFFFFFFFFFFFFFF);
            Marshal.FreeHGlobal(fencePtr);
        }

        private void ResetFenceRaw(IntPtr fence)
        {
            var fencePtr = Marshal.AllocHGlobal(IntPtr.Size);
            Marshal.WriteIntPtr(fencePtr, fence);
            Marshal.GetDelegateForFunctionPointer<ResetFencesDel>(_vkResetFences)(_rhi.Device, 1, ref fencePtr);
            Marshal.FreeHGlobal(fencePtr);
        }

        private uint AcquireNextImageRaw(IntPtr semaphore)
        {
            uint imageIndex = 0;
            Marshal.GetDelegateForFunctionPointer<AcquireNextImageKHRDel>(_vkAcquireNextImageKHR)(
                _rhi.Device, _rhi.Swapchain.Handle, 0xFFFFFFFFFFFFFFFF, semaphore, IntPtr.Zero, ref imageIndex);
            return imageIndex;
        }

        private unsafe void SubmitRaw(VulkanCommandBuffer cmd, IntPtr waitSemaphore, IntPtr signalSemaphore, IntPtr fence)
        {
            var waitStage = PipelineStageFlag.ColorAttachmentOutput;
            var waitStagePtr = Marshal.AllocHGlobal(sizeof(uint));
            Marshal.WriteInt32(waitStagePtr, (int)waitStage);

            IntPtr waitSemaphorePtr = IntPtr.Zero;
            if (waitSemaphore != IntPtr.Zero)
            {
                waitSemaphorePtr = Marshal.AllocHGlobal(IntPtr.Size);
                Marshal.WriteIntPtr(waitSemaphorePtr, waitSemaphore);
            }

            IntPtr signalSemaphorePtr = IntPtr.Zero;
            if (signalSemaphore != IntPtr.Zero)
            {
                signalSemaphorePtr = Marshal.AllocHGlobal(IntPtr.Size);
                Marshal.WriteIntPtr(signalSemaphorePtr, signalSemaphore);
            }

            var cmdBufferArrayPtr = Marshal.AllocHGlobal(IntPtr.Size);
            Marshal.WriteIntPtr(cmdBufferArrayPtr, cmd.Handle);

            var submitInfo = new VkSubmitInfoRaw
            {
                sType = 4,
                commandBufferCount = 1,
                pCommandBuffers = cmdBufferArrayPtr,
                waitSemaphoreCount = waitSemaphore != IntPtr.Zero ? 1u : 0u,
                pWaitSemaphores = waitSemaphorePtr,
                pWaitDstStageMask = waitStagePtr,
                signalSemaphoreCount = signalSemaphore != IntPtr.Zero ? 1u : 0u,
                pSignalSemaphores = signalSemaphorePtr
            };

            Marshal.GetDelegateForFunctionPointer<QueueSubmitDel>(_vkQueueSubmit)(_rhi.Queue, 1, ref submitInfo, fence);

            Marshal.FreeHGlobal(cmdBufferArrayPtr);
            Marshal.FreeHGlobal(waitStagePtr);
            if (waitSemaphorePtr != IntPtr.Zero) Marshal.FreeHGlobal(waitSemaphorePtr);
            if (signalSemaphorePtr != IntPtr.Zero) Marshal.FreeHGlobal(signalSemaphorePtr);
        }

        private unsafe void PresentRaw(IntPtr waitSemaphore, uint imageIndex)
        {
            var presentInfo = new VkPresentInfoKERRaw
            {
                sType = 1000001001,
                swapchainCount = 1,
                pSwapchains = Marshal.AllocHGlobal(IntPtr.Size),
                pImageIndices = Marshal.AllocHGlobal(sizeof(uint)),
            };

            if (waitSemaphore != IntPtr.Zero)
            {
                presentInfo.waitSemaphoreCount = 1;
                presentInfo.pWaitSemaphores = Marshal.AllocHGlobal(IntPtr.Size);
                Marshal.WriteIntPtr(presentInfo.pWaitSemaphores, waitSemaphore);
            }

            Marshal.WriteIntPtr(presentInfo.pSwapchains, _rhi.Swapchain.Handle);
            Marshal.WriteInt32(presentInfo.pImageIndices, (int)imageIndex);

            Marshal.GetDelegateForFunctionPointer<QueuePresentKHRDel>(_vkQueuePresentKHR)(_rhi.Queue, ref presentInfo);

            Marshal.FreeHGlobal(presentInfo.pSwapchains);
            Marshal.FreeHGlobal(presentInfo.pImageIndices);
            if (presentInfo.pWaitSemaphores != IntPtr.Zero) Marshal.FreeHGlobal(presentInfo.pWaitSemaphores);
        }

        [StructLayout(LayoutKind.Sequential)] private struct VkSemaphoreCreateInfoRaw { public uint sType; public IntPtr pNext; public uint flags; }
        [StructLayout(LayoutKind.Sequential)] private struct VkFenceCreateInfoRaw { public uint sType; public IntPtr pNext; public uint flags; }
        [StructLayout(LayoutKind.Sequential)] private struct VkSubmitInfoRaw { public uint sType; public IntPtr pNext; public uint waitSemaphoreCount; public IntPtr pWaitSemaphores; public IntPtr pWaitDstStageMask; public uint commandBufferCount; public IntPtr pCommandBuffers; public uint signalSemaphoreCount; public IntPtr pSignalSemaphores; }
        [StructLayout(LayoutKind.Sequential)] private struct VkPresentInfoKERRaw { public uint sType; public IntPtr pNext; public uint waitSemaphoreCount; public IntPtr pWaitSemaphores; public uint swapchainCount; public IntPtr pSwapchains; public IntPtr pImageIndices; public IntPtr pResults; }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateSemaphoreDel(IntPtr device, ref VkSemaphoreCreateInfoRaw ci, IntPtr a, ref IntPtr s);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult CreateFenceDel(IntPtr device, ref VkFenceCreateInfoRaw ci, IntPtr a, ref IntPtr f);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult WaitForFencesDel(IntPtr device, uint c, ref IntPtr f, uint w, ulong t);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult ResetFencesDel(IntPtr device, uint c, ref IntPtr f);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult AcquireNextImageKHRDel(IntPtr d, IntPtr sw, ulong t, IntPtr sem, IntPtr f, ref uint idx);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult QueueSubmitDel(IntPtr q, uint c, ref VkSubmitInfoRaw s, IntPtr f);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult QueuePresentKHRDel(IntPtr q, ref VkPresentInfoKERRaw p);

        private void CreateRenderPass()
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

            var depthAttachment = new AttachmentDescription
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

            var subpass = new SubpassDescription
            {
                PipelineBindPoint = PipelineBindPoint.Graphics,
                ColorAttachments = new[] { new AttachmentReference(0, ImageLayout.ColorAttachmentOptimal) },
                DepthStencilAttachment = new AttachmentReference(1, ImageLayout.DepthStencilAttachmentOptimal)
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

            _renderPass = _rhi.CreateRenderPass(new RenderPassDescription
            {
                Attachments = new[] { colorAttachment, depthAttachment },
                Subpasses = new[] { subpass },
                Dependencies = new[] { dependency }
            });
        }

        private void CreateDepthResources()
        {
            var extent = _rhi.Swapchain.Extent;
            var images = _rhi.Swapchain.GetImages();
            _swapchainDepthImages = new VulkanTexture[images.Length];

            for (int i = 0; i < images.Length; i++)
            {
                _swapchainDepthImages[i] = _rhi.CreateTexture(new TextureDescription
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

        private void CreateFramebuffers()
        {
            var extent = _rhi.Swapchain.Extent;
            var images = _rhi.Swapchain.GetImages();
            _framebuffers = new VulkanFramebuffer[images.Length];

            for (int i = 0; i < images.Length; i++)
            {
                var attachments = new IntPtr[2];
                attachments[0] = _rhi.Swapchain.GetImageView((uint)i);
                attachments[1] = _swapchainDepthImages[i].GetImageView();

                _framebuffers[i] = _rhi.CreateFramebuffer(new FramebufferDescription
                {
                    RenderPass = _renderPass.Handle,
                    Attachments = attachments,
                    Width = extent.Width, Height = extent.Height, Layers = 1
                });
            }
        }

        private void CreateDescriptorResources()
        {
            _descriptorSetLayout = _rhi.CreateDescriptorSetLayout(new LayoutDescription
            {
                Bindings = new[]
                {
                    new DescriptorSetLayoutBinding { Binding = 0, DescriptorType = DescriptorType.UniformBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlag.Vertex },
                }
            });

            _descriptorPool = _rhi.CreateDescriptorPool(new PoolDescription
            {
                PoolSizes = new[] { new DescriptorPoolSize { Type = DescriptorType.UniformBuffer, DescriptorCount = MAX_FRAMES_IN_FLIGHT } },
                MaxSets = MAX_FRAMES_IN_FLIGHT
            });

            var layouts = new IntPtr[MAX_FRAMES_IN_FLIGHT];
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) layouts[i] = _descriptorSetLayout.Handle;

            _descriptorSets = _rhi.AllocateDescriptorSets(new DescriptorSetAllocation { Pool = _descriptorPool.Handle, Layouts = layouts });

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _rhi.UpdateDescriptorSets(new[]
                {
                    new DescriptorWrite
                    {
                        DescriptorSet = _descriptorSets[i].Handle, DstBinding = 0, DstArrayElement = 0,
                        DescriptorType = DescriptorType.UniformBuffer,
                        BufferInfos = new[] { new DescriptorBufferInfo { Buffer = _uniformBuffers[i].Handle, Offset = 0, Range = (ulong)(Marshal.SizeOf<Matrix4x4>() * 2) } }
                    }
                });
            }
        }

        private void CreateGraphicsPipeline()
        {
            var vertSpv = EmbeddedShaders.CompileVertexShader();
            var fragSpv = EmbeddedShaders.CompileFragmentShader();

            var vertModule = _rhi.CreateShaderModule(vertSpv);
            var fragModule = _rhi.CreateShaderModule(fragSpv);

            _pipelineLayout = _rhi.CreatePipelineLayout(new[] { _descriptorSetLayout.Handle });

            var shaderStages = new PipelineShaderStageCreateInfo[]
            {
                new PipelineShaderStageCreateInfo { Stage = ShaderStageFlag.Vertex, Module = vertModule.Handle, Name = "main" },
                new PipelineShaderStageCreateInfo { Stage = ShaderStageFlag.Fragment, Module = fragModule.Handle, Name = "main" }
            };

            _graphicsPipeline = _rhi.CreatePipeline(new PipelineDescription
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
                InputAssemblyState = new PipelineInputAssemblyStateCreateInfo { Topology = PrimitiveTopology.TriangleList, PrimitiveRestartEnable = false },
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
                ColorBlendState = new PipelineColorBlendStateCreateInfo
                {
                    Attachments = new[] { PipelineColorBlendAttachmentState.Disabled() }
                },
                DynamicState = new PipelineDynamicStateCreateInfo
                {
                    DynamicStates = new[] { DynamicState.Viewport, DynamicState.Scissor }
                },
                PipelineLayout = _pipelineLayout.Handle,
                RenderPass = _renderPass.Handle,
                Subpass = 0
            });
        }

        private void CreateSceneGeometry()
        {
            float s = 1.0f;
            var vertices = new float[]
            {
                -s,-s, s,  0, 0, 1,  0,0,  1,0.2f,0.2f,
                 s,-s, s,  0, 0, 1,  1,0,  0.2f,1,0.2f,
                 s, s, s,  0, 0, 1,  1,1,  0.2f,0.2f,1,
                -s, s, s,  0, 0, 1,  0,1,  1,1,0.2f,

                 s,-s,-s,  0, 0,-1,  0,0,  1,0.2f,0.2f,
                -s,-s,-s,  0, 0,-1,  1,0,  0.2f,1,0.2f,
                -s, s,-s,  0, 0,-1,  1,1,  0.2f,0.2f,1,
                 s, s,-s,  0, 0,-1,  0,1,  1,1,0.2f,

                 s,-s, s,  1, 0, 0,  0,0,  0.2f,1,0.2f,
                 s,-s,-s,  1, 0, 0,  1,0,  1,0.2f,0.2f,
                 s, s,-s,  1, 0, 0,  1,1,  0.2f,0.2f,1,
                 s, s, s,  1, 0, 0,  0,1,  1,1,0.2f,

                -s,-s,-s, -1, 0, 0,  0,0,  0.2f,1,0.2f,
                -s,-s, s, -1, 0, 0,  1,0,  1,0.2f,0.2f,
                -s, s, s, -1, 0, 0,  1,1,  0.2f,0.2f,1,
                -s, s,-s, -1, 0, 0,  0,1,  1,1,0.2f,

                -s, s, s,  0, 1, 0,  0,0,  0.2f,0.2f,1,
                 s, s, s,  0, 1, 0,  1,0,  1,1,0.2f,
                 s, s,-s,  0, 1, 0,  1,1,  0.2f,1,0.2f,
                -s, s,-s,  0, 1, 0,  0,1,  1,0.2f,0.2f,

                -s,-s,-s,  0,-1, 0,  0,0,  0.2f,0.2f,1,
                 s,-s,-s,  0,-1, 0,  1,0,  1,1,0.2f,
                 s,-s, s,  0,-1, 0,  1,1,  0.2f,1,0.2f,
                -s,-s, s,  0,-1, 0,  0,1,  1,0.2f,0.2f,
            };

            var indices = new ushort[]
            {
                0,1,2, 2,3,0, 4,5,6, 6,7,4, 8,9,10, 10,11,8,
                12,13,14, 14,15,12, 16,17,18, 18,19,16, 20,21,22, 22,23,20,
            };
            _indexCount = (uint)indices.Length;

            var vertexData = MemoryMarshal.AsBytes(vertices.AsSpan());
            _vertexBuffer = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)vertexData.Length,
                Usage = BufferUsageFlag.VertexBuffer | BufferUsageFlag.TransferDst,
                MemoryProperties = MemoryPropertyFlag.DeviceLocal
            });

            var indexData = MemoryMarshal.AsBytes(indices.AsSpan());
            _indexBuffer = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)indexData.Length,
                Usage = BufferUsageFlag.IndexBuffer | BufferUsageFlag.TransferDst,
                MemoryProperties = MemoryPropertyFlag.DeviceLocal
            });

            UploadToGpu(_vertexBuffer, vertexData);
            UploadToGpu(_indexBuffer, indexData);
        }

        private unsafe void UploadToGpu(VulkanBuffer dst, ReadOnlySpan<byte> data)
        {
            var staging = _rhi.CreateBuffer(new BufferDescription
            {
                Size = (ulong)data.Length, Usage = BufferUsageFlag.TransferSrc,
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

        private void CreateUniformBuffers()
        {
            ulong bufferSize = (ulong)(Marshal.SizeOf<Matrix4x4>() * 2);
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _uniformBuffers[i] = _rhi.CreateBuffer(new BufferDescription
                {
                    Size = bufferSize, Usage = BufferUsageFlag.UniformBuffer,
                    MemoryProperties = MemoryPropertyFlag.HostVisible | MemoryPropertyFlag.HostCoherent
                });
                _uniformMapped[i] = _uniformBuffers[i].Map();
            }
        }

        private void CreateSyncObjects()
        {
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                _imageAvailableSemaphores[i] = CreateSemaphoreRaw();
                _renderFinishedSemaphores[i] = CreateSemaphoreRaw();
                _inFlightFences[i] = CreateFenceRaw(true);
            }
        }

        private void AllocateCommandBuffers()
        {
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
                _commandBuffers[i] = _rhi.CreateCommandBuffer();
        }

        public void RenderFrame()
        {
            if (_paused || _disposed) return;

            _deltaTime = (float)_frameTimer.Elapsed.TotalSeconds;
            _frameTimer.Restart();
            _totalTime += _deltaTime;

            _fpsFrameCount++;
            if (_fpsTimer.Elapsed.TotalSeconds >= 1.0)
            {
                _fps = _fpsFrameCount / (float)_fpsTimer.Elapsed.TotalSeconds;
                _fpsFrameCount = 0;
                _fpsTimer.Restart();
            }

            if (_useGlfw)
            {
                GlfwWindow.PollEvents();
                if (GlfwWindow.ShouldClose(_window)) { _disposed = true; return; }
                HandleInput();
            }

            WaitForFenceRaw(_inFlightFences[_currentFrame]);
            ResetFenceRaw(_inFlightFences[_currentFrame]);

            uint imageIndex = AcquireNextImageRaw(_imageAvailableSemaphores[_currentFrame]);
            GetFramebufferSize(out int fbW, out int fbH);
            if (fbW == 0 || fbH == 0) fbW = fbH = 1;
            var view = Matrix4x4.CreateLookAt(_cameraPos, _cameraPos + _cameraFront, _cameraUp);
            var proj = Matrix4x4.CreatePerspectiveFieldOfView(MathHelper.Deg2Rad(_fov), (float)fbW / fbH, 0.1f, 100.0f);
            proj.M11 *= -1;
            UpdateUniformBuffer((int)_currentFrame);
            if (_sceneRenderer != null && _sceneRenderer.IsInitialized)
            {
                _sceneRenderer.UpdateUniforms((int)_currentFrame, view, proj, _cameraPos, _totalTime);
                _sceneRenderer.RecordCommandBuffer(_commandBuffers[_currentFrame], imageIndex, (int)_currentFrame);
            }
            else
            {
                RecordCommandBuffer(_commandBuffers[_currentFrame], imageIndex);
            }
            SubmitRaw(_commandBuffers[_currentFrame], _imageAvailableSemaphores[_currentFrame], _renderFinishedSemaphores[_currentFrame], _inFlightFences[_currentFrame]);
            PresentRaw(_renderFinishedSemaphores[_currentFrame], imageIndex);

            _currentFrame = (_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        private void GetFramebufferSize(out int width, out int height)
        {
            if (_useGlfw && _window != IntPtr.Zero)
            {
                GlfwWindow.GetFramebufferSize(_window, out width, out height);
            }
            else
            {
                width = _externalWidth;
                height = _externalHeight;
            }
        }

        private void HandleInput()
        {
            double mouseX, mouseY;
            GlfwWindow.GetCursorPos(_window, out mouseX, out mouseY);

            if (_firstMouse) { _lastMouseX = mouseX; _lastMouseY = mouseY; _firstMouse = false; }

            float xoffset = (float)(mouseX - _lastMouseX) * 0.1f;
            float yoffset = (float)(_lastMouseY - mouseY) * 0.1f;
            _lastMouseX = mouseX; _lastMouseY = mouseY;

            _yaw += xoffset; _pitch += yoffset;
            _pitch = Math.Clamp(_pitch, -89.0f, 89.0f);

            _cameraFront = Vector3.Normalize(new Vector3(
                MathF.Cos(MathHelper.Deg2Rad(_yaw)) * MathF.Cos(MathHelper.Deg2Rad(_pitch)),
                MathF.Sin(MathHelper.Deg2Rad(_pitch)),
                MathF.Sin(MathHelper.Deg2Rad(_yaw)) * MathF.Cos(MathHelper.Deg2Rad(_pitch))));

            float speed = 3.0f * _deltaTime;
            if (GlfwWindow.GetKey(_window, GlfwWindow.GLFW_KEY_W) == GlfwWindow.GLFW_PRESS) _cameraPos += speed * _cameraFront;
            if (GlfwWindow.GetKey(_window, GlfwWindow.GLFW_KEY_S) == GlfwWindow.GLFW_PRESS) _cameraPos -= speed * _cameraFront;
            if (GlfwWindow.GetKey(_window, GlfwWindow.GLFW_KEY_A) == GlfwWindow.GLFW_PRESS) _cameraPos -= Vector3.Normalize(Vector3.Cross(_cameraFront, _cameraUp)) * speed;
            if (GlfwWindow.GetKey(_window, GlfwWindow.GLFW_KEY_D) == GlfwWindow.GLFW_PRESS) _cameraPos += Vector3.Normalize(Vector3.Cross(_cameraFront, _cameraUp)) * speed;
            if (GlfwWindow.GetKey(_window, GlfwWindow.GLFW_KEY_SPACE) == GlfwWindow.GLFW_PRESS) _cameraPos += _cameraUp * speed;
            if (GlfwWindow.GetKey(_window, GlfwWindow.GLFW_KEY_ESCAPE) == GlfwWindow.GLFW_PRESS) GlfwWindow.SetWindowShouldClose(_window, 1);
        }

        private unsafe void UpdateUniformBuffer(int idx)
        {
            GetFramebufferSize(out int w, out int h);
            if (w == 0 || h == 0) return;

            var view = Matrix4x4.CreateLookAt(_cameraPos, _cameraPos + _cameraFront, _cameraUp);
            var proj = Matrix4x4.CreatePerspectiveFieldOfView(MathHelper.Deg2Rad(_fov), (float)w / h, 0.1f, 100.0f);
            proj.M11 *= -1;
            var model = Matrix4x4.CreateRotationY(_totalTime * 0.5f) * Matrix4x4.CreateRotationX(MathHelper.Deg2Rad(20));

            int matrixSize = Marshal.SizeOf<Matrix4x4>();
            var data = stackalloc byte[matrixSize * 2];
            Marshal.StructureToPtr(model, (IntPtr)data, false);
            Marshal.StructureToPtr(view * proj, (IntPtr)(data + matrixSize), false);
            fixed (byte* pSrc = new ReadOnlySpan<byte>(data, matrixSize * 2))
            {
                System.Buffer.MemoryCopy(pSrc, (void*)_uniformMapped[idx], matrixSize * 2, matrixSize * 2);
            }
        }

        private void RecordCommandBuffer(VulkanCommandBuffer cmd, uint imageIndex)
        {
            cmd.Begin(CommandBufferUsageFlag.OneTimeSubmit);
            var extent = _rhi.Swapchain.Extent;
            var clearValues = new ClearValue[]
            {
                ClearValue.ColorClear(0.02f, 0.02f, 0.04f, 1.0f),
                ClearValue.DepthStencilClear(1.0f, 0)
            };

            cmd.BeginRenderPass(_renderPass, _framebuffers[imageIndex], clearValues);
            cmd.BindPipeline(_graphicsPipeline);
            cmd.SetViewport(new Viewport { X = 0, Y = 0, Width = extent.Width, Height = extent.Height, MinDepth = 0, MaxDepth = 1 });
            cmd.SetScissor(new Rect2D { Extent = new Extent2D(extent.Width, extent.Height) });
            cmd.BindVertexBuffer(_vertexBuffer);
            cmd.BindIndexBuffer(_indexBuffer, 0, IndexType.Uint16);
            cmd.DrawIndexed(_indexCount, 1, 0, 0, 0);
            cmd.EndRenderPass();
            cmd.End();
        }

        public bool ShouldQuit => _useGlfw && _window != IntPtr.Zero && GlfwWindow.ShouldClose(_window);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_rhi?.Device != IntPtr.Zero && _vkDeviceWaitIdle != IntPtr.Zero)
                Marshal.GetDelegateForFunctionPointer<DeviceWaitIdleDel>(_vkDeviceWaitIdle)(_rhi.Device);

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                if (_uniformBuffers[i] != null) { _uniformBuffers[i].Unmap(); _uniformBuffers[i].Dispose(); }
            }
            _vertexBuffer?.Dispose(); _indexBuffer?.Dispose();
            _sceneRenderer?.Dispose();
            _graphicsPipeline?.Dispose(); _pipelineLayout?.Dispose();
            _renderPass?.Dispose(); _descriptorPool?.Dispose(); _descriptorSetLayout?.Dispose();
            foreach (var fb in _framebuffers) fb?.Dispose();
            foreach (var di in _swapchainDepthImages) di?.Dispose();
            _rhi?.Dispose();
            if (_useGlfw && _window != IntPtr.Zero) GlfwWindow.DestroyWindow(_window);
        }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)] private delegate VulkanResult DeviceWaitIdleDel(IntPtr device);
    }

    internal static class MathHelper
    {
        public static float Deg2Rad(float deg) => deg * MathF.PI / 180.0f;
    }
}
