// =============================================================================
// VulkanDeviceManager.cs — Synapse Omnia: Unified Vulkan Device & Memory Management
// Implements device fusion, VMA integration, and VRAM optimization using VK_EXT_memory_budget.
// =============================================================================

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using GDNN.Platform;

namespace GDNN.RHI.Vulkan
{
    /// <summary>
    /// Memory budget information from VK_EXT_memory_budget extension.
    /// </summary>
    public struct MemoryBudget
    {
        public ulong HeapBudget;
        public ulong HeapUsage;
        public float UsagePercentage => HeapUsage / (float)HeapBudget * 100f;
    }

    /// <summary>
    /// Information about a Vulkan memory heap.
    /// </summary>
    public struct MemoryHeapInfo
    {
        public ulong Size;
        public MemoryPropertyFlag Flags;
        public MemoryBudget Budget;
    }

    /// <summary>
    /// Configuration for Vulkan device creation.
    /// </summary>
    public class VulkanDeviceConfig
    {
        public string ApplicationName { get; set; } = "Synapse Engine";
        public uint ApplicationVersion { get; set; } = 0x00010000;
        public string EngineName { get; set; } = "Synapse";
        public uint EngineVersion { get; set; } = 0x00010000;
        public uint ApiVersion { get; set; } = VulkanApi.VK_API_VERSION_1_3;
        public bool EnableValidation { get; set; } = true;
        public string[] RequiredInstanceExtensions { get; set; } = Array.Empty<string>();
        public string[] RequiredDeviceExtensions { get; set; } = Array.Empty<string>();
        public Feature Requirements { get; set; } = new Feature();
    }

    /// <summary>
    /// Unified Vulkan device manager that handles instance, physical device, logical device,
    /// and memory allocation through VMA (Vulkan Memory Allocator).
    /// This implements device fusion to reduce memory overhead and synchronization costs.
    /// </summary>
    public class VulkanDeviceManager : IDisposable
    {
        // Vulkan instance and devices
        private IntPtr _instance;
        private IntPtr _physicalDevice;
        private IntPtr _device;
        
        // VMA allocator
        private IntPtr _vmaAllocator;
        
        // Queues
        private QueueInfo _graphicsQueue;
        private QueueInfo _computeQueue;
        private QueueInfo _transferQueue;
        
        // Memory information
        private MemoryHeapInfo[] _memoryHeaps;
        private MemoryBudget[] _memoryBudgets;
        
        // Capabilities
        private PhysicalDeviceFeatures _features;
        private PhysicalDeviceProperties _properties;
        private PhysicalDeviceMemoryProperties _memoryProperties;
        
        // Extension support
        private HashSet<string> _instanceExtensions;
        private HashSet<string> _deviceExtensions;
        
        // Debug messenger
        private IntPtr _debugMessenger;
        
        // Platform capabilities
        private PlatformCapabilities _platformCapabilities;
        
        // Disposed flag
        private bool _disposed;

        /// <summary>Vulkan instance handle.</summary>
        public IntPtr Instance => _instance;

        /// <summary>Physical device handle.</summary>
        public IntPtr PhysicalDevice => _physicalDevice;

        /// <summary>Logical device handle.</summary>
        public IntPtr Device => _device;

        /// <summary>VMA allocator handle.</summary>
        public IntPtr VmaAllocator => _vmaAllocator;

        /// <summary>Graphics queue information.</summary>
        public QueueInfo GraphicsQueue => _graphicsQueue;

        /// <summary>Compute queue information.</summary>
        public QueueInfo ComputeQueue => _computeQueue;

        /// <summary>Transfer queue information.</summary>
        public QueueInfo TransferQueue => _transferQueue;

        /// <summary>Device features.</summary>
        public PhysicalDeviceFeatures Features => _features;

        /// <summary>Device properties.</summary>
        public PhysicalDeviceProperties Properties => _properties;

        /// <summary>Memory properties.</summary>
        public PhysicalDeviceMemoryProperties MemoryProperties => _memoryProperties;

        /// <summary>Platform capabilities.</summary>
        public PlatformCapabilities PlatformCapabilities => _platformCapabilities;

        /// <summary>
        /// Initializes a new VulkanDeviceManager with the specified configuration.
        /// </summary>
        public VulkanDeviceManager(VulkanDeviceConfig? config = null)
        {
            _platformCapabilities = NativePlatform.Probe();
            config ??= new VulkanDeviceConfig();
            
            // Ensure required extensions are included
            var requiredExtensions = new HashSet<string>(config.RequiredInstanceExtensions);
            var surfaceExtensions = VulkanSurfaceExtensions.GetRequiredSurfaceExtensions();
            foreach (var ext in surfaceExtensions)
                requiredExtensions.Add(ext);
            
            // Add debug utils if validation is enabled
            if (config.EnableValidation)
                requiredExtensions.Add(VulkanApi.VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            
            // Add memory budget extension if available
            requiredExtensions.Add(VulkanApi.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
            
            config.RequiredInstanceExtensions = requiredExtensions.ToArray();
            
            InitializeInstance(config);
            PickPhysicalDevice(config);
            InitializeDevice(config);
            InitializeVmaAllocator();
            QueryMemoryInformation();
        }

        /// <summary>
        /// Initializes the Vulkan instance.
        /// </summary>
        private unsafe void InitializeInstance(VulkanDeviceConfig config)
        {
            var appInfo = new VkApplicationInfo
            {
                sType = VkStructureType.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pApplicationName = Marshal.StringToHGlobalAnsi(config.ApplicationName),
                applicationVersion = config.ApplicationVersion,
                pEngineName = Marshal.StringToHGlobalAnsi(config.EngineName),
                engineVersion = config.EngineVersion,
                apiVersion = config.ApiVersion
            };

            var instanceInfo = new VkInstanceCreateInfo
            {
                sType = VkStructureType.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo = ref appInfo
            };

            try
            {
                // Set up instance extensions
                var extensions = config.RequiredInstanceExtensions;
                if (extensions.Length > 0)
                {
                    var extensionNames = new IntPtr[extensions.Length];
                    for (int i = 0; i < extensions.Length; i++)
                        extensionNames[i] = Marshal.StringToHGlobalAnsi(extensions[i]);
                    
                    instanceInfo.enabledExtensionCount = (uint)extensions.Length;
                    instanceInfo.ppEnabledExtensionNames = extensionNames;
                }

                // Set up validation layers if enabled
                if (config.EnableValidation)
                {
                    var validationLayer = VulkanApi.VK_LAYER_KHRONOS_VALIDATION;
                    instanceInfo.enabledLayerCount = 1;
                    instanceInfo.ppEnabledLayerNames = new IntPtr[] { Marshal.StringToHGlobalAnsi(validationLayer) };
                }

                var vkCreateInstance = NativeLibraryResolver.GetVulkanExport("vkCreateInstance");
                var result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkCreateInstance>(vkCreateInstance)(
                    ref instanceInfo, IntPtr.Zero, out _instance);

                if (result != VulkanResult.Success)
                    throw new VulkanException("Failed to create Vulkan instance", result);

                // Store instance extensions
                _instanceExtensions = new HashSet<string>(extensions);

                // Initialize debug messenger if validation is enabled
                if (config.EnableValidation)
                    SetupDebugMessenger();
            }
            finally
            {
                Marshal.FreeHGlobal(appInfo.pApplicationName);
                Marshal.FreeHGlobal(appInfo.pEngineName);
            }
        }

        /// <summary>
        /// Sets up the debug messenger for validation layers.
        /// </summary>
        private unsafe void SetupDebugMessenger()
        {
            var messengerInfo = new VkDebugUtilsMessengerCreateInfoEXT
            {
                sType = VkStructureType.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                messageSeverity = VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                messageType = VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                pfnUserCallback = Marshal.GetFunctionPointerForDelegate((PFN_vkDebugUtilsCallbackEXT)DebugCallback)
            };

            var vkCreateDebugUtilsMessenger = NativeLibraryResolver.GetVulkanExport("vkCreateDebugUtilsMessengerEXT");
            if (vkCreateDebugUtilsMessenger != IntPtr.Zero)
            {
                var result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkCreateDebugUtilsMessengerEXT>(vkCreateDebugUtilsMessenger)(
                    _instance, ref messengerInfo, IntPtr.Zero, out _debugMessenger);
                
                if (result != VulkanResult.Success)
                    _debugMessenger = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Debug callback for validation layers.
        /// </summary>
        private static uint DebugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagBitsEXT messageTypes,
            IntPtr pCallbackData,
            IntPtr pUserData)
        {
            var callbackData = Marshal.PtrToStructure<VkDebugUtilsMessengerCallbackDataEXT>(pCallbackData);
            var message = Marshal.PtrToStringAnsi(callbackData.pMessage);
            
            if (messageSeverity == VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                Console.Error.WriteLine($"[Vulkan Validation Error] {message}");
            else
                Console.WriteLine($"[Vulkan Validation] {message}");
            
            return VulkanApi.VK_FALSE;
        }

        /// <summary>
        /// Picks the best physical device based on the configuration requirements.
        /// </summary>
        private unsafe void PickPhysicalDevice(VulkanDeviceConfig config)
        {
            uint deviceCount = 0;
            var vkEnumeratePhysicalDevices = NativeLibraryResolver.GetVulkanExport("vkEnumeratePhysicalDevices");
            var result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkEnumeratePhysicalDevices>(vkEnumeratePhysicalDevices)(
                _instance, ref deviceCount, IntPtr.Zero);

            if (result != VulkanResult.Success || deviceCount == 0)
                throw new VulkanException("Failed to enumerate physical devices", result);

            var devices = new IntPtr[deviceCount];
            fixed (IntPtr* devicesPtr = devices)
            {
                result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkEnumeratePhysicalDevices>(vkEnumeratePhysicalDevices)(
                    _instance, ref deviceCount, (IntPtr)devicesPtr);
                
                if (result != VulkanResult.Success)
                    throw new VulkanException("Failed to enumerate physical devices", result);
            }

            // Evaluate each device
            int bestScore = -1;
            for (int i = 0; i < devices.Length; i++)
            {
                int score = RateDevice(devices[i], config);
                if (score > bestScore)
                {
                    bestScore = score;
                    _physicalDevice = devices[i];
                }
            }

            if (_physicalDevice == IntPtr.Zero)
                throw new VulkanException("No suitable physical device found");

            // Query device properties and features
            QueryDeviceProperties();
            QueryDeviceFeatures();
            QueryDeviceExtensions();
        }

        /// <summary>
        /// Rates a physical device based on its capabilities.
        /// </summary>
        private int RateDevice(IntPtr device, VulkanDeviceConfig config)
        {
            int score = 0;

            // Query basic properties
            var vkGetPhysicalDeviceProperties = NativeLibraryResolver.GetVulkanExport("vkGetPhysicalDeviceProperties");
            var properties = Marshal.PtrToStructure<VkPhysicalDeviceProperties>(device);

            // Discrete GPUs have a significant performance advantage
            if (properties.deviceType == VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                score += 1000;
            else if (properties.deviceType == VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
                score += 500;

            // Check API version support
            if (properties.apiVersion >= VulkanApi.VK_API_VERSION_1_3)
                score += 100;

            // Query features
            var vkGetPhysicalDeviceFeatures = NativeLibraryResolver.GetVulkanExport("vkGetPhysicalDeviceFeatures");
            var features = Marshal.PtrToStructure<VkPhysicalDeviceFeatures>(device);

            // Check required features
            if (config.Requirements.GeometryShader && features.geometryShader == VulkanApi.VK_TRUE)
                score += 10;
            if (config.Requirements.TessellationShader && features.tessellationShader == VulkanApi.VK_TRUE)
                score += 5;
            if (features.fillModeNonSolid == VulkanApi.VK_TRUE)
                score += 5;
            if (features.wideLines == VulkanApi.VK_TRUE)
                score += 2;
            if (features.largePoints == VulkanApi.VK_TRUE)
                score += 2;

            // Check for required extensions
            uint extensionCount = 0;
            var vkEnumerateDeviceExtensionProperties = NativeLibraryResolver.GetVulkanExport("vkEnumerateDeviceExtensionProperties");
            var extResult = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkEnumerateDeviceExtensionProperties>(vkEnumerateDeviceExtensionProperties)(
                device, IntPtr.Zero, ref extensionCount, IntPtr.Zero);

            if (extResult == VulkanResult.Success && extensionCount > 0)
            {
                var extensions = new VkExtensionProperties[extensionCount];
                fixed (VkExtensionProperties* extPtr = extensions)
                {
                    extResult = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkEnumerateDeviceExtensionProperties>(vkEnumerateDeviceExtensionProperties)(
                        device, IntPtr.Zero, ref extensionCount, (IntPtr)extPtr);
                }

                if (extResult == VulkanResult.Success)
                {
                    var extensionNames = new HashSet<string>();
                    for (int i = 0; i < extensions.Length; i++)
                        extensionNames.Add(Marshal.PtrToStringAnsi(extensions[i].extensionName));

                    foreach (var requiredExt in config.RequiredDeviceExtensions)
                    {
                        if (extensionNames.Contains(requiredExt))
                            score += 50;
                        else
                            score -= 100; // Penalize if required extension is missing
                    }

                    // Check for mesh shader support (Vulkan 1.3+)
                    if (extensionNames.Contains(VulkanApi.VK_EXT_MESH_SHADER_EXTENSION_NAME))
                        score += 200;

                    // Check for memory budget support
                    if (extensionNames.Contains(VulkanApi.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
                        score += 50;

                    // Check for ray tracing support
                    if (extensionNames.Contains(VulkanApi.VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
                        score += 100;
                }
            }

            // Query memory properties
            var vkGetPhysicalDeviceMemoryProperties = NativeLibraryResolver.GetVulkanExport("vkGetPhysicalDeviceMemoryProperties");
            var memoryProps = Marshal.PtrToStructure<VkPhysicalDeviceMemoryProperties>(device);

            // Prefer devices with more VRAM
            for (int i = 0; i < memoryProps.memoryHeapCount; i++)
            {
                if ((memoryProps.memoryHeaps[i].flags & VkMemoryHeapFlagBits.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
                {
                    // Add score based on heap size (in MB)
                    score += (int)(memoryProps.memoryHeaps[i].size / (1024 * 1024));
                }
            }

            return Math.Max(score, 0);
        }

        /// <summary>
        /// Queries device properties.
        /// </summary>
        private unsafe void QueryDeviceProperties()
        {
            var vkGetPhysicalDeviceProperties = NativeLibraryResolver.GetVulkanExport("vkGetPhysicalDeviceProperties");
            var props = Marshal.PtrToStructure<VkPhysicalDeviceProperties>(_physicalDevice);
            
            _properties = new PhysicalDeviceProperties
            {
                ApiVersion = props.apiVersion,
                DriverVersion = props.driverVersion,
                VendorId = props.vendorID,
                DeviceId = props.deviceID,
                DeviceType = props.deviceType,
                DeviceName = Marshal.PtrToStringAnsi(props.deviceName),
                PipelineCacheUuid = new byte[VulkanApi.VK_UUID_SIZE]
            };
            
            fixed (byte* uuidPtr = _properties.PipelineCacheUuid)
            {
                Buffer.MemoryCopy((void*)props.pipelineCacheUUID, uuidPtr, VulkanApi.VK_UUID_SIZE, VulkanApi.VK_UUID_SIZE);
            }
        }

        /// <summary>
        /// Queries device features.
        /// </summary>
        private unsafe void QueryDeviceFeatures()
        {
            var vkGetPhysicalDeviceFeatures = NativeLibraryResolver.GetVulkanExport("vkGetPhysicalDeviceFeatures");
            var features = Marshal.PtrToStructure<VkPhysicalDeviceFeatures>(_physicalDevice);
            
            _features = new PhysicalDeviceFeatures
            {
                RobustBufferAccess = features.robustBufferAccess == VulkanApi.VK_TRUE,
                FullDrawIndexUint32 = features.fullDrawIndexUint32 == VulkanApi.VK_TRUE,
                ImageCubeArray = features.imageCubeArray == VulkanApi.VK_TRUE,
                IndependentBlend = features.independentBlend == VulkanApi.VK_TRUE,
                GeometryShader = features.geometryShader == VulkanApi.VK_TRUE,
                TessellationShader = features.tessellationShader == VulkanApi.VK_TRUE,
                SampleRateShading = features.sampleRateShading == VulkanApi.VK_TRUE,
                DualSrcBlend = features.dualSrcBlend == VulkanApi.VK_TRUE,
                LogicOp = features.logicOp == VulkanApi.VK_TRUE,
                MultiDrawIndirect = features.multiDrawIndirect == VulkanApi.VK_TRUE,
                DrawIndirectFirstInstance = features.drawIndirectFirstInstance == VulkanApi.VK_TRUE,
                DepthClamp = features.depthClamp == VulkanApi.VK_TRUE,
                DepthBiasClamp = features.depthBiasClamp == VulkanApi.VK_TRUE,
                FillModeNonSolid = features.fillModeNonSolid == VulkanApi.VK_TRUE,
                DepthBounds = features.depthBounds == VulkanApi.VK_TRUE,
                WideLines = features.wideLines == VulkanApi.VK_TRUE,
                LargePoints = features.largePoints == VulkanApi.VK_TRUE,
                AlphaToOne = features.alphaToOne == VulkanApi.VK_TRUE,
                MultiViewport = features.multiViewport == VulkanApi.VK_TRUE,
                SamplerAnisotropy = features.samplerAnisotropy == VulkanApi.VK_TRUE,
                TextureCompressionETC2 = features.textureCompressionETC2 == VulkanApi.VK_TRUE,
                TextureCompressionASTC_LDR = features.textureCompressionASTC_LDR == VulkanApi.VK_TRUE,
                TextureCompressionBC = features.textureCompressionBC == VulkanApi.VK_TRUE,
                OcclusionQueryPrecise = features.occlusionQueryPrecise == VulkanApi.VK_TRUE,
                PipelineStatisticsQuery = features.pipelineStatisticsQuery == VulkanApi.VK_TRUE,
                VertexPipelineStoresAndAtomics = features.vertexPipelineStoresAndAtomics == VulkanApi.VK_TRUE,
                FragmentStoresAndAtomics = features.fragmentStoresAndAtomics == VulkanApi.VK_TRUE,
                ShaderTessellationAndGeometryPointSize = features.shaderTessellationAndGeometryPointSize == VulkanApi.VK_TRUE,
                ShaderImageGatherExtended = features.shaderImageGatherExtended == VulkanApi.VK_TRUE,
                ShaderStorageImageExtendedFormats = features.shaderStorageImageExtendedFormats == VulkanApi.VK_TRUE,
                ShaderStorageImageMultisample = features.shaderStorageImageMultisample == VulkanApi.VK_TRUE,
                ShaderStorageImageReadWithoutFormat = features.shaderStorageImageReadWithoutFormat == VulkanApi.VK_TRUE,
                ShaderStorageImageWriteWithoutFormat = features.shaderStorageImageWriteWithoutFormat == VulkanApi.VK_TRUE,
                ShaderUniformBufferArrayDynamicIndexing = features.shaderUniformBufferArrayDynamicIndexing == VulkanApi.VK_TRUE,
                ShaderSampledImageArrayDynamicIndexing = features.shaderSampledImageArrayDynamicIndexing == VulkanApi.VK_TRUE,
                ShaderStorageBufferArrayDynamicIndexing = features.shaderStorageBufferArrayDynamicIndexing == VulkanApi.VK_TRUE,
                ShaderStorageImageArrayDynamicIndexing = features.shaderStorageImageArrayDynamicIndexing == VulkanApi.VK_TRUE,
                ShaderClipDistance = features.shaderClipDistance == VulkanApi.VK_TRUE,
                ShaderCullDistance = features.shaderCullDistance == VulkanApi.VK_TRUE,
                ShaderFloat64 = features.shaderFloat64 == VulkanApi.VK_TRUE,
                ShaderInt64 = features.shaderInt64 == VulkanApi.VK_TRUE,
                ShaderInt16 = features.shaderInt16 == VulkanApi.VK_TRUE,
                ShaderResourceResidency = features.shaderResourceResidency == VulkanApi.VK_TRUE,
                ShaderResourceMinLod = features.shaderResourceMinLod == VulkanApi.VK_TRUE,
                SparseBinding = features.sparseBinding == VulkanApi.VK_TRUE,
                SparseResidencyBuffer = features.sparseResidencyBuffer == VulkanApi.VK_TRUE,
                SparseResidencyImage2D = features.sparseResidencyImage2D == VulkanApi.VK_TRUE,
                SparseResidencyImage3D = features.sparseResidencyImage3D == VulkanApi.VK_TRUE,
                SparseResidency2Samples = features.sparseResidency2Samples == VulkanApi.VK_TRUE,
                SparseResidency4Samples = features.sparseResidency4Samples == VulkanApi.VK_TRUE,
                SparseResidency8Samples = features.sparseResidency8Samples == VulkanApi.VK_TRUE,
                SparseResidency16Samples = features.sparseResidency16Samples == VulkanApi.VK_TRUE,
                SparseResidencyAliased = features.sparseResidencyAliased == VulkanApi.VK_TRUE,
                VariableMultisampleRate = features.variableMultisampleRate == VulkanApi.VK_TRUE,
                InheritedQueries = features.inheritedQueries == VulkanApi.VK_TRUE
            };
        }

        /// <summary>
        /// Queries device extensions.
        /// </summary>
        private void QueryDeviceExtensions()
        {
            _deviceExtensions = new HashSet<string>();
            
            uint extensionCount = 0;
            var vkEnumerateDeviceExtensionProperties = NativeLibraryResolver.GetVulkanExport("vkEnumerateDeviceExtensionProperties");
            var result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkEnumerateDeviceExtensionProperties>(vkEnumerateDeviceExtensionProperties)(
                _physicalDevice, IntPtr.Zero, ref extensionCount, IntPtr.Zero);

            if (result == VulkanResult.Success && extensionCount > 0)
            {
                var extensions = new VkExtensionProperties[extensionCount];
                fixed (VkExtensionProperties* extPtr = extensions)
                {
                    result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkEnumerateDeviceExtensionProperties>(vkEnumerateDeviceExtensionProperties)(
                        _physicalDevice, IntPtr.Zero, ref extensionCount, (IntPtr)extPtr);
                }

                if (result == VulkanResult.Success)
                {
                    for (int i = 0; i < extensions.Length; i++)
                    {
                        var extName = Marshal.PtrToStringAnsi(extensions[i].extensionName);
                        if (!string.IsNullOrEmpty(extName))
                            _deviceExtensions.Add(extName);
                    }
                }
            }
        }

        /// <summary>
        /// Initializes the logical device.
        /// </summary>
        private unsafe void InitializeDevice(VulkanDeviceConfig config)
        {
            // Find queue families
            uint queueFamilyCount = 0;
            var vkGetPhysicalDeviceQueueFamilyProperties = NativeLibraryResolver.GetVulkanExport("vkGetPhysicalDeviceQueueFamilyProperties");
            Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkGetPhysicalDeviceQueueFamilyProperties>(vkGetPhysicalDeviceQueueFamilyProperties)(
                _physicalDevice, ref queueFamilyCount, IntPtr.Zero);

            var queueFamilies = new VkQueueFamilyProperties[queueFamilyCount];
            fixed (VkQueueFamilyProperties* qfPtr = queueFamilies)
            {
                Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkGetPhysicalDeviceQueueFamilyProperties>(vkGetPhysicalDeviceQueueFamilyProperties)(
                    _physicalDevice, ref queueFamilyCount, (IntPtr)qfPtr);
            }

            // Find suitable queue families
            int graphicsFamily = -1;
            int computeFamily = -1;
            int transferFamily = -1;
            int presentFamily = -1;

            for (int i = 0; i < queueFamilies.Length; i++)
            {
                if ((queueFamilies[i].queueFlags & VkQueueFlagBits.VK_QUEUE_GRAPHICS_BIT) != 0)
                {
                    if (graphicsFamily < 0) graphicsFamily = i;
                }
                if ((queueFamilies[i].queueFlags & VkQueueFlagBits.VK_QUEUE_COMPUTE_BIT) != 0)
                {
                    if (computeFamily < 0) computeFamily = i;
                }
                if ((queueFamilies[i].queueFlags & VkQueueFlagBits.VK_QUEUE_TRANSFER_BIT) != 0)
                {
                    if (transferFamily < 0) transferFamily = i;
                }
            }

            // Check for presentation support
            var vkGetPhysicalDeviceSurfaceSupportKHR = NativeLibraryResolver.GetVulkanExport("vkGetPhysicalDeviceSurfaceSupportKHR");
            if (vkGetPhysicalDeviceSurfaceSupportKHR != IntPtr.Zero && _platformCapabilities.GlfwAvailable)
            {
                for (int i = 0; i < queueFamilies.Length; i++)
                {
                    uint supported;
                    var result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(vkGetPhysicalDeviceSurfaceSupportKHR)(
                        _physicalDevice, i, _surface, out supported);
                    
                    if (result == VulkanResult.Success && supported == VulkanApi.VK_TRUE)
                    {
                        presentFamily = i;
                        break;
                    }
                }
            }

            // If no dedicated families found, use the first one that supports everything
            if (graphicsFamily < 0 || computeFamily < 0 || transferFamily < 0)
            {
                for (int i = 0; i < queueFamilies.Length; i++)
                {
                    if (graphicsFamily < 0 && (queueFamilies[i].queueFlags & VkQueueFlagBits.VK_QUEUE_GRAPHICS_BIT) != 0)
                        graphicsFamily = i;
                    if (computeFamily < 0 && (queueFamilies[i].queueFlags & VkQueueFlagBits.VK_QUEUE_COMPUTE_BIT) != 0)
                        computeFamily = i;
                    if (transferFamily < 0 && (queueFamilies[i].queueFlags & VkQueueFlagBits.VK_QUEUE_TRANSFER_BIT) != 0)
                        transferFamily = i;
                }
            }

            if (graphicsFamily < 0)
                throw new VulkanException("No graphics queue family found");

            // Use separate queues if available for better performance
            var queueCreateInfos = new List<VkDeviceQueueCreateInfo>();
            var queuePriorities = new float[] { 1.0f };

            if (graphicsFamily == computeFamily && graphicsFamily == transferFamily)
            {
                // Single queue family for all operations
                queueCreateInfos.Add(new VkDeviceQueueCreateInfo
                {
                    sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex = (uint)graphicsFamily,
                    queueCount = 1,
                    pQueuePriorities = (IntPtr)queuePriorities
                });
                
                _graphicsQueue = new QueueInfo(graphicsFamily, 0);
                _computeQueue = new QueueInfo(graphicsFamily, 0);
                _transferQueue = new QueueInfo(graphicsFamily, 0);
            }
            else
            {
                // Separate queues for different operations
                queueCreateInfos.Add(new VkDeviceQueueCreateInfo
                {
                    sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex = (uint)graphicsFamily,
                    queueCount = 1,
                    pQueuePriorities = (IntPtr)queuePriorities
                });
                
                if (computeFamily != graphicsFamily)
                {
                    queueCreateInfos.Add(new VkDeviceQueueCreateInfo
                    {
                        sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                        queueFamilyIndex = (uint)computeFamily,
                        queueCount = 1,
                        pQueuePriorities = (IntPtr)queuePriorities
                    });
                }

                if (transferFamily != graphicsFamily && transferFamily != computeFamily)
                {
                    queueCreateInfos.Add(new VkDeviceQueueCreateInfo
                    {
                        sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                        queueFamilyIndex = (uint)transferFamily,
                        queueCount = 1,
                        pQueuePriorities = (IntPtr)queuePriorities
                    });
                }

                _graphicsQueue = new QueueInfo(graphicsFamily, 0);
                _computeQueue = new QueueInfo(computeFamily >= 0 ? computeFamily : graphicsFamily, computeFamily >= 0 ? 0 : 0);
                _transferQueue = new QueueInfo(transferFamily >= 0 ? transferFamily : graphicsFamily, transferFamily >= 0 ? 0 : 0);
            }

            // Device features
            var deviceFeatures = new VkPhysicalDeviceFeatures();
            // TODO: Enable features based on config

            var deviceInfo = new VkDeviceCreateInfo
            {
                sType = VkStructureType.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                queueCreateInfoCount = (uint)queueCreateInfos.Count,
                pQueueCreateInfos = (IntPtr)queueCreateInfos.ToArray(),
                pEnabledFeatures = ref deviceFeatures
            };

            // Enable required extensions
            var enabledExtensions = new List<string>(config.RequiredDeviceExtensions);
            
            // Always enable swapchain extension
            enabledExtensions.Add(VulkanApi.VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            
            // Enable mesh shader if supported
            if (_deviceExtensions.Contains(VulkanApi.VK_EXT_MESH_SHADER_EXTENSION_NAME))
                enabledExtensions.Add(VulkanApi.VK_EXT_MESH_SHADER_EXTENSION_NAME);
            
            // Enable ray tracing if supported
            if (_deviceExtensions.Contains(VulkanApi.VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))
                enabledExtensions.Add(VulkanApi.VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
            
            // Enable memory budget if supported
            if (_deviceExtensions.Contains(VulkanApi.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
                enabledExtensions.Add(VulkanApi.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

            if (enabledExtensions.Count > 0)
            {
                var extensionNames = new IntPtr[enabledExtensions.Count];
                for (int i = 0; i < enabledExtensions.Count; i++)
                    extensionNames[i] = Marshal.StringToHGlobalAnsi(enabledExtensions[i]);
                
                deviceInfo.enabledExtensionCount = (uint)enabledExtensions.Count;
                deviceInfo.ppEnabledExtensionNames = extensionNames;
            }

            var vkCreateDevice = NativeLibraryResolver.GetVulkanExport("vkCreateDevice");
            var result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkCreateDevice>(vkCreateDevice)(
                _physicalDevice, ref deviceInfo, IntPtr.Zero, out _device);

            if (result != VulkanResult.Success)
                throw new VulkanException("Failed to create logical device", result);

            // Store enabled device extensions
            _deviceExtensions.UnionWith(enabledExtensions);

            // Get queues
            var vkGetDeviceQueue = NativeLibraryResolver.GetVulkanExport("vkGetDeviceQueue");
            _graphicsQueue.Queue = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkGetDeviceQueue>(vkGetDeviceQueue)(
                _device, _graphicsQueue.FamilyIndex, _graphicsQueue.QueueIndex);
            _computeQueue.Queue = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkGetDeviceQueue>(vkGetDeviceQueue)(
                _device, _computeQueue.FamilyIndex, _computeQueue.QueueIndex);
            _transferQueue.Queue = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkGetDeviceQueue>(vkGetDeviceQueue)(
                _device, _transferQueue.FamilyIndex, _transferQueue.QueueIndex);
        }

        /// <summary>
        /// Initializes the VMA (Vulkan Memory Allocator) allocator.
        /// </summary>
        private unsafe void InitializeVmaAllocator()
        {
            // Load VMA library
            var vmaLib = NativeLibraryResolver.LoadVma();
            if (vmaLib == IntPtr.Zero)
            {
                Console.WriteLine("Warning: VMA library not found. Falling back to manual memory management.");
                return;
            }

            var allocatorInfo = new VmaAllocatorCreateInfo
            {
                physicalDevice = _physicalDevice,
                device = _device,
                instance = _instance,
                // Use default allocation callbacks
                pAllocationCallbacks = IntPtr.Zero,
                // Enable buffer device address for Vulkan 1.2+
                flags = VmaAllocatorCreateFlagBits.VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT | 
                       VmaAllocatorCreateFlagBits.VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT
            };

            var vmaCreateAllocator = NativeLibraryResolver.GetExport(vmaLib, "vmaCreateAllocator");
            var result = Marshal.GetDelegateForFunctionPointer<Vma.PFN_vmaCreateAllocator>(vmaCreateAllocator)(
                ref allocatorInfo, out _vmaAllocator);

            if (result != VulkanResult.Success)
            {
                Console.WriteLine("Warning: Failed to create VMA allocator. Falling back to manual memory management.");
                _vmaAllocator = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Queries memory information from the physical device.
        /// </summary>
        private unsafe void QueryMemoryInformation()
        {
            var vkGetPhysicalDeviceMemoryProperties = NativeLibraryResolver.GetVulkanExport("vkGetPhysicalDeviceMemoryProperties");
            var memoryProps = Marshal.PtrToStructure<VkPhysicalDeviceMemoryProperties>(_physicalDevice);

            _memoryProperties = new PhysicalDeviceMemoryProperties
            {
                MemoryTypes = new MemoryType[memoryProps.memoryTypeCount],
                MemoryHeaps = new MemoryHeap[memoryProps.memoryHeapCount]
            };

            for (int i = 0; i < memoryProps.memoryTypeCount; i++)
            {
                _memoryProperties.MemoryTypes[i] = new MemoryType
                {
                    HeapIndex = memoryProps.memoryTypes[i].heapIndex,
                    PropertyFlags = memoryProps.memoryTypes[i].propertyFlags
                };
            }

            for (int i = 0; i < memoryProps.memoryHeapCount; i++)
            {
                _memoryProperties.MemoryHeaps[i] = new MemoryHeap
                {
                    Size = memoryProps.memoryHeaps[i].size,
                    Flags = memoryProps.memoryHeaps[i].flags
                };
            }

            // Query memory budgets if extension is supported
            if (_deviceExtensions.Contains(VulkanApi.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
            {
                _memoryBudgets = new MemoryBudget[memoryProps.memoryHeapCount];
                var vkGetPhysicalDeviceMemoryBudgetPropertiesEXT = NativeLibraryResolver.GetVulkanExport("vkGetPhysicalDeviceMemoryBudgetPropertiesEXT");
                
                for (int i = 0; i < memoryProps.memoryHeapCount; i++)
                {
                    var budgetProps = new VkPhysicalDeviceMemoryBudgetPropertiesEXT
                    {
                        sType = VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
                    };
                    
                    var nextPtr = (IntPtr)(&budgetProps);
                    var props = new VkPhysicalDeviceProperties2
                    {
                        sType = VkStructureType.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                        pNext = nextPtr
                    };
                    
                    Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkGetPhysicalDeviceProperties2>(vkGetPhysicalDeviceProperties2)(
                        _physicalDevice, ref props);
                    
                    _memoryBudgets[i] = new MemoryBudget
                    {
                        HeapBudget = budgetProps.heapBudget[i],
                        HeapUsage = budgetProps.heapUsage[i]
                    };
                }
            }
        }

        /// <summary>
        /// Allocates memory using VMA.
        /// </summary>
        public unsafe IntPtr AllocateMemory(ulong size, ulong memoryTypeBits, MemoryPropertyFlag requiredFlags, MemoryPropertyFlag preferredFlags = 0)
        {
            if (_vmaAllocator == IntPtr.Zero)
                throw new InvalidOperationException("VMA allocator not initialized");

            var allocInfo = new VmaAllocationCreateInfo
            {
                flags = VmaAllocationCreateFlagBits.VMA_ALLOCATION_CREATE_MAPPED_BIT,
                usage = VmaMemoryUsage.VMA_MEMORY_USAGE_AUTO,
                requiredFlags = requiredFlags,
                preferredFlags = preferredFlags,
                memoryTypeBits = memoryTypeBits
            };

            var vmaAllocateMemory = NativeLibraryResolver.GetExport(NativeLibraryResolver.VmaLibrary, "vmaAllocateMemory");
            IntPtr allocation;
            IntPtr memory;
            var result = Marshal.GetDelegateForFunctionPointer<Vma.PFN_vmaAllocateMemory>(vmaAllocateMemory)(
                _vmaAllocator, size, ref allocInfo, out memory, out allocation);

            if (result != VulkanResult.Success)
                throw new VulkanException("Failed to allocate memory", result);

            return memory;
        }

        /// <summary>
        /// Allocates and binds memory to a buffer.
        /// </summary>
        public unsafe VulkanBuffer CreateBuffer(ulong size, BufferUsageFlag usage, MemoryPropertyFlag memoryProperties)
        {
            if (_vmaAllocator == IntPtr.Zero)
                throw new InvalidOperationException("VMA allocator not initialized");

            var bufferInfo = new VkBufferCreateInfo
            {
                sType = VkStructureType.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size = size,
                usage = usage,
                sharingMode = VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount = 0,
                pQueueFamilyIndices = IntPtr.Zero
            };

            var vkCreateBuffer = NativeLibraryResolver.GetVulkanExport("vkCreateBuffer");
            IntPtr buffer;
            var result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkCreateBuffer>(vkCreateBuffer)(
                _device, ref bufferInfo, IntPtr.Zero, out buffer);

            if (result != VulkanResult.Success)
                throw new VulkanException("Failed to create buffer", result);

            // Get memory requirements
            var vkGetBufferMemoryRequirements = NativeLibraryResolver.GetVulkanExport("vkGetBufferMemoryRequirements");
            VkMemoryRequirements memRequirements;
            Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkGetBufferMemoryRequirements>(vkGetBufferMemoryRequirements)(
                _device, buffer, out memRequirements);

            // Find memory type
            uint memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, memoryProperties);

            // Allocate memory using VMA
            var allocInfo = new VmaAllocationCreateInfo
            {
                usage = VmaMemoryUsage.VMA_MEMORY_USAGE_AUTO,
                requiredFlags = memoryProperties,
                memoryTypeBits = memRequirements.memoryTypeBits
            };

            var vmaAllocateMemoryForBuffer = NativeLibraryResolver.GetExport(NativeLibraryResolver.VmaLibrary, "vmaAllocateMemoryForBuffer");
            IntPtr allocation;
            result = Marshal.GetDelegateForFunctionPointer<Vma.PFN_vmaAllocateMemoryForBuffer>(vmaAllocateMemoryForBuffer)(
                _vmaAllocator, buffer, ref allocInfo, out allocation);

            if (result != VulkanResult.Success)
                throw new VulkanException("Failed to allocate memory for buffer", result);

            // Bind memory to buffer
            var vkBindBufferMemory = NativeLibraryResolver.GetVulkanExport("vkBindBufferMemory");
            result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkBindBufferMemory>(vkBindBufferMemory)(
                _device, buffer, allocation, 0);

            if (result != VulkanResult.Success)
                throw new VulkanException("Failed to bind buffer memory", result);

            return new VulkanBuffer(buffer, allocation, size, usage, memoryProperties);
        }

        /// <summary>
        /// Creates an image with allocated memory.
        /// </summary>
        public unsafe VulkanTexture CreateTexture(TextureDescription description)
        {
            if (_vmaAllocator == IntPtr.Zero)
                throw new InvalidOperationException("VMA allocator not initialized");

            var imageInfo = new VkImageCreateInfo
            {
                sType = VkStructureType.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType = description.Dimension == TextureDimension.Dim2D ? VkImageType.VK_IMAGE_TYPE_2D : VkImageType.VK_IMAGE_TYPE_3D,
                extent = new VkExtent3D { width = (uint)description.Width, height = (uint)description.Height, depth = (uint)description.Depth },
                mipLevels = description.MipLevels,
                arrayLayers = description.ArrayLayers,
                format = description.Format,
                tiling = description.Tiling,
                initialLayout = description.InitialLayout,
                usage = description.Usage,
                sharingMode = VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
                samples = description.Samples,
                flags = description.Flags
            };

            var vkCreateImage = NativeLibraryResolver.GetVulkanExport("vkCreateImage");
            IntPtr image;
            var result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkCreateImage>(vkCreateImage)(
                _device, ref imageInfo, IntPtr.Zero, out image);

            if (result != VulkanResult.Success)
                throw new VulkanException("Failed to create image", result);

            // Get memory requirements
            var vkGetImageMemoryRequirements = NativeLibraryResolver.GetVulkanExport("vkGetImageMemoryRequirements");
            VkMemoryRequirements memRequirements;
            Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkGetImageMemoryRequirements>(vkGetImageMemoryRequirements)(
                _device, image, out memRequirements);

            // Allocate memory using VMA
            var allocInfo = new VmaAllocationCreateInfo
            {
                usage = VmaMemoryUsage.VMA_MEMORY_USAGE_AUTO,
                requiredFlags = MemoryPropertyFlag.DeviceLocal,
                memoryTypeBits = memRequirements.memoryTypeBits
            };

            var vmaAllocateMemoryForImage = NativeLibraryResolver.GetExport(NativeLibraryResolver.VmaLibrary, "vmaAllocateMemoryForImage");
            IntPtr allocation;
            result = Marshal.GetDelegateForFunctionPointer<Vma.PFN_vmaAllocateMemoryForImage>(vmaAllocateMemoryForImage)(
                _vmaAllocator, image, ref allocInfo, out allocation);

            if (result != VulkanResult.Success)
                throw new VulkanException("Failed to allocate memory for image", result);

            // Bind memory to image
            var vkBindImageMemory = NativeLibraryResolver.GetVulkanExport("vkBindImageMemory");
            result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkBindImageMemory>(vkBindImageMemory)(
                _device, image, allocation, 0);

            if (result != VulkanResult.Success)
                throw new VulkanException("Failed to bind image memory", result);

            return new VulkanTexture(image, allocation, description);
        }

        /// <summary>
        /// Finds a memory type that satisfies the specified requirements.
        /// </summary>
        private uint FindMemoryType(uint typeFilter, MemoryPropertyFlag properties)
        {
            for (uint i = 0; i < _memoryProperties.MemoryTypeCount; i++)
            {
                if ((typeFilter & (1 << (int)i)) != 0 &&
                    (_memoryProperties.MemoryTypes[i].PropertyFlags & properties) == properties)
                {
                    return i;
                }
            }
            throw new VulkanException("Failed to find suitable memory type");
        }

        /// <summary>
        /// Gets the current VRAM usage percentage.
        /// </summary>
        public float GetVramUsagePercentage()
        {
            if (_memoryBudgets == null || _memoryBudgets.Length == 0)
                return 0f;

            // Find device-local heap
            for (int i = 0; i < _memoryBudgets.Length; i++)
            {
                if ((_memoryProperties.MemoryHeaps[i].Flags & MemoryHeapFlag.DeviceLocal) != 0)
                {
                    return _memoryBudgets[i].UsagePercentage;
                }
            }
            return 0f;
        }

        /// <summary>
        /// Gets the available VRAM in bytes.
        /// </summary>
        public ulong GetAvailableVram()
        {
            if (_memoryBudgets == null || _memoryBudgets.Length == 0)
                return 0;

            for (int i = 0; i < _memoryBudgets.Length; i++)
            {
                if ((_memoryProperties.MemoryHeaps[i].Flags & MemoryHeapFlag.DeviceLocal) != 0)
                {
                    return _memoryBudgets[i].HeapBudget - _memoryBudgets[i].HeapUsage;
                }
            }
            return 0;
        }

        /// <summary>
        /// Checks if a specific extension is supported.
        /// </summary>
        public bool IsExtensionSupported(string extensionName) => _deviceExtensions.Contains(extensionName);

        /// <summary>
        /// Checks if mesh shaders are supported.
        /// </summary>
        public bool IsMeshShaderSupported() => IsExtensionSupported(VulkanApi.VK_EXT_MESH_SHADER_EXTENSION_NAME);

        /// <summary>
        /// Checks if ray tracing is supported.
        /// </summary>
        public bool IsRayTracingSupported() => IsExtensionSupported(VulkanApi.VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);

        /// <summary>
        /// Checks if memory budget extension is supported.
        /// </summary>
        public bool IsMemoryBudgetSupported() => IsExtensionSupported(VulkanApi.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

        /// <summary>
        /// Waits for the device to become idle.
        /// </summary>
        public unsafe void WaitForIdle()
        {
            var vkDeviceWaitIdle = NativeLibraryResolver.GetVulkanExport("vkDeviceWaitIdle");
            var result = Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkDeviceWaitIdle>(vkDeviceWaitIdle)(_device);
            
            if (result != VulkanResult.Success)
                throw new VulkanException("Failed to wait for device idle", result);
        }

        /// <summary>
        /// Disposes the VulkanDeviceManager and releases all resources.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;

            // Destroy VMA allocator
            if (_vmaAllocator != IntPtr.Zero)
            {
                var vmaDestroyAllocator = NativeLibraryResolver.GetExport(NativeLibraryResolver.VmaLibrary, "vmaDestroyAllocator");
                Marshal.GetDelegateForFunctionPointer<Vma.PFN_vmaDestroyAllocator>(vmaDestroyAllocator)(_vmaAllocator);
                _vmaAllocator = IntPtr.Zero;
            }

            // Destroy debug messenger
            if (_debugMessenger != IntPtr.Zero)
            {
                var vkDestroyDebugUtilsMessenger = NativeLibraryResolver.GetVulkanExport("vkDestroyDebugUtilsMessengerEXT");
                if (vkDestroyDebugUtilsMessenger != IntPtr.Zero)
                {
                    Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkDestroyDebugUtilsMessengerEXT>(vkDestroyDebugUtilsMessenger)(
                        _instance, _debugMessenger, IntPtr.Zero);
                }
                _debugMessenger = IntPtr.Zero;
            }

            // Destroy device
            if (_device != IntPtr.Zero)
            {
                var vkDestroyDevice = NativeLibraryResolver.GetVulkanExport("vkDestroyDevice");
                Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkDestroyDevice>(vkDestroyDevice)(_device, IntPtr.Zero);
                _device = IntPtr.Zero;
            }

            // Destroy instance
            if (_instance != IntPtr.Zero)
            {
                var vkDestroyInstance = NativeLibraryResolver.GetVulkanExport("vkDestroyInstance");
                Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkDestroyInstance>(vkDestroyInstance)(_instance, IntPtr.Zero);
                _instance = IntPtr.Zero;
            }

            _physicalDevice = IntPtr.Zero;
            GC.SuppressFinalize(this);
        }

        ~VulkanDeviceManager() => Dispose();
    }

    /// <summary>
    /// Information about a Vulkan queue.
    /// </summary>
    public readonly struct QueueInfo
    {
        public readonly int FamilyIndex;
        public readonly int QueueIndex;
        public IntPtr Queue;

        public QueueInfo(int familyIndex, int queueIndex)
        {
            FamilyIndex = familyIndex;
            QueueIndex = queueIndex;
            Queue = IntPtr.Zero;
        }
    }

    /// <summary>
    /// Vulkan buffer with allocated memory.
    /// </summary>
    public class VulkanBuffer : IDisposable
    {
        public IntPtr Handle { get; }
        public IntPtr Allocation { get; }
        public ulong Size { get; }
        public BufferUsageFlag Usage { get; }
        public MemoryPropertyFlag MemoryProperties { get; }

        public VulkanBuffer(IntPtr handle, IntPtr allocation, ulong size, BufferUsageFlag usage, MemoryPropertyFlag memoryProperties)
        {
            Handle = handle;
            Allocation = allocation;
            Size = size;
            Usage = usage;
            MemoryProperties = memoryProperties;
        }

        public unsafe IntPtr Map()
        {
            var vmaMapMemory = NativeLibraryResolver.GetExport(NativeLibraryResolver.VmaLibrary, "vmaMapMemory");
            IntPtr data;
            var result = Marshal.GetDelegateForFunctionPointer<Vma.PFN_vmaMapMemory>(vmaMapMemory)(
                Allocation, out data);
            
            if (result != VulkanResult.Success)
                throw new VulkanException("Failed to map memory", result);
            
            return data;
        }

        public unsafe void Unmap()
        {
            var vmaUnmapMemory = NativeLibraryResolver.GetExport(NativeLibraryResolver.VmaLibrary, "vmaUnmapMemory");
            Marshal.GetDelegateForFunctionPointer<Vma.PFN_vmaUnmapMemory>(vmaUnmapMemory)(Allocation);
        }

        public void Dispose()
        {
            if (Handle != IntPtr.Zero)
            {
                var vkDestroyBuffer = NativeLibraryResolver.GetVulkanExport("vkDestroyBuffer");
                Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkDestroyBuffer>(vkDestroyBuffer)(
                    IntPtr.Zero, Handle, IntPtr.Zero);
            }
            
            if (Allocation != IntPtr.Zero)
            {
                var vmaFreeMemory = NativeLibraryResolver.GetExport(NativeLibraryResolver.VmaLibrary, "vmaFreeMemory");
                Marshal.GetDelegateForFunctionPointer<Vma.PFN_vmaFreeMemory>(vmaFreeMemory)(Allocation);
            }
        }
    }

    /// <summary>
    /// Vulkan texture with allocated memory.
    /// </summary>
    public class VulkanTexture : IDisposable
    {
        public IntPtr Handle { get; }
        public IntPtr Allocation { get; }
        public TextureDescription Description { get; }

        public VulkanTexture(IntPtr handle, IntPtr allocation, TextureDescription description)
        {
            Handle = handle;
            Allocation = allocation;
            Description = description;
        }

        public void Dispose()
        {
            if (Handle != IntPtr.Zero)
            {
                var vkDestroyImage = NativeLibraryResolver.GetVulkanExport("vkDestroyImage");
                Marshal.GetDelegateForFunctionPointer<VulkanApi.PFN_vkDestroyImage>(vkDestroyImage)(
                    IntPtr.Zero, Handle, IntPtr.Zero);
            }
            
            if (Allocation != IntPtr.Zero)
            {
                var vmaFreeMemory = NativeLibraryResolver.GetExport(NativeLibraryResolver.VmaLibrary, "vmaFreeMemory");
                Marshal.GetDelegateForFunctionPointer<Vma.PFN_vmaFreeMemory>(vmaFreeMemory)(Allocation);
            }
        }
    }

    /// <summary>
    /// Feature flags for Vulkan device.
    /// </summary>
    public class Feature
    {
        public bool RobustBufferAccess { get; set; }
        public bool FullDrawIndexUint32 { get; set; }
        public bool ImageCubeArray { get; set; }
        public bool IndependentBlend { get; set; }
        public bool GeometryShader { get; set; }
        public bool TessellationShader { get; set; }
        public bool SampleRateShading { get; set; }
        public bool DualSrcBlend { get; set; }
        public bool LogicOp { get; set; }
        public bool MultiDrawIndirect { get; set; }
        public bool DrawIndirectFirstInstance { get; set; }
        public bool DepthClamp { get; set; }
        public bool DepthBiasClamp { get; set; }
        public bool FillModeNonSolid { get; set; }
        public bool DepthBounds { get; set; }
        public bool WideLines { get; set; }
        public bool LargePoints { get; set; }
        public bool AlphaToOne { get; set; }
        public bool MultiViewport { get; set; }
        public bool SamplerAnisotropy { get; set; }
        public bool TextureCompressionETC2 { get; set; }
        public bool TextureCompressionASTC_LDR { get; set; }
        public bool TextureCompressionBC { get; set; }
        public bool OcclusionQueryPrecise { get; set; }
        public bool PipelineStatisticsQuery { get; set; }
        public bool VertexPipelineStoresAndAtomics { get; set; }
        public bool FragmentStoresAndAtomics { get; set; }
        public bool ShaderTessellationAndGeometryPointSize { get; set; }
        public bool ShaderImageGatherExtended { get; set; }
        public bool ShaderStorageImageExtendedFormats { get; set; }
        public bool ShaderStorageImageMultisample { get; set; }
        public bool ShaderStorageImageReadWithoutFormat { get; set; }
        public bool ShaderStorageImageWriteWithoutFormat { get; set; }
        public bool ShaderUniformBufferArrayDynamicIndexing { get; set; }
        public bool ShaderSampledImageArrayDynamicIndexing { get; set; }
        public bool ShaderStorageBufferArrayDynamicIndexing { get; set; }
        public bool ShaderStorageImageArrayDynamicIndexing { get; set; }
        public bool ShaderClipDistance { get; set; }
        public bool ShaderCullDistance { get; set; }
        public bool ShaderFloat64 { get; set; }
        public bool ShaderInt64 { get; set; }
        public bool ShaderInt16 { get; set; }
        public bool ShaderResourceResidency { get; set; }
        public bool ShaderResourceMinLod { get; set; }
        public bool SparseBinding { get; set; }
        public bool SparseResidencyBuffer { get; set; }
        public bool SparseResidencyImage2D { get; set; }
        public bool SparseResidencyImage3D { get; set; }
        public bool SparseResidency2Samples { get; set; }
        public bool SparseResidency4Samples { get; set; }
        public bool SparseResidency8Samples { get; set; }
        public bool SparseResidency16Samples { get; set; }
        public bool SparseResidencyAliased { get; set; }
        public bool VariableMultisampleRate { get; set; }
        public bool InheritedQueries { get; set; }
    }

    /// <summary>
    /// Physical device properties.
    /// </summary>
    public class PhysicalDeviceProperties
    {
        public uint ApiVersion { get; set; }
        public uint DriverVersion { get; set; }
        public uint VendorId { get; set; }
        public uint DeviceId { get; set; }
        public VkPhysicalDeviceType DeviceType { get; set; }
        public string DeviceName { get; set; } = string.Empty;
        public byte[] PipelineCacheUuid { get; set; } = Array.Empty<byte>();
    }

    /// <summary>
    /// Physical device memory properties.
    /// </summary>
    public class PhysicalDeviceMemoryProperties
    {
        public MemoryType[] MemoryTypes { get; set; } = Array.Empty<MemoryType>();
        public MemoryHeap[] MemoryHeaps { get; set; } = Array.Empty<MemoryHeap>();
        public uint MemoryTypeCount => (uint)MemoryTypes.Length;
        public uint MemoryHeapCount => (uint)MemoryHeaps.Length;
    }

    /// <summary>
    /// Memory type information.
    /// </summary>
    public class MemoryType
    {
        public uint HeapIndex { get; set; }
        public MemoryPropertyFlag PropertyFlags { get; set; }
    }

    /// <summary>
    /// Memory heap information.
    /// </summary>
    public class MemoryHeap
    {
        public ulong Size { get; set; }
        public MemoryHeapFlag Flags { get; set; }
    }

    /// <summary>
    /// Texture description for image creation.
    /// </summary>
    public class TextureDescription
    {
        public uint Width { get; set; } = 1;
        public uint Height { get; set; } = 1;
        public uint Depth { get; set; } = 1;
        public uint MipLevels { get; set; } = 1;
        public uint ArrayLayers { get; set; } = 1;
        public VkFormat Format { get; set; } = VkFormat.VK_FORMAT_R8G8B8A8_UNORM;
        public VkImageType ImageType { get; set; } = VkImageType.VK_IMAGE_TYPE_2D;
        public VkImageTiling Tiling { get; set; } = VkImageTiling.VK_IMAGE_TILING_OPTIMAL;
        public VkImageLayout InitialLayout { get; set; } = VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED;
        public ImageUsageFlag Usage { get; set; } = ImageUsageFlag.Sampled;
        public SampleCountFlag Samples { get; set; } = SampleCountFlag.Count1;
        public VkImageCreateFlags Flags { get; set; } = 0;
        public TextureDimension Dimension { get; set; } = TextureDimension.Dim2D;
    }

    /// <summary>
    /// Texture dimension.
    /// </summary>
    public enum TextureDimension
    {
        Dim1D,
        Dim2D,
        Dim3D
    }

    /// <summary>
    /// Exception thrown for Vulkan errors.
    /// </summary>
    public class VulkanException : Exception
    {
        public VulkanResult Result { get; }

        public VulkanException(string message, VulkanResult result) : base($"{message} (Result: {result})")
        {
            Result = result;
        }
    }
}