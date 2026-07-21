using System.Runtime.InteropServices;
using Synapse.Infrastructure.Logging;

namespace Synapse.VR;

/// <summary>OpenXR session lifecycle with Vulkan2 backend (native when bound; simulated fallback for CI).</summary>
public interface IVrSession : IAsyncDisposable
{
    bool IsAvailable { get; }
    bool IsRunning { get; }
    bool IsSimulated { get; }
    bool UsesNativeOpenXr { get; }
    string RuntimeName { get; }
    OpenXrVulkanSwapchain? Swapchain { get; }

    Task<bool> TryInitializeAsync(
        int width = 1280,
        int height = 720,
        OpenXrVulkanBinding? vulkanBinding = null,
        CancellationToken cancellationToken = default);

    Task BeginFrameAsync(CancellationToken cancellationToken = default);

    Task EndFrameAsync(CancellationToken cancellationToken = default);

    Task PollEventsAsync(CancellationToken cancellationToken = default);
}

public sealed class OpenXrVulkanSession : IVrSession
{
    private readonly ISynapseLogger? _logger;
    private bool _initialized;
    private int _frameCounter;
    private ulong _instance;
    private ulong _session;
    private IntPtr _extensionNamesBlock;
    private IntPtr _vulkanExtName;

    public OpenXrVulkanSession(ISynapseLogger? logger = null) => _logger = logger;

    public bool IsAvailable { get; private set; }
    public bool IsRunning { get; private set; }
    public bool IsSimulated { get; private set; }
    public bool UsesNativeOpenXr { get; private set; }
    public string RuntimeName { get; private set; } = "none";
    public OpenXrVulkanSwapchain? Swapchain { get; private set; }

    public Task<bool> TryInitializeAsync(
        int width = 1280,
        int height = 720,
        OpenXrVulkanBinding? vulkanBinding = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (string.Equals(Environment.GetEnvironmentVariable("SYNAPSE_VR_FORCE_SIMULATED"), "1", StringComparison.Ordinal)
            || string.Equals(Environment.GetEnvironmentVariable("SYNAPSE_VR_SIMULATE"), "1", StringComparison.Ordinal))
            return Task.FromResult(InitializeSimulated(width, height, "forced-simulated"));

        if (!TryLoadOpenXrLoader())
            return Task.FromResult(InitializeSimulated(width, height, "no-loader"));

        if (TryInitializeNative(width, height, vulkanBinding))
        {
            UsesNativeOpenXr = true;
            IsSimulated = false;
            return Task.FromResult(true);
        }

        if (string.Equals(
            Environment.GetEnvironmentVariable("SYNAPSE_VR_SIMULATE"),
            "1",
            StringComparison.Ordinal)
            || string.Equals(
                Environment.GetEnvironmentVariable("SYNAPSE_VR_FORCE_SIMULATED"),
                "1",
                StringComparison.Ordinal))
            return Task.FromResult(InitializeSimulated(width, height, "OpenXR-Loader (simulated-fallback)"));

        IsAvailable = false;
        IsRunning = false;
        UsesNativeOpenXr = false;
        Swapchain = null;
        return Task.FromResult(false);
    }

    private bool InitializeSimulated(int width, int height, string runtimeLabel)
    {
        IsAvailable = true;
        IsRunning = true;
        IsSimulated = true;
        UsesNativeOpenXr = false;
        RuntimeName = runtimeLabel;
        Swapchain = OpenXrVulkanSwapchain.CreateSimulated(3, width, height);
        _initialized = true;
        _logger?.Info("VR", $"OpenXR simulated swapchain {width}x{height} ({RuntimeName})");
        return true;
    }

    private unsafe bool TryInitializeNative(int width, int height, OpenXrVulkanBinding? vulkanBinding)
    {
        try
        {
            // XR_KHR_vulkan_enable2
            const string extName = "XR_KHR_vulkan_enable2";
            _vulkanExtName = Marshal.StringToHGlobalAnsi(extName);
            _extensionNamesBlock = Marshal.AllocHGlobal(IntPtr.Size);
            Marshal.WriteIntPtr(_extensionNamesBlock, _vulkanExtName);

            var createInfo = new OpenXrNative.XrInstanceCreateInfo
            {
                type = OpenXrNative.XR_TYPE_INSTANCE_CREATE_INFO,
                next = IntPtr.Zero,
                createFlags = 0,
                enabledApiLayerCount = 0,
                enabledApiLayerNames = IntPtr.Zero,
                enabledExtensionCount = 1,
                enabledExtensionNames = _extensionNamesBlock
            };
            createInfo.applicationInfo.applicationVersion = 1;
            createInfo.applicationInfo.engineVersion = 1;
            // XR_MAKE_VERSION(1,0,0)
            createInfo.applicationInfo.apiVersion = (1UL << 48) | (0UL << 32) | 0UL;
            OpenXrNative.WriteAscii(createInfo.applicationInfo.applicationName, 128, "Synapse OMNIA");
            OpenXrNative.WriteAscii(createInfo.applicationInfo.engineName, 128, "Synapse");

            int rc = OpenXrNative.xrCreateInstance(in createInfo, out _instance);
            if (rc != OpenXrNative.XR_SUCCESS || _instance == OpenXrNative.XR_NULL_HANDLE)
            {
                _logger?.Debug("VR", $"xrCreateInstance failed: {rc}");
                CleanupNativePartial();
                return false;
            }

            var sysInfo = new OpenXrNative.XrSystemGetInfo
            {
                type = OpenXrNative.XR_TYPE_SYSTEM_GET_INFO,
                next = IntPtr.Zero,
                formFactor = OpenXrNative.XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY
            };
            rc = OpenXrNative.xrGetSystem(_instance, in sysInfo, out ulong systemId);
            if (rc == OpenXrNative.XR_ERROR_FORM_FACTOR_UNAVAILABLE ||
                rc != OpenXrNative.XR_SUCCESS ||
                systemId == 0)
            {
                _logger?.Debug("VR", $"xrGetSystem HMD unavailable: {rc}");
                CleanupNativePartial();
                return false;
            }

            // Query Vulkan requirements when the extension proc is available.
            if (OpenXrNative.TryGetProc<OpenXrNative.XrGetVulkanGraphicsRequirements2KHR>(
                    _instance,
                    "xrGetVulkanGraphicsRequirements2KHR",
                    out var getReqs) && getReqs != null)
            {
                var reqs = new OpenXrNative.XrGraphicsRequirementsVulkan2KHR
                {
                    type = OpenXrNative.XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR,
                    next = IntPtr.Zero
                };
                _ = getReqs(_instance, systemId, ref reqs);
            }

            if (vulkanBinding is not { IsValid: true } bind)
            {
                // Native runtime + HMD exist, but caller did not supply Vulkan handles yet.
                CleanupNativePartial();
                return false;
            }

            var gfx = new OpenXrNative.XrGraphicsBindingVulkan2KHR
            {
                type = OpenXrNative.XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR,
                next = IntPtr.Zero,
                instance = bind.Instance,
                physicalDevice = bind.PhysicalDevice,
                device = bind.Device,
                queueFamilyIndex = bind.QueueFamilyIndex,
                queueIndex = bind.QueueIndex
            };

            IntPtr gfxPtr = Marshal.AllocHGlobal(Marshal.SizeOf<OpenXrNative.XrGraphicsBindingVulkan2KHR>());
            try
            {
                Marshal.StructureToPtr(gfx, gfxPtr, false);
                var sessionInfo = new OpenXrNative.XrSessionCreateInfo
                {
                    type = OpenXrNative.XR_TYPE_SESSION_CREATE_INFO,
                    next = gfxPtr,
                    createFlags = 0,
                    systemId = systemId
                };
                rc = OpenXrNative.xrCreateSession(_instance, in sessionInfo, out _session);
            }
            finally
            {
                Marshal.FreeHGlobal(gfxPtr);
            }

            if (rc != OpenXrNative.XR_SUCCESS || _session == OpenXrNative.XR_NULL_HANDLE)
            {
                _logger?.Debug("VR", $"xrCreateSession failed: {rc}");
                CleanupNativePartial();
                return false;
            }

            var swap = OpenXrVulkanSwapchain.TryCreateNative(_session, width, height);
            if (swap == null)
            {
                CleanupNativePartial();
                return false;
            }

            IsAvailable = true;
            IsRunning = true;
            IsSimulated = false;
            UsesNativeOpenXr = true;
            RuntimeName = Environment.GetEnvironmentVariable("XR_RUNTIME") ?? "OpenXR-Vulkan2";
            Swapchain = swap;
            _initialized = true;
            _logger?.Info("VR", $"OpenXR native Vulkan2 swapchain {width}x{height} ({RuntimeName}, images={swap.ImageCount})");
            return true;
        }
        catch (DllNotFoundException ex)
        {
            _logger?.Debug("VR", $"OpenXR P/Invoke missing: {ex.Message}");
            CleanupNativePartial();
            return false;
        }
        catch (Exception ex)
        {
            _logger?.Warn("VR", $"OpenXR native init error: {ex.Message}");
            CleanupNativePartial();
            return false;
        }
    }

    private void CleanupNativePartial()
    {
        if (_session != OpenXrNative.XR_NULL_HANDLE)
        {
            _ = OpenXrNative.xrDestroySession(_session);
            _session = OpenXrNative.XR_NULL_HANDLE;
        }

        if (_instance != OpenXrNative.XR_NULL_HANDLE)
        {
            _ = OpenXrNative.xrDestroyInstance(_instance);
            _instance = OpenXrNative.XR_NULL_HANDLE;
        }

        if (_extensionNamesBlock != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(_extensionNamesBlock);
            _extensionNamesBlock = IntPtr.Zero;
        }

        if (_vulkanExtName != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(_vulkanExtName);
            _vulkanExtName = IntPtr.Zero;
        }
    }

    public Task BeginFrameAsync(CancellationToken cancellationToken = default)
    {
        if (!_initialized || Swapchain == null)
            return Task.CompletedTask;

        if (!Swapchain.TryAcquire(out var index))
            _logger?.Warn("VR", "Swapchain acquire skipped (already acquired or native fail)");
        else
            _logger?.Debug("VR", $"Acquire swapchain image {index} (sim={Swapchain.IsSimulated})");

        return Task.CompletedTask;
    }

    public Task EndFrameAsync(CancellationToken cancellationToken = default)
    {
        if (!_initialized || Swapchain == null)
            return Task.CompletedTask;

        var frame = Swapchain.PrepareSubmit(vulkanQueueFamily: 0, vulkanQueue: 0);
        _frameCounter++;
        _logger?.Debug("VR", $"Submit XR frame #{_frameCounter} image={frame.ImageHandle:x} sim={frame.IsSimulated}");
        Swapchain.Release();
        return Task.CompletedTask;
    }

    public Task PollEventsAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public ValueTask DisposeAsync()
    {
        IsRunning = false;
        _initialized = false;
        Swapchain?.Dispose();
        Swapchain = null;
        CleanupNativePartial();
        return ValueTask.CompletedTask;
    }

    /// <summary>
    /// True when the OpenXR loader native library resolves.
    /// <c>XR_RUNTIME_JSON</c> alone is not enough — it only hints the active runtime.
    /// </summary>
    internal static bool TryLoadOpenXrLoader()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return NativeLibrary.TryLoad("openxr_loader", out _) ||
                   NativeLibrary.TryLoad("openxr_loader.dll", out _);

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return NativeLibrary.TryLoad("openxr_loader", out _) ||
                   NativeLibrary.TryLoad("libopenxr_loader.so", out _) ||
                   NativeLibrary.TryLoad("libopenxr_loader.so.1", out _);

        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return NativeLibrary.TryLoad("openxr_loader", out _) ||
                   NativeLibrary.TryLoad("libopenxr_loader.dylib", out _);

        return false;
    }
}

public sealed class HeadlessVrSession : IVrSession
{
    public bool IsAvailable { get; private set; }
    public bool IsRunning { get; private set; }
    public bool IsSimulated => true;
    public bool UsesNativeOpenXr => false;
    public string RuntimeName { get; private set; } = "headless";
    public OpenXrVulkanSwapchain? Swapchain { get; private set; }

    public Task<bool> TryInitializeAsync(
        int width = 1280,
        int height = 720,
        OpenXrVulkanBinding? vulkanBinding = null,
        CancellationToken cancellationToken = default)
    {
        IsAvailable = false;
        IsRunning = false;
        Swapchain = null;
        return Task.FromResult(false);
    }

    public Task BeginFrameAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task EndFrameAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task PollEventsAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public ValueTask DisposeAsync()
    {
        IsRunning = false;
        return ValueTask.CompletedTask;
    }
}

public static class VrSessionFactory
{
    public static IVrSession Create(ISynapseLogger? logger = null) => new OpenXrVulkanSession(logger);

    public static IVrSession CreateHeadless() => new HeadlessVrSession();
}
