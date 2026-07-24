using System;
using System.Runtime.CompilerServices;
using System.Text;
using Silk.NET.Core;
using Silk.NET.Core.Native;
using Silk.NET.OpenXR;
using Synapse.Infrastructure.Logging;

namespace Synapse.VR;

/// <summary>
/// Native OpenXR runtime (Silk.NET). Creates a real OpenXR instance/system/session when a
/// loader + runtime are available. Prefers <c>XR_MND_headless</c> when present (CI / no HMD).
/// </summary>
public sealed unsafe class NativeOpenXrRuntime : IDisposable
{
    public const string VulkanEnable2Extension = "XR_KHR_vulkan_enable2";
    public const string VulkanEnableExtension = "XR_KHR_vulkan_enable";
    public const string HeadlessExtension = "XR_MND_headless";

    private XR? _xr;
    private Instance _instance;
    private Session _session;
    private Swapchain _swapchain;
    private ulong _systemId;
    private bool _sessionRunning;
    private bool _disposed;
    private readonly ISynapseLogger? _logger;

    public NativeOpenXrRuntime(ISynapseLogger? logger = null) => _logger = logger;

    public bool IsInitialized => _instance.Handle != 0;
    public bool IsSessionRunning => _sessionRunning;
    public string RuntimeName { get; private set; } = "none";
    public string SystemName { get; private set; } = "none";
    public bool UsesHeadless { get; private set; }
    public int RecommendedWidth { get; private set; }
    public int RecommendedHeight { get; private set; }
    public ulong[] SwapchainImageHandles { get; private set; } = Array.Empty<ulong>();
    public int CurrentSwapchainIndex { get; private set; } = -1;
    public long PredictedDisplayTime { get; private set; }

    /// <summary>
    /// Attempts to create a native OpenXR instance, system, session and (when possible) swapchain.
    /// </summary>
    public bool TryInitialize(int preferredWidth, int preferredHeight)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        try
        {
            _xr = XR.GetApi();
            if (_xr is null)
            {
                _logger?.Warn("VR", "Silk.NET OpenXR API failed to load.");
                return false;
            }

            if (!TryCreateInstance(out var enabledHeadless))
                return false;

            UsesHeadless = enabledHeadless;
            if (!TryGetSystem())
                return false;

            if (!TryCreateSession())
                return false;

            if (!TryCreateSwapchain(preferredWidth, preferredHeight))
            {
                _logger?.Warn("VR", "Native OpenXR session created but swapchain enumeration failed; continuing without images.");
            }

            _logger?.Info("VR",
                $"Native OpenXR ready — runtime={RuntimeName}, system={SystemName}, headless={UsesHeadless}, " +
                $"views={RecommendedWidth}x{RecommendedHeight}, images={SwapchainImageHandles.Length}");
            return true;
        }
        catch (DllNotFoundException ex)
        {
            _logger?.Warn("VR", $"OpenXR loader not found: {ex.Message}");
            TearDown();
            return false;
        }
        catch (Exception ex)
        {
            _logger?.Warn("VR", $"Native OpenXR init failed: {ex.Message}");
            TearDown();
            return false;
        }
    }

    public bool TryBeginFrame()
    {
        if (_xr is null || _session.Handle == 0)
            return false;

        PollEvents();

        if (!_sessionRunning)
            return false;

        var frameWait = new FrameWaitInfo { Type = StructureType.FrameWaitInfo };
        var frameState = new FrameState { Type = StructureType.FrameState };
        var waitResult = _xr.WaitFrame(_session, in frameWait, ref frameState);
        if (waitResult != Result.Success)
            return false;

        PredictedDisplayTime = frameState.PredictedDisplayTime;

        var begin = new FrameBeginInfo { Type = StructureType.FrameBeginInfo };
        if (_xr.BeginFrame(_session, in begin) != Result.Success)
            return false;

        if (_swapchain.Handle != 0)
        {
            var acquire = new SwapchainImageAcquireInfo { Type = StructureType.SwapchainImageAcquireInfo };
            uint index = 0;
            if (_xr.AcquireSwapchainImage(_swapchain, in acquire, ref index) == Result.Success)
            {
                var waitInfo = new SwapchainImageWaitInfo
                {
                    Type = StructureType.SwapchainImageWaitInfo,
                    Timeout = 100_000_000 // 100ms in nanoseconds
                };
                _xr.WaitSwapchainImage(_swapchain, in waitInfo);
                CurrentSwapchainIndex = (int)index;
                return true;
            }
        }

        CurrentSwapchainIndex = -1;
        return true;
    }

    public void EndFrame()
    {
        if (_xr is null || _session.Handle == 0)
            return;

        if (_swapchain.Handle != 0 && CurrentSwapchainIndex >= 0)
        {
            var release = new SwapchainImageReleaseInfo { Type = StructureType.SwapchainImageReleaseInfo };
            _xr.ReleaseSwapchainImage(_swapchain, in release);
            CurrentSwapchainIndex = -1;
        }

        var end = new FrameEndInfo
        {
            Type = StructureType.FrameEndInfo,
            DisplayTime = PredictedDisplayTime > 0 ? PredictedDisplayTime : 1,
            EnvironmentBlendMode = EnvironmentBlendMode.Opaque,
            LayerCount = 0,
            Layers = null
        };
        _xr.EndFrame(_session, in end);
    }

    public void PollEvents()
    {
        if (_xr is null || _instance.Handle == 0)
            return;

        while (true)
        {
            var eventData = new EventDataBuffer { Type = StructureType.EventDataBuffer };
            var result = _xr.PollEvent(_instance, ref eventData);
            if (result == Result.EventUnavailable)
                break;
            if (result != Result.Success)
                break;

            if (eventData.Type == StructureType.EventDataSessionStateChanged)
            {
                ref var stateEvent = ref Unsafe.As<EventDataBuffer, EventDataSessionStateChanged>(ref eventData);
                ApplySessionState(stateEvent.State);
            }
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        TearDown();
    }

    private void TearDown()
    {
        if (_xr == null)
            return;

        if (_swapchain.Handle != 0)
        {
            _xr.DestroySwapchain(_swapchain);
            _swapchain = default;
        }

        if (_session.Handle != 0)
        {
            if (_sessionRunning)
            {
                _xr.EndSession(_session);
                _sessionRunning = false;
            }
            _xr.DestroySession(_session);
            _session = default;
        }

        if (_instance.Handle != 0)
        {
            _xr.DestroyInstance(_instance);
            _instance = default;
        }

        _xr.Dispose();
        _xr = null;
    }

    private bool TryCreateInstance(out bool headlessEnabled)
    {
        headlessEnabled = false;
        var xr = _xr!;

        uint extCount = 0;
        xr.EnumerateInstanceExtensionProperties((string?)null, 0, ref extCount, null);
        var extensions = new ExtensionProperties[Math.Max(1, (int)extCount)];
        for (int i = 0; i < extensions.Length; i++)
            extensions[i].Type = StructureType.ExtensionProperties;

        if (extCount > 0)
        {
            fixed (ExtensionProperties* extPtr = extensions)
                xr.EnumerateInstanceExtensionProperties((string?)null, extCount, ref extCount, extPtr);
        }

        bool hasHeadless = false;
        bool hasVulkan2 = false;
        bool hasVulkan1 = false;
        for (int i = 0; i < (int)extCount; i++)
        {
            string name;
            fixed (ExtensionProperties* p = &extensions[i])
                name = ReadFixedString(p->ExtensionName, 128);
            if (name == HeadlessExtension)
                hasHeadless = true;
            else if (name == VulkanEnable2Extension)
                hasVulkan2 = true;
            else if (name == VulkanEnableExtension)
                hasVulkan1 = true;
        }

        // Prefer headless for environments without a compositor/HMD (CI, servers).
        string[] enabled;
        if (hasHeadless)
        {
            enabled = [HeadlessExtension];
            headlessEnabled = true;
        }
        else if (hasVulkan2)
        {
            enabled = [VulkanEnable2Extension];
        }
        else if (hasVulkan1)
        {
            enabled = [VulkanEnableExtension];
        }
        else
        {
            enabled = [];
        }

        var appInfo = new ApplicationInfo
        {
            ApplicationVersion = 1,
            EngineVersion = 1,
            ApiVersion = new Version64(1, 0, 34)
        };
        {
            ApplicationInfo* pApp = &appInfo;
            WriteFixedString(pApp->ApplicationName, "Synapse OMNIA", 128);
            WriteFixedString(pApp->EngineName, "Synapse", 128);
        }

        nint extNames = enabled.Length > 0 ? SilkMarshal.StringArrayToPtr(enabled) : 0;
        try
        {
            var createInfo = new InstanceCreateInfo
            {
                Type = StructureType.InstanceCreateInfo,
                ApplicationInfo = appInfo,
                EnabledExtensionCount = (uint)enabled.Length,
                EnabledExtensionNames = enabled.Length > 0 ? (byte**)extNames : null
            };

            Instance instance = default;
            var result = xr.CreateInstance(in createInfo, ref instance);
            if (result != Result.Success)
            {
                _logger?.Warn("VR", $"xrCreateInstance failed: {result}");
                return false;
            }

            _instance = instance;

            var props = new InstanceProperties { Type = StructureType.InstanceProperties };
            xr.GetInstanceProperties(_instance, ref props);
            {
                InstanceProperties* pProps = &props;
                RuntimeName = ReadFixedString(pProps->RuntimeName, 128);
            }
            return true;
        }
        finally
        {
            if (extNames != 0)
                SilkMarshal.Free(extNames);
        }
    }

    private bool TryGetSystem()
    {
        var getInfo = new SystemGetInfo
        {
            Type = StructureType.SystemGetInfo,
            FormFactor = FormFactor.HeadMountedDisplay
        };
        ulong systemId = 0;
        var result = _xr!.GetSystem(_instance, in getInfo, ref systemId);
        if (result != Result.Success || systemId == 0)
        {
            _logger?.Warn("VR", $"xrGetSystem failed: {result}");
            return false;
        }

        _systemId = systemId;
        var props = new SystemProperties { Type = StructureType.SystemProperties };
        _xr.GetSystemProperties(_instance, _systemId, ref props);
        {
            SystemProperties* pProps = &props;
            SystemName = ReadFixedString(pProps->SystemName, 128);
        }

        var viewConfigType = ViewConfigurationType.PrimaryStereo;
        uint viewCount = 0;
        _xr.EnumerateViewConfigurationView(_instance, _systemId, viewConfigType, 0, ref viewCount, null);
        if (viewCount > 0)
        {
            var views = new ViewConfigurationView[viewCount];
            for (int i = 0; i < views.Length; i++)
                views[i].Type = StructureType.ViewConfigurationView;
            fixed (ViewConfigurationView* viewPtr = views)
                _xr.EnumerateViewConfigurationView(_instance, _systemId, viewConfigType, viewCount, ref viewCount, viewPtr);
            RecommendedWidth = (int)views[0].RecommendedImageRectWidth;
            RecommendedHeight = (int)views[0].RecommendedImageRectHeight;
        }

        return true;
    }

    private bool TryCreateSession()
    {
        // Headless sessions do not require a graphics binding.
        var createInfo = new SessionCreateInfo
        {
            Type = StructureType.SessionCreateInfo,
            SystemId = _systemId,
            Next = null
        };

        Session session = default;
        var result = _xr!.CreateSession(_instance, in createInfo, ref session);
        if (result != Result.Success)
        {
            _logger?.Warn("VR",
                $"xrCreateSession failed: {result}. Install an OpenXR runtime (SteamVR/Monado) " +
                $"or enable {HeadlessExtension} for graphics-less sessions.");
            return false;
        }

        _session = session;
        PollEvents();
        if (!_sessionRunning)
        {
            var begin = new SessionBeginInfo
            {
                Type = StructureType.SessionBeginInfo,
                PrimaryViewConfigurationType = ViewConfigurationType.PrimaryStereo
            };
            var beginResult = _xr.BeginSession(_session, in begin);
            if (beginResult == Result.Success)
                _sessionRunning = true;
            else
                _logger?.Debug("VR", $"xrBeginSession deferred ({beginResult}); waiting for READY event.");
        }

        return true;
    }

    private bool TryCreateSwapchain(int preferredWidth, int preferredHeight)
    {
        if (_session.Handle == 0)
            return false;

        uint formatCount = 0;
        _xr!.EnumerateSwapchainFormats(_session, 0, ref formatCount, null);
        if (formatCount == 0)
            return false;

        var formats = new long[formatCount];
        fixed (long* formatPtr = formats)
            _xr.EnumerateSwapchainFormats(_session, formatCount, ref formatCount, formatPtr);

        // Prefer VK_FORMAT_R8G8B8A8_SRGB (43) or B8G8R8A8_SRGB (44).
        long chosen = formats[0];
        foreach (var f in formats)
        {
            if (f is 43 or 44 or 37 or 50)
            {
                chosen = f;
                break;
            }
        }

        int width = RecommendedWidth > 0 ? RecommendedWidth : preferredWidth;
        int height = RecommendedHeight > 0 ? RecommendedHeight : preferredHeight;

        var createInfo = new SwapchainCreateInfo
        {
            Type = StructureType.SwapchainCreateInfo,
            UsageFlags = SwapchainUsageFlags.ColorAttachmentBit | SwapchainUsageFlags.SampledBit,
            Format = chosen,
            SampleCount = 1,
            Width = (uint)Math.Max(64, width),
            Height = (uint)Math.Max(64, height),
            FaceCount = 1,
            ArraySize = 1,
            MipCount = 1
        };

        Swapchain swapchain = default;
        var result = _xr.CreateSwapchain(_session, in createInfo, ref swapchain);
        if (result != Result.Success)
        {
            _logger?.Warn("VR", $"xrCreateSwapchain failed: {result}");
            return false;
        }

        _swapchain = swapchain;
        uint imageCount = 0;
        _xr.EnumerateSwapchainImages(_swapchain, 0, ref imageCount, null);
        if (imageCount == 0)
            return false;

        // For headless (no Vulkan binding), images may be opaque placeholders from the runtime.
        // When Vulkan enable is active, runtimes return SwapchainImageVulkanKHR with a real VkImage.
        var images = new SwapchainImageVulkanKHR[imageCount];
        for (int i = 0; i < images.Length; i++)
            images[i].Type = StructureType.SwapchainImageVulkanKhr;

        fixed (SwapchainImageVulkanKHR* imgPtr = images)
        {
            var enumResult = _xr.EnumerateSwapchainImages(
                _swapchain,
                imageCount,
                ref imageCount,
                (SwapchainImageBaseHeader*)imgPtr);
            if (enumResult != Result.Success)
                return false;
        }

        SwapchainImageHandles = new ulong[imageCount];
        for (uint i = 0; i < imageCount; i++)
        {
            ulong handle = images[i].Image;
            // Headless / non-Vulkan runtimes may leave Image=0 — keep a stable native-backed token.
            SwapchainImageHandles[i] = handle != 0 ? handle : (0xA000_0000UL + i);
        }

        RecommendedWidth = width;
        RecommendedHeight = height;
        return true;
    }

    private void ApplySessionState(SessionState state)
    {
        switch (state)
        {
            case SessionState.Ready:
                if (_xr != null && _session.Handle != 0 && !_sessionRunning)
                {
                    var begin = new SessionBeginInfo
                    {
                        Type = StructureType.SessionBeginInfo,
                        PrimaryViewConfigurationType = ViewConfigurationType.PrimaryStereo
                    };
                    if (_xr.BeginSession(_session, in begin) == Result.Success)
                        _sessionRunning = true;
                }
                break;
            case SessionState.Stopping:
                if (_xr != null && _sessionRunning)
                {
                    _xr.EndSession(_session);
                    _sessionRunning = false;
                }
                break;
            case SessionState.Exiting:
            case SessionState.LossPending:
                _sessionRunning = false;
                break;
        }
    }

    private static string ReadFixedString(byte* ptr, int max)
    {
        if (ptr == null)
            return string.Empty;
        int len = 0;
        while (len < max && ptr[len] != 0)
            len++;
        return Encoding.UTF8.GetString(ptr, len);
    }

    private static void WriteFixedString(byte* ptr, string value, int max)
    {
        if (ptr == null)
            return;
        var bytes = Encoding.UTF8.GetBytes(value);
        int copy = Math.Min(bytes.Length, max - 1);
        for (int i = 0; i < copy; i++)
            ptr[i] = bytes[i];
        for (int i = copy; i < max; i++)
            ptr[i] = 0;
    }
}
