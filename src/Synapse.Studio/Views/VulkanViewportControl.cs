using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform;
using Avalonia.Threading;
using GDNN.Platform;

namespace Synapse.Studio.Views
{
    /// <summary>
    /// Hosts a Win32 child HWND on Windows, otherwise falls back to a GLFW sibling window
    /// (Linux/macOS and Windows fallback).
    /// The native frame pipeline timer always starts after module init so Physics /
    /// Simulation advance even when Vulkan fails to load (headless / missing vulkan-1.dll).
    /// </summary>
    public sealed class VulkanViewportControl : NativeControlHost
    {
        private IntPtr _childHwnd;
        private bool _startAttempted;
        private bool _timerStarted;
        private DispatcherTimer? _timer;
        private int _width = 800;
        private int _height = 600;

        public VulkanViewportControl()
        {
            PointerPressed += OnPointerPressed;
            PointerMoved += OnPointerMoved;
            PointerReleased += OnPointerReleased;
        }

        protected override IPlatformHandle CreateNativeControlCore(IPlatformHandle parent)
        {
            if (StudioRuntime.IsScreenshotMode)
                return new PlatformHandle(IntPtr.Zero, "HEADLESS");

            if (OperatingSystem.IsWindows())
            {
                try
                {
                    _childHwnd = Win32VulkanSurface.CreateChildWindow(parent.Handle, _width, _height);
                    StartEngine(_childHwnd, embedded: true);
                    return new PlatformHandle(_childHwnd, "HWND");
                }
                catch (Exception ex)
                {
                    App.Logger.Warn("Viewport", $"HWND path failed: {ex.Message}. Using GLFW.");
                }
            }

            StartEngine(IntPtr.Zero, embedded: false);
            return base.CreateNativeControlCore(parent);
        }

        protected override void DestroyNativeControlCore(IPlatformHandle control)
        {
            _timer?.Stop();
            if (_childHwnd != IntPtr.Zero && OperatingSystem.IsWindows())
            {
                Win32VulkanSurface.DestroyWindow(_childHwnd);
                _childHwnd = IntPtr.Zero;
            }
            base.DestroyNativeControlCore(control);
        }

        protected override void OnSizeChanged(SizeChangedEventArgs e)
        {
            base.OnSizeChanged(e);
            _width = Math.Max(1, (int)e.NewSize.Width);
            _height = Math.Max(1, (int)e.NewSize.Height);
            if (_childHwnd != IntPtr.Zero && OperatingSystem.IsWindows())
                Win32VulkanSurface.ResizeChild(_childHwnd, _width, _height);
            App.Host?.RenderEngine?.NotifyExternalResize(_width, _height);
        }

        private void OnPointerPressed(object? sender, PointerPressedEventArgs e)
        {
            if (App.Host == null)
                return;
            var pt = e.GetPosition(this);
            var props = e.GetCurrentPoint(this).Properties;
            App.Host.HandleViewportPointerDown((float)pt.X, (float)pt.Y, _width, _height, props.IsRightButtonPressed);
            e.Handled = true;
        }

        private void OnPointerMoved(object? sender, PointerEventArgs e)
        {
            if (App.Host == null)
                return;
            if (!e.GetCurrentPoint(this).Properties.IsLeftButtonPressed &&
                !e.GetCurrentPoint(this).Properties.IsRightButtonPressed)
                return;
            var pt = e.GetPosition(this);
            App.Host.HandleViewportPointerMove((float)pt.X, (float)pt.Y, _width, _height);
        }

        private void OnPointerReleased(object? sender, PointerReleasedEventArgs e)
        {
            App.Host?.HandleViewportPointerUp();
        }

        private void StartEngine(IntPtr hwnd, bool embedded)
        {
            if (_startAttempted || App.Host == null || App.Orchestrator == null)
                return;
            _startAttempted = true;

            try
            {
                if (embedded && hwnd != IntPtr.Zero && OperatingSystem.IsWindows())
                    App.Host.InitializeRenderFromHwnd(hwnd, _width, _height, App.Config.EnableValidation);
                else
                    App.Host.InitializeRender(App.Config.Width, App.Config.Height, App.Config.EnableValidation);
            }
            catch (Exception ex)
            {
                App.Logger.Error("Viewport", "Failed to start render engine", ex);
                // Modules remain initialized — native pipeline still ticks Physics/Simulation.
            }

            EnsureFrameTimer();
        }

        /// <summary>
        /// Starts the native FrameOrchestrator timer even when Vulkan init failed,
        /// so Physics / Simulation / Quality keep advancing in Studio.
        /// </summary>
        private void EnsureFrameTimer()
        {
            if (_timerStarted || App.Orchestrator == null)
                return;

            _timerStarted = true;
            _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(16) };
            _timer.Tick += async (_, _) =>
            {
                if (App.Orchestrator == null)
                    return;
                try
                { await App.Orchestrator.TickAsync(); }
                catch (Exception ex) { App.Logger.Warn("Viewport", ex.Message); }
            };
            _timer.Start();
            App.Logger.Info("Viewport",
                App.Host?.IsRenderInitialized == true
                    ? "Native frame pipeline started (Physics + Simulation + FrameGraph)"
                    : "Native frame pipeline started headless (Physics + Simulation; render deferred)");
        }
    }
}
