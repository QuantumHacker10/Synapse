using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Platform;
using Avalonia.Threading;
using GDNN.Platform;

namespace GDNN.Studio.Views
{
    /// <summary>
    /// Hosts a Win32 child HWND on Windows, otherwise falls back to a GLFW sibling window
    /// (Linux/macOS and Windows fallback).
    /// </summary>
    public sealed class VulkanViewportControl : NativeControlHost
    {
        private IntPtr _childHwnd;
        private bool _engineStarted;
        private DispatcherTimer? _timer;
        private int _width = 800;
        private int _height = 600;

        protected override IPlatformHandle CreateNativeControlCore(IPlatformHandle parent)
        {
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

        private void StartEngine(IntPtr hwnd, bool embedded)
        {
            if (_engineStarted || App.Host == null || App.Orchestrator == null) return;

            try
            {
                if (embedded && hwnd != IntPtr.Zero && OperatingSystem.IsWindows())
                    App.Host.InitializeRenderFromHwnd(hwnd, _width, _height, App.Config.EnableValidation);
                else
                    App.Host.InitializeRender(App.Config.Width, App.Config.Height, App.Config.EnableValidation);

                _engineStarted = true;
                _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(16) };
                _timer.Tick += async (_, _) =>
                {
                    if (App.Orchestrator == null) return;
                    try { await App.Orchestrator.TickAsync(); }
                    catch (Exception ex) { App.Logger.Warn("Viewport", ex.Message); }
                };
                _timer.Start();
            }
            catch (Exception ex)
            {
                App.Logger.Error("Viewport", "Failed to start render engine", ex);
            }
        }
    }
}
