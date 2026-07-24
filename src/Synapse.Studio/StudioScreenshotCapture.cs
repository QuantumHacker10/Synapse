using System.IO;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Headless;
using Avalonia.Threading;
using Synapse.Infrastructure.Configuration;
using Synapse.Runtime;
using Synapse.Studio.ViewModels;
using Synapse.Studio.Views;

namespace Synapse.Studio;

/// <summary>Captures a live Avalonia render of Synapse Studio via headless platform.</summary>
public static class StudioScreenshotCapture
{
    public static int Capture(string outputPath, string[] args)
    {
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        StudioRuntime.IsScreenshotMode = true;
        SimulationReproducibility.ResetFromEnvironment();

        EngineHost? host = null;
        IDisposable? logger = null;
        try
        {
            var lifetime = new ClassicDesktopStyleApplicationLifetime
            {
                Args = args,
                ShutdownMode = ShutdownMode.OnMainWindowClose
            };

            AppBuilder.Configure<App>()
                .UseSkia()
                .UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false })
                .WithInterFont()
                .AfterSetup(builder =>
                {
                    if (builder.Instance is App app)
                        app.Initialize();
                })
                .SetupWithLifetime(lifetime);

            var (bootHost, orchestrator, bootLogger, config, plugins) = App.BootstrapFull(args);
            host = bootHost;
            logger = bootLogger as IDisposable;
            App.Plugins = plugins;

            var width = Math.Max(640, config.Width);
            var height = Math.Max(480, config.Height);

            var window = new MainWindow
            {
                Width = width,
                Height = height,
                DataContext = new MainWindowViewModel(host, orchestrator, bootLogger, config)
            };
            lifetime.MainWindow = window;
            window.Show();
            Dispatcher.UIThread.RunJobs();

            window.Measure(new Size(width, height));
            window.Arrange(new Rect(0, 0, width, height));
            Dispatcher.UIThread.RunJobs();

            var full = Path.GetFullPath(outputPath);
            Directory.CreateDirectory(Path.GetDirectoryName(full)!);

            var frame = window.CaptureRenderedFrame();
            if (frame is null)
                throw new InvalidOperationException("Headless renderer produced no frame.");

            frame.Save(full);
            Console.WriteLine($"[Screenshot] Saved live Studio render -> {full}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[Screenshot] Failed: {ex.Message}");
            return 1;
        }
        finally
        {
            try
            {
                App.Plugins?.Dispose();
            }
            catch
            {
                // ignore
            }

            if (host != null)
            {
                try
                {
                    host.DisposeAsync().AsTask().GetAwaiter().GetResult();
                }
                catch
                {
                    // ignore
                }
            }

            try
            {
                logger?.Dispose();
            }
            catch
            {
                // ignore
            }

            StudioRuntime.IsScreenshotMode = false;
        }
    }
}
