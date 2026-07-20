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

/// <summary>Captures a live Avalonia render of Synapse Studio via headless platform (v2.2).</summary>
public static class StudioScreenshotCapture
{
    public static int Capture(string outputPath, string[] args)
    {
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        StudioRuntime.IsScreenshotMode = true;
        SimulationReproducibility.ResetFromEnvironment();

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

            var config = SynapseConfig.Load(args: args);
            var (host, orchestrator, logger, loadedConfig) = App.Bootstrap(args);
            config = loadedConfig;

            var width = Math.Max(640, config.Width);
            var height = Math.Max(480, config.Height);

            var window = new MainWindow
            {
                Width = width,
                Height = height,
                DataContext = new MainWindowViewModel(host, orchestrator, logger, config)
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

            host.DisposeAsync().AsTask().GetAwaiter().GetResult();
            (logger as IDisposable)?.Dispose();

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
            StudioRuntime.IsScreenshotMode = false;
        }
    }
}
