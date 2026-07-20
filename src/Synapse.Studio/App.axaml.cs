using System;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;
using Synapse.Studio.ViewModels;
using Synapse.Studio.Views;

namespace Synapse.Studio
{
    public partial class App : Application
    {
        public static EngineHost? Host { get; private set; }
        public static FrameOrchestrator? Orchestrator { get; private set; }
        public static ISynapseLogger Logger { get; private set; } = SynapseLogger.Default;
        public static SynapseConfig Config { get; private set; } = new();

        public override void Initialize() => AvaloniaXamlLoader.Load(this);

        public override void OnFrameworkInitializationCompleted()
        {
            if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            {
                var (host, orchestrator, logger, config) = Bootstrap(Environment.GetCommandLineArgs());
                Host = host;
                Orchestrator = orchestrator;
                Logger = logger;
                Config = config;

                desktop.MainWindow = new MainWindow
                {
                    DataContext = new MainWindowViewModel(host, orchestrator, logger, config)
                };

                desktop.Exit += async (_, _) =>
                {
                    if (Host != null)
                        await Host.DisposeAsync();
                    (Logger as System.IDisposable)?.Dispose();
                };
            }

            base.OnFrameworkInitializationCompleted();
        }

        public static (EngineHost Host, FrameOrchestrator Orchestrator, ISynapseLogger Logger, SynapseConfig Config)
            Bootstrap(string[] args)
        {
            var config = SynapseConfig.Load(args: args);
            var logger = new SynapseLogger(
                System.IO.Path.Combine(
                    System.Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData),
                    "Synapse", "logs"),
                System.Enum.TryParse<LogLevel>(config.LogLevel, true, out var lvl) ? lvl : LogLevel.Information,
                consoleEnabled: !StudioRuntime.IsScreenshotMode);

            var host = new EngineHost(config, logger);
            host.InitializeModules();
            if (!string.IsNullOrWhiteSpace(config.ScenePath))
                host.LoadSceneAsync(config.ScenePath).GetAwaiter().GetResult();

            var orchestrator = new FrameOrchestrator(host, logger);
            return (host, orchestrator, logger, config);
        }
    }
}
