using System;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Plugins;
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
        public static PluginHost? PluginHost { get; private set; }
        public static PluginHost? Plugins { get; set; }

        public override void Initialize() => AvaloniaXamlLoader.Load(this);

        public override void OnFrameworkInitializationCompleted()
        {
            if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            {
                var (host, orchestrator, logger, config, plugins) = BootstrapFull(Environment.GetCommandLineArgs());
                Host = host;
                Orchestrator = orchestrator;
                Logger = logger;
                Config = config;
                Plugins = plugins;

                var pluginHost = new PluginHost(logger);
                if (!string.IsNullOrWhiteSpace(config.PluginDirectory))
                    pluginHost.LoadFromDirectory(config.PluginDirectory, host);
                PluginHost = pluginHost;

                desktop.MainWindow = new MainWindow
                {
                    DataContext = new MainWindowViewModel(host, orchestrator, logger, config, pluginHost)
                };

                desktop.Exit += async (_, _) =>
                {
                    PluginHost?.Dispose();
                    try
                    {
                        Plugins?.Dispose();
                    }
                    catch (Exception ex)
                    {
                        Logger.Warn("Studio", $"Plugin dispose: {ex.Message}");
                    }

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
            var full = BootstrapFull(args);
            return (full.Host, full.Orchestrator, full.Logger, full.Config);
        }

        public static (EngineHost Host, FrameOrchestrator Orchestrator, ISynapseLogger Logger, SynapseConfig Config, PluginHost Plugins)
            BootstrapFull(string[] args)
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
            host.ApplyOptionalCollaborationFromConfigAsync().GetAwaiter().GetResult();

            var plugins = new PluginHost(logger);
            if (!string.IsNullOrWhiteSpace(config.PluginDirectory))
            {
                plugins.LoadFromDirectory(config.PluginDirectory, host);
                PluginMarketplace.FromDirectory(config.PluginDirectory, logger).VerifyInstalledOrWarn();
            }

            var orchestrator = new FrameOrchestrator(host, logger);
            return (host, orchestrator, logger, config, plugins);
        }
    }
}
