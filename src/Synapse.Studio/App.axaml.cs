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
                Config = SynapseConfig.Load(args: Environment.GetCommandLineArgs());
                Logger = new SynapseLogger(
                    System.IO.Path.Combine(
                        System.Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData),
                        "Synapse", "logs"),
                    System.Enum.TryParse<LogLevel>(Config.LogLevel, true, out var lvl) ? lvl : LogLevel.Information);

                Host = new EngineHost(Config, Logger);
                Host.InitializeModules();
                Orchestrator = new FrameOrchestrator(Host, Logger);

                desktop.MainWindow = new MainWindow
                {
                    DataContext = new MainWindowViewModel(Host, Orchestrator, Logger, Config)
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
    }
}
