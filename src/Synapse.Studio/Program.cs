using System;
using Avalonia;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;

namespace Synapse.Studio
{
    internal static class Program
    {
        [STAThread]
        public static void Main(string[] args)
        {
            // Headless / console engine mode for CI and scripting
            if (Array.Exists(args, a => a is "--engine" or "--glfw"))
            {
                RunEngineOnly(args);
                return;
            }

            BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        }

        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .WithInterFont()
                .LogToTrace();

        private static void RunEngineOnly(string[] args)
        {
            Console.WriteLine("=================================================");
            Console.WriteLine("  SYNAPSE OMNIA — Moteur de simulation 3D · v1.1");
            Console.WriteLine("  Apprendre · Réécrire · Cultiver");
            Console.WriteLine("=================================================");

            var config = SynapseConfig.Load(args: args);
            using var logger = new SynapseLogger(null, LogLevel.Information);
            var host = new EngineHost(config, logger);
            var orchestrator = new FrameOrchestrator(host, logger);

            try
            {
                host.InitializeModules();
                host.LoadSceneAsync(config.ScenePath).GetAwaiter().GetResult();
                host.InitializeRender(config.Width, config.Height, config.EnableValidation);

                Console.WriteLine("[Engine] WASD move, Mouse look, ESC quit");
                while (host.RenderEngine is { IsRunning: true } && !host.RenderEngine.ShouldQuit)
                {
                    orchestrator.TickAsync().GetAwaiter().GetResult();
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[FATAL] {ex.Message}");
                Console.WriteLine(ex.StackTrace);
                Console.ResetColor();
                Environment.ExitCode = 1;
            }
            finally
            {
                host.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
        }
    }
}
