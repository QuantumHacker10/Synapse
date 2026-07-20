using System;
using System.IO;
using Avalonia;
using Synapse.Infrastructure;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Plugins;
using Synapse.Runtime;

namespace Synapse.Studio
{
    internal static class Program
    {
        [STAThread]
        public static void Main(string[] args)
        {
            if (Array.Exists(args, a => a is "--engine" or "--glfw"))
            {
                RunEngineOnly(args);
                return;
            }

            if (Array.Exists(args, a => a == "--benchmark"))
            {
                RunBenchmark(args);
                return;
            }

            if (TryGetScreenshotPath(args, out var screenshotPath))
            {
                Environment.ExitCode = StudioScreenshotCapture.Capture(screenshotPath, args);
                return;
            }

            BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        }

        private static bool TryGetScreenshotPath(string[] args, out string path)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--screenshot" && i + 1 < args.Length)
                {
                    path = args[i + 1];
                    return true;
                }
            }

            path = "";
            return false;
        }

        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .WithInterFont()
                .LogToTrace();

        private static void RunBenchmark(string[] args)
        {
            var config = SynapseConfig.Load(args: args);
            SimulationReproducibility.SetSeed(config.SimulationSeed);
            using var logger = new SynapseLogger(null, LogLevel.Information);
            var runner = new BenchmarkRunner(logger);

            var suitePath = config.BenchmarkConfigPath
                ?? Path.Combine(AppContext.BaseDirectory, "samples", "benchmarks", "default.json");
            if (!File.Exists(suitePath))
                suitePath = Path.Combine(Directory.GetCurrentDirectory(), "samples", "benchmarks", "default.json");

            var suite = File.Exists(suitePath)
                ? BenchmarkRunner.LoadConfig(suitePath)
                : new BenchmarkSuiteConfig { Name = "inline", SimulationSeed = config.SimulationSeed };

            if (!string.IsNullOrWhiteSpace(config.ScenePath))
                suite.ScenePath = config.ScenePath;
            suite.SimulationSeed = config.SimulationSeed;

            var report = runner.RunAsync(suite).GetAwaiter().GetResult();
            var outPath = config.BenchmarkOutputPath
                ?? Path.Combine(Directory.GetCurrentDirectory(), $"benchmark-{report.SuiteName}.json");
            runner.SaveReportAsync(report, outPath).GetAwaiter().GetResult();
            Console.WriteLine($"Benchmark complete: physics={report.PhysicsMsAvg:F2}ms sim={report.SimulationMsAvg:F2}ms -> {outPath}");
        }

        private static void RunEngineOnly(string[] args)
        {
            Console.WriteLine("=================================================");
            Console.WriteLine($"  {SynapseProduct.Name} — Moteur de simulation 3D · v{SynapseProduct.Version}");
            Console.WriteLine("  Apprendre · Réécrire · Cultiver");
            Console.WriteLine("=================================================");

            var config = SynapseConfig.Load(args: args);
            SimulationReproducibility.SetSeed(config.SimulationSeed);
            using var logger = new SynapseLogger(null, LogLevel.Information);
            using var pluginHost = new PluginHost(logger);
            var host = new EngineHost(config, logger);
            var orchestrator = new FrameOrchestrator(host, logger);

            try
            {
                host.InitializeModules();

                if (!string.IsNullOrWhiteSpace(config.PluginDirectory))
                    pluginHost.LoadFromDirectory(config.PluginDirectory, host);

                host.LoadSceneAsync(config.ScenePath).GetAwaiter().GetResult();

                if (!string.IsNullOrWhiteSpace(config.ExportScenePath))
                {
                    var export = SceneGlTFExporter.ExportAsync(host.Scene, config.ExportScenePath, host.MeshProvider)
                        .GetAwaiter().GetResult();
                    Console.WriteLine(export.Success
                        ? $"[Export] Scene -> {config.ExportScenePath} ({export.EntityCount} entities)"
                        : $"[Export] Failed: {export.ErrorMessage}");
                    if (config.Headless && string.IsNullOrWhiteSpace(config.ScenePath))
                        return;
                }

                if (config.Headless && !string.IsNullOrWhiteSpace(config.BenchmarkConfigPath))
                {
                    RunBenchmark(args);
                    return;
                }

                if (config.Headless)
                {
                    for (int i = 0; i < 120; i++)
                        orchestrator.TickAsync().GetAwaiter().GetResult();
                    Console.WriteLine("[Engine] Headless tick complete.");
                    return;
                }

                host.InitializeRender(config.Width, config.Height, config.EnableValidation);

                Console.WriteLine("[Engine] WASD move, Mouse look, ESC quit");
                while (host.RenderEngine is { IsRunning: true } && !host.RenderEngine.ShouldQuit)
                    orchestrator.TickAsync().GetAwaiter().GetResult();
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
