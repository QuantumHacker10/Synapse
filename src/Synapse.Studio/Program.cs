using System;
using System.IO;
using Avalonia;
using Synapse.Infrastructure;
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Network;
using Synapse.Physics;
using Synapse.Plugins;
using Synapse.Runtime;

namespace Synapse.Studio
{
    internal static class Program
    {
        [STAThread]
        public static void Main(string[] args)
        {
            if (Array.Exists(args, a => a == "--health"))
            {
                RunHealthCheck(args);
                return;
            }

            if (Array.Exists(args, a => a is "--engine" or "--glfw"))
            {
                RunEngineOnly(args);
                return;
            }

            if (Array.Exists(args, a => a == "--law-benchmark"))
            {
                RunLawBenchmark(args);
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

        private static void RunHealthCheck(string[] args)
        {
            var config = SynapseConfig.Load(args: args);
            using var logger = new SynapseLogger(null, LogLevel.Warning, consoleEnabled: false);
            var host = new EngineHost(config, logger);
            try
            {
                host.InitializeModules();
                var report = host.GetProductionHealth();
                Console.WriteLine(report.ToString());
                foreach (var note in report.ExperimentalNotes)
                    Console.WriteLine($"  · {note}");
                Environment.ExitCode = report.IsCoreReady ? 0 : 2;
            }
            finally
            {
                host.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
        }

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

        private static void RunLawBenchmark(string[] args)
        {
            var config = SynapseConfig.Load(args: args);
            var report = LawCompilationBenchmarkRunner.Run();
            var outPath = config.LawBenchmarkOutputPath
                ?? Path.Combine(Directory.GetCurrentDirectory(), "law-compilation-report.json");
            LawCompilationBenchmarkExporter.SaveAsync(report, outPath).GetAwaiter().GetResult();
            Console.WriteLine(
                $"Law compilation benchmark: total={report.TotalLaws} direct={report.CompiledDirect} " +
                $"fallback={report.CompiledWithFallback} failed={report.Failed} " +
                $"rate={report.DirectCompileRate:P1} -> {outPath}");
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
            WanSimulationPeerHub? wan = null;

            try
            {
                host.InitializeModules();
                Console.WriteLine($"[Health] {host.GetProductionHealth()}");

                if (!string.IsNullOrWhiteSpace(config.PluginDirectory))
                    pluginHost.LoadFromDirectory(config.PluginDirectory, host);

                // Optional local plugin marketplace catalog (manifest + hashes beside plugin-dir).
                if (!string.IsNullOrWhiteSpace(config.PluginDirectory))
                {
                    if (!string.IsNullOrWhiteSpace(config.PluginMarketplaceUrl))
                    {
                        var remote = new RemotePluginMarketplace(logger);
                        var n = remote.SyncAsync(config.PluginMarketplaceUrl, config.PluginDirectory)
                            .GetAwaiter().GetResult();
                        Console.WriteLine($"[Plugins] Remote marketplace sync: {n} installed from {config.PluginMarketplaceUrl}");
                    }

                    var market = PluginMarketplace.FromDirectory(config.PluginDirectory, logger);
                    market.VerifyInstalledOrWarn();
                }

                host.LoadSceneAsync(config.ScenePath).GetAwaiter().GetResult();

                if (!string.IsNullOrWhiteSpace(config.WanSessionCode))
                {
                    if (!System.Net.IPAddress.TryParse(config.WanRendezvousHost, out var rvHost))
                        rvHost = System.Net.IPAddress.Loopback;

                    wan = new WanSimulationPeerHub(
                        logger,
                        config.WanSessionCode,
                        rendezvousAddress: rvHost,
                        rendezvousPort: config.WanRendezvousPort,
                        hostRelay: !config.WanJoin,
                        ice: new NatIceOptions
                        {
                            StunServer = config.StunServer,
                            TurnServer = config.TurnServer,
                            TurnUsername = config.TurnUsername,
                            TurnPassword = config.TurnPassword,
                            PreferTurn = config.WanPreferTurn
                        });

                    if (config.WanJoin)
                    {
                        wan.JoinAsync(rvHost, config.WanRendezvousPort).GetAwaiter().GetResult();
                        Console.WriteLine($"[WAN] Joined via rendezvous {rvHost}:{config.WanRendezvousPort} mode={wan.TransportMode}");
                    }
                    else
                    {
                        wan.StartHostAsync(config.WanPort).GetAwaiter().GetResult();
                        Console.WriteLine(
                            $"[WAN] Authenticated host on 0.0.0.0:{wan.ListenPort} mode={wan.TransportMode} " +
                            $"(rendezvous {rvHost}:{config.WanRendezvousPort})");
                    }
                }

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

                if (!string.IsNullOrWhiteSpace(config.ExportWebPath))
                {
                    var result = host.ExportWebStudioAsync(config.ExportWebPath).GetAwaiter().GetResult();
                    Console.WriteLine(result.UsedDotnetPublish
                        ? $"[Export-Web] WASM Studio publié -> {result.OutputDirectory}"
                        : $"[Export-Web] Fallback WebGPU site -> {result.OutputDirectory}");
                    if (config.Headless)
                        return;
                }

                host.ApplyOptionalCollaborationFromConfigAsync().GetAwaiter().GetResult();
                if (host.IsWanConnected)
                    Console.WriteLine($"[WAN] {host.WanStatusText}");
                if (host.IsVrActive)
                    Console.WriteLine($"[VR] {host.VrStatusText}");

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
                if (wan != null)
                    wan.DisposeAsync().AsTask().GetAwaiter().GetResult();
                host.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
        }
    }
}
